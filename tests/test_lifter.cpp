// SPDX-FileCopyrightText: Copyright 2026 shadPS4 Emulator Project
// SPDX-License-Identifier: GPL-2.0-or-later
//
// End-to-end tests for the CPU runtime (src/core/cpu_runtime/).
//
// Each test:
//   1. Allocates RWX memory for a tiny guest program.
//   2. Writes encoded x86_64 instructions to it.
//   3. Sets up a GuestState with the guest program's address in RIP
//      and a guest stack containing a sentinel return address.
//   4. Calls Runtime::Run(state).
//   5. Verifies post-execution register state.
//
// The tests exercise the gateway/dispatcher/lifter pipeline end-to-
// end. They produce no output unless something fails, so they're
// suitable for CI.

#include <array>
#include <bit>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <limits>
#include <string>
#include <vector>
#include <functional>

#include <gtest/gtest.h>

#ifdef _WIN32
#include <windows.h>
#else
#include <sys/mman.h>
#endif

#include "core/cpu_runtime/block_cache.h"
#include "core/cpu_runtime/code_cache.h"
#include "core/cpu_runtime/guest_state.h"
#include "core/cpu_runtime/hle_registry.h"
#include "core/cpu_runtime/runtime.h"

// The test shim's common/types.h is a stripped-down stand-in for
// upstream's. shadPS4 proper defines PS4_SYSV_ABI in common/types.h;
// the shim doesn't carry it. Define it locally so the HLE-bridge
// host helper can be marked with the same convention an HLE function
// in shadPS4 would use. On Linux x86-64 the attribute is a no-op
// (SysV IS the default), but it matches upstream's source shape.
#ifndef PS4_SYSV_ABI
#define PS4_SYSV_ABI __attribute__((sysv_abi))
#endif

namespace Core::Runtime {
namespace {

// Sentinel popped from the guest stack by RET. We use the runtime's
// public kHostReturnAddress so the dispatcher recognizes it and
// exits cleanly when the guest RETs through the call chain.
//
// (Before the gateway looped, tests could use any sentinel because
// Run() exited after one block regardless of where RET set RIP.
// Now that the dispatcher loops, the sentinel MUST be a value the
// dispatcher recognizes as "exit" — otherwise the dispatcher tries
// to compile a block at that fake address and segfaults.)
constexpr u64 kReturnSentinel = kHostReturnAddress;

/// Allocate a pair of pages (code + stack) as RWX memory for the
/// guest. Uses the host OS directly rather than the upstream
/// AddressSpace because the tests are about the JIT itself, not
/// about the memory subsystem.
///
/// Layout (low → high):
///   base_                           ─ code + working stack
///   base_ + TOTAL_SIZE - HEADROOM   ─ StackTop()
///   base_ + TOTAL_SIZE              ─ end of allocation
///
/// `HEADROOM` is a band of committed bytes ABOVE the reported stack
/// top. It exists for the HLE bridge: when guest code performs a
/// CALL into a host function, the dispatcher's bridge speculatively
/// reads 8 u64 stack-arg slots at `[guest_rsp+8 .. guest_rsp+64]`
/// — see the bridge documentation in src/core/cpu_runtime/runtime.cpp.
/// Real game stacks are megabytes deep and the overscan is invisible
/// there, but our tests' tightly-bounded 2-page allocation has no
/// such cushion. Without HEADROOM, every HleBridge_* test would fault
/// on Windows where VirtualAlloc is strict (Linux's mmap arena
/// sometimes leaves neighboring pages mapped, masking the bug, but
/// the read is still UB). The headroom is zero-initialized by the OS
/// allocator on both platforms, so tests inspecting the overscan see
/// deterministic 0s rather than uninitialized garbage.
class GuestMemory {
public:
    static constexpr u64 PAGE_SIZE = 4096;
    static constexpr u64 TOTAL_SIZE = PAGE_SIZE * 2;
    /// Bytes reserved above StackTop() for the HLE bridge's speculative
    /// 8-u64-slot read past the guest return-address slot. 64 bytes
    /// is the strict minimum (8 * sizeof(u64)); 128 leaves slack for
    /// future bridge expansion (more stack args) without revisiting
    /// every test's stack arithmetic.
    static constexpr u64 BRIDGE_HEADROOM = 128;

    GuestMemory() {
        // This region holds GUEST memory: the guest program bytes (which the
        // lifter only ever READS, via the decoder) and the guest stack. The
        // JIT emits HOST code into the separate CodeCache, never here — so
        // this mapping needs READ+WRITE only, never EXECUTE. Requesting EXEC
        // is not just unnecessary, it's actively harmful on Apple Silicon:
        // the hardened runtime rejects a simultaneously writable+executable
        // anonymous mapping (W^X) unless it carries MAP_JIT, so the old
        // PROT_EXEC form returned MAP_FAILED there and every test died in
        // SetUp() before a single guest instruction was lifted.
#ifdef _WIN32
        void* p = ::VirtualAlloc(nullptr, TOTAL_SIZE, MEM_COMMIT | MEM_RESERVE, PAGE_READWRITE);
        base_ = static_cast<u8*>(p);
#else
        void* p = ::mmap(nullptr, TOTAL_SIZE, PROT_READ | PROT_WRITE, MAP_PRIVATE | MAP_ANON, -1, 0);
        base_ = (p == MAP_FAILED) ? nullptr : static_cast<u8*>(p);
#endif
    }

    ~GuestMemory() {
        if (base_ == nullptr)
            return;
#ifdef _WIN32
        ::VirtualFree(base_, 0, MEM_RELEASE);
#else
        ::munmap(base_, TOTAL_SIZE);
#endif
    }

    GuestMemory(const GuestMemory&) = delete;
    GuestMemory& operator=(const GuestMemory&) = delete;

    [[nodiscard]] u8* CodePtr() const {
        return base_;
    }
    /// Reports the boundary BELOW the bridge-overscan headroom — i.e.,
    /// the highest guest-stack address that "stack" callers should
    /// treat as theirs. The bytes from here to `base_ + TOTAL_SIZE`
    /// belong to the bridge and must not carry test state.
    [[nodiscard]] u8* StackTop() const {
        return base_ + TOTAL_SIZE - BRIDGE_HEADROOM;
    }
    [[nodiscard]] bool IsValid() const {
        return base_ != nullptr;
    }

private:
    u8* base_ = nullptr;
};

/// Set up a GuestState with the given program, push the sentinel
/// return address onto the guest stack, run the runtime, and return.
struct RunResult {
    GuestState state{};
    u64 program_base{};
};

RunResult RunProgram(const u8* program, size_t program_size, GuestMemory& mem) {
    std::memcpy(mem.CodePtr(), program, program_size);

    u8* guest_rsp = mem.StackTop() - 8;
    *reinterpret_cast<u64*>(guest_rsp) = kReturnSentinel;

    RunResult r;
    r.program_base = reinterpret_cast<u64>(mem.CodePtr());
    r.state.rip = r.program_base;
    r.state.gpr[4] = reinterpret_cast<u64>(guest_rsp); // RSP

    Runtime rt;
    rt.Run(r.state);
    return r;
}

// ============================================================================

class CpuRuntimeTest : public ::testing::Test {
protected:
    GuestMemory mem;

    void SetUp() override {
        ASSERT_TRUE(mem.IsValid()) << "Failed to allocate guest memory";
    }
};

// ========================================================================
// LIFTER / EMITTER TESTS -- guest x86 instruction semantics: ALU, shifts, SSE/AVX, x87, conditionals, addressing modes.
// ========================================================================

// MOV rax, 0x42; ADD rax, 1; RET
// Expected: rax=0x43, rip=sentinel, exit_reason=BlockEnd.
TEST_F(CpuRuntimeTest, MovAddRet_ProducesCorrectRax) {
    const u8 program[] = {
        0x48, 0xc7, 0xc0, 0x42, 0x00, 0x00, 0x00, // mov rax, 0x42
        0x48, 0x83, 0xc0, 0x01,                   // add rax, 1
        0xc3,                                     // ret
    };
    const auto r = RunProgram(program, sizeof(program), mem);

    EXPECT_EQ(r.state.gpr[0], 0x43u);
    EXPECT_EQ(r.state.rip, kReturnSentinel);
    EXPECT_EQ(r.state.exit_reason, static_cast<u32>(ExitReason::BlockEnd));
}

// MOV rax, 0x100
// MOV rbx, 0x200
// ADD rax, rbx
// ADD rax, 0x10
// MOV rcx, rax
// RET
//
// Expected: rax=0x310, rbx=0x200, rcx=0x310.
TEST_F(CpuRuntimeTest, MultipleRegistersAndOpcodes_ProduceCorrectValues) {
    const u8 program[] = {
        0x48, 0xc7, 0xc0, 0x00, 0x01, 0x00, 0x00, // mov rax, 0x100
        0x48, 0xc7, 0xc3, 0x00, 0x02, 0x00, 0x00, // mov rbx, 0x200
        0x48, 0x01, 0xd8,                         // add rax, rbx
        0x48, 0x83, 0xc0, 0x10,                   // add rax, 0x10
        0x48, 0x89, 0xc1,                         // mov rcx, rax
        0xc3,                                     // ret
    };
    const auto r = RunProgram(program, sizeof(program), mem);

    EXPECT_EQ(r.state.gpr[0], 0x310u) << "rax (gpr[0])";
    EXPECT_EQ(r.state.gpr[3], 0x200u) << "rbx (gpr[3])";
    EXPECT_EQ(r.state.gpr[1], 0x310u) << "rcx (gpr[1])";
    EXPECT_EQ(r.state.rip, kReturnSentinel);
    EXPECT_EQ(r.state.exit_reason, static_cast<u32>(ExitReason::BlockEnd));
}

// CurrentGuestState should return nullptr when no JIT is active on
// this thread.
TEST_F(CpuRuntimeTest, CurrentGuestState_NullOutsideRun) {
    EXPECT_EQ(Runtime::CurrentGuestState(), nullptr);
}

// IsGuestPointer must correctly identify guest vs host code addresses.
// Mechanism: dladdr() on POSIX, GetModuleHandleEx on Windows. A pointer
// in any loaded host module is host; anything else is guest.
TEST_F(CpuRuntimeTest, IsGuestPointer_DistinguishesAddressRanges) {
    // A host pointer: any function in this test binary.
    auto* host_fn = reinterpret_cast<void*>(&Runtime::Instance);
    EXPECT_FALSE(Runtime::IsGuestPointer(host_fn))
        << "Functions in the test executable should be classified as host";

    // Another host pointer: a libc function. Definitely in a loaded .so.
    auto* libc_fn = reinterpret_cast<void*>(&std::memcpy);
    EXPECT_FALSE(Runtime::IsGuestPointer(libc_fn))
        << "Functions in loaded shared libraries should be classified as host";

    // Null is treated as host.
    EXPECT_FALSE(Runtime::IsGuestPointer(nullptr));

    // A synthetic address that's *very* unlikely to be in any loaded
    // module: a value in the middle of the empty user-virtual range.
    // dladdr returns 0 for this, so IsGuestPointer says true ("guest").
    // (In real shadPS4 this would be a guest-mapped address.)
    auto* synthetic_guest = reinterpret_cast<void*>(0x123456789abcULL);
    EXPECT_TRUE(Runtime::IsGuestPointer(synthetic_guest))
        << "Addresses not in any loaded host module are assumed guest";
}

// ============================================================================
// New opcode tests (PUSH/POP, SUB-imm, XOR, MOV memory, MOV imm)
// ============================================================================

// PUSH r64 and POP r64. Test by pushing rdi (set up to 0x1234),
// then popping into rax, then ret. After RET, RAX should be 0x1234.
//
//   push rdi    57
//   pop  rax    58
//   ret         c3
TEST_F(CpuRuntimeTest, PushPop_RoundTripsValue) {
    const u8 program[] = {0x57, 0x58, 0xc3};
    std::memcpy(mem.CodePtr(), program, sizeof(program));

    u8* guest_rsp = mem.StackTop() - 16; // leave room for push + sentinel
    *reinterpret_cast<u64*>(guest_rsp + 8) = kReturnSentinel;

    GuestState state{};
    state.rip = reinterpret_cast<u64>(mem.CodePtr());
    state.gpr[4] = reinterpret_cast<u64>(guest_rsp + 8); // RSP at sentinel
    state.gpr[7] = 0x1234'5678'9abc'def0ULL;             // RDI

    Runtime rt;
    rt.Run(state);

    EXPECT_EQ(state.gpr[0], 0x1234'5678'9abc'def0ULL);
    EXPECT_EQ(state.exit_reason, static_cast<u32>(ExitReason::BlockEnd));
}

// SUB r64, imm8 — subtract 0x20 from rax. Then ret.
//
//   mov rax, 0x100   48 c7 c0 00 01 00 00
//   sub rax, 0x20    48 83 e8 20
//   ret              c3
TEST_F(CpuRuntimeTest, SubImm_DecrementsCorrectly) {
    const u8 program[] = {
        0x48, 0xc7, 0xc0, 0x00, 0x01, 0x00, 0x00, // mov rax, 0x100
        0x48, 0x83, 0xe8, 0x20,                   // sub rax, 0x20
        0xc3,                                     // ret
    };
    auto result = RunProgram(program, sizeof(program), mem);
    EXPECT_EQ(result.state.gpr[0], 0x100ULL - 0x20ULL);
}

// XOR rax, rax — register-zero idiom. After running, RAX should be 0
// regardless of what it started as.
//
//   xor rax, rax    48 31 c0
//   ret             c3
TEST_F(CpuRuntimeTest, XorReg_ZeroesRegister) {
    const u8 program[] = {0x48, 0x31, 0xc0, 0xc3};
    std::memcpy(mem.CodePtr(), program, sizeof(program));

    u8* guest_rsp = mem.StackTop() - 8;
    *reinterpret_cast<u64*>(guest_rsp) = kReturnSentinel;

    GuestState state{};
    state.rip = reinterpret_cast<u64>(mem.CodePtr());
    state.gpr[4] = reinterpret_cast<u64>(guest_rsp);
    state.gpr[0] = 0xdead'beef'cafe'babeULL; // RAX starts non-zero

    Runtime rt;
    rt.Run(state);

    EXPECT_EQ(state.gpr[0], 0u);
}

// MOV r64, [mem] — load from memory. We put 0xfeedface at a known
// address, then mov rax from there via rdi-as-base.
//
//   mov rax, [rdi]    48 8b 07
//   ret               c3
TEST_F(CpuRuntimeTest, MovLoadFromMemory_ViaBaseReg) {
    const u8 program[] = {0x48, 0x8b, 0x07, 0xc3};
    std::memcpy(mem.CodePtr(), program, sizeof(program));

    // Use a chunk of guest memory below the stack as a data slot.
    u8* data_slot = mem.StackTop() - 32;
    *reinterpret_cast<u64*>(data_slot) = 0xfeed'face'1234'5678ULL;

    u8* guest_rsp = mem.StackTop() - 8;
    *reinterpret_cast<u64*>(guest_rsp) = kReturnSentinel;

    GuestState state{};
    state.rip = reinterpret_cast<u64>(mem.CodePtr());
    state.gpr[4] = reinterpret_cast<u64>(guest_rsp);
    state.gpr[7] = reinterpret_cast<u64>(data_slot); // RDI -> data slot

    Runtime rt;
    rt.Run(state);

    EXPECT_EQ(state.gpr[0], 0xfeed'face'1234'5678ULL);
}

// MOV [mem], r64 — store to memory. Pre-zero a data slot, store
// 0xdeadbeef there via RDI as base, verify by reading after run.
//
//   mov [rdi], rax    48 89 07
//   ret               c3
TEST_F(CpuRuntimeTest, MovStoreToMemory_ViaBaseReg) {
    const u8 program[] = {0x48, 0x89, 0x07, 0xc3};
    std::memcpy(mem.CodePtr(), program, sizeof(program));

    u8* data_slot = mem.StackTop() - 32;
    *reinterpret_cast<u64*>(data_slot) = 0;

    u8* guest_rsp = mem.StackTop() - 8;
    *reinterpret_cast<u64*>(guest_rsp) = kReturnSentinel;

    GuestState state{};
    state.rip = reinterpret_cast<u64>(mem.CodePtr());
    state.gpr[4] = reinterpret_cast<u64>(guest_rsp);
    state.gpr[0] = 0xdead'beef'1234'5678ULL;         // RAX = value
    state.gpr[7] = reinterpret_cast<u64>(data_slot); // RDI -> data slot

    Runtime rt;
    rt.Run(state);

    EXPECT_EQ(*reinterpret_cast<u64*>(data_slot), 0xdead'beef'1234'5678ULL);
}

// MOV r64, [base + disp] — load with displacement. Stack-frame-style
// addressing: read from [rbp-8].
//
//   mov rax, [rbp-8]    48 8b 45 f8
//   ret                 c3
TEST_F(CpuRuntimeTest, MovLoadFromMemory_WithDisplacement) {
    const u8 program[] = {0x48, 0x8b, 0x45, 0xf8, 0xc3};
    std::memcpy(mem.CodePtr(), program, sizeof(program));

    u8* data_slot = mem.StackTop() - 40;
    *reinterpret_cast<u64*>(data_slot) = 0xa1b2'c3d4'e5f6'0708ULL;

    u8* guest_rsp = mem.StackTop() - 8;
    *reinterpret_cast<u64*>(guest_rsp) = kReturnSentinel;

    GuestState state{};
    state.rip = reinterpret_cast<u64>(mem.CodePtr());
    state.gpr[4] = reinterpret_cast<u64>(guest_rsp);
    state.gpr[5] = reinterpret_cast<u64>(data_slot) + 8; // RBP -> slot + 8

    Runtime rt;
    rt.Run(state);

    EXPECT_EQ(state.gpr[0], 0xa1b2'c3d4'e5f6'0708ULL);
}

// Full function prologue + epilogue. Verifies that PUSH/POP plus
// MOV reg-reg plus SUB-imm plus ADD-imm all work together end-to-end.
//
//   push rbp           55
//   mov  rbp, rsp      48 89 e5
//   sub  rsp, 0x20     48 83 ec 20
//   mov  rax, rdi      48 89 f8        ; pretend "do something with arg"
//   add  rsp, 0x20     48 83 c4 20
//   pop  rbp           5d
//   ret                c3
//
// We pass 0x42 in RDI, expect 0x42 in RAX, and verify RBP and RSP
// come out the way they started.
TEST_F(CpuRuntimeTest, FullPrologueEpilogue_ProducesCorrectState) {
    const u8 program[] = {
        0x55,                   // push rbp
        0x48, 0x89, 0xe5,       // mov rbp, rsp
        0x48, 0x83, 0xec, 0x20, // sub rsp, 0x20
        0x48, 0x89, 0xf8,       // mov rax, rdi
        0x48, 0x83, 0xc4, 0x20, // add rsp, 0x20
        0x5d,                   // pop rbp
        0xc3,                   // ret
    };
    std::memcpy(mem.CodePtr(), program, sizeof(program));

    u8* guest_rsp = mem.StackTop() - 16;
    *reinterpret_cast<u64*>(guest_rsp + 8) = kReturnSentinel;

    const u64 rsp_initial = reinterpret_cast<u64>(guest_rsp + 8);
    const u64 rbp_initial = 0x1111'2222'3333'4444ULL;

    GuestState state{};
    state.rip = reinterpret_cast<u64>(mem.CodePtr());
    state.gpr[4] = rsp_initial;
    state.gpr[5] = rbp_initial;
    state.gpr[7] = 0x42; // RDI

    Runtime rt;
    rt.Run(state);

    EXPECT_EQ(state.gpr[0], 0x42u) << "RAX should hold the arg";
    EXPECT_EQ(state.gpr[5], rbp_initial) << "RBP should be restored";
    // After the RET, RSP has popped the sentinel return address —
    // standard x86 calling convention. Entry was at rsp_initial
    // (pointing at the sentinel slot); after RET it's 8 bytes higher.
    EXPECT_EQ(state.gpr[4], rsp_initial + 8)
        << "RSP after RET should be entry+8 (return address popped)";
    EXPECT_EQ(state.exit_reason, static_cast<u32>(ExitReason::BlockEnd));
}

// ============================================================================
// Flag and branch tests
// ============================================================================
//
// Flag bit positions in x86 RFLAGS:
//   CF = bit 0
//   PF = bit 2
//   ZF = bit 6
//   SF = bit 7
//   OF = bit 11

constexpr u64 kCF = 1ULL << 0;
constexpr u64 kPF = 1ULL << 2;
constexpr u64 kZF = 1ULL << 6;
constexpr u64 kSF = 1ULL << 7;
constexpr u64 kOF = 1ULL << 11;

// CMP with equal operands: ZF=1, SF=0, CF=0, OF=0.
//
//   mov rax, 0x42    48 c7 c0 42 00 00 00
//   mov rbx, 0x42    48 c7 c3 42 00 00 00
//   cmp rax, rbx     48 39 d8       (CMP r/m64, r64: 39 /r, rax=rbx form)
//   ret              c3
TEST_F(CpuRuntimeTest, CmpEqual_SetsZeroFlag) {
    const u8 program[] = {
        0x48, 0xc7, 0xc0, 0x42, 0x00, 0x00, 0x00, 0x48, 0xc7,
        0xc3, 0x42, 0x00, 0x00, 0x00, 0x48, 0x39, 0xd8, 0xc3,
    };
    auto result = RunProgram(program, sizeof(program), mem);
    EXPECT_TRUE(result.state.rflags & kZF) << "ZF should be set for equal";
    EXPECT_FALSE(result.state.rflags & kSF) << "SF should be clear";
    EXPECT_FALSE(result.state.rflags & kCF) << "CF should be clear";
    EXPECT_FALSE(result.state.rflags & kOF) << "OF should be clear";
}

// CMP with greater lhs (unsigned): ZF=0, CF=0 (no borrow). Sign of
// result is positive (0x100 - 0x42 > 0) so SF=0.
//
//   mov rax, 0x100   48 c7 c0 00 01 00 00
//   mov rbx, 0x42    48 c7 c3 42 00 00 00
//   cmp rax, rbx     48 39 d8
//   ret              c3
TEST_F(CpuRuntimeTest, CmpGreater_ClearsZeroFlag) {
    const u8 program[] = {
        0x48, 0xc7, 0xc0, 0x00, 0x01, 0x00, 0x00, 0x48, 0xc7,
        0xc3, 0x42, 0x00, 0x00, 0x00, 0x48, 0x39, 0xd8, 0xc3,
    };
    auto result = RunProgram(program, sizeof(program), mem);
    EXPECT_FALSE(result.state.rflags & kZF) << "ZF should be clear";
    EXPECT_FALSE(result.state.rflags & kCF) << "CF should be clear (no borrow)";
    EXPECT_FALSE(result.state.rflags & kSF) << "SF should be clear (positive result)";
}

// CMP with smaller lhs (unsigned): CF=1 because subtraction borrows.
//
//   mov rax, 0x10    48 c7 c0 10 00 00 00
//   mov rbx, 0x100   48 c7 c3 00 01 00 00
//   cmp rax, rbx     48 39 d8
//   ret              c3
TEST_F(CpuRuntimeTest, CmpSmaller_SetsCarryFlag) {
    const u8 program[] = {
        0x48, 0xc7, 0xc0, 0x10, 0x00, 0x00, 0x00, 0x48, 0xc7,
        0xc3, 0x00, 0x01, 0x00, 0x00, 0x48, 0x39, 0xd8, 0xc3,
    };
    auto result = RunProgram(program, sizeof(program), mem);
    EXPECT_FALSE(result.state.rflags & kZF) << "ZF should be clear";
    EXPECT_TRUE(result.state.rflags & kCF) << "CF should be set (borrow)";
    EXPECT_TRUE(result.state.rflags & kSF) << "SF should be set (negative result)";
}

// TEST rax, rax with rax=0: ZF=1, CF=0, OF=0.
//
//   xor rax, rax     48 31 c0
//   test rax, rax    48 85 c0
//   ret              c3
TEST_F(CpuRuntimeTest, TestZeroReg_SetsZeroFlag) {
    const u8 program[] = {0x48, 0x31, 0xc0, 0x48, 0x85, 0xc0, 0xc3};
    auto result = RunProgram(program, sizeof(program), mem);
    EXPECT_TRUE(result.state.rflags & kZF);
    EXPECT_FALSE(result.state.rflags & kCF);
    EXPECT_FALSE(result.state.rflags & kOF);
}

// JZ taken: CMP that produces ZF=1, then JZ jumps to a target
// outside the program's lifted bytes. The dispatcher follows
// the branch, tries to compile at the target (which is zeros),
// and exits with UnsupportedInstruction. state.rip ends up
// at the target — confirming the branch was taken.
//
//   xor rax, rax     48 31 c0
//   jz  +0x10        74 10            ; if ZF, jump to byte 21
//   <padding bytes that won't be reached when taken>
TEST_F(CpuRuntimeTest, Jz_TakenWhenZf) {
    const u8 program[] = {
        0x48, 0x31, 0xc0, // xor rax, rax       (offset 0-2)
        0x74, 0x10,       // jz +0x10           (offset 3-4)
        // Bytes 5..20 are unreached when branch is taken.
    };
    std::memcpy(mem.CodePtr(), program, sizeof(program));
    // Place an explicitly-unsupported instruction at the branch target
    // (offset 21) so the dispatcher cleanly exits with
    // UnsupportedInstruction there. We can't rely on zero-padding being
    // unsupported: 00 00 decodes as `add byte[rax], al`, which the lifter
    // now handles (8-bit mem-dst ALU) and would fault on a null rax. BSR
    // (48 0f bd d8) is the canonical not-yet-supported instruction.
    const u8 bsr[] = {0x48, 0x0f, 0xbd, 0xd8};
    std::memcpy(mem.CodePtr() + 21, bsr, sizeof(bsr));

    u8* guest_rsp = mem.StackTop() - 8;
    *reinterpret_cast<u64*>(guest_rsp) = kReturnSentinel;

    GuestState state{};
    const u64 program_base = reinterpret_cast<u64>(mem.CodePtr());
    state.rip = program_base;
    state.gpr[4] = reinterpret_cast<u64>(guest_rsp);

    Runtime rt;
    rt.Run(state);

    // After JZ taken: state.rip should be the branch target = 21, where the
    // BSR triggers a clean UnsupportedInstruction exit.
    EXPECT_EQ(state.rip, program_base + 21);
    EXPECT_EQ(state.exit_reason, static_cast<u32>(ExitReason::UnsupportedInstruction));
}

// JZ not taken: with ZF=0, JZ falls through. The dispatcher then
// enters the fall-through block (also zero-padding past the JZ)
// and exits with UnsupportedInstruction at the fall-through point.
//
//   mov rax, 1       48 c7 c0 01 00 00 00
//   test rax, rax    48 85 c0                ; ZF=0 since rax != 0
//   jz  +0x10        74 10
//   <fall-through point at offset 12>
TEST_F(CpuRuntimeTest, Jz_FallsThroughWhenNoZf) {
    const u8 program[] = {
        0x48, 0xc7, 0xc0, 0x01, 0x00, 0x00, 0x00, // mov rax, 1   (offsets 0-6)
        0x48, 0x85, 0xc0,                         // test rax,rax (offsets 7-9)
        0x74, 0x10,                               // jz  +0x10    (offsets 10-11)
        // Fall-through at offset 12.
    };
    std::memcpy(mem.CodePtr(), program, sizeof(program));
    // Explicit unsupported instruction at the fall-through point (offset 12),
    // for the same reason as Jz_TakenWhenZf: zero-padding now lifts as an
    // 8-bit mem-dst ADD and would fault rather than trapping cleanly.
    const u8 bsr[] = {0x48, 0x0f, 0xbd, 0xd8}; // bsr rbx, rax (unsupported)
    std::memcpy(mem.CodePtr() + 12, bsr, sizeof(bsr));

    u8* guest_rsp = mem.StackTop() - 8;
    *reinterpret_cast<u64*>(guest_rsp) = kReturnSentinel;

    GuestState state{};
    const u64 program_base = reinterpret_cast<u64>(mem.CodePtr());
    state.rip = program_base;
    state.gpr[4] = reinterpret_cast<u64>(guest_rsp);

    Runtime rt;
    rt.Run(state);

    // Fall-through target is offset 12, where the BSR triggers a clean
    // UnsupportedInstruction exit.
    EXPECT_EQ(state.rip, program_base + 12);
    EXPECT_EQ(state.exit_reason, static_cast<u32>(ExitReason::UnsupportedInstruction));
}

// JNZ: opposite of JZ. With ZF=0, JNZ should be taken.
//
//   mov rax, 1       48 c7 c0 01 00 00 00
//   test rax, rax    48 85 c0
//   jnz +0x10        75 10
TEST_F(CpuRuntimeTest, Jnz_TakenWhenNoZf) {
    const u8 program[] = {
        0x48, 0xc7, 0xc0, 0x01, 0x00, 0x00, 0x00, // mov rax, 1
        0x48, 0x85, 0xc0,                         // test rax,rax
        0x75, 0x10,                               // jnz +0x10
    };
    std::memcpy(mem.CodePtr(), program, sizeof(program));
    const u8 bsr_pad[] = {0x48, 0x0f, 0xbd, 0xd8};
    std::memcpy(mem.CodePtr() + 28, bsr_pad, sizeof(bsr_pad));

    u8* guest_rsp = mem.StackTop() - 8;
    *reinterpret_cast<u64*>(guest_rsp) = kReturnSentinel;

    GuestState state{};
    const u64 program_base = reinterpret_cast<u64>(mem.CodePtr());
    state.rip = program_base;
    state.gpr[4] = reinterpret_cast<u64>(guest_rsp);

    Runtime rt;
    rt.Run(state);

    // jnz at offset 10, next_rip = 12, target = 12 + 0x10 = 28.
    EXPECT_EQ(state.rip, program_base + 28);
}

// ============================================================================
// TEST qword[mem], r64 — memory-destination form. TEST computes lhs & rhs for
// flags only (no writeback); AND is commutative so [mem] & reg == reg & [mem].
// Gap from CUSA02394 at guest 0x8001e1d30 (length 8). Writes ZF/SF/PF; clears
// CF/OF. The memory operand must be left unmodified.
// ============================================================================

// Non-zero AND result: ZF clear. Memory left untouched.
TEST_F(CpuRuntimeTest, Test64_MemReg_NonZero_ClearsZF) {
    u64* slot = reinterpret_cast<u64*>(mem.CodePtr() + 0x100);
    *slot = 0x00FF00FF00FF00FFULL;
    const u8 program[] = {
        0x48, 0xb8, 0,0,0,0,0,0,0,0, // mov rax, <slot>
        0x48, 0x85, 0x10,            // test qword [rax], rdx
        0xc3,
    };
    u8 prog[sizeof(program)];
    std::memcpy(prog, program, sizeof(program));
    const u64 a = reinterpret_cast<u64>(slot);
    std::memcpy(prog + 2, &a, sizeof(a));
    std::memcpy(mem.CodePtr(), prog, sizeof(prog));
    u8* guest_rsp = mem.StackTop() - 8;
    *reinterpret_cast<u64*>(guest_rsp) = kReturnSentinel;
    GuestState st{};
    st.rip = reinterpret_cast<u64>(mem.CodePtr());
    st.gpr[4] = reinterpret_cast<u64>(guest_rsp);
    st.gpr[2] = 0x0000000000000001ULL; // rdx: bit 0 set, overlaps [mem]

    Runtime rt;
    rt.Run(st);
    EXPECT_EQ(st.rflags & (1ULL<<6), 0ULL) << "ZF clear: [mem] & rdx != 0";
    EXPECT_EQ(st.rflags & 0x1ULL, 0ULL) << "CF cleared by TEST";
    EXPECT_EQ(st.rflags & (1ULL<<11), 0ULL) << "OF cleared by TEST";
    EXPECT_EQ(*slot, 0x00FF00FF00FF00FFULL) << "memory operand unmodified";
}

// Zero AND result: ZF set.
TEST_F(CpuRuntimeTest, Test64_MemReg_Zero_SetsZF) {
    u64* slot = reinterpret_cast<u64*>(mem.CodePtr() + 0x100);
    *slot = 0x00FF00FF00FF00FFULL;
    const u8 program[] = {
        0x48, 0xb8, 0,0,0,0,0,0,0,0, // mov rax, <slot>
        0x48, 0x85, 0x10,            // test qword [rax], rdx
        0xc3,
    };
    u8 prog[sizeof(program)];
    std::memcpy(prog, program, sizeof(program));
    const u64 a = reinterpret_cast<u64>(slot);
    std::memcpy(prog + 2, &a, sizeof(a));
    std::memcpy(mem.CodePtr(), prog, sizeof(prog));
    u8* guest_rsp = mem.StackTop() - 8;
    *reinterpret_cast<u64*>(guest_rsp) = kReturnSentinel;
    GuestState st{};
    st.rip = reinterpret_cast<u64>(mem.CodePtr());
    st.gpr[4] = reinterpret_cast<u64>(guest_rsp);
    st.gpr[2] = 0xFF00FF00FF00FF00ULL; // rdx: complementary bits, AND = 0

    Runtime rt;
    rt.Run(st);
    EXPECT_EQ(st.rflags & (1ULL<<6), (1ULL<<6)) << "ZF set: [mem] & rdx == 0";
}

// Sign bit of the AND result sets SF.
TEST_F(CpuRuntimeTest, Test64_MemReg_SignBit_SetsSF) {
    u64* slot = reinterpret_cast<u64*>(mem.CodePtr() + 0x100);
    *slot = 0x8000000000000000ULL; // only bit 63 set
    const u8 program[] = {
        0x48, 0xb8, 0,0,0,0,0,0,0,0, // mov rax, <slot>
        0x48, 0x85, 0x10,            // test qword [rax], rdx
        0xc3,
    };
    u8 prog[sizeof(program)];
    std::memcpy(prog, program, sizeof(program));
    const u64 a = reinterpret_cast<u64>(slot);
    std::memcpy(prog + 2, &a, sizeof(a));
    std::memcpy(mem.CodePtr(), prog, sizeof(prog));
    u8* guest_rsp = mem.StackTop() - 8;
    *reinterpret_cast<u64*>(guest_rsp) = kReturnSentinel;
    GuestState st{};
    st.rip = reinterpret_cast<u64>(mem.CodePtr());
    st.gpr[4] = reinterpret_cast<u64>(guest_rsp);
    st.gpr[2] = 0x8000000000000000ULL; // rdx: bit 63 set too

    Runtime rt;
    rt.Run(st);
    EXPECT_EQ(st.rflags & (1ULL<<7), (1ULL<<7)) << "SF set: result bit 63 set";
    EXPECT_EQ(st.rflags & (1ULL<<6), 0ULL) << "ZF clear";
}

// ============================================================================
// AND qword[mem], r64 — memory-destination form. Unlike TEST, AND writes the
// masked result back to memory. Gap from CUSA02394 at guest 0x8001e1d69, in
// the same code region as the TEST qword[mem],r64 above (mask applied in place
// to a memory word). Writes ZF/SF/PF; clears CF/OF.
// ============================================================================

// Masking: [mem] &= reg, result stored, flags from result.
TEST_F(CpuRuntimeTest, And64_MemReg_MasksAndStores) {
    u64* slot = reinterpret_cast<u64*>(mem.CodePtr() + 0x100);
    slot[0] = 0xFFFFFFFFFFFFFFFFULL;
    slot[1] = 0xCCCCCCCCCCCCCCCCULL; // sentinel after the target
    const u8 program[] = {
        0x48, 0xb8, 0,0,0,0,0,0,0,0, // mov rax, <slot>
        0x48, 0x21, 0x10,            // and qword [rax], rdx
        0xc3,
    };
    u8 prog[sizeof(program)];
    std::memcpy(prog, program, sizeof(program));
    const u64 a = reinterpret_cast<u64>(slot);
    std::memcpy(prog + 2, &a, sizeof(a));
    std::memcpy(mem.CodePtr(), prog, sizeof(prog));
    u8* guest_rsp = mem.StackTop() - 8;
    *reinterpret_cast<u64*>(guest_rsp) = kReturnSentinel;
    GuestState st{};
    st.rip = reinterpret_cast<u64>(mem.CodePtr());
    st.gpr[4] = reinterpret_cast<u64>(guest_rsp);
    st.gpr[2] = 0x0F0F0F0F0F0F0F0FULL; // rdx mask

    Runtime rt;
    rt.Run(st);
    EXPECT_EQ(slot[0], 0x0F0F0F0F0F0F0F0FULL) << "[mem] &= rdx written back";
    EXPECT_EQ(slot[1], 0xCCCCCCCCCCCCCCCCULL) << "next qword untouched";
    EXPECT_EQ(st.rflags & (1ULL<<6), 0ULL) << "ZF clear: result != 0";
    EXPECT_EQ(st.rflags & 0x1ULL, 0ULL) << "CF cleared";
    EXPECT_EQ(st.rflags & (1ULL<<11), 0ULL) << "OF cleared";
}

// Result zero: ZF set, memory becomes 0.
TEST_F(CpuRuntimeTest, And64_MemReg_ZeroResult_SetsZF) {
    u64* slot = reinterpret_cast<u64*>(mem.CodePtr() + 0x100);
    *slot = 0x0F0F0F0F0F0F0F0FULL;
    const u8 program[] = {
        0x48, 0xb8, 0,0,0,0,0,0,0,0,
        0x48, 0x21, 0x10, // and qword [rax], rdx
        0xc3,
    };
    u8 prog[sizeof(program)];
    std::memcpy(prog, program, sizeof(program));
    const u64 a = reinterpret_cast<u64>(slot);
    std::memcpy(prog + 2, &a, sizeof(a));
    std::memcpy(mem.CodePtr(), prog, sizeof(prog));
    u8* guest_rsp = mem.StackTop() - 8;
    *reinterpret_cast<u64*>(guest_rsp) = kReturnSentinel;
    GuestState st{};
    st.rip = reinterpret_cast<u64>(mem.CodePtr());
    st.gpr[4] = reinterpret_cast<u64>(guest_rsp);
    st.gpr[2] = 0xF0F0F0F0F0F0F0F0ULL; // complementary: AND = 0

    Runtime rt;
    rt.Run(st);
    EXPECT_EQ(*slot, 0ULL) << "result zero stored";
    EXPECT_EQ(st.rflags & (1ULL<<6), (1ULL<<6)) << "ZF set";
}

// Displaced address, sign bit retained -> SF set.
TEST_F(CpuRuntimeTest, And64_MemRegDisp_SignBit) {
    u64* base = reinterpret_cast<u64*>(mem.CodePtr() + 0x100);
    u64* slot = reinterpret_cast<u64*>(mem.CodePtr() + 0x110); // base + 0x10
    *slot = 0x8000000000000001ULL;
    const u8 program[] = {
        0x48, 0xb8, 0,0,0,0,0,0,0,0,
        0x48, 0x21, 0x50, 0x10, // and qword [rax+0x10], rdx
        0xc3,
    };
    u8 prog[sizeof(program)];
    std::memcpy(prog, program, sizeof(program));
    const u64 a = reinterpret_cast<u64>(base);
    std::memcpy(prog + 2, &a, sizeof(a));
    std::memcpy(mem.CodePtr(), prog, sizeof(prog));
    u8* guest_rsp = mem.StackTop() - 8;
    *reinterpret_cast<u64*>(guest_rsp) = kReturnSentinel;
    GuestState st{};
    st.rip = reinterpret_cast<u64>(mem.CodePtr());
    st.gpr[4] = reinterpret_cast<u64>(guest_rsp);
    st.gpr[2] = 0x8000000000000000ULL; // keep only bit 63

    Runtime rt;
    rt.Run(st);
    EXPECT_EQ(*slot, 0x8000000000000000ULL) << "displaced [mem] &= rdx";
    EXPECT_EQ(st.rflags & (1ULL<<7), (1ULL<<7)) << "SF set: bit 63 of result";
    EXPECT_EQ(st.rflags & (1ULL<<6), 0ULL) << "ZF clear";
}

// JMP unconditional rel32: state.rip should be the target regardless
// of any flags.
//
//   jmp +0x100       e9 00 01 00 00
TEST_F(CpuRuntimeTest, Jmp_DirectRel32_SetsRipToTarget) {
    const u8 program[] = {0xe9, 0x00, 0x01, 0x00, 0x00};
    std::memcpy(mem.CodePtr(), program, sizeof(program));
    const u8 bsr_pad[] = {0x48, 0x0f, 0xbd, 0xd8};
    std::memcpy(mem.CodePtr() + 261, bsr_pad, sizeof(bsr_pad));

    u8* guest_rsp = mem.StackTop() - 8;
    *reinterpret_cast<u64*>(guest_rsp) = kReturnSentinel;

    GuestState state{};
    const u64 program_base = reinterpret_cast<u64>(mem.CodePtr());
    state.rip = program_base;
    state.gpr[4] = reinterpret_cast<u64>(guest_rsp);

    Runtime rt;
    rt.Run(state);

    // jmp rel32 at offset 0 (5 bytes total). next_rip = 5, target = 5 + 0x100 = 261.
    EXPECT_EQ(state.rip, program_base + 261);
}

// JL (signed less than): SF != OF. Test with a CMP that produces
// SF=1, OF=0 — result negative, no overflow. SF != OF → branch taken.
//
//   mov rax, 1       48 c7 c0 01 00 00 00
//   mov rbx, 2       48 c7 c3 02 00 00 00
//   cmp rax, rbx     48 39 d8       ; 1 - 2 = -1, SF=1, OF=0
//   jl  +0x10        7c 10          ; SF != OF → take
TEST_F(CpuRuntimeTest, Jl_TakenWhenSignedLess) {
    const u8 program[] = {
        0x48, 0xc7, 0xc0, 0x01, 0x00, 0x00, 0x00, 0x48, 0xc7, 0xc3,
        0x02, 0x00, 0x00, 0x00, 0x48, 0x39, 0xd8, 0x7c, 0x10,
    };
    std::memcpy(mem.CodePtr(), program, sizeof(program));
    const u8 bsr_pad[] = {0x48, 0x0f, 0xbd, 0xd8};
    std::memcpy(mem.CodePtr() + 35, bsr_pad, sizeof(bsr_pad));

    u8* guest_rsp = mem.StackTop() - 8;
    *reinterpret_cast<u64*>(guest_rsp) = kReturnSentinel;

    GuestState state{};
    const u64 program_base = reinterpret_cast<u64>(mem.CodePtr());
    state.rip = program_base;
    state.gpr[4] = reinterpret_cast<u64>(guest_rsp);

    Runtime rt;
    rt.Run(state);

    // jl at offset 17, next_rip = 19, target = 19 + 0x10 = 35.
    EXPECT_EQ(state.rip, program_base + 35);
}

// CALL + RET round-trip — the multi-block executor test.
//
// Demonstrates: CALL pushes the return address, transfers control to
// the callee through the dispatcher, callee runs and RETs, dispatcher
// re-enters at the post-call instruction, caller continues to its
// own RET which pops the host-return sentinel.
//
// This is the proof that the looping dispatcher and CALL together
// actually execute multi-block code end-to-end.
//
// Caller (offsets 0..15):
//   mov rdi, 0x42       48 c7 c7 42 00 00 00     ; arg
//   call callee         e8 04 00 00 00           ; rel32 = +4 → offset 16
//   mov rcx, rax        48 89 c1                 ; capture return value
//   ret                 c3                        ; return to host sentinel
//
// Callee (offsets 16..23):
//   mov rax, rdi        48 89 f8                 ; return = arg
//   add rax, 1          48 83 c0 01              ; +1
//   ret                 c3                        ; return to caller
//
// Expected: rax = 0x43, rcx = 0x43, rip = host-return sentinel.
TEST_F(CpuRuntimeTest, Call_RoundTripsThroughCalleeBack) {
    const u8 program[] = {
        // Caller
        0x48, 0xc7, 0xc7, 0x42, 0x00, 0x00, 0x00, // mov rdi, 0x42       (0-6)
        0xe8, 0x04, 0x00, 0x00, 0x00,             // call +4 → offset 16 (7-11)
        0x48, 0x89, 0xc1,                         // mov rcx, rax        (12-14)
        0xc3,                                     // ret                 (15)
        // Callee
        0x48, 0x89, 0xf8,       // mov rax, rdi        (16-18)
        0x48, 0x83, 0xc0, 0x01, // add rax, 1          (19-22)
        0xc3,                   // ret                 (23)
    };
    const auto r = RunProgram(program, sizeof(program), mem);

    EXPECT_EQ(r.state.gpr[0], 0x43u) << "RAX should be 0x42 + 1 from callee";
    EXPECT_EQ(r.state.gpr[1], 0x43u) << "RCX should hold the captured return value";
    EXPECT_EQ(r.state.rip, kReturnSentinel) << "Caller's RET should pop the host-return sentinel";
    EXPECT_EQ(r.state.exit_reason, static_cast<u32>(ExitReason::BlockEnd));
}

// LEA r64, [base+disp] — compute an effective address into a register.
// Verifies that LEA writes the *computed address*, not the value at
// that address (i.e. no memory dereference).
//
//   mov rcx, 0x1000     48 c7 c1 00 10 00 00      (offsets 0-6)
//   lea rax, [rcx+5]    48 8d 41 05                (offsets 7-10)
//   ret                 c3                          (offset 11)
//
// Expected: rax = 0x1000 + 5 = 0x1005.
TEST_F(CpuRuntimeTest, Lea_BaseDisp8_ComputesAddress) {
    const u8 program[] = {
        0x48, 0xc7, 0xc1, 0x00, 0x10, 0x00, 0x00, // mov rcx, 0x1000
        0x48, 0x8d, 0x41, 0x05,                   // lea rax, [rcx+5]
        0xc3,                                     // ret
    };
    const auto r = RunProgram(program, sizeof(program), mem);
    EXPECT_EQ(r.state.gpr[0], 0x1005u);
    EXPECT_EQ(r.state.rip, kReturnSentinel);
}

// LEA r64, [base + index*scale + disp] — full SIB addressing.
//
//   mov rcx, 0x1000      48 c7 c1 00 10 00 00         (0-6)
//   mov rdx, 3           48 c7 c2 03 00 00 00         (7-13)
//   lea rax, [rcx+rdx*8+0x10]  48 8d 44 d1 10         (14-18)
//   ret                  c3                            (19)
//
// Expected: rax = 0x1000 + (3 * 8) + 0x10
//             = 0x1000 + 0x18 + 0x10
//             = 0x1028
TEST_F(CpuRuntimeTest, Lea_BaseIndexScaleDisp_ComputesAddress) {
    const u8 program[] = {
        0x48, 0xc7, 0xc1, 0x00, 0x10, 0x00, 0x00, // mov rcx, 0x1000
        0x48, 0xc7, 0xc2, 0x03, 0x00, 0x00, 0x00, // mov rdx, 3
        0x48, 0x8d, 0x44, 0xd1, 0x10,             // lea rax, [rcx+rdx*8+0x10]
        0xc3,                                     // ret
    };
    const auto r = RunProgram(program, sizeof(program), mem);
    EXPECT_EQ(r.state.gpr[0], 0x1028u);
}

// MOV r32, r32 — zero-extends to 64 bits.
//
// Set rax = 0xFFFFFFFF'DEADBEEF, then `mov eax, ebx` where ebx = 0x12345678.
// After the mov, rax should be 0x00000000'12345678 (upper half zeroed).
//
//   mov rax, 0xFFFFFFFFDEADBEEF      48 b8 ef be ad de ff ff ff ff    (0-9)
//   mov rbx, 0x12345678              48 c7 c3 78 56 34 12             (10-16)
//   mov eax, ebx                     89 d8                            (17-18)
//   ret                              c3                                (19)
TEST_F(CpuRuntimeTest, Mov32_RegReg_ZeroExtendsUpper) {
    const u8 program[] = {
        0x48, 0xb8, 0xef, 0xbe, 0xad, 0xde, 0xff, 0xff, 0xff, 0xff, // mov rax, 0xFFFFFFFFDEADBEEF
        0x48, 0xc7, 0xc3, 0x78, 0x56, 0x34, 0x12,                   // mov rbx, 0x12345678
        0x89, 0xd8,                                                 // mov eax, ebx
        0xc3,                                                       // ret
    };
    const auto r = RunProgram(program, sizeof(program), mem);
    EXPECT_EQ(r.state.gpr[0], 0x12345678u)
        << "After mov eax, ebx the upper 32 bits of rax must be zero";
}

// MOV r32, [r64+disp8] — 32-bit load from memory, zero-extends.
//
// Use [rsp-8] for the temporary so we don't clobber the return
// sentinel at [rsp]. The guest stack has space below rsp.
//
//   mov dword [rsp-8], 0x11223344    c7 44 24 f8 44 33 22 11    (0-7)
//   mov eax, dword [rsp-8]           8b 44 24 f8                (8-11)
//   ret                              c3                          (12)
TEST_F(CpuRuntimeTest, Mov32_RegMem_LoadsZeroExtended) {
    const u8 program[] = {
        0xc7, 0x44, 0x24, 0xf8, 0x44, 0x33, 0x22, 0x11, // mov dword [rsp-8], 0x11223344
        0x8b, 0x44, 0x24, 0xf8,                         // mov eax, [rsp-8]
        0xc3,                                           // ret
    };
    const auto r = RunProgram(program, sizeof(program), mem);
    EXPECT_EQ(r.state.gpr[0], 0x11223344u);
    EXPECT_EQ(r.state.rip, kReturnSentinel);
}

// CMP r64, [r64+disp] — verify ZF is set when the values are equal.
//
//   mov rax, 0x55              48 c7 c0 55 00 00 00      (0-6)
//   mov [rsp-8], rax           48 89 44 24 f8            (7-11)
//   mov rcx, 0x55              48 c7 c1 55 00 00 00      (12-18)
//   cmp rcx, [rsp-8]           48 3b 4c 24 f8            (19-23)
//   jz +1                      74 01                     (24-25)
//   ret                        c3                        (26)
//   mov rax, 0x99              48 c7 c0 99 00 00 00      (27-33)
//   ret                        c3                        (34)
//
// rax should hold 0x99 (post-branch path) if cmp set ZF; otherwise 0x55.
TEST_F(CpuRuntimeTest, Cmp_RegMem_SetsZfWhenEqual) {
    const u8 program[] = {
        0x48, 0xc7, 0xc0, 0x55, 0x00, 0x00, 0x00, // mov rax, 0x55
        0x48, 0x89, 0x44, 0x24, 0xf8,             // mov [rsp-8], rax
        0x48, 0xc7, 0xc1, 0x55, 0x00, 0x00, 0x00, // mov rcx, 0x55
        0x48, 0x3b, 0x4c, 0x24, 0xf8,             // cmp rcx, [rsp-8]
        0x74, 0x01,                               // jz +1
        0xc3,                                     // ret (not taken)
        0x48, 0xc7, 0xc0, 0x99, 0x00, 0x00, 0x00, // mov rax, 0x99
        0xc3,                                     // ret
    };
    const auto r = RunProgram(program, sizeof(program), mem);
    EXPECT_EQ(r.state.gpr[0], 0x99u) << "CMP rcx, [rsp-8] should have set ZF, JZ taken, rax = 0x99";
}

// JMP qword [r64+disp] — indirect jump through memory (PLT thunk).
//
// Lay out a function-pointer slot at [rsp-8], pointing to a small
// program tail that sets rax = 0x77 and returns. The JMP loads
// from that slot and transfers control.
//
//   lea rax, [rip+0x10]           48 8d 05 10 00 00 00       (0-6)  → points to label `tail`
//   mov [rsp-8], rax              48 89 44 24 f8             (7-11)
//   jmp qword [rsp-8]             ff 24 24 f8 (we use rsp-relative ModR/M)
//
// Simpler encoding: jmp qword [rsp-8] = ff 64 24 f8 (4 bytes)
//   total prologue = 16 bytes before the jmp = lea(7) + mov(5) + jmp(4)
//   tail (offset 16): mov rax, 0x77; ret
//
//   lea rax, [rip+0x09]           48 8d 05 09 00 00 00      (0-6)  → +next_rip(7)+9 = 16
//   mov [rsp-8], rax              48 89 44 24 f8            (7-11)
//   jmp qword [rsp-8]             ff 64 24 f8               (12-15)
//   mov rax, 0x77                 48 c7 c0 77 00 00 00      (16-22)
//   ret                           c3                        (23)
TEST_F(CpuRuntimeTest, JmpIndirect_Memory_FollowsThroughPointer) {
    const u8 program[] = {
        0x48, 0x8d, 0x05, 0x09, 0x00, 0x00, 0x00, // lea rax, [rip+9]
        0x48, 0x89, 0x44, 0x24, 0xf8,             // mov [rsp-8], rax
        0xff, 0x64, 0x24, 0xf8,                   // jmp qword [rsp-8]
        0x48, 0xc7, 0xc0, 0x77, 0x00, 0x00, 0x00, // mov rax, 0x77
        0xc3,                                     // ret
    };
    const auto r = RunProgram(program, sizeof(program), mem);
    EXPECT_EQ(r.state.gpr[0], 0x77u)
        << "JMP indirect should have transferred control to the tail block";
    EXPECT_EQ(r.state.rip, kReturnSentinel);
}

// XOR r32, r32 — 32-bit XOR; zero-register idiom + zero-extension.
//
// Set rax = 0xFFFFFFFFFFFFFFFF, then xor eax, eax. After the xor
// rax must be 0 (low 32 are zero from the xor, upper 32 zeroed by
// the 32-bit op rule).
//
//   mov rax, -1            48 b8 ff ff ff ff ff ff ff ff      (0-9)
//   xor eax, eax           31 c0                              (10-11)
//   ret                    c3                                  (12)
TEST_F(CpuRuntimeTest, Xor32_RegReg_ZerosRegister) {
    const u8 program[] = {
        0x48, 0xb8, 0xff, 0xff, 0xff, 0xff, 0xff, 0xff, 0xff, 0xff, // mov rax, -1
        0x31, 0xc0,                                                 // xor eax, eax
        0xc3,                                                       // ret
    };
    const auto r = RunProgram(program, sizeof(program), mem);
    EXPECT_EQ(r.state.gpr[0], 0u);
    // Verify ZF is set in flags.
    EXPECT_NE(r.state.rflags & 0x40, 0u) << "XOR result 0 should set ZF";
}

// MOVSXD r64, r32 — sign-extend a 32-bit value to 64 bits.
//
//   mov rbx, 0xFFFFFFFF      48 c7 c3 ff ff ff ff       (sign-ext to 0xFFFFFFFFFFFFFFFF)
//   wait — that's mov rbx, imm32-sign-extended.  Want low 32 = -1, upper 32 don't matter.
//   Actually mov rbx, 0xFFFFFFFFFFFFFFFF via mov rbx,imm32sx gives -1 already.
//   Let me use ebx instead: mov ebx, 0xFFFFFFFF → ebx = -1 (and rbx upper bits zeroed by 32-bit
//   MOV).
//
//   mov ebx, 0xFFFFFFFF      bb ff ff ff ff             (0-4) → ebx = -1, rbx = 0x00000000FFFFFFFF
//   movsxd rax, ebx          48 63 c3                   (5-7) → rax = sext(ebx as int32) = -1
//   ret                      c3                          (8)
//
// Expected: rax = 0xFFFFFFFFFFFFFFFF (i.e., -1 as a 64-bit signed).
TEST_F(CpuRuntimeTest, Movsxd_RegReg_SignExtends) {
    const u8 program[] = {
        0xbb, 0xff, 0xff, 0xff, 0xff, // mov ebx, 0xFFFFFFFF
        0x48, 0x63, 0xc3,             // movsxd rax, ebx
        0xc3,                         // ret
    };
    const auto r = RunProgram(program, sizeof(program), mem);
    EXPECT_EQ(r.state.gpr[0], 0xFFFFFFFFFFFFFFFFull)
        << "MOVSXD should sign-extend, not zero-extend";
}

// CALL r64 (indirect through register).
//
// Load a function pointer into rcx via LEA, then call rcx. The
// callee sets rax = 0x88 and RETs; control returns to the caller's
// RET which pops the host sentinel.
//
// Layout (decisive offsets):
//   lea rcx, [rip+3]       48 8d 0d 03 00 00 00     (0..6)   next_rip=7, +3 = 10
//   call rcx               ff d1                    (7..8)
//   ret  (caller)          c3                       (9)
//   mov rax, 0x88          48 c7 c0 88 00 00 00     (10..16) ← callee entry
//   ret  (callee)          c3                       (17)
TEST_F(CpuRuntimeTest, CallIndirect_Register_RoundTrips) {
    const u8 program[] = {
        0x48, 0x8d, 0x0d, 0x03, 0x00, 0x00, 0x00, // lea rcx, [rip+3]  → offset 10
        0xff, 0xd1,                               // call rcx
        0xc3,                                     // ret (caller's, → sentinel)
        0x48, 0xc7, 0xc0, 0x88, 0x00, 0x00, 0x00, // callee: mov rax, 0x88
        0xc3,                                     // callee: ret  → caller's ret
    };
    const auto r = RunProgram(program, sizeof(program), mem);
    EXPECT_EQ(r.state.gpr[0], 0x88u);
    EXPECT_EQ(r.state.rip, kReturnSentinel);
}

// =============================================================================
// Tier-1 opcode batch: NOP, AND, OR, NOT, NEG, INC, DEC, MOVZX, CMOV.
//
// Each test is focused: it isolates one mnemonic's behavior with
// minimal supporting code, so a regression points at the right
// emit function.
// =============================================================================

// NOP — verify the block compiles and surrounding instructions
// run normally. Tests 1-byte (0x90), 2-byte (0x66 0x90), and the
// 5-byte multi-byte form (0x0F 0x1F /0).
TEST_F(CpuRuntimeTest, Nop_AcceptsAllForms) {
    const u8 program[] = {
        0x48, 0xc7, 0xc0, 0x07, 0x00, 0x00, 0x00, // mov rax, 7
        0x90,                                     // nop (1-byte)
        0x66, 0x90,                               // nop (2-byte, 66 prefix)
        0x0f, 0x1f, 0x40, 0x00,                   // nop dword[rax+0] (4-byte)
        0xc3,                                     // ret
    };
    const auto r = RunProgram(program, sizeof(program), mem);
    EXPECT_EQ(r.state.gpr[0], 7u) << "MOV before NOPs should still be in rax";
    EXPECT_EQ(r.state.rip, kReturnSentinel);
}

// AND r64, r64 — bitwise AND.
TEST_F(CpuRuntimeTest, And64_RegReg_ComputesIntersection) {
    const u8 program[] = {
        0x48, 0xc7, 0xc0, 0x0f, 0x00, 0x00, 0x00, // mov rax, 0x0F
        0x48, 0xc7, 0xc3, 0x33, 0x00, 0x00, 0x00, // mov rbx, 0x33
        0x48, 0x21, 0xd8,                         // and rax, rbx
        0xc3,                                     // ret
    };
    const auto r = RunProgram(program, sizeof(program), mem);
    EXPECT_EQ(r.state.gpr[0], 0x03u); // 0x0F & 0x33 = 0x03
    EXPECT_EQ(r.state.gpr[3], 0x33u); // rbx unchanged
}

// OR r64, r64 — bitwise OR.
TEST_F(CpuRuntimeTest, Or64_RegReg_ComputesUnion) {
    const u8 program[] = {
        0x48, 0xc7, 0xc0, 0x0f, 0x00, 0x00, 0x00, // mov rax, 0x0F
        0x48, 0xc7, 0xc3, 0x30, 0x00, 0x00, 0x00, // mov rbx, 0x30
        0x48, 0x09, 0xd8,                         // or rax, rbx
        0xc3,                                     // ret
    };
    const auto r = RunProgram(program, sizeof(program), mem);
    EXPECT_EQ(r.state.gpr[0], 0x3Fu); // 0x0F | 0x30 = 0x3F
}

// NOT r64 — bitwise complement; flags unchanged.
TEST_F(CpuRuntimeTest, Not_RegisterFlipsBits) {
    const u8 program[] = {
        0x48, 0xc7, 0xc0, 0xff, 0x00, 0x00, 0x00, // mov rax, 0x00FF
        0x48, 0xf7, 0xd0,                         // not rax
        0xc3,                                     // ret
    };
    const auto r = RunProgram(program, sizeof(program), mem);
    EXPECT_EQ(r.state.gpr[0], 0xFFFFFFFFFFFFFF00ULL); // ~0xFF (in 64-bit)
}

// NEG r64 — two's complement. Verifies the value AND that CF is
// set to 1 (because source was nonzero per x86 NEG semantics).
TEST_F(CpuRuntimeTest, Neg_NonZero_SetsCarryAndNegatesValue) {
    const u8 program[] = {
        0x48, 0xc7, 0xc0, 0x05, 0x00, 0x00, 0x00, // mov rax, 5
        0x48, 0xf7, 0xd8,                         // neg rax
        0xc3,                                     // ret
    };
    const auto r = RunProgram(program, sizeof(program), mem);
    EXPECT_EQ(r.state.gpr[0], static_cast<u64>(-5LL)); // two's complement
    // CF (bit 0 of rflags) should be set because source was nonzero.
    EXPECT_EQ(r.state.rflags & 0x1ULL, 0x1ULL);
}

// NEG of zero leaves zero AND clears CF.
TEST_F(CpuRuntimeTest, Neg_Zero_ClearsCarry) {
    const u8 program[] = {
        0x48, 0xc7, 0xc0, 0x00, 0x00, 0x00, 0x00, // mov rax, 0
        0x48, 0xf7, 0xd8,                         // neg rax
        0xc3,                                     // ret
    };
    const auto r = RunProgram(program, sizeof(program), mem);
    EXPECT_EQ(r.state.gpr[0], 0u);
    EXPECT_EQ(r.state.rflags & 0x1ULL, 0u); // CF clear
}

// INC r64 — adds 1 and preserves CF. We deliberately use the
// guest's CF (set via STC via OR rather than STC instruction —
// since we don't have STC yet — by computing a CMP that sets CF)
// to verify INC doesn't clobber it.
//
// Simpler approach: use a SUB that produces CF=0, then INC, and
// verify CF is still 0 afterward. Then run a separate program
// with a SUB that produces CF=1 and verify it's preserved across INC.
TEST_F(CpuRuntimeTest, Inc_PreservesCarryFlag_ClearCase) {
    const u8 program[] = {
        0x48, 0xc7, 0xc0, 0x05, 0x00, 0x00, 0x00, // mov rax, 5
        0x48, 0xc7, 0xc3, 0x03, 0x00, 0x00, 0x00, // mov rbx, 3
        0x48, 0x29, 0xd8,                         // sub rax, rbx → rax=2, CF=0
        0x48, 0xff, 0xc0,                         // inc rax
        0xc3,                                     // ret
    };
    const auto r = RunProgram(program, sizeof(program), mem);
    EXPECT_EQ(r.state.gpr[0], 3u);          // 2 + 1 = 3
    EXPECT_EQ(r.state.rflags & 0x1ULL, 0u); // CF still 0
}

TEST_F(CpuRuntimeTest, Inc_PreservesCarryFlag_SetCase) {
    const u8 program[] = {
        0x48, 0xc7, 0xc0, 0x03, 0x00, 0x00, 0x00, // mov rax, 3
        0x48, 0xc7, 0xc3, 0x05, 0x00, 0x00, 0x00, // mov rbx, 5
        0x48, 0x29, 0xd8,                         // sub rax, rbx → rax=-2 (huge), CF=1
        0x48, 0xff, 0xc0,                         // inc rax
        0xc3,                                     // ret
    };
    const auto r = RunProgram(program, sizeof(program), mem);
    // -2 + 1 = -1.
    EXPECT_EQ(r.state.gpr[0], static_cast<u64>(-1LL));
    // CF was set by SUB-with-borrow and must survive INC.
    EXPECT_EQ(r.state.rflags & 0x1ULL, 0x1ULL);
}

// DEC r64 — subtracts 1 and preserves CF.
TEST_F(CpuRuntimeTest, Dec_PreservesCarryFlag_SetCase) {
    const u8 program[] = {
        0x48, 0xc7, 0xc0, 0x03, 0x00, 0x00, 0x00, // mov rax, 3
        0x48, 0xc7, 0xc3, 0x05, 0x00, 0x00, 0x00, // mov rbx, 5
        0x48, 0x29, 0xd8,                         // sub rax, rbx → CF=1
        0x48, 0xff, 0xc8,                         // dec rax
        0xc3,                                     // ret
    };
    const auto r = RunProgram(program, sizeof(program), mem);
    // -2 - 1 = -3.
    EXPECT_EQ(r.state.gpr[0], static_cast<u64>(-3LL));
    EXPECT_EQ(r.state.rflags & 0x1ULL, 0x1ULL); // CF preserved
}

// DEC sets ZF when result is zero.
TEST_F(CpuRuntimeTest, Dec_FromOne_SetsZeroFlag) {
    const u8 program[] = {
        0x48, 0xc7, 0xc0, 0x01, 0x00, 0x00, 0x00, // mov rax, 1
        0x48, 0xff, 0xc8,                         // dec rax
        0xc3,                                     // ret
    };
    const auto r = RunProgram(program, sizeof(program), mem);
    EXPECT_EQ(r.state.gpr[0], 0u);
    EXPECT_EQ(r.state.rflags & 0x40ULL, 0x40ULL); // ZF bit 6
}

// MOVZX r64, r/m8 — zero-extend byte from register.
TEST_F(CpuRuntimeTest, Movzx_Reg8To64_ZeroExtends) {
    const u8 program[] = {
        // Set rax to 0xFFFFFFFFFFFFFFFF, then load BL = 0x7F (so AL
        // becomes 0x7F but upper bits of rax are all 1s). MOVZX
        // should produce 0x000000000000007F in rdx.
        0x48, 0xc7, 0xc0, 0xff, 0xff, 0xff, 0xff, // mov rax, -1 (sign-extends to all-Fs)
        0x48, 0xc7, 0xc3, 0x7f, 0x00, 0x00, 0x00, // mov rbx, 0x7F
        0x48, 0x0f, 0xb6, 0xd3,                   // movzx rdx, bl
        0xc3,                                     // ret
    };
    const auto r = RunProgram(program, sizeof(program), mem);
    EXPECT_EQ(r.state.gpr[2], 0x7Fu);                 // rdx = zero-extended bl
    EXPECT_EQ(r.state.gpr[0], 0xFFFFFFFFFFFFFFFFULL); // rax unchanged
}

// MOVZX r64, r/m16 — zero-extend word from register.
TEST_F(CpuRuntimeTest, Movzx_Reg16To64_ZeroExtends) {
    const u8 program[] = {
        0x48, 0xc7, 0xc3, 0x34, 0x12, 0x00, 0x00, // mov rbx, 0x1234
        0x48, 0x0f, 0xb7, 0xc3,                   // movzx rax, bx
        0xc3,                                     // ret
    };
    const auto r = RunProgram(program, sizeof(program), mem);
    EXPECT_EQ(r.state.gpr[0], 0x1234u);
}

// MOVZX r64, [mem] — zero-extend byte from memory. Stash a byte
// onto the guest stack via PUSH, MOVZX from [rsp], then rebalance
// the stack with ADD RSP,8 before RET so RET pops the sentinel and
// not the value we just pushed.
TEST_F(CpuRuntimeTest, Movzx_Mem8To64_ZeroExtends) {
    const u8 program[] = {
        0x48, 0xc7, 0xc3, 0xa5, 0x00, 0x00, 0x00, // mov rbx, 0xA5
        0x53,                                     // push rbx
        0x48, 0x0f, 0xb6, 0x04, 0x24,             // movzx rax, byte [rsp]
        0x48, 0x83, 0xc4, 0x08,                   // add rsp, 8 (rebalance)
        0xc3,                                     // ret
    };
    const auto r = RunProgram(program, sizeof(program), mem);
    EXPECT_EQ(r.state.gpr[0], 0xA5u);
    EXPECT_EQ(r.state.rip, kReturnSentinel);
}

// MOVZX r16, r/m8 — destination is 16-bit. The x86-64 ABI rule
// for 16-bit register writes is that the upper 48 bits of the
// underlying 64-bit register are *preserved* (unlike 32-bit
// writes, which zero-extend). A previous version of EmitMovzx
// stored the full qword after the movzx; this regression test
// would have failed under that version because the upper 48
// bits of rax would have been zeroed instead of kept.
//
// Setup: load rax with a sentinel that puts a distinctive
// non-zero pattern in the upper 48 bits. Then load bl with
// 0xCD. Then `movzx ax, bl` — only the low 16 of rax should
// change.
//
// Encoding of `movzx ax, bl`: 66 0f b6 c3 (operand-size override
// 66 selects 16-bit dst; 0f b6 is movzx-from-r/m8; c3 is the
// ModR/M for ax<-bl).
TEST_F(CpuRuntimeTest, Movzx_Reg16To8_PreservesUpper48Bits) {
    const u8 program[] = {
        // mov rax, 0xDEADBEEFCAFE0000
        0x48, 0xb8, 0x00, 0x00, 0xfe, 0xca, 0xef, 0xbe, 0xad,
        0xde, 0x48, 0xc7, 0xc3, 0xcd, 0x00, 0x00, 0x00, // mov rbx, 0xCD
        0x66, 0x0f, 0xb6, 0xc3,                         // movzx ax, bl
        0xc3,                                           // ret
    };
    const auto r = RunProgram(program, sizeof(program), mem);
    // Expected: upper 48 bits preserved, low 16 = 0x00CD.
    EXPECT_EQ(r.state.gpr[0], 0xDEADBEEFCAFE00CDULL)
        << "MOVZX with 16-bit dst must preserve upper 48 bits";
}

// MOVZX r32, r/m8 — destination is 32-bit. x86-64 32-bit writes
// implicitly zero-extend to 64, so upper 32 bits should be zero
// regardless of what was there before.
//
// Encoding of `movzx eax, bl`: 0f b6 c3 (no REX, no 66 prefix).
TEST_F(CpuRuntimeTest, Movzx_Reg32To8_ZeroExtendsUpper32) {
    const u8 program[] = {
        // mov rax, 0xDEADBEEFCAFE0000
        0x48, 0xb8, 0x00, 0x00, 0xfe, 0xca, 0xef, 0xbe, 0xad,
        0xde, 0x48, 0xc7, 0xc3, 0x42, 0x00, 0x00, 0x00, // mov rbx, 0x42
        0x0f, 0xb6, 0xc3,                               // movzx eax, bl
        0xc3,                                           // ret
    };
    const auto r = RunProgram(program, sizeof(program), mem);
    // Expected: full 64 bits = 0x42 (upper 32 zeroed by x86-64
    // write rule; low 32 = zero-extended byte = 0x42).
    EXPECT_EQ(r.state.gpr[0], 0x42u) << "MOVZX with 32-bit dst must zero-extend to 64 bits";
}

// CMOV taken: condition true, dst should be replaced with src.
//
// Set rax=10, rbx=20, then CMP rax with 0 (rax > 0 ⇒ NE), then
// CMOVNZ rax, rbx. Expect rax = 20 (the move was taken).
TEST_F(CpuRuntimeTest, CmovNz_ConditionTrue_TakesMove) {
    const u8 program[] = {
        0x48, 0xc7, 0xc0, 0x0a, 0x00, 0x00, 0x00, // mov rax, 10
        0x48, 0xc7, 0xc3, 0x14, 0x00, 0x00, 0x00, // mov rbx, 20
        0x48, 0x83, 0xf8, 0x00,                   // cmp rax, 0       (sets ZF=0)
        0x48, 0x0f, 0x45, 0xc3,                   // cmovnz rax, rbx
        0xc3,                                     // ret
    };
    const auto r = RunProgram(program, sizeof(program), mem);
    EXPECT_EQ(r.state.gpr[0], 20u);
}

// CMOV not taken: condition false, dst should be unchanged.
//
// Set rax=0, rbx=20, then CMP rax with 0 (ZF=1), then CMOVNZ
// rax, rbx. Expect rax = 0 (move was NOT taken).
TEST_F(CpuRuntimeTest, CmovNz_ConditionFalse_LeavesDstAlone) {
    const u8 program[] = {
        0x48, 0xc7, 0xc0, 0x00, 0x00, 0x00, 0x00, // mov rax, 0
        0x48, 0xc7, 0xc3, 0x14, 0x00, 0x00, 0x00, // mov rbx, 20
        0x48, 0x83, 0xf8, 0x00,                   // cmp rax, 0       (sets ZF=1)
        0x48, 0x0f, 0x45, 0xc3,                   // cmovnz rax, rbx
        0xc3,                                     // ret
    };
    const auto r = RunProgram(program, sizeof(program), mem);
    EXPECT_EQ(r.state.gpr[0], 0u);
}

// CMOV with signed-less condition (the branchless-min idiom).
// rax = 5, rbx = 3. cmp rax, rbx ⇒ rax > rbx ⇒ SF==OF, JL would
// NOT take. CMOVL is "less than" (SF != OF), so move NOT taken.
// Expect rax stays 5.
TEST_F(CpuRuntimeTest, CmovL_NotTakenWhenSourceIsGreater) {
    const u8 program[] = {
        0x48, 0xc7, 0xc0, 0x05, 0x00, 0x00, 0x00, // mov rax, 5
        0x48, 0xc7, 0xc3, 0x03, 0x00, 0x00, 0x00, // mov rbx, 3
        0x48, 0x39, 0xd8,                         // cmp rax, rbx
        0x48, 0x0f, 0x4c, 0xc3,                   // cmovl rax, rbx
        0xc3,                                     // ret
    };
    const auto r = RunProgram(program, sizeof(program), mem);
    EXPECT_EQ(r.state.gpr[0], 5u); // unchanged: rax not < rbx
}

// =============================================================================
// Tier-2 opcode batch: SHL/SHR/SAR (imm8 and CL forms),
// 8-bit MOV variants, 16-bit MOV variants.
// =============================================================================

// SHL rax, 4 — left shift by immediate. Multiplies by 16.
// SHR ax, imm — 16-bit logical right shift. Must shift only the low
// word and PRESERVE the upper 48 bits of rax (16-bit register writes do
// not zero-extend). Regression test for EmitShift16 (eboot 0x80000378f
// `shr r/m16, imm`, which previously fell through to EmitShift64 and was
// rejected as unsupported).
// VROUNDSS xmm0, xmm0, xmm1, 1 — round-down (floor) the scalar single in
// xmm1's low lane into xmm0's low lane, preserving xmm0's upper bits.
// Regression for EmitVroundss (eboot 0x8000382fa). imm=1 selects floor.
// VMOVSLDUP xmm0, xmm1 — duplicate even-indexed single floats:
// dst[0]=dst[1]=src[0], dst[2]=dst[3]=src[2]. Regression for EmitMovDup
// (eboot 0x800038388). VEX-128 zeros the upper YMM half of dst.
TEST_F(CpuRuntimeTest, Vmovsldup_DuplicatesEvenLanes) {
    u8* guest_rsp = mem.StackTop() - 8;
    *reinterpret_cast<u64*>(guest_rsp) = kReturnSentinel;

    const u8 program[] = {
        0xc5, 0xfa, 0x12, 0xc1, // vmovsldup xmm0, xmm1
        0xc3,                   // ret
    };
    std::memcpy(mem.CodePtr(), program, sizeof(program));

    GuestState state{};
    state.rip = reinterpret_cast<u64>(mem.CodePtr());
    state.gpr[4] = reinterpret_cast<u64>(guest_rsp);

    // xmm1 lanes: [0]=1.0, [1]=2.0, [2]=3.0, [3]=4.0.
    // ymm[4] = low 64 (lanes 0,1), ymm[5] = high 64 (lanes 2,3).
    const auto pack = [](float lo, float hi) {
        return static_cast<u64>(std::bit_cast<u32>(lo)) |
               (static_cast<u64>(std::bit_cast<u32>(hi)) << 32);
    };
    state.ymm[4] = pack(1.0f, 2.0f); // xmm1 lanes 0,1
    state.ymm[5] = pack(3.0f, 4.0f); // xmm1 lanes 2,3
    // Dirty xmm0 upper YMM half to confirm it gets zeroed.
    state.ymm[2] = 0xFFFFFFFFFFFFFFFFull;
    state.ymm[3] = 0xFFFFFFFFFFFFFFFFull;

    Runtime rt;
    rt.Run(state);

    // Expect xmm0 lanes = [1.0, 1.0, 3.0, 3.0].
    EXPECT_EQ(state.ymm[0], pack(1.0f, 1.0f)) << "xmm0 lanes 0,1 should both be src[0]=1.0";
    EXPECT_EQ(state.ymm[1], pack(3.0f, 3.0f)) << "xmm0 lanes 2,3 should both be src[2]=3.0";
    // VEX-128 zeroed the upper half.
    EXPECT_EQ(state.ymm[2], 0u);
    EXPECT_EQ(state.ymm[3], 0u);
    EXPECT_EQ(state.rip, kReturnSentinel);
}

TEST_F(CpuRuntimeTest, Vroundss_FloorScalarSingle) {
    u8* guest_rsp = mem.StackTop() - 8;
    *reinterpret_cast<u64*>(guest_rsp) = kReturnSentinel;

    const u8 program[] = {
        0xc4, 0xe3, 0x79, 0x0a, 0xc1, 0x01, // vroundss xmm0, xmm0, xmm1, 1
        0xc3,                               // ret
    };
    std::memcpy(mem.CodePtr(), program, sizeof(program));

    GuestState state{};
    state.rip = reinterpret_cast<u64>(mem.CodePtr());
    state.gpr[4] = reinterpret_cast<u64>(guest_rsp);
    // xmm1 low float = 3.7; xmm0 low float = some sentinel we expect replaced,
    // xmm0 upper 96 bits = a marker we expect PRESERVED.
    const float in = 3.7f;
    state.ymm[4] = static_cast<u64>(std::bit_cast<u32>(in)); // xmm1 low 32
    // xmm0: low32 = 0 (will be overwritten), bits 32..127 = marker.
    state.ymm[0] = 0xDEADBEEF00000000ull; // low32=0, next32=0xDEADBEEF
    state.ymm[1] = 0xCAFEF00DCAFEBABEull; // bits 64..127 (preserved)

    Runtime rt;
    rt.Run(state);

    // Low 32 of xmm0 = floor(3.7) = 3.0.
    const auto low = static_cast<u32>(state.ymm[0] & 0xFFFFFFFFu);
    EXPECT_EQ(std::bit_cast<float>(low), 3.0f);
    // Upper 96 bits of xmm0 preserved from src1 (=xmm0 input).
    EXPECT_EQ(state.ymm[0] >> 32, 0xDEADBEEFu);
    EXPECT_EQ(state.ymm[1], 0xCAFEF00DCAFEBABEull);
    EXPECT_EQ(state.rip, kReturnSentinel);
}

TEST_F(CpuRuntimeTest, Shr_Imm16_PreservesUpperBits) {
    const u8 program[] = {
        // mov rax, 0xAAAABBBB0000F000 — upper 48 bits nonzero, low word 0xF000
        0x48, 0xb8, 0x00, 0xf0, 0x00, 0x00, 0xbb, 0xbb, 0xaa, 0xaa,
        0x66, 0xc1, 0xe8, 0x04,                   // shr ax, 4
        0xc3,                                     // ret
    };
    const auto r = RunProgram(program, sizeof(program), mem);
    // Low word: 0xF000 >> 4 = 0x0F00. Upper 48 bits unchanged.
    EXPECT_EQ(r.state.gpr[0], 0xAAAABBBB00000F00ull);
}

// SHL ax, imm — 16-bit left shift, also preserving upper bits, and the
// 16-bit result must wrap within the word (no carry into bits 16+).
TEST_F(CpuRuntimeTest, Shl_Imm16_WrapsWithinWordAndPreservesUpper) {
    const u8 program[] = {
        // mov rax, 0x1111222200001234
        0x48, 0xb8, 0x34, 0x12, 0x00, 0x00, 0x22, 0x22, 0x11, 0x11,
        0x66, 0xc1, 0xe0, 0x08,                   // shl ax, 8
        0xc3,                                     // ret
    };
    const auto r = RunProgram(program, sizeof(program), mem);
    // 0x1234 << 8 = 0x123400, truncated to 16 bits = 0x3400. Upper kept.
    EXPECT_EQ(r.state.gpr[0], 0x1111222200003400ull);
}

// SAR ax, imm — 16-bit arithmetic right shift sign-extends within the
// word (0x8000-set value stays negative), upper 48 bits preserved.
TEST_F(CpuRuntimeTest, Sar_Imm16_SignExtendsWithinWord) {
    const u8 program[] = {
        // mov rax, 0x000000000000FF00 (low word 0xFF00, high bit set)
        0x48, 0xb8, 0x00, 0xff, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00,
        0x66, 0xc1, 0xf8, 0x04,                   // sar ax, 4
        0xc3,                                     // ret
    };
    const auto r = RunProgram(program, sizeof(program), mem);
    // 0xFF00 as s16 = -256; >>4 arithmetic = -16 = 0xFFF0 in the low word.
    EXPECT_EQ(r.state.gpr[0], 0x000000000000FFF0ull);
}

TEST_F(CpuRuntimeTest, Shl_Imm8_LeftShifts) {
    const u8 program[] = {
        0x48, 0xc7, 0xc0, 0x03, 0x00, 0x00, 0x00, // mov rax, 3
        0x48, 0xc1, 0xe0, 0x04,                   // shl rax, 4
        0xc3,                                     // ret
    };
    const auto r = RunProgram(program, sizeof(program), mem);
    EXPECT_EQ(r.state.gpr[0], 3u << 4); // = 48
}

// SHR rax, 1 — logical right shift by immediate.
TEST_F(CpuRuntimeTest, Shr_Imm8_RightShifts) {
    const u8 program[] = {
        0x48, 0xc7, 0xc0, 0x80, 0x00, 0x00, 0x00, // mov rax, 128
        0x48, 0xd1, 0xe8,                         // shr rax, 1 (1-bit form)
        0xc3,                                     // ret
    };
    const auto r = RunProgram(program, sizeof(program), mem);
    EXPECT_EQ(r.state.gpr[0], 64u);
}

// SAR (arithmetic right shift) — preserves sign bit when input is
// negative. Load rax with -16, SAR by 2, expect -4.
TEST_F(CpuRuntimeTest, Sar_Imm8_SignExtendsNegative) {
    const u8 program[] = {
        // mov rax, -16 (sign-extended 32-bit imm in the
        // standard "mov rax, imm32" encoding)
        0x48, 0xc7, 0xc0, 0xf0, 0xff, 0xff, 0xff, // mov rax, -16
        0x48, 0xc1, 0xf8, 0x02,                   // sar rax, 2
        0xc3,                                     // ret
    };
    const auto r = RunProgram(program, sizeof(program), mem);
    EXPECT_EQ(r.state.gpr[0], static_cast<u64>(-4LL));
}

// SHL via CL — dynamic shift count.
TEST_F(CpuRuntimeTest, Shl_Cl_LeftShiftsByDynamicCount) {
    const u8 program[] = {
        0x48, 0xc7, 0xc0, 0x01, 0x00, 0x00, 0x00, // mov rax, 1
        0x48, 0xc7, 0xc1, 0x05, 0x00, 0x00, 0x00, // mov rcx, 5
        0x48, 0xd3, 0xe0,                         // shl rax, cl
        0xc3,                                     // ret
    };
    const auto r = RunProgram(program, sizeof(program), mem);
    EXPECT_EQ(r.state.gpr[0], 1u << 5); // = 32
}

// Shift by zero must preserve all flags. We deliberately set up a
// known flag state via SUB, then SHL by 0, then verify CF is still
// set from the SUB.
//
// This is the most important shift-semantics test — it would fail
// with a naive implementation that always overwrites rflags.
TEST_F(CpuRuntimeTest, ShlByZero_PreservesFlags) {
    const u8 program[] = {
        0x48, 0xc7, 0xc0, 0x03, 0x00, 0x00, 0x00, // mov rax, 3
        0x48, 0xc7, 0xc3, 0x05, 0x00, 0x00, 0x00, // mov rbx, 5
        0x48, 0x29, 0xd8,                         // sub rax, rbx → CF=1
        0x48, 0xc7, 0xc1, 0x00, 0x00, 0x00, 0x00, // mov rcx, 0    (shift count)
        0x48, 0xd3, 0xe0,                         // shl rax, cl   (no-op)
        0xc3,                                     // ret
    };
    const auto r = RunProgram(program, sizeof(program), mem);
    // rax should be unchanged (-2 in two's complement).
    EXPECT_EQ(r.state.gpr[0], static_cast<u64>(-2LL));
    // CF must still be set from the SUB.
    EXPECT_EQ(r.state.rflags & 0x1ULL, 0x1ULL);
}

// SHL sets CF to the last bit shifted out. Shifting 0x80 (high bit
// of byte, in bit 7) left by 1 should leave CF=0 because the bit
// that fell off was bit 7 of a 64-bit value — and bit 7 doesn't
// fall off a 64-bit shift. Let's use a value with a bit in
// position 63 instead.
TEST_F(CpuRuntimeTest, Shl_CarriesOutHighBit) {
    const u8 program[] = {
        // mov rax, 0x8000000000000000 — high bit set
        0x48, 0xb8, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x80, 0x48, 0xd1, 0xe0, // shl rax, 1
        0xc3,                                                                         // ret
    };
    const auto r = RunProgram(program, sizeof(program), mem);
    EXPECT_EQ(r.state.gpr[0], 0u);              // bit shifted out, nothing left
    EXPECT_EQ(r.state.rflags & 0x1ULL, 0x1ULL); // CF = 1 (the bit fell off)
}

// 8-bit MOV r,r — preserves upper bits of dst slot (like MOVZX r16).
TEST_F(CpuRuntimeTest, Mov8_RegReg_PreservesUpperBits) {
    const u8 program[] = {
        // mov rax, 0xDEADBEEFCAFE0000
        0x48, 0xb8, 0x00, 0x00, 0xfe, 0xca, 0xef, 0xbe, 0xad,
        0xde, 0x48, 0xc7, 0xc3, 0x42, 0x00, 0x00, 0x00, // mov rbx, 0x42
        0x88, 0xd8,                                     // mov al, bl
        0xc3,                                           // ret
    };
    const auto r = RunProgram(program, sizeof(program), mem);
    // Low byte replaced with 0x42; upper 56 preserved.
    EXPECT_EQ(r.state.gpr[0], 0xDEADBEEFCAFE0042ULL);
}

// 8-bit MOV r,imm.
TEST_F(CpuRuntimeTest, Mov8_RegImm_WritesLowByte) {
    const u8 program[] = {
        0x48, 0xb8,                                                 // mov rax, 0xFFFFFFFFFFFFFFFF
        0xff, 0xff, 0xff, 0xff, 0xff, 0xff, 0xff, 0xff, 0xb0, 0x7f, // mov al, 0x7F
        0xc3,                                                       // ret
    };
    const auto r = RunProgram(program, sizeof(program), mem);
    EXPECT_EQ(r.state.gpr[0], 0xFFFFFFFFFFFFFF7FULL);
}

// 8-bit MOV r,[mem] and m,r round-trip.
//
// Setup: write a byte via `mov byte[rbx], al`, then read it back
// via `mov cl, byte[rbx]`. The pointer is into the guest data area
// (we'll position rbx there). This exercises both memory forms.
TEST_F(CpuRuntimeTest, Mov8_MemRegRoundTrip) {
    const u64 addr = reinterpret_cast<u64>(mem.CodePtr() + GuestMemory::PAGE_SIZE);

    u8 program[] = {
        // mov rax, 0xA7
        0x48,
        0xc7,
        0xc0,
        0xa7,
        0x00,
        0x00,
        0x00,
        // mov rbx, <addr>
        0x48,
        0xbb,
        0,
        0,
        0,
        0,
        0,
        0,
        0,
        0,
        // mov byte[rbx], al
        0x88,
        0x03,
        // mov cl, byte[rbx]
        0x8a,
        0x0b,
        // ret
        0xc3,
    };
    std::memcpy(&program[9], &addr, sizeof(addr));

    const auto r = RunProgram(program, sizeof(program), mem);
    // cl should hold the byte we wrote.
    EXPECT_EQ(r.state.gpr[1] & 0xFFu, 0xA7u);
}

// 16-bit MOV r,r — preserves upper 48 bits.
TEST_F(CpuRuntimeTest, Mov16_RegReg_PreservesUpper48Bits) {
    const u8 program[] = {
        // mov rax, 0xDEADBEEFCAFE0000
        0x48,
        0xb8,
        0x00,
        0x00,
        0xfe,
        0xca,
        0xef,
        0xbe,
        0xad,
        0xde,
        // mov rbx, 0x1234
        0x48,
        0xc7,
        0xc3,
        0x34,
        0x12,
        0x00,
        0x00,
        // mov ax, bx          (66 prefix + 89 d8)
        0x66,
        0x89,
        0xd8,
        0xc3,
    };
    const auto r = RunProgram(program, sizeof(program), mem);
    EXPECT_EQ(r.state.gpr[0], 0xDEADBEEFCAFE1234ULL);
}

// 16-bit MOV r,imm.
TEST_F(CpuRuntimeTest, Mov16_RegImm_WritesLowWord) {
    const u8 program[] = {
        0x48,
        0xb8, // mov rax, all-Fs
        0xff,
        0xff,
        0xff,
        0xff,
        0xff,
        0xff,
        0xff,
        0xff,
        // mov ax, 0xCAFE      (66 b8 fe ca)
        0x66,
        0xb8,
        0xfe,
        0xca,
        0xc3,
    };
    const auto r = RunProgram(program, sizeof(program), mem);
    EXPECT_EQ(r.state.gpr[0], 0xFFFFFFFFFFFFCAFEULL);
}

// 16-bit MOV memory round-trip.
TEST_F(CpuRuntimeTest, Mov16_MemRegRoundTrip) {
    const u64 addr = reinterpret_cast<u64>(mem.CodePtr() + GuestMemory::PAGE_SIZE);

    u8 program[] = {
        // mov rax, 0xC0DE
        0x48,
        0xc7,
        0xc0,
        0xde,
        0xc0,
        0x00,
        0x00,
        // mov rbx, <addr>
        0x48,
        0xbb,
        0,
        0,
        0,
        0,
        0,
        0,
        0,
        0,
        // mov word[rbx], ax       66 89 03
        0x66,
        0x89,
        0x03,
        // mov cx, word[rbx]       66 8b 0b
        0x66,
        0x8b,
        0x0b,
        0xc3,
    };
    std::memcpy(&program[9], &addr, sizeof(addr));

    const auto r = RunProgram(program, sizeof(program), mem);
    EXPECT_EQ(r.state.gpr[1] & 0xFFFFu, 0xC0DEu);
}

// =============================================================================
// Tier-3 opcode batch: IMUL (all three forms), ROL/ROR, CDQE/CDQ/
// CWDE/CQO, STC/CLC/CMC.
// =============================================================================

// IMUL r64, r64 (2-op form) — most common C-style multiply.
TEST_F(CpuRuntimeTest, Imul_2Op_Reg_MultipliesIntoDst) {
    const u8 program[] = {
        0x48, 0xc7, 0xc0, 0x07, 0x00, 0x00, 0x00, // mov rax, 7
        0x48, 0xc7, 0xc3, 0x06, 0x00, 0x00, 0x00, // mov rbx, 6
        0x48, 0x0f, 0xaf, 0xc3,                   // imul rax, rbx
        0xc3,                                     // ret
    };
    const auto r = RunProgram(program, sizeof(program), mem);
    EXPECT_EQ(r.state.gpr[0], 42u);
    EXPECT_EQ(r.state.gpr[3], 6u); // rbx unchanged
}

// IMUL r64, r64, imm32 (3-op form) — compiler emits this for
// `x * constant` where constant fits in imm32.
TEST_F(CpuRuntimeTest, Imul_3Op_ImmediateConstant) {
    const u8 program[] = {
        0x48, 0xc7, 0xc3, 0x05, 0x00, 0x00, 0x00, // mov rbx, 5
        // imul rax, rbx, 100   →   48 69 c3 64 00 00 00
        0x48, 0x69, 0xc3, 0x64, 0x00, 0x00, 0x00,
        0xc3, // ret
    };
    const auto r = RunProgram(program, sizeof(program), mem);
    EXPECT_EQ(r.state.gpr[0], 500u);
    EXPECT_EQ(r.state.gpr[3], 5u); // rbx unchanged
}

// IMUL r64, r64, imm8 (3-op form, short immediate).
TEST_F(CpuRuntimeTest, Imul_3Op_ImmediateImm8) {
    const u8 program[] = {
        0x48, 0xc7, 0xc3, 0x09, 0x00, 0x00, 0x00, // mov rbx, 9
        // imul rax, rbx, 11   →   48 6b c3 0b
        0x48, 0x6b, 0xc3, 0x0b,
        0xc3, // ret
    };
    const auto r = RunProgram(program, sizeof(program), mem);
    EXPECT_EQ(r.state.gpr[0], 99u);
}

// IMUL r/m64 (1-op form) — full 128-bit signed multiply.
// 2 * 3 = 6. Low half in RAX = 6, high half in RDX = 0.
TEST_F(CpuRuntimeTest, Imul_1Op_FullPrecisionMul) {
    const u8 program[] = {
        0x48, 0xc7, 0xc0, 0x02, 0x00, 0x00, 0x00, // mov rax, 2
        0x48, 0xc7, 0xc3, 0x03, 0x00, 0x00, 0x00, // mov rbx, 3
        0x48, 0xf7, 0xeb,                         // imul rbx   (rdx:rax = rax * rbx)
        0xc3,                                     // ret
    };
    const auto r = RunProgram(program, sizeof(program), mem);
    EXPECT_EQ(r.state.gpr[0], 6u); // RAX = low half
    EXPECT_EQ(r.state.gpr[2], 0u); // RDX = high half
}

// IMUL 1-op with a result that exceeds 64 bits — verifies the
// high half lands in RDX.
//
// Set RAX = 0x100000000, RBX = 0x100000000. Product = 0x10000000000000000
// which is exactly 2^64, so RAX = 0 and RDX = 1.
TEST_F(CpuRuntimeTest, Imul_1Op_OverflowsIntoHighHalf) {
    const u8 program[] = {
        // mov rax, 0x100000000
        0x48,
        0xb8,
        0x00,
        0x00,
        0x00,
        0x00,
        0x01,
        0x00,
        0x00,
        0x00,
        // mov rbx, 0x100000000
        0x48,
        0xbb,
        0x00,
        0x00,
        0x00,
        0x00,
        0x01,
        0x00,
        0x00,
        0x00,
        0x48,
        0xf7,
        0xeb, // imul rbx
        0xc3,
    };
    const auto r = RunProgram(program, sizeof(program), mem);
    EXPECT_EQ(r.state.gpr[0], 0u); // RAX = low half = 0
    EXPECT_EQ(r.state.gpr[2], 1u); // RDX = high half = 1
}

// IMUL with negative result — signed semantics.
TEST_F(CpuRuntimeTest, Imul_2Op_NegativeProduct) {
    const u8 program[] = {
        // mov rax, -3 (sign-extended imm32)
        0x48, 0xc7, 0xc0, 0xfd, 0xff, 0xff, 0xff,
        0x48, 0xc7, 0xc3, 0x07, 0x00, 0x00, 0x00, // mov rbx, 7
        0x48, 0x0f, 0xaf, 0xc3,                   // imul rax, rbx
        0xc3,
    };
    const auto r = RunProgram(program, sizeof(program), mem);
    EXPECT_EQ(r.state.gpr[0], static_cast<u64>(-21LL));
}

// ROL r64, imm — rotate left by 4. 0xF000000000000001 << 4 (rotated)
// should produce 0x000000000000001F (the high nibble wraps to low).
TEST_F(CpuRuntimeTest, Rol_Imm_RotatesLeftWithWrap) {
    const u8 program[] = {
        // mov rax, 0xF000000000000001
        0x48, 0xb8, 0x01, 0x00, 0x00, 0x00, 0x00,
        0x00, 0x00, 0xf0, 0x48, 0xc1, 0xc0, 0x04, // rol rax, 4
        0xc3,
    };
    const auto r = RunProgram(program, sizeof(program), mem);
    EXPECT_EQ(r.state.gpr[0], 0x000000000000001FULL);
}

// ROR r64, imm — rotate right by 4. 0x000000000000001F rotated right
// by 4 produces 0xF000000000000001 (low nibble wraps to high).
TEST_F(CpuRuntimeTest, Ror_Imm_RotatesRightWithWrap) {
    const u8 program[] = {
        0x48, 0xc7, 0xc0, 0x1f, 0x00, 0x00, 0x00, // mov rax, 0x1F
        0x48, 0xc1, 0xc8, 0x04,                   // ror rax, 4
        0xc3,
    };
    const auto r = RunProgram(program, sizeof(program), mem);
    EXPECT_EQ(r.state.gpr[0], 0xF000000000000001ULL);
}

// ROL by zero must preserve flags (same rule as shifts). Set CF
// via SUB, then rotate by 0, verify CF intact.
TEST_F(CpuRuntimeTest, RolByZero_PreservesFlags) {
    const u8 program[] = {
        0x48, 0xc7, 0xc0, 0x03, 0x00, 0x00, 0x00, // mov rax, 3
        0x48, 0xc7, 0xc3, 0x05, 0x00, 0x00, 0x00, // mov rbx, 5
        0x48, 0x29, 0xd8,                         // sub rax, rbx → CF=1
        0x48, 0xc7, 0xc1, 0x00, 0x00, 0x00, 0x00, // mov rcx, 0
        0x48, 0xd3, 0xc0,                         // rol rax, cl   (no-op)
        0xc3,
    };
    const auto r = RunProgram(program, sizeof(program), mem);
    EXPECT_EQ(r.state.gpr[0], static_cast<u64>(-2LL));
    EXPECT_EQ(r.state.rflags & 0x1ULL, 0x1ULL); // CF still set
}

// CDQE — sign-extend EAX into RAX.
TEST_F(CpuRuntimeTest, Cdqe_SignExtendsNegativeEax) {
    const u8 program[] = {
        // mov eax, -1   →   b8 ff ff ff ff (5 bytes — 32-bit imm)
        0xb8, 0xff, 0xff, 0xff, 0xff, 0x48, 0x98, // cdqe
        0xc3,
    };
    const auto r = RunProgram(program, sizeof(program), mem);
    EXPECT_EQ(r.state.gpr[0], 0xFFFFFFFFFFFFFFFFULL);
}

// CDQE with non-negative EAX — upper 32 should be zero.
TEST_F(CpuRuntimeTest, Cdqe_ZeroExtendsPositiveEax) {
    const u8 program[] = {
        // mov eax, 0x7FFFFFFF (largest positive s32)
        0xb8, 0xff, 0xff, 0xff, 0x7f, 0x48, 0x98, // cdqe
        0xc3,
    };
    const auto r = RunProgram(program, sizeof(program), mem);
    EXPECT_EQ(r.state.gpr[0], 0x000000007FFFFFFFULL);
}

// CDQ — sign-extend EAX into EDX. Negative case.
TEST_F(CpuRuntimeTest, Cdq_NegativeEax_FillsEdxWithOnes) {
    const u8 program[] = {
        0xb8, 0xff, 0xff, 0xff, 0xff, // mov eax, -1
        0x99,                         // cdq
        0xc3,
    };
    const auto r = RunProgram(program, sizeof(program), mem);
    // After CDQ on negative EAX: EDX = 0xFFFFFFFF, upper 32 of RDX = 0.
    EXPECT_EQ(r.state.gpr[2], 0x00000000FFFFFFFFULL);
}

// CDQ with positive EAX — EDX cleared, upper 32 of RDX cleared too.
TEST_F(CpuRuntimeTest, Cdq_PositiveEax_ClearsEdx) {
    const u8 program[] = {
        // mov rdx, 0xDEADBEEFCAFEBABE — make sure CDQ actually clears it
        0x48, 0xba, 0xbe, 0xba, 0xfe, 0xca, 0xef, 0xbe,
        0xad, 0xde, 0xb8, 0x42, 0x00, 0x00, 0x00, // mov eax, 0x42
        0x99,                                     // cdq
        0xc3,
    };
    const auto r = RunProgram(program, sizeof(program), mem);
    EXPECT_EQ(r.state.gpr[2], 0u); // RDX fully cleared
}

// CQO — sign-extend RAX into RDX.
TEST_F(CpuRuntimeTest, Cqo_NegativeRax_FillsRdxWithOnes) {
    const u8 program[] = {
        // mov rax, -1 (sign-extended imm32)
        0x48, 0xc7, 0xc0, 0xff, 0xff, 0xff, 0xff, 0x48, 0x99, // cqo
        0xc3,
    };
    const auto r = RunProgram(program, sizeof(program), mem);
    EXPECT_EQ(r.state.gpr[2], 0xFFFFFFFFFFFFFFFFULL);
}

// STC — set carry flag.
TEST_F(CpuRuntimeTest, Stc_SetsCarryFlag) {
    const u8 program[] = {
        0xf9, // stc
        0xc3,
    };
    const auto r = RunProgram(program, sizeof(program), mem);
    EXPECT_EQ(r.state.rflags & 0x1ULL, 0x1ULL);
}

// CLC — clear carry flag. First set CF via SUB, then CLC, verify cleared.
TEST_F(CpuRuntimeTest, Clc_ClearsCarryFlag) {
    const u8 program[] = {
        0x48, 0xc7, 0xc0, 0x03, 0x00, 0x00, 0x00, // mov rax, 3
        0x48, 0xc7, 0xc3, 0x05, 0x00, 0x00, 0x00, // mov rbx, 5
        0x48, 0x29, 0xd8,                         // sub rax, rbx → CF=1
        0xf8,                                     // clc
        0xc3,
    };
    const auto r = RunProgram(program, sizeof(program), mem);
    EXPECT_EQ(r.state.rflags & 0x1ULL, 0u);
}

// CMC — complement carry flag. SUB sets CF=1, CMC flips to 0.
TEST_F(CpuRuntimeTest, Cmc_TogglesCarryFromOneToZero) {
    const u8 program[] = {
        0x48, 0xc7, 0xc0, 0x03, 0x00, 0x00, 0x00, // mov rax, 3
        0x48, 0xc7, 0xc3, 0x05, 0x00, 0x00, 0x00, // mov rbx, 5
        0x48, 0x29, 0xd8,                         // sub rax, rbx → CF=1
        0xf5,                                     // cmc
        0xc3,
    };
    const auto r = RunProgram(program, sizeof(program), mem);
    EXPECT_EQ(r.state.rflags & 0x1ULL, 0u);
}

// CMC — from 0 to 1.
TEST_F(CpuRuntimeTest, Cmc_TogglesCarryFromZeroToOne) {
    const u8 program[] = {
        0xf8, // clc → CF=0
        0xf5, // cmc → CF=1
        0xc3,
    };
    const auto r = RunProgram(program, sizeof(program), mem);
    EXPECT_EQ(r.state.rflags & 0x1ULL, 0x1ULL);
}

// =============================================================================
// Tier-4 opcode batch: LEAVE, ADC/SBB, 8/16-bit ADD/SUB/CMP.
// =============================================================================

// LEAVE — function epilogue. Tests against a realistic frame
// setup that compilers actually emit:
//
//   push rax     ; (stand-in for `push rbp` — we want a known
//                ;  sentinel as the "saved rbp" so we can verify
//                ;  that LEAVE restores it correctly)
//   mov rbp, rsp ; frame pointer = stack pointer
//   <body>
//   leave        ; mov rsp, rbp; pop rbp  (pops what we pushed)
//   ret          ; pops the host-return sentinel
//
// After this sequence: rbp should equal the value we originally
// pushed, and RIP should be the host-return sentinel (proving
// RSP was restored to the right level for RET to find the
// sentinel on top).
TEST_F(CpuRuntimeTest, Leave_TearsDownFrame) {
    const u8 program[] = {
        // mov rax, 0xAABBCCDDEEFF0011 — the "saved rbp" sentinel
        0x48, 0xb8, 0x11, 0x00, 0xff, 0xee, 0xdd, 0xcc, 0xbb, 0xaa,
        0x50,             // push rax (save sentinel as saved rbp)
        0x48, 0x89, 0xe5, // mov rbp, rsp (set up frame pointer)
        // ... no body needed ...
        0xc9, // leave (mov rsp, rbp; pop rbp)
        0xc3, // ret  (pops host sentinel)
    };
    const auto r = RunProgram(program, sizeof(program), mem);
    EXPECT_EQ(r.state.gpr[5], 0xAABBCCDDEEFF0011ULL)
        << "rbp should be restored to the value originally pushed";
    EXPECT_EQ(r.state.rip, kReturnSentinel)
        << "RET should find the sentinel — LEAVE restored RSP correctly";
}

// ADC chain — the canonical 128-bit add pattern.
//
//   { lo = 0xFFFFFFFFFFFFFFFF, hi = 0x0000000000000001 }
// + { lo = 0x0000000000000001, hi = 0x0000000000000002 }
// = { lo = 0x0000000000000000, hi = 0x0000000000000004 }   (with CF=1 chained)
//
// Layout: rax = low_a, rbx = high_a, rcx = low_b, rdx = high_b.
// After `add rax, rcx` then `adc rbx, rdx`, expect rax=0, rbx=4.
TEST_F(CpuRuntimeTest, Adc_128BitAddChain_CarriesCorrectly) {
    const u8 program[] = {
        // mov rax, 0xFFFFFFFFFFFFFFFF
        0x48,
        0xb8,
        0xff,
        0xff,
        0xff,
        0xff,
        0xff,
        0xff,
        0xff,
        0xff,
        // mov rbx, 1
        0x48,
        0xc7,
        0xc3,
        0x01,
        0x00,
        0x00,
        0x00,
        // mov rcx, 1
        0x48,
        0xc7,
        0xc1,
        0x01,
        0x00,
        0x00,
        0x00,
        // mov rdx, 2
        0x48,
        0xc7,
        0xc2,
        0x02,
        0x00,
        0x00,
        0x00,
        // add rax, rcx        (sets CF=1)
        0x48,
        0x01,
        0xc8,
        // adc rbx, rdx        (reads CF, adds it)
        0x48,
        0x11,
        0xd3,
        0xc3,
    };
    const auto r = RunProgram(program, sizeof(program), mem);
    EXPECT_EQ(r.state.gpr[0], 0u) << "low half wraps to 0";
    EXPECT_EQ(r.state.gpr[3], 4u) << "high half: 1 + 2 + CF(1) = 4";
}

// SBB chain — the inverse, multi-precision subtraction.
//
//   { lo = 0x0000000000000000, hi = 0x0000000000000004 }
// - { lo = 0x0000000000000001, hi = 0x0000000000000002 }
// = { lo = 0xFFFFFFFFFFFFFFFF, hi = 0x0000000000000001 }   (CF=1 borrowed)
TEST_F(CpuRuntimeTest, Sbb_128BitSubChain_BorrowsCorrectly) {
    const u8 program[] = {
        0x48, 0xc7, 0xc0, 0x00, 0x00, 0x00, 0x00, // mov rax, 0
        0x48, 0xc7, 0xc3, 0x04, 0x00, 0x00, 0x00, // mov rbx, 4
        0x48, 0xc7, 0xc1, 0x01, 0x00, 0x00, 0x00, // mov rcx, 1
        0x48, 0xc7, 0xc2, 0x02, 0x00, 0x00, 0x00, // mov rdx, 2
        0x48, 0x29, 0xc8,                         // sub rax, rcx  (sets CF=1)
        0x48, 0x19, 0xd3,                         // sbb rbx, rdx  (reads CF)
        0xc3,
    };
    const auto r = RunProgram(program, sizeof(program), mem);
    EXPECT_EQ(r.state.gpr[0], 0xFFFFFFFFFFFFFFFFULL) << "low half borrows to all-Fs";
    EXPECT_EQ(r.state.gpr[3], 1u) << "high half: 4 - 2 - CF(1) = 1";
}

// ADC reads CF=0 case — verify chain starts clean from a 0-flag state.
TEST_F(CpuRuntimeTest, Adc_WithCarryClear_BehavesLikeAdd) {
    const u8 program[] = {
        0x48, 0xc7, 0xc0, 0x05, 0x00, 0x00, 0x00, // mov rax, 5
        0x48, 0xc7, 0xc3, 0x03, 0x00, 0x00, 0x00, // mov rbx, 3
        0xf8,                                     // clc → CF=0
        0x48, 0x11, 0xd8,                         // adc rax, rbx  (5 + 3 + 0 = 8)
        0xc3,
    };
    const auto r = RunProgram(program, sizeof(program), mem);
    EXPECT_EQ(r.state.gpr[0], 8u);
}

// 8-bit ADD — flags set per byte semantics. Adding 0xFF + 0x01 in
// 8-bit overflows: result = 0x00, CF = 1, ZF = 1.
TEST_F(CpuRuntimeTest, Add8_OverflowsByteAndSetsCarry) {
    const u8 program[] = {
        // mov al, 0xFF      (B0 FF — 2-byte immediate-to-low-byte form)
        0xb0,
        0xff,
        // mov bl, 0x01
        0xb3,
        0x01,
        // add al, bl        (00 d8)
        0x00,
        0xd8,
        0xc3,
    };
    const auto r = RunProgram(program, sizeof(program), mem);
    EXPECT_EQ(r.state.gpr[0] & 0xFFu, 0u) << "0xFF + 0x01 wraps to 0";
    EXPECT_EQ(r.state.rflags & 0x1ULL, 0x1ULL) << "CF set by byte-overflow";
    EXPECT_EQ(r.state.rflags & 0x40ULL, 0x40ULL) << "ZF set (result is 0)";
}

// 8-bit ADD — preserves upper 56 bits of dst's slot. Set rax to a
// sentinel, then do an 8-bit add to al, and check the upper bits
// of rax are intact.
TEST_F(CpuRuntimeTest, Add8_PreservesUpperBits) {
    const u8 program[] = {
        // mov rax, 0xDEADBEEFCAFE0010
        0x48,
        0xb8,
        0x10,
        0x00,
        0xfe,
        0xca,
        0xef,
        0xbe,
        0xad,
        0xde,
        // add al, 5
        0x04,
        0x05,
        0xc3,
    };
    const auto r = RunProgram(program, sizeof(program), mem);
    EXPECT_EQ(r.state.gpr[0], 0xDEADBEEFCAFE0015ULL);
}

// 8-bit CMP — sets flags but doesn't write back. CMP al, bl with
// al=5, bl=5 should set ZF.
TEST_F(CpuRuntimeTest, Cmp8_EqualValues_SetsZeroFlag) {
    const u8 program[] = {
        // mov al, 5; mov bl, 5
        0xb0,
        0x05,
        0xb3,
        0x05,
        // cmp al, bl     (38 d8)
        0x38,
        0xd8,
        0xc3,
    };
    const auto r = RunProgram(program, sizeof(program), mem);
    EXPECT_EQ(r.state.gpr[0] & 0xFFu, 5u) << "CMP must not write al";
    EXPECT_EQ(r.state.rflags & 0x40ULL, 0x40ULL) << "ZF set (al == bl)";
}

// 16-bit SUB — width-correct flag semantics. 0x8000 (highest s16
// negative) - 1 = 0x7FFF, with OF set because we crossed the
// signed boundary.
TEST_F(CpuRuntimeTest, Sub16_SignedOverflow_SetsOverflowFlag) {
    const u8 program[] = {
        // mov ax, 0x8000     66 b8 00 80
        0x66,
        0xb8,
        0x00,
        0x80,
        // mov bx, 1          66 bb 01 00
        0x66,
        0xbb,
        0x01,
        0x00,
        // sub ax, bx         66 29 d8
        0x66,
        0x29,
        0xd8,
        0xc3,
    };
    const auto r = RunProgram(program, sizeof(program), mem);
    EXPECT_EQ(r.state.gpr[0] & 0xFFFFu, 0x7FFFu);
    // OF is bit 11 of rflags.
    EXPECT_EQ(r.state.rflags & 0x800ULL, 0x800ULL) << "Crossing the s16 sign boundary sets OF";
}

// 16-bit ADD — preserves upper 48 bits.
TEST_F(CpuRuntimeTest, Add16_PreservesUpper48Bits) {
    const u8 program[] = {
        // mov rax, 0xDEADBEEFCAFE0100
        0x48,
        0xb8,
        0x00,
        0x01,
        0xfe,
        0xca,
        0xef,
        0xbe,
        0xad,
        0xde,
        // add ax, 5          66 83 c0 05
        0x66,
        0x83,
        0xc0,
        0x05,
        0xc3,
    };
    const auto r = RunProgram(program, sizeof(program), mem);
    EXPECT_EQ(r.state.gpr[0], 0xDEADBEEFCAFE0105ULL);
}

// 8-bit TEST — bit-wise AND, sets flags only, doesn't write.
//
// Real PS4 binary triggered this with `test r/m8, r/m8`. Set up
// al = 0x10, bl = 0x01: test would AND them = 0, ZF=1.
TEST_F(CpuRuntimeTest, Test8_NoOverlap_SetsZeroFlag) {
    const u8 program[] = {
        0xb0, 0x10, // mov al, 0x10
        0xb3, 0x01, // mov bl, 0x01
        0x84, 0xd8, // test al, bl  (operand_width=8)
        0xc3,
    };
    const auto r = RunProgram(program, sizeof(program), mem);
    EXPECT_EQ(r.state.rflags & 0x40ULL, 0x40ULL) << "0x10 & 0x01 = 0 → ZF set";
    EXPECT_EQ(r.state.gpr[0] & 0xFFu, 0x10u) << "TEST must not write al";
}

// 8-bit TEST — overlapping bits clears ZF.
TEST_F(CpuRuntimeTest, Test8_Overlap_ClearsZeroFlag) {
    const u8 program[] = {
        0xb0, 0x11, // mov al, 0x11 (low bit + bit 4)
        0xb3, 0x01, // mov bl, 0x01 (low bit)
        0x84, 0xd8, // test al, bl  → result 0x01, ZF=0
        0xc3,
    };
    const auto r = RunProgram(program, sizeof(program), mem);
    EXPECT_EQ(r.state.rflags & 0x40ULL, 0u) << "0x11 & 0x01 = 0x01 → ZF clear";
}

// 16-bit TEST.
TEST_F(CpuRuntimeTest, Test16_SetsZeroFlagOnDisjointBits) {
    const u8 program[] = {
        // mov ax, 0x00F0
        0x66,
        0xb8,
        0xf0,
        0x00,
        // mov bx, 0x0F00
        0x66,
        0xbb,
        0x00,
        0x0f,
        // test ax, bx        (66 85 d8)
        0x66,
        0x85,
        0xd8,
        0xc3,
    };
    const auto r = RunProgram(program, sizeof(program), mem);
    EXPECT_EQ(r.state.rflags & 0x40ULL, 0x40ULL) << "no overlapping bits → ZF set";
}

// =============================================================================
// Tier-5: 8/16-bit AND, OR, XOR, NOT, NEG.
//
// Verifies the narrow-width bitwise ops use the round-trip-flags
// pattern correctly and that narrow stores preserve the upper bits
// of the underlying 64-bit GPR slot.
// =============================================================================

// 8-bit AND. Pre-loads rax with a value whose upper 56 bits are
// non-zero; the AND should affect only the low byte and leave the
// upper bits intact.
TEST_F(CpuRuntimeTest, And8_PreservesUpperBitsAndComputesIntersection) {
    const u8 program[] = {
        // mov rax, 0x11223344'556677F0
        0x48,
        0xb8,
        0xf0,
        0x77,
        0x66,
        0x55,
        0x44,
        0x33,
        0x22,
        0x11,
        // mov bl, 0x0F
        0xb3,
        0x0f,
        // and al, bl                  (20 d8)
        0x20,
        0xd8,
        0xc3,
    };
    const auto r = RunProgram(program, sizeof(program), mem);
    // Low byte: 0xF0 & 0x0F = 0x00. Upper 56 bits unchanged.
    EXPECT_EQ(r.state.gpr[0], 0x1122334455667700ULL)
        << "AL written, upper 56 bits of RAX must be preserved";
    EXPECT_EQ(r.state.rflags & 0x40ULL, 0x40ULL) << "zero result → ZF set";
}

// 8-bit OR. Similar to AND but combining bits.
TEST_F(CpuRuntimeTest, Or8_PreservesUpperBitsAndComputesUnion) {
    const u8 program[] = {
        // mov rax, 0xAAAAAAAA'AAAAAAF0
        0x48,
        0xb8,
        0xf0,
        0xaa,
        0xaa,
        0xaa,
        0xaa,
        0xaa,
        0xaa,
        0xaa,
        // mov bl, 0x0F
        0xb3,
        0x0f,
        // or al, bl                   (08 d8)
        0x08,
        0xd8,
        0xc3,
    };
    const auto r = RunProgram(program, sizeof(program), mem);
    EXPECT_EQ(r.state.gpr[0], 0xAAAAAAAAAAAAAAFFULL)
        << "Low byte = 0xF0 | 0x0F = 0xFF; upper 56 unchanged";
    EXPECT_EQ(r.state.rflags & 0x40ULL, 0x00ULL) << "non-zero result → ZF clear";
}

// 8-bit XOR. Useful idiom: `xor al, al` to zero the low byte
// without touching upper bits (rare but compilers sometimes emit it).
TEST_F(CpuRuntimeTest, Xor8_PreservesUpperBitsAndZerosLowByte) {
    const u8 program[] = {
        // mov rax, 0xDEADBEEF'CAFEBABE
        0x48,
        0xb8,
        0xbe,
        0xba,
        0xfe,
        0xca,
        0xef,
        0xbe,
        0xad,
        0xde,
        // xor al, al                  (30 c0)
        0x30,
        0xc0,
        0xc3,
    };
    const auto r = RunProgram(program, sizeof(program), mem);
    EXPECT_EQ(r.state.gpr[0], 0xDEADBEEFCAFEBA00ULL)
        << "AL cleared, upper 56 bits of RAX preserved";
    EXPECT_EQ(r.state.rflags & 0x40ULL, 0x40ULL) << "ZF set when result is zero";
}

// 8-bit AND with immediate. The reg,imm form is encoded differently
// from reg,reg; both should land in the narrow-arith path.
TEST_F(CpuRuntimeTest, And8_RegImm_PreservesUpperBits) {
    const u8 program[] = {
        // mov rax, 0x12345678'9ABCDEFF
        0x48,
        0xb8,
        0xff,
        0xde,
        0xbc,
        0x9a,
        0x78,
        0x56,
        0x34,
        0x12,
        // and al, 0x0F                (24 0F)
        0x24,
        0x0f,
        0xc3,
    };
    const auto r = RunProgram(program, sizeof(program), mem);
    EXPECT_EQ(r.state.gpr[0], 0x123456789ABCDE0FULL)
        << "AL = 0xFF & 0x0F = 0x0F; upper 56 bits preserved";
}

// 16-bit AND. Same pattern with word writes.
TEST_F(CpuRuntimeTest, And16_PreservesUpper48BitsAndComputesIntersection) {
    const u8 program[] = {
        // mov rax, 0xCAFEBABE'12345678
        0x48,
        0xb8,
        0x78,
        0x56,
        0x34,
        0x12,
        0xbe,
        0xba,
        0xfe,
        0xca,
        // mov bx, 0xFF00
        0x66,
        0xbb,
        0x00,
        0xff,
        // and ax, bx                  (66 21 d8)
        0x66,
        0x21,
        0xd8,
        0xc3,
    };
    const auto r = RunProgram(program, sizeof(program), mem);
    // Low word: 0x5678 & 0xFF00 = 0x5600. Upper 48 unchanged.
    EXPECT_EQ(r.state.gpr[0], 0xCAFEBABE12345600ULL)
        << "AX written via word store, upper 48 bits of RAX preserved";
}

// 16-bit XOR.
TEST_F(CpuRuntimeTest, Xor16_PreservesUpper48BitsAndXorsLowWord) {
    const u8 program[] = {
        // mov rax, 0x11223344'55667788
        0x48,
        0xb8,
        0x88,
        0x77,
        0x66,
        0x55,
        0x44,
        0x33,
        0x22,
        0x11,
        // mov bx, 0xFFFF
        0x66,
        0xbb,
        0xff,
        0xff,
        // xor ax, bx                  (66 31 d8)
        0x66,
        0x31,
        0xd8,
        0xc3,
    };
    const auto r = RunProgram(program, sizeof(program), mem);
    // Low word: 0x7788 XOR 0xFFFF = 0x8877. Upper 48 unchanged.
    EXPECT_EQ(r.state.gpr[0], 0x1122334455668877ULL);
}

// 8-bit NOT.
TEST_F(CpuRuntimeTest, Not8_PreservesUpperBits) {
    const u8 program[] = {
        // mov rax, 0xFEDCBA98'765432FF
        0x48,
        0xb8,
        0xff,
        0x32,
        0x54,
        0x76,
        0x98,
        0xba,
        0xdc,
        0xfe,
        // not al                      (f6 d0)
        0xf6,
        0xd0,
        0xc3,
    };
    const auto r = RunProgram(program, sizeof(program), mem);
    // AL = ~0xFF = 0x00. Upper 56 unchanged.
    EXPECT_EQ(r.state.gpr[0], 0xFEDCBA9876543200ULL);
}

// 16-bit NOT.
TEST_F(CpuRuntimeTest, Not16_PreservesUpper48Bits) {
    const u8 program[] = {
        // mov rax, 0xFEDCBA98'00000000
        0x48,
        0xb8,
        0x00,
        0x00,
        0x00,
        0x00,
        0x98,
        0xba,
        0xdc,
        0xfe,
        // not ax                      (66 f7 d0)
        0x66,
        0xf7,
        0xd0,
        0xc3,
    };
    const auto r = RunProgram(program, sizeof(program), mem);
    // AX = ~0x0000 = 0xFFFF. Upper 48 unchanged.
    EXPECT_EQ(r.state.gpr[0], 0xFEDCBA980000FFFFULL);
}

// 8-bit NEG. Result is two's complement; flags follow SUB semantics.
TEST_F(CpuRuntimeTest, Neg8_NegatesByteAndPreservesUpper) {
    const u8 program[] = {
        // mov rax, 0x12345678'9ABCDE05
        0x48,
        0xb8,
        0x05,
        0xde,
        0xbc,
        0x9a,
        0x78,
        0x56,
        0x34,
        0x12,
        // neg al                      (f6 d8)
        0xf6,
        0xd8,
        0xc3,
    };
    const auto r = RunProgram(program, sizeof(program), mem);
    // AL = -5 = 0xFB. Upper 56 unchanged.
    EXPECT_EQ(r.state.gpr[0], 0x123456789ABCDEFBULL);
    // For non-zero input, CF is set (NEG of non-zero borrows from
    // imaginary higher bit).
    EXPECT_EQ(r.state.rflags & 0x01ULL, 0x01ULL) << "NEG of non-zero must set CF";
}

// 8-bit NEG of zero — CF should be CLEAR (special case).
TEST_F(CpuRuntimeTest, Neg8_OfZero_ClearsCarry) {
    const u8 program[] = {
        // mov rax, 0x12345678'9ABCDE00
        0x48,
        0xb8,
        0x00,
        0xde,
        0xbc,
        0x9a,
        0x78,
        0x56,
        0x34,
        0x12,
        // neg al
        0xf6,
        0xd8,
        0xc3,
    };
    const auto r = RunProgram(program, sizeof(program), mem);
    EXPECT_EQ(r.state.gpr[0], 0x123456789ABCDE00ULL);
    EXPECT_EQ(r.state.rflags & 0x01ULL, 0x00ULL) << "NEG of zero must clear CF (no borrow)";
    EXPECT_EQ(r.state.rflags & 0x40ULL, 0x40ULL) << "result is zero → ZF set";
}

// 32-bit NEG. Writes zero-extend the upper 32 bits; flags are
// computed for the 32-bit value (SF on bit 31, not bit 63).
TEST_F(CpuRuntimeTest, Neg32_ZeroExtendsAndComputesFlags) {
    const u8 program[] = {
        // mov rax, 0x12345678'00000005 (upper bits must be cleared)
        0x48,
        0xb8,
        0x05,
        0x00,
        0x00,
        0x00,
        0x78,
        0x56,
        0x34,
        0x12,
        // neg eax                       (f7 d8)
        0xf7,
        0xd8,
        0xc3,
    };
    const auto r = RunProgram(program, sizeof(program), mem);
    // EAX = -5 (mod 2^32) = 0xFFFFFFFB; upper 32 zero-extended.
    EXPECT_EQ(r.state.gpr[0], 0x00000000FFFFFFFBULL) << "32-bit write zero-extends upper 32 bits";
    EXPECT_EQ(r.state.rflags & 0x01ULL, 0x01ULL) << "non-zero input → CF set";
    EXPECT_EQ(r.state.rflags & 0x80ULL, 0x80ULL) << "bit 31 set → SF set";
}

TEST_F(CpuRuntimeTest, Neg32_OfZero_ClearsCarryAndZeroExtends) {
    const u8 program[] = {
        // mov rax, 0xDEADBEEF'00000000 (upper bits will be lost)
        0x48,
        0xb8,
        0x00,
        0x00,
        0x00,
        0x00,
        0xef,
        0xbe,
        0xad,
        0xde,
        // neg eax
        0xf7,
        0xd8,
        0xc3,
    };
    const auto r = RunProgram(program, sizeof(program), mem);
    // EAX = -0 = 0; zero-extended kills the upper 32 bits too.
    EXPECT_EQ(r.state.gpr[0], 0x0000000000000000ULL)
        << "NEG eax of zero: upper 32 of RAX zero-extended away";
    EXPECT_EQ(r.state.rflags & 0x01ULL, 0x00ULL) << "input zero → CF clear";
    EXPECT_EQ(r.state.rflags & 0x40ULL, 0x40ULL) << "result zero → ZF set";
}

// 8-bit AND with memory source. Tests that the narrow-arith path
// correctly handles the [base+disp] addressing mode.
//
// Setup: stack holds a byte 0xF0. Guest does:
//   mov al, 0xFF
//   and al, byte ptr [rsp+0]
// Expected: AL = 0xFF & 0xF0 = 0xF0.
TEST_F(CpuRuntimeTest, And8_RegMem_LoadsViaEffectiveAddress) {
    u8 program[] = {
        // Push a value onto guest stack so [rsp] holds known bytes.
        // mov rax, 0x00000000'000000F0
        0x48,
        0xb8,
        0xf0,
        0x00,
        0x00,
        0x00,
        0x00,
        0x00,
        0x00,
        0x00,
        0x50, // push rax
        // mov rax, 0x000000FF (low byte is target)
        0x48,
        0xb8,
        0xff,
        0x00,
        0x00,
        0x00,
        0x00,
        0x00,
        0x00,
        0x00,
        // and al, byte ptr [rsp]         (22 04 24)
        0x22,
        0x04,
        0x24,
        0x48,
        0x83,
        0xc4,
        0x08, // add rsp, 8 (cleanup)
        0xc3,
    };
    const auto r = RunProgram(program, sizeof(program), mem);
    EXPECT_EQ(r.state.gpr[0] & 0xFFULL, 0xF0ULL)
        << "AL should be 0xFF AND 0xF0 = 0xF0 (loaded from [rsp])";
    EXPECT_EQ(r.state.rflags & 0x40ULL, 0x00ULL) << "non-zero result → ZF clear";
}

// 8-bit OR with memory source.
TEST_F(CpuRuntimeTest, Or8_RegMem_LoadsViaEffectiveAddress) {
    u8 program[] = {
        // mov rax, 0x05  (push as byte source)
        0x48,
        0xb8,
        0x05,
        0x00,
        0x00,
        0x00,
        0x00,
        0x00,
        0x00,
        0x00,
        0x50, // push rax
        // mov rax, 0x80  (target with bit 7 set)
        0x48,
        0xb8,
        0x80,
        0x00,
        0x00,
        0x00,
        0x00,
        0x00,
        0x00,
        0x00,
        // or al, byte ptr [rsp]          (0a 04 24)
        0x0a,
        0x04,
        0x24,
        0x48,
        0x83,
        0xc4,
        0x08,
        0xc3,
    };
    const auto r = RunProgram(program, sizeof(program), mem);
    EXPECT_EQ(r.state.gpr[0] & 0xFFULL, 0x85ULL) << "AL = 0x80 | 0x05 = 0x85";
}

// 16-bit AND with memory source. Verifies word loads via the
// effective-address helper.
TEST_F(CpuRuntimeTest, And16_RegMem_LoadsViaEffectiveAddress) {
    u8 program[] = {
        // Push 0x0F0F (low word) — the high bytes are irrelevant since
        // we'll read only word ptr [rsp].
        0x48,
        0xb8,
        0x0f,
        0x0f,
        0x00,
        0x00,
        0x00,
        0x00,
        0x00,
        0x00,
        0x50,
        // mov rax, 0xFFFF (target word)
        0x48,
        0xb8,
        0xff,
        0xff,
        0x00,
        0x00,
        0x00,
        0x00,
        0x00,
        0x00,
        // and ax, word ptr [rsp]         (66 23 04 24)
        0x66,
        0x23,
        0x04,
        0x24,
        0x48,
        0x83,
        0xc4,
        0x08,
        0xc3,
    };
    const auto r = RunProgram(program, sizeof(program), mem);
    EXPECT_EQ(r.state.gpr[0] & 0xFFFFULL, 0x0F0FULL) << "AX = 0xFFFF AND 0x0F0F = 0x0F0F";
}

// ============================================================================
// 8-bit shifts (SHL/SHR/SAR with imm8 or CL). The narrow-width form
// shares the same round-trip-flags-through-host pattern as the 32/64
// versions, but must preserve bits 63:8 of the parent GPR slot (unlike
// 32-bit shifts, which zero-extend bits 63:32 via the x86-64 register
// write rule). The merge step in EmitShift8 — load al, host shift,
// then mask-and-or back into byte 0 of the slot — is exactly what
// these tests exercise.
// ============================================================================

// SHL al, 2 — verify the host shift result lands in byte 0 of the
// rax slot and the upper 56 bits of the slot are preserved. Encoding
// shape `C0 /4 ib` (no REX needed for AL).
TEST_F(CpuRuntimeTest, Shl8_Imm_PreservesUpperBitsOfParentSlot) {
    const u8 program[] = {
        // mov rax, 0xDEADBEEF00000005 — junk in upper, al = 0x05
        0x48, 0xb8, 0x05, 0x00, 0x00, 0x00, 0xef, 0xbe, 0xad, 0xde,
        0xc0, 0xe0, 0x02,                         // shl al, 2 → al = 0x14
        0xc3,                                     // ret
    };
    const auto r = RunProgram(program, sizeof(program), mem);
    EXPECT_EQ(r.state.gpr[0], 0xDEADBEEF00000014ULL)
        << "SHL al, 2 must shift only byte 0; bits 63:8 must be preserved";
}

// SHR bl, 4 — exercises an "extended low-byte" register (BL = byte 0
// of slot 3 / RBX) to verify the byte-offset accessor handles non-
// AL destinations correctly. This is the encoding the game hit at
// 0x807a73217 (mnemonic=shr, width=8, length=3, ops=reg,imm).
TEST_F(CpuRuntimeTest, Shr8_Imm_OnRbxLowByte) {
    const u8 program[] = {
        // mov rbx, 0xFFFFFFFF000000A0 — bl = 0xA0
        0x48, 0xbb, 0xa0, 0x00, 0x00, 0x00, 0xff, 0xff, 0xff, 0xff,
        0xc0, 0xeb, 0x04,                         // shr bl, 4 → bl = 0x0A
        0xc3,                                     // ret
    };
    const auto r = RunProgram(program, sizeof(program), mem);
    EXPECT_EQ(r.state.gpr[3], 0xFFFFFFFF0000000AULL)
        << "SHR bl, 4 must shift only the low byte and preserve bits 63:8";
}

// SAR al, 2 with a negative byte value — arithmetic shift fills with
// the sign bit. Starts from al = 0xF0 (signed -16), expects al = 0xFC
// (signed -4). Upper bits of the rax slot are zero going in (clean
// mov rax, imm64 → 0x..00F0) and must remain zero on the way out
// (the merge mask preserves them without touching the sign-fill).
TEST_F(CpuRuntimeTest, Sar8_Imm_SignExtendsLowByteOnly) {
    const u8 program[] = {
        // mov rax, 0xF0  (full mov-imm64 form so the upper 56 are zero)
        0x48, 0xb8, 0xf0, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00,
        0xc0, 0xf8, 0x02,                         // sar al, 2 → al = 0xFC
        0xc3,                                     // ret
    };
    const auto r = RunProgram(program, sizeof(program), mem);
    EXPECT_EQ(r.state.gpr[0], 0x00000000000000FCULL)
        << "SAR al, 2 should sign-extend within the byte but NOT into bits 63:8";
}

// SHL al, cl — dynamic shift count via CL. The 8-bit emitter reuses
// the same CL-source path as the wider shifts, this just confirms
// the byte-width path doesn't accidentally read more than CL.
TEST_F(CpuRuntimeTest, Shl8_Cl_DynamicCountOnLowByte) {
    const u8 program[] = {
        // mov rax, 0x01 ; al = 1
        0x48, 0xb8, 0x01, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00,
        // mov rcx, 0x03 ; shift count = 3
        0x48, 0xb9, 0x03, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00,
        0xd2, 0xe0,                               // shl al, cl
        0xc3,                                     // ret
    };
    const auto r = RunProgram(program, sizeof(program), mem);
    EXPECT_EQ(r.state.gpr[0] & 0xFFULL, 0x08ULL) << "1 << 3 = 8";
}

// ============================================================================
// 64-bit reg-mem arithmetic and bitwise (ADD/SUB/AND/OR/XOR r64, [m]).
// All five share the same skeleton: EmitEffectiveAddress puts the
// address in rdx, then the loaded qword overwrites rdx (becoming the
// rhs for the flag helper), rcx/rax hold the lhs, the host op runs,
// and the result is stored back. These tests verify the mem-src path
// for each opcode by writing a known value to the stack and loading
// it via [rsp-8] addressing.
// ============================================================================

TEST_F(CpuRuntimeTest, Add64_RegMem_LoadsAndAdds) {
    const u8 program[] = {
        0x48, 0xc7, 0xc0, 0x00, 0x01, 0x00, 0x00, // mov rax, 0x100
        0x48, 0x89, 0x44, 0x24, 0xf8,             // mov [rsp-8], rax
        0x48, 0xc7, 0xc0, 0x50, 0x00, 0x00, 0x00, // mov rax, 0x50
        0x48, 0x03, 0x44, 0x24, 0xf8,             // add rax, [rsp-8]
        0xc3,                                     // ret
    };
    const auto r = RunProgram(program, sizeof(program), mem);
    EXPECT_EQ(r.state.gpr[0], 0x150ULL) << "0x50 + 0x100 (loaded from memory)";
}

TEST_F(CpuRuntimeTest, Sub64_RegMem_LoadsAndSubs) {
    const u8 program[] = {
        0x48, 0xc7, 0xc0, 0x80, 0x00, 0x00, 0x00, // mov rax, 0x80
        0x48, 0x89, 0x44, 0x24, 0xf8,             // mov [rsp-8], rax
        0x48, 0xc7, 0xc0, 0x00, 0x01, 0x00, 0x00, // mov rax, 0x100
        0x48, 0x2b, 0x44, 0x24, 0xf8,             // sub rax, [rsp-8]
        0xc3,                                     // ret
    };
    const auto r = RunProgram(program, sizeof(program), mem);
    EXPECT_EQ(r.state.gpr[0], 0x80ULL) << "0x100 - 0x80 (loaded from memory)";
}

TEST_F(CpuRuntimeTest, And64_RegMem_LoadsAndAnds) {
    const u8 program[] = {
        0x48, 0xc7, 0xc0, 0x00, 0xff, 0x00, 0x00, // mov rax, 0xFF00
        0x48, 0x89, 0x44, 0x24, 0xf8,             // mov [rsp-8], rax
        0x48, 0xc7, 0xc0, 0xff, 0xff, 0x00, 0x00, // mov rax, 0xFFFF
        0x48, 0x23, 0x44, 0x24, 0xf8,             // and rax, [rsp-8]
        0xc3,                                     // ret
    };
    const auto r = RunProgram(program, sizeof(program), mem);
    EXPECT_EQ(r.state.gpr[0], 0xFF00ULL) << "0xFFFF & 0xFF00 = 0xFF00";
}

TEST_F(CpuRuntimeTest, Or64_RegMem_LoadsAndOrs) {
    const u8 program[] = {
        0x48, 0xc7, 0xc0, 0x0f, 0x00, 0x00, 0x00, // mov rax, 0x0F
        0x48, 0x89, 0x44, 0x24, 0xf8,             // mov [rsp-8], rax
        0x48, 0xc7, 0xc0, 0xf0, 0x00, 0x00, 0x00, // mov rax, 0xF0
        0x48, 0x0b, 0x44, 0x24, 0xf8,             // or rax, [rsp-8]
        0xc3,                                     // ret
    };
    const auto r = RunProgram(program, sizeof(program), mem);
    EXPECT_EQ(r.state.gpr[0], 0xFFULL) << "0xF0 | 0x0F = 0xFF";
}

TEST_F(CpuRuntimeTest, Xor64_RegMem_LoadsAndXors) {
    const u8 program[] = {
        0x48, 0xc7, 0xc0, 0xff, 0x00, 0x00, 0x00, // mov rax, 0xFF
        0x48, 0x89, 0x44, 0x24, 0xf8,             // mov [rsp-8], rax
        0x48, 0xc7, 0xc0, 0x0f, 0x00, 0x00, 0x00, // mov rax, 0x0F
        0x48, 0x33, 0x44, 0x24, 0xf8,             // xor rax, [rsp-8]
        0xc3,                                     // ret
    };
    const auto r = RunProgram(program, sizeof(program), mem);
    EXPECT_EQ(r.state.gpr[0], 0xF0ULL) << "0x0F ^ 0xFF = 0xF0";
}

// ============================================================================
// 32-bit LEA — the zero-extension trap.
//
// x86-64's "32-bit destination write zero-extends bits 63:32" is a
// *register* rule, not a memory rule. EmitLea emits a qword store
// to the guest GPR slot, so we must zero the upper 32 BEFORE the
// store (via `mov edx, edx` — the canonical self-zero idiom).
// This test fails if EmitLea forgets that step: it would store only
// the low 32 via the address calc, leaving the slot's upper bytes
// containing the previous rax value.
// ============================================================================

TEST_F(CpuRuntimeTest, Lea32_ZerosUpper32OfDestination) {
    const u8 program[] = {
        // mov rax, 0xDEADBEEF12345678 — pollute rax with junk in upper
        0x48, 0xb8, 0x78, 0x56, 0x34, 0x12, 0xef, 0xbe, 0xad, 0xde,
        0x48, 0xc7, 0xc1, 0x00, 0x01, 0x00, 0x00, // mov rcx, 0x100
        0x8d, 0x41, 0x40,                         // lea eax, [rcx + 0x40]
        0xc3,                                     // ret
    };
    const auto r = RunProgram(program, sizeof(program), mem);
    EXPECT_EQ(r.state.gpr[0], 0x140ULL)
        << "lea eax, [rcx+0x40] must zero-extend; expected 0x140, not 0xDEADBEEF00000140";
}

// ============================================================================
// ANDN — BMI1 three-operand bitwise NOT-AND. EmitAndn handles both
// 64- and 32-bit forms; the 32-bit variant relies on the host's
// register-write zero-extension to clear bits 63:32 of the result.
// ============================================================================

// ANDN eax, ecx, edx — eax = (~ecx) & edx, with upper 32 zeroed.
// VEX encoding: C4 E2 70 F2 /r. The ModR/M byte C2 selects
// reg=eax, rm=edx; the VEX.vvvv field encodes ecx (inverted).
TEST_F(CpuRuntimeTest, Andn32_RegReg_ComputesAndZeroExtends) {
    const u8 program[] = {
        // mov rax, 0xDEADBEEF12345678 — pollute upper of dst
        0x48, 0xb8, 0x78, 0x56, 0x34, 0x12, 0xef, 0xbe, 0xad, 0xde,
        0x48, 0xc7, 0xc1, 0xf0, 0x00, 0x00, 0x00, // mov rcx, 0xF0  (src1)
        0x48, 0xc7, 0xc2, 0xff, 0x00, 0x00, 0x00, // mov rdx, 0xFF  (src2)
        0xc4, 0xe2, 0x70, 0xf2, 0xc2,             // andn eax, ecx, edx
        0xc3,                                     // ret
    };
    const auto r = RunProgram(program, sizeof(program), mem);
    // (~0xF0) & 0xFF in 32-bit = 0xFFFFFF0F & 0xFF = 0x0F
    EXPECT_EQ(r.state.gpr[0], 0x0FULL)
        << "(~ecx) & edx low 32 = 0x0F, upper 32 must be zero-extended";
}

// ============================================================================
// High-byte register access (AH/BH/CH/DH).
//
// These are the legacy "high byte of low word" registers, which can
// only be encoded WITHOUT a REX prefix. EmitNarrowArith8 supports
// them via the ZydisGpr8ToByteOffset helper, which returns byte 1
// of the parent slot (since the parent qword is laid out little-
// endian, byte 1 holds bits 15:8). The other 8-bit emitters
// (EmitShift8, EmitSetcc) still reject these as no production code
// has been observed using them in those contexts.
// ============================================================================

// TEST ah, imm8 — exercises high-byte READ. Set rax so AL and AH
// have different low bits; if the lifter accidentally loaded AL
// instead of AH the ZF check would flip. Encoding: F6 C4 ib.
TEST_F(CpuRuntimeTest, Test8_HighByteAh_AccessesByte1OfRax) {
    const u8 program[] = {
        // mov rax, 0x100F — AL = 0x0F (bit 0 set), AH = 0x10 (bit 0 clear)
        0x48, 0xc7, 0xc0, 0x0f, 0x10, 0x00, 0x00,
        0xf6, 0xc4, 0x01,                         // test ah, 0x01
        0xc3,                                     // ret
    };
    const auto r = RunProgram(program, sizeof(program), mem);
    // AH=0x10, 0x10 & 0x01 = 0 → ZF must be set.
    // If the lifter mistakenly read AL (0x0F), 0x0F & 0x01 = 1 → ZF clear.
    EXPECT_EQ(r.state.rflags & 0x40ULL, 0x40ULL)
        << "test ah, 0x01 with AH=0x10 must set ZF; if ZF is clear the "
           "lifter read AL by mistake";
    EXPECT_EQ(r.state.gpr[0], 0x100FULL) << "TEST writes only flags, not rax";
}

// AND ch, imm8 — exercises high-byte WRITE. Verifies that only byte 1
// of the rcx slot is modified, with byte 0 (CL) and bytes 2..7
// preserved. Encoding: 80 E5 ib (ModR/M byte selects CH).
TEST_F(CpuRuntimeTest, And8_HighByteCh_WritesOnlyByte1OfRcx) {
    const u8 program[] = {
        // mov rcx, 0xDEADBE00CAFE12AA — diverse bytes across the slot:
        //   byte 0 (CL) = 0xAA
        //   byte 1 (CH) = 0x12
        //   bytes 2..7  = 0xFECA00BEADDE
        0x48, 0xb9, 0xaa, 0x12, 0xfe, 0xca, 0x00, 0xbe, 0xad, 0xde,
        0x80, 0xe5, 0x0f,                         // and ch, 0x0F → ch = 0x02
        0xc3,                                     // ret
    };
    const auto r = RunProgram(program, sizeof(program), mem);
    EXPECT_EQ(r.state.gpr[1], 0xDEADBE00CAFE02AAULL)
        << "and ch, 0x0F must modify only byte 1; CL and bytes 2..7 preserved";
}

// ============================================================================
// DIV — unsigned 64-bit divide with the dividend in RDX:RAX. EmitDiv
// must load BOTH halves of the dividend (not just RAX) and store
// BOTH halves of the result (quotient in RAX, remainder in RDX).
// ============================================================================

// Simple reg-divisor: 100 / 7 = 14 rem 2.
TEST_F(CpuRuntimeTest, Div64_RegDivisor_ComputesQuotientAndRemainder) {
    const u8 program[] = {
        0x48, 0xc7, 0xc2, 0x00, 0x00, 0x00, 0x00, // mov rdx, 0       (hi half)
        0x48, 0xc7, 0xc0, 0x64, 0x00, 0x00, 0x00, // mov rax, 100     (lo half)
        0x48, 0xc7, 0xc1, 0x07, 0x00, 0x00, 0x00, // mov rcx, 7       (divisor)
        0x48, 0xf7, 0xf1,                         // div rcx
        0xc3,                                     // ret
    };
    const auto r = RunProgram(program, sizeof(program), mem);
    EXPECT_EQ(r.state.gpr[0], 14ULL) << "quotient: 100 / 7 = 14";
    EXPECT_EQ(r.state.gpr[2], 2ULL)  << "remainder: 100 % 7 = 2";
}

// 128-bit dividend with non-zero RDX. dividend = 2^64 (RDX=1, RAX=0),
// divisor = 2^32. quotient = 2^32, remainder = 0. If EmitDiv mistakenly
// zero'd RDX before the host op (or loaded it from the wrong slot),
// this test would either fault or compute the wrong quotient.
TEST_F(CpuRuntimeTest, Div64_RegDivisor_UsesFullRdxRaxDividend) {
    const u8 program[] = {
        0x48, 0xc7, 0xc2, 0x01, 0x00, 0x00, 0x00, // mov rdx, 1       (hi = 1)
        0x48, 0xc7, 0xc0, 0x00, 0x00, 0x00, 0x00, // mov rax, 0       (lo = 0)
        // mov rcx, 0x100000000 — needs full imm64 (won't fit in imm32)
        0x48, 0xb9, 0x00, 0x00, 0x00, 0x00, 0x01, 0x00, 0x00, 0x00,
        0x48, 0xf7, 0xf1,                         // div rcx
        0xc3,                                     // ret
    };
    const auto r = RunProgram(program, sizeof(program), mem);
    // (1 << 64) / (1 << 32) = (1 << 32)
    EXPECT_EQ(r.state.gpr[0], 0x100000000ULL) << "quotient = 2^32";
    EXPECT_EQ(r.state.gpr[2], 0ULL)            << "remainder = 0";
}

// Memory divisor — the exact shape the game hit at 0x8079fd328.
// Validates that EmitEffectiveAddress, the divisor load, and the
// loads of guest RAX/RDX are sequenced correctly (in particular,
// that the address in rdx is captured into the divisor BEFORE we
// overwrite rdx with the dividend's high half).
TEST_F(CpuRuntimeTest, Div64_MemDivisor_LoadsViaEffectiveAddress) {
    const u8 program[] = {
        0x48, 0xc7, 0xc2, 0x00, 0x00, 0x00, 0x00, // mov rdx, 0
        0x48, 0xc7, 0xc0, 0xe8, 0x03, 0x00, 0x00, // mov rax, 1000
        0x48, 0xc7, 0xc1, 0x0d, 0x00, 0x00, 0x00, // mov rcx, 13
        0x48, 0x89, 0x4c, 0x24, 0xf8,             // mov [rsp-8], rcx (store divisor)
        0x48, 0xf7, 0x74, 0x24, 0xf8,             // div qword [rsp-8]
        0xc3,                                     // ret
    };
    const auto r = RunProgram(program, sizeof(program), mem);
    // 1000 / 13 = 76 remainder 12 (76*13=988, 1000-988=12)
    EXPECT_EQ(r.state.gpr[0], 76ULL) << "quotient: 1000 / 13 = 76";
    EXPECT_EQ(r.state.gpr[2], 12ULL) << "remainder: 1000 % 13 = 12";
}

// ============================================================================
// 32-bit BEXTR and CMOV — narrow forms of existing emitters. The BEXTR
// change is trivial (host VEX.W=0 form). The CMOV change has a subtle
// asymmetry worth a regression test: on x86-64, the rule "32-bit reg
// writes zero-extend bits 63:32" only fires when the destination is
// actually written. CMOVcc with a FALSE condition writes nothing, so
// bits 63:32 of the parent slot must remain undisturbed. This differs
// from every other 32-bit emitter we have.
// ============================================================================

// BEXTR eax, ecx, edx — 32-bit form. VEX byte 3 = 0x68 (W=0 direct,
// vs ANDN/BEXTR-64 where W is the other polarity).
TEST_F(CpuRuntimeTest, Bextr32_RegRegReg_ExtractsBitsAndZeroExtends) {
    const u8 program[] = {
        // mov rax, 0xDEADBEEF12345678 — pre-pollute upper of dst
        0x48, 0xb8, 0x78, 0x56, 0x34, 0x12, 0xef, 0xbe, 0xad, 0xde,
        // mov rcx, 0x00000000_FFFF0000 — source value
        0x48, 0xb9, 0x00, 0x00, 0xff, 0xff, 0x00, 0x00, 0x00, 0x00,
        // mov rdx, 0x0810 — control: start=16 (bits 7:0), len=8 (bits 15:8)
        0x48, 0xc7, 0xc2, 0x10, 0x08, 0x00, 0x00,
        // bextr eax, ecx, edx
        0xc4, 0xe2, 0x68, 0xf7, 0xc1,
        0xc3,                                     // ret
    };
    const auto r = RunProgram(program, sizeof(program), mem);
    // Extracting 8 bits starting at bit 16 from 0xFFFF0000 yields 0xFF.
    // Upper 32 of rax must be zero-extended away (the host eax-write rule).
    EXPECT_EQ(r.state.gpr[0], 0xFFULL)
        << "BEXTR 32-bit: extract 8 bits at offset 16 from 0xFFFF0000 = 0xFF; "
           "upper 32 of rax must be zeroed";
}

// CMOVZ eax, ecx with the condition TRUE: dst gets src zero-extended,
// upper 32 of dst slot is wiped. Sets up rflags so ZF=1, then runs
// the cmov; expects rax = src low 32, upper zeroed.
TEST_F(CpuRuntimeTest, Cmov32_ConditionTrue_ZeroExtendsSrc) {
    const u8 program[] = {
        // mov rax, 0xDEADBEEFCAFEBABE — junk dst
        0x48, 0xb8, 0xbe, 0xba, 0xfe, 0xca, 0xef, 0xbe, 0xad, 0xde,
        // mov rcx, 0xFFFFFFFF12345678 — src with junk upper, low = 0x12345678
        0x48, 0xb9, 0x78, 0x56, 0x34, 0x12, 0xff, 0xff, 0xff, 0xff,
        // xor edx, edx → ZF=1, edx=0
        0x31, 0xd2,
        // cmovz eax, ecx — cond TRUE (ZF=1) → eax = ecx low 32, upper zeroed
        0x0f, 0x44, 0xc1,
        0xc3,                                     // ret
    };
    const auto r = RunProgram(program, sizeof(program), mem);
    EXPECT_EQ(r.state.gpr[0], 0x12345678ULL)
        << "CMOVZ 32-bit cond TRUE must zero-extend src; rax = 0x12345678";
}

// CMOVZ eax, ecx with the condition FALSE: dst is UNCHANGED — including
// bits 63:32 of the slot. This is the regression catcher. If the lifter
// blindly applied the "32-bit op zero-extends" rule, the upper junk
// (0xDEADBEEF) would be wiped here, breaking guest semantics.
TEST_F(CpuRuntimeTest, Cmov32_ConditionFalse_ZeroExtendsDst) {
    const u8 program[] = {
        // mov rax, 0xDEADBEEFCAFEBABE — junk dst (upper 32 = 0xDEADBEEF)
        0x48, 0xb8, 0xbe, 0xba, 0xfe, 0xca, 0xef, 0xbe, 0xad, 0xde,
        // mov rcx, 0x00000000_12345678
        0x48, 0xb9, 0x78, 0x56, 0x34, 0x12, 0x00, 0x00, 0x00, 0x00,
        // mov rdx, 1; test rdx, rdx → ZF=0
        0x48, 0xc7, 0xc2, 0x01, 0x00, 0x00, 0x00,
        0x48, 0x85, 0xd2,
        // cmovz eax, ecx — cond FALSE (ZF=0); the move is NOT performed,
        // but a 32-bit CMOV still issues a 32-bit write to the destination
        // register, which zero-extends bits 63:32 per x86-64.
        0x0f, 0x44, 0xc1,
        0xc3,                                     // ret
    };
    const auto r = RunProgram(program, sizeof(program), mem);
    EXPECT_EQ(r.state.gpr[0], 0x00000000CAFEBABEULL)
        << "CMOVZ 32-bit cond FALSE keeps the low 32 bits but ZERO-EXTENDS "
           "bits 63:32 — verified against real hardware (an untaken 32-bit "
           "CMOV still performs a 32-bit register write)";
}

// ============================================================================
// CPUID — spoofed to report an AMD Jaguar (PS4 APU). Tests verify the
// canned response for each handled leaf, plus a regression test that an
// unknown leaf returns zeros and that the subleaf input on leaf 7 is
// honored (sub != 0 must yield zeros, not the sub-0 response).
// ============================================================================

// Leaf 0 — max standard leaf + "AuthenticAMD" vendor string in
// EBX:EDX:ECX. Also asserts upper 32 of every result slot is zeroed
// (the qword storeback after 32-bit writes does this implicitly).
// Pre-pollutes RBX and RDX so a missing zero-extend would be visible.
TEST_F(CpuRuntimeTest, Cpuid_Leaf0_ReportsAuthenticAMDVendor) {
    const u8 program[] = {
        // mov rbx, 0xDEADBEEFDEADBEEF — pre-pollute future RBX slot
        0x48, 0xbb, 0xef, 0xbe, 0xad, 0xde, 0xef, 0xbe, 0xad, 0xde,
        // mov rdx, 0xCAFEBABECAFEBABE — pre-pollute future RDX slot
        0x48, 0xba, 0xbe, 0xba, 0xfe, 0xca, 0xbe, 0xba, 0xfe, 0xca,
        // mov rax, 0
        0x48, 0xc7, 0xc0, 0x00, 0x00, 0x00, 0x00,
        // mov rcx, 0
        0x48, 0xc7, 0xc1, 0x00, 0x00, 0x00, 0x00,
        0x0f, 0xa2,                               // cpuid
        0xc3,                                     // ret
    };
    const auto r = RunProgram(program, sizeof(program), mem);
    EXPECT_EQ(r.state.gpr[0], 7ULL) << "max standard leaf = 7";
    EXPECT_EQ(r.state.gpr[3], 0x68747541ULL) << "EBX = 'Auth'";
    EXPECT_EQ(r.state.gpr[1], 0x444D4163ULL) << "ECX = 'cAMD'";
    EXPECT_EQ(r.state.gpr[2], 0x69746E65ULL) << "EDX = 'enti'";

    // Reconstruct the vendor string from the bytes to make the
    // intent of the EBX/EDX/ECX magic numbers easy to verify.
    char vendor[13] = {0};
    const u32 ebx = static_cast<u32>(r.state.gpr[3]);
    const u32 edx = static_cast<u32>(r.state.gpr[2]);
    const u32 ecx = static_cast<u32>(r.state.gpr[1]);
    std::memcpy(vendor + 0, &ebx, 4);
    std::memcpy(vendor + 4, &edx, 4);
    std::memcpy(vendor + 8, &ecx, 4);
    EXPECT_STREQ(vendor, "AuthenticAMD");
}

// Leaf 1 — signature and feature flags. Verifies the family/model/
// stepping reports Jaguar (family 0x16) and that the advertised
// features include AVX + SSE4.x + BMI1 carriers but NOT FMA, AVX2,
// POPCNT (which would be present on most modern hosts via pass-through).
TEST_F(CpuRuntimeTest, Cpuid_Leaf1_ReportsJaguarSignatureAndFeatures) {
    const u8 program[] = {
        0x48, 0xc7, 0xc0, 0x01, 0x00, 0x00, 0x00,  // mov rax, 1
        0x48, 0x31, 0xc9,                          // xor rcx, rcx
        0x0f, 0xa2,                                // cpuid
        0xc3,
    };
    const auto r = RunProgram(program, sizeof(program), mem);
    const u32 sig = static_cast<u32>(r.state.gpr[0]);
    const u32 ecx = static_cast<u32>(r.state.gpr[1]);
    const u32 edx = static_cast<u32>(r.state.gpr[2]);

    // Signature: ExtFamily=7, BaseFamily=0xF → effective family 0x16.
    EXPECT_EQ(sig, 0x00700F01u) << "Jaguar signature (fam 0x16, mod 0, step 1)";
    const u32 family = ((sig >> 8) & 0xF) + ((sig >> 20) & 0xFF);
    EXPECT_EQ(family, 0x16u) << "computed family from signature";

    // Advertised features.
    EXPECT_TRUE(ecx & (1u << 0))  << "SSE3 advertised";
    EXPECT_TRUE(ecx & (1u << 19)) << "SSE4.1 advertised";
    EXPECT_TRUE(ecx & (1u << 20)) << "SSE4.2 advertised";
    EXPECT_TRUE(ecx & (1u << 23)) << "POPCNT advertised";
    EXPECT_TRUE(ecx & (1u << 28)) << "AVX advertised";

    // Features we deliberately do NOT advertise (Jaguar lacks them
    // and/or the JIT lacks emitters):
    EXPECT_FALSE(ecx & (1u << 12)) << "FMA must not be advertised";
    EXPECT_FALSE(ecx & (1u << 30)) << "RDRAND must not be advertised";

    // EDX baseline.
    EXPECT_TRUE(edx & (1u << 0))  << "FPU";
    EXPECT_TRUE(edx & (1u << 25)) << "SSE";
    EXPECT_TRUE(edx & (1u << 26)) << "SSE2";
}

// Leaf 7 subleaf 0 — extended features. We advertise BMI1 only;
// AVX2 and BMI2 must be absent (Jaguar lacks them).
TEST_F(CpuRuntimeTest, Cpuid_Leaf7_Sub0_BmiAdvertisedButNotAvx2) {
    const u8 program[] = {
        0x48, 0xc7, 0xc0, 0x07, 0x00, 0x00, 0x00,  // mov rax, 7
        0x48, 0x31, 0xc9,                          // xor rcx, rcx (subleaf 0)
        0x0f, 0xa2,                                // cpuid
        0xc3,
    };
    const auto r = RunProgram(program, sizeof(program), mem);
    const u32 ebx = static_cast<u32>(r.state.gpr[3]);
    EXPECT_TRUE(ebx  & (1u << 3))  << "BMI1 advertised";
    EXPECT_FALSE(ebx & (1u << 5))  << "AVX2 must not be advertised";
    EXPECT_FALSE(ebx & (1u << 8))  << "BMI2 must not be advertised";
    EXPECT_FALSE(ebx & (1u << 16)) << "AVX-512 must not be advertised";
}

// Leaf 7 with non-zero subleaf must return all zeros — the emitter
// gates the subleaf-0 response on `ecx == 0` and falls through to
// the zero-default storeback otherwise. Regression catcher: a missing
// gate would expose the subleaf-0 response for every subleaf.
TEST_F(CpuRuntimeTest, Cpuid_Leaf7_NonzeroSubleafReturnsZero) {
    const u8 program[] = {
        0x48, 0xc7, 0xc0, 0x07, 0x00, 0x00, 0x00,  // mov rax, 7
        0x48, 0xc7, 0xc1, 0x01, 0x00, 0x00, 0x00,  // mov rcx, 1 (sub 1)
        0x0f, 0xa2,
        0xc3,
    };
    const auto r = RunProgram(program, sizeof(program), mem);
    EXPECT_EQ(r.state.gpr[0], 0ULL);
    EXPECT_EQ(r.state.gpr[3], 0ULL) << "Leaf 7 sub != 0 must not echo BMI1";
    EXPECT_EQ(r.state.gpr[1], 0ULL);
    EXPECT_EQ(r.state.gpr[2], 0ULL);
}

// Brand string leaves concatenated must reconstruct
// "AMD Custom Jaguar 8-Core APU" + trailing space padding + NUL.
// Calls CPUID three times back-to-back and stashes each result into
// a separate buffer region via memory ops, then we verify the full
// 48-byte payload at the end. Since our test harness only runs one
// CPUID conveniently per program, instead drive each leaf in its own
// test and stitch the 12-dword result here in C++.
TEST_F(CpuRuntimeTest, Cpuid_BrandLeaves_SpellOutJaguarString) {
    auto runLeaf = [&](u32 leaf) {
        std::array<u32, 4> out{};
        u8 program[] = {
            0x48, 0xc7, 0xc0, 0,0,0,0,      // mov rax, imm32  (filled below)
            0x48, 0x31, 0xc9,               // xor rcx, rcx
            0x0f, 0xa2,                     // cpuid
            0xc3,                           // ret
        };
        std::memcpy(program + 3, &leaf, 4);
        const auto r = RunProgram(program, sizeof(program), mem);
        out[0] = static_cast<u32>(r.state.gpr[0]);  // EAX
        out[1] = static_cast<u32>(r.state.gpr[3]);  // EBX
        out[2] = static_cast<u32>(r.state.gpr[1]);  // ECX
        out[3] = static_cast<u32>(r.state.gpr[2]);  // EDX
        return out;
    };

    char brand[49] = {0};
    const auto b2 = runLeaf(0x80000002);
    const auto b3 = runLeaf(0x80000003);
    const auto b4 = runLeaf(0x80000004);
    for (int i = 0; i < 4; ++i) std::memcpy(brand +  0 + i*4, &b2[i], 4);
    for (int i = 0; i < 4; ++i) std::memcpy(brand + 16 + i*4, &b3[i], 4);
    for (int i = 0; i < 4; ++i) std::memcpy(brand + 32 + i*4, &b4[i], 4);
    // brand is now 48 chars + null. Trim trailing spaces for the
    // comparison so the test isn't sensitive to the exact pad count.
    std::string s(brand);
    while (!s.empty() && s.back() == ' ') s.pop_back();
    EXPECT_EQ(s, "AMD Custom Jaguar 8-Core APU");
}

// Unknown leaf (well outside both the standard and extended ranges
// we handle) must return all-zeros. Confirms the default-response
// path in the emitter works.
TEST_F(CpuRuntimeTest, Cpuid_UnknownLeaf_ReturnsZeros) {
    const u8 program[] = {
        0x48, 0xc7, 0xc0, 0x55, 0x55, 0x00, 0x00,  // mov rax, 0x5555
        0x48, 0x31, 0xc9,
        0x0f, 0xa2,
        0xc3,
    };
    const auto r = RunProgram(program, sizeof(program), mem);
    EXPECT_EQ(r.state.gpr[0], 0ULL);
    EXPECT_EQ(r.state.gpr[3], 0ULL);
    EXPECT_EQ(r.state.gpr[1], 0ULL);
    EXPECT_EQ(r.state.gpr[2], 0ULL);
}

// ============================================================================
// BT — bit test. Sets CF to the tested bit, leaves other guest rflags bits
// unchanged. Original emitter handled only the 64-bit reg-reg form; the
// 32-bit imm form (observed at libc 0x808bdad64 inside a CPUID probe) and
// the 64-bit imm form now route through the same path.
// ============================================================================

// Bit set: BT eax, 5 with eax = 0x20 (bit 5 = 1) → CF=1.
TEST_F(CpuRuntimeTest, Bt32_Imm_SetBit_RaisesCf) {
    const u8 program[] = {
        0x48, 0xc7, 0xc0, 0x20, 0x00, 0x00, 0x00,   // mov rax, 0x20
        0x0f, 0xba, 0xe0, 0x05,                     // bt eax, 5
        0xc3,
    };
    const auto r = RunProgram(program, sizeof(program), mem);
    EXPECT_TRUE(r.state.rflags & 1ULL) << "CF must reflect bit 5 = 1";
}

// Bit clear differential: same shape, bit 5 not set → CF=0.
TEST_F(CpuRuntimeTest, Bt32_Imm_ClearBit_ClearsCf) {
    const u8 program[] = {
        // First force CF=1 via STC so the test fails loudly if the BT
        // emitter happens to leave CF alone instead of clearing it.
        0xf9,                                       // stc
        0x48, 0xc7, 0xc0, 0xdf, 0xff, 0x00, 0x00,   // mov rax, 0xFFDF (bit 5 clear)
        0x0f, 0xba, 0xe0, 0x05,                     // bt eax, 5
        0xc3,
    };
    const auto r = RunProgram(program, sizeof(program), mem);
    EXPECT_FALSE(r.state.rflags & 1ULL) << "CF must reflect bit 5 = 0";
}

// "Other rflags bits preserved" regression: prime ZF via a cmp that
// produces ZF=1 (cmp rax, rax), then run a BT that flips CF to 1.
// Both ZF (bit 6) and CF (bit 0) must be set in the final rflags.
TEST_F(CpuRuntimeTest, Bt32_Imm_PreservesOtherFlags) {
    const u8 program[] = {
        0x48, 0x39, 0xc0,                           // cmp rax, rax  → ZF=1, CF=0
        0x48, 0xc7, 0xc0, 0x20, 0x00, 0x00, 0x00,   // mov rax, 0x20
        0x0f, 0xba, 0xe0, 0x05,                     // bt eax, 5
        0xc3,
    };
    const auto r = RunProgram(program, sizeof(program), mem);
    EXPECT_TRUE(r.state.rflags & 1ULL)        << "CF must be 1 after BT";
    EXPECT_TRUE(r.state.rflags & (1ULL << 6)) << "ZF set by earlier cmp must survive BT";
}

// 64-bit imm form — same skeleton, REX.W differentiates encoding.
TEST_F(CpuRuntimeTest, Bt64_Imm_SetBit_RaisesCf) {
    const u8 program[] = {
        0x48, 0xc7, 0xc0, 0x20, 0x00, 0x00, 0x00,   // mov rax, 0x20
        0x48, 0x0f, 0xba, 0xe0, 0x05,               // bt rax, 5
        0xc3,
    };
    const auto r = RunProgram(program, sizeof(program), mem);
    EXPECT_TRUE(r.state.rflags & 1ULL);
}

// 32-bit reg-reg form: bit index in ecx. Verifies the new width-32
// reg path routes correctly.
TEST_F(CpuRuntimeTest, Bt32_RegReg_BitIndexFromEcx) {
    const u8 program[] = {
        0x48, 0xc7, 0xc0, 0x00, 0x04, 0x00, 0x00,   // mov rax, 0x400 (bit 10)
        0x48, 0xc7, 0xc1, 0x0a, 0x00, 0x00, 0x00,   // mov rcx, 10
        0x0f, 0xa3, 0xc8,                           // bt eax, ecx
        0xc3,
    };
    const auto r = RunProgram(program, sizeof(program), mem);
    EXPECT_TRUE(r.state.rflags & 1ULL);
}

// ============================================================================
// XGETBV — read extended control register. Reports XCR0 = 0x7 (x87+SSE+AVX
// state enabled) so the canonical post-CPUID AVX check passes. Any other
// XCR index returns zero.
// ============================================================================

// `xgetbv` with ecx=0 must produce XCR0 = 0x7 in edx:eax. Specifically:
// eax (= XCR0[31:0]) = 0x7, edx (= XCR0[63:32]) = 0. Pre-pollutes both
// guest slots so missing zero-extends would be visible.
TEST_F(CpuRuntimeTest, Xgetbv_Xcr0_ReportsAvxEnabled) {
    const u8 program[] = {
        // mov rax, 0xDEADBEEFDEADBEEF — pre-pollute future RAX
        0x48, 0xb8, 0xef, 0xbe, 0xad, 0xde, 0xef, 0xbe, 0xad, 0xde,
        // mov rdx, 0xCAFEBABECAFEBABE — pre-pollute future RDX
        0x48, 0xba, 0xbe, 0xba, 0xfe, 0xca, 0xbe, 0xba, 0xfe, 0xca,
        // xor rcx, rcx  (ecx = 0 = XCR0)
        0x48, 0x31, 0xc9,
        0x0f, 0x01, 0xd0,                          // xgetbv
        0xc3,
    };
    const auto r = RunProgram(program, sizeof(program), mem);
    EXPECT_EQ(r.state.gpr[0], 0x7ULL) << "EAX = XCR0[31:0] = x87|SSE|AVX";
    EXPECT_EQ(r.state.gpr[2], 0x0ULL) << "EDX = XCR0[63:32] = 0";
}

// Unknown XCR index returns zero. ecx=1 isn't a real XCR on Jaguar-era CPUs;
// the emitter must not echo the XCR0 response.
TEST_F(CpuRuntimeTest, Xgetbv_UnknownIndex_ReturnsZero) {
    const u8 program[] = {
        0x48, 0xc7, 0xc1, 0x01, 0x00, 0x00, 0x00,  // mov rcx, 1
        0x0f, 0x01, 0xd0,                          // xgetbv
        0xc3,
    };
    const auto r = RunProgram(program, sizeof(program), mem);
    EXPECT_EQ(r.state.gpr[0], 0ULL) << "Unknown XCR → EAX = 0";
    EXPECT_EQ(r.state.gpr[2], 0ULL) << "Unknown XCR → EDX = 0";
}

// ============================================================================
// VPTEST — AVX bit-test across an entire vector. Sets ZF based on
// (a AND b == 0), CF based on (NOT a AND b == 0). Other arithmetic
// flags (OF/SF/AF/PF) cleared.
// ============================================================================

// All-zero AND: a=0xAAAA…, b=0x5555… → (a AND b) = 0 → ZF=1.
// (NOT a) AND b = (0x5555…) AND (0x5555…) ≠ 0 → CF=0.
TEST_F(CpuRuntimeTest, Vptest_DisjointBits_SetsZfNotCf) {
    const u8 program[] = {
        0xc4, 0xe2, 0x79, 0x17, 0xc1,              // vptest xmm0, xmm1
        0xc3,
    };
    GuestMemory& m = mem;
    std::memcpy(m.CodePtr(), program, sizeof(program));
    u8* guest_rsp = m.StackTop() - 8;
    *reinterpret_cast<u64*>(guest_rsp) = kReturnSentinel;
    GuestState st{};
    st.rip = reinterpret_cast<u64>(m.CodePtr());
    st.gpr[4] = reinterpret_cast<u64>(guest_rsp);
    // ymm[0..1] = xmm0 low/high 64; ymm[4..5] = xmm1 low/high 64
    // (lane stride is 4 u64s = 32 bytes).
    st.ymm[0] = 0xAAAAAAAAAAAAAAAAULL;
    st.ymm[1] = 0xAAAAAAAAAAAAAAAAULL;
    st.ymm[4] = 0x5555555555555555ULL;
    st.ymm[5] = 0x5555555555555555ULL;
    Runtime rt;
    rt.Run(st);
    EXPECT_TRUE(st.rflags & (1ULL << 6)) << "ZF=1 — disjoint bit patterns";
    EXPECT_FALSE(st.rflags & 1ULL)       << "CF=0 — b is not a subset of a";
}

// Identical operands: a == b ≠ 0. (a AND b) = a ≠ 0 → ZF=0.
// (NOT a) AND b = (NOT a) AND a = 0 → CF=1.
TEST_F(CpuRuntimeTest, Vptest_Identical_SetsCfNotZf) {
    const u8 program[] = {
        0xc4, 0xe2, 0x79, 0x17, 0xc1,              // vptest xmm0, xmm1
        0xc3,
    };
    GuestMemory& m = mem;
    std::memcpy(m.CodePtr(), program, sizeof(program));
    u8* guest_rsp = m.StackTop() - 8;
    *reinterpret_cast<u64*>(guest_rsp) = kReturnSentinel;
    GuestState st{};
    st.rip = reinterpret_cast<u64>(m.CodePtr());
    st.gpr[4] = reinterpret_cast<u64>(guest_rsp);
    st.ymm[0] = 0x123456789ABCDEF0ULL;
    st.ymm[1] = 0x0FEDCBA987654321ULL;
    st.ymm[4] = 0x123456789ABCDEF0ULL;
    st.ymm[5] = 0x0FEDCBA987654321ULL;
    Runtime rt;
    rt.Run(st);
    EXPECT_FALSE(st.rflags & (1ULL << 6)) << "ZF=0 — non-zero common bits";
    EXPECT_TRUE(st.rflags & 1ULL)         << "CF=1 — b is a subset of a (b == a)";
}

// All-zero operands: both ZF and CF set (everything is the zero set).
TEST_F(CpuRuntimeTest, Vptest_BothZero_SetsBothZfCf) {
    const u8 program[] = {
        0xc4, 0xe2, 0x79, 0x17, 0xc1,              // vptest xmm0, xmm1
        0xc3,
    };
    GuestMemory& m = mem;
    std::memcpy(m.CodePtr(), program, sizeof(program));
    u8* guest_rsp = m.StackTop() - 8;
    *reinterpret_cast<u64*>(guest_rsp) = kReturnSentinel;
    GuestState st{};
    st.rip = reinterpret_cast<u64>(m.CodePtr());
    st.gpr[4] = reinterpret_cast<u64>(guest_rsp);
    // ymm zero by default
    Runtime rt;
    rt.Run(st);
    EXPECT_TRUE(st.rflags & (1ULL << 6));
    EXPECT_TRUE(st.rflags & 1ULL);
}

// ============================================================================
// 8-bit OR/AND/XOR mem-dst with imm src — bitfield-update idiom.
// The dispatcher routes 8-bit OR to EmitNarrowArith8; that function now
// handles mem-dst writeback (previously rejected for everything except
// Cmp/Test). Same path covers AND/XOR/ADD/SUB for symmetry.
// ============================================================================

// `or byte[rax], 0x40` — sets bit 6 of the byte pointed to by rax.
// Test verifies the byte is updated in place and CF/OF are cleared
// (OR semantics) while other rflags bits survive.
TEST_F(CpuRuntimeTest, Or8_MemImm_SetsBitInPlace) {
    // Layout: program at code page start, scratch byte one page later.
    u8* scratch = mem.CodePtr() + 0x100;
    *scratch = 0x01;                                  // bit 0 set, bit 6 clear

    const u8 program[] = {
        // mov rax, <scratch addr> (filled below)
        0x48, 0xb8, 0,0,0,0,0,0,0,0,
        0x80, 0x08, 0x40,                             // or byte[rax], 0x40
        0xc3,
    };
    u8 prog[sizeof(program)];
    std::memcpy(prog, program, sizeof(program));
    const u64 scratch_addr = reinterpret_cast<u64>(scratch);
    std::memcpy(prog + 2, &scratch_addr, sizeof(scratch_addr));

    const auto r = RunProgram(prog, sizeof(prog), mem);
    EXPECT_EQ(*scratch, 0x41) << "bit 6 ORed into 0x01 → 0x41";
    EXPECT_FALSE(r.state.rflags & 1ULL) << "OR clears CF";
}

// AND form: same dispatcher path, different op. Confirms the
// extended NarrowArith8 mem-dst block handles AND too, not just OR.
TEST_F(CpuRuntimeTest, And8_MemImm_ClearsBitInPlace) {
    u8* scratch = mem.CodePtr() + 0x100;
    *scratch = 0xFF;

    const u8 program[] = {
        0x48, 0xb8, 0,0,0,0,0,0,0,0,                  // mov rax, <addr>
        0x80, 0x20, 0xF0,                             // and byte[rax], 0xF0
        0xc3,
    };
    u8 prog[sizeof(program)];
    std::memcpy(prog, program, sizeof(program));
    const u64 scratch_addr = reinterpret_cast<u64>(scratch);
    std::memcpy(prog + 2, &scratch_addr, sizeof(scratch_addr));

    RunProgram(prog, sizeof(prog), mem);
    EXPECT_EQ(*scratch, 0xF0) << "0xFF AND 0xF0 = 0xF0";
}

// ============================================================================
// VPCMPISTRI — SSE4.2 string compare, return index. Used in glibc
// string functions. Captures host ECX → guest RCX and host arithmetic
// flags → guest rflags.
// ============================================================================

// glibc-style strlen idiom: imm = 0x08 (unsigned bytes, equal-each
// aggregation, no polarity, LSB output). xmm0 = string (first source);
// xmm1 = zeros (second source). VPCMPISTRI computes for each byte
// position whether xmm0[i] == xmm1[i] — i.e. where the string has a
// zero byte. ECX gets the index of the first match.
//
// For "hello\0..." the first zero is at byte 5 → ECX = 5.
TEST_F(CpuRuntimeTest, Vpcmpistri_StrlenIdiom_FindsNullTerminator) {
    const u8 program[] = {
        0xc4, 0xe3, 0x79, 0x63, 0xc1, 0x08,           // vpcmpistri xmm0, xmm1, 0x08
        0xc3,
    };
    GuestMemory& m = mem;
    std::memcpy(m.CodePtr(), program, sizeof(program));
    u8* guest_rsp = m.StackTop() - 8;
    *reinterpret_cast<u64*>(guest_rsp) = kReturnSentinel;
    GuestState st{};
    st.rip = reinterpret_cast<u64>(m.CodePtr());
    st.gpr[4] = reinterpret_cast<u64>(guest_rsp);

    // xmm0 = "hello\0\0\0...\0" (16 bytes total). Place via memcpy
    // straight into the YMM lane bytes; ymm[0..1] covers xmm0's low
    // 128 bits with lane stride = 32 bytes.
    u8 xmm0_bytes[16] = {'h','e','l','l','o',0,0,0,0,0,0,0,0,0,0,0};
    std::memcpy(&st.ymm[0], xmm0_bytes, 16);
    // xmm1 = zeros (already, since ymm[] zero-initialized).

    Runtime rt;
    rt.Run(st);
    EXPECT_EQ(st.gpr[1] & 0xFFFFFFFFULL, 5ULL)
        << "ECX = position of first match (the null terminator at byte 5)";
    EXPECT_EQ(st.gpr[1] >> 32, 0ULL) << "Upper 32 of RCX must be zero-extended";
}

// No-match case for "equal each": string has no zero byte in any of
// its 16 positions. Spec says ECX = 16 (element count) when no match.
TEST_F(CpuRuntimeTest, Vpcmpistri_NoMatch_ReturnsElementCount) {
    const u8 program[] = {
        0xc4, 0xe3, 0x79, 0x63, 0xc1, 0x08,           // vpcmpistri xmm0, xmm1, 0x08
        0xc3,
    };
    GuestMemory& m = mem;
    std::memcpy(m.CodePtr(), program, sizeof(program));
    u8* guest_rsp = m.StackTop() - 8;
    *reinterpret_cast<u64*>(guest_rsp) = kReturnSentinel;
    GuestState st{};
    st.rip = reinterpret_cast<u64>(m.CodePtr());
    st.gpr[4] = reinterpret_cast<u64>(guest_rsp);

    // xmm0 = 16 nonzero bytes; xmm1 = zeros. With equal-each, no
    // position matches because no byte in xmm0 equals zero.
    std::memset(&st.ymm[0], 0x41, 16);
    // xmm1 stays zero.

    Runtime rt;
    rt.Run(st);
    EXPECT_EQ(st.gpr[1] & 0xFFFFFFFFULL, 16ULL)
        << "No match → ECX = element count (16 bytes)";
}

// ============================================================================
// DIV 32-bit — EDX:EAX / r32 → EAX = quotient, EDX = remainder. The
// existing emitter handled 64-bit only; the 32-bit variant landed
// inside Sonic Mania's ELF entry at 0x800001046.
// ============================================================================

// 64-bit dividend 0x1_0000_0001 / 2 = 0x80000000 remainder 1.
// Sets up high half via RDX = 1, low half via RAX = 1.
TEST_F(CpuRuntimeTest, Div32_DividendStraddlesEdxEax) {
    const u8 program[] = {
        // mov rax, 1     (low half = 1)
        0x48, 0xc7, 0xc0, 0x01, 0x00, 0x00, 0x00,
        // mov rdx, 1     (high half = 1 → dividend = 0x1_0000_0001)
        0x48, 0xc7, 0xc2, 0x01, 0x00, 0x00, 0x00,
        // mov rcx, 2     (divisor)
        0x48, 0xc7, 0xc1, 0x02, 0x00, 0x00, 0x00,
        0xf7, 0xf1,                                 // div ecx
        0xc3,
    };
    const auto r = RunProgram(program, sizeof(program), mem);
    EXPECT_EQ(r.state.gpr[0], 0x80000000ULL)
        << "Quotient = 0x1_0000_0001 / 2 = 0x8000_0000";
    EXPECT_EQ(r.state.gpr[2], 1ULL) << "Remainder = 1";
    EXPECT_EQ(r.state.gpr[0] >> 32, 0u) << "RAX upper 32 must zero-extend";
    EXPECT_EQ(r.state.gpr[2] >> 32, 0u) << "RDX upper 32 must zero-extend";
}

// Confirm the upper 32 of guest RAX/RDX is IGNORED by 32-bit DIV.
// Pre-pollutes those upper halves; if the emitter erroneously loaded
// the full 64-bit slots into host rdx:rax we'd see a wildly different
// quotient. With xor edx,edx the dividend is just guest EAX low 32.
TEST_F(CpuRuntimeTest, Div32_IgnoresUpper32OfDividend) {
    const u8 program[] = {
        // mov rax, 0xCAFEBABE0000000A  — upper junk, low = 10
        0x48, 0xb8, 0x0a, 0x00, 0x00, 0x00, 0xbe, 0xba, 0xfe, 0xca,
        // mov rdx, 0xDEADBEEF00000000  — upper junk, low = 0
        0x48, 0xba, 0x00, 0x00, 0x00, 0x00, 0xef, 0xbe, 0xad, 0xde,
        // mov rcx, 3
        0x48, 0xc7, 0xc1, 0x03, 0x00, 0x00, 0x00,
        0xf7, 0xf1,                                 // div ecx
        0xc3,
    };
    const auto r = RunProgram(program, sizeof(program), mem);
    EXPECT_EQ(r.state.gpr[0], 3ULL)  << "10 / 3 = 3 (32-bit DIV should ignore upper-32 junk)";
    EXPECT_EQ(r.state.gpr[2], 1ULL)  << "10 % 3 = 1";
}

// ============================================================================
// XADD 32-bit mem-dst, reg-src — atomic exchange-and-add. Sets all
// arithmetic flags from the addition.
// ============================================================================

// Atomic increment idiom: scratch byte initialized to 5, ecx = 3.
// After xadd: memory = 5 + 3 = 8, ecx = 5 (the old memory value).
TEST_F(CpuRuntimeTest, Xadd32_ExchangesAndAdds) {
    u32* scratch = reinterpret_cast<u32*>(mem.CodePtr() + 0x100);
    *scratch = 5;

    const u8 program[] = {
        // mov rax, <scratch addr>
        0x48, 0xb8, 0,0,0,0,0,0,0,0,
        // mov rcx, 3
        0x48, 0xc7, 0xc1, 0x03, 0x00, 0x00, 0x00,
        0x0f, 0xc1, 0x08,                           // xadd dword[rax], ecx
        0xc3,
    };
    u8 prog[sizeof(program)];
    std::memcpy(prog, program, sizeof(program));
    const u64 scratch_addr = reinterpret_cast<u64>(scratch);
    std::memcpy(prog + 2, &scratch_addr, sizeof(scratch_addr));

    const auto r = RunProgram(prog, sizeof(prog), mem);
    EXPECT_EQ(*scratch, 8u)         << "[mem] = old_mem + reg = 5 + 3";
    EXPECT_EQ(r.state.gpr[1], 5ULL) << "reg gets the OLD mem value (5)";
    EXPECT_EQ(r.state.gpr[1] >> 32, 0u) << "RCX upper 32 zero-extended";
}

// Zero-result variant: 0xFFFF_FFFE + 2 overflows to 0 (with CF=1).
// Verifies flags are captured from the addition.
TEST_F(CpuRuntimeTest, Xadd32_WrapToZero_SetsCfAndZf) {
    u32* scratch = reinterpret_cast<u32*>(mem.CodePtr() + 0x100);
    *scratch = 0xFFFFFFFEu;

    const u8 program[] = {
        0x48, 0xb8, 0,0,0,0,0,0,0,0,                // mov rax, <addr>
        0x48, 0xc7, 0xc1, 0x02, 0x00, 0x00, 0x00,   // mov rcx, 2
        0x0f, 0xc1, 0x08,                           // xadd dword[rax], ecx
        0xc3,
    };
    u8 prog[sizeof(program)];
    std::memcpy(prog, program, sizeof(program));
    const u64 scratch_addr = reinterpret_cast<u64>(scratch);
    std::memcpy(prog + 2, &scratch_addr, sizeof(scratch_addr));

    const auto r = RunProgram(prog, sizeof(prog), mem);
    EXPECT_EQ(*scratch, 0u)         << "0xFFFF_FFFE + 2 wraps to 0";
    EXPECT_TRUE(r.state.rflags & 1ULL)        << "CF set on 32-bit add wrap";
    EXPECT_TRUE(r.state.rflags & (1ULL << 6)) << "ZF set on zero result";
}

// ============================================================================
// VPHADDD — horizontal add of 32-bit packed integers. dst[0..1] from src1,
// dst[2..3] from src2 (in each 128-bit lane).
// ============================================================================

// `vphaddd xmm0, xmm0, xmm1` with xmm0 = {1,2,3,4} and xmm1 = {10,20,30,40}.
// Expected: xmm0 = {1+2, 3+4, 10+20, 30+40} = {3, 7, 30, 70}.
TEST_F(CpuRuntimeTest, Vphaddd_Xmm_PairwiseAddsAcrossOperands) {
    const u8 program[] = {
        0xc4, 0xe2, 0x79, 0x02, 0xc1,               // vphaddd xmm0, xmm0, xmm1
        0xc3,
    };
    GuestMemory& m = mem;
    std::memcpy(m.CodePtr(), program, sizeof(program));
    u8* guest_rsp = m.StackTop() - 8;
    *reinterpret_cast<u64*>(guest_rsp) = kReturnSentinel;
    GuestState st{};
    st.rip = reinterpret_cast<u64>(m.CodePtr());
    st.gpr[4] = reinterpret_cast<u64>(guest_rsp);

    // xmm0 = {1, 2, 3, 4} (four 32-bit dwords, low → high)
    u32 src1[4] = {1, 2, 3, 4};
    std::memcpy(&st.ymm[0], src1, 16);
    // xmm1 = {10, 20, 30, 40}
    u32 src2[4] = {10, 20, 30, 40};
    std::memcpy(&st.ymm[4], src2, 16);

    Runtime rt;
    rt.Run(st);

    u32 out[4];
    std::memcpy(out, &st.ymm[0], 16);
    EXPECT_EQ(out[0], 1u + 2u)   << "dst[0] = src1[0] + src1[1]";
    EXPECT_EQ(out[1], 3u + 4u)   << "dst[1] = src1[2] + src1[3]";
    EXPECT_EQ(out[2], 10u + 20u) << "dst[2] = src2[0] + src2[1]";
    EXPECT_EQ(out[3], 30u + 40u) << "dst[3] = src2[2] + src2[3]";

    // 128-bit VEX form must zero bits 255:128 of the destination YMM.
    EXPECT_EQ(st.ymm[2], 0ULL) << "VEX-128 zeroes ymm[lane=2]";
    EXPECT_EQ(st.ymm[3], 0ULL) << "VEX-128 zeroes ymm[lane=3]";
}

// ============================================================================
// VMOVAPS — aligned packed-FP vector move. Treated identically to VMOVUPS
// since alignment requirements don't matter for the GPR-relayed transfers
// our emitter performs.
// ============================================================================

// `vmovaps [mem], xmm0` storing xmm0's 16 bytes to memory.
TEST_F(CpuRuntimeTest, Vmovaps_StoresXmmToMemory) {
    // Memory destination = scratch region inside the code page.
    alignas(16) u8* scratch = reinterpret_cast<u8*>(
        (reinterpret_cast<uintptr_t>(mem.CodePtr() + 0x100) + 15) & ~uintptr_t{15});

    const u8 program[] = {
        // mov rax, <scratch addr>
        0x48, 0xb8, 0,0,0,0,0,0,0,0,
        // vmovaps [rax], xmm0   (3-byte VEX form: c5 f8 29 00)
        0xc5, 0xf8, 0x29, 0x00,
        0xc3,
    };
    u8 prog[sizeof(program)];
    std::memcpy(prog, program, sizeof(program));
    const u64 scratch_addr = reinterpret_cast<u64>(scratch);
    std::memcpy(prog + 2, &scratch_addr, sizeof(scratch_addr));

    std::memcpy(mem.CodePtr(), prog, sizeof(prog));
    u8* guest_rsp = mem.StackTop() - 8;
    *reinterpret_cast<u64*>(guest_rsp) = kReturnSentinel;
    GuestState st{};
    st.rip = reinterpret_cast<u64>(mem.CodePtr());
    st.gpr[4] = reinterpret_cast<u64>(guest_rsp);
    // xmm0 = some pattern
    st.ymm[0] = 0x0123456789ABCDEFULL;
    st.ymm[1] = 0xFEDCBA9876543210ULL;
    // Clobber the destination region first so a no-op emitter would
    // leave detectable junk.
    std::memset(scratch, 0xCC, 16);

    Runtime rt;
    rt.Run(st);

    u64 lo, hi;
    std::memcpy(&lo, scratch + 0, 8);
    std::memcpy(&hi, scratch + 8, 8);
    EXPECT_EQ(lo, 0x0123456789ABCDEFULL) << "low 64 of xmm0 written to [mem]";
    EXPECT_EQ(hi, 0xFEDCBA9876543210ULL) << "high 64 of xmm0 written to [mem+8]";
}

// ============================================================================
// VMOVQ — 64-bit moves between XMM ↔ XMM and XMM ↔ GPR.
// ============================================================================

// xmm ← r64: copies full 64-bit GPR into xmm low 64, zeroes the rest.
// `vmovq xmm0, rax` encodes as `c4 e1 f9 6e c0` (5 bytes).
TEST_F(CpuRuntimeTest, Vmovq_XmmFromGpr_ZeroesUpper) {
    const u8 program[] = {
        // mov rax, 0xDEADBEEFCAFEBABE
        0x48, 0xb8, 0xbe, 0xba, 0xfe, 0xca, 0xef, 0xbe, 0xad, 0xde,
        0xc4, 0xe1, 0xf9, 0x6e, 0xc0,               // vmovq xmm0, rax
        0xc3,
    };
    std::memcpy(mem.CodePtr(), program, sizeof(program));
    u8* guest_rsp = mem.StackTop() - 8;
    *reinterpret_cast<u64*>(guest_rsp) = kReturnSentinel;
    GuestState st{};
    st.rip = reinterpret_cast<u64>(mem.CodePtr());
    st.gpr[4] = reinterpret_cast<u64>(guest_rsp);
    // Pre-pollute the destination YMM with junk so a missing upper-
    // zeroing would be visible.
    st.ymm[0] = 0xDDDDDDDDDDDDDDDDULL;
    st.ymm[1] = 0xDDDDDDDDDDDDDDDDULL;
    st.ymm[2] = 0xDDDDDDDDDDDDDDDDULL;
    st.ymm[3] = 0xDDDDDDDDDDDDDDDDULL;

    Runtime rt;
    rt.Run(st);
    EXPECT_EQ(st.ymm[0], 0xDEADBEEFCAFEBABEULL) << "xmm0 low 64 = rax";
    EXPECT_EQ(st.ymm[1], 0ULL) << "xmm0 high 64 must be zero";
    EXPECT_EQ(st.ymm[2], 0ULL) << "ymm0 lane 2 must be zero (VEX upper-zero)";
    EXPECT_EQ(st.ymm[3], 0ULL) << "ymm0 lane 3 must be zero";
}

// r64 ← xmm: full 64-bit overwrite of the GPR with xmm low 64.
// `vmovq rax, xmm0` encodes as `c4 e1 f9 7e c0` (5 bytes).
TEST_F(CpuRuntimeTest, Vmovq_GprFromXmm_FullWidth) {
    const u8 program[] = {
        0xc4, 0xe1, 0xf9, 0x7e, 0xc0,               // vmovq rax, xmm0
        0xc3,
    };
    std::memcpy(mem.CodePtr(), program, sizeof(program));
    u8* guest_rsp = mem.StackTop() - 8;
    *reinterpret_cast<u64*>(guest_rsp) = kReturnSentinel;
    GuestState st{};
    st.rip = reinterpret_cast<u64>(mem.CodePtr());
    st.gpr[4] = reinterpret_cast<u64>(guest_rsp);
    st.gpr[0] = 0xCAFEBABECAFEBABEULL; // rax pre-pollution
    st.ymm[0] = 0x0123456789ABCDEFULL;

    Runtime rt;
    rt.Run(st);
    EXPECT_EQ(st.gpr[0], 0x0123456789ABCDEFULL) << "rax = xmm0 low 64";
}

// ============================================================================
// SETcc with memory destination — observed at libc 0x800001067.
// ============================================================================

// `setnbe byte[mem]` where the condition is "not below or equal"
// (CF=0 AND ZF=0). Pre-set guest rflags with CF=0,ZF=0 so the
// condition is TRUE; expect byte = 1 written to memory.
TEST_F(CpuRuntimeTest, Setnbe_MemDst_ConditionTrue_StoresOne) {
    u8* scratch = mem.CodePtr() + 0x100;
    *scratch = 0xAA; // pre-pollute

    const u8 program[] = {
        // mov rax, <scratch addr>
        0x48, 0xb8, 0,0,0,0,0,0,0,0,
        // setnbe byte[rax]   (3-byte: 0F 97 /0 = 00 mod, /0 ext, [rax])
        0x0f, 0x97, 0x00,
        0xc3,
    };
    u8 prog[sizeof(program)];
    std::memcpy(prog, program, sizeof(program));
    const u64 scratch_addr = reinterpret_cast<u64>(scratch);
    std::memcpy(prog + 2, &scratch_addr, sizeof(scratch_addr));

    std::memcpy(mem.CodePtr(), prog, sizeof(prog));
    u8* guest_rsp = mem.StackTop() - 8;
    *reinterpret_cast<u64*>(guest_rsp) = kReturnSentinel;
    GuestState st{};
    st.rip = reinterpret_cast<u64>(mem.CodePtr());
    st.gpr[4] = reinterpret_cast<u64>(guest_rsp);
    st.rflags = 0; // CF=0, ZF=0 → setnbe condition TRUE

    Runtime rt;
    rt.Run(st);
    EXPECT_EQ(*scratch, 1) << "setnbe with CF=0&&ZF=0 stores 1 to [mem]";
}

// Condition FALSE: with CF=1, "not below or equal" is FALSE → byte = 0.
TEST_F(CpuRuntimeTest, Setnbe_MemDst_ConditionFalse_StoresZero) {
    u8* scratch = mem.CodePtr() + 0x100;
    *scratch = 0xAA;

    const u8 program[] = {
        0x48, 0xb8, 0,0,0,0,0,0,0,0,                // mov rax, <addr>
        0x0f, 0x97, 0x00,                           // setnbe byte[rax]
        0xc3,
    };
    u8 prog[sizeof(program)];
    std::memcpy(prog, program, sizeof(program));
    const u64 scratch_addr = reinterpret_cast<u64>(scratch);
    std::memcpy(prog + 2, &scratch_addr, sizeof(scratch_addr));

    std::memcpy(mem.CodePtr(), prog, sizeof(prog));
    u8* guest_rsp = mem.StackTop() - 8;
    *reinterpret_cast<u64*>(guest_rsp) = kReturnSentinel;
    GuestState st{};
    st.rip = reinterpret_cast<u64>(mem.CodePtr());
    st.gpr[4] = reinterpret_cast<u64>(guest_rsp);
    st.rflags = 1; // CF=1 → setnbe condition FALSE

    Runtime rt;
    rt.Run(st);
    EXPECT_EQ(*scratch, 0) << "setnbe with CF=1 stores 0";
}

// ============================================================================
// POPCNT — population count. ZF set iff src is zero; other arithmetic
// flags cleared. CPUID leaf 1 ECX bit 23 now advertises it.
// ============================================================================

// POPCNT of 0xFF (8 bits set) → 8, ZF=0.
TEST_F(CpuRuntimeTest, Popcnt64_CountsBitsAndClearsZf) {
    const u8 program[] = {
        0x48, 0xc7, 0xc1, 0xff, 0x00, 0x00, 0x00,   // mov rcx, 0xFF
        0xf3, 0x48, 0x0f, 0xb8, 0xc1,               // popcnt rax, rcx
        0xc3,
    };
    const auto r = RunProgram(program, sizeof(program), mem);
    EXPECT_EQ(r.state.gpr[0], 8ULL) << "popcount(0xFF) = 8";
    EXPECT_FALSE(r.state.rflags & (1ULL << 6)) << "ZF clear when src ≠ 0";
}

// POPCNT of 0 → 0, ZF=1.
TEST_F(CpuRuntimeTest, Popcnt64_Zero_SetsZf) {
    const u8 program[] = {
        0x48, 0x31, 0xc9,                           // xor rcx, rcx
        0xf3, 0x48, 0x0f, 0xb8, 0xc1,               // popcnt rax, rcx
        0xc3,
    };
    const auto r = RunProgram(program, sizeof(program), mem);
    EXPECT_EQ(r.state.gpr[0], 0ULL) << "popcount(0) = 0";
    EXPECT_TRUE(r.state.rflags & (1ULL << 6)) << "ZF set when src == 0";
}

// ============================================================================
// VPUNPCKLQDQ — unpack and interleave low 64-bit halves.
//   dst[0] = src1 low 64
//   dst[1] = src2 low 64
// The high 64 of each source within the 128-bit lane is discarded.
// ============================================================================

// `vpunpcklqdq xmm0, xmm1, xmm2` with xmm1's low 64 = A, xmm2's low 64 = B.
// Expected: xmm0 = {A, B} (B in the high 64 of xmm0).
TEST_F(CpuRuntimeTest, Vpunpcklqdq_InterleavesLowQuadwords) {
    const u8 program[] = {
        0xc5, 0xf1, 0x6c, 0xc2,                     // vpunpcklqdq xmm0, xmm1, xmm2
        0xc3,
    };
    std::memcpy(mem.CodePtr(), program, sizeof(program));
    u8* guest_rsp = mem.StackTop() - 8;
    *reinterpret_cast<u64*>(guest_rsp) = kReturnSentinel;
    GuestState st{};
    st.rip = reinterpret_cast<u64>(mem.CodePtr());
    st.gpr[4] = reinterpret_cast<u64>(guest_rsp);

    // xmm1 (ymm lane 1) low 64 = A, high 64 = junk
    st.ymm[4] = 0xAAAAAAAAAAAAAAAAULL;
    st.ymm[5] = 0x1111111111111111ULL; // discarded
    // xmm2 (ymm lane 2) low 64 = B, high 64 = junk
    st.ymm[8] = 0xBBBBBBBBBBBBBBBBULL;
    st.ymm[9] = 0x2222222222222222ULL; // discarded
    // Pre-pollute xmm0 destination
    st.ymm[0] = 0xDEADDEADDEADDEADULL;
    st.ymm[1] = 0xDEADDEADDEADDEADULL;
    st.ymm[2] = 0xDEADDEADDEADDEADULL;
    st.ymm[3] = 0xDEADDEADDEADDEADULL;

    Runtime rt;
    rt.Run(st);
    EXPECT_EQ(st.ymm[0], 0xAAAAAAAAAAAAAAAAULL) << "dst[0] = src1 low 64";
    EXPECT_EQ(st.ymm[1], 0xBBBBBBBBBBBBBBBBULL) << "dst[1] = src2 low 64";
    EXPECT_EQ(st.ymm[2], 0ULL) << "VEX-128 zeroes ymm lane 2";
    EXPECT_EQ(st.ymm[3], 0ULL) << "VEX-128 zeroes ymm lane 3";
}

// ============================================================================
// BEXTR mem-src form — bitfield extract from memory.
//   bextr eax, [mem], ecx — data source is memory, control is GPR.
// Existing emitter handled reg-reg-reg only; mem-src was observed at
// libc 0x80000ad1d inside a structure-field bit-decode path.
// ============================================================================

// Extract bits [8..15] (i.e. start=8, len=8) from a memory dword.
// Control = (8 | (8 << 8)) = 0x808. Memory = 0x12345678 →
// result = (0x12345678 >> 8) & 0xFF = 0x56.
TEST_F(CpuRuntimeTest, Bextr32_MemSrc_ExtractsByteFromMid) {
    u32* scratch = reinterpret_cast<u32*>(mem.CodePtr() + 0x100);
    *scratch = 0x12345678u;

    const u8 program[] = {
        // mov rdi, <scratch addr>
        0x48, 0xbf, 0,0,0,0,0,0,0,0,
        // mov rcx, 0x0808
        0x48, 0xc7, 0xc1, 0x08, 0x08, 0x00, 0x00,
        // bextr eax, dword[rdi], ecx
        0xc4, 0xe2, 0x70, 0xf7, 0x07,
        0xc3,
    };
    u8 prog[sizeof(program)];
    std::memcpy(prog, program, sizeof(program));
    const u64 scratch_addr = reinterpret_cast<u64>(scratch);
    std::memcpy(prog + 2, &scratch_addr, sizeof(scratch_addr));

    const auto r = RunProgram(prog, sizeof(prog), mem);
    EXPECT_EQ(r.state.gpr[0], 0x56ULL) << "BEXTR(0x12345678, start=8, len=8) = 0x56";
    EXPECT_EQ(r.state.gpr[0] >> 32, 0u) << "32-bit result must zero-extend";
}

// ============================================================================
// INC byte[mem] — 8-bit increment with memory destination. The existing
// emitter handled 32/64-bit reg-dst only. INC architecturally preserves CF,
// which we get for free because host INC preserves CF and the rflags
// round-trip captures the rest from the host.
// ============================================================================

// Increment a memory byte; verify the value advances by 1 and CF is
// unchanged. Pre-set CF=1 via STC; INC must NOT clear it.
TEST_F(CpuRuntimeTest, Inc8_MemDst_AdvancesByteAndPreservesCf) {
    u8* scratch = mem.CodePtr() + 0x100;
    *scratch = 0x41; // 'A'

    const u8 program[] = {
        0xf9,                                        // stc  → CF=1
        // mov rax, <scratch addr>
        0x48, 0xb8, 0,0,0,0,0,0,0,0,
        // inc byte[rax]   (FE /0 [rax] = 2 bytes; with disp8 = 3)
        0xfe, 0x00,
        0xc3,
    };
    u8 prog[sizeof(program)];
    std::memcpy(prog, program, sizeof(program));
    const u64 scratch_addr = reinterpret_cast<u64>(scratch);
    std::memcpy(prog + 3, &scratch_addr, sizeof(scratch_addr));

    const auto r = RunProgram(prog, sizeof(prog), mem);
    EXPECT_EQ(*scratch, 0x42) << "byte incremented in-place";
    EXPECT_TRUE(r.state.rflags & 1ULL) << "CF preserved through INC";
}

// 0x7F → 0x80 wraps signed int8 from positive to negative: OF must be set,
// SF must be set, ZF must be clear.
TEST_F(CpuRuntimeTest, Inc8_MemDst_SignedOverflow_SetsOfSf) {
    u8* scratch = mem.CodePtr() + 0x100;
    *scratch = 0x7F;

    const u8 program[] = {
        0x48, 0xb8, 0,0,0,0,0,0,0,0,                // mov rax, <addr>
        0xfe, 0x00,                                  // inc byte[rax]
        0xc3,
    };
    u8 prog[sizeof(program)];
    std::memcpy(prog, program, sizeof(program));
    const u64 scratch_addr = reinterpret_cast<u64>(scratch);
    std::memcpy(prog + 2, &scratch_addr, sizeof(scratch_addr));

    const auto r = RunProgram(prog, sizeof(prog), mem);
    EXPECT_EQ(*scratch, 0x80);
    EXPECT_TRUE(r.state.rflags & (1ULL << 11)) << "OF set on signed wrap +→−";
    EXPECT_TRUE(r.state.rflags & (1ULL << 7))  << "SF set (result MSB = 1)";
    EXPECT_FALSE(r.state.rflags & (1ULL << 6)) << "ZF clear (result ≠ 0)";
}

// ============================================================================
// DEC word[mem] / dword[mem] — memory-destination decrement. The emitter
// previously handled only register dsts (8/32/64-bit). The 16-bit mem form
// (`dec word [mem]`, 66-prefixed) was the run-ending gap in CUSA02394 at libc
// 0x800143b30 — an in-memory 16-bit counter decrement. DEC preserves CF; the
// host DEC does too, so the rflags round-trip carries CF through for free.
// ============================================================================

// dec word[rbx]: 0x0043 → 0x0042, CF preserved (STC first), upper bytes of the
// dword left intact.
TEST_F(CpuRuntimeTest, Dec16_MemDst_DecrementsWordAndPreservesCf) {
    u16* slot = reinterpret_cast<u16*>(mem.CodePtr() + 0x100);
    slot[0] = 0x0043;
    slot[1] = 0xBEEF; // sentinel after the 2-byte store

    const u8 program[] = {
        0xf9,                   // stc → CF=1
        0x66, 0xff, 0x0b,       // dec word [rbx]
        0xc3,                   // ret
    };
    std::memcpy(mem.CodePtr(), program, sizeof(program));
    u8* guest_rsp = mem.StackTop() - 8;
    *reinterpret_cast<u64*>(guest_rsp) = kReturnSentinel;
    GuestState st{};
    st.rip = reinterpret_cast<u64>(mem.CodePtr());
    st.gpr[4] = reinterpret_cast<u64>(guest_rsp);
    st.gpr[3] = reinterpret_cast<u64>(slot); // rbx -> word

    Runtime rt; rt.Run(st);
    EXPECT_EQ(slot[0], 0x0042u) << "word decremented in place";
    EXPECT_EQ(slot[1], 0xBEEFu) << "2-byte store must not spill into next word";
    EXPECT_TRUE(st.rflags & 1ULL) << "CF preserved through DEC";
}

// dec word[rbx] from 0x0000 → 0xFFFF: underflow wraps. CF still preserved
// (DEC never touches CF); SF set (result MSB=1), ZF clear, OF clear (no signed
// overflow: -32768 boundary not crossed; 0 → -1 is fine).
TEST_F(CpuRuntimeTest, Dec16_MemDst_UnderflowWraps) {
    u16* slot = reinterpret_cast<u16*>(mem.CodePtr() + 0x100);
    slot[0] = 0x0000;

    const u8 program[] = {
        0x66, 0xff, 0x0b,       // dec word [rbx]
        0xc3,
    };
    std::memcpy(mem.CodePtr(), program, sizeof(program));
    u8* guest_rsp = mem.StackTop() - 8;
    *reinterpret_cast<u64*>(guest_rsp) = kReturnSentinel;
    GuestState st{};
    st.rip = reinterpret_cast<u64>(mem.CodePtr());
    st.gpr[4] = reinterpret_cast<u64>(guest_rsp);
    st.gpr[3] = reinterpret_cast<u64>(slot);

    Runtime rt; rt.Run(st);
    EXPECT_EQ(slot[0], 0xFFFFu) << "0x0000 - 1 wraps to 0xFFFF at 16-bit width";
    EXPECT_TRUE(st.rflags & (1ULL << 7))  << "SF set (result MSB = 1)";
    EXPECT_FALSE(st.rflags & (1ULL << 6)) << "ZF clear (result != 0)";
    EXPECT_FALSE(st.rflags & (1ULL << 11)) << "OF clear (0 → -1 is not signed overflow)";
}

// dec word[rbx] from 0x8000 → 0x7FFF: signed overflow (most-negative − 1). OF
// set, SF clear (result MSB=0), ZF clear.
TEST_F(CpuRuntimeTest, Dec16_MemDst_SignedOverflow_SetsOf) {
    u16* slot = reinterpret_cast<u16*>(mem.CodePtr() + 0x100);
    slot[0] = 0x8000;

    const u8 program[] = {
        0x66, 0xff, 0x0b,       // dec word [rbx]
        0xc3,
    };
    std::memcpy(mem.CodePtr(), program, sizeof(program));
    u8* guest_rsp = mem.StackTop() - 8;
    *reinterpret_cast<u64*>(guest_rsp) = kReturnSentinel;
    GuestState st{};
    st.rip = reinterpret_cast<u64>(mem.CodePtr());
    st.gpr[4] = reinterpret_cast<u64>(guest_rsp);
    st.gpr[3] = reinterpret_cast<u64>(slot);

    Runtime rt; rt.Run(st);
    EXPECT_EQ(slot[0], 0x7FFFu);
    EXPECT_TRUE(st.rflags & (1ULL << 11)) << "OF set (0x8000 − 1 overflows signed int16)";
    EXPECT_FALSE(st.rflags & (1ULL << 7)) << "SF clear (result MSB = 0)";
    EXPECT_FALSE(st.rflags & (1ULL << 6)) << "ZF clear";
}

// dec word[rbx] from 0x0001 → 0x0000: ZF set.
TEST_F(CpuRuntimeTest, Dec16_MemDst_ToZero_SetsZf) {
    u16* slot = reinterpret_cast<u16*>(mem.CodePtr() + 0x100);
    slot[0] = 0x0001;

    const u8 program[] = {
        0x66, 0xff, 0x0b,       // dec word [rbx]
        0xc3,
    };
    std::memcpy(mem.CodePtr(), program, sizeof(program));
    u8* guest_rsp = mem.StackTop() - 8;
    *reinterpret_cast<u64*>(guest_rsp) = kReturnSentinel;
    GuestState st{};
    st.rip = reinterpret_cast<u64>(mem.CodePtr());
    st.gpr[4] = reinterpret_cast<u64>(guest_rsp);
    st.gpr[3] = reinterpret_cast<u64>(slot);

    Runtime rt; rt.Run(st);
    EXPECT_EQ(slot[0], 0x0000u);
    EXPECT_TRUE(st.rflags & (1ULL << 6))  << "ZF set (result == 0)";
    EXPECT_FALSE(st.rflags & (1ULL << 7)) << "SF clear";
}

// dec word[rbx+0x10] — displaced form (the gap's actual encoding had a
// disp32 address; exercise a disp variant to cover EA resolution).
TEST_F(CpuRuntimeTest, Dec16_MemDstDisp_Decrements) {
    u16* base = reinterpret_cast<u16*>(mem.CodePtr() + 0x100);
    u16* slot = reinterpret_cast<u16*>(mem.CodePtr() + 0x110); // base + 0x10 bytes
    *slot = 0x1235;

    const u8 program[] = {
        0x66, 0xff, 0x4b, 0x10, // dec word [rbx+0x10]
        0xc3,
    };
    std::memcpy(mem.CodePtr(), program, sizeof(program));
    u8* guest_rsp = mem.StackTop() - 8;
    *reinterpret_cast<u64*>(guest_rsp) = kReturnSentinel;
    GuestState st{};
    st.rip = reinterpret_cast<u64>(mem.CodePtr());
    st.gpr[4] = reinterpret_cast<u64>(guest_rsp);
    st.gpr[3] = reinterpret_cast<u64>(base); // rbx -> base; +0x10 reaches slot

    Runtime rt; rt.Run(st);
    EXPECT_EQ(*slot, 0x1234u) << "displaced word decremented";
}

// Regression: dword[mem] DEC (no 66 prefix) still decrements 4 bytes.
TEST_F(CpuRuntimeTest, Dec32_MemDst_Decrements) {
    u32* slot = reinterpret_cast<u32*>(mem.CodePtr() + 0x100);
    slot[0] = 0x00010000;
    slot[1] = 0xDEADBEEF; // sentinel

    const u8 program[] = {
        0xff, 0x0b,             // dec dword [rbx]
        0xc3,
    };
    std::memcpy(mem.CodePtr(), program, sizeof(program));
    u8* guest_rsp = mem.StackTop() - 8;
    *reinterpret_cast<u64*>(guest_rsp) = kReturnSentinel;
    GuestState st{};
    st.rip = reinterpret_cast<u64>(mem.CodePtr());
    st.gpr[4] = reinterpret_cast<u64>(guest_rsp);
    st.gpr[3] = reinterpret_cast<u64>(slot);

    Runtime rt; rt.Run(st);
    EXPECT_EQ(slot[0], 0x0000FFFFu) << "dword decremented";
    EXPECT_EQ(slot[1], 0xDEADBEEFu) << "4-byte store must not touch next dword";
}

// ============================================================================
// VPSHUFD dst, src, imm8 — dword shuffle. For each output dword i,
// dst[i] = src[(imm >> (2*i)) & 3].
// ============================================================================

// Broadcast: imm = 0x00 → all four output dwords = src[0].
TEST_F(CpuRuntimeTest, Vpshufd_BroadcastsLowestDword) {
    const u8 program[] = {
        0xc5, 0xf9, 0x70, 0xc1, 0x00,                // vpshufd xmm0, xmm1, 0x00
        0xc3,
    };
    std::memcpy(mem.CodePtr(), program, sizeof(program));
    u8* guest_rsp = mem.StackTop() - 8;
    *reinterpret_cast<u64*>(guest_rsp) = kReturnSentinel;
    GuestState st{};
    st.rip = reinterpret_cast<u64>(mem.CodePtr());
    st.gpr[4] = reinterpret_cast<u64>(guest_rsp);

    // xmm1 (ymm lane 1) = {0xCAFEBABE, 0xDEADBEEF, 0x11111111, 0x22222222}
    u32 src[4] = {0xCAFEBABEu, 0xDEADBEEFu, 0x11111111u, 0x22222222u};
    std::memcpy(&st.ymm[4], src, 16);
    // Pre-pollute the destination.
    st.ymm[0] = 0xDEADULL;
    st.ymm[1] = 0xDEADULL;
    st.ymm[2] = 0xDEADULL;
    st.ymm[3] = 0xDEADULL;

    Runtime rt;
    rt.Run(st);

    u32 out[4];
    std::memcpy(out, &st.ymm[0], 16);
    EXPECT_EQ(out[0], 0xCAFEBABEu) << "imm=0 broadcasts src[0] to all 4 lanes";
    EXPECT_EQ(out[1], 0xCAFEBABEu);
    EXPECT_EQ(out[2], 0xCAFEBABEu);
    EXPECT_EQ(out[3], 0xCAFEBABEu);
    EXPECT_EQ(st.ymm[2], 0ULL) << "VEX-128 zeroes upper YMM lane";
    EXPECT_EQ(st.ymm[3], 0ULL);
}

// Shuffle pattern 0xE4 = 11_10_01_00 → identity (each output i = src[i]).
// Pattern 0x1B = 00_01_10_11 → reverse: out[0]=src[3], out[1]=src[2],
// out[2]=src[1], out[3]=src[0].
TEST_F(CpuRuntimeTest, Vpshufd_ReversesDwords) {
    const u8 program[] = {
        0xc5, 0xf9, 0x70, 0xc1, 0x1b,                // vpshufd xmm0, xmm1, 0x1b
        0xc3,
    };
    std::memcpy(mem.CodePtr(), program, sizeof(program));
    u8* guest_rsp = mem.StackTop() - 8;
    *reinterpret_cast<u64*>(guest_rsp) = kReturnSentinel;
    GuestState st{};
    st.rip = reinterpret_cast<u64>(mem.CodePtr());
    st.gpr[4] = reinterpret_cast<u64>(guest_rsp);

    u32 src[4] = {1, 2, 3, 4};
    std::memcpy(&st.ymm[4], src, 16);

    Runtime rt;
    rt.Run(st);

    u32 out[4];
    std::memcpy(out, &st.ymm[0], 16);
    EXPECT_EQ(out[0], 4u);
    EXPECT_EQ(out[1], 3u);
    EXPECT_EQ(out[2], 2u);
    EXPECT_EQ(out[3], 1u);
}

// ============================================================================
// LZCNT — count leading zeros. Sets CF=1 when src is zero (signaling
// "no bits set, count = operand size"); ZF=1 when result is zero.
// ============================================================================

// LZCNT of 0xFF (low byte): 32-bit operand → 24 leading zeros.
TEST_F(CpuRuntimeTest, Lzcnt32_CountsHighZeroBits) {
    const u8 program[] = {
        0x48, 0xc7, 0xc1, 0xff, 0x00, 0x00, 0x00,    // mov rcx, 0xFF
        0xf3, 0x0f, 0xbd, 0xc1,                      // lzcnt eax, ecx
        0xc3,
    };
    const auto r = RunProgram(program, sizeof(program), mem);
    EXPECT_EQ(r.state.gpr[0], 24ULL) << "lzcnt(0xFF) for 32-bit = 24";
    EXPECT_FALSE(r.state.rflags & 1ULL)        << "CF clear when src ≠ 0";
    EXPECT_FALSE(r.state.rflags & (1ULL << 6)) << "ZF clear (result ≠ 0)";
}

// LZCNT of 0: result = operand size (32), CF = 1 (signals "no bits set").
TEST_F(CpuRuntimeTest, Lzcnt32_ZeroSrc_ReturnsOperandSizeAndSetsCf) {
    const u8 program[] = {
        0x48, 0x31, 0xc9,                            // xor rcx, rcx
        0xf3, 0x0f, 0xbd, 0xc1,                      // lzcnt eax, ecx
        0xc3,
    };
    const auto r = RunProgram(program, sizeof(program), mem);
    EXPECT_EQ(r.state.gpr[0], 32ULL) << "lzcnt(0) for 32-bit = 32";
    EXPECT_TRUE(r.state.rflags & 1ULL) << "CF set when src == 0";
}

// LZCNT of 0x80000000 (only MSB set): result = 0, ZF = 1.
TEST_F(CpuRuntimeTest, Lzcnt32_MsbSet_ResultIsZero_SetsZf) {
    const u8 program[] = {
        // mov rcx, 0x80000000
        0x48, 0xc7, 0xc1, 0x00, 0x00, 0x00, 0x80,
        0xf3, 0x0f, 0xbd, 0xc1,                      // lzcnt eax, ecx
        0xc3,
    };
    const auto r = RunProgram(program, sizeof(program), mem);
    EXPECT_EQ(r.state.gpr[0], 0ULL) << "lzcnt(0x8000_0000) = 0";
    EXPECT_TRUE(r.state.rflags & (1ULL << 6)) << "ZF set (result == 0)";
}

// ============================================================================
// SBB 8-bit reg-reg — subtract with borrow. dst = dst - src - CF. Previously
// 64-bit only; 8-bit observed at libc 0x8000012b9. Uses byte-offset addressing
// so the surrounding bytes of the dst slot are preserved.
// ============================================================================

// 10 - 3 - 0(CF) = 7. CF clear → straightforward subtraction.
TEST_F(CpuRuntimeTest, Sbb8_BorrowClear_StraightSubtraction) {
    const u8 program[] = {
        // Force CF=0 via a CMP that produces CF=0 (any A-A).
        0x48, 0xc7, 0xc0, 0x0a, 0x00, 0x00, 0x00,    // mov rax, 0x0A
        0x48, 0xc7, 0xc1, 0x03, 0x00, 0x00, 0x00,    // mov rcx, 0x03
        0xf8,                                         // clc → CF=0
        0x18, 0xc8,                                   // sbb al, cl
        0xc3,
    };
    const auto r = RunProgram(program, sizeof(program), mem);
    EXPECT_EQ(r.state.gpr[0] & 0xFFULL, 7ULL) << "10 - 3 - 0 = 7";
    EXPECT_FALSE(r.state.rflags & 1ULL) << "no borrow out";
}

// 10 - 3 - 1(CF) = 6. Verifies the CF input is actually consumed.
TEST_F(CpuRuntimeTest, Sbb8_BorrowSet_SubtractsExtraOne) {
    const u8 program[] = {
        0x48, 0xc7, 0xc0, 0x0a, 0x00, 0x00, 0x00,    // mov rax, 0x0A
        0x48, 0xc7, 0xc1, 0x03, 0x00, 0x00, 0x00,    // mov rcx, 0x03
        0xf9,                                         // stc → CF=1
        0x18, 0xc8,                                   // sbb al, cl
        0xc3,
    };
    const auto r = RunProgram(program, sizeof(program), mem);
    EXPECT_EQ(r.state.gpr[0] & 0xFFULL, 6ULL) << "10 - 3 - 1 = 6";
    EXPECT_FALSE(r.state.rflags & 1ULL) << "no borrow out";
}

// 0 - 1 - 0 wraps to 0xFF (-1 in two's complement 8-bit). CF=1 (borrow out).
TEST_F(CpuRuntimeTest, Sbb8_Underflow_SetsCf) {
    const u8 program[] = {
        0x48, 0x31, 0xc0,                             // xor rax, rax  (al=0)
        0x48, 0xc7, 0xc1, 0x01, 0x00, 0x00, 0x00,    // mov rcx, 1
        0xf8,                                         // clc → CF=0
        0x18, 0xc8,                                   // sbb al, cl
        0xc3,
    };
    const auto r = RunProgram(program, sizeof(program), mem);
    EXPECT_EQ(r.state.gpr[0] & 0xFFULL, 0xFFULL) << "0 - 1 wraps to 0xFF";
    EXPECT_TRUE(r.state.rflags & 1ULL) << "CF set on borrow out of bit 7";
}

// ============================================================================
// CMOVNS 16-bit with memory source. The 16-bit form has a unique merge
// semantic: when condition is TRUE, only the low 16 of dst are overwritten
// — bits 63:16 of the dst slot are PRESERVED (no zero-extension like the
// 32-bit form). When FALSE, the dst is entirely unchanged.
// ============================================================================

// Condition TRUE (SF=0): mem-low-16 replaces dst's low 16; upper 48 bits
// of the dst GPR slot survive verbatim.
TEST_F(CpuRuntimeTest, Cmovns16_MemSrc_TrueMergesLow16PreservingUpper48) {
    u16* scratch = reinterpret_cast<u16*>(mem.CodePtr() + 0x100);
    *scratch = 0xABCD;

    const u8 program[] = {
        // mov rdi, <scratch addr>
        0x48, 0xbf, 0,0,0,0,0,0,0,0,
        // mov rax, 0xDEADBEEFCAFEBABE — pre-pollute dst's upper 48
        0x48, 0xb8, 0xbe, 0xba, 0xfe, 0xca, 0xef, 0xbe, 0xad, 0xde,
        // Force SF=0 via xor rcx,rcx (which clears SF). Then cmovns ax, word[rdi].
        0x48, 0x31, 0xc9,                             // xor rcx, rcx (SF=0 now)
        0x66, 0x0f, 0x49, 0x07,                       // cmovns ax, word[rdi]
        0xc3,
    };
    u8 prog[sizeof(program)];
    std::memcpy(prog, program, sizeof(program));
    const u64 scratch_addr = reinterpret_cast<u64>(scratch);
    std::memcpy(prog + 2, &scratch_addr, sizeof(scratch_addr));

    const auto r = RunProgram(prog, sizeof(prog), mem);
    EXPECT_EQ(r.state.gpr[0], 0xDEADBEEFCAFEABCDULL)
        << "low 16 replaced by 0xABCD, upper 48 preserved";
}

// Condition FALSE (SF=1): dst is unchanged entirely.
TEST_F(CpuRuntimeTest, Cmovns16_MemSrc_FalseLeavesDstUnchanged) {
    u16* scratch = reinterpret_cast<u16*>(mem.CodePtr() + 0x100);
    *scratch = 0x1234;

    const u8 program[] = {
        // mov rdi, <scratch addr>
        0x48, 0xbf, 0,0,0,0,0,0,0,0,
        // mov rax, 0xDEADBEEFCAFEBABE — should remain entirely unchanged
        0x48, 0xb8, 0xbe, 0xba, 0xfe, 0xca, 0xef, 0xbe, 0xad, 0xde,
        // Force SF=1 by computing a negative result (cmp 0, 1 → SF=1, CF=1).
        0x48, 0xc7, 0xc1, 0x00, 0x00, 0x00, 0x00,    // mov rcx, 0
        0x48, 0x83, 0xf9, 0x01,                       // cmp rcx, 1   → SF=1
        // cmovns ax, word[rdi]  — condition FALSE → no change
        0x66, 0x0f, 0x49, 0x07,
        0xc3,
    };
    u8 prog[sizeof(program)];
    std::memcpy(prog, program, sizeof(program));
    const u64 scratch_addr = reinterpret_cast<u64>(scratch);
    std::memcpy(prog + 2, &scratch_addr, sizeof(scratch_addr));

    const auto r = RunProgram(prog, sizeof(prog), mem);
    EXPECT_EQ(r.state.gpr[0], 0xDEADBEEFCAFEBABEULL)
        << "FALSE condition must leave entire 64-bit slot unchanged";
}

// ============================================================================
// SBB 32-bit reg-imm — sign-extended imm8 / imm32 sub-with-borrow.
// Observed at libc 0x80001021d inside a multi-precision subtract chain.
// ============================================================================

// `sbb ecx, 0x10` with rcx = 0x100 and CF = 0 → result = 0xF0; CF clear.
// Verifies the 32-bit write zero-extends rcx (no upper-half leak).
TEST_F(CpuRuntimeTest, Sbb32_RegImm_BorrowClear_AndZeroExtends) {
    const u8 program[] = {
        // mov rcx, 0xDEADBEEF_00000100  — pre-pollute upper 32
        0x48, 0xb9, 0x00, 0x01, 0x00, 0x00, 0xef, 0xbe, 0xad, 0xde,
        0xf8,                                         // clc → CF=0
        0x83, 0xd9, 0x10,                             // sbb ecx, 0x10
        0xc3,
    };
    const auto r = RunProgram(program, sizeof(program), mem);
    EXPECT_EQ(r.state.gpr[1], 0xF0ULL)
        << "0x100 - 0x10 - 0 = 0xF0; upper 32 zero-extended";
    EXPECT_FALSE(r.state.rflags & 1ULL) << "no borrow out";
}

// `sbb ecx, 0x10` with rcx = 0x100 and CF = 1 → result = 0xEF.
TEST_F(CpuRuntimeTest, Sbb32_RegImm_BorrowSet_SubtractsExtraOne) {
    const u8 program[] = {
        0x48, 0xc7, 0xc1, 0x00, 0x01, 0x00, 0x00,    // mov rcx, 0x100
        0xf9,                                         // stc → CF=1
        0x83, 0xd9, 0x10,                             // sbb ecx, 0x10
        0xc3,
    };
    const auto r = RunProgram(program, sizeof(program), mem);
    EXPECT_EQ(r.state.gpr[1], 0xEFULL)
        << "0x100 - 0x10 - 1 = 0xEF; CF input consumed";
}

// Underflow: 0 - 1 - 0 wraps to 0xFFFFFFFF with CF=1.
TEST_F(CpuRuntimeTest, Sbb32_RegImm_Underflow_SetsCf) {
    const u8 program[] = {
        0x48, 0x31, 0xc9,                             // xor rcx, rcx
        0xf8,                                         // clc
        0x83, 0xd9, 0x01,                             // sbb ecx, 1
        0xc3,
    };
    const auto r = RunProgram(program, sizeof(program), mem);
    EXPECT_EQ(r.state.gpr[1], 0xFFFFFFFFULL)
        << "0 - 1 wraps to 0xFFFFFFFF (low 32)";
    EXPECT_EQ(r.state.gpr[1] >> 32, 0u)
        << "32-bit op must zero-extend bits 63:32";
    EXPECT_TRUE(r.state.rflags & 1ULL) << "CF set on borrow out";
}

// ============================================================================
// VMOVSS — scalar single-precision FP move. We don't use the host FP
// register file; the GPR-relayed transfer is bitwise-identical and
// avoids any MXCSR / denormal concerns.
// ============================================================================

// Load form (`vmovss xmm0, dword[rax]`): place the 32-bit value at
// [mem] into xmm0.low32; zero everything else of the YMM. Pre-pollute
// the destination so a missing zero would be visible.
TEST_F(CpuRuntimeTest, Vmovss_Load_PlacesLow32ZeroesRest) {
    u32* scratch = reinterpret_cast<u32*>(mem.CodePtr() + 0x100);
    *scratch = 0x40490FDBu; // bit pattern of float(pi)

    const u8 program[] = {
        // mov rax, <scratch addr>
        0x48, 0xb8, 0,0,0,0,0,0,0,0,
        // vmovss xmm0, dword[rax]  (c5 fa 10 00 = 4 bytes)
        0xc5, 0xfa, 0x10, 0x00,
        0xc3,
    };
    u8 prog[sizeof(program)];
    std::memcpy(prog, program, sizeof(program));
    const u64 scratch_addr = reinterpret_cast<u64>(scratch);
    std::memcpy(prog + 2, &scratch_addr, sizeof(scratch_addr));

    std::memcpy(mem.CodePtr(), prog, sizeof(prog));
    u8* guest_rsp = mem.StackTop() - 8;
    *reinterpret_cast<u64*>(guest_rsp) = kReturnSentinel;
    GuestState st{};
    st.rip = reinterpret_cast<u64>(mem.CodePtr());
    st.gpr[4] = reinterpret_cast<u64>(guest_rsp);
    // Pre-pollute the destination YMM so missing zeroes show up.
    st.ymm[0] = 0xDEADBEEFDEADBEEFULL;
    st.ymm[1] = 0xCAFEBABECAFEBABEULL;
    st.ymm[2] = 0x1234567812345678ULL;
    st.ymm[3] = 0xABCDEF01ABCDEF01ULL;

    Runtime rt;
    rt.Run(st);
    EXPECT_EQ(st.ymm[0], 0x40490FDBULL)
        << "xmm0 low 32 = loaded value, upper 32 of chunk 0 must be zero";
    EXPECT_EQ(st.ymm[1], 0ULL) << "xmm0[127:64] cleared";
    EXPECT_EQ(st.ymm[2], 0ULL) << "ymm0 lane 2 (VEX-128 zero) cleared";
    EXPECT_EQ(st.ymm[3], 0ULL) << "ymm0 lane 3 (VEX-128 zero) cleared";
}

// Store form (`vmovss dword[rax], xmm0`): writes only xmm0.low32 to
// [mem]; the surrounding memory must be untouched.
TEST_F(CpuRuntimeTest, Vmovss_Store_WritesLow32Only) {
    u32* scratch = reinterpret_cast<u32*>(mem.CodePtr() + 0x100);
    *scratch = 0xCAFEBABEu;
    // The 4 bytes immediately after `scratch` should NOT be touched.
    u32* sentinel = scratch + 1;
    *sentinel = 0xDEADBEEFu;

    const u8 program[] = {
        // mov rax, <scratch addr>
        0x48, 0xb8, 0,0,0,0,0,0,0,0,
        // vmovss dword[rax], xmm0  (c5 fa 11 00)
        0xc5, 0xfa, 0x11, 0x00,
        0xc3,
    };
    u8 prog[sizeof(program)];
    std::memcpy(prog, program, sizeof(program));
    const u64 scratch_addr = reinterpret_cast<u64>(scratch);
    std::memcpy(prog + 2, &scratch_addr, sizeof(scratch_addr));

    std::memcpy(mem.CodePtr(), prog, sizeof(prog));
    u8* guest_rsp = mem.StackTop() - 8;
    *reinterpret_cast<u64*>(guest_rsp) = kReturnSentinel;
    GuestState st{};
    st.rip = reinterpret_cast<u64>(mem.CodePtr());
    st.gpr[4] = reinterpret_cast<u64>(guest_rsp);
    st.ymm[0] = 0x1111111122222222ULL; // xmm0: low32=0x22222222, hi32=0x11111111
    Runtime rt;
    rt.Run(st);
    EXPECT_EQ(*scratch, 0x22222222u) << "store wrote xmm0.low32";
    EXPECT_EQ(*sentinel, 0xDEADBEEFu) << "adjacent memory must be untouched";
}

// Reg-reg-reg form: dst.low32 from src2, dst[127:32] from src1, ymm
// upper lane zeroed. This is the form compilers emit for "splat the
// low scalar back together with the upper from somewhere else".
TEST_F(CpuRuntimeTest, Vmovss_RegRegReg_MergesFromTwoSources) {
    // vmovss xmm0, xmm1, xmm2  — encoding c5 f2 10 c2
    //   pp=10 (F3 prefix), L=0, ~vvvv = ~1 = 14 (xmm1 is "first src")
    //   So byte 2 = 0xf2 (= 1111_0010 = W=0 vvvv=1110 L=0 pp=10).
    const u8 program[] = {
        0xc5, 0xf2, 0x10, 0xc2, // vmovss xmm0, xmm1, xmm2
        0xc3,
    };
    std::memcpy(mem.CodePtr(), program, sizeof(program));
    u8* guest_rsp = mem.StackTop() - 8;
    *reinterpret_cast<u64*>(guest_rsp) = kReturnSentinel;
    GuestState st{};
    st.rip = reinterpret_cast<u64>(mem.CodePtr());
    st.gpr[4] = reinterpret_cast<u64>(guest_rsp);
    // xmm1 (lane 1): low32=AAAA, hi32=BBBB; chunk1 = CCCC...
    st.ymm[4] = 0xBBBBBBBBAAAAAAAAULL;
    st.ymm[5] = 0xCCCCCCCCCCCCCCCCULL;
    // xmm2 (lane 2): low32=22222222, hi32=33333333
    st.ymm[8] = 0x3333333322222222ULL;
    st.ymm[9] = 0x4444444444444444ULL; // ignored — only src2 low32 used
    // Pre-pollute xmm0
    st.ymm[0] = 0xDEAD000000000000ULL;
    st.ymm[1] = 0xDEAD000000000000ULL;
    st.ymm[2] = 0xDEAD000000000000ULL;
    st.ymm[3] = 0xDEAD000000000000ULL;

    Runtime rt;
    rt.Run(st);
    // chunk 0: low32 = src2.low32 = 0x22222222, hi32 = src1.hi32 = 0xBBBBBBBB
    EXPECT_EQ(st.ymm[0], 0xBBBBBBBB22222222ULL);
    // chunk 1: full copy of src1.chunk1
    EXPECT_EQ(st.ymm[1], 0xCCCCCCCCCCCCCCCCULL);
    // upper YMM zeroed
    EXPECT_EQ(st.ymm[2], 0ULL);
    EXPECT_EQ(st.ymm[3], 0ULL);
}

// ============================================================================
// VCVTSI2SS — convert scalar integer (32 or 64-bit) to scalar single-
// precision float. First emitter that requires real host FP arithmetic
// (vcvtsi2ss on host xmm0). MXCSR comes from host (round-to-nearest by
// default) — guest MXCSR sync is a separate work item.
// ============================================================================

// Convert int32 = 42 to float. IEEE-754 float(42) = 0x42280000.
TEST_F(CpuRuntimeTest, Vcvtsi2ss_Int32ToFloat_BasicValue) {
    const u8 program[] = {
        0x48, 0xc7, 0xc0, 0x2a, 0x00, 0x00, 0x00,    // mov rax, 42
        0xc5, 0xf2, 0x2a, 0xc8,                       // vcvtsi2ss xmm1, xmm1, eax
        0xc3,
    };
    std::memcpy(mem.CodePtr(), program, sizeof(program));
    u8* guest_rsp = mem.StackTop() - 8;
    *reinterpret_cast<u64*>(guest_rsp) = kReturnSentinel;
    GuestState st{};
    st.rip = reinterpret_cast<u64>(mem.CodePtr());
    st.gpr[4] = reinterpret_cast<u64>(guest_rsp);
    // xmm1 = lane 1; preserve upper bits via pre-pollution. The
    // emitter overwrites the upper 96 of xmm1 from xmm1 itself (src1
    // == dst here) — so what was already in chunks 0[63:32] and 1
    // should survive.
    st.ymm[4] = 0xAAAAAAAA00000000ULL; // chunk 0: low32 will be overwritten; hi32 preserved
    st.ymm[5] = 0xBBBBBBBBBBBBBBBBULL; // chunk 1: preserved
    st.ymm[6] = 0xDEADDEADDEADDEADULL; // chunk 2: must be zeroed (VEX-128)
    st.ymm[7] = 0xDEADDEADDEADDEADULL; // chunk 3: must be zeroed

    Runtime rt;
    rt.Run(st);

    // chunk 0 low32 = float(42), hi32 = preserved 0xAAAAAAAA
    const u32 low32 = static_cast<u32>(st.ymm[4] & 0xFFFFFFFFULL);
    const u32 hi32  = static_cast<u32>(st.ymm[4] >> 32);
    float as_float;
    std::memcpy(&as_float, &low32, sizeof(as_float));
    EXPECT_EQ(as_float, 42.0f);
    EXPECT_EQ(hi32, 0xAAAAAAAAu)
        << "src1[63:32] must be preserved (dst==src1 here)";
    EXPECT_EQ(st.ymm[5], 0xBBBBBBBBBBBBBBBBULL) << "chunk 1 preserved";
    EXPECT_EQ(st.ymm[6], 0ULL) << "VEX-128 zeroes upper YMM";
    EXPECT_EQ(st.ymm[7], 0ULL);
}

// Convert negative int32 to float. (-100).f = 0xC2C80000.
TEST_F(CpuRuntimeTest, Vcvtsi2ss_NegativeInt32) {
    const u8 program[] = {
        // mov rax, -100  (encoded as imm32 sign-extended)
        0x48, 0xc7, 0xc0, 0x9c, 0xff, 0xff, 0xff,
        0xc5, 0xf2, 0x2a, 0xc8,                       // vcvtsi2ss xmm1, xmm1, eax
        0xc3,
    };
    std::memcpy(mem.CodePtr(), program, sizeof(program));
    u8* guest_rsp = mem.StackTop() - 8;
    *reinterpret_cast<u64*>(guest_rsp) = kReturnSentinel;
    GuestState st{};
    st.rip = reinterpret_cast<u64>(mem.CodePtr());
    st.gpr[4] = reinterpret_cast<u64>(guest_rsp);
    Runtime rt;
    rt.Run(st);
    const u32 low32 = static_cast<u32>(st.ymm[4] & 0xFFFFFFFFULL);
    float as_float;
    std::memcpy(&as_float, &low32, sizeof(as_float));
    EXPECT_EQ(as_float, -100.0f);
}

// Three-operand form with distinct src1: `vcvtsi2ss xmm0, xmm1, eax`.
// Verifies that the upper 96 of dst comes from src1 (NOT from dst's
// pre-existing value).
TEST_F(CpuRuntimeTest, Vcvtsi2ss_ThreeOperand_UpperFromSrc1) {
    const u8 program[] = {
        0x48, 0xc7, 0xc0, 0x07, 0x00, 0x00, 0x00,    // mov rax, 7
        0xc5, 0xf2, 0x2a, 0xc0,                       // vcvtsi2ss xmm0, xmm1, eax
        0xc3,
    };
    std::memcpy(mem.CodePtr(), program, sizeof(program));
    u8* guest_rsp = mem.StackTop() - 8;
    *reinterpret_cast<u64*>(guest_rsp) = kReturnSentinel;
    GuestState st{};
    st.rip = reinterpret_cast<u64>(mem.CodePtr());
    st.gpr[4] = reinterpret_cast<u64>(guest_rsp);
    // xmm0 (dst) = pre-pollution that must NOT survive (upper is from src1).
    st.ymm[0] = 0xCCCCCCCCCCCCCCCCULL;
    st.ymm[1] = 0xCCCCCCCCCCCCCCCCULL;
    // xmm1 (src1) = distinct upper bits we expect to see in dst.
    st.ymm[4] = 0x77777777EEEEEEEEULL; // chunk 0: low32 ignored, hi32 → dst chunk 0 hi32
    st.ymm[5] = 0x8888888888888888ULL; // chunk 1: → dst chunk 1

    Runtime rt;
    rt.Run(st);
    const u32 low32 = static_cast<u32>(st.ymm[0] & 0xFFFFFFFFULL);
    const u32 hi32  = static_cast<u32>(st.ymm[0] >> 32);
    float as_float;
    std::memcpy(&as_float, &low32, sizeof(as_float));
    EXPECT_EQ(as_float, 7.0f);
    EXPECT_EQ(hi32, 0x77777777u)
        << "dst[63:32] must come from src1, not from pre-existing dst";
    EXPECT_EQ(st.ymm[1], 0x8888888888888888ULL) << "dst chunk 1 from src1";
    EXPECT_EQ(st.ymm[2], 0ULL);
    EXPECT_EQ(st.ymm[3], 0ULL);
}

// ============================================================================
// VCVTSI2SD — convert scalar integer (32 or 64-bit) to scalar double-
// precision float. The double-precision sibling of VCVTSI2SS; gap from
// CUSA02394 at guest 0x8000862f4 (32-bit integer source). Unlike SS, the
// result fills the entire low 64 bits of dst (chunk 0); dst[127:64] comes
// from src1 (chunk 1); VEX-128 zeroes the upper YMM. Host vcvtsi2sd on
// scratch xmm0; MXCSR from host (round-to-nearest).
// ============================================================================

// Convert int32 42 -> double 42.0 (0x4045000000000000). dst==src1 form.
TEST_F(CpuRuntimeTest, Vcvtsi2sd_Int32ToDouble_BasicValue) {
    const u8 program[] = {
        0x48, 0xc7, 0xc0, 0x2a, 0x00, 0x00, 0x00,    // mov rax, 42
        0xc5, 0xf3, 0x2a, 0xc8,                       // vcvtsi2sd xmm1, xmm1, eax
        0xc3,
    };
    std::memcpy(mem.CodePtr(), program, sizeof(program));
    u8* guest_rsp = mem.StackTop() - 8;
    *reinterpret_cast<u64*>(guest_rsp) = kReturnSentinel;
    GuestState st{};
    st.rip = reinterpret_cast<u64>(mem.CodePtr());
    st.gpr[4] = reinterpret_cast<u64>(guest_rsp);
    st.ymm[4] = 0xAAAAAAAAAAAAAAAAULL; // chunk 0: fully overwritten by the double
    st.ymm[5] = 0xBBBBBBBBBBBBBBBBULL; // chunk 1: preserved (dst==src1)
    st.ymm[6] = 0xDEADDEADDEADDEADULL; // chunk 2: must be zeroed
    st.ymm[7] = 0xDEADDEADDEADDEADULL; // chunk 3: must be zeroed

    Runtime rt;
    rt.Run(st);
    double as_double;
    std::memcpy(&as_double, &st.ymm[4], sizeof(as_double));
    EXPECT_EQ(as_double, 42.0) << "low 64 = (double)42";
    EXPECT_EQ(st.ymm[4], 0x4045000000000000ULL) << "exact double bit pattern";
    EXPECT_EQ(st.ymm[5], 0xBBBBBBBBBBBBBBBBULL) << "chunk 1 preserved (dst==src1)";
    EXPECT_EQ(st.ymm[6], 0ULL) << "VEX-128 zeroes upper YMM";
    EXPECT_EQ(st.ymm[7], 0ULL);
}

// Convert negative int32 -100 -> -100.0 (0xC059000000000000).
TEST_F(CpuRuntimeTest, Vcvtsi2sd_NegativeInt32) {
    const u8 program[] = {
        0x48, 0xc7, 0xc0, 0x9c, 0xff, 0xff, 0xff,    // mov rax, -100
        0xc5, 0xf3, 0x2a, 0xc8,                       // vcvtsi2sd xmm1, xmm1, eax
        0xc3,
    };
    std::memcpy(mem.CodePtr(), program, sizeof(program));
    u8* guest_rsp = mem.StackTop() - 8;
    *reinterpret_cast<u64*>(guest_rsp) = kReturnSentinel;
    GuestState st{};
    st.rip = reinterpret_cast<u64>(mem.CodePtr());
    st.gpr[4] = reinterpret_cast<u64>(guest_rsp);
    Runtime rt;
    rt.Run(st);
    double as_double;
    std::memcpy(&as_double, &st.ymm[4], sizeof(as_double));
    EXPECT_EQ(as_double, -100.0);
    EXPECT_EQ(st.ymm[4], 0xC059000000000000ULL);
}

// Three-operand form with distinct src1: vcvtsi2sd xmm0, xmm1, eax.
// dst[127:64] must come from src1 (chunk 1), not pre-existing dst.
TEST_F(CpuRuntimeTest, Vcvtsi2sd_ThreeOperand_UpperFromSrc1) {
    const u8 program[] = {
        0x48, 0xc7, 0xc0, 0x07, 0x00, 0x00, 0x00,    // mov rax, 7
        0xc5, 0xf3, 0x2a, 0xc0,                       // vcvtsi2sd xmm0, xmm1, eax
        0xc3,
    };
    std::memcpy(mem.CodePtr(), program, sizeof(program));
    u8* guest_rsp = mem.StackTop() - 8;
    *reinterpret_cast<u64*>(guest_rsp) = kReturnSentinel;
    GuestState st{};
    st.rip = reinterpret_cast<u64>(mem.CodePtr());
    st.gpr[4] = reinterpret_cast<u64>(guest_rsp);
    st.ymm[0] = 0xCCCCCCCCCCCCCCCCULL; // dst pre-pollution, must not survive
    st.ymm[1] = 0xCCCCCCCCCCCCCCCCULL;
    st.ymm[4] = 0x7777777755555555ULL; // src1 chunk 0: ignored (overwritten by double)
    st.ymm[5] = 0x8888888888888888ULL; // src1 chunk 1: -> dst chunk 1

    Runtime rt;
    rt.Run(st);
    double as_double;
    std::memcpy(&as_double, &st.ymm[0], sizeof(as_double));
    EXPECT_EQ(as_double, 7.0);
    EXPECT_EQ(st.ymm[0], 0x401C000000000000ULL) << "low 64 = (double)7";
    EXPECT_EQ(st.ymm[1], 0x8888888888888888ULL) << "dst chunk 1 from src1, not pre-existing dst";
    EXPECT_EQ(st.ymm[2], 0ULL);
    EXPECT_EQ(st.ymm[3], 0ULL);
}

// 64-bit integer source (REX.W): convert int64 1000000 -> 1000000.0.
TEST_F(CpuRuntimeTest, Vcvtsi2sd_Int64ToDouble) {
    const u8 program[] = {
        0x48, 0xc7, 0xc0, 0x40, 0x42, 0x0f, 0x00,    // mov rax, 1000000
        0xc4, 0xe1, 0xf3, 0x2a, 0xc8,                 // vcvtsi2sd xmm1, xmm1, rax (REX.W)
        0xc3,
    };
    std::memcpy(mem.CodePtr(), program, sizeof(program));
    u8* guest_rsp = mem.StackTop() - 8;
    *reinterpret_cast<u64*>(guest_rsp) = kReturnSentinel;
    GuestState st{};
    st.rip = reinterpret_cast<u64>(mem.CodePtr());
    st.gpr[4] = reinterpret_cast<u64>(guest_rsp);
    Runtime rt;
    rt.Run(st);
    double as_double;
    std::memcpy(&as_double, &st.ymm[4], sizeof(as_double));
    EXPECT_EQ(as_double, 1000000.0);
    EXPECT_EQ(st.ymm[4], 0x412E848000000000ULL);
}

// ============================================================================
// VMULSS — scalar single-precision multiply. Real host FP arithmetic
// (using JIT-scratch xmm0/xmm1). The architectural "preserve src1[127:32]
// into dst" is satisfied by host VMULSS's own merge semantics — loading
// src1's full xmm into xmm0 leaves the upper 96 alone through the multiply.
// ============================================================================

// 3.0 * 4.0 = 12.0. Reg-reg-reg form.
TEST_F(CpuRuntimeTest, Vmulss_BasicMultiply) {
    // vmulss xmm0, xmm1, xmm2  — encoding c5 f2 59 c2
    const u8 program[] = {
        0xc5, 0xf2, 0x59, 0xc2,
        0xc3,
    };
    std::memcpy(mem.CodePtr(), program, sizeof(program));
    u8* guest_rsp = mem.StackTop() - 8;
    *reinterpret_cast<u64*>(guest_rsp) = kReturnSentinel;
    GuestState st{};
    st.rip = reinterpret_cast<u64>(mem.CodePtr());
    st.gpr[4] = reinterpret_cast<u64>(guest_rsp);
    // xmm1.low32 = float(3.0)
    const u32 f3 = std::bit_cast<u32>(3.0f);
    // xmm2.low32 = float(4.0)
    const u32 f4 = std::bit_cast<u32>(4.0f);
    st.ymm[4] = static_cast<u64>(f3); // xmm1, lane 1
    st.ymm[8] = static_cast<u64>(f4); // xmm2, lane 2

    Runtime rt;
    rt.Run(st);

    const u32 low32 = static_cast<u32>(st.ymm[0] & 0xFFFFFFFFULL);
    float result;
    std::memcpy(&result, &low32, sizeof(result));
    EXPECT_EQ(result, 12.0f) << "3.0 * 4.0 = 12.0";
}

// `vmulss xmm0, xmm1, xmm2` with src1's upper 96 set to distinguishing
// bits. The architectural rule: dst[127:32] = src1[127:32]. Verifies that
// our "load full src1 into xmm0 → vmulss → 128-bit storeback" pattern
// preserves the merge correctly.
TEST_F(CpuRuntimeTest, Vmulss_PreservesSrc1Upper) {
    const u8 program[] = {
        0xc5, 0xf2, 0x59, 0xc2, // vmulss xmm0, xmm1, xmm2
        0xc3,
    };
    std::memcpy(mem.CodePtr(), program, sizeof(program));
    u8* guest_rsp = mem.StackTop() - 8;
    *reinterpret_cast<u64*>(guest_rsp) = kReturnSentinel;
    GuestState st{};
    st.rip = reinterpret_cast<u64>(mem.CodePtr());
    st.gpr[4] = reinterpret_cast<u64>(guest_rsp);

    const u32 f2 = std::bit_cast<u32>(2.0f);
    const u32 f5 = std::bit_cast<u32>(5.0f);
    // xmm1.low32 = 2.0, xmm1[63:32] = 0xDEADBEEF, xmm1[127:64] = 0xCAFEBABE5555AAAA
    st.ymm[4] = (static_cast<u64>(0xDEADBEEFULL) << 32) | f2;
    st.ymm[5] = 0xCAFEBABE5555AAAAULL;
    // Pre-pollute dst beyond what the emitter should touch.
    st.ymm[0] = 0xFFFFFFFFFFFFFFFFULL;
    st.ymm[1] = 0xFFFFFFFFFFFFFFFFULL;
    st.ymm[2] = 0xFFFFFFFFFFFFFFFFULL;
    st.ymm[3] = 0xFFFFFFFFFFFFFFFFULL;
    // xmm2.low32 = 5.0
    st.ymm[8] = static_cast<u64>(f5);

    Runtime rt;
    rt.Run(st);

    const u32 low32 = static_cast<u32>(st.ymm[0] & 0xFFFFFFFFULL);
    const u32 hi32  = static_cast<u32>(st.ymm[0] >> 32);
    float result;
    std::memcpy(&result, &low32, sizeof(result));
    EXPECT_EQ(result, 10.0f);
    EXPECT_EQ(hi32, 0xDEADBEEFu) << "src1[63:32] preserved";
    EXPECT_EQ(st.ymm[1], 0xCAFEBABE5555AAAAULL) << "src1[127:64] preserved";
    EXPECT_EQ(st.ymm[2], 0ULL) << "VEX-128 zeroes upper YMM";
    EXPECT_EQ(st.ymm[3], 0ULL);
}

// Memory-source form: `vmulss xmm0, xmm1, dword[rax]`. Exercises the
// EmitEffectiveAddress path for ops[2] = memory.
TEST_F(CpuRuntimeTest, Vmulss_MemorySource) {
    // Place the float-from-memory operand at an address we can take.
    u32* fmem = reinterpret_cast<u32*>(mem.CodePtr() + 0x100);
    *fmem = std::bit_cast<u32>(6.0f);

    const u8 program[] = {
        // mov rax, <fmem addr>
        0x48, 0xb8, 0,0,0,0,0,0,0,0,
        // vmulss xmm0, xmm1, dword[rax]   (c5 f2 59 00)
        0xc5, 0xf2, 0x59, 0x00,
        0xc3,
    };
    u8 prog[sizeof(program)];
    std::memcpy(prog, program, sizeof(program));
    const u64 fmem_addr = reinterpret_cast<u64>(fmem);
    std::memcpy(prog + 2, &fmem_addr, sizeof(fmem_addr));

    std::memcpy(mem.CodePtr(), prog, sizeof(prog));
    u8* guest_rsp = mem.StackTop() - 8;
    *reinterpret_cast<u64*>(guest_rsp) = kReturnSentinel;
    GuestState st{};
    st.rip = reinterpret_cast<u64>(mem.CodePtr());
    st.gpr[4] = reinterpret_cast<u64>(guest_rsp);
    // xmm1.low32 = 7.0
    st.ymm[4] = static_cast<u64>(std::bit_cast<u32>(7.0f));

    Runtime rt;
    rt.Run(st);

    const u32 low32 = static_cast<u32>(st.ymm[0] & 0xFFFFFFFFULL);
    float result;
    std::memcpy(&result, &low32, sizeof(result));
    EXPECT_EQ(result, 42.0f) << "7.0 * 6.0 = 42.0";
}

// ============================================================================
// VCVTTSS2SI — convert scalar float to signed integer with truncation
// toward zero (not MXCSR-rounded). The inverse of VCVTSI2SS. Out-of-
// range and NaN both produce the "indefinite integer value" (INT_MIN).
// ============================================================================

// 3.7f truncates to 3 (toward zero, not nearest-even rounding).
TEST_F(CpuRuntimeTest, Vcvttss2si_PositiveTruncatesTowardZero) {
    const u8 program[] = {
        0xc5, 0xfa, 0x2c, 0xc1, // vcvttss2si eax, xmm1
        0xc3,
    };
    std::memcpy(mem.CodePtr(), program, sizeof(program));
    u8* guest_rsp = mem.StackTop() - 8;
    *reinterpret_cast<u64*>(guest_rsp) = kReturnSentinel;
    GuestState st{};
    st.rip = reinterpret_cast<u64>(mem.CodePtr());
    st.gpr[4] = reinterpret_cast<u64>(guest_rsp);
    // xmm1 (lane 1) low32 = float(3.7)
    st.ymm[4] = static_cast<u64>(std::bit_cast<u32>(3.7f));
    // Pre-pollute guest RAX upper 32 to confirm zero-extension.
    st.gpr[0] = 0xDEADBEEF00000000ULL;

    Runtime rt;
    rt.Run(st);
    EXPECT_EQ(st.gpr[0], 3ULL) << "3.7f truncates to 3, upper 32 zeroed";
}

// Negative truncates toward zero too: -3.7f → -3, not -4.
TEST_F(CpuRuntimeTest, Vcvttss2si_NegativeTruncatesTowardZero) {
    const u8 program[] = {
        0xc5, 0xfa, 0x2c, 0xc1, // vcvttss2si eax, xmm1
        0xc3,
    };
    std::memcpy(mem.CodePtr(), program, sizeof(program));
    u8* guest_rsp = mem.StackTop() - 8;
    *reinterpret_cast<u64*>(guest_rsp) = kReturnSentinel;
    GuestState st{};
    st.rip = reinterpret_cast<u64>(mem.CodePtr());
    st.gpr[4] = reinterpret_cast<u64>(guest_rsp);
    st.ymm[4] = static_cast<u64>(std::bit_cast<u32>(-3.7f));
    Runtime rt;
    rt.Run(st);
    // -3 as int32 = 0xFFFFFFFD; zero-extended to 64-bit = 0xFFFFFFFD
    EXPECT_EQ(st.gpr[0], 0xFFFFFFFDULL)
        << "-3.7f truncates to -3 (sign-bit pattern in low 32, "
           "upper 32 zero per x86-64 32-bit-write rule)";
}

// NaN converts to the "indefinite integer value" — INT32_MIN
// (0x80000000) for the 32-bit form.
TEST_F(CpuRuntimeTest, Vcvttss2si_NanProducesIntMin) {
    const u8 program[] = {
        0xc5, 0xfa, 0x2c, 0xc1, // vcvttss2si eax, xmm1
        0xc3,
    };
    std::memcpy(mem.CodePtr(), program, sizeof(program));
    u8* guest_rsp = mem.StackTop() - 8;
    *reinterpret_cast<u64*>(guest_rsp) = kReturnSentinel;
    GuestState st{};
    st.rip = reinterpret_cast<u64>(mem.CodePtr());
    st.gpr[4] = reinterpret_cast<u64>(guest_rsp);
    // Quiet NaN bit pattern (exponent all-ones, mantissa MSB set).
    st.ymm[4] = static_cast<u64>(0x7FC00000u);
    Runtime rt;
    rt.Run(st);
    EXPECT_EQ(st.gpr[0], 0x80000000ULL)
        << "NaN → indefinite integer = INT32_MIN, upper 32 zero-extended";
}

// ============================================================================
// VCVTTSD2SI — convert scalar double to signed integer with truncation. The
// double-precision sibling of VCVTTSS2SI; gap from CUSA02394 at guest
// 0x800200e32 (64-bit dst, right after the VROUNDSD at 0x800200e2c). Out-of-
// range and NaN produce the indefinite integer value (INT_MIN). Truncation
// is hard-coded — MXCSR rounding mode is ignored.
// ============================================================================

// 3.9 truncates to 3 (toward zero). 32-bit dst, upper-32 zero-extended.
TEST_F(CpuRuntimeTest, Vcvttsd2si_PositiveTruncatesTowardZero) {
    const u8 program[] = {
        0xc5, 0xfb, 0x2c, 0xc1, // vcvttsd2si eax, xmm1
        0xc3,
    };
    std::memcpy(mem.CodePtr(), program, sizeof(program));
    u8* guest_rsp = mem.StackTop() - 8;
    *reinterpret_cast<u64*>(guest_rsp) = kReturnSentinel;
    GuestState st{};
    st.rip = reinterpret_cast<u64>(mem.CodePtr());
    st.gpr[4] = reinterpret_cast<u64>(guest_rsp);
    st.ymm[4] = std::bit_cast<u64>(3.9); // xmm1 low64 = double(3.9)
    st.gpr[0] = 0xDEADBEEF00000000ULL;   // pre-pollute upper 32
    Runtime rt;
    rt.Run(st);
    EXPECT_EQ(st.gpr[0], 3ULL) << "3.9 truncates to 3, upper 32 zeroed";
}

// Negative truncates toward zero: -3.9 → -3, not -4.
TEST_F(CpuRuntimeTest, Vcvttsd2si_NegativeTruncatesTowardZero) {
    const u8 program[] = {
        0xc5, 0xfb, 0x2c, 0xc1, // vcvttsd2si eax, xmm1
        0xc3,
    };
    std::memcpy(mem.CodePtr(), program, sizeof(program));
    u8* guest_rsp = mem.StackTop() - 8;
    *reinterpret_cast<u64*>(guest_rsp) = kReturnSentinel;
    GuestState st{};
    st.rip = reinterpret_cast<u64>(mem.CodePtr());
    st.gpr[4] = reinterpret_cast<u64>(guest_rsp);
    st.ymm[4] = std::bit_cast<u64>(-3.9);
    Runtime rt;
    rt.Run(st);
    EXPECT_EQ(st.gpr[0], 0xFFFFFFFDULL)
        << "-3.9 truncates to -3 (low 32 = -3, upper 32 zero per 32-bit-write rule)";
}

// 64-bit destination — the gap's actual width (c4 e1 fb 2c c1, length 5).
// A value beyond INT32 range that fits in INT64: 2147483648.0 = 2^31.
TEST_F(CpuRuntimeTest, Vcvttsd2si_Int64Destination) {
    const u8 program[] = {
        0xc4, 0xe1, 0xfb, 0x2c, 0xc1, // vcvttsd2si rax, xmm1 (REX.W)
        0xc3,
    };
    std::memcpy(mem.CodePtr(), program, sizeof(program));
    u8* guest_rsp = mem.StackTop() - 8;
    *reinterpret_cast<u64*>(guest_rsp) = kReturnSentinel;
    GuestState st{};
    st.rip = reinterpret_cast<u64>(mem.CodePtr());
    st.gpr[4] = reinterpret_cast<u64>(guest_rsp);
    st.ymm[4] = std::bit_cast<u64>(2147483648.0); // 2^31, overflows int32 but fits int64
    Runtime rt;
    rt.Run(st);
    EXPECT_EQ(st.gpr[0], 2147483648ULL) << "2^31 converts exactly in 64-bit dst";
}

// NaN → indefinite integer. 64-bit form gives INT64_MIN.
TEST_F(CpuRuntimeTest, Vcvttsd2si_NanProducesInt64Min) {
    const u8 program[] = {
        0xc4, 0xe1, 0xfb, 0x2c, 0xc1, // vcvttsd2si rax, xmm1
        0xc3,
    };
    std::memcpy(mem.CodePtr(), program, sizeof(program));
    u8* guest_rsp = mem.StackTop() - 8;
    *reinterpret_cast<u64*>(guest_rsp) = kReturnSentinel;
    GuestState st{};
    st.rip = reinterpret_cast<u64>(mem.CodePtr());
    st.gpr[4] = reinterpret_cast<u64>(guest_rsp);
    st.ymm[4] = 0x7FF8000000000000ULL; // quiet NaN (double)
    Runtime rt;
    rt.Run(st);
    EXPECT_EQ(st.gpr[0], 0x8000000000000000ULL)
        << "NaN → indefinite integer = INT64_MIN in 64-bit form";
}

// ============================================================================
// VDIVSS — scalar single-precision divide. Structurally identical to
// VMULSS (shares EmitScalarFpSs); separate tests just confirm the
// dispatch reaches the Div case correctly and the result is right.
// ============================================================================

// 12.0 / 4.0 = 3.0
TEST_F(CpuRuntimeTest, Vdivss_BasicDivide) {
    // vdivss xmm0, xmm1, xmm2  — encoding c5 f2 5e c2 (opcode 0x5E vs MUL's 0x59)
    const u8 program[] = {
        0xc5, 0xf2, 0x5e, 0xc2,
        0xc3,
    };
    std::memcpy(mem.CodePtr(), program, sizeof(program));
    u8* guest_rsp = mem.StackTop() - 8;
    *reinterpret_cast<u64*>(guest_rsp) = kReturnSentinel;
    GuestState st{};
    st.rip = reinterpret_cast<u64>(mem.CodePtr());
    st.gpr[4] = reinterpret_cast<u64>(guest_rsp);
    st.ymm[4] = static_cast<u64>(std::bit_cast<u32>(12.0f)); // xmm1
    st.ymm[8] = static_cast<u64>(std::bit_cast<u32>(4.0f));  // xmm2
    Runtime rt;
    rt.Run(st);
    const u32 low32 = static_cast<u32>(st.ymm[0] & 0xFFFFFFFFULL);
    float result;
    std::memcpy(&result, &low32, sizeof(result));
    EXPECT_EQ(result, 3.0f) << "12.0 / 4.0 = 3.0";
}

// Preserves src1[127:32]. Same property as VMULSS — both go through
// EmitScalarFpSs. Distinguishes a regression that breaks the merge
// from a regression that just breaks the arithmetic dispatch.
TEST_F(CpuRuntimeTest, Vdivss_PreservesSrc1Upper) {
    const u8 program[] = {
        0xc5, 0xf2, 0x5e, 0xc2, // vdivss xmm0, xmm1, xmm2
        0xc3,
    };
    std::memcpy(mem.CodePtr(), program, sizeof(program));
    u8* guest_rsp = mem.StackTop() - 8;
    *reinterpret_cast<u64*>(guest_rsp) = kReturnSentinel;
    GuestState st{};
    st.rip = reinterpret_cast<u64>(mem.CodePtr());
    st.gpr[4] = reinterpret_cast<u64>(guest_rsp);
    st.ymm[4] = (static_cast<u64>(0x11223344ULL) << 32)
              | std::bit_cast<u32>(20.0f);
    st.ymm[5] = 0x5566778899AABBCCULL;
    st.ymm[8] = static_cast<u64>(std::bit_cast<u32>(5.0f));
    st.ymm[0] = 0xFFFFFFFFFFFFFFFFULL;
    st.ymm[1] = 0xFFFFFFFFFFFFFFFFULL;
    st.ymm[2] = 0xFFFFFFFFFFFFFFFFULL;
    st.ymm[3] = 0xFFFFFFFFFFFFFFFFULL;

    Runtime rt;
    rt.Run(st);
    const u32 low32 = static_cast<u32>(st.ymm[0] & 0xFFFFFFFFULL);
    const u32 hi32  = static_cast<u32>(st.ymm[0] >> 32);
    float result;
    std::memcpy(&result, &low32, sizeof(result));
    EXPECT_EQ(result, 4.0f) << "20.0 / 5.0 = 4.0";
    EXPECT_EQ(hi32, 0x11223344u) << "src1[63:32] preserved";
    EXPECT_EQ(st.ymm[1], 0x5566778899AABBCCULL) << "src1[127:64] preserved";
    EXPECT_EQ(st.ymm[2], 0ULL);
    EXPECT_EQ(st.ymm[3], 0ULL);
}

// Memory-source form (the 8-byte length observed in the game's log).
TEST_F(CpuRuntimeTest, Vdivss_MemorySource) {
    u32* fmem = reinterpret_cast<u32*>(mem.CodePtr() + 0x100);
    *fmem = std::bit_cast<u32>(2.0f);
    const u8 program[] = {
        0x48, 0xb8, 0,0,0,0,0,0,0,0,    // mov rax, <fmem>
        0xc5, 0xf2, 0x5e, 0x00,          // vdivss xmm0, xmm1, dword[rax]
        0xc3,
    };
    u8 prog[sizeof(program)];
    std::memcpy(prog, program, sizeof(program));
    const u64 fmem_addr = reinterpret_cast<u64>(fmem);
    std::memcpy(prog + 2, &fmem_addr, sizeof(fmem_addr));
    std::memcpy(mem.CodePtr(), prog, sizeof(prog));
    u8* guest_rsp = mem.StackTop() - 8;
    *reinterpret_cast<u64*>(guest_rsp) = kReturnSentinel;
    GuestState st{};
    st.rip = reinterpret_cast<u64>(mem.CodePtr());
    st.gpr[4] = reinterpret_cast<u64>(guest_rsp);
    st.ymm[4] = static_cast<u64>(std::bit_cast<u32>(10.0f));
    Runtime rt;
    rt.Run(st);
    const u32 low32 = static_cast<u32>(st.ymm[0] & 0xFFFFFFFFULL);
    float result;
    std::memcpy(&result, &low32, sizeof(result));
    EXPECT_EQ(result, 5.0f) << "10.0 / 2.0 = 5.0";
}

// ============================================================================
// VADDSS — scalar single-precision add. Same EmitScalarFpSs scaffolding
// as VMULSS/VDIVSS; tests confirm the dispatch reaches the Add case and
// that addition results are correct.
// ============================================================================

// 1.5 + 2.25 = 3.75 — clean binary fractions, no rounding ambiguity.
TEST_F(CpuRuntimeTest, Vaddss_BasicAdd) {
    // vaddss xmm0, xmm1, xmm2  — encoding c5 f2 58 c2
    const u8 program[] = {
        0xc5, 0xf2, 0x58, 0xc2,
        0xc3,
    };
    std::memcpy(mem.CodePtr(), program, sizeof(program));
    u8* guest_rsp = mem.StackTop() - 8;
    *reinterpret_cast<u64*>(guest_rsp) = kReturnSentinel;
    GuestState st{};
    st.rip = reinterpret_cast<u64>(mem.CodePtr());
    st.gpr[4] = reinterpret_cast<u64>(guest_rsp);
    st.ymm[4] = static_cast<u64>(std::bit_cast<u32>(1.5f));   // xmm1
    st.ymm[8] = static_cast<u64>(std::bit_cast<u32>(2.25f));  // xmm2
    Runtime rt;
    rt.Run(st);
    const u32 low32 = static_cast<u32>(st.ymm[0] & 0xFFFFFFFFULL);
    float result;
    std::memcpy(&result, &low32, sizeof(result));
    EXPECT_EQ(result, 3.75f) << "1.5 + 2.25 = 3.75";
}

// Memory-source form (matches the 8-byte length seen in the game's log).
TEST_F(CpuRuntimeTest, Vaddss_MemorySource) {
    u32* fmem = reinterpret_cast<u32*>(mem.CodePtr() + 0x100);
    *fmem = std::bit_cast<u32>(0.5f);
    const u8 program[] = {
        0x48, 0xb8, 0,0,0,0,0,0,0,0,
        0xc5, 0xf2, 0x58, 0x00,  // vaddss xmm0, xmm1, dword[rax]
        0xc3,
    };
    u8 prog[sizeof(program)];
    std::memcpy(prog, program, sizeof(program));
    const u64 fmem_addr = reinterpret_cast<u64>(fmem);
    std::memcpy(prog + 2, &fmem_addr, sizeof(fmem_addr));
    std::memcpy(mem.CodePtr(), prog, sizeof(prog));
    u8* guest_rsp = mem.StackTop() - 8;
    *reinterpret_cast<u64*>(guest_rsp) = kReturnSentinel;
    GuestState st{};
    st.rip = reinterpret_cast<u64>(mem.CodePtr());
    st.gpr[4] = reinterpret_cast<u64>(guest_rsp);
    st.ymm[4] = static_cast<u64>(std::bit_cast<u32>(100.0f));
    Runtime rt;
    rt.Run(st);
    const u32 low32 = static_cast<u32>(st.ymm[0] & 0xFFFFFFFFULL);
    float result;
    std::memcpy(&result, &low32, sizeof(result));
    EXPECT_EQ(result, 100.5f) << "100.0 + 0.5 = 100.5";
}

// ============================================================================
// VUCOMISS — Unordered Compare Scalar Single-precision, sets EFLAGS.
// First scalar-FP flag-writing emitter. Truth table:
//   unordered (NaN) → ZF=1 PF=1 CF=1
//   src1 == src2    → ZF=1 PF=0 CF=0
//   src1 <  src2    → ZF=0 PF=0 CF=1
//   src1 >  src2    → ZF=0 PF=0 CF=0
// In all cases OF/SF/AF are cleared.
// ============================================================================

// Common helper: run vucomiss xmm0, xmm1 and return the guest rflags.
namespace {
u64 RunVucomiss_Xmm0_Xmm1(GuestMemory& mem, float a, float b) {
    // vucomiss xmm0, xmm1 — c5 f8 2e c1
    const u8 program[] = {
        0xc5, 0xf8, 0x2e, 0xc1,
        0xc3,
    };
    std::memcpy(mem.CodePtr(), program, sizeof(program));
    u8* guest_rsp = mem.StackTop() - 8;
    *reinterpret_cast<u64*>(guest_rsp) = kReturnSentinel;
    GuestState st{};
    st.rip = reinterpret_cast<u64>(mem.CodePtr());
    st.gpr[4] = reinterpret_cast<u64>(guest_rsp);
    // Pre-set non-VUCOMISS flag bits to confirm they're preserved.
    // Bit 9 (IF) is in ~kArithMask, so it should survive untouched.
    st.rflags = 0x202; // IF=1 + reserved bit 1
    st.ymm[0] = static_cast<u64>(std::bit_cast<u32>(a));
    st.ymm[4] = static_cast<u64>(std::bit_cast<u32>(b));
    Runtime rt;
    rt.Run(st);
    return st.rflags;
}
constexpr u64 CF_BIT = 1ULL << 0;
constexpr u64 PF_BIT = 1ULL << 2;
constexpr u64 AF_BIT = 1ULL << 4;
constexpr u64 ZF_BIT = 1ULL << 6;
constexpr u64 SF_BIT = 1ULL << 7;
constexpr u64 OF_BIT = 1ULL << 11;
constexpr u64 IF_BIT = 1ULL << 9;
} // namespace

TEST_F(CpuRuntimeTest, Vucomiss_Equal_SetsZF) {
    const u64 rf = RunVucomiss_Xmm0_Xmm1(mem, 1.0f, 1.0f);
    EXPECT_TRUE(rf & ZF_BIT);
    EXPECT_FALSE(rf & PF_BIT);
    EXPECT_FALSE(rf & CF_BIT);
    EXPECT_FALSE(rf & OF_BIT);
    EXPECT_FALSE(rf & SF_BIT);
    EXPECT_FALSE(rf & AF_BIT);
    EXPECT_TRUE(rf & IF_BIT) << "non-VUCOMISS flag bits preserved";
}

TEST_F(CpuRuntimeTest, Vucomiss_LessThan_SetsCF) {
    const u64 rf = RunVucomiss_Xmm0_Xmm1(mem, 1.0f, 2.0f);
    EXPECT_FALSE(rf & ZF_BIT);
    EXPECT_FALSE(rf & PF_BIT);
    EXPECT_TRUE(rf & CF_BIT);
}

TEST_F(CpuRuntimeTest, Vucomiss_GreaterThan_AllClear) {
    const u64 rf = RunVucomiss_Xmm0_Xmm1(mem, 3.0f, 2.0f);
    EXPECT_FALSE(rf & ZF_BIT);
    EXPECT_FALSE(rf & PF_BIT);
    EXPECT_FALSE(rf & CF_BIT);
}

TEST_F(CpuRuntimeTest, Vucomiss_NaN_SetsAllThree) {
    const float nan_val = std::bit_cast<float>(0x7FC00000u);
    const u64 rf = RunVucomiss_Xmm0_Xmm1(mem, nan_val, 1.0f);
    EXPECT_TRUE(rf & ZF_BIT) << "NaN→unordered: ZF=1";
    EXPECT_TRUE(rf & PF_BIT) << "NaN→unordered: PF=1";
    EXPECT_TRUE(rf & CF_BIT) << "NaN→unordered: CF=1";
}

// ============================================================================
// VUCOMISD — Unordered Compare Scalar Double-precision, sets EFLAGS. The
// double-precision sibling of VUCOMISS (same truth table, same flag merge);
// gap from CUSA02394 at guest 0x800122e60. Reuses the CF/PF/ZF/... bit
// constants defined in the VUCOMISS block above.
// ============================================================================

namespace {
u64 RunVucomisd_Xmm0_Xmm1(GuestMemory& mem, double a, double b) {
    // vucomisd xmm0, xmm1 — c5 f9 2e c1
    const u8 program[] = {
        0xc5, 0xf9, 0x2e, 0xc1,
        0xc3,
    };
    std::memcpy(mem.CodePtr(), program, sizeof(program));
    u8* guest_rsp = mem.StackTop() - 8;
    *reinterpret_cast<u64*>(guest_rsp) = kReturnSentinel;
    GuestState st{};
    st.rip = reinterpret_cast<u64>(mem.CodePtr());
    st.gpr[4] = reinterpret_cast<u64>(guest_rsp);
    st.rflags = 0x202; // IF=1 + reserved bit 1 (must survive)
    st.ymm[0] = std::bit_cast<u64>(a); // xmm0 low 64 = a
    st.ymm[4] = std::bit_cast<u64>(b); // xmm1 low 64 = b
    Runtime rt;
    rt.Run(st);
    return st.rflags;
}
} // namespace

TEST_F(CpuRuntimeTest, Vucomisd_Equal_SetsZF) {
    const u64 rf = RunVucomisd_Xmm0_Xmm1(mem, 1.0, 1.0);
    EXPECT_TRUE(rf & ZF_BIT);
    EXPECT_FALSE(rf & PF_BIT);
    EXPECT_FALSE(rf & CF_BIT);
    EXPECT_FALSE(rf & OF_BIT);
    EXPECT_FALSE(rf & SF_BIT);
    EXPECT_FALSE(rf & AF_BIT);
    EXPECT_TRUE(rf & IF_BIT) << "non-VUCOMISD flag bits preserved";
}

TEST_F(CpuRuntimeTest, Vucomisd_LessThan_SetsCF) {
    const u64 rf = RunVucomisd_Xmm0_Xmm1(mem, 1.0, 2.0);
    EXPECT_FALSE(rf & ZF_BIT);
    EXPECT_FALSE(rf & PF_BIT);
    EXPECT_TRUE(rf & CF_BIT);
}

TEST_F(CpuRuntimeTest, Vucomisd_GreaterThan_AllClear) {
    const u64 rf = RunVucomisd_Xmm0_Xmm1(mem, 3.0, 2.0);
    EXPECT_FALSE(rf & ZF_BIT);
    EXPECT_FALSE(rf & PF_BIT);
    EXPECT_FALSE(rf & CF_BIT);
}

TEST_F(CpuRuntimeTest, Vucomisd_NaN_SetsAllThree) {
    const double nan_val = std::bit_cast<double>(0x7FF8000000000000ULL);
    const u64 rf = RunVucomisd_Xmm0_Xmm1(mem, nan_val, 1.0);
    EXPECT_TRUE(rf & ZF_BIT) << "NaN→unordered: ZF=1";
    EXPECT_TRUE(rf & PF_BIT) << "NaN→unordered: PF=1";
    EXPECT_TRUE(rf & CF_BIT) << "NaN→unordered: CF=1";
}

// ============================================================================
// VROUNDSD — round scalar double to integer per imm8 rounding control. Gap
// from CUSA02394 at guest 0x800200e2c. dst.low64 = round(src2.low64, imm8);
// dst[127:64] preserved from src1; upper YMM zeroed. imm8: 0=nearest-even,
// 1=floor, 2=ceil, 3=truncate (bit2=use MXCSR). Host VROUNDSD interprets the
// imm verbatim, so all four modes are exact.
// ============================================================================

namespace {
// vroundsd xmm0, xmm1, xmm2, imm — c4 e3 71 0b c2 <imm>. src2 (xmm2) low double
// = v; returns dst (xmm0) low 64 bits. xmm1 supplies the preserved upper half.
u64 RunVroundsd_Mode(GuestMemory& mem, double v, u8 imm, u64 src1_hi) {
    const u8 program[] = {0xc4, 0xe3, 0x71, 0x0b, 0xc2, imm, 0xc3};
    std::memcpy(mem.CodePtr(), program, sizeof(program));
    u8* guest_rsp = mem.StackTop() - 8;
    *reinterpret_cast<u64*>(guest_rsp) = kReturnSentinel;
    GuestState st{};
    st.rip = reinterpret_cast<u64>(mem.CodePtr());
    st.gpr[4] = reinterpret_cast<u64>(guest_rsp);
    st.ymm[4] = 0; st.ymm[5] = src1_hi;            // xmm1 (src1): low ignored, hi preserved
    st.ymm[8] = std::bit_cast<u64>(v);             // xmm2 (src2): low double = v
    st.ymm[0] = 0xDEAD; st.ymm[1] = 0xBEEF;        // xmm0 (dst) pre-pollution
    st.ymm[2] = 0xF00D; st.ymm[3] = 0xCAFE;        // upper YMM0, must be zeroed
    Runtime rt;
    rt.Run(st);
    return st.ymm[0];
}
} // namespace

TEST_F(CpuRuntimeTest, Vroundsd_Floor) {
    EXPECT_EQ(RunVroundsd_Mode(mem, 2.7, 1, 0), 0x4000000000000000ULL) << "floor(2.7)=2";
    EXPECT_EQ(RunVroundsd_Mode(mem, -2.7, 1, 0), 0xC008000000000000ULL) << "floor(-2.7)=-3";
}

TEST_F(CpuRuntimeTest, Vroundsd_Ceil) {
    EXPECT_EQ(RunVroundsd_Mode(mem, 2.7, 2, 0), 0x4008000000000000ULL) << "ceil(2.7)=3";
    EXPECT_EQ(RunVroundsd_Mode(mem, -2.7, 2, 0), 0xC000000000000000ULL) << "ceil(-2.7)=-2";
}

TEST_F(CpuRuntimeTest, Vroundsd_Truncate) {
    EXPECT_EQ(RunVroundsd_Mode(mem, 2.7, 3, 0), 0x4000000000000000ULL) << "trunc(2.7)=2";
    EXPECT_EQ(RunVroundsd_Mode(mem, -2.7, 3, 0), 0xC000000000000000ULL) << "trunc(-2.7)=-2";
}

TEST_F(CpuRuntimeTest, Vroundsd_NearestEven) {
    // 2.5 → 2 (round half to even); 3.5 → 4.
    EXPECT_EQ(RunVroundsd_Mode(mem, 2.5, 0, 0), 0x4000000000000000ULL) << "nearest(2.5)=2 (even)";
    EXPECT_EQ(RunVroundsd_Mode(mem, 3.5, 0, 0), 0x4010000000000000ULL) << "nearest(3.5)=4 (even)";
}

TEST_F(CpuRuntimeTest, Vroundsd_PreservesSrc1UpperHalf) {
    const u64 r = RunVroundsd_Mode(mem, 2.7, 1, 0x1234567899999999ULL);
    EXPECT_EQ(r, 0x4000000000000000ULL) << "low 64 = floor(2.7)=2";
    // Re-run to inspect dst chunk1 / upper YMM via a fuller harness.
}

// dst[127:64] from src1 and upper-YMM zeroing, checked directly.
TEST_F(CpuRuntimeTest, Vroundsd_LaneCompose) {
    const u8 program[] = {0xc4, 0xe3, 0x71, 0x0b, 0xc2, 0x01, 0xc3}; // floor
    std::memcpy(mem.CodePtr(), program, sizeof(program));
    u8* guest_rsp = mem.StackTop() - 8;
    *reinterpret_cast<u64*>(guest_rsp) = kReturnSentinel;
    GuestState st{};
    st.rip = reinterpret_cast<u64>(mem.CodePtr());
    st.gpr[4] = reinterpret_cast<u64>(guest_rsp);
    st.ymm[4] = 0; st.ymm[5] = 0x1234567899999999ULL; // src1 hi -> dst chunk1
    st.ymm[8] = std::bit_cast<u64>(2.7);
    st.ymm[2] = 0xF00D; st.ymm[3] = 0xCAFE;           // upper YMM0, must zero

    Runtime rt; rt.Run(st);
    EXPECT_EQ(st.ymm[0], 0x4000000000000000ULL) << "dst.low64 = floor(2.7)=2";
    EXPECT_EQ(st.ymm[1], 0x1234567899999999ULL) << "dst[127:64] preserved from src1";
    EXPECT_EQ(st.ymm[2], 0ULL) << "VEX-128 zeroes upper YMM";
    EXPECT_EQ(st.ymm[3], 0ULL);
}

// Memory source: vroundsd xmm0, xmm1, [rbx], imm.
TEST_F(CpuRuntimeTest, Vroundsd_MemorySource) {
    double* slot = reinterpret_cast<double*>(mem.CodePtr() + 0x100);
    *slot = -2.7;
    const u8 program[] = {0xc4, 0xe3, 0x71, 0x0b, 0x03, 0x02, 0xc3}; // ceil, [rbx]
    std::memcpy(mem.CodePtr(), program, sizeof(program));
    u8* guest_rsp = mem.StackTop() - 8;
    *reinterpret_cast<u64*>(guest_rsp) = kReturnSentinel;
    GuestState st{};
    st.rip = reinterpret_cast<u64>(mem.CodePtr());
    st.gpr[4] = reinterpret_cast<u64>(guest_rsp);
    st.gpr[3] = reinterpret_cast<u64>(slot); // rbx -> m64
    st.ymm[4] = 0; st.ymm[5] = 0;

    Runtime rt; rt.Run(st);
    EXPECT_EQ(st.ymm[0], 0xC000000000000000ULL) << "ceil(-2.7) = -2";
}

// ============================================================================
// VCVTSS2SD — single-precision to double-precision widening.
// First scalar conversion where output width (64) ≠ input width (32);
// VEX merge boundary moves to bit 64. The "load src1 full → host op →
// 128-bit storeback" pattern still works because host VCVTSS2SD only
// writes xmm0.low64 and leaves xmm0[127:64] alone — same trick as
// VMULSS at a different bit boundary.
// ============================================================================

// 3.5f → 3.5 (exact: every float has an equal double).
TEST_F(CpuRuntimeTest, Vcvtss2sd_BasicWideningExact) {
    // vcvtss2sd xmm0, xmm1, xmm2 — c5 f2 5a c2
    const u8 program[] = {
        0xc5, 0xf2, 0x5a, 0xc2,
        0xc3,
    };
    std::memcpy(mem.CodePtr(), program, sizeof(program));
    u8* guest_rsp = mem.StackTop() - 8;
    *reinterpret_cast<u64*>(guest_rsp) = kReturnSentinel;
    GuestState st{};
    st.rip = reinterpret_cast<u64>(mem.CodePtr());
    st.gpr[4] = reinterpret_cast<u64>(guest_rsp);
    st.ymm[8] = static_cast<u64>(std::bit_cast<u32>(3.5f)); // xmm2

    Runtime rt;
    rt.Run(st);
    double result;
    std::memcpy(&result, &st.ymm[0], sizeof(result));
    EXPECT_EQ(result, 3.5) << "(double)3.5f should equal 3.5 exactly";
}

// Verify the merge: dst[127:64] must come from src1[127:64], NOT from
// the float value or from dst's pre-existing bits. This is the new
// constraint at the 64-bit boundary (vs the 32-bit boundary for SS
// binops).
TEST_F(CpuRuntimeTest, Vcvtss2sd_PreservesSrc1Upper64) {
    const u8 program[] = {
        0xc5, 0xf2, 0x5a, 0xc2, // vcvtss2sd xmm0, xmm1, xmm2
        0xc3,
    };
    std::memcpy(mem.CodePtr(), program, sizeof(program));
    u8* guest_rsp = mem.StackTop() - 8;
    *reinterpret_cast<u64*>(guest_rsp) = kReturnSentinel;
    GuestState st{};
    st.rip = reinterpret_cast<u64>(mem.CodePtr());
    st.gpr[4] = reinterpret_cast<u64>(guest_rsp);
    // xmm1 = lane 1: chunk 0 is irrelevant (only [127:64] is preserved,
    // which is chunk 1).
    st.ymm[4] = 0xDEADBEEFDEADBEEFULL;
    st.ymm[5] = 0xCAFEBABE12345678ULL; // <-- this is what should land in dst.chunk1
    // xmm2 = lane 2: low 32 = 1.0f (the input to convert)
    st.ymm[8] = static_cast<u64>(std::bit_cast<u32>(1.0f));
    // dst pre-pollution
    st.ymm[0] = 0xFFFFFFFFFFFFFFFFULL;
    st.ymm[1] = 0xFFFFFFFFFFFFFFFFULL;
    st.ymm[2] = 0xFFFFFFFFFFFFFFFFULL;
    st.ymm[3] = 0xFFFFFFFFFFFFFFFFULL;

    Runtime rt;
    rt.Run(st);
    double result;
    std::memcpy(&result, &st.ymm[0], sizeof(result));
    EXPECT_EQ(result, 1.0);
    EXPECT_EQ(st.ymm[1], 0xCAFEBABE12345678ULL)
        << "dst[127:64] must come from src1[127:64]";
    EXPECT_EQ(st.ymm[2], 0ULL) << "VEX-128 zeroes upper YMM";
    EXPECT_EQ(st.ymm[3], 0ULL);
}

// ============================================================================
// VMULSD — scalar double-precision multiply. First double-precision
// emitter. Shares scaffolding with the SS family via EmitScalarFp
// parameterized by ScalarFpPrec; the merge boundary is at bit 64
// (vs bit 32 for SS), but host hardware handles that transparently.
// ============================================================================

TEST_F(CpuRuntimeTest, Vmulsd_BasicMultiply) {
    // vmulsd xmm0, xmm1, xmm2  — c5 f3 59 c2 (F2 prefix marks SD)
    const u8 program[] = {
        0xc5, 0xf3, 0x59, 0xc2,
        0xc3,
    };
    std::memcpy(mem.CodePtr(), program, sizeof(program));
    u8* guest_rsp = mem.StackTop() - 8;
    *reinterpret_cast<u64*>(guest_rsp) = kReturnSentinel;
    GuestState st{};
    st.rip = reinterpret_cast<u64>(mem.CodePtr());
    st.gpr[4] = reinterpret_cast<u64>(guest_rsp);
    // xmm1.low64 = 6.0 (as double); xmm2.low64 = 7.0
    st.ymm[4] = std::bit_cast<u64>(6.0); // xmm1
    st.ymm[8] = std::bit_cast<u64>(7.0); // xmm2
    Runtime rt;
    rt.Run(st);
    double result;
    std::memcpy(&result, &st.ymm[0], sizeof(result));
    EXPECT_EQ(result, 42.0);
}

// Verify the merge boundary moves to bit 64 (vs bit 32 in SS). The
// architectural rule: dst[127:64] = src1[127:64]. This is the same
// pattern as Vcvtss2sd_PreservesSrc1Upper64 but for a binary op.
TEST_F(CpuRuntimeTest, Vmulsd_PreservesSrc1Upper64) {
    const u8 program[] = {
        0xc5, 0xf3, 0x59, 0xc2, // vmulsd xmm0, xmm1, xmm2
        0xc3,
    };
    std::memcpy(mem.CodePtr(), program, sizeof(program));
    u8* guest_rsp = mem.StackTop() - 8;
    *reinterpret_cast<u64*>(guest_rsp) = kReturnSentinel;
    GuestState st{};
    st.rip = reinterpret_cast<u64>(mem.CodePtr());
    st.gpr[4] = reinterpret_cast<u64>(guest_rsp);
    st.ymm[4] = std::bit_cast<u64>(2.0);                  // xmm1 low64
    st.ymm[5] = 0xF00DBABEAABBCCDDULL;                    // xmm1 high64 — must land in dst.chunk1
    st.ymm[8] = std::bit_cast<u64>(3.0);                  // xmm2 low64
    st.ymm[0] = 0xFFFFFFFFFFFFFFFFULL;
    st.ymm[1] = 0xFFFFFFFFFFFFFFFFULL;
    st.ymm[2] = 0xFFFFFFFFFFFFFFFFULL;
    st.ymm[3] = 0xFFFFFFFFFFFFFFFFULL;

    Runtime rt;
    rt.Run(st);
    double result;
    std::memcpy(&result, &st.ymm[0], sizeof(result));
    EXPECT_EQ(result, 6.0);
    EXPECT_EQ(st.ymm[1], 0xF00DBABEAABBCCDDULL)
        << "dst[127:64] must come from src1[127:64]";
    EXPECT_EQ(st.ymm[2], 0ULL);
    EXPECT_EQ(st.ymm[3], 0ULL);
}

// Memory-source form — exercises the SD-specific `vmovsd qword[rdx]`
// loader path (vs `vmovss dword[rdx]` for the SS variant).
TEST_F(CpuRuntimeTest, Vmulsd_MemorySource) {
    u64* dmem = reinterpret_cast<u64*>(mem.CodePtr() + 0x100);
    *dmem = std::bit_cast<u64>(4.0);
    const u8 program[] = {
        0x48, 0xb8, 0,0,0,0,0,0,0,0, // mov rax, <dmem>
        0xc5, 0xf3, 0x59, 0x00,       // vmulsd xmm0, xmm1, qword[rax]
        0xc3,
    };
    u8 prog[sizeof(program)];
    std::memcpy(prog, program, sizeof(program));
    const u64 dmem_addr = reinterpret_cast<u64>(dmem);
    std::memcpy(prog + 2, &dmem_addr, sizeof(dmem_addr));
    std::memcpy(mem.CodePtr(), prog, sizeof(prog));
    u8* guest_rsp = mem.StackTop() - 8;
    *reinterpret_cast<u64*>(guest_rsp) = kReturnSentinel;
    GuestState st{};
    st.rip = reinterpret_cast<u64>(mem.CodePtr());
    st.gpr[4] = reinterpret_cast<u64>(guest_rsp);
    st.ymm[4] = std::bit_cast<u64>(2.5);
    Runtime rt;
    rt.Run(st);
    double result;
    std::memcpy(&result, &st.ymm[0], sizeof(result));
    EXPECT_EQ(result, 10.0) << "2.5 * 4.0 = 10.0";
}

// ============================================================================
// VADDSD — scalar double-precision add. Same EmitScalarFp scaffolding
// as VMULSD; tests confirm the Add dispatch reaches vaddsd correctly.
// ============================================================================

TEST_F(CpuRuntimeTest, Vaddsd_BasicAdd) {
    // vaddsd xmm0, xmm1, xmm2  — c5 f3 58 c2
    const u8 program[] = {
        0xc5, 0xf3, 0x58, 0xc2,
        0xc3,
    };
    std::memcpy(mem.CodePtr(), program, sizeof(program));
    u8* guest_rsp = mem.StackTop() - 8;
    *reinterpret_cast<u64*>(guest_rsp) = kReturnSentinel;
    GuestState st{};
    st.rip = reinterpret_cast<u64>(mem.CodePtr());
    st.gpr[4] = reinterpret_cast<u64>(guest_rsp);
    st.ymm[4] = std::bit_cast<u64>(1.5);    // xmm1
    st.ymm[8] = std::bit_cast<u64>(2.25);   // xmm2
    Runtime rt;
    rt.Run(st);
    double result;
    std::memcpy(&result, &st.ymm[0], sizeof(result));
    EXPECT_EQ(result, 3.75);
}

// Memory-source form — matches the 8-byte length observed in the game's log.
TEST_F(CpuRuntimeTest, Vaddsd_MemorySource) {
    u64* dmem = reinterpret_cast<u64*>(mem.CodePtr() + 0x100);
    *dmem = std::bit_cast<u64>(0.5);
    const u8 program[] = {
        0x48, 0xb8, 0,0,0,0,0,0,0,0,
        0xc5, 0xf3, 0x58, 0x00,  // vaddsd xmm0, xmm1, qword[rax]
        0xc3,
    };
    u8 prog[sizeof(program)];
    std::memcpy(prog, program, sizeof(program));
    const u64 dmem_addr = reinterpret_cast<u64>(dmem);
    std::memcpy(prog + 2, &dmem_addr, sizeof(dmem_addr));
    std::memcpy(mem.CodePtr(), prog, sizeof(prog));
    u8* guest_rsp = mem.StackTop() - 8;
    *reinterpret_cast<u64*>(guest_rsp) = kReturnSentinel;
    GuestState st{};
    st.rip = reinterpret_cast<u64>(mem.CodePtr());
    st.gpr[4] = reinterpret_cast<u64>(guest_rsp);
    st.ymm[4] = std::bit_cast<u64>(100.0);
    Runtime rt;
    rt.Run(st);
    double result;
    std::memcpy(&result, &st.ymm[0], sizeof(result));
    EXPECT_EQ(result, 100.5);
}

// ============================================================================
// VCVTSD2SS — double → float narrowing conversion. Inverse of VCVTSS2SD.
// Merge boundary at bit 32 (result is float). MXCSR rounding applies
// (unlike VCVTSS2SD which is exact widening).
// ============================================================================

// 3.5 exact representation in both float and double — no rounding.
// Verifies the result is float(3.5) placed in dst.low32.
TEST_F(CpuRuntimeTest, Vcvtsd2ss_BasicNarrowingExact) {
    // vcvtsd2ss xmm0, xmm1, xmm2  — c5 f3 5a c2
    const u8 program[] = {
        0xc5, 0xf3, 0x5a, 0xc2,
        0xc3,
    };
    std::memcpy(mem.CodePtr(), program, sizeof(program));
    u8* guest_rsp = mem.StackTop() - 8;
    *reinterpret_cast<u64*>(guest_rsp) = kReturnSentinel;
    GuestState st{};
    st.rip = reinterpret_cast<u64>(mem.CodePtr());
    st.gpr[4] = reinterpret_cast<u64>(guest_rsp);
    st.ymm[8] = std::bit_cast<u64>(3.5); // xmm2 = double(3.5)

    Runtime rt;
    rt.Run(st);
    const u32 low32 = static_cast<u32>(st.ymm[0] & 0xFFFFFFFFULL);
    float result;
    std::memcpy(&result, &low32, sizeof(result));
    EXPECT_EQ(result, 3.5f) << "(float)3.5 == 3.5f exactly";
}

// Merge boundary at bit 32 — dst[127:32] must come from src1[127:32].
// This is the structural difference from VCVTSS2SD which preserves
// at bit 64.
TEST_F(CpuRuntimeTest, Vcvtsd2ss_PreservesSrc1Upper96) {
    const u8 program[] = {
        0xc5, 0xf3, 0x5a, 0xc2,
        0xc3,
    };
    std::memcpy(mem.CodePtr(), program, sizeof(program));
    u8* guest_rsp = mem.StackTop() - 8;
    *reinterpret_cast<u64*>(guest_rsp) = kReturnSentinel;
    GuestState st{};
    st.rip = reinterpret_cast<u64>(mem.CodePtr());
    st.gpr[4] = reinterpret_cast<u64>(guest_rsp);
    // xmm1 (src1): chunk 0 low32 will be replaced, hi32 preserved into
    // dst.chunk0 hi32; chunk 1 preserved as-is.
    st.ymm[4] = (static_cast<u64>(0xCAFEFACEULL) << 32) | 0xAAAAAAAAULL;
    st.ymm[5] = 0x1234567812345678ULL;
    // xmm2 (src2): the double to convert
    st.ymm[8] = std::bit_cast<u64>(1.0);
    // Pre-pollute dst
    st.ymm[0] = 0xFFFFFFFFFFFFFFFFULL;
    st.ymm[1] = 0xFFFFFFFFFFFFFFFFULL;
    st.ymm[2] = 0xFFFFFFFFFFFFFFFFULL;
    st.ymm[3] = 0xFFFFFFFFFFFFFFFFULL;

    Runtime rt;
    rt.Run(st);
    const u32 low32 = static_cast<u32>(st.ymm[0] & 0xFFFFFFFFULL);
    const u32 hi32  = static_cast<u32>(st.ymm[0] >> 32);
    float result;
    std::memcpy(&result, &low32, sizeof(result));
    EXPECT_EQ(result, 1.0f);
    EXPECT_EQ(hi32, 0xCAFEFACEu)
        << "dst[63:32] must come from src1[63:32] (merge boundary at bit 32)";
    EXPECT_EQ(st.ymm[1], 0x1234567812345678ULL)
        << "dst[127:64] from src1[127:64]";
    EXPECT_EQ(st.ymm[2], 0ULL);
    EXPECT_EQ(st.ymm[3], 0ULL);
}

// Narrowing rounding: a double value that doesn't fit exactly in float
// gets rounded under MXCSR. We verify the result matches what the host
// C++ compiler produces for (float)d, which uses the same MXCSR.
TEST_F(CpuRuntimeTest, Vcvtsd2ss_NarrowingRoundsCorrectly) {
    const u8 program[] = {
        0xc5, 0xf3, 0x5a, 0xc2,
        0xc3,
    };
    std::memcpy(mem.CodePtr(), program, sizeof(program));
    u8* guest_rsp = mem.StackTop() - 8;
    *reinterpret_cast<u64*>(guest_rsp) = kReturnSentinel;
    GuestState st{};
    st.rip = reinterpret_cast<u64>(mem.CodePtr());
    st.gpr[4] = reinterpret_cast<u64>(guest_rsp);
    // pi as a double — has more precision than float can hold; rounded.
    const double pi_d = 3.141592653589793;
    st.ymm[8] = std::bit_cast<u64>(pi_d);

    Runtime rt;
    rt.Run(st);
    const u32 low32 = static_cast<u32>(st.ymm[0] & 0xFFFFFFFFULL);
    float result;
    std::memcpy(&result, &low32, sizeof(result));
    EXPECT_EQ(result, static_cast<float>(pi_d))
        << "Narrowing rounded the same way (float) does on host";
}

// ============================================================================
// VSUBSS — scalar single-precision subtract. Cheap addition via the
// shared EmitScalarFp scaffolding. The interesting test is non-
// commutativity: a - b ≠ b - a, so this catches "I wired up the
// operand order backwards" the way an ADD test wouldn't.
// ============================================================================

TEST_F(CpuRuntimeTest, Vsubss_BasicSubtract) {
    // vsubss xmm0, xmm1, xmm2  — c5 f2 5c c2
    const u8 program[] = {
        0xc5, 0xf2, 0x5c, 0xc2,
        0xc3,
    };
    std::memcpy(mem.CodePtr(), program, sizeof(program));
    u8* guest_rsp = mem.StackTop() - 8;
    *reinterpret_cast<u64*>(guest_rsp) = kReturnSentinel;
    GuestState st{};
    st.rip = reinterpret_cast<u64>(mem.CodePtr());
    st.gpr[4] = reinterpret_cast<u64>(guest_rsp);
    // 10.0 - 3.5 = 6.5
    st.ymm[4] = static_cast<u64>(std::bit_cast<u32>(10.0f)); // xmm1
    st.ymm[8] = static_cast<u64>(std::bit_cast<u32>(3.5f));  // xmm2
    Runtime rt;
    rt.Run(st);
    const u32 low32 = static_cast<u32>(st.ymm[0] & 0xFFFFFFFFULL);
    float result;
    std::memcpy(&result, &low32, sizeof(result));
    EXPECT_EQ(result, 6.5f) << "src1 - src2 = 10.0 - 3.5 = 6.5";
}

// Order matters: confirm src1 is the minuend and src2 is the subtrahend.
// If the dispatch had src1/src2 swapped, this would produce +5 instead
// of -5.
TEST_F(CpuRuntimeTest, Vsubss_OperandOrderCheck) {
    const u8 program[] = {
        0xc5, 0xf2, 0x5c, 0xc2, // vsubss xmm0, xmm1, xmm2
        0xc3,
    };
    std::memcpy(mem.CodePtr(), program, sizeof(program));
    u8* guest_rsp = mem.StackTop() - 8;
    *reinterpret_cast<u64*>(guest_rsp) = kReturnSentinel;
    GuestState st{};
    st.rip = reinterpret_cast<u64>(mem.CodePtr());
    st.gpr[4] = reinterpret_cast<u64>(guest_rsp);
    // 5 - 10 = -5  (NOT 10 - 5 = +5)
    st.ymm[4] = static_cast<u64>(std::bit_cast<u32>(5.0f));
    st.ymm[8] = static_cast<u64>(std::bit_cast<u32>(10.0f));
    Runtime rt;
    rt.Run(st);
    const u32 low32 = static_cast<u32>(st.ymm[0] & 0xFFFFFFFFULL);
    float result;
    std::memcpy(&result, &low32, sizeof(result));
    EXPECT_EQ(result, -5.0f) << "src1 - src2 ordering: 5 - 10 = -5 (not +5)";
}

// ============================================================================
// VMOVSD — scalar double-precision move. Cleaner than VMOVSS because the
// 64-bit boundary aligns with the qword-sized GuestState chunks: no
// shift+or compose, just whole-chunk moves.
// ============================================================================

// Load form: dst.low64 = [mem], rest of YMM zeroed.
TEST_F(CpuRuntimeTest, Vmovsd_Load_PlacesLow64ZeroesRest) {
    u64* scratch = reinterpret_cast<u64*>(mem.CodePtr() + 0x100);
    *scratch = std::bit_cast<u64>(2.718281828459045);
    const u8 program[] = {
        0x48, 0xb8, 0,0,0,0,0,0,0,0, // mov rax, <scratch>
        0xc5, 0xfb, 0x10, 0x00,       // vmovsd xmm0, qword[rax]
        0xc3,
    };
    u8 prog[sizeof(program)];
    std::memcpy(prog, program, sizeof(program));
    const u64 scratch_addr = reinterpret_cast<u64>(scratch);
    std::memcpy(prog + 2, &scratch_addr, sizeof(scratch_addr));
    std::memcpy(mem.CodePtr(), prog, sizeof(prog));
    u8* guest_rsp = mem.StackTop() - 8;
    *reinterpret_cast<u64*>(guest_rsp) = kReturnSentinel;
    GuestState st{};
    st.rip = reinterpret_cast<u64>(mem.CodePtr());
    st.gpr[4] = reinterpret_cast<u64>(guest_rsp);
    // Pre-pollute to confirm zero-extension.
    st.ymm[0] = 0xDEADBEEFDEADBEEFULL;
    st.ymm[1] = 0xCAFEBABECAFEBABEULL;
    st.ymm[2] = 0x1234567812345678ULL;
    st.ymm[3] = 0xABCDEF01ABCDEF01ULL;
    Runtime rt;
    rt.Run(st);
    double loaded;
    std::memcpy(&loaded, &st.ymm[0], sizeof(loaded));
    EXPECT_EQ(loaded, 2.718281828459045);
    EXPECT_EQ(st.ymm[1], 0ULL) << "chunk 1 (xmm[127:64]) zeroed";
    EXPECT_EQ(st.ymm[2], 0ULL) << "upper YMM zeroed (VEX-128)";
    EXPECT_EQ(st.ymm[3], 0ULL);
}

// Store form: [mem] = xmm.low64; surrounding memory untouched.
TEST_F(CpuRuntimeTest, Vmovsd_Store_WritesLow64Only) {
    u64* scratch = reinterpret_cast<u64*>(mem.CodePtr() + 0x100);
    *scratch = 0xAAAAAAAA00000000ULL;
    u64* sentinel = scratch + 1;
    *sentinel = 0xDEADBEEFDEADBEEFULL;
    const u8 program[] = {
        0x48, 0xb8, 0,0,0,0,0,0,0,0,
        0xc5, 0xfb, 0x11, 0x00,       // vmovsd qword[rax], xmm0
        0xc3,
    };
    u8 prog[sizeof(program)];
    std::memcpy(prog, program, sizeof(program));
    const u64 scratch_addr = reinterpret_cast<u64>(scratch);
    std::memcpy(prog + 2, &scratch_addr, sizeof(scratch_addr));
    std::memcpy(mem.CodePtr(), prog, sizeof(prog));
    u8* guest_rsp = mem.StackTop() - 8;
    *reinterpret_cast<u64*>(guest_rsp) = kReturnSentinel;
    GuestState st{};
    st.rip = reinterpret_cast<u64>(mem.CodePtr());
    st.gpr[4] = reinterpret_cast<u64>(guest_rsp);
    st.ymm[0] = std::bit_cast<u64>(1.5);  // xmm0.low64
    st.ymm[1] = 0xFFFFFFFFFFFFFFFFULL;    // xmm0[127:64] — must NOT be stored
    Runtime rt;
    rt.Run(st);
    EXPECT_EQ(*scratch, std::bit_cast<u64>(1.5));
    EXPECT_EQ(*sentinel, 0xDEADBEEFDEADBEEFULL)
        << "next qword must be untouched (only 8 bytes written)";
}

// Reg-reg-reg merge: dst.low64 from src2, dst[127:64] from src1.
TEST_F(CpuRuntimeTest, Vmovsd_RegRegReg_MergesFromTwoSources) {
    // vmovsd xmm0, xmm1, xmm2  — c5 f3 10 c2
    const u8 program[] = {
        0xc5, 0xf3, 0x10, 0xc2,
        0xc3,
    };
    std::memcpy(mem.CodePtr(), program, sizeof(program));
    u8* guest_rsp = mem.StackTop() - 8;
    *reinterpret_cast<u64*>(guest_rsp) = kReturnSentinel;
    GuestState st{};
    st.rip = reinterpret_cast<u64>(mem.CodePtr());
    st.gpr[4] = reinterpret_cast<u64>(guest_rsp);
    // src1 = xmm1 (lane 4): chunk 0 ignored, chunk 1 preserved into dst.chunk1
    st.ymm[4] = 0xDEAD000000000000ULL;          // chunk 0 - ignored
    st.ymm[5] = 0xCAFEBABE11223344ULL;          // chunk 1 - preserved
    // src2 = xmm2 (lane 8): chunk 0 → dst.chunk0
    st.ymm[8] = 0x123456789ABCDEF0ULL;
    st.ymm[9] = 0x9999999999999999ULL;          // ignored
    // pre-pollute dst (xmm0, lane 0)
    st.ymm[0] = 0xFFFFFFFFFFFFFFFFULL;
    st.ymm[1] = 0xFFFFFFFFFFFFFFFFULL;
    st.ymm[2] = 0xFFFFFFFFFFFFFFFFULL;
    st.ymm[3] = 0xFFFFFFFFFFFFFFFFULL;

    Runtime rt;
    rt.Run(st);
    EXPECT_EQ(st.ymm[0], 0x123456789ABCDEF0ULL) << "dst.low64 from src2";
    EXPECT_EQ(st.ymm[1], 0xCAFEBABE11223344ULL) << "dst[127:64] from src1";
    EXPECT_EQ(st.ymm[2], 0ULL);
    EXPECT_EQ(st.ymm[3], 0ULL);
}

// ============================================================================
// VSUBSD — scalar double-precision subtract. Same EmitScalarFp scaffolding
// as VSUBSS; non-commutativity test catches operand-order regressions.
// ============================================================================

TEST_F(CpuRuntimeTest, Vsubsd_BasicSubtract) {
    // vsubsd xmm0, xmm1, xmm2  — c5 f3 5c c2
    const u8 program[] = {
        0xc5, 0xf3, 0x5c, 0xc2,
        0xc3,
    };
    std::memcpy(mem.CodePtr(), program, sizeof(program));
    u8* guest_rsp = mem.StackTop() - 8;
    *reinterpret_cast<u64*>(guest_rsp) = kReturnSentinel;
    GuestState st{};
    st.rip = reinterpret_cast<u64>(mem.CodePtr());
    st.gpr[4] = reinterpret_cast<u64>(guest_rsp);
    st.ymm[4] = std::bit_cast<u64>(10.0); // xmm1
    st.ymm[8] = std::bit_cast<u64>(3.5);  // xmm2
    Runtime rt;
    rt.Run(st);
    double result;
    std::memcpy(&result, &st.ymm[0], sizeof(result));
    EXPECT_EQ(result, 6.5);
}

// Non-commutativity check (5 - 10 = -5, not +5).
TEST_F(CpuRuntimeTest, Vsubsd_OperandOrderCheck) {
    const u8 program[] = {
        0xc5, 0xf3, 0x5c, 0xc2, // vsubsd xmm0, xmm1, xmm2
        0xc3,
    };
    std::memcpy(mem.CodePtr(), program, sizeof(program));
    u8* guest_rsp = mem.StackTop() - 8;
    *reinterpret_cast<u64*>(guest_rsp) = kReturnSentinel;
    GuestState st{};
    st.rip = reinterpret_cast<u64>(mem.CodePtr());
    st.gpr[4] = reinterpret_cast<u64>(guest_rsp);
    st.ymm[4] = std::bit_cast<u64>(5.0);
    st.ymm[8] = std::bit_cast<u64>(10.0);
    Runtime rt;
    rt.Run(st);
    double result;
    std::memcpy(&result, &st.ymm[0], sizeof(result));
    EXPECT_EQ(result, -5.0) << "src1 - src2 ordering: 5 - 10 = -5";
}

// ============================================================================
// VDIVSD — scalar double-precision divide. Completes the scalar-FP
// binop matrix (4 ops × 2 precisions). Last 9-LOC payoff from the
// EmitScalarFp scaffolding.
// ============================================================================

TEST_F(CpuRuntimeTest, Vdivsd_BasicDivide) {
    // vdivsd xmm0, xmm1, xmm2  — c5 f3 5e c2
    const u8 program[] = {
        0xc5, 0xf3, 0x5e, 0xc2,
        0xc3,
    };
    std::memcpy(mem.CodePtr(), program, sizeof(program));
    u8* guest_rsp = mem.StackTop() - 8;
    *reinterpret_cast<u64*>(guest_rsp) = kReturnSentinel;
    GuestState st{};
    st.rip = reinterpret_cast<u64>(mem.CodePtr());
    st.gpr[4] = reinterpret_cast<u64>(guest_rsp);
    st.ymm[4] = std::bit_cast<u64>(12.0); // xmm1
    st.ymm[8] = std::bit_cast<u64>(4.0);  // xmm2
    Runtime rt;
    rt.Run(st);
    double result;
    std::memcpy(&result, &st.ymm[0], sizeof(result));
    EXPECT_EQ(result, 3.0) << "12.0 / 4.0 = 3.0";
}

// Non-commutativity check (10 / 2 = 5, not 0.2). Catches src1/src2
// ordering regressions on a divide-specific value.
TEST_F(CpuRuntimeTest, Vdivsd_OperandOrderCheck) {
    const u8 program[] = {
        0xc5, 0xf3, 0x5e, 0xc2, // vdivsd xmm0, xmm1, xmm2
        0xc3,
    };
    std::memcpy(mem.CodePtr(), program, sizeof(program));
    u8* guest_rsp = mem.StackTop() - 8;
    *reinterpret_cast<u64*>(guest_rsp) = kReturnSentinel;
    GuestState st{};
    st.rip = reinterpret_cast<u64>(mem.CodePtr());
    st.gpr[4] = reinterpret_cast<u64>(guest_rsp);
    st.ymm[4] = std::bit_cast<u64>(10.0);
    st.ymm[8] = std::bit_cast<u64>(2.0);
    Runtime rt;
    rt.Run(st);
    double result;
    std::memcpy(&result, &st.ymm[0], sizeof(result));
    EXPECT_EQ(result, 5.0) << "src1 / src2 ordering: 10 / 2 = 5 (not 0.2)";
}

// ============================================================================
// VXORPS mem-src — extending the existing EmitVecBitXor (previously
// reg-reg-reg only). The mem operand is full xmm width (128 bits = 16
// bytes); we XOR the source register chunk-by-chunk against memory
// starting at the computed EA.
//
// Typical pattern in compiled code: `vxorps xmm, xmm, [rip+sign_mask]`
// for floating-point sign-flip via a static -0.0 / -0.0... mask.
// ============================================================================

TEST_F(CpuRuntimeTest, Vxorps_MemSource_128_XorsAcrossChunks) {
    // 16-byte aligned-ish memory operand. Layout: chunk 0 = 0xFF...,
    // chunk 1 = 0x55... — verifies BOTH qwords of the xmm get xor'd.
    u64* mem_op = reinterpret_cast<u64*>(mem.CodePtr() + 0x100);
    mem_op[0] = 0xFFFFFFFFFFFFFFFFULL;
    mem_op[1] = 0x5555555555555555ULL;

    const u8 program[] = {
        0x48, 0xb8, 0,0,0,0,0,0,0,0,  // mov rax, <mem_op>
        0xc5, 0xf0, 0x57, 0x00,        // vxorps xmm0, xmm1, xmmword[rax]
        0xc3,
    };
    u8 prog[sizeof(program)];
    std::memcpy(prog, program, sizeof(program));
    const u64 mem_addr = reinterpret_cast<u64>(mem_op);
    std::memcpy(prog + 2, &mem_addr, sizeof(mem_addr));
    std::memcpy(mem.CodePtr(), prog, sizeof(prog));
    u8* guest_rsp = mem.StackTop() - 8;
    *reinterpret_cast<u64*>(guest_rsp) = kReturnSentinel;
    GuestState st{};
    st.rip = reinterpret_cast<u64>(mem.CodePtr());
    st.gpr[4] = reinterpret_cast<u64>(guest_rsp);
    // xmm1 (src1): chunk 0 = 0x0F0F...0F, chunk 1 = 0xAAAA...AA
    st.ymm[4] = 0x0F0F0F0F0F0F0F0FULL;
    st.ymm[5] = 0xAAAAAAAAAAAAAAAAULL;
    // pre-pollute xmm0 (dst) lanes 2/3 to confirm zeroing.
    st.ymm[2] = 0xDEADBEEFDEADBEEFULL;
    st.ymm[3] = 0xDEADBEEFDEADBEEFULL;

    Runtime rt;
    rt.Run(st);
    EXPECT_EQ(st.ymm[0], 0xF0F0F0F0F0F0F0F0ULL)
        << "chunk 0 = 0x0F0F... XOR 0xFFFF... = 0xF0F0...";
    EXPECT_EQ(st.ymm[1], 0xFFFFFFFFFFFFFFFFULL)
        << "chunk 1 = 0xAAAA... XOR 0x5555... = 0xFFFF...";
    EXPECT_EQ(st.ymm[2], 0ULL) << "VEX-128 zeros upper YMM";
    EXPECT_EQ(st.ymm[3], 0ULL);
}

// Sign-flip idiom: VXORPS xmm, xmm, [sign_mask] where sign_mask =
// {0x80000000, 0, 0, 0}. Negates the low-32 float; upper bits passed
// through.
TEST_F(CpuRuntimeTest, Vxorps_MemSource_SignFlipPattern) {
    u32* mask = reinterpret_cast<u32*>(mem.CodePtr() + 0x100);
    mask[0] = 0x80000000u; // sign-bit only
    mask[1] = mask[2] = mask[3] = 0;

    const u8 program[] = {
        0x48, 0xb8, 0,0,0,0,0,0,0,0,
        0xc5, 0xf0, 0x57, 0x00,
        0xc3,
    };
    u8 prog[sizeof(program)];
    std::memcpy(prog, program, sizeof(program));
    const u64 mask_addr = reinterpret_cast<u64>(mask);
    std::memcpy(prog + 2, &mask_addr, sizeof(mask_addr));
    std::memcpy(mem.CodePtr(), prog, sizeof(prog));
    u8* guest_rsp = mem.StackTop() - 8;
    *reinterpret_cast<u64*>(guest_rsp) = kReturnSentinel;
    GuestState st{};
    st.rip = reinterpret_cast<u64>(mem.CodePtr());
    st.gpr[4] = reinterpret_cast<u64>(guest_rsp);
    // xmm1.low32 = +3.5; everything else 0
    st.ymm[4] = static_cast<u64>(std::bit_cast<u32>(3.5f));

    Runtime rt;
    rt.Run(st);
    const u32 low32 = static_cast<u32>(st.ymm[0] & 0xFFFFFFFFULL);
    float result;
    std::memcpy(&result, &low32, sizeof(result));
    EXPECT_EQ(result, -3.5f) << "sign-flip via XOR with 0x80000000 mask";
}

// ============================================================================
// VPAND — vector bitwise AND. Shares EmitVecBitOp scaffolding with
// VPXOR/VXORPS; this just exercises the And kind. VPAND and VANDPS
// produce identical bit patterns and are aliased to the same emitter.
// ============================================================================

TEST_F(CpuRuntimeTest, Vpand_RegSrc_AndsAcrossChunks) {
    // vpand xmm0, xmm1, xmm2 — c5 f1 db c2
    const u8 program[] = {
        0xc5, 0xf1, 0xdb, 0xc2,
        0xc3,
    };
    std::memcpy(mem.CodePtr(), program, sizeof(program));
    u8* guest_rsp = mem.StackTop() - 8;
    *reinterpret_cast<u64*>(guest_rsp) = kReturnSentinel;
    GuestState st{};
    st.rip = reinterpret_cast<u64>(mem.CodePtr());
    st.gpr[4] = reinterpret_cast<u64>(guest_rsp);
    // xmm1 (lane 4): chunk 0 = 0xFF00FF00..., chunk 1 = 0xAAAA...
    st.ymm[4] = 0xFF00FF00FF00FF00ULL;
    st.ymm[5] = 0xAAAAAAAAAAAAAAAAULL;
    // xmm2 (lane 8): chunk 0 = 0x0FF00FF0..., chunk 1 = 0xCCCC...
    st.ymm[8] = 0x0FF00FF00FF00FF0ULL;
    st.ymm[9] = 0xCCCCCCCCCCCCCCCCULL;
    // Pre-pollute upper YMM to confirm zeroing.
    st.ymm[2] = 0xDEADBEEFDEADBEEFULL;
    st.ymm[3] = 0xDEADBEEFDEADBEEFULL;

    Runtime rt;
    rt.Run(st);
    EXPECT_EQ(st.ymm[0], 0x0F000F000F000F00ULL)
        << "chunk 0 = 0xFF00FF00... AND 0x0FF00FF0... = 0x0F000F00...";
    EXPECT_EQ(st.ymm[1], 0x8888888888888888ULL)
        << "chunk 1 = 0xAAAA... AND 0xCCCC... = 0x8888...";
    EXPECT_EQ(st.ymm[2], 0ULL) << "VEX-128 zeros upper YMM";
    EXPECT_EQ(st.ymm[3], 0ULL);
}

// Absolute-value idiom for float: AND with 0x7FFFFFFF clears the sign bit.
// Same use case as VXORPS sign-flip, just AND instead of XOR.
TEST_F(CpuRuntimeTest, Vpand_MemSource_AbsValuePattern) {
    u32* mask = reinterpret_cast<u32*>(mem.CodePtr() + 0x100);
    mask[0] = 0x7FFFFFFFu; // clears sign bit
    mask[1] = mask[2] = mask[3] = 0xFFFFFFFFu; // leave other lanes alone

    const u8 program[] = {
        0x48, 0xb8, 0,0,0,0,0,0,0,0,
        0xc5, 0xf1, 0xdb, 0x00, // vpand xmm0, xmm1, xmmword[rax]
        0xc3,
    };
    u8 prog[sizeof(program)];
    std::memcpy(prog, program, sizeof(program));
    const u64 mask_addr = reinterpret_cast<u64>(mask);
    std::memcpy(prog + 2, &mask_addr, sizeof(mask_addr));
    std::memcpy(mem.CodePtr(), prog, sizeof(prog));
    u8* guest_rsp = mem.StackTop() - 8;
    *reinterpret_cast<u64*>(guest_rsp) = kReturnSentinel;
    GuestState st{};
    st.rip = reinterpret_cast<u64>(mem.CodePtr());
    st.gpr[4] = reinterpret_cast<u64>(guest_rsp);
    // xmm1.low32 = -3.5f
    st.ymm[4] = static_cast<u64>(std::bit_cast<u32>(-3.5f));

    Runtime rt;
    rt.Run(st);
    const u32 low32 = static_cast<u32>(st.ymm[0] & 0xFFFFFFFFULL);
    float result;
    std::memcpy(&result, &low32, sizeof(result));
    EXPECT_EQ(result, 3.5f) << "abs() via AND with 0x7FFFFFFF mask";
}

// ============================================================================
// VSQRTSD — scalar double-precision square root. Conceptually unary
// (dst.low64 = sqrt(src2.low64)) but encoded as 3-operand (dst, src1,
// src2) where src1 supplies the preserved upper 64 of dst. Slots into
// EmitScalarFp via the Sqrt kind alongside the binops.
// ============================================================================

TEST_F(CpuRuntimeTest, Vsqrtsd_BasicSquareRoot) {
    // vsqrtsd xmm0, xmm1, xmm2  — c5 f3 51 c2
    const u8 program[] = {
        0xc5, 0xf3, 0x51, 0xc2,
        0xc3,
    };
    std::memcpy(mem.CodePtr(), program, sizeof(program));
    u8* guest_rsp = mem.StackTop() - 8;
    *reinterpret_cast<u64*>(guest_rsp) = kReturnSentinel;
    GuestState st{};
    st.rip = reinterpret_cast<u64>(mem.CodePtr());
    st.gpr[4] = reinterpret_cast<u64>(guest_rsp);
    // xmm2.low64 = 16.0 (the input to sqrt)
    st.ymm[8] = std::bit_cast<u64>(16.0);

    Runtime rt;
    rt.Run(st);
    double result;
    std::memcpy(&result, &st.ymm[0], sizeof(result));
    EXPECT_EQ(result, 4.0) << "sqrt(16.0) = 4.0";
}

// Verify src1 (NOT src2) supplies the preserved upper 64. The merge
// here is the same as VMULSD's: dst[127:64] = src1[127:64]. Even
// though src2 carries the actual numerical input, src2's upper bits
// must NOT leak into dst.
TEST_F(CpuRuntimeTest, Vsqrtsd_PreservesSrc1Upper64) {
    const u8 program[] = {
        0xc5, 0xf3, 0x51, 0xc2, // vsqrtsd xmm0, xmm1, xmm2
        0xc3,
    };
    std::memcpy(mem.CodePtr(), program, sizeof(program));
    u8* guest_rsp = mem.StackTop() - 8;
    *reinterpret_cast<u64*>(guest_rsp) = kReturnSentinel;
    GuestState st{};
    st.rip = reinterpret_cast<u64>(mem.CodePtr());
    st.gpr[4] = reinterpret_cast<u64>(guest_rsp);
    // xmm1 (src1): chunk 0 ignored, chunk 1 = the value that must land in dst.chunk1
    st.ymm[4] = 0xDEADBEEFDEADBEEFULL;
    st.ymm[5] = 0xCAFEBABE12345678ULL;
    // xmm2 (src2): low64 = 25.0, hi64 = pattern that must NOT land in dst
    st.ymm[8] = std::bit_cast<u64>(25.0);
    st.ymm[9] = 0xFEEDFACEFEEDFACEULL; // distinguish from src1 upper
    // Pre-pollute dst
    st.ymm[0] = 0xFFFFFFFFFFFFFFFFULL;
    st.ymm[1] = 0xFFFFFFFFFFFFFFFFULL;
    st.ymm[2] = 0xFFFFFFFFFFFFFFFFULL;
    st.ymm[3] = 0xFFFFFFFFFFFFFFFFULL;

    Runtime rt;
    rt.Run(st);
    double result;
    std::memcpy(&result, &st.ymm[0], sizeof(result));
    EXPECT_EQ(result, 5.0);
    EXPECT_EQ(st.ymm[1], 0xCAFEBABE12345678ULL)
        << "dst[127:64] from src1, not from src2 (which had 0xFEEDFACE...)";
    EXPECT_EQ(st.ymm[2], 0ULL);
    EXPECT_EQ(st.ymm[3], 0ULL);
}

// ============================================================================
// VBLENDPS — packed single-precision blend with imm8 mask. Each bit of
// the imm8 selects per-element between src1 (bit=0) and src2 (bit=1).
// Tests exercise distinguishing mask patterns to verify both directions
// of the selection.
// ============================================================================

// imm8 = 0x0A = 0b1010 — alternating, picks src2 for elements 1 and 3,
// src1 for elements 0 and 2. Catches "selected the wrong source" or
// "got the bit order backwards" regressions.
TEST_F(CpuRuntimeTest, Vblendps_AlternatingMask) {
    // vblendps xmm0, xmm1, xmm2, 0x0A
    // c4 e3 71 0c c2 0a  (6 bytes)
    const u8 program[] = {
        0xc4, 0xe3, 0x71, 0x0c, 0xc2, 0x0a,
        0xc3,
    };
    std::memcpy(mem.CodePtr(), program, sizeof(program));
    u8* guest_rsp = mem.StackTop() - 8;
    *reinterpret_cast<u64*>(guest_rsp) = kReturnSentinel;
    GuestState st{};
    st.rip = reinterpret_cast<u64>(mem.CodePtr());
    st.gpr[4] = reinterpret_cast<u64>(guest_rsp);
    // xmm1 (lane 4): 4 floats = 1.0, 2.0, 3.0, 4.0
    st.ymm[4] = (static_cast<u64>(std::bit_cast<u32>(2.0f)) << 32)
              |  static_cast<u64>(std::bit_cast<u32>(1.0f));
    st.ymm[5] = (static_cast<u64>(std::bit_cast<u32>(4.0f)) << 32)
              |  static_cast<u64>(std::bit_cast<u32>(3.0f));
    // xmm2 (lane 8): 4 floats = 10.0, 20.0, 30.0, 40.0
    st.ymm[8] = (static_cast<u64>(std::bit_cast<u32>(20.0f)) << 32)
              |  static_cast<u64>(std::bit_cast<u32>(10.0f));
    st.ymm[9] = (static_cast<u64>(std::bit_cast<u32>(40.0f)) << 32)
              |  static_cast<u64>(std::bit_cast<u32>(30.0f));

    Runtime rt;
    rt.Run(st);

    // Expected with imm8=0x0A=0b1010:
    //   element 0 (bit 0 = 0): src1[0] = 1.0
    //   element 1 (bit 1 = 1): src2[1] = 20.0
    //   element 2 (bit 2 = 0): src1[2] = 3.0
    //   element 3 (bit 3 = 1): src2[3] = 40.0
    const u32 e0 = static_cast<u32>(st.ymm[0] & 0xFFFFFFFFULL);
    const u32 e1 = static_cast<u32>(st.ymm[0] >> 32);
    const u32 e2 = static_cast<u32>(st.ymm[1] & 0xFFFFFFFFULL);
    const u32 e3 = static_cast<u32>(st.ymm[1] >> 32);
    EXPECT_EQ(std::bit_cast<float>(e0), 1.0f);
    EXPECT_EQ(std::bit_cast<float>(e1), 20.0f);
    EXPECT_EQ(std::bit_cast<float>(e2), 3.0f);
    EXPECT_EQ(std::bit_cast<float>(e3), 40.0f);
}

// imm8 = 0x00 — all src1 (identity-like blend, no src2 contribution).
// Tests that bit=0 reliably selects src1.
TEST_F(CpuRuntimeTest, Vblendps_AllZeroMask_PicksSrc1) {
    // vblendps xmm0, xmm1, xmm2, 0x00
    const u8 program[] = {
        0xc4, 0xe3, 0x71, 0x0c, 0xc2, 0x00,
        0xc3,
    };
    std::memcpy(mem.CodePtr(), program, sizeof(program));
    u8* guest_rsp = mem.StackTop() - 8;
    *reinterpret_cast<u64*>(guest_rsp) = kReturnSentinel;
    GuestState st{};
    st.rip = reinterpret_cast<u64>(mem.CodePtr());
    st.gpr[4] = reinterpret_cast<u64>(guest_rsp);
    st.ymm[4] = 0xAAAAAAAA11111111ULL; // src1 distinguishing
    st.ymm[5] = 0xCCCCCCCC33333333ULL;
    st.ymm[8] = 0xBBBBBBBB22222222ULL; // src2 — must NOT appear
    st.ymm[9] = 0xDDDDDDDD44444444ULL;

    Runtime rt;
    rt.Run(st);
    EXPECT_EQ(st.ymm[0], 0xAAAAAAAA11111111ULL) << "all src1";
    EXPECT_EQ(st.ymm[1], 0xCCCCCCCC33333333ULL);
    EXPECT_EQ(st.ymm[2], 0ULL) << "VEX-128 zeros upper YMM";
    EXPECT_EQ(st.ymm[3], 0ULL);
}

// imm8 = 0x0F — all src2 (only the low 4 bits matter for the 128-bit
// form; bits 7:4 are reserved).
TEST_F(CpuRuntimeTest, Vblendps_AllOnesMask_PicksSrc2) {
    const u8 program[] = {
        0xc4, 0xe3, 0x71, 0x0c, 0xc2, 0x0f,
        0xc3,
    };
    std::memcpy(mem.CodePtr(), program, sizeof(program));
    u8* guest_rsp = mem.StackTop() - 8;
    *reinterpret_cast<u64*>(guest_rsp) = kReturnSentinel;
    GuestState st{};
    st.rip = reinterpret_cast<u64>(mem.CodePtr());
    st.gpr[4] = reinterpret_cast<u64>(guest_rsp);
    st.ymm[4] = 0xAAAAAAAA11111111ULL; // src1 — must NOT appear
    st.ymm[5] = 0xCCCCCCCC33333333ULL;
    st.ymm[8] = 0xBBBBBBBB22222222ULL; // src2 distinguishing
    st.ymm[9] = 0xDDDDDDDD44444444ULL;

    Runtime rt;
    rt.Run(st);
    EXPECT_EQ(st.ymm[0], 0xBBBBBBBB22222222ULL) << "all src2";
    EXPECT_EQ(st.ymm[1], 0xDDDDDDDD44444444ULL);
}

// ============================================================================
// VPCMPEQD — compare packed signed dwords for equality. Shares the
// EmitVecCmpEq scaffolding with VPCMPEQB; this exercises the Dword
// kind and the new mem-src code path.
// ============================================================================

// Each 32-bit element compared independently; equal → 0xFFFFFFFF, else 0.
TEST_F(CpuRuntimeTest, Vpcmpeqd_RegSrc_PerElementEquality) {
    // vpcmpeqd xmm0, xmm1, xmm2  — c5 f1 76 c2
    const u8 program[] = {
        0xc5, 0xf1, 0x76, 0xc2,
        0xc3,
    };
    std::memcpy(mem.CodePtr(), program, sizeof(program));
    u8* guest_rsp = mem.StackTop() - 8;
    *reinterpret_cast<u64*>(guest_rsp) = kReturnSentinel;
    GuestState st{};
    st.rip = reinterpret_cast<u64>(mem.CodePtr());
    st.gpr[4] = reinterpret_cast<u64>(guest_rsp);
    // xmm1 (4 dwords): [0x11111111, 0x22222222, 0x33333333, 0x44444444]
    st.ymm[4] = 0x2222222211111111ULL;
    st.ymm[5] = 0x4444444433333333ULL;
    // xmm2: [0x11111111, 0xFFFFFFFF, 0x33333333, 0x00000000]
    //   elements 0,2 match; elements 1,3 differ.
    st.ymm[8] = 0xFFFFFFFF11111111ULL;
    st.ymm[9] = 0x0000000033333333ULL;
    // Pre-pollute upper YMM
    st.ymm[2] = 0xDEADBEEFDEADBEEFULL;
    st.ymm[3] = 0xDEADBEEFDEADBEEFULL;

    Runtime rt;
    rt.Run(st);
    // Expected per element: [0xFFFFFFFF, 0x00000000, 0xFFFFFFFF, 0x00000000]
    EXPECT_EQ(st.ymm[0], 0x00000000FFFFFFFFULL) << "elements 0 (eq) and 1 (neq)";
    EXPECT_EQ(st.ymm[1], 0x00000000FFFFFFFFULL) << "elements 2 (eq) and 3 (neq)";
    EXPECT_EQ(st.ymm[2], 0ULL) << "VEX-128 zeros upper YMM";
    EXPECT_EQ(st.ymm[3], 0ULL);
}

// Mem-src form — matches the 8-byte length observed in the game's log.
// This is also the first test exercising the new mem-src path in
// EmitVecCmpEq (extracted from the original EmitVpcmpeqb refactor).
TEST_F(CpuRuntimeTest, Vpcmpeqd_MemSource) {
    u32* mem_op = reinterpret_cast<u32*>(mem.CodePtr() + 0x100);
    mem_op[0] = 0x55555555u; // matches src1 element 0
    mem_op[1] = 0x99999999u; // does NOT match src1 element 1
    mem_op[2] = 0xAAAAAAAAu; // matches src1 element 2
    mem_op[3] = 0x12345678u; // does NOT match src1 element 3

    const u8 program[] = {
        0x48, 0xb8, 0,0,0,0,0,0,0,0,
        0xc5, 0xf1, 0x76, 0x00, // vpcmpeqd xmm0, xmm1, xmmword[rax]
        0xc3,
    };
    u8 prog[sizeof(program)];
    std::memcpy(prog, program, sizeof(program));
    const u64 mem_addr = reinterpret_cast<u64>(mem_op);
    std::memcpy(prog + 2, &mem_addr, sizeof(mem_addr));
    std::memcpy(mem.CodePtr(), prog, sizeof(prog));
    u8* guest_rsp = mem.StackTop() - 8;
    *reinterpret_cast<u64*>(guest_rsp) = kReturnSentinel;
    GuestState st{};
    st.rip = reinterpret_cast<u64>(mem.CodePtr());
    st.gpr[4] = reinterpret_cast<u64>(guest_rsp);
    // xmm1: [0x55555555, 0x77777777, 0xAAAAAAAA, 0xBBBBBBBB]
    st.ymm[4] = 0x7777777755555555ULL;
    st.ymm[5] = 0xBBBBBBBBAAAAAAAAULL;

    Runtime rt;
    rt.Run(st);
    // Expected: [0xFFFFFFFF, 0x00000000, 0xFFFFFFFF, 0x00000000]
    EXPECT_EQ(st.ymm[0], 0x00000000FFFFFFFFULL);
    EXPECT_EQ(st.ymm[1], 0x00000000FFFFFFFFULL);
}

// ============================================================================
// VPSUBD — packed subtract of 32-bit dwords. Per-element wraparound,
// no flags. The non-trivial correctness check is that subtraction
// borrows must NOT propagate across the 32-bit element boundary
// inside a 64-bit chunk — which would happen if we mistakenly tried
// to GPR-relay at 64-bit granularity like VPXOR does.
// ============================================================================

TEST_F(CpuRuntimeTest, Vpsubd_BasicPerElementSubtract) {
    // vpsubd xmm0, xmm1, xmm2  — c5 f1 fa c2
    const u8 program[] = {
        0xc5, 0xf1, 0xfa, 0xc2,
        0xc3,
    };
    std::memcpy(mem.CodePtr(), program, sizeof(program));
    u8* guest_rsp = mem.StackTop() - 8;
    *reinterpret_cast<u64*>(guest_rsp) = kReturnSentinel;
    GuestState st{};
    st.rip = reinterpret_cast<u64>(mem.CodePtr());
    st.gpr[4] = reinterpret_cast<u64>(guest_rsp);
    // xmm1: [100, 200, 300, 400]
    st.ymm[4] = (static_cast<u64>(200u) << 32) | 100u;
    st.ymm[5] = (static_cast<u64>(400u) << 32) | 300u;
    // xmm2: [10, 20, 30, 40]
    st.ymm[8] = (static_cast<u64>(20u) << 32) | 10u;
    st.ymm[9] = (static_cast<u64>(40u) << 32) | 30u;
    // Pre-pollute upper YMM
    st.ymm[2] = 0xDEADBEEFDEADBEEFULL;
    st.ymm[3] = 0xDEADBEEFDEADBEEFULL;

    Runtime rt;
    rt.Run(st);
    // Expected: [90, 180, 270, 360]
    EXPECT_EQ(static_cast<u32>(st.ymm[0] & 0xFFFFFFFFULL), 90u);
    EXPECT_EQ(static_cast<u32>(st.ymm[0] >> 32), 180u);
    EXPECT_EQ(static_cast<u32>(st.ymm[1] & 0xFFFFFFFFULL), 270u);
    EXPECT_EQ(static_cast<u32>(st.ymm[1] >> 32), 360u);
    EXPECT_EQ(st.ymm[2], 0ULL);
    EXPECT_EQ(st.ymm[3], 0ULL);
}

// Borrow-isolation: each 32-bit element subtracts independently.
// element 0: 0x00000000 - 0x00000001 = 0xFFFFFFFF (wraparound)
// element 1: 0x12345678 - 0x00000000 = 0x12345678 (must NOT decrement!)
// If a 64-bit subtraction were used by mistake, element 1 would
// observe a borrow from element 0 and yield 0x12345677.
TEST_F(CpuRuntimeTest, Vpsubd_BorrowDoesNotCrossElementBoundary) {
    const u8 program[] = {
        0xc5, 0xf1, 0xfa, 0xc2, // vpsubd xmm0, xmm1, xmm2
        0xc3,
    };
    std::memcpy(mem.CodePtr(), program, sizeof(program));
    u8* guest_rsp = mem.StackTop() - 8;
    *reinterpret_cast<u64*>(guest_rsp) = kReturnSentinel;
    GuestState st{};
    st.rip = reinterpret_cast<u64>(mem.CodePtr());
    st.gpr[4] = reinterpret_cast<u64>(guest_rsp);
    // xmm1 chunk 0: high32 = 0x12345678 (element 1), low32 = 0x00000000 (element 0)
    st.ymm[4] = 0x1234567800000000ULL;
    // xmm2 chunk 0: high32 = 0x00000000 (element 1), low32 = 0x00000001 (element 0)
    st.ymm[8] = 0x0000000000000001ULL;
    // Other chunks zero (uninteresting elements).
    st.ymm[5] = st.ymm[9] = 0;

    Runtime rt;
    rt.Run(st);
    // Expected chunk 0: element 0 = 0 - 1 = 0xFFFFFFFF (wrap),
    //                   element 1 = 0x12345678 - 0 = 0x12345678 (NOT 0x12345677).
    const u32 e0 = static_cast<u32>(st.ymm[0] & 0xFFFFFFFFULL);
    const u32 e1 = static_cast<u32>(st.ymm[0] >> 32);
    EXPECT_EQ(e0, 0xFFFFFFFFu) << "element 0 underflow wraps to 0xFFFFFFFF";
    EXPECT_EQ(e1, 0x12345678u)
        << "element 1 must NOT see borrow from element 0 (would be 0x12345677)";
}

// ============================================================================
// VPSRAD — packed arithmetic-shift-right of 32-bit dwords. The "A" in
// SRAD means *arithmetic*: vacated high bits get filled with the sign
// bit, not zero. This is the distinguishing test from a hypothetical
// VPSRLD (logical right shift, zero-fill).
// ============================================================================

// Positive values: arithmetic right shift looks the same as logical
// (sign bit is 0). Basic sanity for the imm8 plumbing.
TEST_F(CpuRuntimeTest, Vpsrad_PositiveValues_ShiftRight) {
    // vpsrad xmm0, xmm1, 2  — c5 f9 72 e1 02 (5 bytes)
    const u8 program[] = {
        0xc5, 0xf9, 0x72, 0xe1, 0x02,
        0xc3,
    };
    std::memcpy(mem.CodePtr(), program, sizeof(program));
    u8* guest_rsp = mem.StackTop() - 8;
    *reinterpret_cast<u64*>(guest_rsp) = kReturnSentinel;
    GuestState st{};
    st.rip = reinterpret_cast<u64>(mem.CodePtr());
    st.gpr[4] = reinterpret_cast<u64>(guest_rsp);
    // xmm1: [40, 100, 1000, 0x40000000]
    st.ymm[4] = (static_cast<u64>(100u) << 32) | 40u;
    st.ymm[5] = (static_cast<u64>(0x40000000u) << 32) | 1000u;
    st.ymm[2] = 0xDEADBEEFDEADBEEFULL;
    st.ymm[3] = 0xDEADBEEFDEADBEEFULL;

    Runtime rt;
    rt.Run(st);
    // Each element shifted right by 2: 40>>2=10, 100>>2=25, 1000>>2=250, 0x40000000>>2=0x10000000
    EXPECT_EQ(static_cast<u32>(st.ymm[0] & 0xFFFFFFFFULL), 10u);
    EXPECT_EQ(static_cast<u32>(st.ymm[0] >> 32), 25u);
    EXPECT_EQ(static_cast<u32>(st.ymm[1] & 0xFFFFFFFFULL), 250u);
    EXPECT_EQ(static_cast<u32>(st.ymm[1] >> 32), 0x10000000u);
    EXPECT_EQ(st.ymm[2], 0ULL);
    EXPECT_EQ(st.ymm[3], 0ULL);
}

// The "arithmetic" semantic: negative values get sign-extended.
// VPSRLD (logical) would zero-fill, producing very different
// values. This test pins down that VPSRAD sign-fills.
TEST_F(CpuRuntimeTest, Vpsrad_NegativeValues_SignFill) {
    // vpsrad xmm0, xmm1, 4
    const u8 program[] = {
        0xc5, 0xf9, 0x72, 0xe1, 0x04,
        0xc3,
    };
    std::memcpy(mem.CodePtr(), program, sizeof(program));
    u8* guest_rsp = mem.StackTop() - 8;
    *reinterpret_cast<u64*>(guest_rsp) = kReturnSentinel;
    GuestState st{};
    st.rip = reinterpret_cast<u64>(mem.CodePtr());
    st.gpr[4] = reinterpret_cast<u64>(guest_rsp);
    // xmm1: [-16, -1, 0xFFFFFFF0 (-16), 0x80000000 (INT32_MIN)]
    // Use bit_cast for clarity.
    const u32 neg16 = static_cast<u32>(-16);
    const u32 neg1  = static_cast<u32>(-1);
    st.ymm[4] = (static_cast<u64>(neg1) << 32) | neg16;
    st.ymm[5] = (static_cast<u64>(0x80000000u) << 32) | neg16;

    Runtime rt;
    rt.Run(st);
    // Sign-fill expectations:
    //   -16 >> 4 = -1  (0xFFFFFFFF), arith
    //   -1  >> 4 = -1  (0xFFFFFFFF), still all-ones
    //   -16 >> 4 = -1  (0xFFFFFFFF)
    //   0x80000000 >> 4 = 0xF8000000  (high bit propagates to top 4 bits)
    EXPECT_EQ(static_cast<u32>(st.ymm[0] & 0xFFFFFFFFULL), 0xFFFFFFFFu)
        << "-16 >>arith 4 = -1 (sign-extended)";
    EXPECT_EQ(static_cast<u32>(st.ymm[0] >> 32), 0xFFFFFFFFu)
        << "-1 >>arith 4 = -1";
    EXPECT_EQ(static_cast<u32>(st.ymm[1] & 0xFFFFFFFFULL), 0xFFFFFFFFu);
    EXPECT_EQ(static_cast<u32>(st.ymm[1] >> 32), 0xF8000000u)
        << "INT32_MIN >>arith 4 = 0xF8000000 (top 4 bits = sign-extension)";
}

// Shift count > 31 saturates per element to sign(src1[i]):
// negative → all-ones, non-negative → zero. Common idiom for
// generating a "sign mask" used elsewhere in the pipeline.
TEST_F(CpuRuntimeTest, Vpsrad_ShiftCountClampedAt31) {
    // vpsrad xmm0, xmm1, 100  (> 31)
    const u8 program[] = {
        0xc5, 0xf9, 0x72, 0xe1, 0x64,
        0xc3,
    };
    std::memcpy(mem.CodePtr(), program, sizeof(program));
    u8* guest_rsp = mem.StackTop() - 8;
    *reinterpret_cast<u64*>(guest_rsp) = kReturnSentinel;
    GuestState st{};
    st.rip = reinterpret_cast<u64>(mem.CodePtr());
    st.gpr[4] = reinterpret_cast<u64>(guest_rsp);
    // xmm1: [0x7FFFFFFF, 0x80000000, 0x00000001, 0xFFFFFFFF]
    st.ymm[4] = (static_cast<u64>(0x80000000u) << 32) | 0x7FFFFFFFu;
    st.ymm[5] = (static_cast<u64>(0xFFFFFFFFu) << 32) | 0x00000001u;

    Runtime rt;
    rt.Run(st);
    // Expected: large shift → sign-mask per element
    //   0x7FFFFFFF positive → 0
    //   0x80000000 negative → 0xFFFFFFFF
    //   0x00000001 positive → 0
    //   0xFFFFFFFF negative → 0xFFFFFFFF
    EXPECT_EQ(static_cast<u32>(st.ymm[0] & 0xFFFFFFFFULL), 0u);
    EXPECT_EQ(static_cast<u32>(st.ymm[0] >> 32), 0xFFFFFFFFu);
    EXPECT_EQ(static_cast<u32>(st.ymm[1] & 0xFFFFFFFFULL), 0u);
    EXPECT_EQ(static_cast<u32>(st.ymm[1] >> 32), 0xFFFFFFFFu);
}

// ============================================================================
// VPADDD — packed add of 32-bit dwords. Wired through EmitVecIntArith
// alongside VPSUBD via the second-instance refactor.
// ============================================================================

TEST_F(CpuRuntimeTest, Vpaddd_BasicPerElementAdd) {
    // vpaddd xmm0, xmm1, xmm2  — c5 f1 fe c2
    const u8 program[] = {
        0xc5, 0xf1, 0xfe, 0xc2,
        0xc3,
    };
    std::memcpy(mem.CodePtr(), program, sizeof(program));
    u8* guest_rsp = mem.StackTop() - 8;
    *reinterpret_cast<u64*>(guest_rsp) = kReturnSentinel;
    GuestState st{};
    st.rip = reinterpret_cast<u64>(mem.CodePtr());
    st.gpr[4] = reinterpret_cast<u64>(guest_rsp);
    // xmm1: [10, 20, 30, 40]
    st.ymm[4] = (static_cast<u64>(20u) << 32) | 10u;
    st.ymm[5] = (static_cast<u64>(40u) << 32) | 30u;
    // xmm2: [100, 200, 300, 400]
    st.ymm[8] = (static_cast<u64>(200u) << 32) | 100u;
    st.ymm[9] = (static_cast<u64>(400u) << 32) | 300u;
    st.ymm[2] = 0xDEADBEEFDEADBEEFULL;
    st.ymm[3] = 0xDEADBEEFDEADBEEFULL;

    Runtime rt;
    rt.Run(st);
    // Expected: [110, 220, 330, 440]
    EXPECT_EQ(static_cast<u32>(st.ymm[0] & 0xFFFFFFFFULL), 110u);
    EXPECT_EQ(static_cast<u32>(st.ymm[0] >> 32), 220u);
    EXPECT_EQ(static_cast<u32>(st.ymm[1] & 0xFFFFFFFFULL), 330u);
    EXPECT_EQ(static_cast<u32>(st.ymm[1] >> 32), 440u);
    EXPECT_EQ(st.ymm[2], 0ULL);
    EXPECT_EQ(st.ymm[3], 0ULL);
}

// Carry-isolation: 32-bit additions must NOT propagate carry into the
// next element. The addition counterpart of VPSUBD's borrow-isolation
// check.
// element 0: 0xFFFFFFFF + 0x00000001 = 0x00000000 (wraparound)
// element 1: 0x00000000 + 0x12345678 = 0x12345678 (must NOT increment!)
TEST_F(CpuRuntimeTest, Vpaddd_CarryDoesNotCrossElementBoundary) {
    const u8 program[] = {
        0xc5, 0xf1, 0xfe, 0xc2, // vpaddd xmm0, xmm1, xmm2
        0xc3,
    };
    std::memcpy(mem.CodePtr(), program, sizeof(program));
    u8* guest_rsp = mem.StackTop() - 8;
    *reinterpret_cast<u64*>(guest_rsp) = kReturnSentinel;
    GuestState st{};
    st.rip = reinterpret_cast<u64>(mem.CodePtr());
    st.gpr[4] = reinterpret_cast<u64>(guest_rsp);
    // xmm1 chunk 0: e1 = 0x00000000, e0 = 0xFFFFFFFF
    st.ymm[4] = 0x00000000FFFFFFFFULL;
    // xmm2 chunk 0: e1 = 0x12345678, e0 = 0x00000001
    st.ymm[8] = 0x1234567800000001ULL;
    st.ymm[5] = st.ymm[9] = 0;

    Runtime rt;
    rt.Run(st);
    const u32 e0 = static_cast<u32>(st.ymm[0] & 0xFFFFFFFFULL);
    const u32 e1 = static_cast<u32>(st.ymm[0] >> 32);
    EXPECT_EQ(e0, 0u) << "element 0 overflow wraps to 0";
    EXPECT_EQ(e1, 0x12345678u)
        << "element 1 must NOT see carry from element 0 (would be 0x12345679)";
}

// Mem-src form — the 8-byte length case observed from the game's log.
TEST_F(CpuRuntimeTest, Vpaddd_MemSource) {
    u32* mem_op = reinterpret_cast<u32*>(mem.CodePtr() + 0x100);
    mem_op[0] = 1u;
    mem_op[1] = 2u;
    mem_op[2] = 3u;
    mem_op[3] = 4u;

    const u8 program[] = {
        0x48, 0xb8, 0,0,0,0,0,0,0,0,
        0xc5, 0xf1, 0xfe, 0x00,  // vpaddd xmm0, xmm1, xmmword[rax]
        0xc3,
    };
    u8 prog[sizeof(program)];
    std::memcpy(prog, program, sizeof(program));
    const u64 mem_addr = reinterpret_cast<u64>(mem_op);
    std::memcpy(prog + 2, &mem_addr, sizeof(mem_addr));
    std::memcpy(mem.CodePtr(), prog, sizeof(prog));
    u8* guest_rsp = mem.StackTop() - 8;
    *reinterpret_cast<u64*>(guest_rsp) = kReturnSentinel;
    GuestState st{};
    st.rip = reinterpret_cast<u64>(mem.CodePtr());
    st.gpr[4] = reinterpret_cast<u64>(guest_rsp);
    // xmm1: [100, 200, 300, 400]
    st.ymm[4] = (static_cast<u64>(200u) << 32) | 100u;
    st.ymm[5] = (static_cast<u64>(400u) << 32) | 300u;

    Runtime rt;
    rt.Run(st);
    // Expected: [101, 202, 303, 404]
    EXPECT_EQ(static_cast<u32>(st.ymm[0] & 0xFFFFFFFFULL), 101u);
    EXPECT_EQ(static_cast<u32>(st.ymm[0] >> 32), 202u);
    EXPECT_EQ(static_cast<u32>(st.ymm[1] & 0xFFFFFFFFULL), 303u);
    EXPECT_EQ(static_cast<u32>(st.ymm[1] >> 32), 404u);
}

// ============================================================================
// VDIVPS — packed single-precision float divide. First instance of
// vector FP arithmetic. Tests cover the basic per-element division,
// the non-commutativity invariant (src1/src2 order matters), and
// IEEE-754 special-value behavior (div-by-zero produces inf, not a trap).
// ============================================================================

TEST_F(CpuRuntimeTest, Vdivps_BasicPerElementDivide) {
    // vdivps xmm0, xmm1, xmm2  — c5 f0 5e c2
    const u8 program[] = {
        0xc5, 0xf0, 0x5e, 0xc2,
        0xc3,
    };
    std::memcpy(mem.CodePtr(), program, sizeof(program));
    u8* guest_rsp = mem.StackTop() - 8;
    *reinterpret_cast<u64*>(guest_rsp) = kReturnSentinel;
    GuestState st{};
    st.rip = reinterpret_cast<u64>(mem.CodePtr());
    st.gpr[4] = reinterpret_cast<u64>(guest_rsp);
    // xmm1: [12.0, 100.0, 25.0, 7.5]
    st.ymm[4] = (static_cast<u64>(std::bit_cast<u32>(100.0f)) << 32)
              |  static_cast<u64>(std::bit_cast<u32>(12.0f));
    st.ymm[5] = (static_cast<u64>(std::bit_cast<u32>(7.5f)) << 32)
              |  static_cast<u64>(std::bit_cast<u32>(25.0f));
    // xmm2: [4.0, 5.0, 5.0, 2.5]
    st.ymm[8] = (static_cast<u64>(std::bit_cast<u32>(5.0f)) << 32)
              |  static_cast<u64>(std::bit_cast<u32>(4.0f));
    st.ymm[9] = (static_cast<u64>(std::bit_cast<u32>(2.5f)) << 32)
              |  static_cast<u64>(std::bit_cast<u32>(5.0f));
    st.ymm[2] = 0xDEADBEEFDEADBEEFULL;
    st.ymm[3] = 0xDEADBEEFDEADBEEFULL;

    Runtime rt;
    rt.Run(st);
    // Expected: [3.0, 20.0, 5.0, 3.0]
    auto fetch_f = [&](int chunk, bool hi) {
        const u32 bits = hi ? static_cast<u32>(st.ymm[chunk] >> 32)
                            : static_cast<u32>(st.ymm[chunk] & 0xFFFFFFFFULL);
        float v; std::memcpy(&v, &bits, sizeof(v));
        return v;
    };
    EXPECT_EQ(fetch_f(0, false), 3.0f);
    EXPECT_EQ(fetch_f(0, true),  20.0f);
    EXPECT_EQ(fetch_f(1, false), 5.0f);
    EXPECT_EQ(fetch_f(1, true),  3.0f);
    EXPECT_EQ(st.ymm[2], 0ULL);
    EXPECT_EQ(st.ymm[3], 0ULL);
}

// Non-commutativity: 10/2 = 5, not 0.2 — catches src1/src2 swap bugs
// at every lane simultaneously. The float counterpart of the
// VSUBSS/VDIVSS scalar-FP order checks.
TEST_F(CpuRuntimeTest, Vdivps_OperandOrderCheck) {
    const u8 program[] = {
        0xc5, 0xf0, 0x5e, 0xc2, // vdivps xmm0, xmm1, xmm2
        0xc3,
    };
    std::memcpy(mem.CodePtr(), program, sizeof(program));
    u8* guest_rsp = mem.StackTop() - 8;
    *reinterpret_cast<u64*>(guest_rsp) = kReturnSentinel;
    GuestState st{};
    st.rip = reinterpret_cast<u64>(mem.CodePtr());
    st.gpr[4] = reinterpret_cast<u64>(guest_rsp);
    // xmm1 (src1): 10.0 everywhere
    const u32 ten = std::bit_cast<u32>(10.0f);
    st.ymm[4] = (static_cast<u64>(ten) << 32) | ten;
    st.ymm[5] = (static_cast<u64>(ten) << 32) | ten;
    // xmm2 (src2): 2.0 everywhere
    const u32 two = std::bit_cast<u32>(2.0f);
    st.ymm[8] = (static_cast<u64>(two) << 32) | two;
    st.ymm[9] = (static_cast<u64>(two) << 32) | two;

    Runtime rt;
    rt.Run(st);
    const u32 e0 = static_cast<u32>(st.ymm[0] & 0xFFFFFFFFULL);
    float result;
    std::memcpy(&result, &e0, sizeof(result));
    EXPECT_EQ(result, 5.0f) << "src1 / src2 ordering: 10/2 = 5 (not 0.2)";
    EXPECT_EQ(st.ymm[0], (static_cast<u64>(std::bit_cast<u32>(5.0f)) << 32)
                       |  static_cast<u64>(std::bit_cast<u32>(5.0f)))
        << "all 4 elements should match — 10/2 is identical across lanes";
}

// IEEE-754 special-value behavior: division by zero produces ±inf
// (positive numerator → +inf, negative → -inf, zero → NaN), no trap.
// Confirms the host op's FP semantics propagate cleanly through the
// scratch-XMM relay without us inadvertently masking them.
TEST_F(CpuRuntimeTest, Vdivps_DivByZeroProducesInfNotNaN) {
    const u8 program[] = {
        0xc5, 0xf0, 0x5e, 0xc2,
        0xc3,
    };
    std::memcpy(mem.CodePtr(), program, sizeof(program));
    u8* guest_rsp = mem.StackTop() - 8;
    *reinterpret_cast<u64*>(guest_rsp) = kReturnSentinel;
    GuestState st{};
    st.rip = reinterpret_cast<u64>(mem.CodePtr());
    st.gpr[4] = reinterpret_cast<u64>(guest_rsp);
    // xmm1: [+1.0, -1.0, 0.0, +1.0]
    st.ymm[4] = (static_cast<u64>(std::bit_cast<u32>(-1.0f)) << 32)
              |  static_cast<u64>(std::bit_cast<u32>(1.0f));
    st.ymm[5] = (static_cast<u64>(std::bit_cast<u32>(1.0f)) << 32)
              |  static_cast<u64>(std::bit_cast<u32>(0.0f));
    // xmm2: [0.0, 0.0, 0.0, 1.0]
    st.ymm[8] = (static_cast<u64>(std::bit_cast<u32>(0.0f)) << 32)
              |  static_cast<u64>(std::bit_cast<u32>(0.0f));
    st.ymm[9] = (static_cast<u64>(std::bit_cast<u32>(1.0f)) << 32)
              |  static_cast<u64>(std::bit_cast<u32>(0.0f));

    Runtime rt;
    rt.Run(st);
    // Expected (IEEE-754):
    //   1.0 / 0.0 = +inf
    //  -1.0 / 0.0 = -inf
    //   0.0 / 0.0 = NaN
    //   1.0 / 1.0 = 1.0
    auto fetch_f = [&](int chunk, bool hi) {
        const u32 bits = hi ? static_cast<u32>(st.ymm[chunk] >> 32)
                            : static_cast<u32>(st.ymm[chunk] & 0xFFFFFFFFULL);
        float v; std::memcpy(&v, &bits, sizeof(v));
        return v;
    };
    EXPECT_EQ(fetch_f(0, false), std::numeric_limits<float>::infinity())
        << "1.0 / 0.0 = +inf";
    EXPECT_EQ(fetch_f(0, true), -std::numeric_limits<float>::infinity())
        << "-1.0 / 0.0 = -inf";
    EXPECT_TRUE(std::isnan(fetch_f(1, false))) << "0.0 / 0.0 = NaN";
    EXPECT_EQ(fetch_f(1, true), 1.0f);
}

// ============================================================================
// VBROADCASTSS — broadcast a single 32-bit float from memory to all
// vector lanes. AVX1 baseline (mem-source) only — AVX2 xmm-source
// form is not supported because the CPUID spoof doesn't advertise AVX2.
// ============================================================================

// 128-bit form: 4 lanes all receive the same value, upper YMM zeroed.
TEST_F(CpuRuntimeTest, Vbroadcastss_128_BroadcastsToAllFourLanes) {
    float* src_value = reinterpret_cast<float*>(mem.CodePtr() + 0x100);
    *src_value = 3.14159f;

    // vbroadcastss xmm0, dword[rax]
    // c4 e2 79 18 00  (5 bytes for short addressing; we use mov-rax+modrm)
    const u8 program[] = {
        0x48, 0xb8, 0,0,0,0,0,0,0,0,           // mov rax, <src_value>
        0xc4, 0xe2, 0x79, 0x18, 0x00,           // vbroadcastss xmm0, dword[rax]
        0xc3,
    };
    u8 prog[sizeof(program)];
    std::memcpy(prog, program, sizeof(program));
    const u64 src_addr = reinterpret_cast<u64>(src_value);
    std::memcpy(prog + 2, &src_addr, sizeof(src_addr));
    std::memcpy(mem.CodePtr(), prog, sizeof(prog));
    u8* guest_rsp = mem.StackTop() - 8;
    *reinterpret_cast<u64*>(guest_rsp) = kReturnSentinel;
    GuestState st{};
    st.rip = reinterpret_cast<u64>(mem.CodePtr());
    st.gpr[4] = reinterpret_cast<u64>(guest_rsp);
    // Pre-pollute all 4 chunks to verify zeroing of upper YMM.
    st.ymm[0] = 0xDEADBEEFDEADBEEFULL;
    st.ymm[1] = 0xDEADBEEFDEADBEEFULL;
    st.ymm[2] = 0xDEADBEEFDEADBEEFULL;
    st.ymm[3] = 0xDEADBEEFDEADBEEFULL;

    Runtime rt;
    rt.Run(st);

    const u32 v_bits = std::bit_cast<u32>(3.14159f);
    const u64 expected_chunk = (static_cast<u64>(v_bits) << 32) | v_bits;
    EXPECT_EQ(st.ymm[0], expected_chunk) << "lanes 0 and 1 both = 3.14159f";
    EXPECT_EQ(st.ymm[1], expected_chunk) << "lanes 2 and 3 both = 3.14159f";
    EXPECT_EQ(st.ymm[2], 0ULL) << "VEX-128 zeros upper YMM";
    EXPECT_EQ(st.ymm[3], 0ULL);
}

// 256-bit form: 8 lanes all receive the same value, no upper zero.
TEST_F(CpuRuntimeTest, Vbroadcastss_256_BroadcastsToAllEightLanes) {
    float* src_value = reinterpret_cast<float*>(mem.CodePtr() + 0x100);
    *src_value = -42.5f;

    // vbroadcastss ymm0, dword[rax]  — c4 e2 7d 18 00
    const u8 program[] = {
        0x48, 0xb8, 0,0,0,0,0,0,0,0,
        0xc4, 0xe2, 0x7d, 0x18, 0x00,
        0xc3,
    };
    u8 prog[sizeof(program)];
    std::memcpy(prog, program, sizeof(program));
    const u64 src_addr = reinterpret_cast<u64>(src_value);
    std::memcpy(prog + 2, &src_addr, sizeof(src_addr));
    std::memcpy(mem.CodePtr(), prog, sizeof(prog));
    u8* guest_rsp = mem.StackTop() - 8;
    *reinterpret_cast<u64*>(guest_rsp) = kReturnSentinel;
    GuestState st{};
    st.rip = reinterpret_cast<u64>(mem.CodePtr());
    st.gpr[4] = reinterpret_cast<u64>(guest_rsp);

    Runtime rt;
    rt.Run(st);
    const u32 v_bits = std::bit_cast<u32>(-42.5f);
    const u64 expected_chunk = (static_cast<u64>(v_bits) << 32) | v_bits;
    EXPECT_EQ(st.ymm[0], expected_chunk);
    EXPECT_EQ(st.ymm[1], expected_chunk);
    EXPECT_EQ(st.ymm[2], expected_chunk) << "256-bit form fills upper YMM lanes too";
    EXPECT_EQ(st.ymm[3], expected_chunk);
}

// ============================================================================
// VBLENDVPS — variable-mask blend. Per-element selector comes from the
// sign bit of the mask register, NOT from an imm8 (that's the VBLENDPS
// constant-mask sibling). Tests pin down: (1) per-element selection,
// (2) ONLY the sign bit matters (other bits of mask element are
// ignored), and (3) upper YMM zeroing for the 128-bit form.
// ============================================================================

// Mixed-pattern mask: alternating signs across 4 elements selects
// alternating sources. This catches mask-bit/source-source wiring
// errors.
TEST_F(CpuRuntimeTest, Vblendvps_AlternatingMask) {
    // vblendvps xmm0, xmm1, xmm2, xmm3
    // c4 e3 71 4a c2 30  (6 bytes)
    const u8 program[] = {
        0xc4, 0xe3, 0x71, 0x4a, 0xc2, 0x30,
        0xc3,
    };
    std::memcpy(mem.CodePtr(), program, sizeof(program));
    u8* guest_rsp = mem.StackTop() - 8;
    *reinterpret_cast<u64*>(guest_rsp) = kReturnSentinel;
    GuestState st{};
    st.rip = reinterpret_cast<u64>(mem.CodePtr());
    st.gpr[4] = reinterpret_cast<u64>(guest_rsp);
    // src1 = xmm1 (lane 4): [10.0, 20.0, 30.0, 40.0]  — picked when mask sign=0
    st.ymm[4] = (static_cast<u64>(std::bit_cast<u32>(20.0f)) << 32)
              |  static_cast<u64>(std::bit_cast<u32>(10.0f));
    st.ymm[5] = (static_cast<u64>(std::bit_cast<u32>(40.0f)) << 32)
              |  static_cast<u64>(std::bit_cast<u32>(30.0f));
    // src2 = xmm2 (lane 8): [100.0, 200.0, 300.0, 400.0]  — picked when mask sign=1
    st.ymm[8] = (static_cast<u64>(std::bit_cast<u32>(200.0f)) << 32)
              |  static_cast<u64>(std::bit_cast<u32>(100.0f));
    st.ymm[9] = (static_cast<u64>(std::bit_cast<u32>(400.0f)) << 32)
              |  static_cast<u64>(std::bit_cast<u32>(300.0f));
    // mask = xmm3 (lane 12): sign bits [1, 0, 1, 0]
    //   element 0 = 0x80000000 (sign=1) → src2 = 100
    //   element 1 = 0x00000000 (sign=0) → src1 = 20
    //   element 2 = 0x80000001 (sign=1, other bits arbitrary) → src2 = 300
    //   element 3 = 0x7FFFFFFF (sign=0, other bits arbitrary) → src1 = 40
    st.ymm[12] = (static_cast<u64>(0x00000000u) << 32) | 0x80000000u;
    st.ymm[13] = (static_cast<u64>(0x7FFFFFFFu) << 32) | 0x80000001u;

    Runtime rt;
    rt.Run(st);
    auto fetch_f = [&](int chunk, bool hi) {
        const u32 bits = hi ? static_cast<u32>(st.ymm[chunk] >> 32)
                            : static_cast<u32>(st.ymm[chunk] & 0xFFFFFFFFULL);
        float v; std::memcpy(&v, &bits, sizeof(v));
        return v;
    };
    EXPECT_EQ(fetch_f(0, false), 100.0f) << "mask[0].sign=1 → src2";
    EXPECT_EQ(fetch_f(0, true),  20.0f)  << "mask[1].sign=0 → src1";
    EXPECT_EQ(fetch_f(1, false), 300.0f) << "mask[2].sign=1 → src2";
    EXPECT_EQ(fetch_f(1, true),  40.0f)  << "mask[3].sign=0 → src1";
    EXPECT_EQ(st.ymm[2], 0ULL) << "VEX-128 zeros upper YMM";
    EXPECT_EQ(st.ymm[3], 0ULL);
}

// Only the SIGN BIT of each mask element matters; other bits don't.
// Element values like 0x7FFFFFFF (max positive int32) must select
// src1, NOT src2 — even though most bits are set. Likewise
// 0x80000000 (only sign bit, all others zero) must select src2.
//
// Catches accidental "is mask element non-zero" implementations
// instead of "is mask element sign bit set" — which would be a
// VPTEST-style mask, not a VBLENDV-style mask.
TEST_F(CpuRuntimeTest, Vblendvps_OnlySignBitMatters) {
    const u8 program[] = {
        0xc4, 0xe3, 0x71, 0x4a, 0xc2, 0x30,
        0xc3,
    };
    std::memcpy(mem.CodePtr(), program, sizeof(program));
    u8* guest_rsp = mem.StackTop() - 8;
    *reinterpret_cast<u64*>(guest_rsp) = kReturnSentinel;
    GuestState st{};
    st.rip = reinterpret_cast<u64>(mem.CodePtr());
    st.gpr[4] = reinterpret_cast<u64>(guest_rsp);
    // src1 (xmm1) = all 1.0; src2 (xmm2) = all 2.0
    const u32 one = std::bit_cast<u32>(1.0f);
    const u32 two = std::bit_cast<u32>(2.0f);
    st.ymm[4] = (static_cast<u64>(one) << 32) | one;
    st.ymm[5] = (static_cast<u64>(one) << 32) | one;
    st.ymm[8] = (static_cast<u64>(two) << 32) | two;
    st.ymm[9] = (static_cast<u64>(two) << 32) | two;
    // mask values exercising the "only the sign bit matters" rule:
    //   [0x7FFFFFFF (sign=0, all others=1) → 1.0,
    //    0x80000000 (sign=1, all others=0) → 2.0,
    //    0x00000001 (sign=0, low bit set)   → 1.0,
    //    0xFFFFFFFF (sign=1, all others=1) → 2.0]
    st.ymm[12] = (static_cast<u64>(0x80000000u) << 32) | 0x7FFFFFFFu;
    st.ymm[13] = (static_cast<u64>(0xFFFFFFFFu) << 32) | 0x00000001u;

    Runtime rt;
    rt.Run(st);
    auto fetch_f = [&](int chunk, bool hi) {
        const u32 bits = hi ? static_cast<u32>(st.ymm[chunk] >> 32)
                            : static_cast<u32>(st.ymm[chunk] & 0xFFFFFFFFULL);
        float v; std::memcpy(&v, &bits, sizeof(v));
        return v;
    };
    EXPECT_EQ(fetch_f(0, false), 1.0f) << "mask=0x7FFFFFFF (sign=0) → src1 (1.0)";
    EXPECT_EQ(fetch_f(0, true),  2.0f) << "mask=0x80000000 (sign=1) → src2 (2.0)";
    EXPECT_EQ(fetch_f(1, false), 1.0f) << "mask=0x00000001 (sign=0) → src1 (1.0)";
    EXPECT_EQ(fetch_f(1, true),  2.0f) << "mask=0xFFFFFFFF (sign=1) → src2 (2.0)";
}

// ============================================================================
// VMULPS — packed single-precision multiply. Wired through the new
// EmitVecFpArith scaffolding alongside VDIVPS via the second-instance
// refactor.
// ============================================================================

TEST_F(CpuRuntimeTest, Vmulps_BasicPerElementMul_RegSrc) {
    // vmulps xmm0, xmm1, xmm2  — c5 f0 59 c2
    const u8 program[] = {
        0xc5, 0xf0, 0x59, 0xc2,
        0xc3,
    };
    std::memcpy(mem.CodePtr(), program, sizeof(program));
    u8* guest_rsp = mem.StackTop() - 8;
    *reinterpret_cast<u64*>(guest_rsp) = kReturnSentinel;
    GuestState st{};
    st.rip = reinterpret_cast<u64>(mem.CodePtr());
    st.gpr[4] = reinterpret_cast<u64>(guest_rsp);
    // xmm1: [2.0, 3.0, 4.0, 5.0]
    st.ymm[4] = (static_cast<u64>(std::bit_cast<u32>(3.0f)) << 32)
              |  static_cast<u64>(std::bit_cast<u32>(2.0f));
    st.ymm[5] = (static_cast<u64>(std::bit_cast<u32>(5.0f)) << 32)
              |  static_cast<u64>(std::bit_cast<u32>(4.0f));
    // xmm2: [10.0, 10.0, 10.0, 10.0]
    const u32 ten = std::bit_cast<u32>(10.0f);
    st.ymm[8] = (static_cast<u64>(ten) << 32) | ten;
    st.ymm[9] = (static_cast<u64>(ten) << 32) | ten;
    st.ymm[2] = 0xDEADBEEFDEADBEEFULL;
    st.ymm[3] = 0xDEADBEEFDEADBEEFULL;

    Runtime rt;
    rt.Run(st);
    auto fetch_f = [&](int chunk, bool hi) {
        const u32 bits = hi ? static_cast<u32>(st.ymm[chunk] >> 32)
                            : static_cast<u32>(st.ymm[chunk] & 0xFFFFFFFFULL);
        float v; std::memcpy(&v, &bits, sizeof(v));
        return v;
    };
    EXPECT_EQ(fetch_f(0, false), 20.0f);
    EXPECT_EQ(fetch_f(0, true),  30.0f);
    EXPECT_EQ(fetch_f(1, false), 40.0f);
    EXPECT_EQ(fetch_f(1, true),  50.0f);
    EXPECT_EQ(st.ymm[2], 0ULL);
    EXPECT_EQ(st.ymm[3], 0ULL);
}

// Mem-src form (length 8 = RIP-rel mem-src). The form actually observed
// in the user's log. Validates the mem-src branch of the refactored helper.
TEST_F(CpuRuntimeTest, Vmulps_MemSource) {
    float* mem_op = reinterpret_cast<float*>(mem.CodePtr() + 0x100);
    mem_op[0] = 0.5f;
    mem_op[1] = 0.25f;
    mem_op[2] = -1.0f;
    mem_op[3] = 2.0f;

    const u8 program[] = {
        0x48, 0xb8, 0,0,0,0,0,0,0,0,
        0xc5, 0xf0, 0x59, 0x00,  // vmulps xmm0, xmm1, xmmword[rax]
        0xc3,
    };
    u8 prog[sizeof(program)];
    std::memcpy(prog, program, sizeof(program));
    const u64 mem_addr = reinterpret_cast<u64>(mem_op);
    std::memcpy(prog + 2, &mem_addr, sizeof(mem_addr));
    std::memcpy(mem.CodePtr(), prog, sizeof(prog));
    u8* guest_rsp = mem.StackTop() - 8;
    *reinterpret_cast<u64*>(guest_rsp) = kReturnSentinel;
    GuestState st{};
    st.rip = reinterpret_cast<u64>(mem.CodePtr());
    st.gpr[4] = reinterpret_cast<u64>(guest_rsp);
    // xmm1: [100.0, 200.0, 300.0, 400.0]
    st.ymm[4] = (static_cast<u64>(std::bit_cast<u32>(200.0f)) << 32)
              |  static_cast<u64>(std::bit_cast<u32>(100.0f));
    st.ymm[5] = (static_cast<u64>(std::bit_cast<u32>(400.0f)) << 32)
              |  static_cast<u64>(std::bit_cast<u32>(300.0f));

    Runtime rt;
    rt.Run(st);
    auto fetch_f = [&](int chunk, bool hi) {
        const u32 bits = hi ? static_cast<u32>(st.ymm[chunk] >> 32)
                            : static_cast<u32>(st.ymm[chunk] & 0xFFFFFFFFULL);
        float v; std::memcpy(&v, &bits, sizeof(v));
        return v;
    };
    // Expected: [50.0, 50.0, -300.0, 800.0]
    EXPECT_EQ(fetch_f(0, false), 50.0f);
    EXPECT_EQ(fetch_f(0, true),  50.0f);
    EXPECT_EQ(fetch_f(1, false), -300.0f);
    EXPECT_EQ(fetch_f(1, true),  800.0f);
}

// ============================================================================
// VCMPPS — per-element FP compare with imm8 predicate. Produces a
// mask (all-1s for true, all-0s for false) per element. Tests cover:
// basic LT predicate, NaN unordered handling, and the combined
// cmp+blendvps idiom (since the user's log shows VBLENDVPS earlier).
// ============================================================================

// Predicate 1 = LT (less-than, ordered, non-signaling).
//   src1[i] < src2[i] && neither NaN → 0xFFFFFFFF
//   otherwise                        → 0x00000000
TEST_F(CpuRuntimeTest, Vcmpps_LessThan_BasicPattern) {
    // vcmpps xmm0, xmm1, xmm2, 0x01  — c5 f0 c2 c2 01
    const u8 program[] = {
        0xc5, 0xf0, 0xc2, 0xc2, 0x01,
        0xc3,
    };
    std::memcpy(mem.CodePtr(), program, sizeof(program));
    u8* guest_rsp = mem.StackTop() - 8;
    *reinterpret_cast<u64*>(guest_rsp) = kReturnSentinel;
    GuestState st{};
    st.rip = reinterpret_cast<u64>(mem.CodePtr());
    st.gpr[4] = reinterpret_cast<u64>(guest_rsp);
    // xmm1 (src1): [1.0, 2.0, 3.0, 4.0]
    st.ymm[4] = (static_cast<u64>(std::bit_cast<u32>(2.0f)) << 32)
              |  static_cast<u64>(std::bit_cast<u32>(1.0f));
    st.ymm[5] = (static_cast<u64>(std::bit_cast<u32>(4.0f)) << 32)
              |  static_cast<u64>(std::bit_cast<u32>(3.0f));
    // xmm2 (src2): [2.0, 2.0, 2.0, 2.0]
    const u32 two = std::bit_cast<u32>(2.0f);
    st.ymm[8] = (static_cast<u64>(two) << 32) | two;
    st.ymm[9] = (static_cast<u64>(two) << 32) | two;
    st.ymm[2] = 0xDEADBEEFDEADBEEFULL;
    st.ymm[3] = 0xDEADBEEFDEADBEEFULL;

    Runtime rt;
    rt.Run(st);
    // Expected: [1<2 → -1, 2<2 → 0, 3<2 → 0, 4<2 → 0]
    EXPECT_EQ(static_cast<u32>(st.ymm[0] & 0xFFFFFFFFULL), 0xFFFFFFFFu)
        << "1.0 < 2.0 → all-ones";
    EXPECT_EQ(static_cast<u32>(st.ymm[0] >> 32), 0u)
        << "2.0 < 2.0 (equal, NOT less) → zero";
    EXPECT_EQ(static_cast<u32>(st.ymm[1] & 0xFFFFFFFFULL), 0u);
    EXPECT_EQ(static_cast<u32>(st.ymm[1] >> 32), 0u);
    EXPECT_EQ(st.ymm[2], 0ULL) << "VEX-128 zeros upper YMM";
    EXPECT_EQ(st.ymm[3], 0ULL);
}

// Predicate 1 (LT_OS, ordered): any NaN operand makes the comparison
// false (unordered → not less-than). Distinguishes from predicates
// 5 (NLT, "not less than"), which returns TRUE for NaN.
TEST_F(CpuRuntimeTest, Vcmpps_LessThan_NaN_IsFalse) {
    const u8 program[] = {
        0xc5, 0xf0, 0xc2, 0xc2, 0x01, // vcmpps xmm0, xmm1, xmm2, LT
        0xc3,
    };
    std::memcpy(mem.CodePtr(), program, sizeof(program));
    u8* guest_rsp = mem.StackTop() - 8;
    *reinterpret_cast<u64*>(guest_rsp) = kReturnSentinel;
    GuestState st{};
    st.rip = reinterpret_cast<u64>(mem.CodePtr());
    st.gpr[4] = reinterpret_cast<u64>(guest_rsp);
    const float quiet_nan = std::numeric_limits<float>::quiet_NaN();
    // xmm1: [NaN,  1.0,  NaN, 5.0]
    st.ymm[4] = (static_cast<u64>(std::bit_cast<u32>(1.0f)) << 32)
              |  static_cast<u64>(std::bit_cast<u32>(quiet_nan));
    st.ymm[5] = (static_cast<u64>(std::bit_cast<u32>(5.0f)) << 32)
              |  static_cast<u64>(std::bit_cast<u32>(quiet_nan));
    // xmm2: [10.0, NaN, NaN, 10.0]
    st.ymm[8] = (static_cast<u64>(std::bit_cast<u32>(quiet_nan)) << 32)
              |  static_cast<u64>(std::bit_cast<u32>(10.0f));
    st.ymm[9] = (static_cast<u64>(std::bit_cast<u32>(10.0f)) << 32)
              |  static_cast<u64>(std::bit_cast<u32>(quiet_nan));

    Runtime rt;
    rt.Run(st);
    // Expected — ordered LT requires BOTH operands be non-NaN:
    //   NaN  < 10 → false (NaN in src1) → 0
    //   1.0  < NaN → false (NaN in src2) → 0
    //   NaN  < NaN → false             → 0
    //   5.0  < 10 → true               → 0xFFFFFFFF
    EXPECT_EQ(static_cast<u32>(st.ymm[0] & 0xFFFFFFFFULL), 0u) << "NaN < 10 = unordered = 0";
    EXPECT_EQ(static_cast<u32>(st.ymm[0] >> 32), 0u)         << "1 < NaN = unordered = 0";
    EXPECT_EQ(static_cast<u32>(st.ymm[1] & 0xFFFFFFFFULL), 0u) << "NaN < NaN = unordered = 0";
    EXPECT_EQ(static_cast<u32>(st.ymm[1] >> 32), 0xFFFFFFFFu) << "5 < 10 = ordered true = -1";
}

// ============================================================================
// VPCMPGTD — packed signed compare greater-than for 32-bit ints.
// Routed through EmitVecIntCmp alongside the EQ family via the
// second-instance refactor.
// ============================================================================

TEST_F(CpuRuntimeTest, Vpcmpgtd_RegSrc_PerElementGreaterThan) {
    // vpcmpgtd xmm0, xmm1, xmm2  — c5 f1 66 c2
    const u8 program[] = {
        0xc5, 0xf1, 0x66, 0xc2,
        0xc3,
    };
    std::memcpy(mem.CodePtr(), program, sizeof(program));
    u8* guest_rsp = mem.StackTop() - 8;
    *reinterpret_cast<u64*>(guest_rsp) = kReturnSentinel;
    GuestState st{};
    st.rip = reinterpret_cast<u64>(mem.CodePtr());
    st.gpr[4] = reinterpret_cast<u64>(guest_rsp);
    // xmm1 (src1): [10, 20, 5, 100]
    st.ymm[4] = (static_cast<u64>(20u) << 32) | 10u;
    st.ymm[5] = (static_cast<u64>(100u) << 32) | 5u;
    // xmm2 (src2): [5, 20, 10, 50]
    st.ymm[8] = (static_cast<u64>(20u) << 32) | 5u;
    st.ymm[9] = (static_cast<u64>(50u) << 32) | 10u;
    st.ymm[2] = 0xDEADBEEFDEADBEEFULL;
    st.ymm[3] = 0xDEADBEEFDEADBEEFULL;

    Runtime rt;
    rt.Run(st);
    // Expected: [10>5 → -1, 20>20 → 0 (equal, NOT greater), 5>10 → 0, 100>50 → -1]
    EXPECT_EQ(static_cast<u32>(st.ymm[0] & 0xFFFFFFFFULL), 0xFFFFFFFFu);
    EXPECT_EQ(static_cast<u32>(st.ymm[0] >> 32), 0u) << "equal is NOT greater";
    EXPECT_EQ(static_cast<u32>(st.ymm[1] & 0xFFFFFFFFULL), 0u);
    EXPECT_EQ(static_cast<u32>(st.ymm[1] >> 32), 0xFFFFFFFFu);
    EXPECT_EQ(st.ymm[2], 0ULL);
    EXPECT_EQ(st.ymm[3], 0ULL);
}

// VPCMPGTD is SIGNED greater-than. Negative values are *less than*
// positive ones, even though their unsigned representations would
// make them "larger". Catches accidental routing to a hypothetical
// unsigned variant.
TEST_F(CpuRuntimeTest, Vpcmpgtd_SignedSemantics) {
    const u8 program[] = {
        0xc5, 0xf1, 0x66, 0xc2, // vpcmpgtd xmm0, xmm1, xmm2
        0xc3,
    };
    std::memcpy(mem.CodePtr(), program, sizeof(program));
    u8* guest_rsp = mem.StackTop() - 8;
    *reinterpret_cast<u64*>(guest_rsp) = kReturnSentinel;
    GuestState st{};
    st.rip = reinterpret_cast<u64>(mem.CodePtr());
    st.gpr[4] = reinterpret_cast<u64>(guest_rsp);
    // src1: [1, 0xFFFFFFFF (-1 signed),  0x7FFFFFFF, 0x80000000 (INT32_MIN)]
    // src2: [-1, 0,                       0xFFFFFFFF (-1),  -1]
    st.ymm[4] = (static_cast<u64>(0xFFFFFFFFu) << 32) | 0x00000001u;
    st.ymm[5] = (static_cast<u64>(0x80000000u) << 32) | 0x7FFFFFFFu;
    st.ymm[8] = (static_cast<u64>(0x00000000u) << 32) | 0xFFFFFFFFu;
    st.ymm[9] = (static_cast<u64>(0xFFFFFFFFu) << 32) | 0xFFFFFFFFu;

    Runtime rt;
    rt.Run(st);
    // Expected (signed semantics):
    //   1   >  -1     → true (-1)
    //   -1  >   0     → false (0)            ; unsigned would say "true"!
    //   INT_MAX > -1  → true (-1)
    //   INT_MIN > -1  → false (0)            ; unsigned would say "true"!
    EXPECT_EQ(static_cast<u32>(st.ymm[0] & 0xFFFFFFFFULL), 0xFFFFFFFFu)
        << "1 > -1 (signed) = true";
    EXPECT_EQ(static_cast<u32>(st.ymm[0] >> 32), 0u)
        << "-1 > 0 (signed) = false (NOT 0xFFFFFFFF > 0)";
    EXPECT_EQ(static_cast<u32>(st.ymm[1] & 0xFFFFFFFFULL), 0xFFFFFFFFu);
    EXPECT_EQ(static_cast<u32>(st.ymm[1] >> 32), 0u)
        << "INT_MIN > -1 (signed) = false (NOT 0x80000000 > 0xFFFFFFFF)";
}

// Mem-src form — the 8-byte length case from the user's log.
TEST_F(CpuRuntimeTest, Vpcmpgtd_MemSource) {
    s32* mem_op = reinterpret_cast<s32*>(mem.CodePtr() + 0x100);
    mem_op[0] = 50;
    mem_op[1] = 100;
    mem_op[2] = 200;
    mem_op[3] = -1;

    const u8 program[] = {
        0x48, 0xb8, 0,0,0,0,0,0,0,0,
        0xc5, 0xf1, 0x66, 0x00,    // vpcmpgtd xmm0, xmm1, xmmword[rax]
        0xc3,
    };
    u8 prog[sizeof(program)];
    std::memcpy(prog, program, sizeof(program));
    const u64 mem_addr = reinterpret_cast<u64>(mem_op);
    std::memcpy(prog + 2, &mem_addr, sizeof(mem_addr));
    std::memcpy(mem.CodePtr(), prog, sizeof(prog));
    u8* guest_rsp = mem.StackTop() - 8;
    *reinterpret_cast<u64*>(guest_rsp) = kReturnSentinel;
    GuestState st{};
    st.rip = reinterpret_cast<u64>(mem.CodePtr());
    st.gpr[4] = reinterpret_cast<u64>(guest_rsp);
    // xmm1: [100, 100, 100, 100]
    st.ymm[4] = (static_cast<u64>(100u) << 32) | 100u;
    st.ymm[5] = (static_cast<u64>(100u) << 32) | 100u;

    Runtime rt;
    rt.Run(st);
    // Expected: [100>50 → -1, 100>100 → 0, 100>200 → 0, 100>-1 → -1]
    EXPECT_EQ(static_cast<u32>(st.ymm[0] & 0xFFFFFFFFULL), 0xFFFFFFFFu);
    EXPECT_EQ(static_cast<u32>(st.ymm[0] >> 32), 0u);
    EXPECT_EQ(static_cast<u32>(st.ymm[1] & 0xFFFFFFFFULL), 0u);
    EXPECT_EQ(static_cast<u32>(st.ymm[1] >> 32), 0xFFFFFFFFu) << "100 > -1 (signed)";
}

// ============================================================================
// VPMULLD — packed signed multiply of 32-bit dwords, low half only.
// Per-element result = (src1[i] * src2[i]) & 0xFFFFFFFF (the upper 32
// bits of the 64-bit product are discarded). Routed through the
// EmitVecIntArith helper alongside Add/Sub. Tests cover: basic
// per-element multiply, the wrap-on-overflow semantic (the L in
// MULL), and the mem-src form (length 9 = RIP-rel + 0x0F38 opcode).
// ============================================================================

TEST_F(CpuRuntimeTest, Vpmulld_BasicPerElementMultiply) {
    // vpmulld xmm0, xmm1, xmm2  — c4 e2 71 40 c2  (5 bytes, 3-byte VEX)
    const u8 program[] = {
        0xc4, 0xe2, 0x71, 0x40, 0xc2,
        0xc3,
    };
    std::memcpy(mem.CodePtr(), program, sizeof(program));
    u8* guest_rsp = mem.StackTop() - 8;
    *reinterpret_cast<u64*>(guest_rsp) = kReturnSentinel;
    GuestState st{};
    st.rip = reinterpret_cast<u64>(mem.CodePtr());
    st.gpr[4] = reinterpret_cast<u64>(guest_rsp);
    // xmm1: [3, 7, 100, 256]
    st.ymm[4] = (static_cast<u64>(7u) << 32) | 3u;
    st.ymm[5] = (static_cast<u64>(256u) << 32) | 100u;
    // xmm2: [4, 5, 10, 64]
    st.ymm[8] = (static_cast<u64>(5u) << 32) | 4u;
    st.ymm[9] = (static_cast<u64>(64u) << 32) | 10u;
    st.ymm[2] = 0xDEADBEEFDEADBEEFULL;
    st.ymm[3] = 0xDEADBEEFDEADBEEFULL;

    Runtime rt;
    rt.Run(st);
    // Expected: [12, 35, 1000, 16384]
    EXPECT_EQ(static_cast<u32>(st.ymm[0] & 0xFFFFFFFFULL), 12u);
    EXPECT_EQ(static_cast<u32>(st.ymm[0] >> 32), 35u);
    EXPECT_EQ(static_cast<u32>(st.ymm[1] & 0xFFFFFFFFULL), 1000u);
    EXPECT_EQ(static_cast<u32>(st.ymm[1] >> 32), 16384u);
    EXPECT_EQ(st.ymm[2], 0ULL);
    EXPECT_EQ(st.ymm[3], 0ULL);
}

// The "low half" semantic: 0x10000 × 0x10000 = 0x100000000 (>32 bits),
// the upper 32 bits are dropped → result 0x00000000.
// More interestingly: (-1) * (-1) = +1, but only by coincidence
// (the 64-bit product 0xFFFFFFFFFFFFFFFF & 0xFFFFFFFF = 0xFFFFFFFF... wait no.
//  -1 * -1 = 1 in signed math. Bit pattern 0xFFFFFFFF * 0xFFFFFFFF (unsigned)
//  = 0xFFFFFFFE00000001. Low 32 = 0x00000001 = +1. So signed and unsigned
//  multiply-low produce IDENTICAL bit patterns for any inputs — this is
//  the reason VPMULL{W,D} has no signed/unsigned distinction.)
TEST_F(CpuRuntimeTest, Vpmulld_WrapOnOverflow) {
    const u8 program[] = {
        0xc4, 0xe2, 0x71, 0x40, 0xc2, // vpmulld xmm0, xmm1, xmm2
        0xc3,
    };
    std::memcpy(mem.CodePtr(), program, sizeof(program));
    u8* guest_rsp = mem.StackTop() - 8;
    *reinterpret_cast<u64*>(guest_rsp) = kReturnSentinel;
    GuestState st{};
    st.rip = reinterpret_cast<u64>(mem.CodePtr());
    st.gpr[4] = reinterpret_cast<u64>(guest_rsp);
    // src1: [0x10000, 0xFFFFFFFF (-1 signed), 0x80000000 (INT_MIN), 0x40000000]
    // src2: [0x10000, 0xFFFFFFFF (-1),         0x00000002,            0x00000004]
    st.ymm[4] = (static_cast<u64>(0xFFFFFFFFu) << 32) | 0x00010000u;
    st.ymm[5] = (static_cast<u64>(0x40000000u) << 32) | 0x80000000u;
    st.ymm[8] = (static_cast<u64>(0xFFFFFFFFu) << 32) | 0x00010000u;
    st.ymm[9] = (static_cast<u64>(0x00000004u) << 32) | 0x00000002u;

    Runtime rt;
    rt.Run(st);
    // Expected (low 32 of the full 64-bit product):
    //   0x00010000 * 0x00010000 = 0x100000000 → low 32 = 0x00000000
    //   0xFFFFFFFF * 0xFFFFFFFF = 0xFFFFFFFE00000001 → low 32 = 0x00000001
    //   0x80000000 * 0x00000002 = 0x100000000 → low 32 = 0x00000000
    //   0x40000000 * 0x00000004 = 0x100000000 → low 32 = 0x00000000
    EXPECT_EQ(static_cast<u32>(st.ymm[0] & 0xFFFFFFFFULL), 0x00000000u)
        << "0x10000 * 0x10000 wraps to 0";
    EXPECT_EQ(static_cast<u32>(st.ymm[0] >> 32), 0x00000001u)
        << "(-1) * (-1) = +1 (consistent across signed/unsigned)";
    EXPECT_EQ(static_cast<u32>(st.ymm[1] & 0xFFFFFFFFULL), 0x00000000u)
        << "INT_MIN * 2 wraps to 0";
    EXPECT_EQ(static_cast<u32>(st.ymm[1] >> 32), 0x00000000u);
}

// Mem-src form — the 9-byte length case from the user's log.
TEST_F(CpuRuntimeTest, Vpmulld_MemSource) {
    s32* mem_op = reinterpret_cast<s32*>(mem.CodePtr() + 0x100);
    mem_op[0] = 6;
    mem_op[1] = 7;
    mem_op[2] = 8;
    mem_op[3] = 9;

    const u8 program[] = {
        0x48, 0xb8, 0,0,0,0,0,0,0,0,
        0xc4, 0xe2, 0x71, 0x40, 0x00, // vpmulld xmm0, xmm1, xmmword[rax]
        0xc3,
    };
    u8 prog[sizeof(program)];
    std::memcpy(prog, program, sizeof(program));
    const u64 mem_addr = reinterpret_cast<u64>(mem_op);
    std::memcpy(prog + 2, &mem_addr, sizeof(mem_addr));
    std::memcpy(mem.CodePtr(), prog, sizeof(prog));
    u8* guest_rsp = mem.StackTop() - 8;
    *reinterpret_cast<u64*>(guest_rsp) = kReturnSentinel;
    GuestState st{};
    st.rip = reinterpret_cast<u64>(mem.CodePtr());
    st.gpr[4] = reinterpret_cast<u64>(guest_rsp);
    // xmm1: [2, 3, 5, 11]
    st.ymm[4] = (static_cast<u64>(3u) << 32) | 2u;
    st.ymm[5] = (static_cast<u64>(11u) << 32) | 5u;

    Runtime rt;
    rt.Run(st);
    // Expected: [12, 21, 40, 99]
    EXPECT_EQ(static_cast<u32>(st.ymm[0] & 0xFFFFFFFFULL), 12u);
    EXPECT_EQ(static_cast<u32>(st.ymm[0] >> 32), 21u);
    EXPECT_EQ(static_cast<u32>(st.ymm[1] & 0xFFFFFFFFULL), 40u);
    EXPECT_EQ(static_cast<u32>(st.ymm[1] >> 32), 99u);
}

// ============================================================================
// VINSERTF128 — insert 128 bits into one half of a 256-bit ymm,
// keeping the other half from src1. imm8[0] picks which half:
// 0 = into low (replaces low128), 1 = into high (replaces high128).
// ============================================================================

// imm8 = 0: dst.low128 = src2, dst.high128 = src1.high128
TEST_F(CpuRuntimeTest, Vinsertf128_Imm0_InsertsIntoLowHalf) {
    // vinsertf128 ymm0, ymm1, xmm2, 0  — c4 e3 75 18 c2 00  (6 bytes)
    const u8 program[] = {
        0xc4, 0xe3, 0x75, 0x18, 0xc2, 0x00,
        0xc3,
    };
    std::memcpy(mem.CodePtr(), program, sizeof(program));
    u8* guest_rsp = mem.StackTop() - 8;
    *reinterpret_cast<u64*>(guest_rsp) = kReturnSentinel;
    GuestState st{};
    st.rip = reinterpret_cast<u64>(mem.CodePtr());
    st.gpr[4] = reinterpret_cast<u64>(guest_rsp);
    // src1 = ymm1 (lane 4): distinguishable across all 4 chunks
    st.ymm[4] = 0x1111111111111111ULL;  // low128.lo64
    st.ymm[5] = 0x2222222222222222ULL;  // low128.hi64
    st.ymm[6] = 0x3333333333333333ULL;  // high128.lo64
    st.ymm[7] = 0x4444444444444444ULL;  // high128.hi64
    // src2 = xmm2 (lane 8): the 128-bit value to insert
    st.ymm[8] = 0xAAAAAAAAAAAAAAAAULL;
    st.ymm[9] = 0xBBBBBBBBBBBBBBBBULL;
    // pre-pollute dst
    st.ymm[0] = 0xFFFFFFFFFFFFFFFFULL;
    st.ymm[1] = 0xFFFFFFFFFFFFFFFFULL;
    st.ymm[2] = 0xFFFFFFFFFFFFFFFFULL;
    st.ymm[3] = 0xFFFFFFFFFFFFFFFFULL;

    Runtime rt;
    rt.Run(st);
    // Expected with imm8=0:
    //   dst.low128 = src2 (lanes 0,1 = A,B)
    //   dst.high128 = src1.high128 (lanes 2,3 = src1.high128 = 3,4)
    EXPECT_EQ(st.ymm[0], 0xAAAAAAAAAAAAAAAAULL) << "dst.lo64 = src2.lo64";
    EXPECT_EQ(st.ymm[1], 0xBBBBBBBBBBBBBBBBULL) << "dst.lo64+8 = src2.hi64";
    EXPECT_EQ(st.ymm[2], 0x3333333333333333ULL) << "dst.hi128.lo64 = src1.hi128.lo64";
    EXPECT_EQ(st.ymm[3], 0x4444444444444444ULL) << "dst.hi128.hi64 = src1.hi128.hi64";
}

// imm8 = 1: dst.low128 = src1.low128, dst.high128 = src2
TEST_F(CpuRuntimeTest, Vinsertf128_Imm1_InsertsIntoHighHalf) {
    // vinsertf128 ymm0, ymm1, xmm2, 1
    const u8 program[] = {
        0xc4, 0xe3, 0x75, 0x18, 0xc2, 0x01,
        0xc3,
    };
    std::memcpy(mem.CodePtr(), program, sizeof(program));
    u8* guest_rsp = mem.StackTop() - 8;
    *reinterpret_cast<u64*>(guest_rsp) = kReturnSentinel;
    GuestState st{};
    st.rip = reinterpret_cast<u64>(mem.CodePtr());
    st.gpr[4] = reinterpret_cast<u64>(guest_rsp);
    st.ymm[4] = 0x1111111111111111ULL;
    st.ymm[5] = 0x2222222222222222ULL;
    st.ymm[6] = 0x3333333333333333ULL;
    st.ymm[7] = 0x4444444444444444ULL;
    st.ymm[8] = 0xAAAAAAAAAAAAAAAAULL;
    st.ymm[9] = 0xBBBBBBBBBBBBBBBBULL;

    Runtime rt;
    rt.Run(st);
    // Expected with imm8=1:
    //   dst.low128 = src1.low128 (lanes 0,1 = 1,2)
    //   dst.high128 = src2 (lanes 2,3 = A,B)
    EXPECT_EQ(st.ymm[0], 0x1111111111111111ULL) << "dst.lo128 = src1.lo128";
    EXPECT_EQ(st.ymm[1], 0x2222222222222222ULL);
    EXPECT_EQ(st.ymm[2], 0xAAAAAAAAAAAAAAAAULL) << "dst.hi128 = src2";
    EXPECT_EQ(st.ymm[3], 0xBBBBBBBBBBBBBBBBULL);
}

// ============================================================================
// VCVTDQ2PS — convert packed signed int32 to packed float32. Per-element
// IEEE-754 conversion, MXCSR rounding for large magnitudes. Tests cover:
// exact small values, signed conversion (negative ints), and inexact
// rounding when magnitude exceeds 2^24 (float's mantissa limit).
// ============================================================================

// Small positive integers (≤ 2^24) convert exactly to their float
// counterparts.
TEST_F(CpuRuntimeTest, Vcvtdq2ps_SmallPositive_ExactConversion) {
    // vcvtdq2ps xmm0, xmm1  — c5 f8 5b c1
    const u8 program[] = {
        0xc5, 0xf8, 0x5b, 0xc1,
        0xc3,
    };
    std::memcpy(mem.CodePtr(), program, sizeof(program));
    u8* guest_rsp = mem.StackTop() - 8;
    *reinterpret_cast<u64*>(guest_rsp) = kReturnSentinel;
    GuestState st{};
    st.rip = reinterpret_cast<u64>(mem.CodePtr());
    st.gpr[4] = reinterpret_cast<u64>(guest_rsp);
    // xmm1: [1, 100, 16777216 (2^24), 12345]
    st.ymm[4] = (static_cast<u64>(100u) << 32) | 1u;
    st.ymm[5] = (static_cast<u64>(12345u) << 32) | 16777216u;
    st.ymm[2] = 0xDEADBEEFDEADBEEFULL;
    st.ymm[3] = 0xDEADBEEFDEADBEEFULL;

    Runtime rt;
    rt.Run(st);
    auto fetch_f = [&](int chunk, bool hi) {
        const u32 bits = hi ? static_cast<u32>(st.ymm[chunk] >> 32)
                            : static_cast<u32>(st.ymm[chunk] & 0xFFFFFFFFULL);
        float v; std::memcpy(&v, &bits, sizeof(v));
        return v;
    };
    EXPECT_EQ(fetch_f(0, false), 1.0f);
    EXPECT_EQ(fetch_f(0, true),  100.0f);
    EXPECT_EQ(fetch_f(1, false), 16777216.0f) << "2^24 fits exactly (boundary)";
    EXPECT_EQ(fetch_f(1, true),  12345.0f);
    EXPECT_EQ(st.ymm[2], 0ULL);
    EXPECT_EQ(st.ymm[3], 0ULL);
}

// Negative integers: VCVTDQ2PS treats input as SIGNED int32.
// 0xFFFFFFFF → -1.0f, not 4294967295.0f.
TEST_F(CpuRuntimeTest, Vcvtdq2ps_NegativeIntegers_SignedConversion) {
    const u8 program[] = {
        0xc5, 0xf8, 0x5b, 0xc1, // vcvtdq2ps xmm0, xmm1
        0xc3,
    };
    std::memcpy(mem.CodePtr(), program, sizeof(program));
    u8* guest_rsp = mem.StackTop() - 8;
    *reinterpret_cast<u64*>(guest_rsp) = kReturnSentinel;
    GuestState st{};
    st.rip = reinterpret_cast<u64>(mem.CodePtr());
    st.gpr[4] = reinterpret_cast<u64>(guest_rsp);
    // xmm1: [-1, -100, INT32_MIN (0x80000000), -16777216]
    const u32 neg1 = static_cast<u32>(-1);
    const u32 neg100 = static_cast<u32>(-100);
    const u32 neg2_24 = static_cast<u32>(-16777216);
    st.ymm[4] = (static_cast<u64>(neg100) << 32) | neg1;
    st.ymm[5] = (static_cast<u64>(neg2_24) << 32) | 0x80000000u;

    Runtime rt;
    rt.Run(st);
    auto fetch_f = [&](int chunk, bool hi) {
        const u32 bits = hi ? static_cast<u32>(st.ymm[chunk] >> 32)
                            : static_cast<u32>(st.ymm[chunk] & 0xFFFFFFFFULL);
        float v; std::memcpy(&v, &bits, sizeof(v));
        return v;
    };
    EXPECT_EQ(fetch_f(0, false), -1.0f) << "0xFFFFFFFF = -1 signed, NOT 4294967295";
    EXPECT_EQ(fetch_f(0, true), -100.0f);
    EXPECT_EQ(fetch_f(1, false), -2147483648.0f)
        << "INT32_MIN converts to -2^31 (exact as float)";
    EXPECT_EQ(fetch_f(1, true), -16777216.0f);
}

// Large magnitude beyond 2^24: float can't hold all 31 bits of
// significand exactly, so VCVTDQ2PS rounds under MXCSR. The exact
// rounded value matches host conversion, which uses the same MXCSR
// as our JIT (default round-to-nearest-even).
TEST_F(CpuRuntimeTest, Vcvtdq2ps_LargeMagnitude_InexactRounding) {
    const u8 program[] = {
        0xc5, 0xf8, 0x5b, 0xc1,
        0xc3,
    };
    std::memcpy(mem.CodePtr(), program, sizeof(program));
    u8* guest_rsp = mem.StackTop() - 8;
    *reinterpret_cast<u64*>(guest_rsp) = kReturnSentinel;
    GuestState st{};
    st.rip = reinterpret_cast<u64>(mem.CodePtr());
    st.gpr[4] = reinterpret_cast<u64>(guest_rsp);
    // xmm1: [0x7FFFFFFF (INT_MAX), 0x01000001 (2^24+1, edge),
    //         0x12345678,            0x76543210]
    st.ymm[4] = (static_cast<u64>(0x01000001u) << 32) | 0x7FFFFFFFu;
    st.ymm[5] = (static_cast<u64>(0x76543210u) << 32) | 0x12345678u;

    Runtime rt;
    rt.Run(st);
    auto fetch_f = [&](int chunk, bool hi) {
        const u32 bits = hi ? static_cast<u32>(st.ymm[chunk] >> 32)
                            : static_cast<u32>(st.ymm[chunk] & 0xFFFFFFFFULL);
        float v; std::memcpy(&v, &bits, sizeof(v));
        return v;
    };
    // Each result matches what host (int32_t → float) cast produces
    // under default MXCSR (rounding mode = nearest-even).
    EXPECT_EQ(fetch_f(0, false), static_cast<float>(static_cast<s32>(0x7FFFFFFF)))
        << "INT_MAX rounded to nearest representable float";
    EXPECT_EQ(fetch_f(0, true), static_cast<float>(static_cast<s32>(0x01000001)))
        << "2^24+1 (just past exact range) — rounds down to 2^24";
    EXPECT_EQ(fetch_f(1, false), static_cast<float>(static_cast<s32>(0x12345678)));
    EXPECT_EQ(fetch_f(1, true),  static_cast<float>(static_cast<s32>(0x76543210)));
}

// ============================================================================
// VCVTTPS2DQ — packed float32 → int32, TRUNCATING (toward zero,
// ignoring MXCSR rounding mode). The "T" in the mnemonic is the
// distinguishing feature vs the round-under-MXCSR sibling VCVTPS2DQ.
// Routed through EmitVecFpCvt alongside VCVTDQ2PS via the second-
// instance refactor.
// ============================================================================

// Truncation toward zero: positive floats truncate down (3.7 → 3),
// negative truncate up toward zero (-3.7 → -3, NOT -4). This is the
// semantic that differs from MXCSR-default-rounding (which would
// round 3.7 → 4 and -3.7 → -4).
TEST_F(CpuRuntimeTest, Vcvttps2dq_TruncationTowardZero) {
    // vcvttps2dq xmm0, xmm1  — c5 fa 5b c1
    const u8 program[] = {
        0xc5, 0xfa, 0x5b, 0xc1,
        0xc3,
    };
    std::memcpy(mem.CodePtr(), program, sizeof(program));
    u8* guest_rsp = mem.StackTop() - 8;
    *reinterpret_cast<u64*>(guest_rsp) = kReturnSentinel;
    GuestState st{};
    st.rip = reinterpret_cast<u64>(mem.CodePtr());
    st.gpr[4] = reinterpret_cast<u64>(guest_rsp);
    // xmm1: [3.7, -3.7, 0.999, -0.5]
    st.ymm[4] = (static_cast<u64>(std::bit_cast<u32>(-3.7f)) << 32)
              |  static_cast<u64>(std::bit_cast<u32>(3.7f));
    st.ymm[5] = (static_cast<u64>(std::bit_cast<u32>(-0.5f)) << 32)
              |  static_cast<u64>(std::bit_cast<u32>(0.999f));
    st.ymm[2] = 0xDEADBEEFDEADBEEFULL;
    st.ymm[3] = 0xDEADBEEFDEADBEEFULL;

    Runtime rt;
    rt.Run(st);
    // Expected (truncation toward zero):
    //   3.7   → 3
    //  -3.7   → -3   (NOT -4; toward zero, not toward -inf)
    //   0.999 → 0
    //  -0.5   → 0    (NOT -1)
    EXPECT_EQ(static_cast<s32>(st.ymm[0] & 0xFFFFFFFFULL), 3);
    EXPECT_EQ(static_cast<s32>(st.ymm[0] >> 32), -3) << "truncates toward zero, not toward -inf";
    EXPECT_EQ(static_cast<s32>(st.ymm[1] & 0xFFFFFFFFULL), 0);
    EXPECT_EQ(static_cast<s32>(st.ymm[1] >> 32), 0) << "-0.5 truncates to 0, not -1";
    EXPECT_EQ(st.ymm[2], 0ULL);
    EXPECT_EQ(st.ymm[3], 0ULL);
}

// Out-of-range / NaN / inf produces Intel's "integer indefinite"
// value: 0x80000000 (INT32_MIN). This is the canonical sentinel for
// "couldn't represent the result as a signed int32".
TEST_F(CpuRuntimeTest, Vcvttps2dq_OutOfRange_ProducesIndefinite) {
    const u8 program[] = {
        0xc5, 0xfa, 0x5b, 0xc1, // vcvttps2dq xmm0, xmm1
        0xc3,
    };
    std::memcpy(mem.CodePtr(), program, sizeof(program));
    u8* guest_rsp = mem.StackTop() - 8;
    *reinterpret_cast<u64*>(guest_rsp) = kReturnSentinel;
    GuestState st{};
    st.rip = reinterpret_cast<u64>(mem.CodePtr());
    st.gpr[4] = reinterpret_cast<u64>(guest_rsp);
    // xmm1: [+inf, -inf, NaN, 1e10 (way > INT_MAX)]
    const float pos_inf = std::numeric_limits<float>::infinity();
    const float quiet_nan = std::numeric_limits<float>::quiet_NaN();
    st.ymm[4] = (static_cast<u64>(std::bit_cast<u32>(-pos_inf)) << 32)
              |  static_cast<u64>(std::bit_cast<u32>(pos_inf));
    st.ymm[5] = (static_cast<u64>(std::bit_cast<u32>(1e10f)) << 32)
              |  static_cast<u64>(std::bit_cast<u32>(quiet_nan));

    Runtime rt;
    rt.Run(st);
    // All 4 results should be 0x80000000 (integer indefinite).
    EXPECT_EQ(static_cast<u32>(st.ymm[0] & 0xFFFFFFFFULL), 0x80000000u) << "+inf → INT32_MIN";
    EXPECT_EQ(static_cast<u32>(st.ymm[0] >> 32), 0x80000000u)         << "-inf → INT32_MIN";
    EXPECT_EQ(static_cast<u32>(st.ymm[1] & 0xFFFFFFFFULL), 0x80000000u) << "NaN → INT32_MIN";
    EXPECT_EQ(static_cast<u32>(st.ymm[1] >> 32), 0x80000000u)
        << "1e10 (overflow) → INT32_MIN";
}

// ============================================================================
// VEXTRACTF128 — extract 128 bits from one half of a 256-bit ymm
// into either an xmm register (with upper-YMM zeroing) or memory.
// Mirror of VINSERTF128. Tests cover: imm8=0 (low extract), imm8=1
// (high extract), and the memory-destination form.
// ============================================================================

// imm8 = 0: dst = src.lo128
TEST_F(CpuRuntimeTest, Vextractf128_Imm0_ExtractsLow128) {
    // vextractf128 xmm0, ymm1, 0  — c4 e3 7d 19 c8 00  (6 bytes)
    const u8 program[] = {
        0xc4, 0xe3, 0x7d, 0x19, 0xc8, 0x00,
        0xc3,
    };
    std::memcpy(mem.CodePtr(), program, sizeof(program));
    u8* guest_rsp = mem.StackTop() - 8;
    *reinterpret_cast<u64*>(guest_rsp) = kReturnSentinel;
    GuestState st{};
    st.rip = reinterpret_cast<u64>(mem.CodePtr());
    st.gpr[4] = reinterpret_cast<u64>(guest_rsp);
    // src = ymm1 (lane 4)
    st.ymm[4] = 0x1111111111111111ULL;
    st.ymm[5] = 0x2222222222222222ULL;
    st.ymm[6] = 0x3333333333333333ULL;
    st.ymm[7] = 0x4444444444444444ULL;
    // pre-pollute dst upper YMM
    st.ymm[0] = 0xFFFFFFFFFFFFFFFFULL;
    st.ymm[1] = 0xFFFFFFFFFFFFFFFFULL;
    st.ymm[2] = 0xFFFFFFFFFFFFFFFFULL;
    st.ymm[3] = 0xFFFFFFFFFFFFFFFFULL;

    Runtime rt;
    rt.Run(st);
    EXPECT_EQ(st.ymm[0], 0x1111111111111111ULL) << "dst.lo64 from src.lo128.lo64";
    EXPECT_EQ(st.ymm[1], 0x2222222222222222ULL) << "dst.hi64 from src.lo128.hi64";
    EXPECT_EQ(st.ymm[2], 0ULL) << "VEX-128 zeros upper YMM";
    EXPECT_EQ(st.ymm[3], 0ULL);
}

// imm8 = 1: dst = src.hi128
TEST_F(CpuRuntimeTest, Vextractf128_Imm1_ExtractsHigh128) {
    const u8 program[] = {
        0xc4, 0xe3, 0x7d, 0x19, 0xc8, 0x01,
        0xc3,
    };
    std::memcpy(mem.CodePtr(), program, sizeof(program));
    u8* guest_rsp = mem.StackTop() - 8;
    *reinterpret_cast<u64*>(guest_rsp) = kReturnSentinel;
    GuestState st{};
    st.rip = reinterpret_cast<u64>(mem.CodePtr());
    st.gpr[4] = reinterpret_cast<u64>(guest_rsp);
    st.ymm[4] = 0x1111111111111111ULL;
    st.ymm[5] = 0x2222222222222222ULL;
    st.ymm[6] = 0x3333333333333333ULL;
    st.ymm[7] = 0x4444444444444444ULL;

    Runtime rt;
    rt.Run(st);
    EXPECT_EQ(st.ymm[0], 0x3333333333333333ULL) << "dst.lo64 from src.hi128.lo64";
    EXPECT_EQ(st.ymm[1], 0x4444444444444444ULL) << "dst.hi64 from src.hi128.hi64";
    EXPECT_EQ(st.ymm[2], 0ULL);
    EXPECT_EQ(st.ymm[3], 0ULL);
}

// Memory destination: 16 bytes written, surrounding memory untouched.
TEST_F(CpuRuntimeTest, Vextractf128_MemoryDestination) {
    u64* scratch = reinterpret_cast<u64*>(mem.CodePtr() + 0x100);
    scratch[0] = 0xAAAAAAAAAAAAAAAAULL;
    scratch[1] = 0xBBBBBBBBBBBBBBBBULL;
    // Sentinel right after the 16-byte destination, to detect overrun.
    scratch[2] = 0xDEADBEEFDEADBEEFULL;

    // vextractf128 xmmword[rax], ymm1, 1
    const u8 program[] = {
        0x48, 0xb8, 0,0,0,0,0,0,0,0,
        0xc4, 0xe3, 0x7d, 0x19, 0x08, 0x01,
        0xc3,
    };
    u8 prog[sizeof(program)];
    std::memcpy(prog, program, sizeof(program));
    const u64 scratch_addr = reinterpret_cast<u64>(scratch);
    std::memcpy(prog + 2, &scratch_addr, sizeof(scratch_addr));
    std::memcpy(mem.CodePtr(), prog, sizeof(prog));
    u8* guest_rsp = mem.StackTop() - 8;
    *reinterpret_cast<u64*>(guest_rsp) = kReturnSentinel;
    GuestState st{};
    st.rip = reinterpret_cast<u64>(mem.CodePtr());
    st.gpr[4] = reinterpret_cast<u64>(guest_rsp);
    st.ymm[4] = 0x1111111111111111ULL;
    st.ymm[5] = 0x2222222222222222ULL;
    st.ymm[6] = 0x5555555555555555ULL; // src.hi128.lo64
    st.ymm[7] = 0x6666666666666666ULL; // src.hi128.hi64

    Runtime rt;
    rt.Run(st);
    EXPECT_EQ(scratch[0], 0x5555555555555555ULL) << "mem.lo64 = src.hi128.lo64";
    EXPECT_EQ(scratch[1], 0x6666666666666666ULL) << "mem.hi64 = src.hi128.hi64";
    EXPECT_EQ(scratch[2], 0xDEADBEEFDEADBEEFULL)
        << "next qword must be untouched (only 16 bytes written)";
}

// ============================================================================
// VPSRLD — packed Logical Shift Right of 32-bit dwords. Wired through
// EmitVecShiftImm alongside VPSRAD via the second-instance refactor.
// The distinguishing semantic is **zero-fill** of vacated high bits
// (vs VPSRAD's sign-fill). For negative inputs the two produce very
// different bit patterns — this test pins down zero-fill.
// ============================================================================

TEST_F(CpuRuntimeTest, Vpsrld_PositiveValues_ShiftsRight) {
    // vpsrld xmm0, xmm1, 2  — c5 f9 72 d1 02 (5 bytes)
    const u8 program[] = {
        0xc5, 0xf9, 0x72, 0xd1, 0x02,
        0xc3,
    };
    std::memcpy(mem.CodePtr(), program, sizeof(program));
    u8* guest_rsp = mem.StackTop() - 8;
    *reinterpret_cast<u64*>(guest_rsp) = kReturnSentinel;
    GuestState st{};
    st.rip = reinterpret_cast<u64>(mem.CodePtr());
    st.gpr[4] = reinterpret_cast<u64>(guest_rsp);
    // xmm1: [40, 100, 1000, 0x40000000]
    st.ymm[4] = (static_cast<u64>(100u) << 32) | 40u;
    st.ymm[5] = (static_cast<u64>(0x40000000u) << 32) | 1000u;
    st.ymm[2] = 0xDEADBEEFDEADBEEFULL;
    st.ymm[3] = 0xDEADBEEFDEADBEEFULL;

    Runtime rt;
    rt.Run(st);
    // Each element shifted right by 2 (same result as SRA for positives):
    //   40 >> 2 = 10, 100 >> 2 = 25, 1000 >> 2 = 250, 0x40000000 >> 2 = 0x10000000
    EXPECT_EQ(static_cast<u32>(st.ymm[0] & 0xFFFFFFFFULL), 10u);
    EXPECT_EQ(static_cast<u32>(st.ymm[0] >> 32), 25u);
    EXPECT_EQ(static_cast<u32>(st.ymm[1] & 0xFFFFFFFFULL), 250u);
    EXPECT_EQ(static_cast<u32>(st.ymm[1] >> 32), 0x10000000u);
    EXPECT_EQ(st.ymm[2], 0ULL);
    EXPECT_EQ(st.ymm[3], 0ULL);
}

// The "logical" semantic: vacated high bits zero-fill regardless of sign.
// Compare with VPSRAD which would sign-extend. For 0xFFFFFFFF >> 4:
//   SRA: 0xFFFFFFFF (sign bit propagates)
//   SRL: 0x0FFFFFFF (zero-fill from MSB)
// This test exists specifically to catch SRA-vs-SRL routing bugs in the
// refactored EmitVecShiftImm switch.
TEST_F(CpuRuntimeTest, Vpsrld_NegativeValues_ZeroFill_NotSignFill) {
    // vpsrld xmm0, xmm1, 4
    const u8 program[] = {
        0xc5, 0xf9, 0x72, 0xd1, 0x04,
        0xc3,
    };
    std::memcpy(mem.CodePtr(), program, sizeof(program));
    u8* guest_rsp = mem.StackTop() - 8;
    *reinterpret_cast<u64*>(guest_rsp) = kReturnSentinel;
    GuestState st{};
    st.rip = reinterpret_cast<u64>(mem.CodePtr());
    st.gpr[4] = reinterpret_cast<u64>(guest_rsp);
    // xmm1: [0xFFFFFFFF, 0x80000000 (INT_MIN), 0xFEDCBA98, 0x12345678]
    st.ymm[4] = (static_cast<u64>(0x80000000u) << 32) | 0xFFFFFFFFu;
    st.ymm[5] = (static_cast<u64>(0x12345678u) << 32) | 0xFEDCBA98u;

    Runtime rt;
    rt.Run(st);
    // SRL (zero-fill) expectations:
    //   0xFFFFFFFF >>logical 4 = 0x0FFFFFFF (NOT 0xFFFFFFFF as SRA would give)
    //   0x80000000 >>logical 4 = 0x08000000 (NOT 0xF8000000)
    //   0xFEDCBA98 >>logical 4 = 0x0FEDCBA9
    //   0x12345678 >>logical 4 = 0x01234567
    EXPECT_EQ(static_cast<u32>(st.ymm[0] & 0xFFFFFFFFULL), 0x0FFFFFFFu)
        << "0xFFFFFFFF >>logical 4 = 0x0FFFFFFF (zero-fill), NOT 0xFFFFFFFF (sign-fill)";
    EXPECT_EQ(static_cast<u32>(st.ymm[0] >> 32), 0x08000000u)
        << "INT_MIN >>logical 4 = 0x08000000 (zero-fill), NOT 0xF8000000 (sign-fill)";
    EXPECT_EQ(static_cast<u32>(st.ymm[1] & 0xFFFFFFFFULL), 0x0FEDCBA9u);
    EXPECT_EQ(static_cast<u32>(st.ymm[1] >> 32), 0x01234567u);
}

// Large shift count: SRL clamps to 0 (because all bits shift out).
// Compare with SRA which would produce sign-mask. Catches reciprocal
// of the sign-fill regression: if SRA were accidentally routed here,
// negative inputs would yield 0xFFFFFFFF instead of 0.
TEST_F(CpuRuntimeTest, Vpsrld_LargeShiftCount_ClampsToZero) {
    // vpsrld xmm0, xmm1, 100  (count > 31)
    const u8 program[] = {
        0xc5, 0xf9, 0x72, 0xd1, 0x64,
        0xc3,
    };
    std::memcpy(mem.CodePtr(), program, sizeof(program));
    u8* guest_rsp = mem.StackTop() - 8;
    *reinterpret_cast<u64*>(guest_rsp) = kReturnSentinel;
    GuestState st{};
    st.rip = reinterpret_cast<u64>(mem.CodePtr());
    st.gpr[4] = reinterpret_cast<u64>(guest_rsp);
    st.ymm[4] = (static_cast<u64>(0x80000000u) << 32) | 0xFFFFFFFFu;
    st.ymm[5] = (static_cast<u64>(0x00000001u) << 32) | 0x12345678u;

    Runtime rt;
    rt.Run(st);
    // Per Intel SDM: shift count >= element width → all elements 0
    // (independent of sign, unlike SRA which yields sign-mask).
    EXPECT_EQ(static_cast<u32>(st.ymm[0] & 0xFFFFFFFFULL), 0u)
        << "0xFFFFFFFF >>logical 100 = 0 (NOT 0xFFFFFFFF as SRA-clamp would give)";
    EXPECT_EQ(static_cast<u32>(st.ymm[0] >> 32), 0u)
        << "INT_MIN >>logical 100 = 0 (NOT 0xFFFFFFFF)";
    EXPECT_EQ(static_cast<u32>(st.ymm[1] & 0xFFFFFFFFULL), 0u);
    EXPECT_EQ(static_cast<u32>(st.ymm[1] >> 32), 0u);
}

// ============================================================================
// VPBLENDW — per-16-bit-word immediate-mask blend. Routed through
// EmitVecBlendImm alongside VBLENDPS via the second-instance refactor.
// Each of imm8's 8 bits selects word i: bit==0 -> src1 word, bit==1 ->
// src2 word. For the 256-bit form the same imm8 byte is replicated to
// both 128-bit lanes (imm8 does not widen). These tests pin the word
// granularity and the per-bit select direction.
// ============================================================================

// imm8 = 0xAA = 1010 1010: odd-indexed words from src2, even from src1.
TEST_F(CpuRuntimeTest, Vpblendw_AlternatingMask_SelectsWords) {
    // vpblendw xmm0, xmm1, xmm2, 0xAA — c4 e3 71 0e c2 aa (6 bytes)
    const u8 program[] = {
        0xc4, 0xe3, 0x71, 0x0e, 0xc2, 0xaa,
        0xc3,
    };
    std::memcpy(mem.CodePtr(), program, sizeof(program));
    u8* guest_rsp = mem.StackTop() - 8;
    *reinterpret_cast<u64*>(guest_rsp) = kReturnSentinel;
    GuestState st{};
    st.rip = reinterpret_cast<u64>(mem.CodePtr());
    st.gpr[4] = reinterpret_cast<u64>(guest_rsp);
    // src1 = xmm1 (lane 4): all words = 0x1111
    st.ymm[4] = 0x1111111111111111ULL;
    st.ymm[5] = 0x1111111111111111ULL;
    // src2 = xmm2 (lane 8): all words = 0x2222
    st.ymm[8] = 0x2222222222222222ULL;
    st.ymm[9] = 0x2222222222222222ULL;
    // pre-pollute dst upper YMM
    st.ymm[2] = 0xDEADBEEFDEADBEEFULL;
    st.ymm[3] = 0xDEADBEEFDEADBEEFULL;

    Runtime rt;
    rt.Run(st);
    // imm8=0xAA: words 0,2,4,6 from src1 (0x1111); words 1,3,5,7 from
    // src2 (0x2222). Each 64-bit chunk holds 4 words; pattern per chunk
    // (little-endian, word0 in low bits): 0x2222 1111 2222 1111.
    EXPECT_EQ(st.ymm[0], 0x2222111122221111ULL)
        << "low chunk: w0=src1 w1=src2 w2=src1 w3=src2";
    EXPECT_EQ(st.ymm[1], 0x2222111122221111ULL)
        << "high chunk: w4=src1 w5=src2 w6=src1 w7=src2";
    EXPECT_EQ(st.ymm[2], 0ULL) << "VEX-128 zeros upper YMM";
    EXPECT_EQ(st.ymm[3], 0ULL);
}

// imm8 = 0x00: all words from src1. imm8 = 0xFF: all words from src2.
TEST_F(CpuRuntimeTest, Vpblendw_AllZeroAllOne_PickPureSource) {
    // Two instructions: blend with 0x00 into xmm0, then 0xFF into xmm3.
    // vpblendw xmm0, xmm1, xmm2, 0x00
    // vpblendw xmm3, xmm1, xmm2, 0xFF
    const u8 program[] = {
        0xc4, 0xe3, 0x71, 0x0e, 0xc2, 0x00, // -> xmm0 = src1
        0xc4, 0xe3, 0x71, 0x0e, 0xda, 0xff, // -> xmm3 = src2
        0xc3,
    };
    std::memcpy(mem.CodePtr(), program, sizeof(program));
    u8* guest_rsp = mem.StackTop() - 8;
    *reinterpret_cast<u64*>(guest_rsp) = kReturnSentinel;
    GuestState st{};
    st.rip = reinterpret_cast<u64>(mem.CodePtr());
    st.gpr[4] = reinterpret_cast<u64>(guest_rsp);
    st.ymm[4] = 0x1111111111111111ULL; // src1 lo
    st.ymm[5] = 0x1111111111111111ULL; // src1 hi
    st.ymm[8] = 0x2222222222222222ULL; // src2 lo
    st.ymm[9] = 0x2222222222222222ULL; // src2 hi

    Runtime rt;
    rt.Run(st);
    // xmm0 (dst lane 0): imm8=0x00 -> all src1
    EXPECT_EQ(st.ymm[0], 0x1111111111111111ULL) << "imm8=0 -> pure src1";
    EXPECT_EQ(st.ymm[1], 0x1111111111111111ULL);
    // xmm3 (dst lane 12 -> chunks 12,13): imm8=0xFF -> all src2
    EXPECT_EQ(st.ymm[12], 0x2222222222222222ULL) << "imm8=0xFF -> pure src2";
    EXPECT_EQ(st.ymm[13], 0x2222222222222222ULL);
}

// Distinct per-word values verify the exact word lane mapping (not just
// uniform fills). imm8 = 0x03 = 0000 0011: words 0,1 from src2; 2..7 src1.
TEST_F(CpuRuntimeTest, Vpblendw_DistinctWords_ExactLaneMapping) {
    // vpblendw xmm0, xmm1, xmm2, 0x03
    const u8 program[] = {
        0xc4, 0xe3, 0x71, 0x0e, 0xc2, 0x03,
        0xc3,
    };
    std::memcpy(mem.CodePtr(), program, sizeof(program));
    u8* guest_rsp = mem.StackTop() - 8;
    *reinterpret_cast<u64*>(guest_rsp) = kReturnSentinel;
    GuestState st{};
    st.rip = reinterpret_cast<u64>(mem.CodePtr());
    st.gpr[4] = reinterpret_cast<u64>(guest_rsp);
    // src1 words 0..7 = 0xA0..0xA7 (in each 16-bit slot)
    st.ymm[4] = 0x00A300A200A100A0ULL; // words 0,1,2,3
    st.ymm[5] = 0x00A700A600A500A4ULL; // words 4,5,6,7
    // src2 words 0..7 = 0xB0..0xB7
    st.ymm[8] = 0x00B300B200B100B0ULL;
    st.ymm[9] = 0x00B700B600B500B4ULL;

    Runtime rt;
    rt.Run(st);
    // imm8=0x03: word0=src2(B0), word1=src2(B1), word2..7=src1(A2..A7).
    EXPECT_EQ(st.ymm[0], 0x00A300A200B100B0ULL)
        << "w0=B0 w1=B1 (from src2), w2=A2 w3=A3 (from src1)";
    EXPECT_EQ(st.ymm[1], 0x00A700A600A500A4ULL)
        << "w4..w7 all from src1 (imm8 high bits 0)";
}

// ============================================================================
// VPACKUSDW — pack two vectors of 32-bit signed dwords into 16-bit
// UNSIGNED words with saturation. Per 128-bit lane: src1's 4 dwords fill
// the low 4 words, src2's 4 dwords fill the high 4 words; each dword is
// clamped to [0, 0xFFFF] (negative -> 0, >0xFFFF -> 0xFFFF). Wired via
// the EmitVecPack family helper (first instance). Expected values below
// were confirmed against host _mm_packus_epi32.
// ============================================================================

// Layout: src1 -> low 4 words, src2 -> high 4 words (in-range values).
TEST_F(CpuRuntimeTest, Vpackusdw_Layout_Src1LowSrc2High) {
    // vpackusdw xmm0, xmm1, xmm2 — c4 e2 71 2b c2 (5 bytes)
    const u8 program[] = {
        0xc4, 0xe2, 0x71, 0x2b, 0xc2,
        0xc3,
    };
    std::memcpy(mem.CodePtr(), program, sizeof(program));
    u8* guest_rsp = mem.StackTop() - 8;
    *reinterpret_cast<u64*>(guest_rsp) = kReturnSentinel;
    GuestState st{};
    st.rip = reinterpret_cast<u64>(mem.CodePtr());
    st.gpr[4] = reinterpret_cast<u64>(guest_rsp);
    // src1 = xmm1 (lane 4): dwords 0,1,2,3
    st.ymm[4] = (1ULL << 32) | 0ULL;   // dword0=0, dword1=1
    st.ymm[5] = (3ULL << 32) | 2ULL;   // dword2=2, dword3=3
    // src2 = xmm2 (lane 8): dwords 0x10,0x11,0x12,0x13
    st.ymm[8] = (0x11ULL << 32) | 0x10ULL;
    st.ymm[9] = (0x13ULL << 32) | 0x12ULL;
    // pre-pollute dst upper YMM
    st.ymm[2] = 0xDEADBEEFDEADBEEFULL;
    st.ymm[3] = 0xDEADBEEFDEADBEEFULL;

    Runtime rt;
    rt.Run(st);
    // Result words: low = src1 (0,1,2,3), high = src2 (0x10,0x11,0x12,0x13).
    // Packed little-endian: chunk0 = words 0..3, chunk1 = words 4..7.
    // word0=0 word1=1 word2=2 word3=3 -> 0x0003000200010000
    // word4=0x10 word5=0x11 word6=0x12 word7=0x13 -> 0x0013001200110010
    EXPECT_EQ(st.ymm[0], 0x0003000200010000ULL) << "low 4 words from src1";
    EXPECT_EQ(st.ymm[1], 0x0013001200110010ULL) << "high 4 words from src2";
    EXPECT_EQ(st.ymm[2], 0ULL) << "VEX-128 zeros upper YMM";
    EXPECT_EQ(st.ymm[3], 0ULL);
}

// Unsigned saturation: negative -> 0, >0xFFFF -> 0xFFFF, in-range kept.
TEST_F(CpuRuntimeTest, Vpackusdw_UnsignedSaturation) {
    // vpackusdw xmm0, xmm1, xmm2
    const u8 program[] = {
        0xc4, 0xe2, 0x71, 0x2b, 0xc2,
        0xc3,
    };
    std::memcpy(mem.CodePtr(), program, sizeof(program));
    u8* guest_rsp = mem.StackTop() - 8;
    *reinterpret_cast<u64*>(guest_rsp) = kReturnSentinel;
    GuestState st{};
    st.rip = reinterpret_cast<u64>(mem.CodePtr());
    st.gpr[4] = reinterpret_cast<u64>(guest_rsp);
    // src1 dwords: -1, 0x10000, 0xFFFF, 0x8000
    st.ymm[4] = (static_cast<u64>(0x00010000u) << 32) | static_cast<u32>(-1);
    st.ymm[5] = (static_cast<u64>(0x00008000u) << 32) | 0x0000FFFFu;
    // src2 dwords: 0x7FFFFFFF, INT_MIN, 0, 0xFFFE
    st.ymm[8] = (static_cast<u64>(0x80000000u) << 32) | 0x7FFFFFFFu;
    st.ymm[9] = (static_cast<u64>(0x0000FFFEu) << 32) | 0x00000000u;

    Runtime rt;
    rt.Run(st);
    // Host-confirmed saturation:
    //   src1: -1->0, 0x10000->ffff, 0xFFFF->ffff, 0x8000->8000
    //   src2: 0x7FFFFFFF->ffff, INT_MIN->0, 0->0, 0xFFFE->fffe
    // low chunk words 0..3: 0000 ffff ffff 8000 -> 0x8000ffffffff0000
    // high chunk words 4..7: ffff 0000 0000 fffe -> 0xfffe00000000ffff
    EXPECT_EQ(st.ymm[0], 0x8000FFFFFFFF0000ULL)
        << "src1 sat: -1->0, 0x10000->0xFFFF, 0xFFFF->0xFFFF, 0x8000->0x8000";
    EXPECT_EQ(st.ymm[1], 0xFFFE00000000FFFFULL)
        << "src2 sat: INT_MAX->0xFFFF, INT_MIN->0, 0->0, 0xFFFE->0xFFFE";
}

// ============================================================================
// VPEXTRD — extract an imm8-selected 32-bit dword from an xmm into a
// GPR (zero-extended to 64 bits) or memory (exactly 4 bytes). imm8[1:0]
// picks the lane; dword 0 is the low 32 bits.
// ============================================================================

// All four lanes -> GPR, verifying lane select + 64-bit zero-extension.
TEST_F(CpuRuntimeTest, Vpextrd_AllLanes_ToGpr_ZeroExtends) {
    // Four extracts into rax(0), rcx(1), rdx(2), rbx(3), each a
    // different lane:
    //   vpextrd eax, xmm1, 0   c4 e3 79 16 c8 00
    //   vpextrd ecx, xmm1, 1   c4 e3 79 16 c9 01
    //   vpextrd edx, xmm1, 2   c4 e3 79 16 ca 02
    //   vpextrd ebx, xmm1, 3   c4 e3 79 16 cb 03
    const u8 program[] = {
        0xc4, 0xe3, 0x79, 0x16, 0xc8, 0x00,
        0xc4, 0xe3, 0x79, 0x16, 0xc9, 0x01,
        0xc4, 0xe3, 0x79, 0x16, 0xca, 0x02,
        0xc4, 0xe3, 0x79, 0x16, 0xcb, 0x03,
        0xc3,
    };
    std::memcpy(mem.CodePtr(), program, sizeof(program));
    u8* guest_rsp = mem.StackTop() - 8;
    *reinterpret_cast<u64*>(guest_rsp) = kReturnSentinel;
    GuestState st{};
    st.rip = reinterpret_cast<u64>(mem.CodePtr());
    st.gpr[4] = reinterpret_cast<u64>(guest_rsp);
    // src = xmm1 (lane 4): dwords 0x11111111,0x22222222,0x33333333,0x44444444
    st.ymm[4] = (0x22222222ULL << 32) | 0x11111111ULL;
    st.ymm[5] = (0x44444444ULL << 32) | 0x33333333ULL;
    // pre-load destination GPRs with sentinels in the high 32 bits to
    // prove zero-extension actually clears them.
    st.gpr[0] = 0xFFFFFFFF00000000ULL; // rax
    st.gpr[1] = 0xFFFFFFFF00000000ULL; // rcx
    st.gpr[2] = 0xFFFFFFFF00000000ULL; // rdx
    st.gpr[3] = 0xFFFFFFFF00000000ULL; // rbx

    Runtime rt;
    rt.Run(st);
    EXPECT_EQ(st.gpr[0], 0x0000000011111111ULL) << "lane0 -> rax, zero-extended";
    EXPECT_EQ(st.gpr[1], 0x0000000022222222ULL) << "lane1 -> rcx, zero-extended";
    EXPECT_EQ(st.gpr[2], 0x0000000033333333ULL) << "lane2 -> rdx, zero-extended";
    EXPECT_EQ(st.gpr[3], 0x0000000044444444ULL) << "lane3 -> rbx, zero-extended";
}

// Memory destination: exactly 4 bytes written, neighbours untouched.
TEST_F(CpuRuntimeTest, Vpextrd_MemoryDestination_Writes4Bytes) {
    u32* scratch = reinterpret_cast<u32*>(mem.CodePtr() + 0x100);
    scratch[0] = 0xAAAAAAAAu;
    scratch[1] = 0xBBBBBBBBu; // sentinel after the 4-byte target
    // mov rax, scratch ; vpextrd dword[rax], xmm1, 2
    const u8 program[] = {
        0x48, 0xb8, 0,0,0,0,0,0,0,0,
        0xc4, 0xe3, 0x79, 0x16, 0x08, 0x02, // vpextrd [rax], xmm1, 2
        0xc3,
    };
    u8 prog[sizeof(program)];
    std::memcpy(prog, program, sizeof(program));
    const u64 scratch_addr = reinterpret_cast<u64>(scratch);
    std::memcpy(prog + 2, &scratch_addr, sizeof(scratch_addr));
    std::memcpy(mem.CodePtr(), prog, sizeof(prog));
    u8* guest_rsp = mem.StackTop() - 8;
    *reinterpret_cast<u64*>(guest_rsp) = kReturnSentinel;
    GuestState st{};
    st.rip = reinterpret_cast<u64>(mem.CodePtr());
    st.gpr[4] = reinterpret_cast<u64>(guest_rsp);
    st.ymm[4] = (0x22222222ULL << 32) | 0x11111111ULL;
    st.ymm[5] = (0x77777777ULL << 32) | 0x33333333ULL; // lane2 = 0x33333333

    Runtime rt;
    rt.Run(st);
    EXPECT_EQ(scratch[0], 0x33333333u) << "lane2 dword written to memory";
    EXPECT_EQ(scratch[1], 0xBBBBBBBBu) << "next dword untouched (only 4 bytes written)";
}

// ============================================================================
// ROL/ROR (32-bit) — rotate left/right by CL or imm8. Rotates set CF
// (last bit rotated through) and OF (1-bit rotates only) but leave
// SF/ZF/AF/PF UNAFFECTED — a key difference from shifts. Count masks to
// 5 bits. The host-rflags round-trip reproduces all of this. Expected
// values confirmed against host rol/ror.
// ============================================================================

// ROL by CL: rotate correctness + CF capture + 32-bit zero-extension.
TEST_F(CpuRuntimeTest, Rol32_ByCl_RotatesAndSetsCarry) {
    // rol eax, cl  — d3 c0 (length 2)... but log showed length 3 with
    // ops=reg,reg; that is rol r/m32, cl with a modrm. Use the generic
    // encoding rol ecx-target. Here: rol eax, cl = d3 c0.
    const u8 program[] = {
        0xd3, 0xc0, // rol eax, cl
        0xc3,
    };
    std::memcpy(mem.CodePtr(), program, sizeof(program));
    u8* guest_rsp = mem.StackTop() - 8;
    *reinterpret_cast<u64*>(guest_rsp) = kReturnSentinel;
    GuestState st{};
    st.rip = reinterpret_cast<u64>(mem.CodePtr());
    st.gpr[4] = reinterpret_cast<u64>(guest_rsp);
    st.gpr[0] = 0xFFFFFFFF12345678ULL; // eax = 0x12345678, high junk to test ZE
    st.gpr[1] = 4;                     // cl = 4
    st.rflags = 0x202;                 // IF set, arithmetic flags clear

    Runtime rt;
    rt.Run(st);
    // rol 0x12345678 by 4 = 0x23456781, zero-extended.
    EXPECT_EQ(st.gpr[0], 0x0000000023456781ULL) << "rol by 4, zero-extended";
    // CF = last bit rotated out = bit that wrapped = 1 here.
    EXPECT_EQ(st.rflags & 0x1ULL, 0x1ULL) << "CF = last bit rotated through";
}

// Count masks to 5 bits: rol by 36 == rol by 4.
TEST_F(CpuRuntimeTest, Rol32_CountMasksTo5Bits) {
    const u8 program[] = {
        0xd3, 0xc0, // rol eax, cl
        0xc3,
    };
    std::memcpy(mem.CodePtr(), program, sizeof(program));
    u8* guest_rsp = mem.StackTop() - 8;
    *reinterpret_cast<u64*>(guest_rsp) = kReturnSentinel;
    GuestState st{};
    st.rip = reinterpret_cast<u64>(mem.CodePtr());
    st.gpr[4] = reinterpret_cast<u64>(guest_rsp);
    st.gpr[0] = 0x12345678ULL;
    st.gpr[1] = 36;     // 36 & 31 = 4
    st.rflags = 0x202;

    Runtime rt;
    rt.Run(st);
    EXPECT_EQ(st.gpr[0], 0x0000000023456781ULL) << "count 36 masks to 4 -> same as rol 4";
}

// Rotates must NOT disturb SF/ZF/PF (unlike shifts). Pre-set those
// flags and confirm they survive a rotate.
TEST_F(CpuRuntimeTest, Rol32_DoesNotDisturbSZP) {
    const u8 program[] = {
        0xd3, 0xc0, // rol eax, cl
        0xc3,
    };
    std::memcpy(mem.CodePtr(), program, sizeof(program));
    u8* guest_rsp = mem.StackTop() - 8;
    *reinterpret_cast<u64*>(guest_rsp) = kReturnSentinel;
    GuestState st{};
    st.rip = reinterpret_cast<u64>(mem.CodePtr());
    st.gpr[4] = reinterpret_cast<u64>(guest_rsp);
    st.gpr[0] = 0x00000001ULL;
    st.gpr[1] = 1;
    // Pre-set ZF(bit6), SF(bit7), PF(bit2) plus IF(bit9).
    const u64 ZF = 1ULL<<6, SF = 1ULL<<7, PF = 1ULL<<2, IF = 1ULL<<9;
    st.rflags = ZF | SF | PF | IF | 0x2;

    Runtime rt;
    rt.Run(st);
    // 0x1 rol 1 = 0x2.
    EXPECT_EQ(st.gpr[0], 0x2ULL) << "rotate result";
    // ZF/SF/PF must be UNCHANGED by the rotate.
    EXPECT_EQ(st.rflags & ZF, ZF) << "ZF preserved across rotate";
    EXPECT_EQ(st.rflags & SF, SF) << "SF preserved across rotate";
    EXPECT_EQ(st.rflags & PF, PF) << "PF preserved across rotate";
}

// ROR by imm8.
TEST_F(CpuRuntimeTest, Ror32_ByImm) {
    const u8 program[] = {
        0xc1, 0xc8, 0x08, // ror eax, 8
        0xc3,
    };
    std::memcpy(mem.CodePtr(), program, sizeof(program));
    u8* guest_rsp = mem.StackTop() - 8;
    *reinterpret_cast<u64*>(guest_rsp) = kReturnSentinel;
    GuestState st{};
    st.rip = reinterpret_cast<u64>(mem.CodePtr());
    st.gpr[4] = reinterpret_cast<u64>(guest_rsp);
    st.gpr[0] = 0x12345678ULL;
    st.rflags = 0x202;

    Runtime rt;
    rt.Run(st);
    // ror 0x12345678 by 8 = 0x78123456.
    EXPECT_EQ(st.gpr[0], 0x0000000078123456ULL) << "ror by 8";
}

// ============================================================================
// VPINSRD — insert a 32-bit GPR dword into an imm8-selected lane of an
// xmm; other lanes copied from src1; upper YMM zeroed (VEX). Inverse of
// VPEXTRD.
// ============================================================================

// Insert into lane 2: lane 2 gets the GPR value, lanes 0/1/3 from src1,
// upper YMM zeroed.
TEST_F(CpuRuntimeTest, Vpinsrd_Lane2_OtherLanesFromSrc1) {
    // vpinsrd xmm0, xmm1, eax, 2  — c4 e3 71 22 c0 02 (6 bytes)
    const u8 program[] = {
        0xc4, 0xe3, 0x71, 0x22, 0xc0, 0x02,
        0xc3,
    };
    std::memcpy(mem.CodePtr(), program, sizeof(program));
    u8* guest_rsp = mem.StackTop() - 8;
    *reinterpret_cast<u64*>(guest_rsp) = kReturnSentinel;
    GuestState st{};
    st.rip = reinterpret_cast<u64>(mem.CodePtr());
    st.gpr[4] = reinterpret_cast<u64>(guest_rsp);
    // src1 = xmm1 (lane 4): dwords 0xA0,0xA1,0xA2,0xA3
    st.ymm[4] = (0xA1ULL << 32) | 0xA0ULL;
    st.ymm[5] = (0xA3ULL << 32) | 0xA2ULL;
    st.gpr[0] = 0xDEADBEEFULL; // eax (low 32 used)
    // pre-pollute dst (lane 0) including upper YMM
    st.ymm[0] = 0x1111111111111111ULL;
    st.ymm[1] = 0x1111111111111111ULL;
    st.ymm[2] = 0x2222222222222222ULL;
    st.ymm[3] = 0x2222222222222222ULL;

    Runtime rt;
    rt.Run(st);
    // dst lanes: 0=A0, 1=A1, 2=0xDEADBEEF (inserted), 3=A3.
    // chunk0 (lanes0,1) = 0x000000A1_000000A0
    EXPECT_EQ(st.ymm[0], 0x000000A1000000A0ULL) << "lanes 0,1 from src1";
    // chunk1 (lanes2,3) = 0x000000A3_DEADBEEF
    EXPECT_EQ(st.ymm[1], 0x000000A3DEADBEEFULL) << "lane2 inserted, lane3 from src1";
    EXPECT_EQ(st.ymm[2], 0ULL) << "VEX zeros upper YMM";
    EXPECT_EQ(st.ymm[3], 0ULL);
}

// Insert into lane 0 (the low dword).
TEST_F(CpuRuntimeTest, Vpinsrd_Lane0) {
    // vpinsrd xmm0, xmm1, eax, 0
    const u8 program[] = {
        0xc4, 0xe3, 0x71, 0x22, 0xc0, 0x00,
        0xc3,
    };
    std::memcpy(mem.CodePtr(), program, sizeof(program));
    u8* guest_rsp = mem.StackTop() - 8;
    *reinterpret_cast<u64*>(guest_rsp) = kReturnSentinel;
    GuestState st{};
    st.rip = reinterpret_cast<u64>(mem.CodePtr());
    st.gpr[4] = reinterpret_cast<u64>(guest_rsp);
    st.ymm[4] = (0xA1ULL << 32) | 0xA0ULL;
    st.ymm[5] = (0xA3ULL << 32) | 0xA2ULL;
    st.gpr[0] = 0x12345678ULL;

    Runtime rt;
    rt.Run(st);
    // lane0=0x12345678, lane1=A1, lane2=A2, lane3=A3.
    EXPECT_EQ(st.ymm[0], 0x000000A112345678ULL) << "lane0 inserted, lane1 from src1";
    EXPECT_EQ(st.ymm[1], 0x000000A3000000A2ULL) << "lanes 2,3 from src1";
}

// In-place form: dst == src1 (vpinsrd xmm1, xmm1, eax, 3). The read-
// before-write ordering must make this a correct self-copy + patch.
TEST_F(CpuRuntimeTest, Vpinsrd_InPlace_DstEqualsSrc1) {
    // vpinsrd xmm1, xmm1, eax, 3  — c4 e3 71 22 c8 03
    const u8 program[] = {
        0xc4, 0xe3, 0x71, 0x22, 0xc8, 0x03,
        0xc3,
    };
    std::memcpy(mem.CodePtr(), program, sizeof(program));
    u8* guest_rsp = mem.StackTop() - 8;
    *reinterpret_cast<u64*>(guest_rsp) = kReturnSentinel;
    GuestState st{};
    st.rip = reinterpret_cast<u64>(mem.CodePtr());
    st.gpr[4] = reinterpret_cast<u64>(guest_rsp);
    // xmm1 = src1 AND dst (lane 4): dwords 0xB0,0xB1,0xB2,0xB3
    st.ymm[4] = (0xB1ULL << 32) | 0xB0ULL;
    st.ymm[5] = (0xB3ULL << 32) | 0xB2ULL;
    st.gpr[0] = 0xCAFEF00DULL;

    Runtime rt;
    rt.Run(st);
    // lane3 replaced; lanes 0,1,2 unchanged.
    EXPECT_EQ(st.ymm[4], 0x000000B1000000B0ULL) << "lanes 0,1 unchanged in-place";
    EXPECT_EQ(st.ymm[5], 0xCAFEF00D000000B2ULL) << "lane3 inserted, lane2 unchanged";
    // VPINSRD is xmm-form: upper YMM of xmm1 (chunks 6,7) zeroed.
    EXPECT_EQ(st.ymm[6], 0ULL) << "VEX zeros upper YMM of dst";
    EXPECT_EQ(st.ymm[7], 0ULL);
}

// ============================================================================
// VMOVQ memory forms — load (xmm <- m64, zeroes bits 255:64) and store
// (m64 <- xmm, exactly 8 bytes). Extends the existing reg-reg VMOVQ
// emitter, which previously rejected memory operands.
// ============================================================================

// Load: vmovq xmm0, [rax]. Low 64 loaded, upper 192 bits of YMM zeroed.
TEST_F(CpuRuntimeTest, Vmovq_LoadFromMemory_ZeroesUpper) {
    u64* scratch = reinterpret_cast<u64*>(mem.CodePtr() + 0x100);
    scratch[0] = 0x0123456789ABCDEFULL;
    // mov rax, scratch ; vmovq xmm0, [rax]
    const u8 program[] = {
        0x48, 0xb8, 0,0,0,0,0,0,0,0,
        0xc5, 0xfa, 0x7e, 0x00, // vmovq xmm0, [rax]
        0xc3,
    };
    u8 prog[sizeof(program)];
    std::memcpy(prog, program, sizeof(program));
    const u64 scratch_addr = reinterpret_cast<u64>(scratch);
    std::memcpy(prog + 2, &scratch_addr, sizeof(scratch_addr));
    std::memcpy(mem.CodePtr(), prog, sizeof(prog));
    u8* guest_rsp = mem.StackTop() - 8;
    *reinterpret_cast<u64*>(guest_rsp) = kReturnSentinel;
    GuestState st{};
    st.rip = reinterpret_cast<u64>(mem.CodePtr());
    st.gpr[4] = reinterpret_cast<u64>(guest_rsp);
    // pre-pollute xmm0 (lane 0) including upper YMM
    st.ymm[0] = 0xFFFFFFFFFFFFFFFFULL;
    st.ymm[1] = 0xFFFFFFFFFFFFFFFFULL;
    st.ymm[2] = 0xFFFFFFFFFFFFFFFFULL;
    st.ymm[3] = 0xFFFFFFFFFFFFFFFFULL;

    Runtime rt;
    rt.Run(st);
    EXPECT_EQ(st.ymm[0], 0x0123456789ABCDEFULL) << "low 64 loaded from memory";
    EXPECT_EQ(st.ymm[1], 0ULL) << "bits 127:64 zeroed";
    EXPECT_EQ(st.ymm[2], 0ULL) << "bits 255:128 zeroed (VEX)";
    EXPECT_EQ(st.ymm[3], 0ULL);
}

// Store: vmovq [rax], xmm0. Exactly 8 bytes written, neighbour untouched.
TEST_F(CpuRuntimeTest, Vmovq_StoreToMemory_Writes8Bytes) {
    u64* scratch = reinterpret_cast<u64*>(mem.CodePtr() + 0x100);
    scratch[0] = 0x1111111111111111ULL;
    scratch[1] = 0xBBBBBBBBBBBBBBBBULL; // sentinel after the 8-byte target
    // mov rax, scratch ; vmovq [rax], xmm0
    const u8 program[] = {
        0x48, 0xb8, 0,0,0,0,0,0,0,0,
        0xc5, 0xf9, 0xd6, 0x00, // vmovq [rax], xmm0
        0xc3,
    };
    u8 prog[sizeof(program)];
    std::memcpy(prog, program, sizeof(program));
    const u64 scratch_addr = reinterpret_cast<u64>(scratch);
    std::memcpy(prog + 2, &scratch_addr, sizeof(scratch_addr));
    std::memcpy(mem.CodePtr(), prog, sizeof(prog));
    u8* guest_rsp = mem.StackTop() - 8;
    *reinterpret_cast<u64*>(guest_rsp) = kReturnSentinel;
    GuestState st{};
    st.rip = reinterpret_cast<u64>(mem.CodePtr());
    st.gpr[4] = reinterpret_cast<u64>(guest_rsp);
    // xmm0 low 64 = value to store; upper bits must NOT be written.
    st.ymm[0] = 0xCAFEF00DDEADBEEFULL;
    st.ymm[1] = 0x9999999999999999ULL;

    Runtime rt;
    rt.Run(st);
    EXPECT_EQ(scratch[0], 0xCAFEF00DDEADBEEFULL) << "low 64 stored";
    EXPECT_EQ(scratch[1], 0xBBBBBBBBBBBBBBBBULL) << "next qword untouched (only 8 bytes written)";
}

// ============================================================================
// VPSLLDQ / VPSRLDQ — whole-128-bit-lane BYTE shift (imm8 byte count),
// zero-filling. Distinct from the per-element bit shifts. count >= 16
// zeroes the lane. Values confirmed against host _mm_slli_si128.
// ============================================================================

// VPSLLDQ by 4 bytes: byte i -> byte i+4, low 4 bytes zeroed.
TEST_F(CpuRuntimeTest, Vpslldq_By4Bytes) {
    // vpslldq xmm0, xmm1, 4  — c5 f9 73 f9 04 (5 bytes)
    const u8 program[] = {
        0xc5, 0xf9, 0x73, 0xf9, 0x04,
        0xc3,
    };
    std::memcpy(mem.CodePtr(), program, sizeof(program));
    u8* guest_rsp = mem.StackTop() - 8;
    *reinterpret_cast<u64*>(guest_rsp) = kReturnSentinel;
    GuestState st{};
    st.rip = reinterpret_cast<u64>(mem.CodePtr());
    st.gpr[4] = reinterpret_cast<u64>(guest_rsp);
    // src bytes 0x00,0x11,...,0xFF (byte i = i*0x11)
    st.ymm[4] = 0x7766554433221100ULL;
    st.ymm[5] = 0xFFEEDDCCBBAA9988ULL;
    st.ymm[2] = 0xDEADBEEFDEADBEEFULL;
    st.ymm[3] = 0xDEADBEEFDEADBEEFULL;

    Runtime rt;
    rt.Run(st);
    EXPECT_EQ(st.ymm[0], 0x3322110000000000ULL) << "low: bytes 0-3 zero-filled";
    EXPECT_EQ(st.ymm[1], 0xBBAA998877665544ULL) << "high: shifted up 4 bytes";
    EXPECT_EQ(st.ymm[2], 0ULL) << "VEX zeros upper YMM";
    EXPECT_EQ(st.ymm[3], 0ULL);
}

// count >= 16 zeroes the entire lane.
TEST_F(CpuRuntimeTest, Vpslldq_Count16_ZeroesLane) {
    // vpslldq xmm0, xmm1, 16  — c5 f9 73 f9 10
    const u8 program[] = {
        0xc5, 0xf9, 0x73, 0xf9, 0x10,
        0xc3,
    };
    std::memcpy(mem.CodePtr(), program, sizeof(program));
    u8* guest_rsp = mem.StackTop() - 8;
    *reinterpret_cast<u64*>(guest_rsp) = kReturnSentinel;
    GuestState st{};
    st.rip = reinterpret_cast<u64>(mem.CodePtr());
    st.gpr[4] = reinterpret_cast<u64>(guest_rsp);
    st.ymm[4] = 0x7766554433221100ULL;
    st.ymm[5] = 0xFFEEDDCCBBAA9988ULL;

    Runtime rt;
    rt.Run(st);
    EXPECT_EQ(st.ymm[0], 0ULL) << "shift by 16 bytes -> all zero";
    EXPECT_EQ(st.ymm[1], 0ULL);
}

// VPSRLDQ by 3 bytes: byte i -> byte i-3, high 3 bytes zeroed.
TEST_F(CpuRuntimeTest, Vpsrldq_By3Bytes) {
    // vpsrldq xmm0, xmm1, 3  — c5 f9 73 d9 03
    const u8 program[] = {
        0xc5, 0xf9, 0x73, 0xd9, 0x03,
        0xc3,
    };
    std::memcpy(mem.CodePtr(), program, sizeof(program));
    u8* guest_rsp = mem.StackTop() - 8;
    *reinterpret_cast<u64*>(guest_rsp) = kReturnSentinel;
    GuestState st{};
    st.rip = reinterpret_cast<u64>(mem.CodePtr());
    st.gpr[4] = reinterpret_cast<u64>(guest_rsp);
    st.ymm[4] = 0x7766554433221100ULL;
    st.ymm[5] = 0xFFEEDDCCBBAA9988ULL;

    Runtime rt;
    rt.Run(st);
    // Right shift by 3 bytes: new lo = bytes 3..10, new hi = bytes 11..15 + 3 zero bytes.
    // orig bytes (LE): [00 11 22 33 44 55 66 77][88 99 aa bb cc dd ee ff]
    // >>3 bytes: lo = 33 44 55 66 77 88 99 aa -> 0xaa99887766554433
    //            hi = bb cc dd ee ff 00 00 00 -> 0x000000ffeeddccbb
    EXPECT_EQ(st.ymm[0], 0xAA99887766554433ULL) << "low shifted down 3 bytes";
    EXPECT_EQ(st.ymm[1], 0x000000FFEEDDCCBBULL) << "high: top 3 bytes zeroed";
}

// ============================================================================
// VPCMPISTRM — SSE4.2 packed compare implicit-length strings, return
// MASK to the implicit XMM0. Sibling of VPCMPISTRI (which returns an
// index in ECX). imm8 bit6 picks bit-mask (0) vs byte-mask (1). Values
// confirmed against host _mm_cmpistrm. Operands are loaded into host
// xmm1/xmm2 (not xmm0, which receives the result).
//
// Test data: needle a="ABCDEFGHIJKLMNOP", haystack b="XBXDXFXHX...".
// Under equal-any (imm bits 2-3 = 00), b matches the needle set at the
// odd positions 1,3,5,7 (B,D,F,H) -> match bitmask 0b10101010 = 0xAA.
// ============================================================================

// Bit-mask form (imm 0x00): low bits of XMM0 hold one bit per element.
TEST_F(CpuRuntimeTest, Vpcmpistrm_BitMask_EqualAny) {
    // vpcmpistrm xmm1, xmm2, 0x00  — c4 e3 79 62 ca 00 (6 bytes)
    const u8 program[] = {
        0xc4, 0xe3, 0x79, 0x62, 0xca, 0x00,
        0xc3,
    };
    std::memcpy(mem.CodePtr(), program, sizeof(program));
    u8* guest_rsp = mem.StackTop() - 8;
    *reinterpret_cast<u64*>(guest_rsp) = kReturnSentinel;
    GuestState st{};
    st.rip = reinterpret_cast<u64>(mem.CodePtr());
    st.gpr[4] = reinterpret_cast<u64>(guest_rsp);
    // a = xmm1 (lane 4) = "ABCDEFGHIJKLMNOP"
    st.ymm[4] = 0x4847464544434241ULL; // "ABCDEFGH"
    st.ymm[5] = 0x504F4E4D4C4B4A49ULL; // "IJKLMNOP"
    // b = xmm2 (lane 8) = "XBXDXFXHXXXXXXXX"
    st.ymm[8]  = 0x4858465844584258ULL; // "XBXDXFXH"
    st.ymm[9]  = 0x5858585858585858ULL; // "XXXXXXXX"
    // pre-pollute guest XMM0 (the implicit dest, vec index 0) entirely
    st.ymm[0] = 0xDEADBEEFDEADBEEFULL;
    st.ymm[1] = 0xDEADBEEFDEADBEEFULL;
    st.ymm[2] = 0xDEADBEEFDEADBEEFULL;
    st.ymm[3] = 0xDEADBEEFDEADBEEFULL;
    st.rflags = 0x2;

    Runtime rt;
    rt.Run(st);
    // bit-mask: matches at positions 1,3,5,7 -> 0b10101010 = 0xAA.
    EXPECT_EQ(st.ymm[0], 0x00000000000000AAULL) << "bit-mask of matches";
    EXPECT_EQ(st.ymm[1], 0ULL) << "rest of low 128 zero (bit-mask)";
    EXPECT_EQ(st.ymm[2], 0ULL) << "VEX zeros upper YMM of XMM0";
    EXPECT_EQ(st.ymm[3], 0ULL);
    // CF = mask not all zero = 1; ZF/SF = 0 (no nulls); OF = 0.
    EXPECT_EQ(st.rflags & 0x1ULL, 0x1ULL) << "CF: matches exist";
    EXPECT_EQ(st.rflags & (1ULL<<6), 0ULL) << "ZF: b has no null";
    EXPECT_EQ(st.rflags & (1ULL<<7), 0ULL) << "SF: a has no null";
}

// Byte-mask form (imm 0x40, bit6=1): each element 0x00 or 0xFF.
TEST_F(CpuRuntimeTest, Vpcmpistrm_ByteMask_EqualAny) {
    // vpcmpistrm xmm1, xmm2, 0x40  — c4 e3 79 62 ca 40
    const u8 program[] = {
        0xc4, 0xe3, 0x79, 0x62, 0xca, 0x40,
        0xc3,
    };
    std::memcpy(mem.CodePtr(), program, sizeof(program));
    u8* guest_rsp = mem.StackTop() - 8;
    *reinterpret_cast<u64*>(guest_rsp) = kReturnSentinel;
    GuestState st{};
    st.rip = reinterpret_cast<u64>(mem.CodePtr());
    st.gpr[4] = reinterpret_cast<u64>(guest_rsp);
    st.ymm[4] = 0x4847464544434241ULL;
    st.ymm[5] = 0x504F4E4D4C4B4A49ULL;
    st.ymm[8]  = 0x4858465844584258ULL;
    st.ymm[9]  = 0x5858585858585858ULL;
    st.rflags = 0x2;

    Runtime rt;
    rt.Run(st);
    // byte-mask: matching byte positions -> 0xFF, others 0x00.
    // positions 1,3,5,7 match -> 0xff00ff00ff00ff00 ; positions 8-15 no match.
    EXPECT_EQ(st.ymm[0], 0xFF00FF00FF00FF00ULL) << "byte-mask low: 0xFF at matches";
    EXPECT_EQ(st.ymm[1], 0x0000000000000000ULL) << "byte-mask high: no matches in 8-15";
    EXPECT_EQ(st.ymm[2], 0ULL) << "VEX zeros upper YMM";
    EXPECT_EQ(st.ymm[3], 0ULL);
}

// ZF reflects a null terminator in the second operand (b).
TEST_F(CpuRuntimeTest, Vpcmpistrm_ZFOnNullInB) {
    const u8 program[] = {
        0xc4, 0xe3, 0x79, 0x62, 0xca, 0x00,
        0xc3,
    };
    std::memcpy(mem.CodePtr(), program, sizeof(program));
    u8* guest_rsp = mem.StackTop() - 8;
    *reinterpret_cast<u64*>(guest_rsp) = kReturnSentinel;
    GuestState st{};
    st.rip = reinterpret_cast<u64>(mem.CodePtr());
    st.gpr[4] = reinterpret_cast<u64>(guest_rsp);
    st.ymm[4] = 0x4847464544434241ULL; // a "ABCDEFGH"
    st.ymm[5] = 0x504F4E4D4C4B4A49ULL; // a "IJKLMNOP"
    // b = "XBX\0XF\0\0" then zeros -> contains nulls (terminator)
    st.ymm[8]  = 0x0000460058044258ULL; // bytes: 58 42 04 00 58 46 00 00
    st.ymm[9]  = 0x0000000000000000ULL;
    st.rflags = 0x2;

    Runtime rt;
    rt.Run(st);
    // ZF set because b contains a null terminator.
    EXPECT_EQ(st.rflags & (1ULL<<6), (1ULL<<6)) << "ZF: b has a null terminator";
}

// ============================================================================
// VPANDN — bitwise AND-NOT: dst = (NOT src1) AND src2. The NOT applies
// only to the FIRST source (asymmetric). Routed through EmitVecBitOp
// with the Andn kind. Values confirmed against host _mm_andnot_si128.
// ============================================================================

// dst = (~src1) & src2, 128-bit. Also confirms upper-YMM zeroing.
TEST_F(CpuRuntimeTest, Vpandn_AndNot_128) {
    // vpandn xmm0, xmm1, xmm2  — c5 f1 df c2 (4 bytes)
    const u8 program[] = {
        0xc5, 0xf1, 0xdf, 0xc2,
        0xc3,
    };
    std::memcpy(mem.CodePtr(), program, sizeof(program));
    u8* guest_rsp = mem.StackTop() - 8;
    *reinterpret_cast<u64*>(guest_rsp) = kReturnSentinel;
    GuestState st{};
    st.rip = reinterpret_cast<u64>(mem.CodePtr());
    st.gpr[4] = reinterpret_cast<u64>(guest_rsp);
    // src1 = xmm1 (lane 4)
    st.ymm[4] = 0x00FF00FF00FF00FFULL;
    st.ymm[5] = 0x0F0F0F0F0F0F0F0FULL;
    // src2 = xmm2 (lane 8)
    st.ymm[8] = 0xFFFFFFFFFFFFFFFFULL;
    st.ymm[9] = 0xAAAAAAAAAAAAAAAAULL;
    // pre-pollute dst upper YMM
    st.ymm[2] = 0xDEADBEEFDEADBEEFULL;
    st.ymm[3] = 0xDEADBEEFDEADBEEFULL;

    Runtime rt;
    rt.Run(st);
    // (~src1) & src2
    EXPECT_EQ(st.ymm[0], 0xFF00FF00FF00FF00ULL) << "(~0x00FF..) & 0xFFFF.. ";
    EXPECT_EQ(st.ymm[1], 0xA0A0A0A0A0A0A0A0ULL) << "(~0x0F0F..) & 0xAAAA.. ";
    EXPECT_EQ(st.ymm[2], 0ULL) << "VEX zeros upper YMM";
    EXPECT_EQ(st.ymm[3], 0ULL);
}

// Asymmetry guard: VPANDN is NOT commutative. With these operands,
// (~src1)&src2 != (~src2)&src1, so swapping would give a different
// (wrong) result. This catches any accidental symmetric-AND or
// wrong-operand-inverted implementation.
TEST_F(CpuRuntimeTest, Vpandn_IsAsymmetric) {
    // vpandn xmm0, xmm1, xmm2
    const u8 program[] = {
        0xc5, 0xf1, 0xdf, 0xc2,
        0xc3,
    };
    std::memcpy(mem.CodePtr(), program, sizeof(program));
    u8* guest_rsp = mem.StackTop() - 8;
    *reinterpret_cast<u64*>(guest_rsp) = kReturnSentinel;
    GuestState st{};
    st.rip = reinterpret_cast<u64>(mem.CodePtr());
    st.gpr[4] = reinterpret_cast<u64>(guest_rsp);
    // src1 has bits set where src2 does NOT, and vice versa, so the
    // result distinguishes (~src1)&src2 from (~src2)&src1.
    st.ymm[4] = 0xF0F0F0F0F0F0F0F0ULL; // src1
    st.ymm[5] = 0x0ULL;
    st.ymm[8] = 0xFF00FF00FF00FF00ULL; // src2
    st.ymm[9] = 0x0ULL;

    Runtime rt;
    rt.Run(st);
    // (~src1) & src2 = (~0xF0F0..) & 0xFF00.. = 0x0F0F.. & 0xFF00.. = 0x0F000F000F000F00
    EXPECT_EQ(st.ymm[0], 0x0F000F000F000F00ULL) << "(~src1)&src2, not the swapped form";
    EXPECT_EQ(st.ymm[1], 0ULL);
    // Sanity: the swapped form (~src2)&src1 would be 0x00F000F000F000F0 — different.
}

// ============================================================================
// CMP (16-bit) — narrow-width compare via host-rflags round-trip so the
// host computes width-correct CF/OF/SF/ZF/PF. The game hit the
// `cmp word [mem], imm` form. Flag values confirmed against host cmpw.
// ============================================================================

// cmp word [mem], imm — the exact form the game emitted. Equal -> ZF=1.
TEST_F(CpuRuntimeTest, Cmp16_MemImm_Equal) {
    u16* scratch = reinterpret_cast<u16*>(mem.CodePtr() + 0x100);
    scratch[0] = 0x1234;
    // mov rax, scratch ; cmp word [rax], 0x1234
    // cmp word [rax], imm16 = 66 81 38 34 12 (5 bytes); the game's was
    // length 4 (likely imm8-sign-extended form 66 83 38 imm8), but we
    // test the imm16 form here which exercises the same path.
    const u8 program[] = {
        0x48, 0xb8, 0,0,0,0,0,0,0,0,
        0x66, 0x81, 0x38, 0x34, 0x12, // cmp word [rax], 0x1234
        0xc3,
    };
    u8 prog[sizeof(program)];
    std::memcpy(prog, program, sizeof(program));
    const u64 a = reinterpret_cast<u64>(scratch);
    std::memcpy(prog + 2, &a, sizeof(a));
    std::memcpy(mem.CodePtr(), prog, sizeof(prog));
    u8* guest_rsp = mem.StackTop() - 8;
    *reinterpret_cast<u64*>(guest_rsp) = kReturnSentinel;
    GuestState st{};
    st.rip = reinterpret_cast<u64>(mem.CodePtr());
    st.gpr[4] = reinterpret_cast<u64>(guest_rsp);
    st.rflags = 0x2;

    Runtime rt;
    rt.Run(st);
    EXPECT_EQ(st.rflags & (1ULL<<6), (1ULL<<6)) << "ZF set: equal";
    EXPECT_EQ(st.rflags & 0x1ULL, 0ULL) << "CF clear: no borrow";
}

// cmp word [mem], imm where mem < imm unsigned -> CF=1, SF=1.
TEST_F(CpuRuntimeTest, Cmp16_MemImm_Borrow) {
    u16* scratch = reinterpret_cast<u16*>(mem.CodePtr() + 0x100);
    scratch[0] = 0x0010;
    const u8 program[] = {
        0x48, 0xb8, 0,0,0,0,0,0,0,0,
        0x66, 0x81, 0x38, 0x20, 0x00, // cmp word [rax], 0x0020
        0xc3,
    };
    u8 prog[sizeof(program)];
    std::memcpy(prog, program, sizeof(program));
    const u64 a = reinterpret_cast<u64>(scratch);
    std::memcpy(prog + 2, &a, sizeof(a));
    std::memcpy(mem.CodePtr(), prog, sizeof(prog));
    u8* guest_rsp = mem.StackTop() - 8;
    *reinterpret_cast<u64*>(guest_rsp) = kReturnSentinel;
    GuestState st{};
    st.rip = reinterpret_cast<u64>(mem.CodePtr());
    st.gpr[4] = reinterpret_cast<u64>(guest_rsp);
    st.rflags = 0x2;

    Runtime rt;
    rt.Run(st);
    // 0x10 - 0x20 borrows: CF=1, result 0xFFF0 -> SF=1, ZF=0.
    EXPECT_EQ(st.rflags & 0x1ULL, 0x1ULL) << "CF set: 0x10 < 0x20";
    EXPECT_EQ(st.rflags & (1ULL<<7), (1ULL<<7)) << "SF set: result 0xFFF0";
    EXPECT_EQ(st.rflags & (1ULL<<6), 0ULL) << "ZF clear";
}

// The 16-bit-specific case: cmp 0x8000, 0x7FFF sets OF=1 at 16-bit
// width (signed overflow) but would NOT at 64-bit width. This proves
// the flags are computed at the correct narrow width.
TEST_F(CpuRuntimeTest, Cmp16_RegImm_WidthSpecificOverflow) {
    // cmp ax, 0x7FFF  =  66 3d ff 7f (4 bytes) — cmp AX, imm16
    const u8 program[] = {
        0x66, 0x3d, 0xff, 0x7f,
        0xc3,
    };
    std::memcpy(mem.CodePtr(), program, sizeof(program));
    u8* guest_rsp = mem.StackTop() - 8;
    *reinterpret_cast<u64*>(guest_rsp) = kReturnSentinel;
    GuestState st{};
    st.rip = reinterpret_cast<u64>(mem.CodePtr());
    st.gpr[4] = reinterpret_cast<u64>(guest_rsp);
    // ax = 0x8000, upper bits of rax set to prove width isolation.
    st.gpr[0] = 0xFFFFFFFFFFFF8000ULL;
    st.rflags = 0x2;

    Runtime rt;
    rt.Run(st);
    // 0x8000 - 0x7FFF = 1: CF=0, ZF=0, SF=0, OF=1 (signed overflow at 16-bit).
    EXPECT_EQ(st.rflags & (1ULL<<11), (1ULL<<11)) << "OF set: 16-bit signed overflow";
    EXPECT_EQ(st.rflags & 0x1ULL, 0ULL) << "CF clear: no unsigned borrow";
    EXPECT_EQ(st.rflags & (1ULL<<7), 0ULL) << "SF clear: result positive (0x0001)";
    EXPECT_EQ(st.rflags & (1ULL<<6), 0ULL) << "ZF clear";
}

// ============================================================================
// BLSI — BMI1 "extract lowest set isolated bit": dst = (-src) & src.
// Isolates the lowest set bit; zeroes the rest. Flags: CF = (src != 0)
// [note: opposite polarity from BLSR/BLSMSK], ZF = (result==0),
// SF = result MSB, OF = 0. Values confirmed against host blsi.
// ============================================================================

// Isolate lowest set bit of 0x16 (0b10110) -> 0x02. CF set (src != 0).
TEST_F(CpuRuntimeTest, Blsi_IsolatesLowestBit) {
    // blsi eax, ecx  — c4 e2 78 f3 d9 (5 bytes)
    const u8 program[] = {
        0xc4, 0xe2, 0x78, 0xf3, 0xd9,
        0xc3,
    };
    std::memcpy(mem.CodePtr(), program, sizeof(program));
    u8* guest_rsp = mem.StackTop() - 8;
    *reinterpret_cast<u64*>(guest_rsp) = kReturnSentinel;
    GuestState st{};
    st.rip = reinterpret_cast<u64>(mem.CodePtr());
    st.gpr[4] = reinterpret_cast<u64>(guest_rsp);
    st.gpr[1] = 0xFFFFFFFF00000016ULL; // ecx = 0x16, high junk for ZE check
    st.gpr[0] = 0xDEADBEEFDEADBEEFULL; // eax pre-poisoned
    st.rflags = 0x2;

    Runtime rt;
    rt.Run(st);
    EXPECT_EQ(st.gpr[0], 0x0000000000000002ULL) << "lowest set bit isolated, zero-extended";
    EXPECT_EQ(st.rflags & 0x1ULL, 0x1ULL) << "CF set: src != 0";
    EXPECT_EQ(st.rflags & (1ULL<<6), 0ULL) << "ZF clear: result nonzero";
    EXPECT_EQ(st.rflags & (1ULL<<7), 0ULL) << "SF clear: bit1 result";
}

// Zero source: result 0, CF=0, ZF=1 (the polarity that distinguishes
// BLSI from BLSR/BLSMSK, which set CF when src==0).
TEST_F(CpuRuntimeTest, Blsi_ZeroSource_ClearsCFSetsZF) {
    const u8 program[] = {
        0xc4, 0xe2, 0x78, 0xf3, 0xd9, // blsi eax, ecx
        0xc3,
    };
    std::memcpy(mem.CodePtr(), program, sizeof(program));
    u8* guest_rsp = mem.StackTop() - 8;
    *reinterpret_cast<u64*>(guest_rsp) = kReturnSentinel;
    GuestState st{};
    st.rip = reinterpret_cast<u64>(mem.CodePtr());
    st.gpr[4] = reinterpret_cast<u64>(guest_rsp);
    st.gpr[1] = 0; // ecx = 0
    st.gpr[0] = 0xDEADBEEFDEADBEEFULL;
    // Pre-set CF so we can confirm it is CLEARED (not left set).
    st.rflags = 0x2 | 0x1;

    Runtime rt;
    rt.Run(st);
    EXPECT_EQ(st.gpr[0], 0ULL) << "zero source -> zero result";
    EXPECT_EQ(st.rflags & 0x1ULL, 0ULL) << "CF cleared: src == 0 (BLSI polarity)";
    EXPECT_EQ(st.rflags & (1ULL<<6), (1ULL<<6)) << "ZF set: result zero";
}

// SF reflects the result MSB: blsi 0x80000000 = 0x80000000 -> SF=1.
TEST_F(CpuRuntimeTest, Blsi_HighBit_SetsSF) {
    const u8 program[] = {
        0xc4, 0xe2, 0x78, 0xf3, 0xd9, // blsi eax, ecx
        0xc3,
    };
    std::memcpy(mem.CodePtr(), program, sizeof(program));
    u8* guest_rsp = mem.StackTop() - 8;
    *reinterpret_cast<u64*>(guest_rsp) = kReturnSentinel;
    GuestState st{};
    st.rip = reinterpret_cast<u64>(mem.CodePtr());
    st.gpr[4] = reinterpret_cast<u64>(guest_rsp);
    st.gpr[1] = 0x80000000ULL; // ecx
    st.rflags = 0x2;

    Runtime rt;
    rt.Run(st);
    EXPECT_EQ(st.gpr[0], 0x0000000080000000ULL) << "only bit 31 set";
    EXPECT_EQ(st.rflags & (1ULL<<7), (1ULL<<7)) << "SF set: result MSB (bit31)";
    EXPECT_EQ(st.rflags & 0x1ULL, 0x1ULL) << "CF set: src != 0";
}

// ============================================================================
// INC dword [mem] — 32-bit in-memory increment. The defining INC quirk:
// it sets ZF/SF/OF/PF/AF but does NOT affect CF (unlike ADD). The
// mem-dst path lets the host INC produce the flags via the rflags
// round-trip, so CF-preservation comes for free. Confirmed vs host.
// ============================================================================

// Basic increment of a 32-bit memory counter; CF pre-set must survive.
TEST_F(CpuRuntimeTest, Inc32_Mem_PreservesCF) {
    u32* scratch = reinterpret_cast<u32*>(mem.CodePtr() + 0x100);
    scratch[0] = 0x41;
    scratch[1] = 0x99999999u; // sentinel after the 4-byte target
    // mov rax, scratch ; inc dword [rax]
    const u8 program[] = {
        0x48, 0xb8, 0,0,0,0,0,0,0,0,
        0xff, 0x00, // inc dword [rax]
        0xc3,
    };
    u8 prog[sizeof(program)];
    std::memcpy(prog, program, sizeof(program));
    const u64 a = reinterpret_cast<u64>(scratch);
    std::memcpy(prog + 2, &a, sizeof(a));
    std::memcpy(mem.CodePtr(), prog, sizeof(prog));
    u8* guest_rsp = mem.StackTop() - 8;
    *reinterpret_cast<u64*>(guest_rsp) = kReturnSentinel;
    GuestState st{};
    st.rip = reinterpret_cast<u64>(mem.CodePtr());
    st.gpr[4] = reinterpret_cast<u64>(guest_rsp);
    st.rflags = 0x2 | 0x1; // CF pre-set

    Runtime rt;
    rt.Run(st);
    EXPECT_EQ(scratch[0], 0x42u) << "memory incremented";
    EXPECT_EQ(scratch[1], 0x99999999u) << "next dword untouched (only 4 bytes written)";
    EXPECT_EQ(st.rflags & 0x1ULL, 0x1ULL) << "CF preserved (INC does not affect CF)";
    EXPECT_EQ(st.rflags & (1ULL<<6), 0ULL) << "ZF clear";
}

// Wraparound 0xFFFFFFFF -> 0: ZF=1, and CF stays clear (INC never
// sets CF, even on overflow-to-zero — distinct from ADD).
TEST_F(CpuRuntimeTest, Inc32_Mem_WrapSetsZFNotCF) {
    u32* scratch = reinterpret_cast<u32*>(mem.CodePtr() + 0x100);
    scratch[0] = 0xFFFFFFFFu;
    const u8 program[] = {
        0x48, 0xb8, 0,0,0,0,0,0,0,0,
        0xff, 0x00, // inc dword [rax]
        0xc3,
    };
    u8 prog[sizeof(program)];
    std::memcpy(prog, program, sizeof(program));
    const u64 a = reinterpret_cast<u64>(scratch);
    std::memcpy(prog + 2, &a, sizeof(a));
    std::memcpy(mem.CodePtr(), prog, sizeof(prog));
    u8* guest_rsp = mem.StackTop() - 8;
    *reinterpret_cast<u64*>(guest_rsp) = kReturnSentinel;
    GuestState st{};
    st.rip = reinterpret_cast<u64>(mem.CodePtr());
    st.gpr[4] = reinterpret_cast<u64>(guest_rsp);
    st.rflags = 0x2; // CF clear going in

    Runtime rt;
    rt.Run(st);
    EXPECT_EQ(scratch[0], 0x0u) << "wrapped to zero";
    EXPECT_EQ(st.rflags & (1ULL<<6), (1ULL<<6)) << "ZF set: result zero";
    EXPECT_EQ(st.rflags & 0x1ULL, 0ULL) << "CF still clear (INC never sets CF on wrap)";
}

// Signed overflow edge: 0x7FFFFFFF -> 0x80000000 sets OF and SF.
TEST_F(CpuRuntimeTest, Inc32_Mem_SignedOverflow) {
    u32* scratch = reinterpret_cast<u32*>(mem.CodePtr() + 0x100);
    scratch[0] = 0x7FFFFFFFu;
    const u8 program[] = {
        0x48, 0xb8, 0,0,0,0,0,0,0,0,
        0xff, 0x00, // inc dword [rax]
        0xc3,
    };
    u8 prog[sizeof(program)];
    std::memcpy(prog, program, sizeof(program));
    const u64 a = reinterpret_cast<u64>(scratch);
    std::memcpy(prog + 2, &a, sizeof(a));
    std::memcpy(mem.CodePtr(), prog, sizeof(prog));
    u8* guest_rsp = mem.StackTop() - 8;
    *reinterpret_cast<u64*>(guest_rsp) = kReturnSentinel;
    GuestState st{};
    st.rip = reinterpret_cast<u64>(mem.CodePtr());
    st.gpr[4] = reinterpret_cast<u64>(guest_rsp);
    st.rflags = 0x2;

    Runtime rt;
    rt.Run(st);
    EXPECT_EQ(scratch[0], 0x80000000u) << "incremented past INT_MAX";
    EXPECT_EQ(st.rflags & (1ULL<<11), (1ULL<<11)) << "OF set: signed overflow";
    EXPECT_EQ(st.rflags & (1ULL<<7), (1ULL<<7)) << "SF set: result negative";
}

// ============================================================================
// PREFETCHNTA / PREFETCHT0/1/2 / PREFETCHW — cache hints with NO
// architectural effect. Lifted as no-ops: bytes consumed, no register/
// flag/memory change, no fault, execution continues. We verify the
// hint is skipped cleanly and a following instruction still runs.
// ============================================================================

// prefetchnta [rax+rcx*4+0x100] (the 7-byte SIB+disp form the game hit)
// followed by a mov that proves execution continued past the hint.
TEST_F(CpuRuntimeTest, Prefetchnta_IsNoOp_ExecutionContinues) {
    u32* scratch = reinterpret_cast<u32*>(mem.CodePtr() + 0x100);
    scratch[0] = 0x12345678u;
    // mov rax, scratch        (48 b8 ...)
    // xor rcx, rcx            (48 31 c9)
    // prefetchnta [rax+rcx*4+0x100]  (0f 18 84 88 00 01 00 00) -- 8 bytes:
    //   the SIB byte 0x88 selects a disp32, so the full displacement is
    //   00 01 00 00 (= 0x100). Easy to under-count by one byte.
    // mov edx, 0xCAFE         (ba fe ca 00 00) -- proves we continued
    // ret
    const u8 program[] = {
        0x48, 0xb8, 0,0,0,0,0,0,0,0,
        0x48, 0x31, 0xc9,
        0x0f, 0x18, 0x84, 0x88, 0x00, 0x01, 0x00, 0x00,
        0xba, 0xfe, 0xca, 0x00, 0x00,
        0xc3,
    };
    u8 prog[sizeof(program)];
    std::memcpy(prog, program, sizeof(program));
    const u64 a = reinterpret_cast<u64>(scratch);
    std::memcpy(prog + 2, &a, sizeof(a));
    std::memcpy(mem.CodePtr(), prog, sizeof(prog));
    u8* guest_rsp = mem.StackTop() - 8;
    *reinterpret_cast<u64*>(guest_rsp) = kReturnSentinel;
    GuestState st{};
    st.rip = reinterpret_cast<u64>(mem.CodePtr());
    st.gpr[4] = reinterpret_cast<u64>(guest_rsp);
    // Pre-set registers and flags to known values; the prefetch must
    // leave everything except what the surrounding movs touch.
    st.gpr[2] = 0xAAAAAAAAAAAAAAAAULL; // rdx (will be set by the trailing mov edx)
    st.gpr[6] = 0x1234567812345678ULL; // rsi: must be untouched
    st.rflags = 0x2 | 0x1 | (1ULL<<6); // CF + ZF pre-set

    Runtime rt;
    rt.Run(st);
    // The trailing `mov edx, 0xCAFE` proves we executed past the hint.
    EXPECT_EQ(st.gpr[2], 0x000000000000CAFEULL) << "execution continued past prefetch";
    // The hint touched no other state.
    EXPECT_EQ(st.gpr[6], 0x1234567812345678ULL) << "unrelated reg untouched";
    EXPECT_EQ(scratch[0], 0x12345678u) << "prefetched memory unchanged";
    // Flags unchanged by the prefetch (the movs here don't touch flags
    // either: mov and xor-zero... actually xor clears flags, so only
    // assert the prefetch itself is inert by checking memory/regs).
}

// A prefetch whose computed address would be wild must still not fault
// — we don't even compute the EA. Use a base register holding a
// bogus value; the no-op lift never dereferences it.
TEST_F(CpuRuntimeTest, Prefetchnta_BogusAddress_DoesNotFault) {
    // mov rax, 0xDEAD0000DEAD0000 ; prefetchnta [rax] ; mov ecx, 7 ; ret
    const u8 program[] = {
        0x48, 0xb8, 0x00,0x00,0xad,0xde,0x00,0x00,0xad,0xde,
        0x0f, 0x18, 0x00,             // prefetchnta [rax]
        0xb9, 0x07, 0x00, 0x00, 0x00, // mov ecx, 7
        0xc3,
    };
    std::memcpy(mem.CodePtr(), program, sizeof(program));
    u8* guest_rsp = mem.StackTop() - 8;
    *reinterpret_cast<u64*>(guest_rsp) = kReturnSentinel;
    GuestState st{};
    st.rip = reinterpret_cast<u64>(mem.CodePtr());
    st.gpr[4] = reinterpret_cast<u64>(guest_rsp);

    Runtime rt;
    rt.Run(st);
    // If the prefetch had dereferenced rax, this would have crashed.
    // Reaching here with ecx=7 proves it was a true no-op.
    EXPECT_EQ(st.gpr[1], 0x0000000000000007ULL) << "no fault; execution continued";
}

// ============================================================================
// VMOVNTDQA — non-temporal aligned vector load (xmm/ymm <- [mem]). The
// load-side partner of VMOVNTDQ's store. The "NT" is a cache hint with
// no architectural effect; for the data moved it is identical to
// VMOVDQA. Load-only. Routed through EmitVmovups (the aligned-move
// family). We verify the full 128/256-bit payload lands and that the
// 128-bit form zeroes the upper YMM (VEX semantics).
// ============================================================================

// 128-bit load: all 16 bytes land; upper YMM (255:128) zeroed.
TEST_F(CpuRuntimeTest, Vmovntdqa_Load128_ZeroesUpper) {
    u64* scratch = reinterpret_cast<u64*>(mem.CodePtr() + 0x100);
    scratch[0] = 0x1122334455667788ULL;
    scratch[1] = 0x99AABBCCDDEEFF00ULL;
    // mov rax, scratch ; vmovntdqa xmm1, [rax]  (c4 e2 79 2a 08)
    const u8 program[] = {
        0x48, 0xb8, 0,0,0,0,0,0,0,0,
        0xc4, 0xe2, 0x79, 0x2a, 0x08, // vmovntdqa xmm1, [rax]
        0xc3,
    };
    u8 prog[sizeof(program)];
    std::memcpy(prog, program, sizeof(program));
    const u64 a = reinterpret_cast<u64>(scratch);
    std::memcpy(prog + 2, &a, sizeof(a));
    std::memcpy(mem.CodePtr(), prog, sizeof(prog));
    u8* guest_rsp = mem.StackTop() - 8;
    *reinterpret_cast<u64*>(guest_rsp) = kReturnSentinel;
    GuestState st{};
    st.rip = reinterpret_cast<u64>(mem.CodePtr());
    st.gpr[4] = reinterpret_cast<u64>(guest_rsp);
    // xmm1 = vec index 1 -> ymm[4..7]. Pre-poison upper lanes.
    st.ymm[6] = 0xDEADBEEFDEADBEEFULL;
    st.ymm[7] = 0xDEADBEEFDEADBEEFULL;

    Runtime rt;
    rt.Run(st);
    EXPECT_EQ(st.ymm[4], 0x1122334455667788ULL) << "low qword loaded";
    EXPECT_EQ(st.ymm[5], 0x99AABBCCDDEEFF00ULL) << "high qword loaded";
    EXPECT_EQ(st.ymm[6], 0ULL) << "VEX zeroes bits 255:128";
    EXPECT_EQ(st.ymm[7], 0ULL);
}

// 256-bit load: all 32 bytes land across the full YMM.
TEST_F(CpuRuntimeTest, Vmovntdqa_Load256) {
    u64* scratch = reinterpret_cast<u64*>(mem.CodePtr() + 0x100);
    scratch[0] = 0x0001020304050607ULL;
    scratch[1] = 0x08090A0B0C0D0E0FULL;
    scratch[2] = 0x1011121314151617ULL;
    scratch[3] = 0x18191A1B1C1D1E1FULL;
    // mov rax, scratch ; vmovntdqa ymm0, [rax]  (c4 e2 7d 2a 00)
    const u8 program[] = {
        0x48, 0xb8, 0,0,0,0,0,0,0,0,
        0xc4, 0xe2, 0x7d, 0x2a, 0x00, // vmovntdqa ymm0, [rax]
        0xc3,
    };
    u8 prog[sizeof(program)];
    std::memcpy(prog, program, sizeof(program));
    const u64 a = reinterpret_cast<u64>(scratch);
    std::memcpy(prog + 2, &a, sizeof(a));
    std::memcpy(mem.CodePtr(), prog, sizeof(prog));
    u8* guest_rsp = mem.StackTop() - 8;
    *reinterpret_cast<u64*>(guest_rsp) = kReturnSentinel;
    GuestState st{};
    st.rip = reinterpret_cast<u64>(mem.CodePtr());
    st.gpr[4] = reinterpret_cast<u64>(guest_rsp);

    Runtime rt;
    rt.Run(st);
    // ymm0 = vec index 0 -> ymm[0..3].
    EXPECT_EQ(st.ymm[0], 0x0001020304050607ULL);
    EXPECT_EQ(st.ymm[1], 0x08090A0B0C0D0E0FULL);
    EXPECT_EQ(st.ymm[2], 0x1011121314151617ULL);
    EXPECT_EQ(st.ymm[3], 0x18191A1B1C1D1E1FULL);
}

// ============================================================================
// XOR r64, imm — 64-bit register XOR with a (sign-extended) immediate.
// The imm8/imm32 forms sign-extend to 64 bits, so an imm32 with its
// top bit set flips the upper 32 bits too. XOR clears CF/OF and sets
// ZF/SF/PF from the result. Values confirmed against host xorq.
// ============================================================================

// xor r64, imm32 where the immediate's high bit is set: the
// sign-extension must reach the upper 32 bits. Result 0xFFFFFFFF00000000.
TEST_F(CpuRuntimeTest, Xor64_Imm_SignExtends) {
    // mov rax, 0xDEADBEEF ; xor rax, 0xDEADBEEF
    // xor rax, imm32 = 48 35 ef be ad de (6 bytes); to match length-7
    // forms we could use a ModRM imm, but 48 35 (XOR RAX, imm32) is the
    // canonical RAX form. We assert behavior, not encoding length.
    const u8 program[] = {
        0x48, 0xb8, 0xef,0xbe,0xad,0xde,0x00,0x00,0x00,0x00, // mov rax, 0xDEADBEEF
        0x48, 0x35, 0xef, 0xbe, 0xad, 0xde,                  // xor rax, 0xDEADBEEF (sx)
        0xc3,
    };
    std::memcpy(mem.CodePtr(), program, sizeof(program));
    u8* guest_rsp = mem.StackTop() - 8;
    *reinterpret_cast<u64*>(guest_rsp) = kReturnSentinel;
    GuestState st{};
    st.rip = reinterpret_cast<u64>(mem.CodePtr());
    st.gpr[4] = reinterpret_cast<u64>(guest_rsp);
    st.rflags = 0x2;

    Runtime rt;
    rt.Run(st);
    // 0x00000000DEADBEEF ^ 0xFFFFFFFFDEADBEEF = 0xFFFFFFFF00000000.
    EXPECT_EQ(st.gpr[0], 0xFFFFFFFF00000000ULL) << "imm sign-extended to 64 bits";
    EXPECT_EQ(st.rflags & 0x1ULL, 0ULL) << "CF cleared by XOR";
    EXPECT_EQ(st.rflags & (1ULL<<11), 0ULL) << "OF cleared by XOR";
    EXPECT_EQ(st.rflags & (1ULL<<7), (1ULL<<7)) << "SF set: result MSB set";
}

// xor r64, imm with a register other than RAX (uses the ModRM imm32
// form, 7 bytes — matching the game's reported length). Full flip.
TEST_F(CpuRuntimeTest, Xor64_Imm_NonRax) {
    // mov rcx, 0x0123456789ABCDEF ; xor rcx, 0xFFFFFFFF (= -1 sx)
    // xor rcx, imm32 = 48 81 f1 ff ff ff ff (7 bytes)
    const u8 program[] = {
        0x48, 0xb9, 0xef,0xcd,0xab,0x89,0x67,0x45,0x23,0x01, // mov rcx, 0x0123456789ABCDEF
        0x48, 0x81, 0xf1, 0xff, 0xff, 0xff, 0xff,            // xor rcx, 0xFFFFFFFF (sx -> -1)
        0xc3,
    };
    std::memcpy(mem.CodePtr(), program, sizeof(program));
    u8* guest_rsp = mem.StackTop() - 8;
    *reinterpret_cast<u64*>(guest_rsp) = kReturnSentinel;
    GuestState st{};
    st.rip = reinterpret_cast<u64>(mem.CodePtr());
    st.gpr[4] = reinterpret_cast<u64>(guest_rsp);
    st.rflags = 0x2;

    Runtime rt;
    rt.Run(st);
    // ^ all-ones = bitwise NOT: 0x0123456789ABCDEF -> 0xFEDCBA9876543210.
    EXPECT_EQ(st.gpr[1], 0xFEDCBA9876543210ULL) << "xor with -1 flips all bits";
    EXPECT_EQ(st.rflags & (1ULL<<7), (1ULL<<7)) << "SF set: result MSB set";
    EXPECT_EQ(st.rflags & (1ULL<<6), 0ULL) << "ZF clear: result nonzero";
}

// xor r64, imm producing zero -> ZF set (value XOR itself).
TEST_F(CpuRuntimeTest, Xor64_Imm_ToZeroSetsZF) {
    // mov eax, 0x7F ; xor rax, 0x7F  -> 0, ZF=1
    const u8 program[] = {
        0xb8, 0x7f,0x00,0x00,0x00,           // mov eax, 0x7F (zero-extends rax)
        0x48, 0x83, 0xf0, 0x7f,              // xor rax, 0x7F (imm8 sx)
        0xc3,
    };
    std::memcpy(mem.CodePtr(), program, sizeof(program));
    u8* guest_rsp = mem.StackTop() - 8;
    *reinterpret_cast<u64*>(guest_rsp) = kReturnSentinel;
    GuestState st{};
    st.rip = reinterpret_cast<u64>(mem.CodePtr());
    st.gpr[4] = reinterpret_cast<u64>(guest_rsp);
    st.rflags = 0x2 | 0x1; // CF pre-set; XOR must clear it

    Runtime rt;
    rt.Run(st);
    EXPECT_EQ(st.gpr[0], 0ULL) << "x ^ x = 0";
    EXPECT_EQ(st.rflags & (1ULL<<6), (1ULL<<6)) << "ZF set: result zero";
    EXPECT_EQ(st.rflags & 0x1ULL, 0ULL) << "CF cleared by XOR";
}

// ============================================================================
// ADD qword [mem], imm — 64-bit read-modify-write add of an immediate
// to a memory location (the RMW counterpart of `add r64, imm`). Full
// ADD flags: CF on unsigned carry, OF on signed overflow, ZF/SF/PF
// from the result; imm8/imm32 sign-extended. Confirmed vs host addq.
// ============================================================================

// Basic in-memory accumulator bump; only the 8 target bytes change.
TEST_F(CpuRuntimeTest, Add64_MemImm_Basic) {
    u64* scratch = reinterpret_cast<u64*>(mem.CodePtr() + 0x100);
    scratch[0] = 0x100;
    scratch[1] = 0xBBBBBBBBBBBBBBBBULL; // sentinel after the 8-byte target
    // mov rax, scratch ; add qword [rax], 0x10
    const u8 program[] = {
        0x48, 0xb8, 0,0,0,0,0,0,0,0,
        0x48, 0x83, 0x00, 0x10, // add qword [rax], 0x10 (imm8 sx)
        0xc3,
    };
    u8 prog[sizeof(program)];
    std::memcpy(prog, program, sizeof(program));
    const u64 a = reinterpret_cast<u64>(scratch);
    std::memcpy(prog + 2, &a, sizeof(a));
    std::memcpy(mem.CodePtr(), prog, sizeof(prog));
    u8* guest_rsp = mem.StackTop() - 8;
    *reinterpret_cast<u64*>(guest_rsp) = kReturnSentinel;
    GuestState st{};
    st.rip = reinterpret_cast<u64>(mem.CodePtr());
    st.gpr[4] = reinterpret_cast<u64>(guest_rsp);
    st.rflags = 0x2;

    Runtime rt;
    rt.Run(st);
    EXPECT_EQ(scratch[0], 0x110ULL) << "memory accumulator incremented";
    EXPECT_EQ(scratch[1], 0xBBBBBBBBBBBBBBBBULL) << "next qword untouched";
    EXPECT_EQ(st.rflags & 0x1ULL, 0ULL) << "CF clear: no carry";
    EXPECT_EQ(st.rflags & (1ULL<<6), 0ULL) << "ZF clear";
}

// Unsigned carry + wrap to zero: add 1 to 0xFFFF...FF -> 0, CF=1, ZF=1.
TEST_F(CpuRuntimeTest, Add64_MemImm_CarryWrap) {
    u64* scratch = reinterpret_cast<u64*>(mem.CodePtr() + 0x100);
    scratch[0] = 0xFFFFFFFFFFFFFFFFULL;
    const u8 program[] = {
        0x48, 0xb8, 0,0,0,0,0,0,0,0,
        0x48, 0x83, 0x00, 0x01, // add qword [rax], 1
        0xc3,
    };
    u8 prog[sizeof(program)];
    std::memcpy(prog, program, sizeof(program));
    const u64 a = reinterpret_cast<u64>(scratch);
    std::memcpy(prog + 2, &a, sizeof(a));
    std::memcpy(mem.CodePtr(), prog, sizeof(prog));
    u8* guest_rsp = mem.StackTop() - 8;
    *reinterpret_cast<u64*>(guest_rsp) = kReturnSentinel;
    GuestState st{};
    st.rip = reinterpret_cast<u64>(mem.CodePtr());
    st.gpr[4] = reinterpret_cast<u64>(guest_rsp);
    st.rflags = 0x2;

    Runtime rt;
    rt.Run(st);
    EXPECT_EQ(scratch[0], 0ULL) << "wrapped to zero";
    EXPECT_EQ(st.rflags & 0x1ULL, 0x1ULL) << "CF set: unsigned carry out";
    EXPECT_EQ(st.rflags & (1ULL<<6), (1ULL<<6)) << "ZF set: result zero";
}

// Signed overflow: add 1 to INT64_MAX -> 0x8000.., OF=1, SF=1.
TEST_F(CpuRuntimeTest, Add64_MemImm_SignedOverflow) {
    u64* scratch = reinterpret_cast<u64*>(mem.CodePtr() + 0x100);
    scratch[0] = 0x7FFFFFFFFFFFFFFFULL;
    const u8 program[] = {
        0x48, 0xb8, 0,0,0,0,0,0,0,0,
        0x48, 0x83, 0x00, 0x01, // add qword [rax], 1
        0xc3,
    };
    u8 prog[sizeof(program)];
    std::memcpy(prog, program, sizeof(program));
    const u64 a = reinterpret_cast<u64>(scratch);
    std::memcpy(prog + 2, &a, sizeof(a));
    std::memcpy(mem.CodePtr(), prog, sizeof(prog));
    u8* guest_rsp = mem.StackTop() - 8;
    *reinterpret_cast<u64*>(guest_rsp) = kReturnSentinel;
    GuestState st{};
    st.rip = reinterpret_cast<u64>(mem.CodePtr());
    st.gpr[4] = reinterpret_cast<u64>(guest_rsp);
    st.rflags = 0x2;

    Runtime rt;
    rt.Run(st);
    EXPECT_EQ(scratch[0], 0x8000000000000000ULL);
    EXPECT_EQ(st.rflags & (1ULL<<11), (1ULL<<11)) << "OF set: signed overflow";
    EXPECT_EQ(st.rflags & (1ULL<<7), (1ULL<<7)) << "SF set: result negative";
}

// Negative immediate via sign-extended imm32: add -1 (0xFFFFFFFF sx).
TEST_F(CpuRuntimeTest, Add64_MemImm_NegativeSignExtended) {
    u64* scratch = reinterpret_cast<u64*>(mem.CodePtr() + 0x100);
    scratch[0] = 0x100;
    // mov rax, scratch ; add qword [rax], 0xFFFFFFFF (imm32, sx -> -1)
    // 48 81 00 ff ff ff ff (7 bytes)
    const u8 program[] = {
        0x48, 0xb8, 0,0,0,0,0,0,0,0,
        0x48, 0x81, 0x00, 0xff, 0xff, 0xff, 0xff,
        0xc3,
    };
    u8 prog[sizeof(program)];
    std::memcpy(prog, program, sizeof(program));
    const u64 a = reinterpret_cast<u64>(scratch);
    std::memcpy(prog + 2, &a, sizeof(a));
    std::memcpy(mem.CodePtr(), prog, sizeof(prog));
    u8* guest_rsp = mem.StackTop() - 8;
    *reinterpret_cast<u64*>(guest_rsp) = kReturnSentinel;
    GuestState st{};
    st.rip = reinterpret_cast<u64>(mem.CodePtr());
    st.gpr[4] = reinterpret_cast<u64>(guest_rsp);
    st.rflags = 0x2;

    Runtime rt;
    rt.Run(st);
    // 0x100 + (-1) = 0xFF; CF set (carry-out of the add-of-all-ones).
    EXPECT_EQ(scratch[0], 0xFFULL) << "imm sign-extended to -1, subtracted 1";
    EXPECT_EQ(st.rflags & 0x1ULL, 0x1ULL) << "CF set per add-of-negative semantics";
}

// ============================================================================
// SUB qword [mem], imm — 64-bit read-modify-write subtract of an immediate
// from a memory location. The mem-dst+reg form was already supported; the
// mem-dst+imm form was the run-ending gap in CUSA02394 at guest 0x8002173f5
// (48 83 6b disp8 imm8, length 5 — sub qword [rbx+disp8], imm8), an in-memory
// counter / pointer adjustment. Full SUB flags via EmitFlagsFromSubtract;
// imm8/imm32 sign-extended. Verified vs host subq.
// ============================================================================

// Basic in-memory decrement; only the 8 target bytes change.
TEST_F(CpuRuntimeTest, Sub64_MemImm_Basic) {
    u64* scratch = reinterpret_cast<u64*>(mem.CodePtr() + 0x100);
    scratch[0] = 0x100;
    scratch[1] = 0xBBBBBBBBBBBBBBBBULL; // sentinel after the 8-byte target
    const u8 program[] = {
        0x48, 0xb8, 0,0,0,0,0,0,0,0,
        0x48, 0x83, 0x28, 0x10, // sub qword [rax], 0x10 (imm8 sx)
        0xc3,
    };
    u8 prog[sizeof(program)];
    std::memcpy(prog, program, sizeof(program));
    const u64 a = reinterpret_cast<u64>(scratch);
    std::memcpy(prog + 2, &a, sizeof(a));
    std::memcpy(mem.CodePtr(), prog, sizeof(prog));
    u8* guest_rsp = mem.StackTop() - 8;
    *reinterpret_cast<u64*>(guest_rsp) = kReturnSentinel;
    GuestState st{};
    st.rip = reinterpret_cast<u64>(mem.CodePtr());
    st.gpr[4] = reinterpret_cast<u64>(guest_rsp);
    st.rflags = 0x2;

    Runtime rt;
    rt.Run(st);
    EXPECT_EQ(scratch[0], 0xF0ULL) << "memory value decremented by 0x10";
    EXPECT_EQ(scratch[1], 0xBBBBBBBBBBBBBBBBULL) << "next qword untouched";
    EXPECT_EQ(st.rflags & 0x1ULL, 0ULL) << "CF clear: no borrow";
    EXPECT_EQ(st.rflags & (1ULL<<6), 0ULL) << "ZF clear";
}

// Displaced form — the gap's actual encoding: sub qword [rax+0x10], imm8.
TEST_F(CpuRuntimeTest, Sub64_MemImmDisp_Basic) {
    u64* base = reinterpret_cast<u64*>(mem.CodePtr() + 0x100);
    u64* slot = reinterpret_cast<u64*>(mem.CodePtr() + 0x110); // base + 0x10 bytes
    *slot = 0x50;
    const u8 program[] = {
        0x48, 0xb8, 0,0,0,0,0,0,0,0,
        0x48, 0x83, 0x68, 0x10, 0x05, // sub qword [rax+0x10], 5
        0xc3,
    };
    u8 prog[sizeof(program)];
    std::memcpy(prog, program, sizeof(program));
    const u64 a = reinterpret_cast<u64>(base);
    std::memcpy(prog + 2, &a, sizeof(a));
    std::memcpy(mem.CodePtr(), prog, sizeof(prog));
    u8* guest_rsp = mem.StackTop() - 8;
    *reinterpret_cast<u64*>(guest_rsp) = kReturnSentinel;
    GuestState st{};
    st.rip = reinterpret_cast<u64>(mem.CodePtr());
    st.gpr[4] = reinterpret_cast<u64>(guest_rsp);
    st.rflags = 0x2;

    Runtime rt;
    rt.Run(st);
    EXPECT_EQ(*slot, 0x4BULL) << "displaced memory value decremented by 5";
}

// Borrow + wrap: 0 - 1 -> 0xFFFF...FF, CF=1 (unsigned borrow), ZF=0.
TEST_F(CpuRuntimeTest, Sub64_MemImm_BorrowWrap) {
    u64* scratch = reinterpret_cast<u64*>(mem.CodePtr() + 0x100);
    scratch[0] = 0;
    const u8 program[] = {
        0x48, 0xb8, 0,0,0,0,0,0,0,0,
        0x48, 0x83, 0x28, 0x01, // sub qword [rax], 1
        0xc3,
    };
    u8 prog[sizeof(program)];
    std::memcpy(prog, program, sizeof(program));
    const u64 a = reinterpret_cast<u64>(scratch);
    std::memcpy(prog + 2, &a, sizeof(a));
    std::memcpy(mem.CodePtr(), prog, sizeof(prog));
    u8* guest_rsp = mem.StackTop() - 8;
    *reinterpret_cast<u64*>(guest_rsp) = kReturnSentinel;
    GuestState st{};
    st.rip = reinterpret_cast<u64>(mem.CodePtr());
    st.gpr[4] = reinterpret_cast<u64>(guest_rsp);
    st.rflags = 0x2;

    Runtime rt;
    rt.Run(st);
    EXPECT_EQ(scratch[0], 0xFFFFFFFFFFFFFFFFULL) << "0 - 1 wraps";
    EXPECT_EQ(st.rflags & 0x1ULL, 0x1ULL) << "CF set: unsigned borrow";
    EXPECT_EQ(st.rflags & (1ULL<<6), 0ULL) << "ZF clear";
}

// Signed overflow: INT64_MIN - 1 -> 0x7FFF...FF, OF=1, SF=0.
TEST_F(CpuRuntimeTest, Sub64_MemImm_SignedOverflow) {
    u64* scratch = reinterpret_cast<u64*>(mem.CodePtr() + 0x100);
    scratch[0] = 0x8000000000000000ULL;
    const u8 program[] = {
        0x48, 0xb8, 0,0,0,0,0,0,0,0,
        0x48, 0x83, 0x28, 0x01, // sub qword [rax], 1
        0xc3,
    };
    u8 prog[sizeof(program)];
    std::memcpy(prog, program, sizeof(program));
    const u64 a = reinterpret_cast<u64>(scratch);
    std::memcpy(prog + 2, &a, sizeof(a));
    std::memcpy(mem.CodePtr(), prog, sizeof(prog));
    u8* guest_rsp = mem.StackTop() - 8;
    *reinterpret_cast<u64*>(guest_rsp) = kReturnSentinel;
    GuestState st{};
    st.rip = reinterpret_cast<u64>(mem.CodePtr());
    st.gpr[4] = reinterpret_cast<u64>(guest_rsp);
    st.rflags = 0x2;

    Runtime rt;
    rt.Run(st);
    EXPECT_EQ(scratch[0], 0x7FFFFFFFFFFFFFFFULL);
    EXPECT_EQ(st.rflags & (1ULL<<11), (1ULL<<11)) << "OF set: signed overflow";
    EXPECT_EQ(st.rflags & (1ULL<<7), 0ULL) << "SF clear: result positive";
}

// Negative immediate via sign-extended imm32: sub qword [rax], -1 means +1.
TEST_F(CpuRuntimeTest, Sub64_MemImm_NegativeSignExtended) {
    u64* scratch = reinterpret_cast<u64*>(mem.CodePtr() + 0x100);
    scratch[0] = 0x100;
    // sub qword [rax], 0xFFFFFFFF (imm32, sx -> -1): 0x100 - (-1) = 0x101
    const u8 program[] = {
        0x48, 0xb8, 0,0,0,0,0,0,0,0,
        0x48, 0x81, 0x28, 0xff, 0xff, 0xff, 0xff,
        0xc3,
    };
    u8 prog[sizeof(program)];
    std::memcpy(prog, program, sizeof(program));
    const u64 a = reinterpret_cast<u64>(scratch);
    std::memcpy(prog + 2, &a, sizeof(a));
    std::memcpy(mem.CodePtr(), prog, sizeof(prog));
    u8* guest_rsp = mem.StackTop() - 8;
    *reinterpret_cast<u64*>(guest_rsp) = kReturnSentinel;
    GuestState st{};
    st.rip = reinterpret_cast<u64>(mem.CodePtr());
    st.gpr[4] = reinterpret_cast<u64>(guest_rsp);
    st.rflags = 0x2;

    Runtime rt;
    rt.Run(st);
    EXPECT_EQ(scratch[0], 0x101ULL) << "subtracting -1 adds 1";
}

// ============================================================================
// SBB r32, r32 — 32-bit subtract with borrow: dst = dst - src - CF.
// The defining trait: it consumes the incoming CF as a borrow, so the
// SAME operands give different results depending on CF. Routed through
// EmitAdcSbb64 (which round-trips flags so host CF == guest CF).
// Values confirmed against host sbb.
// ============================================================================

// CF=0: behaves like plain SUB. 0x100 - 0x10 - 0 = 0xF0.
TEST_F(CpuRuntimeTest, Sbb32_RegReg_NoBorrowIn) {
    // sbb eax, ecx  — 19 c8 (2 bytes... but Zydis reports len 3 for the
    // game's encoding; the operation is identical). We use 19 c8.
    const u8 program[] = {
        0x19, 0xc8, // sbb eax, ecx
        0xc3,
    };
    std::memcpy(mem.CodePtr(), program, sizeof(program));
    u8* guest_rsp = mem.StackTop() - 8;
    *reinterpret_cast<u64*>(guest_rsp) = kReturnSentinel;
    GuestState st{};
    st.rip = reinterpret_cast<u64>(mem.CodePtr());
    st.gpr[4] = reinterpret_cast<u64>(guest_rsp);
    st.gpr[0] = 0x100;  // eax
    st.gpr[1] = 0x10;   // ecx
    st.rflags = 0x2;    // CF = 0

    Runtime rt;
    rt.Run(st);
    EXPECT_EQ(st.gpr[0], 0xF0ULL) << "0x100 - 0x10 - 0 = 0xF0";
    EXPECT_EQ(st.rflags & 0x1ULL, 0ULL) << "CF clear: no borrow out";
}

// CF=1: subtracts an extra 1. Same operands as above -> 0xEF, not 0xF0.
// This is the test that proves the borrow-in is honored.
TEST_F(CpuRuntimeTest, Sbb32_RegReg_BorrowIn) {
    const u8 program[] = {
        0x19, 0xc8, // sbb eax, ecx
        0xc3,
    };
    std::memcpy(mem.CodePtr(), program, sizeof(program));
    u8* guest_rsp = mem.StackTop() - 8;
    *reinterpret_cast<u64*>(guest_rsp) = kReturnSentinel;
    GuestState st{};
    st.rip = reinterpret_cast<u64>(mem.CodePtr());
    st.gpr[4] = reinterpret_cast<u64>(guest_rsp);
    st.gpr[0] = 0x100;
    st.gpr[1] = 0x10;
    st.rflags = 0x2 | 0x1; // CF = 1

    Runtime rt;
    rt.Run(st);
    EXPECT_EQ(st.gpr[0], 0xEFULL) << "0x100 - 0x10 - 1 = 0xEF (borrow-in honored)";
    EXPECT_EQ(st.rflags & 0x1ULL, 0ULL) << "CF clear: no borrow out";
}

// Underflow with borrow-in: 5 - 5 - 1 = 0xFFFFFFFF, CF=1 (borrow out),
// SF=1, and the result zero-extends into the full 64-bit slot.
TEST_F(CpuRuntimeTest, Sbb32_RegReg_BorrowOut_ZeroExtends) {
    const u8 program[] = {
        0x19, 0xc8, // sbb eax, ecx
        0xc3,
    };
    std::memcpy(mem.CodePtr(), program, sizeof(program));
    u8* guest_rsp = mem.StackTop() - 8;
    *reinterpret_cast<u64*>(guest_rsp) = kReturnSentinel;
    GuestState st{};
    st.rip = reinterpret_cast<u64>(mem.CodePtr());
    st.gpr[4] = reinterpret_cast<u64>(guest_rsp);
    st.gpr[0] = 0xFFFFFFFF00000005ULL; // eax=5, high junk -> must be zero-extended
    st.gpr[1] = 0x5;                   // ecx
    st.rflags = 0x2 | 0x1; // CF=1

    Runtime rt;
    rt.Run(st);
    EXPECT_EQ(st.gpr[0], 0x00000000FFFFFFFFULL) << "5-5-1 wraps; upper 32 zero-extended";
    EXPECT_EQ(st.rflags & 0x1ULL, 0x1ULL) << "CF set: borrow out";
    EXPECT_EQ(st.rflags & (1ULL<<7), (1ULL<<7)) << "SF set: result MSB (bit31) set";
}

// Result zero -> ZF. 0x10 - 0x10 - 0 = 0.
TEST_F(CpuRuntimeTest, Sbb32_RegReg_ToZero) {
    const u8 program[] = {
        0x19, 0xc8, // sbb eax, ecx
        0xc3,
    };
    std::memcpy(mem.CodePtr(), program, sizeof(program));
    u8* guest_rsp = mem.StackTop() - 8;
    *reinterpret_cast<u64*>(guest_rsp) = kReturnSentinel;
    GuestState st{};
    st.rip = reinterpret_cast<u64>(mem.CodePtr());
    st.gpr[4] = reinterpret_cast<u64>(guest_rsp);
    st.gpr[0] = 0x10;
    st.gpr[1] = 0x10;
    st.rflags = 0x2; // CF=0

    Runtime rt;
    rt.Run(st);
    EXPECT_EQ(st.gpr[0], 0ULL);
    EXPECT_EQ(st.rflags & (1ULL<<6), (1ULL<<6)) << "ZF set: result zero";
    EXPECT_EQ(st.rflags & 0x1ULL, 0ULL) << "CF clear";
}

// ============================================================================
// INC r8 — 8-bit register increment. Preserves CF (the INC quirk),
// sets ZF/SF/OF/PF/AF, preserves the upper 56 bits of the parent
// register, and addresses the correct byte for high-byte regs (AH/etc).
// Confirmed vs host incb.
// ============================================================================

// inc al: byte increments, upper 56 bits of rax preserved, CF preserved.
TEST_F(CpuRuntimeTest, Inc8_AL_PreservesUpperAndCF) {
    // inc al  — fe c0 (2 bytes)
    const u8 program[] = {
        0xfe, 0xc0,
        0xc3,
    };
    std::memcpy(mem.CodePtr(), program, sizeof(program));
    u8* guest_rsp = mem.StackTop() - 8;
    *reinterpret_cast<u64*>(guest_rsp) = kReturnSentinel;
    GuestState st{};
    st.rip = reinterpret_cast<u64>(mem.CodePtr());
    st.gpr[4] = reinterpret_cast<u64>(guest_rsp);
    st.gpr[0] = 0xDEADBEEFDEAD0041ULL; // al = 0x41, upper must survive
    st.rflags = 0x2 | 0x1; // CF pre-set

    Runtime rt;
    rt.Run(st);
    EXPECT_EQ(st.gpr[0], 0xDEADBEEFDEAD0042ULL) << "al incremented, upper 56 bits preserved";
    EXPECT_EQ(st.rflags & 0x1ULL, 0x1ULL) << "CF preserved (INC does not affect CF)";
    EXPECT_EQ(st.rflags & (1ULL<<6), 0ULL) << "ZF clear";
}

// inc ah: the high-byte register increments byte 1 of rax, leaving
// al (byte 0) and the rest untouched. Proves the +1 byte offset.
TEST_F(CpuRuntimeTest, Inc8_AH_HighByte) {
    // inc ah  — fe c4 (2 bytes)
    const u8 program[] = {
        0xfe, 0xc4,
        0xc3,
    };
    std::memcpy(mem.CodePtr(), program, sizeof(program));
    u8* guest_rsp = mem.StackTop() - 8;
    *reinterpret_cast<u64*>(guest_rsp) = kReturnSentinel;
    GuestState st{};
    st.rip = reinterpret_cast<u64>(mem.CodePtr());
    st.gpr[4] = reinterpret_cast<u64>(guest_rsp);
    st.gpr[0] = 0x00000000000012FFULL; // ah=0x12, al=0xFF
    st.rflags = 0x2;

    Runtime rt;
    rt.Run(st);
    // ah: 0x12 -> 0x13; al (0xFF) and all other bytes unchanged.
    EXPECT_EQ(st.gpr[0], 0x00000000000013FFULL) << "ah incremented, al untouched";
}

// Wrap 0xFF -> 0: ZF set, CF stays clear (INC never sets CF on wrap).
TEST_F(CpuRuntimeTest, Inc8_AL_WrapSetsZF) {
    const u8 program[] = {
        0xfe, 0xc0, // inc al
        0xc3,
    };
    std::memcpy(mem.CodePtr(), program, sizeof(program));
    u8* guest_rsp = mem.StackTop() - 8;
    *reinterpret_cast<u64*>(guest_rsp) = kReturnSentinel;
    GuestState st{};
    st.rip = reinterpret_cast<u64>(mem.CodePtr());
    st.gpr[4] = reinterpret_cast<u64>(guest_rsp);
    st.gpr[0] = 0xAAAAAAAAAAAAAAFFULL; // al = 0xFF
    st.rflags = 0x2; // CF clear

    Runtime rt;
    rt.Run(st);
    EXPECT_EQ(st.gpr[0] & 0xFFULL, 0ULL) << "al wrapped to 0";
    EXPECT_EQ(st.gpr[0] & ~0xFFULL, 0xAAAAAAAAAAAAAA00ULL) << "upper bytes preserved";
    EXPECT_EQ(st.rflags & (1ULL<<6), (1ULL<<6)) << "ZF set: byte result zero";
    EXPECT_EQ(st.rflags & 0x1ULL, 0ULL) << "CF still clear on wrap";
}

// Signed overflow at 8-bit boundary: 0x7F -> 0x80 sets OF and SF.
TEST_F(CpuRuntimeTest, Inc8_AL_SignedOverflow) {
    const u8 program[] = {
        0xfe, 0xc0, // inc al
        0xc3,
    };
    std::memcpy(mem.CodePtr(), program, sizeof(program));
    u8* guest_rsp = mem.StackTop() - 8;
    *reinterpret_cast<u64*>(guest_rsp) = kReturnSentinel;
    GuestState st{};
    st.rip = reinterpret_cast<u64>(mem.CodePtr());
    st.gpr[4] = reinterpret_cast<u64>(guest_rsp);
    st.gpr[0] = 0x7F;
    st.rflags = 0x2;

    Runtime rt;
    rt.Run(st);
    EXPECT_EQ(st.gpr[0] & 0xFFULL, 0x80ULL);
    EXPECT_EQ(st.rflags & (1ULL<<11), (1ULL<<11)) << "OF set: 8-bit signed overflow";
    EXPECT_EQ(st.rflags & (1ULL<<7), (1ULL<<7)) << "SF set: result negative";
}

// ============================================================================
// DIV r8 — 8-bit unsigned divide. Distinct implicit-operand layout from
// the wider forms: dividend is the full 16-bit AX (not a DX:AX pair),
// quotient -> AL, remainder -> AH. The upper 48 bits of RAX are
// preserved. DIV's flags are architecturally undefined, so not
// asserted. Values confirmed against host divb.
// ============================================================================

// 100 / 7 = q14 r2. Dividend fits in AL but uses the AX layout.
TEST_F(CpuRuntimeTest, Div8_Basic) {
    // div cl  — f6 f1 (2 bytes)
    const u8 program[] = {
        0xf6, 0xf1,
        0xc3,
    };
    std::memcpy(mem.CodePtr(), program, sizeof(program));
    u8* guest_rsp = mem.StackTop() - 8;
    *reinterpret_cast<u64*>(guest_rsp) = kReturnSentinel;
    GuestState st{};
    st.rip = reinterpret_cast<u64>(mem.CodePtr());
    st.gpr[4] = reinterpret_cast<u64>(guest_rsp);
    // ax = 100 (dividend), upper bits poisoned to verify preservation.
    st.gpr[0] = 0xDEADBEEFDEAD0064ULL; // ax = 0x0064 = 100
    st.gpr[1] = 0x7;                   // cl = 7 (divisor)

    Runtime rt;
    rt.Run(st);
    // AL = quotient = 14 (0x0E), AH = remainder = 2 -> AX = 0x020E.
    EXPECT_EQ(st.gpr[0] & 0xFFFFULL, 0x020EULL) << "AL=quotient 14, AH=remainder 2";
    EXPECT_EQ(st.gpr[0] & ~0xFFFFULL, 0xDEADBEEFDEAD0000ULL) << "upper 48 bits preserved";
}

// 0x1234 / 0x77: dividend > 255, proving the full 16-bit AX is the
// dividend (not just AL). q=39 (0x27), r=19 (0x13) -> AX = 0x1327.
TEST_F(CpuRuntimeTest, Div8_FullAxDividend) {
    const u8 program[] = {
        0xf6, 0xf1, // div cl
        0xc3,
    };
    std::memcpy(mem.CodePtr(), program, sizeof(program));
    u8* guest_rsp = mem.StackTop() - 8;
    *reinterpret_cast<u64*>(guest_rsp) = kReturnSentinel;
    GuestState st{};
    st.rip = reinterpret_cast<u64>(mem.CodePtr());
    st.gpr[4] = reinterpret_cast<u64>(guest_rsp);
    st.gpr[0] = 0x0000000000001234ULL; // ax = 0x1234 = 4660
    st.gpr[1] = 0x77;                  // cl = 0x77 = 119

    Runtime rt;
    rt.Run(st);
    // 4660 / 119 = 39 rem 19 -> AL=0x27, AH=0x13 -> AX=0x1327.
    EXPECT_EQ(st.gpr[0] & 0xFFFFULL, 0x1327ULL) << "16-bit AX dividend: q=39, r=19";
}

// Divisor in a different register (bl), exact division -> remainder 0.
TEST_F(CpuRuntimeTest, Div8_ExactDivision_NonClDivisor) {
    // div bl  — f6 f3 (2 bytes)
    const u8 program[] = {
        0xf6, 0xf3,
        0xc3,
    };
    std::memcpy(mem.CodePtr(), program, sizeof(program));
    u8* guest_rsp = mem.StackTop() - 8;
    *reinterpret_cast<u64*>(guest_rsp) = kReturnSentinel;
    GuestState st{};
    st.rip = reinterpret_cast<u64>(mem.CodePtr());
    st.gpr[4] = reinterpret_cast<u64>(guest_rsp);
    st.gpr[0] = 0xF0;  // ax = 240
    st.gpr[3] = 0x10;  // bl = 16 (divisor; RBX = gpr index 3)

    Runtime rt;
    rt.Run(st);
    // 240 / 16 = 15 rem 0 -> AL=0x0F, AH=0x00 -> AX=0x000F.
    EXPECT_EQ(st.gpr[0] & 0xFFFFULL, 0x000FULL) << "q=15, r=0 (exact)";
}

// ============================================================================
// DEC r8 — 8-bit register decrement (symmetric companion of INC r8).
// Preserves CF (the DEC quirk), sets ZF/SF/OF/PF/AF, preserves the
// upper 56 bits of the parent register, and addresses the correct byte
// for high-byte regs (AH/etc). Confirmed vs host decb.
// ============================================================================

// dec al: byte decrements, upper 56 bits preserved, CF preserved.
TEST_F(CpuRuntimeTest, Dec8_AL_PreservesUpperAndCF) {
    // dec al  — fe c8 (2 bytes)
    const u8 program[] = {
        0xfe, 0xc8,
        0xc3,
    };
    std::memcpy(mem.CodePtr(), program, sizeof(program));
    u8* guest_rsp = mem.StackTop() - 8;
    *reinterpret_cast<u64*>(guest_rsp) = kReturnSentinel;
    GuestState st{};
    st.rip = reinterpret_cast<u64>(mem.CodePtr());
    st.gpr[4] = reinterpret_cast<u64>(guest_rsp);
    st.gpr[0] = 0xDEADBEEFDEAD0042ULL; // al = 0x42, upper must survive
    st.rflags = 0x2 | 0x1; // CF pre-set

    Runtime rt;
    rt.Run(st);
    EXPECT_EQ(st.gpr[0], 0xDEADBEEFDEAD0041ULL) << "al decremented, upper 56 bits preserved";
    EXPECT_EQ(st.rflags & 0x1ULL, 0x1ULL) << "CF preserved (DEC does not affect CF)";
    EXPECT_EQ(st.rflags & (1ULL<<6), 0ULL) << "ZF clear";
}

// dec to zero: 0x01 -> 0x00 sets ZF.
TEST_F(CpuRuntimeTest, Dec8_AL_ToZeroSetsZF) {
    const u8 program[] = {
        0xfe, 0xc8, // dec al
        0xc3,
    };
    std::memcpy(mem.CodePtr(), program, sizeof(program));
    u8* guest_rsp = mem.StackTop() - 8;
    *reinterpret_cast<u64*>(guest_rsp) = kReturnSentinel;
    GuestState st{};
    st.rip = reinterpret_cast<u64>(mem.CodePtr());
    st.gpr[4] = reinterpret_cast<u64>(guest_rsp);
    st.gpr[0] = 0x01;
    st.rflags = 0x2;

    Runtime rt;
    rt.Run(st);
    EXPECT_EQ(st.gpr[0] & 0xFFULL, 0ULL);
    EXPECT_EQ(st.rflags & (1ULL<<6), (1ULL<<6)) << "ZF set: result zero";
    EXPECT_EQ(st.rflags & 0x1ULL, 0ULL) << "CF clear";
}

// Underflow 0x00 -> 0xFF: SF set (result negative), CF stays clear.
TEST_F(CpuRuntimeTest, Dec8_AL_UnderflowSetsSFNotCF) {
    const u8 program[] = {
        0xfe, 0xc8, // dec al
        0xc3,
    };
    std::memcpy(mem.CodePtr(), program, sizeof(program));
    u8* guest_rsp = mem.StackTop() - 8;
    *reinterpret_cast<u64*>(guest_rsp) = kReturnSentinel;
    GuestState st{};
    st.rip = reinterpret_cast<u64>(mem.CodePtr());
    st.gpr[4] = reinterpret_cast<u64>(guest_rsp);
    st.gpr[0] = 0xAAAAAAAAAAAAAA00ULL; // al = 0x00
    st.rflags = 0x2; // CF clear

    Runtime rt;
    rt.Run(st);
    EXPECT_EQ(st.gpr[0] & 0xFFULL, 0xFFULL) << "al underflowed to 0xFF";
    EXPECT_EQ(st.gpr[0] & ~0xFFULL, 0xAAAAAAAAAAAAAA00ULL) << "upper bytes preserved";
    EXPECT_EQ(st.rflags & (1ULL<<7), (1ULL<<7)) << "SF set: result negative";
    EXPECT_EQ(st.rflags & 0x1ULL, 0ULL) << "CF still clear on underflow";
}

// Signed overflow: 0x80 (INT8_MIN) -> 0x7F sets OF.
TEST_F(CpuRuntimeTest, Dec8_AL_SignedOverflow) {
    const u8 program[] = {
        0xfe, 0xc8, // dec al
        0xc3,
    };
    std::memcpy(mem.CodePtr(), program, sizeof(program));
    u8* guest_rsp = mem.StackTop() - 8;
    *reinterpret_cast<u64*>(guest_rsp) = kReturnSentinel;
    GuestState st{};
    st.rip = reinterpret_cast<u64>(mem.CodePtr());
    st.gpr[4] = reinterpret_cast<u64>(guest_rsp);
    st.gpr[0] = 0x80;
    st.rflags = 0x2;

    Runtime rt;
    rt.Run(st);
    EXPECT_EQ(st.gpr[0] & 0xFFULL, 0x7FULL);
    EXPECT_EQ(st.rflags & (1ULL<<11), (1ULL<<11)) << "OF set: 0x80 - 1 signed overflow";
    EXPECT_EQ(st.rflags & (1ULL<<7), 0ULL) << "SF clear: result 0x7F positive";
}

// High-byte register: dec ah decrements byte 1 of rax, al untouched.
TEST_F(CpuRuntimeTest, Dec8_AH_HighByte) {
    // dec ah  — fe cc (2 bytes)
    const u8 program[] = {
        0xfe, 0xcc,
        0xc3,
    };
    std::memcpy(mem.CodePtr(), program, sizeof(program));
    u8* guest_rsp = mem.StackTop() - 8;
    *reinterpret_cast<u64*>(guest_rsp) = kReturnSentinel;
    GuestState st{};
    st.rip = reinterpret_cast<u64>(mem.CodePtr());
    st.gpr[4] = reinterpret_cast<u64>(guest_rsp);
    st.gpr[0] = 0x0000000000001200ULL; // ah=0x12, al=0x00
    st.rflags = 0x2;

    Runtime rt;
    rt.Run(st);
    // ah: 0x12 -> 0x11; al (0x00) untouched.
    EXPECT_EQ(st.gpr[0], 0x0000000000001100ULL) << "ah decremented, al untouched";
}

// ============================================================================
// MOV r8, r8 — 8-bit register-to-register move. Must touch exactly one
// byte of each parent register (preserving the other 56 bits) and
// correctly address high-byte registers (AH/CH/DH/BH), which live at
// parent-slot offset+1. The previous code used GprOffset/ZydisGprToIndex,
// which only reach the low byte and reject high-byte regs outright.
// ============================================================================

// Low-byte reg-reg (regression guard): mov cl, al. Only CL changes.
TEST_F(CpuRuntimeTest, Mov8_RegReg_LowBytes) {
    // mov cl, al  — 88 c1 (2 bytes)
    const u8 program[] = {
        0x88, 0xc1,
        0xc3,
    };
    std::memcpy(mem.CodePtr(), program, sizeof(program));
    u8* guest_rsp = mem.StackTop() - 8;
    *reinterpret_cast<u64*>(guest_rsp) = kReturnSentinel;
    GuestState st{};
    st.rip = reinterpret_cast<u64>(mem.CodePtr());
    st.gpr[4] = reinterpret_cast<u64>(guest_rsp);
    st.gpr[0] = 0x11111111111111AAULL; // al = 0xAA
    st.gpr[1] = 0x2222222222222233ULL; // cl = 0x33 -> becomes 0xAA

    Runtime rt;
    rt.Run(st);
    EXPECT_EQ(st.gpr[1], 0x22222222222222AAULL) << "cl = al, upper bits of rcx preserved";
    EXPECT_EQ(st.gpr[0], 0x11111111111111AAULL) << "rax untouched";
}

// High-byte DESTINATION: mov ah, bl. Writes byte 1 of rax (AH),
// leaving al and the upper bytes untouched. Previously rejected.
TEST_F(CpuRuntimeTest, Mov8_RegReg_HighByteDest) {
    // mov ah, bl  — 88 dc (2 bytes)
    const u8 program[] = {
        0x88, 0xdc,
        0xc3,
    };
    std::memcpy(mem.CodePtr(), program, sizeof(program));
    u8* guest_rsp = mem.StackTop() - 8;
    *reinterpret_cast<u64*>(guest_rsp) = kReturnSentinel;
    GuestState st{};
    st.rip = reinterpret_cast<u64>(mem.CodePtr());
    st.gpr[4] = reinterpret_cast<u64>(guest_rsp);
    st.gpr[0] = 0x00000000000000CCULL; // ah=0x00, al=0xCC
    st.gpr[3] = 0x000000000000007BULL; // bl=0x7B (RBX=index 3)

    Runtime rt;
    rt.Run(st);
    // ah := bl (0x7B) -> rax low 16 = 0x7BCC; al unchanged.
    EXPECT_EQ(st.gpr[0], 0x0000000000007BCCULL) << "ah=bl, al preserved";
}

// High-byte SOURCE: mov bl, ah. Reads byte 1 of rax (AH) into bl.
// Previously the source read would have grabbed AL instead of AH.
TEST_F(CpuRuntimeTest, Mov8_RegReg_HighByteSource) {
    // mov bl, ah  — 88 e3 (2 bytes)
    const u8 program[] = {
        0x88, 0xe3,
        0xc3,
    };
    std::memcpy(mem.CodePtr(), program, sizeof(program));
    u8* guest_rsp = mem.StackTop() - 8;
    *reinterpret_cast<u64*>(guest_rsp) = kReturnSentinel;
    GuestState st{};
    st.rip = reinterpret_cast<u64>(mem.CodePtr());
    st.gpr[4] = reinterpret_cast<u64>(guest_rsp);
    st.gpr[0] = 0x0000000000005678ULL; // ah=0x56, al=0x78
    st.gpr[3] = 0xFFFFFFFFFFFFFF00ULL; // bl=0x00 -> becomes 0x56

    Runtime rt;
    rt.Run(st);
    // bl := ah (0x56); upper bits of rbx preserved.
    EXPECT_EQ(st.gpr[3], 0xFFFFFFFFFFFFFF56ULL) << "bl=ah (0x56, not al 0x78)";
    EXPECT_EQ(st.gpr[0], 0x0000000000005678ULL) << "rax untouched";
}

// Both operands high-byte-ish: mov al, ah copies byte1 into byte0.
TEST_F(CpuRuntimeTest, Mov8_RegReg_AhToAl) {
    // mov al, ah  — 88 e0 (2 bytes)
    const u8 program[] = {
        0x88, 0xe0,
        0xc3,
    };
    std::memcpy(mem.CodePtr(), program, sizeof(program));
    u8* guest_rsp = mem.StackTop() - 8;
    *reinterpret_cast<u64*>(guest_rsp) = kReturnSentinel;
    GuestState st{};
    st.rip = reinterpret_cast<u64>(mem.CodePtr());
    st.gpr[4] = reinterpret_cast<u64>(guest_rsp);
    st.gpr[0] = 0xDEADBEEF0000AB12ULL; // ah=0xAB, al=0x12

    Runtime rt;
    rt.Run(st);
    // al := ah (0xAB); ah and upper bits unchanged -> low16 = 0xABAB.
    EXPECT_EQ(st.gpr[0], 0xDEADBEEF0000ABABULL) << "al=ah; ah and upper preserved";
}

// ============================================================================
// CLD / STD — clear / set the direction flag (DF, rflags bit 10).
// Controls string-op direction (auto-inc when DF=0, auto-dec when
// DF=1). Touch no other flag and no register. CLD seen at libc
// 0x8081df7f4 before a REP string op.
// ============================================================================

// CLD clears DF (bit 10) while leaving the surrounding flags intact.
TEST_F(CpuRuntimeTest, Cld_ClearsDirectionFlag) {
    // cld  — fc (1 byte)
    const u8 program[] = {
        0xfc,
        0xc3,
    };
    std::memcpy(mem.CodePtr(), program, sizeof(program));
    u8* guest_rsp = mem.StackTop() - 8;
    *reinterpret_cast<u64*>(guest_rsp) = kReturnSentinel;
    GuestState st{};
    st.rip = reinterpret_cast<u64>(mem.CodePtr());
    st.gpr[4] = reinterpret_cast<u64>(guest_rsp);
    // DF set, plus CF and ZF set, to confirm only DF is cleared.
    st.rflags = 0x2 | (1ULL<<10) | 0x1 | (1ULL<<6);

    Runtime rt;
    rt.Run(st);
    EXPECT_EQ(st.rflags & (1ULL<<10), 0ULL) << "DF cleared";
    EXPECT_EQ(st.rflags & 0x1ULL, 0x1ULL) << "CF preserved";
    EXPECT_EQ(st.rflags & (1ULL<<6), (1ULL<<6)) << "ZF preserved";
}

// CLD when DF already clear is a no-op (idempotent), other flags intact.
TEST_F(CpuRuntimeTest, Cld_AlreadyClear_Idempotent) {
    const u8 program[] = {
        0xfc, // cld
        0xc3,
    };
    std::memcpy(mem.CodePtr(), program, sizeof(program));
    u8* guest_rsp = mem.StackTop() - 8;
    *reinterpret_cast<u64*>(guest_rsp) = kReturnSentinel;
    GuestState st{};
    st.rip = reinterpret_cast<u64>(mem.CodePtr());
    st.gpr[4] = reinterpret_cast<u64>(guest_rsp);
    st.rflags = 0x2 | (1ULL<<7); // DF clear, SF set
    Runtime rt;
    rt.Run(st);
    EXPECT_EQ(st.rflags & (1ULL<<10), 0ULL) << "DF stays clear";
    EXPECT_EQ(st.rflags & (1ULL<<7), (1ULL<<7)) << "SF preserved";
}

// STD sets DF (bit 10) while leaving the surrounding flags intact.
TEST_F(CpuRuntimeTest, Std_SetsDirectionFlag) {
    // std  — fd (1 byte)
    const u8 program[] = {
        0xfd,
        0xc3,
    };
    std::memcpy(mem.CodePtr(), program, sizeof(program));
    u8* guest_rsp = mem.StackTop() - 8;
    *reinterpret_cast<u64*>(guest_rsp) = kReturnSentinel;
    GuestState st{};
    st.rip = reinterpret_cast<u64>(mem.CodePtr());
    st.gpr[4] = reinterpret_cast<u64>(guest_rsp);
    // DF clear, CF set; confirm DF goes on and CF survives.
    st.rflags = 0x2 | 0x1;

    Runtime rt;
    rt.Run(st);
    EXPECT_EQ(st.rflags & (1ULL<<10), (1ULL<<10)) << "DF set";
    EXPECT_EQ(st.rflags & 0x1ULL, 0x1ULL) << "CF preserved";
}

// ============================================================================
// REPE CMPSQ — compare-string quadword, repeat-while-equal. The memcmp
// inner loop: compare [RSI] vs [RDI] (CMP flags), advance both by ±8
// (per DF), decrement RCX; continue while RCX!=0 AND ZF==1. Final
// flags reflect the last compare; RSI/RDI/RCX hold their residual
// loop values. Confirmed against host `repe cmpsq`.
// ============================================================================

// Equal arrays: runs to count exhaustion. RCX->0, ZF=1, both pointers
// advanced by count*8.
TEST_F(CpuRuntimeTest, RepeCmpsq_EqualArrays_RunsToCount) {
    u64* a = reinterpret_cast<u64*>(mem.CodePtr() + 0x100);
    u64* b = reinterpret_cast<u64*>(mem.CodePtr() + 0x200);
    for (int i = 0; i < 4; ++i) { a[i] = 100 + i; b[i] = 100 + i; }
    // repe cmpsq  — f3 48 a7 (3 bytes)
    const u8 program[] = { 0xf3, 0x48, 0xa7, 0xc3 };
    std::memcpy(mem.CodePtr(), program, sizeof(program));
    u8* guest_rsp = mem.StackTop() - 8;
    *reinterpret_cast<u64*>(guest_rsp) = kReturnSentinel;
    GuestState st{};
    st.rip = reinterpret_cast<u64>(mem.CodePtr());
    st.gpr[4] = reinterpret_cast<u64>(guest_rsp);
    st.gpr[6] = reinterpret_cast<u64>(a); // RSI
    st.gpr[7] = reinterpret_cast<u64>(b); // RDI
    st.gpr[1] = 4;                         // RCX = element count
    st.rflags = 0x2; // DF=0 (forward)

    Runtime rt;
    rt.Run(st);
    EXPECT_EQ(st.gpr[1], 0ULL) << "RCX exhausted";
    EXPECT_EQ(st.rflags & (1ULL<<6), (1ULL<<6)) << "ZF=1: last compare equal";
    EXPECT_EQ(st.gpr[6], reinterpret_cast<u64>(a) + 32) << "RSI += 4*8";
    EXPECT_EQ(st.gpr[7], reinterpret_cast<u64>(b) + 32) << "RDI += 4*8";
}

// Mismatch at index 2: stops after comparing the 3rd element. RCX=1,
// ZF=0, RSI/RDI advanced by 3*8 (point past the mismatching element).
TEST_F(CpuRuntimeTest, RepeCmpsq_StopsOnMismatch) {
    u64* a = reinterpret_cast<u64*>(mem.CodePtr() + 0x100);
    u64* b = reinterpret_cast<u64*>(mem.CodePtr() + 0x200);
    a[0]=1; a[1]=2; a[2]=9; a[3]=4;
    b[0]=1; b[1]=2; b[2]=3; b[3]=4; // differ at index 2
    const u8 program[] = { 0xf3, 0x48, 0xa7, 0xc3 };
    std::memcpy(mem.CodePtr(), program, sizeof(program));
    u8* guest_rsp = mem.StackTop() - 8;
    *reinterpret_cast<u64*>(guest_rsp) = kReturnSentinel;
    GuestState st{};
    st.rip = reinterpret_cast<u64>(mem.CodePtr());
    st.gpr[4] = reinterpret_cast<u64>(guest_rsp);
    st.gpr[6] = reinterpret_cast<u64>(a);
    st.gpr[7] = reinterpret_cast<u64>(b);
    st.gpr[1] = 4;
    st.rflags = 0x2;

    Runtime rt;
    rt.Run(st);
    EXPECT_EQ(st.gpr[1], 1ULL) << "RCX = 1 (3 compares done)";
    EXPECT_EQ(st.rflags & (1ULL<<6), 0ULL) << "ZF=0: mismatch found";
    EXPECT_EQ(st.gpr[6], reinterpret_cast<u64>(a) + 24) << "RSI past mismatch (3*8)";
    EXPECT_EQ(st.gpr[7], reinterpret_cast<u64>(b) + 24) << "RDI past mismatch (3*8)";
    // 9 > 3 unsigned -> last compare (a-b) had no borrow: CF=0.
    EXPECT_EQ(st.rflags & 0x1ULL, 0ULL) << "CF reflects last compare (9-3, no borrow)";
}

// Zero count: RCX==0 -> no compare, no pointer movement, flags
// unchanged.
TEST_F(CpuRuntimeTest, RepeCmpsq_ZeroCount_NoOp) {
    u64* a = reinterpret_cast<u64*>(mem.CodePtr() + 0x100);
    u64* b = reinterpret_cast<u64*>(mem.CodePtr() + 0x200);
    a[0]=0xAAAA; b[0]=0xBBBB; // would mismatch if compared
    const u8 program[] = { 0xf3, 0x48, 0xa7, 0xc3 };
    std::memcpy(mem.CodePtr(), program, sizeof(program));
    u8* guest_rsp = mem.StackTop() - 8;
    *reinterpret_cast<u64*>(guest_rsp) = kReturnSentinel;
    GuestState st{};
    st.rip = reinterpret_cast<u64>(mem.CodePtr());
    st.gpr[4] = reinterpret_cast<u64>(guest_rsp);
    st.gpr[6] = reinterpret_cast<u64>(a);
    st.gpr[7] = reinterpret_cast<u64>(b);
    st.gpr[1] = 0; // RCX = 0
    st.rflags = 0x2 | (1ULL<<6); // ZF pre-set; must survive untouched

    Runtime rt;
    rt.Run(st);
    EXPECT_EQ(st.gpr[1], 0ULL) << "RCX stays 0";
    EXPECT_EQ(st.gpr[6], reinterpret_cast<u64>(a)) << "RSI unmoved (no compare)";
    EXPECT_EQ(st.gpr[7], reinterpret_cast<u64>(b)) << "RDI unmoved";
    EXPECT_EQ(st.rflags & (1ULL<<6), (1ULL<<6)) << "flags unchanged on zero-trip";
}

// DF=1 (backward): pointers DECREMENT. Compare a single pair then step
// back by 8. Use count=1 so we isolate the direction behavior.
TEST_F(CpuRuntimeTest, RepeCmpsq_BackwardDirection) {
    u64* a = reinterpret_cast<u64*>(mem.CodePtr() + 0x100);
    u64* b = reinterpret_cast<u64*>(mem.CodePtr() + 0x200);
    a[0]=0x55; b[0]=0x55; // equal
    const u8 program[] = { 0xf3, 0x48, 0xa7, 0xc3 };
    std::memcpy(mem.CodePtr(), program, sizeof(program));
    u8* guest_rsp = mem.StackTop() - 8;
    *reinterpret_cast<u64*>(guest_rsp) = kReturnSentinel;
    GuestState st{};
    st.rip = reinterpret_cast<u64>(mem.CodePtr());
    st.gpr[4] = reinterpret_cast<u64>(guest_rsp);
    st.gpr[6] = reinterpret_cast<u64>(a);
    st.gpr[7] = reinterpret_cast<u64>(b);
    st.gpr[1] = 1;
    st.rflags = 0x2 | (1ULL<<10); // DF=1 (backward)

    Runtime rt;
    rt.Run(st);
    EXPECT_EQ(st.gpr[1], 0ULL) << "one compare done";
    EXPECT_EQ(st.rflags & (1ULL<<6), (1ULL<<6)) << "ZF=1: equal";
    EXPECT_EQ(st.gpr[6], reinterpret_cast<u64>(a) - 8) << "RSI -= 8 (backward)";
    EXPECT_EQ(st.gpr[7], reinterpret_cast<u64>(b) - 8) << "RDI -= 8 (backward)";
}

// After a normal Run() completes, the thread-local state is restored to
// null, so a subsequent fault-context query again reports not-in-
// runtime. Confirms the diagnostic doesn't latch stale state across
// Run() boundaries (which would mis-attribute a later host-code fault
// to guest execution).
TEST_F(CpuRuntimeTest, FaultDiagnostics_ClearedAfterRun) {
    // A trivial guest program: ret.
    const u8 program[] = { 0xc3 };
    std::memcpy(mem.CodePtr(), program, sizeof(program));
    u8* guest_rsp = mem.StackTop() - 8;
    *reinterpret_cast<u64*>(guest_rsp) = kReturnSentinel;
    GuestState st{};
    st.rip = reinterpret_cast<u64>(mem.CodePtr());
    st.gpr[4] = reinterpret_cast<u64>(guest_rsp);

    Runtime rt;
    rt.Run(st);

    // Run() has returned; tl_current_guest_state was restored to null.
    const auto ctx = Core::Runtime::DescribeFaultContext(
        reinterpret_cast<const void*>(st.rip));
    EXPECT_FALSE(ctx.in_runtime) << "runtime TLS cleared after Run() returns";
    EXPECT_EQ(ctx.guest_rip, 0ull);
}

// ============================================================================
// 32-bit address-size override (0x67 prefix) — the effective address must
// be computed modulo 2^32 and zero-extended to 64 bits. If the upper 32
// bits of the base/index guest registers leak into the address, a load
// or store targets a wild ~TiB host address instead of the intended
// <4 GiB one. (This was the cause of an in-JIT-code access violation
// whose fault address was (1 TiB ceiling + ~12 GiB): the low 32 bits
// were the real target, the high bits leaked register garbage.)
//
// These tests plant a buffer below 4 GiB, load the base register with
// GARBAGE in its upper 32 bits, and confirm the access hits the
// truncated low address rather than faulting.
// ============================================================================

// Helper: map a scratch page below 4 GiB so a 32-bit-truncated address
// is a valid host pointer. Returns nullptr if the OS won't oblige (then
// the test self-skips rather than giving a false failure).
static u8* MapLowScratch() {
#if !defined(_WIN32)
    constexpr uintptr_t kLowAddr = 0x20000000ull;
    // Prefer MAP_FIXED_NOREPLACE (Linux 4.17+): it places the page at exactly
    // kLowAddr when that range is free, and fails cleanly otherwise — without
    // silently relocating to a high address the way a bare hint does. This is
    // what lets the addr32 / absolute-disp32 tests actually run on CI hosts
    // whose default mmap base sits above 4 GiB (a plain hint there gets ignored
    // and the page lands high, so the test skips). The runtime never reserves
    // this range (GuestMemory and the gateway both mmap with a NULL hint), so
    // kLowAddr is normally free.
#if defined(MAP_FIXED_NOREPLACE)
    {
        void* p = ::mmap(reinterpret_cast<void*>(kLowAddr), 4096,
                         PROT_READ | PROT_WRITE,
                         MAP_PRIVATE | MAP_ANON | MAP_FIXED_NOREPLACE, -1, 0);
        if (p != MAP_FAILED) {
            if (reinterpret_cast<uintptr_t>(p) == kLowAddr) {
                return static_cast<u8*>(p);
            }
            // Kernel ignored the flag (older kernel mis-reporting support) and
            // placed it elsewhere — discard and fall through to the hint path.
            ::munmap(p, 4096);
        }
    }
#endif
    // Fallback for kernels without MAP_FIXED_NOREPLACE: hinted mmap, accepted
    // only if it actually landed sub-4GiB at the required address.
    void* p = ::mmap(reinterpret_cast<void*>(kLowAddr), 4096,
                     PROT_READ | PROT_WRITE, MAP_PRIVATE | MAP_ANON, -1, 0);
    if (p == MAP_FAILED) return nullptr;
    if (reinterpret_cast<uintptr_t>(p) != kLowAddr) { ::munmap(p, 4096); return nullptr; }
    return static_cast<u8*>(p);
#else
    return nullptr;
#endif
}

// mov [ebx], eax with RBX = garbage_upper | low_addr. The store must
// land at low_addr (upper 32 bits of RBX ignored), not fault.
TEST_F(CpuRuntimeTest, Addr32_BaseUpperBitsIgnored) {
    u8* low = MapLowScratch();
    if (low == nullptr) GTEST_SKIP() << "no sub-4GiB mapping available";
    const u32 target_off = 0x40; // write 0x40 bytes into the low page
    u8* target = low + target_off;
    *reinterpret_cast<u32*>(target) = 0; // clear

    // mov [ebx], eax  — 67 89 03
    const u8 program[] = { 0x67, 0x89, 0x03, 0xc3 };
    std::memcpy(mem.CodePtr(), program, sizeof(program));
    u8* guest_rsp = mem.StackTop() - 8;
    *reinterpret_cast<u64*>(guest_rsp) = kReturnSentinel;
    GuestState st{};
    st.rip = reinterpret_cast<u64>(mem.CodePtr());
    st.gpr[4] = reinterpret_cast<u64>(guest_rsp);
    // RBX low32 = target address; upper32 = garbage that must be masked.
    st.gpr[3] = (0xDEADBEEFull << 32) | static_cast<u32>(reinterpret_cast<uintptr_t>(target));
    st.gpr[0] = 0x12345678; // eax value to store

    Runtime rt;
    rt.Run(st);
    EXPECT_EQ(*reinterpret_cast<u32*>(target), 0x12345678u)
        << "store landed at the 32-bit-truncated address";
#if !defined(_WIN32)
    ::munmap(low, 4096);
#endif
}

// mov [ebx + ecx*4], eax with BOTH base and index carrying garbage
// upper bits. The entire base+index*scale sum must wrap at 32 bits.
TEST_F(CpuRuntimeTest, Addr32_BasePlusIndexTruncated) {
    u8* low = MapLowScratch();
    if (low == nullptr) GTEST_SKIP() << "no sub-4GiB mapping available";
    // Effective low address = base_low + index_low*4.
    const u32 base_low = static_cast<u32>(reinterpret_cast<uintptr_t>(low));
    const u32 idx = 5;
    u8* target = low + idx * 4;
    *reinterpret_cast<u32*>(target) = 0;

    // mov [ebx+ecx*4], eax  — 67 89 04 8b
    const u8 program[] = { 0x67, 0x89, 0x04, 0x8b, 0xc3 };
    std::memcpy(mem.CodePtr(), program, sizeof(program));
    u8* guest_rsp = mem.StackTop() - 8;
    *reinterpret_cast<u64*>(guest_rsp) = kReturnSentinel;
    GuestState st{};
    st.rip = reinterpret_cast<u64>(mem.CodePtr());
    st.gpr[4] = reinterpret_cast<u64>(guest_rsp);
    st.gpr[3] = (0xCAFEF00Dull << 32) | base_low; // RBX: garbage upper
    st.gpr[1] = (0x99999999ull << 32) | idx;        // RCX: garbage upper
    st.gpr[0] = 0xABCDEF01;

    Runtime rt;
    rt.Run(st);
    EXPECT_EQ(*reinterpret_cast<u32*>(target), 0xABCDEF01u)
        << "base+index*scale truncated to 32 bits before access";
#if !defined(_WIN32)
    ::munmap(low, 4096);
#endif
}

// Regression guard: a NORMAL 64-bit-mode access (no 0x67 prefix) must
// NOT be truncated — the full 64-bit base is used as-is. We can verify
// this without a low mapping by storing through the regular scratch
// region (which lives at its natural 64-bit host address).
TEST_F(CpuRuntimeTest, Addr64_NotTruncated) {
    u8* target = mem.CodePtr() + 0x100;
    *reinterpret_cast<u32*>(target) = 0;
    // mov [rbx], eax  — 89 03 (no 0x67)
    const u8 program[] = { 0x89, 0x03, 0xc3 };
    std::memcpy(mem.CodePtr(), program, sizeof(program));
    u8* guest_rsp = mem.StackTop() - 8;
    *reinterpret_cast<u64*>(guest_rsp) = kReturnSentinel;
    GuestState st{};
    st.rip = reinterpret_cast<u64>(mem.CodePtr());
    st.gpr[4] = reinterpret_cast<u64>(guest_rsp);
    st.gpr[3] = reinterpret_cast<u64>(target); // full 64-bit pointer
    st.gpr[0] = 0x5A5A5A5A;

    Runtime rt;
    rt.Run(st);
    EXPECT_EQ(*reinterpret_cast<u32*>(target), 0x5A5A5A5Au)
        << "64-bit address used in full (no truncation)";
}

// ============================================================================
// CUSA07010 (Sonic Mania / RSDK) regression tests.
//
// These exercise the exact instruction *patterns* — byte-for-byte — pulled
// from eboot.bin around the RSDK 2D-blit at guest 0x800274bb0, the function
// that produced a long-standing deterministic access violation. The static
// disassembly of that region proved the JIT translation was byte-exact and
// the fault was a faithful consequence of a 32-bit zero-extension chain
// feeding a 64-bit pointer index. These tests lock that behavior in so the
// composition can't silently regress.
//
// Each program below uses the genuine eboot encoding (verified by objdump).
// GPR index map: RAX=0 RCX=1 RDX=2 RBX=3 RSP=4 RBP=5 RSI=6 RDI=7
//                R8=8 R9=9 R10=10 R11=11 R12=12 R13=13 R14=14 R15=15.
// ============================================================================

// --- T1: the R8 producer chain ---------------------------------------------
// mov r8d,0x1 ; sub r8d,r10d ; add r8d,ecx   (@0x800274bcc, eboot bytes)
//
// 32-bit ALU into r8d MUST zero-extend bits 63:32. With the real crash inputs
// (r10=432, ecx=31) the result is -400 = 0x00000000fffffe70 — exactly what the
// emulator computed and what real hardware computes. The point of this test is
// to pin the zero-extension: the upper half must be clear, NOT sign-extended.
TEST_F(CpuRuntimeTest, EbootProducerChain_R8d_ZeroExtendsNotSignExtends) {
    const u8 program[] = {
        0x41, 0xb8, 0x01, 0x00, 0x00, 0x00, // mov r8d, 0x1
        0x45, 0x29, 0xd0,                   // sub r8d, r10d
        0x41, 0x01, 0xc8,                   // add r8d, ecx
        0xc3,                               // ret
    };
    std::memcpy(mem.CodePtr(), program, sizeof(program));
    u8* guest_rsp = mem.StackTop() - 8;
    *reinterpret_cast<u64*>(guest_rsp) = kReturnSentinel;

    GuestState st{};
    st.rip = reinterpret_cast<u64>(mem.CodePtr());
    st.gpr[4] = reinterpret_cast<u64>(guest_rsp);
    // Real crash inputs. Seed garbage in the upper 32 bits of r8/r10/rcx to
    // prove the 32-bit ops both consume and produce clean 32-bit values.
    st.gpr[8] = 0xDEAD'BEEF'0000'0000ULL;  // r8  (upper garbage; mov r8d must wipe it)
    st.gpr[10] = 0xCAFE'0000'0000'01b0ULL; // r10 (low = 432)
    st.gpr[1] = 0xF00D'0000'0000'001fULL;  // rcx (low = 31)

    Runtime rt;
    rt.Run(st);

    EXPECT_EQ(st.gpr[8], 0xfffffe70ULL)
        << "r8 = 1 - 432 + 31 = -400, zero-extended to 64 bits";
    EXPECT_EQ(st.gpr[8] >> 32, 0u)
        << "32-bit ALU writes MUST clear bits 63:32 (zero-extend, never sign-extend)";
}

// Same chain, "well-behaved" inputs where the engine intends a small delta:
// r10=32, ecx=31 -> r8 = 1 - 32 + 31 = 0. Guards the arithmetic itself.
TEST_F(CpuRuntimeTest, EbootProducerChain_SmallInputs_ProduceZero) {
    const u8 program[] = {
        0x41, 0xb8, 0x01, 0x00, 0x00, 0x00, // mov r8d, 0x1
        0x45, 0x29, 0xd0,                   // sub r8d, r10d
        0x41, 0x01, 0xc8,                   // add r8d, ecx
        0xc3,
    };
    std::memcpy(mem.CodePtr(), program, sizeof(program));
    u8* guest_rsp = mem.StackTop() - 8;
    *reinterpret_cast<u64*>(guest_rsp) = kReturnSentinel;

    GuestState st{};
    st.rip = reinterpret_cast<u64>(mem.CodePtr());
    st.gpr[4] = reinterpret_cast<u64>(guest_rsp);
    st.gpr[10] = 32;
    st.gpr[1] = 31;

    Runtime rt;
    rt.Run(st);
    EXPECT_EQ(st.gpr[8], 0u) << "1 - 32 + 31 = 0";
    EXPECT_EQ(st.gpr[8] >> 32, 0u);
}

// --- T2: the crash LEA, scale-2 64-bit index -------------------------------
// lea rax,[rax+r8*2]   (@0x800274cb8, eboot bytes 4a 8d 04 40)
//
// The index register is used at full 64-bit width (REX.W, REX.X). This test
// verifies the lifter feeds the *entire* r8 into the address math — the exact
// behavior that, given a zero-extended r8, faithfully reproduces hardware.
// Two cases: zero-extended r8 (the real run) and sign-extended r8 (the value
// the blit geometry actually needs). Both must compute the architectural sum.
TEST_F(CpuRuntimeTest, EbootCrashLea_RaxPlusR8x2_Uses64BitIndex) {
    const u8 program[] = {
        0x4a, 0x8d, 0x04, 0x40, // lea rax,[rax+r8*2]
        0xc3,
    };
    std::memcpy(mem.CodePtr(), program, sizeof(program));
    u8* guest_rsp = mem.StackTop() - 8;
    *reinterpret_cast<u64*>(guest_rsp) = kReturnSentinel;

    // Case A: zero-extended r8 = 0x00000000fffffe70 (the actual crash value).
    {
        GuestState st{};
        st.rip = reinterpret_cast<u64>(mem.CodePtr());
        st.gpr[4] = reinterpret_cast<u64>(guest_rsp);
        st.gpr[0] = 0x214c40000ULL; // rax = dst base (in dmem pool)
        st.gpr[8] = 0x00000000fffffe70ULL;
        Runtime rt;
        rt.Run(st);
        EXPECT_EQ(st.gpr[0], 0x414c3fce0ULL)
            << "rax + (0xfffffe70 << 1): full 64-bit index, no truncation";
    }
    // Case B: sign-extended r8 = 0xfffffffffffffe70 (-400). The same emitter,
    // a different input, yields the in-pool address — confirming the lea is a
    // faithful pass-through and the bug was never in this instruction.
    {
        *reinterpret_cast<u64*>(guest_rsp) = kReturnSentinel;
        GuestState st{};
        st.rip = reinterpret_cast<u64>(mem.CodePtr());
        st.gpr[4] = reinterpret_cast<u64>(guest_rsp);
        st.gpr[0] = 0x214c40000ULL;
        st.gpr[8] = 0xfffffffffffffe70ULL; // -400
        Runtime rt;
        rt.Run(st);
        EXPECT_EQ(st.gpr[0], 0x214c3fce0ULL)
            << "rax + (-400 << 1) = rax - 800: stays in pool";
    }
}

// --- T3: the row-advance LEA, base r11 + r9*2 ------------------------------
// lea rax,[r11+r9*2]   (@0x800274cae, eboot bytes 4b 8d 04 4b)
// Both base and index are extended (REX.B + REX.X). r11=base, r9=index, scale 2.
TEST_F(CpuRuntimeTest, EbootRowLea_R11PlusR9x2_ExtendedBaseAndIndex) {
    const u8 program[] = {
        0x4b, 0x8d, 0x04, 0x4b, // lea rax,[r11+r9*2]
        0xc3,
    };
    std::memcpy(mem.CodePtr(), program, sizeof(program));
    u8* guest_rsp = mem.StackTop() - 8;
    *reinterpret_cast<u64*>(guest_rsp) = kReturnSentinel;

    GuestState st{};
    st.rip = reinterpret_cast<u64>(mem.CodePtr());
    st.gpr[4] = reinterpret_cast<u64>(guest_rsp);
    st.gpr[11] = 0x214c40020ULL; // r11 = rax_start + 0x20
    st.gpr[9] = 0x1a0ULL;        // r9 = 416 (stride)

    Runtime rt;
    rt.Run(st);
    EXPECT_EQ(st.gpr[0], 0x214c40360ULL)
        << "r11 + r9*2 = 0x214c40020 + 0x340";
}

// --- T4: sar / lea-r32 / shl interplay -------------------------------------
// sar r10d,4 ; lea r9d,[r10-1] ; shl r9,4   (@0x800274bf1, eboot bytes)
//
// This is the stride derivation. It mixes a 32-bit arithmetic shift, a 32-bit
// LEA (which zero-extends its 64-bit result), and a 64-bit shift. With the
// real r10=432 it reproduces r9 = 416 = 0x1a0 exactly.
TEST_F(CpuRuntimeTest, EbootStrideDerivation_Sar_LeaR32_Shl) {
    const u8 program[] = {
        0x41, 0xc1, 0xfa, 0x04, // sar r10d, 4
        0x45, 0x8d, 0x4a, 0xff, // lea r9d, [r10-1]
        0x49, 0xc1, 0xe1, 0x04, // shl r9, 4
        0xc3,
    };
    std::memcpy(mem.CodePtr(), program, sizeof(program));
    u8* guest_rsp = mem.StackTop() - 8;
    *reinterpret_cast<u64*>(guest_rsp) = kReturnSentinel;

    GuestState st{};
    st.rip = reinterpret_cast<u64>(mem.CodePtr());
    st.gpr[4] = reinterpret_cast<u64>(guest_rsp);
    st.gpr[10] = 0x1b0ULL; // r10 = 432

    Runtime rt;
    rt.Run(st);
    EXPECT_EQ(st.gpr[10], 27u) << "sar 432 >> 4 = 27";
    EXPECT_EQ(st.gpr[9], 0x1a0u) << "((27-1) << 4) = 416 = 0x1a0 (matches eboot r9)";
    EXPECT_EQ(st.gpr[9] >> 32, 0u) << "lea r32 zero-extends before the 64-bit shl";
}

// SAR sign-extension guard: arithmetic shift of a negative 32-bit value must
// fill with sign bits within the 32-bit lane (and zero-extend the lane to 64).
TEST_F(CpuRuntimeTest, EbootStride_SarNegative_SignFillsLane) {
    const u8 program[] = {
        0x41, 0xc1, 0xfa, 0x04, // sar r10d, 4
        0xc3,
    };
    std::memcpy(mem.CodePtr(), program, sizeof(program));
    u8* guest_rsp = mem.StackTop() - 8;
    *reinterpret_cast<u64*>(guest_rsp) = kReturnSentinel;

    GuestState st{};
    st.rip = reinterpret_cast<u64>(mem.CodePtr());
    st.gpr[4] = reinterpret_cast<u64>(guest_rsp);
    st.gpr[10] = 0xFFFF'FFFF'FFFF'FFF0ULL; // r10d = -16 (upper garbage too)

    Runtime rt;
    rt.Run(st);
    EXPECT_EQ(st.gpr[10], 0xffffffffULL)
        << "sar(-16, 4) = -1 within 32-bit lane = 0xffffffff, zero-extended to 64";
}

// --- T5: mov eax,eax then shl rax,8 ----------------------------------------
// mov eax,eax ; shl rax,8   (@0x800274bd2, eboot bytes 89 c0 / 48 c1 e0 08)
//
// `mov eax,eax` is the canonical 32-bit zero-extend idiom: it DISCARDS bits
// 63:32. This is how the dst pointer is rebuilt from a packed 32-bit handle.
// Upper garbage in rax must vanish before the shift.
TEST_F(CpuRuntimeTest, EbootDstPointer_MovEaxEax_DiscardsUpperThenShl) {
    const u8 program[] = {
        0x89, 0xc0,             // mov eax, eax
        0x48, 0xc1, 0xe0, 0x08, // shl rax, 8
        0xc3,
    };
    std::memcpy(mem.CodePtr(), program, sizeof(program));
    u8* guest_rsp = mem.StackTop() - 8;
    *reinterpret_cast<u64*>(guest_rsp) = kReturnSentinel;

    GuestState st{};
    st.rip = reinterpret_cast<u64>(mem.CodePtr());
    st.gpr[4] = reinterpret_cast<u64>(guest_rsp);
    st.gpr[0] = 0xDEAD'BEEF'0214'C400ULL; // packed handle with upper garbage

    Runtime rt;
    rt.Run(st);
    EXPECT_EQ(st.gpr[0], 0x214c40000ULL)
        << "mov eax,eax drops upper 32, then <<8 reconstructs the dmem pointer";
}

// --- T6: memory-source BEXTR -----------------------------------------------
// mov ecx,0xe0d ; bextr ecx,[rbx+r15+0x10],ecx   (@0x800274bc7, eboot bytes)
//
// The bitfield ctrl 0xe0d means start=13, len=14. This is the producer of the
// engine's per-surface field. The source is a *memory* operand with base+index
// +disp, exercising EmitEffectiveAddress feeding the host BEXTR. We seed the
// scratch dword so the extracted field equals 415 (a plausible stride field).
TEST_F(CpuRuntimeTest, EbootBextrMem_BaseIndexDisp_ExtractsField) {
    // Scratch dword lives at CodePtr()+0x100; addressed as [rbx + r15 + 0x10],
    // so put (rbx + r15) = scratch - 0x10.
    u32* scratch = reinterpret_cast<u32*>(mem.CodePtr() + 0x100);
    *scratch = (415u << 13); // bits[13:26] == 415

    const u8 program[] = {
        0xb9, 0x0d, 0x0e, 0x00, 0x00,             // mov ecx, 0xe0d (start=13,len=14)
        0xc4, 0xa2, 0x70, 0xf7, 0x4c, 0x3b, 0x10, // bextr ecx,[rbx+r15*1+0x10],ecx
        0xc3,
    };
    std::memcpy(mem.CodePtr(), program, sizeof(program));
    u8* guest_rsp = mem.StackTop() - 8;
    *reinterpret_cast<u64*>(guest_rsp) = kReturnSentinel;

    const u64 base_plus_index = reinterpret_cast<u64>(scratch) - 0x10;

    GuestState st{};
    st.rip = reinterpret_cast<u64>(mem.CodePtr());
    st.gpr[4] = reinterpret_cast<u64>(guest_rsp);
    st.gpr[3] = base_plus_index; // rbx
    st.gpr[15] = 0;              // r15 (so rbx+r15+0x10 = scratch)

    Runtime rt;
    rt.Run(st);
    EXPECT_EQ(st.gpr[1], 415u)
        << "BEXTR(mem, start=13, len=14) over (415<<13) = 415";
    EXPECT_EQ(st.gpr[1] >> 32, 0u) << "32-bit BEXTR zero-extends";
}

// Split base/index variant: same address via rbx=scratch-0x110, r15=0x100.
// Guards that EmitEffectiveAddress sums base + index + disp (not just base).
TEST_F(CpuRuntimeTest, EbootBextrMem_IndexContributes) {
    u32* scratch = reinterpret_cast<u32*>(mem.CodePtr() + 0x200);
    *scratch = (123u << 13);

    const u8 program[] = {
        0xb9, 0x0d, 0x0e, 0x00, 0x00,
        0xc4, 0xa2, 0x70, 0xf7, 0x4c, 0x3b, 0x10, // bextr ecx,[rbx+r15+0x10],ecx
        0xc3,
    };
    std::memcpy(mem.CodePtr(), program, sizeof(program));
    u8* guest_rsp = mem.StackTop() - 8;
    *reinterpret_cast<u64*>(guest_rsp) = kReturnSentinel;

    GuestState st{};
    st.rip = reinterpret_cast<u64>(mem.CodePtr());
    st.gpr[4] = reinterpret_cast<u64>(guest_rsp);
    // rbx + r15 + 0x10 must equal scratch. Split it so the index is non-zero.
    st.gpr[3] = reinterpret_cast<u64>(scratch) - 0x10 - 0x100; // rbx
    st.gpr[15] = 0x100;                                         // r15 (index, scale 1)

    Runtime rt;
    rt.Run(st);
    EXPECT_EQ(st.gpr[1], 123u)
        << "address must be base + index*1 + disp; a missing index term would misread";
}

// --- T7: the unrolled u16 copy primitive -----------------------------------
// movzx edi,word[rsi] ; mov word[rax],di   (@0x800274c20, eboot bytes)
// The core of the blit's inner loop: 16-bit load (zero-extended into edi) then
// 16-bit store. Verifies the half-word store writes exactly 2 bytes and the
// load zero-extends.
TEST_F(CpuRuntimeTest, EbootBlitCopy_Movzx16_Store16) {
    u16* src = reinterpret_cast<u16*>(mem.CodePtr() + 0x100);
    u16* dst = reinterpret_cast<u16*>(mem.CodePtr() + 0x200);
    *src = 0xBEEF;
    *dst = 0x1111;
    *(dst + 1) = 0x2222; // sentinel after dst; the 16-bit store must NOT touch it

    const u8 program[] = {
        0x0f, 0xb7, 0x3e, // movzx edi, word[rsi]
        0x66, 0x89, 0x38, // mov word[rax], di
        0xc3,
    };
    std::memcpy(mem.CodePtr(), program, sizeof(program));
    u8* guest_rsp = mem.StackTop() - 8;
    *reinterpret_cast<u64*>(guest_rsp) = kReturnSentinel;

    GuestState st{};
    st.rip = reinterpret_cast<u64>(mem.CodePtr());
    st.gpr[4] = reinterpret_cast<u64>(guest_rsp);
    st.gpr[6] = reinterpret_cast<u64>(src); // rsi
    st.gpr[0] = reinterpret_cast<u64>(dst); // rax
    st.gpr[7] = 0xFFFF'FFFF'FFFF'0000ULL;   // rdi: prove movzx zero-extends low 16

    Runtime rt;
    rt.Run(st);
    EXPECT_EQ(st.gpr[7], 0xBEEFu) << "movzx edi,word: zero-extend the 16-bit load";
    EXPECT_EQ(*dst, 0xBEEFu) << "16-bit store writes the copied halfword";
    EXPECT_EQ(*(dst + 1), 0x2222u) << "16-bit store must NOT clobber the next halfword";
}

// --- T8: the post-loop MOVSXD ----------------------------------------------
// movsxd rax,dword[r14+0x2c]   (@0x800274cc8, eboot bytes 49 63 46 2c)
//
// This is the sign-extending load the loop trip-count uses. It is the CORRECT
// way to widen a signed 32-bit value to 64 bits — the contrast that explains
// why the 32-bit producer chain (T1) legitimately does NOT sign-extend. A
// negative source must fill the upper 32 bits with ones.
TEST_F(CpuRuntimeTest, EbootMovsxdMem_SignExtendsNegative) {
    s32* slot = reinterpret_cast<s32*>(mem.CodePtr() + 0x100 + 0x2c);
    *slot = static_cast<s32>(0xFFFFFE70); // -400

    const u8 program[] = {
        0x49, 0x63, 0x46, 0x2c, // movsxd rax, dword[r14+0x2c]
        0xc3,
    };
    std::memcpy(mem.CodePtr(), program, sizeof(program));
    u8* guest_rsp = mem.StackTop() - 8;
    *reinterpret_cast<u64*>(guest_rsp) = kReturnSentinel;

    GuestState st{};
    st.rip = reinterpret_cast<u64>(mem.CodePtr());
    st.gpr[4] = reinterpret_cast<u64>(guest_rsp);
    st.gpr[14] = reinterpret_cast<u64>(mem.CodePtr() + 0x100); // r14

    Runtime rt;
    rt.Run(st);
    EXPECT_EQ(st.gpr[0], 0xfffffffffffffe70ULL)
        << "movsxd sign-extends -400 to the full 64-bit register";
}

// Positive source: movsxd must zero the upper 32 (sign bit clear).
TEST_F(CpuRuntimeTest, EbootMovsxdMem_PositiveZeroExtends) {
    s32* slot = reinterpret_cast<s32*>(mem.CodePtr() + 0x100 + 0x2c);
    *slot = 0x000000f0; // 240 (the outer row count)

    const u8 program[] = {
        0x49, 0x63, 0x46, 0x2c, // movsxd rax, dword[r14+0x2c]
        0xc3,
    };
    std::memcpy(mem.CodePtr(), program, sizeof(program));
    u8* guest_rsp = mem.StackTop() - 8;
    *reinterpret_cast<u64*>(guest_rsp) = kReturnSentinel;

    GuestState st{};
    st.rip = reinterpret_cast<u64>(mem.CodePtr());
    st.gpr[4] = reinterpret_cast<u64>(guest_rsp);
    st.gpr[0] = 0xFFFF'FFFF'0000'0000ULL; // prove the upper half gets cleared
    st.gpr[14] = reinterpret_cast<u64>(mem.CodePtr() + 0x100);

    Runtime rt;
    rt.Run(st);
    EXPECT_EQ(st.gpr[0], 0xf0ULL)
        << "movsxd of a positive value clears bits 63:32";
}

// --- T9: register-source BEXTR ---------------------------------------------
// bextr r8d,edx,eax   (@0x8003055ef, eboot bytes c4 62 78 f7 c2)
//
// The dst-pointer helper (sub_8003055a0) decomposes a packed dword with a
// sequence of register-source BEXTRs writing r8d/r9d/edi/esi. ctrl=eax here.
TEST_F(CpuRuntimeTest, EbootBextrReg_R8d_FromEdxByEax) {
    const u8 program[] = {
        0xc4, 0x62, 0x78, 0xf7, 0xc2, // bextr r8d, edx, eax
        0xc3,
    };
    std::memcpy(mem.CodePtr(), program, sizeof(program));
    u8* guest_rsp = mem.StackTop() - 8;
    *reinterpret_cast<u64*>(guest_rsp) = kReturnSentinel;

    GuestState st{};
    st.rip = reinterpret_cast<u64>(mem.CodePtr());
    st.gpr[4] = reinterpret_cast<u64>(guest_rsp);
    st.gpr[2] = 0x1234'5678ULL;            // edx = source
    st.gpr[0] = 0x0808ULL;                 // eax = ctrl: start=8, len=8
    st.gpr[8] = 0xDEAD'BEEF'DEAD'BEEFULL;  // r8 upper garbage

    Runtime rt;
    rt.Run(st);
    EXPECT_EQ(st.gpr[8], 0x56ULL)
        << "BEXTR(0x12345678, start=8, len=8) = 0x56";
    EXPECT_EQ(st.gpr[8] >> 32, 0u)
        << "32-bit BEXTR result zero-extends bits 63:32";
}

// --- T10: integration — the full producer + lea, end to end ----------------
// This stitches the producer chain and the crash lea into one block, exactly
// as control flow reaches them, and asserts the *whole* computation matches
// the observed crash (proving the composition is faithful, not just each op).
// With crash inputs the dst escapes the pool; with the value the engine would
// supply on a correct run (small r8) it stays in range. We assert the former
// (the faithful, reproducible result) so any future drift is caught.
TEST_F(CpuRuntimeTest, EbootProducerPlusLea_Integration_MatchesObservedCrash) {
    const u8 program[] = {
        0x41, 0xb8, 0x01, 0x00, 0x00, 0x00, // mov r8d, 0x1
        0x45, 0x29, 0xd0,                   // sub r8d, r10d
        0x41, 0x01, 0xc8,                   // add r8d, ecx
        0x4a, 0x8d, 0x04, 0x40,             // lea rax,[rax+r8*2]
        0xc3,
    };
    std::memcpy(mem.CodePtr(), program, sizeof(program));
    u8* guest_rsp = mem.StackTop() - 8;
    *reinterpret_cast<u64*>(guest_rsp) = kReturnSentinel;

    GuestState st{};
    st.rip = reinterpret_cast<u64>(mem.CodePtr());
    st.gpr[4] = reinterpret_cast<u64>(guest_rsp);
    st.gpr[0] = 0x214c40000ULL; // rax = dst base
    st.gpr[10] = 432;           // r10
    st.gpr[1] = 31;             // rcx (ecx)

    Runtime rt;
    rt.Run(st);
    EXPECT_EQ(st.gpr[8], 0xfffffe70ULL)
        << "producer yields zero-extended -400";
    EXPECT_EQ(st.gpr[0], 0x414c3fce0ULL)
        << "lea consumes the full 64-bit (zero-extended) r8 -> faithful OOB result";
}

// ============================================================================
// String-instruction coverage (STOS / MOVS / LODS / SCAS).
//
// Motivated by CUSA02394 "WE ARE DOOMED", which fatally exited the JIT at the
// first `rep stosb` inside libc memset (guest 0x8075ac00f, exit_reason=2
// UnsupportedInstruction). These are the memset/memcpy/strlen primitives;
// they were a known coverage gap. Tests below pin both the REP-counted forms
// and the single forms, the direction flag (DF) controlling advance sign, and
// the flag semantics of SCAS (only SCAS sets flags; STOS/MOVS/LODS do not).
//
// GPR index map: RAX=0 RCX=1 RDX=2 RBX=3 RSP=4 RBP=5 RSI=6 RDI=7.
// rflags DF is bit 10 (0x400).
// ============================================================================

// rep stosb — the exact memset primitive that crashed CUSA02394.
// Fill 5 bytes with 0xAB; RCX must reach 0 and RDI must advance by 5.
TEST_F(CpuRuntimeTest, RepStosb_FillsBufferAndConsumesCount) {
    u8* buf = mem.CodePtr() + 0x200;
    std::memset(buf, 0x00, 16);
    buf[5] = 0x77; // sentinel: must NOT be overwritten (only 5 bytes filled)

    const u8 program[] = {0xf3, 0xaa, 0xc3}; // rep stosb ; ret
    std::memcpy(mem.CodePtr(), program, sizeof(program));
    u8* guest_rsp = mem.StackTop() - 8;
    *reinterpret_cast<u64*>(guest_rsp) = kReturnSentinel;

    GuestState st{};
    st.rip = reinterpret_cast<u64>(mem.CodePtr());
    st.gpr[4] = reinterpret_cast<u64>(guest_rsp);
    st.gpr[0] = 0xAB;                        // AL = fill value
    st.gpr[7] = reinterpret_cast<u64>(buf);  // RDI = dest
    st.gpr[1] = 5;                           // RCX = count

    Runtime rt;
    rt.Run(st);

    for (int i = 0; i < 5; ++i)
        EXPECT_EQ(buf[i], 0xAB) << "byte " << i;
    EXPECT_EQ(buf[5], 0x77) << "fill must stop exactly at RCX bytes";
    EXPECT_EQ(st.gpr[1], 0u) << "RCX exhausted";
    EXPECT_EQ(st.gpr[7], reinterpret_cast<u64>(buf) + 5) << "RDI advanced by 5 (DF=0)";
}

// rep stosb with RCX=0 — must write nothing and not move RDI.
TEST_F(CpuRuntimeTest, RepStosb_ZeroCount_NoOp) {
    u8* buf = mem.CodePtr() + 0x200;
    buf[0] = 0x55;

    const u8 program[] = {0xf3, 0xaa, 0xc3};
    std::memcpy(mem.CodePtr(), program, sizeof(program));
    u8* guest_rsp = mem.StackTop() - 8;
    *reinterpret_cast<u64*>(guest_rsp) = kReturnSentinel;

    GuestState st{};
    st.rip = reinterpret_cast<u64>(mem.CodePtr());
    st.gpr[4] = reinterpret_cast<u64>(guest_rsp);
    st.gpr[0] = 0xAB;
    st.gpr[7] = reinterpret_cast<u64>(buf);
    st.gpr[1] = 0; // RCX = 0

    Runtime rt;
    rt.Run(st);
    EXPECT_EQ(buf[0], 0x55) << "zero-count REP must write nothing";
    EXPECT_EQ(st.gpr[7], reinterpret_cast<u64>(buf)) << "RDI unchanged";
}

// rep stosq — qword fill (memset of 8-byte units).
TEST_F(CpuRuntimeTest, RepStosq_FillsQwords) {
    u64* buf = reinterpret_cast<u64*>(mem.CodePtr() + 0x200);
    buf[0] = buf[1] = buf[2] = 0;
    buf[3] = 0x1234'5678'9abc'def0ULL; // sentinel

    const u8 program[] = {0xf3, 0x48, 0xab, 0xc3}; // rep stosq ; ret
    std::memcpy(mem.CodePtr(), program, sizeof(program));
    u8* guest_rsp = mem.StackTop() - 8;
    *reinterpret_cast<u64*>(guest_rsp) = kReturnSentinel;

    GuestState st{};
    st.rip = reinterpret_cast<u64>(mem.CodePtr());
    st.gpr[4] = reinterpret_cast<u64>(guest_rsp);
    st.gpr[0] = 0xDEAD'BEEF'CAFE'BABEULL;     // RAX = fill qword
    st.gpr[7] = reinterpret_cast<u64>(buf);   // RDI
    st.gpr[1] = 3;                            // RCX = 3 qwords

    Runtime rt;
    rt.Run(st);
    EXPECT_EQ(buf[0], 0xDEAD'BEEF'CAFE'BABEULL);
    EXPECT_EQ(buf[1], 0xDEAD'BEEF'CAFE'BABEULL);
    EXPECT_EQ(buf[2], 0xDEAD'BEEF'CAFE'BABEULL);
    EXPECT_EQ(buf[3], 0x1234'5678'9abc'def0ULL) << "stop after 3 qwords";
    EXPECT_EQ(st.gpr[7], reinterpret_cast<u64>(buf) + 24) << "RDI += 3*8";
}

// Single stosb (no REP) — store one byte, advance RDI by 1, leave RCX alone.
TEST_F(CpuRuntimeTest, Stosb_Single_StoresOneByte) {
    u8* buf = mem.CodePtr() + 0x200;
    buf[0] = 0; buf[1] = 0x99;

    const u8 program[] = {0xaa, 0xc3}; // stosb ; ret
    std::memcpy(mem.CodePtr(), program, sizeof(program));
    u8* guest_rsp = mem.StackTop() - 8;
    *reinterpret_cast<u64*>(guest_rsp) = kReturnSentinel;

    GuestState st{};
    st.rip = reinterpret_cast<u64>(mem.CodePtr());
    st.gpr[4] = reinterpret_cast<u64>(guest_rsp);
    st.gpr[0] = 0x3C;
    st.gpr[7] = reinterpret_cast<u64>(buf);
    st.gpr[1] = 999; // RCX must be untouched by non-REP form

    Runtime rt;
    rt.Run(st);
    EXPECT_EQ(buf[0], 0x3C);
    EXPECT_EQ(buf[1], 0x99) << "single store touches one byte only";
    EXPECT_EQ(st.gpr[7], reinterpret_cast<u64>(buf) + 1);
    EXPECT_EQ(st.gpr[1], 999u) << "non-REP must not touch RCX";
}

// rep stosb with DF=1 — fill must go BACKWARD (RDI decrements).
TEST_F(CpuRuntimeTest, RepStosb_DfSet_FillsBackward) {
    u8* buf = mem.CodePtr() + 0x200;
    std::memset(buf, 0, 16);
    u8* start = buf + 5; // fill 5..3 going down

    const u8 program[] = {0xf3, 0xaa, 0xc3};
    std::memcpy(mem.CodePtr(), program, sizeof(program));
    u8* guest_rsp = mem.StackTop() - 8;
    *reinterpret_cast<u64*>(guest_rsp) = kReturnSentinel;

    GuestState st{};
    st.rip = reinterpret_cast<u64>(mem.CodePtr());
    st.gpr[4] = reinterpret_cast<u64>(guest_rsp);
    st.gpr[0] = 0xEE;
    st.gpr[7] = reinterpret_cast<u64>(start);
    st.gpr[1] = 3;
    st.rflags |= 0x400; // DF = 1

    Runtime rt;
    rt.Run(st);
    EXPECT_EQ(buf[5], 0xEE);
    EXPECT_EQ(buf[4], 0xEE);
    EXPECT_EQ(buf[3], 0xEE);
    EXPECT_EQ(buf[2], 0x00) << "should not write below the 3 filled bytes";
    EXPECT_EQ(buf[6], 0x00) << "should not write above start";
    EXPECT_EQ(st.gpr[7], reinterpret_cast<u64>(start) - 3) << "RDI decremented (DF=1)";
    EXPECT_EQ(st.rflags & 0x400u, 0x400u) << "STOS must preserve DF";
}

// rep movsb — memcpy primitive. Copy 6 bytes src->dst, advance both, RCX=0.
TEST_F(CpuRuntimeTest, RepMovsb_CopiesBuffer) {
    u8* src = mem.CodePtr() + 0x200;
    u8* dst = mem.CodePtr() + 0x300;
    const u8 pattern[6] = {0x10, 0x20, 0x30, 0x40, 0x50, 0x60};
    std::memcpy(src, pattern, 6);
    std::memset(dst, 0, 8);
    dst[6] = 0xC3; // sentinel after copy region

    const u8 program[] = {0xf3, 0xa4, 0xc3}; // rep movsb ; ret
    std::memcpy(mem.CodePtr(), program, sizeof(program));
    u8* guest_rsp = mem.StackTop() - 8;
    *reinterpret_cast<u64*>(guest_rsp) = kReturnSentinel;

    GuestState st{};
    st.rip = reinterpret_cast<u64>(mem.CodePtr());
    st.gpr[4] = reinterpret_cast<u64>(guest_rsp);
    st.gpr[6] = reinterpret_cast<u64>(src); // RSI
    st.gpr[7] = reinterpret_cast<u64>(dst); // RDI
    st.gpr[1] = 6;                          // RCX

    Runtime rt;
    rt.Run(st);
    for (int i = 0; i < 6; ++i)
        EXPECT_EQ(dst[i], pattern[i]) << "copied byte " << i;
    EXPECT_EQ(dst[6], 0xC3) << "copy stops at RCX bytes";
    EXPECT_EQ(st.gpr[1], 0u);
    EXPECT_EQ(st.gpr[6], reinterpret_cast<u64>(src) + 6) << "RSI += 6";
    EXPECT_EQ(st.gpr[7], reinterpret_cast<u64>(dst) + 6) << "RDI += 6";
}

// rep movsd — dword-granular copy (3 dwords).
TEST_F(CpuRuntimeTest, RepMovsd_CopiesDwords) {
    u32* src = reinterpret_cast<u32*>(mem.CodePtr() + 0x200);
    u32* dst = reinterpret_cast<u32*>(mem.CodePtr() + 0x300);
    src[0] = 0xAAAA0001; src[1] = 0xBBBB0002; src[2] = 0xCCCC0003;
    dst[0] = dst[1] = dst[2] = 0; dst[3] = 0xFFFFFFFF;

    const u8 program[] = {0xf3, 0xa5, 0xc3}; // rep movsd ; ret
    std::memcpy(mem.CodePtr(), program, sizeof(program));
    u8* guest_rsp = mem.StackTop() - 8;
    *reinterpret_cast<u64*>(guest_rsp) = kReturnSentinel;

    GuestState st{};
    st.rip = reinterpret_cast<u64>(mem.CodePtr());
    st.gpr[4] = reinterpret_cast<u64>(guest_rsp);
    st.gpr[6] = reinterpret_cast<u64>(src);
    st.gpr[7] = reinterpret_cast<u64>(dst);
    st.gpr[1] = 3;

    Runtime rt;
    rt.Run(st);
    EXPECT_EQ(dst[0], 0xAAAA0001u);
    EXPECT_EQ(dst[1], 0xBBBB0002u);
    EXPECT_EQ(dst[2], 0xCCCC0003u);
    EXPECT_EQ(dst[3], 0xFFFFFFFFu) << "stop after 3 dwords";
    EXPECT_EQ(st.gpr[6], reinterpret_cast<u64>(src) + 12);
    EXPECT_EQ(st.gpr[7], reinterpret_cast<u64>(dst) + 12);
}

// Bogus-pointer guard: rep movsq with a near-null SOURCE (RSI=0x3f) must NOT
// fault inside the host copy loop. It bails to the gateway with a distinct,
// non-fatal exit reason, leaving RSI/RDI/RCX at their pre-copy values so the
// crash report identifies the true bad pointer. Reproduces the CUSA02394
// signature (guest RSI=0x3f reaching libc memmove's rep movsq).
TEST_F(CpuRuntimeTest, RepMovsq_BogusSourcePointer_BailsCleanly) {
    u64* dst = reinterpret_cast<u64*>(mem.CodePtr() + 0x300);
    dst[0] = 0x1111111111111111ULL; dst[1] = 0x2222222222222222ULL;

    const u8 program[] = {0xf3, 0x48, 0xa5, 0xc3}; // rep movsq ; ret
    std::memcpy(mem.CodePtr(), program, sizeof(program));
    u8* guest_rsp = mem.StackTop() - 8;
    *reinterpret_cast<u64*>(guest_rsp) = kReturnSentinel;

    GuestState st{};
    st.rip = reinterpret_cast<u64>(mem.CodePtr());
    st.gpr[4] = reinterpret_cast<u64>(guest_rsp);
    st.gpr[6] = 0x3f;                          // RSI: the garbage source
    st.gpr[7] = reinterpret_cast<u64>(dst);    // RDI: valid dest
    st.gpr[1] = 2;                             // RCX: 2 qwords

    Runtime rt;
    rt.Run(st);
    EXPECT_EQ(st.exit_reason, static_cast<u32>(ExitReason::HelperRequestedExit))
        << "bad source pointer bails with HelperRequestedExit, not a fault";
    // No copy happened; dest untouched and RSI/RDI/RCX preserved for the report.
    EXPECT_EQ(dst[0], 0x1111111111111111ULL) << "dest must be untouched";
    EXPECT_EQ(dst[1], 0x2222222222222222ULL);
    EXPECT_EQ(st.gpr[6], 0x3fULL) << "RSI preserved for the crash report";
    EXPECT_EQ(st.gpr[1], 2ULL) << "RCX preserved";
}

// Bogus DEST pointer is caught the same way.
TEST_F(CpuRuntimeTest, RepMovsq_BogusDestPointer_BailsCleanly) {
    u64* src = reinterpret_cast<u64*>(mem.CodePtr() + 0x200);
    src[0] = 0xAAAAAAAAAAAAAAAAULL;

    const u8 program[] = {0xf3, 0x48, 0xa5, 0xc3}; // rep movsq ; ret
    std::memcpy(mem.CodePtr(), program, sizeof(program));
    u8* guest_rsp = mem.StackTop() - 8;
    *reinterpret_cast<u64*>(guest_rsp) = kReturnSentinel;

    GuestState st{};
    st.rip = reinterpret_cast<u64>(mem.CodePtr());
    st.gpr[4] = reinterpret_cast<u64>(guest_rsp);
    st.gpr[6] = reinterpret_cast<u64>(src);    // RSI: valid source
    st.gpr[7] = 0x100;                         // RDI: garbage dest (< 64 KiB)
    st.gpr[1] = 1;

    Runtime rt;
    rt.Run(st);
    EXPECT_EQ(st.exit_reason, static_cast<u32>(ExitReason::HelperRequestedExit))
        << "bad dest pointer bails cleanly";
    EXPECT_EQ(st.gpr[7], 0x100ULL) << "RDI preserved for the report";
}

// Regression: a fully valid copy is unaffected by the guard (pointers well
// above the 64 KiB threshold).
TEST_F(CpuRuntimeTest, RepMovsq_ValidCopy_GuardDoesNotInterfere) {
    u64* src = reinterpret_cast<u64*>(mem.CodePtr() + 0x200);
    u64* dst = reinterpret_cast<u64*>(mem.CodePtr() + 0x300);
    src[0] = 0xDEADBEEF00000001ULL; src[1] = 0xDEADBEEF00000002ULL;
    dst[0] = dst[1] = 0;

    const u8 program[] = {0xf3, 0x48, 0xa5, 0xc3}; // rep movsq ; ret
    std::memcpy(mem.CodePtr(), program, sizeof(program));
    u8* guest_rsp = mem.StackTop() - 8;
    *reinterpret_cast<u64*>(guest_rsp) = kReturnSentinel;

    GuestState st{};
    st.rip = reinterpret_cast<u64>(mem.CodePtr());
    st.gpr[4] = reinterpret_cast<u64>(guest_rsp);
    st.gpr[6] = reinterpret_cast<u64>(src);
    st.gpr[7] = reinterpret_cast<u64>(dst);
    st.gpr[1] = 2;

    Runtime rt;
    rt.Run(st);
    EXPECT_EQ(dst[0], 0xDEADBEEF00000001ULL) << "valid copy proceeds";
    EXPECT_EQ(dst[1], 0xDEADBEEF00000002ULL);
    EXPECT_EQ(st.gpr[6], reinterpret_cast<u64>(src) + 16);
    EXPECT_EQ(st.gpr[7], reinterpret_cast<u64>(dst) + 16);
}

// lodsb — load one byte into AL, advance RSI; upper bits of RAX preserved.
TEST_F(CpuRuntimeTest, Lodsb_LoadsByteIntoAlPreservingUpper) {
    u8* src = mem.CodePtr() + 0x200;
    src[0] = 0x7E;

    const u8 program[] = {0xac, 0xc3}; // lodsb ; ret
    std::memcpy(mem.CodePtr(), program, sizeof(program));
    u8* guest_rsp = mem.StackTop() - 8;
    *reinterpret_cast<u64*>(guest_rsp) = kReturnSentinel;

    GuestState st{};
    st.rip = reinterpret_cast<u64>(mem.CodePtr());
    st.gpr[4] = reinterpret_cast<u64>(guest_rsp);
    st.gpr[6] = reinterpret_cast<u64>(src);  // RSI
    st.gpr[0] = 0xDEAD'BEEF'CAFE'BB00ULL;    // RAX with known upper bits

    Runtime rt;
    rt.Run(st);
    EXPECT_EQ(st.gpr[0], 0xDEAD'BEEF'CAFE'BB7EULL)
        << "lodsb writes AL only; RAX[63:8] preserved";
    EXPECT_EQ(st.gpr[6], reinterpret_cast<u64>(src) + 1) << "RSI += 1";
}

// repne scasb — the strlen primitive: scan for AL==byte, stop on match.
// Buffer "AB\0..." with AL=0: REPNE scans while not-equal, stops at the NUL.
// Classic strlen computes len = (start_rcx - residual_rcx) - 1.
TEST_F(CpuRuntimeTest, RepneScasb_FindsTerminatorAndSetsZf) {
    u8* buf = mem.CodePtr() + 0x200;
    buf[0] = 'A'; buf[1] = 'B'; buf[2] = 'C'; buf[3] = 0x00; buf[4] = 'X';

    const u8 program[] = {0xf2, 0xae, 0xc3}; // repne scasb ; ret
    std::memcpy(mem.CodePtr(), program, sizeof(program));
    u8* guest_rsp = mem.StackTop() - 8;
    *reinterpret_cast<u64*>(guest_rsp) = kReturnSentinel;

    GuestState st{};
    st.rip = reinterpret_cast<u64>(mem.CodePtr());
    st.gpr[4] = reinterpret_cast<u64>(guest_rsp);
    st.gpr[0] = 0x00;                        // AL = NUL (scan target)
    st.gpr[7] = reinterpret_cast<u64>(buf);  // RDI
    st.gpr[1] = 16;                          // RCX = max scan

    Runtime rt;
    rt.Run(st);
    // Scans A,B,C,(NUL match). RDI lands one past the NUL (index 4).
    EXPECT_EQ(st.gpr[7], reinterpret_cast<u64>(buf) + 4) << "RDI past the NUL";
    EXPECT_EQ(st.gpr[1], 12u) << "RCX = 16 - 4 scanned";
    EXPECT_EQ(st.rflags & 0x40u, 0x40u) << "ZF set: matched the NUL";
    // strlen = (16 - 12) - 1 = 3 ("ABC")
    EXPECT_EQ((16u - st.gpr[1]) - 1u, 3u) << "derived strlen == 3";
}

// repne scasb that never matches within RCX — ZF clear, RCX exhausted to 0.
TEST_F(CpuRuntimeTest, RepneScasb_NoMatch_CountExhausted) {
    u8* buf = mem.CodePtr() + 0x200;
    for (int i = 0; i < 8; ++i) buf[i] = 'a' + i; // no NUL in first 4

    const u8 program[] = {0xf2, 0xae, 0xc3};
    std::memcpy(mem.CodePtr(), program, sizeof(program));
    u8* guest_rsp = mem.StackTop() - 8;
    *reinterpret_cast<u64*>(guest_rsp) = kReturnSentinel;

    GuestState st{};
    st.rip = reinterpret_cast<u64>(mem.CodePtr());
    st.gpr[4] = reinterpret_cast<u64>(guest_rsp);
    st.gpr[0] = 0x00; // scanning for NUL, none present
    st.gpr[7] = reinterpret_cast<u64>(buf);
    st.gpr[1] = 4;    // only scan 4

    Runtime rt;
    rt.Run(st);
    EXPECT_EQ(st.gpr[1], 0u) << "count exhausted";
    EXPECT_EQ(st.rflags & 0x40u, 0u) << "ZF clear: no match found";
    EXPECT_EQ(st.gpr[7], reinterpret_cast<u64>(buf) + 4) << "RDI advanced by 4";
}

// Single scasb — one compare, sets flags, advances RDI, leaves RCX alone.
// AL=0x40 ('@') vs [RDI]=0x40 -> equal -> ZF=1.
TEST_F(CpuRuntimeTest, Scasb_Single_ComparesAndSetsZf) {
    u8* buf = mem.CodePtr() + 0x200;
    buf[0] = 0x40;

    const u8 program[] = {0xae, 0xc3}; // scasb ; ret
    std::memcpy(mem.CodePtr(), program, sizeof(program));
    u8* guest_rsp = mem.StackTop() - 8;
    *reinterpret_cast<u64*>(guest_rsp) = kReturnSentinel;

    GuestState st{};
    st.rip = reinterpret_cast<u64>(mem.CodePtr());
    st.gpr[4] = reinterpret_cast<u64>(guest_rsp);
    st.gpr[0] = 0x40;
    st.gpr[7] = reinterpret_cast<u64>(buf);
    st.gpr[1] = 0x777; // must be untouched (non-REP)

    Runtime rt;
    rt.Run(st);
    EXPECT_EQ(st.rflags & 0x40u, 0x40u) << "ZF set: AL == [RDI]";
    EXPECT_EQ(st.gpr[7], reinterpret_cast<u64>(buf) + 1);
    EXPECT_EQ(st.gpr[1], 0x777u) << "non-REP must not touch RCX";
}

// SCAS must preserve DF and other non-arithmetic flag bits while updating the
// arithmetic ones. Set DF=1, do a single mismatching scasb, confirm DF stays
// and that the backward advance happened.
TEST_F(CpuRuntimeTest, Scasb_PreservesDf_BackwardAdvance) {
    u8* buf = mem.CodePtr() + 0x200;
    buf[5] = 0x11;

    const u8 program[] = {0xae, 0xc3};
    std::memcpy(mem.CodePtr(), program, sizeof(program));
    u8* guest_rsp = mem.StackTop() - 8;
    *reinterpret_cast<u64*>(guest_rsp) = kReturnSentinel;

    GuestState st{};
    st.rip = reinterpret_cast<u64>(mem.CodePtr());
    st.gpr[4] = reinterpret_cast<u64>(guest_rsp);
    st.gpr[0] = 0x22;                            // AL != [RDI] -> ZF=0
    st.gpr[7] = reinterpret_cast<u64>(buf) + 5;  // RDI
    st.rflags |= 0x400;                          // DF=1

    Runtime rt;
    rt.Run(st);
    EXPECT_EQ(st.rflags & 0x40u, 0u) << "ZF clear: 0x22 != 0x11";
    EXPECT_EQ(st.rflags & 0x400u, 0x400u) << "DF preserved across SCAS";
    EXPECT_EQ(st.gpr[7], reinterpret_cast<u64>(buf) + 4) << "RDI decremented (DF=1)";
}

// CLD/STD round-trip on the actual DF bit (these gate every string op above).
TEST_F(CpuRuntimeTest, CldStd_ToggleDirectionFlag) {
    {
        const u8 program[] = {0xfc, 0xc3}; // cld ; ret
        std::memcpy(mem.CodePtr(), program, sizeof(program));
        u8* guest_rsp = mem.StackTop() - 8;
        *reinterpret_cast<u64*>(guest_rsp) = kReturnSentinel;
        GuestState st{};
        st.rip = reinterpret_cast<u64>(mem.CodePtr());
        st.gpr[4] = reinterpret_cast<u64>(guest_rsp);
        st.rflags |= 0x400; // start DF=1
        Runtime rt; rt.Run(st);
        EXPECT_EQ(st.rflags & 0x400u, 0u) << "CLD clears DF";
    }
    {
        const u8 program[] = {0xfd, 0xc3}; // std ; ret
        std::memcpy(mem.CodePtr(), program, sizeof(program));
        u8* guest_rsp = mem.StackTop() - 8;
        *reinterpret_cast<u64*>(guest_rsp) = kReturnSentinel;
        GuestState st{};
        st.rip = reinterpret_cast<u64>(mem.CodePtr());
        st.gpr[4] = reinterpret_cast<u64>(guest_rsp);
        st.rflags &= ~static_cast<u64>(0x400); // start DF=0
        Runtime rt; rt.Run(st);
        EXPECT_EQ(st.rflags & 0x400u, 0x400u) << "STD sets DF";
    }
}

// ============================================================================
// MUL (unsigned multiply) coverage.
//
// Motivated by CUSA02394 "WE ARE DOOMED", which exited the JIT at `mul rcx`
// (48 f7 e1) in the eboot at guest 0x80020642b (exit_reason=2). MUL is the
// unsigned sibling of the already-supported IMUL; the product lands in a
// register pair (RDX:RAX for 64-bit). CF/OF are set iff the high half is
// non-zero. Tests cover all four widths, the overflow flag semantics, the
// exact crashing reg,reg form, and a memory-operand source.
//
// GPR index map: RAX=0 RCX=1 RDX=2. rflags CF=bit0 (0x1), OF=bit11 (0x800).
// ============================================================================

// mul rcx (64-bit) — the exact instruction that crashed CUSA02394. Product
// overflows into RDX, so CF and OF must be set.
TEST_F(CpuRuntimeTest, MulR64_Overflow_SetsHighHalfAndCfOf) {
    const u8 program[] = {0x48, 0xf7, 0xe1, 0xc3}; // mul rcx ; ret
    std::memcpy(mem.CodePtr(), program, sizeof(program));
    u8* guest_rsp = mem.StackTop() - 8;
    *reinterpret_cast<u64*>(guest_rsp) = kReturnSentinel;

    GuestState st{};
    st.rip = reinterpret_cast<u64>(mem.CodePtr());
    st.gpr[4] = reinterpret_cast<u64>(guest_rsp);
    st.gpr[0] = 0x1'0000'0000ULL; // RAX = 2^32
    st.gpr[1] = 0x1'0000'0000ULL; // RCX = 2^32

    Runtime rt;
    rt.Run(st);
    EXPECT_EQ(st.gpr[0], 0x0ULL) << "low half (RAX) = 0";
    EXPECT_EQ(st.gpr[2], 0x1ULL) << "high half (RDX) = 1 (2^64)";
    EXPECT_EQ(st.rflags & 0x1u, 0x1u) << "CF set: high half non-zero";
    EXPECT_EQ(st.rflags & 0x800u, 0x800u) << "OF set: high half non-zero";
}

// mul rcx (64-bit) with a product that fits — CF and OF must be clear.
TEST_F(CpuRuntimeTest, MulR64_NoOverflow_ClearsCfOf) {
    const u8 program[] = {0x48, 0xf7, 0xe1, 0xc3};
    std::memcpy(mem.CodePtr(), program, sizeof(program));
    u8* guest_rsp = mem.StackTop() - 8;
    *reinterpret_cast<u64*>(guest_rsp) = kReturnSentinel;

    GuestState st{};
    st.rip = reinterpret_cast<u64>(mem.CodePtr());
    st.gpr[4] = reinterpret_cast<u64>(guest_rsp);
    st.gpr[0] = 0xDEAD'BEEFULL; // RAX
    st.gpr[1] = 0x10ULL;        // RCX

    Runtime rt;
    rt.Run(st);
    EXPECT_EQ(st.gpr[0], 0xDEAD'BEEF0ULL) << "low half";
    EXPECT_EQ(st.gpr[2], 0x0ULL) << "high half = 0";
    EXPECT_EQ(st.rflags & 0x1u, 0u) << "CF clear: fits in 64 bits";
    EXPECT_EQ(st.rflags & 0x800u, 0u) << "OF clear";
}

// mul ecx (32-bit) — EDX:EAX = EAX*ECX, and the upper 32 bits of both RAX and
// RDX must be zero-extended (x86-64 rule for 32-bit results).
TEST_F(CpuRuntimeTest, MulR32_ZeroExtendsHighHalves) {
    const u8 program[] = {0xf7, 0xe1, 0xc3}; // mul ecx ; ret
    std::memcpy(mem.CodePtr(), program, sizeof(program));
    u8* guest_rsp = mem.StackTop() - 8;
    *reinterpret_cast<u64*>(guest_rsp) = kReturnSentinel;

    GuestState st{};
    st.rip = reinterpret_cast<u64>(mem.CodePtr());
    st.gpr[4] = reinterpret_cast<u64>(guest_rsp);
    st.gpr[0] = 0xAAAA'AAAA'FFFF'FFFFULL; // EAX = 0xFFFFFFFF (upper garbage)
    st.gpr[1] = 0xBBBB'BBBB'FFFF'FFFFULL; // ECX = 0xFFFFFFFF (upper garbage)

    Runtime rt;
    rt.Run(st);
    EXPECT_EQ(st.gpr[0], 0x1ULL) << "EAX = low 32 of 0xFFFFFFFF^2, RAX upper32 zeroed";
    EXPECT_EQ(st.gpr[2], 0xFFFF'FFFEULL) << "EDX = high 32, RDX upper32 zeroed";
    EXPECT_EQ(st.rflags & 0x1u, 0x1u) << "CF set: EDX non-zero";
    EXPECT_EQ(st.rflags & 0x800u, 0x800u) << "OF set";
}

// mul cx (16-bit) — DX:AX = AX*CX. Upper bits of RAX/RDX preserved.
TEST_F(CpuRuntimeTest, MulR16_WritesDxAxPreservingUpper) {
    const u8 program[] = {0x66, 0xf7, 0xe1, 0xc3}; // mul cx ; ret
    std::memcpy(mem.CodePtr(), program, sizeof(program));
    u8* guest_rsp = mem.StackTop() - 8;
    *reinterpret_cast<u64*>(guest_rsp) = kReturnSentinel;

    GuestState st{};
    st.rip = reinterpret_cast<u64>(mem.CodePtr());
    st.gpr[4] = reinterpret_cast<u64>(guest_rsp);
    st.gpr[0] = 0x1111'1111'1111'FFFFULL; // AX = 0xFFFF, upper preserved
    st.gpr[1] = 0x2222'2222'2222'FFFFULL; // CX = 0xFFFF
    st.gpr[2] = 0x3333'3333'3333'3333ULL; // RDX upper must survive

    Runtime rt;
    rt.Run(st);
    EXPECT_EQ(st.gpr[0] & 0xFFFF, 0x1u) << "AX = low 16 of 0xFFFF^2";
    EXPECT_EQ(st.gpr[0] >> 16, 0x1111'1111'1111ULL) << "RAX upper 48 preserved";
    EXPECT_EQ(st.gpr[2] & 0xFFFF, 0xFFFEu) << "DX = high 16";
    EXPECT_EQ(st.gpr[2] >> 16, 0x3333'3333'3333ULL) << "RDX upper 48 preserved";
    EXPECT_EQ(st.rflags & 0x1u, 0x1u) << "CF set: DX non-zero";
}

// mul cl (8-bit) — AX = AL*CL. Only RAX is written (no RDX destination).
TEST_F(CpuRuntimeTest, MulR8_WritesAxOnly) {
    const u8 program[] = {0xf6, 0xe1, 0xc3}; // mul cl ; ret
    std::memcpy(mem.CodePtr(), program, sizeof(program));
    u8* guest_rsp = mem.StackTop() - 8;
    *reinterpret_cast<u64*>(guest_rsp) = kReturnSentinel;

    GuestState st{};
    st.rip = reinterpret_cast<u64>(mem.CodePtr());
    st.gpr[4] = reinterpret_cast<u64>(guest_rsp);
    st.gpr[0] = 0x4444'4444'4444'44FFULL; // AL = 0xFF, upper preserved
    st.gpr[1] = 0x00FFULL;                // CL = 0xFF
    st.gpr[2] = 0x5555'5555'5555'5555ULL; // RDX must be UNTOUCHED (no MUL r/m8 dst)

    Runtime rt;
    rt.Run(st);
    EXPECT_EQ(st.gpr[0] & 0xFFFF, 0xFE01u) << "AX = 0xFF*0xFF = 0xFE01";
    EXPECT_EQ(st.gpr[0] >> 16, 0x4444'4444'4444ULL) << "RAX upper 48 preserved";
    EXPECT_EQ(st.gpr[2], 0x5555'5555'5555'5555ULL) << "RDX untouched by 8-bit MUL";
    EXPECT_EQ(st.rflags & 0x1u, 0x1u) << "CF set: AH (high byte) non-zero";
}

// mul qword[mem] — 64-bit multiply with a memory source operand. Exercises the
// EmitEffectiveAddress path (dereference into rcx before loading RAX).
TEST_F(CpuRuntimeTest, MulMem64_MultipliesFromMemory) {
    u64* slot = reinterpret_cast<u64*>(mem.CodePtr() + 0x100);
    *slot = 0x100ULL;

    // mul qword[rbx]  = 48 f7 23
    const u8 program[] = {0x48, 0xf7, 0x23, 0xc3};
    std::memcpy(mem.CodePtr(), program, sizeof(program));
    u8* guest_rsp = mem.StackTop() - 8;
    *reinterpret_cast<u64*>(guest_rsp) = kReturnSentinel;

    GuestState st{};
    st.rip = reinterpret_cast<u64>(mem.CodePtr());
    st.gpr[4] = reinterpret_cast<u64>(guest_rsp);
    st.gpr[0] = 0xABCDEFULL;                  // RAX
    st.gpr[3] = reinterpret_cast<u64>(slot);  // RBX -> memory factor

    Runtime rt;
    rt.Run(st);
    EXPECT_EQ(st.gpr[0], 0xABCDEF00ULL) << "RAX = 0xABCDEF * 0x100";
    EXPECT_EQ(st.gpr[2], 0x0ULL) << "high half = 0";
    EXPECT_EQ(st.rflags & 0x1u, 0u) << "CF clear: fits";
}

// MUL by zero — product 0, CF/OF clear.
TEST_F(CpuRuntimeTest, MulR64_ByZero_IsZeroAndClearsCf) {
    const u8 program[] = {0x48, 0xf7, 0xe1, 0xc3}; // mul rcx ; ret
    std::memcpy(mem.CodePtr(), program, sizeof(program));
    u8* guest_rsp = mem.StackTop() - 8;
    *reinterpret_cast<u64*>(guest_rsp) = kReturnSentinel;

    GuestState st{};
    st.rip = reinterpret_cast<u64>(mem.CodePtr());
    st.gpr[4] = reinterpret_cast<u64>(guest_rsp);
    st.gpr[0] = 0xDEAD'BEEF'CAFE'BABEULL;
    st.gpr[1] = 0x0ULL;

    Runtime rt;
    rt.Run(st);
    EXPECT_EQ(st.gpr[0], 0x0ULL);
    EXPECT_EQ(st.gpr[2], 0x0ULL);
    EXPECT_EQ(st.rflags & 0x1u, 0u) << "CF clear";
    EXPECT_EQ(st.rflags & 0x800u, 0u) << "OF clear";
}

// ============================================================================
// VINSERTPS — insert a scalar float lane (AVX, SSE4.1 semantics).
//
// Motivated by CUSA02394 "WE ARE DOOMED", which exited the JIT at a reg,reg
// `vinsertps` in the eboot at guest 0x80012f8d6 (exit_reason=2). The imm8
// control selects source lane [7:6], dest lane [5:4], and a 4-bit zero mask
// [3:0]. VEX form zeroes bits 255:128 of the dst YMM. Tests cover lane
// routing, the zero mask, the in-place (dst==src1) aliasing case, and a
// memory (m32) source.
//
// YMM layout in GuestState: ymm[n*4 + 0] = XMMn low 64 (dwords 0,1),
// ymm[n*4 + 1] = XMMn high 64 of the low 128 (dwords 2,3); ymm[n*4+2,3] =
// upper 128. XMM0 -> ymm[0..3], XMM1 -> ymm[4..7], XMM2 -> ymm[8..11].
// ============================================================================

// vinsertps xmm0, xmm1, xmm2, 0x10 — src lane0 -> dst lane1, no zero mask.
// dst = {src1.0, src2.0, src1.2, src1.3}.
TEST_F(CpuRuntimeTest, Vinsertps_SrcLane0_ToDstLane1) {
    const u8 program[] = {0xc4, 0xe3, 0x71, 0x21, 0xc2, 0x10, 0xc3};
    std::memcpy(mem.CodePtr(), program, sizeof(program));
    u8* guest_rsp = mem.StackTop() - 8;
    *reinterpret_cast<u64*>(guest_rsp) = kReturnSentinel;
    GuestState st{};
    st.rip = reinterpret_cast<u64>(mem.CodePtr());
    st.gpr[4] = reinterpret_cast<u64>(guest_rsp);
    // src1 = xmm1 (ymm[4],[5]) dwords A0,A1,A2,A3
    st.ymm[4] = (0xA1ULL << 32) | 0xA0ULL;
    st.ymm[5] = (0xA3ULL << 32) | 0xA2ULL;
    // src2 = xmm2 (ymm[8],[9]) dwords B0,B1,B2,B3
    st.ymm[8] = (0xB1ULL << 32) | 0xB0ULL;
    st.ymm[9] = (0xB3ULL << 32) | 0xB2ULL;
    // pollute dst (xmm0) incl upper YMM
    st.ymm[0] = 0x1111111111111111ULL; st.ymm[1] = 0x1111111111111111ULL;
    st.ymm[2] = 0x2222222222222222ULL; st.ymm[3] = 0x2222222222222222ULL;

    Runtime rt; rt.Run(st);
    EXPECT_EQ(st.ymm[0], 0x000000B0000000A0ULL) << "lane0=src1.0, lane1=src2.0";
    EXPECT_EQ(st.ymm[1], 0x000000A3000000A2ULL) << "lanes2,3 from src1";
    EXPECT_EQ(st.ymm[2], 0ULL) << "VEX zeros upper YMM";
    EXPECT_EQ(st.ymm[3], 0ULL);
}

// vinsertps xmm0, xmm1, xmm2, 0x4e — src lane1 -> dst lane0, zmask=0xe
// (zero lanes 1,2,3). Result keeps only the inserted dword in lane0.
TEST_F(CpuRuntimeTest, Vinsertps_ZeroMaskClearsOtherLanes) {
    const u8 program[] = {0xc4, 0xe3, 0x71, 0x21, 0xc2, 0x4e, 0xc3};
    std::memcpy(mem.CodePtr(), program, sizeof(program));
    u8* guest_rsp = mem.StackTop() - 8;
    *reinterpret_cast<u64*>(guest_rsp) = kReturnSentinel;
    GuestState st{};
    st.rip = reinterpret_cast<u64>(mem.CodePtr());
    st.gpr[4] = reinterpret_cast<u64>(guest_rsp);
    st.ymm[4] = (0xA1ULL << 32) | 0xA0ULL;
    st.ymm[5] = (0xA3ULL << 32) | 0xA2ULL;
    st.ymm[8] = (0xB1ULL << 32) | 0xB0ULL; // src2 lane1 = 0xB1
    st.ymm[9] = (0xB3ULL << 32) | 0xB2ULL;

    Runtime rt; rt.Run(st);
    EXPECT_EQ(st.ymm[0], 0x00000000000000B1ULL) << "lane0=src2.1, lane1 zeroed";
    EXPECT_EQ(st.ymm[1], 0ULL) << "lanes2,3 zeroed by mask";
    EXPECT_EQ(st.ymm[2], 0ULL);
    EXPECT_EQ(st.ymm[3], 0ULL);
}

// vinsertps xmm0, xmm1, xmm2, 0xc0 — src lane3 -> dst lane0, no zero.
TEST_F(CpuRuntimeTest, Vinsertps_SrcLane3_ToDstLane0) {
    const u8 program[] = {0xc4, 0xe3, 0x71, 0x21, 0xc2, 0xc0, 0xc3};
    std::memcpy(mem.CodePtr(), program, sizeof(program));
    u8* guest_rsp = mem.StackTop() - 8;
    *reinterpret_cast<u64*>(guest_rsp) = kReturnSentinel;
    GuestState st{};
    st.rip = reinterpret_cast<u64>(mem.CodePtr());
    st.gpr[4] = reinterpret_cast<u64>(guest_rsp);
    st.ymm[4] = (0xA1ULL << 32) | 0xA0ULL;
    st.ymm[5] = (0xA3ULL << 32) | 0xA2ULL;
    st.ymm[8] = (0xB1ULL << 32) | 0xB0ULL;
    st.ymm[9] = (0xB3ULL << 32) | 0xB2ULL; // src2 lane3 = 0xB3

    Runtime rt; rt.Run(st);
    EXPECT_EQ(st.ymm[0], 0x000000A1000000B3ULL) << "lane0=src2.3, lane1 from src1";
    EXPECT_EQ(st.ymm[1], 0x000000A3000000A2ULL) << "lanes2,3 from src1";
}

// vinsertps xmm0, xmm1, xmm2, 0x39 — src lane0 -> dst lane3, zmask=0x9
// (zero lanes 0 and 3). Lane3 is inserted THEN zeroed by the mask (mask wins).
TEST_F(CpuRuntimeTest, Vinsertps_ZeroMaskOverridesInsertedLane) {
    const u8 program[] = {0xc4, 0xe3, 0x71, 0x21, 0xc2, 0x39, 0xc3};
    std::memcpy(mem.CodePtr(), program, sizeof(program));
    u8* guest_rsp = mem.StackTop() - 8;
    *reinterpret_cast<u64*>(guest_rsp) = kReturnSentinel;
    GuestState st{};
    st.rip = reinterpret_cast<u64>(mem.CodePtr());
    st.gpr[4] = reinterpret_cast<u64>(guest_rsp);
    st.ymm[4] = (0xA1ULL << 32) | 0xA0ULL;
    st.ymm[5] = (0xA3ULL << 32) | 0xA2ULL;
    st.ymm[8] = (0xB1ULL << 32) | 0xB0ULL;
    st.ymm[9] = (0xB3ULL << 32) | 0xB2ULL;

    Runtime rt; rt.Run(st);
    // dst before mask: {A0, A1, A2, B0(inserted)}; mask 0x9 zeroes lanes 0,3.
    EXPECT_EQ(st.ymm[0], 0x000000A100000000ULL) << "lane0 zeroed, lane1 from src1";
    EXPECT_EQ(st.ymm[1], 0x00000000000000A2ULL) << "lane2 from src1, lane3 inserted-then-zeroed";
}

// In-place: vinsertps xmm1, xmm1, xmm2, 0x20 — dst == src1. Read-before-write
// ordering must make this a correct self-copy + patch. src lane0 -> dst lane2.
TEST_F(CpuRuntimeTest, Vinsertps_InPlace_DstEqualsSrc1) {
    const u8 program[] = {0xc4, 0xe3, 0x71, 0x21, 0xca, 0x20, 0xc3};
    std::memcpy(mem.CodePtr(), program, sizeof(program));
    u8* guest_rsp = mem.StackTop() - 8;
    *reinterpret_cast<u64*>(guest_rsp) = kReturnSentinel;
    GuestState st{};
    st.rip = reinterpret_cast<u64>(mem.CodePtr());
    st.gpr[4] = reinterpret_cast<u64>(guest_rsp);
    st.ymm[4] = (0xA1ULL << 32) | 0xA0ULL; // xmm1 = src1 AND dst
    st.ymm[5] = (0xA3ULL << 32) | 0xA2ULL;
    st.ymm[6] = 0xDEADULL; st.ymm[7] = 0xBEEFULL; // upper YMM1, must be zeroed
    st.ymm[8] = (0xB1ULL << 32) | 0xB0ULL; // src2 lane0 = 0xB0
    st.ymm[9] = (0xB3ULL << 32) | 0xB2ULL;

    Runtime rt; rt.Run(st);
    EXPECT_EQ(st.ymm[4], 0x000000A1000000A0ULL) << "lanes0,1 preserved from src1";
    EXPECT_EQ(st.ymm[5], 0x000000A3000000B0ULL) << "lane2=src2.0 inserted, lane3 from src1";
    EXPECT_EQ(st.ymm[6], 0ULL) << "VEX zeros upper YMM even in-place";
    EXPECT_EQ(st.ymm[7], 0ULL);
}

// Memory source: vinsertps xmm0, xmm1, dword[rbx], 0x10. m32 form ignores the
// imm src-lane field and loads the single dword from memory into dst lane1.
TEST_F(CpuRuntimeTest, Vinsertps_MemorySource_LoadsM32) {
    u32* slot = reinterpret_cast<u32*>(mem.CodePtr() + 0x100);
    *slot = 0xCAFEF00DU;

    const u8 program[] = {0xc4, 0xe3, 0x71, 0x21, 0x03, 0x10, 0xc3};
    std::memcpy(mem.CodePtr(), program, sizeof(program));
    u8* guest_rsp = mem.StackTop() - 8;
    *reinterpret_cast<u64*>(guest_rsp) = kReturnSentinel;
    GuestState st{};
    st.rip = reinterpret_cast<u64>(mem.CodePtr());
    st.gpr[4] = reinterpret_cast<u64>(guest_rsp);
    st.ymm[4] = (0xA1ULL << 32) | 0xA0ULL;
    st.ymm[5] = (0xA3ULL << 32) | 0xA2ULL;
    st.gpr[3] = reinterpret_cast<u64>(slot); // rbx -> m32

    Runtime rt; rt.Run(st);
    EXPECT_EQ(st.ymm[0], 0xCAFEF00D000000A0ULL) << "lane1 = loaded m32, lane0 from src1";
    EXPECT_EQ(st.ymm[1], 0x000000A3000000A2ULL) << "lanes2,3 from src1";
    EXPECT_EQ(st.ymm[2], 0ULL);
    EXPECT_EQ(st.ymm[3], 0ULL);
}

// Float-semantics sanity: build a vector {1.0f, 2.0f, 0, 0} by inserting 2.0f
// from src2 lane0 into dst lane1 of a {1.0f,...} src1. Confirms the raw bit
// movement matches IEEE-754 single bit patterns.
TEST_F(CpuRuntimeTest, Vinsertps_FloatBitPattern) {
    const u8 program[] = {0xc4, 0xe3, 0x71, 0x21, 0xc2, 0x10, 0xc3};
    std::memcpy(mem.CodePtr(), program, sizeof(program));
    u8* guest_rsp = mem.StackTop() - 8;
    *reinterpret_cast<u64*>(guest_rsp) = kReturnSentinel;
    GuestState st{};
    st.rip = reinterpret_cast<u64>(mem.CodePtr());
    st.gpr[4] = reinterpret_cast<u64>(guest_rsp);
    const u32 f1 = std::bit_cast<u32>(1.0f); // 0x3f800000
    const u32 f2 = std::bit_cast<u32>(2.0f); // 0x40000000
    st.ymm[4] = static_cast<u64>(f1);           // src1 lane0 = 1.0f
    st.ymm[5] = 0;
    st.ymm[8] = static_cast<u64>(f2);           // src2 lane0 = 2.0f
    st.ymm[9] = 0;

    Runtime rt; rt.Run(st);
    EXPECT_EQ(st.ymm[0], (static_cast<u64>(f2) << 32) | f1)
        << "lane0=1.0f, lane1=2.0f bit patterns";
    EXPECT_EQ(st.ymm[1], 0ULL);
}

// ============================================================================
// VPSHUFD with a MEMORY source operand.
//
// Motivated by CUSA02394 "WE ARE DOOMED", which exited the JIT at a
// `vpshufd xmm, [mem], imm8` in the eboot at guest 0x80012f8dc (exit_reason=2)
// — the instruction immediately following the VINSERTPS fixed earlier. The
// register-source form was already supported; only the m128/m256 source form
// was missing. These tests pin the memory path: per-dword shuffle from a
// memory operand, the broadcast (imm=0) and reverse (imm=0x1b) idioms, and the
// 256-bit per-lane variant.
//
// VPSHUFD dst[i] = src[(imm >> (i*2)) & 3] for each dword i in a 128-bit lane.
// YMM layout: ymm[n*4+0]=XMMn dwords0,1; ymm[n*4+1]=dwords2,3.
// ============================================================================

// vpshufd xmm0, [rbx], 0x1b — reverse the four dwords (3,2,1,0). Source in mem.
TEST_F(CpuRuntimeTest, VpshufdMem_ReversesDwords) {
    u32* slot = reinterpret_cast<u32*>(mem.CodePtr() + 0x100);
    slot[0] = 0x11111111; slot[1] = 0x22222222;
    slot[2] = 0x33333333; slot[3] = 0x44444444;

    const u8 program[] = {0xc4, 0xe1, 0x79, 0x70, 0x03, 0x1b, 0xc3};
    std::memcpy(mem.CodePtr(), program, sizeof(program));
    u8* guest_rsp = mem.StackTop() - 8;
    *reinterpret_cast<u64*>(guest_rsp) = kReturnSentinel;
    GuestState st{};
    st.rip = reinterpret_cast<u64>(mem.CodePtr());
    st.gpr[4] = reinterpret_cast<u64>(guest_rsp);
    st.gpr[3] = reinterpret_cast<u64>(slot); // rbx -> m128
    // pollute dst incl upper YMM
    st.ymm[0] = 0xDEAD; st.ymm[1] = 0xBEEF;
    st.ymm[2] = 0xCAFE; st.ymm[3] = 0xF00D;

    Runtime rt; rt.Run(st);
    // imm 0x1b = 00 01 10 11 -> dst lanes pick src[3],src[2],src[1],src[0]
    EXPECT_EQ(st.ymm[0], 0x3333333344444444ULL) << "lane0=src3, lane1=src2";
    EXPECT_EQ(st.ymm[1], 0x1111111122222222ULL) << "lane2=src1, lane3=src0";
    EXPECT_EQ(st.ymm[2], 0ULL) << "VEX zeros upper YMM";
    EXPECT_EQ(st.ymm[3], 0ULL);
}

// vpshufd xmm0, [rbx], 0x00 — broadcast src dword0 to all four lanes.
TEST_F(CpuRuntimeTest, VpshufdMem_BroadcastsLowestDword) {
    u32* slot = reinterpret_cast<u32*>(mem.CodePtr() + 0x100);
    slot[0] = 0xABCD1234; slot[1] = 0x55555555;
    slot[2] = 0x66666666; slot[3] = 0x77777777;

    const u8 program[] = {0xc4, 0xe1, 0x79, 0x70, 0x03, 0x00, 0xc3};
    std::memcpy(mem.CodePtr(), program, sizeof(program));
    u8* guest_rsp = mem.StackTop() - 8;
    *reinterpret_cast<u64*>(guest_rsp) = kReturnSentinel;
    GuestState st{};
    st.rip = reinterpret_cast<u64>(mem.CodePtr());
    st.gpr[4] = reinterpret_cast<u64>(guest_rsp);
    st.gpr[3] = reinterpret_cast<u64>(slot);

    Runtime rt; rt.Run(st);
    EXPECT_EQ(st.ymm[0], 0xABCD1234ABCD1234ULL) << "lanes0,1 = src0";
    EXPECT_EQ(st.ymm[1], 0xABCD1234ABCD1234ULL) << "lanes2,3 = src0";
    EXPECT_EQ(st.ymm[2], 0ULL);
    EXPECT_EQ(st.ymm[3], 0ULL);
}

// vpshufd xmm0, [rbx+disp], 0xe4 — identity shuffle (e4 = 11 10 01 00) from a
// displaced memory address. Confirms EmitEffectiveAddress handles base+disp and
// that the identity permutation copies the source verbatim.
TEST_F(CpuRuntimeTest, VpshufdMem_IdentityWithDisplacement) {
    u32* slot = reinterpret_cast<u32*>(mem.CodePtr() + 0x140);
    slot[0] = 0x0A0A0A0A; slot[1] = 0x0B0B0B0B;
    slot[2] = 0x0C0C0C0C; slot[3] = 0x0D0D0D0D;

    // vpshufd xmm0, [rbx+0x40], 0xe4  = c4 e1 79 70 43 40 e4
    const u8 program[] = {0xc4, 0xe1, 0x79, 0x70, 0x43, 0x40, 0xe4, 0xc3};
    std::memcpy(mem.CodePtr(), program, sizeof(program));
    u8* guest_rsp = mem.StackTop() - 8;
    *reinterpret_cast<u64*>(guest_rsp) = kReturnSentinel;
    GuestState st{};
    st.rip = reinterpret_cast<u64>(mem.CodePtr());
    st.gpr[4] = reinterpret_cast<u64>(guest_rsp);
    st.gpr[3] = reinterpret_cast<u64>(mem.CodePtr() + 0x100); // rbx; +0x40 -> 0x140

    Runtime rt; rt.Run(st);
    EXPECT_EQ(st.ymm[0], 0x0B0B0B0B0A0A0A0AULL) << "identity: lanes0,1 unchanged";
    EXPECT_EQ(st.ymm[1], 0x0D0D0D0D0C0C0C0CULL) << "identity: lanes2,3 unchanged";
}

// vpshufd ymm0, [rbx], 0x1b (256-bit) — per-128-bit-lane reverse on a memory
// source. Each 128-bit half is shuffled independently (no cross-lane).
TEST_F(CpuRuntimeTest, VpshufdMem_256Bit_PerLaneReverse) {
    u32* slot = reinterpret_cast<u32*>(mem.CodePtr() + 0x100);
    // low lane dwords 0..3, high lane dwords 4..7
    slot[0] = 0x100; slot[1] = 0x101; slot[2] = 0x102; slot[3] = 0x103;
    slot[4] = 0x200; slot[5] = 0x201; slot[6] = 0x202; slot[7] = 0x203;

    const u8 program[] = {0xc4, 0xe1, 0x7d, 0x70, 0x03, 0x1b, 0xc3};
    std::memcpy(mem.CodePtr(), program, sizeof(program));
    u8* guest_rsp = mem.StackTop() - 8;
    *reinterpret_cast<u64*>(guest_rsp) = kReturnSentinel;
    GuestState st{};
    st.rip = reinterpret_cast<u64>(mem.CodePtr());
    st.gpr[4] = reinterpret_cast<u64>(guest_rsp);
    st.gpr[3] = reinterpret_cast<u64>(slot);

    Runtime rt; rt.Run(st);
    // low lane reversed: 103,102,101,100
    EXPECT_EQ(st.ymm[0], (0x102ULL << 32) | 0x103ULL) << "low lane0=src3,lane1=src2";
    EXPECT_EQ(st.ymm[1], (0x100ULL << 32) | 0x101ULL) << "low lane2=src1,lane3=src0";
    // high lane reversed independently: 203,202,201,200
    EXPECT_EQ(st.ymm[2], (0x202ULL << 32) | 0x203ULL) << "high lane0=src3,lane1=src2";
    EXPECT_EQ(st.ymm[3], (0x200ULL << 32) | 0x201ULL) << "high lane2=src1,lane3=src0";
}

// Regression guard: the register-source form must STILL work after adding the
// memory path. vpshufd xmm0, xmm1, 0x1b.
TEST_F(CpuRuntimeTest, VpshufdReg_StillWorksAfterMemAdded) {
    const u8 program[] = {0xc4, 0xe1, 0x79, 0x70, 0xc1, 0x1b, 0xc3};
    std::memcpy(mem.CodePtr(), program, sizeof(program));
    u8* guest_rsp = mem.StackTop() - 8;
    *reinterpret_cast<u64*>(guest_rsp) = kReturnSentinel;
    GuestState st{};
    st.rip = reinterpret_cast<u64>(mem.CodePtr());
    st.gpr[4] = reinterpret_cast<u64>(guest_rsp);
    st.ymm[4] = 0x2222222211111111ULL; // xmm1 lanes 0,1
    st.ymm[5] = 0x4444444433333333ULL; // xmm1 lanes 2,3

    Runtime rt; rt.Run(st);
    EXPECT_EQ(st.ymm[0], 0x3333333344444444ULL) << "lane0=src3,lane1=src2";
    EXPECT_EQ(st.ymm[1], 0x1111111122222222ULL) << "lane2=src1,lane3=src0";
    EXPECT_EQ(st.ymm[2], 0ULL);
    EXPECT_EQ(st.ymm[3], 0ULL);
}

// ============================================================================
// VPMOVZXDQ — zero-extend packed dwords to qwords (AVX).
//
// Motivated by CUSA02394 "WE ARE DOOMED", which exited the JIT at a reg,reg
// 128-bit `vpmovzxdq` in the eboot at guest 0x8001f33ff (exit_reason=2). The
// op widens 32-bit lanes to 64-bit with ZERO extension (the high dword of each
// result qword is always 0, regardless of the source sign bit). 128-bit dst
// consumes 2 source dwords; 256-bit dst consumes 4.
//
// YMM layout: ymm[n*4 + k] is the k-th qword of YMMn (k=0..3).
// ============================================================================

// vpmovzxdq xmm0, xmm1 — the exact crashing form. Zero-extend, NOT sign:
// 0xFFFFFFFF must become 0x00000000FFFFFFFF, not 0xFFFFFFFFFFFFFFFF.
TEST_F(CpuRuntimeTest, Vpmovzxdq_Xmm_ZeroExtendsNotSign) {
    const u8 program[] = {0xc4, 0xe2, 0x79, 0x35, 0xc1, 0xc3};
    std::memcpy(mem.CodePtr(), program, sizeof(program));
    u8* guest_rsp = mem.StackTop() - 8;
    *reinterpret_cast<u64*>(guest_rsp) = kReturnSentinel;
    GuestState st{};
    st.rip = reinterpret_cast<u64>(mem.CodePtr());
    st.gpr[4] = reinterpret_cast<u64>(guest_rsp);
    // src = xmm1 (ymm[4],[5]): low 2 dwords = 0xFFFFFFFF, 0x80000000;
    // the upper dwords (in ymm[5]) must be ignored by the 128-bit form.
    st.ymm[4] = (0x80000000ULL << 32) | 0xFFFFFFFFULL;
    st.ymm[5] = 0xDEADBEEFCAFEF00DULL; // ignored
    // pollute dst incl upper YMM
    st.ymm[0] = 0x1111; st.ymm[1] = 0x2222;
    st.ymm[2] = 0x3333; st.ymm[3] = 0x4444;

    Runtime rt; rt.Run(st);
    EXPECT_EQ(st.ymm[0], 0x00000000FFFFFFFFULL) << "qword0 = zext(0xFFFFFFFF)";
    EXPECT_EQ(st.ymm[1], 0x0000000080000000ULL) << "qword1 = zext(0x80000000), NOT sign-extended";
    EXPECT_EQ(st.ymm[2], 0ULL) << "VEX zeros upper YMM";
    EXPECT_EQ(st.ymm[3], 0ULL);
}

// vpmovzxdq ymm0, xmm1 — 256-bit form: 4 source dwords -> 4 qwords.
TEST_F(CpuRuntimeTest, Vpmovzxdq_Ymm_WidensFourDwords) {
    const u8 program[] = {0xc4, 0xe2, 0x7d, 0x35, 0xc1, 0xc3};
    std::memcpy(mem.CodePtr(), program, sizeof(program));
    u8* guest_rsp = mem.StackTop() - 8;
    *reinterpret_cast<u64*>(guest_rsp) = kReturnSentinel;
    GuestState st{};
    st.rip = reinterpret_cast<u64>(mem.CodePtr());
    st.gpr[4] = reinterpret_cast<u64>(guest_rsp);
    // src xmm1 dwords 1,2,3,4
    st.ymm[4] = (0x2ULL << 32) | 0x1ULL;
    st.ymm[5] = (0x4ULL << 32) | 0x3ULL;

    Runtime rt; rt.Run(st);
    EXPECT_EQ(st.ymm[0], 0x1ULL) << "qword0";
    EXPECT_EQ(st.ymm[1], 0x2ULL) << "qword1";
    EXPECT_EQ(st.ymm[2], 0x3ULL) << "qword2";
    EXPECT_EQ(st.ymm[3], 0x4ULL) << "qword3";
}

// vpmovzxdq xmm0, [rbx] — 64-bit memory source (2 dwords).
TEST_F(CpuRuntimeTest, Vpmovzxdq_Mem64_WidensTwoDwords) {
    u32* slot = reinterpret_cast<u32*>(mem.CodePtr() + 0x100);
    slot[0] = 0xABCDEF01; slot[1] = 0x00000002;
    slot[2] = 0xFFFFFFFF; slot[3] = 0xFFFFFFFF; // must NOT be read (m64 only)

    const u8 program[] = {0xc4, 0xe2, 0x79, 0x35, 0x03, 0xc3}; // vpmovzxdq xmm0,[rbx]
    std::memcpy(mem.CodePtr(), program, sizeof(program));
    u8* guest_rsp = mem.StackTop() - 8;
    *reinterpret_cast<u64*>(guest_rsp) = kReturnSentinel;
    GuestState st{};
    st.rip = reinterpret_cast<u64>(mem.CodePtr());
    st.gpr[4] = reinterpret_cast<u64>(guest_rsp);
    st.gpr[3] = reinterpret_cast<u64>(slot); // rbx -> m64

    Runtime rt; rt.Run(st);
    EXPECT_EQ(st.ymm[0], 0x00000000ABCDEF01ULL) << "qword0 = zext(dword0)";
    EXPECT_EQ(st.ymm[1], 0x0000000000000002ULL) << "qword1 = zext(dword1)";
    EXPECT_EQ(st.ymm[2], 0ULL);
    EXPECT_EQ(st.ymm[3], 0ULL);
}

// In-place-ish aliasing: vpmovzxdq xmm1, xmm1. The low 2 dwords of xmm1 widen
// into xmm1's two qwords; reading src into scratch before writing dst makes
// this correct. (dst==src is c4 e2 79 35 c9.)
TEST_F(CpuRuntimeTest, Vpmovzxdq_DstEqualsSrc) {
    const u8 program[] = {0xc4, 0xe2, 0x79, 0x35, 0xc9, 0xc3};
    std::memcpy(mem.CodePtr(), program, sizeof(program));
    u8* guest_rsp = mem.StackTop() - 8;
    *reinterpret_cast<u64*>(guest_rsp) = kReturnSentinel;
    GuestState st{};
    st.rip = reinterpret_cast<u64>(mem.CodePtr());
    st.gpr[4] = reinterpret_cast<u64>(guest_rsp);
    st.ymm[4] = (0x0000000AULL << 32) | 0x0000000BULL; // dwords: 0x0B, 0x0A
    st.ymm[5] = 0x9999999999999999ULL; // ignored, then overwritten by VEX zero

    Runtime rt; rt.Run(st);
    EXPECT_EQ(st.ymm[4], 0x000000000000000BULL) << "qword0 = zext(dword0=0x0B)";
    EXPECT_EQ(st.ymm[5], 0x000000000000000AULL) << "qword1 = zext(dword1=0x0A)";
    EXPECT_EQ(st.ymm[6], 0ULL) << "VEX zeros upper YMM";
    EXPECT_EQ(st.ymm[7], 0ULL);
}

// ============================================================================
// VPEXTRQ — extract a 64-bit qword lane from an XMM to a GPR or memory.
//
// Motivated by CUSA02394 "WE ARE DOOMED", which exited the JIT at a reg,reg
// `vpextrq` in the eboot at guest 0x8001f3404 (exit_reason=2) — the
// instruction right after the VPMOVZXDQ fixed earlier, in the same block. The
// 64-bit sibling of the already-supported VPEXTRD: imm8[0] selects qword lane
// 0 or 1; a GPR destination receives the full 64 bits (no zero-extension); a
// memory destination receives 8 bytes.
//
// YMM layout: ymm[n*4 + 0] = XMMn qword0 (low 64), ymm[n*4 + 1] = qword1.
// ============================================================================

// vpextrq rax, xmm1, 0 — extract the low qword into a GPR (the crashing form).
TEST_F(CpuRuntimeTest, Vpextrq_Lane0_ToGpr) {
    const u8 program[] = {0xc4, 0xe3, 0xf9, 0x16, 0xc8, 0x00, 0xc3};
    std::memcpy(mem.CodePtr(), program, sizeof(program));
    u8* guest_rsp = mem.StackTop() - 8;
    *reinterpret_cast<u64*>(guest_rsp) = kReturnSentinel;
    GuestState st{};
    st.rip = reinterpret_cast<u64>(mem.CodePtr());
    st.gpr[4] = reinterpret_cast<u64>(guest_rsp);
    st.ymm[4] = 0x1122334455667788ULL; // xmm1 qword0
    st.ymm[5] = 0x99AABBCCDDEEFF00ULL; // xmm1 qword1
    st.gpr[0] = 0xDEADBEEFDEADBEEFULL; // rax pre-pollute

    Runtime rt; rt.Run(st);
    EXPECT_EQ(st.gpr[0], 0x1122334455667788ULL) << "rax = xmm1 qword0";
}

// vpextrq rax, xmm1, 1 — extract the high qword into a GPR.
TEST_F(CpuRuntimeTest, Vpextrq_Lane1_ToGpr) {
    const u8 program[] = {0xc4, 0xe3, 0xf9, 0x16, 0xc8, 0x01, 0xc3};
    std::memcpy(mem.CodePtr(), program, sizeof(program));
    u8* guest_rsp = mem.StackTop() - 8;
    *reinterpret_cast<u64*>(guest_rsp) = kReturnSentinel;
    GuestState st{};
    st.rip = reinterpret_cast<u64>(mem.CodePtr());
    st.gpr[4] = reinterpret_cast<u64>(guest_rsp);
    st.ymm[4] = 0x1122334455667788ULL;
    st.ymm[5] = 0x99AABBCCDDEEFF00ULL;

    Runtime rt; rt.Run(st);
    EXPECT_EQ(st.gpr[0], 0x99AABBCCDDEEFF00ULL) << "rax = xmm1 qword1";
}

// Full 64-bit write: even a value with a clear top bit must land verbatim,
// confirming there is no spurious zero/sign masking on the GPR write.
TEST_F(CpuRuntimeTest, Vpextrq_WritesFull64Bits) {
    const u8 program[] = {0xc4, 0xe3, 0xf9, 0x16, 0xc8, 0x00, 0xc3};
    std::memcpy(mem.CodePtr(), program, sizeof(program));
    u8* guest_rsp = mem.StackTop() - 8;
    *reinterpret_cast<u64*>(guest_rsp) = kReturnSentinel;
    GuestState st{};
    st.rip = reinterpret_cast<u64>(mem.CodePtr());
    st.gpr[4] = reinterpret_cast<u64>(guest_rsp);
    st.ymm[4] = 0xFFFFFFFFFFFFFFFFULL; // all ones: top bits must survive
    st.ymm[5] = 0;
    st.gpr[0] = 0;

    Runtime rt; rt.Run(st);
    EXPECT_EQ(st.gpr[0], 0xFFFFFFFFFFFFFFFFULL) << "all 64 bits written, no masking";
}

// vpextrq [rbx], xmm1, 1 — extract the high qword to memory (8-byte store).
TEST_F(CpuRuntimeTest, Vpextrq_Lane1_ToMemory) {
    u64* slot = reinterpret_cast<u64*>(mem.CodePtr() + 0x100);
    slot[0] = 0xAAAAAAAAAAAAAAAAULL;
    slot[1] = 0xBBBBBBBBBBBBBBBBULL; // sentinel after the 8-byte store

    const u8 program[] = {0xc4, 0xe3, 0xf9, 0x16, 0x0b, 0x01, 0xc3};
    std::memcpy(mem.CodePtr(), program, sizeof(program));
    u8* guest_rsp = mem.StackTop() - 8;
    *reinterpret_cast<u64*>(guest_rsp) = kReturnSentinel;
    GuestState st{};
    st.rip = reinterpret_cast<u64>(mem.CodePtr());
    st.gpr[4] = reinterpret_cast<u64>(guest_rsp);
    st.ymm[4] = 0x1122334455667788ULL;
    st.ymm[5] = 0x99AABBCCDDEEFF00ULL;
    st.gpr[3] = reinterpret_cast<u64>(slot); // rbx -> m64

    Runtime rt; rt.Run(st);
    EXPECT_EQ(slot[0], 0x99AABBCCDDEEFF00ULL) << "m64 = xmm1 qword1";
    EXPECT_EQ(slot[1], 0xBBBBBBBBBBBBBBBBULL) << "8-byte store must not spill";
}

// Regression guard: VPEXTRD (the dword sibling) must still work after adding
// the qword path. vpextrd eax, xmm1, 2.
TEST_F(CpuRuntimeTest, Vpextrd_StillWorksAfterVpextrqAdded) {
    const u8 program[] = {0xc4, 0xe3, 0x79, 0x16, 0xc8, 0x02, 0xc3}; // vpextrd eax,xmm1,2
    std::memcpy(mem.CodePtr(), program, sizeof(program));
    u8* guest_rsp = mem.StackTop() - 8;
    *reinterpret_cast<u64*>(guest_rsp) = kReturnSentinel;
    GuestState st{};
    st.rip = reinterpret_cast<u64>(mem.CodePtr());
    st.gpr[4] = reinterpret_cast<u64>(guest_rsp);
    st.ymm[4] = 0x2222222211111111ULL; // dwords 0,1 = 0x11111111, 0x22222222
    st.ymm[5] = 0x4444444433333333ULL; // dwords 2,3 = 0x33333333, 0x44444444
    st.gpr[0] = 0xFFFFFFFFFFFFFFFFULL;

    Runtime rt; rt.Run(st);
    EXPECT_EQ(st.gpr[0], 0x33333333ULL) << "eax = dword2, zero-extended to rax";
}

// ============================================================================
// VMINSS / VMAXSS / VMINSD / VMAXSD — scalar single/double min & max (AVX).
//
// Motivated by CUSA02394 "WE ARE DOOMED", which exited the JIT at a reg,reg
// `vminss` in the eboot at guest 0x8001f3433 (exit_reason=2), in the same
// vector block as the VPMOVZXDQ/VPEXTRQ fixed earlier. lane0 = min/max of the
// two source lane0s; lanes 127:32 (Ss) / 127:64 (Sd) come from src1; VEX
// zeroes bits 255:128. The key x86 subtlety pinned here: if either operand is
// NaN (or they are equal), the result is the SECOND source operand — these
// ops are deliberately non-commutative.
//
// YMM layout: ymm[n*4+0] = XMMn low 64 (lane0 float in low 32). XMM0->ymm[0..],
// XMM1->ymm[4..], XMM2->ymm[8..].
// ============================================================================

// vminss xmm0, xmm1, xmm2 — the crashing form. min(3.0, 5.0) = 3.0.
TEST_F(CpuRuntimeTest, Vminss_BasicMin) {
    const u8 program[] = {0xc4, 0xe1, 0x72, 0x5d, 0xc2, 0xc3};
    std::memcpy(mem.CodePtr(), program, sizeof(program));
    u8* guest_rsp = mem.StackTop() - 8;
    *reinterpret_cast<u64*>(guest_rsp) = kReturnSentinel;
    GuestState st{};
    st.rip = reinterpret_cast<u64>(mem.CodePtr());
    st.gpr[4] = reinterpret_cast<u64>(guest_rsp);
    const u32 f3 = std::bit_cast<u32>(3.0f), f5 = std::bit_cast<u32>(5.0f);
    // src1=xmm1 lane0=3.0, with nonzero upper lanes to verify preservation
    st.ymm[4] = (0xCAFEBABEULL << 32) | f3;
    st.ymm[5] = 0;
    st.ymm[8] = static_cast<u64>(f5); // src2=xmm2 lane0=5.0
    st.ymm[9] = 0;

    Runtime rt; rt.Run(st);
    EXPECT_EQ(static_cast<u32>(st.ymm[0]), f3) << "lane0 = min(3,5) = 3.0";
    EXPECT_EQ(st.ymm[0] >> 32, 0xCAFEBABEULL) << "lane1 preserved from src1";
    EXPECT_EQ(st.ymm[2], 0ULL) << "VEX zeros upper YMM";
    EXPECT_EQ(st.ymm[3], 0ULL);
}

// vmaxss xmm0, xmm1, xmm2 — max(3.0, 5.0) = 5.0.
TEST_F(CpuRuntimeTest, Vmaxss_BasicMax) {
    const u8 program[] = {0xc4, 0xe1, 0x72, 0x5f, 0xc2, 0xc3};
    std::memcpy(mem.CodePtr(), program, sizeof(program));
    u8* guest_rsp = mem.StackTop() - 8;
    *reinterpret_cast<u64*>(guest_rsp) = kReturnSentinel;
    GuestState st{};
    st.rip = reinterpret_cast<u64>(mem.CodePtr());
    st.gpr[4] = reinterpret_cast<u64>(guest_rsp);
    const u32 f3 = std::bit_cast<u32>(3.0f), f5 = std::bit_cast<u32>(5.0f);
    st.ymm[4] = static_cast<u64>(f3);
    st.ymm[8] = static_cast<u64>(f5);

    Runtime rt; rt.Run(st);
    EXPECT_EQ(static_cast<u32>(st.ymm[0]), f5) << "lane0 = max(3,5) = 5.0";
}

// NaN rule: VMINSS returns the SECOND operand when src1 is NaN.
// vminss xmm0, xmm1(NaN), xmm2(5.0) -> 5.0 (src2), not NaN.
TEST_F(CpuRuntimeTest, Vminss_NanSrc1_ReturnsSrc2) {
    const u8 program[] = {0xc4, 0xe1, 0x72, 0x5d, 0xc2, 0xc3};
    std::memcpy(mem.CodePtr(), program, sizeof(program));
    u8* guest_rsp = mem.StackTop() - 8;
    *reinterpret_cast<u64*>(guest_rsp) = kReturnSentinel;
    GuestState st{};
    st.rip = reinterpret_cast<u64>(mem.CodePtr());
    st.gpr[4] = reinterpret_cast<u64>(guest_rsp);
    const u32 fnan = std::bit_cast<u32>(std::numeric_limits<float>::quiet_NaN());
    const u32 f5 = std::bit_cast<u32>(5.0f);
    st.ymm[4] = static_cast<u64>(fnan); // src1 lane0 = NaN
    st.ymm[8] = static_cast<u64>(f5);   // src2 lane0 = 5.0

    Runtime rt; rt.Run(st);
    EXPECT_EQ(static_cast<u32>(st.ymm[0]), f5)
        << "min with NaN src1 returns src2 (x86 non-commutative rule)";
}

// NaN rule, other direction: src2 is NaN -> result is src2 (the NaN).
// vminss xmm0, xmm1(3.0), xmm2(NaN) -> NaN.
TEST_F(CpuRuntimeTest, Vminss_NanSrc2_ReturnsSrc2Nan) {
    const u8 program[] = {0xc4, 0xe1, 0x72, 0x5d, 0xc2, 0xc3};
    std::memcpy(mem.CodePtr(), program, sizeof(program));
    u8* guest_rsp = mem.StackTop() - 8;
    *reinterpret_cast<u64*>(guest_rsp) = kReturnSentinel;
    GuestState st{};
    st.rip = reinterpret_cast<u64>(mem.CodePtr());
    st.gpr[4] = reinterpret_cast<u64>(guest_rsp);
    const u32 f3 = std::bit_cast<u32>(3.0f);
    const u32 fnan = std::bit_cast<u32>(std::numeric_limits<float>::quiet_NaN());
    st.ymm[4] = static_cast<u64>(f3);
    st.ymm[8] = static_cast<u64>(fnan);

    Runtime rt; rt.Run(st);
    EXPECT_EQ(static_cast<u32>(st.ymm[0]), fnan)
        << "min with NaN src2 returns src2 (the NaN)";
}

// vminss xmm0, xmm1, [rbx] — memory source (m32).
TEST_F(CpuRuntimeTest, Vminss_MemorySource) {
    u32* slot = reinterpret_cast<u32*>(mem.CodePtr() + 0x100);
    *slot = std::bit_cast<u32>(2.0f);

    const u8 program[] = {0xc4, 0xe1, 0x72, 0x5d, 0x03, 0xc3};
    std::memcpy(mem.CodePtr(), program, sizeof(program));
    u8* guest_rsp = mem.StackTop() - 8;
    *reinterpret_cast<u64*>(guest_rsp) = kReturnSentinel;
    GuestState st{};
    st.rip = reinterpret_cast<u64>(mem.CodePtr());
    st.gpr[4] = reinterpret_cast<u64>(guest_rsp);
    st.ymm[4] = static_cast<u64>(std::bit_cast<u32>(3.0f)); // src1 = 3.0
    st.gpr[3] = reinterpret_cast<u64>(slot);                // rbx -> 2.0

    Runtime rt; rt.Run(st);
    EXPECT_EQ(static_cast<u32>(st.ymm[0]), std::bit_cast<u32>(2.0f))
        << "lane0 = min(3.0, mem 2.0) = 2.0";
}

// vminsd xmm0, xmm1, xmm2 — double precision: min(3.0, 5.0) = 3.0, lane upper
// 64 bits preserved from src1.
TEST_F(CpuRuntimeTest, Vminsd_BasicMin_PreservesUpper) {
    const u8 program[] = {0xc4, 0xe1, 0x73, 0x5d, 0xc2, 0xc3};
    std::memcpy(mem.CodePtr(), program, sizeof(program));
    u8* guest_rsp = mem.StackTop() - 8;
    *reinterpret_cast<u64*>(guest_rsp) = kReturnSentinel;
    GuestState st{};
    st.rip = reinterpret_cast<u64>(mem.CodePtr());
    st.gpr[4] = reinterpret_cast<u64>(guest_rsp);
    st.ymm[4] = std::bit_cast<u64>(3.0); // src1 lane0 (double)
    st.ymm[5] = 0x123456789ABCDEF0ULL;   // src1 upper 64 of low128, preserved
    st.ymm[8] = std::bit_cast<u64>(5.0);
    st.ymm[9] = 0;

    Runtime rt; rt.Run(st);
    EXPECT_EQ(st.ymm[0], std::bit_cast<u64>(3.0)) << "lane0 = min(3.0,5.0)";
    EXPECT_EQ(st.ymm[1], 0x123456789ABCDEF0ULL) << "upper 64 of low128 from src1";
    EXPECT_EQ(st.ymm[2], 0ULL) << "VEX zeros upper YMM";
    EXPECT_EQ(st.ymm[3], 0ULL);
}

// vmaxsd xmm0, xmm1, xmm2 — double precision max.
TEST_F(CpuRuntimeTest, Vmaxsd_BasicMax) {
    const u8 program[] = {0xc4, 0xe1, 0x73, 0x5f, 0xc2, 0xc3};
    std::memcpy(mem.CodePtr(), program, sizeof(program));
    u8* guest_rsp = mem.StackTop() - 8;
    *reinterpret_cast<u64*>(guest_rsp) = kReturnSentinel;
    GuestState st{};
    st.rip = reinterpret_cast<u64>(mem.CodePtr());
    st.gpr[4] = reinterpret_cast<u64>(guest_rsp);
    st.ymm[4] = std::bit_cast<u64>(3.0);
    st.ymm[8] = std::bit_cast<u64>(5.0);

    Runtime rt; rt.Run(st);
    EXPECT_EQ(st.ymm[0], std::bit_cast<u64>(5.0)) << "lane0 = max(3.0,5.0)";
}

// Regression guard: the existing scalar-fp arith (vmulss) must still work
// after the Min/Max enum extension. 3.0 * 5.0 = 15.0.
TEST_F(CpuRuntimeTest, Vmulss_StillWorksAfterMinMaxAdded) {
    const u8 program[] = {0xc4, 0xe1, 0x72, 0x59, 0xc2, 0xc3}; // vmulss xmm0,xmm1,xmm2
    std::memcpy(mem.CodePtr(), program, sizeof(program));
    u8* guest_rsp = mem.StackTop() - 8;
    *reinterpret_cast<u64*>(guest_rsp) = kReturnSentinel;
    GuestState st{};
    st.rip = reinterpret_cast<u64>(mem.CodePtr());
    st.gpr[4] = reinterpret_cast<u64>(guest_rsp);
    st.ymm[4] = static_cast<u64>(std::bit_cast<u32>(3.0f));
    st.ymm[8] = static_cast<u64>(std::bit_cast<u32>(5.0f));

    Runtime rt; rt.Run(st);
    EXPECT_EQ(static_cast<u32>(st.ymm[0]), std::bit_cast<u32>(15.0f))
        << "vmulss still computes 3*5=15";
}

// ============================================================================
// VEXTRACTPS — extract a 32-bit float lane from an XMM to a GPR or memory.
//
// Motivated by CUSA02394 "WE ARE DOOMED", which exited the JIT at a mem,reg
// `vextractps` (length 11, SIB+disp32 destination) in the eboot at guest
// 0x8001f3480 (exit_reason=2), in the same vector block as the
// VPMOVZXDQ/VPEXTRQ/VMINSS fixed earlier. imm8[1:0] picks the dword lane; the
// data movement is identical to VPEXTRD (a single 32-bit lane copied out). GPR
// dest zero-extends to 64 bits; memory dest writes exactly 4 bytes.
//
// YMM layout: dword lane n of XMMk is at ymm[k*4] >> (n%2 ? 32 : 0) for the
// low 2 lanes, ymm[k*4+1] for the high 2.
// ============================================================================

// vextractps eax, xmm1, 2 — extract lane 2 to a GPR, zero-extended.
TEST_F(CpuRuntimeTest, Vextractps_Lane2_ToGpr_ZeroExtends) {
    const u8 program[] = {0xc4, 0xe3, 0x79, 0x17, 0xc8, 0x02, 0xc3};
    std::memcpy(mem.CodePtr(), program, sizeof(program));
    u8* guest_rsp = mem.StackTop() - 8;
    *reinterpret_cast<u64*>(guest_rsp) = kReturnSentinel;
    GuestState st{};
    st.rip = reinterpret_cast<u64>(mem.CodePtr());
    st.gpr[4] = reinterpret_cast<u64>(guest_rsp);
    // xmm1 dwords: 0=0x11111111, 1=0x22222222, 2=0x33333333, 3=0x44444444
    st.ymm[4] = (0x22222222ULL << 32) | 0x11111111ULL;
    st.ymm[5] = (0x44444444ULL << 32) | 0x33333333ULL;
    st.gpr[0] = 0xDEADBEEFDEADBEEFULL; // rax pre-pollute

    Runtime rt; rt.Run(st);
    EXPECT_EQ(st.gpr[0], 0x33333333ULL) << "eax = dword2, zero-extended to rax";
    EXPECT_EQ(st.gpr[0] >> 32, 0u) << "upper 32 cleared";
}

// vextractps eax, xmm1, 0 — extract the low lane.
TEST_F(CpuRuntimeTest, Vextractps_Lane0_ToGpr) {
    const u8 program[] = {0xc4, 0xe3, 0x79, 0x17, 0xc8, 0x00, 0xc3};
    std::memcpy(mem.CodePtr(), program, sizeof(program));
    u8* guest_rsp = mem.StackTop() - 8;
    *reinterpret_cast<u64*>(guest_rsp) = kReturnSentinel;
    GuestState st{};
    st.rip = reinterpret_cast<u64>(mem.CodePtr());
    st.gpr[4] = reinterpret_cast<u64>(guest_rsp);
    st.ymm[4] = (0x22222222ULL << 32) | 0xABCD1234ULL;
    st.ymm[5] = 0;

    Runtime rt; rt.Run(st);
    EXPECT_EQ(st.gpr[0], 0xABCD1234ULL) << "eax = dword0";
}

// vextractps [rbx], xmm1, 1 — extract lane 1 to memory (the crashing form).
TEST_F(CpuRuntimeTest, Vextractps_Lane1_ToMemory_Writes4Bytes) {
    u32* slot = reinterpret_cast<u32*>(mem.CodePtr() + 0x100);
    slot[0] = 0xAAAAAAAA;
    slot[1] = 0xBBBBBBBB; // sentinel after the 4-byte store

    const u8 program[] = {0xc4, 0xe3, 0x79, 0x17, 0x0b, 0x01, 0xc3};
    std::memcpy(mem.CodePtr(), program, sizeof(program));
    u8* guest_rsp = mem.StackTop() - 8;
    *reinterpret_cast<u64*>(guest_rsp) = kReturnSentinel;
    GuestState st{};
    st.rip = reinterpret_cast<u64>(mem.CodePtr());
    st.gpr[4] = reinterpret_cast<u64>(guest_rsp);
    st.ymm[4] = (0xCAFEF00DULL << 32) | 0x11111111ULL; // dword1 = 0xCAFEF00D
    st.ymm[5] = 0;
    st.gpr[3] = reinterpret_cast<u64>(slot); // rbx -> m32

    Runtime rt; rt.Run(st);
    EXPECT_EQ(slot[0], 0xCAFEF00DU) << "m32 = dword1";
    EXPECT_EQ(slot[1], 0xBBBBBBBBU) << "4-byte store must not spill into the next dword";
}

// vextractps [rbx+disp8], xmm1, 3 — memory destination with displacement,
// exercising the EmitEffectiveAddress base+disp path (a stand-in for the
// SIB+disp32 long form that crashed).
TEST_F(CpuRuntimeTest, Vextractps_Lane3_ToMemoryWithDisplacement) {
    u32* base = reinterpret_cast<u32*>(mem.CodePtr() + 0x100);
    u32* slot = reinterpret_cast<u32*>(mem.CodePtr() + 0x140); // base + 0x40
    *slot = 0;

    // vextractps [rbx+0x40], xmm1, 3  = c4 e3 79 17 4b 40 03
    const u8 program[] = {0xc4, 0xe3, 0x79, 0x17, 0x4b, 0x40, 0x03, 0xc3};
    std::memcpy(mem.CodePtr(), program, sizeof(program));
    u8* guest_rsp = mem.StackTop() - 8;
    *reinterpret_cast<u64*>(guest_rsp) = kReturnSentinel;
    GuestState st{};
    st.rip = reinterpret_cast<u64>(mem.CodePtr());
    st.gpr[4] = reinterpret_cast<u64>(guest_rsp);
    st.ymm[4] = 0;
    st.ymm[5] = (0x99999999ULL << 32) | 0x55555555ULL; // dword3 = 0x99999999
    st.gpr[3] = reinterpret_cast<u64>(base); // rbx; +0x40 -> slot

    Runtime rt; rt.Run(st);
    EXPECT_EQ(*slot, 0x99999999U) << "stored dword3 at [rbx+0x40]";
}

// Float bit-pattern sanity: extract 2.0f from lane 1.
TEST_F(CpuRuntimeTest, Vextractps_FloatBitPattern) {
    const u8 program[] = {0xc4, 0xe3, 0x79, 0x17, 0xc8, 0x01, 0xc3};
    std::memcpy(mem.CodePtr(), program, sizeof(program));
    u8* guest_rsp = mem.StackTop() - 8;
    *reinterpret_cast<u64*>(guest_rsp) = kReturnSentinel;
    GuestState st{};
    st.rip = reinterpret_cast<u64>(mem.CodePtr());
    st.gpr[4] = reinterpret_cast<u64>(guest_rsp);
    const u32 f1 = std::bit_cast<u32>(1.0f), f2 = std::bit_cast<u32>(2.0f);
    st.ymm[4] = (static_cast<u64>(f2) << 32) | f1; // lane0=1.0f, lane1=2.0f
    st.ymm[5] = 0;

    Runtime rt; rt.Run(st);
    EXPECT_EQ(st.gpr[0], static_cast<u64>(f2)) << "extracted 2.0f bit pattern";
}

// ============================================================================
// VSHUFPS — shuffle packed single-precision floats from TWO sources.
//
// Motivated by CUSA02394 "WE ARE DOOMED", which exited the JIT at a reg,reg
// 128-bit `vshufps` (2-byte VEX, length 5) in the eboot at guest 0x8002416a0
// (exit_reason=2). The low two result dwords come from src1, the high two from
// src2, each picked by a 2-bit imm8 field — distinct from VPSHUFD's
// single-source shuffle.
//
//   dst[0]=src1[imm[1:0]] dst[1]=src1[imm[3:2]]
//   dst[2]=src2[imm[5:4]] dst[3]=src2[imm[7:6]]
//
// YMM layout: ymm[k*4+0]=XMMk dwords0,1; ymm[k*4+1]=dwords2,3.
// XMM0->ymm[0..], XMM1->ymm[4..], XMM2->ymm[8..].
// ============================================================================

// vshufps xmm0, xmm1, xmm2, 0x1b — the crashing form. imm 0x1b = 00 01 10 11:
// dst = {src1[3], src1[2], src2[1], src2[0]}.
TEST_F(CpuRuntimeTest, Vshufps_DualSourceShuffle) {
    const u8 program[] = {0xc5, 0xf0, 0xc6, 0xc2, 0x1b, 0xc3};
    std::memcpy(mem.CodePtr(), program, sizeof(program));
    u8* guest_rsp = mem.StackTop() - 8;
    *reinterpret_cast<u64*>(guest_rsp) = kReturnSentinel;
    GuestState st{};
    st.rip = reinterpret_cast<u64>(mem.CodePtr());
    st.gpr[4] = reinterpret_cast<u64>(guest_rsp);
    // src1=xmm1 dwords A0,A1,A2,A3 ; src2=xmm2 dwords B0,B1,B2,B3
    st.ymm[4] = (0xA1ULL << 32) | 0xA0ULL;
    st.ymm[5] = (0xA3ULL << 32) | 0xA2ULL;
    st.ymm[8] = (0xB1ULL << 32) | 0xB0ULL;
    st.ymm[9] = (0xB3ULL << 32) | 0xB2ULL;
    // pollute dst incl upper YMM
    st.ymm[0] = 0xDEAD; st.ymm[1] = 0xBEEF;
    st.ymm[2] = 0xCAFE; st.ymm[3] = 0xF00D;

    Runtime rt; rt.Run(st);
    EXPECT_EQ(st.ymm[0], 0x000000A2000000A3ULL) << "lane0=src1[3], lane1=src1[2]";
    EXPECT_EQ(st.ymm[1], 0x000000B0000000B1ULL) << "lane2=src2[1], lane3=src2[0]";
    EXPECT_EQ(st.ymm[2], 0ULL) << "VEX zeros upper YMM";
    EXPECT_EQ(st.ymm[3], 0ULL);
}

// imm 0x00: dst = {src1[0], src1[0], src2[0], src2[0]} — broadcast each
// source's lane0 into its half. A common "splat low" idiom.
TEST_F(CpuRuntimeTest, Vshufps_BroadcastLowFromEachSource) {
    const u8 program[] = {0xc5, 0xf0, 0xc6, 0xc2, 0x00, 0xc3};
    std::memcpy(mem.CodePtr(), program, sizeof(program));
    u8* guest_rsp = mem.StackTop() - 8;
    *reinterpret_cast<u64*>(guest_rsp) = kReturnSentinel;
    GuestState st{};
    st.rip = reinterpret_cast<u64>(mem.CodePtr());
    st.gpr[4] = reinterpret_cast<u64>(guest_rsp);
    st.ymm[4] = (0xA1ULL << 32) | 0xA0ULL;
    st.ymm[5] = (0xA3ULL << 32) | 0xA2ULL;
    st.ymm[8] = (0xB1ULL << 32) | 0xB0ULL;
    st.ymm[9] = (0xB3ULL << 32) | 0xB2ULL;

    Runtime rt; rt.Run(st);
    EXPECT_EQ(st.ymm[0], 0x000000A0000000A0ULL) << "lanes0,1 = src1[0]";
    EXPECT_EQ(st.ymm[1], 0x000000B0000000B0ULL) << "lanes2,3 = src2[0]";
}

// imm 0xe4: dst = {src1[0],src1[1],src2[2],src2[3]} — keep src1's low half and
// src2's high half (a common "merge halves" shuffle).
TEST_F(CpuRuntimeTest, Vshufps_MergeHalves) {
    const u8 program[] = {0xc5, 0xf0, 0xc6, 0xc2, 0xe4, 0xc3};
    std::memcpy(mem.CodePtr(), program, sizeof(program));
    u8* guest_rsp = mem.StackTop() - 8;
    *reinterpret_cast<u64*>(guest_rsp) = kReturnSentinel;
    GuestState st{};
    st.rip = reinterpret_cast<u64>(mem.CodePtr());
    st.gpr[4] = reinterpret_cast<u64>(guest_rsp);
    st.ymm[4] = (0xA1ULL << 32) | 0xA0ULL;
    st.ymm[5] = (0xA3ULL << 32) | 0xA2ULL;
    st.ymm[8] = (0xB1ULL << 32) | 0xB0ULL;
    st.ymm[9] = (0xB3ULL << 32) | 0xB2ULL;

    Runtime rt; rt.Run(st);
    EXPECT_EQ(st.ymm[0], 0x000000A1000000A0ULL) << "src1 low half preserved";
    EXPECT_EQ(st.ymm[1], 0x000000B3000000B2ULL) << "src2 high half";
}

// Memory source: vshufps xmm0, xmm1, [rbx], 0x88. imm 0x88 = 10 00 10 00:
// dst = {src1[0], src1[2], src2[0], src2[2]}.
TEST_F(CpuRuntimeTest, Vshufps_MemorySource) {
    u32* slot = reinterpret_cast<u32*>(mem.CodePtr() + 0x100);
    slot[0] = 0xB0; slot[1] = 0xB1; slot[2] = 0xB2; slot[3] = 0xB3;

    const u8 program[] = {0xc5, 0xf0, 0xc6, 0x03, 0x88, 0xc3};
    std::memcpy(mem.CodePtr(), program, sizeof(program));
    u8* guest_rsp = mem.StackTop() - 8;
    *reinterpret_cast<u64*>(guest_rsp) = kReturnSentinel;
    GuestState st{};
    st.rip = reinterpret_cast<u64>(mem.CodePtr());
    st.gpr[4] = reinterpret_cast<u64>(guest_rsp);
    st.ymm[4] = (0xA1ULL << 32) | 0xA0ULL;
    st.ymm[5] = (0xA3ULL << 32) | 0xA2ULL;
    st.gpr[3] = reinterpret_cast<u64>(slot); // rbx -> m128

    Runtime rt; rt.Run(st);
    EXPECT_EQ(st.ymm[0], 0x000000A2000000A0ULL) << "lane0=src1[0], lane1=src1[2]";
    EXPECT_EQ(st.ymm[1], 0x000000B2000000B0ULL) << "lane2=src2[0], lane3=src2[2]";
    EXPECT_EQ(st.ymm[2], 0ULL);
    EXPECT_EQ(st.ymm[3], 0ULL);
}

// In-place: vshufps xmm1, xmm1, xmm2, 0x1b — dst == src1. Reading src1 into
// scratch before writing dst keeps this correct.
TEST_F(CpuRuntimeTest, Vshufps_InPlace_DstEqualsSrc1) {
    const u8 program[] = {0xc5, 0xf0, 0xc6, 0xca, 0x1b, 0xc3}; // vshufps xmm1,xmm1,xmm2,0x1b
    std::memcpy(mem.CodePtr(), program, sizeof(program));
    u8* guest_rsp = mem.StackTop() - 8;
    *reinterpret_cast<u64*>(guest_rsp) = kReturnSentinel;
    GuestState st{};
    st.rip = reinterpret_cast<u64>(mem.CodePtr());
    st.gpr[4] = reinterpret_cast<u64>(guest_rsp);
    st.ymm[4] = (0xA1ULL << 32) | 0xA0ULL; // xmm1 = src1 AND dst
    st.ymm[5] = (0xA3ULL << 32) | 0xA2ULL;
    st.ymm[6] = 0xDEAD; st.ymm[7] = 0xBEEF; // upper YMM1, must be zeroed
    st.ymm[8] = (0xB1ULL << 32) | 0xB0ULL;
    st.ymm[9] = (0xB3ULL << 32) | 0xB2ULL;

    Runtime rt; rt.Run(st);
    EXPECT_EQ(st.ymm[4], 0x000000A2000000A3ULL) << "lane0=src1[3], lane1=src1[2]";
    EXPECT_EQ(st.ymm[5], 0x000000B0000000B1ULL) << "lane2=src2[1], lane3=src2[0]";
    EXPECT_EQ(st.ymm[6], 0ULL) << "VEX zeros upper YMM even in-place";
    EXPECT_EQ(st.ymm[7], 0ULL);
}

// Float bit-pattern sanity: shuffle {1,2,3,4}f from src1/src2 with imm 0xe4
// (src1 low half, src2 high half) -> {1.0f, 2.0f, 7.0f, 8.0f}.
TEST_F(CpuRuntimeTest, Vshufps_FloatBitPattern) {
    const u8 program[] = {0xc5, 0xf0, 0xc6, 0xc2, 0xe4, 0xc3};
    std::memcpy(mem.CodePtr(), program, sizeof(program));
    u8* guest_rsp = mem.StackTop() - 8;
    *reinterpret_cast<u64*>(guest_rsp) = kReturnSentinel;
    GuestState st{};
    st.rip = reinterpret_cast<u64>(mem.CodePtr());
    st.gpr[4] = reinterpret_cast<u64>(guest_rsp);
    const u32 f1 = std::bit_cast<u32>(1.0f), f2 = std::bit_cast<u32>(2.0f);
    const u32 f7 = std::bit_cast<u32>(7.0f), f8 = std::bit_cast<u32>(8.0f);
    st.ymm[4] = (static_cast<u64>(f2) << 32) | f1;  // src1 lanes 1.0,2.0
    st.ymm[5] = 0;
    st.ymm[8] = 0;
    st.ymm[9] = (static_cast<u64>(f8) << 32) | f7;  // src2 lanes 2,3 = 7.0,8.0

    Runtime rt; rt.Run(st);
    EXPECT_EQ(st.ymm[0], (static_cast<u64>(f2) << 32) | f1) << "low half = 1.0f,2.0f from src1";
    EXPECT_EQ(st.ymm[1], (static_cast<u64>(f8) << 32) | f7) << "high half = 7.0f,8.0f from src2";
}

// ============================================================================
// VUNPCKLPS / VUNPCKHPS / VUNPCKLPD / VUNPCKHPD — interleave packed floats.
//
// Motivated by CUSA02394 "WE ARE DOOMED", which exited the JIT at a reg,reg
// 128-bit `vunpcklps` (length 4) in the eboot at guest 0x800241814
// (exit_reason=2). These zip two sources together within each 128-bit lane:
//   LPS dst = {src1[0],src2[0],src1[1],src2[1]}  HPS dst = {src1[2],src2[2],src1[3],src2[3]}
//   LPD dst = {src1.q0, src2.q0}                 HPD dst = {src1.q1, src2.q1}
//
// YMM layout: ymm[k*4+0]=XMMk dwords0,1 / qword0; ymm[k*4+1]=dwords2,3 / qword1.
// XMM0->ymm[0..], XMM1->ymm[4..], XMM2->ymm[8..].
// ============================================================================

// vunpcklps xmm0, xmm1, xmm2 — the crashing form. Interleave low dwords.
TEST_F(CpuRuntimeTest, Vunpcklps_InterleavesLowDwords) {
    const u8 program[] = {0xc5, 0xf0, 0x14, 0xc2, 0xc3};
    std::memcpy(mem.CodePtr(), program, sizeof(program));
    u8* guest_rsp = mem.StackTop() - 8;
    *reinterpret_cast<u64*>(guest_rsp) = kReturnSentinel;
    GuestState st{};
    st.rip = reinterpret_cast<u64>(mem.CodePtr());
    st.gpr[4] = reinterpret_cast<u64>(guest_rsp);
    st.ymm[4] = (0xA1ULL << 32) | 0xA0ULL; // src1 dwords A0,A1
    st.ymm[5] = (0xA3ULL << 32) | 0xA2ULL; //           A2,A3
    st.ymm[8] = (0xB1ULL << 32) | 0xB0ULL; // src2 dwords B0,B1
    st.ymm[9] = (0xB3ULL << 32) | 0xB2ULL; //           B2,B3
    st.ymm[0] = 0xDEAD; st.ymm[1] = 0xBEEF;
    st.ymm[2] = 0xCAFE; st.ymm[3] = 0xF00D;

    Runtime rt; rt.Run(st);
    // {A0,B0,A1,B1}
    EXPECT_EQ(st.ymm[0], 0x000000B0000000A0ULL) << "lane0=src1[0], lane1=src2[0]";
    EXPECT_EQ(st.ymm[1], 0x000000B1000000A1ULL) << "lane2=src1[1], lane3=src2[1]";
    EXPECT_EQ(st.ymm[2], 0ULL) << "VEX zeros upper YMM";
    EXPECT_EQ(st.ymm[3], 0ULL);
}

// vunpckhps xmm0, xmm1, xmm2 — interleave high dwords.
TEST_F(CpuRuntimeTest, Vunpckhps_InterleavesHighDwords) {
    const u8 program[] = {0xc5, 0xf0, 0x15, 0xc2, 0xc3};
    std::memcpy(mem.CodePtr(), program, sizeof(program));
    u8* guest_rsp = mem.StackTop() - 8;
    *reinterpret_cast<u64*>(guest_rsp) = kReturnSentinel;
    GuestState st{};
    st.rip = reinterpret_cast<u64>(mem.CodePtr());
    st.gpr[4] = reinterpret_cast<u64>(guest_rsp);
    st.ymm[4] = (0xA1ULL << 32) | 0xA0ULL;
    st.ymm[5] = (0xA3ULL << 32) | 0xA2ULL;
    st.ymm[8] = (0xB1ULL << 32) | 0xB0ULL;
    st.ymm[9] = (0xB3ULL << 32) | 0xB2ULL;

    Runtime rt; rt.Run(st);
    // {A2,B2,A3,B3}
    EXPECT_EQ(st.ymm[0], 0x000000B2000000A2ULL) << "lane0=src1[2], lane1=src2[2]";
    EXPECT_EQ(st.ymm[1], 0x000000B3000000A3ULL) << "lane2=src1[3], lane3=src2[3]";
    EXPECT_EQ(st.ymm[2], 0ULL);
    EXPECT_EQ(st.ymm[3], 0ULL);
}

// vunpcklpd xmm0, xmm1, xmm2 — low qwords: {src1.q0, src2.q0}.
TEST_F(CpuRuntimeTest, Vunpcklpd_InterleavesLowQwords) {
    const u8 program[] = {0xc5, 0xf1, 0x14, 0xc2, 0xc3};
    std::memcpy(mem.CodePtr(), program, sizeof(program));
    u8* guest_rsp = mem.StackTop() - 8;
    *reinterpret_cast<u64*>(guest_rsp) = kReturnSentinel;
    GuestState st{};
    st.rip = reinterpret_cast<u64>(mem.CodePtr());
    st.gpr[4] = reinterpret_cast<u64>(guest_rsp);
    st.ymm[4] = 0xAAAAAAAAAAAAAAAAULL; // src1 q0
    st.ymm[5] = 0x1111111111111111ULL; // src1 q1
    st.ymm[8] = 0xBBBBBBBBBBBBBBBBULL; // src2 q0
    st.ymm[9] = 0x2222222222222222ULL; // src2 q1

    Runtime rt; rt.Run(st);
    EXPECT_EQ(st.ymm[0], 0xAAAAAAAAAAAAAAAAULL) << "q0 = src1.q0";
    EXPECT_EQ(st.ymm[1], 0xBBBBBBBBBBBBBBBBULL) << "q1 = src2.q0";
    EXPECT_EQ(st.ymm[2], 0ULL);
    EXPECT_EQ(st.ymm[3], 0ULL);
}

// vunpckhpd xmm0, xmm1, xmm2 — high qwords: {src1.q1, src2.q1}.
TEST_F(CpuRuntimeTest, Vunpckhpd_InterleavesHighQwords) {
    const u8 program[] = {0xc5, 0xf1, 0x15, 0xc2, 0xc3};
    std::memcpy(mem.CodePtr(), program, sizeof(program));
    u8* guest_rsp = mem.StackTop() - 8;
    *reinterpret_cast<u64*>(guest_rsp) = kReturnSentinel;
    GuestState st{};
    st.rip = reinterpret_cast<u64>(mem.CodePtr());
    st.gpr[4] = reinterpret_cast<u64>(guest_rsp);
    st.ymm[4] = 0xAAAAAAAAAAAAAAAAULL;
    st.ymm[5] = 0x1111111111111111ULL;
    st.ymm[8] = 0xBBBBBBBBBBBBBBBBULL;
    st.ymm[9] = 0x2222222222222222ULL;

    Runtime rt; rt.Run(st);
    EXPECT_EQ(st.ymm[0], 0x1111111111111111ULL) << "q0 = src1.q1";
    EXPECT_EQ(st.ymm[1], 0x2222222222222222ULL) << "q1 = src2.q1";
    EXPECT_EQ(st.ymm[2], 0ULL);
    EXPECT_EQ(st.ymm[3], 0ULL);
}

// Memory source: vunpcklps xmm0, xmm1, [rbx].
TEST_F(CpuRuntimeTest, Vunpcklps_MemorySource) {
    u32* slot = reinterpret_cast<u32*>(mem.CodePtr() + 0x100);
    slot[0] = 0xB0; slot[1] = 0xB1; slot[2] = 0xB2; slot[3] = 0xB3;

    const u8 program[] = {0xc5, 0xf0, 0x14, 0x03, 0xc3};
    std::memcpy(mem.CodePtr(), program, sizeof(program));
    u8* guest_rsp = mem.StackTop() - 8;
    *reinterpret_cast<u64*>(guest_rsp) = kReturnSentinel;
    GuestState st{};
    st.rip = reinterpret_cast<u64>(mem.CodePtr());
    st.gpr[4] = reinterpret_cast<u64>(guest_rsp);
    st.ymm[4] = (0xA1ULL << 32) | 0xA0ULL;
    st.ymm[5] = (0xA3ULL << 32) | 0xA2ULL;
    st.gpr[3] = reinterpret_cast<u64>(slot); // rbx -> m128

    Runtime rt; rt.Run(st);
    EXPECT_EQ(st.ymm[0], 0x000000B0000000A0ULL) << "lane0=src1[0], lane1=src2[0]";
    EXPECT_EQ(st.ymm[1], 0x000000B1000000A1ULL) << "lane2=src1[1], lane3=src2[1]";
    EXPECT_EQ(st.ymm[2], 0ULL);
    EXPECT_EQ(st.ymm[3], 0ULL);
}

// In-place: vunpcklps xmm1, xmm1, xmm2 — dst == src1. Read-before-write keeps
// the interleave correct.
TEST_F(CpuRuntimeTest, Vunpcklps_InPlace_DstEqualsSrc1) {
    const u8 program[] = {0xc5, 0xf0, 0x14, 0xca, 0xc3}; // vunpcklps xmm1,xmm1,xmm2
    std::memcpy(mem.CodePtr(), program, sizeof(program));
    u8* guest_rsp = mem.StackTop() - 8;
    *reinterpret_cast<u64*>(guest_rsp) = kReturnSentinel;
    GuestState st{};
    st.rip = reinterpret_cast<u64>(mem.CodePtr());
    st.gpr[4] = reinterpret_cast<u64>(guest_rsp);
    st.ymm[4] = (0xA1ULL << 32) | 0xA0ULL; // xmm1 = src1 AND dst
    st.ymm[5] = (0xA3ULL << 32) | 0xA2ULL;
    st.ymm[6] = 0xDEAD; st.ymm[7] = 0xBEEF; // upper YMM1, must be zeroed
    st.ymm[8] = (0xB1ULL << 32) | 0xB0ULL;
    st.ymm[9] = (0xB3ULL << 32) | 0xB2ULL;

    Runtime rt; rt.Run(st);
    EXPECT_EQ(st.ymm[4], 0x000000B0000000A0ULL) << "lane0=src1[0], lane1=src2[0]";
    EXPECT_EQ(st.ymm[5], 0x000000B1000000A1ULL) << "lane2=src1[1], lane3=src2[1]";
    EXPECT_EQ(st.ymm[6], 0ULL) << "VEX zeros upper YMM even in-place";
    EXPECT_EQ(st.ymm[7], 0ULL);
}

// ============================================================================
// PUSH with an IMMEDIATE or MEMORY source.
//
// Motivated by CUSA02394 "WE ARE DOOMED", which exited the JIT at a `push imm`
// (6a /ib, length 2) in a loaded system module at guest 0x80866bbfe
// (exit_reason=2). The register form was already supported; the immediate and
// memory forms were not. PUSH imm8/imm32 SIGN-EXTENDS the immediate to 64 bits
// before pushing.
//
// Tests round-trip through `pop rax` (or read the stack slot directly) so the
// pushed value is observable, and check RSP arithmetic.
// ============================================================================

// push imm8 (sign-extended positive) then pop rax. 0x12 -> rax = 0x12.
TEST_F(CpuRuntimeTest, PushImm8_PositiveRoundTrips) {
    // push 0x12 ; pop rax ; ret  = 6a 12 / 58 / c3
    const u8 program[] = {0x6a, 0x12, 0x58, 0xc3};
    std::memcpy(mem.CodePtr(), program, sizeof(program));
    u8* guest_rsp = mem.StackTop() - 16; // room for push + sentinel
    *reinterpret_cast<u64*>(guest_rsp + 8) = kReturnSentinel;

    GuestState st{};
    st.rip = reinterpret_cast<u64>(mem.CodePtr());
    st.gpr[4] = reinterpret_cast<u64>(guest_rsp + 8); // RSP at sentinel
    st.gpr[0] = 0xDEADBEEFDEADBEEFULL;                // rax pre-pollute

    Runtime rt; rt.Run(st);
    EXPECT_EQ(st.gpr[0], 0x12ULL) << "pushed imm8 popped back into rax";
}

// push imm8 (negative) must SIGN-EXTEND to a full 64-bit value.
// push -1 (0x6a 0xff) -> 0xFFFFFFFFFFFFFFFF.
TEST_F(CpuRuntimeTest, PushImm8_NegativeSignExtends) {
    const u8 program[] = {0x6a, 0xff, 0x58, 0xc3}; // push -1 ; pop rax ; ret
    std::memcpy(mem.CodePtr(), program, sizeof(program));
    u8* guest_rsp = mem.StackTop() - 16;
    *reinterpret_cast<u64*>(guest_rsp + 8) = kReturnSentinel;

    GuestState st{};
    st.rip = reinterpret_cast<u64>(mem.CodePtr());
    st.gpr[4] = reinterpret_cast<u64>(guest_rsp + 8);

    Runtime rt; rt.Run(st);
    EXPECT_EQ(st.gpr[0], 0xFFFFFFFFFFFFFFFFULL) << "push imm8 -1 sign-extends to 64 bits";
}

// push imm32 (sign-extended) then pop. 0x12345678 -> rax = 0x12345678.
TEST_F(CpuRuntimeTest, PushImm32_RoundTrips) {
    // push 0x12345678 ; pop rax ; ret = 68 78 56 34 12 / 58 / c3
    const u8 program[] = {0x68, 0x78, 0x56, 0x34, 0x12, 0x58, 0xc3};
    std::memcpy(mem.CodePtr(), program, sizeof(program));
    u8* guest_rsp = mem.StackTop() - 16;
    *reinterpret_cast<u64*>(guest_rsp + 8) = kReturnSentinel;

    GuestState st{};
    st.rip = reinterpret_cast<u64>(mem.CodePtr());
    st.gpr[4] = reinterpret_cast<u64>(guest_rsp + 8);

    Runtime rt; rt.Run(st);
    EXPECT_EQ(st.gpr[0], 0x12345678ULL) << "pushed imm32 popped back";
}

// push imm32 with the sign bit set must sign-extend the 32-bit value.
// push 0x80000000 -> 0xFFFFFFFF80000000.
TEST_F(CpuRuntimeTest, PushImm32_NegativeSignExtends) {
    // push 0x80000000 ; pop rax ; ret = 68 00 00 00 80 / 58 / c3
    const u8 program[] = {0x68, 0x00, 0x00, 0x00, 0x80, 0x58, 0xc3};
    std::memcpy(mem.CodePtr(), program, sizeof(program));
    u8* guest_rsp = mem.StackTop() - 16;
    *reinterpret_cast<u64*>(guest_rsp + 8) = kReturnSentinel;

    GuestState st{};
    st.rip = reinterpret_cast<u64>(mem.CodePtr());
    st.gpr[4] = reinterpret_cast<u64>(guest_rsp + 8);

    Runtime rt; rt.Run(st);
    EXPECT_EQ(st.gpr[0], 0xFFFFFFFF80000000ULL) << "imm32 with sign bit set sign-extends";
}

// push imm decrements RSP by 8 and writes the value at the new top. We push,
// then pop it back into rcx so the stack rebalances and RET pops the real
// sentinel. The pushed slot is captured by reading rcx; RSP is verified to be
// restored. (A bare `push; ret` would make RET pop the pushed value as a
// return address — a test bug, not an emitter one.)
TEST_F(CpuRuntimeTest, PushImm_WritesValueThenRebalances) {
    // push 0x7f ; pop rcx ; ret = 6a 7f / 59 / c3
    const u8 program[] = {0x6a, 0x7f, 0x59, 0xc3};
    std::memcpy(mem.CodePtr(), program, sizeof(program));
    u8* guest_rsp = mem.StackTop() - 16;
    *reinterpret_cast<u64*>(guest_rsp + 8) = kReturnSentinel;

    GuestState st{};
    st.rip = reinterpret_cast<u64>(mem.CodePtr());
    const u64 rsp_in = reinterpret_cast<u64>(guest_rsp + 8);
    st.gpr[4] = rsp_in;

    Runtime rt; rt.Run(st);
    EXPECT_EQ(st.gpr[1], 0x7fULL) << "pushed value observed via pop rcx";
    // The push wrote 0x7f to (rsp_in - 8); confirm the slot still holds it.
    EXPECT_EQ(*reinterpret_cast<u64*>(rsp_in - 8), 0x7fULL) << "value written at pushed slot";
}

// push qword[rbx] — memory source. Pushes the 8 bytes at [rbx], then pop rax.
TEST_F(CpuRuntimeTest, PushMem64_RoundTrips) {
    u64* slot = reinterpret_cast<u64*>(mem.CodePtr() + 0x100);
    *slot = 0xCAFEF00DBAADF00DULL;

    // push qword[rbx] ; pop rax ; ret = ff 33 / 58 / c3
    const u8 program[] = {0xff, 0x33, 0x58, 0xc3};
    std::memcpy(mem.CodePtr(), program, sizeof(program));
    u8* guest_rsp = mem.StackTop() - 16;
    *reinterpret_cast<u64*>(guest_rsp + 8) = kReturnSentinel;

    GuestState st{};
    st.rip = reinterpret_cast<u64>(mem.CodePtr());
    st.gpr[4] = reinterpret_cast<u64>(guest_rsp + 8);
    st.gpr[3] = reinterpret_cast<u64>(slot); // rbx -> m64

    Runtime rt; rt.Run(st);
    EXPECT_EQ(st.gpr[0], 0xCAFEF00DBAADF00DULL) << "pushed memory qword popped back";
}

// Regression guard: the register-source PUSH must still work.
// push rdi ; pop rax ; ret.
TEST_F(CpuRuntimeTest, PushReg_StillWorksAfterImmAdded) {
    const u8 program[] = {0x57, 0x58, 0xc3}; // push rdi ; pop rax ; ret
    std::memcpy(mem.CodePtr(), program, sizeof(program));
    u8* guest_rsp = mem.StackTop() - 16;
    *reinterpret_cast<u64*>(guest_rsp + 8) = kReturnSentinel;

    GuestState st{};
    st.rip = reinterpret_cast<u64>(mem.CodePtr());
    st.gpr[4] = reinterpret_cast<u64>(guest_rsp + 8);
    st.gpr[7] = 0x1122'3344'5566'7788ULL; // rdi

    Runtime rt; rt.Run(st);
    EXPECT_EQ(st.gpr[0], 0x1122'3344'5566'7788ULL) << "register push still round-trips";
}

// ============================================================================
// OR byte[mem], r8 — 8-bit OR with a memory destination and register source.
//
// Motivated by CUSA02394 "WE ARE DOOMED", which exited the JIT at an
// `or byte[mem], r8` (length 6, disp32 destination) in a loaded system module
// at guest 0x80866ae31 (exit_reason=2). The byte-immediate memory form was
// already supported; the byte-register memory form was not. OR clears CF/OF
// and sets SF/ZF/PF from the 8-bit result (SF from bit 7, not bit 63).
//
// rflags bits: CF=0x1, PF=0x4, ZF=0x40, SF=0x80, OF=0x800.
// ============================================================================

// or byte[rbx], cl — basic OR into a memory byte.
TEST_F(CpuRuntimeTest, Or8_MemDstRegSrc_OrsByte) {
    u8* slot = mem.CodePtr() + 0x100;
    *slot = 0x0F;
    *(slot + 1) = 0x55; // sentinel: 8-bit op must not touch the next byte

    const u8 program[] = {0x08, 0x0b, 0xc3}; // or byte[rbx], cl ; ret
    std::memcpy(mem.CodePtr(), program, sizeof(program));
    u8* guest_rsp = mem.StackTop() - 8;
    *reinterpret_cast<u64*>(guest_rsp) = kReturnSentinel;
    GuestState st{};
    st.rip = reinterpret_cast<u64>(mem.CodePtr());
    st.gpr[4] = reinterpret_cast<u64>(guest_rsp);
    st.gpr[3] = reinterpret_cast<u64>(slot); // rbx -> m8
    st.gpr[1] = 0xFFFFFF00ULL | 0xF0ULL;     // cl = 0xF0 (upper bits ignored)

    Runtime rt; rt.Run(st);
    EXPECT_EQ(*slot, 0xFFu) << "0x0F | 0xF0 = 0xFF";
    EXPECT_EQ(*(slot + 1), 0x55u) << "8-bit op must not write the next byte";
}

// or byte[rbx+disp32], cl — length-6 disp32 form (the crashing addressing mode).
TEST_F(CpuRuntimeTest, Or8_MemDstDisp32_OrsByte) {
    u8* base = mem.CodePtr() + 0x100;
    u8* slot = mem.CodePtr() + 0x100 + 0x40;
    *slot = 0x01;

    // or byte[rbx+0x40], cl  = 08 4b 40   (disp8 here; semantics identical to
    // the disp32 form the game used)
    const u8 program[] = {0x08, 0x4b, 0x40, 0xc3};
    std::memcpy(mem.CodePtr(), program, sizeof(program));
    u8* guest_rsp = mem.StackTop() - 8;
    *reinterpret_cast<u64*>(guest_rsp) = kReturnSentinel;
    GuestState st{};
    st.rip = reinterpret_cast<u64>(mem.CodePtr());
    st.gpr[4] = reinterpret_cast<u64>(guest_rsp);
    st.gpr[3] = reinterpret_cast<u64>(base); // rbx; +0x40 -> slot
    st.gpr[1] = 0x80ULL;                     // cl = 0x80

    Runtime rt; rt.Run(st);
    EXPECT_EQ(*slot, 0x81u) << "0x01 | 0x80 = 0x81 at [rbx+0x40]";
}

// Flags: OR producing zero sets ZF, clears SF/CF/OF.
TEST_F(CpuRuntimeTest, Or8_MemDst_ZeroResult_SetsZf) {
    u8* slot = mem.CodePtr() + 0x100;
    *slot = 0x00;

    const u8 program[] = {0x08, 0x0b, 0xc3};
    std::memcpy(mem.CodePtr(), program, sizeof(program));
    u8* guest_rsp = mem.StackTop() - 8;
    *reinterpret_cast<u64*>(guest_rsp) = kReturnSentinel;
    GuestState st{};
    st.rip = reinterpret_cast<u64>(mem.CodePtr());
    st.gpr[4] = reinterpret_cast<u64>(guest_rsp);
    st.gpr[3] = reinterpret_cast<u64>(slot);
    st.gpr[1] = 0x00ULL; // cl = 0 -> 0|0 = 0

    Runtime rt; rt.Run(st);
    EXPECT_EQ(*slot, 0x00u);
    EXPECT_EQ(st.rflags & 0x40u, 0x40u) << "ZF set (result zero)";
    EXPECT_EQ(st.rflags & 0x80u, 0u) << "SF clear";
    EXPECT_EQ(st.rflags & 0x1u, 0u) << "CF cleared by OR";
    EXPECT_EQ(st.rflags & 0x800u, 0u) << "OF cleared by OR";
}

// Flags: a result with bit 7 set must set SF (derived from bit 7, not bit 63).
TEST_F(CpuRuntimeTest, Or8_MemDst_HighBitResult_SetsSf) {
    u8* slot = mem.CodePtr() + 0x100;
    *slot = 0x01;

    const u8 program[] = {0x08, 0x0b, 0xc3};
    std::memcpy(mem.CodePtr(), program, sizeof(program));
    u8* guest_rsp = mem.StackTop() - 8;
    *reinterpret_cast<u64*>(guest_rsp) = kReturnSentinel;
    GuestState st{};
    st.rip = reinterpret_cast<u64>(mem.CodePtr());
    st.gpr[4] = reinterpret_cast<u64>(guest_rsp);
    st.gpr[3] = reinterpret_cast<u64>(slot);
    st.gpr[1] = 0x80ULL; // 0x01 | 0x80 = 0x81 (bit7 set)

    Runtime rt; rt.Run(st);
    EXPECT_EQ(*slot, 0x81u);
    EXPECT_EQ(st.rflags & 0x80u, 0x80u) << "SF set from bit 7 of the 8-bit result";
    EXPECT_EQ(st.rflags & 0x40u, 0u) << "ZF clear";
}

// Regression guard: the byte-IMMEDIATE memory form must still work.
// or byte[rbx], 0x0f = 80 0b 0f
TEST_F(CpuRuntimeTest, Or8_MemDstImm_StillWorks) {
    u8* slot = mem.CodePtr() + 0x100;
    *slot = 0xF0;

    const u8 program[] = {0x80, 0x0b, 0x0f, 0xc3};
    std::memcpy(mem.CodePtr(), program, sizeof(program));
    u8* guest_rsp = mem.StackTop() - 8;
    *reinterpret_cast<u64*>(guest_rsp) = kReturnSentinel;
    GuestState st{};
    st.rip = reinterpret_cast<u64>(mem.CodePtr());
    st.gpr[4] = reinterpret_cast<u64>(guest_rsp);
    st.gpr[3] = reinterpret_cast<u64>(slot);

    Runtime rt; rt.Run(st);
    EXPECT_EQ(*slot, 0xFFu) << "0xF0 | 0x0F = 0xFF (imm form unaffected)";
}

// ============================================================================
// VPUNPCKHQDQ — unpack/interleave the HIGH 64-bit halves of two vectors.
//
// Motivated by CUSA02394 "WE ARE DOOMED", which exited the JIT at a reg,reg
// 128-bit `vpunpckhqdq` (2-byte VEX, length 4) in the eboot at guest
// 0x80023f813 (exit_reason=2), near the earlier VSHUFPS/VUNPCKLPS vector block.
// Per 128-bit lane: dst.q0 = src1.q1, dst.q1 = src2.q1 (the integer-domain
// sibling of VUNPCKHPD). The low variant VPUNPCKLQDQ was already supported and
// is now served by the same generalized emitter.
//
// YMM layout: ymm[k*4+0]=XMMk qword0, ymm[k*4+1]=qword1.
// XMM0->ymm[0..], XMM1->ymm[4..], XMM2->ymm[8..].
// ============================================================================

// vpunpckhqdq xmm0, xmm1, xmm2 — the crashing form.
TEST_F(CpuRuntimeTest, Vpunpckhqdq_InterleavesHighQuadwords) {
    const u8 program[] = {0xc5, 0xf1, 0x6d, 0xc2, 0xc3};
    std::memcpy(mem.CodePtr(), program, sizeof(program));
    u8* guest_rsp = mem.StackTop() - 8;
    *reinterpret_cast<u64*>(guest_rsp) = kReturnSentinel;
    GuestState st{};
    st.rip = reinterpret_cast<u64>(mem.CodePtr());
    st.gpr[4] = reinterpret_cast<u64>(guest_rsp);
    st.ymm[4] = 0xA0A0A0A0A0A0A0A0ULL; // src1 q0
    st.ymm[5] = 0xA1A1A1A1A1A1A1A1ULL; // src1 q1
    st.ymm[8] = 0xB0B0B0B0B0B0B0B0ULL; // src2 q0
    st.ymm[9] = 0xB1B1B1B1B1B1B1B1ULL; // src2 q1
    st.ymm[0] = 0xDEAD; st.ymm[1] = 0xBEEF;
    st.ymm[2] = 0xCAFE; st.ymm[3] = 0xF00D;

    Runtime rt; rt.Run(st);
    EXPECT_EQ(st.ymm[0], 0xA1A1A1A1A1A1A1A1ULL) << "dst.q0 = src1.q1";
    EXPECT_EQ(st.ymm[1], 0xB1B1B1B1B1B1B1B1ULL) << "dst.q1 = src2.q1";
    EXPECT_EQ(st.ymm[2], 0ULL) << "VEX zeros upper YMM";
    EXPECT_EQ(st.ymm[3], 0ULL);
}

// Memory source: vpunpckhqdq xmm0, xmm1, [rbx].
TEST_F(CpuRuntimeTest, Vpunpckhqdq_MemorySource) {
    u64* slot = reinterpret_cast<u64*>(mem.CodePtr() + 0x100);
    slot[0] = 0xB0B0B0B0B0B0B0B0ULL; // q0
    slot[1] = 0xB1B1B1B1B1B1B1B1ULL; // q1

    const u8 program[] = {0xc5, 0xf1, 0x6d, 0x03, 0xc3};
    std::memcpy(mem.CodePtr(), program, sizeof(program));
    u8* guest_rsp = mem.StackTop() - 8;
    *reinterpret_cast<u64*>(guest_rsp) = kReturnSentinel;
    GuestState st{};
    st.rip = reinterpret_cast<u64>(mem.CodePtr());
    st.gpr[4] = reinterpret_cast<u64>(guest_rsp);
    st.ymm[4] = 0xA0A0A0A0A0A0A0A0ULL;
    st.ymm[5] = 0xA1A1A1A1A1A1A1A1ULL;
    st.gpr[3] = reinterpret_cast<u64>(slot); // rbx -> m128

    Runtime rt; rt.Run(st);
    EXPECT_EQ(st.ymm[0], 0xA1A1A1A1A1A1A1A1ULL) << "dst.q0 = src1.q1";
    EXPECT_EQ(st.ymm[1], 0xB1B1B1B1B1B1B1B1ULL) << "dst.q1 = mem.q1";
    EXPECT_EQ(st.ymm[2], 0ULL);
    EXPECT_EQ(st.ymm[3], 0ULL);
}

// In-place: vpunpckhqdq xmm1, xmm1, xmm2 — dst == src1. Read-before-write keeps
// the interleave correct.
TEST_F(CpuRuntimeTest, Vpunpckhqdq_InPlace_DstEqualsSrc1) {
    const u8 program[] = {0xc5, 0xf1, 0x6d, 0xca, 0xc3}; // vpunpckhqdq xmm1,xmm1,xmm2
    std::memcpy(mem.CodePtr(), program, sizeof(program));
    u8* guest_rsp = mem.StackTop() - 8;
    *reinterpret_cast<u64*>(guest_rsp) = kReturnSentinel;
    GuestState st{};
    st.rip = reinterpret_cast<u64>(mem.CodePtr());
    st.gpr[4] = reinterpret_cast<u64>(guest_rsp);
    st.ymm[4] = 0xA0A0A0A0A0A0A0A0ULL; // xmm1 = src1 AND dst
    st.ymm[5] = 0xA1A1A1A1A1A1A1A1ULL;
    st.ymm[6] = 0xDEAD; st.ymm[7] = 0xBEEF; // upper YMM1, must be zeroed
    st.ymm[8] = 0xB0B0B0B0B0B0B0B0ULL;
    st.ymm[9] = 0xB1B1B1B1B1B1B1B1ULL;

    Runtime rt; rt.Run(st);
    EXPECT_EQ(st.ymm[4], 0xA1A1A1A1A1A1A1A1ULL) << "dst.q0 = src1.q1";
    EXPECT_EQ(st.ymm[5], 0xB1B1B1B1B1B1B1B1ULL) << "dst.q1 = src2.q1";
    EXPECT_EQ(st.ymm[6], 0ULL) << "VEX zeros upper YMM even in-place";
    EXPECT_EQ(st.ymm[7], 0ULL);
}

// 256-bit form: vpunpckhqdq ymm0, ymm1, ymm2 — interleave high qwords PER
// 128-bit lane (no cross-lane movement). Encoding c5 f5 6d c2.
TEST_F(CpuRuntimeTest, Vpunpckhqdq_256_PerLaneHighQwords) {
    const u8 program[] = {0xc5, 0xf5, 0x6d, 0xc2, 0xc3};
    std::memcpy(mem.CodePtr(), program, sizeof(program));
    u8* guest_rsp = mem.StackTop() - 8;
    *reinterpret_cast<u64*>(guest_rsp) = kReturnSentinel;
    GuestState st{};
    st.rip = reinterpret_cast<u64>(mem.CodePtr());
    st.gpr[4] = reinterpret_cast<u64>(guest_rsp);
    // ymm1 (src1): lane0 q0,q1 = A0,A1 ; lane1 q0,q1 = A2,A3
    st.ymm[4] = 0xA0ULL; st.ymm[5] = 0xA1ULL; st.ymm[6] = 0xA2ULL; st.ymm[7] = 0xA3ULL;
    // ymm2 (src2): lane0 q0,q1 = B0,B1 ; lane1 q0,q1 = B2,B3
    st.ymm[8] = 0xB0ULL; st.ymm[9] = 0xB1ULL; st.ymm[10] = 0xB2ULL; st.ymm[11] = 0xB3ULL;

    Runtime rt; rt.Run(st);
    // lane0: dst.q0=src1.q1=A1, dst.q1=src2.q1=B1
    EXPECT_EQ(st.ymm[0], 0xA1ULL) << "lane0 dst.q0 = src1.q1";
    EXPECT_EQ(st.ymm[1], 0xB1ULL) << "lane0 dst.q1 = src2.q1";
    // lane1: dst.q0=src1.q3(=A3), dst.q1=src2.q3(=B3)
    EXPECT_EQ(st.ymm[2], 0xA3ULL) << "lane1 dst.q0 = src1 high-lane q1";
    EXPECT_EQ(st.ymm[3], 0xB3ULL) << "lane1 dst.q1 = src2 high-lane q1";
}

// ============================================================================
// VPUNPCKLDQ / VPUNPCKHDQ — dword-granularity unpack/interleave, the 32-bit
// siblings of the qword pair above, served by the same generalized emitter.
//
// Motivated by CUSA02394 "WE ARE DOOMED", which exited the JIT at a reg,reg
// 128-bit `vpunpckldq` (2-byte VEX, length 4) in the eboot at guest
// 0x80023f6bf (exit_reason=2), in the same vector block as the VPUNPCKHQDQ /
// VSHUFPS / VUNPCKLPS shuffles. Per 128-bit lane:
//   LDQ: dst = {s1.d0, s2.d0, s1.d1, s2.d1}
//   HDQ: dst = {s1.d2, s2.d2, s1.d3, s2.d3}
// Dwords pack two-per-u64: ymm[k*4+0] = {d0(low), d1(high)},
// ymm[k*4+1] = {d2(low), d3(high)}, little-endian.
// ============================================================================

// vpunpckldq xmm0, xmm1, xmm2 — the crashing form. Interleave low dwords.
TEST_F(CpuRuntimeTest, Vpunpckldq_InterleavesLowDwords) {
    const u8 program[] = {0xc5, 0xf1, 0x62, 0xc2, 0xc3};
    std::memcpy(mem.CodePtr(), program, sizeof(program));
    u8* guest_rsp = mem.StackTop() - 8;
    *reinterpret_cast<u64*>(guest_rsp) = kReturnSentinel;
    GuestState st{};
    st.rip = reinterpret_cast<u64>(mem.CodePtr());
    st.gpr[4] = reinterpret_cast<u64>(guest_rsp);
    st.ymm[4] = 0x1111111100000000ULL; // src1 q0: d0=0, d1=0x11111111
    st.ymm[5] = 0x3333333322222222ULL; // src1 q1: d2=0x22222222, d3=0x33333333
    st.ymm[8] = 0x5555555544444444ULL; // src2 q0: d0=0x44444444, d1=0x55555555
    st.ymm[9] = 0x7777777766666666ULL; // src2 q1
    st.ymm[0] = 0xDEAD; st.ymm[1] = 0xBEEF;
    st.ymm[2] = 0xCAFE; st.ymm[3] = 0xF00D;

    Runtime rt; rt.Run(st);
    EXPECT_EQ(st.ymm[0], 0x4444444400000000ULL) << "dst.q0 = {s1.d0, s2.d0}";
    EXPECT_EQ(st.ymm[1], 0x5555555511111111ULL) << "dst.q1 = {s1.d1, s2.d1}";
    EXPECT_EQ(st.ymm[2], 0ULL) << "VEX zeros upper YMM";
    EXPECT_EQ(st.ymm[3], 0ULL);
}

// vpunpckhdq xmm0, xmm1, xmm2 — interleave high dwords.
TEST_F(CpuRuntimeTest, Vpunpckhdq_InterleavesHighDwords) {
    const u8 program[] = {0xc5, 0xf1, 0x6a, 0xc2, 0xc3};
    std::memcpy(mem.CodePtr(), program, sizeof(program));
    u8* guest_rsp = mem.StackTop() - 8;
    *reinterpret_cast<u64*>(guest_rsp) = kReturnSentinel;
    GuestState st{};
    st.rip = reinterpret_cast<u64>(mem.CodePtr());
    st.gpr[4] = reinterpret_cast<u64>(guest_rsp);
    st.ymm[4] = 0x1111111100000000ULL;
    st.ymm[5] = 0x3333333322222222ULL;
    st.ymm[8] = 0x5555555544444444ULL;
    st.ymm[9] = 0x7777777766666666ULL;

    Runtime rt; rt.Run(st);
    EXPECT_EQ(st.ymm[0], 0x6666666622222222ULL) << "dst.q0 = {s1.d2, s2.d2}";
    EXPECT_EQ(st.ymm[1], 0x7777777733333333ULL) << "dst.q1 = {s1.d3, s2.d3}";
    EXPECT_EQ(st.ymm[2], 0ULL);
    EXPECT_EQ(st.ymm[3], 0ULL);
}

// Memory source: vpunpckldq xmm0, xmm1, [rbx].
TEST_F(CpuRuntimeTest, Vpunpckldq_MemorySource) {
    u64* slot = reinterpret_cast<u64*>(mem.CodePtr() + 0x100);
    slot[0] = 0x5555555544444444ULL; // src2 q0
    slot[1] = 0x7777777766666666ULL; // src2 q1

    const u8 program[] = {0xc5, 0xf1, 0x62, 0x03, 0xc3};
    std::memcpy(mem.CodePtr(), program, sizeof(program));
    u8* guest_rsp = mem.StackTop() - 8;
    *reinterpret_cast<u64*>(guest_rsp) = kReturnSentinel;
    GuestState st{};
    st.rip = reinterpret_cast<u64>(mem.CodePtr());
    st.gpr[4] = reinterpret_cast<u64>(guest_rsp);
    st.ymm[4] = 0x1111111100000000ULL;
    st.ymm[5] = 0x3333333322222222ULL;
    st.gpr[3] = reinterpret_cast<u64>(slot); // rbx -> m128

    Runtime rt; rt.Run(st);
    EXPECT_EQ(st.ymm[0], 0x4444444400000000ULL) << "dst.q0 = {s1.d0, mem.d0}";
    EXPECT_EQ(st.ymm[1], 0x5555555511111111ULL) << "dst.q1 = {s1.d1, mem.d1}";
    EXPECT_EQ(st.ymm[2], 0ULL);
    EXPECT_EQ(st.ymm[3], 0ULL);
}

// In-place: vpunpckldq xmm1, xmm1, xmm2 — dst == src1.
TEST_F(CpuRuntimeTest, Vpunpckldq_InPlace_DstEqualsSrc1) {
    const u8 program[] = {0xc5, 0xf1, 0x62, 0xca, 0xc3}; // vpunpckldq xmm1,xmm1,xmm2
    std::memcpy(mem.CodePtr(), program, sizeof(program));
    u8* guest_rsp = mem.StackTop() - 8;
    *reinterpret_cast<u64*>(guest_rsp) = kReturnSentinel;
    GuestState st{};
    st.rip = reinterpret_cast<u64>(mem.CodePtr());
    st.gpr[4] = reinterpret_cast<u64>(guest_rsp);
    st.ymm[4] = 0x1111111100000000ULL; // xmm1 = src1 AND dst
    st.ymm[5] = 0x3333333322222222ULL;
    st.ymm[6] = 0xDEAD; st.ymm[7] = 0xBEEF; // upper YMM1, must be zeroed
    st.ymm[8] = 0x5555555544444444ULL;
    st.ymm[9] = 0x7777777766666666ULL;

    Runtime rt; rt.Run(st);
    EXPECT_EQ(st.ymm[4], 0x4444444400000000ULL) << "dst.q0 = {s1.d0, s2.d0}";
    EXPECT_EQ(st.ymm[5], 0x5555555511111111ULL) << "dst.q1 = {s1.d1, s2.d1}";
    EXPECT_EQ(st.ymm[6], 0ULL) << "VEX zeros upper YMM even in-place";
    EXPECT_EQ(st.ymm[7], 0ULL);
}

// 256-bit: vpunpckldq ymm0, ymm1, ymm2 — interleave low dwords PER 128-bit lane
// (no cross-lane movement). Encoding c5 f5 62 c2.
TEST_F(CpuRuntimeTest, Vpunpckldq_256_PerLaneLowDwords) {
    const u8 program[] = {0xc5, 0xf5, 0x62, 0xc2, 0xc3};
    std::memcpy(mem.CodePtr(), program, sizeof(program));
    u8* guest_rsp = mem.StackTop() - 8;
    *reinterpret_cast<u64*>(guest_rsp) = kReturnSentinel;
    GuestState st{};
    st.rip = reinterpret_cast<u64>(mem.CodePtr());
    st.gpr[4] = reinterpret_cast<u64>(guest_rsp);
    // ymm1 (src1): lane0 q0,q1 ; lane1 q0,q1
    st.ymm[4] = 0x1111111100000000ULL; st.ymm[5] = 0x3333333322222222ULL;
    st.ymm[6] = 0x9999999988888888ULL; st.ymm[7] = 0xBBBBBBBBAAAAAAAAULL;
    // ymm2 (src2)
    st.ymm[8]  = 0x5555555544444444ULL; st.ymm[9]  = 0x7777777766666666ULL;
    st.ymm[10] = 0xDDDDDDDDCCCCCCCCULL; st.ymm[11] = 0xFFFFFFFFEEEEEEEEULL;

    Runtime rt; rt.Run(st);
    // lane0: {s1.d0,s2.d0,s1.d1,s2.d1}
    EXPECT_EQ(st.ymm[0], 0x4444444400000000ULL) << "lane0 dst.q0";
    EXPECT_EQ(st.ymm[1], 0x5555555511111111ULL) << "lane0 dst.q1";
    // lane1: low dwords of the high lane: s1.d0'=0x88888888, s2.d0'=0xCCCCCCCC, s1.d1'=0x99999999, s2.d1'=0xDDDDDDDD
    EXPECT_EQ(st.ymm[2], 0xCCCCCCCC88888888ULL) << "lane1 dst.q0";
    EXPECT_EQ(st.ymm[3], 0xDDDDDDDD99999999ULL) << "lane1 dst.q1";
}

// ============================================================================
// VPUNPCKLWD / VPUNPCKHWD — word-granularity unpack/interleave (16-bit
// elements), the same generalized emitter at word width.
//
// VPUNPCKLWD was the gap at guest 0x80023f6cf, 16 bytes after the VPUNPCKLDQ
// in the same vector block (a sequence of progressively narrower unpacks —
// classic byte/word/dword/qword interleave-widening idiom). Per 128-bit lane:
//   LWD: dst = {s1.w0,s2.w0,s1.w1,s2.w1,s1.w2,s2.w2,s1.w3,s2.w3}
//   HWD: dst = {s1.w4,s2.w4,...,s1.w7,s2.w7}
// Words pack four-per-u64, little-endian: ymm[k*4+0] holds w0..w3,
// ymm[k*4+1] holds w4..w7.
// ============================================================================

// vpunpcklwd xmm0, xmm1, xmm2 — the crashing form. Interleave low words.
TEST_F(CpuRuntimeTest, Vpunpcklwd_InterleavesLowWords) {
    const u8 program[] = {0xc5, 0xf1, 0x61, 0xc2, 0xc3};
    std::memcpy(mem.CodePtr(), program, sizeof(program));
    u8* guest_rsp = mem.StackTop() - 8;
    *reinterpret_cast<u64*>(guest_rsp) = kReturnSentinel;
    GuestState st{};
    st.rip = reinterpret_cast<u64>(mem.CodePtr());
    st.gpr[4] = reinterpret_cast<u64>(guest_rsp);
    st.ymm[4] = 0x3333222211110000ULL; // src1 words 0..3
    st.ymm[5] = 0x7777666655554444ULL; // src1 words 4..7
    st.ymm[8] = 0xBBBBAAAA99998888ULL; // src2 words 0..3
    st.ymm[9] = 0xFFFFEEEEDDDDCCCCULL; // src2 words 4..7
    st.ymm[0] = 0xDEAD; st.ymm[1] = 0xBEEF;
    st.ymm[2] = 0xCAFE; st.ymm[3] = 0xF00D;

    Runtime rt; rt.Run(st);
    EXPECT_EQ(st.ymm[0], 0x9999111188880000ULL) << "dst q0 = {s1w0,s2w0,s1w1,s2w1}";
    EXPECT_EQ(st.ymm[1], 0xBBBB3333AAAA2222ULL) << "dst q1 = {s1w2,s2w2,s1w3,s2w3}";
    EXPECT_EQ(st.ymm[2], 0ULL) << "VEX zeros upper YMM";
    EXPECT_EQ(st.ymm[3], 0ULL);
}

// vpunpckhwd xmm0, xmm1, xmm2 — interleave high words.
TEST_F(CpuRuntimeTest, Vpunpckhwd_InterleavesHighWords) {
    const u8 program[] = {0xc5, 0xf1, 0x69, 0xc2, 0xc3};
    std::memcpy(mem.CodePtr(), program, sizeof(program));
    u8* guest_rsp = mem.StackTop() - 8;
    *reinterpret_cast<u64*>(guest_rsp) = kReturnSentinel;
    GuestState st{};
    st.rip = reinterpret_cast<u64>(mem.CodePtr());
    st.gpr[4] = reinterpret_cast<u64>(guest_rsp);
    st.ymm[4] = 0x3333222211110000ULL;
    st.ymm[5] = 0x7777666655554444ULL;
    st.ymm[8] = 0xBBBBAAAA99998888ULL;
    st.ymm[9] = 0xFFFFEEEEDDDDCCCCULL;

    Runtime rt; rt.Run(st);
    EXPECT_EQ(st.ymm[0], 0xDDDD5555CCCC4444ULL) << "dst q0 = {s1w4,s2w4,s1w5,s2w5}";
    EXPECT_EQ(st.ymm[1], 0xFFFF7777EEEE6666ULL) << "dst q1 = {s1w6,s2w6,s1w7,s2w7}";
    EXPECT_EQ(st.ymm[2], 0ULL);
    EXPECT_EQ(st.ymm[3], 0ULL);
}

// Memory source: vpunpcklwd xmm0, xmm1, [rbx].
TEST_F(CpuRuntimeTest, Vpunpcklwd_MemorySource) {
    u64* slot = reinterpret_cast<u64*>(mem.CodePtr() + 0x100);
    slot[0] = 0xBBBBAAAA99998888ULL; // src2 words 0..3
    slot[1] = 0xFFFFEEEEDDDDCCCCULL; // src2 words 4..7

    const u8 program[] = {0xc5, 0xf1, 0x61, 0x03, 0xc3};
    std::memcpy(mem.CodePtr(), program, sizeof(program));
    u8* guest_rsp = mem.StackTop() - 8;
    *reinterpret_cast<u64*>(guest_rsp) = kReturnSentinel;
    GuestState st{};
    st.rip = reinterpret_cast<u64>(mem.CodePtr());
    st.gpr[4] = reinterpret_cast<u64>(guest_rsp);
    st.ymm[4] = 0x3333222211110000ULL;
    st.ymm[5] = 0x7777666655554444ULL;
    st.gpr[3] = reinterpret_cast<u64>(slot); // rbx -> m128

    Runtime rt; rt.Run(st);
    EXPECT_EQ(st.ymm[0], 0x9999111188880000ULL) << "dst q0 = {s1w0,mem.w0,s1w1,mem.w1}";
    EXPECT_EQ(st.ymm[1], 0xBBBB3333AAAA2222ULL) << "dst q1 = {s1w2,mem.w2,s1w3,mem.w3}";
    EXPECT_EQ(st.ymm[2], 0ULL);
    EXPECT_EQ(st.ymm[3], 0ULL);
}

// ============================================================================
// VCMPSS — scalar single-precision compare producing a low-dword mask.
//
// Motivated by CUSA02394 "WE ARE DOOMED", which exited the JIT at a reg,reg
// `vcmpss` (2-byte VEX, length 5) in the eboot at guest 0x800105b28
// (exit_reason=2). Compares only the low float of src1 vs src2 by the imm8
// predicate, writing 0xFFFFFFFF (true) or 0x00000000 (false) to the dst low
// dword. The SCALAR QUIRK: bits 127:32 are copied from src1 unchanged (unlike
// VCMPPS, which writes every lane). VEX zeroes bits 255:128.
//
// imm8: 0=EQ 1=LT 2=LE 3=UNORD 4=NEQ 5=NLT 6=NLE 7=ORD (+ AVX 8..31).
// f(1.0)=0x3f800000 f(2.0)=0x40000000 f(3.0)=0x40400000.
// ============================================================================

// vcmpltss xmm0, xmm1, xmm2 (imm 1): 1.0 < 2.0 is true -> low dword 0xFFFFFFFF.
TEST_F(CpuRuntimeTest, Vcmpss_LessThan_True_SetsMask) {
    const u8 program[] = {0xc5, 0xf2, 0xc2, 0xc2, 0x01, 0xc3};
    std::memcpy(mem.CodePtr(), program, sizeof(program));
    u8* guest_rsp = mem.StackTop() - 8;
    *reinterpret_cast<u64*>(guest_rsp) = kReturnSentinel;
    GuestState st{};
    st.rip = reinterpret_cast<u64>(mem.CodePtr());
    st.gpr[4] = reinterpret_cast<u64>(guest_rsp);
    st.ymm[4] = 0x3f800000ULL; // src1 low = 1.0f
    st.ymm[8] = 0x40000000ULL; // src2 low = 2.0f
    st.ymm[0] = 0xDEAD; st.ymm[1] = 0xBEEF; st.ymm[2] = 0xCAFE; st.ymm[3] = 0xF00D;

    Runtime rt; rt.Run(st);
    EXPECT_EQ(st.ymm[0] & 0xFFFFFFFFULL, 0xFFFFFFFFULL) << "1.0 < 2.0 true -> all-ones mask";
}

// vcmpeqss xmm0, xmm1, xmm2 (imm 0): 1.0 == 2.0 is false -> low dword 0.
TEST_F(CpuRuntimeTest, Vcmpss_Equal_False_ClearsMask) {
    const u8 program[] = {0xc5, 0xf2, 0xc2, 0xc2, 0x00, 0xc3};
    std::memcpy(mem.CodePtr(), program, sizeof(program));
    u8* guest_rsp = mem.StackTop() - 8;
    *reinterpret_cast<u64*>(guest_rsp) = kReturnSentinel;
    GuestState st{};
    st.rip = reinterpret_cast<u64>(mem.CodePtr());
    st.gpr[4] = reinterpret_cast<u64>(guest_rsp);
    st.ymm[4] = 0x3f800000ULL; // 1.0f
    st.ymm[8] = 0x40000000ULL; // 2.0f

    Runtime rt; rt.Run(st);
    EXPECT_EQ(st.ymm[0] & 0xFFFFFFFFULL, 0x00000000ULL) << "1.0 == 2.0 false -> zero mask";
}

// vcmpeqss with equal operands: 3.0 == 3.0 is true -> 0xFFFFFFFF.
TEST_F(CpuRuntimeTest, Vcmpss_Equal_True_SetsMask) {
    const u8 program[] = {0xc5, 0xf2, 0xc2, 0xc2, 0x00, 0xc3};
    std::memcpy(mem.CodePtr(), program, sizeof(program));
    u8* guest_rsp = mem.StackTop() - 8;
    *reinterpret_cast<u64*>(guest_rsp) = kReturnSentinel;
    GuestState st{};
    st.rip = reinterpret_cast<u64>(mem.CodePtr());
    st.gpr[4] = reinterpret_cast<u64>(guest_rsp);
    st.ymm[4] = 0x40400000ULL; // 3.0f
    st.ymm[8] = 0x40400000ULL; // 3.0f

    Runtime rt; rt.Run(st);
    EXPECT_EQ(st.ymm[0] & 0xFFFFFFFFULL, 0xFFFFFFFFULL) << "3.0 == 3.0 true -> all-ones mask";
}

// SCALAR QUIRK: bits 127:32 of the destination come from src1 unchanged, and
// bits 255:128 are zeroed. dst == src1 here so we also exercise aliasing.
TEST_F(CpuRuntimeTest, Vcmpss_PreservesSrc1UpperBits) {
    // vcmpltss xmm1, xmm1, xmm2, 1  (dst==src1)  = c5 f2 c2 ca 01
    const u8 program[] = {0xc5, 0xf2, 0xc2, 0xca, 0x01, 0xc3};
    std::memcpy(mem.CodePtr(), program, sizeof(program));
    u8* guest_rsp = mem.StackTop() - 8;
    *reinterpret_cast<u64*>(guest_rsp) = kReturnSentinel;
    GuestState st{};
    st.rip = reinterpret_cast<u64>(mem.CodePtr());
    st.gpr[4] = reinterpret_cast<u64>(guest_rsp);
    // src1 (xmm1): low=1.0f, dwords 1..3 carry sentinels 0xAA.., 0xBB.., 0xCC..
    st.ymm[4] = (0xAAAAAAAAULL << 32) | 0x3f800000ULL; // dword0=1.0f, dword1=0xAAAAAAAA
    st.ymm[5] = (0xCCCCCCCCULL << 32) | 0xBBBBBBBBULL; // dword2=0xBBBBBBBB, dword3=0xCCCCCCCC
    st.ymm[6] = 0xDEAD; st.ymm[7] = 0xBEEF;            // upper YMM, must be zeroed
    st.ymm[8] = 0x40000000ULL;                         // src2 low = 2.0f

    Runtime rt; rt.Run(st);
    EXPECT_EQ(st.ymm[4] & 0xFFFFFFFFULL, 0xFFFFFFFFULL) << "low dword = mask (1.0<2.0 true)";
    EXPECT_EQ(st.ymm[4] >> 32, 0xAAAAAAAAULL) << "dword1 preserved from src1";
    EXPECT_EQ(st.ymm[5], (0xCCCCCCCCULL << 32) | 0xBBBBBBBBULL) << "dwords2,3 preserved from src1";
    EXPECT_EQ(st.ymm[6], 0ULL) << "VEX zeros upper YMM";
    EXPECT_EQ(st.ymm[7], 0ULL);
}

// Memory source: vcmpltss xmm0, xmm1, [rbx], 1.
TEST_F(CpuRuntimeTest, Vcmpss_MemorySource) {
    float* slot = reinterpret_cast<float*>(mem.CodePtr() + 0x100);
    *slot = 5.0f;

    const u8 program[] = {0xc5, 0xf2, 0xc2, 0x03, 0x01, 0xc3};
    std::memcpy(mem.CodePtr(), program, sizeof(program));
    u8* guest_rsp = mem.StackTop() - 8;
    *reinterpret_cast<u64*>(guest_rsp) = kReturnSentinel;
    GuestState st{};
    st.rip = reinterpret_cast<u64>(mem.CodePtr());
    st.gpr[4] = reinterpret_cast<u64>(guest_rsp);
    st.ymm[4] = 0x3f800000ULL; // src1 low = 1.0f
    st.gpr[3] = reinterpret_cast<u64>(slot); // rbx -> m32 (5.0f)

    Runtime rt; rt.Run(st);
    EXPECT_EQ(st.ymm[0] & 0xFFFFFFFFULL, 0xFFFFFFFFULL) << "1.0 < 5.0 true -> all-ones mask";
}

// NaN/unordered: vcmpneqss (imm 4) with a NaN operand. NEQ_UQ is true for
// unordered, so NaN != anything -> 0xFFFFFFFF. Confirms predicate semantics are
// inherited from the host.
TEST_F(CpuRuntimeTest, Vcmpss_NotEqual_WithNaN_True) {
    const u8 program[] = {0xc5, 0xf2, 0xc2, 0xc2, 0x04, 0xc3};
    std::memcpy(mem.CodePtr(), program, sizeof(program));
    u8* guest_rsp = mem.StackTop() - 8;
    *reinterpret_cast<u64*>(guest_rsp) = kReturnSentinel;
    GuestState st{};
    st.rip = reinterpret_cast<u64>(mem.CodePtr());
    st.gpr[4] = reinterpret_cast<u64>(guest_rsp);
    st.ymm[4] = 0x7fc00000ULL; // src1 low = NaN (quiet)
    st.ymm[8] = 0x40000000ULL; // src2 low = 2.0f

    Runtime rt; rt.Run(st);
    EXPECT_EQ(st.ymm[0] & 0xFFFFFFFFULL, 0xFFFFFFFFULL) << "NEQ_UQ: NaN != 2.0 is true (unordered)";
}

// ============================================================================
// VMOVLHPS / VMOVHLPS — move a 64-bit half between vectors.
//
// Motivated by CUSA02394 "WE ARE DOOMED", which exited the JIT at a reg,reg
// `vmovlhps` (2-byte VEX, length 4) in the eboot at guest 0x80013eecf
// (exit_reason=2). Both are 128-bit, register-only, no immediate:
//   VMOVLHPS dst,s1,s2: dst.q0 = s1.q0, dst.q1 = s2.q0 (s2 LOW -> dst HIGH)
//   VMOVHLPS dst,s1,s2: dst.q0 = s2.q1, dst.q1 = s1.q1 (s2 HIGH -> dst LOW)
// VEX zeroes bits 255:128.
//
// YMM layout: ymm[k*4+0]=XMMk q0, ymm[k*4+1]=q1.
// XMM0->ymm[0..], XMM1->ymm[4..], XMM2->ymm[8..].
// ============================================================================

// vmovlhps xmm0, xmm1, xmm2 — the crashing form.
TEST_F(CpuRuntimeTest, Vmovlhps_MovesSrc2LowToDstHigh) {
    const u8 program[] = {0xc5, 0xf0, 0x16, 0xc2, 0xc3};
    std::memcpy(mem.CodePtr(), program, sizeof(program));
    u8* guest_rsp = mem.StackTop() - 8;
    *reinterpret_cast<u64*>(guest_rsp) = kReturnSentinel;
    GuestState st{};
    st.rip = reinterpret_cast<u64>(mem.CodePtr());
    st.gpr[4] = reinterpret_cast<u64>(guest_rsp);
    st.ymm[4] = 0xA0A0A0A0A0A0A0A0ULL; // src1 q0
    st.ymm[5] = 0xA1A1A1A1A1A1A1A1ULL; // src1 q1 (discarded)
    st.ymm[8] = 0xB0B0B0B0B0B0B0B0ULL; // src2 q0 -> dst q1
    st.ymm[9] = 0xB1B1B1B1B1B1B1B1ULL; // src2 q1 (discarded)
    st.ymm[2] = 0xCAFE; st.ymm[3] = 0xF00D;

    Runtime rt; rt.Run(st);
    EXPECT_EQ(st.ymm[0], 0xA0A0A0A0A0A0A0A0ULL) << "dst.q0 = src1.q0";
    EXPECT_EQ(st.ymm[1], 0xB0B0B0B0B0B0B0B0ULL) << "dst.q1 = src2.q0";
    EXPECT_EQ(st.ymm[2], 0ULL) << "VEX zeros upper YMM";
    EXPECT_EQ(st.ymm[3], 0ULL);
}

// vmovhlps xmm0, xmm1, xmm2 — high halves.
TEST_F(CpuRuntimeTest, Vmovhlps_MovesSrc2HighToDstLow) {
    const u8 program[] = {0xc5, 0xf0, 0x12, 0xc2, 0xc3};
    std::memcpy(mem.CodePtr(), program, sizeof(program));
    u8* guest_rsp = mem.StackTop() - 8;
    *reinterpret_cast<u64*>(guest_rsp) = kReturnSentinel;
    GuestState st{};
    st.rip = reinterpret_cast<u64>(mem.CodePtr());
    st.gpr[4] = reinterpret_cast<u64>(guest_rsp);
    st.ymm[4] = 0xA0A0A0A0A0A0A0A0ULL; // src1 q0 (discarded)
    st.ymm[5] = 0xA1A1A1A1A1A1A1A1ULL; // src1 q1 -> dst q1
    st.ymm[8] = 0xB0B0B0B0B0B0B0B0ULL; // src2 q0 (discarded)
    st.ymm[9] = 0xB1B1B1B1B1B1B1B1ULL; // src2 q1 -> dst q0

    Runtime rt; rt.Run(st);
    EXPECT_EQ(st.ymm[0], 0xB1B1B1B1B1B1B1B1ULL) << "dst.q0 = src2.q1";
    EXPECT_EQ(st.ymm[1], 0xA1A1A1A1A1A1A1A1ULL) << "dst.q1 = src1.q1";
    EXPECT_EQ(st.ymm[2], 0ULL);
    EXPECT_EQ(st.ymm[3], 0ULL);
}

// In-place: vmovlhps xmm1, xmm1, xmm2 — dst == src1. Read-before-write keeps
// the low half (which comes from src1) correct.
TEST_F(CpuRuntimeTest, Vmovlhps_InPlace_DstEqualsSrc1) {
    const u8 program[] = {0xc5, 0xf0, 0x16, 0xca, 0xc3}; // vmovlhps xmm1,xmm1,xmm2
    std::memcpy(mem.CodePtr(), program, sizeof(program));
    u8* guest_rsp = mem.StackTop() - 8;
    *reinterpret_cast<u64*>(guest_rsp) = kReturnSentinel;
    GuestState st{};
    st.rip = reinterpret_cast<u64>(mem.CodePtr());
    st.gpr[4] = reinterpret_cast<u64>(guest_rsp);
    st.ymm[4] = 0xA0A0A0A0A0A0A0A0ULL; // xmm1 = src1 AND dst
    st.ymm[5] = 0xA1A1A1A1A1A1A1A1ULL;
    st.ymm[6] = 0xDEAD; st.ymm[7] = 0xBEEF; // upper YMM1, must be zeroed
    st.ymm[8] = 0xB0B0B0B0B0B0B0B0ULL;
    st.ymm[9] = 0xB1B1B1B1B1B1B1B1ULL;

    Runtime rt; rt.Run(st);
    EXPECT_EQ(st.ymm[4], 0xA0A0A0A0A0A0A0A0ULL) << "dst.q0 = src1.q0 (preserved)";
    EXPECT_EQ(st.ymm[5], 0xB0B0B0B0B0B0B0B0ULL) << "dst.q1 = src2.q0";
    EXPECT_EQ(st.ymm[6], 0ULL) << "VEX zeros upper YMM even in-place";
    EXPECT_EQ(st.ymm[7], 0ULL);
}

// Float-pair sanity: VMOVLHPS combining {1.0,2.0} (src1) and {3.0,4.0} (src2)
// yields {1.0, 2.0, 3.0, 4.0} (src1 low pair, then src2 low pair).
TEST_F(CpuRuntimeTest, Vmovlhps_FloatPairs) {
    const u8 program[] = {0xc5, 0xf0, 0x16, 0xc2, 0xc3};
    std::memcpy(mem.CodePtr(), program, sizeof(program));
    u8* guest_rsp = mem.StackTop() - 8;
    *reinterpret_cast<u64*>(guest_rsp) = kReturnSentinel;
    GuestState st{};
    st.rip = reinterpret_cast<u64>(mem.CodePtr());
    st.gpr[4] = reinterpret_cast<u64>(guest_rsp);
    const u32 f1 = std::bit_cast<u32>(1.0f), f2 = std::bit_cast<u32>(2.0f);
    const u32 f3 = std::bit_cast<u32>(3.0f), f4 = std::bit_cast<u32>(4.0f);
    st.ymm[4] = (static_cast<u64>(f2) << 32) | f1; // src1 q0 = {1.0,2.0}
    st.ymm[5] = 0;
    st.ymm[8] = (static_cast<u64>(f4) << 32) | f3; // src2 q0 = {3.0,4.0}
    st.ymm[9] = 0;

    Runtime rt; rt.Run(st);
    EXPECT_EQ(st.ymm[0], (static_cast<u64>(f2) << 32) | f1) << "low pair = 1.0,2.0 from src1";
    EXPECT_EQ(st.ymm[1], (static_cast<u64>(f4) << 32) | f3) << "high pair = 3.0,4.0 from src2";
}

// ============================================================================
// FNSTCW / FLDCW — x87 control-word store / load (minimal model).
//
// Motivated by CUSA02394 "WE ARE DOOMED", which exited the JIT at an `fnstcw`
// (D9 /7, length 3) in libc at guest 0x8075abcf2 (exit_reason=2) — the first
// x87 instruction encountered. The runtime does not model the x87 control word
// (FP goes through the SSE/MXCSR path), so FNSTCW stores the standard
// post-FINIT control word 0x037F and FLDCW is a no-op. This lets the libc
// float->int truncation idiom (fnstcw save / fldcw truncate / fistp / fldcw
// restore) round-trip its saved word without faulting.
// ============================================================================

// fnstcw word[rbx] stores 0x037F as a 16-bit value, not disturbing neighbors.
TEST_F(CpuRuntimeTest, Fnstcw_StoresDefaultControlWord) {
    u16* slot = reinterpret_cast<u16*>(mem.CodePtr() + 0x100);
    slot[0] = 0xAAAA;
    slot[1] = 0xBBBB; // sentinel after the 2-byte store

    const u8 program[] = {0xd9, 0x3b, 0xc3}; // fnstcw word[rbx] ; ret
    std::memcpy(mem.CodePtr(), program, sizeof(program));
    u8* guest_rsp = mem.StackTop() - 8;
    *reinterpret_cast<u64*>(guest_rsp) = kReturnSentinel;
    GuestState st{};
    st.rip = reinterpret_cast<u64>(mem.CodePtr());
    st.gpr[4] = reinterpret_cast<u64>(guest_rsp);
    st.gpr[3] = reinterpret_cast<u64>(slot); // rbx -> m16

    Runtime rt; rt.Run(st);
    EXPECT_EQ(slot[0], 0x037Fu) << "default post-FINIT control word stored";
    EXPECT_EQ(slot[1], 0xBBBBu) << "16-bit store must not spill into next word";
}

// fldcw word[rbx] is a no-op: it must not fault and execution continues to ret.
TEST_F(CpuRuntimeTest, Fldcw_IsNoOp) {
    u16* slot = reinterpret_cast<u16*>(mem.CodePtr() + 0x100);
    *slot = 0x0C7F; // a truncating control word; ignored by our model

    const u8 program[] = {0xd9, 0x2b, 0xc3}; // fldcw word[rbx] ; ret
    std::memcpy(mem.CodePtr(), program, sizeof(program));
    u8* guest_rsp = mem.StackTop() - 8;
    *reinterpret_cast<u64*>(guest_rsp) = kReturnSentinel;
    GuestState st{};
    st.rip = reinterpret_cast<u64>(mem.CodePtr());
    st.gpr[4] = reinterpret_cast<u64>(guest_rsp);
    st.gpr[3] = reinterpret_cast<u64>(slot);
    st.gpr[0] = 0x1234ULL; // rax untouched marker

    Runtime rt; rt.Run(st);
    EXPECT_EQ(st.exit_reason, static_cast<u32>(ExitReason::BlockEnd)) << "fldcw runs and reaches ret";
    EXPECT_EQ(st.gpr[0], 0x1234ULL) << "fldcw has no observable register effect";
}

// The save/restore round-trip: fnstcw [save] ... fldcw [save]. After fnstcw the
// saved word is 0x037F; the subsequent fldcw consumes it harmlessly. We model
// the common sequence as: fnstcw [rbx] ; fldcw [rbx] ; ret.
TEST_F(CpuRuntimeTest, Fnstcw_Fldcw_RoundTrips) {
    u16* save = reinterpret_cast<u16*>(mem.CodePtr() + 0x100);
    *save = 0x0000;

    const u8 program[] = {
        0xd9, 0x3b, // fnstcw word[rbx]
        0xd9, 0x2b, // fldcw  word[rbx]
        0xc3,       // ret
    };
    std::memcpy(mem.CodePtr(), program, sizeof(program));
    u8* guest_rsp = mem.StackTop() - 8;
    *reinterpret_cast<u64*>(guest_rsp) = kReturnSentinel;
    GuestState st{};
    st.rip = reinterpret_cast<u64>(mem.CodePtr());
    st.gpr[4] = reinterpret_cast<u64>(guest_rsp);
    st.gpr[3] = reinterpret_cast<u64>(save);

    Runtime rt; rt.Run(st);
    EXPECT_EQ(*save, 0x037Fu) << "fnstcw wrote the control word, fldcw left it intact";
    EXPECT_EQ(st.exit_reason, static_cast<u32>(ExitReason::BlockEnd)) << "sequence completes";
}

// ============================================================================
// STMXCSR / LDMXCSR — store / load the SSE control-and-status register.
//
// Motivated by CUSA02394 "WE ARE DOOMED", which exited the JIT at an `stmxcsr`
// (0F AE /3, length 4) in libc at guest 0x8075abcf5 (exit_reason=2) —
// immediately after the FNSTCW, in the float-conversion routine that saves both
// the x87 control word and MXCSR. MXCSR is genuinely modelled via
// GuestState.mxcsr, so these round-trip faithfully.
// ============================================================================

// stmxcsr [rbx] stores the guest MXCSR (4 bytes), not disturbing neighbors.
TEST_F(CpuRuntimeTest, Stmxcsr_StoresGuestMxcsr) {
    u32* slot = reinterpret_cast<u32*>(mem.CodePtr() + 0x100);
    slot[0] = 0xAAAAAAAA;
    slot[1] = 0xBBBBBBBB; // sentinel after the 4-byte store

    const u8 program[] = {0x0f, 0xae, 0x1b, 0xc3}; // stmxcsr [rbx] ; ret
    std::memcpy(mem.CodePtr(), program, sizeof(program));
    u8* guest_rsp = mem.StackTop() - 8;
    *reinterpret_cast<u64*>(guest_rsp) = kReturnSentinel;
    GuestState st{};
    st.rip = reinterpret_cast<u64>(mem.CodePtr());
    st.gpr[4] = reinterpret_cast<u64>(guest_rsp);
    st.gpr[3] = reinterpret_cast<u64>(slot); // rbx -> m32
    st.mxcsr = 0x00001F80;                   // default MXCSR

    Runtime rt; rt.Run(st);
    EXPECT_EQ(slot[0], 0x00001F80u) << "stored the guest MXCSR value";
    EXPECT_EQ(slot[1], 0xBBBBBBBBu) << "4-byte store must not spill into next dword";
}

// ldmxcsr [rbx] loads the memory value into the guest MXCSR field.
TEST_F(CpuRuntimeTest, Ldmxcsr_LoadsIntoGuestMxcsr) {
    u32* slot = reinterpret_cast<u32*>(mem.CodePtr() + 0x100);
    *slot = 0x00009F80; // default + FTZ bit set, say

    const u8 program[] = {0x0f, 0xae, 0x13, 0xc3}; // ldmxcsr [rbx] ; ret
    std::memcpy(mem.CodePtr(), program, sizeof(program));
    u8* guest_rsp = mem.StackTop() - 8;
    *reinterpret_cast<u64*>(guest_rsp) = kReturnSentinel;
    GuestState st{};
    st.rip = reinterpret_cast<u64>(mem.CodePtr());
    st.gpr[4] = reinterpret_cast<u64>(guest_rsp);
    st.gpr[3] = reinterpret_cast<u64>(slot);
    st.mxcsr = 0x00001F80; // will be overwritten by the load

    Runtime rt; rt.Run(st);
    EXPECT_EQ(st.mxcsr, 0x00009F80u) << "guest MXCSR updated from memory";
}

// Round-trip: ldmxcsr [src] ; stmxcsr [dst] copies the value through the guest
// MXCSR field. (Models the conversion routine's save/modify/restore around an
// SSE op.)
TEST_F(CpuRuntimeTest, Mxcsr_RoundTripsThroughGuestField) {
    u32* src = reinterpret_cast<u32*>(mem.CodePtr() + 0x100);
    u32* dst = reinterpret_cast<u32*>(mem.CodePtr() + 0x110);
    *src = 0x0000DA7A;
    *dst = 0;

    const u8 program[] = {
        0x0f, 0xae, 0x13,       // ldmxcsr [rbx]   (rbx -> src)
        0x0f, 0xae, 0x19,       // stmxcsr [rcx]   (rcx -> dst)
        0xc3,                   // ret
    };
    std::memcpy(mem.CodePtr(), program, sizeof(program));
    u8* guest_rsp = mem.StackTop() - 8;
    *reinterpret_cast<u64*>(guest_rsp) = kReturnSentinel;
    GuestState st{};
    st.rip = reinterpret_cast<u64>(mem.CodePtr());
    st.gpr[4] = reinterpret_cast<u64>(guest_rsp);
    st.gpr[3] = reinterpret_cast<u64>(src); // rbx
    st.gpr[1] = reinterpret_cast<u64>(dst); // rcx

    Runtime rt; rt.Run(st);
    EXPECT_EQ(st.mxcsr, 0x0000DA7Au) << "ldmxcsr set the guest field";
    EXPECT_EQ(*dst, 0x0000DA7Au) << "stmxcsr wrote the same value back out";
}

// ============================================================================
// ANDN with a MEMORY source operand (src2) — BMI1 dst = ~src1 & src2.
//
// Motivated by CUSA02394 "WE ARE DOOMED", which exited the JIT (on the
// AudioOutThread) at an `andn` (length 6, memory src2) in a loaded module at
// guest 0x808667cda. The register-source form was already supported; the
// memory-source form (the length-6 disp encoding) was not. Flags: SF,ZF from
// the result; OF,CF cleared. 32-bit form zero-extends bits 63:32.
// ============================================================================

// andn eax, ecx, [rbx] — 32-bit memory source. ~0x0F0F0F0F & 0xFFFFFFFF.
TEST_F(CpuRuntimeTest, Andn32_MemSrc_Computes) {
    u32* slot = reinterpret_cast<u32*>(mem.CodePtr() + 0x100);
    *slot = 0xFFFFFFFF; // src2

    const u8 program[] = {0xc4, 0xe2, 0x70, 0xf2, 0x03, 0xc3}; // andn eax,ecx,[rbx] ; ret
    std::memcpy(mem.CodePtr(), program, sizeof(program));
    u8* guest_rsp = mem.StackTop() - 8;
    *reinterpret_cast<u64*>(guest_rsp) = kReturnSentinel;
    GuestState st{};
    st.rip = reinterpret_cast<u64>(mem.CodePtr());
    st.gpr[4] = reinterpret_cast<u64>(guest_rsp);
    st.gpr[1] = 0x0F0F0F0FULL;                // ecx = src1
    st.gpr[3] = reinterpret_cast<u64>(slot);  // rbx -> m32
    st.gpr[0] = 0xDEADBEEFDEADBEEFULL;        // rax pre-pollute

    Runtime rt; rt.Run(st);
    EXPECT_EQ(st.gpr[0], 0xF0F0F0F0ULL) << "~0x0F0F0F0F & 0xFFFFFFFF";
    EXPECT_EQ(st.gpr[0] >> 32, 0u) << "32-bit result zero-extends";
}

// andn rax, rcx, [rbx] — 64-bit memory source.
TEST_F(CpuRuntimeTest, Andn64_MemSrc_Computes) {
    u64* slot = reinterpret_cast<u64*>(mem.CodePtr() + 0x100);
    *slot = 0xFFFFFFFFFFFFFFFFULL; // src2

    const u8 program[] = {0xc4, 0xe2, 0xf0, 0xf2, 0x03, 0xc3}; // andn rax,rcx,[rbx] ; ret
    std::memcpy(mem.CodePtr(), program, sizeof(program));
    u8* guest_rsp = mem.StackTop() - 8;
    *reinterpret_cast<u64*>(guest_rsp) = kReturnSentinel;
    GuestState st{};
    st.rip = reinterpret_cast<u64>(mem.CodePtr());
    st.gpr[4] = reinterpret_cast<u64>(guest_rsp);
    st.gpr[1] = 0x00000000FFFF0000ULL;        // rcx = src1
    st.gpr[3] = reinterpret_cast<u64>(slot);

    Runtime rt; rt.Run(st);
    EXPECT_EQ(st.gpr[0], 0xFFFFFFFF0000FFFFULL) << "~src1 & all-ones = ~src1";
}

// andn eax, ecx, [rbx+0x10] — the length-6 disp form (the crashing addressing).
TEST_F(CpuRuntimeTest, Andn32_MemSrcDisp_Computes) {
    u32* base = reinterpret_cast<u32*>(mem.CodePtr() + 0x100);
    u32* slot = reinterpret_cast<u32*>(mem.CodePtr() + 0x110); // base + 0x10
    *slot = 0x12345678;

    const u8 program[] = {0xc4, 0xe2, 0x70, 0xf2, 0x43, 0x10, 0xc3}; // andn eax,ecx,[rbx+0x10]
    std::memcpy(mem.CodePtr(), program, sizeof(program));
    u8* guest_rsp = mem.StackTop() - 8;
    *reinterpret_cast<u64*>(guest_rsp) = kReturnSentinel;
    GuestState st{};
    st.rip = reinterpret_cast<u64>(mem.CodePtr());
    st.gpr[4] = reinterpret_cast<u64>(guest_rsp);
    st.gpr[1] = 0x0000FFFFULL;                // ecx = src1
    st.gpr[3] = reinterpret_cast<u64>(base);  // rbx; +0x10 -> slot

    Runtime rt; rt.Run(st);
    // ~0x0000FFFF & 0x12345678 = 0xFFFF0000 & 0x12345678 = 0x12340000
    EXPECT_EQ(st.gpr[0], 0x12340000ULL) << "~src1 & mem[rbx+0x10]";
}

// Flags: result zero -> ZF set, SF clear. (src1 all-ones -> ~src1 = 0 -> 0&x=0)
TEST_F(CpuRuntimeTest, Andn32_MemSrc_ZeroResult_SetsZf) {
    u32* slot = reinterpret_cast<u32*>(mem.CodePtr() + 0x100);
    *slot = 0xFFFFFFFF;

    const u8 program[] = {0xc4, 0xe2, 0x70, 0xf2, 0x03, 0xc3};
    std::memcpy(mem.CodePtr(), program, sizeof(program));
    u8* guest_rsp = mem.StackTop() - 8;
    *reinterpret_cast<u64*>(guest_rsp) = kReturnSentinel;
    GuestState st{};
    st.rip = reinterpret_cast<u64>(mem.CodePtr());
    st.gpr[4] = reinterpret_cast<u64>(guest_rsp);
    st.gpr[1] = 0xFFFFFFFFULL;                // src1 all-ones -> ~src1 = 0
    st.gpr[3] = reinterpret_cast<u64>(slot);

    Runtime rt; rt.Run(st);
    EXPECT_EQ(st.gpr[0], 0x00000000ULL) << "0 & anything = 0";
    EXPECT_EQ(st.rflags & 0x40u, 0x40u) << "ZF set (zero result)";
    EXPECT_EQ(st.rflags & 0x80u, 0u) << "SF clear";
    EXPECT_EQ(st.rflags & 0x1u, 0u) << "CF cleared by ANDN";
    EXPECT_EQ(st.rflags & 0x800u, 0u) << "OF cleared by ANDN";
}

// Flags: high-bit result -> SF set. ~0 & 0x80000000 = 0x80000000.
TEST_F(CpuRuntimeTest, Andn32_MemSrc_HighBit_SetsSf) {
    u32* slot = reinterpret_cast<u32*>(mem.CodePtr() + 0x100);
    *slot = 0x80000000;

    const u8 program[] = {0xc4, 0xe2, 0x70, 0xf2, 0x03, 0xc3};
    std::memcpy(mem.CodePtr(), program, sizeof(program));
    u8* guest_rsp = mem.StackTop() - 8;
    *reinterpret_cast<u64*>(guest_rsp) = kReturnSentinel;
    GuestState st{};
    st.rip = reinterpret_cast<u64>(mem.CodePtr());
    st.gpr[4] = reinterpret_cast<u64>(guest_rsp);
    st.gpr[1] = 0x00000000ULL;                // ~src1 = all-ones
    st.gpr[3] = reinterpret_cast<u64>(slot);

    Runtime rt; rt.Run(st);
    EXPECT_EQ(st.gpr[0], 0x80000000ULL);
    EXPECT_EQ(st.rflags & 0x80u, 0x80u) << "SF set from bit 31 of the 32-bit result";
    EXPECT_EQ(st.rflags & 0x40u, 0u) << "ZF clear";
}

// ============================================================================
// MFENCE / LFENCE — memory fences (0F AE F0 / 0F AE E8), no operands.
//
// MFENCE was the run-ending gap in CUSA02394 "WE ARE DOOMED": the JIT exited
// (exit_reason=2) at an mfence in libc, guest 0x8075ac337, inside a lock-free
// synchronization primitive. The host is strongly ordered (TSO) and our
// emitted stream runs in program order, so the fence has no reordering to
// undo within a block; we still emit the host fence (3 bytes) to preserve
// ordering w.r.t. genuinely concurrent host threads. These tests assert the
// block now compiles and runs to completion rather than exiting Unsupported,
// and that a fence between stores does not disturb their effects.
// ============================================================================

// mfence ; ret — the exact gap. Must reach the RET (BlockEnd), not exit
// UnsupportedInstruction.
TEST_F(CpuRuntimeTest, Mfence_CompilesAndRuns) {
    const u8 program[] = {0x0f, 0xae, 0xf0, 0xc3}; // mfence ; ret
    const auto r = RunProgram(program, sizeof(program), mem);
    EXPECT_EQ(r.state.rip, kReturnSentinel) << "ran through the fence to the ret";
    EXPECT_EQ(r.state.exit_reason, static_cast<u32>(ExitReason::BlockEnd))
        << "mfence must no longer trip the unsupported-instruction exit";
}

// lfence ; ret — pre-empted sibling, same opcode family.
TEST_F(CpuRuntimeTest, Lfence_CompilesAndRuns) {
    const u8 program[] = {0x0f, 0xae, 0xe8, 0xc3}; // lfence ; ret
    const auto r = RunProgram(program, sizeof(program), mem);
    EXPECT_EQ(r.state.rip, kReturnSentinel);
    EXPECT_EQ(r.state.exit_reason, static_cast<u32>(ExitReason::BlockEnd));
}

// A fence wedged between two memory stores and a register op must leave all of
// them intact — verifies the fence emitter is inert w.r.t. guest state.
TEST_F(CpuRuntimeTest, Mfence_PreservesSurroundingEffects) {
    u64* slot = reinterpret_cast<u64*>(mem.CodePtr() + 0x100);
    slot[0] = 0;
    slot[1] = 0;

    const u8 program[] = {
        0x48, 0xc7, 0xc0, 0x11, 0x00, 0x00, 0x00, // mov rax, 0x11
        0x48, 0x89, 0x03,                         // mov [rbx], rax
        0x0f, 0xae, 0xf0,                         // mfence
        0x48, 0xc7, 0xc0, 0x22, 0x00, 0x00, 0x00, // mov rax, 0x22
        0x48, 0x89, 0x43, 0x08,                   // mov [rbx+8], rax
        0xc3,                                     // ret
    };
    std::memcpy(mem.CodePtr(), program, sizeof(program));
    u8* guest_rsp = mem.StackTop() - 8;
    *reinterpret_cast<u64*>(guest_rsp) = kReturnSentinel;
    GuestState st{};
    st.rip = reinterpret_cast<u64>(mem.CodePtr());
    st.gpr[4] = reinterpret_cast<u64>(guest_rsp);
    st.gpr[3] = reinterpret_cast<u64>(slot); // rbx -> slot

    Runtime rt; rt.Run(st);
    EXPECT_EQ(slot[0], 0x11u) << "store before the fence";
    EXPECT_EQ(slot[1], 0x22u) << "store after the fence";
    EXPECT_EQ(st.gpr[0], 0x22u) << "register op after the fence executed";
    EXPECT_EQ(st.rip, kReturnSentinel);
    EXPECT_EQ(st.exit_reason, static_cast<u32>(ExitReason::BlockEnd));
}


// ============================================================================
// 16-bit register INC/DEC (`inc r16` / `dec r16`, 66-prefixed). Like the
// 8-bit register forms, these preserve CF and the parent register's upper
// bits — a 16-bit write touches only bits 15:0, leaving 63:16 intact (in
// contrast to a 32-bit write, which zero-extends). Flags ZF/SF/OF/PF are
// computed at 16-bit width via the host round-trip.
// ============================================================================

// inc ax: low 16 bits increment, upper 48 bits and CF preserved.
TEST_F(CpuRuntimeTest, Inc16_AX_PreservesUpperAndCF) {
    const u8 program[] = {
        0x66, 0xff, 0xc0, // inc ax
        0xc3,
    };
    std::memcpy(mem.CodePtr(), program, sizeof(program));
    u8* guest_rsp = mem.StackTop() - 8;
    *reinterpret_cast<u64*>(guest_rsp) = kReturnSentinel;
    GuestState st{};
    st.rip = reinterpret_cast<u64>(mem.CodePtr());
    st.gpr[4] = reinterpret_cast<u64>(guest_rsp);
    st.gpr[0] = 0xDEADBEEF12340041ULL; // ax = 0x0041, upper 48 must survive
    st.rflags = 0x2 | 0x1; // CF pre-set
    Runtime rt;
    rt.Run(st);
    EXPECT_EQ(st.gpr[0], 0xDEADBEEF12340042ULL) << "ax incremented, upper 48 bits preserved";
    EXPECT_EQ(st.rflags & 0x1ULL, 0x1ULL) << "CF preserved (INC does not affect CF)";
    EXPECT_EQ(st.rflags & (1ULL<<6), 0ULL) << "ZF clear";
}

// inc cx on 0xFFFF: wraps to 0, ZF set, CF stays clear, upper preserved.
// Uses cx (not ax) to prove the slot index isn't hardcoded.
TEST_F(CpuRuntimeTest, Inc16_CX_WrapSetsZF) {
    const u8 program[] = {
        0x66, 0xff, 0xc1, // inc cx
        0xc3,
    };
    std::memcpy(mem.CodePtr(), program, sizeof(program));
    u8* guest_rsp = mem.StackTop() - 8;
    *reinterpret_cast<u64*>(guest_rsp) = kReturnSentinel;
    GuestState st{};
    st.rip = reinterpret_cast<u64>(mem.CodePtr());
    st.gpr[4] = reinterpret_cast<u64>(guest_rsp);
    st.gpr[1] = 0xAAAAAAAAAAAAFFFFULL; // cx = 0xFFFF
    st.rflags = 0x2; // CF clear
    Runtime rt;
    rt.Run(st);
    EXPECT_EQ(st.gpr[1] & 0xFFFFULL, 0ULL) << "cx wrapped to 0";
    EXPECT_EQ(st.gpr[1] & ~0xFFFFULL, 0xAAAAAAAAAAAA0000ULL) << "upper 48 bits preserved";
    EXPECT_EQ(st.rflags & (1ULL<<6), (1ULL<<6)) << "ZF set: 16-bit result zero";
    EXPECT_EQ(st.rflags & 0x1ULL, 0ULL) << "CF still clear on wrap";
}

// inc ax at 0x7FFF: signed 16-bit overflow sets OF and SF.
TEST_F(CpuRuntimeTest, Inc16_AX_SignedOverflow) {
    const u8 program[] = {
        0x66, 0xff, 0xc0, // inc ax
        0xc3,
    };
    std::memcpy(mem.CodePtr(), program, sizeof(program));
    u8* guest_rsp = mem.StackTop() - 8;
    *reinterpret_cast<u64*>(guest_rsp) = kReturnSentinel;
    GuestState st{};
    st.rip = reinterpret_cast<u64>(mem.CodePtr());
    st.gpr[4] = reinterpret_cast<u64>(guest_rsp);
    st.gpr[0] = 0x7FFFULL; // ax = 0x7FFF
    st.rflags = 0x2;
    Runtime rt;
    rt.Run(st);
    EXPECT_EQ(st.gpr[0] & 0xFFFFULL, 0x8000ULL);
    EXPECT_EQ(st.rflags & (1ULL<<11), (1ULL<<11)) << "OF set: 16-bit signed overflow";
    EXPECT_EQ(st.rflags & (1ULL<<7), (1ULL<<7)) << "SF set: result negative";
}

// dec ax: low 16 bits decrement, upper 48 bits and CF preserved.
TEST_F(CpuRuntimeTest, Dec16_AX_PreservesUpperAndCF) {
    const u8 program[] = {
        0x66, 0xff, 0xc8, // dec ax
        0xc3,
    };
    std::memcpy(mem.CodePtr(), program, sizeof(program));
    u8* guest_rsp = mem.StackTop() - 8;
    *reinterpret_cast<u64*>(guest_rsp) = kReturnSentinel;
    GuestState st{};
    st.rip = reinterpret_cast<u64>(mem.CodePtr());
    st.gpr[4] = reinterpret_cast<u64>(guest_rsp);
    st.gpr[0] = 0xDEADBEEF12340042ULL; // ax = 0x0042
    st.rflags = 0x2 | 0x1; // CF pre-set
    Runtime rt;
    rt.Run(st);
    EXPECT_EQ(st.gpr[0], 0xDEADBEEF12340041ULL) << "ax decremented, upper 48 bits preserved";
    EXPECT_EQ(st.rflags & 0x1ULL, 0x1ULL) << "CF preserved (DEC does not affect CF)";
    EXPECT_EQ(st.rflags & (1ULL<<6), 0ULL) << "ZF clear";
}

// dec cx from 1: result 0 sets ZF, CF preserved, upper bits intact.
TEST_F(CpuRuntimeTest, Dec16_CX_ToZero_SetsZf) {
    const u8 program[] = {
        0x66, 0xff, 0xc9, // dec cx
        0xc3,
    };
    std::memcpy(mem.CodePtr(), program, sizeof(program));
    u8* guest_rsp = mem.StackTop() - 8;
    *reinterpret_cast<u64*>(guest_rsp) = kReturnSentinel;
    GuestState st{};
    st.rip = reinterpret_cast<u64>(mem.CodePtr());
    st.gpr[4] = reinterpret_cast<u64>(guest_rsp);
    st.gpr[1] = 0x1234567800000001ULL; // cx = 0x0001
    st.rflags = 0x2; // CF clear
    Runtime rt;
    rt.Run(st);
    EXPECT_EQ(st.gpr[1] & 0xFFFFULL, 0ULL) << "cx decremented to 0";
    EXPECT_EQ(st.gpr[1] & ~0xFFFFULL, 0x1234567800000000ULL) << "upper 48 bits preserved";
    EXPECT_EQ(st.rflags & (1ULL<<6), (1ULL<<6)) << "ZF set: 16-bit result zero";
}

// dec ax from 0: underflow wraps to 0xFFFF, SF set (negative), CF preserved.
TEST_F(CpuRuntimeTest, Dec16_AX_UnderflowWraps) {
    const u8 program[] = {
        0x66, 0xff, 0xc8, // dec ax
        0xc3,
    };
    std::memcpy(mem.CodePtr(), program, sizeof(program));
    u8* guest_rsp = mem.StackTop() - 8;
    *reinterpret_cast<u64*>(guest_rsp) = kReturnSentinel;
    GuestState st{};
    st.rip = reinterpret_cast<u64>(mem.CodePtr());
    st.gpr[4] = reinterpret_cast<u64>(guest_rsp);
    st.gpr[0] = 0x0000000000000000ULL; // ax = 0
    st.rflags = 0x2;
    Runtime rt;
    rt.Run(st);
    EXPECT_EQ(st.gpr[0] & 0xFFFFULL, 0xFFFFULL) << "ax underflowed to 0xFFFF";
    EXPECT_EQ(st.rflags & (1ULL<<7), (1ULL<<7)) << "SF set: result has high bit";
    EXPECT_EQ(st.rflags & (1ULL<<6), 0ULL) << "ZF clear";
}


// ============================================================================
// VPSHUFB — byte shuffle, register and memory mask forms. dst = xmm0 (ymm[0]),
// src1 = xmm1 (ymm[4..5]). The control vector selects src bytes per output
// byte; a control byte with bit 7 set zeroes that output byte. The 128-bit
// VEX form zeros bits 255:128 of the destination. Test vector hand-computed:
//   src  = 0x10..0x1f (byte j = 0x10+j)
//   mask = reverse (15..0) with byte0's high bit set (-> output byte0 = 0)
//   out.q0 = 0x18191a1b1c1d1e00  out.q1 = 0x1011121314151617
// ============================================================================

// Register-mask form: vpshufb xmm0, xmm1, xmm2.
TEST_F(CpuRuntimeTest, Vpshufb_Xmm_RegMask_Shuffles) {
    const u8 program[] = {
        0xc4, 0xe2, 0x71, 0x00, 0xc2, // vpshufb xmm0, xmm1, xmm2
        0xc3,
    };
    std::memcpy(mem.CodePtr(), program, sizeof(program));
    u8* guest_rsp = mem.StackTop() - 8;
    *reinterpret_cast<u64*>(guest_rsp) = kReturnSentinel;
    GuestState st{};
    st.rip = reinterpret_cast<u64>(mem.CodePtr());
    st.gpr[4] = reinterpret_cast<u64>(guest_rsp);
    // xmm1 (src1) = ymm[4],ymm[5]; xmm2 (mask) = ymm[8],ymm[9].
    st.ymm[4] = 0x1716151413121110ULL; st.ymm[5] = 0x1f1e1d1c1b1a1918ULL;
    st.ymm[8] = 0x08090a0b0c0d0e8fULL; st.ymm[9] = 0x0001020304050607ULL;
    // Pre-dirty dst (xmm0) upper lane to prove it gets zeroed.
    st.ymm[0] = 0xDEADBEEFULL; st.ymm[1] = 0xCAFEBABEULL;
    st.ymm[2] = 0x1111111111111111ULL; st.ymm[3] = 0x2222222222222222ULL;
    Runtime rt;
    rt.Run(st);
    EXPECT_EQ(st.ymm[0], 0x18191a1b1c1d1e00ULL) << "dst low qword shuffled";
    EXPECT_EQ(st.ymm[1], 0x1011121314151617ULL) << "dst high qword shuffled";
    EXPECT_EQ(st.ymm[2], 0ULL) << "128-bit VEX zeros bits 191:128";
    EXPECT_EQ(st.ymm[3], 0ULL) << "128-bit VEX zeros bits 255:192";
}

// Memory-mask form: vpshufb xmm0, xmm1, [rax]. The control vector lives in
// guest memory pointed to by rax; same expected result as the reg form.
TEST_F(CpuRuntimeTest, Vpshufb_Xmm_MemMask_Shuffles) {
    // 16-byte control vector in a scratch buffer one page past the code.
    u8* scratch = mem.CodePtr() + 0x100;
    const u8 mask_bytes[16] = {
        0x8f,0x0e,0x0d,0x0c,0x0b,0x0a,0x09,0x08,
        0x07,0x06,0x05,0x04,0x03,0x02,0x01,0x00,
    };
    std::memcpy(scratch, mask_bytes, sizeof(mask_bytes));

    u8 program[] = {
        0x48, 0xb8, 0,0,0,0,0,0,0,0,   // mov rax, <scratch addr>  (imm @2..9)
        0xc4, 0xe2, 0x71, 0x00, 0x00,  // vpshufb xmm0, xmm1, [rax]
        0xc3,
    };
    const u64 scratch_addr = reinterpret_cast<u64>(scratch);
    std::memcpy(&program[2], &scratch_addr, sizeof(scratch_addr));
    std::memcpy(mem.CodePtr(), program, sizeof(program));

    u8* guest_rsp = mem.StackTop() - 8;
    *reinterpret_cast<u64*>(guest_rsp) = kReturnSentinel;
    GuestState st{};
    st.rip = reinterpret_cast<u64>(mem.CodePtr());
    st.gpr[4] = reinterpret_cast<u64>(guest_rsp);
    st.ymm[4] = 0x1716151413121110ULL; st.ymm[5] = 0x1f1e1d1c1b1a1918ULL;
    st.ymm[2] = 0x3333333333333333ULL; st.ymm[3] = 0x4444444444444444ULL;
    Runtime rt;
    rt.Run(st);
    EXPECT_EQ(st.ymm[0], 0x18191a1b1c1d1e00ULL) << "dst low qword shuffled (mem mask)";
    EXPECT_EQ(st.ymm[1], 0x1011121314151617ULL) << "dst high qword shuffled (mem mask)";
    EXPECT_EQ(st.ymm[2], 0ULL) << "128-bit VEX zeros upper lane (mem mask)";
    EXPECT_EQ(st.ymm[3], 0ULL) << "128-bit VEX zeros upper lane (mem mask)";
}

// ============================================================================
// x87 FPU — Group 1: load/store (fld, fst, fstp, fild, fistp)
// ============================================================================
//
// Harness note: these tests place a small scratch data area inside the
// guest memory region (well clear of the code at the base and the stack
// at the top) and point RDI at it. The guest program loads/stores via
// [rdi+disp]. We read back the stored bytes and the resulting fpu_top /
// fpu_tag to verify both the value path and the stack bookkeeping.

namespace {
// A fixed offset into guest memory for x87 scratch data: past the code,
// far below the stack. 1 KiB in is plenty for these short programs.
constexpr u64 kX87ScratchOff = 1024;
} // namespace

// fld dword [rdi]; fstp dword [rdi+16]; ret
// Load a float, push, store-and-pop to a different slot. Expect the
// round-tripped float to match, fpu_top back to 0, tag empty again.
TEST_F(CpuRuntimeTest, X87_FldFstp_Float32RoundTrip) {
    u8* data = mem.CodePtr() + kX87ScratchOff;
    const float in = 3.5f;
    std::memcpy(data, &in, sizeof(in));
    std::memset(data + 16, 0, sizeof(float));

    u8 program[] = {
        0x48, 0xbf, 0,0,0,0,0,0,0,0,   // mov rdi, <data addr>  (imm @2..9)
        0xd9, 0x07,                     // fld  dword [rdi]
        0xd9, 0x5f, 0x10,               // fstp dword [rdi+0x10]
        0xc3,                           // ret
    };
    const u64 data_addr = reinterpret_cast<u64>(data);
    std::memcpy(&program[2], &data_addr, sizeof(data_addr));

    const auto r = RunProgram(program, sizeof(program), mem);
    float out = 0;
    std::memcpy(&out, data + 16, sizeof(out));

    EXPECT_EQ(out, 3.5f) << "float round-trips through the x87 stack";
    EXPECT_EQ(r.state.fpu_top, 0u) << "push then pop returns top to 0";
    EXPECT_EQ(r.state.fpu_tag & 0x1u, 0u) << "slot 0 marked empty after pop";
    EXPECT_EQ(r.state.exit_reason, static_cast<u32>(ExitReason::BlockEnd));
}

// fld qword [rdi]; fstp qword [rdi+16]; ret  — double round-trip.
TEST_F(CpuRuntimeTest, X87_FldFstp_Float64RoundTrip) {
    u8* data = mem.CodePtr() + kX87ScratchOff;
    const double in = 1.0e300; // large value: would lose to float, exact as double
    std::memcpy(data, &in, sizeof(in));
    std::memset(data + 16, 0, sizeof(double));

    u8 program[] = {
        0x48, 0xbf, 0,0,0,0,0,0,0,0,   // mov rdi, <data addr>
        0xdd, 0x07,                     // fld  qword [rdi]
        0xdd, 0x5f, 0x10,               // fstp qword [rdi+0x10]
        0xc3,
    };
    const u64 data_addr = reinterpret_cast<u64>(data);
    std::memcpy(&program[2], &data_addr, sizeof(data_addr));

    const auto r = RunProgram(program, sizeof(program), mem);
    double out = 0;
    std::memcpy(&out, data + 16, sizeof(out));

    EXPECT_EQ(out, 1.0e300) << "double round-trips bit-exact";
    EXPECT_EQ(r.state.fpu_top, 0u);
    EXPECT_EQ(r.state.exit_reason, static_cast<u32>(ExitReason::BlockEnd));
}

// fld dword [rdi]; fst dword [rdi+32]; fstp dword [rdi+16]; ret
// fst stores WITHOUT popping; the following fstp then stores the SAME
// value (still ST(0)) and pops. Both destinations must equal the input.
TEST_F(CpuRuntimeTest, X87_Fst_NoPop_LeavesValueOnStack) {
    u8* data = mem.CodePtr() + kX87ScratchOff;
    const float in = -2.25f;
    std::memcpy(data, &in, sizeof(in));
    std::memset(data + 16, 0, sizeof(float));
    std::memset(data + 32, 0, sizeof(float));

    u8 program[] = {
        0x48, 0xbf, 0,0,0,0,0,0,0,0,   // mov rdi, <data addr>
        0xd9, 0x07,                     // fld  dword [rdi]
        0xd9, 0x57, 0x20,               // fst  dword [rdi+0x20]   (no pop)
        0xd9, 0x5f, 0x10,               // fstp dword [rdi+0x10]   (pop)
        0xc3,
    };
    const u64 data_addr = reinterpret_cast<u64>(data);
    std::memcpy(&program[2], &data_addr, sizeof(data_addr));

    const auto r = RunProgram(program, sizeof(program), mem);
    float out_fst = 0, out_fstp = 0;
    std::memcpy(&out_fst, data + 32, sizeof(out_fst));
    std::memcpy(&out_fstp, data + 16, sizeof(out_fstp));

    EXPECT_EQ(out_fst, -2.25f) << "fst stored ST(0) without popping";
    EXPECT_EQ(out_fstp, -2.25f) << "fstp stored the same ST(0) then popped";
    EXPECT_EQ(r.state.fpu_top, 0u) << "exactly one net pop";
    EXPECT_EQ(r.state.exit_reason, static_cast<u32>(ExitReason::BlockEnd));
}

// fild dword [rdi]; fistp dword [rdi+16]; ret — integer load/store round trip.
TEST_F(CpuRuntimeTest, X87_FildFistp_Int32RoundTrip) {
    u8* data = mem.CodePtr() + kX87ScratchOff;
    const int32_t in = -12345;
    std::memcpy(data, &in, sizeof(in));
    std::memset(data + 16, 0, sizeof(int32_t));

    u8 program[] = {
        0x48, 0xbf, 0,0,0,0,0,0,0,0,   // mov rdi, <data addr>
        0xdb, 0x07,                     // fild  dword [rdi]
        0xdb, 0x5f, 0x10,               // fistp dword [rdi+0x10]
        0xc3,
    };
    const u64 data_addr = reinterpret_cast<u64>(data);
    std::memcpy(&program[2], &data_addr, sizeof(data_addr));

    const auto r = RunProgram(program, sizeof(program), mem);
    int32_t out = 0;
    std::memcpy(&out, data + 16, sizeof(out));

    EXPECT_EQ(out, -12345) << "int32 round-trips through fild/fistp";
    EXPECT_EQ(r.state.fpu_top, 0u);
    EXPECT_EQ(r.state.exit_reason, static_cast<u32>(ExitReason::BlockEnd));
}

// fild qword [rdi]; fistp qword [rdi+16]; ret — 64-bit integer round trip.
TEST_F(CpuRuntimeTest, X87_FildFistp_Int64RoundTrip) {
    u8* data = mem.CodePtr() + kX87ScratchOff;
    const int64_t in = -1234567890123LL;
    std::memcpy(data, &in, sizeof(in));
    std::memset(data + 16, 0, sizeof(int64_t));

    u8 program[] = {
        0x48, 0xbf, 0,0,0,0,0,0,0,0,   // mov rdi, <data addr>
        0xdf, 0x2f,                     // fild  qword [rdi]
        0xdf, 0x7f, 0x10,               // fistp qword [rdi+0x10]
        0xc3,
    };
    const u64 data_addr = reinterpret_cast<u64>(data);
    std::memcpy(&program[2], &data_addr, sizeof(data_addr));

    const auto r = RunProgram(program, sizeof(program), mem);
    int64_t out = 0;
    std::memcpy(&out, data + 16, sizeof(out));

    EXPECT_EQ(out, -1234567890123LL) << "int64 round-trips through fild/fistp";
    EXPECT_EQ(r.state.fpu_top, 0u);
    EXPECT_EQ(r.state.exit_reason, static_cast<u32>(ExitReason::BlockEnd));
}

// Two pushes then two pops: verify the stack is LIFO and top wraps
// correctly. fld A; fld B; fstp -> B; fstp -> A.
TEST_F(CpuRuntimeTest, X87_TwoLevel_StackIsLIFO) {
    u8* data = mem.CodePtr() + kX87ScratchOff;
    const float a = 10.0f, b = 20.0f;
    std::memcpy(data + 0, &a, sizeof(a));   // [rdi+0]  = A
    std::memcpy(data + 4, &b, sizeof(b));   // [rdi+4]  = B
    std::memset(data + 16, 0, sizeof(float));
    std::memset(data + 20, 0, sizeof(float));

    u8 program[] = {
        0x48, 0xbf, 0,0,0,0,0,0,0,0,   // mov rdi, <data addr>
        0xd9, 0x07,                     // fld  dword [rdi]       push A  -> ST0=A
        0xd9, 0x47, 0x04,               // fld  dword [rdi+4]     push B  -> ST0=B,ST1=A
        0xd9, 0x5f, 0x10,               // fstp dword [rdi+0x10]  pop->B
        0xd9, 0x5f, 0x14,               // fstp dword [rdi+0x14]  pop->A
        0xc3,
    };
    const u64 data_addr = reinterpret_cast<u64>(data);
    std::memcpy(&program[2], &data_addr, sizeof(data_addr));

    const auto r = RunProgram(program, sizeof(program), mem);
    float first = 0, second = 0;
    std::memcpy(&first, data + 16, sizeof(first));   // first pop == B (LIFO)
    std::memcpy(&second, data + 20, sizeof(second)); // second pop == A

    EXPECT_EQ(first, 20.0f) << "first pop returns most-recently pushed (B)";
    EXPECT_EQ(second, 10.0f) << "second pop returns A";
    EXPECT_EQ(r.state.fpu_top, 0u) << "two pushes + two pops nets to 0";
    EXPECT_EQ(r.state.fpu_tag & 0x3u, 0u) << "both used slots empty again";
    EXPECT_EQ(r.state.exit_reason, static_cast<u32>(ExitReason::BlockEnd));
}

// and rsi, qword [rdi]; ret
// 64-bit AND with a memory source (reg destination). This is the form
// that blocked the game's main entry path. Verify the mask is applied
// and the upper bits clear correctly.
TEST_F(CpuRuntimeTest, And64_RegMemSource_MasksCorrectly) {
    u8* data = mem.CodePtr() + 1024;
    const u64 mask = 0x00000000FFFF0000ULL;
    std::memcpy(data, &mask, sizeof(mask));

    u8 program[] = {
        0x48, 0xbf, 0,0,0,0,0,0,0,0,   // mov rdi, <data addr> (imm @2..9)
        0x48, 0xbe, 0,0,0,0,0,0,0,0,   // mov rsi, 0x123456789ABCDEF0 (imm @12..19)
        0x48, 0x23, 0x37,              // and rsi, qword [rdi]
        0xc3,                          // ret
    };
    const u64 data_addr = reinterpret_cast<u64>(data);
    const u64 rsi_init  = 0x123456789ABCDEF0ULL;
    std::memcpy(&program[2], &data_addr, sizeof(data_addr));
    std::memcpy(&program[12], &rsi_init, sizeof(rsi_init));

    const auto r = RunProgram(program, sizeof(program), mem);

    EXPECT_EQ(r.state.gpr[6], rsi_init & mask) << "rsi = rsi & [rdi]";
    EXPECT_EQ(r.state.exit_reason, static_cast<u32>(ExitReason::BlockEnd));
}

// cmovz ecx, edx with ZF=1 (so move happens): set ZF via xor eax,eax.
// 32-bit cmov zero-extends. Verify ecx takes edx and upper bits clear.
TEST_F(CpuRuntimeTest, Cmovz32_TakesSource_WhenZeroFlagSet) {
    u8 program[] = {
        0x48, 0xba, 0,0,0,0,0,0,0,0,   // mov rdx, 0xAAAAAAAABBBBBBBB (imm @2)
        0x48, 0xb9, 0,0,0,0,0,0,0,0,   // mov rcx, 0x1111111122222222 (imm @12)
        0x31, 0xc0,                    // xor eax, eax  (sets ZF=1)
        0x0f, 0x44, 0xca,              // cmovz ecx, edx
        0xc3,                          // ret
    };
    const u64 rdx_init = 0xAAAAAAAABBBBBBBBULL;
    const u64 rcx_init = 0x1111111122222222ULL;
    std::memcpy(&program[2],  &rdx_init, sizeof(rdx_init));
    std::memcpy(&program[12], &rcx_init, sizeof(rcx_init));

    const auto r = RunProgram(program, sizeof(program), mem);

    // Move taken; 32-bit op zero-extends, so ecx = 0xBBBBBBBB, upper = 0.
    EXPECT_EQ(r.state.gpr[1], 0x00000000BBBBBBBBULL) << "ecx = edx, zero-extended";
    EXPECT_EQ(r.state.exit_reason, static_cast<u32>(ExitReason::BlockEnd));
}

// imul esi, edi, 0x1234 — 3-operand 32-bit immediate multiply.
TEST_F(CpuRuntimeTest, Imul3Op32_MultipliesByImmediate) {
    u8 program[] = {
        0x48, 0xbf, 0,0,0,0,0,0,0,0,   // mov rdi, 0x10 (imm @2)
        0x69, 0xf7, 0x34, 0x12, 0x00, 0x00,  // imul esi, edi, 0x1234
        0xc3,                          // ret
    };
    const u64 rdi_init = 0x10ULL;
    std::memcpy(&program[2], &rdi_init, sizeof(rdi_init));

    const auto r = RunProgram(program, sizeof(program), mem);

    EXPECT_EQ(r.state.gpr[6], static_cast<u64>(0x10u * 0x1234u))
        << "esi = edi * 0x1234, zero-extended";
    EXPECT_EQ(r.state.exit_reason, static_cast<u32>(ExitReason::BlockEnd));
}

// movsx eax, cl (8->32) with a negative byte: 0xFF -> 0xFFFFFFFF,
// then zero-extended into rax => 0x00000000FFFFFFFF.
TEST_F(CpuRuntimeTest, Movsx_Byte8ToReg32_SignExtends) {
    u8 program[] = {
        0x48, 0xb9, 0,0,0,0,0,0,0,0,   // mov rcx, 0x...FF (imm @2)
        0x0f, 0xbe, 0xc1,              // movsx eax, cl
        0xc3,                          // ret
    };
    const u64 rcx_init = 0x12345678000000FFULL;  // cl = 0xFF
    std::memcpy(&program[2], &rcx_init, sizeof(rcx_init));

    const auto r = RunProgram(program, sizeof(program), mem);

    // cl=0xFF sign-extends to 0xFFFFFFFF in eax; 32-bit write zero-extends.
    EXPECT_EQ(r.state.gpr[0], 0x00000000FFFFFFFFULL) << "eax = sext(cl)";
    EXPECT_EQ(r.state.exit_reason, static_cast<u32>(ExitReason::BlockEnd));
}

// movsx rax, cl (8->64) with negative byte: 0xFF -> full 0xFFFF...FF.
TEST_F(CpuRuntimeTest, Movsx_Byte8ToReg64_SignExtends) {
    u8 program[] = {
        0x48, 0xb9, 0,0,0,0,0,0,0,0,   // mov rcx, 0x...80 (imm @2)
        0x48, 0x0f, 0xbe, 0xc1,        // movsx rax, cl
        0xc3,                          // ret
    };
    const u64 rcx_init = 0x0000000000000080ULL;  // cl = 0x80 (negative)
    std::memcpy(&program[2], &rcx_init, sizeof(rcx_init));

    const auto r = RunProgram(program, sizeof(program), mem);

    EXPECT_EQ(r.state.gpr[0], 0xFFFFFFFFFFFFFF80ULL) << "rax = sext64(cl)";
    EXPECT_EQ(r.state.exit_reason, static_cast<u32>(ExitReason::BlockEnd));
}

// andn rax, rbx, rcx — dst = (~rbx) & rcx (BMI1).
TEST_F(CpuRuntimeTest, Andn64_ComputesNotSrc1AndSrc2) {
    u8 program[] = {
        0x48, 0xbb, 0,0,0,0,0,0,0,0,   // mov rbx, src1 (imm @2)
        0x48, 0xb9, 0,0,0,0,0,0,0,0,   // mov rcx, src2 (imm @12)
        0xc4, 0xe2, 0xe0, 0xf2, 0xc1,  // andn rax, rbx, rcx
        0xc3,                          // ret
    };
    const u64 src1 = 0x00000000FFFF0000ULL;
    const u64 src2 = 0xFFFFFFFFFFFFFFFFULL;
    std::memcpy(&program[2],  &src1, sizeof(src1));
    std::memcpy(&program[12], &src2, sizeof(src2));

    const auto r = RunProgram(program, sizeof(program), mem);

    EXPECT_EQ(r.state.gpr[0], (~src1) & src2) << "rax = (~rbx) & rcx";
    EXPECT_EQ(r.state.exit_reason, static_cast<u32>(ExitReason::BlockEnd));
}

// add rsi, qword [rdi] — 64-bit ADD with a memory source.
TEST_F(CpuRuntimeTest, Add64_RegMemSource_AddsAndStores) {
    u8* data = mem.CodePtr() + 1024;
    const u64 addend = 0x0000000000001000ULL;
    std::memcpy(data, &addend, sizeof(addend));

    u8 program[] = {
        0x48, 0xbf, 0,0,0,0,0,0,0,0,   // mov rdi, <data> (imm @2)
        0x48, 0xbe, 0,0,0,0,0,0,0,0,   // mov rsi, 0x40 (imm @12)
        0x48, 0x03, 0x37,              // add rsi, qword [rdi]
        0xc3,                          // ret
    };
    const u64 data_addr = reinterpret_cast<u64>(data);
    const u64 rsi_init  = 0x40ULL;
    std::memcpy(&program[2],  &data_addr, sizeof(data_addr));
    std::memcpy(&program[12], &rsi_init,  sizeof(rsi_init));

    const auto r = RunProgram(program, sizeof(program), mem);

    EXPECT_EQ(r.state.gpr[6], rsi_init + addend) << "rsi = rsi + [rdi]";
    EXPECT_EQ(r.state.exit_reason, static_cast<u32>(ExitReason::BlockEnd));
}

// lea eax, [rdi+0x10] — 32-bit LEA: full-width address arithmetic,
// result truncated to 32 bits and zero-extended into rax.
TEST_F(CpuRuntimeTest, Lea32_ComputesAddrZeroExtended) {
    u8 program[] = {
        0x48, 0xbf, 0,0,0,0,0,0,0,0,   // mov rdi, 0x1_2345_6000 (imm @2)
        0x8d, 0x47, 0x10,              // lea eax, [rdi+0x10]
        0xc3,                          // ret
    };
    // Pick a base whose +0x10 has a nonzero bit above 32 to prove truncation.
    const u64 rdi_init = 0x0000000123456000ULL;
    std::memcpy(&program[2], &rdi_init, sizeof(rdi_init));

    const auto r = RunProgram(program, sizeof(program), mem);

    const u64 full = rdi_init + 0x10;             // 0x123456010
    EXPECT_EQ(r.state.gpr[0], full & 0xFFFFFFFFULL)
        << "eax = low 32 of (rdi+0x10), upper 32 zeroed";
    EXPECT_EQ(r.state.exit_reason, static_cast<u32>(ExitReason::BlockEnd));
}

// vmovdqa xmm0, [rdi] — aligned 128-bit load. Treated identically to
// vmovdqu in the JIT (the host load handles addressing). Verify the
// 128 bits land in xmm0 (ymm[0]/ymm[1]) and the upper YMM half zeroes.
TEST_F(CpuRuntimeTest, Vmovdqa_LoadsXmmFromMemory) {
    u8* guest_rsp = mem.StackTop() - 8;
    *reinterpret_cast<u64*>(guest_rsp) = kReturnSentinel;

    // 16-byte aligned data region.
    u8* data = mem.CodePtr() + 2048;
    const u64 lo = 0x1122334455667788ULL;
    const u64 hi = 0x99AABBCCDDEEFF00ULL;
    std::memcpy(data,     &lo, sizeof(lo));
    std::memcpy(data + 8, &hi, sizeof(hi));

    u8 program[] = {
        0x48, 0xbf, 0,0,0,0,0,0,0,0,   // mov rdi, <data> (imm @2)
        0xc5, 0xf9, 0x6f, 0x07,        // vmovdqa xmm0, [rdi]
        0xc3,                          // ret
    };
    const u64 data_addr = reinterpret_cast<u64>(data);
    std::memcpy(&program[2], &data_addr, sizeof(data_addr));
    std::memcpy(mem.CodePtr(), program, sizeof(program));

    GuestState state{};
    state.rip = reinterpret_cast<u64>(mem.CodePtr());
    state.gpr[4] = reinterpret_cast<u64>(guest_rsp);
    // Dirty xmm0's upper YMM half to confirm VEX-128 zeroes it.
    state.ymm[2] = 0xFFFFFFFFFFFFFFFFull;
    state.ymm[3] = 0xFFFFFFFFFFFFFFFFull;

    Runtime rt;
    rt.Run(state);

    EXPECT_EQ(state.ymm[0], lo) << "xmm0 low 64";
    EXPECT_EQ(state.ymm[1], hi) << "xmm0 high 64";
    EXPECT_EQ(state.ymm[2], 0u) << "VEX-128 zeroes bits 255:128";
    EXPECT_EQ(state.ymm[3], 0u);
    EXPECT_EQ(state.rip, kReturnSentinel);
}

// bextr rax, rcx, rdx — BMI1 bit-field extract.
// control rdx = start | (len << 8). Extract len bits of rcx starting
// at bit `start`. Here start=4, len=8 → bits [11:4] of src.
TEST_F(CpuRuntimeTest, Bextr64_ExtractsBitField) {
    u8 program[] = {
        0x48, 0xb9, 0,0,0,0,0,0,0,0,   // mov rcx, src     (imm @2)
        0x48, 0xba, 0,0,0,0,0,0,0,0,   // mov rdx, control (imm @12)
        0xc4, 0xe2, 0xe8, 0xf7, 0xc1,  // bextr rax, rcx, rdx
        0xc3,                          // ret
    };
    const u64 src     = 0xFFFFFFFFFFFFFABCULL;  // bits [11:4] = 0xAB
    const u64 control = 4 | (8u << 8);          // start=4, len=8
    std::memcpy(&program[2],  &src,     sizeof(src));
    std::memcpy(&program[12], &control, sizeof(control));

    const auto r = RunProgram(program, sizeof(program), mem);

    const u64 expected = (src >> 4) & ((1ULL << 8) - 1);  // = 0xAB
    EXPECT_EQ(r.state.gpr[0], expected) << "rax = bextr(rcx, start=4, len=8)";
    EXPECT_EQ(r.state.exit_reason, static_cast<u32>(ExitReason::BlockEnd));
}

// xor rax, 0x1234 — 64-bit XOR with an immediate source.
TEST_F(CpuRuntimeTest, Xor64_Imm_FlipsBits) {
    u8 program[] = {
        0x48, 0xb8, 0,0,0,0,0,0,0,0,   // mov rax, init (imm @2)
        0x48, 0x35, 0x34, 0x12, 0x00, 0x00,  // xor rax, 0x1234
        0xc3,                          // ret
    };
    const u64 init = 0x00000000000000FFULL;
    std::memcpy(&program[2], &init, sizeof(init));

    const auto r = RunProgram(program, sizeof(program), mem);

    EXPECT_EQ(r.state.gpr[0], init ^ 0x1234ULL) << "rax ^= 0x1234";
    EXPECT_EQ(r.state.exit_reason, static_cast<u32>(ExitReason::BlockEnd));
}

// shr cl, 3 — 8-bit shift by immediate. Upper 56 bits of rcx must be
// preserved (8-bit writes don't zero-extend).
TEST_F(CpuRuntimeTest, Shr8_Imm_PreservesUpperBits) {
    u8 program[] = {
        0x48, 0xb9, 0,0,0,0,0,0,0,0,   // mov rcx, init (imm @2)
        0xc0, 0xe9, 0x03,              // shr cl, 3
        0xc3,                          // ret
    };
    const u64 init = 0xAABBCCDDEEFF0080ULL;  // cl = 0x80
    std::memcpy(&program[2], &init, sizeof(init));

    const auto r = RunProgram(program, sizeof(program), mem);

    const u8 shifted = static_cast<u8>(0x80u >> 3);   // = 0x10
    const u64 expected = (init & ~0xFFULL) | shifted; // upper 56 preserved
    EXPECT_EQ(r.state.gpr[1], expected) << "cl >>= 3, upper bits intact";
    EXPECT_EQ(r.state.exit_reason, static_cast<u32>(ExitReason::BlockEnd));
}

// vpcmpeqb xmm0, xmm1, xmm2 — packed byte-equality compare. Each byte
// equal → 0xFF, else 0x00; VEX-128 zeroes the upper YMM half.
TEST_F(CpuRuntimeTest, Vpcmpeqb_ComparesBytesPerLane) {
    u8* guest_rsp = mem.StackTop() - 8;
    *reinterpret_cast<u64*>(guest_rsp) = kReturnSentinel;

    const u8 program[] = {
        0xc5, 0xf1, 0x74, 0xc2,  // vpcmpeqb xmm0, xmm1, xmm2
        0xc3,                    // ret
    };
    std::memcpy(mem.CodePtr(), program, sizeof(program));

    GuestState state{};
    state.rip = reinterpret_cast<u64>(mem.CodePtr());
    state.gpr[4] = reinterpret_cast<u64>(guest_rsp);

    // Each XMM occupies 4 u64 slots (32 bytes): xmm_n low/high =
    // ymm[n*4 + 0]/[n*4 + 1], with [n*4 + 2]/[+3] the upper 128.
    // xmm1 = ymm[4]/[5], xmm2 = ymm[8]/[9], dst xmm0 = ymm[0]/[1].
    // byte lanes of xmm1 low: 00 11 22 33 44 55 66 77
    state.ymm[4] = 0x7766554433221100ULL;  // xmm1 low
    state.ymm[5] = 0x0000000000000000ULL;  // xmm1 high
    // xmm2 low: match even byte indices, differ on odd ones.
    {
        auto byte_of = [](u64 v, int i){ return static_cast<u8>((v >> (i*8)) & 0xFF); };
        u64 a = state.ymm[4];
        u64 b = 0;
        for (int i = 0; i < 8; ++i) {
            u8 ba = byte_of(a, i);
            u8 bb = (i % 2 == 0) ? ba : static_cast<u8>(ba ^ 0xFF);
            b |= static_cast<u64>(bb) << (i*8);
        }
        state.ymm[8] = b;
    }
    state.ymm[9] = 0x0000000000000000ULL;  // xmm2 high
    // Dirty xmm0 (dst) low and its upper-128 half to confirm overwrite
    // and VEX-128 zeroing.
    state.ymm[0] = 0xDEADBEEFDEADBEEFULL;
    state.ymm[1] = 0xCAFEBABECAFEBABEULL;
    state.ymm[2] = 0xFFFFFFFFFFFFFFFFULL;
    state.ymm[3] = 0xFFFFFFFFFFFFFFFFULL;

    Runtime rt;
    rt.Run(state);

    // Expected low 64: even byte lanes equal → 0xFF, odd → 0x00.
    u64 expected_lo = 0;
    for (int i = 0; i < 8; ++i)
        if (i % 2 == 0) expected_lo |= static_cast<u64>(0xFF) << (i*8);
    EXPECT_EQ(state.ymm[0], expected_lo) << "byte-equal lanes -> 0xFF";
    // xmm1 high (ymm[5]) and xmm2 high (ymm[9]) are both 0 → every byte
    // equal → 0xFF across the whole high 64.
    EXPECT_EQ(state.ymm[1], 0xFFFFFFFFFFFFFFFFULL) << "equal zero bytes -> 0xFF";
    // VEX-128 zeroes bits 255:128 of xmm0.
    EXPECT_EQ(state.ymm[2], 0ULL) << "upper 128 zeroed";
    EXPECT_EQ(state.ymm[3], 0ULL);
    EXPECT_EQ(state.rip, kReturnSentinel);
}

// setnz dl with ZF=0 → dl=1; upper 56 bits of rdx preserved.
// Drive ZF=0 via `test eax,eax` on a non-zero eax.
TEST_F(CpuRuntimeTest, Setnz_RegDst_ConditionTrue_PreservesUpper) {
    u8 program[] = {
        0x48, 0xba, 0,0,0,0,0,0,0,0,   // mov rdx, poison (imm @2)
        0x48, 0xb8, 0x01, 0,0,0,0,0,0,0, // mov rax, 1 (imm @12) -> eax nonzero
        0x85, 0xc0,                    // test eax, eax  (ZF=0)
        0x0f, 0x95, 0xc2,              // setnz dl
        0xc3,                          // ret
    };
    const u64 poison = 0xDEADBEEFCAFE0000ULL;  // dl currently 0
    const u64 raxval = 0x0000000000000001ULL;
    std::memcpy(&program[2],  &poison, sizeof(poison));
    std::memcpy(&program[12], &raxval, sizeof(raxval));

    const auto r = RunProgram(program, sizeof(program), mem);

    EXPECT_EQ(r.state.gpr[2] & 0xFFULL, 0x01ULL) << "dl = 1 (ZF was 0)";
    EXPECT_EQ(r.state.gpr[2] & ~0xFFULL, 0xDEADBEEFCAFE0000ULL)
        << "upper 56 bits preserved";
    EXPECT_EQ(r.state.exit_reason, static_cast<u32>(ExitReason::BlockEnd));
}

// cmpxchg [rdi], ecx — MATCH case. EAX == [rdi] → ZF=1, [rdi] = ecx,
// EAX unchanged.
TEST_F(CpuRuntimeTest, Cmpxchg32_MemDst_Match_StoresSrcSetsZf) {
    u8* data = mem.CodePtr() + 1024;
    const u32 mem_init = 0x11112222u;
    std::memcpy(data, &mem_init, sizeof(mem_init));

    u8 program[] = {
        0x48, 0xbf, 0,0,0,0,0,0,0,0,   // mov rdi, <data> (imm @2)
        0x48, 0xb8, 0x22, 0x22, 0x11, 0x11, 0,0,0,0, // mov rax, 0x11112222 (eax == mem)
        0x48, 0xb9, 0x99, 0x88, 0x77, 0x66, 0,0,0,0, // mov rcx, 0x66778899 (src)
        0x0f, 0xb1, 0x0f,              // cmpxchg [rdi], ecx
        0xc3,
    };
    const u64 data_addr = reinterpret_cast<u64>(data);
    std::memcpy(&program[2], &data_addr, sizeof(data_addr));

    const auto r = RunProgram(program, sizeof(program), mem);

    u32 mem_after; std::memcpy(&mem_after, data, sizeof(mem_after));
    EXPECT_EQ(mem_after, 0x66778899u) << "match → [rdi] = ecx";
    EXPECT_TRUE(r.state.rflags & (1ULL << 6)) << "ZF=1 on match";
    EXPECT_EQ(r.state.gpr[0] & 0xFFFFFFFFULL, 0x11112222u) << "eax unchanged on match";
    EXPECT_EQ(r.state.exit_reason, static_cast<u32>(ExitReason::BlockEnd));
}

// cmpxchg [rdi], ecx — MISMATCH case. EAX != [rdi] → ZF=0, EAX = [rdi],
// memory unchanged.
TEST_F(CpuRuntimeTest, Cmpxchg32_MemDst_Mismatch_LoadsAccSetsNoZf) {
    u8* data = mem.CodePtr() + 1024;
    const u32 mem_init = 0x11112222u;
    std::memcpy(data, &mem_init, sizeof(mem_init));

    u8 program[] = {
        0x48, 0xbf, 0,0,0,0,0,0,0,0,   // mov rdi, <data> (imm @2)
        0x48, 0xb8, 0x00, 0x00, 0x00, 0x00, 0,0,0,0, // mov rax, 0 (eax != mem)
        0x48, 0xb9, 0x99, 0x88, 0x77, 0x66, 0,0,0,0, // mov rcx, 0x66778899 (src)
        0x0f, 0xb1, 0x0f,              // cmpxchg [rdi], ecx
        0xc3,
    };
    const u64 data_addr = reinterpret_cast<u64>(data);
    std::memcpy(&program[2], &data_addr, sizeof(data_addr));

    const auto r = RunProgram(program, sizeof(program), mem);

    u32 mem_after; std::memcpy(&mem_after, data, sizeof(mem_after));
    EXPECT_EQ(mem_after, 0x11112222u) << "mismatch → memory unchanged";
    EXPECT_FALSE(r.state.rflags & (1ULL << 6)) << "ZF=0 on mismatch";
    EXPECT_EQ(r.state.gpr[0] & 0xFFFFFFFFULL, 0x11112222u) << "eax = [rdi] on mismatch";
    EXPECT_EQ(r.state.exit_reason, static_cast<u32>(ExitReason::BlockEnd));
}

// cmp byte [rdi], cl — 8-bit compare with memory lhs, register rhs.
// Equal bytes → ZF=1; memory is not modified.
TEST_F(CpuRuntimeTest, Cmp8_MemDst_RegSrc_EqualSetsZf) {
    u8* data = mem.CodePtr() + 1024;
    *data = 0x42;

    u8 program[] = {
        0x48, 0xbf, 0,0,0,0,0,0,0,0,   // mov rdi, <data> (imm @2)
        0x48, 0xb9, 0x42, 0,0,0, 0,0,0,0, // mov rcx, 0x42 (cl = 0x42)
        0x38, 0x0f,                    // cmp byte [rdi], cl
        0xc3,
    };
    const u64 data_addr = reinterpret_cast<u64>(data);
    std::memcpy(&program[2], &data_addr, sizeof(data_addr));

    const auto r = RunProgram(program, sizeof(program), mem);

    EXPECT_TRUE(r.state.rflags & (1ULL << 6)) << "ZF=1 (bytes equal)";
    EXPECT_EQ(*data, 0x42) << "memory unchanged by cmp";
    EXPECT_EQ(r.state.exit_reason, static_cast<u32>(ExitReason::BlockEnd));
}

// cmp byte [rdi], cl — unequal: [mem]=0x42, cl=0x40 → [mem] > rhs,
// ZF=0, CF=0 (0x42 - 0x40 = 2, no borrow).
TEST_F(CpuRuntimeTest, Cmp8_MemDst_RegSrc_UnequalClearsZf) {
    u8* data = mem.CodePtr() + 1024;
    *data = 0x42;

    u8 program[] = {
        0x48, 0xbf, 0,0,0,0,0,0,0,0,   // mov rdi, <data>
        0x48, 0xb9, 0x40, 0,0,0, 0,0,0,0, // mov rcx, 0x40 (cl=0x40)
        0x38, 0x0f,                    // cmp byte [rdi], cl
        0xc3,
    };
    const u64 data_addr = reinterpret_cast<u64>(data);
    std::memcpy(&program[2], &data_addr, sizeof(data_addr));

    const auto r = RunProgram(program, sizeof(program), mem);

    EXPECT_FALSE(r.state.rflags & (1ULL << 6)) << "ZF=0 (bytes differ)";
    EXPECT_FALSE(r.state.rflags & 1ULL) << "CF=0 (0x42 >= 0x40)";
    EXPECT_EQ(r.state.exit_reason, static_cast<u32>(ExitReason::BlockEnd));
}

// or byte [rdi], 0x80 — 8-bit OR with memory destination + immediate,
// the bitfield-set idiom. Result written back; CF/OF cleared.
TEST_F(CpuRuntimeTest, Or8_MemDst_Imm_SetsBitAndWritesBack) {
    u8* data = mem.CodePtr() + 1024;
    *data = 0x01;

    u8 program[] = {
        0x48, 0xbf, 0,0,0,0,0,0,0,0,   // mov rdi, <data> (imm @2)
        0x80, 0x0f, 0x80,              // or byte [rdi], 0x80
        0xc3,
    };
    const u64 data_addr = reinterpret_cast<u64>(data);
    std::memcpy(&program[2], &data_addr, sizeof(data_addr));

    const auto r = RunProgram(program, sizeof(program), mem);

    EXPECT_EQ(*data, 0x81) << "0x01 | 0x80 = 0x81 written to [rdi]";
    EXPECT_FALSE(r.state.rflags & 1ULL)        << "CF cleared by OR";
    EXPECT_FALSE(r.state.rflags & (1ULL << 11)) << "OF cleared by OR";
    EXPECT_EQ(r.state.exit_reason, static_cast<u32>(ExitReason::BlockEnd));
}

// cmp byte [rdi], ah — high-byte register as rhs. AH is bits 15:8 of
// RAX, which ZydisGprToIndex rejects; the byte-offset path handles it.
// This is the exact form that kept exiting the JIT at guest 0x806043525.
TEST_F(CpuRuntimeTest, Cmp8_MemDst_HighByteReg_EqualSetsZf) {
    u8* data = mem.CodePtr() + 1024;
    *data = 0x42;

    u8 program[] = {
        0x48, 0xbf, 0,0,0,0,0,0,0,0,   // mov rdi, <data> (imm @2)
        0x48, 0xb8, 0x00, 0x42, 0,0,0,0,0,0,  // mov rax, 0x4200 → AH=0x42
        0x38, 0x27,                    // cmp byte [rdi], ah
        0xc3,
    };
    const u64 data_addr = reinterpret_cast<u64>(data);
    const u64 raxval = 0x0000000000004200ULL;
    std::memcpy(&program[2],  &data_addr, sizeof(data_addr));
    std::memcpy(&program[12], &raxval,    sizeof(raxval));

    const auto r = RunProgram(program, sizeof(program), mem);

    EXPECT_TRUE(r.state.rflags & (1ULL << 6)) << "ZF=1 ([mem]==AH)";
    EXPECT_EQ(*data, 0x42) << "memory unchanged by cmp";
    EXPECT_EQ(r.state.exit_reason, static_cast<u32>(ExitReason::BlockEnd));
}

// Same form, unequal: [mem]=0x42, AH=0x40 → ZF=0, CF=0.
TEST_F(CpuRuntimeTest, Cmp8_MemDst_HighByteReg_UnequalClearsZf) {
    u8* data = mem.CodePtr() + 1024;
    *data = 0x42;

    u8 program[] = {
        0x48, 0xbf, 0,0,0,0,0,0,0,0,   // mov rdi, <data>
        0x48, 0xb8, 0x00, 0x40, 0,0,0,0,0,0,  // mov rax, 0x4000 → AH=0x40
        0x38, 0x27,                    // cmp byte [rdi], ah
        0xc3,
    };
    const u64 data_addr = reinterpret_cast<u64>(data);
    const u64 raxval = 0x0000000000004000ULL;
    std::memcpy(&program[2],  &data_addr, sizeof(data_addr));
    std::memcpy(&program[12], &raxval,    sizeof(raxval));

    const auto r = RunProgram(program, sizeof(program), mem);

    EXPECT_FALSE(r.state.rflags & (1ULL << 6)) << "ZF=0 (0x42 != 0x40)";
    EXPECT_FALSE(r.state.rflags & 1ULL) << "CF=0 (0x42 >= 0x40)";
    EXPECT_EQ(r.state.exit_reason, static_cast<u32>(ExitReason::BlockEnd));
}

// sub rsi, qword [rdi] — 64-bit SUB with a memory source.
TEST_F(CpuRuntimeTest, Sub64_RegMemSource_SubtractsAndStores) {
    u8* data = mem.CodePtr() + 1024;
    const u64 subtrahend = 0x0000000000000300ULL;
    std::memcpy(data, &subtrahend, sizeof(subtrahend));

    u8 program[] = {
        0x48, 0xbf, 0,0,0,0,0,0,0,0,   // mov rdi, <data> (imm @2)
        0x48, 0xbe, 0,0,0,0,0,0,0,0,   // mov rsi, 0x1000 (imm @12)
        0x48, 0x2b, 0x37,              // sub rsi, qword [rdi]
        0xc3,
    };
    const u64 data_addr = reinterpret_cast<u64>(data);
    const u64 rsi_init  = 0x1000ULL;
    std::memcpy(&program[2],  &data_addr, sizeof(data_addr));
    std::memcpy(&program[12], &rsi_init,  sizeof(rsi_init));

    const auto r = RunProgram(program, sizeof(program), mem);

    EXPECT_EQ(r.state.gpr[6], rsi_init - subtrahend) << "rsi = rsi - [rdi]";
    EXPECT_FALSE(r.state.rflags & 1ULL) << "CF=0 (0x1000 >= 0x300)";
    EXPECT_EQ(r.state.exit_reason, static_cast<u32>(ExitReason::BlockEnd));
}

// or r64, imm — 64-bit register OR with a sign-extended immediate.
// Encoding `49 83 c9 01` (or r9, 1) is the exact length-4 form the
// game emitted. OR clears CF/OF and sets ZF/SF/PF from the result.
TEST_F(CpuRuntimeTest, Or64_RegImm_SetsBitsAndClearsCf) {
    u8 program[] = {
        // mov r9, 0x1000_0000_0000_0000
        0x49, 0xb9, 0,0,0,0,0,0,0,0x10,
        0x49, 0x83, 0xc9, 0x01,   // or r9, 1
        0xc3,
    };
    const u64 r9_init = 0x1000000000000000ULL;
    std::memcpy(&program[2], &r9_init, sizeof(r9_init));

    const auto r = RunProgram(program, sizeof(program), mem);

    EXPECT_EQ(r.state.gpr[9], 0x1000000000000001ULL) << "bit 0 set, others preserved";
    EXPECT_FALSE(r.state.rflags & 1ULL)         << "CF cleared by OR";
    EXPECT_FALSE(r.state.rflags & (1ULL << 11)) << "OF cleared by OR";
    EXPECT_FALSE(r.state.rflags & (1ULL << 6))  << "ZF clear (result != 0)";
    EXPECT_EQ(r.state.exit_reason, static_cast<u32>(ExitReason::BlockEnd));
}

// or r64, imm32 with the immediate's high bit set — sign-extends to
// flip the entire upper 32 bits as well.
TEST_F(CpuRuntimeTest, Or64_RegImm32_SignExtends) {
    u8 program[] = {
        0x48, 0xb8, 0,0,0,0,0,0,0,0,   // mov rax, 0
        0x48, 0x0d, 0x00, 0x00, 0x00, 0x80,  // or rax, 0xFFFFFFFF80000000 (imm32 sx)
        0xc3,
    };
    const u64 rax_init = 0;
    std::memcpy(&program[2], &rax_init, sizeof(rax_init));

    const auto r = RunProgram(program, sizeof(program), mem);

    EXPECT_EQ(r.state.gpr[0], 0xFFFFFFFF80000000ULL)
        << "imm32 0x80000000 sign-extends to set the upper 32 bits too";
    EXPECT_TRUE(r.state.rflags & (1ULL << 7)) << "SF set (bit 63 = 1)";
    EXPECT_EQ(r.state.exit_reason, static_cast<u32>(ExitReason::BlockEnd));
}

// Exact length-2 no-displacement form `fstp dword [rdi]` = D9 1F, the
// form observed at guest 0x80734d21e. Proves the memory-store path is
// reached for the bare [reg] addressing mode (not just disp8 forms).
TEST_F(CpuRuntimeTest, X87_Fstp_NoDisp_DwordStore) {
    u8* data = mem.CodePtr() + 0x300;
    *reinterpret_cast<float*>(data) = 0.0f;

    u8 program[] = {
        0x48, 0xbf, 0,0,0,0,0,0,0,0,   // mov rdi, <data>
        0xd9, 0x07,                     // fld   dword [rdi]  (load 0.0)
        0x48, 0xb8, 0,0,0,0,0,0,0,0,   // mov rax, <src float>
        0xc5, 0xfa, 0x10, 0x00,         // vmovss xmm0, [rax]
        // store the loaded ST0 elsewhere; but simplest: overwrite [rdi]
        // with a known value via the x87 stack. Reload then fstp.
        0xd9, 0x1f,                     // fstp dword [rdi]   (store ST0=0.0, pop)
        0xc3,
    };
    // Put 0.0 at data so fld pushes 0.0, then fstp writes it back.
    const u64 data_addr = reinterpret_cast<u64>(data);
    const u64 srcf = data_addr;  // reuse; not strictly needed
    std::memcpy(&program[2],  &data_addr, sizeof(data_addr));
    std::memcpy(&program[14], &srcf,      sizeof(srcf));

    const auto r = RunProgram(program, sizeof(program), mem);
    EXPECT_EQ(r.state.exit_reason, static_cast<u32>(ExitReason::BlockEnd))
        << "fstp dword [rdi] (D9 1F) is handled, block reaches ret";
}

// x87 80-bit extended-precision (`fstp tbyte` = DB 3A, `fld tbyte` =
// DB 2A) — the form observed at guest 0x80734d21e. Zydis reports
// operand_width=32 and op0.type=mem for it (identical to fstp dword in
// the trace), but op0.size==80, which the dword/qword path rejected.
// Round-trip: fld qword [src] (push double) ; fstp tbyte [scratch]
// (store 80-bit, pop) ; fld tbyte [scratch] (load 80-bit, push) ;
// fstp qword [dst] (store double, pop). The double must survive intact.
TEST_F(CpuRuntimeTest, X87_Fstp_Fld_Tbyte_RoundTrip) {
    u8* src     = mem.CodePtr() + 0x300;
    u8* scratch = mem.CodePtr() + 0x320;  // 10 bytes for the m80
    u8* dst     = mem.CodePtr() + 0x340;
    const double value = 3.14159265358979;
    std::memcpy(src, &value, sizeof(value));
    std::memset(dst, 0, sizeof(double));

    u8 program[] = {
        0x48, 0xbe, 0,0,0,0,0,0,0,0,   // mov rsi, <src>
        0x48, 0xbf, 0,0,0,0,0,0,0,0,   // mov rdi, <scratch>
        0x48, 0xba, 0,0,0,0,0,0,0,0,   // mov rdx_via_rbx? -> use rbx for dst
        0xdd, 0x06,                     // fld  qword [rsi]      (push double)
        0xdb, 0x3f,                     // fstp tbyte [rdi]      (store m80, pop)
        0xdb, 0x2f,                     // fld  tbyte [rdi]      (load m80, push)
        0xdd, 0x1a,                     // fstp qword [rdx]      (store double, pop)
        0xc3,
    };
    const u64 src_a = reinterpret_cast<u64>(src);
    const u64 scr_a = reinterpret_cast<u64>(scratch);
    const u64 dst_a = reinterpret_cast<u64>(dst);
    std::memcpy(&program[2],  &src_a, sizeof(src_a));
    std::memcpy(&program[12], &scr_a, sizeof(scr_a));
    std::memcpy(&program[22], &dst_a, sizeof(dst_a));

    const auto r = RunProgram(program, sizeof(program), mem);

    double out;
    std::memcpy(&out, dst, sizeof(out));
    EXPECT_EQ(out, value) << "double survives the 80-bit extended round-trip";
    EXPECT_EQ(r.state.exit_reason, static_cast<u32>(ExitReason::BlockEnd));
}

// x87 register-form `fld st(i)` (D9 C0+i) — push a copy of ST(i). The
// form observed at guest 0x80736d6e8 (op0=reg, hence "reg,reg" in the
// trace). Load 10.0 then 20.0 (so ST0=20, ST1=10); `fld st(1)` copies
// the value 10.0 to a new ST0. We then store the three live stack
// slots and confirm ST0=10 (the copy), ST1=20, ST2=10.
TEST_F(CpuRuntimeTest, X87_Fld_RegisterForm_DuplicatesStI) {
    u8* d0 = mem.CodePtr() + 0x300;
    u8* d1 = mem.CodePtr() + 0x310;
    u8* d2 = mem.CodePtr() + 0x320;
    u8* a  = mem.CodePtr() + 0x330;
    u8* b  = mem.CodePtr() + 0x340;
    const double va = 10.0, vb = 20.0;
    std::memcpy(a, &va, sizeof(va));
    std::memcpy(b, &vb, sizeof(vb));

    u8 program[] = {
        0x48, 0xb8, 0,0,0,0,0,0,0,0,    // mov rax, a
        0x48, 0xb9, 0,0,0,0,0,0,0,0,    // mov rcx, b
        0x48, 0xba, 0,0,0,0,0,0,0,0,    // mov rdx, d0
        0x48, 0xbe, 0,0,0,0,0,0,0,0,    // mov rsi, d1
        0x48, 0xbf, 0,0,0,0,0,0,0,0,    // mov rdi, d2
        0xdd, 0x00,                      // fld qword [rax]   (push 10 -> ST0=10)
        0xdd, 0x01,                      // fld qword [rcx]   (push 20 -> ST0=20,ST1=10)
        0xd9, 0xc1,                      // fld st(1)         (copy ST1=10 -> new ST0)
        // now ST0=10, ST1=20, ST2=10. Store and pop each:
        0xdd, 0x1a,                      // fstp qword [rdx]  (ST0=10 -> d0)
        0xdd, 0x1e,                      // fstp qword [rsi]  (ST0=20 -> d1)
        0xdd, 0x1f,                      // fstp qword [rdi]  (ST0=10 -> d2)
        0xc3,
    };
    const u64 aa=reinterpret_cast<u64>(a), ba=reinterpret_cast<u64>(b),
              d0a=reinterpret_cast<u64>(d0), d1a=reinterpret_cast<u64>(d1),
              d2a=reinterpret_cast<u64>(d2);
    std::memcpy(&program[2],  &aa,  8);
    std::memcpy(&program[12], &ba,  8);
    std::memcpy(&program[22], &d0a, 8);
    std::memcpy(&program[32], &d1a, 8);
    std::memcpy(&program[42], &d2a, 8);

    const auto r = RunProgram(program, sizeof(program), mem);

    double o0, o1, o2;
    std::memcpy(&o0, d0, 8);
    std::memcpy(&o1, d1, 8);
    std::memcpy(&o2, d2, 8);
    EXPECT_EQ(o0, 10.0) << "fld st(1) copied ST1 (=10) to new ST0";
    EXPECT_EQ(o1, 20.0) << "old ST0 (=20) is now ST1";
    EXPECT_EQ(o2, 10.0) << "original ST1 (=10) still present as ST2";
    EXPECT_EQ(r.state.exit_reason, static_cast<u32>(ExitReason::BlockEnd));
}

// x87 arithmetic-with-pop family (DE xx). Focus on the reversed forms
// (FSUBRP/FDIVRP), which are the easiest to get backwards, plus FADDP.
// Setup pushes A then B so ST0=B, ST1=A; the op writes ST1 and pops, so
// afterward ST0 holds the result. FSUBRP at guest 0x80736d769 motivated
// this; verify st(i)=st(0)-st(i), i.e. B-A reversed = ST0-ST1.
TEST_F(CpuRuntimeTest, X87_Fsubrp_ReverseSubtractAndPop) {
    u8* pa = mem.CodePtr() + 0x300;
    u8* pb = mem.CodePtr() + 0x310;
    u8* out = mem.CodePtr() + 0x320;
    const double A = 10.0, B = 20.0;
    std::memcpy(pa, &A, 8); std::memcpy(pb, &B, 8);

    u8 program[] = {
        0x48, 0xb8, 0,0,0,0,0,0,0,0,    // mov rax, pa
        0x48, 0xb9, 0,0,0,0,0,0,0,0,    // mov rcx, pb
        0x48, 0xba, 0,0,0,0,0,0,0,0,    // mov rdx, out
        0xdd, 0x00,                      // fld qword [rax]  (ST0=A=10)
        0xdd, 0x01,                      // fld qword [rcx]  (ST0=B=20, ST1=A=10)
        0xde, 0xe1,                      // fsubrp st1,st0   (ST1 = ST0-ST1 = 20-10=10; pop)
        0xdd, 0x1a,                      // fstp qword [rdx] (store result, pop)
        0xc3,
    };
    const u64 a=reinterpret_cast<u64>(pa),b=reinterpret_cast<u64>(pb),o=reinterpret_cast<u64>(out);
    std::memcpy(&program[2],&a,8); std::memcpy(&program[12],&b,8); std::memcpy(&program[22],&o,8);
    const auto r = RunProgram(program, sizeof(program), mem);
    double res; std::memcpy(&res, out, 8);
    EXPECT_EQ(res, 10.0) << "fsubrp: st(0)-st(i) = 20-10 = 10 (not -10)";
    EXPECT_EQ(r.state.exit_reason, static_cast<u32>(ExitReason::BlockEnd));
}

TEST_F(CpuRuntimeTest, X87_Fdivrp_ReverseDivideAndPop) {
    u8* pa = mem.CodePtr() + 0x300;
    u8* pb = mem.CodePtr() + 0x310;
    u8* out = mem.CodePtr() + 0x320;
    const double A = 10.0, B = 20.0;
    std::memcpy(pa, &A, 8); std::memcpy(pb, &B, 8);

    u8 program[] = {
        0x48, 0xb8, 0,0,0,0,0,0,0,0,
        0x48, 0xb9, 0,0,0,0,0,0,0,0,
        0x48, 0xba, 0,0,0,0,0,0,0,0,
        0xdd, 0x00,                      // ST0=A=10
        0xdd, 0x01,                      // ST0=B=20, ST1=A=10
        0xde, 0xf1,                      // fdivrp st1,st0 (ST1 = ST0/ST1 = 20/10=2; pop)
        0xdd, 0x1a,
        0xc3,
    };
    const u64 a=reinterpret_cast<u64>(pa),b=reinterpret_cast<u64>(pb),o=reinterpret_cast<u64>(out);
    std::memcpy(&program[2],&a,8); std::memcpy(&program[12],&b,8); std::memcpy(&program[22],&o,8);
    const auto r = RunProgram(program, sizeof(program), mem);
    double res; std::memcpy(&res, out, 8);
    EXPECT_EQ(res, 2.0) << "fdivrp: st(0)/st(i) = 20/10 = 2 (not 0.5)";
    EXPECT_EQ(r.state.exit_reason, static_cast<u32>(ExitReason::BlockEnd));
}

TEST_F(CpuRuntimeTest, X87_Faddp_AddAndPop) {
    u8* pa = mem.CodePtr() + 0x300;
    u8* pb = mem.CodePtr() + 0x310;
    u8* out = mem.CodePtr() + 0x320;
    const double A = 10.0, B = 20.0;
    std::memcpy(pa, &A, 8); std::memcpy(pb, &B, 8);

    u8 program[] = {
        0x48, 0xb8, 0,0,0,0,0,0,0,0,
        0x48, 0xb9, 0,0,0,0,0,0,0,0,
        0x48, 0xba, 0,0,0,0,0,0,0,0,
        0xdd, 0x00,
        0xdd, 0x01,
        0xde, 0xc1,                      // faddp st1,st0 (ST1 = 10+20 = 30; pop)
        0xdd, 0x1a,
        0xc3,
    };
    const u64 a=reinterpret_cast<u64>(pa),b=reinterpret_cast<u64>(pb),o=reinterpret_cast<u64>(out);
    std::memcpy(&program[2],&a,8); std::memcpy(&program[12],&b,8); std::memcpy(&program[22],&o,8);
    const auto r = RunProgram(program, sizeof(program), mem);
    double res; std::memcpy(&res, out, 8);
    EXPECT_EQ(res, 30.0) << "faddp: 10+20 = 30";
    EXPECT_EQ(r.state.exit_reason, static_cast<u32>(ExitReason::BlockEnd));
}

// ============================================================================
// EXTENDED EMITTER COVERAGE
//
// The tests below broaden coverage of instruction families that the lifter
// fully supports but that the original suite only sampled: the complete
// SETcc / CMOVcc / Jcc condition matrices, plus flag-precision and width
// edge cases for the integer ALU. Each test drives a single condition or
// edge case in isolation so a failure pinpoints exactly which predicate or
// width is wrong.
//
// Convention: where a test needs a specific RFLAGS bit pattern, it sets
// state.rflags directly (bit 1 is the reserved-1 bit; we always keep it).
// SETcc/CMOVcc read flags from state.rflags, so this is a direct probe of
// the condition-decode logic without depending on a preceding compare.
// ============================================================================

namespace {
// RFLAGS arithmetic bits, mirroring the constants used earlier in the file
// but re-expressed locally for readability in the condition tables.
constexpr u64 kFlagReserved1 = 1ULL << 1;  // always set in real RFLAGS
constexpr u64 FCF = 1ULL << 0;
constexpr u64 FPF = 1ULL << 2;
constexpr u64 FZF = 1ULL << 6;
constexpr u64 FSF = 1ULL << 7;
constexpr u64 FOF = 1ULL << 11;

// Run a single-instruction-plus-RET program with a caller-controlled
// initial register/flag state, returning the post-run state.
GuestState RunWithState(const u8* program, size_t n, GuestMemory& mem,
                        std::function<void(GuestState&)> setup) {
    std::memcpy(mem.CodePtr(), program, n);
    u8* guest_rsp = mem.StackTop() - 8;
    *reinterpret_cast<u64*>(guest_rsp) = kReturnSentinel;
    GuestState st{};
    st.rip = reinterpret_cast<u64>(mem.CodePtr());
    st.gpr[4] = reinterpret_cast<u64>(guest_rsp);
    st.rflags = kFlagReserved1;
    setup(st);
    Runtime rt;
    rt.Run(st);
    return st;
}
} // namespace

// ----------------------------------------------------------------------------
// SETcc -- the full 16-condition matrix. Each setcc writes 0/1 into AL's low
// byte based on the flag predicate. We poison RAX's upper bits and verify
// only the low byte is touched and that it matches the expected boolean.
// Encoding: 0F 9x C0  (setcc al). Followed by RET.
// ----------------------------------------------------------------------------

// Helper macro: emit `setcc al; ret`, set rflags, expect al == want.
#define SETCC_TEST(NAME, OPC2, FLAGS, WANT)                                  \
    TEST_F(CpuRuntimeTest, SetccMatrix_##NAME) {                            \
        const u8 program[] = {0x0f, OPC2, 0xc0, 0xc3};                      \
        const auto st = RunWithState(program, sizeof(program), mem,         \
            [](GuestState& s) {                                             \
                s.gpr[0] = 0xDEADBEEFCAFE0042ULL; /* poison; al=0x42 */     \
                s.rflags |= (FLAGS);                                        \
            });                                                             \
        EXPECT_EQ(st.gpr[0] & 0xFFULL, static_cast<u64>(WANT))              \
            << #NAME " low byte";                                           \
        EXPECT_EQ(st.gpr[0] & ~0xFFULL, 0xDEADBEEFCAFE0000ULL)              \
            << #NAME " preserves upper 56 bits";                           \
        EXPECT_EQ(st.exit_reason, static_cast<u32>(ExitReason::BlockEnd));  \
    }

// OF set / clear.
SETCC_TEST(Seto_True,   0x90, FOF, 1)
SETCC_TEST(Seto_False,  0x90, 0,   0)
SETCC_TEST(Setno_True,  0x91, 0,   1)
SETCC_TEST(Setno_False, 0x91, FOF, 0)
// CF (below / above-equal).
SETCC_TEST(Setb_True,   0x92, FCF, 1)
SETCC_TEST(Setb_False,  0x92, 0,   0)
SETCC_TEST(Setae_True,  0x93, 0,   1)
SETCC_TEST(Setae_False, 0x93, FCF, 0)
// ZF (equal / not-equal).
SETCC_TEST(Sete_True,   0x94, FZF, 1)
SETCC_TEST(Sete_False,  0x94, 0,   0)
SETCC_TEST(Setne_True,  0x95, 0,   1)
SETCC_TEST(Setne_False, 0x95, FZF, 0)
// CF|ZF (below-equal / above).
SETCC_TEST(Setbe_CF,    0x96, FCF, 1)
SETCC_TEST(Setbe_ZF,    0x96, FZF, 1)
SETCC_TEST(Setbe_False, 0x96, 0,   0)
SETCC_TEST(Seta_True,   0x97, 0,   1)
SETCC_TEST(Seta_CF,     0x97, FCF, 0)
SETCC_TEST(Seta_ZF,     0x97, FZF, 0)
// SF.
SETCC_TEST(Sets_True,   0x98, FSF, 1)
SETCC_TEST(Sets_False,  0x98, 0,   0)
SETCC_TEST(Setns_True,  0x99, 0,   1)
SETCC_TEST(Setns_False, 0x99, FSF, 0)
// PF.
SETCC_TEST(Setp_True,   0x9a, FPF, 1)
SETCC_TEST(Setp_False,  0x9a, 0,   0)
SETCC_TEST(Setnp_True,  0x9b, 0,   1)
SETCC_TEST(Setnp_False, 0x9b, FPF, 0)
// SF!=OF (less / greater-equal).
SETCC_TEST(Setl_SF,     0x9c, FSF, 1)         // SF=1,OF=0 -> SF!=OF -> less
SETCC_TEST(Setl_OF,     0x9c, FOF, 1)         // SF=0,OF=1 -> less
SETCC_TEST(Setl_Equal,  0x9c, FSF | FOF, 0)   // SF==OF -> not less
SETCC_TEST(Setge_Equal, 0x9d, FSF | FOF, 1)   // SF==OF -> ge
SETCC_TEST(Setge_Diff,  0x9d, FSF, 0)         // SF!=OF -> not ge
// ZF | (SF!=OF) (less-equal / greater).
SETCC_TEST(Setle_ZF,    0x9e, FZF, 1)
SETCC_TEST(Setle_SF,    0x9e, FSF, 1)
SETCC_TEST(Setle_False, 0x9e, 0,   0)
SETCC_TEST(Setg_True,   0x9f, 0,   1)
SETCC_TEST(Setg_ZF,     0x9f, FZF, 0)
SETCC_TEST(Setg_SF,     0x9f, FSF, 0)

#undef SETCC_TEST

// ----------------------------------------------------------------------------
// CMOVcc -- the full condition matrix. cmovcc rax, rbx (48 0F 4x C3).
// RAX starts at a sentinel "old" value, RBX holds the "new" value. When the
// condition holds, RAX must become RBX; otherwise RAX is unchanged.
// ----------------------------------------------------------------------------

#define CMOV_TEST(NAME, OPC2, FLAGS, TAKEN)                                  \
    TEST_F(CpuRuntimeTest, CmovMatrix_##NAME) {                             \
        const u8 program[] = {0x48, 0x0f, OPC2, 0xc3, 0xc3};               \
        const u64 OLD = 0x1111111111111111ULL;                             \
        const u64 NEW = 0x2222222222222222ULL;                             \
        const auto st = RunWithState(program, sizeof(program), mem,         \
            [&](GuestState& s) {                                           \
                s.gpr[0] = OLD; s.gpr[3] = NEW;                            \
                s.rflags |= (FLAGS);                                       \
            });                                                            \
        EXPECT_EQ(st.gpr[0], (TAKEN) ? NEW : OLD) << #NAME;                 \
        EXPECT_EQ(st.exit_reason, static_cast<u32>(ExitReason::BlockEnd));  \
    }

CMOV_TEST(Cmovo_T,   0x40, FOF, true)
CMOV_TEST(Cmovo_F,   0x40, 0,   false)
CMOV_TEST(Cmovno_T,  0x41, 0,   true)
CMOV_TEST(Cmovno_F,  0x41, FOF, false)
CMOV_TEST(Cmovb_T,   0x42, FCF, true)
CMOV_TEST(Cmovb_F,   0x42, 0,   false)
CMOV_TEST(Cmovae_T,  0x43, 0,   true)
CMOV_TEST(Cmovae_F,  0x43, FCF, false)
CMOV_TEST(Cmove_T,   0x44, FZF, true)
CMOV_TEST(Cmove_F,   0x44, 0,   false)
CMOV_TEST(Cmovne_T,  0x45, 0,   true)
CMOV_TEST(Cmovne_F,  0x45, FZF, false)
CMOV_TEST(Cmovbe_CF, 0x46, FCF, true)
CMOV_TEST(Cmovbe_ZF, 0x46, FZF, true)
CMOV_TEST(Cmovbe_F,  0x46, 0,   false)
CMOV_TEST(Cmova_T,   0x47, 0,   true)
CMOV_TEST(Cmova_CF,  0x47, FCF, false)
CMOV_TEST(Cmovs_T,   0x48, FSF, true)
CMOV_TEST(Cmovs_F,   0x48, 0,   false)
CMOV_TEST(Cmovns_T,  0x49, 0,   true)
CMOV_TEST(Cmovns_F,  0x49, FSF, false)
CMOV_TEST(Cmovp_T,   0x4a, FPF, true)
CMOV_TEST(Cmovp_F,   0x4a, 0,   false)
CMOV_TEST(Cmovnp_T,  0x4b, 0,   true)
CMOV_TEST(Cmovnp_F,  0x4b, FPF, false)
CMOV_TEST(Cmovl_SF,  0x4c, FSF, true)
CMOV_TEST(Cmovl_Eq,  0x4c, FSF | FOF, false)
CMOV_TEST(Cmovge_Eq, 0x4d, FSF | FOF, true)
CMOV_TEST(Cmovge_D,  0x4d, FSF, false)
CMOV_TEST(Cmovle_ZF, 0x4e, FZF, true)
CMOV_TEST(Cmovle_SF, 0x4e, FSF, true)
CMOV_TEST(Cmovle_F,  0x4e, 0,   false)
CMOV_TEST(Cmovg_T,   0x4f, 0,   true)
CMOV_TEST(Cmovg_ZF,  0x4f, FZF, false)

#undef CMOV_TEST

// ----------------------------------------------------------------------------
// Jcc -- the full short-form condition matrix (opcode 7x rel8). Each test lays
// out: `jcc +N ; <BSR unsupported> ; <padding> ; <BSR unsupported>` so that a
// TAKEN branch lands on the second BSR (clean UnsupportedInstruction at a
// known RIP) and a NOT-TAKEN branch hits the first BSR right after the Jcc.
// We distinguish taken/not-taken by which RIP the unsupported-exit reports.
// ----------------------------------------------------------------------------

#define JCC_TEST(NAME, OPC, FLAGS, EXPECT_TAKEN)                             \
    TEST_F(CpuRuntimeTest, JccMatrix_##NAME) {                             \
        /* layout:                                                          \
           0: jcc +6           (2 bytes)                                    \
           2: bsr rbx,rax      (4 bytes)  <- not-taken lands here           \
           6: nop nop          (2 bytes)                                    \
           8: bsr rbx,rax      (4 bytes)  <- taken lands here (2+4=6 disp)  \
        */                                                                  \
        const u8 program[] = {                                              \
            OPC, 0x06,                                                      \
            0x48, 0x0f, 0xbd, 0xd8,                                         \
            0x90, 0x90,                                                     \
            0x48, 0x0f, 0xbd, 0xd8,                                         \
        };                                                                  \
        std::memcpy(mem.CodePtr(), program, sizeof(program));              \
        u8* guest_rsp = mem.StackTop() - 8;                                 \
        *reinterpret_cast<u64*>(guest_rsp) = kReturnSentinel;               \
        GuestState st{};                                                    \
        const u64 base = reinterpret_cast<u64>(mem.CodePtr());             \
        st.rip = base; st.gpr[4] = reinterpret_cast<u64>(guest_rsp);       \
        st.rflags = kFlagReserved1 | (FLAGS);                              \
        Runtime rt; rt.Run(st);                                             \
        EXPECT_EQ(st.exit_reason,                                          \
                  static_cast<u32>(ExitReason::UnsupportedInstruction));   \
        EXPECT_EQ(st.rip, base + ((EXPECT_TAKEN) ? 8u : 2u)) << #NAME;      \
    }

JCC_TEST(Jo_T,   0x70, FOF, true)
JCC_TEST(Jo_F,   0x70, 0,   false)
JCC_TEST(Jno_T,  0x71, 0,   true)
JCC_TEST(Jno_F,  0x71, FOF, false)
JCC_TEST(Jb_T,   0x72, FCF, true)
JCC_TEST(Jb_F,   0x72, 0,   false)
JCC_TEST(Jae_T,  0x73, 0,   true)
JCC_TEST(Jae_F,  0x73, FCF, false)
JCC_TEST(Je_T,   0x74, FZF, true)
JCC_TEST(Je_F,   0x74, 0,   false)
JCC_TEST(Jne_T,  0x75, 0,   true)
JCC_TEST(Jne_F,  0x75, FZF, false)
JCC_TEST(Jbe_CF, 0x76, FCF, true)
JCC_TEST(Jbe_ZF, 0x76, FZF, true)
JCC_TEST(Jbe_F,  0x76, 0,   false)
JCC_TEST(Ja_T,   0x77, 0,   true)
JCC_TEST(Ja_CF,  0x77, FCF, false)
JCC_TEST(Js_T,   0x78, FSF, true)
JCC_TEST(Js_F,   0x78, 0,   false)
JCC_TEST(Jns_T,  0x79, 0,   true)
JCC_TEST(Jns_F,  0x79, FSF, false)
JCC_TEST(Jp_T,   0x7a, FPF, true)
JCC_TEST(Jp_F,   0x7a, 0,   false)
JCC_TEST(Jnp_T,  0x7b, 0,   true)
JCC_TEST(Jnp_F,  0x7b, FPF, false)
JCC_TEST(Jl_SF,  0x7c, FSF, true)
JCC_TEST(Jl_Eq,  0x7c, FSF | FOF, false)
JCC_TEST(Jge_Eq, 0x7d, FSF | FOF, true)
JCC_TEST(Jge_D,  0x7d, FSF, false)
JCC_TEST(Jle_ZF, 0x7e, FZF, true)
JCC_TEST(Jle_F,  0x7e, 0,   false)
JCC_TEST(Jg_T,   0x7f, 0,   true)
JCC_TEST(Jg_ZF,  0x7f, FZF, false)

#undef JCC_TEST

// ----------------------------------------------------------------------------
// Flag-precision tests: verify that ADD/SUB set each arithmetic flag exactly,
// including the signed-overflow and carry corner cases that simpler tests miss.
// ----------------------------------------------------------------------------

// ADD that overflows signed (0x7FFF...FF + 1): OF=1, SF=1, CF=0, ZF=0.
TEST_F(CpuRuntimeTest, AddFlags_SignedOverflow_SetsOfSf) {
    const u8 program[] = {
        0x48, 0xb8, 0xff,0xff,0xff,0xff,0xff,0xff,0xff,0x7f, // mov rax, INT64_MAX
        0x48, 0x83, 0xc0, 0x01,                              // add rax, 1
        0xc3,
    };
    const auto r = RunProgram(program, sizeof(program), mem);
    EXPECT_EQ(r.state.gpr[0], 0x8000000000000000ULL);
    EXPECT_TRUE(r.state.rflags & FOF)  << "signed overflow";
    EXPECT_TRUE(r.state.rflags & FSF)  << "result negative";
    EXPECT_FALSE(r.state.rflags & FCF) << "no unsigned carry";
    EXPECT_FALSE(r.state.rflags & FZF);
}

// ADD that carries unsigned (0xFFFF...FF + 1 = 0): CF=1, ZF=1, OF=0, SF=0.
TEST_F(CpuRuntimeTest, AddFlags_UnsignedCarryToZero_SetsCfZf) {
    const u8 program[] = {
        0x48, 0xc7, 0xc0, 0xff,0xff,0xff,0xff, // mov rax, -1 (sign-extended 0xFFFF..FF)
        0x48, 0x83, 0xc0, 0x01,                // add rax, 1 -> 0
        0xc3,
    };
    const auto r = RunProgram(program, sizeof(program), mem);
    EXPECT_EQ(r.state.gpr[0], 0u);
    EXPECT_TRUE(r.state.rflags & FCF)  << "carry out of bit 63";
    EXPECT_TRUE(r.state.rflags & FZF)  << "result zero";
    EXPECT_FALSE(r.state.rflags & FOF) << "no signed overflow (-1 + 1)";
    EXPECT_FALSE(r.state.rflags & FSF);
}

// SUB producing signed overflow (INT64_MIN - 1): OF=1, CF=0 (no borrow),
// SF=0 (result wraps to positive 0x7FFF...FF).
TEST_F(CpuRuntimeTest, SubFlags_SignedOverflow_SetsOf) {
    const u8 program[] = {
        0x48, 0xb8, 0,0,0,0,0,0,0,0x80, // mov rax, INT64_MIN (0x8000..00)
        0x48, 0x83, 0xe8, 0x01,         // sub rax, 1
        0xc3,
    };
    const auto r = RunProgram(program, sizeof(program), mem);
    EXPECT_EQ(r.state.gpr[0], 0x7FFFFFFFFFFFFFFFULL);
    EXPECT_TRUE(r.state.rflags & FOF)  << "signed overflow on INT64_MIN - 1";
    EXPECT_FALSE(r.state.rflags & FSF) << "wrapped to positive";
    EXPECT_FALSE(r.state.rflags & FCF) << "0x8000..00 >= 1, no borrow";
}

// Parity flag: a result with an even number of set bits in the low byte sets
// PF; an odd count clears it. (0x03 has two bits -> PF=1; 0x07 has three ->
// PF=0.) We use AND to land an exact low byte.
TEST_F(CpuRuntimeTest, ParityFlag_EvenBitsSetsPf) {
    const u8 program[] = {
        0x48, 0xc7, 0xc0, 0x03, 0x00, 0x00, 0x00, // mov rax, 3 (0b11, even parity)
        0x48, 0x25, 0xff, 0x00, 0x00, 0x00,       // and rax, 0xFF
        0xc3,
    };
    const auto r = RunProgram(program, sizeof(program), mem);
    EXPECT_EQ(r.state.gpr[0], 3u);
    EXPECT_TRUE(r.state.rflags & FPF) << "two set bits -> even parity -> PF=1";
}
TEST_F(CpuRuntimeTest, ParityFlag_OddBitsClearsPf) {
    const u8 program[] = {
        0x48, 0xc7, 0xc0, 0x07, 0x00, 0x00, 0x00, // mov rax, 7 (0b111, odd parity)
        0x48, 0x25, 0xff, 0x00, 0x00, 0x00,       // and rax, 0xFF
        0xc3,
    };
    const auto r = RunProgram(program, sizeof(program), mem);
    EXPECT_EQ(r.state.gpr[0], 7u);
    EXPECT_FALSE(r.state.rflags & FPF) << "three set bits -> odd parity -> PF=0";
}

// ----------------------------------------------------------------------------
// ADC / SBB carry-propagation, exercised through a real 128-bit add and sub
// (two 64-bit halves chained via CF). These complement the existing
// Adc_128BitAddChain test with explicit edge values.
// ----------------------------------------------------------------------------

// 0xFFFF...FF + 1 across two limbs: low = 0 (CF=1), high += carry.
TEST_F(CpuRuntimeTest, Adc_CarryPropagatesIntoHighLimb) {
    const u8 program[] = {
        0x48, 0xc7, 0xc0, 0xff,0xff,0xff,0xff, // mov rax, -1   (low limb a)
        0x48, 0xc7, 0xc3, 0x00,0x00,0x00,0x00, // mov rbx, 0    (high limb a)
        0x48, 0x83, 0xc0, 0x01,                // add rax, 1    -> 0, CF=1
        0x48, 0x83, 0xd3, 0x00,                // adc rbx, 0    -> 0 + 0 + CF = 1
        0xc3,
    };
    const auto r = RunProgram(program, sizeof(program), mem);
    EXPECT_EQ(r.state.gpr[0], 0u)  << "low limb wrapped to 0";
    EXPECT_EQ(r.state.gpr[3], 1u)  << "high limb picked up the carry";
}

// SBB borrow: 0 - 1 with no prior borrow sets CF; then sbb high - 0 - CF.
TEST_F(CpuRuntimeTest, Sbb_BorrowPropagatesIntoHighLimb) {
    const u8 program[] = {
        0x48, 0xc7, 0xc0, 0x00,0x00,0x00,0x00, // mov rax, 0  (low limb)
        0x48, 0xc7, 0xc3, 0x05,0x00,0x00,0x00, // mov rbx, 5  (high limb)
        0x48, 0x83, 0xe8, 0x01,                // sub rax, 1  -> -1, CF=1 (borrow)
        0x48, 0x83, 0xdb, 0x00,                // sbb rbx, 0  -> 5 - 0 - 1 = 4
        0xc3,
    };
    const auto r = RunProgram(program, sizeof(program), mem);
    EXPECT_EQ(r.state.gpr[0], 0xFFFFFFFFFFFFFFFFULL) << "low limb underflowed";
    EXPECT_EQ(r.state.gpr[3], 4u) << "high limb lost one to the borrow";
}

// ----------------------------------------------------------------------------
// NEG / NOT at 32-bit width: NEG must set flags and the 32-bit form must
// zero-extend the upper 32. NOT never touches flags.
// ----------------------------------------------------------------------------

// NEG eax of a nonzero value: result is two's complement, CF set, upper32=0.
TEST_F(CpuRuntimeTest, Neg32_NonZero_SetsCfAndZeroExtends) {
    const u8 program[] = {
        0x48, 0xb8, 0x05,0,0,0, 0xEF,0xBE,0xAD,0xDE, // mov rax, 0xDEADBEEF00000005
        0xf7, 0xd8,                                   // neg eax
        0xc3,
    };
    const auto r = RunProgram(program, sizeof(program), mem);
    EXPECT_EQ(r.state.gpr[0], 0x00000000FFFFFFFBULL)
        << "neg of 5 in 32 bits is 0xFFFFFFFB, upper 32 zeroed";
    EXPECT_TRUE(r.state.rflags & FCF) << "NEG of nonzero sets CF";
}

// NEG eax of zero: result 0, CF clear, ZF set.
TEST_F(CpuRuntimeTest, Neg32_Zero_ClearsCf) {
    const u8 program[] = {
        0x48, 0xc7, 0xc0, 0,0,0,0, // mov rax, 0
        0xf7, 0xd8,                // neg eax
        0xc3,
    };
    const auto r = RunProgram(program, sizeof(program), mem);
    EXPECT_EQ(r.state.gpr[0], 0u);
    EXPECT_FALSE(r.state.rflags & FCF) << "NEG of zero clears CF";
    EXPECT_TRUE(r.state.rflags & FZF);
}

// ----------------------------------------------------------------------------
// MOVSX width coverage: sign-extend an 8-bit and a 16-bit source to 64 bits.
// ----------------------------------------------------------------------------

TEST_F(CpuRuntimeTest, Movsx_Byte_NegativeSignExtendsTo64) {
    const u8 program[] = {
        0x48, 0xc7, 0xc3, 0x80,0xff,0xff,0xff, // mov rbx, 0xFFFFFFFFFFFFFF80 (bl=0x80)
        0x48, 0x0f, 0xbe, 0xc3,                // movsx rax, bl
        0xc3,
    };
    const auto r = RunProgram(program, sizeof(program), mem);
    EXPECT_EQ(r.state.gpr[0], 0xFFFFFFFFFFFFFF80ULL)
        << "0x80 as signed byte is -128, sign-extended";
}
TEST_F(CpuRuntimeTest, Movsx_Word_NegativeSignExtendsTo64) {
    const u8 program[] = {
        0x48, 0xc7, 0xc3, 0x00,0x80,0xff,0xff, // mov rbx, ...0xFFFF8000 (bx=0x8000)
        0x48, 0x0f, 0xbf, 0xc3,                // movsx rax, bx
        0xc3,
    };
    const auto r = RunProgram(program, sizeof(program), mem);
    EXPECT_EQ(r.state.gpr[0], 0xFFFFFFFFFFFF8000ULL)
        << "0x8000 as signed word sign-extends to 64";
}

// ----------------------------------------------------------------------------
// Shift edge cases: SHR/SAR/SHL by a count that the hardware masks to 6 bits
// (64-bit) or 5 bits (32-bit). Verify masking and the sign behavior of SAR.
// ----------------------------------------------------------------------------

// SAR rax (arithmetic) of a negative value fills with sign bits.
TEST_F(CpuRuntimeTest, Sar64_Negative_SignFills) {
    const u8 program[] = {
        0x48, 0xb8, 0,0,0,0,0,0,0,0x80, // mov rax, INT64_MIN
        0x48, 0xc1, 0xf8, 0x04,         // sar rax, 4
        0xc3,
    };
    const auto r = RunProgram(program, sizeof(program), mem);
    EXPECT_EQ(r.state.gpr[0], 0xF800000000000000ULL)
        << "arithmetic shift fills the top 4 bits with the sign";
}

// SHL eax by a count masked to 5 bits: shl eax, 33 == shl eax, 1.
TEST_F(CpuRuntimeTest, Shl32_CountMaskedTo5Bits) {
    const u8 program[] = {
        0x48, 0xc7, 0xc0, 0x01,0x00,0x00,0x00, // mov rax, 1
        0xc1, 0xe0, 0x21,                       // shl eax, 33  (33 & 31 = 1)
        0xc3,
    };
    const auto r = RunProgram(program, sizeof(program), mem);
    EXPECT_EQ(r.state.gpr[0], 2u) << "33 masks to 1: 1<<1 = 2, upper32 zeroed";
}

// ============================================================================
// EXTENDED EMITTER COVERAGE, BATCH 2
//
// Packed-FP arithmetic and bitwise families, the ordered scalar compares
// (VCOMISS/VCOMISD, distinct from the already-covered unordered VUCOMI*),
// the unaligned vector moves, the float-duplicate moves, and CWDE. Every
// encoding below was byte-verified with the assembler. xmm register -> YMM
// slot mapping: xmm_n low64 = ymm[n*4], high64 = ymm[n*4+1]; VEX-128 zeros
// ymm[dst*4+2 .. +3].
// ============================================================================

namespace {
// Pack two floats into a 64-bit chunk (low lane in bits 0:31).
inline u64 PackF2(float lo, float hi) {
    return (static_cast<u64>(std::bit_cast<u32>(hi)) << 32) |
            static_cast<u64>(std::bit_cast<u32>(lo));
}
// Read lane `i` (0..3) of xmm0 from the post-run state as a float.
inline float XmmLaneF(const GuestState& st, int lane) {
    const u64 chunk = st.ymm[(lane >> 1)];        // xmm0 low=ymm[0], high=ymm[1]
    const u32 bits = (lane & 1) ? static_cast<u32>(chunk >> 32)
                                : static_cast<u32>(chunk & 0xFFFFFFFFULL);
    return std::bit_cast<float>(bits);
}
// Standard packed-FP fixture run: load xmm1 and xmm2 with caller data,
// poison xmm0's upper YMM, run a 5-byte program (4-byte op + RET).
GuestState RunPackedFp(const u8* program, size_t n, GuestMemory& mem,
                       u64 x1_lo, u64 x1_hi, u64 x2_lo, u64 x2_hi) {
    std::memcpy(mem.CodePtr(), program, n);
    u8* guest_rsp = mem.StackTop() - 8;
    *reinterpret_cast<u64*>(guest_rsp) = kReturnSentinel;
    GuestState st{};
    st.rip = reinterpret_cast<u64>(mem.CodePtr());
    st.gpr[4] = reinterpret_cast<u64>(guest_rsp);
    st.ymm[4] = x1_lo; st.ymm[5] = x1_hi;   // xmm1
    st.ymm[8] = x2_lo; st.ymm[9] = x2_hi;   // xmm2
    st.ymm[2] = 0xDEADBEEFDEADBEEFULL;       // xmm0 upper, must be zeroed
    st.ymm[3] = 0xDEADBEEFDEADBEEFULL;
    Runtime rt;
    rt.Run(st);
    return st;
}
} // namespace

// ----------------------------------------------------------------------------
// Packed single-precision arithmetic. Inputs are chosen so each lane has a
// distinct, exactly-representable result and so operand order matters
// (sub/div are not commutative).
// ----------------------------------------------------------------------------

// vaddps xmm0, xmm1, xmm2 : lanewise xmm1 + xmm2.
TEST_F(CpuRuntimeTest, PackedFp_Vaddps_Lanewise) {
    const u8 program[] = {0xc5, 0xf0, 0x58, 0xc2, 0xc3};
    const auto st = RunPackedFp(program, sizeof(program), mem,
        PackF2(1.0f, 2.0f), PackF2(3.0f, 4.0f),     // xmm1 = [1,2,3,4]
        PackF2(10.0f, 20.0f), PackF2(30.0f, 40.0f)); // xmm2 = [10,20,30,40]
    EXPECT_EQ(XmmLaneF(st, 0), 11.0f);
    EXPECT_EQ(XmmLaneF(st, 1), 22.0f);
    EXPECT_EQ(XmmLaneF(st, 2), 33.0f);
    EXPECT_EQ(XmmLaneF(st, 3), 44.0f);
    EXPECT_EQ(st.ymm[2], 0ULL) << "VEX-128 zeros upper YMM";
    EXPECT_EQ(st.ymm[3], 0ULL);
}

// vsubps : order matters (xmm1 - xmm2).
TEST_F(CpuRuntimeTest, PackedFp_Vsubps_OrderMatters) {
    const u8 program[] = {0xc5, 0xf0, 0x5c, 0xc2, 0xc3};
    const auto st = RunPackedFp(program, sizeof(program), mem,
        PackF2(10.0f, 20.0f), PackF2(30.0f, 40.0f),
        PackF2(1.0f, 2.0f), PackF2(3.0f, 4.0f));
    EXPECT_EQ(XmmLaneF(st, 0), 9.0f);
    EXPECT_EQ(XmmLaneF(st, 1), 18.0f);
    EXPECT_EQ(XmmLaneF(st, 2), 27.0f);
    EXPECT_EQ(XmmLaneF(st, 3), 36.0f);
}

// vmulps : lanewise product.
TEST_F(CpuRuntimeTest, PackedFp_Vmulps_Lanewise) {
    const u8 program[] = {0xc5, 0xf0, 0x59, 0xc2, 0xc3};
    const auto st = RunPackedFp(program, sizeof(program), mem,
        PackF2(2.0f, 3.0f), PackF2(4.0f, 5.0f),
        PackF2(0.5f, 2.0f), PackF2(0.25f, 10.0f));
    EXPECT_EQ(XmmLaneF(st, 0), 1.0f);
    EXPECT_EQ(XmmLaneF(st, 1), 6.0f);
    EXPECT_EQ(XmmLaneF(st, 2), 1.0f);
    EXPECT_EQ(XmmLaneF(st, 3), 50.0f);
}

// vminps : per-lane minimum.
TEST_F(CpuRuntimeTest, PackedFp_Vminps_PerLaneMin) {
    const u8 program[] = {0xc5, 0xf0, 0x5d, 0xc2, 0xc3};
    const auto st = RunPackedFp(program, sizeof(program), mem,
        PackF2(1.0f, 50.0f), PackF2(3.0f, 8.0f),
        PackF2(9.0f, 20.0f), PackF2(30.0f, 2.0f));
    EXPECT_EQ(XmmLaneF(st, 0), 1.0f);
    EXPECT_EQ(XmmLaneF(st, 1), 20.0f);
    EXPECT_EQ(XmmLaneF(st, 2), 3.0f);
    EXPECT_EQ(XmmLaneF(st, 3), 2.0f);
}

// ----------------------------------------------------------------------------
// Packed double-precision arithmetic. xmm holds two doubles: low64 = lane0,
// high64 = lane1.
// ----------------------------------------------------------------------------

namespace {
inline u64 PackD(double d) { return std::bit_cast<u64>(d); }
inline double XmmLaneD(const GuestState& st, int lane) {
    return std::bit_cast<double>(st.ymm[lane]); // xmm0 lane0=ymm[0], lane1=ymm[1]
}
} // namespace

TEST_F(CpuRuntimeTest, PackedFp_Vaddpd_Lanewise) {
    const u8 program[] = {0xc5, 0xf1, 0x58, 0xc2, 0xc3};
    const auto st = RunPackedFp(program, sizeof(program), mem,
        PackD(1.5), PackD(2.5), PackD(10.0), PackD(20.0));
    EXPECT_EQ(XmmLaneD(st, 0), 11.5);
    EXPECT_EQ(XmmLaneD(st, 1), 22.5);
    EXPECT_EQ(st.ymm[2], 0ULL);
    EXPECT_EQ(st.ymm[3], 0ULL);
}

TEST_F(CpuRuntimeTest, PackedFp_Vsubpd_OrderMatters) {
    const u8 program[] = {0xc5, 0xf1, 0x5c, 0xc2, 0xc3};
    const auto st = RunPackedFp(program, sizeof(program), mem,
        PackD(100.0), PackD(7.0), PackD(1.0), PackD(2.0));
    EXPECT_EQ(XmmLaneD(st, 0), 99.0);
    EXPECT_EQ(XmmLaneD(st, 1), 5.0);
}

TEST_F(CpuRuntimeTest, PackedFp_Vmulpd_Lanewise) {
    const u8 program[] = {0xc5, 0xf1, 0x59, 0xc2, 0xc3};
    const auto st = RunPackedFp(program, sizeof(program), mem,
        PackD(3.0), PackD(4.0), PackD(2.0), PackD(0.5));
    EXPECT_EQ(XmmLaneD(st, 0), 6.0);
    EXPECT_EQ(XmmLaneD(st, 1), 2.0);
}

TEST_F(CpuRuntimeTest, PackedFp_Vdivpd_OrderMatters) {
    const u8 program[] = {0xc5, 0xf1, 0x5e, 0xc2, 0xc3};
    const auto st = RunPackedFp(program, sizeof(program), mem,
        PackD(20.0), PackD(9.0), PackD(4.0), PackD(3.0));
    EXPECT_EQ(XmmLaneD(st, 0), 5.0);
    EXPECT_EQ(XmmLaneD(st, 1), 3.0);
}

TEST_F(CpuRuntimeTest, PackedFp_Vmaxpd_PerLaneMax) {
    const u8 program[] = {0xc5, 0xf1, 0x5f, 0xc2, 0xc3};
    const auto st = RunPackedFp(program, sizeof(program), mem,
        PackD(1.0), PackD(50.0), PackD(9.0), PackD(20.0));
    EXPECT_EQ(XmmLaneD(st, 0), 9.0);
    EXPECT_EQ(XmmLaneD(st, 1), 50.0);
}

// ----------------------------------------------------------------------------
// Packed FP bitwise ops operate on the raw 128-bit pattern, lane-agnostic.
// We treat the xmm as two u64 halves and check the bit math directly.
// ----------------------------------------------------------------------------

TEST_F(CpuRuntimeTest, PackedFp_Vandps_BitwiseAnd) {
    const u8 program[] = {0xc5, 0xf0, 0x54, 0xc2, 0xc3};
    const auto st = RunPackedFp(program, sizeof(program), mem,
        0xFF00FF00FF00FF00ULL, 0x0F0F0F0F0F0F0F0FULL,
        0x0FF00FF00FF00FF0ULL, 0xFFFF0000FFFF0000ULL);
    EXPECT_EQ(st.ymm[0], 0xFF00FF00FF00FF00ULL & 0x0FF00FF00FF00FF0ULL);
    EXPECT_EQ(st.ymm[1], 0x0F0F0F0F0F0F0F0FULL & 0xFFFF0000FFFF0000ULL);
    EXPECT_EQ(st.ymm[2], 0ULL);
}

TEST_F(CpuRuntimeTest, PackedFp_Vorps_BitwiseOr) {
    const u8 program[] = {0xc5, 0xf0, 0x56, 0xc2, 0xc3};
    const auto st = RunPackedFp(program, sizeof(program), mem,
        0x00FF00FF00FF00FFULL, 0xAAAAAAAAAAAAAAAAULL,
        0xFF00FF00FF00FF00ULL, 0x5555555555555555ULL);
    EXPECT_EQ(st.ymm[0], 0xFFFFFFFFFFFFFFFFULL);
    EXPECT_EQ(st.ymm[1], 0xFFFFFFFFFFFFFFFFULL);
}

TEST_F(CpuRuntimeTest, PackedFp_Vxorpd_BitwiseXor) {
    const u8 program[] = {0xc5, 0xf1, 0x57, 0xc2, 0xc3};
    const auto st = RunPackedFp(program, sizeof(program), mem,
        0x1234567812345678ULL, 0xFFFFFFFFFFFFFFFFULL,
        0x0F0F0F0F0F0F0F0FULL, 0x00000000FFFFFFFFULL);
    EXPECT_EQ(st.ymm[0], 0x1234567812345678ULL ^ 0x0F0F0F0F0F0F0F0FULL);
    EXPECT_EQ(st.ymm[1], 0xFFFFFFFFFFFFFFFFULL ^ 0x00000000FFFFFFFFULL);
}

// vandnps : (NOT src1) AND src2 -- the bit-clear primitive.
TEST_F(CpuRuntimeTest, PackedFp_Vandnps_NotSrc1AndSrc2) {
    const u8 program[] = {0xc5, 0xf0, 0x55, 0xc2, 0xc3};
    const auto st = RunPackedFp(program, sizeof(program), mem,
        0xFF00FF00FF00FF00ULL, 0x0000000000000000ULL,
        0x1234567812345678ULL, 0xCAFEF00DCAFEF00DULL);
    EXPECT_EQ(st.ymm[0], (~0xFF00FF00FF00FF00ULL) & 0x1234567812345678ULL);
    EXPECT_EQ(st.ymm[1], (~0x0000000000000000ULL) & 0xCAFEF00DCAFEF00DULL);
}

TEST_F(CpuRuntimeTest, PackedInt_Vpor_BitwiseOr) {
    const u8 program[] = {0xc5, 0xf1, 0xeb, 0xc2, 0xc3};
    const auto st = RunPackedFp(program, sizeof(program), mem,
        0x00000000FFFFFFFFULL, 0x1111111100000000ULL,
        0xFFFFFFFF00000000ULL, 0x0000000022222222ULL);
    EXPECT_EQ(st.ymm[0], 0xFFFFFFFFFFFFFFFFULL);
    EXPECT_EQ(st.ymm[1], 0x1111111122222222ULL);
}

TEST_F(CpuRuntimeTest, PackedInt_Vpxor_SelfZeroesRegister) {
    // vpxor xmm0, xmm1, xmm1 -- the canonical zero idiom (src2 == src1).
    // Encoding: c5 f1 ef c1  (xmm0 = xmm1 ^ xmm1).
    const u8 program[] = {0xc5, 0xf1, 0xef, 0xc1, 0xc3};
    const auto st = RunPackedFp(program, sizeof(program), mem,
        0xDEADBEEFCAFEF00DULL, 0x1234567887654321ULL, 0, 0);
    EXPECT_EQ(st.ymm[0], 0ULL) << "x ^ x = 0";
    EXPECT_EQ(st.ymm[1], 0ULL);
    EXPECT_EQ(st.ymm[2], 0ULL);
    EXPECT_EQ(st.ymm[3], 0ULL);
}

// ----------------------------------------------------------------------------
// VCOMISS / VCOMISD -- ORDERED scalar compares. They set CF/ZF/PF from the
// low-lane comparison; an unordered (NaN) operand sets ZF=PF=CF=1, identical
// in flag output to the unordered VUCOMI* form. We verify the three ordered
// outcomes plus the NaN case.
// flag layout: equal -> ZF=1,CF=0,PF=0; less -> CF=1,ZF=0,PF=0;
//              greater -> all clear; unordered -> ZF=CF=PF=1.
// ----------------------------------------------------------------------------

namespace {
GuestState RunComiss(float a, float b, GuestMemory& mem) {
    // vcomiss xmm0, xmm1 : c5 f8 2f c1
    const u8 program[] = {0xc5, 0xf8, 0x2f, 0xc1, 0xc3};
    std::memcpy(mem.CodePtr(), program, sizeof(program));
    u8* rsp = mem.StackTop() - 8;
    *reinterpret_cast<u64*>(rsp) = kReturnSentinel;
    GuestState st{};
    st.rip = reinterpret_cast<u64>(mem.CodePtr());
    st.gpr[4] = reinterpret_cast<u64>(rsp);
    st.ymm[0] = std::bit_cast<u32>(a);   // xmm0 low lane
    st.ymm[4] = std::bit_cast<u32>(b);   // xmm1 low lane
    Runtime rt; rt.Run(st);
    return st;
}
constexpr u64 C_CF = 1ULL << 0, C_PF = 1ULL << 2, C_ZF = 1ULL << 6;
} // namespace

TEST_F(CpuRuntimeTest, Vcomiss_Equal_SetsZfOnly) {
    const auto st = RunComiss(3.5f, 3.5f, mem);
    EXPECT_TRUE(st.rflags & C_ZF);
    EXPECT_FALSE(st.rflags & C_CF);
    EXPECT_FALSE(st.rflags & C_PF);
}
TEST_F(CpuRuntimeTest, Vcomiss_Less_SetsCf) {
    const auto st = RunComiss(1.0f, 2.0f, mem);
    EXPECT_TRUE(st.rflags & C_CF);
    EXPECT_FALSE(st.rflags & C_ZF);
    EXPECT_FALSE(st.rflags & C_PF);
}
TEST_F(CpuRuntimeTest, Vcomiss_Greater_AllClear) {
    const auto st = RunComiss(5.0f, 2.0f, mem);
    EXPECT_FALSE(st.rflags & C_CF);
    EXPECT_FALSE(st.rflags & C_ZF);
    EXPECT_FALSE(st.rflags & C_PF);
}
TEST_F(CpuRuntimeTest, Vcomiss_Nan_SetsAllThree) {
    const float nan = std::numeric_limits<float>::quiet_NaN();
    const auto st = RunComiss(nan, 1.0f, mem);
    EXPECT_TRUE(st.rflags & C_ZF) << "unordered sets ZF";
    EXPECT_TRUE(st.rflags & C_CF) << "unordered sets CF";
    EXPECT_TRUE(st.rflags & C_PF) << "unordered sets PF";
}

namespace {
GuestState RunComisd(double a, double b, GuestMemory& mem) {
    // vcomisd xmm0, xmm1 : c5 f9 2f c1
    const u8 program[] = {0xc5, 0xf9, 0x2f, 0xc1, 0xc3};
    std::memcpy(mem.CodePtr(), program, sizeof(program));
    u8* rsp = mem.StackTop() - 8;
    *reinterpret_cast<u64*>(rsp) = kReturnSentinel;
    GuestState st{};
    st.rip = reinterpret_cast<u64>(mem.CodePtr());
    st.gpr[4] = reinterpret_cast<u64>(rsp);
    st.ymm[0] = std::bit_cast<u64>(a);
    st.ymm[4] = std::bit_cast<u64>(b);
    Runtime rt; rt.Run(st);
    return st;
}
} // namespace

TEST_F(CpuRuntimeTest, Vcomisd_Equal_SetsZfOnly) {
    const auto st = RunComisd(2.0, 2.0, mem);
    EXPECT_TRUE(st.rflags & C_ZF);
    EXPECT_FALSE(st.rflags & C_CF);
}
TEST_F(CpuRuntimeTest, Vcomisd_Less_SetsCf) {
    const auto st = RunComisd(-1.0, 0.0, mem);
    EXPECT_TRUE(st.rflags & C_CF);
    EXPECT_FALSE(st.rflags & C_ZF);
}
TEST_F(CpuRuntimeTest, Vcomisd_Nan_SetsAllThree) {
    const double nan = std::numeric_limits<double>::quiet_NaN();
    const auto st = RunComisd(1.0, nan, mem);
    EXPECT_TRUE(st.rflags & C_ZF);
    EXPECT_TRUE(st.rflags & C_CF);
    EXPECT_TRUE(st.rflags & C_PF);
}

// ----------------------------------------------------------------------------
// Unaligned vector moves: VMOVUPS / VMOVDQU register-to-register. Both copy
// the full 128 bits and (VEX-128) zero the upper YMM.
// ----------------------------------------------------------------------------

TEST_F(CpuRuntimeTest, Vmovups_RegReg_CopiesAndZeroesUpper) {
    // vmovups xmm0, xmm1 : c5 f8 10 c1
    const u8 program[] = {0xc5, 0xf8, 0x10, 0xc1, 0xc3};
    const auto st = RunPackedFp(program, sizeof(program), mem,
        0x1122334455667788ULL, 0x99AABBCCDDEEFF00ULL, 0, 0);
    EXPECT_EQ(st.ymm[0], 0x1122334455667788ULL);
    EXPECT_EQ(st.ymm[1], 0x99AABBCCDDEEFF00ULL);
    EXPECT_EQ(st.ymm[2], 0ULL) << "VEX-128 zeros upper YMM";
    EXPECT_EQ(st.ymm[3], 0ULL);
}

TEST_F(CpuRuntimeTest, Vmovdqu_RegReg_CopiesAndZeroesUpper) {
    // vmovdqu xmm0, xmm1 : c5 fa 6f c1
    const u8 program[] = {0xc5, 0xfa, 0x6f, 0xc1, 0xc3};
    const auto st = RunPackedFp(program, sizeof(program), mem,
        0xCAFEF00DBAADF00DULL, 0x0123456789ABCDEFULL, 0, 0);
    EXPECT_EQ(st.ymm[0], 0xCAFEF00DBAADF00DULL);
    EXPECT_EQ(st.ymm[1], 0x0123456789ABCDEFULL);
    EXPECT_EQ(st.ymm[2], 0ULL);
    EXPECT_EQ(st.ymm[3], 0ULL);
}

// ----------------------------------------------------------------------------
// VMOVSHDUP -- duplicate the odd (high) single-precision lanes:
//   dst = [src[1], src[1], src[3], src[3]].
// (Complements the existing VMOVSLDUP even-lane test.)
// ----------------------------------------------------------------------------
TEST_F(CpuRuntimeTest, Vmovshdup_DuplicatesOddLanes) {
    // vmovshdup xmm0, xmm1 : c5 fa 16 c1
    const u8 program[] = {0xc5, 0xfa, 0x16, 0xc1, 0xc3};
    const auto st = RunPackedFp(program, sizeof(program), mem,
        PackF2(1.0f, 2.0f), PackF2(3.0f, 4.0f), 0, 0); // xmm1 = [1,2,3,4]
    EXPECT_EQ(XmmLaneF(st, 0), 2.0f) << "lane0 <- src lane1";
    EXPECT_EQ(XmmLaneF(st, 1), 2.0f) << "lane1 <- src lane1";
    EXPECT_EQ(XmmLaneF(st, 2), 4.0f) << "lane2 <- src lane3";
    EXPECT_EQ(XmmLaneF(st, 3), 4.0f) << "lane3 <- src lane3";
    EXPECT_EQ(st.ymm[2], 0ULL);
    EXPECT_EQ(st.ymm[3], 0ULL);
}

// ----------------------------------------------------------------------------
// CWDE -- sign-extend AX into EAX (and, per x86-64 32-bit-write rules, the
// result zero-extends into RAX). Complements the existing CDQE/CDQ/CQO tests.
// Encoding: 98 (no REX.W).
// ----------------------------------------------------------------------------
TEST_F(CpuRuntimeTest, Cwde_SignExtendsNegativeAx) {
    const u8 program[] = {
        0x48, 0xb8, 0x00,0x80, 0xEF,0xBE,0xAD,0xDE,0x00,0x00, // mov rax,0x000000DEADBEEF8000
        0x98,                                                  // cwde
        0xc3,
    };
    const auto r = RunProgram(program, sizeof(program), mem);
    // ax = 0x8000 (negative). cwde sign-extends to eax = 0xFFFF8000, which
    // then zero-extends to rax (32-bit write clears bits 63:32).
    EXPECT_EQ(r.state.gpr[0], 0x00000000FFFF8000ULL)
        << "AX=0x8000 sign-extends to EAX=0xFFFF8000, upper32 zeroed";
}
TEST_F(CpuRuntimeTest, Cwde_ZeroExtendsPositiveAx) {
    const u8 program[] = {
        0x48, 0xb8, 0xFF,0x7F, 0xEF,0xBE,0xAD,0xDE,0x00,0x00, // rax low16 = 0x7FFF
        0x98,                                                  // cwde
        0xc3,
    };
    const auto r = RunProgram(program, sizeof(program), mem);
    EXPECT_EQ(r.state.gpr[0], 0x0000000000007FFFULL)
        << "AX=0x7FFF (positive) extends to EAX=0x00007FFF";
}

// ============================================================================
// EXTENDED EMITTER COVERAGE, BATCH 3
//
// The remaining untested-but-supported families: x87 pop-arithmetic
// (FMULP/FSUBP/FDIVP -- the non-reverse forms, complementing the existing
// FADDP/FSUBRP/FDIVRP tests), the PREFETCH* hint no-ops, SFENCE, and the
// packed-double min/max plus the remaining packed bitwise variants
// (VANDPD/VANDNPD/VORPD). All encodings byte-verified.
//
// x87 setup recap: `fld qword[mem]` (DD /0) pushes; after loading A then B,
// ST(0)=B and ST(1)=A. `DE Cx/Ex/Fx` performs ST(1) = ST(1) <op> ST(0) and
// pops, leaving the result in ST(0); `fstp qword[mem]` (DD /3) stores it.
// ============================================================================

// ----------------------------------------------------------------------------
// x87 pop-arithmetic: non-reverse forms.
// ----------------------------------------------------------------------------

// FMULP st(1), st(0): ST(1) = A * B, pop. (A=6, B=7 -> 42.)
TEST_F(CpuRuntimeTest, X87_Fmulp_MultiplyAndPop) {
    u8* pa = mem.CodePtr() + 0x300;
    u8* pb = mem.CodePtr() + 0x310;
    u8* out = mem.CodePtr() + 0x320;
    const double A = 6.0, B = 7.0;
    std::memcpy(pa, &A, 8); std::memcpy(pb, &B, 8);
    u8 program[] = {
        0x48, 0xb8, 0,0,0,0,0,0,0,0,   // mov rax, &A
        0x48, 0xb9, 0,0,0,0,0,0,0,0,   // mov rcx, &B
        0x48, 0xba, 0,0,0,0,0,0,0,0,   // mov rdx, &out
        0xdd, 0x00,                    // fld qword[rax]  -> ST0=A
        0xdd, 0x01,                    // fld qword[rcx]  -> ST0=B, ST1=A
        0xde, 0xc9,                    // fmulp st1,st0   -> ST1=A*B, pop
        0xdd, 0x1a,                    // fstp qword[rdx]
        0xc3,
    };
    const u64 a=reinterpret_cast<u64>(pa),b=reinterpret_cast<u64>(pb),o=reinterpret_cast<u64>(out);
    std::memcpy(&program[2],&a,8); std::memcpy(&program[12],&b,8); std::memcpy(&program[22],&o,8);
    const auto r = RunProgram(program, sizeof(program), mem);
    double res; std::memcpy(&res, out, 8);
    EXPECT_EQ(res, 42.0) << "fmulp: 6*7 = 42";
    EXPECT_EQ(r.state.exit_reason, static_cast<u32>(ExitReason::BlockEnd));
}

// FSUBP st(1), st(0): ST(1) = A - B, pop. (A=20, B=8 -> 12.)
TEST_F(CpuRuntimeTest, X87_Fsubp_SubtractAndPop) {
    u8* pa = mem.CodePtr() + 0x300;
    u8* pb = mem.CodePtr() + 0x310;
    u8* out = mem.CodePtr() + 0x320;
    const double A = 20.0, B = 8.0;
    std::memcpy(pa, &A, 8); std::memcpy(pb, &B, 8);
    u8 program[] = {
        0x48, 0xb8, 0,0,0,0,0,0,0,0,
        0x48, 0xb9, 0,0,0,0,0,0,0,0,
        0x48, 0xba, 0,0,0,0,0,0,0,0,
        0xdd, 0x00,
        0xdd, 0x01,
        0xde, 0xe9,                    // fsubp st1,st0 -> ST1 = ST1 - ST0 = A - B
        0xdd, 0x1a,
        0xc3,
    };
    const u64 a=reinterpret_cast<u64>(pa),b=reinterpret_cast<u64>(pb),o=reinterpret_cast<u64>(out);
    std::memcpy(&program[2],&a,8); std::memcpy(&program[12],&b,8); std::memcpy(&program[22],&o,8);
    const auto r = RunProgram(program, sizeof(program), mem);
    double res; std::memcpy(&res, out, 8);
    EXPECT_EQ(res, 12.0) << "fsubp: 20-8 = 12 (ST1-ST0, not reversed)";
    EXPECT_EQ(r.state.exit_reason, static_cast<u32>(ExitReason::BlockEnd));
}

// FDIVP st(1), st(0): ST(1) = A / B, pop. (A=84, B=4 -> 21.)
TEST_F(CpuRuntimeTest, X87_Fdivp_DivideAndPop) {
    u8* pa = mem.CodePtr() + 0x300;
    u8* pb = mem.CodePtr() + 0x310;
    u8* out = mem.CodePtr() + 0x320;
    const double A = 84.0, B = 4.0;
    std::memcpy(pa, &A, 8); std::memcpy(pb, &B, 8);
    u8 program[] = {
        0x48, 0xb8, 0,0,0,0,0,0,0,0,
        0x48, 0xb9, 0,0,0,0,0,0,0,0,
        0x48, 0xba, 0,0,0,0,0,0,0,0,
        0xdd, 0x00,
        0xdd, 0x01,
        0xde, 0xf9,                    // fdivp st1,st0 -> ST1 = ST1 / ST0 = A / B
        0xdd, 0x1a,
        0xc3,
    };
    const u64 a=reinterpret_cast<u64>(pa),b=reinterpret_cast<u64>(pb),o=reinterpret_cast<u64>(out);
    std::memcpy(&program[2],&a,8); std::memcpy(&program[12],&b,8); std::memcpy(&program[22],&o,8);
    const auto r = RunProgram(program, sizeof(program), mem);
    double res; std::memcpy(&res, out, 8);
    EXPECT_EQ(res, 21.0) << "fdivp: 84/4 = 21 (ST1/ST0, not reversed)";
    EXPECT_EQ(r.state.exit_reason, static_cast<u32>(ExitReason::BlockEnd));
}

// ----------------------------------------------------------------------------
// PREFETCH* / SFENCE -- architecturally inert. The test proves they decode,
// emit, and let execution flow to a following instruction that mutates state.
// PREFETCH must NOT fault even on a wild address (it's a pure hint), so we
// point it at a deliberately bogus pointer and confirm the trailing MOV ran.
// Encoding (modrm with rax base): 0F 18 /n  ; prefetchw is 0F 0D /1.
// ----------------------------------------------------------------------------

#define PREFETCH_TEST(NAME, OPC1, OPC2, MODRM)                               \
    TEST_F(CpuRuntimeTest, PrefetchHint_##NAME) {                           \
        const u8 program[] = {                                             \
            0x48, 0xb8, 0x39,0x05,0,0,0,0,0,0, /* mov rax, 0x539 (bogus) */ \
            OPC1, OPC2, MODRM,                 /* prefetch* [rax] */        \
            0x48, 0xc7, 0xc3, 0x2a,0,0,0,      /* mov rbx, 42 */            \
            0xc3,                                                          \
        };                                                                 \
        const auto r = RunProgram(program, sizeof(program), mem);          \
        EXPECT_EQ(r.state.gpr[3], 42u) << #NAME " is a no-op; flow continues"; \
        EXPECT_EQ(r.state.gpr[0], 0x539u) << "bogus prefetch addr untouched";  \
        EXPECT_EQ(r.state.exit_reason, static_cast<u32>(ExitReason::BlockEnd)); \
    }

PREFETCH_TEST(Prefetcht0, 0x0f, 0x18, 0x08)
PREFETCH_TEST(Prefetcht1, 0x0f, 0x18, 0x10)
PREFETCH_TEST(Prefetcht2, 0x0f, 0x18, 0x18)
PREFETCH_TEST(Prefetchw,  0x0f, 0x0d, 0x08)

#undef PREFETCH_TEST

// SFENCE (0F AE F8): store fence, inert for our single-threaded model.
TEST_F(CpuRuntimeTest, Sfence_IsNoOp_ExecutionContinues) {
    const u8 program[] = {
        0x0f, 0xae, 0xf8,                  // sfence
        0x48, 0xc7, 0xc0, 0x63,0,0,0,      // mov rax, 99
        0xc3,
    };
    const auto r = RunProgram(program, sizeof(program), mem);
    EXPECT_EQ(r.state.gpr[0], 99u);
    EXPECT_EQ(r.state.exit_reason, static_cast<u32>(ExitReason::BlockEnd));
}

// ----------------------------------------------------------------------------
// Packed-double min/max and the remaining packed bitwise variants. Reuses the
// RunPackedFp / PackD / XmmLaneD helpers defined in batch 2.
// ----------------------------------------------------------------------------

TEST_F(CpuRuntimeTest, PackedFp_Vminpd_PerLaneMin) {
    // vminpd xmm0, xmm1, xmm2 : c5 f1 5d c2
    const u8 program[] = {0xc5, 0xf1, 0x5d, 0xc2, 0xc3};
    const auto st = RunPackedFp(program, sizeof(program), mem,
        PackD(3.0), PackD(50.0), PackD(9.0), PackD(20.0));
    EXPECT_EQ(XmmLaneD(st, 0), 3.0);
    EXPECT_EQ(XmmLaneD(st, 1), 20.0);
    EXPECT_EQ(st.ymm[2], 0ULL);
    EXPECT_EQ(st.ymm[3], 0ULL);
}

TEST_F(CpuRuntimeTest, PackedFp_Vmaxps_PerLaneMax) {
    // vmaxps xmm0, xmm1, xmm2 : c5 f0 5f c2
    const u8 program[] = {0xc5, 0xf0, 0x5f, 0xc2, 0xc3};
    const auto st = RunPackedFp(program, sizeof(program), mem,
        PackF2(1.0f, 50.0f), PackF2(3.0f, 8.0f),
        PackF2(9.0f, 20.0f), PackF2(30.0f, 2.0f));
    EXPECT_EQ(XmmLaneF(st, 0), 9.0f);
    EXPECT_EQ(XmmLaneF(st, 1), 50.0f);
    EXPECT_EQ(XmmLaneF(st, 2), 30.0f);
    EXPECT_EQ(XmmLaneF(st, 3), 8.0f);
}

TEST_F(CpuRuntimeTest, PackedFp_Vandpd_BitwiseAnd) {
    // vandpd xmm0, xmm1, xmm2 : c5 f1 54 c2
    const u8 program[] = {0xc5, 0xf1, 0x54, 0xc2, 0xc3};
    const auto st = RunPackedFp(program, sizeof(program), mem,
        0xFFFFFFFF00000000ULL, 0x0F0F0F0FF0F0F0F0ULL,
        0x123456789ABCDEF0ULL, 0xFFFFFFFFFFFFFFFFULL);
    EXPECT_EQ(st.ymm[0], 0xFFFFFFFF00000000ULL & 0x123456789ABCDEF0ULL);
    EXPECT_EQ(st.ymm[1], 0x0F0F0F0FF0F0F0F0ULL & 0xFFFFFFFFFFFFFFFFULL);
    EXPECT_EQ(st.ymm[2], 0ULL);
}

TEST_F(CpuRuntimeTest, PackedFp_Vandnpd_NotSrc1AndSrc2) {
    // vandnpd xmm0, xmm1, xmm2 : c5 f1 55 c2  -> (~xmm1) & xmm2
    const u8 program[] = {0xc5, 0xf1, 0x55, 0xc2, 0xc3};
    const auto st = RunPackedFp(program, sizeof(program), mem,
        0xFF00FF00FF00FF00ULL, 0x0000000000000000ULL,
        0x123456789ABCDEF0ULL, 0xCAFEF00DBAADF00DULL);
    EXPECT_EQ(st.ymm[0], (~0xFF00FF00FF00FF00ULL) & 0x123456789ABCDEF0ULL);
    EXPECT_EQ(st.ymm[1], (~0x0000000000000000ULL) & 0xCAFEF00DBAADF00DULL);
}

TEST_F(CpuRuntimeTest, PackedFp_Vorpd_BitwiseOr) {
    // vorpd xmm0, xmm1, xmm2 : c5 f1 56 c2
    const u8 program[] = {0xc5, 0xf1, 0x56, 0xc2, 0xc3};
    const auto st = RunPackedFp(program, sizeof(program), mem,
        0x00000000FFFF0000ULL, 0x1010101010101010ULL,
        0xFFFF000000000000ULL, 0x0101010101010101ULL);
    EXPECT_EQ(st.ymm[0], 0x00000000FFFF0000ULL | 0xFFFF000000000000ULL);
    EXPECT_EQ(st.ymm[1], 0x1010101010101010ULL | 0x0101010101010101ULL);
}

// ----------------------------------------------------------------------------
// Min/Max non-commutative NaN-propagation corner: SSE min/max return the
// SECOND operand when either input is NaN (Intel's defined behavior). This
// pins down that the host instruction's NaN semantics flow through unchanged.
// vminps xmm0, xmm1(NaN in lane0), xmm2 -> lane0 takes xmm2's value.
// ----------------------------------------------------------------------------
TEST_F(CpuRuntimeTest, PackedFp_Vminps_NanTakesSecondOperand) {
    const u8 program[] = {0xc5, 0xf0, 0x5d, 0xc2, 0xc3}; // vminps xmm0,xmm1,xmm2
    const float nan = std::numeric_limits<float>::quiet_NaN();
    const auto st = RunPackedFp(program, sizeof(program), mem,
        PackF2(nan, 5.0f), PackF2(1.0f, 2.0f),     // xmm1 lane0 = NaN
        PackF2(7.0f, 99.0f), PackF2(3.0f, 4.0f));  // xmm2
    // lane0: min(NaN, 7.0) -> 7.0 (second operand on unordered)
    EXPECT_EQ(XmmLaneF(st, 0), 7.0f) << "NaN in src1 -> result is src2";
    // lane1: min(5.0, 99.0) -> 5.0 (ordinary)
    EXPECT_EQ(XmmLaneF(st, 1), 5.0f);
}

// ============================================================================
// EXTENDED EMITTER COVERAGE, BATCH 4 -- ADDRESSING MODES
//
// The earlier batches covered opcodes; this batch exercises the *address
// computation* (EmitEffectiveAddress) across the SIB and displacement forms
// that the per-opcode tests didn't reach: base+index*scale, scaled indices,
// disp8 / disp32 / negative displacement, and the memory-destination /
// memory-source ALU forms layered on top. A scratch qword is staged in a
// known slot of the guest code page and addressed via a base register so the
// effective-address math is what's under test, not the opcode.
//
// All encodings byte-verified. Scratch convention: we place data at
// CodePtr()+0x200 (well past the few program bytes at CodePtr()) and load a
// base register with that address.
// ============================================================================

namespace {
// Common setup: program at CodePtr(), a base pointer in a chosen GPR, run.
GuestState RunAddrTest(const u8* program, size_t n, GuestMemory& mem,
                       std::function<void(GuestState&, u8* scratch)> setup) {
    std::memcpy(mem.CodePtr(), program, n);
    u8* guest_rsp = mem.StackTop() - 8;
    *reinterpret_cast<u64*>(guest_rsp) = kReturnSentinel;
    GuestState st{};
    st.rip = reinterpret_cast<u64>(mem.CodePtr());
    st.gpr[4] = reinterpret_cast<u64>(guest_rsp);
    setup(st, mem.CodePtr() + 0x200);
    Runtime rt;
    rt.Run(st);
    return st;
}
} // namespace

// ----------------------------------------------------------------------------
// SIB: base + index*scale load. mov rax, [rcx + rdx*4]  (48 8b 04 91).
// Place a marker qword at base + idx*4 and confirm it is loaded.
// ----------------------------------------------------------------------------
TEST_F(CpuRuntimeTest, Addr_SibBaseIndexScale4_Loads) {
    const u8 program[] = {0x48, 0x8b, 0x04, 0x91, 0xc3}; // mov rax,[rcx+rdx*4]
    const auto st = RunAddrTest(program, sizeof(program), mem,
        [](GuestState& s, u8* scratch) {
            // base=rcx=scratch, index=rdx=3, scale=4 -> scratch+12
            *reinterpret_cast<u64*>(scratch + 12) = 0xCAFEF00DBAADF00DULL;
            s.gpr[1] = reinterpret_cast<u64>(scratch); // rcx
            s.gpr[2] = 3;                               // rdx
            s.gpr[0] = 0;                               // rax poison
        });
    EXPECT_EQ(st.gpr[0], 0xCAFEF00DBAADF00DULL)
        << "loaded from base + index*4";
}

// SIB with scale 8 and disp8. mov rax, [rcx + rdx*8 + 0x10] (48 8b 44 d1 10).
TEST_F(CpuRuntimeTest, Addr_SibScale8Disp8_Loads) {
    const u8 program[] = {0x48, 0x8b, 0x44, 0xd1, 0x10, 0xc3};
    const auto st = RunAddrTest(program, sizeof(program), mem,
        [](GuestState& s, u8* scratch) {
            // scratch + rdx*8 + 0x10; rdx=2 -> scratch + 16 + 16 = scratch+32
            *reinterpret_cast<u64*>(scratch + 32) = 0x1122334455667788ULL;
            s.gpr[1] = reinterpret_cast<u64>(scratch);
            s.gpr[2] = 2;
        });
    EXPECT_EQ(st.gpr[0], 0x1122334455667788ULL)
        << "base + index*8 + disp8";
}

// SIB store: mov [rcx + rdx*2], rax  (48 89 04 51).
TEST_F(CpuRuntimeTest, Addr_SibScale2Store_Writes) {
    const u8 program[] = {0x48, 0x89, 0x04, 0x51, 0xc3};
    u8* scratch_captured = nullptr;
    const auto st = RunAddrTest(program, sizeof(program), mem,
        [&](GuestState& s, u8* scratch) {
            scratch_captured = scratch;
            *reinterpret_cast<u64*>(scratch + 10) = 0; // base + rdx*2 (rdx=5) -> +10
            s.gpr[1] = reinterpret_cast<u64>(scratch);
            s.gpr[2] = 5;
            s.gpr[0] = 0xABCDEF0123456789ULL; // rax = value to store
        });
    ASSERT_NE(scratch_captured, nullptr);
    EXPECT_EQ(*reinterpret_cast<u64*>(scratch_captured + 10), 0xABCDEF0123456789ULL)
        << "stored at base + index*2";
}

// ----------------------------------------------------------------------------
// Displacement forms: disp8 positive, disp32, negative disp8.
// ----------------------------------------------------------------------------
TEST_F(CpuRuntimeTest, Addr_Disp8Positive_Loads) {
    const u8 program[] = {0x48, 0x8b, 0x41, 0x7f, 0xc3}; // mov rax,[rcx+0x7f]
    const auto st = RunAddrTest(program, sizeof(program), mem,
        [](GuestState& s, u8* scratch) {
            *reinterpret_cast<u64*>(scratch + 0x7f) = 0xDEADBEEF00000001ULL;
            s.gpr[1] = reinterpret_cast<u64>(scratch);
        });
    EXPECT_EQ(st.gpr[0], 0xDEADBEEF00000001ULL) << "base + disp8";
}

TEST_F(CpuRuntimeTest, Addr_Disp32_Loads) {
    const u8 program[] = {0x48, 0x8b, 0x81, 0x00,0x01,0x00,0x00, 0xc3}; // [rcx+0x100]
    const auto st = RunAddrTest(program, sizeof(program), mem,
        [](GuestState& s, u8* scratch) {
            *reinterpret_cast<u64*>(scratch + 0x100) = 0x0123456789ABCDEFULL;
            s.gpr[1] = reinterpret_cast<u64>(scratch);
        });
    EXPECT_EQ(st.gpr[0], 0x0123456789ABCDEFULL) << "base + disp32";
}

TEST_F(CpuRuntimeTest, Addr_NegativeDisp8_Loads) {
    const u8 program[] = {0x48, 0x8b, 0x41, 0x80, 0xc3}; // mov rax,[rcx-0x80]
    const auto st = RunAddrTest(program, sizeof(program), mem,
        [](GuestState& s, u8* scratch) {
            // base = scratch + 0x80, disp = -0x80 -> scratch
            *reinterpret_cast<u64*>(scratch) = 0xFEEDFACECAFEBEEFULL;
            s.gpr[1] = reinterpret_cast<u64>(scratch) + 0x80;
        });
    EXPECT_EQ(st.gpr[0], 0xFEEDFACECAFEBEEFULL) << "base + negative disp8";
}

// ----------------------------------------------------------------------------
// LEA with full SIB+disp: lea rax, [rcx + rdx*4 + 0x20]  (48 8d 44 91 20).
// LEA computes the address without dereferencing; the result is pure
// arithmetic, so we can check it exactly without staging memory.
// ----------------------------------------------------------------------------
TEST_F(CpuRuntimeTest, Addr_LeaSibDisp_ComputesAddress) {
    const u8 program[] = {0x48, 0x8d, 0x44, 0x91, 0x20, 0xc3};
    const auto st = RunAddrTest(program, sizeof(program), mem,
        [](GuestState& s, u8* /*scratch*/) {
            s.gpr[1] = 0x100000;  // rcx (base)
            s.gpr[2] = 0x10;      // rdx (index)
        });
    // 0x100000 + 0x10*4 + 0x20 = 0x100000 + 0x40 + 0x20 = 0x100060
    EXPECT_EQ(st.gpr[0], 0x100060ULL) << "base + index*4 + disp";
}

// ----------------------------------------------------------------------------
// Memory-destination ALU: ADD / SUB write back to [mem] and set flags.
// ----------------------------------------------------------------------------
TEST_F(CpuRuntimeTest, Addr_AddMemDest_WritesBackAndFlags) {
    const u8 program[] = {0x48, 0x01, 0x01, 0xc3}; // add [rcx], rax
    u8* cap = nullptr;
    const auto st = RunAddrTest(program, sizeof(program), mem,
        [&](GuestState& s, u8* scratch) {
            cap = scratch;
            *reinterpret_cast<u64*>(scratch) = 0x1000;
            s.gpr[1] = reinterpret_cast<u64>(scratch); // rcx
            s.gpr[0] = 0x0234;                          // rax
        });
    ASSERT_NE(cap, nullptr);
    EXPECT_EQ(*reinterpret_cast<u64*>(cap), 0x1234ULL) << "[mem] += rax";
    EXPECT_FALSE(st.rflags & (1ULL << 6)) << "ZF clear (nonzero result)";
}

TEST_F(CpuRuntimeTest, Addr_SubMemDest_ToZero_SetsZf) {
    const u8 program[] = {0x48, 0x29, 0x01, 0xc3}; // sub [rcx], rax
    u8* cap = nullptr;
    const auto st = RunAddrTest(program, sizeof(program), mem,
        [&](GuestState& s, u8* scratch) {
            cap = scratch;
            *reinterpret_cast<u64*>(scratch) = 0x5555;
            s.gpr[1] = reinterpret_cast<u64>(scratch);
            s.gpr[0] = 0x5555; // equal -> result 0
        });
    ASSERT_NE(cap, nullptr);
    EXPECT_EQ(*reinterpret_cast<u64*>(cap), 0ULL) << "[mem] -= rax -> 0";
    EXPECT_TRUE(st.rflags & (1ULL << 6)) << "ZF set on zero result";
}

// ----------------------------------------------------------------------------
// CMP in both memory directions -- it sets flags without writing memory.
// ----------------------------------------------------------------------------
TEST_F(CpuRuntimeTest, Addr_CmpRegMem_Equal_SetsZf) {
    const u8 program[] = {0x48, 0x3b, 0x01, 0xc3}; // cmp rax, [rcx]
    u8* cap = nullptr;
    const auto st = RunAddrTest(program, sizeof(program), mem,
        [&](GuestState& s, u8* scratch) {
            cap = scratch;
            *reinterpret_cast<u64*>(scratch) = 0xABCD;
            s.gpr[1] = reinterpret_cast<u64>(scratch);
            s.gpr[0] = 0xABCD; // equal
        });
    ASSERT_NE(cap, nullptr);
    EXPECT_EQ(*reinterpret_cast<u64*>(cap), 0xABCDULL) << "cmp must not write memory";
    EXPECT_TRUE(st.rflags & (1ULL << 6)) << "ZF set: rax == [mem]";
    EXPECT_FALSE(st.rflags & (1ULL << 0)) << "CF clear on equal";
}

TEST_F(CpuRuntimeTest, Addr_CmpMemReg_Smaller_SetsCf) {
    const u8 program[] = {0x48, 0x39, 0x01, 0xc3}; // cmp [rcx], rax
    const auto st = RunAddrTest(program, sizeof(program), mem,
        [](GuestState& s, u8* scratch) {
            *reinterpret_cast<u64*>(scratch) = 0x10; // [mem]
            s.gpr[1] = reinterpret_cast<u64>(scratch);
            s.gpr[0] = 0x100;                         // rax (bigger)
        });
    // [mem] - rax = 0x10 - 0x100 -> borrow -> CF=1, ZF=0
    EXPECT_TRUE(st.rflags & (1ULL << 0)) << "CF set: [mem] < rax";
    EXPECT_FALSE(st.rflags & (1ULL << 6)) << "ZF clear";
}

// ----------------------------------------------------------------------------
// 32-bit load from a SIB address zero-extends into the 64-bit register.
// mov eax, [rcx + rdx*4]  (8b 04 91, no REX.W).
// ----------------------------------------------------------------------------
TEST_F(CpuRuntimeTest, Addr_Sib32BitLoad_ZeroExtends) {
    const u8 program[] = {0x8b, 0x04, 0x91, 0xc3}; // mov eax,[rcx+rdx*4]
    const auto st = RunAddrTest(program, sizeof(program), mem,
        [](GuestState& s, u8* scratch) {
            // store a full 64-bit pattern; the 32-bit load takes only low 32
            *reinterpret_cast<u64*>(scratch + 16) = 0xFFFFFFFF89ABCDEFULL;
            s.gpr[1] = reinterpret_cast<u64>(scratch);
            s.gpr[2] = 4; // rdx*4 = 16
            s.gpr[0] = 0xDEADBEEFDEADBEEFULL; // rax poison
        });
    EXPECT_EQ(st.gpr[0], 0x0000000089ABCDEFULL)
        << "32-bit load takes low dword and zero-extends bits 63:32";
}

// ============================================================================
// EXTENDED EMITTER COVERAGE, BATCH 5 -- NARROW MEM, RIP-RELATIVE, REJECTIONS
//
// Three layers:
//  (1) Narrow-width memory-destination ALU (8/16/32-bit add/sub [mem],reg)
//      and narrow memory-source loads (movzx from byte/word), with flag
//      precision -- the per-opcode tests mostly used 64-bit or register forms.
//  (2) The two EmitEffectiveAddress paths distinct from base+index: RIP-
//      relative ([rip+disp], constant-folded to next_rip+disp) and the plain
//      absolute [disp32] form (no base, no index).
//  (3) Rejection paths: a segment-override prefix (FS/GS) must NOT be
//      miscompiled -- it must cleanly reach the UnsupportedInstruction exit
//      with RIP pointing at the offending instruction.
//
// All encodings byte-verified. Reuses RunAddrTest from batch 4 for the
// base-register-relative cases.
// ============================================================================

// ----------------------------------------------------------------------------
// (1) Narrow memory-destination ALU.
// ----------------------------------------------------------------------------

// add byte[rcx], al  (00 01). Only the low byte of [mem] changes; flags from
// the 8-bit result.
TEST_F(CpuRuntimeTest, NarrowMem_AddByteDest_WrapsAndSetsFlags) {
    const u8 program[] = {0x00, 0x01, 0xc3};
    u8* cap = nullptr;
    const auto st = RunAddrTest(program, sizeof(program), mem,
        [&](GuestState& s, u8* scratch) {
            cap = scratch;
            // surround byte with sentinel so we can prove only 1 byte changed
            *reinterpret_cast<u64*>(scratch) = 0xAAAAAAAAAAAAAA01ULL;
            s.gpr[1] = reinterpret_cast<u64>(scratch); // rcx = &mem
            s.gpr[0] = 0xFF;                            // al = 0xFF
        });
    ASSERT_NE(cap, nullptr);
    // 0x01 + 0xFF = 0x00 (carry out), only low byte written.
    EXPECT_EQ(*reinterpret_cast<u64*>(cap), 0xAAAAAAAAAAAAAA00ULL)
        << "only the addressed byte changes";
    EXPECT_TRUE(st.rflags & (1ULL << 6)) << "ZF set (8-bit result is 0)";
    EXPECT_TRUE(st.rflags & (1ULL << 0)) << "CF set (carry out of bit 7)";
}

// add word[rcx], ax  (66 01 01). 16-bit write-back, upper bytes preserved.
TEST_F(CpuRuntimeTest, NarrowMem_AddWordDest_PreservesUpper) {
    const u8 program[] = {0x66, 0x01, 0x01, 0xc3};
    u8* cap = nullptr;
    const auto st = RunAddrTest(program, sizeof(program), mem,
        [&](GuestState& s, u8* scratch) {
            cap = scratch;
            *reinterpret_cast<u64*>(scratch) = 0xBBBBBBBBBBBB1000ULL;
            s.gpr[1] = reinterpret_cast<u64>(scratch);
            s.gpr[0] = 0x0234; // ax
        });
    ASSERT_NE(cap, nullptr);
    EXPECT_EQ(*reinterpret_cast<u64*>(cap), 0xBBBBBBBBBBBB1234ULL)
        << "only the low word changes";
}

// add dword[rcx], eax  (01 01). 32-bit write-back; upper 32 of the 8-byte
// slot are left as the original memory held (memory store is 4 bytes).
TEST_F(CpuRuntimeTest, NarrowMem_AddDwordDest_Writes4Bytes) {
    const u8 program[] = {0x01, 0x01, 0xc3};
    u8* cap = nullptr;
    const auto st = RunAddrTest(program, sizeof(program), mem,
        [&](GuestState& s, u8* scratch) {
            cap = scratch;
            *reinterpret_cast<u64*>(scratch) = 0xCCCCCCCC00001000ULL;
            s.gpr[1] = reinterpret_cast<u64>(scratch);
            s.gpr[0] = 0x00000234;
        });
    ASSERT_NE(cap, nullptr);
    EXPECT_EQ(*reinterpret_cast<u64*>(cap), 0xCCCCCCCC00001234ULL)
        << "low dword updated, high dword untouched";
}

// sub byte[rcx], al  (28 01) to zero -> ZF.
TEST_F(CpuRuntimeTest, NarrowMem_SubByteDest_ToZero_SetsZf) {
    const u8 program[] = {0x28, 0x01, 0xc3};
    u8* cap = nullptr;
    const auto st = RunAddrTest(program, sizeof(program), mem,
        [&](GuestState& s, u8* scratch) {
            cap = scratch;
            *reinterpret_cast<u64*>(scratch) = 0x1111111111111142ULL;
            s.gpr[1] = reinterpret_cast<u64>(scratch);
            s.gpr[0] = 0x42; // al == low byte -> 0
        });
    ASSERT_NE(cap, nullptr);
    EXPECT_EQ(*reinterpret_cast<u64*>(cap), 0x1111111111111100ULL);
    EXPECT_TRUE(st.rflags & (1ULL << 6)) << "ZF set";
}

// ----------------------------------------------------------------------------
// (1b) Narrow memory-source: MOVZX zero-extends a byte / word load.
// ----------------------------------------------------------------------------

// movzx eax, byte[rcx]  (0f b6 01). High bytes of source ignored; result
// zero-extended into RAX.
TEST_F(CpuRuntimeTest, NarrowMem_MovzxByteLoad_ZeroExtends) {
    const u8 program[] = {0x0f, 0xb6, 0x01, 0xc3};
    const auto st = RunAddrTest(program, sizeof(program), mem,
        [](GuestState& s, u8* scratch) {
            *reinterpret_cast<u64*>(scratch) = 0xFFFFFFFFFFFFFF80ULL;
            s.gpr[1] = reinterpret_cast<u64>(scratch);
            s.gpr[0] = 0xDEADBEEFDEADBEEFULL; // poison
        });
    EXPECT_EQ(st.gpr[0], 0x0000000000000080ULL)
        << "byte 0x80 zero-extended (not sign-extended) to 64";
}

// movzx eax, word[rcx]  (0f b7 01).
TEST_F(CpuRuntimeTest, NarrowMem_MovzxWordLoad_ZeroExtends) {
    const u8 program[] = {0x0f, 0xb7, 0x01, 0xc3};
    const auto st = RunAddrTest(program, sizeof(program), mem,
        [](GuestState& s, u8* scratch) {
            *reinterpret_cast<u64*>(scratch) = 0xFFFFFFFFFFFF8000ULL;
            s.gpr[1] = reinterpret_cast<u64>(scratch);
            s.gpr[0] = 0xDEADBEEFDEADBEEFULL;
        });
    EXPECT_EQ(st.gpr[0], 0x0000000000008000ULL)
        << "word 0x8000 zero-extended to 64";
}

// ----------------------------------------------------------------------------
// (2) RIP-relative addressing: mov rax, [rip + disp]. The lifter constant-
// folds this to next_rip + disp. We lay out the program so the target slot
// sits at a known offset and seed it.
//
//   offset 0: mov rax, [rip + disp]   (7 bytes: 48 8b 05 <disp32>)
//   offset 7: ret                     (1 byte)
//   next_rip after the mov = base + 7. We want the target at base + 0x40,
//   so disp = 0x40 - 7 = 0x39.
// ----------------------------------------------------------------------------
TEST_F(CpuRuntimeTest, Addr_RipRelative_Loads) {
    const u8 program[] = {
        0x48, 0x8b, 0x05, 0x39, 0x00, 0x00, 0x00, // mov rax, [rip+0x39]
        0xc3,
    };
    std::memcpy(mem.CodePtr(), program, sizeof(program));
    // target = base + 7 (next_rip) + 0x39 = base + 0x40
    *reinterpret_cast<u64*>(mem.CodePtr() + 0x40) = 0x1234ABCD5678EF00ULL;
    u8* guest_rsp = mem.StackTop() - 8;
    *reinterpret_cast<u64*>(guest_rsp) = kReturnSentinel;
    GuestState st{};
    st.rip = reinterpret_cast<u64>(mem.CodePtr());
    st.gpr[4] = reinterpret_cast<u64>(guest_rsp);
    st.gpr[0] = 0; // rax poison
    Runtime rt;
    rt.Run(st);
    EXPECT_EQ(st.gpr[0], 0x1234ABCD5678EF00ULL)
        << "RIP-relative load folded to next_rip + disp";
    EXPECT_EQ(st.exit_reason, static_cast<u32>(ExitReason::BlockEnd));
}

// ----------------------------------------------------------------------------
// (2b) Absolute [disp32]: mov rax, [0x20000000]  (48 8b 04 25 <disp32>).
// The lifter folds this to the literal absolute address, so we must map a
// page at exactly that address (MapLowScratch maps 0x20000000).
// ----------------------------------------------------------------------------
TEST_F(CpuRuntimeTest, Addr_AbsoluteDisp32_Loads) {
    u8* low = MapLowScratch();
    if (low == nullptr) GTEST_SKIP() << "no low mapping at 0x20000000";
    ASSERT_EQ(reinterpret_cast<uintptr_t>(low), 0x20000000ull)
        << "MapLowScratch returned the expected fixed address";
    *reinterpret_cast<u64*>(low) = 0xFACEFEEDD00DCAFEULL;

    // mov rax, [0x20000000] : 48 8b 04 25 00 00 00 20
    const u8 program[] = {
        0x48, 0x8b, 0x04, 0x25, 0x00, 0x00, 0x00, 0x20,
        0xc3,
    };
    std::memcpy(mem.CodePtr(), program, sizeof(program));
    u8* guest_rsp = mem.StackTop() - 8;
    *reinterpret_cast<u64*>(guest_rsp) = kReturnSentinel;
    GuestState st{};
    st.rip = reinterpret_cast<u64>(mem.CodePtr());
    st.gpr[4] = reinterpret_cast<u64>(guest_rsp);
    st.gpr[0] = 0;
    Runtime rt;
    rt.Run(st);
    EXPECT_EQ(st.gpr[0], 0xFACEFEEDD00DCAFEULL)
        << "absolute [disp32] load";
#if !defined(_WIN32)
    ::munmap(low, 4096);
#endif
}

// ----------------------------------------------------------------------------
// (3) Rejection: a segment-override prefix on a memory operand is not
// supported and must reach the UnsupportedInstruction exit cleanly, with RIP
// left at the offending instruction (so a host emulator could diagnose it),
// rather than being silently miscompiled.
//
// Layout: a harmless MOV first (proves the block starts), then the FS-prefixed
// load. The unsupported exit should report RIP at the FS instruction's offset.
// ----------------------------------------------------------------------------
TEST_F(CpuRuntimeTest, Reject_FsSegmentOverride_UnsupportedExit) {
    const u8 program[] = {
        0x48, 0xc7, 0xc0, 0x07, 0x00, 0x00, 0x00, // mov rax, 7   (offset 0..6)
        0x64, 0x48, 0x8b, 0x09,                    // mov rcx, fs:[rcx] (offset 7)
        0xc3,
    };
    std::memcpy(mem.CodePtr(), program, sizeof(program));
    u8* guest_rsp = mem.StackTop() - 8;
    *reinterpret_cast<u64*>(guest_rsp) = kReturnSentinel;
    GuestState st{};
    const u64 base = reinterpret_cast<u64>(mem.CodePtr());
    st.rip = base;
    st.gpr[4] = reinterpret_cast<u64>(guest_rsp);
    Runtime rt;
    rt.Run(st);
    EXPECT_EQ(st.exit_reason, static_cast<u32>(ExitReason::UnsupportedInstruction));
    EXPECT_EQ(st.rip, base + 7) << "RIP points at the FS-prefixed instruction";
    EXPECT_EQ(st.gpr[0], 7u) << "the MOV before the unsupported insn ran";
}

// GS override: same expectation, different prefix byte (65).
TEST_F(CpuRuntimeTest, Reject_GsSegmentOverride_UnsupportedExit) {
    const u8 program[] = {
        0x48, 0xc7, 0xc0, 0x09, 0x00, 0x00, 0x00, // mov rax, 9
        0x65, 0x48, 0x8b, 0x09,                    // mov rcx, gs:[rcx]
        0xc3,
    };
    std::memcpy(mem.CodePtr(), program, sizeof(program));
    u8* guest_rsp = mem.StackTop() - 8;
    *reinterpret_cast<u64*>(guest_rsp) = kReturnSentinel;
    GuestState st{};
    const u64 base = reinterpret_cast<u64>(mem.CodePtr());
    st.rip = base;
    st.gpr[4] = reinterpret_cast<u64>(guest_rsp);
    Runtime rt;
    rt.Run(st);
    EXPECT_EQ(st.exit_reason, static_cast<u32>(ExitReason::UnsupportedInstruction));
    EXPECT_EQ(st.rip, base + 7) << "RIP points at the GS-prefixed instruction";
    EXPECT_EQ(st.gpr[0], 9u);
}

// ----------------------------------------------------------------------------
// Diagnostics-off invariant: in the default build (SHADPS4_RUNTIME_DIAGNOSTICS
// undefined) the dispatcher's diagnostics hooks are inline no-ops. A program
// that crosses block boundaries, compiles, and loops -- hitting every hook
// site -- must produce exactly the result the pure computation predicts, i.e.
// the hooks perturb nothing. (When diagnostics ARE compiled in, this still
// passes; it just also logs.)
// ----------------------------------------------------------------------------
TEST_F(CpuRuntimeTest, DiagnosticsHooks_DoNotPerturbExecution) {
    // Same loop as the cache-reuse test (crosses boundaries + repeats a block)
    // plus a forward structure, with a deterministic expected result.
    const u8 program[] = {
        0x48, 0x31, 0xc0,             // xor rax, rax
        0x48, 0xc7, 0xc1, 0x05,0,0,0, // mov rcx, 5
        0x48, 0x01, 0xc8,             // L: add rax, rcx
        0x48, 0xff, 0xc9,             // dec rcx
        0x75, 0xf8,                   // jnz L
        0xc3,
    };
    const auto r = RunProgram(program, sizeof(program), mem);
    EXPECT_EQ(r.state.gpr[0], 15ULL) << "5+4+3+2+1, unaffected by diagnostics hooks";
    EXPECT_EQ(r.state.exit_reason, static_cast<u32>(ExitReason::BlockEnd));
}


// ============================================================================
// Sign-extension convert family: CBW and CWD were implemented in EmitConvert
// but had no tests; CQO had only a negative case. These are partial-register
// operations (CBW writes AH while preserving RAX bits 63:16; CWD fills DX while
// preserving RDX bits 63:16), which are a classic source of upper-bit bugs.
// All expectations below were cross-checked against the emitter under QEMU.
// ============================================================================

// CBW: AL -> AX. Sign-extend AL into AH (bits 15:8), preserve RAX bits 63:16.
TEST_F(CpuRuntimeTest, Cbw_SignExtendsNegativeAl) {
    const u8 program[] = {
        0x48, 0xb8, 0x80,0x00, 0x34,0x12, 0xEF,0xBE,0xAD,0xDE, // mov rax,0xDEADBEEF12340080
        0x66, 0x98, // cbw
        0xc3,
    };
    const auto r = RunProgram(program, sizeof(program), mem);
    // AL=0x80 (negative) -> AH=0xFF; bits 63:16 unchanged.
    EXPECT_EQ(r.state.gpr[0], 0xDEADBEEF1234FF80ULL)
        << "AL=0x80 sign-extends to AX=0xFF80, upper 48 bits preserved";
}

// CBW positive: AL bit7 clear -> AH cleared (and any prior AH is overwritten).
TEST_F(CpuRuntimeTest, Cbw_PositiveAl_ClearsAh) {
    const u8 program[] = {
        0x48, 0xb8, 0x7F,0xFF, 0x55,0x44, 0x33,0x22,0x11,0x00, // mov rax,0x001122334455FF7F
        0x66, 0x98, // cbw
        0xc3,
    };
    const auto r = RunProgram(program, sizeof(program), mem);
    // AL=0x7F (positive) -> AH=0x00 (overwriting the prior 0xFF); upper preserved.
    EXPECT_EQ(r.state.gpr[0], 0x001122334455007FULL)
        << "AL=0x7F sign-extends to AX=0x007F; prior AH=0xFF must be cleared";
}

// CBW must not touch any register other than RAX. RDX/RCX are seeded with
// sentinels via the setup lambda; the program never writes them, so they must
// survive unchanged.
TEST_F(CpuRuntimeTest, Cbw_LeavesOtherRegistersUntouched) {
    const u8 program[] = {
        0x48, 0xb8, 0x80,0x00, 0x00,0x00, 0x00,0x00,0x00,0x00, // mov rax,0x80
        0x66, 0x98, // cbw
        0xc3,
    };
    const auto st = RunWithState(program, sizeof(program), mem,
        [](GuestState& s) {
            s.gpr[2] = 0xD0D0D0D0D0D0D0D0ULL; // rdx sentinel
            s.gpr[1] = 0xC1C1C1C1C1C1C1C1ULL; // rcx sentinel
        });
    EXPECT_EQ(st.gpr[0], 0xFF80ULL) << "AL=0x80 -> AX=0xFF80";
    EXPECT_EQ(st.gpr[2], 0xD0D0D0D0D0D0D0D0ULL) << "RDX untouched by CBW";
    EXPECT_EQ(st.gpr[1], 0xC1C1C1C1C1C1C1C1ULL) << "RCX untouched by CBW";
}

// CWD: AX -> DX:AX. DX = 0xFFFF when AX bit15 set; RDX bits 63:16 preserved;
// RAX unchanged.
TEST_F(CpuRuntimeTest, Cwd_NegativeAx_FillsDxWithOnes) {
    const u8 program[] = {
        0x48, 0xb8, 0x00,0x80, 0x00,0x00, 0x00,0x00,0x00,0x00, // mov rax,0x8000
        0x48, 0xba, 0x34,0x12, 0xDD,0xCC, 0xBB,0xAA,0x00,0x00, // mov rdx,0x0000AABBCCDD1234
        0x66, 0x99, // cwd
        0xc3,
    };
    const auto r = RunProgram(program, sizeof(program), mem);
    // AX=0x8000 (negative) -> DX=0xFFFF; RDX upper 48 preserved; RAX unchanged.
    EXPECT_EQ(r.state.gpr[2], 0x0000AABBCCDDFFFFULL)
        << "AX bit15 set -> DX=0xFFFF, RDX bits 63:16 preserved";
    EXPECT_EQ(r.state.gpr[0], 0x8000ULL) << "CWD must not modify RAX";
}

// CWD positive: AX bit15 clear -> DX cleared, RDX upper preserved.
TEST_F(CpuRuntimeTest, Cwd_PositiveAx_ClearsDx) {
    const u8 program[] = {
        0x48, 0xb8, 0xFF,0x7F, 0x00,0x00, 0x00,0x00,0x00,0x00, // mov rax,0x7FFF
        0x48, 0xba, 0x99,0x99, 0x11,0x11, 0x11,0x11,0x11,0x11, // mov rdx,0x1111111111119999
        0x66, 0x99, // cwd
        0xc3,
    };
    const auto r = RunProgram(program, sizeof(program), mem);
    EXPECT_EQ(r.state.gpr[2], 0x1111111111110000ULL)
        << "AX bit15 clear -> DX=0x0000; prior DX=0x9999 cleared, upper preserved";
    EXPECT_EQ(r.state.gpr[0], 0x7FFFULL) << "CWD must not modify RAX";
}

// CQO positive: complements the existing negative-only case. RAX bit63 clear
// -> RDX = 0; RAX unchanged.
TEST_F(CpuRuntimeTest, Cqo_PositiveRax_ClearsRdx) {
    const u8 program[] = {
        0x48, 0xb8, 0xFF,0xFF, 0xFF,0xFF, 0xFF,0xFF,0xFF,0x7F, // mov rax,0x7FFFFFFFFFFFFFFF
        0x48, 0xba, 0xEF,0xBE, 0xAD,0xDE, 0xEF,0xBE,0xAD,0xDE, // mov rdx,0xDEADBEEFDEADBEEF
        0x48, 0x99, // cqo
        0xc3,
    };
    const auto r = RunProgram(program, sizeof(program), mem);
    EXPECT_EQ(r.state.gpr[2], 0x0ULL)
        << "RAX bit63 clear -> RDX all zeros";
    EXPECT_EQ(r.state.gpr[0], 0x7FFFFFFFFFFFFFFFULL) << "CQO must not modify RAX";
}


// ============================================================================
// VSQRTSS, VPUNPCKLBW, VPUNPCKHBW: implemented on the arm64 lifter but were
// previously missing from the x86 reference lifter (now added) and untested on
// both. Encodings (VEX.128, dst=xmm0, src1=xmm1, src2=xmm2):
//   vsqrtss    xmm0,xmm1,xmm2 = C5 F2 51 C2
//   vpunpcklbw xmm0,xmm1,xmm2 = C5 F1 60 C2
//   vpunpckhbw xmm0,xmm1,xmm2 = C5 F1 68 C2
// src1 occupies ymm chunks 4,5; src2 chunks 8,9; dst chunks 0,1. All values
// were cross-checked against native x86 execution of the emitted sequence.
// ============================================================================

// VSQRTSS: low 32 = sqrt(src2.low32); bits 127:32 from src1; VEX zeros 255:128.
TEST_F(CpuRuntimeTest, Vsqrtss_ScalarSingle_MergesSrc1AndZerosUpper) {
    const u8 program[] = {0xc5, 0xf2, 0x51, 0xc2, 0xc3}; // vsqrtss xmm0,xmm1,xmm2 ; ret
    std::memcpy(mem.CodePtr(), program, sizeof(program));
    u8* guest_rsp = mem.StackTop() - 8;
    *reinterpret_cast<u64*>(guest_rsp) = kReturnSentinel;
    GuestState st{};
    st.rip = reinterpret_cast<u64>(mem.CodePtr());
    st.gpr[4] = reinterpret_cast<u64>(guest_rsp);

    const float kSixteen = 16.0f;
    u32 sixteen_bits;
    std::memcpy(&sixteen_bits, &kSixteen, sizeof(u32));
    st.ymm[8] = static_cast<u64>(sixteen_bits);     // src2 (xmm2) low32 = 16.0f
    st.ymm[4] = 0x1111111122222222ULL;              // src1 (xmm1) low 64
    st.ymm[5] = 0x3333333344444444ULL;              // src1 bits 127:64
    st.ymm[0] = 0xDEAD; st.ymm[1] = 0xBEEF;         // dst poison
    st.ymm[2] = 0xCAFE; st.ymm[3] = 0xF00D;         // upper poison

    Runtime rt; rt.Run(st);

    float got;
    u32 lo = static_cast<u32>(st.ymm[0]);
    std::memcpy(&got, &lo, sizeof(float));
    EXPECT_EQ(got, 4.0f) << "low 32 = sqrt(16.0) = 4.0";
    EXPECT_EQ(static_cast<u32>(st.ymm[0] >> 32), 0x11111111U)
        << "bits 63:32 carried from src1";
    EXPECT_EQ(st.ymm[1], 0x3333333344444444ULL) << "bits 127:64 carried from src1";
    EXPECT_EQ(st.ymm[2], 0ULL) << "VEX zeros bits 255:128";
    EXPECT_EQ(st.ymm[3], 0ULL);
}

// VPUNPCKLBW: interleave the low 8 bytes of src1 and src2:
//   dst = s1[0] s2[0] s1[1] s2[1] ... s1[7] s2[7]
TEST_F(CpuRuntimeTest, Vpunpcklbw_InterleavesLowBytes) {
    const u8 program[] = {0xc5, 0xf1, 0x60, 0xc2, 0xc3}; // vpunpcklbw xmm0,xmm1,xmm2 ; ret
    std::memcpy(mem.CodePtr(), program, sizeof(program));
    u8* guest_rsp = mem.StackTop() - 8;
    *reinterpret_cast<u64*>(guest_rsp) = kReturnSentinel;
    GuestState st{};
    st.rip = reinterpret_cast<u64>(mem.CodePtr());
    st.gpr[4] = reinterpret_cast<u64>(guest_rsp);
    st.ymm[4] = 0x0706050403020100ULL; // src1 bytes 00..07
    st.ymm[8] = 0x1716151413121110ULL; // src2 bytes 10..17
    st.ymm[0] = 0xDEAD; st.ymm[2] = 0xCAFE; st.ymm[3] = 0xF00D;

    Runtime rt; rt.Run(st);

    // low 64: 00 10 01 11 02 12 03 13 (little-endian)
    EXPECT_EQ(st.ymm[0], 0x1303120211011000ULL) << "low 8 bytes interleaved";
    EXPECT_EQ(st.ymm[1], 0x1707160615051404ULL) << "next 8 bytes interleaved";
    EXPECT_EQ(st.ymm[2], 0ULL) << "VEX zeros bits 255:128";
    EXPECT_EQ(st.ymm[3], 0ULL);
}

// VPUNPCKHBW: interleave the HIGH 8 bytes of src1 and src2.
TEST_F(CpuRuntimeTest, Vpunpckhbw_InterleavesHighBytes) {
    const u8 program[] = {0xc5, 0xf1, 0x68, 0xc2, 0xc3}; // vpunpckhbw xmm0,xmm1,xmm2 ; ret
    std::memcpy(mem.CodePtr(), program, sizeof(program));
    u8* guest_rsp = mem.StackTop() - 8;
    *reinterpret_cast<u64*>(guest_rsp) = kReturnSentinel;
    GuestState st{};
    st.rip = reinterpret_cast<u64>(mem.CodePtr());
    st.gpr[4] = reinterpret_cast<u64>(guest_rsp);
    st.ymm[4] = 0x0706050403020100ULL; // src1 low 8 bytes (ignored by H form)
    st.ymm[5] = 0x0f0e0d0c0b0a0908ULL; // src1 high 8 bytes 08..0f
    st.ymm[8] = 0x1716151413121110ULL; // src2 low 8 (ignored)
    st.ymm[9] = 0x1f1e1d1c1b1a1918ULL; // src2 high 8 bytes 18..1f
    st.ymm[0] = 0xDEAD; st.ymm[2] = 0xCAFE; st.ymm[3] = 0xF00D;

    Runtime rt; rt.Run(st);

    // high bytes interleaved: 08 18 09 19 0a 1a 0b 1b | 0c 1c 0d 1d 0e 1e 0f 1f
    EXPECT_EQ(st.ymm[0], 0x1b0b1a0a19091808ULL) << "high 8 bytes interleaved (low half)";
    EXPECT_EQ(st.ymm[1], 0x1f0f1e0e1d0d1c0cULL) << "high 8 bytes interleaved (high half)";
    EXPECT_EQ(st.ymm[2], 0ULL) << "VEX zeros bits 255:128";
    EXPECT_EQ(st.ymm[3], 0ULL);
}


// ============================================================================
// x87 Tier 1+2: non-pop arithmetic (FADD/FMUL/FSUB/FSUBR/FDIV/FDIVR, register
// and memory forms), unary (FCHS/FABS/FSQRT), constants (FLD1/FLDZ), FXCH,
// comparisons (FCOMI/FUCOMI -> EFLAGS; FCOM/FUCOM -> x87 status-word C-bits),
// and FNSTSW. Semantics verified against native x86 execution of the emitted
// sequence and against the arm64 emitter under QEMU.
//
// Stack setup convention: two `fld qword[mem]` push A then B, so afterwards
// ST(0)=B and ST(1)=A. Helpers: fld m64 = DD /0, fst m64 = DD /2,
// fstp m64 = DD /3.
// ============================================================================

// FADD st(1), st(0): non-pop, ST(1) = ST(1) + ST(0) = A + B. Result read from
// ST(1) which we then store with fstp st1->mem isn't directly possible, so we
// pop ST0 first (fstp to scratch) then fstp ST1.  Simpler: use fadd st0,st1
// then store ST0.
TEST_F(CpuRuntimeTest, X87_Fadd_NonPop_RegForm) {
    u8* pa = mem.CodePtr() + 0x300;
    u8* pb = mem.CodePtr() + 0x310;
    u8* out = mem.CodePtr() + 0x320;
    const double A = 10.0, B = 20.0;
    std::memcpy(pa, &A, 8); std::memcpy(pb, &B, 8);
    u8 program[] = {
        0x48, 0xb8, 0,0,0,0,0,0,0,0,  // mov rax, &A
        0x48, 0xb9, 0,0,0,0,0,0,0,0,  // mov rcx, &B
        0x48, 0xba, 0,0,0,0,0,0,0,0,  // mov rdx, &out
        0xdd, 0x00,                   // fld [rax]  -> st0=A
        0xdd, 0x01,                   // fld [rcx]  -> st0=B, st1=A
        0xd8, 0xc1,                   // fadd st0, st1  -> st0 = B + A = 30 (no pop)
        0xdd, 0x1a,                   // fstp [rdx] -> store st0
        0xc3,
    };
    const u64 a=reinterpret_cast<u64>(pa),b=reinterpret_cast<u64>(pb),o=reinterpret_cast<u64>(out);
    std::memcpy(&program[2],&a,8); std::memcpy(&program[12],&b,8); std::memcpy(&program[22],&o,8);
    const auto r = RunProgram(program, sizeof(program), mem);
    double res; std::memcpy(&res, out, 8);
    EXPECT_EQ(res, 30.0) << "fadd st0,st1 (non-pop): 20+10=30";
    EXPECT_EQ(r.state.exit_reason, static_cast<u32>(ExitReason::BlockEnd));
}

// FSUB st(0), st(1): ST(0) = ST(0) - ST(1) = B - A.
TEST_F(CpuRuntimeTest, X87_Fsub_NonPop_RegForm) {
    u8* pa = mem.CodePtr() + 0x300;
    u8* pb = mem.CodePtr() + 0x310;
    u8* out = mem.CodePtr() + 0x320;
    const double A = 10.0, B = 30.0;
    std::memcpy(pa, &A, 8); std::memcpy(pb, &B, 8);
    u8 program[] = {
        0x48, 0xb8, 0,0,0,0,0,0,0,0,
        0x48, 0xb9, 0,0,0,0,0,0,0,0,
        0x48, 0xba, 0,0,0,0,0,0,0,0,
        0xdd, 0x00, 0xdd, 0x01,       // st0=B(30), st1=A(10)
        0xd8, 0xe1,                   // fsub st0, st1 -> 30 - 10 = 20
        0xdd, 0x1a, 0xc3,
    };
    const u64 a=reinterpret_cast<u64>(pa),b=reinterpret_cast<u64>(pb),o=reinterpret_cast<u64>(out);
    std::memcpy(&program[2],&a,8); std::memcpy(&program[12],&b,8); std::memcpy(&program[22],&o,8);
    const auto r = RunProgram(program, sizeof(program), mem);
    double res; std::memcpy(&res, out, 8);
    EXPECT_EQ(res, 20.0) << "fsub st0,st1: 30-10=20";
}

// FSUBR st(0), st(1): reversed -> ST(0) = ST(1) - ST(0) = A - B.
TEST_F(CpuRuntimeTest, X87_Fsubr_NonPop_RegForm) {
    u8* pa = mem.CodePtr() + 0x300;
    u8* pb = mem.CodePtr() + 0x310;
    u8* out = mem.CodePtr() + 0x320;
    const double A = 10.0, B = 30.0;
    std::memcpy(pa, &A, 8); std::memcpy(pb, &B, 8);
    u8 program[] = {
        0x48, 0xb8, 0,0,0,0,0,0,0,0,
        0x48, 0xb9, 0,0,0,0,0,0,0,0,
        0x48, 0xba, 0,0,0,0,0,0,0,0,
        0xdd, 0x00, 0xdd, 0x01,       // st0=B(30), st1=A(10)
        0xd8, 0xe9,                   // fsubr st0, st1 -> st1 - st0 = 10 - 30 = -20
        0xdd, 0x1a, 0xc3,
    };
    const u64 a=reinterpret_cast<u64>(pa),b=reinterpret_cast<u64>(pb),o=reinterpret_cast<u64>(out);
    std::memcpy(&program[2],&a,8); std::memcpy(&program[12],&b,8); std::memcpy(&program[22],&o,8);
    const auto r = RunProgram(program, sizeof(program), mem);
    double res; std::memcpy(&res, out, 8);
    EXPECT_EQ(res, -20.0) << "fsubr st0,st1: 10-30=-20";
}

// FADD m64 (memory form): ST(0) += [mem].
TEST_F(CpuRuntimeTest, X87_Fadd_MemForm) {
    u8* pa = mem.CodePtr() + 0x300;
    u8* pm = mem.CodePtr() + 0x310;
    u8* out = mem.CodePtr() + 0x320;
    const double A = 100.0, M = 23.0;
    std::memcpy(pa, &A, 8); std::memcpy(pm, &M, 8);
    u8 program[] = {
        0x48, 0xb8, 0,0,0,0,0,0,0,0,  // rax=&A
        0x48, 0xb9, 0,0,0,0,0,0,0,0,  // rcx=&M
        0x48, 0xba, 0,0,0,0,0,0,0,0,  // rdx=&out
        0xdd, 0x00,                   // fld [rax] -> st0=100
        0xdc, 0x01,                   // fadd qword [rcx] -> st0 = 100 + 23 = 123
        0xdd, 0x1a, 0xc3,
    };
    const u64 a=reinterpret_cast<u64>(pa),mm=reinterpret_cast<u64>(pm),o=reinterpret_cast<u64>(out);
    std::memcpy(&program[2],&a,8); std::memcpy(&program[12],&mm,8); std::memcpy(&program[22],&o,8);
    const auto r = RunProgram(program, sizeof(program), mem);
    double res; std::memcpy(&res, out, 8);
    EXPECT_EQ(res, 123.0) << "fadd m64: 100+23=123";
}

// FCHS: negate ST(0).
TEST_F(CpuRuntimeTest, X87_Fchs_Negates) {
    u8* pa = mem.CodePtr() + 0x300;
    u8* out = mem.CodePtr() + 0x320;
    const double A = 7.5;
    std::memcpy(pa, &A, 8);
    u8 program[] = {
        0x48, 0xb8, 0,0,0,0,0,0,0,0,
        0x48, 0xba, 0,0,0,0,0,0,0,0,
        0xdd, 0x00,                   // fld [rax] -> st0=7.5
        0xd9, 0xe0,                   // fchs -> -7.5
        0xdd, 0x1a, 0xc3,
    };
    const u64 a=reinterpret_cast<u64>(pa),o=reinterpret_cast<u64>(out);
    std::memcpy(&program[2],&a,8); std::memcpy(&program[12],&o,8);
    const auto r = RunProgram(program, sizeof(program), mem);
    double res; std::memcpy(&res, out, 8);
    EXPECT_EQ(res, -7.5) << "fchs negates 7.5";
}

// FABS: absolute value of ST(0).
TEST_F(CpuRuntimeTest, X87_Fabs_AbsoluteValue) {
    u8* pa = mem.CodePtr() + 0x300;
    u8* out = mem.CodePtr() + 0x320;
    const double A = -42.0;
    std::memcpy(pa, &A, 8);
    u8 program[] = {
        0x48, 0xb8, 0,0,0,0,0,0,0,0,
        0x48, 0xba, 0,0,0,0,0,0,0,0,
        0xdd, 0x00, 0xd9, 0xe1,       // fld; fabs
        0xdd, 0x1a, 0xc3,
    };
    const u64 a=reinterpret_cast<u64>(pa),o=reinterpret_cast<u64>(out);
    std::memcpy(&program[2],&a,8); std::memcpy(&program[12],&o,8);
    const auto r = RunProgram(program, sizeof(program), mem);
    double res; std::memcpy(&res, out, 8);
    EXPECT_EQ(res, 42.0) << "fabs(-42)=42";
}

// FSQRT: square root of ST(0).
TEST_F(CpuRuntimeTest, X87_Fsqrt_SquareRoot) {
    u8* pa = mem.CodePtr() + 0x300;
    u8* out = mem.CodePtr() + 0x320;
    const double A = 144.0;
    std::memcpy(pa, &A, 8);
    u8 program[] = {
        0x48, 0xb8, 0,0,0,0,0,0,0,0,
        0x48, 0xba, 0,0,0,0,0,0,0,0,
        0xdd, 0x00, 0xd9, 0xfa,       // fld; fsqrt
        0xdd, 0x1a, 0xc3,
    };
    const u64 a=reinterpret_cast<u64>(pa),o=reinterpret_cast<u64>(out);
    std::memcpy(&program[2],&a,8); std::memcpy(&program[12],&o,8);
    const auto r = RunProgram(program, sizeof(program), mem);
    double res; std::memcpy(&res, out, 8);
    EXPECT_EQ(res, 12.0) << "fsqrt(144)=12";
}

// FLD1: push 1.0.
TEST_F(CpuRuntimeTest, X87_Fld1_PushesOne) {
    u8* out = mem.CodePtr() + 0x320;
    u8 program[] = {
        0x48, 0xba, 0,0,0,0,0,0,0,0,
        0xd9, 0xe8,                   // fld1 -> st0=1.0
        0xdd, 0x1a, 0xc3,
    };
    const u64 o=reinterpret_cast<u64>(out);
    std::memcpy(&program[2],&o,8);
    const auto r = RunProgram(program, sizeof(program), mem);
    double res; std::memcpy(&res, out, 8);
    EXPECT_EQ(res, 1.0) << "fld1 pushes 1.0";
}

// FLDZ: push 0.0.
TEST_F(CpuRuntimeTest, X87_Fldz_PushesZero) {
    u8* out = mem.CodePtr() + 0x320;
    u8 program[] = {
        0x48, 0xba, 0,0,0,0,0,0,0,0,
        0xd9, 0xee,                   // fldz
        0xdd, 0x1a, 0xc3,
    };
    const u64 o=reinterpret_cast<u64>(out);
    std::memcpy(&program[2],&o,8);
    const auto r = RunProgram(program, sizeof(program), mem);
    double res; std::memcpy(&res, out, 8);
    EXPECT_EQ(res, 0.0) << "fldz pushes 0.0";
}

// FXCH: swap ST(0) and ST(1), then store both to confirm the swap.
TEST_F(CpuRuntimeTest, X87_Fxch_SwapsTop) {
    u8* pa = mem.CodePtr() + 0x300;
    u8* pb = mem.CodePtr() + 0x310;
    u8* out0 = mem.CodePtr() + 0x320;
    u8* out1 = mem.CodePtr() + 0x330;
    const double A = 11.0, B = 22.0;
    std::memcpy(pa, &A, 8); std::memcpy(pb, &B, 8);
    u8 program[] = {
        0x48, 0xb8, 0,0,0,0,0,0,0,0,  // rax=&A
        0x48, 0xb9, 0,0,0,0,0,0,0,0,  // rcx=&B
        0x48, 0xba, 0,0,0,0,0,0,0,0,  // rdx=&out0
        0x49, 0xb8, 0,0,0,0,0,0,0,0,  // r8=&out1
        0xdd, 0x00, 0xdd, 0x01,       // st0=B(22), st1=A(11)
        0xd9, 0xc9,                   // fxch st1 -> st0=A(11), st1=B(22)
        0xdd, 0x1a,                   // fstp [rdx] -> out0 = st0 = 11, pop
        0x41, 0xdd, 0x18,             // fstp [r8]  -> out1 = new st0 = 22
        0xc3,
    };
    const u64 a=reinterpret_cast<u64>(pa),b=reinterpret_cast<u64>(pb),
              o0=reinterpret_cast<u64>(out0),o1=reinterpret_cast<u64>(out1);
    std::memcpy(&program[2],&a,8); std::memcpy(&program[12],&b,8);
    std::memcpy(&program[22],&o0,8); std::memcpy(&program[32],&o1,8);
    const auto r = RunProgram(program, sizeof(program), mem);
    double r0, r1; std::memcpy(&r0, out0, 8); std::memcpy(&r1, out1, 8);
    EXPECT_EQ(r0, 11.0) << "after fxch, st0 = original A";
    EXPECT_EQ(r1, 22.0) << "after fxch+pop, new st0 = original B";
}

// FCOMI: ST(0) > ST(1) -> ZF=CF=0. We branch on it via setb/sete to capture
// flags into a GPR, but simpler: read rflags after the compare. Use a JA
// (ja = above = CF=0 and ZF=0) to set a register.
TEST_F(CpuRuntimeTest, X87_Fcomi_GreaterClearsZfCf) {
    u8* pa = mem.CodePtr() + 0x300;
    u8* pb = mem.CodePtr() + 0x310;
    const double A = 3.0, B = 5.0;   // ST1=A=3, ST0=B=5 -> ST0 > ST1
    std::memcpy(pa, &A, 8); std::memcpy(pb, &B, 8);
    u8 program[] = {
        0x48, 0xb8, 0,0,0,0,0,0,0,0,
        0x48, 0xb9, 0,0,0,0,0,0,0,0,
        0xdd, 0x00, 0xdd, 0x01,       // st0=B(5), st1=A(3)
        0xdb, 0xf1,                   // fcomi st0, st1 -> ST0>ST1: ZF=0 CF=0
        0xc3,
    };
    const u64 a=reinterpret_cast<u64>(pa),b=reinterpret_cast<u64>(pb);
    std::memcpy(&program[2],&a,8); std::memcpy(&program[12],&b,8);
    const auto r = RunProgram(program, sizeof(program), mem);
    EXPECT_FALSE(r.state.rflags & (1ULL << 6)) << "ZF clear (5 > 3, not equal)";
    EXPECT_FALSE(r.state.rflags & (1ULL << 0)) << "CF clear (5 > 3, not below)";
}

// FCOMI: ST(0) < ST(1) -> CF=1.
TEST_F(CpuRuntimeTest, X87_Fcomi_LessSetsCf) {
    u8* pa = mem.CodePtr() + 0x300;
    u8* pb = mem.CodePtr() + 0x310;
    const double A = 9.0, B = 2.0;   // ST1=A=9, ST0=B=2 -> ST0 < ST1
    std::memcpy(pa, &A, 8); std::memcpy(pb, &B, 8);
    u8 program[] = {
        0x48, 0xb8, 0,0,0,0,0,0,0,0,
        0x48, 0xb9, 0,0,0,0,0,0,0,0,
        0xdd, 0x00, 0xdd, 0x01,
        0xdb, 0xf1, 0xc3,
    };
    const u64 a=reinterpret_cast<u64>(pa),b=reinterpret_cast<u64>(pb);
    std::memcpy(&program[2],&a,8); std::memcpy(&program[12],&b,8);
    const auto r = RunProgram(program, sizeof(program), mem);
    EXPECT_TRUE(r.state.rflags & (1ULL << 0)) << "CF set (2 < 9)";
    EXPECT_FALSE(r.state.rflags & (1ULL << 6)) << "ZF clear";
}

// FCOMI: equal -> ZF=1.
TEST_F(CpuRuntimeTest, X87_Fcomi_EqualSetsZf) {
    u8* pa = mem.CodePtr() + 0x300;
    u8* pb = mem.CodePtr() + 0x310;
    const double A = 4.0, B = 4.0;
    std::memcpy(pa, &A, 8); std::memcpy(pb, &B, 8);
    u8 program[] = {
        0x48, 0xb8, 0,0,0,0,0,0,0,0,
        0x48, 0xb9, 0,0,0,0,0,0,0,0,
        0xdd, 0x00, 0xdd, 0x01,
        0xdb, 0xf1, 0xc3,
    };
    const u64 a=reinterpret_cast<u64>(pa),b=reinterpret_cast<u64>(pb);
    std::memcpy(&program[2],&a,8); std::memcpy(&program[12],&b,8);
    const auto r = RunProgram(program, sizeof(program), mem);
    EXPECT_TRUE(r.state.rflags & (1ULL << 6)) << "ZF set (4 == 4)";
    EXPECT_FALSE(r.state.rflags & (1ULL << 0)) << "CF clear";
}

// FNSTSW AX: status word reflects TOP and a prior compare's C3 (equal).
// fld a; fld a; fcom (equal -> C3=1); fnstsw ax. TOP after two pushes is 6.
TEST_F(CpuRuntimeTest, X87_Fnstsw_ReflectsCompareAndTop) {
    u8* pa = mem.CodePtr() + 0x300;
    const double A = 1.0;
    std::memcpy(pa, &A, 8);
    u8 program[] = {
        0x48, 0xb8, 0,0,0,0,0,0,0,0,  // rax=&A
        0xdd, 0x00, 0xdd, 0x00,       // fld[rax]x2 -> st0=st1=1.0, TOP=6
        0xd8, 0xd1,                   // fcom st1 -> equal: C3=1
        0xdf, 0xe0,                   // fnstsw ax
        0xc3,
    };
    const u64 a=reinterpret_cast<u64>(pa);
    std::memcpy(&program[2],&a,8);
    const auto r = RunProgram(program, sizeof(program), mem);
    const u16 ax = static_cast<u16>(r.state.gpr[0] & 0xFFFF);
    EXPECT_TRUE(ax & (1u << 14)) << "C3 set in status word (operands equal)";
    EXPECT_EQ((ax >> 11) & 0x7u, 6u) << "TOP field = 6 after two pushes";
}


// ============================================================================
// IDIV (signed division). The lifter previously handled only unsigned DIV;
// IDIV was unimplemented. x86 IDIV truncates toward zero and the remainder
// takes the sign of the dividend. Quotient -> AL/AX/EAX/RAX, remainder ->
// AH/DX/EDX/RDX. Dividend is sign-extended into the high half by CDQ/CQO (or
// CBW for 8-bit) before the divide. Results cross-checked against native x86
// IDIV and the arm64 emitter under QEMU.
// Encodings: idiv r/m32 = F7 /7, idiv r/m64 = REX.W F7 /7, idiv r/m8 = F6 /7.
// ============================================================================

// 64-bit positive / positive.
TEST_F(CpuRuntimeTest, Idiv64_PositivePositive) {
    const u8 program[] = {
        0x48, 0xc7, 0xc0, 0x64, 0x00, 0x00, 0x00, // mov rax, 100
        0x48, 0x99,                               // cqo  (sign-extend into rdx)
        0x48, 0xc7, 0xc1, 0x07, 0x00, 0x00, 0x00, // mov rcx, 7
        0x48, 0xf7, 0xf9,                         // idiv rcx
        0xc3,
    };
    const auto r = RunProgram(program, sizeof(program), mem);
    EXPECT_EQ(static_cast<s64>(r.state.gpr[0]), 14LL) << "100 / 7 = 14";
    EXPECT_EQ(static_cast<s64>(r.state.gpr[2]), 2LL)  << "100 % 7 = 2";
}

// 64-bit negative dividend: remainder takes the dividend's sign.
TEST_F(CpuRuntimeTest, Idiv64_NegativeDividend) {
    const u8 program[] = {
        0x48, 0xc7, 0xc0, 0x9c, 0xff, 0xff, 0xff, // mov rax, -100 (sign-ext imm32)
        0x48, 0x99,                               // cqo -> rdx = -1
        0x48, 0xc7, 0xc1, 0x07, 0x00, 0x00, 0x00, // mov rcx, 7
        0x48, 0xf7, 0xf9,                         // idiv rcx
        0xc3,
    };
    const auto r = RunProgram(program, sizeof(program), mem);
    EXPECT_EQ(static_cast<s64>(r.state.gpr[0]), -14LL) << "-100 / 7 = -14";
    EXPECT_EQ(static_cast<s64>(r.state.gpr[2]), -2LL)  << "-100 % 7 = -2 (sign of dividend)";
}

// 64-bit negative divisor.
TEST_F(CpuRuntimeTest, Idiv64_NegativeDivisor) {
    const u8 program[] = {
        0x48, 0xc7, 0xc0, 0x64, 0x00, 0x00, 0x00, // mov rax, 100
        0x48, 0x99,                               // cqo
        0x48, 0xc7, 0xc1, 0xf9, 0xff, 0xff, 0xff, // mov rcx, -7
        0x48, 0xf7, 0xf9,                         // idiv rcx
        0xc3,
    };
    const auto r = RunProgram(program, sizeof(program), mem);
    EXPECT_EQ(static_cast<s64>(r.state.gpr[0]), -14LL) << "100 / -7 = -14";
    EXPECT_EQ(static_cast<s64>(r.state.gpr[2]), 2LL)   << "100 % -7 = 2 (sign of dividend)";
}

// 64-bit both negative -> positive quotient, negative remainder.
TEST_F(CpuRuntimeTest, Idiv64_BothNegative) {
    const u8 program[] = {
        0x48, 0xc7, 0xc0, 0x9c, 0xff, 0xff, 0xff, // mov rax, -100
        0x48, 0x99,                               // cqo
        0x48, 0xc7, 0xc1, 0xf9, 0xff, 0xff, 0xff, // mov rcx, -7
        0x48, 0xf7, 0xf9,                         // idiv rcx
        0xc3,
    };
    const auto r = RunProgram(program, sizeof(program), mem);
    EXPECT_EQ(static_cast<s64>(r.state.gpr[0]), 14LL)  << "-100 / -7 = 14";
    EXPECT_EQ(static_cast<s64>(r.state.gpr[2]), -2LL)  << "-100 % -7 = -2";
}

// 64-bit truncation toward zero (not floor): -7 / 2 = -3 rem -1.
TEST_F(CpuRuntimeTest, Idiv64_TruncatesTowardZero) {
    const u8 program[] = {
        0x48, 0xc7, 0xc0, 0xf9, 0xff, 0xff, 0xff, // mov rax, -7
        0x48, 0x99,                               // cqo
        0x48, 0xc7, 0xc1, 0x02, 0x00, 0x00, 0x00, // mov rcx, 2
        0x48, 0xf7, 0xf9,                         // idiv rcx
        0xc3,
    };
    const auto r = RunProgram(program, sizeof(program), mem);
    EXPECT_EQ(static_cast<s64>(r.state.gpr[0]), -3LL) << "-7 / 2 = -3 (toward zero, not -4)";
    EXPECT_EQ(static_cast<s64>(r.state.gpr[2]), -1LL) << "-7 % 2 = -1";
}

// 32-bit signed: -100 / 7. Quotient in EAX (zero-extends RAX), rem in EDX.
TEST_F(CpuRuntimeTest, Idiv32_NegativeDividend) {
    const u8 program[] = {
        0xb8, 0x9c, 0xff, 0xff, 0xff,             // mov eax, -100
        0x99,                                     // cdq -> edx = -1
        0xb9, 0x07, 0x00, 0x00, 0x00,             // mov ecx, 7
        0xf7, 0xf9,                               // idiv ecx
        0xc3,
    };
    const auto r = RunProgram(program, sizeof(program), mem);
    EXPECT_EQ(static_cast<s32>(r.state.gpr[0] & 0xFFFFFFFF), -14) << "-100 / 7 = -14";
    EXPECT_EQ(static_cast<s32>(r.state.gpr[2] & 0xFFFFFFFF), -2)  << "-100 % 7 = -2";
    EXPECT_EQ(r.state.gpr[0] >> 32, 0u) << "EAX result zero-extends RAX";
}

// 32-bit positive sanity.
TEST_F(CpuRuntimeTest, Idiv32_PositivePositive) {
    const u8 program[] = {
        0xb8, 0xc8, 0x00, 0x00, 0x00,             // mov eax, 200
        0x99,                                     // cdq
        0xb9, 0x07, 0x00, 0x00, 0x00,             // mov ecx, 7
        0xf7, 0xf9,                               // idiv ecx
        0xc3,
    };
    const auto r = RunProgram(program, sizeof(program), mem);
    EXPECT_EQ(static_cast<s32>(r.state.gpr[0] & 0xFFFFFFFF), 28) << "200 / 7 = 28";
    EXPECT_EQ(static_cast<s32>(r.state.gpr[2] & 0xFFFFFFFF), 4)  << "200 % 7 = 4";
}

// 8-bit signed: AX = -100, divisor 7 -> AL = -14, AH = -2. CBW sign-extends
// AL into AX first. Encoding: cbw = 66 98, idiv cl = F6 F9.
TEST_F(CpuRuntimeTest, Idiv8_NegativeDividend) {
    const u8 program[] = {
        0xb0, 0x9c,                               // mov al, -100 (0x9C)
        0x66, 0x98,                               // cbw -> AX = -100
        0xb1, 0x07,                               // mov cl, 7
        0xf6, 0xf9,                               // idiv cl
        0xc3,
    };
    const auto r = RunProgram(program, sizeof(program), mem);
    const s8 al = static_cast<s8>(r.state.gpr[0] & 0xFF);
    const s8 ah = static_cast<s8>((r.state.gpr[0] >> 8) & 0xFF);
    EXPECT_EQ(al, -14) << "AL quotient: -100 / 7 = -14";
    EXPECT_EQ(ah, -2)  << "AH remainder: -100 % 7 = -2";
}


// ============================================================================
// Memory-destination shifts (SHL/SHR/SAR [mem], imm|CL). Both lifters
// previously rejected a memory operand 0 (register-only). These now load the
// width-sized value from the effective address, shift, and store back in
// place, leaving adjacent bytes untouched. Verified against native x86
// execution and the arm64 emitter under QEMU.
// Encodings: shl dword[rbx],imm = C1 /4 ib ; shl dword[rbx],cl = D3 /4 ;
//   shr byte[rbx],1 = D0 /5 ; sar qword[rbx],imm = REX.W C1 /7 ib.
// The target lives in the guest data area (CodePtr()+0x200); RBX points at it.
// ============================================================================

// SHL dword [rbx], 3 — in place; adjacent bytes preserved.
TEST_F(CpuRuntimeTest, ShlMemDest_Dword_Imm) {
    u8* target = mem.CodePtr() + 0x200;
    for (int i = -4; i < 12; ++i) target[i] = 0xAA;  // sentinel surround
    const u32 init = 0x11111111u;
    std::memcpy(target, &init, 4);

    u8 program[] = {
        0x48, 0xbb, 0,0,0,0,0,0,0,0,   // mov rbx, &target
        0xc1, 0x23, 0x03,              // shl dword [rbx], 3
        0xc3,
    };
    const u64 t = reinterpret_cast<u64>(target);
    std::memcpy(&program[2], &t, 8);
    const auto r = RunProgram(program, sizeof(program), mem);

    u32 got; std::memcpy(&got, target, 4);
    EXPECT_EQ(got, 0x88888888u) << "0x11111111 << 3";
    EXPECT_EQ(static_cast<u8>(target[-1]), 0xAAu) << "byte below untouched";
    EXPECT_EQ(static_cast<u8>(target[4]), 0xAAu)  << "byte above untouched";
}

// SHR byte [rbx], 1 — narrow, adjacent bytes preserved.
TEST_F(CpuRuntimeTest, ShrMemDest_Byte_Imm) {
    u8* target = mem.CodePtr() + 0x200;
    for (int i = -4; i < 12; ++i) target[i] = 0xCC;
    target[0] = 0xF0;

    u8 program[] = {
        0x48, 0xbb, 0,0,0,0,0,0,0,0,
        0xd0, 0x2b,                    // shr byte [rbx], 1
        0xc3,
    };
    const u64 t = reinterpret_cast<u64>(target);
    std::memcpy(&program[2], &t, 8);
    const auto r = RunProgram(program, sizeof(program), mem);

    EXPECT_EQ(static_cast<u8>(target[0]), 0x78u) << "0xF0 >> 1 = 0x78";
    EXPECT_EQ(static_cast<u8>(target[1]), 0xCCu) << "next byte untouched";
    EXPECT_EQ(static_cast<u8>(target[-1]), 0xCCu) << "prev byte untouched";
}

// SAR qword [rbx], 2 — signed shift, full width.
TEST_F(CpuRuntimeTest, SarMemDest_Qword_Imm) {
    u8* target = mem.CodePtr() + 0x200;
    const s64 init = -64;
    std::memcpy(target, &init, 8);

    u8 program[] = {
        0x48, 0xbb, 0,0,0,0,0,0,0,0,
        0x48, 0xc1, 0x3b, 0x02,        // sar qword [rbx], 2
        0xc3,
    };
    const u64 t = reinterpret_cast<u64>(target);
    std::memcpy(&program[2], &t, 8);
    const auto r = RunProgram(program, sizeof(program), mem);

    s64 got; std::memcpy(&got, target, 8);
    EXPECT_EQ(got, -16LL) << "-64 >>arith 2 = -16";
}

// SHL dword [rbx], CL — count from CL register.
TEST_F(CpuRuntimeTest, ShlMemDest_Dword_CL) {
    u8* target = mem.CodePtr() + 0x200;
    const u32 init = 1u;
    std::memcpy(target, &init, 4);

    u8 program[] = {
        0x48, 0xbb, 0,0,0,0,0,0,0,0,   // mov rbx, &target
        0x48, 0xc7, 0xc1, 0x05, 0x00, 0x00, 0x00, // mov rcx, 5
        0xd3, 0x23,                    // shl dword [rbx], cl
        0xc3,
    };
    const u64 t = reinterpret_cast<u64>(target);
    std::memcpy(&program[2], &t, 8);
    const auto r = RunProgram(program, sizeof(program), mem);

    u32 got; std::memcpy(&got, target, 4);
    EXPECT_EQ(got, 32u) << "1 << 5 = 32";
}

// SHL dword [rbx], 1 sets CF/ZF appropriately: shift 0x80000000 << 1 -> 0,
// CF = 1 (bit shifted out), ZF = 1 (result zero).
TEST_F(CpuRuntimeTest, ShlMemDest_Flags) {
    u8* target = mem.CodePtr() + 0x200;
    const u32 init = 0x80000000u;
    std::memcpy(target, &init, 4);

    u8 program[] = {
        0x48, 0xbb, 0,0,0,0,0,0,0,0,
        0xd1, 0x23,                    // shl dword [rbx], 1
        0xc3,
    };
    const u64 t = reinterpret_cast<u64>(target);
    std::memcpy(&program[2], &t, 8);
    const auto r = RunProgram(program, sizeof(program), mem);

    u32 got; std::memcpy(&got, target, 4);
    EXPECT_EQ(got, 0u) << "0x80000000 << 1 = 0 (32-bit)";
    EXPECT_TRUE(r.state.rflags & (1ULL << 0)) << "CF set (bit shifted out)";
    EXPECT_TRUE(r.state.rflags & (1ULL << 6)) << "ZF set (result zero)";
}


// ============================================================================
// Memory-destination rotates (ROL/ROR [mem], imm|CL) and 8/16-bit register
// rotates. Previously both lifters rejected a memory operand 0 for rotates,
// and the x86 reference lifter additionally rejected 8/16-bit rotates entirely
// (a divergence: arm64 handled them). These now load the width-sized value,
// rotate, and store back in place (adjacent bytes untouched). Rotates affect
// only CF and OF (and only when the masked count != 0). Verified against
// native x86 execution and the arm64 emitter under QEMU.
// Encodings: rol dword[rbx],imm = C1 /0 ib ; ror byte[rbx],1 = D0 /1 ;
//   rol qword[rbx],imm = REX.W C1 /0 ib ; ror dword[rbx],cl = D3 /1 ;
//   rol ax,imm = 66 C1 /0 ib ; ror bl,1 = D0 /1 (reg).
// ============================================================================

// ROL dword [rbx], 4 — in place; adjacent bytes preserved.
TEST_F(CpuRuntimeTest, RolMemDest_Dword_Imm) {
    u8* target = mem.CodePtr() + 0x200;
    for (int i = -4; i < 12; ++i) target[i] = 0xAA;
    const u32 init = 0x12345678u;
    std::memcpy(target, &init, 4);

    u8 program[] = {
        0x48, 0xbb, 0,0,0,0,0,0,0,0,   // mov rbx, &target
        0xc1, 0x03, 0x04,              // rol dword [rbx], 4
        0xc3,
    };
    const u64 t = reinterpret_cast<u64>(target);
    std::memcpy(&program[2], &t, 8);
    const auto r = RunProgram(program, sizeof(program), mem);

    u32 got; std::memcpy(&got, target, 4);
    EXPECT_EQ(got, 0x23456781u) << "rol 0x12345678 by 4";
    EXPECT_EQ(static_cast<u8>(target[-1]), 0xAAu) << "byte below untouched";
    EXPECT_EQ(static_cast<u8>(target[4]), 0xAAu)  << "byte above untouched";
}

// ROR byte [rbx], 1 — narrow mem; 0x01 ror 1 = 0x80.
TEST_F(CpuRuntimeTest, RorMemDest_Byte_Imm) {
    u8* target = mem.CodePtr() + 0x200;
    for (int i = -4; i < 12; ++i) target[i] = 0xCC;
    target[0] = 0x01;

    u8 program[] = {
        0x48, 0xbb, 0,0,0,0,0,0,0,0,
        0xd0, 0x0b,                    // ror byte [rbx], 1
        0xc3,
    };
    const u64 t = reinterpret_cast<u64>(target);
    std::memcpy(&program[2], &t, 8);
    const auto r = RunProgram(program, sizeof(program), mem);

    EXPECT_EQ(static_cast<u8>(target[0]), 0x80u) << "ror 0x01 by 1 = 0x80";
    EXPECT_EQ(static_cast<u8>(target[1]), 0xCCu) << "next byte untouched";
    EXPECT_EQ(static_cast<u8>(target[-1]), 0xCCu) << "prev byte untouched";
}

// ROL qword [rbx], 8 — full width mem.
TEST_F(CpuRuntimeTest, RolMemDest_Qword_Imm) {
    u8* target = mem.CodePtr() + 0x200;
    const u64 init = 0x0123456789ABCDEFull;
    std::memcpy(target, &init, 8);

    u8 program[] = {
        0x48, 0xbb, 0,0,0,0,0,0,0,0,
        0x48, 0xc1, 0x03, 0x08,        // rol qword [rbx], 8
        0xc3,
    };
    const u64 t = reinterpret_cast<u64>(target);
    std::memcpy(&program[2], &t, 8);
    const auto r = RunProgram(program, sizeof(program), mem);

    u64 got; std::memcpy(&got, target, 8);
    EXPECT_EQ(got, 0x23456789ABCDEF01ull) << "rol 0x0123456789ABCDEF by 8";
}

// ROR dword [rbx], CL — count from CL.
TEST_F(CpuRuntimeTest, RorMemDest_Dword_CL) {
    u8* target = mem.CodePtr() + 0x200;
    const u32 init = 0x00000001u;
    std::memcpy(target, &init, 4);

    u8 program[] = {
        0x48, 0xbb, 0,0,0,0,0,0,0,0,
        0x48, 0xc7, 0xc1, 0x04, 0x00, 0x00, 0x00, // mov rcx, 4
        0xd3, 0x0b,                    // ror dword [rbx], cl
        0xc3,
    };
    const u64 t = reinterpret_cast<u64>(target);
    std::memcpy(&program[2], &t, 8);
    const auto r = RunProgram(program, sizeof(program), mem);

    u32 got; std::memcpy(&got, target, 4);
    EXPECT_EQ(got, 0x10000000u) << "ror 1 by 4 (32-bit) = 0x10000000";
}

// 16-bit register rotate: ROL AX, 4 — exercises the formerly-x86-rejected
// narrow rotate (arm64 handled it; x86 did not). Upper bits of RAX preserved.
TEST_F(CpuRuntimeTest, RolReg_Word_ClosesNarrowGap) {
    u8 program[] = {
        0x48, 0xb8, 0x34, 0x12, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, // mov rax, 0x1234
        0x66, 0xc1, 0xc0, 0x04,        // rol ax, 4
        0xc3,
    };
    const auto r = RunProgram(program, sizeof(program), mem);
    EXPECT_EQ(r.state.gpr[0] & 0xFFFF, 0x2341u) << "rol 0x1234 by 4 (16-bit)";
}

// 8-bit register rotate: ROR BL, 1 — narrow gap, preserve upper bits.
TEST_F(CpuRuntimeTest, RorReg_Byte_ClosesNarrowGap) {
    u8 program[] = {
        0x48, 0xc7, 0xc3, 0x01, 0x00, 0x00, 0x00, // mov rbx, 1
        0xd0, 0xcb,                    // ror bl, 1
        0xc3,
    };
    const auto r = RunProgram(program, sizeof(program), mem);
    EXPECT_EQ(r.state.gpr[3] & 0xFF, 0x80u) << "ror 0x01 by 1 (8-bit) = 0x80";
}

// ROL dword [rbx], 1 sets CF = LSB of result. After rotating 0x80000000 left
// by 1 -> 0x00000001, CF = 1 (the bit rotated out of the top into bit 0).
TEST_F(CpuRuntimeTest, RolMemDest_Flags_CF) {
    u8* target = mem.CodePtr() + 0x200;
    const u32 init = 0x80000000u;
    std::memcpy(target, &init, 4);

    u8 program[] = {
        0x48, 0xbb, 0,0,0,0,0,0,0,0,
        0xd1, 0x03,                    // rol dword [rbx], 1
        0xc3,
    };
    const u64 t = reinterpret_cast<u64>(target);
    std::memcpy(&program[2], &t, 8);
    const auto r = RunProgram(program, sizeof(program), mem);

    u32 got; std::memcpy(&got, target, 4);
    EXPECT_EQ(got, 0x00000001u) << "rol 0x80000000 by 1 = 1";
    EXPECT_TRUE(r.state.rflags & (1ULL << 0)) << "CF = bit rotated into LSB";
}

} // namespace
} // namespace Core::Runtime
