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
#include <cstdint>
#include <cstring>
#include <string>
#include <vector>

#include <gtest/gtest.h>

#ifdef _WIN32
#include <windows.h>
#else
#include <sys/mman.h>
#endif

#include "core/cpu_runtime/block_cache.h"
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
#ifdef _WIN32
        void* p =
            ::VirtualAlloc(nullptr, TOTAL_SIZE, MEM_COMMIT | MEM_RESERVE, PAGE_EXECUTE_READWRITE);
        base_ = static_cast<u8*>(p);
#else
        void* p = ::mmap(nullptr, TOTAL_SIZE, PROT_READ | PROT_WRITE | PROT_EXEC,
                         MAP_PRIVATE | MAP_ANON, -1, 0);
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

// MOV rax, 1; XOR rax, rax (unsupported); RET (unreached)
//
// Expected: exit_reason=UnsupportedInstruction, rip points to the XOR
// instruction, rax=1 (MOV before XOR executed).
//
// This exercises the lifter's failure path: when a guest instruction
// is encountered that the initial subset doesn't handle, the lifter
// emits an exit sequence that sets exit_reason and stores the
// un-lifted RIP so callers can diagnose where execution stopped.
TEST_F(CpuRuntimeTest, UnsupportedOpcode_ExitsCleanly) {
    // BSR is still unsupported in the current lifter slice. Use it
    // to verify the unsupported-exit path still works. If/when BSR
    // becomes supported, replace with another not-yet-supported
    // instruction (BSF / POPCNT / SHLD / DIV / IDIV are all good
    // candidates).
    //
    // Encoding: 48 0f bd d8 = BSR rbx, rax (4 bytes).
    const u8 program[] = {
        0x48, 0xc7, 0xc0, 0x01, 0x00, 0x00, 0x00, // mov rax, 1
        0x48, 0x0f, 0xbd, 0xd8,                   // bsr rbx, rax   (unsupported)
        0xc3,                                     // ret  (unreached)
    };
    const auto r = RunProgram(program, sizeof(program), mem);

    // RIP should point to the BSR instruction (at offset 7).
    EXPECT_EQ(r.state.rip, r.program_base + 7);
    EXPECT_EQ(r.state.exit_reason, static_cast<u32>(ExitReason::UnsupportedInstruction));
    // MOV before BSR should have executed.
    EXPECT_EQ(r.state.gpr[0], 1u);
}

// Stress test: many runs through the same Runtime instance to verify
// the block cache reuses compiled blocks and doesn't leak host code.
TEST_F(CpuRuntimeTest, MultipleRuns_ShareBlockCache) {
    const u8 program[] = {
        0x48, 0xc7, 0xc0, 0x42, 0x00, 0x00, 0x00, // mov rax, 0x42
        0xc3,                                     // ret
    };
    std::memcpy(mem.CodePtr(), program, sizeof(program));

    u8* guest_rsp = mem.StackTop() - 8;
    *reinterpret_cast<u64*>(guest_rsp) = kReturnSentinel;

    Runtime rt;

    // Run the same program ten times. The block cache should compile
    // it once and reuse the host code on subsequent calls.
    for (int i = 0; i < 10; ++i) {
        GuestState state{};
        state.rip = reinterpret_cast<u64>(mem.CodePtr());
        state.gpr[4] = reinterpret_cast<u64>(guest_rsp);
        // Reset the stack each iteration since RET advanced rsp.
        *reinterpret_cast<u64*>(guest_rsp) = kReturnSentinel;

        rt.Run(state);

        EXPECT_EQ(state.gpr[0], 0x42u) << "iteration " << i;
        EXPECT_EQ(state.rip, kReturnSentinel) << "iteration " << i;
    }

    // Block cache should contain exactly one block (the one compiled
    // on the first iteration). The other nine runs hit the cache.
    EXPECT_EQ(rt.GetBlockCache().Size(), 1u)
        << "Block cache should have one entry after 10 runs of one block";
}

// CallGuestSimple test: verify the helper invokes a guest function
// with the right argument registers and returns the right value.
//
// Guest function: takes RDI as input, adds 1 to it, returns in RAX.
//   mov rax, rdi     48 89 f8
//   add rax, 1       48 83 c0 01
//   ret              c3
//
// Calling with a0=0x100 should return 0x101.
TEST_F(CpuRuntimeTest, CallGuestSimple_PassesArgumentsAndReturnsValue) {
    const u8 program[] = {
        0x48, 0x89, 0xf8,       // mov rax, rdi
        0x48, 0x83, 0xc0, 0x01, // add rax, 1
        0xc3,                   // ret
    };
    std::memcpy(mem.CodePtr(), program, sizeof(program));

    // Use a local Runtime instance for test isolation. The singleton
    // would share its block cache with other tests, and since mmap
    // may return overlapping addresses across tests, the cache could
    // serve stale compiled code from a previous test's program.
    Runtime rt;
    u64 result = rt.CallGuestSimple(reinterpret_cast<VAddr>(mem.CodePtr()), mem.StackTop(),
                                    /*a0=*/0x100);

    EXPECT_EQ(result, 0x101u);
}

// CallGuest with custom setup: verify the setup callback is called
// and that we can read non-RAX return values from the state.
//
// Guest function uses RDI+RSI, returns sum in RAX, also leaves
// RSI unchanged so we can verify the setup callback put it there.
//
//   mov rax, rdi     48 89 f8
//   add rax, rsi     48 01 f0
//   ret              c3
TEST_F(CpuRuntimeTest, CallGuest_SetupCallbackPopulatesRegisters) {
    const u8 program[] = {
        0x48, 0x89, 0xf8, // mov rax, rdi
        0x48, 0x01, 0xf0, // add rax, rsi
        0xc3,             // ret
    };
    std::memcpy(mem.CodePtr(), program, sizeof(program));

    struct Args {
        u64 x, y;
    };
    Args args{0x200, 0x40};

    auto setup = [](GuestState& state, void* user_data) {
        const auto* a = static_cast<const Args*>(user_data);
        state.gpr[7] = a->x; // RDI
        state.gpr[6] = a->y; // RSI
    };

    // Local Runtime — see note in CallGuestSimple test above.
    Runtime rt;
    GuestState result =
        rt.CallGuest(reinterpret_cast<VAddr>(mem.CodePtr()), mem.StackTop(), setup, &args);

    EXPECT_EQ(result.gpr[0], 0x240u); // RAX = x + y
    EXPECT_EQ(result.gpr[6], args.y); // RSI unchanged
}

// Singleton test: verify Runtime::Instance() returns the same object
// on every call, and the singleton can be invoked through CallGuest.
TEST_F(CpuRuntimeTest, Instance_ReturnsSameRuntimeAcrossCalls) {
    Runtime& a = Runtime::Instance();
    Runtime& b = Runtime::Instance();
    EXPECT_EQ(&a, &b);
}

// CurrentGuestState should return nullptr when no JIT is active on
// this thread.
TEST_F(CpuRuntimeTest, CurrentGuestState_NullOutsideRun) {
    EXPECT_EQ(Runtime::CurrentGuestState(), nullptr);
}

// CallGuestOnCallerStack preserves callee-saved registers and RSP.
//
// Guest function clobbers RBX (callee-saved) and RAX (caller-saved).
// After CallGuestOnCallerStack, RBX should be restored to its pre-call
// value but RAX should hold whatever the function put there.
//
// Encoding:
//   mov rax, 0xdead       48 c7 c0 ad de 00 00
//   mov rbx, 0xbeef       48 c7 c3 ef be 00 00
//   ret                   c3
TEST_F(CpuRuntimeTest, CallGuestOnCallerStack_PreservesCalleeSaved) {
    const u8 program[] = {
        0x48, 0xc7, 0xc0, 0xad, 0xde, 0x00, 0x00, // mov rax, 0xdead
        0x48, 0xc7, 0xc3, 0xef, 0xbe, 0x00, 0x00, // mov rbx, 0xbeef
        0xc3,                                     // ret
    };
    std::memcpy(mem.CodePtr(), program, sizeof(program));

    Runtime rt;
    // Set up a "caller" GuestState with known callee-saved values and a
    // valid stack. The stack just has to be a real memory region the
    // sentinel-push will write to.
    GuestState caller{};
    caller.gpr[3] = 0x1111'1111'1111'1111ULL;              // RBX (callee-saved)
    caller.gpr[4] = reinterpret_cast<u64>(mem.StackTop()); // RSP
    caller.gpr[5] = 0x2222'2222'2222'2222ULL;              // RBP (callee-saved)
    caller.gpr[12] = 0x3333'3333'3333'3333ULL;             // R12 (callee-saved)
    caller.gpr[0] = 0xcafe'cafe'cafe'cafeULL;              // RAX (caller-saved)
    caller.rip = 0x4242'4242ULL;                           // Distinct RIP

    const u64 saved_rsp = caller.gpr[4];
    const u64 saved_rip = caller.rip;

    rt.CallGuestOnCallerStack(caller, reinterpret_cast<VAddr>(mem.CodePtr()),
                              nullptr, // no arg setup needed
                              nullptr);

    // Callee-saved should be restored exactly:
    EXPECT_EQ(caller.gpr[3], 0x1111'1111'1111'1111ULL);
    EXPECT_EQ(caller.gpr[5], 0x2222'2222'2222'2222ULL);
    EXPECT_EQ(caller.gpr[12], 0x3333'3333'3333'3333ULL);
    EXPECT_EQ(caller.gpr[4], saved_rsp);
    EXPECT_EQ(caller.rip, saved_rip);

    // RAX (caller-saved) should hold the callback's value (0xdead),
    // not the original (0xcafecafecafecafe).
    EXPECT_EQ(caller.gpr[0], 0xdeadULL);
}

// CallGuestSimpleOnCallerStack convenience wrapper passes args and
// returns RAX.
//
// Guest function: mov rax, rdi; add rax, 1; ret
TEST_F(CpuRuntimeTest, CallGuestSimpleOnCallerStack_PassesArgsAndReturnsRax) {
    const u8 program[] = {
        0x48, 0x89, 0xf8,       // mov rax, rdi
        0x48, 0x83, 0xc0, 0x01, // add rax, 1
        0xc3,                   // ret
    };
    std::memcpy(mem.CodePtr(), program, sizeof(program));

    Runtime rt;
    GuestState caller{};
    caller.gpr[4] = reinterpret_cast<u64>(mem.StackTop());

    u64 result = rt.CallGuestSimpleOnCallerStack(caller, reinterpret_cast<VAddr>(mem.CodePtr()),
                                                 /*a0=*/0x500);

    EXPECT_EQ(result, 0x501u);
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

// InvokeGuestCallback dispatches to CallGuestSimple in the post-JIT
// (no caller) path. The mid-JIT path is exercised implicitly through
// the dual-context call sites; here we verify the post-JIT branch
// works in isolation.
//
// Guest function: takes RDI as input, returns RDI+0x10 in RAX.
//   mov rax, rdi    48 89 f8
//   add rax, 0x10   48 83 c0 10
//   ret             c3
TEST_F(CpuRuntimeTest, InvokeGuestCallback_PostJitPathUsesFreshStack) {
    const u8 program[] = {
        0x48, 0x89, 0xf8,       // mov rax, rdi
        0x48, 0x83, 0xc0, 0x10, // add rax, 0x10
        0xc3,                   // ret
    };
    std::memcpy(mem.CodePtr(), program, sizeof(program));

    // Note: we use the singleton runtime here, not a local Runtime,
    // because InvokeGuestCallback is a method that internally calls
    // CurrentGuestState() — which is a TLS read, not bound to any
    // specific instance. This means previous tests in this suite
    // may have populated the singleton's block cache, but since
    // we're targeting a fresh program at a fresh address, that's
    // fine.
    Runtime& rt = Runtime::Instance();

    // No caller GuestState is active — we're not inside Run() — so
    // CurrentGuestState() returns nullptr and InvokeGuestCallback
    // takes the malloc path.
    EXPECT_EQ(Runtime::CurrentGuestState(), nullptr);

    u64 result = rt.InvokeGuestCallback(reinterpret_cast<VAddr>(mem.CodePtr()),
                                        /*a0=*/0x40);

    EXPECT_EQ(result, 0x50u);
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

    u8* guest_rsp = mem.StackTop() - 8;
    *reinterpret_cast<u64*>(guest_rsp) = kReturnSentinel;

    GuestState state{};
    const u64 program_base = reinterpret_cast<u64>(mem.CodePtr());
    state.rip = program_base;
    state.gpr[4] = reinterpret_cast<u64>(guest_rsp);

    Runtime rt;
    rt.Run(state);

    // After JZ taken: state.rip should be the branch target = 21.
    // (Now that the dispatcher loops, control then enters the
    // block at offset 21, which is zero-padding; the lifter
    // exits with UnsupportedInstruction at offset 21.)
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
        // Fall-through at offset 12: zero-padding.
    };
    std::memcpy(mem.CodePtr(), program, sizeof(program));

    u8* guest_rsp = mem.StackTop() - 8;
    *reinterpret_cast<u64*>(guest_rsp) = kReturnSentinel;

    GuestState state{};
    const u64 program_base = reinterpret_cast<u64>(mem.CodePtr());
    state.rip = program_base;
    state.gpr[4] = reinterpret_cast<u64>(guest_rsp);

    Runtime rt;
    rt.Run(state);

    // Fall-through target is offset 12. The dispatcher tries to
    // continue there and exits with UnsupportedInstruction
    // (zero-padding starts with bytes that aren't 64-bit
    // instructions in this lifter's repertoire).
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

// JMP unconditional rel32: state.rip should be the target regardless
// of any flags.
//
//   jmp +0x100       e9 00 01 00 00
TEST_F(CpuRuntimeTest, Jmp_DirectRel32_SetsRipToTarget) {
    const u8 program[] = {0xe9, 0x00, 0x01, 0x00, 0x00};
    std::memcpy(mem.CodePtr(), program, sizeof(program));

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
// HLE bridge tests
// =============================================================================
//
// These exercise the dispatcher's host-pointer detection: when
// guest code calls into an address that lives inside the host
// binary (any loaded module, per `Runtime::IsGuestPointer`), the
// dispatcher should bypass the JIT lifter, call the host function
// via the SysV-ABI bridge, write the return value to guest RAX,
// pop the guest return address, and continue.
//
// The bridge is the key abstraction that lets the same JIT
// architecture work on a future ARM64 host backend, where lifting
// AArch64 host code as if it were x86 would be nonsense. By
// routing host calls through an explicit ABI-correct C++ bridge,
// the lifter only ever sees real guest x86 code.

// Host function used as the call target. Declared PS4_SYSV_ABI to
// match shadPS4's HLE convention. Sums its six arguments and adds
// a constant so the return value is sensitive to argument order.
static PS4_SYSV_ABI u64 HleBridgeTestFn_SumArgs(u64 a, u64 b, u64 c, u64 d, u64 e, u64 f) {
    // The constants below let us verify each arg landed in the right
    // SysV slot. If we got the marshal order wrong, the sum would
    // still match but checking specific values rules that out.
    return a * 1 + b * 10 + c * 100 + d * 1000 + e * 10000 + f * 100000;
}

// A second host function we *intentionally* leave unregistered, to
// exercise the bridge's "WARNING unregistered host call" path. It
// returns a sentinel so we can confirm execution actually reached
// the function despite the lack of registration.
static PS4_SYSV_ABI u64 HleBridgeTestFn_Unregistered(u64 a, u64, u64, u64, u64, u64) {
    return a + 0xDEAD0000ULL;
}

// Takes a double; converts to int and doubles it. Used to verify
// the bridge marshals state.ymm[0] (low 64) into xmm0 for the call.
static PS4_SYSV_ABI u64 HleBridgeTestFn_DoubleToU64(double x) {
    return static_cast<u64>(x * 2.0);
}

// Returns a double computed from its int arg. Used to verify the
// bridge captures xmm0 (the double's return slot) back into
// state.ymm[0] via the (rax, xmm0) struct-return trick.
static PS4_SYSV_ABI double HleBridgeTestFn_ReturnsDouble(u64 i) {
    return static_cast<double>(i) * 1.5;
}

// Mixed int/float args. SysV allocates integer args and SSE args
// from independent pools, so a=rdi, b=xmm0, c=rsi, d=xmm1. The
// return computes a position-sensitive sum so wrong slots show up
// as obviously wrong values.
static PS4_SYSV_ABI u64 HleBridgeTestFn_MixedArgs(u64 a, double b, u64 c, double d) {
    return a * 1 + static_cast<u64>(b) * 10 + c * 100 + static_cast<u64>(d) * 1000;
}

// Seven integer args: 6 in registers, 1 spilled to stack at
// [rsp+8]. Position-weighted multipliers make wrong slot
// allocation obvious in the failure message.
static PS4_SYSV_ABI u64 HleBridgeTestFn_SevenArgs(u64 a, u64 b, u64 c, u64 d, u64 e, u64 f, u64 g) {
    return a * 1 + b * 10 + c * 100 + d * 1000 + e * 10000 + f * 100000 + g * 1000000;
}

// Eight integer args: 6 in registers, 2 spilled to stack at
// [rsp+8, +16]. Verifies the ORDER of stack args (lowest addr =
// 7th arg = `g`; next = 8th arg = `h`).
static PS4_SYSV_ABI u64 HleBridgeTestFn_EightArgs(u64 a, u64 b, u64 c, u64 d, u64 e, u64 f, u64 g,
                                                  u64 h) {
    return a * 1 + b * 10 + c * 100 + d * 1000 + e * 10000 + f * 100000 + g * 1000000 +
           h * 10000000;
}

// HLE bridge: guest calls a PS4_SYSV_ABI host function with 6
// distinct int args, captures the return value.
//
// Program (RDI=1, RSI=2, RDX=3, RCX=4, R8=5, R9=6 → call host_fn):
//   mov rdi, 1            48 c7 c7 01 00 00 00     (0..6)
//   mov rsi, 2            48 c7 c6 02 00 00 00     (7..13)
//   mov rdx, 3            48 c7 c2 03 00 00 00     (14..20)
//   mov rcx, 4            48 c7 c1 04 00 00 00     (21..27)
//   mov r8,  5            49 c7 c0 05 00 00 00     (28..34)
//   mov r9,  6            49 c7 c1 06 00 00 00     (35..41)
//   mov rax, <host_fn>    48 b8 [8-byte addr]      (42..51)
//   call rax              ff d0                    (52..53)
//   ret                   c3                       (54)
TEST_F(CpuRuntimeTest, HleBridge_GuestCallsHostFunction_MarshalsArgsAndReturn) {
    const u64 host_fn_addr = reinterpret_cast<u64>(&HleBridgeTestFn_SumArgs);

    // Register the test function so the bridge prints "[bridge] call
    // <name>" instead of the warning path. This both confirms the
    // bridge looks up names correctly and avoids polluting the test
    // log with a spurious WARNING.
    HleRegistry::Instance().Register(host_fn_addr, "HleBridgeTestFn_SumArgs");

    u8 program[] = {
        0x48, 0xc7, 0xc7, 0x01, 0x00, 0x00, 0x00,                   // mov rdi, 1
        0x48, 0xc7, 0xc6, 0x02, 0x00, 0x00, 0x00,                   // mov rsi, 2
        0x48, 0xc7, 0xc2, 0x03, 0x00, 0x00, 0x00,                   // mov rdx, 3
        0x48, 0xc7, 0xc1, 0x04, 0x00, 0x00, 0x00,                   // mov rcx, 4
        0x49, 0xc7, 0xc0, 0x05, 0x00, 0x00, 0x00,                   // mov r8,  5
        0x49, 0xc7, 0xc1, 0x06, 0x00, 0x00, 0x00,                   // mov r9,  6
        0x48, 0xb8, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, // mov rax, <imm64>
        0xff, 0xd0,                                                 // call rax
        0xc3,                                                       // ret
    };
    // Patch the 8 bytes after the `48 b8` (offset 42, then prefix 48 b8 = 2)
    // → addr bytes at offsets 44..51.
    std::memcpy(&program[44], &host_fn_addr, sizeof(host_fn_addr));

    const auto r = RunProgram(program, sizeof(program), mem);

    const u64 expected = 1 + 20 + 300 + 4000 + 50000 + 600000;
    EXPECT_EQ(r.state.gpr[0], expected)
        << "Args should be marshaled in SysV order; RAX should hold the host return";
    EXPECT_EQ(r.state.rip, kReturnSentinel)
        << "After host call returns, post-call block's RET should pop sentinel";
}

// HLE bridge: confirm IsGuestPointer correctly classifies the
// host helper. If this fails, the bridge wouldn't fire and the
// JIT would try to lift host code (which is the bug ARM64 would
// hit). The test is here so a regression in host/guest discrimination
// shows up loudly.
TEST_F(CpuRuntimeTest, HleBridge_IsGuestPointer_ClassifiesHostFunction) {
    Runtime rt;
    const void* host_fn = reinterpret_cast<const void*>(&HleBridgeTestFn_SumArgs);
    EXPECT_FALSE(rt.IsGuestPointer(host_fn))
        << "A function in the test binary must be classified as host code";
    EXPECT_TRUE(rt.IsGuestPointer(mem.CodePtr()))
        << "An mmap'd guest code page must be classified as guest";
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

// =============================================================================
// HLE bridge: XMM marshaling (float/double args + double return).
// =============================================================================

// Float arg in xmm0 is passed through to the host function.
//
// Pre-populates state.ymm[0] (low 64 bits = xmm0 low) with the
// bit pattern of 3.14 before running. The bridge should marshal
// this into xmm0 when calling the host fn, which receives 3.14
// as its double arg and returns 6 (3.14 * 2 truncated to u64).
TEST_F(CpuRuntimeTest, HleBridge_FloatArg_PassesViaXmm0) {
    HleRegistry::Instance().ClearForTesting();
    const u64 host_fn_addr = reinterpret_cast<u64>(&HleBridgeTestFn_DoubleToU64);
    HleRegistry::Instance().Register(host_fn_addr, "HleBridgeTestFn_DoubleToU64");

    u8 program[] = {
        0x48, 0xb8, 0, 0, 0, 0, 0, 0, 0, 0, // mov rax, <addr>     (0..9; imm 2..9)
        0xff, 0xd0,                         // call rax            (10..11)
        0xc3,                               // ret                 (12)
    };
    std::memcpy(&program[2], &host_fn_addr, sizeof(host_fn_addr));
    std::memcpy(mem.CodePtr(), program, sizeof(program));

    u8* guest_rsp = mem.StackTop() - 8;
    *reinterpret_cast<u64*>(guest_rsp) = kReturnSentinel;

    GuestState state{};
    state.rip = reinterpret_cast<u64>(mem.CodePtr());
    state.gpr[4] = reinterpret_cast<u64>(guest_rsp);
    // xmm0 = low 64 of ymm[0]; set to bit pattern of 3.14.
    state.ymm[0] = std::bit_cast<u64>(3.14);

    Runtime rt;
    rt.Run(state);

    EXPECT_EQ(state.gpr[0], static_cast<u64>(3.14 * 2.0))
        << "Host fn received xmm0=3.14, should return 6";
    EXPECT_EQ(state.rip, kReturnSentinel);
}

// Double return value lands in state.ymm[0].
//
// Host fn declared to return double; SysV places the return in
// xmm0. Our bridge's HostReturn struct trick captures both rax
// and xmm0 (since (INTEGER, SSE) classified struct is returned
// in those two registers). We verify the xmm0 bit pattern is
// preserved into state.ymm[0].
TEST_F(CpuRuntimeTest, HleBridge_DoubleReturn_CapturedInXmm0) {
    HleRegistry::Instance().ClearForTesting();
    const u64 host_fn_addr = reinterpret_cast<u64>(&HleBridgeTestFn_ReturnsDouble);
    HleRegistry::Instance().Register(host_fn_addr, "HleBridgeTestFn_ReturnsDouble");

    u8 program[] = {
        0x48, 0xc7, 0xc7, 0x04, 0x00, 0x00, 0x00,          // mov rdi, 4         (0..6)
        0x48, 0xb8, 0,    0,    0,    0,    0,    0, 0, 0, // mov rax, <addr>    (7..16; imm 9..16)
        0xff, 0xd0,                                        // call rax           (17..18)
        0xc3,                                              // ret                (19)
    };
    std::memcpy(&program[9], &host_fn_addr, sizeof(host_fn_addr));
    std::memcpy(mem.CodePtr(), program, sizeof(program));

    u8* guest_rsp = mem.StackTop() - 8;
    *reinterpret_cast<u64*>(guest_rsp) = kReturnSentinel;

    GuestState state{};
    state.rip = reinterpret_cast<u64>(mem.CodePtr());
    state.gpr[4] = reinterpret_cast<u64>(guest_rsp);

    Runtime rt;
    rt.Run(state);

    const u64 expected_bits = std::bit_cast<u64>(4.0 * 1.5); // 6.0
    EXPECT_EQ(state.ymm[0], expected_bits)
        << "state.ymm[0] should hold the bit pattern of 6.0 after a "
        << "double-returning HLE call";
    // We deliberately do NOT check state.gpr[0] (rax) here: for a
    // double-returning function, rax is caller-saved and the
    // callee may have left arbitrary values there. The bridge
    // writes both rax and xmm0 regardless; the guest, knowing
    // the function returns double, reads only xmm0.
    EXPECT_EQ(state.rip, kReturnSentinel);
}

// Mixed int and float args land in their independent register
// pools per SysV. fn(u64 a, double b, u64 c, double d) →
// a=rdi, b=xmm0, c=rsi, d=xmm1.
//
// The host function's return uses position-weighted multipliers
// so a wrong slot (e.g. b ending up in xmm1 instead of xmm0)
// produces an obviously different number, not a coincidentally
// correct one.
TEST_F(CpuRuntimeTest, HleBridge_MixedIntAndFloatArgs) {
    HleRegistry::Instance().ClearForTesting();
    const u64 host_fn_addr = reinterpret_cast<u64>(&HleBridgeTestFn_MixedArgs);
    HleRegistry::Instance().Register(host_fn_addr, "HleBridgeTestFn_MixedArgs");

    u8 program[] = {
        0x48, 0xc7, 0xc7, 0x02, 0x00, 0x00, 0x00, // mov rdi, 2 (a)     (0..6)
        0x48, 0xc7, 0xc6, 0x05, 0x00, 0x00, 0x00, // mov rsi, 5 (c)     (7..13)
        0x48, 0xb8, 0,    0,    0,    0,    0,
        0,    0,    0, // mov rax, <addr>    (14..23; imm 16..23)
        0xff, 0xd0,    // call rax           (24..25)
        0xc3,          // ret                (26)
    };
    std::memcpy(&program[16], &host_fn_addr, sizeof(host_fn_addr));
    std::memcpy(mem.CodePtr(), program, sizeof(program));

    u8* guest_rsp = mem.StackTop() - 8;
    *reinterpret_cast<u64*>(guest_rsp) = kReturnSentinel;

    GuestState state{};
    state.rip = reinterpret_cast<u64>(mem.CodePtr());
    state.gpr[4] = reinterpret_cast<u64>(guest_rsp);
    state.ymm[0] = std::bit_cast<u64>(3.0); // xmm0 = b = 3.0
    state.ymm[4] = std::bit_cast<u64>(7.0); // xmm1 = d = 7.0

    Runtime rt;
    rt.Run(state);

    // Expected: 2*1 + 3*10 + 5*100 + 7*1000 = 2 + 30 + 500 + 7000 = 7532
    EXPECT_EQ(state.gpr[0], 7532ULL)
        << "Mixed args should be: rdi=2(a), xmm0=3(b), rsi=5(c), xmm1=7(d)";
    EXPECT_EQ(state.rip, kReturnSentinel);
}

// HLE bridge: 7th integer arg arrives via guest stack at [rsp+8]
// (just past the return address) and lands as the function's 7th
// arg.
//
// Program sets up rdi=1, rsi=2, rdx=3, rcx=4, r8=5, r9=6, pushes
// 7, calls host fn, pops stack arg, returns.
TEST_F(CpuRuntimeTest, HleBridge_StackArg_PassedViaGuestStack) {
    HleRegistry::Instance().ClearForTesting();
    const u64 host_fn_addr = reinterpret_cast<u64>(&HleBridgeTestFn_SevenArgs);
    HleRegistry::Instance().Register(host_fn_addr, "HleBridgeTestFn_SevenArgs");

    u8 program[] = {
        0x48, 0xc7, 0xc7, 0x01, 0x00, 0x00, 0x00,          // mov rdi, 1     (0..6)
        0x48, 0xc7, 0xc6, 0x02, 0x00, 0x00, 0x00,          // mov rsi, 2     (7..13)
        0x48, 0xc7, 0xc2, 0x03, 0x00, 0x00, 0x00,          // mov rdx, 3     (14..20)
        0x48, 0xc7, 0xc1, 0x04, 0x00, 0x00, 0x00,          // mov rcx, 4     (21..27)
        0x49, 0xc7, 0xc0, 0x05, 0x00, 0x00, 0x00,          // mov r8, 5      (28..34)
        0x49, 0xc7, 0xc1, 0x06, 0x00, 0x00, 0x00,          // mov r9, 6      (35..41)
        0x48, 0xc7, 0xc0, 0x07, 0x00, 0x00, 0x00,          // mov rax, 7     (42..48)
        0x50,                                              // push rax       (49)
        0x48, 0xb8, 0,    0,    0,    0,    0,    0, 0, 0, // mov rax, <fn>  (50..59; imm 52..59)
        0xff, 0xd0,                                        // call rax       (60..61)
        0x48, 0x83, 0xc4, 0x08,                            // add rsp, 8     (62..65)
        0xc3,                                              // ret            (66)
    };
    std::memcpy(&program[52], &host_fn_addr, sizeof(host_fn_addr));
    std::memcpy(mem.CodePtr(), program, sizeof(program));

    u8* guest_rsp = mem.StackTop() - 8;
    *reinterpret_cast<u64*>(guest_rsp) = kReturnSentinel;

    GuestState state{};
    state.rip = reinterpret_cast<u64>(mem.CodePtr());
    state.gpr[4] = reinterpret_cast<u64>(guest_rsp);

    Runtime rt;
    rt.Run(state);

    // 1*1 + 2*10 + 3*100 + 4*1000 + 5*10000 + 6*100000 + 7*1000000
    // = 1 + 20 + 300 + 4000 + 50000 + 600000 + 7000000 = 7654321
    EXPECT_EQ(state.gpr[0], 7654321ULL)
        << "7th arg (=7) should reach the host fn via guest stack [rsp+8]";
    EXPECT_EQ(state.rip, kReturnSentinel);
}

// HLE bridge: 8th integer arg arrives via guest stack at [rsp+16].
// Verifies stack arg ORDER: arg 7 pushed last (closest to return
// addr), arg 8 pushed first.
TEST_F(CpuRuntimeTest, HleBridge_TwoStackArgs_OrderedCorrectly) {
    HleRegistry::Instance().ClearForTesting();
    const u64 host_fn_addr = reinterpret_cast<u64>(&HleBridgeTestFn_EightArgs);
    HleRegistry::Instance().Register(host_fn_addr, "HleBridgeTestFn_EightArgs");

    u8 program[] = {
        0x48, 0xc7, 0xc7, 0x01, 0x00, 0x00, 0x00,          // mov rdi, 1     (0..6)
        0x48, 0xc7, 0xc6, 0x02, 0x00, 0x00, 0x00,          // mov rsi, 2     (7..13)
        0x48, 0xc7, 0xc2, 0x03, 0x00, 0x00, 0x00,          // mov rdx, 3     (14..20)
        0x48, 0xc7, 0xc1, 0x04, 0x00, 0x00, 0x00,          // mov rcx, 4     (21..27)
        0x49, 0xc7, 0xc0, 0x05, 0x00, 0x00, 0x00,          // mov r8, 5      (28..34)
        0x49, 0xc7, 0xc1, 0x06, 0x00, 0x00, 0x00,          // mov r9, 6      (35..41)
        0x48, 0xc7, 0xc0, 0x08, 0x00, 0x00, 0x00,          // mov rax, 8     (42..48)
        0x50,                                              // push rax       (49) ; arg 8
        0x48, 0xc7, 0xc0, 0x07, 0x00, 0x00, 0x00,          // mov rax, 7     (50..56)
        0x50,                                              // push rax       (57) ; arg 7 (lowest)
        0x48, 0xb8, 0,    0,    0,    0,    0,    0, 0, 0, // mov rax, <fn>  (58..67; imm 60..67)
        0xff, 0xd0,                                        // call rax       (68..69)
        0x48, 0x83, 0xc4, 0x10,                            // add rsp, 16    (70..73)
        0xc3,                                              // ret            (74)
    };
    std::memcpy(&program[60], &host_fn_addr, sizeof(host_fn_addr));
    std::memcpy(mem.CodePtr(), program, sizeof(program));

    u8* guest_rsp = mem.StackTop() - 8;
    *reinterpret_cast<u64*>(guest_rsp) = kReturnSentinel;

    GuestState state{};
    state.rip = reinterpret_cast<u64>(mem.CodePtr());
    state.gpr[4] = reinterpret_cast<u64>(guest_rsp);

    Runtime rt;
    rt.Run(state);

    // 1 + 20 + 300 + 4000 + 50000 + 600000 + 7000000 + 80000000 = 87654321
    EXPECT_EQ(state.gpr[0], 87654321ULL)
        << "8th arg (=8) should be at [rsp+16], 7th arg (=7) at [rsp+8]; "
        << "if order swapped, would get 78654321";
    EXPECT_EQ(state.rip, kReturnSentinel);
}

//
// Tests focus on the data structure's contract, not the bridge
// integration (which is already covered by HleBridge_* tests above).
// =============================================================================

// Basic register/lookup round-trip.
TEST(HleRegistryTest, RegisterAndLookup) {
    HleRegistry::Instance().ClearForTesting();
    HleRegistry::Instance().Register(0xDEADBEEFULL, "sceKernelFoo");
    EXPECT_EQ(HleRegistry::Instance().Lookup(0xDEADBEEFULL), "sceKernelFoo");
    EXPECT_TRUE(HleRegistry::Instance().Contains(0xDEADBEEFULL));
}

// Unknown addresses return empty.
TEST(HleRegistryTest, UnknownAddressReturnsEmpty) {
    HleRegistry::Instance().ClearForTesting();
    EXPECT_TRUE(HleRegistry::Instance().Lookup(0xCAFEBABEULL).empty());
    EXPECT_FALSE(HleRegistry::Instance().Contains(0xCAFEBABEULL));
}

// Re-registration is idempotent (first write wins). This matters
// because the linker may resolve the same import in multiple
// modules; we want the first registration to stick rather than
// silently overwriting.
TEST(HleRegistryTest, ReRegistrationKeepsFirstName) {
    HleRegistry::Instance().ClearForTesting();
    HleRegistry::Instance().Register(0x1234ULL, "first_name");
    HleRegistry::Instance().Register(0x1234ULL, "second_name");
    EXPECT_EQ(HleRegistry::Instance().Lookup(0x1234ULL), "first_name");
}

// Size grows as new entries are added; unchanged on re-registration.
TEST(HleRegistryTest, SizeReflectsUniqueEntries) {
    HleRegistry::Instance().ClearForTesting();
    EXPECT_EQ(HleRegistry::Instance().Size(), 0u);
    HleRegistry::Instance().Register(0xA, "a");
    HleRegistry::Instance().Register(0xB, "b");
    HleRegistry::Instance().Register(0xA, "a-again"); // duplicate addr
    EXPECT_EQ(HleRegistry::Instance().Size(), 2u);
}

// HLE bridge: confirm unregistered host calls still execute (we
// warn, but don't refuse). This is the "warn + continue" policy:
// killing execution on an unregistered call would prevent us from
// gathering data on edge cases (e.g. callbacks, runtime helpers).
//
// Setup: ensure HleBridgeTestFn_Unregistered is NOT in the registry,
// then call it from guest code. Expect the result to come back
// despite the warning being logged.
TEST_F(CpuRuntimeTest, HleBridge_UnregisteredHostCall_StillExecutes) {
    // Defensive: clear the registry of any prior registration for
    // this function. Tests run in unpredictable order; an earlier
    // test might have registered it.
    HleRegistry::Instance().ClearForTesting();

    const u64 host_fn_addr = reinterpret_cast<u64>(&HleBridgeTestFn_Unregistered);
    ASSERT_TRUE(HleRegistry::Instance().Lookup(host_fn_addr).empty())
        << "test setup invariant: target must be unregistered";

    u8 program[] = {
        0x48, 0xc7, 0xc7, 0x42, 0x00, 0x00, 0x00, // mov rdi, 0x42       (0..6)
        0x48, 0xb8, 0,    0,    0,    0,    0,
        0,    0,    0, // mov rax, <addr>     (7..16; imm at 9..16)
        0xff, 0xd0,    // call rax            (17..18)
        0xc3,          // ret                 (19)
    };
    // imm starts after the 2-byte `48 b8` prefix at offset 9.
    std::memcpy(&program[9], &host_fn_addr, sizeof(host_fn_addr));

    const auto r = RunProgram(program, sizeof(program), mem);
    EXPECT_EQ(r.state.gpr[0], 0x42ULL + 0xDEAD0000ULL)
        << "Unregistered call should still run and return its computed value";
    EXPECT_EQ(r.state.rip, kReturnSentinel);
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
        0x48, 0xb8, 0x05, 0x00, 0x00, 0x00, 0xef,
        0xbe, 0xad, 0xde, 0xc0, 0xe0, 0x02, // shl al, 2 → al = 0x14
        0xc3,                               // ret
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
        0x48, 0xbb, 0xa0, 0x00, 0x00, 0x00, 0xff,
        0xff, 0xff, 0xff, 0xc0, 0xeb, 0x04, // shr bl, 4 → bl = 0x0A
        0xc3,                               // ret
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
        0x48, 0xb8, 0xf0, 0x00, 0x00, 0x00, 0x00,
        0x00, 0x00, 0x00, 0xc0, 0xf8, 0x02, // sar al, 2 → al = 0xFC
        0xc3,                               // ret
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
        0x48, 0xb9, 0x03, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0xd2, 0xe0, // shl al, cl
        0xc3,                                                                   // ret
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
        0x48, 0xb8, 0x78, 0x56, 0x34, 0x12, 0xef, 0xbe, 0xad,
        0xde, 0x48, 0xc7, 0xc1, 0x00, 0x01, 0x00, 0x00, // mov rcx, 0x100
        0x8d, 0x41, 0x40,                               // lea eax, [rcx + 0x40]
        0xc3,                                           // ret
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
        0x48, 0xb8, 0x78, 0x56, 0x34, 0x12, 0xef, 0xbe, 0xad,
        0xde, 0x48, 0xc7, 0xc1, 0xf0, 0x00, 0x00, 0x00, // mov rcx, 0xF0  (src1)
        0x48, 0xc7, 0xc2, 0xff, 0x00, 0x00, 0x00,       // mov rdx, 0xFF  (src2)
        0xc4, 0xe2, 0x70, 0xf2, 0xc2,                   // andn eax, ecx, edx
        0xc3,                                           // ret
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
        0x48, 0xc7, 0xc0, 0x0f, 0x10, 0x00, 0x00, 0xf6, 0xc4, 0x01, // test ah, 0x01
        0xc3,                                                       // ret
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
        0x48, 0xb9, 0xaa, 0x12, 0xfe, 0xca, 0x00,
        0xbe, 0xad, 0xde, 0x80, 0xe5, 0x0f, // and ch, 0x0F → ch = 0x02
        0xc3,                               // ret
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
    EXPECT_EQ(r.state.gpr[2], 2ULL) << "remainder: 100 % 7 = 2";
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
        0x48, 0xb9, 0x00, 0x00, 0x00, 0x00, 0x01, 0x00, 0x00, 0x00, 0x48, 0xf7, 0xf1, // div rcx
        0xc3,                                                                         // ret
    };
    const auto r = RunProgram(program, sizeof(program), mem);
    // (1 << 64) / (1 << 32) = (1 << 32)
    EXPECT_EQ(r.state.gpr[0], 0x100000000ULL) << "quotient = 2^32";
    EXPECT_EQ(r.state.gpr[2], 0ULL) << "remainder = 0";
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
        0xc3, // ret
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
        0xc3, // ret
    };
    const auto r = RunProgram(program, sizeof(program), mem);
    EXPECT_EQ(r.state.gpr[0], 0x12345678ULL)
        << "CMOVZ 32-bit cond TRUE must zero-extend src; rax = 0x12345678";
}

// CMOVZ eax, ecx with the condition FALSE: dst is UNCHANGED — including
// bits 63:32 of the slot. This is the regression catcher. If the lifter
// blindly applied the "32-bit op zero-extends" rule, the upper junk
// (0xDEADBEEF) would be wiped here, breaking guest semantics.
TEST_F(CpuRuntimeTest, Cmov32_ConditionFalse_LeavesUpper32Untouched) {
    const u8 program[] = {
        // mov rax, 0xDEADBEEFCAFEBABE — junk dst (upper 32 = 0xDEADBEEF)
        0x48, 0xb8, 0xbe, 0xba, 0xfe, 0xca, 0xef, 0xbe, 0xad, 0xde,
        // mov rcx, 0x00000000_12345678
        0x48, 0xb9, 0x78, 0x56, 0x34, 0x12, 0x00, 0x00, 0x00, 0x00,
        // mov rdx, 1; test rdx, rdx → ZF=0
        0x48, 0xc7, 0xc2, 0x01, 0x00, 0x00, 0x00, 0x48, 0x85, 0xd2,
        // cmovz eax, ecx — cond FALSE (ZF=0) → no write to rax at all
        0x0f, 0x44, 0xc1,
        0xc3, // ret
    };
    const auto r = RunProgram(program, sizeof(program), mem);
    EXPECT_EQ(r.state.gpr[0], 0xDEADBEEFCAFEBABEULL)
        << "CMOVZ 32-bit cond FALSE must leave the WHOLE qword unchanged, "
           "including bits 63:32 — different from ordinary 32-bit ops";
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
        0x48, 0xc7, 0xc1, 0x00, 0x00, 0x00, 0x00, 0x0f, 0xa2, // cpuid
        0xc3,                                                 // ret
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
        0x48, 0xc7, 0xc0, 0x01, 0x00, 0x00, 0x00, // mov rax, 1
        0x48, 0x31, 0xc9,                         // xor rcx, rcx
        0x0f, 0xa2,                               // cpuid
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
    EXPECT_TRUE(ecx & (1u << 0)) << "SSE3 advertised";
    EXPECT_TRUE(ecx & (1u << 19)) << "SSE4.1 advertised";
    EXPECT_TRUE(ecx & (1u << 20)) << "SSE4.2 advertised";
    EXPECT_TRUE(ecx & (1u << 23)) << "POPCNT advertised";
    EXPECT_TRUE(ecx & (1u << 28)) << "AVX advertised";

    // Features we deliberately do NOT advertise (Jaguar lacks them
    // and/or the JIT lacks emitters):
    EXPECT_FALSE(ecx & (1u << 12)) << "FMA must not be advertised";
    EXPECT_FALSE(ecx & (1u << 30)) << "RDRAND must not be advertised";

    // EDX baseline.
    EXPECT_TRUE(edx & (1u << 0)) << "FPU";
    EXPECT_TRUE(edx & (1u << 25)) << "SSE";
    EXPECT_TRUE(edx & (1u << 26)) << "SSE2";
}

// Leaf 7 subleaf 0 — extended features. We advertise BMI1 only;
// AVX2 and BMI2 must be absent (Jaguar lacks them).
TEST_F(CpuRuntimeTest, Cpuid_Leaf7_Sub0_BmiAdvertisedButNotAvx2) {
    const u8 program[] = {
        0x48, 0xc7, 0xc0, 0x07, 0x00, 0x00, 0x00, // mov rax, 7
        0x48, 0x31, 0xc9,                         // xor rcx, rcx (subleaf 0)
        0x0f, 0xa2,                               // cpuid
        0xc3,
    };
    const auto r = RunProgram(program, sizeof(program), mem);
    const u32 ebx = static_cast<u32>(r.state.gpr[3]);
    EXPECT_TRUE(ebx & (1u << 3)) << "BMI1 advertised";
    EXPECT_FALSE(ebx & (1u << 5)) << "AVX2 must not be advertised";
    EXPECT_FALSE(ebx & (1u << 8)) << "BMI2 must not be advertised";
    EXPECT_FALSE(ebx & (1u << 16)) << "AVX-512 must not be advertised";
}

// Leaf 7 with non-zero subleaf must return all zeros — the emitter
// gates the subleaf-0 response on `ecx == 0` and falls through to
// the zero-default storeback otherwise. Regression catcher: a missing
// gate would expose the subleaf-0 response for every subleaf.
TEST_F(CpuRuntimeTest, Cpuid_Leaf7_NonzeroSubleafReturnsZero) {
    const u8 program[] = {
        0x48, 0xc7, 0xc0, 0x07, 0x00, 0x00, 0x00, // mov rax, 7
        0x48, 0xc7, 0xc1, 0x01, 0x00, 0x00, 0x00, // mov rcx, 1 (sub 1)
        0x0f, 0xa2, 0xc3,
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
            0x48, 0xc7, 0xc0, 0, 0, 0, 0, // mov rax, imm32  (filled below)
            0x48, 0x31, 0xc9,             // xor rcx, rcx
            0x0f, 0xa2,                   // cpuid
            0xc3,                         // ret
        };
        std::memcpy(program + 3, &leaf, 4);
        const auto r = RunProgram(program, sizeof(program), mem);
        out[0] = static_cast<u32>(r.state.gpr[0]); // EAX
        out[1] = static_cast<u32>(r.state.gpr[3]); // EBX
        out[2] = static_cast<u32>(r.state.gpr[1]); // ECX
        out[3] = static_cast<u32>(r.state.gpr[2]); // EDX
        return out;
    };

    char brand[49] = {0};
    const auto b2 = runLeaf(0x80000002);
    const auto b3 = runLeaf(0x80000003);
    const auto b4 = runLeaf(0x80000004);
    for (int i = 0; i < 4; ++i)
        std::memcpy(brand + 0 + i * 4, &b2[i], 4);
    for (int i = 0; i < 4; ++i)
        std::memcpy(brand + 16 + i * 4, &b3[i], 4);
    for (int i = 0; i < 4; ++i)
        std::memcpy(brand + 32 + i * 4, &b4[i], 4);
    // brand is now 48 chars + null. Trim trailing spaces for the
    // comparison so the test isn't sensitive to the exact pad count.
    std::string s(brand);
    while (!s.empty() && s.back() == ' ')
        s.pop_back();
    EXPECT_EQ(s, "AMD Custom Jaguar 8-Core APU");
}

// Unknown leaf (well outside both the standard and extended ranges
// we handle) must return all-zeros. Confirms the default-response
// path in the emitter works.
TEST_F(CpuRuntimeTest, Cpuid_UnknownLeaf_ReturnsZeros) {
    const u8 program[] = {
        0x48, 0xc7, 0xc0, 0x55, 0x55, 0x00, 0x00, // mov rax, 0x5555
        0x48, 0x31, 0xc9, 0x0f, 0xa2, 0xc3,
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
        0x48, 0xc7, 0xc0, 0x20, 0x00, 0x00, 0x00, // mov rax, 0x20
        0x0f, 0xba, 0xe0, 0x05,                   // bt eax, 5
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
        0xf9,                                     // stc
        0x48, 0xc7, 0xc0, 0xdf, 0xff, 0x00, 0x00, // mov rax, 0xFFDF (bit 5 clear)
        0x0f, 0xba, 0xe0, 0x05,                   // bt eax, 5
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
        0x48, 0x39, 0xc0,                         // cmp rax, rax  → ZF=1, CF=0
        0x48, 0xc7, 0xc0, 0x20, 0x00, 0x00, 0x00, // mov rax, 0x20
        0x0f, 0xba, 0xe0, 0x05,                   // bt eax, 5
        0xc3,
    };
    const auto r = RunProgram(program, sizeof(program), mem);
    EXPECT_TRUE(r.state.rflags & 1ULL) << "CF must be 1 after BT";
    EXPECT_TRUE(r.state.rflags & (1ULL << 6)) << "ZF set by earlier cmp must survive BT";
}

// 64-bit imm form — same skeleton, REX.W differentiates encoding.
TEST_F(CpuRuntimeTest, Bt64_Imm_SetBit_RaisesCf) {
    const u8 program[] = {
        0x48, 0xc7, 0xc0, 0x20, 0x00, 0x00, 0x00, // mov rax, 0x20
        0x48, 0x0f, 0xba, 0xe0, 0x05,             // bt rax, 5
        0xc3,
    };
    const auto r = RunProgram(program, sizeof(program), mem);
    EXPECT_TRUE(r.state.rflags & 1ULL);
}

// 32-bit reg-reg form: bit index in ecx. Verifies the new width-32
// reg path routes correctly.
TEST_F(CpuRuntimeTest, Bt32_RegReg_BitIndexFromEcx) {
    const u8 program[] = {
        0x48, 0xc7, 0xc0, 0x00, 0x04, 0x00, 0x00, // mov rax, 0x400 (bit 10)
        0x48, 0xc7, 0xc1, 0x0a, 0x00, 0x00, 0x00, // mov rcx, 10
        0x0f, 0xa3, 0xc8,                         // bt eax, ecx
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
        0x48,
        0xb8,
        0xef,
        0xbe,
        0xad,
        0xde,
        0xef,
        0xbe,
        0xad,
        0xde,
        // mov rdx, 0xCAFEBABECAFEBABE — pre-pollute future RDX
        0x48,
        0xba,
        0xbe,
        0xba,
        0xfe,
        0xca,
        0xbe,
        0xba,
        0xfe,
        0xca,
        // xor rcx, rcx  (ecx = 0 = XCR0)
        0x48,
        0x31,
        0xc9,
        0x0f,
        0x01,
        0xd0, // xgetbv
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
        0x48, 0xc7, 0xc1, 0x01, 0x00, 0x00, 0x00, // mov rcx, 1
        0x0f, 0x01, 0xd0,                         // xgetbv
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
        0xc4, 0xe2, 0x79, 0x17, 0xc1, // vptest xmm0, xmm1
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
    EXPECT_FALSE(st.rflags & 1ULL) << "CF=0 — b is not a subset of a";
}

// Identical operands: a == b ≠ 0. (a AND b) = a ≠ 0 → ZF=0.
// (NOT a) AND b = (NOT a) AND a = 0 → CF=1.
TEST_F(CpuRuntimeTest, Vptest_Identical_SetsCfNotZf) {
    const u8 program[] = {
        0xc4, 0xe2, 0x79, 0x17, 0xc1, // vptest xmm0, xmm1
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
    EXPECT_TRUE(st.rflags & 1ULL) << "CF=1 — b is a subset of a (b == a)";
}

// All-zero operands: both ZF and CF set (everything is the zero set).
TEST_F(CpuRuntimeTest, Vptest_BothZero_SetsBothZfCf) {
    const u8 program[] = {
        0xc4, 0xe2, 0x79, 0x17, 0xc1, // vptest xmm0, xmm1
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
    *scratch = 0x01; // bit 0 set, bit 6 clear

    const u8 program[] = {
        // mov rax, <scratch addr> (filled below)
        0x48, 0xb8, 0, 0, 0, 0, 0, 0, 0, 0, 0x80, 0x08, 0x40, // or byte[rax], 0x40
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
        0x48, 0xb8, 0,    0, 0, 0, 0, 0, 0, 0, // mov rax, <addr>
        0x80, 0x20, 0xF0,                      // and byte[rax], 0xF0
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
        0xc4, 0xe3, 0x79, 0x63, 0xc1, 0x08, // vpcmpistri xmm0, xmm1, 0x08
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
    u8 xmm0_bytes[16] = {'h', 'e', 'l', 'l', 'o', 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0};
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
        0xc4, 0xe3, 0x79, 0x63, 0xc1, 0x08, // vpcmpistri xmm0, xmm1, 0x08
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
    EXPECT_EQ(st.gpr[1] & 0xFFFFFFFFULL, 16ULL) << "No match → ECX = element count (16 bytes)";
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
        0x48,
        0xc7,
        0xc0,
        0x01,
        0x00,
        0x00,
        0x00,
        // mov rdx, 1     (high half = 1 → dividend = 0x1_0000_0001)
        0x48,
        0xc7,
        0xc2,
        0x01,
        0x00,
        0x00,
        0x00,
        // mov rcx, 2     (divisor)
        0x48,
        0xc7,
        0xc1,
        0x02,
        0x00,
        0x00,
        0x00,
        0xf7,
        0xf1, // div ecx
        0xc3,
    };
    const auto r = RunProgram(program, sizeof(program), mem);
    EXPECT_EQ(r.state.gpr[0], 0x80000000ULL) << "Quotient = 0x1_0000_0001 / 2 = 0x8000_0000";
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
        0x48,
        0xb8,
        0x0a,
        0x00,
        0x00,
        0x00,
        0xbe,
        0xba,
        0xfe,
        0xca,
        // mov rdx, 0xDEADBEEF00000000  — upper junk, low = 0
        0x48,
        0xba,
        0x00,
        0x00,
        0x00,
        0x00,
        0xef,
        0xbe,
        0xad,
        0xde,
        // mov rcx, 3
        0x48,
        0xc7,
        0xc1,
        0x03,
        0x00,
        0x00,
        0x00,
        0xf7,
        0xf1, // div ecx
        0xc3,
    };
    const auto r = RunProgram(program, sizeof(program), mem);
    EXPECT_EQ(r.state.gpr[0], 3ULL) << "10 / 3 = 3 (32-bit DIV should ignore upper-32 junk)";
    EXPECT_EQ(r.state.gpr[2], 1ULL) << "10 % 3 = 1";
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
        0x48,
        0xb8,
        0,
        0,
        0,
        0,
        0,
        0,
        0,
        0,
        // mov rcx, 3
        0x48,
        0xc7,
        0xc1,
        0x03,
        0x00,
        0x00,
        0x00,
        0x0f,
        0xc1,
        0x08, // xadd dword[rax], ecx
        0xc3,
    };
    u8 prog[sizeof(program)];
    std::memcpy(prog, program, sizeof(program));
    const u64 scratch_addr = reinterpret_cast<u64>(scratch);
    std::memcpy(prog + 2, &scratch_addr, sizeof(scratch_addr));

    const auto r = RunProgram(prog, sizeof(prog), mem);
    EXPECT_EQ(*scratch, 8u) << "[mem] = old_mem + reg = 5 + 3";
    EXPECT_EQ(r.state.gpr[1], 5ULL) << "reg gets the OLD mem value (5)";
    EXPECT_EQ(r.state.gpr[1] >> 32, 0u) << "RCX upper 32 zero-extended";
}

// Zero-result variant: 0xFFFF_FFFE + 2 overflows to 0 (with CF=1).
// Verifies flags are captured from the addition.
TEST_F(CpuRuntimeTest, Xadd32_WrapToZero_SetsCfAndZf) {
    u32* scratch = reinterpret_cast<u32*>(mem.CodePtr() + 0x100);
    *scratch = 0xFFFFFFFEu;

    const u8 program[] = {
        0x48, 0xb8, 0,    0,    0,    0,    0,    0, 0, 0, // mov rax, <addr>
        0x48, 0xc7, 0xc1, 0x02, 0x00, 0x00, 0x00,          // mov rcx, 2
        0x0f, 0xc1, 0x08,                                  // xadd dword[rax], ecx
        0xc3,
    };
    u8 prog[sizeof(program)];
    std::memcpy(prog, program, sizeof(program));
    const u64 scratch_addr = reinterpret_cast<u64>(scratch);
    std::memcpy(prog + 2, &scratch_addr, sizeof(scratch_addr));

    const auto r = RunProgram(prog, sizeof(prog), mem);
    EXPECT_EQ(*scratch, 0u) << "0xFFFF_FFFE + 2 wraps to 0";
    EXPECT_TRUE(r.state.rflags & 1ULL) << "CF set on 32-bit add wrap";
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
        0xc4, 0xe2, 0x79, 0x02, 0xc1, // vphaddd xmm0, xmm0, xmm1
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
    EXPECT_EQ(out[0], 1u + 2u) << "dst[0] = src1[0] + src1[1]";
    EXPECT_EQ(out[1], 3u + 4u) << "dst[1] = src1[2] + src1[3]";
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
        0x48,
        0xb8,
        0,
        0,
        0,
        0,
        0,
        0,
        0,
        0,
        // vmovaps [rax], xmm0   (3-byte VEX form: c5 f8 29 00)
        0xc5,
        0xf8,
        0x29,
        0x00,
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
        0x48, 0xb8, 0xbe, 0xba, 0xfe, 0xca, 0xef, 0xbe,
        0xad, 0xde, 0xc4, 0xe1, 0xf9, 0x6e, 0xc0, // vmovq xmm0, rax
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
        0xc4, 0xe1, 0xf9, 0x7e, 0xc0, // vmovq rax, xmm0
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
        0x48,
        0xb8,
        0,
        0,
        0,
        0,
        0,
        0,
        0,
        0,
        // setnbe byte[rax]   (3-byte: 0F 97 /0 = 00 mod, /0 ext, [rax])
        0x0f,
        0x97,
        0x00,
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
        0x48, 0xb8, 0,    0, 0, 0, 0, 0, 0, 0, // mov rax, <addr>
        0x0f, 0x97, 0x00,                      // setnbe byte[rax]
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
        0x48, 0xc7, 0xc1, 0xff, 0x00, 0x00, 0x00, // mov rcx, 0xFF
        0xf3, 0x48, 0x0f, 0xb8, 0xc1,             // popcnt rax, rcx
        0xc3,
    };
    const auto r = RunProgram(program, sizeof(program), mem);
    EXPECT_EQ(r.state.gpr[0], 8ULL) << "popcount(0xFF) = 8";
    EXPECT_FALSE(r.state.rflags & (1ULL << 6)) << "ZF clear when src ≠ 0";
}

// POPCNT of 0 → 0, ZF=1.
TEST_F(CpuRuntimeTest, Popcnt64_Zero_SetsZf) {
    const u8 program[] = {
        0x48, 0x31, 0xc9,             // xor rcx, rcx
        0xf3, 0x48, 0x0f, 0xb8, 0xc1, // popcnt rax, rcx
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
        0xc5, 0xf1, 0x6c, 0xc2, // vpunpcklqdq xmm0, xmm1, xmm2
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
        0x48,
        0xbf,
        0,
        0,
        0,
        0,
        0,
        0,
        0,
        0,
        // mov rcx, 0x0808
        0x48,
        0xc7,
        0xc1,
        0x08,
        0x08,
        0x00,
        0x00,
        // bextr eax, dword[rdi], ecx
        0xc4,
        0xe2,
        0x70,
        0xf7,
        0x07,
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
        0xf9, // stc  → CF=1
        // mov rax, <scratch addr>
        0x48,
        0xb8,
        0,
        0,
        0,
        0,
        0,
        0,
        0,
        0,
        // inc byte[rax]   (FE /0 [rax] = 2 bytes; with disp8 = 3)
        0xfe,
        0x00,
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
        0x48, 0xb8, 0, 0, 0, 0, 0, 0, 0, 0, // mov rax, <addr>
        0xfe, 0x00,                         // inc byte[rax]
        0xc3,
    };
    u8 prog[sizeof(program)];
    std::memcpy(prog, program, sizeof(program));
    const u64 scratch_addr = reinterpret_cast<u64>(scratch);
    std::memcpy(prog + 2, &scratch_addr, sizeof(scratch_addr));

    const auto r = RunProgram(prog, sizeof(prog), mem);
    EXPECT_EQ(*scratch, 0x80);
    EXPECT_TRUE(r.state.rflags & (1ULL << 11)) << "OF set on signed wrap +→−";
    EXPECT_TRUE(r.state.rflags & (1ULL << 7)) << "SF set (result MSB = 1)";
    EXPECT_FALSE(r.state.rflags & (1ULL << 6)) << "ZF clear (result ≠ 0)";
}

// ============================================================================
// VPSHUFD dst, src, imm8 — dword shuffle. For each output dword i,
// dst[i] = src[(imm >> (2*i)) & 3].
// ============================================================================

// Broadcast: imm = 0x00 → all four output dwords = src[0].
TEST_F(CpuRuntimeTest, Vpshufd_BroadcastsLowestDword) {
    const u8 program[] = {
        0xc5, 0xf9, 0x70, 0xc1, 0x00, // vpshufd xmm0, xmm1, 0x00
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
        0xc5, 0xf9, 0x70, 0xc1, 0x1b, // vpshufd xmm0, xmm1, 0x1b
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
        0x48, 0xc7, 0xc1, 0xff, 0x00, 0x00, 0x00, // mov rcx, 0xFF
        0xf3, 0x0f, 0xbd, 0xc1,                   // lzcnt eax, ecx
        0xc3,
    };
    const auto r = RunProgram(program, sizeof(program), mem);
    EXPECT_EQ(r.state.gpr[0], 24ULL) << "lzcnt(0xFF) for 32-bit = 24";
    EXPECT_FALSE(r.state.rflags & 1ULL) << "CF clear when src ≠ 0";
    EXPECT_FALSE(r.state.rflags & (1ULL << 6)) << "ZF clear (result ≠ 0)";
}

// LZCNT of 0: result = operand size (32), CF = 1 (signals "no bits set").
TEST_F(CpuRuntimeTest, Lzcnt32_ZeroSrc_ReturnsOperandSizeAndSetsCf) {
    const u8 program[] = {
        0x48, 0x31, 0xc9,       // xor rcx, rcx
        0xf3, 0x0f, 0xbd, 0xc1, // lzcnt eax, ecx
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
        0x48, 0xc7, 0xc1, 0x00, 0x00, 0x00, 0x80, 0xf3, 0x0f, 0xbd, 0xc1, // lzcnt eax, ecx
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
        0x48, 0xc7, 0xc0, 0x0a, 0x00, 0x00, 0x00, // mov rax, 0x0A
        0x48, 0xc7, 0xc1, 0x03, 0x00, 0x00, 0x00, // mov rcx, 0x03
        0xf8,                                     // clc → CF=0
        0x18, 0xc8,                               // sbb al, cl
        0xc3,
    };
    const auto r = RunProgram(program, sizeof(program), mem);
    EXPECT_EQ(r.state.gpr[0] & 0xFFULL, 7ULL) << "10 - 3 - 0 = 7";
    EXPECT_FALSE(r.state.rflags & 1ULL) << "no borrow out";
}

// 10 - 3 - 1(CF) = 6. Verifies the CF input is actually consumed.
TEST_F(CpuRuntimeTest, Sbb8_BorrowSet_SubtractsExtraOne) {
    const u8 program[] = {
        0x48, 0xc7, 0xc0, 0x0a, 0x00, 0x00, 0x00, // mov rax, 0x0A
        0x48, 0xc7, 0xc1, 0x03, 0x00, 0x00, 0x00, // mov rcx, 0x03
        0xf9,                                     // stc → CF=1
        0x18, 0xc8,                               // sbb al, cl
        0xc3,
    };
    const auto r = RunProgram(program, sizeof(program), mem);
    EXPECT_EQ(r.state.gpr[0] & 0xFFULL, 6ULL) << "10 - 3 - 1 = 6";
    EXPECT_FALSE(r.state.rflags & 1ULL) << "no borrow out";
}

// 0 - 1 - 0 wraps to 0xFF (-1 in two's complement 8-bit). CF=1 (borrow out).
TEST_F(CpuRuntimeTest, Sbb8_Underflow_SetsCf) {
    const u8 program[] = {
        0x48, 0x31, 0xc0,                         // xor rax, rax  (al=0)
        0x48, 0xc7, 0xc1, 0x01, 0x00, 0x00, 0x00, // mov rcx, 1
        0xf8,                                     // clc → CF=0
        0x18, 0xc8,                               // sbb al, cl
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
        0x48,
        0xbf,
        0,
        0,
        0,
        0,
        0,
        0,
        0,
        0,
        // mov rax, 0xDEADBEEFCAFEBABE — pre-pollute dst's upper 48
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
        // Force SF=0 via xor rcx,rcx (which clears SF). Then cmovns ax, word[rdi].
        0x48,
        0x31,
        0xc9, // xor rcx, rcx (SF=0 now)
        0x66,
        0x0f,
        0x49,
        0x07, // cmovns ax, word[rdi]
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
        0x48,
        0xbf,
        0,
        0,
        0,
        0,
        0,
        0,
        0,
        0,
        // mov rax, 0xDEADBEEFCAFEBABE — should remain entirely unchanged
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
        // Force SF=1 by computing a negative result (cmp 0, 1 → SF=1, CF=1).
        0x48,
        0xc7,
        0xc1,
        0x00,
        0x00,
        0x00,
        0x00, // mov rcx, 0
        0x48,
        0x83,
        0xf9,
        0x01, // cmp rcx, 1   → SF=1
        // cmovns ax, word[rdi]  — condition FALSE → no change
        0x66,
        0x0f,
        0x49,
        0x07,
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
        0xf8,             // clc → CF=0
        0x83, 0xd9, 0x10, // sbb ecx, 0x10
        0xc3,
    };
    const auto r = RunProgram(program, sizeof(program), mem);
    EXPECT_EQ(r.state.gpr[1], 0xF0ULL) << "0x100 - 0x10 - 0 = 0xF0; upper 32 zero-extended";
    EXPECT_FALSE(r.state.rflags & 1ULL) << "no borrow out";
}

// `sbb ecx, 0x10` with rcx = 0x100 and CF = 1 → result = 0xEF.
TEST_F(CpuRuntimeTest, Sbb32_RegImm_BorrowSet_SubtractsExtraOne) {
    const u8 program[] = {
        0x48, 0xc7, 0xc1, 0x00, 0x01, 0x00, 0x00, // mov rcx, 0x100
        0xf9,                                     // stc → CF=1
        0x83, 0xd9, 0x10,                         // sbb ecx, 0x10
        0xc3,
    };
    const auto r = RunProgram(program, sizeof(program), mem);
    EXPECT_EQ(r.state.gpr[1], 0xEFULL) << "0x100 - 0x10 - 1 = 0xEF; CF input consumed";
}

// Underflow: 0 - 1 - 0 wraps to 0xFFFFFFFF with CF=1.
TEST_F(CpuRuntimeTest, Sbb32_RegImm_Underflow_SetsCf) {
    const u8 program[] = {
        0x48, 0x31, 0xc9, // xor rcx, rcx
        0xf8,             // clc
        0x83, 0xd9, 0x01, // sbb ecx, 1
        0xc3,
    };
    const auto r = RunProgram(program, sizeof(program), mem);
    EXPECT_EQ(r.state.gpr[1], 0xFFFFFFFFULL) << "0 - 1 wraps to 0xFFFFFFFF (low 32)";
    EXPECT_EQ(r.state.gpr[1] >> 32, 0u) << "32-bit op must zero-extend bits 63:32";
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
        0x48,
        0xb8,
        0,
        0,
        0,
        0,
        0,
        0,
        0,
        0,
        // vmovss xmm0, dword[rax]  (c5 fa 10 00 = 4 bytes)
        0xc5,
        0xfa,
        0x10,
        0x00,
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
        0x48,
        0xb8,
        0,
        0,
        0,
        0,
        0,
        0,
        0,
        0,
        // vmovss dword[rax], xmm0  (c5 fa 11 00)
        0xc5,
        0xfa,
        0x11,
        0x00,
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
        0x48, 0xc7, 0xc0, 0x2a, 0x00, 0x00, 0x00, // mov rax, 42
        0xc5, 0xf2, 0x2a, 0xc8,                   // vcvtsi2ss xmm1, xmm1, eax
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
    const u32 hi32 = static_cast<u32>(st.ymm[4] >> 32);
    float as_float;
    std::memcpy(&as_float, &low32, sizeof(as_float));
    EXPECT_EQ(as_float, 42.0f);
    EXPECT_EQ(hi32, 0xAAAAAAAAu) << "src1[63:32] must be preserved (dst==src1 here)";
    EXPECT_EQ(st.ymm[5], 0xBBBBBBBBBBBBBBBBULL) << "chunk 1 preserved";
    EXPECT_EQ(st.ymm[6], 0ULL) << "VEX-128 zeroes upper YMM";
    EXPECT_EQ(st.ymm[7], 0ULL);
}

// Convert negative int32 to float. (-100).f = 0xC2C80000.
TEST_F(CpuRuntimeTest, Vcvtsi2ss_NegativeInt32) {
    const u8 program[] = {
        // mov rax, -100  (encoded as imm32 sign-extended)
        0x48, 0xc7, 0xc0, 0x9c, 0xff, 0xff,
        0xff, 0xc5, 0xf2, 0x2a, 0xc8, // vcvtsi2ss xmm1, xmm1, eax
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
        0x48, 0xc7, 0xc0, 0x07, 0x00, 0x00, 0x00, // mov rax, 7
        0xc5, 0xf2, 0x2a, 0xc0,                   // vcvtsi2ss xmm0, xmm1, eax
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
    const u32 hi32 = static_cast<u32>(st.ymm[0] >> 32);
    float as_float;
    std::memcpy(&as_float, &low32, sizeof(as_float));
    EXPECT_EQ(as_float, 7.0f);
    EXPECT_EQ(hi32, 0x77777777u) << "dst[63:32] must come from src1, not from pre-existing dst";
    EXPECT_EQ(st.ymm[1], 0x8888888888888888ULL) << "dst chunk 1 from src1";
    EXPECT_EQ(st.ymm[2], 0ULL);
    EXPECT_EQ(st.ymm[3], 0ULL);
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
        0xc5, 0xf2, 0x59, 0xc2, 0xc3,
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
    const u32 hi32 = static_cast<u32>(st.ymm[0] >> 32);
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
        0x48,
        0xb8,
        0,
        0,
        0,
        0,
        0,
        0,
        0,
        0,
        // vmulss xmm0, xmm1, dword[rax]   (c5 f2 59 00)
        0xc5,
        0xf2,
        0x59,
        0x00,
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
    EXPECT_EQ(st.gpr[0], 0xFFFFFFFDULL) << "-3.7f truncates to -3 (sign-bit pattern in low 32, "
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
// VDIVSS — scalar single-precision divide. Structurally identical to
// VMULSS (shares EmitScalarFpSs); separate tests just confirm the
// dispatch reaches the Div case correctly and the result is right.
// ============================================================================

// 12.0 / 4.0 = 3.0
TEST_F(CpuRuntimeTest, Vdivss_BasicDivide) {
    // vdivss xmm0, xmm1, xmm2  — encoding c5 f2 5e c2 (opcode 0x5E vs MUL's 0x59)
    const u8 program[] = {
        0xc5, 0xf2, 0x5e, 0xc2, 0xc3,
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
    st.ymm[4] = (static_cast<u64>(0x11223344ULL) << 32) | std::bit_cast<u32>(20.0f);
    st.ymm[5] = 0x5566778899AABBCCULL;
    st.ymm[8] = static_cast<u64>(std::bit_cast<u32>(5.0f));
    st.ymm[0] = 0xFFFFFFFFFFFFFFFFULL;
    st.ymm[1] = 0xFFFFFFFFFFFFFFFFULL;
    st.ymm[2] = 0xFFFFFFFFFFFFFFFFULL;
    st.ymm[3] = 0xFFFFFFFFFFFFFFFFULL;

    Runtime rt;
    rt.Run(st);
    const u32 low32 = static_cast<u32>(st.ymm[0] & 0xFFFFFFFFULL);
    const u32 hi32 = static_cast<u32>(st.ymm[0] >> 32);
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
        0x48, 0xb8, 0,    0,    0, 0, 0, 0, 0, 0, // mov rax, <fmem>
        0xc5, 0xf2, 0x5e, 0x00,                   // vdivss xmm0, xmm1, dword[rax]
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
        0xc5, 0xf2, 0x58, 0xc2, 0xc3,
    };
    std::memcpy(mem.CodePtr(), program, sizeof(program));
    u8* guest_rsp = mem.StackTop() - 8;
    *reinterpret_cast<u64*>(guest_rsp) = kReturnSentinel;
    GuestState st{};
    st.rip = reinterpret_cast<u64>(mem.CodePtr());
    st.gpr[4] = reinterpret_cast<u64>(guest_rsp);
    st.ymm[4] = static_cast<u64>(std::bit_cast<u32>(1.5f));  // xmm1
    st.ymm[8] = static_cast<u64>(std::bit_cast<u32>(2.25f)); // xmm2
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
        0x48, 0xb8, 0, 0, 0, 0, 0, 0, 0, 0, 0xc5, 0xf2, 0x58, 0x00, // vaddss xmm0, xmm1, dword[rax]
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
        0xc5, 0xf8, 0x2e, 0xc1, 0xc3,
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
        0xc5, 0xf2, 0x5a, 0xc2, 0xc3,
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
    EXPECT_EQ(st.ymm[1], 0xCAFEBABE12345678ULL) << "dst[127:64] must come from src1[127:64]";
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
        0xc5, 0xf3, 0x59, 0xc2, 0xc3,
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
    st.ymm[4] = std::bit_cast<u64>(2.0); // xmm1 low64
    st.ymm[5] = 0xF00DBABEAABBCCDDULL;   // xmm1 high64 — must land in dst.chunk1
    st.ymm[8] = std::bit_cast<u64>(3.0); // xmm2 low64
    st.ymm[0] = 0xFFFFFFFFFFFFFFFFULL;
    st.ymm[1] = 0xFFFFFFFFFFFFFFFFULL;
    st.ymm[2] = 0xFFFFFFFFFFFFFFFFULL;
    st.ymm[3] = 0xFFFFFFFFFFFFFFFFULL;

    Runtime rt;
    rt.Run(st);
    double result;
    std::memcpy(&result, &st.ymm[0], sizeof(result));
    EXPECT_EQ(result, 6.0);
    EXPECT_EQ(st.ymm[1], 0xF00DBABEAABBCCDDULL) << "dst[127:64] must come from src1[127:64]";
    EXPECT_EQ(st.ymm[2], 0ULL);
    EXPECT_EQ(st.ymm[3], 0ULL);
}

// Memory-source form — exercises the SD-specific `vmovsd qword[rdx]`
// loader path (vs `vmovss dword[rdx]` for the SS variant).
TEST_F(CpuRuntimeTest, Vmulsd_MemorySource) {
    u64* dmem = reinterpret_cast<u64*>(mem.CodePtr() + 0x100);
    *dmem = std::bit_cast<u64>(4.0);
    const u8 program[] = {
        0x48, 0xb8, 0,    0,    0, 0, 0, 0, 0, 0, // mov rax, <dmem>
        0xc5, 0xf3, 0x59, 0x00,                   // vmulsd xmm0, xmm1, qword[rax]
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
        0xc5, 0xf3, 0x58, 0xc2, 0xc3,
    };
    std::memcpy(mem.CodePtr(), program, sizeof(program));
    u8* guest_rsp = mem.StackTop() - 8;
    *reinterpret_cast<u64*>(guest_rsp) = kReturnSentinel;
    GuestState st{};
    st.rip = reinterpret_cast<u64>(mem.CodePtr());
    st.gpr[4] = reinterpret_cast<u64>(guest_rsp);
    st.ymm[4] = std::bit_cast<u64>(1.5);  // xmm1
    st.ymm[8] = std::bit_cast<u64>(2.25); // xmm2
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
        0x48, 0xb8, 0, 0, 0, 0, 0, 0, 0, 0, 0xc5, 0xf3, 0x58, 0x00, // vaddsd xmm0, xmm1, qword[rax]
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
        0xc5, 0xf3, 0x5a, 0xc2, 0xc3,
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
        0xc5, 0xf3, 0x5a, 0xc2, 0xc3,
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
    const u32 hi32 = static_cast<u32>(st.ymm[0] >> 32);
    float result;
    std::memcpy(&result, &low32, sizeof(result));
    EXPECT_EQ(result, 1.0f);
    EXPECT_EQ(hi32, 0xCAFEFACEu)
        << "dst[63:32] must come from src1[63:32] (merge boundary at bit 32)";
    EXPECT_EQ(st.ymm[1], 0x1234567812345678ULL) << "dst[127:64] from src1[127:64]";
    EXPECT_EQ(st.ymm[2], 0ULL);
    EXPECT_EQ(st.ymm[3], 0ULL);
}

// Narrowing rounding: a double value that doesn't fit exactly in float
// gets rounded under MXCSR. We verify the result matches what the host
// C++ compiler produces for (float)d, which uses the same MXCSR.
TEST_F(CpuRuntimeTest, Vcvtsd2ss_NarrowingRoundsCorrectly) {
    const u8 program[] = {
        0xc5, 0xf3, 0x5a, 0xc2, 0xc3,
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
        0xc5, 0xf2, 0x5c, 0xc2, 0xc3,
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
        0x48, 0xb8, 0,    0,    0, 0, 0, 0, 0, 0, // mov rax, <scratch>
        0xc5, 0xfb, 0x10, 0x00,                   // vmovsd xmm0, qword[rax]
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
        0x48, 0xb8, 0, 0, 0, 0, 0, 0, 0, 0, 0xc5, 0xfb, 0x11, 0x00, // vmovsd qword[rax], xmm0
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
    st.ymm[0] = std::bit_cast<u64>(1.5); // xmm0.low64
    st.ymm[1] = 0xFFFFFFFFFFFFFFFFULL;   // xmm0[127:64] — must NOT be stored
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
        0xc5, 0xf3, 0x10, 0xc2, 0xc3,
    };
    std::memcpy(mem.CodePtr(), program, sizeof(program));
    u8* guest_rsp = mem.StackTop() - 8;
    *reinterpret_cast<u64*>(guest_rsp) = kReturnSentinel;
    GuestState st{};
    st.rip = reinterpret_cast<u64>(mem.CodePtr());
    st.gpr[4] = reinterpret_cast<u64>(guest_rsp);
    // src1 = xmm1 (lane 4): chunk 0 ignored, chunk 1 preserved into dst.chunk1
    st.ymm[4] = 0xDEAD000000000000ULL; // chunk 0 - ignored
    st.ymm[5] = 0xCAFEBABE11223344ULL; // chunk 1 - preserved
    // src2 = xmm2 (lane 8): chunk 0 → dst.chunk0
    st.ymm[8] = 0x123456789ABCDEF0ULL;
    st.ymm[9] = 0x9999999999999999ULL; // ignored
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
        0xc5, 0xf3, 0x5c, 0xc2, 0xc3,
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
        0xc5, 0xf3, 0x5e, 0xc2, 0xc3,
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
        0x48, 0xb8, 0,    0,    0, 0, 0, 0, 0, 0, // mov rax, <mem_op>
        0xc5, 0xf0, 0x57, 0x00,                   // vxorps xmm0, xmm1, xmmword[rax]
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
    EXPECT_EQ(st.ymm[0], 0xF0F0F0F0F0F0F0F0ULL) << "chunk 0 = 0x0F0F... XOR 0xFFFF... = 0xF0F0...";
    EXPECT_EQ(st.ymm[1], 0xFFFFFFFFFFFFFFFFULL) << "chunk 1 = 0xAAAA... XOR 0x5555... = 0xFFFF...";
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
        0x48, 0xb8, 0, 0, 0, 0, 0, 0, 0, 0, 0xc5, 0xf0, 0x57, 0x00, 0xc3,
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
        0xc5, 0xf1, 0xdb, 0xc2, 0xc3,
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
    EXPECT_EQ(st.ymm[1], 0x8888888888888888ULL) << "chunk 1 = 0xAAAA... AND 0xCCCC... = 0x8888...";
    EXPECT_EQ(st.ymm[2], 0ULL) << "VEX-128 zeros upper YMM";
    EXPECT_EQ(st.ymm[3], 0ULL);
}

// Absolute-value idiom for float: AND with 0x7FFFFFFF clears the sign bit.
// Same use case as VXORPS sign-flip, just AND instead of XOR.
TEST_F(CpuRuntimeTest, Vpand_MemSource_AbsValuePattern) {
    u32* mask = reinterpret_cast<u32*>(mem.CodePtr() + 0x100);
    mask[0] = 0x7FFFFFFFu;                     // clears sign bit
    mask[1] = mask[2] = mask[3] = 0xFFFFFFFFu; // leave other lanes alone

    const u8 program[] = {
        0x48, 0xb8, 0, 0,    0,    0,    0,
        0,    0,    0, 0xc5, 0xf1, 0xdb, 0x00, // vpand xmm0, xmm1, xmmword[rax]
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
        0xc5, 0xf3, 0x51, 0xc2, 0xc3,
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
        0xc4, 0xe3, 0x71, 0x0c, 0xc2, 0x0a, 0xc3,
    };
    std::memcpy(mem.CodePtr(), program, sizeof(program));
    u8* guest_rsp = mem.StackTop() - 8;
    *reinterpret_cast<u64*>(guest_rsp) = kReturnSentinel;
    GuestState st{};
    st.rip = reinterpret_cast<u64>(mem.CodePtr());
    st.gpr[4] = reinterpret_cast<u64>(guest_rsp);
    // xmm1 (lane 4): 4 floats = 1.0, 2.0, 3.0, 4.0
    st.ymm[4] = (static_cast<u64>(std::bit_cast<u32>(2.0f)) << 32) |
                static_cast<u64>(std::bit_cast<u32>(1.0f));
    st.ymm[5] = (static_cast<u64>(std::bit_cast<u32>(4.0f)) << 32) |
                static_cast<u64>(std::bit_cast<u32>(3.0f));
    // xmm2 (lane 8): 4 floats = 10.0, 20.0, 30.0, 40.0
    st.ymm[8] = (static_cast<u64>(std::bit_cast<u32>(20.0f)) << 32) |
                static_cast<u64>(std::bit_cast<u32>(10.0f));
    st.ymm[9] = (static_cast<u64>(std::bit_cast<u32>(40.0f)) << 32) |
                static_cast<u64>(std::bit_cast<u32>(30.0f));

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
        0xc4, 0xe3, 0x71, 0x0c, 0xc2, 0x00, 0xc3,
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
        0xc4, 0xe3, 0x71, 0x0c, 0xc2, 0x0f, 0xc3,
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
        0xc5, 0xf1, 0x76, 0xc2, 0xc3,
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
        0x48, 0xb8, 0, 0,    0,    0,    0,
        0,    0,    0, 0xc5, 0xf1, 0x76, 0x00, // vpcmpeqd xmm0, xmm1, xmmword[rax]
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
        0xc5, 0xf1, 0xfa, 0xc2, 0xc3,
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
        0xc5, 0xf9, 0x72, 0xe1, 0x02, 0xc3,
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
        0xc5, 0xf9, 0x72, 0xe1, 0x04, 0xc3,
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
    const u32 neg1 = static_cast<u32>(-1);
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
    EXPECT_EQ(static_cast<u32>(st.ymm[0] >> 32), 0xFFFFFFFFu) << "-1 >>arith 4 = -1";
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
        0xc5, 0xf9, 0x72, 0xe1, 0x64, 0xc3,
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
        0xc5, 0xf1, 0xfe, 0xc2, 0xc3,
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
        0x48, 0xb8, 0, 0,    0,    0,    0,
        0,    0,    0, 0xc5, 0xf1, 0xfe, 0x00, // vpaddd xmm0, xmm1, xmmword[rax]
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

} // namespace
} // namespace Core::Runtime
