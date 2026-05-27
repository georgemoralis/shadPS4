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

#include <cstdint>
#include <cstring>
#include <vector>

#include <gtest/gtest.h>

#ifdef _WIN32
#include <windows.h>
#else
#include <sys/mman.h>
#endif

#include "core/cpu_runtime/block_cache.h"
#include "core/cpu_runtime/guest_state.h"
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
class GuestMemory {
public:
    static constexpr u64 PAGE_SIZE = 4096;
    static constexpr u64 TOTAL_SIZE = PAGE_SIZE * 2;

    GuestMemory() {
#ifdef _WIN32
        void* p = ::VirtualAlloc(nullptr, TOTAL_SIZE, MEM_COMMIT | MEM_RESERVE,
                                 PAGE_EXECUTE_READWRITE);
        base_ = static_cast<u8*>(p);
#else
        void* p = ::mmap(nullptr, TOTAL_SIZE, PROT_READ | PROT_WRITE | PROT_EXEC,
                         MAP_PRIVATE | MAP_ANON, -1, 0);
        base_ = (p == MAP_FAILED) ? nullptr : static_cast<u8*>(p);
#endif
    }

    ~GuestMemory() {
        if (base_ == nullptr) return;
#ifdef _WIN32
        ::VirtualFree(base_, 0, MEM_RELEASE);
#else
        ::munmap(base_, TOTAL_SIZE);
#endif
    }

    GuestMemory(const GuestMemory&) = delete;
    GuestMemory& operator=(const GuestMemory&) = delete;

    [[nodiscard]] u8* CodePtr() const { return base_; }
    [[nodiscard]] u8* StackTop() const { return base_ + TOTAL_SIZE; }
    [[nodiscard]] bool IsValid() const { return base_ != nullptr; }

private:
    u8* base_ = nullptr;
};

/// Set up a GuestState with the given program, push the sentinel
/// return address onto the guest stack, run the runtime, and return.
struct RunResult {
    GuestState state{};
    u64 program_base{};
};

RunResult RunProgram(const u8* program, size_t program_size,
                     GuestMemory& mem) {
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
        0x48, 0xc7, 0xc0, 0x42, 0x00, 0x00, 0x00,  // mov rax, 0x42
        0x48, 0x83, 0xc0, 0x01,                     // add rax, 1
        0xc3,                                       // ret
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
        0x48, 0xc7, 0xc0, 0x00, 0x01, 0x00, 0x00,  // mov rax, 0x100
        0x48, 0xc7, 0xc3, 0x00, 0x02, 0x00, 0x00,  // mov rbx, 0x200
        0x48, 0x01, 0xd8,                           // add rax, rbx
        0x48, 0x83, 0xc0, 0x10,                     // add rax, 0x10
        0x48, 0x89, 0xc1,                           // mov rcx, rax
        0xc3,                                       // ret
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
        0x48, 0xc7, 0xc0, 0x01, 0x00, 0x00, 0x00,  // mov rax, 1
        0x48, 0x0f, 0xbd, 0xd8,                     // bsr rbx, rax   (unsupported)
        0xc3,                                       // ret  (unreached)
    };
    const auto r = RunProgram(program, sizeof(program), mem);

    // RIP should point to the BSR instruction (at offset 7).
    EXPECT_EQ(r.state.rip, r.program_base + 7);
    EXPECT_EQ(r.state.exit_reason,
              static_cast<u32>(ExitReason::UnsupportedInstruction));
    // MOV before BSR should have executed.
    EXPECT_EQ(r.state.gpr[0], 1u);
}

// Stress test: many runs through the same Runtime instance to verify
// the block cache reuses compiled blocks and doesn't leak host code.
TEST_F(CpuRuntimeTest, MultipleRuns_ShareBlockCache) {
    const u8 program[] = {
        0x48, 0xc7, 0xc0, 0x42, 0x00, 0x00, 0x00,  // mov rax, 0x42
        0xc3,                                       // ret
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
        0x48, 0x89, 0xf8,                           // mov rax, rdi
        0x48, 0x83, 0xc0, 0x01,                     // add rax, 1
        0xc3,                                       // ret
    };
    std::memcpy(mem.CodePtr(), program, sizeof(program));

    // Use a local Runtime instance for test isolation. The singleton
    // would share its block cache with other tests, and since mmap
    // may return overlapping addresses across tests, the cache could
    // serve stale compiled code from a previous test's program.
    Runtime rt;
    u64 result = rt.CallGuestSimple(
        reinterpret_cast<VAddr>(mem.CodePtr()),
        mem.StackTop(),
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
        0x48, 0x89, 0xf8,                           // mov rax, rdi
        0x48, 0x01, 0xf0,                           // add rax, rsi
        0xc3,                                       // ret
    };
    std::memcpy(mem.CodePtr(), program, sizeof(program));

    struct Args { u64 x, y; };
    Args args{0x200, 0x40};

    auto setup = [](GuestState& state, void* user_data) {
        const auto* a = static_cast<const Args*>(user_data);
        state.gpr[7] = a->x;  // RDI
        state.gpr[6] = a->y;  // RSI
    };

    // Local Runtime — see note in CallGuestSimple test above.
    Runtime rt;
    GuestState result = rt.CallGuest(
        reinterpret_cast<VAddr>(mem.CodePtr()),
        mem.StackTop(),
        setup, &args);

    EXPECT_EQ(result.gpr[0], 0x240u);              // RAX = x + y
    EXPECT_EQ(result.gpr[6], args.y);              // RSI unchanged
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
        0x48, 0xc7, 0xc0, 0xad, 0xde, 0x00, 0x00,    // mov rax, 0xdead
        0x48, 0xc7, 0xc3, 0xef, 0xbe, 0x00, 0x00,    // mov rbx, 0xbeef
        0xc3,                                         // ret
    };
    std::memcpy(mem.CodePtr(), program, sizeof(program));

    Runtime rt;
    // Set up a "caller" GuestState with known callee-saved values and a
    // valid stack. The stack just has to be a real memory region the
    // sentinel-push will write to.
    GuestState caller{};
    caller.gpr[3] = 0x1111'1111'1111'1111ULL;        // RBX (callee-saved)
    caller.gpr[4] = reinterpret_cast<u64>(mem.StackTop());  // RSP
    caller.gpr[5] = 0x2222'2222'2222'2222ULL;        // RBP (callee-saved)
    caller.gpr[12] = 0x3333'3333'3333'3333ULL;       // R12 (callee-saved)
    caller.gpr[0] = 0xcafe'cafe'cafe'cafeULL;        // RAX (caller-saved)
    caller.rip = 0x4242'4242ULL;                     // Distinct RIP

    const u64 saved_rsp = caller.gpr[4];
    const u64 saved_rip = caller.rip;

    rt.CallGuestOnCallerStack(
        caller,
        reinterpret_cast<VAddr>(mem.CodePtr()),
        nullptr,  // no arg setup needed
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
        0x48, 0x89, 0xf8,                           // mov rax, rdi
        0x48, 0x83, 0xc0, 0x01,                     // add rax, 1
        0xc3,                                       // ret
    };
    std::memcpy(mem.CodePtr(), program, sizeof(program));

    Runtime rt;
    GuestState caller{};
    caller.gpr[4] = reinterpret_cast<u64>(mem.StackTop());

    u64 result = rt.CallGuestSimpleOnCallerStack(
        caller,
        reinterpret_cast<VAddr>(mem.CodePtr()),
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
        0x48, 0x89, 0xf8,                          // mov rax, rdi
        0x48, 0x83, 0xc0, 0x10,                    // add rax, 0x10
        0xc3,                                       // ret
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

    u64 result = rt.InvokeGuestCallback(
        reinterpret_cast<VAddr>(mem.CodePtr()),
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

    u8* guest_rsp = mem.StackTop() - 16;  // leave room for push + sentinel
    *reinterpret_cast<u64*>(guest_rsp + 8) = kReturnSentinel;

    GuestState state{};
    state.rip = reinterpret_cast<u64>(mem.CodePtr());
    state.gpr[4] = reinterpret_cast<u64>(guest_rsp + 8);  // RSP at sentinel
    state.gpr[7] = 0x1234'5678'9abc'def0ULL;              // RDI

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
        0x48, 0xc7, 0xc0, 0x00, 0x01, 0x00, 0x00,  // mov rax, 0x100
        0x48, 0x83, 0xe8, 0x20,                    // sub rax, 0x20
        0xc3,                                       // ret
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
    state.gpr[0] = 0xdead'beef'cafe'babeULL;  // RAX starts non-zero

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
    state.gpr[7] = reinterpret_cast<u64>(data_slot);  // RDI -> data slot

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
    state.gpr[0] = 0xdead'beef'1234'5678ULL;            // RAX = value
    state.gpr[7] = reinterpret_cast<u64>(data_slot);    // RDI -> data slot

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
    state.gpr[5] = reinterpret_cast<u64>(data_slot) + 8;  // RBP -> slot + 8

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
        0x55,                                        // push rbp
        0x48, 0x89, 0xe5,                            // mov rbp, rsp
        0x48, 0x83, 0xec, 0x20,                      // sub rsp, 0x20
        0x48, 0x89, 0xf8,                            // mov rax, rdi
        0x48, 0x83, 0xc4, 0x20,                      // add rsp, 0x20
        0x5d,                                        // pop rbp
        0xc3,                                        // ret
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
    state.gpr[7] = 0x42;  // RDI

    Runtime rt;
    rt.Run(state);

    EXPECT_EQ(state.gpr[0], 0x42u)       << "RAX should hold the arg";
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
        0x48, 0xc7, 0xc0, 0x42, 0x00, 0x00, 0x00,
        0x48, 0xc7, 0xc3, 0x42, 0x00, 0x00, 0x00,
        0x48, 0x39, 0xd8,
        0xc3,
    };
    auto result = RunProgram(program, sizeof(program), mem);
    EXPECT_TRUE(result.state.rflags & kZF)  << "ZF should be set for equal";
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
        0x48, 0xc7, 0xc0, 0x00, 0x01, 0x00, 0x00,
        0x48, 0xc7, 0xc3, 0x42, 0x00, 0x00, 0x00,
        0x48, 0x39, 0xd8,
        0xc3,
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
        0x48, 0xc7, 0xc0, 0x10, 0x00, 0x00, 0x00,
        0x48, 0xc7, 0xc3, 0x00, 0x01, 0x00, 0x00,
        0x48, 0x39, 0xd8,
        0xc3,
    };
    auto result = RunProgram(program, sizeof(program), mem);
    EXPECT_FALSE(result.state.rflags & kZF) << "ZF should be clear";
    EXPECT_TRUE(result.state.rflags & kCF)  << "CF should be set (borrow)";
    EXPECT_TRUE(result.state.rflags & kSF)  << "SF should be set (negative result)";
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
        0x48, 0x31, 0xc0,          // xor rax, rax       (offset 0-2)
        0x74, 0x10,                // jz +0x10           (offset 3-4)
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
        0x48, 0xc7, 0xc0, 0x01, 0x00, 0x00, 0x00,  // mov rax, 1   (offsets 0-6)
        0x48, 0x85, 0xc0,                           // test rax,rax (offsets 7-9)
        0x74, 0x10,                                 // jz  +0x10    (offsets 10-11)
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
        0x48, 0xc7, 0xc0, 0x01, 0x00, 0x00, 0x00,  // mov rax, 1
        0x48, 0x85, 0xc0,                           // test rax,rax
        0x75, 0x10,                                 // jnz +0x10
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
        0x48, 0xc7, 0xc0, 0x01, 0x00, 0x00, 0x00,
        0x48, 0xc7, 0xc3, 0x02, 0x00, 0x00, 0x00,
        0x48, 0x39, 0xd8,
        0x7c, 0x10,
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
        0x48, 0xc7, 0xc7, 0x42, 0x00, 0x00, 0x00,  // mov rdi, 0x42       (0-6)
        0xe8, 0x04, 0x00, 0x00, 0x00,              // call +4 → offset 16 (7-11)
        0x48, 0x89, 0xc1,                           // mov rcx, rax        (12-14)
        0xc3,                                       // ret                 (15)
        // Callee
        0x48, 0x89, 0xf8,                           // mov rax, rdi        (16-18)
        0x48, 0x83, 0xc0, 0x01,                     // add rax, 1          (19-22)
        0xc3,                                       // ret                 (23)
    };
    const auto r = RunProgram(program, sizeof(program), mem);

    EXPECT_EQ(r.state.gpr[0], 0x43u) << "RAX should be 0x42 + 1 from callee";
    EXPECT_EQ(r.state.gpr[1], 0x43u) << "RCX should hold the captured return value";
    EXPECT_EQ(r.state.rip, kReturnSentinel)
        << "Caller's RET should pop the host-return sentinel";
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
        0x48, 0xc7, 0xc1, 0x00, 0x10, 0x00, 0x00,  // mov rcx, 0x1000
        0x48, 0x8d, 0x41, 0x05,                     // lea rax, [rcx+5]
        0xc3,                                       // ret
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
        0x48, 0xc7, 0xc1, 0x00, 0x10, 0x00, 0x00,  // mov rcx, 0x1000
        0x48, 0xc7, 0xc2, 0x03, 0x00, 0x00, 0x00,  // mov rdx, 3
        0x48, 0x8d, 0x44, 0xd1, 0x10,              // lea rax, [rcx+rdx*8+0x10]
        0xc3,                                       // ret
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
        0x89, 0xd8,                                                  // mov eax, ebx
        0xc3,                                                        // ret
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
        0xc7, 0x44, 0x24, 0xf8, 0x44, 0x33, 0x22, 0x11,  // mov dword [rsp-8], 0x11223344
        0x8b, 0x44, 0x24, 0xf8,                           // mov eax, [rsp-8]
        0xc3,                                              // ret
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
        0x48, 0xc7, 0xc0, 0x55, 0x00, 0x00, 0x00,        // mov rax, 0x55
        0x48, 0x89, 0x44, 0x24, 0xf8,                     // mov [rsp-8], rax
        0x48, 0xc7, 0xc1, 0x55, 0x00, 0x00, 0x00,        // mov rcx, 0x55
        0x48, 0x3b, 0x4c, 0x24, 0xf8,                     // cmp rcx, [rsp-8]
        0x74, 0x01,                                       // jz +1
        0xc3,                                             // ret (not taken)
        0x48, 0xc7, 0xc0, 0x99, 0x00, 0x00, 0x00,        // mov rax, 0x99
        0xc3,                                             // ret
    };
    const auto r = RunProgram(program, sizeof(program), mem);
    EXPECT_EQ(r.state.gpr[0], 0x99u)
        << "CMP rcx, [rsp-8] should have set ZF, JZ taken, rax = 0x99";
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
        0x48, 0x8d, 0x05, 0x09, 0x00, 0x00, 0x00,  // lea rax, [rip+9]
        0x48, 0x89, 0x44, 0x24, 0xf8,              // mov [rsp-8], rax
        0xff, 0x64, 0x24, 0xf8,                    // jmp qword [rsp-8]
        0x48, 0xc7, 0xc0, 0x77, 0x00, 0x00, 0x00,  // mov rax, 0x77
        0xc3,                                       // ret
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
        0x48, 0xb8, 0xff, 0xff, 0xff, 0xff, 0xff, 0xff, 0xff, 0xff,  // mov rax, -1
        0x31, 0xc0,                                                   // xor eax, eax
        0xc3,                                                         // ret
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
//   Let me use ebx instead: mov ebx, 0xFFFFFFFF → ebx = -1 (and rbx upper bits zeroed by 32-bit MOV).
//
//   mov ebx, 0xFFFFFFFF      bb ff ff ff ff             (0-4) → ebx = -1, rbx = 0x00000000FFFFFFFF
//   movsxd rax, ebx          48 63 c3                   (5-7) → rax = sext(ebx as int32) = -1
//   ret                      c3                          (8)
//
// Expected: rax = 0xFFFFFFFFFFFFFFFF (i.e., -1 as a 64-bit signed).
TEST_F(CpuRuntimeTest, Movsxd_RegReg_SignExtends) {
    const u8 program[] = {
        0xbb, 0xff, 0xff, 0xff, 0xff,  // mov ebx, 0xFFFFFFFF
        0x48, 0x63, 0xc3,              // movsxd rax, ebx
        0xc3,                          // ret
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
        0x48, 0x8d, 0x0d, 0x03, 0x00, 0x00, 0x00,  // lea rcx, [rip+3]  → offset 10
        0xff, 0xd1,                                 // call rcx
        0xc3,                                       // ret (caller's, → sentinel)
        0x48, 0xc7, 0xc0, 0x88, 0x00, 0x00, 0x00,  // callee: mov rax, 0x88
        0xc3,                                       // callee: ret  → caller's ret
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
static PS4_SYSV_ABI u64 HleBridgeTestFn_SumArgs(u64 a, u64 b, u64 c,
                                                u64 d, u64 e, u64 f) {
    // The constants below let us verify each arg landed in the right
    // SysV slot. If we got the marshal order wrong, the sum would
    // still match but checking specific values rules that out.
    return a * 1 + b * 10 + c * 100 + d * 1000 + e * 10000 + f * 100000;
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

    u8 program[] = {
        0x48, 0xc7, 0xc7, 0x01, 0x00, 0x00, 0x00,  // mov rdi, 1
        0x48, 0xc7, 0xc6, 0x02, 0x00, 0x00, 0x00,  // mov rsi, 2
        0x48, 0xc7, 0xc2, 0x03, 0x00, 0x00, 0x00,  // mov rdx, 3
        0x48, 0xc7, 0xc1, 0x04, 0x00, 0x00, 0x00,  // mov rcx, 4
        0x49, 0xc7, 0xc0, 0x05, 0x00, 0x00, 0x00,  // mov r8,  5
        0x49, 0xc7, 0xc1, 0x06, 0x00, 0x00, 0x00,  // mov r9,  6
        0x48, 0xb8, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, // mov rax, <imm64>
        0xff, 0xd0,                                 // call rax
        0xc3,                                       // ret
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
        0x48, 0xc7, 0xc0, 0x07, 0x00, 0x00, 0x00,  // mov rax, 7
        0x90,                                       // nop (1-byte)
        0x66, 0x90,                                 // nop (2-byte, 66 prefix)
        0x0f, 0x1f, 0x40, 0x00,                     // nop dword[rax+0] (4-byte)
        0xc3,                                       // ret
    };
    const auto r = RunProgram(program, sizeof(program), mem);
    EXPECT_EQ(r.state.gpr[0], 7u) << "MOV before NOPs should still be in rax";
    EXPECT_EQ(r.state.rip, kReturnSentinel);
}

// AND r64, r64 — bitwise AND.
TEST_F(CpuRuntimeTest, And64_RegReg_ComputesIntersection) {
    const u8 program[] = {
        0x48, 0xc7, 0xc0, 0x0f, 0x00, 0x00, 0x00,  // mov rax, 0x0F
        0x48, 0xc7, 0xc3, 0x33, 0x00, 0x00, 0x00,  // mov rbx, 0x33
        0x48, 0x21, 0xd8,                           // and rax, rbx
        0xc3,                                       // ret
    };
    const auto r = RunProgram(program, sizeof(program), mem);
    EXPECT_EQ(r.state.gpr[0], 0x03u);  // 0x0F & 0x33 = 0x03
    EXPECT_EQ(r.state.gpr[3], 0x33u);  // rbx unchanged
}

// OR r64, r64 — bitwise OR.
TEST_F(CpuRuntimeTest, Or64_RegReg_ComputesUnion) {
    const u8 program[] = {
        0x48, 0xc7, 0xc0, 0x0f, 0x00, 0x00, 0x00,  // mov rax, 0x0F
        0x48, 0xc7, 0xc3, 0x30, 0x00, 0x00, 0x00,  // mov rbx, 0x30
        0x48, 0x09, 0xd8,                           // or rax, rbx
        0xc3,                                       // ret
    };
    const auto r = RunProgram(program, sizeof(program), mem);
    EXPECT_EQ(r.state.gpr[0], 0x3Fu);  // 0x0F | 0x30 = 0x3F
}

// NOT r64 — bitwise complement; flags unchanged.
TEST_F(CpuRuntimeTest, Not_RegisterFlipsBits) {
    const u8 program[] = {
        0x48, 0xc7, 0xc0, 0xff, 0x00, 0x00, 0x00,  // mov rax, 0x00FF
        0x48, 0xf7, 0xd0,                           // not rax
        0xc3,                                       // ret
    };
    const auto r = RunProgram(program, sizeof(program), mem);
    EXPECT_EQ(r.state.gpr[0], 0xFFFFFFFFFFFFFF00ULL);  // ~0xFF (in 64-bit)
}

// NEG r64 — two's complement. Verifies the value AND that CF is
// set to 1 (because source was nonzero per x86 NEG semantics).
TEST_F(CpuRuntimeTest, Neg_NonZero_SetsCarryAndNegatesValue) {
    const u8 program[] = {
        0x48, 0xc7, 0xc0, 0x05, 0x00, 0x00, 0x00,  // mov rax, 5
        0x48, 0xf7, 0xd8,                           // neg rax
        0xc3,                                       // ret
    };
    const auto r = RunProgram(program, sizeof(program), mem);
    EXPECT_EQ(r.state.gpr[0], static_cast<u64>(-5LL));  // two's complement
    // CF (bit 0 of rflags) should be set because source was nonzero.
    EXPECT_EQ(r.state.rflags & 0x1ULL, 0x1ULL);
}

// NEG of zero leaves zero AND clears CF.
TEST_F(CpuRuntimeTest, Neg_Zero_ClearsCarry) {
    const u8 program[] = {
        0x48, 0xc7, 0xc0, 0x00, 0x00, 0x00, 0x00,  // mov rax, 0
        0x48, 0xf7, 0xd8,                           // neg rax
        0xc3,                                       // ret
    };
    const auto r = RunProgram(program, sizeof(program), mem);
    EXPECT_EQ(r.state.gpr[0], 0u);
    EXPECT_EQ(r.state.rflags & 0x1ULL, 0u);  // CF clear
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
        0x48, 0xc7, 0xc0, 0x05, 0x00, 0x00, 0x00,  // mov rax, 5
        0x48, 0xc7, 0xc3, 0x03, 0x00, 0x00, 0x00,  // mov rbx, 3
        0x48, 0x29, 0xd8,                           // sub rax, rbx → rax=2, CF=0
        0x48, 0xff, 0xc0,                           // inc rax
        0xc3,                                       // ret
    };
    const auto r = RunProgram(program, sizeof(program), mem);
    EXPECT_EQ(r.state.gpr[0], 3u);                  // 2 + 1 = 3
    EXPECT_EQ(r.state.rflags & 0x1ULL, 0u);         // CF still 0
}

TEST_F(CpuRuntimeTest, Inc_PreservesCarryFlag_SetCase) {
    const u8 program[] = {
        0x48, 0xc7, 0xc0, 0x03, 0x00, 0x00, 0x00,  // mov rax, 3
        0x48, 0xc7, 0xc3, 0x05, 0x00, 0x00, 0x00,  // mov rbx, 5
        0x48, 0x29, 0xd8,                           // sub rax, rbx → rax=-2 (huge), CF=1
        0x48, 0xff, 0xc0,                           // inc rax
        0xc3,                                       // ret
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
        0x48, 0xc7, 0xc0, 0x03, 0x00, 0x00, 0x00,  // mov rax, 3
        0x48, 0xc7, 0xc3, 0x05, 0x00, 0x00, 0x00,  // mov rbx, 5
        0x48, 0x29, 0xd8,                           // sub rax, rbx → CF=1
        0x48, 0xff, 0xc8,                           // dec rax
        0xc3,                                       // ret
    };
    const auto r = RunProgram(program, sizeof(program), mem);
    // -2 - 1 = -3.
    EXPECT_EQ(r.state.gpr[0], static_cast<u64>(-3LL));
    EXPECT_EQ(r.state.rflags & 0x1ULL, 0x1ULL);     // CF preserved
}

// DEC sets ZF when result is zero.
TEST_F(CpuRuntimeTest, Dec_FromOne_SetsZeroFlag) {
    const u8 program[] = {
        0x48, 0xc7, 0xc0, 0x01, 0x00, 0x00, 0x00,  // mov rax, 1
        0x48, 0xff, 0xc8,                           // dec rax
        0xc3,                                       // ret
    };
    const auto r = RunProgram(program, sizeof(program), mem);
    EXPECT_EQ(r.state.gpr[0], 0u);
    EXPECT_EQ(r.state.rflags & 0x40ULL, 0x40ULL);   // ZF bit 6
}

// MOVZX r64, r/m8 — zero-extend byte from register.
TEST_F(CpuRuntimeTest, Movzx_Reg8To64_ZeroExtends) {
    const u8 program[] = {
        // Set rax to 0xFFFFFFFFFFFFFFFF, then load BL = 0x7F (so AL
        // becomes 0x7F but upper bits of rax are all 1s). MOVZX
        // should produce 0x000000000000007F in rdx.
        0x48, 0xc7, 0xc0, 0xff, 0xff, 0xff, 0xff,  // mov rax, -1 (sign-extends to all-Fs)
        0x48, 0xc7, 0xc3, 0x7f, 0x00, 0x00, 0x00,  // mov rbx, 0x7F
        0x48, 0x0f, 0xb6, 0xd3,                     // movzx rdx, bl
        0xc3,                                       // ret
    };
    const auto r = RunProgram(program, sizeof(program), mem);
    EXPECT_EQ(r.state.gpr[2], 0x7Fu);  // rdx = zero-extended bl
    EXPECT_EQ(r.state.gpr[0], 0xFFFFFFFFFFFFFFFFULL);  // rax unchanged
}

// MOVZX r64, r/m16 — zero-extend word from register.
TEST_F(CpuRuntimeTest, Movzx_Reg16To64_ZeroExtends) {
    const u8 program[] = {
        0x48, 0xc7, 0xc3, 0x34, 0x12, 0x00, 0x00,  // mov rbx, 0x1234
        0x48, 0x0f, 0xb7, 0xc3,                     // movzx rax, bx
        0xc3,                                       // ret
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
        0x48, 0xc7, 0xc3, 0xa5, 0x00, 0x00, 0x00,  // mov rbx, 0xA5
        0x53,                                       // push rbx
        0x48, 0x0f, 0xb6, 0x04, 0x24,               // movzx rax, byte [rsp]
        0x48, 0x83, 0xc4, 0x08,                     // add rsp, 8 (rebalance)
        0xc3,                                       // ret
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
        0x48, 0xb8,
        0x00, 0x00, 0xfe, 0xca, 0xef, 0xbe, 0xad, 0xde,
        0x48, 0xc7, 0xc3, 0xcd, 0x00, 0x00, 0x00,   // mov rbx, 0xCD
        0x66, 0x0f, 0xb6, 0xc3,                      // movzx ax, bl
        0xc3,                                        // ret
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
        0x48, 0xb8,
        0x00, 0x00, 0xfe, 0xca, 0xef, 0xbe, 0xad, 0xde,
        0x48, 0xc7, 0xc3, 0x42, 0x00, 0x00, 0x00,   // mov rbx, 0x42
        0x0f, 0xb6, 0xc3,                            // movzx eax, bl
        0xc3,                                        // ret
    };
    const auto r = RunProgram(program, sizeof(program), mem);
    // Expected: full 64 bits = 0x42 (upper 32 zeroed by x86-64
    // write rule; low 32 = zero-extended byte = 0x42).
    EXPECT_EQ(r.state.gpr[0], 0x42u)
        << "MOVZX with 32-bit dst must zero-extend to 64 bits";
}

// CMOV taken: condition true, dst should be replaced with src.
//
// Set rax=10, rbx=20, then CMP rax with 0 (rax > 0 ⇒ NE), then
// CMOVNZ rax, rbx. Expect rax = 20 (the move was taken).
TEST_F(CpuRuntimeTest, CmovNz_ConditionTrue_TakesMove) {
    const u8 program[] = {
        0x48, 0xc7, 0xc0, 0x0a, 0x00, 0x00, 0x00,  // mov rax, 10
        0x48, 0xc7, 0xc3, 0x14, 0x00, 0x00, 0x00,  // mov rbx, 20
        0x48, 0x83, 0xf8, 0x00,                     // cmp rax, 0       (sets ZF=0)
        0x48, 0x0f, 0x45, 0xc3,                     // cmovnz rax, rbx
        0xc3,                                       // ret
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
        0x48, 0xc7, 0xc0, 0x00, 0x00, 0x00, 0x00,  // mov rax, 0
        0x48, 0xc7, 0xc3, 0x14, 0x00, 0x00, 0x00,  // mov rbx, 20
        0x48, 0x83, 0xf8, 0x00,                     // cmp rax, 0       (sets ZF=1)
        0x48, 0x0f, 0x45, 0xc3,                     // cmovnz rax, rbx
        0xc3,                                       // ret
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
        0x48, 0xc7, 0xc0, 0x05, 0x00, 0x00, 0x00,  // mov rax, 5
        0x48, 0xc7, 0xc3, 0x03, 0x00, 0x00, 0x00,  // mov rbx, 3
        0x48, 0x39, 0xd8,                           // cmp rax, rbx
        0x48, 0x0f, 0x4c, 0xc3,                     // cmovl rax, rbx
        0xc3,                                       // ret
    };
    const auto r = RunProgram(program, sizeof(program), mem);
    EXPECT_EQ(r.state.gpr[0], 5u);  // unchanged: rax not < rbx
}

// =============================================================================
// Tier-2 opcode batch: SHL/SHR/SAR (imm8 and CL forms),
// 8-bit MOV variants, 16-bit MOV variants.
// =============================================================================

// SHL rax, 4 — left shift by immediate. Multiplies by 16.
TEST_F(CpuRuntimeTest, Shl_Imm8_LeftShifts) {
    const u8 program[] = {
        0x48, 0xc7, 0xc0, 0x03, 0x00, 0x00, 0x00,  // mov rax, 3
        0x48, 0xc1, 0xe0, 0x04,                     // shl rax, 4
        0xc3,                                       // ret
    };
    const auto r = RunProgram(program, sizeof(program), mem);
    EXPECT_EQ(r.state.gpr[0], 3u << 4);  // = 48
}

// SHR rax, 1 — logical right shift by immediate.
TEST_F(CpuRuntimeTest, Shr_Imm8_RightShifts) {
    const u8 program[] = {
        0x48, 0xc7, 0xc0, 0x80, 0x00, 0x00, 0x00,  // mov rax, 128
        0x48, 0xd1, 0xe8,                           // shr rax, 1 (1-bit form)
        0xc3,                                       // ret
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
        0x48, 0xc7, 0xc0, 0xf0, 0xff, 0xff, 0xff,  // mov rax, -16
        0x48, 0xc1, 0xf8, 0x02,                     // sar rax, 2
        0xc3,                                       // ret
    };
    const auto r = RunProgram(program, sizeof(program), mem);
    EXPECT_EQ(r.state.gpr[0], static_cast<u64>(-4LL));
}

// SHL via CL — dynamic shift count.
TEST_F(CpuRuntimeTest, Shl_Cl_LeftShiftsByDynamicCount) {
    const u8 program[] = {
        0x48, 0xc7, 0xc0, 0x01, 0x00, 0x00, 0x00,  // mov rax, 1
        0x48, 0xc7, 0xc1, 0x05, 0x00, 0x00, 0x00,  // mov rcx, 5
        0x48, 0xd3, 0xe0,                           // shl rax, cl
        0xc3,                                       // ret
    };
    const auto r = RunProgram(program, sizeof(program), mem);
    EXPECT_EQ(r.state.gpr[0], 1u << 5);  // = 32
}

// Shift by zero must preserve all flags. We deliberately set up a
// known flag state via SUB, then SHL by 0, then verify CF is still
// set from the SUB.
//
// This is the most important shift-semantics test — it would fail
// with a naive implementation that always overwrites rflags.
TEST_F(CpuRuntimeTest, ShlByZero_PreservesFlags) {
    const u8 program[] = {
        0x48, 0xc7, 0xc0, 0x03, 0x00, 0x00, 0x00,  // mov rax, 3
        0x48, 0xc7, 0xc3, 0x05, 0x00, 0x00, 0x00,  // mov rbx, 5
        0x48, 0x29, 0xd8,                           // sub rax, rbx → CF=1
        0x48, 0xc7, 0xc1, 0x00, 0x00, 0x00, 0x00,  // mov rcx, 0    (shift count)
        0x48, 0xd3, 0xe0,                           // shl rax, cl   (no-op)
        0xc3,                                       // ret
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
        0x48, 0xb8,
        0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x80,
        0x48, 0xd1, 0xe0,                            // shl rax, 1
        0xc3,                                        // ret
    };
    const auto r = RunProgram(program, sizeof(program), mem);
    EXPECT_EQ(r.state.gpr[0], 0u);                  // bit shifted out, nothing left
    EXPECT_EQ(r.state.rflags & 0x1ULL, 0x1ULL);     // CF = 1 (the bit fell off)
}

// 8-bit MOV r,r — preserves upper bits of dst slot (like MOVZX r16).
TEST_F(CpuRuntimeTest, Mov8_RegReg_PreservesUpperBits) {
    const u8 program[] = {
        // mov rax, 0xDEADBEEFCAFE0000
        0x48, 0xb8,
        0x00, 0x00, 0xfe, 0xca, 0xef, 0xbe, 0xad, 0xde,
        0x48, 0xc7, 0xc3, 0x42, 0x00, 0x00, 0x00,    // mov rbx, 0x42
        0x88, 0xd8,                                   // mov al, bl
        0xc3,                                         // ret
    };
    const auto r = RunProgram(program, sizeof(program), mem);
    // Low byte replaced with 0x42; upper 56 preserved.
    EXPECT_EQ(r.state.gpr[0], 0xDEADBEEFCAFE0042ULL);
}

// 8-bit MOV r,imm.
TEST_F(CpuRuntimeTest, Mov8_RegImm_WritesLowByte) {
    const u8 program[] = {
        0x48, 0xb8,                                   // mov rax, 0xFFFFFFFFFFFFFFFF
        0xff, 0xff, 0xff, 0xff, 0xff, 0xff, 0xff, 0xff,
        0xb0, 0x7f,                                   // mov al, 0x7F
        0xc3,                                         // ret
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
        0x48, 0xc7, 0xc0, 0xa7, 0x00, 0x00, 0x00,
        // mov rbx, <addr>
        0x48, 0xbb, 0,0,0,0,0,0,0,0,
        // mov byte[rbx], al
        0x88, 0x03,
        // mov cl, byte[rbx]
        0x8a, 0x0b,
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
        0x48, 0xb8,
        0x00, 0x00, 0xfe, 0xca, 0xef, 0xbe, 0xad, 0xde,
        // mov rbx, 0x1234
        0x48, 0xc7, 0xc3, 0x34, 0x12, 0x00, 0x00,
        // mov ax, bx          (66 prefix + 89 d8)
        0x66, 0x89, 0xd8,
        0xc3,
    };
    const auto r = RunProgram(program, sizeof(program), mem);
    EXPECT_EQ(r.state.gpr[0], 0xDEADBEEFCAFE1234ULL);
}

// 16-bit MOV r,imm.
TEST_F(CpuRuntimeTest, Mov16_RegImm_WritesLowWord) {
    const u8 program[] = {
        0x48, 0xb8,                                   // mov rax, all-Fs
        0xff, 0xff, 0xff, 0xff, 0xff, 0xff, 0xff, 0xff,
        // mov ax, 0xCAFE      (66 b8 fe ca)
        0x66, 0xb8, 0xfe, 0xca,
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
        0x48, 0xc7, 0xc0, 0xde, 0xc0, 0x00, 0x00,
        // mov rbx, <addr>
        0x48, 0xbb, 0,0,0,0,0,0,0,0,
        // mov word[rbx], ax       66 89 03
        0x66, 0x89, 0x03,
        // mov cx, word[rbx]       66 8b 0b
        0x66, 0x8b, 0x0b,
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
        0x48, 0xc7, 0xc0, 0x07, 0x00, 0x00, 0x00,  // mov rax, 7
        0x48, 0xc7, 0xc3, 0x06, 0x00, 0x00, 0x00,  // mov rbx, 6
        0x48, 0x0f, 0xaf, 0xc3,                     // imul rax, rbx
        0xc3,                                       // ret
    };
    const auto r = RunProgram(program, sizeof(program), mem);
    EXPECT_EQ(r.state.gpr[0], 42u);
    EXPECT_EQ(r.state.gpr[3], 6u);  // rbx unchanged
}

// IMUL r64, r64, imm32 (3-op form) — compiler emits this for
// `x * constant` where constant fits in imm32.
TEST_F(CpuRuntimeTest, Imul_3Op_ImmediateConstant) {
    const u8 program[] = {
        0x48, 0xc7, 0xc3, 0x05, 0x00, 0x00, 0x00,  // mov rbx, 5
        // imul rax, rbx, 100   →   48 69 c3 64 00 00 00
        0x48, 0x69, 0xc3, 0x64, 0x00, 0x00, 0x00,
        0xc3,                                       // ret
    };
    const auto r = RunProgram(program, sizeof(program), mem);
    EXPECT_EQ(r.state.gpr[0], 500u);
    EXPECT_EQ(r.state.gpr[3], 5u);  // rbx unchanged
}

// IMUL r64, r64, imm8 (3-op form, short immediate).
TEST_F(CpuRuntimeTest, Imul_3Op_ImmediateImm8) {
    const u8 program[] = {
        0x48, 0xc7, 0xc3, 0x09, 0x00, 0x00, 0x00,  // mov rbx, 9
        // imul rax, rbx, 11   →   48 6b c3 0b
        0x48, 0x6b, 0xc3, 0x0b,
        0xc3,                                       // ret
    };
    const auto r = RunProgram(program, sizeof(program), mem);
    EXPECT_EQ(r.state.gpr[0], 99u);
}

// IMUL r/m64 (1-op form) — full 128-bit signed multiply.
// 2 * 3 = 6. Low half in RAX = 6, high half in RDX = 0.
TEST_F(CpuRuntimeTest, Imul_1Op_FullPrecisionMul) {
    const u8 program[] = {
        0x48, 0xc7, 0xc0, 0x02, 0x00, 0x00, 0x00,  // mov rax, 2
        0x48, 0xc7, 0xc3, 0x03, 0x00, 0x00, 0x00,  // mov rbx, 3
        0x48, 0xf7, 0xeb,                           // imul rbx   (rdx:rax = rax * rbx)
        0xc3,                                       // ret
    };
    const auto r = RunProgram(program, sizeof(program), mem);
    EXPECT_EQ(r.state.gpr[0], 6u);  // RAX = low half
    EXPECT_EQ(r.state.gpr[2], 0u);  // RDX = high half
}

// IMUL 1-op with a result that exceeds 64 bits — verifies the
// high half lands in RDX.
//
// Set RAX = 0x100000000, RBX = 0x100000000. Product = 0x10000000000000000
// which is exactly 2^64, so RAX = 0 and RDX = 1.
TEST_F(CpuRuntimeTest, Imul_1Op_OverflowsIntoHighHalf) {
    const u8 program[] = {
        // mov rax, 0x100000000
        0x48, 0xb8, 0x00, 0x00, 0x00, 0x00, 0x01, 0x00, 0x00, 0x00,
        // mov rbx, 0x100000000
        0x48, 0xbb, 0x00, 0x00, 0x00, 0x00, 0x01, 0x00, 0x00, 0x00,
        0x48, 0xf7, 0xeb,                           // imul rbx
        0xc3,
    };
    const auto r = RunProgram(program, sizeof(program), mem);
    EXPECT_EQ(r.state.gpr[0], 0u);  // RAX = low half = 0
    EXPECT_EQ(r.state.gpr[2], 1u);  // RDX = high half = 1
}

// IMUL with negative result — signed semantics.
TEST_F(CpuRuntimeTest, Imul_2Op_NegativeProduct) {
    const u8 program[] = {
        // mov rax, -3 (sign-extended imm32)
        0x48, 0xc7, 0xc0, 0xfd, 0xff, 0xff, 0xff,
        0x48, 0xc7, 0xc3, 0x07, 0x00, 0x00, 0x00,  // mov rbx, 7
        0x48, 0x0f, 0xaf, 0xc3,                     // imul rax, rbx
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
        0x48, 0xb8, 0x01, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0xf0,
        0x48, 0xc1, 0xc0, 0x04,                     // rol rax, 4
        0xc3,
    };
    const auto r = RunProgram(program, sizeof(program), mem);
    EXPECT_EQ(r.state.gpr[0], 0x000000000000001FULL);
}

// ROR r64, imm — rotate right by 4. 0x000000000000001F rotated right
// by 4 produces 0xF000000000000001 (low nibble wraps to high).
TEST_F(CpuRuntimeTest, Ror_Imm_RotatesRightWithWrap) {
    const u8 program[] = {
        0x48, 0xc7, 0xc0, 0x1f, 0x00, 0x00, 0x00,  // mov rax, 0x1F
        0x48, 0xc1, 0xc8, 0x04,                     // ror rax, 4
        0xc3,
    };
    const auto r = RunProgram(program, sizeof(program), mem);
    EXPECT_EQ(r.state.gpr[0], 0xF000000000000001ULL);
}

// ROL by zero must preserve flags (same rule as shifts). Set CF
// via SUB, then rotate by 0, verify CF intact.
TEST_F(CpuRuntimeTest, RolByZero_PreservesFlags) {
    const u8 program[] = {
        0x48, 0xc7, 0xc0, 0x03, 0x00, 0x00, 0x00,  // mov rax, 3
        0x48, 0xc7, 0xc3, 0x05, 0x00, 0x00, 0x00,  // mov rbx, 5
        0x48, 0x29, 0xd8,                           // sub rax, rbx → CF=1
        0x48, 0xc7, 0xc1, 0x00, 0x00, 0x00, 0x00,  // mov rcx, 0
        0x48, 0xd3, 0xc0,                           // rol rax, cl   (no-op)
        0xc3,
    };
    const auto r = RunProgram(program, sizeof(program), mem);
    EXPECT_EQ(r.state.gpr[0], static_cast<u64>(-2LL));
    EXPECT_EQ(r.state.rflags & 0x1ULL, 0x1ULL);  // CF still set
}

// CDQE — sign-extend EAX into RAX.
TEST_F(CpuRuntimeTest, Cdqe_SignExtendsNegativeEax) {
    const u8 program[] = {
        // mov eax, -1   →   b8 ff ff ff ff (5 bytes — 32-bit imm)
        0xb8, 0xff, 0xff, 0xff, 0xff,
        0x48, 0x98,                                 // cdqe
        0xc3,
    };
    const auto r = RunProgram(program, sizeof(program), mem);
    EXPECT_EQ(r.state.gpr[0], 0xFFFFFFFFFFFFFFFFULL);
}

// CDQE with non-negative EAX — upper 32 should be zero.
TEST_F(CpuRuntimeTest, Cdqe_ZeroExtendsPositiveEax) {
    const u8 program[] = {
        // mov eax, 0x7FFFFFFF (largest positive s32)
        0xb8, 0xff, 0xff, 0xff, 0x7f,
        0x48, 0x98,                                 // cdqe
        0xc3,
    };
    const auto r = RunProgram(program, sizeof(program), mem);
    EXPECT_EQ(r.state.gpr[0], 0x000000007FFFFFFFULL);
}

// CDQ — sign-extend EAX into EDX. Negative case.
TEST_F(CpuRuntimeTest, Cdq_NegativeEax_FillsEdxWithOnes) {
    const u8 program[] = {
        0xb8, 0xff, 0xff, 0xff, 0xff,               // mov eax, -1
        0x99,                                       // cdq
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
        0x48, 0xba, 0xbe, 0xba, 0xfe, 0xca, 0xef, 0xbe, 0xad, 0xde,
        0xb8, 0x42, 0x00, 0x00, 0x00,               // mov eax, 0x42
        0x99,                                       // cdq
        0xc3,
    };
    const auto r = RunProgram(program, sizeof(program), mem);
    EXPECT_EQ(r.state.gpr[2], 0u);  // RDX fully cleared
}

// CQO — sign-extend RAX into RDX.
TEST_F(CpuRuntimeTest, Cqo_NegativeRax_FillsRdxWithOnes) {
    const u8 program[] = {
        // mov rax, -1 (sign-extended imm32)
        0x48, 0xc7, 0xc0, 0xff, 0xff, 0xff, 0xff,
        0x48, 0x99,                                 // cqo
        0xc3,
    };
    const auto r = RunProgram(program, sizeof(program), mem);
    EXPECT_EQ(r.state.gpr[2], 0xFFFFFFFFFFFFFFFFULL);
}

// STC — set carry flag.
TEST_F(CpuRuntimeTest, Stc_SetsCarryFlag) {
    const u8 program[] = {
        0xf9,                                       // stc
        0xc3,
    };
    const auto r = RunProgram(program, sizeof(program), mem);
    EXPECT_EQ(r.state.rflags & 0x1ULL, 0x1ULL);
}

// CLC — clear carry flag. First set CF via SUB, then CLC, verify cleared.
TEST_F(CpuRuntimeTest, Clc_ClearsCarryFlag) {
    const u8 program[] = {
        0x48, 0xc7, 0xc0, 0x03, 0x00, 0x00, 0x00,  // mov rax, 3
        0x48, 0xc7, 0xc3, 0x05, 0x00, 0x00, 0x00,  // mov rbx, 5
        0x48, 0x29, 0xd8,                           // sub rax, rbx → CF=1
        0xf8,                                       // clc
        0xc3,
    };
    const auto r = RunProgram(program, sizeof(program), mem);
    EXPECT_EQ(r.state.rflags & 0x1ULL, 0u);
}

// CMC — complement carry flag. SUB sets CF=1, CMC flips to 0.
TEST_F(CpuRuntimeTest, Cmc_TogglesCarryFromOneToZero) {
    const u8 program[] = {
        0x48, 0xc7, 0xc0, 0x03, 0x00, 0x00, 0x00,  // mov rax, 3
        0x48, 0xc7, 0xc3, 0x05, 0x00, 0x00, 0x00,  // mov rbx, 5
        0x48, 0x29, 0xd8,                           // sub rax, rbx → CF=1
        0xf5,                                       // cmc
        0xc3,
    };
    const auto r = RunProgram(program, sizeof(program), mem);
    EXPECT_EQ(r.state.rflags & 0x1ULL, 0u);
}

// CMC — from 0 to 1.
TEST_F(CpuRuntimeTest, Cmc_TogglesCarryFromZeroToOne) {
    const u8 program[] = {
        0xf8,                                       // clc → CF=0
        0xf5,                                       // cmc → CF=1
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
        0x50,                    // push rax (save sentinel as saved rbp)
        0x48, 0x89, 0xe5,        // mov rbp, rsp (set up frame pointer)
        // ... no body needed ...
        0xc9,                    // leave (mov rsp, rbp; pop rbp)
        0xc3,                    // ret  (pops host sentinel)
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
        0x48, 0xb8, 0xff, 0xff, 0xff, 0xff, 0xff, 0xff, 0xff, 0xff,
        // mov rbx, 1
        0x48, 0xc7, 0xc3, 0x01, 0x00, 0x00, 0x00,
        // mov rcx, 1
        0x48, 0xc7, 0xc1, 0x01, 0x00, 0x00, 0x00,
        // mov rdx, 2
        0x48, 0xc7, 0xc2, 0x02, 0x00, 0x00, 0x00,
        // add rax, rcx        (sets CF=1)
        0x48, 0x01, 0xc8,
        // adc rbx, rdx        (reads CF, adds it)
        0x48, 0x11, 0xd3,
        0xc3,
    };
    const auto r = RunProgram(program, sizeof(program), mem);
    EXPECT_EQ(r.state.gpr[0], 0u)                   << "low half wraps to 0";
    EXPECT_EQ(r.state.gpr[3], 4u)                   << "high half: 1 + 2 + CF(1) = 4";
}

// SBB chain — the inverse, multi-precision subtraction.
//
//   { lo = 0x0000000000000000, hi = 0x0000000000000004 }
// - { lo = 0x0000000000000001, hi = 0x0000000000000002 }
// = { lo = 0xFFFFFFFFFFFFFFFF, hi = 0x0000000000000001 }   (CF=1 borrowed)
TEST_F(CpuRuntimeTest, Sbb_128BitSubChain_BorrowsCorrectly) {
    const u8 program[] = {
        0x48, 0xc7, 0xc0, 0x00, 0x00, 0x00, 0x00,   // mov rax, 0
        0x48, 0xc7, 0xc3, 0x04, 0x00, 0x00, 0x00,   // mov rbx, 4
        0x48, 0xc7, 0xc1, 0x01, 0x00, 0x00, 0x00,   // mov rcx, 1
        0x48, 0xc7, 0xc2, 0x02, 0x00, 0x00, 0x00,   // mov rdx, 2
        0x48, 0x29, 0xc8,                            // sub rax, rcx  (sets CF=1)
        0x48, 0x19, 0xd3,                            // sbb rbx, rdx  (reads CF)
        0xc3,
    };
    const auto r = RunProgram(program, sizeof(program), mem);
    EXPECT_EQ(r.state.gpr[0], 0xFFFFFFFFFFFFFFFFULL) << "low half borrows to all-Fs";
    EXPECT_EQ(r.state.gpr[3], 1u)                   << "high half: 4 - 2 - CF(1) = 1";
}

// ADC reads CF=0 case — verify chain starts clean from a 0-flag state.
TEST_F(CpuRuntimeTest, Adc_WithCarryClear_BehavesLikeAdd) {
    const u8 program[] = {
        0x48, 0xc7, 0xc0, 0x05, 0x00, 0x00, 0x00,   // mov rax, 5
        0x48, 0xc7, 0xc3, 0x03, 0x00, 0x00, 0x00,   // mov rbx, 3
        0xf8,                                        // clc → CF=0
        0x48, 0x11, 0xd8,                            // adc rax, rbx  (5 + 3 + 0 = 8)
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
        0xb0, 0xff,
        // mov bl, 0x01
        0xb3, 0x01,
        // add al, bl        (00 d8)
        0x00, 0xd8,
        0xc3,
    };
    const auto r = RunProgram(program, sizeof(program), mem);
    EXPECT_EQ(r.state.gpr[0] & 0xFFu, 0u)           << "0xFF + 0x01 wraps to 0";
    EXPECT_EQ(r.state.rflags & 0x1ULL, 0x1ULL)      << "CF set by byte-overflow";
    EXPECT_EQ(r.state.rflags & 0x40ULL, 0x40ULL)    << "ZF set (result is 0)";
}

// 8-bit ADD — preserves upper 56 bits of dst's slot. Set rax to a
// sentinel, then do an 8-bit add to al, and check the upper bits
// of rax are intact.
TEST_F(CpuRuntimeTest, Add8_PreservesUpperBits) {
    const u8 program[] = {
        // mov rax, 0xDEADBEEFCAFE0010
        0x48, 0xb8, 0x10, 0x00, 0xfe, 0xca, 0xef, 0xbe, 0xad, 0xde,
        // add al, 5
        0x04, 0x05,
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
        0xb0, 0x05,
        0xb3, 0x05,
        // cmp al, bl     (38 d8)
        0x38, 0xd8,
        0xc3,
    };
    const auto r = RunProgram(program, sizeof(program), mem);
    EXPECT_EQ(r.state.gpr[0] & 0xFFu, 5u)           << "CMP must not write al";
    EXPECT_EQ(r.state.rflags & 0x40ULL, 0x40ULL)    << "ZF set (al == bl)";
}

// 16-bit SUB — width-correct flag semantics. 0x8000 (highest s16
// negative) - 1 = 0x7FFF, with OF set because we crossed the
// signed boundary.
TEST_F(CpuRuntimeTest, Sub16_SignedOverflow_SetsOverflowFlag) {
    const u8 program[] = {
        // mov ax, 0x8000     66 b8 00 80
        0x66, 0xb8, 0x00, 0x80,
        // mov bx, 1          66 bb 01 00
        0x66, 0xbb, 0x01, 0x00,
        // sub ax, bx         66 29 d8
        0x66, 0x29, 0xd8,
        0xc3,
    };
    const auto r = RunProgram(program, sizeof(program), mem);
    EXPECT_EQ(r.state.gpr[0] & 0xFFFFu, 0x7FFFu);
    // OF is bit 11 of rflags.
    EXPECT_EQ(r.state.rflags & 0x800ULL, 0x800ULL)
        << "Crossing the s16 sign boundary sets OF";
}

// 16-bit ADD — preserves upper 48 bits.
TEST_F(CpuRuntimeTest, Add16_PreservesUpper48Bits) {
    const u8 program[] = {
        // mov rax, 0xDEADBEEFCAFE0100
        0x48, 0xb8, 0x00, 0x01, 0xfe, 0xca, 0xef, 0xbe, 0xad, 0xde,
        // add ax, 5          66 83 c0 05
        0x66, 0x83, 0xc0, 0x05,
        0xc3,
    };
    const auto r = RunProgram(program, sizeof(program), mem);
    EXPECT_EQ(r.state.gpr[0], 0xDEADBEEFCAFE0105ULL);
}

} // namespace
} // namespace Core::Runtime
