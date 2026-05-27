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
    // INC is still unsupported in the current lifter slice. Use it
    // to verify the unsupported-exit path still works. If/when INC
    // becomes supported, replace with another not-yet-supported
    // instruction.
    const u8 program[] = {
        0x48, 0xc7, 0xc0, 0x01, 0x00, 0x00, 0x00,  // mov rax, 1
        0x48, 0xff, 0xc3,                           // inc rbx       (unsupported)
        0xc3,                                       // ret  (unreached)
    };
    const auto r = RunProgram(program, sizeof(program), mem);

    // RIP should point to the INC instruction (at offset 7).
    EXPECT_EQ(r.state.rip, r.program_base + 7);
    EXPECT_EQ(r.state.exit_reason,
              static_cast<u32>(ExitReason::UnsupportedInstruction));
    // MOV before INC should have executed.
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

} // namespace
} // namespace Core::Runtime
