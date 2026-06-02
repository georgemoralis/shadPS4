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
// RUNTIME / INFRASTRUCTURE TESTS -- dispatcher, block & code cache, HLE bridge, callbacks, fault diagnostics, exit reasons.
// ========================================================================


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

// ---- Additional HLE bridge test host functions (new coverage) -------------

// Eight double args, all passed in xmm0..xmm7 (SysV SSE pool). Position-
// weighted so a misrouted lane is obvious. Used to exercise the bridge's
// full f0..f7 marshalling (load_xmm_low(0..7)); the prior suite only set
// xmm0/xmm1.
static PS4_SYSV_ABI double HleBridgeTestFn_EightDoubles(double a, double b, double c, double d,
                                                        double e, double f, double g, double h) {
    return a * 1.0 + b * 10.0 + c * 100.0 + d * 1000.0 +
           e * 10000.0 + f * 100000.0 + g * 1000000.0 + h * 10000000.0;
}

// Fourteen integer args: 6 in registers (rdi,rsi,rdx,rcx,r8,r9) and 8 spilled
// to the guest stack at [rsp+8 .. rsp+64] (args 7..14). Exercises the bridge's
// full s0..s7 stack-marshalling; the prior suite only reached 2 stack args.
// Multipliers are powers of two so the 64-bit sum stays exact and a wrong slot
// is identifiable from the bit that is off.
static PS4_SYSV_ABI u64 HleBridgeTestFn_FourteenArgs(u64 a, u64 b, u64 c, u64 d, u64 e, u64 f,
                                                     u64 g, u64 h, u64 i, u64 j, u64 k, u64 l,
                                                     u64 m, u64 n) {
    return (a << 0)  | (b << 1)  | (c << 2)  | (d << 3)  | (e << 4)  | (f << 5)  | (g << 6)  |
           (h << 7)  | (i << 8)  | (j << 9)  | (k << 10) | (l << 11) | (m << 12) | (n << 13);
}

// Returns a fixed high-bit-set 64-bit value regardless of args. Verifies the
// integer return round-trips through ret.rax with no truncation or sign games,
// and (on AArch64) that the x0 capture is byte-exact for a value with bit 63 set.
static PS4_SYSV_ABI u64 HleBridgeTestFn_ReturnsHighBits(u64) {
    return 0xFEEDFACECAFEBEEFULL;
}

// Pure integer function whose result is in rax; it does NOT compute or return
// a double, so the host's d0/xmm0 on return is undefined. The guest must read
// rax (gpr[0]); the meaningless xmm0 write-back must not corrupt the int path.
static PS4_SYSV_ABI u64 HleBridgeTestFn_IntReturnIgnoresXmm(u64 a, u64 b) {
    return a + b;
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
// Code-cache flush/recycle invariant — the mechanism behind the
// dispatcher's "flush and retry on full cache" path. When the bump
// allocator is exhausted (Allocate returns nullptr), flushing the code
// cache (reset bump pointer) together with clearing the block cache
// (drop all host pointers, which would otherwise dangle into recycled
// memory) must make allocation succeed again. This is what lets the
// game keep running once it has compiled more code than the cache
// holds at once — instead of dying at "code cache full".
// ============================================================================

TEST(CodeCacheFlush, RecycleAfterFull) {
    // Small cache so we can exhaust it quickly. 64 KB.
    CodeCache cache(64 * 1024);
    BlockCache blocks;

    // Allocate fixed-size chunks until the cache is full, recording a
    // fake block-cache mapping for each (as the dispatcher would).
    const u64 chunk = 4096;
    u64 fake_rip = 0x40000;
    int allocated = 0;
    for (;;) {
        u8* p = cache.Allocate(chunk);
        if (p == nullptr) break;
        blocks.Insert(fake_rip, p);
        fake_rip += 0x100;
        ++allocated;
        ASSERT_LT(allocated, 100000) << "cache never filled (unexpected)";
    }
    EXPECT_GT(allocated, 0) << "should have fit at least one chunk";
    EXPECT_GT(cache.Used(), 0u) << "cache shows usage before flush";
    // Confirm it is genuinely full: another allocation fails.
    EXPECT_EQ(cache.Allocate(chunk), nullptr) << "cache is full";

    // The recycle: clear the block cache, flush the code cache.
    blocks.Clear();
    cache.Flush();

    // After flush, allocation succeeds again — this is the retry the
    // dispatcher performs to keep executing.
    u8* after = cache.Allocate(chunk);
    EXPECT_NE(after, nullptr) << "allocation succeeds after flush+clear";
    EXPECT_LE(cache.Used(), cache.Capacity()) << "usage sane after recycle";

    // The block cache is empty after the clear: the old (now-dangling)
    // mappings are gone, so a lookup of a previously-inserted rip misses.
    EXPECT_EQ(blocks.Lookup(0x40000), nullptr)
        << "stale block-cache entry must not survive a flush";
}

// A single allocation larger than the whole cache fails even when the
// cache is empty — this is the genuine-failure case the dispatcher
// distinguishes from a capacity recycle (it does NOT loop on it).
TEST(CodeCacheFlush, OversizeAllocationFailsEvenWhenEmpty) {
    CodeCache cache(64 * 1024);
    EXPECT_EQ(cache.Used(), 0u) << "fresh cache is empty";
    // Request more than the entire capacity.
    u8* p = cache.Allocate(128 * 1024);
    EXPECT_EQ(p, nullptr) << "an over-capacity block cannot be allocated";
    // Used() is still ~0, which is exactly the signal the dispatcher
    // uses to decline flushing (flushing an empty cache won't help).
    EXPECT_EQ(cache.Used(), 0u) << "failed oversize alloc left cache empty";
}

// ============================================================================
// DescribeFaultContext — the async-signal-safe crash-diagnostic helper
// the signal/SEH handler consults to turn a raw host fault address into
// CPU-runtime context (was this thread in the runtime? what guest RIP?
// was the fault inside JIT code?). We can only observe it from outside a
// live fault here, but the out-of-runtime behavior and the null-address
// path are exactly the safety-critical defaults, so we pin those.
// ============================================================================

// Outside any Run(), the calling thread has no active GuestState, so
// the context reports not-in-runtime with zeroed fields. This is the
// path taken when a fault happens in pure host code (the common case
// for the audio/pad/user-service HLE threads).
TEST(FaultDiagnostics, OutsideRuntime_ReportsNotInRuntime) {
    const auto ctx = Core::Runtime::DescribeFaultContext(
        reinterpret_cast<const void*>(0x1234ull));
    EXPECT_FALSE(ctx.in_runtime) << "no active Run() on this thread";
    EXPECT_EQ(ctx.guest_rip, 0ull) << "no guest RIP when not in runtime";
    EXPECT_EQ(ctx.guest_exit_reason, 0u);
    EXPECT_FALSE(ctx.have_gprs) << "no GPR snapshot when not in runtime";
    EXPECT_EQ(ctx.guest_gpr[0], 0ull) << "GPR array zeroed by default";
    EXPECT_EQ(ctx.guest_gpr[15], 0ull);
    EXPECT_FALSE(ctx.faulting_insn_decoded) << "no instruction decoded when not in runtime";
    EXPECT_EQ(ctx.faulting_mnemonic, 0u) << "mnemonic id zero (INVALID) by default";
    EXPECT_EQ(ctx.faulting_insn_length, 0) << "insn length zero by default";
}

// A null host address must never be treated as inside JIT code (the
// Contains range check must not fire on null), regardless of runtime
// state. Defensive: the caller may not always have a code address.
TEST(FaultDiagnostics, NullHostAddr_NotInJitCode) {
    const auto ctx = Core::Runtime::DescribeFaultContext(nullptr);
    EXPECT_FALSE(ctx.in_jit_code) << "null address is never inside the code cache";
}

// A host address far outside any plausible code-cache mapping (the
// kind of wild ~1 TiB address a bad-pointer dereference produces) is
// reported as NOT in JIT code — i.e. the fault is elsewhere (HLE or a
// bad guest-pointer deref), which is the correct triage signal.
TEST(FaultDiagnostics, WildAddr_NotInJitCode) {
    const auto ctx = Core::Runtime::DescribeFaultContext(
        reinterpret_cast<const void*>(0x1030afb0035ull));
    EXPECT_FALSE(ctx.in_jit_code)
        << "a wild far-out address is not inside the JIT code cache";
}

// ============================================================================
// RUNTIME-LAYER BEHAVIOR TESTS
//
// These exercise the runtime *machinery* -- the dispatcher loop, block/code
// caches, exit-reason plumbing, and the (compiled-out) diagnostics hooks --
// rather than individual lifted opcodes. They drive Runtime directly (instead
// of the RunProgram helper) where they need to introspect cache state after a
// run.
// ============================================================================

namespace {
// Like RunProgram, but exposes the Runtime so the test can inspect its caches
// after execution. Returns nothing; caller holds the Runtime and GuestState.
void RunOn(Runtime& rt, GuestMemory& mem, const u8* program, size_t n,
           GuestState& out_state) {
    std::memcpy(mem.CodePtr(), program, n);
    u8* guest_rsp = mem.StackTop() - 8;
    *reinterpret_cast<u64*>(guest_rsp) = kReturnSentinel;
    out_state = GuestState{};
    out_state.rip = reinterpret_cast<u64>(mem.CodePtr());
    out_state.gpr[4] = reinterpret_cast<u64>(guest_rsp);
    rt.Run(out_state);
}
} // namespace

// ----------------------------------------------------------------------------
// Multi-block dispatch: a forward jump splits execution across two distinct
// guest RIPs. The dispatcher must compile and cache *both* blocks and chain
// from the first into the second. Observable via the final register state and
// a clean BlockEnd exit.
//   xor rax,rax; jmp +N; <dead>; target: mov rax,0x2a; ret
// ----------------------------------------------------------------------------
TEST_F(CpuRuntimeTest, Dispatch_ForwardJump_ChainsTwoBlocks) {
    const u8 program[] = {
        0x48, 0x31, 0xc0,             // xor rax, rax           (block A, off 0)
        0xeb, 0x07,                   // jmp +7  -> off 0x0c    (ends block A)
        0x48, 0xc7, 0xc3, 0x09,0,0,0, // mov rbx, 9 (dead code, jumped over)
        0x48, 0xc7, 0xc0, 0x2a,0,0,0, // target: mov rax, 0x2a  (block B, off 0x0c)
        0xc3,                         // ret
    };
    const auto r = RunProgram(program, sizeof(program), mem);
    EXPECT_EQ(r.state.gpr[0], 0x2aULL) << "block B ran";
    EXPECT_EQ(r.state.gpr[3], 0ULL)    << "dead code between blocks did not run";
    EXPECT_EQ(r.state.rip, kReturnSentinel);
    EXPECT_EQ(r.state.exit_reason, static_cast<u32>(ExitReason::BlockEnd));
}

// ----------------------------------------------------------------------------
// Block-cache reuse: a loop re-enters the same block RIP three times. The
// block is compiled once and served from the cache on the 2nd and 3rd entry.
// Correctness of the accumulated result proves the cached block runs each time.
//   xor rax,rax; mov rcx,3; L: add rax,rcx; dec rcx; jnz L; ret
//   -> rax = 3 + 2 + 1 = 6
// ----------------------------------------------------------------------------
TEST_F(CpuRuntimeTest, Dispatch_Loop_ReusesCachedBlock) {
    const u8 program[] = {
        0x48, 0x31, 0xc0,             // xor rax, rax
        0x48, 0xc7, 0xc1, 0x03,0,0,0, // mov rcx, 3
        0x48, 0x01, 0xc8,             // L: add rax, rcx
        0x48, 0xff, 0xc9,             // dec rcx
        0x75, 0xf8,                   // jnz L
        0xc3,                         // ret
    };
    const auto r = RunProgram(program, sizeof(program), mem);
    EXPECT_EQ(r.state.gpr[0], 6ULL) << "3+2+1 accumulated across loop iterations";
    EXPECT_EQ(r.state.gpr[1], 0ULL) << "counter decremented to 0";
    EXPECT_EQ(r.state.exit_reason, static_cast<u32>(ExitReason::BlockEnd));
}

// ----------------------------------------------------------------------------
// The block cache populates and serves a compiled block: after a run, the
// entry RIP resolves in the block cache, and that host pointer is inside the
// code cache.
// ----------------------------------------------------------------------------
TEST_F(CpuRuntimeTest, BlockCache_PopulatedAfterRun_AndInsideCodeCache) {
    const u8 program[] = {
        0x48, 0xc7, 0xc0, 0x07,0,0,0, // mov rax, 7
        0xc3,                         // ret
    };
    Runtime rt;
    GuestState st{};
    RunOn(rt, mem, program, sizeof(program), st);
    ASSERT_EQ(st.gpr[0], 7ULL);

    const u64 entry_rip = reinterpret_cast<u64>(mem.CodePtr());
    void* host = const_cast<BlockCache&>(rt.GetBlockCache()).Lookup(entry_rip);
    ASSERT_NE(host, nullptr) << "entry block must be cached after the run";
    EXPECT_TRUE(rt.GetCodeCache().Contains(host))
        << "the cached block's host code lives in the code cache";
}

// A second Run() of the same program on the SAME Runtime resolves the entry to
// the identical host pointer -- i.e. it is served from cache, not recompiled
// to a new location.
TEST_F(CpuRuntimeTest, BlockCache_SecondRun_ServesSameHostPointer) {
    const u8 program[] = {
        0x48, 0xc7, 0xc0, 0x07,0,0,0, // mov rax, 7
        0xc3,
    };
    Runtime rt;
    GuestState st1{}, st2{};
    RunOn(rt, mem, program, sizeof(program), st1);
    const u64 entry_rip = reinterpret_cast<u64>(mem.CodePtr());
    void* host1 = const_cast<BlockCache&>(rt.GetBlockCache()).Lookup(entry_rip);

    RunOn(rt, mem, program, sizeof(program), st2);
    void* host2 = const_cast<BlockCache&>(rt.GetBlockCache()).Lookup(entry_rip);

    ASSERT_NE(host1, nullptr);
    EXPECT_EQ(host1, host2) << "same RIP must map to the same cached host block";
    EXPECT_EQ(st1.gpr[0], st2.gpr[0]) << "and produce the same result";
}

// ----------------------------------------------------------------------------
// Exit-reason plumbing: each reachable terminal reason surfaces in
// state.exit_reason.
// ----------------------------------------------------------------------------

// BlockEnd: a normal RET through the return sentinel.
TEST_F(CpuRuntimeTest, ExitReason_BlockEnd_OnReturnToSentinel) {
    const u8 program[] = {0x48, 0x31, 0xc0, 0xc3}; // xor rax,rax; ret
    const auto r = RunProgram(program, sizeof(program), mem);
    EXPECT_EQ(r.state.exit_reason, static_cast<u32>(ExitReason::BlockEnd));
    EXPECT_EQ(r.state.rip, kReturnSentinel);
}

// UnsupportedInstruction: an instruction the lifter does not handle stops the
// run with the offending RIP preserved. (FS-segment-override load -- verified
// elsewhere as unsupported.)
TEST_F(CpuRuntimeTest, ExitReason_Unsupported_StopsWithRipAtFault) {
    const u8 program[] = {
        0x48, 0xc7, 0xc0, 0x05,0,0,0, // mov rax, 5  (off 0)
        0x64, 0x48, 0x8b, 0x09,        // mov rcx, fs:[rcx]  (off 7, unsupported)
        0xc3,
    };
    const auto r = RunProgram(program, sizeof(program), mem);
    EXPECT_EQ(r.state.exit_reason,
              static_cast<u32>(ExitReason::UnsupportedInstruction));
    EXPECT_EQ(r.state.rip, r.program_base + 7) << "rip at the unsupported insn";
    EXPECT_EQ(r.state.gpr[0], 5ULL) << "instructions before it executed";
}

// ----------------------------------------------------------------------------
// Cross-run cache sharing on one Runtime: two *different* programs each get
// their own cached entry block, and both entries coexist (the cache is not
// clobbered by the second compile).
// ----------------------------------------------------------------------------
TEST_F(CpuRuntimeTest, BlockCache_TwoProgramsCoexist) {
    Runtime rt;

    // Program 1 at CodePtr(): mov rax, 0x11; ret
    const u8 p1[] = {0x48, 0xc7, 0xc0, 0x11,0,0,0, 0xc3};
    GuestState s1{};
    RunOn(rt, mem, p1, sizeof(p1), s1);
    const u64 rip1 = reinterpret_cast<u64>(mem.CodePtr());
    void* h1 = const_cast<BlockCache&>(rt.GetBlockCache()).Lookup(rip1);

    // Program 2 at CodePtr()+0x40 (distinct RIP): mov rax, 0x22; ret
    u8* p2_base = mem.CodePtr() + 0x40;
    const u8 p2[] = {0x48, 0xc7, 0xc0, 0x22,0,0,0, 0xc3};
    std::memcpy(p2_base, p2, sizeof(p2));
    u8* rsp = mem.StackTop() - 8;
    *reinterpret_cast<u64*>(rsp) = kReturnSentinel;
    GuestState s2{};
    s2.rip = reinterpret_cast<u64>(p2_base);
    s2.gpr[4] = reinterpret_cast<u64>(rsp);
    rt.Run(s2);

    const u64 rip2 = reinterpret_cast<u64>(p2_base);
    void* h2 = const_cast<BlockCache&>(rt.GetBlockCache()).Lookup(rip2);

    EXPECT_EQ(s1.gpr[0], 0x11ULL);
    EXPECT_EQ(s2.gpr[0], 0x22ULL);
    ASSERT_NE(h1, nullptr);
    ASSERT_NE(h2, nullptr);
    EXPECT_NE(h1, h2) << "distinct programs get distinct cached blocks";
    // Program 1's block is still resolvable after program 2 compiled.
    EXPECT_EQ(const_cast<BlockCache&>(rt.GetBlockCache()).Lookup(rip1), h1)
        << "first block survived the second compile";
}

// ----------------------------------------------------------------------------
// CallGuestSimple round-trips integer args (RDI..R9 per SysV) into the guest
// and returns RAX. Guest: lea rax,[rdi+rsi]; ret  (rax = a0 + a1).
// ----------------------------------------------------------------------------
TEST_F(CpuRuntimeTest, CallGuestSimple_SumsTwoArgsViaLea) {
    const u8 program[] = {
        0x48, 0x8d, 0x04, 0x37, // lea rax, [rdi + rsi]
        0xc3,
    };
    std::memcpy(mem.CodePtr(), program, sizeof(program));
    Runtime rt;
    const u64 fn = reinterpret_cast<u64>(mem.CodePtr());
    const u64 result = rt.CallGuestSimple(fn, mem.StackTop(),
                                          /*a0=*/40, /*a1=*/2);
    EXPECT_EQ(result, 42ULL) << "RDI(40) + RSI(2) via lea, returned in RAX";
}

// ============================================================================
// HLE bridge: extended marshalling coverage.
//
// The original bridge suite exercised up to xmm1 for float args and up to 2
// stack args. CallHostFromGuest actually marshals 8 float args (f0..f7 from
// state.ymm[i*4]) and 8 stack args (s0..s7 from [guest_rsp+8..+64]), and
// returns both an integer (ret.rax -> gpr[0]) and a double (ret.xmm0 ->
// ymm[0] low 64). These tests cover the previously-untested far slots and the
// return-value edge cases, including the AArch64 x0/d0 capture path that this
// series rewrote.
// ============================================================================

// --- Group 1: all 8 XMM float args (f0..f7) --------------------------------
// The bridge reads xmm_i from state.ymm[i*4], so the test sets those slots
// directly; no guest instructions are needed to populate the XMM registers.
// The guest just loads the fn pointer and `call rax`. Position-weighted result
// makes a misrouted lane obvious.
TEST_F(CpuRuntimeTest, HleBridge_EightFloatArgs_AllXmmLanes) {
    HleRegistry::Instance().ClearForTesting();
    const u64 host_fn_addr = reinterpret_cast<u64>(&HleBridgeTestFn_EightDoubles);
    HleRegistry::Instance().Register(host_fn_addr, "HleBridgeTestFn_EightDoubles");

    u8 program[] = {
        0x48, 0xb8, 0,0,0,0,0,0,0,0, // mov rax, <fn> (imm @2..9)
        0xff, 0xd0,                  // call rax
        0xc3,                        // ret
    };
    std::memcpy(&program[2], &host_fn_addr, sizeof(host_fn_addr));
    std::memcpy(mem.CodePtr(), program, sizeof(program));

    u8* guest_rsp = mem.StackTop() - 8;
    *reinterpret_cast<u64*>(guest_rsp) = kReturnSentinel;

    GuestState state{};
    state.rip = reinterpret_cast<u64>(mem.CodePtr());
    state.gpr[4] = reinterpret_cast<u64>(guest_rsp);
    // xmm0..xmm7 = a..h = 1..8 (low 64 of each = ymm[i*4]).
    for (unsigned i = 0; i < 8; ++i)
        state.ymm[i * 4] = std::bit_cast<u64>(static_cast<double>(i + 1));

    Runtime rt;
    rt.Run(state);

    // 1*1 + 2*10 + 3*100 + 4*1000 + 5*10000 + 6*100000 + 7*1000000 + 8*10000000
    //   = 1 + 20 + 300 + 4000 + 50000 + 600000 + 7000000 + 80000000 = 87654321
    const double got = std::bit_cast<double>(state.ymm[0]);
    EXPECT_EQ(got, 87654321.0)
        << "all 8 xmm lanes must route to f0..f7 in order";
    EXPECT_EQ(state.rip, kReturnSentinel);
}

// Single far lane: only xmm5 (f5) nonzero proves f5 specifically is wired
// (not just an aggregate that could mask a swap among the high lanes).
TEST_F(CpuRuntimeTest, HleBridge_FloatArg_Xmm5_RoutesToF5) {
    HleRegistry::Instance().ClearForTesting();
    const u64 host_fn_addr = reinterpret_cast<u64>(&HleBridgeTestFn_EightDoubles);
    HleRegistry::Instance().Register(host_fn_addr, "HleBridgeTestFn_EightDoubles");

    u8 program[] = {
        0x48, 0xb8, 0,0,0,0,0,0,0,0,
        0xff, 0xd0,
        0xc3,
    };
    std::memcpy(&program[2], &host_fn_addr, sizeof(host_fn_addr));
    std::memcpy(mem.CodePtr(), program, sizeof(program));

    u8* guest_rsp = mem.StackTop() - 8;
    *reinterpret_cast<u64*>(guest_rsp) = kReturnSentinel;

    GuestState state{};
    state.rip = reinterpret_cast<u64>(mem.CodePtr());
    state.gpr[4] = reinterpret_cast<u64>(guest_rsp);
    state.ymm[5 * 4] = std::bit_cast<u64>(3.0); // xmm5 = f = 3.0, all others 0

    Runtime rt;
    rt.Run(state);

    // f has multiplier 100000 -> 3 * 100000 = 300000.
    EXPECT_EQ(std::bit_cast<double>(state.ymm[0]), 300000.0)
        << "xmm5 must map to the 6th float param (f), multiplier 100000";
}

// --- Group 2: full 14-arg call (6 register + 8 stack) ----------------------
// Pushes args 14..7 (highest address first), leaving arg7 at [rsp+8]=s0 and
// arg14 at [rsp+64]=s7. Powers-of-two multipliers keep the sum exact and make
// any misrouted slot a single wrong bit.
TEST_F(CpuRuntimeTest, HleBridge_FourteenArgs_EightStackSlots) {
    HleRegistry::Instance().ClearForTesting();
    const u64 host_fn_addr = reinterpret_cast<u64>(&HleBridgeTestFn_FourteenArgs);
    HleRegistry::Instance().Register(host_fn_addr, "HleBridgeTestFn_FourteenArgs");

    // Each stack arg pushed via `mov rax, imm32; push rax`. We push args
    // 14,13,...,7 so that after all pushes arg7 is at the lowest address
    // ([rsp+8] once the call pushes the return addr). All arg values are 1, so
    // the result is the OR of all 14 weight bits = (1<<14)-1 = 0x3FFF.
    auto push_imm = [](std::vector<u8>& p, u32 v) {
        p.push_back(0x48); p.push_back(0xc7); p.push_back(0xc0); // mov rax, imm32
        p.push_back(v & 0xFF); p.push_back((v >> 8) & 0xFF);
        p.push_back((v >> 16) & 0xFF); p.push_back((v >> 24) & 0xFF);
        p.push_back(0x50); // push rax
    };
    std::vector<u8> prog;
    // Register args a..f = 1 (rdi,rsi,rdx,rcx,r8,r9).
    auto mov_reg = [&](u8 op0, u8 op1, u8 op2, u32 v) {
        prog.push_back(op0); prog.push_back(op1); prog.push_back(op2);
        prog.push_back(v & 0xFF); prog.push_back((v >> 8) & 0xFF);
        prog.push_back((v >> 16) & 0xFF); prog.push_back((v >> 24) & 0xFF);
    };
    mov_reg(0x48, 0xc7, 0xc7, 1); // mov rdi, 1
    mov_reg(0x48, 0xc7, 0xc6, 1); // mov rsi, 1
    mov_reg(0x48, 0xc7, 0xc2, 1); // mov rdx, 1
    mov_reg(0x48, 0xc7, 0xc1, 1); // mov rcx, 1
    mov_reg(0x49, 0xc7, 0xc0, 1); // mov r8, 1
    mov_reg(0x49, 0xc7, 0xc1, 1); // mov r9, 1
    // Push stack args 14 down to 7 (each = 1).
    for (int arg = 14; arg >= 7; --arg) push_imm(prog, 1);
    // mov rax, <fn> ; call rax
    prog.push_back(0x48); prog.push_back(0xb8);
    for (int i = 0; i < 8; ++i) prog.push_back((host_fn_addr >> (i * 8)) & 0xFF);
    prog.push_back(0xff); prog.push_back(0xd0); // call rax
    prog.push_back(0x48); prog.push_back(0x83); prog.push_back(0xc4); prog.push_back(0x40); // add rsp, 64
    prog.push_back(0xc3); // ret

    std::memcpy(mem.CodePtr(), prog.data(), prog.size());

    u8* guest_rsp = mem.StackTop() - 8;
    *reinterpret_cast<u64*>(guest_rsp) = kReturnSentinel;

    GuestState state{};
    state.rip = reinterpret_cast<u64>(mem.CodePtr());
    state.gpr[4] = reinterpret_cast<u64>(guest_rsp);

    Runtime rt;
    rt.Run(state);

    // All 14 args = 1 -> OR of bits 0..13 = 0x3FFF.
    EXPECT_EQ(state.gpr[0], 0x3FFFULL)
        << "all 14 args (6 reg + 8 stack) must reach their slots; a missing "
           "slot leaves its weight bit clear";
}

// Distinct stack-arg values to pin ORDER across all 8 stack slots: arg7..arg14
// = 7..14, so the result's set bits are exactly {arg<<(arg-1)} ORed. We use
// powers of two values too, but here verify a specific high slot (arg14) lands
// at s7 by making only it nonzero.
TEST_F(CpuRuntimeTest, HleBridge_FourteenArgs_HighestStackSlotIsArg14) {
    HleRegistry::Instance().ClearForTesting();
    const u64 host_fn_addr = reinterpret_cast<u64>(&HleBridgeTestFn_FourteenArgs);
    HleRegistry::Instance().Register(host_fn_addr, "HleBridgeTestFn_FourteenArgs");

    std::vector<u8> prog;
    auto mov_reg = [&](u8 a, u8 b, u8 c, u32 v) {
        prog.insert(prog.end(), {a, b, c,
            static_cast<u8>(v & 0xFF), static_cast<u8>((v >> 8) & 0xFF),
            static_cast<u8>((v >> 16) & 0xFF), static_cast<u8>((v >> 24) & 0xFF)});
    };
    auto push_imm = [&](u32 v) {
        prog.insert(prog.end(), {0x48, 0xc7, 0xc0,
            static_cast<u8>(v & 0xFF), static_cast<u8>((v >> 8) & 0xFF),
            static_cast<u8>((v >> 16) & 0xFF), static_cast<u8>((v >> 24) & 0xFF), 0x50});
    };
    // Register args all 0.
    mov_reg(0x48, 0xc7, 0xc7, 0); mov_reg(0x48, 0xc7, 0xc6, 0);
    mov_reg(0x48, 0xc7, 0xc2, 0); mov_reg(0x48, 0xc7, 0xc1, 0);
    mov_reg(0x49, 0xc7, 0xc0, 0); mov_reg(0x49, 0xc7, 0xc1, 0);
    // Push args 14..7: only arg14 (=1) nonzero, rest 0. arg14 pushed first
    // (highest address) -> lands at s7 = [rsp+64].
    push_imm(1); // arg14 = 1
    for (int arg = 13; arg >= 7; --arg) push_imm(0);
    prog.push_back(0x48); prog.push_back(0xb8);
    for (int i = 0; i < 8; ++i) prog.push_back((host_fn_addr >> (i * 8)) & 0xFF);
    prog.push_back(0xff); prog.push_back(0xd0);
    prog.insert(prog.end(), {0x48, 0x83, 0xc4, 0x40}); // add rsp, 64
    prog.push_back(0xc3);

    std::memcpy(mem.CodePtr(), prog.data(), prog.size());
    u8* guest_rsp = mem.StackTop() - 8;
    *reinterpret_cast<u64*>(guest_rsp) = kReturnSentinel;
    GuestState state{};
    state.rip = reinterpret_cast<u64>(mem.CodePtr());
    state.gpr[4] = reinterpret_cast<u64>(guest_rsp);

    Runtime rt;
    rt.Run(state);

    // arg14 has weight bit 13 -> result = 1 << 13 = 0x2000.
    EXPECT_EQ(state.gpr[0], 0x2000ULL)
        << "arg14 must occupy the highest stack slot (s7); a wrong slot yields "
           "a different single bit";
}

// --- Group 3: return-value edge cases --------------------------------------

// A host fn returning a value with bit 63 set must round-trip through ret.rax
// into gpr[0] with no truncation or sign mangling. This specifically guards
// the AArch64 x0 capture (register u64 asm("x0")) rewritten this series.
TEST_F(CpuRuntimeTest, HleBridge_HighBitIntReturn_RoundTripsExactly) {
    HleRegistry::Instance().ClearForTesting();
    const u64 host_fn_addr = reinterpret_cast<u64>(&HleBridgeTestFn_ReturnsHighBits);
    HleRegistry::Instance().Register(host_fn_addr, "HleBridgeTestFn_ReturnsHighBits");

    u8 program[] = {
        0x48, 0xb8, 0,0,0,0,0,0,0,0,
        0xff, 0xd0,
        0xc3,
    };
    std::memcpy(&program[2], &host_fn_addr, sizeof(host_fn_addr));
    std::memcpy(mem.CodePtr(), program, sizeof(program));

    u8* guest_rsp = mem.StackTop() - 8;
    *reinterpret_cast<u64*>(guest_rsp) = kReturnSentinel;
    GuestState state{};
    state.rip = reinterpret_cast<u64>(mem.CodePtr());
    state.gpr[4] = reinterpret_cast<u64>(guest_rsp);
    state.gpr[0] = 0; // rax pre-cleared

    Runtime rt;
    rt.Run(state);

    EXPECT_EQ(state.gpr[0], 0xFEEDFACECAFEBEEFULL)
        << "high-bit-set 64-bit return must land in gpr[0] byte-exact";
    EXPECT_EQ(state.rip, kReturnSentinel);
}

// A pure-integer host fn leaves xmm0/d0 undefined on return. The guest reads
// rax (gpr[0]); the bridge's unconditional xmm0 write-back must not corrupt
// the integer result, and gpr[0] must equal the true sum.
TEST_F(CpuRuntimeTest, HleBridge_IntReturn_UnaffectedByXmmWriteback) {
    HleRegistry::Instance().ClearForTesting();
    const u64 host_fn_addr = reinterpret_cast<u64>(&HleBridgeTestFn_IntReturnIgnoresXmm);
    HleRegistry::Instance().Register(host_fn_addr, "HleBridgeTestFn_IntReturnIgnoresXmm");

    u8 program[] = {
        0x48, 0xc7, 0xc7, 0x0a, 0x00, 0x00, 0x00, // mov rdi, 10
        0x48, 0xc7, 0xc6, 0x20, 0x00, 0x00, 0x00, // mov rsi, 32
        0x48, 0xb8, 0,0,0,0,0,0,0,0,              // mov rax, <fn>
        0xff, 0xd0,                                // call rax
        0xc3,
    };
    std::memcpy(&program[16], &host_fn_addr, sizeof(host_fn_addr));
    std::memcpy(mem.CodePtr(), program, sizeof(program));

    u8* guest_rsp = mem.StackTop() - 8;
    *reinterpret_cast<u64*>(guest_rsp) = kReturnSentinel;
    GuestState state{};
    state.rip = reinterpret_cast<u64>(mem.CodePtr());
    state.gpr[4] = reinterpret_cast<u64>(guest_rsp);

    Runtime rt;
    rt.Run(state);

    EXPECT_EQ(state.gpr[0], 42ULL) << "int return (10+32) must reach gpr[0] intact";
    EXPECT_EQ(state.rip, kReturnSentinel);
}



// ----------------------------------------------------------------------------
// Dense block does not overflow the host code buffer. A long straight-line run
// of flag-setting ALU ops expands to far more host bytes than guest bytes (each
// add materializes flags). Under the old 4 KiB host-size cap with no per-insn
// guard, the x86 lifter overran the Xbyak CodeGenerator buffer and Xbyak threw
// ERR_CODE_IS_TOO_BIG mid-compile (surfacing on Windows as a C++ exception
// 0xe06d7363 followed by an access violation). The fix enlarges the cap to
// 16 KiB and adds a per-instruction size guard that ends the block early with a
// clean BlockEnd fallthrough, matching the arm64 lifter. Here we compile a
// block big enough to have overflowed the old cap; surviving compilation and
// producing the right sum proves the guard/cap fix holds.
//   xor rax,rax; mov rbx,1; (add rax,rbx) x N; ret   -> rax = N
// ----------------------------------------------------------------------------
TEST_F(CpuRuntimeTest, Dense_Block_DoesNotOverflowHostBuffer) {
    constexpr int kAdds = 300;  // ~900 guest bytes of adds; >4 KiB host bytes
    std::vector<u8> program;
    // xor rax, rax
    program.insert(program.end(), {0x48, 0x31, 0xc0});
    // mov rbx, 1
    program.insert(program.end(), {0x48, 0xc7, 0xc3, 0x01, 0x00, 0x00, 0x00});
    // add rax, rbx  (x kAdds)
    for (int i = 0; i < kAdds; ++i) {
        program.insert(program.end(), {0x48, 0x01, 0xd8});
    }
    program.push_back(0xc3);  // ret

    const auto r = RunProgram(program.data(), program.size(), mem);

    // If the block exceeded the host-size cap it ends early at BlockEnd; the
    // dispatcher then recompiles from the next RIP and continues, so the final
    // accumulated value is still kAdds regardless of how many sub-blocks it took.
    EXPECT_EQ(r.state.gpr[0], static_cast<u64>(kAdds))
        << "all adds executed across however many sub-blocks the cap produced";
    EXPECT_EQ(r.state.gpr[3], 1ULL) << "rbx held the addend";
}

} // namespace
} // namespace Core::Runtime
