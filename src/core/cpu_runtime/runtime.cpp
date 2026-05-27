// SPDX-FileCopyrightText: Copyright 2026 shadPS4 Emulator Project
// SPDX-License-Identifier: GPL-2.0-or-later

#include "core/cpu_runtime/runtime.h"

#include <cstdlib>

#include "common/assert.h"
#include "common/logging/log.h"
#include "core/cpu_runtime/block_cache.h"
#include "core/cpu_runtime/code_cache.h"
#include "core/cpu_runtime/gateway/gateway.h"
#include "core/cpu_runtime/lifter/lifter.h"

namespace Core::Runtime {

namespace {

// Thread-local pointer to the Runtime instance that owns the currently
// executing gateway. Used by the dispatcher trampoline to find the
// BlockCache and Lifter. We use TLS rather than passing through
// GuestState because the dispatcher signature is fixed (state -> ptr)
// and we don't want to grow it.
thread_local Runtime* tl_active_runtime = nullptr;

// Thread-local pointer to the GuestState of the JIT execution currently
// active on this thread. Set by Run() on entry, restored on exit (so
// nested Run() calls behave correctly). Exposed via the public
// Runtime::CurrentGuestState() accessor for HLE shims that need to
// invoke guest callbacks on the caller's stack.
//
// nullptr when no JIT execution is active on this thread.
thread_local GuestState* tl_current_guest_state = nullptr;

/// HLE bridge — calls a host function as if it were the next-block
/// continuation of guest execution.
///
/// We use a `PS4_SYSV_ABI` function pointer so the *call instruction*
/// emitted at the call site below uses System V x86-64 calling
/// convention (args in RDI/RSI/RDX/RCX/R8/R9, return in RAX). This
/// matches the convention shadPS4's HLE functions are declared with.
///
///   - On Linux x86-64, SysV is the host C ABI by default — the
///     attribute is a no-op but harmless.
///   - On Windows x86-64, the host C ABI is MS x64 (args in
///     RCX/RDX/R8/R9 + shadow space). Without `PS4_SYSV_ABI` the
///     compiler would generate an MS-x64 call here, and arguments
///     would land in the wrong registers from the HLE function's
///     point of view. With the attribute, clang generates a SysV
///     call and arguments flow correctly.
///
/// Arguments are unpacked from canonical SysV arg-passing slots in
/// `state.gpr[]`:
///
///   1st arg  →  gpr[7]  (RDI)
///   2nd arg  →  gpr[6]  (RSI)
///   3rd arg  →  gpr[2]  (RDX)
///   4th arg  →  gpr[1]  (RCX)
///   5th arg  →  gpr[8]  (R8)
///   6th arg  →  gpr[9]  (R9)
///
/// The return value (RAX in the SysV ABI) is written to `gpr[0]`.
///
/// Deferred (not yet handled; documented as limitations):
///
///   - Float/double args (XMM0..7 in the SSE arg-passing class).
///   - More than 6 integer args (would spill to the stack and we
///     don't marshal stack args yet).
///   - Variadic functions (RAX = SSE arg count is required).
///   - Aggregate returns via a hidden RDI pointer (would shift all
///     subsequent args by one slot — silent misalignment).
///
/// In practice shadPS4's HLE shims are integer-arg/integer-return
/// functions, so the minimal bridge above covers them. We'll widen
/// the bridge when a real HLE function trips one of the gaps.
typedef u64 (*HostHleFn)(u64, u64, u64, u64, u64, u64) __attribute__((sysv_abi));

u64 CallHostFromGuest(VAddr host_fn,
                      u64 a0, u64 a1, u64 a2, u64 a3, u64 a4, u64 a5) {
    auto fn = reinterpret_cast<HostHleFn>(host_fn);
    return fn(a0, a1, a2, a3, a4, a5);
}

/// Dispatcher trampoline. Called from the gateway with the current
/// state; returns the host code pointer for `state.rip`, compiling
/// on demand if needed. Returning nullptr causes the gateway to
/// exit the dispatch loop and return to C.
///
/// Iterates over consecutive host-function calls without going back
/// to the gateway: each time the popped return address is itself a
/// host pointer (e.g. one HLE shim returning into another), the
/// loop body handles it inline. Only when `state.rip` ends up
/// pointing at a guest block (or at the host-return sentinel) does
/// the function return.
void* DispatcherTrampoline(GuestState* state) {
    Runtime* rt = tl_active_runtime;
    ASSERT_MSG(rt != nullptr, "Dispatcher called with no active runtime");

    while (true) {
        // Sentinel check: if guest code RET'd through the call chain
        // back to the host-return address, exit cleanly.
        if (state->rip == kHostReturnAddress) {
            state->exit_reason = static_cast<u32>(ExitReason::BlockEnd);
            return nullptr;
        }

        // HLE bridge: if state.rip points at host code (anywhere in
        // a loaded host module — shadps4.exe itself, any system DLL,
        // etc.), call it via the SysV-ABI bridge instead of trying
        // to JIT-compile bytes from host memory.
        //
        // After the call:
        //   1. Write the return value into guest RAX.
        //   2. Pop the return address from the guest stack into
        //      state.rip. (Guest code's pre-call setup pushed it.)
        //   3. Loop. The new state.rip might be a guest block (we
        //      fall through to JIT it), another host function
        //      (another bridge iteration), or the host-return
        //      sentinel (we exit).
        if (!rt->IsGuestPointer(reinterpret_cast<const void*>(state->rip))) {
            const VAddr host_fn = state->rip;
            const u64 ret = CallHostFromGuest(
                host_fn,
                state->gpr[7],   // RDI
                state->gpr[6],   // RSI
                state->gpr[2],   // RDX
                state->gpr[1],   // RCX
                state->gpr[8],   // R8
                state->gpr[9]);  // R9
            state->gpr[0] = ret; // RAX

            // Pop guest return address.
            const u64 rsp = state->gpr[4];
            state->rip = *reinterpret_cast<const u64*>(rsp);
            state->gpr[4] = rsp + 8;
            continue;
        }

        // Guest path: cache lookup, compile on miss.
        BlockCache& bc = const_cast<BlockCache&>(rt->GetBlockCache());
        if (void* host_ptr = bc.Lookup(state->rip); host_ptr != nullptr) {
            return host_ptr;
        }
        void* host_ptr = rt->CompileBlockForDispatcher(state->rip);
        if (host_ptr == nullptr) {
            // Compile failed. Returning nullptr causes the gateway
            // to exit with whatever exit_reason the lifter set.
            return nullptr;
        }
        bc.Insert(state->rip, host_ptr);
        return host_ptr;
    }
}

} // namespace

Runtime::Runtime()
    : block_cache_(std::make_unique<BlockCache>()),
      code_cache_(std::make_unique<CodeCache>()),
      gateway_(std::make_unique<Gateway>()),
      lifter_(std::make_unique<Lifter>(*code_cache_)) {
    LOG_INFO(Core, "CPU runtime initialized (gateway + lifter ready)");
}

Runtime::~Runtime() = default;

void Runtime::Run(GuestState& state) {
    // Save the previous active-runtime and current-state pointers for
    // this thread. Setting these is per-Run nesting, not a global
    // lock: nested Run() calls (typical for HLE → guest callback →
    // HLE → guest pattern) restore the outer Run's pointers on exit.
    Runtime* const saved_rt = tl_active_runtime;
    GuestState* const saved_state = tl_current_guest_state;
    tl_active_runtime = this;
    tl_current_guest_state = &state;
    gateway_->Enter(state, &DispatcherTrampoline);
    tl_current_guest_state = saved_state;
    tl_active_runtime = saved_rt;
}

GuestState* Runtime::CurrentGuestState() noexcept {
    return tl_current_guest_state;
}

void Runtime::AsyncBreak() {
    // Stub. When the dispatcher loop adds break-in support, this
    // sets a flag the dispatcher checks at each entry.
    LOG_DEBUG(Core, "Runtime::AsyncBreak — not yet implemented");
}

void* Runtime::CompileBlockForDispatcher(u64 guest_rip) {
    return lifter_->CompileBlock(guest_rip);
}

// ============================================================================
// Singleton
// ============================================================================

Runtime& Runtime::Instance() {
    // Magic statics — C++11 guarantees thread-safe initialization.
    static Runtime instance;
    return instance;
}

// ============================================================================
// CallGuest helpers
// ============================================================================

GuestState Runtime::CallGuest(VAddr guest_fn, void* guest_stack_top,
                              SetupFn setup, void* user_data) {
    ASSERT_MSG(guest_fn != 0, "CallGuest: null guest_fn");
    ASSERT_MSG(guest_stack_top != nullptr, "CallGuest: null guest_stack_top");

    GuestState state{};

    // Align guest stack to 16, misalign by 8 (PS4_SYSV_ABI: caller's
    // stack pointer is 8-byte-misaligned at function entry, becoming
    // 16-byte-aligned after the implicit return-address push).
    u64 rsp = reinterpret_cast<u64>(guest_stack_top);
    rsp &= ~static_cast<u64>(0xF);
    rsp -= 8;

    // Push the host-return sentinel address onto the guest stack.
    // When guest code RETs through the full call chain, RSP points
    // here, RET pops it into state.rip, and the dispatcher
    // recognizes it as the signal to return control.
    rsp -= 8;
    *reinterpret_cast<u64*>(rsp) = kHostReturnAddress;

    state.gpr[4] = rsp;             // RSP
    state.rip = guest_fn;

    // Let the caller populate argument registers.
    if (setup != nullptr) {
        setup(state, user_data);
    }

    // Run.
    Run(state);

    // At this point state.rip should equal kReturnSentinel (clean RET)
    // or some other value (unsupported instruction, fault). The caller
    // inspects state to determine which.
    return state;
}

namespace {
// User-data struct for CallGuestSimple's setup callback.
struct SimpleArgs {
    u64 a0, a1, a2, a3, a4, a5;
};

// Populate registers per PS4_SYSV_ABI integer-argument convention.
// First six pointer-sized args go in RDI, RSI, RDX, RCX, R8, R9
// (GPR indices 7, 6, 2, 1, 8, 9 in canonical AMD64 ordering).
void SimpleSetup(GuestState& state, void* user_data) {
    const auto* args = static_cast<const SimpleArgs*>(user_data);
    state.gpr[7] = args->a0;  // RDI
    state.gpr[6] = args->a1;  // RSI
    state.gpr[2] = args->a2;  // RDX
    state.gpr[1] = args->a3;  // RCX
    state.gpr[8] = args->a4;  // R8
    state.gpr[9] = args->a5;  // R9
}
} // namespace

u64 Runtime::CallGuestSimple(VAddr guest_fn, void* guest_stack_top,
                             u64 a0, u64 a1, u64 a2,
                             u64 a3, u64 a4, u64 a5) {
    SimpleArgs args{a0, a1, a2, a3, a4, a5};
    GuestState state = CallGuest(guest_fn, guest_stack_top,
                                 &SimpleSetup, &args);
    return state.gpr[0];  // RAX
}

// ============================================================================
// CallGuestOnCallerStack — reuses caller's GuestState
// ============================================================================
//
// PS4_SYSV_ABI register classification (AMD64 SysV):
//   Callee-saved (preserved across calls): RBX, RBP, R12, R13, R14, R15
//   Caller-saved (may be clobbered):       RAX, RCX, RDX, RSI, RDI, R8-R11
//
// GuestState gpr[] index mapping (canonical AMD64 ordering):
//   RAX=0, RCX=1, RDX=2, RBX=3, RSP=4, RBP=5, RSI=6, RDI=7,
//   R8=8, R9=9, R10=10, R11=11, R12=12, R13=13, R14=14, R15=15

void Runtime::CallGuestOnCallerStack(GuestState& caller, VAddr guest_fn,
                                     SetupFn setup, void* user_data) {
    ASSERT_MSG(guest_fn != 0, "CallGuestOnCallerStack: null guest_fn");
    ASSERT_MSG(caller.gpr[4] != 0,
               "CallGuestOnCallerStack: caller has null RSP (uninitialized?)");

    // Snapshot callee-saved registers, plus RSP and RIP (which we'll
    // restore to whatever the caller had before the callback).
    const u64 saved_rbx = caller.gpr[3];
    const u64 saved_rsp = caller.gpr[4];
    const u64 saved_rbp = caller.gpr[5];
    const u64 saved_r12 = caller.gpr[12];
    const u64 saved_r13 = caller.gpr[13];
    const u64 saved_r14 = caller.gpr[14];
    const u64 saved_r15 = caller.gpr[15];
    const u64 saved_rip = caller.rip;

    // Set up RIP for the callback. Then let the caller-supplied setup
    // populate argument registers.
    caller.rip = guest_fn;
    if (setup != nullptr) {
        setup(caller, user_data);
    }

    // Push a sentinel return address onto the caller's stack so the
    // callback's RET exits the JIT cleanly. We align to 16 then push
    // (so the callback enters with the standard 8-misaligned RSP that
    // SysV ABI expects at function entry, post-push-return-address).
    constexpr u64 kReturnSentinel = 0xCB'CB'CB'CB'00'00'00'00ULL;
    u64 rsp = caller.gpr[4];
    rsp &= ~static_cast<u64>(0xF);
    rsp -= 8;
    *reinterpret_cast<u64*>(rsp) = kReturnSentinel;
    caller.gpr[4] = rsp;

    // Run the callback through the JIT on the caller's stack.
    Run(caller);

    // Restore callee-saved registers, RSP, and RIP. Leave caller-saved
    // registers (including RAX, which holds the callback's return
    // value) in whatever state the callback left them — that's the
    // ABI contract.
    caller.gpr[3] = saved_rbx;
    caller.gpr[4] = saved_rsp;
    caller.gpr[5] = saved_rbp;
    caller.gpr[12] = saved_r12;
    caller.gpr[13] = saved_r13;
    caller.gpr[14] = saved_r14;
    caller.gpr[15] = saved_r15;
    caller.rip = saved_rip;
}

u64 Runtime::CallGuestSimpleOnCallerStack(GuestState& caller, VAddr guest_fn,
                                          u64 a0, u64 a1, u64 a2,
                                          u64 a3, u64 a4, u64 a5) {
    SimpleArgs args{a0, a1, a2, a3, a4, a5};
    CallGuestOnCallerStack(caller, guest_fn, &SimpleSetup, &args);
    return caller.gpr[0];  // RAX
}

// ============================================================================
// Dual-context dispatch (the shared HLE-callback helper)
// ============================================================================

u64 Runtime::InvokeGuestCallback(VAddr guest_fn,
                                 u64 a0, u64 a1, u64 a2,
                                 u64 a3, u64 a4, u64 a5) {
    GuestState* caller_state = CurrentGuestState();
    if (caller_state != nullptr) {
        return CallGuestSimpleOnCallerStack(*caller_state, guest_fn,
                                            a0, a1, a2, a3, a4, a5);
    }

    // Post-JIT path: HLE worker thread invoking a guest callback.
    // 256 KB matches the size used by pthread cleanup, ThreadDtors,
    // and the AvPlayer wrappers.
    constexpr u64 kCallbackStackSize = 256 * 1024;
    void* guest_stack = std::malloc(kCallbackStackSize);
    if (guest_stack == nullptr) {
        LOG_ERROR(Core, "InvokeGuestCallback: failed to allocate guest stack");
        return 0;
    }
    void* guest_stack_top = static_cast<u8*>(guest_stack) + kCallbackStackSize;
    const u64 result = CallGuestSimple(guest_fn, guest_stack_top,
                                       a0, a1, a2, a3, a4, a5);
    std::free(guest_stack);
    return result;
}

// ============================================================================
// Host-vs-guest pointer discrimination
// ============================================================================
//
// We use OS APIs to answer "is this address in a loaded host module?"
//   - POSIX: dladdr(ptr, &info) returns non-zero if ptr is in any loaded
//            shared object (executable or .so). 0 means not in any loaded
//            module — which is the case for guest memory mapped via
//            shadPS4's loader.
//   - Windows: GetModuleHandleExW(GET_MODULE_HANDLE_EX_FLAG_FROM_ADDRESS,
//              ptr, &handle) succeeds if ptr is in any loaded module (.exe
//              or .dll), fails otherwise.
//
// A naive address-range check (e.g. "ptr >= 0x800000000") was tempting
// but wrong: under PIE+ASLR on Linux, host code itself often lives well
// above that threshold. dladdr is the right primitive.

#ifdef _WIN32
#include <windows.h>
#else
#include <dlfcn.h>
#endif

bool Runtime::IsGuestPointer(const void* ptr) noexcept {
    if (ptr == nullptr) {
        return false;
    }
#ifdef _WIN32
    HMODULE handle = nullptr;
    if (GetModuleHandleExW(
            GET_MODULE_HANDLE_EX_FLAG_FROM_ADDRESS |
                GET_MODULE_HANDLE_EX_FLAG_UNCHANGED_REFCOUNT,
            reinterpret_cast<LPCWSTR>(ptr), &handle)) {
        // ptr is inside a loaded host module — it's host code.
        return false;
    }
    // Not in any loaded module — assume guest.
    return true;
#else
    Dl_info info{};
    if (dladdr(ptr, &info) != 0) {
        // dladdr found a containing module — it's host code.
        return false;
    }
    // dladdr returned 0 — ptr is not in any loaded host module.
    // Assume guest.
    return true;
#endif
}

} // namespace Core::Runtime
