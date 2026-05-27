// SPDX-FileCopyrightText: Copyright 2026 shadPS4 Emulator Project
// SPDX-License-Identifier: GPL-2.0-or-later

#include "core/cpu_runtime/runtime.h"

#include <bit>
#include <cstdio>
#include <cstdlib>
#include <cstring>

#include "common/assert.h"
#include "common/logging/log.h"
#include "core/cpu_runtime/block_cache.h"
#include "core/cpu_runtime/code_cache.h"
#include "core/cpu_runtime/gateway/gateway.h"
#include "core/cpu_runtime/lifter/lifter.h"
#include "core/cpu_runtime/hle_registry.h"

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
/// HLE bridge return value: captures *both* rax and xmm0 from
/// the called host function.
///
/// We don't know at runtime whether the called function returns an
/// int (rax-used), a double (xmm0-used), a void (neither used), or
/// something else. By declaring the bridge to return a struct
/// containing one INTEGER-class field (u64) and one SSE-class
/// field (double), the SysV ABI classifies the struct as
/// (INTEGER, SSE) — returned in rax + xmm0 respectively. The
/// compiler emits a call site that reads *both* rax and xmm0 from
/// the callee, regardless of which the callee actually populated.
///
/// We then write both back into guest state. For int-returning
/// callees: guest.rax = real return value, guest.xmm0 = garbage.
/// For double-returning callees: guest.rax = garbage, guest.xmm0
/// = real return. The guest knows the function's signature and
/// reads only the correct slot; the "garbage" slot is one that
/// SysV says is caller-saved anyway, so the guest doesn't rely on
/// the pre-call value being preserved across the call.
struct HostReturn {
    u64 rax;
    double xmm0;
};

/// HLE bridge — calls a host function with full SysV-ABI marshaling.
///
/// Arguments are unpacked from canonical SysV slots:
///
///   Integer args 0..5 →  rdi, rsi, rdx, rcx, r8, r9
///                        (state.gpr[7,6,2,1,8,9])
///   Float args   0..7 →  xmm0..xmm7
///                        (low u64 of each state.ymm[i*4])
///   Integer args 6..13 → stack at [rsp+8..+64]
///                        (read from guest stack at [guest_rsp+8..+64])
///
/// The 8 stack-spilled slots are declared as additional named u64
/// args AFTER the 6 register slots are full. Per SysV classification
/// each is INTEGER, but with the integer register pool already
/// exhausted by args 0..5, the compiler emits stores to the host
/// stack at [rsp+0, +8, ..., +56] before the call. The called HLE
/// function reads them as its 7th..14th positional args via the
/// same SysV layout.
///
/// The function pointer is variadic. This serves two purposes:
///
///   1. It makes the compiler set AL = number of XMM args used (8)
///      when emitting the call instruction. AL is required by the
///      SysV ABI for variadic targets (printf, sprintf, ...) so
///      they can locate their float args via va_arg.
///   2. For non-variadic targets the AL value is ignored, so we
///      get full coverage of both cases from one trampoline.
///
/// The `__attribute__((sysv_abi))` annotation forces SysV calling
/// convention on the call instruction. On Linux x86-64 this is the
/// default; on Windows x86-64 the default is MS x64 (args in
/// RCX/RDX/R8/R9 + shadow space) — without this attribute, args
/// would land in the wrong registers from the HLE target's POV.
///
/// Deferred (still not handled; documented as limitations):
///
///   - More than 14 integer args (would require reading further
///     into guest stack and adding more named slots). Rare; defer
///     until observed.
///   - More than 8 float args spill to stack; not yet handled.
///   - Aggregate returns via a hidden RDI pointer would silently
///     shift all args by one slot — not handled. The bridge
///     assumes scalar or empty returns.
///   - Float (32-bit) values are passed via the same XMM slots as
///     doubles; we pass the low 64 bits of each YMM lane. The host
///     function reads only the low 32 if it expects a float, which
///     is correct as long as the guest set up the float bit pattern
///     in the low 32 of the corresponding XMM. NaN canonicalization
///     by the host compiler may quiet signaling NaNs through the
///     double-typed pass-through; benign for non-NaN values.
///   - Guest stack args are read unconditionally as 8 qwords past
///     the return address. If the guest's actual stack-arg count
///     is < 8, the extra reads pull whatever happens to be on the
///     guest stack beyond the args (caller's spill area, padding).
///     The called HLE function ignores slots beyond its declared
///     arg count, so the extra reads are harmless. The only risk
///     is reading past the end of the guest stack mapping; in
///     practice guest stacks are megabytes deep and the 64-byte
///     overscan is negligible.
typedef HostReturn (*HostHleFn)(u64, u64, u64, u64, u64, u64,
                                double, double, double, double,
                                double, double, double, double,
                                u64, u64, u64, u64,
                                u64, u64, u64, u64, ...)
                              __attribute__((sysv_abi));

HostReturn CallHostFromGuest(VAddr host_fn,
                             u64 a0, u64 a1, u64 a2, u64 a3, u64 a4, u64 a5,
                             double f0, double f1, double f2, double f3,
                             double f4, double f5, double f6, double f7,
                             u64 s0, u64 s1, u64 s2, u64 s3,
                             u64 s4, u64 s5, u64 s6, u64 s7) {
    auto fn = reinterpret_cast<HostHleFn>(host_fn);
    return fn(a0, a1, a2, a3, a4, a5,
              f0, f1, f2, f3, f4, f5, f6, f7,
              s0, s1, s2, s3, s4, s5, s6, s7);
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

            // Read the guest return address now (before the call)
            // so we can log it. If the host fn crashes, this log
            // line will be the last thing emitted — telling us
            // *which* host fn crashed and *which* guest block
            // called it.
            //
            // We use fprintf+fflush rather than LOG_* because the
            // dispatcher is reached from a JIT-emitted gateway
            // frame that doesn't have Windows unwind info
            // registered; spdlog/fmt internal SEH walks would
            // fault. The same workaround is used in CompileBlock.
            const u64 guest_rsp = state->gpr[4];
            const u64 guest_return_addr =
                *reinterpret_cast<const u64*>(guest_rsp);

            // Look up the resolved HLE function name. If the
            // address isn't registered, that's either a JIT bug
            // (we computed a wrong target) or a non-import host
            // call (callback machinery, runtime helpers, etc.).
            // Either way, emit a loud warning that's easy to grep
            // for in crash logs.
            const std::string_view name =
                HleRegistry::Instance().Lookup(host_fn);

            // Marshal the 8 XMM-arg slots from state.ymm. Each YMM
            // register occupies 4 u64 lanes; the low 64 bits of
            // xmm[i] are state.ymm[i*4]. The compiler will load
            // these into xmm0..xmm7 per SysV when emitting the call.
            const auto load_xmm_low = [&](unsigned i) -> double {
                return std::bit_cast<double>(state->ymm[i * 4]);
            };
            const double f0 = load_xmm_low(0);
            const double f1 = load_xmm_low(1);
            const double f2 = load_xmm_low(2);
            const double f3 = load_xmm_low(3);
            const double f4 = load_xmm_low(4);
            const double f5 = load_xmm_low(5);
            const double f6 = load_xmm_low(6);
            const double f7 = load_xmm_low(7);

            // Marshal up to 8 stack-spilled integer args from guest
            // stack. SysV places them at [rsp+8, +16, +24, ...]
            // (just past the return address). The compiler will
            // re-spill them onto the host stack via our named u64
            // params s0..s7.
            //
            // If the guest's actual stack-arg count is < 8, the
            // extra reads pull whatever lives beyond the args. The
            // called HLE function ignores them per its declared
            // signature.
            const u64* guest_stack_args =
                reinterpret_cast<const u64*>(guest_rsp) + 1;
            const u64 s0 = guest_stack_args[0];
            const u64 s1 = guest_stack_args[1];
            const u64 s2 = guest_stack_args[2];
            const u64 s3 = guest_stack_args[3];
            const u64 s4 = guest_stack_args[4];
            const u64 s5 = guest_stack_args[5];
            const u64 s6 = guest_stack_args[6];
            const u64 s7 = guest_stack_args[7];

            if (name.empty()) {
                std::fprintf(stderr,
                    "[bridge] WARNING unregistered host=0x%llx ret=0x%llx | "
                    "rdi=0x%llx rsi=0x%llx rdx=0x%llx "
                    "rcx=0x%llx r8=0x%llx r9=0x%llx\n",
                    (unsigned long long)host_fn,
                    (unsigned long long)guest_return_addr,
                    (unsigned long long)state->gpr[7],
                    (unsigned long long)state->gpr[6],
                    (unsigned long long)state->gpr[2],
                    (unsigned long long)state->gpr[1],
                    (unsigned long long)state->gpr[8],
                    (unsigned long long)state->gpr[9]);
            } else {
                // %.*s lets us print a non-null-terminated
                // string_view without copying it. The cast to int
                // is required because precision uses int width.
                std::fprintf(stderr,
                    "[bridge] call %.*s host=0x%llx ret=0x%llx | "
                    "rdi=0x%llx rsi=0x%llx rdx=0x%llx "
                    "rcx=0x%llx r8=0x%llx r9=0x%llx\n",
                    static_cast<int>(name.size()), name.data(),
                    (unsigned long long)host_fn,
                    (unsigned long long)guest_return_addr,
                    (unsigned long long)state->gpr[7],
                    (unsigned long long)state->gpr[6],
                    (unsigned long long)state->gpr[2],
                    (unsigned long long)state->gpr[1],
                    (unsigned long long)state->gpr[8],
                    (unsigned long long)state->gpr[9]);
            }
            // Log XMM args on a continuation line. Print as hex bit
            // patterns rather than decimals — they're easier to
            // recognize as "this is 1.0" vs "this is garbage" by eye.
            std::fprintf(stderr,
                "[bridge]   xmm0=0x%llx xmm1=0x%llx xmm2=0x%llx xmm3=0x%llx "
                "xmm4=0x%llx xmm5=0x%llx xmm6=0x%llx xmm7=0x%llx\n",
                (unsigned long long)state->ymm[0],
                (unsigned long long)state->ymm[4],
                (unsigned long long)state->ymm[8],
                (unsigned long long)state->ymm[12],
                (unsigned long long)state->ymm[16],
                (unsigned long long)state->ymm[20],
                (unsigned long long)state->ymm[24],
                (unsigned long long)state->ymm[28]);
            // Log stack-spilled args on a third continuation line.
            // Same caveats as XMM line: if the function takes few
            // stack args, the latter slots are whatever the guest
            // happened to leave there. Still useful for diagnosis.
            std::fprintf(stderr,
                "[bridge]   stk0=0x%llx stk1=0x%llx stk2=0x%llx stk3=0x%llx "
                "stk4=0x%llx stk5=0x%llx stk6=0x%llx stk7=0x%llx\n",
                (unsigned long long)s0, (unsigned long long)s1,
                (unsigned long long)s2, (unsigned long long)s3,
                (unsigned long long)s4, (unsigned long long)s5,
                (unsigned long long)s6, (unsigned long long)s7);
            std::fflush(stderr);

            const HostReturn ret = CallHostFromGuest(
                host_fn,
                state->gpr[7],   // RDI
                state->gpr[6],   // RSI
                state->gpr[2],   // RDX
                state->gpr[1],   // RCX
                state->gpr[8],   // R8
                state->gpr[9],   // R9
                f0, f1, f2, f3, f4, f5, f6, f7,
                s0, s1, s2, s3, s4, s5, s6, s7);
            // Write both rax and xmm0 back to guest state. The
            // guest knows which one is meaningful based on the
            // called function's signature; the "other" slot may
            // have been clobbered by the call but x86-64 calling
            // convention says rax/xmm0 are caller-saved, so the
            // guest doesn't rely on the pre-call values surviving.
            state->gpr[0] = ret.rax;
            const u64 xmm0_bits = std::bit_cast<u64>(ret.xmm0);
            std::memcpy(&state->ymm[0], &xmm0_bits, sizeof(u64));

            // Print the post-return rax/xmm0. If we never see this
            // line for a given host_fn, that function crashed.
            if (name.empty()) {
                std::fprintf(stderr,
                    "[bridge] ret  host=0x%llx -> rax=0x%llx xmm0=0x%llx\n",
                    (unsigned long long)host_fn,
                    (unsigned long long)ret.rax,
                    (unsigned long long)xmm0_bits);
            } else {
                std::fprintf(stderr,
                    "[bridge] ret  %.*s -> rax=0x%llx xmm0=0x%llx\n",
                    static_cast<int>(name.size()), name.data(),
                    (unsigned long long)ret.rax,
                    (unsigned long long)xmm0_bits);
            }
            std::fflush(stderr);

            // Pop guest return address.
            state->rip = guest_return_addr;
            state->gpr[4] = guest_rsp + 8;
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
