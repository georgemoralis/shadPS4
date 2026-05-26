// SPDX-FileCopyrightText: Copyright 2026 shadPS4 Emulator Project
// SPDX-License-Identifier: GPL-2.0-or-later

#include "core/cpu_runtime/runtime.h"

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

/// Dispatcher trampoline. Called from the gateway with the current
/// state; returns the host code pointer for state.rip, compiling on
/// demand if needed.
void* DispatcherTrampoline(GuestState* state) {
    Runtime* rt = tl_active_runtime;
    ASSERT_MSG(rt != nullptr, "Dispatcher called with no active runtime");

    BlockCache& bc = const_cast<BlockCache&>(rt->GetBlockCache());
    if (void* host_ptr = bc.Lookup(state->rip); host_ptr != nullptr) {
        return host_ptr;
    }

    // Cache miss — compile the block.
    // Friend access: Runtime exposes its lifter via a private hook.
    void* host_ptr = rt->CompileBlockForDispatcher(state->rip);
    if (host_ptr == nullptr) {
        // Compile failed. Returning nullptr causes the gateway to
        // exit to C with whatever exit_reason the lifter set in
        // its fallback.
        return nullptr;
    }
    bc.Insert(state->rip, host_ptr);
    return host_ptr;
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
    // Save the previous active-runtime pointer for this thread. Setting
    // tl_active_runtime is per-Run nesting, not a global lock: nested
    // Run() calls (typical for HLE → guest callback → HLE → guest pattern)
    // restore the outer Run's runtime pointer on exit.
    Runtime* const saved = tl_active_runtime;
    tl_active_runtime = this;
    gateway_->Enter(state, &DispatcherTrampoline);
    tl_active_runtime = saved;
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

    // Push a sentinel return address that will exit the runtime cleanly
    // when the guest function executes its terminal RET. We use a
    // recognizable invalid address; the lifter's RET handler will set
    // state.rip to this value and exit_reason to BlockEnd, at which
    // point we return.
    constexpr u64 kReturnSentinel = 0xCB'CB'CB'CB'00'00'00'00ULL;
    rsp -= 8;
    *reinterpret_cast<u64*>(rsp) = kReturnSentinel;

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

} // namespace Core::Runtime
