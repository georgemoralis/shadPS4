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
    ASSERT_MSG(tl_active_runtime == nullptr,
               "Runtime::Run already active on this thread "
               "(re-entry not yet supported)");
    tl_active_runtime = this;
    gateway_->Enter(state, &DispatcherTrampoline);
    tl_active_runtime = nullptr;
}

void Runtime::AsyncBreak() {
    // Stub. When the dispatcher loop adds break-in support, this
    // sets a flag the dispatcher checks at each entry.
    LOG_DEBUG(Core, "Runtime::AsyncBreak — not yet implemented");
}

void* Runtime::CompileBlockForDispatcher(u64 guest_rip) {
    return lifter_->CompileBlock(guest_rip);
}

} // namespace Core::Runtime
