// SPDX-FileCopyrightText: Copyright 2026 shadPS4 Emulator Project
// SPDX-License-Identifier: GPL-2.0-or-later

#pragma once

#include <memory>
#include "common/types.h"
#include "core/cpu_runtime/guest_state.h"

namespace Core::Runtime {

class BlockCache;
class CodeCache;
class Gateway;
class Lifter;

/// Public entry point for the CPU runtime. Created once during
/// emulator init, used by Linker::Execute as the alternative to
/// native execution when SHADPS4_CPU_BACKEND=runtime.
///
/// Threading: methods on this class are thread-safe. Multiple guest
/// threads dispatching through Run() share the block cache and code
/// cache; concurrent compilation of distinct blocks works without
/// additional synchronization on the caller's part.
class Runtime {
public:
    Runtime();
    ~Runtime();

    Runtime(const Runtime&) = delete;
    Runtime& operator=(const Runtime&) = delete;

    /// Execute guest code starting at the given guest RIP, with the
    /// provided initial GuestState. Returns when the guest hits a
    /// halt-equivalent (typically the kernel's exit shim), at which
    /// point the state's exit_reason is set and state.rip points to
    /// the next un-executed guest instruction (or 0 if cleanly
    /// terminated).
    ///
    /// This call doesn't return until the guest is done. For typical
    /// game workloads that's the lifetime of the game; for the
    /// future debugger path, the runtime supports asynchronous
    /// break-in via AsyncBreak() which causes this to return early.
    void Run(GuestState& state);

    /// Request that the currently-executing Run() return at the next
    /// safe point (block boundary). May be called from another
    /// thread. Idempotent.
    void AsyncBreak();

    /// Diagnostic accessors. Provided for debugging and for the
    /// devtools widget that displays runtime state.
    [[nodiscard]] const BlockCache& GetBlockCache() const noexcept {
        return *block_cache_;
    }
    [[nodiscard]] const CodeCache& GetCodeCache() const noexcept {
        return *code_cache_;
    }

    /// Internal use by the dispatcher trampoline: compile a block at
    /// the given guest RIP and return its host code pointer. Exposed
    /// because the trampoline is a free function (gateway signature
    /// constraints) and needs access to the lifter.
    void* CompileBlockForDispatcher(u64 guest_rip);

    // ========================================================================
    // Process-wide singleton.
    //
    // The runtime is heavyweight (allocates a 64 MB code cache, generates
    // the gateway via xbyak) so we don't want to construct a fresh one for
    // every guest-entry site. All sites share one runtime via Instance().
    //
    // Thread-safety: Instance() is safe to call from any thread; the
    // static-local initialization is C++11-guaranteed thread-safe. The
    // returned Runtime is itself thread-safe for concurrent Run() calls
    // from different threads (each thread enters its own gateway stack
    // frame).
    // ========================================================================
    static Runtime& Instance();

    // ========================================================================
    // Helpers for invoking guest code at sites other than the main entry.
    //
    // These exist to centralize the "set up GuestState, call Run(),
    // extract result" boilerplate. Without them, each integration site
    // would duplicate the same dozen lines of GuestState manipulation,
    // and the ABI conventions (which GPR is RDI, which is RSI, where the
    // return value lands) would be open-coded everywhere.
    //
    // The caller provides:
    //   - The guest function address.
    //   - A guest stack address (allocated by the caller; the runtime
    //     does not manage stacks for callbacks because each site has
    //     different stack-lifetime concerns).
    //   - A setup callback that takes GuestState& and sets up RDI/RSI/etc.
    //     per the PS4_SYSV_ABI for the specific signature being invoked.
    //
    // CallGuest returns the GuestState after execution. The caller reads
    // the return value out of state.gpr[0] (RAX) for integer/pointer
    // returns, or state.xmm[0] for float/SSE returns.
    //
    // CallGuestSimple is a convenience wrapper for the common case of a
    // function taking up to 6 pointer-sized arguments and returning a
    // pointer-sized value. It does the argument marshalling itself.
    // ========================================================================
    using SetupFn = void (*)(GuestState& state, void* user_data);

    /// Invoke a guest function at `guest_fn`. The `setup` callback is
    /// called with a fresh GuestState; it should populate the argument
    /// registers per PS4_SYSV_ABI. `guest_stack_top` is the high end of
    /// a stack region the caller has allocated; the runtime sets up
    /// GuestState::gpr[4] (RSP) to a properly-aligned address near it.
    ///
    /// Returns the GuestState after execution. The caller is responsible
    /// for extracting return values from the appropriate register fields.
    GuestState CallGuest(VAddr guest_fn, void* guest_stack_top,
                         SetupFn setup, void* user_data);

    /// Convenience: call a guest function with up to 6 pointer-sized
    /// arguments. Returns the value of RAX (which holds the return
    /// value for integer/pointer returns under PS4_SYSV_ABI). Caller
    /// provides the guest stack.
    u64 CallGuestSimple(VAddr guest_fn, void* guest_stack_top,
                        u64 a0 = 0, u64 a1 = 0, u64 a2 = 0,
                        u64 a3 = 0, u64 a4 = 0, u64 a5 = 0);

private:
    std::unique_ptr<BlockCache> block_cache_;
    std::unique_ptr<CodeCache> code_cache_;
    std::unique_ptr<Gateway> gateway_;
    std::unique_ptr<Lifter> lifter_;
};

} // namespace Core::Runtime
