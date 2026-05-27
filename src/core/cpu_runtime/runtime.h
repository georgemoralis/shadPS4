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
    GuestState CallGuest(VAddr guest_fn, void* guest_stack_top, SetupFn setup, void* user_data);

    /// Convenience: call a guest function with up to 6 pointer-sized
    /// arguments. Returns the value of RAX (which holds the return
    /// value for integer/pointer returns under PS4_SYSV_ABI). Caller
    /// provides the guest stack.
    u64 CallGuestSimple(VAddr guest_fn, void* guest_stack_top, u64 a0 = 0, u64 a1 = 0, u64 a2 = 0,
                        u64 a3 = 0, u64 a4 = 0, u64 a5 = 0);

    // ========================================================================
    // Caller-context infrastructure for HLE callbacks invoked from
    // mid-JIT execution.
    //
    // When an HLE shim is called from JIT-executing guest code and the
    // shim itself invokes a guest callback, the callback must run with
    // the caller's RSP and (after return) leave the caller's callee-
    // saved registers intact. Using CallGuestSimple here would lose
    // the caller's stack chain.
    //
    // CurrentGuestState() returns the GuestState* of the JIT execution
    // active on the calling thread, or nullptr if not inside a JIT
    // call. Set by Run() on entry, restored on exit (handles nesting).
    //
    // CallGuestOnCallerStack invokes a guest function reusing the
    // caller's GuestState. Per PS4_SYSV_ABI, callee-saved registers
    // (RBX, RBP, R12-R15) and RSP/RIP are preserved across the call;
    // caller-saved registers may be clobbered. The callback's return
    // value lands in caller.gpr[0] (RAX) per the ABI.
    //
    // CallGuestSimpleOnCallerStack is the integer-args convenience
    // wrapper, analogous to CallGuestSimple.
    // ========================================================================

    /// Returns the GuestState pointer for the JIT execution currently
    /// active on the calling thread, or nullptr if no JIT execution
    /// is active. Cheap (single TLS read).
    [[nodiscard]] static GuestState* CurrentGuestState() noexcept;

    /// Invoke a guest function on the caller's stack, preserving the
    /// caller's callee-saved registers, RSP, and RIP per the
    /// PS4_SYSV_ABI. The `setup` callback receives the caller's
    /// GuestState; it should set up argument registers (RDI, RSI,
    /// etc.) for the callback. Caller-saved registers (RAX, RCX,
    /// RDX, RDI, RSI, R8-R11) may be clobbered after this call.
    void CallGuestOnCallerStack(GuestState& caller, VAddr guest_fn, SetupFn setup, void* user_data);

    /// Convenience: invoke a guest function on the caller's stack
    /// with up to 6 pointer-sized integer arguments. Returns the
    /// callback's return value (caller.gpr[0] after the call).
    u64 CallGuestSimpleOnCallerStack(GuestState& caller, VAddr guest_fn, u64 a0 = 0, u64 a1 = 0,
                                     u64 a2 = 0, u64 a3 = 0, u64 a4 = 0, u64 a5 = 0);

    // ========================================================================
    // Dual-context dispatch — the common HLE-callback pattern
    //
    // Most HLE callback sites have the same shape:
    //   - If CurrentGuestState() returns non-null (mid-JIT, the HLE
    //     wrapper was called from JIT code), use the caller's stack.
    //   - Otherwise (post-JIT, the HLE wrapper was called from a host
    //     thread like libusb's worker or AvPlayer's controller),
    //     allocate a fresh stack via malloc.
    //
    // Writing this dispatch by hand at each call site (as PR 1.5d-1
    // and PR 1.5d-2 do) produces ~15 lines of `#ifdef`-gated
    // boilerplate per site. With ~14 more sites to convert, that's
    // 200+ lines of identical-shape code. This helper collapses it
    // to a single call per site.
    //
    // The helper allocates a 256 KB stack for the post-JIT case
    // (matches the size used by other post-JIT sites in this PR).
    // The allocation can fail (returns 0 in that case with an error
    // logged); AvPlayer-style callers that can't recover should
    // check for that, but most callers ignore the return value
    // because the callbacks are void.
    //
    // Returns the callback's RAX. For void callbacks, the value is
    // meaningless and can be discarded.
    // ========================================================================

    /// Dual-context guest-callback invocation. Picks
    /// CallGuestSimpleOnCallerStack or CallGuestSimple based on
    /// whether a caller GuestState is active. Returns RAX after the
    /// call (0 if allocation failed in the post-JIT path).
    u64 InvokeGuestCallback(VAddr guest_fn, u64 a0 = 0, u64 a1 = 0, u64 a2 = 0, u64 a3 = 0,
                            u64 a4 = 0, u64 a5 = 0);

    // ========================================================================
    // Host-vs-guest function-pointer discrimination
    //
    // Several integration boundaries receive a function pointer of
    // unknown origin: it could point to guest code (loaded by
    // shadPS4's ELF loader into a mapped memory region) or host code
    // (C++ thunks, HOST_CALL wrappers, etc., living in the host
    // process's own .text segment or shared libraries).
    //
    // Under the runtime backend, guest pointers must be invoked
    // through the JIT (via Run/CallGuest*); host pointers must be
    // invoked natively (the JIT can't lift host bytes as guest
    // instructions). The native backend doesn't care since both
    // kinds are just function pointers, but the runtime backend
    // needs to distinguish them.
    //
    // We use OS APIs to ask "is this address in a loaded host
    // module?":
    //   - POSIX: dladdr() succeeds for addresses in loaded .so/.exe
    //   - Windows: GetModuleHandleExW with FROM_ADDRESS flag
    // Addresses not in any loaded host module are assumed to be
    // guest (shadPS4 maps guest code via its own loader, outside
    // any host module registry).
    //
    // A naive address-range check (e.g. "ptr >= 0x800000000") was
    // tempting but wrong: PIE+ASLR puts host code at high addresses
    // too. dladdr is the right primitive.
    // ========================================================================

    /// Returns true if `ptr` points into guest-loaded code, false if
    /// it points into a loaded host module (executable or shared
    /// library), or is null. O(1) amortized via the OS's module
    /// tables, thread-safe.
    [[nodiscard]] static bool IsGuestPointer(const void* ptr) noexcept;

private:
    std::unique_ptr<BlockCache> block_cache_;
    std::unique_ptr<CodeCache> code_cache_;
    std::unique_ptr<Gateway> gateway_;
    std::unique_ptr<Lifter> lifter_;
};

} // namespace Core::Runtime
