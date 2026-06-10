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
    ///
    /// MUST be called with the calling thread's code-cache claim held
    /// (the dispatcher's guest path acquires it before the lookup). On a
    /// full cache this drops the claim, performs a stop-the-world recycle
    /// via FlushCachesForRecycle, re-claims, and retries (bounded).
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
    void CallGuestOnCallerStack(GuestState& caller, VAddr guest_fn,
                                SetupFn setup, void* user_data);

    /// Convenience: invoke a guest function on the caller's stack
    /// with up to 6 pointer-sized integer arguments. Returns the
    /// callback's return value (caller.gpr[0] after the call).
    u64 CallGuestSimpleOnCallerStack(GuestState& caller, VAddr guest_fn,
                                     u64 a0 = 0, u64 a1 = 0, u64 a2 = 0,
                                     u64 a3 = 0, u64 a4 = 0, u64 a5 = 0);

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
    u64 InvokeGuestCallback(VAddr guest_fn,
                            u64 a0 = 0, u64 a1 = 0, u64 a2 = 0,
                            u64 a3 = 0, u64 a4 = 0, u64 a5 = 0);

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
    /// Stop-the-world code-cache recycle. Serialized across flushing
    /// threads; `observed_generation` is the cache generation the caller
    /// saw when its compile failed -- if it no longer matches, another
    /// thread already recycled and this is a no-op. Caller must NOT hold a
    /// code-cache claim (it would deadlock waiting on itself). See the
    /// flush-handshake design block in runtime.cpp for the full protocol.
    void FlushCachesForRecycle(u64 observed_generation);

    std::unique_ptr<BlockCache> block_cache_;
    std::unique_ptr<CodeCache> code_cache_;
    std::unique_ptr<Gateway> gateway_;
    std::unique_ptr<Lifter> lifter_;
};

// ============================================================================
// Crash-diagnostic support.
//
// When the process takes an unhandled fault (SIGSEGV/access violation,
// SIGILL, etc.), the signal/SEH handler runs in a severely constrained
// async context: the faulting thread is frozen mid-instruction, the
// heap may be inconsistent, and almost nothing is safe to call. The raw
// host fault address alone is nearly useless for triage — it doesn't
// say whether the fault was inside JIT-compiled guest code, inside an
// HLE shim, or in plain host code, nor which guest instruction was
// executing.
//
// DescribeFaultContext answers those questions using ONLY async-signal-
// safe operations: it reads a thread-local pointer (the live GuestState,
// if this thread is mid-Run), reads integer fields from it, and does a
// pointer-range comparison against the code cache. No allocation, no
// locks, no fmt/spdlog calls — the returned POD carries only integers,
// which the caller may format after deciding it's safe to do so.
// ============================================================================
struct FaultContext {
    /// True if this thread was executing inside the CPU runtime (i.e.
    /// tl_current_guest_state was set) at the time of the fault.
    bool in_runtime = false;
    /// True if `host_addr` (the faulting code/instruction address the
    /// caller passed in) lies inside the JIT code cache — meaning the
    /// fault happened while *executing* lifted guest code, as opposed
    /// to inside an HLE shim or while dereferencing a bad guest pointer.
    bool in_jit_code = false;
    /// The guest RIP the runtime was executing (GuestState::rip) when
    /// the fault occurred. Valid only if `in_runtime` is true. This is
    /// the single most useful datum for triage: it pins the fault to a
    /// specific guest instruction address regardless of where in the
    /// host the deref actually landed.
    u64 guest_rip = 0;
    /// The guest exit_reason field at fault time (valid if in_runtime).
    /// Helps distinguish "faulted mid-block" from a stale state.
    u32 guest_exit_reason = 0;
    /// Snapshot of the 16 guest GPRs (canonical AMD64 order: RAX=0,
    /// RCX=1, RDX=2, RBX=3, RSP=4, RBP=5, RSI=6, RDI=7, R8..R15=8..15)
    /// at fault time. Valid only if `in_runtime`. This is what turns
    /// "faulted dereferencing a wild address" into "which register held
    /// the bad pointer" — the caller can match the faulting data
    /// address against these to identify the base/index register that
    /// carried garbage, and trace it back to its producer.
    u64 guest_gpr[16] = {};
    /// True if the guest_gpr snapshot was populated.
    bool have_gprs = false;
    /// Decoded mnemonic of the faulting guest instruction (the bytes at
    /// guest_rip), as a Zydis mnemonic id, plus its length. Valid only
    /// if `faulting_insn_decoded`. This distinguishes, e.g., a load that
    /// brought a bad pointer in (mov reg,[mem]) from a store/use of a
    /// locally-miscomputed pointer (mov [reg],... / op on [reg]) —
    /// resolving "did the value come from memory or did we compute it
    /// wrong" without needing the guest binary on hand.
    u32 faulting_mnemonic = 0;     // ZydisMnemonic (0 = ZYDIS_MNEMONIC_INVALID)
    u8 faulting_insn_length = 0;
    bool faulting_insn_decoded = false;
    /// First bytes at the host fault PC, captured raw. This is the
    /// ultimate fallback: even when decode fails (wrong machine mode,
    /// mid-instruction PC, or a build/version mismatch), the raw bytes
    /// let a human disassemble offline. `raw_byte_count` is how many
    /// were captured (0 if host_addr wasn't a readable in-cache addr).
    u8 faulting_raw_bytes[16] = {};
    u8 raw_byte_count = 0;
    /// Host bytes immediately PRECEDING the fault PC. The faulting
    /// instruction usually only consumes the bad pointer; the code that
    /// computed it sits just before. Disassembling this window reveals
    /// the lifter sequence that produced the address (e.g. where a stray
    /// high bit is introduced). `pre_byte_count` is how many were grabbed.
    u8 pre_fault_bytes[32] = {};
    u8 pre_byte_count = 0;
    /// Decoded shape of the faulting memory access (valid when
    /// `have_mem_operand`). `mem_base_reg` is the ZydisRegister used as
    /// the EA base; `mem_is_read`/`mem_is_write` say whether the faulting
    /// access was a load or a store. A read means the bad value/pointer
    /// was being brought in FROM the address; a write means an
    /// already-bad pointer was used as the destination.
    u32 mem_base_reg = 0;          // ZydisRegister (0 = none)
    bool mem_is_read = false;
    bool mem_is_write = false;
    bool have_mem_operand = false;
};

/// Compute fault context for the CURRENT thread. `host_addr` is the
/// host address the fault reported (the code/instruction pointer, e.g.
/// EXCEPTION_ADDRESS on Windows or the context RIP on POSIX). May be
/// null if the caller doesn't have it; the in_jit_code result is then
/// simply false. Async-signal-safe; never throws, allocates, or locks.
[[nodiscard]] FaultContext DescribeFaultContext(const void* host_addr) noexcept;

} // namespace Core::Runtime
