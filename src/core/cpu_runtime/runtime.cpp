// SPDX-FileCopyrightText: Copyright 2026 shadPS4 Emulator Project
// SPDX-License-Identifier: GPL-2.0-or-later

#include "core/cpu_runtime/runtime.h"

#include <array>
#include <atomic>
#include <bit>
#include <condition_variable>
#include <cstddef>
#include <cstdlib>
#include <cstring>
#include <mutex>
#include <type_traits>

#include "common/assert.h"
#include "common/arch.h"
#include "common/logging/log.h"
#include "core/cpu_runtime/block_cache.h"
#include "core/cpu_runtime/code_cache.h"
#include "core/cpu_runtime/gateway/gateway.h"
#include "core/cpu_runtime/lifter/lifter.h"
#include "core/cpu_runtime/hle_registry.h"
#include "core/cpu_runtime/runtime_diagnostics.h"
#include "core/signals.h"
#include "core/tls.h"

#ifdef ARCH_X86_64
#include <Zydis/Zydis.h>
#include <xmmintrin.h> // _mm_getcsr/_mm_setcsr (guest-rounding swap)
#endif

// Platform headers for IsGuestPointer's host-module query. These MUST be at
// file scope: when included inside `namespace Core::Runtime` they bind Win32/COM
// types like _GUID/IID into the namespace, breaking QueryInterface/Resolve.
// (Previously this was masked because windows.h was pulled in earlier at global
// scope in the monolithic TU; as a standalone library it no longer is.)
#ifdef _WIN32
#include <windows.h>
#else
#include <dlfcn.h>
#endif

namespace Core::Runtime {

namespace {

// ============================================================================
// Compile-time invariants
// ============================================================================
//
// The dispatcher's HLE bridge unmarshals integer arg slots directly
// from `state->gpr[]` and stores the SysV return value into
// `state->gpr[0]`. The XMM-arg path indexes `state->ymm[]` in
// strides of 4 (one YMM = 4 u64 lanes). Both depend on a specific
// GuestState layout that's a convention but isn't enforced by the
// header. These asserts pin down the relationship:

// The dispatcher reads SysV arg slots as gpr[7,6,2,1,8,9] = (rdi,
// rsi, rdx, rcx, r8, r9) and writes the return to gpr[0] = rax.
// These match the canonical AMD64 register-number ordering Zydis
// uses (verified separately in the lifter), so reordering the slot
// numbering would break a long chain of assumptions across the
// whole runtime.
//
// We don't have a header constant for "RAX index" — gpr is a flat
// array. So we assert against GuestState::gpr's static type. If
// someone replaces it with a non-array (e.g., separate named
// fields), this breaks the build at this assert and forces the
// dispatcher to be updated alongside.
static_assert(std::is_same_v<decltype(GuestState::gpr), std::array<u64, 16>>,
              "GuestState::gpr must be std::array<u64,16> in canonical "
              "AMD64 order; the dispatcher's HLE bridge depends on this");

// YMM/XMM file: 32 ymm registers × 4 u64 lanes each. The HLE
// bridge reads state.ymm[i*4] as the low 64 bits of xmm[i] for
// i=0..7. If the lane count per ymm register ever changes (e.g.,
// AVX-512 expansion that re-uses this field), the stride would
// need to change too.
static_assert(std::is_same_v<decltype(GuestState::ymm),
                             std::array<u64, 32 * 4>>,
              "GuestState::ymm must hold 32 ymm registers of 4 u64 lanes; "
              "the HLE bridge's XMM marshaling assumes this layout");

// ---------------- Named SysV arg-slot indices ----------------
//
// These match the canonical AMD64 register numbering (verified in
// the lifter) but using named constants makes the dispatcher's
// intent legible at the call site and centralises the convention.
constexpr int kSysvArg0 = 7;  // RDI
constexpr int kSysvArg1 = 6;  // RSI
constexpr int kSysvArg2 = 2;  // RDX
constexpr int kSysvArg3 = 1;  // RCX
constexpr int kSysvArg4 = 8;  // R8
constexpr int kSysvArg5 = 9;  // R9
constexpr int kSysvRet  = 0;  // RAX
constexpr int kGuestRsp = 4;  // RSP (host stack of the guest)

// ---------------- end of compile-time invariants ----------------

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

// ============================================================================
// Code-cache flush handshake (stop-the-world recycling)
// ============================================================================
//
// PROBLEM. The code cache is a bump allocator; on overflow it is recycled by
// resetting the bump pointer, which invalidates EVERY host pointer in it.
// Multiple guest threads run JIT code concurrently. Recycling while another
// thread is (a) executing a block inside the cache, or (b) between a block-
// cache lookup/compile and the moment it jumps into the result, overwrites or
// dangles code that thread is about to run. The old comment ("single compiler
// thread, no other thread is dispatching") described a world this runtime no
// longer lives in -- pthread.cpp launches arbitrarily many guest threads
// through Run().
//
// SCHEME. A thread holds a CLAIM for exactly the window in which it may touch
// code-cache memory:
//
//     dispatcher guest path:  AcquireClaim()
//         -> BlockCache lookup / Lifter compile / BlockCache insert
//         -> return host_ptr to the gateway
//         -> gateway jmps into the block; the block executes; jmp r14
//     next dispatcher entry:  ReleaseClaim()
//
// Run() also releases after the gateway returns, covering the fatal-exit path
// (jmp r15) that bypasses the dispatcher. Everything else a thread can do --
// the HLE bridge (including arbitrarily long blocking host calls), sentinel
// handling, watchdog, sitting parked -- holds NO claim, so a thread asleep in
// sceKernelUsleep for a minute cannot stall a flush; it simply parks at
// AcquireClaim when it next wants a block, if a flush is mid-flight.
//
// FLUSH. The thread that hits a full cache releases its own claim, then:
//   1. takes g_flush_mutex (serializes flushers),
//   2. re-checks the generation counter -- if another thread already
//      recycled, skip (its blocks are gone; just recompile),
//   3. sets g_flush_pending (new claims now park), waits for the active
//      claim count to drain to zero,
//   4. clears the BlockCache, THEN resets the CodeCache bump pointer
//      (order: no lookup may observe a mapping into recycled memory),
//   5. bumps the generation, clears pending, wakes everyone.
//
// HOT-PATH COST. One thread-local bool test per dispatcher entry; one
// relaxed-ish fetch_add + two loads per block transition in the guest path.
// The mutex/condvar are touched only when a flush is actually pending (rare:
// once per 64 MB of emitted code) or on the last-claim-out wakeup.
//
// FUTURE. Block chaining (blocks jumping directly to each other without
// returning to the dispatcher) breaks the "every transition passes the
// dispatcher" premise this relies on; when chaining lands, claims must be
// released by the chained code itself or the handshake replaced with a
// signal-based suspend. Leave this comment as the tripwire.
// ============================================================================

// MEMORY ORDERING NOTE. The pending/claims pair forms two Dekker-shaped
// store-buffer races:
//   flusher:  W(pending=1)  then  R(claims)     -- "anyone still in?"
//   releaser: W(claims-1)   then  R(pending)    -- "should I wake a flusher?"
//   claimant: W(claims+1)   then  R(pending)    -- "did a flush just start?"
// With acquire/release only, BOTH sides can read stale ("flusher sees the old
// claim count AND releaser sees pending==false") -- the missed notify parks
// the flusher forever with claims at zero. seq_cst on these four access
// kinds forbids that outcome (store-buffer interleaving is impossible under
// a single total order). Cost on the hot path is nil on x86 (a seq_cst LOAD
// is a plain mov; the RMWs are lock-prefixed either way) and a stronger
// barrier flavor on ARM64, paid once per block transition.
std::mutex g_flush_mutex;
std::condition_variable g_flush_cv;
std::atomic<u64> g_active_claims{0};
std::atomic<bool> g_flush_pending{false};
std::atomic<u64> g_cache_generation{0};
thread_local bool tl_has_claim = false;

// ============================================================================
// Guest-thread progress registry (watchdog cross-thread verdicts)
// ============================================================================
//
// The stuck-RIP watchdog (see DispatcherTrampoline) needs to distinguish two
// situations that look identical from one thread's chair:
//
//   (a) a thread spin-reading a memory flag that ANOTHER guest thread will
//       eventually write (raw spinlock, no register mutation per pass) --
//       a CORRECT program under contention, must not be killed;
//   (b) the whole guest frozen -- every thread spinning or wedged -- a true
//       hang where a fatal exit with register dumps is the useful outcome.
//
// Each thread that enters Run() takes a cache-line-sized slot here and bumps
// its transition counter (relaxed, on an exclusively-owned line: a couple of
// cycles) at every dispatcher entry; the HLE bridge marks the slot while the
// thread is parked inside a host call. A stuck thread's verdict then reads
// the OTHER slots: anyone advancing, or anyone parked in HLE (presumed
// legitimately waiting -- sema, sleep, io), means the system is alive and
// the watchdog demotes to log-and-continue. Fatal fires only when this
// thread is provably frozen AND every other guest thread is too. The
// single-threaded case degenerates to the original behavior exactly (no
// other slots -> fatal), which the true-spin runtime test pins.
//
// Slots are registered on the OUTERMOST Run() of a thread (nested Run via
// caller-stack callbacks reuses the slot; depth-counted) and released at its
// exit. The fixed pool bounds the cost; a thread beyond the pool runs
// without a slot -- it is invisible to other threads' verdicts, and its own
// watchdog demotes to log-and-continue rather than risk a false kill on
// partial information (degradation documented over silent wrongness).

struct alignas(64) ThreadProgressSlot {
    std::atomic<u64> transitions{0};
    std::atomic<bool> in_hle{false};
    std::atomic<bool> active{false};
};
constexpr size_t kMaxProgressSlots = 512; // 32 KiB static; >n guest threads
std::array<ThreadProgressSlot, kMaxProgressSlots> g_progress_slots{};
std::mutex g_progress_registry_mutex; // register/release only (rare)
thread_local ThreadProgressSlot* tl_progress_slot = nullptr;
thread_local u32 tl_run_depth = 0;

ThreadProgressSlot* AcquireProgressSlot() {
    std::lock_guard lk{g_progress_registry_mutex};
    for (auto& slot : g_progress_slots) {
        if (!slot.active.load(std::memory_order_relaxed)) {
            slot.transitions.store(0, std::memory_order_relaxed);
            slot.in_hle.store(false, std::memory_order_relaxed);
            slot.active.store(true, std::memory_order_release);
            return &slot;
        }
    }
    return nullptr; // pool exhausted: run slotless (see block comment)
}

void ReleaseProgressSlot(ThreadProgressSlot* slot) {
    if (slot == nullptr) {
        return;
    }
    std::lock_guard lk{g_progress_registry_mutex};
    slot->in_hle.store(false, std::memory_order_relaxed);
    slot->active.store(false, std::memory_order_release);
}

/// Sum of all OTHER active threads' transition counters, plus liveness
/// census. Per-slot counters are monotonic, so cross-thread progress
/// strictly changes the sum -- but the SUM itself is not monotonic: a slot
/// RELEASE (thread exit) removes its contribution. The consequence is
/// bounded and self-correcting: a thread-exit between a verdict's two
/// samples reads as "others changed" and demotes that one episode; the
/// next episode re-snapshots the post-exit sum, and a genuinely frozen
/// remainder then goes fatal one ~10 M-transition cycle later than ideal.
/// Erring toward one extra demotion beats erring toward a false kill.
struct OthersSnapshot {
    u64 transition_sum = 0;
    u32 active_count = 0;
    u32 in_hle_count = 0;
};
OthersSnapshot SnapshotOthers(const ThreadProgressSlot* self) noexcept {
    OthersSnapshot s;
    for (const auto& slot : g_progress_slots) {
        if (&slot == self || !slot.active.load(std::memory_order_acquire)) {
            continue;
        }
        ++s.active_count;
        s.transition_sum += slot.transitions.load(std::memory_order_relaxed);
        if (slot.in_hle.load(std::memory_order_relaxed)) {
            ++s.in_hle_count;
        }
    }
    return s;
}

/// Drop this thread's claim, if held. Called at every dispatcher entry and at
/// Run() exit. If this was the last claim out while a flusher is waiting,
/// wake it. Hot path when no flush is pending: one branch + one fetch_sub.
void ReleaseClaim() noexcept {
    if (!tl_has_claim) {
        return;
    }
    tl_has_claim = false;
    const u64 prev = g_active_claims.fetch_sub(1, std::memory_order_seq_cst);
    if (prev == 1 && g_flush_pending.load(std::memory_order_seq_cst)) {
        // Last claim out with a flush waiting: the flusher sleeps on the cv
        // under g_flush_mutex; taking the lock here orders the notify after
        // its predicate re-check window.
        std::lock_guard lk{g_flush_mutex};
        g_flush_cv.notify_all();
    }
}

// ============================================================================
// Guest-rounding swap (x86 host)
// ============================================================================
//
// Guest LDMXCSR applies a SANITIZED value to the host MXCSR inside JIT code
// (see EmitMxcsr in the x86 lifter), so the SSE/AVX instructions we emit run
// under the guest's rounding mode and FTZ/DAZ — but host C++ (the dispatcher,
// the HLE bridge, every library call those make) must NEVER run under guest
// rounding: RC != nearest or FTZ inside printf/maths-adjacent host code is a
// recipe for untraceable numeric corruption. So the dispatcher swaps:
//
//   dispatcher entry:   guest -> HOST DEFAULT   (we just left JIT code)
//   handing out a block: HOST DEFAULT -> guest  (about to enter JIT code)
//   Run() after Enter:  guest -> HOST DEFAULT   (covers jmp-r15 fatal exits)
//
// The swap is gated on Sanitize(state->mxcsr) != default, so a guest that
// never touches MXCSR (or writes a default-equivalent) pays one load + mask
// + compare per block transition and zero ldmxcsr. Sanitization keeps RC
// (bits 14:13), FTZ (15), DAZ (6) and forces all exception masks set; the
// constants MUST match EmitMxcsr in lifter_x86_host.cpp.
//
// ARM64 host: not yet wired — the equivalent is mapping guest MXCSR.RC/FTZ
// onto FPCR.RMode/FZ with the same swap discipline. Until then the arm64
// backend records guest MXCSR writes without applying them (its pre-existing
// behavior), which is a documented fidelity gap, not a regression.

#ifdef ARCH_X86_64
constexpr u32 kHostDefaultMxcsr = 0x1F80;
constexpr u32 kGuestMxcsrMask = (3u << 13) | (1u << 15) | (1u << 6); // RC|FTZ|DAZ

inline u32 SanitizeGuestMxcsr(u32 guest) noexcept {
    return kHostDefaultMxcsr | (guest & kGuestMxcsrMask);
}

/// Entering host C++ from JIT context: put the host default back if the
/// guest's MXCSR could have been live.
inline void RestoreHostRounding(const GuestState* state) noexcept {
    if (SanitizeGuestMxcsr(state->mxcsr) != kHostDefaultMxcsr) {
        _mm_setcsr(kHostDefaultMxcsr);
    }
}

/// About to hand a block pointer to the gateway: arm the guest's rounding
/// for the JIT code that is about to run.
inline void ApplyGuestRounding(const GuestState* state) noexcept {
    const u32 s = SanitizeGuestMxcsr(state->mxcsr);
    if (s != kHostDefaultMxcsr) {
        _mm_setcsr(s);
    }
}
#else
inline void RestoreHostRounding(const GuestState*) noexcept {}
inline void ApplyGuestRounding(const GuestState*) noexcept {}
#endif

/// FNV-1a over the guest's vector and x87 state, for the watchdog's lazy
/// second-stage progress check. Reads ~1.1 KB; deliberately NOT called on
/// the per-transition hot path -- only at the two sampling points of a
/// suspected-stuck episode (see the watchdog below). A 64-bit digest
/// instead of a stored snapshot keeps the thread_local footprint at 8
/// bytes; the 2^-64 per-episode collision odds are noise next to the
/// equivalent aliasing the GPR comparison already accepts.
u64 HashGuestVectorState(const GuestState* state) noexcept {
    constexpr u64 kFnvOffset = 0xCBF29CE484222325ULL;
    constexpr u64 kFnvPrime = 0x100000001B3ULL;
    u64 h = kFnvOffset;
    const auto fold = [&](const void* p, size_t n) {
        const u8* b = static_cast<const u8*>(p);
        for (size_t i = 0; i < n; ++i) {
            h ^= b[i];
            h *= kFnvPrime;
        }
    };
    fold(state->ymm.data(), state->ymm.size() * sizeof(u64));
    fold(state->st.data(), state->st.size() * sizeof(u64));
    fold(&state->fpu_top, sizeof(state->fpu_top));
    fold(&state->fpu_tag, sizeof(state->fpu_tag));
    fold(&state->fpu_sw_cc, sizeof(state->fpu_sw_cc));
    fold(&state->fpu_cw, sizeof(state->fpu_cw));
    fold(&state->mxcsr, sizeof(state->mxcsr));
    return h;
}

/// Take a claim, parking while a flush is pending. The increment-then-
/// re-check dance closes the race where a flusher sets pending between our
/// pending load and our increment: if we lost that race we back out (undoing
/// the increment, waking the flusher if we were what it was waiting on) and
/// go around again.
void AcquireClaim() {
    for (;;) {
        if (g_flush_pending.load(std::memory_order_seq_cst)) {
            std::unique_lock lk{g_flush_mutex};
            g_flush_cv.wait(lk, [] {
                return !g_flush_pending.load(std::memory_order_seq_cst);
            });
        }
        g_active_claims.fetch_add(1, std::memory_order_seq_cst);
        if (!g_flush_pending.load(std::memory_order_seq_cst)) {
            tl_has_claim = true;
            return;
        }
        // Raced with a starting flush: back out and park. A transient claim
        // taken here is harmless to an in-progress flush -- we never touch
        // the cache while holding it -- and the prev==1 notify below covers
        // the case where the flusher was waiting on exactly this count.
        const u64 prev = g_active_claims.fetch_sub(1, std::memory_order_seq_cst);
        if (prev == 1) {
            std::lock_guard lk{g_flush_mutex};
            g_flush_cv.notify_all();
        }
    }
}

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

// ─────────────────────────────────────────────────────────────────────────
// DO NOT "SIMPLIFY" HostReturn. The two-field layout is load-bearing.
//
// The entire HLE return path depends on this struct being classified by the
// SysV AMD64 ABI into exactly two eightbytes — the first INTEGER (→ rax), the
// second SSE (→ xmm0) — so that a single call site can recover *both* a
// possible integer return and a possible floating return without knowing the
// callee's declared return type. Any of the following "cleanups" silently
// breaks that and produces garbage returns that are extremely hard to trace:
//
//   * Replacing it with a bare u64 (loses the xmm0 eightbyte; double-returning
//     HLE callees would have their result dropped).
//   * Reordering the members (the INTEGER eightbyte MUST be first so it maps
//     to rax; swapping puts the double in rax and the int in xmm0).
//   * Adding a third member, or widening either field past 8 bytes (pushes the
//     aggregate over two eightbytes → returned via hidden pointer in rdi, not
//     rax:xmm0, which this path does not implement).
//   * Making it non-trivial / non-standard-layout (e.g. adding a constructor),
//     which can change ABI classification on some toolchains.
//
// The static_asserts below pin the invariants the ABI trick relies on. If one
// trips, the fix is to restore the layout, NOT to relax the assert.
static_assert(std::is_standard_layout_v<HostReturn>,
              "HostReturn must be standard-layout for predictable SysV "
              "eightbyte classification (INTEGER+SSE → rax:xmm0).");
static_assert(std::is_trivially_copyable_v<HostReturn>,
              "HostReturn must be trivially copyable; a non-trivial type can "
              "be classified as MEMORY and returned via hidden pointer.");
static_assert(sizeof(HostReturn) == 16,
              "HostReturn must occupy exactly two eightbytes (16 bytes). A "
              "different size changes ABI return classification.");
static_assert(offsetof(HostReturn, rax) == 0,
              "The INTEGER eightbyte (rax) MUST be the first member so it maps "
              "to the rax return register, not xmm0.");
static_assert(offsetof(HostReturn, xmm0) == 8,
              "The SSE eightbyte (xmm0) MUST be the second member so it maps "
              "to the xmm0 return register.");
static_assert(std::is_same_v<decltype(HostReturn::rax), u64>,
              "The first eightbyte must be an integer type to classify INTEGER "
              "(→ rax).");
static_assert(std::is_same_v<decltype(HostReturn::xmm0), double>,
              "The second eightbyte must be a floating type to classify SSE "
              "(→ xmm0).");
// ─────────────────────────────────────────────────────────────────────────

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
///     is reading past the end of the guest stack mapping; the
///     reads are bounded by GuestState::stack_top (exact) when the
///     state came through CallGuest, with a same-page clamp as the
///     fallback for unknown / switched stacks -- see safe_read_qword
///     in the bridge for the full bound-selection rationale.
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
#if defined(ARCH_X86_64) || defined(__x86_64__) || defined(_M_X64)
    // x86-64 SysV: the HostHleFn return type {u64; double} is classified into
    // two eightbytes (INTEGER→rax, SSE→xmm0), so a single typed call recovers
    // both a possible integer and a possible floating return.
    auto fn = reinterpret_cast<HostHleFn>(host_fn);
    return fn(a0, a1, a2, a3, a4, a5,
              f0, f1, f2, f3, f4, f5, f6, f7,
              s0, s1, s2, s3, s4, s5, s6, s7);
#elif defined(__aarch64__)
    // AArch64 AAPCS: the {u64; double} struct-return trick does NOT apply — a
    // mixed aggregate returns in x0:x1 (both GPRs), not x0:d0, and the
    // x86-only sysv_abi attribute is ignored here. Instead we let the compiler
    // marshal all arguments via a normal typed call (AAPCS already routes the
    // integer args to x0..x7/stack and the double args to d0..d7), then capture
    // BOTH the integer return (x0) and the floating return (d0) — exactly the
    // two registers a callee of unknown return type may have written.
    //
    // The call is made through a pointer typed to return void so the compiler
    // emits a standard AAPCS call and arg setup; immediately afterward, inline
    // asm reads x0 and d0 before any other code can clobber them.
    using HostHleFnA64 = void (*)(u64, u64, u64, u64, u64, u64,
                                  double, double, double, double,
                                  double, double, double, double,
                                  u64, u64, u64, u64, u64, u64, u64, u64);
    auto fn = reinterpret_cast<HostHleFnA64>(host_fn);
    fn(a0, a1, a2, a3, a4, a5,
       f0, f1, f2, f3, f4, f5, f6, f7,
       s0, s1, s2, s3, s4, s5, s6, s7);
    HostReturn ret;
    // Capture x0 (integer return) and d0 (floating return) post-call.
    register u64 x0_val asm("x0");
    register double d0_val asm("d0");
    asm volatile("" : "=r"(x0_val), "=w"(d0_val));
    ret.rax = x0_val;
    ret.xmm0 = d0_val;
    return ret;
#else
#error "CallHostFromGuest: unsupported host architecture"
#endif
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
    // We just left JIT code (or this is the first entry of a Run, in which
    // case this is a no-op): drop the claim taken when the previous block's
    // pointer was handed out. From here until the guest path's AcquireClaim
    // below, this thread touches no code-cache memory, so a concurrent
    // flush is safe -- including for the whole duration of any HLE bridge
    // call this invocation makes.
    ReleaseClaim();

    // Publish forward progress for OTHER threads' watchdog verdicts: one
    // relaxed increment on this thread's own cache line per block
    // transition. (Self-progress is judged by the register/vector checks
    // below, not this counter -- a spinning thread bumps it too.)
    if (tl_progress_slot != nullptr) {
        tl_progress_slot->transitions.fetch_add(1, std::memory_order_relaxed);
    }

    // We may arrive here with the guest's MXCSR live (a block ran after a
    // guest LDMXCSR). Host C++ -- this function, the HLE bridge, anything
    // they call -- must run under the host default. No-op when the guest's
    // value sanitizes to the default. See the guest-rounding swap block.
    RestoreHostRounding(state);

    // Hot path: the dispatcher trampoline is called once per JIT block
    // exit, which in tight loops (memset/memcpy/scan) is once per loop
    // iteration. The per-entry diagnostics use LOG_TRACE, which expands
    // to (void(0)) outside _DEBUG — so in release builds this hot path
    // touches no logging machinery at all (important: the dispatcher is
    // reached from a JIT-emitted gateway frame; on the x86/Windows host
    // an spdlog/fmt SEH stack walk could fault there, and on macOS/arm64
    // there is no SEH walker but we still want zero hot-path cost).
    //
    // In _DEBUG, gate the per-entry trace behind a "RIP changed since
    // last entry" check so back-edge loops stay quiet: new RIPs get the
    // full A/B/C dump; repeat entries to the same RIP are silent. This
    // keeps debug tracing useful without drowning tight loops.
    static thread_local u64 prev_logged_rip = 0;
    const u64 cur_rip = state->rip;
    const bool rip_changed = (cur_rip != prev_logged_rip);
    if (rip_changed) {
        prev_logged_rip = cur_rip;
        LOG_TRACE(Core, "Dispatcher: enter state={} rip={:#x}",
                  static_cast<void*>(state), cur_rip);
    }

    Runtime* rt = tl_active_runtime;
    if (rip_changed) {
        LOG_TRACE(Core, "Dispatcher: active runtime={}", static_cast<void*>(rt));
    }

    ASSERT_MSG(rt != nullptr, "Dispatcher called with no active runtime");

    // Stuck-RIP watchdog: detect TRUE livelock — the same block re-entered
    // with no forward progress at all. Progress is judged by the guest state
    // changing (below): a long but advancing loop (memset, zero-fill, timeout
    // poll) repeats the same RIP every iteration yet mutates a register each
    // pass, so it keeps resetting the counter and never trips, no matter how
    // many iterations it runs. The threshold then only has to ignore brief
    // identical-state stutters; 10 M re-entries with byte-identical guest
    // state is unambiguously a hang (a spin on a memory/sync condition that
    // nothing is advancing), not a slow loop.
    //
    // The watchdog fires regardless of build flavor (it's a real fault
    // condition, not a trace), so it uses LOG_ERROR. It runs at most
    // once per stuck site (the == comparison, not >=), so the logging
    // cost is a non-issue.
    {
        static thread_local u64 last_rip = 0;
        static thread_local u64 same_rip_hits = 0;
        static thread_local std::array<u64, 16> last_gpr{};
        static thread_local u64 last_rflags = 0;
        // Forward progress, stage 1 (every transition, cheap): any change in
        // the integer register file or flags between consecutive entries at
        // the same block. A true spin reads memory and branches without
        // mutating a register, so its state is identical every pass and the
        // counter climbs; an advancing loop changes a counter/pointer and
        // resets it.
        //
        // Stage 2 (lazy, two reads per suspected episode): loops whose
        // induction state lives ENTIRELY in vector or x87 registers -- a
        // convergence loop accumulating in xmm0 with ucomisd producing the
        // same flag pattern every pass, SIMD DSP/PRNG kernels, x87
        // accumulation -- are invisible to stage 1 and were previously
        // KILLED as stuck at 10 M iterations. Comparing the ~1.1 KB of
        // vector state per transition would tax the hot path, so it is
        // hashed only twice: a digest at the half-threshold, re-checked at
        // the threshold. Any difference proves vector-only progress and
        // resets the episode (it re-arms automatically if the counter
        // climbs back). Cost: two 1.1 KB hashes per ~10 M transitions of a
        // suspected loop; zero when loops advance through integer state.
        //
        // Stage 3 (cross-thread, evaluated only at the threshold): a thread
        // spinning on a memory flag that ANOTHER guest thread will write
        // (raw spinlock under contention) is byte-identical state from this
        // chair yet a CORRECT program. The verdict consults the progress
        // registry: if any other guest thread advanced during the episode,
        // or is parked inside an HLE call (sema/sleep/io -- presumed
        // legitimately waiting), the system is alive and the watchdog
        // DEMOTES to a one-per-RIP LOG_ERROR and continues. FATAL is
        // reserved for whole-system freeze: this thread provably frozen
        // (integer + vector state byte-identical) and every other guest
        // thread equally silent. Killing a correct program is strictly
        // worse than logging a hang. Slotless threads (registry pool
        // exhausted) demote unconditionally -- partial information must
        // not kill.
        const bool advanced =
            (last_gpr != state->gpr) || (last_rflags != state->rflags);
        if (cur_rip == last_rip && !advanced) {
            ++same_rip_hits;
        } else {
            same_rip_hits = 1;
        }
        last_rip = cur_rip;
        last_gpr = state->gpr;
        last_rflags = state->rflags;
        constexpr u64 kStuckThreshold = 10'000'000;
        constexpr u64 kSuspectThreshold = kStuckThreshold / 2;
        static thread_local u64 suspect_vec_hash = 0;
        static thread_local u64 suspect_others_sum = 0;
        if (same_rip_hits == kSuspectThreshold) {
            suspect_vec_hash = HashGuestVectorState(state);
            suspect_others_sum =
                SnapshotOthers(tl_progress_slot).transition_sum;
        }
        if (same_rip_hits == kStuckThreshold &&
            HashGuestVectorState(state) != suspect_vec_hash) {
            // Vector/x87 state moved between the two sampling points:
            // the loop is advancing, just not through integer state.
            same_rip_hits = 1;
        }
        if (same_rip_hits == kStuckThreshold) {
            // This thread is provably frozen. Cross-thread verdict (stage 3).
            const OthersSnapshot others = SnapshotOthers(tl_progress_slot);
            const bool others_advanced =
                others.transition_sum != suspect_others_sum;
            const bool others_waiting = others.in_hle_count > 0;
            const bool slotless = (tl_progress_slot == nullptr);
            if (others_advanced || others_waiting || slotless) {
                // System alive (or unknowable): demote. Log once per RIP so
                // a minutes-long contended spin doesn't flood the log at one
                // line per ~10 M transitions.
                static thread_local u64 last_warned_spin_rip = 0;
                if (last_warned_spin_rip != cur_rip) {
                    last_warned_spin_rip = cur_rip;
                    LOG_ERROR(Core,
                              "Dispatcher: thread spin-waiting at {:#x} ({} "
                              "re-entries, registers/vectors byte-identical) "
                              "but the system is alive ({} other guest "
                              "threads: progressed={}, parked-in-HLE={}{}); "
                              "continuing",
                              cur_rip, same_rip_hits, others.active_count,
                              others_advanced, others.in_hle_count,
                              slotless ? ", self slotless" : "");
                }
                same_rip_hits = 1; // re-arm; re-verdicts every ~10 M
            }
        }
        if (same_rip_hits == kStuckThreshold) {
            LOG_ERROR(Core,
                      "Dispatcher: STUCK at {:#x} after {} re-entries with "
                      "no forward progress (integer AND vector/x87 state "
                      "byte-identical across the episode; no other guest "
                      "thread progressed or is waiting in HLE -- whole-"
                      "system freeze); dumping guest GPRs and exiting "
                      "fatally",
                      cur_rip, same_rip_hits);
            static const char* const kGprNames[16] = {
                "rax","rcx","rdx","rbx","rsp","rbp","rsi","rdi",
                "r8","r9","r10","r11","r12","r13","r14","r15",
            };
            for (int i = 0; i < 16; ++i) {
                LOG_ERROR(Core, "  {} = {:#x}", kGprNames[i], state->gpr[i]);
            }
            LOG_ERROR(Core, "  rflags = {:#x}", state->rflags);
            state->exit_reason = static_cast<u32>(ExitReason::UnsupportedInstruction);
            return nullptr;
        }
    }

    // Corruption-investigation instrumentation. These observers read the
    // guest state at this block boundary and log; they never alter
    // execution. The whole subsystem compiles out unless
    // SHADPS4_RUNTIME_DIAGNOSTICS is defined (see runtime_diagnostics.h),
    // so in normal builds the following calls vanish entirely and the hot
    // path carries zero cost. Order matters: RecordBlockBoundary must run
    // before CheckRegisterCorruption, which consumes the snapshot and ring
    // it records.
    Diagnostics::AnnounceOnce(cur_rip);
    Diagnostics::RecordBlockBoundary(state, cur_rip);
    Diagnostics::CheckRegisterCorruption(rt, state, cur_rip);

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
            // so we can log it. If the host fn crashes, this trace
            // line will be the last thing emitted — telling us
            // *which* host fn crashed and *which* guest block
            // called it.
            //
            // Logging here uses LOG_TRACE for the per-call dump (the
            // common case, compiled out in release so the bridge hot
            // path is free) and LOG_WARNING only for the unregistered-
            // target case (rare, a real diagnostic worth keeping in
            // release). The dispatcher is reached from a JIT-emitted
            // gateway frame; LOG_TRACE being (void(0)) in release means
            // the per-call path never invokes spdlog/fmt there, and on
            // the macOS/arm64 target there is no SEH walker to trip.
            const u64 guest_rsp = state->gpr[kGuestRsp];

            // Reading the guest stack (return address + spilled args) can
            // fault if guest_rsp sits near the top of a mapped page and the
            // next page is unmapped — the bridge would then crash inside the
            // marshalling reads rather than in the called function.
            //
            // BOUND SELECTION. SysV stack args live in the CALLER's frame at
            // [rsp+8 ..]; whenever an argument actually exists there, that
            // memory is mapped by construction (the caller wrote it). The
            // only range a speculative read can fault on is PAST THE TOP OF
            // THE STACK MAPPING -- reachable exactly when the callee takes
            // fewer stack args than the 8 we read unconditionally AND the
            // call sits in the outermost frames, within 72 bytes of the top.
            //
            //   - Preferred bound: state->stack_top, the true first-byte-
            //     past-the-stack recorded by CallGuest. Exact: no fault is
            //     possible at or below it, and no legitimate argument can
            //     live above it. Used when nonzero and consistent with rsp.
            //
            //   - Fallback (stack_top == 0, or rsp above it -- e.g. the
            //     guest switched stacks via sceFiber/ucontext, which we do
            //     not track): the old same-page clamp. It can never fault
            //     (rsp's own page is mapped: the guest is executing on it)
            //     but it FALSE-ZEROES real 7th+ arguments whenever rsp
            //     lands in the top 72 bytes of any page -- a 64-in-4096,
            //     stack-depth-dependent argument-corruption heisenbug. That
            //     is why the exact bound is preferred whenever available.
            //
            // A VMA/MM query would subsume both, but the bridge runs from a
            // JIT gateway frame where taking the memory manager's mutex is
            // unsafe (same reason this path avoids spdlog). If a LOCK-FREE
            // "is this address mapped" query becomes available, this is the
            // place to use it -- it would also cover switched stacks, where
            // the exact bound degrades to no protection above the fiber's
            // (unknown) top.
            constexpr u64 kPageSize = 0x1000;
            const u64 read_limit =
                (state->stack_top != 0 && guest_rsp < state->stack_top)
                    ? state->stack_top
                    : (guest_rsp & ~(kPageSize - 1)) + kPageSize;
            const auto safe_read_qword = [&](u64 addr) -> u64 {
                if (addr < guest_rsp || addr + sizeof(u64) > read_limit) {
                    // Below RSP or past the bound: don't touch it.
                    return 0;
                }
                return *reinterpret_cast<const u64*>(addr);
            };

            const u64 guest_return_addr = safe_read_qword(guest_rsp);

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
            // signature. Each read is bounded by safe_read_qword above
            // (the recorded stack top when known, page clamp otherwise):
            // slots past the bound default to 0 rather than faulting.
            const u64 s0 = safe_read_qword(guest_rsp + 1 * sizeof(u64));
            const u64 s1 = safe_read_qword(guest_rsp + 2 * sizeof(u64));
            const u64 s2 = safe_read_qword(guest_rsp + 3 * sizeof(u64));
            const u64 s3 = safe_read_qword(guest_rsp + 4 * sizeof(u64));
            const u64 s4 = safe_read_qword(guest_rsp + 5 * sizeof(u64));
            const u64 s5 = safe_read_qword(guest_rsp + 6 * sizeof(u64));
            const u64 s6 = safe_read_qword(guest_rsp + 7 * sizeof(u64));
            const u64 s7 = safe_read_qword(guest_rsp + 8 * sizeof(u64));

            if (name.empty()) {
                LOG_WARNING(Core,
                    "Bridge: unregistered host={:#x} ret={:#x} | "
                    "rdi={:#x} rsi={:#x} rdx={:#x} rcx={:#x} r8={:#x} r9={:#x}",
                    host_fn, guest_return_addr,
                    state->gpr[7], state->gpr[6], state->gpr[2],
                    state->gpr[1], state->gpr[8], state->gpr[9]);
            } else {
                // fmt prints a std::string_view natively via {} — no
                // null terminator or %.*s width dance needed.
                LOG_TRACE(Core,
                    "Bridge: call {} host={:#x} ret={:#x} | "
                    "rdi={:#x} rsi={:#x} rdx={:#x} rcx={:#x} r8={:#x} r9={:#x}",
                    name, host_fn, guest_return_addr,
                    state->gpr[7], state->gpr[6], state->gpr[2],
                    state->gpr[1], state->gpr[8], state->gpr[9]);
            }
            // XMM and stack args as TRACE-level detail (hex bit patterns
            // — easier to eyeball "this is 1.0" vs "this is garbage").
            // These are pure diagnostics, dropped in release; the
            // actionable unregistered-target signal above is WARNING.
            LOG_TRACE(Core,
                "Bridge:   xmm0={:#x} xmm1={:#x} xmm2={:#x} xmm3={:#x} "
                "xmm4={:#x} xmm5={:#x} xmm6={:#x} xmm7={:#x}",
                state->ymm[0], state->ymm[4], state->ymm[8], state->ymm[12],
                state->ymm[16], state->ymm[20], state->ymm[24], state->ymm[28]);
            LOG_TRACE(Core,
                "Bridge:   stk0={:#x} stk1={:#x} stk2={:#x} stk3={:#x} "
                "stk4={:#x} stk5={:#x} stk6={:#x} stk7={:#x}",
                s0, s1, s2, s3, s4, s5, s6, s7);

            // Unregistered host addresses get a loud WARNING from
            // the log block above, but the call still proceeds. The
            // bridge's job is to mechanically translate a guest CALL
            // into a host function invocation; whether HleRegistry
            // has a name for the target is a diagnostic concern, not
            // a gating one. (An earlier revision short-circuited
            // unregistered targets to rax=0 as a defense against an
            // unrelated spdlog/fmt SEH-walk crash in LOG_ERROR; that
            // crash is handled now — the hot-path logging here is
            // LOG_TRACE which is absent in release, and the one
            // release-visible call is the LOG_WARNING above — so the
            // short-circuit was removed. Gating host calls on prior
            // registration is the wrong contract for a JIT bridge.)
            // Mark this thread parked-in-HLE for the duration of the host
            // call: a thread blocked here (sema wait, sleep, io) is
            // presumed legitimately waiting, and other threads' watchdog
            // verdicts treat its presence as "system alive" even though
            // its transition counter is static.
            if (tl_progress_slot != nullptr) {
                tl_progress_slot->in_hle.store(true, std::memory_order_relaxed);
            }
            HostReturn ret = CallHostFromGuest(
                host_fn,
                state->gpr[kSysvArg0],   // RDI
                state->gpr[kSysvArg1],   // RSI
                state->gpr[kSysvArg2],   // RDX
                state->gpr[kSysvArg3],   // RCX
                state->gpr[kSysvArg4],   // R8
                state->gpr[kSysvArg5],   // R9
                f0, f1, f2, f3, f4, f5, f6, f7,
                s0, s1, s2, s3, s4, s5, s6, s7);
            // Write both rax and xmm0 back to guest state. The
            // guest knows which one is meaningful based on the
            // called function's signature; the "other" slot may
            // have been clobbered by the call but x86-64 calling
            // convention says rax/xmm0 are caller-saved, so the
            // guest doesn't rely on the pre-call values surviving.
            if (tl_progress_slot != nullptr) {
                tl_progress_slot->in_hle.store(false, std::memory_order_relaxed);
            }
            state->gpr[kSysvRet] = ret.rax;
            const u64 xmm0_bits = std::bit_cast<u64>(ret.xmm0);
            std::memcpy(&state->ymm[0], &xmm0_bits, sizeof(u64));

            // Diagnostics: flag an HLE stub that returned a gap-range value in
            // RAX (a likely bad-pointer source). No-op unless diagnostics are
            // compiled in.
            Diagnostics::CheckHleReturn(name, host_fn, ret.rax);

            // Post-return rax/xmm0 as TRACE detail. If a debug trace
            // shows the "call" line for a host_fn but never this "ret"
            // line, that function crashed inside the call.
            if (name.empty()) {
                LOG_TRACE(Core, "Bridge: ret  host={:#x} -> rax={:#x} xmm0={:#x}",
                          host_fn, ret.rax, xmm0_bits);
            } else {
                LOG_TRACE(Core, "Bridge: ret  {} -> rax={:#x} xmm0={:#x}",
                          name, ret.rax, xmm0_bits);
            }

            // Pop guest return address.
            state->rip = guest_return_addr;
            state->gpr[kGuestRsp] = guest_rsp + 8;

            // Diagnostics: a block entered by returning from an HLE bridge call
            // is reached inside this loop, NOT via a fresh DispatchOne entry, so
            // the boundary snapshot taken at function entry (cur_rip) missed it.
            // Record the boundary here too so the value-origin ring stays
            // complete across HLE calls. No-op unless diagnostics are compiled in.
            Diagnostics::RecordBlockBoundary(state, state->rip);
            continue;
        }

        // Guest path: cache lookup, compile on miss.
        BlockCache& bc = const_cast<BlockCache&>(rt->GetBlockCache());

        // Diagnostics: if the block we're about to enter is one under
        // investigation, dump guest GPRs at this safe point. No-op unless
        // diagnostics are compiled in.
        Diagnostics::MaybeDumpPreBlock(state);

        // Claim the code cache BEFORE the lookup. The claim must cover the
        // entire lookup -> compile -> insert -> execute window: a pointer
        // obtained from either the block cache or the lifter is only valid
        // for the cache generation it was created in, and the claim is what
        // holds that generation alive. (Acquiring after the lookup re-opens
        // the race where a flush recycles the cache between Lookup() and the
        // gateway's jmp.) The matching release happens at the next
        // dispatcher entry, or in Run() for fatal exits that bypass it.
        AcquireClaim();

        if (void* host_ptr = bc.Lookup(state->rip); host_ptr != nullptr) {
            ApplyGuestRounding(state);
            return host_ptr;
        }
        void* host_ptr = rt->CompileBlockForDispatcher(state->rip);
        if (host_ptr == nullptr) {
            // Compile failed terminally (hard per-block error, or recycle
            // attempts exhausted). state->rip already points at the block
            // we couldn't compile; stamp exit_reason so the gateway exit
            // reports the truth instead of whatever stale value the last
            // successful block left behind. UnsupportedInstruction is the
            // honest fit: the lifter could not produce code for this RIP.
            state->exit_reason =
                static_cast<u32>(ExitReason::UnsupportedInstruction);
            // We will not be executing cache code: hand the claim back.
            // (No rounding swap: we exit to host C++, where the default is
            // already live.)
            ReleaseClaim();
            return nullptr;
        }
        bc.Insert(state->rip, host_ptr);
        ApplyGuestRounding(state);
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
    //
    // These R0–R3 checkpoints are bring-up trace granularity
    // (LOG_TRACE: absent in release). Run() is the OUTER entry — it is
    // not itself in JIT-dispatched context — so logging here is safe
    // at any level; TRACE is chosen purely because this is checkpoint
    // detail, not something a release build needs.
    LOG_TRACE(Core, "Run: R0 enter, state.rip={:#x} state.gpr[4]={:#x}",
              state.rip, state.gpr[4]);
    Runtime* const saved_rt = tl_active_runtime;
    GuestState* const saved_state = tl_current_guest_state;
    LOG_TRACE(Core, "Run: R1 tls read ok, saved_rt={}", static_cast<void*>(saved_rt));
    tl_active_runtime = this;
    tl_current_guest_state = &state;
    LOG_TRACE(Core, "Run: R2 tls write ok");

    // Per-thread setup before entering JIT execution:
    //  (1) Ensure this thread has its dedicated VEH exception-handler stack, so
    //      a guest fault never has to run the handler on an exhausted guest
    //      stack. Idempotent per thread; no-op off-Windows (POSIX uses an
    //      alternate signal stack via SA_ONSTACK). See signals.cpp.
    //  (2) Point the guest's fs_base at this thread's TCB so the lifter's
    //      fs:/gs: (TLS) addressing resolves to real thread-local storage.
    //      RunThread installs the TCB via InitializeTLS/SetTcbBase before the
    //      first guest call, so GetTcbBase() is valid here. If TLS isn't set up
    //      on this thread yet (null) we leave fs_base as-is rather than point it
    //      at garbage. gs_base is left as-is (PS4/FreeBSD uses fs for TLS; gs is
    //      effectively unused).
    Core::EnsureVehStack();
    if (auto* tcb = Core::GetTcbBase()) {
        state.fs_base = reinterpret_cast<u64>(tcb);
    }

    //  (3) Register this thread in the progress registry (outermost Run only;
    //      nested Run via caller-stack callbacks reuses the slot). The slot
    //      is what lets OTHER threads' watchdogs see this thread advancing,
    //      and vice versa. RAII so the slot is released on every exit path.
    struct ProgressRegistration {
        ProgressRegistration() {
            if (tl_run_depth++ == 0) {
                tl_progress_slot = AcquireProgressSlot();
            }
        }
        ~ProgressRegistration() {
            if (--tl_run_depth == 0) {
                ReleaseProgressSlot(tl_progress_slot);
                tl_progress_slot = nullptr;
            }
        }
    } progress_registration;

    gateway_->Enter(state, &DispatcherTrampoline);
    LOG_TRACE(Core, "Run: R3 gateway returned");
    // A block that exits via the FATAL stub (jmp r15: unsupported insn,
    // string-op pointer bail) never re-enters the dispatcher, so the claim
    // taken when its pointer was handed out is still held here -- and the
    // guest's MXCSR may still be live for the same reason. Normal exits
    // released/restored at the last dispatcher entry, making both no-ops.
    ReleaseClaim();
    RestoreHostRounding(&state);
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
    // Called from the dispatcher's guest path WITH this thread's claim held
    // (see the flush-handshake block above). Compiling under the claim is
    // what makes the emitted bytes safe to hand back: a concurrent flush
    // cannot recycle the cache out from under the emission or the caller's
    // subsequent BlockCache::Insert + jmp.
    //
    // Bounded retry: a compile can fail because the bump allocator is full;
    // we then recycle via the stop-the-world handshake and try again. Two
    // recycle attempts are plenty -- if a freshly emptied 64 MB cache cannot
    // fit one block (<= BLOCK_HOST_SIZE_CAP), no number of flushes will.
    for (int attempt = 0; attempt < 3; ++attempt) {
        const u64 used_before = code_cache_->Used();
        void* host_ptr = lifter_->CompileBlock(guest_rip);
        if (host_ptr != nullptr) {
            // Diagnostics: optional full per-instruction disassembly of the
            // block (env-gated dev tool) and, for blocks under investigation,
            // a raw host-byte dump. No-ops unless diagnostics are compiled in.
            Diagnostics::OnBlockCompiled(guest_rip, host_ptr,
                                         code_cache_->Used() - used_before);
            return host_ptr;
        }

        // CompileBlock returned nullptr. Two distinct causes, two distinct
        // responses:
        //
        //   1. Cache exhaustion: Allocate() couldn't reserve a block-sized
        //      chunk. Detectable as free space < BlockReservationSize().
        //      Benign -- recycle and retry.
        //   2. Hard per-block failure: the assembler threw (oversized
        //      block, emitter bug) or some other non-capacity error, with
        //      room still available. Recycling wipes EVERY compiled block
        //      and cannot possibly help -- and because such failures are
        //      deterministic, the old undiscriminated retry would have
        //      flushed the entire cache twice per occurrence before giving
        //      up. Propagate immediately instead.
        //
        // (Used() is read while other threads may be allocating; the value
        // is monotonic within a generation, so the heuristic can only err
        // toward one unnecessary recycle in a tight race, never toward
        // missing a needed one: the retry loop re-evaluates.)
        const u64 free_bytes = code_cache_->Capacity() - code_cache_->Used();
        if (free_bytes >= Lifter::BlockReservationSize()) {
            return nullptr;
        }

        LOG_INFO(Core,
                 "Code cache full at guest_rip={:#x} ({} / {} bytes used); "
                 "stop-the-world recycle (attempt {})",
                 guest_rip, code_cache_->Used(), code_cache_->Capacity(),
                 attempt + 1);

        // Snapshot the generation BEFORE dropping our claim: if it has moved
        // by the time we hold the flush mutex, another thread already
        // recycled and we must not flush again (we'd wipe its fresh blocks).
        const u64 observed_gen =
            g_cache_generation.load(std::memory_order_acquire);

        // Drop our own claim for the duration of the handshake -- the flusher
        // must not wait on itself -- then re-claim before retrying so the
        // recompile is protected again.
        ReleaseClaim();
        FlushCachesForRecycle(observed_gen);
        AcquireClaim();
    }

    LOG_ERROR(Core,
              "Code cache recompile still failed at guest_rip={:#x} after "
              "recycling; block may exceed cache capacity",
              guest_rip);
    return nullptr;
}

void Runtime::FlushCachesForRecycle(u64 observed_generation) {
    // Caller holds NO claim (it released before calling) but may be one of
    // several threads that hit the full cache at once: the mutex serializes
    // them, and the generation re-check makes all but the first a no-op.
    std::unique_lock lk{g_flush_mutex};
    if (g_cache_generation.load(std::memory_order_acquire) !=
        observed_generation) {
        // Someone else recycled between our failed compile and here. The
        // cache is fresh; our caller just recompiles into it.
        return;
    }

    // Park new claimants, then wait for every in-flight claim -- threads
    // executing JIT code or mid-compile -- to drain. Threads blocked inside
    // HLE calls hold no claim and cannot stall this; they park at
    // AcquireClaim if they come back while we work.
    g_flush_pending.store(true, std::memory_order_seq_cst);
    g_flush_cv.wait(lk, [] {
        return g_active_claims.load(std::memory_order_seq_cst) == 0;
    });

    // Order: drop every guest_rip -> host_ptr mapping FIRST, so no lookup
    // can ever observe a pointer into recycled memory, then reset the bump
    // pointer. (With all claims drained nobody can look up concurrently,
    // but the ordering keeps the invariant locally checkable.)
    block_cache_->Clear();
    code_cache_->Flush();

    g_cache_generation.fetch_add(1, std::memory_order_seq_cst);
    g_flush_pending.store(false, std::memory_order_seq_cst);
    // Wake both populations: parked claimants and any queued flushers (who
    // will see the bumped generation and no-op).
    g_flush_cv.notify_all();
}

// ============================================================================
// Crash-diagnostic support (see runtime.h). Async-signal-safe: only
// thread-local pointer reads, integer field reads, and a pointer-range
// comparison. No allocation, no locks, no logging.
// ============================================================================
FaultContext DescribeFaultContext(const void* host_addr) noexcept {
    FaultContext ctx;

    // Was the host fault/code address inside the JIT code cache? Compute
    // this FIRST — the instruction-decode below depends on it. This
    // distinguishes "faulted while executing lifted guest code" from
    // "faulted in an HLE shim / while dereferencing a bad guest pointer
    // (the code pointer is in normal host code, the *data* address is
    // wild)". We reach the code cache via the thread-local active
    // runtime pointer rather than Instance() to avoid the static-local
    // init guard in signal context. Both the pointer read and
    // CodeCache::Contains (a pure range compare) are signal-safe.
    if (host_addr != nullptr) {
        Runtime* rt = tl_active_runtime;
        if (rt != nullptr) {
            ctx.in_jit_code = rt->GetCodeCache().Contains(host_addr);
        }
    }

    // tl_current_guest_state is the live GuestState for any Run() active
    // on THIS thread. A plain thread-local pointer read — signal-safe.
    // If it's null, this thread wasn't executing guest code (the fault
    // is in pure host code), and we leave ctx at its defaults.
    GuestState* gs = tl_current_guest_state;
    if (gs != nullptr) {
        ctx.in_runtime = true;
        ctx.guest_rip = gs->rip;
        ctx.guest_exit_reason = gs->exit_reason;
        // Snapshot the 16 GPRs. Fixed-size integer copy — signal-safe.
        // This lets the crash report show which register carried the
        // bad pointer that the faulting instruction dereferenced.
        for (int i = 0; i < 16; ++i) {
            ctx.guest_gpr[i] = gs->gpr[i];
        }
        ctx.have_gprs = true;

        // Decode the faulting HOST instruction (the bytes at host_addr,
        // i.e. the JIT'd code that actually faulted). We deliberately
        // decode host_addr rather than the guest bytes at gs->rip: the
        // lifter only syncs gs->rip at block boundaries and branches, so
        // mid-block gs->rip is the BLOCK-ENTRY guest RIP, not the
        // faulting instruction — decoding there is misleading (and may
        // not even land on an instruction boundary). host_addr is the
        // precise fault PC, it lives in the code cache (readable), and
        // the host instruction reveals the access shape directly: e.g.
        // `mov rXX, [rYY]` (a load that brought a bad pointer in) vs.
        // `mov [rYY], rXX` / an op on `[rYY]` (a store/use through a
        // pointer computed earlier). Combined with the guest GPR dump
        // and data_addr, that pins down whether the bad pointer was
        // loaded from guest memory or computed by lifted code.
        //
        // Only decode when host_addr is inside the code cache — anywhere
        // else it isn't our JIT code and the bytes are meaningless here.
#ifdef ARCH_X86_64
        if (host_addr != nullptr && ctx.in_jit_code) {
            // Capture raw bytes first — this always succeeds for an
            // in-cache address and is the fallback when decode doesn't.
            const auto* hb = reinterpret_cast<const u8*>(host_addr);
            for (int i = 0; i < 16; ++i) {
                ctx.faulting_raw_bytes[i] = hb[i];
            }
            ctx.raw_byte_count = 16;

            // Also capture the host bytes immediately PRECEDING the fault
            // PC. The faulting instruction is only the *consumer* of the
            // bad pointer (e.g. `mov rXX,[rax]`); the instruction(s) that
            // *computed* that pointer are just before it in the emitted
            // block. Capturing this window lets us disassemble the lifter
            // sequence that produced the address and see where a stray bit
            // (e.g. bit 33) is introduced. Reading backwards within the
            // code cache is safe — it's all our own committed code.
            // The window must be clamped to the cache BASE: when the fault
            // PC sits in the first 32 bytes of the cache (first compiled
            // block), an unclamped hb[i - 32] reads before the mapping and
            // faults inside the fault handler. Bytes are right-aligned in
            // pre_fault_bytes so offline disassembly still ends at the PC.
            {
                const u8* cache_base = tl_active_runtime->GetCodeCache().Base();
                const u64 avail_back = static_cast<u64>(hb - cache_base);
                const int back = avail_back >= 32 ? 32 : static_cast<int>(avail_back);
                for (int i = 0; i < back; ++i) {
                    ctx.pre_fault_bytes[(32 - back) + i] = hb[i - back];
                }
                ctx.pre_byte_count = static_cast<u8>(back);
            }

            ZydisDecoder dec;
            if (ZYAN_SUCCESS(ZydisDecoderInit(&dec, ZYDIS_MACHINE_MODE_LONG_64,
                                              ZYDIS_STACK_WIDTH_64))) {
                ZydisDecodedInstruction insn;
                ZydisDecodedOperand ops[ZYDIS_MAX_OPERAND_COUNT];
                if (ZYAN_SUCCESS(ZydisDecoderDecodeFull(
                        &dec, host_addr, 15, &insn, ops))) {
                    ctx.faulting_mnemonic = static_cast<u32>(insn.mnemonic);
                    ctx.faulting_insn_length = insn.length;
                    ctx.faulting_insn_decoded = true;
                    // Determine the memory-operand direction: is the
                    // faulting access a READ (load that brought the bad
                    // pointer/value in) or a WRITE (store through an
                    // already-bad pointer)? And which base register feeds
                    // the effective address? This pins whether the bad
                    // value arrived from guest memory or from a prior
                    // computed register.
                    for (u32 i = 0; i < insn.operand_count; ++i) {
                        if (ops[i].type == ZYDIS_OPERAND_TYPE_MEMORY) {
                            ctx.mem_base_reg = static_cast<u32>(ops[i].mem.base);
                            ctx.mem_is_write =
                                (ops[i].actions & ZYDIS_OPERAND_ACTION_MASK_WRITE) != 0;
                            ctx.mem_is_read =
                                (ops[i].actions & ZYDIS_OPERAND_ACTION_MASK_READ) != 0;
                            ctx.have_mem_operand = true;
                            break;
                        }
                    }
                }
            }
        }
#endif
    }

    return ctx;
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
    // FP control state (mxcsr / fpu_cw) comes up in the architectural reset
    // values via GuestState's default member initializers -- every
    // construction site gets them, not just this one. fpu_tag = 0 is
    // correct for the lifter's 1-bit "in use" model (all slots empty).

    // Record the true top of this guest stack for the HLE bridge's
    // stack-argument bound (see GuestState::stack_top). The raw, pre-
    // alignment top: nothing above it is part of the allocation.
    state.stack_top = reinterpret_cast<u64>(guest_stack_top);

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
    // kHostReturnAddress is the single shared constant (guest_state.h);
    // a duplicated local literal here previously risked silently diverging
    // from the dispatcher's recognition check.
    u64 rsp = caller.gpr[4];
    rsp &= ~static_cast<u64>(0xF);
    rsp -= 8;
    *reinterpret_cast<u64*>(rsp) = kHostReturnAddress;
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
//
// KNOWN LIMITATION — this is a heuristic, not a definitive test. The
// assumption is "not in any loaded host module ⇒ guest". That holds for
// everything in the tree today (host code is in loaded .so/.exe/.dll
// modules; guest memory is mapped by shadPS4's loader and belongs to no
// module), but it can misclassify any *executable host memory that is not
// part of a loaded module*, which would be reported as "guest". Future
// sources of such memory include:
//   - JIT-generated host trampolines / stubs not registered with the loader
//   - executable memory allocated by third-party libraries
//   - LLVM-generated stubs, if an LLVM-backed path is ever added
//   - additional runtime helper code allocated at run time
// None of these exist today, so the heuristic is currently safe.
//
// TODO: once a guest-address range is queryable from the memory manager,
// prefer a positive test — MemoryManager::ContainsGuestAddress(ptr) — over
// this negative "not a host module" inference. That test must be callable
// from the fault path without taking locks (this can run in a signal/fault
// context), so it depends on a lock-free range query being available.

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
