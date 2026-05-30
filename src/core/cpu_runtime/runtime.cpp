// SPDX-FileCopyrightText: Copyright 2026 shadPS4 Emulator Project
// SPDX-License-Identifier: GPL-2.0-or-later

#include "core/cpu_runtime/runtime.h"

#include <array>
#include <bit>
#include <cstdlib>
#include <cstring>
#include <type_traits>
#include <unordered_set>

#include "common/assert.h"
#include "common/arch.h"
#include "common/logging/log.h"
#include "core/cpu_runtime/block_cache.h"
#include "core/cpu_runtime/code_cache.h"
#include "core/cpu_runtime/gateway/gateway.h"
#include "core/cpu_runtime/lifter/lifter.h"
#include "core/cpu_runtime/hle_registry.h"

#ifdef ARCH_X86_64
#include <Zydis/Zydis.h>
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

    // Stuck-RIP watchdog. With diagnostics silenced on repeat entries
    // a real memset can run at hundreds of thousands of iterations
    // per second, so the threshold needs to accommodate genuinely
    // long loops. A 16 MB region cleared 16 bytes at a time is ~1 M
    // iterations — well within normal operation. Set the threshold
    // at 10 M to catch only true infinite loops (no forward progress
    // at all) rather than slow ones.
    //
    // The watchdog fires regardless of build flavor (it's a real fault
    // condition, not a trace), so it uses LOG_ERROR. It runs at most
    // once per stuck site (the == comparison, not >=), so the logging
    // cost is a non-issue.
    {
        static thread_local u64 last_rip = 0;
        static thread_local u64 same_rip_hits = 0;
        if (cur_rip == last_rip) {
            ++same_rip_hits;
        } else {
            last_rip = cur_rip;
            same_rip_hits = 1;
        }
        constexpr u64 kStuckThreshold = 10'000'000;
        if (same_rip_hits == kStuckThreshold) {
            LOG_ERROR(Core,
                      "Dispatcher: STUCK at {:#x} after {} re-entries; "
                      "dumping guest GPRs and exiting fatally",
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

    // [DIAG-RAXTRACE] Corruption tracer. The crash at guest_rip
    // 0x800274c09 dereferences a guest GPR (RAX) holding 0x414c40040 —
    // an address ABOVE all mapped guest memory (which tops out around
    // 0x8'xxxxxxxx for code/heap and 0x2'xxxxxxxx for dmem). That value
    // is produced by an EARLIER block and persists in GuestState. To pin
    // the producing instruction, watch for any GPR transitioning INTO the
    // impossible range and log the RIP of the block that just ran (the
    // culprit), plus the prior RIP for context. The dispatcher runs
    // between blocks (a safe point, plain C++), so reading GPRs here is
    // safe. Fires at most once per (gpr,rip) site to avoid loop spam.
    {
        // [RAXTRACE] One-time presence beacon: proves THIS runtime.cpp
        // (with the tracer) is the binary actually running. If you see a
        // crash but never see this line, runtime.o was not rebuilt.
        static thread_local bool announced = false;
        if (!announced) {
            announced = true;
            LOG_ERROR(Core, "[RAXTRACE] tracer-v3 active (dispatcher first entry, rip={:#x})",
                      cur_rip);
        }
        // A guest VA is "impossible" if it's at/above 0x10'00000000 (the
        // GPU carveout floor) yet not a normal code/heap (0x8__) or dmem
        // (0x2__) pointer. Simplest robust filter: flag the specific high
        // nibble pattern we keep seeing (bits >= 0x4'00000000 that aren't
        // 0x8__ code/heap). We log any GPR whose top byte is in [0x3,0x7]
        // — squarely in the unmapped gap between dmem and code.
        static thread_local u64 last_block_rip = 0;
        static thread_local u64 reported_sites = 0;  // bitset of gpr indices already reported
        const char* const kGprNames[16] = {
            "rax","rcx","rdx","rbx","rsp","rbp","rsi","rdi",
            "r8","r9","r10","r11","r12","r13","r14","r15",
        };
        for (int i = 0; i < 16; ++i) {
            if (i == 4 || i == 5) continue;  // skip rsp/rbp (host-ish stack values)
            const u64 v = state->gpr[i];
            const u64 top = v >> 32;
            // Unmapped gap: high dword in [0x3 .. 0x7] (above dmem 0x2__,
            // below code/heap 0x8__). 0x414c40040 -> top = 0x4.
            // EXCLUDE the legitimately-mapped high "anon" stack region
            // (~0x7ef000000..0x7f0000000) which also has top byte 0x7 —
            // those are valid pointers, not corruption (false positives).
            const bool in_anon_stack = (v >= 0x7ef000000ull && v < 0x800000000ull);
            if (top >= 0x3 && top <= 0x7 && !in_anon_stack) {
                if (!(reported_sites & (1ull << i))) {
                    reported_sites |= (1ull << i);
                    LOG_ERROR(Core,
                              "[RAXTRACE] guest {} = {:#x} (UNMAPPED gap) first seen after "
                              "block rip={:#x}; next rip={:#x}",
                              kGprNames[i], v, last_block_rip, cur_rip);
                    // Auto-dump the culprit block's emitted host code. Look
                    // up its host pointer in the block cache and dump a
                    // fixed window (the block plus a little of the next —
                    // fine for offline disassembly). This removes the need
                    // for a per-RIP allowlist and any compile-time timing
                    // dependence: whatever block produced the bad value
                    // gets its code dumped right here.
                    if (rt != nullptr && last_block_rip != 0) {
                        BlockCache& bc = const_cast<BlockCache&>(rt->GetBlockCache());
                        if (void* hp = bc.Lookup(last_block_rip)) {
                            const u8* p = reinterpret_cast<const u8*>(hp);
                            const char* digits = "0123456789abcdef";
                            char hex[256 * 3 + 1];
                            for (int k = 0; k < 256; ++k) {
                                hex[k * 3 + 0] = digits[(p[k] >> 4) & 0xF];
                                hex[k * 3 + 1] = digits[p[k] & 0xF];
                                hex[k * 3 + 2] = ' ';
                            }
                            hex[256 * 3] = '\0';
                            LOG_ERROR(Core,
                                      "[RAXTRACE] culprit block {:#x} host code (256 bytes): {}",
                                      last_block_rip, hex);
                        }
                    }
                }
            }
        }
        last_block_rip = cur_rip;

        // [GPRSNAP] Per-block full-GPR snapshot ring + value-origin walk.
        //
        // The gap detector above names the block AFTER which a register was
        // first seen holding a bad value, but it cannot say where that value
        // was actually WRITTEN — the bad register often persists unchanged
        // across many blocks before the gap detector first inspects it, and
        // the block-RIP ring records only RIPs, never values. This snapshot
        // ring records ALL 16 GPRs at every block boundary. When a register
        // first enters the unmapped gap, we walk the ring backward to find
        // the FIRST boundary where that register's value differs from the
        // boundary before it — i.e. the block that actually produced the
        // bad value — and report old->new with the producing block's RIP.
        //
        // This is the decisive attribution the prior tracers could not give:
        // it converts "corruption happened somewhere in the last N blocks"
        // into "block X wrote <bad> into <reg> (was <good>)".
        {
            constexpr int kSnapDepth = 64;
            struct GprSnap {
                u64 rip;            // block RIP recorded at this boundary
                u64 gpr[16];        // full GPR file at this boundary
                bool valid;
            };
            static thread_local GprSnap snaps[kSnapDepth] = {};
            static thread_local u32 snap_pos = 0;
            static thread_local u64 origin_reported = 0;  // bitset per gpr idx

            // Record this boundary's snapshot.
            const u32 cur_slot = snap_pos % kSnapDepth;
            snaps[cur_slot].rip = cur_rip;
            for (int i = 0; i < 16; ++i) snaps[cur_slot].gpr[i] = state->gpr[i];
            snaps[cur_slot].valid = true;
            snap_pos++;

            // For each GPR now sitting in the unmapped gap that we haven't
            // already attributed, walk the ring backward to find where it
            // first took on its current (bad) value.
            for (int i = 0; i < 16; ++i) {
                if (i == 4 || i == 5) continue;  // rsp/rbp: stack-ish, skip
                const u64 bad = state->gpr[i];
                const u64 top = bad >> 32;
                const bool in_anon_stack =
                    (bad >= 0x7ef000000ull && bad < 0x800000000ull);
                const bool is_bad = (top >= 0x3 && top <= 0x7 && !in_anon_stack);
                if (!is_bad) continue;
                if (origin_reported & (1ull << i)) continue;

                // Walk backward over recorded snapshots (newest first).
                // We have snap_pos total writes; only the last min(snap_pos,
                // kSnapDepth) are live. Find the oldest CONTIGUOUS run in
                // which gpr[i] == bad, then the snapshot just before that run
                // holds the prior (good) value and its RIP is the producer.
                const u32 live = (snap_pos < (u32)kSnapDepth) ? snap_pos
                                                              : (u32)kSnapDepth;
                if (live < 2) continue;

                // newest live snapshot is at (snap_pos-1).
                u64 prev_val = 0;
                u64 producer_rip = 0;
                bool found = false;
                bool seen_bad_run = false;
                for (u32 k = 1; k <= live; ++k) {
                    const u32 slot = (snap_pos - k) % kSnapDepth;
                    if (!snaps[slot].valid) break;
                    const u64 v = snaps[slot].gpr[i];
                    if (v == bad) {
                        seen_bad_run = true;
                        // keep walking back; this boundary still had the bad value
                        continue;
                    }
                    // First boundary (going back) where the value was NOT bad.
                    if (seen_bad_run) {
                        // Snapshot timing: snaps[].rip labels the block ABOUT
                        // TO RUN at that boundary, while snaps[].gpr is the
                        // state the PREVIOUS block produced. So the block that
                        // actually produced the good->bad transition is the one
                        // about-to-run at THIS (last-good) boundary — i.e.
                        // snaps[slot].rip, not the next-newer slot. (The newer
                        // slot's value is already bad, so its about-to-run block
                        // merely inherited the bad value.)
                        prev_val = v;
                        producer_rip = snaps[slot].rip;
                        found = true;
                    }
                    break;
                }

                if (found) {
                    origin_reported |= (1ull << i);
                    LOG_ERROR(Core,
                              "[GPRSNAP] {} VALUE-ORIGIN: block {:#x} changed it "
                              "from {:#x} -> {:#x} (bad now persists into rip={:#x})",
                              kGprNames[i], producer_rip, prev_val, bad, cur_rip);
                    // Dump the producing block's emitted host code for offline
                    // disassembly, same format as the gap detector above.
                    if (rt != nullptr && producer_rip != 0) {
                        BlockCache& bc = const_cast<BlockCache&>(rt->GetBlockCache());
                        if (void* hp = bc.Lookup(producer_rip)) {
                            const u8* p = reinterpret_cast<const u8*>(hp);
                            const char* digits = "0123456789abcdef";
                            char hex[256 * 3 + 1];
                            for (int k = 0; k < 256; ++k) {
                                hex[k * 3 + 0] = digits[(p[k] >> 4) & 0xF];
                                hex[k * 3 + 1] = digits[p[k] & 0xF];
                                hex[k * 3 + 2] = ' ';
                            }
                            hex[256 * 3] = '\0';
                            LOG_ERROR(Core,
                                      "[GPRSNAP] producer block {:#x} host code "
                                      "(256 bytes): {}",
                                      producer_rip, hex);
                        }
                    }
                } else if (seen_bad_run) {
                    // The bad value was already present at the oldest live
                    // snapshot — the producer is older than our ring depth.
                    // Report that so we know to deepen the ring if needed.
                    origin_reported |= (1ull << i);
                    LOG_ERROR(Core,
                              "[GPRSNAP] {} VALUE-ORIGIN: bad value {:#x} predates "
                              "the {}-deep snapshot ring (producer older than "
                              "rip={:#x}); increase kSnapDepth to catch it",
                              kGprNames[i], bad, kSnapDepth,
                              snaps[(snap_pos - live) % kSnapDepth].rip);
                }
            }
        }

        // [RAXTRACE] Block-RIP ring buffer. Records the last 32 block
        // entries on this thread. When a corruption is first reported
        // (above), we also dump this ring so the actual loop CYCLE is
        // visible — which blocks repeat, in what order. This reveals the
        // OUTER loop (the one resetting RBX and restarting the copy) that
        // the per-block attribution can't name on its own. Dumped once,
        // right after the first corruption report.
        {
            static thread_local u64 ring[32] = {};
            static thread_local u32 ring_pos = 0;
            static thread_local bool ring_dumped = false;
            ring[ring_pos & 31] = cur_rip;
            ring_pos++;
            if (reported_sites != 0 && !ring_dumped) {
                ring_dumped = true;
                // Emit the last 32 block RIPs oldest→newest.
                for (u32 k = 0; k < 32; ++k) {
                    u32 idx = (ring_pos + k) & 31;
                    LOG_ERROR(Core, "[RAXTRACE] ring[{}] block rip={:#x}", k, ring[idx]);
                }
            }
        }
    }

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
            // marshalling reads rather than in the called function. Guard the
            // reads with a page bound: a qword is only dereferenced if its
            // full 8 bytes lie within the same page as guest_rsp; otherwise it
            // defaults to 0. The return address and SysV stack args are pushed
            // immediately below RSP, so in practice they're always in-page and
            // this never trips — it only protects the pathological boundary
            // case. We use a page bound rather than a VMA query because the
            // bridge runs from a JIT gateway frame where taking the memory
            // manager's mutex is unsafe (same reason this path avoids spdlog).
            constexpr u64 kPageSize = 0x1000;
            const u64 rsp_page_end = (guest_rsp & ~(kPageSize - 1)) + kPageSize;
            const auto safe_read_qword = [&](u64 addr) -> u64 {
                if (addr < guest_rsp || addr + sizeof(u64) > rsp_page_end) {
                    // Below RSP or crossing the page boundary: don't touch it.
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
            // signature. Each read is page-bounded (see safe_read_qword
            // above): args past the end of guest_rsp's page default to 0
            // rather than faulting the bridge.
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
            state->gpr[kSysvRet] = ret.rax;
            const u64 xmm0_bits = std::bit_cast<u64>(ret.xmm0);
            std::memcpy(&state->ymm[0], &xmm0_bits, sizeof(u64));

            // [RAXTRACE] If an HLE stub just returned a value in the
            // unmapped gap (high dword 0x3..0x7) into guest RAX, that is
            // very likely the source of the bad pointer dereferenced later
            // at 0x800274c09. Name the function so we know which stub to
            // fix. Fires once.
            {
                static thread_local bool hle_bad_reported = false;
                const u64 top = ret.rax >> 32;
                if (!hle_bad_reported && top >= 0x3 && top <= 0x7) {
                    hle_bad_reported = true;
                    LOG_ERROR(Core,
                              "[RAXTRACE] HLE stub '{}' (host={:#x}) returned rax={:#x} "
                              "in UNMAPPED gap — likely the bad-pointer source",
                              name.empty() ? std::string_view{"<unregistered>"} : name,
                              host_fn, ret.rax);
                }
            }

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
            continue;
        }

        // Guest path: cache lookup, compile on miss.
        BlockCache& bc = const_cast<BlockCache&>(rt->GetBlockCache());

        // [RAXTRACE] Targeted pre-block dump: if we are about to enter the
        // known faulting block, dump all guest GPRs HERE — this is the
        // exact state feeding the crash, captured at a safe point before
        // the block runs. Shows which register already holds the bad
        // pointer and what the others look like one block earlier than the
        // fault handler sees. Fires once.
        {
            static thread_local bool dumped = false;
            if (!dumped && state->rip == 0x800274c09ull) {
                dumped = true;
                static const char* const kN[16] = {
                    "rax","rcx","rdx","rbx","rsp","rbp","rsi","rdi",
                    "r8","r9","r10","r11","r12","r13","r14","r15"};
                LOG_ERROR(Core, "[RAXTRACE] about to enter faulting block 0x800274c09; guest GPRs:");
                for (int i = 0; i < 16; ++i)
                    LOG_ERROR(Core, "[RAXTRACE]   {} = {:#x}", kN[i], state->gpr[i]);
            }
        }

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
    gateway_->Enter(state, &DispatcherTrampoline);
    LOG_TRACE(Core, "Run: R3 gateway returned");
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
    const u64 used_before = code_cache_->Used();
    void* host_ptr = lifter_->CompileBlock(guest_rip);
    if (host_ptr != nullptr) {
#ifdef ARCH_X86_64
        // Optional: disassemble EVERY compiled block straight into the log,
        // once per guest RIP, so blocks are human-readable without any offline
        // tool. Off by default (this floods the log); enable by setting the
        // environment variable SHADPS4_DUMP_BLOCKS=1 before launching. Output
        // is one header line per block plus one line per host instruction:
        //   block 0x800274c09 (32 host bytes):
        //     +0x00  mov eax, [r13+0x50]        ; guest R10
        //     +0x04  mov [r13+0x10], rax        ; guest RDX
        //     ...
        // The r13-relative annotation maps the access back to the guest
        // register / state field, since r13 is pinned to the GuestState base.
        // x86-only: it decodes the emitted HOST instructions with Zydis, whose
        // formatter is x86. On the arm64 host the emitted code is AArch64, which
        // Zydis can't format, so the feature is compiled out there (the Zydis
        // include itself is also gated to ARCH_X86_64).
        static const bool dump_all_blocks = [] {
            const char* e = std::getenv("SHADPS4_DUMP_BLOCKS");
            return e != nullptr && e[0] != '\0' && e[0] != '0';
        }();
        if (dump_all_blocks) {
            // De-dup per guest RIP so re-dispatched blocks don't re-print.
            static thread_local std::unordered_set<u64> seen_blocks;
            if (seen_blocks.insert(guest_rip).second) {
                const u64 used_after = code_cache_->Used();
                u64 n = used_after - used_before;
                constexpr u64 kMaxDump = 1024;
                if (n > kMaxDump) n = kMaxDump;
                const auto* code = reinterpret_cast<const u8*>(host_ptr);

                LOG_ERROR(Core, "block {:#x} ({} host bytes):", guest_rip, n);

                // Map an r13-relative displacement to a guest register / state
                // field name (GuestState layout: gpr[0..15] at 0x00..0x78,
                // then rip/rflags/lazy-flag fields/fs_base/gs_base).
                auto slot_name = [](u64 off) -> const char* {
                    static const char* kGpr[16] = {
                        "RAX","RCX","RDX","RBX","RSP","RBP","RSI","RDI",
                        "R8","R9","R10","R11","R12","R13","R14","R15"};
                    if (off < 0x80 && (off % 8) == 0) return kGpr[off / 8];
                    switch (off) {
                    case 0x80: return "rip";
                    case 0x88: return "rflags";
                    case 0x90: return "flag_op";
                    case 0x98: return "flag_lhs";
                    case 0xa0: return "flag_rhs";
                    case 0xa8: return "flag_result";
                    case 0xb0: return "fs_base";
                    case 0xb8: return "gs_base";
                    default:   return nullptr;
                    }
                };

                ZydisDecoder dec;
                ZydisFormatter fmt;
                if (ZYAN_SUCCESS(ZydisDecoderInit(&dec, ZYDIS_MACHINE_MODE_LONG_64,
                                                  ZYDIS_STACK_WIDTH_64)) &&
                    ZYAN_SUCCESS(ZydisFormatterInit(&fmt, ZYDIS_FORMATTER_STYLE_INTEL))) {
                    u64 off = 0;
                    while (off < n) {
                        // Stop at zero-padding: the code-cache "used" delta
                        // can over-report a block's true length (alignment /
                        // post-flush bump), and the tail is zeroed cache that
                        // decodes as a meaningless run of `add [rax], al`
                        // (opcode 00 00). A 00 00 here is never real emitted
                        // code, so treat it as the end of the block.
                        if (off + 1 < n && code[off] == 0x00 && code[off + 1] == 0x00) {
                            break;
                        }
                        ZydisDecodedInstruction insn;
                        ZydisDecodedOperand ops[ZYDIS_MAX_OPERAND_COUNT];
                        if (!ZYAN_SUCCESS(ZydisDecoderDecodeFull(
                                &dec, code + off, n - off, &insn, ops))) {
                            LOG_ERROR(Core, "  +{:#04x}  <bad decode>", off);
                            break;
                        }
                        char text[160];
                        ZydisFormatterFormatInstruction(&fmt, &insn, ops,
                                                        insn.operand_count_visible,
                                                        text, sizeof(text),
                                                        /*runtime_address=*/off,
                                                        ZYAN_NULL);
                        // Annotate the first r13-relative memory operand with
                        // the guest slot it touches. We read disp.value
                        // directly: a zero displacement is a valid slot
                        // access ([r13] == guest RAX), and the
                        // `has_displacement` flag isn't present across all
                        // Zydis versions in the tree.
                        const char* note = nullptr;
                        for (u32 i = 0; i < insn.operand_count; ++i) {
                            if (ops[i].type == ZYDIS_OPERAND_TYPE_MEMORY &&
                                ops[i].mem.base == ZYDIS_REGISTER_R13) {
                                const u64 d =
                                    static_cast<u64>(ops[i].mem.disp.value);
                                note = slot_name(d);
                                break;
                            }
                        }
                        if (note != nullptr) {
                            LOG_ERROR(Core, "  +{:#04x}  {}    ; guest {}",
                                      off, text, note);
                        } else {
                            LOG_ERROR(Core, "  +{:#04x}  {}", off, text);
                        }
                        off += insn.length;

                        // A block ends with its terminator: an indirect jump
                        // through r14 (dispatcher loop) or r15 (clean/fatal
                        // exit). Once we've emitted that, everything after is
                        // the next block or padding — stop here so we don't
                        // walk past the real end.
                        if (insn.mnemonic == ZYDIS_MNEMONIC_JMP &&
                            ops[0].type == ZYDIS_OPERAND_TYPE_REGISTER &&
                            (ops[0].reg.value == ZYDIS_REGISTER_R14 ||
                             ops[0].reg.value == ZYDIS_REGISTER_R15)) {
                            break;
                        }
                    }
                }
            }
        }
#endif // ARCH_X86_64
        // [RAXTRACE] Dump emitted host bytes for the loop blocks involved
        // in the bit-33 corruption (0x800274c00 / ...c09 / ...cae), so the
        // exact lifted sequence can be disassembled. Computed from the
        // code-cache bump delta; host_ptr is the block's start. Fires once
        // per guest_rip of interest.
        if (guest_rip == 0x800274c00ull || guest_rip == 0x800274c09ull ||
            guest_rip == 0x800274caeull || guest_rip == 0x800302f60ull) {
            static thread_local u64 dumped_mask = 0;
            int slot = (guest_rip == 0x800274c00ull) ? 0
                     : (guest_rip == 0x800274c09ull) ? 1
                     : (guest_rip == 0x800274caeull) ? 2 : 3;
            if (!(dumped_mask & (1ull << slot))) {
                dumped_mask |= (1ull << slot);
                const u64 used_after = code_cache_->Used();
                u64 n = used_after - used_before;
                if (n > 256) n = 256;
                const u8* p = reinterpret_cast<const u8*>(host_ptr);
                const char* digits = "0123456789abcdef";
                char hex[256 * 3 + 1];
                u64 m = (n > 256) ? 256 : n;
                for (u64 i = 0; i < m; ++i) {
                    hex[i * 3 + 0] = digits[(p[i] >> 4) & 0xF];
                    hex[i * 3 + 1] = digits[p[i] & 0xF];
                    hex[i * 3 + 2] = ' ';
                }
                hex[m * 3] = '\0';
                LOG_ERROR(Core, "[RAXTRACE] emitted block {:#x} ({} host bytes): {}",
                          guest_rip, n, hex);
            }
        }
        return host_ptr;
    }
    // CompileBlock returned nullptr. The common (and benign) cause is
    // a full code cache — the bump allocator ran out of room. This is
    // not an error: we recycle the cache and recompile. The code
    // cache and block cache MUST be flushed together — Flush() resets
    // the code-cache bump pointer, invalidating every host pointer the
    // block cache holds, so any surviving block-cache entry would
    // dangle into recycled memory.
    //
    // Safety: this runs in the dispatcher (plain C++) between guest
    // blocks — the gateway has returned here to resolve the next
    // block, so no JIT code is executing in the cache. Per
    // CodeCache::Flush's contract this is the safe point to recycle.
    //
    // We do NOT flush if the cache simply couldn't fit a single block
    // when already empty (a block larger than the whole cache) — that
    // is a genuine failure, not a capacity-recycling case. We detect
    // it by checking whether the cache had any prior usage: if Used()
    // is already ~0, flushing won't help, so we bail.
    if (code_cache_->Used() == 0) {
        // Empty cache yet compile still failed → not a capacity issue.
        // Propagate the failure (lifter set the exit_reason).
        return nullptr;
    }

    LOG_INFO(Core, "Code cache full at guest_rip={:#x} ({} / {} bytes used); "
                   "flushing and recompiling",
             guest_rip, code_cache_->Used(), code_cache_->Capacity());

    // Flush the block cache first (drop all guest_rip -> host_ptr
    // mappings), then reset the code-cache bump pointer. Order matters
    // only for clarity here — both complete before we recompile, and
    // no other thread is dispatching (single compiler thread).
    block_cache_->Clear();
    code_cache_->Flush();

    // Retry the compile into the now-empty cache. With a sane cache
    // size relative to BLOCK_HOST_SIZE_CAP this always succeeds; if it
    // somehow doesn't, propagate the failure rather than looping.
    host_ptr = lifter_->CompileBlock(guest_rip);
    if (host_ptr == nullptr) {
        LOG_ERROR(Core, "Code cache recompile still failed at guest_rip={:#x} "
                        "after flush; block may exceed cache capacity",
                  guest_rip);
    }
    return host_ptr;
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
            for (int i = 0; i < 32; ++i) {
                ctx.pre_fault_bytes[i] = hb[i - 32];
            }
            ctx.pre_byte_count = 32;

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
