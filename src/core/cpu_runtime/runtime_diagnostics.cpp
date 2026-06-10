// SPDX-FileCopyrightText: Copyright 2024 shadPS4 Emulator Project
// SPDX-License-Identifier: GPL-2.0-or-later

#include "core/cpu_runtime/runtime_diagnostics.h"

// The whole translation unit is empty unless diagnostics are enabled, so a
// release build links in nothing from here.
#ifdef SHADPS4_RUNTIME_DIAGNOSTICS

#include <cstring>
#include <string_view>
#include <unordered_set>
#include <cstddef>
#include <cstdlib>

#include "common/arch.h"
#include "common/logging/log.h"
#include "core/cpu_runtime/block_cache.h"
#include "core/cpu_runtime/code_cache.h"
#include "core/cpu_runtime/runtime.h"

#ifdef ARCH_X86_64
#include <Zydis/Zydis.h>
#endif

#ifdef _WIN32
#include <windows.h>
// SEH-guarded 64-bit read. Returns true and sets *out if the address is mapped
// and readable; returns false if the access would fault. Used by the field
// watchpoint, whose target may be unmapped early in boot before the owning
// object is allocated. clang-cl supports __try/__except.
static bool SafeReadU64(unsigned long long addr, unsigned long long* out) {
    __try {
        *out = *reinterpret_cast<volatile unsigned long long*>(addr);
        return true;
    } __except (EXCEPTION_EXECUTE_HANDLER) {
        return false;
    }
}
#else
static bool SafeReadU64(unsigned long long addr, unsigned long long* out) {
    // Non-Windows fallback: best-effort direct read (the target platform for
    // this diagnostic is the Windows clang-cl build; other platforms simply do
    // the plain read).
    *out = *reinterpret_cast<volatile unsigned long long*>(addr);
    return true;
}
#endif

namespace Core::Runtime {
namespace Diagnostics {

namespace {

const char* const kGprNames[16] = {
    "rax", "rcx", "rdx", "rbx", "rsp", "rbp", "rsi", "rdi",
    "r8",  "r9",  "r10", "r11", "r12", "r13", "r14", "r15",
};

// A guest VA is treated as "impossible" (corruption) if its high dword sits in
// [0x3 .. 0x7] — above dmem (0x2__) and below code/heap (0x8__) — EXCEPT the
// legitimately-mapped anonymous stack region (~0x7ef000000..0x800000000),
// which also has a top byte of 0x7 and must not be flagged.
bool IsCorruptGuestValue(u64 v) {
    const u64 top = v >> 32;
    const bool in_anon_stack = (v >= 0x7ef000000ull && v < 0x800000000ull);
    return top >= 0x3 && top <= 0x7 && !in_anon_stack;
}

// Hex-dump a fixed window of a compiled block's emitted host code for offline
// disassembly. Runs inside corruption diagnostics, so it must not itself fault:
// before touching any byte it verifies, via CodeCache::Contains() (a signal-
// safe pure range compare), that BOTH the first and last byte of the window
// lie inside the code cache. If the block sits near the end of a cache region,
// the cache was flushed, or a bad pointer entered the block cache, the read
// would otherwise walk off mapped memory and crash while diagnosing another
// crash. On any failure we log the reason and read nothing.
void SafeDumpBlockCode(Runtime* rt, u64 block_rip, const void* hp,
                       const char* tag) {
    constexpr int kWindow = 768;
    if (rt == nullptr || hp == nullptr) {
        return;
    }
    const u8* p = reinterpret_cast<const u8*>(hp);
    const CodeCache& cc = rt->GetCodeCache();
    if (!cc.Contains(p) || !cc.Contains(p + (kWindow - 1))) {
        LOG_ERROR(Core,
                  "[{}] block {:#x}: host code window [{}, +{}) is not fully "
                  "inside the code cache; skipping dump to avoid faulting in "
                  "diagnostics",
                  tag, block_rip, hp, kWindow);
        return;
    }
    const char* digits = "0123456789abcdef";
    char hex[kWindow * 3 + 1];
    for (int k = 0; k < kWindow; ++k) {
        hex[k * 3 + 0] = digits[(p[k] >> 4) & 0xF];
        hex[k * 3 + 1] = digits[p[k] & 0xF];
        hex[k * 3 + 2] = ' ';
    }
    hex[kWindow * 3] = '\0';
    LOG_ERROR(Core, "[{}] block {:#x} host code ({} bytes): {}",
              tag, block_rip, kWindow, hex);
}

// ---- Shared per-thread diagnostic state -----------------------------------
//
// These were function-local thread_locals inside the dispatcher; moving the
// logic here brings the state with it. RecordBlockBoundary writes the snapshot
// ring and block-RIP ring; CheckRegisterCorruption reads them and consults
// g_reported_sites (also set here) to trigger the one-shot ring dump.

constexpr int kSnapDepth = 64;
struct GprSnap {
    u64 rip;       // block RIP recorded at this boundary
    u64 gpr[16];   // full GPR file at this boundary
    bool valid;
};
thread_local GprSnap g_snaps[kSnapDepth] = {};
thread_local u32 g_snap_pos = 0;
thread_local u64 g_origin_reported = 0;  // bitset per gpr idx (value-origin walk)

thread_local u64 g_last_block_rip = 0;
thread_local u64 g_reported_sites = 0;   // bitset per gpr idx (gap detector)

thread_local u64 g_rip_ring[32] = {};
thread_local u32 g_rip_ring_pos = 0;
thread_local bool g_rip_ring_dumped = false;

// Field-watchpoint -> corruption-check handoff. When the watchpoint observes the
// tracked field being zeroed, it records the block that ran just before (the
// writer) here; CheckRegisterCorruption (which has the Runtime* needed to look
// up and hex-dump compiled host bytes) consumes it and dumps that block once.
thread_local u64 g_watch_zeroing_block = 0;
thread_local bool g_watch_dump_done = false;

// Rolling window of the most recent HLE bridge calls on this thread. We don't
// log them live (early-boot HLE traffic would flood); instead we keep the last
// kHleRingDepth and dump them at the moment the fault block is about to run
// (from MaybeDumpPreBlock), so the log shows exactly which stubs ran most
// recently before the bad struct read. thread_local so the crash thread's
// window isn't polluted by other threads' HLE traffic.
struct HleCallRec {
    char     name[64];
    u64      host_fn;
    u64      ret_to;
    u64      args[6];   // rdi, rsi, rdx, rcx, r8, r9
    bool     valid;
};
constexpr int kHleRingDepth = 32;
thread_local HleCallRec g_hle_ring[kHleRingDepth] = {};
thread_local u32 g_hle_ring_pos = 0;

// Ring of recent caller(0x80030e680) invocations. We record every call's key
// inputs and dump the window at the real fault, so the true failing invocation
// is always captured regardless of dataflow-reconstruction error.
struct FaultCallRec {
    u64  r14;
    u64  rax;
    u64  rdx;
    u64  base;
    u64  predicted_r15;
    u32  V;
    bool base_ok;
    bool valid;
};
constexpr int kFaultRingDepth = 16;
thread_local FaultCallRec g_fault_ring[kFaultRingDepth] = {};
thread_local u32 g_fault_ring_pos = 0;

}  // namespace

// Forward declaration: RecordBlockBoundary (below) calls DumpFaultStruct when it
// sees the caller block. DumpFaultStruct is defined later in this Diagnostics
// namespace (not the anonymous one — that's what previously caused an undefined-
// symbol link error), after IsImplausiblePointer which it uses as a read guard.
void DumpFaultStruct(GuestState* state);
void DumpFaultRing();

void AnnounceOnce(u64 first_rip) {
    static thread_local bool announced = false;
    if (!announced) {
        announced = true;
        LOG_ERROR(Core,
                  "[RAXTRACE] tracer-v3 active (dispatcher first entry, rip={:#x})",
                  first_rip);
    }
}

void RecordBlockBoundary(GuestState* state, u64 cur_rip) {
    // Differential execution trace. When SHADPS4_TRACE=1, emit one compact line
    // per block: the module-relative RIP plus all 16 GPRs, so the control-flow
    // path and register evolution can be diffed against a reference run (e.g.
    // mainline shadPS4). RIPs are normalized to module-relative (subtract the
    // 0x800000000 load base) so they match across runs/ASLR. Bounded so a long
    // session can't fill the disk; the first divergence from the reference is
    // the bug. Gated + cached so the getenv cost is paid once per thread.
    {
        static thread_local int trace_on = -1;       // -1 = unknown, 0/1 cached
        if (trace_on == -1) {
            const char* e = std::getenv("SHADPS4_TRACE");
            trace_on = (e && e[0] == '1') ? 1 : 0;
        }
        if (trace_on == 1) {
            static thread_local u64 trace_seq = 0;
            static thread_local u64 trace_budget = 2000000;  // ~2M blocks cap
            if (trace_budget > 0) {
                --trace_budget;
                constexpr u64 kModuleBase = 0x800000000ull;
                const u64 rel = (cur_rip >= kModuleBase) ? (cur_rip - kModuleBase)
                                                         : cur_rip;
                const u64* g = state->gpr.data();
                LOG_ERROR(Core,
                    "TRACE {} {:#x} {:x} {:x} {:x} {:x} {:x} {:x} {:x} {:x} "
                    "{:x} {:x} {:x} {:x} {:x} {:x} {:x} {:x}",
                    trace_seq++, rel,
                    g[0], g[1], g[2], g[3], g[4], g[5], g[6], g[7],
                    g[8], g[9], g[10], g[11], g[12], g[13], g[14], g[15]);
                if (trace_budget == 0) {
                    LOG_ERROR(Core, "TRACE budget exhausted; tracing stopped");
                }
            }
        }
    }

    // Snapshot ring: record all 16 GPRs at this boundary.
    const u32 cur_slot = g_snap_pos % kSnapDepth;
    g_snaps[cur_slot].rip = cur_rip;
    for (int i = 0; i < 16; ++i) {
        g_snaps[cur_slot].gpr[i] = state->gpr[i];
    }
    g_snaps[cur_slot].valid = true;
    g_snap_pos++;

    // Block-RIP ring: record this entry.
    g_rip_ring[g_rip_ring_pos & 31] = cur_rip;
    g_rip_ring_pos++;

    // Field write-watchpoint. The recurring null-deref faults on a load of
    // [obj+0x18] that comes back 0; for this title the object is at guest
    // Guest memory is identity-mapped to host VAs (the lifter dereferences
    // guest addresses directly), so watched fields read straight from their
    // guest addresses. Polled at every block boundary: first observed value
    // plus every change, attributed to the block that ran just before the
    // observing boundary. v6 watches the two RSDK records implicated by the
    // [VERDICT] (see the slot table below); the original 0x8016a2270 lead is
    // closed and retired. SHADPS4_WATCH=0 disables.
    {
        static thread_local int watch_on = -1;
        if (watch_on == -1) {
            // Default ON in diagnostic builds (this whole file only compiles
            // when SHADPS4_RUNTIME_DIAGNOSTICS is set, so being here already
            // means a deliberate debug build). SHADPS4_WATCH=0 disables it;
            // anything else (including unset) leaves it on. This avoids losing
            // a run to a forgotten environment export.
            const char* e = std::getenv("SHADPS4_WATCH");
            watch_on = (e && e[0] == '0') ? 0 : 1;
        }
        if (watch_on == 1) {
            // v6: multi-slot change logger. The v5 [VERDICT] proved the
            // faulting argument is a faithful read of an UNINITIALIZED
            // record in main-module static data: {id=0, q@+8=0, q@+0x10=0x3f}
            // at blob(0x800655d10)+0x278, while its sibling record at
            // blob+0x0 held a live heap pointer one loop iteration earlier.
            // So the question is no longer "who corrupted a byte" but "who
            // initializes these records, and why did entry1's init not run
            // (or get diverted)". We watch BOTH records:
            //   slots 0-2: the BROKEN record  (blob+0x278: id / qA / qB)
            //   slots 3-5: the HEALTHY record (blob+0x000: id / qA / qB)
            // When the healthy record's writer fires, its block RIP names
            // the init function -- then the broken record either (a) shows
            // a write too (writer found directly), or (b) never changes
            // from its image/boot value, proving the init path skipped it,
            // and the init function's own branches become the next focus.
            // The original 0x8016a2270 watch is retired (that lead closed).
            struct WatchSlot {
                u64 addr;
                const char* tag;
                bool inited;
                u64 last;
                u64 prev_rip;
                u32 hits;
            };
            static thread_local WatchSlot wp_slots[] = {
                {0x800655f88ull, "broken.id ", false, 0, 0, 0},
                {0x800655f90ull, "broken.qA ", false, 0, 0, 0},
                {0x800655f98ull, "broken.qB ", false, 0, 0, 0},
                {0x800655d10ull, "healthy.id", false, 0, 0, 0},
                {0x800655d18ull, "healthy.qA", false, 0, 0, 0},
                {0x800655d20ull, "healthy.qB", false, 0, 0, 0},
            };
            for (auto& s : wp_slots) {
                u64 cur = 0;
                if (!SafeReadU64(s.addr, &cur)) {
                    continue; // not mapped yet; poll again next boundary
                }
                if (!s.inited) {
                    s.inited = true;
                    s.last = cur;
                    LOG_ERROR(Core,
                              "[WATCH] {} first readable [{:#x}] = {:#x} "
                              "(at block {:#x})",
                              s.tag, s.addr, cur, cur_rip);
                } else if (cur != s.last) {
                    if (s.hits < 256) {
                        ++s.hits;
                        // Poll runs at each block entry: the value now
                        // reflects everything the PREVIOUS block (and any
                        // HLE it called) did -- that block is the writer.
                        LOG_ERROR(Core,
                                  "[WATCH] {} [{:#x}] changed {:#x} -> {:#x} "
                                  "(writer block = {:#x}; observed at "
                                  "boundary {:#x})",
                                  s.tag, s.addr, s.last, cur, s.prev_rip,
                                  cur_rip);
                        // Queue the writer's host code for a one-shot dump
                        // (reuses the existing zeroing-block dump machinery
                        // in CheckRegisterCorruption). Any write to the
                        // BROKEN record is the headline event; the healthy
                        // record's first write identifies the init function.
                        if (g_watch_zeroing_block == 0) {
                            g_watch_zeroing_block = s.prev_rip;
                        }
                    }
                    s.last = cur;
                }
                s.prev_rip = cur_rip;
            }
        }
    }

    // Keep recording 0x80030e680 invocations (turned out NOT to be the failing
    // caller, but harmless to keep for context).
    if (cur_rip == 0x80030e680ull) {
        DumpFaultStruct(state);
    }

    // THE failing call: entry to the prologue 0x8002f04d0 with the bad incoming
    // rcx (= guest gpr index 1) that the prologue copies into r15. r15 ends 0x3f,
    // and the prologue does r15 = rcx, so rcx == 0x3f here identifies the exact
    // failing invocation. Dump the recent block-RIP ring so the predecessor block
    // (the REAL caller, which set rcx) is revealed, plus the incoming GPRs.
    // One-shot.
    if (cur_rip == 0x8002f04d0ull && state->gpr[1] == 0x3full) {
        static thread_local bool caller_dumped = false;
        if (!caller_dumped) {
            caller_dumped = true;
            LOG_ERROR(Core,
                      "[CALLER] entering prologue 0x8002f04d0 with rcx=0x3f "
                      "(-> r15). Incoming GPRs:");
            for (int i = 0; i < 16; ++i) {
                LOG_ERROR(Core, "[CALLER]   {} = {:#x}", kGprNames[i], state->gpr[i]);
            }
            LOG_ERROR(Core, "[CALLER] recent block sequence (newest last):");
            // g_rip_ring_pos has already been incremented for this (current)
            // entry; walk the last 16 entries oldest-first. The entry just before
            // the current one (0x8002f04d0) is the real caller block.
            for (int k = 16; k >= 1; --k) {
                const u32 idx = (g_rip_ring_pos - (u32)k) & 31;
                LOG_ERROR(Core, "[CALLER]   block {:#x}", g_rip_ring[idx]);
            }

            // The snapshot ring records all 16 GPRs at every boundary, so it
            // already holds rcx (gpr[1]) at each recent block. Dump (rip, rcx)
            // oldest-first so we can SEE the block at which rcx first became 0x3f
            // — that block is the exact producer, no hand-reconstruction needed.
            LOG_ERROR(Core, "[CALLER] rcx history at recent boundaries (rip : rcx):");
            const u32 live = (g_snap_pos < (u32)kSnapDepth) ? g_snap_pos
                                                            : (u32)kSnapDepth;
            const u32 show = (live < 20u) ? live : 20u;
            for (u32 k = show; k >= 1; --k) {
                const u32 slot = (g_snap_pos - k) % kSnapDepth;
                if (!g_snaps[slot].valid) continue;
                LOG_ERROR(Core, "[CALLER]   {:#x} : rcx={:#x}",
                          g_snaps[slot].rip, g_snaps[slot].gpr[1]);
            }

            // v5 [VERDICT]: replay the caller's argument computation FROM
            // MEMORY and compare against the actual rcx. The producer chain
            // (caller block 0x80030e680, fully disassembled in the v4
            // session) is:
            //
            //   off = dword[r14+rax+0x30]
            //   blob = qword[rbp-0x68]
            //   id  = word[blob+off]
            //   ZF  = ((dword[r14+rax+0x3c] & id) == 0)
            //   sel = blob + off + (ZF ? 0x10 : 8)        ; test/cmove
            //   rdx = qword[sel]
            //   idx = dword[r14+rax+0x38]
            //   rcx_arg = (idx << 4) + rdx
            //
            // r14/rax/rbp are live and untouched through the caller's tail
            // (verified against the v4 disassembly: no writes before the
            // call), so every input is recomputable right here at the
            // boundary. Outcomes:
            //   replay == actual rcx (0x3f)  -> the selected qword GENUINELY
            //       holds 0x3f: the JIT executed the selection faithfully
            //       and the corruption is UPSTREAM in whoever wrote
            //       blob+off+{8,0x10}. Next: point [WATCH] at that address.
            //   replay != actual rcx         -> the data says a different
            //       qword should have been picked: ZF was WRONG at runtime.
            //       The only instructions between the flag-setting test and
            //       the cmove are two LEAs -> audit EmitLea /
            //       EmitNarrowArith32's flag store / EmitJccCondition.
            // Both qwords are dumped either way so the record shape is
            // visible (RSDK two-storage select: bit 15 of id picks).
            {
                const u64 r14v = state->gpr[14];
                const u64 raxv = state->gpr[0];
                const u64 rbpv = state->gpr[5];
                const u64 base = r14v + raxv;
                u32 off = 0, mask = 0, idx = 0;
                u64 blob = 0;
                std::memcpy(&off,  reinterpret_cast<const void*>(base + 0x30), 4);
                std::memcpy(&mask, reinterpret_cast<const void*>(base + 0x3c), 4);
                std::memcpy(&idx,  reinterpret_cast<const void*>(base + 0x38), 4);
                std::memcpy(&blob, reinterpret_cast<const void*>(rbpv - 0x68), 8);
                u16 id = 0;
                u64 qa = 0, qb = 0;
                if (blob != 0) {
                    std::memcpy(&id, reinterpret_cast<const void*>(blob + off), 2);
                    std::memcpy(&qa, reinterpret_cast<const void*>(blob + off + 8), 8);
                    std::memcpy(&qb, reinterpret_cast<const void*>(blob + off + 0x10), 8);
                }
                const bool zf = ((mask & id) == 0);
                const u64 sel = zf ? qb : qa;
                const u64 replay = (static_cast<u64>(idx) << 4) + sel;
                LOG_ERROR(Core,
                          "[VERDICT] off={:#x} blob={:#x} id={:#x} mask={:#x} "
                          "idx={:#x} q@+8={:#x} q@+0x10={:#x} zf(expected)={} "
                          "sel={:#x} replay={:#x} actual_rcx={:#x} => {}",
                          off, blob, id, mask, idx, qa, qb, zf, sel, replay,
                          state->gpr[1],
                          (replay == state->gpr[1])
                              ? "FAITHFUL SELECTION - corrupt qword in memory; "
                                "hunt the WRITER of the selected slot"
                              : "SELECTION MISMATCH - ZF was wrong at runtime; "
                                "audit test/lea/cmove emitters");
            }
        }
    }
}

void CheckRegisterCorruption(Runtime* rt, GuestState* state, u64 cur_rip) {
    // Field-watchpoint handoff: if the watched field was just zeroed, dump the
    // host bytes of the block that wrote it (we have Runtime* here, which the
    // watchpoint in RecordBlockBoundary does not). One-shot.
    if (!g_watch_dump_done && g_watch_zeroing_block != 0 && rt != nullptr) {
        g_watch_dump_done = true;
        const u64 writer = g_watch_zeroing_block;
        LOG_ERROR(Core,
                  "[WATCH] dumping host bytes of field-zeroing block {:#x}:",
                  writer);
        BlockCache& bc = const_cast<BlockCache&>(rt->GetBlockCache());
        if (void* hp = bc.Lookup(writer)) {
            SafeDumpBlockCode(rt, writer, hp, "WATCH");
        } else {
            LOG_ERROR(Core, "[WATCH]   block {:#x} not found in block cache", writer);
        }
    }

    // Gap detector: flag any GPR (except rsp/rbp) that has entered the
    // unmapped gap, naming the block after which it was first seen.
    for (int i = 0; i < 16; ++i) {
        if (i == 4 || i == 5) continue;  // skip rsp/rbp
        const u64 v = state->gpr[i];
        if (!IsCorruptGuestValue(v)) continue;
        if (g_reported_sites & (1ull << i)) continue;

        g_reported_sites |= (1ull << i);
        LOG_ERROR(Core,
                  "[RAXTRACE] guest {} = {:#x} (UNMAPPED gap) first seen after "
                  "block rip={:#x}; next rip={:#x}",
                  kGprNames[i], v, g_last_block_rip, cur_rip);
        if (rt != nullptr && g_last_block_rip != 0) {
            BlockCache& bc = const_cast<BlockCache&>(rt->GetBlockCache());
            if (void* hp = bc.Lookup(g_last_block_rip)) {
                SafeDumpBlockCode(rt, g_last_block_rip, hp, "RAXTRACE");
            }
        }
    }
    g_last_block_rip = cur_rip;

    // Value-origin walk: for each GPR now in the gap and not yet attributed,
    // walk the snapshot ring backward to find the block that produced the
    // good->bad transition.
    for (int i = 0; i < 16; ++i) {
        if (i == 4 || i == 5) continue;
        const u64 bad = state->gpr[i];
        if (!IsCorruptGuestValue(bad)) continue;
        if (g_origin_reported & (1ull << i)) continue;

        const u32 live = (g_snap_pos < (u32)kSnapDepth) ? g_snap_pos
                                                        : (u32)kSnapDepth;
        if (live < 2) continue;

        u64 prev_val = 0;
        u64 producer_rip = 0;
        bool found = false;
        bool seen_bad_run = false;
        for (u32 k = 1; k <= live; ++k) {
            const u32 slot = (g_snap_pos - k) % kSnapDepth;
            if (!g_snaps[slot].valid) break;
            const u64 v = g_snaps[slot].gpr[i];
            if (v == bad) {
                seen_bad_run = true;
                continue;
            }
            if (seen_bad_run) {
                // The block about-to-run at this last-good boundary is the
                // producer (snaps[].rip labels the block about to run; its
                // gpr[] is the prior block's output).
                prev_val = v;
                producer_rip = g_snaps[slot].rip;
                found = true;
            }
            break;
        }

        if (found) {
            g_origin_reported |= (1ull << i);
            LOG_ERROR(Core,
                      "[GPRSNAP] {} VALUE-ORIGIN: block {:#x} changed it "
                      "from {:#x} -> {:#x} (bad now persists into rip={:#x})",
                      kGprNames[i], producer_rip, prev_val, bad, cur_rip);
            if (rt != nullptr && producer_rip != 0) {
                BlockCache& bc = const_cast<BlockCache&>(rt->GetBlockCache());
                if (void* hp = bc.Lookup(producer_rip)) {
                    SafeDumpBlockCode(rt, producer_rip, hp, "GPRSNAP");
                }
            }
        } else if (seen_bad_run) {
            g_origin_reported |= (1ull << i);
            LOG_ERROR(Core,
                      "[GPRSNAP] {} VALUE-ORIGIN: bad value {:#x} predates "
                      "the {}-deep snapshot ring (producer older than "
                      "rip={:#x}); increase kSnapDepth to catch it",
                      kGprNames[i], bad, kSnapDepth,
                      g_snaps[(g_snap_pos - live) % kSnapDepth].rip);
        }
    }

    // One-shot block-RIP ring dump: once any corruption has been reported,
    // emit the last 32 block RIPs oldest->newest so the loop cycle is visible.
    if (g_reported_sites != 0 && !g_rip_ring_dumped) {
        g_rip_ring_dumped = true;
        for (u32 k = 0; k < 32; ++k) {
            const u32 idx = (g_rip_ring_pos + k) & 31;
            LOG_ERROR(Core, "[RAXTRACE] ring[{}] block rip={:#x}", k, g_rip_ring[idx]);
        }
    }
}

void CheckHleReturn(std::string_view name, u64 host_fn, u64 rax) {
    static thread_local bool hle_bad_reported = false;
    const u64 top = rax >> 32;
    if (!hle_bad_reported && top >= 0x3 && top <= 0x7) {
        hle_bad_reported = true;
        LOG_ERROR(Core,
                  "[RAXTRACE] HLE stub '{}' (host={:#x}) returned rax={:#x} "
                  "in UNMAPPED gap — likely the bad-pointer source",
                  name.empty() ? std::string_view{"<unregistered>"} : name,
                  host_fn, rax);
    }
}

// Bridge ABI-integrity check. The HLE bridge calls a host C++ function that is
// declared sysv_abi; per AMD64 SysV the callee MUST preserve RBX, RBP, R12,
// R13, R14, R15. The guest's copies of those live in GuestState.gpr[] and the
// bridge never touches them, so after the call they must be byte-identical to
// before. If any differs, either (a) the host callee violated the ABI, (b) a
// guest callback re-entered on this GuestState and its save/restore is wrong,
// or (c) something else mutated GuestState across the call — all genuine
// runtime bugs, not upstream guest divergence. We capture a 6-register snapshot
// before the call (caller passes it in) and compare here, after.
//
// One-shot per distinct (name, register) so a hot call that legitimately... no:
// there is no legitimate divergence. We report every distinct offending call
// (capped) with the exact register, before/after values, and the HLE name, so a
// real corruption is impossible to miss and easy to localize.
// (BridgeCalleeSaved is declared in runtime_diagnostics.h so the bridge can
// build the snapshot; we just define the checker here.)
void BridgeCheckCalleeSaved(std::string_view name, u64 host_fn,
                            const BridgeCalleeSaved& before,
                            GuestState* state) {
    if (state == nullptr) return;
    static thread_local u32 reports = 0;
    if (reports >= 64) return;

    // GuestState gpr[] index mapping: RBX=3, RBP=5, R12=12, R13=13, R14=14, R15=15.
    struct Chk { const char* nm; u64 was; u64 now; };
    const Chk checks[] = {
        {"rbx", before.rbx, state->gpr[3]},
        {"rbp", before.rbp, state->gpr[5]},
        {"r12", before.r12, state->gpr[12]},
        {"r13", before.r13, state->gpr[13]},
        {"r14", before.r14, state->gpr[14]},
        {"r15", before.r15, state->gpr[15]},
    };
    for (const Chk& c : checks) {
        if (c.was != c.now) {
            ++reports;
            LOG_ERROR(Core,
                      "[BRIDGE-ABI] callee-saved {} CORRUPTED across HLE call "
                      "'{}' (host={:#x}): {:#x} -> {:#x}",
                      c.nm,
                      name.empty() ? std::string_view{"<unregistered>"} : name,
                      host_fn, c.was, c.now);
            if (reports >= 64) {
                LOG_ERROR(Core, "[BRIDGE-ABI] (further reports suppressed)");
                break;
            }
        }
    }
}

// Record an HLE bridge call into the rolling ring (name + the six SysV integer-
// arg registers). We don't log live — early-boot HLE traffic would flood — but
// keep the last kHleRingDepth and dump them at the fault point (DumpHleRing).
// Hunting a stub that receives an output-struct pointer it never populates: the
// guest later reads a stale field and computes the bad r15. rdi is arg0; the
// struct base the faulting computation reads (r14+rax+0x38) is reachable from an
// arg or from r14 itself.
void LogHleCall(std::string_view name, u64 host_fn, u64 guest_return_addr,
                u64 rdi, u64 rsi, u64 rdx, u64 rcx, u64 r8, u64 r9) {
    HleCallRec& r = g_hle_ring[g_hle_ring_pos % kHleRingDepth];
    // Copy the (possibly very long, NID-mangled) name into the fixed record,
    // truncating safely. Empty name -> "<unregistered>".
    std::string_view n = name.empty() ? std::string_view{"<unregistered>"} : name;
    const size_t cap = sizeof(r.name) - 1;
    const size_t len = (n.size() < cap) ? n.size() : cap;
    for (size_t i = 0; i < len; ++i) r.name[i] = n[i];
    r.name[len] = '\0';
    r.host_fn = host_fn;
    r.ret_to  = guest_return_addr;
    r.args[0] = rdi; r.args[1] = rsi; r.args[2] = rdx;
    r.args[3] = rcx; r.args[4] = r8;  r.args[5] = r9;
    r.valid = true;
    g_hle_ring_pos++;
}

// Dump the rolling HLE-call window oldest-first. Called from MaybeDumpPreBlock
// when the fault block is about to run, so the log shows the stubs that ran most
// recently before the bad struct read — the prime suspects for a stub that was
// handed an output buffer it never populated.
void DumpHleRing() {
    const u32 live = (g_hle_ring_pos < (u32)kHleRingDepth) ? g_hle_ring_pos
                                                           : (u32)kHleRingDepth;
    LOG_ERROR(Core, "[HLECALL] last {} HLE calls before fault (oldest first):", live);
    for (u32 k = live; k >= 1; --k) {
        const u32 slot = (g_hle_ring_pos - k) % kHleRingDepth;
        const HleCallRec& r = g_hle_ring[slot];
        if (!r.valid) continue;
        LOG_ERROR(Core,
                  "[HLECALL]   {} host={:#x} ret_to={:#x} | rdi={:#x} rsi={:#x} "
                  "rdx={:#x} rcx={:#x} r8={:#x} r9={:#x}",
                  r.name, r.host_fn, r.ret_to,
                  r.args[0], r.args[1], r.args[2], r.args[3], r.args[4], r.args[5]);
    }
}

// Public fault-time dump: callable from the signal handler at the moment of an
// unhandled fault. Unlike MaybeDumpPreBlock (gated on a hard-coded block-entry
// RIP), this fires for ANY crash including mid-block faults, since the handler
// always reaches it. Dumps the HLE-call window plus the recent block-RIP
// sequence so we can see which stub ran last and the control-flow path into the
// fault. Reads thread-local rings only — no allocation, handler-safe.
void DumpHleRingNow() {
    DumpHleRing();
    const u32 live = (g_rip_ring_pos < 32u) ? g_rip_ring_pos : 32u;
    LOG_ERROR(Core, "[HLECALL] recent block-RIP sequence before fault (oldest first):");
    for (u32 k = live; k >= 1; --k) {
        const u32 idx = (g_rip_ring_pos - k) & 31;
        LOG_ERROR(Core, "[HLECALL]   block {:#x}", g_rip_ring[idx]);
    }
}

// The block that actually faults. Reached via a jump-table/dispatch path;
// it executes `mov eax,[rdx]` with a poisoned guest rdx (a small non-pointer
// like 0x6), so the load dereferences unmapped low memory. The earlier
// 0x800274c09 target entered with clean registers, so it was upstream of the
// damage; this is the block where the bad value is first *used*.
constexpr u64 kFaultBlockRip = 0x8002f05e2ull;

// The faulting load is `mov eax,[r15+0xc]`: its base is guest r15 (GuestState
// offset 0x78 = index 15), and the crash shows r15 = 0x3f (a small integer, not
// a pointer) producing data_addr 0x4b. r15 — NOT rdx — is the corrupted base.
// (An earlier revision watched rdx, the host scratch reg the EA helper computes
// into; that is unrelated to the guest base and produced a misleading origin.)
constexpr int kBaseIdx = 15;

// A "plausible guest pointer" is anything that could legitimately be loaded
// from: dmem (top dword 0x1/0x2), code/heap (0x8), or the anon stack window.
// The faulting rdx values (0x6, 0x3f, 0x0) are none of these — they are
// small integers that were never meant to be an address. We treat
// "about to deref a non-pointer" as the trap condition rather than reusing
// IsCorruptGuestValue, whose gap-pointer test (top dword 0x3..0x7) does NOT
// match a small integer like 0x6 and so would never fire here.
bool IsImplausiblePointer(u64 v) {
    const u64 top = v >> 32;
    const bool dmem      = (top == 0x1 || top == 0x2);
    const bool code_heap = (top == 0x8);
    const bool anon_stack = (v >= 0x7ef000000ull && v < 0x800000000ull);
    return !(dmem || code_heap || anon_stack);
}

// Dump the struct the caller block (0x80030e680) reads to compute r15. From its
// disassembly: base = guest r14 + guest rax, then it reads fields at base+0x30,
// +0x38 (the value V that feeds r15 = (V<<4)+rdx), +0x3c, +0x40, +0x44. We log
// base and a window of dwords around those fields so we can see whether the
// struct is populated or stale. Fires once. Guest memory is mapped in the
// emulator's host address space, so a guest VA is directly dereferenceable IF
// mapped; we guard with IsImplausiblePointer so the diagnostic can't itself
// fault on an unmapped/garbage base. The read is best-effort: a plausible-looking
// but actually-unmapped base could still fault, but the prior crash shows this
// path's pointers are in mapped dmem/code, so the guard is adequate here.
void DumpFaultStruct(GuestState* state) {
    // Record every invocation's key inputs into a small ring, dumped at the real
    // fault. The caller 0x80030e680 loops over a 0x30-stride array at r14 and, on
    // the failing iteration, loads rcx = *(u32*)(r14 + rax + 0x60) (disassembly
    // of 0x80030e680: rdx=r14+rax; rdx+=0x30; ... the field that feeds rcx is at
    // element-relative +0x30, i.e. base+0x60 from the array start when rax=0x30).
    // So record that field, not the earlier mis-identified +0x38.
    const u64 r14 = state->gpr[14];
    const u64 rax = state->gpr[0];
    const u64 rdx = state->gpr[2];
    const u64 base = r14 + rax;
    const bool base_ok = !IsImplausiblePointer(base);
    const u32 V = base_ok ? *reinterpret_cast<const volatile u32*>(base + 0x30) : 0xFFFFFFFFu;

    FaultCallRec& r = g_fault_ring[g_fault_ring_pos % kFaultRingDepth];
    r.r14 = r14; r.rax = rax; r.rdx = rdx; r.base = base;
    r.base_ok = base_ok; r.V = V;
    r.predicted_r15 = V;  // v3 leftover; kept populated but no longer
                          // reported -- the first-load value is NOT the
                          // call argument (see the v4 note at the dump).
    r.valid = true;
    g_fault_ring_pos++;

    // One-shot: dump the whole array region from r14 (several 0x30-byte elements)
    // so we see element 0, element 1 (the failing one, field +0x30 => array+0x60),
    // and context. r14 is the stable array base across iterations.
    static thread_local bool array_dumped = false;
    if (!array_dumped && !IsImplausiblePointer(r14)) {
        array_dumped = true;
        LOG_ERROR(Core, "[ARRAY] dump from r14={:#x} (0x30-stride elements):", r14);
        for (int off = 0x0; off <= 0xC0; off += 0x4) {
            const u32 w = *reinterpret_cast<const volatile u32*>(r14 + off);
            const int elem = off / 0x30;
            const int eoff = off % 0x30;
            LOG_ERROR(Core, "[ARRAY]   [r14+{:#04x}] (elem{} +{:#04x}) = {:#010x}",
                      off, elem, eoff, w);
        }
    }
}

void DumpFaultRing() {
    const u32 live = (g_fault_ring_pos < (u32)kFaultRingDepth) ? g_fault_ring_pos
                                                              : (u32)kFaultRingDepth;
    LOG_ERROR(Core, "[STRUCT] last {} caller(0x80030e680) invocations (oldest first):",
              live);
    for (u32 k = live; k >= 1; --k) {
        const u32 slot = (g_fault_ring_pos - k) % kFaultRingDepth;
        const FaultCallRec& r = g_fault_ring[slot];
        if (!r.valid) continue;
        if (r.base_ok) {
            // v4 NOTE: "-> rcx(=r15)" was the v3 provenance hypothesis and
            // is WRONG for this block. Disassembly shows rcx is rewritten
            // at least twice after the first load within 45 bytes (lea/
            // cmove at +0x17..+0x1c, mov ecx,[rcx] at +0x28), and the call
            // argument's real producer is a 64-bit instruction in the
            // block's tail (the known-good arg 0x211601cc4 exceeds 32
            // bits; every rcx write in the old window was 32-bit). The
            // value below is the FIRST-LOAD offset field only.
            LOG_ERROR(Core,
                      "[STRUCT]   r14={:#x} rax={:#x} base={:#x} rdx={:#x} "
                      "first-load [base+0x30]={:#x} (NOT the call arg; see v4 note)",
                      r.r14, r.rax, r.base, r.rdx, r.V);
        } else {
            LOG_ERROR(Core,
                      "[STRUCT]   r14={:#x} rax={:#x} base={:#x} rdx={:#x} "
                      "(base implausible — not read)",
                      r.r14, r.rax, r.base, r.rdx);
        }
    }
}

void MaybeDumpPreBlock(GuestState* state) {
    static thread_local bool dumped = false;
    if (dumped || state->rip != kFaultBlockRip) {
        return;
    }

    // Only treat this as THE event when r15 is actually unusable. The block is
    // entered cleanly (valid r15) many times before the faulting entry; if we
    // consumed the one-shot on the first clean visit we'd suppress the crash
    // dump. So gate the one-shot on the bad value, and on clean visits fall
    // through silently without setting `dumped`.
    const u64 bad = state->gpr[kBaseIdx];
    if (!IsImplausiblePointer(bad)) {
        return;
    }
    dumped = true;

    LOG_ERROR(Core,
              "[RAXTRACE] about to enter faulting block {:#x}; guest GPRs:",
              kFaultBlockRip);
    for (int i = 0; i < 16; ++i) {
        LOG_ERROR(Core, "[RAXTRACE]   {} = {:#x}", kGprNames[i], state->gpr[i]);
    }

    // Walk the snapshot ring backward to find the block that last changed r15
    // from a plausible pointer to the current bad value. snaps[].rip labels the
    // block ABOUT to run at that boundary, and snaps[].gpr[] is the prior
    // block's *output* — so the rip recorded at the last-good boundary names
    // the producer.
    const u32 live = (g_snap_pos < (u32)kSnapDepth) ? g_snap_pos
                                                    : (u32)kSnapDepth;
    bool seen_bad_run = false;
    for (u32 k = 1; k <= live; ++k) {
        const u32 slot = (g_snap_pos - k) % kSnapDepth;
        if (!g_snaps[slot].valid) break;
        const u64 v = g_snaps[slot].gpr[kBaseIdx];
        if (v == bad) {
            seen_bad_run = true;
            continue;
        }
        if (seen_bad_run) {
            LOG_ERROR(Core,
                      "[GPRSNAP] r15 VALUE-ORIGIN: block {:#x} changed it "
                      "from {:#x} -> {:#x} before faulting block {:#x}",
                      g_snaps[slot].rip, v, bad, kFaultBlockRip);
        }
        break;
    }
    static thread_local bool dumped_guest_bytes = false;
    if (!dumped_guest_bytes && state->rip == 0x8002f05e2ull) {
        dumped_guest_bytes = true;
        // v4: the 48-byte window proved too small -- the caller block
        // 0x80030e680's FINAL rcx producer (the instruction that built the
        // corrupt 0x3f argument) lies past byte 45, as does its terminator.
        // Every rcx write inside the old window was 32-bit, and the known-
        // good argument 0x211601cc4 cannot come from a 32-bit load, so the
        // producer is provably in the tail. Dump 0x150 bytes per block,
        // chunked 48/line for log sanity; capstone the result offline.
        constexpr int kWin = 0x150;
        for (u64 rip : {0x8002f04d0ull, 0x8002f05e2ull, 0x80030e680ull}) {
            const u8* p = reinterpret_cast<const u8*>(rip); // identity-mapped guest VA
            for (int base = 0; base < kWin; base += 48) {
                char hex[48 * 3 + 1];
                for (int k = 0; k < 48; ++k) {
                    static const char* d = "0123456789abcdef";
                    hex[k * 3 + 0] = d[(p[base + k] >> 4) & 0xF];
                    hex[k * 3 + 1] = d[p[base + k] & 0xF];
                    hex[k * 3 + 2] = ' ';
                }
                hex[48 * 3] = '\0';
                LOG_ERROR(Core, "[GUESTBYTES] {:#x}+{:#x}: {}", rip, base, hex);
            }
        }
    }
    static thread_local bool dumped_loop = false;
    if (!dumped_loop && state->rip == 0x80030e680ull) {
        const u64 r14 = state->gpr[14]; // container base
        const u64 rax = state->gpr[0];  // entry offset
        const u8* entry = reinterpret_cast<const u8*>(r14 + rax);
        if (*reinterpret_cast<const u32*>(entry + 0x38) == 0x3fu) { // the bad iteration
            dumped_loop = true;
            LOG_ERROR(Core, "[LOOP] r14={:#x} rax={:#x} rbp={:#x}", r14, rax, state->gpr[5]);
            // the two loop-advance blocks + a re-dump of the lookup block
            for (u64 rip : {0x80030e650ull, 0x80030e6c7ull, 0x80030e680ull}) {
                const u8* p = reinterpret_cast<const u8*>(rip);
                char hex[64 * 3 + 1];
                static const char* d = "0123456789abcdef";
                for (int k = 0; k < 64; ++k) {
                    hex[k * 3] = d[(p[k] >> 4) & 0xF];
                    hex[k * 3 + 1] = d[p[k] & 0xF];
                    hex[k * 3 + 2] = ' ';
                }
                hex[64 * 3] = '\0';
                LOG_ERROR(Core, "[LOOPBYTES] {:#x}: {}", rip, hex);
            }
            // the container entry the bad iteration is reading (96 bytes around it)
            const u8* e = entry;
            char hx[96 * 3 + 1];
            static const char* d2 = "0123456789abcdef";
            for (int k = 0; k < 96; ++k) {
                hx[k * 3] = d2[(e[k] >> 4) & 0xF];
                hx[k * 3 + 1] = d2[e[k] & 0xF];
                hx[k * 3 + 2] = ' ';
            }
            hx[96 * 3] = '\0';
            LOG_ERROR(Core, "[ENTRY] {:#x}: {}", (u64)entry, hx);
        }
    }
    static thread_local bool dumped_302 = false;
    if (!dumped_302 && state->rip == 0x800302f60ull) {
        dumped_302 = true;
        const u8* p = reinterpret_cast<const u8*>(0x800302f60ull);
        char hex[96 * 3 + 1];
        static const char* d = "0123456789abcdef";
        for (int k = 0; k < 96; ++k) {
            hex[k * 3] = d[(p[k] >> 4) & 0xF];
            hex[k * 3 + 1] = d[p[k] & 0xF];
            hex[k * 3 + 2] = ' ';
        }
        hex[96 * 3] = '\0';
        LOG_ERROR(Core, "[GUESTBYTES] 0x800302f60: {}", hex);
    }
    // Dump the recent HLE-call window: a stub that was handed an output buffer
    // it never populated is the prime suspect for the stale struct field the
    // caller read to compute the bad r15.
    DumpHleRing();

    // Dump the recent caller(0x80030e680) invocations so we see the actual
    // inputs (r14/rax/rdx/base/V) of the call that produced the bad r15.
    DumpFaultRing();
}

#ifdef ARCH_X86_64
// Disassemble a freshly compiled block straight into the log, once per guest
// RIP, so blocks are human-readable without any offline tool. Off by default
// (this floods the log); enable with SHADPS4_DUMP_BLOCKS=1. Output is one
// header line plus one line per host instruction, with r13-relative memory
// operands annotated by the guest register/state field they touch (r13 is
// pinned to the GuestState base). x86-only: it decodes the emitted HOST
// instructions with Zydis, whose formatter is x86; on arm64 the emitted code
// is AArch64, which Zydis can't format, so this is compiled out there.
void DumpCompiledBlockDisasm(u64 guest_rip, const void* host_ptr,
                             u64 emitted_size) {
    static const bool dump_all_blocks = [] {
        const char* e = std::getenv("SHADPS4_DUMP_BLOCKS");
        return e != nullptr && e[0] != '\0' && e[0] != '0';
    }();
    if (!dump_all_blocks) {
        return;
    }
    // De-dup per guest RIP so re-dispatched blocks don't re-print.
    static thread_local std::unordered_set<u64> seen_blocks;
    if (!seen_blocks.insert(guest_rip).second) {
        return;
    }
    constexpr u64 kMaxDump = 1024;
    u64 n = (emitted_size > kMaxDump) ? kMaxDump : emitted_size;
    const auto* code = reinterpret_cast<const u8*>(host_ptr);

    LOG_ERROR(Core, "block {:#x} ({} host bytes):", guest_rip, n);

    // Map an r13-relative displacement to a guest register / state field name.
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
            // Stop at zero-padding: the "used" delta can over-report a block's
            // true length, and the zeroed tail decodes as a meaningless run of
            // `add [rax], al` (00 00). A 00 00 here is never real emitted code.
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
            // Annotate the first r13-relative memory operand with the guest
            // slot it touches. A zero displacement is a valid slot access
            // ([r13] == guest RAX), so we read disp.value directly.
            const char* note = nullptr;
            for (u32 i = 0; i < insn.operand_count; ++i) {
                if (ops[i].type == ZYDIS_OPERAND_TYPE_MEMORY &&
                    ops[i].mem.base == ZYDIS_REGISTER_R13) {
                    note = slot_name(static_cast<u64>(ops[i].mem.disp.value));
                    break;
                }
            }
            if (note != nullptr) {
                LOG_ERROR(Core, "  +{:#04x}  {}    ; guest {}", off, text, note);
            } else {
                LOG_ERROR(Core, "  +{:#04x}  {}", off, text);
            }
            off += insn.length;

            // A block ends with its terminator: an indirect jump through r14
            // (dispatcher loop) or r15 (clean/fatal exit). Stop after it.
            if (insn.mnemonic == ZYDIS_MNEMONIC_JMP &&
                ops[0].type == ZYDIS_OPERAND_TYPE_REGISTER &&
                (ops[0].reg.value == ZYDIS_REGISTER_R14 ||
                 ops[0].reg.value == ZYDIS_REGISTER_R15)) {
                break;
            }
        }
    }
}
#else
void DumpCompiledBlockDisasm(u64, const void*, u64) {}
#endif  // ARCH_X86_64

void OnBlockCompiled(u64 guest_rip, const void* host_ptr, u64 emitted_size) {
    // General dev tooling: full per-instruction disassembly, env-gated.
    DumpCompiledBlockDisasm(guest_rip, host_ptr, emitted_size);

    // Investigation-specific: raw host-byte dump for the blocks under active
    // investigation. Current target: the field-zeroing bug. The watchpoint
    // showed [0x8016a2270] going 0x801698770 -> 0 right after block 0x8008adec0,
    // and the fault is block 0x8008ae120 reading that now-null field. Dump both
    // so we can read the lifted store (0x8008adec0) and the lifted deref
    // (0x8008ae120) and decide mis-lift vs faithful-store-of-zero.
    if (guest_rip != 0x8008adec0ull && guest_rip != 0x8008ae120ull) {
        return;
    }
    static thread_local u64 dumped_mask = 0;
    const int slot = (guest_rip == 0x8008adec0ull) ? 0 : 1;
    if (dumped_mask & (1ull << slot)) {
        return;
    }
    dumped_mask |= (1ull << slot);
    // 768 bytes covers the largest blocks under investigation (the caller block
    // is 506 host bytes); the hex buffer scales with this constant. A spdlog line
    // of ~2.3 KB is fine for a one-shot diagnostic dump.
    constexpr u64 kMaxDump = 768;
    const u64 n = (emitted_size > kMaxDump) ? kMaxDump : emitted_size;
    const u8* p = reinterpret_cast<const u8*>(host_ptr);
    const char* digits = "0123456789abcdef";
    char hex[kMaxDump * 3 + 1];
    for (u64 i = 0; i < n; ++i) {
        hex[i * 3 + 0] = digits[(p[i] >> 4) & 0xF];
        hex[i * 3 + 1] = digits[p[i] & 0xF];
        hex[i * 3 + 2] = ' ';
    }
    hex[n * 3] = '\0';
    LOG_ERROR(Core, "[RAXTRACE] emitted block {:#x} ({} host bytes): {}",
              guest_rip, emitted_size, hex);
}

}  // namespace Diagnostics
}  // namespace Core::Runtime

#endif  // SHADPS4_RUNTIME_DIAGNOSTICS
