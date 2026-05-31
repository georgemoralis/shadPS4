// SPDX-FileCopyrightText: Copyright 2024 shadPS4 Emulator Project
// SPDX-License-Identifier: GPL-2.0-or-later

#include "core/cpu_runtime/runtime_diagnostics.h"

// The whole translation unit is empty unless diagnostics are enabled, so a
// release build links in nothing from here.
#ifdef SHADPS4_RUNTIME_DIAGNOSTICS

#include <string_view>
#include <unordered_set>

#include "common/arch.h"
#include "common/logging/log.h"
#include "core/cpu_runtime/block_cache.h"
#include "core/cpu_runtime/code_cache.h"
#include "core/cpu_runtime/runtime.h"

#ifdef ARCH_X86_64
#include <Zydis/Zydis.h>
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
    constexpr int kWindow = 256;
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

}  // namespace

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
}

void CheckRegisterCorruption(Runtime* rt, GuestState* state, u64 cur_rip) {
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

void MaybeDumpPreBlock(GuestState* state) {
    static thread_local bool dumped = false;
    if (!dumped && state->rip == 0x800274c09ull) {
        dumped = true;
        LOG_ERROR(Core,
                  "[RAXTRACE] about to enter faulting block 0x800274c09; guest GPRs:");
        for (int i = 0; i < 16; ++i) {
            LOG_ERROR(Core, "[RAXTRACE]   {} = {:#x}", kGprNames[i], state->gpr[i]);
        }
    }
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

    // Investigation-specific: raw host-byte dump for the bit-33 corruption
    // loop blocks only.
    if (guest_rip != 0x800274c00ull && guest_rip != 0x800274c09ull &&
        guest_rip != 0x800274caeull && guest_rip != 0x800302f60ull) {
        return;
    }
    static thread_local u64 dumped_mask = 0;
    const int slot = (guest_rip == 0x800274c00ull) ? 0
                   : (guest_rip == 0x800274c09ull) ? 1
                   : (guest_rip == 0x800274caeull) ? 2 : 3;
    if (dumped_mask & (1ull << slot)) {
        return;
    }
    dumped_mask |= (1ull << slot);
    constexpr u64 kMaxDump = 256;
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
