// SPDX-FileCopyrightText: Copyright 2026 shadPS4 Emulator Project
// SPDX-License-Identifier: GPL-2.0-or-later

#include "core/cpu_runtime/lifter/lifter.h"

#include <cstddef>
#include <cstdio>
#include <cstdlib> // getenv (SHADPS4_LIFTER_TRACE gate)
#include <Zydis/Zydis.h>
#include <xbyak/xbyak.h>

#include "common/assert.h"
#include "common/logging/log.h"
#include "core/cpu_runtime/code_cache.h"
#include "core/cpu_runtime/guest_state.h"

namespace Core::Runtime {

namespace {

// ============================================================================
// Constants and helpers
// ============================================================================

/// Maximum bytes emitted per block before we force a fallthrough.
/// Keeps compile time bounded for runaway code. Must be large enough
/// that a full BLOCK_GUEST_SIZE_CAP of dense SSE/AVX guest code (each
/// vector op can expand to many host bytes) fits with margin; 4 KiB was
/// too small and let Xbyak throw ERR_CODE_IS_TOO_BIG mid-block. Matches
/// the arm64 lifter's cap.
constexpr u64 BLOCK_HOST_SIZE_CAP = 16384;

/// Maximum guest bytes consumed per block before forcing a
/// fallthrough. Pathologically long basic blocks are bad for
/// dispatcher latency (no break-in checks until exit).
constexpr u64 BLOCK_GUEST_SIZE_CAP = 1024;

// ============================================================================
// Compile-time invariants
// ============================================================================
//
// The lifter makes several assumptions about external constants — the
// Zydis register enum, the GuestState layout, the ExitReason values.
// These are stable today and depended on throughout the file. Locking
// them in via static_assert means a future Zydis bump or guest-state
// refactor breaks the build at the assertion, rather than producing
// silently-wrong machine code at runtime.

// ---------------- Zydis register enum ordering ----------------
//
// `ZydisGprToIndex` (below) is the single point of register-name
// translation. It assumes Zydis lays out each width's GPRs in
// contiguous canonical AMD64 order:
//
//   64-bit:    RAX, RCX, RDX, RBX, RSP, RBP, RSI, RDI, R8..R15
//   32-bit:    EAX, ECX, EDX, EBX, ESP, EBP, ESI, EDI, R8D..R15D
//   16-bit:    AX,  CX,  DX,  BX,  SP,  BP,  SI,  DI,  R8W..R15W
//    8-bit-low:           AL,  CL,  DL,  BL
//    8-bit-low-ext:       SPL, BPL, SIL, DIL
//    8-bit-rex:           R8B..R15B
//
// If Zydis ever renumbers these (would also break every other
// consumer of the library), the asserts below catch it at compile
// time instead of letting the lifter emit wrong slot indices.

static_assert(ZYDIS_REGISTER_RCX - ZYDIS_REGISTER_RAX == 1,
              "Zydis 64-bit GPR enum order changed");
static_assert(ZYDIS_REGISTER_R15 - ZYDIS_REGISTER_RAX == 15,
              "Zydis 64-bit GPRs no longer contiguous over 16 entries");
static_assert(ZYDIS_REGISTER_ECX - ZYDIS_REGISTER_EAX == 1,
              "Zydis 32-bit GPR enum order changed");
static_assert(ZYDIS_REGISTER_R15D - ZYDIS_REGISTER_EAX == 15,
              "Zydis 32-bit GPRs no longer contiguous");
static_assert(ZYDIS_REGISTER_CX - ZYDIS_REGISTER_AX == 1,
              "Zydis 16-bit GPR enum order changed");
static_assert(ZYDIS_REGISTER_R15W - ZYDIS_REGISTER_AX == 15,
              "Zydis 16-bit GPRs no longer contiguous");
static_assert(ZYDIS_REGISTER_BL - ZYDIS_REGISTER_AL == 3,
              "Zydis 8-bit-low GPR enum (AL..BL) ordering changed");
static_assert(ZYDIS_REGISTER_DIL - ZYDIS_REGISTER_SPL == 3,
              "Zydis 8-bit extended-low GPR enum (SPL..DIL) ordering changed");
static_assert(ZYDIS_REGISTER_R15B - ZYDIS_REGISTER_R8B == 7,
              "Zydis 8-bit REX-prefixed GPR enum (R8B..R15B) ordering changed");

// Specific position invariants the lifter depends on. RSP must be at
// position 4 in the canonical AMD64 ordering (Zydis encodes it this
// way; we use kGuestRspIdx = 4 to mirror it in guest state).
static_assert((ZYDIS_REGISTER_RAX - ZYDIS_REGISTER_RAX) == 0,
              "RAX must map to slot 0");
static_assert((ZYDIS_REGISTER_RSP - ZYDIS_REGISTER_RAX) == 4,
              "RSP must map to slot 4 (canonical AMD64 ordering)");
static_assert((ZYDIS_REGISTER_RDI - ZYDIS_REGISTER_RAX) == 7,
              "RDI must map to slot 7 (SysV arg 1; HLE bridge depends on this)");
static_assert((ZYDIS_REGISTER_R15 - ZYDIS_REGISTER_RAX) == 15,
              "R15 must map to slot 15");

// ---------------- ExitReason values used by JIT-emitted code ----------------
//
// The lifter emits `mov dword[r13 + offsetof(exit_reason)], <imm>`
// at several exit paths. The constants come from casting ExitReason
// enumerators; the dispatcher and the test harness compare against
// the same enumerators. If ExitReason gets renumbered, JIT-emitted
// stores would write values the dispatcher no longer recognises.
// Lock the wire-format values down here so both sides have to be
// updated together.

static_assert(static_cast<u32>(ExitReason::BlockEnd) == 0,
              "ExitReason::BlockEnd must remain 0 (lifter normal-exit constant)");
static_assert(static_cast<u32>(ExitReason::UnsupportedInstruction) == 2,
              "ExitReason::UnsupportedInstruction must remain 2 (lifter unsupported-path)");

// ---------------- end of compile-time invariants ----------------

/// Map a Zydis GPR enum to a guest-state GPR index 0..15.
/// Returns -1 for non-GPR or unsupported registers.
///
/// Every width variant of the same physical register maps to the
/// same slot. The caller decides which bytes of the slot to read
/// or write based on `insn.operand_width` (or `ops[i].size`).
///
///   - 64-bit: RAX..R15        → 0..15
///   - 32-bit: EAX..R15D       → 0..15
///   - 16-bit: AX..R15W        → 0..15
///   -  8-bit (low):
///       AL/CL/DL/BL           → 0/1/2/3
///       SPL/BPL/SIL/DIL       → 4/5/6/7
///       R8B..R15B             → 8..15
///
/// The high-8 registers (AH/CH/DH/BH) are deliberately NOT handled.
/// They alias byte 1 of registers 0..3, requiring a different access
/// pattern. Compilers rarely emit them in modern 64-bit code. If a
/// real binary needs them, the caller falls through to the
/// unsupported-instruction path with diagnostics.
int ZydisGprToIndex(ZydisRegister r) {
    if (r >= ZYDIS_REGISTER_RAX && r <= ZYDIS_REGISTER_R15) {
        return r - ZYDIS_REGISTER_RAX;
    }
    if (r >= ZYDIS_REGISTER_EAX && r <= ZYDIS_REGISTER_R15D) {
        return r - ZYDIS_REGISTER_EAX;
    }
    if (r >= ZYDIS_REGISTER_AX && r <= ZYDIS_REGISTER_R15W) {
        return r - ZYDIS_REGISTER_AX;
    }
    // 8-bit low: AL, CL, DL, BL = indices 0..3.
    if (r >= ZYDIS_REGISTER_AL && r <= ZYDIS_REGISTER_BL) {
        return r - ZYDIS_REGISTER_AL;
    }
    // 8-bit "extended" low: SPL, BPL, SIL, DIL = indices 4..7.
    if (r >= ZYDIS_REGISTER_SPL && r <= ZYDIS_REGISTER_DIL) {
        return (r - ZYDIS_REGISTER_SPL) + 4;
    }
    // 8-bit REX-prefixed: R8B..R15B = indices 8..15.
    if (r >= ZYDIS_REGISTER_R8B && r <= ZYDIS_REGISTER_R15B) {
        return (r - ZYDIS_REGISTER_R8B) + 8;
    }
    // AH/CH/DH/BH and non-GPR registers (segment, MMX, XMM, etc.)
    // fall through to "unsupported".
    return -1;
}

/// Byte offset within GuestState for the n-th GPR.
constexpr u32 GprOffset(int idx) {
    return Offsets::Gpr + static_cast<u32>(idx) * 8;
}

/// Byte offset within GuestState for an 8-bit register, including the
/// high-byte registers AH/CH/DH/BH. Returns -1 if `r` is not an 8-bit
/// GPR. Low-byte and REX-low registers map to byte 0 of their parent
/// slot; AH/CH/DH/BH map to byte 1 of slots 0..3 (the parent qword is
/// little-endian, so bits 15:8 live at offset+1).
int ZydisGpr8ToByteOffset(ZydisRegister r) {
    // High-byte regs first (they're outside the contiguous low ranges).
    switch (r) {
        case ZYDIS_REGISTER_AH: return static_cast<int>(GprOffset(0)) + 1;
        case ZYDIS_REGISTER_CH: return static_cast<int>(GprOffset(1)) + 1;
        case ZYDIS_REGISTER_DH: return static_cast<int>(GprOffset(2)) + 1;
        case ZYDIS_REGISTER_BH: return static_cast<int>(GprOffset(3)) + 1;
        default: break;
    }
    if (r >= ZYDIS_REGISTER_AL && r <= ZYDIS_REGISTER_BL)
        return static_cast<int>(GprOffset(r - ZYDIS_REGISTER_AL));
    if (r >= ZYDIS_REGISTER_SPL && r <= ZYDIS_REGISTER_DIL)
        return static_cast<int>(GprOffset((r - ZYDIS_REGISTER_SPL) + 4));
    if (r >= ZYDIS_REGISTER_R8B && r <= ZYDIS_REGISTER_R15B)
        return static_cast<int>(GprOffset((r - ZYDIS_REGISTER_R8B) + 8));
    return -1;
}

/// Guest RSP index (canonical AMD64 order: RAX=0, RCX=1, RDX=2,
/// RBX=3, RSP=4, RBP=5, RSI=6, RDI=7, R8..R15=8..15).
constexpr int kGuestRspIdx = 4;

static_assert(kGuestRspIdx == (ZYDIS_REGISTER_RSP - ZYDIS_REGISTER_RAX),
              "kGuestRspIdx must match the canonical AMD64 RSP slot index "
              "(also assumed by the dispatcher, gateway, and HLE bridge)");

using namespace Xbyak::util;

/// Compute the effective address of a guest memory operand into a
/// host scratch register (always rdx). Uses rax/rcx as transient
/// scratch during the computation. After this returns, rdx holds
/// the guest virtual address described by the operand.
///
/// Supports:
///   - [base]
///   - [base + disp]
///   - [base + index*scale]
///   - [base + index*scale + disp]
///   - [disp32]                          (no base, no index)
///   - [base = RIP + disp]               (RIP-relative)
///
/// Where `base` is a 64-bit guest GPR, `index` is a 64-bit guest
/// GPR with scale 1/2/4/8, `disp` is a signed displacement, and
/// `RIP` is the address of the *next* instruction (so RIP-relative
/// disp resolves to next_rip + disp).
///
/// `next_rip` is the address immediately after the current
/// instruction (used for RIP-relative addressing). The caller
/// passes guest_rip + insn.length.
///
/// Returns true on success; false if the addressing mode isn't
/// supported (e.g. segment override, non-GPR base).
bool EmitEffectiveAddress(const ZydisDecodedOperandMem& mem,
                          u64 next_rip,
                          Xbyak::CodeGenerator& c) {
    // Segment overrides. DS/SS/CS/ES are flat (base 0) in long mode and need
    // no adjustment. FS/GS carry a guest-visible base used for TLS: the
    // effective address is seg_base + (base + index*scale + disp), where the
    // guest's fs_base/gs_base live in GuestState (fs_base is the thread's TCB,
    // installed by InitializeTLS/SetTcbBase). We compute the rest of the
    // address as usual and add the segment base at the end. The native backend
    // handles fs:/gs: by patching the guest code (cpu_patches); the JIT models
    // the segment bases directly, so it needs no patching and this also covers
    // the stack-canary (fs:[0x28]) and TCB (fs:[0]) accesses generically.
    int seg_base_off = -1;
    if (mem.segment == ZYDIS_REGISTER_FS) {
        seg_base_off = static_cast<int>(Offsets::FsBase);
    } else if (mem.segment == ZYDIS_REGISTER_GS) {
        seg_base_off = static_cast<int>(Offsets::GsBase);
    } else if (mem.segment != ZYDIS_REGISTER_DS &&
               mem.segment != ZYDIS_REGISTER_SS &&
               mem.segment != ZYDIS_REGISTER_CS &&
               mem.segment != ZYDIS_REGISTER_ES) {
        return false;  // genuinely unsupported segment
    }

    const bool has_base = (mem.base != ZYDIS_REGISTER_NONE);
    const bool has_index = (mem.index != ZYDIS_REGISTER_NONE);
    // Note: Zydis 4.1+ exposes mem.disp.has_displacement to distinguish
    // [base] from [base+0]. shadPS4's bundled Zydis (4.0.x) doesn't.
    // The distinction doesn't matter for emit: a zero displacement
    // produces no `add` (see the `if (disp != 0)` guard below).
    const s64 disp = mem.disp.value;

    // RIP-relative: base == RIP, no index. Address = next_rip + disp.
    // We constant-fold this into a single mov.
    if (has_base && mem.base == ZYDIS_REGISTER_RIP) {
        if (has_index) return false;  // RIP-relative with index is not a thing
        if (seg_base_off >= 0) return false;  // segment + RIP-relative: degenerate
        c.mov(rdx, static_cast<u64>(static_cast<s64>(next_rip) + disp));
        return true;
    }

    // Plain [disp32] absolute (no base, no index). With a segment override this
    // is the common TLS form (e.g. fs:[0], fs:[0x28]).
    if (!has_base && !has_index) {
        c.mov(rdx, static_cast<u64>(disp));
        if (seg_base_off >= 0) {
            c.add(rdx, qword[r13 + seg_base_off]);
        }
        return true;
    }

    // General case: rdx = base + index*scale + disp.
    // Start rdx at the base value (or 0 if no base).
    if (has_base) {
        const int base_idx = ZydisGprToIndex(mem.base);
        if (base_idx < 0) return false;
        c.mov(rdx, qword[r13 + GprOffset(base_idx)]);
    } else {
        c.xor_(rdx, rdx);
    }

    if (has_index) {
        const int index_idx = ZydisGprToIndex(mem.index);
        if (index_idx < 0) return false;
        // Load index into rax, scale it, add to rdx.
        c.mov(rax, qword[r13 + GprOffset(index_idx)]);
        switch (mem.scale) {
            case 1:  break;  // no shift
            case 2:  c.shl(rax, 1); break;
            case 4:  c.shl(rax, 2); break;
            case 8:  c.shl(rax, 3); break;
            default: return false;  // invalid SIB scale
        }
        c.add(rdx, rax);
    }

    if (disp != 0) {
        if (disp >= INT32_MIN && disp <= INT32_MAX) {
            c.add(rdx, static_cast<int>(disp));
        } else {
            c.mov(rax, static_cast<u64>(disp));
            c.add(rdx, rax);
        }
    }

    // Add the FS/GS segment base last, after the index/disp math.
    if (seg_base_off >= 0) {
        c.add(rdx, qword[r13 + seg_base_off]);
    }

    return true;
}

// ============================================================================
// Per-opcode emit functions
// ============================================================================
//
// Each emit function:
//   - Returns true if it handled the instruction and emitted code.
//   - Returns false if it couldn't handle the operand combination,
//     in which case the dispatcher falls through to EmitUnsupported.
//
// Conventions inside emit code:
//   - rax/rcx/rdx are scratch. JIT code is free to clobber them
//     between instruction boundaries.
//   - r13 is the GuestState pointer (set by the gateway, never
//     modified by JIT).
//   - r12, r14, r15 are reserved (dispatcher, exit stub, future use).
//
// `next_rip` is the guest RIP of the next instruction after this
// one. Used by RIP-relative addressing.

/// MOV: handles
///   - r64, r64         (reg-to-reg)
///   - r64, imm64       (full 10-byte form)
///   - r64, imm32-sx    (sign-extended)
///   - r64, [mem]       (load from memory)
///   - [mem], r64       (store to memory)
///   - [mem], imm32-sx  (store immediate to memory)
bool EmitMov(const ZydisDecodedInstruction& insn,
             const ZydisDecodedOperand* ops,
             u64 next_rip,
             Xbyak::CodeGenerator& c) {
    if (insn.operand_width != 64) {
        return false;  // 32/16/8-bit MOVs not in initial slice
    }
    const auto& dst = ops[0];
    const auto& src = ops[1];

    // ----- Memory destination -----
    if (dst.type == ZYDIS_OPERAND_TYPE_MEMORY) {
        // Compute effective address into rdx.
        if (!EmitEffectiveAddress(dst.mem, next_rip, c)) return false;

        if (src.type == ZYDIS_OPERAND_TYPE_REGISTER) {
            const int src_idx = ZydisGprToIndex(src.reg.value);
            if (src_idx < 0) return false;
            // Load src value into rax; store at [rdx].
            c.mov(rax, qword[r13 + GprOffset(src_idx)]);
            c.mov(qword[rdx], rax);
            return true;
        }
        if (src.type == ZYDIS_OPERAND_TYPE_IMMEDIATE) {
            // imm32 sign-extended to 64 bits. Load via rax (mov
            // qword[mem], imm32 doesn't exist as a single insn).
            c.mov(rax, src.imm.value.s);
            c.mov(qword[rdx], rax);
            return true;
        }
        return false;
    }

    // ----- Register destination -----
    if (dst.type != ZYDIS_OPERAND_TYPE_REGISTER) return false;
    const int dst_idx = ZydisGprToIndex(dst.reg.value);
    if (dst_idx < 0) return false;

    if (src.type == ZYDIS_OPERAND_TYPE_REGISTER) {
        const int src_idx = ZydisGprToIndex(src.reg.value);
        if (src_idx < 0) return false;
        c.mov(rax, qword[r13 + GprOffset(src_idx)]);
        c.mov(qword[r13 + GprOffset(dst_idx)], rax);
        return true;
    }

    if (src.type == ZYDIS_OPERAND_TYPE_IMMEDIATE) {
        // Both forms: REX.W + B8+r io (imm64), and REX.W + C7 /0 id
        // (imm32 sign-extended). Zydis gives us the resolved value.
        c.mov(rax, src.imm.value.s);
        c.mov(qword[r13 + GprOffset(dst_idx)], rax);
        return true;
    }

    if (src.type == ZYDIS_OPERAND_TYPE_MEMORY) {
        if (!EmitEffectiveAddress(src.mem, next_rip, c)) return false;
        c.mov(rax, qword[rdx]);
        c.mov(qword[r13 + GprOffset(dst_idx)], rax);
        return true;
    }

    return false;
}

/// MOV with 32-bit operand width.
///
/// x86-64 quirk worth being explicit about: writing to a 32-bit
/// register zero-extends into the full 64-bit register. So
/// `mov eax, ebx` actually clears the upper 32 bits of rax. We get
/// this for free by:
///   1. Loading the 32-bit source value into a host 32-bit reg
///      (e.g. `mov eax, dword[...]`), which zero-extends rax.
///   2. Storing rax as a full 64-bit qword into the dst's GPR slot.
///
/// Memory operands are 4 bytes for 32-bit MOVs (not 8). The dst's
/// upper 32 bits get zeroed only when the dst is a register; for
/// memory dst, only 4 bytes are written.
bool EmitMov32(const ZydisDecodedInstruction& insn,
               const ZydisDecodedOperand* ops,
               u64 next_rip,
               Xbyak::CodeGenerator& c) {
    if (insn.operand_width != 32) return false;
    const auto& dst = ops[0];
    const auto& src = ops[1];

    // ----- Memory destination -----
    if (dst.type == ZYDIS_OPERAND_TYPE_MEMORY) {
        // Compute effective address into rdx.
        if (!EmitEffectiveAddress(dst.mem, next_rip, c)) return false;

        if (src.type == ZYDIS_OPERAND_TYPE_REGISTER) {
            const int src_idx = ZydisGprToIndex(src.reg.value);
            if (src_idx < 0) return false;
            // Load low 32 bits of src; store 4 bytes at [rdx].
            c.mov(eax, dword[r13 + GprOffset(src_idx)]);
            c.mov(dword[rdx], eax);
            return true;
        }
        if (src.type == ZYDIS_OPERAND_TYPE_IMMEDIATE) {
            c.mov(dword[rdx], static_cast<s32>(src.imm.value.s));
            return true;
        }
        return false;
    }

    // ----- Register destination -----
    if (dst.type != ZYDIS_OPERAND_TYPE_REGISTER) return false;
    const int dst_idx = ZydisGprToIndex(dst.reg.value);
    if (dst_idx < 0) return false;

    if (src.type == ZYDIS_OPERAND_TYPE_REGISTER) {
        const int src_idx = ZydisGprToIndex(src.reg.value);
        if (src_idx < 0) return false;
        // 32-bit load into eax zero-extends rax. Storing rax as
        // qword gives the dst's upper 32 bits the required zero.
        c.mov(eax, dword[r13 + GprOffset(src_idx)]);
        c.mov(qword[r13 + GprOffset(dst_idx)], rax);
        return true;
    }

    if (src.type == ZYDIS_OPERAND_TYPE_IMMEDIATE) {
        // `mov eax, imm32`. The 32-bit move zero-extends rax.
        c.mov(eax, static_cast<s32>(src.imm.value.s));
        c.mov(qword[r13 + GprOffset(dst_idx)], rax);
        return true;
    }

    if (src.type == ZYDIS_OPERAND_TYPE_MEMORY) {
        if (!EmitEffectiveAddress(src.mem, next_rip, c)) return false;
        // 32-bit load zero-extends rax. Store full 64.
        c.mov(eax, dword[rdx]);
        c.mov(qword[r13 + GprOffset(dst_idx)], rax);
        return true;
    }

    return false;
}

/// MOVBE — move with byte-swap (big-endian load/store). Two operands, one
/// register and one memory; the direction is set by which operand is the
/// register:
///   movbe r, m   reg = byteswap(load[m])
///   movbe m, r   store[m] = byteswap(reg)
/// Widths 16/32/64 only (MOVBE has no 8-bit form). Emulated as a sized
/// load/store plus a byte reversal — BSWAP for 32/64, and ROR by 8 for the
/// 16-bit case (BSWAP is undefined on a 16-bit register). This needs no host
/// MOVBE support. No flags are affected. The effective address is computed
/// first (it uses rax/rcx as scratch), then the value is moved through rax.
bool EmitMovbe(const ZydisDecodedInstruction& insn,
               const ZydisDecodedOperand* ops,
               u64 next_rip,
               Xbyak::CodeGenerator& c) {
    const u32 w = insn.operand_width;
    if (w != 16 && w != 32 && w != 64) return false;

    const auto& a = ops[0];
    const auto& b = ops[1];

    // Load form: movbe reg, [mem].
    if (a.type == ZYDIS_OPERAND_TYPE_REGISTER &&
        b.type == ZYDIS_OPERAND_TYPE_MEMORY) {
        const int dst_idx = ZydisGprToIndex(a.reg.value);
        if (dst_idx < 0) return false;
        if (!EmitEffectiveAddress(b.mem, next_rip, c)) return false;  // rdx = EA
        if (w == 64) {
            c.mov(rax, qword[rdx]);
            c.bswap(rax);
            c.mov(qword[r13 + GprOffset(dst_idx)], rax);
        } else if (w == 32) {
            c.mov(eax, dword[rdx]);
            c.bswap(eax);                                 // 32-bit write zero-extends rax
            c.mov(qword[r13 + GprOffset(dst_idx)], rax);
        } else {  // w == 16
            c.mov(ax, word[rdx]);
            c.ror(ax, 8);                                 // swap the two bytes
            c.mov(word[r13 + GprOffset(dst_idx)], ax);    // preserve upper 48 bits
        }
        return true;
    }

    // Store form: movbe [mem], reg.
    if (a.type == ZYDIS_OPERAND_TYPE_MEMORY &&
        b.type == ZYDIS_OPERAND_TYPE_REGISTER) {
        const int src_idx = ZydisGprToIndex(b.reg.value);
        if (src_idx < 0) return false;
        if (!EmitEffectiveAddress(a.mem, next_rip, c)) return false;  // rdx = EA
        if (w == 64) {
            c.mov(rax, qword[r13 + GprOffset(src_idx)]);
            c.bswap(rax);
            c.mov(qword[rdx], rax);
        } else if (w == 32) {
            c.mov(eax, dword[r13 + GprOffset(src_idx)]);
            c.bswap(eax);
            c.mov(dword[rdx], eax);
        } else {  // w == 16
            c.mov(ax, word[r13 + GprOffset(src_idx)]);
            c.ror(ax, 8);
            c.mov(word[rdx], ax);
        }
        return true;
    }

    return false;  // reg,reg or any other shape is not a valid MOVBE
}

/// BSWAP r32/r64 — reverse the byte order of a register, in place. Defined
/// only for 32- and 64-bit operands (BSWAP on a 16-bit register is undefined
/// on x86, so we bail rather than guess). A 32-bit BSWAP zero-extends into the
/// full 64-bit register, like any 32-bit write. No flags affected.
bool EmitBswap(const ZydisDecodedInstruction& insn,
               const ZydisDecodedOperand* ops,
               u64 next_rip,
               Xbyak::CodeGenerator& c) {
    (void)next_rip;
    const u32 w = insn.operand_width;
    if (w != 32 && w != 64) return false;
    if (ops[0].type != ZYDIS_OPERAND_TYPE_REGISTER) return false;
    const int idx = ZydisGprToIndex(ops[0].reg.value);
    if (idx < 0) return false;
    if (w == 64) {
        c.mov(rax, qword[r13 + GprOffset(idx)]);
        c.bswap(rax);
        c.mov(qword[r13 + GprOffset(idx)], rax);
    } else {  // w == 32
        c.mov(eax, dword[r13 + GprOffset(idx)]);
        c.bswap(eax);                                 // 32-bit write zero-extends rax
        c.mov(qword[r13 + GprOffset(idx)], rax);
    }
    return true;
}

/// LEA r64, [mem] — compute effective address into a register.
///
/// No memory access. The effective-address computation already lives
/// in EmitEffectiveAddress (it puts the address into rdx); for LEA
/// we just store rdx to the destination GPR slot.
///
/// Common patterns:
///   lea rax, [rip+disp32]   — pointer to globals (REX.W + 8d /0)
///   lea rbp, [rsp+disp8]    — frame pointer via offset
///   lea rcx, [rax + rbx*8]  — array indexing into a register
/// All flow through EmitEffectiveAddress; we don't care which.
///
/// 32-bit LEA (`lea r32, [m]`) is not yet handled; its semantics
/// require truncating the 32-bit result and zero-extending to 64.
bool EmitLea(const ZydisDecodedInstruction& insn,
             const ZydisDecodedOperand* ops,
             u64 next_rip,
             Xbyak::CodeGenerator& c) {
    if (insn.operand_width != 64 && insn.operand_width != 32) return false;
    if (ops[0].type != ZYDIS_OPERAND_TYPE_REGISTER) return false;
    if (ops[1].type != ZYDIS_OPERAND_TYPE_MEMORY) return false;

    const int dst_idx = ZydisGprToIndex(ops[0].reg.value);
    if (dst_idx < 0) return false;

    if (!EmitEffectiveAddress(ops[1].mem, next_rip, c)) return false;
    // Address arithmetic is computed at full width regardless of the
    // operand size; the operand size only governs the destination write.
    // 32-bit LEA writes the low 32 bits and zero-extends bits 63:32.
    if (insn.operand_width == 64) {
        c.mov(qword[r13 + GprOffset(dst_idx)], rdx);
    } else {
        // 32-bit store via edx zero-extends rdx; write the full qword
        // slot so the upper 32 bits are explicitly cleared.
        c.mov(eax, edx);
        c.mov(qword[r13 + GprOffset(dst_idx)], rax);
    }
    return true;
}

/// MOVSXD r64, r/m32 — sign-extend a 32-bit value to 64 bits.
///
/// Encoding: REX.W + 63 + ModR/M (+ optional SIB/disp). Common in
/// PS4 code for promoting `int` indices to pointer-sized values:
///
///   movsxd rax, dword [rbx+rcx*4]   ; rax = sext(table[rcx])
///
/// Both register and memory sources are supported. xbyak provides
/// `movsxd` directly.
bool EmitMovsxd(const ZydisDecodedInstruction& insn,
                const ZydisDecodedOperand* ops,
                u64 next_rip,
                Xbyak::CodeGenerator& c) {
    if (insn.operand_width != 64) return false;
    if (ops[0].type != ZYDIS_OPERAND_TYPE_REGISTER) return false;

    const int dst_idx = ZydisGprToIndex(ops[0].reg.value);
    if (dst_idx < 0) return false;

    if (ops[1].type == ZYDIS_OPERAND_TYPE_REGISTER) {
        const int src_idx = ZydisGprToIndex(ops[1].reg.value);
        if (src_idx < 0) return false;
        // Sign-extend low 32 of guest src register to 64 bits.
        c.movsxd(rax, dword[r13 + GprOffset(src_idx)]);
        c.mov(qword[r13 + GprOffset(dst_idx)], rax);
        return true;
    }

    if (ops[1].type == ZYDIS_OPERAND_TYPE_MEMORY) {
        if (!EmitEffectiveAddress(ops[1].mem, next_rip, c)) return false;
        c.movsxd(rax, dword[rdx]);
        c.mov(qword[r13 + GprOffset(dst_idx)], rax);
        return true;
    }

    return false;
}

// ============================================================================
// Flag computation helpers (eager)
//
// We compute flag bits eagerly into state.rflags after every
// flag-affecting operation. This is the simpler choice over lazy
// flag evaluation; if profiling later shows flag computation is hot,
// switching to lazy is a localized change (the consumers all read
// `state.rflags`, so the source can be reformulated transparently).
//
// We compute five of the six "arithmetic" flags: ZF (zero), SF
// (sign), CF (carry/borrow), OF (signed overflow), PF (parity of
// low byte). AF (auxiliary carry, decimal-arithmetic helper) is
// deliberately skipped — modern code essentially never reads it,
// and the only consumer is the `JP/JPE/JPO` family for parity,
// which uses PF not AF.
//
// Input register convention for these helpers:
//   rcx = lhs (original destination value before the op)
//   rdx = rhs (source value)
//   rax = result (after the op)
//
// Scratch used:
//   r8, r9, r10 internally (clobbered)
//
// Output:
//   state.rflags has the five flag bits written, other bits
//   preserved (well — preserved relative to whatever was there
//   before, which is also from a previous flag-writing op).
// ============================================================================

namespace RflagsBits {
constexpr u64 CF = 1ULL << 0;
constexpr u64 PF = 1ULL << 2;
constexpr u64 ZF = 1ULL << 6;
constexpr u64 SF = 1ULL << 7;
constexpr u64 OF = 1ULL << 11;
constexpr u64 AF = 1ULL << 4;
constexpr u64 AllArith = CF | PF | ZF | SF | OF;
// Mask applied to guest rflags before any popfq into the HOST flags
// register. Only the six arithmetic flags may pass through: the guest can
// legitimately carry DF=1 (after STD) -- and, in principle, TF/AC/NT -- in
// state.rflags, and popfq'ing those into the host is catastrophic. A host
// DF=1 leaking past the block boundary breaks the SysV "DF clear on call"
// contract: glibc/MSVC memcpy (rep movsb) then runs BACKWARDS in the
// dispatcher and every HLE call after it. TF would single-step-trap host
// execution; AC would fault on any misaligned access. String ops never
// consult host DF (they read state.rflags bit 10 directly), so masking
// here loses nothing.
constexpr u64 HostLoadMask = CF | PF | AF | ZF | SF | OF;
}  // namespace RflagsBits

/// Store host-derived ARITHMETIC flags into guest rflags, preserving the
/// guest's non-arith bits. The dual of the HostLoadMask applied before every
/// popfq: a wholesale `mov [rflags], host_pushfq_value` writes host DF=0 over
/// a guest DF=1 (set by STD; the string ops read it from state.rflags) and
/// imports the host's IF/reserved bits into the guest image. Every emitter
/// that captures flags via pushfq MUST store through this merge.
///
/// host_flags is masked in place (callers treat it as consumed). The memory
/// AND uses an imm32 that sign-extends to the full ~HostLoadMask -- bit 31
/// of 0xFFFFF72A is set, so the hardware extension reproduces the intended
/// 64-bit mask exactly.
inline void EmitStoreArithFlags(Xbyak::CodeGenerator& c,
                                const Xbyak::Reg64& host_flags) {
    using namespace Xbyak::util;
    c.and_(host_flags, RflagsBits::HostLoadMask);
    c.and_(qword[r13 + Offsets::Rflags],
           static_cast<Xbyak::uint32>(~RflagsBits::HostLoadMask));
    c.or_(qword[r13 + Offsets::Rflags], host_flags);
}

/// Emit code that computes PF (parity of the low byte of rax) and
/// writes the bit into r8 with rflags-position alignment.
/// Uses r9 as additional scratch.
///
/// x86 already provides a way to compute parity: do an XOR or AND
/// of the value with itself (which sets PF), then `setp` to extract.
/// But we already have the result in rax — the original op that
/// produced rax also set host PF (since we used a host instruction
/// to produce it). However, host flags are not stable across the
/// xbyak-emitted sequence (every host insn could alter them). So
/// we recompute PF from scratch using AND.
void EmitWritePF(Xbyak::CodeGenerator& c) {
    // Set host PF from low byte of rax.
    c.test(al, al);
    // setp r8b sets the byte to 1 if PF, else 0.
    c.setp(r8b);
    c.movzx(r8, r8b);
    c.shl(r8, 2);                       // r8 = PF_bit << 2 (PF is at bit 2)
    // OR into rflags.
    c.or_(qword[r13 + Offsets::Rflags], r8);
}

/// Emit code that clears the five arithmetic flag bits in
/// state.rflags. Caller then ORs in each computed bit.
void EmitClearArithFlags(Xbyak::CodeGenerator& c) {
    c.mov(r9, ~RflagsBits::AllArith);
    c.and_(qword[r13 + Offsets::Rflags], r9);
}

/// Compute flags for a subtract (SUB, CMP).
/// Inputs:
///   rcx = lhs, rdx = rhs, rax = lhs - rhs
/// `width` is the operand width in bits (8/16/32/64). ZF/SF/CF/OF are
/// computed against that width: a 32-bit op's result is zero-extended in
/// rax, so SF/OF must use bit (width-1), not bit 63, and unsigned CF must
/// compare only the low `width` bits.
/// Clobbers r8, r9, r10. Writes rflags.
void EmitFlagsFromSubtract(Xbyak::CodeGenerator& c, u32 width = 64) {
    EmitClearArithFlags(c);
    const int sign_bit = static_cast<int>(width) - 1;   // 7/15/31/63
    // Mask of the low `width` bits (for width<64; 64 needs no mask).
    const u64 wmask = (width >= 64) ? ~0ULL : ((1ULL << width) - 1);

    // ZF: low `width` bits of result == 0.
    c.mov(r8, rax);
    if (width < 64) { c.mov(r9, wmask); c.and_(r8, r9); }
    c.test(r8, r8);
    c.setz(r8b);
    c.movzx(r8, r8b);
    c.shl(r8, 6);                       // ZF at bit 6
    c.or_(qword[r13 + Offsets::Rflags], r8);

    // SF: bit (width-1) of result.
    c.mov(r8, rax);
    c.shr(r8, sign_bit);
    c.and_(r8, 1);
    c.shl(r8, 7);                       // SF at bit 7
    c.or_(qword[r13 + Offsets::Rflags], r8);

    // CF for subtract: lhs < rhs (unsigned), compared at operand width.
    if (width < 64) {
        c.mov(r8, rcx); c.mov(r9, wmask); c.and_(r8, r9);   // r8 = lhs & wmask
        c.mov(r10, rdx); c.and_(r10, r9);                   // r10 = rhs & wmask
        c.cmp(r8, r10);
    } else {
        c.cmp(rcx, rdx);
    }
    c.setb(r8b);                        // setb = "set if below", unsigned <
    c.movzx(r8, r8b);
    c.or_(qword[r13 + Offsets::Rflags], r8);

    // OF for subtract: ((lhs ^ rhs) & (lhs ^ result)) at bit (width-1).
    c.mov(r8, rcx);
    c.xor_(r8, rdx);                    // r8 = lhs ^ rhs
    c.mov(r9, rcx);
    c.xor_(r9, rax);                    // r9 = lhs ^ result
    c.and_(r8, r9);
    c.shr(r8, sign_bit);
    c.and_(r8, 1);
    c.shl(r8, 11);                      // OF at bit 11
    c.or_(qword[r13 + Offsets::Rflags], r8);

    // PF: parity of low byte of result.
    EmitWritePF(c);
}

///   rcx = lhs, rdx = rhs, rax = lhs + rhs
/// `width` is the operand width in bits; see EmitFlagsFromSubtract.
/// Clobbers r8, r9, r10. Writes rflags.
void EmitFlagsFromAdd(Xbyak::CodeGenerator& c, u32 width = 64) {
    EmitClearArithFlags(c);
    const int sign_bit = static_cast<int>(width) - 1;
    const u64 wmask = (width >= 64) ? ~0ULL : ((1ULL << width) - 1);

    // ZF: low `width` bits of result == 0.
    c.mov(r8, rax);
    if (width < 64) { c.mov(r9, wmask); c.and_(r8, r9); }
    c.test(r8, r8);
    c.setz(r8b);
    c.movzx(r8, r8b);
    c.shl(r8, 6);
    c.or_(qword[r13 + Offsets::Rflags], r8);

    // SF: bit (width-1) of result.
    c.mov(r8, rax);
    c.shr(r8, sign_bit);
    c.and_(r8, 1);
    c.shl(r8, 7);
    c.or_(qword[r13 + Offsets::Rflags], r8);

    // CF for add: low-`width` result < low-`width` lhs (unsigned wrap).
    if (width < 64) {
        c.mov(r8, rax); c.mov(r9, wmask); c.and_(r8, r9);   // r8 = res & wmask
        c.mov(r10, rcx); c.and_(r10, r9);                   // r10 = lhs & wmask
        c.cmp(r8, r10);
    } else {
        c.cmp(rax, rcx);
    }
    c.setb(r8b);
    c.movzx(r8, r8b);
    c.or_(qword[r13 + Offsets::Rflags], r8);

    // OF for add: (~(lhs ^ rhs) & (lhs ^ result)) at bit (width-1).
    c.mov(r8, rcx);
    c.xor_(r8, rdx);                    // r8 = lhs ^ rhs
    c.not_(r8);                         // r8 = ~(lhs ^ rhs)
    c.mov(r9, rcx);
    c.xor_(r9, rax);                    // r9 = lhs ^ result
    c.and_(r8, r9);
    c.shr(r8, sign_bit);
    c.and_(r8, 1);
    c.shl(r8, 11);
    c.or_(qword[r13 + Offsets::Rflags], r8);

    // PF: parity of low byte of result.
    EmitWritePF(c);
}

/// Compute flags for a bitwise op (AND, TEST, XOR, OR).
/// Inputs: rax = result. (lhs and rhs unused — CF and OF are
/// always 0 for bitwise ops per x86 spec.)
/// `width` is the operand width in bits; SF/ZF are computed against it.
/// Clobbers r8, r9. Writes rflags.
void EmitFlagsFromBitwise(Xbyak::CodeGenerator& c, u32 width = 64) {
    EmitClearArithFlags(c);
    const int sign_bit = static_cast<int>(width) - 1;
    const u64 wmask = (width >= 64) ? ~0ULL : ((1ULL << width) - 1);

    // ZF: low `width` bits of result == 0.
    c.mov(r8, rax);
    if (width < 64) { c.mov(r9, wmask); c.and_(r8, r9); }
    c.test(r8, r8);
    c.setz(r8b);
    c.movzx(r8, r8b);
    c.shl(r8, 6);
    c.or_(qword[r13 + Offsets::Rflags], r8);

    // SF: bit (width-1) of result.
    c.mov(r8, rax);
    c.shr(r8, sign_bit);
    c.and_(r8, 1);
    c.shl(r8, 7);
    c.or_(qword[r13 + Offsets::Rflags], r8);

    // CF, OF: always 0 for bitwise ops. Already cleared by
    // EmitClearArithFlags above.

    // PF: parity of low byte of result.
    EmitWritePF(c);
}

/// ADD r64, r64 / ADD r64, imm32-sx / ADD qword[mem], r64.
/// Writes ZF/SF/CF/OF/PF to state.rflags (eager flag computation).
/// `next_rip` is needed by the mem-dst form's address calculation
/// (RIP-relative case); reg-dst forms ignore it.
bool EmitAdd(const ZydisDecodedInstruction& insn,
             const ZydisDecodedOperand* ops,
             u64 next_rip,
             Xbyak::CodeGenerator& c) {
    const auto& dst = ops[0];
    const auto& src = ops[1];

    // 64-bit ADD with memory destination: `add qword[mem], r64`.
    // Mirrors EmitOr's mem-dst pattern. We stash the computed address
    // into r10 before reusing rdx as the rhs operand for the flag
    // helper. EmitFlagsFromAdd clobbers r8/r9 only, so r10 survives
    // until after the writeback (and beyond — we don't actually need
    // r10 after the store).
    if (insn.operand_width == 64 &&
        dst.type == ZYDIS_OPERAND_TYPE_MEMORY &&
        (src.type == ZYDIS_OPERAND_TYPE_REGISTER ||
         src.type == ZYDIS_OPERAND_TYPE_IMMEDIATE)) {
        if (!EmitEffectiveAddress(dst.mem, next_rip, c)) return false;
        // rdx = address; preserve it in r10 so the flag helper can
        // use rdx for the rhs.
        c.mov(r10, rdx);
        c.mov(rcx, qword[r10]);                       // rcx = lhs (orig [mem])
        if (src.type == ZYDIS_OPERAND_TYPE_REGISTER) {
            const int src_idx = ZydisGprToIndex(src.reg.value);
            if (src_idx < 0) return false;
            c.mov(rdx, qword[r13 + GprOffset(src_idx)]);  // rdx = rhs
        } else {
            c.mov(rdx, src.imm.value.s);              // sign-extended imm
        }
        c.mov(rax, rcx);
        c.add(rax, rdx);
        c.mov(qword[r10], rax);                       // store result
        EmitFlagsFromAdd(c);
        return true;
    }

    if (dst.type != ZYDIS_OPERAND_TYPE_REGISTER) return false;
    const int dst_idx = ZydisGprToIndex(dst.reg.value);
    if (dst_idx < 0) return false;
    if (insn.operand_width != 64) return false;

    // 64-bit ADD with memory SOURCE: `add r64, qword[mem]`. Symmetric
    // to the mem-dst case above but the register is the destination.
    // Appears in the game's entry path. Flag helper wants lhs in rcx,
    // rhs in rdx, result in rax.
    if (src.type == ZYDIS_OPERAND_TYPE_MEMORY) {
        if (!EmitEffectiveAddress(src.mem, next_rip, c)) return false;
        c.mov(rdx, qword[rdx]);                       // rdx = rhs ([mem])
        c.mov(rcx, qword[r13 + GprOffset(dst_idx)]);  // rcx = lhs (dst reg)
        c.mov(rax, rcx);
        c.add(rax, rdx);
        c.mov(qword[r13 + GprOffset(dst_idx)], rax);
        EmitFlagsFromAdd(c);
        return true;
    }

    if (src.type == ZYDIS_OPERAND_TYPE_REGISTER) {
        const int src_idx = ZydisGprToIndex(src.reg.value);
        if (src_idx < 0) return false;

        // Load inputs: rcx = lhs, rdx = rhs, then rax = sum.
        // (Flag helpers want lhs in rcx, rhs in rdx, result in rax.)
        c.mov(rcx, qword[r13 + GprOffset(dst_idx)]);
        c.mov(rdx, qword[r13 + GprOffset(src_idx)]);
        c.mov(rax, rcx);
        c.add(rax, rdx);
        c.mov(qword[r13 + GprOffset(dst_idx)], rax);

        EmitFlagsFromAdd(c);
        return true;
    }

    if (src.type == ZYDIS_OPERAND_TYPE_IMMEDIATE) {
        c.mov(rcx, qword[r13 + GprOffset(dst_idx)]);
        const auto imm = src.imm.value.s;
        // Materialize the immediate into rdx so the flag helper
        // sees the same rhs the operation used.
        c.mov(rdx, imm);
        c.mov(rax, rcx);
        c.add(rax, rdx);
        c.mov(qword[r13 + GprOffset(dst_idx)], rax);

        EmitFlagsFromAdd(c);
        return true;
    }

    return false;
}

/// SUB r64, imm32-sx — for stack adjustment in function prologue.
/// Also SUB r64, r64, and SUB qword[mem], r64. Writes ZF/SF/CF/OF/PF.
/// `next_rip` is required by the mem-dst form's address calculation
/// (RIP-relative case); reg-dst forms ignore it.
bool EmitSub(const ZydisDecodedInstruction& insn,
             const ZydisDecodedOperand* ops,
             u64 next_rip,
             Xbyak::CodeGenerator& c) {
    if (insn.operand_width != 64) return false;
    const auto& dst = ops[0];
    const auto& src = ops[1];

    // 64-bit SUB with memory destination: `sub qword[mem], r64` or
    // `sub qword[mem], imm`. Mirrors EmitAdd's mem-dst pattern; stash
    // address in r10 to free rdx for the flag helper's rhs slot.
    if (dst.type == ZYDIS_OPERAND_TYPE_MEMORY &&
        (src.type == ZYDIS_OPERAND_TYPE_REGISTER ||
         src.type == ZYDIS_OPERAND_TYPE_IMMEDIATE)) {
        if (!EmitEffectiveAddress(dst.mem, next_rip, c)) return false;
        c.mov(r10, rdx);                              // r10 = addr
        c.mov(rcx, qword[r10]);                       // rcx = lhs ([mem])
        if (src.type == ZYDIS_OPERAND_TYPE_REGISTER) {
            const int src_idx = ZydisGprToIndex(src.reg.value);
            if (src_idx < 0) return false;
            c.mov(rdx, qword[r13 + GprOffset(src_idx)]);  // rdx = rhs
        } else {
            c.mov(rdx, src.imm.value.s);              // sign-extended imm
        }
        c.mov(rax, rcx);
        c.sub(rax, rdx);
        c.mov(qword[r10], rax);                       // store result
        EmitFlagsFromSubtract(c);
        return true;
    }

    if (dst.type != ZYDIS_OPERAND_TYPE_REGISTER) return false;
    const int dst_idx = ZydisGprToIndex(dst.reg.value);
    if (dst_idx < 0) return false;

    // 64-bit SUB with memory SOURCE: `sub r64, qword[mem]`. Symmetric
    // to the mem-dst case but the register is the destination. Flag
    // helper wants lhs in rcx, rhs in rdx, result in rax.
    if (src.type == ZYDIS_OPERAND_TYPE_MEMORY) {
        if (!EmitEffectiveAddress(src.mem, next_rip, c)) return false;
        c.mov(rdx, qword[rdx]);                       // rdx = rhs ([mem])
        c.mov(rcx, qword[r13 + GprOffset(dst_idx)]);  // rcx = lhs (dst reg)
        c.mov(rax, rcx);
        c.sub(rax, rdx);
        c.mov(qword[r13 + GprOffset(dst_idx)], rax);
        EmitFlagsFromSubtract(c);
        return true;
    }

    if (src.type == ZYDIS_OPERAND_TYPE_REGISTER) {
        const int src_idx = ZydisGprToIndex(src.reg.value);
        if (src_idx < 0) return false;
        c.mov(rcx, qword[r13 + GprOffset(dst_idx)]);
        c.mov(rdx, qword[r13 + GprOffset(src_idx)]);
        c.mov(rax, rcx);
        c.sub(rax, rdx);
        c.mov(qword[r13 + GprOffset(dst_idx)], rax);

        EmitFlagsFromSubtract(c);
        return true;
    }

    if (src.type == ZYDIS_OPERAND_TYPE_IMMEDIATE) {
        c.mov(rcx, qword[r13 + GprOffset(dst_idx)]);
        const auto imm = src.imm.value.s;
        c.mov(rdx, imm);
        c.mov(rax, rcx);
        c.sub(rax, rdx);
        c.mov(qword[r13 + GprOffset(dst_idx)], rax);

        EmitFlagsFromSubtract(c);
        return true;
    }

    return false;
}

/// CMP r64, r64 / CMP r64, imm32-sx.
/// Like SUB but doesn't write the result back — only writes flags.
/// CMP — sets flags by computing lhs - rhs without storing the result.
///
/// Supported forms:
///   cmp r64, r64
///   cmp r64, imm32 (sign-extended)
///   cmp r64, [m]                      ← new
///   cmp [m], r64                      ← new
///   cmp [m], imm32                    ← new
///
/// Memory operands flow through EmitEffectiveAddress which puts the
/// computed address into rdx. We then load the 8 bytes at [rdx] and
/// shuffle into rcx/rdx as needed for the subtract step.
bool EmitCmp(const ZydisDecodedInstruction& insn,
             const ZydisDecodedOperand* ops,
             u64 next_rip,
             Xbyak::CodeGenerator& c) {
    if (insn.operand_width != 64) return false;
    const auto& lhs_op = ops[0];
    const auto& rhs_op = ops[1];

    // ----- lhs is a register -----
    if (lhs_op.type == ZYDIS_OPERAND_TYPE_REGISTER) {
        const int lhs_idx = ZydisGprToIndex(lhs_op.reg.value);
        if (lhs_idx < 0) return false;

        if (rhs_op.type == ZYDIS_OPERAND_TYPE_REGISTER) {
            const int rhs_idx = ZydisGprToIndex(rhs_op.reg.value);
            if (rhs_idx < 0) return false;
            c.mov(rcx, qword[r13 + GprOffset(lhs_idx)]);
            c.mov(rdx, qword[r13 + GprOffset(rhs_idx)]);
            c.mov(rax, rcx);
            c.sub(rax, rdx);
            EmitFlagsFromSubtract(c);
            return true;
        }

        if (rhs_op.type == ZYDIS_OPERAND_TYPE_IMMEDIATE) {
            c.mov(rcx, qword[r13 + GprOffset(lhs_idx)]);
            c.mov(rdx, rhs_op.imm.value.s);
            c.mov(rax, rcx);
            c.sub(rax, rdx);
            EmitFlagsFromSubtract(c);
            return true;
        }

        if (rhs_op.type == ZYDIS_OPERAND_TYPE_MEMORY) {
            // EmitEffectiveAddress → rdx = addr (clobbers rax).
            // Then rdx = *rdx (the value). Then load lhs into rcx.
            if (!EmitEffectiveAddress(rhs_op.mem, next_rip, c)) return false;
            c.mov(rdx, qword[rdx]);
            c.mov(rcx, qword[r13 + GprOffset(lhs_idx)]);
            c.mov(rax, rcx);
            c.sub(rax, rdx);
            EmitFlagsFromSubtract(c);
            return true;
        }
        return false;
    }

    // ----- lhs is memory -----
    if (lhs_op.type == ZYDIS_OPERAND_TYPE_MEMORY) {
        // Compute address into rdx, then load the value into rcx (it
        // becomes the lhs of the subtract).
        if (!EmitEffectiveAddress(lhs_op.mem, next_rip, c)) return false;
        c.mov(rcx, qword[rdx]);

        if (rhs_op.type == ZYDIS_OPERAND_TYPE_REGISTER) {
            const int rhs_idx = ZydisGprToIndex(rhs_op.reg.value);
            if (rhs_idx < 0) return false;
            c.mov(rdx, qword[r13 + GprOffset(rhs_idx)]);
            c.mov(rax, rcx);
            c.sub(rax, rdx);
            EmitFlagsFromSubtract(c);
            return true;
        }

        if (rhs_op.type == ZYDIS_OPERAND_TYPE_IMMEDIATE) {
            c.mov(rdx, rhs_op.imm.value.s);
            c.mov(rax, rcx);
            c.sub(rax, rdx);
            EmitFlagsFromSubtract(c);
            return true;
        }
        return false;
    }

    return false;
}

/// TEST r64, r64 / TEST r64, imm32-sx.
/// Like AND but doesn't write the result back — only writes flags.
/// CF and OF are always 0 (per x86 spec).
bool EmitTest(const ZydisDecodedInstruction& insn,
              const ZydisDecodedOperand* ops,
              u64 next_rip,
              Xbyak::CodeGenerator& c) {
    if (insn.operand_width != 64) return false;
    const auto& lhs_op = ops[0];
    const auto& rhs_op = ops[1];

    // Memory lhs: `test qword[mem], reg`. TEST computes lhs & rhs purely
    // for flags (no writeback). Load [mem] into rax, the reg into rdx,
    // AND, set flags, discard the result.
    if (lhs_op.type == ZYDIS_OPERAND_TYPE_MEMORY) {
        if (rhs_op.type != ZYDIS_OPERAND_TYPE_REGISTER) return false;
        const int rhs_idx = ZydisGprToIndex(rhs_op.reg.value);
        if (rhs_idx < 0) return false;
        if (!EmitEffectiveAddress(lhs_op.mem, next_rip, c)) return false;
        c.mov(rcx, qword[rdx]);                      // rcx = [mem]
        c.mov(rax, qword[r13 + GprOffset(rhs_idx)]);
        c.and_(rax, rcx);
        EmitFlagsFromBitwise(c);
        return true;
    }

    if (lhs_op.type != ZYDIS_OPERAND_TYPE_REGISTER) return false;
    const int lhs_idx = ZydisGprToIndex(lhs_op.reg.value);
    if (lhs_idx < 0) return false;

    if (rhs_op.type == ZYDIS_OPERAND_TYPE_REGISTER) {
        const int rhs_idx = ZydisGprToIndex(rhs_op.reg.value);
        if (rhs_idx < 0) return false;
        c.mov(rax, qword[r13 + GprOffset(lhs_idx)]);
        c.mov(rdx, qword[r13 + GprOffset(rhs_idx)]);
        c.and_(rax, rdx);
        EmitFlagsFromBitwise(c);
        return true;
    }

    if (rhs_op.type == ZYDIS_OPERAND_TYPE_IMMEDIATE) {
        c.mov(rax, qword[r13 + GprOffset(lhs_idx)]);
        c.mov(rdx, rhs_op.imm.value.s);
        c.and_(rax, rdx);
        EmitFlagsFromBitwise(c);
        return true;
    }

    return false;
}

/// XOR r64, r64 — common register-zero idiom (`xor rax, rax`).
/// Writes ZF/SF/PF (CF, OF = 0 per x86 spec for bitwise ops).
/// We don't yet implement XOR r64, imm.
/// XOR — bitwise exclusive or, writes flags (ZF/SF/PF; CF=OF=0).
///
/// Supported forms:
///   xor r64, r64
///   xor r32, r32           ← `xor eax, eax` zero-register idiom
///
/// 32-bit XOR zero-extends the destination's upper 32 (the x86-64
/// rule for any 32-bit operation writing a register). We get this
/// for free by using a host 32-bit op into rax then storing as 64.
bool EmitXor(const ZydisDecodedInstruction& insn,
             const ZydisDecodedOperand* ops,
             u64 next_rip,
             Xbyak::CodeGenerator& c) {
    const auto& dst = ops[0];
    const auto& src = ops[1];

    if (dst.type != ZYDIS_OPERAND_TYPE_REGISTER) return false;
    const int dst_idx = ZydisGprToIndex(dst.reg.value);
    if (dst_idx < 0) return false;

    // Register-source forms.
    if (src.type == ZYDIS_OPERAND_TYPE_REGISTER) {
        const int src_idx = ZydisGprToIndex(src.reg.value);
        if (src_idx < 0) return false;

        if (insn.operand_width == 64) {
            c.mov(rax, qword[r13 + GprOffset(dst_idx)]);
            c.mov(rcx, qword[r13 + GprOffset(src_idx)]);
            c.xor_(rax, rcx);
            c.mov(qword[r13 + GprOffset(dst_idx)], rax);
            EmitFlagsFromBitwise(c);
            return true;
        }

        if (insn.operand_width == 32) {
            c.mov(eax, dword[r13 + GprOffset(dst_idx)]);
            c.mov(ecx, dword[r13 + GprOffset(src_idx)]);
            c.xor_(eax, ecx);
            // Storing rax as qword writes the zero-extended 64-bit value.
            c.mov(qword[r13 + GprOffset(dst_idx)], rax);
            // Flag helpers read rax; the 32-bit XOR already zeroed the
            // upper half, so ZF/SF/PF are computed against the full 64-bit
            // value, which matches the 32-bit-result semantics.
            EmitFlagsFromBitwise(c);
            return true;
        }

        return false;
    }

    // Immediate-source forms: `xor r64, imm32-sx` and `xor r32, imm`.
    // The architecture only encodes imm8/imm32 (sign-extended for the
    // 64-bit form); we materialize the value into rdx/edx to avoid
    // xbyak immediate-size foot-guns, matching EmitAnd's imm pattern.
    if (src.type == ZYDIS_OPERAND_TYPE_IMMEDIATE) {
        if (insn.operand_width == 64) {
            const auto imm = src.imm.value.s;
            c.mov(rax, qword[r13 + GprOffset(dst_idx)]);
            c.mov(rdx, imm);
            c.xor_(rax, rdx);
            c.mov(qword[r13 + GprOffset(dst_idx)], rax);
            EmitFlagsFromBitwise(c);
            return true;
        }
        if (insn.operand_width == 32) {
            const u32 imm = static_cast<u32>(src.imm.value.u & 0xFFFFFFFFu);
            c.mov(eax, dword[r13 + GprOffset(dst_idx)]);
            c.xor_(eax, imm);                          // zero-extends rax
            c.mov(qword[r13 + GprOffset(dst_idx)], rax);
            EmitFlagsFromBitwise(c);
            return true;
        }
        return false;
    }

    // Memory-source forms: `xor r64, qword[mem]` / `xor r32, dword[mem]`.
    if (src.type == ZYDIS_OPERAND_TYPE_MEMORY) {
        if (insn.operand_width == 64) {
            if (!EmitEffectiveAddress(src.mem, next_rip, c)) return false;
            c.mov(rcx, qword[rdx]);                    // rcx = [mem]
            c.mov(rax, qword[r13 + GprOffset(dst_idx)]);
            c.xor_(rax, rcx);
            c.mov(qword[r13 + GprOffset(dst_idx)], rax);
            EmitFlagsFromBitwise(c);
            return true;
        }
        if (insn.operand_width == 32) {
            if (!EmitEffectiveAddress(src.mem, next_rip, c)) return false;
            c.mov(ecx, dword[rdx]);
            c.mov(eax, dword[r13 + GprOffset(dst_idx)]);
            c.xor_(eax, ecx);
            c.mov(qword[r13 + GprOffset(dst_idx)], rax);   // zero-extend
            EmitFlagsFromBitwise(c);
            return true;
        }
        return false;
    }

    return false;
}

// =============================================================================
// AND, OR — bitwise binary ops, mirror EmitXor but produce different
// results. Both clear CF/OF and set ZF/SF/PF based on the result.
// =============================================================================

/// Generic helper for 64-bit reg-reg bitwise dispatch. The actual
/// host op (`and`/`or`/`xor`) is supplied as a lambda so all three
/// instructions share the same load/store/flag scaffolding.
template <typename HostOp>
bool EmitBitwise64RegReg(const ZydisDecodedOperand& dst,
                         const ZydisDecodedOperand& src,
                         Xbyak::CodeGenerator& c,
                         HostOp host_op) {
    if (dst.type != ZYDIS_OPERAND_TYPE_REGISTER) return false;
    if (src.type != ZYDIS_OPERAND_TYPE_REGISTER) return false;
    const int dst_idx = ZydisGprToIndex(dst.reg.value);
    const int src_idx = ZydisGprToIndex(src.reg.value);
    if (dst_idx < 0 || src_idx < 0) return false;

    c.mov(rax, qword[r13 + GprOffset(dst_idx)]);
    c.mov(rcx, qword[r13 + GprOffset(src_idx)]);
    host_op(rax, rcx);
    c.mov(qword[r13 + GprOffset(dst_idx)], rax);
    EmitFlagsFromBitwise(c);
    return true;
}

/// Like the 64-bit version but for 32-bit width. Writing to a 32-bit
/// destination implicitly zeros the upper 32 of the guest GPR.
template <typename HostOp>
bool EmitBitwise32RegReg(const ZydisDecodedOperand& dst,
                         const ZydisDecodedOperand& src,
                         Xbyak::CodeGenerator& c,
                         HostOp host_op) {
    if (dst.type != ZYDIS_OPERAND_TYPE_REGISTER) return false;
    if (src.type != ZYDIS_OPERAND_TYPE_REGISTER) return false;
    const int dst_idx = ZydisGprToIndex(dst.reg.value);
    const int src_idx = ZydisGprToIndex(src.reg.value);
    if (dst_idx < 0 || src_idx < 0) return false;

    c.mov(eax, dword[r13 + GprOffset(dst_idx)]);
    c.mov(ecx, dword[r13 + GprOffset(src_idx)]);
    host_op(eax, ecx);                                  // 32-bit op zero-extends rax
    c.mov(qword[r13 + GprOffset(dst_idx)], rax);
    EmitFlagsFromBitwise(c);
    return true;
}

/// AND — bitwise and. Supported: r64,r64; r32,r32; r32,imm; r64,imm;
/// and r64,[mem] (reg destination, memory source).
bool EmitAnd(const ZydisDecodedInstruction& insn,
             const ZydisDecodedOperand* ops,
             u64 next_rip,
             Xbyak::CodeGenerator& c) {
    // 64-bit memory destination: `and qword[mem], reg`. Load [mem] into
    // rax, AND with the source reg, write back, set flags. (mem-dst +
    // immediate is deferred until a test needs it.)
    if (insn.operand_width == 64 &&
        ops[0].type == ZYDIS_OPERAND_TYPE_MEMORY &&
        ops[1].type == ZYDIS_OPERAND_TYPE_REGISTER) {
        const int src_idx = ZydisGprToIndex(ops[1].reg.value);
        if (src_idx < 0) return false;
        if (!EmitEffectiveAddress(ops[0].mem, next_rip, c)) return false;
        c.mov(rcx, qword[r13 + GprOffset(src_idx)]);
        c.mov(rax, qword[rdx]);
        c.and_(rax, rcx);
        c.mov(qword[rdx], rax);
        EmitFlagsFromBitwise(c);
        return true;
    }

    // 64-bit memory destination with immediate source: `and qword[mem], imm`.
    // The architecture encodes imm8 (sign-extended) or imm32 (sign-extended);
    // Zydis hands us the sign-extended s64, which we materialize into a
    // register to avoid xbyak immediate-size foot-guns. Stash the address in
    // r10 across the value setup (EmitEffectiveAddress returns it in rdx and
    // clobbers rax/rdx during computation).
    if (insn.operand_width == 64 &&
        ops[0].type == ZYDIS_OPERAND_TYPE_MEMORY &&
        ops[1].type == ZYDIS_OPERAND_TYPE_IMMEDIATE) {
        if (!EmitEffectiveAddress(ops[0].mem, next_rip, c)) return false;
        c.mov(r10, rdx);                          // r10 = &dst
        const auto imm = ops[1].imm.value.s;
        c.mov(rcx, imm);                          // rcx = sign-extended imm
        c.mov(rax, qword[r10]);                   // rax = [mem]
        c.and_(rax, rcx);
        c.mov(qword[r10], rax);                   // store result
        EmitFlagsFromBitwise(c);
        return true;
    }

    // 32-bit register destination with immediate source: very common
    // masking idiom (`and eax, 0xFF`, `and ecx, 0x3F`, etc.). Mirrors
    // the 32-bit reg-reg path's flag-handling: EmitFlagsFromBitwise
    // produces the same lazy flag computation, so behavior matches
    // the existing 32-bit AND.
    if (insn.operand_width == 32 &&
        ops[0].type == ZYDIS_OPERAND_TYPE_REGISTER &&
        ops[1].type == ZYDIS_OPERAND_TYPE_IMMEDIATE) {
        const int dst_idx = ZydisGprToIndex(ops[0].reg.value);
        if (dst_idx < 0) return false;
        // Zydis sign-extends imm8 forms into a u64 for us; truncate to
        // u32 — the bit pattern is preserved for the masking case.
        const u32 imm = static_cast<u32>(ops[1].imm.value.u & 0xFFFFFFFFu);
        c.mov(eax, dword[r13 + GprOffset(dst_idx)]);
        c.and_(eax, imm);                          // 32-bit op zero-extends rax
        c.mov(qword[r13 + GprOffset(dst_idx)], rax);
        EmitFlagsFromBitwise(c);
        return true;
    }

    // 64-bit register destination with immediate source: `and r64, imm`.
    // Zydis presents the (sign-extended) value as s64; the architecture
    // only allows imm8 and imm32-sx forms, so the value always fits
    // in a signed 32-bit window — but we materialize the full s64 into
    // rdx anyway to match EmitAdd's 64-bit imm pattern and avoid xbyak
    // encoding-size foot-guns.
    if (insn.operand_width == 64 &&
        ops[0].type == ZYDIS_OPERAND_TYPE_REGISTER &&
        ops[1].type == ZYDIS_OPERAND_TYPE_IMMEDIATE) {
        const int dst_idx = ZydisGprToIndex(ops[0].reg.value);
        if (dst_idx < 0) return false;
        const auto imm = ops[1].imm.value.s;
        c.mov(rax, qword[r13 + GprOffset(dst_idx)]);
        c.mov(rdx, imm);
        c.and_(rax, rdx);
        c.mov(qword[r13 + GprOffset(dst_idx)], rax);
        EmitFlagsFromBitwise(c);
        return true;
    }

    // 64-bit AND with memory SOURCE: `and r64, qword[mem]`. Mirror of
    // EmitOr's mem-dst path, but here the destination is the register
    // and the memory operand is the source. Common in the game's entry
    // path (masking a register against a value loaded from a table).
    if (insn.operand_width == 64 &&
        ops[0].type == ZYDIS_OPERAND_TYPE_REGISTER &&
        ops[1].type == ZYDIS_OPERAND_TYPE_MEMORY) {
        const int dst_idx = ZydisGprToIndex(ops[0].reg.value);
        if (dst_idx < 0) return false;
        if (!EmitEffectiveAddress(ops[1].mem, next_rip, c)) return false;
        // rdx = src address. Load lhs (dst reg) into rax, AND with
        // [rdx], store back to the guest dst slot, flags from result.
        c.mov(rcx, qword[rdx]);
        c.mov(rax, qword[r13 + GprOffset(dst_idx)]);
        c.and_(rax, rcx);
        c.mov(qword[r13 + GprOffset(dst_idx)], rax);
        EmitFlagsFromBitwise(c);
        return true;
    }

    if (insn.operand_width == 64) {
        return EmitBitwise64RegReg(ops[0], ops[1], c,
            [&](Xbyak::Reg64 a, Xbyak::Reg64 b) { c.and_(a, b); });
    }
    if (insn.operand_width == 32) {
        return EmitBitwise32RegReg(ops[0], ops[1], c,
            [&](Xbyak::Reg32 a, Xbyak::Reg32 b) { c.and_(a, b); });
    }
    return false;
}

/// OR — bitwise or. Same forms as AND, plus `or qword[mem], r64`
/// for the lock-free-bit-set idiom the game uses on shared state.
/// `next_rip` is needed for the mem-dst form's address calculation
/// (RIP-relative case); other forms ignore it.
bool EmitOr(const ZydisDecodedInstruction& insn,
            const ZydisDecodedOperand* ops,
            u64 next_rip,
            Xbyak::CodeGenerator& c) {
    // 64-bit OR with memory destination: `or qword[mem], r64`.
    // Strict mem-dst + reg-src for now; mem-dst + imm and 32-bit
    // memory forms are deferred to keep the diff focused.
    if (insn.operand_width == 64 &&
        ops[0].type == ZYDIS_OPERAND_TYPE_MEMORY &&
        ops[1].type == ZYDIS_OPERAND_TYPE_REGISTER) {
        const int src_idx = ZydisGprToIndex(ops[1].reg.value);
        if (src_idx < 0) return false;
        if (!EmitEffectiveAddress(ops[0].mem, next_rip, c)) return false;
        // rdx = address. Load src into rcx, lhs from [rdx] into rax,
        // OR, store back, flags from result. The flag helper uses
        // rax as the result input and r8/r9 as transients; rdx is
        // free to repurpose after the store but we don't need it.
        c.mov(rcx, qword[r13 + GprOffset(src_idx)]);
        c.mov(rax, qword[rdx]);
        c.or_(rax, rcx);
        c.mov(qword[rdx], rax);
        EmitFlagsFromBitwise(c);
        return true;
    }

    if (insn.operand_width == 64) {
        // Immediate source: `or r64, imm32-sx`.
        if (ops[0].type == ZYDIS_OPERAND_TYPE_REGISTER &&
            ops[1].type == ZYDIS_OPERAND_TYPE_IMMEDIATE) {
            const int dst_idx = ZydisGprToIndex(ops[0].reg.value);
            if (dst_idx < 0) return false;
            c.mov(rax, qword[r13 + GprOffset(dst_idx)]);
            c.mov(rcx, ops[1].imm.value.s);   // sign-extended imm in a reg
            c.or_(rax, rcx);
            c.mov(qword[r13 + GprOffset(dst_idx)], rax);
            EmitFlagsFromBitwise(c);
            return true;
        }
        // Memory source: `or r64, qword[mem]`.
        if (ops[0].type == ZYDIS_OPERAND_TYPE_REGISTER &&
            ops[1].type == ZYDIS_OPERAND_TYPE_MEMORY) {
            const int dst_idx = ZydisGprToIndex(ops[0].reg.value);
            if (dst_idx < 0) return false;
            if (!EmitEffectiveAddress(ops[1].mem, next_rip, c)) return false;
            c.mov(rcx, qword[rdx]);                     // rcx = [mem]
            c.mov(rax, qword[r13 + GprOffset(dst_idx)]);
            c.or_(rax, rcx);
            c.mov(qword[r13 + GprOffset(dst_idx)], rax);
            EmitFlagsFromBitwise(c);
            return true;
        }
        return EmitBitwise64RegReg(ops[0], ops[1], c,
            [&](Xbyak::Reg64 a, Xbyak::Reg64 b) { c.or_(a, b); });
    }
    if (insn.operand_width == 32) {
        // Immediate source: `or r32, imm`.
        if (ops[0].type == ZYDIS_OPERAND_TYPE_REGISTER &&
            ops[1].type == ZYDIS_OPERAND_TYPE_IMMEDIATE) {
            const int dst_idx = ZydisGprToIndex(ops[0].reg.value);
            if (dst_idx < 0) return false;
            c.mov(eax, dword[r13 + GprOffset(dst_idx)]);
            c.mov(ecx, static_cast<u32>(ops[1].imm.value.u & 0xFFFFFFFFu));
            c.or_(eax, ecx);
            c.mov(qword[r13 + GprOffset(dst_idx)], rax);  // zero-extend
            EmitFlagsFromBitwise(c);
            return true;
        }
        // Memory source: `or r32, dword[mem]`.
        if (ops[0].type == ZYDIS_OPERAND_TYPE_REGISTER &&
            ops[1].type == ZYDIS_OPERAND_TYPE_MEMORY) {
            const int dst_idx = ZydisGprToIndex(ops[0].reg.value);
            if (dst_idx < 0) return false;
            if (!EmitEffectiveAddress(ops[1].mem, next_rip, c)) return false;
            c.mov(ecx, dword[rdx]);
            c.mov(eax, dword[r13 + GprOffset(dst_idx)]);
            c.or_(eax, ecx);
            c.mov(qword[r13 + GprOffset(dst_idx)], rax);  // zero-extend
            EmitFlagsFromBitwise(c);
            return true;
        }
        return EmitBitwise32RegReg(ops[0], ops[1], c,
            [&](Xbyak::Reg32 a, Xbyak::Reg32 b) { c.or_(a, b); });
    }
    return false;
}

/// ANDN dst, src1, src2 — BMI1: dst = (~src1) & src2. A 3-operand
/// VEX-encoded instruction. Flags: SF and ZF are set from the result,
/// OF and CF are cleared, PF is undefined. Those are exactly the
/// bitwise-AND flag semantics, so EmitFlagsFromBitwise applies. We
/// compute the result manually (not + and) rather than using the host
/// ANDN so the result lands in rax for the flag helper, which expects
/// its input there.
bool EmitAndn(const ZydisDecodedInstruction& insn,
              const ZydisDecodedOperand* ops,
              u64 next_rip,
              Xbyak::CodeGenerator& c) {
    if (insn.operand_width != 64 && insn.operand_width != 32) return false;
    if (ops[0].type != ZYDIS_OPERAND_TYPE_REGISTER) return false;
    if (ops[1].type != ZYDIS_OPERAND_TYPE_REGISTER) return false;
    const int dst_idx  = ZydisGprToIndex(ops[0].reg.value);
    const int src1_idx = ZydisGprToIndex(ops[1].reg.value);
    if (dst_idx < 0 || src1_idx < 0) return false;

    // src2 may be register or memory.
    if (ops[2].type == ZYDIS_OPERAND_TYPE_REGISTER) {
        const int src2_idx = ZydisGprToIndex(ops[2].reg.value);
        if (src2_idx < 0) return false;
        c.mov(rcx, qword[r13 + GprOffset(src2_idx)]);
    } else if (ops[2].type == ZYDIS_OPERAND_TYPE_MEMORY) {
        if (!EmitEffectiveAddress(ops[2].mem, next_rip, c)) return false;
        c.mov(rcx, qword[rdx]);
    } else {
        return false;
    }

    c.mov(rax, qword[r13 + GprOffset(src1_idx)]);
    if (insn.operand_width == 64) {
        c.not_(rax);
        c.and_(rax, rcx);
        c.mov(qword[r13 + GprOffset(dst_idx)], rax);
        EmitFlagsFromBitwise(c);
    } else {
        c.not_(eax);          // operate at 32-bit width
        c.and_(eax, ecx);     // eax result zero-extends rax
        c.mov(qword[r13 + GprOffset(dst_idx)], rax);   // store zero-extended
        // EmitFlagsFromBitwise reads rax at 64-bit width (ZF from all 64,
        // SF from bit 63). Sign-extend the 32-bit result into rax FIRST so
        // ZF reflects the low 32 and SF reflects bit 31 — the store above
        // already captured the architectural zero-extended value.
        c.movsxd(rax, eax);
        EmitFlagsFromBitwise(c);
    }
    return true;
}

/// BEXTR dst, src, control — BMI1 bit-field extract:
///   start = control[7:0], len = control[15:8]
///   dst = (src >> start) & ((1 << len) - 1)
/// A 3-operand VEX instruction (dst, src, control). Flags: ZF is set
/// from the result, CF and OF are cleared, SF/AF/PF are undefined —
/// matching the bitwise-AND flag profile, so EmitFlagsFromBitwise
/// applies. We emit the host BEXTR directly: it implements all the
/// edge cases (start/len beyond the operand width) bit-exactly, and
/// we land the result in rax for the flag helper.
bool EmitBextr(const ZydisDecodedInstruction& insn,
               const ZydisDecodedOperand* ops,
               u64 next_rip,
               Xbyak::CodeGenerator& c) {
    if (insn.operand_width != 64 && insn.operand_width != 32) return false;
    if (ops[0].type != ZYDIS_OPERAND_TYPE_REGISTER) return false;
    if (ops[2].type != ZYDIS_OPERAND_TYPE_REGISTER) return false;
    const int dst_idx  = ZydisGprToIndex(ops[0].reg.value);
    const int ctrl_idx = ZydisGprToIndex(ops[2].reg.value);
    if (dst_idx < 0 || ctrl_idx < 0) return false;

    // Control register into rcx (host BEXTR's third operand).
    c.mov(rcx, qword[r13 + GprOffset(ctrl_idx)]);

    // src may be register or memory.
    if (ops[1].type == ZYDIS_OPERAND_TYPE_REGISTER) {
        const int src_idx = ZydisGprToIndex(ops[1].reg.value);
        if (src_idx < 0) return false;
        c.mov(rdx, qword[r13 + GprOffset(src_idx)]);
    } else if (ops[1].type == ZYDIS_OPERAND_TYPE_MEMORY) {
        if (!EmitEffectiveAddress(ops[1].mem, next_rip, c)) return false;
        c.mov(rdx, qword[rdx]);
    } else {
        return false;
    }

    if (insn.operand_width == 64) {
        c.bextr(rax, rdx, rcx);
    } else {
        c.bextr(eax, edx, ecx);  // 32-bit form zero-extends rax
    }
    c.mov(qword[r13 + GprOffset(dst_idx)], rax);
    EmitFlagsFromBitwise(c);
    return true;
}
// =============================================================================

/// LZCNT dst, src — count leading zero bits (BMI/ABM). For a zero
/// source the result is the operand size (32 or 64) and CF is set;
/// otherwise CF is clear. ZF is set when the result is zero (the MSB
/// of the source was set). We emit the host LZCNT and capture CF/ZF
/// via a flags round-trip.
bool EmitLzcnt(const ZydisDecodedInstruction& insn,
               const ZydisDecodedOperand* ops,
               u64 next_rip,
               Xbyak::CodeGenerator& c) {
    const u32 w = insn.operand_width;
    if (w != 32 && w != 64) return false;
    if (ops[0].type != ZYDIS_OPERAND_TYPE_REGISTER) return false;
    const int dst_idx = ZydisGprToIndex(ops[0].reg.value);
    if (dst_idx < 0) return false;

    if (ops[1].type == ZYDIS_OPERAND_TYPE_REGISTER) {
        const int src_idx = ZydisGprToIndex(ops[1].reg.value);
        if (src_idx < 0) return false;
        c.mov(rcx, qword[r13 + GprOffset(src_idx)]);
    } else if (ops[1].type == ZYDIS_OPERAND_TYPE_MEMORY) {
        if (!EmitEffectiveAddress(ops[1].mem, next_rip, c)) return false;
        // Width-sized load: a qword read for a 32-bit operand over-reads 4
        // bytes and can fault when the operand sits at the end of a mapping.
        if (w == 32) c.mov(ecx, dword[rdx]);
        else         c.mov(rcx, qword[rdx]);
    } else {
        return false;
    }

    if (w == 32) c.lzcnt(eax, ecx);   // result zero-extends rax
    else         c.lzcnt(rax, rcx);

    // Merge: arithmetic flags from the host op, everything else (DF and the
    // other control bits) from the prior guest rflags. A wholesale store of
    // host pushfq output would clear a guest DF=1 and import host IF etc.
    c.pushfq();
    c.pop(r8);
    c.mov(r9, RflagsBits::HostLoadMask);
    c.and_(r8, r9);
    c.mov(r9, qword[r13 + Offsets::Rflags]);
    c.mov(r10, ~RflagsBits::HostLoadMask);
    c.and_(r9, r10);
    c.or_(r8, r9);
    c.mov(qword[r13 + Offsets::Rflags], r8);

    c.mov(qword[r13 + GprOffset(dst_idx)], rax);
    return true;
}

/// TZCNT dst, src — count trailing zero bits (BMI1). For a zero source
/// the result is the operand size (32 or 64) and CF is set; otherwise CF
/// is clear. ZF is set when the result is zero (bit 0 of the source was
/// set). We emit the host TZCNT and capture CF/ZF via a flags round-trip,
/// exactly as LZCNT does.
bool EmitTzcnt(const ZydisDecodedInstruction& insn,
               const ZydisDecodedOperand* ops,
               u64 next_rip,
               Xbyak::CodeGenerator& c) {
    const u32 w = insn.operand_width;
    if (w != 32 && w != 64) return false;
    if (ops[0].type != ZYDIS_OPERAND_TYPE_REGISTER) return false;
    const int dst_idx = ZydisGprToIndex(ops[0].reg.value);
    if (dst_idx < 0) return false;

    if (ops[1].type == ZYDIS_OPERAND_TYPE_REGISTER) {
        const int src_idx = ZydisGprToIndex(ops[1].reg.value);
        if (src_idx < 0) return false;
        c.mov(rcx, qword[r13 + GprOffset(src_idx)]);
    } else if (ops[1].type == ZYDIS_OPERAND_TYPE_MEMORY) {
        if (!EmitEffectiveAddress(ops[1].mem, next_rip, c)) return false;
        // Width-sized load: a qword read for a 32-bit operand over-reads 4
        // bytes and can fault when the operand sits at the end of a mapping.
        if (w == 32) c.mov(ecx, dword[rdx]);
        else         c.mov(rcx, qword[rdx]);
    } else {
        return false;
    }

    if (w == 32) c.tzcnt(eax, ecx);   // result zero-extends rax
    else         c.tzcnt(rax, rcx);

    // Merge: arithmetic flags from the host op, everything else (DF and the
    // other control bits) from the prior guest rflags. A wholesale store of
    // host pushfq output would clear a guest DF=1 and import host IF etc.
    c.pushfq();
    c.pop(r8);
    c.mov(r9, RflagsBits::HostLoadMask);
    c.and_(r8, r9);
    c.mov(r9, qword[r13 + Offsets::Rflags]);
    c.mov(r10, ~RflagsBits::HostLoadMask);
    c.and_(r9, r10);
    c.or_(r8, r9);
    c.mov(qword[r13 + Offsets::Rflags], r8);

    c.mov(qword[r13 + GprOffset(dst_idx)], rax);
    return true;
}

/// POPCNT dst, src — population count (number of set bits). Sets ZF
/// when the source is zero and clears CF/OF/SF/AF/PF. The host POPCNT
/// implements this exactly; we capture its flags via a round-trip.
bool EmitPopcnt(const ZydisDecodedInstruction& insn,
                const ZydisDecodedOperand* ops,
                u64 next_rip,
                Xbyak::CodeGenerator& c) {
    const u32 w = insn.operand_width;
    if (w != 32 && w != 64) return false;
    if (ops[0].type != ZYDIS_OPERAND_TYPE_REGISTER) return false;
    const int dst_idx = ZydisGprToIndex(ops[0].reg.value);
    if (dst_idx < 0) return false;

    if (ops[1].type == ZYDIS_OPERAND_TYPE_REGISTER) {
        const int src_idx = ZydisGprToIndex(ops[1].reg.value);
        if (src_idx < 0) return false;
        c.mov(rcx, qword[r13 + GprOffset(src_idx)]);
    } else if (ops[1].type == ZYDIS_OPERAND_TYPE_MEMORY) {
        if (!EmitEffectiveAddress(ops[1].mem, next_rip, c)) return false;
        // Width-sized load: a qword read for a 32-bit operand over-reads 4
        // bytes and can fault when the operand sits at the end of a mapping.
        if (w == 32) c.mov(ecx, dword[rdx]);
        else         c.mov(rcx, qword[rdx]);
    } else {
        return false;
    }

    if (w == 32) c.popcnt(eax, ecx);   // result zero-extends rax
    else         c.popcnt(rax, rcx);

    // Merge: arithmetic flags from the host op, everything else (DF and the
    // other control bits) from the prior guest rflags. A wholesale store of
    // host pushfq output would clear a guest DF=1 and import host IF etc.
    c.pushfq();
    c.pop(r8);
    c.mov(r9, RflagsBits::HostLoadMask);
    c.and_(r8, r9);
    c.mov(r9, qword[r13 + Offsets::Rflags]);
    c.mov(r10, ~RflagsBits::HostLoadMask);
    c.and_(r9, r10);
    c.or_(r8, r9);
    c.mov(qword[r13 + Offsets::Rflags], r8);

    c.mov(qword[r13 + GprOffset(dst_idx)], rax);
    return true;
}
// =============================================================================

/// NOT r/m — bitwise complement. Per x86 spec, NOT does NOT affect
/// any flags at any width. So we skip the round-trip-flags pattern
/// the binary narrow-arith ops use.
bool EmitNot(const ZydisDecodedInstruction& insn,
             const ZydisDecodedOperand* ops,
             u64 next_rip,
             Xbyak::CodeGenerator& c) {
    // ---- Memory destination: not byte/word/dword/qword [mem]. NOT affects
    //      no flags, so we just complement the operand in place at its width.
    if (ops[0].type == ZYDIS_OPERAND_TYPE_MEMORY) {
        const u32 w = insn.operand_width;
        if (w != 8 && w != 16 && w != 32 && w != 64) return false;
        if (!EmitEffectiveAddress(ops[0].mem, next_rip, c)) return false; // rdx = &dst
        switch (w) {
            case 8:  c.not_(byte[rdx]);  break;
            case 16: c.not_(word[rdx]);  break;
            case 32: c.not_(dword[rdx]); break;
            default: c.not_(qword[rdx]); break;
        }
        return true;
    }
    if (ops[0].type != ZYDIS_OPERAND_TYPE_REGISTER) return false;
    const int idx = ZydisGprToIndex(ops[0].reg.value);
    if (idx < 0) return false;

    switch (insn.operand_width) {
        case 64:
            c.mov(rax, qword[r13 + GprOffset(idx)]);
            c.not_(rax);
            c.mov(qword[r13 + GprOffset(idx)], rax);
            return true;
        case 32:
            // 32-bit writes zero-extend. mov-eax then writeback covers
            // both halves of the 64-bit slot atomically.
            c.mov(eax, dword[r13 + GprOffset(idx)]);
            c.not_(eax);
            c.mov(qword[r13 + GprOffset(idx)], rax);
            return true;
        case 16:
            // 16-bit writes preserve upper 48 bits — use word memory operand.
            c.mov(ax, word[r13 + GprOffset(idx)]);
            c.not_(ax);
            c.mov(word[r13 + GprOffset(idx)], ax);
            return true;
        case 8:
            // 8-bit writes preserve upper 56 bits.
            c.mov(al, byte[r13 + GprOffset(idx)]);
            c.not_(al);
            c.mov(byte[r13 + GprOffset(idx)], al);
            return true;
        default:
            return false;
    }
}

/// NEG r/m — two's complement negate; equivalent to `0 - src`.
/// Flags follow SUB semantics: CF = (src != 0), ZF/SF/OF/PF computed
/// from the result with width-specific semantics.
///
/// For 64-bit we use the lazy-flag helper (EmitFlagsFromSubtract).
/// For 8/16-bit we round-trip flags through the host CPU so it
/// computes correct narrow-width flag values — same pattern the
/// narrow-arith ops (ADD/SUB/...) use.
bool EmitNeg(const ZydisDecodedInstruction& insn,
             const ZydisDecodedOperand* ops,
             u64 next_rip,
             Xbyak::CodeGenerator& c) {
    // ---- Memory destination: neg byte/word/dword/qword [mem]. NEG sets ALL
    //      arithmetic flags (CF = src != 0), so let the host compute them: run
    //      the width-sized neg in place and round-trip rflags around it.
    if (ops[0].type == ZYDIS_OPERAND_TYPE_MEMORY) {
        const u32 w = insn.operand_width;
        if (w != 8 && w != 16 && w != 32 && w != 64) return false;
        if (!EmitEffectiveAddress(ops[0].mem, next_rip, c)) return false; // rdx = &dst
        c.mov(r10, rdx);                          // r10 = &dst (rdx reused below)
        c.mov(rdx, qword[r13 + Offsets::Rflags]);
        c.and_(rdx, RflagsBits::HostLoadMask);
        c.push(rdx);
        c.popfq();
        switch (w) {
            case 8:  c.neg(byte[r10]);  break;
            case 16: c.neg(word[r10]);  break;
            case 32: c.neg(dword[r10]); break;
            default: c.neg(qword[r10]); break;
        }
        c.pushfq();
        c.pop(rdx);
        EmitStoreArithFlags(c, rdx);
        return true;
    }
    if (ops[0].type != ZYDIS_OPERAND_TYPE_REGISTER) return false;
    const int idx = ZydisGprToIndex(ops[0].reg.value);
    if (idx < 0) return false;

    if (insn.operand_width == 64) {
        // Set up flag helper inputs: lhs=0 in rcx, rhs=src in rdx,
        // result in rax.
        c.xor_(rcx, rcx);
        c.mov(rdx, qword[r13 + GprOffset(idx)]);
        c.mov(rax, rcx);
        c.sub(rax, rdx);
        c.mov(qword[r13 + GprOffset(idx)], rax);
        EmitFlagsFromSubtract(c);
        return true;
    }

    // Narrow widths (8/16/32): round-trip flags through host so the
    // host CPU computes width-correct flag bits (CF, ZF, SF, PF).
    if (insn.operand_width != 8 &&
        insn.operand_width != 16 &&
        insn.operand_width != 32) {
        return false;
    }

    c.mov(rdx, qword[r13 + Offsets::Rflags]);
    c.and_(rdx, RflagsBits::HostLoadMask);
    c.push(rdx);
    c.popfq();

    if (insn.operand_width == 8) {
        c.mov(al, byte[r13 + GprOffset(idx)]);
        c.neg(al);
        c.mov(byte[r13 + GprOffset(idx)], al);
    } else if (insn.operand_width == 16) {
        c.mov(ax, word[r13 + GprOffset(idx)]);
        c.neg(ax);
        c.mov(word[r13 + GprOffset(idx)], ax);
    } else { // 32
        // 32-bit writes zero-extend the upper 32 bits of the
        // underlying 64-bit slot. We load into eax (which clears
        // upper 32 of rax automatically), negate, then qword-store
        // so the zero extension reaches memory.
        c.mov(eax, dword[r13 + GprOffset(idx)]);
        c.neg(eax);
        c.mov(qword[r13 + GprOffset(idx)], rax);
    }

    c.pushfq();
    c.pop(rdx);
    EmitStoreArithFlags(c, rdx);
    return true;
}

// =============================================================================
// INC, DEC — single-operand add/sub by 1.
//
// Important x86 quirk: INC and DEC do *not* affect CF. They set
// ZF/SF/OF/PF/AF only. This matters for multi-precision arithmetic
// patterns. We snapshot CF before the operation and restore it
// after so the flag helper (which clobbers CF) can still be reused.
// =============================================================================

/// INC r64 — add 1, preserve CF.
bool EmitInc(const ZydisDecodedInstruction& insn,
             const ZydisDecodedOperand* ops,
             u64 next_rip,
             Xbyak::CodeGenerator& c) {
    const u32 w = insn.operand_width;
    if (w != 64 && w != 32 && w != 16 && w != 8) return false;

    // ---- Memory destination (any width): inc byte/word/dword/qword
    //      [mem]. INC preserves CF and sets ZF/SF/OF/AF/PF. We stash
    //      the address in r10 across the flag round-trip, run the host
    //      inc at the operand width, and restore CF.
    if (ops[0].type == ZYDIS_OPERAND_TYPE_MEMORY) {
        if (!EmitEffectiveAddress(ops[0].mem, next_rip, c)) return false;
        c.mov(r10, rdx);                          // r10 = &dst
        // Load current value (width-sized) into a host reg.
        // Round-trip rflags so the host computes correct narrow flags.
        c.mov(rdx, qword[r13 + Offsets::Rflags]);
        c.mov(r11, rdx);
        c.and_(r11, 0x1);                         // r11 = saved CF
        c.and_(rdx, RflagsBits::HostLoadMask);
        c.push(rdx);
        c.popfq();
        switch (w) {
            case 8:  c.inc(byte[r10]);  break;
            case 16: c.inc(word[r10]);  break;
            case 32: c.inc(dword[r10]); break;
            default: c.inc(qword[r10]); break;
        }
        c.pushfq();
        c.pop(rdx);
        c.btr(rdx, 0);                            // clear CF in new flags
        c.or_(rdx, r11);                          // restore saved CF
        EmitStoreArithFlags(c, rdx);
        return true;
    }

    if (ops[0].type != ZYDIS_OPERAND_TYPE_REGISTER) return false;

    // ---- 8/16-bit register (incl. high-byte AH/CH/DH/BH). Narrow
    //      writes preserve the parent slot's upper bits; CF preserved.
    if (w == 8 || w == 16) {
        c.mov(r10, qword[r13 + Offsets::Rflags]);
        c.and_(r10, 0x1);                         // saved CF
        if (w == 8) {
            const int off = ZydisGpr8ToByteOffset(ops[0].reg.value);
            if (off < 0) return false;
            c.mov(al, byte[r13 + off]);
            c.mov(rdx, qword[r13 + Offsets::Rflags]);
            c.and_(rdx, RflagsBits::HostLoadMask);
            c.push(rdx); c.popfq();
            c.inc(al);
            c.pushfq(); c.pop(rdx);
            c.mov(byte[r13 + off], al);
        } else {  // 16
            const int idx = ZydisGprToIndex(ops[0].reg.value);
            if (idx < 0) return false;
            c.mov(ax, word[r13 + GprOffset(idx)]);
            c.mov(rdx, qword[r13 + Offsets::Rflags]);
            c.and_(rdx, RflagsBits::HostLoadMask);
            c.push(rdx); c.popfq();
            c.inc(ax);
            c.pushfq(); c.pop(rdx);
            c.mov(word[r13 + GprOffset(idx)], ax);
        }
        EmitStoreArithFlags(c, rdx);
        // Restore CF.
        c.mov(r11, qword[r13 + Offsets::Rflags]);
        c.btr(r11, 0);
        c.or_(r11, r10);
        c.mov(qword[r13 + Offsets::Rflags], r11);
        return true;
    }

    const int idx = ZydisGprToIndex(ops[0].reg.value);
    if (idx < 0) return false;

    // Snapshot the existing CF (bit 0 of rflags) into r10. INC must
    // preserve CF, which neither EmitFlagsFromAdd nor the host's
    // own ADD do — we restore it explicitly at the end.
    c.mov(r10, qword[r13 + Offsets::Rflags]);
    c.and_(r10, 0x1);

    if (insn.operand_width == 64) {
        // 64-bit: compute via host 64-bit ADD with the same lazy-flag
        // helper inputs the other arith ops use (lhs/rhs in rcx/rdx,
        // result in rax).
        c.mov(rcx, qword[r13 + GprOffset(idx)]);
        c.mov(rdx, 1);
        c.mov(rax, rcx);
        c.add(rax, rdx);
        c.mov(qword[r13 + GprOffset(idx)], rax);
        EmitFlagsFromAdd(c);
    } else {
        // 32-bit: do the work in eax so the result zero-extends rax
        // (x86-64 32-bit-write semantics). Store the full qword to
        // propagate the zero-extension into the guest slot. Flags
        // are computed by rolling rflags through the host CPU's
        // 32-bit ADD — same round-trip pattern as the narrow-arith
        // path used for 8/16/32-bit ops elsewhere in this file.
        c.mov(eax, dword[r13 + GprOffset(idx)]);
        c.mov(ecx, 1);

        // Round-trip guest rflags through host flags so the host
        // computes correct 32-bit-width flags (CF/OF/SF/ZF/PF/AF).
        c.mov(rdx, qword[r13 + Offsets::Rflags]);
        c.and_(rdx, RflagsBits::HostLoadMask);
        c.push(rdx);
        c.popfq();

        c.add(eax, ecx);

        c.pushfq();
        c.pop(rdx);
        EmitStoreArithFlags(c, rdx);

        // Writeback: store the full rax (low 32 = result, upper 32 =
        // zero from the eax write above) so the guest slot gets the
        // correct x86-64 zero-extension semantic.
        c.mov(qword[r13 + GprOffset(idx)], rax);
    }

    // Restore CF: clear bit 0 then OR in our snapshot. We load
    // rflags into a host register first because xbyak's
    // `and_(qword[mem], imm32)` sign-extends the immediate, and
    // `~0x1ULL = 0xFFFFFFFFFFFFFFFE` doesn't fit in a signed 32-bit
    // immediate. Going through a register sidesteps the encoding limit.
    c.mov(r11, qword[r13 + Offsets::Rflags]);
    c.btr(r11, 0);                                       // clear bit 0 (CF)
    c.or_(r11, r10);                                     // OR in saved CF
    c.mov(qword[r13 + Offsets::Rflags], r11);
    return true;
}

/// DEC r/m — subtract 1, preserve CF. 64- and 32-bit register forms.
bool EmitDec(const ZydisDecodedInstruction& insn,
             const ZydisDecodedOperand* ops,
             u64 next_rip,
             Xbyak::CodeGenerator& c) {
    const u32 w = insn.operand_width;
    if (w != 64 && w != 32 && w != 16 && w != 8) return false;

    // ---- Memory destination (any width): dec byte/word/dword/qword
    //      [mem]. DEC preserves CF and sets ZF/SF/OF/AF/PF. Address in
    //      r10 across the flag round-trip; restore CF afterward.
    if (ops[0].type == ZYDIS_OPERAND_TYPE_MEMORY) {
        if (!EmitEffectiveAddress(ops[0].mem, next_rip, c)) return false;
        c.mov(r10, rdx);
        c.mov(rdx, qword[r13 + Offsets::Rflags]);
        c.mov(r11, rdx);
        c.and_(r11, 0x1);                         // saved CF
        c.and_(rdx, RflagsBits::HostLoadMask);
        c.push(rdx);
        c.popfq();
        switch (w) {
            case 8:  c.dec(byte[r10]);  break;
            case 16: c.dec(word[r10]);  break;
            case 32: c.dec(dword[r10]); break;
            default: c.dec(qword[r10]); break;
        }
        c.pushfq();
        c.pop(rdx);
        c.btr(rdx, 0);
        c.or_(rdx, r11);
        EmitStoreArithFlags(c, rdx);
        return true;
    }

    if (ops[0].type != ZYDIS_OPERAND_TYPE_REGISTER) return false;

    // 8-bit DEC (including high-byte regs AH/CH/DH/BH). DEC preserves
    // CF and sets ZF/SF/OF/PF/AF; an 8-bit write preserves the parent
    // slot's upper 56 bits. We round-trip rflags through the host so
    // the narrow-width flags are computed correctly, then restore CF.
    if (w == 8) {
        const int byte_off = ZydisGpr8ToByteOffset(ops[0].reg.value);
        if (byte_off < 0) return false;

        c.mov(al, byte[r13 + byte_off]);
        c.mov(rdx, qword[r13 + Offsets::Rflags]);
        c.and_(rdx, RflagsBits::HostLoadMask);
        c.push(rdx);
        c.popfq();
        c.dec(al);
        c.pushfq();
        c.pop(rdx);
        // DEC leaves CF unchanged: take CF from the saved guest rflags,
        // and everything else ARITH from the host result -- non-arith bits
        // (DF!) stay the guest's, via the merge store.
        c.mov(r10, qword[r13 + Offsets::Rflags]);
        c.and_(r10, 0x1);          // r10 = old CF
        c.btr(rdx, 0);             // clear CF in the new flags
        c.or_(rdx, r10);           // restore old CF
        EmitStoreArithFlags(c, rdx);
        c.mov(byte[r13 + byte_off], al);
        return true;
    }

    // 16-bit register DEC: preserve upper 48 bits and CF.
    if (w == 16) {
        const int idx16 = ZydisGprToIndex(ops[0].reg.value);
        if (idx16 < 0) return false;
        c.mov(ax, word[r13 + GprOffset(idx16)]);
        c.mov(rdx, qword[r13 + Offsets::Rflags]);
        c.and_(rdx, RflagsBits::HostLoadMask);
        c.push(rdx);
        c.popfq();
        c.dec(ax);
        c.pushfq();
        c.pop(rdx);
        c.mov(r10, qword[r13 + Offsets::Rflags]);
        c.and_(r10, 0x1);
        c.btr(rdx, 0);
        c.or_(rdx, r10);
        EmitStoreArithFlags(c, rdx);
        c.mov(word[r13 + GprOffset(idx16)], ax);
        return true;
    }

    const int idx = ZydisGprToIndex(ops[0].reg.value);
    if (idx < 0) return false;

    // Snapshot CF (bit 0 of rflags) into r10 so we can restore it
    // after the SUB clobbers it. DEC must preserve CF.
    c.mov(r10, qword[r13 + Offsets::Rflags]);
    c.and_(r10, 0x1);

    if (insn.operand_width == 64) {
        c.mov(rcx, qword[r13 + GprOffset(idx)]);
        c.mov(rdx, 1);
        c.mov(rax, rcx);
        c.sub(rax, rdx);
        c.mov(qword[r13 + GprOffset(idx)], rax);
        EmitFlagsFromSubtract(c);
    } else {
        // 32-bit: same round-trip-through-host-flags pattern used by
        // the 32-bit INC path. The host's 32-bit SUB computes
        // CF/OF/SF/ZF/PF/AF at 32-bit width; we capture all of them
        // by snapshotting rflags before and after the host op. The
        // 32-bit write into eax zero-extends rax, and we store the
        // full qword so the guest slot inherits the zero-extension.
        c.mov(eax, dword[r13 + GprOffset(idx)]);
        c.mov(ecx, 1);

        c.mov(rdx, qword[r13 + Offsets::Rflags]);
        c.and_(rdx, RflagsBits::HostLoadMask);
        c.push(rdx);
        c.popfq();

        c.sub(eax, ecx);

        c.pushfq();
        c.pop(rdx);
        EmitStoreArithFlags(c, rdx);

        c.mov(qword[r13 + GprOffset(idx)], rax);
    }

    // Restore CF: clear bit 0 of rflags, OR in the saved CF. Going
    // through r11 sidesteps xbyak's sign-extension of imm32 in the
    // mem-form AND (see the same trick in EmitInc).
    c.mov(r11, qword[r13 + Offsets::Rflags]);
    c.btr(r11, 0);
    c.or_(r11, r10);
    c.mov(qword[r13 + Offsets::Rflags], r11);
    return true;
}

// Forward declaration: EmitJccCondition (defined alongside EmitJcc)
// computes a 0/1 condition indicator into rcx from guest rflags.
bool EmitJccCondition(ZydisMnemonic mnemonic, Xbyak::CodeGenerator& c);

/// Map a SETcc mnemonic to the Jcc that shares its condition, so we
/// can reuse EmitJccCondition. (Same table as CmovToJcc but keyed on
/// the SETcc mnemonics.)
ZydisMnemonic SetccToJcc(ZydisMnemonic m) {
    switch (m) {
        case ZYDIS_MNEMONIC_SETZ:   return ZYDIS_MNEMONIC_JZ;
        case ZYDIS_MNEMONIC_SETNZ:  return ZYDIS_MNEMONIC_JNZ;
        case ZYDIS_MNEMONIC_SETS:   return ZYDIS_MNEMONIC_JS;
        case ZYDIS_MNEMONIC_SETNS:  return ZYDIS_MNEMONIC_JNS;
        case ZYDIS_MNEMONIC_SETO:   return ZYDIS_MNEMONIC_JO;
        case ZYDIS_MNEMONIC_SETNO:  return ZYDIS_MNEMONIC_JNO;
        case ZYDIS_MNEMONIC_SETP:   return ZYDIS_MNEMONIC_JP;
        case ZYDIS_MNEMONIC_SETNP:  return ZYDIS_MNEMONIC_JNP;
        case ZYDIS_MNEMONIC_SETB:   return ZYDIS_MNEMONIC_JB;
        case ZYDIS_MNEMONIC_SETNB:  return ZYDIS_MNEMONIC_JNB;
        case ZYDIS_MNEMONIC_SETBE:  return ZYDIS_MNEMONIC_JBE;
        case ZYDIS_MNEMONIC_SETNBE: return ZYDIS_MNEMONIC_JNBE;
        case ZYDIS_MNEMONIC_SETL:   return ZYDIS_MNEMONIC_JL;
        case ZYDIS_MNEMONIC_SETNL:  return ZYDIS_MNEMONIC_JNL;
        case ZYDIS_MNEMONIC_SETLE:  return ZYDIS_MNEMONIC_JLE;
        case ZYDIS_MNEMONIC_SETNLE: return ZYDIS_MNEMONIC_JNLE;
        default: return ZYDIS_MNEMONIC_INVALID;
    }
}

/// SETcc r/m8 — set the byte operand to 1 if the condition holds, else
/// 0. We compute the 0/1 indicator via EmitJccCondition (lands in rcx)
/// and store the low byte. Register low-byte destinations preserve the
/// upper 56 bits of the parent slot; memory destinations store one byte.
/// (High-byte registers like AH are not produced by observed code in a
/// SETcc context and are rejected, matching the other 8-bit emitters.)
bool EmitSetcc(const ZydisDecodedInstruction& insn,
               const ZydisDecodedOperand* ops,
               u64 next_rip,
               Xbyak::CodeGenerator& c) {
    const ZydisMnemonic jcc_equiv = SetccToJcc(insn.mnemonic);
    if (jcc_equiv == ZYDIS_MNEMONIC_INVALID) return false;

    if (ops[0].type == ZYDIS_OPERAND_TYPE_MEMORY) {
        // Compute condition first (uses rax/rcx), then the address.
        // EmitEffectiveAddress clobbers rax and returns the address in
        // rdx; condition is preserved in rcx across it.
        if (!EmitJccCondition(jcc_equiv, c)) return false;
        c.mov(r8, rcx);                       // stash indicator
        if (!EmitEffectiveAddress(ops[0].mem, next_rip, c)) return false;
        c.mov(byte[rdx], r8b);
        return true;
    }

    if (ops[0].type != ZYDIS_OPERAND_TYPE_REGISTER) return false;
    const int dst_idx = ZydisGprToIndex(ops[0].reg.value);
    if (dst_idx < 0) return false;

    if (!EmitJccCondition(jcc_equiv, c)) return false;
    // Store low byte (cl) into the parent slot's byte 0, preserving the
    // upper 56 bits.
    c.mov(byte[r13 + GprOffset(dst_idx)], cl);
    return true;
}

/// DIV r/m8 — unsigned 8-bit divide. Dividend is the 16-bit AX; the
/// quotient goes to AL and the remainder to AH. The upper 48 bits of
/// RAX are preserved. The single operand is the divisor (register or
/// memory). Flags are undefined after DIV, so we leave guest rflags
/// untouched. (Wider DIV forms are not yet observed; this handles the
/// 8-bit form seen in the boot path.)
bool EmitDiv(const ZydisDecodedInstruction& insn,
             const ZydisDecodedOperand* ops,
             u64 next_rip,
             Xbyak::CodeGenerator& c) {
    const u32 w = insn.operand_width;
    if (w != 8 && w != 32 && w != 64) return false;

    // ---- 8-bit: dividend AX, quotient AL, remainder AH. ----
    if (w == 8) {
        // Load AX (16-bit dividend) into host ax. We must preserve RAX's
        // upper 48 bits, so we save them, do the divide, then merge.
        c.mov(r8, qword[r13 + GprOffset(0)]);     // r8 = full guest RAX
        c.mov(ax, word[r13 + GprOffset(0)]);      // ax = dividend (AX)

        if (ops[0].type == ZYDIS_OPERAND_TYPE_REGISTER) {
            const int d_idx = ZydisGprToIndex(ops[0].reg.value);
            if (d_idx < 0) return false;
            c.mov(cl, byte[r13 + GprOffset(d_idx)]);
        } else if (ops[0].type == ZYDIS_OPERAND_TYPE_MEMORY) {
            if (!EmitEffectiveAddress(ops[0].mem, next_rip, c)) return false;
            c.mov(cl, byte[rdx]);
            c.mov(ax, r8w);                       // EA clobbered rax; reload
        } else {
            return false;
        }

        c.div(cl);                                // AL = AX/cl, AH = AX%cl

        c.movzx(rdx, ax);
        c.mov(rcx, 0xFFFFFFFFFFFF0000ULL);
        c.and_(r8, rcx);
        c.or_(r8, rdx);
        c.mov(qword[r13 + GprOffset(0)], r8);
        return true;
    }

    // ---- 32/64-bit: dividend (E/R)DX:(E/R)AX, quotient (E/R)AX,
    //      remainder (E/R)DX. ----
    // We load the divisor into a scratch (r9) first, then set up the
    // host RDX:RAX from the guest slots. The 32-bit form must load only
    // the low 32 of each (upper 32 are ignored by 32-bit DIV) and its
    // results zero-extend the guest RAX/RDX slots.
    if (ops[0].type == ZYDIS_OPERAND_TYPE_REGISTER) {
        const int d_idx = ZydisGprToIndex(ops[0].reg.value);
        if (d_idx < 0) return false;
        c.mov(r9, qword[r13 + GprOffset(d_idx)]);
    } else if (ops[0].type == ZYDIS_OPERAND_TYPE_MEMORY) {
        if (!EmitEffectiveAddress(ops[0].mem, next_rip, c)) return false;
        // Width-sized read: an 8-byte load for `div dword [m]` over-reads
        // past the operand and can fault at a mapping boundary.
        if (w == 32) c.mov(r9d, dword[rdx]);
        else         c.mov(r9,  qword[rdx]);
    } else {
        return false;
    }

    if (w == 32) {
        c.mov(eax, dword[r13 + GprOffset(0)]);    // EAX = dividend low
        c.mov(edx, dword[r13 + GprOffset(2)]);    // EDX = dividend high
        c.div(r9d);                               // EAX=quot, EDX=rem
        // 32-bit results already zero-extend rax/rdx; store full slots.
        c.mov(qword[r13 + GprOffset(0)], rax);
        c.mov(qword[r13 + GprOffset(2)], rdx);
        return true;
    }

    // w == 64: dividend RDX:RAX, quotient RAX, remainder RDX.
    c.mov(rax, qword[r13 + GprOffset(0)]);
    c.mov(rdx, qword[r13 + GprOffset(2)]);
    c.div(r9);
    c.mov(qword[r13 + GprOffset(0)], rax);
    c.mov(qword[r13 + GprOffset(2)], rdx);
    return true;
}

/// IDIV r/m — signed division. Mirrors EmitDiv exactly but uses the host's
/// native `idiv`, which already matches x86 IDIV semantics (the host IS x86).
/// Dividend implicit in (R/E)DX:(R/E)AX (AX for 8-bit); quotient -> (R/E)AX/AL,
/// remainder -> (R/E)DX/AH. The guest is responsible for having sign-extended
/// the dividend into the high half (via CWD/CDQ/CQO/CBW) before IDIV.
bool EmitIdiv(const ZydisDecodedInstruction& insn,
              const ZydisDecodedOperand* ops,
              u64 next_rip,
              Xbyak::CodeGenerator& c) {
    const u32 w = insn.operand_width;
    if (w != 8 && w != 32 && w != 64) return false;

    // ---- 8-bit: dividend AX, quotient AL, remainder AH. ----
    if (w == 8) {
        c.mov(r8, qword[r13 + GprOffset(0)]);     // r8 = full guest RAX
        c.mov(ax, word[r13 + GprOffset(0)]);      // ax = dividend (AX)

        if (ops[0].type == ZYDIS_OPERAND_TYPE_REGISTER) {
            const int d_idx = ZydisGprToIndex(ops[0].reg.value);
            if (d_idx < 0) return false;
            c.mov(cl, byte[r13 + GprOffset(d_idx)]);
        } else if (ops[0].type == ZYDIS_OPERAND_TYPE_MEMORY) {
            if (!EmitEffectiveAddress(ops[0].mem, next_rip, c)) return false;
            c.mov(cl, byte[rdx]);
            c.mov(ax, r8w);                       // EA clobbered rax; reload
        } else {
            return false;
        }

        c.idiv(cl);                               // AL = AX/cl, AH = AX%cl (signed)

        c.movzx(rdx, ax);
        c.mov(rcx, 0xFFFFFFFFFFFF0000ULL);
        c.and_(r8, rcx);
        c.or_(r8, rdx);
        c.mov(qword[r13 + GprOffset(0)], r8);
        return true;
    }

    // ---- 32/64-bit: load divisor into r9 (handling reg/mem), then set up
    //      host (R/E)DX:(R/E)AX from the guest slots and run idiv. ----
    if (ops[0].type == ZYDIS_OPERAND_TYPE_REGISTER) {
        const int d_idx = ZydisGprToIndex(ops[0].reg.value);
        if (d_idx < 0) return false;
        c.mov(r9, qword[r13 + GprOffset(d_idx)]);
    } else if (ops[0].type == ZYDIS_OPERAND_TYPE_MEMORY) {
        if (!EmitEffectiveAddress(ops[0].mem, next_rip, c)) return false;
        // Width-sized read: an 8-byte load for `div dword [m]` over-reads
        // past the operand and can fault at a mapping boundary.
        if (w == 32) c.mov(r9d, dword[rdx]);
        else         c.mov(r9,  qword[rdx]);
    } else {
        return false;
    }

    if (w == 32) {
        c.mov(eax, dword[r13 + GprOffset(0)]);    // EAX = dividend low
        c.mov(edx, dword[r13 + GprOffset(2)]);    // EDX = dividend high
        c.idiv(r9d);                              // EAX=quot, EDX=rem (signed)
        c.mov(qword[r13 + GprOffset(0)], rax);    // results zero-extend
        c.mov(qword[r13 + GprOffset(2)], rdx);
        return true;
    }

    // w == 64: dividend RDX:RAX, quotient RAX, remainder RDX.
    c.mov(rax, qword[r13 + GprOffset(0)]);
    c.mov(rdx, qword[r13 + GprOffset(2)]);
    c.idiv(r9);
    c.mov(qword[r13 + GprOffset(0)], rax);
    c.mov(qword[r13 + GprOffset(2)], rdx);
    return true;
}

/// XADD dst, src — exchange-and-add:
///   tmp = dst + src; src = dst (old); dst = tmp
/// Sets all arithmetic flags (CF/OF/SF/ZF/AF/PF) from the addition.
/// dst is register or memory; src is a register. We compute with the
/// host XADD so flags and the exchange happen exactly per spec. 32-
/// and 64-bit forms supported.
///
/// Note: this is the NON-LOCKED path. A LOCK-prefixed XADD is intercepted by
/// the dispatcher and emitted atomically via EmitLockedRmw; this handler
/// covers the unlocked form (and register-destination XADD, where LOCK is
/// illegal).
bool EmitXadd(const ZydisDecodedInstruction& insn,
              const ZydisDecodedOperand* ops,
              u64 next_rip,
              Xbyak::CodeGenerator& c) {
    const u32 w = insn.operand_width;
    if (w != 64 && w != 32) return false;
    if (ops[1].type != ZYDIS_OPERAND_TYPE_REGISTER) return false;
    const int src_idx = ZydisGprToIndex(ops[1].reg.value);
    if (src_idx < 0) return false;

    // src value into rcx (this register both supplies the addend and
    // receives the old dst value after XADD).
    c.mov(rcx, qword[r13 + GprOffset(src_idx)]);

    if (ops[0].type == ZYDIS_OPERAND_TYPE_REGISTER) {
        const int dst_idx = ZydisGprToIndex(ops[0].reg.value);
        if (dst_idx < 0) return false;
        c.mov(rax, qword[r13 + GprOffset(dst_idx)]);   // rax = dst
        if (w == 64) c.xadd(rax, rcx);
        else         c.xadd(eax, ecx);
        c.pushfq();
        c.pop(r8);
        EmitStoreArithFlags(c, r8);
        // rax = sum (new dst); rcx = old dst (new src). 32-bit results
        // already zero-extend; store full qword slots.
        c.mov(qword[r13 + GprOffset(dst_idx)], rax);
        c.mov(qword[r13 + GprOffset(src_idx)], rcx);
        return true;
    }

    if (ops[0].type == ZYDIS_OPERAND_TYPE_MEMORY) {
        if (!EmitEffectiveAddress(ops[0].mem, next_rip, c)) return false;
        c.mov(r10, rdx);                               // r10 = &dst
        if (w == 64) c.xadd(qword[r10], rcx);
        else         c.xadd(dword[r10], ecx);
        c.pushfq();
        c.pop(r8);
        EmitStoreArithFlags(c, r8);
        // Memory now holds the sum; rcx holds the old memory value,
        // which XADD wrote back into the source register.
        c.mov(qword[r13 + GprOffset(src_idx)], rcx);
        return true;
    }

    return false;
}

/// XCHG dst, src — exchange the two operands:
///   tmp = dst; dst = src; src = tmp
/// XCHG affects NO flags. When one operand is memory the operation is
/// XCHG — exchange a register with a register or memory operand. Widths
/// 8/16/32/64. Register destinations zero-extend in the 32-bit form (any
/// 32-bit register write clears bits 63:32); 8/16-bit forms preserve the
/// surrounding bits, and 8-bit resolves AH/CH/DH/BH to byte 1 of the parent
/// slot. We never touch the guest rflags slot.
///
/// XCHG with a memory operand is implicitly atomic (the CPU asserts LOCK with
/// no prefix needed). We honor that by emitting the host XCHG against memory
/// rather than a plain load/store pair, so a guest spinlock / test-and-set on
/// a shared location stays correct if more than one thread runs through the
/// JIT. The reg,reg form has no memory and is a simple slot swap.
bool EmitXchg(const ZydisDecodedInstruction& insn,
              const ZydisDecodedOperand* ops,
              u64 next_rip,
              Xbyak::CodeGenerator& c) {
    const u32 w = insn.operand_width;
    if (w != 8 && w != 16 && w != 32 && w != 64) return false;

    // Byte offset of a GPR operand of width `w` in the register file. For
    // 8-bit operands this resolves the high-byte regs (AH/CH/DH/BH) to byte 1
    // of their parent slot; wider operands sit at the slot base. -1 = bad reg.
    auto reg_off = [&](ZydisRegister r) -> int {
        if (w == 8) return ZydisGpr8ToByteOffset(r);
        const int i = ZydisGprToIndex(r);
        return (i < 0) ? -1 : static_cast<int>(GprOffset(i));
    };

    // reg, reg — straight swap of two guest slots (no atomicity concern).
    if (ops[0].type == ZYDIS_OPERAND_TYPE_REGISTER &&
        ops[1].type == ZYDIS_OPERAND_TYPE_REGISTER) {
        const int a_off = reg_off(ops[0].reg.value);
        const int b_off = reg_off(ops[1].reg.value);
        if (a_off < 0 || b_off < 0) return false;
        if (a_off == b_off) return true;   // same location — no-op
        switch (w) {
            case 8:
                c.mov(al, byte[r13 + a_off]); c.mov(cl, byte[r13 + b_off]);
                c.mov(byte[r13 + a_off], cl); c.mov(byte[r13 + b_off], al); break;
            case 16:
                c.mov(ax, word[r13 + a_off]); c.mov(cx, word[r13 + b_off]);
                c.mov(word[r13 + a_off], cx); c.mov(word[r13 + b_off], ax); break;
            case 32:
                c.mov(eax, dword[r13 + a_off]); c.mov(ecx, dword[r13 + b_off]);
                c.mov(qword[r13 + a_off], rcx); c.mov(qword[r13 + b_off], rax); break;  // zero-extend both
            case 64:
                c.mov(rax, qword[r13 + a_off]); c.mov(rcx, qword[r13 + b_off]);
                c.mov(qword[r13 + a_off], rcx); c.mov(qword[r13 + b_off], rax); break;
        }
        return true;
    }

    // mem, reg  or  reg, mem — identify which operand is which.
    const ZydisDecodedOperand* reg_op = nullptr;
    const ZydisDecodedOperand* mem_op = nullptr;
    if (ops[0].type == ZYDIS_OPERAND_TYPE_MEMORY &&
        ops[1].type == ZYDIS_OPERAND_TYPE_REGISTER) {
        mem_op = &ops[0]; reg_op = &ops[1];
    } else if (ops[0].type == ZYDIS_OPERAND_TYPE_REGISTER &&
               ops[1].type == ZYDIS_OPERAND_TYPE_MEMORY) {
        reg_op = &ops[0]; mem_op = &ops[1];
    } else {
        return false;
    }
    const int roff = reg_off(reg_op->reg.value);
    if (roff < 0) return false;

    // Address into r10 (EmitEffectiveAddress returns it in rdx and uses
    // rax/rcx as scratch). Then atomically swap reg <-> [mem] with the host
    // XCHG; afterwards the host scratch holds the old memory value, which we
    // write back to the guest reg slot with the correct width semantics.
    if (!EmitEffectiveAddress(mem_op->mem, next_rip, c)) return false;
    c.mov(r10, rdx);                                          // r10 = &mem
    switch (w) {
        case 8:
            c.mov(al,  byte [r13 + roff]); c.xchg(byte [r10], al);  c.mov(byte [r13 + roff], al);  break;
        case 16:
            c.mov(ax,  word [r13 + roff]); c.xchg(word [r10], ax);  c.mov(word [r13 + roff], ax);  break;
        case 32:
            c.mov(eax, dword[r13 + roff]); c.xchg(dword[r10], eax); c.mov(qword[r13 + roff], rax); break;  // zero-extend
        case 64:
            c.mov(rax, qword[r13 + roff]); c.xchg(qword[r10], rax); c.mov(qword[r13 + roff], rax); break;
    }
    return true;
}

/// CMPXCHG dst, src — compare accumulator (AL/AX/EAX/RAX) with dst:
///   if (acc == dst) { ZF=1; dst = src; }
///   else            { ZF=0; acc = dst; }
/// Flags are set exactly as `cmp acc, dst`. We emit the host CMPXCHG
/// directly (it implements the whole semantics, including the flag
/// rules) on host rax + a scratch src register, with a guest-rflags
/// round-trip so the host's ZF/CF/etc. land in the guest. dst may be a
/// register or memory; the accumulator and result writebacks go to the
/// guest slots. 32- and 64-bit forms supported (the boot path uses 32).
///
/// Note: this is the NON-LOCKED path. A LOCK-prefixed CMPXCHG never reaches
/// here — the dispatcher routes it to EmitLockedRmw, which emits a real atomic
/// host CMPXCHG against guest memory (guest threads run concurrently on
/// parallel host threads over shared memory). This handler covers the
/// unlocked form and the register-destination form (where LOCK is illegal).
bool EmitCmpxchg(const ZydisDecodedInstruction& insn,
                 const ZydisDecodedOperand* ops,
                 u64 next_rip,
                 Xbyak::CodeGenerator& c) {
    const u32 w = insn.operand_width;
    if (w != 64 && w != 32) return false;
    if (ops[1].type != ZYDIS_OPERAND_TYPE_REGISTER) return false;
    const int src_idx = ZydisGprToIndex(ops[1].reg.value);
    if (src_idx < 0) return false;

    // CMPXCHG fully defines ZF/CF/SF/OF/AF/PF from the compare, so we
    // don't need to preload guest flags — we just capture the host
    // flags afterward. src goes into rcx, accumulator into rax.
    c.mov(rcx, qword[r13 + GprOffset(src_idx)]);

    if (ops[0].type == ZYDIS_OPERAND_TYPE_REGISTER) {
        const int dst_idx = ZydisGprToIndex(ops[0].reg.value);
        if (dst_idx < 0) return false;
        c.mov(rax, qword[r13 + GprOffset(0)]);
        c.mov(r9, qword[r13 + GprOffset(dst_idx)]);
        if (w == 64) c.cmpxchg(r9, rcx);
        else         c.cmpxchg(r9d, ecx);
        c.pushfq();
        c.pop(r8);
        EmitStoreArithFlags(c, r8);
        // dst and accumulator may both have changed; 32-bit results are
        // already zero-extended in r9/rax, so a full-qword store is right.
        c.mov(qword[r13 + GprOffset(dst_idx)], r9);
        c.mov(qword[r13 + GprOffset(0)], rax);
        return true;
    }

    if (ops[0].type == ZYDIS_OPERAND_TYPE_MEMORY) {
        // Address first (clobbers rax); stash in r10, then load acc.
        if (!EmitEffectiveAddress(ops[0].mem, next_rip, c)) return false;
        c.mov(r10, rdx);
        c.mov(rax, qword[r13 + GprOffset(0)]);
        if (w == 64) c.cmpxchg(qword[r10], rcx);
        else         c.cmpxchg(dword[r10], ecx);
        c.pushfq();
        c.pop(r8);
        EmitStoreArithFlags(c, r8);
        c.mov(qword[r13 + GprOffset(0)], rax);  // acc updated on mismatch
        return true;
    }

    return false;
}

/// EmitLockedRmw — atomic read-modify-write for LOCK-prefixed memory ops.
///
/// Guest threads run on parallel host threads over shared guest memory, so a
/// LOCK-prefixed op (mutex/spinlock/refcount/CAS primitive) must be a single
/// atomic step, not the load-op-store the per-op handlers emit. We honor that
/// by emitting the host instruction with a real LOCK prefix (db 0xF0) directly
/// against guest memory; on an x86 (TSO) host that is sufficient — no extra
/// fences needed. The effective address is computed into r10 first.
///
/// Flags: we round-trip the guest rflags through the host (popfq before,
/// pushfq after) so the host instruction's own flag input/output behavior is
/// exactly the guest semantics — this gives CF-preservation for INC/DEC,
/// CF-consumption for ADC/SBB, and full arithmetic flags for the rest, with no
/// per-op special-casing. The round-trip uses r8 so it never clobbers the rax
/// accumulator that CMPXCHG/XADD rely on.
///
/// Covered: ADD/ADC/SUB/SBB/AND/OR/XOR (reg or imm source), INC/DEC/NEG/NOT,
/// XADD, CMPXCHG. Anything else (BTS/BTR/BTC, CMPXCHG8B/16B) returns false and
/// the caller falls back to the existing handler — same behavior as before,
/// just not yet atomic.
bool EmitLockedRmw(const ZydisDecodedInstruction& insn, const ZydisDecodedOperand* ops,
                   u64 next_rip, Xbyak::CodeGenerator& c) {
    if ((insn.attributes & ZYDIS_ATTRIB_HAS_LOCK) == 0)
        return false;
    if (ops[0].type != ZYDIS_OPERAND_TYPE_MEMORY)
        return false; // LOCK ⇒ mem dest
    const u32 w = insn.operand_width;
    if (w != 8 && w != 16 && w != 32 && w != 64)
        return false;
    const ZydisMnemonic m = insn.mnemonic;

    // Width-sized memory operand at the computed address (in r10).
    auto mem = [&]() -> Xbyak::Address {
        switch (w) {
        case 8:
            return byte[r10];
        case 16:
            return word[r10];
        case 32:
            return dword[r10];
        default:
            return qword[r10];
        }
    };
    // Width-sized views of the rcx (source) and rax (accumulator) scratch regs.
    auto rcxw = [&]() -> Xbyak::Reg {
        switch (w) {
        case 8:
            return cl;
        case 16:
            return cx;
        case 32:
            return ecx;
        default:
            return rcx;
        }
    };
    auto raxw = [&]() -> Xbyak::Reg {
        switch (w) {
        case 8:
            return al;
        case 16:
            return ax;
        case 32:
            return eax;
        default:
            return rax;
        }
    };
    // Byte offset of a GPR operand of width w in the register file (resolves
    // AH/CH/DH/BH for the 8-bit case). -1 on an unsupported register.
    auto gpr_off = [&](ZydisRegister r) -> int {
        if (w == 8)
            return ZydisGpr8ToByteOffset(r);
        const int i = ZydisGprToIndex(r);
        return (i < 0) ? -1 : static_cast<int>(GprOffset(i));
    };
    auto load_rcx = [&](int off) {
        switch (w) {
        case 8:
            c.mov(cl, byte[r13 + off]);
            break;
        case 16:
            c.mov(cx, word[r13 + off]);
            break;
        case 32:
            c.mov(ecx, dword[r13 + off]);
            break;
        default:
            c.mov(rcx, qword[r13 + off]);
        }
    };
    auto load_rax = [&](int off) {
        switch (w) {
        case 8:
            c.mov(al, byte[r13 + off]);
            break;
        case 16:
            c.mov(ax, word[r13 + off]);
            break;
        case 32:
            c.mov(eax, dword[r13 + off]);
            break;
        default:
            c.mov(rax, qword[r13 + off]);
        }
    };
    // Write back a result reg to a guest slot with the right width semantics
    // (8/16 preserve surrounding bits; 32 already zero-extended by the host op).
    auto store_rcx = [&](int off) {
        switch (w) {
        case 8:
            c.mov(byte[r13 + off], cl);
            break;
        case 16:
            c.mov(word[r13 + off], cx);
            break;
        default:
            c.mov(qword[r13 + off], rcx);
        }
    };
    auto store_rax = [&](int off) {
        switch (w) {
        case 8:
            c.mov(byte[r13 + off], al);
            break;
        case 16:
            c.mov(word[r13 + off], ax);
            break;
        default:
            c.mov(qword[r13 + off], rax);
        }
    };
    // Guest rflags <-> host, via r8 (keeps rax/rcx free for operands/results).
    auto load_flags = [&] {
        c.mov(r8, qword[r13 + Offsets::Rflags]);
        c.and_(r8, RflagsBits::HostLoadMask);
        c.push(r8);
        c.popfq();
    };
    auto save_flags = [&] {
        c.pushfq();
        c.pop(r8);
        EmitStoreArithFlags(c, r8);
    };

    // Effective address into r10 first (EmitEffectiveAddress returns rdx and
    // uses rax/rcx as scratch and perturbs host flags, so it must precede the
    // flag load below).
    if (!EmitEffectiveAddress(ops[0].mem, next_rip, c))
        return false;
    c.mov(r10, rdx);

    switch (m) {
    // ---- binary RMW: [mem] op= (reg | imm) ----
    case ZYDIS_MNEMONIC_ADD:
    case ZYDIS_MNEMONIC_ADC:
    case ZYDIS_MNEMONIC_SUB:
    case ZYDIS_MNEMONIC_SBB:
    case ZYDIS_MNEMONIC_AND:
    case ZYDIS_MNEMONIC_OR:
    case ZYDIS_MNEMONIC_XOR: {
        const bool src_imm = (ops[1].type == ZYDIS_OPERAND_TYPE_IMMEDIATE);
        const u32 immv = src_imm ? static_cast<u32>(ops[1].imm.value.u) : 0;
        if (!src_imm) {
            if (ops[1].type != ZYDIS_OPERAND_TYPE_REGISTER)
                return false;
            const int so = gpr_off(ops[1].reg.value);
            if (so < 0)
                return false;
            load_rcx(so);
        }
        load_flags();
#define LOCK_BIN(host)                                                                             \
    do {                                                                                           \
        c.db(0xF0);                                                                                \
        if (src_imm)                                                                               \
            c.host(mem(), immv);                                                                   \
        else                                                                                       \
            c.host(mem(), rcxw());                                                                 \
    } while (0)
        switch (m) {
        case ZYDIS_MNEMONIC_ADD:
            LOCK_BIN(add);
            break;
        case ZYDIS_MNEMONIC_ADC:
            LOCK_BIN(adc);
            break;
        case ZYDIS_MNEMONIC_SUB:
            LOCK_BIN(sub);
            break;
        case ZYDIS_MNEMONIC_SBB:
            LOCK_BIN(sbb);
            break;
        case ZYDIS_MNEMONIC_AND:
            LOCK_BIN(and_);
            break;
        case ZYDIS_MNEMONIC_OR:
            LOCK_BIN(or_);
            break;
        case ZYDIS_MNEMONIC_XOR:
            LOCK_BIN(xor_);
            break;
        default:
            return false;
        }
#undef LOCK_BIN
        save_flags();
        return true;
    }

    // ---- unary RMW ----
    case ZYDIS_MNEMONIC_INC:
    case ZYDIS_MNEMONIC_DEC:
    case ZYDIS_MNEMONIC_NEG: {
        load_flags(); // INC/DEC preserve CF; NEG sets all
        c.db(0xF0);
        if (m == ZYDIS_MNEMONIC_INC)
            c.inc(mem());
        else if (m == ZYDIS_MNEMONIC_DEC)
            c.dec(mem());
        else
            c.neg(mem());
        save_flags();
        return true;
    }
    case ZYDIS_MNEMONIC_NOT: {
        c.db(0xF0);
        c.not_(mem()); // NOT affects no flags
        return true;
    }

    // ---- XADD: [mem] and reg exchange-add; reg receives old [mem] ----
    case ZYDIS_MNEMONIC_XADD: {
        if (ops[1].type != ZYDIS_OPERAND_TYPE_REGISTER)
            return false;
        const int ro = gpr_off(ops[1].reg.value);
        if (ro < 0)
            return false;
        load_rcx(ro);
        load_flags();
        c.db(0xF0);
        c.xadd(mem(), rcxw()); // rcx <- old [mem]; [mem] += rcx
        save_flags();
        store_rcx(ro);
        return true;
    }

    // ---- CMPXCHG: compare RAX-accumulator with [mem]; swap or reload ----
    case ZYDIS_MNEMONIC_CMPXCHG: {
        if (ops[1].type != ZYDIS_OPERAND_TYPE_REGISTER)
            return false;
        const int so = gpr_off(ops[1].reg.value);
        if (so < 0)
            return false;
        load_rax(static_cast<int>(GprOffset(0))); // accumulator = guest RAX
        load_rcx(so);                             // source register
        load_flags();
        c.db(0xF0);
        c.cmpxchg(mem(), rcxw()); // implicit rax compare
        save_flags();
        store_rax(static_cast<int>(GprOffset(0))); // rax updated on mismatch
        return true;
    }

    // ---- BTS/BTR/BTC: atomic bit test-and-{set,reset,complement} ----
    //
    // Host passthrough: the host is x86, so `lock bts/btr/btc` reproduces the
    // guest instruction EXACTLY -- including the register-offset form's
    // bit-string addressing (the bit index is a SIGNED offset; the accessed
    // byte is EA + (index >> 3), which may sit well outside, and even below,
    // the nominal operand) and the immediate form's modulo-operand-size
    // masking. Both are done by hardware identically on host and guest, so
    // no emulation of the addressing is needed or wanted. CF receives the
    // pre-update bit value; the remaining arith flags are architecturally
    // undefined after the BT family, so capturing whatever the host produced
    // is a conforming choice. No load_flags(): these read no flags, and the
    // store goes through the arith-only merge, so guest DF and friends are
    // untouched.
    case ZYDIS_MNEMONIC_BTS:
    case ZYDIS_MNEMONIC_BTR:
    case ZYDIS_MNEMONIC_BTC: {
        if (w == 8)
            return false; // no 8-bit form architecturally
        if (ops[1].type == ZYDIS_OPERAND_TYPE_REGISTER) {
            const int so = gpr_off(ops[1].reg.value);
            if (so < 0)
                return false;
            load_rcx(so);
            c.db(0xF0);
            switch (m) {
            case ZYDIS_MNEMONIC_BTS: c.bts(mem(), rcxw()); break;
            case ZYDIS_MNEMONIC_BTR: c.btr(mem(), rcxw()); break;
            default:                 c.btc(mem(), rcxw()); break;
            }
        } else if (ops[1].type == ZYDIS_OPERAND_TYPE_IMMEDIATE) {
            // Hardware masks the immediate to the operand size on both
            // sides; pass it through raw.
            const u8 bit = static_cast<u8>(ops[1].imm.value.u);
            c.db(0xF0);
            switch (m) {
            case ZYDIS_MNEMONIC_BTS: c.bts(mem(), bit); break;
            case ZYDIS_MNEMONIC_BTR: c.btr(mem(), bit); break;
            default:                 c.btc(mem(), bit); break;
            }
        } else {
            return false;
        }
        c.pushfq();
        c.pop(r8);
        EmitStoreArithFlags(c, r8);
        return true;
    }

    default:
        return false; // not covered here — caller falls back to the per-op handler
    }
}
// that bit, leaves other flags undefined per Intel SDM.
// =============================================================================

/// BT r/m, r — register-register form, 64-bit width.
/// Only the dst,src reg-reg form is implemented for now; the imm
/// form (`bt r64, imm8`) and the mem-dst form can be added when seen.
///
/// Implementation notes:
/// - We never let the host's `bt` set guest CF directly because BT
///   leaves OF/SF/ZF/AF/PF "undefined" — Intel allows arbitrary
///   values. To stay deterministic we compute CF explicitly and
///   leave the other guest flags unchanged.
/// - The bit index is masked to (opsize - 1) by host BT already
///   when src is a register operand, but Zydis-decoded BT may
///   present a 64-bit register holding a value > 63. The host
///   instruction also masks by opsize-1 in that case, so we mirror
///   the architectural semantics for free.
bool EmitBt(const ZydisDecodedInstruction& insn,
            const ZydisDecodedOperand* ops,
            Xbyak::CodeGenerator& c) {
    if (insn.operand_width != 64 && insn.operand_width != 32 &&
        insn.operand_width != 16) return false;
    if (ops[0].type != ZYDIS_OPERAND_TYPE_REGISTER) return false;
    const int dst_idx = ZydisGprToIndex(ops[0].reg.value);
    if (dst_idx < 0) return false;

    // Load the value to test. We operate at the host's 64-bit width but
    // mask the bit index to the operand size so out-of-width indices
    // wrap exactly as the guest's BT would (BT reg,reg masks the index
    // by operand-size-1; BT reg,imm uses imm mod operand-size).
    c.mov(rax, qword[r13 + GprOffset(dst_idx)]);

    const u32 width = insn.operand_width;
    const u64 idx_mask = width - 1;  // 63 / 31 / 15

    if (ops[1].type == ZYDIS_OPERAND_TYPE_IMMEDIATE) {
        const u64 bit = ops[1].imm.value.u & idx_mask;
        c.mov(rcx, bit);
        c.bt(rax, rcx);
    } else if (ops[1].type == ZYDIS_OPERAND_TYPE_REGISTER) {
        const int src_idx = ZydisGprToIndex(ops[1].reg.value);
        if (src_idx < 0) return false;
        c.mov(rcx, qword[r13 + GprOffset(src_idx)]);
        c.and_(rcx, idx_mask);     // mask to operand width
        c.bt(rax, rcx);
    } else {
        return false;
    }

    // Capture host CF into r11, then merge into guest rflags (CF only;
    // other flags are left as-is per the SDM's "undefined" allowance,
    // matching the prior behavior the tests lock in).
    c.setc(r11b);
    c.movzx(r11, r11b);
    c.mov(r10, qword[r13 + Offsets::Rflags]);
    c.btr(r10, 0);                                  // clear guest CF
    c.or_(r10, r11);                                // OR in new CF
    c.mov(qword[r13 + Offsets::Rflags], r10);
    return true;
}

/// MOVZX — read 8 or 16 bits from src (register or memory),
/// zero-extend to the destination's width, write to dst.
///
/// We always emit the zero-extension into rax (a 64-bit host
/// register). The store back to the guest dst slot is width-aware,
/// because x86-64 has *different* "what happens to the upper bits"
/// rules at each width when writing a register:
///
///   - 64-bit dst: the whole register is written (trivial).
///   - 32-bit dst: writing the low 32 zero-extends to 64 (the
///     "no surprise" rule of x86-64 32-bit writes). Since rax
///     already has zeros above the loaded source, storing the
///     full qword reproduces this — both observably yield
///     "low 32 = value, upper 32 = 0" in the guest slot.
///   - 16-bit dst: writing the low 16 **preserves** the upper
///     48 bits of the underlying 64-bit register. A qword store
///     would silently zero those upper 48 bits and corrupt
///     guest state — so we narrow the store to a word.
///
/// Compilers rarely emit `movzx r16, ...` in 64-bit code (it's
/// shorter to emit `movzx r32, ...`, which is also zero-extending
/// and gets the same low-16 result for free). But it's legal and
/// would be a silent miscompile if we got it wrong, so it's
/// worth handling correctly rather than gambling on absence.
bool EmitMovzx(const ZydisDecodedInstruction& insn,
               const ZydisDecodedOperand* ops,
               u64 next_rip,
               Xbyak::CodeGenerator& c) {
    if (ops[0].type != ZYDIS_OPERAND_TYPE_REGISTER) return false;
    const int dst_idx = ZydisGprToIndex(ops[0].reg.value);
    if (dst_idx < 0) return false;

    // Destination width must be one of 16/32/64. (Per the x86 spec,
    // those are the only legal MOVZX destinations.)
    const u32 dst_size = insn.operand_width;
    if (dst_size != 16 && dst_size != 32 && dst_size != 64) return false;

    // Source operand size in bits (8 or 16 for any legal MOVZX).
    const u32 src_size = ops[1].size;
    if (src_size != 8 && src_size != 16) return false;

    if (ops[1].type == ZYDIS_OPERAND_TYPE_REGISTER) {
        if (src_size == 8) {
            // 8-bit source may be a high-byte reg (AH/CH/DH/BH); use the byte
            // offset helper, which accounts for the +1 of high-byte regs.
            const int src_off = ZydisGpr8ToByteOffset(ops[1].reg.value);
            if (src_off < 0) return false;
            c.movzx(rax, byte[r13 + src_off]);
        } else {
            const int src_idx = ZydisGprToIndex(ops[1].reg.value);
            if (src_idx < 0) return false;
            c.movzx(rax, word[r13 + GprOffset(src_idx)]);
        }
    } else if (ops[1].type == ZYDIS_OPERAND_TYPE_MEMORY) {
        if (!EmitEffectiveAddress(ops[1].mem, next_rip, c)) return false;
        if (src_size == 8) {
            c.movzx(rax, byte[rdx]);
        } else {
            c.movzx(rax, word[rdx]);
        }
    } else {
        return false;
    }

    // Width-aware store back to the guest slot.
    if (dst_size == 16) {
        // Preserve upper 48 bits — write only the low word.
        c.mov(word[r13 + GprOffset(dst_idx)], ax);
    } else {
        // 32-bit dst: qword store works because rax is already
        // zero-extended; the upper 32 bits we write are zeros,
        // matching x86-64 32-bit-write semantics.
        // 64-bit dst: qword store is the whole register.
        c.mov(qword[r13 + GprOffset(dst_idx)], rax);
    }
    return true;
}

/// MOVSX r/m — move with sign-extension. Mirror of EmitMovzx, but
/// sign-extends the 8/16-bit source into the wider destination. Note:
/// MOVSX is a distinct mnemonic from MOVSXD (which sign-extends 32->64
/// and is handled by EmitMovsxd). MOVSX handles 8->16, 8->32, 8->64,
/// 16->32, and 16->64.
bool EmitMovsx(const ZydisDecodedInstruction& insn,
               const ZydisDecodedOperand* ops,
               u64 next_rip,
               Xbyak::CodeGenerator& c) {
    if (ops[0].type != ZYDIS_OPERAND_TYPE_REGISTER) return false;
    const int dst_idx = ZydisGprToIndex(ops[0].reg.value);
    if (dst_idx < 0) return false;

    const u32 dst_size = insn.operand_width;
    if (dst_size != 16 && dst_size != 32 && dst_size != 64) return false;

    const u32 src_size = ops[1].size;
    if (src_size != 8 && src_size != 16) return false;

    // Sign-extend the source into a host register matching the
    // destination width, so the high bits carry the correct sign.
    // For a 16-bit destination we must produce a 16-bit result while
    // preserving the guest register's upper 48 bits, so we extend into
    // ax. For 32/64-bit destinations we extend into eax/rax — eax
    // zero-extends rax (the sign already lives in bit 31 of eax for the
    // 32-bit case, which is exactly what a 32-bit write stores), and
    // rax holds the full 64-bit sign-extended value for the 64-bit case.
    if (ops[1].type == ZYDIS_OPERAND_TYPE_REGISTER) {
        if (src_size == 8) {
            // 8-bit source may be a high-byte reg (AH/CH/DH/BH).
            const int src_off = ZydisGpr8ToByteOffset(ops[1].reg.value);
            if (src_off < 0) return false;
            if (dst_size == 16)      c.movsx(ax,  byte[r13 + src_off]);
            else if (dst_size == 32) c.movsx(eax, byte[r13 + src_off]);
            else                     c.movsx(rax, byte[r13 + src_off]);
        } else {  // src_size == 16
            const int src_idx = ZydisGprToIndex(ops[1].reg.value);
            if (src_idx < 0) return false;
            if (dst_size == 32)      c.movsx(eax, word[r13 + GprOffset(src_idx)]);
            else                     c.movsx(rax, word[r13 + GprOffset(src_idx)]);
        }
    } else if (ops[1].type == ZYDIS_OPERAND_TYPE_MEMORY) {
        if (!EmitEffectiveAddress(ops[1].mem, next_rip, c)) return false;
        if (src_size == 8) {
            if (dst_size == 16)      c.movsx(ax,  byte[rdx]);
            else if (dst_size == 32) c.movsx(eax, byte[rdx]);
            else                     c.movsx(rax, byte[rdx]);
        } else {  // src_size == 16
            if (dst_size == 32)      c.movsx(eax, word[rdx]);
            else                     c.movsx(rax, word[rdx]);
        }
    } else {
        return false;
    }

    // Width-aware store back. 16-bit dst preserves upper 48 bits;
    // 32-bit dst zero-extends (the qword store writes the zero-extended
    // rax from the 32-bit movsx); 64-bit dst stores the full register.
    if (dst_size == 16) {
        c.mov(word[r13 + GprOffset(dst_idx)], ax);
    } else {
        c.mov(qword[r13 + GprOffset(dst_idx)], rax);
    }
    return true;
}
//
// Same 16 condition codes as Jcc. We reuse EmitJccCondition to
// compute a 0/1 indicator in rcx, then conditionally pick between
// the existing dst value and the src value based on that indicator.
//
// Uses host CMOV on the 0/1 indicator's zero-flag for the selection.
// =============================================================================

// Forward declaration: EmitJccCondition is defined later in this
// file alongside EmitJcc; EmitCmov uses it for condition encoding.
bool EmitJccCondition(ZydisMnemonic mnemonic, Xbyak::CodeGenerator& c);

/// Map a CMOVcc mnemonic to the corresponding Jcc mnemonic. The
/// condition encoding is identical between the two families; we
/// share the condition-computation code.
ZydisMnemonic CmovToJcc(ZydisMnemonic m) {
    switch (m) {
        case ZYDIS_MNEMONIC_CMOVZ:   return ZYDIS_MNEMONIC_JZ;
        case ZYDIS_MNEMONIC_CMOVNZ:  return ZYDIS_MNEMONIC_JNZ;
        case ZYDIS_MNEMONIC_CMOVS:   return ZYDIS_MNEMONIC_JS;
        case ZYDIS_MNEMONIC_CMOVNS:  return ZYDIS_MNEMONIC_JNS;
        case ZYDIS_MNEMONIC_CMOVO:   return ZYDIS_MNEMONIC_JO;
        case ZYDIS_MNEMONIC_CMOVNO:  return ZYDIS_MNEMONIC_JNO;
        case ZYDIS_MNEMONIC_CMOVP:   return ZYDIS_MNEMONIC_JP;
        case ZYDIS_MNEMONIC_CMOVNP:  return ZYDIS_MNEMONIC_JNP;
        case ZYDIS_MNEMONIC_CMOVB:   return ZYDIS_MNEMONIC_JB;
        case ZYDIS_MNEMONIC_CMOVNB:  return ZYDIS_MNEMONIC_JNB;
        case ZYDIS_MNEMONIC_CMOVBE:  return ZYDIS_MNEMONIC_JBE;
        case ZYDIS_MNEMONIC_CMOVNBE: return ZYDIS_MNEMONIC_JNBE;
        case ZYDIS_MNEMONIC_CMOVL:   return ZYDIS_MNEMONIC_JL;
        case ZYDIS_MNEMONIC_CMOVNL:  return ZYDIS_MNEMONIC_JNL;
        case ZYDIS_MNEMONIC_CMOVLE:  return ZYDIS_MNEMONIC_JLE;
        case ZYDIS_MNEMONIC_CMOVNLE: return ZYDIS_MNEMONIC_JNLE;
        default: return ZYDIS_MNEMONIC_INVALID;
    }
}

/// CMOVcc r64, r/m64 — if condition true, dst = src.
/// 64-bit only for now; 32-bit CMOV would follow the same shape.
bool EmitCmov(const ZydisDecodedInstruction& insn,
              const ZydisDecodedOperand* ops,
              u64 next_rip,
              Xbyak::CodeGenerator& c) {
    if (insn.operand_width != 64 && insn.operand_width != 32 &&
        insn.operand_width != 16) return false;
    if (ops[0].type != ZYDIS_OPERAND_TYPE_REGISTER) return false;
    const int dst_idx = ZydisGprToIndex(ops[0].reg.value);
    if (dst_idx < 0) return false;

    const ZydisMnemonic jcc_equiv = CmovToJcc(insn.mnemonic);
    if (jcc_equiv == ZYDIS_MNEMONIC_INVALID) return false;

    // Load src into r8 first — we'll need rax/rcx/rdx for the
    // condition computation.
    if (ops[1].type == ZYDIS_OPERAND_TYPE_REGISTER) {
        const int src_idx = ZydisGprToIndex(ops[1].reg.value);
        if (src_idx < 0) return false;
        c.mov(r8, qword[r13 + GprOffset(src_idx)]);
    } else if (ops[1].type == ZYDIS_OPERAND_TYPE_MEMORY) {
        if (!EmitEffectiveAddress(ops[1].mem, next_rip, c)) return false;
        c.mov(r8, qword[rdx]);
    } else {
        return false;
    }

    // Load current dst value into r9 (the candidate "no-change" result).
    c.mov(r9, qword[r13 + GprOffset(dst_idx)]);

    // Compute condition into rcx (0 or 1).
    if (!EmitJccCondition(jcc_equiv, c)) return false;

    // host_test sets ZF on the indicator; cmovnz picks r8 (src) when
    // the condition was true (rcx != 0), otherwise keeps r9 (old dst).
    // The conditional move is performed at the operand's width so the
    // x86-64 width rules apply to the writeback: 64-bit keeps the full
    // value; 32-bit always zero-extends bits 63:32 (taken or not); and
    // 16-bit writes only the low word, preserving bits 63:16.
    c.test(rcx, rcx);
    if (insn.operand_width == 64) {
        c.cmovnz(r9, r8);
        c.mov(qword[r13 + GprOffset(dst_idx)], r9);
    } else if (insn.operand_width == 32) {
        c.cmovnz(r9d, r8d);  // 32-bit cmov zero-extends r9
        c.mov(qword[r13 + GprOffset(dst_idx)], r9);
    } else {  // 16-bit
        // 16-bit CMOV writes only the low 16 bits of the register,
        // preserving bits 63:16 — whether or not the move is taken.
        // cmovnz r9w, r8w merges r8's low word into r9's low word and
        // leaves r9's upper 48 bits intact, so a word-sized store back
        // gives the correct merge.
        c.cmovnz(r9w, r8w);
        c.mov(word[r13 + GprOffset(dst_idx)], r9w);
    }
    return true;
}

// =============================================================================
// SHL / SHR / SAR — shift instructions.
//
// Shifts have a subtle semantic that's worth getting right: per
// Intel's spec, *if the shift count is zero, no flags are affected*.
// This applies to both immediate-zero and "CL=0 at runtime".
// Computing this explicitly with a runtime branch would be ugly.
// Instead, we round-trip rflags through the host CPU:
//
//   1. Load guest rflags into host rflags (push + popfq).
//   2. Execute the host shift on a host scratch register. The host
//      CPU implements the same "shift-by-zero preserves flags" rule
//      that the guest expects, so flags either get updated or stay
//      put — matching guest semantics in both cases.
//   3. Capture host rflags back into guest rflags (pushfq + pop).
//
// The shift count is masked to 6 bits (for 64-bit operands) by the
// host CPU automatically, matching guest behavior. We don't need
// to mask explicitly.
//
// Limitations: only 64-bit operand width and only register
// destinations. 32-bit shifts and memory destinations follow the
// same pattern and can be added on demand.
// =============================================================================

enum class ShiftKind { Shl, Shr, Sar };

/// Common 64-bit shift emitter, parameterised by which host shift
/// to use. Source operand 0 = destination register; operand 1 =
/// either an 8-bit immediate or the CL register.
bool EmitShift64(const ZydisDecodedInstruction& insn,
                 const ZydisDecodedOperand* ops,
                 u64 next_rip,
                 Xbyak::CodeGenerator& c,
                 ShiftKind kind) {
    if (insn.operand_width != 64) return false;
    const bool dst_mem = (ops[0].type == ZYDIS_OPERAND_TYPE_MEMORY);
    int dst_idx = 0;
    if (dst_mem) {
        if (!EmitEffectiveAddress(ops[0].mem, next_rip, c)) return false;  // -> rdx
        c.mov(r8, rdx);                            // stable value address
    } else {
        if (ops[0].type != ZYDIS_OPERAND_TYPE_REGISTER) return false;
        dst_idx = ZydisGprToIndex(ops[0].reg.value);
        if (dst_idx < 0) return false;
    }

    // Load shift count into host cl.
    if (ops[1].type == ZYDIS_OPERAND_TYPE_IMMEDIATE) {
        // Zydis presents the imm value as unsigned u64; only low
        // 6 bits matter for 64-bit shifts, but the host CPU masks
        // anyway. Use a byte move for clarity.
        c.mov(cl, static_cast<u8>(ops[1].imm.value.u & 0xFF));
    } else if (ops[1].type == ZYDIS_OPERAND_TYPE_REGISTER) {
        // The only legal shift-count register is CL itself. (BL/DL
        // etc. are not allowed by the x86 ISA in this slot.)
        if (ops[1].reg.value != ZYDIS_REGISTER_CL) return false;
        // Load guest CL = low byte of guest rcx (index 1).
        c.mov(cl, byte[r13 + GprOffset(1)]);
    } else {
        return false;
    }

    // Load destination value into rax (register slot or memory).
    if (dst_mem) c.mov(rax, qword[r8]);
    else         c.mov(rax, qword[r13 + GprOffset(dst_idx)]);

    // Round-trip rflags: load guest → host. Use rdx (not in use yet
    // and not aliased to cl).
    c.mov(rdx, qword[r13 + Offsets::Rflags]);
    c.and_(rdx, RflagsBits::HostLoadMask);
    c.push(rdx);
    c.popfq();

    // Perform the shift. Host x86 shifts consume the count from cl
    // implicitly when given a register operand.
    switch (kind) {
        case ShiftKind::Shl: c.shl(rax, cl); break;
        case ShiftKind::Shr: c.shr(rax, cl); break;
        case ShiftKind::Sar: c.sar(rax, cl); break;
    }

    // Capture host → guest.
    c.pushfq();
    c.pop(rdx);
    EmitStoreArithFlags(c, rdx);

    // Store the shifted value back (register slot or memory).
    if (dst_mem) c.mov(qword[r8], rax);
    else         c.mov(qword[r13 + GprOffset(dst_idx)], rax);
    return true;
}

/// 32-bit shift emitter. Same shape as EmitShift64 but operates at
/// 32-bit width, which matters for two reasons:
///   - The host shift opcode reads count from CL but masks it with
///     0x1F (vs 0x3F for the 64-bit form). The host CPU handles
///     this; we don't need to mask ourselves.
///   - A 32-bit write into eax zero-extends rax. We store the full
///     qword back so the guest GPR slot inherits the zero-extension
///     — matching how the guest's host CPU would have updated the
///     register.
/// Flags: SF is correctly captured because EmitShift32 lets the host
/// CPU compute it at 32-bit width via the rflags round-trip — bit 31
/// of the result lands in host SF naturally. (This is the same
/// reason the 32-bit narrow-arith path is SF-correct while the
/// EmitFlagsFromBitwise wide-path is not.)
bool EmitShift32(const ZydisDecodedInstruction& insn,
                 const ZydisDecodedOperand* ops,
                 u64 next_rip,
                 Xbyak::CodeGenerator& c,
                 ShiftKind kind) {
    if (insn.operand_width != 32) return false;
    // Destination: register or memory. For memory, compute the EA FIRST into a
    // stable host reg (r8) so the flag round-trip / shift can't clobber it; the
    // same address is reused for the read and write-back.
    const bool dst_mem = (ops[0].type == ZYDIS_OPERAND_TYPE_MEMORY);
    int dst_idx = 0;
    if (dst_mem) {
        if (!EmitEffectiveAddress(ops[0].mem, next_rip, c)) return false;  // -> rdx
        c.mov(r8, rdx);
    } else {
        if (ops[0].type != ZYDIS_OPERAND_TYPE_REGISTER) return false;
        dst_idx = ZydisGprToIndex(ops[0].reg.value);
        if (dst_idx < 0) return false;
    }

    if (ops[1].type == ZYDIS_OPERAND_TYPE_IMMEDIATE) {
        c.mov(cl, static_cast<u8>(ops[1].imm.value.u & 0xFF));
    } else if (ops[1].type == ZYDIS_OPERAND_TYPE_REGISTER) {
        if (ops[1].reg.value != ZYDIS_REGISTER_CL) return false;
        c.mov(cl, byte[r13 + GprOffset(1)]);
    } else {
        return false;
    }

    if (dst_mem) c.mov(eax, dword[r8]);
    else         c.mov(eax, dword[r13 + GprOffset(dst_idx)]);

    c.mov(rdx, qword[r13 + Offsets::Rflags]);
    c.and_(rdx, RflagsBits::HostLoadMask);
    c.push(rdx);
    c.popfq();

    switch (kind) {
        case ShiftKind::Shl: c.shl(eax, cl); break;
        case ShiftKind::Shr: c.shr(eax, cl); break;
        case ShiftKind::Sar: c.sar(eax, cl); break;
    }

    c.pushfq();
    c.pop(rdx);
    EmitStoreArithFlags(c, rdx);

    if (dst_mem) {
        c.mov(dword[r8], eax);   // 32-bit memory store; adjacent bytes untouched
    } else {
        // Full-qword store so the host's 32-bit zero-extension is
        // recorded in the guest GPR slot.
        c.mov(qword[r13 + GprOffset(dst_idx)], rax);
    }
    return true;
}

/// 8/16-bit shift emitter. Same round-trip-flags shape as the wider
/// forms, but narrow writes PRESERVE the upper bits of the guest GPR
/// (an 8-bit write keeps bits 63:8; a 16-bit write keeps bits 63:16),
/// so we read/operate/store at the operand's native width. The host
/// CPU masks the count to 5 bits for both 8- and 16-bit shifts and
/// computes the narrow-width flags (including SF from the top bit of
/// the 8/16-bit result) through the rflags round-trip.
bool EmitShiftNarrow(const ZydisDecodedInstruction& insn,
                 const ZydisDecodedOperand* ops,
                 u64 next_rip,
                 Xbyak::CodeGenerator& c,
                 ShiftKind kind) {
    if (insn.operand_width != 8 && insn.operand_width != 16) return false;
    const bool dst_mem = (ops[0].type == ZYDIS_OPERAND_TYPE_MEMORY);
    int dst_idx = 0;
    if (dst_mem) {
        if (!EmitEffectiveAddress(ops[0].mem, next_rip, c)) return false;  // -> rdx
        c.mov(r8, rdx);
    } else {
        if (ops[0].type != ZYDIS_OPERAND_TYPE_REGISTER) return false;
        dst_idx = ZydisGprToIndex(ops[0].reg.value);
        if (dst_idx < 0) return false;
    }

    if (ops[1].type == ZYDIS_OPERAND_TYPE_IMMEDIATE) {
        c.mov(cl, static_cast<u8>(ops[1].imm.value.u & 0xFF));
    } else if (ops[1].type == ZYDIS_OPERAND_TYPE_REGISTER) {
        if (ops[1].reg.value != ZYDIS_REGISTER_CL) return false;
        c.mov(cl, byte[r13 + GprOffset(1)]);
    } else {
        return false;
    }

    const bool is8 = (insn.operand_width == 8);
    if (dst_mem) {
        if (is8) c.mov(al, byte[r8]);
        else     c.mov(ax, word[r8]);
    } else {
        if (is8) c.mov(al, byte[r13 + GprOffset(dst_idx)]);
        else     c.mov(ax, word[r13 + GprOffset(dst_idx)]);
    }

    c.mov(rdx, qword[r13 + Offsets::Rflags]);
    c.and_(rdx, RflagsBits::HostLoadMask);
    c.push(rdx);
    c.popfq();

    switch (kind) {
        case ShiftKind::Shl: if (is8) c.shl(al, cl); else c.shl(ax, cl); break;
        case ShiftKind::Shr: if (is8) c.shr(al, cl); else c.shr(ax, cl); break;
        case ShiftKind::Sar: if (is8) c.sar(al, cl); else c.sar(ax, cl); break;
    }

    c.pushfq();
    c.pop(rdx);
    EmitStoreArithFlags(c, rdx);

    // Narrow store preserves surrounding bytes (memory) or the upper bits of
    // the guest GPR slot (register).
    if (dst_mem) {
        if (is8) c.mov(byte[r8], al);
        else     c.mov(word[r8], ax);
    } else {
        if (is8) c.mov(byte[r13 + GprOffset(dst_idx)], al);
        else     c.mov(word[r13 + GprOffset(dst_idx)], ax);
    }
    return true;
}
//
// Bit-rotation flavours of the shift family. Same encoding shape
// as SHL/SHR/SAR (count from imm8 or CL), same round-trip-flags
// technique. The only differences vs shifts:
//
//   - The host opcode is `rol` / `ror` instead of `shl` etc.
//   - Rotates set CF (= the bit rotated through) and OF (for
//     1-bit rotates only), but unlike shifts they do NOT modify
//     ZF/SF/PF. The popfq round-trip naturally preserves those
//     flags through the host CPU's rotate, so we get this for free.
//   - Like shifts, rotate-by-zero affects no flags at all.
// =============================================================================

enum class RotateKind { Rol, Ror };

bool EmitRotate64(const ZydisDecodedInstruction& insn,
                  const ZydisDecodedOperand* ops,
                  u64 next_rip,
                  Xbyak::CodeGenerator& c,
                  RotateKind kind) {
    const u32 w = insn.operand_width;
    if (w != 8 && w != 16 && w != 32 && w != 64) return false;

    // Destination: register or memory. For memory, compute the EA FIRST into a
    // stable host reg (r8) so the flag round-trip can't clobber it; the same
    // address serves the read and the write-back.
    const bool dst_mem = (ops[0].type == ZYDIS_OPERAND_TYPE_MEMORY);
    int dst_idx = 0;
    if (dst_mem) {
        if (!EmitEffectiveAddress(ops[0].mem, next_rip, c)) return false;  // -> rdx
        c.mov(r8, rdx);
    } else {
        if (ops[0].type != ZYDIS_OPERAND_TYPE_REGISTER) return false;
        dst_idx = ZydisGprToIndex(ops[0].reg.value);
        if (dst_idx < 0) return false;
    }

    // Shift count from imm or guest CL — identical to EmitShift64.
    if (ops[1].type == ZYDIS_OPERAND_TYPE_IMMEDIATE) {
        c.mov(cl, static_cast<u8>(ops[1].imm.value.u & 0xFF));
    } else if (ops[1].type == ZYDIS_OPERAND_TYPE_REGISTER) {
        if (ops[1].reg.value != ZYDIS_REGISTER_CL) return false;
        c.mov(cl, byte[r13 + GprOffset(1)]);
    } else {
        return false;
    }

    // Load the destination value into rax at the operand width.
    if (dst_mem) {
        if (w == 8)       c.mov(al, byte[r8]);
        else if (w == 16) c.mov(ax, word[r8]);
        else if (w == 32) c.mov(eax, dword[r8]);
        else              c.mov(rax, qword[r8]);
    } else {
        if (w == 8)       c.mov(al, byte[r13 + GprOffset(dst_idx)]);
        else if (w == 16) c.mov(ax, word[r13 + GprOffset(dst_idx)]);
        else if (w == 32) c.mov(eax, dword[r13 + GprOffset(dst_idx)]);
        else              c.mov(rax, qword[r13 + GprOffset(dst_idx)]);
    }

    // Round-trip flags via host.
    c.mov(rdx, qword[r13 + Offsets::Rflags]);
    c.and_(rdx, RflagsBits::HostLoadMask);
    c.push(rdx);
    c.popfq();

    switch (w) {
        case 8:  if (kind == RotateKind::Rol) c.rol(al, cl);  else c.ror(al, cl);  break;
        case 16: if (kind == RotateKind::Rol) c.rol(ax, cl);  else c.ror(ax, cl);  break;
        case 32: if (kind == RotateKind::Rol) c.rol(eax, cl); else c.ror(eax, cl); break;  // zero-extends rax
        default: if (kind == RotateKind::Rol) c.rol(rax, cl); else c.ror(rax, cl); break;
    }

    c.pushfq();
    c.pop(rdx);
    EmitStoreArithFlags(c, rdx);

    // Store back. Memory: width-sized store (adjacent bytes untouched).
    // Register: 8/16 preserve the upper GPR bits; 32/64 store the full qword
    // (the 32-bit rotate already zero-extended rax).
    if (dst_mem) {
        if (w == 8)       c.mov(byte[r8], al);
        else if (w == 16) c.mov(word[r8], ax);
        else if (w == 32) c.mov(dword[r8], eax);
        else              c.mov(qword[r8], rax);
    } else {
        if (w == 8)       c.mov(byte[r13 + GprOffset(dst_idx)], al);
        else if (w == 16) c.mov(word[r13 + GprOffset(dst_idx)], ax);
        else              c.mov(qword[r13 + GprOffset(dst_idx)], rax);
    }
    return true;
}

// =============================================================================
// IMUL — three encoding forms, dispatched by Zydis's visible
// operand count.
//
//   1-op (operand_count_visible == 1):
//     IMUL r/m64  →  RDX:RAX = sign_extend(RAX) * sign_extend(src)
//     Full-precision 128-bit signed multiply. CF/OF set if the
//     upper half is significant (i.e. the result doesn't fit in
//     64 bits). ZF/SF/PF/AF are *undefined* per Intel.
//
//   2-op (operand_count_visible == 2):
//     IMUL r64, r/m64  →  dst = (dst * src) truncated to 64 bits
//     The common C `*` operator path. CF/OF set if the truncation
//     dropped significant bits (i.e. the signed product doesn't
//     fit in 64 bits).
//
//   3-op (operand_count_visible == 3):
//     IMUL r64, r/m64, imm  →  dst = (src * imm) truncated to 64
//     Compiler emits this for `x * constant` when the constant
//     fits in imm32 (sign-extended).
//
// All three use the host IMUL opcode and round-trip flags so
// CF/OF come out correct.
// =============================================================================

/// 1-op IMUL: rdx:rax = rax * src.
bool EmitImul1Op(const ZydisDecodedInstruction& insn,
                 const ZydisDecodedOperand* ops,
                 u64 next_rip,
                 Xbyak::CodeGenerator& c) {
    if (insn.operand_width != 64) return false;

    // Load src into rcx. For memory operands, EmitEffectiveAddress
    // writes rdx (the address) and clobbers rax — so we must
    // dereference the address into rcx *before* loading rax.
    if (ops[0].type == ZYDIS_OPERAND_TYPE_REGISTER) {
        const int src_idx = ZydisGprToIndex(ops[0].reg.value);
        if (src_idx < 0) return false;
        c.mov(rcx, qword[r13 + GprOffset(src_idx)]);
    } else if (ops[0].type == ZYDIS_OPERAND_TYPE_MEMORY) {
        if (!EmitEffectiveAddress(ops[0].mem, next_rip, c)) return false;
        c.mov(rcx, qword[rdx]);
    } else {
        return false;
    }

    // Load guest RAX into host rax.
    c.mov(rax, qword[r13 + GprOffset(0)]);

    // Flag round-trip.
    c.mov(r8, qword[r13 + Offsets::Rflags]);
    c.and_(r8, RflagsBits::HostLoadMask);
    c.push(r8);
    c.popfq();

    c.imul(rcx);  // implicit rax operand; rdx:rax = rax * rcx

    c.pushfq();
    c.pop(r8);
    EmitStoreArithFlags(c, r8);

    // Write both halves of the result.
    c.mov(qword[r13 + GprOffset(0)], rax);  // low → RAX
    c.mov(qword[r13 + GprOffset(2)], rdx);  // high → RDX
    return true;
}

/// 2-op IMUL: dst = dst * src (low 64 bits).
bool EmitImul2Op(const ZydisDecodedInstruction& insn,
                 const ZydisDecodedOperand* ops,
                 u64 next_rip,
                 Xbyak::CodeGenerator& c) {
    if (insn.operand_width != 64 && insn.operand_width != 32) return false;
    if (ops[0].type != ZYDIS_OPERAND_TYPE_REGISTER) return false;
    const int dst_idx = ZydisGprToIndex(ops[0].reg.value);
    if (dst_idx < 0) return false;

    // Load src into rcx (handle memory case first to avoid clobbering rax).
    if (ops[1].type == ZYDIS_OPERAND_TYPE_REGISTER) {
        const int src_idx = ZydisGprToIndex(ops[1].reg.value);
        if (src_idx < 0) return false;
        c.mov(rcx, qword[r13 + GprOffset(src_idx)]);
    } else if (ops[1].type == ZYDIS_OPERAND_TYPE_MEMORY) {
        if (!EmitEffectiveAddress(ops[1].mem, next_rip, c)) return false;
        c.mov(rcx, qword[rdx]);
    } else {
        return false;
    }

    c.mov(rax, qword[r13 + GprOffset(dst_idx)]);

    c.mov(r8, qword[r13 + Offsets::Rflags]);
    c.and_(r8, RflagsBits::HostLoadMask);
    c.push(r8);
    c.popfq();

    // 64-bit IMUL or 32-bit IMUL (the latter zero-extends rax via
    // a 32-bit write into eax). Flags differ slightly (host computes
    // OF/CF based on whether the high half of the product is a
    // sign-extension of the low half — same rule both widths,
    // just at different widths), and the rflags round-trip captures
    // whichever variant the host produced.
    if (insn.operand_width == 64) {
        c.imul(rax, rcx);
    } else {
        c.imul(eax, ecx);
    }

    c.pushfq();
    c.pop(r8);
    EmitStoreArithFlags(c, r8);

    // Full-qword writeback. For the 32-bit form the 32-bit IMUL
    // already zero-extended rax above, so this stores the canonical
    // zero-extended value into the guest GPR slot.
    c.mov(qword[r13 + GprOffset(dst_idx)], rax);
    return true;
}

/// 3-op IMUL: dst = src * imm.
bool EmitImul3Op(const ZydisDecodedInstruction& insn,
                 const ZydisDecodedOperand* ops,
                 u64 next_rip,
                 Xbyak::CodeGenerator& c) {
    if (insn.operand_width != 64 && insn.operand_width != 32) return false;
    if (ops[0].type != ZYDIS_OPERAND_TYPE_REGISTER) return false;
    if (ops[2].type != ZYDIS_OPERAND_TYPE_IMMEDIATE) return false;
    const int dst_idx = ZydisGprToIndex(ops[0].reg.value);
    if (dst_idx < 0) return false;

    // Load src into rcx.
    if (ops[1].type == ZYDIS_OPERAND_TYPE_REGISTER) {
        const int src_idx = ZydisGprToIndex(ops[1].reg.value);
        if (src_idx < 0) return false;
        c.mov(rcx, qword[r13 + GprOffset(src_idx)]);
    } else if (ops[1].type == ZYDIS_OPERAND_TYPE_MEMORY) {
        if (!EmitEffectiveAddress(ops[1].mem, next_rip, c)) return false;
        c.mov(rcx, qword[rdx]);
    } else {
        return false;
    }

    // Zydis returns the sign-extended 64-bit value in imm.value.s
    // regardless of whether the encoded immediate was imm8 or imm32.
    const s64 imm_val = ops[2].imm.value.s;
    c.mov(rax, imm_val);  // xbyak picks 32-bit immediate form when possible

    c.mov(r8, qword[r13 + Offsets::Rflags]);
    c.and_(r8, RflagsBits::HostLoadMask);
    c.push(r8);
    c.popfq();

    // 64-bit or 32-bit multiply. The 32-bit form (imul eax, ecx)
    // zero-extends rax per the x86-64 32-bit-write rule, so the
    // full-qword writeback below stores the canonical value.
    if (insn.operand_width == 64) {
        c.imul(rax, rcx);  // rax = imm * src
    } else {
        c.imul(eax, ecx);  // eax = imm * src, zero-extends rax
    }

    c.pushfq();
    c.pop(r8);
    EmitStoreArithFlags(c, r8);

    c.mov(qword[r13 + GprOffset(dst_idx)], rax);
    return true;
}

/// Top-level IMUL dispatcher. Routes by Zydis's visible-operand
/// count, which is reliable: 1/2/3 maps cleanly to the three IMUL
/// encoding families.
bool EmitImul(const ZydisDecodedInstruction& insn,
              const ZydisDecodedOperand* ops,
              u64 next_rip,
              Xbyak::CodeGenerator& c) {
    switch (insn.operand_count_visible) {
        case 1: return EmitImul1Op(insn, ops, next_rip, c);
        case 2: return EmitImul2Op(insn, ops, next_rip, c);
        case 3: return EmitImul3Op(insn, ops, next_rip, c);
        default: return false;
    }
}

/// MUL — unsigned multiply (F7 /4, F6 /4). One operand; the implicit
/// other operand and the result destination are width-dependent:
///   8:  AX        = AL  * r/m8
///   16: DX:AX     = AX  * r/m16
///   32: EDX:EAX   = EAX * r/m32   (writes zero-extend to RDX:RAX)
///   64: RDX:RAX   = RAX * r/m64
/// CF/OF set iff the high half is nonzero; other arithmetic flags are
/// undefined (we let the host set them and copy through). The host MUL
/// mirrors guest MUL exactly, so we run it on staged registers.
bool EmitMul(const ZydisDecodedInstruction& insn,
             const ZydisDecodedOperand* ops,
             u64 next_rip,
             Xbyak::CodeGenerator& c) {
    const u32 w = insn.operand_width;
    if (w != 8 && w != 16 && w != 32 && w != 64) return false;

    // Source into rcx (deref memory before touching rax).
    if (ops[0].type == ZYDIS_OPERAND_TYPE_REGISTER) {
        const int src_idx = ZydisGprToIndex(ops[0].reg.value);
        if (src_idx < 0) return false;
        c.mov(rcx, qword[r13 + GprOffset(src_idx)]);
    } else if (ops[0].type == ZYDIS_OPERAND_TYPE_MEMORY) {
        if (!EmitEffectiveAddress(ops[0].mem, next_rip, c)) return false;
        c.mov(rcx, qword[rdx]);
    } else {
        return false;
    }

    // Load the implicit accumulator (guest RAX) into host rax.
    c.mov(rax, qword[r13 + GprOffset(0)]);

    // Flag round-trip around the host op.
    c.mov(r8, qword[r13 + Offsets::Rflags]);
    c.and_(r8, RflagsBits::HostLoadMask);
    c.push(r8);
    c.popfq();

    switch (w) {
        case 8:  c.mul(cl);  break;   // AX = AL * cl
        case 16: c.mul(cx);  break;   // DX:AX
        case 32: c.mul(ecx); break;   // EDX:EAX (zero-extends)
        case 64: c.mul(rcx); break;   // RDX:RAX
    }

    c.pushfq();
    c.pop(r8);
    EmitStoreArithFlags(c, r8);

    // Write results per width.
    if (w == 8) {
        // AX holds the 16-bit product; write low 16 of RAX, preserve upper.
        c.mov(rdx, qword[r13 + GprOffset(0)]);
        c.mov(dx, ax);
        c.mov(qword[r13 + GprOffset(0)], rdx);
    } else if (w == 16) {
        // DX:AX — merge low 16 of each into guest RAX/RDX preserving upper.
        c.mov(r9, qword[r13 + GprOffset(0)]);
        c.mov(r9w, ax);
        c.mov(qword[r13 + GprOffset(0)], r9);
        c.mov(r9, qword[r13 + GprOffset(2)]);
        c.mov(r9w, dx);
        c.mov(qword[r13 + GprOffset(2)], r9);
    } else if (w == 32) {
        // EDX:EAX zero-extend to 64.
        c.mov(r9, rax);
        c.and_(r9, 0xFFFFFFFF);
        c.mov(qword[r13 + GprOffset(0)], r9);
        c.mov(r9, rdx);
        c.and_(r9, 0xFFFFFFFF);
        c.mov(qword[r13 + GprOffset(2)], r9);
    } else {
        c.mov(qword[r13 + GprOffset(0)], rax);
        c.mov(qword[r13 + GprOffset(2)], rdx);
    }
    return true;
}

// =============================================================================
// Sign-extension family: CWDE / CDQE / CDQ / CQO.
//
// (Inserted just above: BLSI, CLD/STD, PREFETCH*, XGETBV, VPMOVZXDQ.)
//
// BLSI dst, src — BMI1: dst = src & (-src), isolating the lowest set
// bit. CF = (src != 0); ZF/SF from the result; OF cleared. 32/64-bit,
// reg or mem source, zero-extends on 32-bit.
bool EmitBlsi(const ZydisDecodedInstruction& insn,
              const ZydisDecodedOperand* ops,
              u64 next_rip,
              Xbyak::CodeGenerator& c) {
    if (insn.mnemonic != ZYDIS_MNEMONIC_BLSI) return false;
    const u32 w = insn.operand_width;
    if (w != 32 && w != 64) return false;
    if (ops[0].type != ZYDIS_OPERAND_TYPE_REGISTER) return false;
    const int dst_idx = ZydisGprToIndex(ops[0].reg.value);
    if (dst_idx < 0) return false;

    // src into rax.
    if (ops[1].type == ZYDIS_OPERAND_TYPE_REGISTER) {
        const int s = ZydisGprToIndex(ops[1].reg.value);
        if (s < 0) return false;
        c.mov(rax, qword[r13 + GprOffset(s)]);
    } else if (ops[1].type == ZYDIS_OPERAND_TYPE_MEMORY) {
        if (!EmitEffectiveAddress(ops[1].mem, next_rip, c)) return false;
        c.mov(rax, qword[rdx]);
    } else {
        return false;
    }

    // Host BLSI via the same encoding: use blsi if available; otherwise
    // compute (src & -src) and set flags. xbyak provides blsi.
    if (w == 32) {
        c.blsi(eax, eax);                 // zero-extends rax
    } else {
        c.blsi(rax, rax);
    }
    // Capture host flags (CF/ZF/SF/OF set per BLSI spec).
    c.pushfq();
    c.pop(r8);
    EmitStoreArithFlags(c, r8);
    c.mov(qword[r13 + GprOffset(dst_idx)], rax);
    return true;
}

// BLSR dst, src — BMI1: dst = src & (src - 1), resetting the lowest set
// bit. CF = (src != 0); ZF/SF from the result; OF cleared. 32/64-bit,
// reg or mem source, zero-extends on 32-bit. Same shape as BLSI; the host
// BLSR instruction implements the whole semantics including flags.
bool EmitBlsr(const ZydisDecodedInstruction& insn,
              const ZydisDecodedOperand* ops,
              u64 next_rip,
              Xbyak::CodeGenerator& c) {
    if (insn.mnemonic != ZYDIS_MNEMONIC_BLSR) return false;
    const u32 w = insn.operand_width;
    if (w != 32 && w != 64) return false;
    if (ops[0].type != ZYDIS_OPERAND_TYPE_REGISTER) return false;
    const int dst_idx = ZydisGprToIndex(ops[0].reg.value);
    if (dst_idx < 0) return false;

    // src into rax.
    if (ops[1].type == ZYDIS_OPERAND_TYPE_REGISTER) {
        const int s = ZydisGprToIndex(ops[1].reg.value);
        if (s < 0) return false;
        c.mov(rax, qword[r13 + GprOffset(s)]);
    } else if (ops[1].type == ZYDIS_OPERAND_TYPE_MEMORY) {
        if (!EmitEffectiveAddress(ops[1].mem, next_rip, c)) return false;
        c.mov(rax, qword[rdx]);
    } else {
        return false;
    }

    if (w == 32) {
        c.blsr(eax, eax);                 // zero-extends rax
    } else {
        c.blsr(rax, rax);
    }
    // Capture host flags (CF/ZF/SF/OF set per BLSR spec).
    c.pushfq();
    c.pop(r8);
    EmitStoreArithFlags(c, r8);
    c.mov(qword[r13 + GprOffset(dst_idx)], rax);
    return true;
}
// flags preserved.
bool EmitCldStd(const ZydisDecodedInstruction& insn, Xbyak::CodeGenerator& c) {
    constexpr u64 DF = 1ULL << 10;
    if (insn.mnemonic == ZYDIS_MNEMONIC_CLD) {
        c.mov(r8, ~DF);
        c.and_(qword[r13 + Offsets::Rflags], r8);
        return true;
    }
    if (insn.mnemonic == ZYDIS_MNEMONIC_STD) {
        c.mov(r8, DF);
        c.or_(qword[r13 + Offsets::Rflags], r8);
        return true;
    }
    return false;
}

// PREFETCH* / PREFETCHNTA / PREFETCHW — hint instructions; architecturally
// no-ops for correctness. Consume the operand (its address may be bogus,
// which is fine — prefetch never faults) and continue.
bool EmitPrefetch(const ZydisDecodedInstruction& insn,
                  const ZydisDecodedOperand* ops,
                  Xbyak::CodeGenerator& c) {
    switch (insn.mnemonic) {
        case ZYDIS_MNEMONIC_PREFETCHNTA:
        case ZYDIS_MNEMONIC_PREFETCHT0:
        case ZYDIS_MNEMONIC_PREFETCHT1:
        case ZYDIS_MNEMONIC_PREFETCHT2:
        case ZYDIS_MNEMONIC_PREFETCHW:
            (void)ops; (void)c;
            return true;   // pure no-op; do NOT compute the address
        default:
            return false;
    }
}

// XGETBV — read extended control register. We spoof XCR0 = 0x7
// (x87|SSE|AVX enabled) for index 0 (in ECX), zero for any other index.
// Writes EDX:EAX (zero-extended to RDX:RAX).
bool EmitXgetbv(const ZydisDecodedInstruction& insn, Xbyak::CodeGenerator& c) {
    if (insn.mnemonic != ZYDIS_MNEMONIC_XGETBV) return false;
    // ECX selects the register. index 0 -> 0x7, else 0.
    Xbyak::Label nonzero, done;
    c.mov(eax, dword[r13 + GprOffset(1)]);   // ecx (guest RCX low 32)
    c.test(eax, eax);
    c.jnz(nonzero);
    c.mov(rax, 0x7);
    c.mov(qword[r13 + GprOffset(0)], rax);   // RAX = 0x7
    c.xor_(rax, rax);
    c.mov(qword[r13 + GprOffset(2)], rax);   // RDX = 0
    c.jmp(done);
    c.L(nonzero);
    c.xor_(rax, rax);
    c.mov(qword[r13 + GprOffset(0)], rax);
    c.mov(qword[r13 + GprOffset(2)], rax);
    c.L(done);
    return true;
}

// The packed widening moves (VPMOVZX**/VPMOVSX**, including VPMOVZXDQ) are
// handled later by EmitPmovExtend, among the vector emitters (it needs
// ZydisVecToIndex / YmmChunkOffset which are declared further down).

// =============================================================================
// IMUL — three encoding forms (the comment block below documents IMUL).
// =============================================================================
//
// These operate on the implicit accumulator (RAX). They have no
// flag effects. We just defer to the host's identically-named
// instruction — its semantics line up exactly with the guest's.
//
//   CWDE : AX  → EAX (sign-extend 16→32, upper 32 of RAX = 0 by
//          x86-64's 32-bit-write zero-extension rule)
//   CDQE : EAX → RAX (sign-extend 32→64)
//   CDQ  : EAX → EDX:EAX (sign-extension of EAX into EDX, upper
//          32 of RDX = 0 by 32-bit-write rule)
//   CQO  : RAX → RDX:RAX (sign-extension of RAX into RDX)
// =============================================================================

bool EmitCbw(Xbyak::CodeGenerator& c) {
    // AL -> AX: sign-extend the low byte of guest RAX into AH, preserving
    // RAX bits 63:16. CBW writes only the 16-bit AX, so we read AL, run the
    // native cbw, and store back ONLY the 16-bit word — the in-memory upper
    // 48 bits are untouched. (Unlike CWDE/CDQE, this must not blow away the
    // high bits with a qword store.)
    c.mov(al, byte[r13 + GprOffset(0)]);
    c.cbw();
    c.mov(word[r13 + GprOffset(0)], ax);
    return true;
}

bool EmitCwde(Xbyak::CodeGenerator& c) {
    // Load low 16 of guest RAX into host AX, sign-extend to EAX,
    // store qword (upper 32 of rax is naturally zero after CWDE).
    c.mov(ax, word[r13 + GprOffset(0)]);
    c.cwde();
    c.mov(qword[r13 + GprOffset(0)], rax);
    return true;
}

bool EmitCwd(Xbyak::CodeGenerator& c) {
    // AX -> DX:AX: sign-extend AX's bit15 into DX, preserving RDX bits 63:16
    // and leaving RAX untouched. CWD writes only the 16-bit DX, so we load AX,
    // run native cwd, and store back ONLY the 16-bit DX word — the in-memory
    // upper 48 bits of RDX are preserved.
    c.mov(ax, word[r13 + GprOffset(0)]);
    c.cwd();
    c.mov(word[r13 + GprOffset(2)], dx);
    return true;
}

bool EmitCdqe(Xbyak::CodeGenerator& c) {
    // Load low 32 of guest RAX into host EAX, sign-extend to RAX,
    // store qword.
    c.mov(eax, dword[r13 + GprOffset(0)]);
    c.cdqe();
    c.mov(qword[r13 + GprOffset(0)], rax);
    return true;
}

bool EmitCdq(Xbyak::CodeGenerator& c) {
    // Load low 32 of RAX, CDQ sign-extends into EDX (with EDX
    // zero-extending its upper 32 per x86-64). Store rdx as
    // qword — host's zero-extension is preserved.
    c.mov(eax, dword[r13 + GprOffset(0)]);
    c.cdq();
    c.mov(qword[r13 + GprOffset(2)], rdx);
    return true;
}

bool EmitCqo(Xbyak::CodeGenerator& c) {
    // Load RAX, CQO sign-extends into RDX. Store rdx as qword.
    c.mov(rax, qword[r13 + GprOffset(0)]);
    c.cqo();
    c.mov(qword[r13 + GprOffset(2)], rdx);
    return true;
}

// =============================================================================
// STC / CLC / CMC — direct CF manipulation. No operands, no other
// flag effects, just bit 0 of rflags.
// =============================================================================

bool EmitStc(Xbyak::CodeGenerator& c) {
    // Set CF: OR rflags with 1. Imm8 fits the encoding directly.
    c.or_(qword[r13 + Offsets::Rflags], 1);
    return true;
}

bool EmitClc(Xbyak::CodeGenerator& c) {
    // Clear CF: load, BTR bit 0, store. Same dodge as INC/DEC's
    // CF restore, for the same `and qword[mem], imm32 sign-extends`
    // encoding-limit reason.
    c.mov(r10, qword[r13 + Offsets::Rflags]);
    c.btr(r10, 0);
    c.mov(qword[r13 + Offsets::Rflags], r10);
    return true;
}

bool EmitCmc(Xbyak::CodeGenerator& c) {
    // Complement CF: load, BTC bit 0, store.
    c.mov(r10, qword[r13 + Offsets::Rflags]);
    c.btc(r10, 0);
    c.mov(qword[r13 + Offsets::Rflags], r10);
    return true;
}

// =============================================================================
// LEAVE — function epilogue shorthand.
//
// Semantics: `mov rsp, rbp; pop rbp`. Tears down a standard
// function frame. One byte (0xC9), no operands, no flags affected.
// Extremely common at the tail of non-leaf C functions.
// =============================================================================

bool EmitLeave(Xbyak::CodeGenerator& c) {
    // Read current rbp — this will be the new rsp value (and the
    // address from which we pop the saved rbp).
    c.mov(rax, qword[r13 + GprOffset(5)]);  // rax = old rbp

    // Load *rax (the saved rbp value) into rcx — this is the new rbp.
    c.mov(rcx, qword[rax]);

    // Write new rbp.
    c.mov(qword[r13 + GprOffset(5)], rcx);

    // Compute new rsp = old rbp + 8 (the pop advanced past saved rbp).
    c.add(rax, 8);
    c.mov(qword[r13 + GprOffset(4)], rax);
    return true;
}

// =============================================================================
// ADC / SBB — add and subtract with carry input.
//
// ADC dst, src  →  dst = dst + src + CF
// SBB dst, src  →  dst = dst - src - CF
//
// These are how multi-precision arithmetic gets done:
//
//   add  rax, rcx      ; lo + lo, sets CF
//   adc  rbx, rdx      ; hi + hi + CF
//
// We round-trip flags through host rflags so the host CPU reads
// the existing guest CF as its ADC/SBB input AND writes the new
// CF correctly. Same flag-handling pattern as shifts/rotates/IMUL.
//
// Supported forms (matching ADD/SUB scope): r64,r64 and r64,imm32.
// Memory-source forms would follow but require careful operand
// ordering — deferred until a real binary hits them.
// =============================================================================

enum class AdcSbbKind { Adc, Sbb };

bool EmitAdcSbb64(const ZydisDecodedInstruction& insn,
                  const ZydisDecodedOperand* ops,
                  Xbyak::CodeGenerator& c,
                  AdcSbbKind kind) {
    if (insn.operand_width != 64) return false;
    if (ops[0].type != ZYDIS_OPERAND_TYPE_REGISTER) return false;
    const int dst_idx = ZydisGprToIndex(ops[0].reg.value);
    if (dst_idx < 0) return false;

    // Load src into rcx.
    if (ops[1].type == ZYDIS_OPERAND_TYPE_REGISTER) {
        const int src_idx = ZydisGprToIndex(ops[1].reg.value);
        if (src_idx < 0) return false;
        c.mov(rcx, qword[r13 + GprOffset(src_idx)]);
    } else if (ops[1].type == ZYDIS_OPERAND_TYPE_IMMEDIATE) {
        // Sign-extended from imm32 / imm8 — Zydis exposes the
        // sign-extended s64 in imm.value.s.
        c.mov(rcx, ops[1].imm.value.s);
    } else {
        return false;
    }

    // Load dst into rax.
    c.mov(rax, qword[r13 + GprOffset(dst_idx)]);

    // Round-trip flags so host CF is the same as guest CF.
    c.mov(r8, qword[r13 + Offsets::Rflags]);
    c.and_(r8, RflagsBits::HostLoadMask);
    c.push(r8);
    c.popfq();

    switch (kind) {
        case AdcSbbKind::Adc: c.adc(rax, rcx); break;
        case AdcSbbKind::Sbb: c.sbb(rax, rcx); break;
    }

    c.pushfq();
    c.pop(r8);
    EmitStoreArithFlags(c, r8);

    c.mov(qword[r13 + GprOffset(dst_idx)], rax);
    return true;
}

// =============================================================================
// Narrow-width (8-bit and 16-bit) ADD / SUB / CMP.
//
// Existing 64- and 32-bit handlers compute flags via the
// EmitFlagsFromAdd/Subtract helpers, which work in 64-bit terms.
// For narrow widths the flag-set rules differ:
//
//   - CF: carry out of bit 7 (8-bit) or bit 15 (16-bit), not bit 63.
//   - SF: bit 7 / bit 15 of the result, not bit 63.
//   - OF: signed-overflow detected at the narrow boundary.
//
// Recomputing all of these manually would mean three new flag
// helpers per width. Cleaner: use the host CPU's own narrow-width
// arithmetic instruction and round-trip flags. The host implements
// the exact same flag rules the guest expects.
//
// CMP is structurally a SUB that throws away the result but keeps
// the flags — handled by passing a "discard result" flag through
// the same emit function.
// =============================================================================

// Operation kind for `EmitNarrowArith8`/`EmitNarrowArith16`.
//
// The name "arith" is historical — these operations share the same
// round-trip-through-host-flags pattern (so the host CPU computes
// narrow-width flags correctly), but they include both arithmetic
// (ADD/SUB) and bitwise (AND/OR/XOR) operations. CMP and TEST are
// the "discard result" variants of SUB and AND respectively.
enum class NarrowArithKind { Add, Sub, Cmp, Test, And, Or, Xor, Sbb, Adc };

bool EmitNarrowArith8(const ZydisDecodedInstruction& insn,
                      const ZydisDecodedOperand* ops,
                      u64 next_rip,
                      Xbyak::CodeGenerator& c,
                      NarrowArithKind kind) {
    if (insn.operand_width != 8) return false;

    // ----- Memory destination/lhs -----
    //
    // Two sub-cases:
    //   * Cmp/Test discard the result (no writeback) — e.g. "cmp byte
    //     [mem], r8" / "test byte [mem], imm8".
    //   * Or/And/Xor/Add/Sub update memory in place — e.g. the
    //     bitfield-set idiom "or byte [mem], imm8".
    // For the writeback ops we keep the effective address in r10 across
    // the flag round-trip (which uses rdx), then store the result byte.
    if (ops[0].type == ZYDIS_OPERAND_TYPE_MEMORY) {
        const bool writeback =
            (kind == NarrowArithKind::Or  || kind == NarrowArithKind::And ||
             kind == NarrowArithKind::Xor || kind == NarrowArithKind::Add ||
             kind == NarrowArithKind::Sub);
        if (kind != NarrowArithKind::Cmp && kind != NarrowArithKind::Test &&
            !writeback) {
            return false;
        }
        if (!EmitEffectiveAddress(ops[0].mem, next_rip, c)) return false;
        c.mov(r10, rdx);              // stash address (survives popfq/pushfq)
        c.mov(al, byte[r10]);         // al = lhs ([mem])

        // Load rhs into cl. Mem-mem doesn't exist.
        if (ops[1].type == ZYDIS_OPERAND_TYPE_REGISTER) {
            // Use the 8-bit byte offset so high-byte regs (AH/CH/DH/BH)
            // resolve correctly — ZydisGprToIndex rejects those.
            const int src_off = ZydisGpr8ToByteOffset(ops[1].reg.value);
            if (src_off < 0) return false;
            c.mov(cl, byte[r13 + src_off]);
        } else if (ops[1].type == ZYDIS_OPERAND_TYPE_IMMEDIATE) {
            c.mov(cl, static_cast<u8>(ops[1].imm.value.u & 0xFF));
        } else {
            return false;
        }

        // Flag round-trip through the host CPU so it computes correct
        // 8-bit-width flags.
        c.mov(rdx, qword[r13 + Offsets::Rflags]);
        c.and_(rdx, RflagsBits::HostLoadMask);
        c.push(rdx);
        c.popfq();

        switch (kind) {
            case NarrowArithKind::Cmp:  c.cmp(al, cl); break;
            case NarrowArithKind::Test: c.test(al, cl); break;
            case NarrowArithKind::Or:   c.or_(al, cl); break;
            case NarrowArithKind::And:  c.and_(al, cl); break;
            case NarrowArithKind::Xor:  c.xor_(al, cl); break;
            case NarrowArithKind::Add:  c.add(al, cl); break;
            case NarrowArithKind::Sub:  c.sub(al, cl); break;
            default: return false;
        }

        c.pushfq();
        c.pop(rdx);
        EmitStoreArithFlags(c, rdx);

        if (writeback) {
            c.mov(byte[r10], al);     // store result back to [mem]
        }
        return true;
    }

    if (ops[0].type != ZYDIS_OPERAND_TYPE_REGISTER) return false;
    // 8-bit byte offset so high-byte regs (AH/CH/DH/BH) resolve right.
    const int dst_off = ZydisGpr8ToByteOffset(ops[0].reg.value);
    if (dst_off < 0) return false;

    // Load dst byte into al.
    c.mov(al, byte[r13 + dst_off]);

    // Load src byte into cl. Three source forms:
    //   - reg: read from the guest GPR slot (high-byte aware).
    //   - imm: literal byte.
    //   - mem: compute guest effective address into rdx, deref to cl.
    //
    // For mem source, EmitEffectiveAddress trashes rdx (and uses
    // rax/rcx as transients); we use rdx and cl AFTER, which is
    // safe — al is already loaded and cl is what we're populating.
    // The subsequent flag round-trip reuses rdx, also safe because
    // we no longer need the address by then.
    if (ops[1].type == ZYDIS_OPERAND_TYPE_REGISTER) {
        const int src_off = ZydisGpr8ToByteOffset(ops[1].reg.value);
        if (src_off < 0) return false;
        c.mov(cl, byte[r13 + src_off]);
    } else if (ops[1].type == ZYDIS_OPERAND_TYPE_IMMEDIATE) {
        c.mov(cl, static_cast<u8>(ops[1].imm.value.u & 0xFF));
    } else if (ops[1].type == ZYDIS_OPERAND_TYPE_MEMORY) {
        if (!EmitEffectiveAddress(ops[1].mem, next_rip, c)) return false;
        c.mov(cl, byte[rdx]);
    } else {
        return false;
    }

    // Round-trip flags (so host computes narrow-width flags).
    c.mov(rdx, qword[r13 + Offsets::Rflags]);
    c.and_(rdx, RflagsBits::HostLoadMask);
    c.push(rdx);
    c.popfq();

    switch (kind) {
        case NarrowArithKind::Add: c.add(al, cl); break;
        case NarrowArithKind::Sub: c.sub(al, cl); break;
        case NarrowArithKind::Cmp: c.cmp(al, cl); break;  // sets flags, no write
        case NarrowArithKind::Test: c.test(al, cl); break; // (al & cl), flags only
        case NarrowArithKind::And: c.and_(al, cl); break;
        case NarrowArithKind::Or:  c.or_(al, cl); break;
        case NarrowArithKind::Xor: c.xor_(al, cl); break;
        case NarrowArithKind::Sbb: c.sbb(al, cl); break;  // consumes CF (round-tripped)
        case NarrowArithKind::Adc: c.adc(al, cl); break;  // consumes CF
    }

    c.pushfq();
    c.pop(rdx);
    EmitStoreArithFlags(c, rdx);

    // CMP and TEST discard the result — only the others write back.
    // Narrow store preserves upper 56 bits per x86-64 semantics; writes
    // to the correct byte (incl. high-byte regs via dst_off).
    if (kind != NarrowArithKind::Cmp && kind != NarrowArithKind::Test) {
        c.mov(byte[r13 + dst_off], al);
    }
    return true;
}

bool EmitNarrowArith16(const ZydisDecodedInstruction& insn,
                       const ZydisDecodedOperand* ops,
                       u64 next_rip,
                       Xbyak::CodeGenerator& c,
                       NarrowArithKind kind) {
    if (insn.operand_width != 16) return false;

    // ----- Memory destination/lhs: cmp/test/and/or/xor/add/sub/sbb/adc
    //       word [mem], (reg|imm). Cmp/Test discard the result; the rest
    //       write the 16-bit result back. Address stashed in r10 across
    //       the flag round-trip (which uses rdx). -----
    if (ops[0].type == ZYDIS_OPERAND_TYPE_MEMORY) {
        const bool writeback =
            (kind == NarrowArithKind::Or  || kind == NarrowArithKind::And ||
             kind == NarrowArithKind::Xor || kind == NarrowArithKind::Add ||
             kind == NarrowArithKind::Sub || kind == NarrowArithKind::Sbb ||
             kind == NarrowArithKind::Adc);
        if (kind != NarrowArithKind::Cmp && kind != NarrowArithKind::Test &&
            !writeback) {
            return false;
        }
        if (!EmitEffectiveAddress(ops[0].mem, next_rip, c)) return false;
        c.mov(r10, rdx);                  // stash address
        c.mov(ax, word[r10]);             // ax = lhs ([mem])

        if (ops[1].type == ZYDIS_OPERAND_TYPE_REGISTER) {
            const int src_idx = ZydisGprToIndex(ops[1].reg.value);
            if (src_idx < 0) return false;
            c.mov(cx, word[r13 + GprOffset(src_idx)]);
        } else if (ops[1].type == ZYDIS_OPERAND_TYPE_IMMEDIATE) {
            c.mov(cx, static_cast<u16>(ops[1].imm.value.u & 0xFFFF));
        } else {
            return false;
        }

        c.mov(rdx, qword[r13 + Offsets::Rflags]);
        c.and_(rdx, RflagsBits::HostLoadMask);
        c.push(rdx);
        c.popfq();

        switch (kind) {
            case NarrowArithKind::Cmp:  c.cmp(ax, cx);  break;
            case NarrowArithKind::Test: c.test(ax, cx); break;
            case NarrowArithKind::Or:   c.or_(ax, cx);  break;
            case NarrowArithKind::And:  c.and_(ax, cx); break;
            case NarrowArithKind::Xor:  c.xor_(ax, cx); break;
            case NarrowArithKind::Add:  c.add(ax, cx);  break;
            case NarrowArithKind::Sub:  c.sub(ax, cx);  break;
            case NarrowArithKind::Sbb:  c.sbb(ax, cx);  break;
            case NarrowArithKind::Adc:  c.adc(ax, cx);  break;
            default: return false;
        }

        c.pushfq();
        c.pop(rdx);
        EmitStoreArithFlags(c, rdx);

        if (writeback) {
            c.mov(word[r10], ax);
        }
        return true;
    }

    if (ops[0].type != ZYDIS_OPERAND_TYPE_REGISTER) return false;
    const int dst_idx = ZydisGprToIndex(ops[0].reg.value);
    if (dst_idx < 0) return false;

    c.mov(ax, word[r13 + GprOffset(dst_idx)]);

    if (ops[1].type == ZYDIS_OPERAND_TYPE_REGISTER) {
        const int src_idx = ZydisGprToIndex(ops[1].reg.value);
        if (src_idx < 0) return false;
        c.mov(cx, word[r13 + GprOffset(src_idx)]);
    } else if (ops[1].type == ZYDIS_OPERAND_TYPE_IMMEDIATE) {
        c.mov(cx, static_cast<u16>(ops[1].imm.value.u & 0xFFFF));
    } else if (ops[1].type == ZYDIS_OPERAND_TYPE_MEMORY) {
        if (!EmitEffectiveAddress(ops[1].mem, next_rip, c)) return false;
        c.mov(cx, word[rdx]);
    } else {
        return false;
    }

    c.mov(rdx, qword[r13 + Offsets::Rflags]);
    c.and_(rdx, RflagsBits::HostLoadMask);
    c.push(rdx);
    c.popfq();

    switch (kind) {
        case NarrowArithKind::Add: c.add(ax, cx); break;
        case NarrowArithKind::Sub: c.sub(ax, cx); break;
        case NarrowArithKind::Cmp: c.cmp(ax, cx); break;
        case NarrowArithKind::Test: c.test(ax, cx); break;
        case NarrowArithKind::And: c.and_(ax, cx); break;
        case NarrowArithKind::Or:  c.or_(ax, cx); break;
        case NarrowArithKind::Xor: c.xor_(ax, cx); break;
        case NarrowArithKind::Sbb: c.sbb(ax, cx); break;  // consumes CF (round-tripped)
        case NarrowArithKind::Adc: c.adc(ax, cx); break;  // consumes CF
    }

    c.pushfq();
    c.pop(rdx);
    EmitStoreArithFlags(c, rdx);

    if (kind != NarrowArithKind::Cmp && kind != NarrowArithKind::Test) {
        c.mov(word[r13 + GprOffset(dst_idx)], ax);
    }
    return true;
}

// 32-bit narrow form. Same flag-round-trip pattern as the 8/16-bit
// variants, but with one critical writeback difference: x86-64
// semantics REQUIRE that a 32-bit write to a register zero-extend
// into the upper 32 bits of the 64-bit register slot. We achieve
// this by doing the operation in `eax` (which automatically zeros
// `rax`'s high half on the host) and writing the full 64-bit `rax`
// back to the GuestState slot. Cmp/Test discard the result and so
// don't store anything.
//
// Supported operand shapes:
//   - reg dst + (reg | imm | mem) src      — all kinds
//   - mem dst + (reg | imm) src             — Cmp/Test only (no writeback)
//
// Mem destination with ADD/SUB/AND/OR/XOR (which need a 4-byte
// store-back) is deferred — it requires preserving the address
// across the flag round-trip.
bool EmitNarrowArith32(const ZydisDecodedInstruction& insn,
                       const ZydisDecodedOperand* ops,
                       u64 next_rip,
                       Xbyak::CodeGenerator& c,
                       NarrowArithKind kind) {
    if (insn.operand_width != 32) return false;

    // ----- Memory destination/lhs — Cmp/Test only -----
    //
    // Common in compiled code as "cmp dword ptr [reg+disp8], imm8"
    // for struct-field comparisons. The lhs value is loaded from
    // memory into eax; rhs comes from a register or immediate
    // (memory-memory operands don't exist in x86 cmp/test). Since
    // Cmp/Test discard the result there's no writeback, which
    // lets us reuse rdx for the flag round-trip without saving
    // the address first.
    //
    // ----- Memory destination — ADD/SUB/AND/OR/XOR with writeback -----
    //
    // For the arithmetic kinds we additionally need to preserve the
    // address across the flag round-trip so we can store the result
    // back. We stash it in r10 (no other code path in this emitter
    // touches r10), then reuse rdx for the flag round-trip.
    if (ops[0].type == ZYDIS_OPERAND_TYPE_MEMORY) {
        if (!EmitEffectiveAddress(ops[0].mem, next_rip, c)) return false;
        // For arithmetic kinds, stash address in r10 for writeback.
        const bool needs_writeback =
            (kind == NarrowArithKind::Add ||
             kind == NarrowArithKind::Sub ||
             kind == NarrowArithKind::And ||
             kind == NarrowArithKind::Or  ||
             kind == NarrowArithKind::Xor);
        if (needs_writeback) {
            c.mov(r10, rdx);
            c.mov(eax, dword[r10]);
        } else {
            c.mov(eax, dword[rdx]);
        }

        // Load rhs into ecx.
        if (ops[1].type == ZYDIS_OPERAND_TYPE_REGISTER) {
            const int src_idx = ZydisGprToIndex(ops[1].reg.value);
            if (src_idx < 0) return false;
            c.mov(ecx, dword[r13 + GprOffset(src_idx)]);
        } else if (ops[1].type == ZYDIS_OPERAND_TYPE_IMMEDIATE) {
            c.mov(ecx, static_cast<u32>(ops[1].imm.value.u & 0xFFFFFFFFu));
        } else {
            // Mem-mem doesn't exist in x86; bail.
            return false;
        }

        // Flag round-trip. rdx is now free (address saved in r10
        // if we need writeback; flag-only paths never needed to
        // keep it past this point).
        c.mov(rdx, qword[r13 + Offsets::Rflags]);
        c.and_(rdx, RflagsBits::HostLoadMask);
        c.push(rdx);
        c.popfq();

        switch (kind) {
            case NarrowArithKind::Cmp:  c.cmp(eax, ecx);  break;
            case NarrowArithKind::Test: c.test(eax, ecx); break;
            case NarrowArithKind::Add:  c.add(eax, ecx);  break;
            case NarrowArithKind::Sub:  c.sub(eax, ecx);  break;
            case NarrowArithKind::And:  c.and_(eax, ecx); break;
            case NarrowArithKind::Or:   c.or_(eax, ecx);  break;
            case NarrowArithKind::Xor:  c.xor_(eax, ecx); break;
            case NarrowArithKind::Sbb:  c.sbb(eax, ecx);  break;
            case NarrowArithKind::Adc:  c.adc(eax, ecx);  break;
        }

        c.pushfq();
        c.pop(rdx);
        EmitStoreArithFlags(c, rdx);

        // Store the result back. dword store leaves the upper 32
        // bits of the surrounding qword untouched, which is the
        // correct semantics: memory writes don't have the
        // zero-extension behavior of register writes.
        if (needs_writeback) {
            c.mov(dword[r10], eax);
        }
        return true;
    }

    if (ops[0].type != ZYDIS_OPERAND_TYPE_REGISTER) return false;
    const int dst_idx = ZydisGprToIndex(ops[0].reg.value);
    if (dst_idx < 0) return false;

    // Load dst's low 32 bits into eax. Reading 32-bit from a 64-bit
    // slot is fine — we're explicitly working at 32-bit width.
    c.mov(eax, dword[r13 + GprOffset(dst_idx)]);

    if (ops[1].type == ZYDIS_OPERAND_TYPE_REGISTER) {
        const int src_idx = ZydisGprToIndex(ops[1].reg.value);
        if (src_idx < 0) return false;
        c.mov(ecx, dword[r13 + GprOffset(src_idx)]);
    } else if (ops[1].type == ZYDIS_OPERAND_TYPE_IMMEDIATE) {
        // 32-bit forms with imm operands use either imm32 or imm8-sx;
        // Zydis hands us the sign-extended u64. Truncate to u32 for
        // the host 32-bit move; the bit pattern is preserved.
        c.mov(ecx, static_cast<u32>(ops[1].imm.value.u & 0xFFFFFFFFu));
    } else if (ops[1].type == ZYDIS_OPERAND_TYPE_MEMORY) {
        if (!EmitEffectiveAddress(ops[1].mem, next_rip, c)) return false;
        c.mov(ecx, dword[rdx]);
    } else {
        return false;
    }

    // Round-trip guest rflags through host flags so the host CPU
    // computes correct 32-bit-width flags (CF/OF/SF/ZF/PF/AF).
    c.mov(rdx, qword[r13 + Offsets::Rflags]);
    c.and_(rdx, RflagsBits::HostLoadMask);
    c.push(rdx);
    c.popfq();

    switch (kind) {
        case NarrowArithKind::Add: c.add(eax, ecx); break;
        case NarrowArithKind::Sub: c.sub(eax, ecx); break;
        case NarrowArithKind::Cmp: c.cmp(eax, ecx); break;
        case NarrowArithKind::Test: c.test(eax, ecx); break;
        case NarrowArithKind::And: c.and_(eax, ecx); break;
        case NarrowArithKind::Or:  c.or_(eax, ecx); break;
        case NarrowArithKind::Xor: c.xor_(eax, ecx); break;
        case NarrowArithKind::Sbb: c.sbb(eax, ecx); break;  // consumes CF (round-tripped)
        case NarrowArithKind::Adc: c.adc(eax, ecx); break;  // consumes CF
    }

    c.pushfq();
    c.pop(rdx);
    EmitStoreArithFlags(c, rdx);

    if (kind != NarrowArithKind::Cmp && kind != NarrowArithKind::Test) {
        // 32-bit write to a register zeros the upper 32 bits. Our
        // `mov eax, ...` and the subsequent host op already did that
        // on the host side (writing to eax zeros bits 63:32 of rax),
        // so storing the full `rax` qword propagates that zeroing
        // into the guest slot.
        c.mov(qword[r13 + GprOffset(dst_idx)], rax);
    }
    return true;
}

// =============================================================================
// MOV — 8-bit and 16-bit forms.
//
// Mirror of the 64-bit MOV but with narrowed loads/stores. Key
// x86-64 semantic to preserve: writing an 8-bit or 16-bit register
// destination *preserves the upper bits* of the underlying 64-bit
// register slot (unlike 32-bit writes, which zero-extend). We
// achieve this by using narrowed memory operands (`byte`/`word`)
// for the store back into the guest state.
//
// Five operand combinations each, mirroring MOV64:
//   - r,  r
//   - r,  imm
//   - r,  m
//   - m,  r
//   - m,  imm
// =============================================================================

bool EmitMov8(const ZydisDecodedInstruction&,
              const ZydisDecodedOperand* ops,
              u64 next_rip,
              Xbyak::CodeGenerator& c) {
    const auto& dst = ops[0];
    const auto& src = ops[1];

    // Register destination.
    if (dst.type == ZYDIS_OPERAND_TYPE_REGISTER) {
        // Use the byte-offset helper so AH/CH/DH/BH (high-byte regs,
        // which ZydisGprToIndex can't map) resolve to the correct
        // byte within their parent slot.
        const int dst_off = ZydisGpr8ToByteOffset(dst.reg.value);
        if (dst_off < 0) return false;

        if (src.type == ZYDIS_OPERAND_TYPE_REGISTER) {
            // mov r8, r8 — copy one byte, preserving the rest of dst.
            const int src_off = ZydisGpr8ToByteOffset(src.reg.value);
            if (src_off < 0) return false;
            c.mov(al, byte[r13 + src_off]);
            c.mov(byte[r13 + dst_off], al);
            return true;
        }
        if (src.type == ZYDIS_OPERAND_TYPE_IMMEDIATE) {
            c.mov(byte[r13 + dst_off],
                  static_cast<u8>(src.imm.value.u & 0xFF));
            return true;
        }
        if (src.type == ZYDIS_OPERAND_TYPE_MEMORY) {
            if (!EmitEffectiveAddress(src.mem, next_rip, c)) return false;
            c.mov(cl, byte[rdx]);
            c.mov(byte[r13 + dst_off], cl);
            return true;
        }
        return false;
    }

    // Memory destination.
    if (dst.type == ZYDIS_OPERAND_TYPE_MEMORY) {
        if (src.type == ZYDIS_OPERAND_TYPE_REGISTER) {
            const int src_off = ZydisGpr8ToByteOffset(src.reg.value);
            if (src_off < 0) return false;
            if (!EmitEffectiveAddress(dst.mem, next_rip, c)) return false;
            c.mov(cl, byte[r13 + src_off]);
            c.mov(byte[rdx], cl);
            return true;
        }
        if (src.type == ZYDIS_OPERAND_TYPE_IMMEDIATE) {
            if (!EmitEffectiveAddress(dst.mem, next_rip, c)) return false;
            c.mov(byte[rdx], static_cast<u8>(src.imm.value.u & 0xFF));
            return true;
        }
        return false;
    }

    return false;
}

bool EmitMov16(const ZydisDecodedInstruction&,
               const ZydisDecodedOperand* ops,
               u64 next_rip,
               Xbyak::CodeGenerator& c) {
    const auto& dst = ops[0];
    const auto& src = ops[1];

    if (dst.type == ZYDIS_OPERAND_TYPE_REGISTER) {
        const int dst_idx = ZydisGprToIndex(dst.reg.value);
        if (dst_idx < 0) return false;

        if (src.type == ZYDIS_OPERAND_TYPE_REGISTER) {
            const int src_idx = ZydisGprToIndex(src.reg.value);
            if (src_idx < 0) return false;
            c.mov(ax, word[r13 + GprOffset(src_idx)]);
            c.mov(word[r13 + GprOffset(dst_idx)], ax);
            return true;
        }
        if (src.type == ZYDIS_OPERAND_TYPE_IMMEDIATE) {
            c.mov(word[r13 + GprOffset(dst_idx)],
                  static_cast<u16>(src.imm.value.u & 0xFFFF));
            return true;
        }
        if (src.type == ZYDIS_OPERAND_TYPE_MEMORY) {
            if (!EmitEffectiveAddress(src.mem, next_rip, c)) return false;
            c.mov(cx, word[rdx]);
            c.mov(word[r13 + GprOffset(dst_idx)], cx);
            return true;
        }
        return false;
    }

    if (dst.type == ZYDIS_OPERAND_TYPE_MEMORY) {
        if (src.type == ZYDIS_OPERAND_TYPE_REGISTER) {
            const int src_idx = ZydisGprToIndex(src.reg.value);
            if (src_idx < 0) return false;
            if (!EmitEffectiveAddress(dst.mem, next_rip, c)) return false;
            c.mov(cx, word[r13 + GprOffset(src_idx)]);
            c.mov(word[rdx], cx);
            return true;
        }
        if (src.type == ZYDIS_OPERAND_TYPE_IMMEDIATE) {
            if (!EmitEffectiveAddress(dst.mem, next_rip, c)) return false;
            c.mov(word[rdx], static_cast<u16>(src.imm.value.u & 0xFFFF));
            return true;
        }
        return false;
    }

    return false;
}

/// PUSH r64 — pushes a register onto the guest stack.
/// Semantics: guest_rsp -= 8; *guest_rsp = reg.
bool EmitPush(const ZydisDecodedInstruction& insn,
              const ZydisDecodedOperand* ops,
              u64 next_rip,
              Xbyak::CodeGenerator& c) {
    if (insn.operand_width != 64) return false;
    const auto& src = ops[0];

    // Resolve the value to push into rax.
    if (src.type == ZYDIS_OPERAND_TYPE_REGISTER) {
        const int src_idx = ZydisGprToIndex(src.reg.value);
        if (src_idx < 0) return false;
        c.mov(rax, qword[r13 + GprOffset(src_idx)]);
    } else if (src.type == ZYDIS_OPERAND_TYPE_IMMEDIATE) {
        // push imm8 / imm32 — both sign-extend to 64 bits.
        c.mov(rax, static_cast<u64>(src.imm.value.s));
    } else if (src.type == ZYDIS_OPERAND_TYPE_MEMORY) {
        // push qword[mem]. EmitEffectiveAddress writes rdx; deref into rax.
        if (!EmitEffectiveAddress(src.mem, next_rip, c)) return false;
        c.mov(rax, qword[rdx]);
    } else {
        return false;
    }

    c.mov(rdx, qword[r13 + GprOffset(kGuestRspIdx)]);
    c.sub(rdx, 8);
    c.mov(qword[rdx], rax);                              // write to guest stack
    c.mov(qword[r13 + GprOffset(kGuestRspIdx)], rdx);    // update RSP
    return true;
}

/// POP r64 — pops top of guest stack into a register.
/// Semantics: reg = *guest_rsp; guest_rsp += 8.
bool EmitPop(const ZydisDecodedInstruction& insn,
             const ZydisDecodedOperand* ops,
             Xbyak::CodeGenerator& c) {
    if (insn.operand_width != 64) return false;
    const auto& dst = ops[0];
    if (dst.type != ZYDIS_OPERAND_TYPE_REGISTER) return false;
    const int dst_idx = ZydisGprToIndex(dst.reg.value);
    if (dst_idx < 0) return false;

    // rdx = guest_rsp, rax = *rdx, increment rsp, store.
    c.mov(rdx, qword[r13 + GprOffset(kGuestRspIdx)]);
    c.mov(rax, qword[rdx]);
    c.add(rdx, 8);
    c.mov(qword[r13 + GprOffset(kGuestRspIdx)], rdx);
    c.mov(qword[r13 + GprOffset(dst_idx)], rax);
    return true;
}

/// RET — single-byte (0xC3) form. Pops the return address from the
/// guest stack and updates state.rip.
///
/// Semantics:
///   1. Load qword from [rsp_guest] into a scratch reg.
///   2. Set state.rip to that scratch reg.
///   3. Add 8 to rsp_guest.
///   4. Set state.exit_reason to BlockEnd.
///   5. Jump to the gateway dispatch-loop top via r14 (normal block
///      end; r15 is the FATAL exit stub). The dispatcher then resolves
///      the popped return address — guest block, HLE bridge target, or
///      the host-return sentinel.
///
/// Note: rsp_guest is GPR[4] (RSP per AMD64 ABI).
bool EmitRet(const ZydisDecodedInstruction& insn,
             const ZydisDecodedOperand* /*ops*/,
             Xbyak::CodeGenerator& c) {
    if (insn.mnemonic != ZYDIS_MNEMONIC_RET) return false;
    // Only the no-immediate form. RET imm16 (0xC2) shows up with
    // operand_count_visible > 0.
    if (insn.operand_count_visible != 0) return false;

    constexpr int RSP_IDX = 4; // GPR[4] = RSP in canonical order

    // Load guest's RSP into rax, then the return address at [rsp]
    // into rcx, then write rcx to state.rip.
    c.mov(rax, qword[r13 + GprOffset(RSP_IDX)]);
    c.mov(rcx, qword[rax]);
    c.mov(qword[r13 + Offsets::Rip], rcx);

    // Pop: guest_rsp += 8.
    c.add(rax, 8);
    c.mov(qword[r13 + GprOffset(RSP_IDX)], rax);

    // Set exit_reason = BlockEnd. The field is u32; use a 32-bit
    // store via dword[r13 + offsetof(...)].
    constexpr u32 EXIT_BLOCK_END = static_cast<u32>(ExitReason::BlockEnd);
    c.mov(dword[r13 + offsetof(GuestState, exit_reason)], EXIT_BLOCK_END);

    // Exit to gateway.
    c.jmp(r14);
    return true;
}

/// Helper: given a Jcc mnemonic, emit code that reads the relevant
/// flag(s) from state.rflags and produces the *condition* (1 if
/// the branch should be taken, 0 if not) in the low bit of rax.
/// All other bits of rax are zeroed.
///
/// Returns true on success; false if the mnemonic isn't a Jcc we
/// recognize.
///
/// We could be cleverer here — read state.rflags into a host
/// register and use `setcc` based on host flags — but that requires
/// `popfq` from a memory location, which would clobber other flag
/// state we don't want to touch. The explicit bit-test approach is
/// slightly more code but doesn't disturb anything.
bool EmitJccCondition(ZydisMnemonic mnemonic, Xbyak::CodeGenerator& c) {
    using namespace RflagsBits;
    // Load rflags into rax (we're going to compute the condition
    // into rax anyway).
    c.mov(rax, qword[r13 + Offsets::Rflags]);

    auto test_bit = [&](u64 bit) {
        // result in rcx = (rflags & bit) ? 1 : 0
        c.mov(rcx, rax);
        c.and_(rcx, bit);
        c.setnz(cl);
        c.movzx(rcx, cl);
    };
    auto test_not_bit = [&](u64 bit) {
        c.mov(rcx, rax);
        c.and_(rcx, bit);
        c.setz(cl);
        c.movzx(rcx, cl);
    };

    switch (mnemonic) {
        // Equal / not equal: ZF
        case ZYDIS_MNEMONIC_JZ:                          // JE / JZ: ZF=1
            test_bit(ZF); break;
        case ZYDIS_MNEMONIC_JNZ:                         // JNE / JNZ: ZF=0
            test_not_bit(ZF); break;

        // Sign-based: SF
        case ZYDIS_MNEMONIC_JS:                          // JS: SF=1
            test_bit(SF); break;
        case ZYDIS_MNEMONIC_JNS:                         // JNS: SF=0
            test_not_bit(SF); break;

        // Overflow: OF
        case ZYDIS_MNEMONIC_JO:                          // JO: OF=1
            test_bit(OF); break;
        case ZYDIS_MNEMONIC_JNO:                         // JNO: OF=0
            test_not_bit(OF); break;

        // Parity: PF
        case ZYDIS_MNEMONIC_JP:                          // JP / JPE: PF=1
            test_bit(PF); break;
        case ZYDIS_MNEMONIC_JNP:                         // JNP / JPO: PF=0
            test_not_bit(PF); break;

        // Unsigned comparison: CF, ZF
        case ZYDIS_MNEMONIC_JB:                          // JB / JC / JNAE: CF=1
            test_bit(CF); break;
        case ZYDIS_MNEMONIC_JNB:                         // JNB / JNC / JAE: CF=0
            test_not_bit(CF); break;
        case ZYDIS_MNEMONIC_JBE: {                       // JBE / JNA: CF=1 OR ZF=1
            c.mov(rcx, rax);
            c.and_(rcx, CF | ZF);
            c.setnz(cl);
            c.movzx(rcx, cl);
            break;
        }
        case ZYDIS_MNEMONIC_JNBE: {                      // JNBE / JA: CF=0 AND ZF=0
            c.mov(rcx, rax);
            c.and_(rcx, CF | ZF);
            c.setz(cl);
            c.movzx(rcx, cl);
            break;
        }

        // Signed comparison: SF, OF, ZF
        case ZYDIS_MNEMONIC_JL: {                        // JL / JNGE: SF != OF
            // rcx = (SF >> 7) XOR (OF >> 11), both in low bit.
            c.mov(rcx, rax);
            c.shr(rcx, 7);                               // SF -> bit 0
            c.mov(rdx, rax);
            c.shr(rdx, 11);                              // OF -> bit 0
            c.xor_(rcx, rdx);
            c.and_(rcx, 1);
            break;
        }
        case ZYDIS_MNEMONIC_JNL: {                       // JNL / JGE: SF == OF
            c.mov(rcx, rax);
            c.shr(rcx, 7);
            c.mov(rdx, rax);
            c.shr(rdx, 11);
            c.xor_(rcx, rdx);
            c.not_(rcx);
            c.and_(rcx, 1);
            break;
        }
        case ZYDIS_MNEMONIC_JLE: {                       // JLE / JNG: ZF=1 OR SF != OF
            // (SF != OF) into rdx. NOTE: must not use r8 here — EmitCmov
            // stashes the CMOV source operand in r8 across this call, so
            // clobbering r8 corrupted CMOVLE/CMOVG. rdx is free scratch.
            c.mov(rdx, rax);
            c.shr(rdx, 7);
            c.mov(rcx, rax);
            c.shr(rcx, 11);
            c.xor_(rdx, rcx);
            c.and_(rdx, 1);
            // Then ZF into rcx.
            c.mov(rcx, rax);
            c.shr(rcx, 6);
            c.and_(rcx, 1);
            // OR them.
            c.or_(rcx, rdx);
            break;
        }
        case ZYDIS_MNEMONIC_JNLE: {                      // JNLE / JG: ZF=0 AND SF == OF
            // (SF != OF) into rdx (see JLE note: avoid r8).
            c.mov(rdx, rax);
            c.shr(rdx, 7);
            c.mov(rcx, rax);
            c.shr(rcx, 11);
            c.xor_(rdx, rcx);
            c.and_(rdx, 1);
            // rdx = (SF != OF). We want NOT(ZF=1 OR (SF!=OF)).
            c.mov(rcx, rax);
            c.shr(rcx, 6);
            c.and_(rcx, 1);
            c.or_(rcx, rdx);
            // rcx is now (ZF=1 OR SF!=OF). Invert.
            c.not_(rcx);
            c.and_(rcx, 1);
            break;
        }

        default:
            return false;
    }
    return true;
}

/// Conditional jump (Jcc) — block terminator.
/// Reads flags, picks between branch-taken and fall-through target,
/// writes state.rip, exits to gateway.
bool EmitJcc(const ZydisDecodedInstruction& insn,
             const ZydisDecodedOperand* ops,
             u64 next_rip,
             Xbyak::CodeGenerator& c) {
    // Zydis emits the absolute target for near jumps in
    // ops[0].imm.is_relative + .value.s. The decoder normalizes to
    // an absolute address when ZYDIS_DECODER_FLAG_NORMALIZED is set;
    // here we calculate manually: target = next_rip + imm (relative)
    // or just .value.s (absolute).
    if (ops[0].type != ZYDIS_OPERAND_TYPE_IMMEDIATE) {
        return false;  // indirect Jcc isn't a real x86 form anyway
    }
    const u64 target = ops[0].imm.is_relative
        ? static_cast<u64>(static_cast<s64>(next_rip) + ops[0].imm.value.s)
        : static_cast<u64>(ops[0].imm.value.s);

    // Compute condition (1 or 0) into rcx. Bails if mnemonic isn't
    // a Jcc we recognize.
    if (!EmitJccCondition(insn.mnemonic, c)) return false;

    // Select target via conditional move. rdx = target, rax = next_rip;
    // if rcx != 0, set rax = rdx.
    c.mov(rax, next_rip);
    c.mov(rdx, target);
    c.test(rcx, rcx);
    c.cmovnz(rax, rdx);
    c.mov(qword[r13 + Offsets::Rip], rax);

    // Exit to gateway with BlockEnd.
    constexpr u32 EXIT_BLOCK_END = static_cast<u32>(ExitReason::BlockEnd);
    c.mov(dword[r13 + offsetof(GuestState, exit_reason)], EXIT_BLOCK_END);
    c.jmp(r14);
    return true;
}

/// JMP — direct (rel) and indirect (r/m64) forms. Block terminator.
///
/// Direct:  target = next_rip + disp32
/// Indirect (register): target = guest_reg
/// Indirect (memory):   target = *(guest_addr)  — the PLT pattern
///                      `jmp qword [rip+disp32]` is by far the most
///                      common shape here.
bool EmitJmp(const ZydisDecodedInstruction& insn,
             const ZydisDecodedOperand* ops,
             u64 next_rip,
             Xbyak::CodeGenerator& c) {
    if (insn.mnemonic != ZYDIS_MNEMONIC_JMP) return false;

    if (ops[0].type == ZYDIS_OPERAND_TYPE_IMMEDIATE) {
        // Direct relative jump.
        const u64 target = ops[0].imm.is_relative
            ? static_cast<u64>(static_cast<s64>(next_rip) + ops[0].imm.value.s)
            : static_cast<u64>(ops[0].imm.value.s);
        c.mov(rax, target);
        c.mov(qword[r13 + Offsets::Rip], rax);
    } else if (ops[0].type == ZYDIS_OPERAND_TYPE_REGISTER) {
        // Indirect through register: target = guest reg value.
        const int reg_idx = ZydisGprToIndex(ops[0].reg.value);
        if (reg_idx < 0) return false;
        c.mov(rax, qword[r13 + GprOffset(reg_idx)]);
        c.mov(qword[r13 + Offsets::Rip], rax);
    } else if (ops[0].type == ZYDIS_OPERAND_TYPE_MEMORY) {
        // Indirect through memory: target = *(effective address).
        // EmitEffectiveAddress puts the address in rdx; we then load
        // the 8-byte target from [rdx] into rax.
        if (!EmitEffectiveAddress(ops[0].mem, next_rip, c)) return false;
        c.mov(rax, qword[rdx]);
        c.mov(qword[r13 + Offsets::Rip], rax);
    } else {
        return false;
    }

    constexpr u32 EXIT_BLOCK_END = static_cast<u32>(ExitReason::BlockEnd);
    c.mov(dword[r13 + offsetof(GuestState, exit_reason)], EXIT_BLOCK_END);
    c.jmp(r14);
    return true;
}

/// CALL rel — direct near call (block terminator).
///
/// Semantics:
///   1. Push next_rip (the return address) onto the guest stack.
///   2. Set state.rip = target.
///   3. Exit to gateway. The dispatcher then enters the callee
///      block; when the callee RETs, it pops next_rip into
///      state.rip and the dispatcher re-enters the block after our
///      call site.
///
/// This call/return matching only works because the gateway now
/// loops (PR: this one). In the previous one-block-per-Run model,
/// CALL couldn't return.
///
/// CALL r/m64 (indirect) is not handled yet — it'd be the same
/// shape but with the target loaded from a register or memory.
/// CALL — direct (rel32) and indirect (r/m64) forms. Block terminator.
///
/// Semantics in all cases:
///   1. Push next_rip onto guest stack (the return address).
///   2. Set state.rip = target.
///   3. Exit to gateway. The dispatcher then enters the callee
///      block; when the callee RETs, it pops next_rip into
///      state.rip and the dispatcher re-enters the post-call block.
///
/// Direct:   target = next_rip + disp32
/// Indirect (register):  target = guest_reg
/// Indirect (memory):    target = *(effective_addr)  — vtable / GOT
bool EmitCall(const ZydisDecodedInstruction& insn,
              const ZydisDecodedOperand* ops,
              u64 next_rip,
              Xbyak::CodeGenerator& c) {
    if (insn.mnemonic != ZYDIS_MNEMONIC_CALL) return false;

    // Step 1: compute target into rax. We do this BEFORE the stack
    // push because the memory form uses rdx as scratch via
    // EmitEffectiveAddress; the push step also uses rdx and would
    // collide.
    if (ops[0].type == ZYDIS_OPERAND_TYPE_IMMEDIATE) {
        const u64 target = ops[0].imm.is_relative
            ? static_cast<u64>(static_cast<s64>(next_rip) + ops[0].imm.value.s)
            : static_cast<u64>(ops[0].imm.value.s);
        c.mov(rax, target);
    } else if (ops[0].type == ZYDIS_OPERAND_TYPE_REGISTER) {
        const int reg_idx = ZydisGprToIndex(ops[0].reg.value);
        if (reg_idx < 0) return false;
        c.mov(rax, qword[r13 + GprOffset(reg_idx)]);
    } else if (ops[0].type == ZYDIS_OPERAND_TYPE_MEMORY) {
        if (!EmitEffectiveAddress(ops[0].mem, next_rip, c)) return false;
        c.mov(rax, qword[rdx]);
    } else {
        return false;
    }

    // Step 2: push next_rip onto guest stack. rax holds target;
    // use rdx and rcx as scratch.
    c.mov(rdx, qword[r13 + GprOffset(kGuestRspIdx)]);
    c.sub(rdx, 8);
    c.mov(rcx, next_rip);
    c.mov(qword[rdx], rcx);
    c.mov(qword[r13 + GprOffset(kGuestRspIdx)], rdx);

    // Step 3: set state.rip = rax (the target).
    c.mov(qword[r13 + Offsets::Rip], rax);

    // Exit to gateway.
    constexpr u32 EXIT_BLOCK_END = static_cast<u32>(ExitReason::BlockEnd);
    c.mov(dword[r13 + offsetof(GuestState, exit_reason)], EXIT_BLOCK_END);
    c.jmp(r14);
    return true;
}

// ============================================================================
// AVX / VEX-encoded 128/256-bit vector instructions
// ============================================================================
//
// First-class AVX support, scoped tightly to what the game actually
// hits early: VMOVUPS (load/store/reg-reg) and VXORPS / VPXOR
// (3-operand bitwise xor). We deliberately do NOT use host XMM/YMM
// registers — instead we model each operation as a sequence of
// 64-bit integer loads, integer ops, and 64-bit stores. The trade-off:
//
//   pro:  no host-SIMD save/restore around helper calls; lifter
//         logic stays purely GPR-based and matches the rest of the
//         file; deterministic register allocation (rax/rdx only).
//   pro:  bypasses an xbyak-VEX-encoding-API correctness question we
//         haven't yet validated.
//   con:  2× or 4× more host instructions per guest insn (16- or
//         32-byte vector → 2 or 4 mov/xor pairs). Acceptable for
//         the initial cut; can be tightened to host vmovups+vpxor
//         once block chaining is in and we benchmark.
//
// One subtlety: VEX-encoded 128-bit ops zero the upper 128 bits of
// the destination YMM. Legacy SSE 128-bit ops preserve them. Since
// every mnemonic here is VEX-only, the zero-upper write is
// unconditional for the xmm form.

/// Map a Zydis XMM*/YMM* register enum to a 0..31 lane index.
/// XMM and YMM are aliased: xmm0 is the low 128 bits of ymm0, so
/// both map to the same lane in `GuestState::ymm`.
int ZydisVecToIndex(ZydisRegister reg) {
    if (reg >= ZYDIS_REGISTER_XMM0 && reg <= ZYDIS_REGISTER_XMM31) {
        return static_cast<int>(reg) - static_cast<int>(ZYDIS_REGISTER_XMM0);
    }
    if (reg >= ZYDIS_REGISTER_YMM0 && reg <= ZYDIS_REGISTER_YMM31) {
        return static_cast<int>(reg) - static_cast<int>(ZYDIS_REGISTER_YMM0);
    }
    return -1;
}

/// Byte offset to the `chunk`'th 64-bit word of YMM lane `lane_idx`.
/// Lanes are 32 bytes; chunks are 8 bytes (chunk in 0..3).
constexpr u32 YmmChunkOffset(int lane_idx, int chunk) {
    return static_cast<u32>(offsetof(GuestState, ymm) +
                            static_cast<size_t>(lane_idx) * 32u +
                            static_cast<size_t>(chunk) * 8u);
}

// ============================================================================
// x87 FPU — Group 1: load / store (fld, fst, fstp, fild, fistp)
// ============================================================================
//
// The x87 register file is a stack of 8 registers, st[0..7], accessed
// relative to a top-of-stack pointer GuestState::fpu_top (0..7):
//   ST(i)  ==  st[(fpu_top + i) & 7]
// A push (fld/fild) decrements top then writes the new ST(0); a pop
// (fstp/fistp) reads ST(0) then increments top.
//
// We store each register as a 64-bit double bit-pattern (see the
// precision note in guest_state.h). Memory loads of m32fp are widened
// to double via cvtss2sd; m64fp is moved verbatim. Stores narrow back
// via cvtsd2ss (m32fp) or move verbatim (m64fp). Integer loads/stores
// (fild/fistp) convert via cvtsi2sd / cvtsd2si.
//
// Register/scratch usage inside these emitters:
//   r13 = GuestState (never modified)
//   rdx = effective address of the memory operand (from
//         EmitEffectiveAddress; also clobbers rax)
//   rcx = scratch for the top-of-stack index math
//   rax = scratch (clobbered by EmitEffectiveAddress)
//   xmm0 = transfer register for the FP value
//
// IMPORTANT ordering rule: EmitEffectiveAddress clobbers rax and writes
// rdx. The fpu_top index math uses rcx (and a fresh rax load), so we
// always compute the address FIRST, move the loaded value into xmm0,
// and only THEN touch fpu_top — never interleaving in a way that lets
// the address computation stomp a value we still need.

/// Emit code that computes, into `dst_addr_reg`, the host address of
/// the x87 register st[(fpu_top + delta) & 7]. Uses rcx as scratch for
/// the index. `delta` is a small signed constant (e.g. -1 for a push
/// target, 0 for ST(0)). The masking with 7 keeps the index in range.
///
/// dst_addr_reg must not be rcx. Typical callers pass r8 or r9.
void EmitX87RegAddr(Xbyak::CodeGenerator& c, const Xbyak::Reg64& dst_addr_reg,
                    int delta) {
    // rcx = fpu_top
    c.mov(ecx, dword[r13 + offsetof(GuestState, fpu_top)]);
    if (delta != 0) {
        c.add(ecx, delta);
    }
    c.and_(ecx, 7);
    // dst = r13 + offsetof(st) + rcx*8
    c.lea(dst_addr_reg, ptr[r13 + offsetof(GuestState, st)]);
    c.lea(dst_addr_reg, ptr[dst_addr_reg + rcx * 8]);
}

/// Emit a push: fpu_top = (fpu_top - 1) & 7, then store xmm0 (a double)
/// into the new ST(0) = st[fpu_top], and mark that slot valid in the
/// tag word. Assumes the value to push is already in xmm0.
void EmitX87Push(Xbyak::CodeGenerator& c) {
    // top = (top - 1) & 7
    c.mov(ecx, dword[r13 + offsetof(GuestState, fpu_top)]);
    c.sub(ecx, 1);
    c.and_(ecx, 7);
    c.mov(dword[r13 + offsetof(GuestState, fpu_top)], ecx);
    // st[top] = xmm0   (address = r13 + off(st) + top*8, top now in ecx)
    c.lea(r8, ptr[r13 + offsetof(GuestState, st)]);
    c.movsd(ptr[r8 + rcx * 8], xmm0);
    // Mark slot valid: clear the empty bit. We track a simple
    // per-physical-register "in use" model in fpu_tag (bit i = 1 means
    // st[i] is non-empty). Set bit `top`.
    c.mov(r9d, 1);
    // shift left by cl (cl = top); shl r9d, cl
    c.shl(r9d, cl);
    c.or_(word[r13 + offsetof(GuestState, fpu_tag)], r9w);
}

/// Emit a pop: load ST(0) = st[fpu_top] into xmm0, mark that slot
/// empty, then fpu_top = (fpu_top + 1) & 7. Used after a store that
/// pops (fstp/fistp). Callers that need the value must read xmm0 (or
/// the memory store) BEFORE calling this; here we only need to clear
/// the tag and advance top, the value having already been consumed.
void EmitX87PopDiscardValue(Xbyak::CodeGenerator& c) {
    // ecx = top
    c.mov(ecx, dword[r13 + offsetof(GuestState, fpu_top)]);
    // mark slot empty: clear bit `top` in fpu_tag.
    c.mov(r9d, 1);
    c.shl(r9d, cl);
    c.not_(r9d);
    c.and_(word[r13 + offsetof(GuestState, fpu_tag)], r9w);
    // top = (top + 1) & 7
    c.add(ecx, 1);
    c.and_(ecx, 7);
    c.mov(dword[r13 + offsetof(GuestState, fpu_top)], ecx);
}

/// FLD m32fp / m64fp — load a float/double from memory, push onto the
/// x87 stack. (FLD ST(i) — the register form — is handled in Group 2.)
bool EmitFld(const ZydisDecodedInstruction& insn,
             const ZydisDecodedOperand* ops,
             u64 next_rip,
             Xbyak::CodeGenerator& c) {
    // Register form: `fld st(i)` (D9 C0+i) — push a copy of ST(i) onto
    // the stack. The index i is relative to the CURRENT top, evaluated
    // BEFORE the push, so we read st[(top+i)&7] into xmm0 first, then
    // push (which decrements top and stores xmm0 into the new ST(0)).
    if (ops[0].type == ZYDIS_OPERAND_TYPE_REGISTER) {
        if (ops[0].reg.value < ZYDIS_REGISTER_ST0 ||
            ops[0].reg.value > ZYDIS_REGISTER_ST7) {
            return false;
        }
        const int i = ops[0].reg.value - ZYDIS_REGISTER_ST0;
        EmitX87RegAddr(c, r8, i);          // r8 = &st[(top+i)&7]
        c.movsd(xmm0, ptr[r8]);
        EmitX87Push(c);                    // top--, st[top] = xmm0
        return true;
    }
    // operand width: 32 = m32fp (float), 64 = m64fp (double).
    // NOTE: Zydis reports operand_width==32 for ALL x87 memory operands
    // (dword and qword alike); the true access size is in ops[0].size.
    // Keying off operand_width would silently truncate every m64 form.
    const u32 w = ops[0].size;
    if (w != 32 && w != 64 && w != 80) return false;

    // 80-bit extended-precision load (`fld tbyte`). xbyak has no m80
    // address form and our guest slots hold doubles, so we round-trip
    // through the host x87 FPU: load the 80-bit value onto the host x87
    // stack, allocate the guest slot (top decrement + tag), then store
    // it as a 64-bit double straight into that slot via host `fstp
    // qword`. This both narrows to double and pops the host x87 stack,
    // keeping it balanced.
    if (w == 80) {
        if (!EmitEffectiveAddress(ops[0].mem, next_rip, c)) return false;
        c.db(0xDB); c.db(0x2A);               // fld tbyte [rdx]  -> host x87 st0
        // Allocate the guest slot: top = (top-1)&7, mark tag, r8 = &st[top].
        c.mov(ecx, dword[r13 + offsetof(GuestState, fpu_top)]);
        c.sub(ecx, 1);
        c.and_(ecx, 7);
        c.mov(dword[r13 + offsetof(GuestState, fpu_top)], ecx);
        c.mov(r9d, 1);
        c.shl(r9d, cl);
        c.or_(word[r13 + offsetof(GuestState, fpu_tag)], r9w);
        c.lea(r8, ptr[r13 + offsetof(GuestState, st)]);
        c.lea(r8, ptr[r8 + rcx * 8]);         // r8 = &st[top]
        c.db(0x41); c.db(0xDD); c.db(0x18);   // fstp qword [r8] (narrow+pop host)
        return true;
    }

    if (!EmitEffectiveAddress(ops[0].mem, next_rip, c)) return false;
    // rdx = address. Load + widen into xmm0.
    if (w == 32) {
        c.movss(xmm0, dword[rdx]);
        c.cvtss2sd(xmm0, xmm0);
    } else {
        c.movsd(xmm0, qword[rdx]);
    }
    EmitX87Push(c);
    return true;
}

/// FST / FSTP m32fp / m64fp — store ST(0) to memory (narrowing to
/// float for m32fp). FSTP additionally pops. (Register forms deferred.)
bool EmitFstOrFstp(const ZydisDecodedInstruction& insn,
                   const ZydisDecodedOperand* ops,
                   u64 next_rip,
                   Xbyak::CodeGenerator& c) {
    const bool pop = (insn.mnemonic == ZYDIS_MNEMONIC_FSTP);

    // Register form: FST ST(i) / FSTP ST(i) copies ST(0) into ST(i), then pops
    // for FSTP. Zydis may present one visible ST operand (the dest) or two
    // (dest + implicit ST0); take the destination from the first ST-register
    // operand. FSTP ST(0) is the common "discard top" idiom.
    if (ops[0].type == ZYDIS_OPERAND_TYPE_REGISTER) {
        int di = -1;
        for (int i = 0; i < insn.operand_count_visible; ++i) {
            if (ops[i].type == ZYDIS_OPERAND_TYPE_REGISTER &&
                ops[i].reg.value >= ZYDIS_REGISTER_ST0 &&
                ops[i].reg.value <= ZYDIS_REGISTER_ST7) {
                di = ops[i].reg.value - ZYDIS_REGISTER_ST0;
                break;
            }
        }
        if (di < 0) return false;
        EmitX87RegAddr(c, r8, 0);          // r8 = &ST(0)
        c.movsd(xmm0, ptr[r8]);
        EmitX87RegAddr(c, r8, di);         // r8 = &ST(di)
        c.movsd(ptr[r8], xmm0);            // ST(di) = ST(0) (self-copy if di==0)
        if (pop) EmitX87PopDiscardValue(c);
        return true;
    }

    if (ops[0].type != ZYDIS_OPERAND_TYPE_MEMORY) {
        return false;
    }
    // NOTE: Zydis reports operand_width==32 for ALL x87 memory operands
    // (dword and qword alike); the true access size is in ops[0].size.
    // Keying off operand_width would silently truncate every m64 form.
    const u32 w = ops[0].size;
    if (w != 32 && w != 64 && w != 80) return false;

    // 80-bit extended-precision store (m80 / `fstp tbyte`). xbyak has no
    // m80 address form, and our guest x87 slots hold 64-bit doubles, so
    // we round-trip through the host x87 FPU: push the guest ST(0)
    // double onto the host x87 stack (fld qword), then store+pop it as
    // 80-bit extended to guest memory (fstp tbyte). FST (no-pop) to m80
    // is not encodable on x86 — only FSTP — so `pop` is always true here.
    if (w == 80) {
        EmitX87RegAddr(c, r8, 0);             // r8 = &st[top] (the double)
        if (!EmitEffectiveAddress(ops[0].mem, next_rip, c)) return false;
        // rdx = destination address. Host x87: fld qword [r8]; fstp tbyte [rdx].
        c.db(0x41); c.db(0xDD); c.db(0x00);   // fld qword [r8]
        c.db(0xDB); c.db(0x3A);               // fstp tbyte [rdx]
        // Pop the guest stack (the architectural FSTP popped ST(0)).
        EmitX87PopDiscardValue(c);
        return true;
    }

    // Load ST(0) into xmm0 first (address math uses rcx + r8).
    EmitX87RegAddr(c, r8, 0);     // r8 = &st[top]
    c.movsd(xmm0, ptr[r8]);

    // Compute the destination address (clobbers rax, writes rdx).
    if (!EmitEffectiveAddress(ops[0].mem, next_rip, c)) return false;
    if (w == 32) {
        c.cvtsd2ss(xmm0, xmm0);
        c.movss(dword[rdx], xmm0);
    } else {
        c.movsd(qword[rdx], xmm0);
    }

    if (pop) {
        EmitX87PopDiscardValue(c);
    }
    return true;
}

/// FILD m32/m64 — load a signed integer from memory, convert to double,
/// push. FILD m16 (16-bit) is rarer; deferred until observed.
bool EmitFild(const ZydisDecodedInstruction& insn,
              const ZydisDecodedOperand* ops,
              u64 next_rip,
              Xbyak::CodeGenerator& c) {
    if (ops[0].type != ZYDIS_OPERAND_TYPE_MEMORY) return false;
    // NOTE: Zydis reports operand_width==32 for ALL x87 memory operands
    // (dword and qword alike); the true access size is in ops[0].size.
    // Keying off operand_width would silently truncate every m64 form.
    const u32 w = ops[0].size;
    if (w != 32 && w != 64) return false;

    if (!EmitEffectiveAddress(ops[0].mem, next_rip, c)) return false;
    // rdx = address. Load signed integer into a GPR, convert to double.
    if (w == 32) {
        c.movsxd(rax, dword[rdx]);   // sign-extend 32->64
        c.cvtsi2sd(xmm0, rax);
    } else {
        c.mov(rax, qword[rdx]);
        c.cvtsi2sd(xmm0, rax);
    }
    EmitX87Push(c);
    return true;
}

/// FISTP m32/m64 — convert ST(0) to a signed integer, store, pop.
/// Uses cvtsd2si (honors the host rounding mode, which mirrors the
/// guest control word's rounding field for the common round-to-nearest
/// case). FIST (no pop) and m16 are deferred.
bool EmitFistp(const ZydisDecodedInstruction& insn,
               const ZydisDecodedOperand* ops,
               u64 next_rip,
               Xbyak::CodeGenerator& c) {
    // FISTP rounds ST(0) per the x87 CONTROL WORD's RC field (fpu_cw bits
    // 11:10) — NOT per MXCSR. cvtsd2si follows MXCSR.RC, so to make the host
    // convert honor the guest's x87 rounding we transplant cw.RC into
    // MXCSR.RC around the conversion. The two fields use the same encoding
    // (00 nearest, 01 down, 10 up, 11 chop), so the transplant is a mask,
    // shift, and OR — no translation table. The save/modify/restore goes
    // through GuestState::scratch[0..1] (stmxcsr/ldmxcsr need a memory
    // operand); scratch is documented as not-preserved-across-dispatch,
    // which a single instruction's emission satisfies trivially. This also
    // composes with a guest LDMXCSR value applied to the host (the
    // dispatcher's rounding swap): we save whatever MXCSR is live, override
    // only RC, and put the live value back.
    //
    // FISTTP (SSE3) always truncates regardless of cw.RC: cvttsd2si needs
    // no MXCSR help.
    const bool is_truncating = (insn.mnemonic == ZYDIS_MNEMONIC_FISTTP);
    if (insn.mnemonic != ZYDIS_MNEMONIC_FISTP && !is_truncating) return false;
    if (ops[0].type != ZYDIS_OPERAND_TYPE_MEMORY) return false;
    // NOTE: Zydis reports operand_width==32 for ALL x87 memory operands
    // (dword and qword alike); the true access size is in ops[0].size.
    // Keying off operand_width would silently truncate every m64 form.
    const u32 w = ops[0].size;
    if (w != 16 && w != 32 && w != 64) return false;

    // Load ST(0) into xmm0 (address math uses rcx + r8).
    EmitX87RegAddr(c, r8, 0);
    c.movsd(xmm0, ptr[r8]);

    // Destination address (clobbers rax, writes rdx).
    if (!EmitEffectiveAddress(ops[0].mem, next_rip, c)) return false;

    constexpr int kScratch0 = static_cast<int>(offsetof(GuestState, scratch));
    constexpr int kScratch1 = kScratch0 + 8;
    if (!is_truncating) {
        // Transplant cw.RC (bits 11:10) into MXCSR.RC (bits 14:13): same
        // encoding, 3-bit left shift. eax/ecx are free here (the EA above
        // already consumed rax; the value lives in xmm0, the address in rdx).
        c.stmxcsr(dword[r13 + kScratch0]);              // save live MXCSR
        c.mov(eax, dword[r13 + kScratch0]);
        c.and_(eax, ~(3u << 13));                       // clear RC
        c.movzx(ecx, word[r13 + offsetof(GuestState, fpu_cw)]);
        c.and_(ecx, 3u << 10);                          // cw.RC in place
        c.shl(ecx, 3);                                  // -> MXCSR.RC position
        c.or_(eax, ecx);
        c.mov(dword[r13 + kScratch1], eax);
        c.ldmxcsr(dword[r13 + kScratch1]);              // apply
    }

    if (w == 64) {
        if (is_truncating) c.cvttsd2si(rax, xmm0);   // double -> 64-bit signed, truncate
        else               c.cvtsd2si(rax, xmm0);    // double -> 64-bit signed, per cw.RC
        if (!is_truncating) c.ldmxcsr(dword[r13 + kScratch0]); // restore live MXCSR
        c.mov(qword[rdx], rax);
    } else {
        // 16- and 32-bit both convert to a 32-bit signed int; the store width
        // differs (m16 writes a word, preserving adjacent bytes). NOTE: the
        // m16 form's overflow behavior diverges from hardware (x86 writes the
        // 0x8000 integer-indefinite on 16-bit overflow; we store the low word
        // of the 32-bit conversion) — pre-existing, out of scope here.
        if (is_truncating) c.cvttsd2si(eax, xmm0);
        else               c.cvtsd2si(eax, xmm0);
        if (!is_truncating) c.ldmxcsr(dword[r13 + kScratch0]); // restore live MXCSR
        if (w == 16) c.mov(word[rdx], ax);
        else         c.mov(dword[rdx], eax);
    }

    EmitX87PopDiscardValue(c);
    return true;
}

/// FADDP / FMULP / FSUBP / FSUBRP / FDIVP / FDIVRP st(i), st(0) —
/// x87 arithmetic with pop (DE xx). The operation writes ST(i), then
/// ST(0) is popped. Operand order (Intel):
///   FADDP : st(i) = st(i) + st(0)
///   FMULP : st(i) = st(i) * st(0)
///   FSUBP : st(i) = st(i) - st(0)
///   FSUBRP: st(i) = st(0) - st(i)   (reversed)
///   FDIVP : st(i) = st(i) / st(0)
///   FDIVRP: st(i) = st(0) / st(i)   (reversed)
/// i is relative to the CURRENT top (before the pop). We load st(0)
/// into xmm0 and st(i) into xmm1, compute on host SSE2 doubles, write
/// the result back to st(i)'s slot, then pop. Addresses are captured in
/// callee-stable regs (r8 = &st(0), r10 = &st(i)) so EmitX87RegAddr's
/// rcx/lea scratch use doesn't clobber them between the two lookups.
bool EmitFpuArithPop(const ZydisDecodedInstruction& insn,
                     const ZydisDecodedOperand* ops,
                     Xbyak::CodeGenerator& c) {
    const ZydisMnemonic m = insn.mnemonic;
    switch (m) {
        case ZYDIS_MNEMONIC_FADDP:  case ZYDIS_MNEMONIC_FMULP:
        case ZYDIS_MNEMONIC_FSUBP:  case ZYDIS_MNEMONIC_FSUBRP:
        case ZYDIS_MNEMONIC_FDIVP:  case ZYDIS_MNEMONIC_FDIVRP:
            break;
        default: return false;
    }
    if (ops[0].type != ZYDIS_OPERAND_TYPE_REGISTER) return false;
    if (ops[0].reg.value < ZYDIS_REGISTER_ST0 ||
        ops[0].reg.value > ZYDIS_REGISTER_ST7) {
        return false;
    }
    const int i = ops[0].reg.value - ZYDIS_REGISTER_ST0;
    if (i == 0) return false;  // st(0),st(0) — not a meaningful pop form

    // r8 = &st(0); load into xmm0. r10 = &st(i); load into xmm1.
    EmitX87RegAddr(c, r8, 0);
    c.movsd(xmm0, ptr[r8]);          // xmm0 = st(0)
    EmitX87RegAddr(c, r10, i);
    c.movsd(xmm1, ptr[r10]);         // xmm1 = st(i)

    // Compute into xmm1 (= the st(i) result). Non-reversed ops are
    // st(i) <op> st(0); reversed subtract/divide swap the operands.
    switch (m) {
        case ZYDIS_MNEMONIC_FADDP:  c.addsd(xmm1, xmm0); break;  // i + 0
        case ZYDIS_MNEMONIC_FMULP:  c.mulsd(xmm1, xmm0); break;  // i * 0
        case ZYDIS_MNEMONIC_FSUBP:  c.subsd(xmm1, xmm0); break;  // i - 0
        case ZYDIS_MNEMONIC_FDIVP:  c.divsd(xmm1, xmm0); break;  // i / 0
        case ZYDIS_MNEMONIC_FSUBRP:                              // 0 - i
            c.subsd(xmm0, xmm1);     // xmm0 = st(0) - st(i)
            c.movsd(xmm1, xmm0);
            break;
        case ZYDIS_MNEMONIC_FDIVRP:                             // 0 / i
            c.divsd(xmm0, xmm1);     // xmm0 = st(0) / st(i)
            c.movsd(xmm1, xmm0);
            break;
        default: return false;
    }
    c.movsd(ptr[r10], xmm1);         // st(i) = result

    // Pop st(0).
    EmitX87PopDiscardValue(c);
    return true;
}

/// FADD/FMUL/FSUB/FSUBR/FDIV/FDIVR (non-pop). Forms:
///   2 reg ops: `fadd st(0),st(i)` (dst=ST0) or `fadd st(i),st(0)` (dst=STi).
///   1 mem op:  `fadd m32/m64` (dst=ST0, operand from memory).
/// FSUBR/FDIVR reverse operand order. Computes on host SSE2 doubles; writes
/// the result to dst's slot. No pop.
bool EmitFpuArith(const ZydisDecodedInstruction& insn,
                  const ZydisDecodedOperand* ops,
                  u64 next_rip,
                  Xbyak::CodeGenerator& c) {
    const ZydisMnemonic m = insn.mnemonic;
    const bool rev   = (m == ZYDIS_MNEMONIC_FSUBR || m == ZYDIS_MNEMONIC_FDIVR);
    const bool isadd = (m == ZYDIS_MNEMONIC_FADD);
    const bool ismul = (m == ZYDIS_MNEMONIC_FMUL);
    const bool issub = (m == ZYDIS_MNEMONIC_FSUB || m == ZYDIS_MNEMONIC_FSUBR);
    const bool isdiv = (m == ZYDIS_MNEMONIC_FDIV || m == ZYDIS_MNEMONIC_FDIVR);
    if (!(isadd || ismul || issub || isdiv)) return false;

    auto do_op = [&](void) {
        // xmm0 = dst, xmm1 = other. Compute dst <- result into xmm0.
        if (isadd)      c.addsd(xmm0, xmm1);
        else if (ismul) c.mulsd(xmm0, xmm1);
        else if (issub) { if (rev) { c.subsd(xmm1, xmm0); c.movsd(xmm0, xmm1); } else c.subsd(xmm0, xmm1); }
        else            { if (rev) { c.divsd(xmm1, xmm0); c.movsd(xmm0, xmm1); } else c.divsd(xmm0, xmm1); }
    };

    // Memory form: dst = ST0, other = mem.
    if (insn.operand_count_visible == 1 && ops[0].type == ZYDIS_OPERAND_TYPE_MEMORY) {
        if (!EmitEffectiveAddress(ops[0].mem, next_rip, c)) return false;
        // rdx = &mem (EmitEffectiveAddress leaves the address in rdx).
        c.mov(r11, rdx);                 // stable mem addr
        const int w = ops[0].size;
        if (w == 32) { c.movss(xmm1, ptr[r11]); c.cvtss2sd(xmm1, xmm1); }
        else         { c.movsd(xmm1, ptr[r11]); }   // m64
        EmitX87RegAddr(c, r8, 0);
        c.movsd(xmm0, ptr[r8]);          // xmm0 = ST0
        do_op();
        c.movsd(ptr[r8], xmm0);
        return true;
    }

    // Register form: two ST operands.
    if (insn.operand_count_visible != 2 ||
        ops[0].type != ZYDIS_OPERAND_TYPE_REGISTER ||
        ops[1].type != ZYDIS_OPERAND_TYPE_REGISTER) return false;
    if (ops[0].reg.value < ZYDIS_REGISTER_ST0 || ops[0].reg.value > ZYDIS_REGISTER_ST7) return false;
    if (ops[1].reg.value < ZYDIS_REGISTER_ST0 || ops[1].reg.value > ZYDIS_REGISTER_ST7) return false;
    const int di = ops[0].reg.value - ZYDIS_REGISTER_ST0;
    const int si = ops[1].reg.value - ZYDIS_REGISTER_ST0;

    EmitX87RegAddr(c, r8, di);
    c.movsd(xmm0, ptr[r8]);              // xmm0 = dst
    EmitX87RegAddr(c, r10, si);
    c.movsd(xmm1, ptr[r10]);             // xmm1 = other
    do_op();
    c.movsd(ptr[r8], xmm0);
    return true;
}

/// FCHS (negate) / FABS / FSQRT on ST(0).
bool EmitFpuUnary(const ZydisDecodedInstruction& insn, Xbyak::CodeGenerator& c) {
    EmitX87RegAddr(c, r8, 0);
    c.movsd(xmm0, ptr[r8]);
    switch (insn.mnemonic) {
        case ZYDIS_MNEMONIC_FCHS: {
            // Flip sign bit (bit 63). xmm1 = mask 0x8000000000000000.
            c.mov(rax, 0x8000000000000000ULL);
            c.movq(xmm1, rax);
            c.xorpd(xmm0, xmm1);
            break;
        }
        case ZYDIS_MNEMONIC_FABS: {
            // Clear sign bit. mask = 0x7FFFFFFFFFFFFFFF.
            c.mov(rax, 0x7FFFFFFFFFFFFFFFULL);
            c.movq(xmm1, rax);
            c.andpd(xmm0, xmm1);
            break;
        }
        case ZYDIS_MNEMONIC_FSQRT: c.sqrtsd(xmm0, xmm0); break;
        default: return false;
    }
    c.movsd(ptr[r8], xmm0);
    return true;
}

/// FLD1 (push 1.0) / FLDZ (push 0.0).
bool EmitFpuLoadConst(const ZydisDecodedInstruction& insn, Xbyak::CodeGenerator& c) {
    u64 bits;
    switch (insn.mnemonic) {
        case ZYDIS_MNEMONIC_FLD1: bits = 0x3FF0000000000000ULL; break;  // 1.0
        case ZYDIS_MNEMONIC_FLDZ: bits = 0x0000000000000000ULL; break;  // 0.0
        default: return false;
    }
    c.mov(rax, bits);
    c.movq(xmm0, rax);
    EmitX87Push(c);
    return true;
}

/// FXCH st(i): swap ST(0) and ST(i). Default i=1.
bool EmitFxch(const ZydisDecodedInstruction& insn, const ZydisDecodedOperand* ops,
              Xbyak::CodeGenerator& c) {
    int i = 1;
    if (insn.operand_count_visible >= 1 && ops[0].type == ZYDIS_OPERAND_TYPE_REGISTER &&
        ops[0].reg.value >= ZYDIS_REGISTER_ST0 && ops[0].reg.value <= ZYDIS_REGISTER_ST7)
        i = ops[0].reg.value - ZYDIS_REGISTER_ST0;
    if (i == 0) return true;   // no-op
    EmitX87RegAddr(c, r8, 0);
    c.movsd(xmm0, ptr[r8]);
    EmitX87RegAddr(c, r10, i);
    c.movsd(xmm1, ptr[r10]);
    c.movsd(ptr[r8], xmm1);
    c.movsd(ptr[r10], xmm0);
    return true;
}

/// x87 compare. FCOMI/FUCOMI(/P) -> EFLAGS ZF/PF/CF; FCOM/FUCOM(/P/PP) ->
/// x87 condition codes C3/C2/C0 in fpu_sw_cc. Comparison is ST(0) vs other.
/// We use the host's native UCOMISD which sets EFLAGS exactly as the x87
/// FCOMI would; for the cc forms we read those EFLAGS back and re-encode
/// them into C3/C2/C0.
bool EmitFpuCompare(const ZydisDecodedInstruction& insn, const ZydisDecodedOperand* ops,
                    Xbyak::CodeGenerator& c) {
    const ZydisMnemonic m = insn.mnemonic;
    const bool to_eflags = (m == ZYDIS_MNEMONIC_FCOMI || m == ZYDIS_MNEMONIC_FUCOMI ||
                            m == ZYDIS_MNEMONIC_FCOMIP || m == ZYDIS_MNEMONIC_FUCOMIP);
    const bool to_cc     = (m == ZYDIS_MNEMONIC_FCOM || m == ZYDIS_MNEMONIC_FCOMP ||
                            m == ZYDIS_MNEMONIC_FCOMPP || m == ZYDIS_MNEMONIC_FUCOM ||
                            m == ZYDIS_MNEMONIC_FUCOMP);
    if (!to_eflags && !to_cc) return false;
    const int npop = (m == ZYDIS_MNEMONIC_FCOMPP) ? 2 :
                     (m == ZYDIS_MNEMONIC_FCOMIP || m == ZYDIS_MNEMONIC_FUCOMIP ||
                      m == ZYDIS_MNEMONIC_FCOMP  || m == ZYDIS_MNEMONIC_FUCOMP) ? 1 : 0;

    int other = 1;
    if (insn.operand_count_visible == 2 && ops[1].type == ZYDIS_OPERAND_TYPE_REGISTER &&
        ops[1].reg.value >= ZYDIS_REGISTER_ST0 && ops[1].reg.value <= ZYDIS_REGISTER_ST7) {
        other = ops[1].reg.value - ZYDIS_REGISTER_ST0;
    } else if (insn.operand_count_visible == 1 && ops[0].type == ZYDIS_OPERAND_TYPE_REGISTER &&
               ops[0].reg.value >= ZYDIS_REGISTER_ST0 && ops[0].reg.value <= ZYDIS_REGISTER_ST7) {
        other = ops[0].reg.value - ZYDIS_REGISTER_ST0;
    }

    EmitX87RegAddr(c, r8, 0);
    c.movsd(xmm0, ptr[r8]);              // ST0
    EmitX87RegAddr(c, r10, other);
    c.movsd(xmm1, ptr[r10]);            // other
    // ucomisd xmm0, xmm1 sets ZF/PF/CF per the FCOMI mapping exactly.
    c.ucomisd(xmm0, xmm1);

    if (to_eflags) {
        // Capture ZF/PF/CF immediately (before any flag-clobbering op) via
        // setcc into bytes, then merge into guest rflags at the same bit
        // positions (guest CF=0, PF=2, ZF=6 — matching x86).
        c.setz(r8b);  c.movzx(r8d, r8b);    // ZF
        c.setp(r9b);  c.movzx(r9d, r9b);    // PF
        c.setc(r10b); c.movzx(r10d, r10b);  // CF
        c.mov(rdx, qword[r13 + offsetof(GuestState, rflags)]);
        c.mov(rax, ~((1ULL<<6)|(1ULL<<0)|(1ULL<<2)|(1ULL<<7)|(1ULL<<11)|(1ULL<<4)));
        c.and_(rdx, rax);
        c.shl(r8d, 6);                       // ZF -> bit6
        c.shl(r9d, 2);                       // PF -> bit2
        // r10d already CF at bit0
        c.or_(rdx, r8);
        c.or_(rdx, r9);
        c.or_(rdx, r10);
        c.mov(qword[r13 + offsetof(GuestState, rflags)], rdx);
    } else {
        // cc form: map ZF/PF/CF (host) -> C3(bit14)/C2(bit10)/C0(bit8).
        // setz/setp/setc into bytes, then place.
        c.setz(al);   c.movzx(eax, al);   // ZF -> eax bit0
        c.setp(cl);   c.movzx(ecx, cl);   // PF -> ecx bit0
        c.setc(dl);   c.movzx(edx, dl);   // CF -> edx bit0
        c.shl(eax, 14);                   // C3
        c.shl(ecx, 10);                   // C2
        c.shl(edx, 8);                    // C0
        c.or_(eax, ecx);
        c.or_(eax, edx);
        // merge into fpu_sw_cc, clearing old C3/C2/C1/C0.
        c.movzx(ecx, word[r13 + offsetof(GuestState, fpu_sw_cc)]);
        c.and_(ecx, ~((1<<14)|(1<<10)|(1<<9)|(1<<8)));
        c.or_(ecx, eax);
        c.mov(word[r13 + offsetof(GuestState, fpu_sw_cc)], cx);
    }

    for (int p = 0; p < npop; ++p) EmitX87PopDiscardValue(c);
    return true;
}

/// FNSTSW ax/m16: status word = fpu_sw_cc | (fpu_top << 11).
bool EmitFnstsw(const ZydisDecodedInstruction& insn, const ZydisDecodedOperand* ops,
                u64 next_rip, Xbyak::CodeGenerator& c) {
    c.movzx(eax, word[r13 + offsetof(GuestState, fpu_sw_cc)]);
    c.mov(ecx, dword[r13 + offsetof(GuestState, fpu_top)]);
    c.shl(ecx, 11);
    c.or_(eax, ecx);
    c.and_(eax, 0xFFFF);
    if (ops[0].type == ZYDIS_OPERAND_TYPE_REGISTER) {
        // AX = guest gpr[0] low 16; preserve bits 63:16.
        c.mov(rdx, qword[r13 + GprOffset(0)]);
        c.mov(rcx, ~0xFFFFULL);
        c.and_(rdx, rcx);
        c.movzx(rax, ax);
        c.or_(rdx, rax);
        c.mov(qword[r13 + GprOffset(0)], rdx);
        return true;
    }
    if (ops[0].type == ZYDIS_OPERAND_TYPE_MEMORY) {
        if (!EmitEffectiveAddress(ops[0].mem, next_rip, c)) return false;
        c.mov(word[rdx], ax);
        return true;
    }
    return false;
}

/// x87 conditional move: FCMOVcc ST(0), ST(i). If the EFLAGS condition holds,
/// copy ST(i) into ST(0); otherwise leave ST(0) unchanged. No pop, no flag
/// change. Modeled on host SSE2 doubles: build the condition into a register,
/// load both slots, and select with a host CMOVcc on integer GPRs holding the
/// raw double bits (movsd<->gpr round-trip). Conditions use guest rflags
/// CF=bit0, PF=bit2, ZF=bit6. Pairs with FCOMI/FUCOMI.
bool EmitFcmov(const ZydisDecodedInstruction& insn, const ZydisDecodedOperand* ops,
               Xbyak::CodeGenerator& c) {
    int si = -1;
    for (int i = insn.operand_count_visible - 1; i >= 0; --i) {
        if (ops[i].type == ZYDIS_OPERAND_TYPE_REGISTER &&
            ops[i].reg.value >= ZYDIS_REGISTER_ST0 &&
            ops[i].reg.value <= ZYDIS_REGISTER_ST7) {
            si = ops[i].reg.value - ZYDIS_REGISTER_ST0;
            break;
        }
    }
    if (si < 0) return false;

    // Compute the two stable slot addresses FIRST: EmitX87RegAddr clobbers rcx,
    // which we use below for the condition, so addresses must precede it.
    EmitX87RegAddr(c, r8, 0);          // r8 = &ST0
    EmitX87RegAddr(c, r11, si);        // r11 = &ST(i)
    c.mov(rax, qword[r8]);             // ST0 raw bits
    c.mov(r9, qword[r11]);             // ST(i) raw bits

    // Build CF/ZF/PF condition into rcx (1 = perform the move).
    c.mov(rdx, qword[r13 + Offsets::Rflags]);
    c.xor_(rcx, rcx);
    switch (insn.mnemonic) {
        case ZYDIS_MNEMONIC_FCMOVB:   c.bt(rdx, 0); c.setc(cl); break;            // CF
        case ZYDIS_MNEMONIC_FCMOVE:   c.bt(rdx, 6); c.setc(cl); break;            // ZF
        case ZYDIS_MNEMONIC_FCMOVU:   c.bt(rdx, 2); c.setc(cl); break;            // PF
        case ZYDIS_MNEMONIC_FCMOVNB:  c.bt(rdx, 0); c.setnc(cl); break;           // !CF
        case ZYDIS_MNEMONIC_FCMOVNE:  c.bt(rdx, 6); c.setnc(cl); break;           // !ZF
        case ZYDIS_MNEMONIC_FCMOVNU:  c.bt(rdx, 2); c.setnc(cl); break;           // !PF
        case ZYDIS_MNEMONIC_FCMOVBE: {                                            // CF|ZF
            c.mov(r10, rdx); c.shr(r10, 6); c.and_(r10, 1);   // ZF
            c.mov(rcx, rdx); c.and_(rcx, 1);                  // CF
            c.or_(rcx, r10); break;
        }
        case ZYDIS_MNEMONIC_FCMOVNBE: {                                           // !(CF|ZF)
            c.mov(r10, rdx); c.shr(r10, 6); c.and_(r10, 1);
            c.mov(rcx, rdx); c.and_(rcx, 1);
            c.or_(rcx, r10); c.xor_(rcx, 1); break;
        }
        default: return false;
    }
    c.and_(rcx, 1);

    // If cond != 0, rax := r9 (ST(i)); else keep ST0. Write back to ST0.
    c.test(rcx, rcx);
    c.cmovne(rax, r9);
    c.mov(qword[r8], rax);
    return true;
}

/// host SSE2 doubles and does not honor the guest's precision-control
/// or rounding-mode bits, so loading the control word has no observable
/// effect; we just consume the operand (compute its address for side
/// effects/faults parity, then ignore the value). FNSTCW reports the
/// architectural default 0x037F.
/// FLDCW m16 — load the x87 control word into GuestState::fpu_cw.
///
/// What is and is not honored, by design:
///   - RC (bits 11:10, rounding control) IS honored — EmitFistp consults it,
///     which covers the classic pre-SSE float->int truncation idiom
///     `fldcw [chop]; fistp; fldcw [restore]`. Dropping it (as an earlier
///     revision did) silently turned every such conversion into
///     round-to-nearest: off-by-one casts all over legacy x87 code.
///   - PC (precision control) is ignored: our x87 registers are 64-bit
///     doubles (see guest_state.h's precision note), so the 24/53/64-bit
///     significand selection has no representation to act on.
///   - Exception masks are ignored: we never raise x87 exceptions.
bool EmitFldcw(const ZydisDecodedInstruction& insn,
               const ZydisDecodedOperand* ops,
               u64 next_rip,
               Xbyak::CodeGenerator& c) {
    if (insn.mnemonic != ZYDIS_MNEMONIC_FLDCW) return false;
    if (ops[0].type != ZYDIS_OPERAND_TYPE_MEMORY) return false;
    if (!EmitEffectiveAddress(ops[0].mem, next_rip, c)) return false;
    c.mov(ax, word[rdx]);
    c.mov(word[r13 + offsetof(GuestState, fpu_cw)], ax);
    return true;
}

/// FNSTCW m16 — store the x87 control word from GuestState::fpu_cw.
/// CallGuest initializes the field to the FNINIT default (0x037F), so a
/// guest that never executed FLDCW reads the architectural reset value;
/// after an FLDCW, the save/restore idiom round-trips the guest's own
/// word instead of a hardcoded constant (which broke the restore half of
/// `fnstcw [save]; fldcw [chop]; ...; fldcw [save]` whenever the guest's
/// ambient cw differed from 0x037F).
bool EmitFnstcw(const ZydisDecodedInstruction& insn,
                const ZydisDecodedOperand* ops,
                u64 next_rip,
                Xbyak::CodeGenerator& c) {
    if (insn.mnemonic != ZYDIS_MNEMONIC_FNSTCW) return false;
    if (ops[0].type != ZYDIS_OPERAND_TYPE_MEMORY) return false;
    if (!EmitEffectiveAddress(ops[0].mem, next_rip, c)) return false;
    c.mov(ax, word[r13 + offsetof(GuestState, fpu_cw)]);
    c.mov(word[rdx], ax);
    return true;
}

/// STMXCSR m32 / LDMXCSR m32 — store/load the SSE control-and-status
/// register.
///
/// STMXCSR reports GuestState::mxcsr (the guest's own last-written value,
/// initialized to the architectural 0x1F80 by CallGuest), so guest fenv
/// save/restore round-trips.
///
/// LDMXCSR does two things: records the raw value in GuestState::mxcsr,
/// and APPLIES a sanitized version to the host MXCSR so the SSE/AVX
/// instructions we emit actually run under the guest's rounding mode and
/// FTZ/DAZ settings. (An earlier revision only recorded the value — guest
/// rounding-mode changes and the near-universal game FTZ enable had zero
/// hardware effect, a silent numeric divergence.) Sanitization keeps RC
/// (14:13), FTZ (15), and DAZ (6) from the guest and forces all exception
/// masks set — a guest unmasking SSE exceptions must not arm host SIGFPE.
/// The dispatcher mirrors this: it restores the host default on every
/// entry (so HLE / host C++ never runs under guest rounding) and
/// re-applies the sanitized guest value before jumping into the next
/// block — see the rounding-swap block in runtime.cpp. The mask and
/// default constants here MUST stay in sync with that code.
bool EmitMxcsr(const ZydisDecodedInstruction& insn,
               const ZydisDecodedOperand* ops,
               u64 next_rip,
               Xbyak::CodeGenerator& c) {
    // VEX-encoded forms (VSTMXCSR/VLDMXCSR) are behaviorally identical to
    // the legacy ones -- same m32 operand, same semantics -- they just
    // decode to distinct Zydis mnemonics.
    const bool store = (insn.mnemonic == ZYDIS_MNEMONIC_STMXCSR ||
                        insn.mnemonic == ZYDIS_MNEMONIC_VSTMXCSR);
    const bool load  = (insn.mnemonic == ZYDIS_MNEMONIC_LDMXCSR ||
                        insn.mnemonic == ZYDIS_MNEMONIC_VLDMXCSR);
    if (!store && !load) return false;
    if (ops[0].type != ZYDIS_OPERAND_TYPE_MEMORY) return false;
    if (!EmitEffectiveAddress(ops[0].mem, next_rip, c)) return false;
    if (store) {
        c.mov(eax, dword[r13 + offsetof(GuestState, mxcsr)]);
        c.mov(dword[rdx], eax);
    } else {
        constexpr u32 kHostDefaultMxcsr = 0x1F80;
        constexpr u32 kGuestMxcsrMask = (3u << 13) | (1u << 15) | (1u << 6);
        constexpr int kScratch0 = static_cast<int>(offsetof(GuestState, scratch));
        c.mov(eax, dword[rdx]);
        c.mov(dword[r13 + offsetof(GuestState, mxcsr)], eax);
        // Sanitize and apply: keep RC/FTZ/DAZ, force exception masks.
        c.and_(eax, kGuestMxcsrMask);
        c.or_(eax, kHostDefaultMxcsr);
        c.mov(dword[r13 + kScratch0], eax);
        c.ldmxcsr(dword[r13 + kScratch0]);
    }
    return true;
}

/// VMOVUPS / VMOVDQU — three forms: reg ← mem, mem ← reg, reg ← reg.
/// Both 128-bit (xmm) and 256-bit (ymm).
///
/// These mnemonics differ only in float-vs-int (and single-vs-double)
/// operand-type hinting on the host CPU, plus the aligned/unaligned
/// hint; the actual bits moved and the upper-zero behavior are
/// identical, and we never differentiate by type or alignment since
/// nothing here observes float-vs-int semantics and the GPR-relayed
/// copy is alignment-agnostic. One emitter covers VMOVUPS/VMOVUPD,
/// VMOVAPS/VMOVAPD, VMOVDQU/VMOVDQA, and the non-temporal VMOVNTDQ(A).
///
/// The vector size comes from `ops[0].size` (in bits): 128 or 256.
/// We deliberately ignore alignment (the U in MOVUPS/MOVDQU =
/// Unaligned; the A forms' #GP-on-misalignment is not modelled).
///
/// ATOMICITY. Guest-memory traffic goes through a host XMM (movups), one
/// 16-byte host op per 16-byte chunk -- NOT a 64-bit GPR relay. The relay
/// (an earlier revision) made every guest vector store visible to other
/// guest threads as two separate 8-byte writes with a multi-instruction
/// window between them. The native form is the conservative direction:
/// aligned 16-byte SSE/AVX accesses are single-copy atomic on every
/// AVX-era host (Intel SDM / AMD APM both document this for cacheable
/// memory), which is strictly FEWER observable interleavings than the
/// original hardware -- the PS4's Jaguar cracked 128-bit accesses into
/// two 64-bit halves, so guest code never had 16-byte atomicity to rely
/// on. Accepted theoretical corner: an 8-aligned-but-not-16-aligned
/// VMOVUPS got two atomic 8-byte halves from the relay (and from Jaguar),
/// while an unaligned host movups carries no sub-access guarantee; code
/// depending on the half-atomicity of deliberately misaligned vector
/// stores would be relying on an accident on real hardware too. The
/// aligned-required forms (VMOVAPS/VMOVDQA/NT) can't hit the corner.
/// Non-temporal forms still lower to plain movups (keeps the
/// LFENCE/SFENCE-no-op reasoning valid: every store we emit is an
/// ordinary store).
bool EmitVmovups(const ZydisDecodedInstruction& insn,
                 const ZydisDecodedOperand* ops,
                 u64 next_rip,
                 Xbyak::CodeGenerator& c) {
    if (insn.mnemonic != ZYDIS_MNEMONIC_VMOVUPS &&
        insn.mnemonic != ZYDIS_MNEMONIC_VMOVUPD &&
        insn.mnemonic != ZYDIS_MNEMONIC_VMOVDQU &&
        insn.mnemonic != ZYDIS_MNEMONIC_VMOVDQA &&
        insn.mnemonic != ZYDIS_MNEMONIC_VMOVAPS &&
        insn.mnemonic != ZYDIS_MNEMONIC_VMOVAPD &&
        insn.mnemonic != ZYDIS_MNEMONIC_VMOVNTDQ &&
        insn.mnemonic != ZYDIS_MNEMONIC_VMOVNTDQA) {
        return false;
    }

    // Vector size: 128 or 256 bits → 2 or 4 chunks of 64 bits.
    int vec_bits = 0;
    if (ops[0].type == ZYDIS_OPERAND_TYPE_REGISTER) {
        vec_bits = ops[0].size;
    } else if (ops[1].type == ZYDIS_OPERAND_TYPE_REGISTER) {
        vec_bits = ops[1].size;
    } else {
        return false;
    }
    if (vec_bits != 128 && vec_bits != 256) return false;
    const int chunks = vec_bits / 64;

    // reg ← mem
    if (ops[0].type == ZYDIS_OPERAND_TYPE_REGISTER &&
        ops[1].type == ZYDIS_OPERAND_TYPE_MEMORY) {
        const int dst_idx = ZydisVecToIndex(ops[0].reg.value);
        if (dst_idx < 0) return false;
        if (!EmitEffectiveAddress(ops[1].mem, next_rip, c)) return false;
        // rdx = src address; xmm0 is the 16-byte transfer register (host
        // XMMs are never live across emitted instructions -- everything
        // round-trips through GuestState -- so xmm0 is free scratch, same
        // convention as EmitFistp).
        for (int i = 0; i < chunks / 2; ++i) {
            c.movups(xmm0, xword[rdx + i * 16]);
            c.movups(xword[r13 + YmmChunkOffset(dst_idx, i * 2)], xmm0);
        }
        // VEX 128-bit form zeros bits 255:128 of the destination YMM.
        // (State is thread-private; no atomicity concern in the zeroing.)
        if (vec_bits == 128) {
            c.xorps(xmm0, xmm0);
            c.movups(xword[r13 + YmmChunkOffset(dst_idx, 2)], xmm0);
        }
        return true;
    }

    // mem ← reg
    if (ops[0].type == ZYDIS_OPERAND_TYPE_MEMORY &&
        ops[1].type == ZYDIS_OPERAND_TYPE_REGISTER) {
        const int src_idx = ZydisVecToIndex(ops[1].reg.value);
        if (src_idx < 0) return false;
        if (!EmitEffectiveAddress(ops[0].mem, next_rip, c)) return false;
        // rdx = dst address; one 16-byte host store per chunk -- this is
        // the direction where the GPR relay was observably torn by
        // concurrent guest threads (see the atomicity note above).
        for (int i = 0; i < chunks / 2; ++i) {
            c.movups(xmm0, xword[r13 + YmmChunkOffset(src_idx, i * 2)]);
            c.movups(xword[rdx + i * 16], xmm0);
        }
        return true;
    }

    // reg ← reg
    if (ops[0].type == ZYDIS_OPERAND_TYPE_REGISTER &&
        ops[1].type == ZYDIS_OPERAND_TYPE_REGISTER) {
        const int dst_idx = ZydisVecToIndex(ops[0].reg.value);
        const int src_idx = ZydisVecToIndex(ops[1].reg.value);
        if (dst_idx < 0 || src_idx < 0) return false;
        for (int i = 0; i < chunks / 2; ++i) {
            c.movups(xmm0, xword[r13 + YmmChunkOffset(src_idx, i * 2)]);
            c.movups(xword[r13 + YmmChunkOffset(dst_idx, i * 2)], xmm0);
        }
        if (vec_bits == 128) {
            c.xorps(xmm0, xmm0);
            c.movups(xword[r13 + YmmChunkOffset(dst_idx, 2)], xmm0);
        }
        return true;
    }

    return false;
}

/// VMOVD — 32-bit move between the low 32 bits of an XMM and a GPR or
/// memory. All AVX-encoded (VEX prefix), so every XMM-destination form
/// also zeros bits 255:32 of the destination YMM — critical for
/// correctness, since leaving stale upper bits would corrupt later
/// wide-vector reads of the same register.
///
/// Encodings handled:
///   - `vmovd r32, xmm`   : GPR ← low 32 of XMM
///   - `vmovd xmm, r32`   : low 32 of XMM ← GPR   (rest of YMM zeroed)
///   - `vmovd xmm, m32`   : low 32 of XMM ← memory (rest of YMM zeroed)
///   - `vmovd m32, xmm`   : memory ← low 32 of XMM (exactly 4 bytes)
bool EmitVmovd(const ZydisDecodedInstruction& insn,
               const ZydisDecodedOperand* ops,
               u64 next_rip,
               Xbyak::CodeGenerator& c) {
    if (insn.mnemonic != ZYDIS_MNEMONIC_VMOVD) return false;

    // Reg-reg forms: GPR <-> XMM low 32.
    if (ops[0].type == ZYDIS_OPERAND_TYPE_REGISTER &&
        ops[1].type == ZYDIS_OPERAND_TYPE_REGISTER) {
        const int dst_gpr = ZydisGprToIndex(ops[0].reg.value);
        const int src_gpr = ZydisGprToIndex(ops[1].reg.value);
        const int dst_vec = ZydisVecToIndex(ops[0].reg.value);
        const int src_vec = ZydisVecToIndex(ops[1].reg.value);

        // GPR ← XMM low 32 bits.
        if (dst_gpr >= 0 && src_vec >= 0) {
            c.mov(eax, dword[r13 + YmmChunkOffset(src_vec, 0)]);
            // Writing eax zero-extends rax; store full qword so the
            // guest GPR slot picks up the canonical zero-extension.
            c.mov(qword[r13 + GprOffset(dst_gpr)], rax);
            return true;
        }

        // XMM ← GPR low 32 bits, with upper YMM zeroed (VEX-encoded).
        if (dst_vec >= 0 && src_gpr >= 0) {
            // Going through rax (32-bit load sets bits 63:32 to 0, then
            // qword store) covers chunk 0 cleanly; zero chunks 1..3.
            c.mov(eax, dword[r13 + GprOffset(src_gpr)]);
            c.mov(qword[r13 + YmmChunkOffset(dst_vec, 0)], rax);
            c.xor_(rax, rax);
            c.mov(qword[r13 + YmmChunkOffset(dst_vec, 1)], rax);
            c.mov(qword[r13 + YmmChunkOffset(dst_vec, 2)], rax);
            c.mov(qword[r13 + YmmChunkOffset(dst_vec, 3)], rax);
            return true;
        }

        return false;
    }

    // XMM ← memory (load low 32, zero the rest of the YMM).
    if (ops[0].type == ZYDIS_OPERAND_TYPE_REGISTER &&
        ops[1].type == ZYDIS_OPERAND_TYPE_MEMORY) {
        const int dst_vec = ZydisVecToIndex(ops[0].reg.value);
        if (dst_vec < 0) return false;
        if (!EmitEffectiveAddress(ops[1].mem, next_rip, c)) return false;
        c.mov(eax, dword[rdx]);   // 32-bit load zero-extends into rax
        c.mov(qword[r13 + YmmChunkOffset(dst_vec, 0)], rax);
        c.xor_(rax, rax);
        c.mov(qword[r13 + YmmChunkOffset(dst_vec, 1)], rax);
        c.mov(qword[r13 + YmmChunkOffset(dst_vec, 2)], rax);
        c.mov(qword[r13 + YmmChunkOffset(dst_vec, 3)], rax);
        return true;
    }

    // memory ← XMM (store low 32 bits, exactly 4 bytes).
    if (ops[0].type == ZYDIS_OPERAND_TYPE_MEMORY &&
        ops[1].type == ZYDIS_OPERAND_TYPE_REGISTER) {
        const int src_vec = ZydisVecToIndex(ops[1].reg.value);
        if (src_vec < 0) return false;
        // Compute EA first (it clobbers rax/rdx), stash &dst, then load value.
        if (!EmitEffectiveAddress(ops[0].mem, next_rip, c)) return false;
        c.mov(r10, rdx);                          // r10 = &dst
        c.mov(eax, dword[r13 + YmmChunkOffset(src_vec, 0)]);
        c.mov(dword[r10], eax);
        return true;
    }

    return false;
}

/// VMOVQ — 64-bit move involving the low 64 bits of an XMM register.
/// Four forms (all VEX-encoded, so XMM destinations zero bits 255:64):
///   - `vmovq r64, xmm`   : GPR ← low 64 of XMM
///   - `vmovq xmm, r64`   : low 64 of XMM ← GPR (rest of YMM zeroed)
///   - `vmovq xmm, [mem]` : low 64 of XMM ← memory (rest zeroed)
///   - `vmovq [mem], xmm` : memory ← low 64 of XMM (8 bytes only)
/// We move the 64-bit value directly between guest GPR/YMM slots (and
/// memory) with plain GPR moves — no host SIMD needed.
bool EmitVmovq(const ZydisDecodedInstruction& insn,
               const ZydisDecodedOperand* ops,
               u64 next_rip,
               Xbyak::CodeGenerator& c) {
    if (insn.mnemonic != ZYDIS_MNEMONIC_VMOVQ &&
        insn.mnemonic != ZYDIS_MNEMONIC_MOVQ) {
        return false;
    }

    // Reg-reg forms: GPR<->XMM.
    if (ops[0].type == ZYDIS_OPERAND_TYPE_REGISTER &&
        ops[1].type == ZYDIS_OPERAND_TYPE_REGISTER) {
        const int dst_gpr = ZydisGprToIndex(ops[0].reg.value);
        const int src_gpr = ZydisGprToIndex(ops[1].reg.value);
        const int dst_vec = ZydisVecToIndex(ops[0].reg.value);
        const int src_vec = ZydisVecToIndex(ops[1].reg.value);

        // GPR ← XMM low 64.
        if (dst_gpr >= 0 && src_vec >= 0) {
            c.mov(rax, qword[r13 + YmmChunkOffset(src_vec, 0)]);
            c.mov(qword[r13 + GprOffset(dst_gpr)], rax);
            return true;
        }
        // XMM ← GPR low 64, upper YMM zeroed.
        if (dst_vec >= 0 && src_gpr >= 0) {
            c.mov(rax, qword[r13 + GprOffset(src_gpr)]);
            c.mov(qword[r13 + YmmChunkOffset(dst_vec, 0)], rax);
            c.xor_(rax, rax);
            c.mov(qword[r13 + YmmChunkOffset(dst_vec, 1)], rax);
            c.mov(qword[r13 + YmmChunkOffset(dst_vec, 2)], rax);
            c.mov(qword[r13 + YmmChunkOffset(dst_vec, 3)], rax);
            return true;
        }
        // XMM ← XMM (low 64, upper zeroed) — also a legal vmovq form.
        if (dst_vec >= 0 && src_vec >= 0) {
            c.mov(rax, qword[r13 + YmmChunkOffset(src_vec, 0)]);
            c.mov(qword[r13 + YmmChunkOffset(dst_vec, 0)], rax);
            c.xor_(rax, rax);
            c.mov(qword[r13 + YmmChunkOffset(dst_vec, 1)], rax);
            c.mov(qword[r13 + YmmChunkOffset(dst_vec, 2)], rax);
            c.mov(qword[r13 + YmmChunkOffset(dst_vec, 3)], rax);
            return true;
        }
        return false;
    }

    // XMM ← memory (load low 64, zero the rest of the YMM).
    if (ops[0].type == ZYDIS_OPERAND_TYPE_REGISTER &&
        ops[1].type == ZYDIS_OPERAND_TYPE_MEMORY) {
        const int dst_vec = ZydisVecToIndex(ops[0].reg.value);
        if (dst_vec < 0) return false;
        if (!EmitEffectiveAddress(ops[1].mem, next_rip, c)) return false;
        c.mov(rax, qword[rdx]);
        c.mov(qword[r13 + YmmChunkOffset(dst_vec, 0)], rax);
        c.xor_(rax, rax);
        c.mov(qword[r13 + YmmChunkOffset(dst_vec, 1)], rax);
        c.mov(qword[r13 + YmmChunkOffset(dst_vec, 2)], rax);
        c.mov(qword[r13 + YmmChunkOffset(dst_vec, 3)], rax);
        return true;
    }

    // memory ← XMM (store low 64, exactly 8 bytes).
    if (ops[0].type == ZYDIS_OPERAND_TYPE_MEMORY &&
        ops[1].type == ZYDIS_OPERAND_TYPE_REGISTER) {
        const int src_vec = ZydisVecToIndex(ops[1].reg.value);
        if (src_vec < 0) return false;
        // Load the value first (EmitEffectiveAddress clobbers rax).
        if (!EmitEffectiveAddress(ops[0].mem, next_rip, c)) return false;
        c.mov(r10, rdx);                          // r10 = &dst
        c.mov(rax, qword[r13 + YmmChunkOffset(src_vec, 0)]);
        c.mov(qword[r10], rax);
        return true;
    }

    return false;
}

/// VMOVSS / VMOVSD — scalar single/double float moves. Despite the FP
/// naming these are pure bit moves (no arithmetic, no rounding), so we
/// implement them with plain integer moves between guest YMM slots and
/// memory. ELEM = 32 (ss) or 64 (sd) bits is the scalar element size.
///
/// Three forms:
///   Load  `vmovss/sd xmm, [mem]`  (2 visible ops, op1 = mem):
///       dst.elem = [mem]; ALL other bits of the dst (rest of chunk 0
///       for ss, chunk 1, and the upper YMM) are zeroed.
///   Store `vmovss/sd [mem], xmm`  (2 visible ops, op0 = mem):
///       [mem] = src.elem (only ELEM bits written).
///   Merge `vmovss/sd xmm0, xmm1, xmm2` (3 visible reg ops):
///       dst.elem            = src2.elem
///       dst[127:ELEM]       = src1[127:ELEM]   (rest of low 128 lane)
///       dst[255:128]        = 0                (VEX upper-zero)
/// The 2-op register-register form (`vmovss xmm0, xmm1`) is a
/// load-style move that, for the reg-reg case, behaves like the merge
/// with src1 == dst; it is not observed yet and folds into the 3-op
/// path if it appears with operand_count_visible == 2 and a reg src.
bool EmitVmovss(const ZydisDecodedInstruction& insn,
                const ZydisDecodedOperand* ops,
                u64 next_rip,
                Xbyak::CodeGenerator& c) {
    const bool ss = (insn.mnemonic == ZYDIS_MNEMONIC_VMOVSS);
    const bool sd = (insn.mnemonic == ZYDIS_MNEMONIC_VMOVSD);
    if (!ss && !sd) return false;

    // Merge form: 3 visible register operands.
    if (insn.operand_count_visible == 3 &&
        ops[0].type == ZYDIS_OPERAND_TYPE_REGISTER &&
        ops[1].type == ZYDIS_OPERAND_TYPE_REGISTER &&
        ops[2].type == ZYDIS_OPERAND_TYPE_REGISTER) {
        const int dst = ZydisVecToIndex(ops[0].reg.value);
        const int s1  = ZydisVecToIndex(ops[1].reg.value);
        const int s2  = ZydisVecToIndex(ops[2].reg.value);
        if (dst < 0 || s1 < 0 || s2 < 0) return false;

        if (sd) {
            // chunk0 = src2.chunk0 ; chunk1 = src1.chunk1
            c.mov(rax, qword[r13 + YmmChunkOffset(s2, 0)]);
            c.mov(rcx, qword[r13 + YmmChunkOffset(s1, 1)]);
            c.mov(qword[r13 + YmmChunkOffset(dst, 0)], rax);
            c.mov(qword[r13 + YmmChunkOffset(dst, 1)], rcx);
        } else {
            // chunk0 = (src1.hi32 << 32) | src2.lo32 ; chunk1 = src1.chunk1
            // Keep src1 high32, replace low32 with src2 low32.
            c.mov(rcx, qword[r13 + YmmChunkOffset(s1, 0)]);
            c.shr(rcx, 32);
            c.shl(rcx, 32);                                   // rcx = src1.hi32 << 32
            c.mov(eax, dword[r13 + YmmChunkOffset(s2, 0)]);   // eax = src2.lo32 (zero-ext)
            c.or_(rax, rcx);
            c.mov(qword[r13 + YmmChunkOffset(dst, 0)], rax);
            c.mov(rcx, qword[r13 + YmmChunkOffset(s1, 1)]);   // chunk1 = src1.chunk1
            c.mov(qword[r13 + YmmChunkOffset(dst, 1)], rcx);
        }
        c.xor_(rax, rax);
        c.mov(qword[r13 + YmmChunkOffset(dst, 2)], rax);
        c.mov(qword[r13 + YmmChunkOffset(dst, 3)], rax);
        return true;
    }

    // Load: xmm <- [mem] (op0 reg, op1 mem).
    if (ops[0].type == ZYDIS_OPERAND_TYPE_REGISTER &&
        ops[1].type == ZYDIS_OPERAND_TYPE_MEMORY) {
        const int dst = ZydisVecToIndex(ops[0].reg.value);
        if (dst < 0) return false;
        if (!EmitEffectiveAddress(ops[1].mem, next_rip, c)) return false;
        if (sd) {
            c.mov(rax, qword[rdx]);
            c.mov(qword[r13 + YmmChunkOffset(dst, 0)], rax);
        } else {
            c.mov(eax, dword[rdx]);                           // zero-extends rax
            c.mov(qword[r13 + YmmChunkOffset(dst, 0)], rax);
        }
        c.xor_(rax, rax);
        c.mov(qword[r13 + YmmChunkOffset(dst, 1)], rax);
        c.mov(qword[r13 + YmmChunkOffset(dst, 2)], rax);
        c.mov(qword[r13 + YmmChunkOffset(dst, 3)], rax);
        return true;
    }

    // Store: [mem] <- xmm (op0 mem, op1 reg).
    if (ops[0].type == ZYDIS_OPERAND_TYPE_MEMORY &&
        ops[1].type == ZYDIS_OPERAND_TYPE_REGISTER) {
        const int src = ZydisVecToIndex(ops[1].reg.value);
        if (src < 0) return false;
        if (!EmitEffectiveAddress(ops[0].mem, next_rip, c)) return false;
        c.mov(r10, rdx);                                      // r10 = &dst
        if (sd) {
            c.mov(rax, qword[r13 + YmmChunkOffset(src, 0)]);
            c.mov(qword[r10], rax);
        } else {
            c.mov(eax, dword[r13 + YmmChunkOffset(src, 0)]);
            c.mov(dword[r10], eax);
        }
        return true;
    }

    return false;
}

/// VCVTSI2SS / VCVTSI2SD — convert a scalar integer (32- or 64-bit,
/// from a GPR or memory) to scalar single/double float, merging into
/// the low element of the destination XMM while taking the rest of the
/// low 128-bit lane from src1. This is the first emitter that needs
/// real host FP: we run the host VCVTSI2SS/SD on a scratch xmm so the
/// IEEE-754 conversion (and host MXCSR rounding) is exact.
///
/// Strategy: stage src1's full low-128 into host xmm0 (so its upper 96
/// bits, ss, or upper 64, sd, survive the convert via the host
/// instruction's own merge), load the integer into a host GPR at the
/// correct width, run `vcvtsi2ss/sd xmm0, xmm0, <gpr>`, then store
/// xmm0 back to the guest dst and zero the upper YMM (VEX-128).
///
/// Operands: 3 visible (dst, src1, int-src) is the AVX form; the
/// folded 2-visible (dst=src1, int-src) is also accepted.
bool EmitVcvtsi2(const ZydisDecodedInstruction& insn,
                 const ZydisDecodedOperand* ops,
                 u64 next_rip,
                 Xbyak::CodeGenerator& c) {
    const bool ss = (insn.mnemonic == ZYDIS_MNEMONIC_VCVTSI2SS);
    const bool sd = (insn.mnemonic == ZYDIS_MNEMONIC_VCVTSI2SD);
    if (!ss && !sd) return false;

    int dst_idx, src1_idx;
    const ZydisDecodedOperand* isrc = nullptr;
    if (insn.operand_count_visible == 3) {
        if (ops[0].type != ZYDIS_OPERAND_TYPE_REGISTER ||
            ops[1].type != ZYDIS_OPERAND_TYPE_REGISTER) return false;
        dst_idx  = ZydisVecToIndex(ops[0].reg.value);
        src1_idx = ZydisVecToIndex(ops[1].reg.value);
        isrc     = &ops[2];
    } else if (insn.operand_count_visible == 2) {
        if (ops[0].type != ZYDIS_OPERAND_TYPE_REGISTER) return false;
        dst_idx  = ZydisVecToIndex(ops[0].reg.value);
        src1_idx = dst_idx;
        isrc     = &ops[1];
    } else {
        return false;
    }
    if (dst_idx < 0 || src1_idx < 0) return false;

    // Integer source width: 64 if the operand is 64 bits, else 32.
    const bool src64 = (isrc->size == 64);

    // Stage src1 into host xmm0 (carries the bits to preserve).
    c.vmovdqu(xmm0, ptr[r13 + YmmChunkOffset(src1_idx, 0)]);

    // Load the integer source into rcx at the right width.
    if (isrc->type == ZYDIS_OPERAND_TYPE_REGISTER) {
        const int gi = ZydisGprToIndex(isrc->reg.value);
        if (gi < 0) return false;
        if (src64) c.mov(rcx, qword[r13 + GprOffset(gi)]);
        else       c.mov(ecx, dword[r13 + GprOffset(gi)]);
    } else if (isrc->type == ZYDIS_OPERAND_TYPE_MEMORY) {
        if (!EmitEffectiveAddress(isrc->mem, next_rip, c)) return false;
        if (src64) c.mov(rcx, qword[rdx]);
        else       c.mov(ecx, dword[rdx]);
    } else {
        return false;
    }

    // Convert into xmm0's low element; host merge keeps xmm0's upper.
    if (ss) {
        if (src64) c.vcvtsi2ss(xmm0, xmm0, rcx);
        else       c.vcvtsi2ss(xmm0, xmm0, ecx);
    } else {
        if (src64) c.vcvtsi2sd(xmm0, xmm0, rcx);
        else       c.vcvtsi2sd(xmm0, xmm0, ecx);
    }

    // Store back to guest dst, zero the upper YMM (VEX-128).
    c.vmovdqu(ptr[r13 + YmmChunkOffset(dst_idx, 0)], xmm0);
    c.xor_(rax, rax);
    c.mov(qword[r13 + YmmChunkOffset(dst_idx, 2)], rax);
    c.mov(qword[r13 + YmmChunkOffset(dst_idx, 3)], rax);
    return true;
}

/// VCVTTSS2SI / VCVTTSD2SI (VEX, truncating) and the non-VEX SSE forms
/// CVTSS2SI / CVTSD2SI (rounding, honoring MXCSR) and CVTTSS2SI /
/// CVTTSD2SI (truncating) — convert a scalar single/double float to a
/// signed integer, writing the result to a GPR. The destination width
/// (32 or 64) is the GPR operand size; a 32-bit write zero-extends bits
/// 63:32 per x86-64. Out-of-range and NaN inputs yield the "integer
/// indefinite" value (INT_MIN at the dst width). The truncating forms
/// always round toward zero regardless of MXCSR; the non-truncating
/// CVTSS2SI/CVTSD2SI honor the current MXCSR rounding mode. We run the
/// matching host instruction on a staged xmm (or memory) and store the
/// result, so the host reproduces all of this exactly. The VEX and
/// non-VEX encodings are semantically identical for these (the non-VEX
/// forms leave the upper FP state alone, which we don't model anyway).
bool EmitVcvtt2si(const ZydisDecodedInstruction& insn,
                  const ZydisDecodedOperand* ops,
                  u64 next_rip,
                  Xbyak::CodeGenerator& c) {
    const ZydisMnemonic m = insn.mnemonic;
    bool ss, sd, trunc;
    switch (m) {
        case ZYDIS_MNEMONIC_VCVTTSS2SI: ss = true;  sd = false; trunc = true;  break;
        case ZYDIS_MNEMONIC_VCVTTSD2SI: ss = false; sd = true;  trunc = true;  break;
        case ZYDIS_MNEMONIC_CVTTSS2SI:  ss = true;  sd = false; trunc = true;  break;
        case ZYDIS_MNEMONIC_CVTTSD2SI:  ss = false; sd = true;  trunc = true;  break;
        case ZYDIS_MNEMONIC_CVTSS2SI:   ss = true;  sd = false; trunc = false; break;
        case ZYDIS_MNEMONIC_CVTSD2SI:   ss = false; sd = true;  trunc = false; break;
        default: return false;
    }
    (void)ss; (void)sd;
    if (ops[0].type != ZYDIS_OPERAND_TYPE_REGISTER) return false;

    const int dst_gpr = ZydisGprToIndex(ops[0].reg.value);
    if (dst_gpr < 0) return false;
    const bool dst64 = (ops[0].size == 64);

    // Stage the FP source into host xmm1 (register) or read from memory.
    bool src_mem = false;
    if (ops[1].type == ZYDIS_OPERAND_TYPE_REGISTER) {
        const int s = ZydisVecToIndex(ops[1].reg.value);
        if (s < 0) return false;
        c.vmovdqu(xmm1, ptr[r13 + YmmChunkOffset(s, 0)]);
    } else if (ops[1].type == ZYDIS_OPERAND_TYPE_MEMORY) {
        if (!EmitEffectiveAddress(ops[1].mem, next_rip, c)) return false;
        src_mem = true;
    } else {
        return false;
    }

    // Convert into rax/eax with the matching host instruction. We always
    // use the VEX-encoded host mnemonics (cvt... vs vcvt...) — they are
    // architecturally equivalent and avoid an SSE/AVX transition penalty
    // on the host. Single vs double picks dword vs qword memory width.
    if (trunc) {
        if (ss) {
            if (dst64) { if (src_mem) c.vcvttss2si(rax, dword[rdx]); else c.vcvttss2si(rax, xmm1); }
            else       { if (src_mem) c.vcvttss2si(eax, dword[rdx]); else c.vcvttss2si(eax, xmm1); }
        } else {
            if (dst64) { if (src_mem) c.vcvttsd2si(rax, qword[rdx]); else c.vcvttsd2si(rax, xmm1); }
            else       { if (src_mem) c.vcvttsd2si(eax, qword[rdx]); else c.vcvttsd2si(eax, xmm1); }
        }
    } else {
        if (ss) {
            if (dst64) { if (src_mem) c.vcvtss2si(rax, dword[rdx]); else c.vcvtss2si(rax, xmm1); }
            else       { if (src_mem) c.vcvtss2si(eax, dword[rdx]); else c.vcvtss2si(eax, xmm1); }
        } else {
            if (dst64) { if (src_mem) c.vcvtsd2si(rax, qword[rdx]); else c.vcvtsd2si(rax, xmm1); }
            else       { if (src_mem) c.vcvtsd2si(eax, qword[rdx]); else c.vcvtsd2si(eax, xmm1); }
        }
    }

    // Store result. 32-bit dst: writing eax already zeroed bits 63:32
    // of rax, so a qword store carries the x86-64 zero-extension.
    c.mov(qword[r13 + GprOffset(dst_gpr)], rax);
    return true;
}

/// VUCOMISS / VUCOMISD / VCOMISS / VCOMISD — scalar FP compare that
/// writes EFLAGS rather than producing a value. The result truth table
/// (per Intel): GT → ZF=PF=CF=0; LT → CF=1; EQ → ZF=1; UNORDERED (NaN)
/// → ZF=PF=CF=1. OF/SF/AF are always cleared. The (U)COMISS pair differ
/// only in which NaN classes raise the invalid exception, which we
/// don't surface, so they behave identically here.
///
/// We must touch ONLY the arithmetic flag bits and preserve the rest of
/// guest rflags (e.g. IF). So: clear the arith bits (CF/PF/AF/ZF/SF/OF)
/// in the guest rflags, run the host compare on staged xmm regs, take
/// just those bits from the host result, and OR them in.
bool EmitFpCompare(const ZydisDecodedInstruction& insn,
                   const ZydisDecodedOperand* ops,
                   u64 next_rip,
                   Xbyak::CodeGenerator& c) {
    const ZydisMnemonic m = insn.mnemonic;
    bool dbl;
    switch (m) {
        case ZYDIS_MNEMONIC_VUCOMISS: case ZYDIS_MNEMONIC_VCOMISS: dbl = false; break;
        case ZYDIS_MNEMONIC_VUCOMISD: case ZYDIS_MNEMONIC_VCOMISD: dbl = true;  break;
        default: return false;
    }
    const bool ucomi = (m == ZYDIS_MNEMONIC_VUCOMISS ||
                        m == ZYDIS_MNEMONIC_VUCOMISD);
    if (ops[0].type != ZYDIS_OPERAND_TYPE_REGISTER) return false;
    const int a_idx = ZydisVecToIndex(ops[0].reg.value);
    if (a_idx < 0) return false;

    // Stage op0 (the lhs) into xmm0.
    c.vmovdqu(xmm0, ptr[r13 + YmmChunkOffset(a_idx, 0)]);

    // Stage op1 (rhs) into xmm1 or take it from memory.
    bool rhs_mem = false;
    if (ops[1].type == ZYDIS_OPERAND_TYPE_REGISTER) {
        const int b_idx = ZydisVecToIndex(ops[1].reg.value);
        if (b_idx < 0) return false;
        c.vmovdqu(xmm1, ptr[r13 + YmmChunkOffset(b_idx, 0)]);
    } else if (ops[1].type == ZYDIS_OPERAND_TYPE_MEMORY) {
        if (!EmitEffectiveAddress(ops[1].mem, next_rip, c)) return false;
        rhs_mem = true;
    } else {
        return false;
    }

    // The flag bits this instruction defines (all cleared except as set
    // by the compare): CF(0) PF(2) AF(4) ZF(6) SF(7) OF(11).
    constexpr u64 kFpCmpMask =
        (1ULL << 0) | (1ULL << 2) | (1ULL << 4) |
        (1ULL << 6) | (1ULL << 7) | (1ULL << 11);

    // Clear those bits in the guest rflags (preserve the rest).
    c.mov(r9, ~kFpCmpMask);
    c.and_(qword[r13 + Offsets::Rflags], r9);

    // Run the host compare; it sets ZF/PF/CF and clears OF/SF/AF.
    if (ucomi) {
        if (dbl) { if (rhs_mem) c.vucomisd(xmm0, qword[rdx]); else c.vucomisd(xmm0, xmm1); }
        else     { if (rhs_mem) c.vucomiss(xmm0, dword[rdx]); else c.vucomiss(xmm0, xmm1); }
    } else {
        if (dbl) { if (rhs_mem) c.vcomisd(xmm0, qword[rdx]); else c.vcomisd(xmm0, xmm1); }
        else     { if (rhs_mem) c.vcomiss(xmm0, dword[rdx]); else c.vcomiss(xmm0, xmm1); }
    }

    // Capture host flags, keep only the FP-compare bits, OR them in.
    c.pushfq();
    c.pop(r8);
    c.mov(r9, kFpCmpMask);
    c.and_(r8, r9);
    c.or_(qword[r13 + Offsets::Rflags], r8);
    return true;
}

/// VCVTSS2SD / VCVTSD2SS — convert a scalar single<->double float,
/// merging into the destination's low element while taking the rest of
/// the low 128-bit lane from src1:
///   ss2sd: dst.lo64 = (double)src2.f32; dst[127:64] = src1[127:64].
///   sd2ss: dst.lo32 = (float)src2.f64;  dst[127:32] = src1[127:32].
/// Upper YMM is zeroed (VEX-128). The host conversion (and its MXCSR
/// rounding for the narrowing sd2ss case) is exact; we stage src1 into
/// xmm0 so the host op's own merge preserves the right upper bits.
///
/// Operands: 3 visible (dst, src1, src2) — src2 reg or mem — and the
/// folded 2-visible (dst=src1, src2).
bool EmitVcvtScalar(const ZydisDecodedInstruction& insn,
                    const ZydisDecodedOperand* ops,
                    u64 next_rip,
                    Xbyak::CodeGenerator& c) {
    const bool ss2sd = (insn.mnemonic == ZYDIS_MNEMONIC_VCVTSS2SD);
    const bool sd2ss = (insn.mnemonic == ZYDIS_MNEMONIC_VCVTSD2SS);
    if (!ss2sd && !sd2ss) return false;

    int dst_idx, src1_idx;
    const ZydisDecodedOperand* src2 = nullptr;
    if (insn.operand_count_visible == 3) {
        if (ops[0].type != ZYDIS_OPERAND_TYPE_REGISTER ||
            ops[1].type != ZYDIS_OPERAND_TYPE_REGISTER) return false;
        dst_idx  = ZydisVecToIndex(ops[0].reg.value);
        src1_idx = ZydisVecToIndex(ops[1].reg.value);
        src2     = &ops[2];
    } else if (insn.operand_count_visible == 2) {
        if (ops[0].type != ZYDIS_OPERAND_TYPE_REGISTER) return false;
        dst_idx  = ZydisVecToIndex(ops[0].reg.value);
        src1_idx = dst_idx;
        src2     = &ops[1];
    } else {
        return false;
    }
    if (dst_idx < 0 || src1_idx < 0) return false;

    // Stage src1 into xmm0 (carries the lane bits to preserve).
    c.vmovdqu(xmm0, ptr[r13 + YmmChunkOffset(src1_idx, 0)]);

    bool src2_mem = false;
    if (src2->type == ZYDIS_OPERAND_TYPE_REGISTER) {
        const int s2 = ZydisVecToIndex(src2->reg.value);
        if (s2 < 0) return false;
        c.vmovdqu(xmm1, ptr[r13 + YmmChunkOffset(s2, 0)]);
    } else if (src2->type == ZYDIS_OPERAND_TYPE_MEMORY) {
        if (!EmitEffectiveAddress(src2->mem, next_rip, c)) return false;
        src2_mem = true;
    } else {
        return false;
    }

    if (ss2sd) {
        if (src2_mem) c.vcvtss2sd(xmm0, xmm0, dword[rdx]);  // source is f32
        else          c.vcvtss2sd(xmm0, xmm0, xmm1);
    } else {
        if (src2_mem) c.vcvtsd2ss(xmm0, xmm0, qword[rdx]);  // source is f64
        else          c.vcvtsd2ss(xmm0, xmm0, xmm1);
    }

    c.vmovdqu(ptr[r13 + YmmChunkOffset(dst_idx, 0)], xmm0);
    c.xor_(rax, rax);
    c.mov(qword[r13 + YmmChunkOffset(dst_idx, 2)], rax);
    c.mov(qword[r13 + YmmChunkOffset(dst_idx, 3)], rax);
    return true;
}

/// Scalar FP binary arithmetic: VMULSS/SD, VADDSS/SD, VSUBSS/SD,
/// VDIVSS/SD, VMINSS, VMAXSS. All share the same shape: compute
/// `dst.elem = src1.elem <op> src2.elem`, with the rest of the low
/// 128-bit lane (dst[127:ELEM]) taken from src1 and the upper YMM
/// zeroed (VEX-128). src2 may be a register or memory.
///
/// We use real host FP: stage src1's full low-128 into host xmm0,
/// stage src2 into xmm1 (or read it from memory), and run the matching
/// host instruction `vop xmm0, xmm0, xmm1`. The host op writes only the
/// low element and preserves xmm0's upper bits, which is exactly the
/// architectural merge — so a 128-bit storeback of xmm0 is correct.
/// Operand order is src1 <op> src2 (e.g. vsubss = src1 - src2).
bool EmitScalarFpArith(const ZydisDecodedInstruction& insn,
                       const ZydisDecodedOperand* ops,
                       u64 next_rip,
                       Xbyak::CodeGenerator& c) {
    const ZydisMnemonic m = insn.mnemonic;
    bool ss; // true = single (32-bit elem), false = double (64-bit)
    switch (m) {
        case ZYDIS_MNEMONIC_VMULSS: case ZYDIS_MNEMONIC_VADDSS:
        case ZYDIS_MNEMONIC_VSUBSS: case ZYDIS_MNEMONIC_VDIVSS:
        case ZYDIS_MNEMONIC_VMINSS: case ZYDIS_MNEMONIC_VMAXSS:
            ss = true; break;
        case ZYDIS_MNEMONIC_VMULSD: case ZYDIS_MNEMONIC_VADDSD:
        case ZYDIS_MNEMONIC_VSUBSD: case ZYDIS_MNEMONIC_VDIVSD:
        case ZYDIS_MNEMONIC_VMINSD: case ZYDIS_MNEMONIC_VMAXSD:
            ss = false; break;
        default:
            return false;
    }
    (void)ss;

    int dst_idx, src1_idx;
    const ZydisDecodedOperand* src2 = nullptr;
    if (insn.operand_count_visible == 3) {
        if (ops[0].type != ZYDIS_OPERAND_TYPE_REGISTER ||
            ops[1].type != ZYDIS_OPERAND_TYPE_REGISTER) return false;
        dst_idx  = ZydisVecToIndex(ops[0].reg.value);
        src1_idx = ZydisVecToIndex(ops[1].reg.value);
        src2     = &ops[2];
    } else if (insn.operand_count_visible == 2) {
        if (ops[0].type != ZYDIS_OPERAND_TYPE_REGISTER) return false;
        dst_idx  = ZydisVecToIndex(ops[0].reg.value);
        src1_idx = dst_idx;
        src2     = &ops[1];
    } else {
        return false;
    }
    if (dst_idx < 0 || src1_idx < 0) return false;

    // Stage src1 into xmm0 (carries the lane bits to preserve).
    c.vmovdqu(xmm0, ptr[r13 + YmmChunkOffset(src1_idx, 0)]);

    // Stage src2 into xmm1 (register) or read scalar from memory.
    bool src2_mem = false;
    if (src2->type == ZYDIS_OPERAND_TYPE_REGISTER) {
        const int s2 = ZydisVecToIndex(src2->reg.value);
        if (s2 < 0) return false;
        c.vmovdqu(xmm1, ptr[r13 + YmmChunkOffset(s2, 0)]);
    } else if (src2->type == ZYDIS_OPERAND_TYPE_MEMORY) {
        if (!EmitEffectiveAddress(src2->mem, next_rip, c)) return false;
        src2_mem = true;
    } else {
        return false;
    }

    // Run the host scalar-FP op: xmm0 = xmm0 <op> (xmm1 | [rdx]).
    // The memory forms take the address operand directly; only the low
    // element is read by a scalar op, so a dword/qword deref is safe.
    if (src2_mem) {
        switch (m) {
            case ZYDIS_MNEMONIC_VMULSS: c.vmulss(xmm0, xmm0, dword[rdx]); break;
            case ZYDIS_MNEMONIC_VADDSS: c.vaddss(xmm0, xmm0, dword[rdx]); break;
            case ZYDIS_MNEMONIC_VSUBSS: c.vsubss(xmm0, xmm0, dword[rdx]); break;
            case ZYDIS_MNEMONIC_VDIVSS: c.vdivss(xmm0, xmm0, dword[rdx]); break;
            case ZYDIS_MNEMONIC_VMINSS: c.vminss(xmm0, xmm0, dword[rdx]); break;
            case ZYDIS_MNEMONIC_VMAXSS: c.vmaxss(xmm0, xmm0, dword[rdx]); break;
            case ZYDIS_MNEMONIC_VMULSD: c.vmulsd(xmm0, xmm0, qword[rdx]); break;
            case ZYDIS_MNEMONIC_VADDSD: c.vaddsd(xmm0, xmm0, qword[rdx]); break;
            case ZYDIS_MNEMONIC_VSUBSD: c.vsubsd(xmm0, xmm0, qword[rdx]); break;
            case ZYDIS_MNEMONIC_VDIVSD: c.vdivsd(xmm0, xmm0, qword[rdx]); break;
            case ZYDIS_MNEMONIC_VMINSD: c.vminsd(xmm0, xmm0, qword[rdx]); break;
            case ZYDIS_MNEMONIC_VMAXSD: c.vmaxsd(xmm0, xmm0, qword[rdx]); break;
            default: return false;
        }
    } else {
        switch (m) {
            case ZYDIS_MNEMONIC_VMULSS: c.vmulss(xmm0, xmm0, xmm1); break;
            case ZYDIS_MNEMONIC_VADDSS: c.vaddss(xmm0, xmm0, xmm1); break;
            case ZYDIS_MNEMONIC_VSUBSS: c.vsubss(xmm0, xmm0, xmm1); break;
            case ZYDIS_MNEMONIC_VDIVSS: c.vdivss(xmm0, xmm0, xmm1); break;
            case ZYDIS_MNEMONIC_VMINSS: c.vminss(xmm0, xmm0, xmm1); break;
            case ZYDIS_MNEMONIC_VMAXSS: c.vmaxss(xmm0, xmm0, xmm1); break;
            case ZYDIS_MNEMONIC_VMULSD: c.vmulsd(xmm0, xmm0, xmm1); break;
            case ZYDIS_MNEMONIC_VADDSD: c.vaddsd(xmm0, xmm0, xmm1); break;
            case ZYDIS_MNEMONIC_VSUBSD: c.vsubsd(xmm0, xmm0, xmm1); break;
            case ZYDIS_MNEMONIC_VDIVSD: c.vdivsd(xmm0, xmm0, xmm1); break;
            case ZYDIS_MNEMONIC_VMINSD: c.vminsd(xmm0, xmm0, xmm1); break;
            case ZYDIS_MNEMONIC_VMAXSD: c.vmaxsd(xmm0, xmm0, xmm1); break;
            default: return false;
        }
    }

    // Store xmm0 back to the guest dst; zero the upper YMM (VEX-128).
    c.vmovdqu(ptr[r13 + YmmChunkOffset(dst_idx, 0)], xmm0);
    c.xor_(rax, rax);
    c.mov(qword[r13 + YmmChunkOffset(dst_idx, 2)], rax);
    c.mov(qword[r13 + YmmChunkOffset(dst_idx, 3)], rax);
    return true;
}

/// Packed FP binary arithmetic: VMULPS/PD, VADDPS/PD, VSUBPS/PD,
/// VDIVPS/PD, VMINPS/PD, VMAXPS/PD. Unlike the scalar *ss/*sd ops,
/// these operate on EVERY lane of the vector (no merge), so the whole
/// 128/256-bit result is computed. We stage src1 into host xmm0/ymm0,
/// run the host packed op against src2 (register or full-width
/// memory), and store back. 128-bit VEX zeroes bits 255:128.
/// Operand order is src1 <op> src2 (e.g. vsubps = src1 - src2).
bool EmitPackedFpArith(const ZydisDecodedInstruction& insn,
                       const ZydisDecodedOperand* ops,
                       u64 next_rip,
                       Xbyak::CodeGenerator& c) {
    const ZydisMnemonic m = insn.mnemonic;
    switch (m) {
        case ZYDIS_MNEMONIC_VMULPS: case ZYDIS_MNEMONIC_VADDPS:
        case ZYDIS_MNEMONIC_VSUBPS: case ZYDIS_MNEMONIC_VDIVPS:
        case ZYDIS_MNEMONIC_VMINPS: case ZYDIS_MNEMONIC_VMAXPS:
        case ZYDIS_MNEMONIC_VMULPD: case ZYDIS_MNEMONIC_VADDPD:
        case ZYDIS_MNEMONIC_VSUBPD: case ZYDIS_MNEMONIC_VDIVPD:
        case ZYDIS_MNEMONIC_VMINPD: case ZYDIS_MNEMONIC_VMAXPD:
        // Horizontal add/sub and alternating add/sub. Same 3-operand binary
        // shape and host staging; the hardware computes the cross-lane result.
        case ZYDIS_MNEMONIC_VHADDPS: case ZYDIS_MNEMONIC_VHADDPD:
        case ZYDIS_MNEMONIC_VHSUBPS: case ZYDIS_MNEMONIC_VHSUBPD:
        case ZYDIS_MNEMONIC_VADDSUBPS: case ZYDIS_MNEMONIC_VADDSUBPD:
            break;
        default: return false;
    }

    int dst_idx, src1_idx;
    const ZydisDecodedOperand* src2 = nullptr;
    if (insn.operand_count_visible == 3) {
        if (ops[0].type != ZYDIS_OPERAND_TYPE_REGISTER ||
            ops[1].type != ZYDIS_OPERAND_TYPE_REGISTER) return false;
        dst_idx  = ZydisVecToIndex(ops[0].reg.value);
        src1_idx = ZydisVecToIndex(ops[1].reg.value);
        src2     = &ops[2];
    } else if (insn.operand_count_visible == 2) {
        if (ops[0].type != ZYDIS_OPERAND_TYPE_REGISTER) return false;
        dst_idx  = ZydisVecToIndex(ops[0].reg.value);
        src1_idx = dst_idx;
        src2     = &ops[1];
    } else {
        return false;
    }
    if (dst_idx < 0 || src1_idx < 0) return false;

    const int vec_bits = ops[0].size;
    if (vec_bits != 128 && vec_bits != 256) return false;
    const bool ymm = (vec_bits == 256);

    // Stage src1 into xmm0/ymm0.
    if (ymm) c.vmovdqu(ymm0, ptr[r13 + YmmChunkOffset(src1_idx, 0)]);
    else     c.vmovdqu(xmm0, ptr[r13 + YmmChunkOffset(src1_idx, 0)]);

    bool src2_mem = false;
    if (src2->type == ZYDIS_OPERAND_TYPE_REGISTER) {
        const int s2 = ZydisVecToIndex(src2->reg.value);
        if (s2 < 0) return false;
        if (ymm) c.vmovdqu(ymm1, ptr[r13 + YmmChunkOffset(s2, 0)]);
        else     c.vmovdqu(xmm1, ptr[r13 + YmmChunkOffset(s2, 0)]);
    } else if (src2->type == ZYDIS_OPERAND_TYPE_MEMORY) {
        if (!EmitEffectiveAddress(src2->mem, next_rip, c)) return false;
        if (ymm) c.vmovdqu(ymm1, ptr[rdx]);
        else     c.vmovdqu(xmm1, ptr[rdx]);
        src2_mem = true;
    } else {
        return false;
    }
    (void)src2_mem;

    if (ymm) {
        switch (m) {
            case ZYDIS_MNEMONIC_VMULPS: c.vmulps(ymm0, ymm0, ymm1); break;
            case ZYDIS_MNEMONIC_VADDPS: c.vaddps(ymm0, ymm0, ymm1); break;
            case ZYDIS_MNEMONIC_VSUBPS: c.vsubps(ymm0, ymm0, ymm1); break;
            case ZYDIS_MNEMONIC_VDIVPS: c.vdivps(ymm0, ymm0, ymm1); break;
            case ZYDIS_MNEMONIC_VMINPS: c.vminps(ymm0, ymm0, ymm1); break;
            case ZYDIS_MNEMONIC_VMAXPS: c.vmaxps(ymm0, ymm0, ymm1); break;
            case ZYDIS_MNEMONIC_VMULPD: c.vmulpd(ymm0, ymm0, ymm1); break;
            case ZYDIS_MNEMONIC_VADDPD: c.vaddpd(ymm0, ymm0, ymm1); break;
            case ZYDIS_MNEMONIC_VSUBPD: c.vsubpd(ymm0, ymm0, ymm1); break;
            case ZYDIS_MNEMONIC_VDIVPD: c.vdivpd(ymm0, ymm0, ymm1); break;
            case ZYDIS_MNEMONIC_VMINPD: c.vminpd(ymm0, ymm0, ymm1); break;
            case ZYDIS_MNEMONIC_VMAXPD: c.vmaxpd(ymm0, ymm0, ymm1); break;
            case ZYDIS_MNEMONIC_VHADDPS: c.vhaddps(ymm0, ymm0, ymm1); break;
            case ZYDIS_MNEMONIC_VHADDPD: c.vhaddpd(ymm0, ymm0, ymm1); break;
            case ZYDIS_MNEMONIC_VHSUBPS: c.vhsubps(ymm0, ymm0, ymm1); break;
            case ZYDIS_MNEMONIC_VHSUBPD: c.vhsubpd(ymm0, ymm0, ymm1); break;
            case ZYDIS_MNEMONIC_VADDSUBPS: c.vaddsubps(ymm0, ymm0, ymm1); break;
            case ZYDIS_MNEMONIC_VADDSUBPD: c.vaddsubpd(ymm0, ymm0, ymm1); break;
            default: return false;
        }
        c.vmovdqu(ptr[r13 + YmmChunkOffset(dst_idx, 0)], ymm0);
    } else {
        switch (m) {
            case ZYDIS_MNEMONIC_VMULPS: c.vmulps(xmm0, xmm0, xmm1); break;
            case ZYDIS_MNEMONIC_VADDPS: c.vaddps(xmm0, xmm0, xmm1); break;
            case ZYDIS_MNEMONIC_VSUBPS: c.vsubps(xmm0, xmm0, xmm1); break;
            case ZYDIS_MNEMONIC_VDIVPS: c.vdivps(xmm0, xmm0, xmm1); break;
            case ZYDIS_MNEMONIC_VMINPS: c.vminps(xmm0, xmm0, xmm1); break;
            case ZYDIS_MNEMONIC_VMAXPS: c.vmaxps(xmm0, xmm0, xmm1); break;
            case ZYDIS_MNEMONIC_VMULPD: c.vmulpd(xmm0, xmm0, xmm1); break;
            case ZYDIS_MNEMONIC_VADDPD: c.vaddpd(xmm0, xmm0, xmm1); break;
            case ZYDIS_MNEMONIC_VSUBPD: c.vsubpd(xmm0, xmm0, xmm1); break;
            case ZYDIS_MNEMONIC_VDIVPD: c.vdivpd(xmm0, xmm0, xmm1); break;
            case ZYDIS_MNEMONIC_VMINPD: c.vminpd(xmm0, xmm0, xmm1); break;
            case ZYDIS_MNEMONIC_VMAXPD: c.vmaxpd(xmm0, xmm0, xmm1); break;
            case ZYDIS_MNEMONIC_VHADDPS: c.vhaddps(xmm0, xmm0, xmm1); break;
            case ZYDIS_MNEMONIC_VHADDPD: c.vhaddpd(xmm0, xmm0, xmm1); break;
            case ZYDIS_MNEMONIC_VHSUBPS: c.vhsubps(xmm0, xmm0, xmm1); break;
            case ZYDIS_MNEMONIC_VHSUBPD: c.vhsubpd(xmm0, xmm0, xmm1); break;
            case ZYDIS_MNEMONIC_VADDSUBPS: c.vaddsubps(xmm0, xmm0, xmm1); break;
            case ZYDIS_MNEMONIC_VADDSUBPD: c.vaddsubpd(xmm0, xmm0, xmm1); break;
            default: return false;
        }
        c.vmovdqu(ptr[r13 + YmmChunkOffset(dst_idx, 0)], xmm0);
        c.xor_(rax, rax);
        c.mov(qword[r13 + YmmChunkOffset(dst_idx, 2)], rax);
        c.mov(qword[r13 + YmmChunkOffset(dst_idx, 3)], rax);
    }
    return true;
}

/// Unary packed-FP: VSQRTPS/VSQRTPD (IEEE square root) and VRSQRTPS/VRCPPS
/// (~12-bit hardware approximations of 1/sqrt(x) and 1/x; PS-only, no PD
/// form). Single source (register or full-width memory), single dest; same
/// host staging as the packed-FP arith family. The approximate ops emit the
/// host instruction directly, so the result is the host CPU's estimate — the
/// low bits may differ from the PS4's Jaguar, but code that uses RCP/RSQRT
/// expects an approximation, so this is the standard and acceptable behavior.
bool EmitPackedFpUnary(const ZydisDecodedInstruction& insn,
                       const ZydisDecodedOperand* ops,
                       u64 next_rip,
                       Xbyak::CodeGenerator& c) {
    const ZydisMnemonic m = insn.mnemonic;
    switch (m) {
        case ZYDIS_MNEMONIC_VSQRTPS:
        case ZYDIS_MNEMONIC_VSQRTPD:
        case ZYDIS_MNEMONIC_VRSQRTPS:
        case ZYDIS_MNEMONIC_VRCPPS:
            break;
        default: return false;
    }
    if (ops[0].type != ZYDIS_OPERAND_TYPE_REGISTER) return false;
    const int dst_idx = ZydisVecToIndex(ops[0].reg.value);
    if (dst_idx < 0) return false;

    const int vec_bits = ops[0].size;
    if (vec_bits != 128 && vec_bits != 256) return false;
    const bool ymm = (vec_bits == 256);

    // Stage the single source into xmm1/ymm1 (register or full-width memory).
    if (ops[1].type == ZYDIS_OPERAND_TYPE_REGISTER) {
        const int s = ZydisVecToIndex(ops[1].reg.value);
        if (s < 0) return false;
        if (ymm) c.vmovdqu(ymm1, ptr[r13 + YmmChunkOffset(s, 0)]);
        else     c.vmovdqu(xmm1, ptr[r13 + YmmChunkOffset(s, 0)]);
    } else if (ops[1].type == ZYDIS_OPERAND_TYPE_MEMORY) {
        if (!EmitEffectiveAddress(ops[1].mem, next_rip, c)) return false;
        if (ymm) c.vmovdqu(ymm1, ptr[rdx]);
        else     c.vmovdqu(xmm1, ptr[rdx]);
    } else {
        return false;
    }

    if (ymm) {
        switch (m) {
            case ZYDIS_MNEMONIC_VSQRTPS:  c.vsqrtps(ymm0, ymm1);  break;
            case ZYDIS_MNEMONIC_VSQRTPD:  c.vsqrtpd(ymm0, ymm1);  break;
            case ZYDIS_MNEMONIC_VRSQRTPS: c.vrsqrtps(ymm0, ymm1); break;
            case ZYDIS_MNEMONIC_VRCPPS:   c.vrcpps(ymm0, ymm1);   break;
            default: return false;
        }
        c.vmovdqu(ptr[r13 + YmmChunkOffset(dst_idx, 0)], ymm0);
    } else {
        switch (m) {
            case ZYDIS_MNEMONIC_VSQRTPS:  c.vsqrtps(xmm0, xmm1);  break;
            case ZYDIS_MNEMONIC_VSQRTPD:  c.vsqrtpd(xmm0, xmm1);  break;
            case ZYDIS_MNEMONIC_VRSQRTPS: c.vrsqrtps(xmm0, xmm1); break;
            case ZYDIS_MNEMONIC_VRCPPS:   c.vrcpps(xmm0, xmm1);   break;
            default: return false;
        }
        c.vmovdqu(ptr[r13 + YmmChunkOffset(dst_idx, 0)], xmm0);
        c.xor_(rax, rax);
        c.mov(qword[r13 + YmmChunkOffset(dst_idx, 2)], rax);
        c.mov(qword[r13 + YmmChunkOffset(dst_idx, 3)], rax);
    }
    return true;
}
/// compares VPCMPEQD/VPCMPGTD and the widening unsigned multiply VPMULUDQ.
/// Per-lane with no carry/borrow crossing lane boundaries — the host
/// instruction enforces that. VPMULUDQ is the odd one out in granularity:
/// it multiplies the even-indexed 32-bit lanes (dwords 0,2,...) producing
/// 64-bit results in each qword lane; but it has the same 2-source operand
/// shape and the same host staging, and the hardware computes the exact
/// semantics, so it fits this template unchanged. Same host-staging as the
/// packed-FP family: stage src1 into xmm0/ymm0, run the host op against src2
/// (reg or full-width memory), store back, zero the upper YMM for VEX-128.
bool EmitPackedIntArith(const ZydisDecodedInstruction& insn,
                        const ZydisDecodedOperand* ops,
                        u64 next_rip,
                        Xbyak::CodeGenerator& c) {
    const ZydisMnemonic m = insn.mnemonic;
    switch (m) {
        case ZYDIS_MNEMONIC_VPADDD:
        case ZYDIS_MNEMONIC_VPSUBD:
        case ZYDIS_MNEMONIC_VPADDQ:
        case ZYDIS_MNEMONIC_VPSUBQ:
        case ZYDIS_MNEMONIC_VPMULLD:
        case ZYDIS_MNEMONIC_VPMULUDQ:
        case ZYDIS_MNEMONIC_VPMINUD:
        case ZYDIS_MNEMONIC_VPMINSD:
        case ZYDIS_MNEMONIC_VPMAXUD:
        case ZYDIS_MNEMONIC_VPMAXSD:
        case ZYDIS_MNEMONIC_VPCMPEQD:
        case ZYDIS_MNEMONIC_VPCMPGTD:
        case ZYDIS_MNEMONIC_VPCMPEQQ:
        case ZYDIS_MNEMONIC_VPCMPGTQ:
        // Byte/word-width variants of the above (same template; the hardware
        // supplies the per-element semantics).
        case ZYDIS_MNEMONIC_VPADDB:
        case ZYDIS_MNEMONIC_VPADDW:
        case ZYDIS_MNEMONIC_VPSUBB:
        case ZYDIS_MNEMONIC_VPSUBW:
        case ZYDIS_MNEMONIC_VPMULLW:
        case ZYDIS_MNEMONIC_VPMINUB:
        case ZYDIS_MNEMONIC_VPMINSB:
        case ZYDIS_MNEMONIC_VPMINUW:
        case ZYDIS_MNEMONIC_VPMINSW:
        case ZYDIS_MNEMONIC_VPMAXUB:
        case ZYDIS_MNEMONIC_VPMAXSB:
        case ZYDIS_MNEMONIC_VPMAXUW:
        case ZYDIS_MNEMONIC_VPMAXSW:
        case ZYDIS_MNEMONIC_VPCMPEQW:
        case ZYDIS_MNEMONIC_VPCMPGTB:
        case ZYDIS_MNEMONIC_VPCMPGTW:
            break;
        default: return false;
    }

    int dst_idx, src1_idx;
    const ZydisDecodedOperand* src2 = nullptr;
    if (insn.operand_count_visible == 3) {
        if (ops[0].type != ZYDIS_OPERAND_TYPE_REGISTER ||
            ops[1].type != ZYDIS_OPERAND_TYPE_REGISTER) return false;
        dst_idx  = ZydisVecToIndex(ops[0].reg.value);
        src1_idx = ZydisVecToIndex(ops[1].reg.value);
        src2     = &ops[2];
    } else if (insn.operand_count_visible == 2) {
        if (ops[0].type != ZYDIS_OPERAND_TYPE_REGISTER) return false;
        dst_idx  = ZydisVecToIndex(ops[0].reg.value);
        src1_idx = dst_idx;
        src2     = &ops[1];
    } else {
        return false;
    }
    if (dst_idx < 0 || src1_idx < 0) return false;

    const int vec_bits = ops[0].size;
    if (vec_bits != 128 && vec_bits != 256) return false;
    const bool ymm = (vec_bits == 256);

    if (ymm) c.vmovdqu(ymm0, ptr[r13 + YmmChunkOffset(src1_idx, 0)]);
    else     c.vmovdqu(xmm0, ptr[r13 + YmmChunkOffset(src1_idx, 0)]);

    if (src2->type == ZYDIS_OPERAND_TYPE_REGISTER) {
        const int s2 = ZydisVecToIndex(src2->reg.value);
        if (s2 < 0) return false;
        if (ymm) c.vmovdqu(ymm1, ptr[r13 + YmmChunkOffset(s2, 0)]);
        else     c.vmovdqu(xmm1, ptr[r13 + YmmChunkOffset(s2, 0)]);
    } else if (src2->type == ZYDIS_OPERAND_TYPE_MEMORY) {
        if (!EmitEffectiveAddress(src2->mem, next_rip, c)) return false;
        if (ymm) c.vmovdqu(ymm1, ptr[rdx]);
        else     c.vmovdqu(xmm1, ptr[rdx]);
    } else {
        return false;
    }

    if (ymm) {
        switch (m) {
            case ZYDIS_MNEMONIC_VPADDD:  c.vpaddd(ymm0, ymm0, ymm1);  break;
            case ZYDIS_MNEMONIC_VPSUBD:  c.vpsubd(ymm0, ymm0, ymm1);  break;
            case ZYDIS_MNEMONIC_VPADDQ:  c.vpaddq(ymm0, ymm0, ymm1);  break;
            case ZYDIS_MNEMONIC_VPSUBQ:  c.vpsubq(ymm0, ymm0, ymm1);  break;
            case ZYDIS_MNEMONIC_VPMULLD: c.vpmulld(ymm0, ymm0, ymm1); break;
            case ZYDIS_MNEMONIC_VPMULUDQ: c.vpmuludq(ymm0, ymm0, ymm1); break;
            case ZYDIS_MNEMONIC_VPMINUD:  c.vpminud(ymm0, ymm0, ymm1);  break;
            case ZYDIS_MNEMONIC_VPMINSD:  c.vpminsd(ymm0, ymm0, ymm1);  break;
            case ZYDIS_MNEMONIC_VPMAXUD:  c.vpmaxud(ymm0, ymm0, ymm1);  break;
            case ZYDIS_MNEMONIC_VPMAXSD:  c.vpmaxsd(ymm0, ymm0, ymm1);  break;
            case ZYDIS_MNEMONIC_VPCMPEQD: c.vpcmpeqd(ymm0, ymm0, ymm1); break;
            case ZYDIS_MNEMONIC_VPCMPGTD: c.vpcmpgtd(ymm0, ymm0, ymm1); break;
            case ZYDIS_MNEMONIC_VPCMPEQQ: c.vpcmpeqq(ymm0, ymm0, ymm1); break;
            case ZYDIS_MNEMONIC_VPCMPGTQ: c.vpcmpgtq(ymm0, ymm0, ymm1); break;
            case ZYDIS_MNEMONIC_VPADDB:   c.vpaddb(ymm0, ymm0, ymm1);   break;
            case ZYDIS_MNEMONIC_VPADDW:   c.vpaddw(ymm0, ymm0, ymm1);   break;
            case ZYDIS_MNEMONIC_VPSUBB:   c.vpsubb(ymm0, ymm0, ymm1);   break;
            case ZYDIS_MNEMONIC_VPSUBW:   c.vpsubw(ymm0, ymm0, ymm1);   break;
            case ZYDIS_MNEMONIC_VPMULLW:  c.vpmullw(ymm0, ymm0, ymm1);  break;
            case ZYDIS_MNEMONIC_VPMINUB:  c.vpminub(ymm0, ymm0, ymm1);  break;
            case ZYDIS_MNEMONIC_VPMINSB:  c.vpminsb(ymm0, ymm0, ymm1);  break;
            case ZYDIS_MNEMONIC_VPMINUW:  c.vpminuw(ymm0, ymm0, ymm1);  break;
            case ZYDIS_MNEMONIC_VPMINSW:  c.vpminsw(ymm0, ymm0, ymm1);  break;
            case ZYDIS_MNEMONIC_VPMAXUB:  c.vpmaxub(ymm0, ymm0, ymm1);  break;
            case ZYDIS_MNEMONIC_VPMAXSB:  c.vpmaxsb(ymm0, ymm0, ymm1);  break;
            case ZYDIS_MNEMONIC_VPMAXUW:  c.vpmaxuw(ymm0, ymm0, ymm1);  break;
            case ZYDIS_MNEMONIC_VPMAXSW:  c.vpmaxsw(ymm0, ymm0, ymm1);  break;
            case ZYDIS_MNEMONIC_VPCMPEQW: c.vpcmpeqw(ymm0, ymm0, ymm1); break;
            case ZYDIS_MNEMONIC_VPCMPGTB: c.vpcmpgtb(ymm0, ymm0, ymm1); break;
            case ZYDIS_MNEMONIC_VPCMPGTW: c.vpcmpgtw(ymm0, ymm0, ymm1); break;
            default: return false;
        }
        c.vmovdqu(ptr[r13 + YmmChunkOffset(dst_idx, 0)], ymm0);
    } else {
        switch (m) {
            case ZYDIS_MNEMONIC_VPADDD:  c.vpaddd(xmm0, xmm0, xmm1);  break;
            case ZYDIS_MNEMONIC_VPSUBD:  c.vpsubd(xmm0, xmm0, xmm1);  break;
            case ZYDIS_MNEMONIC_VPADDQ:  c.vpaddq(xmm0, xmm0, xmm1);  break;
            case ZYDIS_MNEMONIC_VPSUBQ:  c.vpsubq(xmm0, xmm0, xmm1);  break;
            case ZYDIS_MNEMONIC_VPMULLD: c.vpmulld(xmm0, xmm0, xmm1); break;
            case ZYDIS_MNEMONIC_VPMULUDQ: c.vpmuludq(xmm0, xmm0, xmm1); break;
            case ZYDIS_MNEMONIC_VPMINUD:  c.vpminud(xmm0, xmm0, xmm1);  break;
            case ZYDIS_MNEMONIC_VPMINSD:  c.vpminsd(xmm0, xmm0, xmm1);  break;
            case ZYDIS_MNEMONIC_VPMAXUD:  c.vpmaxud(xmm0, xmm0, xmm1);  break;
            case ZYDIS_MNEMONIC_VPMAXSD:  c.vpmaxsd(xmm0, xmm0, xmm1);  break;
            case ZYDIS_MNEMONIC_VPCMPEQD: c.vpcmpeqd(xmm0, xmm0, xmm1); break;
            case ZYDIS_MNEMONIC_VPCMPGTD: c.vpcmpgtd(xmm0, xmm0, xmm1); break;
            case ZYDIS_MNEMONIC_VPCMPEQQ: c.vpcmpeqq(xmm0, xmm0, xmm1); break;
            case ZYDIS_MNEMONIC_VPCMPGTQ: c.vpcmpgtq(xmm0, xmm0, xmm1); break;
            case ZYDIS_MNEMONIC_VPADDB:   c.vpaddb(xmm0, xmm0, xmm1);   break;
            case ZYDIS_MNEMONIC_VPADDW:   c.vpaddw(xmm0, xmm0, xmm1);   break;
            case ZYDIS_MNEMONIC_VPSUBB:   c.vpsubb(xmm0, xmm0, xmm1);   break;
            case ZYDIS_MNEMONIC_VPSUBW:   c.vpsubw(xmm0, xmm0, xmm1);   break;
            case ZYDIS_MNEMONIC_VPMULLW:  c.vpmullw(xmm0, xmm0, xmm1);  break;
            case ZYDIS_MNEMONIC_VPMINUB:  c.vpminub(xmm0, xmm0, xmm1);  break;
            case ZYDIS_MNEMONIC_VPMINSB:  c.vpminsb(xmm0, xmm0, xmm1);  break;
            case ZYDIS_MNEMONIC_VPMINUW:  c.vpminuw(xmm0, xmm0, xmm1);  break;
            case ZYDIS_MNEMONIC_VPMINSW:  c.vpminsw(xmm0, xmm0, xmm1);  break;
            case ZYDIS_MNEMONIC_VPMAXUB:  c.vpmaxub(xmm0, xmm0, xmm1);  break;
            case ZYDIS_MNEMONIC_VPMAXSB:  c.vpmaxsb(xmm0, xmm0, xmm1);  break;
            case ZYDIS_MNEMONIC_VPMAXUW:  c.vpmaxuw(xmm0, xmm0, xmm1);  break;
            case ZYDIS_MNEMONIC_VPMAXSW:  c.vpmaxsw(xmm0, xmm0, xmm1);  break;
            case ZYDIS_MNEMONIC_VPCMPEQW: c.vpcmpeqw(xmm0, xmm0, xmm1); break;
            case ZYDIS_MNEMONIC_VPCMPGTB: c.vpcmpgtb(xmm0, xmm0, xmm1); break;
            case ZYDIS_MNEMONIC_VPCMPGTW: c.vpcmpgtw(xmm0, xmm0, xmm1); break;
            default: return false;
        }
        c.vmovdqu(ptr[r13 + YmmChunkOffset(dst_idx, 0)], xmm0);
        c.xor_(rax, rax);
        c.mov(qword[r13 + YmmChunkOffset(dst_idx, 2)], rax);
        c.mov(qword[r13 + YmmChunkOffset(dst_idx, 3)], rax);
    }
    return true;
}

// Small helpers shared by the vector-op batch below.
namespace {
// Resolve a 3-op-or-folded-2-op vector instruction's operands.
// Returns false if shapes are unsupported. On success: dst_idx, src1_idx
// set; *src2 points at the src2 operand (reg or mem).
inline bool ResolveVec3(const ZydisDecodedInstruction& insn,
                        const ZydisDecodedOperand* ops,
                        int& dst_idx, int& src1_idx,
                        const ZydisDecodedOperand*& src2) {
    if (insn.operand_count_visible == 3) {
        if (ops[0].type != ZYDIS_OPERAND_TYPE_REGISTER ||
            ops[1].type != ZYDIS_OPERAND_TYPE_REGISTER) return false;
        dst_idx  = ZydisVecToIndex(ops[0].reg.value);
        src1_idx = ZydisVecToIndex(ops[1].reg.value);
        src2     = &ops[2];
    } else if (insn.operand_count_visible == 2) {
        if (ops[0].type != ZYDIS_OPERAND_TYPE_REGISTER) return false;
        dst_idx  = ZydisVecToIndex(ops[0].reg.value);
        src1_idx = dst_idx;
        src2     = &ops[1];
    } else {
        return false;
    }
    return dst_idx >= 0 && src1_idx >= 0;
}
inline void ZeroUpperYmm(int dst_idx, Xbyak::CodeGenerator& c) {
    c.xor_(rax, rax);
    c.mov(qword[r13 + YmmChunkOffset(dst_idx, 2)], rax);
    c.mov(qword[r13 + YmmChunkOffset(dst_idx, 3)], rax);
}
} // namespace

/// Host-staged vector ops with no immediate, sharing the 3-op/folded-2-op
/// shape: VUNPCKLPS/HPS/LPD/HPD, VPACKUSDW, VMOVLHPS, VMOVHLPS, VSQRTSD.
/// (VSQRTSD/VMOVLHPS/VMOVHLPS are scalar/partial but use the same staging;
/// the host instruction's own semantics produce the right merge.)
bool EmitVecHostStaged(const ZydisDecodedInstruction& insn,
                       const ZydisDecodedOperand* ops,
                       u64 next_rip,
                       Xbyak::CodeGenerator& c) {
    const ZydisMnemonic m = insn.mnemonic;
    switch (m) {
        case ZYDIS_MNEMONIC_VUNPCKLPS: case ZYDIS_MNEMONIC_VUNPCKHPS:
        case ZYDIS_MNEMONIC_VUNPCKLPD: case ZYDIS_MNEMONIC_VUNPCKHPD:
        case ZYDIS_MNEMONIC_VPACKUSDW:
        case ZYDIS_MNEMONIC_VMOVLHPS:  case ZYDIS_MNEMONIC_VMOVHLPS:
        case ZYDIS_MNEMONIC_VSQRTSD:
        case ZYDIS_MNEMONIC_VSQRTSS:
        case ZYDIS_MNEMONIC_VRSQRTSS:
        case ZYDIS_MNEMONIC_VRCPSS:
            break;
        default: return false;
    }
    int dst_idx, src1_idx;
    const ZydisDecodedOperand* src2 = nullptr;
    if (!ResolveVec3(insn, ops, dst_idx, src1_idx, src2)) return false;

    const int vec_bits = ops[0].size;
    const bool ymm = (vec_bits == 256);
    if (vec_bits != 128 && vec_bits != 256) {
        // scalar (sd) ops report 128; treat <256 as xmm.
    }

    if (ymm) c.vmovdqu(ymm0, ptr[r13 + YmmChunkOffset(src1_idx, 0)]);
    else     c.vmovdqu(xmm0, ptr[r13 + YmmChunkOffset(src1_idx, 0)]);

    bool mem = false;
    if (src2->type == ZYDIS_OPERAND_TYPE_REGISTER) {
        const int s2 = ZydisVecToIndex(src2->reg.value);
        if (s2 < 0) return false;
        if (ymm) c.vmovdqu(ymm1, ptr[r13 + YmmChunkOffset(s2, 0)]);
        else     c.vmovdqu(xmm1, ptr[r13 + YmmChunkOffset(s2, 0)]);
    } else if (src2->type == ZYDIS_OPERAND_TYPE_MEMORY) {
        if (!EmitEffectiveAddress(src2->mem, next_rip, c)) return false;
        if (ymm) c.vmovdqu(ymm1, ptr[rdx]);
        else     c.vmovdqu(xmm1, ptr[rdx]);
        mem = true;
    } else {
        return false;
    }
    (void)mem;

#define VEC_OP(opname) \
    do { if (ymm) c.opname(ymm0, ymm0, ymm1); else c.opname(xmm0, xmm0, xmm1); } while (0)
    switch (m) {
        case ZYDIS_MNEMONIC_VUNPCKLPS: VEC_OP(vunpcklps); break;
        case ZYDIS_MNEMONIC_VUNPCKHPS: VEC_OP(vunpckhps); break;
        case ZYDIS_MNEMONIC_VUNPCKLPD: VEC_OP(vunpcklpd); break;
        case ZYDIS_MNEMONIC_VUNPCKHPD: VEC_OP(vunpckhpd); break;
        case ZYDIS_MNEMONIC_VPACKUSDW: VEC_OP(vpackusdw); break;
        case ZYDIS_MNEMONIC_VMOVLHPS:  c.vmovlhps(xmm0, xmm0, xmm1); break;
        case ZYDIS_MNEMONIC_VMOVHLPS:  c.vmovhlps(xmm0, xmm0, xmm1); break;
        case ZYDIS_MNEMONIC_VSQRTSD:   c.vsqrtsd(xmm0, xmm0, xmm1); break;
        case ZYDIS_MNEMONIC_VSQRTSS:   c.vsqrtss(xmm0, xmm0, xmm1); break;
        // Scalar ~12-bit approximations (single-precision only); host estimate.
        case ZYDIS_MNEMONIC_VRSQRTSS:  c.vrsqrtss(xmm0, xmm0, xmm1); break;
        case ZYDIS_MNEMONIC_VRCPSS:    c.vrcpss(xmm0, xmm0, xmm1); break;
        default: return false;
    }
#undef VEC_OP

    if (ymm) {
        c.vmovdqu(ptr[r13 + YmmChunkOffset(dst_idx, 0)], ymm0);
    } else {
        c.vmovdqu(ptr[r13 + YmmChunkOffset(dst_idx, 0)], xmm0);
        ZeroUpperYmm(dst_idx, c);
    }
    return true;
}

/// Host-staged vector ops WITH an imm8: VSHUFPS, VBLENDPS, VPBLENDW.
/// 4-op (dst, src1, src2, imm8) or folded 3-op (dst=src1, src2, imm8).
bool EmitVecImm8(const ZydisDecodedInstruction& insn,
                 const ZydisDecodedOperand* ops,
                 u64 next_rip,
                 Xbyak::CodeGenerator& c) {
    const ZydisMnemonic m = insn.mnemonic;
    switch (m) {
        case ZYDIS_MNEMONIC_VSHUFPS: case ZYDIS_MNEMONIC_VBLENDPS:
        case ZYDIS_MNEMONIC_VPBLENDW: case ZYDIS_MNEMONIC_VCMPPS:
        case ZYDIS_MNEMONIC_VSHUFPD: case ZYDIS_MNEMONIC_VBLENDPD:
        case ZYDIS_MNEMONIC_VPBLENDD:
            break;
        default: return false;
    }
    int dst_idx, src1_idx;
    const ZydisDecodedOperand* src2 = nullptr;
    u8 imm = 0;
    if (insn.operand_count_visible == 4) {
        if (ops[0].type != ZYDIS_OPERAND_TYPE_REGISTER ||
            ops[1].type != ZYDIS_OPERAND_TYPE_REGISTER) return false;
        dst_idx  = ZydisVecToIndex(ops[0].reg.value);
        src1_idx = ZydisVecToIndex(ops[1].reg.value);
        src2     = &ops[2];
        if (ops[3].type != ZYDIS_OPERAND_TYPE_IMMEDIATE) return false;
        imm = static_cast<u8>(ops[3].imm.value.u);
    } else if (insn.operand_count_visible == 3) {
        if (ops[0].type != ZYDIS_OPERAND_TYPE_REGISTER) return false;
        dst_idx  = ZydisVecToIndex(ops[0].reg.value);
        src1_idx = dst_idx;
        src2     = &ops[1];
        if (ops[2].type != ZYDIS_OPERAND_TYPE_IMMEDIATE) return false;
        imm = static_cast<u8>(ops[2].imm.value.u);
    } else {
        return false;
    }
    if (dst_idx < 0 || src1_idx < 0) return false;

    const bool ymm = (ops[0].size == 256);
    if (ymm) c.vmovdqu(ymm0, ptr[r13 + YmmChunkOffset(src1_idx, 0)]);
    else     c.vmovdqu(xmm0, ptr[r13 + YmmChunkOffset(src1_idx, 0)]);

    if (src2->type == ZYDIS_OPERAND_TYPE_REGISTER) {
        const int s2 = ZydisVecToIndex(src2->reg.value);
        if (s2 < 0) return false;
        if (ymm) c.vmovdqu(ymm1, ptr[r13 + YmmChunkOffset(s2, 0)]);
        else     c.vmovdqu(xmm1, ptr[r13 + YmmChunkOffset(s2, 0)]);
    } else if (src2->type == ZYDIS_OPERAND_TYPE_MEMORY) {
        if (!EmitEffectiveAddress(src2->mem, next_rip, c)) return false;
        if (ymm) c.vmovdqu(ymm1, ptr[rdx]);
        else     c.vmovdqu(xmm1, ptr[rdx]);
    } else {
        return false;
    }

#define VEC_IMM(opname) \
    do { if (ymm) c.opname(ymm0, ymm0, ymm1, imm); else c.opname(xmm0, xmm0, xmm1, imm); } while (0)
    switch (m) {
        case ZYDIS_MNEMONIC_VSHUFPS:  VEC_IMM(vshufps);  break;
        case ZYDIS_MNEMONIC_VBLENDPS: VEC_IMM(vblendps); break;
        case ZYDIS_MNEMONIC_VPBLENDW: VEC_IMM(vpblendw); break;
        case ZYDIS_MNEMONIC_VCMPPS:   VEC_IMM(vcmpps);   break;
        case ZYDIS_MNEMONIC_VSHUFPD:  VEC_IMM(vshufpd);  break;
        case ZYDIS_MNEMONIC_VBLENDPD: VEC_IMM(vblendpd); break;
        case ZYDIS_MNEMONIC_VPBLENDD: VEC_IMM(vpblendd); break;
        default: return false;
    }
#undef VEC_IMM

    if (ymm) {
        c.vmovdqu(ptr[r13 + YmmChunkOffset(dst_idx, 0)], ymm0);
    } else {
        c.vmovdqu(ptr[r13 + YmmChunkOffset(dst_idx, 0)], xmm0);
        ZeroUpperYmm(dst_idx, c);
    }
    return true;
}

/// Packed 32-bit shifts by immediate or by xmm count: VPSRAD, VPSRLD.
/// (Arithmetic vs logical right shift; the host instruction handles the
/// sign-fill / zero-fill and the >=32 count clamping/zeroing.)
/// Also the whole-register byte shifts VPSLLDQ / VPSRLDQ (imm8 only).
bool EmitVecShift(const ZydisDecodedInstruction& insn,
                  const ZydisDecodedOperand* ops,
                  u64 next_rip,
                  Xbyak::CodeGenerator& c) {
    const ZydisMnemonic m = insn.mnemonic;
    switch (m) {
        case ZYDIS_MNEMONIC_VPSRAD: case ZYDIS_MNEMONIC_VPSRLD:
        case ZYDIS_MNEMONIC_VPSLLDQ: case ZYDIS_MNEMONIC_VPSRLDQ:
        case ZYDIS_MNEMONIC_VPSLLW: case ZYDIS_MNEMONIC_VPSLLD:
        case ZYDIS_MNEMONIC_VPSLLQ: case ZYDIS_MNEMONIC_VPSRLW:
        case ZYDIS_MNEMONIC_VPSRLQ: case ZYDIS_MNEMONIC_VPSRAW:
            break;
        default: return false;
    }
    // dst, src1, (imm8 | xmm count). 3 visible operands.
    if (insn.operand_count_visible != 3) return false;
    if (ops[0].type != ZYDIS_OPERAND_TYPE_REGISTER ||
        ops[1].type != ZYDIS_OPERAND_TYPE_REGISTER) return false;
    const int dst_idx  = ZydisVecToIndex(ops[0].reg.value);
    const int src1_idx = ZydisVecToIndex(ops[1].reg.value);
    if (dst_idx < 0 || src1_idx < 0) return false;
    const bool ymm = (ops[0].size == 256);

    if (ymm) c.vmovdqu(ymm0, ptr[r13 + YmmChunkOffset(src1_idx, 0)]);
    else     c.vmovdqu(xmm0, ptr[r13 + YmmChunkOffset(src1_idx, 0)]);

    if (ops[2].type == ZYDIS_OPERAND_TYPE_IMMEDIATE) {
        const u8 imm = static_cast<u8>(ops[2].imm.value.u);
        switch (m) {
            case ZYDIS_MNEMONIC_VPSRAD:
                if (ymm) c.vpsrad(ymm0, ymm0, imm); else c.vpsrad(xmm0, xmm0, imm); break;
            case ZYDIS_MNEMONIC_VPSRLD:
                if (ymm) c.vpsrld(ymm0, ymm0, imm); else c.vpsrld(xmm0, xmm0, imm); break;
            case ZYDIS_MNEMONIC_VPSLLDQ:
                if (ymm) c.vpslldq(ymm0, ymm0, imm); else c.vpslldq(xmm0, xmm0, imm); break;
            case ZYDIS_MNEMONIC_VPSRLDQ:
                if (ymm) c.vpsrldq(ymm0, ymm0, imm); else c.vpsrldq(xmm0, xmm0, imm); break;
            case ZYDIS_MNEMONIC_VPSLLW:
                if (ymm) c.vpsllw(ymm0, ymm0, imm); else c.vpsllw(xmm0, xmm0, imm); break;
            case ZYDIS_MNEMONIC_VPSLLD:
                if (ymm) c.vpslld(ymm0, ymm0, imm); else c.vpslld(xmm0, xmm0, imm); break;
            case ZYDIS_MNEMONIC_VPSLLQ:
                if (ymm) c.vpsllq(ymm0, ymm0, imm); else c.vpsllq(xmm0, xmm0, imm); break;
            case ZYDIS_MNEMONIC_VPSRLW:
                if (ymm) c.vpsrlw(ymm0, ymm0, imm); else c.vpsrlw(xmm0, xmm0, imm); break;
            case ZYDIS_MNEMONIC_VPSRLQ:
                if (ymm) c.vpsrlq(ymm0, ymm0, imm); else c.vpsrlq(xmm0, xmm0, imm); break;
            case ZYDIS_MNEMONIC_VPSRAW:
                if (ymm) c.vpsraw(ymm0, ymm0, imm); else c.vpsraw(xmm0, xmm0, imm); break;
            default: return false;
        }
    } else if (ops[2].type == ZYDIS_OPERAND_TYPE_REGISTER &&
               m != ZYDIS_MNEMONIC_VPSLLDQ && m != ZYDIS_MNEMONIC_VPSRLDQ) {
        // Per-element shifts accept an xmm count (low 64 bits = scalar count for
        // all lanes). The whole-register byte shifts VPSLLDQ/VPSRLDQ have no
        // register-count form, so they are excluded above.
        const int cnt = ZydisVecToIndex(ops[2].reg.value);
        if (cnt < 0) return false;
        c.vmovdqu(xmm1, ptr[r13 + YmmChunkOffset(cnt, 0)]);
        switch (m) {
            case ZYDIS_MNEMONIC_VPSRAD:
                if (ymm) c.vpsrad(ymm0, ymm0, xmm1); else c.vpsrad(xmm0, xmm0, xmm1); break;
            case ZYDIS_MNEMONIC_VPSRLD:
                if (ymm) c.vpsrld(ymm0, ymm0, xmm1); else c.vpsrld(xmm0, xmm0, xmm1); break;
            case ZYDIS_MNEMONIC_VPSLLW:
                if (ymm) c.vpsllw(ymm0, ymm0, xmm1); else c.vpsllw(xmm0, xmm0, xmm1); break;
            case ZYDIS_MNEMONIC_VPSLLD:
                if (ymm) c.vpslld(ymm0, ymm0, xmm1); else c.vpslld(xmm0, xmm0, xmm1); break;
            case ZYDIS_MNEMONIC_VPSLLQ:
                if (ymm) c.vpsllq(ymm0, ymm0, xmm1); else c.vpsllq(xmm0, xmm0, xmm1); break;
            case ZYDIS_MNEMONIC_VPSRLW:
                if (ymm) c.vpsrlw(ymm0, ymm0, xmm1); else c.vpsrlw(xmm0, xmm0, xmm1); break;
            case ZYDIS_MNEMONIC_VPSRLQ:
                if (ymm) c.vpsrlq(ymm0, ymm0, xmm1); else c.vpsrlq(xmm0, xmm0, xmm1); break;
            case ZYDIS_MNEMONIC_VPSRAW:
                if (ymm) c.vpsraw(ymm0, ymm0, xmm1); else c.vpsraw(xmm0, xmm0, xmm1); break;
            default: return false;
        }
    } else {
        return false;
    }

    if (ymm) {
        c.vmovdqu(ptr[r13 + YmmChunkOffset(dst_idx, 0)], ymm0);
    } else {
        c.vmovdqu(ptr[r13 + YmmChunkOffset(dst_idx, 0)], xmm0);
        ZeroUpperYmm(dst_idx, c);
    }
    return true;
}

/// Packed conversions: VCVTDQ2PS (int32->float), VCVTTPS2DQ (float->int32
/// truncating). 2-op (dst, src reg/mem). Host MXCSR governs rounding for
/// the dq2ps direction; ps2dq always truncates.
bool EmitVecConvert(const ZydisDecodedInstruction& insn,
                    const ZydisDecodedOperand* ops,
                    u64 next_rip,
                    Xbyak::CodeGenerator& c) {
    const ZydisMnemonic m = insn.mnemonic;
    if (m != ZYDIS_MNEMONIC_VCVTDQ2PS && m != ZYDIS_MNEMONIC_VCVTTPS2DQ)
        return false;
    if (ops[0].type != ZYDIS_OPERAND_TYPE_REGISTER) return false;
    const int dst_idx = ZydisVecToIndex(ops[0].reg.value);
    if (dst_idx < 0) return false;
    const bool ymm = (ops[0].size == 256);

    if (ops[1].type == ZYDIS_OPERAND_TYPE_REGISTER) {
        const int s = ZydisVecToIndex(ops[1].reg.value);
        if (s < 0) return false;
        if (ymm) c.vmovdqu(ymm0, ptr[r13 + YmmChunkOffset(s, 0)]);
        else     c.vmovdqu(xmm0, ptr[r13 + YmmChunkOffset(s, 0)]);
    } else if (ops[1].type == ZYDIS_OPERAND_TYPE_MEMORY) {
        if (!EmitEffectiveAddress(ops[1].mem, next_rip, c)) return false;
        if (ymm) c.vmovdqu(ymm0, ptr[rdx]);
        else     c.vmovdqu(xmm0, ptr[rdx]);
    } else {
        return false;
    }

    if (m == ZYDIS_MNEMONIC_VCVTDQ2PS) {
        if (ymm) c.vcvtdq2ps(ymm0, ymm0); else c.vcvtdq2ps(xmm0, xmm0);
    } else {
        if (ymm) c.vcvttps2dq(ymm0, ymm0); else c.vcvttps2dq(xmm0, xmm0);
    }

    if (ymm) {
        c.vmovdqu(ptr[r13 + YmmChunkOffset(dst_idx, 0)], ymm0);
    } else {
        c.vmovdqu(ptr[r13 + YmmChunkOffset(dst_idx, 0)], xmm0);
        ZeroUpperYmm(dst_idx, c);
    }
    return true;
}

/// VBROADCASTSS — replicate a single float (from xmm[0] or m32) to all
/// lanes. 2-op (dst, src reg/mem).
bool EmitVbroadcastss(const ZydisDecodedInstruction& insn,
                      const ZydisDecodedOperand* ops,
                      u64 next_rip,
                      Xbyak::CodeGenerator& c) {
    if (insn.mnemonic != ZYDIS_MNEMONIC_VBROADCASTSS) return false;
    if (ops[0].type != ZYDIS_OPERAND_TYPE_REGISTER) return false;
    const int dst_idx = ZydisVecToIndex(ops[0].reg.value);
    if (dst_idx < 0) return false;
    const bool ymm = (ops[0].size == 256);

    if (ops[1].type == ZYDIS_OPERAND_TYPE_MEMORY) {
        if (!EmitEffectiveAddress(ops[1].mem, next_rip, c)) return false;
        if (ymm) c.vbroadcastss(ymm0, dword[rdx]); else c.vbroadcastss(xmm0, dword[rdx]);
    } else if (ops[1].type == ZYDIS_OPERAND_TYPE_REGISTER) {
        const int s = ZydisVecToIndex(ops[1].reg.value);
        if (s < 0) return false;
        c.vmovdqu(xmm1, ptr[r13 + YmmChunkOffset(s, 0)]);
        if (ymm) c.vbroadcastss(ymm0, xmm1); else c.vbroadcastss(xmm0, xmm1);
    } else {
        return false;
    }

    if (ymm) {
        c.vmovdqu(ptr[r13 + YmmChunkOffset(dst_idx, 0)], ymm0);
    } else {
        c.vmovdqu(ptr[r13 + YmmChunkOffset(dst_idx, 0)], xmm0);
        ZeroUpperYmm(dst_idx, c);
    }
    return true;
}

/// Sign-mask extraction to a GPR: VMOVMSKPS / VMOVMSKPD (one bit per
/// single/double lane) and VPMOVMSKB (one bit per byte). The source is
/// always a vector register (no memory form); the destination is a GPR that
/// receives the mask zero-extended into the full 64-bit register. xmm gives
/// 4/2/16 bits, ymm gives 8/4/32 bits. The host instruction writes a 32-bit
/// GPR (zero-extending), so a single qword store covers both r32 and r64
/// destinations.
bool EmitMovmsk(const ZydisDecodedInstruction& insn,
                const ZydisDecodedOperand* ops,
                u64 next_rip,
                Xbyak::CodeGenerator& c) {
    (void)next_rip;
    const ZydisMnemonic m = insn.mnemonic;
    switch (m) {
        case ZYDIS_MNEMONIC_VMOVMSKPS:
        case ZYDIS_MNEMONIC_VMOVMSKPD:
        case ZYDIS_MNEMONIC_VPMOVMSKB:
            break;
        default: return false;
    }
    if (ops[0].type != ZYDIS_OPERAND_TYPE_REGISTER) return false;
    const int dst_idx = ZydisGprToIndex(ops[0].reg.value);
    if (dst_idx < 0) return false;
    if (ops[1].type != ZYDIS_OPERAND_TYPE_REGISTER) return false;  // no memory form
    const int src_idx = ZydisVecToIndex(ops[1].reg.value);
    if (src_idx < 0) return false;
    const bool ymm = (ops[1].size == 256);

    if (ymm) c.vmovdqu(ymm1, ptr[r13 + YmmChunkOffset(src_idx, 0)]);
    else     c.vmovdqu(xmm1, ptr[r13 + YmmChunkOffset(src_idx, 0)]);

    switch (m) {
        case ZYDIS_MNEMONIC_VMOVMSKPS: if (ymm) c.vmovmskps(eax, ymm1); else c.vmovmskps(eax, xmm1); break;
        case ZYDIS_MNEMONIC_VMOVMSKPD: if (ymm) c.vmovmskpd(eax, ymm1); else c.vmovmskpd(eax, xmm1); break;
        case ZYDIS_MNEMONIC_VPMOVMSKB: if (ymm) c.vpmovmskb(eax, ymm1); else c.vpmovmskb(eax, xmm1); break;
        default: return false;
    }
    c.mov(qword[r13 + GprOffset(dst_idx)], rax);  // eax result zero-extends rax
    return true;
}

/// VROUNDPS / VROUNDPD — round packed singles/doubles under an imm8
/// rounding-control byte. Three operands: dst (reg), src (reg or full-width
/// memory), imm8. The imm8 encoding (bits[1:0] = rounding mode, bit2 =
/// suppress-precision-exception, bit3 = use MXCSR.RC) is identical on the host,
/// so the guest imm passes straight through. Scalar VROUNDSS/SD have a
/// different (dst, src1, src2, imm8) shape and are handled elsewhere.
bool EmitRoundPacked(const ZydisDecodedInstruction& insn,
                     const ZydisDecodedOperand* ops,
                     u64 next_rip,
                     Xbyak::CodeGenerator& c) {
    const ZydisMnemonic m = insn.mnemonic;
    bool pd;
    switch (m) {
        case ZYDIS_MNEMONIC_VROUNDPS: pd = false; break;
        case ZYDIS_MNEMONIC_VROUNDPD: pd = true;  break;
        default: return false;
    }
    if (insn.operand_count_visible < 3) return false;
    if (ops[0].type != ZYDIS_OPERAND_TYPE_REGISTER) return false;
    const int dst_idx = ZydisVecToIndex(ops[0].reg.value);
    if (dst_idx < 0) return false;
    if (ops[2].type != ZYDIS_OPERAND_TYPE_IMMEDIATE) return false;
    const u8 imm = static_cast<u8>(ops[2].imm.value.u);
    const bool ymm = (ops[0].size == 256);

    const ZydisDecodedOperand* src = &ops[1];
    if (src->type == ZYDIS_OPERAND_TYPE_REGISTER) {
        const int s = ZydisVecToIndex(src->reg.value);
        if (s < 0) return false;
        if (ymm) c.vmovdqu(ymm1, ptr[r13 + YmmChunkOffset(s, 0)]);
        else     c.vmovdqu(xmm1, ptr[r13 + YmmChunkOffset(s, 0)]);
    } else if (src->type == ZYDIS_OPERAND_TYPE_MEMORY) {
        if (!EmitEffectiveAddress(src->mem, next_rip, c)) return false;
        if (ymm) c.vmovdqu(ymm1, ptr[rdx]);
        else     c.vmovdqu(xmm1, ptr[rdx]);
    } else {
        return false;
    }

    if (ymm) {
        if (pd) c.vroundpd(ymm0, ymm1, imm); else c.vroundps(ymm0, ymm1, imm);
        c.vmovdqu(ptr[r13 + YmmChunkOffset(dst_idx, 0)], ymm0);
    } else {
        if (pd) c.vroundpd(xmm0, xmm1, imm); else c.vroundps(xmm0, xmm1, imm);
        c.vmovdqu(ptr[r13 + YmmChunkOffset(dst_idx, 0)], xmm0);
        ZeroUpperYmm(dst_idx, c);
    }
    return true;
}

/// Packed integer widening moves: VPMOVZX** (zero-extend) and VPMOVSX**
/// (sign-extend), for every source->dest element pair (b->w, b->d, b->q,
/// w->d, w->q, d->q). The source is half/quarter/eighth the dest element
/// width, so a 128-bit dest reads m64/m32/m16 of source and a 256-bit dest
/// reads m128/m64/m32. Register source: load the full guest reg into xmm1 and
/// let the host instruction take the low bits it needs. The host op performs
/// the exact zero/sign extension, so this one template covers all 12.
bool EmitPmovExtend(const ZydisDecodedInstruction& insn,
                    const ZydisDecodedOperand* ops,
                    u64 next_rip,
                    Xbyak::CodeGenerator& c) {
    const ZydisMnemonic m = insn.mnemonic;
    switch (m) {
        case ZYDIS_MNEMONIC_VPMOVZXBW: case ZYDIS_MNEMONIC_VPMOVZXBD:
        case ZYDIS_MNEMONIC_VPMOVZXBQ: case ZYDIS_MNEMONIC_VPMOVZXWD:
        case ZYDIS_MNEMONIC_VPMOVZXWQ: case ZYDIS_MNEMONIC_VPMOVZXDQ:
        case ZYDIS_MNEMONIC_VPMOVSXBW: case ZYDIS_MNEMONIC_VPMOVSXBD:
        case ZYDIS_MNEMONIC_VPMOVSXBQ: case ZYDIS_MNEMONIC_VPMOVSXWD:
        case ZYDIS_MNEMONIC_VPMOVSXWQ: case ZYDIS_MNEMONIC_VPMOVSXDQ:
            break;
        default: return false;
    }
    if (ops[0].type != ZYDIS_OPERAND_TYPE_REGISTER) return false;
    const int dst_idx = ZydisVecToIndex(ops[0].reg.value);
    if (dst_idx < 0) return false;
    const bool ymm = (ops[0].size == 256);

    bool is_mem = false;
    if (ops[1].type == ZYDIS_OPERAND_TYPE_REGISTER) {
        const int s = ZydisVecToIndex(ops[1].reg.value);
        if (s < 0) return false;
        c.vmovdqu(xmm1, ptr[r13 + YmmChunkOffset(s, 0)]);
    } else if (ops[1].type == ZYDIS_OPERAND_TYPE_MEMORY) {
        if (!EmitEffectiveAddress(ops[1].mem, next_rip, c)) return false;
        is_mem = true;
    } else {
        return false;
    }

    // Emit the host op into xmm0/ymm0. For a memory source the deref width is
    // the source-operand size: dst xmm reads (xmm_sz), dst ymm reads (ymm_sz).
    // For a register source the full reg sits in xmm1 and the op takes the low
    // bits it needs.
#define PMOV(op, xmm_sz, ymm_sz)                                       \
    do {                                                               \
        if (ymm) { if (is_mem) c.op(ymm0, ymm_sz[rdx]); else c.op(ymm0, xmm1); } \
        else     { if (is_mem) c.op(xmm0, xmm_sz[rdx]); else c.op(xmm0, xmm1); } \
    } while (0)
    switch (m) {
        case ZYDIS_MNEMONIC_VPMOVZXBW: PMOV(vpmovzxbw, qword, xword); break;
        case ZYDIS_MNEMONIC_VPMOVZXBD: PMOV(vpmovzxbd, dword, qword); break;
        case ZYDIS_MNEMONIC_VPMOVZXBQ: PMOV(vpmovzxbq, word,  dword); break;
        case ZYDIS_MNEMONIC_VPMOVZXWD: PMOV(vpmovzxwd, qword, xword); break;
        case ZYDIS_MNEMONIC_VPMOVZXWQ: PMOV(vpmovzxwq, dword, qword); break;
        case ZYDIS_MNEMONIC_VPMOVZXDQ: PMOV(vpmovzxdq, qword, xword); break;
        case ZYDIS_MNEMONIC_VPMOVSXBW: PMOV(vpmovsxbw, qword, xword); break;
        case ZYDIS_MNEMONIC_VPMOVSXBD: PMOV(vpmovsxbd, dword, qword); break;
        case ZYDIS_MNEMONIC_VPMOVSXBQ: PMOV(vpmovsxbq, word,  dword); break;
        case ZYDIS_MNEMONIC_VPMOVSXWD: PMOV(vpmovsxwd, qword, xword); break;
        case ZYDIS_MNEMONIC_VPMOVSXWQ: PMOV(vpmovsxwq, dword, qword); break;
        case ZYDIS_MNEMONIC_VPMOVSXDQ: PMOV(vpmovsxdq, qword, xword); break;
        default: return false;
    }
#undef PMOV

    if (ymm) {
        c.vmovdqu(ptr[r13 + YmmChunkOffset(dst_idx, 0)], ymm0);
    } else {
        c.vmovdqu(ptr[r13 + YmmChunkOffset(dst_idx, 0)], xmm0);
        ZeroUpperYmm(dst_idx, c);
    }
    return true;
}
///   vinsertf128 ymm_dst, ymm_src1, xmm/m128 src2, imm8(lane)
///   vextractf128 xmm/m128 dst, ymm_src, imm8(lane)
bool EmitVlane128(const ZydisDecodedInstruction& insn,
                  const ZydisDecodedOperand* ops,
                  u64 next_rip,
                  Xbyak::CodeGenerator& c) {
    const ZydisMnemonic m = insn.mnemonic;
    if (m == ZYDIS_MNEMONIC_VINSERTF128) {
        // dst(ymm), src1(ymm), src2(xmm/m128), imm8
        if (insn.operand_count_visible != 4) return false;
        if (ops[0].type != ZYDIS_OPERAND_TYPE_REGISTER ||
            ops[1].type != ZYDIS_OPERAND_TYPE_REGISTER) return false;
        const int dst_idx  = ZydisVecToIndex(ops[0].reg.value);
        const int src1_idx = ZydisVecToIndex(ops[1].reg.value);
        if (dst_idx < 0 || src1_idx < 0) return false;
        if (ops[3].type != ZYDIS_OPERAND_TYPE_IMMEDIATE) return false;
        const u8 imm = static_cast<u8>(ops[3].imm.value.u);
        c.vmovdqu(ymm0, ptr[r13 + YmmChunkOffset(src1_idx, 0)]);
        if (ops[2].type == ZYDIS_OPERAND_TYPE_REGISTER) {
            const int s2 = ZydisVecToIndex(ops[2].reg.value);
            if (s2 < 0) return false;
            c.vmovdqu(xmm1, ptr[r13 + YmmChunkOffset(s2, 0)]);
            c.vinsertf128(ymm0, ymm0, xmm1, imm);
        } else if (ops[2].type == ZYDIS_OPERAND_TYPE_MEMORY) {
            if (!EmitEffectiveAddress(ops[2].mem, next_rip, c)) return false;
            c.vinsertf128(ymm0, ymm0, ptr[rdx], imm);
        } else {
            return false;
        }
        c.vmovdqu(ptr[r13 + YmmChunkOffset(dst_idx, 0)], ymm0);
        return true;
    }
    if (m == ZYDIS_MNEMONIC_VEXTRACTF128) {
        // dst(xmm/m128), src(ymm), imm8
        if (insn.operand_count_visible != 3) return false;
        if (ops[1].type != ZYDIS_OPERAND_TYPE_REGISTER) return false;
        const int src_idx = ZydisVecToIndex(ops[1].reg.value);
        if (src_idx < 0) return false;
        if (ops[2].type != ZYDIS_OPERAND_TYPE_IMMEDIATE) return false;
        const u8 imm = static_cast<u8>(ops[2].imm.value.u);
        c.vmovdqu(ymm0, ptr[r13 + YmmChunkOffset(src_idx, 0)]);
        if (ops[0].type == ZYDIS_OPERAND_TYPE_REGISTER) {
            const int dst_idx = ZydisVecToIndex(ops[0].reg.value);
            if (dst_idx < 0) return false;
            c.vextractf128(xmm1, ymm0, imm);
            c.vmovdqu(ptr[r13 + YmmChunkOffset(dst_idx, 0)], xmm1);
            ZeroUpperYmm(dst_idx, c);
        } else if (ops[0].type == ZYDIS_OPERAND_TYPE_MEMORY) {
            if (!EmitEffectiveAddress(ops[0].mem, next_rip, c)) return false;
            c.vextractf128(ptr[rdx], ymm0, imm);
        } else {
            return false;
        }
        return true;
    }
    return false;
}

/// VBLENDVPS / VBLENDVPD / VPBLENDVB — variable blend, per-element select
/// by the sign/high bit of a mask register's elements. 4-op: dst, src1,
/// src2, mask(xmm). VBLENDVPS selects per 32-bit element, VBLENDVPD per
/// 64-bit element, VPBLENDVB per byte — all share the same operand shape
/// and staging; only the host instruction differs.
bool EmitVblendvps(const ZydisDecodedInstruction& insn,
                   const ZydisDecodedOperand* ops,
                   u64 next_rip,
                   Xbyak::CodeGenerator& c) {
    const ZydisMnemonic m = insn.mnemonic;
    if (m != ZYDIS_MNEMONIC_VBLENDVPS &&
        m != ZYDIS_MNEMONIC_VBLENDVPD &&
        m != ZYDIS_MNEMONIC_VPBLENDVB) return false;
    if (insn.operand_count_visible != 4) return false;
    if (ops[0].type != ZYDIS_OPERAND_TYPE_REGISTER ||
        ops[1].type != ZYDIS_OPERAND_TYPE_REGISTER ||
        ops[3].type != ZYDIS_OPERAND_TYPE_REGISTER) return false;
    const int dst_idx  = ZydisVecToIndex(ops[0].reg.value);
    const int src1_idx = ZydisVecToIndex(ops[1].reg.value);
    const int mask_idx = ZydisVecToIndex(ops[3].reg.value);
    if (dst_idx < 0 || src1_idx < 0 || mask_idx < 0) return false;
    const bool ymm = (ops[0].size == 256);

    if (ymm) c.vmovdqu(ymm0, ptr[r13 + YmmChunkOffset(src1_idx, 0)]);
    else     c.vmovdqu(xmm0, ptr[r13 + YmmChunkOffset(src1_idx, 0)]);
    // mask -> ymm2/xmm2 (avoid xmm0/1 used for src1/src2)
    if (ymm) c.vmovdqu(ymm2, ptr[r13 + YmmChunkOffset(mask_idx, 0)]);
    else     c.vmovdqu(xmm2, ptr[r13 + YmmChunkOffset(mask_idx, 0)]);

    if (ops[2].type == ZYDIS_OPERAND_TYPE_REGISTER) {
        const int s2 = ZydisVecToIndex(ops[2].reg.value);
        if (s2 < 0) return false;
        if (ymm) c.vmovdqu(ymm1, ptr[r13 + YmmChunkOffset(s2, 0)]);
        else     c.vmovdqu(xmm1, ptr[r13 + YmmChunkOffset(s2, 0)]);
    } else if (ops[2].type == ZYDIS_OPERAND_TYPE_MEMORY) {
        if (!EmitEffectiveAddress(ops[2].mem, next_rip, c)) return false;
        if (ymm) c.vmovdqu(ymm1, ptr[rdx]);
        else     c.vmovdqu(xmm1, ptr[rdx]);
    } else {
        return false;
    }

    if (ymm) {
        switch (m) {
            case ZYDIS_MNEMONIC_VBLENDVPS: c.vblendvps(ymm0, ymm0, ymm1, ymm2); break;
            case ZYDIS_MNEMONIC_VBLENDVPD: c.vblendvpd(ymm0, ymm0, ymm1, ymm2); break;
            case ZYDIS_MNEMONIC_VPBLENDVB: c.vpblendvb(ymm0, ymm0, ymm1, ymm2); break;
            default: return false;
        }
    } else {
        switch (m) {
            case ZYDIS_MNEMONIC_VBLENDVPS: c.vblendvps(xmm0, xmm0, xmm1, xmm2); break;
            case ZYDIS_MNEMONIC_VBLENDVPD: c.vblendvpd(xmm0, xmm0, xmm1, xmm2); break;
            case ZYDIS_MNEMONIC_VPBLENDVB: c.vpblendvb(xmm0, xmm0, xmm1, xmm2); break;
            default: return false;
        }
    }

    if (ymm) {
        c.vmovdqu(ptr[r13 + YmmChunkOffset(dst_idx, 0)], ymm0);
    } else {
        c.vmovdqu(ptr[r13 + YmmChunkOffset(dst_idx, 0)], xmm0);
        ZeroUpperYmm(dst_idx, c);
    }
    return true;
}

/// GPR<->lane transfers: VPEXTRD/VPEXTRQ (lane -> GPR or m32/m64),
/// VEXTRACTPS (float lane -> GPR or m32), VPINSRD (GPR/m32 -> lane).
/// The host instructions encode the lane in imm8; we stage the xmm into
/// xmm0, point the host op at a GPR or computed memory address.
bool EmitLaneGpr(const ZydisDecodedInstruction& insn,
                 const ZydisDecodedOperand* ops,
                 u64 next_rip,
                 Xbyak::CodeGenerator& c) {
    const ZydisMnemonic m = insn.mnemonic;

    if (m == ZYDIS_MNEMONIC_VPEXTRD || m == ZYDIS_MNEMONIC_VPEXTRQ ||
        m == ZYDIS_MNEMONIC_VEXTRACTPS) {
        // dst(reg/mem), src(xmm), imm8(lane)
        if (insn.operand_count_visible != 3) return false;
        if (ops[1].type != ZYDIS_OPERAND_TYPE_REGISTER) return false;
        const int src_idx = ZydisVecToIndex(ops[1].reg.value);
        if (src_idx < 0) return false;
        if (ops[2].type != ZYDIS_OPERAND_TYPE_IMMEDIATE) return false;
        const u8 imm = static_cast<u8>(ops[2].imm.value.u);
        const bool q = (m == ZYDIS_MNEMONIC_VPEXTRQ);
        c.vmovdqu(xmm0, ptr[r13 + YmmChunkOffset(src_idx, 0)]);

        if (ops[0].type == ZYDIS_OPERAND_TYPE_REGISTER) {
            const int dst_gpr = ZydisGprToIndex(ops[0].reg.value);
            if (dst_gpr < 0) return false;
            if (q) {
                c.vpextrq(rax, xmm0, imm);
                c.mov(qword[r13 + GprOffset(dst_gpr)], rax);
            } else {
                // 32-bit extract: write eax then store full rax (zero-extend).
                if (m == ZYDIS_MNEMONIC_VPEXTRD) c.vpextrd(eax, xmm0, imm);
                else                              c.vextractps(eax, xmm0, imm);
                c.mov(qword[r13 + GprOffset(dst_gpr)], rax);
            }
            return true;
        }
        if (ops[0].type == ZYDIS_OPERAND_TYPE_MEMORY) {
            if (!EmitEffectiveAddress(ops[0].mem, next_rip, c)) return false;
            if (q) c.vpextrq(qword[rdx], xmm0, imm);
            else if (m == ZYDIS_MNEMONIC_VPEXTRD) c.vpextrd(dword[rdx], xmm0, imm);
            else c.vextractps(dword[rdx], xmm0, imm);
            return true;
        }
        return false;
    }

    if (m == ZYDIS_MNEMONIC_VPINSRD || m == ZYDIS_MNEMONIC_VPINSRW ||
        m == ZYDIS_MNEMONIC_VPINSRB) {
        // dst(xmm), src1(xmm), src2(reg32/mem), imm8(lane). The GPR source
        // supplies the low dword/word/byte; xbyak's vpinsr{d,w,b} all take a
        // 32-bit register operand and use only the low bits. Memory width is
        // dword/word/byte respectively.
        if (insn.operand_count_visible != 4) return false;
        if (ops[0].type != ZYDIS_OPERAND_TYPE_REGISTER ||
            ops[1].type != ZYDIS_OPERAND_TYPE_REGISTER) return false;
        const int dst_idx  = ZydisVecToIndex(ops[0].reg.value);
        const int src1_idx = ZydisVecToIndex(ops[1].reg.value);
        if (dst_idx < 0 || src1_idx < 0) return false;
        if (ops[3].type != ZYDIS_OPERAND_TYPE_IMMEDIATE) return false;
        const u8 imm = static_cast<u8>(ops[3].imm.value.u);
        c.vmovdqu(xmm0, ptr[r13 + YmmChunkOffset(src1_idx, 0)]);
        if (ops[2].type == ZYDIS_OPERAND_TYPE_REGISTER) {
            const int s2 = ZydisGprToIndex(ops[2].reg.value);
            if (s2 < 0) return false;
            c.mov(eax, dword[r13 + GprOffset(s2)]);   // low bits hold the value
            if (m == ZYDIS_MNEMONIC_VPINSRD)      c.vpinsrd(xmm0, xmm0, eax, imm);
            else if (m == ZYDIS_MNEMONIC_VPINSRW) c.vpinsrw(xmm0, xmm0, eax, imm);
            else                                  c.vpinsrb(xmm0, xmm0, eax, imm);
        } else if (ops[2].type == ZYDIS_OPERAND_TYPE_MEMORY) {
            if (!EmitEffectiveAddress(ops[2].mem, next_rip, c)) return false;
            if (m == ZYDIS_MNEMONIC_VPINSRD)      c.vpinsrd(xmm0, xmm0, dword[rdx], imm);
            else if (m == ZYDIS_MNEMONIC_VPINSRW) c.vpinsrw(xmm0, xmm0, word[rdx], imm);
            else                                  c.vpinsrb(xmm0, xmm0, byte[rdx], imm);
        } else {
            return false;
        }
        c.vmovdqu(ptr[r13 + YmmChunkOffset(dst_idx, 0)], xmm0);
        ZeroUpperYmm(dst_idx, c);
        return true;
    }
    return false;
}

/// VCMPSS — scalar single compare with imm8 predicate, producing an
/// all-ones/all-zeros 32-bit mask in lane 0 and preserving src1[127:32].
/// Host vcmpss does exactly this merge; we stage src1 into xmm0.
bool EmitVcmpss(const ZydisDecodedInstruction& insn,
                const ZydisDecodedOperand* ops,
                u64 next_rip,
                Xbyak::CodeGenerator& c) {
    if (insn.mnemonic != ZYDIS_MNEMONIC_VCMPSS) return false;
    int dst_idx, src1_idx;
    const ZydisDecodedOperand* src2 = nullptr;
    u8 imm = 0;
    if (insn.operand_count_visible == 4) {
        if (ops[0].type != ZYDIS_OPERAND_TYPE_REGISTER ||
            ops[1].type != ZYDIS_OPERAND_TYPE_REGISTER) return false;
        dst_idx  = ZydisVecToIndex(ops[0].reg.value);
        src1_idx = ZydisVecToIndex(ops[1].reg.value);
        src2     = &ops[2];
        if (ops[3].type != ZYDIS_OPERAND_TYPE_IMMEDIATE) return false;
        imm = static_cast<u8>(ops[3].imm.value.u);
    } else if (insn.operand_count_visible == 3) {
        if (ops[0].type != ZYDIS_OPERAND_TYPE_REGISTER) return false;
        dst_idx  = ZydisVecToIndex(ops[0].reg.value);
        src1_idx = dst_idx;
        src2     = &ops[1];
        if (ops[2].type != ZYDIS_OPERAND_TYPE_IMMEDIATE) return false;
        imm = static_cast<u8>(ops[2].imm.value.u);
    } else {
        return false;
    }
    if (dst_idx < 0 || src1_idx < 0) return false;

    c.vmovdqu(xmm0, ptr[r13 + YmmChunkOffset(src1_idx, 0)]);
    if (src2->type == ZYDIS_OPERAND_TYPE_REGISTER) {
        const int s2 = ZydisVecToIndex(src2->reg.value);
        if (s2 < 0) return false;
        c.vmovdqu(xmm1, ptr[r13 + YmmChunkOffset(s2, 0)]);
        c.vcmpss(xmm0, xmm0, xmm1, imm);
    } else if (src2->type == ZYDIS_OPERAND_TYPE_MEMORY) {
        if (!EmitEffectiveAddress(src2->mem, next_rip, c)) return false;
        c.vcmpss(xmm0, xmm0, dword[rdx], imm);
    } else {
        return false;
    }
    c.vmovdqu(ptr[r13 + YmmChunkOffset(dst_idx, 0)], xmm0);
    ZeroUpperYmm(dst_idx, c);
    return true;
}

/// VROUNDSS / VROUNDSD — round a scalar single/double to an integral
/// value using the rounding mode selected by the imm8 control (bit 2 =
/// "use MXCSR rounding mode", low 2 bits = 0:nearest-even, 1:floor,
/// 2:ceil, 3:truncate; bit 3 suppresses the precision exception). The
/// host VROUNDSS/SD takes the identical imm8, so we pass it through.
///
/// Form: `vround{ss,sd} dst, src1, src2, imm8` — dst.elem =
/// round(src2.elem); dst[127:ELEM] = src1[127:ELEM]; upper YMM zeroed.
/// (Folded 2-operand `vroundss dst, src2, imm8` with dst==src1 is
/// also accepted.) Implemented with the real host instruction: stage
/// src1 into xmm0 (for the merge), src2 into xmm1, round in place.
bool EmitVround(const ZydisDecodedInstruction& insn,
                const ZydisDecodedOperand* ops,
                u64 next_rip,
                Xbyak::CodeGenerator& c) {
    const bool ss = (insn.mnemonic == ZYDIS_MNEMONIC_VROUNDSS);
    const bool sd = (insn.mnemonic == ZYDIS_MNEMONIC_VROUNDSD);
    if (!ss && !sd) return false;

    int dst_idx, src1_idx;
    const ZydisDecodedOperand* src2 = nullptr;
    const ZydisDecodedOperand* immop = nullptr;
    if (insn.operand_count_visible == 4) {
        if (ops[0].type != ZYDIS_OPERAND_TYPE_REGISTER ||
            ops[1].type != ZYDIS_OPERAND_TYPE_REGISTER) return false;
        dst_idx  = ZydisVecToIndex(ops[0].reg.value);
        src1_idx = ZydisVecToIndex(ops[1].reg.value);
        src2     = &ops[2];
        immop    = &ops[3];
    } else if (insn.operand_count_visible == 3) {
        if (ops[0].type != ZYDIS_OPERAND_TYPE_REGISTER) return false;
        dst_idx  = ZydisVecToIndex(ops[0].reg.value);
        src1_idx = dst_idx;
        src2     = &ops[1];
        immop    = &ops[2];
    } else {
        return false;
    }
    if (dst_idx < 0 || src1_idx < 0) return false;
    if (immop->type != ZYDIS_OPERAND_TYPE_IMMEDIATE) return false;
    const u8 imm = static_cast<u8>(immop->imm.value.u & 0xFF);

    // Stage src1 into xmm0 (carries the lane bits to preserve).
    c.vmovdqu(xmm0, ptr[r13 + YmmChunkOffset(src1_idx, 0)]);

    bool src2_mem = false;
    if (src2->type == ZYDIS_OPERAND_TYPE_REGISTER) {
        const int s2 = ZydisVecToIndex(src2->reg.value);
        if (s2 < 0) return false;
        c.vmovdqu(xmm1, ptr[r13 + YmmChunkOffset(s2, 0)]);
    } else if (src2->type == ZYDIS_OPERAND_TYPE_MEMORY) {
        if (!EmitEffectiveAddress(src2->mem, next_rip, c)) return false;
        src2_mem = true;
    } else {
        return false;
    }

    if (ss) {
        if (src2_mem) c.vroundss(xmm0, xmm0, dword[rdx], imm);
        else          c.vroundss(xmm0, xmm0, xmm1, imm);
    } else {
        if (src2_mem) c.vroundsd(xmm0, xmm0, qword[rdx], imm);
        else          c.vroundsd(xmm0, xmm0, xmm1, imm);
    }

    c.vmovdqu(ptr[r13 + YmmChunkOffset(dst_idx, 0)], xmm0);
    c.xor_(rax, rax);
    c.mov(qword[r13 + YmmChunkOffset(dst_idx, 2)], rax);
    c.mov(qword[r13 + YmmChunkOffset(dst_idx, 3)], rax);
    return true;
}

/// VINSERTPS — insert a single-precision float into one of the four
/// dword lanes of a 128-bit vector, with an imm8 control:
///   imm[7:6] = source-element select (which dword of src2; ignored
///              when src2 is a 32-bit memory operand)
///   imm[5:4] = destination dword select
///   imm[3:0] = zero mask (each set bit zeroes that output dword AFTER
///              the insert)
/// The rest of dst's low 128 comes from src1; the upper YMM is zeroed
/// (VEX-128). The host VINSERTPS implements the full imm semantics, so
/// we stage src1 into xmm0, src2 into xmm1 (or use the m32 operand),
/// run the host op with the same imm, and store back.
///
/// Operands: 4 visible (dst, src1, src2, imm8); the folded 3-visible
/// (dst=src1, src2, imm8) is also accepted.
bool EmitVinsertps(const ZydisDecodedInstruction& insn,
                   const ZydisDecodedOperand* ops,
                   u64 next_rip,
                   Xbyak::CodeGenerator& c) {
    if (insn.mnemonic != ZYDIS_MNEMONIC_VINSERTPS) return false;

    int dst_idx, src1_idx;
    const ZydisDecodedOperand* src2 = nullptr;
    const ZydisDecodedOperand* immop = nullptr;
    if (insn.operand_count_visible == 4) {
        if (ops[0].type != ZYDIS_OPERAND_TYPE_REGISTER ||
            ops[1].type != ZYDIS_OPERAND_TYPE_REGISTER) return false;
        dst_idx  = ZydisVecToIndex(ops[0].reg.value);
        src1_idx = ZydisVecToIndex(ops[1].reg.value);
        src2     = &ops[2];
        immop    = &ops[3];
    } else if (insn.operand_count_visible == 3) {
        if (ops[0].type != ZYDIS_OPERAND_TYPE_REGISTER) return false;
        dst_idx  = ZydisVecToIndex(ops[0].reg.value);
        src1_idx = dst_idx;
        src2     = &ops[1];
        immop    = &ops[2];
    } else {
        return false;
    }
    if (dst_idx < 0 || src1_idx < 0) return false;
    if (immop->type != ZYDIS_OPERAND_TYPE_IMMEDIATE) return false;
    const u8 imm = static_cast<u8>(immop->imm.value.u & 0xFF);

    c.vmovdqu(xmm0, ptr[r13 + YmmChunkOffset(src1_idx, 0)]);

    if (src2->type == ZYDIS_OPERAND_TYPE_REGISTER) {
        const int s2 = ZydisVecToIndex(src2->reg.value);
        if (s2 < 0) return false;
        c.vmovdqu(xmm1, ptr[r13 + YmmChunkOffset(s2, 0)]);
        c.vinsertps(xmm0, xmm0, xmm1, imm);
    } else if (src2->type == ZYDIS_OPERAND_TYPE_MEMORY) {
        if (!EmitEffectiveAddress(src2->mem, next_rip, c)) return false;
        c.vinsertps(xmm0, xmm0, dword[rdx], imm);
    } else {
        return false;
    }

    c.vmovdqu(ptr[r13 + YmmChunkOffset(dst_idx, 0)], xmm0);
    c.xor_(rax, rax);
    c.mov(qword[r13 + YmmChunkOffset(dst_idx, 2)], rax);
    c.mov(qword[r13 + YmmChunkOffset(dst_idx, 3)], rax);
    return true;
}

/// VMOVSLDUP / VMOVSHDUP / VMOVDDUP — replicate elements while moving:
///   VMOVSLDUP: dst[0]=dst[1]=src[0], dst[2]=dst[3]=src[2] (dup even
///              single lanes), per 128-bit lane.
///   VMOVSHDUP: dst[0]=dst[1]=src[1], dst[2]=dst[3]=src[3] (dup odd
///              single lanes), per 128-bit lane.
///   VMOVDDUP : dst.lo64 = dst.hi64 = src.lo64 (dup the low double),
///              per 128-bit lane.
/// 2-operand (dst, src) with src reg or mem. The host instructions do
/// the replication exactly; we stage src into xmm0/ymm0 (or read from
/// memory), run the host op, and store back with VEX-128 upper-zero.
bool EmitMovDup(const ZydisDecodedInstruction& insn,
                const ZydisDecodedOperand* ops,
                u64 next_rip,
                Xbyak::CodeGenerator& c) {
    const ZydisMnemonic m = insn.mnemonic;
    if (m != ZYDIS_MNEMONIC_VMOVSLDUP &&
        m != ZYDIS_MNEMONIC_VMOVSHDUP &&
        m != ZYDIS_MNEMONIC_VMOVDDUP) {
        return false;
    }
    if (ops[0].type != ZYDIS_OPERAND_TYPE_REGISTER) return false;
    const int dst_idx = ZydisVecToIndex(ops[0].reg.value);
    if (dst_idx < 0) return false;

    const int vec_bits = ops[0].size;
    if (vec_bits != 128 && vec_bits != 256) return false;
    const bool ymm = (vec_bits == 256);

    // Stage the source into host xmm0/ymm0 (register) or read mem.
    bool src_mem = false;
    if (ops[1].type == ZYDIS_OPERAND_TYPE_REGISTER) {
        const int s = ZydisVecToIndex(ops[1].reg.value);
        if (s < 0) return false;
        if (ymm) c.vmovdqu(ymm0, ptr[r13 + YmmChunkOffset(s, 0)]);
        else     c.vmovdqu(xmm0, ptr[r13 + YmmChunkOffset(s, 0)]);
    } else if (ops[1].type == ZYDIS_OPERAND_TYPE_MEMORY) {
        if (!EmitEffectiveAddress(ops[1].mem, next_rip, c)) return false;
        if (ymm) c.vmovdqu(ymm0, ptr[rdx]);
        else     c.vmovdqu(xmm0, ptr[rdx]);
        src_mem = true;
    } else {
        return false;
    }
    (void)src_mem;

    if (ymm) {
        switch (m) {
            case ZYDIS_MNEMONIC_VMOVSLDUP: c.vmovsldup(ymm0, ymm0); break;
            case ZYDIS_MNEMONIC_VMOVSHDUP: c.vmovshdup(ymm0, ymm0); break;
            case ZYDIS_MNEMONIC_VMOVDDUP:  c.vmovddup(ymm0, ymm0);  break;
            default: return false;
        }
        c.vmovdqu(ptr[r13 + YmmChunkOffset(dst_idx, 0)], ymm0);
    } else {
        switch (m) {
            case ZYDIS_MNEMONIC_VMOVSLDUP: c.vmovsldup(xmm0, xmm0); break;
            case ZYDIS_MNEMONIC_VMOVSHDUP: c.vmovshdup(xmm0, xmm0); break;
            case ZYDIS_MNEMONIC_VMOVDDUP:  c.vmovddup(xmm0, xmm0);  break;
            default: return false;
        }
        c.vmovdqu(ptr[r13 + YmmChunkOffset(dst_idx, 0)], xmm0);
        c.xor_(rax, rax);
        c.mov(qword[r13 + YmmChunkOffset(dst_idx, 2)], rax);
        c.mov(qword[r13 + YmmChunkOffset(dst_idx, 3)], rax);
    }
    return true;
}
///
/// Supported operand shape: reg, reg, reg (the common case in the
/// game's prologue zeroing idiom: `vxorps xmm0, xmm0, xmm0`).
/// Reg + reg + mem can be added when we see it.
/// Vector bitwise logic: AND/OR/XOR/ANDN at full vector width. Covers
/// both the float-named encodings (VANDPS/PD, VORPS/PD, VXORPS/PD,
/// VANDNPS/PD) and the integer-named ones (VPAND, VPOR, VPXOR, VPANDN).
/// These are pure bit operations — no FP arithmetic, no flags — so we
/// do them with 64-bit GPR chunk ops (no host SIMD needed). ANDN is the
/// asymmetric (~src1) & src2.
///
/// Operand shapes: 3 visible (dst, src1, src2) with src2 reg or mem, and
/// the assembler-folded 2 visible (dst=src1, src2). VEX-128 zeroes the
/// upper YMM half of the destination.
bool EmitVecBitLogic(const ZydisDecodedInstruction& insn,
                     const ZydisDecodedOperand* ops,
                     u64 next_rip,
                     Xbyak::CodeGenerator& c) {
    enum class Op { And, Or, Xor, AndN };
    Op op;
    switch (insn.mnemonic) {
        case ZYDIS_MNEMONIC_VANDPS: case ZYDIS_MNEMONIC_VANDPD:
        case ZYDIS_MNEMONIC_VPAND:  op = Op::And;  break;
        case ZYDIS_MNEMONIC_VORPS:  case ZYDIS_MNEMONIC_VORPD:
        case ZYDIS_MNEMONIC_VPOR:   op = Op::Or;   break;
        case ZYDIS_MNEMONIC_VXORPS: case ZYDIS_MNEMONIC_VXORPD:
        case ZYDIS_MNEMONIC_VPXOR:  op = Op::Xor;  break;
        case ZYDIS_MNEMONIC_VANDNPS: case ZYDIS_MNEMONIC_VANDNPD:
        case ZYDIS_MNEMONIC_VPANDN:  op = Op::AndN; break;
        default: return false;
    }

    int dst_idx, src1_idx;
    const ZydisDecodedOperand* src2 = nullptr;
    if (insn.operand_count_visible == 3) {
        if (ops[0].type != ZYDIS_OPERAND_TYPE_REGISTER ||
            ops[1].type != ZYDIS_OPERAND_TYPE_REGISTER) return false;
        dst_idx  = ZydisVecToIndex(ops[0].reg.value);
        src1_idx = ZydisVecToIndex(ops[1].reg.value);
        src2     = &ops[2];
    } else if (insn.operand_count_visible == 2) {
        if (ops[0].type != ZYDIS_OPERAND_TYPE_REGISTER) return false;
        dst_idx  = ZydisVecToIndex(ops[0].reg.value);
        src1_idx = dst_idx;
        src2     = &ops[1];
    } else {
        return false;
    }
    if (dst_idx < 0 || src1_idx < 0) return false;

    const int vec_bits = ops[0].size;
    if (vec_bits != 128 && vec_bits != 256) return false;
    const int chunks = vec_bits / 64;

    // Resolve src2: either a vector register lane base, or a memory
    // address stashed in r10 (so per-chunk derefs use a stable base).
    int src2_idx = -1;
    bool src2_mem = false;
    if (src2->type == ZYDIS_OPERAND_TYPE_REGISTER) {
        src2_idx = ZydisVecToIndex(src2->reg.value);
        if (src2_idx < 0) return false;
    } else if (src2->type == ZYDIS_OPERAND_TYPE_MEMORY) {
        if (!EmitEffectiveAddress(src2->mem, next_rip, c)) return false;
        c.mov(r10, rdx);
        src2_mem = true;
    } else {
        return false;
    }

    // dst[i] = src1[i] <op> src2[i] for each 64-bit chunk.
    for (int i = 0; i < chunks; ++i) {
        c.mov(rax, qword[r13 + YmmChunkOffset(src1_idx, i)]);   // rax = src1
        if (src2_mem) c.mov(rcx, qword[r10 + i * 8]);
        else          c.mov(rcx, qword[r13 + YmmChunkOffset(src2_idx, i)]);
        switch (op) {
            case Op::And:  c.and_(rax, rcx); break;
            case Op::Or:   c.or_(rax, rcx);  break;
            case Op::Xor:  c.xor_(rax, rcx); break;
            case Op::AndN: c.not_(rax); c.and_(rax, rcx); break; // (~src1) & src2
        }
        c.mov(qword[r13 + YmmChunkOffset(dst_idx, i)], rax);
    }
    if (vec_bits == 128) {
        c.xor_(rax, rax);
        c.mov(qword[r13 + YmmChunkOffset(dst_idx, 2)], rax);
        c.mov(qword[r13 + YmmChunkOffset(dst_idx, 3)], rax);
    }
    return true;
}

/// VPSHUFB — byte-granularity shuffle. For each output byte i:
///   dst[i] = (mask[i] & 0x80) ? 0 : src[mask[i] & 0x0F]
/// AVX2 extends this to 256-bit width, but the shuffle operates
/// independently on each 128-bit lane (no cross-lane movement),
/// so we emit it as two 128-bit halves.
///
/// Unlike the GPR-relayed AVX emitters above, VPSHUFB is hard to
/// emulate efficiently in scalar code (16 conditional byte loads
/// per lane). We use host VPSHUFB directly on host xmm0/xmm1.
/// Those registers are caller-saved on both SysV (xmm0-15) and
/// Windows (xmm0-5) ABIs; the JIT owns the full XMM file inside
/// a block, since neither the gateway nor the dispatcher
/// preserves any XMM state across block boundaries — each block
/// reloads what it needs from the GuestState slot.
///
/// Operand-count handling: Zydis reports `operand_count_visible`
/// based on assembler syntax. For a destructive guest encoding
/// like `vpshufb xmm5, xmm5, xmm6` an assembler may fold to
/// 2-operand syntax (`vpshufb xmm5, xmm6`) and Zydis will report
/// 2 visible ops. We handle both shapes: 3 visible → ops[0,1,2]
/// = dst, src1, mask; 2 visible → ops[0,1] = dst(=src1), mask.
bool EmitVpshufb(const ZydisDecodedInstruction& insn,
                 const ZydisDecodedOperand* ops,
                 u64 next_rip,
                 Xbyak::CodeGenerator& c) {
    if (insn.mnemonic != ZYDIS_MNEMONIC_VPSHUFB) return false;

    int dst_idx, src_idx;
    const ZydisDecodedOperand* mask = nullptr;
    if (insn.operand_count_visible == 3) {
        if (ops[0].type != ZYDIS_OPERAND_TYPE_REGISTER ||
            ops[1].type != ZYDIS_OPERAND_TYPE_REGISTER) return false;
        dst_idx = ZydisVecToIndex(ops[0].reg.value);
        src_idx = ZydisVecToIndex(ops[1].reg.value);
        mask    = &ops[2];
    } else if (insn.operand_count_visible == 2) {
        if (ops[0].type != ZYDIS_OPERAND_TYPE_REGISTER) return false;
        dst_idx = ZydisVecToIndex(ops[0].reg.value);
        src_idx = dst_idx;
        mask    = &ops[1];
    } else {
        return false;
    }
    if (dst_idx < 0 || src_idx < 0) return false;

    const int vec_bits = ops[0].size;
    if (vec_bits != 128 && vec_bits != 256) return false;

    // Resolve the mask: register (per-lane slot) or memory. For a memory
    // mask we load the whole vector into r10-based staging once: compute
    // the base address (rdx) and load each lane from rdx + lane*16.
    int mask_idx = -1;
    bool mask_mem = false;
    if (mask->type == ZYDIS_OPERAND_TYPE_REGISTER) {
        mask_idx = ZydisVecToIndex(mask->reg.value);
        if (mask_idx < 0) return false;
    } else if (mask->type == ZYDIS_OPERAND_TYPE_MEMORY) {
        if (!EmitEffectiveAddress(mask->mem, next_rip, c)) return false;
        c.mov(r10, rdx);   // stash mask base address (rdx reused below)
        mask_mem = true;
    } else {
        return false;
    }

    for (int lane = 0; lane < vec_bits / 128; ++lane) {
        const u32 src_off = YmmChunkOffset(src_idx, lane * 2);
        const u32 dst_off = YmmChunkOffset(dst_idx, lane * 2);
        c.vmovdqu(xmm0, ptr[r13 + src_off]);
        if (mask_mem) {
            c.vmovdqu(xmm1, ptr[r10 + lane * 16]);
        } else {
            c.vmovdqu(xmm1, ptr[r13 + YmmChunkOffset(mask_idx, lane * 2)]);
        }
        c.vpshufb(xmm0, xmm0, xmm1);
        c.vmovdqu(ptr[r13 + dst_off], xmm0);
    }
    if (vec_bits == 128) {
        c.xor_(rax, rax);
        c.mov(qword[r13 + YmmChunkOffset(dst_idx, 2)], rax);
        c.mov(qword[r13 + YmmChunkOffset(dst_idx, 3)], rax);
    }
    return true;
}

/// VPCMPEQB — packed byte-equality compare. For each byte i:
///   dst[i] = (src1[i] == src2[i]) ? 0xFF : 0x00
/// Byte-granular compare is awkward in scalar GPR code (16 compares
/// per lane), so we use the host VPCMPEQB on host xmm0/xmm1 — same
/// approach as VPSHUFB. Operates independently per 128-bit lane, so
/// the 256-bit AVX2 form is just two lanes.
///
/// Operand-count handling mirrors VPSHUFB: a destructive guest
/// encoding `vpcmpeqb xmm, xmm, xmm/m` may be folded by the assembler
/// to 2-operand syntax, so accept both 3-visible (dst, src1, src2)
/// and 2-visible (dst=src1, src2). src2 may be a register or memory.
bool EmitVpcmpeqb(const ZydisDecodedInstruction& insn,
                  const ZydisDecodedOperand* ops,
                  u64 next_rip,
                  Xbyak::CodeGenerator& c) {
    if (insn.mnemonic != ZYDIS_MNEMONIC_VPCMPEQB) return false;

    int dst_idx, src1_idx;
    const ZydisDecodedOperand* src2 = nullptr;
    if (insn.operand_count_visible == 3) {
        if (ops[0].type != ZYDIS_OPERAND_TYPE_REGISTER ||
            ops[1].type != ZYDIS_OPERAND_TYPE_REGISTER) return false;
        dst_idx  = ZydisVecToIndex(ops[0].reg.value);
        src1_idx = ZydisVecToIndex(ops[1].reg.value);
        src2     = &ops[2];
    } else if (insn.operand_count_visible == 2) {
        if (ops[0].type != ZYDIS_OPERAND_TYPE_REGISTER) return false;
        dst_idx  = ZydisVecToIndex(ops[0].reg.value);
        src1_idx = dst_idx;
        src2     = &ops[1];
    } else {
        return false;
    }
    if (dst_idx < 0 || src1_idx < 0) return false;

    const int vec_bits = ops[0].size;
    if (vec_bits != 128 && vec_bits != 256) return false;

    // Memory src2 is only supported for the single-lane 128-bit form
    // here (the address points at a 128-bit operand). Register src2
    // handles both widths lane-by-lane.
    if (src2->type == ZYDIS_OPERAND_TYPE_MEMORY) {
        if (vec_bits != 128) return false;
        if (!EmitEffectiveAddress(src2->mem, next_rip, c)) return false;
        c.vmovdqu(xmm0, ptr[r13 + YmmChunkOffset(src1_idx, 0)]);
        c.vmovdqu(xmm1, ptr[rdx]);
        c.vpcmpeqb(xmm0, xmm0, xmm1);
        c.vmovdqu(ptr[r13 + YmmChunkOffset(dst_idx, 0)], xmm0);
        c.xor_(rax, rax);
        c.mov(qword[r13 + YmmChunkOffset(dst_idx, 2)], rax);
        c.mov(qword[r13 + YmmChunkOffset(dst_idx, 3)], rax);
        return true;
    }

    if (src2->type != ZYDIS_OPERAND_TYPE_REGISTER) return false;
    const int src2_idx = ZydisVecToIndex(src2->reg.value);
    if (src2_idx < 0) return false;

    for (int lane = 0; lane < vec_bits / 128; ++lane) {
        const u32 a_off = YmmChunkOffset(src1_idx, lane * 2);
        const u32 b_off = YmmChunkOffset(src2_idx, lane * 2);
        const u32 d_off = YmmChunkOffset(dst_idx,  lane * 2);
        c.vmovdqu(xmm0, ptr[r13 + a_off]);
        c.vmovdqu(xmm1, ptr[r13 + b_off]);
        c.vpcmpeqb(xmm0, xmm0, xmm1);
        c.vmovdqu(ptr[r13 + d_off], xmm0);
    }
    if (vec_bits == 128) {
        c.xor_(rax, rax);
        c.mov(qword[r13 + YmmChunkOffset(dst_idx, 2)], rax);
        c.mov(qword[r13 + YmmChunkOffset(dst_idx, 3)], rax);
    }
    return true;
}

/// VPUNPCK{L,H}{WD,DQ,QDQ} — unpack/interleave from two source vectors
/// at word / dword / qword granularity, operating independently within
/// each 128-bit lane. We stage the operands into host xmm/ymm 0/1 and
/// emit the host unpack, which implements the per-lane interleave (and
/// the AVX2 per-lane behavior for the 256-bit form) exactly. VEX-128
/// zeroes bits 255:128 of the destination.
///
/// Operand shapes: 3-visible (dst, src1, src2) — src2 reg or mem — and
/// the assembler-folded 2-visible (dst=src1, src2).
bool EmitVpunpck(const ZydisDecodedInstruction& insn,
                 const ZydisDecodedOperand* ops,
                 u64 next_rip,
                 Xbyak::CodeGenerator& c) {
    const ZydisMnemonic m = insn.mnemonic;
    if (m != ZYDIS_MNEMONIC_VPUNPCKLQDQ && m != ZYDIS_MNEMONIC_VPUNPCKHQDQ &&
        m != ZYDIS_MNEMONIC_VPUNPCKLDQ  && m != ZYDIS_MNEMONIC_VPUNPCKHDQ  &&
        m != ZYDIS_MNEMONIC_VPUNPCKLWD  && m != ZYDIS_MNEMONIC_VPUNPCKHWD  &&
        m != ZYDIS_MNEMONIC_VPUNPCKLBW  && m != ZYDIS_MNEMONIC_VPUNPCKHBW) {
        return false;
    }

    int dst_idx, src1_idx;
    const ZydisDecodedOperand* src2 = nullptr;
    if (insn.operand_count_visible == 3) {
        if (ops[0].type != ZYDIS_OPERAND_TYPE_REGISTER ||
            ops[1].type != ZYDIS_OPERAND_TYPE_REGISTER) return false;
        dst_idx  = ZydisVecToIndex(ops[0].reg.value);
        src1_idx = ZydisVecToIndex(ops[1].reg.value);
        src2     = &ops[2];
    } else if (insn.operand_count_visible == 2) {
        if (ops[0].type != ZYDIS_OPERAND_TYPE_REGISTER) return false;
        dst_idx  = ZydisVecToIndex(ops[0].reg.value);
        src1_idx = dst_idx;
        src2     = &ops[1];
    } else {
        return false;
    }
    if (dst_idx < 0 || src1_idx < 0) return false;

    const int vec_bits = ops[0].size;
    if (vec_bits != 128 && vec_bits != 256) return false;
    const bool ymm = (vec_bits == 256);

    // Stage src1 into host xmm0/ymm0.
    if (ymm) c.vmovdqu(ymm0, ptr[r13 + YmmChunkOffset(src1_idx, 0)]);
    else     c.vmovdqu(xmm0, ptr[r13 + YmmChunkOffset(src1_idx, 0)]);

    // Stage src2 into host xmm1/ymm1 (register or memory).
    if (src2->type == ZYDIS_OPERAND_TYPE_REGISTER) {
        const int s2 = ZydisVecToIndex(src2->reg.value);
        if (s2 < 0) return false;
        if (ymm) c.vmovdqu(ymm1, ptr[r13 + YmmChunkOffset(s2, 0)]);
        else     c.vmovdqu(xmm1, ptr[r13 + YmmChunkOffset(s2, 0)]);
    } else if (src2->type == ZYDIS_OPERAND_TYPE_MEMORY) {
        if (ymm) return false;  // 256-bit mem-src not observed; add when seen
        if (!EmitEffectiveAddress(src2->mem, next_rip, c)) return false;
        c.vmovdqu(xmm1, ptr[rdx]);
    } else {
        return false;
    }

    // Emit the host unpack: xmm0 = unpack(xmm0, xmm1).
    if (ymm) {
        switch (m) {
            case ZYDIS_MNEMONIC_VPUNPCKLQDQ: c.vpunpcklqdq(ymm0, ymm0, ymm1); break;
            case ZYDIS_MNEMONIC_VPUNPCKHQDQ: c.vpunpckhqdq(ymm0, ymm0, ymm1); break;
            case ZYDIS_MNEMONIC_VPUNPCKLDQ:  c.vpunpckldq (ymm0, ymm0, ymm1); break;
            case ZYDIS_MNEMONIC_VPUNPCKHDQ:  c.vpunpckhdq (ymm0, ymm0, ymm1); break;
            case ZYDIS_MNEMONIC_VPUNPCKLWD:  c.vpunpcklwd (ymm0, ymm0, ymm1); break;
            case ZYDIS_MNEMONIC_VPUNPCKHWD:  c.vpunpckhwd (ymm0, ymm0, ymm1); break;
            case ZYDIS_MNEMONIC_VPUNPCKLBW:  c.vpunpcklbw (ymm0, ymm0, ymm1); break;
            case ZYDIS_MNEMONIC_VPUNPCKHBW:  c.vpunpckhbw (ymm0, ymm0, ymm1); break;
            default: return false;
        }
        c.vmovdqu(ptr[r13 + YmmChunkOffset(dst_idx, 0)], ymm0);
    } else {
        switch (m) {
            case ZYDIS_MNEMONIC_VPUNPCKLQDQ: c.vpunpcklqdq(xmm0, xmm0, xmm1); break;
            case ZYDIS_MNEMONIC_VPUNPCKHQDQ: c.vpunpckhqdq(xmm0, xmm0, xmm1); break;
            case ZYDIS_MNEMONIC_VPUNPCKLDQ:  c.vpunpckldq (xmm0, xmm0, xmm1); break;
            case ZYDIS_MNEMONIC_VPUNPCKHDQ:  c.vpunpckhdq (xmm0, xmm0, xmm1); break;
            case ZYDIS_MNEMONIC_VPUNPCKLWD:  c.vpunpcklwd (xmm0, xmm0, xmm1); break;
            case ZYDIS_MNEMONIC_VPUNPCKHWD:  c.vpunpckhwd (xmm0, xmm0, xmm1); break;
            case ZYDIS_MNEMONIC_VPUNPCKLBW:  c.vpunpcklbw (xmm0, xmm0, xmm1); break;
            case ZYDIS_MNEMONIC_VPUNPCKHBW:  c.vpunpckhbw (xmm0, xmm0, xmm1); break;
            default: return false;
        }
        c.vmovdqu(ptr[r13 + YmmChunkOffset(dst_idx, 0)], xmm0);
        // VEX-128 zeroes bits 255:128.
        c.xor_(rax, rax);
        c.mov(qword[r13 + YmmChunkOffset(dst_idx, 2)], rax);
        c.mov(qword[r13 + YmmChunkOffset(dst_idx, 3)], rax);
    }
    return true;
}

/// VPHADDD — horizontal add of packed 32-bit integers. Within each
/// 128-bit lane: dst = { s1.d0+s1.d1, s1.d2+s1.d3, s2.d0+s2.d1,
/// s2.d2+s2.d3 }. We stage into host xmm/ymm 0/1 and emit the host
/// VPHADDD. VEX-128 zeroes bits 255:128.
bool EmitVphaddd(const ZydisDecodedInstruction& insn,
                 const ZydisDecodedOperand* ops,
                 u64 next_rip,
                 Xbyak::CodeGenerator& c) {
    if (insn.mnemonic != ZYDIS_MNEMONIC_VPHADDD) return false;

    int dst_idx, src1_idx;
    const ZydisDecodedOperand* src2 = nullptr;
    if (insn.operand_count_visible == 3) {
        if (ops[0].type != ZYDIS_OPERAND_TYPE_REGISTER ||
            ops[1].type != ZYDIS_OPERAND_TYPE_REGISTER) return false;
        dst_idx  = ZydisVecToIndex(ops[0].reg.value);
        src1_idx = ZydisVecToIndex(ops[1].reg.value);
        src2     = &ops[2];
    } else if (insn.operand_count_visible == 2) {
        if (ops[0].type != ZYDIS_OPERAND_TYPE_REGISTER) return false;
        dst_idx  = ZydisVecToIndex(ops[0].reg.value);
        src1_idx = dst_idx;
        src2     = &ops[1];
    } else {
        return false;
    }
    if (dst_idx < 0 || src1_idx < 0) return false;

    const int vec_bits = ops[0].size;
    if (vec_bits != 128 && vec_bits != 256) return false;
    const bool ymm = (vec_bits == 256);

    if (ymm) c.vmovdqu(ymm0, ptr[r13 + YmmChunkOffset(src1_idx, 0)]);
    else     c.vmovdqu(xmm0, ptr[r13 + YmmChunkOffset(src1_idx, 0)]);

    if (src2->type == ZYDIS_OPERAND_TYPE_REGISTER) {
        const int s2 = ZydisVecToIndex(src2->reg.value);
        if (s2 < 0) return false;
        if (ymm) c.vmovdqu(ymm1, ptr[r13 + YmmChunkOffset(s2, 0)]);
        else     c.vmovdqu(xmm1, ptr[r13 + YmmChunkOffset(s2, 0)]);
    } else if (src2->type == ZYDIS_OPERAND_TYPE_MEMORY) {
        if (ymm) return false;
        if (!EmitEffectiveAddress(src2->mem, next_rip, c)) return false;
        c.vmovdqu(xmm1, ptr[rdx]);
    } else {
        return false;
    }

    if (ymm) {
        c.vphaddd(ymm0, ymm0, ymm1);
        c.vmovdqu(ptr[r13 + YmmChunkOffset(dst_idx, 0)], ymm0);
    } else {
        c.vphaddd(xmm0, xmm0, xmm1);
        c.vmovdqu(ptr[r13 + YmmChunkOffset(dst_idx, 0)], xmm0);
        c.xor_(rax, rax);
        c.mov(qword[r13 + YmmChunkOffset(dst_idx, 2)], rax);
        c.mov(qword[r13 + YmmChunkOffset(dst_idx, 3)], rax);
    }
    return true;
}

/// VPSHUFD / VPSHUFLW / VPSHUFHW dst, src, imm8 — single-source in-lane
/// shuffle with imm8 control. VPSHUFD shuffles the four dwords of each
/// 128-bit lane; VPSHUFLW shuffles the low four words (high four copied);
/// VPSHUFHW shuffles the high four words (low four copied). All share the
/// same (dst, src, imm) shape and the same host staging. We stage src
/// into host xmm/ymm1 and emit the matching host instruction with the
/// same imm. VEX-128 zeroes the upper YMM half.
bool EmitVpshufd(const ZydisDecodedInstruction& insn,
                 const ZydisDecodedOperand* ops,
                 u64 next_rip,
                 Xbyak::CodeGenerator& c) {
    const ZydisMnemonic m = insn.mnemonic;
    if (m != ZYDIS_MNEMONIC_VPSHUFD &&
        m != ZYDIS_MNEMONIC_VPSHUFLW &&
        m != ZYDIS_MNEMONIC_VPSHUFHW) {
        return false;
    }
    if (ops[0].type != ZYDIS_OPERAND_TYPE_REGISTER) return false;
    const int dst_idx = ZydisVecToIndex(ops[0].reg.value);
    if (dst_idx < 0) return false;

    // imm8 control is the last operand.
    u8 imm = 0;
    if (ops[2].type == ZYDIS_OPERAND_TYPE_IMMEDIATE) {
        imm = static_cast<u8>(ops[2].imm.value.u & 0xFF);
    } else {
        return false;
    }

    const int vec_bits = ops[0].size;
    if (vec_bits != 128 && vec_bits != 256) return false;
    const bool ymm = (vec_bits == 256);

    // Stage src (reg or mem) into host xmm1/ymm1.
    if (ops[1].type == ZYDIS_OPERAND_TYPE_REGISTER) {
        const int s = ZydisVecToIndex(ops[1].reg.value);
        if (s < 0) return false;
        if (ymm) c.vmovdqu(ymm1, ptr[r13 + YmmChunkOffset(s, 0)]);
        else     c.vmovdqu(xmm1, ptr[r13 + YmmChunkOffset(s, 0)]);
    } else if (ops[1].type == ZYDIS_OPERAND_TYPE_MEMORY) {
        if (!EmitEffectiveAddress(ops[1].mem, next_rip, c)) return false;
        if (ymm) c.vmovdqu(ymm1, ptr[rdx]);
        else     c.vmovdqu(xmm1, ptr[rdx]);
    } else {
        return false;
    }

#define VPSHUF_EMIT(dstreg, srcreg)                                  \
    do {                                                             \
        switch (m) {                                                 \
            case ZYDIS_MNEMONIC_VPSHUFD:  c.vpshufd(dstreg, srcreg, imm);  break; \
            case ZYDIS_MNEMONIC_VPSHUFLW: c.vpshuflw(dstreg, srcreg, imm); break; \
            case ZYDIS_MNEMONIC_VPSHUFHW: c.vpshufhw(dstreg, srcreg, imm); break; \
            default: break;                                          \
        }                                                            \
    } while (0)

    if (ymm) {
        VPSHUF_EMIT(ymm0, ymm1);
        c.vmovdqu(ptr[r13 + YmmChunkOffset(dst_idx, 0)], ymm0);
    } else {
        VPSHUF_EMIT(xmm0, xmm1);
        c.vmovdqu(ptr[r13 + YmmChunkOffset(dst_idx, 0)], xmm0);
        c.xor_(rax, rax);
        c.mov(qword[r13 + YmmChunkOffset(dst_idx, 2)], rax);
        c.mov(qword[r13 + YmmChunkOffset(dst_idx, 3)], rax);
    }
#undef VPSHUF_EMIT
    return true;
}

/// VPERMILPS / VPERMILPD — in-lane permute of packed single/double floats.
/// Two encodings each, both in-lane (the permute never crosses a 128-bit
/// lane boundary, so the 256-bit form is just two independent lanes):
///
///   imm8 form:      vpermilp{s,d} dst, src, imm8
///   variable form:  vpermilp{s,d} dst, src, ctrl   (ctrl a vector reg/mem)
///
/// PS permutes 32-bit elements (selector = 2 bits each); PD permutes
/// 64-bit elements (selector = 1 bit each). We stage operands into host
/// xmm0/xmm1 and emit the matching host instruction — identical
/// semantics, so no manual element shuffling is needed. Mirrors the
/// staging approach used by VPSHUFD / VPSHUFB. VEX-128 zeroes bits
/// 255:128 of the destination.
bool EmitVpermilps(const ZydisDecodedInstruction& insn,
                   const ZydisDecodedOperand* ops,
                   u64 next_rip,
                   Xbyak::CodeGenerator& c) {
    const bool is_pd = (insn.mnemonic == ZYDIS_MNEMONIC_VPERMILPD);
    if (insn.mnemonic != ZYDIS_MNEMONIC_VPERMILPS && !is_pd) return false;
    if (ops[0].type != ZYDIS_OPERAND_TYPE_REGISTER) return false;
    const int dst_idx = ZydisVecToIndex(ops[0].reg.value);
    if (dst_idx < 0) return false;

    const int vec_bits = ops[0].size;
    if (vec_bits != 128 && vec_bits != 256) return false;
    const bool ymm = (vec_bits == 256);

    // Operand 1 is the source vector (register or memory).
    auto stage_src = [&](const Xbyak::Ymm& yreg, const Xbyak::Xmm& xreg) -> bool {
        if (ops[1].type == ZYDIS_OPERAND_TYPE_REGISTER) {
            const int s = ZydisVecToIndex(ops[1].reg.value);
            if (s < 0) return false;
            if (ymm) c.vmovdqu(yreg, ptr[r13 + YmmChunkOffset(s, 0)]);
            else     c.vmovdqu(xreg, ptr[r13 + YmmChunkOffset(s, 0)]);
        } else if (ops[1].type == ZYDIS_OPERAND_TYPE_MEMORY) {
            if (!EmitEffectiveAddress(ops[1].mem, next_rip, c)) return false;
            if (ymm) c.vmovdqu(yreg, ptr[rdx]);
            else     c.vmovdqu(xreg, ptr[rdx]);
        } else {
            return false;
        }
        return true;
    };

    // Operand 2 is the control: imm8 OR a vector (register/memory).
    if (ops[2].type == ZYDIS_OPERAND_TYPE_IMMEDIATE) {
        const u8 imm = static_cast<u8>(ops[2].imm.value.u & 0xFF);
        if (!stage_src(ymm1, xmm1)) return false;
        if (ymm) {
            if (is_pd) c.vpermilpd(ymm0, ymm1, imm);
            else       c.vpermilps(ymm0, ymm1, imm);
            c.vmovdqu(ptr[r13 + YmmChunkOffset(dst_idx, 0)], ymm0);
        } else {
            if (is_pd) c.vpermilpd(xmm0, xmm1, imm);
            else       c.vpermilps(xmm0, xmm1, imm);
            c.vmovdqu(ptr[r13 + YmmChunkOffset(dst_idx, 0)], xmm0);
        }
    } else {
        // Variable form: control is a vector operand. Stage src into
        // xmm0/ymm0 and control into xmm1/ymm1, run host vpermilps.
        // src -> 0
        if (ops[1].type == ZYDIS_OPERAND_TYPE_REGISTER) {
            const int s = ZydisVecToIndex(ops[1].reg.value);
            if (s < 0) return false;
            if (ymm) c.vmovdqu(ymm0, ptr[r13 + YmmChunkOffset(s, 0)]);
            else     c.vmovdqu(xmm0, ptr[r13 + YmmChunkOffset(s, 0)]);
        } else if (ops[1].type == ZYDIS_OPERAND_TYPE_MEMORY) {
            if (!EmitEffectiveAddress(ops[1].mem, next_rip, c)) return false;
            if (ymm) c.vmovdqu(ymm0, ptr[rdx]);
            else     c.vmovdqu(xmm0, ptr[rdx]);
        } else {
            return false;
        }
        // control -> 1
        if (ops[2].type == ZYDIS_OPERAND_TYPE_REGISTER) {
            const int ctl = ZydisVecToIndex(ops[2].reg.value);
            if (ctl < 0) return false;
            if (ymm) c.vmovdqu(ymm1, ptr[r13 + YmmChunkOffset(ctl, 0)]);
            else     c.vmovdqu(xmm1, ptr[r13 + YmmChunkOffset(ctl, 0)]);
        } else if (ops[2].type == ZYDIS_OPERAND_TYPE_MEMORY) {
            if (!EmitEffectiveAddress(ops[2].mem, next_rip, c)) return false;
            if (ymm) c.vmovdqu(ymm1, ptr[rdx]);
            else     c.vmovdqu(xmm1, ptr[rdx]);
        } else {
            return false;
        }
        if (ymm) {
            if (is_pd) c.vpermilpd(ymm0, ymm0, ymm1);
            else       c.vpermilps(ymm0, ymm0, ymm1);
            c.vmovdqu(ptr[r13 + YmmChunkOffset(dst_idx, 0)], ymm0);
        } else {
            if (is_pd) c.vpermilpd(xmm0, xmm0, xmm1);
            else       c.vpermilps(xmm0, xmm0, xmm1);
            c.vmovdqu(ptr[r13 + YmmChunkOffset(dst_idx, 0)], xmm0);
        }
    }

    // VEX-128 zeroes the upper 128 bits of the destination YMM.
    if (!ymm) {
        c.xor_(rax, rax);
        c.mov(qword[r13 + YmmChunkOffset(dst_idx, 2)], rax);
        c.mov(qword[r13 + YmmChunkOffset(dst_idx, 3)], rax);
    }
    return true;
}

/// VPTEST a, b (128- or 256-bit; also legacy PTEST) — bitwise test
/// that defines only ZF and CF:
///   ZF = ((a AND b) == 0)
///   CF = ((NOT a AND b) == 0)
/// OF/SF/PF/AF are cleared. We stage the operands into host xmm/ymm
/// 0/1, run the host VPTEST, then merge ZF/CF from the host result
/// into guest rflags and clear the other defined bits.
bool EmitVptest(const ZydisDecodedInstruction& insn,
                const ZydisDecodedOperand* ops,
                u64 next_rip,
                Xbyak::CodeGenerator& c) {
    if (insn.mnemonic != ZYDIS_MNEMONIC_VPTEST &&
        insn.mnemonic != ZYDIS_MNEMONIC_PTEST) {
        return false;
    }
    if (ops[0].type != ZYDIS_OPERAND_TYPE_REGISTER) return false;
    const int a_idx = ZydisVecToIndex(ops[0].reg.value);
    if (a_idx < 0) return false;

    const int vec_bits = ops[0].size;
    if (vec_bits != 128 && vec_bits != 256) return false;

    if (vec_bits == 128) {
        c.vmovdqu(xmm0, ptr[r13 + YmmChunkOffset(a_idx, 0)]);
        if (ops[1].type == ZYDIS_OPERAND_TYPE_REGISTER) {
            const int b_idx = ZydisVecToIndex(ops[1].reg.value);
            if (b_idx < 0) return false;
            c.vmovdqu(xmm1, ptr[r13 + YmmChunkOffset(b_idx, 0)]);
        } else if (ops[1].type == ZYDIS_OPERAND_TYPE_MEMORY) {
            if (!EmitEffectiveAddress(ops[1].mem, next_rip, c)) return false;
            c.vmovdqu(xmm1, ptr[rdx]);
        } else {
            return false;
        }
        c.vptest(xmm0, xmm1);
    } else {
        if (ops[1].type != ZYDIS_OPERAND_TYPE_REGISTER) return false;
        const int b_idx = ZydisVecToIndex(ops[1].reg.value);
        if (b_idx < 0) return false;
        c.vmovdqu(ymm0, ptr[r13 + YmmChunkOffset(a_idx, 0)]);
        c.vmovdqu(ymm1, ptr[r13 + YmmChunkOffset(b_idx, 0)]);
        c.vptest(ymm0, ymm1);
    }

    // Merge ZF/CF from host rflags; clear the bits VPTEST defines.
    c.pushfq();
    c.pop(rax);                               // rax = host rflags
    using namespace RflagsBits;
    constexpr u64 AF = 1ULL << 4;
    const u64 zf_cf   = ZF | CF;
    const u64 cleared = ZF | CF | OF | SF | PF | AF;
    c.mov(rdx, qword[r13 + Offsets::Rflags]);
    c.mov(rcx, ~cleared);
    c.and_(rdx, rcx);
    c.mov(rcx, zf_cf);
    c.and_(rax, rcx);
    c.or_(rdx, rax);
    c.mov(qword[r13 + Offsets::Rflags], rdx);
    return true;
}

/// VPCMPISTRI xmm1, xmm2/m128, imm8 (and legacy PCMPISTRI) — SSE4.2
/// "packed compare implicit-length strings, return index". The control
/// imm8 selects the comparison mode; the instruction writes the result
/// index to ECX and sets ZF/CF/SF/OF accordingly. The string functions
/// in the PS4 libc use this for strlen/strchr-style scans.
///
/// Implementing the full PCMPISTRI control matrix by hand is large and
/// error-prone, so we emit the host instruction directly on host
/// xmm0/xmm1 with the same imm8, then copy host ECX into guest ECX
/// (zero-extended) and capture the flags. Correct for every control
/// byte, not just the strlen idiom.
bool EmitVpcmpistri(const ZydisDecodedInstruction& insn,
                    const ZydisDecodedOperand* ops,
                    u64 next_rip,
                    Xbyak::CodeGenerator& c) {
    const bool is_vex = (insn.mnemonic == ZYDIS_MNEMONIC_VPCMPISTRI);
    const bool is_sse = (insn.mnemonic == ZYDIS_MNEMONIC_PCMPISTRI);
    if (!is_vex && !is_sse) return false;
    if (ops[0].type != ZYDIS_OPERAND_TYPE_REGISTER) return false;
    const int a_idx = ZydisVecToIndex(ops[0].reg.value);
    if (a_idx < 0) return false;

    // The imm8 control is the last operand. Zydis exposes it as an
    // immediate; find it (it's ops[2] for the 3-operand syntax).
    u8 imm = 0;
    if (ops[2].type == ZYDIS_OPERAND_TYPE_IMMEDIATE) {
        imm = static_cast<u8>(ops[2].imm.value.u & 0xFF);
    } else if (ops[1].type == ZYDIS_OPERAND_TYPE_IMMEDIATE) {
        imm = static_cast<u8>(ops[1].imm.value.u & 0xFF);
    } else {
        return false;
    }

    // Stage operand a (xmm0) and operand b (xmm1).
    c.vmovdqu(xmm0, ptr[r13 + YmmChunkOffset(a_idx, 0)]);
    if (ops[1].type == ZYDIS_OPERAND_TYPE_REGISTER) {
        const int b_idx = ZydisVecToIndex(ops[1].reg.value);
        if (b_idx < 0) return false;
        c.vmovdqu(xmm1, ptr[r13 + YmmChunkOffset(b_idx, 0)]);
    } else if (ops[1].type == ZYDIS_OPERAND_TYPE_MEMORY) {
        if (!EmitEffectiveAddress(ops[1].mem, next_rip, c)) return false;
        c.vmovdqu(xmm1, ptr[rdx]);
    } else {
        return false;
    }

    // Host instruction writes ECX implicitly and sets flags.
    if (is_vex) c.vpcmpistri(xmm0, xmm1, imm);
    else        c.pcmpistri(xmm0, xmm1, imm);

    // Capture flags, then store host ECX → guest ECX (zero-extended).
    c.pushfq();
    c.pop(r8);
    EmitStoreArithFlags(c, r8);
    c.mov(eax, ecx);                          // zero-extends rax
    c.mov(qword[r13 + GprOffset(1)], rax);    // guest RCX/ECX
    return true;
}

/// VPCMPISTRM / PCMPISTRM — packed compare implicit-length strings,
/// returning a MASK in XMM0 (vs the index-in-ECX of the ...ISTRI form).
/// imm8[6] selects byte-mask (0xFF/0x00 per element) vs bit-mask (low
/// bits of XMM0). The host instruction writes XMM0 implicitly and sets
/// flags. We stage operand a in xmm2 and b in xmm1 (leaving host XMM0
/// free as the destination), then copy host XMM0 to the guest xmm0 slot.
bool EmitVpcmpistrm(const ZydisDecodedInstruction& insn,
                    const ZydisDecodedOperand* ops,
                    u64 next_rip,
                    Xbyak::CodeGenerator& c) {
    const bool is_vex = (insn.mnemonic == ZYDIS_MNEMONIC_VPCMPISTRM);
    const bool is_sse = (insn.mnemonic == ZYDIS_MNEMONIC_PCMPISTRM);
    if (!is_vex && !is_sse) return false;
    if (ops[0].type != ZYDIS_OPERAND_TYPE_REGISTER) return false;
    const int a_idx = ZydisVecToIndex(ops[0].reg.value);
    if (a_idx < 0) return false;

    u8 imm = 0;
    if (ops[2].type == ZYDIS_OPERAND_TYPE_IMMEDIATE) {
        imm = static_cast<u8>(ops[2].imm.value.u & 0xFF);
    } else if (ops[1].type == ZYDIS_OPERAND_TYPE_IMMEDIATE) {
        imm = static_cast<u8>(ops[1].imm.value.u & 0xFF);
    } else {
        return false;
    }

    c.vmovdqu(xmm2, ptr[r13 + YmmChunkOffset(a_idx, 0)]);   // operand a
    if (ops[1].type == ZYDIS_OPERAND_TYPE_REGISTER) {
        const int b_idx = ZydisVecToIndex(ops[1].reg.value);
        if (b_idx < 0) return false;
        c.vmovdqu(xmm1, ptr[r13 + YmmChunkOffset(b_idx, 0)]);
    } else if (ops[1].type == ZYDIS_OPERAND_TYPE_MEMORY) {
        if (!EmitEffectiveAddress(ops[1].mem, next_rip, c)) return false;
        c.vmovdqu(xmm1, ptr[rdx]);
    } else {
        return false;
    }

    if (is_vex) c.vpcmpistrm(xmm2, xmm1, imm);   // writes XMM0 implicitly
    else        c.pcmpistrm(xmm2, xmm1, imm);

    c.pushfq();
    c.pop(r8);
    EmitStoreArithFlags(c, r8);

    // Copy host XMM0 (the result mask) to the guest xmm0 slot; the mask
    // is 128-bit, upper YMM unaffected by the legacy/VEX.128 form -> zero.
    c.vmovdqu(ptr[r13 + YmmChunkOffset(0, 0)], xmm0);
    c.xor_(rax, rax);
    c.mov(qword[r13 + YmmChunkOffset(0, 2)], rax);
    c.mov(qword[r13 + YmmChunkOffset(0, 3)], rax);
    return true;
}

/// String operations: STOS / MOVS / LODS / SCAS / CMPS, in their 8/16/
/// 32/64-bit element widths, with optional REP / REPE / REPNE prefixes.
///
/// Register roles (x86-64): RDI(7)=dest pointer, RSI(6)=source pointer,
/// RAX(0)=accumulator (for stos/lods/scas), RCX(1)=rep count. The
/// direction flag (RFLAGS bit 10) selects forward (+) or backward (-)
/// advance of RSI/RDI by the element size.
///
/// REP repeats RCX times; REPE/REPNE additionally terminate early when
/// the per-iteration compare's ZF disagrees with the prefix (REPE stops
/// when ZF=0, REPNE stops when ZF=1). SCAS/CMPS set arithmetic flags via
/// the materialized-from-result helper; STOS/MOVS/LODS leave flags alone.
///
/// Pointer guard: before dereferencing RSI/RDI, near-null pointers (below
/// the conventional low-address reservation) bail cleanly with
/// HelperRequestedExit, leaving all guest state untouched so the host can
/// produce a crash report. Valid high pointers proceed.
bool EmitStringOp(const ZydisDecodedInstruction& insn,
                  const ZydisDecodedOperand* /*ops*/,
                  u64 next_rip,
                  Xbyak::CodeGenerator& c) {
    const ZydisMnemonic m = insn.mnemonic;

    // MOVSD / CMPSD mnemonics are shared between the string ops (implicit
    // RSI/RDI, no visible operands) and the legacy SSE scalar-double ops
    // (explicit xmm operands). Disambiguate by visible-operand count: the
    // true string forms expose none.
    if (insn.operand_count_visible != 0) return false;

    enum class Kind { Stos, Movs, Lods, Scas, Cmps } kind;
    bool uses_rsi = false, uses_rdi = false, is_cmp = false;
    switch (m) {
        case ZYDIS_MNEMONIC_STOSB: case ZYDIS_MNEMONIC_STOSW:
        case ZYDIS_MNEMONIC_STOSD: case ZYDIS_MNEMONIC_STOSQ:
            kind = Kind::Stos; uses_rdi = true; break;
        case ZYDIS_MNEMONIC_MOVSB: case ZYDIS_MNEMONIC_MOVSW:
        case ZYDIS_MNEMONIC_MOVSD: case ZYDIS_MNEMONIC_MOVSQ:
            kind = Kind::Movs; uses_rsi = uses_rdi = true; break;
        case ZYDIS_MNEMONIC_LODSB: case ZYDIS_MNEMONIC_LODSW:
        case ZYDIS_MNEMONIC_LODSD: case ZYDIS_MNEMONIC_LODSQ:
            kind = Kind::Lods; uses_rsi = true; break;
        case ZYDIS_MNEMONIC_SCASB: case ZYDIS_MNEMONIC_SCASW:
        case ZYDIS_MNEMONIC_SCASD: case ZYDIS_MNEMONIC_SCASQ:
            kind = Kind::Scas; uses_rdi = true; is_cmp = true; break;
        case ZYDIS_MNEMONIC_CMPSB: case ZYDIS_MNEMONIC_CMPSW:
        case ZYDIS_MNEMONIC_CMPSD: case ZYDIS_MNEMONIC_CMPSQ:
            kind = Kind::Cmps; uses_rsi = uses_rdi = true; is_cmp = true; break;
        default: return false;
    }

    // Element width in bytes from operand_width.
    int esz;
    switch (insn.operand_width) {
        case 8:  esz = 1; break;
        case 16: esz = 2; break;
        case 32: esz = 4; break;
        case 64: esz = 8; break;
        default: return false;
    }

    const bool has_rep   = (insn.attributes & ZYDIS_ATTRIB_HAS_REP)  != 0;
    const bool has_repe  = (insn.attributes & ZYDIS_ATTRIB_HAS_REPE) != 0;
    const bool has_repne = (insn.attributes & ZYDIS_ATTRIB_HAS_REPNE)!= 0;
    const bool repeated  = has_rep || has_repe || has_repne;

    constexpr int RAX_IDX = 0, RCX_IDX = 1, RSI_IDX = 6, RDI_IDX = 7;
    const u64 cur_rip = next_rip - insn.length;

    Xbyak::Label loop_top, loop_done, do_bail;

    // ---- REP setup: if RCX==0, skip the whole thing. ----
    if (repeated) {
        c.mov(rcx, qword[r13 + GprOffset(RCX_IDX)]);
        c.test(rcx, rcx);
        c.jz(loop_done, c.T_NEAR);
    }

    c.L(loop_top);

    // ---- Pointer guard on the live RSI/RDI before any deref. ----
    // Near-null pointers (below 0x10000) bail cleanly. Real guest
    // mappings sit far above this.
    constexpr u64 kMinGuestPtr = 0x10000;
    if (uses_rdi) {
        c.mov(rax, qword[r13 + GprOffset(RDI_IDX)]);
        c.cmp(rax, kMinGuestPtr);
        c.jb(do_bail, c.T_NEAR);
    }
    if (uses_rsi) {
        c.mov(rax, qword[r13 + GprOffset(RSI_IDX)]);
        c.cmp(rax, kMinGuestPtr);
        c.jb(do_bail, c.T_NEAR);
    }

    // ---- The per-iteration body. ----
    // Load pointers we need into r10 (dest/RDI) and r11 (src/RSI).
    if (uses_rdi) c.mov(r10, qword[r13 + GprOffset(RDI_IDX)]);
    if (uses_rsi) c.mov(r11, qword[r13 + GprOffset(RSI_IDX)]);

    auto load_acc = [&](Xbyak::Reg64 dstreg, Xbyak::Reg64 ptrreg) {
        switch (esz) {
            case 1: c.movzx(dstreg, c.byte[ptrreg]); break;
            case 2: c.movzx(dstreg, c.word[ptrreg]); break;
            case 4: c.mov(Xbyak::Reg32(dstreg.getIdx()), c.dword[ptrreg]); break;
            case 8: c.mov(dstreg, c.qword[ptrreg]); break;
        }
    };

    if (kind == Kind::Stos) {
        // [RDI] = AL/AX/EAX/RAX
        c.mov(rax, qword[r13 + GprOffset(RAX_IDX)]);
        switch (esz) {
            case 1: c.mov(c.byte[r10],  al);  break;
            case 2: c.mov(c.word[r10],  ax);  break;
            case 4: c.mov(c.dword[r10], eax); break;
            case 8: c.mov(c.qword[r10], rax); break;
        }
    } else if (kind == Kind::Movs) {
        // [RDI] = [RSI]
        switch (esz) {
            case 1: c.mov(al,  c.byte[r11]);  c.mov(c.byte[r10],  al);  break;
            case 2: c.mov(ax,  c.word[r11]);  c.mov(c.word[r10],  ax);  break;
            case 4: c.mov(eax, c.dword[r11]); c.mov(c.dword[r10], eax); break;
            case 8: c.mov(rax, c.qword[r11]); c.mov(c.qword[r10], rax); break;
        }
    } else if (kind == Kind::Lods) {
        // AL/AX/EAX/RAX = [RSI], preserving upper bits for 8/16.
        if (esz == 1) {
            c.mov(al, c.byte[r11]);
            c.mov(byte[r13 + GprOffset(RAX_IDX)], al);  // preserve upper 56
        } else if (esz == 2) {
            c.mov(ax, c.word[r11]);
            c.mov(word[r13 + GprOffset(RAX_IDX)], ax);  // preserve upper 48
        } else if (esz == 4) {
            c.mov(eax, c.dword[r11]);
            c.mov(qword[r13 + GprOffset(RAX_IDX)], rax); // zero-extend
        } else {
            c.mov(rax, c.qword[r11]);
            c.mov(qword[r13 + GprOffset(RAX_IDX)], rax);
        }
    } else if (kind == Kind::Scas) {
        // compare AL/AX/EAX/RAX - [RDI]; set flags.
        c.mov(rcx, qword[r13 + GprOffset(RAX_IDX)]);   // lhs = acc
        load_acc(rdx, r10);                      // rhs = [RDI]
        // sub for flags (result in rax), via the subtract flag helper.
        if (esz <= 4) {
            // operate at element width: mask both to esz bits then sub.
            c.mov(rax, rcx);
            switch (esz) {
                case 1: c.and_(rax, 0xFF); c.and_(rdx, 0xFF); break;
                case 2: c.and_(rax, 0xFFFF); c.and_(rdx, 0xFFFF); break;
                case 4: c.mov(eax, ecx); c.mov(edx, edx); break;
            }
            c.sub(rax, rdx);
        } else {
            c.mov(rax, rcx);
            c.sub(rax, rdx);
        }
        // Width matters: with the operands masked to esz bytes, a default
        // 64-bit derivation computes SF/OF from bit 63 instead of bit
        // esz*8-1 -- breaking signed jcc after SCASB/W/D.
        EmitFlagsFromSubtract(c, static_cast<u32>(esz) * 8);
    } else { // Cmps
        // compare [RSI] - [RDI]; set flags.
        load_acc(rcx, r11);   // lhs = [RSI]
        load_acc(rdx, r10);   // rhs = [RDI]
        c.mov(rax, rcx);
        c.sub(rax, rdx);
        EmitFlagsFromSubtract(c, static_cast<u32>(esz) * 8);
    }

    // ---- Advance pointers per DF. ----
    // DF in rflags bit 10. delta = DF ? -esz : +esz.
    {
        Xbyak::Label df_set, adv_done;
        c.mov(rax, qword[r13 + Offsets::Rflags]);
        c.bt(rax, 10);
        c.jc(df_set);
        // forward
        if (uses_rdi) { c.add(qword[r13 + GprOffset(RDI_IDX)], esz); }
        if (uses_rsi) { c.add(qword[r13 + GprOffset(RSI_IDX)], esz); }
        c.jmp(adv_done);
        c.L(df_set);
        if (uses_rdi) { c.sub(qword[r13 + GprOffset(RDI_IDX)], esz); }
        if (uses_rsi) { c.sub(qword[r13 + GprOffset(RSI_IDX)], esz); }
        c.L(adv_done);
    }

    // ---- REP bookkeeping. ----
    if (repeated) {
        // RCX-- ; if RCX==0 done.
        c.mov(rcx, qword[r13 + GprOffset(RCX_IDX)]);
        c.sub(rcx, 1);
        c.mov(qword[r13 + GprOffset(RCX_IDX)], rcx);

        if (is_cmp && (has_repe || has_repne)) {
            // Early-exit on ZF condition. ZF is in the freshly-written
            // rflags. REPE continues while ZF=1 (stop on ZF=0);
            // REPNE continues while ZF=0 (stop on ZF=1).
            c.mov(rax, qword[r13 + Offsets::Rflags]);
            c.bt(rax, 6);   // CF = ZF
            if (has_repe) {
                // stop if ZF == 0
                c.jnc(loop_done, c.T_NEAR);
            } else {
                // repne: stop if ZF == 1
                c.jc(loop_done, c.T_NEAR);
            }
        }
        // continue if RCX != 0
        c.test(rcx, rcx);
        c.jnz(loop_top, c.T_NEAR);
    }

    c.L(loop_done);
    // Normal completion: set RIP to next instruction, exit_reason=BlockEnd,
    // return to dispatcher (it will re-enter at next_rip).
    c.mov(rax, next_rip);
    c.mov(qword[r13 + Offsets::Rip], rax);
    c.mov(dword[r13 + offsetof(GuestState, exit_reason)],
          static_cast<u32>(ExitReason::BlockEnd));
    c.jmp(r14);

    // Clean bail for a bogus pointer: preserve all state (we wrote nothing
    // this iteration before the guard), point RIP at this instruction so
    // the report identifies it, signal HelperRequestedExit, and exit to the
    // gateway via r15 (the exit stub) — NOT r14 (the dispatcher loop), which
    // would just recompile and re-run this same instruction forever.
    c.L(do_bail);
    c.mov(rax, cur_rip);
    c.mov(qword[r13 + Offsets::Rip], rax);
    c.mov(dword[r13 + offsetof(GuestState, exit_reason)],
          static_cast<u32>(ExitReason::HelperRequestedExit));
    c.jmp(r15);
    return true;
}

/// CPUID — spoofed to report an AMD Jaguar (the PS4's APU), so guest
/// code that probes the CPU sees a consistent, period-correct vendor
/// rather than whatever host the JIT happens to run on. Inputs are
/// EAX (leaf) and ECX (subleaf); outputs go to EAX/EBX/ECX/EDX, which
/// are guest GPR slots 0/3/1/2. Each 32-bit write zero-extends its
/// 64-bit slot (we store the canned dword as a qword with the upper
/// half cleared), matching how the host CPU updates the registers.
///
/// Handled leaves: 0 (vendor), 1 (signature+features), 7/sub0
/// (extended features — BMI1 only), and 0x80000002..4 (brand string
/// "AMD Custom Jaguar 8-Core APU"). Every other leaf — and leaf 7
/// with a non-zero subleaf — returns all zeros.
/// RDTSC / RDTSCP -- host passthrough.
///
/// Guest semantics: TSC low 32 -> EAX, high 32 -> EDX, upper halves of
/// RAX/RDX zeroed; RDTSCP additionally writes IA32_TSC_AUX -> ECX. No
/// flags. The host instruction produces exactly this shape, so the
/// values stream straight into the guest slots.
///
/// Passthrough (rather than a synthesized counter) is deliberate: the
/// kernel HLE exposes the host TSC via sceKernelReadTsc /
/// sceKernelGetTscFrequency, and game calibration loops compare rdtsc
/// deltas against those -- the two MUST read the same counter or
/// guest timing code diverges.
bool EmitRdtsc(const ZydisDecodedInstruction& insn,
               const ZydisDecodedOperand* ops,
               Xbyak::CodeGenerator& c,
               bool with_aux) {
    (void)insn;
    (void)ops;
    if (with_aux) {
        c.rdtscp();  // edx:eax = TSC, ecx = TSC_AUX
        c.mov(qword[r13 + GprOffset(1)], rcx);
    } else {
        c.rdtsc();   // edx:eax = TSC
    }
    c.mov(qword[r13 + GprOffset(0)], rax);
    c.mov(qword[r13 + GprOffset(2)], rdx);
    return true;
}


/// LAHF / SAHF -- AH <-> low flags byte.
///
/// LAHF composes AH = SF:ZF:0:AF:0:PF:1:CF from state.rflags; SAHF
/// writes those five flag bits from AH back into state.rflags,
/// preserving OF (bit 11), DF, and everything else above bit 7. Both
/// are direct byte operations against the guest RAX slot's AH byte
/// (GprOffset(0)+1) -- no full-register round trip, and no host-flag
/// round trip (the and/or here clobber only HOST flags, which carry
/// no guest state between emitted sequences).
///
/// AF fidelity: the helper-computed flag paths skip AF (see the flag
/// model comment above RflagsBits), so LAHF reports AF=0 after ops
/// lifted through those helpers but a real AF after emitters that
/// capture host flags via pushfq (mask 0x8D5 includes AF). Same
/// tradeoff the rest of the JIT already makes; LAHF consumers in the
/// wild use it for CF/ZF save-restore, not AF.
bool EmitLahf(const ZydisDecodedInstruction& insn,
              const ZydisDecodedOperand* ops,
              Xbyak::CodeGenerator& c) {
    (void)insn;
    (void)ops;
    c.mov(rdx, qword[r13 + Offsets::Rflags]);
    c.and_(edx, 0xD5);                            // SF|ZF|AF|PF|CF
    c.or_(edx, 0x02);                             // bit 1 reads as 1
    c.mov(byte[r13 + GprOffset(0) + 1], dl);      // AH of guest RAX
    return true;
}

bool EmitSahf(const ZydisDecodedInstruction& insn,
              const ZydisDecodedOperand* ops,
              Xbyak::CodeGenerator& c) {
    (void)insn;
    (void)ops;
    c.movzx(edx, byte[r13 + GprOffset(0) + 1]);   // AH of guest RAX
    c.and_(edx, 0xD5);                            // only SF|ZF|AF|PF|CF transfer
    // imm32 sign-extends: clears bits 0,2,4,6,7 and preserves all
    // higher bits (OF, DF, ...) -- same encoding trick as
    // EmitStoreArithFlags.
    c.and_(qword[r13 + Offsets::Rflags], static_cast<Xbyak::uint32>(~0xD5u));
    c.or_(qword[r13 + Offsets::Rflags], rdx);
    return true;
}

bool EmitCpuid(const ZydisDecodedInstruction& insn,
               const ZydisDecodedOperand* ops,
               Xbyak::CodeGenerator& c) {
    (void)ops;
    (void)insn;

    // Write helper: store a canned (EAX,EBX,ECX,EDX) quad into the
    // guest GPR slots (0=EAX, 3=EBX, 1=ECX, 2=EDX). Each value is
    // materialized into eax then stored as a zero-extended qword so
    // the upper 32 bits of every slot are cleared.
    auto put = [&](u32 v_eax, u32 v_ebx, u32 v_ecx, u32 v_edx) {
        c.mov(eax, v_eax); c.mov(qword[r13 + GprOffset(0)], rax);
        c.mov(eax, v_ebx); c.mov(qword[r13 + GprOffset(3)], rax);
        c.mov(eax, v_ecx); c.mov(qword[r13 + GprOffset(1)], rax);
        c.mov(eax, v_edx); c.mov(qword[r13 + GprOffset(2)], rax);
    };

    // Keep the requested leaf in r8d and subleaf in r9d so put() is
    // free to reuse eax/rax.
    c.mov(r8d, dword[r13 + GprOffset(0)]);   // leaf    = EAX
    c.mov(r9d, dword[r13 + GprOffset(1)]);   // subleaf = ECX

    Xbyak::Label l1, l7, lb2, lb3, lb4, l_default, l_done;

    // Leaf 0 — vendor "AuthenticAMD", max standard leaf = 7.
    c.cmp(r8d, 0x00000000u);
    c.jne(l1, Xbyak::CodeGenerator::T_NEAR);
    put(0x00000007u, 0x68747541u, 0x444D4163u, 0x69746E65u);
    c.jmp(l_done, Xbyak::CodeGenerator::T_NEAR);

    // Leaf 1 — Jaguar signature + advertised features.
    c.L(l1);
    c.cmp(r8d, 0x00000001u);
    c.jne(l7, Xbyak::CodeGenerator::T_NEAR);
    {
        const u32 sig  = 0x00700F01u;
        const u32 ecxf = (1u<<0)|(1u<<19)|(1u<<20)|(1u<<23)|(1u<<28);
        const u32 edxf = (1u<<0)|(1u<<25)|(1u<<26);
        put(sig, 0x00000000u, ecxf, edxf);
    }
    c.jmp(l_done, Xbyak::CodeGenerator::T_NEAR);

    // Leaf 7 — extended features; only subleaf 0 responds.
    c.L(l7);
    c.cmp(r8d, 0x00000007u);
    c.jne(lb2, Xbyak::CodeGenerator::T_NEAR);
    c.test(r9d, r9d);
    c.jnz(l_default, Xbyak::CodeGenerator::T_NEAR);
    put(0x00000000u, (1u<<3), 0x00000000u, 0x00000000u);  // EBX: BMI1
    c.jmp(l_done, Xbyak::CodeGenerator::T_NEAR);

    // Brand string leaves 0x80000002..4 → "AMD Custom Jaguar 8-Core APU".
    c.L(lb2);
    c.cmp(r8d, 0x80000002u);
    c.jne(lb3, Xbyak::CodeGenerator::T_NEAR);
    put(0x20444D41u, 0x74737543u, 0x4A206D6Fu, 0x61756761u);
    c.jmp(l_done, Xbyak::CodeGenerator::T_NEAR);

    c.L(lb3);
    c.cmp(r8d, 0x80000003u);
    c.jne(lb4, Xbyak::CodeGenerator::T_NEAR);
    put(0x2D382072u, 0x65726F43u, 0x55504120u, 0x00000000u);
    c.jmp(l_done, Xbyak::CodeGenerator::T_NEAR);

    c.L(lb4);
    c.cmp(r8d, 0x80000004u);
    c.jne(l_default, Xbyak::CodeGenerator::T_NEAR);
    put(0x00000000u, 0x00000000u, 0x00000000u, 0x00000000u);
    c.jmp(l_done, Xbyak::CodeGenerator::T_NEAR);

    // Default — unknown leaf (or leaf 7 sub != 0): all zeros.
    c.L(l_default);
    put(0,0,0,0);

    c.L(l_done);
    return true;
}

} // namespace

// ============================================================================
// Lifter class — block-compile loop and dispatch
// ============================================================================

Lifter::Lifter(CodeCache& code_cache) : code_cache_(code_cache) {
    LOG_INFO(Core, "Lifter (x86 host) initialized");
}

Lifter::~Lifter() {
    // Use fprintf rather than LOG_INFO here: by the time the lifter
    // destructor runs at program shutdown, shadPS4's logging
    // subsystem has often been torn down, and LOG_INFO degrades to
    // emitting the format string verbatim with the `{}` placeholders
    // un-substituted. fprintf works at any teardown phase.
    std::fprintf(stderr,
                 "[lifter] %llu blocks compiled, %llu bytes emitted, %llu unsupported\n",
                 (unsigned long long)blocks_compiled_,
                 (unsigned long long)bytes_emitted_,
                 (unsigned long long)unsupported_hits_);
    std::fflush(stderr);
}

u64 Lifter::BlockReservationSize() noexcept {
    return BLOCK_HOST_SIZE_CAP;
}

// Function-try-block: the emit loop below calls hundreds of xbyak emitters,
// any of which can throw Xbyak::Error (ERR_CODE_IS_TOO_BIG past the margin
// check, label misuse, bad operand combinations). An escaped exception would
// unwind into the dispatcher and then into the JIT gateway frame, which C++
// unwinding cannot cross -- std::terminate with no diagnostics. The catch
// converts any throw into the nullptr compile-failure path; the dispatcher
// distinguishes it from cache exhaustion via BlockReservationSize() (a hard
// failure with cache room left is NOT retried, so a deterministic oversized
// block can't wipe the cache pointlessly). The cap-sized reservation made
// below is abandoned on throw -- it was never inserted anywhere, no claim
// ever points at it, and the next recycle reclaims it.
void* Lifter::CompileBlock(u64 guest_rip) try {
    // Diagnostic: trace the compile path via fprintf(stderr).
    //
    // We deliberately do NOT use LOG_INFO here, even though it would
    // be the natural fit. The reason: this function is called from
    // inside the gateway-dispatched code path (Runtime::Run -> gateway
    // -> dispatcher trampoline -> here). The gateway is JIT-emitted
    // x86 code with no registered Windows unwind info (.pdata /
    // .xdata). Any spdlog/fmt operation that triggers SEH stack
    // walking — RTC1 checks, RAII destructor cleanup paths, debug
    // checks — fails when the walker reaches the JIT gateway frame
    // and reads garbage from a missing function table entry.
    //
    // Empirically: LOG_INFO from constructors (before JIT execution)
    // works; LOG_INFO from inside CompileBlock crashes with
    // "access violation reading 0xFFFFFFFFFFFFFFFF". The fprintf path
    // doesn't walk the stack and is safe.
    //
    // The proper long-term fix is registering unwind info for the
    // gateway via RtlAddFunctionTable on Windows. That's a separate
    // piece of work. Until then, JIT-dispatched-context code uses
    // fprintf for diagnostics.
    // Per-block / per-instruction tracing is opt-in: two stderr writes plus
    // a flush per decoded instruction is a heavy compile-time tax and a log
    // firehose in release. The fprintf transport itself stays (see the SEH
    // rationale above for why not LOG_*); only the volume is gated. The
    // unsupported-instruction report further down remains UNCONDITIONAL --
    // it is rare and is the primary porting signal.
    static const bool lifter_trace = [] {
        const char* e = std::getenv("SHADPS4_LIFTER_TRACE");
        return e != nullptr && e[0] != '0';
    }();
    if (lifter_trace) {
        std::fprintf(stderr, "[lifter] CompileBlock: guest_rip = 0x%llx\n",
                     static_cast<unsigned long long>(guest_rip));
        std::fflush(stderr);
    }

    // Reserve a chunk of code cache for this block. We don't know
    // the final size yet; conservatively reserve the size cap and
    // commit only what we use. (For a real impl we'd use xbyak's
    // internal buffer and copy out, but the bump allocator's
    // overhead is tiny.)
    u8* code_buf = code_cache_.Allocate(BLOCK_HOST_SIZE_CAP);
    if (code_buf == nullptr) {
        std::fprintf(stderr, "[lifter] code cache full at RIP 0x%llx\n",
                     static_cast<unsigned long long>(guest_rip));
        return nullptr;
    }

    Xbyak::CodeGenerator c{BLOCK_HOST_SIZE_CAP, code_buf};

    // Init Zydis decoder. (Done once per block; could be hoisted to
    // Lifter member for efficiency, but the cost is negligible.)
    ZydisDecoder decoder;
    ZydisDecoderInit(&decoder, ZYDIS_MACHINE_MODE_LONG_64, ZYDIS_STACK_WIDTH_64);

    u64 rip = guest_rip;
    const u64 block_end_cap = guest_rip + BLOCK_GUEST_SIZE_CAP;
    bool emitted_terminator = false;

    while (rip < block_end_cap) {
        // Decode one instruction from guest memory at `rip`.
        // For this slice we trust that guest memory at the lift
        // RIP is valid and accessible. Production code adds a
        // safe-decode path that catches faults.
        ZydisDecodedInstruction insn;
        ZydisDecodedOperand ops[ZYDIS_MAX_OPERAND_COUNT];
        if (lifter_trace) {
            std::fprintf(stderr, "[lifter] about to decode at 0x%llx\n",
                         static_cast<unsigned long long>(rip));
            std::fflush(stderr);
        }
        const auto status = ZydisDecoderDecodeFull(
            &decoder, reinterpret_cast<const void*>(rip), 15,
            &insn, ops);
        if (lifter_trace) {
            std::fprintf(stderr, "[lifter] decoded at 0x%llx ok=%d mnemonic=%s\n",
                         static_cast<unsigned long long>(rip),
                         ZYAN_SUCCESS(status) ? 1 : 0,
                         ZYAN_SUCCESS(status)
                             ? ZydisMnemonicGetString(insn.mnemonic)
                             : "(decode-failed)");
            std::fflush(stderr);
        }
        if (!ZYAN_SUCCESS(status)) {
            std::fprintf(stderr, "[lifter] decode FAILED at 0x%llx\n",
                         static_cast<unsigned long long>(rip));
            std::fflush(stderr);
            ++unsupported_hits_;
            // Emit a clean exit so the host program doesn't die.
            // Use r15 (fatal exit) rather than r14 (dispatcher loop)
            // because retrying the same bad address would just loop.
            c.mov(rax, rip);
            c.mov(qword[r13 + Offsets::Rip], rax);
            c.mov(dword[r13 + offsetof(GuestState, exit_reason)],
                  static_cast<u32>(ExitReason::UnsupportedInstruction));
            c.jmp(r15);
            emitted_terminator = true;
            break;
        }

        // Compute next_rip for RIP-relative addressing.
        const u64 next_rip = rip + insn.length;

        // Dispatch on mnemonic.
        bool handled = false;
        // Address-size override (0x67) is unsupported. In 64-bit mode it
        // truncates effective-address computation to 32 bits; PS4 (LP64) code
        // never emits it. Rather than mask inconsistently across the ~100
        // EmitEffectiveAddress call sites, reject it uniformly here so any such
        // op bails as UnsupportedInstruction (visible in the log) instead of
        // being silently mis-addressed.
        if ((insn.attributes & ZYDIS_ATTRIB_HAS_LOCK) != 0 && insn.address_width != 32) {
            // Atomic read-modify-write. A LOCK-prefixed memory op must be a
            // single atomic step (guest threads run on parallel host threads
            // over shared guest memory), not the load-op-store the per-op
            // handlers emit. Try the atomic path first; if it doesn't cover
            // this op, fall through to the normal handler below — same result
            // as before, just not yet atomic.
            handled = EmitLockedRmw(insn, ops, next_rip, c);
        }
        if (handled) {
            // Already emitted atomically above.
        } else if (insn.address_width == 32) {
            // handled stays false -> falls through to the unsupported path.
        } else
            switch (insn.mnemonic) {
            case ZYDIS_MNEMONIC_MOV: // basic
                if (insn.operand_width == 64) {
                    handled = EmitMov(insn, ops, next_rip, c);
                } else if (insn.operand_width == 32) {
                    handled = EmitMov32(insn, ops, next_rip, c);
                } else if (insn.operand_width == 16) {
                    handled = EmitMov16(insn, ops, next_rip, c);
                } else if (insn.operand_width == 8) {
                    handled = EmitMov8(insn, ops, next_rip, c);
                }
                break;
            case ZYDIS_MNEMONIC_LEA:    handled = EmitLea(insn, ops, next_rip, c); break; // basic
            case ZYDIS_MNEMONIC_MOVBE:  handled = EmitMovbe(insn, ops, next_rip, c); break; // basic
            case ZYDIS_MNEMONIC_BSWAP:  handled = EmitBswap(insn, ops, next_rip, c); break; // basic
            case ZYDIS_MNEMONIC_MOVSXD: handled = EmitMovsxd(insn, ops, next_rip, c); break; // basic
            case ZYDIS_MNEMONIC_MOVSX:  handled = EmitMovsx(insn, ops, next_rip, c); break; // basic
            case ZYDIS_MNEMONIC_ANDN:   handled = EmitAndn(insn, ops, next_rip, c); break; // BMI
            case ZYDIS_MNEMONIC_BEXTR:  handled = EmitBextr(insn, ops, next_rip, c); break; // BMI
            case ZYDIS_MNEMONIC_LZCNT:  handled = EmitLzcnt(insn, ops, next_rip, c); break; // BMI
            case ZYDIS_MNEMONIC_TZCNT:  handled = EmitTzcnt(insn, ops, next_rip, c); break; // BMI
            case ZYDIS_MNEMONIC_POPCNT: handled = EmitPopcnt(insn, ops, next_rip, c); break; // BMI
            case ZYDIS_MNEMONIC_ADD: // basic
                // 64-bit goes through the wide eager-flag path.
                // 32-bit takes the narrow round-trip path
                // (now including the mem-dst form). 8- and 16-bit
                // have always been narrow.
                if (insn.operand_width == 8) {
                    handled = EmitNarrowArith8(insn, ops, next_rip, c, NarrowArithKind::Add);
                } else if (insn.operand_width == 16) {
                    handled = EmitNarrowArith16(insn, ops, next_rip, c, NarrowArithKind::Add);
                } else if (insn.operand_width == 32) {
                    handled = EmitNarrowArith32(insn, ops, next_rip, c, NarrowArithKind::Add);
                } else {
                    handled = EmitAdd(insn, ops, next_rip, c);
                }
                break;
            case ZYDIS_MNEMONIC_SUB: // basic
                if (insn.operand_width == 8) {
                    handled = EmitNarrowArith8(insn, ops, next_rip, c, NarrowArithKind::Sub);
                } else if (insn.operand_width == 16) {
                    handled = EmitNarrowArith16(insn, ops, next_rip, c, NarrowArithKind::Sub);
                } else if (insn.operand_width == 32) {
                    handled = EmitNarrowArith32(insn, ops, next_rip, c, NarrowArithKind::Sub);
                } else {
                    handled = EmitSub(insn, ops, next_rip, c);
                }
                break;
            case ZYDIS_MNEMONIC_CMP: // basic
                if (insn.operand_width == 8) {
                    handled = EmitNarrowArith8(insn, ops, next_rip, c, NarrowArithKind::Cmp);
                } else if (insn.operand_width == 16) {
                    handled = EmitNarrowArith16(insn, ops, next_rip, c, NarrowArithKind::Cmp);
                } else if (insn.operand_width == 32) {
                    handled = EmitNarrowArith32(insn, ops, next_rip, c, NarrowArithKind::Cmp);
                } else {
                    handled = EmitCmp(insn, ops, next_rip, c);
                }
                break;

            // Add/sub with carry input — multi-precision arithmetic.
            case ZYDIS_MNEMONIC_ADC: // basic
                if (insn.operand_width == 8) {
                    handled = EmitNarrowArith8(insn, ops, next_rip, c, NarrowArithKind::Adc);
                } else if (insn.operand_width == 16) {
                    handled = EmitNarrowArith16(insn, ops, next_rip, c, NarrowArithKind::Adc);
                } else if (insn.operand_width == 32) {
                    handled = EmitNarrowArith32(insn, ops, next_rip, c, NarrowArithKind::Adc);
                } else {
                    handled = EmitAdcSbb64(insn, ops, c, AdcSbbKind::Adc);
                }
                break;
            case ZYDIS_MNEMONIC_SBB: // basic
                if (insn.operand_width == 8) {
                    handled = EmitNarrowArith8(insn, ops, next_rip, c, NarrowArithKind::Sbb);
                } else if (insn.operand_width == 16) {
                    handled = EmitNarrowArith16(insn, ops, next_rip, c, NarrowArithKind::Sbb);
                } else if (insn.operand_width == 32) {
                    handled = EmitNarrowArith32(insn, ops, next_rip, c, NarrowArithKind::Sbb);
                } else {
                    handled = EmitAdcSbb64(insn, ops, c, AdcSbbKind::Sbb);
                }
                break;
            case ZYDIS_MNEMONIC_VPUNPCKLQDQ: // AVX
            case ZYDIS_MNEMONIC_VPUNPCKHQDQ: // AVX
            case ZYDIS_MNEMONIC_VPUNPCKLDQ: // AVX
            case ZYDIS_MNEMONIC_VPUNPCKHDQ: // AVX
            case ZYDIS_MNEMONIC_VPUNPCKLWD: // AVX
            case ZYDIS_MNEMONIC_VPUNPCKHWD: // AVX
            case ZYDIS_MNEMONIC_VPUNPCKLBW: // AVX
            case ZYDIS_MNEMONIC_VPUNPCKHBW: // AVX
                handled = EmitVpunpck(insn, ops, next_rip, c);
                break;
            case ZYDIS_MNEMONIC_VPHADDD: // AVX
                handled = EmitVphaddd(insn, ops, next_rip, c);
                break;
            case ZYDIS_MNEMONIC_VPSHUFD: // AVX
                handled = EmitVpshufd(insn, ops, next_rip, c);
                break;
            case ZYDIS_MNEMONIC_VPSHUFLW: // AVX
            case ZYDIS_MNEMONIC_VPSHUFHW: // AVX
                handled = EmitVpshufd(insn, ops, next_rip, c);
                break;
            case ZYDIS_MNEMONIC_VPERMILPS: // AVX
                handled = EmitVpermilps(insn, ops, next_rip, c);
                break;
            case ZYDIS_MNEMONIC_VPERMILPD: // AVX
                handled = EmitVpermilps(insn, ops, next_rip, c);
                break;

            // Function epilogue shorthand: mov rsp, rbp; pop rbp.
            case ZYDIS_MNEMONIC_LEAVE: handled = EmitLeave(c); break; // basic
            case ZYDIS_MNEMONIC_TEST: // basic
                if (insn.operand_width == 8) {
                    handled = EmitNarrowArith8(insn, ops, next_rip, c, NarrowArithKind::Test);
                } else if (insn.operand_width == 16) {
                    handled = EmitNarrowArith16(insn, ops, next_rip, c, NarrowArithKind::Test);
                } else if (insn.operand_width == 32) {
                    handled = EmitNarrowArith32(insn, ops, next_rip, c, NarrowArithKind::Test);
                } else {
                    handled = EmitTest(insn, ops, next_rip, c);
                }
                break;
            case ZYDIS_MNEMONIC_XOR: // basic
                if (insn.operand_width == 8) {
                    handled = EmitNarrowArith8(insn, ops, next_rip, c, NarrowArithKind::Xor);
                } else if (insn.operand_width == 16) {
                    handled = EmitNarrowArith16(insn, ops, next_rip, c, NarrowArithKind::Xor);
                } else if (insn.operand_width == 32) {
                    handled = EmitNarrowArith32(insn, ops, next_rip, c, NarrowArithKind::Xor);
                } else {
                    handled = EmitXor(insn, ops, next_rip, c);
                }
                break;
            case ZYDIS_MNEMONIC_AND: // basic
                if (insn.operand_width == 8) {
                    handled = EmitNarrowArith8(insn, ops, next_rip, c, NarrowArithKind::And);
                } else if (insn.operand_width == 16) {
                    handled = EmitNarrowArith16(insn, ops, next_rip, c, NarrowArithKind::And);
                } else if (insn.operand_width == 32) {
                    handled = EmitNarrowArith32(insn, ops, next_rip, c, NarrowArithKind::And);
                } else {
                    handled = EmitAnd(insn, ops, next_rip, c);
                }
                break;
            case ZYDIS_MNEMONIC_OR: // basic
                if (insn.operand_width == 8) {
                    handled = EmitNarrowArith8(insn, ops, next_rip, c, NarrowArithKind::Or);
                } else if (insn.operand_width == 16) {
                    handled = EmitNarrowArith16(insn, ops, next_rip, c, NarrowArithKind::Or);
                } else if (insn.operand_width == 32) {
                    handled = EmitNarrowArith32(insn, ops, next_rip, c, NarrowArithKind::Or);
                } else {
                    handled = EmitOr(insn, ops, next_rip, c);
                }
                break;
            case ZYDIS_MNEMONIC_NOT:  handled = EmitNot(insn, ops, next_rip, c); break; // basic
            case ZYDIS_MNEMONIC_NEG:  handled = EmitNeg(insn, ops, next_rip, c); break; // basic
            case ZYDIS_MNEMONIC_INC:  handled = EmitInc(insn, ops, next_rip, c); break; // basic
            case ZYDIS_MNEMONIC_DEC:  handled = EmitDec(insn, ops, next_rip, c); break; // basic
            case ZYDIS_MNEMONIC_BT:   handled = EmitBt(insn, ops, c); break; // basic
            case ZYDIS_MNEMONIC_DIV:  handled = EmitDiv(insn, ops, next_rip, c); break; // basic
            case ZYDIS_MNEMONIC_IDIV: handled = EmitIdiv(insn, ops, next_rip, c); break; // basic
            case ZYDIS_MNEMONIC_CMPXCHG: handled = EmitCmpxchg(insn, ops, next_rip, c); break; // basic
            case ZYDIS_MNEMONIC_XADD:    handled = EmitXadd(insn, ops, next_rip, c); break; // basic
            case ZYDIS_MNEMONIC_XCHG:    handled = EmitXchg(insn, ops, next_rip, c); break; // basic
            // SETcc family — all 16 conditions route through EmitSetcc.
            case ZYDIS_MNEMONIC_SETZ:  case ZYDIS_MNEMONIC_SETNZ: // basic
            case ZYDIS_MNEMONIC_SETS:  case ZYDIS_MNEMONIC_SETNS: // basic
            case ZYDIS_MNEMONIC_SETO:  case ZYDIS_MNEMONIC_SETNO: // basic
            case ZYDIS_MNEMONIC_SETP:  case ZYDIS_MNEMONIC_SETNP: // basic
            case ZYDIS_MNEMONIC_SETB:  case ZYDIS_MNEMONIC_SETNB: // basic
            case ZYDIS_MNEMONIC_SETBE: case ZYDIS_MNEMONIC_SETNBE: // basic
            case ZYDIS_MNEMONIC_SETL:  case ZYDIS_MNEMONIC_SETNL: // basic
            case ZYDIS_MNEMONIC_SETLE: case ZYDIS_MNEMONIC_SETNLE: // basic
                handled = EmitSetcc(insn, ops, next_rip, c); break;
            case ZYDIS_MNEMONIC_MOVZX: handled = EmitMovzx(insn, ops, next_rip, c); break; // basic

            // Shifts. All three use the same emit with a kind tag.
            case ZYDIS_MNEMONIC_SHL: // basic
                handled = (insn.operand_width == 32)
                    ? EmitShift32(insn, ops, next_rip, c, ShiftKind::Shl)
                    : (insn.operand_width == 64)
                        ? EmitShift64(insn, ops, next_rip, c, ShiftKind::Shl)
                        : EmitShiftNarrow(insn, ops, next_rip, c, ShiftKind::Shl);
                break;
            case ZYDIS_MNEMONIC_SHR: // basic
                handled = (insn.operand_width == 32)
                    ? EmitShift32(insn, ops, next_rip, c, ShiftKind::Shr)
                    : (insn.operand_width == 64)
                        ? EmitShift64(insn, ops, next_rip, c, ShiftKind::Shr)
                        : EmitShiftNarrow(insn, ops, next_rip, c, ShiftKind::Shr);
                break;
            case ZYDIS_MNEMONIC_SAR: // basic
                handled = (insn.operand_width == 32)
                    ? EmitShift32(insn, ops, next_rip, c, ShiftKind::Sar)
                    : (insn.operand_width == 64)
                        ? EmitShift64(insn, ops, next_rip, c, ShiftKind::Sar)
                        : EmitShiftNarrow(insn, ops, next_rip, c, ShiftKind::Sar);
                break;

            // Rotates. Same shape as shifts.
            case ZYDIS_MNEMONIC_ROL: handled = EmitRotate64(insn, ops, next_rip, c, RotateKind::Rol); break; // basic
            case ZYDIS_MNEMONIC_ROR: handled = EmitRotate64(insn, ops, next_rip, c, RotateKind::Ror); break; // basic

            // Multiplication. EmitImul dispatches by operand_count_visible.
            case ZYDIS_MNEMONIC_IMUL: handled = EmitImul(insn, ops, next_rip, c); break; // basic
            case ZYDIS_MNEMONIC_MUL:  handled = EmitMul(insn, ops, next_rip, c); break; // basic
            case ZYDIS_MNEMONIC_BLSI: handled = EmitBlsi(insn, ops, next_rip, c); break; // BMI
            case ZYDIS_MNEMONIC_BLSR: handled = EmitBlsr(insn, ops, next_rip, c); break; // BMI
            case ZYDIS_MNEMONIC_STOSB: case ZYDIS_MNEMONIC_STOSW: // string
            case ZYDIS_MNEMONIC_STOSD: case ZYDIS_MNEMONIC_STOSQ: // string
            case ZYDIS_MNEMONIC_MOVSB: case ZYDIS_MNEMONIC_MOVSW: // string
            case ZYDIS_MNEMONIC_MOVSD: case ZYDIS_MNEMONIC_MOVSQ: // string
            case ZYDIS_MNEMONIC_LODSB: case ZYDIS_MNEMONIC_LODSW: // string
            case ZYDIS_MNEMONIC_LODSD: case ZYDIS_MNEMONIC_LODSQ: // string
            case ZYDIS_MNEMONIC_SCASB: case ZYDIS_MNEMONIC_SCASW: // string
            case ZYDIS_MNEMONIC_SCASD: case ZYDIS_MNEMONIC_SCASQ: // string
            case ZYDIS_MNEMONIC_CMPSB: case ZYDIS_MNEMONIC_CMPSW: // string
            case ZYDIS_MNEMONIC_CMPSD: case ZYDIS_MNEMONIC_CMPSQ: // string
                handled = EmitStringOp(insn, ops, next_rip, c);
                // EmitStringOp emits its own exit (jmp r14 / r15): it IS the
                // block terminator. Without this flag the loop kept decoding
                // the bytes AFTER the string op -- dead host code at best,
                // and at worst the decoder walks past the end of the function
                // into unmapped guest memory and faults inside the compiler.
                if (handled) emitted_terminator = true;
                break;
            case ZYDIS_MNEMONIC_CLD: // basic
            case ZYDIS_MNEMONIC_STD:  handled = EmitCldStd(insn, c); break; // basic
            case ZYDIS_MNEMONIC_PREFETCHNTA: // system
            case ZYDIS_MNEMONIC_PREFETCHT0: // system
            case ZYDIS_MNEMONIC_PREFETCHT1: // system
            case ZYDIS_MNEMONIC_PREFETCHT2: // system
            case ZYDIS_MNEMONIC_PREFETCHW: handled = EmitPrefetch(insn, ops, c); break; // system
            case ZYDIS_MNEMONIC_XGETBV: handled = EmitXgetbv(insn, c); break; // system
            case ZYDIS_MNEMONIC_VPMOVZXBW: // AVX
            case ZYDIS_MNEMONIC_VPMOVZXBD: // AVX
            case ZYDIS_MNEMONIC_VPMOVZXBQ: // AVX
            case ZYDIS_MNEMONIC_VPMOVZXWD: // AVX
            case ZYDIS_MNEMONIC_VPMOVZXWQ: // AVX
            case ZYDIS_MNEMONIC_VPMOVZXDQ: // AVX
            case ZYDIS_MNEMONIC_VPMOVSXBW: // AVX
            case ZYDIS_MNEMONIC_VPMOVSXBD: // AVX
            case ZYDIS_MNEMONIC_VPMOVSXBQ: // AVX
            case ZYDIS_MNEMONIC_VPMOVSXWD: // AVX
            case ZYDIS_MNEMONIC_VPMOVSXWQ: // AVX
            case ZYDIS_MNEMONIC_VPMOVSXDQ: // AVX
                handled = EmitPmovExtend(insn, ops, next_rip, c); break;
            case ZYDIS_MNEMONIC_VMOVMSKPS: // AVX
            case ZYDIS_MNEMONIC_VMOVMSKPD: // AVX
            case ZYDIS_MNEMONIC_VPMOVMSKB: // AVX
                handled = EmitMovmsk(insn, ops, next_rip, c); break;
            case ZYDIS_MNEMONIC_VROUNDPS: // AVX
            case ZYDIS_MNEMONIC_VROUNDPD: // AVX
                handled = EmitRoundPacked(insn, ops, next_rip, c); break;

            // Sign-extension family. No operands; operate on RAX/RDX.
            case ZYDIS_MNEMONIC_CBW:  handled = EmitCbw(c);  break; // basic
            case ZYDIS_MNEMONIC_CWDE: handled = EmitCwde(c); break; // basic
            case ZYDIS_MNEMONIC_CDQE: handled = EmitCdqe(c); break; // basic
            case ZYDIS_MNEMONIC_CWD:  handled = EmitCwd(c);  break; // basic
            case ZYDIS_MNEMONIC_CDQ:  handled = EmitCdq(c);  break; // basic
            case ZYDIS_MNEMONIC_CQO:  handled = EmitCqo(c);  break; // basic

            // Direct carry-flag manipulation.
            case ZYDIS_MNEMONIC_STC: handled = EmitStc(c); break; // basic
            case ZYDIS_MNEMONIC_CLC: handled = EmitClc(c); break; // basic
            case ZYDIS_MNEMONIC_CMC: handled = EmitCmc(c); break; // basic

            // NOP — no semantic effect, just consume the bytes.
            // Common forms: 90 (1-byte), 66 90 (2-byte),
            // 0F 1F /0 (multi-byte padding). All decode as NOP.
            case ZYDIS_MNEMONIC_NOP: handled = true; break; // basic
            // Memory-ordering fences. Guest threads run on parallel host
            // threads over shared guest memory (same premise as the locked
            // RMW path), so MFENCE must be preserved: even on a TSO host,
            // dropping it loses guest StoreLoad ordering (Dekker-style
            // lock-free code). SFENCE/LFENCE are no-ops here because every
            // store we emit is an ordinary x86 store (the non-temporal
            // VMOVNT* forms are lowered to plain moves) and TSO already
            // gives StoreStore / LoadLoad ordering.
            case ZYDIS_MNEMONIC_SFENCE: // system
            case ZYDIS_MNEMONIC_LFENCE: // system
                handled = true; break;
            case ZYDIS_MNEMONIC_MFENCE: // system
                c.mfence();
                handled = true; break;

            // All CMOVcc variants go through EmitCmov, which maps
            // the mnemonic to the matching Jcc condition encoding.
            case ZYDIS_MNEMONIC_CMOVZ: // basic
            case ZYDIS_MNEMONIC_CMOVNZ: // basic
            case ZYDIS_MNEMONIC_CMOVS: // basic
            case ZYDIS_MNEMONIC_CMOVNS: // basic
            case ZYDIS_MNEMONIC_CMOVO: // basic
            case ZYDIS_MNEMONIC_CMOVNO: // basic
            case ZYDIS_MNEMONIC_CMOVP: // basic
            case ZYDIS_MNEMONIC_CMOVNP: // basic
            case ZYDIS_MNEMONIC_CMOVB: // basic
            case ZYDIS_MNEMONIC_CMOVNB: // basic
            case ZYDIS_MNEMONIC_CMOVBE: // basic
            case ZYDIS_MNEMONIC_CMOVNBE: // basic
            case ZYDIS_MNEMONIC_CMOVL: // basic
            case ZYDIS_MNEMONIC_CMOVNL: // basic
            case ZYDIS_MNEMONIC_CMOVLE: // basic
            case ZYDIS_MNEMONIC_CMOVNLE: // basic
                handled = EmitCmov(insn, ops, next_rip, c);
                break;

            case ZYDIS_MNEMONIC_PUSH: handled = EmitPush(insn, ops, next_rip, c); break; // basic
            case ZYDIS_MNEMONIC_POP:  handled = EmitPop(insn, ops, c); break; // basic
            case ZYDIS_MNEMONIC_RET: // control
                handled = EmitRet(insn, ops, c);
                if (handled) emitted_terminator = true;
                break;
            case ZYDIS_MNEMONIC_JMP: // control
                handled = EmitJmp(insn, ops, next_rip, c);
                if (handled) emitted_terminator = true;
                break;
            case ZYDIS_MNEMONIC_CALL: // control
                handled = EmitCall(insn, ops, next_rip, c);
                if (handled) emitted_terminator = true;
                break;
            // All conditional jumps go through EmitJcc.
            case ZYDIS_MNEMONIC_JZ: // control
            case ZYDIS_MNEMONIC_JNZ: // control
            case ZYDIS_MNEMONIC_JS: // control
            case ZYDIS_MNEMONIC_JNS: // control
            case ZYDIS_MNEMONIC_JO: // control
            case ZYDIS_MNEMONIC_JNO: // control
            case ZYDIS_MNEMONIC_JP: // control
            case ZYDIS_MNEMONIC_JNP: // control
            case ZYDIS_MNEMONIC_JB: // control
            case ZYDIS_MNEMONIC_JNB: // control
            case ZYDIS_MNEMONIC_JBE: // control
            case ZYDIS_MNEMONIC_JNBE: // control
            case ZYDIS_MNEMONIC_JL: // control
            case ZYDIS_MNEMONIC_JNL: // control
            case ZYDIS_MNEMONIC_JLE: // control
            case ZYDIS_MNEMONIC_JNLE: // control
                handled = EmitJcc(insn, ops, next_rip, c);
                if (handled) emitted_terminator = true;
                break;

            // AVX VEX-encoded 128/256-bit vector instructions. These
            // operate on GuestState::ymm[] via 64-bit GPR transfers
            // (see EmitVmovups / EmitVecBitXor for the design notes).
            case ZYDIS_MNEMONIC_VMOVUPS: // AVX
            case ZYDIS_MNEMONIC_VMOVUPD: // AVX
            case ZYDIS_MNEMONIC_VMOVDQU: // AVX
            case ZYDIS_MNEMONIC_VMOVDQA: // AVX
            case ZYDIS_MNEMONIC_VMOVAPS: // AVX
            case ZYDIS_MNEMONIC_VMOVAPD: // AVX
            case ZYDIS_MNEMONIC_VMOVNTDQ: // AVX
            case ZYDIS_MNEMONIC_VMOVNTDQA: // AVX
                handled = EmitVmovups(insn, ops, next_rip, c);
                break;
            case ZYDIS_MNEMONIC_VXORPS: case ZYDIS_MNEMONIC_VXORPD: // AVX
            case ZYDIS_MNEMONIC_VPXOR: // AVX
            case ZYDIS_MNEMONIC_VANDPS: case ZYDIS_MNEMONIC_VANDPD: // AVX
            case ZYDIS_MNEMONIC_VPAND: // AVX
            case ZYDIS_MNEMONIC_VORPS:  case ZYDIS_MNEMONIC_VORPD: // AVX
            case ZYDIS_MNEMONIC_VPOR: // AVX
            case ZYDIS_MNEMONIC_VANDNPS: case ZYDIS_MNEMONIC_VANDNPD: // AVX
            case ZYDIS_MNEMONIC_VPANDN: // AVX
                handled = EmitVecBitLogic(insn, ops, next_rip, c);
                break;
            case ZYDIS_MNEMONIC_VMOVD: // AVX
                handled = EmitVmovd(insn, ops, next_rip, c);
                break;
            case ZYDIS_MNEMONIC_VMOVQ: // AVX
            case ZYDIS_MNEMONIC_MOVQ: // SSE
                handled = EmitVmovq(insn, ops, next_rip, c);
                break;
            case ZYDIS_MNEMONIC_VMOVSS: // AVX
            case ZYDIS_MNEMONIC_VMOVSD: // AVX
                handled = EmitVmovss(insn, ops, next_rip, c);
                break;
            case ZYDIS_MNEMONIC_VCVTSI2SS: // AVX
            case ZYDIS_MNEMONIC_VCVTSI2SD: // AVX
                handled = EmitVcvtsi2(insn, ops, next_rip, c);
                break;
            case ZYDIS_MNEMONIC_VCVTTSS2SI: // AVX
            case ZYDIS_MNEMONIC_VCVTTSD2SI: // AVX
            case ZYDIS_MNEMONIC_CVTTSS2SI: // SSE
            case ZYDIS_MNEMONIC_CVTTSD2SI: // SSE
            case ZYDIS_MNEMONIC_CVTSS2SI: // SSE
            case ZYDIS_MNEMONIC_CVTSD2SI: // SSE
                handled = EmitVcvtt2si(insn, ops, next_rip, c);
                break;
            case ZYDIS_MNEMONIC_VUCOMISS: case ZYDIS_MNEMONIC_VUCOMISD: // AVX
            case ZYDIS_MNEMONIC_VCOMISS:  case ZYDIS_MNEMONIC_VCOMISD: // AVX
                handled = EmitFpCompare(insn, ops, next_rip, c);
                break;
            case ZYDIS_MNEMONIC_VCVTSS2SD: // AVX
            case ZYDIS_MNEMONIC_VCVTSD2SS: // AVX
                handled = EmitVcvtScalar(insn, ops, next_rip, c);
                break;
            case ZYDIS_MNEMONIC_VMULSS: case ZYDIS_MNEMONIC_VMULSD: // AVX
            case ZYDIS_MNEMONIC_VADDSS: case ZYDIS_MNEMONIC_VADDSD: // AVX
            case ZYDIS_MNEMONIC_VSUBSS: case ZYDIS_MNEMONIC_VSUBSD: // AVX
            case ZYDIS_MNEMONIC_VDIVSS: case ZYDIS_MNEMONIC_VDIVSD: // AVX
            case ZYDIS_MNEMONIC_VMINSS: case ZYDIS_MNEMONIC_VMINSD: // AVX
            case ZYDIS_MNEMONIC_VMAXSS: case ZYDIS_MNEMONIC_VMAXSD: // AVX
                handled = EmitScalarFpArith(insn, ops, next_rip, c);
                break;
            case ZYDIS_MNEMONIC_VMULPS: case ZYDIS_MNEMONIC_VMULPD: // AVX
            case ZYDIS_MNEMONIC_VADDPS: case ZYDIS_MNEMONIC_VADDPD: // AVX
            case ZYDIS_MNEMONIC_VSUBPS: case ZYDIS_MNEMONIC_VSUBPD: // AVX
            case ZYDIS_MNEMONIC_VDIVPS: case ZYDIS_MNEMONIC_VDIVPD: // AVX
            case ZYDIS_MNEMONIC_VMINPS: case ZYDIS_MNEMONIC_VMINPD: // AVX
            case ZYDIS_MNEMONIC_VMAXPS: case ZYDIS_MNEMONIC_VMAXPD: // AVX
            case ZYDIS_MNEMONIC_VHADDPS: case ZYDIS_MNEMONIC_VHADDPD: // AVX
            case ZYDIS_MNEMONIC_VHSUBPS: case ZYDIS_MNEMONIC_VHSUBPD: // AVX
            case ZYDIS_MNEMONIC_VADDSUBPS: case ZYDIS_MNEMONIC_VADDSUBPD: // AVX
                handled = EmitPackedFpArith(insn, ops, next_rip, c);
                break;
            case ZYDIS_MNEMONIC_VSQRTPS: // AVX
            case ZYDIS_MNEMONIC_VSQRTPD: // AVX
            case ZYDIS_MNEMONIC_VRSQRTPS: // AVX
            case ZYDIS_MNEMONIC_VRCPPS: // AVX
                handled = EmitPackedFpUnary(insn, ops, next_rip, c);
                break;
            case ZYDIS_MNEMONIC_VPADDD: // AVX
            case ZYDIS_MNEMONIC_VPSUBD: // AVX
            case ZYDIS_MNEMONIC_VPADDQ: // AVX
            case ZYDIS_MNEMONIC_VPSUBQ: // AVX
            case ZYDIS_MNEMONIC_VPMULLD: // AVX
            case ZYDIS_MNEMONIC_VPMULUDQ: // AVX
            case ZYDIS_MNEMONIC_VPMINUD: // AVX
            case ZYDIS_MNEMONIC_VPMINSD: // AVX
            case ZYDIS_MNEMONIC_VPMAXUD: // AVX
            case ZYDIS_MNEMONIC_VPMAXSD: // AVX
            case ZYDIS_MNEMONIC_VPCMPEQD: // AVX
            case ZYDIS_MNEMONIC_VPCMPGTD: // AVX
            case ZYDIS_MNEMONIC_VPCMPEQQ: // AVX
            case ZYDIS_MNEMONIC_VPCMPGTQ: // AVX
            case ZYDIS_MNEMONIC_VPADDB: // AVX
            case ZYDIS_MNEMONIC_VPADDW: // AVX
            case ZYDIS_MNEMONIC_VPSUBB: // AVX
            case ZYDIS_MNEMONIC_VPSUBW: // AVX
            case ZYDIS_MNEMONIC_VPMULLW: // AVX
            case ZYDIS_MNEMONIC_VPMINUB: // AVX
            case ZYDIS_MNEMONIC_VPMINSB: // AVX
            case ZYDIS_MNEMONIC_VPMINUW: // AVX
            case ZYDIS_MNEMONIC_VPMINSW: // AVX
            case ZYDIS_MNEMONIC_VPMAXUB: // AVX
            case ZYDIS_MNEMONIC_VPMAXSB: // AVX
            case ZYDIS_MNEMONIC_VPMAXUW: // AVX
            case ZYDIS_MNEMONIC_VPMAXSW: // AVX
            case ZYDIS_MNEMONIC_VPCMPEQW: // AVX
            case ZYDIS_MNEMONIC_VPCMPGTB: // AVX
            case ZYDIS_MNEMONIC_VPCMPGTW: // AVX
                handled = EmitPackedIntArith(insn, ops, next_rip, c);
                break;
            case ZYDIS_MNEMONIC_VUNPCKLPS: case ZYDIS_MNEMONIC_VUNPCKHPS: // AVX
            case ZYDIS_MNEMONIC_VUNPCKLPD: case ZYDIS_MNEMONIC_VUNPCKHPD: // AVX
            case ZYDIS_MNEMONIC_VPACKUSDW: // AVX
            case ZYDIS_MNEMONIC_VMOVLHPS:  case ZYDIS_MNEMONIC_VMOVHLPS: // AVX
            case ZYDIS_MNEMONIC_VSQRTSD: // AVX
            case ZYDIS_MNEMONIC_VSQRTSS: // AVX
            case ZYDIS_MNEMONIC_VRSQRTSS: // AVX
            case ZYDIS_MNEMONIC_VRCPSS: // AVX
                handled = EmitVecHostStaged(insn, ops, next_rip, c);
                break;
            case ZYDIS_MNEMONIC_VSHUFPS: case ZYDIS_MNEMONIC_VBLENDPS: // AVX
            case ZYDIS_MNEMONIC_VPBLENDW: case ZYDIS_MNEMONIC_VCMPPS: // AVX
            case ZYDIS_MNEMONIC_VSHUFPD: case ZYDIS_MNEMONIC_VBLENDPD: // AVX
            case ZYDIS_MNEMONIC_VPBLENDD: // AVX
                handled = EmitVecImm8(insn, ops, next_rip, c);
                break;
            case ZYDIS_MNEMONIC_VPEXTRD: case ZYDIS_MNEMONIC_VPEXTRQ: // AVX
            case ZYDIS_MNEMONIC_VEXTRACTPS: case ZYDIS_MNEMONIC_VPINSRD: // AVX
            case ZYDIS_MNEMONIC_VPINSRW: case ZYDIS_MNEMONIC_VPINSRB: // AVX
                handled = EmitLaneGpr(insn, ops, next_rip, c);
                break;
            case ZYDIS_MNEMONIC_VCMPSS: // AVX
                handled = EmitVcmpss(insn, ops, next_rip, c);
                break;
            case ZYDIS_MNEMONIC_VPSRAD: case ZYDIS_MNEMONIC_VPSRLD: // AVX
            case ZYDIS_MNEMONIC_VPSLLDQ: case ZYDIS_MNEMONIC_VPSRLDQ: // AVX
            case ZYDIS_MNEMONIC_VPSLLW: case ZYDIS_MNEMONIC_VPSLLD: // AVX
            case ZYDIS_MNEMONIC_VPSLLQ: case ZYDIS_MNEMONIC_VPSRLW: // AVX
            case ZYDIS_MNEMONIC_VPSRLQ: case ZYDIS_MNEMONIC_VPSRAW: // AVX
                handled = EmitVecShift(insn, ops, next_rip, c);
                break;
            case ZYDIS_MNEMONIC_VCVTDQ2PS: case ZYDIS_MNEMONIC_VCVTTPS2DQ: // AVX
                handled = EmitVecConvert(insn, ops, next_rip, c);
                break;
            case ZYDIS_MNEMONIC_VBROADCASTSS: // AVX
                handled = EmitVbroadcastss(insn, ops, next_rip, c);
                break;
            case ZYDIS_MNEMONIC_VINSERTF128: case ZYDIS_MNEMONIC_VEXTRACTF128: // AVX
                handled = EmitVlane128(insn, ops, next_rip, c);
                break;
            case ZYDIS_MNEMONIC_VBLENDVPS: // AVX
            case ZYDIS_MNEMONIC_VBLENDVPD: // AVX
            case ZYDIS_MNEMONIC_VPBLENDVB: // AVX
                handled = EmitVblendvps(insn, ops, next_rip, c);
                break;
            case ZYDIS_MNEMONIC_VROUNDSS: // AVX
            case ZYDIS_MNEMONIC_VROUNDSD: // AVX
                handled = EmitVround(insn, ops, next_rip, c);
                break;
            case ZYDIS_MNEMONIC_VINSERTPS: // AVX
                handled = EmitVinsertps(insn, ops, next_rip, c);
                break;
            case ZYDIS_MNEMONIC_VMOVSLDUP: // AVX
            case ZYDIS_MNEMONIC_VMOVSHDUP: // AVX
            case ZYDIS_MNEMONIC_VMOVDDUP: // AVX
                handled = EmitMovDup(insn, ops, next_rip, c);
                break;

            // x87 FPU — Group 1: load/store.
            case ZYDIS_MNEMONIC_FLD: // x87
                handled = EmitFld(insn, ops, next_rip, c);
                break;
            case ZYDIS_MNEMONIC_FST: // x87
            case ZYDIS_MNEMONIC_FSTP: // x87
                handled = EmitFstOrFstp(insn, ops, next_rip, c);
                break;
            case ZYDIS_MNEMONIC_FADDP:  case ZYDIS_MNEMONIC_FMULP: // x87
            case ZYDIS_MNEMONIC_FSUBP:  case ZYDIS_MNEMONIC_FSUBRP: // x87
            case ZYDIS_MNEMONIC_FDIVP:  case ZYDIS_MNEMONIC_FDIVRP: // x87
                handled = EmitFpuArithPop(insn, ops, c);
                break;
            case ZYDIS_MNEMONIC_FADD:   case ZYDIS_MNEMONIC_FMUL: // x87
            case ZYDIS_MNEMONIC_FSUB:   case ZYDIS_MNEMONIC_FSUBR: // x87
            case ZYDIS_MNEMONIC_FDIV:   case ZYDIS_MNEMONIC_FDIVR: // x87
                handled = EmitFpuArith(insn, ops, next_rip, c);
                break;
            case ZYDIS_MNEMONIC_FCHS:   case ZYDIS_MNEMONIC_FABS: // x87
            case ZYDIS_MNEMONIC_FSQRT: // x87
                handled = EmitFpuUnary(insn, c);
                break;
            case ZYDIS_MNEMONIC_FLD1:   case ZYDIS_MNEMONIC_FLDZ: // x87
                handled = EmitFpuLoadConst(insn, c);
                break;
            case ZYDIS_MNEMONIC_FXCH: // x87
                handled = EmitFxch(insn, ops, c);
                break;
            case ZYDIS_MNEMONIC_FCOMI:  case ZYDIS_MNEMONIC_FUCOMI: // x87
            case ZYDIS_MNEMONIC_FCOMIP: case ZYDIS_MNEMONIC_FUCOMIP: // x87
            case ZYDIS_MNEMONIC_FCOM:   case ZYDIS_MNEMONIC_FCOMP: // x87
            case ZYDIS_MNEMONIC_FCOMPP: case ZYDIS_MNEMONIC_FUCOM: // x87
            case ZYDIS_MNEMONIC_FUCOMP: // x87
                handled = EmitFpuCompare(insn, ops, c);
                break;
            case ZYDIS_MNEMONIC_FNSTSW: // x87
                handled = EmitFnstsw(insn, ops, next_rip, c);
                break;
            case ZYDIS_MNEMONIC_FCMOVB: case ZYDIS_MNEMONIC_FCMOVE: // x87
            case ZYDIS_MNEMONIC_FCMOVBE: case ZYDIS_MNEMONIC_FCMOVU: // x87
            case ZYDIS_MNEMONIC_FCMOVNB: case ZYDIS_MNEMONIC_FCMOVNE: // x87
            case ZYDIS_MNEMONIC_FCMOVNBE: case ZYDIS_MNEMONIC_FCMOVNU: // x87
                handled = EmitFcmov(insn, ops, c);
                break;
            case ZYDIS_MNEMONIC_FLDCW: // x87
                handled = EmitFldcw(insn, ops, next_rip, c);
                break;
            case ZYDIS_MNEMONIC_FNSTCW: // x87
                handled = EmitFnstcw(insn, ops, next_rip, c);
                break;
            case ZYDIS_MNEMONIC_STMXCSR:  // SSE
            case ZYDIS_MNEMONIC_LDMXCSR:  // SSE
            case ZYDIS_MNEMONIC_VSTMXCSR: // AVX (VEX-encoded, same semantics)
            case ZYDIS_MNEMONIC_VLDMXCSR: // AVX (VEX-encoded, same semantics)
                handled = EmitMxcsr(insn, ops, next_rip, c);
                break;
            case ZYDIS_MNEMONIC_FILD: // x87
                handled = EmitFild(insn, ops, next_rip, c);
                break;
            case ZYDIS_MNEMONIC_FISTP: // x87
            case ZYDIS_MNEMONIC_FISTTP: // x87
                handled = EmitFistp(insn, ops, next_rip, c);
                break;
            case ZYDIS_MNEMONIC_VPSHUFB: // AVX
                handled = EmitVpshufb(insn, ops, next_rip, c);
                break;
            case ZYDIS_MNEMONIC_VPCMPEQB: // AVX
                handled = EmitVpcmpeqb(insn, ops, next_rip, c);
                break;
            case ZYDIS_MNEMONIC_VPTEST: // AVX
            case ZYDIS_MNEMONIC_PTEST: // SSE
                handled = EmitVptest(insn, ops, next_rip, c);
                break;
            case ZYDIS_MNEMONIC_VPCMPISTRI: // AVX
            case ZYDIS_MNEMONIC_PCMPISTRI: // SSE
                handled = EmitVpcmpistri(insn, ops, next_rip, c);
                break;
            case ZYDIS_MNEMONIC_VPCMPISTRM: // AVX
            case ZYDIS_MNEMONIC_PCMPISTRM: // SSE
                handled = EmitVpcmpistrm(insn, ops, next_rip, c);
                break;
            case ZYDIS_MNEMONIC_RDTSC: // system
                handled = EmitRdtsc(insn, ops, c, /*with_aux=*/false);
                break;
            case ZYDIS_MNEMONIC_RDTSCP: // system
                handled = EmitRdtsc(insn, ops, c, /*with_aux=*/true);
                break;
            case ZYDIS_MNEMONIC_CPUID: // system
                handled = EmitCpuid(insn, ops, c);
                break;
            case ZYDIS_MNEMONIC_LAHF: // flags
                handled = EmitLahf(insn, ops, c);
                break;
            case ZYDIS_MNEMONIC_SAHF: // flags
                handled = EmitSahf(insn, ops, c);
                break;
            default:
                handled = false;
                break;
            }

        if (!handled) {
            // Operand-type accessor for the diagnostic. Maps Zydis
            // operand-type enum to a short string. Helps tell apart
            // "narrow-width register form unimplemented" from
            // "narrow-width memory form unimplemented" — same
            // mnemonic, very different work to add.
            auto op_type_name = [](ZydisOperandType t) -> const char* {
                switch (t) {
                    case ZYDIS_OPERAND_TYPE_REGISTER:  return "reg";
                    case ZYDIS_OPERAND_TYPE_MEMORY:    return "mem";
                    case ZYDIS_OPERAND_TYPE_IMMEDIATE: return "imm";
                    case ZYDIS_OPERAND_TYPE_POINTER:   return "ptr";
                    default:                           return "?";
                }
            };
            std::fprintf(stderr,
                         "[lifter] unsupported insn at 0x%llx (mnemonic=%s, "
                         "width=%u, length=%u, ops=%s,%s)\n",
                         static_cast<unsigned long long>(rip),
                         ZydisMnemonicGetString(insn.mnemonic),
                         static_cast<unsigned>(insn.operand_width),
                         static_cast<unsigned>(insn.length),
                         op_type_name(ops[0].type),
                         op_type_name(ops[1].type));
            std::fflush(stderr);
            ++unsupported_hits_;
            // Update state.rip to the un-lifted instruction so a
            // post-mortem caller knows where it stopped, then exit
            // via r15 (fatal exit). Using r14 here would loop the
            // dispatcher on the same bad address.
            c.mov(rax, rip);
            c.mov(qword[r13 + Offsets::Rip], rax);
            c.mov(dword[r13 + offsetof(GuestState, exit_reason)],
                  static_cast<u32>(ExitReason::UnsupportedInstruction));
            c.jmp(r15);
            emitted_terminator = true;
            break;
        }

        rip += insn.length;
        if (emitted_terminator) break;

        // Stop before the next instruction if emitting it could overrun the
        // host buffer. Vector ops (SSE/AVX) can expand to several hundred host
        // bytes each, so the margin must comfortably cover one worst-case
        // instruction plus the fall-through terminator emitted below. Without
        // this guard a dense block overflowed the CodeGenerator buffer and
        // Xbyak threw ERR_CODE_IS_TOO_BIG mid-compile. (Mirrors arm64.)
        constexpr u64 HOST_SIZE_MARGIN = 1024;
        if (c.getSize() + HOST_SIZE_MARGIN >= BLOCK_HOST_SIZE_CAP) break;
    }

    if (!emitted_terminator) {
        // Block hit the size cap without finding a terminator.
        // Emit a fall-through exit. Set exit_reason to BlockEnd
        // (not "unsupported" — fallthrough is normal block exit
        // when we have a real dispatcher loop).
        c.mov(rax, rip);
        c.mov(qword[r13 + Offsets::Rip], rax);
        c.mov(dword[r13 + offsetof(GuestState, exit_reason)],
              static_cast<u32>(ExitReason::BlockEnd));
        c.jmp(r14);
    }

    const u64 emitted = c.getSize();
    // Hand the unused remainder of the cap-sized reservation back to the
    // bump allocator (succeeds whenever no other thread allocated since;
    // see CodeCache::ReturnTail). Multiplies effective cache capacity by
    // the cap-to-typical-block ratio, making stop-the-world recycles
    // correspondingly rarer.
    code_cache_.ReturnTail(code_buf, BLOCK_HOST_SIZE_CAP, emitted);

    bytes_emitted_ += emitted;
    ++blocks_compiled_;

    if (lifter_trace) {
        std::fprintf(stderr,
                     "[lifter] compiled block 0x%llx -> %p (%llu guest bytes -> %llu host bytes)\n",
                     static_cast<unsigned long long>(guest_rip),
                     static_cast<void*>(code_buf),
                     static_cast<unsigned long long>(rip - guest_rip),
                     static_cast<unsigned long long>(emitted));
        std::fflush(stderr);
    }

    return code_buf;
} catch (const std::exception& e) {
    // The only thrower in scope is the assembler (Xbyak::Error derives from
    // std::exception; what() carries the error string). fprintf, not LOG_*:
    // same JIT-dispatched-context rule as the rest of this function.
    std::fprintf(stderr,
                 "[lifter] compile threw at guest RIP 0x%llx: %s\n",
                 static_cast<unsigned long long>(guest_rip), e.what());
    std::fflush(stderr);
    return nullptr;
}

} // namespace Core::Runtime
