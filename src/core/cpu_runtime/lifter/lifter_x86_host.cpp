// SPDX-FileCopyrightText: Copyright 2026 shadPS4 Emulator Project
// SPDX-License-Identifier: GPL-2.0-or-later

#include "core/cpu_runtime/lifter/lifter.h"

#include <cstddef>
#include <cstdio>
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
/// Keeps compile time bounded for runaway code.
constexpr u64 BLOCK_HOST_SIZE_CAP = 4096;

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

static_assert(ZYDIS_REGISTER_RCX - ZYDIS_REGISTER_RAX == 1, "Zydis 64-bit GPR enum order changed");
static_assert(ZYDIS_REGISTER_R15 - ZYDIS_REGISTER_RAX == 15,
              "Zydis 64-bit GPRs no longer contiguous over 16 entries");
static_assert(ZYDIS_REGISTER_ECX - ZYDIS_REGISTER_EAX == 1, "Zydis 32-bit GPR enum order changed");
static_assert(ZYDIS_REGISTER_R15D - ZYDIS_REGISTER_EAX == 15,
              "Zydis 32-bit GPRs no longer contiguous");
static_assert(ZYDIS_REGISTER_CX - ZYDIS_REGISTER_AX == 1, "Zydis 16-bit GPR enum order changed");
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
static_assert((ZYDIS_REGISTER_RAX - ZYDIS_REGISTER_RAX) == 0, "RAX must map to slot 0");
static_assert((ZYDIS_REGISTER_RSP - ZYDIS_REGISTER_RAX) == 4,
              "RSP must map to slot 4 (canonical AMD64 ordering)");
static_assert((ZYDIS_REGISTER_RDI - ZYDIS_REGISTER_RAX) == 7,
              "RDI must map to slot 7 (SysV arg 1; HLE bridge depends on this)");
static_assert((ZYDIS_REGISTER_R15 - ZYDIS_REGISTER_RAX) == 15, "R15 must map to slot 15");

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

/// Byte offset within GuestState for an 8-bit register reference.
///
/// GuestState stores GPRs as little-endian qword slots, so the low
/// byte of each parent (AL/CL/DL/BL/SPL/BPL/SIL/DIL/R8B..R15B)
/// lives at byte 0 of its slot, and the legacy high bytes
/// (AH/CH/DH/BH) live at byte 1 of slot 0/1/2/3 (RAX/RCX/RDX/RBX).
///
/// Returns -1 if `r` isn't an 8-bit GPR. Distinct from
/// ZydisGprToIndex, which intentionally returns -1 for high-byte
/// registers because callers that round-trip through host
/// registers can't generally encode AH/BH/CH/DH (REX prefix makes
/// SPL/BPL/SIL/DIL claim the same byte-3 slot of the ModR/M
/// encoding). For callers like EmitNarrowArith8 that load and
/// store via explicit byte offsets, high-byte regs are fine.
int ZydisGpr8ToByteOffset(ZydisRegister r) {
    if (r >= ZYDIS_REGISTER_AL && r <= ZYDIS_REGISTER_BL) {
        return static_cast<int>(GprOffset(r - ZYDIS_REGISTER_AL));
    }
    if (r >= ZYDIS_REGISTER_AH && r <= ZYDIS_REGISTER_BH) {
        // Zydis enum order: AH, CH, DH, BH — which maps to parent
        // slots 0 (RAX), 1 (RCX), 2 (RDX), 3 (RBX) in that order.
        return static_cast<int>(GprOffset(r - ZYDIS_REGISTER_AH)) + 1;
    }
    if (r >= ZYDIS_REGISTER_SPL && r <= ZYDIS_REGISTER_DIL) {
        return static_cast<int>(GprOffset((r - ZYDIS_REGISTER_SPL) + 4));
    }
    if (r >= ZYDIS_REGISTER_R8B && r <= ZYDIS_REGISTER_R15B) {
        return static_cast<int>(GprOffset((r - ZYDIS_REGISTER_R8B) + 8));
    }
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
bool EmitEffectiveAddress(const ZydisDecodedOperandMem& mem, u64 next_rip,
                          Xbyak::CodeGenerator& c) {
    // Segment overrides other than the standard DS/SS aren't
    // supported. shadPS4 guest code rarely uses FS/GS, and when it
    // does it's for TLS via specific helper sequences we'd lift
    // specially. CS/ES are flat in long mode.
    if (mem.segment != ZYDIS_REGISTER_DS && mem.segment != ZYDIS_REGISTER_SS &&
        mem.segment != ZYDIS_REGISTER_CS && mem.segment != ZYDIS_REGISTER_ES) {
        return false;
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
        if (has_index)
            return false; // RIP-relative with index is not a thing
        c.mov(rdx, static_cast<u64>(static_cast<s64>(next_rip) + disp));
        return true;
    }

    // Plain [disp32] absolute (no base, no index).
    if (!has_base && !has_index) {
        c.mov(rdx, static_cast<u64>(disp));
        return true;
    }

    // General case: rdx = base + index*scale + disp.
    // Start rdx at the base value (or 0 if no base).
    if (has_base) {
        const int base_idx = ZydisGprToIndex(mem.base);
        if (base_idx < 0)
            return false;
        c.mov(rdx, qword[r13 + GprOffset(base_idx)]);
    } else {
        c.xor_(rdx, rdx);
    }

    if (has_index) {
        const int index_idx = ZydisGprToIndex(mem.index);
        if (index_idx < 0)
            return false;
        // Load index into rax, scale it, add to rdx.
        c.mov(rax, qword[r13 + GprOffset(index_idx)]);
        switch (mem.scale) {
        case 1:
            break; // no shift
        case 2:
            c.shl(rax, 1);
            break;
        case 4:
            c.shl(rax, 2);
            break;
        case 8:
            c.shl(rax, 3);
            break;
        default:
            return false; // invalid SIB scale
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
bool EmitMov(const ZydisDecodedInstruction& insn, const ZydisDecodedOperand* ops, u64 next_rip,
             Xbyak::CodeGenerator& c) {
    if (insn.operand_width != 64) {
        return false; // 32/16/8-bit MOVs not in initial slice
    }
    const auto& dst = ops[0];
    const auto& src = ops[1];

    // ----- Memory destination -----
    if (dst.type == ZYDIS_OPERAND_TYPE_MEMORY) {
        // Compute effective address into rdx.
        if (!EmitEffectiveAddress(dst.mem, next_rip, c))
            return false;

        if (src.type == ZYDIS_OPERAND_TYPE_REGISTER) {
            const int src_idx = ZydisGprToIndex(src.reg.value);
            if (src_idx < 0)
                return false;
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
    if (dst.type != ZYDIS_OPERAND_TYPE_REGISTER)
        return false;
    const int dst_idx = ZydisGprToIndex(dst.reg.value);
    if (dst_idx < 0)
        return false;

    if (src.type == ZYDIS_OPERAND_TYPE_REGISTER) {
        const int src_idx = ZydisGprToIndex(src.reg.value);
        if (src_idx < 0)
            return false;
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
        if (!EmitEffectiveAddress(src.mem, next_rip, c))
            return false;
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
bool EmitMov32(const ZydisDecodedInstruction& insn, const ZydisDecodedOperand* ops, u64 next_rip,
               Xbyak::CodeGenerator& c) {
    if (insn.operand_width != 32)
        return false;
    const auto& dst = ops[0];
    const auto& src = ops[1];

    // ----- Memory destination -----
    if (dst.type == ZYDIS_OPERAND_TYPE_MEMORY) {
        // Compute effective address into rdx.
        if (!EmitEffectiveAddress(dst.mem, next_rip, c))
            return false;

        if (src.type == ZYDIS_OPERAND_TYPE_REGISTER) {
            const int src_idx = ZydisGprToIndex(src.reg.value);
            if (src_idx < 0)
                return false;
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
    if (dst.type != ZYDIS_OPERAND_TYPE_REGISTER)
        return false;
    const int dst_idx = ZydisGprToIndex(dst.reg.value);
    if (dst_idx < 0)
        return false;

    if (src.type == ZYDIS_OPERAND_TYPE_REGISTER) {
        const int src_idx = ZydisGprToIndex(src.reg.value);
        if (src_idx < 0)
            return false;
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
        if (!EmitEffectiveAddress(src.mem, next_rip, c))
            return false;
        // 32-bit load zero-extends rax. Store full 64.
        c.mov(eax, dword[rdx]);
        c.mov(qword[r13 + GprOffset(dst_idx)], rax);
        return true;
    }

    return false;
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
bool EmitLea(const ZydisDecodedInstruction& insn, const ZydisDecodedOperand* ops, u64 next_rip,
             Xbyak::CodeGenerator& c) {
    if (insn.operand_width != 64 && insn.operand_width != 32)
        return false;
    if (ops[0].type != ZYDIS_OPERAND_TYPE_REGISTER)
        return false;
    if (ops[1].type != ZYDIS_OPERAND_TYPE_MEMORY)
        return false;

    const int dst_idx = ZydisGprToIndex(ops[0].reg.value);
    if (dst_idx < 0)
        return false;

    if (!EmitEffectiveAddress(ops[1].mem, next_rip, c))
        return false;
    // For 32-bit LEA, x86-64's "32-bit destination write zero-extends
    // bits 63:32" rule applies: only the low 32 bits of the effective
    // address go into the destination, with the upper 32 zeroed.
    // We can't rely on a `mov dword[slot], edx` here because memory
    // writes don't get the zero-extension rule — only register
    // writes do — so a dword store would leave bytes 4..7 of the
    // slot stale. Instead we self-mov edx into edx, which is the
    // canonical x86-64 idiom for "clear upper 32 of rdx", then
    // qword-store the now-correct value.
    if (insn.operand_width == 32) {
        c.mov(edx, edx);
    }
    c.mov(qword[r13 + GprOffset(dst_idx)], rdx);
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
bool EmitMovsxd(const ZydisDecodedInstruction& insn, const ZydisDecodedOperand* ops, u64 next_rip,
                Xbyak::CodeGenerator& c) {
    if (insn.operand_width != 64)
        return false;
    if (ops[0].type != ZYDIS_OPERAND_TYPE_REGISTER)
        return false;

    const int dst_idx = ZydisGprToIndex(ops[0].reg.value);
    if (dst_idx < 0)
        return false;

    if (ops[1].type == ZYDIS_OPERAND_TYPE_REGISTER) {
        const int src_idx = ZydisGprToIndex(ops[1].reg.value);
        if (src_idx < 0)
            return false;
        // Sign-extend low 32 of guest src register to 64 bits.
        c.movsxd(rax, dword[r13 + GprOffset(src_idx)]);
        c.mov(qword[r13 + GprOffset(dst_idx)], rax);
        return true;
    }

    if (ops[1].type == ZYDIS_OPERAND_TYPE_MEMORY) {
        if (!EmitEffectiveAddress(ops[1].mem, next_rip, c))
            return false;
        c.movsxd(rax, dword[rdx]);
        c.mov(qword[r13 + GprOffset(dst_idx)], rax);
        return true;
    }

    return false;
}

/// MOVSX r{16,32,64}, r/m{8,16} — sign-extend a narrower source into a
/// wider destination. The 32→64 sign-extend is encoded as MOVSXD and
/// handled by EmitMovsxd above; this covers the MOVSX opcode family
/// (8→{16,32,64} and 16→{32,64}). MOVSX doesn't affect flags.
///
/// We use the host MOVSX directly. The destination width determines
/// the writeback width: 32-bit dst zero-extends to 64 (canonical x86-64
/// 32-bit write semantics), 64-bit dst is a full qword store, 16-bit
/// dst writes only the low word and preserves the upper 48 bits.
bool EmitMovsx(const ZydisDecodedInstruction& insn, const ZydisDecodedOperand* ops, u64 next_rip,
               Xbyak::CodeGenerator& c) {
    if (ops[0].type != ZYDIS_OPERAND_TYPE_REGISTER)
        return false;
    const int dst_idx = ZydisGprToIndex(ops[0].reg.value);
    if (dst_idx < 0)
        return false;

    const u32 dst_size = insn.operand_width;
    if (dst_size != 16 && dst_size != 32 && dst_size != 64)
        return false;

    const u32 src_size = ops[1].size;
    if (src_size != 8 && src_size != 16)
        return false;

    // Resolve source operand into a "source addressing context": either
    // a guest GPR slot, or a host memory address (rdx) after
    // EmitEffectiveAddress. We then dispatch host MOVSX at the right
    // width — using the narrow-dst form (eax/ax) for narrow dst,
    // since the host's 32-bit-write zero-extension semantic gives us
    // the exact guest behavior for dst_size == 32 without an extra
    // explicit zero-extend step.
    bool src_is_mem = (ops[1].type == ZYDIS_OPERAND_TYPE_MEMORY);
    int src_idx = -1;
    if (ops[1].type == ZYDIS_OPERAND_TYPE_REGISTER) {
        src_idx = ZydisGprToIndex(ops[1].reg.value);
        if (src_idx < 0)
            return false;
    } else if (src_is_mem) {
        if (!EmitEffectiveAddress(ops[1].mem, next_rip, c))
            return false;
    } else {
        return false;
    }

    if (dst_size == 16) {
        if (src_size == 8) {
            if (src_is_mem)
                c.movsx(ax, byte[rdx]);
            else
                c.movsx(ax, byte[r13 + GprOffset(src_idx)]);
        } else {
            // 16-bit dst with 16-bit src is a degenerate MOV; just
            // copy the low word. (Not expected in real code.)
            if (src_is_mem)
                c.mov(ax, word[rdx]);
            else
                c.mov(ax, word[r13 + GprOffset(src_idx)]);
        }
        c.mov(word[r13 + GprOffset(dst_idx)], ax);
        return true;
    }

    if (dst_size == 32) {
        if (src_size == 8) {
            if (src_is_mem)
                c.movsx(eax, byte[rdx]);
            else
                c.movsx(eax, byte[r13 + GprOffset(src_idx)]);
        } else {
            if (src_is_mem)
                c.movsx(eax, word[rdx]);
            else
                c.movsx(eax, word[r13 + GprOffset(src_idx)]);
        }
        // Host 32-bit MOVSX zero-extends bits 63:32 of rax;
        // qword writeback stores the canonical 32-bit-write
        // representation into the guest slot.
        c.mov(qword[r13 + GprOffset(dst_idx)], rax);
        return true;
    }

    // dst_size == 64
    if (src_size == 8) {
        if (src_is_mem)
            c.movsx(rax, byte[rdx]);
        else
            c.movsx(rax, byte[r13 + GprOffset(src_idx)]);
    } else {
        if (src_is_mem)
            c.movsx(rax, word[rdx]);
        else
            c.movsx(rax, word[r13 + GprOffset(src_idx)]);
    }
    c.mov(qword[r13 + GprOffset(dst_idx)], rax);
    return true;
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
constexpr u64 AllArith = CF | PF | ZF | SF | OF;
} // namespace RflagsBits

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
    c.shl(r8, 2); // r8 = PF_bit << 2 (PF is at bit 2)
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
/// Clobbers r8, r9, r10. Writes rflags.
void EmitFlagsFromSubtract(Xbyak::CodeGenerator& c) {
    EmitClearArithFlags(c);

    // ZF: result == 0
    c.test(rax, rax);
    c.setz(r8b);
    c.movzx(r8, r8b);
    c.shl(r8, 6); // ZF at bit 6
    c.or_(qword[r13 + Offsets::Rflags], r8);

    // SF: result >> 63
    c.mov(r8, rax);
    c.shr(r8, 63);
    c.shl(r8, 7); // SF at bit 7
    c.or_(qword[r13 + Offsets::Rflags], r8);

    // CF for subtract: lhs < rhs (unsigned).
    c.cmp(rcx, rdx);
    c.setb(r8b); // setb = "set if below", unsigned <
    c.movzx(r8, r8b);
    // CF is at bit 0; no shift needed.
    c.or_(qword[r13 + Offsets::Rflags], r8);

    // OF for subtract: ((lhs ^ rhs) & (lhs ^ result)) >> 63
    c.mov(r8, rcx);
    c.xor_(r8, rdx); // r8 = lhs ^ rhs
    c.mov(r9, rcx);
    c.xor_(r9, rax); // r9 = lhs ^ result
    c.and_(r8, r9);
    c.shr(r8, 63);
    c.shl(r8, 11); // OF at bit 11
    c.or_(qword[r13 + Offsets::Rflags], r8);

    // PF: parity of low byte of result.
    EmitWritePF(c);
}

/// Compute flags for an add (ADD).
/// Inputs:
///   rcx = lhs, rdx = rhs, rax = lhs + rhs
/// Clobbers r8, r9, r10. Writes rflags.
void EmitFlagsFromAdd(Xbyak::CodeGenerator& c) {
    EmitClearArithFlags(c);

    // ZF: result == 0
    c.test(rax, rax);
    c.setz(r8b);
    c.movzx(r8, r8b);
    c.shl(r8, 6);
    c.or_(qword[r13 + Offsets::Rflags], r8);

    // SF: result >> 63
    c.mov(r8, rax);
    c.shr(r8, 63);
    c.shl(r8, 7);
    c.or_(qword[r13 + Offsets::Rflags], r8);

    // CF for add: result < lhs (unsigned add overflow).
    c.cmp(rax, rcx);
    c.setb(r8b);
    c.movzx(r8, r8b);
    c.or_(qword[r13 + Offsets::Rflags], r8);

    // OF for add: (~(lhs ^ rhs) & (lhs ^ result)) >> 63
    c.mov(r8, rcx);
    c.xor_(r8, rdx); // r8 = lhs ^ rhs
    c.not_(r8);      // r8 = ~(lhs ^ rhs)
    c.mov(r9, rcx);
    c.xor_(r9, rax); // r9 = lhs ^ result
    c.and_(r8, r9);
    c.shr(r8, 63);
    c.shl(r8, 11);
    c.or_(qword[r13 + Offsets::Rflags], r8);

    // PF: parity of low byte of result.
    EmitWritePF(c);
}

/// Compute flags for a bitwise op (AND, TEST, XOR, OR).
/// Inputs: rax = result. (lhs and rhs unused — CF and OF are
/// always 0 for bitwise ops per x86 spec.)
/// Clobbers r8, r9. Writes rflags.
void EmitFlagsFromBitwise(Xbyak::CodeGenerator& c) {
    EmitClearArithFlags(c);

    // ZF: result == 0
    c.test(rax, rax);
    c.setz(r8b);
    c.movzx(r8, r8b);
    c.shl(r8, 6);
    c.or_(qword[r13 + Offsets::Rflags], r8);

    // SF: result >> 63
    c.mov(r8, rax);
    c.shr(r8, 63);
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
bool EmitAdd(const ZydisDecodedInstruction& insn, const ZydisDecodedOperand* ops, u64 next_rip,
             Xbyak::CodeGenerator& c) {
    const auto& dst = ops[0];
    const auto& src = ops[1];

    // 64-bit ADD with memory destination: `add qword[mem], r64`.
    // Mirrors EmitOr's mem-dst pattern. We stash the computed address
    // into r10 before reusing rdx as the rhs operand for the flag
    // helper. EmitFlagsFromAdd clobbers r8/r9 only, so r10 survives
    // until after the writeback (and beyond — we don't actually need
    // r10 after the store).
    if (insn.operand_width == 64 && dst.type == ZYDIS_OPERAND_TYPE_MEMORY &&
        src.type == ZYDIS_OPERAND_TYPE_REGISTER) {
        const int src_idx = ZydisGprToIndex(src.reg.value);
        if (src_idx < 0)
            return false;
        if (!EmitEffectiveAddress(dst.mem, next_rip, c))
            return false;
        // rdx = address; preserve it in r10 so the flag helper can
        // use rdx for the rhs.
        c.mov(r10, rdx);
        c.mov(rcx, qword[r10]);                      // rcx = lhs (orig [mem])
        c.mov(rdx, qword[r13 + GprOffset(src_idx)]); // rdx = rhs
        c.mov(rax, rcx);
        c.add(rax, rdx);
        c.mov(qword[r10], rax); // store result
        EmitFlagsFromAdd(c);
        return true;
    }

    if (dst.type != ZYDIS_OPERAND_TYPE_REGISTER)
        return false;
    const int dst_idx = ZydisGprToIndex(dst.reg.value);
    if (dst_idx < 0)
        return false;
    if (insn.operand_width != 64)
        return false;

    if (src.type == ZYDIS_OPERAND_TYPE_REGISTER) {
        const int src_idx = ZydisGprToIndex(src.reg.value);
        if (src_idx < 0)
            return false;

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

    // reg-mem: add r64, qword[mem]. EmitEffectiveAddress yields the
    // address in rdx; we overwrite rdx with the loaded value so it
    // can serve as the rhs for the flag helper (which wants rhs in
    // rdx). Order matters here: we must compute the address before
    // touching rcx/rax, since EmitEffectiveAddress may use rax for
    // index*scale arithmetic.
    if (src.type == ZYDIS_OPERAND_TYPE_MEMORY) {
        if (!EmitEffectiveAddress(src.mem, next_rip, c))
            return false;
        c.mov(rdx, qword[rdx]);                      // rdx = rhs (loaded)
        c.mov(rcx, qword[r13 + GprOffset(dst_idx)]); // rcx = lhs (dst current)
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
bool EmitSub(const ZydisDecodedInstruction& insn, const ZydisDecodedOperand* ops, u64 next_rip,
             Xbyak::CodeGenerator& c) {
    if (insn.operand_width != 64)
        return false;
    const auto& dst = ops[0];
    const auto& src = ops[1];

    // 64-bit SUB with memory destination: `sub qword[mem], r64`.
    // Mirrors EmitAdd's mem-dst pattern; stash address in r10 to free
    // rdx for the flag helper's rhs slot.
    if (dst.type == ZYDIS_OPERAND_TYPE_MEMORY && src.type == ZYDIS_OPERAND_TYPE_REGISTER) {
        const int src_idx = ZydisGprToIndex(src.reg.value);
        if (src_idx < 0)
            return false;
        if (!EmitEffectiveAddress(dst.mem, next_rip, c))
            return false;
        c.mov(r10, rdx);                             // r10 = addr
        c.mov(rcx, qword[r10]);                      // rcx = lhs ([mem])
        c.mov(rdx, qword[r13 + GprOffset(src_idx)]); // rdx = rhs
        c.mov(rax, rcx);
        c.sub(rax, rdx);
        c.mov(qword[r10], rax); // store result
        EmitFlagsFromSubtract(c);
        return true;
    }

    if (dst.type != ZYDIS_OPERAND_TYPE_REGISTER)
        return false;
    const int dst_idx = ZydisGprToIndex(dst.reg.value);
    if (dst_idx < 0)
        return false;

    if (src.type == ZYDIS_OPERAND_TYPE_REGISTER) {
        const int src_idx = ZydisGprToIndex(src.reg.value);
        if (src_idx < 0)
            return false;
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

    // reg-mem: sub r64, qword[mem]. Same skeleton as EmitAdd's
    // mem-src branch — compute the address, overwrite rdx with the
    // loaded qword (rhs), load dst into rcx (lhs), subtract into rax,
    // store back, flags via the existing helper.
    if (src.type == ZYDIS_OPERAND_TYPE_MEMORY) {
        if (!EmitEffectiveAddress(src.mem, next_rip, c))
            return false;
        c.mov(rdx, qword[rdx]);                      // rdx = rhs (loaded)
        c.mov(rcx, qword[r13 + GprOffset(dst_idx)]); // rcx = lhs (dst current)
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
bool EmitCmp(const ZydisDecodedInstruction& insn, const ZydisDecodedOperand* ops, u64 next_rip,
             Xbyak::CodeGenerator& c) {
    if (insn.operand_width != 64)
        return false;
    const auto& lhs_op = ops[0];
    const auto& rhs_op = ops[1];

    // ----- lhs is a register -----
    if (lhs_op.type == ZYDIS_OPERAND_TYPE_REGISTER) {
        const int lhs_idx = ZydisGprToIndex(lhs_op.reg.value);
        if (lhs_idx < 0)
            return false;

        if (rhs_op.type == ZYDIS_OPERAND_TYPE_REGISTER) {
            const int rhs_idx = ZydisGprToIndex(rhs_op.reg.value);
            if (rhs_idx < 0)
                return false;
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
            if (!EmitEffectiveAddress(rhs_op.mem, next_rip, c))
                return false;
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
        if (!EmitEffectiveAddress(lhs_op.mem, next_rip, c))
            return false;
        c.mov(rcx, qword[rdx]);

        if (rhs_op.type == ZYDIS_OPERAND_TYPE_REGISTER) {
            const int rhs_idx = ZydisGprToIndex(rhs_op.reg.value);
            if (rhs_idx < 0)
                return false;
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
bool EmitTest(const ZydisDecodedInstruction& insn, const ZydisDecodedOperand* ops,
              Xbyak::CodeGenerator& c) {
    if (insn.operand_width != 64)
        return false;
    const auto& lhs_op = ops[0];
    const auto& rhs_op = ops[1];

    if (lhs_op.type != ZYDIS_OPERAND_TYPE_REGISTER)
        return false;
    const int lhs_idx = ZydisGprToIndex(lhs_op.reg.value);
    if (lhs_idx < 0)
        return false;

    if (rhs_op.type == ZYDIS_OPERAND_TYPE_REGISTER) {
        const int rhs_idx = ZydisGprToIndex(rhs_op.reg.value);
        if (rhs_idx < 0)
            return false;
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
bool EmitXor(const ZydisDecodedInstruction& insn, const ZydisDecodedOperand* ops, u64 next_rip,
             Xbyak::CodeGenerator& c) {
    const auto& dst = ops[0];
    const auto& src = ops[1];

    if (dst.type != ZYDIS_OPERAND_TYPE_REGISTER)
        return false;
    const int dst_idx = ZydisGprToIndex(dst.reg.value);
    if (dst_idx < 0)
        return false;

    // 64-bit reg-mem: `xor r64, qword[mem]`. Same skeleton as the
    // ADD/SUB reg-mem branches — compute address into rdx, overwrite
    // with the loaded value, load dst into rax, xor, store back.
    if (insn.operand_width == 64 && src.type == ZYDIS_OPERAND_TYPE_MEMORY) {
        if (!EmitEffectiveAddress(src.mem, next_rip, c))
            return false;
        c.mov(rdx, qword[rdx]);
        c.mov(rax, qword[r13 + GprOffset(dst_idx)]);
        c.xor_(rax, rdx);
        c.mov(qword[r13 + GprOffset(dst_idx)], rax);
        EmitFlagsFromBitwise(c);
        return true;
    }

    if (src.type != ZYDIS_OPERAND_TYPE_REGISTER)
        return false;
    const int src_idx = ZydisGprToIndex(src.reg.value);
    if (src_idx < 0)
        return false;

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

// =============================================================================
// AND, OR — bitwise binary ops, mirror EmitXor but produce different
// results. Both clear CF/OF and set ZF/SF/PF based on the result.
// =============================================================================

/// Generic helper for 64-bit reg-reg bitwise dispatch. The actual
/// host op (`and`/`or`/`xor`) is supplied as a lambda so all three
/// instructions share the same load/store/flag scaffolding.
template <typename HostOp>
bool EmitBitwise64RegReg(const ZydisDecodedOperand& dst, const ZydisDecodedOperand& src,
                         Xbyak::CodeGenerator& c, HostOp host_op) {
    if (dst.type != ZYDIS_OPERAND_TYPE_REGISTER)
        return false;
    if (src.type != ZYDIS_OPERAND_TYPE_REGISTER)
        return false;
    const int dst_idx = ZydisGprToIndex(dst.reg.value);
    const int src_idx = ZydisGprToIndex(src.reg.value);
    if (dst_idx < 0 || src_idx < 0)
        return false;

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
bool EmitBitwise32RegReg(const ZydisDecodedOperand& dst, const ZydisDecodedOperand& src,
                         Xbyak::CodeGenerator& c, HostOp host_op) {
    if (dst.type != ZYDIS_OPERAND_TYPE_REGISTER)
        return false;
    if (src.type != ZYDIS_OPERAND_TYPE_REGISTER)
        return false;
    const int dst_idx = ZydisGprToIndex(dst.reg.value);
    const int src_idx = ZydisGprToIndex(src.reg.value);
    if (dst_idx < 0 || src_idx < 0)
        return false;

    c.mov(eax, dword[r13 + GprOffset(dst_idx)]);
    c.mov(ecx, dword[r13 + GprOffset(src_idx)]);
    host_op(eax, ecx); // 32-bit op zero-extends rax
    c.mov(qword[r13 + GprOffset(dst_idx)], rax);
    EmitFlagsFromBitwise(c);
    return true;
}

/// AND — bitwise and. Supported: r64,r64; r32,r32; r32,imm; r64,imm;
/// r64,[mem]. `next_rip` is needed for the mem-src form's address
/// calculation (RIP-relative case).
bool EmitAnd(const ZydisDecodedInstruction& insn, const ZydisDecodedOperand* ops, u64 next_rip,
             Xbyak::CodeGenerator& c) {
    // 32-bit register destination with immediate source: very common
    // masking idiom (`and eax, 0xFF`, `and ecx, 0x3F`, etc.). Mirrors
    // the 32-bit reg-reg path's flag-handling: EmitFlagsFromBitwise
    // produces the same lazy flag computation, so behavior matches
    // the existing 32-bit AND.
    if (insn.operand_width == 32 && ops[0].type == ZYDIS_OPERAND_TYPE_REGISTER &&
        ops[1].type == ZYDIS_OPERAND_TYPE_IMMEDIATE) {
        const int dst_idx = ZydisGprToIndex(ops[0].reg.value);
        if (dst_idx < 0)
            return false;
        // Zydis sign-extends imm8 forms into a u64 for us; truncate to
        // u32 — the bit pattern is preserved for the masking case.
        const u32 imm = static_cast<u32>(ops[1].imm.value.u & 0xFFFFFFFFu);
        c.mov(eax, dword[r13 + GprOffset(dst_idx)]);
        c.and_(eax, imm); // 32-bit op zero-extends rax
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
    if (insn.operand_width == 64 && ops[0].type == ZYDIS_OPERAND_TYPE_REGISTER &&
        ops[1].type == ZYDIS_OPERAND_TYPE_IMMEDIATE) {
        const int dst_idx = ZydisGprToIndex(ops[0].reg.value);
        if (dst_idx < 0)
            return false;
        const auto imm = ops[1].imm.value.s;
        c.mov(rax, qword[r13 + GprOffset(dst_idx)]);
        c.mov(rdx, imm);
        c.and_(rax, rdx);
        c.mov(qword[r13 + GprOffset(dst_idx)], rax);
        EmitFlagsFromBitwise(c);
        return true;
    }

    // 64-bit reg-mem: `and r64, qword[mem]`. Same skeleton as ADD/SUB
    // reg-mem — compute address into rdx, overwrite with loaded value,
    // load dst into rax, AND, store back.
    if (insn.operand_width == 64 && ops[0].type == ZYDIS_OPERAND_TYPE_REGISTER &&
        ops[1].type == ZYDIS_OPERAND_TYPE_MEMORY) {
        const int dst_idx = ZydisGprToIndex(ops[0].reg.value);
        if (dst_idx < 0)
            return false;
        if (!EmitEffectiveAddress(ops[1].mem, next_rip, c))
            return false;
        c.mov(rdx, qword[rdx]);
        c.mov(rax, qword[r13 + GprOffset(dst_idx)]);
        c.and_(rax, rdx);
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
bool EmitOr(const ZydisDecodedInstruction& insn, const ZydisDecodedOperand* ops, u64 next_rip,
            Xbyak::CodeGenerator& c) {
    // 64-bit OR with memory destination: `or qword[mem], r64`.
    // Strict mem-dst + reg-src for now; mem-dst + imm and 32-bit
    // memory forms are deferred to keep the diff focused.
    if (insn.operand_width == 64 && ops[0].type == ZYDIS_OPERAND_TYPE_MEMORY &&
        ops[1].type == ZYDIS_OPERAND_TYPE_REGISTER) {
        const int src_idx = ZydisGprToIndex(ops[1].reg.value);
        if (src_idx < 0)
            return false;
        if (!EmitEffectiveAddress(ops[0].mem, next_rip, c))
            return false;
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

    // 64-bit reg-mem: `or r64, qword[mem]`. Same skeleton as the
    // ADD/SUB/AND/XOR reg-mem branches.
    if (insn.operand_width == 64 && ops[0].type == ZYDIS_OPERAND_TYPE_REGISTER &&
        ops[1].type == ZYDIS_OPERAND_TYPE_MEMORY) {
        const int dst_idx = ZydisGprToIndex(ops[0].reg.value);
        if (dst_idx < 0)
            return false;
        if (!EmitEffectiveAddress(ops[1].mem, next_rip, c))
            return false;
        c.mov(rdx, qword[rdx]);
        c.mov(rax, qword[r13 + GprOffset(dst_idx)]);
        c.or_(rax, rdx);
        c.mov(qword[r13 + GprOffset(dst_idx)], rax);
        EmitFlagsFromBitwise(c);
        return true;
    }

    // 64-bit register destination with immediate source: `or r64, imm`.
    // Same shape as EmitAnd's reg-imm 64-bit branch. Materialise the
    // immediate into rdx so the OR has a register operand (xbyak can't
    // encode `or rax, imm64`; imm32-sx via rdx is the canonical route).
    if (insn.operand_width == 64 && ops[0].type == ZYDIS_OPERAND_TYPE_REGISTER &&
        ops[1].type == ZYDIS_OPERAND_TYPE_IMMEDIATE) {
        const int dst_idx = ZydisGprToIndex(ops[0].reg.value);
        if (dst_idx < 0)
            return false;
        const auto imm = ops[1].imm.value.s;
        c.mov(rax, qword[r13 + GprOffset(dst_idx)]);
        c.mov(rdx, imm);
        c.or_(rax, rdx);
        c.mov(qword[r13 + GprOffset(dst_idx)], rax);
        EmitFlagsFromBitwise(c);
        return true;
    }

    // 32-bit register destination with immediate source: `or r32, imm`.
    // Mirrors the AND 32-bit reg-imm path; 32-bit write zero-extends rax.
    if (insn.operand_width == 32 && ops[0].type == ZYDIS_OPERAND_TYPE_REGISTER &&
        ops[1].type == ZYDIS_OPERAND_TYPE_IMMEDIATE) {
        const int dst_idx = ZydisGprToIndex(ops[0].reg.value);
        if (dst_idx < 0)
            return false;
        const u32 imm = static_cast<u32>(ops[1].imm.value.u & 0xFFFFFFFFu);
        c.mov(eax, dword[r13 + GprOffset(dst_idx)]);
        c.or_(eax, imm); // 32-bit op zero-extends rax
        c.mov(qword[r13 + GprOffset(dst_idx)], rax);
        EmitFlagsFromBitwise(c);
        return true;
    }

    // 8-bit OR with memory destination and byte immediate: `or byte[mem], imm8`.
    // Observed in Sonic Mania (libc flags field update pattern).
    //
    // Flags: OR clears CF/OF; ZF/SF/PF from the 8-bit result. We
    // round-trip rflags through the host CPU (same pattern as 8-bit
    // shifts / NEG / INC) so the host `or al, imm8` computes SF from
    // bit 7 of the result — the wide EmitFlagsFromBitwise helper would
    // incorrectly derive SF from bit 63 instead.
    if (insn.operand_width == 8 && ops[0].type == ZYDIS_OPERAND_TYPE_MEMORY &&
        ops[1].type == ZYDIS_OPERAND_TYPE_IMMEDIATE) {
        if (!EmitEffectiveAddress(ops[0].mem, next_rip, c))
            return false;
        // rdx = effective address. Stash it in r10 so the rflags
        // round-trip (which uses rdx as the intermediate) doesn't
        // clobber it.
        c.mov(r10, rdx);
        const u8 imm = static_cast<u8>(ops[1].imm.value.u & 0xFFu);
        c.mov(al, byte[r10]);

        c.mov(rdx, qword[r13 + Offsets::Rflags]);
        c.push(rdx);
        c.popfq();

        c.or_(al, imm);

        c.pushfq();
        c.pop(rdx);
        c.mov(qword[r13 + Offsets::Rflags], rdx);

        c.mov(byte[r10], al);
        return true;
    }

    if (insn.operand_width == 64) {
        return EmitBitwise64RegReg(ops[0], ops[1], c,
                                   [&](Xbyak::Reg64 a, Xbyak::Reg64 b) { c.or_(a, b); });
    }
    if (insn.operand_width == 32) {
        return EmitBitwise32RegReg(ops[0], ops[1], c,
                                   [&](Xbyak::Reg32 a, Xbyak::Reg32 b) { c.or_(a, b); });
    }
    return false;
}

// =============================================================================
// NOT, NEG — unary ops.
// =============================================================================

/// NOT r/m — bitwise complement. Per x86 spec, NOT does NOT affect
/// any flags at any width. So we skip the round-trip-flags pattern
/// the binary narrow-arith ops use.
bool EmitNot(const ZydisDecodedInstruction& insn, const ZydisDecodedOperand* ops,
             Xbyak::CodeGenerator& c) {
    if (ops[0].type != ZYDIS_OPERAND_TYPE_REGISTER)
        return false;
    const int idx = ZydisGprToIndex(ops[0].reg.value);
    if (idx < 0)
        return false;

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
bool EmitNeg(const ZydisDecodedInstruction& insn, const ZydisDecodedOperand* ops,
             Xbyak::CodeGenerator& c) {
    if (ops[0].type != ZYDIS_OPERAND_TYPE_REGISTER)
        return false;
    const int idx = ZydisGprToIndex(ops[0].reg.value);
    if (idx < 0)
        return false;

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
    if (insn.operand_width != 8 && insn.operand_width != 16 && insn.operand_width != 32) {
        return false;
    }

    c.mov(rdx, qword[r13 + Offsets::Rflags]);
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
    c.mov(qword[r13 + Offsets::Rflags], rdx);
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
bool EmitInc(const ZydisDecodedInstruction& insn, const ZydisDecodedOperand* ops,
             Xbyak::CodeGenerator& c) {
    if (insn.operand_width != 64 && insn.operand_width != 32)
        return false;
    if (ops[0].type != ZYDIS_OPERAND_TYPE_REGISTER)
        return false;
    const int idx = ZydisGprToIndex(ops[0].reg.value);
    if (idx < 0)
        return false;

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
        c.push(rdx);
        c.popfq();

        c.add(eax, ecx);

        c.pushfq();
        c.pop(rdx);
        c.mov(qword[r13 + Offsets::Rflags], rdx);

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
    c.btr(r11, 0);   // clear bit 0 (CF)
    c.or_(r11, r10); // OR in saved CF
    c.mov(qword[r13 + Offsets::Rflags], r11);
    return true;
}

/// DEC r/m — subtract 1, preserve CF. 64- and 32-bit register forms.
bool EmitDec(const ZydisDecodedInstruction& insn, const ZydisDecodedOperand* ops,
             Xbyak::CodeGenerator& c) {
    if (insn.operand_width != 64 && insn.operand_width != 32)
        return false;
    if (ops[0].type != ZYDIS_OPERAND_TYPE_REGISTER)
        return false;
    const int idx = ZydisGprToIndex(ops[0].reg.value);
    if (idx < 0)
        return false;

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
        c.push(rdx);
        c.popfq();

        c.sub(eax, ecx);

        c.pushfq();
        c.pop(rdx);
        c.mov(qword[r13 + Offsets::Rflags], rdx);

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

// =============================================================================
// BT — bit test. Reads bit `src mod opsize` from `dst`, sets CF to
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
bool EmitBt(const ZydisDecodedInstruction& insn, const ZydisDecodedOperand* ops,
            Xbyak::CodeGenerator& c) {
    if (ops[0].type != ZYDIS_OPERAND_TYPE_REGISTER)
        return false;
    if (insn.operand_width != 64 && insn.operand_width != 32)
        return false;

    const int src_idx = ZydisGprToIndex(ops[0].reg.value);
    if (src_idx < 0)
        return false;

    if (ops[1].type == ZYDIS_OPERAND_TYPE_REGISTER) {
        // bt r32/r64, r32/r64 — bit index from a GPR. Host BT
        // architecturally masks the index by opsize-1 (cl & 63 for
        // 64-bit, cl & 31 for 32-bit), matching guest semantics for
        // both widths.
        const int bit_idx = ZydisGprToIndex(ops[1].reg.value);
        if (bit_idx < 0)
            return false;
        if (insn.operand_width == 64) {
            c.mov(rax, qword[r13 + GprOffset(src_idx)]);
            c.mov(rcx, qword[r13 + GprOffset(bit_idx)]);
            c.bt(rax, rcx);
        } else { // 32-bit
            c.mov(eax, dword[r13 + GprOffset(src_idx)]);
            c.mov(ecx, dword[r13 + GprOffset(bit_idx)]);
            c.bt(eax, ecx);
        }
    } else if (ops[1].type == ZYDIS_OPERAND_TYPE_IMMEDIATE) {
        // bt r32/r64, imm8 — Sonic Mania fast single-bit-test idiom.
        // Observed as `bt r32, imm` / `jb target` in the Hello World
        // and video-out init paths.
        const unsigned int bit_pos = static_cast<unsigned int>(ops[1].imm.value.u & 0xFFu);
        if (insn.operand_width == 64) {
            c.mov(rax, qword[r13 + GprOffset(src_idx)]);
            c.bt(rax, bit_pos);
        } else { // 32-bit
            c.mov(eax, dword[r13 + GprOffset(src_idx)]);
            c.bt(eax, bit_pos);
        }
    } else {
        return false;
    }

    // ZF/SF/PF/OF are undefined after BT (Intel SDM Vol. 2A). Capture
    // only CF via setc+btr+or so other guest flags are undisturbed.
    c.setc(r11b);
    c.movzx(r11, r11b);
    c.mov(r10, qword[r13 + Offsets::Rflags]);
    c.btr(r10, 0);   // clear guest CF
    c.or_(r10, r11); // OR in new CF
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
bool EmitMovzx(const ZydisDecodedInstruction& insn, const ZydisDecodedOperand* ops, u64 next_rip,
               Xbyak::CodeGenerator& c) {
    if (ops[0].type != ZYDIS_OPERAND_TYPE_REGISTER)
        return false;
    const int dst_idx = ZydisGprToIndex(ops[0].reg.value);
    if (dst_idx < 0)
        return false;

    // Destination width must be one of 16/32/64. (Per the x86 spec,
    // those are the only legal MOVZX destinations.)
    const u32 dst_size = insn.operand_width;
    if (dst_size != 16 && dst_size != 32 && dst_size != 64)
        return false;

    // Source operand size in bits (8 or 16 for any legal MOVZX).
    const u32 src_size = ops[1].size;
    if (src_size != 8 && src_size != 16)
        return false;

    if (ops[1].type == ZYDIS_OPERAND_TYPE_REGISTER) {
        const int src_idx = ZydisGprToIndex(ops[1].reg.value);
        if (src_idx < 0)
            return false;
        if (src_size == 8) {
            c.movzx(rax, byte[r13 + GprOffset(src_idx)]);
        } else {
            c.movzx(rax, word[r13 + GprOffset(src_idx)]);
        }
    } else if (ops[1].type == ZYDIS_OPERAND_TYPE_MEMORY) {
        if (!EmitEffectiveAddress(ops[1].mem, next_rip, c))
            return false;
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

// =============================================================================
// CMOV — conditional move (branchless if-then).
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
    case ZYDIS_MNEMONIC_CMOVZ:
        return ZYDIS_MNEMONIC_JZ;
    case ZYDIS_MNEMONIC_CMOVNZ:
        return ZYDIS_MNEMONIC_JNZ;
    case ZYDIS_MNEMONIC_CMOVS:
        return ZYDIS_MNEMONIC_JS;
    case ZYDIS_MNEMONIC_CMOVNS:
        return ZYDIS_MNEMONIC_JNS;
    case ZYDIS_MNEMONIC_CMOVO:
        return ZYDIS_MNEMONIC_JO;
    case ZYDIS_MNEMONIC_CMOVNO:
        return ZYDIS_MNEMONIC_JNO;
    case ZYDIS_MNEMONIC_CMOVP:
        return ZYDIS_MNEMONIC_JP;
    case ZYDIS_MNEMONIC_CMOVNP:
        return ZYDIS_MNEMONIC_JNP;
    case ZYDIS_MNEMONIC_CMOVB:
        return ZYDIS_MNEMONIC_JB;
    case ZYDIS_MNEMONIC_CMOVNB:
        return ZYDIS_MNEMONIC_JNB;
    case ZYDIS_MNEMONIC_CMOVBE:
        return ZYDIS_MNEMONIC_JBE;
    case ZYDIS_MNEMONIC_CMOVNBE:
        return ZYDIS_MNEMONIC_JNBE;
    case ZYDIS_MNEMONIC_CMOVL:
        return ZYDIS_MNEMONIC_JL;
    case ZYDIS_MNEMONIC_CMOVNL:
        return ZYDIS_MNEMONIC_JNL;
    case ZYDIS_MNEMONIC_CMOVLE:
        return ZYDIS_MNEMONIC_JLE;
    case ZYDIS_MNEMONIC_CMOVNLE:
        return ZYDIS_MNEMONIC_JNLE;
    default:
        return ZYDIS_MNEMONIC_INVALID;
    }
}

/// CMOVcc r64, r/m64 — if condition true, dst = src.
/// 64-bit only for now; 32-bit CMOV would follow the same shape.
bool EmitCmov(const ZydisDecodedInstruction& insn, const ZydisDecodedOperand* ops, u64 next_rip,
              Xbyak::CodeGenerator& c) {
    if (insn.operand_width != 64 && insn.operand_width != 32)
        return false;
    if (ops[0].type != ZYDIS_OPERAND_TYPE_REGISTER)
        return false;
    const int dst_idx = ZydisGprToIndex(ops[0].reg.value);
    if (dst_idx < 0)
        return false;

    const ZydisMnemonic jcc_equiv = CmovToJcc(insn.mnemonic);
    if (jcc_equiv == ZYDIS_MNEMONIC_INVALID)
        return false;

    // Load src into r8 first — we'll need rax/rcx/rdx for the
    // condition computation. For 32-bit, we load the full qword from
    // the slot and rely on the eax-write further down to perform the
    // zero-extension when the condition is true.
    if (ops[1].type == ZYDIS_OPERAND_TYPE_REGISTER) {
        const int src_idx = ZydisGprToIndex(ops[1].reg.value);
        if (src_idx < 0)
            return false;
        c.mov(r8, qword[r13 + GprOffset(src_idx)]);
    } else if (ops[1].type == ZYDIS_OPERAND_TYPE_MEMORY) {
        if (!EmitEffectiveAddress(ops[1].mem, next_rip, c))
            return false;
        c.mov(r8, qword[rdx]);
    } else {
        return false;
    }

    // Load current dst slot into r9 (the candidate "no-change" result).
    // For 32-bit CMOVcc with the condition FALSE, the destination is
    // unchanged ENTIRELY — including bits 63:32. This is a quirk of
    // CMOVcc relative to ordinary 32-bit ops, which always zero-extend.
    // So r9 must be the full qword, not the low 32.
    c.mov(r9, qword[r13 + GprOffset(dst_idx)]);

    // Compute condition into rcx (0 or 1).
    if (!EmitJccCondition(jcc_equiv, c))
        return false;

    // For 32-bit, build the cond-TRUE result by zero-extending the low
    // 32 of the src into rax via a same-name reg-reg mov (mov eax, r8d).
    // This is the standard idiom to clear bits 63:32 of a host GPR.
    // Then cmov picks rax (zero-extended src) when the cond was true,
    // r9 (unchanged dst slot) otherwise. For 64-bit, we use r8 as-is.
    c.test(rcx, rcx);
    if (insn.operand_width == 32) {
        c.mov(eax, r8d); // rax = src & 0xFFFFFFFF, upper 32 = 0
        c.cmovnz(r9, rax);
    } else {
        c.cmovnz(r9, r8);
    }
    c.mov(qword[r13 + GprOffset(dst_idx)], r9);
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
bool EmitShift64(const ZydisDecodedInstruction& insn, const ZydisDecodedOperand* ops,
                 Xbyak::CodeGenerator& c, ShiftKind kind) {
    if (insn.operand_width != 64)
        return false;
    if (ops[0].type != ZYDIS_OPERAND_TYPE_REGISTER)
        return false;
    const int dst_idx = ZydisGprToIndex(ops[0].reg.value);
    if (dst_idx < 0)
        return false;

    // Load shift count into host cl.
    if (ops[1].type == ZYDIS_OPERAND_TYPE_IMMEDIATE) {
        // Zydis presents the imm value as unsigned u64; only low
        // 6 bits matter for 64-bit shifts, but the host CPU masks
        // anyway. Use a byte move for clarity.
        c.mov(cl, static_cast<u8>(ops[1].imm.value.u & 0xFF));
    } else if (ops[1].type == ZYDIS_OPERAND_TYPE_REGISTER) {
        // The only legal shift-count register is CL itself. (BL/DL
        // etc. are not allowed by the x86 ISA in this slot.)
        if (ops[1].reg.value != ZYDIS_REGISTER_CL)
            return false;
        // Load guest CL = low byte of guest rcx (index 1).
        c.mov(cl, byte[r13 + GprOffset(1)]);
    } else {
        return false;
    }

    // Load destination value into rax.
    c.mov(rax, qword[r13 + GprOffset(dst_idx)]);

    // Round-trip rflags: load guest → host. Use rdx (not in use yet
    // and not aliased to cl).
    c.mov(rdx, qword[r13 + Offsets::Rflags]);
    c.push(rdx);
    c.popfq();

    // Perform the shift. Host x86 shifts consume the count from cl
    // implicitly when given a register operand.
    switch (kind) {
    case ShiftKind::Shl:
        c.shl(rax, cl);
        break;
    case ShiftKind::Shr:
        c.shr(rax, cl);
        break;
    case ShiftKind::Sar:
        c.sar(rax, cl);
        break;
    }

    // Capture host → guest.
    c.pushfq();
    c.pop(rdx);
    c.mov(qword[r13 + Offsets::Rflags], rdx);

    // Store the shifted value back.
    c.mov(qword[r13 + GprOffset(dst_idx)], rax);
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
bool EmitShift32(const ZydisDecodedInstruction& insn, const ZydisDecodedOperand* ops,
                 Xbyak::CodeGenerator& c, ShiftKind kind) {
    if (insn.operand_width != 32)
        return false;
    if (ops[0].type != ZYDIS_OPERAND_TYPE_REGISTER)
        return false;
    const int dst_idx = ZydisGprToIndex(ops[0].reg.value);
    if (dst_idx < 0)
        return false;

    if (ops[1].type == ZYDIS_OPERAND_TYPE_IMMEDIATE) {
        c.mov(cl, static_cast<u8>(ops[1].imm.value.u & 0xFF));
    } else if (ops[1].type == ZYDIS_OPERAND_TYPE_REGISTER) {
        if (ops[1].reg.value != ZYDIS_REGISTER_CL)
            return false;
        c.mov(cl, byte[r13 + GprOffset(1)]);
    } else {
        return false;
    }

    c.mov(eax, dword[r13 + GprOffset(dst_idx)]);

    c.mov(rdx, qword[r13 + Offsets::Rflags]);
    c.push(rdx);
    c.popfq();

    switch (kind) {
    case ShiftKind::Shl:
        c.shl(eax, cl);
        break;
    case ShiftKind::Shr:
        c.shr(eax, cl);
        break;
    case ShiftKind::Sar:
        c.sar(eax, cl);
        break;
    }

    c.pushfq();
    c.pop(rdx);
    c.mov(qword[r13 + Offsets::Rflags], rdx);

    // Full-qword store so the host's 32-bit zero-extension is
    // recorded in the guest GPR slot.
    c.mov(qword[r13 + GprOffset(dst_idx)], rax);
    return true;
}

/// 8-bit shift emitter. Same structure as EmitShift32, but the
/// destination is a byte register and the upper 56 bits of the
/// parent GPR slot must be preserved (unlike the 32-bit form,
/// where x86-64's zero-extend rule does the upper-half clearing
/// for free).
///
/// Observed at libSceLibc 0x807a73217 as `shr r8b, imm` — a
/// byte-field decompose, presumably part of a parser or hash
/// loop. The imm path is what the game uses; CL-source is wired
/// up for free since the body is shared.
///
/// Flag semantics: identical to wider shifts — CF gets the last
/// bit out, ZF/SF/PF from result, OF defined only for 1-bit
/// shifts. The rflags round-trip carries all of this through the
/// host's own 8-bit shift, which matches Intel's spec exactly.
/// As with wider shifts, a runtime count of zero preserves all
/// flags, again courtesy of the round-trip.
///
/// Only register destinations are handled. AH/BH/CH/DH (the
/// high-byte variants) are rejected via ZydisGprToIndex returning
/// -1, since handling them needs a shift in the merge step that
/// we haven't observed a need for.
bool EmitShift8(const ZydisDecodedInstruction& insn, const ZydisDecodedOperand* ops,
                Xbyak::CodeGenerator& c, ShiftKind kind) {
    if (insn.operand_width != 8)
        return false;
    if (ops[0].type != ZYDIS_OPERAND_TYPE_REGISTER)
        return false;
    const int dst_idx = ZydisGprToIndex(ops[0].reg.value);
    if (dst_idx < 0)
        return false; // refuses AH/BH/CH/DH

    if (ops[1].type == ZYDIS_OPERAND_TYPE_IMMEDIATE) {
        c.mov(cl, static_cast<u8>(ops[1].imm.value.u & 0xFF));
    } else if (ops[1].type == ZYDIS_OPERAND_TYPE_REGISTER) {
        if (ops[1].reg.value != ZYDIS_REGISTER_CL)
            return false;
        c.mov(cl, byte[r13 + GprOffset(1)]);
    } else {
        return false;
    }

    // Load the source byte into al. We deliberately load it into
    // al (not into r8b via a wider register name) because that
    // way the host's 8-bit shift naturally targets the same al
    // we'll merge back, and we avoid clobbering rax's upper bits
    // before the merge.
    c.mov(al, byte[r13 + GprOffset(dst_idx)]);

    c.mov(rdx, qword[r13 + Offsets::Rflags]);
    c.push(rdx);
    c.popfq();

    switch (kind) {
    case ShiftKind::Shl:
        c.shl(al, cl);
        break;
    case ShiftKind::Shr:
        c.shr(al, cl);
        break;
    case ShiftKind::Sar:
        c.sar(al, cl);
        break;
    }

    c.pushfq();
    c.pop(rdx);
    c.mov(qword[r13 + Offsets::Rflags], rdx);

    // Merge: dst_slot = (dst_slot & ~0xFF) | (al & 0xFF). Same
    // pattern as EmitSetcc, but al already has the right value;
    // we mask defensively in case any host implementation surprises
    // us with non-zero upper bits in the byte register's parent
    // (it shouldn't, but the cost is one instruction).
    c.movzx(rax, al); // zero-extend the result byte into rax
    c.mov(r8, qword[r13 + GprOffset(dst_idx)]);
    c.mov(rdx, ~static_cast<u64>(0xFF));
    c.and_(r8, rdx);
    c.or_(r8, rax);
    c.mov(qword[r13 + GprOffset(dst_idx)], r8);
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

bool EmitRotate64(const ZydisDecodedInstruction& insn, const ZydisDecodedOperand* ops,
                  Xbyak::CodeGenerator& c, RotateKind kind) {
    if (insn.operand_width != 64)
        return false;
    if (ops[0].type != ZYDIS_OPERAND_TYPE_REGISTER)
        return false;
    const int dst_idx = ZydisGprToIndex(ops[0].reg.value);
    if (dst_idx < 0)
        return false;

    // Shift count from imm or guest CL — identical to EmitShift64.
    if (ops[1].type == ZYDIS_OPERAND_TYPE_IMMEDIATE) {
        c.mov(cl, static_cast<u8>(ops[1].imm.value.u & 0xFF));
    } else if (ops[1].type == ZYDIS_OPERAND_TYPE_REGISTER) {
        if (ops[1].reg.value != ZYDIS_REGISTER_CL)
            return false;
        c.mov(cl, byte[r13 + GprOffset(1)]);
    } else {
        return false;
    }

    c.mov(rax, qword[r13 + GprOffset(dst_idx)]);

    // Round-trip flags via host.
    c.mov(rdx, qword[r13 + Offsets::Rflags]);
    c.push(rdx);
    c.popfq();

    switch (kind) {
    case RotateKind::Rol:
        c.rol(rax, cl);
        break;
    case RotateKind::Ror:
        c.ror(rax, cl);
        break;
    }

    c.pushfq();
    c.pop(rdx);
    c.mov(qword[r13 + Offsets::Rflags], rdx);

    c.mov(qword[r13 + GprOffset(dst_idx)], rax);
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
bool EmitImul1Op(const ZydisDecodedInstruction& insn, const ZydisDecodedOperand* ops, u64 next_rip,
                 Xbyak::CodeGenerator& c) {
    if (insn.operand_width != 64)
        return false;

    // Load src into rcx. For memory operands, EmitEffectiveAddress
    // writes rdx (the address) and clobbers rax — so we must
    // dereference the address into rcx *before* loading rax.
    if (ops[0].type == ZYDIS_OPERAND_TYPE_REGISTER) {
        const int src_idx = ZydisGprToIndex(ops[0].reg.value);
        if (src_idx < 0)
            return false;
        c.mov(rcx, qword[r13 + GprOffset(src_idx)]);
    } else if (ops[0].type == ZYDIS_OPERAND_TYPE_MEMORY) {
        if (!EmitEffectiveAddress(ops[0].mem, next_rip, c))
            return false;
        c.mov(rcx, qword[rdx]);
    } else {
        return false;
    }

    // Load guest RAX into host rax.
    c.mov(rax, qword[r13 + GprOffset(0)]);

    // Flag round-trip.
    c.mov(r8, qword[r13 + Offsets::Rflags]);
    c.push(r8);
    c.popfq();

    c.imul(rcx); // implicit rax operand; rdx:rax = rax * rcx

    c.pushfq();
    c.pop(r8);
    c.mov(qword[r13 + Offsets::Rflags], r8);

    // Write both halves of the result.
    c.mov(qword[r13 + GprOffset(0)], rax); // low → RAX
    c.mov(qword[r13 + GprOffset(2)], rdx); // high → RDX
    return true;
}

// =============================================================================
// DIV — unsigned divide. Single explicit divisor (r/m); the dividend
// is implicit RDX:RAX. Quotient → RAX, remainder → RDX.
//
// First observed in libSceLibc at 0x8079fd328 with a memory divisor.
// The shape mirrors EmitImul1Op (one explicit operand + implicit
// RAX/RDX) with two differences:
//
//   - DIV's dividend is RDX:RAX (both halves), so we load BOTH guest
//     slots before the host op. IMUL's dividend was just RAX.
//   - All flags are documented as "undefined" after DIV (Intel SDM
//     Vol. 2A). We deliberately skip the rflags round-trip so the
//     guest sees its pre-DIV flag state preserved — this matches the
//     letter of the spec (undefined ≡ any value, including unchanged)
//     and avoids the cost of two push/pop pairs around the host op.
//
// Divide-by-zero and quotient-overflow #DE faults propagate as host
// SIGFPE; until we wire up a signal handler that maps these to a
// guest exception, programs that actually divide-by-zero will crash
// the runtime. Real PS4 binaries don't intentionally do this.
// =============================================================================

bool EmitDiv(const ZydisDecodedInstruction& insn, const ZydisDecodedOperand* ops, u64 next_rip,
             Xbyak::CodeGenerator& c) {
    if (insn.operand_width != 64 && insn.operand_width != 32)
        return false;

    // Load divisor into rcx. For memory operands, EmitEffectiveAddress
    // writes rdx (the address) and clobbers rax — so we must
    // dereference the address into rcx *before* loading guest RAX/RDX.
    if (ops[0].type == ZYDIS_OPERAND_TYPE_REGISTER) {
        const int src_idx = ZydisGprToIndex(ops[0].reg.value);
        if (src_idx < 0)
            return false;
        c.mov(rcx, qword[r13 + GprOffset(src_idx)]);
    } else if (ops[0].type == ZYDIS_OPERAND_TYPE_MEMORY) {
        if (!EmitEffectiveAddress(ops[0].mem, next_rip, c))
            return false;
        c.mov(rcx, qword[rdx]);
    } else {
        return false;
    }

    if (insn.operand_width == 64) {
        // Load guest RDX:RAX into host rdx:rax. Order matters: rdx is
        // loaded LAST so it overwrites the address (memory-divisor path)
        // or the unrelated rdx (register-divisor path) only after we've
        // captured everything we need.
        c.mov(rax, qword[r13 + GprOffset(0)]); // RAX = low half of dividend
        c.mov(rdx, qword[r13 + GprOffset(2)]); // RDX = high half of dividend

        c.div(rcx); // implicit rdx:rax; rax = quotient, rdx = remainder

        c.mov(qword[r13 + GprOffset(0)], rax);
        c.mov(qword[r13 + GprOffset(2)], rdx);
        return true;
    }

    // 32-bit DIV: EDX:EAX / ECX → quotient in EAX, remainder in EDX.
    // We load 32-bit slices of guest RAX/RDX into host eax/edx so the
    // host DIV operates on the same 64-bit dividend the guest would.
    // 32-bit writes zero-extend bits 63:32 of the host registers, so
    // the qword storebacks below land clean values in the guest slots
    // (matching the architectural rule that EAX/EDX writes zero the
    // upper 32 of the underlying 64-bit registers).
    c.mov(eax, dword[r13 + GprOffset(0)]);
    c.mov(edx, dword[r13 + GprOffset(2)]);
    c.div(ecx); // edx:eax / ecx → eax,edx
    c.mov(qword[r13 + GprOffset(0)], rax);
    c.mov(qword[r13 + GprOffset(2)], rdx);
    return true;
}

/// 2-op IMUL: dst = dst * src (low 64 bits).
bool EmitImul2Op(const ZydisDecodedInstruction& insn, const ZydisDecodedOperand* ops, u64 next_rip,
                 Xbyak::CodeGenerator& c) {
    if (insn.operand_width != 64 && insn.operand_width != 32)
        return false;
    if (ops[0].type != ZYDIS_OPERAND_TYPE_REGISTER)
        return false;
    const int dst_idx = ZydisGprToIndex(ops[0].reg.value);
    if (dst_idx < 0)
        return false;

    // Load src into rcx (handle memory case first to avoid clobbering rax).
    if (ops[1].type == ZYDIS_OPERAND_TYPE_REGISTER) {
        const int src_idx = ZydisGprToIndex(ops[1].reg.value);
        if (src_idx < 0)
            return false;
        c.mov(rcx, qword[r13 + GprOffset(src_idx)]);
    } else if (ops[1].type == ZYDIS_OPERAND_TYPE_MEMORY) {
        if (!EmitEffectiveAddress(ops[1].mem, next_rip, c))
            return false;
        c.mov(rcx, qword[rdx]);
    } else {
        return false;
    }

    c.mov(rax, qword[r13 + GprOffset(dst_idx)]);

    c.mov(r8, qword[r13 + Offsets::Rflags]);
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
    c.mov(qword[r13 + Offsets::Rflags], r8);

    // Full-qword writeback. For the 32-bit form the 32-bit IMUL
    // already zero-extended rax above, so this stores the canonical
    // zero-extended value into the guest GPR slot.
    c.mov(qword[r13 + GprOffset(dst_idx)], rax);
    return true;
}

/// 3-op IMUL: dst = src * imm.
bool EmitImul3Op(const ZydisDecodedInstruction& insn, const ZydisDecodedOperand* ops, u64 next_rip,
                 Xbyak::CodeGenerator& c) {
    // 32-bit and 64-bit 3-operand forms have identical structure;
    // 16-bit exists too but is rare and not yet observed.
    if (insn.operand_width != 32 && insn.operand_width != 64)
        return false;
    if (ops[0].type != ZYDIS_OPERAND_TYPE_REGISTER)
        return false;
    if (ops[2].type != ZYDIS_OPERAND_TYPE_IMMEDIATE)
        return false;
    const int dst_idx = ZydisGprToIndex(ops[0].reg.value);
    if (dst_idx < 0)
        return false;

    const bool is32 = (insn.operand_width == 32);

    // Load src into ecx/rcx.
    if (ops[1].type == ZYDIS_OPERAND_TYPE_REGISTER) {
        const int src_idx = ZydisGprToIndex(ops[1].reg.value);
        if (src_idx < 0)
            return false;
        if (is32)
            c.mov(ecx, dword[r13 + GprOffset(src_idx)]);
        else
            c.mov(rcx, qword[r13 + GprOffset(src_idx)]);
    } else if (ops[1].type == ZYDIS_OPERAND_TYPE_MEMORY) {
        if (!EmitEffectiveAddress(ops[1].mem, next_rip, c))
            return false;
        if (is32)
            c.mov(ecx, dword[rdx]);
        else
            c.mov(rcx, qword[rdx]);
    } else {
        return false;
    }

    // Zydis returns the sign-extended 64-bit value in imm.value.s
    // regardless of whether the encoded immediate was imm8 or imm32.
    const s64 imm_val = ops[2].imm.value.s;
    if (is32) {
        // 32-bit form: imm sign-extends to 32 bits. xbyak's 32-bit
        // mov takes a signed 32-bit literal; we mask to that width
        // (the value is already a proper sign-extended 32-bit per
        // Zydis's contract, but be explicit).
        c.mov(eax, static_cast<s32>(imm_val));
    } else {
        c.mov(rax, imm_val);
    }

    c.mov(r8, qword[r13 + Offsets::Rflags]);
    c.push(r8);
    c.popfq();

    if (is32)
        c.imul(eax, ecx); // eax = imm * src (32-bit signed)
    else
        c.imul(rax, rcx); // rax = imm * src (64-bit signed)

    c.pushfq();
    c.pop(r8);
    c.mov(qword[r13 + Offsets::Rflags], r8);

    // 32-bit write zero-extends bits 63:32 of rax per x86-64 rules,
    // so storing the full qword captures the correct guest value
    // (32-bit operand width means upper bits of the guest GPR are
    // zeroed).
    c.mov(qword[r13 + GprOffset(dst_idx)], rax);
    return true;
}

/// Top-level IMUL dispatcher. Routes by Zydis's visible-operand
/// count, which is reliable: 1/2/3 maps cleanly to the three IMUL
/// encoding families.
bool EmitImul(const ZydisDecodedInstruction& insn, const ZydisDecodedOperand* ops, u64 next_rip,
              Xbyak::CodeGenerator& c) {
    switch (insn.operand_count_visible) {
    case 1:
        return EmitImul1Op(insn, ops, next_rip, c);
    case 2:
        return EmitImul2Op(insn, ops, next_rip, c);
    case 3:
        return EmitImul3Op(insn, ops, next_rip, c);
    default:
        return false;
    }
}

// =============================================================================
// Sign-extension family: CWDE / CDQE / CDQ / CQO.
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

bool EmitCwde(Xbyak::CodeGenerator& c) {
    // Load low 16 of guest RAX into host AX, sign-extend to EAX,
    // store qword (upper 32 of rax is naturally zero after CWDE).
    c.mov(ax, word[r13 + GprOffset(0)]);
    c.cwde();
    c.mov(qword[r13 + GprOffset(0)], rax);
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
    c.mov(rax, qword[r13 + GprOffset(5)]); // rax = old rbp

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

bool EmitAdcSbb64(const ZydisDecodedInstruction& insn, const ZydisDecodedOperand* ops,
                  Xbyak::CodeGenerator& c, AdcSbbKind kind) {
    if (insn.operand_width != 64)
        return false;
    if (ops[0].type != ZYDIS_OPERAND_TYPE_REGISTER)
        return false;
    const int dst_idx = ZydisGprToIndex(ops[0].reg.value);
    if (dst_idx < 0)
        return false;

    // Load src into rcx.
    if (ops[1].type == ZYDIS_OPERAND_TYPE_REGISTER) {
        const int src_idx = ZydisGprToIndex(ops[1].reg.value);
        if (src_idx < 0)
            return false;
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
    c.push(r8);
    c.popfq();

    switch (kind) {
    case AdcSbbKind::Adc:
        c.adc(rax, rcx);
        break;
    case AdcSbbKind::Sbb:
        c.sbb(rax, rcx);
        break;
    }

    c.pushfq();
    c.pop(r8);
    c.mov(qword[r13 + Offsets::Rflags], r8);

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
enum class NarrowArithKind { Add, Sub, Cmp, Test, And, Or, Xor };

bool EmitNarrowArith8(const ZydisDecodedInstruction& insn, const ZydisDecodedOperand* ops,
                      u64 next_rip, Xbyak::CodeGenerator& c, NarrowArithKind kind) {
    if (insn.operand_width != 8)
        return false;

    // ----- Memory destination/lhs -----
    //
    // Cmp/Test are read-only — no writeback. They're common as
    // "test byte ptr [reg+disp], imm8" boolean checks and similar.
    //
    // Or/And/Xor/Add/Sub with mem-dst + imm-src DO write back. The
    // typical pattern is `or byte[reg+disp], imm8` (set-flag idiom)
    // observed in glibc-style bitfield updates. The reg-src form
    // exists too but isn't observed yet — when seen, add `mov cl,
    // byte[r13+src_off]` and use the cl register variant of the op.
    if (ops[0].type == ZYDIS_OPERAND_TYPE_MEMORY) {
        const bool is_readonly = (kind == NarrowArithKind::Cmp || kind == NarrowArithKind::Test);
        const bool is_writeback_imm =
            (ops[1].type == ZYDIS_OPERAND_TYPE_IMMEDIATE) &&
            (kind == NarrowArithKind::Or || kind == NarrowArithKind::And ||
             kind == NarrowArithKind::Xor || kind == NarrowArithKind::Add ||
             kind == NarrowArithKind::Sub);
        if (!is_readonly && !is_writeback_imm) {
            return false;
        }
        if (!EmitEffectiveAddress(ops[0].mem, next_rip, c))
            return false;

        if (is_readonly) {
            // rdx = effective address. Load 8-bit lhs into al.
            c.mov(al, byte[rdx]);

            // Load rhs into cl. Mem-mem doesn't exist for cmp/test.
            if (ops[1].type == ZYDIS_OPERAND_TYPE_REGISTER) {
                const int src_off = ZydisGpr8ToByteOffset(ops[1].reg.value);
                if (src_off < 0)
                    return false;
                c.mov(cl, byte[r13 + src_off]);
            } else if (ops[1].type == ZYDIS_OPERAND_TYPE_IMMEDIATE) {
                c.mov(cl, static_cast<u8>(ops[1].imm.value.u & 0xFF));
            } else {
                return false;
            }

            // Flag round-trip through host CPU (so it computes correct
            // 8-bit-width flags). rdx is free now — address is no longer
            // needed since cmp/test don't write back.
            c.mov(rdx, qword[r13 + Offsets::Rflags]);
            c.push(rdx);
            c.popfq();

            switch (kind) {
            case NarrowArithKind::Cmp:
                c.cmp(al, cl);
                break;
            case NarrowArithKind::Test:
                c.test(al, cl);
                break;
            default:
                return false; // unreachable: filtered above
            }

            c.pushfq();
            c.pop(rdx);
            c.mov(qword[r13 + Offsets::Rflags], rdx);
            return true;
        }

        // Writeback path: read the byte, do op with imm, store back.
        // Stash EA in r10 so the rflags round-trip's use of rdx
        // doesn't clobber it before we store back.
        c.mov(r10, rdx);
        c.mov(al, byte[r10]);
        const u8 imm = static_cast<u8>(ops[1].imm.value.u & 0xFFu);

        c.mov(rdx, qword[r13 + Offsets::Rflags]);
        c.push(rdx);
        c.popfq();

        switch (kind) {
        case NarrowArithKind::Or:
            c.or_(al, imm);
            break;
        case NarrowArithKind::And:
            c.and_(al, imm);
            break;
        case NarrowArithKind::Xor:
            c.xor_(al, imm);
            break;
        case NarrowArithKind::Add:
            c.add(al, imm);
            break;
        case NarrowArithKind::Sub:
            c.sub(al, imm);
            break;
        default:
            return false; // unreachable: filtered above
        }

        c.pushfq();
        c.pop(rdx);
        c.mov(qword[r13 + Offsets::Rflags], rdx);

        c.mov(byte[r10], al);
        return true;
    }

    if (ops[0].type != ZYDIS_OPERAND_TYPE_REGISTER)
        return false;
    const int dst_off = ZydisGpr8ToByteOffset(ops[0].reg.value);
    if (dst_off < 0)
        return false;

    // Load dst byte into al. dst_off addresses the specific byte
    // within the parent qword slot — byte 0 for low-byte regs, byte
    // 1 for AH/BH/CH/DH. The store at the end uses the same offset,
    // so the high-byte case correctly updates only byte 1 of the
    // slot and leaves the surrounding bytes untouched.
    c.mov(al, byte[r13 + dst_off]);

    // Load src byte into cl. Three source forms:
    //   - reg: read from the guest GPR slot (supports high-byte too).
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
        if (src_off < 0)
            return false;
        c.mov(cl, byte[r13 + src_off]);
    } else if (ops[1].type == ZYDIS_OPERAND_TYPE_IMMEDIATE) {
        c.mov(cl, static_cast<u8>(ops[1].imm.value.u & 0xFF));
    } else if (ops[1].type == ZYDIS_OPERAND_TYPE_MEMORY) {
        if (!EmitEffectiveAddress(ops[1].mem, next_rip, c))
            return false;
        c.mov(cl, byte[rdx]);
    } else {
        return false;
    }

    // Round-trip flags (so host computes narrow-width flags).
    c.mov(rdx, qword[r13 + Offsets::Rflags]);
    c.push(rdx);
    c.popfq();

    switch (kind) {
    case NarrowArithKind::Add:
        c.add(al, cl);
        break;
    case NarrowArithKind::Sub:
        c.sub(al, cl);
        break;
    case NarrowArithKind::Cmp:
        c.cmp(al, cl);
        break; // sets flags, no write
    case NarrowArithKind::Test:
        c.test(al, cl);
        break; // (al & cl), flags only
    case NarrowArithKind::And:
        c.and_(al, cl);
        break;
    case NarrowArithKind::Or:
        c.or_(al, cl);
        break;
    case NarrowArithKind::Xor:
        c.xor_(al, cl);
        break;
    }

    c.pushfq();
    c.pop(rdx);
    c.mov(qword[r13 + Offsets::Rflags], rdx);

    // CMP and TEST discard the result — only the others write back.
    // Narrow store preserves upper 56 bits per x86-64 semantics. For
    // high-byte regs, dst_off points at byte 1, so the store updates
    // only byte 1 of the parent slot.
    if (kind != NarrowArithKind::Cmp && kind != NarrowArithKind::Test) {
        c.mov(byte[r13 + dst_off], al);
    }
    return true;
}

bool EmitNarrowArith16(const ZydisDecodedInstruction& insn, const ZydisDecodedOperand* ops,
                       u64 next_rip, Xbyak::CodeGenerator& c, NarrowArithKind kind) {
    if (insn.operand_width != 16)
        return false;
    if (ops[0].type != ZYDIS_OPERAND_TYPE_REGISTER)
        return false;
    const int dst_idx = ZydisGprToIndex(ops[0].reg.value);
    if (dst_idx < 0)
        return false;

    c.mov(ax, word[r13 + GprOffset(dst_idx)]);

    if (ops[1].type == ZYDIS_OPERAND_TYPE_REGISTER) {
        const int src_idx = ZydisGprToIndex(ops[1].reg.value);
        if (src_idx < 0)
            return false;
        c.mov(cx, word[r13 + GprOffset(src_idx)]);
    } else if (ops[1].type == ZYDIS_OPERAND_TYPE_IMMEDIATE) {
        c.mov(cx, static_cast<u16>(ops[1].imm.value.u & 0xFFFF));
    } else if (ops[1].type == ZYDIS_OPERAND_TYPE_MEMORY) {
        if (!EmitEffectiveAddress(ops[1].mem, next_rip, c))
            return false;
        c.mov(cx, word[rdx]);
    } else {
        return false;
    }

    c.mov(rdx, qword[r13 + Offsets::Rflags]);
    c.push(rdx);
    c.popfq();

    switch (kind) {
    case NarrowArithKind::Add:
        c.add(ax, cx);
        break;
    case NarrowArithKind::Sub:
        c.sub(ax, cx);
        break;
    case NarrowArithKind::Cmp:
        c.cmp(ax, cx);
        break;
    case NarrowArithKind::Test:
        c.test(ax, cx);
        break;
    case NarrowArithKind::And:
        c.and_(ax, cx);
        break;
    case NarrowArithKind::Or:
        c.or_(ax, cx);
        break;
    case NarrowArithKind::Xor:
        c.xor_(ax, cx);
        break;
    }

    c.pushfq();
    c.pop(rdx);
    c.mov(qword[r13 + Offsets::Rflags], rdx);

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
bool EmitNarrowArith32(const ZydisDecodedInstruction& insn, const ZydisDecodedOperand* ops,
                       u64 next_rip, Xbyak::CodeGenerator& c, NarrowArithKind kind) {
    if (insn.operand_width != 32)
        return false;

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
        if (!EmitEffectiveAddress(ops[0].mem, next_rip, c))
            return false;
        // For arithmetic kinds, stash address in r10 for writeback.
        const bool needs_writeback =
            (kind == NarrowArithKind::Add || kind == NarrowArithKind::Sub ||
             kind == NarrowArithKind::And || kind == NarrowArithKind::Or ||
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
            if (src_idx < 0)
                return false;
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
        c.push(rdx);
        c.popfq();

        switch (kind) {
        case NarrowArithKind::Cmp:
            c.cmp(eax, ecx);
            break;
        case NarrowArithKind::Test:
            c.test(eax, ecx);
            break;
        case NarrowArithKind::Add:
            c.add(eax, ecx);
            break;
        case NarrowArithKind::Sub:
            c.sub(eax, ecx);
            break;
        case NarrowArithKind::And:
            c.and_(eax, ecx);
            break;
        case NarrowArithKind::Or:
            c.or_(eax, ecx);
            break;
        case NarrowArithKind::Xor:
            c.xor_(eax, ecx);
            break;
        }

        c.pushfq();
        c.pop(rdx);
        c.mov(qword[r13 + Offsets::Rflags], rdx);

        // Store the result back. dword store leaves the upper 32
        // bits of the surrounding qword untouched, which is the
        // correct semantics: memory writes don't have the
        // zero-extension behavior of register writes.
        if (needs_writeback) {
            c.mov(dword[r10], eax);
        }
        return true;
    }

    if (ops[0].type != ZYDIS_OPERAND_TYPE_REGISTER)
        return false;
    const int dst_idx = ZydisGprToIndex(ops[0].reg.value);
    if (dst_idx < 0)
        return false;

    // Load dst's low 32 bits into eax. Reading 32-bit from a 64-bit
    // slot is fine — we're explicitly working at 32-bit width.
    c.mov(eax, dword[r13 + GprOffset(dst_idx)]);

    if (ops[1].type == ZYDIS_OPERAND_TYPE_REGISTER) {
        const int src_idx = ZydisGprToIndex(ops[1].reg.value);
        if (src_idx < 0)
            return false;
        c.mov(ecx, dword[r13 + GprOffset(src_idx)]);
    } else if (ops[1].type == ZYDIS_OPERAND_TYPE_IMMEDIATE) {
        // 32-bit forms with imm operands use either imm32 or imm8-sx;
        // Zydis hands us the sign-extended u64. Truncate to u32 for
        // the host 32-bit move; the bit pattern is preserved.
        c.mov(ecx, static_cast<u32>(ops[1].imm.value.u & 0xFFFFFFFFu));
    } else if (ops[1].type == ZYDIS_OPERAND_TYPE_MEMORY) {
        if (!EmitEffectiveAddress(ops[1].mem, next_rip, c))
            return false;
        c.mov(ecx, dword[rdx]);
    } else {
        return false;
    }

    // Round-trip guest rflags through host flags so the host CPU
    // computes correct 32-bit-width flags (CF/OF/SF/ZF/PF/AF).
    c.mov(rdx, qword[r13 + Offsets::Rflags]);
    c.push(rdx);
    c.popfq();

    switch (kind) {
    case NarrowArithKind::Add:
        c.add(eax, ecx);
        break;
    case NarrowArithKind::Sub:
        c.sub(eax, ecx);
        break;
    case NarrowArithKind::Cmp:
        c.cmp(eax, ecx);
        break;
    case NarrowArithKind::Test:
        c.test(eax, ecx);
        break;
    case NarrowArithKind::And:
        c.and_(eax, ecx);
        break;
    case NarrowArithKind::Or:
        c.or_(eax, ecx);
        break;
    case NarrowArithKind::Xor:
        c.xor_(eax, ecx);
        break;
    }

    c.pushfq();
    c.pop(rdx);
    c.mov(qword[r13 + Offsets::Rflags], rdx);

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

bool EmitMov8(const ZydisDecodedInstruction&, const ZydisDecodedOperand* ops, u64 next_rip,
              Xbyak::CodeGenerator& c) {
    const auto& dst = ops[0];
    const auto& src = ops[1];

    // Register destination.
    if (dst.type == ZYDIS_OPERAND_TYPE_REGISTER) {
        const int dst_idx = ZydisGprToIndex(dst.reg.value);
        if (dst_idx < 0)
            return false;

        if (src.type == ZYDIS_OPERAND_TYPE_REGISTER) {
            // mov r8, r8 — read low byte of src slot, write low byte
            // of dst slot. Upper bits of dst preserved by `byte` store.
            const int src_idx = ZydisGprToIndex(src.reg.value);
            if (src_idx < 0)
                return false;
            c.mov(al, byte[r13 + GprOffset(src_idx)]);
            c.mov(byte[r13 + GprOffset(dst_idx)], al);
            return true;
        }
        if (src.type == ZYDIS_OPERAND_TYPE_IMMEDIATE) {
            // mov r8, imm8 — store immediate byte into low byte of slot.
            c.mov(byte[r13 + GprOffset(dst_idx)], static_cast<u8>(src.imm.value.u & 0xFF));
            return true;
        }
        if (src.type == ZYDIS_OPERAND_TYPE_MEMORY) {
            // mov r8, byte[mem] — compute address (clobbers rax),
            // then load byte and write low byte of dst slot.
            // We use cl for the loaded byte because rcx is preserved
            // by EmitEffectiveAddress; al would be valid too but is
            // a scratch the helper actively uses.
            if (!EmitEffectiveAddress(src.mem, next_rip, c))
                return false;
            c.mov(cl, byte[rdx]);
            c.mov(byte[r13 + GprOffset(dst_idx)], cl);
            return true;
        }
        return false;
    }

    // Memory destination.
    if (dst.type == ZYDIS_OPERAND_TYPE_MEMORY) {
        if (src.type == ZYDIS_OPERAND_TYPE_REGISTER) {
            // mov byte[mem], r8 — compute address (rdx), then read
            // src low byte into cl (rcx preserved through
            // EmitEffectiveAddress), then store.
            const int src_idx = ZydisGprToIndex(src.reg.value);
            if (src_idx < 0)
                return false;
            if (!EmitEffectiveAddress(dst.mem, next_rip, c))
                return false;
            c.mov(cl, byte[r13 + GprOffset(src_idx)]);
            c.mov(byte[rdx], cl);
            return true;
        }
        if (src.type == ZYDIS_OPERAND_TYPE_IMMEDIATE) {
            // mov byte[mem], imm8.
            if (!EmitEffectiveAddress(dst.mem, next_rip, c))
                return false;
            c.mov(byte[rdx], static_cast<u8>(src.imm.value.u & 0xFF));
            return true;
        }
        return false;
    }

    return false;
}

bool EmitMov16(const ZydisDecodedInstruction&, const ZydisDecodedOperand* ops, u64 next_rip,
               Xbyak::CodeGenerator& c) {
    const auto& dst = ops[0];
    const auto& src = ops[1];

    if (dst.type == ZYDIS_OPERAND_TYPE_REGISTER) {
        const int dst_idx = ZydisGprToIndex(dst.reg.value);
        if (dst_idx < 0)
            return false;

        if (src.type == ZYDIS_OPERAND_TYPE_REGISTER) {
            const int src_idx = ZydisGprToIndex(src.reg.value);
            if (src_idx < 0)
                return false;
            c.mov(ax, word[r13 + GprOffset(src_idx)]);
            c.mov(word[r13 + GprOffset(dst_idx)], ax);
            return true;
        }
        if (src.type == ZYDIS_OPERAND_TYPE_IMMEDIATE) {
            c.mov(word[r13 + GprOffset(dst_idx)], static_cast<u16>(src.imm.value.u & 0xFFFF));
            return true;
        }
        if (src.type == ZYDIS_OPERAND_TYPE_MEMORY) {
            if (!EmitEffectiveAddress(src.mem, next_rip, c))
                return false;
            c.mov(cx, word[rdx]);
            c.mov(word[r13 + GprOffset(dst_idx)], cx);
            return true;
        }
        return false;
    }

    if (dst.type == ZYDIS_OPERAND_TYPE_MEMORY) {
        if (src.type == ZYDIS_OPERAND_TYPE_REGISTER) {
            const int src_idx = ZydisGprToIndex(src.reg.value);
            if (src_idx < 0)
                return false;
            if (!EmitEffectiveAddress(dst.mem, next_rip, c))
                return false;
            c.mov(cx, word[r13 + GprOffset(src_idx)]);
            c.mov(word[rdx], cx);
            return true;
        }
        if (src.type == ZYDIS_OPERAND_TYPE_IMMEDIATE) {
            if (!EmitEffectiveAddress(dst.mem, next_rip, c))
                return false;
            c.mov(word[rdx], static_cast<u16>(src.imm.value.u & 0xFFFF));
            return true;
        }
        return false;
    }

    return false;
}

/// PUSH r64 — pushes a register onto the guest stack.
/// Semantics: guest_rsp -= 8; *guest_rsp = reg.
bool EmitPush(const ZydisDecodedInstruction& insn, const ZydisDecodedOperand* ops,
              Xbyak::CodeGenerator& c) {
    if (insn.operand_width != 64)
        return false;
    const auto& src = ops[0];
    if (src.type != ZYDIS_OPERAND_TYPE_REGISTER)
        return false;
    const int src_idx = ZydisGprToIndex(src.reg.value);
    if (src_idx < 0)
        return false;

    // rax = src value, rdx = guest_rsp, decrement, store.
    c.mov(rax, qword[r13 + GprOffset(src_idx)]);
    c.mov(rdx, qword[r13 + GprOffset(kGuestRspIdx)]);
    c.sub(rdx, 8);
    c.mov(qword[rdx], rax);                           // write to guest stack
    c.mov(qword[r13 + GprOffset(kGuestRspIdx)], rdx); // update RSP
    return true;
}

/// POP r64 — pops top of guest stack into a register.
/// Semantics: reg = *guest_rsp; guest_rsp += 8.
bool EmitPop(const ZydisDecodedInstruction& insn, const ZydisDecodedOperand* ops,
             Xbyak::CodeGenerator& c) {
    if (insn.operand_width != 64)
        return false;
    const auto& dst = ops[0];
    if (dst.type != ZYDIS_OPERAND_TYPE_REGISTER)
        return false;
    const int dst_idx = ZydisGprToIndex(dst.reg.value);
    if (dst_idx < 0)
        return false;

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
///   5. Jump to gateway exit stub via r14. The test harness then
///      inspects state.rip to see where the guest returned to.
///
/// Note: rsp_guest is GPR[4] (RSP per AMD64 ABI).
bool EmitRet(const ZydisDecodedInstruction& insn, const ZydisDecodedOperand* /*ops*/,
             Xbyak::CodeGenerator& c) {
    if (insn.mnemonic != ZYDIS_MNEMONIC_RET)
        return false;
    // Only the no-immediate form. RET imm16 (0xC2) shows up with
    // operand_count_visible > 0.
    if (insn.operand_count_visible != 0)
        return false;

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
    case ZYDIS_MNEMONIC_JZ: // JE / JZ: ZF=1
        test_bit(ZF);
        break;
    case ZYDIS_MNEMONIC_JNZ: // JNE / JNZ: ZF=0
        test_not_bit(ZF);
        break;

    // Sign-based: SF
    case ZYDIS_MNEMONIC_JS: // JS: SF=1
        test_bit(SF);
        break;
    case ZYDIS_MNEMONIC_JNS: // JNS: SF=0
        test_not_bit(SF);
        break;

    // Overflow: OF
    case ZYDIS_MNEMONIC_JO: // JO: OF=1
        test_bit(OF);
        break;
    case ZYDIS_MNEMONIC_JNO: // JNO: OF=0
        test_not_bit(OF);
        break;

    // Parity: PF
    case ZYDIS_MNEMONIC_JP: // JP / JPE: PF=1
        test_bit(PF);
        break;
    case ZYDIS_MNEMONIC_JNP: // JNP / JPO: PF=0
        test_not_bit(PF);
        break;

    // Unsigned comparison: CF, ZF
    case ZYDIS_MNEMONIC_JB: // JB / JC / JNAE: CF=1
        test_bit(CF);
        break;
    case ZYDIS_MNEMONIC_JNB: // JNB / JNC / JAE: CF=0
        test_not_bit(CF);
        break;
    case ZYDIS_MNEMONIC_JBE: { // JBE / JNA: CF=1 OR ZF=1
        c.mov(rcx, rax);
        c.and_(rcx, CF | ZF);
        c.setnz(cl);
        c.movzx(rcx, cl);
        break;
    }
    case ZYDIS_MNEMONIC_JNBE: { // JNBE / JA: CF=0 AND ZF=0
        c.mov(rcx, rax);
        c.and_(rcx, CF | ZF);
        c.setz(cl);
        c.movzx(rcx, cl);
        break;
    }

    // Signed comparison: SF, OF, ZF
    case ZYDIS_MNEMONIC_JL: { // JL / JNGE: SF != OF
        // rcx = (SF >> 7) XOR (OF >> 11), both in low bit.
        c.mov(rcx, rax);
        c.shr(rcx, 7); // SF -> bit 0
        c.mov(rdx, rax);
        c.shr(rdx, 11); // OF -> bit 0
        c.xor_(rcx, rdx);
        c.and_(rcx, 1);
        break;
    }
    case ZYDIS_MNEMONIC_JNL: { // JNL / JGE: SF == OF
        c.mov(rcx, rax);
        c.shr(rcx, 7);
        c.mov(rdx, rax);
        c.shr(rdx, 11);
        c.xor_(rcx, rdx);
        c.not_(rcx);
        c.and_(rcx, 1);
        break;
    }
    case ZYDIS_MNEMONIC_JLE: { // JLE / JNG: ZF=1 OR SF != OF
        // First (SF != OF) into r8.
        c.mov(r8, rax);
        c.shr(r8, 7);
        c.mov(rcx, rax);
        c.shr(rcx, 11);
        c.xor_(r8, rcx);
        c.and_(r8, 1);
        // Then ZF into rcx.
        c.mov(rcx, rax);
        c.shr(rcx, 6);
        c.and_(rcx, 1);
        // OR them.
        c.or_(rcx, r8);
        break;
    }
    case ZYDIS_MNEMONIC_JNLE: { // JNLE / JG: ZF=0 AND SF == OF
        c.mov(r8, rax);
        c.shr(r8, 7);
        c.mov(rcx, rax);
        c.shr(rcx, 11);
        c.xor_(r8, rcx);
        c.and_(r8, 1);
        // r8 = (SF != OF). We want NOT(ZF=1 OR (SF!=OF)).
        c.mov(rcx, rax);
        c.shr(rcx, 6);
        c.and_(rcx, 1);
        c.or_(rcx, r8);
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
bool EmitJcc(const ZydisDecodedInstruction& insn, const ZydisDecodedOperand* ops, u64 next_rip,
             Xbyak::CodeGenerator& c) {
    // Zydis emits the absolute target for near jumps in
    // ops[0].imm.is_relative + .value.s. The decoder normalizes to
    // an absolute address when ZYDIS_DECODER_FLAG_NORMALIZED is set;
    // here we calculate manually: target = next_rip + imm (relative)
    // or just .value.s (absolute).
    if (ops[0].type != ZYDIS_OPERAND_TYPE_IMMEDIATE) {
        return false; // indirect Jcc isn't a real x86 form anyway
    }
    const u64 target = ops[0].imm.is_relative
                           ? static_cast<u64>(static_cast<s64>(next_rip) + ops[0].imm.value.s)
                           : static_cast<u64>(ops[0].imm.value.s);

    // Compute condition (1 or 0) into rcx. Bails if mnemonic isn't
    // a Jcc we recognize.
    if (!EmitJccCondition(insn.mnemonic, c))
        return false;

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
bool EmitJmp(const ZydisDecodedInstruction& insn, const ZydisDecodedOperand* ops, u64 next_rip,
             Xbyak::CodeGenerator& c) {
    if (insn.mnemonic != ZYDIS_MNEMONIC_JMP)
        return false;

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
        if (reg_idx < 0)
            return false;
        c.mov(rax, qword[r13 + GprOffset(reg_idx)]);
        c.mov(qword[r13 + Offsets::Rip], rax);
    } else if (ops[0].type == ZYDIS_OPERAND_TYPE_MEMORY) {
        // Indirect through memory: target = *(effective address).
        // EmitEffectiveAddress puts the address in rdx; we then load
        // the 8-byte target from [rdx] into rax.
        if (!EmitEffectiveAddress(ops[0].mem, next_rip, c))
            return false;
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
bool EmitCall(const ZydisDecodedInstruction& insn, const ZydisDecodedOperand* ops, u64 next_rip,
              Xbyak::CodeGenerator& c) {
    if (insn.mnemonic != ZYDIS_MNEMONIC_CALL)
        return false;

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
        if (reg_idx < 0)
            return false;
        c.mov(rax, qword[r13 + GprOffset(reg_idx)]);
    } else if (ops[0].type == ZYDIS_OPERAND_TYPE_MEMORY) {
        if (!EmitEffectiveAddress(ops[0].mem, next_rip, c))
            return false;
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
    return static_cast<u32>(offsetof(GuestState, ymm) + static_cast<size_t>(lane_idx) * 32u +
                            static_cast<size_t>(chunk) * 8u);
}

/// VMOVUPS / VMOVDQU — three forms: reg ← mem, mem ← reg, reg ← reg.
/// Both 128-bit (xmm) and 256-bit (ymm).
///
/// These two mnemonics differ only in float-vs-int operand-type
/// hinting on the host CPU; the actual bits moved and the upper-zero
/// behavior are identical, and we never differentiate by type since
/// nothing here observes float-vs-int semantics. One emitter covers
/// both.
///
/// The vector size comes from `ops[0].size` (in bits): 128 or 256.
/// We deliberately ignore alignment (the U in MOVUPS/MOVDQU =
/// Unaligned), which suits 64-bit-granular GPR-relayed moves.
bool EmitVmovups(const ZydisDecodedInstruction& insn, const ZydisDecodedOperand* ops, u64 next_rip,
                 Xbyak::CodeGenerator& c) {
    if (insn.mnemonic != ZYDIS_MNEMONIC_VMOVUPS && insn.mnemonic != ZYDIS_MNEMONIC_VMOVDQU &&
        insn.mnemonic != ZYDIS_MNEMONIC_VMOVDQA && insn.mnemonic != ZYDIS_MNEMONIC_VMOVAPS) {
        // VMOVDQA differs from VMOVDQU only in that it #GPs on
        // misaligned addresses on real hardware. We do all transfers
        // via 64-bit GPR chunks anyway, so alignment is irrelevant
        // to our emitted code; treating VMOVDQA as VMOVDQU is bitwise
        // identical for aligned inputs (which is what the guest is
        // expected to give us — it'd crash on its own hardware if it
        // weren't aligned).
        //
        // VMOVAPS is similarly the aligned packed-float variant of
        // VMOVUPS. Float-vs-int operand typing is irrelevant for
        // GPR-relayed bitwise moves, and the alignment requirement
        // doesn't matter for the same reason as VMOVDQA.
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
    if (vec_bits != 128 && vec_bits != 256)
        return false;
    const int chunks = vec_bits / 64;

    // reg ← mem
    if (ops[0].type == ZYDIS_OPERAND_TYPE_REGISTER && ops[1].type == ZYDIS_OPERAND_TYPE_MEMORY) {
        const int dst_idx = ZydisVecToIndex(ops[0].reg.value);
        if (dst_idx < 0)
            return false;
        if (!EmitEffectiveAddress(ops[1].mem, next_rip, c))
            return false;
        // rdx = src address; we use rax as the transfer register.
        for (int i = 0; i < chunks; ++i) {
            c.mov(rax, qword[rdx + i * 8]);
            c.mov(qword[r13 + YmmChunkOffset(dst_idx, i)], rax);
        }
        // VEX 128-bit form zeros bits 255:128 of the destination YMM.
        if (vec_bits == 128) {
            c.xor_(rax, rax);
            c.mov(qword[r13 + YmmChunkOffset(dst_idx, 2)], rax);
            c.mov(qword[r13 + YmmChunkOffset(dst_idx, 3)], rax);
        }
        return true;
    }

    // mem ← reg
    if (ops[0].type == ZYDIS_OPERAND_TYPE_MEMORY && ops[1].type == ZYDIS_OPERAND_TYPE_REGISTER) {
        const int src_idx = ZydisVecToIndex(ops[1].reg.value);
        if (src_idx < 0)
            return false;
        if (!EmitEffectiveAddress(ops[0].mem, next_rip, c))
            return false;
        // rdx = dst address; rax is the transfer register.
        for (int i = 0; i < chunks; ++i) {
            c.mov(rax, qword[r13 + YmmChunkOffset(src_idx, i)]);
            c.mov(qword[rdx + i * 8], rax);
        }
        return true;
    }

    // reg ← reg
    if (ops[0].type == ZYDIS_OPERAND_TYPE_REGISTER && ops[1].type == ZYDIS_OPERAND_TYPE_REGISTER) {
        const int dst_idx = ZydisVecToIndex(ops[0].reg.value);
        const int src_idx = ZydisVecToIndex(ops[1].reg.value);
        if (dst_idx < 0 || src_idx < 0)
            return false;
        for (int i = 0; i < chunks; ++i) {
            c.mov(rax, qword[r13 + YmmChunkOffset(src_idx, i)]);
            c.mov(qword[r13 + YmmChunkOffset(dst_idx, i)], rax);
        }
        if (vec_bits == 128) {
            c.xor_(rax, rax);
            c.mov(qword[r13 + YmmChunkOffset(dst_idx, 2)], rax);
            c.mov(qword[r13 + YmmChunkOffset(dst_idx, 3)], rax);
        }
        return true;
    }

    return false;
}

/// VMOVD — 32-bit move between a GPR and the low 32 bits of an XMM
/// register. Two directions, both AVX-encoded (VEX prefix, so the
/// XMM-destination form also zeros bits 255:32 of the destination
/// YMM, which is critical for correctness — naively copying just
/// 32 bits would leave the upper YMM bits with stale data).
///
/// Encodings handled:
///   - `vmovd r32, xmm`   : GPR ← low 32 of XMM
///   - `vmovd xmm, r32`   : low 32 of XMM ← GPR (rest of YMM zeroed)
///
/// MOVD r/m32 with memory operands isn't included yet; add when
/// observed. Same for VMOVQ (64-bit variant).
bool EmitVmovd(const ZydisDecodedInstruction& insn, const ZydisDecodedOperand* ops,
               Xbyak::CodeGenerator& c) {
    if (insn.mnemonic != ZYDIS_MNEMONIC_VMOVD)
        return false;
    if (ops[0].type != ZYDIS_OPERAND_TYPE_REGISTER || ops[1].type != ZYDIS_OPERAND_TYPE_REGISTER) {
        return false;
    }

    // Decide direction by sniffing whether op[0] is a GPR or a vec.
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
        // Write low 32 bits; zero the rest of chunk 0 explicitly,
        // then zero chunks 1..3. Going through rax (32-bit load
        // sets bits 63:32 of rax to 0, then qword store) covers
        // chunk 0 cleanly in one step.
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

/// VMOVQ — 64-bit move between an XMM register and either another
/// XMM register or a GPR. Three reg-reg directions handled here; the
/// mem-source/dest forms (xmm ↔ m64) can be added when seen.
///
///   xmm ← xmm: copies low 64 of src to low 64 of dst, zeroes the
///              rest of the dst YMM (bits 255:64).
///   xmm ← r64: copies r64 to dst low 64, zeroes bits 255:64 of YMM.
///   r64 ← xmm: copies xmm low 64 to r64 (full 64-bit write, no
///              zero-extend needed since it IS the full GPR).
///
/// The user log showed reg,reg with length=5 — that's the 3-byte-
/// VEX form, used for the GPR↔XMM variants. The xmm←xmm form is 4
/// bytes (2-byte VEX). All three variants share this emitter and
/// route by sniffing the operand register classes via the same
/// trick VMOVD uses.
bool EmitVmovq(const ZydisDecodedInstruction& insn, const ZydisDecodedOperand* ops,
               Xbyak::CodeGenerator& c) {
    if (insn.mnemonic != ZYDIS_MNEMONIC_VMOVQ)
        return false;
    if (ops[0].type != ZYDIS_OPERAND_TYPE_REGISTER || ops[1].type != ZYDIS_OPERAND_TYPE_REGISTER)
        return false;

    const int dst_gpr = ZydisGprToIndex(ops[0].reg.value);
    const int src_gpr = ZydisGprToIndex(ops[1].reg.value);
    const int dst_vec = ZydisVecToIndex(ops[0].reg.value);
    const int src_vec = ZydisVecToIndex(ops[1].reg.value);

    // GPR ← XMM low 64 bits. Full 64-bit write to the guest GPR;
    // upper bits of the GPR are overwritten (this is the qword
    // semantic, no zero-extension question).
    if (dst_gpr >= 0 && src_vec >= 0) {
        c.mov(rax, qword[r13 + YmmChunkOffset(src_vec, 0)]);
        c.mov(qword[r13 + GprOffset(dst_gpr)], rax);
        return true;
    }

    // XMM ← GPR. Copies the full 64-bit GPR into the dst xmm's
    // low 64, zeroes bits 255:64 of the dst YMM.
    if (dst_vec >= 0 && src_gpr >= 0) {
        c.mov(rax, qword[r13 + GprOffset(src_gpr)]);
        c.mov(qword[r13 + YmmChunkOffset(dst_vec, 0)], rax);
        c.xor_(rax, rax);
        c.mov(qword[r13 + YmmChunkOffset(dst_vec, 1)], rax);
        c.mov(qword[r13 + YmmChunkOffset(dst_vec, 2)], rax);
        c.mov(qword[r13 + YmmChunkOffset(dst_vec, 3)], rax);
        return true;
    }

    // XMM ← XMM. Same shape as the GPR-src case, just sourcing
    // from a YMM chunk instead of a GPR slot.
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
/// Both are bitwise XOR; they differ only in whether they're classed
/// as float (VXORPS) or int (VPXOR), which we don't care about
/// because no flags are written either way.
///
/// Supported operand shape: reg, reg, reg (the common case in the
/// game's prologue zeroing idiom: `vxorps xmm0, xmm0, xmm0`).
/// Reg + reg + mem can be added when we see it.
bool EmitVecBitXor(const ZydisDecodedInstruction& insn, const ZydisDecodedOperand* ops,
                   Xbyak::CodeGenerator& c) {
    if (insn.mnemonic != ZYDIS_MNEMONIC_VXORPS && insn.mnemonic != ZYDIS_MNEMONIC_VPXOR) {
        return false;
    }
    // Strictly 3-operand reg form for this initial cut.
    if (ops[0].type != ZYDIS_OPERAND_TYPE_REGISTER)
        return false;
    if (ops[1].type != ZYDIS_OPERAND_TYPE_REGISTER)
        return false;
    if (ops[2].type != ZYDIS_OPERAND_TYPE_REGISTER)
        return false;

    const int dst_idx = ZydisVecToIndex(ops[0].reg.value);
    const int src1_idx = ZydisVecToIndex(ops[1].reg.value);
    const int src2_idx = ZydisVecToIndex(ops[2].reg.value);
    if (dst_idx < 0 || src1_idx < 0 || src2_idx < 0)
        return false;

    const int vec_bits = ops[0].size;
    if (vec_bits != 128 && vec_bits != 256)
        return false;
    const int chunks = vec_bits / 64;

    // dst[i] = src1[i] XOR src2[i] for each 64-bit chunk.
    for (int i = 0; i < chunks; ++i) {
        c.mov(rax, qword[r13 + YmmChunkOffset(src1_idx, i)]);
        c.xor_(rax, qword[r13 + YmmChunkOffset(src2_idx, i)]);
        c.mov(qword[r13 + YmmChunkOffset(dst_idx, i)], rax);
    }
    // 128-bit VEX form zeros bits 255:128 of the destination YMM.
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
bool EmitVpshufb(const ZydisDecodedInstruction& insn, const ZydisDecodedOperand* ops,
                 Xbyak::CodeGenerator& c) {
    if (insn.mnemonic != ZYDIS_MNEMONIC_VPSHUFB)
        return false;

    int dst_idx, src_idx, mask_idx;
    if (insn.operand_count_visible == 3) {
        if (ops[0].type != ZYDIS_OPERAND_TYPE_REGISTER ||
            ops[1].type != ZYDIS_OPERAND_TYPE_REGISTER ||
            ops[2].type != ZYDIS_OPERAND_TYPE_REGISTER)
            return false;
        dst_idx = ZydisVecToIndex(ops[0].reg.value);
        src_idx = ZydisVecToIndex(ops[1].reg.value);
        mask_idx = ZydisVecToIndex(ops[2].reg.value);
    } else if (insn.operand_count_visible == 2) {
        if (ops[0].type != ZYDIS_OPERAND_TYPE_REGISTER ||
            ops[1].type != ZYDIS_OPERAND_TYPE_REGISTER)
            return false;
        dst_idx = ZydisVecToIndex(ops[0].reg.value);
        src_idx = dst_idx;
        mask_idx = ZydisVecToIndex(ops[1].reg.value);
    } else {
        return false;
    }
    if (dst_idx < 0 || src_idx < 0 || mask_idx < 0)
        return false;

    const int vec_bits = ops[0].size;
    if (vec_bits != 128 && vec_bits != 256)
        return false;

    // Process each 128-bit lane. For the 128-bit form there's only
    // one lane (chunks 0..1); for the 256-bit AVX2 form there's a
    // second lane (chunks 2..3) handled identically — VPSHUFB does
    // not cross 128-bit boundaries.
    for (int lane = 0; lane < vec_bits / 128; ++lane) {
        const u32 src_off = YmmChunkOffset(src_idx, lane * 2);
        const u32 mask_off = YmmChunkOffset(mask_idx, lane * 2);
        const u32 dst_off = YmmChunkOffset(dst_idx, lane * 2);
        c.vmovdqu(xmm0, ptr[r13 + src_off]);
        c.vmovdqu(xmm1, ptr[r13 + mask_off]);
        c.vpshufb(xmm0, xmm0, xmm1);
        c.vmovdqu(ptr[r13 + dst_off], xmm0);
    }
    // 128-bit VEX form zeros bits 255:128 of the destination YMM.
    if (vec_bits == 128) {
        c.xor_(rax, rax);
        c.mov(qword[r13 + YmmChunkOffset(dst_idx, 2)], rax);
        c.mov(qword[r13 + YmmChunkOffset(dst_idx, 3)], rax);
    }
    return true;
}

/// VMOVNTDQ m128/m256, xmm/ymm — non-temporal aligned store from a
/// vector register to memory. The "non-temporal" part is a cache
/// hint: it tells the host CPU to bypass the cache hierarchy on the
/// store, which matters for streaming-write workloads (memset-like
/// loops over large buffers).
///
/// For correctness we can ignore the cache hint and emit a regular
/// store — the guest will get the same data at the same address.
/// We may revisit later if profiling shows a streaming-store hot
/// path benefits from preserving the non-temporal semantic on the
/// host (would require host VMOVNTDQ to the guest memory address;
/// the address itself isn't required to be aligned for our store
/// since we're using non-NT moves at the byte level via GPR relay).
///
/// Mem-dst form only; reg-dst would be VMOVDQA reading from a
/// non-temporal source, which isn't this opcode.
bool EmitVmovntdq(const ZydisDecodedInstruction& insn, const ZydisDecodedOperand* ops, u64 next_rip,
                  Xbyak::CodeGenerator& c) {
    if (insn.mnemonic != ZYDIS_MNEMONIC_VMOVNTDQ)
        return false;
    if (ops[0].type != ZYDIS_OPERAND_TYPE_MEMORY || ops[1].type != ZYDIS_OPERAND_TYPE_REGISTER) {
        return false;
    }

    const int src_idx = ZydisVecToIndex(ops[1].reg.value);
    if (src_idx < 0)
        return false;

    // Vector size from the source register operand.
    const int vec_bits = ops[1].size;
    if (vec_bits != 128 && vec_bits != 256)
        return false;
    const int chunks = vec_bits / 64;

    if (!EmitEffectiveAddress(ops[0].mem, next_rip, c))
        return false;
    // rdx = dst address; rax is the GPR-relay transfer register.
    // Identical shape to EmitVmovups mem-dst, minus any
    // upper-YMM-zeroing (which doesn't apply for mem-dst forms).
    for (int i = 0; i < chunks; ++i) {
        c.mov(rax, qword[r13 + YmmChunkOffset(src_idx, i)]);
        c.mov(qword[rdx + i * 8], rax);
    }
    return true;
}

/// ANDN r64, r64, r64 — BMI1 three-operand AND-NOT.
///   dst = ~src1 & src2
///
/// Flags: writes SF, ZF from the result; clears OF, CF; PF undefined.
/// We use the host ANDN instruction (BMI1 is universally present on
/// modern x86-64 hosts), and round-trip rflags so the host's exact
/// flag semantics propagate to the guest's rflags shadow.
///
/// 32-bit ANDN exists too but hasn't been observed; add when seen.
bool EmitAndn(const ZydisDecodedInstruction& insn, const ZydisDecodedOperand* ops,
              Xbyak::CodeGenerator& c) {
    if (insn.mnemonic != ZYDIS_MNEMONIC_ANDN)
        return false;
    if (insn.operand_width != 64 && insn.operand_width != 32)
        return false;
    if (insn.operand_count_visible != 3)
        return false;
    if (ops[0].type != ZYDIS_OPERAND_TYPE_REGISTER || ops[1].type != ZYDIS_OPERAND_TYPE_REGISTER ||
        ops[2].type != ZYDIS_OPERAND_TYPE_REGISTER) {
        return false;
    }

    const int dst_idx = ZydisGprToIndex(ops[0].reg.value);
    const int src1_idx = ZydisGprToIndex(ops[1].reg.value);
    const int src2_idx = ZydisGprToIndex(ops[2].reg.value);
    if (dst_idx < 0 || src1_idx < 0 || src2_idx < 0)
        return false;

    const bool is32 = (insn.operand_width == 32);

    // Load operands into scratch registers we don't need for the
    // flags round-trip. r9/r10 are safe (r10 isn't used by the
    // round-trip itself). 32-bit form uses the d-suffixed names,
    // which zero-extend bits 63:32 into r9/r10 as a side effect —
    // exactly what we want for the 32-bit host ANDN.
    if (is32) {
        c.mov(r9d, dword[r13 + GprOffset(src1_idx)]);
        c.mov(r10d, dword[r13 + GprOffset(src2_idx)]);
    } else {
        c.mov(r9, qword[r13 + GprOffset(src1_idx)]);
        c.mov(r10, qword[r13 + GprOffset(src2_idx)]);
    }

    c.mov(r8, qword[r13 + Offsets::Rflags]);
    c.push(r8);
    c.popfq();

    // Host ANDN. 32-bit form writes 32-bit result and zero-extends
    // bits 63:32 of rax via the standard register-write rule, so
    // the qword storeback below captures the correct guest value.
    if (is32)
        c.andn(eax, r9d, r10d);
    else
        c.andn(rax, r9, r10);

    c.pushfq();
    c.pop(r8);
    c.mov(qword[r13 + Offsets::Rflags], r8);

    c.mov(qword[r13 + GprOffset(dst_idx)], rax);
    return true;
}

/// CMPXCHG [mem], r — atomic compare-and-exchange.
///
/// Semantics: compares the accumulator (RAX/EAX/AX/AL by width) with
/// [mem]. If equal: stores `r` into [mem]. If not equal: loads [mem]
/// into the accumulator. ZF reflects the comparison (1 if equal,
/// 0 otherwise).
///
/// libc primitives (mutexes, atomic counters) use CMPXCHG as the
/// load-locked / store-conditional building block — so the LOCK
/// prefix matters and we issue a real host LOCK CMPXCHG.
///
/// 32-bit mem-dst form (observed at libc 0x807a084ae) is implemented
/// here. 64-bit and 8-bit variants will follow the same shape when
/// observed. Note: CMPXCHG with a register destination (rather than
/// memory) exists but is rare; we don't handle it yet.
bool EmitCmpxchg(const ZydisDecodedInstruction& insn, const ZydisDecodedOperand* ops, u64 next_rip,
                 Xbyak::CodeGenerator& c) {
    if (insn.mnemonic != ZYDIS_MNEMONIC_CMPXCHG)
        return false;
    if (insn.operand_width != 32 && insn.operand_width != 64)
        return false;
    if (ops[0].type != ZYDIS_OPERAND_TYPE_MEMORY)
        return false;
    if (ops[1].type != ZYDIS_OPERAND_TYPE_REGISTER)
        return false;

    const int src_idx = ZydisGprToIndex(ops[1].reg.value);
    if (src_idx < 0)
        return false;

    // Compute effective address FIRST (uses rdx and clobbers rax for
    // index-scale ops), then stash to r10 so subsequent register
    // setup doesn't trample it.
    if (!EmitEffectiveAddress(ops[0].mem, next_rip, c))
        return false;
    c.mov(r10, rdx);

    // Set up host accumulator and source register. Guest RAX is slot 0.
    if (insn.operand_width == 32) {
        c.mov(eax, dword[r13 + GprOffset(0)]);
        c.mov(ecx, dword[r13 + GprOffset(src_idx)]);
    } else {
        c.mov(rax, qword[r13 + GprOffset(0)]);
        c.mov(rcx, qword[r13 + GprOffset(src_idx)]);
    }

    // Round-trip rflags through the host so ZF (and the rest) come
    // out matching what the host CMPXCHG produces.
    c.mov(r8, qword[r13 + Offsets::Rflags]);
    c.push(r8);
    c.popfq();

    // Atomic compare-exchange. The host instruction:
    //   - compares EAX/RAX against [r10]
    //   - on equal:   stores ECX/RCX into [r10], EAX unchanged
    //   - on unequal: loads [r10] into EAX/RAX, [r10] unchanged
    //   - in all cases: writes ZF accordingly.
    c.lock();
    if (insn.operand_width == 32) {
        c.cmpxchg(dword[r10], ecx);
    } else {
        c.cmpxchg(qword[r10], rcx);
    }

    c.pushfq();
    c.pop(r8);
    c.mov(qword[r13 + Offsets::Rflags], r8);

    // Write back guest RAX (which is now either the original value
    // or [mem]'s previous contents). For 32-bit form the host
    // already zero-extended bits 63:32 of rax via the 32-bit write,
    // so qword storeback is canonical.
    c.mov(qword[r13 + GprOffset(0)], rax);
    return true;
}

/// XADD [mem], reg — atomic "exchange and add".
///   tmp     := [mem]
///   [mem]   := [mem] + reg
///   reg     := tmp        (i.e. reg gets the OLD memory value)
/// All standard arithmetic flags (CF/OF/SF/ZF/AF/PF) are set from
/// the addition. Always emitted with LOCK; matches CMPXCHG's
/// conservative approach to atomicity. The non-atomic variant has
/// identical observable behavior in single-threaded code, so adding
/// LOCK is at worst a perf cost — never a correctness change.
///
/// 32-bit width observed first (at libkernel 0x807219de0 inside a
/// reference-count increment). 64-bit gets the same shape with
/// the obvious type substitutions.
bool EmitXadd(const ZydisDecodedInstruction& insn, const ZydisDecodedOperand* ops, u64 next_rip,
              Xbyak::CodeGenerator& c) {
    if (insn.mnemonic != ZYDIS_MNEMONIC_XADD)
        return false;
    if (insn.operand_width != 32 && insn.operand_width != 64)
        return false;
    if (ops[0].type != ZYDIS_OPERAND_TYPE_MEMORY)
        return false;
    if (ops[1].type != ZYDIS_OPERAND_TYPE_REGISTER)
        return false;

    const int src_idx = ZydisGprToIndex(ops[1].reg.value);
    if (src_idx < 0)
        return false;

    // Compute EA into rdx, stash to r10 to survive the rflags
    // round-trip (which uses rdx as the intermediate).
    if (!EmitEffectiveAddress(ops[0].mem, next_rip, c))
        return false;
    c.mov(r10, rdx);

    // Load source register into ecx/rcx. Host XADD will overwrite
    // it with the old memory value after the operation.
    if (insn.operand_width == 32) {
        c.mov(ecx, dword[r13 + GprOffset(src_idx)]);
    } else {
        c.mov(rcx, qword[r13 + GprOffset(src_idx)]);
    }

    // Set host rflags from guest before the op so the captured flags
    // reflect the addition's effect on a clean baseline.
    c.mov(r8, qword[r13 + Offsets::Rflags]);
    c.push(r8);
    c.popfq();

    c.lock();
    if (insn.operand_width == 32) {
        c.xadd(dword[r10], ecx);
    } else {
        c.xadd(qword[r10], rcx);
    }

    c.pushfq();
    c.pop(r8);
    c.mov(qword[r13 + Offsets::Rflags], r8);

    // Write the old memory value (now in ecx/rcx) back to the guest
    // register slot. 32-bit XADD writes to ecx, which zero-extends
    // rcx — so qword storeback is canonical.
    c.mov(qword[r13 + GprOffset(src_idx)], rcx);
    return true;
}

/// Map a SETcc mnemonic to the corresponding Jcc mnemonic. The
/// condition encoding is identical across the Jcc/CMOVcc/SETcc
/// families, so we reuse the shared condition-computation routine.
ZydisMnemonic SetccToJcc(ZydisMnemonic m) {
    switch (m) {
    case ZYDIS_MNEMONIC_SETZ:
        return ZYDIS_MNEMONIC_JZ;
    case ZYDIS_MNEMONIC_SETNZ:
        return ZYDIS_MNEMONIC_JNZ;
    case ZYDIS_MNEMONIC_SETS:
        return ZYDIS_MNEMONIC_JS;
    case ZYDIS_MNEMONIC_SETNS:
        return ZYDIS_MNEMONIC_JNS;
    case ZYDIS_MNEMONIC_SETO:
        return ZYDIS_MNEMONIC_JO;
    case ZYDIS_MNEMONIC_SETNO:
        return ZYDIS_MNEMONIC_JNO;
    case ZYDIS_MNEMONIC_SETP:
        return ZYDIS_MNEMONIC_JP;
    case ZYDIS_MNEMONIC_SETNP:
        return ZYDIS_MNEMONIC_JNP;
    case ZYDIS_MNEMONIC_SETB:
        return ZYDIS_MNEMONIC_JB;
    case ZYDIS_MNEMONIC_SETNB:
        return ZYDIS_MNEMONIC_JNB;
    case ZYDIS_MNEMONIC_SETBE:
        return ZYDIS_MNEMONIC_JBE;
    case ZYDIS_MNEMONIC_SETNBE:
        return ZYDIS_MNEMONIC_JNBE;
    case ZYDIS_MNEMONIC_SETL:
        return ZYDIS_MNEMONIC_JL;
    case ZYDIS_MNEMONIC_SETNL:
        return ZYDIS_MNEMONIC_JNL;
    case ZYDIS_MNEMONIC_SETLE:
        return ZYDIS_MNEMONIC_JLE;
    case ZYDIS_MNEMONIC_SETNLE:
        return ZYDIS_MNEMONIC_JNLE;
    default:
        return ZYDIS_MNEMONIC_INVALID;
    }
}

/// SETcc r/m8 — set the low byte of the destination to 0 or 1 based
/// on the condition. Upper bits of the destination's parent GPR are
/// preserved (unlike most byte ops, SETcc only touches byte 0).
///
/// Observed at libc 0x807a084b2 immediately after a CMPXCHG, as the
/// classic atomic-CAS-success-test idiom:
///
///     cmpxchg [mem], reg
///     setz al
///     test al, al
///     jz   retry
///
/// We compute the condition via the shared EmitJccCondition helper
/// (which yields 0/1 in rcx), then merge into the destination's
/// 8-bit slot while preserving the upper 56 bits.
///
/// Only handles register destinations for now. Memory-destination
/// SETcc exists but is rare; mem-dst would also need an effective-
/// address computation up front to free up rdx.
bool EmitSetcc(const ZydisDecodedInstruction& insn, const ZydisDecodedOperand* ops, u64 next_rip,
               Xbyak::CodeGenerator& c) {
    if (insn.operand_width != 8)
        return false;
    if (insn.operand_count_visible != 1)
        return false;

    const ZydisMnemonic jcc_equiv = SetccToJcc(insn.mnemonic);
    if (jcc_equiv == ZYDIS_MNEMONIC_INVALID)
        return false;

    // Memory-destination form: `setcc byte[mem]`. Observed in PS4
    // binaries around the libc / setjmp probe paths (length=5 with
    // a disp32). Compute the EA, stash it in r10 to survive the
    // condition computation (which uses rax/rcx/rdx/r8), then store
    // the 0/1 result byte.
    if (ops[0].type == ZYDIS_OPERAND_TYPE_MEMORY) {
        if (!EmitEffectiveAddress(ops[0].mem, next_rip, c))
            return false;
        c.mov(r10, rdx);
        if (!EmitJccCondition(jcc_equiv, c))
            return false;
        c.mov(byte[r10], cl);
        return true;
    }

    if (ops[0].type != ZYDIS_OPERAND_TYPE_REGISTER)
        return false;

    // ZydisGprToIndex maps AL/CL/DL/BL/SPL/BPL/SIL/DIL/R8B..R15B to
    // their parent-GPR index 0..15. AH/BH/CH/DH return -1 — we
    // refuse those (high-byte writes need a shift, and they're
    // uncommon in compiler-emitted code).
    const int dst_idx = ZydisGprToIndex(ops[0].reg.value);
    if (dst_idx < 0)
        return false;

    // Compute condition into rcx (0 or 1). Uses rax/rcx/rdx; we
    // need r8 afterward for the merge, so stash and restore.
    if (!EmitJccCondition(jcc_equiv, c))
        return false;

    // Merge: dst_slot = (dst_slot & ~0xFF) | (rcx & 0xFF)
    // rcx is already 0 or 1 from EmitJccCondition; mask defensively.
    c.and_(rcx, 0xFF);
    c.mov(r8, qword[r13 + GprOffset(dst_idx)]);
    c.mov(rax, ~static_cast<u64>(0xFF)); // not encodable as imm32
    c.and_(r8, rax);
    c.or_(r8, rcx);
    c.mov(qword[r13 + GprOffset(dst_idx)], r8);
    return true;
}

/// VPCMPEQB dst, src1, src2 — packed byte compare-equal.
///
/// For each byte position i: dst[i] = (src1[i] == src2[i]) ? 0xFF : 0x00.
///
/// Observed in Sonic Mania (and Hello World / video-out init) at all
/// three test-case sites; the mnemonic is used by libc's memcmp-derived
/// fast-path and by initialisation code checking sentinel byte lanes.
///
/// We use the host VPCMPEQB instruction directly on host XMM registers
/// per 128-bit lane — the same strategy VPSHUFB uses. This is cheaper
/// than a scalar byte loop (16-32 conditional stores vs 3-4 host VEX
/// instructions) and avoids any accuracy questions about scalar
/// emulation of the lane-independence invariant.
///
/// VPCMPEQB does not affect any flags per the x86 spec (unlike scalar
/// CMP / TEST), so no rflags writeback is needed.
///
/// Supported forms:
///   3-operand VEX (normal assembler output): dst, src1, src2.
///   2-operand (dst == src1 folded by assembler): dst, src2.
/// Both 128-bit (xmm) and 256-bit (ymm) widths.
bool EmitVpcmpeqb(const ZydisDecodedInstruction& insn, const ZydisDecodedOperand* ops,
                  Xbyak::CodeGenerator& c) {
    if (insn.mnemonic != ZYDIS_MNEMONIC_VPCMPEQB)
        return false;

    int dst_idx, src1_idx, src2_idx;
    if (insn.operand_count_visible == 3) {
        if (ops[0].type != ZYDIS_OPERAND_TYPE_REGISTER ||
            ops[1].type != ZYDIS_OPERAND_TYPE_REGISTER ||
            ops[2].type != ZYDIS_OPERAND_TYPE_REGISTER)
            return false;
        dst_idx = ZydisVecToIndex(ops[0].reg.value);
        src1_idx = ZydisVecToIndex(ops[1].reg.value);
        src2_idx = ZydisVecToIndex(ops[2].reg.value);
    } else if (insn.operand_count_visible == 2) {
        // Assembler folded dst == src1: `vpcmpeqb xmm0, xmm1` means
        // xmm0 = xmm0 OP xmm1.
        if (ops[0].type != ZYDIS_OPERAND_TYPE_REGISTER ||
            ops[1].type != ZYDIS_OPERAND_TYPE_REGISTER)
            return false;
        dst_idx = ZydisVecToIndex(ops[0].reg.value);
        src1_idx = dst_idx; // implicit src1 == dst
        src2_idx = ZydisVecToIndex(ops[1].reg.value);
    } else {
        return false;
    }
    if (dst_idx < 0 || src1_idx < 0 || src2_idx < 0)
        return false;

    // Zydis reports ops[0].size in bits for vector registers: 128 or 256.
    // (insn.operand_width is the element width which Zydis encodes as a
    // different field for VEX instructions; we ignore it here.)
    const int vec_bits = ops[0].size;
    if (vec_bits != 128 && vec_bits != 256)
        return false;

    // Process each 128-bit lane independently (VPCMPEQB AVX2 256-bit
    // operates on two independent 128-bit halves, with no cross-lane
    // byte movement — same invariant VPSHUFB relies on).
    for (int lane = 0; lane < vec_bits / 128; ++lane) {
        const u32 src1_off = YmmChunkOffset(src1_idx, lane * 2);
        const u32 src2_off = YmmChunkOffset(src2_idx, lane * 2);
        const u32 dst_off = YmmChunkOffset(dst_idx, lane * 2);
        c.vmovdqu(xmm0, ptr[r13 + src1_off]);
        c.vmovdqu(xmm1, ptr[r13 + src2_off]);
        c.vpcmpeqb(xmm0, xmm0, xmm1);
        c.vmovdqu(ptr[r13 + dst_off], xmm0);
    }
    // VEX 128-bit form zeros bits 255:128 of the destination YMM.
    if (vec_bits == 128) {
        c.xor_(rax, rax);
        c.mov(qword[r13 + YmmChunkOffset(dst_idx, 2)], rax);
        c.mov(qword[r13 + YmmChunkOffset(dst_idx, 3)], rax);
    }
    return true;
}

/// SFENCE — serialize all prior non-temporal stores. No operands.
///
/// Observed at libc 0x805f2b42d, immediately after the VMOVNTDQ
/// memset loop we added last iteration. The pairing is canonical
/// for the standard "fast bzero" pattern: non-temporal stores to
/// avoid polluting the cache, then SFENCE to make sure subsequent
/// readers observe the writes.
///
/// On the host, our VMOVNTDQ implementation already issues regular
/// (cache-line-allocating) GPR stores rather than non-temporal
/// stores, so SFENCE's "drain the non-temporal store buffer"
/// semantics are essentially a no-op for the emitted code. We
/// still emit a host SFENCE — it's a single byte after encoding
/// (actually 3 bytes: 0F AE F8) and provides a cheap guarantee
/// that any prior memory effects we generated *do* become visible
/// before the next instruction.
bool EmitSfence(const ZydisDecodedInstruction& insn, Xbyak::CodeGenerator& c) {
    if (insn.mnemonic != ZYDIS_MNEMONIC_SFENCE)
        return false;
    c.sfence();
    return true;
}

/// BEXTR r64, r/m64, r64 — BMI1 three-operand bitfield extract.
///   dst = (src1 >> control[7:0]) & ((1 << control[15:8]) - 1)
///
/// Sister instruction to ANDN; same VEX three-operand structure.
/// Observed at libc 0x807a1c2b2, inside what appears to be a
/// hash-bucket lookup using bit-field decomposition.
///
/// Flags: writes ZF from result; clears CF/OF; others undefined.
/// We round-trip rflags through the host so its exact semantics
/// propagate — same shape as EmitAndn.
///
/// Only the 64-bit register-register form is handled; mem source
/// can be added by routing through EmitEffectiveAddress when
/// observed, and 32-bit BEXTR follows the same skeleton with eax
/// substitutions.
bool EmitBextr(const ZydisDecodedInstruction& insn, const ZydisDecodedOperand* ops, u64 next_rip,
               Xbyak::CodeGenerator& c) {
    if (insn.mnemonic != ZYDIS_MNEMONIC_BEXTR)
        return false;
    if (insn.operand_width != 64 && insn.operand_width != 32)
        return false;
    if (insn.operand_count_visible != 3)
        return false;
    if (ops[0].type != ZYDIS_OPERAND_TYPE_REGISTER)
        return false;
    if (ops[2].type != ZYDIS_OPERAND_TYPE_REGISTER)
        return false;

    const int dst_idx = ZydisGprToIndex(ops[0].reg.value);
    const int src2_idx = ZydisGprToIndex(ops[2].reg.value);
    if (dst_idx < 0 || src2_idx < 0)
        return false;

    // Load src1 (the data to extract from) into r9. It can be either
    // a register (the original observed form) or memory (libc-driven
    // path at 0x80000ad1d that reads the bitfield from a structure
    // field). The mem path uses EmitEffectiveAddress, which writes
    // rdx — stash to r10 then dereference into r9. r9 ends up
    // holding the data either way.
    if (ops[1].type == ZYDIS_OPERAND_TYPE_REGISTER) {
        const int src1_idx = ZydisGprToIndex(ops[1].reg.value);
        if (src1_idx < 0)
            return false;
        c.mov(r9, qword[r13 + GprOffset(src1_idx)]);
    } else if (ops[1].type == ZYDIS_OPERAND_TYPE_MEMORY) {
        if (!EmitEffectiveAddress(ops[1].mem, next_rip, c))
            return false;
        // rdx = effective address. Dereference into r9 directly;
        // for the 32-bit form we still load 64 bits — host BEXTR
        // only reads bits 0..31 in 32-bit mode, so the upper half
        // is ignored. (No alignment requirement either way.)
        if (insn.operand_width == 32) {
            c.mov(r9d, dword[rdx]);
        } else {
            c.mov(r9, qword[rdx]);
        }
    } else {
        return false;
    }

    // src2 = the control register (low 8 bits = start, next 8 = len)
    c.mov(r10, qword[r13 + GprOffset(src2_idx)]);

    c.mov(r8, qword[r13 + Offsets::Rflags]);
    c.push(r8);
    c.popfq();

    // Host BEXTR: dst = bitfield(src, control). xbyak's bextr is
    // (dst, src, control). For the 32-bit form, writing to eax
    // zero-extends bits 63:32 of rax automatically, so the qword
    // storeback below sees a clean high half.
    if (insn.operand_width == 32) {
        c.bextr(eax, r9d, r10d);
    } else {
        c.bextr(rax, r9, r10);
    }

    c.pushfq();
    c.pop(r8);
    c.mov(qword[r13 + Offsets::Rflags], r8);

    c.mov(qword[r13 + GprOffset(dst_idx)], rax);
    return true;
}

/// POPCNT — population count (count of 1 bits in src, written to dst).
/// Flags: ZF set iff src == 0; OF/SF/AF/CF/PF all cleared.
///
/// We let host POPCNT do the arithmetic and set the host rflags
/// directly, then capture the host rflags into the guest slot. The
/// rflags round-trip pattern matches other "host op produces correct
/// guest flags" emitters (TEST, neg etc.).
///
/// Used by Sonic Mania (libc fast paths) at width=64. The CPUID
/// spoof now advertises POPCNT (leaf 1 ECX bit 23) so a guest that
/// gates on the bit will see it as available; games that emit POPCNT
/// without checking — the more common pattern in JIT-compiled
/// targets — get correct behavior regardless.
bool EmitPopcnt(const ZydisDecodedInstruction& insn, const ZydisDecodedOperand* ops, u64 next_rip,
                Xbyak::CodeGenerator& c) {
    if (insn.mnemonic != ZYDIS_MNEMONIC_POPCNT)
        return false;
    if (insn.operand_width != 32 && insn.operand_width != 64)
        return false;
    if (ops[0].type != ZYDIS_OPERAND_TYPE_REGISTER)
        return false;

    const int dst_idx = ZydisGprToIndex(ops[0].reg.value);
    if (dst_idx < 0)
        return false;

    // Load src into r9 (reg or mem).
    if (ops[1].type == ZYDIS_OPERAND_TYPE_REGISTER) {
        const int src_idx = ZydisGprToIndex(ops[1].reg.value);
        if (src_idx < 0)
            return false;
        c.mov(r9, qword[r13 + GprOffset(src_idx)]);
    } else if (ops[1].type == ZYDIS_OPERAND_TYPE_MEMORY) {
        if (!EmitEffectiveAddress(ops[1].mem, next_rip, c))
            return false;
        c.mov(r9, qword[rdx]);
    } else {
        return false;
    }

    // Set host rflags from guest so unrelated flag bits survive
    // (POPCNT only writes ZF and clears the other five arithmetic
    // bits — the non-arithmetic bits like IF/DF must be preserved).
    c.mov(r8, qword[r13 + Offsets::Rflags]);
    c.push(r8);
    c.popfq();

    if (insn.operand_width == 32) {
        c.popcnt(eax, r9d);
    } else {
        c.popcnt(rax, r9);
    }

    c.pushfq();
    c.pop(r8);
    c.mov(qword[r13 + Offsets::Rflags], r8);

    // 32-bit POPCNT writes eax which zero-extends rax automatically;
    // qword storeback lands a clean upper half.
    c.mov(qword[r13 + GprOffset(dst_idx)], rax);
    return true;
}

/// CPUID — spoofed to report an AMD Jaguar (PS4 APU family 16h)
/// rather than passing through to the host. The original pass-through
/// implementation leaked host details: vendor (likely "GenuineIntel"
/// rather than "AuthenticAMD"), feature bits the PS4 doesn't have
/// (AVX2, FMA, AVX-512, BMI2, etc.) and feature bits we haven't
/// emitted yet (e.g. POPCNT). Real PS4 binaries inspect this output
/// to gate code paths; advertising features we don't support causes
/// downstream "unsupported insn" exits — sometimes deep inside hot
/// paths the user can't easily route around.
///
/// Strategy: synthesise responses for the leaves a guest is likely
/// to query, returning zeros for everything else. The leaves
/// covered are 0, 1, 7, 0x80000000, 0x80000001, and the brand-string
/// triplet 0x80000002–0x80000004. Standard max-leaf is reported as
/// 7 (not 0xD or higher), so anything not explicitly handled — leaf
/// 0xB topology, leaf 0xD extended state, etc. — appears unsupported
/// to the guest, which is honest about what we provide.
///
/// Advertised feature bits are intersection( Jaguar, JIT coverage ):
///   Leaf 1 ECX  — SSE3, SSSE3, CMPXCHG16B, SSE4.1, SSE4.2,
///                 OSXSAVE, AVX. Skipped: POPCNT, AES, XSAVE, FMA,
///                 PCLMUL, F16C, RDRAND, MOVBE (no emitters yet).
///   Leaf 1 EDX  — standard baseline (FPU/CMOV/MMX/FXSR/SSE/SSE2/…).
///   Leaf 7 EBX  — BMI1 only (ANDN/BEXTR; we have these). Notably
///                 NOT AVX2 nor BMI2 — Jaguar lacks both, and we
///                 lack emitters either way.
///   Leaf 0x8000_0001 ECX — LahfSahf, ABM (LZCNT), SSE4A.
///   Leaf 0x8000_0001 EDX — LM (long mode), NX, RDTSCP, baseline.
///
/// Brand string is "AMD Custom Jaguar 8-Core APU" + space padding,
/// 48 bytes total including trailing NUL, packed little-endian into
/// the 12 dwords reported across the three brand-string leaves.
///
/// Implementation: branch-and-store table. The lifter loads the
/// guest's leaf into eax and subleaf into ecx, zeroes r8d–r11d as
/// the default (zero-fill) response, dispatches via cmp/je on eax,
/// each leaf body overwrites r8d–r11d with the canned response,
/// then a common tail stores all four to the guest RAX/RBX/RCX/RDX
/// slots (slot indices 0/3/1/2 — Zydis GPR ordering).
///
/// rbx, r12, r14, r15 are all reserved (gateway-saved or
/// dispatcher channels), so we use rax/rcx/r8–r11 as scratch.
/// 32-bit writes to r8d–r11d zero-extend bits 63:32 automatically,
/// so qword stores at the tail land clean values in every slot.
bool EmitCpuid(const ZydisDecodedInstruction& insn, Xbyak::CodeGenerator& c) {
    if (insn.mnemonic != ZYDIS_MNEMONIC_CPUID)
        return false;

    // Vendor string "AuthenticAMD" — note the CPUID convention is
    // EBX:EDX:ECX (not EBX:ECX:EDX), so the substrings are split
    // accordingly. Each dword is little-endian-packed 4 chars.
    constexpr u32 kVendorEbx = 0x68747541; // "Auth"
    constexpr u32 kVendorEdx = 0x69746E65; // "enti"
    constexpr u32 kVendorEcx = 0x444D4163; // "cAMD"

    // Leaf 1 EAX — processor signature.
    //   Format: [ExtFam:8][ExtMod:4][Rsv:2][Type:2][Fam:4][Mod:4][Step:4]
    //   Jaguar reports Family = BaseFam(0xF) + ExtFam(0x7) = 0x16.
    //   Model = 0, Stepping = 1.
    constexpr u32 kLeaf1Eax = 0x00700F01;

    // Leaf 1 EBX — [LocalAPIC:8][MaxAPICIDs:8][CLFLUSH/8:8][BrandIdx:8]
    //   CLFLUSH line size / 8 = 8 (64-byte lines).
    //   Max logical processors per package = 8 (PS4 is octa-core).
    constexpr u32 kLeaf1Ebx = 0x08080000;

    // Leaf 1 ECX — feature flags (advertised subset, see header).
    constexpr u32 kLeaf1Ecx = (1u << 0) |  // SSE3
                              (1u << 9) |  // SSSE3
                              (1u << 13) | // CMPXCHG16B
                              (1u << 19) | // SSE4.1
                              (1u << 20) | // SSE4.2
                              (1u << 23) | // POPCNT
                              (1u << 27) | // OSXSAVE
                              (1u << 28);  // AVX

    // Leaf 1 EDX — baseline features, all set on Jaguar except HTT.
    constexpr u32 kLeaf1Edx =
        (1u << 0) | (1u << 1) | (1u << 2) | (1u << 3) |     // FPU,VME,DE,PSE
        (1u << 4) | (1u << 5) | (1u << 6) | (1u << 7) |     // TSC,MSR,PAE,MCE
        (1u << 8) | (1u << 9) | (1u << 11) | (1u << 12) |   // CX8,APIC,SEP,MTRR
        (1u << 13) | (1u << 14) | (1u << 15) | (1u << 16) | // PGE,MCA,CMOV,PAT
        (1u << 17) | (1u << 19) | (1u << 23) | (1u << 24) | // PSE36,CLFSH,MMX,FXSR
        (1u << 25) | (1u << 26);                            // SSE, SSE2

    // Leaf 7, subleaf 0 — structured extended features.
    constexpr u32 kLeaf7Sub0Ebx = (1u << 3); // BMI1

    // Leaf 0x80000001 ECX — AMD extended feature flags.
    constexpr u32 kLeafExt1Ecx = (1u << 0) | // LahfSahf
                                 (1u << 5) | // ABM (LZCNT/POPCNT extensions)
                                 (1u << 6);  // SSE4A

    // Leaf 0x80000001 EDX — AMD extended/64-bit features.
    constexpr u32 kLeafExt1Edx = kLeaf1Edx | (1u << 20) | // NX (XD bit)
                                 (1u << 27) |             // RDTSCP
                                 (1u << 29);              // LM (64-bit long mode — mandatory)

    // Brand string "AMD Custom Jaguar 8-Core APU" + 19 spaces + NUL,
    // packed little-endian as 12 consecutive dwords across the three
    // 0x80000002–0x80000004 leaves.
    constexpr u32 kBrand2Eax = 0x20444D41; // "AMD "
    constexpr u32 kBrand2Ebx = 0x74737543; // "Cust"
    constexpr u32 kBrand2Ecx = 0x4A206D6F; // "om J"
    constexpr u32 kBrand2Edx = 0x61756761; // "agua"
    constexpr u32 kBrand3Eax = 0x2D382072; // "r 8-"
    constexpr u32 kBrand3Ebx = 0x65726F43; // "Core"
    constexpr u32 kBrand3Ecx = 0x55504120; // " APU"
    constexpr u32 kBrand3Edx = 0x20202020; // "    "
    constexpr u32 kBrand4Eax = 0x20202020; // "    "
    constexpr u32 kBrand4Ebx = 0x20202020; // "    "
    constexpr u32 kBrand4Ecx = 0x20202020; // "    "
    constexpr u32 kBrand4Edx = 0x00202020; // "   \0"

    Xbyak::Label l_0, l_1, l_7, l_e0, l_e1, l_b2, l_b3, l_b4, l_done;
    using LT = Xbyak::CodeGenerator::LabelType;

    // Load inputs: leaf into eax, subleaf into ecx.
    c.mov(eax, dword[r13 + GprOffset(0)]);
    c.mov(ecx, dword[r13 + GprOffset(1)]);

    // Default response — zero-fill. Any leaf not specifically handled
    // by the chain below falls through to the storeback at l_done
    // with all four results still zero.
    c.xor_(r8d, r8d);
    c.xor_(r9d, r9d);
    c.xor_(r10d, r10d);
    c.xor_(r11d, r11d);

    // Dispatch chain. T_NEAR forces a 32-bit-relative jump rather
    // than the xbyak-default 8-bit form — the per-leaf bodies are
    // far enough apart that an 8-bit displacement won't reach.
    c.cmp(eax, 0);
    c.je(l_0, LT::T_NEAR);
    c.cmp(eax, 1);
    c.je(l_1, LT::T_NEAR);
    c.cmp(eax, 7);
    c.je(l_7, LT::T_NEAR);
    c.cmp(eax, 0x80000000);
    c.je(l_e0, LT::T_NEAR);
    c.cmp(eax, 0x80000001);
    c.je(l_e1, LT::T_NEAR);
    c.cmp(eax, 0x80000002);
    c.je(l_b2, LT::T_NEAR);
    c.cmp(eax, 0x80000003);
    c.je(l_b3, LT::T_NEAR);
    c.cmp(eax, 0x80000004);
    c.je(l_b4, LT::T_NEAR);
    c.jmp(l_done, LT::T_NEAR);

    // Leaf 0 — max standard leaf + vendor string.
    c.L(l_0);
    c.mov(r8d, 7);
    c.mov(r9d, kVendorEbx);
    c.mov(r10d, kVendorEcx);
    c.mov(r11d, kVendorEdx);
    c.jmp(l_done, LT::T_NEAR);

    // Leaf 1 — signature + features.
    c.L(l_1);
    c.mov(r8d, kLeaf1Eax);
    c.mov(r9d, kLeaf1Ebx);
    c.mov(r10d, kLeaf1Ecx);
    c.mov(r11d, kLeaf1Edx);
    c.jmp(l_done, LT::T_NEAR);

    // Leaf 7 — structured extended features. Only subleaf 0 has
    // meaningful contents on Jaguar; other subleaves return zero
    // (which is what we'd hit by falling through to l_done with
    // r8d–r11d still zeroed from the default-response setup above).
    c.L(l_7);
    c.test(ecx, ecx);
    c.jnz(l_done, LT::T_NEAR);
    c.mov(r9d, kLeaf7Sub0Ebx);
    c.jmp(l_done, LT::T_NEAR);

    // Leaf 0x80000000 — max extended leaf + vendor string echo.
    c.L(l_e0);
    c.mov(r8d, 0x80000004);
    c.mov(r9d, kVendorEbx);
    c.mov(r10d, kVendorEcx);
    c.mov(r11d, kVendorEdx);
    c.jmp(l_done, LT::T_NEAR);

    // Leaf 0x80000001 — AMD extended features.
    c.L(l_e1);
    c.mov(r8d, kLeaf1Eax);
    c.mov(r10d, kLeafExt1Ecx);
    c.mov(r11d, kLeafExt1Edx);
    c.jmp(l_done, LT::T_NEAR);

    // Leaves 0x80000002–0x80000004 — brand string dwords 0..11.
    c.L(l_b2);
    c.mov(r8d, kBrand2Eax);
    c.mov(r9d, kBrand2Ebx);
    c.mov(r10d, kBrand2Ecx);
    c.mov(r11d, kBrand2Edx);
    c.jmp(l_done, LT::T_NEAR);

    c.L(l_b3);
    c.mov(r8d, kBrand3Eax);
    c.mov(r9d, kBrand3Ebx);
    c.mov(r10d, kBrand3Ecx);
    c.mov(r11d, kBrand3Edx);
    c.jmp(l_done, LT::T_NEAR);

    c.L(l_b4);
    c.mov(r8d, kBrand4Eax);
    c.mov(r9d, kBrand4Ebx);
    c.mov(r10d, kBrand4Ecx);
    c.mov(r11d, kBrand4Edx);
    // Fall through to l_done.

    // Common storeback. Store as qwords — the 32-bit writes above
    // zero-extended bits 63:32, so the upper halves of the guest
    // slots come out clean, matching x86-64 CPUID semantics.
    c.L(l_done);
    c.mov(qword[r13 + GprOffset(0)], r8);  // a → RAX
    c.mov(qword[r13 + GprOffset(3)], r9);  // b → RBX
    c.mov(qword[r13 + GprOffset(1)], r10); // c → RCX
    c.mov(qword[r13 + GprOffset(2)], r11); // d → RDX
    return true;
}

/// XGETBV — read an Extended Control Register (XCR).
///   Input:   ECX = XCR index (only XCR0 is defined in Sandy
///                  Bridge/Jaguar-era CPUs)
///   Outputs: EDX:EAX = 64-bit XCR value
///
/// Games reach this after CPUID reports OSXSAVE+AVX: the canonical
/// Intel-recommended AVX detection is
///     `cpuid -> OSXSAVE && AVX feature set` then
///     `xgetbv ecx=0 -> check bits 1 (SSE state) and 2 (AVX state)`.
/// Without an XGETBV implementation, games that do the full check
/// will disable their AVX paths — or, more commonly on this codebase,
/// hit the unsupported-insn exit and stop dead.
///
/// We don't actually have an XSAVE area; we just need to return
/// values that say "yes, the OS has enabled SSE+AVX state" so the
/// guest's "is AVX usable?" gate succeeds. That's XCR0 = 0x7
/// (bit 0 = x87 mandatory, bit 1 = SSE state, bit 2 = AVX state).
/// Any other XCR index returns 0 — matches an OS that hasn't enabled
/// extended features beyond AVX.
///
/// Implementation: tiny dispatcher. Load guest ECX into a host
/// scratch, branch on ecx==0, write the constants to slots 0/2
/// (RAX/RDX) via r8/r11. 32-bit writes zero-extend so the qword
/// storebacks land clean upper halves, matching the architectural
/// "EAX/EDX written, upper 32 zeroed by x86-64 rule" semantics.
bool EmitXgetbv(const ZydisDecodedInstruction& insn, Xbyak::CodeGenerator& c) {
    if (insn.mnemonic != ZYDIS_MNEMONIC_XGETBV)
        return false;

    using LT = Xbyak::CodeGenerator::LabelType;
    Xbyak::Label l_xcr0, l_done;

    // Load guest ECX.
    c.mov(ecx, dword[r13 + GprOffset(1)]);

    // Default response: zero in both halves.
    c.xor_(r8d, r8d);   // → EAX
    c.xor_(r11d, r11d); // → EDX

    c.test(ecx, ecx);
    c.jz(l_xcr0, LT::T_NEAR);
    c.jmp(l_done, LT::T_NEAR);

    // XCR0 — x87 (bit 0), SSE state (bit 1), AVX state (bit 2).
    // Reporting these as enabled is what an OSXSAVE-aware AVX-capable
    // OS would return; it's what the CPUID we spoofed promises.
    c.L(l_xcr0);
    c.mov(r8d, 0x7);
    // r11d stays zero — upper 32 of the XCR are reserved on Jaguar.

    c.L(l_done);
    c.mov(qword[r13 + GprOffset(0)], r8);  // → RAX
    c.mov(qword[r13 + GprOffset(2)], r11); // → RDX
    return true;
}

/// VPTEST xmm1, xmm2 — AVX bit-test across the entire vector.
///   ZF := ((xmm1 AND xmm2)        == 0)   "all bits common are zero"
///   CF := ((NOT xmm1 AND xmm2)    == 0)   "every bit in xmm2 is in xmm1"
///   OF/SF/AF/PF := 0
///
/// 256-bit ymm form follows the same definition with 256 bits per
/// operand. Games commonly use VPTEST as a "branchless equality" or
/// "any-bits-set" test after a vector compare; missing it cascades
/// into wrong control flow rather than a crash, so it's worth
/// getting right.
///
/// Implementation: stash the two operands into host xmm0/xmm1 via
/// VMOVDQU from the GuestState YMM lanes (the JIT owns the full XMM
/// file inside a block), run host VPTEST, then capture ZF and CF
/// into the guest rflags slot. The pushfq/popfq round-trip is the
/// cleanest way to copy all six arithmetic flags at once — host
/// VPTEST does the right thing for OF/SF/AF/PF (clears them), and
/// we preserve the non-arithmetic guest flag bits (TF/IF/DF/...)
/// by masking.
bool EmitVptest(const ZydisDecodedInstruction& insn, const ZydisDecodedOperand* ops,
                Xbyak::CodeGenerator& c) {
    if (insn.mnemonic != ZYDIS_MNEMONIC_VPTEST)
        return false;
    if (ops[0].type != ZYDIS_OPERAND_TYPE_REGISTER)
        return false;
    if (ops[1].type != ZYDIS_OPERAND_TYPE_REGISTER)
        return false;

    const int a_idx = ZydisVecToIndex(ops[0].reg.value);
    const int b_idx = ZydisVecToIndex(ops[1].reg.value);
    if (a_idx < 0 || b_idx < 0)
        return false;

    const int vec_bits = ops[0].size;
    if (vec_bits != 128 && vec_bits != 256)
        return false;

    // Load operands into host xmm0/xmm1 (or ymm0/ymm1 for 256-bit).
    // Host XMM0..XMM5 are JIT-scratch inside a block per the
    // host-register convention; no save/restore needed.
    if (vec_bits == 128) {
        c.vmovdqu(xmm0, ptr[r13 + YmmChunkOffset(a_idx, 0)]);
        c.vmovdqu(xmm1, ptr[r13 + YmmChunkOffset(b_idx, 0)]);
        c.vptest(xmm0, xmm1);
    } else {
        c.vmovdqu(ymm0, ptr[r13 + YmmChunkOffset(a_idx, 0)]);
        c.vmovdqu(ymm1, ptr[r13 + YmmChunkOffset(b_idx, 0)]);
        c.vptest(ymm0, ymm1);
    }

    // Merge host arithmetic flags into the guest rflags slot. VPTEST
    // produces CF/ZF/PF/AF/SF/OF — but the architectural spec sets
    // OF/AF/PF/SF to 0 explicitly. pushfq captures everything; mask
    // the host result to just the bits VPTEST writes (CF=0, PF=2,
    // AF=4, ZF=6, SF=7, OF=11 → 0x8D5) and OR into the guest rflags
    // with those same bits cleared first.
    constexpr u64 kArithMask = 0x8D5; // CF|PF|AF|ZF|SF|OF
    c.pushfq();
    c.pop(rax);
    c.and_(rax, kArithMask);
    c.mov(rdx, qword[r13 + Offsets::Rflags]);
    c.mov(rcx, ~kArithMask);
    c.and_(rdx, rcx);
    c.or_(rdx, rax);
    c.mov(qword[r13 + Offsets::Rflags], rdx);
    return true;
}

/// VPCMPISTRI xmm1, xmm2/m128, imm8 — SSE4.2 "compare implicit-length
/// strings, return index". Used by glibc string functions (strlen,
/// strchr, strcmp variants) for vectorized inner loops.
///
/// Architectural behavior:
///   ECX  := result index (semantics controlled by imm8 bit 6)
///   CF   := (intermediate boolean vector is not all zero)
///   ZF   := (xmm2 contains a null character / terminator)
///   SF   := (xmm1 contains a null character / terminator)
///   OF   := bit 0 of intermediate boolean vector
///   AF, PF := 0
///
/// Implementation: this is too complex to lift directly — the
/// imm8 control byte selects between four aggregation operations
/// (equal-any, ranges, equal-each, equal-ordered) across two data
/// formats (signed/unsigned, bytes/words), with polarity inversion
/// and output-sense bits. Running host VPCMPISTRI gets all of that
/// for free, including the rflags layout.
///
/// We load the two xmm operands from the GuestState YMM lanes into
/// host xmm0/xmm1 (JIT-scratch inside a block), run host VPCMPISTRI
/// with the same imm8, then capture host ECX → guest RCX (zero-
/// extended into the full slot) and merge the arithmetic flags into
/// the guest rflags using the same mask-and-or pattern as VPTEST.
///
/// Reg-mem second-operand form is deferred — glibc's string loops
/// use the reg-reg form after a load into a register operand.
bool EmitVpcmpistri(const ZydisDecodedInstruction& insn, const ZydisDecodedOperand* ops,
                    Xbyak::CodeGenerator& c) {
    if (insn.mnemonic != ZYDIS_MNEMONIC_VPCMPISTRI)
        return false;
    if (ops[0].type != ZYDIS_OPERAND_TYPE_REGISTER)
        return false;
    if (ops[1].type != ZYDIS_OPERAND_TYPE_REGISTER)
        return false;
    if (ops[2].type != ZYDIS_OPERAND_TYPE_IMMEDIATE)
        return false;

    const int a_idx = ZydisVecToIndex(ops[0].reg.value);
    const int b_idx = ZydisVecToIndex(ops[1].reg.value);
    if (a_idx < 0 || b_idx < 0)
        return false;

    const u8 imm = static_cast<u8>(ops[2].imm.value.u & 0xFFu);

    // Load both 128-bit operands into host scratch xmm0/xmm1.
    c.vmovdqu(xmm0, ptr[r13 + YmmChunkOffset(a_idx, 0)]);
    c.vmovdqu(xmm1, ptr[r13 + YmmChunkOffset(b_idx, 0)]);

    // Run host VPCMPISTRI. Side effects: host ECX = result index,
    // host rflags has CF/ZF/SF/OF set per spec (PF/AF cleared).
    c.vpcmpistri(xmm0, xmm1, imm);

    // Capture host ECX into guest RCX. 32-bit mov zero-extends bits
    // 63:32 of rax, then the qword storeback lands a clean slot
    // (matching the architectural "EAX/ECX writes zero-extend"
    // semantics that VPCMPISTRI's ECX write follows).
    c.mov(eax, ecx);
    c.mov(qword[r13 + GprOffset(1)], rax);

    // Merge host arithmetic flag bits into guest rflags. Same shape
    // as VPTEST. We need rax for the captured flags, but we just
    // used rax above — fine to clobber, the guest RCX slot is
    // already stored.
    constexpr u64 kArithMask = 0x8D5; // CF|PF|AF|ZF|SF|OF
    c.pushfq();
    c.pop(rax);
    c.and_(rax, kArithMask);
    c.mov(rdx, qword[r13 + Offsets::Rflags]);
    c.mov(rcx, ~kArithMask);
    c.and_(rdx, rcx);
    c.or_(rdx, rax);
    c.mov(qword[r13 + Offsets::Rflags], rdx);
    return true;
}

/// VPHADDD — packed horizontal add of 32-bit integers across paired
/// lanes. For each 128-bit half:
///   dst[0] = src1[0] + src1[1]
///   dst[1] = src1[2] + src1[3]
///   dst[2] = src2[0] + src2[1]
///   dst[3] = src2[2] + src2[3]
/// The 256-bit form runs this independently for the low and high
/// 128-bit lanes (no cross-lane shuffle). No flags affected.
///
/// Sonic Mania uses this in vectorised reduction code paths
/// (dot-product / sum-of-elements idioms). Lifting it scalar would
/// mean several GPR-relayed 32-bit adds across lane pairs; instead
/// we run host VPHADDD directly on JIT-scratch xmm0/xmm1 (the same
/// pattern VPSHUFB / VPCMPISTRI use). Requires SSSE3, which the
/// spoofed CPUID already advertises.
///
/// Operand-count handling mirrors VPSHUFB / VPCMPEQB: 3-visible
/// (dst, src1, src2) or 2-visible folded form (dst==src1, src2).
bool EmitVphaddd(const ZydisDecodedInstruction& insn, const ZydisDecodedOperand* ops,
                 Xbyak::CodeGenerator& c) {
    if (insn.mnemonic != ZYDIS_MNEMONIC_VPHADDD)
        return false;

    int dst_idx, src1_idx, src2_idx;
    if (insn.operand_count_visible == 3) {
        if (ops[0].type != ZYDIS_OPERAND_TYPE_REGISTER ||
            ops[1].type != ZYDIS_OPERAND_TYPE_REGISTER ||
            ops[2].type != ZYDIS_OPERAND_TYPE_REGISTER)
            return false;
        dst_idx = ZydisVecToIndex(ops[0].reg.value);
        src1_idx = ZydisVecToIndex(ops[1].reg.value);
        src2_idx = ZydisVecToIndex(ops[2].reg.value);
    } else if (insn.operand_count_visible == 2) {
        if (ops[0].type != ZYDIS_OPERAND_TYPE_REGISTER ||
            ops[1].type != ZYDIS_OPERAND_TYPE_REGISTER)
            return false;
        dst_idx = ZydisVecToIndex(ops[0].reg.value);
        src1_idx = dst_idx;
        src2_idx = ZydisVecToIndex(ops[1].reg.value);
    } else {
        return false;
    }
    if (dst_idx < 0 || src1_idx < 0 || src2_idx < 0)
        return false;

    const int vec_bits = ops[0].size;
    if (vec_bits != 128 && vec_bits != 256)
        return false;

    if (vec_bits == 128) {
        c.vmovdqu(xmm0, ptr[r13 + YmmChunkOffset(src1_idx, 0)]);
        c.vmovdqu(xmm1, ptr[r13 + YmmChunkOffset(src2_idx, 0)]);
        c.vphaddd(xmm0, xmm0, xmm1);
        c.vmovdqu(ptr[r13 + YmmChunkOffset(dst_idx, 0)], xmm0);
        // 128-bit VEX zeroes bits 255:128 of the destination YMM.
        c.xor_(rax, rax);
        c.mov(qword[r13 + YmmChunkOffset(dst_idx, 2)], rax);
        c.mov(qword[r13 + YmmChunkOffset(dst_idx, 3)], rax);
    } else {
        c.vmovdqu(ymm0, ptr[r13 + YmmChunkOffset(src1_idx, 0)]);
        c.vmovdqu(ymm1, ptr[r13 + YmmChunkOffset(src2_idx, 0)]);
        c.vphaddd(ymm0, ymm0, ymm1);
        c.vmovdqu(ptr[r13 + YmmChunkOffset(dst_idx, 0)], ymm0);
    }
    return true;
}

/// VPUNPCKLQDQ — unpack and interleave the low 64-bit halves of two
/// vectors. For each 128-bit lane:
///   dst[0] = src1[0]   (low  64 of src1)
///   dst[1] = src2[0]   (low  64 of src2)
/// The high 64 of each source within the lane is discarded. The
/// 256-bit form runs this independently for the low and high 128-bit
/// lanes (no cross-lane shuffle). No flags affected.
///
/// Compiler-emitted in vectorised gather/broadcast patterns: take
/// two scalar 64-bit values that landed in xmm regs and pack them
/// as a 128-bit vector of two 64-bit elements.
///
/// We run host VPUNPCKLQDQ directly on JIT-scratch xmm0/xmm1 — same
/// pattern as VPSHUFB / VPCMPEQB. Both 3-operand visible form
/// (dst, src1, src2) and the 2-operand folded form (dst==src1, src2)
/// are handled.
bool EmitVpunpcklqdq(const ZydisDecodedInstruction& insn, const ZydisDecodedOperand* ops,
                     Xbyak::CodeGenerator& c) {
    if (insn.mnemonic != ZYDIS_MNEMONIC_VPUNPCKLQDQ)
        return false;

    int dst_idx, src1_idx, src2_idx;
    if (insn.operand_count_visible == 3) {
        if (ops[0].type != ZYDIS_OPERAND_TYPE_REGISTER ||
            ops[1].type != ZYDIS_OPERAND_TYPE_REGISTER ||
            ops[2].type != ZYDIS_OPERAND_TYPE_REGISTER)
            return false;
        dst_idx = ZydisVecToIndex(ops[0].reg.value);
        src1_idx = ZydisVecToIndex(ops[1].reg.value);
        src2_idx = ZydisVecToIndex(ops[2].reg.value);
    } else if (insn.operand_count_visible == 2) {
        if (ops[0].type != ZYDIS_OPERAND_TYPE_REGISTER ||
            ops[1].type != ZYDIS_OPERAND_TYPE_REGISTER)
            return false;
        dst_idx = ZydisVecToIndex(ops[0].reg.value);
        src1_idx = dst_idx;
        src2_idx = ZydisVecToIndex(ops[1].reg.value);
    } else {
        return false;
    }
    if (dst_idx < 0 || src1_idx < 0 || src2_idx < 0)
        return false;

    const int vec_bits = ops[0].size;
    if (vec_bits != 128 && vec_bits != 256)
        return false;

    if (vec_bits == 128) {
        c.vmovdqu(xmm0, ptr[r13 + YmmChunkOffset(src1_idx, 0)]);
        c.vmovdqu(xmm1, ptr[r13 + YmmChunkOffset(src2_idx, 0)]);
        c.vpunpcklqdq(xmm0, xmm0, xmm1);
        c.vmovdqu(ptr[r13 + YmmChunkOffset(dst_idx, 0)], xmm0);
        // VEX-128 zeroes bits 255:128 of the destination YMM.
        c.xor_(rax, rax);
        c.mov(qword[r13 + YmmChunkOffset(dst_idx, 2)], rax);
        c.mov(qword[r13 + YmmChunkOffset(dst_idx, 3)], rax);
    } else {
        c.vmovdqu(ymm0, ptr[r13 + YmmChunkOffset(src1_idx, 0)]);
        c.vmovdqu(ymm1, ptr[r13 + YmmChunkOffset(src2_idx, 0)]);
        c.vpunpcklqdq(ymm0, ymm0, ymm1);
        c.vmovdqu(ptr[r13 + YmmChunkOffset(dst_idx, 0)], ymm0);
    }
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
    std::fprintf(stderr, "[lifter] %llu blocks compiled, %llu bytes emitted, %llu unsupported\n",
                 (unsigned long long)blocks_compiled_, (unsigned long long)bytes_emitted_,
                 (unsigned long long)unsupported_hits_);
    std::fflush(stderr);
}

void* Lifter::CompileBlock(u64 guest_rip) {
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
    std::fprintf(stderr, "[lifter] CompileBlock: guest_rip = 0x%llx\n",
                 static_cast<unsigned long long>(guest_rip));
    std::fflush(stderr);

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
        std::fprintf(stderr, "[lifter] about to decode at 0x%llx\n",
                     static_cast<unsigned long long>(rip));
        std::fflush(stderr);
        const auto status =
            ZydisDecoderDecodeFull(&decoder, reinterpret_cast<const void*>(rip), 15, &insn, ops);
        std::fprintf(stderr, "[lifter] decoded at 0x%llx ok=%d mnemonic=%s\n",
                     static_cast<unsigned long long>(rip), ZYAN_SUCCESS(status) ? 1 : 0,
                     ZYAN_SUCCESS(status) ? ZydisMnemonicGetString(insn.mnemonic)
                                          : "(decode-failed)");
        std::fflush(stderr);
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
        switch (insn.mnemonic) {
        case ZYDIS_MNEMONIC_MOV:
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
        case ZYDIS_MNEMONIC_LEA:
            handled = EmitLea(insn, ops, next_rip, c);
            break;
        case ZYDIS_MNEMONIC_MOVSXD:
            handled = EmitMovsxd(insn, ops, next_rip, c);
            break;
        case ZYDIS_MNEMONIC_MOVSX:
            handled = EmitMovsx(insn, ops, next_rip, c);
            break;
        case ZYDIS_MNEMONIC_ADD:
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
        case ZYDIS_MNEMONIC_SUB:
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
        case ZYDIS_MNEMONIC_CMP:
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
        case ZYDIS_MNEMONIC_ADC:
            handled = EmitAdcSbb64(insn, ops, c, AdcSbbKind::Adc);
            break;
        case ZYDIS_MNEMONIC_SBB:
            handled = EmitAdcSbb64(insn, ops, c, AdcSbbKind::Sbb);
            break;

        // Function epilogue shorthand: mov rsp, rbp; pop rbp.
        case ZYDIS_MNEMONIC_LEAVE:
            handled = EmitLeave(c);
            break;
        case ZYDIS_MNEMONIC_TEST:
            if (insn.operand_width == 8) {
                handled = EmitNarrowArith8(insn, ops, next_rip, c, NarrowArithKind::Test);
            } else if (insn.operand_width == 16) {
                handled = EmitNarrowArith16(insn, ops, next_rip, c, NarrowArithKind::Test);
            } else if (insn.operand_width == 32) {
                handled = EmitNarrowArith32(insn, ops, next_rip, c, NarrowArithKind::Test);
            } else {
                handled = EmitTest(insn, ops, c);
            }
            break;
        case ZYDIS_MNEMONIC_XOR:
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
        case ZYDIS_MNEMONIC_AND:
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
        case ZYDIS_MNEMONIC_OR:
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
        case ZYDIS_MNEMONIC_NOT:
            handled = EmitNot(insn, ops, c);
            break;
        case ZYDIS_MNEMONIC_NEG:
            handled = EmitNeg(insn, ops, c);
            break;
        case ZYDIS_MNEMONIC_INC:
            handled = EmitInc(insn, ops, c);
            break;
        case ZYDIS_MNEMONIC_DEC:
            handled = EmitDec(insn, ops, c);
            break;
        case ZYDIS_MNEMONIC_BT:
            handled = EmitBt(insn, ops, c);
            break;
        case ZYDIS_MNEMONIC_MOVZX:
            handled = EmitMovzx(insn, ops, next_rip, c);
            break;

        // Shifts. All three use the same emit with a kind tag.
        case ZYDIS_MNEMONIC_SHL:
            handled = (insn.operand_width == 8)    ? EmitShift8(insn, ops, c, ShiftKind::Shl)
                      : (insn.operand_width == 32) ? EmitShift32(insn, ops, c, ShiftKind::Shl)
                                                   : EmitShift64(insn, ops, c, ShiftKind::Shl);
            break;
        case ZYDIS_MNEMONIC_SHR:
            handled = (insn.operand_width == 8)    ? EmitShift8(insn, ops, c, ShiftKind::Shr)
                      : (insn.operand_width == 32) ? EmitShift32(insn, ops, c, ShiftKind::Shr)
                                                   : EmitShift64(insn, ops, c, ShiftKind::Shr);
            break;
        case ZYDIS_MNEMONIC_SAR:
            handled = (insn.operand_width == 8)    ? EmitShift8(insn, ops, c, ShiftKind::Sar)
                      : (insn.operand_width == 32) ? EmitShift32(insn, ops, c, ShiftKind::Sar)
                                                   : EmitShift64(insn, ops, c, ShiftKind::Sar);
            break;

        // Rotates. Same shape as shifts.
        case ZYDIS_MNEMONIC_ROL:
            handled = EmitRotate64(insn, ops, c, RotateKind::Rol);
            break;
        case ZYDIS_MNEMONIC_ROR:
            handled = EmitRotate64(insn, ops, c, RotateKind::Ror);
            break;

        // Multiplication. EmitImul dispatches by operand_count_visible.
        case ZYDIS_MNEMONIC_IMUL:
            handled = EmitImul(insn, ops, next_rip, c);
            break;
        case ZYDIS_MNEMONIC_DIV:
            handled = EmitDiv(insn, ops, next_rip, c);
            break;

        // Sign-extension family. No operands; operate on RAX/RDX.
        case ZYDIS_MNEMONIC_CWDE:
            handled = EmitCwde(c);
            break;
        case ZYDIS_MNEMONIC_CDQE:
            handled = EmitCdqe(c);
            break;
        case ZYDIS_MNEMONIC_CDQ:
            handled = EmitCdq(c);
            break;
        case ZYDIS_MNEMONIC_CQO:
            handled = EmitCqo(c);
            break;

        // Direct carry-flag manipulation.
        case ZYDIS_MNEMONIC_STC:
            handled = EmitStc(c);
            break;
        case ZYDIS_MNEMONIC_CLC:
            handled = EmitClc(c);
            break;
        case ZYDIS_MNEMONIC_CMC:
            handled = EmitCmc(c);
            break;

        // NOP — no semantic effect, just consume the bytes.
        // Common forms: 90 (1-byte), 66 90 (2-byte),
        // 0F 1F /0 (multi-byte padding). All decode as NOP.
        case ZYDIS_MNEMONIC_NOP:
            handled = true;
            break;

        // All CMOVcc variants go through EmitCmov, which maps
        // the mnemonic to the matching Jcc condition encoding.
        case ZYDIS_MNEMONIC_CMOVZ:
        case ZYDIS_MNEMONIC_CMOVNZ:
        case ZYDIS_MNEMONIC_CMOVS:
        case ZYDIS_MNEMONIC_CMOVNS:
        case ZYDIS_MNEMONIC_CMOVO:
        case ZYDIS_MNEMONIC_CMOVNO:
        case ZYDIS_MNEMONIC_CMOVP:
        case ZYDIS_MNEMONIC_CMOVNP:
        case ZYDIS_MNEMONIC_CMOVB:
        case ZYDIS_MNEMONIC_CMOVNB:
        case ZYDIS_MNEMONIC_CMOVBE:
        case ZYDIS_MNEMONIC_CMOVNBE:
        case ZYDIS_MNEMONIC_CMOVL:
        case ZYDIS_MNEMONIC_CMOVNL:
        case ZYDIS_MNEMONIC_CMOVLE:
        case ZYDIS_MNEMONIC_CMOVNLE:
            handled = EmitCmov(insn, ops, next_rip, c);
            break;

        case ZYDIS_MNEMONIC_PUSH:
            handled = EmitPush(insn, ops, c);
            break;
        case ZYDIS_MNEMONIC_POP:
            handled = EmitPop(insn, ops, c);
            break;
        case ZYDIS_MNEMONIC_RET:
            handled = EmitRet(insn, ops, c);
            if (handled)
                emitted_terminator = true;
            break;
        case ZYDIS_MNEMONIC_JMP:
            handled = EmitJmp(insn, ops, next_rip, c);
            if (handled)
                emitted_terminator = true;
            break;
        case ZYDIS_MNEMONIC_CALL:
            handled = EmitCall(insn, ops, next_rip, c);
            if (handled)
                emitted_terminator = true;
            break;
        // All conditional jumps go through EmitJcc.
        case ZYDIS_MNEMONIC_JZ:
        case ZYDIS_MNEMONIC_JNZ:
        case ZYDIS_MNEMONIC_JS:
        case ZYDIS_MNEMONIC_JNS:
        case ZYDIS_MNEMONIC_JO:
        case ZYDIS_MNEMONIC_JNO:
        case ZYDIS_MNEMONIC_JP:
        case ZYDIS_MNEMONIC_JNP:
        case ZYDIS_MNEMONIC_JB:
        case ZYDIS_MNEMONIC_JNB:
        case ZYDIS_MNEMONIC_JBE:
        case ZYDIS_MNEMONIC_JNBE:
        case ZYDIS_MNEMONIC_JL:
        case ZYDIS_MNEMONIC_JNL:
        case ZYDIS_MNEMONIC_JLE:
        case ZYDIS_MNEMONIC_JNLE:
            handled = EmitJcc(insn, ops, next_rip, c);
            if (handled)
                emitted_terminator = true;
            break;

        // AVX VEX-encoded 128/256-bit vector instructions. These
        // operate on GuestState::ymm[] via 64-bit GPR transfers
        // (see EmitVmovups / EmitVecBitXor for the design notes).
        case ZYDIS_MNEMONIC_VMOVUPS:
        case ZYDIS_MNEMONIC_VMOVDQU:
        case ZYDIS_MNEMONIC_VMOVDQA:
        case ZYDIS_MNEMONIC_VMOVAPS:
            handled = EmitVmovups(insn, ops, next_rip, c);
            break;
        case ZYDIS_MNEMONIC_VMOVNTDQ:
            handled = EmitVmovntdq(insn, ops, next_rip, c);
            break;
        case ZYDIS_MNEMONIC_SFENCE:
            handled = EmitSfence(insn, c);
            break;
        case ZYDIS_MNEMONIC_VXORPS:
        case ZYDIS_MNEMONIC_VPXOR:
            handled = EmitVecBitXor(insn, ops, c);
            break;
        case ZYDIS_MNEMONIC_VMOVD:
            handled = EmitVmovd(insn, ops, c);
            break;
        case ZYDIS_MNEMONIC_VMOVQ:
            handled = EmitVmovq(insn, ops, c);
            break;
        case ZYDIS_MNEMONIC_VPSHUFB:
            handled = EmitVpshufb(insn, ops, c);
            break;
        case ZYDIS_MNEMONIC_VPTEST:
            handled = EmitVptest(insn, ops, c);
            break;
        case ZYDIS_MNEMONIC_VPCMPISTRI:
            handled = EmitVpcmpistri(insn, ops, c);
            break;
        case ZYDIS_MNEMONIC_VPHADDD:
            handled = EmitVphaddd(insn, ops, c);
            break;
        case ZYDIS_MNEMONIC_VPUNPCKLQDQ:
            handled = EmitVpunpcklqdq(insn, ops, c);
            break;
        case ZYDIS_MNEMONIC_XADD:
            handled = EmitXadd(insn, ops, next_rip, c);
            break;
        case ZYDIS_MNEMONIC_VPCMPEQB:
            handled = EmitVpcmpeqb(insn, ops, c);
            break;
        case ZYDIS_MNEMONIC_ANDN:
            handled = EmitAndn(insn, ops, c);
            break;
        case ZYDIS_MNEMONIC_BEXTR:
            handled = EmitBextr(insn, ops, next_rip, c);
            break;
        case ZYDIS_MNEMONIC_POPCNT:
            handled = EmitPopcnt(insn, ops, next_rip, c);
            break;
        case ZYDIS_MNEMONIC_CPUID:
            handled = EmitCpuid(insn, c);
            break;
        case ZYDIS_MNEMONIC_XGETBV:
            handled = EmitXgetbv(insn, c);
            break;
        case ZYDIS_MNEMONIC_CMPXCHG:
            handled = EmitCmpxchg(insn, ops, next_rip, c);
            break;

        // SETcc family: write 0/1 to dst byte based on flags.
        // All sixteen variants share EmitSetcc via SetccToJcc.
        case ZYDIS_MNEMONIC_SETZ:
        case ZYDIS_MNEMONIC_SETNZ:
        case ZYDIS_MNEMONIC_SETS:
        case ZYDIS_MNEMONIC_SETNS:
        case ZYDIS_MNEMONIC_SETO:
        case ZYDIS_MNEMONIC_SETNO:
        case ZYDIS_MNEMONIC_SETP:
        case ZYDIS_MNEMONIC_SETNP:
        case ZYDIS_MNEMONIC_SETB:
        case ZYDIS_MNEMONIC_SETNB:
        case ZYDIS_MNEMONIC_SETBE:
        case ZYDIS_MNEMONIC_SETNBE:
        case ZYDIS_MNEMONIC_SETL:
        case ZYDIS_MNEMONIC_SETNL:
        case ZYDIS_MNEMONIC_SETLE:
        case ZYDIS_MNEMONIC_SETNLE:
            handled = EmitSetcc(insn, ops, next_rip, c);
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
                case ZYDIS_OPERAND_TYPE_REGISTER:
                    return "reg";
                case ZYDIS_OPERAND_TYPE_MEMORY:
                    return "mem";
                case ZYDIS_OPERAND_TYPE_IMMEDIATE:
                    return "imm";
                case ZYDIS_OPERAND_TYPE_POINTER:
                    return "ptr";
                default:
                    return "?";
                }
            };
            std::fprintf(
                stderr,
                "[lifter] unsupported insn at 0x%llx (mnemonic=%s, "
                "width=%u, length=%u, ops=%s,%s)\n",
                static_cast<unsigned long long>(rip), ZydisMnemonicGetString(insn.mnemonic),
                static_cast<unsigned>(insn.operand_width), static_cast<unsigned>(insn.length),
                op_type_name(ops[0].type), op_type_name(ops[1].type));
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
        if (emitted_terminator)
            break;
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
    bytes_emitted_ += emitted;
    ++blocks_compiled_;

    std::fprintf(
        stderr, "[lifter] compiled block 0x%llx -> %p (%llu guest bytes -> %llu host bytes)\n",
        static_cast<unsigned long long>(guest_rip), static_cast<void*>(code_buf),
        static_cast<unsigned long long>(rip - guest_rip), static_cast<unsigned long long>(emitted));
    std::fflush(stderr);

    return code_buf;
}

} // namespace Core::Runtime
