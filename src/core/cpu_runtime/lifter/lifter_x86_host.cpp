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
    // Segment overrides other than the standard DS/SS aren't
    // supported. shadPS4 guest code rarely uses FS/GS, and when it
    // does it's for TLS via specific helper sequences we'd lift
    // specially. CS/ES are flat in long mode.
    if (mem.segment != ZYDIS_REGISTER_DS &&
        mem.segment != ZYDIS_REGISTER_SS &&
        mem.segment != ZYDIS_REGISTER_CS &&
        mem.segment != ZYDIS_REGISTER_ES) {
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
        if (has_index) return false;  // RIP-relative with index is not a thing
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
    if (insn.operand_width != 64) return false;  // 32-bit LEA deferred
    if (ops[0].type != ZYDIS_OPERAND_TYPE_REGISTER) return false;
    if (ops[1].type != ZYDIS_OPERAND_TYPE_MEMORY) return false;

    const int dst_idx = ZydisGprToIndex(ops[0].reg.value);
    if (dst_idx < 0) return false;

    if (!EmitEffectiveAddress(ops[1].mem, next_rip, c)) return false;
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
constexpr u64 AllArith = CF | PF | ZF | SF | OF;
}  // namespace RflagsBits

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
/// Clobbers r8, r9, r10. Writes rflags.
void EmitFlagsFromSubtract(Xbyak::CodeGenerator& c) {
    EmitClearArithFlags(c);

    // ZF: result == 0
    c.test(rax, rax);
    c.setz(r8b);
    c.movzx(r8, r8b);
    c.shl(r8, 6);                       // ZF at bit 6
    c.or_(qword[r13 + Offsets::Rflags], r8);

    // SF: result >> 63
    c.mov(r8, rax);
    c.shr(r8, 63);
    c.shl(r8, 7);                       // SF at bit 7
    c.or_(qword[r13 + Offsets::Rflags], r8);

    // CF for subtract: lhs < rhs (unsigned).
    c.cmp(rcx, rdx);
    c.setb(r8b);                        // setb = "set if below", unsigned <
    c.movzx(r8, r8b);
    // CF is at bit 0; no shift needed.
    c.or_(qword[r13 + Offsets::Rflags], r8);

    // OF for subtract: ((lhs ^ rhs) & (lhs ^ result)) >> 63
    c.mov(r8, rcx);
    c.xor_(r8, rdx);                    // r8 = lhs ^ rhs
    c.mov(r9, rcx);
    c.xor_(r9, rax);                    // r9 = lhs ^ result
    c.and_(r8, r9);
    c.shr(r8, 63);
    c.shl(r8, 11);                      // OF at bit 11
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
    c.xor_(r8, rdx);                    // r8 = lhs ^ rhs
    c.not_(r8);                         // r8 = ~(lhs ^ rhs)
    c.mov(r9, rcx);
    c.xor_(r9, rax);                    // r9 = lhs ^ result
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

/// ADD r64, r64 / ADD r64, imm32-sx.
/// Writes ZF/SF/CF/OF/PF to state.rflags (eager flag computation).
bool EmitAdd(const ZydisDecodedInstruction& insn,
             const ZydisDecodedOperand* ops,
             Xbyak::CodeGenerator& c) {
    const auto& dst = ops[0];
    const auto& src = ops[1];

    if (dst.type != ZYDIS_OPERAND_TYPE_REGISTER) return false;
    const int dst_idx = ZydisGprToIndex(dst.reg.value);
    if (dst_idx < 0) return false;
    if (insn.operand_width != 64) return false;

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
/// Also SUB r64, r64. Writes ZF/SF/CF/OF/PF.
bool EmitSub(const ZydisDecodedInstruction& insn,
             const ZydisDecodedOperand* ops,
             Xbyak::CodeGenerator& c) {
    if (insn.operand_width != 64) return false;
    const auto& dst = ops[0];
    const auto& src = ops[1];

    if (dst.type != ZYDIS_OPERAND_TYPE_REGISTER) return false;
    const int dst_idx = ZydisGprToIndex(dst.reg.value);
    if (dst_idx < 0) return false;

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
              Xbyak::CodeGenerator& c) {
    if (insn.operand_width != 64) return false;
    const auto& lhs_op = ops[0];
    const auto& rhs_op = ops[1];

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
             Xbyak::CodeGenerator& c) {
    const auto& dst = ops[0];
    const auto& src = ops[1];

    if (dst.type != ZYDIS_OPERAND_TYPE_REGISTER) return false;
    if (src.type != ZYDIS_OPERAND_TYPE_REGISTER) return false;
    const int dst_idx = ZydisGprToIndex(dst.reg.value);
    const int src_idx = ZydisGprToIndex(src.reg.value);
    if (dst_idx < 0 || src_idx < 0) return false;

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

/// AND — bitwise and. Supported: r64,r64; r32,r32; r32,imm.
bool EmitAnd(const ZydisDecodedInstruction& insn,
             const ZydisDecodedOperand* ops,
             Xbyak::CodeGenerator& c) {
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
bool EmitNot(const ZydisDecodedInstruction& insn,
             const ZydisDecodedOperand* ops,
             Xbyak::CodeGenerator& c) {
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
             Xbyak::CodeGenerator& c) {
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
bool EmitInc(const ZydisDecodedInstruction& insn,
             const ZydisDecodedOperand* ops,
             Xbyak::CodeGenerator& c) {
    if (insn.operand_width != 64 && insn.operand_width != 32) return false;
    if (ops[0].type != ZYDIS_OPERAND_TYPE_REGISTER) return false;
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
    c.btr(r11, 0);                                       // clear bit 0 (CF)
    c.or_(r11, r10);                                     // OR in saved CF
    c.mov(qword[r13 + Offsets::Rflags], r11);
    return true;
}

/// DEC r64 — subtract 1, preserve CF.
bool EmitDec(const ZydisDecodedInstruction& insn,
             const ZydisDecodedOperand* ops,
             Xbyak::CodeGenerator& c) {
    if (insn.operand_width != 64) return false;
    if (ops[0].type != ZYDIS_OPERAND_TYPE_REGISTER) return false;
    const int idx = ZydisGprToIndex(ops[0].reg.value);
    if (idx < 0) return false;

    c.mov(r10, qword[r13 + Offsets::Rflags]);
    c.and_(r10, 0x1);

    c.mov(rcx, qword[r13 + GprOffset(idx)]);
    c.mov(rdx, 1);
    c.mov(rax, rcx);
    c.sub(rax, rdx);
    c.mov(qword[r13 + GprOffset(idx)], rax);
    EmitFlagsFromSubtract(c);

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
bool EmitBt(const ZydisDecodedInstruction& insn,
            const ZydisDecodedOperand* ops,
            Xbyak::CodeGenerator& c) {
    if (insn.operand_width != 64) return false;
    if (ops[0].type != ZYDIS_OPERAND_TYPE_REGISTER) return false;
    if (ops[1].type != ZYDIS_OPERAND_TYPE_REGISTER) return false;
    const int dst_idx = ZydisGprToIndex(ops[0].reg.value);
    const int src_idx = ZydisGprToIndex(ops[1].reg.value);
    if (dst_idx < 0 || src_idx < 0) return false;

    // Load value and bit index.
    c.mov(rax, qword[r13 + GprOffset(dst_idx)]);
    c.mov(rcx, qword[r13 + GprOffset(src_idx)]);
    // Host BT with reg-src already masks by opsize-1 (i.e. cl & 63
    // for 64-bit). No explicit `and rcx, 63` needed.
    c.bt(rax, rcx);                                 // host CF := bit N of rax

    // Capture host CF into r11, then merge into guest rflags.
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
        const int src_idx = ZydisGprToIndex(ops[1].reg.value);
        if (src_idx < 0) return false;
        if (src_size == 8) {
            c.movzx(rax, byte[r13 + GprOffset(src_idx)]);
        } else {
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
    if (insn.operand_width != 64) return false;
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
    c.test(rcx, rcx);
    c.cmovnz(r9, r8);
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
bool EmitShift64(const ZydisDecodedInstruction& insn,
                 const ZydisDecodedOperand* ops,
                 Xbyak::CodeGenerator& c,
                 ShiftKind kind) {
    if (insn.operand_width != 64) return false;
    if (ops[0].type != ZYDIS_OPERAND_TYPE_REGISTER) return false;
    const int dst_idx = ZydisGprToIndex(ops[0].reg.value);
    if (dst_idx < 0) return false;

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
        case ShiftKind::Shl: c.shl(rax, cl); break;
        case ShiftKind::Shr: c.shr(rax, cl); break;
        case ShiftKind::Sar: c.sar(rax, cl); break;
    }

    // Capture host → guest.
    c.pushfq();
    c.pop(rdx);
    c.mov(qword[r13 + Offsets::Rflags], rdx);

    // Store the shifted value back.
    c.mov(qword[r13 + GprOffset(dst_idx)], rax);
    return true;
}

// =============================================================================
// ROL / ROR — rotates.
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
                  Xbyak::CodeGenerator& c,
                  RotateKind kind) {
    if (insn.operand_width != 64) return false;
    if (ops[0].type != ZYDIS_OPERAND_TYPE_REGISTER) return false;
    const int dst_idx = ZydisGprToIndex(ops[0].reg.value);
    if (dst_idx < 0) return false;

    // Shift count from imm or guest CL — identical to EmitShift64.
    if (ops[1].type == ZYDIS_OPERAND_TYPE_IMMEDIATE) {
        c.mov(cl, static_cast<u8>(ops[1].imm.value.u & 0xFF));
    } else if (ops[1].type == ZYDIS_OPERAND_TYPE_REGISTER) {
        if (ops[1].reg.value != ZYDIS_REGISTER_CL) return false;
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
        case RotateKind::Rol: c.rol(rax, cl); break;
        case RotateKind::Ror: c.ror(rax, cl); break;
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
    c.push(r8);
    c.popfq();

    c.imul(rcx);  // implicit rax operand; rdx:rax = rax * rcx

    c.pushfq();
    c.pop(r8);
    c.mov(qword[r13 + Offsets::Rflags], r8);

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
    if (insn.operand_width != 64) return false;
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
    c.push(r8);
    c.popfq();

    c.imul(rax, rcx);  // rax = rax * rcx, low 64 bits only

    c.pushfq();
    c.pop(r8);
    c.mov(qword[r13 + Offsets::Rflags], r8);

    c.mov(qword[r13 + GprOffset(dst_idx)], rax);
    return true;
}

/// 3-op IMUL: dst = src * imm.
bool EmitImul3Op(const ZydisDecodedInstruction& insn,
                 const ZydisDecodedOperand* ops,
                 u64 next_rip,
                 Xbyak::CodeGenerator& c) {
    if (insn.operand_width != 64) return false;
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
    c.push(r8);
    c.popfq();

    c.imul(rax, rcx);  // rax = imm * src

    c.pushfq();
    c.pop(r8);
    c.mov(qword[r13 + Offsets::Rflags], r8);

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
    c.push(r8);
    c.popfq();

    switch (kind) {
        case AdcSbbKind::Adc: c.adc(rax, rcx); break;
        case AdcSbbKind::Sbb: c.sbb(rax, rcx); break;
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

bool EmitNarrowArith8(const ZydisDecodedInstruction& insn,
                      const ZydisDecodedOperand* ops,
                      u64 next_rip,
                      Xbyak::CodeGenerator& c,
                      NarrowArithKind kind) {
    if (insn.operand_width != 8) return false;

    // ----- Memory destination/lhs — Cmp/Test only -----
    //
    // Common in compiled code as "test byte ptr [reg+disp], imm8"
    // (boolean field check) and "cmp byte ptr [mem], imm8". Cmp/Test
    // discard the result so there's no writeback, and rdx is free to
    // repurpose for the flag round-trip after we no longer need the
    // address.
    if (ops[0].type == ZYDIS_OPERAND_TYPE_MEMORY) {
        if (kind != NarrowArithKind::Cmp && kind != NarrowArithKind::Test) {
            return false;
        }
        if (!EmitEffectiveAddress(ops[0].mem, next_rip, c)) return false;
        // rdx = effective address. Load 8-bit lhs into al.
        c.mov(al, byte[rdx]);

        // Load rhs into cl. Mem-mem doesn't exist for cmp/test.
        if (ops[1].type == ZYDIS_OPERAND_TYPE_REGISTER) {
            const int src_idx = ZydisGprToIndex(ops[1].reg.value);
            if (src_idx < 0) return false;
            c.mov(cl, byte[r13 + GprOffset(src_idx)]);
        } else if (ops[1].type == ZYDIS_OPERAND_TYPE_IMMEDIATE) {
            c.mov(cl, static_cast<u8>(ops[1].imm.value.u & 0xFF));
        } else {
            return false;
        }

        // Flag round-trip through host CPU (so it computes correct
        // 8-bit-width flags). rdx is free now — address is no longer
        // needed for a writeback.
        c.mov(rdx, qword[r13 + Offsets::Rflags]);
        c.push(rdx);
        c.popfq();

        switch (kind) {
            case NarrowArithKind::Cmp:  c.cmp(al, cl); break;
            case NarrowArithKind::Test: c.test(al, cl); break;
            default: return false;  // unreachable: filtered above
        }

        c.pushfq();
        c.pop(rdx);
        c.mov(qword[r13 + Offsets::Rflags], rdx);
        return true;
    }

    if (ops[0].type != ZYDIS_OPERAND_TYPE_REGISTER) return false;
    const int dst_idx = ZydisGprToIndex(ops[0].reg.value);
    if (dst_idx < 0) return false;

    // Load dst byte into al.
    c.mov(al, byte[r13 + GprOffset(dst_idx)]);

    // Load src byte into cl. Three source forms:
    //   - reg: read from the guest GPR slot.
    //   - imm: literal byte.
    //   - mem: compute guest effective address into rdx, deref to cl.
    //
    // For mem source, EmitEffectiveAddress trashes rdx (and uses
    // rax/rcx as transients); we use rdx and cl AFTER, which is
    // safe — al is already loaded and cl is what we're populating.
    // The subsequent flag round-trip reuses rdx, also safe because
    // we no longer need the address by then.
    if (ops[1].type == ZYDIS_OPERAND_TYPE_REGISTER) {
        const int src_idx = ZydisGprToIndex(ops[1].reg.value);
        if (src_idx < 0) return false;
        c.mov(cl, byte[r13 + GprOffset(src_idx)]);
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
    }

    c.pushfq();
    c.pop(rdx);
    c.mov(qword[r13 + Offsets::Rflags], rdx);

    // CMP and TEST discard the result — only the others write back.
    // Narrow store preserves upper 56 bits per x86-64 semantics.
    if (kind != NarrowArithKind::Cmp && kind != NarrowArithKind::Test) {
        c.mov(byte[r13 + GprOffset(dst_idx)], al);
    }
    return true;
}

bool EmitNarrowArith16(const ZydisDecodedInstruction& insn,
                       const ZydisDecodedOperand* ops,
                       u64 next_rip,
                       Xbyak::CodeGenerator& c,
                       NarrowArithKind kind) {
    if (insn.operand_width != 16) return false;
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
    if (ops[0].type == ZYDIS_OPERAND_TYPE_MEMORY) {
        if (kind != NarrowArithKind::Cmp && kind != NarrowArithKind::Test) {
            return false;
        }
        // rdx = effective address. EmitEffectiveAddress may clobber
        // rax (used as scratch for index*scale); rcx is preserved.
        if (!EmitEffectiveAddress(ops[0].mem, next_rip, c)) return false;
        // Load the 32-bit lhs from memory into eax (zero-extends rax,
        // but that's fine — we only ever use eax below).
        c.mov(eax, dword[rdx]);

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

        // Flag round-trip. We re-use rdx for the rflags load — the
        // address is no longer needed (no writeback path here).
        c.mov(rdx, qword[r13 + Offsets::Rflags]);
        c.push(rdx);
        c.popfq();

        switch (kind) {
            case NarrowArithKind::Cmp:  c.cmp(eax, ecx); break;
            case NarrowArithKind::Test: c.test(eax, ecx); break;
            default: return false;  // unreachable: filtered above
        }

        c.pushfq();
        c.pop(rdx);
        c.mov(qword[r13 + Offsets::Rflags], rdx);
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

bool EmitMov8(const ZydisDecodedInstruction&,
              const ZydisDecodedOperand* ops,
              u64 next_rip,
              Xbyak::CodeGenerator& c) {
    const auto& dst = ops[0];
    const auto& src = ops[1];

    // Register destination.
    if (dst.type == ZYDIS_OPERAND_TYPE_REGISTER) {
        const int dst_idx = ZydisGprToIndex(dst.reg.value);
        if (dst_idx < 0) return false;

        if (src.type == ZYDIS_OPERAND_TYPE_REGISTER) {
            // mov r8, r8 — read low byte of src slot, write low byte
            // of dst slot. Upper bits of dst preserved by `byte` store.
            const int src_idx = ZydisGprToIndex(src.reg.value);
            if (src_idx < 0) return false;
            c.mov(al, byte[r13 + GprOffset(src_idx)]);
            c.mov(byte[r13 + GprOffset(dst_idx)], al);
            return true;
        }
        if (src.type == ZYDIS_OPERAND_TYPE_IMMEDIATE) {
            // mov r8, imm8 — store immediate byte into low byte of slot.
            c.mov(byte[r13 + GprOffset(dst_idx)],
                  static_cast<u8>(src.imm.value.u & 0xFF));
            return true;
        }
        if (src.type == ZYDIS_OPERAND_TYPE_MEMORY) {
            // mov r8, byte[mem] — compute address (clobbers rax),
            // then load byte and write low byte of dst slot.
            // We use cl for the loaded byte because rcx is preserved
            // by EmitEffectiveAddress; al would be valid too but is
            // a scratch the helper actively uses.
            if (!EmitEffectiveAddress(src.mem, next_rip, c)) return false;
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
            if (src_idx < 0) return false;
            if (!EmitEffectiveAddress(dst.mem, next_rip, c)) return false;
            c.mov(cl, byte[r13 + GprOffset(src_idx)]);
            c.mov(byte[rdx], cl);
            return true;
        }
        if (src.type == ZYDIS_OPERAND_TYPE_IMMEDIATE) {
            // mov byte[mem], imm8.
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
              Xbyak::CodeGenerator& c) {
    if (insn.operand_width != 64) return false;
    const auto& src = ops[0];
    if (src.type != ZYDIS_OPERAND_TYPE_REGISTER) return false;
    const int src_idx = ZydisGprToIndex(src.reg.value);
    if (src_idx < 0) return false;

    // rax = src value, rdx = guest_rsp, decrement, store.
    c.mov(rax, qword[r13 + GprOffset(src_idx)]);
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
///   5. Jump to gateway exit stub via r14. The test harness then
///      inspects state.rip to see where the guest returned to.
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
        case ZYDIS_MNEMONIC_JNLE: {                      // JNLE / JG: ZF=0 AND SF == OF
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
bool EmitVmovups(const ZydisDecodedInstruction& insn,
                 const ZydisDecodedOperand* ops,
                 u64 next_rip,
                 Xbyak::CodeGenerator& c) {
    if (insn.mnemonic != ZYDIS_MNEMONIC_VMOVUPS &&
        insn.mnemonic != ZYDIS_MNEMONIC_VMOVDQU) {
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
    if (ops[0].type == ZYDIS_OPERAND_TYPE_MEMORY &&
        ops[1].type == ZYDIS_OPERAND_TYPE_REGISTER) {
        const int src_idx = ZydisVecToIndex(ops[1].reg.value);
        if (src_idx < 0) return false;
        if (!EmitEffectiveAddress(ops[0].mem, next_rip, c)) return false;
        // rdx = dst address; rax is the transfer register.
        for (int i = 0; i < chunks; ++i) {
            c.mov(rax, qword[r13 + YmmChunkOffset(src_idx, i)]);
            c.mov(qword[rdx + i * 8], rax);
        }
        return true;
    }

    // reg ← reg
    if (ops[0].type == ZYDIS_OPERAND_TYPE_REGISTER &&
        ops[1].type == ZYDIS_OPERAND_TYPE_REGISTER) {
        const int dst_idx = ZydisVecToIndex(ops[0].reg.value);
        const int src_idx = ZydisVecToIndex(ops[1].reg.value);
        if (dst_idx < 0 || src_idx < 0) return false;
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

/// VXORPS / VPXOR — three-operand VEX form: dst = src1 XOR src2.
/// Both are bitwise XOR; they differ only in whether they're classed
/// as float (VXORPS) or int (VPXOR), which we don't care about
/// because no flags are written either way.
///
/// Supported operand shape: reg, reg, reg (the common case in the
/// game's prologue zeroing idiom: `vxorps xmm0, xmm0, xmm0`).
/// Reg + reg + mem can be added when we see it.
bool EmitVecBitXor(const ZydisDecodedInstruction& insn,
                   const ZydisDecodedOperand* ops,
                   Xbyak::CodeGenerator& c) {
    if (insn.mnemonic != ZYDIS_MNEMONIC_VXORPS &&
        insn.mnemonic != ZYDIS_MNEMONIC_VPXOR) {
        return false;
    }
    // Strictly 3-operand reg form for this initial cut.
    if (ops[0].type != ZYDIS_OPERAND_TYPE_REGISTER) return false;
    if (ops[1].type != ZYDIS_OPERAND_TYPE_REGISTER) return false;
    if (ops[2].type != ZYDIS_OPERAND_TYPE_REGISTER) return false;

    const int dst_idx  = ZydisVecToIndex(ops[0].reg.value);
    const int src1_idx = ZydisVecToIndex(ops[1].reg.value);
    const int src2_idx = ZydisVecToIndex(ops[2].reg.value);
    if (dst_idx < 0 || src1_idx < 0 || src2_idx < 0) return false;

    const int vec_bits = ops[0].size;
    if (vec_bits != 128 && vec_bits != 256) return false;
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
        const auto status = ZydisDecoderDecodeFull(
            &decoder, reinterpret_cast<const void*>(rip), 15,
            &insn, ops);
        std::fprintf(stderr, "[lifter] decoded at 0x%llx ok=%d mnemonic=%s\n",
                     static_cast<unsigned long long>(rip),
                     ZYAN_SUCCESS(status) ? 1 : 0,
                     ZYAN_SUCCESS(status)
                         ? ZydisMnemonicGetString(insn.mnemonic)
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
            case ZYDIS_MNEMONIC_LEA:    handled = EmitLea(insn, ops, next_rip, c); break;
            case ZYDIS_MNEMONIC_MOVSXD: handled = EmitMovsxd(insn, ops, next_rip, c); break;
            case ZYDIS_MNEMONIC_ADD:
                // 64- and 32-bit go through the existing path (eager
                // flag computation); 8- and 16-bit use the round-trip
                // technique via EmitNarrowArith.
                if (insn.operand_width == 8) {
                    handled = EmitNarrowArith8(insn, ops, next_rip, c, NarrowArithKind::Add);
                } else if (insn.operand_width == 16) {
                    handled = EmitNarrowArith16(insn, ops, next_rip, c, NarrowArithKind::Add);
                } else {
                    handled = EmitAdd(insn, ops, c);
                }
                break;
            case ZYDIS_MNEMONIC_SUB:
                if (insn.operand_width == 8) {
                    handled = EmitNarrowArith8(insn, ops, next_rip, c, NarrowArithKind::Sub);
                } else if (insn.operand_width == 16) {
                    handled = EmitNarrowArith16(insn, ops, next_rip, c, NarrowArithKind::Sub);
                } else {
                    handled = EmitSub(insn, ops, c);
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
            case ZYDIS_MNEMONIC_ADC: handled = EmitAdcSbb64(insn, ops, c, AdcSbbKind::Adc); break;
            case ZYDIS_MNEMONIC_SBB: handled = EmitAdcSbb64(insn, ops, c, AdcSbbKind::Sbb); break;

            // Function epilogue shorthand: mov rsp, rbp; pop rbp.
            case ZYDIS_MNEMONIC_LEAVE: handled = EmitLeave(c); break;
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
                } else {
                    handled = EmitXor(insn, ops, c);
                }
                break;
            case ZYDIS_MNEMONIC_AND:
                if (insn.operand_width == 8) {
                    handled = EmitNarrowArith8(insn, ops, next_rip, c, NarrowArithKind::And);
                } else if (insn.operand_width == 16) {
                    handled = EmitNarrowArith16(insn, ops, next_rip, c, NarrowArithKind::And);
                } else {
                    handled = EmitAnd(insn, ops, c);
                }
                break;
            case ZYDIS_MNEMONIC_OR:
                if (insn.operand_width == 8) {
                    handled = EmitNarrowArith8(insn, ops, next_rip, c, NarrowArithKind::Or);
                } else if (insn.operand_width == 16) {
                    handled = EmitNarrowArith16(insn, ops, next_rip, c, NarrowArithKind::Or);
                } else {
                    handled = EmitOr(insn, ops, next_rip, c);
                }
                break;
            case ZYDIS_MNEMONIC_NOT:  handled = EmitNot(insn, ops, c); break;
            case ZYDIS_MNEMONIC_NEG:  handled = EmitNeg(insn, ops, c); break;
            case ZYDIS_MNEMONIC_INC:  handled = EmitInc(insn, ops, c); break;
            case ZYDIS_MNEMONIC_DEC:  handled = EmitDec(insn, ops, c); break;
            case ZYDIS_MNEMONIC_BT:   handled = EmitBt(insn, ops, c); break;
            case ZYDIS_MNEMONIC_MOVZX: handled = EmitMovzx(insn, ops, next_rip, c); break;

            // Shifts. All three use the same emit with a kind tag.
            case ZYDIS_MNEMONIC_SHL: handled = EmitShift64(insn, ops, c, ShiftKind::Shl); break;
            case ZYDIS_MNEMONIC_SHR: handled = EmitShift64(insn, ops, c, ShiftKind::Shr); break;
            case ZYDIS_MNEMONIC_SAR: handled = EmitShift64(insn, ops, c, ShiftKind::Sar); break;

            // Rotates. Same shape as shifts.
            case ZYDIS_MNEMONIC_ROL: handled = EmitRotate64(insn, ops, c, RotateKind::Rol); break;
            case ZYDIS_MNEMONIC_ROR: handled = EmitRotate64(insn, ops, c, RotateKind::Ror); break;

            // Multiplication. EmitImul dispatches by operand_count_visible.
            case ZYDIS_MNEMONIC_IMUL: handled = EmitImul(insn, ops, next_rip, c); break;

            // Sign-extension family. No operands; operate on RAX/RDX.
            case ZYDIS_MNEMONIC_CWDE: handled = EmitCwde(c); break;
            case ZYDIS_MNEMONIC_CDQE: handled = EmitCdqe(c); break;
            case ZYDIS_MNEMONIC_CDQ:  handled = EmitCdq(c);  break;
            case ZYDIS_MNEMONIC_CQO:  handled = EmitCqo(c);  break;

            // Direct carry-flag manipulation.
            case ZYDIS_MNEMONIC_STC: handled = EmitStc(c); break;
            case ZYDIS_MNEMONIC_CLC: handled = EmitClc(c); break;
            case ZYDIS_MNEMONIC_CMC: handled = EmitCmc(c); break;

            // NOP — no semantic effect, just consume the bytes.
            // Common forms: 90 (1-byte), 66 90 (2-byte),
            // 0F 1F /0 (multi-byte padding). All decode as NOP.
            case ZYDIS_MNEMONIC_NOP: handled = true; break;

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

            case ZYDIS_MNEMONIC_PUSH: handled = EmitPush(insn, ops, c); break;
            case ZYDIS_MNEMONIC_POP:  handled = EmitPop(insn, ops, c); break;
            case ZYDIS_MNEMONIC_RET:
                handled = EmitRet(insn, ops, c);
                if (handled) emitted_terminator = true;
                break;
            case ZYDIS_MNEMONIC_JMP:
                handled = EmitJmp(insn, ops, next_rip, c);
                if (handled) emitted_terminator = true;
                break;
            case ZYDIS_MNEMONIC_CALL:
                handled = EmitCall(insn, ops, next_rip, c);
                if (handled) emitted_terminator = true;
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
                if (handled) emitted_terminator = true;
                break;

            // AVX VEX-encoded 128/256-bit vector instructions. These
            // operate on GuestState::ymm[] via 64-bit GPR transfers
            // (see EmitVmovups / EmitVecBitXor for the design notes).
            case ZYDIS_MNEMONIC_VMOVUPS:
            case ZYDIS_MNEMONIC_VMOVDQU:
                handled = EmitVmovups(insn, ops, next_rip, c);
                break;
            case ZYDIS_MNEMONIC_VXORPS:
            case ZYDIS_MNEMONIC_VPXOR:
                handled = EmitVecBitXor(insn, ops, c);
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

    std::fprintf(stderr,
                 "[lifter] compiled block 0x%llx -> %p (%llu guest bytes -> %llu host bytes)\n",
                 static_cast<unsigned long long>(guest_rip),
                 static_cast<void*>(code_buf),
                 static_cast<unsigned long long>(rip - guest_rip),
                 static_cast<unsigned long long>(emitted));
    std::fflush(stderr);

    return code_buf;
}

} // namespace Core::Runtime
