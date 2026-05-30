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
///
/// Sized for vector-heavy basic blocks: each AVX instruction expands
/// to 50-100 host bytes (vmovdqu loads + host op + vmovdqu store +
/// VEX-128 upper-zero ~20B). A 150-instruction vector block at the
/// pessimistic 100B/insn = 15000 bytes, so 16384 is comfortable.
///
/// Tradeoff: each block pre-reserves this many bytes from the code
/// cache (Allocate is fixed-size; unused tail is wasted until the
/// cache is flushed). Cache utilization degrades linearly with this
/// cap. If utilization becomes a problem, the right fix is a
/// two-pass compile (estimate, then allocate exact size, then copy),
/// not further cap inflation.
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

    // 32-bit address-size detection. When the guest uses a 32-bit
    // address-size override (0x67 prefix), the base/index registers are
    // reported by Zydis as 32-bit GPRs (EAX-class, not RAX-class), and
    // x86 semantics require the WHOLE effective-address computation to
    // be done modulo 2^32 and zero-extended to 64 bits. If we instead
    // read the full 64-bit guest register slots (which we must — the
    // 32-bit regs alias the low halves), any garbage in the upper 32
    // bits leaks into the address, producing a wild ~TiB-range host
    // pointer instead of the intended <4 GiB address. This was the
    // cause of an in-JIT-code access violation at a (1 TiB ceiling +
    // ~12 GiB) fault address: the low 32 bits were the real target, the
    // high bits were leaked register garbage. We detect 32-bit mode
    // from the operand registers themselves (no need to thread
    // address_width through all callers) and truncate the result.
    const bool addr32 =
        (has_base && mem.base != ZYDIS_REGISTER_RIP &&
         ZydisRegisterGetClass(mem.base) == ZYDIS_REGCLASS_GPR32) ||
        (has_index && ZydisRegisterGetClass(mem.index) == ZYDIS_REGCLASS_GPR32);

    // Emits the 32-bit truncation (zero-extend low 32 of rdx into rdx)
    // when 32-bit addressing is in effect. `mov edx, edx` clears the
    // upper 32 bits — exactly the x86 "address mod 2^32" rule.
    auto truncate_if_addr32 = [&]() {
        if (addr32)
            c.mov(edx, edx);
    };

    // RIP-relative: base == RIP, no index. Address = next_rip + disp.
    // We constant-fold this into a single mov. (EIP-relative — 32-bit
    // mode — would also land here via base==EIP; ZydisRegisterGetClass
    // puts EIP in the IP class, not GPR32, so addr32 stays false and we
    // fold the full value. EIP-relative is vanishingly rare in PS4 code
    // and a correct fold of next_rip+disp is already within range.)
    if (has_base && (mem.base == ZYDIS_REGISTER_RIP || mem.base == ZYDIS_REGISTER_EIP)) {
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

    // Apply the modulo-2^32 truncation for 32-bit address mode. Done
    // AFTER base+index*scale+disp so the wraparound matches hardware
    // (the entire sum wraps at 32 bits, not each term).
    truncate_if_addr32();

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
// NOTE: this AF omission is NOT an Intel/AMD undefined-flag case.
// AF is *fully defined and identical on both vendors* for ADD/SUB/
// ADC/SBB/CMP/NEG/INC/DEC (AF = carry/borrow out of bit 3). Skipping
// it is a deliberate correctness shortcut on the bet that no PS4
// title reads AF (it has no purpose outside the legacy BCD adjust
// instructions AAA/AAS/DAA/DAS, none of which exist in 64-bit mode).
// If a title ever depends on AF, the fix is: AF = ((lhs ^ rhs ^
// result) >> 4) & 1, valid for both add and subtract, on both
// vendors. The ARM64 backend may likewise skip AF for the same
// reason — this is not a vendor-divergence concern.
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
constexpr u64 AF = 1ULL << 4; // auxiliary carry (carry/borrow out of bit 3)
constexpr u64 ZF = 1ULL << 6;
constexpr u64 SF = 1ULL << 7;
constexpr u64 DF = 1ULL << 10; // direction flag (string-op direction)
constexpr u64 OF = 1ULL << 11;
// AF is included so EmitClearArithFlags wipes any stale AF before the per-op
// helpers OR in the freshly-computed value. ADD/SUB compute AF exactly
// (carry/borrow out of bit 3); bitwise ops leave it 0 (x86 marks AF undefined
// after AND/OR/XOR/TEST — 0 is a deterministic, conventional choice).
constexpr u64 AllArith = CF | PF | AF | ZF | SF | OF;
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

    // AF for subtract: borrow out of bit 3 = (lhs ^ rhs ^ result) bit 4.
    // Identical formula for add and subtract, vendor-independent.
    c.mov(r8, rcx);
    c.xor_(r8, rdx);
    c.xor_(r8, rax); // r8 = lhs ^ rhs ^ result
    c.shr(r8, 4);    // bring bit 4 down to bit 0
    c.and_(r8, 1);
    c.shl(r8, 4);    // AF at bit 4
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

    // AF for add: carry out of bit 3 = (lhs ^ rhs ^ result) bit 4.
    c.mov(r8, rcx);
    c.xor_(r8, rdx);
    c.xor_(r8, rax); // r8 = lhs ^ rhs ^ result
    c.shr(r8, 4);
    c.and_(r8, 1);
    c.shl(r8, 4);    // AF at bit 4
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

    // 64-bit ADD with memory destination and immediate source:
    // `add qword[mem], imm` (imm8/imm32 sign-extended). The RMW
    // counterpart of the reg-dst imm form — seen at libc 0x808187113
    // (an in-memory 64-bit accumulator/counter bumped by a constant).
    // Same r10-stash discipline as the mem-dst reg form: keep the
    // computed address in r10 so the flag helper can use rdx for rhs.
    if (insn.operand_width == 64 && dst.type == ZYDIS_OPERAND_TYPE_MEMORY &&
        src.type == ZYDIS_OPERAND_TYPE_IMMEDIATE) {
        if (!EmitEffectiveAddress(dst.mem, next_rip, c))
            return false;
        c.mov(r10, rdx);                  // r10 = address (survives helper)
        c.mov(rcx, qword[r10]);           // rcx = lhs (orig [mem])
        c.mov(rdx, src.imm.value.s);      // rdx = rhs (sign-extended imm)
        c.mov(rax, rcx);
        c.add(rax, rdx);
        c.mov(qword[r10], rax);           // store result
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

    // 64-bit SUB with memory destination and immediate source:
    // `sub qword[mem], imm` (imm8/imm32 sign-extended to 64 bits). The
    // gap from CUSA02394 at guest 0x8002173f5 — an in-memory counter /
    // pointer adjustment. Same RMW + r10-stash discipline as the
    // mem-dst+reg form above; the only change is materializing the
    // sign-extended immediate into rdx (the flag helper's rhs slot)
    // instead of reading a guest register. Zydis sign-extends the
    // encoded imm8/imm32 into imm.value.s, matching x86 semantics.
    if (dst.type == ZYDIS_OPERAND_TYPE_MEMORY && src.type == ZYDIS_OPERAND_TYPE_IMMEDIATE) {
        if (!EmitEffectiveAddress(dst.mem, next_rip, c))
            return false;
        c.mov(r10, rdx);            // r10 = addr (survives the load below)
        c.mov(rcx, qword[r10]);     // rcx = lhs ([mem])
        c.mov(rdx, src.imm.value.s); // rdx = sign-extended immediate (rhs)
        c.mov(rax, rcx);
        c.sub(rax, rdx);
        c.mov(qword[r10], rax);     // store result
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

/// TEST r64, r64 / TEST r64, imm32-sx / TEST qword[mem], r64.
/// Like AND but doesn't write the result back — only writes flags.
/// CF and OF are always 0 (per x86 spec).
bool EmitTest(const ZydisDecodedInstruction& insn, const ZydisDecodedOperand* ops,
              u64 next_rip, Xbyak::CodeGenerator& c) {
    if (insn.operand_width != 64)
        return false;
    const auto& lhs_op = ops[0];
    const auto& rhs_op = ops[1];

    // Memory-destination form: `test qword[mem], r64`. TEST computes
    // lhs & rhs purely for flags (no writeback), and AND is commutative,
    // so [mem] & reg gives identical flags to reg & [mem]. Gap from
    // CUSA02394 at guest 0x8001e1d30 (length 8 — SIB/disp32 address).
    // Load the memory operand into rax, AND with the register, set the
    // bitwise flags. rhs must be a register here (the imm-to-memory form
    // is opcode F7 /0 and would be reported with an IMMEDIATE rhs; we
    // handle the reg rhs that this gap uses).
    if (lhs_op.type == ZYDIS_OPERAND_TYPE_MEMORY) {
        if (rhs_op.type == ZYDIS_OPERAND_TYPE_REGISTER) {
            const int rhs_idx = ZydisGprToIndex(rhs_op.reg.value);
            if (rhs_idx < 0)
                return false;
            if (!EmitEffectiveAddress(lhs_op.mem, next_rip, c))
                return false;
            c.mov(rax, qword[rdx]);                       // rax = [mem]
            c.mov(rdx, qword[r13 + GprOffset(rhs_idx)]);  // rdx = reg
            c.and_(rax, rdx);
            EmitFlagsFromBitwise(c);
            return true;
        }
        if (rhs_op.type == ZYDIS_OPERAND_TYPE_IMMEDIATE) {
            if (!EmitEffectiveAddress(lhs_op.mem, next_rip, c))
                return false;
            c.mov(rax, qword[rdx]);            // rax = [mem]
            c.mov(rdx, rhs_op.imm.value.s);    // rdx = sign-extended imm
            c.and_(rax, rdx);
            EmitFlagsFromBitwise(c);
            return true;
        }
        return false;
    }

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

    // 32-bit reg-imm: `xor r32, imm`. Mirrors EmitAnd's 32-bit imm
    // path — truncate the (possibly sign-extended) immediate to u32;
    // the 32-bit op zero-extends rax.
    if (insn.operand_width == 32 && src.type == ZYDIS_OPERAND_TYPE_IMMEDIATE) {
        const u32 imm = static_cast<u32>(src.imm.value.u & 0xFFFFFFFFu);
        c.mov(eax, dword[r13 + GprOffset(dst_idx)]);
        c.xor_(eax, imm);
        c.mov(qword[r13 + GprOffset(dst_idx)], rax);
        EmitFlagsFromBitwise(c);
        return true;
    }

    // 64-bit reg-imm: `xor r64, imm` (imm8/imm32 sign-extended). Seen
    // at libc 0x808186d09. Materialize the full s64 into rdx — same
    // pattern EmitAnd/EmitAdd use to dodge xbyak imm-encoding limits.
    if (insn.operand_width == 64 && src.type == ZYDIS_OPERAND_TYPE_IMMEDIATE) {
        const auto imm = src.imm.value.s;
        c.mov(rax, qword[r13 + GprOffset(dst_idx)]);
        c.mov(rdx, imm);
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

    // 64-bit AND with memory destination: `and qword[mem], r64`.
    // The mem-dst sibling of the reg-dst forms above, and the gap from
    // CUSA02394 at guest 0x8001e1d69 — in the same code region as the
    // TEST qword[mem],r64 just above it (a load-AND-store mask applied
    // in place to a memory word). Unlike TEST, AND writes the result
    // back. Same shape as EmitOr's mem-dst path: stash address in rdx,
    // load src reg into rcx, lhs from [rdx] into rax, AND, store back,
    // flags from the result (CF=OF=0 per the bitwise spec).
    if (insn.operand_width == 64 && ops[0].type == ZYDIS_OPERAND_TYPE_MEMORY &&
        ops[1].type == ZYDIS_OPERAND_TYPE_REGISTER) {
        const int src_idx = ZydisGprToIndex(ops[1].reg.value);
        if (src_idx < 0)
            return false;
        if (!EmitEffectiveAddress(ops[0].mem, next_rip, c))
            return false;
        c.mov(rcx, qword[r13 + GprOffset(src_idx)]); // rcx = src reg
        c.mov(rax, qword[rdx]);                       // rax = [mem]
        c.and_(rax, rcx);
        c.mov(qword[rdx], rax);                       // store result
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

/// INC r/m — add 1, preserve CF. Register forms (32/64-bit) and
/// memory forms (8/16/32/64-bit).
bool EmitInc(const ZydisDecodedInstruction& insn, const ZydisDecodedOperand* ops,
             u64 next_rip, Xbyak::CodeGenerator& c) {
    if (insn.operand_width != 64 && insn.operand_width != 32 &&
        insn.operand_width != 16 && insn.operand_width != 8)
        return false;

    // Memory-destination form: `inc byte/word/dword/qword [mem]`.
    // First seen 8-bit at libc 0x805f6a7f5 (refcount-byte increment);
    // 32-bit form at 0x80027136c (a 32-bit in-memory counter increment).
    // x86 INC architecturally preserves CF — and so does the host INC,
    // so the rflags round-trip gives us CF-preservation automatically,
    // without the explicit save/restore the register paths below need.
    if (ops[0].type == ZYDIS_OPERAND_TYPE_MEMORY) {
        if (!EmitEffectiveAddress(ops[0].mem, next_rip, c))
            return false;
        c.mov(r10, rdx); // stash EA across the rflags round-trip

        // Load the in-memory value at the operand width.
        switch (insn.operand_width) {
        case 8:  c.mov(al,  byte[r10]);  break;
        case 16: c.mov(ax,  word[r10]);  break;
        case 32: c.mov(eax, dword[r10]); break;
        case 64: c.mov(rax, qword[r10]); break;
        default: return false;
        }

        c.mov(rdx, qword[r13 + Offsets::Rflags]);
        c.push(rdx);
        c.popfq();

        switch (insn.operand_width) {
        case 8:  c.inc(al);  break;
        case 16: c.inc(ax);  break;
        case 32: c.inc(eax); break;
        case 64: c.inc(rax); break;
        }

        c.pushfq();
        c.pop(rdx);
        c.mov(qword[r13 + Offsets::Rflags], rdx);

        // Store back exactly the operand width (memory has no
        // zero-extension semantics — only the touched bytes change).
        switch (insn.operand_width) {
        case 8:  c.mov(byte[r10],  al);  break;
        case 16: c.mov(word[r10],  ax);  break;
        case 32: c.mov(dword[r10], eax); break;
        case 64: c.mov(qword[r10], rax); break;
        }
        return true;
    }

    if (ops[0].type != ZYDIS_OPERAND_TYPE_REGISTER)
        return false;

    // 8-bit register form: `inc r8` (AL/CL/.../R15B, incl. AH/CH/DH/BH
    // high-byte regs). First seen at Retro-engine 0x8002ac2cb. Like the
    // mem-dst path, we let the host INC produce the flags via the rflags
    // round-trip — host INC preserves CF, so CF-preservation is
    // automatic (no manual save/restore). The byte-offset helper points
    // at the exact byte slot (handling high-byte regs), and a byte-sized
    // store preserves the surrounding 56 bits of the parent register.
    if (insn.operand_width == 8) {
        const int byte_off = ZydisGpr8ToByteOffset(ops[0].reg.value);
        if (byte_off < 0) return false;

        c.mov(al, byte[r13 + byte_off]);

        c.mov(rdx, qword[r13 + Offsets::Rflags]);
        c.push(rdx);
        c.popfq();

        c.inc(al);

        c.pushfq();
        c.pop(rdx);
        c.mov(qword[r13 + Offsets::Rflags], rdx);

        c.mov(byte[r13 + byte_off], al);
        return true;
    }

    // 16-bit register form: `inc r16` (AX/CX/.../R15W). Modeled on the
    // 8-bit register path above: load the low 16 bits of the parent slot,
    // round-trip guest rflags through the host so `inc ax` produces correct
    // 16-bit-width flags (host INC preserves CF, so CF-preservation is
    // automatic — no manual save/restore), then store back only the low 16
    // bits. A 16-bit register write preserves the upper 48 bits of the
    // parent register (unlike a 32-bit write, which zero-extends), so we use
    // a word-sized store into the slot rather than a full-qword writeback.
    if (insn.operand_width == 16) {
        const int idx = ZydisGprToIndex(ops[0].reg.value);
        if (idx < 0) return false;

        c.mov(ax, word[r13 + GprOffset(idx)]);

        c.mov(rdx, qword[r13 + Offsets::Rflags]);
        c.push(rdx);
        c.popfq();

        c.inc(ax);

        c.pushfq();
        c.pop(rdx);
        c.mov(qword[r13 + Offsets::Rflags], rdx);

        c.mov(word[r13 + GprOffset(idx)], ax); // preserve parent bits 63:16
        return true;
    }

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

/// DEC r/m — subtract 1, preserve CF. 8-bit (register) and 32/64-bit
/// register forms.
bool EmitDec(const ZydisDecodedInstruction& insn, const ZydisDecodedOperand* ops,
             u64 next_rip, Xbyak::CodeGenerator& c) {
    // Memory-destination form: `dec byte/word/dword/qword [mem]`.
    // The 16-bit form (`dec word [mem]`, 66-prefixed, length 8 with a
    // disp32 SIB address) is the run-ending gap from CUSA02394 at
    // libc 0x800143b30 — an in-memory 16-bit counter decrement. This
    // mirrors the INC mem-dst path exactly: x86 DEC preserves CF, and
    // so does the host DEC, so rolling guest rflags through the host
    // CPU around the host DEC gives CF-preservation for free (no
    // explicit save/restore like the register paths below need).
    if (ops[0].type == ZYDIS_OPERAND_TYPE_MEMORY) {
        if (insn.operand_width != 64 && insn.operand_width != 32 &&
            insn.operand_width != 16 && insn.operand_width != 8)
            return false;
        if (!EmitEffectiveAddress(ops[0].mem, next_rip, c))
            return false;
        c.mov(r10, rdx); // stash EA across the rflags round-trip

        switch (insn.operand_width) {
        case 8:  c.mov(al,  byte[r10]);  break;
        case 16: c.mov(ax,  word[r10]);  break;
        case 32: c.mov(eax, dword[r10]); break;
        case 64: c.mov(rax, qword[r10]); break;
        default: return false;
        }

        c.mov(rdx, qword[r13 + Offsets::Rflags]);
        c.push(rdx);
        c.popfq();

        switch (insn.operand_width) {
        case 8:  c.dec(al);  break;
        case 16: c.dec(ax);  break;
        case 32: c.dec(eax); break;
        case 64: c.dec(rax); break;
        }

        c.pushfq();
        c.pop(rdx);
        c.mov(qword[r13 + Offsets::Rflags], rdx);

        // Store back exactly the operand width — memory has no
        // zero-extension; only the touched bytes change.
        switch (insn.operand_width) {
        case 8:  c.mov(byte[r10],  al);  break;
        case 16: c.mov(word[r10],  ax);  break;
        case 32: c.mov(dword[r10], eax); break;
        case 64: c.mov(qword[r10], rax); break;
        }
        return true;
    }

    if (ops[0].type != ZYDIS_OPERAND_TYPE_REGISTER)
        return false;

    // 8-bit register form: `dec r8` (incl. AH/CH/DH/BH high-byte regs).
    // First seen at Retro-engine 0x80030e358 (3 bytes past the DIV r8
    // at ...355 — same routine: divide, then step a byte counter). The
    // symmetric companion of `inc r8`. We let the host DEC produce the
    // flags via the rflags round-trip — host DEC preserves CF, so
    // CF-preservation is automatic (no manual save/restore). The
    // byte-offset helper points at the exact byte slot (handling
    // high-byte regs), and a byte-sized store preserves the surrounding
    // 56 bits of the parent register.
    if (insn.operand_width == 8) {
        const int byte_off = ZydisGpr8ToByteOffset(ops[0].reg.value);
        if (byte_off < 0) return false;

        c.mov(al, byte[r13 + byte_off]);

        c.mov(rdx, qword[r13 + Offsets::Rflags]);
        c.push(rdx);
        c.popfq();

        c.dec(al);

        c.pushfq();
        c.pop(rdx);
        c.mov(qword[r13 + Offsets::Rflags], rdx);

        c.mov(byte[r13 + byte_off], al);
        return true;
    }

    // 16-bit register form: `dec r16` (AX/CX/.../R15W). Symmetric companion
    // of `inc r16`. Load the low 16 bits of the parent slot, round-trip
    // guest rflags through the host so `dec ax` produces correct 16-bit
    // flags (host DEC preserves CF — automatic, no manual save/restore),
    // then store back only the low 16 bits. A 16-bit register write
    // preserves the parent's upper 48 bits (unlike a 32-bit write, which
    // zero-extends), so we use a word-sized store into the slot.
    if (insn.operand_width == 16) {
        const int idx = ZydisGprToIndex(ops[0].reg.value);
        if (idx < 0) return false;

        c.mov(ax, word[r13 + GprOffset(idx)]);

        c.mov(rdx, qword[r13 + Offsets::Rflags]);
        c.push(rdx);
        c.popfq();

        c.dec(ax);

        c.pushfq();
        c.pop(rdx);
        c.mov(qword[r13 + Offsets::Rflags], rdx);

        c.mov(word[r13 + GprOffset(idx)], ax); // preserve parent bits 63:16
        return true;
    }

    if (insn.operand_width != 64 && insn.operand_width != 32)
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
///   leaves OF/SF/ZF/AF/PF "undefined" — both Intel and AMD allow
///   arbitrary values. To stay deterministic we compute CF explicitly
///   and leave the other guest flags unchanged.
///
///   ── INTEL vs AMD DIVERGENCE (BT/BTS/BTR/BTC undefined flags) ──
///   Both vendors document SF/ZF/AF/PF as undefined after the BT
///   family (only CF is defined = the selected bit). Their silicon
///   differs in what's actually left: AMD tends to preserve the
///   prior values; Intel may zero or scramble SF/OF. Our "leave the
///   other guest flags unchanged" policy happens to match AMD's
///   observed preserve-prior behavior, which is the correct target
///   for a Jaguar/AuthenticAMD guest — so this is accurate by
///   construction, not just convenient. The ARM64 backend should
///   keep the same policy: set only CF, preserve the rest.
/// ────────────────────────────────────────────────────────────────
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
        const unsigned int bit_pos =
            static_cast<unsigned int>(ops[1].imm.value.u & 0xFFu);
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
    if (insn.operand_width != 64 && insn.operand_width != 32 && insn.operand_width != 16)
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
        // 16-bit mem-src: only the low 16 bits of [mem] matter for
        // the eventual merge. We still load 64 bits — the high bits
        // get masked away in the cond-true candidate construction.
        c.mov(r8, qword[rdx]);
    } else {
        return false;
    }

    // Load current dst slot into r9 (the candidate "no-change" result).
    // For 32-bit CMOVcc with the condition FALSE, the destination is
    // unchanged ENTIRELY — including bits 63:32. This is a quirk of
    // CMOVcc relative to ordinary 32-bit ops, which always zero-extend.
    // The same "no change on FALSE" rule applies to the 16-bit form,
    // so r9 = full qword in all width cases.
    c.mov(r9, qword[r13 + GprOffset(dst_idx)]);

    // Compute condition into rcx (0 or 1).
    if (!EmitJccCondition(jcc_equiv, c))
        return false;

    // Build the cond-TRUE candidate in rax. Three width-specific
    // shapes — the 16-bit form is the most exotic because the
    // architectural rule is "low 16 of dst gets low 16 of src;
    // bits 63:16 of dst are PRESERVED" (no zero-extension, unlike
    // 32-bit). We compose it by clearing the low 16 of the current
    // dst slot and ORing in the low 16 of src.
    //
    // The build steps for the 16-bit form use AND/OR which DO touch
    // flags — so we must finish the candidate-build BEFORE the
    // `test rcx, rcx` that the cmov consumes. Otherwise the cmov
    // would read ZF from the last AND/OR, not from the condition.
    // The 32/64-bit paths only use MOV (no flag effect) so they're
    // safe to inline around the test.
    if (insn.operand_width == 16) {
        // rax = (cur_dst & ~0xFFFF) | (src & 0xFFFF). Build candidate
        // FIRST while we're free to touch flags.
        c.mov(rax, r9);
        c.mov(rdx, 0xFFFFFFFFFFFF0000ULL);
        c.and_(rax, rdx);
        c.mov(rdx, r8);
        c.and_(rdx, 0xFFFF);
        c.or_(rax, rdx);
        c.test(rcx, rcx);
        c.cmovnz(r9, rax);
    } else if (insn.operand_width == 32) {
        c.test(rcx, rcx);
        c.mov(eax, r8d); // rax = src & 0xFFFFFFFF, upper 32 = 0
        c.cmovnz(r9, rax);
    } else {
        c.test(rcx, rcx);
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
// ── INTEL vs AMD DIVERGENCE (OF and CF on shifts) ──────────────────
// The PS4 is an AMD Jaguar, and our CPUID spoofs AuthenticAMD, so the
// *architecturally correct* reference here is the AMD APM, not the
// Intel SDM. The two diverge on the UNDEFINED cases of shifts:
//
//   * OF (overflow flag): BOTH vendors define OF only for count==1
//     (SHL/SAL: OF = MSB(result) XOR CF; SAR: OF = 0; SHR: OF =
//     MSB(original)). For count>1 BOTH document OF as "undefined" —
//     BUT the actual left-behind value differs between Intel and AMD
//     silicon. AMD Jaguar leaves OF computed as if count==1 (i.e.
//     MSB(result) XOR CF semantics persist); several Intel parts zero
//     it or leave the prior value. Software that reads OF after a
//     multi-bit shift is buggy, but a few real titles do.
//
//   * CF for count >= operand-size (e.g. SHL r32 by 32+): documented
//     "undefined" by both. AMD masks the count FIRST (to 5 or 6 bits)
//     so e.g. "SHL r32, 32" becomes "SHL r32, 0" => CF unchanged.
//     This falls out naturally from the host round-trip on an x86
//     host because the host masks identically.
//
// Because we execute the shift on the HOST CPU and copy its rflags
// verbatim, on an x86 host we inherit *that host's* choice for the
// undefined bits — which is only guaranteed to match AMD Jaguar when
// the host is itself AMD. This is an accepted imprecision on the x86
// host backend (undefined-flag divergence is invisible to correct
// software). The ARM64 backend, which must compute these flags
// explicitly rather than via a round-trip, MUST follow the AMD APM
// rule above: define OF for count==1 per the table, and for count>1
// reproduce the count==1 OF formula (Jaguar behavior) rather than
// Intel's zeroing.
// ───────────────────────────────────────────────────────────────────
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

/// 32-bit rotate emitter. Same structure as EmitRotate64 but operates
/// at 32-bit width: the host `rol`/`ror` on `eax` masks the count to
/// 5 bits (0..31) automatically, matching guest semantics, and the
/// 32-bit write zero-extends so a full-qword storeback records the
/// canonical zero-extension in the guest GPR slot.
///
/// Flag semantics (identical Intel/AMD for rotates): CF = last bit
/// rotated through; OF defined only for 1-bit rotates (undefined for
/// multi-bit, which the host round-trip reproduces); SF/ZF/AF/PF are
/// NOT modified by rotates — the popfq round-trip loads them from the
/// guest rflags before the host rotate, the host rotate leaves them
/// alone, and pushfq carries the unchanged values back. Rotate-by-
/// zero affects no flags, also handled by the round-trip.
bool EmitRotate32(const ZydisDecodedInstruction& insn, const ZydisDecodedOperand* ops,
                  Xbyak::CodeGenerator& c, RotateKind kind) {
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
    case RotateKind::Rol:
        c.rol(eax, cl);
        break;
    case RotateKind::Ror:
        c.ror(eax, cl);
        break;
    }

    c.pushfq();
    c.pop(rdx);
    c.mov(qword[r13 + Offsets::Rflags], rdx);

    // Full-qword store so the host's 32-bit zero-extension lands in
    // the guest GPR slot (bits 63:32 cleared).
    c.mov(qword[r13 + GprOffset(dst_idx)], rax);
    return true;
}
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
//
// ── INTEL vs AMD DIVERGENCE (SF/ZF/AF/PF after IMUL) ───────────────
// CF and OF are defined identically on both vendors (set when the
// product doesn't fit). The other four — SF, ZF, AF, PF — are
// documented "undefined" by both Intel and AMD, but their actual
// silicon behavior differs, and the AMD APM is our reference (PS4 =
// Jaguar, CPUID = AuthenticAMD):
//
//   * AMD Jaguar computes SF/ZF/PF from the LOW half of the result
//     as if it were a normal data write (SF = MSB of low 64, ZF =
//     (low 64 == 0), PF = parity of low byte), and leaves AF
//     cleared. Intel's recent parts also began defining SF this way
//     (post-2014 SDM), but older Intel leaves these genuinely
//     garbage. AMD's "compute from low half" has been stable.
//
// Because we round-trip the HOST rflags, on an x86 host we again
// inherit the host CPU's behavior for these undefined bits — only
// guaranteed AMD-correct on an AMD host. Accepted imprecision here
// (correct software never reads these after IMUL). The ARM64 backend
// MUST, when it computes flags explicitly, follow the AMD rule:
// SF/ZF/PF from the low 64 bits, AF cleared. Do NOT leave them at
// whatever the AArch64 NZCV happened to be.
// ───────────────────────────────────────────────────────────────────
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
// MUL — unsigned multiply. Single explicit operand (r/m); the other factor is
// the implicit accumulator. The double-width product lands in a register pair:
//
//   MUL r/m8 :  AX        = AL  * src      (8-bit:   high=AH,  low=AL)
//   MUL r/m16:  DX:AX     = AX  * src
//   MUL r/m32:  EDX:EAX   = EAX * src      (upper 32 of RDX/RAX zero-extended)
//   MUL r/m64:  RDX:RAX   = RAX * src
//
// CF and OF are SET if the upper half of the product is non-zero (i.e. the
// result didn't fit in the low half), otherwise CLEARED. SF/ZF/AF/PF are
// undefined; we let the host's MUL set whatever it sets and round-trip the
// host rflags, which is the same strategy EmitImul1Op uses for its undefined
// flags. This is the unsigned sibling of EmitImul1Op — identical shape, with
// `c.mul` instead of `c.imul` and per-width accumulator handling.
//
// First observed as `mul rcx` (48 f7 e1) in the CUSA02394 "WE ARE DOOMED"
// eboot at guest 0x80020642b. Host scratch: rcx=src factor, rax/rdx = product.
bool EmitMul(const ZydisDecodedInstruction& insn, const ZydisDecodedOperand* ops, u64 next_rip,
             Xbyak::CodeGenerator& c) {
    const unsigned w = insn.operand_width;
    if (w != 8 && w != 16 && w != 32 && w != 64)
        return false;

    // Load src into rcx. For memory operands, EmitEffectiveAddress writes the
    // address into rdx and may clobber rax — so dereference into rcx BEFORE
    // loading the accumulator (same ordering rule as EmitImul1Op).
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

    // Load the implicit accumulator (guest RAX) into host rax.
    c.mov(rax, qword[r13 + GprOffset(0)]);

    // Flag round-trip: seed host rflags from guest, run, capture back.
    c.mov(r8, qword[r13 + Offsets::Rflags]);
    c.push(r8);
    c.popfq();

    // Host MUL with the matching sub-register width. The implicit accumulator
    // and the high-half destination follow the width automatically.
    switch (w) {
    case 8:  c.mul(cl);  break; // AX      = AL  * CL
    case 16: c.mul(cx);  break; // DX:AX   = AX  * CX
    case 32: c.mul(ecx); break; // EDX:EAX = EAX * ECX
    case 64: c.mul(rcx); break; // RDX:RAX = RAX * RCX
    }

    c.pushfq();
    c.pop(r8);
    c.mov(qword[r13 + Offsets::Rflags], r8);

    // Write the result back to the guest slots.
    //
    // For 8-bit MUL the entire 16-bit product is in AX — there is NO separate
    // high register; AH holds the high byte. Writing the full host rax into
    // guest RAX captures both AL and AH correctly (and matches x86: bits above
    // 16 of the accumulator are not part of the architectural result, but a
    // host `mul cl` leaves rax's upper bits as they were — to stay faithful we
    // only need RAX's low 16 to be right, and they are; RDX is not a MUL r/m8
    // destination, so leave guest RDX untouched).
    //
    // For 16/32/64-bit MUL the low half goes to (E/R)AX and the high half to
    // (E/R)DX. Host `mul` already zero-extends the 32-bit forms into the full
    // 64-bit rax/rdx, so a plain qword store is correct for width 32 and 64.
    if (w == 8) {
        // Only RAX is written (AX = AL*src). Preserve guest RAX's upper 48
        // bits, replace the low 16 with the product. Host AX holds it.
        c.mov(r9, qword[r13 + GprOffset(0)]); // current guest RAX
        c.mov(r10, 0xFFFFFFFFFFFF0000ULL);
        c.and_(r9, r10);                      // clear low 16
        c.movzx(r11d, ax);                    // product in AX (zero-extended)
        c.or_(r9, r11);
        c.mov(qword[r13 + GprOffset(0)], r9);
    } else {
        // 16-bit: low→AX (preserve RAX upper 48), high→DX (preserve RDX upper 48).
        // 32-bit: low→EAX (host zero-extended RAX), high→EDX (host zero-extended RDX).
        // 64-bit: low→RAX, high→RDX.
        if (w == 16) {
            c.mov(r9, qword[r13 + GprOffset(0)]);  // guest RAX
            c.mov(r10, 0xFFFFFFFFFFFF0000ULL);
            c.and_(r9, r10);
            c.movzx(r11d, ax);
            c.or_(r9, r11);
            c.mov(qword[r13 + GprOffset(0)], r9);

            c.mov(r9, qword[r13 + GprOffset(2)]);  // guest RDX
            c.and_(r9, r10);
            c.movzx(r11d, dx);
            c.or_(r9, r11);
            c.mov(qword[r13 + GprOffset(2)], r9);
        } else {
            // 32-bit and 64-bit: host result already in full rax/rdx with the
            // correct zero-extension for the 32-bit case.
            c.mov(qword[r13 + GprOffset(0)], rax); // low  → RAX
            c.mov(qword[r13 + GprOffset(2)], rdx); // high → RDX
        }
    }
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
//   - All flags are documented as "undefined" after DIV (both Intel
//     SDM Vol. 2A and AMD APM Vol. 3 agree here — CF/OF/SF/ZF/AF/PF
//     are all undefined for DIV and IDIV). We deliberately skip the
//     rflags round-trip so the guest sees its pre-DIV flag state
//     preserved — this matches the letter of the spec (undefined ≡
//     any value, including unchanged) and avoids the cost of two
//     push/pop pairs around the host op. Preserve-prior also happens
//     to match AMD Jaguar's observed behavior, so it's the right
//     target for our AuthenticAMD guest. The ARM64 backend should
//     likewise leave flags untouched across DIV/IDIV.
//
// Divide-by-zero and quotient-overflow #DE faults propagate as host
// SIGFPE; until we wire up a signal handler that maps these to a
// guest exception, programs that actually divide-by-zero will crash
// the runtime. Real PS4 binaries don't intentionally do this.
// =============================================================================

bool EmitDiv(const ZydisDecodedInstruction& insn, const ZydisDecodedOperand* ops, u64 next_rip,
             Xbyak::CodeGenerator& c) {
    // 8-bit DIV has a different implicit-operand layout than the wider
    // forms: the dividend is the full 16-bit AX (not a DX:AX pair),
    // the quotient goes to AL, and the remainder to AH. Seen at Retro-
    // engine 0x80030e355. The host `div r8` does exactly this.
    if (insn.operand_width == 8) {
        // Load the 8-bit divisor into cl. The divisor may be a
        // high-byte register (AH/CH/DH/BH), so use the byte-offset
        // helper rather than assuming the low byte of a slot.
        if (ops[0].type == ZYDIS_OPERAND_TYPE_REGISTER) {
            const int div_off = ZydisGpr8ToByteOffset(ops[0].reg.value);
            if (div_off < 0) return false;
            c.mov(cl, byte[r13 + div_off]);
        } else if (ops[0].type == ZYDIS_OPERAND_TYPE_MEMORY) {
            if (!EmitEffectiveAddress(ops[0].mem, next_rip, c))
                return false;
            c.mov(cl, byte[rdx]);
        } else {
            return false;
        }

        // Dividend = guest AX (low 16 bits of RAX slot, GPR index 0).
        c.mov(ax, word[r13 + GprOffset(0)]);

        c.div(cl); // AX / cl -> AL = quotient, AH = remainder

        // Store AL (quotient) and AH (remainder) back into the low 16
        // bits of guest RAX. A 16-bit store writes both AL and AH at
        // once while preserving the upper 48 bits of the slot.
        c.mov(word[r13 + GprOffset(0)], ax);
        return true;
    }

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

// CLD / STD — clear / set the direction flag (DF, rflags bit 10). DF
// controls whether the string instructions (MOVS/STOS/LODS/CMPS/SCAS)
// auto-increment (DF=0) or auto-decrement (DF=1) their pointer
// registers. These touch no other flag and no register. CLD seen at
// libc 0x8081df7f4 (the conventional "forward direction" setup glibc
// string/memory routines emit before a REP-prefixed copy/scan). STD
// is its setter sibling, pre-wired here.
bool EmitCld(Xbyak::CodeGenerator& c) {
    // Clear DF: load, BTR bit 10, store. Through a register (same
    // reason as EmitClc — avoids the `and qword[mem], imm32`
    // sign-extension foot-gun, and keeps the shape uniform).
    c.mov(r10, qword[r13 + Offsets::Rflags]);
    c.btr(r10, 10);
    c.mov(qword[r13 + Offsets::Rflags], r10);
    return true;
}

bool EmitStd(Xbyak::CodeGenerator& c) {
    // Set DF: load, BTS bit 10, store.
    c.mov(r10, qword[r13 + Offsets::Rflags]);
    c.bts(r10, 10);
    c.mov(qword[r13 + Offsets::Rflags], r10);
    return true;
}

// =============================================================================
// CMPS / REP(E|NE) CMPS — compare string operands.
//
// CMPSQ compares qword [RSI] vs qword [RDI] (CMP semantics: sets
// SF/ZF/CF/OF/PF/AF from [RSI]-[RDI]) and advances BOTH RSI and RDI
// by the element size, in the direction set by DF (forward +size when
// DF=0, backward -size when DF=1).
//
// With a REPE/REPNE prefix it becomes a counted loop driven by RCX —
// the building block of memcmp/wmemcmp:
//   REPE  CMPSQ: repeat while RCX!=0 AND ZF==1  (stop on mismatch or count)
//   REPNE CMPSQ: repeat while RCX!=0 AND ZF==0  (stop on match or count)
// Each iteration: compare, advance RSI/RDI by ±size, decrement RCX.
// If RCX starts at 0, nothing happens (no compare, no flag change).
// The final flags reflect the LAST comparison actually performed.
//
// First seen as `repe cmpsq` at libc 0x8081df7fc — a memcmp inner
// loop, right after the CLD that set the forward direction.
//
// We emit a host loop. Implicit operands map to fixed guest slots:
// RSI=6, RDI=7, RCX=1. The element size comes from operand_width.
// Host scratch: r8=RSI, r9=RDI, r10=RCX, r11=step; rax/rdx hold the
// two loaded elements for the compare; flags captured at the end.
bool EmitCmps(const ZydisDecodedInstruction& insn, Xbyak::CodeGenerator& c) {
    using LT = Xbyak::CodeGenerator::LabelType;

    // CRITICAL: ZYDIS_MNEMONIC_CMPSD is shared between the string
    // "compare dword" and the unrelated SSE2 scalar-double CMPSD
    // (compare scalar double-precision, e.g. `cmpsd xmm0, xmm1, imm8`).
    // They are distinguished only by operand visibility: the string
    // forms (CMPSB/W/D/Q) have ALL operands implicit (RSI/RDI/RCX), so
    // operand_count_visible == 0; the SSE2 form has 3 visible operands.
    // Reject anything with visible operands so an SSE2 CMPSD routed
    // here falls through to its own handler / unsupported path.
    if (insn.operand_count_visible != 0)
        return false;

    const unsigned w = insn.operand_width;
    if (w != 8 && w != 16 && w != 32 && w != 64)
        return false;
    const int elem = static_cast<int>(w / 8); // 1/2/4/8 bytes

    const bool has_repe  = (insn.attributes & ZYDIS_ATTRIB_HAS_REPE) != 0;
    const bool has_repne = (insn.attributes & ZYDIS_ATTRIB_HAS_REPNE) != 0;
    const bool is_rep = has_repe || has_repne;

    constexpr int kRsi = 6, kRdi = 7, kRcx = 1;

    // Pointers into r8/r9.
    c.mov(r8, qword[r13 + GprOffset(kRsi)]);
    c.mov(r9, qword[r13 + GprOffset(kRdi)]);

    // Step into r11: +elem if DF=0, -elem if DF=1 (DF = rflags bit 10).
    c.mov(r11, elem);
    {
        Xbyak::Label df_clear;
        c.mov(rax, qword[r13 + Offsets::Rflags]);
        c.bt(rax, 10);
        c.jnc(df_clear, LT::T_NEAR);
        c.neg(r11);
        c.L(df_clear);
    }

    // rbx holds the guest rflags we will ultimately write back. Start
    // it at the current guest rflags so a zero-trip REP (RCX==0) leaves
    // flags unchanged. Each performed compare overwrites it with that
    // compare's full flag state (the architectural rule: final flags
    // reflect the LAST comparison performed). We must preserve rbx
    // across the host op, so save/restore the host's rbx (it is NOT a
    // JIT scratch — r13 is pinned; rbx belongs to the host ABI here).
    c.push(rbx);
    c.mov(rbx, qword[r13 + Offsets::Rflags]);

    // NOTE: we deliberately do NOT popfq the guest rflags into the host
    // here. Doing so would also load the guest's DF (bit 10) into the
    // HOST direction flag, leaving the host in DF=1 mode on return —
    // a calling-convention violation that corrupts any host memcpy/etc.
    // run by the dispatcher afterward. We don't need it anyway: each
    // `cmp` below sets all the arithmetic flags fresh, and rbx already
    // carries the guest flag bits for the zero-trip case. The host DF
    // stays at its canonical 0 throughout (we use explicit add/sub for
    // the pointer stepping, never a host string op).

    // Emits: compare [r8] vs [r9] at width w, then snapshot the host
    // rflags into rbx (so rbx always holds the most-recent compare's
    // flags). Uses rax as the load scratch.
    auto cmp_and_capture = [&]() {
        switch (w) {
        case 8:  c.mov(al,  byte[r8]);  c.cmp(al,  byte[r9]);  break;
        case 16: c.mov(ax,  word[r8]);  c.cmp(ax,  word[r9]);  break;
        case 32: c.mov(eax, dword[r8]); c.cmp(eax, dword[r9]); break;
        case 64: c.mov(rax, qword[r8]); c.cmp(rax, qword[r9]); break;
        }
        c.pushfq();
        c.pop(rbx);                 // rbx = this compare's full rflags
    };

    if (!is_rep) {
        // Plain CMPS: one compare, one advance.
        cmp_and_capture();
        c.add(r8, r11);
        c.add(r9, r11);
    } else {
        c.mov(r10, qword[r13 + GprOffset(kRcx)]);

        Xbyak::Label l_top, l_done;
        c.test(r10, r10);
        c.jz(l_done, LT::T_NEAR);   // RCX==0: zero-trip, rbx stays = guest flags

        c.L(l_top);
        cmp_and_capture();          // rbx = last compare's flags
        c.add(r8, r11);
        c.add(r9, r11);
        c.dec(r10);
        c.jz(l_done, LT::T_NEAR);   // count exhausted

        // Data condition: REPE continues while equal (ZF==1 in rbx),
        // REPNE continues while not-equal (ZF==0). ZF is rbx bit 6.
        c.bt(rbx, 6);               // CF = ZF-of-last-compare
        if (has_repe)
            c.jc(l_top, LT::T_NEAR);   // equal -> keep going
        else
            c.jnc(l_top, LT::T_NEAR);  // not-equal -> keep going
        c.L(l_done);
    }

    // Writeback. Store the advanced pointers and residual count FIRST,
    // while r8/r9/r10 still hold them — then the flag merge is free to
    // use those registers as scratch.
    c.mov(qword[r13 + GprOffset(kRsi)], r8);
    c.mov(qword[r13 + GprOffset(kRdi)], r9);
    if (is_rep)
        c.mov(qword[r13 + GprOffset(kRcx)], r10);

    // Flags: arithmetic bits come from the last compare (rbx); the
    // non-arithmetic bits — crucially DF, which CMPS must NOT modify —
    // are preserved from the guest's current rflags. On a zero-trip
    // REP, rbx still equals the full original guest rflags, so this
    // merge is an exact no-op. Uses rax/r8/r9 as scratch (pointers
    // already stored above).
    constexpr u64 kArith = 0x8D5; // CF|PF|AF|ZF|SF|OF
    c.mov(rax, qword[r13 + Offsets::Rflags]); // original guest rflags
    c.mov(r8, ~kArith);
    c.and_(rax, r8);            // guest non-arith bits (incl DF)
    c.mov(r9, kArith);
    c.and_(rbx, r9);            // compare's arith bits only
    c.or_(rax, rbx);            // merged
    c.mov(qword[r13 + Offsets::Rflags], rax);

    c.pop(rbx);                 // restore host rbx
    return true;
}

// =============================================================================
// STOS / REP STOS — store string. STOSB/W/D/Q store AL/AX/EAX/RAX to [RDI],
// then advance RDI by ±elem (DF). With a REP prefix it is a counted fill
// driven by RCX — the core of memset. No flags are affected (REP STOS has no
// data-condition; it runs exactly RCX times).
//
// First seen as `rep stosb` inside libc memset (CUSA02394 "WE ARE DOOMED",
// guest 0x8075ac00f). Implicit operands: RDI=7 (dest), RAX=0 (value),
// RCX=1 (count for REP). Host scratch: r9=RDI, r10=RCX, r11=step, al/ax/eax/rax
// holds the store value.
bool EmitStos(const ZydisDecodedInstruction& insn, Xbyak::CodeGenerator& c) {
    using LT = Xbyak::CodeGenerator::LabelType;
    if (insn.operand_count_visible != 0)
        return false; // string form has all-implicit operands

    const unsigned w = insn.operand_width;
    if (w != 8 && w != 16 && w != 32 && w != 64)
        return false;
    const int elem = static_cast<int>(w / 8);
    const bool is_rep = (insn.attributes &
                         (ZYDIS_ATTRIB_HAS_REP | ZYDIS_ATTRIB_HAS_REPE |
                          ZYDIS_ATTRIB_HAS_REPNE)) != 0;

    constexpr int kRdi = 7, kRax = 0, kRcx = 1;

    c.mov(r9, qword[r13 + GprOffset(kRdi)]);  // dest pointer
    c.mov(rax, qword[r13 + GprOffset(kRax)]); // store value (low w bits used)

    // step in r11 = ±elem from DF (rflags bit 10).
    c.mov(r11, elem);
    {
        Xbyak::Label df_clear;
        c.mov(r8, qword[r13 + Offsets::Rflags]);
        c.bt(r8, 10);
        c.jnc(df_clear, LT::T_NEAR);
        c.neg(r11);
        c.L(df_clear);
    }

    auto store_one = [&]() {
        switch (w) {
        case 8:  c.mov(byte[r9], al);   break;
        case 16: c.mov(word[r9], ax);   break;
        case 32: c.mov(dword[r9], eax); break;
        case 64: c.mov(qword[r9], rax); break;
        }
        c.add(r9, r11);
    };

    if (!is_rep) {
        store_one();
    } else {
        c.mov(r10, qword[r13 + GprOffset(kRcx)]);
        Xbyak::Label l_top, l_done;
        c.test(r10, r10);
        c.jz(l_done, LT::T_NEAR);   // RCX==0: store nothing
        c.L(l_top);
        store_one();
        c.dec(r10);
        c.jnz(l_top, LT::T_NEAR);
        c.L(l_done);
        c.mov(qword[r13 + GprOffset(kRcx)], r10); // residual count (0)
    }

    c.mov(qword[r13 + GprOffset(kRdi)], r9); // advanced dest
    // STOS affects no flags (DF preserved automatically — never touched).
    return true;
}

// =============================================================================
// MOVS / REP MOVS — move string. MOVSB/W/D/Q copy [RSI] -> [RDI] then advance
// BOTH by ±elem (DF). REP MOVS is the core of memcpy/memmove-forward. No flags.
//
// Implicit operands: RSI=6 (src), RDI=7 (dest), RCX=1 (count). Host scratch:
// r8=RSI, r9=RDI, r10=RCX, r11=step, rax = the copied element.
bool EmitMovs(const ZydisDecodedInstruction& insn, u64 rip, Xbyak::CodeGenerator& c) {
    using LT = Xbyak::CodeGenerator::LabelType;
    if (insn.operand_count_visible != 0)
        return false;

    const unsigned w = insn.operand_width;
    if (w != 8 && w != 16 && w != 32 && w != 64)
        return false;
    const int elem = static_cast<int>(w / 8);
    const bool is_rep = (insn.attributes &
                         (ZYDIS_ATTRIB_HAS_REP | ZYDIS_ATTRIB_HAS_REPE |
                          ZYDIS_ATTRIB_HAS_REPNE)) != 0;

    constexpr int kRsi = 6, kRdi = 7, kRcx = 1;

    c.mov(r8, qword[r13 + GprOffset(kRsi)]);
    c.mov(r9, qword[r13 + GprOffset(kRdi)]);

    // Bogus-pointer guard. A garbage source/dest pointer reaching a string
    // copy (observed in CUSA02394: rep movsq with guest RSI=0x3f, a tiny
    // near-null value produced upstream by a null-base + small-offset address
    // calculation — most likely an HLE/struct field that came back 0) would
    // otherwise fault deep inside the host copy loop and surface as a raw
    // 0xc0000005 with a misleading register dump: the live host r8/r9 are this
    // loop's scratch, NOT the guest-R8 slot the signal handler prints, so the
    // operator sees a "valid" r8 while the real bad pointer (guest RSI/RDI) is
    // elsewhere. Catch it here, before any dereference: if either RSI or RDI
    // is below the guest user base, leave the RSI/RDI/RCX slots untouched (they
    // still hold the pre-copy values, so the crash report is accurate), set a
    // distinct non-fatal exit reason, and bail to the gateway. The runtime
    // then reports the guest RIP of the copy with the true RSI/RDI, which is
    // exactly what's needed to walk back to the producer of the bad pointer.
    // Any legitimate PS4 mapping (flexible/direct/system-managed) is far above
    // 64 KiB, so this never trips on a real copy.
    {
        constexpr u64 kMinValidGuestPtr = 0x10000; // 64 KiB
        constexpr u32 EXIT_BADPTR = static_cast<u32>(ExitReason::HelperRequestedExit);
        Xbyak::Label ptrs_ok, bad_ptr;
        c.mov(rax, r8);
        c.cmp(rax, kMinValidGuestPtr);
        c.jb(bad_ptr, LT::T_NEAR);
        c.mov(rax, r9);
        c.cmp(rax, kMinValidGuestPtr);
        c.jb(bad_ptr, LT::T_NEAR);
        c.jmp(ptrs_ok, LT::T_NEAR);
        c.L(bad_ptr);
        // RSI/RDI/RCX slots are unchanged from the pre-copy load above, so the
        // fault report shows the true guest pointers. Set state.rip to this
        // MOVS so the post-mortem knows where it stopped, record a distinct
        // exit reason, and exit via r15 (fatal exit to the runtime). Using r14
        // (the dispatcher loop) here would re-dispatch this same block at the
        // unchanged RIP and spin forever.
        c.mov(rax, rip);
        c.mov(qword[r13 + Offsets::Rip], rax);
        c.mov(dword[r13 + offsetof(GuestState, exit_reason)], EXIT_BADPTR);
        c.jmp(r15);
        c.L(ptrs_ok);
    }

    c.mov(r11, elem);
    {
        Xbyak::Label df_clear;
        c.mov(rax, qword[r13 + Offsets::Rflags]);
        c.bt(rax, 10);
        c.jnc(df_clear, LT::T_NEAR);
        c.neg(r11);
        c.L(df_clear);
    }

    auto copy_one = [&]() {
        switch (w) {
        case 8:  c.mov(al,  byte[r8]);  c.mov(byte[r9], al);   break;
        case 16: c.mov(ax,  word[r8]);  c.mov(word[r9], ax);   break;
        case 32: c.mov(eax, dword[r8]); c.mov(dword[r9], eax); break;
        case 64: c.mov(rax, qword[r8]); c.mov(qword[r9], rax); break;
        }
        c.add(r8, r11);
        c.add(r9, r11);
    };

    if (!is_rep) {
        copy_one();
    } else {
        c.mov(r10, qword[r13 + GprOffset(kRcx)]);
        Xbyak::Label l_top, l_done;
        c.test(r10, r10);
        c.jz(l_done, LT::T_NEAR);
        c.L(l_top);
        copy_one();
        c.dec(r10);
        c.jnz(l_top, LT::T_NEAR);
        c.L(l_done);
        c.mov(qword[r13 + GprOffset(kRcx)], r10);
    }

    c.mov(qword[r13 + GprOffset(kRsi)], r8);
    c.mov(qword[r13 + GprOffset(kRdi)], r9);
    // MOVS affects no flags.
    return true;
}

// =============================================================================
// LODS / REP LODS — load string. LODSB/W/D/Q load [RSI] into AL/AX/EAX/RAX
// then advance RSI by ±elem (DF). REP LODS is near-useless (it just leaves the
// last element in the accumulator) but is architecturally valid; we honor it.
// No flags. For the 8/16/32-bit forms the accumulator's wider bits follow the
// usual x86 rules: a 32-bit load zero-extends RAX; 8/16-bit forms write only
// the sub-register and preserve the upper bits.
//
// Implicit operands: RSI=6 (src), RAX=0 (dest), RCX=1 (count). Host scratch:
// r8=RSI, r10=RCX, r11=step, rax/eax/ax/al = loaded value (mirrored to slot).
bool EmitLods(const ZydisDecodedInstruction& insn, Xbyak::CodeGenerator& c) {
    using LT = Xbyak::CodeGenerator::LabelType;
    if (insn.operand_count_visible != 0)
        return false;

    const unsigned w = insn.operand_width;
    if (w != 8 && w != 16 && w != 32 && w != 64)
        return false;
    const int elem = static_cast<int>(w / 8);
    const bool is_rep = (insn.attributes &
                         (ZYDIS_ATTRIB_HAS_REP | ZYDIS_ATTRIB_HAS_REPE |
                          ZYDIS_ATTRIB_HAS_REPNE)) != 0;

    constexpr int kRsi = 6, kRax = 0, kRcx = 1;

    c.mov(r8, qword[r13 + GprOffset(kRsi)]);
    // Seed rax with the current guest RAX so 8/16-bit loads preserve the
    // untouched upper bits exactly as hardware does.
    c.mov(rax, qword[r13 + GprOffset(kRax)]);

    c.mov(r11, elem);
    {
        Xbyak::Label df_clear;
        c.mov(rdx, qword[r13 + Offsets::Rflags]);
        c.bt(rdx, 10);
        c.jnc(df_clear, LT::T_NEAR);
        c.neg(r11);
        c.L(df_clear);
    }

    auto load_one = [&]() {
        switch (w) {
        case 8:  c.mov(al,  byte[r8]);  break; // preserves RAX[63:8]
        case 16: c.mov(ax,  word[r8]);  break; // preserves RAX[63:16]
        case 32: c.mov(eax, dword[r8]); break; // zero-extends RAX[63:32]
        case 64: c.mov(rax, qword[r8]); break;
        }
        c.add(r8, r11);
    };

    if (!is_rep) {
        load_one();
    } else {
        c.mov(r10, qword[r13 + GprOffset(kRcx)]);
        Xbyak::Label l_top, l_done;
        c.test(r10, r10);
        c.jz(l_done, LT::T_NEAR);
        c.L(l_top);
        load_one();
        c.dec(r10);
        c.jnz(l_top, LT::T_NEAR);
        c.L(l_done);
        c.mov(qword[r13 + GprOffset(kRcx)], r10);
    }

    c.mov(qword[r13 + GprOffset(kRax)], rax); // accumulator
    c.mov(qword[r13 + GprOffset(kRsi)], r8);  // advanced src
    // LODS affects no flags.
    return true;
}

// =============================================================================
// SCAS / REP(E|NE) SCAS — scan string. SCASB/W/D/Q compare AL/AX/EAX/RAX vs
// [RDI] (CMP semantics, full arithmetic flags) and advance RDI by ±elem (DF).
// REPE SCAS scans while equal (memchr-complement / strlen-ish), REPNE SCAS
// scans while not-equal (the classic strlen / memchr building block):
//   REPNE SCASB: repeat while RCX!=0 AND ZF==0  (stop on match or count)
//   REPE  SCASB: repeat while RCX!=0 AND ZF==1  (stop on mismatch or count)
// Final flags reflect the LAST comparison actually performed; a zero-trip REP
// (RCX==0) leaves flags unchanged.
//
// This mirrors EmitCmps exactly, except the left operand is the accumulator
// (RAX) instead of [RSI], and only RDI advances. Host scratch: r9=RDI,
// r10=RCX, r11=step, rax=accumulator, rbx=captured flags.
bool EmitScas(const ZydisDecodedInstruction& insn, Xbyak::CodeGenerator& c) {
    using LT = Xbyak::CodeGenerator::LabelType;
    if (insn.operand_count_visible != 0)
        return false;

    const unsigned w = insn.operand_width;
    if (w != 8 && w != 16 && w != 32 && w != 64)
        return false;
    const int elem = static_cast<int>(w / 8);

    const bool has_repe  = (insn.attributes & ZYDIS_ATTRIB_HAS_REPE) != 0;
    const bool has_repne = (insn.attributes & ZYDIS_ATTRIB_HAS_REPNE) != 0;
    const bool is_rep = has_repe || has_repne;

    constexpr int kRdi = 7, kRax = 0, kRcx = 1;

    c.mov(r9, qword[r13 + GprOffset(kRdi)]);
    // Accumulator into a callee-saved-free scratch we won't disturb across the
    // loop: keep it in r12? r12 is guest-state; use the host stack instead by
    // reloading each iteration is wasteful — instead pin it in rsi (host
    // scratch here; guest RSI is not an operand of SCAS so we may clobber the
    // host rsi register as long as we don't touch the guest RSI slot, which we
    // don't). We still must preserve host rsi for the ABI: save/restore it.
    c.push(rsi);
    c.mov(rsi, qword[r13 + GprOffset(kRax)]); // rsi = accumulator (host scratch)

    c.mov(r11, elem);
    {
        Xbyak::Label df_clear;
        c.mov(rax, qword[r13 + Offsets::Rflags]);
        c.bt(rax, 10);
        c.jnc(df_clear, LT::T_NEAR);
        c.neg(r11);
        c.L(df_clear);
    }

    c.push(rbx);
    c.mov(rbx, qword[r13 + Offsets::Rflags]); // zero-trip default

    auto cmp_and_capture = [&]() {
        switch (w) {
        case 8:  c.cmp(sil, byte[r9]);  break;
        case 16: c.cmp(si,  word[r9]);  break;
        case 32: c.cmp(esi, dword[r9]); break;
        case 64: c.cmp(rsi, qword[r9]); break;
        }
        c.pushfq();
        c.pop(rbx);
    };

    if (!is_rep) {
        cmp_and_capture();
        c.add(r9, r11);
    } else {
        c.mov(r10, qword[r13 + GprOffset(kRcx)]);
        Xbyak::Label l_top, l_done;
        c.test(r10, r10);
        c.jz(l_done, LT::T_NEAR);
        c.L(l_top);
        cmp_and_capture();
        c.add(r9, r11);
        c.dec(r10);
        c.jz(l_done, LT::T_NEAR);
        c.bt(rbx, 6);               // CF = ZF of last compare
        if (has_repe)
            c.jc(l_top, LT::T_NEAR);    // equal -> keep going
        else
            c.jnc(l_top, LT::T_NEAR);   // not-equal -> keep going
        c.L(l_done);
        c.mov(qword[r13 + GprOffset(kRcx)], r10);
    }

    c.mov(qword[r13 + GprOffset(kRdi)], r9);

    // Merge arithmetic flags from the last compare (rbx); preserve guest
    // non-arith bits (incl DF). Identical scheme to EmitCmps.
    constexpr u64 kArith = 0x8D5; // CF|PF|AF|ZF|SF|OF
    c.mov(rax, qword[r13 + Offsets::Rflags]);
    c.mov(r8, ~kArith);
    c.and_(rax, r8);
    c.mov(r9, kArith);
    c.and_(rbx, r9);
    c.or_(rax, rbx);
    c.mov(qword[r13 + Offsets::Rflags], rax);

    c.pop(rbx);
    c.pop(rsi);
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
    if (insn.operand_width != 64 && insn.operand_width != 32 && insn.operand_width != 8)
        return false;

    // 8-bit reg-reg form: `sbb r8, r8` (or adc). Observed at libc
    // 0x8000012b9 inside a borrow-propagating accumulator.
    // Borrow-in from CF is needed exactly as the host does it.
    if (insn.operand_width == 8) {
        if (ops[0].type != ZYDIS_OPERAND_TYPE_REGISTER ||
            ops[1].type != ZYDIS_OPERAND_TYPE_REGISTER)
            return false;
        const int dst_off = ZydisGpr8ToByteOffset(ops[0].reg.value);
        const int src_off = ZydisGpr8ToByteOffset(ops[1].reg.value);
        if (dst_off < 0 || src_off < 0) return false;

        c.mov(al, byte[r13 + dst_off]);
        c.mov(cl, byte[r13 + src_off]);

        // Round-trip flags so host CF is the same as guest CF.
        c.mov(r8, qword[r13 + Offsets::Rflags]);
        c.push(r8);
        c.popfq();

        switch (kind) {
        case AdcSbbKind::Adc:
            c.adc(al, cl);
            break;
        case AdcSbbKind::Sbb:
            c.sbb(al, cl);
            break;
        }

        c.pushfq();
        c.pop(r8);
        c.mov(qword[r13 + Offsets::Rflags], r8);

        // Store the 8-bit result back to the dst byte slot,
        // preserving the surrounding bytes.
        c.mov(byte[r13 + dst_off], al);
        return true;
    }

    // 32-bit forms: reg-reg (`sbb r32, r32`) or reg-imm
    // (`sbb r32, imm8/imm32`). reg-reg first seen at libc 0x808167c40
    // (a 32-bit borrow-propagating subtract), reg-imm at 0x80001021d.
    // The 32-bit-write zero-extends rax automatically, so the qword
    // storeback lands a clean upper half. Flags round-trip through the
    // host so its ADC/SBB reads the guest CF as borrow/carry-in and
    // writes the new CF/OF/SF/ZF/PF/AF at the correct 32-bit width.
    if (insn.operand_width == 32) {
        if (ops[0].type != ZYDIS_OPERAND_TYPE_REGISTER)
            return false;
        const int dst_idx = ZydisGprToIndex(ops[0].reg.value);
        if (dst_idx < 0) return false;

        c.mov(eax, dword[r13 + GprOffset(dst_idx)]);

        // Load the src into ecx (reg) or use an immediate directly.
        bool src_is_imm = false;
        u32 imm = 0;
        if (ops[1].type == ZYDIS_OPERAND_TYPE_REGISTER) {
            const int src_idx = ZydisGprToIndex(ops[1].reg.value);
            if (src_idx < 0) return false;
            c.mov(ecx, dword[r13 + GprOffset(src_idx)]);
        } else if (ops[1].type == ZYDIS_OPERAND_TYPE_IMMEDIATE) {
            src_is_imm = true;
            imm = static_cast<u32>(ops[1].imm.value.s); // sign-extended
        } else {
            return false;
        }

        c.mov(r8, qword[r13 + Offsets::Rflags]);
        c.push(r8);
        c.popfq();

        switch (kind) {
        case AdcSbbKind::Adc:
            if (src_is_imm) c.adc(eax, imm); else c.adc(eax, ecx);
            break;
        case AdcSbbKind::Sbb:
            if (src_is_imm) c.sbb(eax, imm); else c.sbb(eax, ecx);
            break;
        }

        c.pushfq();
        c.pop(r8);
        c.mov(qword[r13 + Offsets::Rflags], r8);

        c.mov(qword[r13 + GprOffset(dst_idx)], rax);
        return true;
    }

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
    // Or/And/Xor/Add/Sub with mem-dst DO write back. The typical patterns are
    // `or byte[reg+disp], imm8` (set-flag idiom, glibc bitfield updates) and
    // `or byte[mem], r8` (observed in a loaded system module, CUSA02394
    // 0x80866ae31). Both the immediate and register source forms are handled.
    if (ops[0].type == ZYDIS_OPERAND_TYPE_MEMORY) {
        const bool is_readonly = (kind == NarrowArithKind::Cmp ||
                                  kind == NarrowArithKind::Test);
        const bool is_writeback_op =
            (kind == NarrowArithKind::Or  || kind == NarrowArithKind::And ||
             kind == NarrowArithKind::Xor || kind == NarrowArithKind::Add ||
             kind == NarrowArithKind::Sub);
        const bool is_writeback =
            is_writeback_op &&
            (ops[1].type == ZYDIS_OPERAND_TYPE_IMMEDIATE ||
             ops[1].type == ZYDIS_OPERAND_TYPE_REGISTER);
        if (!is_readonly && !is_writeback) {
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

        // Writeback path: read [mem] byte, OP it with the source (imm or r8),
        // store back. Stash EA in r10 so the rflags round-trip's use of rdx
        // doesn't clobber it. Load the source into cl BEFORE the round-trip.
        c.mov(r10, rdx);
        c.mov(al, byte[r10]);
        if (ops[1].type == ZYDIS_OPERAND_TYPE_IMMEDIATE) {
            c.mov(cl, static_cast<u8>(ops[1].imm.value.u & 0xFFu));
        } else { // register source (high-byte aware)
            const int src_off = ZydisGpr8ToByteOffset(ops[1].reg.value);
            if (src_off < 0)
                return false;
            c.mov(cl, byte[r13 + src_off]);
        }

        c.mov(rdx, qword[r13 + Offsets::Rflags]);
        c.push(rdx);
        c.popfq();

        switch (kind) {
        case NarrowArithKind::Or:
            c.or_(al, cl);
            break;
        case NarrowArithKind::And:
            c.and_(al, cl);
            break;
        case NarrowArithKind::Xor:
            c.xor_(al, cl);
            break;
        case NarrowArithKind::Add:
            c.add(al, cl);
            break;
        case NarrowArithKind::Sub:
            c.sub(al, cl);
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

    const bool is_readonly =
        (kind == NarrowArithKind::Cmp || kind == NarrowArithKind::Test);

    // ---- Memory-destination/lhs form (e.g. `cmp word [mem], imm`) ----
    // The lhs is a memory operand. We only support this for the
    // read-only ops (CMP/TEST) right now — a read-modify-write to
    // memory (ADD/SUB/AND/... word[mem], x) would need the result
    // stored back to [mem], which no game has exercised yet. Reject
    // those rather than silently dropping the writeback.
    if (ops[0].type == ZYDIS_OPERAND_TYPE_MEMORY) {
        if (!is_readonly)
            return false;
        // Compute the lhs address (clobbers rax), read the word into
        // ax, then load the rhs into cx. mem,mem is not a real form.
        if (!EmitEffectiveAddress(ops[0].mem, next_rip, c))
            return false;
        c.mov(ax, word[rdx]);

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
        c.push(rdx);
        c.popfq();
        if (kind == NarrowArithKind::Cmp) c.cmp(ax, cx);
        else                              c.test(ax, cx);
        c.pushfq();
        c.pop(rdx);
        c.mov(qword[r13 + Offsets::Rflags], rdx);
        return true;
    }

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

    // Register destination. Each sub-path below computes the precise
    // byte offset (via ZydisGpr8ToByteOffset) so high-byte destination
    // registers (AH/CH/DH/BH) are handled — we deliberately do NOT
    // gate on ZydisGprToIndex here, since it returns -1 for those.
    if (dst.type == ZYDIS_OPERAND_TYPE_REGISTER) {
        if (src.type == ZYDIS_OPERAND_TYPE_REGISTER) {
            // mov r8, r8 — read the source byte, write the dest byte.
            // We must use the 8-bit byte-offset helper, not GprOffset:
            // high-byte registers (AH/CH/DH/BH) live at parent-slot
            // offset+1, and ZydisGprToIndex doesn't even recognize them
            // (it returns -1), so the old GprOffset-based code rejected
            // any mov involving AH/CH/DH/BH. The byte store/load touches
            // exactly one byte, preserving the rest of each parent slot.
            const int dst_off = ZydisGpr8ToByteOffset(dst.reg.value);
            const int src_off = ZydisGpr8ToByteOffset(src.reg.value);
            if (dst_off < 0 || src_off < 0)
                return false;
            c.mov(al, byte[r13 + src_off]);
            c.mov(byte[r13 + dst_off], al);
            return true;
        }
        if (src.type == ZYDIS_OPERAND_TYPE_IMMEDIATE) {
            // mov r8, imm8 — store immediate byte into the dest byte
            // slot (byte-offset helper for high-byte dest regs).
            const int dst_off = ZydisGpr8ToByteOffset(dst.reg.value);
            if (dst_off < 0)
                return false;
            c.mov(byte[r13 + dst_off], static_cast<u8>(src.imm.value.u & 0xFF));
            return true;
        }
        if (src.type == ZYDIS_OPERAND_TYPE_MEMORY) {
            // mov r8, byte[mem] — compute address (clobbers rax),
            // then load byte and write the dest byte slot.
            // We use cl for the loaded byte because rcx is preserved
            // by EmitEffectiveAddress; al would be valid too but is
            // a scratch the helper actively uses.
            const int dst_off = ZydisGpr8ToByteOffset(dst.reg.value);
            if (dst_off < 0)
                return false;
            if (!EmitEffectiveAddress(src.mem, next_rip, c))
                return false;
            c.mov(cl, byte[rdx]);
            c.mov(byte[r13 + dst_off], cl);
            return true;
        }
        return false;
    }

    // Memory destination.
    if (dst.type == ZYDIS_OPERAND_TYPE_MEMORY) {
        if (src.type == ZYDIS_OPERAND_TYPE_REGISTER) {
            // mov byte[mem], r8 — compute address (rdx), then read
            // src byte into cl (rcx preserved through
            // EmitEffectiveAddress), then store. Byte-offset helper so
            // a high-byte source register reads the correct byte.
            const int src_off = ZydisGpr8ToByteOffset(src.reg.value);
            if (src_off < 0)
                return false;
            if (!EmitEffectiveAddress(dst.mem, next_rip, c))
                return false;
            c.mov(cl, byte[r13 + src_off]);
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
              u64 next_rip, Xbyak::CodeGenerator& c) {
    if (insn.operand_width != 64)
        return false;
    const auto& src = ops[0];

    // Load the value to push into rax. Three source forms:
    //   register  — read the guest GPR slot.
    //   immediate — PUSH imm8/imm32 sign-extends to 64 bits; Zydis already
    //               sign-extends into imm.value.s, so use it directly.
    //   memory    — PUSH r/m64; dereference the effective address. Compute the
    //               address FIRST (EmitEffectiveAddress writes rdx, scratches
    //               rax) and load the value into rax before we touch RSP.
    if (src.type == ZYDIS_OPERAND_TYPE_REGISTER) {
        const int src_idx = ZydisGprToIndex(src.reg.value);
        if (src_idx < 0)
            return false;
        c.mov(rax, qword[r13 + GprOffset(src_idx)]);
    } else if (src.type == ZYDIS_OPERAND_TYPE_IMMEDIATE) {
        c.mov(rax, src.imm.value.s); // sign-extended immediate
    } else if (src.type == ZYDIS_OPERAND_TYPE_MEMORY) {
        if (!EmitEffectiveAddress(src.mem, next_rip, c))
            return false;
        c.mov(rax, qword[rdx]);
    } else {
        return false;
    }

    // rdx = guest_rsp, decrement, store, write back.
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
        insn.mnemonic != ZYDIS_MNEMONIC_VMOVDQA && insn.mnemonic != ZYDIS_MNEMONIC_VMOVAPS &&
        insn.mnemonic != ZYDIS_MNEMONIC_VMOVNTDQA) {
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
        //
        // VMOVNTDQA is the *non-temporal* aligned load (the load-side
        // partner of VMOVNTDQ's store). The "NT" is a cache-locality
        // hint with no architectural effect — for the data actually
        // moved it is bitwise identical to VMOVDQA. It is load-only
        // (reg ← mem); there is no non-temporal-load store direction,
        // so only the reg←mem path below applies (a mem-dst VMOVNTDQA
        // isn't a real encoding and would fall through to false).
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

/// VPEXTRD — extract one 32-bit dword from an XMM register, selected
/// by imm8, into a GPR (r32) or memory (m32).
///
///   VPEXTRD r/m32, xmm, imm8
///
/// imm8[1:0] selects the dword lane (0..3); higher imm8 bits are
/// ignored (architecturally masked). dword 0 is the low 32 bits.
///
/// GPR destination zero-extends to the full 64-bit register (writing
/// r32 clears bits 63:32), so we store a full qword to the guest GPR
/// slot with the upper half cleared — same canonical zero-extension
/// VMOVD uses. Memory destination writes exactly 4 bytes.
///
/// Implementation: rather than round-trip through a scratch xmm and
/// run host VPEXTRD, we read the selected dword straight out of the
/// GuestState ymm storage. The 4 dwords live at byte offsets 0,4,8,12
/// within the low 128 bits; we compute the dword's address and load
/// it directly. (This mirrors the VMOVD GPR←XMM path, just at an
/// imm-selected offset instead of always dword 0.) For the memory-
/// destination form we compute the effective address FIRST, because
/// EmitEffectiveAddress uses rax as scratch — so the extracted dword
/// is loaded into ecx after the address is settled in rdx.
///
/// First instance of the PEXTR family (VPEXTRB/W/D/Q). Kept as a
/// standalone emitter rather than a parameterised helper for now —
/// the element widths differ enough (byte/word/dword/qword, with
/// different GPR write widths and zero-extension rules) that a shared
/// helper would be mostly branches. Revisit if a second member lands.
bool EmitVpextrd(const ZydisDecodedInstruction& insn, const ZydisDecodedOperand* ops,
                 u64 next_rip, Xbyak::CodeGenerator& c) {
    if (insn.mnemonic != ZYDIS_MNEMONIC_VPEXTRD) return false;
    if (insn.operand_count_visible != 3) return false;
    if (ops[1].type != ZYDIS_OPERAND_TYPE_REGISTER) return false;
    if (ops[2].type != ZYDIS_OPERAND_TYPE_IMMEDIATE) return false;

    const int src_vec = ZydisVecToIndex(ops[1].reg.value);
    if (src_vec < 0) return false;

    // Lane select: low 2 bits of imm8 (4 dwords in a 128-bit xmm).
    const unsigned lane = static_cast<unsigned>(ops[2].imm.value.u) & 0x3u;
    // Byte offset of the selected dword within the ymm storage:
    // dword `lane` sits at byte (lane*4) from the start of chunk 0.
    const int dword_off = static_cast<int>(YmmChunkOffset(src_vec, 0)) +
                          static_cast<int>(lane) * 4;

    if (ops[0].type == ZYDIS_OPERAND_TYPE_REGISTER) {
        const int dst_gpr = ZydisGprToIndex(ops[0].reg.value);
        if (dst_gpr < 0) return false;
        // Load the 32-bit dword into eax (zero-extends rax to 64 bits),
        // then store full qword: a 32-bit GPR write zero-extends, so
        // the guest slot gets the canonical value with bits 63:32 = 0.
        c.mov(eax, dword[r13 + dword_off]);
        c.mov(qword[r13 + GprOffset(dst_gpr)], rax);
    } else if (ops[0].type == ZYDIS_OPERAND_TYPE_MEMORY) {
        // Compute the destination address FIRST: EmitEffectiveAddress
        // uses rax as scratch (index scaling), so we must not hold the
        // extracted dword in eax across it. Address lands in rdx.
        if (!EmitEffectiveAddress(ops[0].mem, next_rip, c))
            return false;
        // Now load the selected dword and write exactly 4 bytes. The
        // r13-relative load is independent of rdx, and ecx is free
        // scratch here (not used by EmitEffectiveAddress's result).
        c.mov(ecx, dword[r13 + dword_off]);
        c.mov(dword[rdx], ecx);
    } else {
        return false;
    }
    return true;
}

/// VPEXTRQ — extract one 64-bit qword from an XMM register, selected
/// by imm8, into a GPR (r64) or memory (m64).
///
///   VPEXTRQ r/m64, xmm, imm8
///
/// imm8[0] selects the qword lane (0..1); higher bits are ignored
/// (architecturally masked). qword 0 is the low 64 bits of the xmm.
///
/// The 64-bit sibling of VPEXTRD. Differences from the dword form:
///   - only 2 lanes (imm8 low BIT, not low 2 bits); offset is lane*8.
///   - a GPR destination is written at full 64-bit width, so there is
///     no zero-extension step — the qword IS the whole register.
///   - the memory destination writes 8 bytes.
/// We read the selected qword straight out of the GuestState ymm
/// storage (the 2 qwords live at byte offsets 0 and 8 within the low
/// 128 bits), mirroring VPEXTRD. For the memory form we compute the
/// effective address FIRST (EmitEffectiveAddress scratches rax/rdx),
/// then load the extracted qword into rcx and store it.
///
/// First observed as a reg,reg form (length 6) in the CUSA02394
/// "WE ARE DOOMED" eboot at guest 0x8001f3404 — the instruction right
/// after the VPMOVZXDQ in the same block.
bool EmitVpextrq(const ZydisDecodedInstruction& insn, const ZydisDecodedOperand* ops,
                 u64 next_rip, Xbyak::CodeGenerator& c) {
    if (insn.mnemonic != ZYDIS_MNEMONIC_VPEXTRQ) return false;
    if (insn.operand_count_visible != 3) return false;
    if (ops[1].type != ZYDIS_OPERAND_TYPE_REGISTER) return false;
    if (ops[2].type != ZYDIS_OPERAND_TYPE_IMMEDIATE) return false;

    const int src_vec = ZydisVecToIndex(ops[1].reg.value);
    if (src_vec < 0) return false;

    // Lane select: low bit of imm8 (2 qwords in a 128-bit xmm).
    const unsigned lane = static_cast<unsigned>(ops[2].imm.value.u) & 0x1u;
    const int qword_off = static_cast<int>(YmmChunkOffset(src_vec, 0)) +
                          static_cast<int>(lane) * 8;

    if (ops[0].type == ZYDIS_OPERAND_TYPE_REGISTER) {
        const int dst_gpr = ZydisGprToIndex(ops[0].reg.value);
        if (dst_gpr < 0) return false;
        // Full 64-bit move: the qword is the entire destination register.
        c.mov(rax, qword[r13 + qword_off]);
        c.mov(qword[r13 + GprOffset(dst_gpr)], rax);
    } else if (ops[0].type == ZYDIS_OPERAND_TYPE_MEMORY) {
        // Address first (EmitEffectiveAddress scratches rax); result in rdx.
        if (!EmitEffectiveAddress(ops[0].mem, next_rip, c))
            return false;
        // rcx is free scratch; load the selected qword and store 8 bytes.
        c.mov(rcx, qword[r13 + qword_off]);
        c.mov(qword[rdx], rcx);
    } else {
        return false;
    }
    return true;
}

/// VEXTRACTPS — extract one 32-bit (single-precision float) element from an
/// XMM register, selected by imm8, into a GPR (r32) or memory (m32).
///
///   VEXTRACTPS r/m32, xmm, imm8     (VEX.128.66.0F3A.WIG 17 /r ib)
///
/// imm8[1:0] selects the dword lane (0..3); higher bits are ignored. The data
/// movement is bit-identical to VPEXTRD — a single 32-bit lane copied out — so
/// this mirrors EmitVpextrd exactly:
///   - GPR destination zero-extends to the full 64-bit register (writing r32
///     clears bits 63:32), so we store a full qword with the upper half clear.
///   - memory destination writes exactly 4 bytes; we compute the effective
///     address FIRST (EmitEffectiveAddress scratches rax), then load the
///     selected dword into ecx and store it.
///
/// First observed as a mem,reg form (length 11, SIB+disp32 destination) in the
/// CUSA02394 "WE ARE DOOMED" eboot at guest 0x8001f3480, in the same vector
/// block as the VPMOVZXDQ/VPEXTRQ/VMINSS fixed earlier.
bool EmitVextractps(const ZydisDecodedInstruction& insn, const ZydisDecodedOperand* ops,
                    u64 next_rip, Xbyak::CodeGenerator& c) {
    if (insn.mnemonic != ZYDIS_MNEMONIC_VEXTRACTPS) return false;
    if (insn.operand_count_visible != 3) return false;
    if (ops[1].type != ZYDIS_OPERAND_TYPE_REGISTER) return false;
    if (ops[2].type != ZYDIS_OPERAND_TYPE_IMMEDIATE) return false;

    const int src_vec = ZydisVecToIndex(ops[1].reg.value);
    if (src_vec < 0) return false;

    // Lane select: low 2 bits of imm8 (4 dwords in a 128-bit xmm).
    const unsigned lane = static_cast<unsigned>(ops[2].imm.value.u) & 0x3u;
    const int dword_off = static_cast<int>(YmmChunkOffset(src_vec, 0)) +
                          static_cast<int>(lane) * 4;

    if (ops[0].type == ZYDIS_OPERAND_TYPE_REGISTER) {
        const int dst_gpr = ZydisGprToIndex(ops[0].reg.value);
        if (dst_gpr < 0) return false;
        // 32-bit GPR write zero-extends to 64 bits.
        c.mov(eax, dword[r13 + dword_off]);
        c.mov(qword[r13 + GprOffset(dst_gpr)], rax);
    } else if (ops[0].type == ZYDIS_OPERAND_TYPE_MEMORY) {
        // Address first (EmitEffectiveAddress scratches rax); result in rdx.
        if (!EmitEffectiveAddress(ops[0].mem, next_rip, c))
            return false;
        c.mov(ecx, dword[r13 + dword_off]);
        c.mov(dword[rdx], ecx);
    } else {
        return false;
    }
    return true;
}

/// VPINSRD — insert a 32-bit dword (from a GPR or m32) into an
/// imm8-selected lane of an XMM. The inverse of VPEXTRD.
///
///   VPINSRD xmm1, xmm2, r/m32, imm8
///
/// Result: dst = src1, with dword lane (imm8 & 3) replaced by the
/// 32-bit source value. As a VEX-encoded op, bits 255:128 of the
/// destination YMM are zeroed.
///
/// Implementation: copy src1's low 128 bits into dst (chunks 0,1),
/// then overwrite the selected dword, then zero the upper YMM. We
/// must order reads before writes so the dst==src1 in-place form
/// (vpinsrd xmm0, xmm0, eax, imm) is correct — the chunk copy is a
/// self-copy in that case, harmless. The 32-bit source is read into
/// a GPR FIRST when it comes from memory, because EmitEffectiveAddress
/// uses rax/rdx as scratch (same ordering lesson as VPEXTRD).
///
/// First instance of the PINSR family (VPINSRB/W/D/Q). Standalone for
/// now, mirroring the VPEXTRD decision — the per-width differences
/// (insert granularity, source register width) make a shared helper
/// mostly branches. Revisit if a second member lands.
bool EmitVpinsrd(const ZydisDecodedInstruction& insn, const ZydisDecodedOperand* ops,
                 u64 next_rip, Xbyak::CodeGenerator& c) {
    if (insn.mnemonic != ZYDIS_MNEMONIC_VPINSRD) return false;
    if (insn.operand_count_visible != 4) return false;
    if (ops[0].type != ZYDIS_OPERAND_TYPE_REGISTER) return false;
    if (ops[1].type != ZYDIS_OPERAND_TYPE_REGISTER) return false;
    if (ops[3].type != ZYDIS_OPERAND_TYPE_IMMEDIATE) return false;

    const int dst_vec = ZydisVecToIndex(ops[0].reg.value);
    const int src1_vec = ZydisVecToIndex(ops[1].reg.value);
    if (dst_vec < 0 || src1_vec < 0) return false;

    const unsigned lane = static_cast<unsigned>(ops[3].imm.value.u) & 0x3u;

    // Step 1: get the 32-bit insert value into ecx (a register that
    // survives the subsequent chunk copies). For a memory source we
    // must compute the EA and load BEFORE touching dst, and before
    // anything that uses rax/rdx as scratch.
    if (ops[2].type == ZYDIS_OPERAND_TYPE_REGISTER) {
        const int src2_gpr = ZydisGprToIndex(ops[2].reg.value);
        if (src2_gpr < 0) return false;
        c.mov(ecx, dword[r13 + GprOffset(src2_gpr)]);
    } else if (ops[2].type == ZYDIS_OPERAND_TYPE_MEMORY) {
        if (!EmitEffectiveAddress(ops[2].mem, next_rip, c))
            return false;
        c.mov(ecx, dword[rdx]);
    } else {
        return false;
    }

    // Step 2: copy src1's low 128 bits (chunks 0,1) into dst. Reads of
    // src1 happen before writes of dst; if dst==src1 these are
    // self-copies (harmless). Use rax as the transfer register.
    c.mov(rax, qword[r13 + YmmChunkOffset(src1_vec, 0)]);
    c.mov(qword[r13 + YmmChunkOffset(dst_vec, 0)], rax);
    c.mov(rax, qword[r13 + YmmChunkOffset(src1_vec, 1)]);
    c.mov(qword[r13 + YmmChunkOffset(dst_vec, 1)], rax);

    // Step 3: overwrite the selected dword. dword `lane` is at byte
    // offset lane*4 from chunk 0 of dst.
    const int dword_off = static_cast<int>(YmmChunkOffset(dst_vec, 0)) +
                          static_cast<int>(lane) * 4;
    c.mov(dword[r13 + dword_off], ecx);

    // Step 4: VEX zero of bits 255:128 (chunks 2,3).
    c.xor_(rax, rax);
    c.mov(qword[r13 + YmmChunkOffset(dst_vec, 2)], rax);
    c.mov(qword[r13 + YmmChunkOffset(dst_vec, 3)], rax);
    return true;
}

// =============================================================================
// VINSERTPS — insert a single packed single-precision float lane.
//
//   VINSERTPS xmm1, xmm2, xmm3/m32, imm8   (VEX.128.66.0F3A.WIG 21 /r ib)
//
// The imm8 control byte has three fields:
//   bits [7:6] = COUNT_S — source dword lane to read from xmm3 (register form
//                only; for the m32 memory form the value is simply the loaded
//                dword and these bits are ignored).
//   bits [5:4] = COUNT_D — destination dword lane in xmm1 to overwrite.
//   bits [3:0] = ZMASK   — after the insert, each set bit zeroes the
//                corresponding dword lane of the destination (bit0->lane0 …
//                bit3->lane3).
//
// Operation: dst.low128 = src1.low128; dst.dword[COUNT_D] = selected source
// dword; then for each i in 0..3, if ZMASK bit i set, dst.dword[i] = 0; then
// (VEX) bits 255:128 of the dst YMM are zeroed.
//
// First observed as a reg,reg form (length 6) in the CUSA02394 "WE ARE DOOMED"
// eboot at guest 0x80012f8d6. Modeled on EmitVpinsrd's chunk-copy + lane-store
// layout. Host scratch: ecx holds the source dword, rax the transfer/zero reg.
bool EmitVinsertps(const ZydisDecodedInstruction& insn, const ZydisDecodedOperand* ops,
                   u64 next_rip, Xbyak::CodeGenerator& c) {
    if (insn.mnemonic != ZYDIS_MNEMONIC_VINSERTPS) return false;
    if (insn.operand_count_visible != 4) return false;
    if (ops[0].type != ZYDIS_OPERAND_TYPE_REGISTER) return false;
    if (ops[1].type != ZYDIS_OPERAND_TYPE_REGISTER) return false;
    if (ops[3].type != ZYDIS_OPERAND_TYPE_IMMEDIATE) return false;

    const int dst_vec = ZydisVecToIndex(ops[0].reg.value);
    const int src1_vec = ZydisVecToIndex(ops[1].reg.value);
    if (dst_vec < 0 || src1_vec < 0) return false;

    const unsigned imm = static_cast<unsigned>(ops[3].imm.value.u) & 0xFFu;
    const unsigned count_s = (imm >> 6) & 0x3u;
    const unsigned count_d = (imm >> 4) & 0x3u;
    const unsigned zmask = imm & 0xFu;

    // Step 1: load the source dword into ecx BEFORE touching dst (so dst may
    // safely alias src1 or src2). For a register source, pick dword COUNT_S;
    // for a memory source, load the single m32 (COUNT_S ignored per the ISA).
    if (ops[2].type == ZYDIS_OPERAND_TYPE_REGISTER) {
        const int src2_vec = ZydisVecToIndex(ops[2].reg.value);
        if (src2_vec < 0) return false;
        const int src_dword_off = static_cast<int>(YmmChunkOffset(src2_vec, 0)) +
                                  static_cast<int>(count_s) * 4;
        c.mov(ecx, dword[r13 + src_dword_off]);
    } else if (ops[2].type == ZYDIS_OPERAND_TYPE_MEMORY) {
        if (!EmitEffectiveAddress(ops[2].mem, next_rip, c))
            return false;
        c.mov(ecx, dword[rdx]);
    } else {
        return false;
    }

    // Step 2: copy src1's low 128 bits (chunks 0,1) into dst. Self-copy is
    // harmless when dst==src1.
    c.mov(rax, qword[r13 + YmmChunkOffset(src1_vec, 0)]);
    c.mov(qword[r13 + YmmChunkOffset(dst_vec, 0)], rax);
    c.mov(rax, qword[r13 + YmmChunkOffset(src1_vec, 1)]);
    c.mov(qword[r13 + YmmChunkOffset(dst_vec, 1)], rax);

    // Step 3: overwrite the destination dword lane COUNT_D.
    const int dst_dword_off = static_cast<int>(YmmChunkOffset(dst_vec, 0)) +
                              static_cast<int>(count_d) * 4;
    c.mov(dword[r13 + dst_dword_off], ecx);

    // Step 4: apply the zero mask — zero each selected dword lane. This runs
    // AFTER the insert, so a ZMASK bit covering COUNT_D wins (zeroes it).
    if (zmask != 0) {
        c.xor_(eax, eax);
        for (unsigned i = 0; i < 4; ++i) {
            if (zmask & (1u << i)) {
                const int off = static_cast<int>(YmmChunkOffset(dst_vec, 0)) +
                                static_cast<int>(i) * 4;
                c.mov(dword[r13 + off], eax);
            }
        }
    }

    // Step 5: VEX zero of bits 255:128 (chunks 2,3).
    c.xor_(rax, rax);
    c.mov(qword[r13 + YmmChunkOffset(dst_vec, 2)], rax);
    c.mov(qword[r13 + YmmChunkOffset(dst_vec, 3)], rax);
    return true;
}

/// VMOVQ — 64-bit move involving an XMM register. Handles the three
/// reg-reg directions plus the two memory forms.
///
///   xmm ← xmm: copies low 64 of src to low 64 of dst, zeroes the
///              rest of the dst YMM (bits 255:64).
///   xmm ← r64: copies r64 to dst low 64, zeroes bits 255:64 of YMM.
///   r64 ← xmm: copies xmm low 64 to r64 (full 64-bit write, no
///              zero-extend needed since it IS the full GPR).
///   xmm ← m64: loads 64 bits to dst low 64, zeroes bits 255:64.
///   m64 ← xmm: stores xmm low 64 to memory (8 bytes).
///
/// Forms route by sniffing operand register classes (same trick VMOVD
/// uses) and operand types. The GPR↔XMM variants are the 3-byte-VEX
/// reg,reg form (length 5); xmm←xmm is 2-byte VEX (length 4); the
/// memory forms use a ModRM with addressing bytes (length 8 with
/// RIP-relative, as first seen at libc 0x800263e4d). The memory paths
/// compute the effective address before moving data, since
/// EmitEffectiveAddress scratches rax.
bool EmitVmovq(const ZydisDecodedInstruction& insn, const ZydisDecodedOperand* ops,
               u64 next_rip, Xbyak::CodeGenerator& c) {
    if (insn.mnemonic != ZYDIS_MNEMONIC_VMOVQ)
        return false;

    // ---- Memory forms (added when the reg-mem load was first seen) ----
    //
    // VMOVQ xmm, m64  : load 64 bits into dst.low64, zero bits 255:64
    //                   of the dst YMM (VEX architectural).
    // VMOVQ m64, xmm  : store xmm.low64 to memory (8 bytes, no other
    //                   effects).
    //
    // EmitEffectiveAddress writes the address into rdx and uses rax as
    // scratch, so we compute the EA first, then move data through rax.
    if (ops[0].type == ZYDIS_OPERAND_TYPE_REGISTER &&
        ops[1].type == ZYDIS_OPERAND_TYPE_MEMORY) {
        const int dst_vec = ZydisVecToIndex(ops[0].reg.value);
        if (dst_vec < 0) return false;
        if (!EmitEffectiveAddress(ops[1].mem, next_rip, c))
            return false;
        c.mov(rax, qword[rdx]);                              // load m64
        c.mov(qword[r13 + YmmChunkOffset(dst_vec, 0)], rax); // dst.low64
        c.xor_(rax, rax);                                    // zero 255:64
        c.mov(qword[r13 + YmmChunkOffset(dst_vec, 1)], rax);
        c.mov(qword[r13 + YmmChunkOffset(dst_vec, 2)], rax);
        c.mov(qword[r13 + YmmChunkOffset(dst_vec, 3)], rax);
        return true;
    }
    if (ops[0].type == ZYDIS_OPERAND_TYPE_MEMORY &&
        ops[1].type == ZYDIS_OPERAND_TYPE_REGISTER) {
        const int src_vec = ZydisVecToIndex(ops[1].reg.value);
        if (src_vec < 0) return false;
        // Load the source quad BEFORE computing the EA (EA clobbers
        // rax); stash it in rcx, which the EA computation leaves alone.
        c.mov(rcx, qword[r13 + YmmChunkOffset(src_vec, 0)]);
        if (!EmitEffectiveAddress(ops[0].mem, next_rip, c))
            return false;
        c.mov(qword[rdx], rcx);                              // store 8 bytes
        return true;
    }

    // ---- Register-register forms ----
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

/// VMOVSS — scalar single-precision FP move.
///
/// Three operand forms:
///   1. VMOVSS xmm, m32           — load: dst.low32 = [mem], zero the
///                                  rest of the dst XMM, zero the upper
///                                  YMM lane (VEX-128 architectural).
///   2. VMOVSS m32, xmm           — store: [mem] = xmm.low32. No other
///                                  effects (memory is just a u32).
///   3. VMOVSS xmm1, xmm2, xmm3   — reg-reg-reg: dst.low32 = src2.low32,
///                                  dst[127:32] = src1[127:32] (i.e.
///                                  preserves upper from the second
///                                  operand), and bits 255:128 of YMM
///                                  zeroed (VEX-128).
///
/// We don't actually invoke host VMOVSS — the operation is small
/// enough to do directly via 32-bit GPR moves. That keeps us off the
/// host SSE/FP register file entirely, which is simpler and avoids
/// any MXCSR / denormal handling questions (since we're just shuffling
/// bits, no FP arithmetic occurs).
///
/// The 8-byte encoding observed at libc 0x8002a08b3 is the reg-mem
/// load form with RIP-relative addressing (VEX + opcode + ModRM +
/// disp32 = 8 bytes). RIP-relative is handled transparently by
/// EmitEffectiveAddress.
bool EmitVmovss(const ZydisDecodedInstruction& insn, const ZydisDecodedOperand* ops, u64 next_rip,
                Xbyak::CodeGenerator& c) {
    if (insn.mnemonic != ZYDIS_MNEMONIC_VMOVSS) return false;

    // Form 1: VMOVSS xmm, m32 — load from memory into xmm.
    if (insn.operand_count_visible == 2 &&
        ops[0].type == ZYDIS_OPERAND_TYPE_REGISTER &&
        ops[1].type == ZYDIS_OPERAND_TYPE_MEMORY) {
        const int dst_vec = ZydisVecToIndex(ops[0].reg.value);
        if (dst_vec < 0) return false;
        if (!EmitEffectiveAddress(ops[1].mem, next_rip, c))
            return false;
        // rdx = effective address. Load 32-bit value into eax, which
        // zero-extends bits 63:32 of rax — so the qword storeback to
        // chunk 0 lands {value, 0} cleanly with no upper-32 leak.
        c.mov(eax, dword[rdx]);
        c.mov(qword[r13 + YmmChunkOffset(dst_vec, 0)], rax);
        // Zero the remaining 7 bytes of the lane (chunk 0 high half
        // is already zero from the eax-write zero-extension; we need
        // chunks 1, 2, 3 zeroed for the VEX-128 architectural
        // "zero upper YMM" requirement).
        c.xor_(rax, rax);
        c.mov(qword[r13 + YmmChunkOffset(dst_vec, 1)], rax);
        c.mov(qword[r13 + YmmChunkOffset(dst_vec, 2)], rax);
        c.mov(qword[r13 + YmmChunkOffset(dst_vec, 3)], rax);
        return true;
    }

    // Form 2: VMOVSS m32, xmm — store xmm low 32 to memory.
    if (insn.operand_count_visible == 2 &&
        ops[0].type == ZYDIS_OPERAND_TYPE_MEMORY &&
        ops[1].type == ZYDIS_OPERAND_TYPE_REGISTER) {
        const int src_vec = ZydisVecToIndex(ops[1].reg.value);
        if (src_vec < 0) return false;
        if (!EmitEffectiveAddress(ops[0].mem, next_rip, c))
            return false;
        // rdx = address. Pre-load src.low32 into ecx FIRST (the load
        // doesn't touch rdx) then store. Going through ecx keeps the
        // dependency chain on the 32-bit register, matching the
        // architectural "only 32 bits read" semantic.
        c.mov(ecx, dword[r13 + YmmChunkOffset(src_vec, 0)]);
        c.mov(dword[rdx], ecx);
        return true;
    }

    // Form 3: VMOVSS xmm1, xmm2, xmm3 — reg-reg-reg merge.
    if (insn.operand_count_visible == 3 &&
        ops[0].type == ZYDIS_OPERAND_TYPE_REGISTER &&
        ops[1].type == ZYDIS_OPERAND_TYPE_REGISTER &&
        ops[2].type == ZYDIS_OPERAND_TYPE_REGISTER) {
        const int dst_vec = ZydisVecToIndex(ops[0].reg.value);
        const int src1_vec = ZydisVecToIndex(ops[1].reg.value);
        const int src2_vec = ZydisVecToIndex(ops[2].reg.value);
        if (dst_vec < 0 || src1_vec < 0 || src2_vec < 0) return false;

        // dst.low32 = src2.low32; dst[127:32] = src1[127:32].
        // The composite: chunk 0 = (src2.low32 in low half) | (src1
        // upper 32 in high half). Build via two 32-bit reads and a
        // shift+or.
        c.mov(eax, dword[r13 + YmmChunkOffset(src2_vec, 0)]);          // eax = src2.low32 (rax bits 63:32 zeroed)
        c.mov(edx, dword[r13 + YmmChunkOffset(src1_vec, 0) + 4]);      // edx = src1.bits[63:32] (rdx bits 63:32 zeroed)
        c.shl(rdx, 32);
        c.or_(rax, rdx);
        c.mov(qword[r13 + YmmChunkOffset(dst_vec, 0)], rax);
        // chunk 1 = src1[127:64] (preserved from first source).
        c.mov(rax, qword[r13 + YmmChunkOffset(src1_vec, 1)]);
        c.mov(qword[r13 + YmmChunkOffset(dst_vec, 1)], rax);
        // VEX-128: zero bits 255:128 of the dst YMM.
        c.xor_(rax, rax);
        c.mov(qword[r13 + YmmChunkOffset(dst_vec, 2)], rax);
        c.mov(qword[r13 + YmmChunkOffset(dst_vec, 3)], rax);
        return true;
    }

    return false;
}

/// VMOVSD — scalar double-precision FP move. SD counterpart to VMOVSS.
///
/// Three operand forms (same shape as VMOVSS but at the 64-bit boundary):
///   1. VMOVSD xmm, m64          — dst.low64 = [mem], zero rest of xmm,
///                                 zero upper YMM lane (VEX-128).
///   2. VMOVSD m64, xmm          — [mem] = xmm.low64. Surrounding memory
///                                 untouched.
///   3. VMOVSD xmm1, xmm2, xmm3  — dst.low64 = src2.low64,
///                                 dst[127:64] = src1[127:64], upper YMM zeroed.
///
/// Cleaner than VMOVSS at the implementation level because every transfer
/// here is a clean qword (the SD boundary aligns naturally with the 8-byte
/// chunks of GuestState). VMOVSS had to shift+or to compose chunk 0 from
/// two 32-bit halves; VMOVSD just moves whole chunks.
///
/// As with VMOVSS, no host FP register file involved — pure GPR-relayed
/// transfer. Just shuffling 64-bit lumps; no MXCSR/denormal concerns.
bool EmitVmovsd(const ZydisDecodedInstruction& insn, const ZydisDecodedOperand* ops, u64 next_rip,
                Xbyak::CodeGenerator& c) {
    if (insn.mnemonic != ZYDIS_MNEMONIC_VMOVSD) return false;

    // Form 1: VMOVSD xmm, m64 — load from memory into xmm.
    if (insn.operand_count_visible == 2 &&
        ops[0].type == ZYDIS_OPERAND_TYPE_REGISTER &&
        ops[1].type == ZYDIS_OPERAND_TYPE_MEMORY) {
        const int dst_vec = ZydisVecToIndex(ops[0].reg.value);
        if (dst_vec < 0) return false;
        if (!EmitEffectiveAddress(ops[1].mem, next_rip, c))
            return false;
        // rdx = effective address. Single qword load → chunk 0;
        // chunks 1/2/3 zeroed (VEX-128 architectural).
        c.mov(rax, qword[rdx]);
        c.mov(qword[r13 + YmmChunkOffset(dst_vec, 0)], rax);
        c.xor_(rax, rax);
        c.mov(qword[r13 + YmmChunkOffset(dst_vec, 1)], rax);
        c.mov(qword[r13 + YmmChunkOffset(dst_vec, 2)], rax);
        c.mov(qword[r13 + YmmChunkOffset(dst_vec, 3)], rax);
        return true;
    }

    // Form 2: VMOVSD m64, xmm — store xmm low 64 to memory.
    if (insn.operand_count_visible == 2 &&
        ops[0].type == ZYDIS_OPERAND_TYPE_MEMORY &&
        ops[1].type == ZYDIS_OPERAND_TYPE_REGISTER) {
        const int src_vec = ZydisVecToIndex(ops[1].reg.value);
        if (src_vec < 0) return false;
        if (!EmitEffectiveAddress(ops[0].mem, next_rip, c))
            return false;
        // rdx = address. Load src.low64 into rcx first (doesn't touch
        // rdx), then store. Memory above [rdx+8] is untouched.
        c.mov(rcx, qword[r13 + YmmChunkOffset(src_vec, 0)]);
        c.mov(qword[rdx], rcx);
        return true;
    }

    // Form 3: VMOVSD xmm1, xmm2, xmm3 — reg-reg-reg merge.
    if (insn.operand_count_visible == 3 &&
        ops[0].type == ZYDIS_OPERAND_TYPE_REGISTER &&
        ops[1].type == ZYDIS_OPERAND_TYPE_REGISTER &&
        ops[2].type == ZYDIS_OPERAND_TYPE_REGISTER) {
        const int dst_vec = ZydisVecToIndex(ops[0].reg.value);
        const int src1_vec = ZydisVecToIndex(ops[1].reg.value);
        const int src2_vec = ZydisVecToIndex(ops[2].reg.value);
        if (dst_vec < 0 || src1_vec < 0 || src2_vec < 0) return false;

        // chunk 0 = src2.chunk0 (the low 64 of src2 IS chunk 0 entirely).
        c.mov(rax, qword[r13 + YmmChunkOffset(src2_vec, 0)]);
        c.mov(qword[r13 + YmmChunkOffset(dst_vec, 0)], rax);
        // chunk 1 = src1.chunk1 (preserved from VEX.vvvv source).
        c.mov(rax, qword[r13 + YmmChunkOffset(src1_vec, 1)]);
        c.mov(qword[r13 + YmmChunkOffset(dst_vec, 1)], rax);
        // VEX-128: zero upper YMM.
        c.xor_(rax, rax);
        c.mov(qword[r13 + YmmChunkOffset(dst_vec, 2)], rax);
        c.mov(qword[r13 + YmmChunkOffset(dst_vec, 3)], rax);
        return true;
    }

    return false;
}

/// VCVTSI2SS — convert scalar 32-bit or 64-bit integer to scalar
/// single-precision float.
///
///   VCVTSI2SS xmm1, xmm2, r/m32  — dst.low32 = (float)src32
///   VCVTSI2SS xmm1, xmm2, r/m64  — dst.low32 = (float)src64  (REX.W)
///
///   dst[127:32] = src1[127:32]      ; preserved from the second operand
///   ymm.bits[255:128] = 0           ; VEX-128 architectural
///
/// First emitter that requires actual host FP arithmetic — not just
/// bit-shuffling. The conversion respects IEEE-754 rounding, which
/// we delegate to host VCVTSI2SS. Currently we run with whatever
/// MXCSR the host has (round-to-nearest by default); games that
/// depend on a non-default guest MXCSR will need MXCSR sync, which
/// is a separate work item — there's a guest_state.mxcsr slot
/// reserved for that purpose.
///
/// Implementation: load the integer source into a host GPR scratch,
/// run host `vcvtsi2ss xmm0, xmm0, eax/rax`, read the resulting
/// 32-bit float bit pattern out via vmovd, then compose the
/// destination YMM lane from {converted_float, src1[127:32]} +
/// src1[127:64] + zero upper.
bool EmitVcvtsi2ss(const ZydisDecodedInstruction& insn, const ZydisDecodedOperand* ops,
                   u64 next_rip, Xbyak::CodeGenerator& c) {
    if (insn.mnemonic != ZYDIS_MNEMONIC_VCVTSI2SS) return false;

    // Zydis presents this as 3 visible operands: dst, src1 (= the
    // VEX.vvvv "second" operand), src2 (= r/m, the integer). For the
    // 2-operand-syntax legacy form Zydis still synthesizes 3 visible
    // ops with src1 == dst (xbyak does the same on the assembly
    // side). So we always read 3 operands.
    if (insn.operand_count_visible != 3) return false;
    if (ops[0].type != ZYDIS_OPERAND_TYPE_REGISTER) return false;
    if (ops[1].type != ZYDIS_OPERAND_TYPE_REGISTER) return false;

    const int dst_vec = ZydisVecToIndex(ops[0].reg.value);
    const int src1_vec = ZydisVecToIndex(ops[1].reg.value);
    if (dst_vec < 0 || src1_vec < 0) return false;

    // Width here is the INTEGER source width (32 or 64), as Zydis
    // reports it. The destination is always a single-precision
    // float (32 bits).
    const u32 src_width = insn.operand_width;
    if (src_width != 32 && src_width != 64) return false;

    // Load the integer source into a host GPR scratch.
    // Use rax/eax — vcvtsi2ss reads only as many bits as the host
    // operand width specifies.
    if (ops[2].type == ZYDIS_OPERAND_TYPE_REGISTER) {
        const int src_int_idx = ZydisGprToIndex(ops[2].reg.value);
        if (src_int_idx < 0) return false;
        if (src_width == 32) {
            c.mov(eax, dword[r13 + GprOffset(src_int_idx)]);
        } else {
            c.mov(rax, qword[r13 + GprOffset(src_int_idx)]);
        }
    } else if (ops[2].type == ZYDIS_OPERAND_TYPE_MEMORY) {
        if (!EmitEffectiveAddress(ops[2].mem, next_rip, c))
            return false;
        // rdx = address. Load into eax/rax.
        if (src_width == 32) {
            c.mov(eax, dword[rdx]);
        } else {
            c.mov(rax, qword[rdx]);
        }
    } else {
        return false;
    }

    // Run host VCVTSI2SS. xmm0 is JIT-scratch inside a block. The
    // 3-operand form vcvtsi2ss(x1, x2, op) preserves x2's upper 96
    // bits into x1, but since we'll overwrite those from the guest's
    // src1 lane anyway, we don't care about that — passing xmm0 as
    // the merge source is fine.
    if (src_width == 32) {
        c.vcvtsi2ss(xmm0, xmm0, eax);
    } else {
        c.vcvtsi2ss(xmm0, xmm0, rax);
    }

    // Read the converted 32-bit float bit pattern back into r10d.
    // vmovd zero-extends bits 63:32 of r10 by virtue of being a
    // 32-bit move; we'll OR-merge with the high 32 of guest src1
    // below.
    c.vmovd(r10d, xmm0);

    // Compose the destination YMM lane chunk-by-chunk.
    //
    // chunk 0: low32 = converted float (r10d),
    //          hi32  = src1[63:32] (preserved).
    c.mov(eax, dword[r13 + YmmChunkOffset(src1_vec, 0) + 4]); // eax = src1[63:32]; rax bits 63:32 zeroed
    c.shl(rax, 32);
    c.or_(rax, r10);
    c.mov(qword[r13 + YmmChunkOffset(dst_vec, 0)], rax);

    // chunk 1: full copy of src1.chunk1 (preserves dst[127:64]).
    c.mov(rax, qword[r13 + YmmChunkOffset(src1_vec, 1)]);
    c.mov(qword[r13 + YmmChunkOffset(dst_vec, 1)], rax);

    // VEX-128: zero bits 255:128 of the destination YMM.
    c.xor_(rax, rax);
    c.mov(qword[r13 + YmmChunkOffset(dst_vec, 2)], rax);
    c.mov(qword[r13 + YmmChunkOffset(dst_vec, 3)], rax);
    return true;
}

/// VCVTSI2SD — convert scalar 32-bit or 64-bit integer to scalar
/// double-precision float.
///
///   VCVTSI2SD xmm1, xmm2, r/m32  — dst.low64 = (double)src32
///   VCVTSI2SD xmm1, xmm2, r/m64  — dst.low64 = (double)src64  (REX.W)
///
///   dst[127:64] = src1[127:64]      ; preserved from the second operand
///   ymm.bits[255:128] = 0           ; VEX-128 architectural
///
/// The double-precision sibling of VCVTSI2SS, and the gap from CUSA02394
/// at guest 0x8000862f4 (32-bit integer source). Same delegate-to-host
/// strategy: load the integer into a GPR scratch, run host VCVTSI2SD on
/// scratch xmm0, read the 64-bit double bit pattern back with vmovq, and
/// compose the destination lane. Simpler than the SS path because the
/// result fills the entire low 64 bits — no half-lane merge; chunk 0 is
/// the converted double outright, chunk 1 copies src1[127:64].
///
/// MXCSR rounding is delegated to the host (round-to-nearest by default);
/// guest MXCSR sync remains a separate work item — though for the common
/// int→double conversion of values that fit exactly, rounding is moot.
bool EmitVcvtsi2sd(const ZydisDecodedInstruction& insn, const ZydisDecodedOperand* ops,
                   u64 next_rip, Xbyak::CodeGenerator& c) {
    if (insn.mnemonic != ZYDIS_MNEMONIC_VCVTSI2SD) return false;

    if (insn.operand_count_visible != 3) return false;
    if (ops[0].type != ZYDIS_OPERAND_TYPE_REGISTER) return false;
    if (ops[1].type != ZYDIS_OPERAND_TYPE_REGISTER) return false;

    const int dst_vec = ZydisVecToIndex(ops[0].reg.value);
    const int src1_vec = ZydisVecToIndex(ops[1].reg.value);
    if (dst_vec < 0 || src1_vec < 0) return false;

    // Width is the INTEGER source width (32 or 64); the destination is
    // always a double (64 bits).
    const u32 src_width = insn.operand_width;
    if (src_width != 32 && src_width != 64) return false;

    // Load the integer source into a host GPR scratch (eax/rax).
    if (ops[2].type == ZYDIS_OPERAND_TYPE_REGISTER) {
        const int src_int_idx = ZydisGprToIndex(ops[2].reg.value);
        if (src_int_idx < 0) return false;
        if (src_width == 32)
            c.mov(eax, dword[r13 + GprOffset(src_int_idx)]);
        else
            c.mov(rax, qword[r13 + GprOffset(src_int_idx)]);
    } else if (ops[2].type == ZYDIS_OPERAND_TYPE_MEMORY) {
        if (!EmitEffectiveAddress(ops[2].mem, next_rip, c))
            return false;
        if (src_width == 32)
            c.mov(eax, dword[rdx]);
        else
            c.mov(rax, qword[rdx]);
    } else {
        return false;
    }

    // Run host VCVTSI2SD on scratch xmm0 (merge source irrelevant — we
    // overwrite the high half from guest src1 below).
    if (src_width == 32)
        c.vcvtsi2sd(xmm0, xmm0, eax);
    else
        c.vcvtsi2sd(xmm0, xmm0, rax);

    // chunk 0: the full 64-bit converted double.
    c.vmovq(r10, xmm0);
    c.mov(qword[r13 + YmmChunkOffset(dst_vec, 0)], r10);

    // chunk 1: full copy of src1.chunk1 (preserves dst[127:64]).
    c.mov(rax, qword[r13 + YmmChunkOffset(src1_vec, 1)]);
    c.mov(qword[r13 + YmmChunkOffset(dst_vec, 1)], rax);

    // VEX-128: zero bits 255:128 of the destination YMM.
    c.xor_(rax, rax);
    c.mov(qword[r13 + YmmChunkOffset(dst_vec, 2)], rax);
    c.mov(qword[r13 + YmmChunkOffset(dst_vec, 3)], rax);
    return true;
}
/// binary ops. All four share an identical shape:
///
///   VMULSS xmm1, xmm2, xmm3/m32
///     dst.low32     = src1.low32 OP src2.low32   ; IEEE-754 round-to-MXCSR
///     dst[127:32]   = src1[127:32]                ; preserved (VEX merge)
///     ymm.bits[255:128] = 0                       ; VEX-128 architectural
///
/// All use real host FP arithmetic; the architectural "preserve
/// src1[127:32] into dst" rule is satisfied for free because host
/// scalar-FP ops on xmm0 leave xmm0[127:32] untouched while modifying
/// only the low 32. Loading src1's full 128-bit xmm into host xmm0
/// (instead of just the low 32) puts the right bits into the upper
/// half before the operation, then a single 128-bit vmovdqu storeback
/// captures the merge.
///
/// MXCSR: we use whatever the host has (round-to-nearest-even by
/// default). Guest MXCSR sync is a TODO across all FP emitters.
///
/// Operand-count: Zydis presents 3 visible operands (dst, src1, src2)
/// even for 2-operand-syntax assembly. src1 must be a register; src2
/// can be register or m32 (for single-precision Ss) / m64 (for
/// double-precision Sd).
///
/// The same `EmitScalarFp` body covers both precisions. The merge
/// boundary is different (bit 32 for Ss, bit 64 for Sd), but host
/// hardware handles that transparently: in both cases the host op
/// writes only its result-width to xmm0 and leaves the bits above
/// alone — and xmm0 was pre-loaded with src1's full 128 — so the
/// architectural merge falls out for free regardless of precision.
/// The two precisions differ in exactly two emitted lines: the
/// memory load for src2 (vmovss/dword vs vmovsd/qword) and the
/// arithmetic op (vmulss vs vmulsd, etc.).
enum class ScalarFpKind { Mul, Div, Add, Sub, Sqrt, Min, Max };
enum class ScalarFpPrec { Single, Double };

bool EmitScalarFp(const ZydisDecodedInstruction& insn, const ZydisDecodedOperand* ops,
                  u64 next_rip, Xbyak::CodeGenerator& c,
                  ScalarFpKind kind, ScalarFpPrec prec) {
    if (insn.operand_count_visible != 3) return false;
    if (ops[0].type != ZYDIS_OPERAND_TYPE_REGISTER) return false;
    if (ops[1].type != ZYDIS_OPERAND_TYPE_REGISTER) return false;

    const int dst_vec = ZydisVecToIndex(ops[0].reg.value);
    const int src1_vec = ZydisVecToIndex(ops[1].reg.value);
    if (dst_vec < 0 || src1_vec < 0) return false;

    const bool is_double = (prec == ScalarFpPrec::Double);

    // Load src1's full 128-bit xmm into xmm0. The upper bits
    // (127:32 for Ss, 127:64 for Sd) will be preserved into dst by
    // host op's own merge semantics.
    c.vmovdqu(xmm0, ptr[r13 + YmmChunkOffset(src1_vec, 0)]);

    // Load src2 into xmm1. For reg case, full 128; only the low
    // result-width is read by the op. For mem case, load just the
    // m32 / m64 (zero-extends upper, irrelevant since op ignores it).
    if (ops[2].type == ZYDIS_OPERAND_TYPE_REGISTER) {
        const int src2_vec = ZydisVecToIndex(ops[2].reg.value);
        if (src2_vec < 0) return false;
        c.vmovdqu(xmm1, ptr[r13 + YmmChunkOffset(src2_vec, 0)]);
    } else if (ops[2].type == ZYDIS_OPERAND_TYPE_MEMORY) {
        if (!EmitEffectiveAddress(ops[2].mem, next_rip, c))
            return false;
        if (is_double) {
            c.vmovsd(xmm1, qword[rdx]);
        } else {
            c.vmovss(xmm1, dword[rdx]);
        }
    } else {
        return false;
    }

    // Run the host op. For Ss: xmm0.low32 = xmm0.low32 OP xmm1.low32,
    // xmm0[127:32] unchanged. For Sd: xmm0.low64 = xmm0.low64 OP
    // xmm1.low64, xmm0[127:64] unchanged.
    //
    // VSQRTSS / VSQRTSD are conceptually unary (dst.low = sqrt(src2.low))
    // but the host's 3-operand encoding takes (dst, src1, src2) with
    // the same merge semantics as the binops — so they slot in here
    // cleanly. The xmm0 argument that nominally "provides" the upper
    // bits IS our src1-loaded xmm0; src2 is the value being sqrt'd.
    if (is_double) {
        switch (kind) {
        case ScalarFpKind::Mul:  c.vmulsd(xmm0, xmm0, xmm1);  break;
        case ScalarFpKind::Div:  c.vdivsd(xmm0, xmm0, xmm1);  break;
        case ScalarFpKind::Add:  c.vaddsd(xmm0, xmm0, xmm1);  break;
        case ScalarFpKind::Sub:  c.vsubsd(xmm0, xmm0, xmm1);  break;
        case ScalarFpKind::Sqrt: c.vsqrtsd(xmm0, xmm0, xmm1); break;
        case ScalarFpKind::Min:  c.vminsd(xmm0, xmm0, xmm1);  break;
        case ScalarFpKind::Max:  c.vmaxsd(xmm0, xmm0, xmm1);  break;
        }
    } else {
        switch (kind) {
        case ScalarFpKind::Mul:  c.vmulss(xmm0, xmm0, xmm1);  break;
        case ScalarFpKind::Div:  c.vdivss(xmm0, xmm0, xmm1);  break;
        case ScalarFpKind::Add:  c.vaddss(xmm0, xmm0, xmm1);  break;
        case ScalarFpKind::Sub:  c.vsubss(xmm0, xmm0, xmm1);  break;
        case ScalarFpKind::Sqrt: c.vsqrtss(xmm0, xmm0, xmm1); break;
        case ScalarFpKind::Min:  c.vminss(xmm0, xmm0, xmm1);  break;
        case ScalarFpKind::Max:  c.vmaxss(xmm0, xmm0, xmm1);  break;
        }
    }

    // Store the full 128 bits back to dst chunks 0/1.
    c.vmovdqu(ptr[r13 + YmmChunkOffset(dst_vec, 0)], xmm0);

    // VEX-128: zero bits 255:128 of the destination YMM.
    c.xor_(rax, rax);
    c.mov(qword[r13 + YmmChunkOffset(dst_vec, 2)], rax);
    c.mov(qword[r13 + YmmChunkOffset(dst_vec, 3)], rax);
    return true;
}

bool EmitVmulss(const ZydisDecodedInstruction& insn, const ZydisDecodedOperand* ops, u64 next_rip,
                Xbyak::CodeGenerator& c) {
    if (insn.mnemonic != ZYDIS_MNEMONIC_VMULSS) return false;
    return EmitScalarFp(insn, ops, next_rip, c, ScalarFpKind::Mul, ScalarFpPrec::Single);
}

bool EmitVdivss(const ZydisDecodedInstruction& insn, const ZydisDecodedOperand* ops, u64 next_rip,
                Xbyak::CodeGenerator& c) {
    if (insn.mnemonic != ZYDIS_MNEMONIC_VDIVSS) return false;
    return EmitScalarFp(insn, ops, next_rip, c, ScalarFpKind::Div, ScalarFpPrec::Single);
}

bool EmitVaddss(const ZydisDecodedInstruction& insn, const ZydisDecodedOperand* ops, u64 next_rip,
                Xbyak::CodeGenerator& c) {
    if (insn.mnemonic != ZYDIS_MNEMONIC_VADDSS) return false;
    return EmitScalarFp(insn, ops, next_rip, c, ScalarFpKind::Add, ScalarFpPrec::Single);
}

bool EmitVsubss(const ZydisDecodedInstruction& insn, const ZydisDecodedOperand* ops, u64 next_rip,
                Xbyak::CodeGenerator& c) {
    if (insn.mnemonic != ZYDIS_MNEMONIC_VSUBSS) return false;
    return EmitScalarFp(insn, ops, next_rip, c, ScalarFpKind::Sub, ScalarFpPrec::Single);
}

bool EmitVmulsd(const ZydisDecodedInstruction& insn, const ZydisDecodedOperand* ops, u64 next_rip,
                Xbyak::CodeGenerator& c) {
    if (insn.mnemonic != ZYDIS_MNEMONIC_VMULSD) return false;
    return EmitScalarFp(insn, ops, next_rip, c, ScalarFpKind::Mul, ScalarFpPrec::Double);
}

bool EmitVaddsd(const ZydisDecodedInstruction& insn, const ZydisDecodedOperand* ops, u64 next_rip,
                Xbyak::CodeGenerator& c) {
    if (insn.mnemonic != ZYDIS_MNEMONIC_VADDSD) return false;
    return EmitScalarFp(insn, ops, next_rip, c, ScalarFpKind::Add, ScalarFpPrec::Double);
}

bool EmitVsubsd(const ZydisDecodedInstruction& insn, const ZydisDecodedOperand* ops, u64 next_rip,
                Xbyak::CodeGenerator& c) {
    if (insn.mnemonic != ZYDIS_MNEMONIC_VSUBSD) return false;
    return EmitScalarFp(insn, ops, next_rip, c, ScalarFpKind::Sub, ScalarFpPrec::Double);
}

bool EmitVdivsd(const ZydisDecodedInstruction& insn, const ZydisDecodedOperand* ops, u64 next_rip,
                Xbyak::CodeGenerator& c) {
    if (insn.mnemonic != ZYDIS_MNEMONIC_VDIVSD) return false;
    return EmitScalarFp(insn, ops, next_rip, c, ScalarFpKind::Div, ScalarFpPrec::Double);
}

bool EmitVsqrtsd(const ZydisDecodedInstruction& insn, const ZydisDecodedOperand* ops, u64 next_rip,
                 Xbyak::CodeGenerator& c) {
    if (insn.mnemonic != ZYDIS_MNEMONIC_VSQRTSD) return false;
    return EmitScalarFp(insn, ops, next_rip, c, ScalarFpKind::Sqrt, ScalarFpPrec::Double);
}

bool EmitVsqrtss(const ZydisDecodedInstruction& insn, const ZydisDecodedOperand* ops, u64 next_rip,
                 Xbyak::CodeGenerator& c) {
    if (insn.mnemonic != ZYDIS_MNEMONIC_VSQRTSS) return false;
    return EmitScalarFp(insn, ops, next_rip, c, ScalarFpKind::Sqrt, ScalarFpPrec::Single);
}

/// VMINSS / VMAXSS / VMINSD / VMAXSD — scalar single/double min and max.
///
///   VMINSS xmm1, xmm2, xmm3/m32   ; lane0 = min(xmm2.lane0, xmm3.lane0)
///
/// These reuse the EmitScalarFp body verbatim: src1 supplies the upper lanes
/// of the dst (host op merges), src2 (reg or m32/m64) supplies the other
/// operand, VEX zeroes bits 255:128. The host VMINSS/VMAXSS reproduce x86's
/// exact (non-commutative) NaN and signed-zero rule — if either operand is a
/// NaN, or the values are equal, the SECOND source operand is returned — so
/// emitting the host instruction directly is faithful with no extra handling.
///
/// First observed as a reg,reg VMINSS (length 6) in the CUSA02394 "WE ARE
/// DOOMED" eboot at guest 0x8001f3433, in the same vector block as the
/// VPMOVZXDQ/VPEXTRQ fixed earlier.
bool EmitVminss(const ZydisDecodedInstruction& insn, const ZydisDecodedOperand* ops, u64 next_rip,
                Xbyak::CodeGenerator& c) {
    if (insn.mnemonic != ZYDIS_MNEMONIC_VMINSS) return false;
    return EmitScalarFp(insn, ops, next_rip, c, ScalarFpKind::Min, ScalarFpPrec::Single);
}

bool EmitVmaxss(const ZydisDecodedInstruction& insn, const ZydisDecodedOperand* ops, u64 next_rip,
                Xbyak::CodeGenerator& c) {
    if (insn.mnemonic != ZYDIS_MNEMONIC_VMAXSS) return false;
    return EmitScalarFp(insn, ops, next_rip, c, ScalarFpKind::Max, ScalarFpPrec::Single);
}

bool EmitVminsd(const ZydisDecodedInstruction& insn, const ZydisDecodedOperand* ops, u64 next_rip,
                Xbyak::CodeGenerator& c) {
    if (insn.mnemonic != ZYDIS_MNEMONIC_VMINSD) return false;
    return EmitScalarFp(insn, ops, next_rip, c, ScalarFpKind::Min, ScalarFpPrec::Double);
}

bool EmitVmaxsd(const ZydisDecodedInstruction& insn, const ZydisDecodedOperand* ops, u64 next_rip,
                Xbyak::CodeGenerator& c) {
    if (insn.mnemonic != ZYDIS_MNEMONIC_VMAXSD) return false;
    return EmitScalarFp(insn, ops, next_rip, c, ScalarFpKind::Max, ScalarFpPrec::Double);
}

/// VBLENDPS / VPBLENDW — packed blend controlled by an imm8 mask.
/// Both are 4-operand (dst, src1, src2/mem, imm8) immediate-mask
/// blends; they differ only in element granularity and the host
/// mnemonic. Parameterised here at the second instance (VPBLENDW).
///
///   VBLENDPS xmm/ymm, src1, src2/m, imm8   ; 32-bit float-word lanes
///   VPBLENDW xmm/ymm, src1, src2/m, imm8   ; 16-bit integer-word lanes
///
/// Per lane i (of the element width for the kind):
///   imm8 bit for lane i == 0 → dst[i] = src1[i]
///   imm8 bit for lane i == 1 → dst[i] = src2[i]
///
/// imm8 BIT-SEMANTICS differ between the two, and this matters for
/// the 256-bit forms — but in both cases the HOST op interprets imm8,
/// so we pass the raw byte straight through and never decode it
/// ourselves:
///   * VBLENDPS: 1 bit per 32-bit element. 128-bit uses imm8[3:0]
///     (high nibble reserved); 256-bit uses all 8 bits (one per
///     element across both lanes).
///   * VPBLENDW: 1 bit per 16-bit word, covering the 8 words of ONE
///     128-bit lane (imm8[7:0]). For the 256-bit form the SAME imm8
///     pattern is replicated to the upper 128-bit lane — imm8 does
///     NOT widen to 16 bits. The host vpblendw ymm does this
///     replication internally, so passing imm8 unchanged is correct.
///
/// Implementation: the imm8 control logic is non-trivial to lift
/// directly (per-element conditional moves keyed on constant imm8
/// bits). Much simpler to run the host blend — the imm8 is in the
/// guest instruction stream so we have it at lift time. 128-bit form
/// additionally zeros bits 255:128 of the dst YMM (VEX architectural).
///
/// Compilers emit these for branchless selection — picking between
/// two computed alternatives on a fixed compile-time-proven pattern.
enum class VecBlendImm { Ps, W };

bool EmitVecBlendImm(const ZydisDecodedInstruction& insn, const ZydisDecodedOperand* ops,
                     u64 next_rip, Xbyak::CodeGenerator& c, VecBlendImm kind) {
    // Mnemonic gating — defensive routing check.
    auto expected = [&]() -> ZydisMnemonic {
        switch (kind) {
        case VecBlendImm::Ps: return ZYDIS_MNEMONIC_VBLENDPS;
        case VecBlendImm::W:  return ZYDIS_MNEMONIC_VPBLENDW;
        }
        return ZYDIS_MNEMONIC_INVALID;
    }();
    if (insn.mnemonic != expected) return false;

    if (insn.operand_count_visible != 4) return false;
    if (ops[0].type != ZYDIS_OPERAND_TYPE_REGISTER) return false;
    if (ops[1].type != ZYDIS_OPERAND_TYPE_REGISTER) return false;
    if (ops[3].type != ZYDIS_OPERAND_TYPE_IMMEDIATE) return false;

    const int dst_vec = ZydisVecToIndex(ops[0].reg.value);
    const int src1_vec = ZydisVecToIndex(ops[1].reg.value);
    if (dst_vec < 0 || src1_vec < 0) return false;

    const int vec_bits = ops[0].size;
    if (vec_bits != 128 && vec_bits != 256) return false;

    const u8 imm = static_cast<u8>(ops[3].imm.value.u);

    // Dispatch the host blend op for the kind, at the given width.
    auto emit_blend_128 = [&]() {
        switch (kind) {
        case VecBlendImm::Ps: c.vblendps(xmm0, xmm0, xmm1, imm); break;
        case VecBlendImm::W:  c.vpblendw(xmm0, xmm0, xmm1, imm); break;
        }
    };
    auto emit_blend_256 = [&]() {
        switch (kind) {
        case VecBlendImm::Ps: c.vblendps(ymm0, ymm0, ymm1, imm); break;
        case VecBlendImm::W:  c.vpblendw(ymm0, ymm0, ymm1, imm); break;
        }
    };

    // Branch on vector width — different host registers (xmm vs ymm)
    // and different storeback chunk counts.
    if (vec_bits == 128) {
        c.vmovdqu(xmm0, ptr[r13 + YmmChunkOffset(src1_vec, 0)]);

        if (ops[2].type == ZYDIS_OPERAND_TYPE_REGISTER) {
            const int src2_vec = ZydisVecToIndex(ops[2].reg.value);
            if (src2_vec < 0) return false;
            c.vmovdqu(xmm1, ptr[r13 + YmmChunkOffset(src2_vec, 0)]);
        } else if (ops[2].type == ZYDIS_OPERAND_TYPE_MEMORY) {
            if (!EmitEffectiveAddress(ops[2].mem, next_rip, c))
                return false;
            c.vmovdqu(xmm1, ptr[rdx]);
        } else {
            return false;
        }

        emit_blend_128();

        c.vmovdqu(ptr[r13 + YmmChunkOffset(dst_vec, 0)], xmm0);
        c.xor_(rax, rax);
        c.mov(qword[r13 + YmmChunkOffset(dst_vec, 2)], rax);
        c.mov(qword[r13 + YmmChunkOffset(dst_vec, 3)], rax);
    } else { // 256-bit
        c.vmovdqu(ymm0, ptr[r13 + YmmChunkOffset(src1_vec, 0)]);

        if (ops[2].type == ZYDIS_OPERAND_TYPE_REGISTER) {
            const int src2_vec = ZydisVecToIndex(ops[2].reg.value);
            if (src2_vec < 0) return false;
            c.vmovdqu(ymm1, ptr[r13 + YmmChunkOffset(src2_vec, 0)]);
        } else if (ops[2].type == ZYDIS_OPERAND_TYPE_MEMORY) {
            if (!EmitEffectiveAddress(ops[2].mem, next_rip, c))
                return false;
            c.vmovdqu(ymm1, ptr[rdx]);
        } else {
            return false;
        }

        emit_blend_256();
        c.vmovdqu(ptr[r13 + YmmChunkOffset(dst_vec, 0)], ymm0);
    }
    return true;
}

/// VBLENDVPS — variable-mask packed single-precision blend. The
/// runtime-register counterpart of VBLENDPS (constant imm8 mask).
///
///   VBLENDVPS xmm1, xmm2, xmm3/m128, xmm4   ; 4 elements (128-bit)
///   VBLENDVPS ymm1, ymm2, ymm3/m256, ymm4   ; 8 elements (256-bit)
///
/// For each 32-bit element i:
///   if mask[i].sign_bit (bit 31) == 1 → dst[i] = src2[i]
///   if mask[i].sign_bit       == 0 → dst[i] = src1[i]
///
/// The 4th operand (mask register) is encoded in the IS4-immediate
/// byte's upper nibble — a VEX-specific encoding hack. Zydis presents
/// it as ops[3] with REGISTER type.
///
/// This is shape-distinct from VBLENDPS — the selector is data, not
/// a lift-time constant, so we can't fold it into the host op's imm
/// field. Instead we load the mask into scratch xmm2 and let the
/// host's variable-blend hardware do the per-element selection.
///
/// Compilers emit VBLENDVPS for branchless selection where the
/// predicate is a runtime mask — e.g., `if (cond) a else b` lifted
/// across 4 lanes where `cond` is itself a per-lane comparison
/// result. The common pattern is:
///   vcmpltps mask, x, threshold   ; mask[i] = (x[i] < threshold) ? all-1s : 0
///   vblendvps result, b, a, mask  ; result[i] = (mask[i]) ? a[i] : b[i]
bool EmitVblendvps(const ZydisDecodedInstruction& insn, const ZydisDecodedOperand* ops,
                   u64 next_rip, Xbyak::CodeGenerator& c) {
    if (insn.mnemonic != ZYDIS_MNEMONIC_VBLENDVPS) return false;
    if (insn.operand_count_visible != 4) return false;
    if (ops[0].type != ZYDIS_OPERAND_TYPE_REGISTER) return false;
    if (ops[1].type != ZYDIS_OPERAND_TYPE_REGISTER) return false;
    if (ops[3].type != ZYDIS_OPERAND_TYPE_REGISTER) return false;

    const int dst_vec = ZydisVecToIndex(ops[0].reg.value);
    const int src1_vec = ZydisVecToIndex(ops[1].reg.value);
    const int mask_vec = ZydisVecToIndex(ops[3].reg.value);
    if (dst_vec < 0 || src1_vec < 0 || mask_vec < 0) return false;

    const int vec_bits = ops[0].size;
    if (vec_bits != 128 && vec_bits != 256) return false;

    if (vec_bits == 128) {
        c.vmovdqu(xmm0, ptr[r13 + YmmChunkOffset(src1_vec, 0)]);

        if (ops[2].type == ZYDIS_OPERAND_TYPE_REGISTER) {
            const int src2_vec = ZydisVecToIndex(ops[2].reg.value);
            if (src2_vec < 0) return false;
            c.vmovdqu(xmm1, ptr[r13 + YmmChunkOffset(src2_vec, 0)]);
        } else if (ops[2].type == ZYDIS_OPERAND_TYPE_MEMORY) {
            if (!EmitEffectiveAddress(ops[2].mem, next_rip, c))
                return false;
            c.vmovdqu(xmm1, ptr[rdx]);
        } else {
            return false;
        }

        c.vmovdqu(xmm2, ptr[r13 + YmmChunkOffset(mask_vec, 0)]);
        c.vblendvps(xmm0, xmm0, xmm1, xmm2);
        c.vmovdqu(ptr[r13 + YmmChunkOffset(dst_vec, 0)], xmm0);

        c.xor_(rax, rax);
        c.mov(qword[r13 + YmmChunkOffset(dst_vec, 2)], rax);
        c.mov(qword[r13 + YmmChunkOffset(dst_vec, 3)], rax);
    } else { // 256-bit
        c.vmovdqu(ymm0, ptr[r13 + YmmChunkOffset(src1_vec, 0)]);

        if (ops[2].type == ZYDIS_OPERAND_TYPE_REGISTER) {
            const int src2_vec = ZydisVecToIndex(ops[2].reg.value);
            if (src2_vec < 0) return false;
            c.vmovdqu(ymm1, ptr[r13 + YmmChunkOffset(src2_vec, 0)]);
        } else if (ops[2].type == ZYDIS_OPERAND_TYPE_MEMORY) {
            if (!EmitEffectiveAddress(ops[2].mem, next_rip, c))
                return false;
            c.vmovdqu(ymm1, ptr[rdx]);
        } else {
            return false;
        }

        c.vmovdqu(ymm2, ptr[r13 + YmmChunkOffset(mask_vec, 0)]);
        c.vblendvps(ymm0, ymm0, ymm1, ymm2);
        c.vmovdqu(ptr[r13 + YmmChunkOffset(dst_vec, 0)], ymm0);
    }
    return true;
}

/// VINSERTF128 — insert a 128-bit packed FP value into one half of a
/// 256-bit ymm. The other half comes from src1.
///
///   VINSERTF128 ymm1, ymm2, xmm3/m128, imm8
///
/// imm8[0] selects the insertion lane:
///   imm8[0] == 0 → dst.lo128 = src2,            dst.hi128 = src1.hi128
///   imm8[0] == 1 → dst.lo128 = src1.lo128,      dst.hi128 = src2
/// imm8 bits [7:1] are reserved (ignored).
///
/// Compilers emit this for 256-bit-wide computations that combine
/// two 128-bit halves, especially when the lanes come from different
/// sources (e.g. one xmm holds the "real" parts of complex values,
/// another holds the "imaginary" — VINSERTF128 stitches them into a
/// 256-bit YMM for parallel processing).
///
/// Implementation: host VINSERTF128 directly. xbyak's signature takes
/// (Ymm dst, Ymm src1, Xmm/Mem src2, imm8) which matches the guest
/// shape exactly. Load src1 into ymm0, load src2 into xmm1 (or via
/// mem operand), run host op, store all 4 chunks back. No upper-zero
/// needed — this is a full 256-bit operation.
bool EmitVinsertf128(const ZydisDecodedInstruction& insn, const ZydisDecodedOperand* ops,
                     u64 next_rip, Xbyak::CodeGenerator& c) {
    if (insn.mnemonic != ZYDIS_MNEMONIC_VINSERTF128) return false;
    if (insn.operand_count_visible != 4) return false;
    if (ops[0].type != ZYDIS_OPERAND_TYPE_REGISTER) return false;
    if (ops[1].type != ZYDIS_OPERAND_TYPE_REGISTER) return false;
    if (ops[3].type != ZYDIS_OPERAND_TYPE_IMMEDIATE) return false;

    const int dst_vec = ZydisVecToIndex(ops[0].reg.value);
    const int src1_vec = ZydisVecToIndex(ops[1].reg.value);
    if (dst_vec < 0 || src1_vec < 0) return false;

    // dst and src1 must be ymm (256-bit); src2 must be xmm or m128.
    if (ops[0].size != 256 || ops[1].size != 256) return false;

    const u8 imm = static_cast<u8>(ops[3].imm.value.u);

    // Load src1 full ymm into host ymm0 (provides the non-replaced half).
    c.vmovdqu(ymm0, ptr[r13 + YmmChunkOffset(src1_vec, 0)]);

    // Load src2 (128-bit) into host xmm1.
    if (ops[2].type == ZYDIS_OPERAND_TYPE_REGISTER) {
        const int src2_vec = ZydisVecToIndex(ops[2].reg.value);
        if (src2_vec < 0) return false;
        c.vmovdqu(xmm1, ptr[r13 + YmmChunkOffset(src2_vec, 0)]);
    } else if (ops[2].type == ZYDIS_OPERAND_TYPE_MEMORY) {
        if (!EmitEffectiveAddress(ops[2].mem, next_rip, c))
            return false;
        c.vmovdqu(xmm1, ptr[rdx]);
    } else {
        return false;
    }

    // Host op handles the lane-selection logic via imm8.
    c.vinsertf128(ymm0, ymm0, xmm1, imm);

    // Storeback all 4 chunks of the 256-bit result. No upper-zero
    // needed — this IS a 256-bit op, so all of ymm is meaningful.
    c.vmovdqu(ptr[r13 + YmmChunkOffset(dst_vec, 0)], ymm0);
    return true;
}

/// VEXTRACTF128 — extract a 128-bit half from a 256-bit ymm. The
/// mirror image of VINSERTF128.
///
///   VEXTRACTF128 xmm1/m128, ymm2, imm8
///
/// imm8[0] selects which half is extracted:
///   imm8[0] == 0 → dst = src.lo128
///   imm8[0] == 1 → dst = src.hi128
/// imm8 bits [7:1] are reserved (ignored).
///
/// Reg destination: dst's upper 128 bits (the YMM-counterpart) are
/// zeroed (VEX-128 architectural). Mem destination: only 16 bytes
/// written; surrounding memory untouched.
///
/// Operand shape differs from VINSERTF128 — 3 visible ops (dst,
/// src, imm) instead of 4 — and dst can be memory, so we don't share
/// a helper. The sibling relationship (extract vs insert, mem-or-reg
/// dst vs mem-or-reg src) is real but the code paths are short
/// enough that two emitters are clearer than one parameterized one.
///
/// Implementation: host VEXTRACTF128 directly. Load src ymm into
/// ymm0, run host extract into xmm1, then either store 128 bits
/// to a mem dst OR write xmm1 to the reg dst's chunk 0 and zero
/// chunks 1/2/3.
///
/// Common compiler emission: paired with VINSERTF128 for splitting
/// a 256-bit result back into two 128-bit halves for downstream
/// 128-bit-width processing, or for the upper-128 "extract and
/// throw away" pattern that compilers sometimes emit before a
/// dependent xmm-width op.
bool EmitVextractf128(const ZydisDecodedInstruction& insn, const ZydisDecodedOperand* ops,
                      u64 next_rip, Xbyak::CodeGenerator& c) {
    if (insn.mnemonic != ZYDIS_MNEMONIC_VEXTRACTF128) return false;
    if (insn.operand_count_visible != 3) return false;
    if (ops[1].type != ZYDIS_OPERAND_TYPE_REGISTER) return false;
    if (ops[2].type != ZYDIS_OPERAND_TYPE_IMMEDIATE) return false;

    const int src_vec = ZydisVecToIndex(ops[1].reg.value);
    if (src_vec < 0) return false;
    // Note: Zydis reports both operand sizes as 128 here (the data-
    // flow width, not the YMM-register physical width). The source
    // register IS a ymm in the encoding — ZydisVecToIndex picks up
    // the lane index correctly from the register enum either way.

    const u8 imm = static_cast<u8>(ops[2].imm.value.u);

    // Load source ymm (the full 256 bits) into host ymm0.
    c.vmovdqu(ymm0, ptr[r13 + YmmChunkOffset(src_vec, 0)]);

    if (ops[0].type == ZYDIS_OPERAND_TYPE_REGISTER) {
        const int dst_vec = ZydisVecToIndex(ops[0].reg.value);
        if (dst_vec < 0) return false;

        c.vextractf128(xmm1, ymm0, imm);
        c.vmovdqu(ptr[r13 + YmmChunkOffset(dst_vec, 0)], xmm1);
        // VEX-128 zero of dst's upper YMM half.
        c.xor_(rax, rax);
        c.mov(qword[r13 + YmmChunkOffset(dst_vec, 2)], rax);
        c.mov(qword[r13 + YmmChunkOffset(dst_vec, 3)], rax);
    } else if (ops[0].type == ZYDIS_OPERAND_TYPE_MEMORY) {
        if (!EmitEffectiveAddress(ops[0].mem, next_rip, c))
            return false;
        c.vextractf128(ptr[rdx], ymm0, imm);
    } else {
        return false;
    }
    return true;
}

/// VUCOMISS — Unordered Compare Scalar Single-precision and Set EFLAGS.
///
///   VUCOMISS xmm1, xmm2/m32
///     Compares src1.low32 (float) to src2.low32 (float).
///     Writes ZF/PF/CF based on the comparison; clears OF/SF/AF.
///
///   Truth table (per Intel SDM):
///     Unordered  (either is NaN)   → ZF=1 PF=1 CF=1
///     src1 == src2                 → ZF=1 PF=0 CF=0
///     src1  < src2                 → ZF=0 PF=0 CF=1
///     src1  > src2                 → ZF=0 PF=0 CF=0
///
/// The "U" prefix means *unordered* — quiet NaNs go silently into the
/// unordered branch and do NOT raise an exception. (VCOMISS, without
/// the U, raises on signaling NaN; for our purposes both forms are
/// equivalent since we let the host CPU manage MXCSR exception masks.)
///
/// First scalar-FP flag-writing emitter. The mechanism is the same
/// as VPTEST: load src1 and src2 into JIT-scratch xmm0/xmm1, run
/// host VUCOMISS (which sets the host EFLAGS as the architecture
/// requires), then capture pushed-host-rflags and merge into the
/// guest rflags slot using the standard arithmetic mask
/// (CF|PF|AF|ZF|SF|OF = 0x8D5).
bool EmitVucomiss(const ZydisDecodedInstruction& insn, const ZydisDecodedOperand* ops,
                  u64 next_rip, Xbyak::CodeGenerator& c) {
    if (insn.mnemonic != ZYDIS_MNEMONIC_VUCOMISS) return false;
    if (insn.operand_count_visible < 2) return false;
    if (ops[0].type != ZYDIS_OPERAND_TYPE_REGISTER) return false;

    const int src1_vec = ZydisVecToIndex(ops[0].reg.value);
    if (src1_vec < 0) return false;

    // Load src1's full xmm into host xmm0. Only low 32 is read by
    // VUCOMISS but loading 128 is the same cost as 32.
    c.vmovdqu(xmm0, ptr[r13 + YmmChunkOffset(src1_vec, 0)]);

    // Load src2 — either an xmm lane or m32.
    if (ops[1].type == ZYDIS_OPERAND_TYPE_REGISTER) {
        const int src2_vec = ZydisVecToIndex(ops[1].reg.value);
        if (src2_vec < 0) return false;
        c.vmovdqu(xmm1, ptr[r13 + YmmChunkOffset(src2_vec, 0)]);
    } else if (ops[1].type == ZYDIS_OPERAND_TYPE_MEMORY) {
        if (!EmitEffectiveAddress(ops[1].mem, next_rip, c))
            return false;
        c.vmovss(xmm1, dword[rdx]);
    } else {
        return false;
    }

    // Run host VUCOMISS. This sets host EFLAGS.ZF/PF/CF per the
    // comparison and clears OF/SF/AF.
    c.vucomiss(xmm0, xmm1);

    // Merge host arithmetic flags into the guest rflags slot. Same
    // pattern as VPTEST/VPCMPISTRI.
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

/// VUCOMISD — Unordered Compare Scalar Double-precision and Set EFLAGS.
///
///   VUCOMISD xmm1, xmm2/m64
///     Compares src1.low64 (double) to src2.low64 (double).
///     Writes ZF/PF/CF by the same truth table as VUCOMISS; clears OF/SF/AF.
///
/// The double-precision sibling of VUCOMISS, and the gap from CUSA02394 at
/// guest 0x800122e60. Identical mechanism — load both operands into scratch
/// xmm0/xmm1, run host VUCOMISD to set host EFLAGS, then merge the host
/// arithmetic flags into the guest rflags slot with the standard 0x8D5 mask.
/// The only differences from the SS path are the host op (vucomisd) and the
/// memory operand width (m64 via vmovsd instead of m32 via vmovss).
bool EmitVucomisd(const ZydisDecodedInstruction& insn, const ZydisDecodedOperand* ops,
                  u64 next_rip, Xbyak::CodeGenerator& c) {
    if (insn.mnemonic != ZYDIS_MNEMONIC_VUCOMISD) return false;
    if (insn.operand_count_visible < 2) return false;
    if (ops[0].type != ZYDIS_OPERAND_TYPE_REGISTER) return false;

    const int src1_vec = ZydisVecToIndex(ops[0].reg.value);
    if (src1_vec < 0) return false;

    // Load src1's full xmm into host xmm0. Only low 64 is read by VUCOMISD.
    c.vmovdqu(xmm0, ptr[r13 + YmmChunkOffset(src1_vec, 0)]);

    // Load src2 — either an xmm lane or m64.
    if (ops[1].type == ZYDIS_OPERAND_TYPE_REGISTER) {
        const int src2_vec = ZydisVecToIndex(ops[1].reg.value);
        if (src2_vec < 0) return false;
        c.vmovdqu(xmm1, ptr[r13 + YmmChunkOffset(src2_vec, 0)]);
    } else if (ops[1].type == ZYDIS_OPERAND_TYPE_MEMORY) {
        if (!EmitEffectiveAddress(ops[1].mem, next_rip, c))
            return false;
        c.vmovsd(xmm1, qword[rdx]);
    } else {
        return false;
    }

    // Run host VUCOMISD. This sets host EFLAGS.ZF/PF/CF per the comparison
    // and clears OF/SF/AF.
    c.vucomisd(xmm0, xmm1);

    // Merge host arithmetic flags into the guest rflags slot.
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
///
///   VCVTSS2SD xmm1, xmm2, xmm3/m32
///     dst.low64       = (double)src2.low32       ; IEEE-754 widening
///     dst[127:64]     = src1[127:64]             ; preserved (VEX merge)
///     ymm.bits[255:128] = 0                      ; VEX-128 architectural
///
/// First scalar conversion where the output has a *different* width
/// than the input. The VEX merge boundary is at bit 64 (not bit 32 as
/// in the SS-binop family), because dst.low64 is the double result and
/// the architectural preserved-from-src1 region is everything above
/// that.
///
/// Despite the different merge boundary, the "load src1 full xmm into
/// xmm0 → run host op → 128-bit storeback" pattern works unchanged:
/// host VCVTSS2SD writes only xmm0.low64, leaving xmm0[127:64] alone
/// — and those upper 64 bits were loaded from src1[127:64], so the
/// architectural merge is satisfied for free. Same trick as VMULSS
/// at a different bit boundary.
///
/// The widening 32-bit-float → 64-bit-double is exact: every float
/// has a corresponding double with the same value, no rounding occurs.
/// (The inverse VCVTSD2SS does round.) MXCSR rounding-mode field is
/// therefore irrelevant for this direction.
bool EmitVcvtss2sd(const ZydisDecodedInstruction& insn, const ZydisDecodedOperand* ops,
                   u64 next_rip, Xbyak::CodeGenerator& c) {
    if (insn.mnemonic != ZYDIS_MNEMONIC_VCVTSS2SD) return false;
    if (insn.operand_count_visible != 3) return false;
    if (ops[0].type != ZYDIS_OPERAND_TYPE_REGISTER) return false;
    if (ops[1].type != ZYDIS_OPERAND_TYPE_REGISTER) return false;

    const int dst_vec = ZydisVecToIndex(ops[0].reg.value);
    const int src1_vec = ZydisVecToIndex(ops[1].reg.value);
    if (dst_vec < 0 || src1_vec < 0) return false;

    // Load src1's full 128-bit xmm into host xmm0. The upper 64
    // will be preserved into dst by host VCVTSS2SD's merge semantics.
    c.vmovdqu(xmm0, ptr[r13 + YmmChunkOffset(src1_vec, 0)]);

    // Load src2 (the single-precision input). Reg case: full 128
    // into xmm1, only low 32 read. Mem case: vmovss zero-extends.
    if (ops[2].type == ZYDIS_OPERAND_TYPE_REGISTER) {
        const int src2_vec = ZydisVecToIndex(ops[2].reg.value);
        if (src2_vec < 0) return false;
        c.vmovdqu(xmm1, ptr[r13 + YmmChunkOffset(src2_vec, 0)]);
    } else if (ops[2].type == ZYDIS_OPERAND_TYPE_MEMORY) {
        if (!EmitEffectiveAddress(ops[2].mem, next_rip, c))
            return false;
        c.vmovss(xmm1, dword[rdx]);
    } else {
        return false;
    }

    // Convert. xmm0.low64 = (double)xmm1.low32; xmm0[127:64] unchanged.
    c.vcvtss2sd(xmm0, xmm0, xmm1);

    // Store the full 128 bits back. xmm0 now holds
    // {converted_double, src1[127:64]} — exactly the architectural merge.
    c.vmovdqu(ptr[r13 + YmmChunkOffset(dst_vec, 0)], xmm0);

    // VEX-128: zero bits 255:128 of the destination YMM.
    c.xor_(rax, rax);
    c.mov(qword[r13 + YmmChunkOffset(dst_vec, 2)], rax);
    c.mov(qword[r13 + YmmChunkOffset(dst_vec, 3)], rax);
    return true;
}

/// VCVTSD2SS — Convert Scalar Double-precision to Scalar Single-precision.
///
///   VCVTSD2SS xmm1, xmm2, xmm3/m64
///     dst.low32       = (float)src2.low64       ; IEEE-754 narrowing (rounds!)
///     dst[127:32]     = src1[127:32]            ; preserved (VEX merge)
///     ymm.bits[255:128] = 0                     ; VEX-128 architectural
///
/// The inverse of VCVTSS2SD. Mirror image in two ways:
///   1. The source is double (m64 / qword load, vmovsd) instead of
///      single (m32 / dword, vmovss).
///   2. The merge boundary moves back to bit 32 — same as the SS-binop
///      family — because the *output* is single-precision. dst[127:32]
///      is preserved from src1, exactly as it is in VMULSS et al.
///
/// MXCSR matters here, unlike VCVTSS2SD: 64 → 32 bits is a narrowing
/// conversion and the source value may not be representable as a
/// float. The host uses its current MXCSR rounding mode; we run with
/// host default (round-to-nearest-even). Same caveat as the rest of
/// the FP-arithmetic family.
///
/// Implementation pattern is the same "load src1 full xmm → host op
/// → 128-bit storeback" trick: host VCVTSD2SS modifies only
/// xmm0.low32 (writing the converted float), leaving xmm0[127:32]
/// alone — and those are already src1[127:32] from the initial load.
bool EmitVcvtsd2ss(const ZydisDecodedInstruction& insn, const ZydisDecodedOperand* ops,
                   u64 next_rip, Xbyak::CodeGenerator& c) {
    if (insn.mnemonic != ZYDIS_MNEMONIC_VCVTSD2SS) return false;
    if (insn.operand_count_visible != 3) return false;
    if (ops[0].type != ZYDIS_OPERAND_TYPE_REGISTER) return false;
    if (ops[1].type != ZYDIS_OPERAND_TYPE_REGISTER) return false;

    const int dst_vec = ZydisVecToIndex(ops[0].reg.value);
    const int src1_vec = ZydisVecToIndex(ops[1].reg.value);
    if (dst_vec < 0 || src1_vec < 0) return false;

    // Load src1's full 128-bit xmm into host xmm0. The upper 96 will
    // be preserved into dst by host VCVTSD2SS's merge semantics.
    c.vmovdqu(xmm0, ptr[r13 + YmmChunkOffset(src1_vec, 0)]);

    // Load src2 (the double-precision input). Reg case: full 128
    // into xmm1, only low 64 read. Mem case: vmovsd loads the m64
    // (zero-extends bits 127:64 of xmm1, irrelevant since op ignores).
    if (ops[2].type == ZYDIS_OPERAND_TYPE_REGISTER) {
        const int src2_vec = ZydisVecToIndex(ops[2].reg.value);
        if (src2_vec < 0) return false;
        c.vmovdqu(xmm1, ptr[r13 + YmmChunkOffset(src2_vec, 0)]);
    } else if (ops[2].type == ZYDIS_OPERAND_TYPE_MEMORY) {
        if (!EmitEffectiveAddress(ops[2].mem, next_rip, c))
            return false;
        c.vmovsd(xmm1, qword[rdx]);
    } else {
        return false;
    }

    // Convert. xmm0.low32 = (float)xmm1.low64; xmm0[127:32] unchanged.
    c.vcvtsd2ss(xmm0, xmm0, xmm1);

    // Store the full 128 bits back. xmm0 now holds
    // {converted_float, src1[127:32]} — exactly the architectural merge.
    c.vmovdqu(ptr[r13 + YmmChunkOffset(dst_vec, 0)], xmm0);

    // VEX-128: zero bits 255:128 of the destination YMM.
    c.xor_(rax, rax);
    c.mov(qword[r13 + YmmChunkOffset(dst_vec, 2)], rax);
    c.mov(qword[r13 + YmmChunkOffset(dst_vec, 3)], rax);
    return true;
}

/// VCVTTSS2SI — convert scalar single-precision float to signed
/// integer, with truncation toward zero (the extra "T" in the name
/// vs VCVTSS2SI which uses the current MXCSR rounding mode).
///
///   VCVTTSS2SI r32, xmm/m32   — r32 = (int32_t)truncate(src.low32)
///   VCVTTSS2SI r64, xmm/m32   — r64 = (int64_t)truncate(src.low32)
///
/// Overflow / NaN / infinity all produce the "indefinite integer
/// value" — INT32_MIN for the 32-bit form, INT64_MIN for the 64-bit
/// form. Host VCVTTSS2SI implements this correctly; we delegate.
///
/// This is the inverse of VCVTSI2SS. Same delegation strategy:
/// load src into JIT-scratch xmm0, run host VCVTTSS2SI into rax/eax,
/// store back to the guest GPR slot. 32-bit writes zero-extend bits
/// 63:32 of rax so the qword storeback lands a clean upper half.
///
/// MXCSR: the truncating form ignores the MXCSR rounding-mode field
/// entirely (truncation is hard-coded), so even though we run with
/// host MXCSR, this emitter is unaffected by any guest/host MXCSR
/// divergence — unlike VCVTSI2SS which does respect MXCSR rounding.
bool EmitVcvttss2si(const ZydisDecodedInstruction& insn, const ZydisDecodedOperand* ops,
                    u64 next_rip, Xbyak::CodeGenerator& c) {
    if (insn.mnemonic != ZYDIS_MNEMONIC_VCVTTSS2SI) return false;
    if (ops[0].type != ZYDIS_OPERAND_TYPE_REGISTER) return false;

    // operand_width here refers to the DESTINATION integer width
    // (32 or 64) — Zydis reports it based on the result type.
    const u32 dst_width = insn.operand_width;
    if (dst_width != 32 && dst_width != 64) return false;

    const int dst_idx = ZydisGprToIndex(ops[0].reg.value);
    if (dst_idx < 0) return false;

    // Load src.low32 into host xmm0. The reg case loads the full
    // 128-bit lane (cheap, simple); the mem case loads just the
    // 32-bit float via vmovss, zero-extending the upper 96 of xmm0
    // — irrelevant since vcvttss2si only reads low 32.
    if (ops[1].type == ZYDIS_OPERAND_TYPE_REGISTER) {
        const int src_vec = ZydisVecToIndex(ops[1].reg.value);
        if (src_vec < 0) return false;
        c.vmovdqu(xmm0, ptr[r13 + YmmChunkOffset(src_vec, 0)]);
    } else if (ops[1].type == ZYDIS_OPERAND_TYPE_MEMORY) {
        if (!EmitEffectiveAddress(ops[1].mem, next_rip, c))
            return false;
        c.vmovss(xmm0, dword[rdx]);
    } else {
        return false;
    }

    // Run host VCVTTSS2SI. Truncation is enforced by the opcode
    // itself — no MXCSR query.
    if (dst_width == 32) {
        c.vcvttss2si(eax, xmm0);
    } else {
        c.vcvttss2si(rax, xmm0);
    }

    // Store result into guest dst GPR slot. For dst_width == 32, the
    // host eax-write already zero-extended rax bits 63:32, so qword
    // storeback lands a clean upper half. For 64-bit, the full rax
    // is the result.
    c.mov(qword[r13 + GprOffset(dst_idx)], rax);
    return true;
}

/// VCVTTSD2SI — convert scalar double-precision float to signed integer,
/// with truncation toward zero. The double-precision sibling of
/// VCVTTSS2SI, and the gap from CUSA02394 at guest 0x800200e32 (a 64-bit
/// destination, immediately after the VROUNDSD at 0x800200e2c — the
/// double is rounded, then converted to an integer index).
///
///   VCVTTSD2SI r32, xmm/m64   — r32 = (int32_t)truncate(src.low64)
///   VCVTTSD2SI r64, xmm/m64   — r64 = (int64_t)truncate(src.low64)
///
/// Overflow / NaN / infinity all produce the indefinite integer value
/// (INT32_MIN / INT64_MIN). Identical delegation to the SS version:
/// load src into scratch xmm0, run host VCVTTSD2SI into eax/rax, store
/// to the guest GPR slot (the 32-bit form zero-extends rax[63:32] for a
/// clean qword storeback). The only differences from the SS path are
/// the host op and the m64 memory operand width (vmovsd vs vmovss).
/// Truncation is hard-coded in the opcode, so this is unaffected by any
/// guest/host MXCSR rounding-mode divergence.
bool EmitVcvttsd2si(const ZydisDecodedInstruction& insn, const ZydisDecodedOperand* ops,
                    u64 next_rip, Xbyak::CodeGenerator& c) {
    if (insn.mnemonic != ZYDIS_MNEMONIC_VCVTTSD2SI) return false;
    if (ops[0].type != ZYDIS_OPERAND_TYPE_REGISTER) return false;

    // operand_width is the DESTINATION integer width (32 or 64).
    const u32 dst_width = insn.operand_width;
    if (dst_width != 32 && dst_width != 64) return false;

    const int dst_idx = ZydisGprToIndex(ops[0].reg.value);
    if (dst_idx < 0) return false;

    // Load src.low64 into host xmm0. The reg case loads the full 128-bit
    // lane; the mem case loads just the 64-bit double via vmovsd.
    if (ops[1].type == ZYDIS_OPERAND_TYPE_REGISTER) {
        const int src_vec = ZydisVecToIndex(ops[1].reg.value);
        if (src_vec < 0) return false;
        c.vmovdqu(xmm0, ptr[r13 + YmmChunkOffset(src_vec, 0)]);
    } else if (ops[1].type == ZYDIS_OPERAND_TYPE_MEMORY) {
        if (!EmitEffectiveAddress(ops[1].mem, next_rip, c))
            return false;
        c.vmovsd(xmm0, qword[rdx]);
    } else {
        return false;
    }

    // Run host VCVTTSD2SI. Truncation is enforced by the opcode itself.
    if (dst_width == 32) {
        c.vcvttsd2si(eax, xmm0);
    } else {
        c.vcvttsd2si(rax, xmm0);
    }

    // Store result into guest dst GPR slot (32-bit form already
    // zero-extended rax[63:32]).
    c.mov(qword[r13 + GprOffset(dst_idx)], rax);
    return true;
}
/// Same operand shape (dst, src1, src2/m), same chunk-by-chunk loop;
/// only the inner instruction differs (xor_ / and_ / or_). The PS-
/// vs-PI naming distinction is a type-class tag with no semantic
/// effect — XORPS and PXOR produce identical bit patterns and don't
/// touch flags.
///
/// Supported operand shapes:
///   * reg, reg, reg — dst = src1 OP src2
///   * reg, reg, mem — dst = src1 OP [mem]
///
/// VPANDN is the asymmetric member: dst = (NOT src1) AND src2. The
/// inversion applies ONLY to the first source operand, then ANDs with
/// the second — NOT a symmetric op. We realize it by inverting the
/// loaded src1 chunk (`not rax`) before the AND against src2. This is
/// the standard "clear bits of src2 that are set in src1" / "mask
/// complement" idiom string code uses to combine match masks.
///
/// Vector widths: 128 (VEX-128, zeros upper YMM) and 256 (full YMM).
enum class VecBitOpKind { Xor, And, Or, Andn };

bool EmitVecBitOp(const ZydisDecodedInstruction& insn, const ZydisDecodedOperand* ops,
                  u64 next_rip, Xbyak::CodeGenerator& c, VecBitOpKind kind) {
    // Mnemonic gating per kind — defensive check that the dispatcher
    // didn't misroute. If anything ever slips through, the false
    // return drops to the unsupported-instruction path.
    switch (kind) {
    case VecBitOpKind::Xor:
        if (insn.mnemonic != ZYDIS_MNEMONIC_VXORPS &&
            insn.mnemonic != ZYDIS_MNEMONIC_VPXOR)
            return false;
        break;
    case VecBitOpKind::And:
        if (insn.mnemonic != ZYDIS_MNEMONIC_VANDPS &&
            insn.mnemonic != ZYDIS_MNEMONIC_VPAND)
            return false;
        break;
    case VecBitOpKind::Or:
        if (insn.mnemonic != ZYDIS_MNEMONIC_VORPS &&
            insn.mnemonic != ZYDIS_MNEMONIC_VPOR)
            return false;
        break;
    case VecBitOpKind::Andn:
        if (insn.mnemonic != ZYDIS_MNEMONIC_VPANDN)
            return false;
        break;
    }

    if (ops[0].type != ZYDIS_OPERAND_TYPE_REGISTER) return false;
    if (ops[1].type != ZYDIS_OPERAND_TYPE_REGISTER) return false;

    const int dst_idx = ZydisVecToIndex(ops[0].reg.value);
    const int src1_idx = ZydisVecToIndex(ops[1].reg.value);
    if (dst_idx < 0 || src1_idx < 0)
        return false;

    const int vec_bits = ops[0].size;
    if (vec_bits != 128 && vec_bits != 256)
        return false;
    const int chunks = vec_bits / 64;

    // Emit one chunk's worth of {load src1, op against src2-or-mem, store}.
    // src1_chunk and src2_chunk_addr are the per-iteration arguments; the
    // helper handles the kind dispatch. For Andn the binary op is AND
    // (the NOT of src1 is applied separately, right after the load).
    auto emit_op = [&](const Xbyak::Reg64& dst_reg, const Xbyak::Address& src2_mem) {
        switch (kind) {
        case VecBitOpKind::Xor:  c.xor_(dst_reg, src2_mem); break;
        case VecBitOpKind::And:  c.and_(dst_reg, src2_mem); break;
        case VecBitOpKind::Or:   c.or_(dst_reg, src2_mem);  break;
        case VecBitOpKind::Andn: c.and_(dst_reg, src2_mem); break;
        }
    };
    // For VPANDN, invert the loaded src1 chunk before the AND.
    const bool invert_src1 = (kind == VecBitOpKind::Andn);

    if (ops[2].type == ZYDIS_OPERAND_TYPE_REGISTER) {
        const int src2_idx = ZydisVecToIndex(ops[2].reg.value);
        if (src2_idx < 0) return false;

        for (int i = 0; i < chunks; ++i) {
            c.mov(rax, qword[r13 + YmmChunkOffset(src1_idx, i)]);
            if (invert_src1) c.not_(rax);
            emit_op(rax, qword[r13 + YmmChunkOffset(src2_idx, i)]);
            c.mov(qword[r13 + YmmChunkOffset(dst_idx, i)], rax);
        }
    } else if (ops[2].type == ZYDIS_OPERAND_TYPE_MEMORY) {
        // mem-src: compute EA into rdx, then OP src1 chunks against
        // [rdx+i*8] qword-by-qword.
        if (!EmitEffectiveAddress(ops[2].mem, next_rip, c))
            return false;

        for (int i = 0; i < chunks; ++i) {
            c.mov(rax, qword[r13 + YmmChunkOffset(src1_idx, i)]);
            if (invert_src1) c.not_(rax);
            emit_op(rax, qword[rdx + i * 8]);
            c.mov(qword[r13 + YmmChunkOffset(dst_idx, i)], rax);
        }
    } else {
        return false;
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
///
/// The mask (last operand) may be a register OR a memory operand
/// (`vpshufb xmm, xmm, [mem]`). dst and src1 are always registers.
/// For a memory mask we compute the effective address once into
/// rdx and load each 128-bit lane from [rdx + lane*16]; VPSHUFB
/// never crosses a 128-bit lane boundary, so the per-lane loads are
/// independent and need no reordering.
bool EmitVpshufb(const ZydisDecodedInstruction& insn, const ZydisDecodedOperand* ops,
                 u64 next_rip, Xbyak::CodeGenerator& c) {
    if (insn.mnemonic != ZYDIS_MNEMONIC_VPSHUFB)
        return false;

    int dst_idx, src_idx;
    const ZydisDecodedOperand* mask_op;
    if (insn.operand_count_visible == 3) {
        if (ops[0].type != ZYDIS_OPERAND_TYPE_REGISTER ||
            ops[1].type != ZYDIS_OPERAND_TYPE_REGISTER)
            return false;
        dst_idx = ZydisVecToIndex(ops[0].reg.value);
        src_idx = ZydisVecToIndex(ops[1].reg.value);
        mask_op = &ops[2];
    } else if (insn.operand_count_visible == 2) {
        if (ops[0].type != ZYDIS_OPERAND_TYPE_REGISTER)
            return false;
        dst_idx = ZydisVecToIndex(ops[0].reg.value);
        src_idx = dst_idx;
        mask_op = &ops[1];
    } else {
        return false;
    }
    if (dst_idx < 0 || src_idx < 0)
        return false;

    const int vec_bits = ops[0].size;
    if (vec_bits != 128 && vec_bits != 256)
        return false;
    const int lanes = vec_bits / 128;

    // Process each 128-bit lane. For the 128-bit form there's only
    // one lane (chunks 0..1); for the 256-bit AVX2 form there's a
    // second lane (chunks 2..3) handled identically — VPSHUFB does
    // not cross 128-bit boundaries.
    if (mask_op->type == ZYDIS_OPERAND_TYPE_REGISTER) {
        const int mask_idx = ZydisVecToIndex(mask_op->reg.value);
        if (mask_idx < 0)
            return false;
        for (int lane = 0; lane < lanes; ++lane) {
            c.vmovdqu(xmm0, ptr[r13 + YmmChunkOffset(src_idx, lane * 2)]);
            c.vmovdqu(xmm1, ptr[r13 + YmmChunkOffset(mask_idx, lane * 2)]);
            c.vpshufb(xmm0, xmm0, xmm1);
            c.vmovdqu(ptr[r13 + YmmChunkOffset(dst_idx, lane * 2)], xmm0);
        }
    } else if (mask_op->type == ZYDIS_OPERAND_TYPE_MEMORY) {
        // mem-mask: EA → rdx (computed once); each lane loads its
        // 16-byte control vector from [rdx + lane*16].
        if (!EmitEffectiveAddress(mask_op->mem, next_rip, c))
            return false;
        for (int lane = 0; lane < lanes; ++lane) {
            c.vmovdqu(xmm0, ptr[r13 + YmmChunkOffset(src_idx, lane * 2)]);
            c.vmovdqu(xmm1, ptr[rdx + lane * 16]);
            c.vpshufb(xmm0, xmm0, xmm1);
            c.vmovdqu(ptr[r13 + YmmChunkOffset(dst_idx, lane * 2)], xmm0);
        }
    } else {
        return false;
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

/// ANDN r32/r64, r32/r64, r/m32/r/m64 — BMI1 three-operand AND-NOT.
///   dst = ~src1 & src2
///
/// Flags: writes SF, ZF from the result; clears OF, CF; PF undefined.
/// We use the host ANDN instruction (BMI1 is advertised in the CPUID spoof),
/// and round-trip rflags so the host's exact flag semantics propagate to the
/// guest's rflags shadow. Both 32- and 64-bit widths are handled, and src2 may
/// be a register or memory (the mem-src form, length 6 with a disp, was first
/// observed reg,(mem) in a loaded module on the AudioOutThread at guest
/// 0x808667cda, CUSA02394 "WE ARE DOOMED").
bool EmitAndn(const ZydisDecodedInstruction& insn, const ZydisDecodedOperand* ops,
              u64 next_rip, Xbyak::CodeGenerator& c) {
    if (insn.mnemonic != ZYDIS_MNEMONIC_ANDN)
        return false;
    if (insn.operand_width != 64 && insn.operand_width != 32)
        return false;
    if (insn.operand_count_visible != 3)
        return false;
    if (ops[0].type != ZYDIS_OPERAND_TYPE_REGISTER || ops[1].type != ZYDIS_OPERAND_TYPE_REGISTER)
        return false;

    const int dst_idx = ZydisGprToIndex(ops[0].reg.value);
    const int src1_idx = ZydisGprToIndex(ops[1].reg.value);
    if (dst_idx < 0 || src1_idx < 0)
        return false;

    const bool is32 = (insn.operand_width == 32);

    // Load src2 into r10/r10d. Register or memory. For the memory form,
    // EmitEffectiveAddress writes the address into rdx (and scratches rax), so
    // we resolve and dereference it BEFORE loading src1 / touching rflags.
    if (ops[2].type == ZYDIS_OPERAND_TYPE_REGISTER) {
        const int src2_idx = ZydisGprToIndex(ops[2].reg.value);
        if (src2_idx < 0)
            return false;
        if (is32) c.mov(r10d, dword[r13 + GprOffset(src2_idx)]);
        else      c.mov(r10,  qword[r13 + GprOffset(src2_idx)]);
    } else if (ops[2].type == ZYDIS_OPERAND_TYPE_MEMORY) {
        if (!EmitEffectiveAddress(ops[2].mem, next_rip, c))
            return false;
        if (is32) c.mov(r10d, dword[rdx]);
        else      c.mov(r10,  qword[rdx]);
    } else {
        return false;
    }

    // Load src1 into r9/r9d. The 32-bit d-suffixed names zero-extend bits
    // 63:32 — exactly what the 32-bit host ANDN wants.
    if (is32) c.mov(r9d, dword[r13 + GprOffset(src1_idx)]);
    else      c.mov(r9,  qword[r13 + GprOffset(src1_idx)]);

    c.mov(r8, qword[r13 + Offsets::Rflags]);
    c.push(r8);
    c.popfq();

    // Host ANDN. 32-bit form writes a 32-bit result and zero-extends bits
    // 63:32 of rax, so the qword storeback below captures the correct value.
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
    if (insn.mnemonic != ZYDIS_MNEMONIC_XADD) return false;
    if (insn.operand_width != 32 && insn.operand_width != 64) return false;
    if (ops[0].type != ZYDIS_OPERAND_TYPE_MEMORY) return false;
    if (ops[1].type != ZYDIS_OPERAND_TYPE_REGISTER) return false;

    const int src_idx = ZydisGprToIndex(ops[1].reg.value);
    if (src_idx < 0) return false;

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
bool EmitSetcc(const ZydisDecodedInstruction& insn, const ZydisDecodedOperand* ops,
               u64 next_rip, Xbyak::CodeGenerator& c) {
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
/// VPCMPEQ{B,W,D,Q} / VPCMPGT{B,W,D,Q} — packed integer compare-equal
/// or compare-greater-than at byte/word/dword/qword granularity.
///
/// For each element of width E in the vector:
///   dst[E] = (src1[E] PREDICATE src2[E]) ? all-ones : 0
///   PREDICATE: EQ for VPCMPEQ*, signed > for VPCMPGT*
///
/// No flags written (unlike scalar CMP). Per-lane independent — the
/// 256-bit form operates on two 128-bit halves with no cross-lane
/// movement, so we emit it as two 128-bit host ops.
///
/// All 8 cells share the same shape; the only difference is which
/// host op fires. VPCMPEQQ uses the 0x0F38 opcode map and VPCMPGTQ
/// requires SSE4.2 (both advertised in our CPUID spoof); xbyak
/// handles all the encoding selection internally.
///
/// Supported operand shapes:
///   * 3-operand reg-reg-reg
///   * 3-operand reg-reg-mem (the RIP-relative mask-from-memory pattern)
///   * 2-operand reg-reg(/mem) (assembler-folded dst == src1)
///
/// Refactored from EmitVecCmpEq (first 4 cells) at the second instance
/// (VPCMPGTD). Pre-wires the remaining 7 cells — VPCMPGT{B,W,Q} cost
/// nothing beyond their dispatch case + tests when first observed.
enum class VecIntCmp {
    EqB, EqW, EqD, EqQ,
    GtB, GtW, GtD, GtQ,
};

bool EmitVecIntCmp(const ZydisDecodedInstruction& insn, const ZydisDecodedOperand* ops,
                   u64 next_rip, Xbyak::CodeGenerator& c, VecIntCmp kind) {
    // Mnemonic gating — defensive check that the dispatcher routed
    // the right opcode to the right kind.
    auto expected = [&]() -> ZydisMnemonic {
        switch (kind) {
        case VecIntCmp::EqB: return ZYDIS_MNEMONIC_VPCMPEQB;
        case VecIntCmp::EqW: return ZYDIS_MNEMONIC_VPCMPEQW;
        case VecIntCmp::EqD: return ZYDIS_MNEMONIC_VPCMPEQD;
        case VecIntCmp::EqQ: return ZYDIS_MNEMONIC_VPCMPEQQ;
        case VecIntCmp::GtB: return ZYDIS_MNEMONIC_VPCMPGTB;
        case VecIntCmp::GtW: return ZYDIS_MNEMONIC_VPCMPGTW;
        case VecIntCmp::GtD: return ZYDIS_MNEMONIC_VPCMPGTD;
        case VecIntCmp::GtQ: return ZYDIS_MNEMONIC_VPCMPGTQ;
        }
        return ZYDIS_MNEMONIC_INVALID;
    }();
    if (insn.mnemonic != expected) return false;

    // Decode operands. Zydis presents 3 visible for the normal VEX
    // form (dst, src1, src2/mem) or 2 visible if the assembler
    // folded the source-register-equals-destination shorthand.
    int dst_idx, src1_idx;
    const ZydisDecodedOperand* src2_op = nullptr;
    if (insn.operand_count_visible == 3) {
        if (ops[0].type != ZYDIS_OPERAND_TYPE_REGISTER ||
            ops[1].type != ZYDIS_OPERAND_TYPE_REGISTER)
            return false;
        dst_idx  = ZydisVecToIndex(ops[0].reg.value);
        src1_idx = ZydisVecToIndex(ops[1].reg.value);
        src2_op  = &ops[2];
    } else if (insn.operand_count_visible == 2) {
        if (ops[0].type != ZYDIS_OPERAND_TYPE_REGISTER) return false;
        dst_idx  = ZydisVecToIndex(ops[0].reg.value);
        src1_idx = dst_idx; // implicit src1 == dst
        src2_op  = &ops[1];
    } else {
        return false;
    }
    if (dst_idx < 0 || src1_idx < 0) return false;

    const int vec_bits = ops[0].size;
    if (vec_bits != 128 && vec_bits != 256) return false;
    const int lanes = vec_bits / 128;

    // Helper: emit the right host op for the kind on (xmm0, xmm0, xmm1).
    auto emit_op = [&]() {
        switch (kind) {
        case VecIntCmp::EqB: c.vpcmpeqb(xmm0, xmm0, xmm1); break;
        case VecIntCmp::EqW: c.vpcmpeqw(xmm0, xmm0, xmm1); break;
        case VecIntCmp::EqD: c.vpcmpeqd(xmm0, xmm0, xmm1); break;
        case VecIntCmp::EqQ: c.vpcmpeqq(xmm0, xmm0, xmm1); break;
        case VecIntCmp::GtB: c.vpcmpgtb(xmm0, xmm0, xmm1); break;
        case VecIntCmp::GtW: c.vpcmpgtw(xmm0, xmm0, xmm1); break;
        case VecIntCmp::GtD: c.vpcmpgtd(xmm0, xmm0, xmm1); break;
        case VecIntCmp::GtQ: c.vpcmpgtq(xmm0, xmm0, xmm1); break;
        }
    };

    if (src2_op->type == ZYDIS_OPERAND_TYPE_REGISTER) {
        const int src2_idx = ZydisVecToIndex(src2_op->reg.value);
        if (src2_idx < 0) return false;
        for (int lane = 0; lane < lanes; ++lane) {
            c.vmovdqu(xmm0, ptr[r13 + YmmChunkOffset(src1_idx, lane * 2)]);
            c.vmovdqu(xmm1, ptr[r13 + YmmChunkOffset(src2_idx, lane * 2)]);
            emit_op();
            c.vmovdqu(ptr[r13 + YmmChunkOffset(dst_idx, lane * 2)], xmm0);
        }
    } else if (src2_op->type == ZYDIS_OPERAND_TYPE_MEMORY) {
        // mem-src: EA → rdx (computed once), then each 128-bit lane
        // loads from [rdx + lane*16]. Per-lane independence means
        // this is correct without any reordering concerns.
        if (!EmitEffectiveAddress(src2_op->mem, next_rip, c))
            return false;
        for (int lane = 0; lane < lanes; ++lane) {
            c.vmovdqu(xmm0, ptr[r13 + YmmChunkOffset(src1_idx, lane * 2)]);
            c.vmovdqu(xmm1, ptr[rdx + lane * 16]);
            emit_op();
            c.vmovdqu(ptr[r13 + YmmChunkOffset(dst_idx, lane * 2)], xmm0);
        }
    } else {
        return false;
    }

    // VEX 128-bit form zeros bits 255:128 of the destination YMM.
    if (vec_bits == 128) {
        c.xor_(rax, rax);
        c.mov(qword[r13 + YmmChunkOffset(dst_idx, 2)], rax);
        c.mov(qword[r13 + YmmChunkOffset(dst_idx, 3)], rax);
    }
    return true;
}

/// VPADD{B,W,D,Q} / VPSUB{B,W,D,Q} / VPMULL{W,D} — packed integer
/// add / subtract / signed multiply (low half) at byte/word/dword/qword
/// granularity (multiply only at W and D widths — VPMULLB does not
/// exist in AVX1/AVX2; VPMULLQ is AVX-512 only).
///
///   VPxxx xmm1, xmm2, xmm3/m128       ; 16/8/4/2 elements (128-bit)
///   VPxxx ymm1, ymm2, ymm3/m256       ; 32/16/8/4 elements (256-bit)
///
/// For each element i of element-width E:
///   dst[i] = src1[i] ± src2[i]
/// No saturation — values wrap modulo 2^E. No flags written.
/// Per-lane independent (256-bit = two 128-bit halves), VEX-128 zeros
/// upper YMM.
///
/// CANNOT be GPR-relayed at 64-bit chunk granularity the way the
/// bitwise family (VPXOR/VPAND/VPOR) can: carries/borrows would
/// propagate across element boundaries inside each chunk. The
/// scratch-XMM + host-op approach is the same one VPCMPEQ / VPSHUFB
/// already use; here the per-element width is implicit in the host
/// mnemonic, so the lifter doesn't need to know it.
///
/// Refactored from EmitVpsubd (first instance) at the second instance
/// (VPADDD). Pre-wires the rest of the Add/Sub × {B,W,D,Q} matrix —
/// each cell costs one extra enum value and one extra switch case.
/// If VPMUL*, VPMIN*, VPMAX*, etc. show up later, the right move is
/// to widen this enum further rather than create a parallel helper;
/// they all share the exact same per-lane scaffolding.
enum class VecIntArith {
    AddB, AddW, AddD, AddQ,
    SubB, SubW, SubD, SubQ,
    MulW, MulD,
};

bool EmitVecIntArith(const ZydisDecodedInstruction& insn, const ZydisDecodedOperand* ops,
                     u64 next_rip, Xbyak::CodeGenerator& c, VecIntArith kind) {
    // Mnemonic gating — defensive routing check.
    auto expected = [&]() -> ZydisMnemonic {
        switch (kind) {
        case VecIntArith::AddB: return ZYDIS_MNEMONIC_VPADDB;
        case VecIntArith::AddW: return ZYDIS_MNEMONIC_VPADDW;
        case VecIntArith::AddD: return ZYDIS_MNEMONIC_VPADDD;
        case VecIntArith::AddQ: return ZYDIS_MNEMONIC_VPADDQ;
        case VecIntArith::SubB: return ZYDIS_MNEMONIC_VPSUBB;
        case VecIntArith::SubW: return ZYDIS_MNEMONIC_VPSUBW;
        case VecIntArith::SubD: return ZYDIS_MNEMONIC_VPSUBD;
        case VecIntArith::SubQ: return ZYDIS_MNEMONIC_VPSUBQ;
        case VecIntArith::MulW: return ZYDIS_MNEMONIC_VPMULLW;
        case VecIntArith::MulD: return ZYDIS_MNEMONIC_VPMULLD;
        }
        return ZYDIS_MNEMONIC_INVALID;
    }();
    if (insn.mnemonic != expected) return false;

    if (insn.operand_count_visible != 3) return false;
    if (ops[0].type != ZYDIS_OPERAND_TYPE_REGISTER) return false;
    if (ops[1].type != ZYDIS_OPERAND_TYPE_REGISTER) return false;

    const int dst_idx = ZydisVecToIndex(ops[0].reg.value);
    const int src1_idx = ZydisVecToIndex(ops[1].reg.value);
    if (dst_idx < 0 || src1_idx < 0) return false;

    const int vec_bits = ops[0].size;
    if (vec_bits != 128 && vec_bits != 256) return false;
    const int lanes = vec_bits / 128;

    auto emit_op = [&]() {
        switch (kind) {
        case VecIntArith::AddB: c.vpaddb(xmm0, xmm0, xmm1); break;
        case VecIntArith::AddW: c.vpaddw(xmm0, xmm0, xmm1); break;
        case VecIntArith::AddD: c.vpaddd(xmm0, xmm0, xmm1); break;
        case VecIntArith::AddQ: c.vpaddq(xmm0, xmm0, xmm1); break;
        case VecIntArith::SubB: c.vpsubb(xmm0, xmm0, xmm1); break;
        case VecIntArith::SubW: c.vpsubw(xmm0, xmm0, xmm1); break;
        case VecIntArith::SubD: c.vpsubd(xmm0, xmm0, xmm1); break;
        case VecIntArith::SubQ: c.vpsubq(xmm0, xmm0, xmm1); break;
        case VecIntArith::MulW: c.vpmullw(xmm0, xmm0, xmm1); break;
        case VecIntArith::MulD: c.vpmulld(xmm0, xmm0, xmm1); break;
        }
    };

    if (ops[2].type == ZYDIS_OPERAND_TYPE_REGISTER) {
        const int src2_idx = ZydisVecToIndex(ops[2].reg.value);
        if (src2_idx < 0) return false;
        for (int lane = 0; lane < lanes; ++lane) {
            c.vmovdqu(xmm0, ptr[r13 + YmmChunkOffset(src1_idx, lane * 2)]);
            c.vmovdqu(xmm1, ptr[r13 + YmmChunkOffset(src2_idx, lane * 2)]);
            emit_op();
            c.vmovdqu(ptr[r13 + YmmChunkOffset(dst_idx, lane * 2)], xmm0);
        }
    } else if (ops[2].type == ZYDIS_OPERAND_TYPE_MEMORY) {
        if (!EmitEffectiveAddress(ops[2].mem, next_rip, c))
            return false;
        for (int lane = 0; lane < lanes; ++lane) {
            c.vmovdqu(xmm0, ptr[r13 + YmmChunkOffset(src1_idx, lane * 2)]);
            c.vmovdqu(xmm1, ptr[rdx + lane * 16]);
            emit_op();
            c.vmovdqu(ptr[r13 + YmmChunkOffset(dst_idx, lane * 2)], xmm0);
        }
    } else {
        return false;
    }

    if (vec_bits == 128) {
        c.xor_(rax, rax);
        c.mov(qword[r13 + YmmChunkOffset(dst_idx, 2)], rax);
        c.mov(qword[r13 + YmmChunkOffset(dst_idx, 3)], rax);
    }
    return true;
}

/// VMULPS / VDIVPS / VADDPS / VSUBPS — packed single-precision FP
/// VMULPD / VDIVPD / VADDPD / VSUBPD — packed double-precision FP
///
///   VxxxPS xmm1, xmm2, xmm3/m128       ; 4 elements (128-bit)
///   VxxxPS ymm1, ymm2, ymm3/m256       ; 8 elements (256-bit)
///   VxxxPD xmm1, xmm2, xmm3/m128       ; 2 elements (128-bit)
///   VxxxPD ymm1, ymm2, ymm3/m256       ; 4 elements (256-bit)
///
/// For each element i: dst[i] = src1[i] OP src2[i]. IEEE-754 semantics,
/// MXCSR rounding, NaN/inf propagate per spec. No flags written.
/// Per-lane independent (256-bit form = two 128-bit halves), VEX-128
/// zeros upper YMM.
///
/// Refactored from EmitVdivps (first instance) at the second instance
/// (VMULPS). The 4 × 2 = 8 cells share the exact same scaffolding;
/// the only per-cell variation is which host op fires. Mirrors the
/// structure of EmitScalarFp at the scalar-FP layer.
///
/// VSQRTPS / VSQRTPD are NOT in this enum yet — they're unary at the
/// host op level (no 3-operand form, unlike scalar VSQRTSS/SD which
/// have a merge-source third operand). When/if they show up, either
/// add a special case here or split into a parallel helper.
enum class VecFpKind { Mul, Div, Add, Sub };
enum class VecFpPrec { Single, Double };

bool EmitVecFpArith(const ZydisDecodedInstruction& insn, const ZydisDecodedOperand* ops,
                    u64 next_rip, Xbyak::CodeGenerator& c, VecFpKind kind, VecFpPrec prec) {
    // Mnemonic gating — defensive routing check.
    auto expected = [&]() -> ZydisMnemonic {
        const bool is_double = (prec == VecFpPrec::Double);
        switch (kind) {
        case VecFpKind::Mul: return is_double ? ZYDIS_MNEMONIC_VMULPD : ZYDIS_MNEMONIC_VMULPS;
        case VecFpKind::Div: return is_double ? ZYDIS_MNEMONIC_VDIVPD : ZYDIS_MNEMONIC_VDIVPS;
        case VecFpKind::Add: return is_double ? ZYDIS_MNEMONIC_VADDPD : ZYDIS_MNEMONIC_VADDPS;
        case VecFpKind::Sub: return is_double ? ZYDIS_MNEMONIC_VSUBPD : ZYDIS_MNEMONIC_VSUBPS;
        }
        return ZYDIS_MNEMONIC_INVALID;
    }();
    if (insn.mnemonic != expected) return false;

    if (insn.operand_count_visible != 3) return false;
    if (ops[0].type != ZYDIS_OPERAND_TYPE_REGISTER) return false;
    if (ops[1].type != ZYDIS_OPERAND_TYPE_REGISTER) return false;

    const int dst_idx = ZydisVecToIndex(ops[0].reg.value);
    const int src1_idx = ZydisVecToIndex(ops[1].reg.value);
    if (dst_idx < 0 || src1_idx < 0) return false;

    const int vec_bits = ops[0].size;
    if (vec_bits != 128 && vec_bits != 256) return false;
    const int lanes = vec_bits / 128;
    const bool is_double = (prec == VecFpPrec::Double);

    auto emit_op = [&]() {
        if (is_double) {
            switch (kind) {
            case VecFpKind::Mul: c.vmulpd(xmm0, xmm0, xmm1); break;
            case VecFpKind::Div: c.vdivpd(xmm0, xmm0, xmm1); break;
            case VecFpKind::Add: c.vaddpd(xmm0, xmm0, xmm1); break;
            case VecFpKind::Sub: c.vsubpd(xmm0, xmm0, xmm1); break;
            }
        } else {
            switch (kind) {
            case VecFpKind::Mul: c.vmulps(xmm0, xmm0, xmm1); break;
            case VecFpKind::Div: c.vdivps(xmm0, xmm0, xmm1); break;
            case VecFpKind::Add: c.vaddps(xmm0, xmm0, xmm1); break;
            case VecFpKind::Sub: c.vsubps(xmm0, xmm0, xmm1); break;
            }
        }
    };

    if (ops[2].type == ZYDIS_OPERAND_TYPE_REGISTER) {
        const int src2_idx = ZydisVecToIndex(ops[2].reg.value);
        if (src2_idx < 0) return false;
        for (int lane = 0; lane < lanes; ++lane) {
            c.vmovdqu(xmm0, ptr[r13 + YmmChunkOffset(src1_idx, lane * 2)]);
            c.vmovdqu(xmm1, ptr[r13 + YmmChunkOffset(src2_idx, lane * 2)]);
            emit_op();
            c.vmovdqu(ptr[r13 + YmmChunkOffset(dst_idx, lane * 2)], xmm0);
        }
    } else if (ops[2].type == ZYDIS_OPERAND_TYPE_MEMORY) {
        if (!EmitEffectiveAddress(ops[2].mem, next_rip, c))
            return false;
        for (int lane = 0; lane < lanes; ++lane) {
            c.vmovdqu(xmm0, ptr[r13 + YmmChunkOffset(src1_idx, lane * 2)]);
            c.vmovdqu(xmm1, ptr[rdx + lane * 16]);
            emit_op();
            c.vmovdqu(ptr[r13 + YmmChunkOffset(dst_idx, lane * 2)], xmm0);
        }
    } else {
        return false;
    }

    if (vec_bits == 128) {
        c.xor_(rax, rax);
        c.mov(qword[r13 + YmmChunkOffset(dst_idx, 2)], rax);
        c.mov(qword[r13 + YmmChunkOffset(dst_idx, 3)], rax);
    }
    return true;
}

/// VCMPPS — packed single-precision FP comparison with imm8 predicate.
///
///   VCMPPS xmm1, xmm2, xmm3/m128, imm8     ; 4 lanes (128-bit)
///   VCMPPS ymm1, ymm2, ymm3/m256, imm8     ; 8 lanes (256-bit)
///
/// For each 32-bit float element i:
///   dst[i] = (src1[i] PREDICATE src2[i]) ? 0xFFFFFFFF : 0x00000000
///
/// The 5-bit imm8 selects from 32 predicates (signaling/non-signaling
/// × 16 base ops). Low 3 bits cover the legacy SSE 8 predicates:
///   0=EQ, 1=LT, 2=LE, 3=UNORD, 4=NEQ, 5=NLT, 6=NLE, 7=ORD.
/// No flags written.
///
/// This is the FP analog of VPCMPEQ* (which produces the same kind of
/// all-1s/all-0s mask per element) and the predicate-generation half
/// of the VBLENDVPS branchless-select idiom:
///   vcmpps     mask, x, threshold, LT_OQ   ; mask[i] = (x[i] < t) ? -1 : 0
///   vblendvps  result, b, a, mask          ; result[i] = mask[i].sign ? a[i] : b[i]
///
/// Implementation: host VCMPPS on scratch xmm0/xmm1, imm8 read at
/// lift time from ops[3] and baked into the host instruction. Same
/// scaffolding as VBLENDPS, just different host op. First instance
/// of FP vector compare; when VCMPPD surfaces, refactor into
/// EmitVecFpCmp(prec) — they share the entire imm8 predicate space.
bool EmitVcmpps(const ZydisDecodedInstruction& insn, const ZydisDecodedOperand* ops,
                u64 next_rip, Xbyak::CodeGenerator& c) {
    if (insn.mnemonic != ZYDIS_MNEMONIC_VCMPPS) return false;
    if (insn.operand_count_visible != 4) return false;
    if (ops[0].type != ZYDIS_OPERAND_TYPE_REGISTER) return false;
    if (ops[1].type != ZYDIS_OPERAND_TYPE_REGISTER) return false;
    if (ops[3].type != ZYDIS_OPERAND_TYPE_IMMEDIATE) return false;

    const int dst_idx = ZydisVecToIndex(ops[0].reg.value);
    const int src1_idx = ZydisVecToIndex(ops[1].reg.value);
    if (dst_idx < 0 || src1_idx < 0) return false;

    const int vec_bits = ops[0].size;
    if (vec_bits != 128 && vec_bits != 256) return false;

    const u8 imm = static_cast<u8>(ops[3].imm.value.u);

    if (vec_bits == 128) {
        c.vmovdqu(xmm0, ptr[r13 + YmmChunkOffset(src1_idx, 0)]);

        if (ops[2].type == ZYDIS_OPERAND_TYPE_REGISTER) {
            const int src2_idx = ZydisVecToIndex(ops[2].reg.value);
            if (src2_idx < 0) return false;
            c.vmovdqu(xmm1, ptr[r13 + YmmChunkOffset(src2_idx, 0)]);
        } else if (ops[2].type == ZYDIS_OPERAND_TYPE_MEMORY) {
            if (!EmitEffectiveAddress(ops[2].mem, next_rip, c))
                return false;
            c.vmovdqu(xmm1, ptr[rdx]);
        } else {
            return false;
        }

        c.vcmpps(xmm0, xmm0, xmm1, imm);
        c.vmovdqu(ptr[r13 + YmmChunkOffset(dst_idx, 0)], xmm0);

        c.xor_(rax, rax);
        c.mov(qword[r13 + YmmChunkOffset(dst_idx, 2)], rax);
        c.mov(qword[r13 + YmmChunkOffset(dst_idx, 3)], rax);
    } else { // 256
        c.vmovdqu(ymm0, ptr[r13 + YmmChunkOffset(src1_idx, 0)]);

        if (ops[2].type == ZYDIS_OPERAND_TYPE_REGISTER) {
            const int src2_idx = ZydisVecToIndex(ops[2].reg.value);
            if (src2_idx < 0) return false;
            c.vmovdqu(ymm1, ptr[r13 + YmmChunkOffset(src2_idx, 0)]);
        } else if (ops[2].type == ZYDIS_OPERAND_TYPE_MEMORY) {
            if (!EmitEffectiveAddress(ops[2].mem, next_rip, c))
                return false;
            c.vmovdqu(ymm1, ptr[rdx]);
        } else {
            return false;
        }

        c.vcmpps(ymm0, ymm0, ymm1, imm);
        c.vmovdqu(ptr[r13 + YmmChunkOffset(dst_idx, 0)], ymm0);
    }
    return true;
}

/// VCMPSS — scalar single-precision FP comparison with an imm8 predicate.
///
///   VCMPSS xmm1, xmm2, xmm3/m32, imm8     (VEX.LIG.F3.0F.WIG C2 /r ib)
///
/// Compares only the LOW 32-bit float of src1 vs src2 using the predicate in
/// imm8, producing an all-ones (0xFFFFFFFF) or all-zeros (0x00000000) mask in
/// the destination's low dword. Bits 127:32 of the destination are copied
/// UNCHANGED from src1 (this is the scalar quirk — unlike VCMPPS, which writes
/// all lanes). VEX zeroes bits 255:128.
///
/// imm8 predicates (low 3 bits in the legacy SSE range, AVX extends to 5 bits):
/// 0=EQ_OQ 1=LT_OS 2=LE_OS 3=UNORD_Q 4=NEQ_UQ 5=NLT_US 6=NLE_US 7=ORD_Q, plus
/// the AVX signalling/quiet variants 8..31. We delegate the predicate handling
/// (including NaN/ordered/unordered semantics) to the host VCMPSS, which is
/// faithful by construction.
///
/// We run the host op on JIT-scratch xmm0 (=src1) / xmm1 (=src2): the low-dword
/// mask and the preserved src1 upper-96 bits both come out of the host
/// instruction, so a single store of the low 128 captures the whole result.
///
/// First observed reg,reg (2-byte VEX, length 5) in the CUSA02394 "WE ARE
/// DOOMED" eboot at guest 0x800105b28.
bool EmitVcmpss(const ZydisDecodedInstruction& insn, const ZydisDecodedOperand* ops,
                u64 next_rip, Xbyak::CodeGenerator& c) {
    if (insn.mnemonic != ZYDIS_MNEMONIC_VCMPSS) return false;
    if (insn.operand_count_visible != 4) return false;
    if (ops[0].type != ZYDIS_OPERAND_TYPE_REGISTER) return false;
    if (ops[1].type != ZYDIS_OPERAND_TYPE_REGISTER) return false;
    if (ops[3].type != ZYDIS_OPERAND_TYPE_IMMEDIATE) return false;

    const int dst_idx = ZydisVecToIndex(ops[0].reg.value);
    const int src1_idx = ZydisVecToIndex(ops[1].reg.value);
    if (dst_idx < 0 || src1_idx < 0) return false;

    const u8 imm = static_cast<u8>(ops[3].imm.value.u);

    // src1 -> xmm0 (also supplies the preserved upper 96 bits).
    c.vmovdqu(xmm0, ptr[r13 + YmmChunkOffset(src1_idx, 0)]);

    // src2 -> xmm1 (register or m32). For the m32 form, vcmpss only reads the
    // low dword, but loading the full 128 via the scratch is harmless and
    // matches the EmitVcmpps pattern; a 32-bit load would also do.
    if (ops[2].type == ZYDIS_OPERAND_TYPE_REGISTER) {
        const int src2_idx = ZydisVecToIndex(ops[2].reg.value);
        if (src2_idx < 0) return false;
        c.vmovdqu(xmm1, ptr[r13 + YmmChunkOffset(src2_idx, 0)]);
    } else if (ops[2].type == ZYDIS_OPERAND_TYPE_MEMORY) {
        if (!EmitEffectiveAddress(ops[2].mem, next_rip, c))
            return false;
        // Load only the scalar dword into xmm1's low lane (zeroing the rest is
        // fine — vcmpss only consults the low dword of src2).
        c.vmovss(xmm1, dword[rdx]);
    } else {
        return false;
    }

    c.vcmpss(xmm0, xmm0, xmm1, imm);
    c.vmovdqu(ptr[r13 + YmmChunkOffset(dst_idx, 0)], xmm0);

    // VEX zeroes bits 255:128 of the destination YMM.
    c.xor_(rax, rax);
    c.mov(qword[r13 + YmmChunkOffset(dst_idx, 2)], rax);
    c.mov(qword[r13 + YmmChunkOffset(dst_idx, 3)], rax);
    return true;
}

/// VROUNDSD — Round Scalar Double-precision to an integer per an imm8
/// rounding-control field.
///
///   VROUNDSD xmm1, xmm2, xmm3/m64, imm8
///     dst.low64     = round(src2.low64, imm8)   ; result is still a double
///     dst[127:64]   = src1[127:64]              ; preserved (VEX merge)
///     ymm.bits[255:128] = 0                     ; VEX-128 architectural
///
///   imm8 rounding control (Intel SDM):
///     bit 2 (0x04) set  → use the current MXCSR rounding mode
///     bit 2 clear       → bits[1:0] select: 00=nearest-even, 01=down (floor),
///                         10=up (ceil), 11=truncate (toward zero)
///     bit 3 (0x08)      → suppress precision exception (irrelevant here)
///
/// Gap from CUSA02394 at guest 0x800200e2c (reg,reg,imm8 — 3-byte VEX +
/// imm, length 6); compilers emit VROUNDSD for floor()/ceil()/trunc()/
/// round() on doubles, common in UI layout / animation math.
///
/// Same shape as VCMPSS: 4 visible operands (dst, src1, src2, imm8). Load
/// src1 into scratch xmm0 — this both feeds the (ignored) merge source and
/// supplies the preserved upper 96 bits — load src2 into xmm1, run host
/// VROUNDSD with the guest's imm8 verbatim (the host interprets the field
/// identically, including the MXCSR-defer bit), then store the low 128 and
/// zero the upper YMM. dst[127:64] preservation is free: host VROUNDSD
/// writes only xmm0.low64, leaving xmm0[127:64] (= src1's) untouched.
bool EmitVroundsd(const ZydisDecodedInstruction& insn, const ZydisDecodedOperand* ops,
                  u64 next_rip, Xbyak::CodeGenerator& c) {
    if (insn.mnemonic != ZYDIS_MNEMONIC_VROUNDSD) return false;
    if (insn.operand_count_visible != 4) return false;
    if (ops[0].type != ZYDIS_OPERAND_TYPE_REGISTER) return false;
    if (ops[1].type != ZYDIS_OPERAND_TYPE_REGISTER) return false;
    if (ops[3].type != ZYDIS_OPERAND_TYPE_IMMEDIATE) return false;

    const int dst_idx = ZydisVecToIndex(ops[0].reg.value);
    const int src1_idx = ZydisVecToIndex(ops[1].reg.value);
    if (dst_idx < 0 || src1_idx < 0) return false;

    const u8 imm = static_cast<u8>(ops[3].imm.value.u);

    // src1 -> xmm0 (supplies the preserved upper 96 bits).
    c.vmovdqu(xmm0, ptr[r13 + YmmChunkOffset(src1_idx, 0)]);

    // src2 -> xmm1 (register or m64). VROUNDSD reads only the low double.
    if (ops[2].type == ZYDIS_OPERAND_TYPE_REGISTER) {
        const int src2_idx = ZydisVecToIndex(ops[2].reg.value);
        if (src2_idx < 0) return false;
        c.vmovdqu(xmm1, ptr[r13 + YmmChunkOffset(src2_idx, 0)]);
    } else if (ops[2].type == ZYDIS_OPERAND_TYPE_MEMORY) {
        if (!EmitEffectiveAddress(ops[2].mem, next_rip, c))
            return false;
        c.vmovsd(xmm1, qword[rdx]);
    } else {
        return false;
    }

    c.vroundsd(xmm0, xmm0, xmm1, imm);
    c.vmovdqu(ptr[r13 + YmmChunkOffset(dst_idx, 0)], xmm0);

    // VEX zeroes bits 255:128 of the destination YMM.
    c.xor_(rax, rax);
    c.mov(qword[r13 + YmmChunkOffset(dst_idx, 2)], rax);
    c.mov(qword[r13 + YmmChunkOffset(dst_idx, 3)], rax);
    return true;
}

/// VMOVLHPS / VMOVHLPS — move a 64-bit (two-float) half between vectors.
///
///   VMOVLHPS xmm1, xmm2, xmm3   (VEX.128.0F.WIG 16 /r)
///       dst.q0 = src1.q0   (low  half of src1 stays)
///       dst.q1 = src2.q0   (LOW half of src2 moved to HIGH of dst)
///
///   VMOVHLPS xmm1, xmm2, xmm3   (VEX.128.0F.WIG 12 /r)
///       dst.q0 = src2.q1   (HIGH half of src2 moved to LOW of dst)
///       dst.q1 = src1.q1   (high half of src1 stays)
///
/// Both are register-only, 128-bit only (no memory or 256-bit form), take no
/// immediate, and are the classic qword-half move/swap pair compilers emit when
/// assembling or rearranging pairs of doubles / pairs of floats. VEX zeroes
/// bits 255:128 of the destination.
///
/// We delegate to the matching host instruction on JIT-scratch xmm0 (=src1) /
/// xmm1 (=src2): the host op reproduces the exact half-selection, so a single
/// store of the resulting low 128 captures the whole result.
///
/// VMOVLHPS first observed reg,reg (2-byte VEX, length 4) in the CUSA02394
/// "WE ARE DOOMED" eboot at guest 0x80013eecf.
enum class VecMovHalfKind { LhPs, HlPs };

bool EmitVmovLhHlps(const ZydisDecodedInstruction& insn, const ZydisDecodedOperand* ops,
                    Xbyak::CodeGenerator& c, VecMovHalfKind kind) {
    // Visible form is 3-operand (dst, src1, src2), all registers.
    if (insn.operand_count_visible != 3) return false;
    if (ops[0].type != ZYDIS_OPERAND_TYPE_REGISTER) return false;
    if (ops[1].type != ZYDIS_OPERAND_TYPE_REGISTER) return false;
    if (ops[2].type != ZYDIS_OPERAND_TYPE_REGISTER) return false;

    const int dst_idx = ZydisVecToIndex(ops[0].reg.value);
    const int src1_idx = ZydisVecToIndex(ops[1].reg.value);
    const int src2_idx = ZydisVecToIndex(ops[2].reg.value);
    if (dst_idx < 0 || src1_idx < 0 || src2_idx < 0) return false;

    c.vmovdqu(xmm0, ptr[r13 + YmmChunkOffset(src1_idx, 0)]); // src1
    c.vmovdqu(xmm1, ptr[r13 + YmmChunkOffset(src2_idx, 0)]); // src2

    if (kind == VecMovHalfKind::LhPs)
        c.vmovlhps(xmm0, xmm0, xmm1);
    else
        c.vmovhlps(xmm0, xmm0, xmm1);

    c.vmovdqu(ptr[r13 + YmmChunkOffset(dst_idx, 0)], xmm0);

    // VEX zeroes bits 255:128 of the destination YMM.
    c.xor_(rax, rax);
    c.mov(qword[r13 + YmmChunkOffset(dst_idx, 2)], rax);
    c.mov(qword[r13 + YmmChunkOffset(dst_idx, 3)], rax);
    return true;
}

/// FNSTCW / FLDCW — x87 FPU control-word store / load.
///
///   FNSTCW m16   (D9 /7)   store the FPU control word to memory
///   FLDCW  m16   (D9 /5)   load  the FPU control word from memory
///
/// This runtime does not model the x87 register stack or its control word —
/// all floating point is handled through the SSE/AVX path, whose rounding is
/// governed by MXCSR. These two instructions appear almost exclusively as the
/// libc float->int *truncation* idiom:
///
///     fnstcw [save]              ; save current control word
///     mov    ax, [save］ ; or    ; set rounding-control = truncate
///     or     ax, 0x0C00
///     mov    [tmp], ax
///     fldcw  [tmp]               ; install truncating mode
///     fistp  [result]            ; convert
///     fldcw  [save]              ; restore
///
/// Because we don't execute x87 conversions, the only thing the guest can
/// observe is the save/restore round-trip of the saved word. We therefore:
///   - FNSTCW: store the standard post-FINIT control word 0x037F (round to
///     nearest, 64-bit precision, all exceptions masked). This is the value
///     the control word holds in virtually all real programs, so a later
///     `fldcw [save]` restoring it is consistent.
///   - FLDCW: treat as a no-op (loading a control word has no effect on the
///     SSE path).
///
/// Modelling these faithfully (tracking the actual CW and honouring its
/// rounding-control bits on x87 conversions) would require an x87 stack — out
/// of scope here; the minimal model lets the truncation idiom round-trip.
///
/// First observed as FNSTCW (D9 /7, length 3) in libc at guest 0x8075abcf2
/// (CUSA02394 "WE ARE DOOMED").
bool EmitFnstcw(const ZydisDecodedInstruction& insn, const ZydisDecodedOperand* ops,
                u64 next_rip, Xbyak::CodeGenerator& c) {
    if (insn.mnemonic != ZYDIS_MNEMONIC_FNSTCW) return false;
    if (ops[0].type != ZYDIS_OPERAND_TYPE_MEMORY) return false;
    if (!EmitEffectiveAddress(ops[0].mem, next_rip, c))
        return false;
    // Store the default post-FINIT control word (0x037F) as a 16-bit value.
    c.mov(word[rdx], 0x037F);
    return true;
}

bool EmitFldcw(const ZydisDecodedInstruction& insn, const ZydisDecodedOperand* /*ops*/,
               u64 /*next_rip*/, Xbyak::CodeGenerator& /*c*/) {
    if (insn.mnemonic != ZYDIS_MNEMONIC_FLDCW) return false;
    // No-op: x87 rounding control is not modelled (SSE path uses MXCSR).
    return true;
}

/// STMXCSR / LDMXCSR — store / load the SSE control-and-status register.
///
///   STMXCSR m32   (0F AE /3)   store MXCSR to memory
///   LDMXCSR m32   (0F AE /2)   load  MXCSR from memory
///
/// Unlike the x87 control word (see EmitFnstcw), MXCSR is genuinely modelled:
/// GuestState carries an `mxcsr` field. We store/load that field directly, so
/// the save/restore idiom round-trips faithfully and any guest code that reads
/// back its saved MXCSR sees a consistent value.
///
/// Observed immediately after the FNSTCW in the libc float-conversion routine
/// (CUSA02394 "WE ARE DOOMED", guest 0x8075abcf5) — the routine saves both the
/// x87 control word and MXCSR.
///
/// KNOWN LIMITATION: the SSE/AVX lifters execute on the host's MXCSR, which we
/// do not re-sync to guest.mxcsr per instruction. So LDMXCSR records the new
/// value but does not (yet) change the rounding/masking actually applied to
/// subsequent SSE ops. Full per-op MXCSR honouring is a separate work item; for
/// the common save/restore-around-a-conversion pattern this faithful round-trip
/// of the field is sufficient.
bool EmitStmxcsr(const ZydisDecodedInstruction& insn, const ZydisDecodedOperand* ops,
                 u64 next_rip, Xbyak::CodeGenerator& c) {
    if (insn.mnemonic != ZYDIS_MNEMONIC_STMXCSR) return false;
    if (ops[0].type != ZYDIS_OPERAND_TYPE_MEMORY) return false;
    if (!EmitEffectiveAddress(ops[0].mem, next_rip, c))
        return false;
    // rdx = effective address. Store the 32-bit guest MXCSR.
    c.mov(eax, dword[r13 + offsetof(GuestState, mxcsr)]);
    c.mov(dword[rdx], eax);
    return true;
}

bool EmitLdmxcsr(const ZydisDecodedInstruction& insn, const ZydisDecodedOperand* ops,
                 u64 next_rip, Xbyak::CodeGenerator& c) {
    if (insn.mnemonic != ZYDIS_MNEMONIC_LDMXCSR) return false;
    if (ops[0].type != ZYDIS_OPERAND_TYPE_MEMORY) return false;
    if (!EmitEffectiveAddress(ops[0].mem, next_rip, c))
        return false;
    // rdx = effective address. Load the 32-bit value into the guest MXCSR.
    c.mov(eax, dword[rdx]);
    c.mov(dword[r13 + offsetof(GuestState, mxcsr)], eax);
    return true;
}

/// VBROADCASTSS — broadcast a 32-bit float to all lanes.
///
///   VBROADCASTSS xmm1, m32          ; 4 lanes (128-bit form); upper YMM zeroed
///   VBROADCASTSS ymm1, m32          ; 8 lanes (256-bit form)
///   VBROADCASTSS xmm1/ymm1, xmm2    ; AVX2-only; broadcasts xmm2.low32 to all lanes
///
/// The AVX1 baseline form (memory source) is what we lift here —
/// it's the canonical pattern for splatting a single constant or
/// loaded scalar across a vector lane in 4-wide arithmetic
/// (e.g. `x * scale_factor` where `scale_factor` lives in memory).
/// The AVX2 register-source form is not handled because the CPUID
/// spoof doesn't advertise AVX2 — games written for Jaguar without
/// runtime feature checks won't emit it.
///
/// Host VBROADCASTSS handles the broadcast in a single op. We compute
/// the effective address into rdx and let host hardware do the splat
/// into scratch xmm0/ymm0, then store back. For 128-bit form we
/// additionally zero the upper YMM (VEX-128 architectural).
bool EmitVbroadcastss(const ZydisDecodedInstruction& insn, const ZydisDecodedOperand* ops,
                      u64 next_rip, Xbyak::CodeGenerator& c) {
    if (insn.mnemonic != ZYDIS_MNEMONIC_VBROADCASTSS) return false;
    if (insn.operand_count_visible != 2) return false;
    if (ops[0].type != ZYDIS_OPERAND_TYPE_REGISTER) return false;
    // Initial cut: mem-source form only. AVX2 xmm-source form needs
    // a separate codepath but games shouldn't be emitting it given
    // our CPUID spoof.
    if (ops[1].type != ZYDIS_OPERAND_TYPE_MEMORY) return false;

    const int dst_idx = ZydisVecToIndex(ops[0].reg.value);
    if (dst_idx < 0) return false;

    const int vec_bits = ops[0].size;
    if (vec_bits != 128 && vec_bits != 256) return false;

    if (!EmitEffectiveAddress(ops[1].mem, next_rip, c))
        return false;

    if (vec_bits == 128) {
        // Host broadcasts the m32 to all 4 lanes of xmm0; upper YMM
        // not touched at the host op level — we zero it manually.
        c.vbroadcastss(xmm0, dword[rdx]);
        c.vmovdqu(ptr[r13 + YmmChunkOffset(dst_idx, 0)], xmm0);
        c.xor_(rax, rax);
        c.mov(qword[r13 + YmmChunkOffset(dst_idx, 2)], rax);
        c.mov(qword[r13 + YmmChunkOffset(dst_idx, 3)], rax);
    } else { // 256
        c.vbroadcastss(ymm0, dword[rdx]);
        c.vmovdqu(ptr[r13 + YmmChunkOffset(dst_idx, 0)], ymm0);
    }
    return true;
}

/// VCVTDQ2PS / VCVTPS2DQ / VCVTTPS2DQ — same-width (32-bit element) FP
/// conversions between signed int32 and float32.
///
///   VCVTDQ2PS  xmm/ymm, xmm/m | ymm/m       ; int32 → float32, MXCSR rounding
///   VCVTPS2DQ  xmm/ymm, xmm/m | ymm/m       ; float32 → int32, MXCSR rounding
///   VCVTTPS2DQ xmm/ymm, xmm/m | ymm/m       ; float32 → int32, truncating
///
/// For each element i: dst[i] = convert(src[i]) under the per-op
/// rounding rule. The "T" in TPS2DQ means truncation toward zero
/// (regardless of MXCSR). The non-T variant uses MXCSR rounding.
/// Out-of-range / NaN / inf in float→int conversions → 0x80000000
/// (Intel's "integer indefinite" value).
///
/// Per-lane independent. VEX-128 zeros upper YMM.
///
/// Refactored from EmitVcvtdq2ps (first instance) at the second
/// instance (VCVTTPS2DQ). Currently spans the **same-width** cells of
/// the conversion family. The width-changing siblings (DQ2PD,
/// PD2DQ/TPD2DQ, PS2PD, PD2PS) have different vector-width semantics
/// (input and output don't agree on register width) and need a
/// different code path; they'll go in a parallel helper when seen.
enum class VecFpCvt { Dq2Ps, Ps2Dq, TPs2Dq };

bool EmitVecFpCvt(const ZydisDecodedInstruction& insn, const ZydisDecodedOperand* ops,
                  u64 next_rip, Xbyak::CodeGenerator& c, VecFpCvt kind) {
    // Mnemonic gating — defensive routing check.
    auto expected = [&]() -> ZydisMnemonic {
        switch (kind) {
        case VecFpCvt::Dq2Ps:  return ZYDIS_MNEMONIC_VCVTDQ2PS;
        case VecFpCvt::Ps2Dq:  return ZYDIS_MNEMONIC_VCVTPS2DQ;
        case VecFpCvt::TPs2Dq: return ZYDIS_MNEMONIC_VCVTTPS2DQ;
        }
        return ZYDIS_MNEMONIC_INVALID;
    }();
    if (insn.mnemonic != expected) return false;

    if (insn.operand_count_visible != 2) return false;
    if (ops[0].type != ZYDIS_OPERAND_TYPE_REGISTER) return false;

    const int dst_idx = ZydisVecToIndex(ops[0].reg.value);
    if (dst_idx < 0) return false;

    const int vec_bits = ops[0].size;
    if (vec_bits != 128 && vec_bits != 256) return false;

    // Load src into host scratch (xmm0 or ymm0 depending on width).
    if (ops[1].type == ZYDIS_OPERAND_TYPE_REGISTER) {
        const int src_idx = ZydisVecToIndex(ops[1].reg.value);
        if (src_idx < 0) return false;
        if (vec_bits == 128) {
            c.vmovdqu(xmm0, ptr[r13 + YmmChunkOffset(src_idx, 0)]);
        } else {
            c.vmovdqu(ymm0, ptr[r13 + YmmChunkOffset(src_idx, 0)]);
        }
    } else if (ops[1].type == ZYDIS_OPERAND_TYPE_MEMORY) {
        if (!EmitEffectiveAddress(ops[1].mem, next_rip, c))
            return false;
        if (vec_bits == 128) {
            c.vmovdqu(xmm0, ptr[rdx]);
        } else {
            c.vmovdqu(ymm0, ptr[rdx]);
        }
    } else {
        return false;
    }

    // Helper: dispatch the host conversion op for the kind.
    auto emit_op_128 = [&]() {
        switch (kind) {
        case VecFpCvt::Dq2Ps:  c.vcvtdq2ps(xmm0, xmm0);  break;
        case VecFpCvt::Ps2Dq:  c.vcvtps2dq(xmm0, xmm0);  break;
        case VecFpCvt::TPs2Dq: c.vcvttps2dq(xmm0, xmm0); break;
        }
    };
    auto emit_op_256 = [&]() {
        switch (kind) {
        case VecFpCvt::Dq2Ps:  c.vcvtdq2ps(ymm0, ymm0);  break;
        case VecFpCvt::Ps2Dq:  c.vcvtps2dq(ymm0, ymm0);  break;
        case VecFpCvt::TPs2Dq: c.vcvttps2dq(ymm0, ymm0); break;
        }
    };

    if (vec_bits == 128) {
        emit_op_128();
        c.vmovdqu(ptr[r13 + YmmChunkOffset(dst_idx, 0)], xmm0);
        c.xor_(rax, rax);
        c.mov(qword[r13 + YmmChunkOffset(dst_idx, 2)], rax);
        c.mov(qword[r13 + YmmChunkOffset(dst_idx, 3)], rax);
    } else {
        emit_op_256();
        c.vmovdqu(ptr[r13 + YmmChunkOffset(dst_idx, 0)], ymm0);
    }
    return true;
}

/// VPSRA{W,D} / VPSRL{W,D,Q} / VPSLL{W,D,Q} — packed shift by imm8.
/// 8 cells across {SRA, SRL, SLL} × {W, D, Q except SRA-Q which is
/// AVX-512 only}.
///
///   VPxxx xmm1, xmm2, imm8         ; per-element shift by constant count
///   VPxxx ymm1, ymm2, imm8         ; same, 256-bit form
///
/// For each element of width E:
///   SRA: dst[i] = (signed)src[i] >> count, sign-fill; count > E-1 → sign-mask
///   SRL: dst[i] = (unsigned)src[i] >> count, zero-fill; count > E-1 → 0
///   SLL: dst[i] = src[i] << count, zero-fill;          count > E-1 → 0
///
/// No flags written. Per-lane independent (256-bit form = two 128-bit
/// halves). VEX-128 zeros upper YMM.
///
/// Initial cut handles only the imm8 form (length=5 in the encoding).
/// The xmm/mem-count form (`op dst, src, xmm3/m128` where the count
/// comes from xmm3.low64) is a different shape and gets its own
/// helper when first observed — that's a single shift count
/// broadcast across all lanes from the source register's low 64.
///
/// Refactored from EmitVpsrad (first instance) at the second
/// instance (VPSRLD). All cells share the same scaffolding —
/// scratch xmm0 load, host op with imm8, storeback. The 9-LOC-per-
/// cell payoff applies once an op enters the dispatch.
enum class VecShiftImm { SraW, SraD, SrlW, SrlD, SrlQ, SllW, SllD, SllQ };

bool EmitVecShiftImm(const ZydisDecodedInstruction& insn, const ZydisDecodedOperand* ops,
                     Xbyak::CodeGenerator& c, VecShiftImm kind) {
    // Mnemonic gating — defensive routing check.
    auto expected = [&]() -> ZydisMnemonic {
        switch (kind) {
        case VecShiftImm::SraW: return ZYDIS_MNEMONIC_VPSRAW;
        case VecShiftImm::SraD: return ZYDIS_MNEMONIC_VPSRAD;
        case VecShiftImm::SrlW: return ZYDIS_MNEMONIC_VPSRLW;
        case VecShiftImm::SrlD: return ZYDIS_MNEMONIC_VPSRLD;
        case VecShiftImm::SrlQ: return ZYDIS_MNEMONIC_VPSRLQ;
        case VecShiftImm::SllW: return ZYDIS_MNEMONIC_VPSLLW;
        case VecShiftImm::SllD: return ZYDIS_MNEMONIC_VPSLLD;
        case VecShiftImm::SllQ: return ZYDIS_MNEMONIC_VPSLLQ;
        }
        return ZYDIS_MNEMONIC_INVALID;
    }();
    if (insn.mnemonic != expected) return false;

    if (insn.operand_count_visible != 3) return false;
    if (ops[0].type != ZYDIS_OPERAND_TYPE_REGISTER) return false;
    if (ops[1].type != ZYDIS_OPERAND_TYPE_REGISTER) return false;
    // imm8 form only at this layer. The xmm/mem-count form has
    // different shape and goes through a parallel helper.
    if (ops[2].type != ZYDIS_OPERAND_TYPE_IMMEDIATE) return false;

    const int dst_idx = ZydisVecToIndex(ops[0].reg.value);
    const int src1_idx = ZydisVecToIndex(ops[1].reg.value);
    if (dst_idx < 0 || src1_idx < 0) return false;

    const int vec_bits = ops[0].size;
    if (vec_bits != 128 && vec_bits != 256) return false;
    const int lanes = vec_bits / 128;

    const u8 imm = static_cast<u8>(ops[2].imm.value.u);

    // Helper: emit the right host op for the kind on (xmm0, xmm0, imm).
    auto emit_op = [&]() {
        switch (kind) {
        case VecShiftImm::SraW: c.vpsraw(xmm0, xmm0, imm); break;
        case VecShiftImm::SraD: c.vpsrad(xmm0, xmm0, imm); break;
        case VecShiftImm::SrlW: c.vpsrlw(xmm0, xmm0, imm); break;
        case VecShiftImm::SrlD: c.vpsrld(xmm0, xmm0, imm); break;
        case VecShiftImm::SrlQ: c.vpsrlq(xmm0, xmm0, imm); break;
        case VecShiftImm::SllW: c.vpsllw(xmm0, xmm0, imm); break;
        case VecShiftImm::SllD: c.vpslld(xmm0, xmm0, imm); break;
        case VecShiftImm::SllQ: c.vpsllq(xmm0, xmm0, imm); break;
        }
    };

    for (int lane = 0; lane < lanes; ++lane) {
        c.vmovdqu(xmm0, ptr[r13 + YmmChunkOffset(src1_idx, lane * 2)]);
        emit_op();
        c.vmovdqu(ptr[r13 + YmmChunkOffset(dst_idx, lane * 2)], xmm0);
    }

    if (vec_bits == 128) {
        c.xor_(rax, rax);
        c.mov(qword[r13 + YmmChunkOffset(dst_idx, 2)], rax);
        c.mov(qword[r13 + YmmChunkOffset(dst_idx, 3)], rax);
    }
    return true;
}

/// VPSLLDQ / VPSRLDQ — shift the WHOLE 128-bit lane left/right by an
/// imm8 BYTE count, zero-filling. The "DQ" (double-quadword) suffix
/// distinguishes these from the per-element bit shifts (VPSLLD etc.):
/// here the entire 128-bit value moves as one unit, in byte steps.
///
///   VPSLLDQ xmm1, xmm2, imm8   ; dst = src << (imm8 * 8) bits, per lane
///   VPSRLDQ xmm1, xmm2, imm8   ; dst = src >> (imm8 * 8) bits, per lane
///
/// Byte-shift, not bit-shift: VPSLLDQ by 4 moves byte i to byte i+4,
/// zero-filling bytes 0..3 and discarding the top 4 bytes. A count
/// >= 16 zeroes the entire lane (all bytes shifted out).
///
/// LANE behavior (same caveat as the pack family): on the 256-bit
/// form the shift happens INDEPENDENTLY within each 128-bit lane —
/// bytes never cross the 128-bit boundary. The host op reproduces
/// this, so we run host VPSLLDQ/VPSRLDQ on xmm0 (128) or ymm0 (256)
/// rather than reconstructing the byte movement by hand.
///
/// Separate from EmitVecShiftImm: that helper is per-ELEMENT bit
/// shifts (W/D/Q element widths, count in bits); this is a whole-lane
/// BYTE shift. Different operation, different operand semantics —
/// hence its own emitter, with VPSRLDQ as the natural pair.
enum class VecByteShiftKind { Slldq, Srldq };

bool EmitVecByteShift(const ZydisDecodedInstruction& insn, const ZydisDecodedOperand* ops,
                      Xbyak::CodeGenerator& c, VecByteShiftKind kind) {
    const ZydisMnemonic expected =
        (kind == VecByteShiftKind::Slldq) ? ZYDIS_MNEMONIC_VPSLLDQ : ZYDIS_MNEMONIC_VPSRLDQ;
    if (insn.mnemonic != expected) return false;
    if (insn.operand_count_visible != 3) return false;
    if (ops[0].type != ZYDIS_OPERAND_TYPE_REGISTER) return false;
    if (ops[1].type != ZYDIS_OPERAND_TYPE_REGISTER) return false;
    if (ops[2].type != ZYDIS_OPERAND_TYPE_IMMEDIATE) return false;

    const int dst_idx = ZydisVecToIndex(ops[0].reg.value);
    const int src_idx = ZydisVecToIndex(ops[1].reg.value);
    if (dst_idx < 0 || src_idx < 0) return false;

    const u8 imm = static_cast<u8>(ops[2].imm.value.u & 0xFFu);
    const int vec_bits = ops[0].size;
    if (vec_bits != 128 && vec_bits != 256) return false;

    auto emit_op = [&](bool wide) {
        if (kind == VecByteShiftKind::Slldq) {
            if (wide) c.vpslldq(ymm0, ymm0, imm); else c.vpslldq(xmm0, xmm0, imm);
        } else {
            if (wide) c.vpsrldq(ymm0, ymm0, imm); else c.vpsrldq(xmm0, xmm0, imm);
        }
    };

    if (vec_bits == 128) {
        c.vmovdqu(xmm0, ptr[r13 + YmmChunkOffset(src_idx, 0)]);
        emit_op(false);
        c.vmovdqu(ptr[r13 + YmmChunkOffset(dst_idx, 0)], xmm0);
        c.xor_(rax, rax);
        c.mov(qword[r13 + YmmChunkOffset(dst_idx, 2)], rax);
        c.mov(qword[r13 + YmmChunkOffset(dst_idx, 3)], rax);
    } else {
        c.vmovdqu(ymm0, ptr[r13 + YmmChunkOffset(src_idx, 0)]);
        emit_op(true);
        c.vmovdqu(ptr[r13 + YmmChunkOffset(dst_idx, 0)], ymm0);
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

/// MFENCE / LFENCE — full and load memory fences. No operands.
///
/// MFENCE (0F AE F0) is the run-ending gap from CUSA02394: it appears
/// in libc at guest 0x8075ac337, in what looks like an atomic /
/// synchronization primitive (the surrounding block does a
/// cmpxchg-style load-modify sequence). LFENCE (0F AE E8) is its
/// load-ordering sibling; we handle it here pre-emptively since it
/// shares the opcode family and shows up in the same lock-free code.
///
/// Our JIT lowers each guest instruction to host code that executes
/// in program order on a single host thread per guest thread, and the
/// host x86 memory model is already strongly ordered (TSO). A guest
/// fence therefore has no reordering to prevent within our emitted
/// stream. We still emit the corresponding host fence: it is three
/// bytes after encoding, costs almost nothing, and preserves the
/// ordering guarantee with respect to any genuinely concurrent host
/// thread (e.g. the AudioOutThread / Game:Main pair observed in the
/// log) that may be touching shared guest memory.
bool EmitMfence(const ZydisDecodedInstruction& insn, Xbyak::CodeGenerator& c) {
    if (insn.mnemonic == ZYDIS_MNEMONIC_MFENCE) {
        c.mfence();
        return true;
    }
    if (insn.mnemonic == ZYDIS_MNEMONIC_LFENCE) {
        c.lfence();
        return true;
    }
    return false;
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
bool EmitBextr(const ZydisDecodedInstruction& insn, const ZydisDecodedOperand* ops,
               u64 next_rip, Xbyak::CodeGenerator& c) {
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

/// BLSI / BLSR / BLSMSK — the BMI1 "lowest-set-bit" trio. All are
/// 2-operand (dst, src), share VEX opcode 0xF3 (distinguished by the
/// ModRM.reg field), and have closely-related semantics:
///
///   BLSI   dst = (-src) & src        ; isolate lowest set bit
///   BLSR   dst = (src-1) & src       ; reset (clear) lowest set bit
///   BLSMSK dst = (src-1) ^ src       ; mask up to & incl lowest set bit
///
/// Flag semantics (identical Intel/AMD): SF = result MSB, OF = 0,
/// AF/PF undefined. ZF and CF differ across the three:
///   BLSI:   ZF = (result==0); CF = (src != 0)
///   BLSR:   ZF = (result==0); CF = (src == 0)
///   BLSMSK: ZF = 0;           CF = (src == 0)
/// We don't compute any of this by hand — the host op sets all the
/// flags per spec, and the rflags round-trip carries them across,
/// exactly as BEXTR/ANDN do. BMI1 is advertised in the CPUID spoof
/// (Jaguar has it), so a guest gating on the bit sees it available.
///
/// First instance is BLSI (hit in libc bit-manipulation code).
/// Parameterised so BLSR/BLSMSK are one-line dispatch additions.
enum class BlsKind { Blsi, Blsr, Blsmsk };

bool EmitBlsFamily(const ZydisDecodedInstruction& insn, const ZydisDecodedOperand* ops,
                   u64 next_rip, Xbyak::CodeGenerator& c, BlsKind kind) {
    const ZydisMnemonic expected =
        (kind == BlsKind::Blsi)   ? ZYDIS_MNEMONIC_BLSI :
        (kind == BlsKind::Blsr)   ? ZYDIS_MNEMONIC_BLSR :
                                    ZYDIS_MNEMONIC_BLSMSK;
    if (insn.mnemonic != expected) return false;
    if (insn.operand_width != 64 && insn.operand_width != 32) return false;
    if (insn.operand_count_visible != 2) return false;
    if (ops[0].type != ZYDIS_OPERAND_TYPE_REGISTER) return false;

    const int dst_idx = ZydisGprToIndex(ops[0].reg.value);
    if (dst_idx < 0) return false;

    const bool w32 = (insn.operand_width == 32);

    // Load src into r9 (register or memory). For the 32-bit form we
    // load 32 bits; the host op only consumes the low 32 anyway.
    if (ops[1].type == ZYDIS_OPERAND_TYPE_REGISTER) {
        const int src_idx = ZydisGprToIndex(ops[1].reg.value);
        if (src_idx < 0) return false;
        if (w32) c.mov(r9d, dword[r13 + GprOffset(src_idx)]);
        else     c.mov(r9,  qword[r13 + GprOffset(src_idx)]);
    } else if (ops[1].type == ZYDIS_OPERAND_TYPE_MEMORY) {
        if (!EmitEffectiveAddress(ops[1].mem, next_rip, c)) return false;
        if (w32) c.mov(r9d, dword[rdx]);
        else     c.mov(r9,  qword[rdx]);
    } else {
        return false;
    }

    // Round-trip flags through the host so it computes the BMI1
    // SF/ZF/CF per spec (OF cleared, AF/PF as the host leaves them).
    c.mov(r8, qword[r13 + Offsets::Rflags]);
    c.push(r8);
    c.popfq();

    // Run the host op. xbyak's blsi/blsr/blsmsk are (dst, src). A
    // 32-bit destination write zero-extends bits 63:32, so the qword
    // storeback lands a clean upper half.
    switch (kind) {
    case BlsKind::Blsi:
        if (w32) c.blsi(eax, r9d);   else c.blsi(rax, r9);   break;
    case BlsKind::Blsr:
        if (w32) c.blsr(eax, r9d);   else c.blsr(rax, r9);   break;
    case BlsKind::Blsmsk:
        if (w32) c.blsmsk(eax, r9d); else c.blsmsk(rax, r9); break;
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
bool EmitPopcnt(const ZydisDecodedInstruction& insn, const ZydisDecodedOperand* ops,
                u64 next_rip, Xbyak::CodeGenerator& c) {
    if (insn.mnemonic != ZYDIS_MNEMONIC_POPCNT) return false;
    if (insn.operand_width != 32 && insn.operand_width != 64) return false;
    if (ops[0].type != ZYDIS_OPERAND_TYPE_REGISTER) return false;

    const int dst_idx = ZydisGprToIndex(ops[0].reg.value);
    if (dst_idx < 0) return false;

    // Load src into r9 (reg or mem).
    if (ops[1].type == ZYDIS_OPERAND_TYPE_REGISTER) {
        const int src_idx = ZydisGprToIndex(ops[1].reg.value);
        if (src_idx < 0) return false;
        c.mov(r9, qword[r13 + GprOffset(src_idx)]);
    } else if (ops[1].type == ZYDIS_OPERAND_TYPE_MEMORY) {
        if (!EmitEffectiveAddress(ops[1].mem, next_rip, c)) return false;
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

/// LZCNT — count leading zeros (BMI1/ABM extension).
///   dst := number of leading zero bits in src (counting from MSB)
///   If src == 0: dst = operand size (32 or 64), CF = 1
///   Otherwise:   dst = count, CF = 0
///   ZF set iff result is zero (i.e. iff MSB of src was set)
///   OF/SF/PF/AF are architecturally undefined.
///
/// LZCNT's defined ZF/CF behavior is IDENTICAL on Intel and AMD — the
/// only documented difference is vs BSR (a different instruction),
/// not between vendors. So no vendor-divergence handling is needed
/// here beyond the round-trip.
///
/// ── FORWARD NOTE: BSF / BSR / TZCNT (not yet implemented) ──────────
/// When these get emitters, mind the divergences:
///   * BSF/BSR with src==0: dst is "undefined" on BOTH vendors (ZF=1
///     signals the zero-source case). BUT the left-behind dst value
///     differs: AMD leaves the DESTINATION UNCHANGED (documented
///     behavior in the APM — "if src==0, dst is not modified"), while
///     Intel documents it as truly undefined and some parts scribble.
///     For an AuthenticAMD guest, emit "preserve dst on zero source".
///     A few real binaries actually rely on the AMD preserve behavior.
///   * TZCNT/LZCNT vs BSF/BSR on zero source: TZCNT/LZCNT give a
///     DEFINED result (operand size) and set CF; BSF/BSR give
///     undefined dst and set ZF. Don't conflate them — TZCNT is NOT
///     "BSF that always works".
/// ───────────────────────────────────────────────────────────────────
///
/// Note: LZCNT is encoded with `F3` prefix on top of the BSR opcode
/// (`0F BD`). On non-ABM CPUs the prefix is ignored and the
/// instruction silently decodes as BSR — a different operation with
/// different undefined-on-zero semantics. We always behave as LZCNT
/// since the CPUID spoof advertises ABM (leaf 0x80000001 ECX bit 5).
///
/// Implementation mirrors POPCNT: load src, set host rflags, run
/// host LZCNT (which writes ZF/CF per spec and leaves the undefined
/// bits to the host CPU's choice), capture rflags back.
bool EmitLzcnt(const ZydisDecodedInstruction& insn, const ZydisDecodedOperand* ops,
               u64 next_rip, Xbyak::CodeGenerator& c) {
    if (insn.mnemonic != ZYDIS_MNEMONIC_LZCNT) return false;
    if (insn.operand_width != 32 && insn.operand_width != 64) return false;
    if (ops[0].type != ZYDIS_OPERAND_TYPE_REGISTER) return false;

    const int dst_idx = ZydisGprToIndex(ops[0].reg.value);
    if (dst_idx < 0) return false;

    if (ops[1].type == ZYDIS_OPERAND_TYPE_REGISTER) {
        const int src_idx = ZydisGprToIndex(ops[1].reg.value);
        if (src_idx < 0) return false;
        c.mov(r9, qword[r13 + GprOffset(src_idx)]);
    } else if (ops[1].type == ZYDIS_OPERAND_TYPE_MEMORY) {
        if (!EmitEffectiveAddress(ops[1].mem, next_rip, c)) return false;
        c.mov(r9, qword[rdx]);
    } else {
        return false;
    }

    c.mov(r8, qword[r13 + Offsets::Rflags]);
    c.push(r8);
    c.popfq();

    if (insn.operand_width == 32) {
        c.lzcnt(eax, r9d);
    } else {
        c.lzcnt(rax, r9);
    }

    c.pushfq();
    c.pop(r8);
    c.mov(qword[r13 + Offsets::Rflags], r8);

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
    //
    // ── INTEL vs AMD DIVERGENCE (cache-topology leaves) ───────────
    // We report max standard leaf = 7. A guest probing the cache
    // hierarchy therefore gets ZERO for leaf 4 (the default-response
    // path). This is CORRECT for an AuthenticAMD guest: leaf 4
    // (Deterministic Cache Parameters) is an INTEL leaf — AMD does
    // not implement it. AMD exposes cache topology through the
    // extended leaves 0x80000005 (L1), 0x80000006 (L2/L3), and
    // 0x8000001d (with TOPOEXT). A well-behaved AMD-aware guest never
    // queries leaf 4; one that does should see it absent.
    //
    // DO NOT "fix" missing cache info by adding an Intel-style leaf 4
    // here — that would make the spoof internally inconsistent
    // (AuthenticAMD vendor string + Intel-only leaf). If a title
    // actually needs cache geometry, add the AMD extended leaves
    // 0x80000005/6 instead. Reporting max-leaf 7 with zeroed leaf 4
    // is the accurate Jaguar behavior and is intentional.
    // ───────────────────────────────────────────────────────────────
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
bool EmitXgetbv(const ZydisDecodedInstruction& insn,
                Xbyak::CodeGenerator& c) {
    if (insn.mnemonic != ZYDIS_MNEMONIC_XGETBV) return false;

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
    if (insn.mnemonic != ZYDIS_MNEMONIC_VPTEST) return false;
    if (ops[0].type != ZYDIS_OPERAND_TYPE_REGISTER) return false;
    if (ops[1].type != ZYDIS_OPERAND_TYPE_REGISTER) return false;

    const int a_idx = ZydisVecToIndex(ops[0].reg.value);
    const int b_idx = ZydisVecToIndex(ops[1].reg.value);
    if (a_idx < 0 || b_idx < 0) return false;

    const int vec_bits = ops[0].size;
    if (vec_bits != 128 && vec_bits != 256) return false;

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
    if (insn.mnemonic != ZYDIS_MNEMONIC_VPCMPISTRI) return false;
    if (ops[0].type != ZYDIS_OPERAND_TYPE_REGISTER) return false;
    if (ops[1].type != ZYDIS_OPERAND_TYPE_REGISTER) return false;
    if (ops[2].type != ZYDIS_OPERAND_TYPE_IMMEDIATE) return false;

    const int a_idx = ZydisVecToIndex(ops[0].reg.value);
    const int b_idx = ZydisVecToIndex(ops[1].reg.value);
    if (a_idx < 0 || b_idx < 0) return false;

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

/// VPCMPISTRM xmm1, xmm2/m128, imm8 — SSE4.2 "compare implicit-length
/// strings, return MASK". The sibling of VPCMPISTRI: identical compare
/// machinery, but the result is a MASK written to the implicit XMM0
/// rather than an index in ECX. Used by glibc string functions
/// (strpbrk / strcspn / strspn-style scans) that want the full
/// match mask rather than just the first index.
///
/// Architectural behavior:
///   XMM0 := result mask. imm8 bit 6 selects the format:
///            bit6 = 0  → bit-mask: one bit per element in the low
///                        bits of XMM0, rest zeroed.
///            bit6 = 1  → byte/word-mask: each element position is
///                        0x00 or 0xFF (byte mode) / 0x0000 or 0xFFFF
///                        (word mode), filling the full 128 bits.
///   CF/ZF/SF/OF := same string-comparison flags as VPCMPISTRI.
///   AF, PF := 0
///
/// CRITICAL: host VPCMPISTRM writes its result to the implicit,
/// hard-wired physical XMM0. We must therefore NOT use host xmm0 as
/// an operand-load scratch (as VPCMPISTRI does) — the result would
/// overwrite an operand before the op even runs. We load the two
/// operands into host xmm1/xmm2 instead, run the op (which writes
/// xmm0), then store xmm0 back to the guest XMM0 slot.
///
/// imm8 bit 6 (mask format) needs no special handling here: the host
/// op produces the correctly-formatted XMM0 either way, and we copy
/// the full 128 bits back verbatim.
///
/// Reg-mem second-operand form is deferred (same rationale as
/// VPCMPISTRI — the string loops use reg-reg after a load).
bool EmitVpcmpistrm(const ZydisDecodedInstruction& insn, const ZydisDecodedOperand* ops,
                    Xbyak::CodeGenerator& c) {
    if (insn.mnemonic != ZYDIS_MNEMONIC_VPCMPISTRM) return false;
    if (ops[0].type != ZYDIS_OPERAND_TYPE_REGISTER) return false;
    if (ops[1].type != ZYDIS_OPERAND_TYPE_REGISTER) return false;
    if (ops[2].type != ZYDIS_OPERAND_TYPE_IMMEDIATE) return false;

    const int a_idx = ZydisVecToIndex(ops[0].reg.value);
    const int b_idx = ZydisVecToIndex(ops[1].reg.value);
    if (a_idx < 0 || b_idx < 0) return false;

    const u8 imm = static_cast<u8>(ops[2].imm.value.u & 0xFFu);

    // Load operands into xmm1/xmm2 — NOT xmm0, which receives the
    // implicit mask result.
    c.vmovdqu(xmm1, ptr[r13 + YmmChunkOffset(a_idx, 0)]);
    c.vmovdqu(xmm2, ptr[r13 + YmmChunkOffset(b_idx, 0)]);

    // Run host VPCMPISTRM. Writes the mask to physical xmm0 and sets
    // CF/ZF/SF/OF per spec (PF/AF cleared).
    c.vpcmpistrm(xmm1, xmm2, imm);

    // Capture the flags IMMEDIATELY — before any flag-affecting
    // instruction below. The upper-YMM-zeroing `xor` would otherwise
    // clobber CF/ZF before we read them. Stash the raw rflags in r8.
    c.pushfq();
    c.pop(r8);

    // Store host xmm0 (the mask) into the guest XMM0 slot (vec index
    // 0, low 128). VEX semantics zero bits 255:128 of that YMM. The
    // `xor` here trashes host flags, which is why we captured above.
    c.vmovdqu(ptr[r13 + YmmChunkOffset(0, 0)], xmm0);
    c.xor_(rax, rax);
    c.mov(qword[r13 + YmmChunkOffset(0, 2)], rax);
    c.mov(qword[r13 + YmmChunkOffset(0, 3)], rax);

    // Merge the captured arithmetic flag bits into guest rflags —
    // identical mask-and-or pattern as VPCMPISTRI / VPTEST.
    constexpr u64 kArithMask = 0x8D5; // CF|PF|AF|ZF|SF|OF
    c.and_(r8, kArithMask);
    c.mov(rdx, qword[r13 + Offsets::Rflags]);
    c.mov(rcx, ~kArithMask);
    c.and_(rdx, rcx);
    c.or_(rdx, r8);
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
    if (insn.mnemonic != ZYDIS_MNEMONIC_VPHADDD) return false;

    int dst_idx, src1_idx, src2_idx;
    if (insn.operand_count_visible == 3) {
        if (ops[0].type != ZYDIS_OPERAND_TYPE_REGISTER ||
            ops[1].type != ZYDIS_OPERAND_TYPE_REGISTER ||
            ops[2].type != ZYDIS_OPERAND_TYPE_REGISTER)
            return false;
        dst_idx  = ZydisVecToIndex(ops[0].reg.value);
        src1_idx = ZydisVecToIndex(ops[1].reg.value);
        src2_idx = ZydisVecToIndex(ops[2].reg.value);
    } else if (insn.operand_count_visible == 2) {
        if (ops[0].type != ZYDIS_OPERAND_TYPE_REGISTER ||
            ops[1].type != ZYDIS_OPERAND_TYPE_REGISTER)
            return false;
        dst_idx  = ZydisVecToIndex(ops[0].reg.value);
        src1_idx = dst_idx;
        src2_idx = ZydisVecToIndex(ops[1].reg.value);
    } else {
        return false;
    }
    if (dst_idx < 0 || src1_idx < 0 || src2_idx < 0) return false;

    const int vec_bits = ops[0].size;
    if (vec_bits != 128 && vec_bits != 256) return false;

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

/// VPUNPCKLQDQ / VPUNPCKHQDQ — unpack and interleave the low or high 64-bit
/// halves of two vectors. For each 128-bit lane:
///   VPUNPCKLQDQ: dst[0] = src1.q0, dst[1] = src2.q0   (low  64 of each)
///   VPUNPCKHQDQ: dst[0] = src1.q1, dst[1] = src2.q1   (high 64 of each)
/// The unused 64-bit half of each source within the lane is discarded. The
/// 256-bit form runs this independently for the low and high 128-bit lanes
/// (no cross-lane shuffle). No flags affected.
///
/// VPUNPCKLDQ / VPUNPCKHDQ are the dword-granularity siblings (interleave
/// 32-bit elements instead of 64-bit). For each 128-bit lane:
///   VPUNPCKLDQ: dst = {src1.d0, src2.d0, src1.d1, src2.d1}  (low  dwords)
///   VPUNPCKHDQ: dst = {src1.d2, src2.d2, src1.d3, src2.d3}  (high dwords)
/// VPUNPCKLDQ is the gap from CUSA02394 at guest 0x80023f6bf (reg,reg, 2-byte
/// VEX, length 4) — in the same vector block as the VPUNPCKHQDQ / VSHUFPS /
/// VUNPCKLPS shuffles. The dword and qword forms share identical structure;
/// only the host mnemonic differs, so they fold into one emitter.
///
/// Compiler-emitted in vectorised gather/broadcast and transpose patterns:
/// take values that landed in xmm regs and pack/interleave them. We run the
/// matching host op directly on JIT-scratch xmm0/xmm1 — same pattern as
/// VSHUFPS / VUNPCK*. Both the 3-operand visible form (dst, src1, src2) and
/// the 2-operand folded form (dst==src1, src2) are handled; src2 may be a
/// register or m128/m256.
/// VPUNPCKLWD / VPUNPCKHWD are the word-granularity siblings (interleave
/// 16-bit elements). For each 128-bit lane:
///   VPUNPCKLWD: dst = {s1.w0,s2.w0,s1.w1,s2.w1,s1.w2,s2.w2,s1.w3,s2.w3}
///   VPUNPCKHWD: dst = {s1.w4,s2.w4,...,s1.w7,s2.w7}
/// VPUNPCKLWD is the gap at guest 0x80023f6cf, 16 bytes after the VPUNPCKLDQ
/// in the same vector block. Same structure; only the host mnemonic differs.
enum class VecPunpckKind { LowQ, HighQ, LowD, HighD, LowW, HighW };

bool EmitVpunpckDQdq(const ZydisDecodedInstruction& insn, const ZydisDecodedOperand* ops,
                     u64 next_rip, Xbyak::CodeGenerator& c, VecPunpckKind kind) {
    int dst_idx, src1_idx;
    const ZydisDecodedOperand* src2 = nullptr;
    if (insn.operand_count_visible == 3) {
        if (ops[0].type != ZYDIS_OPERAND_TYPE_REGISTER ||
            ops[1].type != ZYDIS_OPERAND_TYPE_REGISTER)
            return false;
        dst_idx  = ZydisVecToIndex(ops[0].reg.value);
        src1_idx = ZydisVecToIndex(ops[1].reg.value);
        src2 = &ops[2];
    } else if (insn.operand_count_visible == 2) {
        if (ops[0].type != ZYDIS_OPERAND_TYPE_REGISTER)
            return false;
        dst_idx  = ZydisVecToIndex(ops[0].reg.value);
        src1_idx = dst_idx;
        src2 = &ops[1];
    } else {
        return false;
    }
    if (dst_idx < 0 || src1_idx < 0) return false;

    const int vec_bits = ops[0].size;
    if (vec_bits != 128 && vec_bits != 256) return false;
    const bool is256 = (vec_bits == 256);

    // src1 -> xmm0/ymm0.
    if (is256)
        c.vmovdqu(ymm0, ptr[r13 + YmmChunkOffset(src1_idx, 0)]);
    else
        c.vmovdqu(xmm0, ptr[r13 + YmmChunkOffset(src1_idx, 0)]);

    // src2 -> xmm1/ymm1 (register or memory).
    if (src2->type == ZYDIS_OPERAND_TYPE_REGISTER) {
        const int src2_idx = ZydisVecToIndex(src2->reg.value);
        if (src2_idx < 0) return false;
        if (is256)
            c.vmovdqu(ymm1, ptr[r13 + YmmChunkOffset(src2_idx, 0)]);
        else
            c.vmovdqu(xmm1, ptr[r13 + YmmChunkOffset(src2_idx, 0)]);
    } else if (src2->type == ZYDIS_OPERAND_TYPE_MEMORY) {
        if (!EmitEffectiveAddress(src2->mem, next_rip, c))
            return false;
        if (is256)
            c.vmovdqu(ymm1, ptr[rdx]);
        else
            c.vmovdqu(xmm1, ptr[rdx]);
    } else {
        return false;
    }

    if (is256) {
        switch (kind) {
        case VecPunpckKind::LowQ:  c.vpunpcklqdq(ymm0, ymm0, ymm1); break;
        case VecPunpckKind::HighQ: c.vpunpckhqdq(ymm0, ymm0, ymm1); break;
        case VecPunpckKind::LowD:  c.vpunpckldq(ymm0, ymm0, ymm1);  break;
        case VecPunpckKind::HighD: c.vpunpckhdq(ymm0, ymm0, ymm1);  break;
        case VecPunpckKind::LowW:  c.vpunpcklwd(ymm0, ymm0, ymm1);  break;
        case VecPunpckKind::HighW: c.vpunpckhwd(ymm0, ymm0, ymm1);  break;
        }
        c.vmovdqu(ptr[r13 + YmmChunkOffset(dst_idx, 0)], ymm0);
    } else {
        switch (kind) {
        case VecPunpckKind::LowQ:  c.vpunpcklqdq(xmm0, xmm0, xmm1); break;
        case VecPunpckKind::HighQ: c.vpunpckhqdq(xmm0, xmm0, xmm1); break;
        case VecPunpckKind::LowD:  c.vpunpckldq(xmm0, xmm0, xmm1);  break;
        case VecPunpckKind::HighD: c.vpunpckhdq(xmm0, xmm0, xmm1);  break;
        case VecPunpckKind::LowW:  c.vpunpcklwd(xmm0, xmm0, xmm1);  break;
        case VecPunpckKind::HighW: c.vpunpckhwd(xmm0, xmm0, xmm1);  break;
        }
        c.vmovdqu(ptr[r13 + YmmChunkOffset(dst_idx, 0)], xmm0);
        // VEX-128 zeroes bits 255:128 of the destination YMM.
        c.xor_(rax, rax);
        c.mov(qword[r13 + YmmChunkOffset(dst_idx, 2)], rax);
        c.mov(qword[r13 + YmmChunkOffset(dst_idx, 3)], rax);
    }
    return true;
}

/// VPACKSSDW / VPACKSSWB / VPACKUSDW / VPACKUSWB — pack two vectors of
/// wider integer elements into narrower ones with saturation. The four
/// members of the pack family share an identical 3-operand shape and
/// differ only in element widths and signed-vs-unsigned saturation:
///
///   VPACKSSDW dst, s1, s2/m  : 32-bit signed   -> 16-bit SIGNED   sat
///   VPACKSSWB dst, s1, s2/m  : 16-bit signed   -> 8-bit  SIGNED   sat
///   VPACKUSDW dst, s1, s2/m  : 32-bit signed   -> 16-bit UNSIGNED sat
///   VPACKUSWB dst, s1, s2/m  : 16-bit signed   -> 8-bit  UNSIGNED sat
///
/// Saturation clamps each source element to the destination element's
/// representable range before narrowing. For the unsigned-dword form
/// (VPACKUSDW): src < 0 -> 0, src > 0xFFFF -> 0xFFFF, else src.
///
/// LAYOUT (the part that's easy to get wrong): pack operates PER
/// 128-BIT LANE, not across the whole register. Within each 128-bit
/// lane, the narrowed elements of src1 fill the LOW half of that lane
/// and the narrowed elements of src2 fill the HIGH half. So for the
/// 128-bit form: dst.low = saturate(src1), dst.high = saturate(src2).
/// For the 256-bit form the same low=src1/high=src2 rule applies
/// INDEPENDENTLY within each of the two 128-bit lanes — it does NOT
/// concatenate all of src1 then all of src2. We sidestep having to
/// reproduce that interleave by hand: running the host pack op on
/// xmm (128) or ymm (256) reproduces the architectural lane behavior
/// exactly, since the host instruction has the same per-lane semantics.
///
/// First instance is VPACKUSDW (hit in pixel/format-conversion code).
/// Parameterised as EmitVecPack(kind) so the sibling members slot in
/// as one-line dispatch additions when first seen.
enum class VecPackKind { Ssdw, Sswb, Usdw, Uswb };

bool EmitVecPack(const ZydisDecodedInstruction& insn, const ZydisDecodedOperand* ops,
                 u64 next_rip, Xbyak::CodeGenerator& c, VecPackKind kind) {
    // Mnemonic gating — defensive routing check.
    auto expected = [&]() -> ZydisMnemonic {
        switch (kind) {
        case VecPackKind::Ssdw: return ZYDIS_MNEMONIC_VPACKSSDW;
        case VecPackKind::Sswb: return ZYDIS_MNEMONIC_VPACKSSWB;
        case VecPackKind::Usdw: return ZYDIS_MNEMONIC_VPACKUSDW;
        case VecPackKind::Uswb: return ZYDIS_MNEMONIC_VPACKUSWB;
        }
        return ZYDIS_MNEMONIC_INVALID;
    }();
    if (insn.mnemonic != expected) return false;

    if (insn.operand_count_visible != 3) return false;
    if (ops[0].type != ZYDIS_OPERAND_TYPE_REGISTER) return false;
    if (ops[1].type != ZYDIS_OPERAND_TYPE_REGISTER) return false;

    const int dst_idx = ZydisVecToIndex(ops[0].reg.value);
    const int src1_idx = ZydisVecToIndex(ops[1].reg.value);
    if (dst_idx < 0 || src1_idx < 0) return false;

    const int vec_bits = ops[0].size;
    if (vec_bits != 128 && vec_bits != 256) return false;

    // Dispatch the host pack op for the kind (operates on xmm0/xmm1
    // or ymm0/ymm1; result lands in the first operand).
    auto emit_pack_128 = [&]() {
        switch (kind) {
        case VecPackKind::Ssdw: c.vpackssdw(xmm0, xmm0, xmm1); break;
        case VecPackKind::Sswb: c.vpacksswb(xmm0, xmm0, xmm1); break;
        case VecPackKind::Usdw: c.vpackusdw(xmm0, xmm0, xmm1); break;
        case VecPackKind::Uswb: c.vpackuswb(xmm0, xmm0, xmm1); break;
        }
    };
    auto emit_pack_256 = [&]() {
        switch (kind) {
        case VecPackKind::Ssdw: c.vpackssdw(ymm0, ymm0, ymm1); break;
        case VecPackKind::Sswb: c.vpacksswb(ymm0, ymm0, ymm1); break;
        case VecPackKind::Usdw: c.vpackusdw(ymm0, ymm0, ymm1); break;
        case VecPackKind::Uswb: c.vpackuswb(ymm0, ymm0, ymm1); break;
        }
    };

    if (vec_bits == 128) {
        c.vmovdqu(xmm0, ptr[r13 + YmmChunkOffset(src1_idx, 0)]);

        if (ops[2].type == ZYDIS_OPERAND_TYPE_REGISTER) {
            const int src2_idx = ZydisVecToIndex(ops[2].reg.value);
            if (src2_idx < 0) return false;
            c.vmovdqu(xmm1, ptr[r13 + YmmChunkOffset(src2_idx, 0)]);
        } else if (ops[2].type == ZYDIS_OPERAND_TYPE_MEMORY) {
            if (!EmitEffectiveAddress(ops[2].mem, next_rip, c))
                return false;
            c.vmovdqu(xmm1, ptr[rdx]);
        } else {
            return false;
        }

        emit_pack_128();

        c.vmovdqu(ptr[r13 + YmmChunkOffset(dst_idx, 0)], xmm0);
        c.xor_(rax, rax);
        c.mov(qword[r13 + YmmChunkOffset(dst_idx, 2)], rax);
        c.mov(qword[r13 + YmmChunkOffset(dst_idx, 3)], rax);
    } else { // 256-bit
        c.vmovdqu(ymm0, ptr[r13 + YmmChunkOffset(src1_idx, 0)]);

        if (ops[2].type == ZYDIS_OPERAND_TYPE_REGISTER) {
            const int src2_idx = ZydisVecToIndex(ops[2].reg.value);
            if (src2_idx < 0) return false;
            c.vmovdqu(ymm1, ptr[r13 + YmmChunkOffset(src2_idx, 0)]);
        } else if (ops[2].type == ZYDIS_OPERAND_TYPE_MEMORY) {
            if (!EmitEffectiveAddress(ops[2].mem, next_rip, c))
                return false;
            c.vmovdqu(ymm1, ptr[rdx]);
        } else {
            return false;
        }

        emit_pack_256();
        c.vmovdqu(ptr[r13 + YmmChunkOffset(dst_idx, 0)], ymm0);
    }
    return true;
}

/// VPSHUFD dst, src, imm8 — packed dword shuffle.
/// For each 32-bit element i in 0..3:
///   dst[i] = src[(imm8 >> (i*2)) & 3]
/// The imm8 picks which of the 4 source dwords goes to each
/// destination position. 256-bit form runs this independently for
/// the low and high 128-bit lanes (per-lane selection, no cross-
/// lane shuffle). No flags affected.
///
/// Common compiler-emitted idiom for "broadcast a single dword" via
/// imm = 0x00 (all four positions get src[0]) and for swizzles
/// after gather/load sequences.
///
/// We run host VPSHUFD directly on JIT-scratch xmm0. The src and
/// dst can be the same register at the guest level; the host code
/// always writes dst from a single source so aliasing isn't an
/// issue (we load src into xmm0 before storing back to dst).
bool EmitVpshufd(const ZydisDecodedInstruction& insn, const ZydisDecodedOperand* ops,
                 u64 next_rip, Xbyak::CodeGenerator& c) {
    if (insn.mnemonic != ZYDIS_MNEMONIC_VPSHUFD) return false;
    if (ops[0].type != ZYDIS_OPERAND_TYPE_REGISTER) return false;
    // src (ops[1]) may be a register OR a memory operand; imm is ops[2].
    if (ops[1].type != ZYDIS_OPERAND_TYPE_REGISTER &&
        ops[1].type != ZYDIS_OPERAND_TYPE_MEMORY)
        return false;
    if (ops[2].type != ZYDIS_OPERAND_TYPE_IMMEDIATE) return false;

    const int dst_idx = ZydisVecToIndex(ops[0].reg.value);
    if (dst_idx < 0) return false;

    const u8 imm = static_cast<u8>(ops[2].imm.value.u & 0xFFu);
    const int vec_bits = ops[0].size;
    if (vec_bits != 128 && vec_bits != 256) return false;

    const bool src_is_mem = (ops[1].type == ZYDIS_OPERAND_TYPE_MEMORY);
    int src_idx = -1;
    if (!src_is_mem) {
        src_idx = ZydisVecToIndex(ops[1].reg.value);
        if (src_idx < 0) return false;
    }

    // Load the source into the host scratch vector. For a memory source we
    // compute the effective address first (EmitEffectiveAddress writes rdx
    // and scratches rax), then load the 128/256-bit operand from [rdx].
    if (vec_bits == 128) {
        if (src_is_mem) {
            if (!EmitEffectiveAddress(ops[1].mem, next_rip, c))
                return false;
            c.vmovdqu(xmm0, ptr[rdx]);
        } else {
            c.vmovdqu(xmm0, ptr[r13 + YmmChunkOffset(src_idx, 0)]);
        }
        c.vpshufd(xmm0, xmm0, imm);
        c.vmovdqu(ptr[r13 + YmmChunkOffset(dst_idx, 0)], xmm0);
        // VEX-128 zeroes bits 255:128 of the destination YMM.
        c.xor_(rax, rax);
        c.mov(qword[r13 + YmmChunkOffset(dst_idx, 2)], rax);
        c.mov(qword[r13 + YmmChunkOffset(dst_idx, 3)], rax);
    } else {
        if (src_is_mem) {
            if (!EmitEffectiveAddress(ops[1].mem, next_rip, c))
                return false;
            c.vmovdqu(ymm0, ptr[rdx]);
        } else {
            c.vmovdqu(ymm0, ptr[r13 + YmmChunkOffset(src_idx, 0)]);
        }
        c.vpshufd(ymm0, ymm0, imm);
        c.vmovdqu(ptr[r13 + YmmChunkOffset(dst_idx, 0)], ymm0);
    }
    return true;
}

// =============================================================================
// VPMOVZXDQ — zero-extend packed dwords to qwords.
//
//   VPMOVZXDQ xmm1, xmm2/m64    (VEX.128.66.0F38.WIG 35 /r)
//       dst.qword[0] = zext(src.dword[0]); dst.qword[1] = zext(src.dword[1]);
//       bits 255:128 of the dst YMM are zeroed (VEX).
//   VPMOVZXDQ ymm1, xmm2/m128   (VEX.256.66.0F38.WIG 35 /r)
//       dst.qword[i] = zext(src.dword[i]) for i in 0..3.
//
// The source is half the width of the destination: a 128-bit dst reads 64 bits
// (2 dwords) of source; a 256-bit dst reads 128 bits (4 dwords). No flags.
//
// First observed as a reg,reg 128-bit form (length 5) in the CUSA02394
// "WE ARE DOOMED" eboot at guest 0x8001f33ff. We run the host VPMOVZXDQ on
// JIT-scratch vectors. Source is loaded into xmm1 before writing dst, so a
// guest-level dst==src aliasing is safe. Host scratch: xmm0=result, xmm1=src.
bool EmitVpmovzxdq(const ZydisDecodedInstruction& insn, const ZydisDecodedOperand* ops,
                   u64 next_rip, Xbyak::CodeGenerator& c) {
    if (insn.mnemonic != ZYDIS_MNEMONIC_VPMOVZXDQ) return false;
    if (ops[0].type != ZYDIS_OPERAND_TYPE_REGISTER) return false;
    if (ops[1].type != ZYDIS_OPERAND_TYPE_REGISTER &&
        ops[1].type != ZYDIS_OPERAND_TYPE_MEMORY)
        return false;

    const int dst_idx = ZydisVecToIndex(ops[0].reg.value);
    if (dst_idx < 0) return false;

    const int dst_bits = ops[0].size; // 128 -> xmm dst, 256 -> ymm dst
    if (dst_bits != 128 && dst_bits != 256) return false;

    const bool src_is_mem = (ops[1].type == ZYDIS_OPERAND_TYPE_MEMORY);

    // Load the source operand into host scratch xmm1. The source width is half
    // the destination: 64 bits (m64 / low 2 dwords of an xmm) for a 128-bit
    // dst, 128 bits (m128 / full xmm) for a 256-bit dst. Loading the full low
    // 128 bits of the source register is always safe — VPMOVZXDQ only consumes
    // the low half it needs and ignores the rest.
    if (src_is_mem) {
        if (!EmitEffectiveAddress(ops[1].mem, next_rip, c))
            return false;
        if (dst_bits == 128)
            c.vmovq(xmm1, ptr[rdx]);    // 64-bit source
        else
            c.vmovdqu(xmm1, ptr[rdx]);  // 128-bit source
    } else {
        const int src_idx = ZydisVecToIndex(ops[1].reg.value);
        if (src_idx < 0) return false;
        c.vmovdqu(xmm1, ptr[r13 + YmmChunkOffset(src_idx, 0)]);
    }

    if (dst_bits == 128) {
        c.vpmovzxdq(xmm0, xmm1);
        c.vmovdqu(ptr[r13 + YmmChunkOffset(dst_idx, 0)], xmm0);
        // VEX-128 zeroes bits 255:128 of the destination YMM.
        c.xor_(rax, rax);
        c.mov(qword[r13 + YmmChunkOffset(dst_idx, 2)], rax);
        c.mov(qword[r13 + YmmChunkOffset(dst_idx, 3)], rax);
    } else {
        c.vpmovzxdq(ymm0, xmm1); // 4 dwords -> 4 qwords (full 256-bit dst)
        c.vmovdqu(ptr[r13 + YmmChunkOffset(dst_idx, 0)], ymm0);
    }
    return true;
}

/// VSHUFPS — shuffle packed single-precision floats from two sources.
///
///   VSHUFPS xmm1, xmm2, xmm3/m128, imm8   (VEX.128.0F.WIG C6 /r ib)
///
/// Within each 128-bit lane the destination dwords are selected as:
///   dst[0] = src1[(imm >> 0) & 3]
///   dst[1] = src1[(imm >> 2) & 3]
///   dst[2] = src2[(imm >> 4) & 3]
///   dst[3] = src2[(imm >> 6) & 3]
/// i.e. the LOW two result dwords come from src1, the HIGH two from src2.
/// (Contrast VPSHUFD, which shuffles four dwords from a single source.) The
/// 256-bit form applies this independently to each 128-bit lane. No flags.
///
/// We delegate the lane selection to the host VSHUFPS on JIT-scratch vectors:
/// load src1 into xmm0, src2 into xmm1, run vshufps with the same imm, store
/// the result to dst, and (VEX) zero bits 255:128 for the 128-bit form. The
/// host instruction reproduces the selection exactly, so no manual lane
/// plumbing is needed. src2 may be a register or m128.
///
/// First observed as a reg,reg 128-bit form (2-byte VEX, length 5) in the
/// CUSA02394 "WE ARE DOOMED" eboot at guest 0x8002416a0.
bool EmitVshufps(const ZydisDecodedInstruction& insn, const ZydisDecodedOperand* ops,
                 u64 next_rip, Xbyak::CodeGenerator& c) {
    if (insn.mnemonic != ZYDIS_MNEMONIC_VSHUFPS) return false;
    if (insn.operand_count_visible != 4) return false;
    if (ops[0].type != ZYDIS_OPERAND_TYPE_REGISTER) return false;
    if (ops[1].type != ZYDIS_OPERAND_TYPE_REGISTER) return false;
    if (ops[3].type != ZYDIS_OPERAND_TYPE_IMMEDIATE) return false;

    const int dst_idx = ZydisVecToIndex(ops[0].reg.value);
    const int src1_idx = ZydisVecToIndex(ops[1].reg.value);
    if (dst_idx < 0 || src1_idx < 0) return false;

    const u8 imm = static_cast<u8>(ops[3].imm.value.u & 0xFFu);
    const int vec_bits = ops[0].size;
    if (vec_bits != 128 && vec_bits != 256) return false;

    const bool is256 = (vec_bits == 256);

    // Load src1 into xmm0/ymm0.
    if (is256)
        c.vmovdqu(ymm0, ptr[r13 + YmmChunkOffset(src1_idx, 0)]);
    else
        c.vmovdqu(xmm0, ptr[r13 + YmmChunkOffset(src1_idx, 0)]);

    // Load src2 into xmm1/ymm1 (register or memory).
    if (ops[2].type == ZYDIS_OPERAND_TYPE_REGISTER) {
        const int src2_idx = ZydisVecToIndex(ops[2].reg.value);
        if (src2_idx < 0) return false;
        if (is256)
            c.vmovdqu(ymm1, ptr[r13 + YmmChunkOffset(src2_idx, 0)]);
        else
            c.vmovdqu(xmm1, ptr[r13 + YmmChunkOffset(src2_idx, 0)]);
    } else if (ops[2].type == ZYDIS_OPERAND_TYPE_MEMORY) {
        if (!EmitEffectiveAddress(ops[2].mem, next_rip, c))
            return false;
        if (is256)
            c.vmovdqu(ymm1, ptr[rdx]);
        else
            c.vmovdqu(xmm1, ptr[rdx]);
    } else {
        return false;
    }

    if (is256) {
        c.vshufps(ymm0, ymm0, ymm1, imm);
        c.vmovdqu(ptr[r13 + YmmChunkOffset(dst_idx, 0)], ymm0);
    } else {
        c.vshufps(xmm0, xmm0, xmm1, imm);
        c.vmovdqu(ptr[r13 + YmmChunkOffset(dst_idx, 0)], xmm0);
        // VEX-128: zero bits 255:128 of the destination YMM.
        c.xor_(rax, rax);
        c.mov(qword[r13 + YmmChunkOffset(dst_idx, 2)], rax);
        c.mov(qword[r13 + YmmChunkOffset(dst_idx, 3)], rax);
    }
    return true;
}

/// VUNPCKLPS / VUNPCKHPS / VUNPCKLPD / VUNPCKHPD — interleave (unpack) packed
/// floats from two sources. Within each 128-bit lane:
///
///   VUNPCKLPS dst = {src1[0], src2[0], src1[1], src2[1]}  (low dwords)
///   VUNPCKHPS dst = {src1[2], src2[2], src1[3], src2[3]}  (high dwords)
///   VUNPCKLPD dst = {src1.q0, src2.q0}                    (low qwords)
///   VUNPCKHPD dst = {src1.q1, src2.q1}                    (high qwords)
///
/// The 256-bit forms apply the same interleave independently per 128-bit lane.
/// No flags. These are the standard "transpose / zip" building blocks compilers
/// emit to assemble structures-of-arrays from interleaved data.
///
/// As with VSHUFPS, we delegate the interleave to the host instruction on
/// JIT-scratch vectors: load src1 into xmm0/ymm0, src2 into xmm1/ymm1, run the
/// matching host op, store to dst, and (VEX) zero bits 255:128 for the 128-bit
/// form. src2 may be a register or m128/m256. Parameterised over the four
/// members since they differ only in the host mnemonic.
///
/// First observed as a reg,reg 128-bit VUNPCKLPS (length 4) in the CUSA02394
/// "WE ARE DOOMED" eboot at guest 0x800241814.
enum class VecUnpackKind { Lps, Hps, Lpd, Hpd };

bool EmitVunpck(const ZydisDecodedInstruction& insn, const ZydisDecodedOperand* ops,
                u64 next_rip, Xbyak::CodeGenerator& c, VecUnpackKind kind) {
    if (insn.operand_count_visible != 3) return false;
    if (ops[0].type != ZYDIS_OPERAND_TYPE_REGISTER) return false;
    if (ops[1].type != ZYDIS_OPERAND_TYPE_REGISTER) return false;

    const int dst_idx = ZydisVecToIndex(ops[0].reg.value);
    const int src1_idx = ZydisVecToIndex(ops[1].reg.value);
    if (dst_idx < 0 || src1_idx < 0) return false;

    const int vec_bits = ops[0].size;
    if (vec_bits != 128 && vec_bits != 256) return false;
    const bool is256 = (vec_bits == 256);

    // src1 -> xmm0/ymm0.
    if (is256)
        c.vmovdqu(ymm0, ptr[r13 + YmmChunkOffset(src1_idx, 0)]);
    else
        c.vmovdqu(xmm0, ptr[r13 + YmmChunkOffset(src1_idx, 0)]);

    // src2 -> xmm1/ymm1 (register or memory).
    if (ops[2].type == ZYDIS_OPERAND_TYPE_REGISTER) {
        const int src2_idx = ZydisVecToIndex(ops[2].reg.value);
        if (src2_idx < 0) return false;
        if (is256)
            c.vmovdqu(ymm1, ptr[r13 + YmmChunkOffset(src2_idx, 0)]);
        else
            c.vmovdqu(xmm1, ptr[r13 + YmmChunkOffset(src2_idx, 0)]);
    } else if (ops[2].type == ZYDIS_OPERAND_TYPE_MEMORY) {
        if (!EmitEffectiveAddress(ops[2].mem, next_rip, c))
            return false;
        if (is256)
            c.vmovdqu(ymm1, ptr[rdx]);
        else
            c.vmovdqu(xmm1, ptr[rdx]);
    } else {
        return false;
    }

    auto run_128 = [&]() {
        switch (kind) {
        case VecUnpackKind::Lps: c.vunpcklps(xmm0, xmm0, xmm1); break;
        case VecUnpackKind::Hps: c.vunpckhps(xmm0, xmm0, xmm1); break;
        case VecUnpackKind::Lpd: c.vunpcklpd(xmm0, xmm0, xmm1); break;
        case VecUnpackKind::Hpd: c.vunpckhpd(xmm0, xmm0, xmm1); break;
        }
    };
    auto run_256 = [&]() {
        switch (kind) {
        case VecUnpackKind::Lps: c.vunpcklps(ymm0, ymm0, ymm1); break;
        case VecUnpackKind::Hps: c.vunpckhps(ymm0, ymm0, ymm1); break;
        case VecUnpackKind::Lpd: c.vunpcklpd(ymm0, ymm0, ymm1); break;
        case VecUnpackKind::Hpd: c.vunpckhpd(ymm0, ymm0, ymm1); break;
        }
    };

    if (is256) {
        run_256();
        c.vmovdqu(ptr[r13 + YmmChunkOffset(dst_idx, 0)], ymm0);
    } else {
        run_128();
        c.vmovdqu(ptr[r13 + YmmChunkOffset(dst_idx, 0)], xmm0);
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
    // Deliberate fprintf exception (the only one left in this file).
    // By the time the lifter destructor runs at program shutdown,
    // shadPS4's logging subsystem has often already been torn down, so
    // LOG_INFO would degrade to emitting the format string verbatim
    // with the `{}` placeholders un-substituted (or worse, touch a
    // half-destructed sink). fprintf to stderr works at any teardown
    // phase and needs no live logging backend. This is platform-
    // independent and unrelated to the JIT-context SEH concern.
    std::fprintf(stderr, "[lifter] %llu blocks compiled, %llu bytes emitted, %llu unsupported\n",
                 (unsigned long long)blocks_compiled_, (unsigned long long)bytes_emitted_,
                 (unsigned long long)unsupported_hits_);
    std::fflush(stderr);
}

void* Lifter::CompileBlock(u64 guest_rip) {
    // Per-block compile trace. LOG_TRACE compiles to (void(0)) in
    // release builds, so this has zero cost there and only emits in
    // _DEBUG. That matters because CompileBlock runs in JIT-dispatched
    // context (Runtime::Run -> gateway -> dispatcher trampoline ->
    // here), and on the x86/Windows host an spdlog/fmt call that
    // triggers an SEH stack walk can fault when the walker reaches the
    // JIT gateway frame (no registered .pdata/.xdata). Because
    // LOG_TRACE is gone entirely in release, the hot path never touches
    // spdlog. On the macOS/arm64 target there is no SEH walker at all,
    // so even _DEBUG tracing here is safe. The error/warning paths
    // below use LOG_WARNING/LOG_ERROR, which only fire on real problems
    // (low frequency) and carry the same release-build caveat handled
    // by the gateway-unwind work tracked separately for Windows.
    LOG_TRACE(Core, "Lifter: compiling block at guest_rip={:#x}", guest_rip);

    // Reserve a chunk of code cache for this block. We don't know
    // the final size yet; conservatively reserve the size cap and
    // commit only what we use. (For a real impl we'd use xbyak's
    // internal buffer and copy out, but the bump allocator's
    // overhead is tiny.)
    u8* code_buf = code_cache_.Allocate(BLOCK_HOST_SIZE_CAP);
    if (code_buf == nullptr) {
        LOG_WARNING(Core, "Lifter: code cache full at guest_rip={:#x}; "
                          "caller should flush and retry", guest_rip);
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
        LOG_TRACE(Core, "Lifter: decoding at {:#x}", rip);
        const auto status =
            ZydisDecoderDecodeFull(&decoder, reinterpret_cast<const void*>(rip), 15, &insn, ops);
        LOG_TRACE(Core, "Lifter: decoded at {:#x} ok={} mnemonic={}", rip,
                  ZYAN_SUCCESS(status) ? 1 : 0,
                  ZYAN_SUCCESS(status) ? ZydisMnemonicGetString(insn.mnemonic)
                                       : "(decode-failed)");
        if (!ZYAN_SUCCESS(status)) {
            LOG_ERROR(Core, "Lifter: decode FAILED at {:#x}", rip);
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

        // [RAXTRACE] R8-writer finder. The crash root cause is guest R8
        // holding 0xfffffe70 (should be sign-extended 0xfffffffffffffe70).
        // R8 is loop-invariant — set ONCE upstream. Log EVERY guest
        // instruction that WRITES R8/R8D in ANY operand position (not just
        // ops[0]) — catches pop r8, xchg, implicit writes, etc. — using
        // Zydis operand action flags.
        {
            bool writes_r8 = false;
            int wr_op = -1;
            for (int oi = 0; oi < insn.operand_count; ++oi) {
                if (ops[oi].type != ZYDIS_OPERAND_TYPE_REGISTER) continue;
                const auto rv = ops[oi].reg.value;
                if (rv != ZYDIS_REGISTER_R8 && rv != ZYDIS_REGISTER_R8D &&
                    rv != ZYDIS_REGISTER_R8W && rv != ZYDIS_REGISTER_R8B)
                    continue;
                if (ops[oi].actions & ZYDIS_OPERAND_ACTION_MASK_WRITE) {
                    writes_r8 = true; wr_op = oi; break;
                }
            }
            if (writes_r8) {
                char srcdesc[160]; int so = 0;
                for (int oi = 0; oi < insn.operand_count && so < 140; ++oi) {
                    const char* tn =
                        (ops[oi].type == ZYDIS_OPERAND_TYPE_REGISTER) ? "reg" :
                        (ops[oi].type == ZYDIS_OPERAND_TYPE_MEMORY)   ? "mem" :
                        (ops[oi].type == ZYDIS_OPERAND_TYPE_IMMEDIATE)? "imm" : "oth";
                    const char* rs = (ops[oi].type == ZYDIS_OPERAND_TYPE_REGISTER)
                        ? ZydisRegisterGetString(ops[oi].reg.value) : "";
                    so += std::snprintf(srcdesc + so, sizeof(srcdesc) - so,
                                        " op%d=%s%s%s", oi, tn, *rs ? ":" : "", rs);
                }
                LOG_ERROR(Core,
                          "[RAXTRACE] R8-WRITER @ {:#x}: {} opw={} wr_op={}{}",
                          rip, ZydisMnemonicGetString(insn.mnemonic),
                          insn.operand_width, wr_op, srcdesc);
            }
        }

        // [RAXTRACE] Guest-instruction trace for the crash loop blocks. Logs
        // the DECODED GUEST instruction (mnemonic + operand widths + index
        // register), which the host-byte dumps cannot reveal. This is what
        // distinguishes "cae's lea is faithful, R8 producer is the bug" from
        // "cae's lift is wrong" — we need to see the guest's actual operand
        // widths and whether the index is R8 (64-bit) or R8D (32-bit).
        if (guest_rip == 0x800274caeull || guest_rip == 0x800274c09ull ||
            guest_rip == 0x800274c00ull) {
            char ops_desc[256];
            int off = 0;
            for (int oi = 0; oi < insn.operand_count && off < 200; ++oi) {
                const char* tn =
                    (ops[oi].type == ZYDIS_OPERAND_TYPE_REGISTER) ? "reg" :
                    (ops[oi].type == ZYDIS_OPERAND_TYPE_MEMORY)   ? "mem" :
                    (ops[oi].type == ZYDIS_OPERAND_TYPE_IMMEDIATE)? "imm" : "oth";
                const char* rs = (ops[oi].type == ZYDIS_OPERAND_TYPE_REGISTER)
                    ? ZydisRegisterGetString(ops[oi].reg.value) : "";
                const char* idxs = (ops[oi].type == ZYDIS_OPERAND_TYPE_MEMORY &&
                                    ops[oi].mem.index != ZYDIS_REGISTER_NONE)
                    ? ZydisRegisterGetString(ops[oi].mem.index) : "";
                off += std::snprintf(ops_desc + off, sizeof(ops_desc) - off,
                                     " [%s %s idx=%s sz%u]", tn, rs, idxs,
                                     ops[oi].size);
            }
            LOG_ERROR(Core,
                      "[RAXTRACE] guest insn @ {:#x}: {} opw={}{}",
                      rip, ZydisMnemonicGetString(insn.mnemonic),
                      insn.operand_width, ops_desc);
        }

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
                handled = EmitTest(insn, ops, next_rip, c);
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
            handled = EmitInc(insn, ops, next_rip, c);
            break;
        case ZYDIS_MNEMONIC_DEC:
            handled = EmitDec(insn, ops, next_rip, c);
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

        // Rotates. Same shape as shifts; width-dispatched like them.
        case ZYDIS_MNEMONIC_ROL:
            handled = (insn.operand_width == 32)
                          ? EmitRotate32(insn, ops, c, RotateKind::Rol)
                          : EmitRotate64(insn, ops, c, RotateKind::Rol);
            break;
        case ZYDIS_MNEMONIC_ROR:
            handled = (insn.operand_width == 32)
                          ? EmitRotate32(insn, ops, c, RotateKind::Ror)
                          : EmitRotate64(insn, ops, c, RotateKind::Ror);
            break;

        // Multiplication. EmitImul dispatches by operand_count_visible.
        case ZYDIS_MNEMONIC_IMUL:
            handled = EmitImul(insn, ops, next_rip, c);
            break;
        case ZYDIS_MNEMONIC_MUL:
            handled = EmitMul(insn, ops, next_rip, c);
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
        case ZYDIS_MNEMONIC_CLD:
            handled = EmitCld(c);
            break;
        case ZYDIS_MNEMONIC_STD:
            handled = EmitStd(c);
            break;
        case ZYDIS_MNEMONIC_CMPSB:
        case ZYDIS_MNEMONIC_CMPSW:
        case ZYDIS_MNEMONIC_CMPSD:
        case ZYDIS_MNEMONIC_CMPSQ:
            handled = EmitCmps(insn, c);
            break;
        case ZYDIS_MNEMONIC_STOSB:
        case ZYDIS_MNEMONIC_STOSW:
        case ZYDIS_MNEMONIC_STOSD:
        case ZYDIS_MNEMONIC_STOSQ:
            handled = EmitStos(insn, c);
            break;
        case ZYDIS_MNEMONIC_MOVSB:
        case ZYDIS_MNEMONIC_MOVSW:
        case ZYDIS_MNEMONIC_MOVSD:
        case ZYDIS_MNEMONIC_MOVSQ:
            handled = EmitMovs(insn, rip, c);
            break;
        case ZYDIS_MNEMONIC_LODSB:
        case ZYDIS_MNEMONIC_LODSW:
        case ZYDIS_MNEMONIC_LODSD:
        case ZYDIS_MNEMONIC_LODSQ:
            handled = EmitLods(insn, c);
            break;
        case ZYDIS_MNEMONIC_SCASB:
        case ZYDIS_MNEMONIC_SCASW:
        case ZYDIS_MNEMONIC_SCASD:
        case ZYDIS_MNEMONIC_SCASQ:
            handled = EmitScas(insn, c);
            break;

        // NOP — no semantic effect, just consume the bytes.
        // Common forms: 90 (1-byte), 66 90 (2-byte),
        // 0F 1F /0 (multi-byte padding). All decode as NOP.
        case ZYDIS_MNEMONIC_NOP:
            handled = true;
            break;

        // PREFETCH* — cache-locality hints with NO architectural
        // effect. They never modify registers, flags, or memory, and
        // never fault — a prefetch of an invalid/unmapped address is
        // silently ignored by the CPU. So the correct lift is to drop
        // them entirely: we don't even compute the effective address
        // (nothing is actually accessed, so there's nothing to fault
        // on and no value to load). Emitting host prefetches would be
        // pointless — the guest's locality hints don't map meaningfully
        // onto our JIT's host memory layout. PREFETCHNTA first seen at
        // libc 0x80817880b (a non-temporal streaming-read hint, likely
        // a memcpy/scan fast path). PREFETCHT0/T1/T2 and PREFETCHW
        // share the same no-op treatment.
        case ZYDIS_MNEMONIC_PREFETCHNTA:
        case ZYDIS_MNEMONIC_PREFETCHT0:
        case ZYDIS_MNEMONIC_PREFETCHT1:
        case ZYDIS_MNEMONIC_PREFETCHT2:
        case ZYDIS_MNEMONIC_PREFETCHW:
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
            handled = EmitPush(insn, ops, next_rip, c);
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
        case ZYDIS_MNEMONIC_VMOVNTDQA:
            handled = EmitVmovups(insn, ops, next_rip, c);
            break;
        case ZYDIS_MNEMONIC_VMOVNTDQ:
            handled = EmitVmovntdq(insn, ops, next_rip, c);
            break;
        case ZYDIS_MNEMONIC_SFENCE:
            handled = EmitSfence(insn, c);
            break;
        case ZYDIS_MNEMONIC_MFENCE:
        case ZYDIS_MNEMONIC_LFENCE:
            handled = EmitMfence(insn, c);
            break;
        case ZYDIS_MNEMONIC_VXORPS:
        case ZYDIS_MNEMONIC_VPXOR:
            handled = EmitVecBitOp(insn, ops, next_rip, c, VecBitOpKind::Xor);
            break;
        case ZYDIS_MNEMONIC_VANDPS:
        case ZYDIS_MNEMONIC_VPAND:
            handled = EmitVecBitOp(insn, ops, next_rip, c, VecBitOpKind::And);
            break;
        case ZYDIS_MNEMONIC_VORPS:
        case ZYDIS_MNEMONIC_VPOR:
            handled = EmitVecBitOp(insn, ops, next_rip, c, VecBitOpKind::Or);
            break;
        case ZYDIS_MNEMONIC_VPANDN:
            handled = EmitVecBitOp(insn, ops, next_rip, c, VecBitOpKind::Andn);
            break;
        case ZYDIS_MNEMONIC_VMOVD:
            handled = EmitVmovd(insn, ops, c);
            break;
        case ZYDIS_MNEMONIC_VPEXTRD:
            handled = EmitVpextrd(insn, ops, next_rip, c);
            break;
        case ZYDIS_MNEMONIC_VPEXTRQ:
            handled = EmitVpextrq(insn, ops, next_rip, c);
            break;
        case ZYDIS_MNEMONIC_VEXTRACTPS:
            handled = EmitVextractps(insn, ops, next_rip, c);
            break;
        case ZYDIS_MNEMONIC_VPINSRD:
            handled = EmitVpinsrd(insn, ops, next_rip, c);
            break;
        case ZYDIS_MNEMONIC_VINSERTPS:
            handled = EmitVinsertps(insn, ops, next_rip, c);
            break;
        case ZYDIS_MNEMONIC_VMOVQ:
            handled = EmitVmovq(insn, ops, next_rip, c);
            break;
        case ZYDIS_MNEMONIC_VMOVSS:
            handled = EmitVmovss(insn, ops, next_rip, c);
            break;
        case ZYDIS_MNEMONIC_VMOVSD:
            handled = EmitVmovsd(insn, ops, next_rip, c);
            break;
        case ZYDIS_MNEMONIC_VCVTSI2SS:
            handled = EmitVcvtsi2ss(insn, ops, next_rip, c);
            break;
        case ZYDIS_MNEMONIC_VCVTSI2SD:
            handled = EmitVcvtsi2sd(insn, ops, next_rip, c);
            break;
        case ZYDIS_MNEMONIC_VMULSS:
            handled = EmitVmulss(insn, ops, next_rip, c);
            break;
        case ZYDIS_MNEMONIC_VDIVSS:
            handled = EmitVdivss(insn, ops, next_rip, c);
            break;
        case ZYDIS_MNEMONIC_VADDSS:
            handled = EmitVaddss(insn, ops, next_rip, c);
            break;
        case ZYDIS_MNEMONIC_VSUBSS:
            handled = EmitVsubss(insn, ops, next_rip, c);
            break;
        case ZYDIS_MNEMONIC_VMULSD:
            handled = EmitVmulsd(insn, ops, next_rip, c);
            break;
        case ZYDIS_MNEMONIC_VADDSD:
            handled = EmitVaddsd(insn, ops, next_rip, c);
            break;
        case ZYDIS_MNEMONIC_VSUBSD:
            handled = EmitVsubsd(insn, ops, next_rip, c);
            break;
        case ZYDIS_MNEMONIC_VDIVSD:
            handled = EmitVdivsd(insn, ops, next_rip, c);
            break;
        case ZYDIS_MNEMONIC_VSQRTSD:
            handled = EmitVsqrtsd(insn, ops, next_rip, c);
            break;
        case ZYDIS_MNEMONIC_VSQRTSS:
            handled = EmitVsqrtss(insn, ops, next_rip, c);
            break;
        case ZYDIS_MNEMONIC_VMINSS:
            handled = EmitVminss(insn, ops, next_rip, c);
            break;
        case ZYDIS_MNEMONIC_VMAXSS:
            handled = EmitVmaxss(insn, ops, next_rip, c);
            break;
        case ZYDIS_MNEMONIC_VMINSD:
            handled = EmitVminsd(insn, ops, next_rip, c);
            break;
        case ZYDIS_MNEMONIC_VMAXSD:
            handled = EmitVmaxsd(insn, ops, next_rip, c);
            break;
        case ZYDIS_MNEMONIC_VBLENDPS:
            handled = EmitVecBlendImm(insn, ops, next_rip, c, VecBlendImm::Ps);
            break;
        case ZYDIS_MNEMONIC_VPBLENDW:
            handled = EmitVecBlendImm(insn, ops, next_rip, c, VecBlendImm::W);
            break;
        case ZYDIS_MNEMONIC_VBLENDVPS:
            handled = EmitVblendvps(insn, ops, next_rip, c);
            break;
        case ZYDIS_MNEMONIC_VINSERTF128:
            handled = EmitVinsertf128(insn, ops, next_rip, c);
            break;
        case ZYDIS_MNEMONIC_VEXTRACTF128:
            handled = EmitVextractf128(insn, ops, next_rip, c);
            break;
        case ZYDIS_MNEMONIC_VUCOMISS:
            handled = EmitVucomiss(insn, ops, next_rip, c);
            break;
        case ZYDIS_MNEMONIC_VUCOMISD:
            handled = EmitVucomisd(insn, ops, next_rip, c);
            break;
        case ZYDIS_MNEMONIC_VROUNDSD:
            handled = EmitVroundsd(insn, ops, next_rip, c);
            break;
        case ZYDIS_MNEMONIC_VCVTSS2SD:
            handled = EmitVcvtss2sd(insn, ops, next_rip, c);
            break;
        case ZYDIS_MNEMONIC_VCVTSD2SS:
            handled = EmitVcvtsd2ss(insn, ops, next_rip, c);
            break;
        case ZYDIS_MNEMONIC_VCVTTSS2SI:
            handled = EmitVcvttss2si(insn, ops, next_rip, c);
            break;
        case ZYDIS_MNEMONIC_VCVTTSD2SI:
            handled = EmitVcvttsd2si(insn, ops, next_rip, c);
            break;
        case ZYDIS_MNEMONIC_VPSHUFB:
            handled = EmitVpshufb(insn, ops, next_rip, c);
            break;
        case ZYDIS_MNEMONIC_VPTEST:
            handled = EmitVptest(insn, ops, c);
            break;
        case ZYDIS_MNEMONIC_VPCMPISTRI:
            handled = EmitVpcmpistri(insn, ops, c);
            break;
        case ZYDIS_MNEMONIC_VPCMPISTRM:
            handled = EmitVpcmpistrm(insn, ops, c);
            break;
        case ZYDIS_MNEMONIC_VPHADDD:
            handled = EmitVphaddd(insn, ops, c);
            break;
        case ZYDIS_MNEMONIC_VPUNPCKLQDQ:
            handled = EmitVpunpckDQdq(insn, ops, next_rip, c, VecPunpckKind::LowQ);
            break;
        case ZYDIS_MNEMONIC_VPUNPCKHQDQ:
            handled = EmitVpunpckDQdq(insn, ops, next_rip, c, VecPunpckKind::HighQ);
            break;
        case ZYDIS_MNEMONIC_VPUNPCKLDQ:
            handled = EmitVpunpckDQdq(insn, ops, next_rip, c, VecPunpckKind::LowD);
            break;
        case ZYDIS_MNEMONIC_VPUNPCKHDQ:
            handled = EmitVpunpckDQdq(insn, ops, next_rip, c, VecPunpckKind::HighD);
            break;
        case ZYDIS_MNEMONIC_VPUNPCKLWD:
            handled = EmitVpunpckDQdq(insn, ops, next_rip, c, VecPunpckKind::LowW);
            break;
        case ZYDIS_MNEMONIC_VPUNPCKHWD:
            handled = EmitVpunpckDQdq(insn, ops, next_rip, c, VecPunpckKind::HighW);
            break;
        case ZYDIS_MNEMONIC_VPACKUSDW:
            handled = EmitVecPack(insn, ops, next_rip, c, VecPackKind::Usdw);
            break;
        case ZYDIS_MNEMONIC_VPACKSSDW:
            handled = EmitVecPack(insn, ops, next_rip, c, VecPackKind::Ssdw);
            break;
        case ZYDIS_MNEMONIC_VPACKSSWB:
            handled = EmitVecPack(insn, ops, next_rip, c, VecPackKind::Sswb);
            break;
        case ZYDIS_MNEMONIC_VPACKUSWB:
            handled = EmitVecPack(insn, ops, next_rip, c, VecPackKind::Uswb);
            break;
        case ZYDIS_MNEMONIC_VPSHUFD:
            handled = EmitVpshufd(insn, ops, next_rip, c);
            break;
        case ZYDIS_MNEMONIC_VSHUFPS:
            handled = EmitVshufps(insn, ops, next_rip, c);
            break;
        case ZYDIS_MNEMONIC_VUNPCKLPS:
            handled = EmitVunpck(insn, ops, next_rip, c, VecUnpackKind::Lps);
            break;
        case ZYDIS_MNEMONIC_VUNPCKHPS:
            handled = EmitVunpck(insn, ops, next_rip, c, VecUnpackKind::Hps);
            break;
        case ZYDIS_MNEMONIC_VUNPCKLPD:
            handled = EmitVunpck(insn, ops, next_rip, c, VecUnpackKind::Lpd);
            break;
        case ZYDIS_MNEMONIC_VUNPCKHPD:
            handled = EmitVunpck(insn, ops, next_rip, c, VecUnpackKind::Hpd);
            break;
        case ZYDIS_MNEMONIC_VPMOVZXDQ:
            handled = EmitVpmovzxdq(insn, ops, next_rip, c);
            break;
        case ZYDIS_MNEMONIC_XADD:
            handled = EmitXadd(insn, ops, next_rip, c);
            break;
        case ZYDIS_MNEMONIC_VPCMPEQB:
            handled = EmitVecIntCmp(insn, ops, next_rip, c, VecIntCmp::EqB);
            break;
        case ZYDIS_MNEMONIC_VPCMPEQW:
            handled = EmitVecIntCmp(insn, ops, next_rip, c, VecIntCmp::EqW);
            break;
        case ZYDIS_MNEMONIC_VPCMPEQD:
            handled = EmitVecIntCmp(insn, ops, next_rip, c, VecIntCmp::EqD);
            break;
        case ZYDIS_MNEMONIC_VPCMPEQQ:
            handled = EmitVecIntCmp(insn, ops, next_rip, c, VecIntCmp::EqQ);
            break;
        case ZYDIS_MNEMONIC_VPCMPGTB:
            handled = EmitVecIntCmp(insn, ops, next_rip, c, VecIntCmp::GtB);
            break;
        case ZYDIS_MNEMONIC_VPCMPGTW:
            handled = EmitVecIntCmp(insn, ops, next_rip, c, VecIntCmp::GtW);
            break;
        case ZYDIS_MNEMONIC_VPCMPGTD:
            handled = EmitVecIntCmp(insn, ops, next_rip, c, VecIntCmp::GtD);
            break;
        case ZYDIS_MNEMONIC_VPCMPGTQ:
            handled = EmitVecIntCmp(insn, ops, next_rip, c, VecIntCmp::GtQ);
            break;
        case ZYDIS_MNEMONIC_VPADDB:
            handled = EmitVecIntArith(insn, ops, next_rip, c, VecIntArith::AddB);
            break;
        case ZYDIS_MNEMONIC_VPADDW:
            handled = EmitVecIntArith(insn, ops, next_rip, c, VecIntArith::AddW);
            break;
        case ZYDIS_MNEMONIC_VPADDD:
            handled = EmitVecIntArith(insn, ops, next_rip, c, VecIntArith::AddD);
            break;
        case ZYDIS_MNEMONIC_VPADDQ:
            handled = EmitVecIntArith(insn, ops, next_rip, c, VecIntArith::AddQ);
            break;
        case ZYDIS_MNEMONIC_VPSUBB:
            handled = EmitVecIntArith(insn, ops, next_rip, c, VecIntArith::SubB);
            break;
        case ZYDIS_MNEMONIC_VPSUBW:
            handled = EmitVecIntArith(insn, ops, next_rip, c, VecIntArith::SubW);
            break;
        case ZYDIS_MNEMONIC_VPSUBD:
            handled = EmitVecIntArith(insn, ops, next_rip, c, VecIntArith::SubD);
            break;
        case ZYDIS_MNEMONIC_VPSUBQ:
            handled = EmitVecIntArith(insn, ops, next_rip, c, VecIntArith::SubQ);
            break;
        case ZYDIS_MNEMONIC_VPMULLW:
            handled = EmitVecIntArith(insn, ops, next_rip, c, VecIntArith::MulW);
            break;
        case ZYDIS_MNEMONIC_VPMULLD:
            handled = EmitVecIntArith(insn, ops, next_rip, c, VecIntArith::MulD);
            break;
        case ZYDIS_MNEMONIC_VPSRAW:
            handled = EmitVecShiftImm(insn, ops, c, VecShiftImm::SraW);
            break;
        case ZYDIS_MNEMONIC_VPSRAD:
            handled = EmitVecShiftImm(insn, ops, c, VecShiftImm::SraD);
            break;
        case ZYDIS_MNEMONIC_VPSRLW:
            handled = EmitVecShiftImm(insn, ops, c, VecShiftImm::SrlW);
            break;
        case ZYDIS_MNEMONIC_VPSRLD:
            handled = EmitVecShiftImm(insn, ops, c, VecShiftImm::SrlD);
            break;
        case ZYDIS_MNEMONIC_VPSRLQ:
            handled = EmitVecShiftImm(insn, ops, c, VecShiftImm::SrlQ);
            break;
        case ZYDIS_MNEMONIC_VPSLLW:
            handled = EmitVecShiftImm(insn, ops, c, VecShiftImm::SllW);
            break;
        case ZYDIS_MNEMONIC_VPSLLD:
            handled = EmitVecShiftImm(insn, ops, c, VecShiftImm::SllD);
            break;
        case ZYDIS_MNEMONIC_VPSLLQ:
            handled = EmitVecShiftImm(insn, ops, c, VecShiftImm::SllQ);
            break;
        case ZYDIS_MNEMONIC_VPSLLDQ:
            handled = EmitVecByteShift(insn, ops, c, VecByteShiftKind::Slldq);
            break;
        case ZYDIS_MNEMONIC_VPSRLDQ:
            handled = EmitVecByteShift(insn, ops, c, VecByteShiftKind::Srldq);
            break;
        case ZYDIS_MNEMONIC_VMULPS:
            handled = EmitVecFpArith(insn, ops, next_rip, c, VecFpKind::Mul, VecFpPrec::Single);
            break;
        case ZYDIS_MNEMONIC_VDIVPS:
            handled = EmitVecFpArith(insn, ops, next_rip, c, VecFpKind::Div, VecFpPrec::Single);
            break;
        case ZYDIS_MNEMONIC_VADDPS:
            handled = EmitVecFpArith(insn, ops, next_rip, c, VecFpKind::Add, VecFpPrec::Single);
            break;
        case ZYDIS_MNEMONIC_VSUBPS:
            handled = EmitVecFpArith(insn, ops, next_rip, c, VecFpKind::Sub, VecFpPrec::Single);
            break;
        case ZYDIS_MNEMONIC_VMULPD:
            handled = EmitVecFpArith(insn, ops, next_rip, c, VecFpKind::Mul, VecFpPrec::Double);
            break;
        case ZYDIS_MNEMONIC_VDIVPD:
            handled = EmitVecFpArith(insn, ops, next_rip, c, VecFpKind::Div, VecFpPrec::Double);
            break;
        case ZYDIS_MNEMONIC_VADDPD:
            handled = EmitVecFpArith(insn, ops, next_rip, c, VecFpKind::Add, VecFpPrec::Double);
            break;
        case ZYDIS_MNEMONIC_VSUBPD:
            handled = EmitVecFpArith(insn, ops, next_rip, c, VecFpKind::Sub, VecFpPrec::Double);
            break;
        case ZYDIS_MNEMONIC_VCMPPS:
            handled = EmitVcmpps(insn, ops, next_rip, c);
            break;
        case ZYDIS_MNEMONIC_VCMPSS:
            handled = EmitVcmpss(insn, ops, next_rip, c);
            break;
        case ZYDIS_MNEMONIC_VMOVLHPS:
            handled = EmitVmovLhHlps(insn, ops, c, VecMovHalfKind::LhPs);
            break;
        case ZYDIS_MNEMONIC_VMOVHLPS:
            handled = EmitVmovLhHlps(insn, ops, c, VecMovHalfKind::HlPs);
            break;
        case ZYDIS_MNEMONIC_FNSTCW:
            handled = EmitFnstcw(insn, ops, next_rip, c);
            break;
        case ZYDIS_MNEMONIC_FLDCW:
            handled = EmitFldcw(insn, ops, next_rip, c);
            break;
        case ZYDIS_MNEMONIC_STMXCSR:
            handled = EmitStmxcsr(insn, ops, next_rip, c);
            break;
        case ZYDIS_MNEMONIC_LDMXCSR:
            handled = EmitLdmxcsr(insn, ops, next_rip, c);
            break;
        case ZYDIS_MNEMONIC_VBROADCASTSS:
            handled = EmitVbroadcastss(insn, ops, next_rip, c);
            break;
        case ZYDIS_MNEMONIC_VCVTDQ2PS:
            handled = EmitVecFpCvt(insn, ops, next_rip, c, VecFpCvt::Dq2Ps);
            break;
        case ZYDIS_MNEMONIC_VCVTPS2DQ:
            handled = EmitVecFpCvt(insn, ops, next_rip, c, VecFpCvt::Ps2Dq);
            break;
        case ZYDIS_MNEMONIC_VCVTTPS2DQ:
            handled = EmitVecFpCvt(insn, ops, next_rip, c, VecFpCvt::TPs2Dq);
            break;
        case ZYDIS_MNEMONIC_ANDN:
            handled = EmitAndn(insn, ops, next_rip, c);
            break;
        case ZYDIS_MNEMONIC_BEXTR:
            handled = EmitBextr(insn, ops, next_rip, c);
            break;
        case ZYDIS_MNEMONIC_BLSI:
            handled = EmitBlsFamily(insn, ops, next_rip, c, BlsKind::Blsi);
            break;
        case ZYDIS_MNEMONIC_BLSR:
            handled = EmitBlsFamily(insn, ops, next_rip, c, BlsKind::Blsr);
            break;
        case ZYDIS_MNEMONIC_BLSMSK:
            handled = EmitBlsFamily(insn, ops, next_rip, c, BlsKind::Blsmsk);
            break;
        case ZYDIS_MNEMONIC_POPCNT:
            handled = EmitPopcnt(insn, ops, next_rip, c);
            break;
        case ZYDIS_MNEMONIC_LZCNT:
            handled = EmitLzcnt(insn, ops, next_rip, c);
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
            LOG_ERROR(Core,
                      "Lifter: unsupported insn at {:#x} (mnemonic={}, "
                      "width={}, length={}, ops={},{})",
                      rip, ZydisMnemonicGetString(insn.mnemonic),
                      static_cast<unsigned>(insn.operand_width),
                      static_cast<unsigned>(insn.length),
                      op_type_name(ops[0].type), op_type_name(ops[1].type));
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

        // Host-size guard. Without this, a runaway vector-heavy block
        // can pile up emitted code until xbyak throws
        // ERR_CODE_IS_TOO_BIG mid-emission, killing the host process
        // with an exception that has no clean recovery point. Instead,
        // when we're within MARGIN bytes of the cap, force a clean
        // fallthrough exit at this guest RIP. The dispatcher will
        // re-enter for the next compile, splitting the long block
        // into two shorter blocks.
        //
        // MARGIN must cover the post-loop fallthrough emit (mov-rip,
        // mov-exit-reason, jmp r14 = ~30 bytes) with comfortable
        // headroom for any single big-emitter overshoot. 256 is
        // generous; the largest single-instruction emit observed is
        // around 120 bytes.
        constexpr u64 HOST_SIZE_MARGIN = 256;
        if (c.getSize() + HOST_SIZE_MARGIN >= BLOCK_HOST_SIZE_CAP) {
            break;
        }
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

    LOG_TRACE(Core,
              "Lifter: compiled block {:#x} -> {} ({} guest bytes -> {} host bytes)",
              guest_rip, static_cast<void*>(code_buf), rip - guest_rip, emitted);

    return code_buf;
}

} // namespace Core::Runtime
