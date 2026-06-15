// SPDX-FileCopyrightText: Copyright 2026 shadPS4 Emulator Project
// SPDX-License-Identifier: GPL-2.0-or-later
//
// Copy-verbatim gate for the x86_64->x86_64 "block-copy" backend.
//
// Same-ISA translation lets us memcpy a guest instruction's bytes straight into
// the code cache whenever it touches none of the runtime-reserved machinery.
// This predicate is the ENTIRE correctness surface of that shortcut: anything
// it wrongly accepts is silent guest corruption, so it is deliberately a
// default-REJECT design -- an instruction is copyable only if every check below
// affirmatively clears it.
//
// Reserved host registers (and, by the same names, the four virtualized guest
// registers that live in [r13_state+off] instead of a host reg):
//   RSP  -> host stack is the runtime's internal stack; guest RSP virtualized.
//   R13  -> pointer to the virtualized-register storage (GuestState spill area).
//   R14, R15 -> scratch for the mangler's fixups.
//
// An instruction is copy-verbatim iff ALL hold:
//   * it decodes cleanly;
//   * no operand (explicit, implicit, or SIB base/index) names RSP/R13/R14/R15
//     -- this single scan also catches the implicit-RSP stack ops (push/pop/
//     call/ret/leave/enter/pushfq/popfq), since Zydis lists RSP and [RSP] as
//     hidden operands of those;
//   * no memory operand is RIP-relative (host code RIP != guest RIP);
//   * no memory operand carries an FS/GS segment override (segment base differs);
//   * it is not a control transfer (call/ret/branch/int/syscall) -- those are
//     block terminators and, for rel-immediate branches, carry no reserved-reg
//     operand for the scan to catch;
//   * it is not environment-sensing / privileged (cpuid, rdtsc(p), rd/wr fs/gs
//     base, xgetbv/xsetbv, swapgs, rd/wrmsr, hlt, int/iret, sysenter/sysexit).
//
// Everything else -- ordinary ALU, data movement, SSE/AVX, partial-register and
// high-byte writes, absolute-addressed and rep-string memory ops over guest
// pointers -- copies byte-for-byte and runs correctly because guest VA == host
// VA (identity mapping) and the same-named host GPR holds the guest value.

#pragma once

#include <Zydis/Zydis.h>

namespace Core::Runtime::copyjit {

// Why an instruction was rejected (diagnostics / harness assertions). kAccept
// means copy the bytes verbatim.
enum class CopyVerdict {
    kAccept,
    kDecodeError,
    kReservedReg,     // operand / SIB base|index names RSP/R13/R14/R15
    kRipRelative,     // memory operand is RIP-relative
    kSegmentFsGs,     // memory operand uses FS/GS override
    kControlFlow,     // call/ret/branch/int/syscall terminator
    kPrivileged,      // environment-sensing or privileged
};

namespace detail {

inline bool IsReservedEnclosing(ZydisRegister reg) {
    if (reg == ZYDIS_REGISTER_NONE) return false;
    const ZydisRegister enc =
        ZydisRegisterGetLargestEnclosing(ZYDIS_MACHINE_MODE_LONG_64, reg);
    return enc == ZYDIS_REGISTER_RSP || enc == ZYDIS_REGISTER_R13 ||
           enc == ZYDIS_REGISTER_R14 || enc == ZYDIS_REGISTER_R15;
}

// Control-transfer categories: terminators, never copied mid-block.
inline bool IsControlFlowCategory(ZydisInstructionCategory cat) {
    switch (cat) {
        case ZYDIS_CATEGORY_CALL:
        case ZYDIS_CATEGORY_RET:
        case ZYDIS_CATEGORY_COND_BR:
        case ZYDIS_CATEGORY_UNCOND_BR:
        case ZYDIS_CATEGORY_INTERRUPT:
        case ZYDIS_CATEGORY_SYSCALL:
        case ZYDIS_CATEGORY_SYSRET:
        case ZYDIS_CATEGORY_SYSTEM:  // sysenter/sysexit, swapgs, hlt, etc.
            return true;
        default:
            return false;
    }
}

// Environment-sensing / privileged mnemonics whose verbatim copy would diverge
// (host vs guest CPUID/TSC/segment-base/MSR/control-register state) or fault.
inline bool IsPrivilegedOrSensing(ZydisMnemonic m) {
    switch (m) {
        case ZYDIS_MNEMONIC_CPUID:
        case ZYDIS_MNEMONIC_RDTSC:
        case ZYDIS_MNEMONIC_RDTSCP:
        case ZYDIS_MNEMONIC_RDPMC:
        case ZYDIS_MNEMONIC_RDFSBASE:
        case ZYDIS_MNEMONIC_WRFSBASE:
        case ZYDIS_MNEMONIC_RDGSBASE:
        case ZYDIS_MNEMONIC_WRGSBASE:
        case ZYDIS_MNEMONIC_XGETBV:
        case ZYDIS_MNEMONIC_XSETBV:
        case ZYDIS_MNEMONIC_SWAPGS:
        case ZYDIS_MNEMONIC_RDMSR:
        case ZYDIS_MNEMONIC_WRMSR:
        case ZYDIS_MNEMONIC_HLT:
        case ZYDIS_MNEMONIC_INT:
        case ZYDIS_MNEMONIC_INT3:
        case ZYDIS_MNEMONIC_INTO:
        case ZYDIS_MNEMONIC_IRET:
        case ZYDIS_MNEMONIC_IRETD:
        case ZYDIS_MNEMONIC_IRETQ:
        case ZYDIS_MNEMONIC_SYSENTER:
        case ZYDIS_MNEMONIC_SYSEXIT:
        case ZYDIS_MNEMONIC_SYSCALL:
        case ZYDIS_MNEMONIC_SYSRET:
            return true;
        default:
            return false;
    }
}

}  // namespace detail

// Classify an ALREADY-DECODED instruction. CompileBlock decodes once (it needs
// the decoded form for the terminator anyway) and calls this directly.
inline CopyVerdict ClassifyDecoded(const ZydisDecodedInstruction& insn,
                                   const ZydisDecodedOperand* operands,
                                   ZyanU8* out_len = nullptr) {
    // Control flow first: rel-immediate branches carry no reserved-reg operand,
    // so the operand scan below would miss them.
    if (detail::IsControlFlowCategory(insn.meta.category)) {
        return CopyVerdict::kControlFlow;
    }
    if (detail::IsPrivilegedOrSensing(insn.mnemonic)) {
        return CopyVerdict::kPrivileged;
    }

    // Scan EVERY operand (operand_count includes implicit/hidden operands, which
    // is how the implicit-RSP stack ops get caught).
    for (ZyanU8 i = 0; i < insn.operand_count; ++i) {
        const ZydisDecodedOperand& op = operands[i];
        if (op.type == ZYDIS_OPERAND_TYPE_REGISTER) {
            if (detail::IsReservedEnclosing(op.reg.value)) {
                return CopyVerdict::kReservedReg;
            }
        } else if (op.type == ZYDIS_OPERAND_TYPE_MEMORY) {
            // RIP-relative: base is the instruction pointer.
            if (op.mem.base == ZYDIS_REGISTER_RIP ||
                op.mem.base == ZYDIS_REGISTER_EIP) {
                return CopyVerdict::kRipRelative;
            }
            if (detail::IsReservedEnclosing(op.mem.base) ||
                detail::IsReservedEnclosing(op.mem.index)) {
                return CopyVerdict::kReservedReg;
            }
            if (op.mem.segment == ZYDIS_REGISTER_FS ||
                op.mem.segment == ZYDIS_REGISTER_GS) {
                return CopyVerdict::kSegmentFsGs;
            }
        }
    }

    if (out_len) *out_len = insn.length;
    return CopyVerdict::kAccept;
}

// Decode-and-classify. `data`/`length` point at one guest instruction.
// On kAccept, `*out_len` (if non-null) receives the instruction length so the
// caller can memcpy exactly that many bytes and advance.
inline CopyVerdict ClassifyForCopy(const ZydisDecoder& decoder,
                                   const void* data, ZyanUSize length,
                                   ZyanU8* out_len = nullptr) {
    ZydisDecodedInstruction insn;
    ZydisDecodedOperand operands[ZYDIS_MAX_OPERAND_COUNT];
    if (!ZYAN_SUCCESS(ZydisDecoderDecodeFull(&decoder, data, length, &insn,
                                             operands))) {
        return CopyVerdict::kDecodeError;
    }
    return ClassifyDecoded(insn, operands, out_len);
}

inline const char* VerdictName(CopyVerdict v) {
    switch (v) {
        case CopyVerdict::kAccept:      return "ACCEPT";
        case CopyVerdict::kDecodeError: return "decode-error";
        case CopyVerdict::kReservedReg: return "reserved-reg";
        case CopyVerdict::kRipRelative: return "rip-relative";
        case CopyVerdict::kSegmentFsGs: return "fs/gs-segment";
        case CopyVerdict::kControlFlow: return "control-flow";
        case CopyVerdict::kPrivileged:  return "privileged";
    }
    return "?";
}

}  // namespace Core::Runtime::copyjit
