// SPDX-FileCopyrightText: Copyright 2026 shadPS4 Emulator Project
// SPDX-License-Identifier: GPL-2.0-or-later

#include "core/cpu_runtime/lifter/lifter.h"

#include <cstddef>
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

/// Map a Zydis GPR enum to a guest-state GPR index 0..15.
/// Returns -1 for non-GPR or unsupported registers.
int ZydisGprToIndex(ZydisRegister r) {
    if (r >= ZYDIS_REGISTER_RAX && r <= ZYDIS_REGISTER_R15) {
        return r - ZYDIS_REGISTER_RAX;
    }
    // 32-bit variants (EAX..R15D). Same index space.
    if (r >= ZYDIS_REGISTER_EAX && r <= ZYDIS_REGISTER_R15D) {
        return r - ZYDIS_REGISTER_EAX;
    }
    // 16- and 8-bit variants are valid x86 but we don't handle
    // them in the initial subset. Caller falls through to
    // EmitUnsupported.
    return -1;
}

/// Byte offset within GuestState for the n-th GPR.
constexpr u32 GprOffset(int idx) {
    return Offsets::Gpr + static_cast<u32>(idx) * 8;
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
//   - rax/rcx are scratch. JIT code is free to clobber them between
//     instruction boundaries. (More sophisticated allocation comes
//     later.)
//   - r13 is the GuestState pointer (set by the gateway, never
//     modified by JIT).
//   - r12, r14, r15 are reserved (dispatcher, exit stub, future use).

using namespace Xbyak::util;

/// MOV r64, imm64 / MOV r64, r64 / MOV r64, imm32 (sign-extended).
bool EmitMov(const ZydisDecodedInstruction& insn,
             const ZydisDecodedOperand* ops,
             Xbyak::CodeGenerator& c) {
    const auto& dst = ops[0];
    const auto& src = ops[1];

    if (dst.type != ZYDIS_OPERAND_TYPE_REGISTER) {
        return false; // memory destinations not in initial subset
    }
    const int dst_idx = ZydisGprToIndex(dst.reg.value);
    if (dst_idx < 0) {
        return false;
    }

    if (src.type == ZYDIS_OPERAND_TYPE_REGISTER) {
        const int src_idx = ZydisGprToIndex(src.reg.value);
        if (src_idx < 0) return false;
        c.mov(rax, qword[r13 + GprOffset(src_idx)]);
        c.mov(qword[r13 + GprOffset(dst_idx)], rax);
        return true;
    }

    if (src.type == ZYDIS_OPERAND_TYPE_IMMEDIATE) {
        // Zydis gives the immediate value (already sign-extended to
        // 64 bits if applicable). The instruction width tells us
        // the semantics; for our initial subset we only handle the
        // 64-bit-destination case.
        if (insn.operand_width != 64) {
            return false;
        }
        c.mov(rax, src.imm.value.s);
        c.mov(qword[r13 + GprOffset(dst_idx)], rax);
        return true;
    }

    return false;
}

/// ADD r64, r64 / ADD r64, imm.
/// Flag side-effects deferred — the lazy-flag scheme records the
/// op and operands; flag consumers materialize on demand. For the
/// initial slice we set the side-band fields but don't yet have
/// flag consumers wired up, so the values are written and unused.
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

        // Load inputs into scratch. rax = original dst, rcx = src.
        // These are also the lazy-flag inputs.
        c.mov(rax, qword[r13 + GprOffset(dst_idx)]);
        c.mov(rcx, qword[r13 + GprOffset(src_idx)]);

        // Capture lhs/rhs into side-band BEFORE the add, so we don't
        // need to re-derive them later.
        c.mov(qword[r13 + Offsets::FlagLhs], rax);
        c.mov(qword[r13 + Offsets::FlagRhs], rcx);

        // Do the add. rax becomes the result.
        c.add(rax, rcx);

        // Store the new dst value AND the lazy-flag result.
        c.mov(qword[r13 + GprOffset(dst_idx)], rax);
        c.mov(qword[r13 + Offsets::FlagResult], rax);

        // Tag the op.
        constexpr u32 FLAG_OP_ADD64 = 1;
        c.mov(dword[r13 + Offsets::FlagOp], FLAG_OP_ADD64);
        return true;
    }

    if (src.type == ZYDIS_OPERAND_TYPE_IMMEDIATE) {
        c.mov(rax, qword[r13 + GprOffset(dst_idx)]);
        const auto imm = src.imm.value.s;
        // ADD r64, imm32 with sign-extension. If the immediate
        // fits in s32 we use the short form; otherwise mov-and-add.
        if (imm >= INT32_MIN && imm <= INT32_MAX) {
            c.add(rax, static_cast<int>(imm));
        } else {
            c.mov(rcx, imm);
            c.add(rax, rcx);
        }
        c.mov(qword[r13 + GprOffset(dst_idx)], rax);
        // Flag side-band omitted for the imm form in this slice.
        return true;
    }

    return false;
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

} // namespace

// ============================================================================
// Lifter class — block-compile loop and dispatch
// ============================================================================

Lifter::Lifter(CodeCache& code_cache) : code_cache_(code_cache) {
    LOG_INFO(Core, "Lifter (x86 host) initialized");
}

Lifter::~Lifter() {
    LOG_INFO(Core, "Lifter: {} blocks compiled, {} bytes emitted, {} unsupported",
             blocks_compiled_, bytes_emitted_, unsupported_hits_);
}

void* Lifter::CompileBlock(u64 guest_rip) {
    // Reserve a chunk of code cache for this block. We don't know
    // the final size yet; conservatively reserve the size cap and
    // commit only what we use. (For a real impl we'd use xbyak's
    // internal buffer and copy out, but the bump allocator's
    // overhead is tiny.)
    u8* code_buf = code_cache_.Allocate(BLOCK_HOST_SIZE_CAP);
    if (code_buf == nullptr) {
        LOG_ERROR(Core, "Lifter: code cache full at RIP {:#x}", guest_rip);
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
        const auto status = ZydisDecoderDecodeFull(
            &decoder, reinterpret_cast<const void*>(rip), 15,
            &insn, ops);
        if (!ZYAN_SUCCESS(status)) {
            LOG_WARNING(Core, "Lifter: decode failed at {:#x}", rip);
            ++unsupported_hits_;
            // Emit a clean exit so the host program doesn't die.
            c.mov(rax, rip);
            c.mov(qword[r13 + Offsets::Rip], rax);
            c.mov(dword[r13 + offsetof(GuestState, exit_reason)],
                  static_cast<u32>(ExitReason::UnsupportedInstruction));
            c.jmp(r14);
            emitted_terminator = true;
            break;
        }

        // Dispatch on mnemonic.
        bool handled = false;
        switch (insn.mnemonic) {
            case ZYDIS_MNEMONIC_MOV: handled = EmitMov(insn, ops, c); break;
            case ZYDIS_MNEMONIC_ADD: handled = EmitAdd(insn, ops, c); break;
            case ZYDIS_MNEMONIC_RET:
                handled = EmitRet(insn, ops, c);
                if (handled) emitted_terminator = true;
                break;
            default:
                handled = false;
                break;
        }

        if (!handled) {
            LOG_WARNING(Core,
                        "Lifter: unsupported insn at {:#x} (mnemonic={})",
                        rip, static_cast<u32>(insn.mnemonic));
            ++unsupported_hits_;
            // Update state.rip to the un-lifted instruction so a
            // post-mortem caller knows where it stopped, then exit.
            c.mov(rax, rip);
            c.mov(qword[r13 + Offsets::Rip], rax);
            c.mov(dword[r13 + offsetof(GuestState, exit_reason)],
                  static_cast<u32>(ExitReason::UnsupportedInstruction));
            c.jmp(r14);
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

    LOG_DEBUG(Core,
              "Lifter: compiled block {:#x} -> {} ({} guest bytes -> {} host bytes)",
              guest_rip, static_cast<void*>(code_buf),
              rip - guest_rip, emitted);

    return code_buf;
}

} // namespace Core::Runtime
