// SPDX-FileCopyrightText: Copyright 2026 shadPS4 Emulator Project
// SPDX-License-Identifier: GPL-2.0-or-later
//
// ============================================================================
// ARM64 (AArch64) lifter — macOS / Apple Silicon ONLY.
//
// STATUS: minimal emitter set targeting the simplest runtime tests.
// Beyond the verified infrastructure (gateway, dispatch loop, MAP_JIT W^X
// bracketing, icache invalidation, block compilation, unsupported-exit), this
// implements GPR emitters — MOV (reg/imm), ADD r64 (reg/imm), SUB r64 (reg/imm),
// both arith with the lazy-flag side-band, PUSH (reg/imm), POP (reg), and RET —
// which make the simplest runtime tests (MovAddRet, MultipleRegisters, SubImm,
// PushPop) executable end-to-end.
// Every other instruction form routes to EmitUnsupportedExit. Emitters are
// added one at a time, each landing only after a green arm64 test confirms it,
// because an emitter that is subtly wrong silently corrupts guest state rather
// than taking the clean, diagnosable exit. The x86 host lifter is the reference
// for translation shapes; remaining scaffolding (NEON helpers) is kept ready.
//
// The CI macos-arm64 job is the only compile/run check — the x86_64 dev
// sandbox has no xbyak_aarch64.
//
// Two macOS/Apple-Silicon facts drive the structure:
//
//   * W^X: code pages are MAP_JIT. A thread sees them as EITHER writable
//     OR executable, toggled by pthread_jit_write_protect_np. CompileBlock
//     MUST bracket its emission: WriteBegin() (-> writable) before, then
//     WriteEnd() (-> executable) + icache invalidate after. The x86 host
//     never needed this — its pages are permanently RWX.
//
//   * I-cache is not coherent with D-cache for freshly written code. After
//     writing a block we MUST sys_icache_invalidate over its range or the
//     CPU may fetch stale bytes. This is THE classic first-run JIT-on-ARM
//     bug.
//
// And one ISA fact:
//
//   * Unlike the x86 host (guest and host are the same ISA, so many
//     instructions ran near-verbatim), the ARM64 host must TRANSLATE every
//     guest x86 instruction into one-or-more AArch64 instructions. There is
//     no "just run the host op" shortcut for scalar integer work.
//
// MEMORY MODEL / ATOMICITY (known follow-up, NOT handled by this skeleton):
//
//   GuestState is a per-thread execution context, so the plain ldr/str this
//   lifter emits for ordinary guest loads/stores are correct: a naturally
//   aligned <=64-bit GPR access IS single-copy atomic on ARMv8 (no LDAR/LDXR
//   needed), and with no second observer of a thread-local field there is
//   nothing to tear against or to order. The GPR slots are 8-byte aligned and
//   GuestState is alignas(64), so that precondition holds.
//
//   What does NOT carry over from the x86 host is GUEST-VISIBLE concurrency.
//   The x86 host got correct behavior for free from x86's strong (TSO) memory
//   model and its genuinely-atomic LOCK-prefixed ops. AArch64 is a WEAK model,
//   so when the emitter port reaches instructions whose guest semantics imply
//   atomicity or ordering across guest threads / shared memory, a plain
//   ldr/str lowering is WRONG. Specifically, these will each need real ARM
//   primitives rather than the x86 host's plain accesses:
//     - LOCK-prefixed RMW (xchg, cmpxchg, xadd, lock add/or/...) -> LSE atomics
//       (ldadd/swp/cas...) or an ldxr/stxr retry loop; NOT load-then-store.
//     - MFENCE / LFENCE / SFENCE                                 -> dmb ish*.
//     - Acquire/release-ordered guest accesses                   -> ldar/stlr.
//     - 128-bit accesses are NOT single-copy atomic before v8.4/LSE2: a 128-bit
//       guest atomic cannot be a single ldp/stp pre-v8.4 (each 64-bit half is
//       atomic, but the pair can tear). Vector (NEON) loads are not guaranteed
//       atomic at all — fine for thread-local YMM state, wrong if ever shared.
//   None of the above is implemented here; the current emitters assume the
//   thread-local case. Guest atomics/fences must be added before this backend
//   can run real multithreaded guest code correctly.
// ============================================================================

#include "core/cpu_runtime/lifter/lifter.h"

#include <cstdio> // fprintf in the destructor (logging may be torn down)

#include <Zydis/Zydis.h>
#include <xbyak_aarch64/xbyak_aarch64.h>

#include <libkern/OSCacheControl.h> // sys_icache_invalidate

#include "common/assert.h"
#include "common/logging/log.h"
#include "core/cpu_runtime/code_cache.h"
#include "core/cpu_runtime/guest_state.h"

namespace Core::Runtime {

namespace {

using namespace Xbyak_aarch64;

constexpr u64 BLOCK_HOST_SIZE_CAP = 16384;
constexpr u64 BLOCK_GUEST_SIZE_CAP = 1024;

// ============================================================================
// HOST REGISTER CONVENTIONS (AArch64) — mirror of the x86 doc block.
//
// Pinned across all JIT code (set by the gateway, never clobbered):
//   x28 = GuestState*       (mirror of x86 r13)
//   x27 = DispatcherFn      (mirror of x86 r12)
//   x26 = dispatch_loop_top (mirror of x86 r14; `br x26` = normal exit)
//   x25 = exit_stub         (mirror of x86 r15; `br x25` = fatal exit)
//
// Scratch usable inside an emitter (caller-saved x0..x15; avoid x16/x17
// = veneer scratch, x18 = platform reserved). Convention mirroring x86's
// rax/rcx/rdx primary scratch:
//   x9  = primary scratch    (rax analog)
//   x10 = secondary scratch  (rcx analog)
//   x11 = address scratch    (rdx analog; effective addresses land here)
//   x12..x15 = additional transient
//
// Host SIMD scratch (NEON): v0/v1 are the per-lane vector scratch
// (mirror of x86 xmm0/xmm1); v0..v7 are caller-saved.
// ============================================================================
// xbyak_aarch64 does not expose the pre-instantiated register objects
// (x0, w0, sp, xzr, ...) at namespace scope, so `using namespace
// Xbyak_aarch64` does NOT bring `x28` etc. into scope here. Construct the
// registers explicitly by index instead: XReg(i) is the 64-bit register i,
// WReg(i) the 32-bit view. This is the scope-independent, API-guaranteed
// form (the README shows WReg(i) in a loop). Also note XReg is a runtime
// class with no constexpr constructor, so these are `const`, not constexpr.
const XReg kState = XReg(28);     // GuestState* (x86 r13 analog)
const XReg kScratch0 = XReg(9);   // primary scratch (rax analog)
const XReg kScratch1 = XReg(10);  // secondary scratch (rcx analog)
[[maybe_unused]] const XReg kAddr = XReg(11);      // address scratch (rdx analog)
const XReg kScratch2 = XReg(12);  // additional transient
const XReg kExitStub = XReg(25);  // fatal-exit stub (x86 r15 analog)
const XReg kDispatchTop = XReg(26); // dispatcher-loop top (x86 r14 analog)
const WReg kWScratch0 = WReg(9);  // 32-bit view of x9
const WReg kWScratch3 = WReg(13); // 32-bit transient (flag_op write)

constexpr u32 GprOffset(int idx) {
    return static_cast<u32>(Offsets::Gpr + idx * 8);
}

// YMM lane chunk offset — identical layout to x86. lane 0..31,
// chunk 0..3 (each chunk is a u64). Stride 4 u64 = 32 bytes/lane.
[[maybe_unused]] constexpr u32 YmmChunkOffset(int lane, int chunk) {
    return static_cast<u32>(offsetof(GuestState, ymm) + (lane * 4 + chunk) * 8);
}

// Register-name -> canonical AMD64 index. These mirror the x86 host
// lifter's helpers exactly; both live in their TU's anonymous namespace,
// so the arm64 lifter needs its own copy (the x86 definitions are not
// linked into an arm64 build).
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
    // AH/CH/DH/BH and non-GPR registers fall through to "unsupported".
    return -1;
}

[[maybe_unused]] int ZydisVecToIndex(ZydisRegister reg) {
    if (reg >= ZYDIS_REGISTER_XMM0 && reg <= ZYDIS_REGISTER_XMM31) {
        return static_cast<int>(reg) - static_cast<int>(ZYDIS_REGISTER_XMM0);
    }
    if (reg >= ZYDIS_REGISTER_YMM0 && reg <= ZYDIS_REGISTER_YMM31) {
        return static_cast<int>(reg) - static_cast<int>(ZYDIS_REGISTER_YMM0);
    }
    return -1;
}

// Shared lazy-flag op enum value (matches the x86 lifter + the
// host-agnostic materializer in runtime.cpp).
constexpr u32 FLAG_OP_ADD = 1;
constexpr u32 FLAG_OP_SUB = 2;
// Logical ops (AND/OR/XOR/TEST): x86 clears CF and OF and sets ZF/SF/PF
// from the result. The runtime materializer reads (op, result) for this
// tag; lhs/rhs are still stashed for uniformity but are not needed to
// derive the logical flags.
constexpr u32 FLAG_OP_LOGIC = 3;
// INC / DEC: identical OF/SF/ZF/AF/PF derivation to ADD/SUB with rhs==1, but
// x86 INC/DEC leave CF UNCHANGED. The materializer treats these like ADD/SUB
// for the result-derived flags and preserves the prior CF bit instead of
// computing one. (rhs is stashed as 1 so the shared OF/AF math is reused.)
constexpr u32 FLAG_OP_INC = 4;
constexpr u32 FLAG_OP_DEC = 5;

// Forward declaration: EmitEffectiveAddress is defined later (near LEA) but is
// called earlier by the ALU mem-operand paths (EmitAlu). Declaring
// it here keeps the address-emitter grouped with its address-consuming
// emitters while satisfying C++'s define-before-use for the earlier callers.
bool EmitEffectiveAddress(const ZydisDecodedOperandMem& mem, u64 next_rip,
                          CodeGenerator& c, bool addr32 = false);

// ----------------------------------------------------------------------------
// INSTRUCTION EMITTERS.
//
// Scope is deliberately minimal: exactly the forms exercised by the simplest
// runtime tests (MovAddRet_ProducesCorrectRax and
// MultipleRegistersAndOpcodes_ProduceCorrectValues), each mirroring the x86
// host lifter's semantics. Every other form returns false and routes to the
// unsupported-exit. More emitters are added one at a time, each gated behind a
// passing arm64 test. The lazy-flag side-band model is host-agnostic: emitters
// stash (op, lhs, rhs, result) and the runtime materializer derives RFLAGS;
// we do NOT map host NZCV onto guest RFLAGS (different layout; AF/PF absent).
// ----------------------------------------------------------------------------

// ----------------------------------------------------------------------------
// FLAG MATERIALIZATION.
//
// AArch64's NZCV does not map onto x86 RFLAGS (different bit layout, and x86
// has AF/PF which NZCV lacks), so rather than translate host flags we derive
// the x86 arithmetic flags arithmetically from (op, lhs, rhs, result). This is
// the consumer of the lazy-flag side-band the arith/logic emitters stash.
//
// Derives and writes CF(0), PF(2), AF(4), ZF(6), SF(7), OF(11) into
// state.rflags from the side-band fields already in GuestState. The bit
// derivations were validated end-to-end under QEMU (cross-compiled
// aarch64) against the exact x86 reference cases the runtime tests assert.
//
// Scratch: x9..x15 (caller-saved). Reads flag_op/lhs/rhs/result, writes
// rflags. Call AFTER the side-band stores (it reloads them from memory, so
// ordering is by memory, not register liveness).
void EmitMaterializeFlags(CodeGenerator& c) {
    const XReg lhs = XReg(9);
    const XReg rhs = XReg(10);
    const XReg res = XReg(11);
    const XReg fl  = XReg(12);
    const XReg t   = XReg(13);
    const XReg u   = XReg(14);
    const WReg opw = WReg(15);
    // Width-aware extras. sb = sign-bit index (width-1); operands are masked
    // to `width` bits so unsigned compares and the parity/zero tests are
    // width-correct. We reuse memory for these so we don't run out of the
    // x9..x15 caller-saved window.
    const XReg sb  = XReg(5);   // sign-bit position (7/15/31/63)
    const XReg msk = XReg(6);   // (1<<width)-1, or all-ones for width==64
    const WReg wW  = WReg(7);   // flag_width loaded

    c.ldr(opw, ptr(kState, static_cast<u32>(offsetof(GuestState, flag_op))));
    c.ldr(lhs, ptr(kState, static_cast<u32>(offsetof(GuestState, flag_lhs))));
    c.ldr(rhs, ptr(kState, static_cast<u32>(offsetof(GuestState, flag_rhs))));
    c.ldr(res, ptr(kState, static_cast<u32>(offsetof(GuestState, flag_result))));
    c.ldr(wW, ptr(kState, static_cast<u32>(offsetof(GuestState, flag_width))));
    c.mov(fl, 0);

    // Normalize width: 0 -> 64 (legacy producers / 64-bit ops).
    {
        Label nz;
        c.cmp(wW, 0);
        c.b(NE, nz);
        c.mov(wW, 64);
        c.L(nz);
    }
    // sb = width - 1.
    c.sub(sb, XReg(7), 1);  // XReg(7) is the 64-bit view of wW (width<=64)

    // msk = (width==64) ? ~0 : (1<<width)-1.
    {
        Label is64, mdone;
        c.cmp(wW, 64);
        c.b(EQ, is64);
        c.mov(msk, 1);
        c.lsl(msk, msk, XReg(7));   // 1 << width   (width in 1..32 here)
        c.sub(msk, msk, 1);         // (1<<width)-1
        c.b(mdone);
        c.L(is64);
        c.mov(msk, 0);
        c.sub(msk, msk, 1);         // 0 - 1 = all ones
        c.L(mdone);
    }

    // Mask operands/result to width (so width-relative tests are correct).
    c.and_(lhs, lhs, msk);
    c.and_(rhs, rhs, msk);
    c.and_(res, res, msk);

    // ZF: masked result == 0 -> bit 6
    c.cmp(res, 0);
    c.cset(t, EQ);
    c.lsl(t, t, 6);
    c.orr(fl, fl, t);

    // SF: bit (width-1) of result -> bit 7. Variable shift by sb.
    c.lsr(t, res, sb);          // lsr by register
    c.and_(t, t, 1);
    c.lsl(t, t, 7);
    c.orr(fl, fl, t);

    // PF: even parity of the low byte -> bit 2 (always the low 8 bits).
    c.and_(t, res, 0xFF);
    c.eor(t, t, t, LSR, 4);
    c.eor(t, t, t, LSR, 2);
    c.eor(t, t, t, LSR, 1);
    c.mvn(t, t);
    c.and_(t, t, 1);
    c.lsl(t, t, 2);
    c.orr(fl, fl, t);

    // AF: bit 4 of (lhs ^ rhs ^ result) -> bit 4 (always bit 4, width-agnostic).
    c.eor(t, lhs, rhs);
    c.eor(t, t, res);
    c.lsr(t, t, 4);
    c.and_(t, t, 1);
    c.lsl(t, t, 4);
    c.orr(fl, fl, t);

    // CF and OF are operation-dependent. All operands are already width-masked,
    // and OF uses a variable sign-bit shift (sb) so the formulas are the
    // 64-bit ones generalized to the operand width.
    Label isSub, isLogic, isInc, isDec, doneCfOf;
    c.cmp(opw, FLAG_OP_SUB);
    c.b(EQ, isSub);
    c.cmp(opw, FLAG_OP_LOGIC);
    c.b(EQ, isLogic);
    c.cmp(opw, FLAG_OP_INC);
    c.b(EQ, isInc);
    c.cmp(opw, FLAG_OP_DEC);
    c.b(EQ, isDec);

    // ADD: CF = (result < lhs) unsigned at width. OF = (~(lhs^rhs) & (lhs^res)) >> sb.
    c.cmp(res, lhs);
    c.cset(t, CC);
    c.orr(fl, fl, t);
    c.eor(t, lhs, rhs);
    c.mvn(t, t);
    c.eor(u, lhs, res);
    c.and_(t, t, u);
    c.lsr(t, t, sb);
    c.and_(t, t, 1);
    c.lsl(t, t, 11);
    c.orr(fl, fl, t);
    c.b(doneCfOf);

    c.L(isSub);
    // SUB: CF = (lhs < rhs) unsigned at width. OF = ((lhs^rhs) & (lhs^res)) >> sb.
    c.cmp(lhs, rhs);
    c.cset(t, CC);
    c.orr(fl, fl, t);
    c.eor(t, lhs, rhs);
    c.eor(u, lhs, res);
    c.and_(t, t, u);
    c.lsr(t, t, sb);
    c.and_(t, t, 1);
    c.lsl(t, t, 11);
    c.orr(fl, fl, t);
    c.b(doneCfOf);

    c.L(isInc);
    // INC: OF like ADD (rhs==1), CF PRESERVED.
    c.eor(t, lhs, rhs);
    c.mvn(t, t);
    c.eor(u, lhs, res);
    c.and_(t, t, u);
    c.lsr(t, t, sb);
    c.and_(t, t, 1);
    c.lsl(t, t, 11);
    c.orr(fl, fl, t);
    c.ldr(u, ptr(kState, Offsets::Rflags));
    c.and_(u, u, 1);
    c.orr(fl, fl, u);
    c.b(doneCfOf);

    c.L(isDec);
    // DEC: OF like SUB (rhs==1), CF PRESERVED.
    c.eor(t, lhs, rhs);
    c.eor(u, lhs, res);
    c.and_(t, t, u);
    c.lsr(t, t, sb);
    c.and_(t, t, 1);
    c.lsl(t, t, 11);
    c.orr(fl, fl, t);
    c.ldr(u, ptr(kState, Offsets::Rflags));
    c.and_(u, u, 1);
    c.orr(fl, fl, u);
    c.b(doneCfOf);

    c.L(isLogic);
    // LOGIC: CF = OF = 0.

    c.L(doneCfOf);
    c.str(fl, ptr(kState, Offsets::Rflags));
}

// guest: mov r64, r64  /  mov r64, imm
//   reg<-reg  ldr x9,[x28,#src]; str x9,[x28,#dst]
//   reg<-imm  mov x9,#imm (movz/movk); str x9,[x28,#dst]
// guest: mov — supports widths 8/16/32/64 and reg/imm/mem on both sides.
//   Narrow-write semantics: 8/16-bit writes preserve the destination slot's
//   upper bits; 32-bit writes zero-extend bits 63:32; 64-bit is a full write.
//   AH/CH/DH/BH high-byte registers are not handled (return false). No flags.
bool EmitMov(const ZydisDecodedInstruction& insn, const ZydisDecodedOperand* ops,
             u64 next_rip, CodeGenerator& c) {
    if (insn.mnemonic != ZYDIS_MNEMONIC_MOV) return false;
    const u32 w = insn.operand_width;
    if (w != 8 && w != 16 && w != 32 && w != 64) return false;
    if (insn.operand_count_visible != 2) return false;

    const u64 wmask = (w == 64) ? ~0ull : ((1ull << w) - 1);

    // Load the source VALUE (already width-masked for the narrow cases where it
    // matters on store) into kScratch0.
    auto load_src = [&]() -> bool {
        if (ops[1].type == ZYDIS_OPERAND_TYPE_REGISTER) {
            const int src = ZydisGprToIndex(ops[1].reg.value);
            if (src < 0) return false;
            c.ldr(kScratch0, ptr(kState, GprOffset(src)));
            return true;
        }
        if (ops[1].type == ZYDIS_OPERAND_TYPE_IMMEDIATE) {
            c.mov(kScratch0, static_cast<u64>(ops[1].imm.value.s));
            return true;
        }
        if (ops[1].type == ZYDIS_OPERAND_TYPE_MEMORY) {
            if (!EmitEffectiveAddress(ops[1].mem, next_rip, c)) return false;
            switch (w) {
                case 8:  c.ldrb(WReg(9), ptr(kAddr)); break;
                case 16: c.ldrh(WReg(9), ptr(kAddr)); break;
                case 32: c.ldr(WReg(9), ptr(kAddr)); break;   // zero-extends
                default: c.ldr(kScratch0, ptr(kAddr)); break;
            }
            return true;
        }
        return false;
    };

    // ---- Register destination ----
    if (ops[0].type == ZYDIS_OPERAND_TYPE_REGISTER) {
        const int dst = ZydisGprToIndex(ops[0].reg.value);
        if (dst < 0) return false;  // AH/CH/DH/BH unsupported
        if (!load_src()) return false;
        if (w == 64) {
            c.str(kScratch0, ptr(kState, GprOffset(dst)));
        } else if (w == 32) {
            // 32-bit write zero-extends: mask src low 32, store full slot.
            c.mov(WReg(9), WReg(9));  // zero-extend low 32 of kScratch0
            c.str(kScratch0, ptr(kState, GprOffset(dst)));
        } else {
            // 8/16-bit: merge low bits into the existing slot (preserve upper).
            c.and_(kScratch0, kScratch0, wmask);          // src & wmask
            c.ldr(kScratch1, ptr(kState, GprOffset(dst)));
            c.mov(kScratch2, ~wmask);
            c.and_(kScratch1, kScratch1, kScratch2);      // slot & ~wmask
            c.orr(kScratch1, kScratch1, kScratch0);
            c.str(kScratch1, ptr(kState, GprOffset(dst)));
        }
        return true;
    }

    // ---- Memory destination ----
    if (ops[0].type == ZYDIS_OPERAND_TYPE_MEMORY) {
        // Compute the store value first (into kScratch0). If the SOURCE is also
        // memory that's not a real x86 mov form, so source is reg or imm here.
        if (ops[1].type == ZYDIS_OPERAND_TYPE_MEMORY) return false;
        if (!load_src()) return false;          // value -> kScratch0 (does not use EA)
        // Now compute the destination address (EA clobbers x9=kScratch0!), so
        // stash the value in kScratch1 first.
        c.mov(kScratch1, kScratch0);
        if (!EmitEffectiveAddress(ops[0].mem, next_rip, c)) return false;  // -> kAddr
        switch (w) {
            case 8:  c.strb(WReg(10), ptr(kAddr)); break;  // kScratch1 low byte
            case 16: c.strh(WReg(10), ptr(kAddr)); break;
            case 32: c.str(WReg(10), ptr(kAddr)); break;
            default: c.str(kScratch1, ptr(kAddr)); break;
        }
        return true;
    }

    return false;
}

// guest: not r/m — bitwise complement at width 8/16/32/64. No flags.
//   reg dest: 8/16 merge-preserve, 32 zero-extend, 64 full.
//   mem dest: write exactly width bytes.
bool EmitNot(const ZydisDecodedInstruction& insn, const ZydisDecodedOperand* ops,
             u64 next_rip, CodeGenerator& c) {
    if (insn.mnemonic != ZYDIS_MNEMONIC_NOT) return false;
    const u32 w = insn.operand_width;
    if (w != 8 && w != 16 && w != 32 && w != 64) return false;
    const u64 wmask = (w == 64) ? ~0ull : ((1ull << w) - 1);

    if (ops[0].type == ZYDIS_OPERAND_TYPE_MEMORY) {
        if (!EmitEffectiveAddress(ops[0].mem, next_rip, c)) return false;  // -> x11
        c.mov(XReg(13), kAddr);
        switch (w) {
            case 8:  c.ldrb(WReg(12), ptr(XReg(13))); break;
            case 16: c.ldrh(WReg(12), ptr(XReg(13))); break;
            case 32: c.ldr(WReg(12), ptr(XReg(13))); break;
            default: c.ldr(kScratch2, ptr(XReg(13))); break;
        }
        c.mvn(kScratch2, kScratch2);
        switch (w) {
            case 8:  c.strb(WReg(12), ptr(XReg(13))); break;
            case 16: c.strh(WReg(12), ptr(XReg(13))); break;
            case 32: c.str(WReg(12), ptr(XReg(13))); break;
            default: c.str(kScratch2, ptr(XReg(13))); break;
        }
        return true;
    }
    const int d = ZydisGprToIndex(ops[0].reg.value);
    if (d < 0) return false;
    c.ldr(kScratch0, ptr(kState, GprOffset(d)));
    c.mvn(kScratch2, kScratch0);
    if (w == 64) {
        c.str(kScratch2, ptr(kState, GprOffset(d)));
    } else if (w == 32) {
        c.mov(WReg(12), WReg(12));  // zero-extend low 32
        c.str(kScratch2, ptr(kState, GprOffset(d)));
    } else {
        c.and_(kScratch2, kScratch2, wmask);
        c.ldr(kScratch0, ptr(kState, GprOffset(d)));
        c.mov(kScratch1, ~wmask);
        c.and_(kScratch0, kScratch0, kScratch1);
        c.orr(kScratch0, kScratch0, kScratch2);
        c.str(kScratch0, ptr(kState, GprOffset(d)));
    }
    return true;
}

// guest: neg r/m — two's-complement negate at width. Flags as SUB(0, operand):
//   CF = (operand != 0), and ZF/SF/OF/PF/AF as for 0 - operand. Uses the
//   width-aware materializer with FLAG_OP_SUB, lhs=0, rhs=operand.
bool EmitNeg(const ZydisDecodedInstruction& insn, const ZydisDecodedOperand* ops,
             u64 next_rip, CodeGenerator& c) {
    if (insn.mnemonic != ZYDIS_MNEMONIC_NEG) return false;
    const u32 w = insn.operand_width;
    if (w != 8 && w != 16 && w != 32 && w != 64) return false;
    const u64 wmask = (w == 64) ? ~0ull : ((1ull << w) - 1);

    const XReg vOperand = XReg(15);  // rhs
    const XReg vRes = kScratch2;
    const bool mem = (ops[0].type == ZYDIS_OPERAND_TYPE_MEMORY);

    if (mem) {
        if (!EmitEffectiveAddress(ops[0].mem, next_rip, c)) return false;
        c.mov(XReg(13), kAddr);
        switch (w) {
            case 8:  c.ldrb(WReg(15), ptr(XReg(13))); break;
            case 16: c.ldrh(WReg(15), ptr(XReg(13))); break;
            case 32: c.ldr(WReg(15), ptr(XReg(13))); break;
            default: c.ldr(vOperand, ptr(XReg(13))); break;
        }
    } else {
        const int d = ZydisGprToIndex(ops[0].reg.value);
        if (d < 0) return false;
        c.ldr(vOperand, ptr(kState, GprOffset(d)));
    }

    c.mov(kScratch0, 0);                 // lhs = 0
    c.sub(vRes, kScratch0, vOperand);    // result = 0 - operand

    // Write back at width.
    if (mem) {
        switch (w) {
            case 8:  c.strb(WReg(12), ptr(XReg(13))); break;
            case 16: c.strh(WReg(12), ptr(XReg(13))); break;
            case 32: c.str(WReg(12), ptr(XReg(13))); break;
            default: c.str(vRes, ptr(XReg(13))); break;
        }
    } else {
        const int d = ZydisGprToIndex(ops[0].reg.value);
        if (w == 64) {
            c.str(vRes, ptr(kState, GprOffset(d)));
        } else if (w == 32) {
            c.mov(WReg(12), WReg(12));
            c.str(vRes, ptr(kState, GprOffset(d)));
        } else {
            XReg t = XReg(14);
            c.and_(t, vRes, wmask);
            c.ldr(kScratch0, ptr(kState, GprOffset(d)));
            c.mov(kScratch1, ~wmask);
            c.and_(kScratch0, kScratch0, kScratch1);
            c.orr(kScratch0, kScratch0, t);
            c.str(kScratch0, ptr(kState, GprOffset(d)));
        }
    }

    // Flags: SUB(0, operand), width-tagged.
    c.mov(kWScratch3, FLAG_OP_SUB);
    c.str(kWScratch3, ptr(kState, static_cast<u32>(offsetof(GuestState, flag_op))));
    c.mov(kWScratch3, w);
    c.str(kWScratch3, ptr(kState, static_cast<u32>(offsetof(GuestState, flag_width))));
    c.mov(kScratch0, 0);
    c.str(kScratch0, ptr(kState, static_cast<u32>(offsetof(GuestState, flag_lhs))));
    c.str(vOperand, ptr(kState, static_cast<u32>(offsetof(GuestState, flag_rhs))));
    c.str(vRes, ptr(kState, static_cast<u32>(offsetof(GuestState, flag_result))));
    EmitMaterializeFlags(c);
    return true;
}

// guest: movzx r, r/m8|r/m16 — zero-extend source into dest (16/32/64). No flags.
bool EmitMovzx(const ZydisDecodedInstruction& insn, const ZydisDecodedOperand* ops,
               u64 next_rip, CodeGenerator& c) {
    if (insn.mnemonic != ZYDIS_MNEMONIC_MOVZX) return false;
    if (ops[0].type != ZYDIS_OPERAND_TYPE_REGISTER) return false;
    const int d = ZydisGprToIndex(ops[0].reg.value);
    if (d < 0) return false;
    const u32 dst_size = insn.operand_width;
    if (dst_size != 16 && dst_size != 32 && dst_size != 64) return false;
    const u32 src_size = ops[1].size;
    if (src_size != 8 && src_size != 16) return false;

    // Load zero-extended source into kScratch0 (ldrb/ldrh/uxt zero-extend).
    if (ops[1].type == ZYDIS_OPERAND_TYPE_REGISTER) {
        const int s = ZydisGprToIndex(ops[1].reg.value);
        if (s < 0) return false;
        c.ldr(kScratch0, ptr(kState, GprOffset(s)));
        if (src_size == 8) c.uxtb(kScratch0, kScratch0);
        else               c.uxth(kScratch0, kScratch0);
    } else if (ops[1].type == ZYDIS_OPERAND_TYPE_MEMORY) {
        if (!EmitEffectiveAddress(ops[1].mem, next_rip, c)) return false;
        if (src_size == 8) c.ldrb(WReg(9), ptr(kAddr));   // zero-extends to X
        else               c.ldrh(WReg(9), ptr(kAddr));
    } else {
        return false;
    }

    // Width-aware store. 16-bit dst merges low word; 32/64 store full (already
    // zero-extended, so a 64-bit store of a value < 2^16 is correct for both).
    if (dst_size == 16) {
        c.and_(kScratch0, kScratch0, 0xFFFF);
        c.ldr(kScratch1, ptr(kState, GprOffset(d)));
        c.mov(kScratch2, ~0xFFFFull);
        c.and_(kScratch1, kScratch1, kScratch2);
        c.orr(kScratch1, kScratch1, kScratch0);
        c.str(kScratch1, ptr(kState, GprOffset(d)));
    } else {
        c.str(kScratch0, ptr(kState, GprOffset(d)));
    }
    return true;
}

// guest: movsx r, r/m8|r/m16 — sign-extend source into dest (16/32/64). No flags.
//   For a 32-bit dst the value is sign-extended to 32 bits then the upper 32
//   are zeroed (32-bit write semantics); we do this by sign-extending into a
//   W register and storing it zero-extended into the 64-bit slot.
bool EmitMovsx(const ZydisDecodedInstruction& insn, const ZydisDecodedOperand* ops,
               u64 next_rip, CodeGenerator& c) {
    if (insn.mnemonic != ZYDIS_MNEMONIC_MOVSX) return false;
    if (ops[0].type != ZYDIS_OPERAND_TYPE_REGISTER) return false;
    const int d = ZydisGprToIndex(ops[0].reg.value);
    if (d < 0) return false;
    const u32 dst_size = insn.operand_width;
    if (dst_size != 16 && dst_size != 32 && dst_size != 64) return false;
    const u32 src_size = ops[1].size;
    if (src_size != 8 && src_size != 16) return false;

    // Load the raw source bits into kScratch0 (low byte/word significant).
    if (ops[1].type == ZYDIS_OPERAND_TYPE_REGISTER) {
        const int s = ZydisGprToIndex(ops[1].reg.value);
        if (s < 0) return false;
        c.ldr(kScratch0, ptr(kState, GprOffset(s)));
    } else if (ops[1].type == ZYDIS_OPERAND_TYPE_MEMORY) {
        if (!EmitEffectiveAddress(ops[1].mem, next_rip, c)) return false;
        if (src_size == 8) c.ldrb(WReg(9), ptr(kAddr));
        else               c.ldrh(WReg(9), ptr(kAddr));
    } else {
        return false;
    }

    // Sign-extend to the destination width.
    if (dst_size == 64) {
        if (src_size == 8) c.sxtb(kScratch0, WReg(9));
        else               c.sxth(kScratch0, WReg(9));
        c.str(kScratch0, ptr(kState, GprOffset(d)));
    } else if (dst_size == 32) {
        // Sign-extend into a W reg; the 64-bit store of that (already only low
        // 32 meaningful, upper bits zero) gives 32-bit zero-extended write.
        if (src_size == 8) c.sxtb(WReg(9), WReg(9));
        else               c.sxth(WReg(9), WReg(9));
        c.mov(WReg(9), WReg(9));  // ensure upper 32 of X are zero
        c.str(kScratch0, ptr(kState, GprOffset(d)));
    } else {  // dst_size == 16, src_size == 8 (16->16 isn't a movsx)
        c.sxtb(WReg(9), WReg(9));
        c.and_(kScratch0, kScratch0, 0xFFFF);
        c.ldr(kScratch1, ptr(kState, GprOffset(d)));
        c.mov(kScratch2, ~0xFFFFull);
        c.and_(kScratch1, kScratch1, kScratch2);
        c.orr(kScratch1, kScratch1, kScratch0);
        c.str(kScratch1, ptr(kState, GprOffset(d)));
    }
    return true;
}

// guest: movsxd r64, r/m32 — sign-extend 32->64. No flags.
bool EmitMovsxd(const ZydisDecodedInstruction& insn, const ZydisDecodedOperand* ops,
                u64 next_rip, CodeGenerator& c) {
    if (insn.mnemonic != ZYDIS_MNEMONIC_MOVSXD) return false;
    if (ops[0].type != ZYDIS_OPERAND_TYPE_REGISTER) return false;
    const int d = ZydisGprToIndex(ops[0].reg.value);
    if (d < 0) return false;

    if (ops[1].type == ZYDIS_OPERAND_TYPE_REGISTER) {
        const int s = ZydisGprToIndex(ops[1].reg.value);
        if (s < 0) return false;
        c.ldr(WReg(9), ptr(kState, GprOffset(s)));  // low 32
        c.sxtw(kScratch0, WReg(9));
    } else if (ops[1].type == ZYDIS_OPERAND_TYPE_MEMORY) {
        if (!EmitEffectiveAddress(ops[1].mem, next_rip, c)) return false;
        c.ldrsw(kScratch0, ptr(kAddr));  // sign-extending 32->64 load
    } else {
        return false;
    }
    c.str(kScratch0, ptr(kState, GprOffset(d)));
    return true;
}

// ----------------------------------------------------------------------------
// UNIFIED ALU EMITTER (ADD / SUB / AND / OR / XOR).
//
// Handles the full matrix these five share: widths 8/16/32/64, destination =
// register or memory, source = register / immediate / memory. Uses the
// width-aware lazy-flag materializer (stashes op/lhs/rhs/result + flag_width).
//
// Register conventions inside here (all caller-saved, none pinned):
//   x11 (kAddr)     — effective address when an operand is memory (from EA)
//   x13             — held copy of the dest address across RMW (EA clobbers x9)
//   x14             — lhs value
//   x15             — rhs value
//   x12 (kScratch2) — result
// EA internally uses x9 and x11; we read those out immediately, so the only
// live value that must survive an EA call is a previously-computed address,
// which we keep in x13.
//
// Narrow write-back: register dest at width 8/16 merges into the slot
// (preserve upper bits); width 32 zero-extends; width 64 full. Memory dest
// writes exactly `width` bytes (strb/strh/str(W)/str).
//
// High-byte regs (AH/CH/DH/BH) are not handled — ZydisGprToIndex returns -1,
// so those route to the unsupported-exit (a separate batch).
bool EmitAlu(const ZydisDecodedInstruction& insn, const ZydisDecodedOperand* ops,
             u64 next_rip, CodeGenerator& c) {
    u32 flag_op;
    switch (insn.mnemonic) {
        case ZYDIS_MNEMONIC_ADD: flag_op = FLAG_OP_ADD;   break;
        case ZYDIS_MNEMONIC_SUB: flag_op = FLAG_OP_SUB;   break;
        case ZYDIS_MNEMONIC_AND:
        case ZYDIS_MNEMONIC_OR:
        case ZYDIS_MNEMONIC_XOR: flag_op = FLAG_OP_LOGIC; break;
        default: return false;
    }
    const u32 w = insn.operand_width;
    if (w != 8 && w != 16 && w != 32 && w != 64) return false;
    if (insn.operand_count_visible != 2) return false;

    const XReg vLhs = XReg(14);
    const XReg vRhs = XReg(15);
    const XReg vRes = kScratch2;       // x12
    const XReg vAddr = XReg(13);       // held dest address (mem-dest RMW)
    const u64 wmask = (w == 64) ? ~0ull : ((1ull << w) - 1);

    const bool dst_mem = (ops[0].type == ZYDIS_OPERAND_TYPE_MEMORY);
    const bool src_mem = (ops[1].type == ZYDIS_OPERAND_TYPE_MEMORY);
    if (dst_mem && src_mem) return false;  // not a real x86 ALU form

    auto width_load = [&](const XReg& dstreg, const XReg& addr) {
        switch (w) {
            case 8:  c.ldrb(WReg(dstreg.getIdx()), ptr(addr)); break;
            case 16: c.ldrh(WReg(dstreg.getIdx()), ptr(addr)); break;
            case 32: c.ldr(WReg(dstreg.getIdx()), ptr(addr)); break;
            default: c.ldr(dstreg, ptr(addr)); break;
        }
    };

    // ---- Gather lhs (dst operand value) and rhs (src operand value). ----
    if (dst_mem) {
        // dst is memory: compute its address first, keep it in vAddr, load lhs.
        if (!EmitEffectiveAddress(ops[0].mem, next_rip, c)) return false;  // -> x11
        c.mov(vAddr, kAddr);
        width_load(vLhs, vAddr);
        // src is register or immediate (src_mem excluded above).
        if (ops[1].type == ZYDIS_OPERAND_TYPE_REGISTER) {
            const int s = ZydisGprToIndex(ops[1].reg.value);
            if (s < 0) return false;
            c.ldr(vRhs, ptr(kState, GprOffset(s)));
        } else if (ops[1].type == ZYDIS_OPERAND_TYPE_IMMEDIATE) {
            c.mov(vRhs, static_cast<u64>(ops[1].imm.value.s));
        } else {
            return false;
        }
    } else {
        // dst is a register.
        const int d = ZydisGprToIndex(ops[0].reg.value);
        if (d < 0) return false;
        if (src_mem) {
            // Load rhs from memory first (EA clobbers x9/x11), then lhs.
            if (!EmitEffectiveAddress(ops[1].mem, next_rip, c)) return false;
            c.mov(vAddr, kAddr);
            width_load(vRhs, vAddr);
            c.ldr(vLhs, ptr(kState, GprOffset(d)));
        } else {
            c.ldr(vLhs, ptr(kState, GprOffset(d)));
            if (ops[1].type == ZYDIS_OPERAND_TYPE_REGISTER) {
                const int s = ZydisGprToIndex(ops[1].reg.value);
                if (s < 0) return false;
                c.ldr(vRhs, ptr(kState, GprOffset(s)));
            } else if (ops[1].type == ZYDIS_OPERAND_TYPE_IMMEDIATE) {
                c.mov(vRhs, static_cast<u64>(ops[1].imm.value.s));
            } else {
                return false;
            }
        }
    }

    // ---- Compute result. ----
    switch (insn.mnemonic) {
        case ZYDIS_MNEMONIC_ADD: c.add(vRes, vLhs, vRhs);  break;
        case ZYDIS_MNEMONIC_SUB: c.sub(vRes, vLhs, vRhs);  break;
        case ZYDIS_MNEMONIC_AND: c.and_(vRes, vLhs, vRhs); break;
        case ZYDIS_MNEMONIC_OR:  c.orr(vRes, vLhs, vRhs);  break;
        case ZYDIS_MNEMONIC_XOR: c.eor(vRes, vLhs, vRhs);  break;
        default: return false;
    }

    // ---- Lazy-flag side-band (width-tagged), stashed BEFORE write-back.
    //      The narrow merge path below reuses vLhs/vRhs as scratch, so the
    //      operands must be committed to the side-band first. The materializer
    //      reads these from memory and masks to flag_width.
    //      NB: must NOT use kWScratch3 here — it is WReg(13), which aliases
    //      vAddr (XReg(13)); clobbering it would corrupt the mem-dest store
    //      address. Use WReg(9) (kScratch0's W view), free after operand load. ----
    c.mov(WReg(9), flag_op);
    c.str(WReg(9), ptr(kState, static_cast<u32>(offsetof(GuestState, flag_op))));
    c.mov(WReg(9), w);
    c.str(WReg(9), ptr(kState, static_cast<u32>(offsetof(GuestState, flag_width))));
    c.str(vLhs, ptr(kState, static_cast<u32>(offsetof(GuestState, flag_lhs))));
    c.str(vRhs, ptr(kState, static_cast<u32>(offsetof(GuestState, flag_rhs))));
    c.str(vRes, ptr(kState, static_cast<u32>(offsetof(GuestState, flag_result))));

    // ---- Write back at width. (May clobber vLhs/vRhs in the narrow path;
    //      safe now that the side-band is already committed.) ----
    if (dst_mem) {
        switch (w) {
            case 8:  c.strb(WReg(vRes.getIdx()), ptr(vAddr)); break;
            case 16: c.strh(WReg(vRes.getIdx()), ptr(vAddr)); break;
            case 32: c.str(WReg(vRes.getIdx()), ptr(vAddr)); break;
            default: c.str(vRes, ptr(vAddr)); break;
        }
    } else {
        const int d = ZydisGprToIndex(ops[0].reg.value);
        if (w == 64) {
            c.str(vRes, ptr(kState, GprOffset(d)));
        } else if (w == 32) {
            c.mov(WReg(vRes.getIdx()), WReg(vRes.getIdx()));  // zero-extend low 32
            c.str(vRes, ptr(kState, GprOffset(d)));
        } else {
            // 8/16: merge low bits into existing slot, preserve upper.
            c.and_(vRes, vRes, wmask);
            c.ldr(vLhs, ptr(kState, GprOffset(d)));       // reuse vLhs as slot
            c.mov(vRhs, ~wmask);                          // reuse vRhs as mask
            c.and_(vLhs, vLhs, vRhs);
            c.orr(vLhs, vLhs, vRes);
            c.str(vLhs, ptr(kState, GprOffset(d)));
        }
    }

    EmitMaterializeFlags(c);
    return true;
}

// guest: push r64 / push imm.   value = src; rsp -= 8; [rsp] = value.
// (mem source deferred — needs effective-address emitter.) No flags affected.
bool EmitPush(const ZydisDecodedInstruction& insn, const ZydisDecodedOperand* ops,
              CodeGenerator& c) {
    if (insn.mnemonic != ZYDIS_MNEMONIC_PUSH) return false;
    if (insn.operand_width != 64) return false;
    if (insn.operand_count_visible != 1) return false;

    constexpr int RSP_IDX = 4;
    if (ops[0].type == ZYDIS_OPERAND_TYPE_REGISTER) {
        const int src = ZydisGprToIndex(ops[0].reg.value);
        if (src < 0) return false;
        c.ldr(kScratch0, ptr(kState, GprOffset(src)));
    } else if (ops[0].type == ZYDIS_OPERAND_TYPE_IMMEDIATE) {
        c.mov(kScratch0, static_cast<u64>(ops[0].imm.value.s)); // sign-extended
    } else {
        return false; // mem source deferred
    }
    c.ldr(kScratch1, ptr(kState, GprOffset(RSP_IDX))); // x10 = guest RSP
    c.sub(kScratch1, kScratch1, 8);
    c.str(kScratch0, ptr(kScratch1));                  // write to guest stack
    c.str(kScratch1, ptr(kState, GprOffset(RSP_IDX))); // update RSP
    return true;
}

// guest: pop r64.   reg = [rsp]; rsp += 8.   No flags affected.
bool EmitPop(const ZydisDecodedInstruction& insn, const ZydisDecodedOperand* ops,
             CodeGenerator& c) {
    if (insn.mnemonic != ZYDIS_MNEMONIC_POP) return false;
    if (insn.operand_width != 64) return false;
    if (insn.operand_count_visible != 1) return false;
    if (ops[0].type != ZYDIS_OPERAND_TYPE_REGISTER) return false;
    const int dst = ZydisGprToIndex(ops[0].reg.value);
    if (dst < 0) return false;

    constexpr int RSP_IDX = 4;
    c.ldr(kScratch1, ptr(kState, GprOffset(RSP_IDX))); // x10 = guest RSP
    c.ldr(kScratch0, ptr(kScratch1));                  // x9 = [rsp]
    c.add(kScratch1, kScratch1, 8);                    // rsp += 8
    c.str(kScratch1, ptr(kState, GprOffset(RSP_IDX))); // update RSP
    c.str(kScratch0, ptr(kState, GprOffset(dst)));     // dst = popped value
    return true;
}

// guest: ret (no-immediate form only).
//   rip = [rsp]; rsp += 8; exit_reason = BlockEnd; br dispatch-top (normal).
// Mirrors x86 EmitRet exactly. This is the NORMAL block terminator — it sets
// emitted_terminator so CompileBlock does not append a fallthrough exit.
bool EmitRet(const ZydisDecodedInstruction& insn, CodeGenerator& c) {
    if (insn.mnemonic != ZYDIS_MNEMONIC_RET) return false;
    if (insn.operand_count_visible != 0) return false; // not RET imm16

    constexpr int RSP_IDX = 4; // GPR[4] = RSP in canonical order
    c.ldr(kScratch0, ptr(kState, GprOffset(RSP_IDX))); // x9 = guest RSP
    c.ldr(kScratch1, ptr(kScratch0));                  // x10 = [rsp] (ret addr)
    c.str(kScratch1, ptr(kState, Offsets::Rip));       // state.rip = ret addr
    c.add(kScratch0, kScratch0, 8);                    // pop: rsp += 8
    c.str(kScratch0, ptr(kState, GprOffset(RSP_IDX)));

    c.mov(kWScratch0, static_cast<u32>(ExitReason::BlockEnd));
    c.str(kWScratch0, ptr(kState, static_cast<u32>(offsetof(GuestState, exit_reason))));
    c.br(kDispatchTop); // normal dispatcher re-entry (NOT fatal exit)
    return true;
}

// guest: inc/dec r64 (register form; 8/16/32-bit and memory forms deferred —
//   they need width-aware flag derivation and the effective-address emitter).
//   result = lhs +/- 1; CF is PRESERVED (the defining quirk of INC/DEC vs
//   ADD/SUB); OF/SF/ZF/AF/PF derived from the result. rhs is stashed as 1 so
//   the materializer's shared OF/AF math applies unchanged.
bool EmitIncDec64(const ZydisDecodedInstruction& insn, const ZydisDecodedOperand* ops,
                  CodeGenerator& c) {
    const bool is_inc = (insn.mnemonic == ZYDIS_MNEMONIC_INC);
    if (!is_inc && insn.mnemonic != ZYDIS_MNEMONIC_DEC) return false;
    if (insn.operand_width != 64) return false;            // narrow/mem deferred
    if (insn.operand_count_visible != 1) return false;
    if (ops[0].type != ZYDIS_OPERAND_TYPE_REGISTER) return false;
    const int dst = ZydisGprToIndex(ops[0].reg.value);
    if (dst < 0) return false;

    c.ldr(kScratch0, ptr(kState, GprOffset(dst)));         // lhs
    c.mov(kScratch1, 1);                                   // rhs = 1
    if (is_inc) {
        c.add(kScratch2, kScratch0, kScratch1);
    } else {
        c.sub(kScratch2, kScratch0, kScratch1);
    }
    c.str(kScratch2, ptr(kState, GprOffset(dst)));

    c.mov(kWScratch3, is_inc ? FLAG_OP_INC : FLAG_OP_DEC);
    c.str(kWScratch3, ptr(kState, static_cast<u32>(offsetof(GuestState, flag_op))));
    c.mov(kWScratch3, 64);  // operand width (all current ALU emitters are 64-bit)
    c.str(kWScratch3, ptr(kState, static_cast<u32>(offsetof(GuestState, flag_width))));
    c.str(kScratch0, ptr(kState, static_cast<u32>(offsetof(GuestState, flag_lhs))));
    c.str(kScratch1, ptr(kState, static_cast<u32>(offsetof(GuestState, flag_rhs))));
    c.str(kScratch2, ptr(kState, static_cast<u32>(offsetof(GuestState, flag_result))));
    EmitMaterializeFlags(c);
    return true;
}

// ----------------------------------------------------------------------------
// EFFECTIVE ADDRESS.
//
// Computes a guest memory operand's address into kAddr (x11), mirroring the
// x86 host EmitEffectiveAddress. Guest addresses are absolute pointers into
// mapped guest memory, so the computed value is dereferenced directly by a
// plain ldr/str (single-copy atomic for <=64-bit aligned thread-local state;
// see the memory-model note at the top of this file for where that stops
// holding). Clobbers kAddr (x11) and x9 (index scaling). Returns false for
// unsupported forms (non-flat segment override, RIP+index, bad SIB scale),
// routing the caller to the unsupported-exit.
//
// addr32: the 0x67 address-size-override. The whole sum is computed then
// masked to 32 bits — exact because (a+b) mod 2^32 == ((a mod 2^32)+(b mod
// 2^32)) mod 2^32. Lets e.g. `mov [ebx], eax` reach a sub-4GiB address even
// when the upper 32 bits of the base register hold garbage.
bool EmitEffectiveAddress(const ZydisDecodedOperandMem& mem, u64 next_rip,
                          CodeGenerator& c, bool addr32) {
    // Only flat segments (DS/SS/CS/ES). FS/GS would need TLS-base handling.
    if (mem.segment != ZYDIS_REGISTER_DS && mem.segment != ZYDIS_REGISTER_SS &&
        mem.segment != ZYDIS_REGISTER_CS && mem.segment != ZYDIS_REGISTER_ES) {
        return false;
    }

    const bool has_base = (mem.base != ZYDIS_REGISTER_NONE);
    const bool has_index = (mem.index != ZYDIS_REGISTER_NONE);
    const s64 disp = mem.disp.value;

    const XReg idx_scratch = XReg(9);  // index*scale staging (rax analog)

    // RIP-relative: address = next_rip + disp, constant-folded.
    if (has_base && mem.base == ZYDIS_REGISTER_RIP) {
        if (has_index) return false;
        c.mov(kAddr, static_cast<u64>(static_cast<s64>(next_rip) + disp));
        return true;
    }

    // Plain [disp] absolute (no base, no index).
    if (!has_base && !has_index) {
        c.mov(kAddr, static_cast<u64>(disp));
        return true;
    }

    // General: kAddr = base + index*scale + disp.
    if (has_base) {
        const int base_idx = ZydisGprToIndex(mem.base);
        if (base_idx < 0) return false;
        c.ldr(kAddr, ptr(kState, GprOffset(base_idx)));
    } else {
        c.mov(kAddr, 0);
    }

    if (has_index) {
        const int index_idx = ZydisGprToIndex(mem.index);
        if (index_idx < 0) return false;
        c.ldr(idx_scratch, ptr(kState, GprOffset(index_idx)));
        switch (mem.scale) {
            case 1:  break;
            case 2:  c.lsl(idx_scratch, idx_scratch, 1); break;
            case 4:  c.lsl(idx_scratch, idx_scratch, 2); break;
            case 8:  c.lsl(idx_scratch, idx_scratch, 3); break;
            default: return false;  // invalid SIB scale
        }
        c.add(kAddr, kAddr, idx_scratch);
    }

    if (disp != 0) {
        // AArch64 add-immediate is limited to 12 bits (optionally <<12), so
        // materialize the displacement into a scratch and add register-form.
        // (mov handles any 64-bit constant via movz/movk; sign-extended disp
        // is already in s64.)
        c.mov(idx_scratch, static_cast<u64>(disp));
        c.add(kAddr, kAddr, idx_scratch);
    }

    if (addr32) {
        // Mask to 32 bits: a 32-bit (W) register write zero-extends bits
        // 63:32 of the underlying X register. (There is no scalar uxtw in
        // this assembler; the W-move is the canonical equivalent.)
        c.mov(WReg(11), WReg(11));
    }
    return true;
}

// guest: lea r64/r32, [mem]. Computes the effective address (no dereference)
//   into the destination. Address math is full-width regardless of operand
//   size; a 32-bit LEA writes the low 32 and zero-extends bits 63:32.
bool EmitLea(const ZydisDecodedInstruction& insn, const ZydisDecodedOperand* ops,
             u64 next_rip, CodeGenerator& c) {
    if (insn.mnemonic != ZYDIS_MNEMONIC_LEA) return false;
    if (insn.operand_width != 64 && insn.operand_width != 32) return false;
    if (ops[0].type != ZYDIS_OPERAND_TYPE_REGISTER) return false;
    if (ops[1].type != ZYDIS_OPERAND_TYPE_MEMORY) return false;
    const int dst = ZydisGprToIndex(ops[0].reg.value);
    if (dst < 0) return false;

    if (!EmitEffectiveAddress(ops[1].mem, next_rip, c)) return false;  // -> kAddr
    if (insn.operand_width == 32) {
        c.mov(WReg(11), WReg(11));  // zero-extend low 32 into the 64-bit slot
    }
    c.str(kAddr, ptr(kState, GprOffset(dst)));
    return true;
}

// guest: cmp lhs, rhs — SUB that discards the result but sets flags. Supports
//   reg/reg, reg/imm, reg/mem, and mem/reg at widths 8/16/32/64. The flag
//   side-band carries the operation width so the materializer derives
//   width-correct ZF/SF/CF/OF.
bool EmitCmp(const ZydisDecodedInstruction& insn, const ZydisDecodedOperand* ops,
             u64 next_rip, CodeGenerator& c) {
    if (insn.mnemonic != ZYDIS_MNEMONIC_CMP) return false;
    const u32 w = insn.operand_width;
    if (w != 8 && w != 16 && w != 32 && w != 64) return false;
    if (insn.operand_count_visible != 2) return false;

    // Load lhs (ops[0]) into kScratch0, rhs (ops[1]) into kScratch1.
    auto load_operand = [&](const ZydisDecodedOperand& op, const XReg& dstreg) -> bool {
        if (op.type == ZYDIS_OPERAND_TYPE_REGISTER) {
            const int idx = ZydisGprToIndex(op.reg.value);
            if (idx < 0) return false;
            c.ldr(dstreg, ptr(kState, GprOffset(idx)));
            return true;
        }
        if (op.type == ZYDIS_OPERAND_TYPE_IMMEDIATE) {
            c.mov(dstreg, static_cast<u64>(op.imm.value.s));
            return true;
        }
        if (op.type == ZYDIS_OPERAND_TYPE_MEMORY) {
            if (!EmitEffectiveAddress(op.mem, next_rip, c)) return false;  // -> kAddr
            c.ldr(dstreg, ptr(kAddr));
            return true;
        }
        return false;
    };
    // Memory operand uses kAddr/x9 during EA; load mem operand first if either
    // side is memory to avoid clobbering an already-loaded register operand.
    if (ops[1].type == ZYDIS_OPERAND_TYPE_MEMORY) {
        if (!load_operand(ops[1], kScratch1)) return false;
        if (!load_operand(ops[0], kScratch0)) return false;
    } else {
        if (!load_operand(ops[0], kScratch0)) return false;
        if (!load_operand(ops[1], kScratch1)) return false;
    }
    c.sub(kScratch2, kScratch0, kScratch1);  // result (discarded except flags)

    c.mov(kWScratch3, FLAG_OP_SUB);
    c.str(kWScratch3, ptr(kState, static_cast<u32>(offsetof(GuestState, flag_op))));
    c.mov(kWScratch3, w);
    c.str(kWScratch3, ptr(kState, static_cast<u32>(offsetof(GuestState, flag_width))));
    c.str(kScratch0, ptr(kState, static_cast<u32>(offsetof(GuestState, flag_lhs))));
    c.str(kScratch1, ptr(kState, static_cast<u32>(offsetof(GuestState, flag_rhs))));
    c.str(kScratch2, ptr(kState, static_cast<u32>(offsetof(GuestState, flag_result))));
    EmitMaterializeFlags(c);
    return true;
}

// guest: test lhs, rhs — AND that discards the result but sets flags (CF=OF=0,
//   ZF/SF/PF from result). Supports reg/reg, reg/imm, mem/reg at 8/16/32/64.
bool EmitTest(const ZydisDecodedInstruction& insn, const ZydisDecodedOperand* ops,
              u64 next_rip, CodeGenerator& c) {
    if (insn.mnemonic != ZYDIS_MNEMONIC_TEST) return false;
    const u32 w = insn.operand_width;
    if (w != 8 && w != 16 && w != 32 && w != 64) return false;
    if (insn.operand_count_visible != 2) return false;

    auto load_operand = [&](const ZydisDecodedOperand& op, const XReg& dstreg) -> bool {
        if (op.type == ZYDIS_OPERAND_TYPE_REGISTER) {
            const int idx = ZydisGprToIndex(op.reg.value);
            if (idx < 0) return false;
            c.ldr(dstreg, ptr(kState, GprOffset(idx)));
            return true;
        }
        if (op.type == ZYDIS_OPERAND_TYPE_IMMEDIATE) {
            c.mov(dstreg, static_cast<u64>(op.imm.value.s));
            return true;
        }
        if (op.type == ZYDIS_OPERAND_TYPE_MEMORY) {
            if (!EmitEffectiveAddress(op.mem, next_rip, c)) return false;
            c.ldr(dstreg, ptr(kAddr));
            return true;
        }
        return false;
    };
    if (ops[0].type == ZYDIS_OPERAND_TYPE_MEMORY) {
        if (!load_operand(ops[0], kScratch0)) return false;
        if (!load_operand(ops[1], kScratch1)) return false;
    } else {
        if (!load_operand(ops[0], kScratch0)) return false;
        if (!load_operand(ops[1], kScratch1)) return false;
    }
    c.and_(kScratch2, kScratch0, kScratch1);  // result (discarded except flags)

    c.mov(kWScratch3, FLAG_OP_LOGIC);
    c.str(kWScratch3, ptr(kState, static_cast<u32>(offsetof(GuestState, flag_op))));
    c.mov(kWScratch3, w);
    c.str(kWScratch3, ptr(kState, static_cast<u32>(offsetof(GuestState, flag_width))));
    c.str(kScratch0, ptr(kState, static_cast<u32>(offsetof(GuestState, flag_lhs))));
    c.str(kScratch1, ptr(kState, static_cast<u32>(offsetof(GuestState, flag_rhs))));
    c.str(kScratch2, ptr(kState, static_cast<u32>(offsetof(GuestState, flag_result))));
    EmitMaterializeFlags(c);
    return true;
}

// guest: jmp rel / jmp r64 / jmp [mem] — unconditional. Block terminator.
bool EmitJmp(const ZydisDecodedInstruction& insn, const ZydisDecodedOperand* ops,
             u64 next_rip, CodeGenerator& c) {
    if (insn.mnemonic != ZYDIS_MNEMONIC_JMP) return false;
    if (ops[0].type == ZYDIS_OPERAND_TYPE_IMMEDIATE) {
        const u64 target = ops[0].imm.is_relative
            ? static_cast<u64>(static_cast<s64>(next_rip) + ops[0].imm.value.s)
            : static_cast<u64>(ops[0].imm.value.s);
        c.mov(kScratch0, target);
    } else if (ops[0].type == ZYDIS_OPERAND_TYPE_REGISTER) {
        const int idx = ZydisGprToIndex(ops[0].reg.value);
        if (idx < 0) return false;
        c.ldr(kScratch0, ptr(kState, GprOffset(idx)));
    } else if (ops[0].type == ZYDIS_OPERAND_TYPE_MEMORY) {
        if (!EmitEffectiveAddress(ops[0].mem, next_rip, c)) return false;
        c.ldr(kScratch0, ptr(kAddr));  // jmp [mem]: target = *(addr)
    } else {
        return false;
    }
    c.str(kScratch0, ptr(kState, Offsets::Rip));
    c.mov(kWScratch0, static_cast<u32>(ExitReason::BlockEnd));
    c.str(kWScratch0, ptr(kState, static_cast<u32>(offsetof(GuestState, exit_reason))));
    c.br(kDispatchTop);
    return true;
}

// guest: call rel / call r64 / call [mem] — push return addr, jump. Terminator.
//   Target is computed BEFORE the push (the mem form uses kAddr/x9, which the
//   push also touches), mirroring the x86 host emitter's ordering.
bool EmitCall(const ZydisDecodedInstruction& insn, const ZydisDecodedOperand* ops,
              u64 next_rip, CodeGenerator& c) {
    if (insn.mnemonic != ZYDIS_MNEMONIC_CALL) return false;
    constexpr int RSP_IDX = 4;

    // Step 1: target -> kScratch1 (x10), survives the push.
    if (ops[0].type == ZYDIS_OPERAND_TYPE_IMMEDIATE) {
        const u64 target = ops[0].imm.is_relative
            ? static_cast<u64>(static_cast<s64>(next_rip) + ops[0].imm.value.s)
            : static_cast<u64>(ops[0].imm.value.s);
        c.mov(kScratch1, target);
    } else if (ops[0].type == ZYDIS_OPERAND_TYPE_REGISTER) {
        const int idx = ZydisGprToIndex(ops[0].reg.value);
        if (idx < 0) return false;
        c.ldr(kScratch1, ptr(kState, GprOffset(idx)));
    } else if (ops[0].type == ZYDIS_OPERAND_TYPE_MEMORY) {
        if (!EmitEffectiveAddress(ops[0].mem, next_rip, c)) return false;
        c.ldr(kScratch1, ptr(kAddr));
    } else {
        return false;
    }

    // Step 2: push next_rip. rsp -= 8; [rsp] = next_rip.
    c.ldr(kScratch0, ptr(kState, GprOffset(RSP_IDX)));
    c.sub(kScratch0, kScratch0, 8);
    c.mov(kScratch2, next_rip);
    c.str(kScratch2, ptr(kScratch0));
    c.str(kScratch0, ptr(kState, GprOffset(RSP_IDX)));

    // Step 3: rip = target, exit to dispatcher.
    c.str(kScratch1, ptr(kState, Offsets::Rip));
    c.mov(kWScratch0, static_cast<u32>(ExitReason::BlockEnd));
    c.str(kWScratch0, ptr(kState, static_cast<u32>(offsetof(GuestState, exit_reason))));
    c.br(kDispatchTop);
    return true;
}

// ----------------------------------------------------------------------------
// CONDITIONAL FAMILY (Setcc / Cmovcc / Jcc).
//
// All three decode an x86 condition code from the mnemonic and act on the
// guest RFLAGS that a PRIOR instruction already materialized into
// state.rflags (via EmitMaterializeFlags). They do NOT consult the lazy-flag
// side-band directly: by the time a conditional runs, the producing
// instruction has already written the full RFLAGS word, so we read the
// finished bits — exactly mirroring the x86 host lifter's EmitJccCondition,
// which reads rax(=materialized flags) too.
//
// EmitConditionToReg loads state.rflags and computes the boolean (0/1) for the
// given condition into `out`. Scratch used: x9..x14 (caller-saved); `out`
// should be one of those. Returns false if the mnemonic's condition suffix is
// not recognized (caller then routes to unsupported-exit).
//
// RFLAGS bit positions (x86): CF=0, PF=2, ZF=6, SF=7, OF=11.
// ----------------------------------------------------------------------------

// Condition classes, keyed off the shared suffix of the Jcc/SETcc/CMOVcc
// mnemonic. "N" variants are the boolean negation of the base.
enum class CondClass { O, B, Z, BE, S, P, L, LE };

// Map any Jcc/SETcc/CMOVcc mnemonic to (class, negated). Returns false if not
// a recognized conditional.
bool DecodeCondition(ZydisMnemonic m, CondClass& cls, bool& neg) {
    switch (m) {
    // Overflow
    case ZYDIS_MNEMONIC_JO:    case ZYDIS_MNEMONIC_SETO:    case ZYDIS_MNEMONIC_CMOVO:
        cls = CondClass::O;  neg = false; return true;
    case ZYDIS_MNEMONIC_JNO:   case ZYDIS_MNEMONIC_SETNO:   case ZYDIS_MNEMONIC_CMOVNO:
        cls = CondClass::O;  neg = true;  return true;
    // Below / carry
    case ZYDIS_MNEMONIC_JB:    case ZYDIS_MNEMONIC_SETB:    case ZYDIS_MNEMONIC_CMOVB:
        cls = CondClass::B;  neg = false; return true;
    case ZYDIS_MNEMONIC_JNB:   case ZYDIS_MNEMONIC_SETNB:   case ZYDIS_MNEMONIC_CMOVNB:
        cls = CondClass::B;  neg = true;  return true;
    // Zero / equal
    case ZYDIS_MNEMONIC_JZ:    case ZYDIS_MNEMONIC_SETZ:    case ZYDIS_MNEMONIC_CMOVZ:
        cls = CondClass::Z;  neg = false; return true;
    case ZYDIS_MNEMONIC_JNZ:   case ZYDIS_MNEMONIC_SETNZ:   case ZYDIS_MNEMONIC_CMOVNZ:
        cls = CondClass::Z;  neg = true;  return true;
    // Below-or-equal
    case ZYDIS_MNEMONIC_JBE:   case ZYDIS_MNEMONIC_SETBE:   case ZYDIS_MNEMONIC_CMOVBE:
        cls = CondClass::BE; neg = false; return true;
    case ZYDIS_MNEMONIC_JNBE:  case ZYDIS_MNEMONIC_SETNBE:  case ZYDIS_MNEMONIC_CMOVNBE:
        cls = CondClass::BE; neg = true;  return true;
    // Sign
    case ZYDIS_MNEMONIC_JS:    case ZYDIS_MNEMONIC_SETS:    case ZYDIS_MNEMONIC_CMOVS:
        cls = CondClass::S;  neg = false; return true;
    case ZYDIS_MNEMONIC_JNS:   case ZYDIS_MNEMONIC_SETNS:   case ZYDIS_MNEMONIC_CMOVNS:
        cls = CondClass::S;  neg = true;  return true;
    // Parity
    case ZYDIS_MNEMONIC_JP:    case ZYDIS_MNEMONIC_SETP:    case ZYDIS_MNEMONIC_CMOVP:
        cls = CondClass::P;  neg = false; return true;
    case ZYDIS_MNEMONIC_JNP:   case ZYDIS_MNEMONIC_SETNP:   case ZYDIS_MNEMONIC_CMOVNP:
        cls = CondClass::P;  neg = true;  return true;
    // Less (SF != OF)
    case ZYDIS_MNEMONIC_JL:    case ZYDIS_MNEMONIC_SETL:    case ZYDIS_MNEMONIC_CMOVL:
        cls = CondClass::L;  neg = false; return true;
    case ZYDIS_MNEMONIC_JNL:   case ZYDIS_MNEMONIC_SETNL:   case ZYDIS_MNEMONIC_CMOVNL:
        cls = CondClass::L;  neg = true;  return true;
    // Less-or-equal (ZF | (SF != OF))
    case ZYDIS_MNEMONIC_JLE:   case ZYDIS_MNEMONIC_SETLE:   case ZYDIS_MNEMONIC_CMOVLE:
        cls = CondClass::LE; neg = false; return true;
    case ZYDIS_MNEMONIC_JNLE:  case ZYDIS_MNEMONIC_SETNLE:  case ZYDIS_MNEMONIC_CMOVNLE:
        cls = CondClass::LE; neg = true;  return true;
    default:
        return false;
    }
}

// Compute the 0/1 boolean for (cls, neg) into `out` from state.rflags.
// Clobbers x9, x10, x11 (and `out`, which should be one of x9..x14).
void EmitConditionToReg(CondClass cls, bool neg, const XReg& out,
                        CodeGenerator& c) {
    const XReg fl = XReg(9);   // rflags
    const XReg a  = XReg(10);  // term A
    const XReg b  = XReg(11);  // term B
    c.ldr(fl, ptr(kState, Offsets::Rflags));

    auto bit = [&](const XReg& dst, int pos) {
        c.lsr(dst, fl, pos);
        c.and_(dst, dst, 1);
    };

    switch (cls) {
    case CondClass::O:  bit(out, 11); break;                 // OF
    case CondClass::B:  bit(out, 0);  break;                 // CF
    case CondClass::Z:  bit(out, 6);  break;                 // ZF
    case CondClass::S:  bit(out, 7);  break;                 // SF
    case CondClass::P:  bit(out, 2);  break;                 // PF
    case CondClass::BE:                                      // CF | ZF
        bit(a, 0); bit(b, 6); c.orr(out, a, b); break;
    case CondClass::L:                                       // SF ^ OF
        bit(a, 7); bit(b, 11); c.eor(out, a, b); break;
    case CondClass::LE:                                      // ZF | (SF ^ OF)
        bit(a, 7); bit(b, 11); c.eor(a, a, b);   // a = SF^OF
        bit(b, 6);                               // b = ZF
        c.orr(out, a, b); break;
    }
    if (neg) {
        c.eor(out, out, 1);  // boolean negate (out is exactly 0 or 1)
    }
}

// guest: setcc r/m8 (register form only; mem form deferred).
//   Writes the condition (0/1) into the destination byte, preserving the
//   upper 56 bits of the 64-bit GPR slot (x86 SETcc writes only 8 bits).
bool EmitSetcc(const ZydisDecodedInstruction& insn, const ZydisDecodedOperand* ops,
               CodeGenerator& c) {
    CondClass cls; bool neg;
    if (!DecodeCondition(insn.mnemonic, cls, neg)) return false;
    if (insn.operand_count_visible != 1) return false;
    if (ops[0].type != ZYDIS_OPERAND_TYPE_REGISTER) return false;
    const int dst = ZydisGprToIndex(ops[0].reg.value);
    if (dst < 0) return false;  // AH/CH/DH/BH high-byte regs unsupported

    const XReg cond = XReg(12);          // result of condition (0/1)
    EmitConditionToReg(cls, neg, cond, c);
    // Merge low byte: slot = (slot & ~0xFF) | cond.
    const XReg slot = XReg(13);
    c.ldr(slot, ptr(kState, GprOffset(dst)));
    c.and_(slot, slot, ~0xFFULL);
    c.orr(slot, slot, cond);             // cond is 0/1, fits the low byte
    c.str(slot, ptr(kState, GprOffset(dst)));
    return true;
}

// guest: cmovcc r64, r64  (register source; mem source deferred).
//   dst = cond ? src : dst. Full 64-bit move when taken (matches the test
//   coverage; cmov r32 would zero-extend — deferred with other 32-bit forms).
bool EmitCmov(const ZydisDecodedInstruction& insn, const ZydisDecodedOperand* ops,
              CodeGenerator& c) {
    CondClass cls; bool neg;
    if (!DecodeCondition(insn.mnemonic, cls, neg)) return false;
    if (insn.operand_width != 64) return false;
    if (insn.operand_count_visible != 2) return false;
    if (ops[0].type != ZYDIS_OPERAND_TYPE_REGISTER) return false;
    if (ops[1].type != ZYDIS_OPERAND_TYPE_REGISTER) return false;  // mem deferred
    const int dst = ZydisGprToIndex(ops[0].reg.value);
    const int src = ZydisGprToIndex(ops[1].reg.value);
    if (dst < 0 || src < 0) return false;

    const XReg cond = XReg(12);
    EmitConditionToReg(cls, neg, cond, c);
    const XReg dval = XReg(13);
    const XReg sval = XReg(14);
    c.ldr(dval, ptr(kState, GprOffset(dst)));
    c.ldr(sval, ptr(kState, GprOffset(src)));
    // csel: dst = (cond != 0) ? sval : dval.
    c.cmp(cond, 0);
    c.csel(dval, sval, dval, NE);
    c.str(dval, ptr(kState, GprOffset(dst)));
    return true;
}

// guest: jcc rel  — conditional branch. Block terminator. Mirrors the x86
//   host EmitJcc: select (taken target | next_rip) into state.rip, set
//   BlockEnd, and re-enter the dispatcher, which compiles/runs whichever block
//   the chosen RIP names. No in-block branch is emitted.
bool EmitJcc(const ZydisDecodedInstruction& insn, const ZydisDecodedOperand* ops,
             u64 next_rip, CodeGenerator& c) {
    CondClass cls; bool neg;
    if (!DecodeCondition(insn.mnemonic, cls, neg)) return false;
    if (ops[0].type != ZYDIS_OPERAND_TYPE_IMMEDIATE) return false;
    const u64 target = ops[0].imm.is_relative
        ? static_cast<u64>(static_cast<s64>(next_rip) + ops[0].imm.value.s)
        : static_cast<u64>(ops[0].imm.value.s);

    const XReg cond = XReg(12);
    EmitConditionToReg(cls, neg, cond, c);
    const XReg taken = XReg(13);
    const XReg fall  = XReg(14);
    c.mov(taken, target);
    c.mov(fall, next_rip);
    c.cmp(cond, 0);
    c.csel(fall, taken, fall, NE);       // rip = cond ? target : next_rip
    c.str(fall, ptr(kState, Offsets::Rip));

    c.mov(kWScratch0, static_cast<u32>(ExitReason::BlockEnd));
    c.str(kWScratch0, ptr(kState, static_cast<u32>(offsetof(GuestState, exit_reason))));
    c.br(kDispatchTop);  // normal dispatcher re-entry
    return true;
}

// ----------------------------------------------------------------------------
// Fatal-exit tail (unsupported instruction).
//   set state.rip, set exit_reason, br x25.
// ----------------------------------------------------------------------------
void EmitUnsupportedExit(u64 rip, CodeGenerator& c) {
    c.mov(kScratch0, rip); // mov pseudo-op -> movz/movk sequence for 64-bit imm
    c.str(kScratch0, ptr(kState, Offsets::Rip));
    c.mov(kWScratch0, static_cast<u32>(ExitReason::UnsupportedInstruction));
    c.str(kWScratch0, ptr(kState, static_cast<u32>(offsetof(GuestState, exit_reason))));
    c.br(kExitStub); // fatal exit (do not re-dispatch the bad address)
}

} // namespace

// ============================================================================
// CompileBlock — same control structure as the x86 lifter, PLUS the
// Apple-Silicon W^X bracketing and icache invalidation.
// ============================================================================
Lifter::Lifter(CodeCache& code_cache) : code_cache_(code_cache) {
    LOG_INFO(Core, "Lifter (arm64 host) initialized");
}

Lifter::~Lifter() {
    // Use fprintf rather than LOG_INFO here: by the time the lifter
    // destructor runs at program shutdown, shadPS4's logging subsystem
    // has often been torn down, and LOG_INFO would emit the format
    // string verbatim. fprintf works at any teardown phase. (Mirrors the
    // x86 host lifter destructor.)
    std::fprintf(stderr,
                 "[lifter] %llu blocks compiled, %llu bytes emitted, %llu unsupported\n",
                 (unsigned long long)blocks_compiled_,
                 (unsigned long long)bytes_emitted_,
                 (unsigned long long)unsupported_hits_);
    std::fflush(stderr);
}

void* Lifter::CompileBlock(u64 guest_rip) {
    u8* code_buf = code_cache_.Allocate(BLOCK_HOST_SIZE_CAP);
    if (code_buf == nullptr) {
        return nullptr; // caller flushes + retries
    }

    // --- W^X: switch this thread's MAP_JIT pages to WRITABLE for the
    // duration of emission. The x86 path has no equivalent (permanent
    // RWX). WriteEnd() below flips back to executable + barrier.
    code_cache_.WriteBegin();

    CodeGenerator c{BLOCK_HOST_SIZE_CAP, code_buf};

    ZydisDecoder decoder;
    // Guest is x86-64 regardless of host: decoder config is identical to
    // the x86 host lifter.
    ZydisDecoderInit(&decoder, ZYDIS_MACHINE_MODE_LONG_64, ZYDIS_STACK_WIDTH_64);

    u64 rip = guest_rip;
    const u64 block_end_cap = guest_rip + BLOCK_GUEST_SIZE_CAP;
    bool emitted_terminator = false;

    while (rip < block_end_cap) {
        ZydisDecodedInstruction insn;
        ZydisDecodedOperand ops[ZYDIS_MAX_OPERAND_COUNT];
        const auto status =
            ZydisDecoderDecodeFull(&decoder, reinterpret_cast<const void*>(rip), 15,
                                   &insn, ops);
        if (!ZYAN_SUCCESS(status)) {
            ++unsupported_hits_;
            EmitUnsupportedExit(rip, c);
            emitted_terminator = true;
            break;
        }

        const u64 next_rip = rip + insn.length;
        bool handled = false;
        switch (insn.mnemonic) {
        case ZYDIS_MNEMONIC_MOV:
            handled = EmitMov(insn, ops, next_rip, c);
            break;
        case ZYDIS_MNEMONIC_CMP:
            handled = EmitCmp(insn, ops, next_rip, c);
            break;
        case ZYDIS_MNEMONIC_TEST:
            handled = EmitTest(insn, ops, next_rip, c);
            break;
        case ZYDIS_MNEMONIC_NOT:
            handled = EmitNot(insn, ops, next_rip, c);
            break;
        case ZYDIS_MNEMONIC_NEG:
            handled = EmitNeg(insn, ops, next_rip, c);
            break;
        case ZYDIS_MNEMONIC_MOVZX:
            handled = EmitMovzx(insn, ops, next_rip, c);
            break;
        case ZYDIS_MNEMONIC_MOVSX:
            handled = EmitMovsx(insn, ops, next_rip, c);
            break;
        case ZYDIS_MNEMONIC_MOVSXD:
            handled = EmitMovsxd(insn, ops, next_rip, c);
            break;
        case ZYDIS_MNEMONIC_NOP:
            handled = true;  // NOP (incl. multi-byte forms): emit nothing.
            break;
        case ZYDIS_MNEMONIC_JMP:
            handled = EmitJmp(insn, ops, next_rip, c);
            if (handled) emitted_terminator = true;
            break;
        case ZYDIS_MNEMONIC_CALL:
            handled = EmitCall(insn, ops, next_rip, c);
            if (handled) emitted_terminator = true;
            break;
        case ZYDIS_MNEMONIC_ADD:
        case ZYDIS_MNEMONIC_SUB:
        case ZYDIS_MNEMONIC_AND:
        case ZYDIS_MNEMONIC_OR:
        case ZYDIS_MNEMONIC_XOR:
            handled = EmitAlu(insn, ops, next_rip, c);
            break;
        case ZYDIS_MNEMONIC_LEA:
            handled = EmitLea(insn, ops, next_rip, c);
            break;
        case ZYDIS_MNEMONIC_INC:
        case ZYDIS_MNEMONIC_DEC:
            handled = EmitIncDec64(insn, ops, c);
            break;
        case ZYDIS_MNEMONIC_PUSH:
            handled = EmitPush(insn, ops, c);
            break;
        case ZYDIS_MNEMONIC_POP:
            handled = EmitPop(insn, ops, c);
            break;
        case ZYDIS_MNEMONIC_RET:
            handled = EmitRet(insn, c);
            if (handled) emitted_terminator = true; // RET is a block terminator
            break;
        // SETcc (all 16 conditions) — register byte form.
        case ZYDIS_MNEMONIC_SETO:  case ZYDIS_MNEMONIC_SETNO:
        case ZYDIS_MNEMONIC_SETB:  case ZYDIS_MNEMONIC_SETNB:
        case ZYDIS_MNEMONIC_SETZ:  case ZYDIS_MNEMONIC_SETNZ:
        case ZYDIS_MNEMONIC_SETBE: case ZYDIS_MNEMONIC_SETNBE:
        case ZYDIS_MNEMONIC_SETS:  case ZYDIS_MNEMONIC_SETNS:
        case ZYDIS_MNEMONIC_SETP:  case ZYDIS_MNEMONIC_SETNP:
        case ZYDIS_MNEMONIC_SETL:  case ZYDIS_MNEMONIC_SETNL:
        case ZYDIS_MNEMONIC_SETLE: case ZYDIS_MNEMONIC_SETNLE:
            handled = EmitSetcc(insn, ops, c);
            break;
        // CMOVcc (all 16 conditions) — register/register form.
        case ZYDIS_MNEMONIC_CMOVO:  case ZYDIS_MNEMONIC_CMOVNO:
        case ZYDIS_MNEMONIC_CMOVB:  case ZYDIS_MNEMONIC_CMOVNB:
        case ZYDIS_MNEMONIC_CMOVZ:  case ZYDIS_MNEMONIC_CMOVNZ:
        case ZYDIS_MNEMONIC_CMOVBE: case ZYDIS_MNEMONIC_CMOVNBE:
        case ZYDIS_MNEMONIC_CMOVS:  case ZYDIS_MNEMONIC_CMOVNS:
        case ZYDIS_MNEMONIC_CMOVP:  case ZYDIS_MNEMONIC_CMOVNP:
        case ZYDIS_MNEMONIC_CMOVL:  case ZYDIS_MNEMONIC_CMOVNL:
        case ZYDIS_MNEMONIC_CMOVLE: case ZYDIS_MNEMONIC_CMOVNLE:
            handled = EmitCmov(insn, ops, c);
            break;
        // Jcc (all 16 conditions) — relative, block terminator.
        case ZYDIS_MNEMONIC_JO:  case ZYDIS_MNEMONIC_JNO:
        case ZYDIS_MNEMONIC_JB:  case ZYDIS_MNEMONIC_JNB:
        case ZYDIS_MNEMONIC_JZ:  case ZYDIS_MNEMONIC_JNZ:
        case ZYDIS_MNEMONIC_JBE: case ZYDIS_MNEMONIC_JNBE:
        case ZYDIS_MNEMONIC_JS:  case ZYDIS_MNEMONIC_JNS:
        case ZYDIS_MNEMONIC_JP:  case ZYDIS_MNEMONIC_JNP:
        case ZYDIS_MNEMONIC_JL:  case ZYDIS_MNEMONIC_JNL:
        case ZYDIS_MNEMONIC_JLE: case ZYDIS_MNEMONIC_JNLE:
            handled = EmitJcc(insn, ops, next_rip, c);
            if (handled) emitted_terminator = true; // Jcc terminates the block
            break;
        default:
            break;
        }

        if (emitted_terminator) break; // RET already emitted its terminator

        if (!handled) {
            ++unsupported_hits_;
            EmitUnsupportedExit(rip, c);
            emitted_terminator = true;
            break;
        }

        rip += insn.length;
        if (emitted_terminator) break;

        constexpr u64 HOST_SIZE_MARGIN = 256;
        if (c.getSize() + HOST_SIZE_MARGIN >= BLOCK_HOST_SIZE_CAP) break;
    }

    if (!emitted_terminator) {
        c.mov(kScratch0, rip);
        c.str(kScratch0, ptr(kState, Offsets::Rip));
        c.mov(kWScratch0, static_cast<u32>(ExitReason::BlockEnd));
        c.str(kWScratch0, ptr(kState, static_cast<u32>(offsetof(GuestState, exit_reason))));
        c.br(kDispatchTop); // normal dispatcher re-entry
    }

    // Lay out labels in the buffer (no protect — we own W^X here).
    c.ready();

    const u64 emitted = c.getSize();

    // --- W^X: flip back to executable. WriteEnd() does
    // pthread_jit_write_protect_np(1).
    code_cache_.WriteEnd();

    // --- I-cache invalidate over exactly the emitted range. Without
    // this the CPU may fetch stale bytes from the freshly written block.
    // (CodeCache::WriteEnd currently can't know the range; do it here
    // where we have [code_buf, code_buf+emitted). If WriteEnd is later
    // extended to take a range, fold this into it.)
    // xbyak_aarch64 declares sys_icache_invalidate in its own namespace (not
    // the global one), so qualify it explicitly. (`::` does NOT work here.)
    Xbyak_aarch64::sys_icache_invalidate(code_buf, emitted);

    bytes_emitted_ += emitted;
    ++blocks_compiled_;
    return code_buf;
}

} // namespace Core::Runtime
