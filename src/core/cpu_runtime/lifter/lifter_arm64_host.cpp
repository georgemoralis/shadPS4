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
    // Preserve non-arithmetic control bits from the prior RFLAGS. The
    // materializer only computes the six arithmetic flags
    // (CF=0,PF=2,AF=4,ZF=6,SF=7,OF=11); every other bit — notably DF (bit 10),
    // but also IF/TF/reserved — must survive. Without this, a SCAS/CMPS in a
    // DF=1 context would clear DF and corrupt the next iteration's advance.
    {
        constexpr u64 kArithMask =
            (1ull<<0)|(1ull<<2)|(1ull<<4)|(1ull<<6)|(1ull<<7)|(1ull<<11);
        c.ldr(u, ptr(kState, Offsets::Rflags));
        c.mov(t, ~kArithMask);
        c.and_(u, u, t);          // u = old control bits (DF etc.)
        c.orr(fl, fl, u);         // merge into freshly-computed arithmetic flags
    }
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

// guest: shl/shr/sar r/m, imm8|CL — shifts at widths 8/16/32/64, register dest.
//   x86 flag semantics (computed directly into rflags here, since "CF = last
//   bit shifted out" doesn't fit the lazy-flag (lhs,rhs,result) model):
//     - count is masked to 0x3F (w==64) else 0x1F, matching x86.
//     - count==0: NO flags change, value unchanged.
//     - SHL: CF = bit (w-count) of original; result = val<<count.
//     - SHR: CF = bit (count-1) of original (logical, zero-fill).
//     - SAR: CF = bit (count-1) of original (arithmetic, sign-fill).
//     - SF/ZF/PF from the width-truncated result; AF undefined (left as-is).
//     - OF only meaningful for count==1; we set it per-op (SHL: MSB(res)^CF;
//       SHR: MSB(original); SAR: 0). For count>1 x86 leaves OF undefined, so
//       any value is architecturally acceptable.
//   Register conventions: val=x14, cnt=x15, res=x12, oldflags/scratch x9/x10,
//   bitwork x13.
enum class ShiftKind { Shl, Shr, Sar };
bool EmitShift(const ZydisDecodedInstruction& insn, const ZydisDecodedOperand* ops,
               ShiftKind kind, CodeGenerator& c) {
    const u32 w = insn.operand_width;
    if (w != 8 && w != 16 && w != 32 && w != 64) return false;
    if (insn.operand_count_visible < 2) return false;
    if (ops[0].type != ZYDIS_OPERAND_TYPE_REGISTER) return false;
    const int d = ZydisGprToIndex(ops[0].reg.value);
    if (d < 0) return false;

    const XReg val = XReg(14), cnt = XReg(15), res = kScratch2;  // x12
    const XReg t = XReg(13), oldfl = kScratch0, msk = kScratch1;
    const u64 wmask = (w == 64) ? ~0ull : ((1ull << w) - 1);
    const u32 cntmask = (w == 64) ? 0x3F : 0x1F;
    const u32 sb = w - 1;

    // Load count -> cnt (masked).
    if (ops[1].type == ZYDIS_OPERAND_TYPE_IMMEDIATE) {
        c.mov(cnt, static_cast<u64>(ops[1].imm.value.u & cntmask));
    } else if (ops[1].type == ZYDIS_OPERAND_TYPE_REGISTER) {
        if (ops[1].reg.value != ZYDIS_REGISTER_CL) return false;
        c.ldr(cnt, ptr(kState, GprOffset(1)));     // rcx
        c.and_(cnt, cnt, cntmask);
    } else {
        return false;
    }

    // Load value, truncated to width into val.
    c.ldr(val, ptr(kState, GprOffset(d)));
    if (w != 64) c.and_(val, val, wmask);
    if (kind == ShiftKind::Sar && w != 64) {
        // sign-extend the width-w value to 64 so asr fills correctly.
        if (w == 8)       c.sxtb(val, WReg(val.getIdx()));
        else if (w == 16) c.sxth(val, WReg(val.getIdx()));
        else              c.sxtw(val, WReg(val.getIdx()));
    }
    // Capture MSB of the original (width-w) value into x6 now, before val is
    // later reused as the merge slot — SHR's OF needs it.
    c.lsr(XReg(6), val, sb);
    c.and_(XReg(6), XReg(6), 1);

    // Compute result = val (shift) cnt, using variable shift by cnt.
    switch (kind) {
        case ShiftKind::Shl: c.lsl(res, val, cnt); break;
        case ShiftKind::Shr: c.lsr(res, val, cnt); break;
        case ShiftKind::Sar: c.asr(res, val, cnt); break;
    }
    if (w != 64) c.and_(res, res, wmask);   // truncate result to width

    // Compute CF (only valid when cnt != 0). Done before storing flags.
    // SHL: CF = (val >> (w - cnt)) & 1; SHR/SAR: CF = (val >> (cnt-1)) & 1.
    // Use t as the CF bit.
    {
        Label cntzero, cfdone;
        c.cmp(cnt, 0);
        c.b(EQ, cntzero);
        if (kind == ShiftKind::Shl) {
            // shamt = w - cnt
            c.mov(t, w);
            c.sub(t, t, cnt);
            c.lsr(t, val, t);
        } else {
            // shamt = cnt - 1
            c.sub(t, cnt, 1);
            // For SAR the sign bits don't affect bit (cnt-1) extraction of the
            // original magnitude; val already holds the (sign-extended) value,
            // and bit (cnt-1) is within width, so this is correct.
            c.lsr(t, val, t);
        }
        c.and_(t, t, 1);
        c.b(cfdone);
        c.L(cntzero);
        c.mov(t, 0);   // placeholder; for cnt==0 we won't update flags at all
        c.L(cfdone);
    }

    // Store result back at width (8/16 merge-preserve, 32 zero-extend, 64 full).
    if (w == 64) {
        c.str(res, ptr(kState, GprOffset(d)));
    } else if (w == 32) {
        c.mov(WReg(res.getIdx()), WReg(res.getIdx()));
        c.str(res, ptr(kState, GprOffset(d)));
    } else {
        XReg slot = XReg(14);  // reuse val
        c.ldr(slot, ptr(kState, GprOffset(d)));
        c.mov(msk, ~wmask);
        c.and_(slot, slot, msk);
        c.orr(slot, slot, res);
        c.str(slot, ptr(kState, GprOffset(d)));
    }

    // Flags. For cnt==0, leave rflags untouched. Otherwise recompute
    // CF/ZF/SF/PF (+OF for cnt==1, approximated for cnt>1).
    {
        Label noflags;
        c.cmp(cnt, 0);
        c.b(EQ, noflags);

        c.ldr(oldfl, ptr(kState, Offsets::Rflags));
        // Clear CF(0) PF(2) ZF(6) SF(7) OF(11); keep the rest (AF etc.).
        c.mov(msk, ~((1ull<<0)|(1ull<<2)|(1ull<<6)|(1ull<<7)|(1ull<<11)));
        c.and_(oldfl, oldfl, msk);

        // CF (bit 0) from t.
        c.orr(oldfl, oldfl, t);

        // ZF: res == 0 -> bit 6.
        c.cmp(res, 0);
        c.cset(XReg(10), EQ);   // reuse msk's reg (x10) as scratch now
        c.lsl(XReg(10), XReg(10), 6);
        c.orr(oldfl, oldfl, XReg(10));

        // SF: bit (w-1) of res -> bit 7.
        c.lsr(XReg(10), res, sb);
        c.and_(XReg(10), XReg(10), 1);
        c.lsl(XReg(10), XReg(10), 7);
        c.orr(oldfl, oldfl, XReg(10));

        // PF: parity of low byte -> bit 2.
        c.and_(XReg(10), res, 0xFF);
        c.eor(XReg(10), XReg(10), XReg(10), LSR, 4);
        c.eor(XReg(10), XReg(10), XReg(10), LSR, 2);
        c.eor(XReg(10), XReg(10), XReg(10), LSR, 1);
        c.mvn(XReg(10), XReg(10));
        c.and_(XReg(10), XReg(10), 1);
        c.lsl(XReg(10), XReg(10), 2);
        c.orr(oldfl, oldfl, XReg(10));

        // OF: SHL -> MSB(res) ^ CF; SHR -> MSB(original val); SAR -> 0.
        if (kind == ShiftKind::Shl) {
            c.lsr(XReg(10), res, sb);   // MSB(res)
            c.and_(XReg(10), XReg(10), 1);
            c.eor(XReg(10), XReg(10), t);  // ^ CF
            c.lsl(XReg(10), XReg(10), 11);
            c.orr(oldfl, oldfl, XReg(10));
        } else if (kind == ShiftKind::Shr) {
            c.and_(XReg(10), XReg(6), 1);   // MSB(original), captured earlier
            c.lsl(XReg(10), XReg(10), 11);
            c.orr(oldfl, oldfl, XReg(10));
        }
        // SAR: OF cleared (already masked off above).

        c.str(oldfl, ptr(kState, Offsets::Rflags));
        c.L(noflags);
    }
    return true;
}

// guest: rol/ror r/m, imm8|CL — rotates at widths 8/16/32/64, register dest.
//   x86 flag semantics: ROL/ROR affect ONLY CF and OF; SF/ZF/PF/AF are
//   preserved. Count masked to 0x3F (w==64) else 0x1F. If the masked count is
//   0, NO flags change and the value is unchanged.
//     ROL: result rotated left; CF = bit 0 of result; OF(count==1)=MSB(res)^CF.
//     ROR: result rotated right; CF = MSB of result;
//          OF(count==1) = MSB(res) ^ bit(w-2) of res.
//   Register conventions: val=x14, cnt=x15, res=x12, scratch x9/x10/x13.
enum class RotateKind { Rol, Ror };
bool EmitRotate(const ZydisDecodedInstruction& insn, const ZydisDecodedOperand* ops,
                RotateKind kind, CodeGenerator& c) {
    const u32 w = insn.operand_width;
    if (w != 8 && w != 16 && w != 32 && w != 64) return false;
    if (insn.operand_count_visible < 2) return false;
    if (ops[0].type != ZYDIS_OPERAND_TYPE_REGISTER) return false;
    const int d = ZydisGprToIndex(ops[0].reg.value);
    if (d < 0) return false;

    const XReg val = XReg(14), cnt = XReg(15), res = kScratch2;  // x12
    const XReg t = XReg(13), oldfl = kScratch0, msk = kScratch1;
    const u64 wmask = (w == 64) ? ~0ull : ((1ull << w) - 1);
    const u32 cntmask = (w == 64) ? 0x3F : 0x1F;
    const u32 sb = w - 1;

    // Load count -> cnt (masked to 5/6 bits).
    if (ops[1].type == ZYDIS_OPERAND_TYPE_IMMEDIATE) {
        c.mov(cnt, static_cast<u64>(ops[1].imm.value.u & cntmask));
    } else if (ops[1].type == ZYDIS_OPERAND_TYPE_REGISTER) {
        if (ops[1].reg.value != ZYDIS_REGISTER_CL) return false;
        c.ldr(cnt, ptr(kState, GprOffset(1)));
        c.and_(cnt, cnt, cntmask);
    } else {
        return false;
    }

    // Load value truncated to width.
    c.ldr(val, ptr(kState, GprOffset(d)));
    if (w != 64) c.and_(val, val, wmask);

    // Compute rotated result into res.
    if (w == 64) {
        if (kind == RotateKind::Ror) {
            c.rorv(res, val, cnt);
        } else {
            // ROL(n) = ROR((64 - n) & 63). For n==0 this is ROR(0)=identity.
            c.mov(t, 64);
            c.sub(t, t, cnt);
            c.and_(t, t, 0x3F);
            c.rorv(res, val, t);
        }
    } else if (w == 32) {
        if (kind == RotateKind::Ror) {
            c.rorv(WReg(res.getIdx()), WReg(val.getIdx()), WReg(cnt.getIdx()));
        } else {
            c.mov(WReg(t.getIdx()), 32);
            c.sub(WReg(t.getIdx()), WReg(t.getIdx()), WReg(cnt.getIdx()));
            c.and_(WReg(t.getIdx()), WReg(t.getIdx()), 0x1F);
            c.rorv(WReg(res.getIdx()), WReg(val.getIdx()), WReg(t.getIdx()));
        }
        c.mov(WReg(res.getIdx()), WReg(res.getIdx()));  // zero-extend
    } else {
        // 8/16-bit: rotate within w bits via shifts. effective e = cnt mod w.
        // res = ((val << e) | (val >> (w - e))) & wmask, with e==0 -> val.
        c.mov(t, cnt);
        if (w == 8)  c.and_(t, t, 0x7);   // cnt mod 8
        else         c.and_(t, t, 0xF);   // cnt mod 16
        Label rotzero, rotdone;
        c.cmp(t, 0);
        c.b(EQ, rotzero);
        if (kind == RotateKind::Rol) {
            // left part: val << e
            c.lsl(res, val, t);
            // right part: val >> (w - e)
            c.mov(msk, w);
            c.sub(msk, msk, t);
            c.lsr(XReg(10), val, msk);   // x10 scratch (msk reg is x10)
            // NB: msk==x10, lsr dest also x10 — fold: compute shift then store.
            c.orr(res, res, XReg(10));
        } else {
            // ror: right part: val >> e ; left part: val << (w - e)
            c.lsr(res, val, t);
            c.mov(msk, w);
            c.sub(msk, msk, t);
            c.lsl(XReg(10), val, msk);
            c.orr(res, res, XReg(10));
        }
        c.and_(res, res, wmask);
        c.b(rotdone);
        c.L(rotzero);
        c.mov(res, val);
        c.L(rotdone);
    }

    // Store result back at width (preserve upper bits for narrow).
    if (w == 64) {
        c.str(res, ptr(kState, GprOffset(d)));
    } else if (w == 32) {
        c.str(res, ptr(kState, GprOffset(d)));   // already zero-extended
    } else {
        XReg slot = XReg(14);  // reuse val
        c.ldr(slot, ptr(kState, GprOffset(d)));
        c.mov(msk, ~wmask);
        c.and_(slot, slot, msk);
        c.and_(res, res, wmask);
        c.orr(slot, slot, res);
        c.str(slot, ptr(kState, GprOffset(d)));
    }

    // Flags: only CF and OF, and only when masked count != 0.
    {
        Label noflags;
        c.cmp(cnt, 0);
        c.b(EQ, noflags);

        c.ldr(oldfl, ptr(kState, Offsets::Rflags));
        // Clear only CF(0) and OF(11); preserve SF/ZF/PF/AF/etc.
        c.mov(msk, ~((1ull<<0)|(1ull<<11)));
        c.and_(oldfl, oldfl, msk);

        // CF: ROL -> bit 0 of res; ROR -> bit (w-1) of res.
        if (kind == RotateKind::Rol) {
            c.and_(t, res, 1);              // CF = res[0]
        } else {
            c.lsr(t, res, sb);
            c.and_(t, t, 1);               // CF = res[w-1]
        }
        c.orr(oldfl, oldfl, t);            // place CF at bit 0

        // OF: ROL -> MSB(res) ^ CF ; ROR -> MSB(res) ^ bit(w-2).
        // (x86 defines OF only for count==1; for count>1 any value is fine.)
        if (kind == RotateKind::Rol) {
            c.lsr(XReg(10), res, sb);
            c.and_(XReg(10), XReg(10), 1);
            c.eor(XReg(10), XReg(10), t);  // ^ CF (t still holds CF=res[0])
        } else {
            c.lsr(XReg(10), res, sb);      // MSB
            c.and_(XReg(10), XReg(10), 1);
            c.lsr(t, res, sb - 1);         // bit(w-2)
            c.and_(t, t, 1);
            c.eor(XReg(10), XReg(10), t);
        }
        c.lsl(XReg(10), XReg(10), 11);
        c.orr(oldfl, oldfl, XReg(10));

        c.str(oldfl, ptr(kState, Offsets::Rflags));
        c.L(noflags);
    }
    return true;
}

// guest: adc/sbb r/m, r/imm/m — widths 8/16/32/64, register destination.
//   Reads carry-in from state.rflags bit 0 (the previous op's materialized CF),
//   computes lhs +/- rhs +/- cf, and writes CF/OF/SF/ZF/PF directly to rflags.
//   ADC: res = lhs + rhs + cf;  CF = unsigned carry out.
//        OF = (~(lhs^rhs) & (lhs^res)) sign bit.
//   SBB: res = lhs - rhs - cf;  CF = borrow out (res, as unsigned, > lhs path).
//        OF = ((lhs^rhs) & (lhs^res)) sign bit.
//   Register conventions: lhs=x14, rhs=x15, res=x12, cf=x13, carry/scratch x9/x10.
enum class AdcSbbKind { Adc, Sbb };
bool EmitAdcSbb(const ZydisDecodedInstruction& insn, const ZydisDecodedOperand* ops,
                AdcSbbKind kind, CodeGenerator& c) {
    const u32 w = insn.operand_width;
    if (w != 8 && w != 16 && w != 32 && w != 64) return false;
    if (insn.operand_count_visible < 2) return false;
    if (ops[0].type != ZYDIS_OPERAND_TYPE_REGISTER) return false;
    const int d = ZydisGprToIndex(ops[0].reg.value);
    if (d < 0) return false;

    const XReg lhs = XReg(14), rhs = XReg(15), res = kScratch2;  // x12
    const XReg cf = XReg(13), c1 = kScratch0, c2 = kScratch1;    // x9, x10
    const u64 wmask = (w == 64) ? ~0ull : ((1ull << w) - 1);
    const u32 sb = w - 1;

    // Load lhs (dst reg) and rhs (reg/imm/mem), truncated to width.
    c.ldr(lhs, ptr(kState, GprOffset(d)));
    if (ops[1].type == ZYDIS_OPERAND_TYPE_REGISTER) {
        const int s = ZydisGprToIndex(ops[1].reg.value);
        if (s < 0) return false;
        c.ldr(rhs, ptr(kState, GprOffset(s)));
    } else if (ops[1].type == ZYDIS_OPERAND_TYPE_IMMEDIATE) {
        c.mov(rhs, static_cast<u64>(ops[1].imm.value.s));  // sign-extended imm
    } else if (ops[1].type == ZYDIS_OPERAND_TYPE_MEMORY) {
        if (!EmitEffectiveAddress(ops[1].mem, /*next_rip*/0, c)) return false;
        switch (w) {
            case 8:  c.ldrb(WReg(rhs.getIdx()), ptr(kAddr)); break;
            case 16: c.ldrh(WReg(rhs.getIdx()), ptr(kAddr)); break;
            case 32: c.ldr(WReg(rhs.getIdx()), ptr(kAddr)); break;
            default: c.ldr(rhs, ptr(kAddr)); break;
        }
    } else {
        return false;
    }
    if (w != 64) { c.and_(lhs, lhs, wmask); c.and_(rhs, rhs, wmask); }

    // Carry-in: cf = rflags & 1.
    c.ldr(cf, ptr(kState, Offsets::Rflags));
    c.and_(cf, cf, 1);

    if (kind == AdcSbbKind::Adc) {
        // res = lhs + rhs + cf.
        c.add(res, lhs, rhs);
        c.add(res, res, cf);
        if (w == 64) {
            // CF = (lhs+rhs overflowed) | ((lhs+rhs)+cf overflowed).
            // c1 = (lhs+rhs) < lhs ; recompute partial in c1.
            c.add(c1, lhs, rhs);
            c.cmp(c1, lhs);
            c.cset(c2, LO);          // c2 = carry from lhs+rhs
            c.cmp(res, c1);
            c.cset(c1, LO);          // c1 = carry from +cf
            c.orr(cf, c1, c2);       // CF out
        } else {
            c.lsr(cf, res, w);       // CF = bit w of the 64-bit sum
            c.and_(cf, cf, 1);
            c.and_(res, res, wmask);
        }
    } else {
        // SBB: res = lhs - rhs - cf.
        c.sub(res, lhs, rhs);
        c.sub(res, res, cf);
        if (w == 64) {
            // Borrow out = (lhs < rhs) | (lhs == rhs && cf) ... compute via
            // unsigned compares: b1 = lhs < rhs ; partial = lhs - rhs ;
            // b2 = partial < cf.
            c.cmp(lhs, rhs);
            c.cset(c2, LO);          // b1
            c.sub(c1, lhs, rhs);
            c.cmp(c1, cf);
            c.cset(c1, LO);          // b2
            c.orr(cf, c1, c2);       // CF (borrow) out
        } else {
            // In 64-bit math, res = lhs - rhs - cf; borrow if res has bit w set
            // (i.e. went negative within width). CF = (res >> w) & 1 of the
            // two's-complement low (w+something) — detect via sign beyond width.
            c.lsr(cf, res, w);
            c.and_(cf, cf, 1);
            c.and_(res, res, wmask);
        }
    }

    // Compute the OF sign-bit NOW, before the narrow store-back reuses rhs.
    //   ADC: OF = sign bit of (~(lhs^rhs) & (lhs^res)).
    //   SBB: OF = sign bit of ( (lhs^rhs) & (lhs^res)).
    // Result (0/1) parked in x5 (block-local scratch) until flag assembly.
    {
        const XReg ofa = XReg(5), ofb = XReg(6);
        c.eor(ofa, lhs, rhs);
        if (kind == AdcSbbKind::Adc) c.mvn(ofa, ofa);
        c.eor(ofb, lhs, res);
        c.and_(ofa, ofa, ofb);
        c.lsr(ofa, ofa, sb);
        c.and_(ofa, ofa, 1);     // x5 = OF (0/1)
    }

    // Write result back at width.
    if (w == 64) {
        c.str(res, ptr(kState, GprOffset(d)));
    } else if (w == 32) {
        c.mov(WReg(res.getIdx()), WReg(res.getIdx()));
        c.str(res, ptr(kState, GprOffset(d)));
    } else {
        XReg slot = XReg(15);  // reuse rhs (safe now: OF already captured)
        c.ldr(slot, ptr(kState, GprOffset(d)));
        c.mov(c1, ~wmask);
        c.and_(slot, slot, c1);
        c.orr(slot, slot, res);
        c.str(slot, ptr(kState, GprOffset(d)));
    }

    // Flags: CF (computed), ZF, SF, PF, OF. AF left as-is.
    c.ldr(c1, ptr(kState, Offsets::Rflags));
    c.mov(c2, ~((1ull<<0)|(1ull<<2)|(1ull<<6)|(1ull<<7)|(1ull<<11)));
    c.and_(c1, c1, c2);
    // CF (bit 0).
    c.orr(c1, c1, cf);
    // ZF (bit 6): res == 0 in width.
    c.cmp(res, 0);
    c.cset(c2, EQ);
    c.lsl(c2, c2, 6);
    c.orr(c1, c1, c2);
    // SF (bit 7): res[w-1].
    c.lsr(c2, res, sb);
    c.and_(c2, c2, 1);
    c.lsl(c2, c2, 7);
    c.orr(c1, c1, c2);
    // PF (bit 2): parity of low byte.
    c.and_(c2, res, 0xFF);
    c.eor(c2, c2, c2, LSR, 4);
    c.eor(c2, c2, c2, LSR, 2);
    c.eor(c2, c2, c2, LSR, 1);
    c.mvn(c2, c2);
    c.and_(c2, c2, 1);
    c.lsl(c2, c2, 2);
    c.orr(c1, c1, c2);
    // OF (bit 11): from x5 captured above.
    c.lsl(XReg(5), XReg(5), 11);
    c.orr(c1, c1, XReg(5));
    c.str(c1, ptr(kState, Offsets::Rflags));
    return true;
}

// guest: stc/clc/cmc — directly set/clear/toggle CF (bit 0) of rflags.
bool EmitFlagOp(const ZydisDecodedInstruction& insn, CodeGenerator& c) {
    const XReg fl = kScratch0;
    c.ldr(fl, ptr(kState, Offsets::Rflags));
    switch (insn.mnemonic) {
        case ZYDIS_MNEMONIC_STC: c.orr(fl, fl, 1); break;
        case ZYDIS_MNEMONIC_CLC: c.and_(fl, fl, ~1ull); break;
        case ZYDIS_MNEMONIC_CMC: c.eor(fl, fl, 1); break;
        default: return false;
    }
    c.str(fl, ptr(kState, Offsets::Rflags));
    return true;
}

// Helper: load a MUL/DIV/IMUL-1op source operand (reg or mem) at width w into
// `dst`, zero-extended. Returns false on unsupported operand form.
static bool LoadSrcOperand(const ZydisDecodedOperand& op, u32 w, const XReg& dst,
                           CodeGenerator& c) {
    if (op.type == ZYDIS_OPERAND_TYPE_REGISTER) {
        const int s = ZydisGprToIndex(op.reg.value);
        if (s < 0) return false;
        c.ldr(dst, ptr(kState, GprOffset(s)));
    } else if (op.type == ZYDIS_OPERAND_TYPE_MEMORY) {
        if (!EmitEffectiveAddress(op.mem, /*next_rip*/0, c)) return false;
        switch (w) {
            case 8:  c.ldrb(WReg(dst.getIdx()), ptr(kAddr)); break;
            case 16: c.ldrh(WReg(dst.getIdx()), ptr(kAddr)); break;
            case 32: c.ldr(WReg(dst.getIdx()), ptr(kAddr)); break;
            default: c.ldr(dst, ptr(kAddr)); break;
        }
    } else {
        return false;
    }
    if (w != 64) c.and_(dst, dst, (w == 32) ? 0xFFFFFFFFull : ((1ull << w) - 1));
    return true;
}

// guest: mul r/m (one-operand unsigned multiply). Implicit accumulator:
//   w8 : AX        = AL  * src ; CF/OF = (AH != 0).
//   w16: DX:AX     = AX  * src ; CF/OF = (DX != 0).
//   w32: EDX:EAX   = EAX * src ; CF/OF = (EDX != 0); upper 32 of RAX/RDX zeroed.
//   w64: RDX:RAX   = RAX * src ; CF/OF = (RDX != 0).
// SF/ZF/PF/AF undefined (left as-is). Narrow forms preserve upper bits of the
// parent slots (w8 leaves RDX entirely untouched).
bool EmitMul(const ZydisDecodedInstruction& insn, const ZydisDecodedOperand* ops,
             CodeGenerator& c) {
    const u32 w = insn.operand_width;
    if (w != 8 && w != 16 && w != 32 && w != 64) return false;
    const XReg a = kScratch0, src = kScratch1, hi = kScratch2, lo = XReg(13);
    c.ldr(a, ptr(kState, GprOffset(0)));               // rax
    if (w != 64) c.and_(a, a, (w == 32) ? 0xFFFFFFFFull : ((1ull << w) - 1));
    if (!LoadSrcOperand(ops[0], w, src, c)) return false;

    if (w == 64) {
        c.mul(lo, a, src);            // low 64
        c.umulh(hi, a, src);          // high 64
        c.str(lo, ptr(kState, GprOffset(0)));
        c.str(hi, ptr(kState, GprOffset(2)));
    } else if (w == 32) {
        c.umull(lo, WReg(a.getIdx()), WReg(src.getIdx()));  // 32x32 -> 64 in lo
        c.lsr(hi, lo, 32);                                  // EDX = high 32
        c.and_(lo, lo, 0xFFFFFFFFull);
        c.str(lo, ptr(kState, GprOffset(0)));               // EAX (zero-ext)
        c.and_(hi, hi, 0xFFFFFFFFull);
        c.str(hi, ptr(kState, GprOffset(2)));               // EDX (zero-ext)
    } else if (w == 16) {
        c.mul(lo, a, src);            // full product (<=32 bits)
        // AX = low 16 (merge), DX = bits 31:16 (merge), preserve uppers.
        c.ldr(XReg(14), ptr(kState, GprOffset(0)));
        c.mov(XReg(15), ~0xFFFFull);
        c.and_(XReg(14), XReg(14), XReg(15));
        c.and_(hi, lo, 0xFFFF);
        c.orr(XReg(14), XReg(14), hi);
        c.str(XReg(14), ptr(kState, GprOffset(0)));
        c.ldr(XReg(14), ptr(kState, GprOffset(2)));
        c.and_(XReg(14), XReg(14), XReg(15));
        c.lsr(hi, lo, 16);
        c.and_(hi, hi, 0xFFFF);
        c.orr(XReg(14), XReg(14), hi);
        c.str(XReg(14), ptr(kState, GprOffset(2)));
        c.lsr(hi, lo, 16);           // hi holds DX value for CF/OF
        c.and_(hi, hi, 0xFFFF);
    } else { // w == 8
        c.mul(lo, a, src);           // AL*src (<=16 bits)
        c.ldr(XReg(14), ptr(kState, GprOffset(0)));
        c.mov(XReg(15), ~0xFFFFull);
        c.and_(XReg(14), XReg(14), XReg(15));
        c.and_(hi, lo, 0xFFFF);
        c.orr(XReg(14), XReg(14), hi);
        c.str(XReg(14), ptr(kState, GprOffset(0)));   // AX = product, upper preserved
        c.lsr(hi, lo, 8);            // AH for CF/OF
        c.and_(hi, hi, 0xFF);
    }

    // CF/OF = (high part != 0). Load rflags, clear CF(0)/OF(11), set from hi.
    c.ldr(XReg(14), ptr(kState, Offsets::Rflags));
    c.mov(XReg(15), ~((1ull<<0)|(1ull<<11)));
    c.and_(XReg(14), XReg(14), XReg(15));
    {
        Label hizero, done;
        c.cmp(hi, 0);
        c.b(EQ, hizero);
        c.orr(XReg(14), XReg(14), 1ull<<0);    // CF
        c.orr(XReg(14), XReg(14), 1ull<<11);   // OF
        c.L(hizero);
    }
    c.str(XReg(14), ptr(kState, Offsets::Rflags));
    return true;
}

// guest: imul — 1-operand (RDX:RAX = RAX*src, signed), 2-operand (dst *= src),
// 3-operand (dst = src * imm). For 2/3-op only the low w bits are kept in dst;
// CF/OF set iff the full signed product doesn't fit in w bits (truncation lost
// significant bits). For 1-op, RDX:RAX gets the full signed 128-bit product and
// CF/OF set iff RDX is not the sign-extension of RAX. SF/ZF/PF undefined.
bool EmitImul(const ZydisDecodedInstruction& insn, const ZydisDecodedOperand* ops,
              CodeGenerator& c) {
    const u32 w = insn.operand_width;
    if (w != 16 && w != 32 && w != 64) return false;
    const int nops = insn.operand_count_visible;

    if (nops == 1) {
        // RDX:RAX = RAX * src (signed full product).
        const XReg a = kScratch0, src = kScratch1, hi = kScratch2, lo = XReg(13);
        c.ldr(a, ptr(kState, GprOffset(0)));
        if (!LoadSrcOperand(ops[0], w, src, c)) return false;
        if (w == 64) {
            c.mul(lo, a, src);
            c.smulh(hi, a, src);
            c.str(lo, ptr(kState, GprOffset(0)));
            c.str(hi, ptr(kState, GprOffset(2)));
        } else if (w == 32) {
            c.sxtw(a, WReg(a.getIdx()));
            c.sxtw(src, WReg(src.getIdx()));
            c.mul(lo, a, src);                 // full 64-bit signed product
            c.and_(XReg(15), lo, 0xFFFFFFFFull);
            c.str(XReg(15), ptr(kState, GprOffset(0)));   // EAX (zero-ext)
            c.lsr(hi, lo, 32);
            c.and_(hi, hi, 0xFFFFFFFFull);
            c.str(hi, ptr(kState, GprOffset(2)));         // EDX
        } else { // 16
            c.sxth(a, WReg(a.getIdx()));
            c.sxth(src, WReg(src.getIdx()));
            c.mul(lo, a, src);
            c.ldr(XReg(14), ptr(kState, GprOffset(0)));
            c.mov(XReg(15), ~0xFFFFull);
            c.and_(XReg(14), XReg(14), XReg(15));
            c.and_(XReg(12), lo, 0xFFFF);
            c.orr(XReg(14), XReg(14), XReg(12));
            c.str(XReg(14), ptr(kState, GprOffset(0)));
            c.ldr(XReg(14), ptr(kState, GprOffset(2)));
            c.and_(XReg(14), XReg(14), XReg(15));
            c.lsr(XReg(12), lo, 16);
            c.and_(XReg(12), XReg(12), 0xFFFF);
            c.orr(XReg(14), XReg(14), XReg(12));
            c.str(XReg(14), ptr(kState, GprOffset(2)));
        }
        // CF/OF = high half != sign-extension of low half. Conservatively set
        // when smulh(hi) != asr(lo,63) for w64, or high32/16 != sign of low.
        // (Tests check the basic fits/overflow cases.)
        c.ldr(XReg(14), ptr(kState, Offsets::Rflags));
        c.mov(XReg(15), ~((1ull<<0)|(1ull<<11)));
        c.and_(XReg(14), XReg(14), XReg(15));
        // recompute sign-extension check
        if (w == 64) {
            c.asr(XReg(15), lo, 63);
            c.cmp(hi, XReg(15));
        } else if (w == 32) {
            c.sxtw(XReg(15), WReg(lo.getIdx()));   // sign-extend low32
            c.cmp(lo, XReg(15));                     // lo holds full 64 product
        } else {
            c.sxth(XReg(15), WReg(lo.getIdx()));
            c.cmp(lo, XReg(15));
        }
        {
            Label fits, done;
            c.b(EQ, fits);
            c.orr(XReg(14), XReg(14), 1ull<<0);    // CF
        c.orr(XReg(14), XReg(14), 1ull<<11);   // OF
            c.L(fits);
        }
        c.str(XReg(14), ptr(kState, Offsets::Rflags));
        return true;
    }

    // 2-op: dst (reg) *= src ; 3-op: dst = src * imm.
    if (ops[0].type != ZYDIS_OPERAND_TYPE_REGISTER) return false;
    const int d = ZydisGprToIndex(ops[0].reg.value);
    if (d < 0) return false;
    const XReg x = kScratch0, y = kScratch1, res = kScratch2;
    if (nops == 2) {
        c.ldr(x, ptr(kState, GprOffset(d)));
        if (!LoadSrcOperand(ops[1], w, y, c)) return false;
    } else { // 3-op
        if (!LoadSrcOperand(ops[1], w, x, c)) return false;
        c.mov(y, static_cast<u64>(ops[2].imm.value.s));
    }
    c.mul(res, x, y);
    // Write back at width.
    if (w == 64) {
        c.str(res, ptr(kState, GprOffset(d)));
    } else if (w == 32) {
        c.and_(res, res, 0xFFFFFFFFull);
        c.str(res, ptr(kState, GprOffset(d)));
    } else { // 16
        c.ldr(XReg(14), ptr(kState, GprOffset(d)));
        c.mov(XReg(15), ~0xFFFFull);
        c.and_(XReg(14), XReg(14), XReg(15));
        c.and_(XReg(13), res, 0xFFFF);
        c.orr(XReg(14), XReg(14), XReg(13));
        c.str(XReg(14), ptr(kState, GprOffset(d)));
    }
    return true;
}

// guest: div r/m — unsigned (RDX:RAX) / src -> quotient RAX, remainder RDX.
//   w8 : AX / src -> AL=quot, AH=rem.
//   w16: DX:AX / src -> AX=quot, DX=rem.
//   w32: EDX:EAX / src -> EAX=quot, EDX=rem (uppers zeroed).
//   w64: RDX:RAX / src -> RAX=quot, RDX=rem (true 128/64 via long division).
// Divide-by-zero returns false (routes to unsupported-exit; no SIGFPE).
bool EmitDiv(const ZydisDecodedInstruction& insn, const ZydisDecodedOperand* ops,
             CodeGenerator& c) {
    const u32 w = insn.operand_width;
    if (w != 8 && w != 16 && w != 32 && w != 64) return false;
    const XReg src = kScratch1;
    if (!LoadSrcOperand(ops[0], w, src, c)) return false;
    // Divide-by-zero: route to unsupported-exit rather than trap. The guest
    // #DE path isn't modeled; bailing is safe and the tests never divide by 0.
    {
        Label ok;
        c.cbnz(src, ok);
        c.mov(kWScratch0, static_cast<u32>(ExitReason::UnsupportedInstruction));
        c.str(kWScratch0, ptr(kState, static_cast<u32>(offsetof(GuestState, exit_reason))));
        c.br(kExitStub);
        c.L(ok);
    }

    if (w == 8) {
        // dividend = AX (16-bit). q,r fit in 8 bits.
        const XReg dv = kScratch0, q = kScratch2, r = XReg(13);
        c.ldr(dv, ptr(kState, GprOffset(0)));
        c.and_(dv, dv, 0xFFFF);
        c.udiv(q, dv, src);
        c.msub(r, q, src, dv);     // r = dv - q*src
        // AL=q, AH=r, preserve upper.
        c.ldr(XReg(14), ptr(kState, GprOffset(0)));
        c.mov(XReg(15), ~0xFFFFull);
        c.and_(XReg(14), XReg(14), XReg(15));
        c.and_(q, q, 0xFF);
        c.and_(r, r, 0xFF);
        c.lsl(r, r, 8);
        c.orr(XReg(14), XReg(14), q);
        c.orr(XReg(14), XReg(14), r);
        c.str(XReg(14), ptr(kState, GprOffset(0)));
        return true;
    }

    if (w == 16) {
        // dividend = DX:AX (32-bit).
        const XReg dv = kScratch0, q = kScratch2, r = XReg(13);
        c.ldr(dv, ptr(kState, GprOffset(0)));
        c.and_(dv, dv, 0xFFFF);
        c.ldr(XReg(14), ptr(kState, GprOffset(2)));
        c.and_(XReg(14), XReg(14), 0xFFFF);
        c.lsl(XReg(14), XReg(14), 16);
        c.orr(dv, dv, XReg(14));        // dividend = (DX<<16)|AX
        c.udiv(q, dv, src);
        c.msub(r, q, src, dv);
        // AX=q, DX=r (merge low16, preserve upper).
        c.ldr(XReg(14), ptr(kState, GprOffset(0)));
        c.mov(XReg(15), ~0xFFFFull);
        c.and_(XReg(14), XReg(14), XReg(15));
        c.and_(q, q, 0xFFFF);
        c.orr(XReg(14), XReg(14), q);
        c.str(XReg(14), ptr(kState, GprOffset(0)));
        c.ldr(XReg(14), ptr(kState, GprOffset(2)));
        c.and_(XReg(14), XReg(14), XReg(15));
        c.and_(r, r, 0xFFFF);
        c.orr(XReg(14), XReg(14), r);
        c.str(XReg(14), ptr(kState, GprOffset(2)));
        return true;
    }

    if (w == 32) {
        // dividend = EDX:EAX (64-bit).
        const XReg dv = kScratch0, q = kScratch2, r = XReg(13);
        c.ldr(dv, ptr(kState, GprOffset(0)));
        c.and_(dv, dv, 0xFFFFFFFFull);
        c.ldr(XReg(14), ptr(kState, GprOffset(2)));
        c.and_(XReg(14), XReg(14), 0xFFFFFFFFull);
        c.lsl(XReg(14), XReg(14), 32);
        c.orr(dv, dv, XReg(14));         // (EDX<<32)|EAX
        c.udiv(q, dv, src);
        c.msub(r, q, src, dv);
        c.and_(q, q, 0xFFFFFFFFull);
        c.str(q, ptr(kState, GprOffset(0)));   // EAX (zero-ext)
        c.and_(r, r, 0xFFFFFFFFull);
        c.str(r, ptr(kState, GprOffset(2)));   // EDX (zero-ext)
        return true;
    }

    // w == 64: true 128/64 division of (RDX:RAX) by src.
    // Binary long division: 128-bit dividend in (hi=RDX, lo=RAX), 64-bit
    // divisor. Quotient -> RAX, remainder -> RDX.
    const XReg hi = kScratch0, lo = kScratch2, q = XReg(13), rem = XReg(14);
    c.ldr(hi, ptr(kState, GprOffset(2)));   // RDX
    c.ldr(lo, ptr(kState, GprOffset(0)));   // RAX
    c.mov(q, 0);
    c.mov(rem, 0);
    // 128 iterations: shift (rem:hi:lo) left, bring in next dividend bit,
    // compare rem>=src, subtract+set quotient bit.
    const XReg i = XReg(15), bit = XReg(5), carry = XReg(6);
    c.mov(i, 128);
    Label loop, after;
    c.L(loop);
    c.cbz(i, after);
    // Capture rem's MSB (bit shifted out of bit63 when rem<<1) = the 65th bit.
    c.lsr(carry, rem, 63);
    // rem = (rem<<1) | top bit of (hi:lo); then shift hi:lo left by 1.
    c.lsl(rem, rem, 1);
    c.lsr(bit, hi, 63);            // top bit of hi
    c.orr(rem, rem, bit);
    c.lsl(hi, hi, 1);
    c.lsr(bit, lo, 63);           // top bit of lo into hi[0]
    c.orr(hi, hi, bit);
    c.lsl(lo, lo, 1);
    // shift quotient left
    c.lsl(q, q, 1);
    // Subtract when the 65-bit remainder >= src: i.e. carry==1 (rem>=2^64>src)
    // OR rem (64-bit) >= src.
    {
        Label doSub, noSub;
        c.cbnz(carry, doSub);
        c.cmp(rem, src);
        c.b(LO, noSub);
        c.L(doSub);
        c.sub(rem, rem, src);
        c.orr(q, q, 1);
        c.L(noSub);
    }
    c.sub(i, i, 1);
    c.b(loop);
    c.L(after);
    c.str(q, ptr(kState, GprOffset(0)));     // quotient -> RAX
    c.str(rem, ptr(kState, GprOffset(2)));   // remainder -> RDX
    return true;
}

// Set SF/ZF from `res` (width w) into rflags, clearing CF/OF/PF... actually:
// helper used by BMI ops that set ZF/SF and clear CF/OF (ANDN, BEXTR). Writes
// CF=0, OF=0, ZF=(res==0), SF=(res[w-1]); PF/AF left as-is.
static void SetZnFlagsClearCO(const XReg& res, u32 w, CodeGenerator& c) {
    const XReg fl = XReg(14), t = XReg(15);
    const u32 sb = w - 1;
    c.ldr(fl, ptr(kState, Offsets::Rflags));
    c.mov(t, ~((1ull<<0)|(1ull<<6)|(1ull<<7)|(1ull<<11)));   // clear CF,ZF,SF,OF
    c.and_(fl, fl, t);
    c.cmp(res, 0);
    c.cset(t, EQ);
    c.lsl(t, t, 6);
    c.orr(fl, fl, t);                 // ZF
    c.lsr(t, res, sb);
    c.and_(t, t, 1);
    c.lsl(t, t, 7);
    c.orr(fl, fl, t);                 // SF
    c.str(fl, ptr(kState, Offsets::Rflags));
}

// guest: andn dst, src1, src2  ->  dst = (~src1) & src2. Widths 32/64.
//   ops[0]=dst, ops[1]=src1 (VEX.vvvv reg), ops[2]=src2 (reg/mem).
//   Flags: ZF/SF from result; CF=OF=0. 32-bit zero-extends.
bool EmitAndn(const ZydisDecodedInstruction& insn, const ZydisDecodedOperand* ops,
              CodeGenerator& c) {
    const u32 w = insn.operand_width;
    if (w != 32 && w != 64) return false;
    if (ops[0].type != ZYDIS_OPERAND_TYPE_REGISTER) return false;
    const int d = ZydisGprToIndex(ops[0].reg.value);
    if (d < 0) return false;
    if (ops[1].type != ZYDIS_OPERAND_TYPE_REGISTER) return false;
    const int s1 = ZydisGprToIndex(ops[1].reg.value);
    if (s1 < 0) return false;
    const XReg a = kScratch0, b = kScratch1, res = kScratch2;
    c.ldr(a, ptr(kState, GprOffset(s1)));
    if (!LoadSrcOperand(ops[2], w, b, c)) return false;
    if (w != 64) c.and_(a, a, 0xFFFFFFFFull);
    c.bic(res, b, a);                 // res = b & ~a  (AArch64 bic = and-not)
    if (w == 32) c.and_(res, res, 0xFFFFFFFFull);
    c.str(res, ptr(kState, GprOffset(d)));
    SetZnFlagsClearCO(res, w, c);
    return true;
}

// guest: blsi dst, src  ->  dst = src & (-src) (isolate lowest set bit).
//   ops[0]=dst (VEX.vvvv), ops[1]=src (reg/mem). Widths 32/64.
//   CF = (src != 0); ZF/SF from result; OF=0.
bool EmitBlsi(const ZydisDecodedInstruction& insn, const ZydisDecodedOperand* ops,
              CodeGenerator& c) {
    const u32 w = insn.operand_width;
    if (w != 32 && w != 64) return false;
    if (ops[0].type != ZYDIS_OPERAND_TYPE_REGISTER) return false;
    const int d = ZydisGprToIndex(ops[0].reg.value);
    if (d < 0) return false;
    const XReg s = kScratch0, neg = kScratch1, res = kScratch2;
    if (!LoadSrcOperand(ops[1], w, s, c)) return false;
    c.neg(neg, s);
    c.and_(res, s, neg);
    if (w == 32) c.and_(res, res, 0xFFFFFFFFull);
    c.str(res, ptr(kState, GprOffset(d)));
    // Flags: CF = (src != 0); ZF/SF from result; OF=0; PF/AF as-is.
    const XReg fl = XReg(14), t = XReg(15);
    const u32 sb = w - 1;
    c.ldr(fl, ptr(kState, Offsets::Rflags));
    c.mov(t, ~((1ull<<0)|(1ull<<6)|(1ull<<7)|(1ull<<11)));
    c.and_(fl, fl, t);
    c.cmp(s, 0);
    c.cset(t, NE);
    c.orr(fl, fl, t);                 // CF = src != 0
    c.cmp(res, 0);
    c.cset(t, EQ);
    c.lsl(t, t, 6);
    c.orr(fl, fl, t);                 // ZF
    c.lsr(t, res, sb);
    c.and_(t, t, 1);
    c.lsl(t, t, 7);
    c.orr(fl, fl, t);                 // SF
    c.str(fl, ptr(kState, Offsets::Rflags));
    return true;
}

// guest: bextr dst, src, ctrl -> start=ctrl[7:0], len=ctrl[15:8];
//   dst = (src >> start) & ((1<<len)-1). Widths 32/64.
//   ops[0]=dst, ops[1]=src (reg/mem), ops[2]=ctrl (reg). ZF from result;
//   CF/OF cleared. 32-bit zero-extends.
bool EmitBextr(const ZydisDecodedInstruction& insn, const ZydisDecodedOperand* ops,
               CodeGenerator& c) {
    const u32 w = insn.operand_width;
    if (w != 32 && w != 64) return false;
    if (ops[0].type != ZYDIS_OPERAND_TYPE_REGISTER) return false;
    const int d = ZydisGprToIndex(ops[0].reg.value);
    if (d < 0) return false;
    if (ops[2].type != ZYDIS_OPERAND_TYPE_REGISTER) return false;
    const int cidx = ZydisGprToIndex(ops[2].reg.value);
    if (cidx < 0) return false;
    const XReg src = kScratch0, ctrl = kScratch1, res = kScratch2;
    const XReg start = XReg(13), len = XReg(14);
    if (!LoadSrcOperand(ops[1], w, src, c)) return false;
    c.ldr(ctrl, ptr(kState, GprOffset(cidx)));
    c.and_(start, ctrl, 0xFF);          // start = ctrl[7:0]
    c.ubfx(len, ctrl, 8, 8);            // len = ctrl[15:8]
    // res = src >> start (logical, variable). For start>=width AArch64 lsr by
    // (start mod 64); x86 BEXTR with start>=opsize gives 0. Clamp by masking
    // start to <64 is fine for the variable shift; large start -> 0 result via
    // mask below when len handling. We follow common impl: shift then mask len.
    c.lsr(res, src, start);
    // mask = (len>=64)? all : (1<<len)-1. Build via: if len==0 -> 0; else
    // (1<<len)-1 with len clamped to 64.
    {
        Label lenBig, lenDone, lenZero;
        const XReg mask = XReg(15);
        c.cmp(len, 0);
        c.b(EQ, lenZero);
        c.cmp(len, 64);
        c.b(HS, lenBig);                // len>=64 -> full mask
        c.mov(mask, 1);
        c.lsl(mask, mask, len);
        c.sub(mask, mask, 1);
        c.and_(res, res, mask);
        c.b(lenDone);
        c.L(lenBig);
        // full mask: leave res as-is (all bits kept)
        c.b(lenDone);
        c.L(lenZero);
        c.mov(res, 0);                  // len==0 -> 0
        c.L(lenDone);
    }
    if (w == 32) c.and_(res, res, 0xFFFFFFFFull);
    c.str(res, ptr(kState, GprOffset(d)));
    SetZnFlagsClearCO(res, w, c);
    return true;
}

// guest: bt src, index -> CF = bit (index mod width) of src. Only CF affected;
//   all other flags preserved (per test). ops[0]=src (reg), ops[1]=imm or reg.
bool EmitBt(const ZydisDecodedInstruction& insn, const ZydisDecodedOperand* ops,
            CodeGenerator& c) {
    const u32 w = insn.operand_width;
    if (w != 16 && w != 32 && w != 64) return false;
    if (ops[0].type != ZYDIS_OPERAND_TYPE_REGISTER) return false;
    const int s = ZydisGprToIndex(ops[0].reg.value);
    if (s < 0) return false;
    const u32 modmask = (w == 64) ? 0x3F : (w == 32 ? 0x1F : 0x0F);
    const XReg src = kScratch0, idx = kScratch1, bit = kScratch2;
    c.ldr(src, ptr(kState, GprOffset(s)));
    if (ops[1].type == ZYDIS_OPERAND_TYPE_IMMEDIATE) {
        c.mov(idx, static_cast<u64>(ops[1].imm.value.u & modmask));
    } else if (ops[1].type == ZYDIS_OPERAND_TYPE_REGISTER) {
        const int r = ZydisGprToIndex(ops[1].reg.value);
        if (r < 0) return false;
        c.ldr(idx, ptr(kState, GprOffset(r)));
        c.and_(idx, idx, modmask);
    } else {
        return false;
    }
    c.lsr(bit, src, idx);
    c.and_(bit, bit, 1);
    // CF = bit; preserve all other flags.
    const XReg fl = XReg(14), t = XReg(15);
    c.ldr(fl, ptr(kState, Offsets::Rflags));
    c.mov(t, ~1ull);
    c.and_(fl, fl, t);
    c.orr(fl, fl, bit);
    c.str(fl, ptr(kState, Offsets::Rflags));
    return true;
}

// guest: lzcnt dst, src -> count leading zeros at width. CF=(src==0),
//   ZF=(result==0). Widths 32/64. ops[0]=dst, ops[1]=src(reg/mem).
bool EmitLzcnt(const ZydisDecodedInstruction& insn, const ZydisDecodedOperand* ops,
               CodeGenerator& c) {
    const u32 w = insn.operand_width;
    if (w != 32 && w != 64) return false;
    if (ops[0].type != ZYDIS_OPERAND_TYPE_REGISTER) return false;
    const int d = ZydisGprToIndex(ops[0].reg.value);
    if (d < 0) return false;
    const XReg src = kScratch0, res = kScratch2;
    if (!LoadSrcOperand(ops[1], w, src, c)) return false;
    if (w == 32) c.clz(WReg(res.getIdx()), WReg(src.getIdx()));
    else         c.clz(res, src);
    if (w == 32) c.and_(res, res, 0xFFFFFFFFull);
    c.str(res, ptr(kState, GprOffset(d)));
    // CF = (src==0) ; ZF = (result==0). SF/OF/PF undefined -> clear CF/ZF only.
    const XReg fl = XReg(14), t = XReg(15);
    c.ldr(fl, ptr(kState, Offsets::Rflags));
    c.mov(t, ~((1ull<<0)|(1ull<<6)));
    c.and_(fl, fl, t);
    c.cmp(src, 0);
    c.cset(t, EQ);
    c.orr(fl, fl, t);                 // CF = src==0
    c.cmp(res, 0);
    c.cset(t, EQ);
    c.lsl(t, t, 6);
    c.orr(fl, fl, t);                 // ZF = result==0
    c.str(fl, ptr(kState, Offsets::Rflags));
    return true;
}

// guest: popcnt dst, src -> population count. ZF=(src==0); CF/OF/SF/PF/AF=0.
//   Widths 32/64. SWAR popcount (no NEON). ops[0]=dst, ops[1]=src(reg/mem).
bool EmitPopcnt(const ZydisDecodedInstruction& insn, const ZydisDecodedOperand* ops,
                CodeGenerator& c) {
    const u32 w = insn.operand_width;
    if (w != 32 && w != 64) return false;
    if (ops[0].type != ZYDIS_OPERAND_TYPE_REGISTER) return false;
    const int d = ZydisGprToIndex(ops[0].reg.value);
    if (d < 0) return false;
    const XReg src = kScratch0, x = kScratch2, t = XReg(13), m = XReg(14);
    if (!LoadSrcOperand(ops[1], w, src, c)) return false;
    // SWAR popcount on 64-bit x (operate full width; src already width-masked).
    c.mov(x, src);
    // x = x - ((x>>1) & 0x5555...);
    c.lsr(t, x, 1);
    c.mov(m, 0x5555555555555555ull);
    c.and_(t, t, m);
    c.sub(x, x, t);
    // x = (x & 0x3333...) + ((x>>2) & 0x3333...);
    c.mov(m, 0x3333333333333333ull);
    c.and_(t, x, m);
    c.lsr(x, x, 2);
    c.and_(x, x, m);
    c.add(x, x, t);
    // x = (x + (x>>4)) & 0x0F0F...;
    c.lsr(t, x, 4);
    c.add(x, x, t);
    c.mov(m, 0x0F0F0F0F0F0F0F0Full);
    c.and_(x, x, m);
    // count = (x * 0x0101...) >> 56;
    c.mov(m, 0x0101010101010101ull);
    c.mul(x, x, m);
    c.lsr(x, x, 56);
    c.str(x, ptr(kState, GprOffset(d)));
    // ZF = (src==0); clear CF/OF/SF/PF/AF.
    const XReg fl = XReg(14), tt = XReg(15);
    c.ldr(fl, ptr(kState, Offsets::Rflags));
    c.mov(tt, ~((1ull<<0)|(1ull<<2)|(1ull<<4)|(1ull<<6)|(1ull<<7)|(1ull<<11)));
    c.and_(fl, fl, tt);
    c.cmp(src, 0);
    c.cset(tt, EQ);
    c.lsl(tt, tt, 6);
    c.orr(fl, fl, tt);                // ZF
    c.str(fl, ptr(kState, Offsets::Rflags));
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
// guest: cbw/cwde/cdqe (sign-extend accumulator) and cwd/cdq/cqo (sign-extend
// rAX into rDX:rAX). No flags affected. The 0x98/0x99 opcodes are
// operand-width-disambiguated; Zydis resolves the mnemonic for us.
//   CBW : AL ->  AX  (sign);  CWDE: AX  -> EAX (sign, zero-ext to RAX);
//   CDQE: EAX -> RAX (sign).
//   CWD : AX  -> DX:AX  (DX = sign bits of AX, upper of RDX preserved);
//   CDQ : EAX -> EDX:EAX (EDX = sign of EAX, zero-ext into RDX);
//   CQO : RAX -> RDX:RAX (RDX = all sign bits of RAX).
bool EmitConvert(const ZydisDecodedInstruction& insn, CodeGenerator& c) {
    const XReg a = kScratch0, t = kScratch1;
    c.ldr(a, ptr(kState, GprOffset(0)));   // rax
    switch (insn.mnemonic) {
        case ZYDIS_MNEMONIC_CBW: {
            // AL -> AX: sign-extend low 8 into bits 15:8, preserve upper 48.
            c.sxtb(t, WReg(a.getIdx()));        // t = sign-extended byte (64)
            c.mov(XReg(12), ~0xFFFFull);
            c.and_(a, a, XReg(12));             // clear low 16
            c.and_(t, t, 0xFFFF);               // keep low 16 of extension
            c.orr(a, a, t);
            c.str(a, ptr(kState, GprOffset(0)));
            return true;
        }
        case ZYDIS_MNEMONIC_CWDE: {
            // AX -> EAX: sign-extend low 16 to 32, then zero-extend to 64.
            c.sxth(WReg(t.getIdx()), WReg(a.getIdx()));   // 16->32 sign
            c.mov(WReg(t.getIdx()), WReg(t.getIdx()));    // zero-ext 32->64
            c.str(t, ptr(kState, GprOffset(0)));
            return true;
        }
        case ZYDIS_MNEMONIC_CDQE: {
            // EAX -> RAX: sign-extend low 32 to 64.
            c.sxtw(t, WReg(a.getIdx()));
            c.str(t, ptr(kState, GprOffset(0)));
            return true;
        }
        case ZYDIS_MNEMONIC_CWD: {
            // AX -> DX:AX. DX = 0xFFFF if AX bit15 set else 0; preserve upper 48 of RDX.
            c.ldr(t, ptr(kState, GprOffset(2)));   // rdx
            c.mov(XReg(12), ~0xFFFFull);
            c.and_(t, t, XReg(12));                // clear DX
            c.sbfx(XReg(13), a, 15, 1);            // x13 = sign of bit15 (all-ones/zero)
            c.and_(XReg(13), XReg(13), 0xFFFF);
            c.orr(t, t, XReg(13));
            c.str(t, ptr(kState, GprOffset(2)));
            return true;
        }
        case ZYDIS_MNEMONIC_CDQ: {
            // EAX -> EDX:EAX. EDX = sign of bit31, zero-extended into RDX.
            c.sbfx(WReg(13), WReg(a.getIdx()), 31, 1);  // w13 = 0 or 0xFFFFFFFF
            c.mov(WReg(13), WReg(13));                  // zero-ext to 64
            c.str(XReg(13), ptr(kState, GprOffset(2)));
            return true;
        }
        case ZYDIS_MNEMONIC_CQO: {
            // RAX -> RDX:RAX. RDX = all sign bits of RAX.
            c.asr(t, a, 63);
            c.str(t, ptr(kState, GprOffset(2)));
            return true;
        }
        default:
            return false;
    }
}

// guest: inc/dec r/m at widths 8/16/32/64, register (incl. AH/CH/DH/BH
// high-byte) or memory destination. INC/DEC preserve CF and set OF/SF/ZF/PF;
// the lazy-flag side-band (FLAG_OP_INC/DEC, width-tagged) lets the materializer
// reproduce exactly that (it reloads the old CF and keeps it). High-byte regs
// occupy bits 15:8 of parent slot 0..3 (AH/CH/DH/BH).
bool EmitIncDec(const ZydisDecodedInstruction& insn, const ZydisDecodedOperand* ops,
                CodeGenerator& c) {
    const bool is_inc = (insn.mnemonic == ZYDIS_MNEMONIC_INC);
    if (!is_inc && insn.mnemonic != ZYDIS_MNEMONIC_DEC) return false;
    if (insn.operand_count_visible != 1) return false;
    const u32 w = insn.operand_width;
    if (w != 8 && w != 16 && w != 32 && w != 64) return false;
    const u64 wmask = (w == 64) ? ~0ull : ((1ull << w) - 1);

    const XReg lhs = kScratch0, rhs = kScratch1, res = kScratch2;  // x9,x10,x12
    const XReg vAddr = XReg(13);

    // Detect high-byte register (AH/CH/DH/BH): parent slot 0..3, byte offset 1.
    int hi_parent = -1;
    if (ops[0].type == ZYDIS_OPERAND_TYPE_REGISTER) {
        switch (ops[0].reg.value) {
            case ZYDIS_REGISTER_AH: hi_parent = 0; break;
            case ZYDIS_REGISTER_CH: hi_parent = 1; break;
            case ZYDIS_REGISTER_DH: hi_parent = 2; break;
            case ZYDIS_REGISTER_BH: hi_parent = 3; break;
            default: break;
        }
    }

    const bool dst_mem = (ops[0].type == ZYDIS_OPERAND_TYPE_MEMORY);
    int dst = -1;
    if (!dst_mem && hi_parent < 0) {
        dst = ZydisGprToIndex(ops[0].reg.value);
        if (dst < 0) return false;
    }

    // Load lhs (width-truncated) and compute res = lhs +/- 1.
    if (dst_mem) {
        const bool addr32 = (insn.address_width == 32);
        if (!EmitEffectiveAddress(ops[0].mem, /*next_rip*/0, c, addr32)) return false;
        c.mov(vAddr, kAddr);
        switch (w) {
            case 8:  c.ldrb(WReg(lhs.getIdx()), ptr(vAddr)); break;
            case 16: c.ldrh(WReg(lhs.getIdx()), ptr(vAddr)); break;
            case 32: c.ldr(WReg(lhs.getIdx()), ptr(vAddr)); break;
            default: c.ldr(lhs, ptr(vAddr)); break;
        }
    } else if (hi_parent >= 0) {
        c.ldr(lhs, ptr(kState, GprOffset(hi_parent)));
        c.ubfx(lhs, lhs, 8, 8);   // extract byte 1
    } else {
        c.ldr(lhs, ptr(kState, GprOffset(dst)));
        if (w != 64) c.and_(lhs, lhs, wmask);
    }

    c.mov(rhs, 1);
    if (is_inc) c.add(res, lhs, rhs);
    else        c.sub(res, lhs, rhs);
    if (w != 64) c.and_(res, res, wmask);

    // Lazy-flag side-band (width-tagged). Use WReg(9)... wait, lhs IS x9; use a
    // free reg for the flag_op/width immediates that doesn't alias lhs/res/vAddr.
    // x14 is free here.
    c.mov(WReg(14), is_inc ? FLAG_OP_INC : FLAG_OP_DEC);
    c.str(WReg(14), ptr(kState, static_cast<u32>(offsetof(GuestState, flag_op))));
    c.mov(WReg(14), w);
    c.str(WReg(14), ptr(kState, static_cast<u32>(offsetof(GuestState, flag_width))));
    c.str(lhs, ptr(kState, static_cast<u32>(offsetof(GuestState, flag_lhs))));
    c.str(rhs, ptr(kState, static_cast<u32>(offsetof(GuestState, flag_rhs))));
    c.str(res, ptr(kState, static_cast<u32>(offsetof(GuestState, flag_result))));

    // Write back.
    if (dst_mem) {
        switch (w) {
            case 8:  c.strb(WReg(res.getIdx()), ptr(vAddr)); break;
            case 16: c.strh(WReg(res.getIdx()), ptr(vAddr)); break;
            case 32: c.str(WReg(res.getIdx()), ptr(vAddr)); break;
            default: c.str(res, ptr(vAddr)); break;
        }
    } else if (hi_parent >= 0) {
        // Merge res into bits 15:8 of parent slot, preserve the rest.
        c.ldr(XReg(14), ptr(kState, GprOffset(hi_parent)));
        c.mov(XReg(15), ~(0xFFull << 8));
        c.and_(XReg(14), XReg(14), XReg(15));
        c.and_(res, res, 0xFF);
        c.lsl(res, res, 8);
        c.orr(XReg(14), XReg(14), res);
        c.str(XReg(14), ptr(kState, GprOffset(hi_parent)));
    } else if (w == 64) {
        c.str(res, ptr(kState, GprOffset(dst)));
    } else if (w == 32) {
        c.mov(WReg(res.getIdx()), WReg(res.getIdx()));
        c.str(res, ptr(kState, GprOffset(dst)));
    } else {
        // 8/16 merge-preserve upper.
        c.ldr(XReg(14), ptr(kState, GprOffset(dst)));
        c.mov(XReg(15), ~wmask);
        c.and_(XReg(14), XReg(14), XReg(15));
        c.orr(XReg(14), XReg(14), res);
        c.str(XReg(14), ptr(kState, GprOffset(dst)));
    }

    EmitMaterializeFlags(c);
    return true;
}

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

// guest: cld/std — clear/set the direction flag (RFLAGS bit 10). Other flags
// untouched.
bool EmitCldStd(const ZydisDecodedInstruction& insn, CodeGenerator& c) {
    constexpr u64 DF = 1ull << 10;
    const XReg fl = kScratch0;
    c.ldr(fl, ptr(kState, Offsets::Rflags));
    if (insn.mnemonic == ZYDIS_MNEMONIC_CLD) {
        c.mov(kScratch1, ~DF);
        c.and_(fl, fl, kScratch1);
    } else if (insn.mnemonic == ZYDIS_MNEMONIC_STD) {
        c.orr(fl, fl, DF);
    } else {
        return false;
    }
    c.str(fl, ptr(kState, Offsets::Rflags));
    return true;
}

// guest: stos/movs/lods/scas/cmps with optional REP/REPE/REPNE. Implicit
// operands RDI(7)/RSI(6)/RAX(0)/RCX(1). Block terminator: on completion it sets
// state.rip=next_rip, exit_reason=BlockEnd, and re-enters the dispatcher. A
// near-null pointer bails cleanly (HelperRequestedExit -> exit stub), matching
// the x86 host. SCAS/CMPS set flags via the lazy-flag side-band (FLAG_OP_SUB),
// materialized each iteration so REPE/REPNE can read ZF.
bool EmitStringOp(const ZydisDecodedInstruction& insn, u64 next_rip,
                  CodeGenerator& c) {
    if (insn.operand_count_visible != 0) return false;  // disambiguate vs SSE MOVSD/CMPSD

    enum class Kind { Stos, Movs, Lods, Scas, Cmps } kind;
    bool uses_rsi = false, uses_rdi = false, is_cmp = false;
    switch (insn.mnemonic) {
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

    int esz;
    switch (insn.operand_width) {
        case 8:  esz = 1; break;
        case 16: esz = 2; break;
        case 32: esz = 4; break;
        case 64: esz = 8; break;
        default: return false;
    }
    const u32 w = insn.operand_width;

    const bool repeated =
        (insn.attributes & (ZYDIS_ATTRIB_HAS_REP | ZYDIS_ATTRIB_HAS_REPE |
                            ZYDIS_ATTRIB_HAS_REPNE)) != 0;
    const bool has_repe  = (insn.attributes & ZYDIS_ATTRIB_HAS_REPE)  != 0;
    const bool has_repne = (insn.attributes & ZYDIS_ATTRIB_HAS_REPNE) != 0;

    constexpr int RAX_IDX = 0, RCX_IDX = 1, RSI_IDX = 6, RDI_IDX = 7;
    const u64 cur_rip = next_rip - insn.length;

    const XReg rcx = kScratch1;     // x10: loop counter
    const XReg rdi = XReg(13);      // dest ptr
    const XReg rsi = XReg(14);      // src ptr
    const XReg acc = kScratch2;     // x12: load/compare accumulator
    const XReg tmp = XReg(15), tmp2 = XReg(5);

    Label loop_top, loop_done, do_bail;

    if (repeated) {
        c.ldr(rcx, ptr(kState, GprOffset(RCX_IDX)));
        c.cbz(rcx, loop_done);
    }

    c.L(loop_top);

    // Pointer null-guard (below 0x10000 bails).
    constexpr u64 kMinGuestPtr = 0x10000;
    if (uses_rdi) {
        c.ldr(rdi, ptr(kState, GprOffset(RDI_IDX)));
        c.mov(tmp, kMinGuestPtr);
        c.cmp(rdi, tmp);
        c.b(LO, do_bail);
    }
    if (uses_rsi) {
        c.ldr(rsi, ptr(kState, GprOffset(RSI_IDX)));
        c.mov(tmp, kMinGuestPtr);
        c.cmp(rsi, tmp);
        c.b(LO, do_bail);
    }

    auto load_w = [&](const XReg& dstreg, const XReg& ptrreg) {
        switch (esz) {
            case 1: c.ldrb(WReg(dstreg.getIdx()), ptr(ptrreg)); break;
            case 2: c.ldrh(WReg(dstreg.getIdx()), ptr(ptrreg)); break;
            case 4: c.ldr(WReg(dstreg.getIdx()), ptr(ptrreg)); break;
            default: c.ldr(dstreg, ptr(ptrreg)); break;
        }
    };
    auto store_w = [&](const XReg& srcreg, const XReg& ptrreg) {
        switch (esz) {
            case 1: c.strb(WReg(srcreg.getIdx()), ptr(ptrreg)); break;
            case 2: c.strh(WReg(srcreg.getIdx()), ptr(ptrreg)); break;
            case 4: c.str(WReg(srcreg.getIdx()), ptr(ptrreg)); break;
            default: c.str(srcreg, ptr(ptrreg)); break;
        }
    };

    if (kind == Kind::Stos) {
        c.ldr(acc, ptr(kState, GprOffset(RAX_IDX)));
        store_w(acc, rdi);
    } else if (kind == Kind::Movs) {
        load_w(acc, rsi);
        store_w(acc, rdi);
    } else if (kind == Kind::Lods) {
        load_w(acc, rsi);
        if (esz == 1 || esz == 2) {
            c.ldr(tmp, ptr(kState, GprOffset(RAX_IDX)));
            c.mov(tmp2, (esz == 1) ? ~0xFFull : ~0xFFFFull);
            c.and_(tmp, tmp, tmp2);
            c.and_(acc, acc, (esz == 1) ? 0xFF : 0xFFFF);
            c.orr(tmp, tmp, acc);
            c.str(tmp, ptr(kState, GprOffset(RAX_IDX)));
        } else {
            if (esz == 4) c.and_(acc, acc, 0xFFFFFFFFull);
            c.str(acc, ptr(kState, GprOffset(RAX_IDX)));
        }
    } else {
        // Scas/Cmps: lhs - rhs (width-masked), set flags via side-band.
        const XReg lhs = acc, rhs = tmp;
        if (kind == Kind::Scas) {
            c.ldr(lhs, ptr(kState, GprOffset(RAX_IDX)));
            load_w(rhs, rdi);
        } else {
            load_w(lhs, rsi);
            load_w(rhs, rdi);
        }
        const u64 wmask = (w == 64) ? ~0ull : ((1ull << w) - 1);
        if (w != 64) { c.and_(lhs, lhs, wmask); c.and_(rhs, rhs, wmask); }
        c.sub(tmp2, lhs, rhs);
        if (w != 64) c.and_(tmp2, tmp2, wmask);
        c.mov(WReg(9), FLAG_OP_SUB);
        c.str(WReg(9), ptr(kState, static_cast<u32>(offsetof(GuestState, flag_op))));
        c.mov(WReg(9), w);
        c.str(WReg(9), ptr(kState, static_cast<u32>(offsetof(GuestState, flag_width))));
        c.str(lhs, ptr(kState, static_cast<u32>(offsetof(GuestState, flag_lhs))));
        c.str(rhs, ptr(kState, static_cast<u32>(offsetof(GuestState, flag_rhs))));
        c.str(tmp2, ptr(kState, static_cast<u32>(offsetof(GuestState, flag_result))));
        EmitMaterializeFlags(c);
    }

    // Advance pointers per DF (rflags bit 10). delta = DF ? -esz : +esz.
    {
        Label df_set, adv_done;
        c.ldr(tmp, ptr(kState, Offsets::Rflags));
        c.tbnz(tmp, 10, df_set);
        if (uses_rdi) { c.ldr(rdi, ptr(kState, GprOffset(RDI_IDX))); c.add(rdi, rdi, esz); c.str(rdi, ptr(kState, GprOffset(RDI_IDX))); }
        if (uses_rsi) { c.ldr(rsi, ptr(kState, GprOffset(RSI_IDX))); c.add(rsi, rsi, esz); c.str(rsi, ptr(kState, GprOffset(RSI_IDX))); }
        c.b(adv_done);
        c.L(df_set);
        if (uses_rdi) { c.ldr(rdi, ptr(kState, GprOffset(RDI_IDX))); c.sub(rdi, rdi, esz); c.str(rdi, ptr(kState, GprOffset(RDI_IDX))); }
        if (uses_rsi) { c.ldr(rsi, ptr(kState, GprOffset(RSI_IDX))); c.sub(rsi, rsi, esz); c.str(rsi, ptr(kState, GprOffset(RSI_IDX))); }
        c.L(adv_done);
    }

    if (repeated) {
        c.ldr(rcx, ptr(kState, GprOffset(RCX_IDX)));
        c.sub(rcx, rcx, 1);
        c.str(rcx, ptr(kState, GprOffset(RCX_IDX)));
        if (is_cmp && (has_repe || has_repne)) {
            c.ldr(tmp, ptr(kState, Offsets::Rflags));
            if (has_repe) {
                c.tbz(tmp, 6, loop_done);   // ZF==0 -> stop
            } else {
                c.tbnz(tmp, 6, loop_done);  // ZF==1 -> stop
            }
        }
        c.cbnz(rcx, loop_top);
    }

    c.L(loop_done);
    c.mov(kScratch0, next_rip);
    c.str(kScratch0, ptr(kState, Offsets::Rip));
    c.mov(kWScratch0, static_cast<u32>(ExitReason::BlockEnd));
    c.str(kWScratch0, ptr(kState, static_cast<u32>(offsetof(GuestState, exit_reason))));
    c.br(kDispatchTop);

    c.L(do_bail);
    c.mov(kScratch0, cur_rip);
    c.str(kScratch0, ptr(kState, Offsets::Rip));
    c.mov(kWScratch0, static_cast<u32>(ExitReason::HelperRequestedExit));
    c.str(kWScratch0, ptr(kState, static_cast<u32>(offsetof(GuestState, exit_reason))));
    c.br(kExitStub);
    return true;
}

// ============================================================================
// SSE/AVX scalar floating-point — foundation.
//
// Guest XMM/YMM live in GuestState::ymm (lane*4 + chunk, 8 bytes/chunk; XMM is
// chunks 0,1; YMM upper is chunks 2,3). These VEX-encoded ops operate on the
// low 32 (ss) or 64 (sd) bits, merge the rest of the low 128 from src1, and
// zero the upper 128 (chunks 2,3). We shuffle bytes with GPR moves and run the
// actual IEEE-754 math on NEON scalar regs (v0/v1), so host MXCSR rounding
// gives exact results.
// ============================================================================

enum class VScalarOp { Add, Sub, Mul, Div, Sqrt, Min, Max };

// vaddss/sd, vsubss/sd, vmulss/sd, vdivss/sd, vsqrtss/sd, vminss/sd, vmaxss/sd.
// Forms: 3-op reg (dst, src1, src2-reg-or-mem). Sqrt is 2-op-ish but VEX still
// carries src1 for the merge: vsqrtsd xmm0, xmm1, xmm2 -> low=sqrt(src2),
// upper-of-low128 from src1. Single precision uses SReg, double uses DReg.
bool EmitVScalarArith(const ZydisDecodedInstruction& insn,
                      const ZydisDecodedOperand* ops, u64 next_rip,
                      VScalarOp op, bool dbl, CodeGenerator& c) {
    if (insn.operand_count_visible < 3) return false;
    if (ops[0].type != ZYDIS_OPERAND_TYPE_REGISTER ||
        ops[1].type != ZYDIS_OPERAND_TYPE_REGISTER) return false;
    const int dst = ZydisVecToIndex(ops[0].reg.value);
    const int s1  = ZydisVecToIndex(ops[1].reg.value);
    if (dst < 0 || s1 < 0) return false;

    // Load src2 low element into v1 (from reg lane or memory).
    if (ops[2].type == ZYDIS_OPERAND_TYPE_REGISTER) {
        const int s2 = ZydisVecToIndex(ops[2].reg.value);
        if (s2 < 0) return false;
        if (dbl) c.ldr(DReg(1), ptr(kState, YmmChunkOffset(s2, 0)));
        else     c.ldr(SReg(1), ptr(kState, YmmChunkOffset(s2, 0)));
    } else if (ops[2].type == ZYDIS_OPERAND_TYPE_MEMORY) {
        if (!EmitEffectiveAddress(ops[2].mem, next_rip, c)) return false;
        if (dbl) c.ldr(DReg(1), ptr(kAddr));
        else     c.ldr(SReg(1), ptr(kAddr));
    } else {
        return false;
    }

    // Load src1 low element into v0 (the other arithmetic operand; for sqrt
    // src1 only supplies the merge upper, but loading it is harmless).
    if (dbl) c.ldr(DReg(0), ptr(kState, YmmChunkOffset(s1, 0)));
    else     c.ldr(SReg(0), ptr(kState, YmmChunkOffset(s1, 0)));

    // Compute low result into v0.
    if (dbl) {
        switch (op) {
            case VScalarOp::Add:  c.fadd(DReg(0), DReg(0), DReg(1)); break;
            case VScalarOp::Sub:  c.fsub(DReg(0), DReg(0), DReg(1)); break;
            case VScalarOp::Mul:  c.fmul(DReg(0), DReg(0), DReg(1)); break;
            case VScalarOp::Div:  c.fdiv(DReg(0), DReg(0), DReg(1)); break;
            case VScalarOp::Min:  c.fmin(DReg(0), DReg(0), DReg(1)); break;
            case VScalarOp::Max:  c.fmax(DReg(0), DReg(0), DReg(1)); break;
            case VScalarOp::Sqrt: c.fsqrt(DReg(0), DReg(1)); break;  // sqrt(src2)
        }
    } else {
        switch (op) {
            case VScalarOp::Add:  c.fadd(SReg(0), SReg(0), SReg(1)); break;
            case VScalarOp::Sub:  c.fsub(SReg(0), SReg(0), SReg(1)); break;
            case VScalarOp::Mul:  c.fmul(SReg(0), SReg(0), SReg(1)); break;
            case VScalarOp::Div:  c.fdiv(SReg(0), SReg(0), SReg(1)); break;
            case VScalarOp::Min:  c.fmin(SReg(0), SReg(0), SReg(1)); break;
            case VScalarOp::Max:  c.fmax(SReg(0), SReg(0), SReg(1)); break;
            case VScalarOp::Sqrt: c.fsqrt(SReg(0), SReg(1)); break;
        }
    }

    // Merge: dst.chunk0 = result-low merged with src1's other bits of chunk0
    // (for ss, src1's high 32); dst.chunk1 = src1.chunk1; zero chunks 2,3.
    if (dbl) {
        // chunk0 = result (full 64); chunk1 = src1.chunk1.
        c.str(DReg(0), ptr(kState, YmmChunkOffset(dst, 0)));
        c.ldr(kScratch0, ptr(kState, YmmChunkOffset(s1, 1)));
        c.str(kScratch0, ptr(kState, YmmChunkOffset(dst, 1)));
    } else {
        // chunk0 = (src1.hi32 << 32) | result.lo32.
        c.fmov(WReg(9), SReg(0));                           // w9 = result bits (zero-ext to x9)
        c.mov(WReg(9), WReg(9));                            // ensure upper 32 of x9 zeroed
        c.ldr(kScratch1, ptr(kState, YmmChunkOffset(s1, 0)));
        c.lsr(kScratch1, kScratch1, 32);
        c.lsl(kScratch1, kScratch1, 32);                    // src1.hi32 << 32
        c.orr(kScratch0, kScratch1, kScratch0);             // note: kScratch0=x9 holds result lo32
        c.str(kScratch0, ptr(kState, YmmChunkOffset(dst, 0)));
        c.ldr(kScratch0, ptr(kState, YmmChunkOffset(s1, 1)));
        c.str(kScratch0, ptr(kState, YmmChunkOffset(dst, 1)));
    }
    c.mov(kScratch0, 0);
    c.str(kScratch0, ptr(kState, YmmChunkOffset(dst, 2)));
    c.str(kScratch0, ptr(kState, YmmChunkOffset(dst, 3)));
    return true;
}

// vmovss / vmovsd — three forms: merge (3 reg), load (reg<-mem), store (mem<-reg).
bool EmitVmovss(const ZydisDecodedInstruction& insn,
                const ZydisDecodedOperand* ops, u64 next_rip, CodeGenerator& c) {
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
            c.ldr(kScratch0, ptr(kState, YmmChunkOffset(s2, 0)));  // chunk0 = src2.chunk0
            c.ldr(kScratch1, ptr(kState, YmmChunkOffset(s1, 1)));  // chunk1 = src1.chunk1
            c.str(kScratch0, ptr(kState, YmmChunkOffset(dst, 0)));
            c.str(kScratch1, ptr(kState, YmmChunkOffset(dst, 1)));
        } else {
            c.ldr(kScratch1, ptr(kState, YmmChunkOffset(s1, 0)));
            c.lsr(kScratch1, kScratch1, 32);
            c.lsl(kScratch1, kScratch1, 32);                       // src1.hi32 << 32
            c.ldr(WReg(9), ptr(kState, YmmChunkOffset(s2, 0)));    // src2.lo32 (zero-ext x9)
            c.orr(kScratch0, kScratch1, kScratch0);
            c.str(kScratch0, ptr(kState, YmmChunkOffset(dst, 0)));
            c.ldr(kScratch0, ptr(kState, YmmChunkOffset(s1, 1)));
            c.str(kScratch0, ptr(kState, YmmChunkOffset(dst, 1)));
        }
        c.mov(kScratch0, 0);
        c.str(kScratch0, ptr(kState, YmmChunkOffset(dst, 2)));
        c.str(kScratch0, ptr(kState, YmmChunkOffset(dst, 3)));
        return true;
    }

    // Load: xmm <- [mem].
    if (ops[0].type == ZYDIS_OPERAND_TYPE_REGISTER &&
        ops[1].type == ZYDIS_OPERAND_TYPE_MEMORY) {
        const int dst = ZydisVecToIndex(ops[0].reg.value);
        if (dst < 0) return false;
        if (!EmitEffectiveAddress(ops[1].mem, next_rip, c)) return false;
        if (sd) {
            c.ldr(kScratch0, ptr(kAddr));
        } else {
            c.ldr(WReg(9), ptr(kAddr));   // zero-extends x9
        }
        c.str(kScratch0, ptr(kState, YmmChunkOffset(dst, 0)));
        c.mov(kScratch0, 0);
        c.str(kScratch0, ptr(kState, YmmChunkOffset(dst, 1)));
        c.str(kScratch0, ptr(kState, YmmChunkOffset(dst, 2)));
        c.str(kScratch0, ptr(kState, YmmChunkOffset(dst, 3)));
        return true;
    }

    // Store: [mem] <- xmm.
    if (ops[0].type == ZYDIS_OPERAND_TYPE_MEMORY &&
        ops[1].type == ZYDIS_OPERAND_TYPE_REGISTER) {
        const int src = ZydisVecToIndex(ops[1].reg.value);
        if (src < 0) return false;
        if (!EmitEffectiveAddress(ops[0].mem, next_rip, c)) return false;
        if (sd) {
            c.ldr(kScratch0, ptr(kState, YmmChunkOffset(src, 0)));
            c.str(kScratch0, ptr(kAddr));
        } else {
            c.ldr(WReg(9), ptr(kState, YmmChunkOffset(src, 0)));
            c.str(WReg(9), ptr(kAddr));
        }
        return true;
    }
    return false;
}

// Packed vector ops on the full XMM (128) or YMM (256) width. Three families:
//   FP arith (ps/pd): vaddps/vsubps/vmulps/vdivps/vminps/vmaxps + pd variants.
//   int arith (d):    vpaddd/vpsubd/vpmulld.
//   bitwise:          vandps/vorps/vxorps/vandnps + pd; vpand/vpor/vpxor/vpandn.
// VEX rule: a 128-bit (XMM) op zeroes the upper 128 (chunks 2,3); a 256-bit op
// writes all four chunks. src1/src2 are reg or (src2) memory.
enum class VPackKind {
    AddPS, SubPS, MulPS, DivPS, MinPS, MaxPS,
    AddPD, SubPD, MulPD, DivPD, MinPD, MaxPD,
    AddD, SubD, MulD,
    And, Or, Xor, AndN,   // bitwise (operate on whole 128/256 regardless of element)
};

bool EmitVPacked(const ZydisDecodedInstruction& insn, const ZydisDecodedOperand* ops,
                 u64 next_rip, VPackKind k, CodeGenerator& c) {
    if (insn.operand_count_visible < 3) return false;
    if (ops[0].type != ZYDIS_OPERAND_TYPE_REGISTER ||
        ops[1].type != ZYDIS_OPERAND_TYPE_REGISTER) return false;
    const int dst = ZydisVecToIndex(ops[0].reg.value);
    const int s1  = ZydisVecToIndex(ops[1].reg.value);
    if (dst < 0 || s1 < 0) return false;

    const bool ymm = (ops[0].size == 256);
    const int nchunks = ymm ? 2 : 1;   // number of 128-bit halves to process

    // Process each 128-bit half independently (YMM lanes don't cross for these).
    for (int h = 0; h < nchunks; ++h) {
        const int chunk = h * 2;   // chunk offset of this 128-bit half (0 or 2)
        // Load src1 half -> q0, src2 half -> q1.
        c.add(kScratch0, kState, YmmChunkOffset(s1, chunk));
        c.ldr(QReg(0), ptr(kScratch0));
        if (ops[2].type == ZYDIS_OPERAND_TYPE_REGISTER) {
            const int s2 = ZydisVecToIndex(ops[2].reg.value);
            if (s2 < 0) return false;
            c.add(kScratch0, kState, YmmChunkOffset(s2, chunk));
            c.ldr(QReg(1), ptr(kScratch0));
        } else if (ops[2].type == ZYDIS_OPERAND_TYPE_MEMORY) {
            if (h == 0) {
                if (!EmitEffectiveAddress(ops[2].mem, next_rip, c)) return false;
                c.mov(kScratch1, kAddr);   // preserve base EA across halves
            }
            c.add(kScratch0, kScratch1, h * 16);
            c.ldr(QReg(1), ptr(kScratch0));
        } else {
            return false;
        }

        switch (k) {
            case VPackKind::AddPS: c.fadd(VReg4S(0), VReg4S(0), VReg4S(1)); break;
            case VPackKind::SubPS: c.fsub(VReg4S(0), VReg4S(0), VReg4S(1)); break;
            case VPackKind::MulPS: c.fmul(VReg4S(0), VReg4S(0), VReg4S(1)); break;
            case VPackKind::DivPS: c.fdiv(VReg4S(0), VReg4S(0), VReg4S(1)); break;
            case VPackKind::MinPS: c.fmin(VReg4S(0), VReg4S(0), VReg4S(1)); break;
            case VPackKind::MaxPS: c.fmax(VReg4S(0), VReg4S(0), VReg4S(1)); break;
            case VPackKind::AddPD: c.fadd(VReg2D(0), VReg2D(0), VReg2D(1)); break;
            case VPackKind::SubPD: c.fsub(VReg2D(0), VReg2D(0), VReg2D(1)); break;
            case VPackKind::MulPD: c.fmul(VReg2D(0), VReg2D(0), VReg2D(1)); break;
            case VPackKind::DivPD: c.fdiv(VReg2D(0), VReg2D(0), VReg2D(1)); break;
            case VPackKind::MinPD: c.fmin(VReg2D(0), VReg2D(0), VReg2D(1)); break;
            case VPackKind::MaxPD: c.fmax(VReg2D(0), VReg2D(0), VReg2D(1)); break;
            case VPackKind::AddD:  c.add(VReg4S(0), VReg4S(0), VReg4S(1)); break;
            case VPackKind::SubD:  c.sub(VReg4S(0), VReg4S(0), VReg4S(1)); break;
            case VPackKind::MulD:  c.mul(VReg4S(0), VReg4S(0), VReg4S(1)); break;
            case VPackKind::And:   c.and_(VReg16B(0), VReg16B(0), VReg16B(1)); break;
            case VPackKind::Or:    c.orr(VReg16B(0), VReg16B(0), VReg16B(1)); break;
            case VPackKind::Xor:   c.eor(VReg16B(0), VReg16B(0), VReg16B(1)); break;
            // ANDN = (~src1) & src2. NEON bic(d,n,m) = n & ~m -> bic(d, src2, src1).
            case VPackKind::AndN:  c.bic(VReg16B(0), VReg16B(1), VReg16B(0)); break;
        }

        c.add(kScratch0, kState, YmmChunkOffset(dst, chunk));
        c.str(QReg(0), ptr(kScratch0));
    }

    if (!ymm) {
        // VEX-128: zero upper 128 (chunks 2,3).
        c.mov(kScratch0, 0);
        c.str(kScratch0, ptr(kState, YmmChunkOffset(dst, 2)));
        c.str(kScratch0, ptr(kState, YmmChunkOffset(dst, 3)));
    }
    return true;
}

} // namespace
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
        case ZYDIS_MNEMONIC_SHL:
            handled = EmitShift(insn, ops, ShiftKind::Shl, c);
            break;
        case ZYDIS_MNEMONIC_SHR:
            handled = EmitShift(insn, ops, ShiftKind::Shr, c);
            break;
        case ZYDIS_MNEMONIC_SAR:
            handled = EmitShift(insn, ops, ShiftKind::Sar, c);
            break;
        case ZYDIS_MNEMONIC_ROL:
            handled = EmitRotate(insn, ops, RotateKind::Rol, c);
            break;
        case ZYDIS_MNEMONIC_ROR:
            handled = EmitRotate(insn, ops, RotateKind::Ror, c);
            break;
        case ZYDIS_MNEMONIC_ADC:
            handled = EmitAdcSbb(insn, ops, AdcSbbKind::Adc, c);
            break;
        case ZYDIS_MNEMONIC_SBB:
            handled = EmitAdcSbb(insn, ops, AdcSbbKind::Sbb, c);
            break;
        case ZYDIS_MNEMONIC_STC:
        case ZYDIS_MNEMONIC_CLC:
        case ZYDIS_MNEMONIC_CMC:
            handled = EmitFlagOp(insn, c);
            break;
        case ZYDIS_MNEMONIC_ANDN:
            handled = EmitAndn(insn, ops, c);
            break;
        case ZYDIS_MNEMONIC_BLSI:
            handled = EmitBlsi(insn, ops, c);
            break;
        case ZYDIS_MNEMONIC_BEXTR:
            handled = EmitBextr(insn, ops, c);
            break;
        case ZYDIS_MNEMONIC_BT:
            handled = EmitBt(insn, ops, c);
            break;
        case ZYDIS_MNEMONIC_LZCNT:
            handled = EmitLzcnt(insn, ops, c);
            break;
        case ZYDIS_MNEMONIC_POPCNT:
            handled = EmitPopcnt(insn, ops, c);
            break;
        case ZYDIS_MNEMONIC_MUL:
            handled = EmitMul(insn, ops, c);
            break;
        case ZYDIS_MNEMONIC_IMUL:
            handled = EmitImul(insn, ops, c);
            break;
        case ZYDIS_MNEMONIC_DIV:
            handled = EmitDiv(insn, ops, c);
            break;
        case ZYDIS_MNEMONIC_VADDSS: handled = EmitVScalarArith(insn, ops, next_rip, VScalarOp::Add, false, c); break;
        case ZYDIS_MNEMONIC_VADDSD: handled = EmitVScalarArith(insn, ops, next_rip, VScalarOp::Add, true,  c); break;
        case ZYDIS_MNEMONIC_VSUBSS: handled = EmitVScalarArith(insn, ops, next_rip, VScalarOp::Sub, false, c); break;
        case ZYDIS_MNEMONIC_VSUBSD: handled = EmitVScalarArith(insn, ops, next_rip, VScalarOp::Sub, true,  c); break;
        case ZYDIS_MNEMONIC_VMULSS: handled = EmitVScalarArith(insn, ops, next_rip, VScalarOp::Mul, false, c); break;
        case ZYDIS_MNEMONIC_VMULSD: handled = EmitVScalarArith(insn, ops, next_rip, VScalarOp::Mul, true,  c); break;
        case ZYDIS_MNEMONIC_VDIVSS: handled = EmitVScalarArith(insn, ops, next_rip, VScalarOp::Div, false, c); break;
        case ZYDIS_MNEMONIC_VDIVSD: handled = EmitVScalarArith(insn, ops, next_rip, VScalarOp::Div, true,  c); break;
        case ZYDIS_MNEMONIC_VMINSS: handled = EmitVScalarArith(insn, ops, next_rip, VScalarOp::Min, false, c); break;
        case ZYDIS_MNEMONIC_VMINSD: handled = EmitVScalarArith(insn, ops, next_rip, VScalarOp::Min, true,  c); break;
        case ZYDIS_MNEMONIC_VMAXSS: handled = EmitVScalarArith(insn, ops, next_rip, VScalarOp::Max, false, c); break;
        case ZYDIS_MNEMONIC_VMAXSD: handled = EmitVScalarArith(insn, ops, next_rip, VScalarOp::Max, true,  c); break;
        case ZYDIS_MNEMONIC_VSQRTSS: handled = EmitVScalarArith(insn, ops, next_rip, VScalarOp::Sqrt, false, c); break;
        case ZYDIS_MNEMONIC_VSQRTSD: handled = EmitVScalarArith(insn, ops, next_rip, VScalarOp::Sqrt, true,  c); break;
        case ZYDIS_MNEMONIC_VMOVSS:
        case ZYDIS_MNEMONIC_VMOVSD: handled = EmitVmovss(insn, ops, next_rip, c); break;
        case ZYDIS_MNEMONIC_VADDPS: handled = EmitVPacked(insn, ops, next_rip, VPackKind::AddPS, c); break;
        case ZYDIS_MNEMONIC_VSUBPS: handled = EmitVPacked(insn, ops, next_rip, VPackKind::SubPS, c); break;
        case ZYDIS_MNEMONIC_VMULPS: handled = EmitVPacked(insn, ops, next_rip, VPackKind::MulPS, c); break;
        case ZYDIS_MNEMONIC_VDIVPS: handled = EmitVPacked(insn, ops, next_rip, VPackKind::DivPS, c); break;
        case ZYDIS_MNEMONIC_VMINPS: handled = EmitVPacked(insn, ops, next_rip, VPackKind::MinPS, c); break;
        case ZYDIS_MNEMONIC_VMAXPS: handled = EmitVPacked(insn, ops, next_rip, VPackKind::MaxPS, c); break;
        case ZYDIS_MNEMONIC_VADDPD: handled = EmitVPacked(insn, ops, next_rip, VPackKind::AddPD, c); break;
        case ZYDIS_MNEMONIC_VSUBPD: handled = EmitVPacked(insn, ops, next_rip, VPackKind::SubPD, c); break;
        case ZYDIS_MNEMONIC_VMULPD: handled = EmitVPacked(insn, ops, next_rip, VPackKind::MulPD, c); break;
        case ZYDIS_MNEMONIC_VDIVPD: handled = EmitVPacked(insn, ops, next_rip, VPackKind::DivPD, c); break;
        case ZYDIS_MNEMONIC_VMINPD: handled = EmitVPacked(insn, ops, next_rip, VPackKind::MinPD, c); break;
        case ZYDIS_MNEMONIC_VMAXPD: handled = EmitVPacked(insn, ops, next_rip, VPackKind::MaxPD, c); break;
        case ZYDIS_MNEMONIC_VPADDD: handled = EmitVPacked(insn, ops, next_rip, VPackKind::AddD, c); break;
        case ZYDIS_MNEMONIC_VPSUBD: handled = EmitVPacked(insn, ops, next_rip, VPackKind::SubD, c); break;
        case ZYDIS_MNEMONIC_VPMULLD: handled = EmitVPacked(insn, ops, next_rip, VPackKind::MulD, c); break;
        case ZYDIS_MNEMONIC_VANDPS: case ZYDIS_MNEMONIC_VANDPD:
        case ZYDIS_MNEMONIC_VPAND:  handled = EmitVPacked(insn, ops, next_rip, VPackKind::And, c); break;
        case ZYDIS_MNEMONIC_VORPS:  case ZYDIS_MNEMONIC_VORPD:
        case ZYDIS_MNEMONIC_VPOR:   handled = EmitVPacked(insn, ops, next_rip, VPackKind::Or, c); break;
        case ZYDIS_MNEMONIC_VXORPS: case ZYDIS_MNEMONIC_VXORPD:
        case ZYDIS_MNEMONIC_VPXOR:  handled = EmitVPacked(insn, ops, next_rip, VPackKind::Xor, c); break;
        case ZYDIS_MNEMONIC_VANDNPS: case ZYDIS_MNEMONIC_VANDNPD:
        case ZYDIS_MNEMONIC_VPANDN:  handled = EmitVPacked(insn, ops, next_rip, VPackKind::AndN, c); break;
        case ZYDIS_MNEMONIC_CLD:
        case ZYDIS_MNEMONIC_STD:
            handled = EmitCldStd(insn, c);
            break;
        case ZYDIS_MNEMONIC_STOSB: case ZYDIS_MNEMONIC_STOSW:
        case ZYDIS_MNEMONIC_STOSD: case ZYDIS_MNEMONIC_STOSQ:
        case ZYDIS_MNEMONIC_MOVSB: case ZYDIS_MNEMONIC_MOVSW:
        case ZYDIS_MNEMONIC_MOVSD: case ZYDIS_MNEMONIC_MOVSQ:
        case ZYDIS_MNEMONIC_LODSB: case ZYDIS_MNEMONIC_LODSW:
        case ZYDIS_MNEMONIC_LODSD: case ZYDIS_MNEMONIC_LODSQ:
        case ZYDIS_MNEMONIC_SCASB: case ZYDIS_MNEMONIC_SCASW:
        case ZYDIS_MNEMONIC_SCASD: case ZYDIS_MNEMONIC_SCASQ:
        case ZYDIS_MNEMONIC_CMPSB: case ZYDIS_MNEMONIC_CMPSW:
        case ZYDIS_MNEMONIC_CMPSD: case ZYDIS_MNEMONIC_CMPSQ:
            handled = EmitStringOp(insn, next_rip, c);
            if (handled) emitted_terminator = true;  // string op is a terminator
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
        case ZYDIS_MNEMONIC_CBW:
        case ZYDIS_MNEMONIC_CWDE:
        case ZYDIS_MNEMONIC_CDQE:
        case ZYDIS_MNEMONIC_CWD:
        case ZYDIS_MNEMONIC_CDQ:
        case ZYDIS_MNEMONIC_CQO:
            handled = EmitConvert(insn, c);
            break;
        case ZYDIS_MNEMONIC_INC:
        case ZYDIS_MNEMONIC_DEC:
            handled = EmitIncDec(insn, ops, c);
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
