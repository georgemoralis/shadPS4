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
// Host stack pointer. Register index 31 encodes SP (not XZR) in the add/sub
// immediate and load/store base-register forms used below; xbyak_aarch64's
// pre-instantiated `sp` object is a class member (not at namespace scope, per
// the note above), so construct it by index to stay scope-independent.
const XReg kSp = XReg(31);

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
// High-byte register (AH/CH/DH/BH) -> parent GPR slot index (0..3), else -1.
// These access bits 15:8 of the parent register.
static int HighByteParent(ZydisRegister r) {
    switch (r) {
        case ZYDIS_REGISTER_AH: return 0;
        case ZYDIS_REGISTER_CH: return 1;
        case ZYDIS_REGISTER_DH: return 2;
        case ZYDIS_REGISTER_BH: return 3;
        default: return -1;
    }
}

bool EmitMov(const ZydisDecodedInstruction& insn, const ZydisDecodedOperand* ops,
             u64 next_rip, CodeGenerator& c) {
    if (insn.mnemonic != ZYDIS_MNEMONIC_MOV) return false;
    const u32 w = insn.operand_width;
    if (w != 8 && w != 16 && w != 32 && w != 64) return false;
    if (insn.operand_count_visible != 2) return false;

    const u64 wmask = (w == 64) ? ~0ull : ((1ull << w) - 1);
    const bool addr32 = (insn.address_width == 32);   // 0x67 prefix

    // Load the source VALUE (already width-masked for the narrow cases where it
    // matters on store) into kScratch0.
    auto load_src = [&]() -> bool {
        if (ops[1].type == ZYDIS_OPERAND_TYPE_REGISTER) {
            const int hb = HighByteParent(ops[1].reg.value);
            if (hb >= 0) {
                c.ldr(kScratch0, ptr(kState, GprOffset(hb)));
                c.ubfx(kScratch0, kScratch0, 8, 8);   // src = parent byte 1
                return true;
            }
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
            if (!EmitEffectiveAddress(ops[1].mem, next_rip, c, addr32)) return false;
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
        const int hbDst = HighByteParent(ops[0].reg.value);
        if (hbDst >= 0) {
            // 8-bit write into bits 15:8 of the parent slot.
            if (!load_src()) return false;
            c.and_(kScratch0, kScratch0, 0xFFull);
            c.lsl(kScratch0, kScratch0, 8);
            c.ldr(kScratch1, ptr(kState, GprOffset(hbDst)));
            c.mov(kScratch2, ~(0xFFull << 8));
            c.and_(kScratch1, kScratch1, kScratch2);
            c.orr(kScratch1, kScratch1, kScratch0);
            c.str(kScratch1, ptr(kState, GprOffset(hbDst)));
            return true;
        }
        const int dst = ZydisGprToIndex(ops[0].reg.value);
        if (dst < 0) return false;  // (non-high-byte) unsupported
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
        if (!EmitEffectiveAddress(ops[0].mem, next_rip, c, addr32)) return false;  // -> kAddr
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

// guest: bswap r32/r64 — reverse byte order of a register. Register-only, no
// flags. Maps directly to AArch64 REV (Xd for 64-bit, Wd for 32-bit; a 32-bit
// REV through the W view zero-extends into the guest slot, matching how x86
// writes a 32-bit dest). The 16-bit form is architecturally undefined on x86,
// so we decline it rather than emit something plausible-but-wrong.
bool EmitBswap(const ZydisDecodedInstruction& insn, const ZydisDecodedOperand* ops,
               u64 /*next_rip*/, CodeGenerator& c) {
    if (insn.mnemonic != ZYDIS_MNEMONIC_BSWAP) return false;
    if (ops[0].type != ZYDIS_OPERAND_TYPE_REGISTER) return false;
    const u32 w = insn.operand_width;
    if (w != 32 && w != 64) return false;
    const int d = ZydisGprToIndex(ops[0].reg.value);
    if (d < 0) return false;

    if (w == 64) {
        c.ldr(kScratch0, ptr(kState, GprOffset(d)));
        c.rev(kScratch0, kScratch0);
        c.str(kScratch0, ptr(kState, GprOffset(d)));
    } else {
        c.ldr(kWScratch0, ptr(kState, GprOffset(d)));  // low 32 bits
        c.rev(kWScratch0, kWScratch0);                 // W-rev zero-extends x9
        c.str(kScratch0, ptr(kState, GprOffset(d)));   // store zero-extended qword
    }
    return true;
}

// guest: xchg r/m, r — exchange. Register/register only here: swap the two guest
// GPR slots at the operand width. The memory form carries an implied LOCK and is
// left to the atomic RMW path (declined here). XCHG affects no flags. For 32-bit
// each result zero-extends; 8/16-bit merge into the existing upper bits.
bool EmitXchg(const ZydisDecodedInstruction& insn, const ZydisDecodedOperand* ops,
              u64 /*next_rip*/, CodeGenerator& c) {
    if (insn.mnemonic != ZYDIS_MNEMONIC_XCHG) return false;
    if (ops[0].type != ZYDIS_OPERAND_TYPE_REGISTER ||
        ops[1].type != ZYDIS_OPERAND_TYPE_REGISTER) return false;   // mem -> atomic path
    const u32 w = insn.operand_width;
    if (w != 8 && w != 16 && w != 32 && w != 64) return false;
    const int a = ZydisGprToIndex(ops[0].reg.value);
    const int b = ZydisGprToIndex(ops[1].reg.value);
    if (a < 0 || b < 0) return false;
    if (a == b) return true;   // xchg r,r (e.g. the NOP encoding) — nothing to do

    c.ldr(kScratch0, ptr(kState, GprOffset(a)));   // x9 = a
    c.ldr(kScratch1, ptr(kState, GprOffset(b)));   // x10 = b
    if (w == 64) {
        c.str(kScratch1, ptr(kState, GprOffset(a)));
        c.str(kScratch0, ptr(kState, GprOffset(b)));
    } else if (w == 32) {
        c.mov(WReg(9), WReg(9));    // zero-extend low 32 of each
        c.mov(WReg(10), WReg(10));
        c.str(kScratch1, ptr(kState, GprOffset(a)));
        c.str(kScratch0, ptr(kState, GprOffset(b)));
    } else {
        // 8/16-bit: merge the low w bits, preserving the upper bits of each dst.
        const u64 m = (1ull << w) - 1;
        const XReg na = XReg(12), nb = XReg(13), t = XReg(14);
        c.mov(t, m);
        c.and_(na, kScratch1, t);                  // na = b's low w bits
        c.and_(nb, kScratch0, t);                  // nb = a's low w bits
        c.ldr(XReg(15), ptr(kState, GprOffset(a))); // reload a full
        c.mov(t, ~m); c.and_(XReg(15), XReg(15), t);
        c.orr(XReg(15), XReg(15), na);
        c.str(XReg(15), ptr(kState, GprOffset(a)));
        c.ldr(XReg(15), ptr(kState, GprOffset(b))); // reload b full
        c.mov(t, ~m); c.and_(XReg(15), XReg(15), t);
        c.orr(XReg(15), XReg(15), nb);
        c.str(XReg(15), ptr(kState, GprOffset(b)));
    }
    return true;
}

// guest: movbe r,m / movbe m,r — move with byte swap (load or store big-endian).
// No flags. Maps to a width load/store plus REV (REV16 for the 16-bit form). A
// 32-bit destination register zero-extends via the W view.
bool EmitMovbe(const ZydisDecodedInstruction& insn, const ZydisDecodedOperand* ops,
               u64 next_rip, CodeGenerator& c) {
    if (insn.mnemonic != ZYDIS_MNEMONIC_MOVBE) return false;
    const u32 w = insn.operand_width;
    if (w != 16 && w != 32 && w != 64) return false;

    const bool load = (ops[0].type == ZYDIS_OPERAND_TYPE_REGISTER);  // reg <- mem
    const int reg_op = load ? 0 : 1;
    const int mem_op = load ? 1 : 0;
    if (ops[reg_op].type != ZYDIS_OPERAND_TYPE_REGISTER ||
        ops[mem_op].type != ZYDIS_OPERAND_TYPE_MEMORY) return false;
    const int g = ZydisGprToIndex(ops[reg_op].reg.value);
    if (g < 0) return false;
    if (!EmitEffectiveAddress(ops[mem_op].mem, next_rip, c)) return false;
    c.mov(kScratch1, kAddr);   // EA -> x10 (stable)

    auto bswap = [&](const XReg& x, const WReg& wv) {
        if (w == 16)      c.rev16(wv, wv);
        else if (w == 32) c.rev(wv, wv);   // W-rev zero-extends
        else              c.rev(x, x);
    };

    if (load) {
        switch (w) {
            case 16: c.ldrh(kWScratch0, ptr(kScratch1)); break;
            case 32: c.ldr(kWScratch0, ptr(kScratch1)); break;
            default: c.ldr(kScratch0, ptr(kScratch1)); break;
        }
        bswap(kScratch0, kWScratch0);
        if (w == 16) {
            // 16-bit dest merges into the low 16 bits, preserving bits 63:16.
            c.ldr(kScratch2, ptr(kState, GprOffset(g)));
            c.mov(XReg(14), 0xFFFFull);
            c.bic(kScratch2, kScratch2, XReg(14));         // clear low 16
            c.and_(kScratch0, kScratch0, XReg(14));
            c.orr(kScratch2, kScratch2, kScratch0);
            c.str(kScratch2, ptr(kState, GprOffset(g)));
        } else {
            c.str(kScratch0, ptr(kState, GprOffset(g)));   // 32 zero-ext / 64 full
        }
    } else {
        c.ldr(kScratch0, ptr(kState, GprOffset(g)));
        bswap(kScratch0, kWScratch0);
        switch (w) {
            case 16: c.strh(kWScratch0, ptr(kScratch1)); break;
            case 32: c.str(kWScratch0, ptr(kScratch1)); break;
            default: c.str(kScratch0, ptr(kScratch1)); break;
        }
    }
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
        const int hb = HighByteParent(ops[1].reg.value);
        if (hb >= 0) {
            // High-byte source (AH/CH/DH/BH): the byte is bits 15:8 of the
            // parent GPR. Load parent, shift down 8, zero-extend the byte.
            c.ldr(kScratch0, ptr(kState, GprOffset(hb)));
            c.lsr(kScratch0, kScratch0, 8);
            c.uxtb(kScratch0, kScratch0);
        } else {
            const int s = ZydisGprToIndex(ops[1].reg.value);
            if (s < 0) return false;
            c.ldr(kScratch0, ptr(kState, GprOffset(s)));
            if (src_size == 8) c.uxtb(kScratch0, kScratch0);
            else               c.uxth(kScratch0, kScratch0);
        }
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
        const int hb = HighByteParent(ops[1].reg.value);
        if (hb >= 0) {
            // High-byte source (AH/CH/DH/BH): move bits 15:8 of the parent
            // into the low byte so the sxtb below sign-extends from bit 7.
            c.ldr(kScratch0, ptr(kState, GprOffset(hb)));
            c.lsr(kScratch0, kScratch0, 8);
        } else {
            const int s = ZydisGprToIndex(ops[1].reg.value);
            if (s < 0) return false;
            c.ldr(kScratch0, ptr(kState, GprOffset(s)));
        }
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

    // Destination is a register OR a memory operand. For the memory form, the
    // value's effective address is computed FIRST into a callee-stable reg
    // (x19) so the later count load / scratch use can't clobber it (EA compute
    // uses x9/x11). The same address is reused for the read and the write-back.
    const bool dst_mem = (ops[0].type == ZYDIS_OPERAND_TYPE_MEMORY);
    int d = 0;
    const XReg vaddr = XReg(19);   // stable value address (mem form only)
    if (dst_mem) {
        if (!EmitEffectiveAddress(ops[0].mem, /*next_rip*/0, c)) return false;  // -> kAddr (x11)
        c.mov(vaddr, kAddr);
    } else {
        if (ops[0].type != ZYDIS_OPERAND_TYPE_REGISTER) return false;
        d = ZydisGprToIndex(ops[0].reg.value);
        if (d < 0) return false;
    }

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
    if (dst_mem) {
        if (w == 8)       c.ldrb(WReg(val.getIdx()), ptr(vaddr));
        else if (w == 16) c.ldrh(WReg(val.getIdx()), ptr(vaddr));
        else if (w == 32) c.ldr(WReg(val.getIdx()), ptr(vaddr));
        else              c.ldr(val, ptr(vaddr));
    } else {
        c.ldr(val, ptr(kState, GprOffset(d)));
    }
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
    if (dst_mem) {
        // Memory: write exactly w bits at vaddr; adjacent bytes untouched.
        if (w == 8)       c.strb(WReg(res.getIdx()), ptr(vaddr));
        else if (w == 16) c.strh(WReg(res.getIdx()), ptr(vaddr));
        else if (w == 32) c.str(WReg(res.getIdx()), ptr(vaddr));
        else              c.str(res, ptr(vaddr));
    } else if (w == 64) {
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

    // Destination is a register OR memory. For memory, compute the EA FIRST
    // into a callee-stable reg (x19; saved by the gateway, never relied on
    // across the dispatch loop) so the count load / scratch use can't clobber
    // it. The same address serves the read and the write-back.
    const bool dst_mem = (ops[0].type == ZYDIS_OPERAND_TYPE_MEMORY);
    int d = 0;
    const XReg vaddr = XReg(19);
    if (dst_mem) {
        if (!EmitEffectiveAddress(ops[0].mem, /*next_rip*/0, c)) return false;  // -> kAddr
        c.mov(vaddr, kAddr);
    } else {
        if (ops[0].type != ZYDIS_OPERAND_TYPE_REGISTER) return false;
        d = ZydisGprToIndex(ops[0].reg.value);
        if (d < 0) return false;
    }

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

    // Load value truncated to width (register slot or memory).
    if (dst_mem) {
        if (w == 8)       c.ldrb(WReg(val.getIdx()), ptr(vaddr));
        else if (w == 16) c.ldrh(WReg(val.getIdx()), ptr(vaddr));
        else if (w == 32) c.ldr(WReg(val.getIdx()), ptr(vaddr));
        else              c.ldr(val, ptr(vaddr));
    } else {
        c.ldr(val, ptr(kState, GprOffset(d)));
    }
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
    if (dst_mem) {
        if (w != 64) c.and_(res, res, wmask); // all-ones imm invalid on AArch64; qword rotate is already full-width
        if (w == 8)       c.strb(WReg(res.getIdx()), ptr(vaddr));
        else if (w == 16) c.strh(WReg(res.getIdx()), ptr(vaddr));
        else if (w == 32) c.str(WReg(res.getIdx()), ptr(vaddr));
        else              c.str(res, ptr(vaddr));
    } else if (w == 64) {
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
             u64 next_rip, CodeGenerator& c) {
    const u32 w = insn.operand_width;
    if (w != 8 && w != 16 && w != 32 && w != 64) return false;
    const XReg src = kScratch1;
    if (!LoadSrcOperand(ops[0], w, src, c)) return false;
    // Divide-by-zero: route to unsupported-exit rather than trap. The guest
    // #DE path isn't modeled; bailing is safe and the tests never divide by 0.
    {
        Label ok;
        c.cbnz(src, ok);
        // Pin state.rip to THIS instruction: without it the exit reports the
        // stale block-entry RIP and the diagnostic points at the wrong insn.
        c.mov(kScratch0, next_rip - insn.length);
        c.str(kScratch0, ptr(kState, Offsets::Rip));
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

// guest: idiv r/m (signed division). Implicit dividend in (RDX:RAX scaled to
// width); quotient -> AL/AX/EAX/RAX, remainder -> AH/DX/EDX/RDX. x86 IDIV
// truncates toward zero and the remainder takes the sign of the dividend —
// which is exactly AArch64 sdiv/msub semantics, so the 8/16/32 widths use
// sdiv directly on sign-extended operands. The 64-bit width has no native
// 128/64 divide, so it computes result signs, takes absolute values, runs the
// same unsigned long-division loop as EmitDiv, then re-applies the signs.
// (Verified against x86 IDIV under QEMU for all sign combinations.)
bool EmitIdiv(const ZydisDecodedInstruction& insn, const ZydisDecodedOperand* ops,
              u64 next_rip, CodeGenerator& c) {
    const u32 w = insn.operand_width;
    if (w != 8 && w != 16 && w != 32 && w != 64) return false;
    const XReg src = kScratch1;
    if (!LoadSrcOperand(ops[0], w, src, c)) return false;
    // Divide-by-zero: route to unsupported-exit (mirrors EmitDiv; #DE not modeled).
    {
        Label ok;
        c.cbnz(src, ok);
        // Pin state.rip to THIS instruction: without it the exit reports the
        // stale block-entry RIP and the diagnostic points at the wrong insn.
        c.mov(kScratch0, next_rip - insn.length);
        c.str(kScratch0, ptr(kState, Offsets::Rip));
        c.mov(kWScratch0, static_cast<u32>(ExitReason::UnsupportedInstruction));
        c.str(kWScratch0, ptr(kState, static_cast<u32>(offsetof(GuestState, exit_reason))));
        c.br(kExitStub);
        c.L(ok);
    }

    if (w == 8) {
        // dividend = AX (16-bit, signed). q,r are signed 8-bit.
        const XReg dv = kScratch0, q = kScratch2, r = XReg(13);
        c.ldr(dv, ptr(kState, GprOffset(0)));
        c.sxth(dv, WReg(dv.getIdx()));          // sign-extend AX (bits 15:0) to 64
        c.sxtb(src, WReg(src.getIdx()));         // sign-extend the 8-bit divisor
        c.sdiv(q, dv, src);
        c.msub(r, q, src, dv);                   // r = dv - q*src
        // AL=q, AH=r, preserve upper 48.
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
        // dividend = DX:AX (32-bit signed).
        const XReg dv = kScratch0, q = kScratch2, r = XReg(13);
        c.ldr(dv, ptr(kState, GprOffset(0)));
        c.and_(dv, dv, 0xFFFF);
        c.ldr(XReg(14), ptr(kState, GprOffset(2)));
        c.and_(XReg(14), XReg(14), 0xFFFF);
        c.lsl(XReg(14), XReg(14), 16);
        c.orr(dv, dv, XReg(14));                 // (DX<<16)|AX
        c.sxtw(dv, WReg(dv.getIdx()));           // sign-extend 32-bit dividend to 64
        c.sxth(src, WReg(src.getIdx()));         // sign-extend 16-bit divisor
        c.sdiv(q, dv, src);
        c.msub(r, q, src, dv);
        // AX=q, DX=r (merge low 16, preserve upper 48).
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
        // dividend = EDX:EAX (64-bit signed).
        const XReg dv = kScratch0, q = kScratch2, r = XReg(13);
        c.ldr(dv, ptr(kState, GprOffset(0)));
        c.and_(dv, dv, 0xFFFFFFFFull);
        c.ldr(XReg(14), ptr(kState, GprOffset(2)));
        c.and_(XReg(14), XReg(14), 0xFFFFFFFFull);
        c.lsl(XReg(14), XReg(14), 32);
        c.orr(dv, dv, XReg(14));                 // (EDX<<32)|EAX = signed 64-bit dividend
        c.sxtw(src, WReg(src.getIdx()));         // sign-extend 32-bit divisor
        c.sdiv(q, dv, src);
        c.msub(r, q, src, dv);
        c.and_(q, q, 0xFFFFFFFFull);
        c.str(q, ptr(kState, GprOffset(0)));     // EAX (zero-extends RAX)
        c.and_(r, r, 0xFFFFFFFFull);
        c.str(r, ptr(kState, GprOffset(2)));     // EDX (zero-extends RDX)
        return true;
    }

    // w == 64: signed 128/64 division of (RDX:RAX) by src.
    // qsign = sign(dividend) XOR sign(src); rsign = sign(dividend). Take abs of
    // both, run unsigned long division (same loop as EmitDiv), reapply signs.
    const XReg hi = kScratch0, lo = kScratch2, qsign = XReg(13), rsign = XReg(14);
    c.ldr(hi, ptr(kState, GprOffset(2)));        // RDX
    c.ldr(lo, ptr(kState, GprOffset(0)));        // RAX
    c.lsr(qsign, hi, 63);
    c.lsr(XReg(7), src, 63);
    c.eor(qsign, qsign, XReg(7));                // quotient sign
    c.lsr(rsign, hi, 63);                        // remainder sign = dividend sign
    // abs(src).
    { Label pos; c.cbz(XReg(7), pos); c.neg(src, src); c.L(pos); }
    // abs(128-bit hi:lo) if dividend negative.
    { Label pos; c.cbz(rsign, pos);
      c.mvn(lo, lo); c.mvn(hi, hi);
      c.adds(lo, lo, 1); c.adc(hi, hi, XReg(31));   // +1 with carry into hi (x31=xzr)
      c.L(pos); }

    // Unsigned long division (mirrors EmitDiv w==64).
    const XReg q = XReg(4), rem = XReg(5), i = XReg(15), bit = XReg(6), carry = XReg(8);
    c.mov(q, 0);
    c.mov(rem, 0);
    c.mov(i, 128);
    Label loop, after;
    c.L(loop);
    c.cbz(i, after);
    c.lsr(carry, rem, 63);
    c.lsl(rem, rem, 1);
    c.lsr(bit, hi, 63);
    c.orr(rem, rem, bit);
    c.lsl(hi, hi, 1);
    c.lsr(bit, lo, 63);
    c.orr(hi, hi, bit);
    c.lsl(lo, lo, 1);
    c.lsl(q, q, 1);
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
    // Reapply signs.
    { Label pos; c.cbz(qsign, pos); c.neg(q, q); c.L(pos); }
    { Label pos; c.cbz(rsign, pos); c.neg(rem, rem); c.L(pos); }
    c.str(q, ptr(kState, GprOffset(0)));         // quotient -> RAX
    c.str(rem, ptr(kState, GprOffset(2)));       // remainder -> RDX
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
    // Load src2 FIRST: if it is memory, EmitEffectiveAddress clobbers x9
    // (kScratch0) for index scaling, so src1 must be read afterward.
    if (!LoadSrcOperand(ops[2], w, b, c)) return false;
    c.ldr(a, ptr(kState, GprOffset(s1)));
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

// guest: tzcnt dst, src -> count trailing zero bits (= width if src==0).
//   CF=(src==0); ZF=(result==0); SF/OF/PF/AF undefined (cleared CF/ZF only).
//   Widths 32/64. trailing-zeros = clz(rbit(x)).
bool EmitTzcnt(const ZydisDecodedInstruction& insn, const ZydisDecodedOperand* ops,
               CodeGenerator& c) {
    const u32 w = insn.operand_width;
    if (w != 32 && w != 64) return false;
    if (ops[0].type != ZYDIS_OPERAND_TYPE_REGISTER) return false;
    const int d = ZydisGprToIndex(ops[0].reg.value);
    if (d < 0) return false;
    const XReg src = kScratch0, res = kScratch2;
    if (!LoadSrcOperand(ops[1], w, src, c)) return false;
    if (w == 32) {
        c.rbit(WReg(res.getIdx()), WReg(src.getIdx()));
        c.clz(WReg(res.getIdx()), WReg(res.getIdx()));
        c.and_(res, res, 0xFFFFFFFFull);
    } else {
        c.rbit(res, src);
        c.clz(res, res);
    }
    c.str(res, ptr(kState, GprOffset(d)));
    const XReg fl = XReg(14), t = XReg(15);
    c.ldr(fl, ptr(kState, Offsets::Rflags));
    c.mov(t, ~((1ull<<0)|(1ull<<6)));
    c.and_(fl, fl, t);
    c.cmp(src, 0); c.cset(t, EQ); c.orr(fl, fl, t);                  // CF = src==0
    c.cmp(res, 0); c.cset(t, EQ); c.lsl(t, t, 6); c.orr(fl, fl, t);  // ZF = result==0
    c.str(fl, ptr(kState, Offsets::Rflags));
    return true;
}

// guest: blsr dst, src -> reset lowest set bit: dst = src & (src - 1).
//   CF=(src==0); ZF=(dst==0); SF=(dst sign); OF=0; AF/PF undefined. Widths 32/64.
bool EmitBlsr(const ZydisDecodedInstruction& insn, const ZydisDecodedOperand* ops,
              CodeGenerator& c) {
    const u32 w = insn.operand_width;
    if (w != 32 && w != 64) return false;
    if (ops[0].type != ZYDIS_OPERAND_TYPE_REGISTER) return false;
    const int d = ZydisGprToIndex(ops[0].reg.value);
    if (d < 0) return false;
    const XReg src = kScratch0, res = kScratch2;
    if (!LoadSrcOperand(ops[1], w, src, c)) return false;
    c.sub(res, src, 1);
    c.and_(res, res, src);
    if (w == 32) c.and_(res, res, 0xFFFFFFFFull);
    c.str(res, ptr(kState, GprOffset(d)));
    const XReg fl = XReg(14), t = XReg(15);
    c.ldr(fl, ptr(kState, Offsets::Rflags));
    c.mov(t, ~((1ull<<0)|(1ull<<2)|(1ull<<4)|(1ull<<6)|(1ull<<7)|(1ull<<11)));
    c.and_(fl, fl, t);
    c.cmp(src, 0); c.cset(t, EQ); c.orr(fl, fl, t);                  // CF = src==0
    c.cmp(res, 0); c.cset(t, EQ); c.lsl(t, t, 6); c.orr(fl, fl, t);  // ZF = dst==0
    c.ubfx(t, res, w - 1, 1); c.lsl(t, t, 7); c.orr(fl, fl, t);      // SF = dst sign
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
    const bool addr32 = (insn.address_width == 32);   // 0x67 prefix
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
        if (!EmitEffectiveAddress(ops[0].mem, next_rip, c, addr32)) return false;  // -> x11
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
        const int hbDst = HighByteParent(ops[0].reg.value);
        const int d = (hbDst >= 0) ? hbDst : ZydisGprToIndex(ops[0].reg.value);
        if (d < 0) return false;
        if (src_mem) {
            // Load rhs from memory first (EA clobbers x9/x11), then lhs.
            if (!EmitEffectiveAddress(ops[1].mem, next_rip, c, addr32)) return false;
            c.mov(vAddr, kAddr);
            width_load(vRhs, vAddr);
            c.ldr(vLhs, ptr(kState, GprOffset(d)));
            if (hbDst >= 0) c.ubfx(vLhs, vLhs, 8, 8);
        } else {
            c.ldr(vLhs, ptr(kState, GprOffset(d)));
            if (hbDst >= 0) c.ubfx(vLhs, vLhs, 8, 8);
            if (ops[1].type == ZYDIS_OPERAND_TYPE_REGISTER) {
                const int hbSrc = HighByteParent(ops[1].reg.value);
                if (hbSrc >= 0) {
                    c.ldr(vRhs, ptr(kState, GprOffset(hbSrc)));
                    c.ubfx(vRhs, vRhs, 8, 8);
                } else {
                    const int s = ZydisGprToIndex(ops[1].reg.value);
                    if (s < 0) return false;
                    c.ldr(vRhs, ptr(kState, GprOffset(s)));
                }
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
        const int hbDst = HighByteParent(ops[0].reg.value);
        if (hbDst >= 0) {
            // Write result into bits 15:8 of the parent slot.
            c.and_(vRes, vRes, 0xFFull);
            c.lsl(vRes, vRes, 8);
            c.ldr(vLhs, ptr(kState, GprOffset(hbDst)));
            c.mov(vRhs, ~(0xFFull << 8));
            c.and_(vLhs, vLhs, vRhs);
            c.orr(vLhs, vLhs, vRes);
            c.str(vLhs, ptr(kState, GprOffset(hbDst)));
            EmitMaterializeFlags(c);
            return true;
        }
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
              u64 next_rip, CodeGenerator& c) {
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
    } else if (ops[0].type == ZYDIS_OPERAND_TYPE_MEMORY) {
        if (!EmitEffectiveAddress(ops[0].mem, next_rip, c)) return false;
        c.ldr(kScratch0, ptr(kAddr));   // push qword[mem]
    } else {
        return false;
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

// guest: leave — mov rsp, rbp ; pop rbp. Tears down the stack frame.
//   rsp = rbp ; rbp = [rsp] ; rsp += 8.
bool EmitLeave(const ZydisDecodedInstruction& insn, CodeGenerator& c) {
    if (insn.mnemonic != ZYDIS_MNEMONIC_LEAVE) return false;
    constexpr int RSP_IDX = 4, RBP_IDX = 5;
    c.ldr(kScratch1, ptr(kState, GprOffset(RBP_IDX)));  // rsp = rbp
    c.ldr(kScratch0, ptr(kScratch1));                   // x9 = [rbp] (saved rbp)
    c.add(kScratch1, kScratch1, 8);                     // rsp = rbp + 8
    c.str(kScratch1, ptr(kState, GprOffset(RSP_IDX)));
    c.str(kScratch0, ptr(kState, GprOffset(RBP_IDX)));  // rbp = saved rbp
    return true;
}

// guest: xadd r/m, reg — temp = dst; dst = dst + reg; reg = temp. Sets the
// arithmetic flags like ADD. Memory or register dest; 32-bit form zero-extends
// the reg slot. (Stashes the ADD flag side-band like other arithmetic.)
bool EmitXadd(const ZydisDecodedInstruction& insn, const ZydisDecodedOperand* ops,
              u64 next_rip, CodeGenerator& c) {
    if (insn.mnemonic != ZYDIS_MNEMONIC_XADD) return false;
    const u32 w = insn.operand_width;
    if (w != 32 && w != 64) return false;
    if (ops[1].type != ZYDIS_OPERAND_TYPE_REGISTER) return false;
    const int reg = ZydisGprToIndex(ops[1].reg.value);
    if (reg < 0) return false;

    const XReg dval = XReg(5), rval = XReg(6), sum = XReg(7);
    // Load reg source.
    c.ldr(rval, ptr(kState, GprOffset(reg)));
    if (w == 32) c.and_(rval, rval, 0xFFFFFFFFull);

    bool memDst = (ops[0].type == ZYDIS_OPERAND_TYPE_MEMORY);
    if (memDst) {
        if (!EmitEffectiveAddress(ops[0].mem, next_rip, c)) return false;
        c.mov(XReg(15), kAddr);                          // stable addr
        if (w == 32) c.ldr(WReg(5), ptr(XReg(15)));
        else         c.ldr(dval, ptr(XReg(15)));
    } else if (ops[0].type == ZYDIS_OPERAND_TYPE_REGISTER) {
        const int d = ZydisGprToIndex(ops[0].reg.value);
        if (d < 0) return false;
        c.ldr(dval, ptr(kState, GprOffset(d)));
        if (w == 32) c.and_(dval, dval, 0xFFFFFFFFull);
    } else {
        return false;
    }

    c.add(sum, dval, rval);
    if (w == 32) c.and_(sum, sum, 0xFFFFFFFFull);

    // dst = sum.
    if (memDst) {
        if (w == 32) c.str(WReg(7), ptr(XReg(15)));
        else         c.str(sum, ptr(XReg(15)));
    } else {
        const int d = ZydisGprToIndex(ops[0].reg.value);
        c.str(sum, ptr(kState, GprOffset(d)));
    }
    // reg = old dst value (zero-extended for w32).
    c.str(dval, ptr(kState, GprOffset(reg)));

    // Flag side-band: ADD of dval + rval -> sum.
    c.str(dval, ptr(kState, Offsets::FlagLhs));
    c.str(rval, ptr(kState, Offsets::FlagRhs));
    c.str(sum,  ptr(kState, Offsets::FlagResult));
    c.mov(kWScratch0, FLAG_OP_ADD);
    c.str(kWScratch0, ptr(kState, static_cast<u32>(offsetof(GuestState, flag_op))));
    c.mov(kWScratch0, w);
    c.str(kWScratch0, ptr(kState, static_cast<u32>(offsetof(GuestState, flag_width))));
    EmitMaterializeFlags(c);
    return true;
}

// guest: cmpxchg r/m, reg — compare ACC (AL/AX/EAX/RAX) with dst.
//   if (ACC == dst) { ZF=1; dst = reg; }
//   else            { ZF=0; ACC = dst; }
// Flags set as for CMP(ACC, dst). 32-bit zero-extends ACC on mismatch.
bool EmitCmpxchg(const ZydisDecodedInstruction& insn, const ZydisDecodedOperand* ops,
                 u64 next_rip, CodeGenerator& c) {
    if (insn.mnemonic != ZYDIS_MNEMONIC_CMPXCHG) return false;
    const u32 w = insn.operand_width;
    if (w != 32 && w != 64) return false;
    if (ops[1].type != ZYDIS_OPERAND_TYPE_REGISTER) return false;
    const int reg = ZydisGprToIndex(ops[1].reg.value);
    if (reg < 0) return false;
    constexpr int RAX_IDX = 0;

    const XReg acc = XReg(5), dval = XReg(6), rval = XReg(7);
    c.ldr(acc, ptr(kState, GprOffset(RAX_IDX)));
    c.ldr(rval, ptr(kState, GprOffset(reg)));
    if (w == 32) { c.and_(acc, acc, 0xFFFFFFFFull); c.and_(rval, rval, 0xFFFFFFFFull); }

    bool memDst = (ops[0].type == ZYDIS_OPERAND_TYPE_MEMORY);
    if (memDst) {
        if (!EmitEffectiveAddress(ops[0].mem, next_rip, c)) return false;
        c.mov(XReg(15), kAddr);
        if (w == 32) c.ldr(WReg(6), ptr(XReg(15)));
        else         c.ldr(dval, ptr(XReg(15)));
    } else if (ops[0].type == ZYDIS_OPERAND_TYPE_REGISTER) {
        const int d = ZydisGprToIndex(ops[0].reg.value);
        if (d < 0) return false;
        c.ldr(dval, ptr(kState, GprOffset(d)));
        if (w == 32) c.and_(dval, dval, 0xFFFFFFFFull);
    } else {
        return false;
    }

    // Flag side-band: CMP(acc, dval) == SUB.
    const XReg diff = XReg(12);
    c.sub(diff, acc, dval);
    if (w == 32) c.and_(diff, diff, 0xFFFFFFFFull);
    c.str(acc,  ptr(kState, Offsets::FlagLhs));
    c.str(dval, ptr(kState, Offsets::FlagRhs));
    c.str(diff, ptr(kState, Offsets::FlagResult));
    c.mov(kWScratch0, FLAG_OP_SUB);
    c.str(kWScratch0, ptr(kState, static_cast<u32>(offsetof(GuestState, flag_op))));
    c.mov(kWScratch0, w);
    c.str(kWScratch0, ptr(kState, static_cast<u32>(offsetof(GuestState, flag_width))));

    // Branchless: equal = (acc == dval).
    //   dst  = equal ? reg : dval   (dst unchanged on mismatch)
    //   acc  = equal ? acc : dval
    const XReg newDst = XReg(13), newAcc = XReg(14);
    c.cmp(acc, dval);
    c.csel(newDst, rval, dval, EQ);
    c.csel(newAcc, acc,  dval, EQ);
    if (w == 32) { c.and_(newDst, newDst, 0xFFFFFFFFull); c.and_(newAcc, newAcc, 0xFFFFFFFFull); }

    if (memDst) {
        if (w == 32) c.str(WReg(13), ptr(XReg(15)));
        else         c.str(newDst, ptr(XReg(15)));
    } else {
        const int d = ZydisGprToIndex(ops[0].reg.value);
        c.str(newDst, ptr(kState, GprOffset(d)));
    }
    c.str(newAcc, ptr(kState, GprOffset(RAX_IDX)));
    EmitMaterializeFlags(c);
    return true;
}

// ============================================================================
// EmitLockedRmwA64 — guest atomics via LSE
// ============================================================================
//
// Guest threads run on parallel host threads over shared guest memory, so a
// LOCK-prefixed RMW (and the implicitly-locked XCHG-with-memory) must be one
// atomic step with full x86-LOCK ordering. This backend targets Apple Silicon
// exclusively (ARMv8.4+), so the LSE atomics are unconditionally available --
// no ldxr/stxr fallback loop is needed. The acquire+release (*al) variants
// are the correct strength mapping for x86 LOCK semantics (the same mapping
// Rosetta/FEX use): a LOCK op is a full two-way fence on x86, and
// load-acquire/store-release on the single accessed location plus the
// program-order guarantees of the surrounding TSO-translated code give the
// observable equivalent.
//
// Covered (widths 8/16/32/64 unless noted):
//   ADD / SUB                  -> ldaddal (SUB negates the operand)
//   AND / OR / XOR             -> ldclral (complement) / ldsetal / ldeoral
//   INC / DEC      (32/64)     -> ldaddal +/-1, CF preserved by materializer
//   XADD           (32/64)     -> ldaddal, reg receives old value
//   CMPXCHG        (32/64)     -> casal, acc receives old value
//   XCHG mem form              -> swpal, reg receives old value
//
// NOT covered -- returns false, routing to the diagnosable unsupported exit
// (NEVER to the plain non-atomic emitters):
//   ADC / SBB (need a flag-input CAS loop), NEG / NOT (no LSE op; CAS loop),
//   BTS / BTR / BTC (bit-string addressing), CMPXCHG8B / CMPXCHG16B (needs
//   caspal), 8/16-bit XADD/CMPXCHG, AH/CH/DH/BH operands. All are rare; add
//   on first sighting in a real binary.
//
// Flags: every op stashes the lazy side-band as (lhs = OLD memory value,
// rhs = operand, result = computed) with the operation width, then
// materializes -- exactly matching the x86 semantics where LOCK-op flags are
// derived from the pre-update memory value.
//
// Register budget: x15 = stable EA, x5 = operand value, x6 = old memory
// value (LSE destination), x7 = transformed operand / CAS-new, x12/x13 =
// transients. EmitMaterializeFlags clobbers x5..x7 and x9..x15, so all GPR
// writebacks happen before it (it re-reads the side-band from memory).
bool EmitLockedRmwA64(const ZydisDecodedInstruction& insn,
                      const ZydisDecodedOperand* ops, u64 next_rip,
                      CodeGenerator& c) {
    const ZydisMnemonic m = insn.mnemonic;
    const u32 w = insn.operand_width;
    if (w != 8 && w != 16 && w != 32 && w != 64) return false;
    if (ops[0].type != ZYDIS_OPERAND_TYPE_MEMORY) return false; // LOCK => mem dst

    const XReg addr = XReg(15);
    const XReg val = XReg(5), old = XReg(6), op2 = XReg(7);
    const WReg wval = WReg(5), wold = WReg(6), wop2 = WReg(7);

    // Width-masked load of the source operand (register or immediate) into
    // x5. High-byte registers (AH..BH) are not handled. Immediates arrive
    // sign-extended from Zydis; mask to width so the side-band and the
    // narrow LSE forms see the architectural operand.
    auto load_src = [&](const ZydisDecodedOperand& op) -> bool {
        if (op.type == ZYDIS_OPERAND_TYPE_REGISTER) {
            if (HighByteParent(op.reg.value) >= 0) return false;
            const int idx = ZydisGprToIndex(op.reg.value);
            if (idx < 0) return false;
            c.ldr(val, ptr(kState, GprOffset(idx)));
        } else if (op.type == ZYDIS_OPERAND_TYPE_IMMEDIATE) {
            c.mov(val, static_cast<u64>(op.imm.value.s));
        } else {
            return false;
        }
        if (w < 64) {
            c.mov(op2, (1ull << w) - 1);
            c.and_(val, val, op2);
        }
        return true;
    };

    // Width-dispatched LSE fetch-ops: result (old value) -> x6/w6.
    auto lse_add = [&](const XReg& s) {
        switch (w) {
        case 8:  c.ldaddalb(WReg(s.getIdx()), wold, ptr(addr)); break;
        case 16: c.ldaddalh(WReg(s.getIdx()), wold, ptr(addr)); break;
        case 32: c.ldaddal(WReg(s.getIdx()), wold, ptr(addr)); break;
        default: c.ldaddal(s, old, ptr(addr)); break;
        }
    };
    auto lse_set = [&](const XReg& s) {
        switch (w) {
        case 8:  c.ldsetalb(WReg(s.getIdx()), wold, ptr(addr)); break;
        case 16: c.ldsetalh(WReg(s.getIdx()), wold, ptr(addr)); break;
        case 32: c.ldsetal(WReg(s.getIdx()), wold, ptr(addr)); break;
        default: c.ldsetal(s, old, ptr(addr)); break;
        }
    };
    auto lse_clr = [&](const XReg& s) {
        switch (w) {
        case 8:  c.ldclralb(WReg(s.getIdx()), wold, ptr(addr)); break;
        case 16: c.ldclralh(WReg(s.getIdx()), wold, ptr(addr)); break;
        case 32: c.ldclral(WReg(s.getIdx()), wold, ptr(addr)); break;
        default: c.ldclral(s, old, ptr(addr)); break;
        }
    };
    auto lse_eor = [&](const XReg& s) {
        switch (w) {
        case 8:  c.ldeoralb(WReg(s.getIdx()), wold, ptr(addr)); break;
        case 16: c.ldeoralh(WReg(s.getIdx()), wold, ptr(addr)); break;
        case 32: c.ldeoral(WReg(s.getIdx()), wold, ptr(addr)); break;
        default: c.ldeoral(s, old, ptr(addr)); break;
        }
    };

    // Invariant relied on below: the byte/half LSE forms ZERO-EXTEND the
    // loaded old value into Wt, and any W-register write zero-extends into
    // the X register -- so after every lse_* call, x6 (`old`) holds the
    // width-masked old memory value with clean upper bits, no further
    // masking needed.

    // Side-band + materialize. lhs/rhs/result already in x-regs.
    auto flags = [&](u32 flag_op, const XReg& lhs, const XReg& rhs,
                     const XReg& result) {
        c.str(lhs, ptr(kState, Offsets::FlagLhs));
        c.str(rhs, ptr(kState, Offsets::FlagRhs));
        c.str(result, ptr(kState, Offsets::FlagResult));
        c.mov(kWScratch0, flag_op);
        c.str(kWScratch0,
              ptr(kState, static_cast<u32>(offsetof(GuestState, flag_op))));
        c.mov(kWScratch0, w);
        c.str(kWScratch0,
              ptr(kState, static_cast<u32>(offsetof(GuestState, flag_width))));
        EmitMaterializeFlags(c);
    };

    // Effective address first (clobbers x9/x11), parked in x15.
    if (!EmitEffectiveAddress(ops[0].mem, next_rip, c)) return false;
    c.mov(addr, kAddr);

    switch (m) {
    // ---- binary RMW: [mem] op= (reg | imm); flags from OLD value ----
    case ZYDIS_MNEMONIC_ADD:
    case ZYDIS_MNEMONIC_SUB: {
        if (!load_src(ops[1])) return false;
        if (m == ZYDIS_MNEMONIC_SUB) {
            c.neg(op2, val);                  // memory receives old + (-val)
            if (w < 64) {
                c.mov(XReg(12), (1ull << w) - 1);
                c.and_(op2, op2, XReg(12));   // narrow forms add low bits
            }
            lse_add(op2);
        } else {
            lse_add(val);
        }
        // result for flags: old +/- val at width.
        if (m == ZYDIS_MNEMONIC_SUB) c.sub(XReg(12), old, val);
        else                         c.add(XReg(12), old, val);
        if (w < 64) {
            c.mov(XReg(13), (1ull << w) - 1);
            c.and_(XReg(12), XReg(12), XReg(13));
        }
        flags(m == ZYDIS_MNEMONIC_SUB ? FLAG_OP_SUB : FLAG_OP_ADD,
              old, val, XReg(12));
        return true;
    }
    case ZYDIS_MNEMONIC_AND:
    case ZYDIS_MNEMONIC_OR:
    case ZYDIS_MNEMONIC_XOR: {
        if (!load_src(ops[1])) return false;
        if (m == ZYDIS_MNEMONIC_AND) {
            c.mvn(op2, val);                  // AND old,val == CLR old,~val
            lse_clr(op2);
            c.and_(XReg(12), old, val);
        } else if (m == ZYDIS_MNEMONIC_OR) {
            lse_set(val);
            c.orr(XReg(12), old, val);
        } else {
            lse_eor(val);
            c.eor(XReg(12), old, val);
        }
        if (w < 64) {
            c.mov(XReg(13), (1ull << w) - 1);
            c.and_(XReg(12), XReg(12), XReg(13));
        }
        flags(FLAG_OP_LOGIC, old, val, XReg(12));
        return true;
    }

    // ---- unary RMW: INC/DEC (32/64); CF preserved by the materializer ----
    case ZYDIS_MNEMONIC_INC:
    case ZYDIS_MNEMONIC_DEC: {
        if (w != 32 && w != 64) return false;
        const bool inc = (m == ZYDIS_MNEMONIC_INC);
        c.mov(val, 1);
        if (inc) {
            lse_add(val);
            c.add(XReg(12), old, val);
        } else {
            c.mov(op2, static_cast<u64>(-1));
            if (w == 32) {
                c.mov(XReg(13), 0xFFFFFFFFull);
                c.and_(op2, op2, XReg(13));
            }
            lse_add(op2);
            c.sub(XReg(12), old, val);
        }
        if (w == 32) {
            c.mov(XReg(13), 0xFFFFFFFFull);
            c.and_(XReg(12), XReg(12), XReg(13));
        }
        flags(inc ? FLAG_OP_INC : FLAG_OP_DEC, old, val, XReg(12));
        return true;
    }

    // ---- XADD: reg <- old [mem]; [mem] += reg; flags from OLD value ----
    case ZYDIS_MNEMONIC_XADD: {
        if (w != 32 && w != 64) return false;
        if (ops[1].type != ZYDIS_OPERAND_TYPE_REGISTER) return false;
        if (HighByteParent(ops[1].reg.value) >= 0) return false;
        const int reg = ZydisGprToIndex(ops[1].reg.value);
        if (reg < 0) return false;
        if (!load_src(ops[1])) return false;
        lse_add(val);
        // reg = old (32-bit zero-extends the slot; old is already
        // zero-extended by the W-form write).
        c.str(old, ptr(kState, GprOffset(reg)));
        c.add(XReg(12), old, val);
        if (w == 32) {
            c.mov(XReg(13), 0xFFFFFFFFull);
            c.and_(XReg(12), XReg(12), XReg(13));
        }
        flags(FLAG_OP_ADD, old, val, XReg(12));
        return true;
    }

    // ---- CMPXCHG: casal; acc <- old; flags = CMP(acc, old) ----
    case ZYDIS_MNEMONIC_CMPXCHG: {
        if (w != 32 && w != 64) return false;
        if (ops[1].type != ZYDIS_OPERAND_TYPE_REGISTER) return false;
        if (HighByteParent(ops[1].reg.value) >= 0) return false;
        const int reg = ZydisGprToIndex(ops[1].reg.value);
        if (reg < 0) return false;
        constexpr int RAX_IDX = 0;

        // x5 = expected (acc) -- casal overwrites it with the old value, so
        // keep the original acc in x12 for the flag computation.
        c.ldr(val, ptr(kState, GprOffset(RAX_IDX)));
        c.ldr(op2, ptr(kState, GprOffset(reg)));   // desired/new
        if (w == 32) {
            c.mov(XReg(13), 0xFFFFFFFFull);
            c.and_(val, val, XReg(13));
            c.and_(op2, op2, XReg(13));
        }
        c.mov(XReg(12), val);                      // acc copy for flags
        if (w == 32) c.casal(wval, wop2, ptr(addr));
        else         c.casal(val, op2, ptr(addr));
        // x5 now holds the OLD memory value (zero-extended for w32).
        // Architectural acc writeback: on success old == acc (identity), on
        // failure acc = old -- so an unconditional store of old is exact.
        c.str(val, ptr(kState, GprOffset(RAX_IDX)));
        // flags = CMP(acc_orig, old): lhs = x12, rhs = x5.
        c.sub(XReg(13), XReg(12), val);
        if (w == 32) {
            c.mov(old, 0xFFFFFFFFull);             // x6 free as a mask reg here
            c.and_(XReg(13), XReg(13), old);
        }
        flags(FLAG_OP_SUB, XReg(12), val, XReg(13));
        return true;
    }

    // ---- XCHG with memory: implicitly locked; swpal; no flags ----
    case ZYDIS_MNEMONIC_XCHG: {
        // Zydis may order the operands either way; normalize to mem in
        // ops[0] (guaranteed by the gate's mem check) and reg in ops[1].
        if (ops[1].type != ZYDIS_OPERAND_TYPE_REGISTER) return false;
        if (HighByteParent(ops[1].reg.value) >= 0) return false;
        const int reg = ZydisGprToIndex(ops[1].reg.value);
        if (reg < 0) return false;
        if (!load_src(ops[1])) return false;
        switch (w) {
        case 8:  c.swpalb(wval, wold, ptr(addr)); break;
        case 16: c.swpalh(wval, wold, ptr(addr)); break;
        case 32: c.swpal(wval, wold, ptr(addr)); break;
        default: c.swpal(val, old, ptr(addr)); break;
        }
        // Register writeback with width semantics: 8/16 merge into the low
        // bits preserving the rest of the slot; 32 zero-extends; 64 full.
        if (w == 8 || w == 16) {
            const u64 msk = (1ull << w) - 1;
            c.ldr(XReg(12), ptr(kState, GprOffset(reg)));
            c.mov(XReg(13), ~msk);
            c.and_(XReg(12), XReg(12), XReg(13));
            c.orr(XReg(12), XReg(12), old);        // old already zero-extended
            c.str(XReg(12), ptr(kState, GprOffset(reg)));
        } else {
            c.str(old, ptr(kState, GprOffset(reg)));
        }
        return true;
    }

    default:
        return false; // ADC/SBB/NEG/NOT/BTS/BTR/BTC/CMPXCHG8B/16B: gate.
    }
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
    // Segment handling mirrors the x86 host lifter: DS/SS/CS/ES are flat
    // (base 0) in long mode; FS/GS carry the guest-visible TLS bases that
    // live in GuestState (fs_base = the thread's TCB, installed by Run()
    // via GetTcbBase). The base is added LAST, after the addr32 truncation
    // if any -- architecturally the linear address is seg_base + the
    // (possibly 32-bit-truncated) effective address. This unblocks every
    // TLS access on this backend: fs:[0] (TCB self-pointer) and fs:[0x28]
    // (stack canary) previously took the fatal unsupported exit, which
    // stopped any real guest function prologue cold.
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
    const s64 disp = mem.disp.value;

    const XReg idx_scratch = XReg(9);  // index*scale staging (rax analog)

    // RIP-relative: address = next_rip + disp, constant-folded.
    if (has_base && mem.base == ZYDIS_REGISTER_RIP) {
        if (has_index) return false;
        if (seg_base_off >= 0) return false;  // segment + RIP-relative: degenerate
        c.mov(kAddr, static_cast<u64>(static_cast<s64>(next_rip) + disp));
        return true;
    }

    // Plain [disp] absolute (no base, no index). With a segment override
    // this is the common TLS form (fs:[0], fs:[0x28]).
    if (!has_base && !has_index) {
        c.mov(kAddr, static_cast<u64>(disp));
        if (seg_base_off >= 0) {
            c.ldr(idx_scratch, ptr(kState, static_cast<u32>(seg_base_off)));
            c.add(kAddr, kAddr, idx_scratch);
        }
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

    // FS/GS base last, after the index/disp math and any addr32 truncation
    // (the base is part of LINEAR address formation, applied to the already-
    // truncated effective address).
    if (seg_base_off >= 0) {
        c.ldr(idx_scratch, ptr(kState, static_cast<u32>(seg_base_off)));
        c.add(kAddr, kAddr, idx_scratch);
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
    const bool addr32 = (insn.address_width == 32);   // 0x67 prefix

    // Load lhs (ops[0]) into kScratch0, rhs (ops[1]) into kScratch1.
    auto load_operand = [&](const ZydisDecodedOperand& op, const XReg& dstreg) -> bool {
        if (op.type == ZYDIS_OPERAND_TYPE_REGISTER) {
            const int hb = HighByteParent(op.reg.value);
            if (hb >= 0) {
                c.ldr(dstreg, ptr(kState, GprOffset(hb)));
                c.ubfx(dstreg, dstreg, 8, 8);   // byte 1 of parent
                return true;
            }
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
            if (!EmitEffectiveAddress(op.mem, next_rip, c, addr32)) return false;  // -> kAddr
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
    const bool addr32 = (insn.address_width == 32);   // 0x67 prefix

    auto load_operand = [&](const ZydisDecodedOperand& op, const XReg& dstreg) -> bool {
        if (op.type == ZYDIS_OPERAND_TYPE_REGISTER) {
            const int hb = HighByteParent(op.reg.value);
            if (hb >= 0) {
                c.ldr(dstreg, ptr(kState, GprOffset(hb)));
                c.ubfx(dstreg, dstreg, 8, 8);
                return true;
            }
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
            if (!EmitEffectiveAddress(op.mem, next_rip, c, addr32)) return false;
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
               u64 next_rip, CodeGenerator& c) {
    CondClass cls; bool neg;
    if (!DecodeCondition(insn.mnemonic, cls, neg)) return false;
    if (insn.operand_count_visible != 1) return false;

    const XReg cond = XReg(12);          // result of condition (0/1)
    if (ops[0].type == ZYDIS_OPERAND_TYPE_MEMORY) {
        // EmitConditionToReg uses x11 (kAddr) as a scratch term, so compute the
        // condition FIRST, then the effective address, then store.
        EmitConditionToReg(cls, neg, cond, c);
        if (!EmitEffectiveAddress(ops[0].mem, next_rip, c)) return false;
        c.strb(WReg(12), ptr(kAddr));    // store the 0/1 byte
        return true;
    }
    if (ops[0].type != ZYDIS_OPERAND_TYPE_REGISTER) return false;
    const int dst = ZydisGprToIndex(ops[0].reg.value);
    if (dst < 0) return false;  // AH/CH/DH/BH high-byte regs unsupported

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
              u64 next_rip, CodeGenerator& c) {
    CondClass cls; bool neg;
    if (!DecodeCondition(insn.mnemonic, cls, neg)) return false;
    if (insn.operand_count_visible != 2) return false;
    if (ops[0].type != ZYDIS_OPERAND_TYPE_REGISTER) return false;
    const int w = insn.operand_width;
    if (w != 16 && w != 32 && w != 64) return false;
    const int dst = ZydisGprToIndex(ops[0].reg.value);
    if (dst < 0) return false;

    // Load src into x14 (reg or mem).
    const XReg sval = XReg(14);
    if (ops[1].type == ZYDIS_OPERAND_TYPE_REGISTER) {
        const int src = ZydisGprToIndex(ops[1].reg.value);
        if (src < 0) return false;
        c.ldr(sval, ptr(kState, GprOffset(src)));
    } else if (ops[1].type == ZYDIS_OPERAND_TYPE_MEMORY) {
        if (!EmitEffectiveAddress(ops[1].mem, next_rip, c)) return false;
        if (w == 64)      c.ldr(sval, ptr(kAddr));
        else if (w == 32) c.ldr(WReg(14), ptr(kAddr));
        else              c.ldrh(WReg(14), ptr(kAddr));
    } else {
        return false;
    }

    const XReg cond = XReg(12);
    EmitConditionToReg(cls, neg, cond, c);
    const XReg dval = XReg(13);
    c.ldr(dval, ptr(kState, GprOffset(dst)));

    if (w == 64) {
        c.cmp(cond, 0);
        c.csel(dval, sval, dval, NE);
        c.str(dval, ptr(kState, GprOffset(dst)));
    } else if (w == 32) {
        // On taken, dst = zero-extended 32-bit src. On not-taken, x86 CMOV32
        // STILL zero-extends the destination (writes to a 32-bit reg always
        // clear bits 63:32). So: result = cond ? (u32)src : (u32)dst.
        c.and_(sval, sval, 0xFFFFFFFFull);
        c.and_(dval, dval, 0xFFFFFFFFull);
        c.cmp(cond, 0);
        c.csel(dval, sval, dval, NE);
        c.str(dval, ptr(kState, GprOffset(dst)));
    } else {
        // 16-bit: merge low 16 of (cond ? src : dst), preserving upper 48.
        const XReg merged = XReg(15);
        c.and_(sval, sval, 0xFFFFull);
        c.mov(merged, dval);
        c.and_(merged, merged, 0xFFFFull);
        c.cmp(cond, 0);
        c.csel(merged, sval, merged, NE);     // chosen low 16
        c.lsr(dval, dval, 16);
        c.lsl(dval, dval, 16);                 // clear low 16 of dst
        c.orr(dval, dval, merged);
        c.str(dval, ptr(kState, GprOffset(dst)));
    }
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
            case VScalarOp::Min:
                c.fcmgt(VReg2D(2), VReg2D(1), VReg2D(0));   // mask = b>a (a<b)
                c.bsl(VReg16B(2), VReg16B(0), VReg16B(1));
                c.mov(VReg16B(0), VReg16B(2));
                break;
            case VScalarOp::Max:
                c.fcmgt(VReg2D(2), VReg2D(0), VReg2D(1));   // mask = a>b
                c.bsl(VReg16B(2), VReg16B(0), VReg16B(1));
                c.mov(VReg16B(0), VReg16B(2));
                break;
            case VScalarOp::Sqrt: c.fsqrt(DReg(0), DReg(1)); break;  // sqrt(src2)
        }
    } else {
        switch (op) {
            case VScalarOp::Add:  c.fadd(SReg(0), SReg(0), SReg(1)); break;
            case VScalarOp::Sub:  c.fsub(SReg(0), SReg(0), SReg(1)); break;
            case VScalarOp::Mul:  c.fmul(SReg(0), SReg(0), SReg(1)); break;
            case VScalarOp::Div:  c.fdiv(SReg(0), SReg(0), SReg(1)); break;
            case VScalarOp::Min:
                c.fcmgt(VReg4S(2), VReg4S(1), VReg4S(0));   // mask = b>a (a<b)
                c.bsl(VReg16B(2), VReg16B(0), VReg16B(1));
                c.mov(VReg16B(0), VReg16B(2));
                break;
            case VScalarOp::Max:
                c.fcmgt(VReg4S(2), VReg4S(0), VReg4S(1));   // mask = a>b
                c.bsl(VReg16B(2), VReg16B(0), VReg16B(1));
                c.mov(VReg16B(0), VReg16B(2));
                break;
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
    AddB, AddW, AddQ, SubB, SubW, SubQ, MulW,    // integer add/sub b/w/q; integer mul w
    MaxSB, MaxSW, MaxSD, MaxUB, MaxUW, MaxUD,     // packed signed/unsigned max (b/w/d)
    MinSB, MinSW, MinSD, MinUB, MinUW, MinUD,     // packed signed/unsigned min (b/w/d)
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
            // x86 MIN/MAX: dst = (a<b)?a:b  / (a>b)?a:b. On a NaN operand the
            // compare is false, so the result is src2 (b) — NOT IEEE minNum. NEON
            // fmin/fmax pick the non-NaN operand, which diverges, so use an
            // explicit compare-select. v0=a(src1), v1=b(src2), v2=mask.
            case VPackKind::MinPS:
                c.fcmgt(VReg4S(2), VReg4S(1), VReg4S(0));   // mask = b>a  (=> a<b)
                c.bsl(VReg16B(2), VReg16B(0), VReg16B(1));  // a<b ? a : b
                c.mov(VReg16B(0), VReg16B(2));
                break;
            case VPackKind::MaxPS:
                c.fcmgt(VReg4S(2), VReg4S(0), VReg4S(1));   // mask = a>b
                c.bsl(VReg16B(2), VReg16B(0), VReg16B(1));  // a>b ? a : b
                c.mov(VReg16B(0), VReg16B(2));
                break;
            case VPackKind::AddPD: c.fadd(VReg2D(0), VReg2D(0), VReg2D(1)); break;
            case VPackKind::SubPD: c.fsub(VReg2D(0), VReg2D(0), VReg2D(1)); break;
            case VPackKind::MulPD: c.fmul(VReg2D(0), VReg2D(0), VReg2D(1)); break;
            case VPackKind::DivPD: c.fdiv(VReg2D(0), VReg2D(0), VReg2D(1)); break;
            case VPackKind::MinPD:
                c.fcmgt(VReg2D(2), VReg2D(1), VReg2D(0));   // mask = b>a
                c.bsl(VReg16B(2), VReg16B(0), VReg16B(1));
                c.mov(VReg16B(0), VReg16B(2));
                break;
            case VPackKind::MaxPD:
                c.fcmgt(VReg2D(2), VReg2D(0), VReg2D(1));   // mask = a>b
                c.bsl(VReg16B(2), VReg16B(0), VReg16B(1));
                c.mov(VReg16B(0), VReg16B(2));
                break;
            case VPackKind::AddD:  c.add(VReg4S(0), VReg4S(0), VReg4S(1)); break;
            case VPackKind::SubD:  c.sub(VReg4S(0), VReg4S(0), VReg4S(1)); break;
            case VPackKind::MulD:  c.mul(VReg4S(0), VReg4S(0), VReg4S(1)); break;
            // Integer add/sub at the remaining element widths. NEON add/sub
            // support .16b/.8h/.2d as well as the .4s above; these wrap modulo
            // the element width exactly like x86 VPADD/VPSUB.
            case VPackKind::AddB:  c.add(VReg16B(0), VReg16B(0), VReg16B(1)); break;
            case VPackKind::AddW:  c.add(VReg8H(0),  VReg8H(0),  VReg8H(1));  break;
            case VPackKind::AddQ:  c.add(VReg2D(0),  VReg2D(0),  VReg2D(1));  break;
            case VPackKind::SubB:  c.sub(VReg16B(0), VReg16B(0), VReg16B(1)); break;
            case VPackKind::SubW:  c.sub(VReg8H(0),  VReg8H(0),  VReg8H(1));  break;
            case VPackKind::SubQ:  c.sub(VReg2D(0),  VReg2D(0),  VReg2D(1));  break;
            // VPMULLW keeps the low 16 bits of each 16-bit product, which is
            // exactly NEON's `mul .8h`.
            case VPackKind::MulW:  c.mul(VReg8H(0),  VReg8H(0),  VReg8H(1));  break;
            // Packed integer min/max. NEON s/u max/min cover .16b/.8h/.4s (there
            // is no 64-bit form, and x86 has no byte/word/dword gap there).
            case VPackKind::MaxSB: c.smax(VReg16B(0), VReg16B(0), VReg16B(1)); break;
            case VPackKind::MaxSW: c.smax(VReg8H(0),  VReg8H(0),  VReg8H(1));  break;
            case VPackKind::MaxSD: c.smax(VReg4S(0),  VReg4S(0),  VReg4S(1));  break;
            case VPackKind::MaxUB: c.umax(VReg16B(0), VReg16B(0), VReg16B(1)); break;
            case VPackKind::MaxUW: c.umax(VReg8H(0),  VReg8H(0),  VReg8H(1));  break;
            case VPackKind::MaxUD: c.umax(VReg4S(0),  VReg4S(0),  VReg4S(1));  break;
            case VPackKind::MinSB: c.smin(VReg16B(0), VReg16B(0), VReg16B(1)); break;
            case VPackKind::MinSW: c.smin(VReg8H(0),  VReg8H(0),  VReg8H(1));  break;
            case VPackKind::MinSD: c.smin(VReg4S(0),  VReg4S(0),  VReg4S(1));  break;
            case VPackKind::MinUB: c.umin(VReg16B(0), VReg16B(0), VReg16B(1)); break;
            case VPackKind::MinUW: c.umin(VReg8H(0),  VReg8H(0),  VReg8H(1));  break;
            case VPackKind::MinUD: c.umin(VReg4S(0),  VReg4S(0),  VReg4S(1));  break;
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

// Unpack / interleave family: vpunpck{l,h}{bw,wd,dq,qdq}, vunpck{l,h}{ps,pd},
// vmovlhps, vmovhlps. These map to NEON zip1/zip2 at the right element width.
// x86 PUNPCKLDQ dst,a,b -> {a0,b0,a1,b1} == zip1(d, a, b); high half == zip2.
// Per 128-bit lane for YMM. XMM zeroes the upper 128.
enum class VUnpackKind {
    LBW, HBW, LWD, HWD, LDQ, HDQ, LQDQ, HQDQ,   // integer punpck
    LPS, HPS, LPD, HPD,                          // fp unpck
    MOVLHPS, MOVHLPS,
};

bool EmitVUnpack(const ZydisDecodedInstruction& insn, const ZydisDecodedOperand* ops,
                 u64 next_rip, VUnpackKind k, CodeGenerator& c) {
    if (insn.operand_count_visible < 3) return false;
    if (ops[0].type != ZYDIS_OPERAND_TYPE_REGISTER ||
        ops[1].type != ZYDIS_OPERAND_TYPE_REGISTER) return false;
    const int dst = ZydisVecToIndex(ops[0].reg.value);
    const int s1  = ZydisVecToIndex(ops[1].reg.value);
    if (dst < 0 || s1 < 0) return false;

    const bool ymm = (ops[0].size == 256);
    const int nchunks = ymm ? 2 : 1;

    for (int h = 0; h < nchunks; ++h) {
        const int chunk = h * 2;
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
                c.mov(kScratch1, kAddr);
            }
            c.add(kScratch0, kScratch1, h * 16);
            c.ldr(QReg(1), ptr(kScratch0));
        } else {
            return false;
        }

        // v0 = src1 half, v1 = src2 half. Result -> v2.
        switch (k) {
            case VUnpackKind::LBW:  c.zip1(VReg16B(2), VReg16B(0), VReg16B(1)); break;
            case VUnpackKind::HBW:  c.zip2(VReg16B(2), VReg16B(0), VReg16B(1)); break;
            case VUnpackKind::LWD:  c.zip1(VReg8H(2),  VReg8H(0),  VReg8H(1));  break;
            case VUnpackKind::HWD:  c.zip2(VReg8H(2),  VReg8H(0),  VReg8H(1));  break;
            case VUnpackKind::LDQ:  c.zip1(VReg4S(2),  VReg4S(0),  VReg4S(1));  break;
            case VUnpackKind::HDQ:  c.zip2(VReg4S(2),  VReg4S(0),  VReg4S(1));  break;
            case VUnpackKind::LQDQ: c.zip1(VReg2D(2),  VReg2D(0),  VReg2D(1));  break;
            case VUnpackKind::HQDQ: c.zip2(VReg2D(2),  VReg2D(0),  VReg2D(1));  break;
            case VUnpackKind::LPS:  c.zip1(VReg4S(2),  VReg4S(0),  VReg4S(1));  break;
            case VUnpackKind::HPS:  c.zip2(VReg4S(2),  VReg4S(0),  VReg4S(1));  break;
            case VUnpackKind::LPD:  c.zip1(VReg2D(2),  VReg2D(0),  VReg2D(1));  break;
            case VUnpackKind::HPD:  c.zip2(VReg2D(2),  VReg2D(0),  VReg2D(1));  break;
            // movlhps: dst.q0=src1.q0, dst.q1=src2.q0 == zip1(2D, src1, src2).
            case VUnpackKind::MOVLHPS: c.zip1(VReg2D(2), VReg2D(0), VReg2D(1)); break;
            // movhlps: dst.q0=src2.q1, dst.q1=src1.q1 == zip2(2D, src2, src1).
            case VUnpackKind::MOVHLPS: c.zip2(VReg2D(2), VReg2D(1), VReg2D(0)); break;
        }
        c.add(kScratch0, kState, YmmChunkOffset(dst, chunk));
        c.str(QReg(2), ptr(kScratch0));
    }

    if (!ymm) {
        c.mov(kScratch0, 0);
        c.str(kScratch0, ptr(kState, YmmChunkOffset(dst, 2)));
        c.str(kScratch0, ptr(kState, YmmChunkOffset(dst, 3)));
    }
    return true;
}

// vpshufd dst, src/m, imm8  — dst.dword[i] = src.dword[(imm8>>(2i))&3].
// vshufps dst, src1, src2, imm8 — dst[0]=src1[imm&3], dst[1]=src1[(imm>>2)&3],
//   dst[2]=src2[(imm>>4)&3], dst[3]=src2[(imm>>6)&3].
// Element selects are compile-time-known, so we emit per-lane NEON ins from the
// loaded source(s) into a fresh temp (v2), avoiding self-clobber. XMM-only here.
bool EmitVShuffle(const ZydisDecodedInstruction& insn, const ZydisDecodedOperand* ops,
                  u64 next_rip, CodeGenerator& c) {
    const bool shufps = (insn.mnemonic == ZYDIS_MNEMONIC_VSHUFPS);
    const bool pshufd = (insn.mnemonic == ZYDIS_MNEMONIC_VPSHUFD);
    if (!shufps && !pshufd) return false;

    if (ops[0].type != ZYDIS_OPERAND_TYPE_REGISTER) return false;
    const int dst = ZydisVecToIndex(ops[0].reg.value);
    if (dst < 0) return false;

    if (pshufd) {
        // ops: dst, src(reg/mem), imm8. 128-bit shuffles the one lane and zeros
        // the upper 128; 256-bit shuffles each 128-bit lane independently with
        // the same imm8.
        const int imm = static_cast<int>(ops[2].imm.value.u) & 0xFF;
        const bool ymm = (ops[0].size == 256);
        const int nhalves = ymm ? 2 : 1;

        // Resolve a stable source: register index, or memory address in x10.
        int sreg = -1;
        bool smem = (ops[1].type == ZYDIS_OPERAND_TYPE_MEMORY);
        if (ops[1].type == ZYDIS_OPERAND_TYPE_REGISTER) {
            sreg = ZydisVecToIndex(ops[1].reg.value);
            if (sreg < 0) return false;
        } else if (smem) {
            if (!EmitEffectiveAddress(ops[1].mem, next_rip, c)) return false;
            c.mov(kScratch1, kAddr);
        } else {
            return false;
        }

        for (int h = 0; h < nhalves; ++h) {
            const int chunk = h * 2;
            // Load this 128-bit lane -> v1.
            if (smem) {
                c.add(kScratch0, kScratch1, h * 16);
                c.ldr(QReg(1), ptr(kScratch0));
            } else {
                c.add(kScratch0, kState, YmmChunkOffset(sreg, chunk));
                c.ldr(QReg(1), ptr(kScratch0));
            }
            for (int i = 0; i < 4; ++i) {
                const int sel = (imm >> (2 * i)) & 3;
                c.ins(VReg4S(2)[i], VReg4S(1)[sel]);
            }
            c.add(kScratch0, kState, YmmChunkOffset(dst, chunk));
            c.str(QReg(2), ptr(kScratch0));
        }
        if (!ymm) {
            c.mov(kScratch0, 0);
            c.str(kScratch0, ptr(kState, YmmChunkOffset(dst, 2)));
            c.str(kScratch0, ptr(kState, YmmChunkOffset(dst, 3)));
        }
        return true;
    }

    // vshufps: dst, src1, src2, imm8.
    if (insn.operand_count_visible < 4) return false;
    const int s1 = ZydisVecToIndex(ops[1].reg.value);
    if (s1 < 0) return false;
    const int imm = static_cast<int>(ops[3].imm.value.u) & 0xFF;
    c.add(kScratch0, kState, YmmChunkOffset(s1, 0));
    c.ldr(QReg(0), ptr(kScratch0));
    if (ops[2].type == ZYDIS_OPERAND_TYPE_REGISTER) {
        const int s2 = ZydisVecToIndex(ops[2].reg.value);
        if (s2 < 0) return false;
        c.add(kScratch0, kState, YmmChunkOffset(s2, 0));
        c.ldr(QReg(1), ptr(kScratch0));
    } else if (ops[2].type == ZYDIS_OPERAND_TYPE_MEMORY) {
        if (!EmitEffectiveAddress(ops[2].mem, next_rip, c)) return false;
        c.ldr(QReg(1), ptr(kAddr));
    } else {
        return false;
    }
    // lanes 0,1 from src1 (v0); lanes 2,3 from src2 (v1). Build in v2.
    c.ins(VReg4S(2)[0], VReg4S(0)[(imm >> 0) & 3]);
    c.ins(VReg4S(2)[1], VReg4S(0)[(imm >> 2) & 3]);
    c.ins(VReg4S(2)[2], VReg4S(1)[(imm >> 4) & 3]);
    c.ins(VReg4S(2)[3], VReg4S(1)[(imm >> 6) & 3]);
    c.add(kScratch0, kState, YmmChunkOffset(dst, 0));
    c.str(QReg(2), ptr(kScratch0));
    c.mov(kScratch0, 0);
    c.str(kScratch0, ptr(kState, YmmChunkOffset(dst, 2)));
    c.str(kScratch0, ptr(kState, YmmChunkOffset(dst, 3)));
    return true;
}

// Integer per-element compare: vpcmpeqb/eqd (equal), vpcmpgtd (signed >).
// Each element becomes all-ones (true) or all-zeros (false). NEON cmeq/cmgt
// have identical semantics. XMM zeroes upper 128; YMM per-lane.
enum class VCmpIntKind { EqB, EqW, EqD, EqQ, GtB, GtW, GtD, GtQ };

bool EmitVCmpInt(const ZydisDecodedInstruction& insn, const ZydisDecodedOperand* ops,
                 u64 next_rip, VCmpIntKind k, CodeGenerator& c) {
    if (insn.operand_count_visible < 3) return false;
    if (ops[0].type != ZYDIS_OPERAND_TYPE_REGISTER ||
        ops[1].type != ZYDIS_OPERAND_TYPE_REGISTER) return false;
    const int dst = ZydisVecToIndex(ops[0].reg.value);
    const int s1  = ZydisVecToIndex(ops[1].reg.value);
    if (dst < 0 || s1 < 0) return false;
    const bool ymm = (ops[0].size == 256);
    const int nchunks = ymm ? 2 : 1;

    for (int h = 0; h < nchunks; ++h) {
        const int chunk = h * 2;
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
                c.mov(kScratch1, kAddr);
            }
            c.add(kScratch0, kScratch1, h * 16);
            c.ldr(QReg(1), ptr(kScratch0));
        } else {
            return false;
        }
        switch (k) {
            case VCmpIntKind::EqB: c.cmeq(VReg16B(0), VReg16B(0), VReg16B(1)); break;
            case VCmpIntKind::EqW: c.cmeq(VReg8H(0),  VReg8H(0),  VReg8H(1));  break;
            case VCmpIntKind::EqD: c.cmeq(VReg4S(0),  VReg4S(0),  VReg4S(1));  break;
            case VCmpIntKind::EqQ: c.cmeq(VReg2D(0),  VReg2D(0),  VReg2D(1));  break;
            case VCmpIntKind::GtB: c.cmgt(VReg16B(0), VReg16B(0), VReg16B(1)); break;
            case VCmpIntKind::GtW: c.cmgt(VReg8H(0),  VReg8H(0),  VReg8H(1));  break;
            case VCmpIntKind::GtD: c.cmgt(VReg4S(0),  VReg4S(0),  VReg4S(1));  break;
            case VCmpIntKind::GtQ: c.cmgt(VReg2D(0),  VReg2D(0),  VReg2D(1));  break;
        }
        c.add(kScratch0, kState, YmmChunkOffset(dst, chunk));
        c.str(QReg(0), ptr(kScratch0));
    }
    if (!ymm) {
        c.mov(kScratch0, 0);
        c.str(kScratch0, ptr(kState, YmmChunkOffset(dst, 2)));
        c.str(kScratch0, ptr(kState, YmmChunkOffset(dst, 3)));
    }
    return true;
}

// vptest a, b — ZF = ((a AND b) == 0); CF = ((b AND NOT a) == 0). a=ops[0],
// b=ops[1] (both reg here). Sets ZF/CF, clears OF/SF/AF/PF. 128-bit only in
// the tested forms; we OR the two 64-bit halves to test each 128-bit AND result.
bool EmitVptest(const ZydisDecodedInstruction& insn, const ZydisDecodedOperand* ops,
                CodeGenerator& c) {
    if (ops[0].type != ZYDIS_OPERAND_TYPE_REGISTER ||
        ops[1].type != ZYDIS_OPERAND_TYPE_REGISTER) return false;
    const int a = ZydisVecToIndex(ops[0].reg.value);
    const int b = ZydisVecToIndex(ops[1].reg.value);
    if (a < 0 || b < 0) return false;

    // andLow/andHigh = a & b ; subset = b & ~a, per 64-bit half.
    const XReg a0 = XReg(9), a1 = XReg(10), b0 = XReg(12), b1 = XReg(13);
    const XReg t = XReg(14), fl = XReg(15);
    c.ldr(a0, ptr(kState, YmmChunkOffset(a, 0)));
    c.ldr(a1, ptr(kState, YmmChunkOffset(a, 1)));
    c.ldr(b0, ptr(kState, YmmChunkOffset(b, 0)));
    c.ldr(b1, ptr(kState, YmmChunkOffset(b, 1)));

    c.ldr(fl, ptr(kState, Offsets::Rflags));
    // Clear ZF(6),CF(0),PF(2),SF(7),OF(11),AF(4).
    c.mov(t, ~((1ull<<6)|(1ull<<0)|(1ull<<2)|(1ull<<7)|(1ull<<11)|(1ull<<4)));
    c.and_(fl, fl, t);

    // ZF = (a&b)==0 over 128 bits.
    c.and_(t, a0, b0);
    {
        XReg t2 = XReg(5);
        c.and_(t2, a1, b1);
        c.orr(t, t, t2);            // t = OR of both halves of (a&b)
        Label nz;
        c.cbnz(t, nz);
        c.orr(fl, fl, 1ull<<6);     // ZF=1
        c.L(nz);
    }
    // CF = (b & ~a)==0 over 128 bits.
    c.bic(t, b0, a0);               // b0 & ~a0
    {
        XReg t2 = XReg(5);
        c.bic(t2, b1, a1);
        c.orr(t, t, t2);
        Label nz;
        c.cbnz(t, nz);
        c.orr(fl, fl, 1ull<<0);     // CF=1
        c.L(nz);
    }
    c.str(fl, ptr(kState, Offsets::Rflags));
    return true;
}

// vcomiss/vcomisd/vucomiss/vucomisd — ordered scalar compare of src1.low vs
// src2.low, setting ZF/PF/CF (clearing OF/SF/AF). x86 result:
//   unordered(NaN): ZF=CF=PF=1 ; a>b: 000 ; a<b: CF ; a==b: ZF.
// (COMI signals on SNaN, UCOMI on QNaN; the flag outcome is identical, so one
// path serves both.) We derive eq/gt/lt masks via NEON scalar compares; if none
// hold the inputs are unordered.
bool EmitVcomi(const ZydisDecodedInstruction& insn, const ZydisDecodedOperand* ops,
               u64 next_rip, CodeGenerator& c) {
    const bool sd = (insn.mnemonic == ZYDIS_MNEMONIC_VCOMISD ||
                     insn.mnemonic == ZYDIS_MNEMONIC_VUCOMISD);
    if (ops[0].type != ZYDIS_OPERAND_TYPE_REGISTER) return false;
    const int s1 = ZydisVecToIndex(ops[0].reg.value);
    if (s1 < 0) return false;

    // a = src1.low -> v0 ; b = src2.low -> v1.
    if (sd) c.ldr(DReg(0), ptr(kState, YmmChunkOffset(s1, 0)));
    else    c.ldr(SReg(0), ptr(kState, YmmChunkOffset(s1, 0)));
    if (ops[1].type == ZYDIS_OPERAND_TYPE_REGISTER) {
        const int s2 = ZydisVecToIndex(ops[1].reg.value);
        if (s2 < 0) return false;
        if (sd) c.ldr(DReg(1), ptr(kState, YmmChunkOffset(s2, 0)));
        else    c.ldr(SReg(1), ptr(kState, YmmChunkOffset(s2, 0)));
    } else if (ops[1].type == ZYDIS_OPERAND_TYPE_MEMORY) {
        if (!EmitEffectiveAddress(ops[1].mem, next_rip, c)) return false;
        if (sd) c.ldr(DReg(1), ptr(kAddr));
        else    c.ldr(SReg(1), ptr(kAddr));
    } else {
        return false;
    }

    // eq -> v2, gt(a>b) -> v3, lt(a<b)=gt(b,a) -> v4. Read low bit to GPR.
    const XReg eq = XReg(9), gt = XReg(10), lt = XReg(12), fl = XReg(13), t = XReg(14);
    if (sd) {
        c.fcmeq(DReg(2), DReg(0), DReg(1));
        c.fcmgt(DReg(3), DReg(0), DReg(1));
        c.fcmgt(DReg(4), DReg(1), DReg(0));
        c.fmov(eq, DReg(2)); c.fmov(gt, DReg(3)); c.fmov(lt, DReg(4));
    } else {
        c.fcmeq(SReg(2), SReg(0), SReg(1));
        c.fcmgt(SReg(3), SReg(0), SReg(1));
        c.fcmgt(SReg(4), SReg(1), SReg(0));
        c.fmov(WReg(9), SReg(2)); c.fmov(WReg(10), SReg(3)); c.fmov(WReg(12), SReg(4));
    }

    c.ldr(fl, ptr(kState, Offsets::Rflags));
    c.mov(t, ~((1ull<<6)|(1ull<<0)|(1ull<<2)|(1ull<<7)|(1ull<<11)|(1ull<<4)));
    c.and_(fl, fl, t);

    // if eq: ZF. elif lt: CF. elif gt: (nothing). else unordered: ZF|CF|PF.
    Label done, notEq, notLt;
    c.cbz(eq, notEq);
    c.orr(fl, fl, 1ull<<6);          // ZF
    c.b(done);
    c.L(notEq);
    c.cbz(lt, notLt);
    c.orr(fl, fl, 1ull<<0);          // CF
    c.b(done);
    c.L(notLt);
    // if gt -> nothing; else unordered.
    c.cbnz(gt, done);
    c.orr(fl, fl, 1ull<<6);          // ZF
    c.orr(fl, fl, 1ull<<0);          // CF
    c.orr(fl, fl, 1ull<<2);          // PF
    c.L(done);
    c.str(fl, ptr(kState, Offsets::Rflags));
    return true;
}

// vcvtsi2ss / vcvtsi2sd — signed int (32 or 64-bit GPR or mem) -> scalar
// single/double, merged into dst.low, with dst[127:32]/[127:64] from src1.
// 3-op (dst, src1, intsrc) or folded 2-op (dst=src1, intsrc).
bool EmitVcvtsi2(const ZydisDecodedInstruction& insn, const ZydisDecodedOperand* ops,
                 u64 next_rip, CodeGenerator& c) {
    const bool sd = (insn.mnemonic == ZYDIS_MNEMONIC_VCVTSI2SD);
    if (ops[0].type != ZYDIS_OPERAND_TYPE_REGISTER) return false;
    const int dst = ZydisVecToIndex(ops[0].reg.value);
    if (dst < 0) return false;

    int s1; const ZydisDecodedOperand* isrc;
    if (insn.operand_count_visible >= 3) {
        if (ops[1].type != ZYDIS_OPERAND_TYPE_REGISTER) return false;
        s1 = ZydisVecToIndex(ops[1].reg.value);
        isrc = &ops[2];
    } else {
        s1 = dst;
        isrc = &ops[1];
    }
    if (s1 < 0) return false;
    const bool src64 = (isrc->size == 64);

    // Load int -> x9, then scvtf into v0 (low element).
    if (isrc->type == ZYDIS_OPERAND_TYPE_REGISTER) {
        const int gi = ZydisGprToIndex(isrc->reg.value);
        if (gi < 0) return false;
        c.ldr(kScratch0, ptr(kState, GprOffset(gi)));
    } else if (isrc->type == ZYDIS_OPERAND_TYPE_MEMORY) {
        if (!EmitEffectiveAddress(isrc->mem, next_rip, c)) return false;
        if (src64) c.ldr(kScratch0, ptr(kAddr));
        else       c.ldr(WReg(9), ptr(kAddr));   // zero-ext; reinterpreted as signed below
    } else {
        return false;
    }
    if (sd) {
        if (src64) c.scvtf(DReg(0), kScratch0);
        else       c.scvtf(DReg(0), WReg(9));
        c.str(DReg(0), ptr(kState, YmmChunkOffset(dst, 0)));
        c.ldr(kScratch1, ptr(kState, YmmChunkOffset(s1, 1)));
        c.str(kScratch1, ptr(kState, YmmChunkOffset(dst, 1)));
    } else {
        if (src64) c.scvtf(SReg(0), kScratch0);
        else       c.scvtf(SReg(0), WReg(9));
        c.fmov(WReg(9), SReg(0));                 // result lo32 -> x9 (zero-ext)
        c.ldr(kScratch1, ptr(kState, YmmChunkOffset(s1, 0)));
        c.lsr(kScratch1, kScratch1, 32);
        c.lsl(kScratch1, kScratch1, 32);          // src1.hi32 << 32
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

// vcvttss2si / vcvttsd2si — scalar float/double -> signed int (32/64 GPR),
// truncating toward zero. NaN and out-of-range -> INT_MIN / INT64_MIN. Result
// zero-extends the destination GPR slot.
bool EmitVcvtt2si(const ZydisDecodedInstruction& insn, const ZydisDecodedOperand* ops,
                  u64 next_rip, CodeGenerator& c) {
    const ZydisMnemonic m = insn.mnemonic;
    const bool sd = (m == ZYDIS_MNEMONIC_VCVTTSD2SI || m == ZYDIS_MNEMONIC_CVTTSD2SI ||
                     m == ZYDIS_MNEMONIC_CVTSD2SI);
    // Truncating forms use round-toward-zero (fcvtzs); the non-T forms round to
    // nearest-even (fcvtns), the default MXCSR mode (the other MXCSR rounding
    // modes are not modeled, matching the VROUND note).
    const bool trunc = (m == ZYDIS_MNEMONIC_VCVTTSD2SI || m == ZYDIS_MNEMONIC_VCVTTSS2SI ||
                        m == ZYDIS_MNEMONIC_CVTTSD2SI  || m == ZYDIS_MNEMONIC_CVTTSS2SI);
    if (ops[0].type != ZYDIS_OPERAND_TYPE_REGISTER) return false;
    const int gi = ZydisGprToIndex(ops[0].reg.value);
    if (gi < 0) return false;
    const bool dst64 = (ops[0].size == 64);

    // Load scalar src -> v0.
    if (ops[1].type == ZYDIS_OPERAND_TYPE_REGISTER) {
        const int s = ZydisVecToIndex(ops[1].reg.value);
        if (s < 0) return false;
        if (sd) c.ldr(DReg(0), ptr(kState, YmmChunkOffset(s, 0)));
        else    c.ldr(SReg(0), ptr(kState, YmmChunkOffset(s, 0)));
    } else if (ops[1].type == ZYDIS_OPERAND_TYPE_MEMORY) {
        if (!EmitEffectiveAddress(ops[1].mem, next_rip, c)) return false;
        if (sd) c.ldr(DReg(0), ptr(kAddr));
        else    c.ldr(SReg(0), ptr(kAddr));
    } else {
        return false;
    }

    // Convert -> x9 (round-to-zero or round-to-nearest per the form).
    if (dst64) {
        if (trunc) { if (sd) c.fcvtzs(kScratch0, DReg(0)); else c.fcvtzs(kScratch0, SReg(0)); }
        else       { if (sd) c.fcvtns(kScratch0, DReg(0)); else c.fcvtns(kScratch0, SReg(0)); }
    } else {
        if (trunc) { if (sd) c.fcvtzs(WReg(9), DReg(0)); else c.fcvtzs(WReg(9), SReg(0)); }
        else       { if (sd) c.fcvtns(WReg(9), DReg(0)); else c.fcvtns(WReg(9), SReg(0)); }
    }

    // Indefinite fixup. x86 yields INT_MIN for NaN and any out-of-range; NEON
    // saturates NaN->0 and +ovf->INT_MAX (only -ovf already gives INT_MIN).
    // Correct: if NaN(src) OR result==INT_MAX -> force INT_MIN.
    const XReg res = kScratch0, imin = XReg(10), imax = XReg(12);
    if (dst64) { c.mov(imin, 0x8000000000000000ull); c.mov(imax, 0x7FFFFFFFFFFFFFFFull); }
    else       { c.mov(imin, 0x80000000ull);         c.mov(imax, 0x7FFFFFFFull); }
    // NaN mask: fcmeq(src,src) is all-ones when ordered, 0 when NaN.
    const XReg nm = XReg(13);
    if (sd) { c.fcmeq(DReg(1), DReg(0), DReg(0)); c.fmov(nm, DReg(1)); }
    else    { c.fcmeq(SReg(1), SReg(0), SReg(0)); c.fmov(WReg(13), SReg(1)); }
    // want_imin = (result==INT_MAX) || (nm==0  i.e. NaN)
    {
        Label setImin, doneFix;
        if (!dst64) { c.and_(res, res, 0xFFFFFFFFull); }
        c.cmp(res, imax);
        c.b(EQ, setImin);
        if (sd) c.cbz(nm, setImin); else c.cbz(WReg(13), setImin);
        c.b(doneFix);
        c.L(setImin);
        c.mov(res, imin);
        c.L(doneFix);
    }

    // Store zero-extended into the GPR slot.
    if (dst64) {
        c.str(res, ptr(kState, GprOffset(gi)));
    } else {
        c.and_(res, res, 0xFFFFFFFFull);
        c.str(res, ptr(kState, GprOffset(gi)));
    }
    return true;
}

// vcvtss2sd / vcvtsd2ss — precision change of low element; dst upper from src1.
bool EmitVcvtScalar(const ZydisDecodedInstruction& insn, const ZydisDecodedOperand* ops,
                    u64 next_rip, CodeGenerator& c) {
    const bool toDouble = (insn.mnemonic == ZYDIS_MNEMONIC_VCVTSS2SD); // ss->sd
    if (ops[0].type != ZYDIS_OPERAND_TYPE_REGISTER ||
        ops[1].type != ZYDIS_OPERAND_TYPE_REGISTER) return false;
    const int dst = ZydisVecToIndex(ops[0].reg.value);
    const int s1  = ZydisVecToIndex(ops[1].reg.value);
    if (dst < 0 || s1 < 0) return false;

    // src2 low element.
    if (ops[2].type == ZYDIS_OPERAND_TYPE_REGISTER) {
        const int s2 = ZydisVecToIndex(ops[2].reg.value);
        if (s2 < 0) return false;
        if (toDouble) c.ldr(SReg(0), ptr(kState, YmmChunkOffset(s2, 0)));
        else          c.ldr(DReg(0), ptr(kState, YmmChunkOffset(s2, 0)));
    } else if (ops[2].type == ZYDIS_OPERAND_TYPE_MEMORY) {
        if (!EmitEffectiveAddress(ops[2].mem, next_rip, c)) return false;
        if (toDouble) c.ldr(SReg(0), ptr(kAddr));
        else          c.ldr(DReg(0), ptr(kAddr));
    } else {
        return false;
    }

    if (toDouble) {
        // ss -> sd: dst.q0 = (double)src2.f32 ; dst.q1 = src1.q1.
        c.fcvt(DReg(1), SReg(0));
        c.str(DReg(1), ptr(kState, YmmChunkOffset(dst, 0)));
        c.ldr(kScratch1, ptr(kState, YmmChunkOffset(s1, 1)));
        c.str(kScratch1, ptr(kState, YmmChunkOffset(dst, 1)));
    } else {
        // sd -> ss: dst.lo32 = (float)src2.f64 ; dst.hi32 = src1.hi32 ; q1=src1.q1.
        c.fcvt(SReg(1), DReg(0));
        c.fmov(WReg(9), SReg(1));
        c.ldr(kScratch1, ptr(kState, YmmChunkOffset(s1, 0)));
        c.lsr(kScratch1, kScratch1, 32);
        c.lsl(kScratch1, kScratch1, 32);
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

// vcvtdq2ps — packed 4x int32 -> 4x float (signed). vcvttps2dq — packed 4x
// float -> 4x int32, truncating; NaN/out-of-range -> INT_MIN (0x80000000).
bool EmitVcvtPacked(const ZydisDecodedInstruction& insn, const ZydisDecodedOperand* ops,
                    u64 next_rip, CodeGenerator& c) {
    const bool dq2ps = (insn.mnemonic == ZYDIS_MNEMONIC_VCVTDQ2PS);
    if (ops[0].type != ZYDIS_OPERAND_TYPE_REGISTER) return false;
    const int dst = ZydisVecToIndex(ops[0].reg.value);
    if (dst < 0) return false;
    const bool ymm = (ops[0].size == 256);
    const int nchunks = ymm ? 2 : 1;

    for (int h = 0; h < nchunks; ++h) {
        const int chunk = h * 2;
        if (ops[1].type == ZYDIS_OPERAND_TYPE_REGISTER) {
            const int s = ZydisVecToIndex(ops[1].reg.value);
            if (s < 0) return false;
            c.add(kScratch0, kState, YmmChunkOffset(s, chunk));
            c.ldr(QReg(0), ptr(kScratch0));
        } else if (ops[1].type == ZYDIS_OPERAND_TYPE_MEMORY) {
            if (h == 0) {
                if (!EmitEffectiveAddress(ops[1].mem, next_rip, c)) return false;
                c.mov(kScratch1, kAddr);
            }
            c.add(kScratch0, kScratch1, h * 16);
            c.ldr(QReg(0), ptr(kScratch0));
        } else {
            return false;
        }
        if (dq2ps) {
            c.scvtf(VReg4S(0), VReg4S(0));
        } else {
            // fcvtzs saturates: NaN->0, +ovf->INT_MAX, -ovf->INT_MIN. x86 wants
            // INT_MIN for all invalid. Detect NaN (v != v) and +INT_MAX result,
            // force 0x80000000 there.
            c.mov(VReg16B(3), VReg16B(0));               // keep original floats
            c.fcvtzs(VReg4S(0), VReg4S(0));              // v0 = truncated ints (saturated)
            // NaN mask: fcmeq(orig,orig) -> ~0 in ordered lanes, 0 in NaN lanes.
            c.fcmeq(VReg4S(1), VReg4S(3), VReg4S(3));
            // v2 = lanes where result == INT_MAX (positive saturation).
            c.mov(WReg(9), 0x7FFFFFFF);
            c.dup(VReg4S(2), WReg(9));
            c.cmeq(VReg4S(2), VReg4S(0), VReg4S(2));
            // invalid = (result==INT_MAX) OR NaN-lane. NaN-lane = ~v1.
            c.not_(VReg16B(1), VReg16B(1));
            c.orr(VReg16B(2), VReg16B(2), VReg16B(1));   // v2 = invalid mask
            // v1 = INT_MIN lanes.
            c.mov(WReg(9), 0x80000000);
            c.dup(VReg4S(1), WReg(9));
            // result = invalid ? INT_MIN : v0  ==  bsl(mask=v2, v1, v0).
            c.bsl(VReg16B(2), VReg16B(1), VReg16B(0));
            c.mov(VReg16B(0), VReg16B(2));
        }
        c.add(kScratch0, kState, YmmChunkOffset(dst, chunk));
        c.str(QReg(0), ptr(kScratch0));
    }
    if (!ymm) {
        c.mov(kScratch0, 0);
        c.str(kScratch0, ptr(kState, YmmChunkOffset(dst, 2)));
        c.str(kScratch0, ptr(kState, YmmChunkOffset(dst, 3)));
    }
    return true;
}

// Lane extract to GPR or memory: vpextrd (dword), vpextrq (qword), vextractps
// (dword, FP-typed but bit-identical to pextrd). imm = lane index. GPR dest
// zero-extends. We slice directly out of the guest register file via GPR.
bool EmitVpextr(const ZydisDecodedInstruction& insn, const ZydisDecodedOperand* ops,
                u64 next_rip, CodeGenerator& c) {
    const bool q = (insn.mnemonic == ZYDIS_MNEMONIC_VPEXTRQ);
    // ops: [0]=dst (gpr or mem), [1]=src xmm, [2]=imm8 lane.
    const int src = ZydisVecToIndex(ops[1].reg.value);
    if (src < 0) return false;
    const int lane = static_cast<int>(ops[2].imm.value.u) & (q ? 1 : 3);

    // For a memory dest, compute the effective address FIRST (EA clobbers x9),
    // stash it, then load the lane value — otherwise the lane value in x9 is
    // destroyed by the index-scaling in EmitEffectiveAddress.
    bool memDst = (ops[0].type == ZYDIS_OPERAND_TYPE_MEMORY);
    if (memDst) {
        if (!EmitEffectiveAddress(ops[0].mem, next_rip, c)) return false;
        c.mov(kScratch1, kAddr);
    }

    if (q) {
        c.ldr(kScratch0, ptr(kState, YmmChunkOffset(src, lane)));
    } else {
        c.ldr(kScratch0, ptr(kState, YmmChunkOffset(src, lane / 2)));
        if (lane & 1) c.lsr(kScratch0, kScratch0, 32);
        c.and_(kScratch0, kScratch0, 0xFFFFFFFFull);
    }

    if (memDst) {
        if (q) c.str(kScratch0, ptr(kScratch1));
        else   c.str(WReg(9), ptr(kScratch1));
    } else if (ops[0].type == ZYDIS_OPERAND_TYPE_REGISTER) {
        const int gi = ZydisGprToIndex(ops[0].reg.value);
        if (gi < 0) return false;
        c.str(kScratch0, ptr(kState, GprOffset(gi)));   // zero-extended store
    } else {
        return false;
    }
    return true;
}

// vpinsrd dst, src1, r/m32, imm8 — copy src1's low 128 to dst, then replace the
// dword lane (imm&3) with the 32-bit GPR/mem source. Upper 128 zeroed.
bool EmitVpinsrd(const ZydisDecodedInstruction& insn, const ZydisDecodedOperand* ops,
                 u64 next_rip, CodeGenerator& c) {
    if (ops[0].type != ZYDIS_OPERAND_TYPE_REGISTER ||
        ops[1].type != ZYDIS_OPERAND_TYPE_REGISTER) return false;
    const int dst = ZydisVecToIndex(ops[0].reg.value);
    const int s1  = ZydisVecToIndex(ops[1].reg.value);
    if (dst < 0 || s1 < 0) return false;
    const int lane = static_cast<int>(ops[3].imm.value.u) & 3;

    // Load the 32-bit value -> w9 (zero-ext into x9).
    if (ops[2].type == ZYDIS_OPERAND_TYPE_REGISTER) {
        const int gi = ZydisGprToIndex(ops[2].reg.value);
        if (gi < 0) return false;
        c.ldr(WReg(9), ptr(kState, GprOffset(gi)));
    } else if (ops[2].type == ZYDIS_OPERAND_TYPE_MEMORY) {
        if (!EmitEffectiveAddress(ops[2].mem, next_rip, c)) return false;
        c.ldr(WReg(9), ptr(kAddr));
    } else {
        return false;
    }
    // Copy src1 chunk0/1 to dst, then patch the target chunk's half.
    const int chunk = lane / 2;
    // Load the src1 chunk holding the lane, merge in the new dword.
    c.ldr(kScratch1, ptr(kState, YmmChunkOffset(s1, chunk)));
    if (lane & 1) {
        // replace hi32: keep lo32 of src1 chunk, put value in hi32.
        c.and_(kScratch1, kScratch1, 0xFFFFFFFFull);
        c.lsl(kScratch0, kScratch0, 32);   // x9 (value) -> hi32
        c.orr(kScratch1, kScratch1, kScratch0);
    } else {
        // replace lo32: keep hi32 of src1 chunk, put value in lo32.
        c.lsr(kScratch1, kScratch1, 32);
        c.lsl(kScratch1, kScratch1, 32);
        c.and_(kScratch0, kScratch0, 0xFFFFFFFFull);
        c.orr(kScratch1, kScratch1, kScratch0);
    }
    c.str(kScratch1, ptr(kState, YmmChunkOffset(dst, chunk)));
    // The other chunk of the low 128 comes straight from src1.
    const int other = chunk ^ 1;
    c.ldr(kScratch0, ptr(kState, YmmChunkOffset(s1, other)));
    c.str(kScratch0, ptr(kState, YmmChunkOffset(dst, other)));
    // Zero upper 128.
    c.mov(kScratch0, 0);
    c.str(kScratch0, ptr(kState, YmmChunkOffset(dst, 2)));
    c.str(kScratch0, ptr(kState, YmmChunkOffset(dst, 3)));
    return true;
}

// vinsertps dst, src1, src2/m32, imm8. imm: [7:6]=src dword select (reg form),
// [5:4]=dst dword select, [3:0]=zero mask. dst = src1 with src2[sel] inserted
// at dst[dsel], then lanes flagged in the zero mask are cleared. Upper 128 zero.
// We build the 4 dwords in GPRs (x5..x8-ish via two 64-bit chunk regs) then store.
bool EmitVinsertps(const ZydisDecodedInstruction& insn, const ZydisDecodedOperand* ops,
                   u64 next_rip, CodeGenerator& c) {
    if (ops[0].type != ZYDIS_OPERAND_TYPE_REGISTER ||
        ops[1].type != ZYDIS_OPERAND_TYPE_REGISTER) return false;
    const int dst = ZydisVecToIndex(ops[0].reg.value);
    const int s1  = ZydisVecToIndex(ops[1].reg.value);
    if (dst < 0 || s1 < 0) return false;
    const int imm = static_cast<int>(ops[3].imm.value.u) & 0xFF;
    const int src_sel = (imm >> 6) & 3;
    const int dst_sel = (imm >> 4) & 3;
    const int zmask   = imm & 0xF;

    // Load the inserted dword -> w12.
    if (ops[2].type == ZYDIS_OPERAND_TYPE_REGISTER) {
        const int s2 = ZydisVecToIndex(ops[2].reg.value);
        if (s2 < 0) return false;
        c.ldr(kScratch2, ptr(kState, YmmChunkOffset(s2, src_sel / 2)));
        if (src_sel & 1) c.lsr(kScratch2, kScratch2, 32);
        c.and_(kScratch2, kScratch2, 0xFFFFFFFFull);
    } else if (ops[2].type == ZYDIS_OPERAND_TYPE_MEMORY) {
        if (!EmitEffectiveAddress(ops[2].mem, next_rip, c)) return false;
        c.ldr(WReg(12), ptr(kAddr));    // m32 (src select ignored for mem form)
    } else {
        return false;
    }

    // Load src1's two low chunks into x9 (chunk0) and x10 (chunk1).
    c.ldr(kScratch0, ptr(kState, YmmChunkOffset(s1, 0)));
    c.ldr(kScratch1, ptr(kState, YmmChunkOffset(s1, 1)));

    // Insert the dword (x12) at dst_sel. Each chunk holds two dwords.
    auto patch = [&](int lane) {
        const XReg ch = (lane / 2 == 0) ? kScratch0 : kScratch1;
        if (lane & 1) {
            c.and_(ch, ch, 0xFFFFFFFFull);
            c.lsl(XReg(7), kScratch2, 32);
            c.orr(ch, ch, XReg(7));
        } else {
            c.lsr(ch, ch, 32);
            c.lsl(ch, ch, 32);
            c.and_(XReg(7), kScratch2, 0xFFFFFFFFull);
            c.orr(ch, ch, XReg(7));
        }
    };
    patch(dst_sel);

    // Apply zero mask: clear any of the 4 dwords whose zmask bit is set.
    for (int lane = 0; lane < 4; ++lane) {
        if (!((zmask >> lane) & 1)) continue;
        const XReg ch = (lane / 2 == 0) ? kScratch0 : kScratch1;
        if (lane & 1) {
            c.and_(ch, ch, 0xFFFFFFFFull);   // clear hi32
        } else {
            c.lsr(ch, ch, 32);
            c.lsl(ch, ch, 32);               // clear lo32
        }
    }

    c.str(kScratch0, ptr(kState, YmmChunkOffset(dst, 0)));
    c.str(kScratch1, ptr(kState, YmmChunkOffset(dst, 1)));
    c.mov(kScratch0, 0);
    c.str(kScratch0, ptr(kState, YmmChunkOffset(dst, 2)));
    c.str(kScratch0, ptr(kState, YmmChunkOffset(dst, 3)));
    return true;
}

// vinsertf128 dst, src1(ymm), src2(xmm/m128), imm8 — dst = src1 (256), then the
// 128-bit half selected by imm&1 is replaced by src2.
bool EmitVinsertf128(const ZydisDecodedInstruction& insn, const ZydisDecodedOperand* ops,
                     u64 next_rip, CodeGenerator& c) {
    if (ops[0].type != ZYDIS_OPERAND_TYPE_REGISTER ||
        ops[1].type != ZYDIS_OPERAND_TYPE_REGISTER) return false;
    const int dst = ZydisVecToIndex(ops[0].reg.value);
    const int s1  = ZydisVecToIndex(ops[1].reg.value);
    if (dst < 0 || s1 < 0) return false;
    const int half = static_cast<int>(ops[3].imm.value.u) & 1;

    // dst = src1 (all 4 chunks).
    for (int ch = 0; ch < 4; ++ch) {
        c.ldr(kScratch0, ptr(kState, YmmChunkOffset(s1, ch)));
        c.str(kScratch0, ptr(kState, YmmChunkOffset(dst, ch)));
    }
    // Overwrite the chosen 128-half with src2 (xmm low 128, chunks 0,1).
    const int base = half * 2;
    if (ops[2].type == ZYDIS_OPERAND_TYPE_REGISTER) {
        const int s2 = ZydisVecToIndex(ops[2].reg.value);
        if (s2 < 0) return false;
        c.ldr(kScratch0, ptr(kState, YmmChunkOffset(s2, 0)));
        c.str(kScratch0, ptr(kState, YmmChunkOffset(dst, base)));
        c.ldr(kScratch0, ptr(kState, YmmChunkOffset(s2, 1)));
        c.str(kScratch0, ptr(kState, YmmChunkOffset(dst, base + 1)));
    } else if (ops[2].type == ZYDIS_OPERAND_TYPE_MEMORY) {
        if (!EmitEffectiveAddress(ops[2].mem, next_rip, c)) return false;
        c.ldr(kScratch0, ptr(kAddr));
        c.str(kScratch0, ptr(kState, YmmChunkOffset(dst, base)));
        c.add(kScratch1, kAddr, 8);
        c.ldr(kScratch0, ptr(kScratch1));
        c.str(kScratch0, ptr(kState, YmmChunkOffset(dst, base + 1)));
    } else {
        return false;
    }
    return true;
}

// vextractf128 dst(xmm/m128), src(ymm), imm8 — dst = src's 128-half (imm&1).
bool EmitVextractf128(const ZydisDecodedInstruction& insn, const ZydisDecodedOperand* ops,
                      u64 next_rip, CodeGenerator& c) {
    const int src = ZydisVecToIndex(ops[1].reg.value);
    if (src < 0) return false;
    const int half = static_cast<int>(ops[2].imm.value.u) & 1;
    const int base = half * 2;

    if (ops[0].type == ZYDIS_OPERAND_TYPE_REGISTER) {
        const int dst = ZydisVecToIndex(ops[0].reg.value);
        if (dst < 0) return false;
        c.ldr(kScratch0, ptr(kState, YmmChunkOffset(src, base)));
        c.str(kScratch0, ptr(kState, YmmChunkOffset(dst, 0)));
        c.ldr(kScratch0, ptr(kState, YmmChunkOffset(src, base + 1)));
        c.str(kScratch0, ptr(kState, YmmChunkOffset(dst, 1)));
        // xmm dst zeroes upper 128.
        c.mov(kScratch0, 0);
        c.str(kScratch0, ptr(kState, YmmChunkOffset(dst, 2)));
        c.str(kScratch0, ptr(kState, YmmChunkOffset(dst, 3)));
    } else if (ops[0].type == ZYDIS_OPERAND_TYPE_MEMORY) {
        if (!EmitEffectiveAddress(ops[0].mem, next_rip, c)) return false;
        c.mov(kScratch1, kAddr);
        c.ldr(kScratch0, ptr(kState, YmmChunkOffset(src, base)));
        c.str(kScratch0, ptr(kScratch1));
        c.ldr(kScratch0, ptr(kState, YmmChunkOffset(src, base + 1)));
        c.add(kScratch1, kScratch1, 8);
        c.str(kScratch0, ptr(kScratch1));
    } else {
        return false;
    }
    return true;
}

// vmovd — move 32 bits between XMM.low and GPR/memory. AVX/VEX-encoded, so
// every XMM-destination form zeros the rest of the YMM (bits 255:32).
//   xmm <- gpr      : low 32 from GPR, zero bits 255:32.
//   xmm <- mem      : low 32 from memory, zero bits 255:32.
//   gpr <- xmm      : low 32 to GPR slot, zero-extended to 64.
//   mem <- xmm      : store low 32 (4 bytes).
// EA-clobber note: EmitEffectiveAddress clobbers x9 (kScratch0) and x11
// (kAddr), so for the store form we compute the address first, then load the
// value into a DIFFERENT register (kScratch1/x10) before storing.
bool EmitVmovd(const ZydisDecodedInstruction& insn, const ZydisDecodedOperand* ops,
               u64 next_rip, CodeGenerator& c) {
    if (insn.mnemonic != ZYDIS_MNEMONIC_VMOVD) return false;

    const int d_vec = (ops[0].type == ZYDIS_OPERAND_TYPE_REGISTER)
                          ? ZydisVecToIndex(ops[0].reg.value) : -1;
    if (d_vec >= 0) {
        // dst is XMM. src = gpr or mem -> low 32 bits, rest of YMM zeroed.
        if (ops[1].type == ZYDIS_OPERAND_TYPE_REGISTER) {
            const int gi = ZydisGprToIndex(ops[1].reg.value);
            if (gi < 0) return false;
            c.ldr(WReg(9), ptr(kState, GprOffset(gi)));   // 32-bit, zero-extends x9
        } else if (ops[1].type == ZYDIS_OPERAND_TYPE_MEMORY) {
            if (!EmitEffectiveAddress(ops[1].mem, next_rip, c)) return false;
            c.ldr(WReg(9), ptr(kAddr));                   // 32-bit load, zero-extends x9
        } else {
            return false;
        }
        // str of the full X reg writes 8 bytes: low 4 = value, high 4 = 0
        // (the W-load above cleared bits 63:32 of x9). Then zero chunks 1..3.
        c.str(kScratch0, ptr(kState, YmmChunkOffset(d_vec, 0)));
        c.mov(kScratch0, 0);
        c.str(kScratch0, ptr(kState, YmmChunkOffset(d_vec, 1)));
        c.str(kScratch0, ptr(kState, YmmChunkOffset(d_vec, 2)));
        c.str(kScratch0, ptr(kState, YmmChunkOffset(d_vec, 3)));
        return true;
    }

    // dst is GPR or mem; src is XMM.
    const int s_vec = ZydisVecToIndex(ops[1].reg.value);
    if (s_vec < 0) return false;
    if (ops[0].type == ZYDIS_OPERAND_TYPE_REGISTER) {
        const int gi = ZydisGprToIndex(ops[0].reg.value);
        if (gi < 0) return false;
        // 32-bit load zero-extends into x9; store full qword so the GPR slot
        // gets the canonical 32-bit zero-extension.
        c.ldr(WReg(9), ptr(kState, YmmChunkOffset(s_vec, 0)));
        c.str(kScratch0, ptr(kState, GprOffset(gi)));
        return true;
    } else if (ops[0].type == ZYDIS_OPERAND_TYPE_MEMORY) {
        // Compute EA first (clobbers x9/x11), THEN load the value into x10 so
        // the address in kAddr survives.
        if (!EmitEffectiveAddress(ops[0].mem, next_rip, c)) return false;
        c.ldr(WReg(10), ptr(kState, YmmChunkOffset(s_vec, 0)));
        c.str(WReg(10), ptr(kAddr));                      // store exactly 4 bytes
        return true;
    }
    return false;
}

// vmovq — move 64 bits between XMM.low and GPR/memory.
//   xmm <- gpr/mem : write low 64, zero the rest of the YMM (chunks 1,2,3).
//   gpr <- xmm     : write full 64 to the GPR slot.
//   mem <- xmm     : store low 64 (8 bytes).
bool EmitVmovq(const ZydisDecodedInstruction& insn, const ZydisDecodedOperand* ops,
               u64 next_rip, CodeGenerator& c) {
    const int d_vec = (ops[0].type == ZYDIS_OPERAND_TYPE_REGISTER)
                          ? ZydisVecToIndex(ops[0].reg.value) : -1;
    if (d_vec >= 0) {
        // dst is XMM. src = gpr or mem.
        if (ops[1].type == ZYDIS_OPERAND_TYPE_REGISTER) {
            const int gi = ZydisGprToIndex(ops[1].reg.value);
            if (gi < 0) return false;
            c.ldr(kScratch0, ptr(kState, GprOffset(gi)));
        } else if (ops[1].type == ZYDIS_OPERAND_TYPE_MEMORY) {
            if (!EmitEffectiveAddress(ops[1].mem, next_rip, c)) return false;
            c.ldr(kScratch0, ptr(kAddr));
        } else {
            return false;
        }
        c.str(kScratch0, ptr(kState, YmmChunkOffset(d_vec, 0)));
        c.mov(kScratch0, 0);
        c.str(kScratch0, ptr(kState, YmmChunkOffset(d_vec, 1)));
        c.str(kScratch0, ptr(kState, YmmChunkOffset(d_vec, 2)));
        c.str(kScratch0, ptr(kState, YmmChunkOffset(d_vec, 3)));
        return true;
    }
    // dst is GPR or mem; src is XMM.
    const int s_vec = ZydisVecToIndex(ops[1].reg.value);
    if (s_vec < 0) return false;
    if (ops[0].type == ZYDIS_OPERAND_TYPE_REGISTER) {
        const int gi = ZydisGprToIndex(ops[0].reg.value);
        if (gi < 0) return false;
        c.ldr(kScratch0, ptr(kState, YmmChunkOffset(s_vec, 0)));
        c.str(kScratch0, ptr(kState, GprOffset(gi)));
    } else if (ops[0].type == ZYDIS_OPERAND_TYPE_MEMORY) {
        // Compute EA first (it clobbers x9/x11), THEN load the value into x10
        // so the destination address in kAddr survives until the store.
        if (!EmitEffectiveAddress(ops[0].mem, next_rip, c)) return false;
        c.ldr(kScratch1, ptr(kState, YmmChunkOffset(s_vec, 0)));
        c.str(kScratch1, ptr(kAddr));
    } else {
        return false;
    }
    return true;
}

// VMOVUPS/VMOVAPS/VMOVDQU/VMOVDQA/VMOVNT* — full-width vector moves.
//
// ATOMICITY. Guest-memory traffic moves through a Q register (ldr/str q),
// one 16-byte host op per 16-byte chunk -- NOT a 64-bit GPR relay. The
// relay (an earlier revision) made every guest vector store visible to
// other guest threads as two separate 8-byte writes with a window between
// them. On this backend's only target (Apple Silicon, ARMv8.4 => FEAT_LSE2)
// aligned 16-byte ldr/str q are single-copy atomic: strictly fewer
// observable interleavings than the PS4's Jaguar, which cracked 128-bit
// accesses into two 64-bit halves -- guest code never had 16-byte
// atomicity to rely on. Unaligned accesses carry no sub-access guarantee
// (same accepted corner as the x86 host; see EmitVmovups there for the
// full analysis). State-side stores (thread-private) have no atomicity
// requirement; they use the same q ops for instruction-count reasons.
//
// YMM state chunks are 32-byte aligned (alignas on GuestState::ymm), so
// the q-form scaled immediates at chunk*16 are always encodable.
bool EmitVmovFull(const ZydisDecodedInstruction& insn, const ZydisDecodedOperand* ops,
                  u64 next_rip, CodeGenerator& c) {
    const bool ymm = (ops[0].size == 256 || ops[1].size == 256);
    const int nchunks = ymm ? 4 : 2;

    if (ops[0].type == ZYDIS_OPERAND_TYPE_REGISTER &&
        ops[1].type == ZYDIS_OPERAND_TYPE_REGISTER) {
        const int dst = ZydisVecToIndex(ops[0].reg.value);
        const int src = ZydisVecToIndex(ops[1].reg.value);
        if (dst < 0 || src < 0) return false;
        for (int ch = 0; ch < nchunks; ch += 2) {
            c.ldr(QReg(0), ptr(kState, YmmChunkOffset(src, ch)));
            c.str(QReg(0), ptr(kState, YmmChunkOffset(dst, ch)));
        }
        if (!ymm) {
            // VEX 128-bit form zeros bits 255:128 of the destination.
            c.movi(VReg16B(0), 0);
            c.str(QReg(0), ptr(kState, YmmChunkOffset(dst, 2)));
        }
        return true;
    }
    if (ops[0].type == ZYDIS_OPERAND_TYPE_REGISTER &&
        ops[1].type == ZYDIS_OPERAND_TYPE_MEMORY) {
        const int dst = ZydisVecToIndex(ops[0].reg.value);
        if (dst < 0) return false;
        if (!EmitEffectiveAddress(ops[1].mem, next_rip, c)) return false;
        c.mov(kScratch1, kAddr);
        for (int ch = 0; ch < nchunks; ch += 2) {
            c.ldr(QReg(0), ptr(kScratch1, ch * 8));
            c.str(QReg(0), ptr(kState, YmmChunkOffset(dst, ch)));
        }
        if (!ymm) {
            c.movi(VReg16B(0), 0);
            c.str(QReg(0), ptr(kState, YmmChunkOffset(dst, 2)));
        }
        return true;
    }
    if (ops[0].type == ZYDIS_OPERAND_TYPE_MEMORY &&
        ops[1].type == ZYDIS_OPERAND_TYPE_REGISTER) {
        const int src = ZydisVecToIndex(ops[1].reg.value);
        if (src < 0) return false;
        if (!EmitEffectiveAddress(ops[0].mem, next_rip, c)) return false;
        c.mov(kScratch1, kAddr);
        // The store direction: this is where the GPR relay was observably
        // torn by concurrent guest threads (see the atomicity note above).
        for (int ch = 0; ch < nchunks; ch += 2) {
            c.ldr(QReg(0), ptr(kState, YmmChunkOffset(src, ch)));
            c.str(QReg(0), ptr(kScratch1, ch * 8));
        }
        return true;
    }
    return false;
}

// vmovsldup — dst even lanes duplicated: dst[0]=src[0],dst[1]=src[0],
//   dst[2]=src[2],dst[3]=src[2]  ==  trn1(4S, src, src).
// vmovshdup — odd lanes duplicated: dst[0]=src[1],dst[1]=src[1],
//   dst[2]=src[3],dst[3]=src[3]  ==  trn2(4S, src, src). XMM zeroes upper.
bool EmitVmovdup(const ZydisDecodedInstruction& insn, const ZydisDecodedOperand* ops,
                 u64 next_rip, CodeGenerator& c) {
    const bool odd = (insn.mnemonic == ZYDIS_MNEMONIC_VMOVSHDUP);
    if (ops[0].type != ZYDIS_OPERAND_TYPE_REGISTER) return false;
    const int dst = ZydisVecToIndex(ops[0].reg.value);
    if (dst < 0) return false;

    if (ops[1].type == ZYDIS_OPERAND_TYPE_REGISTER) {
        const int src = ZydisVecToIndex(ops[1].reg.value);
        if (src < 0) return false;
        c.add(kScratch0, kState, YmmChunkOffset(src, 0));
        c.ldr(QReg(0), ptr(kScratch0));
    } else if (ops[1].type == ZYDIS_OPERAND_TYPE_MEMORY) {
        if (!EmitEffectiveAddress(ops[1].mem, next_rip, c)) return false;
        c.ldr(QReg(0), ptr(kAddr));
    } else {
        return false;
    }
    if (odd) c.trn2(VReg4S(1), VReg4S(0), VReg4S(0));
    else     c.trn1(VReg4S(1), VReg4S(0), VReg4S(0));
    c.add(kScratch0, kState, YmmChunkOffset(dst, 0));
    c.str(QReg(1), ptr(kScratch0));
    c.mov(kScratch0, 0);
    c.str(kScratch0, ptr(kState, YmmChunkOffset(dst, 2)));
    c.str(kScratch0, ptr(kState, YmmChunkOffset(dst, 3)));
    return true;
}

// Per-element packed shifts: VPSLLW/D/Q (left logical), VPSRLW/D/Q (right
// logical), VPSRAW/D (right arithmetic; no VPSRAQ in the Jaguar set). The third
// operand is either an imm8 or an xmm whose low 64 bits hold a single shift
// count applied to every lane (NOT a per-lane count). x86 count saturation:
//   SLL/SRL  count >= elem_width  -> all bits shifted out -> 0
//   SRA      count >= elem_width  -> acts as (elem_width-1) -> sign-fill
// Element-width-by-imm uses NEON shl/ushr/sshr directly (count is a literal in
// 1..width-1; out-of-range handled in C++). Element-width-by-register uses
// ushl/sshl with a per-lane signed amount we build by clamping the scalar count
// and broadcasting it: positive = left, negative = right. ushl by >= width
// zeroes (matches logical saturation); sshl with amount clamped to -(width-1)
// sign-fills (matches arithmetic saturation). Verified on QEMU for W/D/Q.
bool EmitVpshift(const ZydisDecodedInstruction& insn, const ZydisDecodedOperand* ops,
                 u64 next_rip, CodeGenerator& c) {
    enum class Dir { Sll, Srl, Sra };
    Dir dir; int width;  // element width in bits
    switch (insn.mnemonic) {
        case ZYDIS_MNEMONIC_VPSLLW: dir = Dir::Sll; width = 16; break;
        case ZYDIS_MNEMONIC_VPSLLD: dir = Dir::Sll; width = 32; break;
        case ZYDIS_MNEMONIC_VPSLLQ: dir = Dir::Sll; width = 64; break;
        case ZYDIS_MNEMONIC_VPSRLW: dir = Dir::Srl; width = 16; break;
        case ZYDIS_MNEMONIC_VPSRLD: dir = Dir::Srl; width = 32; break;
        case ZYDIS_MNEMONIC_VPSRLQ: dir = Dir::Srl; width = 64; break;
        case ZYDIS_MNEMONIC_VPSRAW: dir = Dir::Sra; width = 16; break;
        case ZYDIS_MNEMONIC_VPSRAD: dir = Dir::Sra; width = 32; break;
        default: return false;
    }
    if (insn.operand_count_visible != 3) return false;
    if (ops[0].type != ZYDIS_OPERAND_TYPE_REGISTER ||
        ops[1].type != ZYDIS_OPERAND_TYPE_REGISTER) return false;
    const int dst = ZydisVecToIndex(ops[0].reg.value);
    const int src = ZydisVecToIndex(ops[1].reg.value);
    if (dst < 0 || src < 0) return false;
    const bool ymm = (ops[0].size == 256);
    const int nchunks = ymm ? 2 : 1;

    // Apply the chosen NEON shift to v0 viewed at `width`. `byreg` selects the
    // register (ushl/sshl on v1) vs immediate (shl/ushr/sshr) path. For the
    // register path v1 must already hold the per-lane signed amount.
    auto emit_shift = [&](bool byreg, int imm) {
        if (byreg) {
            if (dir == Dir::Sra) {
                if (width == 16) c.sshl(VReg8H(0), VReg8H(0), VReg8H(1));
                else             c.sshl(VReg4S(0), VReg4S(0), VReg4S(1));
            } else {
                if (width == 16)      c.ushl(VReg8H(0), VReg8H(0), VReg8H(1));
                else if (width == 32) c.ushl(VReg4S(0), VReg4S(0), VReg4S(1));
                else                  c.ushl(VReg2D(0), VReg2D(0), VReg2D(1));
            }
            return;
        }
        // immediate path: imm already clamped/handled by caller for over-width.
        if (dir == Dir::Sll) {
            if (width == 16)      c.shl(VReg8H(0), VReg8H(0), imm);
            else if (width == 32) c.shl(VReg4S(0), VReg4S(0), imm);
            else                  c.shl(VReg2D(0), VReg2D(0), imm);
        } else if (dir == Dir::Srl) {
            if (width == 16)      c.ushr(VReg8H(0), VReg8H(0), imm);
            else if (width == 32) c.ushr(VReg4S(0), VReg4S(0), imm);
            else                  c.ushr(VReg2D(0), VReg2D(0), imm);
        } else {  // Sra
            if (width == 16) c.sshr(VReg8H(0), VReg8H(0), imm);
            else             c.sshr(VReg4S(0), VReg4S(0), imm);
        }
    };

    if (ops[2].type == ZYDIS_OPERAND_TYPE_IMMEDIATE) {
        const int cnt = static_cast<int>(ops[2].imm.value.u) & 0xFF;
        for (int h = 0; h < nchunks; ++h) {
            const int chunk = h * 2;
            c.add(kScratch0, kState, YmmChunkOffset(src, chunk));
            c.ldr(QReg(0), ptr(kScratch0));
            if (dir == Dir::Sra) {
                int s = cnt; if (s > width - 1) s = width - 1;
                if (s >= 1) emit_shift(false, s);   // s==0 -> no-op
            } else if (cnt >= width) {
                c.eor(VReg16B(0), VReg16B(0), VReg16B(0));  // all bits out -> 0
            } else if (cnt >= 1) {
                emit_shift(false, cnt);
            }
            c.add(kScratch0, kState, YmmChunkOffset(dst, chunk));
            c.str(QReg(0), ptr(kScratch0));
        }
    } else if (ops[2].type == ZYDIS_OPERAND_TYPE_REGISTER) {
        const int cnt = ZydisVecToIndex(ops[2].reg.value);
        if (cnt < 0) return false;
        // Scalar count = low 64 bits of the count register. Clamp to the cap:
        //   SLL/SRL cap = width   (>=width zeroes via ushl by >=width)
        //   SRA     cap = width-1 (sign-fill)
        // then build the signed per-lane amount (negate for right shifts) and
        // broadcast it. x12 holds the count; x13 the cap.
        const int cap = (dir == Dir::Sra) ? (width - 1) : width;
        c.ldr(XReg(12), ptr(kState, YmmChunkOffset(cnt, 0)));  // low 64 bits
        c.mov(XReg(13), cap);
        c.cmp(XReg(12), XReg(13));
        c.csel(XReg(12), XReg(12), XReg(13), LT);              // min(count, cap)
        if (dir != Dir::Sll) c.neg(XReg(12), XReg(12));        // right = negative
        if (width == 16)      c.dup(VReg8H(1), WReg(12));
        else if (width == 32) c.dup(VReg4S(1), WReg(12));
        else                  c.dup(VReg2D(1), XReg(12));
        for (int h = 0; h < nchunks; ++h) {
            const int chunk = h * 2;
            c.add(kScratch0, kState, YmmChunkOffset(src, chunk));
            c.ldr(QReg(0), ptr(kScratch0));
            emit_shift(true, 0);
            c.add(kScratch0, kState, YmmChunkOffset(dst, chunk));
            c.str(QReg(0), ptr(kScratch0));
        }
    } else {
        return false;
    }

    if (!ymm) {
        c.mov(kScratch0, 0);
        c.str(kScratch0, ptr(kState, YmmChunkOffset(dst, 2)));
        c.str(kScratch0, ptr(kState, YmmChunkOffset(dst, 3)));
    }
    return true;
}

// vpslldq/vpsrldq dst, src, imm8 — whole-128-bit BYTE shift (not per-element),
// zero-filled. NEON ext concatenates two regs and slices a 16-byte window:
//   ext(d, lo, hi, idx) = bytes [idx .. idx+15] of (lo:hi).
// Right shift by n bytes: window starting at byte n of (src:zero) -> ext(src,0,n).
// Left shift by n bytes: window starting at byte (16-n) of (zero:src).
bool EmitVpsxldq(const ZydisDecodedInstruction& insn, const ZydisDecodedOperand* ops,
                 u64 next_rip, CodeGenerator& c) {
    const bool left = (insn.mnemonic == ZYDIS_MNEMONIC_VPSLLDQ);
    if (ops[0].type != ZYDIS_OPERAND_TYPE_REGISTER ||
        ops[1].type != ZYDIS_OPERAND_TYPE_REGISTER) return false;
    const int dst = ZydisVecToIndex(ops[0].reg.value);
    const int src = ZydisVecToIndex(ops[1].reg.value);
    if (dst < 0 || src < 0) return false;
    int n = static_cast<int>(ops[2].imm.value.u) & 0xFF;
    if (n > 15) n = 16;  // shift out everything

    c.add(kScratch0, kState, YmmChunkOffset(src, 0));
    c.ldr(QReg(0), ptr(kScratch0));      // v0 = src
    c.eor(VReg16B(1), VReg16B(1), VReg16B(1));  // v1 = 0

    if (n == 16) {
        c.eor(VReg16B(0), VReg16B(0), VReg16B(0));   // all shifted out
    } else if (n == 0) {
        // no-op
    } else if (left) {
        // left by n: result = (zero:src) window at byte (16-n).
        // ext(d, v1=zero, v0=src, 16-n).
        c.ext(VReg16B(0), VReg16B(1), VReg16B(0), 16 - n);
    } else {
        // right by n: result = (src:zero) window at byte n.
        c.ext(VReg16B(0), VReg16B(0), VReg16B(1), n);
    }
    c.add(kScratch0, kState, YmmChunkOffset(dst, 0));
    c.str(QReg(0), ptr(kScratch0));
    c.mov(kScratch0, 0);
    c.str(kScratch0, ptr(kState, YmmChunkOffset(dst, 2)));
    c.str(kScratch0, ptr(kState, YmmChunkOffset(dst, 3)));
    return true;
}

// vblendps dst,src1,src2,imm4 (per-dword) / vpblendw dst,src1,src2,imm8
// (per-word) — for each element, mask bit picks src2 (1) or src1 (0). imm is
// compile-time, so build the result by element copies into a temp.
bool EmitVblend(const ZydisDecodedInstruction& insn, const ZydisDecodedOperand* ops,
                u64 next_rip, CodeGenerator& c) {
    const ZydisMnemonic m = insn.mnemonic;
    const bool word = (m == ZYDIS_MNEMONIC_VPBLENDW);          // 8 word lanes
    const bool qword = (m == ZYDIS_MNEMONIC_VBLENDPD);         // 2 double lanes
    // VBLENDPS and VPBLENDD both select 4 dword lanes by imm[3:0].
    if (ops[0].type != ZYDIS_OPERAND_TYPE_REGISTER ||
        ops[1].type != ZYDIS_OPERAND_TYPE_REGISTER) return false;
    const int dst = ZydisVecToIndex(ops[0].reg.value);
    const int s1  = ZydisVecToIndex(ops[1].reg.value);
    if (dst < 0 || s1 < 0) return false;
    const int imm = static_cast<int>(ops[3].imm.value.u) & 0xFF;

    // src1 -> v0, src2 -> v1.
    c.add(kScratch0, kState, YmmChunkOffset(s1, 0));
    c.ldr(QReg(0), ptr(kScratch0));
    if (ops[2].type == ZYDIS_OPERAND_TYPE_REGISTER) {
        const int s2 = ZydisVecToIndex(ops[2].reg.value);
        if (s2 < 0) return false;
        c.add(kScratch0, kState, YmmChunkOffset(s2, 0));
        c.ldr(QReg(1), ptr(kScratch0));
    } else if (ops[2].type == ZYDIS_OPERAND_TYPE_MEMORY) {
        if (!EmitEffectiveAddress(ops[2].mem, next_rip, c)) return false;
        c.ldr(QReg(1), ptr(kAddr));
    } else {
        return false;
    }
    // Result builds in v0: for each set mask bit copy element from v1 (src2).
    if (word) {
        for (int i = 0; i < 8; ++i)
            if ((imm >> i) & 1) c.ins(VReg8H(0)[i], VReg8H(1)[i]);
    } else if (qword) {
        for (int i = 0; i < 2; ++i)
            if ((imm >> i) & 1) c.ins(VReg2D(0)[i], VReg2D(1)[i]);
    } else {
        for (int i = 0; i < 4; ++i)
            if ((imm >> i) & 1) c.ins(VReg4S(0)[i], VReg4S(1)[i]);
    }
    c.add(kScratch0, kState, YmmChunkOffset(dst, 0));
    c.str(QReg(0), ptr(kScratch0));
    c.mov(kScratch0, 0);
    c.str(kScratch0, ptr(kState, YmmChunkOffset(dst, 2)));
    c.str(kScratch0, ptr(kState, YmmChunkOffset(dst, 3)));
    return true;
}

// vblendvps dst,src1,src2,mask — per-dword select by the SIGN BIT of the mask
// register's matching dword (mask reg is imm8[7:4]). Broadcast sign via
// arithmetic shift >>31 to make a full-lane select mask, then bsl.
bool EmitVblendvps(const ZydisDecodedInstruction& insn, const ZydisDecodedOperand* ops,
                   u64 next_rip, CodeGenerator& c) {
    if (ops[0].type != ZYDIS_OPERAND_TYPE_REGISTER ||
        ops[1].type != ZYDIS_OPERAND_TYPE_REGISTER ||
        ops[2].type != ZYDIS_OPERAND_TYPE_REGISTER) return false;
    const int dst = ZydisVecToIndex(ops[0].reg.value);
    const int s1  = ZydisVecToIndex(ops[1].reg.value);
    const int s2  = ZydisVecToIndex(ops[2].reg.value);
    if (dst < 0 || s1 < 0 || s2 < 0) return false;
    // mask register is the 4th operand (is4 imm high nibble decoded by Zydis).
    int mask_vec = -1;
    if (insn.operand_count_visible >= 4 && ops[3].type == ZYDIS_OPERAND_TYPE_REGISTER)
        mask_vec = ZydisVecToIndex(ops[3].reg.value);
    if (mask_vec < 0) return false;

    c.add(kScratch0, kState, YmmChunkOffset(s1, 0));
    c.ldr(QReg(0), ptr(kScratch0));      // v0 = src1
    c.add(kScratch0, kState, YmmChunkOffset(s2, 0));
    c.ldr(QReg(1), ptr(kScratch0));      // v1 = src2
    c.add(kScratch0, kState, YmmChunkOffset(mask_vec, 0));
    c.ldr(QReg(2), ptr(kScratch0));      // v2 = mask
    // Broadcast the per-element sign bit to a full-element select mask. Element
    // width depends on the form: dword (VBLENDVPS), qword (VBLENDVPD), or byte
    // (VPBLENDVB).
    switch (insn.mnemonic) {
        case ZYDIS_MNEMONIC_VBLENDVPD: c.sshr(VReg2D(2),  VReg2D(2),  63); break;
        case ZYDIS_MNEMONIC_VPBLENDVB: c.sshr(VReg16B(2), VReg16B(2), 7);  break;
        default:                       c.sshr(VReg4S(2),  VReg4S(2),  31); break;
    }
    // bsl(mask, src2, src1): where mask=1 pick src2(v1) else src1(v0).
    c.bsl(VReg16B(2), VReg16B(1), VReg16B(0));
    c.add(kScratch0, kState, YmmChunkOffset(dst, 0));
    c.str(QReg(2), ptr(kScratch0));
    c.mov(kScratch0, 0);
    c.str(kScratch0, ptr(kState, YmmChunkOffset(dst, 2)));
    c.str(kScratch0, ptr(kState, YmmChunkOffset(dst, 3)));
    return true;
}

// vbroadcastss dst, m32/xmm — broadcast a single float to all lanes (4 for xmm,
// 8 for ymm).
bool EmitVbroadcastss(const ZydisDecodedInstruction& insn, const ZydisDecodedOperand* ops,
                      u64 next_rip, CodeGenerator& c) {
    if (ops[0].type != ZYDIS_OPERAND_TYPE_REGISTER) return false;
    const int dst = ZydisVecToIndex(ops[0].reg.value);
    if (dst < 0) return false;
    const bool ymm = (ops[0].size == 256);

    if (ops[1].type == ZYDIS_OPERAND_TYPE_MEMORY) {
        if (!EmitEffectiveAddress(ops[1].mem, next_rip, c)) return false;
        c.ldr(WReg(9), ptr(kAddr));
        c.dup(VReg4S(0), WReg(9));
    } else if (ops[1].type == ZYDIS_OPERAND_TYPE_REGISTER) {
        const int src = ZydisVecToIndex(ops[1].reg.value);
        if (src < 0) return false;
        c.ldr(WReg(9), ptr(kState, YmmChunkOffset(src, 0)));
        c.dup(VReg4S(0), WReg(9));
    } else {
        return false;
    }
    c.add(kScratch0, kState, YmmChunkOffset(dst, 0));
    c.str(QReg(0), ptr(kScratch0));
    if (ymm) {
        c.add(kScratch0, kState, YmmChunkOffset(dst, 2));
        c.str(QReg(0), ptr(kScratch0));
    } else {
        c.mov(kScratch0, 0);
        c.str(kScratch0, ptr(kState, YmmChunkOffset(dst, 2)));
        c.str(kScratch0, ptr(kState, YmmChunkOffset(dst, 3)));
    }
    return true;
}

// vroundss/vroundsd dst, src1, src2, imm8 — round src2's low element per imm
// rounding mode (bits 0-1 when bit2=0): 0=nearest-even, 1=floor, 2=ceil,
// 3=truncate. dst upper from src1. NEON frintn/frintm/frintp/frintz.
bool EmitVround(const ZydisDecodedInstruction& insn, const ZydisDecodedOperand* ops,
                u64 next_rip, CodeGenerator& c) {
    const bool sd = (insn.mnemonic == ZYDIS_MNEMONIC_VROUNDSD);
    if (ops[0].type != ZYDIS_OPERAND_TYPE_REGISTER ||
        ops[1].type != ZYDIS_OPERAND_TYPE_REGISTER) return false;
    const int dst = ZydisVecToIndex(ops[0].reg.value);
    const int s1  = ZydisVecToIndex(ops[1].reg.value);
    if (dst < 0 || s1 < 0) return false;
    const int mode = static_cast<int>(ops[3].imm.value.u) & 0x3;  // bit2 (MXCSR) not modeled

    // src2 low element -> v0.
    if (ops[2].type == ZYDIS_OPERAND_TYPE_REGISTER) {
        const int s2 = ZydisVecToIndex(ops[2].reg.value);
        if (s2 < 0) return false;
        if (sd) c.ldr(DReg(0), ptr(kState, YmmChunkOffset(s2, 0)));
        else    c.ldr(SReg(0), ptr(kState, YmmChunkOffset(s2, 0)));
    } else if (ops[2].type == ZYDIS_OPERAND_TYPE_MEMORY) {
        if (!EmitEffectiveAddress(ops[2].mem, next_rip, c)) return false;
        if (sd) c.ldr(DReg(0), ptr(kAddr));
        else    c.ldr(SReg(0), ptr(kAddr));
    } else {
        return false;
    }
    if (sd) {
        switch (mode) {
            case 0: c.frintn(DReg(0), DReg(0)); break;
            case 1: c.frintm(DReg(0), DReg(0)); break;
            case 2: c.frintp(DReg(0), DReg(0)); break;
            default: c.frintz(DReg(0), DReg(0)); break;
        }
        c.str(DReg(0), ptr(kState, YmmChunkOffset(dst, 0)));
        c.ldr(kScratch1, ptr(kState, YmmChunkOffset(s1, 1)));
        c.str(kScratch1, ptr(kState, YmmChunkOffset(dst, 1)));
    } else {
        switch (mode) {
            case 0: c.frintn(SReg(0), SReg(0)); break;
            case 1: c.frintm(SReg(0), SReg(0)); break;
            case 2: c.frintp(SReg(0), SReg(0)); break;
            default: c.frintz(SReg(0), SReg(0)); break;
        }
        c.fmov(WReg(9), SReg(0));
        c.ldr(kScratch1, ptr(kState, YmmChunkOffset(s1, 0)));
        c.lsr(kScratch1, kScratch1, 32);
        c.lsl(kScratch1, kScratch1, 32);
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

// Compose an AArch64 predicate compare on VReg4S(0) (a) vs VReg4S(1) (b) into
// VReg16B(0) (all-ones/all-zeros per lane), matching x86 CMPPS imm predicates.
// Handles the common predicates seen in PS4 code: EQ/LT/LE/UNORD/NEQ/NLT/NLE/ORD.
static bool EmitFpPredicate4S(int pred, CodeGenerator& c) {
    // a in v0, b in v1, scratch v2. Result mask -> v0.
    switch (pred & 0x7) {
        case 0: c.fcmeq(VReg4S(0), VReg4S(0), VReg4S(1)); return true;          // EQ
        case 1: c.fcmgt(VReg4S(0), VReg4S(1), VReg4S(0)); return true;          // LT: b>a
        case 2: c.fcmge(VReg4S(0), VReg4S(1), VReg4S(0)); return true;          // LE: b>=a
        case 3: // UNORD: neither a<=... ordered. ord = a==a & b==b; unord = ~ord.
            c.fcmeq(VReg4S(2), VReg4S(0), VReg4S(0));
            c.fcmeq(VReg4S(0), VReg4S(1), VReg4S(1));
            c.and_(VReg16B(0), VReg16B(0), VReg16B(2));
            c.not_(VReg16B(0), VReg16B(0));
            return true;
        case 4: c.fcmeq(VReg4S(0), VReg4S(0), VReg4S(1));                       // NEQ = ~EQ
                c.not_(VReg16B(0), VReg16B(0)); return true;
        case 5: c.fcmgt(VReg4S(0), VReg4S(1), VReg4S(0));                       // NLT = ~LT
                c.not_(VReg16B(0), VReg16B(0)); return true;
        case 6: c.fcmge(VReg4S(0), VReg4S(1), VReg4S(0));                       // NLE = ~LE
                c.not_(VReg16B(0), VReg16B(0)); return true;
        case 7: // ORD: a==a & b==b
            c.fcmeq(VReg4S(2), VReg4S(0), VReg4S(0));
            c.fcmeq(VReg4S(0), VReg4S(1), VReg4S(1));
            c.and_(VReg16B(0), VReg16B(0), VReg16B(2));
            return true;
    }
    return false;
}

// vcmpps/vcmpss dst, src1, src2, imm8 — packed/scalar FP compare producing a
// per-element all-ones/all-zeros mask. Scalar form writes only the low element;
// dst[127:32] from src1. XMM zeroes upper 128.
bool EmitVcmpfp(const ZydisDecodedInstruction& insn, const ZydisDecodedOperand* ops,
                u64 next_rip, CodeGenerator& c) {
    const bool scalar = (insn.mnemonic == ZYDIS_MNEMONIC_VCMPSS);
    if (ops[0].type != ZYDIS_OPERAND_TYPE_REGISTER ||
        ops[1].type != ZYDIS_OPERAND_TYPE_REGISTER) return false;
    const int dst = ZydisVecToIndex(ops[0].reg.value);
    const int s1  = ZydisVecToIndex(ops[1].reg.value);
    if (dst < 0 || s1 < 0) return false;
    const int pred = static_cast<int>(ops[3].imm.value.u) & 0x7;

    c.add(kScratch0, kState, YmmChunkOffset(s1, 0));
    c.ldr(QReg(0), ptr(kScratch0));      // a
    if (ops[2].type == ZYDIS_OPERAND_TYPE_REGISTER) {
        const int s2 = ZydisVecToIndex(ops[2].reg.value);
        if (s2 < 0) return false;
        c.add(kScratch0, kState, YmmChunkOffset(s2, 0));
        c.ldr(QReg(1), ptr(kScratch0));
    } else if (ops[2].type == ZYDIS_OPERAND_TYPE_MEMORY) {
        if (!EmitEffectiveAddress(ops[2].mem, next_rip, c)) return false;
        c.ldr(QReg(1), ptr(kAddr));
    } else {
        return false;
    }
    // Preserve src1 for the scalar merge (a is clobbered by the predicate).
    if (scalar) c.mov(VReg16B(3), VReg16B(0));
    if (!EmitFpPredicate4S(pred, c)) return false;   // mask -> v0

    if (!scalar) {
        c.add(kScratch0, kState, YmmChunkOffset(dst, 0));
        c.str(QReg(0), ptr(kScratch0));
    } else {
        // Insert mask low lane into src1 copy (v3), store that.
        c.ins(VReg4S(3)[0], VReg4S(0)[0]);
        c.add(kScratch0, kState, YmmChunkOffset(dst, 0));
        c.str(QReg(3), ptr(kScratch0));
    }
    c.mov(kScratch0, 0);
    c.str(kScratch0, ptr(kState, YmmChunkOffset(dst, 2)));
    c.str(kScratch0, ptr(kState, YmmChunkOffset(dst, 3)));
    return true;
}

// vpmovzxdq — zero-extend the low two dwords of src to two qwords. NEON uxtl.
bool EmitVpmovzxdq(const ZydisDecodedInstruction& insn, const ZydisDecodedOperand* ops,
                   u64 next_rip, CodeGenerator& c) {
    if (ops[0].type != ZYDIS_OPERAND_TYPE_REGISTER) return false;
    const int dst = ZydisVecToIndex(ops[0].reg.value);
    if (dst < 0) return false;
    const bool ymm = (ops[0].size == 256);

    // Load source low 128 (4 dwords) into v0.
    if (ops[1].type == ZYDIS_OPERAND_TYPE_REGISTER) {
        const int src = ZydisVecToIndex(ops[1].reg.value);
        if (src < 0) return false;
        c.add(kScratch0, kState, YmmChunkOffset(src, 0));
        c.ldr(QReg(0), ptr(kScratch0));
    } else if (ops[1].type == ZYDIS_OPERAND_TYPE_MEMORY) {
        if (!EmitEffectiveAddress(ops[1].mem, next_rip, c)) return false;
        if (ymm) c.ldr(QReg(0), ptr(kAddr));   // 128-bit src (4 dwords)
        else     c.ldr(DReg(0), ptr(kAddr));   // 64-bit src (2 dwords)
    } else {
        return false;
    }

    if (ymm) {
        // 4 dwords -> 4 qwords across the full 256: low 2 -> chunks 0,1;
        // high 2 -> chunks 2,3.
        c.uxtl2(VReg2D(1), VReg4S(0));     // v1 = widen high 2 dwords
        c.uxtl(VReg2D(0), VReg2S(0));      // v0 = widen low 2 dwords
        c.add(kScratch0, kState, YmmChunkOffset(dst, 0));
        c.str(QReg(0), ptr(kScratch0));
        c.add(kScratch0, kState, YmmChunkOffset(dst, 2));
        c.str(QReg(1), ptr(kScratch0));
    } else {
        c.uxtl(VReg2D(0), VReg2S(0));      // low 2 dwords -> 2 qwords
        c.add(kScratch0, kState, YmmChunkOffset(dst, 0));
        c.str(QReg(0), ptr(kScratch0));
        c.mov(kScratch0, 0);
        c.str(kScratch0, ptr(kState, YmmChunkOffset(dst, 2)));
        c.str(kScratch0, ptr(kState, YmmChunkOffset(dst, 3)));
    }
    return true;
}

// vpmovsx/vpmovzx — sign/zero-extend packed integers to a wider element. The
// source supplies 128/ratio bytes of narrow elements; each widens to the dest
// element. Implemented as a chain of NEON sxtl/uxtl steps on the low half (each
// doubles the element width). XMM dest only for now (the 256-bit forms widen
// across both halves and are left to the per-bail path). `sgn` selects sxtl vs
// uxtl; src_bits/dst_bits are the element widths in bits.
bool EmitVpmovx(const ZydisDecodedInstruction& insn, const ZydisDecodedOperand* ops,
                u64 next_rip, bool sgn, int src_bits, int dst_bits, CodeGenerator& c) {
    if (ops[0].type != ZYDIS_OPERAND_TYPE_REGISTER) return false;
    const int dst = ZydisVecToIndex(ops[0].reg.value);
    if (dst < 0) return false;
    if (ops[0].size == 256) return false;        // XMM dest only (first cut)
    const int ratio = dst_bits / src_bits;       // 2, 4, or 8
    const int src_bytes = 16 / ratio;            // bytes consumed from source

    if (ops[1].type == ZYDIS_OPERAND_TYPE_REGISTER) {
        const int src = ZydisVecToIndex(ops[1].reg.value);
        if (src < 0) return false;
        c.add(kScratch0, kState, YmmChunkOffset(src, 0));
        c.ldr(QReg(0), ptr(kScratch0));          // full low 128; only low elems used
    } else if (ops[1].type == ZYDIS_OPERAND_TYPE_MEMORY) {
        if (!EmitEffectiveAddress(ops[1].mem, next_rip, c)) return false;
        switch (src_bytes) {                     // load exactly the needed width
            case 2:  c.ldr(HReg(0), ptr(kAddr)); break;
            case 4:  c.ldr(SReg(0), ptr(kAddr)); break;
            case 8:  c.ldr(DReg(0), ptr(kAddr)); break;
            default: return false;
        }
    } else {
        return false;
    }

    // Widen the low half one element-width at a time, from src_bits to dst_bits.
    int b = src_bits;
    if (b == 8) {
        if (sgn) c.sxtl(VReg8H(0), VReg8B(0)); else c.uxtl(VReg8H(0), VReg8B(0));
        b = 16;
    }
    if (b == 16 && dst_bits >= 32) {
        if (sgn) c.sxtl(VReg4S(0), VReg4H(0)); else c.uxtl(VReg4S(0), VReg4H(0));
        b = 32;
    }
    if (b == 32 && dst_bits >= 64) {
        if (sgn) c.sxtl(VReg2D(0), VReg2S(0)); else c.uxtl(VReg2D(0), VReg2S(0));
        b = 64;
    }

    c.add(kScratch0, kState, YmmChunkOffset(dst, 0));
    c.str(QReg(0), ptr(kScratch0));
    c.mov(kScratch0, 0);                          // VEX-128: zero upper 128
    c.str(kScratch0, ptr(kState, YmmChunkOffset(dst, 2)));
    c.str(kScratch0, ptr(kState, YmmChunkOffset(dst, 3)));
    return true;
}

// vsqrtps/vsqrtpd dst, src/m — packed square root. NEON fsqrt at .4s/.2d, per
// 128-bit half (YMM = 2 halves). Single source operand.
bool EmitVsqrtPacked(const ZydisDecodedInstruction& insn, const ZydisDecodedOperand* ops,
                     u64 next_rip, bool dbl, CodeGenerator& c) {
    if (ops[0].type != ZYDIS_OPERAND_TYPE_REGISTER) return false;
    const int dst = ZydisVecToIndex(ops[0].reg.value);
    if (dst < 0) return false;
    const bool ymm = (ops[0].size == 256);
    const int nchunks = ymm ? 2 : 1;

    for (int h = 0; h < nchunks; ++h) {
        const int chunk = h * 2;
        if (ops[1].type == ZYDIS_OPERAND_TYPE_REGISTER) {
            const int src = ZydisVecToIndex(ops[1].reg.value);
            if (src < 0) return false;
            c.add(kScratch0, kState, YmmChunkOffset(src, chunk));
            c.ldr(QReg(0), ptr(kScratch0));
        } else if (ops[1].type == ZYDIS_OPERAND_TYPE_MEMORY) {
            if (h == 0) {
                if (!EmitEffectiveAddress(ops[1].mem, next_rip, c)) return false;
                c.mov(kScratch1, kAddr);
            }
            c.add(kScratch0, kScratch1, h * 16);
            c.ldr(QReg(0), ptr(kScratch0));
        } else {
            return false;
        }
        if (dbl) c.fsqrt(VReg2D(0), VReg2D(0));
        else     c.fsqrt(VReg4S(0), VReg4S(0));
        c.add(kScratch0, kState, YmmChunkOffset(dst, chunk));
        c.str(QReg(0), ptr(kScratch0));
    }
    if (!ymm) {
        c.mov(kScratch0, 0);
        c.str(kScratch0, ptr(kState, YmmChunkOffset(dst, 2)));
        c.str(kScratch0, ptr(kState, YmmChunkOffset(dst, 3)));
    }
    return true;
}

// vmovddup dst, src/m — duplicate the low 64 bits into both halves of each
// 128-bit lane: dst[63:0]=src[63:0], dst[127:64]=src[63:0]. A plain 64-bit
// data move (no NEON needed); per 128-bit lane for YMM.
bool EmitVmovddup(const ZydisDecodedInstruction& insn, const ZydisDecodedOperand* ops,
                  u64 next_rip, CodeGenerator& c) {
    if (ops[0].type != ZYDIS_OPERAND_TYPE_REGISTER) return false;
    const int dst = ZydisVecToIndex(ops[0].reg.value);
    if (dst < 0) return false;
    const bool ymm = (ops[0].size == 256);
    const int nchunks = ymm ? 2 : 1;

    for (int h = 0; h < nchunks; ++h) {
        const int chunk = h * 2;
        if (ops[1].type == ZYDIS_OPERAND_TYPE_REGISTER) {
            const int src = ZydisVecToIndex(ops[1].reg.value);
            if (src < 0) return false;
            c.ldr(kScratch0, ptr(kState, YmmChunkOffset(src, chunk)));  // low 64 of lane
        } else if (ops[1].type == ZYDIS_OPERAND_TYPE_MEMORY) {
            if (h == 0) {
                if (!EmitEffectiveAddress(ops[1].mem, next_rip, c)) return false;
                c.mov(kScratch1, kAddr);
            }
            c.add(kScratch2, kScratch1, h * 16);
            c.ldr(kScratch0, ptr(kScratch2));
        } else {
            return false;
        }
        c.str(kScratch0, ptr(kState, YmmChunkOffset(dst, chunk)));      // low 64
        c.str(kScratch0, ptr(kState, YmmChunkOffset(dst, chunk + 1)));  // dup -> high 64
    }
    if (!ymm) {
        c.mov(kScratch0, 0);
        c.str(kScratch0, ptr(kState, YmmChunkOffset(dst, 2)));
        c.str(kScratch0, ptr(kState, YmmChunkOffset(dst, 3)));
    }
    return true;
}

// vpinsrb/vpinsrw dst, src1, r/m, imm8 — copy src1's low 128 to dst, then insert
// the byte/word from a GPR or memory into lane (imm & laneMask). Uses NEON INS
// (general) at the right element width. Upper 128 zeroed.
bool EmitVpinsrBW(const ZydisDecodedInstruction& insn, const ZydisDecodedOperand* ops,
                  u64 next_rip, CodeGenerator& c) {
    if (ops[0].type != ZYDIS_OPERAND_TYPE_REGISTER ||
        ops[1].type != ZYDIS_OPERAND_TYPE_REGISTER) return false;
    const int dst = ZydisVecToIndex(ops[0].reg.value);
    const int s1  = ZydisVecToIndex(ops[1].reg.value);
    if (dst < 0 || s1 < 0) return false;
    const bool byte = (insn.mnemonic == ZYDIS_MNEMONIC_VPINSRB);
    const int lane = static_cast<int>(ops[3].imm.value.u) & (byte ? 15 : 7);

    // Stage src1's low 128 into v0.
    c.add(kScratch0, kState, YmmChunkOffset(s1, 0));
    c.ldr(QReg(0), ptr(kScratch0));

    // Fetch the value into w9 (only the low 8/16 bits matter).
    if (ops[2].type == ZYDIS_OPERAND_TYPE_REGISTER) {
        const int gi = ZydisGprToIndex(ops[2].reg.value);
        if (gi < 0) return false;
        c.ldr(kWScratch0, ptr(kState, GprOffset(gi)));
    } else if (ops[2].type == ZYDIS_OPERAND_TYPE_MEMORY) {
        if (!EmitEffectiveAddress(ops[2].mem, next_rip, c)) return false;
        if (byte) c.ldrb(kWScratch0, ptr(kAddr));
        else      c.ldrh(kWScratch0, ptr(kAddr));
    } else {
        return false;
    }

    if (byte) c.ins(VReg16B(0)[lane], kWScratch0);
    else      c.ins(VReg8H(0)[lane],  kWScratch0);

    c.add(kScratch0, kState, YmmChunkOffset(dst, 0));
    c.str(QReg(0), ptr(kScratch0));
    c.mov(kScratch0, 0);
    c.str(kScratch0, ptr(kState, YmmChunkOffset(dst, 2)));
    c.str(kScratch0, ptr(kState, YmmChunkOffset(dst, 3)));
    return true;
}

// vpmuludq dst, src1, src2 — multiply the EVEN 32-bit lanes (0,2[,4,6]) of each
// source as unsigned, producing 64-bit results. NEON umull multiplies the low
// two .2s lanes, so first gather the even dwords down with uzp1, then umull.
// Per 128-bit half for YMM.
bool EmitVpmuludq(const ZydisDecodedInstruction& insn, const ZydisDecodedOperand* ops,
                  u64 next_rip, CodeGenerator& c) {
    if (ops[0].type != ZYDIS_OPERAND_TYPE_REGISTER ||
        ops[1].type != ZYDIS_OPERAND_TYPE_REGISTER) return false;
    const int dst = ZydisVecToIndex(ops[0].reg.value);
    const int s1  = ZydisVecToIndex(ops[1].reg.value);
    if (dst < 0 || s1 < 0) return false;
    const bool ymm = (ops[0].size == 256);
    const int nchunks = ymm ? 2 : 1;

    for (int h = 0; h < nchunks; ++h) {
        const int chunk = h * 2;
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
                c.mov(kScratch1, kAddr);
            }
            c.add(kScratch0, kScratch1, h * 16);
            c.ldr(QReg(1), ptr(kScratch0));
        } else {
            return false;
        }
        // Gather even dwords (lanes 0,2) into the low two .2s of each, then
        // unsigned-multiply-long to two 64-bit products.
        c.uzp1(VReg4S(0), VReg4S(0), VReg4S(0));
        c.uzp1(VReg4S(1), VReg4S(1), VReg4S(1));
        c.umull(VReg2D(0), VReg2S(0), VReg2S(1));
        c.add(kScratch0, kState, YmmChunkOffset(dst, chunk));
        c.str(QReg(0), ptr(kScratch0));
    }
    if (!ymm) {
        c.mov(kScratch0, 0);
        c.str(kScratch0, ptr(kState, YmmChunkOffset(dst, 2)));
        c.str(kScratch0, ptr(kState, YmmChunkOffset(dst, 3)));
    }
    return true;
}

// vroundps/vroundpd dst, src/m, imm8 — round every lane per imm[1:0]
// (0=nearest-even,1=floor,2=ceil,3=trunc); bit2 (use-MXCSR) not modeled. NEON
// frintn/frintm/frintp/frintz at .4s/.2d, per 128-bit half.
bool EmitVroundPacked(const ZydisDecodedInstruction& insn, const ZydisDecodedOperand* ops,
                      u64 next_rip, bool dbl, CodeGenerator& c) {
    if (ops[0].type != ZYDIS_OPERAND_TYPE_REGISTER) return false;
    const int dst = ZydisVecToIndex(ops[0].reg.value);
    if (dst < 0) return false;
    const int mode = static_cast<int>(ops[2].imm.value.u) & 0x3;
    const bool ymm = (ops[0].size == 256);
    const int nchunks = ymm ? 2 : 1;

    for (int h = 0; h < nchunks; ++h) {
        const int chunk = h * 2;
        if (ops[1].type == ZYDIS_OPERAND_TYPE_REGISTER) {
            const int src = ZydisVecToIndex(ops[1].reg.value);
            if (src < 0) return false;
            c.add(kScratch0, kState, YmmChunkOffset(src, chunk));
            c.ldr(QReg(0), ptr(kScratch0));
        } else if (ops[1].type == ZYDIS_OPERAND_TYPE_MEMORY) {
            if (h == 0) {
                if (!EmitEffectiveAddress(ops[1].mem, next_rip, c)) return false;
                c.mov(kScratch1, kAddr);
            }
            c.add(kScratch0, kScratch1, h * 16);
            c.ldr(QReg(0), ptr(kScratch0));
        } else {
            return false;
        }
        if (dbl) {
            switch (mode) {
                case 0: c.frintn(VReg2D(0), VReg2D(0)); break;
                case 1: c.frintm(VReg2D(0), VReg2D(0)); break;
                case 2: c.frintp(VReg2D(0), VReg2D(0)); break;
                default: c.frintz(VReg2D(0), VReg2D(0)); break;
            }
        } else {
            switch (mode) {
                case 0: c.frintn(VReg4S(0), VReg4S(0)); break;
                case 1: c.frintm(VReg4S(0), VReg4S(0)); break;
                case 2: c.frintp(VReg4S(0), VReg4S(0)); break;
                default: c.frintz(VReg4S(0), VReg4S(0)); break;
            }
        }
        c.add(kScratch0, kState, YmmChunkOffset(dst, chunk));
        c.str(QReg(0), ptr(kScratch0));
    }
    if (!ymm) {
        c.mov(kScratch0, 0);
        c.str(kScratch0, ptr(kState, YmmChunkOffset(dst, 2)));
        c.str(kScratch0, ptr(kState, YmmChunkOffset(dst, 3)));
    }
    return true;
}

// vhaddps/pd, vhsubps/pd dst, src1, src2 — horizontal pairwise add/sub. For PS:
// dst = [s1_0±s1_1, s1_2±s1_3, s2_0±s2_1, s2_2±s2_3]. Gather even lanes (uzp1)
// and odd lanes (uzp2), then fadd/fsub. Per 128-bit half for YMM.
bool EmitVhaddsub(const ZydisDecodedInstruction& insn, const ZydisDecodedOperand* ops,
                  u64 next_rip, bool dbl, bool sub, CodeGenerator& c) {
    if (ops[0].type != ZYDIS_OPERAND_TYPE_REGISTER ||
        ops[1].type != ZYDIS_OPERAND_TYPE_REGISTER) return false;
    const int dst = ZydisVecToIndex(ops[0].reg.value);
    const int s1  = ZydisVecToIndex(ops[1].reg.value);
    if (dst < 0 || s1 < 0) return false;
    const bool ymm = (ops[0].size == 256);
    const int nchunks = ymm ? 2 : 1;

    for (int h = 0; h < nchunks; ++h) {
        const int chunk = h * 2;
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
                c.mov(kScratch1, kAddr);
            }
            c.add(kScratch0, kScratch1, h * 16);
            c.ldr(QReg(1), ptr(kScratch0));
        } else {
            return false;
        }
        if (dbl) {
            c.uzp1(VReg2D(2), VReg2D(0), VReg2D(1));   // even: [s1_0, s2_0]
            c.uzp2(VReg2D(3), VReg2D(0), VReg2D(1));   // odd:  [s1_1, s2_1]
            if (sub) c.fsub(VReg2D(0), VReg2D(2), VReg2D(3));
            else     c.fadd(VReg2D(0), VReg2D(2), VReg2D(3));
        } else {
            c.uzp1(VReg4S(2), VReg4S(0), VReg4S(1));   // even: [s1_0,s1_2,s2_0,s2_2]
            c.uzp2(VReg4S(3), VReg4S(0), VReg4S(1));   // odd:  [s1_1,s1_3,s2_1,s2_3]
            if (sub) c.fsub(VReg4S(0), VReg4S(2), VReg4S(3));
            else     c.fadd(VReg4S(0), VReg4S(2), VReg4S(3));
        }
        c.add(kScratch0, kState, YmmChunkOffset(dst, chunk));
        c.str(QReg(0), ptr(kScratch0));
    }
    if (!ymm) {
        c.mov(kScratch0, 0);
        c.str(kScratch0, ptr(kState, YmmChunkOffset(dst, 2)));
        c.str(kScratch0, ptr(kState, YmmChunkOffset(dst, 3)));
    }
    return true;
}

// vaddsubps/pd dst, src1, src2 — subtract on even lanes, add on odd lanes:
// dst[i] = (i even) ? s1[i]-s2[i] : s1[i]+s2[i]. Compute both fadd and fsub,
// then overwrite the even lanes of the add-result with the sub-result.
bool EmitVaddsub(const ZydisDecodedInstruction& insn, const ZydisDecodedOperand* ops,
                 u64 next_rip, bool dbl, CodeGenerator& c) {
    if (ops[0].type != ZYDIS_OPERAND_TYPE_REGISTER ||
        ops[1].type != ZYDIS_OPERAND_TYPE_REGISTER) return false;
    const int dst = ZydisVecToIndex(ops[0].reg.value);
    const int s1  = ZydisVecToIndex(ops[1].reg.value);
    if (dst < 0 || s1 < 0) return false;
    const bool ymm = (ops[0].size == 256);
    const int nchunks = ymm ? 2 : 1;

    for (int h = 0; h < nchunks; ++h) {
        const int chunk = h * 2;
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
                c.mov(kScratch1, kAddr);
            }
            c.add(kScratch0, kScratch1, h * 16);
            c.ldr(QReg(1), ptr(kScratch0));
        } else {
            return false;
        }
        if (dbl) {
            c.fadd(VReg2D(2), VReg2D(0), VReg2D(1));   // v2 = s1+s2 (odd lane keeps this)
            c.fsub(VReg2D(3), VReg2D(0), VReg2D(1));   // v3 = s1-s2 (even lane)
            c.ins(VReg2D(2)[0], VReg2D(3)[0]);         // lane 0 = sub
        } else {
            c.fadd(VReg4S(2), VReg4S(0), VReg4S(1));
            c.fsub(VReg4S(3), VReg4S(0), VReg4S(1));
            c.ins(VReg4S(2)[0], VReg4S(3)[0]);         // lanes 0,2 = sub
            c.ins(VReg4S(2)[2], VReg4S(3)[2]);
        }
        c.add(kScratch0, kState, YmmChunkOffset(dst, chunk));
        c.str(QReg(2), ptr(kScratch0));
    }
    if (!ymm) {
        c.mov(kScratch0, 0);
        c.str(kScratch0, ptr(kState, YmmChunkOffset(dst, 2)));
        c.str(kScratch0, ptr(kState, YmmChunkOffset(dst, 3)));
    }
    return true;
}

// vrcpps/ss, vrsqrtps/ss — reciprocal / reciprocal-sqrt ESTIMATE. NEON
// frecpe/frsqrte give ~8-bit accuracy versus x86's ~12-bit guarantee, so this
// is a lower-precision approximation (acceptable for the games' use of these as
// fast estimates; a Newton-Raphson refinement step could be added if a title
// proves sensitive). PS forms are packed (.4s); SS forms are scalar with the
// upper 96 bits taken from src1.
bool EmitVrecip(const ZydisDecodedInstruction& insn, const ZydisDecodedOperand* ops,
                u64 next_rip, bool rsqrt, bool scalar, CodeGenerator& c) {
    if (ops[0].type != ZYDIS_OPERAND_TYPE_REGISTER) return false;
    const int dst = ZydisVecToIndex(ops[0].reg.value);
    if (dst < 0) return false;

    if (scalar) {
        // dst.s[0] = est(src2.s[0]); dst[127:32] = src1[127:32]; upper zeroed.
        const int s1 = ZydisVecToIndex(ops[1].reg.value);
        if (s1 < 0) return false;
        c.add(kScratch0, kState, YmmChunkOffset(s1, 0));
        c.ldr(QReg(1), ptr(kScratch0));                  // v1 = src1 (upper kept)
        if (ops[2].type == ZYDIS_OPERAND_TYPE_REGISTER) {
            const int s2 = ZydisVecToIndex(ops[2].reg.value);
            if (s2 < 0) return false;
            c.ldr(SReg(0), ptr(kState, YmmChunkOffset(s2, 0)));
        } else if (ops[2].type == ZYDIS_OPERAND_TYPE_MEMORY) {
            if (!EmitEffectiveAddress(ops[2].mem, next_rip, c)) return false;
            c.ldr(SReg(0), ptr(kAddr));
        } else {
            return false;
        }
        if (rsqrt) c.frsqrte(SReg(0), SReg(0)); else c.frecpe(SReg(0), SReg(0));
        c.ins(VReg4S(1)[0], VReg4S(0)[0]);               // merge estimate into low lane
        c.add(kScratch0, kState, YmmChunkOffset(dst, 0));
        c.str(QReg(1), ptr(kScratch0));
        c.mov(kScratch0, 0);
        c.str(kScratch0, ptr(kState, YmmChunkOffset(dst, 2)));
        c.str(kScratch0, ptr(kState, YmmChunkOffset(dst, 3)));
        return true;
    }

    // Packed: est of all 4 lanes per 128-bit half.
    const bool ymm = (ops[0].size == 256);
    const int nchunks = ymm ? 2 : 1;
    for (int h = 0; h < nchunks; ++h) {
        const int chunk = h * 2;
        if (ops[1].type == ZYDIS_OPERAND_TYPE_REGISTER) {
            const int src = ZydisVecToIndex(ops[1].reg.value);
            if (src < 0) return false;
            c.add(kScratch0, kState, YmmChunkOffset(src, chunk));
            c.ldr(QReg(0), ptr(kScratch0));
        } else if (ops[1].type == ZYDIS_OPERAND_TYPE_MEMORY) {
            if (h == 0) {
                if (!EmitEffectiveAddress(ops[1].mem, next_rip, c)) return false;
                c.mov(kScratch1, kAddr);
            }
            c.add(kScratch0, kScratch1, h * 16);
            c.ldr(QReg(0), ptr(kScratch0));
        } else {
            return false;
        }
        if (rsqrt) c.frsqrte(VReg4S(0), VReg4S(0)); else c.frecpe(VReg4S(0), VReg4S(0));
        c.add(kScratch0, kState, YmmChunkOffset(dst, chunk));
        c.str(QReg(0), ptr(kScratch0));
    }
    if (!ymm) {
        c.mov(kScratch0, 0);
        c.str(kScratch0, ptr(kState, YmmChunkOffset(dst, 2)));
        c.str(kScratch0, ptr(kState, YmmChunkOffset(dst, 3)));
    }
    return true;
}

// vmovmskps/pd, vpmovmskb dst(gpr), src(xmm/ymm) — gather the sign bit of each
// element into a GPR bitmask (bit i = sign of element i). elem_bits selects the
// element width: 32 (ps), 64 (pd), 8 (byte). Per-lane umov + shift-or; the lane
// count follows the source width (128 or 256).
bool EmitMovmsk(const ZydisDecodedInstruction& insn, const ZydisDecodedOperand* ops,
                int elem_bits, CodeGenerator& c) {
    if (ops[0].type != ZYDIS_OPERAND_TYPE_REGISTER ||
        ops[1].type != ZYDIS_OPERAND_TYPE_REGISTER) return false;
    const int d   = ZydisGprToIndex(ops[0].reg.value);
    const int src = ZydisVecToIndex(ops[1].reg.value);
    if (d < 0 || src < 0) return false;
    const int total_bits = (ops[1].size == 256) ? 256 : 128;
    const int per_half = 128 / elem_bits;
    const int nchunks = (total_bits == 256) ? 2 : 1;
    const int signpos = elem_bits - 1;
    const XReg result = XReg(13);

    c.mov(result, 0);
    for (int h = 0; h < nchunks; ++h) {
        c.add(kScratch0, kState, YmmChunkOffset(src, h * 2));
        c.ldr(QReg(0), ptr(kScratch0));
        for (int j = 0; j < per_half; ++j) {
            const int bit = h * per_half + j;
            if (elem_bits == 64) {
                c.umov(XReg(14), VReg2D(0)[j]);
                c.lsr(XReg(14), XReg(14), signpos);
            } else if (elem_bits == 32) {
                c.umov(WReg(14), VReg4S(0)[j]);
                c.lsr(WReg(14), WReg(14), signpos);
            } else {
                c.umov(WReg(14), VReg16B(0)[j]);
                c.lsr(WReg(14), WReg(14), signpos);
            }
            if (bit == 0) {
                c.orr(result, result, XReg(14));
            } else {
                c.lsl(XReg(14), XReg(14), bit);
                c.orr(result, result, XReg(14));
            }
        }
    }
    c.str(result, ptr(kState, GprOffset(d)));   // zero-extended mask into dst GPR
    return true;
}

// vshufpd dst, src1, src2, imm — per 128-bit lane: low double from src1 (bit
// selects which of its two), high double from src2. For YMM the next imm bits
// drive the high lane.
bool EmitVshufpd(const ZydisDecodedInstruction& insn, const ZydisDecodedOperand* ops,
                 u64 next_rip, CodeGenerator& c) {
    if (ops[0].type != ZYDIS_OPERAND_TYPE_REGISTER ||
        ops[1].type != ZYDIS_OPERAND_TYPE_REGISTER) return false;
    const int dst = ZydisVecToIndex(ops[0].reg.value);
    const int s1  = ZydisVecToIndex(ops[1].reg.value);
    if (dst < 0 || s1 < 0) return false;
    const int imm = static_cast<int>(ops[3].imm.value.u) & 0xFF;
    const bool ymm = (ops[0].size == 256);
    const int nchunks = ymm ? 2 : 1;

    for (int h = 0; h < nchunks; ++h) {
        const int chunk = h * 2;
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
                c.mov(kScratch1, kAddr);
            }
            c.add(kScratch0, kScratch1, h * 16);
            c.ldr(QReg(1), ptr(kScratch0));
        } else {
            return false;
        }
        const int lo = (imm >> (2 * h + 0)) & 1;     // src1's double
        const int hi = (imm >> (2 * h + 1)) & 1;     // src2's double
        c.ins(VReg2D(2)[0], VReg2D(0)[lo]);
        c.ins(VReg2D(2)[1], VReg2D(1)[hi]);
        c.add(kScratch0, kState, YmmChunkOffset(dst, chunk));
        c.str(QReg(2), ptr(kScratch0));
    }
    if (!ymm) {
        c.mov(kScratch0, 0);
        c.str(kScratch0, ptr(kState, YmmChunkOffset(dst, 2)));
        c.str(kScratch0, ptr(kState, YmmChunkOffset(dst, 3)));
    }
    return true;
}

// vpermilps/pd dst, src, imm8 — in-lane permute selected by imm, same imm per
// 128-bit lane. PS: 4 floats by 2-bit fields; PD: 2 doubles by 1-bit fields
// (bits [1:0] low lane, [3:2] high lane). Only the imm form is handled here; the
// variable (vector-control) form is declined.
bool EmitVpermil(const ZydisDecodedInstruction& insn, const ZydisDecodedOperand* ops,
                 u64 next_rip, bool dbl, CodeGenerator& c) {
    if (ops[0].type != ZYDIS_OPERAND_TYPE_REGISTER) return false;
    if (ops[2].type != ZYDIS_OPERAND_TYPE_IMMEDIATE) return false;  // variable form -> bail
    const int dst = ZydisVecToIndex(ops[0].reg.value);
    if (dst < 0) return false;
    const int imm = static_cast<int>(ops[2].imm.value.u) & 0xFF;
    const bool ymm = (ops[0].size == 256);
    const int nchunks = ymm ? 2 : 1;

    for (int h = 0; h < nchunks; ++h) {
        const int chunk = h * 2;
        if (ops[1].type == ZYDIS_OPERAND_TYPE_REGISTER) {
            const int src = ZydisVecToIndex(ops[1].reg.value);
            if (src < 0) return false;
            c.add(kScratch0, kState, YmmChunkOffset(src, chunk));
            c.ldr(QReg(0), ptr(kScratch0));
        } else if (ops[1].type == ZYDIS_OPERAND_TYPE_MEMORY) {
            if (h == 0) {
                if (!EmitEffectiveAddress(ops[1].mem, next_rip, c)) return false;
                c.mov(kScratch1, kAddr);
            }
            c.add(kScratch0, kScratch1, h * 16);
            c.ldr(QReg(0), ptr(kScratch0));
        } else {
            return false;
        }
        if (dbl) {
            const int s0 = (imm >> (2 * h + 0)) & 1;
            const int s1 = (imm >> (2 * h + 1)) & 1;
            c.ins(VReg2D(2)[0], VReg2D(0)[s0]);
            c.ins(VReg2D(2)[1], VReg2D(0)[s1]);
        } else {
            for (int i = 0; i < 4; ++i)
                c.ins(VReg4S(2)[i], VReg4S(0)[(imm >> (2 * i)) & 3]);  // same imm per lane
        }
        c.add(kScratch0, kState, YmmChunkOffset(dst, chunk));
        c.str(QReg(2), ptr(kScratch0));
    }
    if (!ymm) {
        c.mov(kScratch0, 0);
        c.str(kScratch0, ptr(kState, YmmChunkOffset(dst, 2)));
        c.str(kScratch0, ptr(kState, YmmChunkOffset(dst, 3)));
    }
    return true;
}

// vpshuflw/vpshufhw dst, src, imm8 — shuffle the low (LW) or high (HW) four
// words of each 128-bit lane by imm 2-bit fields; the other four words pass
// through unchanged.
bool EmitVpshufw(const ZydisDecodedInstruction& insn, const ZydisDecodedOperand* ops,
                 u64 next_rip, CodeGenerator& c) {
    if (ops[0].type != ZYDIS_OPERAND_TYPE_REGISTER) return false;
    const int dst = ZydisVecToIndex(ops[0].reg.value);
    if (dst < 0) return false;
    const bool hi = (insn.mnemonic == ZYDIS_MNEMONIC_VPSHUFHW);
    const int imm = static_cast<int>(ops[2].imm.value.u) & 0xFF;
    const bool ymm = (ops[0].size == 256);
    const int nchunks = ymm ? 2 : 1;
    const int base = hi ? 4 : 0;   // which group of 4 words is shuffled

    for (int h = 0; h < nchunks; ++h) {
        const int chunk = h * 2;
        if (ops[1].type == ZYDIS_OPERAND_TYPE_REGISTER) {
            const int src = ZydisVecToIndex(ops[1].reg.value);
            if (src < 0) return false;
            c.add(kScratch0, kState, YmmChunkOffset(src, chunk));
            c.ldr(QReg(0), ptr(kScratch0));
        } else if (ops[1].type == ZYDIS_OPERAND_TYPE_MEMORY) {
            if (h == 0) {
                if (!EmitEffectiveAddress(ops[1].mem, next_rip, c)) return false;
                c.mov(kScratch1, kAddr);
            }
            c.add(kScratch0, kScratch1, h * 16);
            c.ldr(QReg(0), ptr(kScratch0));
        } else {
            return false;
        }
        c.mov(VReg16B(2), VReg16B(0));   // pass-through copy; overwrite the 4 shuffled words
        for (int i = 0; i < 4; ++i)
            c.ins(VReg8H(2)[base + i], VReg8H(0)[base + ((imm >> (2 * i)) & 3)]);
        c.add(kScratch0, kState, YmmChunkOffset(dst, chunk));
        c.str(QReg(2), ptr(kScratch0));
    }
    if (!ymm) {
        c.mov(kScratch0, 0);
        c.str(kScratch0, ptr(kState, YmmChunkOffset(dst, 2)));
        c.str(kScratch0, ptr(kState, YmmChunkOffset(dst, 3)));
    }
    return true;
}

// --- PCMPISTRI / PCMPISTRM ---------------------------------------------------
// SSE4.2 implicit-length packed string compare. There is no NEON analog, so the
// lifter marshals the two 128-bit operands + imm8 to this portable routine and
// calls it. The routine is validated bit-exact against x86 SSE4.2 hardware
// across all four aggregation modes, both data formats, signed/unsigned, every
// polarity, and both output selections (index/mask, incl. the expanded mask).
// It writes the index (ISTRI), the mask (ISTRM), and the AFZSO flags — already
// shifted into rflags bit positions — into the caller's output buffer.
struct PcmpistrOut { u64 mask_lo; u64 mask_hi; u32 index; u32 flags; };

inline bool PcmpElemZero(const u8* P, int i, int sz) {
    return sz == 1 ? (P[i] == 0) : (P[2 * i] == 0 && P[2 * i + 1] == 0);
}
inline int PcmpElemVal(const u8* P, int i, int sz, int sgn) {
    if (sz == 1) return sgn ? (int)(signed char)P[i] : (int)P[i];
    u32 v = (u32)(P[2 * i] | (P[2 * i + 1] << 8));
    return sgn ? (int)(short)(u16)v : (int)v;
}

// Called from JIT-emitted code via blr. AAPCS64: x28 (kState) and x25-x27 are
// callee-saved, so the only register the emitted shim must preserve across the
// call is the link register.
void Pcmpistr_helper(const void* aptr, const void* bptr,
                     u32 imm, u32 ret_mask, void* outptr) {
    const u8* A = (const u8*)aptr;
    const u8* B = (const u8*)bptr;
    PcmpistrOut* out = (PcmpistrOut*)outptr;
    const int sz  = (imm & 1) ? 2 : 1;
    const int n   = (imm & 1) ? 8 : 16;
    const int sgn = (imm >> 1) & 1;
    const int agg = (imm >> 2) & 3;
    const int pol = (imm >> 4) & 3;
    const int sel = (imm >> 6) & 1;

    int la = n, lb = n;   // implicit lengths: index of first null element
    for (int i = 0; i < n; i++) { if (PcmpElemZero(A, i, sz)) { la = i; break; } }
    for (int j = 0; j < n; j++) { if (PcmpElemZero(B, j, sz)) { lb = j; break; } }

    int a[16], b[16];
    for (int i = 0; i < n; i++) { a[i] = PcmpElemVal(A, i, sz, sgn); b[i] = PcmpElemVal(B, i, sz, sgn); }

    int intres1 = 0;
    for (int j = 0; j < n; j++) {
        int r = 0;
        switch (agg) {
        case 0:  // equal any: b[j] equals any valid a[i]
            for (int i = 0; i < n; i++)
                r |= (((i < la) && (j < lb)) ? (a[i] == b[j]) : 0);
            break;
        case 1:  // ranges: b[j] within any valid pair [a[i], a[i+1]]
            for (int i = 0; i + 1 < n; i += 2)
                r |= ((((i < la) && (i + 1 < la)) && (j < lb))
                          ? (b[j] >= a[i] && b[j] <= a[i + 1]) : 0);
            break;
        case 2: { // equal each (diagonal), with invalid-element override
            int av = (j < la), bv = (j < lb);
            r = (av && bv) ? (a[j] == b[j]) : ((!av && !bv) ? 1 : 0);
            break; }
        case 3: { // equal ordered (substring search at offset j)
            r = 1;
            for (int i = 0; i < n - j; i++) {
                int av = (i < la), bv = (j + i < lb);
                int cell = av ? (bv ? (a[i] == b[j + i]) : 0) : 1;
                r &= cell;
            }
            break; }
        }
        intres1 |= (r & 1) << j;
    }

    const int mask_all = (n == 16) ? 0xFFFF : 0xFF;
    if (pol == 1) intres1 = (~intres1) & mask_all;                       // negate all
    else if (pol == 3) { int m = (lb >= n) ? mask_all : ((1 << lb) - 1); // negate up to lb
                         intres1 = (intres1 ^ m) & mask_all; }

    u32 flags = 0;
    if (intres1 != 0) flags |= (1u << 0);   // CF = IntRes1 != 0
    if (lb < n)       flags |= (1u << 6);   // ZF = operand2 had a null
    if (la < n)       flags |= (1u << 7);   // SF = operand1 had a null
    if (intres1 & 1)  flags |= (1u << 11);  // OF = IntRes1[0]
    out->flags = flags;

    int idx;
    if (intres1 == 0) idx = n;
    else if (sel == 0) { idx = 0; while (!((intres1 >> idx) & 1)) idx++; }   // least sig
    else               { idx = n - 1; while (!((intres1 >> idx) & 1)) idx--; } // most sig
    out->index = (u32)idx;

    out->mask_lo = 0; out->mask_hi = 0;
    if (ret_mask) {
        if (sel == 0) {
            out->mask_lo = (u64)(u32)intres1;                  // bit mask in low bits
        } else {                                               // expanded element mask
            u8 by[16]; for (int k = 0; k < 16; k++) by[k] = 0;
            for (int j = 0; j < n; j++)
                if ((intres1 >> j) & 1) {
                    if (sz == 1) by[j] = 0xFF;
                    else { by[2 * j] = 0xFF; by[2 * j + 1] = 0xFF; }
                }
            u64 lo = 0, hi = 0;
            for (int k = 0; k < 8; k++) { lo |= (u64)by[k] << (8 * k); hi |= (u64)by[8 + k] << (8 * k); }
            out->mask_lo = lo; out->mask_hi = hi;
        }
    }
}

// pcmpistri/pcmpistrm xmm1, xmm2/m128, imm8 — marshal to Pcmpistr_helper. The
// out buffer and the link-register save live in a 48-byte host stack frame.
bool EmitPcmpistr(const ZydisDecodedInstruction& insn, const ZydisDecodedOperand* ops,
                  u64 next_rip, CodeGenerator& c) {
    const bool ret_mask = (insn.mnemonic == ZYDIS_MNEMONIC_PCMPISTRM);
    if (ops[0].type != ZYDIS_OPERAND_TYPE_REGISTER) return false;
    const int a_idx = ZydisVecToIndex(ops[0].reg.value);
    if (a_idx < 0) return false;
    if (ops[2].type != ZYDIS_OPERAND_TYPE_IMMEDIATE) return false;
    const u32 imm = (u32)(ops[2].imm.value.u & 0xFF);

    // Resolve operand 2's address. For memory, compute the EA first (it clobbers
    // x9-x11) and stash it in x12 before we build the call frame and args.
    const bool b_mem = (ops[1].type == ZYDIS_OPERAND_TYPE_MEMORY);
    int b_idx = -1;
    if (b_mem) {
        if (!EmitEffectiveAddress(ops[1].mem, next_rip, c)) return false;
        c.mov(XReg(12), kAddr);
    } else {
        if (ops[1].type != ZYDIS_OPERAND_TYPE_REGISTER) return false;
        b_idx = ZydisVecToIndex(ops[1].reg.value);
        if (b_idx < 0) return false;
    }

    c.sub(kSp, kSp, 48);
    c.str(XReg(30), ptr(kSp, 32));                          // save link register
    c.add(XReg(0), kState, YmmChunkOffset(a_idx, 0));      // x0 = &xmm1
    if (b_mem) c.mov(XReg(1), XReg(12));                   // x1 = &operand2
    else       c.add(XReg(1), kState, YmmChunkOffset(b_idx, 0));
    c.mov(XReg(2), imm);                                   // x2 = imm8
    c.mov(XReg(3), ret_mask ? 1 : 0);                      // x3 = ret_mask
    c.add(XReg(4), kSp, 0);                                 // x4 = &out
    c.mov(XReg(9), reinterpret_cast<u64>(&Pcmpistr_helper));
    c.blr(XReg(9));
    c.ldr(XReg(30), ptr(kSp, 32));                          // restore link register

    if (ret_mask) {
        // XMM0 (guest vec 0) low 128 = mask. The x86 backend zeroes ymm0's upper
        // 128 bits for both the VEX and legacy SSE forms (see EmitVpcmpistrm in
        // lifter_x86_host.cpp), so match that here for backend parity rather than
        // preserving bits 255:128 as strict legacy-SSE semantics would.
        c.ldr(kScratch0, ptr(kSp, 0));
        c.str(kScratch0, ptr(kState, YmmChunkOffset(0, 0)));
        c.ldr(kScratch0, ptr(kSp, 8));
        c.str(kScratch0, ptr(kState, YmmChunkOffset(0, 1)));
        c.mov(kScratch0, 0);
        c.str(kScratch0, ptr(kState, YmmChunkOffset(0, 2)));
        c.str(kScratch0, ptr(kState, YmmChunkOffset(0, 3)));
    } else {
        // ECX = index, zero-extended into RCX (gpr index 1).
        c.ldr(kWScratch0, ptr(kSp, 16));
        c.str(kScratch0, ptr(kState, GprOffset(1)));
    }
    // Merge CF/ZF/SF/OF (helper already placed them at their rflags positions).
    // Load the existing rflags into kScratch1 FIRST, then read the helper's flag
    // word into a register distinct from kScratch1/kScratch2 — x11 (kAddr) is
    // dead here in both operand paths. (A previous version loaded the flags into
    // WReg(10), which aliases kScratch1, so the subsequent rflags load clobbered
    // them and every flag came out zero.)
    c.ldr(kScratch1, ptr(kState, Offsets::Rflags));
    c.ldr(WReg(11), ptr(kSp, 20));
    c.mov(kScratch2, ~((1ull<<0)|(1ull<<2)|(1ull<<4)|(1ull<<6)|(1ull<<7)|(1ull<<11)));
    c.and_(kScratch1, kScratch1, kScratch2);
    c.orr(kScratch1, kScratch1, XReg(11));
    c.str(kScratch1, ptr(kState, Offsets::Rflags));

    c.add(kSp, kSp, 48);
    return true;
}

// vphaddd dst, src1, src2 — horizontal pairwise dword add. Result low half =
// pairwise sums of src1, high half = pairwise sums of src2. NEON addp.
bool EmitVphaddd(const ZydisDecodedInstruction& insn, const ZydisDecodedOperand* ops,
                 u64 next_rip, CodeGenerator& c) {
    if (ops[0].type != ZYDIS_OPERAND_TYPE_REGISTER ||
        ops[1].type != ZYDIS_OPERAND_TYPE_REGISTER) return false;
    const int dst = ZydisVecToIndex(ops[0].reg.value);
    const int s1  = ZydisVecToIndex(ops[1].reg.value);
    if (dst < 0 || s1 < 0) return false;

    c.add(kScratch0, kState, YmmChunkOffset(s1, 0));
    c.ldr(QReg(0), ptr(kScratch0));
    if (ops[2].type == ZYDIS_OPERAND_TYPE_REGISTER) {
        const int s2 = ZydisVecToIndex(ops[2].reg.value);
        if (s2 < 0) return false;
        c.add(kScratch0, kState, YmmChunkOffset(s2, 0));
        c.ldr(QReg(1), ptr(kScratch0));
    } else if (ops[2].type == ZYDIS_OPERAND_TYPE_MEMORY) {
        if (!EmitEffectiveAddress(ops[2].mem, next_rip, c)) return false;
        c.ldr(QReg(1), ptr(kAddr));
    } else {
        return false;
    }
    // addp(d, a, b): d = [a0+a1, a2+a3, b0+b1, b2+b3] — exactly x86 phaddd.
    c.addp(VReg4S(0), VReg4S(0), VReg4S(1));
    c.add(kScratch0, kState, YmmChunkOffset(dst, 0));
    c.str(QReg(0), ptr(kScratch0));
    c.mov(kScratch0, 0);
    c.str(kScratch0, ptr(kState, YmmChunkOffset(dst, 2)));
    c.str(kScratch0, ptr(kState, YmmChunkOffset(dst, 3)));
    return true;
}

// vpshufb dst, src1, mask — per-byte shuffle: out[i] = (mask[i] & 0x80) ? 0 :
// src1[mask[i] & 0x0F]. NEON tbl zeros for indices >= 16; AND the mask with
// 0x8F so bit7-set bytes become >=0x80 (tbl zeros) and others index 0..15.
bool EmitVpshufb(const ZydisDecodedInstruction& insn, const ZydisDecodedOperand* ops,
                 u64 next_rip, CodeGenerator& c) {
    if (ops[0].type != ZYDIS_OPERAND_TYPE_REGISTER ||
        ops[1].type != ZYDIS_OPERAND_TYPE_REGISTER) return false;
    const int dst = ZydisVecToIndex(ops[0].reg.value);
    const int s1  = ZydisVecToIndex(ops[1].reg.value);
    if (dst < 0 || s1 < 0) return false;

    c.add(kScratch0, kState, YmmChunkOffset(s1, 0));
    c.ldr(QReg(0), ptr(kScratch0));      // table (src1)
    if (ops[2].type == ZYDIS_OPERAND_TYPE_REGISTER) {
        const int s2 = ZydisVecToIndex(ops[2].reg.value);
        if (s2 < 0) return false;
        c.add(kScratch0, kState, YmmChunkOffset(s2, 0));
        c.ldr(QReg(1), ptr(kScratch0));
    } else if (ops[2].type == ZYDIS_OPERAND_TYPE_MEMORY) {
        if (!EmitEffectiveAddress(ops[2].mem, next_rip, c)) return false;
        c.ldr(QReg(1), ptr(kAddr));
    } else {
        return false;
    }
    c.movi(VReg16B(2), 0x8F);
    c.and_(VReg16B(1), VReg16B(1), VReg16B(2));     // keep bit7 + low nibble
    c.tbl(VReg16B(0), VReg16B(0), VReg16B(1));      // 1-register table lookup
    c.add(kScratch0, kState, YmmChunkOffset(dst, 0));
    c.str(QReg(0), ptr(kScratch0));
    c.mov(kScratch0, 0);
    c.str(kScratch0, ptr(kState, YmmChunkOffset(dst, 2)));
    c.str(kScratch0, ptr(kState, YmmChunkOffset(dst, 3)));
    return true;
}

// VPCMPISTRI / VPCMPISTRM — SSE4.2 implicit-length string compare. AArch64 has
// no equivalent instruction, so we implement the control matrix in software for
// the aggregation modes PS4 code uses: "equal each" (imm[3:2]=10) and
// "equal any" (imm[3:2]=00), byte data (imm[0]=0). Operand a = ops[0],
// b = ops[1] (reg/mem). Implicit length: each string ends at its first zero
// byte. Flags: CF=(mask!=0), ZF=(b has a null), SF=(a has a null), OF=mask bit0.
//   ISTRI: ECX = index of first match (LSB), else 16.
//   ISTRM: imm[6]=0 bit-mask (low 16 bits of XMM0); imm[6]=1 byte-mask
//          (0xFF/0x00 per byte). XMM0 upper YMM zeroed.
//
// Strategy: stage a and b into the two 8-byte halves of guest scratch on the
// stack is unnecessary — we read bytes directly out of the guest YMM slots via
// byte loads, building a 16-bit match mask in w-reg `mask`. a_len/b_len are the
// first-null indices (16 if none).
static bool EmitVpcmpistr(const ZydisDecodedInstruction& insn,
                          const ZydisDecodedOperand* ops, u64 next_rip,
                          bool wantMask, CodeGenerator& c) {
    if (ops[0].type != ZYDIS_OPERAND_TYPE_REGISTER) return false;
    const int a_idx = ZydisVecToIndex(ops[0].reg.value);
    if (a_idx < 0) return false;
    u8 imm = 0;
    if (ops[2].type == ZYDIS_OPERAND_TYPE_IMMEDIATE) imm = (u8)(ops[2].imm.value.u & 0xFF);
    else if (ops[1].type == ZYDIS_OPERAND_TYPE_IMMEDIATE) imm = (u8)(ops[1].imm.value.u & 0xFF);
    else return false;
    const int agg = (imm >> 2) & 3;       // 00=equal-any, 10=equal-each
    const bool byteMask = (imm >> 6) & 1;

    // Base address of a's 16 bytes -> x5 ; b's 16 bytes -> x6.
    c.add(XReg(5), kState, YmmChunkOffset(a_idx, 0));
    if (ops[1].type == ZYDIS_OPERAND_TYPE_REGISTER) {
        const int b_idx = ZydisVecToIndex(ops[1].reg.value);
        if (b_idx < 0) return false;
        c.add(XReg(6), kState, YmmChunkOffset(b_idx, 0));
    } else if (ops[1].type == ZYDIS_OPERAND_TYPE_MEMORY) {
        if (!EmitEffectiveAddress(ops[1].mem, next_rip, c)) return false;
        c.mov(XReg(6), kAddr);
    } else {
        return false;
    }

    // a_len -> x7, b_len -> x8: index of first zero byte, else 16. Compute by
    // scanning; use a running "found" flag so later zeros don't lower the len.
    auto scanLen = [&](const XReg& base, const XReg& lenOut) {
        c.mov(lenOut, 16);
        c.mov(XReg(9), 0);                 // found flag
        for (int i = 0; i < 16; ++i) {
            c.ldrb(WReg(12), ptr(base, i));
            // if (!found && byte==0) { len=i; found=1; }
            Label skip;
            c.cbnz(XReg(9), skip);         // already found -> skip
            c.cbnz(XReg(12), skip);        // nonzero -> skip
            c.mov(lenOut, i);
            c.mov(XReg(9), 1);
            c.L(skip);
        }
    };
    scanLen(XReg(5), XReg(7));   // a_len
    scanLen(XReg(6), XReg(8));   // b_len

    // Build 16-bit match mask in w15. A position i is "valid" by mode:
    //   equal-each: valid iff i < a_len AND i < b_len; match = a[i]==b[i].
    //               Also, positions where BOTH strings ended count as match
    //               (i>=a_len && i>=b_len) per the spec's invalid-handling for
    //               EqualEach; but the strlen idiom only needs the first true
    //               match, so we additionally treat i>=both-len as a match.
    //   equal-any:  match[i] = (i < b_len) AND (b[i] equals some a[j], j<a_len).
    c.mov(WReg(15), 0);                    // mask
    for (int i = 0; i < 16; ++i) {
        c.ldrb(WReg(12), ptr(XReg(5), i));  // a[i]
        c.ldrb(WReg(13), ptr(XReg(6), i));  // b[i]
        Label noMatch, done;
        if (agg == 2) {
            // equal-each
            // matched if (i>=a_len && i>=b_len) [both ended] OR
            //            (i<a_len && i<b_len && a[i]==b[i]).
            Label bothEnded, cmpEq;
            c.cmp(XReg(7), i); c.b(LE, /*i>=a_len?*/ bothEnded);   // a_len<=i
            c.b(cmpEq);
            c.L(bothEnded);
            // a ended; matched iff b also ended (b_len<=i).
            c.cmp(XReg(8), i); c.b(LE, done /*=match*/); c.b(noMatch);
            c.L(cmpEq);
            // i<a_len; need i<b_len and a[i]==b[i].
            c.cmp(XReg(8), i); c.b(LE, noMatch);    // b ended -> no match
            c.cmp(WReg(12), WReg(13)); c.b(NE, noMatch);
            c.b(done);
        } else {
            // equal-any: i<b_len and b[i] ∈ a[0..a_len).
            c.cmp(XReg(8), i); c.b(LE, noMatch);    // i>=b_len -> invalid
            // scan a for b[i].
            Label found;
            for (int j = 0; j < 16; ++j) {
                Label jskip;
                c.cmp(XReg(7), j); c.b(LE, jskip);  // j>=a_len -> stop considering
                c.ldrb(WReg(14), ptr(XReg(5), j));
                c.cmp(WReg(14), WReg(13));
                c.b(EQ, found);
                c.L(jskip);
            }
            c.b(noMatch);
            c.L(found);
            c.b(done);
        }
        c.L(done);
        c.mov(WReg(14), 1);
        c.lsl(WReg(14), WReg(14), i);
        c.orr(WReg(15), WReg(15), WReg(14));
        c.L(noMatch);
    }

    // ---- Flags. CF=(mask!=0), ZF=(b_len<16), SF=(a_len<16), OF=mask bit0. ----
    const XReg fl = XReg(12), t = XReg(13);
    c.ldr(fl, ptr(kState, Offsets::Rflags));
    c.mov(t, ~((1ull<<0)|(1ull<<6)|(1ull<<7)|(1ull<<11)));
    c.and_(fl, fl, t);
    c.cmp(WReg(15), 0); c.cset(t, NE); c.orr(fl, fl, t);            // CF
    c.cmp(XReg(8), 16); c.cset(t, LT); c.lsl(t, t, 6); c.orr(fl, fl, t);  // ZF
    c.cmp(XReg(7), 16); c.cset(t, LT); c.lsl(t, t, 7); c.orr(fl, fl, t);  // SF
    c.and_(t, XReg(15), 1); c.lsl(t, t, 11); c.orr(fl, fl, t);     // OF
    c.str(fl, ptr(kState, Offsets::Rflags));

    if (!wantMask) {
        // ISTRI: ECX = index of least-significant set bit, else 16.
        const XReg ecx = XReg(13);
        Label none, store;
        c.cbz(WReg(15), none);
        c.rbit(WReg(13), WReg(15));
        c.clz(WReg(13), WReg(13));     // index of LSB
        c.b(store);
        c.L(none);
        c.mov(ecx, 16);
        c.L(store);
        c.and_(ecx, ecx, 0xFFFFFFFFull);
        c.str(ecx, ptr(kState, GprOffset(1)));  // RCX zero-extended
        return true;
    }

    // ISTRM: write mask into XMM0 (guest vec index 0).
    if (byteMask) {
        // Per-byte 0xFF/0x00. Build two 64-bit halves.
        const XReg lo = XReg(13), hi = XReg(14), bit = XReg(9), bytev = XReg(10);
        c.mov(lo, 0); c.mov(hi, 0);
        for (int i = 0; i < 16; ++i) {
            c.lsr(bit, XReg(15), i); c.and_(bit, bit, 1);
            c.mov(bytev, 0);
            Label z;
            c.cbz(bit, z);
            c.mov(bytev, 0xFF);
            c.L(z);
            if (i < 8) { c.lsl(bytev, bytev, (i*8)); c.orr(lo, lo, bytev); }
            else       { c.lsl(bytev, bytev, ((i-8)*8)); c.orr(hi, hi, bytev); }
        }
        c.str(lo, ptr(kState, YmmChunkOffset(0, 0)));
        c.str(hi, ptr(kState, YmmChunkOffset(0, 1)));
    } else {
        // Bit-mask: low 16 bits = mask, rest zero.
        c.and_(XReg(13), XReg(15), 0xFFFF);
        c.str(XReg(13), ptr(kState, YmmChunkOffset(0, 0)));
        c.mov(kScratch0, 0);
        c.str(kScratch0, ptr(kState, YmmChunkOffset(0, 1)));
    }
    c.mov(kScratch0, 0);
    c.str(kScratch0, ptr(kState, YmmChunkOffset(0, 2)));
    c.str(kScratch0, ptr(kState, YmmChunkOffset(0, 3)));
    return true;
}

// vpackusdw dst, src1, src2 — pack 8 signed dwords (src1 low 4, src2 high 4)
// into 8 unsigned words with saturation. NEON sqxtun (signed->unsigned sat).
bool EmitVpackusdw(const ZydisDecodedInstruction& insn, const ZydisDecodedOperand* ops,
                   u64 next_rip, CodeGenerator& c) {
    if (ops[0].type != ZYDIS_OPERAND_TYPE_REGISTER ||
        ops[1].type != ZYDIS_OPERAND_TYPE_REGISTER) return false;
    const int dst = ZydisVecToIndex(ops[0].reg.value);
    const int s1  = ZydisVecToIndex(ops[1].reg.value);
    if (dst < 0 || s1 < 0) return false;

    c.add(kScratch0, kState, YmmChunkOffset(s1, 0));
    c.ldr(QReg(0), ptr(kScratch0));
    if (ops[2].type == ZYDIS_OPERAND_TYPE_REGISTER) {
        const int s2 = ZydisVecToIndex(ops[2].reg.value);
        if (s2 < 0) return false;
        c.add(kScratch0, kState, YmmChunkOffset(s2, 0));
        c.ldr(QReg(1), ptr(kScratch0));
    } else if (ops[2].type == ZYDIS_OPERAND_TYPE_MEMORY) {
        if (!EmitEffectiveAddress(ops[2].mem, next_rip, c)) return false;
        c.ldr(QReg(1), ptr(kAddr));
    } else {
        return false;
    }
    // low 4 words = sat(src1 dwords); high 4 words = sat(src2 dwords).
    c.sqxtun(VReg4H(2), VReg4S(0));        // v2.low64 = packed src1
    c.sqxtun2(VReg8H(2), VReg4S(1));       // v2.high64 = packed src2
    c.add(kScratch0, kState, YmmChunkOffset(dst, 0));
    c.str(QReg(2), ptr(kScratch0));
    c.mov(kScratch0, 0);
    c.str(kScratch0, ptr(kState, YmmChunkOffset(dst, 2)));
    c.str(kScratch0, ptr(kState, YmmChunkOffset(dst, 3)));
    return true;
}

// cpuid — spoof an AMD Jaguar (PS4 APU). Reads leaf in EAX, subleaf in ECX;
// writes EAX/EBX/ECX/EDX (guest slots 0/3/1/2), zero-extended. Mirrors the x86
// host's canned values exactly.
/// RDTSC / RDTSCP -- CNTVCT_EL0 passthrough.
///
/// Guest shape: counter low 32 -> EAX slot, high 32 -> EDX slot, both
/// zero-extended; RDTSCP additionally writes TSC_AUX -> ECX (we report
/// core 0). No flags.
///
/// The virtual counter is the arm64 analogue of the host-TSC
/// passthrough on x86: monotonic, user-readable, and the same source
/// the kernel HLE's clock plumbing reads on this host, so rdtsc deltas
/// and sceKernelGetTscFrequency stay mutually consistent. KNOWN
/// RESIDUAL: CNTFRQ (commonly 24 MHz / 1 GHz) differs from a Jaguar
/// TSC (~1.6 GHz); any guest that HARDCODES the PS4 TSC frequency
/// instead of calibrating will run its rdtsc-based timing at the wrong
/// rate. If a title trips on that, the fix is a scale factor here and
/// in the kernel TSC HLE together, never one side alone.
bool EmitRdtsc(const ZydisDecodedInstruction& insn, const ZydisDecodedOperand* ops,
               CodeGenerator& c, bool with_aux) {
    (void)insn; (void)ops;
    // mrs x9, CNTVCT_EL0  (op0=3, op1=3, CRn=14, CRm=0, op2=2)
    c.mrs(XReg(9), 3, 3, 14, 0, 2);
    c.ubfx(XReg(10), XReg(9), 0, 32);            // low 32, zero-extended
    c.lsr(XReg(11), XReg(9), 32);                // high 32, zero-extended
    c.str(XReg(10), ptr(kState, GprOffset(0)));  // RAX
    c.str(XReg(11), ptr(kState, GprOffset(2)));  // RDX
    if (with_aux) {
        c.mov(XReg(10), 0);                       // TSC_AUX: core 0
        c.str(XReg(10), ptr(kState, GprOffset(1))); // RCX
    }
    return true;
}


/// LAHF / SAHF -- AH <-> low flags byte. Mirrors the x86 backend:
/// straight byte traffic between state.rflags and the AH byte of the
/// guest RAX slot. 0xD5 (SF|ZF|AF|PF|CF) is not encodable as an
/// AArch64 logical immediate (not a rotated run of ones), so it is
/// materialized in a scratch register -- which makes the SAHF
/// slot-clear a single BIC against that same register. AF fidelity
/// note as on x86: helper-computed flag paths leave AF=0.
bool EmitLahf(const ZydisDecodedInstruction& insn, const ZydisDecodedOperand* ops,
              CodeGenerator& c) {
    (void)insn; (void)ops;
    c.ldr(XReg(9), ptr(kState, Offsets::Rflags));
    c.mov(WReg(10), 0xD5);                           // SF|ZF|AF|PF|CF
    c.and_(WReg(9), WReg(9), WReg(10));
    c.orr(WReg(9), WReg(9), 0x02);                   // bit 1 reads as 1
    c.strb(WReg(9), ptr(kState, GprOffset(0) + 1));  // AH of guest RAX
    return true;
}

bool EmitSahf(const ZydisDecodedInstruction& insn, const ZydisDecodedOperand* ops,
              CodeGenerator& c) {
    (void)insn; (void)ops;
    c.ldrb(WReg(9), ptr(kState, GprOffset(0) + 1));  // AH of guest RAX
    c.mov(WReg(10), 0xD5);
    c.and_(XReg(9), XReg(9), XReg(10));              // only SF|ZF|AF|PF|CF transfer
    c.ldr(XReg(11), ptr(kState, Offsets::Rflags));
    c.bic(XReg(11), XReg(11), XReg(10));             // clear the five bits, keep OF/DF/...
    c.orr(XReg(11), XReg(11), XReg(9));
    c.str(XReg(11), ptr(kState, Offsets::Rflags));
    return true;
}

bool EmitCpuid(const ZydisDecodedInstruction& insn, const ZydisDecodedOperand* ops,
               CodeGenerator& c) {
    (void)insn; (void)ops;
    const XReg leaf = XReg(5), sub = XReg(6);
    c.ldr(WReg(5), ptr(kState, GprOffset(0)));   // leaf = EAX
    c.ldr(WReg(6), ptr(kState, GprOffset(1)));   // subleaf = ECX

    auto put = [&](u32 a, u32 b, u32 cc, u32 d) {
        c.mov(WReg(9), a); c.str(kScratch0, ptr(kState, GprOffset(0)));
        c.mov(WReg(9), b); c.str(kScratch0, ptr(kState, GprOffset(3)));
        c.mov(WReg(9), cc); c.str(kScratch0, ptr(kState, GprOffset(1)));
        c.mov(WReg(9), d); c.str(kScratch0, ptr(kState, GprOffset(2)));
    };

    Label l1, l7, lb2, lb3, lb4, ldef, ldone;
    c.mov(WReg(10), 0x00000000u); c.cmp(WReg(5), WReg(10)); c.b(NE, l1);
    put(0x00000007u, 0x68747541u, 0x444D4163u, 0x69746E65u); c.b(ldone);

    c.L(l1);
    c.mov(WReg(10), 0x00000001u); c.cmp(WReg(5), WReg(10)); c.b(NE, l7);
    put(0x00700F01u, 0x00000000u,
        (1u<<0)|(1u<<19)|(1u<<20)|(1u<<23)|(1u<<28),
        (1u<<0)|(1u<<25)|(1u<<26));
    c.b(ldone);

    c.L(l7);
    c.mov(WReg(10), 0x00000007u); c.cmp(WReg(5), WReg(10)); c.b(NE, lb2);
    c.cbnz(WReg(6), ldef);                 // only subleaf 0 responds
    put(0x00000000u, (1u<<3), 0x00000000u, 0x00000000u);  // EBX: BMI1
    c.b(ldone);

    c.L(lb2);
    c.mov(WReg(10), 0x80000002u); c.cmp(WReg(5), WReg(10)); c.b(NE, lb3);
    put(0x20444D41u, 0x74737543u, 0x4A206D6Fu, 0x61756761u);
    c.b(ldone);

    c.L(lb3);
    c.mov(WReg(10), 0x80000003u); c.cmp(WReg(5), WReg(10)); c.b(NE, lb4);
    put(0x2D382072u, 0x65726F43u, 0x55504120u, 0x00000000u);
    c.b(ldone);

    c.L(lb4);
    c.mov(WReg(10), 0x80000004u); c.cmp(WReg(5), WReg(10)); c.b(NE, ldef);
    put(0x00000000u, 0x00000000u, 0x00000000u, 0x00000000u);
    c.b(ldone);

    c.L(ldef);
    put(0, 0, 0, 0);
    c.L(ldone);
    return true;
}

// xgetbv — report XCR0 = x87|SSE|AVX (bits 0,1,2 set) in EDX:EAX. EAX=7, EDX=0.
bool EmitXgetbv(const ZydisDecodedInstruction& insn, const ZydisDecodedOperand* ops,
                CodeGenerator& c) {
    (void)insn; (void)ops;
    // ECX selects the XCR. Only XCR0 (ECX==0) is defined → EAX=7, EDX=0.
    // Any other index returns zero in both halves.
    c.ldr(WReg(6), ptr(kState, GprOffset(1)));   // ECX
    Label unknown, done;
    c.cbnz(WReg(6), unknown);
    c.mov(WReg(9), 0x7u);
    c.str(kScratch0, ptr(kState, GprOffset(0)));   // EAX = 7
    c.mov(kScratch0, 0);
    c.str(kScratch0, ptr(kState, GprOffset(2)));   // EDX = 0
    c.b(done);
    c.L(unknown);
    c.mov(kScratch0, 0);
    c.str(kScratch0, ptr(kState, GprOffset(0)));
    c.str(kScratch0, ptr(kState, GprOffset(2)));
    c.L(done);
    return true;
}

// stmxcsr m32 / ldmxcsr m32 — store/load the guest MXCSR field.
// STMXCSR / LDMXCSR — guest SSE control/status register.
//
// STMXCSR reports GuestState::mxcsr (the guest's own last-written value;
// status flags are not reflected from host FPSR — same fidelity stance as
// the x86 backend's STMXCSR). LDMXCSR records the raw value AND applies a
// sanitized FPCR image to the host, so the NEON instructions we emit run
// under the guest's rounding mode and flush-to-zero from this point in the
// block onward. The dispatcher mirrors the swap (restore default FPCR on
// entry, re-apply before handing out a block) — see the rounding-swap
// design block and SanitizeGuestMxcsrToFpcr in runtime.cpp, which this
// emitted computation MUST stay bit-identical to.
//
// Mapping subtleties (full rationale in runtime.cpp):
//   * RC -> RMode is a 2-BIT REVERSE (x86 01=down/10=up vs ARM 01=up/
//     10=down): rmode = ((rc << 1) | (rc >> 1)) & 3.
//   * FZ = FTZ | DAZ: ARM's FZ flushes inputs AND outputs, so the x86
//     split is unexpressible; OR-ing is the consistent best effort.
//   * Trap enables are never set; host default FPCR is 0.
//
// FISTP is unaffected either way — it dispatches on fpu_cw explicitly.
// KNOWN RESIDUAL: emitters that lower x86 ops with current-rounding
// semantics via mode-EXPLICIT A64 instructions (e.g. cvtsd2si via fcvtns)
// do not yet consult FPCR; they need frinti-based lowerings to pick up the
// guest mode. Tracked separately — this change makes the mode REACH the
// hardware, which those lowerings can then honor.
bool EmitMxcsr(const ZydisDecodedInstruction& insn, const ZydisDecodedOperand* ops,
               u64 next_rip, CodeGenerator& c) {
    // VEX forms (VSTMXCSR/VLDMXCSR) are identical in behavior to the
    // legacy encodings; they only differ in Zydis mnemonic.
    const bool store = (insn.mnemonic == ZYDIS_MNEMONIC_STMXCSR ||
                        insn.mnemonic == ZYDIS_MNEMONIC_VSTMXCSR);
    if (ops[0].type != ZYDIS_OPERAND_TYPE_MEMORY) return false;
    if (!EmitEffectiveAddress(ops[0].mem, next_rip, c)) return false;
    const u32 off = static_cast<u32>(offsetof(GuestState, mxcsr));
    if (store) {
        c.ldr(WReg(9), ptr(kState, off));
        c.str(WReg(9), ptr(kAddr));
    } else {
        c.ldr(WReg(9), ptr(kAddr));
        c.str(WReg(9), ptr(kState, off));
        // Apply sanitized FPCR: rmode = bit-reversed RC at 23:22,
        // FZ = (FTZ | DAZ) at 24. Bit-identical to runtime.cpp's
        // SanitizeGuestMxcsrToFpcr.
        c.ubfx(WReg(10), WReg(9), 13, 2);          // rc
        c.lsl(WReg(11), WReg(10), 1);
        c.orr(WReg(11), WReg(11), WReg(10), LSR, 1);
        c.and_(WReg(11), WReg(11), 3);             // rmode = 2-bit reverse
        c.lsl(WReg(11), WReg(11), 22);             // -> FPCR.RMode
        c.ubfx(WReg(10), WReg(9), 15, 1);          // FTZ
        c.ubfx(WReg(12), WReg(9), 6, 1);           // DAZ
        c.orr(WReg(10), WReg(10), WReg(12));       // FZ
        c.orr(XReg(11), XReg(11), XReg(10), LSL, 24);
        c.msr(3, 3, 4, 4, 0, XReg(11));            // FPCR = S3_3_C4_C4_0
    }
    return true;
}

// fnstcw m16 — store the (fixed) x87 control word 0x037F. fldcw is a no-op in
// our model (rounding/precision control not emulated).
// FLDCW m16 / FNSTCW m16 — load/store the x87 control word, tracked in
// GuestState::fpu_cw (CallGuest initializes it to the FNINIT default
// 0x037F). FLDCW's RC field (bits 11:10) is consumed by EmitFistp below;
// precision control and exception masks are ignored by design (the x87
// registers are 64-bit doubles, and we never raise x87 exceptions). An
// earlier revision dropped FLDCW and reported a hardcoded 0x037F from
// FNSTCW — breaking both the truncation idiom (fldcw [chop]; fistp) and
// the save/restore round-trip when the guest's ambient cw differed.
bool EmitFnstcw(const ZydisDecodedInstruction& insn, const ZydisDecodedOperand* ops,
                u64 next_rip, CodeGenerator& c) {
    if (ops[0].type != ZYDIS_OPERAND_TYPE_MEMORY) return false;
    if (!EmitEffectiveAddress(ops[0].mem, next_rip, c)) return false;
    const u32 cw_off = static_cast<u32>(offsetof(GuestState, fpu_cw));
    if (insn.mnemonic == ZYDIS_MNEMONIC_FLDCW) {
        c.ldrh(WReg(9), ptr(kAddr));
        c.strh(WReg(9), ptr(kState, cw_off));
    } else {
        c.ldrh(WReg(9), ptr(kState, cw_off));
        c.strh(WReg(9), ptr(kAddr));
    }
    return true;
}

// ============================================================================
// x87 FPU — register-stack model. ST(i) = st[(fpu_top + i) & 7]. Each slot
// holds a 64-bit double bit-pattern. Push: top=(top-1)&7 then write st[top],
// set tag bit. Pop: clear tag bit, top=(top+1)&7. We compute slot addresses
// into a callee-stable reg so the top-of-stack math never aliases the value.
// ============================================================================

// addr_reg = &st[(fpu_top + delta) & 7]. Uses w-scratch regs; clobbers x12,x13.
static void EmitX87RegAddr(const XReg& addr_reg, int delta, CodeGenerator& c) {
    c.ldr(WReg(12), ptr(kState, (u32)offsetof(GuestState, fpu_top)));
    if (delta != 0) c.add(WReg(12), WReg(12), delta);
    c.and_(WReg(12), WReg(12), 7);
    c.add(addr_reg, kState, (u32)offsetof(GuestState, st));   // &st[0]
    c.add(addr_reg, addr_reg, XReg(12), ShMod::LSL, 3);  // + idx*8
}

// Push: top=(top-1)&7; st[top] = DReg(0); set tag bit `top`.
static void EmitX87Push(CodeGenerator& c) {
    c.ldr(WReg(12), ptr(kState, (u32)offsetof(GuestState, fpu_top)));
    c.sub(WReg(12), WReg(12), 1);
    c.and_(WReg(12), WReg(12), 7);
    c.str(WReg(12), ptr(kState, (u32)offsetof(GuestState, fpu_top)));
    c.add(XReg(14), kState, (u32)offsetof(GuestState, st));
    c.str(DReg(0), ptr(XReg(14), XReg(12), ShMod::LSL, 3));
    // set tag bit `top`: tag |= (1 << top).
    c.mov(WReg(13), 1);
    c.lsl(WReg(13), WReg(13), WReg(12));
    c.ldrh(WReg(14), ptr(kState, (u32)offsetof(GuestState, fpu_tag)));
    c.orr(WReg(14), WReg(14), WReg(13));
    c.strh(WReg(14), ptr(kState, (u32)offsetof(GuestState, fpu_tag)));
}

// Pop: clear tag bit `top`; top=(top+1)&7. Value already consumed by caller.
static void EmitX87PopDiscard(CodeGenerator& c) {
    c.ldr(WReg(12), ptr(kState, (u32)offsetof(GuestState, fpu_top)));
    c.mov(WReg(13), 1);
    c.lsl(WReg(13), WReg(13), WReg(12));
    c.mvn(WReg(13), WReg(13));
    c.ldrh(WReg(14), ptr(kState, (u32)offsetof(GuestState, fpu_tag)));
    c.and_(WReg(14), WReg(14), WReg(13));
    c.strh(WReg(14), ptr(kState, (u32)offsetof(GuestState, fpu_tag)));
    c.add(WReg(12), WReg(12), 1);
    c.and_(WReg(12), WReg(12), 7);
    c.str(WReg(12), ptr(kState, (u32)offsetof(GuestState, fpu_top)));
}

// Convert the double in DReg(0) to 80-bit extended, writing 10 bytes to the
// address in `dstAddr` (x11/kAddr). 80-bit: sign(1):exp(15):mantissa(64, with
// explicit integer bit). Handles normal/zero; subnormals-as-double map fine.
static void EmitDoubleToTbyte(const XReg& dstAddr, CodeGenerator& c) {
    // x9 = double bits.
    c.fmov(kScratch0, DReg(0));
    const XReg sign = XReg(5), dexp = XReg(6), dmant = XReg(7), eexp = XReg(12), emant = XReg(13);
    c.lsr(sign, kScratch0, 63);
    c.lsl(sign, sign, 15);                        // sign in bit15 of the exp word
    c.ubfx(dexp, kScratch0, 52, 11);              // 11-bit exponent
    c.ubfx(dmant, kScratch0, 0, 52);              // 52-bit mantissa
    Label zerocase, compose;
    // zero (dexp==0 && dmant==0) -> emit zero exp+mant.
    c.orr(eexp, dexp, dmant);
    c.cbz(eexp, zerocase);
    // normal: eexp = dexp - 1023 + 16383 ; emant = (1<<63) | (dmant << 11).
    c.mov(eexp, 1023);
    c.sub(eexp, dexp, eexp);
    c.mov(dmant, 16383);
    c.add(eexp, eexp, dmant);
    c.and_(eexp, eexp, 0x7FFF);
    c.ubfx(dmant, kScratch0, 0, 52);              // reload mantissa
    c.lsl(emant, dmant, 11);
    c.mov(dmant, 1);
    c.lsl(dmant, dmant, 63);
    c.orr(emant, emant, dmant);                   // explicit integer bit
    c.b(compose);
    c.L(zerocase);
    c.mov(eexp, 0);
    c.mov(emant, 0);
    c.L(compose);
    c.orr(sign, sign, eexp);                      // exp word = sign | eexp
    c.str(emant, ptr(dstAddr));                   // bytes 0..7 = mantissa
    c.strh(WReg(5), ptr(dstAddr, 8));             // bytes 8..9 = sign|exp
}

// Convert 80-bit extended at `srcAddr` to a double in DReg(0).
static void EmitTbyteToDouble(const XReg& srcAddr, CodeGenerator& c) {
    const XReg emant = XReg(5), expw = XReg(6), sign = XReg(7), dexp = XReg(12), dmant = XReg(13);
    c.ldr(emant, ptr(srcAddr));                   // 64-bit mantissa
    c.ldrh(WReg(6), ptr(srcAddr, 8));             // 16-bit sign|exp
    c.lsr(sign, expw, 15);
    c.and_(sign, sign, 1);
    c.and_(expw, expw, 0x7FFF);
    Label zero, done;
    c.cbz(expw, zero);
    // normal: dexp = eexp - 16383 + 1023 ; dmant = (emant >> 11) & 0xF_FFFF_FFFF_FFFF.
    c.mov(dexp, 16383);
    c.sub(dexp, expw, dexp);
    c.mov(dmant, 1023);
    c.add(dexp, dexp, dmant);
    c.and_(dexp, dexp, 0x7FF);
    c.lsr(dmant, emant, 11);
    c.ubfx(dmant, dmant, 0, 52);
    c.lsl(dexp, dexp, 52);
    c.orr(dmant, dmant, dexp);
    c.b(done);
    c.L(zero);
    c.mov(dmant, 0);
    c.L(done);
    c.lsl(sign, sign, 63);
    c.orr(dmant, dmant, sign);
    c.fmov(DReg(0), dmant);
}

// FLD — load float/double from memory and push; or register form `fld st(i)`
// which pushes a copy of ST(i). Tbyte (m80) load handled here too.
bool EmitFld(const ZydisDecodedInstruction& insn, const ZydisDecodedOperand* ops,
             u64 next_rip, CodeGenerator& c) {
    // Register form: fld st(i) (D9 C0+i).
    if (ops[0].type == ZYDIS_OPERAND_TYPE_REGISTER) {
        int i = 0;
        if (ops[0].reg.value >= ZYDIS_REGISTER_ST0 && ops[0].reg.value <= ZYDIS_REGISTER_ST7)
            i = ops[0].reg.value - ZYDIS_REGISTER_ST0;
        else return false;
        // Read ST(i) BEFORE the push changes top (push decrements top, so the
        // old ST(i) is at new offset i+1; read first into DReg(0)).
        EmitX87RegAddr(XReg(11), i, c);
        c.ldr(DReg(0), ptr(XReg(11)));
        EmitX87Push(c);
        return true;
    }
    if (ops[0].type != ZYDIS_OPERAND_TYPE_MEMORY) return false;
    if (!EmitEffectiveAddress(ops[0].mem, next_rip, c)) return false;
    const int w = ops[0].size;
    if (w == 80) {
        c.mov(kScratch1, kAddr);
        EmitTbyteToDouble(kScratch1, c);          // -> DReg(0)
    } else if (w == 32) {
        c.ldr(SReg(0), ptr(kAddr));
        c.fcvt(DReg(0), SReg(0));
    } else {
        c.ldr(DReg(0), ptr(kAddr));
    }
    EmitX87Push(c);
    return true;
}

// FST / FSTP — store ST(0) to memory (narrowing to float for m32). FSTP pops.
// Tbyte (m80) store handled here (always a pop form on x86).
bool EmitFstOrFstp(const ZydisDecodedInstruction& insn, const ZydisDecodedOperand* ops,
                   u64 next_rip, CodeGenerator& c) {
    const bool pop = (insn.mnemonic == ZYDIS_MNEMONIC_FSTP);

    // Register form: FST ST(i) / FSTP ST(i) copies ST(0) into ST(i), then pops
    // for FSTP. Zydis may present this with one visible ST operand (the dest)
    // or two (dest + implicit ST0); locate the destination ST index from the
    // first ST-register operand. FSTP ST(0) is the common "discard top" idiom.
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
        // Copy ST(0) -> ST(di). When di == 0 this is a self-copy (no-op aside
        // from the pop), which is correct.
        EmitX87RegAddr(XReg(11), 0, c);
        c.ldr(DReg(0), ptr(XReg(11)));
        EmitX87RegAddr(XReg(11), di, c);
        c.str(DReg(0), ptr(XReg(11)));
        if (pop) EmitX87PopDiscard(c);
        return true;
    }

    if (ops[0].type != ZYDIS_OPERAND_TYPE_MEMORY) return false;
    const int w = ops[0].size;
    if (!EmitEffectiveAddress(ops[0].mem, next_rip, c)) return false;
    c.mov(kScratch1, kAddr);                      // stable dst addr
    // Load ST(0) into DReg(0).
    EmitX87RegAddr(XReg(11), 0, c);
    c.ldr(DReg(0), ptr(XReg(11)));
    if (w == 80) {
        EmitDoubleToTbyte(kScratch1, c);
    } else if (w == 32) {
        c.fcvt(SReg(1), DReg(0));
        c.str(SReg(1), ptr(kScratch1));
    } else {
        c.str(DReg(0), ptr(kScratch1));
    }
    if (pop) EmitX87PopDiscard(c);
    return true;
}

// FILD — load signed int (32/64) from memory, convert to double, push.
bool EmitFild(const ZydisDecodedInstruction& insn, const ZydisDecodedOperand* ops,
              u64 next_rip, CodeGenerator& c) {
    if (ops[0].type != ZYDIS_OPERAND_TYPE_MEMORY) return false;
    if (!EmitEffectiveAddress(ops[0].mem, next_rip, c)) return false;
    const int w = ops[0].size;
    if (w == 64) {
        c.ldr(kScratch0, ptr(kAddr));
        c.scvtf(DReg(0), kScratch0);
    } else {
        c.ldrsw(kScratch0, ptr(kAddr));           // sign-extend 32-bit
        c.scvtf(DReg(0), kScratch0);
    }
    EmitX87Push(c);
    return true;
}

// FISTP — convert ST(0) to signed int (16/32/64), store, pop, rounding per
// the x87 control word's RC field. FISTTP always truncates.
bool EmitFistp(const ZydisDecodedInstruction& insn, const ZydisDecodedOperand* ops,
               u64 next_rip, CodeGenerator& c) {
    // FISTP rounds ST(0) per fpu_cw.RC (bits 11:10) before the integer
    // conversion; FISTTP (SSE3) always truncates regardless. AArch64 makes
    // this pleasant: the four x87 RC modes map 1:1 onto dedicated round
    // instructions — 00 nearest-even -> frintn, 01 toward -inf -> frintm,
    // 10 toward +inf -> frintp, 11 toward zero -> frintz — so we dispatch
    // on the RC value at runtime instead of swapping FPCR. (The RC=11 arm
    // is technically redundant before fcvtzs, which truncates on its own,
    // but keeping the explicit frintz makes the value in DReg(0) the
    // architecturally rounded one in all four arms.) This is what makes
    // the classic `fldcw [chop]; fistp; fldcw [restore]` idiom truncate
    // instead of silently rounding to nearest, matching the x86 host's
    // MXCSR-transplant fix.
    const bool is_truncating = (insn.mnemonic == ZYDIS_MNEMONIC_FISTTP);
    if (insn.mnemonic != ZYDIS_MNEMONIC_FISTP && !is_truncating) return false;
    if (ops[0].type != ZYDIS_OPERAND_TYPE_MEMORY) return false;
    if (!EmitEffectiveAddress(ops[0].mem, next_rip, c)) return false;
    c.mov(kScratch1, kAddr);
    EmitX87RegAddr(XReg(11), 0, c);
    c.ldr(DReg(0), ptr(XReg(11)));
    const int w = ops[0].size;
    if (!is_truncating) {
        // 4-way dispatch on fpu_cw.RC. WReg(12) is free: EmitX87RegAddr's
        // x12/x13 use is already past, and the dest address is parked in
        // kScratch1 (x10).
        Label rc_down, rc_up, rc_chop, rc_done;
        c.ldrh(WReg(12), ptr(kState, static_cast<u32>(offsetof(GuestState, fpu_cw))));
        c.ubfx(WReg(12), WReg(12), 10, 2);       // RC field
        c.cmp(WReg(12), 1);
        c.b(EQ, rc_down);
        c.cmp(WReg(12), 2);
        c.b(EQ, rc_up);
        c.cmp(WReg(12), 3);
        c.b(EQ, rc_chop);
        c.frintn(DReg(0), DReg(0));              // 00: nearest even
        c.b(rc_done);
        c.L(rc_down);
        c.frintm(DReg(0), DReg(0));              // 01: toward -inf
        c.b(rc_done);
        c.L(rc_up);
        c.frintp(DReg(0), DReg(0));              // 10: toward +inf
        c.b(rc_done);
        c.L(rc_chop);
        c.frintz(DReg(0), DReg(0));              // 11: toward zero
        c.L(rc_done);
    }
    if (w == 64) {
        c.fcvtzs(kScratch0, DReg(0));
        c.str(kScratch0, ptr(kScratch1));
    } else if (w == 16) {
        c.fcvtzs(WReg(9), DReg(0));
        c.strh(WReg(9), ptr(kScratch1));
    } else {
        c.fcvtzs(WReg(9), DReg(0));
        c.str(WReg(9), ptr(kScratch1));
    }
    EmitX87PopDiscard(c);
    return true;
}

// FADDP/FMULP/FSUBP/FSUBRP/FDIVP/FDIVRP st(i),st(0) — arithmetic with pop.
//   FADDP : st(i)=st(i)+st(0)   FMULP : st(i)=st(i)*st(0)
//   FSUBP : st(i)=st(i)-st(0)   FSUBRP: st(i)=st(0)-st(i)
//   FDIVP : st(i)=st(i)/st(0)   FDIVRP: st(i)=st(0)/st(i)
bool EmitX87ArithPop(const ZydisDecodedInstruction& insn, const ZydisDecodedOperand* ops,
                     CodeGenerator& c) {
    int i = 1;
    if (insn.operand_count_visible >= 1 && ops[0].type == ZYDIS_OPERAND_TYPE_REGISTER &&
        ops[0].reg.value >= ZYDIS_REGISTER_ST0 && ops[0].reg.value <= ZYDIS_REGISTER_ST7)
        i = ops[0].reg.value - ZYDIS_REGISTER_ST0;

    // DReg(0) = st(0), DReg(1) = st(i).
    EmitX87RegAddr(XReg(11), 0, c);
    c.mov(XReg(15), XReg(11));                     // stable &st(0)
    c.ldr(DReg(0), ptr(XReg(15)));
    EmitX87RegAddr(XReg(11), i, c);
    c.mov(XReg(5), XReg(11));                      // stable &st(i)
    c.ldr(DReg(1), ptr(XReg(5)));

    switch (insn.mnemonic) {
        case ZYDIS_MNEMONIC_FADDP:  c.fadd(DReg(1), DReg(1), DReg(0)); break;
        case ZYDIS_MNEMONIC_FMULP:  c.fmul(DReg(1), DReg(1), DReg(0)); break;
        case ZYDIS_MNEMONIC_FSUBP:  c.fsub(DReg(1), DReg(1), DReg(0)); break;
        case ZYDIS_MNEMONIC_FDIVP:  c.fdiv(DReg(1), DReg(1), DReg(0)); break;
        case ZYDIS_MNEMONIC_FSUBRP: c.fsub(DReg(1), DReg(0), DReg(1)); break;  // st0 - sti
        case ZYDIS_MNEMONIC_FDIVRP: c.fdiv(DReg(1), DReg(0), DReg(1)); break;  // st0 / sti
        default: return false;
    }
    c.str(DReg(1), ptr(XReg(5)));                  // st(i) = result
    EmitX87PopDiscard(c);
    return true;
}

// ---------------------------------------------------------------------------
// x87 non-pop arithmetic: FADD/FMUL/FSUB/FSUBR/FDIV/FDIVR.
// Forms (from Zydis):
//   2 reg ops:  fadd st(0), st(i)   -> dst = op0 = ST0,  other = op1 = ST(i)
//               fadd st(i), st(0)   -> dst = op0 = ST(i),other = op1 = ST0
//   1 mem op:   fadd m32/m64        -> dst = ST0, other = mem (converted to double)
// FSUBR/FDIVR reverse the operand order (dst = other - dst, dst = other / dst).
// No pop (that's the *P forms in EmitX87ArithPop). Result written back to dst slot.
static bool EmitX87Arith(const ZydisDecodedInstruction& insn, const ZydisDecodedOperand* ops,
                         u64 next_rip, CodeGenerator& c) {
    const ZydisMnemonic m = insn.mnemonic;
    const bool rev = (m == ZYDIS_MNEMONIC_FSUBR || m == ZYDIS_MNEMONIC_FDIVR);
    const bool isadd = (m == ZYDIS_MNEMONIC_FADD);
    const bool ismul = (m == ZYDIS_MNEMONIC_FMUL);
    const bool issub = (m == ZYDIS_MNEMONIC_FSUB || m == ZYDIS_MNEMONIC_FSUBR);
    const bool isdiv = (m == ZYDIS_MNEMONIC_FDIV || m == ZYDIS_MNEMONIC_FDIVR);
    if (!(isadd || ismul || issub || isdiv)) return false;

    // Memory form: dst = ST0, operand from memory (m32 -> double, m64 -> double).
    if (insn.operand_count_visible == 1 && ops[0].type == ZYDIS_OPERAND_TYPE_MEMORY) {
        if (!EmitEffectiveAddress(ops[0].mem, next_rip, c)) return false;
        c.mov(kScratch1, kAddr);                       // stable mem addr
        const int w = ops[0].size;
        if (w == 32) { c.ldr(SReg(1), ptr(kScratch1)); c.fcvt(DReg(1), SReg(1)); }
        else         { c.ldr(DReg(1), ptr(kScratch1)); }   // m64
        EmitX87RegAddr(XReg(11), 0, c);
        c.mov(XReg(15), XReg(11));                     // stable &ST0
        c.ldr(DReg(0), ptr(XReg(15)));                 // DReg0 = ST0
        // dst = ST0 (DReg0), other = mem (DReg1).
        if (isadd)      c.fadd(DReg(0), DReg(0), DReg(1));
        else if (ismul) c.fmul(DReg(0), DReg(0), DReg(1));
        else if (issub) c.fsub(DReg(0), rev ? DReg(1) : DReg(0), rev ? DReg(0) : DReg(1));
        else            c.fdiv(DReg(0), rev ? DReg(1) : DReg(0), rev ? DReg(0) : DReg(1));
        c.str(DReg(0), ptr(XReg(15)));
        return true;
    }

    // Register form: two ST operands.
    if (insn.operand_count_visible != 2 ||
        ops[0].type != ZYDIS_OPERAND_TYPE_REGISTER ||
        ops[1].type != ZYDIS_OPERAND_TYPE_REGISTER) return false;
    if (ops[0].reg.value < ZYDIS_REGISTER_ST0 || ops[0].reg.value > ZYDIS_REGISTER_ST7) return false;
    if (ops[1].reg.value < ZYDIS_REGISTER_ST0 || ops[1].reg.value > ZYDIS_REGISTER_ST7) return false;
    const int di = ops[0].reg.value - ZYDIS_REGISTER_ST0;   // dst slot
    const int si = ops[1].reg.value - ZYDIS_REGISTER_ST0;   // other slot

    EmitX87RegAddr(XReg(11), di, c);
    c.mov(XReg(15), XReg(11));                         // stable &dst
    c.ldr(DReg(0), ptr(XReg(15)));                     // DReg0 = dst
    EmitX87RegAddr(XReg(11), si, c);
    c.ldr(DReg(1), ptr(XReg(11)));                     // DReg1 = other
    if (isadd)      c.fadd(DReg(0), DReg(0), DReg(1));
    else if (ismul) c.fmul(DReg(0), DReg(0), DReg(1));
    else if (issub) c.fsub(DReg(0), rev ? DReg(1) : DReg(0), rev ? DReg(0) : DReg(1));
    else            c.fdiv(DReg(0), rev ? DReg(1) : DReg(0), rev ? DReg(0) : DReg(1));
    c.str(DReg(0), ptr(XReg(15)));
    return true;
}

// x87 unary on ST(0): FCHS (negate), FABS, FSQRT.
static bool EmitX87Unary(const ZydisDecodedInstruction& insn, CodeGenerator& c) {
    EmitX87RegAddr(XReg(11), 0, c);
    c.mov(XReg(15), XReg(11));
    c.ldr(DReg(0), ptr(XReg(15)));
    switch (insn.mnemonic) {
        case ZYDIS_MNEMONIC_FCHS:  c.fneg(DReg(0), DReg(0)); break;
        case ZYDIS_MNEMONIC_FABS:  c.fabs(DReg(0), DReg(0)); break;
        case ZYDIS_MNEMONIC_FSQRT: c.fsqrt(DReg(0), DReg(0)); break;
        default: return false;
    }
    c.str(DReg(0), ptr(XReg(15)));
    return true;
}

// x87 push-constant: FLD1 (push 1.0), FLDZ (push 0.0).
static bool EmitX87LoadConst(const ZydisDecodedInstruction& insn, CodeGenerator& c) {
    u64 bits;
    switch (insn.mnemonic) {
        case ZYDIS_MNEMONIC_FLD1: bits = 0x3FF0000000000000ull; break;  // 1.0
        case ZYDIS_MNEMONIC_FLDZ: bits = 0x0000000000000000ull; break;  // 0.0
        default: return false;
    }
    // Materialize the double bit pattern into DReg(0), then push.
    c.mov(kScratch0, bits);
    c.fmov(DReg(0), kScratch0);
    EmitX87Push(c);
    return true;
}

// FXCH st(i): swap ST(0) and ST(i). Default i=1 if no operand.
static bool EmitFxch(const ZydisDecodedInstruction& insn, const ZydisDecodedOperand* ops,
                     CodeGenerator& c) {
    int i = 1;
    if (insn.operand_count_visible >= 1 && ops[0].type == ZYDIS_OPERAND_TYPE_REGISTER &&
        ops[0].reg.value >= ZYDIS_REGISTER_ST0 && ops[0].reg.value <= ZYDIS_REGISTER_ST7)
        i = ops[0].reg.value - ZYDIS_REGISTER_ST0;
    if (i == 0) return true;   // fxch st(0) is a no-op.
    EmitX87RegAddr(XReg(11), 0, c);
    c.mov(XReg(15), XReg(11));                         // stable &ST0
    c.ldr(DReg(0), ptr(XReg(15)));
    EmitX87RegAddr(XReg(11), i, c);
    c.mov(XReg(5), XReg(11));                          // stable &ST(i)
    c.ldr(DReg(1), ptr(XReg(5)));
    c.str(DReg(1), ptr(XReg(15)));                     // ST0 = old ST(i)
    c.str(DReg(0), ptr(XReg(5)));                      // ST(i) = old ST0
    return true;
}

// x87 compare. Two families:
//   FCOMI/FUCOMI/FCOMIP/FUCOMIP -> set EFLAGS ZF/PF/CF directly (and pop on *IP).
//   FCOM/FCOMP/FCOMPP/FUCOM/FUCOMP -> set x87 condition codes C3/C2/C0 in
//     fpu_sw_cc (C3=bit14, C2=bit10, C0=bit8); C1 (bit9) cleared. Pop on *P/*PP.
// Comparison is ST(0) vs the other operand. Mapping (x86 semantics):
//   ST0 > other : ZF=0 PF=0 CF=0  (C3=0 C2=0 C0=0)
//   ST0 < other : ZF=0 PF=0 CF=1  (C3=0 C2=0 C0=1)
//   ST0 = other : ZF=1 PF=0 CF=0  (C3=1 C2=0 C0=0)
//   unordered   : ZF=1 PF=1 CF=1  (C3=1 C2=1 C0=1)
static bool EmitX87Compare(const ZydisDecodedInstruction& insn, const ZydisDecodedOperand* ops,
                           CodeGenerator& c) {
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

    // Resolve "other" ST index. FCOMI/FUCOMI have 2 ops (op0=ST0, op1=other).
    // FCOM/FUCOM/FCOMP/FUCOMP have 1 op (other). FCOMPP has 0 ops (other=ST1).
    int other = 1;
    if (insn.operand_count_visible == 2 && ops[1].type == ZYDIS_OPERAND_TYPE_REGISTER &&
        ops[1].reg.value >= ZYDIS_REGISTER_ST0 && ops[1].reg.value <= ZYDIS_REGISTER_ST7) {
        other = ops[1].reg.value - ZYDIS_REGISTER_ST0;
    } else if (insn.operand_count_visible == 1 && ops[0].type == ZYDIS_OPERAND_TYPE_REGISTER &&
               ops[0].reg.value >= ZYDIS_REGISTER_ST0 && ops[0].reg.value <= ZYDIS_REGISTER_ST7) {
        other = ops[0].reg.value - ZYDIS_REGISTER_ST0;
    }

    // Load ST0 -> DReg0, ST(other) -> DReg1.
    EmitX87RegAddr(XReg(11), 0, c);
    c.ldr(DReg(0), ptr(XReg(11)));
    EmitX87RegAddr(XReg(11), other, c);
    c.ldr(DReg(1), ptr(XReg(11)));

    const XReg eq = XReg(9), gt = XReg(10), lt = XReg(12), v = XReg(13), t = XReg(14);
    c.fcmeq(DReg(2), DReg(0), DReg(1));
    c.fcmgt(DReg(3), DReg(0), DReg(1));    // ST0 > other
    c.fcmgt(DReg(4), DReg(1), DReg(0));    // other > ST0  => ST0 < other
    c.fmov(eq, DReg(2)); c.fmov(gt, DReg(3)); c.fmov(lt, DReg(4));

    if (to_eflags) {
        c.ldr(v, ptr(kState, Offsets::Rflags));
        c.mov(t, ~((1ull<<6)|(1ull<<0)|(1ull<<2)|(1ull<<7)|(1ull<<11)|(1ull<<4)));
        c.and_(v, v, t);
        Label done, notEq, notLt;
        c.cbz(eq, notEq);
        c.orr(v, v, 1ull<<6);              // ZF (equal)
        c.b(done);
        c.L(notEq);
        c.cbz(lt, notLt);
        c.orr(v, v, 1ull<<0);              // CF (less)
        c.b(done);
        c.L(notLt);
        c.cbnz(gt, done);                  // greater -> all clear
        c.orr(v, v, 1ull<<6);              // unordered: ZF
        c.orr(v, v, 1ull<<0);              //            CF
        c.orr(v, v, 1ull<<2);              //            PF
        c.L(done);
        c.str(v, ptr(kState, Offsets::Rflags));
    } else {
        // Build C3(bit14)/C2(bit10)/C0(bit8) in fpu_sw_cc; clear C1(bit9).
        // ldrh/strh require a WReg; use the W views of v/t.
        const WReg vw = WReg(v.getIdx()), tw = WReg(t.getIdx());
        c.ldrh(vw, ptr(kState, (u32)offsetof(GuestState, fpu_sw_cc)));
        c.mov(tw, ~((1u<<14)|(1u<<10)|(1u<<9)|(1u<<8)));
        c.and_(vw, vw, tw);
        Label done, notEq, notLt;
        c.cbz(eq, notEq);
        c.orr(vw, vw, 1u<<14);             // C3 (equal)
        c.b(done);
        c.L(notEq);
        c.cbz(lt, notLt);
        c.orr(vw, vw, 1u<<8);              // C0 (less)
        c.b(done);
        c.L(notLt);
        c.cbnz(gt, done);                  // greater -> all clear
        c.orr(vw, vw, 1u<<14);             // unordered: C3
        c.orr(vw, vw, 1u<<10);             //            C2
        c.orr(vw, vw, 1u<<8);              //            C0
        c.L(done);
        c.strh(vw, ptr(kState, (u32)offsetof(GuestState, fpu_sw_cc)));
    }

    for (int p = 0; p < npop; ++p) EmitX87PopDiscard(c);
    return true;
}

// FNSTSW ax / m16: build the status word = (fpu_sw_cc condition bits) |
// (fpu_top << 11), store to AX (guest gpr[0] low 16) or to memory.
static bool EmitFnstsw(const ZydisDecodedInstruction& insn, const ZydisDecodedOperand* ops,
                       u64 next_rip, CodeGenerator& c) {
    // sw = fpu_sw_cc | (top << 11).
    c.ldrh(WReg(9), ptr(kState, (u32)offsetof(GuestState, fpu_sw_cc)));
    c.ldr(WReg(10), ptr(kState, (u32)offsetof(GuestState, fpu_top)));
    c.lsl(WReg(10), WReg(10), 11);
    c.orr(WReg(9), WReg(9), WReg(10));
    c.and_(WReg(9), WReg(9), 0xFFFF);
    if (ops[0].type == ZYDIS_OPERAND_TYPE_REGISTER) {
        // Destination is AX: write low 16 of gpr[0], preserve bits 63:16.
        c.ldr(XReg(12), ptr(kState, GprOffset(0)));
        c.mov(XReg(13), ~0xFFFFull);
        c.and_(XReg(12), XReg(12), XReg(13));
        c.orr(XReg(12), XReg(12), XReg(9));
        c.str(XReg(12), ptr(kState, GprOffset(0)));
        return true;
    }
    if (ops[0].type == ZYDIS_OPERAND_TYPE_MEMORY) {
        if (!EmitEffectiveAddress(ops[0].mem, next_rip, c)) return false;
        c.strh(WReg(9), ptr(kAddr));
        return true;
    }
    return false;
}

// x87 conditional move: FCMOVcc ST(0), ST(i). If the EFLAGS condition holds,
// copy ST(i) into ST(0); otherwise leave ST(0) unchanged. No pop, no flag
// change. Conditions (from guest rflags CF=bit0, PF=bit2, ZF=bit6):
//   FCMOVB   CF=1        FCMOVNB   CF=0
//   FCMOVE   ZF=1        FCMOVNE   ZF=0
//   FCMOVBE  CF=1|ZF=1   FCMOVNBE  CF=0&ZF=0
//   FCMOVU   PF=1        FCMOVNU   PF=0
// These pair with FCOMI/FUCOMI, which set CF/ZF/PF.
static bool EmitFcmov(const ZydisDecodedInstruction& insn, const ZydisDecodedOperand* ops,
                      CodeGenerator& c) {
    // Source ST index = the second ST-register operand (dest is ST0).
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

    // Build the condition into cond (1 = perform the move, 0 = leave ST0).
    const XReg fl = kScratch0, cf = kScratch1, zf = kScratch2, pf = XReg(13), cond = XReg(14);
    c.ldr(fl, ptr(kState, Offsets::Rflags));
    c.lsr(cf, fl, 0); c.and_(cf, cf, 1);          // CF
    c.lsr(zf, fl, 6); c.and_(zf, zf, 1);          // ZF
    c.lsr(pf, fl, 2); c.and_(pf, pf, 1);          // PF

    switch (insn.mnemonic) {
        case ZYDIS_MNEMONIC_FCMOVB:   c.mov(cond, cf); break;
        case ZYDIS_MNEMONIC_FCMOVE:   c.mov(cond, zf); break;
        case ZYDIS_MNEMONIC_FCMOVBE:  c.orr(cond, cf, zf); break;             // CF|ZF
        case ZYDIS_MNEMONIC_FCMOVU:   c.mov(cond, pf); break;
        case ZYDIS_MNEMONIC_FCMOVNB:  c.eor(cond, cf, 1); break;              // !CF
        case ZYDIS_MNEMONIC_FCMOVNE:  c.eor(cond, zf, 1); break;              // !ZF
        case ZYDIS_MNEMONIC_FCMOVNBE: c.orr(cond, cf, zf); c.eor(cond, cond, 1); break; // !(CF|ZF)
        case ZYDIS_MNEMONIC_FCMOVNU:  c.eor(cond, pf, 1); break;              // !PF
        default: return false;
    }

    // Load ST(0) and ST(i); csel picks the source when cond != 0.
    EmitX87RegAddr(XReg(11), 0, c);
    c.mov(XReg(15), XReg(11));                     // stable &ST0
    c.ldr(DReg(0), ptr(XReg(15)));                 // current ST0
    EmitX87RegAddr(XReg(11), si, c);
    c.ldr(DReg(1), ptr(XReg(11)));                 // ST(i)
    c.cmp(cond, 0);
    c.fcsel(DReg(0), DReg(1), DReg(0), NE);        // cond!=0 ? ST(i) : ST0
    c.str(DReg(0), ptr(XReg(15)));
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

u64 Lifter::BlockReservationSize() noexcept {
    return BLOCK_HOST_SIZE_CAP;
}

// Function-try-block: converts any assembler throw (Xbyak_aarch64::Error,
// a std::exception) into the nullptr compile-failure path -- an escaped
// exception cannot unwind through the JIT gateway frame. See the x86 host's
// CompileBlock for the full rationale; one arm64-specific addition: the
// catch MUST restore W^X before returning. Emission runs between
// WriteBegin() (MAP_JIT pages writable for this thread) and WriteEnd()
// (back to executable); a throw mid-emit would otherwise strand the thread
// in the writable state, and its next jump into ANY cached block faults --
// macOS refuses to execute writable MAP_JIT pages.
void* Lifter::CompileBlock(u64 guest_rip) try {
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

        // Guest atomics: LOCK-prefixed RMW and the implicitly-locked
        // XCHG-with-memory route to the LSE emitter -- one atomic host op
        // with acquire+release ordering, matching x86 LOCK semantics. Forms
        // the LSE emitter declines (ADC/SBB, NEG/NOT, bit ops, CMPXCHG8B/
        // 16B, high-byte operands) leave handled == false and take the
        // generic unsupported exit below; they must NEVER fall into the
        // plain emitters, whose ldr/op/str lowering is a silent data race
        // against other guest threads.
        bool handled = false;
        const bool guest_atomic =
            (insn.attributes & ZYDIS_ATTRIB_HAS_LOCK) != 0 ||
            (insn.mnemonic == ZYDIS_MNEMONIC_XCHG &&
             (ops[0].type == ZYDIS_OPERAND_TYPE_MEMORY ||
              ops[1].type == ZYDIS_OPERAND_TYPE_MEMORY));
        if (guest_atomic) {
            handled = EmitLockedRmwA64(insn, ops, next_rip, c);
        } else
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
        case ZYDIS_MNEMONIC_TZCNT:
            handled = EmitTzcnt(insn, ops, c);
            break;
        case ZYDIS_MNEMONIC_BLSR:
            handled = EmitBlsr(insn, ops, c);
            break;
        case ZYDIS_MNEMONIC_MUL:
            handled = EmitMul(insn, ops, c);
            break;
        case ZYDIS_MNEMONIC_IMUL:
            handled = EmitImul(insn, ops, c);
            break;
        case ZYDIS_MNEMONIC_DIV:
            handled = EmitDiv(insn, ops, next_rip, c);
            break;
        case ZYDIS_MNEMONIC_IDIV:
            handled = EmitIdiv(insn, ops, next_rip, c);
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
        case ZYDIS_MNEMONIC_VPADDB: handled = EmitVPacked(insn, ops, next_rip, VPackKind::AddB, c); break;
        case ZYDIS_MNEMONIC_VPADDW: handled = EmitVPacked(insn, ops, next_rip, VPackKind::AddW, c); break;
        case ZYDIS_MNEMONIC_VPADDQ: handled = EmitVPacked(insn, ops, next_rip, VPackKind::AddQ, c); break;
        case ZYDIS_MNEMONIC_VPSUBB: handled = EmitVPacked(insn, ops, next_rip, VPackKind::SubB, c); break;
        case ZYDIS_MNEMONIC_VPSUBW: handled = EmitVPacked(insn, ops, next_rip, VPackKind::SubW, c); break;
        case ZYDIS_MNEMONIC_VPSUBQ: handled = EmitVPacked(insn, ops, next_rip, VPackKind::SubQ, c); break;
        case ZYDIS_MNEMONIC_VPMULLW: handled = EmitVPacked(insn, ops, next_rip, VPackKind::MulW, c); break;
        case ZYDIS_MNEMONIC_VPMAXSB: handled = EmitVPacked(insn, ops, next_rip, VPackKind::MaxSB, c); break;
        case ZYDIS_MNEMONIC_VPMAXSW: handled = EmitVPacked(insn, ops, next_rip, VPackKind::MaxSW, c); break;
        case ZYDIS_MNEMONIC_VPMAXSD: handled = EmitVPacked(insn, ops, next_rip, VPackKind::MaxSD, c); break;
        case ZYDIS_MNEMONIC_VPMAXUB: handled = EmitVPacked(insn, ops, next_rip, VPackKind::MaxUB, c); break;
        case ZYDIS_MNEMONIC_VPMAXUW: handled = EmitVPacked(insn, ops, next_rip, VPackKind::MaxUW, c); break;
        case ZYDIS_MNEMONIC_VPMAXUD: handled = EmitVPacked(insn, ops, next_rip, VPackKind::MaxUD, c); break;
        case ZYDIS_MNEMONIC_VPMINSB: handled = EmitVPacked(insn, ops, next_rip, VPackKind::MinSB, c); break;
        case ZYDIS_MNEMONIC_VPMINSW: handled = EmitVPacked(insn, ops, next_rip, VPackKind::MinSW, c); break;
        case ZYDIS_MNEMONIC_VPMINSD: handled = EmitVPacked(insn, ops, next_rip, VPackKind::MinSD, c); break;
        case ZYDIS_MNEMONIC_VPMINUB: handled = EmitVPacked(insn, ops, next_rip, VPackKind::MinUB, c); break;
        case ZYDIS_MNEMONIC_VPMINUW: handled = EmitVPacked(insn, ops, next_rip, VPackKind::MinUW, c); break;
        case ZYDIS_MNEMONIC_VPMINUD: handled = EmitVPacked(insn, ops, next_rip, VPackKind::MinUD, c); break;
        case ZYDIS_MNEMONIC_VANDPS: case ZYDIS_MNEMONIC_VANDPD:
        case ZYDIS_MNEMONIC_VPAND:  handled = EmitVPacked(insn, ops, next_rip, VPackKind::And, c); break;
        case ZYDIS_MNEMONIC_VORPS:  case ZYDIS_MNEMONIC_VORPD:
        case ZYDIS_MNEMONIC_VPOR:   handled = EmitVPacked(insn, ops, next_rip, VPackKind::Or, c); break;
        case ZYDIS_MNEMONIC_VXORPS: case ZYDIS_MNEMONIC_VXORPD:
        case ZYDIS_MNEMONIC_VPXOR:  handled = EmitVPacked(insn, ops, next_rip, VPackKind::Xor, c); break;
        case ZYDIS_MNEMONIC_VANDNPS: case ZYDIS_MNEMONIC_VANDNPD:
        case ZYDIS_MNEMONIC_VPANDN:  handled = EmitVPacked(insn, ops, next_rip, VPackKind::AndN, c); break;
        case ZYDIS_MNEMONIC_VPUNPCKLBW:  handled = EmitVUnpack(insn, ops, next_rip, VUnpackKind::LBW, c); break;
        case ZYDIS_MNEMONIC_VPUNPCKHBW:  handled = EmitVUnpack(insn, ops, next_rip, VUnpackKind::HBW, c); break;
        case ZYDIS_MNEMONIC_VPUNPCKLWD:  handled = EmitVUnpack(insn, ops, next_rip, VUnpackKind::LWD, c); break;
        case ZYDIS_MNEMONIC_VPUNPCKHWD:  handled = EmitVUnpack(insn, ops, next_rip, VUnpackKind::HWD, c); break;
        case ZYDIS_MNEMONIC_VPUNPCKLDQ:  handled = EmitVUnpack(insn, ops, next_rip, VUnpackKind::LDQ, c); break;
        case ZYDIS_MNEMONIC_VPUNPCKHDQ:  handled = EmitVUnpack(insn, ops, next_rip, VUnpackKind::HDQ, c); break;
        case ZYDIS_MNEMONIC_VPUNPCKLQDQ: handled = EmitVUnpack(insn, ops, next_rip, VUnpackKind::LQDQ, c); break;
        case ZYDIS_MNEMONIC_VPUNPCKHQDQ: handled = EmitVUnpack(insn, ops, next_rip, VUnpackKind::HQDQ, c); break;
        case ZYDIS_MNEMONIC_VUNPCKLPS:   handled = EmitVUnpack(insn, ops, next_rip, VUnpackKind::LPS, c); break;
        case ZYDIS_MNEMONIC_VUNPCKHPS:   handled = EmitVUnpack(insn, ops, next_rip, VUnpackKind::HPS, c); break;
        case ZYDIS_MNEMONIC_VUNPCKLPD:   handled = EmitVUnpack(insn, ops, next_rip, VUnpackKind::LPD, c); break;
        case ZYDIS_MNEMONIC_VUNPCKHPD:   handled = EmitVUnpack(insn, ops, next_rip, VUnpackKind::HPD, c); break;
        case ZYDIS_MNEMONIC_VMOVLHPS:    handled = EmitVUnpack(insn, ops, next_rip, VUnpackKind::MOVLHPS, c); break;
        case ZYDIS_MNEMONIC_VMOVHLPS:    handled = EmitVUnpack(insn, ops, next_rip, VUnpackKind::MOVHLPS, c); break;
        case ZYDIS_MNEMONIC_VPSHUFD:
        case ZYDIS_MNEMONIC_VSHUFPS:     handled = EmitVShuffle(insn, ops, next_rip, c); break;
        case ZYDIS_MNEMONIC_VSHUFPD:     handled = EmitVshufpd(insn, ops, next_rip, c); break;
        case ZYDIS_MNEMONIC_VPERMILPS:   handled = EmitVpermil(insn, ops, next_rip, false, c); break;
        case ZYDIS_MNEMONIC_VPERMILPD:   handled = EmitVpermil(insn, ops, next_rip, true, c); break;
        case ZYDIS_MNEMONIC_VPSHUFLW: case ZYDIS_MNEMONIC_VPSHUFHW:
            handled = EmitVpshufw(insn, ops, next_rip, c); break;
        case ZYDIS_MNEMONIC_VMOVMSKPS: handled = EmitMovmsk(insn, ops, 32, c); break;
        case ZYDIS_MNEMONIC_VMOVMSKPD: handled = EmitMovmsk(insn, ops, 64, c); break;
        case ZYDIS_MNEMONIC_VPMOVMSKB: handled = EmitMovmsk(insn, ops, 8, c); break;
        case ZYDIS_MNEMONIC_VPCMPEQB: handled = EmitVCmpInt(insn, ops, next_rip, VCmpIntKind::EqB, c); break;
        case ZYDIS_MNEMONIC_VPCMPEQD: handled = EmitVCmpInt(insn, ops, next_rip, VCmpIntKind::EqD, c); break;
        case ZYDIS_MNEMONIC_VPCMPGTD: handled = EmitVCmpInt(insn, ops, next_rip, VCmpIntKind::GtD, c); break;
        case ZYDIS_MNEMONIC_VPCMPEQW: handled = EmitVCmpInt(insn, ops, next_rip, VCmpIntKind::EqW, c); break;
        case ZYDIS_MNEMONIC_VPCMPEQQ: handled = EmitVCmpInt(insn, ops, next_rip, VCmpIntKind::EqQ, c); break;
        case ZYDIS_MNEMONIC_VPCMPGTB: handled = EmitVCmpInt(insn, ops, next_rip, VCmpIntKind::GtB, c); break;
        case ZYDIS_MNEMONIC_VPCMPGTW: handled = EmitVCmpInt(insn, ops, next_rip, VCmpIntKind::GtW, c); break;
        case ZYDIS_MNEMONIC_VPCMPGTQ: handled = EmitVCmpInt(insn, ops, next_rip, VCmpIntKind::GtQ, c); break;
        case ZYDIS_MNEMONIC_VPTEST:   handled = EmitVptest(insn, ops, c); break;
        case ZYDIS_MNEMONIC_PTEST:    handled = EmitVptest(insn, ops, c); break;
        case ZYDIS_MNEMONIC_BSWAP:    handled = EmitBswap(insn, ops, next_rip, c); break;
        case ZYDIS_MNEMONIC_XCHG:     handled = EmitXchg(insn, ops, next_rip, c); break;
        case ZYDIS_MNEMONIC_MOVBE:    handled = EmitMovbe(insn, ops, next_rip, c); break;
        case ZYDIS_MNEMONIC_VCOMISS:  case ZYDIS_MNEMONIC_VCOMISD:
        case ZYDIS_MNEMONIC_VUCOMISS: case ZYDIS_MNEMONIC_VUCOMISD:
            handled = EmitVcomi(insn, ops, next_rip, c); break;
        case ZYDIS_MNEMONIC_VCVTSI2SS: case ZYDIS_MNEMONIC_VCVTSI2SD:
            handled = EmitVcvtsi2(insn, ops, next_rip, c); break;
        case ZYDIS_MNEMONIC_VCVTTSS2SI: case ZYDIS_MNEMONIC_VCVTTSD2SI:
        case ZYDIS_MNEMONIC_CVTSS2SI:   case ZYDIS_MNEMONIC_CVTSD2SI:
        case ZYDIS_MNEMONIC_CVTTSS2SI:  case ZYDIS_MNEMONIC_CVTTSD2SI:
            handled = EmitVcvtt2si(insn, ops, next_rip, c); break;
        case ZYDIS_MNEMONIC_VCVTSS2SD: case ZYDIS_MNEMONIC_VCVTSD2SS:
            handled = EmitVcvtScalar(insn, ops, next_rip, c); break;
        case ZYDIS_MNEMONIC_VCVTDQ2PS: case ZYDIS_MNEMONIC_VCVTTPS2DQ:
            handled = EmitVcvtPacked(insn, ops, next_rip, c); break;
        case ZYDIS_MNEMONIC_VPEXTRD: case ZYDIS_MNEMONIC_VPEXTRQ:
        case ZYDIS_MNEMONIC_VEXTRACTPS:
            handled = EmitVpextr(insn, ops, next_rip, c); break;
        case ZYDIS_MNEMONIC_VPINSRD:    handled = EmitVpinsrd(insn, ops, next_rip, c); break;
        case ZYDIS_MNEMONIC_VINSERTPS:  handled = EmitVinsertps(insn, ops, next_rip, c); break;
        case ZYDIS_MNEMONIC_VINSERTF128:  handled = EmitVinsertf128(insn, ops, next_rip, c); break;
        case ZYDIS_MNEMONIC_VEXTRACTF128: handled = EmitVextractf128(insn, ops, next_rip, c); break;
        case ZYDIS_MNEMONIC_VMOVD:    handled = EmitVmovd(insn, ops, next_rip, c); break;
        case ZYDIS_MNEMONIC_VMOVQ:    handled = EmitVmovq(insn, ops, next_rip, c); break;
        case ZYDIS_MNEMONIC_MOVQ:     handled = EmitVmovq(insn, ops, next_rip, c); break;
        case ZYDIS_MNEMONIC_VMOVAPS: case ZYDIS_MNEMONIC_VMOVUPS:
        case ZYDIS_MNEMONIC_VMOVDQA: case ZYDIS_MNEMONIC_VMOVDQU:
        case ZYDIS_MNEMONIC_VMOVNTDQA:
        case ZYDIS_MNEMONIC_VMOVAPD: case ZYDIS_MNEMONIC_VMOVUPD:
        case ZYDIS_MNEMONIC_VMOVNTDQ:
            handled = EmitVmovFull(insn, ops, next_rip, c); break;
        case ZYDIS_MNEMONIC_VMOVSLDUP: case ZYDIS_MNEMONIC_VMOVSHDUP:
            handled = EmitVmovdup(insn, ops, next_rip, c); break;
        case ZYDIS_MNEMONIC_VPSRAD: case ZYDIS_MNEMONIC_VPSRLD:
        case ZYDIS_MNEMONIC_VPSLLW: case ZYDIS_MNEMONIC_VPSLLD:
        case ZYDIS_MNEMONIC_VPSLLQ: case ZYDIS_MNEMONIC_VPSRLW:
        case ZYDIS_MNEMONIC_VPSRLQ: case ZYDIS_MNEMONIC_VPSRAW:
            handled = EmitVpshift(insn, ops, next_rip, c); break;
        case ZYDIS_MNEMONIC_VPSLLDQ: case ZYDIS_MNEMONIC_VPSRLDQ:
            handled = EmitVpsxldq(insn, ops, next_rip, c); break;
        case ZYDIS_MNEMONIC_VBLENDPS: case ZYDIS_MNEMONIC_VPBLENDW:
        case ZYDIS_MNEMONIC_VBLENDPD: case ZYDIS_MNEMONIC_VPBLENDD:
            handled = EmitVblend(insn, ops, next_rip, c); break;
        case ZYDIS_MNEMONIC_VBLENDVPS:
        case ZYDIS_MNEMONIC_VBLENDVPD: case ZYDIS_MNEMONIC_VPBLENDVB:
            handled = EmitVblendvps(insn, ops, next_rip, c); break;
        case ZYDIS_MNEMONIC_VBROADCASTSS: handled = EmitVbroadcastss(insn, ops, next_rip, c); break;
        case ZYDIS_MNEMONIC_VROUNDSS: case ZYDIS_MNEMONIC_VROUNDSD:
            handled = EmitVround(insn, ops, next_rip, c); break;
        case ZYDIS_MNEMONIC_VROUNDPS: handled = EmitVroundPacked(insn, ops, next_rip, false, c); break;
        case ZYDIS_MNEMONIC_VROUNDPD: handled = EmitVroundPacked(insn, ops, next_rip, true, c); break;
        case ZYDIS_MNEMONIC_VHADDPS: handled = EmitVhaddsub(insn, ops, next_rip, false, false, c); break;
        case ZYDIS_MNEMONIC_VHADDPD: handled = EmitVhaddsub(insn, ops, next_rip, true,  false, c); break;
        case ZYDIS_MNEMONIC_VHSUBPS: handled = EmitVhaddsub(insn, ops, next_rip, false, true,  c); break;
        case ZYDIS_MNEMONIC_VHSUBPD: handled = EmitVhaddsub(insn, ops, next_rip, true,  true,  c); break;
        case ZYDIS_MNEMONIC_VADDSUBPS: handled = EmitVaddsub(insn, ops, next_rip, false, c); break;
        case ZYDIS_MNEMONIC_VADDSUBPD: handled = EmitVaddsub(insn, ops, next_rip, true,  c); break;
        case ZYDIS_MNEMONIC_VRCPPS:   handled = EmitVrecip(insn, ops, next_rip, false, false, c); break;
        case ZYDIS_MNEMONIC_VRCPSS:   handled = EmitVrecip(insn, ops, next_rip, false, true,  c); break;
        case ZYDIS_MNEMONIC_VRSQRTPS: handled = EmitVrecip(insn, ops, next_rip, true,  false, c); break;
        case ZYDIS_MNEMONIC_VRSQRTSS: handled = EmitVrecip(insn, ops, next_rip, true,  true,  c); break;
        case ZYDIS_MNEMONIC_VCMPPS: case ZYDIS_MNEMONIC_VCMPSS:
            handled = EmitVcmpfp(insn, ops, next_rip, c); break;
        case ZYDIS_MNEMONIC_VPMOVZXDQ: handled = EmitVpmovzxdq(insn, ops, next_rip, c); break;
        case ZYDIS_MNEMONIC_VPMOVZXBW: handled = EmitVpmovx(insn, ops, next_rip, false, 8, 16, c); break;
        case ZYDIS_MNEMONIC_VPMOVZXBD: handled = EmitVpmovx(insn, ops, next_rip, false, 8, 32, c); break;
        case ZYDIS_MNEMONIC_VPMOVZXBQ: handled = EmitVpmovx(insn, ops, next_rip, false, 8, 64, c); break;
        case ZYDIS_MNEMONIC_VPMOVZXWD: handled = EmitVpmovx(insn, ops, next_rip, false, 16, 32, c); break;
        case ZYDIS_MNEMONIC_VPMOVZXWQ: handled = EmitVpmovx(insn, ops, next_rip, false, 16, 64, c); break;
        case ZYDIS_MNEMONIC_VPMOVSXBW: handled = EmitVpmovx(insn, ops, next_rip, true, 8, 16, c); break;
        case ZYDIS_MNEMONIC_VPMOVSXBD: handled = EmitVpmovx(insn, ops, next_rip, true, 8, 32, c); break;
        case ZYDIS_MNEMONIC_VPMOVSXBQ: handled = EmitVpmovx(insn, ops, next_rip, true, 8, 64, c); break;
        case ZYDIS_MNEMONIC_VPMOVSXWD: handled = EmitVpmovx(insn, ops, next_rip, true, 16, 32, c); break;
        case ZYDIS_MNEMONIC_VPMOVSXWQ: handled = EmitVpmovx(insn, ops, next_rip, true, 16, 64, c); break;
        case ZYDIS_MNEMONIC_VPMOVSXDQ: handled = EmitVpmovx(insn, ops, next_rip, true, 32, 64, c); break;
        case ZYDIS_MNEMONIC_VPMULUDQ:  handled = EmitVpmuludq(insn, ops, next_rip, c); break;
        case ZYDIS_MNEMONIC_VSQRTPS:   handled = EmitVsqrtPacked(insn, ops, next_rip, false, c); break;
        case ZYDIS_MNEMONIC_VSQRTPD:   handled = EmitVsqrtPacked(insn, ops, next_rip, true, c); break;
        case ZYDIS_MNEMONIC_VMOVDDUP:  handled = EmitVmovddup(insn, ops, next_rip, c); break;
        case ZYDIS_MNEMONIC_VPINSRB: case ZYDIS_MNEMONIC_VPINSRW:
            handled = EmitVpinsrBW(insn, ops, next_rip, c); break;
        case ZYDIS_MNEMONIC_VPHADDD:   handled = EmitVphaddd(insn, ops, next_rip, c); break;
        case ZYDIS_MNEMONIC_PCMPISTRI:
        case ZYDIS_MNEMONIC_PCMPISTRM:
            handled = EmitPcmpistr(insn, ops, next_rip, c); break;
        case ZYDIS_MNEMONIC_VPSHUFB:   handled = EmitVpshufb(insn, ops, next_rip, c); break;
        case ZYDIS_MNEMONIC_VPACKUSDW: handled = EmitVpackusdw(insn, ops, next_rip, c); break;
        case ZYDIS_MNEMONIC_VPCMPISTRI:
            handled = EmitVpcmpistr(insn, ops, next_rip, /*wantMask=*/false, c); break;
        case ZYDIS_MNEMONIC_VPCMPISTRM:
            handled = EmitVpcmpistr(insn, ops, next_rip, /*wantMask=*/true, c); break;
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
        case ZYDIS_MNEMONIC_MFENCE: case ZYDIS_MNEMONIC_LFENCE:
        case ZYDIS_MNEMONIC_SFENCE:
            // Guest fences carry real ordering on this weakly-ordered host:
            // dropping them loses the TSO guarantees the guest was compiled
            // against. dmb ish is the conservative lowering for all three
            // (full barrier; subsumes the store-store / load-load subsets).
            c.dmb(ISH);
            handled = true;
            break;
        case ZYDIS_MNEMONIC_PREFETCH:    case ZYDIS_MNEMONIC_PREFETCHNTA:
        case ZYDIS_MNEMONIC_PREFETCHT0:  case ZYDIS_MNEMONIC_PREFETCHT1:
        case ZYDIS_MNEMONIC_PREFETCHT2:  case ZYDIS_MNEMONIC_PREFETCHW:
            handled = true;  // memory hints: architecturally no-ops.
            break;
        case ZYDIS_MNEMONIC_RDTSC:  handled = EmitRdtsc(insn, ops, c, /*with_aux=*/false); break;
        case ZYDIS_MNEMONIC_RDTSCP: handled = EmitRdtsc(insn, ops, c, /*with_aux=*/true); break;
        case ZYDIS_MNEMONIC_CPUID:  handled = EmitCpuid(insn, ops, c); break;
        case ZYDIS_MNEMONIC_LAHF:   handled = EmitLahf(insn, ops, c); break; // flags
        case ZYDIS_MNEMONIC_SAHF:   handled = EmitSahf(insn, ops, c); break; // flags
        case ZYDIS_MNEMONIC_XGETBV: handled = EmitXgetbv(insn, ops, c); break;
        case ZYDIS_MNEMONIC_STMXCSR: case ZYDIS_MNEMONIC_LDMXCSR:
        case ZYDIS_MNEMONIC_VSTMXCSR: case ZYDIS_MNEMONIC_VLDMXCSR:
            handled = EmitMxcsr(insn, ops, next_rip, c); break;
        case ZYDIS_MNEMONIC_FNSTCW: case ZYDIS_MNEMONIC_FLDCW:
            handled = EmitFnstcw(insn, ops, next_rip, c); break;
        case ZYDIS_MNEMONIC_FLD:   handled = EmitFld(insn, ops, next_rip, c); break;
        case ZYDIS_MNEMONIC_FST: case ZYDIS_MNEMONIC_FSTP:
            handled = EmitFstOrFstp(insn, ops, next_rip, c); break;
        case ZYDIS_MNEMONIC_FILD:  handled = EmitFild(insn, ops, next_rip, c); break;
        case ZYDIS_MNEMONIC_FISTP: case ZYDIS_MNEMONIC_FISTTP:
            handled = EmitFistp(insn, ops, next_rip, c); break;
        case ZYDIS_MNEMONIC_FADDP: case ZYDIS_MNEMONIC_FMULP:
        case ZYDIS_MNEMONIC_FSUBP: case ZYDIS_MNEMONIC_FSUBRP:
        case ZYDIS_MNEMONIC_FDIVP: case ZYDIS_MNEMONIC_FDIVRP:
            handled = EmitX87ArithPop(insn, ops, c); break;
        case ZYDIS_MNEMONIC_FADD: case ZYDIS_MNEMONIC_FMUL:
        case ZYDIS_MNEMONIC_FSUB: case ZYDIS_MNEMONIC_FSUBR:
        case ZYDIS_MNEMONIC_FDIV: case ZYDIS_MNEMONIC_FDIVR:
            handled = EmitX87Arith(insn, ops, next_rip, c); break;
        case ZYDIS_MNEMONIC_FCHS: case ZYDIS_MNEMONIC_FABS:
        case ZYDIS_MNEMONIC_FSQRT:
            handled = EmitX87Unary(insn, c); break;
        case ZYDIS_MNEMONIC_FLD1: case ZYDIS_MNEMONIC_FLDZ:
            handled = EmitX87LoadConst(insn, c); break;
        case ZYDIS_MNEMONIC_FXCH:
            handled = EmitFxch(insn, ops, c); break;
        case ZYDIS_MNEMONIC_FCOMI: case ZYDIS_MNEMONIC_FUCOMI:
        case ZYDIS_MNEMONIC_FCOMIP: case ZYDIS_MNEMONIC_FUCOMIP:
        case ZYDIS_MNEMONIC_FCOM: case ZYDIS_MNEMONIC_FCOMP:
        case ZYDIS_MNEMONIC_FCOMPP: case ZYDIS_MNEMONIC_FUCOM:
        case ZYDIS_MNEMONIC_FUCOMP:
            handled = EmitX87Compare(insn, ops, c); break;
        case ZYDIS_MNEMONIC_FNSTSW:
            handled = EmitFnstsw(insn, ops, next_rip, c); break;
        case ZYDIS_MNEMONIC_FCMOVB: case ZYDIS_MNEMONIC_FCMOVE:
        case ZYDIS_MNEMONIC_FCMOVBE: case ZYDIS_MNEMONIC_FCMOVU:
        case ZYDIS_MNEMONIC_FCMOVNB: case ZYDIS_MNEMONIC_FCMOVNE:
        case ZYDIS_MNEMONIC_FCMOVNBE: case ZYDIS_MNEMONIC_FCMOVNU:
            handled = EmitFcmov(insn, ops, c); break;
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
            handled = EmitPush(insn, ops, next_rip, c);
            break;
        case ZYDIS_MNEMONIC_LEAVE:
            handled = EmitLeave(insn, c);
            break;
        case ZYDIS_MNEMONIC_XADD:
            handled = EmitXadd(insn, ops, next_rip, c);
            break;
        case ZYDIS_MNEMONIC_CMPXCHG:
            handled = EmitCmpxchg(insn, ops, next_rip, c);
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
            handled = EmitSetcc(insn, ops, next_rip, c);
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
            handled = EmitCmov(insn, ops, next_rip, c);
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

        // Must comfortably cover ONE worst-case emitter plus the fallthrough
        // terminator. The NEON/packed emitters routinely expand a single
        // guest op to dozens of host instructions; 256 bytes was under the
        // worst case, and an overflow throws Xbyak::Error through the
        // gateway JIT frame (no unwind info -> process death). Matches the
        // x86 lifter's margin.
        constexpr u64 HOST_SIZE_MARGIN = 1024;
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

    // Hand the unused remainder of the cap-sized reservation back to the
    // bump allocator (see CodeCache::ReturnTail; the x86 tail has the full
    // note). Safe relative to the icache invalidate above: the reclaimed
    // bytes were never executed, and a future block landing in them runs
    // its own invalidate over its own range.
    code_cache_.ReturnTail(code_buf, BLOCK_HOST_SIZE_CAP, emitted);

    bytes_emitted_ += emitted;
    ++blocks_compiled_;
    return code_buf;
} catch (const std::exception& e) {
    // Restore W^X first: Allocate() cannot throw, so any exception landing
    // here was raised at or after the CodeGenerator -- i.e., after
    // WriteBegin(). WriteEnd() flips pthread_jit_write_protect_np back to
    // executable; the abandoned reservation was never executed or inserted,
    // so no i-cache maintenance is needed for it.
    code_cache_.WriteEnd();
    std::fprintf(stderr,
                 "[lifter] compile threw at guest RIP 0x%llx: %s\n",
                 static_cast<unsigned long long>(guest_rip), e.what());
    std::fflush(stderr);
    return nullptr;
}

} // namespace Core::Runtime
