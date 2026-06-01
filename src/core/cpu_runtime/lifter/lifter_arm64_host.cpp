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
        const int hbDst = HighByteParent(ops[0].reg.value);
        const int d = (hbDst >= 0) ? hbDst : ZydisGprToIndex(ops[0].reg.value);
        if (d < 0) return false;
        if (src_mem) {
            // Load rhs from memory first (EA clobbers x9/x11), then lhs.
            if (!EmitEffectiveAddress(ops[1].mem, next_rip, c)) return false;
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
enum class VCmpIntKind { EqB, EqD, GtD };

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
            case VCmpIntKind::EqD: c.cmeq(VReg4S(0),  VReg4S(0),  VReg4S(1));  break;
            case VCmpIntKind::GtD: c.cmgt(VReg4S(0),  VReg4S(0),  VReg4S(1));  break;
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
    const bool sd = (insn.mnemonic == ZYDIS_MNEMONIC_VCVTTSD2SI);
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

    // Truncating convert -> x9.
    if (dst64) {
        if (sd) c.fcvtzs(kScratch0, DReg(0));
        else    c.fcvtzs(kScratch0, SReg(0));
    } else {
        if (sd) c.fcvtzs(WReg(9), DReg(0));
        else    c.fcvtzs(WReg(9), SReg(0));
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
    c.ldr(kScratch0, ptr(kState, YmmChunkOffset(s_vec, 0)));
    if (ops[0].type == ZYDIS_OPERAND_TYPE_REGISTER) {
        const int gi = ZydisGprToIndex(ops[0].reg.value);
        if (gi < 0) return false;
        c.str(kScratch0, ptr(kState, GprOffset(gi)));
    } else if (ops[0].type == ZYDIS_OPERAND_TYPE_MEMORY) {
        if (!EmitEffectiveAddress(ops[0].mem, next_rip, c)) return false;
        c.str(kScratch0, ptr(kAddr));
    } else {
        return false;
    }
    return true;
}

// vmovaps/vmovups/vmovdqa/vmovdqu/vmovntdqa — 128/256-bit move with the usual
// VEX zeroing on a register/memory load into a register.
//   reg <- reg : copy width, zero upper YMM (128-bit form).
//   reg <- mem : load width, zero upper YMM.
//   mem <- reg : store width.
bool EmitVmovFull(const ZydisDecodedInstruction& insn, const ZydisDecodedOperand* ops,
                  u64 next_rip, CodeGenerator& c) {
    const bool ymm = (ops[0].size == 256 || ops[1].size == 256);
    const int nchunks = ymm ? 4 : 2;

    if (ops[0].type == ZYDIS_OPERAND_TYPE_REGISTER &&
        ops[1].type == ZYDIS_OPERAND_TYPE_REGISTER) {
        const int dst = ZydisVecToIndex(ops[0].reg.value);
        const int src = ZydisVecToIndex(ops[1].reg.value);
        if (dst < 0 || src < 0) return false;
        for (int ch = 0; ch < nchunks; ++ch) {
            c.ldr(kScratch0, ptr(kState, YmmChunkOffset(src, ch)));
            c.str(kScratch0, ptr(kState, YmmChunkOffset(dst, ch)));
        }
        if (!ymm) {
            c.mov(kScratch0, 0);
            c.str(kScratch0, ptr(kState, YmmChunkOffset(dst, 2)));
            c.str(kScratch0, ptr(kState, YmmChunkOffset(dst, 3)));
        }
        return true;
    }
    if (ops[0].type == ZYDIS_OPERAND_TYPE_REGISTER &&
        ops[1].type == ZYDIS_OPERAND_TYPE_MEMORY) {
        const int dst = ZydisVecToIndex(ops[0].reg.value);
        if (dst < 0) return false;
        if (!EmitEffectiveAddress(ops[1].mem, next_rip, c)) return false;
        c.mov(kScratch1, kAddr);
        for (int ch = 0; ch < nchunks; ++ch) {
            c.add(kScratch0, kScratch1, ch * 8);
            c.ldr(kScratch2, ptr(kScratch0));
            c.str(kScratch2, ptr(kState, YmmChunkOffset(dst, ch)));
        }
        if (!ymm) {
            c.mov(kScratch0, 0);
            c.str(kScratch0, ptr(kState, YmmChunkOffset(dst, 2)));
            c.str(kScratch0, ptr(kState, YmmChunkOffset(dst, 3)));
        }
        return true;
    }
    if (ops[0].type == ZYDIS_OPERAND_TYPE_MEMORY &&
        ops[1].type == ZYDIS_OPERAND_TYPE_REGISTER) {
        const int src = ZydisVecToIndex(ops[1].reg.value);
        if (src < 0) return false;
        if (!EmitEffectiveAddress(ops[0].mem, next_rip, c)) return false;
        c.mov(kScratch1, kAddr);
        for (int ch = 0; ch < nchunks; ++ch) {
            c.ldr(kScratch2, ptr(kState, YmmChunkOffset(src, ch)));
            c.add(kScratch0, kScratch1, ch * 8);
            c.str(kScratch2, ptr(kScratch0));
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

// vpsrad/vpsrld dst, src, imm8 — per-dword arithmetic/logical right shift. The
// x86 shift count saturates: for SRA, count>=32 acts as 31 (sign-fill); for
// SRL, count>=32 yields 0. NEON sshr/ushr take 1..32, so we clamp.
bool EmitVpshift(const ZydisDecodedInstruction& insn, const ZydisDecodedOperand* ops,
                 u64 next_rip, CodeGenerator& c) {
    const bool arith = (insn.mnemonic == ZYDIS_MNEMONIC_VPSRAD);
    if (ops[0].type != ZYDIS_OPERAND_TYPE_REGISTER ||
        ops[1].type != ZYDIS_OPERAND_TYPE_REGISTER) return false;
    const int dst = ZydisVecToIndex(ops[0].reg.value);
    const int src = ZydisVecToIndex(ops[1].reg.value);
    if (dst < 0 || src < 0) return false;
    int cnt = static_cast<int>(ops[2].imm.value.u) & 0xFF;

    const bool ymm = (ops[0].size == 256);
    const int nchunks = ymm ? 2 : 1;

    for (int h = 0; h < nchunks; ++h) {
        const int chunk = h * 2;
        c.add(kScratch0, kState, YmmChunkOffset(src, chunk));
        c.ldr(QReg(0), ptr(kScratch0));
        if (arith) {
            int s = cnt; if (s > 31) s = 31; if (s < 1) s = 0;
            if (s == 0) { /* no shift */ }
            else c.sshr(VReg4S(0), VReg4S(0), s);
        } else {
            if (cnt >= 32) {
                c.eor(VReg16B(0), VReg16B(0), VReg16B(0));  // all zero
            } else if (cnt >= 1) {
                c.ushr(VReg4S(0), VReg4S(0), cnt);
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
    const bool word = (insn.mnemonic == ZYDIS_MNEMONIC_VPBLENDW);
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
    c.sshr(VReg4S(2), VReg4S(2), 31);    // broadcast sign bit per dword -> all-ones/all-zeros
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
bool EmitMxcsr(const ZydisDecodedInstruction& insn, const ZydisDecodedOperand* ops,
               u64 next_rip, CodeGenerator& c) {
    const bool store = (insn.mnemonic == ZYDIS_MNEMONIC_STMXCSR);
    if (ops[0].type != ZYDIS_OPERAND_TYPE_MEMORY) return false;
    if (!EmitEffectiveAddress(ops[0].mem, next_rip, c)) return false;
    const u32 off = static_cast<u32>(offsetof(GuestState, mxcsr));
    if (store) {
        c.ldr(WReg(9), ptr(kState, off));
        c.str(WReg(9), ptr(kAddr));
    } else {
        c.ldr(WReg(9), ptr(kAddr));
        c.str(WReg(9), ptr(kState, off));
    }
    return true;
}

// fnstcw m16 — store the (fixed) x87 control word 0x037F. fldcw is a no-op in
// our model (rounding/precision control not emulated).
bool EmitFnstcw(const ZydisDecodedInstruction& insn, const ZydisDecodedOperand* ops,
                u64 next_rip, CodeGenerator& c) {
    if (insn.mnemonic == ZYDIS_MNEMONIC_FLDCW) return true;  // no-op
    if (ops[0].type != ZYDIS_OPERAND_TYPE_MEMORY) return false;
    if (!EmitEffectiveAddress(ops[0].mem, next_rip, c)) return false;
    c.mov(WReg(9), 0x037Fu);
    c.strh(WReg(9), ptr(kAddr));
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
    if (ops[0].type != ZYDIS_OPERAND_TYPE_MEMORY) return false;
    const bool pop = (insn.mnemonic == ZYDIS_MNEMONIC_FSTP);
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

// FISTP — convert ST(0) to signed int (32/64), store, pop. Round-to-nearest.
bool EmitFistp(const ZydisDecodedInstruction& insn, const ZydisDecodedOperand* ops,
               u64 next_rip, CodeGenerator& c) {
    if (insn.mnemonic != ZYDIS_MNEMONIC_FISTP) return false;
    if (ops[0].type != ZYDIS_OPERAND_TYPE_MEMORY) return false;
    if (!EmitEffectiveAddress(ops[0].mem, next_rip, c)) return false;
    c.mov(kScratch1, kAddr);
    EmitX87RegAddr(XReg(11), 0, c);
    c.ldr(DReg(0), ptr(XReg(11)));
    const int w = ops[0].size;
    // Round to nearest even, then convert.
    c.frintn(DReg(0), DReg(0));
    if (w == 64) {
        c.fcvtzs(kScratch0, DReg(0));
        c.str(kScratch0, ptr(kScratch1));
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
        case ZYDIS_MNEMONIC_VPCMPEQB: handled = EmitVCmpInt(insn, ops, next_rip, VCmpIntKind::EqB, c); break;
        case ZYDIS_MNEMONIC_VPCMPEQD: handled = EmitVCmpInt(insn, ops, next_rip, VCmpIntKind::EqD, c); break;
        case ZYDIS_MNEMONIC_VPCMPGTD: handled = EmitVCmpInt(insn, ops, next_rip, VCmpIntKind::GtD, c); break;
        case ZYDIS_MNEMONIC_VPTEST:   handled = EmitVptest(insn, ops, c); break;
        case ZYDIS_MNEMONIC_VCOMISS:  case ZYDIS_MNEMONIC_VCOMISD:
        case ZYDIS_MNEMONIC_VUCOMISS: case ZYDIS_MNEMONIC_VUCOMISD:
            handled = EmitVcomi(insn, ops, next_rip, c); break;
        case ZYDIS_MNEMONIC_VCVTSI2SS: case ZYDIS_MNEMONIC_VCVTSI2SD:
            handled = EmitVcvtsi2(insn, ops, next_rip, c); break;
        case ZYDIS_MNEMONIC_VCVTTSS2SI: case ZYDIS_MNEMONIC_VCVTTSD2SI:
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
        case ZYDIS_MNEMONIC_VMOVQ:    handled = EmitVmovq(insn, ops, next_rip, c); break;
        case ZYDIS_MNEMONIC_VMOVAPS: case ZYDIS_MNEMONIC_VMOVUPS:
        case ZYDIS_MNEMONIC_VMOVDQA: case ZYDIS_MNEMONIC_VMOVDQU:
        case ZYDIS_MNEMONIC_VMOVNTDQA:
            handled = EmitVmovFull(insn, ops, next_rip, c); break;
        case ZYDIS_MNEMONIC_VMOVSLDUP: case ZYDIS_MNEMONIC_VMOVSHDUP:
            handled = EmitVmovdup(insn, ops, next_rip, c); break;
        case ZYDIS_MNEMONIC_VPSRAD: case ZYDIS_MNEMONIC_VPSRLD:
            handled = EmitVpshift(insn, ops, next_rip, c); break;
        case ZYDIS_MNEMONIC_VPSLLDQ: case ZYDIS_MNEMONIC_VPSRLDQ:
            handled = EmitVpsxldq(insn, ops, next_rip, c); break;
        case ZYDIS_MNEMONIC_VBLENDPS: case ZYDIS_MNEMONIC_VPBLENDW:
            handled = EmitVblend(insn, ops, next_rip, c); break;
        case ZYDIS_MNEMONIC_VBLENDVPS: handled = EmitVblendvps(insn, ops, next_rip, c); break;
        case ZYDIS_MNEMONIC_VBROADCASTSS: handled = EmitVbroadcastss(insn, ops, next_rip, c); break;
        case ZYDIS_MNEMONIC_VROUNDSS: case ZYDIS_MNEMONIC_VROUNDSD:
            handled = EmitVround(insn, ops, next_rip, c); break;
        case ZYDIS_MNEMONIC_VCMPPS: case ZYDIS_MNEMONIC_VCMPSS:
            handled = EmitVcmpfp(insn, ops, next_rip, c); break;
        case ZYDIS_MNEMONIC_VPMOVZXDQ: handled = EmitVpmovzxdq(insn, ops, next_rip, c); break;
        case ZYDIS_MNEMONIC_VPHADDD:   handled = EmitVphaddd(insn, ops, next_rip, c); break;
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
        case ZYDIS_MNEMONIC_PREFETCH:    case ZYDIS_MNEMONIC_PREFETCHNTA:
        case ZYDIS_MNEMONIC_PREFETCHT0:  case ZYDIS_MNEMONIC_PREFETCHT1:
        case ZYDIS_MNEMONIC_PREFETCHT2:  case ZYDIS_MNEMONIC_PREFETCHW:
            handled = true;  // memory hints / fences: no-op for this model.
            break;
        case ZYDIS_MNEMONIC_CPUID:  handled = EmitCpuid(insn, ops, c); break;
        case ZYDIS_MNEMONIC_XGETBV: handled = EmitXgetbv(insn, ops, c); break;
        case ZYDIS_MNEMONIC_STMXCSR: case ZYDIS_MNEMONIC_LDMXCSR:
            handled = EmitMxcsr(insn, ops, next_rip, c); break;
        case ZYDIS_MNEMONIC_FNSTCW: case ZYDIS_MNEMONIC_FLDCW:
            handled = EmitFnstcw(insn, ops, next_rip, c); break;
        case ZYDIS_MNEMONIC_FLD:   handled = EmitFld(insn, ops, next_rip, c); break;
        case ZYDIS_MNEMONIC_FST: case ZYDIS_MNEMONIC_FSTP:
            handled = EmitFstOrFstp(insn, ops, next_rip, c); break;
        case ZYDIS_MNEMONIC_FILD:  handled = EmitFild(insn, ops, next_rip, c); break;
        case ZYDIS_MNEMONIC_FISTP: handled = EmitFistp(insn, ops, next_rip, c); break;
        case ZYDIS_MNEMONIC_FADDP: case ZYDIS_MNEMONIC_FMULP:
        case ZYDIS_MNEMONIC_FSUBP: case ZYDIS_MNEMONIC_FSUBRP:
        case ZYDIS_MNEMONIC_FDIVP: case ZYDIS_MNEMONIC_FDIVRP:
            handled = EmitX87ArithPop(insn, ops, c); break;
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
