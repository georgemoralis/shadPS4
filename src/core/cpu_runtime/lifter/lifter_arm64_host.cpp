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
// both with the lazy-flag side-band, and RET — which make the simplest runtime
// tests (MovAddRet, MultipleRegisters, SubImm) executable end-to-end.
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

// guest: mov r64, r64  /  mov r64, imm
//   reg<-reg  ldr x9,[x28,#src]; str x9,[x28,#dst]
//   reg<-imm  mov x9,#imm (movz/movk); str x9,[x28,#dst]
bool EmitMov(const ZydisDecodedInstruction& insn, const ZydisDecodedOperand* ops,
             CodeGenerator& c) {
    if (insn.mnemonic != ZYDIS_MNEMONIC_MOV) return false;
    if (insn.operand_width != 64) return false;
    if (insn.operand_count_visible != 2) return false;
    if (ops[0].type != ZYDIS_OPERAND_TYPE_REGISTER) return false;
    const int dst = ZydisGprToIndex(ops[0].reg.value);
    if (dst < 0) return false;

    if (ops[1].type == ZYDIS_OPERAND_TYPE_REGISTER) {
        const int src = ZydisGprToIndex(ops[1].reg.value);
        if (src < 0) return false;
        c.ldr(kScratch0, ptr(kState, GprOffset(src)));
        c.str(kScratch0, ptr(kState, GprOffset(dst)));
        return true;
    }
    if (ops[1].type == ZYDIS_OPERAND_TYPE_IMMEDIATE) {
        // Both x86 encodings (B8+r imm64, C7 /0 imm32-sign-extended) arrive
        // pre-resolved in imm.value.s. mov(XReg,u64) materializes any 64-bit
        // constant (movz/movk) — same call the terminator uses for the RIP.
        c.mov(kScratch0, static_cast<u64>(ops[1].imm.value.s));
        c.str(kScratch0, ptr(kState, GprOffset(dst)));
        return true;
    }
    return false; // mem forms deferred
}

// guest: add r64, r64  /  add r64, imm   (with lazy-flag side-band)
//   load lhs(=dst) and rhs, add, store result, then stash op/lhs/rhs/result.
bool EmitAdd64(const ZydisDecodedInstruction& insn, const ZydisDecodedOperand* ops,
               CodeGenerator& c) {
    if (insn.mnemonic != ZYDIS_MNEMONIC_ADD) return false;
    if (insn.operand_width != 64) return false;
    if (insn.operand_count_visible != 2) return false;
    if (ops[0].type != ZYDIS_OPERAND_TYPE_REGISTER) return false;
    const int dst = ZydisGprToIndex(ops[0].reg.value);
    if (dst < 0) return false;

    c.ldr(kScratch0, ptr(kState, GprOffset(dst))); // lhs (== dst current)
    if (ops[1].type == ZYDIS_OPERAND_TYPE_REGISTER) {
        const int src = ZydisGprToIndex(ops[1].reg.value);
        if (src < 0) return false;
        c.ldr(kScratch1, ptr(kState, GprOffset(src))); // rhs
    } else if (ops[1].type == ZYDIS_OPERAND_TYPE_IMMEDIATE) {
        c.mov(kScratch1, static_cast<u64>(ops[1].imm.value.s)); // rhs (sign-ext imm)
    } else {
        return false; // mem forms deferred
    }
    c.add(kScratch2, kScratch0, kScratch1);        // result
    c.str(kScratch2, ptr(kState, GprOffset(dst)));

    // Lazy-flag side-band (host-agnostic; runtime materializes RFLAGS).
    c.mov(kWScratch3, FLAG_OP_ADD);
    c.str(kWScratch3, ptr(kState, static_cast<u32>(offsetof(GuestState, flag_op))));
    c.str(kScratch0, ptr(kState, static_cast<u32>(offsetof(GuestState, flag_lhs))));
    c.str(kScratch1, ptr(kState, static_cast<u32>(offsetof(GuestState, flag_rhs))));
    c.str(kScratch2, ptr(kState, static_cast<u32>(offsetof(GuestState, flag_result))));
    return true;
}

// guest: sub r64, r64  /  sub r64, imm   (with lazy-flag side-band)
//   identical shape to EmitAdd64; result = lhs - rhs, op tag = FLAG_OP_SUB.
// NOTE: like ADD, the side-band (op/lhs/rhs/result) is written for a future
// runtime materializer; flags are NOT yet derived on arm64 (see header note).
// Tests that only check the GPR result (e.g. SubImm_DecrementsCorrectly) pass;
// flag-checking SUB/CMP tests await the materializer.
bool EmitSub64(const ZydisDecodedInstruction& insn, const ZydisDecodedOperand* ops,
               CodeGenerator& c) {
    if (insn.mnemonic != ZYDIS_MNEMONIC_SUB) return false;
    if (insn.operand_width != 64) return false;
    if (insn.operand_count_visible != 2) return false;
    if (ops[0].type != ZYDIS_OPERAND_TYPE_REGISTER) return false;
    const int dst = ZydisGprToIndex(ops[0].reg.value);
    if (dst < 0) return false;

    c.ldr(kScratch0, ptr(kState, GprOffset(dst))); // lhs (== dst current)
    if (ops[1].type == ZYDIS_OPERAND_TYPE_REGISTER) {
        const int src = ZydisGprToIndex(ops[1].reg.value);
        if (src < 0) return false;
        c.ldr(kScratch1, ptr(kState, GprOffset(src))); // rhs
    } else if (ops[1].type == ZYDIS_OPERAND_TYPE_IMMEDIATE) {
        c.mov(kScratch1, static_cast<u64>(ops[1].imm.value.s)); // rhs (sign-ext imm)
    } else {
        return false; // mem forms deferred
    }
    c.sub(kScratch2, kScratch0, kScratch1);        // result = lhs - rhs
    c.str(kScratch2, ptr(kState, GprOffset(dst)));

    // Lazy-flag side-band (host-agnostic; runtime materializes RFLAGS).
    c.mov(kWScratch3, FLAG_OP_SUB);
    c.str(kWScratch3, ptr(kState, static_cast<u32>(offsetof(GuestState, flag_op))));
    c.str(kScratch0, ptr(kState, static_cast<u32>(offsetof(GuestState, flag_lhs))));
    c.str(kScratch1, ptr(kState, static_cast<u32>(offsetof(GuestState, flag_rhs))));
    c.str(kScratch2, ptr(kState, static_cast<u32>(offsetof(GuestState, flag_result))));
    return true;
}
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
        (void)next_rip; // unused until mem-operand emitters (need RIP-relative)
        bool handled = false;
        switch (insn.mnemonic) {
        case ZYDIS_MNEMONIC_MOV:
            handled = EmitMov(insn, ops, c);
            break;
        case ZYDIS_MNEMONIC_ADD:
            handled = EmitAdd64(insn, ops, c);
            break;
        case ZYDIS_MNEMONIC_SUB:
            handled = EmitSub64(insn, ops, c);
            break;
        case ZYDIS_MNEMONIC_RET:
            handled = EmitRet(insn, c);
            if (handled) emitted_terminator = true; // RET is a block terminator
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
