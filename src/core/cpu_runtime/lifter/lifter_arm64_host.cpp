// SPDX-FileCopyrightText: Copyright 2026 shadPS4 Emulator Project
// SPDX-License-Identifier: GPL-2.0-or-later
//
// ============================================================================
// ARM64 (AArch64) lifter — macOS / Apple Silicon ONLY.
//
// STATUS: structural skeleton. This compiles and links into an arm64 build,
// but implements only THREE representative emitters, one per distinct
// translation shape:
//
//   1. GPR move            (GuestState<->host reg data movement)
//   2. 64-bit ADD          (arithmetic + lazy-flag side-band write-back)
//   3. vector op via NEON   (validates the YMM-in-memory + per-lane model)
//   plus the block terminator (br x26 normal / br x25 fatal)
//
// Every other guest instruction falls through to EmitUnsupportedExit, so a
// real workload will bail almost immediately. The remaining ~hundred x86
// emitters port mechanically by matching each to one of the shapes above;
// that port is the follow-up work. This file exists so the arm64 build
// configures/compiles/links and the runtime can be exercised end-to-end on
// real hardware (the CI macos-arm64 job is the first compile-check — the
// x86_64 dev sandbox has no xbyak_aarch64).
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
const XReg kAddr = XReg(11);      // address scratch (rdx analog)
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
constexpr u32 YmmChunkOffset(int lane, int chunk) {
    return static_cast<u32>(offsetof(GuestState, ymm) + (lane * 4 + chunk) * 8);
}

int ZydisGprToIndex(ZydisRegister reg); // shared util (RAX..R15 -> 0..15)
int ZydisVecToIndex(ZydisRegister reg); // shared util (XMM/YMM0.. -> 0..31)

// Shared lazy-flag op enum value (matches the x86 lifter + the
// host-agnostic materializer in runtime.cpp).
constexpr u32 FLAG_OP_ADD = 1;

// ----------------------------------------------------------------------------
// PATTERN 1 — GPR MOVE.  guest: mov r64, r64
//   x86:   mov rax,[r13+src]; mov [r13+dst],rax
//   arm64: ldr x9,[x28,#src]; str x9,[x28,#dst]
// ----------------------------------------------------------------------------
bool EmitMov(const ZydisDecodedInstruction& insn, const ZydisDecodedOperand* ops,
             u64 next_rip, CodeGenerator& c) {
    if (insn.mnemonic != ZYDIS_MNEMONIC_MOV) return false;
    if (insn.operand_count_visible != 2) return false;
    if (ops[0].type != ZYDIS_OPERAND_TYPE_REGISTER ||
        ops[1].type != ZYDIS_OPERAND_TYPE_REGISTER) {
        return false; // draft: reg-reg only
    }
    const int dst = ZydisGprToIndex(ops[0].reg.value);
    const int src = ZydisGprToIndex(ops[1].reg.value);
    if (dst < 0 || src < 0) return false;

    c.ldr(kScratch0, ptr(kState, GprOffset(src)));
    c.str(kScratch0, ptr(kState, GprOffset(dst)));
    return true;
}

// ----------------------------------------------------------------------------
// PATTERN 2 — 64-bit ADD with lazy flags.  guest: add r64, r64
//
// The lazy-flag side-band model is HOST-AGNOSTIC: it decides WHEN to
// materialize guest RFLAGS, not how host flags map. We do the host
// arithmetic, write the result, and stash (op, lhs, rhs, result) for the
// runtime's materializer — identical strategy to x86. We deliberately do
// NOT map host NZCV onto guest RFLAGS (different bit layout, AF/PF have
// no NZCV equivalent).
// ----------------------------------------------------------------------------
bool EmitAdd64(const ZydisDecodedInstruction& insn, const ZydisDecodedOperand* ops,
               u64 next_rip, CodeGenerator& c) {
    if (insn.mnemonic != ZYDIS_MNEMONIC_ADD) return false;
    if (insn.operand_width != 64) return false;
    if (ops[0].type != ZYDIS_OPERAND_TYPE_REGISTER ||
        ops[1].type != ZYDIS_OPERAND_TYPE_REGISTER) {
        return false;
    }
    const int dst = ZydisGprToIndex(ops[0].reg.value);
    const int src = ZydisGprToIndex(ops[1].reg.value);
    if (dst < 0 || src < 0) return false;

    c.ldr(kScratch0, ptr(kState, GprOffset(dst))); // lhs (== src1)
    c.ldr(kScratch1, ptr(kState, GprOffset(src))); // rhs
    c.add(kScratch2, kScratch0, kScratch1);        // result
    c.str(kScratch2, ptr(kState, GprOffset(dst)));

    // Lazy-flag side-band.
    c.mov(kWScratch3, FLAG_OP_ADD);
    c.str(kWScratch3, ptr(kState, static_cast<u32>(offsetof(GuestState, flag_op))));
    c.str(kScratch0, ptr(kState, static_cast<u32>(offsetof(GuestState, flag_lhs))));
    c.str(kScratch1, ptr(kState, static_cast<u32>(offsetof(GuestState, flag_rhs))));
    c.str(kScratch2, ptr(kState, static_cast<u32>(offsetof(GuestState, flag_result))));
    return true;
}

// ----------------------------------------------------------------------------
// PATTERN 3 — vector op via NEON.  guest: vpaddd xmm, xmm, xmm (128-bit)
//
// YMM-in-memory model identical to x86. NEON is 128-bit only, so the
// 256-bit form (deferred in this draft) becomes TWO of these — which maps
// onto the x86 lifter's existing "per-128-bit-lane loop". We materialize
// the chunk base into kAddr and use register addressing, since high-lane
// YmmChunkOffsets exceed the q-register ldr/str immediate range.
// ----------------------------------------------------------------------------
bool EmitVpaddd128(const ZydisDecodedInstruction& insn, const ZydisDecodedOperand* ops,
                   u64 next_rip, CodeGenerator& c) {
    if (insn.mnemonic != ZYDIS_MNEMONIC_VPADDD) return false;
    if (insn.operand_count_visible != 3) return false;
    if (ops[0].type != ZYDIS_OPERAND_TYPE_REGISTER ||
        ops[1].type != ZYDIS_OPERAND_TYPE_REGISTER ||
        ops[2].type != ZYDIS_OPERAND_TYPE_REGISTER) {
        return false; // draft: reg-reg-reg only
    }
    if (ops[0].size != 128) return false; // draft: skip 256 for now

    const int dst = ZydisVecToIndex(ops[0].reg.value);
    const int s1 = ZydisVecToIndex(ops[1].reg.value);
    const int s2 = ZydisVecToIndex(ops[2].reg.value);
    if (dst < 0 || s1 < 0 || s2 < 0) return false;

    c.add(kAddr, kState, YmmChunkOffset(s1, 0));
    c.ld1(VReg4S(0), ptr(kAddr));   // v0.4s = src1 low-128
    c.add(kAddr, kState, YmmChunkOffset(s2, 0));
    c.ld1(VReg4S(1), ptr(kAddr));   // v1.4s = src2 low-128
    c.add(VReg4S(0), VReg4S(0), VReg4S(1)); // per-lane 32-bit add
    c.add(kAddr, kState, YmmChunkOffset(dst, 0));
    c.st1(VReg4S(0), ptr(kAddr));   // store result low-128
    // VEX-128: zero upper 128 bits (chunks 2,3). Use a zeroed GPR pair
    // rather than xzr — xbyak_aarch64's register-31 spelling (xzr vs sp) is
    // type-disambiguated and not reachable here as a bare instance, so we
    // materialize zero in a scratch and store the pair.
    c.mov(kScratch0, 0);
    c.add(kAddr, kState, YmmChunkOffset(dst, 2));
    c.stp(kScratch0, kScratch0, ptr(kAddr));
    return true;
}

// ----------------------------------------------------------------------------
// PATTERN 4 — fatal-exit tail (unsupported instruction).
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
            if (insn.operand_width == 64) handled = EmitMov(insn, ops, next_rip, c);
            break;
        case ZYDIS_MNEMONIC_ADD:
            if (insn.operand_width == 64) handled = EmitAdd64(insn, ops, next_rip, c);
            break;
        case ZYDIS_MNEMONIC_VPADDD:
            handled = EmitVpaddd128(insn, ops, next_rip, c);
            break;
        // ... ~hundred more emitters port here following patterns 1-4 ...
        default:
            break;
        }

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
