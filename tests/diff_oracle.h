// SPDX-FileCopyrightText: Copyright 2026 shadPS4 Emulator Project
// SPDX-License-Identifier: GPL-2.0-or-later
//
// Differential oracle: run the same guest instruction sequence two ways and
// compare the resulting architectural state.
//
//   NATIVE  -- the snippet is executed directly on the host CPU (this is the
//              "normal" shadPS4 model: PS4 guest and host are both x86-64, so
//              normal mode runs guest code natively). This is the reference.
//   JIT     -- the snippet is lifted and run through Core::Runtime.
//
// Comparing the two pins down lifter bugs the synthetic unit tests miss
// (wrong values, scratch-register clobbering, flag-precision, bad addressing).
//
// SCOPE / what a snippet may contain (both runs must agree, so we restrict to
// what is faithfully reproducible by inline native execution):
//   * straight-line only: NO control flow (jmp/jcc/call/ret/loop). Block
//     terminus is supplied by the harness.
//   * NO fs:/gs: overrides: native uses the host's real fs/gs, the JIT uses the
//     guest fs_base/gs_base -- they cannot match. (Covered by the dedicated
//     TLS tests instead.)
//   * NO rip-relative memory: native runs the bytes inline at the stub's
//     address, so its rip differs from the JIT's guest rip.
//   * memory via [reg(+index*scale)+disp] and push/pop ARE fine: guest memory
//     is identity-mapped host memory, so both runs touch the same bytes.
//
// Single-threaded (the native stub uses thread_local scratch slots).

#pragma once

#include <array>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <string>
#include <vector>

#include <xbyak/xbyak.h>

#include "common/types.h"
#include "core/cpu_runtime/guest_state.h"
#include "core/cpu_runtime/runtime.h"

namespace diff {

using GuestState = Core::Runtime::GuestState;
using Runtime = Core::Runtime::Runtime;

// The five status flags the lifter models (CF|PF|ZF|SF|OF). AF is deliberately
// NOT modeled by the JIT, so it is excluded from comparison by default.
constexpr u64 CF = 1ULL << 0;
constexpr u64 PF = 1ULL << 2;
constexpr u64 ZF = 1ULL << 6;
constexpr u64 SF = 1ULL << 7;
constexpr u64 OF = 1ULL << 11;
constexpr u64 kStatusMask = CF | PF | ZF | SF | OF; // 0x8C5

struct Context {
    std::array<u64, 16> gpr{}; // RAX,RCX,RDX,RBX,RSP,RBP,RSI,RDI,R8..R15
    u64 rflags{};              // only kStatusMask bits are meaningful
};

// ----------------------------------------------------------------------------
// NATIVE reference: execute `snippet` (len bytes) on the host CPU with the
// given input context, capturing the resulting GPRs and status flags.
// ----------------------------------------------------------------------------
inline void RunNative(const u8* snippet, size_t len, const Context& in, Context& out) {
    // Per-thread scratch (avoids the chicken-and-egg of capturing 16 live GPRs
    // without a free base register: RAX is spilled via the rax-only absolute
    // moffs store, freeing it to address the rest).
    static thread_local u64 g_host_rsp = 0;
    static thread_local u64 g_rax = 0;
    static thread_local u64 g_flags = 0; // low 16 bits: ah=lahf, al=seto(OF)

    struct Stub : Xbyak::CodeGenerator {
        Stub(const u8* sn, size_t n, const Context& cin,
             u64* host_rsp, u64* rax_slot, u64* flags_slot, u64* out_gpr) {
            using namespace Xbyak::util;

            // --- prologue: preserve host non-volatiles + host rsp -----------
            push(rbp); push(rbx); push(r12); push(r13); push(r14); push(r15);
            push(rsi); push(rdi);
            mov(rax, rsp);
            db(0x48); db(0xA3); dq(reinterpret_cast<u64>(host_rsp)); // mov [abs64], rax

            // Seed status flags on the HOST stack, before switching rsp. Only
            // the settable status bits (+ reserved bit 1); never TF/IF/etc.
            mov(rax, (cin.rflags & kStatusMask) | 0x2);
            push(rax); popfq();

            // --- enter guest context ---------------------------------------
            mov(rsp, cin.gpr[4]);                 // guest stack (caller must map it)
            // mov does not disturb flags, so the seeded flags survive these.
            mov(rcx, cin.gpr[1]);  mov(rdx, cin.gpr[2]);  mov(rbx, cin.gpr[3]);
            mov(rbp, cin.gpr[5]);  mov(rsi, cin.gpr[6]);  mov(rdi, cin.gpr[7]);
            mov(r8,  cin.gpr[8]);  mov(r9,  cin.gpr[9]);  mov(r10, cin.gpr[10]);
            mov(r11, cin.gpr[11]); mov(r12, cin.gpr[12]); mov(r13, cin.gpr[13]);
            mov(r14, cin.gpr[14]); mov(r15, cin.gpr[15]);
            mov(rax, cin.gpr[0]);                 // rax last

            // --- the snippet, inline ---------------------------------------
            for (size_t i = 0; i < n; ++i) db(sn[i]);

            // --- capture (snippet results are live in every GPR + flags) ----
            db(0x48); db(0xA3); dq(reinterpret_cast<u64>(rax_slot)); // [abs]=rax (spill rax)
            lahf();                                // ah = SF ZF * AF * PF * CF
            db(0x0F); db(0x90); db(0xC0);          // seto al  -> al = OF
            db(0x48); db(0xA3); dq(reinterpret_cast<u64>(flags_slot)); // [abs]=rax (flags)
            mov(rax, reinterpret_cast<u64>(out_gpr)); // rax now free -> base for the rest
            mov(ptr[rax + 8 * 1], rcx);  mov(ptr[rax + 8 * 2], rdx);
            mov(ptr[rax + 8 * 3], rbx);  mov(ptr[rax + 8 * 4], rsp);
            mov(ptr[rax + 8 * 5], rbp);  mov(ptr[rax + 8 * 6], rsi);
            mov(ptr[rax + 8 * 7], rdi);  mov(ptr[rax + 8 * 8], r8);
            mov(ptr[rax + 8 * 9], r9);   mov(ptr[rax + 8 * 10], r10);
            mov(ptr[rax + 8 * 11], r11); mov(ptr[rax + 8 * 12], r12);
            mov(ptr[rax + 8 * 13], r13); mov(ptr[rax + 8 * 14], r14);
            mov(ptr[rax + 8 * 15], r15);

            // --- epilogue: restore host rsp + non-volatiles -----------------
            db(0x48); db(0xA1); dq(reinterpret_cast<u64>(host_rsp)); // mov rax,[abs64]
            mov(rsp, rax);
            pop(rdi); pop(rsi); pop(r15); pop(r14); pop(r13); pop(r12);
            pop(rbx); pop(rbp);
            ret();
        }
    };

    out = Context{};
    Stub stub(snippet, len, in, &g_host_rsp, &g_rax, &g_flags, out.gpr.data());
    stub.ready();
    stub.getCode<void (*)()>()();

    out.gpr[0] = g_rax;
    const u8 ah = static_cast<u8>(g_flags >> 8);
    const u8 al = static_cast<u8>(g_flags & 0xFF);
    u64 fl = 0;
    if (ah & 0x01) fl |= CF;
    if (ah & 0x04) fl |= PF;
    if (ah & 0x40) fl |= ZF;
    if (ah & 0x80) fl |= SF;
    if (al & 0x01) fl |= OF;
    out.rflags = fl;
}

// ----------------------------------------------------------------------------
// JIT: lift+run `snippet` via Core::Runtime. `code_buf`/`code_addr` is a guest
// code region (identity-mapped). A BSR terminator ends the block WITHOUT
// perturbing GPRs/flags/RSP (unlike `ret`, which pops RSP), so the captured
// state is exactly the post-snippet state.
// ----------------------------------------------------------------------------
struct JitOutcome {
    Context out;
    u32 exit_reason{};
    u64 rip{};
};

inline JitOutcome RunJit(Runtime& rt, u8* code_buf, u64 code_addr,
                         const u8* snippet, size_t len, const Context& in) {
    static const u8 kBsr[] = {0x48, 0x0f, 0xbd, 0xd8}; // bsr rbx, rax (unsupported)
    std::vector<u8> prog(snippet, snippet + len);
    prog.insert(prog.end(), kBsr, kBsr + sizeof(kBsr));
    std::memcpy(code_buf, prog.data(), prog.size());

    GuestState st{};
    st.gpr = in.gpr;
    st.rip = code_addr;
    st.rflags = (in.rflags & kStatusMask) | 0x2;
    rt.Run(st);

    JitOutcome o;
    o.out.gpr = st.gpr;
    o.out.rflags = st.rflags & kStatusMask;
    o.exit_reason = st.exit_reason;
    o.rip = st.rip;
    return o;
}

// ----------------------------------------------------------------------------
// Compare. Returns an empty string on match, else a human-readable report.
// gpr_mask: bit i selects gpr[i] for comparison (default: all 16).
// flag_mask: which status flags to compare (default: all five the JIT models;
//            narrow it per-op for instructions that leave some flags undefined).
// ----------------------------------------------------------------------------
inline std::string Compare(const Context& jit, const Context& nat,
                           u16 gpr_mask = 0xFFFF, u64 flag_mask = kStatusMask) {
    static const char* kNames[16] = {"rax", "rcx", "rdx", "rbx", "rsp", "rbp",
                                      "rsi", "rdi", "r8",  "r9",  "r10", "r11",
                                      "r12", "r13", "r14", "r15"};
    std::string r;
    char line[128];
    for (int i = 0; i < 16; ++i) {
        if (!(gpr_mask & (1u << i))) continue;
        if (jit.gpr[i] != nat.gpr[i]) {
            std::snprintf(line, sizeof(line),
                          "  %-3s  jit=0x%016llx  native=0x%016llx\n", kNames[i],
                          (unsigned long long)jit.gpr[i],
                          (unsigned long long)nat.gpr[i]);
            r += line;
        }
    }
    if ((jit.rflags & flag_mask) != (nat.rflags & flag_mask)) {
        std::snprintf(line, sizeof(line),
                      "  flags jit=0x%03llx native=0x%03llx (mask=0x%03llx)\n",
                      (unsigned long long)(jit.rflags & flag_mask),
                      (unsigned long long)(nat.rflags & flag_mask),
                      (unsigned long long)flag_mask);
        r += line;
    }
    return r;
}

} // namespace diff
