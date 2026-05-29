// SPDX-FileCopyrightText: Copyright 2026 shadPS4 Emulator Project
// SPDX-License-Identifier: GPL-2.0-or-later
//
// ============================================================================
// ARM64 (AArch64) gateway — macOS / Apple Silicon ONLY.
//
// Scope is deliberately narrow: the only ARM64 host we target is macOS on
// Apple Silicon. That removes all multi-platform branching:
//   - No Windows-on-ARM unwind-info registration.
//   - No Linux/aarch64 PROT_EXEC-without-MAP_JIT path.
// What it ADDS is the Apple-specific W^X (MAP_JIT + per-thread
// pthread_jit_write_protect_np) discipline, which the x86 host never
// needed.
//
// This is a complete gateway implementation (prologue/epilogue, register
// pinning, dispatch loop, W^X + icache discipline). It has not yet been
// compile-checked on real hardware — the x86_64 dev sandbox has no
// xbyak_aarch64 — so any xbyak_aarch64 API/signature drift will surface in
// the CI macos-arm64 build, which is the first real compile-check.
// ============================================================================

#include "core/cpu_runtime/gateway/gateway.h"

#include <xbyak_aarch64/xbyak_aarch64.h>

#include <pthread.h>                  // pthread_jit_write_protect_np
#include <sys/mman.h>                 // mmap, MAP_JIT
#include <libkern/OSCacheControl.h>   // sys_icache_invalidate

#include "common/assert.h"
#include "common/logging/log.h"
#include "core/cpu_runtime/guest_state.h"

#ifndef MAP_JIT
#define MAP_JIT 0x800
#endif

namespace Core::Runtime {

namespace {

constexpr u64 GATEWAY_SIZE = 4096;

// ============================================================================
// REGISTER PINNING (mirror of the x86 r12/r13/r14/r15 scheme).
//
//   x28 = GuestState*           (mirror of x86 r13)
//   x27 = DispatcherFn pointer  (mirror of x86 r12)
//   x26 = dispatch_loop_top     (mirror of x86 r14; `br x26` = normal exit)
//   x25 = exit_stub             (mirror of x86 r15; `br x25` = fatal exit)
//
// x25..x28 are callee-saved (AAPCS64 x19..x28), so they survive the
// dispatcher call without per-call spilling. We avoid x16/x17 (veneer
// scratch) and x18 (platform register; reserved on Darwin).
//
// CALLEE-SAVED PRESERVATION POLICY:
// The gateway is called from C++ and must honor AAPCS64 — it must leave every
// callee-saved register as it found it. Crucially, the lifted JIT blocks run
// "inside" this gateway frame (the dispatch loop branches into them and they
// branch back), so ANY callee-saved register a lifted block clobbers would
// leak out to the C++ caller unless the gateway saved it. Rather than impose a
// "lifter may only touch caller-saved regs" contract on every future emitter
// (easy to violate silently), the gateway saves the FULL callee-saved set up
// front, exactly like the x86 gateway pushes rbx/rbp/r12-r15:
//   GPRs:    x19..x28 (10) + x29(FP) + x30(LR)      = 12 regs, 6 STP pairs
//   FP/SIMD: d8..d15 (low 64 bits of v8..v15 only)  =  8 regs, 4 STP pairs
// Total 160 bytes, 16-byte aligned (a hard AArch64 SP requirement). This makes
// the gateway correct for any emitter the lifter port adds, with no per-block
// spill cost (saved once on entry, restored once on exit).
// ============================================================================
constexpr int kRegGuestState = 28;
constexpr int kRegDispatcher = 27;
constexpr int kRegDispatchTop = 26;
constexpr int kRegExitStub = 25;

/// Allocate one page of MAP_JIT RWX memory for the gateway. Same
/// MAP_JIT discipline as CodeCache::AllocateRwxRegion on Apple Silicon:
/// the page is created RWX-capable but a per-thread write-protect
/// toggle decides whether THIS thread currently sees it as writable or
/// executable. At gateway-generation time we must be in writable mode.
u8* AllocateGatewayRegion() {
    void* p = ::mmap(nullptr, GATEWAY_SIZE, PROT_READ | PROT_WRITE | PROT_EXEC,
                     MAP_PRIVATE | MAP_ANON | MAP_JIT, -1, 0);
    if (p == MAP_FAILED) {
        return nullptr;
    }
    return static_cast<u8*>(p);
}

void FreeGatewayRegion(u8* p) {
    if (p == nullptr) return;
    ::munmap(p, GATEWAY_SIZE);
}

/// Generate the gateway entry stub. Callable as
/// `void (*)(GuestState*, DispatcherFn)`.
///
/// AAPCS64: arg0 = x0 (GuestState*), arg1 = x1 (DispatcherFn),
/// result = x0, callee-saved x19..x28 + FP(x29) + LR(x30).
///
/// IMPORTANT (Apple Silicon W^X): the caller MUST have switched this
/// thread's MAP_JIT pages to WRITABLE before calling, and MUST switch
/// back to EXECUTABLE + invalidate the icache AFTER. See the Gateway
/// constructor.
/// Gateway code generator. xbyak_aarch64's pre-instantiated register
/// instances (x0, x29, sp, ...) and mnemonic functions are MEMBERS of
/// CodeGenerator — they are not reachable through a file-scope
/// `using namespace`, only from within a CodeGenerator-derived class. So we
/// follow the library's documented pattern (see its sample/add.cpp) and
/// emit from a subclass constructor, where `x0`, `sp`, `stp(...)`, etc. all
/// resolve as inherited members. The pinned registers are built by index
/// with XReg(i), which is valid in any scope.
class GatewayGenerator : public Xbyak_aarch64::CodeGenerator {
public:
    GatewayGenerator(u8* code_buf, u64 code_size)
        : Xbyak_aarch64::CodeGenerator(code_size, code_buf) {
        // xbyak_aarch64 splits its symbols: the register CLASSES (XReg, ...)
        // and Label are in the GLOBAL namespace, while the address-helper free
        // functions (pre_ptr/post_ptr/ptr) live in namespace Xbyak_aarch64. A
        // single using-directive makes both reachable here (and the bare
        // register instances x0/sp/x29 resolve as inherited CodeGenerator
        // members).
        using namespace Xbyak_aarch64;

        const XReg rGuestState = XReg(kRegGuestState);
        const XReg rDispatcher = XReg(kRegDispatcher);
        const XReg rDispatchTop = XReg(kRegDispatchTop);
        const XReg rExitStub = XReg(kRegExitStub);

        // ---- PROLOGUE ----
        // Save the full AAPCS64 callee-saved set (see policy note above).
        // STP pre-indexed = push-pair; each pair is 16 bytes so SP stays
        // 16-aligned at every step (a hard AArch64 requirement).
        //
        // GPRs x19..x28 + FP(x29)/LR(x30): 6 pairs, 96 bytes.
        stp(x19, x20, pre_ptr(sp, -16));
        stp(x21, x22, pre_ptr(sp, -16));
        stp(x23, x24, pre_ptr(sp, -16));
        stp(x25, x26, pre_ptr(sp, -16));
        stp(x27, x28, pre_ptr(sp, -16));
        stp(x29, x30, pre_ptr(sp, -16));            // FP, LR
        // Callee-saved FP/SIMD: only the low 64 bits of v8..v15 (the d-regs)
        // are callee-saved. 4 pairs, 64 bytes.
        stp(d8, d9, pre_ptr(sp, -16));
        stp(d10, d11, pre_ptr(sp, -16));
        stp(d12, d13, pre_ptr(sp, -16));
        stp(d14, d15, pre_ptr(sp, -16));
        mov(x29, sp); // frame pointer (points at the saved d14/d15 pair)

        // Normalize args into pinned registers.
        mov(rGuestState, x0); // state
        mov(rDispatcher, x1); // dispatcher

        // Pre-compute exit-path addresses. ADR is PC-relative (+/-1 MiB),
        // far more than the gateway's size.
        Label dispatch_loop_top;
        Label exit_stub;
        adr(rDispatchTop, dispatch_loop_top);
        adr(rExitStub, exit_stub);

        // ---- DISPATCH LOOP ----
        L(dispatch_loop_top);
        mov(x0, rGuestState); // dispatcher(state): arg0 = x0
        blr(rDispatcher);     // branch-with-link to register
        cbz(x0, exit_stub);   // null result -> exit (no flags needed)
        br(x0);               // else branch into JIT block

        // ---- EXIT STUB ----
        L(exit_stub);
        // Pop in EXACT reverse order of the prologue pushes (post-indexed =
        // pop-pair). FP/SIMD first (they were pushed last), then GPRs.
        ldp(d14, d15, post_ptr(sp, 16));
        ldp(d12, d13, post_ptr(sp, 16));
        ldp(d10, d11, post_ptr(sp, 16));
        ldp(d8, d9, post_ptr(sp, 16));
        ldp(x29, x30, post_ptr(sp, 16));            // FP, LR
        ldp(x27, x28, post_ptr(sp, 16));
        ldp(x25, x26, post_ptr(sp, 16));
        ldp(x23, x24, post_ptr(sp, 16));
        ldp(x21, x22, post_ptr(sp, 16));
        ldp(x19, x20, post_ptr(sp, 16));
        ret();

        // Finalize the xbyak_aarch64 buffer (lays out labels). We manage
        // MAP_JIT write-protect + sys_icache_invalidate ourselves in the
        // Gateway constructor.
        ready();
    }
};

void GenerateGateway(u8* code_buf, u64 code_size) {
    GatewayGenerator{code_buf, code_size};
}

} // namespace

Gateway::Gateway() {
    gateway_code_ = AllocateGatewayRegion();
    ASSERT_MSG(gateway_code_ != nullptr,
               "Gateway(arm64): failed to allocate MAP_JIT RWX page");
    gateway_size_ = GATEWAY_SIZE;

    // Apple Silicon W^X: switch this thread's MAP_JIT pages to WRITABLE
    // before emitting, then back to EXECUTABLE + invalidate icache after.
    pthread_jit_write_protect_np(0); // writable
    GenerateGateway(gateway_code_, gateway_size_);
    pthread_jit_write_protect_np(1); // executable

    // Invalidate the instruction cache over the gateway range. On
    // AArch64 the I/D caches are not coherent for freshly-written code;
    // without this the CPU may execute stale bytes.
    // In this TU the libkern declaration is visible via the Xbyak_aarch64
    // namespace (the library includes OSCacheControl.h within its own
    // namespace), so qualify it accordingly.
    Xbyak_aarch64::sys_icache_invalidate(gateway_code_, gateway_size_);

    entry_ = reinterpret_cast<EntryFn>(gateway_code_);

    LOG_INFO(Core, "Gateway(arm64/macOS): generated at {} (size budget {} bytes)",
             static_cast<void*>(gateway_code_), gateway_size_);
}

Gateway::~Gateway() {
    FreeGatewayRegion(gateway_code_);
    gateway_code_ = nullptr;
    entry_ = nullptr;
}

void Gateway::Enter(GuestState& state, DispatcherFn dispatcher) {
    ASSERT_MSG(entry_ != nullptr, "Gateway(arm64): entry not generated");
    entry_(&state, dispatcher);
}

} // namespace Core::Runtime
