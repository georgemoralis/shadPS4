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
        using Xbyak_aarch64::Label;
        using Xbyak_aarch64::XReg;

        const XReg rGuestState = XReg(kRegGuestState);
        const XReg rDispatcher = XReg(kRegDispatcher);
        const XReg rDispatchTop = XReg(kRegDispatchTop);
        const XReg rExitStub = XReg(kRegExitStub);

        // ---- PROLOGUE ----
        // STP pre-indexed = push-pair; each pair is 16 bytes so the stack
        // stays 16-aligned (a hard AArch64 requirement at all times).
        stp(x29, x30, pre_ptr(sp, -16));            // FP, LR
        stp(rGuestState, rDispatcher, pre_ptr(sp, -16));
        stp(rDispatchTop, rExitStub, pre_ptr(sp, -16));
        mov(x29, sp); // frame pointer

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
        // Pop pairs in reverse order (post-indexed = pop-pair).
        ldp(rDispatchTop, rExitStub, post_ptr(sp, 16));
        ldp(rGuestState, rDispatcher, post_ptr(sp, 16));
        ldp(x29, x30, post_ptr(sp, 16));
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
    // Qualify ::sys_icache_invalidate explicitly: xbyak_aarch64 declares its
    // own same-named symbol, which can shadow libkern's via ADL/namespace
    // lookup and fail to resolve. The global libkern one is what we want.
    ::sys_icache_invalidate(gateway_code_, gateway_size_);

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
