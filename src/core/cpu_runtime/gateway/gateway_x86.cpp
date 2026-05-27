// SPDX-FileCopyrightText: Copyright 2026 shadPS4 Emulator Project
// SPDX-License-Identifier: GPL-2.0-or-later

#include "core/cpu_runtime/gateway/gateway.h"

#include <xbyak/xbyak.h>

#include "common/assert.h"
#include "common/logging/log.h"
#include "core/cpu_runtime/guest_state.h"

#ifdef _WIN32
#include <windows.h>
#else
#include <sys/mman.h>
#endif

namespace Core::Runtime {

namespace {

// Gateway code size budget. The full gateway is small (~200 bytes
// in tested form) but allocate a page for it to be safe across
// future expansions.
constexpr u64 GATEWAY_SIZE = 4096;

/// Allocate one page of RWX memory for the gateway. Same shape as
/// CodeCache::AllocateRwxRegion but kept separate so a code-cache
/// flush doesn't take out the gateway.
u8* AllocateGatewayRegion() {
#ifdef _WIN32
    void* p = ::VirtualAlloc(nullptr, GATEWAY_SIZE, MEM_COMMIT | MEM_RESERVE,
                             PAGE_EXECUTE_READWRITE);
    return static_cast<u8*>(p);
#else
    void* p = ::mmap(nullptr, GATEWAY_SIZE, PROT_READ | PROT_WRITE | PROT_EXEC,
                     MAP_PRIVATE | MAP_ANON, -1, 0);
    if (p == MAP_FAILED) {
        return nullptr;
    }
    return static_cast<u8*>(p);
#endif
}

void FreeGatewayRegion(u8* p) {
    if (p == nullptr) return;
#ifdef _WIN32
    ::VirtualFree(p, 0, MEM_RELEASE);
#else
    ::munmap(p, GATEWAY_SIZE);
#endif
}

/// Generate the gateway entry stub. Called once at Gateway
/// construction time; the returned pointer is callable as
/// `void (*)(GuestState*, DispatcherFn)`.
///
/// ABI: System V on Linux/macOS, MS x64 on Windows. The two ABIs
/// differ in argument registers and which registers are callee-
/// saved. We emit the right sequence for each.
///
/// Pinned registers across all JIT code:
///   r13 = GuestState*
///   r12 = DispatcherFn pointer
///   r14 = exit_stub_address (so JIT can `jmp r14` to exit)
///
/// Why r12/r13/r14: they're callee-saved on both ABIs, so we can
/// set them once at gateway entry and they survive through any
/// helper calls the JIT makes. Why those three specifically: they
/// have low encoding cost (REX.B variants) and don't collide with
/// the C ABI's argument registers.
void GenerateGateway(u8* code_buf, u64 code_size) {
    using namespace Xbyak::util;

    Xbyak::CodeGenerator c{code_size, code_buf};

    // ============================================================
    // ENTRY: void gateway_entry(GuestState* state, DispatcherFn d)
    // ============================================================
    //
    // System V:  arg0 = rdi, arg1 = rsi
    // MS x64:    arg0 = rcx, arg1 = rdx
    //
    // We normalize: state into r13, dispatcher into r12.

#ifdef _WIN32
    // MS x64 ABI:
    // - Args in rcx, rdx, r8, r9.
    // - Callee-saved: rbx, rbp, rdi, rsi, r12-r15, xmm6-xmm15.
    // - Shadow space: 32 bytes the caller reserves below rsp.
    //   We don't need to allocate it here (we're the callee) but
    //   we DO need to allocate it before calling C from JIT.
    c.push(rbp);
    c.push(rbx);
    c.push(rdi);
    c.push(rsi);
    c.push(r12);
    c.push(r13);
    c.push(r14);
    c.push(r15);
    // Total pushed: 8 * 8 = 64 bytes. rsp was 16-aligned at function
    // entry (per ABI), after `call` it's misaligned by 8 (return
    // address), then 8*8 = 64 puts it back to 16-aligned. No extra
    // sub-from-rsp needed.

    c.mov(r13, rcx); // state
    c.mov(r12, rdx); // dispatcher
#else
    // System V x86_64 ABI:
    // - Args in rdi, rsi, rdx, rcx, r8, r9.
    // - Callee-saved: rbx, rbp, r12-r15.
    c.push(rbp);
    c.push(rbx);
    c.push(r12);
    c.push(r13);
    c.push(r14);
    c.push(r15);
    // 6 * 8 = 48 bytes pushed. rsp was 16-aligned at entry, after
    // `call` misaligned by 8, plus 48 = misaligned by 8 (since
    // 48 % 16 = 0). So push one more dummy to realign:
    c.sub(rsp, 8);

    c.mov(r13, rdi); // state
    c.mov(r12, rsi); // dispatcher
#endif

    // ============================================================
    // Pre-compute exit-path addresses:
    //   r14 = dispatch_loop_top (normal block-end re-enters dispatcher)
    //   r15 = exit_stub         (fatal exit returns to C)
    //
    // Block terminators (JMP/Jcc/RET/CALL) `jmp r14` to continue
    // dispatching. The dispatcher decides whether to exit (sentinel
    // host-return, compile failure) by returning nullptr, in which
    // case the gateway falls through from the dispatcher call to
    // the exit stub.
    //
    // Lifter-emitted "fatal exit" paths (unsupported instruction,
    // decode failure) `jmp r15` directly, bypassing the dispatcher.
    // This preserves diagnostic state (state.rip holds the offending
    // address) and avoids the dispatcher trying to compile the same
    // bad address again.
    // ============================================================
    Xbyak::Label exit_stub;
    Xbyak::Label dispatch_loop_top;
    c.lea(r14, ptr[rip + dispatch_loop_top]);
    c.lea(r15, ptr[rip + exit_stub]);

    // ============================================================
    // DISPATCH LOOP
    // ============================================================
    //
    // Call dispatcher(state) -> rax = host code pointer.
    //   - If non-null: jmp rax (into JIT code). When the JIT block
    //     exits via a block terminator, it does `jmp r14`, which
    //     lands at dispatch_loop_top and we go around again.
    //   - If null: exit to C via the exit stub.
    //
    // This loop runs entirely in machine code without returning to
    // C until the dispatcher decides to exit. Block chaining (where
    // blocks jump directly to each other, bypassing the dispatcher)
    // is a future optimization layered on top.

    c.L(dispatch_loop_top);

    // Call dispatcher(state). r13 is state. We need state in rdi
    // (System V) or rcx (Windows) for the C call.
#ifdef _WIN32
    c.sub(rsp, 32); // shadow space for the C call
    c.mov(rcx, r13);
    c.call(r12);
    c.add(rsp, 32);
#else
    c.mov(rdi, r13);
    c.call(r12);
#endif

    // Returned host code pointer in rax. If it's null, exit
    // (dispatcher returned null = "we should stop"). Otherwise
    // jmp into JIT; the JIT will eventually `jmp r14` back to
    // dispatch_loop_top.
    c.test(rax, rax);
    c.jz(exit_stub);
    c.jmp(rax);

    // ============================================================
    // EXIT STUB
    // ============================================================
    //
    // JIT code jumps here (via r14) when a block decides to exit.
    // Restore callee-saved regs and return to C.

    c.L(exit_stub);

#ifdef _WIN32
    c.pop(r15);
    c.pop(r14);
    c.pop(r13);
    c.pop(r12);
    c.pop(rsi);
    c.pop(rdi);
    c.pop(rbx);
    c.pop(rbp);
#else
    c.add(rsp, 8); // undo the alignment dummy
    c.pop(r15);
    c.pop(r14);
    c.pop(r13);
    c.pop(r12);
    c.pop(rbx);
    c.pop(rbp);
#endif

    c.ret();

    // ============================================================
    // Done. xbyak has emitted into code_buf. The pointer to use
    // is the start of the buffer (entry point is the first
    // instruction).
    // ============================================================
}

} // namespace

Gateway::Gateway() {
    gateway_code_ = AllocateGatewayRegion();
    ASSERT_MSG(gateway_code_ != nullptr,
               "Gateway: failed to allocate RWX page");
    gateway_size_ = GATEWAY_SIZE;

    GenerateGateway(gateway_code_, gateway_size_);
    entry_ = reinterpret_cast<EntryFn>(gateway_code_);

    LOG_INFO(Core, "Gateway: generated at {} (size budget {} bytes)",
             static_cast<void*>(gateway_code_), gateway_size_);
}

Gateway::~Gateway() {
    FreeGatewayRegion(gateway_code_);
    gateway_code_ = nullptr;
    entry_ = nullptr;
}

void Gateway::Enter(GuestState& state, DispatcherFn dispatcher) {
    ASSERT_MSG(entry_ != nullptr, "Gateway: entry not generated");
    entry_(&state, dispatcher);
}

} // namespace Core::Runtime
