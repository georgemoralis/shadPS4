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
    void* p =
        ::VirtualAlloc(nullptr, GATEWAY_SIZE, MEM_COMMIT | MEM_RESERVE, PAGE_EXECUTE_READWRITE);
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
    if (p == nullptr)
        return;
#ifdef _WIN32
    ::VirtualFree(p, 0, MEM_RELEASE);
#else
    ::munmap(p, GATEWAY_SIZE);
#endif
}

/// Frame allocation made once in the Windows prologue: 32 bytes of MS x64
/// shadow space for the dispatcher call PLUS 8 bytes of alignment. Keeping
/// the whole amount in the prologue (instead of sub/add rsp,32 around every
/// dispatcher call, as an earlier revision did) gives the gateway a CONSTANT
/// rsp for every instruction past the prologue -- which is the property that
/// lets a single UNWIND_INFO describe the frame correctly at any PC. The OS
/// unwinder applies the full code array for any PC past the prologue; an rsp
/// that changes mid-function (per-call shadow space) cannot be expressed in
/// one static UNWIND_INFO, and the walk reconstructed the caller frame 40
/// bytes off exactly when it mattered most: while parked inside the
/// dispatcher call, where the gateway spends nearly all of its time.
#ifdef _WIN32
constexpr u32 WIN_GATEWAY_FRAME_ALLOC = 40; // 32 shadow + 8 align
#endif

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
/// Returns the measured byte length of the (Windows) prologue -- the offset
/// of the first instruction after the frame allocation -- so the unwind
/// registration describes the bytes ACTUALLY emitted rather than a hardcoded
/// guess that an xbyak encoding choice could silently desync. Returns 0 on
/// System V (no unwind tables are registered there).
u32 GenerateGateway(u8* code_buf, u64 code_size) {
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
    //   We don't need it for OUR entry (we're the callee; our caller
    //   provided it), but we must provide it for the dispatcher calls
    //   we make -- done once, in the frame allocation below.
    c.push(rbp);
    c.push(rbx);
    c.push(rdi);
    c.push(rsi);
    c.push(r12);
    c.push(r13);
    c.push(r14);
    c.push(r15);
    // The pushes have fixed encodings (1 byte for the legacy regs, 2 bytes
    // for the REX.B r12..r15 forms); the unwind codes below depend on those
    // offsets, so pin them.
    const u32 pushes_end = static_cast<u32>(c.getSize());
    ASSERT_MSG(pushes_end == 12,
               "Gateway: unexpected push encoding length {} (unwind codes "
               "assume 12)", pushes_end);

    // Frame allocation, ONCE, covering the whole function:
    //
    //   At Enter() entry (immediately after CALL), rsp is misaligned by 8
    //   from a 16-byte boundary. 8 pushes = 64 bytes, still misaligned by
    //   8. Subtracting 40 (= WIN_GATEWAY_FRAME_ALLOC: 32 bytes of MS x64
    //   shadow space + 8 alignment) leaves rsp 16-aligned at every
    //   subsequent CALL site, with the shadow space permanently sitting
    //   just above rsp where the callee expects it.
    //
    //   Allocating per call (sub/add rsp,32 around the dispatcher call)
    //   was rejected: it makes rsp PC-dependent, which a single static
    //   UNWIND_INFO cannot describe -- see the unwind registration below.
    c.sub(rsp, WIN_GATEWAY_FRAME_ALLOC);
    const u32 prolog_end = static_cast<u32>(c.getSize());

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
    // Shadow space was allocated once in the prologue; rsp is already
    // 16-aligned here with 32 free bytes above it. No per-call adjustment
    // (that would break the static unwind description -- see prologue).
    c.mov(rcx, r13);
    c.call(r12);
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
    // Undo the prologue's single frame allocation. `add rsp, imm8` followed
    // by pops + ret matches the OS unwinder's legal-epilogue pattern.
    c.add(rsp, WIN_GATEWAY_FRAME_ALLOC);
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
#ifdef _WIN32
    return prolog_end;
#else
    return 0;
#endif
}

} // namespace

#ifdef _WIN32

// =============================================================================
// Windows unwind info registration for the JIT gateway.
// =============================================================================
//
// Windows x64 requires every executable code range to have an entry in the
// process's unwind tables (.pdata equivalent) so the OS can walk stack
// frames during exception handling, debugger break-in, sampling profilers,
// /GS security checks, and other SEH operations.
//
// For statically compiled code, the linker emits these tables automatically.
// For dynamically generated code (our gateway), we must register the info
// ourselves via `RtlAddFunctionTable`.
//
// Without this, the OS unwinder hits a gateway frame, can't find any
// unwind info, and crashes the process — which is exactly what happens
// to "simple" JIT-using tests like MovAddRet on Windows debug builds,
// where /GS cookie checks and other security mechanisms trigger
// frequent SEH walks even in the absence of explicit exceptions.
//
// We register info for the gateway here. JIT-compiled blocks are
// jumped to (not called), so they have an unusual relationship with
// the unwinder; if gateway-only registration turns out insufficient,
// the next layer is to also describe JIT blocks via
// RtlInstallFunctionTableCallback (deferred).
//
// References:
//   - x64 exception handling overview:
//     learn.microsoft.com/cpp/build/exception-handling-x64
//   - RtlAddFunctionTable:
//     learn.microsoft.com/windows/win32/api/winnt/nf-winnt-rtladdfunctiontable
//
// Structures below mirror those in <winnt.h>. We define our own
// copies because (a) `UNWIND_INFO` is not a documented public type
// (only `RUNTIME_FUNCTION` is), and (b) the layout is fixed and easy
// to express directly.

namespace {

// Operation codes for UNWIND_CODE.UnwindOp.
constexpr u8 UWOP_PUSH_NONVOL = 0;
// Fixed stack allocation of (op_info + 1) * 8 bytes, 8..128. Describes the
// prologue's `sub rsp, WIN_GATEWAY_FRAME_ALLOC`.
constexpr u8 UWOP_ALLOC_SMALL = 2;
// (other UWOPs not used here)

// Register encodings for UWOP_PUSH_NONVOL.OpInfo (matches AMD64 reg-num).
constexpr u8 UWREG_RBX = 3;
constexpr u8 UWREG_RBP = 5;
constexpr u8 UWREG_RSI = 6;
constexpr u8 UWREG_RDI = 7;
constexpr u8 UWREG_R12 = 12;
constexpr u8 UWREG_R13 = 13;
constexpr u8 UWREG_R14 = 14;
constexpr u8 UWREG_R15 = 15;

// One UNWIND_CODE describes a single prologue operation that the OS
// needs to be able to reverse during a stack walk. Layout matches
// the `UNWIND_CODE` union in winnt.h.
struct UnwindCode {
    u8 code_offset;   // offset of the END of the corresponding insn
    u8 unwind_op : 4; // UWOP_*
    u8 op_info : 4;   // op-specific (e.g. UWREG_* for UWOP_PUSH_NONVOL)
};
static_assert(sizeof(UnwindCode) == 2, "UNWIND_CODE must be exactly 2 bytes");

// UNWIND_INFO header + the inline codes array. Layout matches
// winnt.h:_UNWIND_INFO. We use 9 codes (1 ALLOC_SMALL + 8 pushes);
// CountOfCodes tells the OS how many are valid. Per the documented
// format the array always holds an EVEN number of slots for alignment,
// so it is sized 10 with the last entry unused.
struct UnwindInfoGateway {
    u8 version : 3; // = 1
    u8 flags : 5;   // = 0 (no exception handler, no chained)
    u8 size_of_prolog;
    u8 count_of_codes;
    u8 frame_register : 4; // = 0 (no frame register)
    u8 frame_offset : 4;   // = 0
    UnwindCode codes[10];  // 9 used: ALLOC_SMALL + 8 pushes; [9] = pad
};
static_assert(sizeof(UnwindInfoGateway) == 4 + 10 * 2,
              "UNWIND_INFO layout mismatch — check bit-field packing");

// Offsets within the gateway RWX page. We co-locate unwind metadata
// with the gateway code in the same page; placing it at offset 2048
// keeps it comfortably past the ~200-byte gateway code while still
// leaving room for the RUNTIME_FUNCTION descriptor right after.
constexpr u64 GATEWAY_UNWIND_INFO_OFFSET = 2048;
constexpr u64 GATEWAY_RUNTIME_FN_OFFSET = GATEWAY_UNWIND_INFO_OFFSET + sizeof(UnwindInfoGateway);
static_assert(GATEWAY_RUNTIME_FN_OFFSET + sizeof(RUNTIME_FUNCTION) < GATEWAY_SIZE,
              "unwind metadata doesn't fit in the gateway page");

/// Populate unwind metadata for the gateway and register it with the OS.
///
/// Called once per Gateway construction, after the gateway code has
/// been emitted. `prolog_size` is the MEASURED prologue length returned
/// by GenerateGateway (the offset of the first instruction after the
/// frame allocation) -- using the emitted truth instead of a hardcoded
/// constant means an xbyak encoding change cannot silently desync the
/// unwind description from the code. Returns true if the OS accepted
/// the registration. On false, JIT execution may still work for tests
/// that don't trigger SEH walks, but anything that does (debugger
/// break, exception, /GS failure) will crash.
bool RegisterGatewayUnwindInfo(u8* gateway_base, u32 prolog_size) {
    // ----------------------------------------------------------------
    // Build UNWIND_INFO describing the gateway's MS x64 prologue:
    //
    //   offset 0:  push rbp    (1 byte;  rsp after = -8)
    //   offset 1:  push rbx    (1 byte;  rsp after = -16)
    //   offset 2:  push rdi    (1 byte;  rsp after = -24)
    //   offset 3:  push rsi    (1 byte;  rsp after = -32)
    //   offset 4:  push r12    (2 bytes; rsp after = -40)
    //   offset 6:  push r13    (2 bytes; rsp after = -48)
    //   offset 8:  push r14    (2 bytes; rsp after = -56)
    //   offset 10: push r15    (2 bytes; rsp after = -64)
    //   offset 12: sub rsp, 40 (4 bytes; rsp after = -104)
    //   offset 16: (prologue ends = prolog_size; the following movs /
    //              leas don't affect unwind state)
    //
    // The ALLOC_SMALL entry is the load-bearing addition: every PC past
    // the prologue -- including the return address of the dispatcher
    // call, where SEH walks overwhelmingly arrive -- now reconstructs
    // the caller's rsp through the full 104-byte frame. The previous
    // revision described only the pushes while the code subtracted an
    // extra 8 (alignment) plus a per-call 32 (shadow space), so every
    // walk through a parked gateway frame was 40 bytes off and chased
    // a garbage return address: the exact crash this registration
    // exists to prevent.
    //
    // Per Microsoft documentation, codes are stored in REVERSE
    // prologue order: codes[0] describes the LAST prologue op. The
    // CodeOffset field records where the corresponding instruction
    // ENDS (its "rsp value after this op was applied").
    // ----------------------------------------------------------------
    static_assert(WIN_GATEWAY_FRAME_ALLOC % 8 == 0 && WIN_GATEWAY_FRAME_ALLOC <= 128,
                  "UWOP_ALLOC_SMALL encodes (n/8 - 1) in 4 bits: size must be "
                  "a multiple of 8, at most 128");
    constexpr u8 kAllocSmallInfo = WIN_GATEWAY_FRAME_ALLOC / 8 - 1;

    UnwindInfoGateway* ui =
        reinterpret_cast<UnwindInfoGateway*>(gateway_base + GATEWAY_UNWIND_INFO_OFFSET);
    *ui = {};
    ui->version = 1;
    ui->flags = 0;
    ui->size_of_prolog = static_cast<u8>(prolog_size);
    ui->count_of_codes = 9;
    ui->frame_register = 0;
    ui->frame_offset = 0;
    ui->codes[0] = {static_cast<u8>(prolog_size), UWOP_ALLOC_SMALL, kAllocSmallInfo};
    ui->codes[1] = {12, UWOP_PUSH_NONVOL, UWREG_R15};
    ui->codes[2] = {10, UWOP_PUSH_NONVOL, UWREG_R14};
    ui->codes[3] = {8, UWOP_PUSH_NONVOL, UWREG_R13};
    ui->codes[4] = {6, UWOP_PUSH_NONVOL, UWREG_R12};
    ui->codes[5] = {4, UWOP_PUSH_NONVOL, UWREG_RSI};
    ui->codes[6] = {3, UWOP_PUSH_NONVOL, UWREG_RDI};
    ui->codes[7] = {2, UWOP_PUSH_NONVOL, UWREG_RBX};
    ui->codes[8] = {1, UWOP_PUSH_NONVOL, UWREG_RBP};

    // ----------------------------------------------------------------
    // RUNTIME_FUNCTION ties code range -> unwind info. Addresses are
    // RVAs (relative to BaseAddress passed to RtlAddFunctionTable).
    //
    // BeginAddress: 0 (gateway code starts at the page base)
    // EndAddress:   covers the full code budget so any future
    //               extensions in the same page are also described.
    // ----------------------------------------------------------------
    RUNTIME_FUNCTION* rf =
        reinterpret_cast<RUNTIME_FUNCTION*>(gateway_base + GATEWAY_RUNTIME_FN_OFFSET);
    rf->BeginAddress = 0;
    rf->EndAddress = static_cast<DWORD>(GATEWAY_SIZE);
    rf->UnwindInfoAddress = static_cast<DWORD>(GATEWAY_UNWIND_INFO_OFFSET);

    return ::RtlAddFunctionTable(rf, 1, reinterpret_cast<DWORD64>(gateway_base)) != FALSE;
}

void UnregisterGatewayUnwindInfo(u8* gateway_base) {
    RUNTIME_FUNCTION* rf =
        reinterpret_cast<RUNTIME_FUNCTION*>(gateway_base + GATEWAY_RUNTIME_FN_OFFSET);
    ::RtlDeleteFunctionTable(rf);
}

} // namespace

#endif // _WIN32

Gateway::Gateway() {
    gateway_code_ = AllocateGatewayRegion();
    ASSERT_MSG(gateway_code_ != nullptr, "Gateway: failed to allocate RWX page");
    gateway_size_ = GATEWAY_SIZE;

    const u32 prolog_size = GenerateGateway(gateway_code_, gateway_size_);
    entry_ = reinterpret_cast<EntryFn>(gateway_code_);

#ifdef _WIN32
    // Register the gateway's unwind metadata so Windows can walk
    // stack frames through it. Without this, SEH walks (from /GS
    // checks, debugger breaks, etc.) crash the process even for
    // simple JIT operations. The MEASURED prologue size from the
    // generator keeps the description tied to the bytes actually
    // emitted. See RegisterGatewayUnwindInfo for the full rationale.
    if (!RegisterGatewayUnwindInfo(gateway_code_, prolog_size)) {
        LOG_WARNING(Core, "Gateway: RtlAddFunctionTable failed; SEH walks "
                          "through JIT frames will crash. JIT may still "
                          "work for code paths that don't trigger walks.");
    }
#else
    (void)prolog_size; // only consumed by the Windows unwind registration
#endif

    LOG_INFO(Core, "Gateway: generated at {} (size budget {} bytes)",
             static_cast<void*>(gateway_code_), gateway_size_);
}

Gateway::~Gateway() {
#ifdef _WIN32
    if (gateway_code_ != nullptr) {
        UnregisterGatewayUnwindInfo(gateway_code_);
    }
#endif
    FreeGatewayRegion(gateway_code_);
    gateway_code_ = nullptr;
    entry_ = nullptr;
}

void Gateway::Enter(GuestState& state, DispatcherFn dispatcher) {
    ASSERT_MSG(entry_ != nullptr, "Gateway: entry not generated");
    entry_(&state, dispatcher);
}

} // namespace Core::Runtime
