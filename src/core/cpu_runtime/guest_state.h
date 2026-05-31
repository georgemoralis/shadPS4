// SPDX-FileCopyrightText: Copyright 2026 shadPS4 Emulator Project
// SPDX-License-Identifier: GPL-2.0-or-later

#pragma once

#include <array>
#include <cstddef>
#include "common/types.h"

namespace Core::Runtime {

/// PS4 guest CPU state. This struct is shared between all backends; its
/// layout is part of the gateway ABI. Field offsets are used directly
/// from hand-written assembly (gateway_x86.cpp, gateway_arm64.cpp) and
/// from emitted JIT code. Changing the layout requires updating those
/// callers.
///
/// Layout rationale:
///
///   1. Integer GPRs first, in canonical AMD64 order (RAX..R15). This
///      lets backends compute a field address from a 0..15 register
///      index via simple arithmetic, no branches.
///   2. RIP next, hot in the dispatcher loop.
///   3. RFLAGS as a single u64. Lazy-flag handling lifts the
///      flag-producing operands into a side-band (see flag_op /
///      flag_lhs / flag_rhs), so RFLAGS itself is only materialized
///      when an instruction actually reads it.
///   4. Segment bases (fs_base, gs_base). On hosts where the
///      patch-on-fault model handled this, the values lived in TLS;
///      here they're explicit so the JIT can fold them into addressing.
///   5. SSE/AVX register file. Sized to 32 ymm registers (AVX-256)
///      even on hosts that only have 16. This matches the prototype
///      design and keeps the offsets stable across hosts.
///
/// The struct is intentionally trivial and aggregate-constructible.
/// Constructors / destructors / virtual functions would all complicate
/// the gateway path with no benefit.
struct alignas(64) GuestState {
    // ---- Integer registers ----
    // Indexed in canonical AMD64 order. gpr[0] = RAX, gpr[4] = RSP,
    // gpr[5] = RBP, gpr[7] = RDI, etc.
    std::array<u64, 16> gpr;

    // ---- Instruction pointer ----
    u64 rip;

    // ---- Flags ----
    // Materialized RFLAGS value. See flag_op for lazy-flag scheme.
    u64 rflags;

    // ---- Lazy-flag side band ----
    // flag_op encodes the most recent flag-producing operation. When an
    // instruction reads flags, the runtime materializes rflags from
    // (flag_op, flag_lhs, flag_rhs, flag_result). When an instruction
    // both writes flags and is followed immediately by a consumer
    // (fused jcc, etc.), the lifter can short-circuit and emit a
    // direct branch without materializing rflags at all.
    u32 flag_op;
    // flag_width: operand width in bits (8/16/32/64) for the lazy-flag
    // computation. The materializer derives ZF/SF/PF over the low `flag_width`
    // bits and applies width-correct CF/OF. A value of 0 is treated as 64 so
    // that producers which predate this field (e.g. the x86 host lifter, which
    // computes flags natively and never sets it) keep their existing 64-bit
    // behavior. Occupies the former _pad_flag slot — no layout change.
    u32 flag_width;
    u64 flag_lhs;
    u64 flag_rhs;
    u64 flag_result;

    // ---- Segment bases ----
    // fs_base and gs_base are guest-visible. On x86 host, code that
    // reads fs:[0] (or similar) is normally patched by cpu_patches.cpp
    // to a sequence that loads from this field; in the runtime path
    // the lifter does the same translation per-block.
    u64 fs_base;
    u64 gs_base;

    // ---- SSE / AVX register file ----
    // 32 lanes of 256 bits. xmm/ymm are aliases (xmm0 = low 128 of ymm0).
    // On hosts that only have 16 ymm regs, lanes 16..31 are still
    // reachable from the JIT via memory operands; we never assume the
    // host has 32.
    alignas(32) std::array<u64, 32 * 4> ymm;

    // ---- MXCSR (SSE control/status) ----
    u32 mxcsr;

    // ---- Exit code ----
    // Set by the dispatcher / helpers when guest execution should
    // pause and return control to C++. Codes are in `ExitReason`.
    u32 exit_reason;

    // ---- Reserved / scratch ----
    // Available for backends that need a small amount of state-adjacent
    // scratch space (e.g. for save/restore around helper calls).
    // Backends MUST NOT assume any specific value is preserved across
    // dispatcher invocations.
    std::array<u64, 4> scratch;

    // ---- x87 FPU state ----
    // The x87 register file is a stack of 8 registers accessed relative
    // to a top-of-stack pointer (fpu_top): ST(i) lives in
    // st[(fpu_top + i) & 7]. A push (fld) does fpu_top = (fpu_top-1)&7
    // then writes st[fpu_top]; a pop (fstp) reads st[fpu_top] then does
    // fpu_top = (fpu_top+1)&7.
    //
    // PRECISION NOTE: real x87 registers are 80-bit extended precision.
    // We store each as a 64-bit double bit-pattern instead. This is
    // bit-exact for float/double load, store, convert, compare, and
    // ordinary arithmetic on values that fit in a double — which covers
    // essentially all PS4 guest x87 use (compilers emit SSE for real
    // FP work and only fall back to x87 for occasional conversions and
    // legacy routines). Code that depends on the extra 11 mantissa bits
    // or the wider exponent of true 80-bit intermediates will differ.
    // 80-bit storage was rejected because the host long double is only
    // 64-bit on the Windows target, so it cannot represent the format.
    alignas(16) std::array<u64, 8> st;

    // Top-of-stack pointer, 0..7. Authoritative; the x87 status-word TOP
    // field (bits 11..13) is composed from this on demand by fstsw/fnstsw.
    u32 fpu_top;

    // Tag word, 2 bits per PHYSICAL register st[i] (not ST(i)):
    //   0b00 valid, 0b01 zero, 0b10 special (NaN/Inf/denormal),
    //   0b11 empty. Needed so fld can detect stack overflow (push onto a
    //   non-empty slot) and fstp/arith can detect underflow (access of an
    //   empty slot). v1 tracks empty vs valid precisely; the
    //   zero/special distinction is computed only where an instruction
    //   actually observes it.
    u16 fpu_tag;

    // Control word (rounding mode, precision control, exception masks).
    // Read/written by fldcw/fnstcw. Initialized to 0x037F (the x87
    // default: round-to-nearest, 64-bit precision, all exceptions
    // masked) when the FPU is reset (finit).
    u16 fpu_cw;

    // Cached condition codes C0/C1/C2/C3 from the last compare-class
    // instruction (fcom/fucom/fxam/...), in their status-word bit
    // positions (C0=bit8, C1=bit9, C2=bit10, C3=bit14). fstsw/fnstsw
    // ORs these together with the TOP field to build the full status
    // word. Kept separate from fpu_top so a compare doesn't have to
    // read-modify-write a packed status word.
    u16 fpu_sw_cc;

    u16 _pad_fpu;
};

/// Reason the JIT exited and returned control to the dispatcher / host.
enum class ExitReason : u32 {
    /// Normal block boundary; dispatcher should look up the next block
    /// and continue.
    BlockEnd = 0,

    /// Guest executed a HALT-equivalent (typically int3 at the
    /// kernel entry-exit shim). Host should not re-enter.
    Halt = 1,

    /// Lifter encountered an instruction it cannot translate. Host
    /// should log the guest RIP and the un-lifted bytes, then either
    /// fall back to an interpreter (future work) or terminate.
    UnsupportedInstruction = 2,

    /// A helper-call out of the JIT returned a status indicating the
    /// guest should stop (e.g. an HLE library said "exit").
    HelperRequestedExit = 3,

    /// Reserved for asynchronous events (debugger break-in, etc.).
    AsyncBreak = 4,
};

/// Backend-visible field offsets. These mirror offsetof() but are
/// expressed as constants so the assembly gateways can include this
/// header and not have to mirror the struct definition.
namespace Offsets {
constexpr u32 Gpr = 0;
constexpr u32 Rip = Gpr + 16 * 8;
constexpr u32 Rflags = Rip + 8;
constexpr u32 FlagOp = Rflags + 8;
constexpr u32 FlagLhs = FlagOp + 8; // FlagOp is u32 + u32 pad = 8 bytes
constexpr u32 FlagRhs = FlagLhs + 8;
constexpr u32 FlagResult = FlagRhs + 8;
constexpr u32 FsBase = FlagResult + 8;
constexpr u32 GsBase = FsBase + 8;
// Ymm is aligned to 32, so there may be padding between gs_base and ymm.
// Compute via static_assert after the struct is fully defined; for
// gateway code, use offsetof() rather than these constants for the
// ymm and below fields.
} // namespace Offsets

// Sanity checks: field layout MUST match Offsets:: above. If a field
// is reordered or its type changed, these fail at compile time and
// force the gateway assembly to be updated.
static_assert(offsetof(GuestState, gpr) == Offsets::Gpr);
static_assert(offsetof(GuestState, rip) == Offsets::Rip);
static_assert(offsetof(GuestState, rflags) == Offsets::Rflags);
static_assert(offsetof(GuestState, flag_op) == Offsets::FlagOp);
static_assert(offsetof(GuestState, flag_lhs) == Offsets::FlagLhs);
static_assert(offsetof(GuestState, flag_rhs) == Offsets::FlagRhs);
static_assert(offsetof(GuestState, flag_result) == Offsets::FlagResult);
static_assert(offsetof(GuestState, fs_base) == Offsets::FsBase);
static_assert(offsetof(GuestState, gs_base) == Offsets::GsBase);
static_assert(sizeof(GuestState) % 16 == 0,
              "GuestState size must be 16-byte aligned for gateway save area");

/// Magic guest address used as the sentinel return address pushed
/// onto the guest stack by CallGuest. When guest code RETs through
/// the full call chain, RSP eventually points at this value, RET
/// pops it into state.rip, and the dispatcher recognizes it as the
/// signal to return control to the host.
///
/// The value is chosen to be non-canonical (high half nonzero) and
/// far from any real guest mapping, so accidental jumps to nearby
/// addresses don't accidentally exit. The dispatcher's check is by
/// exact equality.
///
/// Tests and CallGuest both use this constant. Changing it
/// requires updating the dispatcher's recognition logic in
/// runtime.cpp.
constexpr u64 kHostReturnAddress = 0xCB'CB'CB'CB'00'00'00'00ULL;

} // namespace Core::Runtime
