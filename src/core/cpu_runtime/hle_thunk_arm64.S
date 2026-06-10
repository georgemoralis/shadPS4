// SPDX-FileCopyrightText: Copyright 2026 shadPS4 Emulator Project
// SPDX-License-Identifier: GPL-2.0-or-later

// HLE-bridge call thunk for AArch64 hosts.
//
// PROBLEM THIS SOLVES. The bridge must call a host function of UNKNOWN
// return type and recover both registers a callee may legally have written
// its result into: x0 (integer) and d0 (floating). The x86-64 path gets
// this for free from the SysV ABI ({u64; double} returns in rax:xmm0), but
// no AAPCS64 C return type splits across x0 AND d0 -- a 16-byte mixed
// composite comes back in x0:x1, an HFA in d0:d1. The previous revision
// called through a void-typed pointer and then read x0/d0 via
// register-asm locals fed by an EMPTY asm statement. That is unsound:
// nothing creates a dependency edge between the call and the asm, so the
// compiler may schedule arbitrary code between them (spills, stack-
// protector checks, sanitizer instrumentation) or reorder the asm relative
// to the call outright. It worked by accident, per compiler, per flag set.
//
// THE THUNK. C++ calls CallHostThunkA64 with the real target as the FIRST
// argument and the 22 marshalled arguments after it, so the compiler does
// all argument placement under the normal AAPCS rules -- no inline asm, no
// register variables. The thunk then shifts the integer arguments down one
// slot (undoing the displacement the prepended target caused), performs
// the call itself, and builds the {u64; double} return: AAPCS returns that
// composite in x0:x1, the callee's integer result is already in x0, and
// one fmov moves d0's raw bits into x1. The C++ caller's HostReturn then
// bit-holds exactly {callee x0, callee d0}.
//
// ARGUMENT GEOMETRY (AAPCS64; integer and FP pools fill independently):
//
//   This thunk's own frame (C++ prototype: target, a0..a5, f0..f7, s0..s7):
//     x0          = target function pointer
//     x1..x6      = a0..a5
//     x7          = s0
//     [sp+0]      = s1            (first incoming stack slot)
//     [sp+8..48]  = s2..s7
//     d0..d7      = f0..f7        (already in final position)
//
//   The target expects (the bridge's canonical 22-arg layout):
//     x0..x5      = a0..a5
//     x6          = s0
//     x7          = s1
//     [sp+0..40]  = s2..s7
//     d0..d7      = f0..f7
//
// So: shift x1..x7 down to x0..x6, pull s1 from the incoming stack into
// x7, copy the remaining six incoming stack slots to the outgoing area.
//
// KNOWN LIMITATION (pre-existing, unchanged by the thunk): variadic HLE
// targets. Apple's AArch64 ABI passes anonymous (variadic) arguments on
// the stack, not in registers, so printf-family targets called through
// this fixed-arg layout receive their variadic args in the wrong places
// on macOS regardless of how the call is made. The x86 path's AL trick
// has no AArch64 equivalent here. Variadic HLE shims need their own
// marshalling when they matter; this thunk preserves the fixed-arg
// contract the bridge has always provided.

#if defined(__aarch64__)

.text
.balign 16

#if defined(__APPLE__)
.globl _CallHostThunkA64
_CallHostThunkA64:
#else
.globl CallHostThunkA64
.type CallHostThunkA64, %function
CallHostThunkA64:
#endif
    .cfi_startproc
    // Frame: save fp/lr, establish fp, carve 48 bytes of outgoing stack
    // args. x29 stays 16-aligned; sp = x29 - 48 stays 16-aligned.
    stp     x29, x30, [sp, #-16]!
    .cfi_def_cfa_offset 16
    .cfi_offset x29, -16
    .cfi_offset x30, -8
    mov     x29, sp
    .cfi_def_cfa_register x29
    sub     sp, sp, #48

    // Target out of the way before the shuffle tramples x0. x9 is a plain
    // caller-saved temporary: not an argument register, and deliberately
    // NOT x16/x17 -- those are the IP0/IP1 intra-procedure-call scratch
    // registers that toolchains reserve for their own linkage glue (PLT
    // stubs, pointer-auth/BTI thunks) and treat as fair game around calls.
    mov     x9, x0

    // Integer args down one slot: a0..a5, then s0 into x6.
    mov     x0, x1
    mov     x1, x2
    mov     x2, x3
    mov     x3, x4
    mov     x4, x5
    mov     x5, x6
    mov     x6, x7

    // s1 was this frame's first incoming stack arg: above the saved
    // fp/lr pair, i.e. [x29 + 16].
    ldr     x7, [x29, #16]

    // s2..s7: incoming [x29 + 24 .. + 64] -> outgoing [sp + 0 .. + 40].
    ldp     x10, x11, [x29, #24]
    stp     x10, x11, [sp]
    ldp     x10, x11, [x29, #40]
    stp     x10, x11, [sp, #16]
    ldp     x10, x11, [x29, #56]
    stp     x10, x11, [sp, #32]

    // d0..d7 already hold f0..f7 untouched. Call.
    blr     x9

    // Build the {u64; double} composite return (AAPCS: x0:x1). The
    // callee's integer result is already in x0; move d0's raw bits into
    // x1 -- fmov transfers the bit pattern, no conversion.
    fmov    x1, d0

    mov     sp, x29
    .cfi_def_cfa_register sp
    ldp     x29, x30, [sp], #16
    .cfi_def_cfa_offset 0
    .cfi_restore x29
    .cfi_restore x30
    ret
    .cfi_endproc

#if !defined(__APPLE__)
.size CallHostThunkA64, . - CallHostThunkA64
// Non-executable stack note for ELF linkers.
.section .note.GNU-stack, "", %progbits
#endif

#endif // __aarch64__
