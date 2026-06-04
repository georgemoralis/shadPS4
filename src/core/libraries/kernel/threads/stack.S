# SPDX-FileCopyrightText: Copyright 2026 shadPS4 Emulator Project
# SPDX-License-Identifier: GPL-2.0-or-later

.global _runOnAnotherStack
_runOnAnotherStack:
  pushq %r12
  pushq %r13
#ifdef _WIN32
  pushq %r14
  pushq %r15

  # save stack check registers
  movq %gs:0x08, %r14 # teb->stack_bottom
  movq %gs:0x10, %r15 # teb->stack_top

  # disable stack checks
  xorq %rcx, %rcx
  movq %rcx, %gs:0x08 # teb->stack_bottom = 0
  movq %rcx, %gs:0x10 # teb->stack_top = 0
#endif
  movq %rsp, %r12
  movq %rbp, %r13
  movq %rdx, %rsp
  movq %rdx, %rbp
  callq *%rsi
  movq %r13, %rbp
  movq %r12, %rsp
#ifdef _WIN32
  # restore stack check registers
  movq %r14, %gs:0x08 # teb->stack_bottom
  movq %r15, %gs:0x10 # teb->stack_top
  popq %r15
  popq %r14
#endif
  popq %r13
  popq %r12
  ret

#ifdef _WIN32
# ---------------------------------------------------------------------------
# CallOnStack — run the Windows vectored exception handler on a dedicated,
# always-committed per-thread stack instead of the (possibly nearly-exhausted)
# guest stack the fault occurred on.
#
# Windows invokes a vectored exception handler on whatever stack is current at
# fault time. The JIT runs guest code on guest stacks (1MB pthread stacks w/ a
# single guard page, or smaller malloc'd callback stacks). A fault while the
# guest stack is deep leaves no room for the handler's own C++ frames (decoder,
# logging, fault-context capture), so the handler itself faults writing its
# locals off the end of the mapped stack. POSIX solves this with SA_ONSTACK /
# sigaltstack; Windows has no equivalent, so we switch stacks by hand here —
# the same spirit as _runOnAnotherStack above.
#
# Windows x64 ABI:  rcx = stack_top (16-byte region top, already committed)
#                   rdx = pExp (EXCEPTION_POINTERS*, passed through to impl)
#                   r8  = impl (LONG (*)(EXCEPTION_POINTERS*))
# Returns: impl's LONG result in eax.
#
# The whole switch is unwound before returning to Windows, so Windows sees a
# normal handler return on the original stack.
.global CallOnStack
CallOnStack:
  movq %rsp, %rax            # rax = old (guest-stack) rsp
  andq $-16, %rcx            # ensure 16-byte alignment of the new stack top
  movq %rcx, %rsp            # switch to the dedicated exception stack
  pushq %rax                 # save old rsp on the new stack
  subq $32, %rsp             # 32B shadow space; rsp is 16-aligned at the call
  movq %rdx, %rcx            # arg0 = pExp
  callq *%r8                 # impl(pExp)  -> result in eax
  addq $32, %rsp
  popq %r10                  # r10 = saved old rsp
  movq %r10, %rsp            # switch back to the original (guest) stack
  ret
#endif
