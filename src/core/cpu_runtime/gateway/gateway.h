// SPDX-FileCopyrightText: Copyright 2026 shadPS4 Emulator Project
// SPDX-License-Identifier: GPL-2.0-or-later

#pragma once

#include "common/types.h"

namespace Core::Runtime {

struct GuestState;

/// Function-pointer type for the block dispatcher. Given the current
/// GuestState (carrying the next guest RIP in `state.rip`), the
/// dispatcher returns the host code pointer for that guest block.
/// Returning nullptr signals "compile this block now and try again",
/// which the gateway handles by calling into C++ compile and re-
/// dispatching.
///
/// The dispatcher is called from JIT code via the pinned r12 register
/// (System V ABI on x86 host). The pinning means the JIT never has to
/// look it up; one indirect call gets to the right place.
using DispatcherFn = void* (*)(GuestState* state);

/// Gateway: enters JIT execution starting at the host code pointer
/// returned by dispatching the current GuestState's RIP. Runs until
/// JIT code triggers an exit (sets state.exit_reason and jumps to
/// the gateway's exit stub).
///
/// Threading: gateway state lives on the C stack (callee-saved regs
/// are pushed there). Multiple guest threads each call this with
/// their own GuestState; no global state is involved.
///
/// Implementation lives in gateway_x86.cpp / gateway_arm64.cpp. The
/// gateway is generated at runtime via xbyak rather than written as
/// inline asm; this lets it tightly coordinate with the JIT's
/// register-pinning conventions without depending on compiler-
/// specific asm syntax.
class Gateway {
public:
    Gateway();
    ~Gateway();

    Gateway(const Gateway&) = delete;
    Gateway& operator=(const Gateway&) = delete;

    /// Enter JIT execution. Returns when JIT code exits (either
    /// normally or due to an unsupported instruction / asynchronous
    /// break). On return, `state.exit_reason` indicates why; `state.rip`
    /// points to the next un-executed guest instruction.
    void Enter(GuestState& state, DispatcherFn dispatcher);

private:
    // Function pointer to the generated gateway entry stub.
    using EntryFn = void (*)(GuestState* state, DispatcherFn dispatcher);
    EntryFn entry_ = nullptr;

    // The generated code lives in its own small RWX region rather
    // than in the main code cache, so it survives code cache
    // flushes. (Flushing the gateway would be catastrophic.)
    u8* gateway_code_ = nullptr;
    u64 gateway_size_ = 0;
};

} // namespace Core::Runtime
