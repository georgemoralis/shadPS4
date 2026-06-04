// SPDX-FileCopyrightText: Copyright 2024-2026 shadPS4 Emulator Project
// SPDX-License-Identifier: GPL-2.0-or-later

#pragma once

#include <set>
#include <signal.h>
#include "common/singleton.h"
#include "common/types.h"

#ifdef _WIN32
#define SIGSLEEP -1
#else
#define SIGSLEEP SIGVTALRM
#endif
namespace Core {

using AccessViolationHandler = bool (*)(void* context, void* fault_address);
using IllegalInstructionHandler = bool (*)(void* context);

#ifndef _WIN32
void SignalHandler(int sig, siginfo_t* info, void* raw_context);
#endif

// Ensure the calling thread has its dedicated vectored-exception-handler stack
// allocated. On Windows the VEH runs on the faulting (guest) stack, which under
// the JIT can be too deep/exhausted to host the handler's own frames; calling
// this from the JIT entry (Runtime::Run) in a healthy-stack context pre-
// allocates a dedicated stack the handler switches onto. Idempotent per thread.
// No-op on non-Windows (POSIX uses an alternate signal stack via SA_ONSTACK).
#ifdef _WIN32
void EnsureVehStack();
#else
inline void EnsureVehStack() {}
#endif

/// Receives OS signals and dispatches to the appropriate handlers.
class SignalDispatch {
public:
    SignalDispatch();
    ~SignalDispatch();

    /// Registers a handler for memory access violation signals.
    void RegisterAccessViolationHandler(const AccessViolationHandler& handler, u32 priority) {
        access_violation_handlers.emplace(handler, priority);
    }

    /// Registers a handler for illegal instruction signals.
    void RegisterIllegalInstructionHandler(const IllegalInstructionHandler& handler, u32 priority) {
        illegal_instruction_handlers.emplace(handler, priority);
    }

    /// Dispatches an access violation signal, returning whether it was successfully handled.
    bool DispatchAccessViolation(void* context, void* fault_address) const;

    /// Dispatches an illegal instruction signal, returning whether it was successfully handled.
    bool DispatchIllegalInstruction(void* context) const;

private:
    template <typename T>
    struct HandlerEntry {
        T handler;
        u32 priority;

        std::strong_ordering operator<=>(const HandlerEntry& right) const {
            return priority <=> right.priority;
        }
    };
    std::set<HandlerEntry<AccessViolationHandler>> access_violation_handlers;
    std::set<HandlerEntry<IllegalInstructionHandler>> illegal_instruction_handlers;

#ifdef _WIN32
    void* handle{};
#endif
};

using Signals = Common::Singleton<SignalDispatch>;

} // namespace Core