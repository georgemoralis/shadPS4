// SPDX-FileCopyrightText: Copyright 2026 shadPS4 Emulator Project
// SPDX-License-Identifier: GPL-2.0-or-later

#pragma once

#include <memory>
#include "common/types.h"
#include "core/cpu_runtime/guest_state.h"

namespace Core::Runtime {

class BlockCache;
class CodeCache;
class Gateway;
class Lifter;

/// Public entry point for the CPU runtime. Created once during
/// emulator init, used by Linker::Execute as the alternative to
/// native execution when SHADPS4_CPU_BACKEND=runtime.
///
/// Threading: methods on this class are thread-safe. Multiple guest
/// threads dispatching through Run() share the block cache and code
/// cache; concurrent compilation of distinct blocks works without
/// additional synchronization on the caller's part.
class Runtime {
public:
    Runtime();
    ~Runtime();

    Runtime(const Runtime&) = delete;
    Runtime& operator=(const Runtime&) = delete;

    /// Execute guest code starting at the given guest RIP, with the
    /// provided initial GuestState. Returns when the guest hits a
    /// halt-equivalent (typically the kernel's exit shim), at which
    /// point the state's exit_reason is set and state.rip points to
    /// the next un-executed guest instruction (or 0 if cleanly
    /// terminated).
    ///
    /// This call doesn't return until the guest is done. For typical
    /// game workloads that's the lifetime of the game; for the
    /// future debugger path, the runtime supports asynchronous
    /// break-in via AsyncBreak() which causes this to return early.
    void Run(GuestState& state);

    /// Request that the currently-executing Run() return at the next
    /// safe point (block boundary). May be called from another
    /// thread. Idempotent.
    void AsyncBreak();

    /// Diagnostic accessors. Provided for debugging and for the
    /// devtools widget that displays runtime state.
    [[nodiscard]] const BlockCache& GetBlockCache() const noexcept {
        return *block_cache_;
    }
    [[nodiscard]] const CodeCache& GetCodeCache() const noexcept {
        return *code_cache_;
    }

    /// Internal use by the dispatcher trampoline: compile a block at
    /// the given guest RIP and return its host code pointer. Exposed
    /// because the trampoline is a free function (gateway signature
    /// constraints) and needs access to the lifter.
    void* CompileBlockForDispatcher(u64 guest_rip);

private:
    std::unique_ptr<BlockCache> block_cache_;
    std::unique_ptr<CodeCache> code_cache_;
    std::unique_ptr<Gateway> gateway_;
    std::unique_ptr<Lifter> lifter_;
};

} // namespace Core::Runtime
