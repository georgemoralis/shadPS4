// SPDX-FileCopyrightText: Copyright 2026 shadPS4 Emulator Project
// SPDX-License-Identifier: GPL-2.0-or-later

#pragma once

#include "common/types.h"

namespace Core::Runtime {

class CodeCache;
struct GuestState;

/// Lifter: translates one guest basic block at a time into host code,
/// emitting through the provided CodeCache.
///
/// One Lifter instance per runtime. The lifter is not stateless — it
/// holds an internal xbyak CodeGenerator and Zydis decoder context
/// that get reused across compilations. Lifter calls are serialized
/// externally (we don't compile blocks in parallel).
///
/// Per-host implementations:
///   lifter_x86_host.cpp   — x86_64 guest -> x86_64 host via xbyak
///   lifter_arm64_host.cpp — x86_64 guest -> ARM64 host via xbyak_aarch64
///
/// Both implement the same Lifter class declaration but only one is
/// compiled per build (chosen by the SHADPS4_CPU_BACKEND flag and the
/// host architecture).
class Lifter {
public:
    explicit Lifter(CodeCache& code_cache);
    ~Lifter();

    Lifter(const Lifter&) = delete;
    Lifter& operator=(const Lifter&) = delete;

    /// Compile the basic block beginning at guest_rip.
    ///
    /// Returns a host code pointer that the dispatcher can jmp into,
    /// or nullptr on failure (out of code cache, unsupported
    /// instruction at the start of the block, etc.).
    ///
    /// The compiled block updates GuestState as it executes; on
    /// reaching its terminator, it jumps either to the dispatcher
    /// (via r12) or to the exit stub (via r14, set by the gateway).
    void* CompileBlock(u64 guest_rip);

    /// Diagnostics: number of blocks compiled, number of bytes of
    /// host code emitted, number of unsupported-instruction
    /// terminations.
    [[nodiscard]] u64 BlocksCompiled() const noexcept { return blocks_compiled_; }
    [[nodiscard]] u64 BytesEmitted() const noexcept { return bytes_emitted_; }
    [[nodiscard]] u64 UnsupportedHits() const noexcept { return unsupported_hits_; }

private:
    CodeCache& code_cache_;
    u64 blocks_compiled_ = 0;
    u64 bytes_emitted_ = 0;
    u64 unsupported_hits_ = 0;
};

} // namespace Core::Runtime
