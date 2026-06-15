// SPDX-FileCopyrightText: Copyright 2026 shadPS4 Emulator Project
// SPDX-License-Identifier: GPL-2.0-or-later
#pragma once
#include "common/types.h"

namespace Core::Runtime {

// Bump-allocated executable arena for emitted blocks. Allocate(cap) reserves a
// fixed per-block chunk; the lifter commits only what it emits via ReturnTail.
// W^X note: this maps RX-writable (RWX) for simplicity; a hardened build should
// map RW, emit, then flip RX per page.
class CodeCache {
public:
    explicit CodeCache(u64 size);
    ~CodeCache();
    CodeCache(const CodeCache&) = delete;
    CodeCache& operator=(const CodeCache&) = delete;

    u8* Allocate(u64 cap) noexcept;
    void ReturnTail(u8* buf, u64 cap, u64 used) noexcept;
    u64 Free() const noexcept { return static_cast<u64>(end_ - cur_); }

private:
    u8 *base_ = nullptr, *cur_ = nullptr, *end_ = nullptr;
    u64 size_ = 0;
};

}  // namespace Core::Runtime
