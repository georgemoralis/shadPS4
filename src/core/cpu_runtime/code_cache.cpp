// SPDX-FileCopyrightText: Copyright 2026 shadPS4 Emulator Project
// SPDX-License-Identifier: GPL-2.0-or-later
#include "core/cpu_runtime/code_cache.h"
#ifdef _WIN32
#include <windows.h>
#else
#include <sys/mman.h>
#endif

namespace Core::Runtime {

CodeCache::CodeCache(u64 size) : size_(size) {
#ifdef _WIN32
    base_ = static_cast<u8*>(VirtualAlloc(nullptr, size, MEM_RESERVE | MEM_COMMIT,
                                          PAGE_EXECUTE_READWRITE));
#else
    base_ = static_cast<u8*>(mmap(nullptr, size, PROT_READ | PROT_WRITE | PROT_EXEC,
                                  MAP_PRIVATE | MAP_ANONYMOUS, -1, 0));
    if (base_ == MAP_FAILED) base_ = nullptr;
#endif
    cur_ = base_;
    end_ = base_ ? base_ + size : nullptr;
}

CodeCache::~CodeCache() {
    if (!base_) return;
#ifdef _WIN32
    VirtualFree(base_, 0, MEM_RELEASE);
#else
    munmap(base_, size_);
#endif
}

u8* CodeCache::Allocate(u64 cap) noexcept {
    if (!cur_ || cur_ + cap > end_) return nullptr;
    u8* p = cur_;
    cur_ += cap;
    return p;
}

void CodeCache::ReturnTail(u8* buf, u64 cap, u64 used) noexcept {
    if (buf + cap == cur_) cur_ = buf + used;  // only the most-recent allocation
}

}  // namespace Core::Runtime
