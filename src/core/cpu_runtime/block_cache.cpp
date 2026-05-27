// SPDX-FileCopyrightText: Copyright 2026 shadPS4 Emulator Project
// SPDX-License-Identifier: GPL-2.0-or-later

#include "core/cpu_runtime/block_cache.h"

#include "common/logging/log.h"

namespace Core::Runtime {

BlockCache::BlockCache() {
    // Reserve enough buckets to avoid early rehashing. A typical PS4
    // game executes 10s-100s of thousands of unique guest blocks
    // across its lifetime; reserving 64K up front prevents the first
    // dozen rehashes during early game startup, when block compilation
    // pressure is highest.
    map_.reserve(64 * 1024);
}

BlockCache::~BlockCache() = default;

void* BlockCache::Lookup(u64 guest_rip) const noexcept {
    std::shared_lock lock{mutex_};
    const auto it = map_.find(guest_rip);
    return (it != map_.end()) ? it->second : nullptr;
}

void BlockCache::Insert(u64 guest_rip, void* host_ptr) {
    std::unique_lock lock{mutex_};
    // Last-write-wins. In the normal flow a guest RIP is compiled
    // exactly once, so collisions are rare. If we ever see one, the
    // most likely cause is two threads racing to compile the same
    // block — both produced valid code, and either is fine to keep.
    // We log at debug level to surface the case without polluting
    // normal logs.
    const auto [_, inserted] = map_.insert_or_assign(guest_rip, host_ptr);
    if (!inserted) {
        LOG_DEBUG(Core, "BlockCache: duplicate insert for RIP {:#x}", guest_rip);
    }
}

u32 BlockCache::InvalidateRange(u64 start_rip, u64 end_rip) {
    std::unique_lock lock{mutex_};
    u32 removed = 0;
    // robin_map doesn't have a range-erase, and the storage is
    // unordered, so we have to iterate. For typical SMC patterns the
    // affected range is small and this is acceptable. If we ever need
    // to invalidate large regions cheaply, we'd add a side index
    // (e.g. a page-keyed sorted map) — but that's not on the path
    // for the initial PR.
    for (auto it = map_.begin(); it != map_.end();) {
        if (it->first >= start_rip && it->first < end_rip) {
            it = map_.erase(it);
            ++removed;
        } else {
            ++it;
        }
    }
    if (removed > 0) {
        LOG_DEBUG(Core, "BlockCache: invalidated {} block(s) in [{:#x}, {:#x})", removed, start_rip,
                  end_rip);
    }
    return removed;
}

void BlockCache::Clear() {
    std::unique_lock lock{mutex_};
    LOG_DEBUG(Core, "BlockCache: cleared {} block(s)", map_.size());
    map_.clear();
}

u64 BlockCache::Size() const noexcept {
    std::shared_lock lock{mutex_};
    return static_cast<u64>(map_.size());
}

} // namespace Core::Runtime
