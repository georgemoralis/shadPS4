// SPDX-FileCopyrightText: Copyright 2026 shadPS4 Emulator Project
// SPDX-License-Identifier: GPL-2.0-or-later

#pragma once

#include <mutex>
#include <shared_mutex>
#include <tsl/robin_map.h>
#include "common/types.h"

namespace Core::Runtime {

/// Maps guest RIP (the address of a guest basic block's first
/// instruction) to a host pointer (the entry of the compiled host
/// code for that block).
///
/// Lookup is on the hot path — it runs once per block transition.
/// Insert is rare (happens once per block, when first compiled).
///
/// Concurrency model:
///
///   - Lookups are read-locked. Multiple guest threads may run JIT
///     code simultaneously, all dispatching through the same cache.
///   - Inserts are write-locked. A single compiler thread runs at a
///     time (we don't currently compile blocks in parallel; if that
///     changes, this is the synchronization point that needs to grow
///     up).
///   - tsl::robin_map is already a dependency of upstream (see
///     CMakeLists target_link_libraries). It has lower lookup latency
///     than std::unordered_map for our access pattern.
///
/// We deliberately do NOT use a flat array indexed by guest RIP. The
/// guest address space is 48 bits; even a 1-bit-per-page bitmap would
/// be 32GB. A hash map is the right shape.
///
/// Block invalidation: when guest code modifies a region (self-
/// modifying code, code patches, module unloading), entries in that
/// region must be removed from the cache. The current API supports
/// removal by guest-RIP range; the SMC detection mechanism that drives
/// invalidation is out of scope for this header.
class BlockCache {
public:
    BlockCache();
    ~BlockCache();

    BlockCache(const BlockCache&) = delete;
    BlockCache& operator=(const BlockCache&) = delete;

    /// Look up the host code pointer for a guest RIP. Returns nullptr
    /// if the block isn't compiled yet. This is the hot path — the
    /// dispatcher calls this once per block transition.
    [[nodiscard]] void* Lookup(u64 guest_rip) const noexcept;

    /// Record that a guest RIP was compiled to a particular host
    /// pointer. Called once per block, after the lifter emits code
    /// and the code cache hands back a pointer. The block is now
    /// dispatchable.
    void Insert(u64 guest_rip, void* host_ptr);

    /// Remove all entries whose guest RIP falls in [start, end). Used
    /// when a guest memory region is modified or unmapped and the
    /// compiled code for blocks in that region is no longer valid.
    /// Does NOT free the code cache memory — that's a separate
    /// concern (code cache is bump-allocated and only freed at
    /// runtime shutdown).
    ///
    /// Returns the number of entries removed.
    u32 InvalidateRange(u64 start_rip, u64 end_rip);

    /// Remove all entries. Used at shutdown and when the user opts to
    /// clear the cache (e.g. for debugging).
    void Clear();

    /// Number of currently-compiled blocks. Provided for diagnostics.
    [[nodiscard]] u64 Size() const noexcept;

private:
    mutable std::shared_mutex mutex_;
    tsl::robin_map<u64, void*> map_;
};

} // namespace Core::Runtime
