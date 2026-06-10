// SPDX-FileCopyrightText: Copyright 2026 shadPS4 Emulator Project
// SPDX-License-Identifier: GPL-2.0-or-later

#pragma once

#include <atomic>
#include <cstddef>
#include <mutex>
#include "common/types.h"

namespace Core::Runtime {

/// Bump-allocated RWX memory pool that JIT-emitted host code lives in.
///
/// Lifecycle: allocated once at runtime init, freed at runtime
/// shutdown. Blocks are never individually freed — the cache is
/// either entirely valid (normal operation) or entirely flushed
/// (cache invalidation, debugging). This is a deliberate trade-off:
/// individual block freeing would require maintaining a free list,
/// which adds complexity and fragmentation risk, for a benefit
/// (memory reclamation) that mostly doesn't matter — the entire
/// pool fits comfortably in a couple hundred megabytes for typical
/// guest workloads.
///
/// Platform notes:
///
///   - On x86 Linux/Windows/macOS: pages are mapped RWX once and
///     stay that way. Cheapest possible path.
///
///   - On Apple Silicon (ARM64 macOS): hardened runtime restricts
///     mapping memory both writable and executable simultaneously
///     (W^X). The Apple-blessed approach uses MAP_JIT plus the
///     pthread_jit_write_protect_np per-thread switch — code is
///     writable from the compiler thread and executable from the
///     dispatcher thread. The interface below has WriteBegin /
///     WriteEnd hooks for this; on platforms where W^X isn't
///     enforced they're no-ops.
///
///   - On ARM64 Linux: PROT_EXEC + PROT_WRITE works (kernel doesn't
///     enforce W^X for userspace).
class CodeCache {
public:
    /// Initial / maximum size for the code cache. 64MB is enough for
    /// the hottest game workloads we've seen in similar emulators;
    /// can be tuned up later if needed. We do NOT grow the cache —
    /// on overflow, we flush and start over (correctness preserved
    /// because the block cache is also flushed).
    static constexpr u64 DEFAULT_SIZE = 64 * 1024 * 1024;

    /// Code cache alignment for individual blocks. 16 bytes matches
    /// the icache line size on most x86 hosts and the optimal branch
    /// target alignment on ARM64.
    static constexpr u32 BLOCK_ALIGN = 16;

    explicit CodeCache(u64 size_bytes = DEFAULT_SIZE);
    ~CodeCache();

    CodeCache(const CodeCache&) = delete;
    CodeCache& operator=(const CodeCache&) = delete;

    /// Reserve `size` bytes of code-cache memory for emission. Returns
    /// nullptr if the cache is full — caller should Flush() and retry
    /// (which invalidates all currently-compiled blocks).
    ///
    /// The returned pointer is aligned to BLOCK_ALIGN. Subsequent
    /// allocations are also aligned (the allocator rounds up).
    [[nodiscard]] u8* Allocate(u64 size);

    /// Return the unused tail of the MOST RECENT allocation to the pool.
    ///
    /// The lifter reserves a fixed per-block cap up front (it cannot know
    /// the emitted size before emitting) and calls this with the actual
    /// byte count afterwards. Without it, every block burned the full cap:
    /// at a 16 KiB cap the 64 MiB cache held at most 4096 blocks while
    /// typical blocks emit well under 1 KiB -- a 20-50x effective-capacity
    /// loss paid for in stop-the-world recycles.
    ///
    /// Concurrency: compiles run in parallel, so the tail can only be
    /// reclaimed if this allocation is still the TOP of the bump -- a
    /// single CAS from (block start + reserved) down to (block start +
    /// aligned used). If another thread allocated in between, the CAS
    /// fails and the tail is simply wasted (the pre-ReturnTail behavior;
    /// now the rare interleaved case instead of every block). A concurrent
    /// failing Allocate's inflate/deflate window likewise just fails the
    /// CAS. Returns whether the tail was reclaimed; callers need not care.
    ///
    /// `block_base` must be a pointer returned by Allocate(`reserved`),
    /// with `used` <= `reserved` bytes actually written.
    bool ReturnTail(const u8* block_base, u64 reserved, u64 used) noexcept;

    /// Reset the bump pointer to the start of the cache. All
    /// previously-allocated regions become invalid; the caller is
    /// responsible for ensuring no host code is executing in the
    /// cache when this is called (typically by invalidating the
    /// BlockCache before / after).
    void Flush();

    /// W^X support. On platforms that enforce it, code emission
    /// must be bracketed: WriteBegin() switches the cache to
    /// writable, WriteEnd() switches it back to executable. On
    /// platforms without W^X, both are no-ops.
    ///
    /// Calls are per-thread. The compiler thread brackets every
    /// emission with WriteBegin/WriteEnd; the dispatcher thread
    /// never calls either.
    void WriteBegin() noexcept;
    void WriteEnd() noexcept;

    /// Size and high-water-mark, for diagnostics.
    [[nodiscard]] u64 Capacity() const noexcept {
        return capacity_;
    }
    [[nodiscard]] u64 Used() const noexcept {
        return used_.load(std::memory_order_acquire);
    }

    /// Range query: is this host pointer inside the code cache?
    /// Used by the signal handler to identify whether a fault
    /// happened in JIT code (and thus needs reverse-mapping to
    /// guest RIP) vs in normal C++ code.
    [[nodiscard]] bool Contains(const void* ptr) const noexcept;

    /// Base pointer of the code cache. Diagnostic only — clients
    /// should not access this directly except for very specific
    /// purposes (e.g. perf-map files for profilers).
    [[nodiscard]] const u8* Base() const noexcept {
        return base_;
    }

private:
    u8* base_ = nullptr;
    u64 capacity_ = 0;
    std::atomic<u64> used_{0};
    std::mutex emit_mutex_;
};

} // namespace Core::Runtime
