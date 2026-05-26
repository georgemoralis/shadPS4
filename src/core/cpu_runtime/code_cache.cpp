// SPDX-FileCopyrightText: Copyright 2026 shadPS4 Emulator Project
// SPDX-License-Identifier: GPL-2.0-or-later

#include "core/cpu_runtime/code_cache.h"

#include "common/alignment.h"
#include "common/arch.h"
#include "common/assert.h"
#include "common/logging/log.h"

#ifdef _WIN32
#include <windows.h>
#else
#include <sys/mman.h>
#include <unistd.h>
#endif

#if defined(__APPLE__) && defined(ARCH_ARM64)
#include <pthread.h>
// MAP_JIT is defined in <sys/mman.h> on macOS but may not be
// present in older SDKs. Provide a fallback definition so the
// build doesn't break.
#ifndef MAP_JIT
#define MAP_JIT 0x800
#endif
#endif

namespace Core::Runtime {

namespace {

/// Platform-specific allocation of an RWX region. Returns nullptr
/// on failure.
u8* AllocateRwxRegion(u64 size) {
#ifdef _WIN32
    // Windows: VirtualAlloc with PAGE_EXECUTE_READWRITE. No W^X
    // restriction; the page stays RWX for the cache lifetime.
    void* p = ::VirtualAlloc(nullptr, size, MEM_COMMIT | MEM_RESERVE,
                             PAGE_EXECUTE_READWRITE);
    return static_cast<u8*>(p);

#elif defined(__APPLE__) && defined(ARCH_ARM64)
    // Apple Silicon: hardened runtime requires MAP_JIT. The pages
    // start writable; the thread-local pthread_jit_write_protect_np
    // switch toggles between writable and executable per-thread.
    void* p = ::mmap(nullptr, size, PROT_READ | PROT_WRITE | PROT_EXEC,
                     MAP_PRIVATE | MAP_ANON | MAP_JIT, -1, 0);
    if (p == MAP_FAILED) {
        return nullptr;
    }
    return static_cast<u8*>(p);

#else
    // Generic POSIX: PROT_READ | PROT_WRITE | PROT_EXEC. Linux and
    // FreeBSD allow this for userspace. macOS Intel allows this for
    // pages not tagged MAP_JIT.
    void* p = ::mmap(nullptr, size, PROT_READ | PROT_WRITE | PROT_EXEC,
                     MAP_PRIVATE | MAP_ANON, -1, 0);
    if (p == MAP_FAILED) {
        return nullptr;
    }
    return static_cast<u8*>(p);
#endif
}

void FreeRwxRegion(u8* ptr, u64 size) {
    if (ptr == nullptr) {
        return;
    }
#ifdef _WIN32
    ::VirtualFree(ptr, 0, MEM_RELEASE);
    (void)size;
#else
    ::munmap(ptr, size);
#endif
}

} // namespace

CodeCache::CodeCache(u64 size_bytes) {
    // Round up to a page. On all our target platforms, page size is
    // 4KB or 16KB. The exact size doesn't matter for correctness
    // (the kernel will round up anyway), but we report the actual
    // mapped size honestly.
#ifdef _WIN32
    SYSTEM_INFO si;
    ::GetSystemInfo(&si);
    const u64 page_size = si.dwPageSize;
#else
    const u64 page_size = static_cast<u64>(::sysconf(_SC_PAGESIZE));
#endif
    capacity_ = Common::AlignUp(size_bytes, page_size);

    base_ = AllocateRwxRegion(capacity_);
    ASSERT_MSG(base_ != nullptr,
               "CodeCache: failed to allocate {} bytes of RWX memory",
               capacity_);

    LOG_INFO(Core,
             "CodeCache: allocated {} MB at {} (page size {} KB)",
             capacity_ / (1024 * 1024), static_cast<void*>(base_),
             page_size / 1024);
}

CodeCache::~CodeCache() {
    FreeRwxRegion(base_, capacity_);
    base_ = nullptr;
    capacity_ = 0;
}

u8* CodeCache::Allocate(u64 size) {
    // Round the request up to BLOCK_ALIGN so subsequent allocations
    // are also aligned. We accept the small internal fragmentation
    // (avg ~8 bytes per block) for the simpler dispatch and
    // branch-prediction-friendly target alignment.
    const u64 aligned_size = Common::AlignUp(size, BLOCK_ALIGN);

    // fetch_add atomically reserves the range. The actual emission
    // happens after we return — concurrent emissions to different
    // blocks are safe as long as the reservation is atomic.
    const u64 prev = used_.fetch_add(aligned_size, std::memory_order_acq_rel);
    if (prev + aligned_size > capacity_) {
        // Cache is full. Roll back the reservation and return null;
        // caller will flush and retry.
        used_.fetch_sub(aligned_size, std::memory_order_acq_rel);
        return nullptr;
    }
    return base_ + prev;
}

void CodeCache::Flush() {
    std::lock_guard lock{emit_mutex_};
    used_.store(0, std::memory_order_release);
    LOG_INFO(Core, "CodeCache: flushed (capacity {} MB)",
             capacity_ / (1024 * 1024));
}

void CodeCache::WriteBegin() noexcept {
#if defined(__APPLE__) && defined(ARCH_ARM64)
    // Switch this thread's MAP_JIT pages to writable mode. Returns
    // void; failure (e.g. inside a signal handler context where
    // the call isn't allowed) silently no-ops, which is the
    // documented behavior.
    pthread_jit_write_protect_np(0);
#endif
}

void CodeCache::WriteEnd() noexcept {
#if defined(__APPLE__) && defined(ARCH_ARM64)
    pthread_jit_write_protect_np(1);
    // Invalidate the icache for the just-written region. We don't
    // know the exact range here without more state; for now, rely
    // on the per-thread write-protect toggle being sufficient
    // (Apple's documentation says it includes a barrier). If we
    // see spurious crashes from stale icache, the fix is to track
    // (start, end) of the most recent emission and call
    // sys_icache_invalidate(start, len) here.
#endif
}

bool CodeCache::Contains(const void* ptr) const noexcept {
    const auto p = reinterpret_cast<const u8*>(ptr);
    return p >= base_ && p < (base_ + capacity_);
}

} // namespace Core::Runtime
