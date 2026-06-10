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
    void* p = ::VirtualAlloc(nullptr, size, MEM_COMMIT | MEM_RESERVE, PAGE_EXECUTE_READWRITE);
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
    void* p =
        ::mmap(nullptr, size, PROT_READ | PROT_WRITE | PROT_EXEC, MAP_PRIVATE | MAP_ANON, -1, 0);
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

#ifdef _WIN32

// =============================================================================
// Windows unwind info registration for JIT-emitted code blocks.
// =============================================================================
//
// The gateway has its own UNWIND_INFO (gateway_x86.cpp) describing
// its 8-push prologue. JIT-emitted blocks are different: they have
// NO prologue, are jumped to (not called) from the gateway, and
// don't establish a stack frame.
//
// From the OS unwinder's perspective they're "leaf functions". We
// register a callback for the entire code cache range that returns
// a static UNWIND_INFO declaring no prologue and no register saves.
// This is sufficient to prevent the unwinder from FAULTING when it
// encounters a RIP in JIT code (the original failure mode).
//
// CAVEAT: Because blocks are jumped to (not called), [rsp] at block
// entry does NOT hold a real return address — it holds the last
// register the gateway pushed (r15). An SEH walk that STARTS inside
// a JIT block will therefore pop a garbage "return address" and
// either fault on the next lookup or chase nonsense. The intent of
// this layer is to handle the common case (walks PASSING THROUGH
// JIT code on their way up from a deeper C++ frame). A fully
// correct fix (UNW_FLAG_CHAININFO pointing to the gateway's unwind
// info, or restructuring the gateway↔block transition to use real
// call/ret) is deferred.

namespace {

// UNWIND_INFO for a "no-prologue leaf function". version=1, no
// flags, zero prologue size, zero codes. Layout matches winnt.h's
// _UNWIND_INFO with an empty UNWIND_CODE array. 4 bytes total.
struct UnwindInfoJitBlock {
    u8 version : 3;        // = 1
    u8 flags : 5;          // = 0 (no exception handler, no chained)
    u8 size_of_prolog;     // = 0
    u8 count_of_codes;     // = 0
    u8 frame_register : 4; // = 0
    u8 frame_offset : 4;   // = 0
    // No UNWIND_CODE entries; size_of_prolog and count_of_codes are
    // both zero so the structure ends here.
};
static_assert(sizeof(UnwindInfoJitBlock) == 4, "UNWIND_INFO header must be exactly 4 bytes");

// Layout inside the code cache mapping:
//   [0 .. capacity_)            — user-visible JIT code region
//   [capacity_ .. capacity_+4K) — reserved for unwind metadata
//
// The UNWIND_INFO sits at offset capacity_ from base_. The
// RUNTIME_FUNCTION is returned dynamically via the callback (per
// thread).
constexpr u64 WIN_UNWIND_RESERVE = 4096;

// The OS retrieves a RUNTIME_FUNCTION* by calling our callback. The
// returned pointer must remain valid until the OS finishes using it.
// In-process unwind walks happen synchronously on the unwinding
// thread, so per-thread storage is sufficient — thread_local here
// lets concurrent threads unwind independently without locking.
thread_local RUNTIME_FUNCTION tls_runtime_function;

/// Callback registered with `RtlInstallFunctionTableCallback`. The
/// OS calls this when it needs unwind info for a RIP in our code
/// cache range. Returns a pointer to a per-thread RUNTIME_FUNCTION
/// describing the JIT block at ControlPc.
///
/// We treat the entire code cache as one big "function" with no
/// prologue. The OS guarantees ControlPc is within the registered
/// range; we just fill in a RUNTIME_FUNCTION pointing at our
/// pre-written UNWIND_INFO and return.
PRUNTIME_FUNCTION CALLBACK GetRuntimeFunctionForJitBlock(DWORD64 ControlPc, PVOID Context) {
    auto* cc = static_cast<const CodeCache*>(Context);
    const auto cap = static_cast<DWORD>(cc->Capacity());

    // ControlPc isn't used — we return the same coverage range for
    // every PC in our region. Silence the "unused" warning.
    (void)ControlPc;

    tls_runtime_function.BeginAddress = 0;
    tls_runtime_function.EndAddress = cap;
    tls_runtime_function.UnwindInfoAddress = cap;
    return &tls_runtime_function;
}

/// Write the static UNWIND_INFO at base+capacity, then register the
/// callback for the full code cache range. Returns true on success.
bool InstallJitBlockUnwindCallback(const CodeCache& cc) {
    u8* base = const_cast<u8*>(cc.Base());
    const u64 cap = cc.Capacity();

    auto* ui = reinterpret_cast<UnwindInfoJitBlock*>(base + cap);
    *ui = {};
    ui->version = 1;
    ui->flags = 0;
    ui->size_of_prolog = 0;
    ui->count_of_codes = 0;
    ui->frame_register = 0;
    ui->frame_offset = 0;

    // TableIdentifier (per RtlInstallFunctionTableCallback docs): the
    // low 2 bits must be set. Use base|3 — easy to reconstruct on
    // the delete side.
    const DWORD64 table_id = reinterpret_cast<DWORD64>(base) | 0x3ULL;
    const DWORD range = static_cast<DWORD>(cap + WIN_UNWIND_RESERVE);

    return ::RtlInstallFunctionTableCallback(
               table_id, reinterpret_cast<DWORD64>(base), range, &GetRuntimeFunctionForJitBlock,
               const_cast<void*>(static_cast<const void*>(&cc)), nullptr) != FALSE;
}

void RemoveJitBlockUnwindCallback(const CodeCache& cc) {
    const DWORD64 table_id = reinterpret_cast<DWORD64>(cc.Base()) | 0x3ULL;
    ::RtlDeleteFunctionTable(reinterpret_cast<PRUNTIME_FUNCTION>(table_id));
}

} // namespace

#endif // _WIN32

namespace {

// Augmented allocation size: the user-visible capacity plus a page
// of reserved space (on Windows only) for unwind metadata. The
// reserve is intentionally outside `capacity_` so Allocate() and
// Contains() naturally exclude it without any extra checks.
#ifdef _WIN32
constexpr u64 PLATFORM_METADATA_RESERVE = 4096;
#else
constexpr u64 PLATFORM_METADATA_RESERVE = 0;
#endif

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

    // Allocate the user-visible capacity PLUS the platform reserve
    // (a 4KB page on Windows for unwind metadata, 0 elsewhere). The
    // reserve lives just past `capacity_`, so Allocate() and
    // Contains() naturally exclude it without checks.
    const u64 total_size = capacity_ + PLATFORM_METADATA_RESERVE;
    base_ = AllocateRwxRegion(total_size);
    ASSERT_MSG(base_ != nullptr, "CodeCache: failed to allocate {} bytes of RWX memory",
               total_size);

    LOG_INFO(Core, "CodeCache: allocated {} MB at {} (page size {} KB)", capacity_ / (1024 * 1024),
             static_cast<void*>(base_), page_size / 1024);

#ifdef _WIN32
    // Register the JIT block unwind callback so the OS unwinder can
    // walk through JIT-emitted code without faulting on missing
    // metadata. See InstallJitBlockUnwindCallback for the full
    // story and limitations.
    if (!InstallJitBlockUnwindCallback(*this)) {
        LOG_WARNING(Core, "CodeCache: RtlInstallFunctionTableCallback failed; "
                          "SEH walks through JIT blocks may crash. The gateway "
                          "unwind info is still installed, so walks that don't "
                          "originate inside JIT code should remain OK.");
    }
#endif
}

CodeCache::~CodeCache() {
#ifdef _WIN32
    if (base_ != nullptr) {
        RemoveJitBlockUnwindCallback(*this);
    }
#endif
    FreeRwxRegion(base_, capacity_ + PLATFORM_METADATA_RESERVE);
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

bool CodeCache::ReturnTail(const u8* block_base, u64 reserved, u64 used) noexcept {
    const u64 begin = static_cast<u64>(block_base - base_);
    // Mirror Allocate's rounding on both figures: the reservation was
    // bumped by AlignUp(reserved), and the new top must keep the next
    // block's BLOCK_ALIGN guarantee.
    const u64 top_if_last = begin + Common::AlignUp(reserved, BLOCK_ALIGN);
    const u64 new_top = begin + Common::AlignUp(used, BLOCK_ALIGN);
    if (new_top >= top_if_last) {
        return false; // nothing to reclaim
    }
    u64 expected = top_if_last;
    // CAS: only shrink if we are still the top allocation. Failure means a
    // concurrent Allocate (or its overflow back-out window) moved the bump;
    // the tail is then unreachable until the next Flush -- accepted.
    return used_.compare_exchange_strong(expected, new_top,
                                         std::memory_order_acq_rel,
                                         std::memory_order_relaxed);
}

void CodeCache::Flush() {
    std::lock_guard lock{emit_mutex_};
    used_.store(0, std::memory_order_release);
    LOG_INFO(Core, "CodeCache: flushed (capacity {} MB)", capacity_ / (1024 * 1024));
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
