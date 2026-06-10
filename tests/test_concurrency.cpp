// Concurrency test: a LOCK-prefixed read-modify-write must be emitted as a
// single atomic step, not the load-op-store the per-op handlers use. PS4 titles
// are multithreaded and shadPS4 runs each guest thread on its own host thread
// over shared guest memory, so a `lock add [shared], 1` compiled non-atomically
// loses updates under contention (broken mutexes / refcounts).
//
// The differential oracle is single-threaded and structurally cannot observe
// this, so atomicity gets its own test: many host threads, each driving a tight
// guest loop of `lock add [counter], 1` through the JIT against ONE shared
// counter, released together by a start barrier. If the emit is atomic the
// final value is exactly threads*iters; a non-atomic regression loses updates
// and the assertion fails.
//
// Why a shared arena is safe here: this is a load-op-store lifter — guest GPRs
// (including RSP) live in memory at r13+offset, so the machine rsp during JIT
// execution is the gateway frame on each *host* thread's native stack, and the
// flag round-trip's push/pop never touches shared memory. Each thread has its
// own GuestState (so r13 and the guest rflags slot are per-thread). The only
// shared mutable memory is the counter, reached solely through the atomic op.

#include <array>
#include <atomic>
#include <cstdint>
#include <cstring>
#include <thread>
#include <vector>

#include <gtest/gtest.h>

#include "common/arch.h"
#include "common/types.h"

// Runs on BOTH host backends. The arm64 gate that used to wrap this file
// existed because the arm64 lifter routed every LOCK-prefixed RMW to the
// unsupported exit; since the LSE atomics landed (lock add lowers to
// ldaddal), the contended-counter test below is exactly the cross-host
// regression net those emitters need -- the guest bytes are x86 either way.

#ifdef _WIN32
#include <windows.h>
#else
#include <sys/mman.h>
#endif

#include "core/cpu_runtime/guest_state.h"
#include "core/cpu_runtime/runtime.h"

namespace {

using Runtime = Core::Runtime::Runtime;
using GuestState = Core::Runtime::GuestState;

// RW arena, identity-mapped (guest addr == host addr): code at the base, a
// shared counter word in the data region.
class Arena {
public:
    static constexpr size_t kSize = 64 * 1024;
    static constexpr size_t kCounterOff = 8 * 1024;
    static constexpr size_t kStackTopOff = 48 * 1024;

    Arena() {
#ifdef _WIN32
        base_ = static_cast<u8*>(
            ::VirtualAlloc(nullptr, kSize, MEM_COMMIT | MEM_RESERVE, PAGE_READWRITE));
#else
        void* p = ::mmap(nullptr, kSize, PROT_READ | PROT_WRITE, MAP_PRIVATE | MAP_ANON, -1, 0);
        base_ = (p == MAP_FAILED) ? nullptr : static_cast<u8*>(p);
#endif
    }
    ~Arena() {
        if (!base_)
            return;
#ifdef _WIN32
        ::VirtualFree(base_, 0, MEM_RELEASE);
#else
        ::munmap(base_, kSize);
#endif
    }
    Arena(const Arena&) = delete;
    Arena& operator=(const Arena&) = delete;

    bool valid() const {
        return base_ != nullptr;
    }
    u8* code() const {
        return base_;
    }
    u64 code_addr() const {
        return reinterpret_cast<u64>(base_);
    }
    u64 counter_addr() const {
        return reinterpret_cast<u64>(base_ + kCounterOff);
    }
    u64 stack_top() const {
        return reinterpret_cast<u64>(base_ + kStackTopOff);
    }
    volatile std::uint64_t* counter() const {
        return reinterpret_cast<volatile std::uint64_t*>(base_ + kCounterOff);
    }

private:
    u8* base_ = nullptr;
};

// Guest loop:  L: lock add qword [rax], 1 ; dec rcx ; jnz L ; bsr rbx, rax
//   rax = counter address (shared), rcx = iteration count (per thread).
// The trailing BSR is the canonical unsupported terminator: it ends the block
// without perturbing guest state, so Run() returns cleanly after the loop.
constexpr u8 kLoop[] = {
    0xF0, 0x48, 0x83, 0x00, 0x01, // lock add qword [rax], 1
    0x48, 0xFF, 0xC9,             // dec rcx
    0x75, 0xF6,                   // jnz -10  (back to the lock add)
    0x48, 0x0F, 0xBD, 0xD8,       // bsr rbx, rax  (terminator)
};

// Run the loop once on the calling thread: rax->counter, rcx=iters.
void RunLoop(Runtime& rt, u64 code_addr, u64 counter_addr, u64 stack_top, u64 iters) {
    GuestState st{};
    st.gpr[0] = counter_addr; // rax
    st.gpr[1] = iters;        // rcx
    st.gpr[4] = stack_top;    // rsp (unused by the snippet; set for sanity)
    st.rip = code_addr;
    st.rflags = 0x2; // reserved bit set; clear status flags
    rt.Run(st);
}

} // namespace

TEST(Concurrency, LockAddIsAtomic) {
    Arena arena;
    ASSERT_TRUE(arena.valid()) << "arena allocation failed";

    std::memcpy(arena.code(), kLoop, sizeof(kLoop));

    Runtime rt;
    const u64 code_addr = arena.code_addr();
    const u64 counter_addr = arena.counter_addr();
    const u64 stack_top = arena.stack_top();

    // Prime: compile the block once, single-threaded, so the worker threads
    // only ever execute it (first-compile happens under the cache lock, but we
    // keep it out of the timed/contended section). Verify it actually counts.
    *arena.counter() = 0;
    RunLoop(rt, code_addr, counter_addr, stack_top, 5);
    ASSERT_EQ(*arena.counter(), 5u) << "primed lock-add loop did not count correctly";

    // Contended run: all threads released together hammer the one counter.
    constexpr int kThreads = 8;
    constexpr u64 kIters = 50000;
    *arena.counter() = 0;

    std::atomic<int> ready{0};
    std::atomic<bool> go{false};
    std::vector<std::thread> workers;
    workers.reserve(kThreads);
    for (int t = 0; t < kThreads; ++t) {
        workers.emplace_back([&] {
            ready.fetch_add(1, std::memory_order_acq_rel);
            while (!go.load(std::memory_order_acquire)) { /* spin to the barrier */
            }
            RunLoop(rt, code_addr, counter_addr, stack_top, kIters);
        });
    }
    while (ready.load(std::memory_order_acquire) < kThreads) { /* wait for spawn */
    }
    go.store(true, std::memory_order_release);
    for (auto& w : workers)
        w.join();

    EXPECT_EQ(*arena.counter(), static_cast<std::uint64_t>(kThreads) * kIters)
        << "lost updates under contention: LOCK-prefixed add was not emitted atomically";
}


