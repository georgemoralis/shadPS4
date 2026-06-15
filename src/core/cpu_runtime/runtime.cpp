// SPDX-FileCopyrightText: Copyright 2026 shadPS4 Emulator Project
// SPDX-License-Identifier: GPL-2.0-or-later
//
// runtime_copy dispatcher glue. NOTE: written against the shadPS4 tree's
// interfaces but NOT yet compiled in-tree. Integration points that must be
// confirmed against the live tree are marked "INTEGRATION:".

#include <array>
#include <atomic>
#include <cstdio>
#include <memory>
#include <unordered_map>

#include "common/types.h"
#include "core/cpu_runtime/code_cache.h"
#include "core/cpu_runtime/cpu_backend.h"
#include "core/cpu_runtime/guest_state.h"
#include "core/cpu_runtime/runtime.h"
#include "core/memory.h"          // INTEGRATION: Memory::Instance(), AddressSpace getters
#include "core/tls.h"            // Core::GetTcbBase() -> guest TLS/canary base
#include "common/logging/log.h" // bring-up tracing (Core_Linker)

namespace Core::Runtime {

// ---------------- guest VA range table / IsGuestPointer ----------------
namespace {
constexpr std::size_t kMaxRanges = 4;
std::array<GuestAddressRange, kMaxRanges> g_ranges{};
std::atomic<u32> g_range_count{0};
}

void RegisterGuestAddressRanges(const GuestAddressRange* ranges, std::size_t count) noexcept {
    const u32 n = static_cast<u32>(count < kMaxRanges ? count : kMaxRanges);
    for (u32 i = 0; i < n; ++i) g_ranges[i] = ranges[i];
    g_range_count.store(n, std::memory_order_release);
}

bool IsGuestPointer(VAddr a) noexcept {
    const u32 n = g_range_count.load(std::memory_order_acquire);
    for (u32 i = 0; i < n; ++i)
        if (a - g_ranges[i].base < g_ranges[i].size) return true;
    return false;
}

#if SHADPS4_HAVE_JIT

namespace {

void RegisterRangesFromMemory() {
    auto& as = Core::Memory::Instance()->GetAddressSpace();   // INTEGRATION: exact accessor
    const GuestAddressRange r[3] = {
        {as.SystemManagedVirtualBase(),  as.SystemManagedVirtualSize()},
        {as.SystemReservedVirtualBase(), as.SystemReservedVirtualSize()},
        {as.UserVirtualBase(),           as.UserVirtualSize()},
    };
    RegisterGuestAddressRanges(r, 3);
}

// INTEGRATION: allocate a guest stack in guest memory and return its top. Native
// mode reuses the host stack; the JIT can't (host rsp is the runtime's internal
// stack), so guest code needs its own. Uses MemoryManager.
VAddr AllocGuestStack(u64 size) {
    void* addr = nullptr;
    Core::Memory::Instance()->MapMemory(&addr, /*vaddr=*/0, size,
                                        Core::MemoryProt::CpuReadWrite,
                                        /*flags=*/{}, Core::VMAType::Flexible);
    return reinterpret_cast<VAddr>(addr) + size;   // top of stack
}

class BlockCache {
public:
    void* GetOrCompile(u64 rip, Backend& be) {
        auto it = map_.find(rip);
        if (it != map_.end()) return it->second;
        void* h = be.CompileBlock(rip);
        map_.emplace(rip, h);
        return h;
    }
private:
    std::unordered_map<u64, void*> map_;
};

// Shared JIT machinery. Lazily created on first guest entry (module_start runs
// before the main entry), then reused so the code cache, backend, and compiled
// blocks are common to every host->guest call. Ranges are registered once here.
struct JitState {
    CodeCache                code_cache;
    std::unique_ptr<Backend> backend;
    EnterBlockFn             enter{};
    HleBridgeFn              bridge{};
    BlockCache               blocks;
    VAddr                    fs_base = 0;   // INTEGRATION: see SetGuestSegmentBases
    VAddr                    gs_base = 0;
    explicit JitState(u64 cache_bytes) : code_cache(cache_bytes) {
        RegisterRangesFromMemory();
        backend = CreateActiveJitBackend(code_cache);
        enter   = backend->GetEnterBlock();
        bridge  = backend->GetHleBridge();
    }
};

JitState& Jit() {
    static JitState state{64ull * 1024 * 1024};
    return state;
}

// Unique, non-guest marker address used as the return address for host->guest
// calls; when the guest's final `ret` pops it into rip, the call has returned.
const u8 g_return_sentinel = 0;

// Bring-up tracing: log the first kTraceBudget blocks of each host->guest call so
// a stop can be localized. "run" prints before executing a block; "done" after.
// If a block crashes inside enter(), you see its "run" with no "done" -> that rip
// is the bad block. If a "done next=X" has no following "run X", compiling X is the
// problem. Set g_jit_trace=false (or lower the budget) once past bring-up.
static bool g_jit_trace = true;
static constexpr u64 kTraceBudget = 256;

// Drive a prepared GuestState until rip reaches stop_rip (a host address outside
// every guest range, so it is never compiled or treated as an HLE target).
void RunUntil(GuestState& gs, VAddr stop_rip, const char* tag) {
    JitState& j = Jit();
    u64 n = 0;
    for (;;) {
        const u64 rip = gs.rip;
        if (rip == stop_rip) {
            if (g_jit_trace)
                LOG_INFO(Core_Linker, "[jit] {} returned after {} block(s)", tag, n);
            return;
        }
        if (!IsGuestPointer(rip)) {
            if (g_jit_trace && n < kTraceBudget)
                LOG_INFO(Core_Linker, "[jit] {} hle  {:#x}", tag, rip);
            j.bridge(&gs);                               // HLE: call host fn, resume at retaddr
            ++n;
            continue;
        }
        if (g_jit_trace && n < kTraceBudget) {
            const u8* b = reinterpret_cast<const u8*>(rip);
            char hex[8 * 3 + 1]; int o = 0;
            for (int i = 0; i < 8; ++i)
                o += std::snprintf(hex + o, sizeof(hex) - o, "%02x ", b[i]);
            LOG_INFO(Core_Linker, "[jit] {} run  {:#x} bytes={}", tag, rip, hex);
        }
        void* host = j.blocks.GetOrCompile(rip, *j.backend);
        if (!host) {
            LOG_CRITICAL(Core_Linker, "[jit] {} COMPILE FAILED @ {:#x}", tag, rip);
            std::abort();
        }
        if (g_jit_trace && n < kTraceBudget)
            LOG_INFO(Core_Linker, "[jit] {} exec {:#x} host={:#x}",
                     tag, rip, reinterpret_cast<u64>(host));
        j.enter(&gs, host);
        if (gs.exit_reason == static_cast<u32>(ExitReason::UnsupportedInstruction)) {
            const u8* ub = reinterpret_cast<const u8*>(gs.rip);
            char uhex[16 * 3 + 1]; int uo = 0;
            for (int i = 0; i < 16; ++i)
                uo += std::snprintf(uhex + uo, sizeof(uhex) - uo, "%02x ", ub[i]);
            LOG_CRITICAL(Core_Linker, "[jit] {} UNSUPPORTED guest instruction @ {:#x} bytes={}",
                         tag, gs.rip, uhex);
            std::abort();
        }
        if (g_jit_trace && n < kTraceBudget)
            LOG_INFO(Core_Linker, "[jit] {} done {:#x} exit={} next={:#x}",
                     tag, rip, gs.exit_reason, gs.rip);
        if (g_jit_trace && n == kTraceBudget)
            LOG_INFO(Core_Linker, "[jit] {} (trace capped at {} blocks)", tag, kTraceBudget);
        ++n;
    }
}

}  // namespace

void SetGuestSegmentBases(VAddr fs_base, VAddr gs_base) noexcept {
    Jit().fs_base = fs_base;
    Jit().gs_base = gs_base;
}

s32 CallGuestEntry(VAddr entry_addr, u64 arg0, const void* arg1, void* arg2) noexcept {
    JitState& j = Jit();   // also registers ranges / builds backend on first use

    // One reusable guest stack for these sequential host->guest calls (module
    // inits run one at a time on the main thread before EnterDispatcher).
    static const VAddr stack_top = AllocGuestStack(8ull * 1024 * 1024);
    const VAddr sentinel = reinterpret_cast<VAddr>(&g_return_sentinel);

    GuestState gs{};
    // fs holds this thread's guest TLS base: fs:[0]=tcb_self, fs:[0x28]=stack
    // canary. The fs/gs mangler reads state->fs_base at runtime, so a fresh
    // per-entry value is correct per thread. SetGuestSegmentBases can override.
    gs.fs_base = j.fs_base ? j.fs_base : reinterpret_cast<u64>(Core::GetTcbBase());
    gs.gs_base = j.gs_base;   // guest (FreeBSD) uses fs for TLS; gs usually unused
    // Emulate a SysV call: 16-aligned rsp, then push the return address so the
    // callee sees rsp%16==8 on entry. The guest's final `ret` lands on sentinel.
    VAddr rsp = stack_top & ~15ull;
    rsp -= 8; *reinterpret_cast<u64*>(rsp) = sentinel;
    gs.rip       = entry_addr;
    gs.gpr[kRsp] = rsp;
    gs.gpr[kRdi] = arg0;                                 // size_t args
    gs.gpr[kRsi] = reinterpret_cast<u64>(arg1);          // const void* argp
    gs.gpr[kRdx] = reinterpret_cast<u64>(arg2);          // void* param
    gs.rflags    = 0x202;

    if (g_jit_trace)
        LOG_INFO(Core_Linker, "[jit] CallGuestEntry entry={:#x} fs_base={:#x} rsp={:#x}",
                 entry_addr, gs.fs_base, gs.gpr[kRsp]);
    if (gs.fs_base == 0)
        LOG_WARNING(Core_Linker, "[jit] fs_base is 0 (no guest TLS on this thread); "
                    "fs:[0]/fs:[0x28] accesses will fault");
    RunUntil(gs, sentinel, "module");
    return static_cast<s32>(gs.gpr[kRax]);
}

[[noreturn]] void EnterDispatcher(VAddr entry_addr, void* params, VAddr exit_func) {
    JitState& j = Jit();

    // Initial GuestState mirrors RunMainEntry's native asm, but into GuestState
    // and onto a dedicated guest stack.
    GuestState gs{};
    // fs holds this thread's guest TLS base: fs:[0]=tcb_self, fs:[0x28]=stack
    // canary. The fs/gs mangler reads state->fs_base at runtime, so a fresh
    // per-entry value is correct per thread. SetGuestSegmentBases can override.
    gs.fs_base = j.fs_base ? j.fs_base : reinterpret_cast<u64>(Core::GetTcbBase());
    gs.gs_base = j.gs_base;   // guest (FreeBSD) uses fs for TLS; gs usually unused
    VAddr rsp = AllocGuestStack(8ull * 1024 * 1024);
    rsp &= ~15ull;                                        // andq $-16,%rsp
    rsp -= 8;                                             // subq $8 (videoout_basic misalign)
    // pushq 8(%1) / pushq 0(%1): copy the first two qwords *of* EntryParams onto
    // the stack (the kernel/OpenOrbis layout), i.e. the values at params+8 then
    // params+0 -- not the params pointer itself.
    auto* p = reinterpret_cast<const u64*>(params);
    rsp -= 8; *reinterpret_cast<u64*>(rsp) = p[1];        // pushq 8(%1)
    rsp -= 8; *reinterpret_cast<u64*>(rsp) = p[0];        // pushq 0(%1)
    gs.rip       = entry_addr;
    gs.gpr[kRsp] = rsp;
    gs.gpr[kRdi] = reinterpret_cast<u64>(params);         // movq %1,%%rdi
    gs.gpr[kRsi] = exit_func;                             // movq %2,%%rsi
    gs.rflags    = 0x202;

    if (g_jit_trace)
        LOG_INFO(Core_Linker, "[jit] EnterDispatcher entry={:#x} fs_base={:#x} rsp={:#x}",
                 entry_addr, gs.fs_base, gs.gpr[kRsp]);
    RunUntil(gs, exit_func, "main");
    std::fprintf(stderr, "[runtime_copy] guest returned to exit function; stopping.\n");
    std::abort();   // INTEGRATION: hook into the emulator's normal shutdown instead.
}

#endif  // SHADPS4_HAVE_JIT

}  // namespace Core::Runtime
