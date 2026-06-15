// SPDX-FileCopyrightText: Copyright 2026 shadPS4 Emulator Project
// SPDX-License-Identifier: GPL-2.0-or-later
#pragma once
#include <cstddef>
#include "common/types.h"
#include "core/cpu_runtime/cpu_backend.h"

namespace Core::Runtime {

// Guest VA ranges registered once at startup; IsGuestPointer is then an exact
// range test (hot path: every dispatcher iteration + fault context).
struct GuestAddressRange { VAddr base; u64 size; };
void RegisterGuestAddressRanges(const GuestAddressRange* ranges, std::size_t count) noexcept;
[[nodiscard]] bool IsGuestPointer(VAddr addr) noexcept;
// Convenience for pointer contexts (e.g. a fault address from mcontext).
[[nodiscard]] inline bool IsGuestPointer(const void* ptr) noexcept {
    return IsGuestPointer(reinterpret_cast<VAddr>(ptr));
}

#if SHADPS4_HAVE_JIT
// Replaces RunMainEntry's direct jmp when a JIT backend is built. Sets up the
// initial GuestState from the loader's entry params, then runs the dispatch
// loop until the guest returns to exit_func. Never returns.
[[noreturn]] void EnterDispatcher(VAddr entry_addr, void* params, VAddr exit_func);

// Run a guest function to completion through the JIT and return its result
// (guest rax). Used wherever the host must call into guest code and get control
// back: module_start/stop, and (later) thread entry points and callbacks. The
// arguments follow the PS4 SysV ABI (rdi, rsi, rdx).
s32 CallGuestEntry(VAddr entry_addr, u64 arg0, const void* arg1, void* arg2) noexcept;

// INTEGRATION: set the guest TLS segment bases the JIT uses to rebase fs/gs
// accesses in copied code. The loader establishes these when it sets up the
// main thread's TLS; until then they are 0 and fs/gs-relative guest accesses
// will compute wrong addresses.
void SetGuestSegmentBases(VAddr fs_base, VAddr gs_base) noexcept;
#endif

}  // namespace Core::Runtime
