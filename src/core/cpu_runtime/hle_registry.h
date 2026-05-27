// SPDX-FileCopyrightText: Copyright 2024-2026 shadPS4 Emulator Project
// SPDX-License-Identifier: GPL-2.0-or-later

#pragma once

#include <cstddef>
#include <shared_mutex>
#include <string>
#include <string_view>
#include <unordered_map>

#include "common/types.h"

namespace Core::Runtime {

/// Registry of resolved HLE function entries.
///
/// The linker populates this as it resolves guest import stubs to
/// their host-side HLE implementations. The CPU runtime's HLE
/// bridge queries it to produce diagnostic output that *names* the
/// host function being called (e.g., "sceKernelAllocateMainDirectMemory"
/// instead of just "0x700000a44249").
///
/// Two clear uses:
///
///  1. **Diagnostic naming.** Every bridge invocation logs the
///     resolved function name. Crash logs become directly
///     actionable.
///
///  2. **Validation.** If the bridge is invoked with a host address
///     that ISN'T in the registry, that's a strong signal something
///     is wrong: either the JIT computed a bogus target (a JIT bug)
///     or guest code is using a host pointer it shouldn't have
///     (also a bug). The bridge can log a loud warning in that
///     case to make the failure mode visible.
///
/// Thread safety: a shared_mutex protects the map. Registrations
/// happen mostly during module loading (rare); lookups happen on
/// every HLE call (frequent). Shared-read / exclusive-write fits
/// this pattern naturally.
///
/// Lifecycle: the registry is a Meyers singleton with a lifetime
/// that exceeds the Runtime's. This is intentional — shadPS4 may
/// load modules and resolve stubs BEFORE the CPU runtime is
/// constructed (e.g. libSceLibcInternal.sprx is loaded first;
/// the runtime is initialized later when the first guest thread
/// starts). The registry needs to be writable from the linker
/// throughout that early phase.
class HleRegistry {
public:
    /// Get the singleton instance.
    static HleRegistry& Instance();

    /// Register a (host address -> name) mapping. Idempotent: a
    /// re-registration for the same address keeps the first name.
    /// Safe to call concurrently with `Lookup`.
    void Register(VAddr host_addr, std::string name);

    /// Look up a host function name by address. Returns an empty
    /// string view if the address is not registered. The returned
    /// view aliases storage inside the registry — it remains valid
    /// for the lifetime of the registry (i.e., the entire program).
    /// Safe to call concurrently with `Register`.
    [[nodiscard]] std::string_view Lookup(VAddr host_addr) const noexcept;

    /// True iff `host_addr` is registered. Convenience wrapper.
    [[nodiscard]] bool Contains(VAddr host_addr) const noexcept;

    /// Total number of registered entries. Mostly for diagnostics
    /// and tests.
    [[nodiscard]] std::size_t Size() const noexcept;

    /// Remove all entries. Test-only; do not call from production
    /// code paths because the linker may have already published
    /// addresses that guest code is about to execute.
    void ClearForTesting();

private:
    HleRegistry() = default;
    HleRegistry(const HleRegistry&) = delete;
    HleRegistry& operator=(const HleRegistry&) = delete;

    mutable std::shared_mutex mutex_;
    std::unordered_map<VAddr, std::string> map_;
};

} // namespace Core::Runtime
