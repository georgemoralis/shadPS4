// SPDX-FileCopyrightText: Copyright 2024-2026 shadPS4 Emulator Project
// SPDX-License-Identifier: GPL-2.0-or-later

#include "core/cpu_runtime/hle_registry.h"

#include <mutex>
#include <utility>

namespace Core::Runtime {

HleRegistry& HleRegistry::Instance() {
    // Meyers singleton — first-use initialization, thread-safe under
    // C++11+. Lifetime ends at program exit AFTER all static
    // destructors that don't depend on us; the registry's data is
    // available throughout normal runtime.
    static HleRegistry registry;
    return registry;
}

void HleRegistry::Register(VAddr host_addr, std::string name) {
    std::unique_lock lock(mutex_);
    map_.try_emplace(host_addr, std::move(name));
}

std::string_view HleRegistry::Lookup(VAddr host_addr) const noexcept {
    std::shared_lock lock(mutex_);
    const auto it = map_.find(host_addr);
    if (it == map_.end()) {
        return {};
    }
    return it->second;
}

bool HleRegistry::Contains(VAddr host_addr) const noexcept {
    std::shared_lock lock(mutex_);
    return map_.find(host_addr) != map_.end();
}

std::size_t HleRegistry::Size() const noexcept {
    std::shared_lock lock(mutex_);
    return map_.size();
}

void HleRegistry::ClearForTesting() {
    std::unique_lock lock(mutex_);
    map_.clear();
}

} // namespace Core::Runtime
