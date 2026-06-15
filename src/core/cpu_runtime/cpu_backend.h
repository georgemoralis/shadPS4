// SPDX-FileCopyrightText: Copyright 2026 shadPS4 Emulator Project
// SPDX-License-Identifier: GPL-2.0-or-later
#pragma once
#include <memory>
#include "common/types.h"

#if defined(SHADPS4_CPU_BACKEND_RUNTIME_COPY) || defined(SHADPS4_CPU_BACKEND_RUNTIME_LIFTER)
#define SHADPS4_HAVE_JIT 1
#elif defined(SHADPS4_CPU_BACKEND_NATIVE)
#define SHADPS4_HAVE_JIT 0
#else
#error "Set the SHADPS4_CPU_BACKEND CMake option (native|runtime_copy|runtime_lifter)."
#endif

namespace Core::Runtime {

class CodeCache;
struct GuestState;

enum class CpuBackend : u32 { Native=0, RuntimeCopy=1, RuntimeLifter=2 };

inline constexpr CpuBackend kActiveBackend =
#if defined(SHADPS4_CPU_BACKEND_RUNTIME_COPY)
    CpuBackend::RuntimeCopy;
#elif defined(SHADPS4_CPU_BACKEND_RUNTIME_LIFTER)
    CpuBackend::RuntimeLifter;
#else
    CpuBackend::Native;
#endif

#if (defined(__APPLE__) && defined(__aarch64__))
static_assert(kActiveBackend == CpuBackend::Native,
              "macOS arm64 builds must use SHADPS4_CPU_BACKEND=native");
#endif

#if SHADPS4_HAVE_JIT

// gateway: run block_entry until its terminator returns, leaving next rip +
// exit_reason in state. hle bridge: invoked when rip is a host function.
using EnterBlockFn = void (*)(GuestState* state, void* block_entry);
using HleBridgeFn  = void (*)(GuestState* state);

class Backend {
public:
    virtual ~Backend() = default;
    Backend(const Backend&) = delete;
    Backend& operator=(const Backend&) = delete;
    [[nodiscard]] virtual void* CompileBlock(u64 guest_rip) = 0;
    [[nodiscard]] virtual u64 BlockReservationSize() const noexcept = 0;
    [[nodiscard]] virtual EnterBlockFn GetEnterBlock() const noexcept = 0;
    [[nodiscard]] virtual HleBridgeFn  GetHleBridge() const noexcept = 0;
    [[nodiscard]] virtual CpuBackend Kind() const noexcept = 0;
protected:
    Backend() = default;
};

[[nodiscard]] std::unique_ptr<Backend> CreateActiveJitBackend(CodeCache& code_cache);

#endif  // SHADPS4_HAVE_JIT

}  // namespace Core::Runtime
