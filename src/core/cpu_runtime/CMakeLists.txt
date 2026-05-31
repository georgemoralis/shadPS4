# SPDX-FileCopyrightText: Copyright 2026 shadPS4 Emulator Project
# SPDX-License-Identifier: GPL-2.0-or-later

# ============================================================================
# CPU runtime (experimental native-execution JIT)
# ----------------------------------------------------------------------------
# The runtime has an architecture-independent core (dispatcher, code/block
# caches, HLE registry, guest-state model) plus exactly ONE host code-
# generation backend, selected by ARCHITECTURE (set in the top-level
# CMakeLists.txt to "x86_64" or "arm64").
#
# This backend selection MUST stay in lock-step with externals/CMakeLists.txt,
# which pulls in only the matching assembler: xbyak for x86_64,
# xbyak_aarch64 for arm64. Compiling exactly one backend per configuration is
# what lets the x86 build avoid the AArch64 assembler header (and vice versa) —
# the arm64 lifter/gateway #include <xbyak_aarch64/...> unconditionally, so
# they must simply not be handed to the compiler on an x86 build.
# ============================================================================

# Opt-in JIT corruption-investigation instrumentation. OFF by default: when
# off, the diagnostics subsystem (runtime_diagnostics.cpp) compiles to an empty
# translation unit and the dispatcher's diagnostics calls become inline no-ops
# (see runtime_diagnostics.h), so there is zero hot-path cost in normal builds.
# Turn on only for targeted debugging:  -DSHADPS4_RUNTIME_DIAGNOSTICS=ON
option(SHADPS4_RUNTIME_DIAGNOSTICS
    "Build the JIT runtime corruption-investigation diagnostics (RAXTRACE / \
GPRSNAP / block dumps). Off by default; adds hot-path tracing when on." OFF)

# Architecture-independent runtime sources — always compiled.
set(CPU_RUNTIME_COMMON_SRCS
    runtime.cpp
    runtime.h
    runtime_diagnostics.cpp
    runtime_diagnostics.h
    code_cache.cpp
    code_cache.h
    block_cache.cpp
    block_cache.h
    hle_registry.cpp
    hle_registry.h
    guest_state.h
    gateway/gateway.h
    lifter/lifter.h
)

# Host-specific code-generation backend: lifter + gateway. Exactly one set is
# compiled, matching the assembler the externals guard added.
if (ARCHITECTURE STREQUAL "x86_64")
    set(CPU_RUNTIME_BACKEND_SRCS
        lifter/lifter_x86_host.cpp
        gateway/gateway_x86.cpp
    )
    set(CPU_RUNTIME_BACKEND_LIB xbyak::xbyak)
elseif (ARCHITECTURE STREQUAL "arm64")
    # NOTE: the arm64 backend is still a draft and is not built by mainline.
    # These files are the promoted (non-"_draft") names the tree's headers
    # already reference (see gateway.h / guest_state.h); move the drafts into
    # place before enabling an arm64 configure.
    set(CPU_RUNTIME_BACKEND_SRCS
        lifter/lifter_arm64_host.cpp
        gateway/gateway_arm64.cpp
    )
    set(CPU_RUNTIME_BACKEND_LIB xbyak_aarch64)
else()
    message(FATAL_ERROR
        "cpu_runtime: unsupported ARCHITECTURE '${ARCHITECTURE}'")
endif()

add_library(cpu_runtime STATIC
    ${CPU_RUNTIME_COMMON_SRCS}
    ${CPU_RUNTIME_BACKEND_SRCS}
)

# Propagate the diagnostics toggle as a compile definition. PUBLIC so that any
# consumer including runtime_diagnostics.h sees the same active/no-op decision
# the library was built with — otherwise the inline stubs in the header (chosen
# by this macro) could disagree with the out-of-line definitions in the
# library, an ODR hazard.
if (SHADPS4_RUNTIME_DIAGNOSTICS)
    target_compile_definitions(cpu_runtime PUBLIC SHADPS4_RUNTIME_DIAGNOSTICS)
endif()

# Zydis (guest x86 decoder) is needed by every configuration — the lifter
# decodes guest x86 regardless of host arch, and the runtime's block
# disassembler and signal-fault decoder use it too. The code-generation
# assembler is the only arch-dependent dependency.
target_link_libraries(cpu_runtime PRIVATE
    Zydis::Zydis
    ${CPU_RUNTIME_BACKEND_LIB}
)

# Headers are resolved via the project-wide `include_directories(src)` set in
# the top-level CMakeLists.txt (includes use "core/cpu_runtime/..." and
# "common/..." paths), so no per-target include directories are needed here.
create_target_directory_groups(cpu_runtime)
