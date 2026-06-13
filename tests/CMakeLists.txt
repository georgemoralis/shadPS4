# SPDX-FileCopyrightText: Copyright 2026 shadPS4 Emulator Project
# SPDX-License-Identifier: GPL-2.0-or-later

# Find or download Google Test
include(FetchContent)
    FetchContent_Declare(
    googletest
    URL https://github.com/google/googletest/archive/refs/tags/v1.17.0.zip
)
FetchContent_MakeAvailable(googletest)

set(TEST_TARGETS shadps4_settings_test shadps4_gcn_test)

set(SETTINGS_TEST_SOURCES
    # Under test
    ${CMAKE_SOURCE_DIR}/src/core/emulator_settings.cpp
    ${CMAKE_SOURCE_DIR}/src/core/emulator_state.cpp

    # Minimal common support
    ${CMAKE_SOURCE_DIR}/src/common/path_util.cpp
    ${CMAKE_SOURCE_DIR}/src/common/assert.cpp
    ${CMAKE_SOURCE_DIR}/src/common/error.cpp
    ${CMAKE_SOURCE_DIR}/src/common/string_util.cpp
    ${CMAKE_SOURCE_DIR}/src/common/logging/log.cpp

    # Stubs that replace dependencies
    stubs/common_stub.cpp
    stubs/scm_rev_stub.cpp
    stubs/sdl_stub.cpp

    # Tests
    test_emulator_settings.cpp
)

set(GCN_TEST_SOURCES
    # Under test
    ${CMAKE_SOURCE_DIR}/src/core/emulator_settings.cpp
    ${CMAKE_SOURCE_DIR}/src/core/emulator_state.cpp
    ${CMAKE_SOURCE_DIR}/src/shader_recompiler/backend/spirv/emit_spirv.cpp
    ${CMAKE_SOURCE_DIR}/src/shader_recompiler/backend/spirv/emit_spirv.h
    ${CMAKE_SOURCE_DIR}/src/shader_recompiler/backend/spirv/emit_spirv_atomic.cpp
    ${CMAKE_SOURCE_DIR}/src/shader_recompiler/backend/spirv/emit_spirv_barriers.cpp
    ${CMAKE_SOURCE_DIR}/src/shader_recompiler/backend/spirv/emit_spirv_bitwise_conversion.cpp
    ${CMAKE_SOURCE_DIR}/src/shader_recompiler/backend/spirv/emit_spirv_composite.cpp
    ${CMAKE_SOURCE_DIR}/src/shader_recompiler/backend/spirv/emit_spirv_context_get_set.cpp
    ${CMAKE_SOURCE_DIR}/src/shader_recompiler/backend/spirv/emit_spirv_convert.cpp
    ${CMAKE_SOURCE_DIR}/src/shader_recompiler/backend/spirv/emit_spirv_floating_point.cpp
    ${CMAKE_SOURCE_DIR}/src/shader_recompiler/backend/spirv/emit_spirv_image.cpp
    ${CMAKE_SOURCE_DIR}/src/shader_recompiler/backend/spirv/emit_spirv_integer.cpp
    ${CMAKE_SOURCE_DIR}/src/shader_recompiler/backend/spirv/emit_spirv_logical.cpp
    ${CMAKE_SOURCE_DIR}/src/shader_recompiler/backend/spirv/emit_spirv_select.cpp
    ${CMAKE_SOURCE_DIR}/src/shader_recompiler/backend/spirv/emit_spirv_shared_memory.cpp
    ${CMAKE_SOURCE_DIR}/src/shader_recompiler/backend/spirv/emit_spirv_special.cpp
    ${CMAKE_SOURCE_DIR}/src/shader_recompiler/backend/spirv/emit_spirv_undefined.cpp
    ${CMAKE_SOURCE_DIR}/src/shader_recompiler/backend/spirv/emit_spirv_warp.cpp
    ${CMAKE_SOURCE_DIR}/src/shader_recompiler/backend/spirv/emit_spirv_instructions.h
    ${CMAKE_SOURCE_DIR}/src/shader_recompiler/backend/spirv/spirv_emit_context.cpp
    ${CMAKE_SOURCE_DIR}/src/shader_recompiler/backend/spirv/spirv_emit_context.h
    ${CMAKE_SOURCE_DIR}/src/shader_recompiler/frontend/format.cpp
    ${CMAKE_SOURCE_DIR}/src/shader_recompiler/frontend/decode.cpp
    ${CMAKE_SOURCE_DIR}/src/shader_recompiler/frontend/decode.h
    ${CMAKE_SOURCE_DIR}/src/shader_recompiler/frontend/fetch_shader.cpp
    ${CMAKE_SOURCE_DIR}/src/shader_recompiler/frontend/fetch_shader.h
    ${CMAKE_SOURCE_DIR}/src/shader_recompiler/frontend/translate/data_share.cpp
    ${CMAKE_SOURCE_DIR}/src/shader_recompiler/frontend/translate/export.cpp
    ${CMAKE_SOURCE_DIR}/src/shader_recompiler/frontend/translate/translate.cpp
    ${CMAKE_SOURCE_DIR}/src/shader_recompiler/frontend/translate/translate.h
    ${CMAKE_SOURCE_DIR}/src/shader_recompiler/frontend/translate/scalar_alu.cpp
    ${CMAKE_SOURCE_DIR}/src/shader_recompiler/frontend/translate/scalar_flow.cpp
    ${CMAKE_SOURCE_DIR}/src/shader_recompiler/frontend/translate/scalar_memory.cpp
    ${CMAKE_SOURCE_DIR}/src/shader_recompiler/frontend/translate/vector_alu.cpp
    ${CMAKE_SOURCE_DIR}/src/shader_recompiler/frontend/translate/vector_interpolation.cpp
    ${CMAKE_SOURCE_DIR}/src/shader_recompiler/frontend/translate/vector_memory.cpp
    ${CMAKE_SOURCE_DIR}/src/shader_recompiler/ir/abstract_syntax_list.cpp
    ${CMAKE_SOURCE_DIR}/src/shader_recompiler/ir/abstract_syntax_list.h
    ${CMAKE_SOURCE_DIR}/src/shader_recompiler/ir/basic_block.cpp
    ${CMAKE_SOURCE_DIR}/src/shader_recompiler/ir/basic_block.h
    ${CMAKE_SOURCE_DIR}/src/shader_recompiler/ir/microinstruction.cpp
    ${CMAKE_SOURCE_DIR}/src/shader_recompiler/ir/attribute.cpp
    ${CMAKE_SOURCE_DIR}/src/shader_recompiler/ir/attribute.h
    ${CMAKE_SOURCE_DIR}/src/shader_recompiler/ir/ir_emitter.cpp
    ${CMAKE_SOURCE_DIR}/src/shader_recompiler/ir/ir_emitter.h
    ${CMAKE_SOURCE_DIR}/src/shader_recompiler/ir/opcodes.cpp
    ${CMAKE_SOURCE_DIR}/src/shader_recompiler/ir/opcodes.h
    ${CMAKE_SOURCE_DIR}/src/shader_recompiler/ir/patch.cpp
    ${CMAKE_SOURCE_DIR}/src/shader_recompiler/ir/patch.h
    ${CMAKE_SOURCE_DIR}/src/shader_recompiler/ir/program.cpp
    ${CMAKE_SOURCE_DIR}/src/shader_recompiler/ir/program.h
    ${CMAKE_SOURCE_DIR}/src/shader_recompiler/ir/passes/ir_passes.h
    ${CMAKE_SOURCE_DIR}/src/shader_recompiler/ir/passes/constant_propagation_pass.cpp
    ${CMAKE_SOURCE_DIR}/src/shader_recompiler/ir/passes/dead_code_elimination_pass.cpp
    ${CMAKE_SOURCE_DIR}/src/shader_recompiler/ir/passes/identity_removal_pass.cpp
    ${CMAKE_SOURCE_DIR}/src/shader_recompiler/ir/passes/shader_info_collection_pass.cpp
    ${CMAKE_SOURCE_DIR}/src/shader_recompiler/ir/passes/ssa_rewrite_pass.cpp
    ${CMAKE_SOURCE_DIR}/src/shader_recompiler/ir/post_order.cpp
    ${CMAKE_SOURCE_DIR}/src/shader_recompiler/ir/post_order.h
    ${CMAKE_SOURCE_DIR}/src/shader_recompiler/ir/type.cpp
    ${CMAKE_SOURCE_DIR}/src/shader_recompiler/ir/type.h
    ${CMAKE_SOURCE_DIR}/src/shader_recompiler/ir/value.cpp
    ${CMAKE_SOURCE_DIR}/src/shader_recompiler/ir/value.h
    ${CMAKE_SOURCE_DIR}/src/video_core/amdgpu/pixel_format.cpp
    ${CMAKE_SOURCE_DIR}/src/video_core/amdgpu/pixel_format.h
    ${CMAKE_SOURCE_DIR}/src/video_core/amdgpu/resource.h

    # Minimal common support
    ${CMAKE_SOURCE_DIR}/src/common/path_util.cpp
    ${CMAKE_SOURCE_DIR}/src/common/assert.cpp
    ${CMAKE_SOURCE_DIR}/src/common/error.cpp
    ${CMAKE_SOURCE_DIR}/src/common/io_file.cpp
    ${CMAKE_SOURCE_DIR}/src/common/string_util.cpp
    ${CMAKE_SOURCE_DIR}/src/common/logging/log.cpp
    ${CMAKE_SOURCE_DIR}/src/common/ntapi.cpp

    # Stubs that replace dependencies
    stubs/common_stub.cpp
    stubs/resource_tracking_pass_stub.cpp
    stubs/scm_rev_stub.cpp

    gcn/gcn_test_runner.hpp
    gcn/gcn_test_runner.cpp
    gcn/translator.hpp
    gcn/translator.cpp

    # Tests
    gcn/test_gcn_instructions.cpp
)

add_executable(shadps4_settings_test ${SETTINGS_TEST_SOURCES})
add_executable(shadps4_gcn_test ${GCN_TEST_SOURCES})

foreach(t ${TEST_TARGETS})
    target_include_directories(${t} PRIVATE
        ${CMAKE_SOURCE_DIR}/src
        ${CMAKE_SOURCE_DIR}
    )
    target_compile_features(${t} PRIVATE cxx_std_23)
    target_compile_definitions(${t} PRIVATE BOOST_ASIO_STANDALONE)
endforeach()

target_link_libraries(shadps4_settings_test PRIVATE
    GTest::gtest_main
    fmt::fmt
    nlohmann_json::nlohmann_json
    toml11::toml11
    SDL3::SDL3
    spdlog::spdlog
)
target_link_libraries(shadps4_gcn_test PRIVATE
    GTest::gtest_main
    fmt::fmt
    nlohmann_json::nlohmann_json
    magic_enum::magic_enum
    toml11::toml11
    Boost::headers
    Vulkan::Headers
    sirit
    SDL3::SDL3
    spdlog::spdlog
    half::half
)

if (CMAKE_CXX_COMPILER_ID STREQUAL "Clang" OR
    CMAKE_CXX_COMPILER_ID STREQUAL "AppleClang")
    include(CheckCXXSymbolExists)
    check_cxx_symbol_exists(_LIBCPP_VERSION version LIBCPP)
    if (LIBCPP)
        foreach(t ${TEST_TARGETS})
            target_compile_options(${t} PRIVATE -fexperimental-library)
        endforeach()
    endif()
endif()

foreach(t ${TEST_TARGETS})
    if (WIN32)
        target_link_libraries(${t} PRIVATE onecore)
        target_compile_definitions(${t} PRIVATE
            NOMINMAX
            WIN32_LEAN_AND_MEAN
            NTDDI_VERSION=0x0A000006
            _WIN32_WINNT=0x0A00
            WINVER=0x0A00
        )
        if (MSVC)
            target_compile_definitions(${t} PRIVATE
                _CRT_SECURE_NO_WARNINGS
                _CRT_NONSTDC_NO_DEPRECATE
                _SCL_SECURE_NO_WARNINGS
                _TIMESPEC_DEFINED
            )
        endif()
    endif()
endforeach()

include(GoogleTest)

foreach(t ${TEST_TARGETS})
    gtest_discover_tests(${t}
        WORKING_DIRECTORY ${CMAKE_BINARY_DIR}
        PROPERTIES TIMEOUT 60
    )
endforeach()

# ===========================================================================
# HTTP tests (libSceHttp - URI helpers and status-line parsing)
# ===========================================================================
# Self-contained block - mirrors the structure of the targets above so the
# new test does not depend on any earlier foreach iterations.

set(HTTP_TEST_SOURCES
    # Under test
    ${CMAKE_SOURCE_DIR}/src/core/libraries/network/http.cpp

    # Required to link RegisterLib's LIB_FUNCTION calls and the logger's
    # access to EmulatorSettings.
    ${CMAKE_SOURCE_DIR}/src/core/emulator_settings.cpp
    ${CMAKE_SOURCE_DIR}/src/core/emulator_state.cpp

    # Minimal common support
    ${CMAKE_SOURCE_DIR}/src/common/path_util.cpp
    ${CMAKE_SOURCE_DIR}/src/common/assert.cpp
    ${CMAKE_SOURCE_DIR}/src/common/error.cpp
    ${CMAKE_SOURCE_DIR}/src/common/string_util.cpp
    ${CMAKE_SOURCE_DIR}/src/common/logging/log.cpp

    # Stubs that replace dependencies
    stubs/common_stub.cpp
    stubs/scm_rev_stub.cpp
    stubs/sdl_stub.cpp
    stubs/loader_stub.cpp

    # Tests
    network/test_http_uri.cpp
    network/test_http_status_line.cpp
    network/test_http_parse_response_header.cpp
)

add_executable(shadps4_http_test ${HTTP_TEST_SOURCES})

list(APPEND TEST_TARGETS shadps4_http_test)

target_include_directories(shadps4_http_test PRIVATE
    ${CMAKE_SOURCE_DIR}/src
    ${CMAKE_SOURCE_DIR}
)
target_compile_features(shadps4_http_test PRIVATE cxx_std_23)
target_compile_definitions(shadps4_http_test PRIVATE BOOST_ASIO_STANDALONE)

target_link_libraries(shadps4_http_test PRIVATE
    GTest::gtest_main
    fmt::fmt
    magic_enum::magic_enum
    nlohmann_json::nlohmann_json
    toml11::toml11
    SDL3::SDL3
    spdlog::spdlog
)

if (CMAKE_CXX_COMPILER_ID STREQUAL "Clang" OR
    CMAKE_CXX_COMPILER_ID STREQUAL "AppleClang")
    include(CheckCXXSymbolExists)
    check_cxx_symbol_exists(_LIBCPP_VERSION version LIBCPP)
    if (LIBCPP)
        target_compile_options(shadps4_http_test PRIVATE -fexperimental-library)
    endif()
endif()

if (WIN32)
    target_link_libraries(shadps4_http_test PRIVATE onecore)
    target_compile_definitions(shadps4_http_test PRIVATE
        NOMINMAX
        WIN32_LEAN_AND_MEAN
        NTDDI_VERSION=0x0A000006
        _WIN32_WINNT=0x0A00
        WINVER=0x0A00
    )
    if (MSVC)
        target_compile_definitions(shadps4_http_test PRIVATE
            _CRT_SECURE_NO_WARNINGS
            _CRT_NONSTDC_NO_DEPRECATE
            _SCL_SECURE_NO_WARNINGS
            _TIMESPEC_DEFINED
        )
    endif()
endif()

gtest_discover_tests(shadps4_http_test
    WORKING_DIRECTORY ${CMAKE_BINARY_DIR}
    PROPERTIES TIMEOUT 60
)

# ===========================================================================
# CPU runtime tests (src/core/cpu_runtime/)
# ===========================================================================
# Built only when the runtime backend is compiled. Mirrors the structure
# of the HTTP test block above.
if (SHADPS4_CPU_BACKEND_RESOLVED STREQUAL "runtime")
    # Shared support: the runtime/backend sources and minimal common deps that
    # both test executables need. Built once into a static lib so the heavy
    # lifter/runtime objects are not recompiled per test binary.
    #
    # NOTE: the test executables are deliberately SEPARATE (not one binary with
    # both test files). Each test translation unit carries its own copy of the
    # test harness (GuestMemory, the CpuRuntimeTest fixture, RunProgram, the
    # kReturnSentinel constant, etc.) in an anonymous namespace. Linking both
    # test TUs into a single program is an ODR hazard: GoogleTest registers a
    # TEST_F by its fixture+test name globally, but each TU defines its own
    # anonymous-namespace `CpuRuntimeTest`, so a combined binary binds all
    # CpuRuntimeTest tests to ONE of the two fixtures and the other TU's tests
    # silently run against the wrong harness/helpers. Keeping them in separate
    # executables gives each TU its own process and its own harness, which is
    # also why the split must not be "collapsed" back into one target later.
    set(CPU_RUNTIME_TEST_SUPPORT_SRCS
        # Under test
        ${CMAKE_SOURCE_DIR}/src/core/cpu_runtime/block_cache.cpp
        ${CMAKE_SOURCE_DIR}/src/core/cpu_runtime/code_cache.cpp
        ${CMAKE_SOURCE_DIR}/src/core/cpu_runtime/hle_registry.cpp
        ${CMAKE_SOURCE_DIR}/src/core/cpu_runtime/runtime.cpp
        ${CMAKE_SOURCE_DIR}/src/core/cpu_runtime/runtime_diagnostics.cpp

        # Required by logger
        ${CMAKE_SOURCE_DIR}/src/core/emulator_settings.cpp
        ${CMAKE_SOURCE_DIR}/src/core/emulator_state.cpp

        # Minimal common support
        ${CMAKE_SOURCE_DIR}/src/common/path_util.cpp
        ${CMAKE_SOURCE_DIR}/src/common/assert.cpp
        ${CMAKE_SOURCE_DIR}/src/common/error.cpp
        ${CMAKE_SOURCE_DIR}/src/common/string_util.cpp
        ${CMAKE_SOURCE_DIR}/src/common/logging/log.cpp

        # Stubs that replace dependencies
        stubs/common_stub.cpp
        stubs/scm_rev_stub.cpp
        stubs/sdl_stub.cpp
        stubs/core_tls_signals_stub.cpp
    )

    if (ARCHITECTURE STREQUAL "x86_64")
        list(APPEND CPU_RUNTIME_TEST_SUPPORT_SRCS
            ${CMAKE_SOURCE_DIR}/src/core/cpu_runtime/gateway/gateway_x86.cpp
            ${CMAKE_SOURCE_DIR}/src/core/cpu_runtime/lifter/lifter_x86_host.cpp)
    elseif (ARCHITECTURE STREQUAL "arm64")
        # arm64 host backend. NOTE: the arm64 lifter is currently a skeleton
        # (a few representative emitters), so most lifter tests in
        # test_lifter.cpp — which assert x86-lifter behavior by running emitted
        # code — will FAIL here until the full emitter port lands. The arm64 CI
        # test job is allowed to fail for exactly this reason; building the
        # target gives compile/link coverage of the arm64 backend in the
        # meantime.
        list(APPEND CPU_RUNTIME_TEST_SUPPORT_SRCS
            ${CMAKE_SOURCE_DIR}/src/core/cpu_runtime/gateway/gateway_arm64.cpp
            ${CMAKE_SOURCE_DIR}/src/core/cpu_runtime/lifter/lifter_arm64_host.cpp
            # The HLE-bridge call thunk runtime.cpp's CallHostFromGuest
            # references on arm64. This support lib compiles runtime.cpp
            # directly instead of linking the cpu_runtime library, so it
            # must carry the .S itself (same reason as the diagnostics
            # define above). Top-level project already enables ASM.
            ${CMAKE_SOURCE_DIR}/src/core/cpu_runtime/hle_thunk_arm64.S)
    endif()

    add_library(shadps4_cpu_runtime_test_support STATIC
        ${CPU_RUNTIME_TEST_SUPPORT_SRCS})
    target_include_directories(shadps4_cpu_runtime_test_support PUBLIC
        ${CMAKE_SOURCE_DIR}/src
        ${CMAKE_SOURCE_DIR}
    )
    target_compile_features(shadps4_cpu_runtime_test_support PUBLIC cxx_std_23)
    target_compile_definitions(shadps4_cpu_runtime_test_support PUBLIC
        BOOST_ASIO_STANDALONE
        SHADPS4_USES_RUNTIME=1)

    # Mirror the runtime diagnostics toggle (see runtime/CMakeLists.txt). The
    # support lib compiles runtime.cpp + runtime_diagnostics.cpp directly rather
    # than linking the cpu_runtime library, so it does not inherit that lib's
    # definition; set it here so runtime_diagnostics.{h,cpp} land on the same
    # side of the #ifdef. PUBLIC so the test TUs that include the header agree.
    if (SHADPS4_RUNTIME_DIAGNOSTICS)
        target_compile_definitions(shadps4_cpu_runtime_test_support PUBLIC
            SHADPS4_RUNTIME_DIAGNOSTICS)
    endif()

    target_link_libraries(shadps4_cpu_runtime_test_support PUBLIC
        fmt::fmt
        magic_enum::magic_enum
        nlohmann_json::nlohmann_json
        toml11::toml11
        tsl::robin_map
        Zydis::Zydis
        SDL3::SDL3
        spdlog::spdlog
    )

    # Code generator for the runtime lifter: x86 xbyak on x86_64,
    # xbyak_aarch64 on arm64 — matching the backend sources added above.
    if (ARCHITECTURE STREQUAL "x86_64")
        target_link_libraries(shadps4_cpu_runtime_test_support PUBLIC xbyak::xbyak)
    elseif (ARCHITECTURE STREQUAL "arm64")
        target_link_libraries(shadps4_cpu_runtime_test_support PUBLIC xbyak_aarch64)
    endif()

    if (CMAKE_CXX_COMPILER_ID STREQUAL "Clang" OR
        CMAKE_CXX_COMPILER_ID STREQUAL "AppleClang")
        include(CheckCXXSymbolExists)
        check_cxx_symbol_exists(_LIBCPP_VERSION version LIBCPP)
        if (LIBCPP)
            target_compile_options(shadps4_cpu_runtime_test_support PUBLIC
                -fexperimental-library)
        endif()
    endif()

    if (WIN32)
        target_link_libraries(shadps4_cpu_runtime_test_support PUBLIC onecore)
        target_compile_definitions(shadps4_cpu_runtime_test_support PUBLIC
            NOMINMAX
            WIN32_LEAN_AND_MEAN
            NTDDI_VERSION=0x0A000006
            _WIN32_WINNT=0x0A00
            WINVER=0x0A00
        )
        if (MSVC)
            target_compile_definitions(shadps4_cpu_runtime_test_support PUBLIC
                _CRT_SECURE_NO_WARNINGS
                _CRT_NONSTDC_NO_DEPRECATE
                _SCL_SECURE_NO_WARNINGS
                _TIMESPEC_DEFINED
            )
        endif()
    endif()

    # Two thin test executables, one per area. Each links the shared support
    # lib (which carries all the include dirs, definitions, compile features,
    # and library deps via its PUBLIC usage requirements) plus gtest_main.
    #
    # The concurrency test spawns host threads (std::thread); link an explicit
    # threading library so it builds on toolchains where pthread is not folded
    # into libc (the runtime itself uses no std::thread, so nothing else pulls
    # it into the link). Threads::Threads is a harmless no-op for the other
    # areas, and THREADS_PREFER_PTHREAD_FLAG applies -pthread at compile+link.
    set(THREADS_PREFER_PTHREAD_FLAG ON)
    find_package(Threads REQUIRED)
    foreach(area runtime lifter differential concurrency differential_sonic)
        add_executable(shadps4_cpu_${area}_test test_${area}.cpp)
        target_link_libraries(shadps4_cpu_${area}_test PRIVATE
            shadps4_cpu_runtime_test_support
            GTest::gtest_main
            Threads::Threads)
        list(APPEND TEST_TARGETS shadps4_cpu_${area}_test)
        gtest_discover_tests(shadps4_cpu_${area}_test
            WORKING_DIRECTORY ${CMAKE_BINARY_DIR}
            PROPERTIES TIMEOUT 60
        )
    endforeach()
endif()
