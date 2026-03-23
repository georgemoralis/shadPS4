# SPDX-FileCopyrightText: Copyright 2025-2026 shadPS4 Emulator Project
# SPDX-License-Identifier: GPL-2.0-or-later

# ---------------------------------------------------------------------------
# Unit-test target: shadps4_settings_test
#
# Minimal source set containing only the files required by EmulatorSettingsImpl:
#   - core/emulator_settings.cpp
#   - core/emulator_state.cpp
#   - common/path_util.cpp
#   - common/assert.cpp
#   - common/error.cpp
#   - common/string_util.cpp
#   - common/logging/filter.cpp
#   - common/logging/text_formatter.cpp
#   - Stubs: log_stub.cpp, scm_rev_stub.cpp, sdl_stub.cpp
#
# Deliberately excluded (pull in SDL, kernel threads, or other heavy deps):
#   - common/config.cpp          (includes input_handler.h → SDL + pad library)
#   - common/logging/backend.cpp (includes thread.cpp → kernel pthreads; also
#                                  circularly includes emulator_settings.h)
#   - common/memory_patcher.cpp  (not needed by settings)
#   - common/thread.cpp          (kernel pthread headers)
#   - common/key_manager.cpp     (not needed by settings)
#   - common/ntapi.cpp           (Windows-only; not needed by settings)
#   - core/user_settings.cpp     (includes user_manager.cpp → pugixml +
#                                  userservice.h)
#   - core/user_manager.cpp      (includes userservice.h — HLE library dep)
# ---------------------------------------------------------------------------

    # Find or download Google Test
    include(FetchContent)
    FetchContent_Declare(
        googletest
        URL https://github.com/google/googletest/archive/refs/tags/v1.14.0.zip
    )
    FetchContent_MakeAvailable(googletest)

# Source files shared with the main executable but safe to compile in isolation
set(SETTINGS_TEST_SOURCES
    # Under test
    ${CMAKE_SOURCE_DIR}/src/core/emulator_settings.cpp
    ${CMAKE_SOURCE_DIR}/src/core/emulator_state.cpp

    # Minimal common support
    ${CMAKE_SOURCE_DIR}/src/common/path_util.cpp
    ${CMAKE_SOURCE_DIR}/src/common/assert.cpp
    ${CMAKE_SOURCE_DIR}/src/common/error.cpp
    ${CMAKE_SOURCE_DIR}/src/common/string_util.cpp
    ${CMAKE_SOURCE_DIR}/src/common/logging/filter.cpp
    ${CMAKE_SOURCE_DIR}/src/common/logging/text_formatter.cpp

    # Stubs that replace heavy / circular dependencies
    stubs/log_stub.cpp      # replaces common/logging/backend.cpp
    stubs/scm_rev_stub.cpp  # replaces generated scm_rev.cpp
    stubs/sdl_stub.cpp      # no-op SDL_ShowMessageBox / SDL_ShowSimpleMessageBox

    # The test suite itself
    test_emulator_settings.cpp
)

add_executable(shadps4_settings_test ${SETTINGS_TEST_SOURCES})

# Mirror the include path used by the main target so all relative includes resolve
target_include_directories(shadps4_settings_test PRIVATE
    ${CMAKE_SOURCE_DIR}/src
    ${CMAKE_SOURCE_DIR}
)

# Only the libraries actually needed at link time for these sources
target_link_libraries(shadps4_settings_test PRIVATE
    GTest::gtest_main
    fmt::fmt
    nlohmann_json::nlohmann_json
    toml11::toml11
    SDL3::SDL3      # SDL_messagebox.h is included by emulator_settings.cpp;
                    # our sdl_stub.cpp satisfies the symbol at link time, but
                    # we still need the header available during compilation.
)

# Inherit the C++23 standard required by the rest of the project
target_compile_features(shadps4_settings_test PRIVATE cxx_std_23)

# Mirror the BOOST_ASIO_STANDALONE definition used project-wide
target_compile_definitions(shadps4_settings_test PRIVATE BOOST_ASIO_STANDALONE)

# Clang / AppleClang: enable std::jthread / std::stop_token on libc++
if (CMAKE_CXX_COMPILER_ID STREQUAL "Clang" OR
    CMAKE_CXX_COMPILER_ID STREQUAL "AppleClang")
    include(CheckCXXSymbolExists)
    check_cxx_symbol_exists(_LIBCPP_VERSION version LIBCPP)
    if (LIBCPP)
        target_compile_options(shadps4_settings_test PRIVATE -fexperimental-library)
    endif()
endif()

# Windows-specific: unicode console + suppress MSVC warnings already suppressed
# for the main target
if (WIN32)
    target_compile_definitions(shadps4_settings_test PRIVATE
        NOMINMAX
        WIN32_LEAN_AND_MEAN
        NTDDI_VERSION=0x0A000006
        _WIN32_WINNT=0x0A00
        WINVER=0x0A00
    )
    if (MSVC)
        target_compile_definitions(shadps4_settings_test PRIVATE
            _CRT_SECURE_NO_WARNINGS
            _CRT_NONSTDC_NO_DEPRECATE
            _SCL_SECURE_NO_WARNINGS
            _TIMESPEC_DEFINED
        )
    endif()
endif()

# Register with CTest so `ctest` and `cmake --build . --target test` work
include(GoogleTest)
gtest_discover_tests(shadps4_settings_test
    WORKING_DIRECTORY ${CMAKE_BINARY_DIR}
    PROPERTIES TIMEOUT 60
)
