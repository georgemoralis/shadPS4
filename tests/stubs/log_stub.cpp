// SPDX-FileCopyrightText: Copyright 2025-2026 shadPS4 Emulator Project
// SPDX-License-Identifier: GPL-2.0-or-later

// Minimal stub for Common::Log::FmtLogMessageImpl.
//
// The real implementation lives in src/common/logging/backend.cpp, which
// pulls in a large thread-safe log queue, io_file, platform thread APIs, and
// emulator_settings itself — creating a circular dependency that makes the
// real backend unsuitable for the settings unit-test target.
//
// This stub satisfies the linker by providing the symbol while forwarding
// every message to stderr so test output is still readable when a failure
// occurs inside the settings code.

#include <cstdio>
#include <string_view>
#include <fmt/format.h>
#include "common/logging/backend.h"
#include "common/logging/filter.h"
#include "common/logging/log.h"

namespace Common::Log {

void FmtLogMessageImpl(Class log_class, Level log_level, const char* filename,
                       unsigned int line_num, const char* function, const char* format,
                       const fmt::format_args& args) {
    // Map level to a short prefix so test stderr is scannable.
    const char* prefix = "[ ]";
    switch (log_level) {
    case Level::Trace:    prefix = "[T]"; break;
    case Level::Debug:    prefix = "[D]"; break;
    case Level::Info:     prefix = "[I]"; break;
    case Level::Warning:  prefix = "[W]"; break;
    case Level::Error:    prefix = "[E]"; break;
    case Level::Critical: prefix = "[!]"; break;
    default: break;
    }
    std::string msg = fmt::vformat(format, args);
    std::fprintf(stderr, "%s %s:%u %s\n", prefix, filename, line_num, msg.c_str());
}

// Stubs for Log::Initialize / Start / Stop / Denitializer referenced by
// emulator_settings.cpp indirectly through the migration path.
void Initialize(std::string_view)   {}
bool IsActive()                     { return false; }
void SetGlobalFilter(const Filter&) {}
void SetColorConsoleBackendEnabled(bool) {}
void Start()                        {}
void Stop()                         {}
void Denitializer()                 {}
void SetAppend()                    {}

} // namespace Common::Log
