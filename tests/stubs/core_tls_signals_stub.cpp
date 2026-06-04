// SPDX-FileCopyrightText: Copyright 2026 shadPS4 Emulator Project
// SPDX-License-Identifier: GPL-2.0-or-later

// Test-support stubs for the core thread-context dependencies that
// Runtime::Run() references but the cpu_runtime test-support library does not
// link. The real definitions live in core/tls.cpp and core/signals.cpp, which
// belong to the full core library; the test-support lib compiles only a minimal
// subset of core, so it provides these instead.
//
// Under the unit tests there is no real guest TCB and no JIT fault path, so the
// behavior here matches the pre-TLS test behavior exactly:
//   - GetTcbBase() returns null  -> Run() leaves the guest fs_base as-is (0).
//     The test programs never touch fs:/gs:, so this is inert.
//   - EnsureVehStack() is a no-op. It is only a real (out-of-line) symbol on
//     Windows; off-Windows signals.h already defines it inline, so we must not
//     redefine it there.

#include "core/signals.h"
#include "core/tls.h"

namespace Core {

Tcb* GetTcbBase() {
    return nullptr;
}

#ifdef _WIN32
void EnsureVehStack() {}
#endif

} // namespace Core
