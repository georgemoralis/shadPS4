// SPDX-FileCopyrightText: Copyright 2025-2026 shadPS4 Emulator Project
// SPDX-License-Identifier: GPL-2.0-or-later

// SDL message-box stub for the settings test target.
//
// EmulatorSettingsImpl::Load() calls SDL_ShowMessageBox() when it finds a
// config.toml but no config.json (the one-time migration dialog).  In the
// test environment config.toml never exists in the temp dir, so the real SDL
// call is never reached at runtime.  However the linker still requires the
// symbol.  This stub satisfies the linker without requiring an SDL window or
// display connection.
//
// If a test ever does end up triggering the migration path by accident, the
// stub returns result=0 ("No" button), which is the safe default that skips
// migration and falls through to SetDefaultValues().

#include <SDL3/SDL_messagebox.h>

extern "C" {

bool SDL_ShowMessageBox(const SDL_MessageBoxData* /* messageboxdata */, int* buttonid) {
    if (buttonid) *buttonid = 0; // "No" — skip migration
    return true;
}

bool SDL_ShowSimpleMessageBox(SDL_MessageBoxFlags /* flags */, const char* /* title */,
                              const char* /* message */, SDL_Window* /* window */) {
    return true;
}

} // extern "C"
