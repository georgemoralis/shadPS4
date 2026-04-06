// SPDX-FileCopyrightText: Copyright 2025 shadPS4 Emulator Project
// SPDX-License-Identifier: GPL-2.0-or-later

#pragma once

#include "common/types.h"

namespace Libraries::Np::NpSignaling::Helpers {

using SignalingIpmiHandler = void (*)(int*);

// Bridge hooks used while the initialize rewrite is being staged. These let the
// helper layer drive the current HLE runtime pieces without keeping their
// orchestration logic in sceNpSignalingInitialize itself.
struct SignalingRuntimeHooks {
    void (*start_dispatch)();
    void (*start_receive)();
    void (*start_ping)();
    void (*stop_ping)();
    void (*stop_receive)();
    void (*stop_dispatch)();
};

// Initialize / terminate helper surface for the staged signaling rewrite.
void SetRuntimeHooks(const SignalingRuntimeHooks& hooks);
s32 CheckInitializeAppType(u32* is_app_type_4);
s32 InitSignalingHeap(s64 pool_size);
void ShutdownSignalingHeap();
s32 RegisterIpmiHandler(SignalingIpmiHandler handler);
void UnregisterIpmiHandler();
s32 CheckAppType();
s32 StartMainRuntime(s32 thread_priority, s32 cpu_affinity_mask, s64 thread_stack_size);
s32 StartEchoRuntime(s32 thread_priority, s32 cpu_affinity_mask);
void ShutdownRuntime();

} // namespace Libraries::Np::NpSignaling::Helpers

namespace Libraries::Np::NpSignaling {

// Called from sceNpSignalingInitialize to register HLE thread start/stop with the Helpers layer.
// Also registers the IPMI handler stub internally.
void RegisterRuntimeHooks();

} // namespace Libraries::Np::NpSignaling
