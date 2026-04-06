// SPDX-FileCopyrightText: Copyright 2025 shadPS4 Emulator Project
// SPDX-License-Identifier: GPL-2.0-or-later

#pragma once

#include <string>

#include "common/types.h"

namespace Core::Loader {
class SymbolsResolver;
}

namespace Libraries::Np::NpSignaling {

using OrbisNpSignalingHandler = PS4_SYSV_ABI void (*)(u32 ctxId, u32 connId, s32 event,
                                                      s32 errorCode, void* userArg);

constexpr s32 ORBIS_NP_SIGNALING_EVENT_DEAD = 0;
constexpr s32 ORBIS_NP_SIGNALING_EVENT_ESTABLISHED = 1;
constexpr s32 ORBIS_NP_SIGNALING_EVENT_NETINFO_ERROR = 2;
constexpr s32 ORBIS_NP_SIGNALING_EVENT_NETINFO_RESULT = 3;
constexpr s32 ORBIS_NP_SIGNALING_EVENT_PEER_ACTIVATED = 10;
constexpr s32 ORBIS_NP_SIGNALING_EVENT_PEER_DEACTIVATED = 11;
constexpr s32 ORBIS_NP_SIGNALING_EVENT_MUTUAL_ACTIVATED = 12;

constexpr s32 ORBIS_NP_SIGNALING_CONN_STATUS_INACTIVE = 0;
constexpr s32 ORBIS_NP_SIGNALING_CONN_STATUS_PENDING = 1;
constexpr s32 ORBIS_NP_SIGNALING_CONN_STATUS_ACTIVE = 2;

constexpr s32 ORBIS_NP_SIGNALING_CONN_INFO_RTT = 1;
constexpr s32 ORBIS_NP_SIGNALING_CONN_INFO_BANDWIDTH = 2;
constexpr s32 ORBIS_NP_SIGNALING_CONN_INFO_PEER_NP_ID = 3;
constexpr s32 ORBIS_NP_SIGNALING_CONN_INFO_PEER_ADDR = 4;
constexpr s32 ORBIS_NP_SIGNALING_CONN_INFO_MAPPED_ADDR = 5;
constexpr s32 ORBIS_NP_SIGNALING_CONN_INFO_PACKET_LOSS = 6;

struct OrbisNpSignalingNetInfo {
    u64 size;       // +0x00: must be sizeof(OrbisNpSignalingNetInfo) = 0x18
    u32 localAddr;  // +0x08
    u32 mappedAddr; // +0x0C
    s32 natStatus;  // +0x10
    u32 _pad_14;    // +0x14
};
static_assert(sizeof(OrbisNpSignalingNetInfo) == 0x18);

// --- PS4 API functions ---

s32 PS4_SYSV_ABI sceNpSignalingActivateConnection(s32 ctxId, void* npId, s32* connId);
s32 PS4_SYSV_ABI sceNpSignalingActivateConnectionA();
s32 PS4_SYSV_ABI sceNpSignalingCancelPeerNetInfo(s32 ctxId, s32 connId);
s32 PS4_SYSV_ABI sceNpSignalingCreateContext(void* npId, void* callback, void* userArg,
                                             s32* context_id);
s32 PS4_SYSV_ABI sceNpSignalingCreateContextA();
s32 PS4_SYSV_ABI sceNpSignalingDeactivateConnection(s32 ctxId, s32 connId);
s32 PS4_SYSV_ABI sceNpSignalingDeleteContext(s32 ctxId);
s32 PS4_SYSV_ABI sceNpSignalingGetConnectionFromNpId(s32 ctxId, void* npId, s32* connId);
s32 PS4_SYSV_ABI sceNpSignalingGetConnectionFromPeerAddress(s32 ctxId, u32 peerAddr, u16 peerPort,
                                                            s32* connId);
s32 PS4_SYSV_ABI sceNpSignalingGetConnectionFromPeerAddressA(s32 ctxId, u32 peerAddr, u16 peerPort,
                                                             s32* connId);
s32 PS4_SYSV_ABI sceNpSignalingGetConnectionInfo(s32 ctxId, s32 connId, s32 infoType, void* info);
s32 PS4_SYSV_ABI sceNpSignalingGetConnectionInfoA();
s32 PS4_SYSV_ABI sceNpSignalingGetConnectionStatistics();
s32 PS4_SYSV_ABI sceNpSignalingGetConnectionStatus(s32 ctxId, s32 connId, s32* connStatus,
                                                   u32* peerAddr, u16* peerPort);
s32 PS4_SYSV_ABI sceNpSignalingGetContextOption(s32 ctxId, s32 opt, void* value, size_t valueSize);
s32 PS4_SYSV_ABI sceNpSignalingGetLocalNetInfo(s32 ctxId, OrbisNpSignalingNetInfo* info);
s32 PS4_SYSV_ABI sceNpSignalingGetMemoryInfo(void* info);
s32 PS4_SYSV_ABI sceNpSignalingGetPeerNetInfo(s32 ctxId, void* npId, s32 reqId);
s32 PS4_SYSV_ABI sceNpSignalingGetPeerNetInfoA(s32 ctxId, void* npId, s32 reqId);
s32 PS4_SYSV_ABI sceNpSignalingGetPeerNetInfoResult(s32 ctxId, s32 reqId, s32* result);
s32 PS4_SYSV_ABI sceNpSignalingInitialize(s64 poolSize, s32 threadPriority, s32 cpuAffinityMask,
                                          s64 threadStackSize);
s32 PS4_SYSV_ABI sceNpSignalingSetContextOption(s32 ctxId, s32 opt, const void* value,
                                                size_t valueSize);
s32 PS4_SYSV_ABI sceNpSignalingTerminate();
s32 PS4_SYSV_ABI sceNpSignalingTerminateConnection();

// Returns the conn_id of an ACTIVE signaling connection for the given peer,
// or 0 if no such connection exists.
s32 GetActiveConnectionIdForPeer(std::string_view online_id);

// Returns the conn_id and actual status of any existing signaling connection for
// the given peer (regardless of status). conn_id is set to 0 if none exists.
// Returns ORBIS_NP_SIGNALING_CONN_STATUS_INACTIVE if no connection is found.
s32 GetConnectionStatusForPeer(std::string_view online_id, s32* out_conn_id);

// Handles a server-driven NpSignaling event for the given peer.
// event_type: ORBIS_NP_SIGNALING_EVENT_ESTABLISHED (1) — sets ACTIVE, fires callback,
//             sends SignalingEstablished (cmd=0x02) to server.
//             Any other value — sets INACTIVE, fires callback.
void HandleServerNpSignalingEvent(std::string_view online_id, u32 event_type);

void RegisterLib(Core::Loader::SymbolsResolver* sym);
} // namespace Libraries::Np::NpSignaling
