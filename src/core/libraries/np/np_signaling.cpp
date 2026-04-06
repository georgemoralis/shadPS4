// SPDX-FileCopyrightText: Copyright 2025 shadPS4 Emulator Project
// SPDX-License-Identifier: GPL-2.0-or-later

// NpSignaling (sig1) HLE implementation.
//
// Responsibilities of this file:
//   1. Maintain signaling contexts and connection bookkeeping.
//   2. Discover the local context's external UDP endpoint via the MM UDP echo.
//   3. Resolve peer endpoints through Matching2 and send the 32-byte
//      ActivateConnection packet emitted by the PS4 library.
//   4. Deliver signaling events through the existing dispatch thread.
//
// Packet layout (all fields little-endian on x86):
//   ActivatePacket (32 bytes):
//     [0x00] u32              type            - always 5
//     [0x04] u32              conn_id         - local signaling connection id
//     [0x08] u32              ctx_tag         - context tag derived from the bound port
//     [0x0C] u8[20]           reserved
//
//   StunPing (21 bytes, client -> server UDP):
//     [0x00] u8   cmd            - always 0x01
//     [0x01] u8   online_id[16]  - null-padded
//     [0x11] u32  local_ip       - informational (ignored by server)
//
//   StunEcho (6 bytes, server -> client UDP):
//     [0x00] u32  ext_ip    - network byte order
//     [0x04] u16  ext_port  - network byte order

#include <cstring>
#include <vector>

#include "common/logging/log.h"
#include "core/libraries/error_codes.h"
#include "core/libraries/kernel/kernel.h"
#include "core/libraries/libs.h"
#include "core/libraries/network/net.h"
#include "core/libraries/network/sockets.h"
#include "core/libraries/np/np_matching2_mm.h"
#include "core/libraries/np/np_signaling.h"
#include "core/libraries/np/np_signaling_error.h"
#include "core/libraries/np/np_signaling_helpers.h"
#include "core/libraries/np/np_signaling_state.h"

namespace Libraries::Np::NpSignaling {

using Libraries::Net::sceNetHtons;
using Libraries::Net::sceNetNtohl;
using Libraries::Net::sceNetNtohs;

s32 PS4_SYSV_ABI sceNpSignalingInitialize(s64 poolSize, s32 threadPriority, s32 cpuAffinityMask,
                                          s64 threadStackSize) {
    std::lock_guard<std::mutex> lock(g_mutex);
    u32 app_type_4 = 0;

    if (g_initialized) {
        return ORBIS_NP_SIGNALING_ERROR_ALREADY_INITIALIZED;
    }

    const s32 app_type_rc = Helpers::CheckInitializeAppType(&app_type_4);
    if (app_type_rc < 0) {
        return app_type_rc;
    }
    if (app_type_4 != 0) {
        return ORBIS_NP_SIGNALING_ERROR_PROHIBITED_TO_USE;
    }

    if (poolSize == 0) {
        poolSize = 0x20000;
    }
    if (threadPriority == 0) {
        threadPriority = 700;
    }
    if (threadStackSize == 0) {
        threadStackSize = 0x4000;
    }

    LOG_INFO(Lib_NpSignaling,
             "Initialize: poolSize={} threadPriority={} cpuAffinityMask={} "
             "threadStackSize={}",
             poolSize, threadPriority, cpuAffinityMask, threadStackSize);

    // Staged FUN_0100c4b0 equivalent. This currently models only the flexible
    // memory portion of signaling heap setup; the libc mspace layer is still
    // intentionally omitted.
    const s32 heap_rc = Helpers::InitSignalingHeap(poolSize);
    if (heap_rc < 0) {
        return heap_rc;
    }

    // Staged FUN_0100bbd0 equivalent. RegisterRuntimeHooks now also calls
    // Helpers::RegisterIpmiHandler internally with the stub handler.
    RegisterRuntimeHooks();

    // Staged FUN_010079b0 equivalent. The control-flow shape is present, but
    // the helper still uses inferred local app-type state until a real NP-int
    // source exists in the codebase.
    const s32 check_app_type_rc = Helpers::CheckAppType();
    if (check_app_type_rc < 0) {
        Helpers::UnregisterIpmiHandler();
        Helpers::ShutdownSignalingHeap();
        return check_app_type_rc;
    }

    g_initialized = true;
    // Staged FUN_010029a0 equivalent. Today this still drives the existing HLE
    // dispatch/receive/ping threads through helper-owned runtime hooks.
    const s32 rc = Helpers::StartMainRuntime(threadPriority, cpuAffinityMask, threadStackSize);
    if (rc < 0) {
        g_initialized = false;
        // Roll back the helper-owned initialize stages in reverse order.
        Helpers::UnregisterIpmiHandler();
        Helpers::ShutdownSignalingHeap();
        return rc;
    }

    // Staged FUN_01004f10 equivalent. This is still a placeholder helper today,
    // but initialize now reflects the separate echo-runtime startup stage from
    // the decomped flow.
    const s32 echo_rc = Helpers::StartEchoRuntime(threadPriority, cpuAffinityMask);
    if (echo_rc < 0) {
        g_initialized = false;
        Helpers::ShutdownRuntime();
        Helpers::UnregisterIpmiHandler();
        Helpers::ShutdownSignalingHeap();
        return echo_rc;
    }
    return ORBIS_OK;
}

s32 PS4_SYSV_ABI sceNpSignalingTerminate() {
    {
        std::lock_guard<std::mutex> lock(g_mutex);
        if (!g_initialized) {
            return ORBIS_NP_SIGNALING_ERROR_NOT_INITIALIZED;
        }
        g_initialized = false;
    }

    // Staged FUN_010031a0 / FUN_0100bcd0 / FUN_0100c5a0 unwind chain. The
    // helper layer now owns the high-level shutdown ordering, even though the
    // runtime helper itself is still only partially implemented.
    Helpers::ShutdownRuntime();
    Helpers::UnregisterIpmiHandler();
    Helpers::ShutdownSignalingHeap();

    {
        std::lock_guard<std::mutex> lock(g_mutex);
        for (auto& [_, ctx] : g_contexts) {
            // No socket to close — P2P transport is shared.
        }
        g_contexts.clear();
        g_connections.clear();
        g_npid_to_conn.clear();
    }

    LOG_INFO(Lib_NpSignaling, "sceNpSignalingTerminate: cleared all state");
    return ORBIS_OK;
}

s32 PS4_SYSV_ABI sceNpSignalingCreateContext(void* npId, void* callback, void* userArg,
                                             s32* context_id) {
    OrbisNpId owner_npid{};
    OrbisNpOnlineId owner_online_id{};

    if (NormalizeNpId(npId, &owner_npid, &owner_online_id) != ORBIS_OK || !context_id) {
        return ORBIS_NP_SIGNALING_ERROR_INVALID_ARGUMENT;
    }

    s32 ctx_id = 0;

    {
        std::lock_guard<std::mutex> lock(g_mutex);
        if (!g_initialized) {
            return ORBIS_NP_SIGNALING_ERROR_NOT_INITIALIZED;
        }

        ctx_id = AllocateContextIdLocked();
        if (ctx_id < 0) {
            return ORBIS_NP_SIGNALING_ERROR_CTXID_NOT_AVAILABLE;
        }

        // No socket creation here — signaling sends and receives through
        // the game's P2P shared transport socket (created by the game via
        // sceNetSocket).  This ensures the NAT hole is punched on the same
        // port the game will use for data, and avoids port conflicts.
        NpSignalingContext& ctx = g_contexts[ctx_id];
        ctx.callback = reinterpret_cast<OrbisNpSignalingHandler>(callback);
        ctx.callback_arg = userArg;
        ctx.active = true;
        ctx.owner_npid = owner_npid;
        ctx.owner_online_id = owner_online_id;
        ctx.bound_port = Net::GetP2PConfiguredPort();
        *context_id = ctx_id;

        LOG_INFO(Lib_NpSignaling,
                 "sceNpSignalingCreateContext: ctxId={} owner='{}' callback={} "
                 "userArg={} p2p_port={}",
                 ctx_id, OnlineIdFromOnlineId(owner_online_id), fmt::ptr(callback),
                 fmt::ptr(userArg), ctx.bound_port);
    }

    // Send STUN ping if P2P transport is already up.
    if (Net::P2PTransportIsReady()) {
        SendStunPing(ctx_id);
    }

    return ORBIS_OK;
}

s32 PS4_SYSV_ABI sceNpSignalingCreateContextA() {
    return ORBIS_OK;
}

s32 PS4_SYSV_ABI sceNpSignalingActivateConnection(s32 ctxId, void* npId, s32* connId) {
    LOG_INFO(Lib_NpSignaling,
             "t={} sceNpSignalingActivateConnection: ctxId={} npId={:p} connId={:p}", NowMs(),
             ctxId, npId, fmt::ptr(connId));

    OrbisNpId peer_npid{};
    OrbisNpOnlineId peer_online_id{};
    if (NormalizeNpId(npId, &peer_npid, &peer_online_id) != ORBIS_OK || !connId) {
        return ORBIS_NP_SIGNALING_ERROR_INVALID_ARGUMENT;
    }
    const std::string peer_online_id_str = OnlineIdFromOnlineId(peer_online_id);

    s32 cid = 0;
    u32 peer_addr = 0;
    u16 peer_port = 0;
    u32 ctx_tag = 0;
    OrbisNpOnlineId my_online_id{};
    {
        std::lock_guard<std::mutex> lock(g_mutex);
        if (!g_initialized) {
            return ORBIS_NP_SIGNALING_ERROR_NOT_INITIALIZED;
        }
        const auto ctx_it = g_contexts.find(ctxId);
        if (ctx_it == g_contexts.end() || !ctx_it->second.active) {
            return ORBIS_NP_SIGNALING_ERROR_CTX_NOT_FOUND;
        }
        if (std::memcmp(&ctx_it->second.owner_npid, &peer_npid, sizeof(peer_npid)) == 0) {
            return ORBIS_NP_SIGNALING_ERROR_OWN_NP_ID;
        }
        if (!ConsumeActivationBudgetLocked(ctx_it->second)) {
            return ORBIS_NP_SIGNALING_ERROR_EXCEED_RATE_LIMIT;
        }
        ctx_tag = static_cast<u32>(ctx_it->second.bound_port);
        my_online_id = ctx_it->second.owner_online_id;

        // Always allocate a fresh connection — connection objects are never reused.
        cid = AllocateConnectionIdLocked();
        if (cid < 0) {
            return ORBIS_NP_SIGNALING_ERROR_OUT_OF_MEMORY;
        }
        ConnectionInfo ci{};
        ci.conn_id = cid;
        ci.ctx_id = ctxId;
        ci.status = ORBIS_NP_SIGNALING_CONN_STATUS_PENDING;
        ci.npid = peer_npid;
        ci.online_id = peer_online_id;
        g_connections[cid] = std::move(ci);
        // Update lookup table to point to the latest conn_id for this (ctx, peer) pair.
        g_npid_to_conn[MakeCtxNpIdKey(ctxId, peer_npid)] = cid;
        LOG_INFO(Lib_NpSignaling,
                 "sceNpSignalingActivateConnection: created connId={} for ctxId={} peer='{}'", cid,
                 ctxId, peer_online_id_str);
    }

    if (!NpMatching2::RequestSignalingInfos(peer_online_id_str, &peer_addr, &peer_port) ||
        peer_addr == 0 || peer_port == 0) {
        std::lock_guard<std::mutex> lock(g_mutex);
        RemoveConnectionLocked(cid);
        return ORBIS_NP_SIGNALING_ERROR_PEER_UNREACHABLE;
    }

    {
        std::lock_guard<std::mutex> lock(g_mutex);
        const auto it = g_connections.find(cid);
        if (it == g_connections.end()) {
            return ORBIS_NP_SIGNALING_ERROR_CONN_NOT_FOUND;
        }
        it->second.addr = peer_addr;
        it->second.port = peer_port;
    }

    *connId = cid;

    LOG_INFO(Lib_NpSignaling,
             "t={} sceNpSignalingActivateConnection: ctxId={} peer='{}' connId={} "
             "peer_addr={:#x} peer_port={} ctx_tag={:#x}",
             NowMs(), ctxId, peer_online_id_str, cid, peer_addr, sceNetNtohs(peer_port), ctx_tag);

    const s32 send_rc = SendActivatePacket(peer_addr, peer_port, static_cast<u32>(cid), ctx_tag);
    if (send_rc != ORBIS_OK) {
        std::lock_guard<std::mutex> lock(g_mutex);
        RemoveConnectionLocked(cid);
        return send_rc;
    }

    // Notify the server of our pending activation intent.
    const u32 server_addr = NpMatching2::GetMmServerAddr();
    const u16 server_udp = NpMatching2::GetMmServerUdpPort();
    if (server_addr != 0 && server_udp != 0) {
        ActivationIntent intent{};
        std::memcpy(intent.online_id_me, my_online_id.data, ORBIS_NP_ONLINEID_MAX_LENGTH);
        std::memcpy(intent.online_id_peer, peer_online_id.data, ORBIS_NP_ONLINEID_MAX_LENGTH);
        intent.ctx_tag = ctx_tag; // host byte order — server parses as little-endian u32
        Net::P2PSignalingSendTo(&intent, sizeof(intent), server_addr, server_udp);
        LOG_INFO(Lib_NpSignaling, "ActivationIntent: me='{}' peer='{}' ctx_tag={:#x}",
                 OnlineIdFromOnlineId(my_online_id), peer_online_id_str, ctx_tag);
    }

    // Connection remains PENDING — event 1 fires when server confirms handshake.
    return ORBIS_OK;
}

s32 PS4_SYSV_ABI sceNpSignalingActivateConnectionA() {
    return ORBIS_OK;
}

s32 PS4_SYSV_ABI sceNpSignalingDeactivateConnection(s32 ctxId, s32 connId) {
    {
        std::lock_guard<std::mutex> lock(g_mutex);
        if (!g_initialized) {
            return ORBIS_NP_SIGNALING_ERROR_NOT_INITIALIZED;
        }
        if (!IsContextValidLocked(ctxId)) {
            return ORBIS_NP_SIGNALING_ERROR_CTX_NOT_FOUND;
        }

        const auto it = g_connections.find(connId);
        if (it == g_connections.end() || it->second.ctx_id != ctxId) {
            return ORBIS_NP_SIGNALING_ERROR_CONN_NOT_FOUND;
        }

        ConnectionInfo& ci = it->second;
        LOG_INFO(Lib_NpSignaling,
                 "t={} sceNpSignalingDeactivateConnection: ctxId={} connId={} "
                 "peer='{}' status={} -> INACTIVE",
                 NowMs(), ctxId, connId, OnlineIdFromOnlineId(ci.online_id), ci.status);
    }

    UpdateConnStatus(ctxId, connId, ORBIS_NP_SIGNALING_CONN_STATUS_INACTIVE);

    return ORBIS_OK;
}

s32 PS4_SYSV_ABI sceNpSignalingDeleteContext(s32 ctxId) {
    std::vector<s32> active_conns;
    {
        std::lock_guard<std::mutex> lock(g_mutex);

        if (!g_initialized) {
            return ORBIS_NP_SIGNALING_ERROR_NOT_INITIALIZED;
        }

        const auto it = g_contexts.find(ctxId);
        if (it == g_contexts.end()) {
            return ORBIS_NP_SIGNALING_ERROR_CTX_NOT_FOUND;
        }

        LOG_INFO(Lib_NpSignaling, "sceNpSignalingDeleteContext: ctxId={}", ctxId);

        for (const auto& [cid, ci] : g_connections) {
            if (ci.ctx_id == ctxId && ci.status != ORBIS_NP_SIGNALING_CONN_STATUS_INACTIVE) {
                active_conns.push_back(cid);
            }
        }
    }

    for (const s32 cid : active_conns) {
        UpdateConnStatus(ctxId, cid, ORBIS_NP_SIGNALING_CONN_STATUS_INACTIVE);
    }

    {
        std::lock_guard<std::mutex> lock(g_mutex);
        const auto it = g_contexts.find(ctxId);
        if (it != g_contexts.end()) {
            RemoveContextConnectionsLocked(ctxId);
            // No socket to close — P2P transport is shared.
            g_contexts.erase(it);
        }
    }
    return ORBIS_OK;
}

s32 PS4_SYSV_ABI sceNpSignalingGetConnectionStatus(s32 ctxId, s32 connId, s32* connStatus,
                                                   u32* peerAddr, u16* peerPort) {
    std::lock_guard<std::mutex> lock(g_mutex);

    if (!g_initialized) {
        return ORBIS_NP_SIGNALING_ERROR_NOT_INITIALIZED;
    }
    if (!connStatus) {
        return ORBIS_NP_SIGNALING_ERROR_INVALID_ARGUMENT;
    }
    if (!IsContextValidLocked(ctxId)) {
        return ORBIS_NP_SIGNALING_ERROR_CTX_NOT_FOUND;
    }

    const auto it = g_connections.find(connId);
    if (it == g_connections.end() || it->second.ctx_id != ctxId) {
        *connStatus = ORBIS_NP_SIGNALING_CONN_STATUS_INACTIVE;
        return ORBIS_OK;
    }

    *connStatus = it->second.status;
    if (it->second.status == ORBIS_NP_SIGNALING_CONN_STATUS_ACTIVE) {
        if (peerAddr) {
            *peerAddr = it->second.addr;
        }
        if (peerPort) {
            *peerPort = it->second.port;
        }
    }

    const char* status_name = (it->second.status == 0)   ? "INACTIVE"
                              : (it->second.status == 1) ? "PENDING"
                              : (it->second.status == 2) ? "ACTIVE"
                                                         : "UNKNOWN";
    LOG_INFO(Lib_NpSignaling,
             "t={} GetConnectionStatus: ctxId={} connId={} peer='{}' status={}({})", NowMs(), ctxId,
             connId, OnlineIdFromOnlineId(it->second.online_id), it->second.status, status_name);
    return ORBIS_OK;
}

s32 PS4_SYSV_ABI sceNpSignalingCancelPeerNetInfo(s32 ctxId, s32 connId) {
    std::lock_guard<std::mutex> lock(g_mutex);
    if (!g_initialized) {
        return ORBIS_NP_SIGNALING_ERROR_NOT_INITIALIZED;
    }
    if (!IsContextValidLocked(ctxId)) {
        return ORBIS_NP_SIGNALING_ERROR_CTX_NOT_FOUND;
    }
    if (g_connections.find(connId) == g_connections.end()) {
        return ORBIS_NP_SIGNALING_ERROR_CONN_NOT_FOUND;
    }
    LOG_INFO(Lib_NpSignaling, "sceNpSignalingCancelPeerNetInfo: ctxId={} connId={}", ctxId, connId);
    return ORBIS_OK;
}

s32 PS4_SYSV_ABI sceNpSignalingGetConnectionFromNpId(s32 ctxId, void* npId, s32* connId) {
    OrbisNpId peer_npid{};
    OrbisNpOnlineId peer_online_id{};
    if (NormalizeNpId(npId, &peer_npid, &peer_online_id) != ORBIS_OK || !connId) {
        return ORBIS_NP_SIGNALING_ERROR_INVALID_ARGUMENT;
    }

    std::lock_guard<std::mutex> lock(g_mutex);
    if (!g_initialized) {
        return ORBIS_NP_SIGNALING_ERROR_NOT_INITIALIZED;
    }
    if (!IsContextValidLocked(ctxId)) {
        return ORBIS_NP_SIGNALING_ERROR_CTX_NOT_FOUND;
    }

    const auto it = g_npid_to_conn.find(MakeCtxNpIdKey(ctxId, peer_npid));
    if (it == g_npid_to_conn.end()) {
        return ORBIS_NP_SIGNALING_ERROR_CONN_NOT_FOUND;
    }

    *connId = it->second;
    LOG_INFO(Lib_NpSignaling, "GetConnectionFromNpId: ctxId={} npid='{}' connId={}", ctxId,
             OnlineIdFromOnlineId(peer_online_id), it->second);
    return ORBIS_OK;
}

s32 PS4_SYSV_ABI sceNpSignalingGetConnectionFromPeerAddress(s32 ctxId, u32 peerAddr, u16 peerPort,
                                                            s32* connId) {
    if (!connId) {
        return ORBIS_NP_SIGNALING_ERROR_INVALID_ARGUMENT;
    }

    std::lock_guard<std::mutex> lock(g_mutex);
    if (!g_initialized) {
        return ORBIS_NP_SIGNALING_ERROR_NOT_INITIALIZED;
    }
    if (!IsContextValidLocked(ctxId)) {
        return ORBIS_NP_SIGNALING_ERROR_CTX_NOT_FOUND;
    }

    for (const auto& [id, ci] : g_connections) {
        if (ci.ctx_id == ctxId && ci.addr == peerAddr && ci.port == peerPort) {
            *connId = id;
            LOG_INFO(Lib_NpSignaling,
                     "GetConnectionFromPeerAddress: ctxId={} addr={:#x} port={} connId={}", ctxId,
                     peerAddr, sceNetNtohs(peerPort), id);
            return ORBIS_OK;
        }
    }
    return ORBIS_NP_SIGNALING_ERROR_CONN_NOT_FOUND;
}

s32 PS4_SYSV_ABI sceNpSignalingGetConnectionFromPeerAddressA(s32 ctxId, u32 peerAddr, u16 peerPort,
                                                             s32* connId) {
    return sceNpSignalingGetConnectionFromPeerAddress(ctxId, peerAddr, peerPort, connId);
}

s32 PS4_SYSV_ABI sceNpSignalingGetConnectionInfo(s32 ctxId, s32 connId, s32 infoType, void* info) {
    if (!info) {
        return ORBIS_NP_SIGNALING_ERROR_INVALID_ARGUMENT;
    }

    std::lock_guard<std::mutex> lock(g_mutex);
    if (!g_initialized) {
        return ORBIS_NP_SIGNALING_ERROR_NOT_INITIALIZED;
    }
    if (!IsContextValidLocked(ctxId)) {
        return ORBIS_NP_SIGNALING_ERROR_CTX_NOT_FOUND;
    }

    const auto it = g_connections.find(connId);
    if (it == g_connections.end() || it->second.ctx_id != ctxId) {
        return ORBIS_NP_SIGNALING_ERROR_CONN_NOT_FOUND;
    }

    const ConnectionInfo& ci = it->second;
    switch (infoType) {
    case ORBIS_NP_SIGNALING_CONN_INFO_RTT:
        *reinterpret_cast<s32*>(info) = static_cast<s32>(ci.rtt_us / 1000); // report in ms
        break;
    case ORBIS_NP_SIGNALING_CONN_INFO_BANDWIDTH:
        *reinterpret_cast<s32*>(info) = static_cast<s32>(ci.bandwidth);
        break;
    case ORBIS_NP_SIGNALING_CONN_INFO_PEER_NP_ID:
        std::memcpy(info, &ci.npid, sizeof(ci.npid));
        break;
    case ORBIS_NP_SIGNALING_CONN_INFO_PEER_ADDR:
        *reinterpret_cast<u32*>(info) = ci.addr;
        *reinterpret_cast<u16*>(reinterpret_cast<u8*>(info) + 4) = ci.port;
        break;
    case ORBIS_NP_SIGNALING_CONN_INFO_MAPPED_ADDR:
        *reinterpret_cast<u32*>(info) = ci.addr;
        *reinterpret_cast<u16*>(reinterpret_cast<u8*>(info) + 4) = ci.port;
        break;
    case ORBIS_NP_SIGNALING_CONN_INFO_PACKET_LOSS:
        *reinterpret_cast<u32*>(info) = ci.packet_loss;
        break;
    default:
        return ORBIS_NP_SIGNALING_ERROR_INVALID_ARGUMENT;
    }

    LOG_INFO(Lib_NpSignaling, "GetConnectionInfo: ctxId={} connId={} infoType={} peer='{}'", ctxId,
             connId, infoType, OnlineIdFromOnlineId(ci.online_id));
    return ORBIS_OK;
}

s32 PS4_SYSV_ABI sceNpSignalingGetConnectionInfoA() {
    return ORBIS_OK;
}

s32 PS4_SYSV_ABI sceNpSignalingGetConnectionStatistics() {
    if (!g_initialized) {
        return ORBIS_NP_SIGNALING_ERROR_NOT_INITIALIZED;
    }
    return ORBIS_OK;
}

s32 PS4_SYSV_ABI sceNpSignalingGetContextOption(s32 ctxId, s32 opt, void* value, size_t valueSize) {
    std::lock_guard<std::mutex> lock(g_mutex);
    if (!g_initialized) {
        return ORBIS_NP_SIGNALING_ERROR_NOT_INITIALIZED;
    }
    if (!IsContextValidLocked(ctxId)) {
        return ORBIS_NP_SIGNALING_ERROR_CTX_NOT_FOUND;
    }
    if (!value && valueSize != 0) {
        return ORBIS_NP_SIGNALING_ERROR_INVALID_ARGUMENT;
    }
    (void)opt;
    return ORBIS_OK;
}

s32 PS4_SYSV_ABI sceNpSignalingGetLocalNetInfo(s32 ctxId, OrbisNpSignalingNetInfo* info) {
    std::lock_guard<std::mutex> lock(g_mutex);
    if (!g_initialized) {
        return ORBIS_NP_SIGNALING_ERROR_NOT_INITIALIZED;
    }
    if (!IsContextValidLocked(ctxId)) {
        return ORBIS_NP_SIGNALING_ERROR_CTX_NOT_FOUND;
    }
    if (!info) {
        return ORBIS_NP_SIGNALING_ERROR_INVALID_ARGUMENT;
    }

    std::memset(info, 0, sizeof(*info));
    info->size = sizeof(OrbisNpSignalingNetInfo);

    // Prefer the STUN-resolved external address; fall back to 0 if not yet known.
    const auto ctx_it = g_contexts.find(ctxId);
    const u32 ext_addr = (ctx_it != g_contexts.end()) ? ctx_it->second.ext_addr.load() : 0u;

    info->localAddr = ext_addr;
    info->mappedAddr = ext_addr;
    info->natStatus = 2; // Type 2: NAT with mapped address (conservative assumption)

    LOG_INFO(Lib_NpSignaling, "GetLocalNetInfo: ctxId={} ext_addr={:#x} natStatus=2", ctxId,
             ext_addr);
    return ORBIS_OK;
}

s32 PS4_SYSV_ABI sceNpSignalingGetMemoryInfo(void* info) {
    if (!g_initialized) {
        return ORBIS_NP_SIGNALING_ERROR_NOT_INITIALIZED;
    }
    if (!info) {
        return ORBIS_NP_SIGNALING_ERROR_INVALID_ARGUMENT;
    }
    return ORBIS_OK;
}

s32 PS4_SYSV_ABI sceNpSignalingGetPeerNetInfo(s32 ctxId, void* npId, s32 reqId) {
    std::lock_guard<std::mutex> lock(g_mutex);
    if (!g_initialized) {
        return ORBIS_NP_SIGNALING_ERROR_NOT_INITIALIZED;
    }
    if (!IsContextValidLocked(ctxId) || !npId || reqId < 0) {
        return ORBIS_NP_SIGNALING_ERROR_INVALID_ARGUMENT;
    }
    return ORBIS_OK;
}

s32 PS4_SYSV_ABI sceNpSignalingGetPeerNetInfoA(s32 ctxId, void* npId, s32 reqId) {
    return sceNpSignalingGetPeerNetInfo(ctxId, npId, reqId);
}

s32 PS4_SYSV_ABI sceNpSignalingGetPeerNetInfoResult(s32 ctxId, s32 reqId, s32* result) {
    std::lock_guard<std::mutex> lock(g_mutex);
    if (!g_initialized) {
        return ORBIS_NP_SIGNALING_ERROR_NOT_INITIALIZED;
    }
    if (!IsContextValidLocked(ctxId) || !result || reqId < 0) {
        return ORBIS_NP_SIGNALING_ERROR_INVALID_ARGUMENT;
    }
    *result = 0;
    return ORBIS_OK;
}

s32 PS4_SYSV_ABI sceNpSignalingSetContextOption(s32 ctxId, s32 opt, const void* value,
                                                size_t valueSize) {
    std::lock_guard<std::mutex> lock(g_mutex);
    if (!g_initialized) {
        return ORBIS_NP_SIGNALING_ERROR_NOT_INITIALIZED;
    }
    if (!IsContextValidLocked(ctxId)) {
        return ORBIS_NP_SIGNALING_ERROR_CTX_NOT_FOUND;
    }
    if (!value && valueSize != 0) {
        return ORBIS_NP_SIGNALING_ERROR_INVALID_ARGUMENT;
    }
    (void)opt;
    return ORBIS_OK;
}

s32 PS4_SYSV_ABI sceNpSignalingTerminateConnection() {
    if (!g_initialized) {
        return ORBIS_NP_SIGNALING_ERROR_NOT_INITIALIZED;
    }
    return ORBIS_OK;
}

void RegisterLib(Core::Loader::SymbolsResolver* sym) {
    LIB_FUNCTION("0UvTFeomAUM", "libSceNpSignaling", 1, "libSceNpSignaling",
                 sceNpSignalingActivateConnection);
    LIB_FUNCTION("ZPLavCKqAB0", "libSceNpSignaling", 1, "libSceNpSignaling",
                 sceNpSignalingActivateConnectionA);
    LIB_FUNCTION("X1G4kkN2R-8", "libSceNpSignaling", 1, "libSceNpSignaling",
                 sceNpSignalingCancelPeerNetInfo);
    LIB_FUNCTION("5yYjEdd4t8Y", "libSceNpSignaling", 1, "libSceNpSignaling",
                 sceNpSignalingCreateContext);
    LIB_FUNCTION("dDLNFdY8dws", "libSceNpSignaling", 1, "libSceNpSignaling",
                 sceNpSignalingCreateContextA);
    LIB_FUNCTION("6UEembipgrM", "libSceNpSignaling", 1, "libSceNpSignaling",
                 sceNpSignalingDeactivateConnection);
    LIB_FUNCTION("hx+LIg-1koI", "libSceNpSignaling", 1, "libSceNpSignaling",
                 sceNpSignalingDeleteContext);
    LIB_FUNCTION("GQ0hqmzj0F4", "libSceNpSignaling", 1, "libSceNpSignaling",
                 sceNpSignalingGetConnectionFromNpId);
    LIB_FUNCTION("CkPxQjSm018", "libSceNpSignaling", 1, "libSceNpSignaling",
                 sceNpSignalingGetConnectionFromPeerAddress);
    LIB_FUNCTION("B7cT9aVby7A", "libSceNpSignaling", 1, "libSceNpSignaling",
                 sceNpSignalingGetConnectionFromPeerAddressA);
    LIB_FUNCTION("AN3h0EBSX7A", "libSceNpSignaling", 1, "libSceNpSignaling",
                 sceNpSignalingGetConnectionInfo);
    LIB_FUNCTION("rcylknsUDwg", "libSceNpSignaling", 1, "libSceNpSignaling",
                 sceNpSignalingGetConnectionInfoA);
    LIB_FUNCTION("C6ZNCDTj00Y", "libSceNpSignaling", 1, "libSceNpSignaling",
                 sceNpSignalingGetConnectionStatistics);
    LIB_FUNCTION("bD-JizUb3JM", "libSceNpSignaling", 1, "libSceNpSignaling",
                 sceNpSignalingGetConnectionStatus);
    LIB_FUNCTION("npU5V56id34", "libSceNpSignaling", 1, "libSceNpSignaling",
                 sceNpSignalingGetContextOption);
    LIB_FUNCTION("U8AQMlOFBc8", "libSceNpSignaling", 1, "libSceNpSignaling",
                 sceNpSignalingGetLocalNetInfo);
    LIB_FUNCTION("tOpqyDyMje4", "libSceNpSignaling", 1, "libSceNpSignaling",
                 sceNpSignalingGetMemoryInfo);
    LIB_FUNCTION("zFgFHId7vAE", "libSceNpSignaling", 1, "libSceNpSignaling",
                 sceNpSignalingGetPeerNetInfo);
    LIB_FUNCTION("Shr7bZq8QHY", "libSceNpSignaling", 1, "libSceNpSignaling",
                 sceNpSignalingGetPeerNetInfoA);
    LIB_FUNCTION("2HajCEGgG4s", "libSceNpSignaling", 1, "libSceNpSignaling",
                 sceNpSignalingGetPeerNetInfoResult);
    LIB_FUNCTION("3KOuC4RmZZU", "libSceNpSignaling", 1, "libSceNpSignaling",
                 sceNpSignalingInitialize);
    LIB_FUNCTION("IHRDvZodPYY", "libSceNpSignaling", 1, "libSceNpSignaling",
                 sceNpSignalingSetContextOption);
    LIB_FUNCTION("NPhw0UXaNrk", "libSceNpSignaling", 1, "libSceNpSignaling",
                 sceNpSignalingTerminate);
    LIB_FUNCTION("b4qaXPzMJxo", "libSceNpSignaling", 1, "libSceNpSignaling",
                 sceNpSignalingTerminateConnection);
}

} // namespace Libraries::Np::NpSignaling
