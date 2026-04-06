// SPDX-FileCopyrightText: Copyright 2025 shadPS4 Emulator Project
// SPDX-License-Identifier: GPL-2.0-or-later

#include <algorithm>
#include <cctype>
#include <cstring>

#include "common/logging/log.h"
#include "core/libraries/error_codes.h"
#include "core/libraries/kernel/kernel.h"
#include "core/libraries/network/net.h"
#include "core/libraries/network/sockets.h"
#include "core/libraries/np/np_matching2_mm.h"
#include "core/libraries/np/np_signaling_error.h"
#include "core/libraries/np/np_signaling_state.h"

namespace Libraries::Np::NpSignaling {

using Libraries::Net::sceNetNtohl;
using Libraries::Net::sceNetNtohs;

std::unordered_map<s32, NpSignalingContext> g_contexts;
std::unordered_map<s32, ConnectionInfo> g_connections;
std::unordered_map<CtxNpIdKey, s32, CtxNpIdKeyHash> g_npid_to_conn;
bool g_initialized = false;
std::mutex g_mutex;
std::multimap<std::chrono::steady_clock::time_point, QueuedDispatch> g_dispatch_queue;
std::mutex g_dispatch_mutex;
std::condition_variable g_dispatch_cv;
bool g_dispatch_stop = false;

long long NowMs() {
    return std::chrono::duration_cast<std::chrono::milliseconds>(
               std::chrono::steady_clock::now().time_since_epoch())
        .count();
}

s64 NowUs() {
    return std::chrono::duration_cast<std::chrono::microseconds>(
               std::chrono::steady_clock::now().time_since_epoch())
        .count();
}

std::string OnlineIdFromOnlineId(const OrbisNpOnlineId& online_id_in) {
    std::string online_id(online_id_in.data, ORBIS_NP_ONLINEID_MAX_LENGTH);
    const auto nul = online_id.find('\0');
    if (nul != std::string::npos) {
        online_id.resize(nul);
    }
    return online_id;
}

std::string OnlineIdFromNpId(const OrbisNpId& np_id) {
    return OnlineIdFromOnlineId(np_id.handle);
}

OrbisNpId NpIdFromOnlineId(const OrbisNpOnlineId& online_id) {
    OrbisNpId npid{};
    npid.handle = online_id;
    return npid;
}

bool OnlineIdEqualsString(const OrbisNpOnlineId& online_id, std::string_view value) {
    return OnlineIdFromOnlineId(online_id) == value;
}

bool IsValidNpId(const OrbisNpId& np_id) {
    const auto id = OnlineIdFromNpId(np_id);
    if (id.size() > ORBIS_NP_ONLINEID_MAX_LENGTH) {
        return false;
    }
    for (unsigned char c : id) {
        if (std::isalnum(c) || c == '_' || c == '-' || c == '.') {
            continue;
        }
        return false;
    }
    return !id.empty();
}

s32 NormalizeNpId(const void* np_id, OrbisNpId* out_npid, OrbisNpOnlineId* out_online_id) {
    if (!np_id || !out_npid) {
        return ORBIS_NP_SIGNALING_ERROR_INVALID_ARGUMENT;
    }
    std::memcpy(out_npid, np_id, sizeof(*out_npid));
    if (!IsValidNpId(*out_npid)) {
        return ORBIS_NP_SIGNALING_ERROR_INVALID_ARGUMENT;
    }
    if (out_online_id) {
        *out_online_id = out_npid->handle;
    }
    return ORBIS_OK;
}

CtxNpIdKey MakeCtxNpIdKey(s32 ctx_id, const OrbisNpId& npid) {
    CtxNpIdKey key{};
    key.ctx_id = ctx_id;
    key.npid = npid;
    return key;
}

bool IsContextValidLocked(s32 ctx_id) {
    const auto it = g_contexts.find(ctx_id);
    return it != g_contexts.end() && it->second.active;
}

s32 AllocateContextIdLocked() {
    for (s32 id = 1; id <= 8; ++id) {
        if (g_contexts.find(id) == g_contexts.end()) {
            return id;
        }
    }
    return -1;
}

s32 AllocateConnectionIdLocked() {
    for (s32 id = 1; id <= 0xffff; ++id) {
        if (g_connections.find(id) == g_connections.end()) {
            return id;
        }
    }
    return -1;
}

void RemoveConnectionLocked(s32 conn_id) {
    const auto it = g_connections.find(conn_id);
    if (it == g_connections.end()) {
        return;
    }
    const auto map_key = MakeCtxNpIdKey(it->second.ctx_id, it->second.npid);
    auto key_it = g_npid_to_conn.find(map_key);
    if (key_it != g_npid_to_conn.end() && key_it->second == conn_id) {
        g_npid_to_conn.erase(key_it);
    }
    g_connections.erase(it);
}

void RemoveContextConnectionsLocked(s32 ctx_id) {
    std::vector<s32> to_remove;
    for (const auto& [cid, ci] : g_connections) {
        if (ci.ctx_id == ctx_id) {
            to_remove.push_back(cid);
        }
    }
    for (s32 cid : to_remove) {
        RemoveConnectionLocked(cid);
    }
}

const char* SignalingEventName(u32 event_type) {
    switch (event_type) {
    case ORBIS_NP_SIGNALING_EVENT_DEAD:
        return "DEAD";
    case ORBIS_NP_SIGNALING_EVENT_ESTABLISHED:
        return "ESTABLISHED";
    case ORBIS_NP_SIGNALING_EVENT_NETINFO_ERROR:
        return "NETINFO_ERROR";
    case ORBIS_NP_SIGNALING_EVENT_NETINFO_RESULT:
        return "NETINFO_RESULT";
    case ORBIS_NP_SIGNALING_EVENT_PEER_ACTIVATED:
        return "PEER_ACTIVATED";
    case ORBIS_NP_SIGNALING_EVENT_PEER_DEACTIVATED:
        return "PEER_DEACTIVATED";
    case ORBIS_NP_SIGNALING_EVENT_MUTUAL_ACTIVATED:
        return "MUTUAL_ACTIVATED";
    default:
        return "UNKNOWN";
    }
}

void DeliverSignalingEventForCtx(s32 ctx_id, s32 conn_id, u32 event_type, u32 delay_ms) {
    OrbisNpSignalingHandler callback = nullptr;
    void* callback_arg = nullptr;

    {
        std::lock_guard<std::mutex> lock(g_mutex);
        const auto it = g_contexts.find(ctx_id);
        if (it != g_contexts.end() && it->second.active) {
            callback = it->second.callback;
            callback_arg = it->second.callback_arg;
        }
    }

    if (!callback) {
        return;
    }

    QueuedDispatch dispatch;
    dispatch.fire_at = std::chrono::steady_clock::now() + std::chrono::milliseconds(delay_ms);
    dispatch.delay_ms = delay_ms;
    dispatch.ctx_id = ctx_id;
    dispatch.conn_id = conn_id;
    dispatch.event_type = event_type;
    dispatch.callback = callback;
    dispatch.callback_arg = callback_arg;

    {
        std::lock_guard<std::mutex> lock(g_dispatch_mutex);
        if (g_dispatch_stop) {
            return;
        }
        g_dispatch_queue.emplace(dispatch.fire_at, std::move(dispatch));
    }
    g_dispatch_cv.notify_all();
}

bool ConsumeActivationBudgetLocked(NpSignalingContext& ctx) {
    const s64 now = NowUs();

    s64 budget = ctx.activate_budget_us;
    if (ctx.activate_last_update_us == 0) {
        budget = kActivateBudgetMaxUs;
    } else {
        const s64 elapsed = std::max<s64>(0, now - ctx.activate_last_update_us);
        budget = std::min<s64>(kActivateBudgetMaxUs, budget + elapsed);
    }

    ctx.activate_last_update_us = now;
    ctx.activate_budget_us = budget;

    if (budget < kActivateCooldownUs) {
        return false;
    }

    ctx.activate_budget_us = budget - kActivateCooldownUs;
    return true;
}

s32 SendActivatePacket(u32 peer_addr, u16 peer_port, u32 conn_id, u32 ctx_tag) {
    ActivatePacket pkt{};
    pkt.type = 5;
    pkt.conn_id = conn_id;
    pkt.ctx_tag = ctx_tag;

    const int rc = Net::P2PSignalingSendTo(&pkt, sizeof(pkt), peer_addr, peer_port);
    if (rc < 0) {
        return *Libraries::Kernel::__Error();
    }
    return ORBIS_OK;
}

void UpdateConnStatus(s32 ctx_id, s32 conn_id, s32 new_status) {
    s32 old_status = ORBIS_NP_SIGNALING_CONN_STATUS_INACTIVE;

    {
        std::lock_guard<std::mutex> lock(g_mutex);
        const auto it = g_connections.find(conn_id);
        if (it == g_connections.end() || it->second.ctx_id != ctx_id) {
            return;
        }

        ConnectionInfo& ci = it->second;
        old_status = ci.status;

        if (old_status == new_status) {
            return;
        }

        ci.status = new_status;
    }

    LOG_INFO(Lib_NpSignaling, "UpdateConnStatus: ctxId={} connId={} {} -> {}", ctx_id, conn_id,
             old_status, new_status);
}

void SendStunPing(s32 ctx_id) {
    OrbisNpOnlineId online_id{};

    {
        std::lock_guard<std::mutex> lock(g_mutex);
        const auto it = g_contexts.find(ctx_id);
        if (it == g_contexts.end() || !it->second.active) {
            return;
        }
        online_id = it->second.owner_online_id;
    }

    if (!Net::P2PTransportIsReady()) {
        return;
    }

    const u32 server_addr = NpMatching2::GetMmServerAddr();
    const u16 server_udp = NpMatching2::GetMmServerUdpPort();

    if (server_addr == 0 || server_udp == 0) {
        LOG_WARNING(Lib_NpSignaling,
                    "SendStunPing: ctxId={} skipped (server_addr={:#x} udp_port={})", ctx_id,
                    server_addr, sceNetNtohs(server_udp));
        return;
    }

    StunPing ping{};
    ping.cmd = 0x01;
    std::memcpy(ping.online_id, online_id.data, ORBIS_NP_ONLINEID_MAX_LENGTH);

    LOG_INFO(Lib_NpSignaling, "SendStunPing: ctxId={} online_id='{}' server={:#x}:{}", ctx_id,
             OnlineIdFromOnlineId(online_id), server_addr, sceNetNtohs(server_udp));

    Net::P2PSignalingSendTo(&ping, sizeof(ping), server_addr, server_udp);
}

s32 GetActiveConnectionIdForPeer(std::string_view online_id) {
    std::lock_guard<std::mutex> lock(g_mutex);
    for (const auto& [conn_id, ci] : g_connections) {
        if (OnlineIdEqualsString(ci.online_id, online_id) &&
            ci.status == ORBIS_NP_SIGNALING_CONN_STATUS_ACTIVE) {
            return conn_id;
        }
    }
    return 0;
}

void HandleServerNpSignalingEvent(std::string_view online_id, u32 event_type) {
    s32 ctx_id = 0;
    s32 conn_id = 0;
    OrbisNpOnlineId my_online_id{};
    OrbisNpOnlineId peer_online_id_stored{};
    {
        std::lock_guard<std::mutex> lock(g_mutex);
        for (auto& [cid, ci] : g_connections) {
            if (!OnlineIdEqualsString(ci.online_id, online_id))
                continue;
            // Skip stale INACTIVE entries from previous sessions. The game tracks
            // the connId it received from ActivateConnection; callbacks must match
            // that connId. A PENDING or ACTIVE entry is the live connection.
            if (ci.status == ORBIS_NP_SIGNALING_CONN_STATUS_INACTIVE)
                continue;
            ctx_id = ci.ctx_id;
            conn_id = cid;
            if (event_type == static_cast<u32>(ORBIS_NP_SIGNALING_EVENT_ESTABLISHED))
                ci.status = ORBIS_NP_SIGNALING_CONN_STATUS_ACTIVE;
            else
                ci.status = ORBIS_NP_SIGNALING_CONN_STATUS_INACTIVE;
            peer_online_id_stored = ci.online_id;
            break;
        }
        if (conn_id != 0) {
            auto ctx_it = g_contexts.find(ctx_id);
            if (ctx_it != g_contexts.end())
                my_online_id = ctx_it->second.owner_online_id;
        }
    }
    if (conn_id == 0) {
        LOG_WARNING(Lib_NpSignaling,
                    "HandleServerNpSignalingEvent: no connection for peer='{}' event={}", online_id,
                    event_type);
        return;
    }

    LOG_INFO(Lib_NpSignaling,
             "t={} HandleServerNpSignalingEvent: peer='{}' event={} ctxId={} connId={}", NowMs(),
             online_id, SignalingEventName(event_type), ctx_id, conn_id);

    DeliverSignalingEventForCtx(ctx_id, conn_id, event_type, 0);

    if (event_type == static_cast<u32>(ORBIS_NP_SIGNALING_EVENT_ESTABLISHED)) {
        // Report to server for mutual-activated pairing.
        const u32 server_addr = NpMatching2::GetMmServerAddr();
        const u16 server_udp = NpMatching2::GetMmServerUdpPort();
        if (server_addr != 0 && server_udp != 0) {
            SignalingEstablished pkt{};
            std::memcpy(pkt.online_id_me, my_online_id.data, ORBIS_NP_ONLINEID_MAX_LENGTH);
            std::memcpy(pkt.online_id_peer, peer_online_id_stored.data,
                        ORBIS_NP_ONLINEID_MAX_LENGTH);
            Net::P2PSignalingSendTo(&pkt, sizeof(pkt), server_addr, server_udp);
            LOG_INFO(Lib_NpSignaling, "SignalingEstablished: me='{}' peer='{}'",
                     OnlineIdFromOnlineId(my_online_id),
                     OnlineIdFromOnlineId(peer_online_id_stored));
        }
    }
}

s32 GetConnectionStatusForPeer(std::string_view online_id, s32* out_conn_id) {
    std::lock_guard<std::mutex> lock(g_mutex);
    for (const auto& [conn_id, ci] : g_connections) {
        if (OnlineIdEqualsString(ci.online_id, online_id)) {
            if (out_conn_id) {
                *out_conn_id = conn_id;
            }
            return ci.status;
        }
    }
    if (out_conn_id) {
        *out_conn_id = 0;
    }
    return ORBIS_NP_SIGNALING_CONN_STATUS_INACTIVE;
}

} // namespace Libraries::Np::NpSignaling
