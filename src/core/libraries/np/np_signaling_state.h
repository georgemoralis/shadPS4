// SPDX-FileCopyrightText: Copyright 2025 shadPS4 Emulator Project
// SPDX-License-Identifier: GPL-2.0-or-later

#pragma once

#include <atomic>
#include <chrono>
#include <condition_variable>
#include <cstring>
#include <map>
#include <mutex>
#include <string>
#include <unordered_map>
#include <vector>

#include "common/types.h"
#include "core/libraries/kernel/threads.h"
#include "core/libraries/np/np_signaling.h"
#include "core/libraries/np/np_types.h"

namespace Libraries::Np::NpSignaling {

constexpr s64 kActivateCooldownUs = 6'000'000;
constexpr s64 kActivateBudgetMaxUs = 600'000'000;
constexpr u32 kSigRetryMs = 500;
constexpr u32 kSigPingMs = 5'000;

// Activation packet emitted by sceNpSignalingActivateConnection.
// The decomp shows a 32-byte send buffer with the first three u32 fields set.
#pragma pack(push, 1)
struct ActivatePacket {
    u32 type = 5;
    u32 conn_id = 0;
    u32 ctx_tag = 0;
    u8 reserved[20]{};
};
static_assert(sizeof(ActivatePacket) == 32, "ActivatePacket must be exactly 32 bytes");
#pragma pack(pop)

// STUN-equivalent ping sent from our signaling socket to the server's UDP port.
// The server observes the external src addr and echoes it back as StunEcho.
#pragma pack(push, 1)
struct StunPing {
    u8 cmd = 0x01;      // Always 0x01 -- identifies as STUN ping
    u8 online_id[16]{}; // Local online_id, null-padded to 16 bytes
    u32 local_ip = 0;   // Local IP in network byte order (informational)
};
static_assert(sizeof(StunPing) == 21, "StunPing must be exactly 21 bytes");
#pragma pack(pop)

// Echo response sent by the server.
#pragma pack(push, 1)
struct StunEcho {
    u32 ext_ip = 0;   // External IP in network byte order
    u16 ext_port = 0; // External UDP port in network byte order
};
static_assert(sizeof(StunEcho) == 6, "StunEcho must be exactly 6 bytes");
#pragma pack(pop)

// Signaling state report: client -> server after ESTABLISHED fires.
// Server uses this to detect mutual activation and send MutualActivated to both.
#pragma pack(push, 1)
struct SignalingEstablished {
    u8 cmd = 0x02;
    u8 online_id_me[16]{};
    u8 online_id_peer[16]{};
};
static_assert(sizeof(SignalingEstablished) == 33);
#pragma pack(pop)

// Mutual activation notification: server -> client when both sides have reported.
#pragma pack(push, 1)
struct MutualActivated {
    u8 cmd = 0x03;
    u8 online_id_peer[16]{};
};
static_assert(sizeof(MutualActivated) == 17);
#pragma pack(pop)

// Client -> server (UDP): Peer A notifies the server that it is activating a connection to Peer B.
#pragma pack(push, 1)
struct ActivationIntent {
    u8 cmd = 0x04;
    u8 online_id_me[16]{};   // Peer A's online_id
    u8 online_id_peer[16]{}; // Peer B's online_id
    u32 ctx_tag = 0;         // Peer A's context tag (host byte order, = bound_port)
};
static_assert(sizeof(ActivationIntent) == 37);
#pragma pack(pop)

struct ConnectionInfo {
    s32 conn_id = 0;
    s32 ctx_id = 0;

    // Peer's externally-routable UDP endpoint (network byte order).
    u32 addr = 0;
    u16 port = 0;

    s32 status = ORBIS_NP_SIGNALING_CONN_STATUS_INACTIVE;

    OrbisNpId npid{};
    OrbisNpOnlineId online_id{};
    // Last-measured round-trip time in microseconds.
    u32 rtt_us = 0;

    // Bandwidth placeholder (games read this; keep it plausible).
    u32 bandwidth = 10'000'000;

    // Packet loss placeholder.
    u32 packet_loss = 0;
};

struct NpSignalingContext {
    OrbisNpSignalingHandler callback{nullptr};
    void* callback_arg{nullptr};
    bool active{false};

    OrbisNpId owner_npid{};
    OrbisNpOnlineId owner_online_id{};

    // The port used for signaling (host byte order).
    // This is the P2P shared transport port, not a socket we own.
    u16 bound_port{0};

    // External endpoint as returned by the STUN echo.
    // Written by the receive thread, read by ActivateConnection.
    std::atomic<u32> ext_addr{0};
    std::atomic<u16> ext_port{0};
    std::mutex stun_mutex{};
    std::condition_variable stun_cv{};

    // FUN_01003900 uses a 6-second token bucket per signaling context.
    s64 activate_budget_us{0};
    s64 activate_last_update_us{0};
};

// Key for the (ctx_id, npid) -> conn_id lookup table.
struct CtxNpIdKey {
    s32 ctx_id = 0;
    OrbisNpId npid{};
};

struct CtxNpIdKeyHash {
    size_t operator()(const CtxNpIdKey& key) const noexcept {
        size_t hash = static_cast<size_t>(key.ctx_id);
        const auto* bytes = reinterpret_cast<const u8*>(&key.npid);
        for (size_t i = 0; i < sizeof(key.npid); ++i) {
            hash ^= static_cast<size_t>(bytes[i]) + 0x9e3779b9 + (hash << 6) + (hash >> 2);
        }
        return hash;
    }
};

inline bool operator==(const CtxNpIdKey& lhs, const CtxNpIdKey& rhs) {
    return lhs.ctx_id == rhs.ctx_id && std::memcmp(&lhs.npid, &rhs.npid, sizeof(lhs.npid)) == 0;
}

struct QueuedDispatch {
    std::chrono::steady_clock::time_point fire_at{};
    u32 delay_ms = 0;
    s32 ctx_id = 0;
    s32 conn_id = 0;
    u32 event_type = 0;
    u32 error_code = 0;
    OrbisNpSignalingHandler callback = nullptr;
    void* callback_arg = nullptr;
};

// Shared globals (defined in np_signaling_state.cpp).
extern std::unordered_map<s32, NpSignalingContext> g_contexts;
extern std::unordered_map<s32, ConnectionInfo> g_connections;
extern std::unordered_map<CtxNpIdKey, s32, CtxNpIdKeyHash> g_npid_to_conn;
extern bool g_initialized;
extern std::mutex g_mutex;
extern std::multimap<std::chrono::steady_clock::time_point, QueuedDispatch> g_dispatch_queue;
extern std::mutex g_dispatch_mutex;
extern std::condition_variable g_dispatch_cv;
extern bool g_dispatch_stop;

long long NowMs();
s64 NowUs();
std::string OnlineIdFromOnlineId(const OrbisNpOnlineId& online_id_in);
std::string OnlineIdFromNpId(const OrbisNpId& np_id);
OrbisNpId NpIdFromOnlineId(const OrbisNpOnlineId& online_id);
bool OnlineIdEqualsString(const OrbisNpOnlineId& online_id, std::string_view value);
bool IsValidNpId(const OrbisNpId& np_id);
s32 NormalizeNpId(const void* np_id, OrbisNpId* out_npid, OrbisNpOnlineId* out_online_id = nullptr);
CtxNpIdKey MakeCtxNpIdKey(s32 ctx_id, const OrbisNpId& npid);

bool IsContextValidLocked(s32 ctx_id);
s32 AllocateContextIdLocked();
s32 AllocateConnectionIdLocked();
void RemoveConnectionLocked(s32 conn_id);
void RemoveContextConnectionsLocked(s32 ctx_id);

const char* SignalingEventName(u32 event_type);
void DeliverSignalingEventForCtx(s32 ctx_id, s32 conn_id, u32 event_type, u32 delay_ms);
bool ConsumeActivationBudgetLocked(NpSignalingContext& ctx); // caller holds g_mutex
void UpdateConnStatus(s32 ctx_id, s32 conn_id, s32 new_status);
s32 SendActivatePacket(u32 peer_addr, u16 peer_port, u32 conn_id, u32 ctx_tag);
void SendStunPing(s32 ctx_id);

} // namespace Libraries::Np::NpSignaling
