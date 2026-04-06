// SPDX-FileCopyrightText: Copyright 2025 shadPS4 Emulator Project
// SPDX-License-Identifier: GPL-2.0-or-later

// MM (matchmaking) client — binary transport connecting the emulator to the
// shadnet C++ server.  All socket I/O goes through the sceNet* family so that
// the same code runs on every host platform without conditionals.

#include <algorithm>
#include <atomic>
#include <chrono>
#include <condition_variable>
#include <cstdio>
#include <cstring>
#include <map>
#include <mutex>
#include <utility>
#include <vector>

#include "common/config.h"
#include "common/logging/log.h"
#include "core/libraries/kernel/threads.h"
#include "core/libraries/libs.h"
#include "core/libraries/network/net.h"
#include "core/libraries/network/sockets.h"
#include "core/libraries/np/np_matching2_internal.h"
#include "core/libraries/np/np_matching2_mm.h"
#include "core/libraries/np/np_signaling.h"

namespace Libraries::Np::NpMatching2 {

namespace {

// ── Shadnet packet header layout ─────────────────────────────────────────────
// Byte 0:    PacketType (u8)
// Byte 1-2:  Command/NotificationType (u16 LE)
// Byte 3-6:  Packet size including header (u32 LE)
// Byte 7-14: Packet ID (u64 LE)
// Byte 15+:  Payload

struct MmClientState {
    std::atomic<bool> running{false};
    Kernel::PthreadT thread = nullptr;
    Net::OrbisNetId sock = -1;
    std::atomic<u64> next_packet_id{1};
    std::mutex send_mutex;
    std::mutex reply_mutex;
    std::condition_variable reply_cv;
    // packetId -> (error_type, reply_payload)
    std::map<u64, std::pair<ShadnetErrorType, std::vector<u8>>> replies;
    MmNotificationHandlers handlers{};
    std::string online_id;
    std::string signaling_addr;
    u16 signaling_port = 0;

    u32 server_addr = 0;
    u16 server_udp_port = 0;
    bool authenticated = false;
};

MmClientState g_mm;

// Fixed password for auto-created shadnet accounts
static constexpr const char* SHADNET_PASSWORD = "shadps4";
static constexpr const char* SHADNET_EMAIL_SUFFIX = "@shadps4.local";

long long NowMs() {
    return std::chrono::duration_cast<std::chrono::milliseconds>(
               std::chrono::steady_clock::now().time_since_epoch())
        .count();
}

u32 IpStringToAddr(std::string_view ip) {
    u32 a, b, c, d;
    if (std::sscanf(std::string(ip).c_str(), "%u.%u.%u.%u", &a, &b, &c, &d) != 4) {
        return 0;
    }
    return (a) | (b << 8) | (c << 16) | (d << 24);
}

std::pair<std::string, int> ParseMmServerHost() {
    std::string host = Config::GetNp2ServerHost();
    if (host.size() >= 7 && host.substr(0, 7) == "http://") {
        host = host.substr(7);
    } else if (host.size() >= 8 && host.substr(0, 8) == "https://") {
        host = host.substr(8);
    }
    const auto colon = host.rfind(':');
    if (colon == std::string::npos) {
        return {host, 31313};
    }
    try {
        return {host.substr(0, colon), std::stoi(host.substr(colon + 1))};
    } catch (...) {
        return {host, 31313};
    }
}

// ── Binary packet building ───────────────────────────────────────────────────

std::vector<u8> BuildPacket(ShadnetPacketType type, u16 command, u64 packetId,
                            const std::vector<u8>& payload) {
    u32 totalSize = SHADNET_HEADER_SIZE + static_cast<u32>(payload.size());
    std::vector<u8> pkt;
    pkt.reserve(totalSize);
    pkt.push_back(static_cast<u8>(type));
    pkt.push_back(static_cast<u8>(command & 0xFF));
    pkt.push_back(static_cast<u8>((command >> 8) & 0xFF));
    for (int i = 0; i < 4; ++i)
        pkt.push_back(static_cast<u8>((totalSize >> (8 * i)) & 0xFF));
    for (int i = 0; i < 8; ++i)
        pkt.push_back(static_cast<u8>((packetId >> (8 * i)) & 0xFF));
    pkt.insert(pkt.end(), payload.begin(), payload.end());
    return pkt;
}

std::vector<u8> BuildRequest(ShadnetCommandType cmd, u64 packetId, const std::vector<u8>& payload) {
    return BuildPacket(ShadnetPacketType::Request, static_cast<u16>(cmd), packetId, payload);
}

// ── Raw socket I/O helpers ───────────────────────────────────────────────────

bool RecvExact(Net::OrbisNetId sock, u8* buf, size_t len) {
    size_t received = 0;
    while (received < len) {
        const auto n =
            Libraries::Net::sceNetRecv(sock, buf + received, static_cast<u64>(len - received), 0);
        if (n <= 0)
            return false;
        received += static_cast<size_t>(n);
    }
    return true;
}

bool SendAll(Net::OrbisNetId sock, const std::vector<u8>& data) {
    size_t sent = 0;
    while (sent < data.size()) {
        const auto n = Libraries::Net::sceNetSend(sock, data.data() + sent,
                                                  static_cast<u64>(data.size() - sent), 0);
        if (n <= 0)
            return false;
        sent += static_cast<size_t>(n);
    }
    return true;
}

// Parse a 15-byte header into its fields
struct PacketHeader {
    ShadnetPacketType type;
    u16 command;
    u32 size;
    u64 packetId;
};

bool ParseHeader(const u8* hdr, PacketHeader& out) {
    out.type = static_cast<ShadnetPacketType>(hdr[0]);
    out.command = static_cast<u16>(hdr[1]) | (static_cast<u16>(hdr[2]) << 8);
    out.size = 0;
    for (int i = 0; i < 4; ++i)
        out.size |= static_cast<u32>(hdr[3 + i]) << (8 * i);
    out.packetId = 0;
    for (int i = 0; i < 8; ++i)
        out.packetId |= static_cast<u64>(hdr[7 + i]) << (8 * i);
    return out.size >= SHADNET_HEADER_SIZE && out.size <= SHADNET_MAX_PACKET_SIZE;
}

// ── Blocking request/reply (used for Login, Create, GetRoomList, etc.) ───────

ShadnetErrorType BlockingRequest(Net::OrbisNetId sock, ShadnetCommandType cmd,
                                 const std::vector<u8>& payload, std::vector<u8>& out_reply) {
    const u64 pktId = g_mm.next_packet_id++;
    const auto pkt = BuildRequest(cmd, pktId, payload);

    {
        std::lock_guard<std::mutex> lock(g_mm.send_mutex);
        if (!SendAll(sock, pkt)) {
            LOG_ERROR(Lib_NpMatching2, "BlockingRequest: send failed cmd={}",
                      static_cast<u16>(cmd));
            return ShadnetErrorType::Malformed;
        }
    }

    // Wait for the reply to arrive (receive loop deposits it in g_mm.replies)
    std::unique_lock<std::mutex> lock(g_mm.reply_mutex);
    const bool got = g_mm.reply_cv.wait_for(lock, std::chrono::seconds(5), [&] {
        return g_mm.replies.count(pktId) > 0 || !g_mm.running;
    });
    if (!got || !g_mm.running) {
        return ShadnetErrorType::Malformed;
    }

    auto it = g_mm.replies.find(pktId);
    auto [errType, replyPayload] = std::move(it->second);
    g_mm.replies.erase(it);
    out_reply = std::move(replyPayload);
    return errType;
}

// ── Pre-auth blocking request (before receive thread, reads reply inline) ────

ShadnetErrorType PreAuthRequest(Net::OrbisNetId sock, ShadnetCommandType cmd,
                                const std::vector<u8>& payload, std::vector<u8>& out_reply) {
    const u64 pktId = g_mm.next_packet_id++;
    const auto pkt = BuildRequest(cmd, pktId, payload);

    if (!SendAll(sock, pkt)) {
        LOG_ERROR(Lib_NpMatching2, "PreAuthRequest: send failed cmd={}", static_cast<u16>(cmd));
        return ShadnetErrorType::Malformed;
    }

    // Read the reply directly (no receive thread running yet)
    u8 hdr[SHADNET_HEADER_SIZE];
    if (!RecvExact(sock, hdr, SHADNET_HEADER_SIZE)) {
        LOG_ERROR(Lib_NpMatching2, "PreAuthRequest: failed to read reply header");
        return ShadnetErrorType::Malformed;
    }

    PacketHeader h;
    if (!ParseHeader(hdr, h) || h.type != ShadnetPacketType::Reply) {
        LOG_ERROR(Lib_NpMatching2, "PreAuthRequest: invalid reply header type={}",
                  static_cast<u8>(h.type));
        return ShadnetErrorType::Malformed;
    }

    u32 payloadSize = h.size - SHADNET_HEADER_SIZE;
    std::vector<u8> replyPayload(payloadSize);
    if (payloadSize > 0 && !RecvExact(sock, replyPayload.data(), payloadSize)) {
        LOG_ERROR(Lib_NpMatching2, "PreAuthRequest: failed to read reply payload");
        return ShadnetErrorType::Malformed;
    }

    // First byte of payload is error type
    ShadnetErrorType errType = ShadnetErrorType::Malformed;
    if (!replyPayload.empty()) {
        errType = static_cast<ShadnetErrorType>(replyPayload[0]);
        out_reply.assign(replyPayload.begin() + 1, replyPayload.end());
    }
    return errType;
}

// ── Notification dispatch ────────────────────────────────────────────────────

void DispatchNotification(u16 notifType, const std::vector<u8>& payload) {
    auto type = static_cast<ShadnetNotificationType>(notifType);

    switch (type) {
    case ShadnetNotificationType::RequestEvent: {
        BinaryReader r(payload);
        const u32 ctx_id = r.GetU32LE();
        const u16 server_id = r.GetU16LE();
        const u16 world_id = r.GetU16LE();
        const u16 lobby_id = r.GetU16LE();
        const u16 req_event = r.GetU16LE();
        const u32 req_id_raw = r.GetU32LE();
        const s32 error_code = static_cast<s32>(r.GetU32LE());
        const u64 room_id = r.GetU64LE();
        const u16 member_id = r.GetU16LE();
        const u16 max_slots = r.GetU16LE();
        const u32 flags = r.GetU32LE();
        const bool is_owner = (r.GetU8() != 0);
        const bool has_response = (r.GetU8() != 0);

        if (r.error()) {
            LOG_ERROR(Lib_NpMatching2, "RequestEvent: malformed notification");
            return;
        }

        // Parse member list from response blob if present
        // Maps memberId -> vector of bin attrs for use in BuildRequestCallbackData
        std::map<u16, std::vector<MemberBinAttr>> member_bin_attrs_map;
        if (has_response) {
            // Skip RoomDataInternal section:
            // u16*5 (publicSlots, privateSlots, openPublicSlots, openPrivateSlots, maxSlot)
            // u16*3 (serverId, worldId, lobbyId) + u64 roomId + u32*2 (passwdSlotMask,
            // joinedSlotMask)
            r.skip(10 + 6 + 8 + 8); // 32 bytes fixed header

            // Room groups (variable)
            u16 groupCount = r.GetU16LE();
            for (u16 g = 0; g < groupCount && !r.error(); ++g) {
                r.skip(2 + 1 + 1 + 8 + 2 + 2); // groupId(2) + hasPasswd(1) + hasLabel(1) + label(8)
                                               // + slots(2) + groupMembers(2)
            }

            r.skip(4); // flags (u32)

            // Internal bin attrs (variable)
            u16 roomBinAttrCount = r.GetU16LE();
            for (u16 a = 0; a < roomBinAttrCount && !r.error(); ++a) {
                r.skip(8 + 2 + 2); // tick(8) + memberId(2) + attrId(2)
                u32 dataSize = r.GetU32LE();
                r.skip(dataSize); // raw data
            }

            // Now at the member list
            u16 memberCount = r.GetU16LE();
            for (u16 m = 0; m < memberCount && !r.error(); ++m) {
                r.skip(1 + 8); // hasNext(1) + joinDateTicks(8)
                r.skip(36);    // NpId: data[16] + term(1) + dummy[3] + opt[8] + reserved[8]
                u16 mid = r.GetU16LE(); // memberId
                r.skip(2 + 1 + 4);      // teamId(2) + natType(1) + memberFlags(4)

                // Room group
                u8 hasGroup = r.GetU8();
                if (hasGroup) {
                    r.skip(2 + 1 + 1 + 8 + 2 + 2); // groupId(2) + hasPasswd(1) + hasLabel(1) +
                                                   // label(8) + slots(2) + groupMembers(2)
                }

                // Member internal bin attrs
                u16 mBinAttrCount = r.GetU16LE();
                std::vector<MemberBinAttr> attrs;
                for (u16 a = 0; a < mBinAttrCount && !r.error(); ++a) {
                    r.skip(8); // tick
                    u16 attrId = r.GetU16LE();
                    u32 dataSize = r.GetU32LE();
                    const u8* dataPtr = r.GetRawPtr(dataSize);
                    MemberBinAttr mba;
                    mba.id = attrId;
                    if (dataPtr && dataSize > 0) {
                        mba.data.assign(dataPtr, dataPtr + dataSize);
                    }
                    attrs.push_back(std::move(mba));
                }
                if (!attrs.empty()) {
                    member_bin_attrs_map[mid] = std::move(attrs);
                }
            }

            if (r.error()) {
                LOG_WARNING(
                    Lib_NpMatching2,
                    "RequestEvent: response blob parse error, member bin attrs may be incomplete");
            }
            // Skip meMemberId(2) + ownerMemberId(2) - not needed
        }

        LOG_INFO(Lib_NpMatching2,
                 "t={} RequestEvent: req_event={:#x} req_id={} room={} member={} "
                 "is_owner={} max_slots={} flags={:#x}",
                 NowMs(), req_event, req_id_raw, room_id, member_id, is_owner, max_slots, flags);

        u32 req_id = req_id_raw;
        if (req_event == ORBIS_NP_MATCHING2_REQUEST_EVENT_LEAVE_ROOM) {
            if (req_id == 0)
                req_id = g_state.next_request_id++;
            auto callback = g_state.per_request_callback ? g_state.per_request_callback
                                                         : g_state.default_request_callback;
            auto callback_arg = g_state.per_request_callback ? g_state.per_request_callback_arg
                                                             : g_state.default_request_callback_arg;
            if (callback) {
                PendingEvent ev{};
                ev.type = PendingEvent::REQUEST_CB;
                ev.fire_at = std::chrono::steady_clock::now();
                ev.req_id = req_id;
                ev.req_event = req_event;
                ev.error_code = error_code;
                ev.request_cb = callback;
                ev.request_cb_arg = callback_arg;
                ev.request_data = nullptr;
                ScheduleEvent(std::move(ev));
            }
        } else {
            std::string local_online_id;
            u16 srv_id = 1;
            {
                std::lock_guard<std::mutex> lock(g_state.mutex);
                g_state.ctx.room_id = room_id;
                g_state.ctx.world_id = world_id;
                g_state.ctx.lobby_id = lobby_id;
                g_state.ctx.my_member_id = member_id;
                g_state.ctx.is_room_owner = is_owner;
                g_state.ctx.max_slot = max_slots > 0 ? max_slots : 5;
                g_state.ctx.flag_attr = flags;

                // Add self to peers
                PeerInfo self_pi{};
                self_pi.member_id = member_id;
                self_pi.addr = IpStringToAddr(g_state.signaling_addr);
                self_pi.port = Libraries::Net::sceNetHtons(g_state.signaling_port);
                self_pi.conn_id = 0;
                self_pi.status = ORBIS_NP_MATCHING2_SIGNALING_CONN_STATUS_ACTIVE;
                self_pi.online_id = g_state.ctx.online_id;
                g_state.peers[member_id] = self_pi;

                local_online_id = g_state.ctx.online_id;
                srv_id = g_state.ctx.server_id;
            }

            // Build member list from peer table (SignalingHelper populates it
            // before this RequestEvent arrives, so all peers are already known).
            std::vector<MemberInfo> all_members;
            {
                std::lock_guard<std::mutex> lock2(g_state.mutex);
                for (const auto& [mid, pi] : g_state.peers) {
                    MemberInfo mi{};
                    mi.member_id = static_cast<int>(mid);
                    mi.online_id = pi.online_id;
                    all_members.push_back(std::move(mi));
                }
            }
            // Ensure self is present
            bool found_self = false;
            for (const auto& m : all_members) {
                if (m.member_id == static_cast<int>(member_id)) {
                    found_self = true;
                    break;
                }
            }
            if (!found_self) {
                MemberInfo self_mi{};
                self_mi.member_id = static_cast<int>(member_id);
                self_mi.online_id = local_online_id;
                all_members.push_back(std::move(self_mi));
            }

            // Merge member bin attrs from response blob into MemberInfo list
            if (!member_bin_attrs_map.empty()) {
                for (auto& mi : all_members) {
                    auto it = member_bin_attrs_map.find(static_cast<u16>(mi.member_id));
                    if (it != member_bin_attrs_map.end()) {
                        mi.bin_attrs = std::move(it->second);
                    }
                }
            }

            // Sort so lowest member_id (host/owner) is first
            std::sort(
                all_members.begin(), all_members.end(),
                [](const MemberInfo& a, const MemberInfo& b) { return a.member_id < b.member_id; });

            const u16 host_member_id =
                !all_members.empty() ? static_cast<u16>(all_members.front().member_id) : 1;

            if (req_id == 0)
                req_id = g_state.next_request_id++;

            void* request_data = BuildRequestCallbackData(
                req_event, req_id, room_id, srv_id, world_id, lobby_id, all_members, host_member_id,
                member_id, g_state.ctx.max_slot, g_state.ctx.flag_attr);

            auto callback = g_state.per_request_callback ? g_state.per_request_callback
                                                         : g_state.default_request_callback;
            auto callback_arg = g_state.per_request_callback ? g_state.per_request_callback_arg
                                                             : g_state.default_request_callback_arg;

            if (callback && request_data) {
                PendingEvent ev{};
                ev.type = PendingEvent::REQUEST_CB;
                ev.fire_at = std::chrono::steady_clock::now();
                ev.req_id = req_id;
                ev.req_event = req_event;
                ev.error_code = error_code;
                ev.request_cb = callback;
                ev.request_cb_arg = callback_arg;
                ev.request_data = request_data;
                ScheduleEvent(std::move(ev));
            }
        }
        break;
    }
    case ShadnetNotificationType::MemberJoined: {
        BinaryReader r(payload);
        if (g_mm.handlers.member_joined)
            g_mm.handlers.member_joined(r);
        break;
    }
    case ShadnetNotificationType::MemberLeft: {
        BinaryReader r(payload);
        if (g_mm.handlers.member_left)
            g_mm.handlers.member_left(r);
        break;
    }
    case ShadnetNotificationType::SignalingHelper: {
        BinaryReader r(payload);
        if (g_mm.handlers.signaling_helper)
            g_mm.handlers.signaling_helper(r);
        break;
    }
    case ShadnetNotificationType::SignalingEvent: {
        BinaryReader r(payload);
        const u16 sig_event = r.GetU16LE();
        const u64 sig_room_id = r.GetU64LE();
        const u16 sig_member_id = r.GetU16LE();
        u32 conn_id = r.GetU32LE();

        if (r.error()) {
            LOG_ERROR(Lib_NpMatching2, "SignalingEvent: malformed notification");
            return;
        }

        {
            std::lock_guard<std::mutex> lock(g_state.mutex);
            auto it = g_state.peers.find(sig_member_id);
            if (it != g_state.peers.end()) {
                if (conn_id == 0 && it->second.conn_id != 0) {
                    conn_id = static_cast<u32>(it->second.conn_id);
                } else if (conn_id != 0) {
                    it->second.conn_id = static_cast<s32>(conn_id);
                }
                it->second.status = (sig_event == ORBIS_NP_MATCHING2_SIGNALING_EVENT_ESTABLISHED)
                                        ? ORBIS_NP_MATCHING2_SIGNALING_CONN_STATUS_ACTIVE
                                        : ORBIS_NP_MATCHING2_SIGNALING_CONN_STATUS_INACTIVE;
            }
        }

        LOG_INFO(Lib_NpMatching2, "t={} SignalingEvent: sig_event={:#x} room={} member={} conn={}",
                 NowMs(), sig_event, sig_room_id, sig_member_id, conn_id);

        PendingEvent ev{};
        ev.type = PendingEvent::SIGNALING_CB;
        ev.fire_at = std::chrono::steady_clock::now();
        ev.room_id = sig_room_id != 0 ? sig_room_id : g_state.ctx.room_id;
        ev.member_id = sig_member_id;
        ev.sig_event = sig_event;
        ev.conn_id = conn_id;
        ScheduleEvent(std::move(ev));
        break;
    }
    case ShadnetNotificationType::RoomDataInternalUpdated: {
        BinaryReader r(payload);
        if (g_mm.handlers.room_data_internal_updated)
            g_mm.handlers.room_data_internal_updated(r);
        break;
    }
    case ShadnetNotificationType::NpSignalingEvent: {
        BinaryReader r(payload);
        const u32 np_event = r.GetU32LE();
        const std::string peer_oid = r.GetCStr();

        if (r.error()) {
            LOG_ERROR(Lib_NpMatching2, "NpSignalingEvent: malformed notification");
            return;
        }

        LOG_INFO(Lib_NpMatching2, "t={} NpSignalingEvent: event={} peer='{}'", NowMs(), np_event,
                 peer_oid);
        NpSignaling::HandleServerNpSignalingEvent(peer_oid, np_event);
        break;
    }
    default:
        LOG_WARNING(Lib_NpMatching2, "Unknown notification type {}", notifType);
        break;
    }
}

// ── Receive thread: reads binary packets from the socket ─────────────────────

PS4_SYSV_ABI void* MmClientThreadFunc(void* /*arg*/) {
    LOG_INFO(Lib_NpMatching2, "MmClient: thread started");

    const auto [host, port] = ParseMmServerHost();
    LOG_INFO(Lib_NpMatching2, "MmClient: connecting to {}:{}", host, port);

    const Net::OrbisNetId sock =
        Libraries::Net::sceNetSocket("mm_client", Net::ORBIS_NET_AF_INET,
                                     Net::ORBIS_NET_SOCK_STREAM, Net::ORBIS_NET_IPPROTO_TCP);
    if (sock < 0) {
        LOG_ERROR(Lib_NpMatching2, "MmClient: sceNetSocket() failed rc={}", sock);
        g_mm.running = false;
        return nullptr;
    }

    u32 server_ip = 0;
    if (Libraries::Net::sceNetInetPton(Net::ORBIS_NET_AF_INET, host.c_str(), &server_ip) <= 0) {
        const Net::OrbisNetId rid = Libraries::Net::sceNetResolverCreate("mm_resolver", 0, 0);
        if (rid < 0) {
            LOG_ERROR(Lib_NpMatching2, "MmClient: sceNetResolverCreate() failed rc={}", rid);
            Libraries::Net::sceNetSocketClose(sock);
            g_mm.running = false;
            return nullptr;
        }
        const int rc = Libraries::Net::sceNetResolverStartNtoa(
            rid, host.c_str(), reinterpret_cast<Net::OrbisNetInAddr*>(&server_ip), 0, 0, 0);
        Libraries::Net::sceNetResolverDestroy(rid);
        if (rc < 0) {
            LOG_ERROR(Lib_NpMatching2, "MmClient: cannot resolve '{}' rc={}", host, rc);
            Libraries::Net::sceNetSocketClose(sock);
            g_mm.running = false;
            return nullptr;
        }
    }

    g_mm.server_addr = server_ip;
    // UDP STUN port: read from config or default to TCP port + 1
    g_mm.server_udp_port = Libraries::Net::sceNetHtons(static_cast<u16>(port + 1));

    Net::OrbisNetSockaddrIn sa{};
    sa.sin_len = sizeof(sa);
    sa.sin_family = Net::ORBIS_NET_AF_INET;
    sa.sin_port = Libraries::Net::sceNetHtons(static_cast<u16>(port));
    sa.sin_addr = server_ip;
    sa.sin_vport = 0;

    const int conn_rc = Libraries::Net::sceNetConnect(
        sock, reinterpret_cast<const Net::OrbisNetSockaddr*>(&sa), sizeof(sa));
    if (conn_rc < 0) {
        LOG_ERROR(Lib_NpMatching2, "MmClient: sceNetConnect() failed to {}:{} rc={}", host, port,
                  conn_rc);
        Libraries::Net::sceNetSocketClose(sock);
        g_mm.running = false;
        return nullptr;
    }

    LOG_INFO(Lib_NpMatching2, "MmClient: connected to {}:{}", host, port);
    g_mm.sock = sock;

    // ── Read ServerInfo packet ───────────────────────────────────────────────
    {
        u8 hdr[SHADNET_HEADER_SIZE];
        if (!RecvExact(sock, hdr, SHADNET_HEADER_SIZE)) {
            LOG_ERROR(Lib_NpMatching2, "MmClient: failed to read ServerInfo header");
            Libraries::Net::sceNetSocketClose(sock);
            g_mm.sock = -1;
            g_mm.running = false;
            return nullptr;
        }
        PacketHeader h;
        if (!ParseHeader(hdr, h) || h.type != ShadnetPacketType::ServerInfo) {
            LOG_ERROR(Lib_NpMatching2, "MmClient: expected ServerInfo, got type={}",
                      static_cast<u8>(h.type));
            Libraries::Net::sceNetSocketClose(sock);
            g_mm.sock = -1;
            g_mm.running = false;
            return nullptr;
        }
        // Read and discard ServerInfo payload (protocol version u32)
        u32 payloadSize = h.size - SHADNET_HEADER_SIZE;
        if (payloadSize > 0) {
            std::vector<u8> siBuf(payloadSize);
            RecvExact(sock, siBuf.data(), payloadSize);
        }
        LOG_INFO(Lib_NpMatching2, "MmClient: received ServerInfo");
    }

    // ── Create account (if not exists) + Login ───────────────────────────────
    {
        // Try to create an account first. If it already exists, that's fine.
        BinaryWriter createPayload;
        createPayload.AppendCStr(g_mm.online_id);   // npid
        createPayload.AppendCStr(SHADNET_PASSWORD); // password
        createPayload.AppendCStr(g_mm.online_id);   // onlineName
        createPayload.AppendCStr("");               // avatarUrl (empty = default)
        createPayload.AppendCStr(g_mm.online_id + SHADNET_EMAIL_SUFFIX); // email

        std::vector<u8> createReply;
        auto createErr =
            PreAuthRequest(sock, ShadnetCommandType::Create, createPayload.data(), createReply);
        if (createErr == ShadnetErrorType::NoError) {
            LOG_INFO(Lib_NpMatching2, "MmClient: account created for '{}'", g_mm.online_id);
        } else if (createErr == ShadnetErrorType::CreationExistingUsername) {
            LOG_INFO(Lib_NpMatching2, "MmClient: account already exists for '{}'", g_mm.online_id);
        } else {
            LOG_WARNING(Lib_NpMatching2, "MmClient: account creation returned error {}",
                        static_cast<u8>(createErr));
        }

        // Now login
        BinaryWriter loginPayload;
        loginPayload.AppendCStr(g_mm.online_id);   // npid
        loginPayload.AppendCStr(SHADNET_PASSWORD); // password
        loginPayload.AppendCStr("");               // token (empty)

        std::vector<u8> loginReply;
        auto loginErr =
            PreAuthRequest(sock, ShadnetCommandType::Login, loginPayload.data(), loginReply);
        if (loginErr != ShadnetErrorType::NoError) {
            LOG_ERROR(Lib_NpMatching2, "MmClient: login failed for '{}' error={}", g_mm.online_id,
                      static_cast<u8>(loginErr));
            Libraries::Net::sceNetSocketClose(sock);
            g_mm.sock = -1;
            g_mm.running = false;
            return nullptr;
        }

        g_mm.authenticated = true;
        LOG_INFO(Lib_NpMatching2, "MmClient: authenticated as '{}'", g_mm.online_id);
        // Flush any callbacks registered before the connection was established.
        SendRegisterHandlers();
    }

    // ── Main receive loop (binary packets) ───────────────────────────────────

    while (g_mm.running) {
        u8 hdr[SHADNET_HEADER_SIZE];
        if (!RecvExact(sock, hdr, SHADNET_HEADER_SIZE)) {
            if (g_mm.running) {
                LOG_INFO(Lib_NpMatching2, "MmClient: connection closed by server");
            }
            break;
        }

        PacketHeader h;
        if (!ParseHeader(hdr, h)) {
            LOG_WARNING(Lib_NpMatching2, "MmClient: invalid packet header — disconnecting");
            break;
        }

        u32 payloadSize = h.size - SHADNET_HEADER_SIZE;
        std::vector<u8> payload(payloadSize);
        if (payloadSize > 0 && !RecvExact(sock, payload.data(), payloadSize)) {
            if (g_mm.running) {
                LOG_INFO(Lib_NpMatching2, "MmClient: connection closed reading payload");
            }
            break;
        }

        if (h.type == ShadnetPacketType::Reply) {
            // Extract error byte, store remaining payload
            ShadnetErrorType errType = ShadnetErrorType::Malformed;
            std::vector<u8> replyData;
            if (!payload.empty()) {
                errType = static_cast<ShadnetErrorType>(payload[0]);
                replyData.assign(payload.begin() + 1, payload.end());
            }
            {
                std::lock_guard<std::mutex> lock(g_mm.reply_mutex);
                g_mm.replies[h.packetId] = {errType, std::move(replyData)};
            }
            g_mm.reply_cv.notify_all();
        } else if (h.type == ShadnetPacketType::Notification) {
            DispatchNotification(h.command, payload);
        } else {
            LOG_WARNING(Lib_NpMatching2, "MmClient: unexpected packet type {}",
                        static_cast<u8>(h.type));
        }
    }

    g_mm.sock = -1;
    g_mm.authenticated = false;
    Libraries::Net::sceNetSocketClose(sock);
    g_mm.reply_cv.notify_all();
    LOG_INFO(Lib_NpMatching2, "MmClient: thread stopped");
    return nullptr;
}

} // namespace

// ── Public API ───────────────────────────────────────────────────────────────

void ConfigureMmNotificationHandlers(MmNotificationHandlers handlers) {
    g_mm.handlers = std::move(handlers);
}

bool StartMmClient(std::string_view online_id, std::string_view signaling_addr,
                   u16 signaling_port) {
    if (g_mm.running)
        return true;

    g_mm.online_id = std::string(online_id);
    g_mm.signaling_addr = std::string(signaling_addr);
    g_mm.signaling_port = signaling_port;
    g_mm.running = true;
    g_mm.authenticated = false;

    const int ret =
        Kernel::posix_pthread_create(&g_mm.thread, nullptr, MmClientThreadFunc, nullptr);
    if (ret != 0) {
        LOG_ERROR(Lib_NpMatching2, "Failed to create MmClient thread: {}", ret);
        g_mm.running = false;
        g_mm.thread = nullptr;
        return false;
    }

    LOG_INFO(Lib_NpMatching2, "MmClient thread created");
    return true;
}

void StopMmClient() {
    g_mm.running = false;
    if (g_mm.sock >= 0) {
        Libraries::Net::sceNetSocketClose(g_mm.sock);
        g_mm.sock = -1;
    }
    g_mm.reply_cv.notify_all();
    if (g_mm.thread != nullptr) {
        Kernel::posix_pthread_join(g_mm.thread, nullptr);
        g_mm.thread = nullptr;
    }
    {
        std::lock_guard<std::mutex> lock(g_mm.reply_mutex);
        g_mm.replies.clear();
    }
    g_mm.authenticated = false;
}

bool IsMmClientRunning() {
    return g_mm.running && g_mm.authenticated;
}

ShadnetErrorType MmRequest(ShadnetCommandType cmd, const BinaryWriter& payload,
                           std::vector<u8>& out_reply) {
    if (g_mm.sock < 0 || !g_mm.authenticated) {
        LOG_ERROR(Lib_NpMatching2, "MmRequest({}): not connected", static_cast<u16>(cmd));
        return ShadnetErrorType::Malformed;
    }
    return BlockingRequest(g_mm.sock, cmd, payload.data(), out_reply);
}

void MmSendFireAndForget(ShadnetCommandType cmd, const BinaryWriter& payload) {
    if (g_mm.sock < 0 || !g_mm.authenticated) {
        LOG_DEBUG(Lib_NpMatching2, "MmSendFireAndForget({}): not connected, dropping",
                  static_cast<u16>(cmd));
        return;
    }
    const u64 pktId = g_mm.next_packet_id++;
    const auto pkt = BuildRequest(cmd, pktId, payload.data());
    std::lock_guard<std::mutex> lock(g_mm.send_mutex);
    SendAll(g_mm.sock, pkt);
    LOG_DEBUG(Lib_NpMatching2, "MmSendFireAndForget: cmd={} pktId={}", static_cast<u16>(cmd),
              pktId);
}

void SendRegisterHandlers() {
    if (!IsMmClientRunning() || g_mm.sock < 0)
        return;

    const bool has_context = g_state.context_callback != nullptr;
    const bool has_request =
        g_state.default_request_callback != nullptr || g_state.per_request_callback != nullptr;
    const bool has_room_event = g_state.room_event_callback != nullptr;
    const bool has_signaling = g_state.signaling_callback != nullptr;
    const bool has_lobby_event = g_state.lobby_event_callback != nullptr;
    const bool has_room_message = g_state.room_message_callback != nullptr;
    const bool has_lobby_message = g_state.lobby_message_callback != nullptr;

    BinaryWriter w;
    w.AppendCStr(g_mm.signaling_addr);  // addr
    w.AppendU16LE(g_mm.signaling_port); // port
    w.AppendU32LE(g_state.ctx.ctx_id);  // ctxId
    w.AppendU32LE(0);                   // serviceLabel
    w.AppendU8(7);                      // handlerCount
    // Handler 0: context
    w.AppendU8(has_context ? 1 : 0);
    w.AppendU64LE(0);
    w.AppendU64LE(0);
    // Handler 1: request
    w.AppendU8(has_request ? 1 : 0);
    w.AppendU64LE(0);
    w.AppendU64LE(0);
    // Handler 2: signaling
    w.AppendU8(has_signaling ? 1 : 0);
    w.AppendU64LE(0);
    w.AppendU64LE(0);
    // Handler 3: room_event
    w.AppendU8(has_room_event ? 1 : 0);
    w.AppendU64LE(0);
    w.AppendU64LE(0);
    // Handler 4: lobby_event
    w.AppendU8(has_lobby_event ? 1 : 0);
    w.AppendU64LE(0);
    w.AppendU64LE(0);
    // Handler 5: room_message
    w.AppendU8(has_room_message ? 1 : 0);
    w.AppendU64LE(0);
    w.AppendU64LE(0);
    // Handler 6: lobby_message
    w.AppendU8(has_lobby_message ? 1 : 0);
    w.AppendU64LE(0);
    w.AppendU64LE(0);

    MmSendFireAndForget(ShadnetCommandType::RegisterHandlers, w);
    LOG_INFO(Lib_NpMatching2,
             "SendRegisterHandlers: context={} request={} room_event={} signaling={} "
             "lobby_event={} room_message={} lobby_message={}",
             has_context, has_request, has_room_event, has_signaling, has_lobby_event,
             has_room_message, has_lobby_message);
}

// ── Notification handlers ────────────────────────────────────────────────────

void OnMmMemberJoined(BinaryReader& reader) {
    const u64 room_id = reader.GetU64LE();
    const u16 member_id = reader.GetU16LE();
    const std::string online_id = reader.GetCStr();
    const std::string addr = reader.GetCStr();
    const u16 port = reader.GetU16LE();

    // Parse member bin attrs (appended by server after the base fields)
    std::vector<MemberBinAttr> member_bin_attrs;
    if (!reader.error() && reader.remaining() >= 2) {
        const u16 bin_attr_count = reader.GetU16LE();
        for (u16 i = 0; i < bin_attr_count && !reader.error(); ++i) {
            MemberBinAttr mba;
            mba.id = reader.GetU16LE();
            const u32 data_size = reader.GetU32LE();
            if (!reader.error() && data_size > 0) {
                const u8* src = reader.GetRawPtr(data_size);
                if (src)
                    mba.data.assign(src, src + data_size);
            }
            member_bin_attrs.push_back(std::move(mba));
        }
    }

    if (reader.error() || member_id == 0)
        return;

    LOG_INFO(Lib_NpMatching2,
             "t={} OnMmMemberJoined: member={} online_id={} addr={}:{} binAttrs={}", NowMs(),
             member_id, online_id, addr, port, member_bin_attrs.size());

    if (member_id != g_state.ctx.my_member_id) {
        PeerInfo pi;
        pi.member_id = member_id;
        pi.addr = IpStringToAddr(addr);
        pi.port = Libraries::Net::sceNetHtons(port);
        pi.conn_id = 0;
        pi.status = ORBIS_NP_MATCHING2_SIGNALING_CONN_STATUS_ACTIVE;
        pi.online_id = online_id;
        g_state.peers[member_id] = pi;

        const s32 sig_conn_id = NpSignaling::GetActiveConnectionIdForPeer(online_id);
        if (sig_conn_id != 0) {
            pi.conn_id = sig_conn_id;
            g_state.peers[member_id] = pi;
        }
    }

    MemberInfo mi{};
    mi.member_id = static_cast<int>(member_id);
    mi.online_id = online_id;
    mi.addr = addr;
    mi.port = static_cast<int>(port);
    mi.bin_attrs = std::move(member_bin_attrs);
    ScheduleRoomEventMemberJoined(mi, std::chrono::steady_clock::now());
}

void OnMmMemberLeft(BinaryReader& reader) {
    const u64 room_id = reader.GetU64LE();
    const u16 member_id = reader.GetU16LE();
    const std::string online_id = reader.GetCStr();

    if (reader.error() || member_id == 0 || member_id == g_state.ctx.my_member_id)
        return;
    if (g_state.peers.count(member_id) == 0) {
        LOG_WARNING(Lib_NpMatching2, "OnMmMemberLeft: unknown member {}", member_id);
        return;
    }

    LOG_INFO(Lib_NpMatching2, "t={} OnMmMemberLeft: member={}", NowMs(), member_id);

    auto now = std::chrono::steady_clock::now();
    {
        const auto& pi = g_state.peers[member_id];
        PendingEvent sig_ev{};
        sig_ev.type = PendingEvent::SIGNALING_CB;
        sig_ev.fire_at = now + std::chrono::milliseconds(100);
        sig_ev.room_id = g_state.ctx.room_id;
        sig_ev.member_id = member_id;
        sig_ev.sig_event = ORBIS_NP_MATCHING2_SIGNALING_EVENT_DEAD;
        sig_ev.conn_id = pi.conn_id > 0 ? static_cast<u32>(pi.conn_id) : 0;
        ScheduleEvent(std::move(sig_ev));
    }

    ScheduleRoomEventMemberLeft(member_id, ORBIS_NP_MATCHING2_EVENT_CAUSE_MEMBER_DISAPPEARED, now);
    g_state.peers.erase(member_id);
}

void OnMmRoomDestroyed(BinaryReader& /*reader*/) {
    LOG_INFO(Lib_NpMatching2, "OnMmRoomDestroyed: firing events for all peers");
    auto now = std::chrono::steady_clock::now();
    for (const auto& [mid, pi] : g_state.peers) {
        if (mid == g_state.ctx.my_member_id)
            continue;
        ScheduleRoomEventMemberLeft(mid, ORBIS_NP_MATCHING2_EVENT_CAUSE_MEMBER_DISAPPEARED, now);
        PendingEvent sig_ev{};
        sig_ev.type = PendingEvent::SIGNALING_CB;
        sig_ev.fire_at = now + std::chrono::milliseconds(100);
        sig_ev.room_id = g_state.ctx.room_id;
        sig_ev.member_id = mid;
        sig_ev.sig_event = ORBIS_NP_MATCHING2_SIGNALING_EVENT_DEAD;
        sig_ev.conn_id = pi.conn_id > 0 ? static_cast<u32>(pi.conn_id) : 0;
        ScheduleEvent(std::move(sig_ev));
    }
}

void OnMmSignalingHelper(BinaryReader& reader) {
    const std::string online_id = reader.GetCStr();
    const u16 member_id = reader.GetU16LE();
    const std::string addr = reader.GetCStr();
    const u16 port = reader.GetU16LE();

    if (reader.error() || member_id == 0 || addr.empty())
        return;

    LOG_INFO(Lib_NpMatching2, "t={} OnMmSignalingHelper: member={} online_id={} addr={}:{}",
             NowMs(), member_id, online_id, addr, port);

    const u32 ip = IpStringToAddr(addr);
    const u16 port_nbo = Libraries::Net::sceNetHtons(port);

    if (g_state.peers.count(member_id) == 0) {
        PeerInfo pi;
        pi.member_id = member_id;
        pi.addr = ip;
        pi.port = port_nbo;
        pi.status = ORBIS_NP_MATCHING2_SIGNALING_CONN_STATUS_INACTIVE;
        pi.online_id = online_id;
        g_state.peers[member_id] = pi;
    } else {
        g_state.peers[member_id].addr = ip;
        g_state.peers[member_id].port = port_nbo;
    }
}

// Static storage for the 0x1106 callback data (lives until next update).
static OrbisNpMatching2RoomDataInternal s_room_data_update{};
static OrbisNpMatching2RoomDataInternalUpdateInfo s_room_update_info{};
static std::vector<OrbisNpMatching2RoomBinAttrInternal> s_room_update_bin_attrs;
static std::vector<std::vector<u8>> s_room_update_bin_buffers;

void OnMmRoomDataInternalUpdated(BinaryReader& reader) {
    const u64 room_id = reader.GetU64LE();
    const u32 new_flags = reader.GetU32LE();
    const u16 bin_attr_count = reader.GetU16LE();

    if (reader.error()) {
        LOG_ERROR(Lib_NpMatching2, "OnMmRoomDataInternalUpdated: malformed");
        return;
    }

    LOG_INFO(Lib_NpMatching2,
             "t={} OnMmRoomDataInternalUpdated: room={} flags={:#x} binAttrCount={}", NowMs(),
             room_id, new_flags, bin_attr_count);

    // Parse bin attr data and build the callback structs the game expects.
    s_room_update_bin_attrs.clear();
    s_room_update_bin_buffers.clear();
    s_room_update_bin_attrs.resize(bin_attr_count);
    s_room_update_bin_buffers.resize(bin_attr_count);

    for (u16 i = 0; i < bin_attr_count; ++i) {
        const u16 attr_id = reader.GetU16LE();
        const u32 data_size = reader.GetU32LE();
        s_room_update_bin_buffers[i].resize(data_size);
        if (data_size > 0) {
            const u8* src = reader.GetRawPtr(data_size);
            if (src)
                std::memcpy(s_room_update_bin_buffers[i].data(), src, data_size);
        }
        auto& ba = s_room_update_bin_attrs[i];
        std::memset(&ba, 0, sizeof(ba));
        ba.binAttr.id = attr_id;
        ba.binAttr.data = s_room_update_bin_buffers[i].data();
        ba.binAttr.dataSize = data_size;
    }

    if (reader.error()) {
        LOG_ERROR(Lib_NpMatching2, "OnMmRoomDataInternalUpdated: malformed bin attrs");
        return;
    }

    // Populate OrbisNpMatching2RoomDataInternal with current context + updated bin attrs.
    std::memset(&s_room_data_update, 0, sizeof(s_room_data_update));
    {
        std::lock_guard<std::mutex> lock(g_state.mutex);
        s_room_data_update.maxSlot = static_cast<u16>(g_state.ctx.max_slot);
        s_room_data_update.serverId = g_state.ctx.server_id;
        s_room_data_update.worldId = g_state.ctx.world_id;
        s_room_data_update.lobbyId = g_state.ctx.lobby_id;
        s_room_data_update.roomId = room_id;
        s_room_data_update.flags = new_flags;
    }
    s_room_data_update.roomBinAttrInternal =
        bin_attr_count > 0 ? s_room_update_bin_attrs.data() : nullptr;
    s_room_data_update.roomBinAttrInternalNum = bin_attr_count;

    std::memset(&s_room_update_info, 0, sizeof(s_room_update_info));
    s_room_update_info.roomDataInternal = &s_room_data_update;
    s_room_update_info.eventCause = 0;

    PendingEvent ev{};
    ev.type = PendingEvent::ROOM_EVENT_CB;
    ev.fire_at = std::chrono::steady_clock::now();
    ev.room_id = room_id;
    ev.room_event = ORBIS_NP_MATCHING2_ROOM_EVENT_UPDATED_ROOM_DATA_INTERNAL;
    ev.room_event_data = &s_room_update_info;
    ScheduleEvent(std::move(ev));
}

u32 GetMmServerAddr() {
    return g_mm.server_addr;
}

u16 GetMmServerUdpPort() {
    return g_mm.server_udp_port;
}

bool RequestSignalingInfos(std::string_view target_online_id, u32* out_addr, u16* out_port) {
    if (!out_addr || !out_port)
        return false;

    BinaryWriter w;
    w.AppendCStr(target_online_id);

    std::vector<u8> reply;
    auto err = MmRequest(ShadnetCommandType::RequestSignalingInfos, w, reply);
    if (err != ShadnetErrorType::NoError) {
        LOG_WARNING(Lib_NpMatching2, "RequestSignalingInfos: error {} for target='{}'",
                    static_cast<u8>(err), target_online_id);
        return false;
    }

    BinaryReader r(reply);
    const std::string npid = r.GetCStr();
    const std::string addr_str = r.GetCStr();
    const u16 port_host = r.GetU16LE();
    const u16 target_member_id = r.GetU16LE();

    if (r.error() || addr_str.empty() || port_host == 0) {
        LOG_WARNING(Lib_NpMatching2, "RequestSignalingInfos: malformed reply for target='{}'",
                    target_online_id);
        return false;
    }

    u32 addr_nbo = 0;
    if (Libraries::Net::sceNetInetPton(Net::ORBIS_NET_AF_INET, addr_str.c_str(), &addr_nbo) <= 0) {
        LOG_WARNING(Lib_NpMatching2,
                    "RequestSignalingInfos: cannot parse addr '{}' for target='{}'", addr_str,
                    target_online_id);
        return false;
    }

    *out_addr = addr_nbo;
    *out_port = Libraries::Net::sceNetHtons(port_host);

    LOG_INFO(Lib_NpMatching2, "RequestSignalingInfos: target='{}' resolved to {}:{}",
             target_online_id, addr_str, port_host);
    return true;
}

bool SendActivationConfirm(std::string_view me_id, std::string_view initiator_ip, u32 ctx_tag) {
    BinaryWriter w;
    w.AppendCStr(me_id);
    w.AppendCStr(initiator_ip);
    w.AppendU32LE(ctx_tag);

    std::vector<u8> reply;
    auto err = MmRequest(ShadnetCommandType::ActivationConfirm, w, reply);
    return err == ShadnetErrorType::NoError;
}

} // namespace Libraries::Np::NpMatching2
