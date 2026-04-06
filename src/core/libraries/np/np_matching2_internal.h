// SPDX-FileCopyrightText: Copyright 2025 shadPS4 Emulator Project
// SPDX-License-Identifier: GPL-2.0-or-later

#pragma once

#include <atomic>
#include <chrono>
#include <map>
#include <mutex>
#include <string>
#include <string_view>
#include <vector>

#include "core/libraries/kernel/threads.h"
#include "core/libraries/np/np_matching2_mm.h"
#include "core/libraries/np/np_matching2_types.h"
#include "core/libraries/np/np_signaling.h"

namespace Libraries::Np::NpMatching2 {

using OrbisNpMatching2RoomEventCallback = OrbisNpMatching2RoomCallback;
using OrbisNpMatching2LobbyEventCallback = PS4_SYSV_ABI void (*)(OrbisNpMatching2ContextId,
                                                                 OrbisNpMatching2LobbyId,
                                                                 OrbisNpMatching2Event, const void*,
                                                                 void*);
using OrbisNpMatching2LobbyMessageCallback = PS4_SYSV_ABI void (*)(OrbisNpMatching2ContextId,
                                                                   OrbisNpMatching2LobbyId, u16,
                                                                   u16, const void*, void*);
using OrbisNpMatching2RoomMessageCallback = PS4_SYSV_ABI void (*)(OrbisNpMatching2ContextId,
                                                                  OrbisNpMatching2RoomId, u16, u16,
                                                                  const void*, void*);

constexpr s32 ORBIS_NP_MATCHING2_SIGNALING_CONN_STATUS_INACTIVE =
    NpSignaling::ORBIS_NP_SIGNALING_CONN_STATUS_INACTIVE;
constexpr s32 ORBIS_NP_MATCHING2_SIGNALING_CONN_STATUS_PENDING =
    NpSignaling::ORBIS_NP_SIGNALING_CONN_STATUS_PENDING;
constexpr s32 ORBIS_NP_MATCHING2_SIGNALING_CONN_STATUS_ACTIVE =
    NpSignaling::ORBIS_NP_SIGNALING_CONN_STATUS_ACTIVE;

struct PeerInfo {
    u32 addr = 0;
    u16 port = 0;
    OrbisNpMatching2RoomMemberId member_id = 0;
    s32 conn_id = 0;
    s32 status = ORBIS_NP_MATCHING2_SIGNALING_CONN_STATUS_INACTIVE;
    bool connected_event_sent = false;
    bool bootstrap_dead_sent = false;
    std::string online_id;
};

struct PendingEvent {
    enum Type {
        CONTEXT_CB,
        REQUEST_CB,
        SIGNALING_CB,
        ROOM_EVENT_CB,
        LOBBY_EVENT_CB,
        LOBBY_MESSAGE_CB,
        ROOM_MESSAGE_CB
    };
    Type type = CONTEXT_CB;
    std::chrono::steady_clock::time_point fire_at;

    OrbisNpMatching2Event ctx_event = 0;
    OrbisNpMatching2EventCause ctx_event_cause = 0;
    s32 error_code = 0;

    OrbisNpMatching2RequestId req_id = 0;
    OrbisNpMatching2Event req_event = 0;
    OrbisNpMatching2RequestCallback request_cb = nullptr;
    void* request_cb_arg = nullptr;
    void* request_data = nullptr;

    OrbisNpMatching2RoomId room_id = 0;
    OrbisNpMatching2RoomMemberId member_id = 0;
    OrbisNpMatching2Event sig_event = 0;
    u32 conn_id = 0;

    OrbisNpMatching2Event room_event = 0;
    void* room_event_data = nullptr;

    OrbisNpMatching2LobbyId lobby_id = 0;
    OrbisNpMatching2Event lobby_event = 0;
    void* lobby_event_data = nullptr;

    OrbisNpMatching2RoomMemberId src_member_id = 0;
    OrbisNpMatching2Event msg_event = 0;
    void* message_data = nullptr;
};

struct NpMatching2Context {
    OrbisNpMatching2ContextId ctx_id = 0;
    bool started = false;
    OrbisNpMatching2ServerId server_id = 1;
    OrbisNpMatching2WorldId world_id = 0;
    OrbisNpMatching2LobbyId lobby_id = 0;
    u16 service_label = 0;
    std::string online_id;
    std::string session_id;
    OrbisNpMatching2RoomId room_id = 0;
    OrbisNpMatching2RoomMemberId my_member_id = 0;
    bool is_room_owner = false;
    u32 max_slot = 5;
    OrbisNpMatching2Flags flag_attr = 0;
};

struct NpMatching2State {
    bool initialized = false;
    u16 signaling_port = 9303;
    std::string signaling_addr = "127.0.0.1";

    OrbisNpMatching2RequestCallback default_request_callback = nullptr;
    void* default_request_callback_arg = nullptr;
    OrbisNpMatching2RequestId last_fired_req_id = 0;
    OrbisNpMatching2Event last_fired_req_event = 0;

    OrbisNpMatching2ContextCallback context_callback = nullptr;
    void* context_callback_arg = nullptr;
    OrbisNpMatching2RoomEventCallback room_event_callback = nullptr;
    void* room_event_callback_arg = nullptr;
    OrbisNpMatching2SignalingCallback signaling_callback = nullptr;
    void* signaling_callback_arg = nullptr;
    OrbisNpMatching2LobbyEventCallback lobby_event_callback = nullptr;
    void* lobby_event_callback_arg = nullptr;
    OrbisNpMatching2LobbyMessageCallback lobby_message_callback = nullptr;
    void* lobby_message_callback_arg = nullptr;
    OrbisNpMatching2RoomMessageCallback room_message_callback = nullptr;
    void* room_message_callback_arg = nullptr;

    NpMatching2Context ctx;
    std::map<OrbisNpMatching2RoomMemberId, PeerInfo> peers;

    OrbisNpMatching2RoomDataInternal* last_room_data = nullptr;
    OrbisNpMatching2RoomDataExternal* last_room_data_external = nullptr;
    OrbisNpMatching2RoomGroup* last_room_groups = nullptr;
    OrbisNpMatching2RoomBinAttrInternal* last_room_bin_attr_internal = nullptr;
    OrbisNpMatching2RoomMemberBinAttrInternal* last_room_member_bin_attr_internal = nullptr;
    OrbisNpMatching2RoomMemberDataInternal* last_member_data = nullptr;
    OrbisNpMatching2CreateJoinRoomResponse* last_create_join_response = nullptr;
    OrbisNpMatching2SearchRoomResponse* last_search_room_response = nullptr;
    OrbisNpMatching2SignalingGetPingInfoResponse* last_ping_info_response = nullptr;
    Libraries::Np::OrbisNpId* last_room_owner_np_ids = nullptr;
    void* last_request_data = nullptr;
    int last_member_count = 0;
    int last_room_data_external_count = 0;
    int last_room_group_count = 0;
    int last_room_bin_attr_internal_count = 0;
    int last_room_member_bin_attr_internal_count = 0;
    std::vector<u8*> last_request_bin_buffers;
    std::vector<void*> callback_data_pointers; // search room attr arrays, freed in FreeCallbackData

    OrbisNpMatching2World world_info_worlds[2]{};
    OrbisNpMatching2GetWorldInfoListResponse world_info_resp{};

    OrbisNpMatching2RequestId next_request_id = 1;
    std::mutex mutex;

    OrbisNpMatching2RequestCallback per_request_callback = nullptr;
    void* per_request_callback_arg = nullptr;

    std::vector<PendingEvent> pending_events;
    std::mutex event_queue_mutex;
    std::atomic<bool> dispatch_running{false};
    Kernel::PthreadT dispatch_thread = nullptr;

    void FreeCallbackData();
    void StopPollThread();
};

extern NpMatching2State g_state;

void ScheduleEvent(PendingEvent ev);
void ScheduleSignalingEventForMemberId(OrbisNpMatching2RoomMemberId member_id,
                                       OrbisNpMatching2Event sig_event);
void ScheduleSignalingEventForMemberId(OrbisNpMatching2RoomMemberId member_id,
                                       OrbisNpMatching2Event sig_event,
                                       std::chrono::steady_clock::time_point fire_at);
void ScheduleRoomEventMemberLeft(OrbisNpMatching2RoomMemberId departed_member_id, u8 reason,
                                 std::chrono::steady_clock::time_point fire_at);
void ScheduleRoomEventMemberJoined(const MemberInfo& member,
                                   std::chrono::steady_clock::time_point fire_at);
void* BuildRequestCallbackData(OrbisNpMatching2Event req_event, OrbisNpMatching2RequestId req_id,
                               OrbisNpMatching2RoomId room_id, OrbisNpMatching2ServerId server_id,
                               OrbisNpMatching2WorldId world_id, OrbisNpMatching2LobbyId lobby_id,
                               const std::vector<MemberInfo>& members,
                               OrbisNpMatching2RoomMemberId host_member_id,
                               OrbisNpMatching2RoomMemberId my_member_id, u32 max_slot = 5,
                               OrbisNpMatching2Flags flag_attr = 0);

void LogRawBuffer(const char* label, const void* ptr, size_t size, size_t max_size = 0x80);
void LogCreateJoinRoomWordView(const void* req_param, size_t size, size_t max_offset = 0x80);
void LogCreateJoinRoomDecoded(const void* req);
void LogCreateJoinRoomOutboundPayload(const OrbisNpMatching2CreateJoinRoomRequest& req,
                                      OrbisNpMatching2RequestId req_id,
                                      std::string_view request_json);

void CacheCreateJoinRoomRequest(const OrbisNpMatching2CreateJoinRoomRequest& req,
                                OrbisNpMatching2RequestId req_id);
OrbisNpMatching2SearchRoomResponse* BuildSearchRoomResponseFromBinary(
    const std::vector<u8>& replyData);

PS4_SYSV_ABI void* EventDispatchThreadFunc(void* arg);

struct AsyncGetWorldInfoArgs {
    OrbisNpMatching2ContextId ctx_id;
    OrbisNpMatching2RequestId req_id;
    OrbisNpMatching2RequestCallback callback;
    void* callback_arg;
};
PS4_SYSV_ABI void* GetWorldInfoListThreadFunc(void* arg);

extern u16 g_next_ctx_id;

} // namespace Libraries::Np::NpMatching2
