// SPDX-FileCopyrightText: Copyright 2025 shadPS4 Emulator Project
// SPDX-License-Identifier: GPL-2.0-or-later

#include <algorithm>
#include <atomic>
#include <cctype>
#include <chrono>
#include <cinttypes>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <map>
#include <mutex>
#include <set>
#include <string>
#include <string_view>
#include <thread>
#include <vector>

#include "common/logging/log.h"
#include "core/libraries/error_codes.h"
#include "core/libraries/kernel/threads.h"
#include "core/libraries/network/net.h"
#include "core/libraries/network/sockets.h"
#include "core/libraries/np/np_matching2_internal.h"
#include "core/tls.h"

namespace {

long long NowMs() {
    return std::chrono::duration_cast<std::chrono::milliseconds>(
               std::chrono::steady_clock::now().time_since_epoch())
        .count();
}

std::string SafeStringPreview(const char* ptr, size_t max_len = 64) {
    if (!ptr) {
        return {};
    }

    std::string out;
    out.reserve(max_len);
    for (size_t i = 0; i < max_len; ++i) {
        const char c = ptr[i];
        if (c == '\0') {
            break;
        }
        if (std::isprint(static_cast<unsigned char>(c)) == 0) {
            out += '.';
            continue;
        }
        out += c;
    }
    return out;
}

std::vector<std::string> JsonGetObjectArray(const std::string& json, const std::string& key) {
    std::vector<std::string> result;
    const std::string search = "\"" + key + "\"";
    auto pos = json.find(search);
    if (pos == std::string::npos) {
        return result;
    }

    pos = json.find('[', pos + search.size());
    if (pos == std::string::npos) {
        return result;
    }

    bool in_string = false;
    bool escaped = false;
    int brace_depth = 0;
    size_t obj_start = std::string::npos;

    for (size_t i = pos + 1; i < json.size(); ++i) {
        const char c = json[i];
        if (escaped) {
            escaped = false;
            continue;
        }
        if (c == '\\') {
            escaped = true;
            continue;
        }
        if (c == '"') {
            in_string = !in_string;
            continue;
        }
        if (in_string) {
            continue;
        }
        if (c == '{') {
            if (brace_depth == 0) {
                obj_start = i;
            }
            ++brace_depth;
            continue;
        }
        if (c == '}') {
            if (brace_depth == 0) {
                continue;
            }
            --brace_depth;
            if (brace_depth == 0 && obj_start != std::string::npos) {
                result.push_back(json.substr(obj_start, i - obj_start + 1));
                obj_start = std::string::npos;
            }
            continue;
        }
        if (c == ']' && brace_depth == 0) {
            break;
        }
    }

    return result;
}

void LogCreateJoinRoomSignalingOptions(
    const Libraries::Np::NpMatching2::OrbisNpMatching2SignalingParam* opt) {
    if (!opt) {
        return;
    }

    LOG_INFO(Lib_NpMatching2,
             "CreateJoinRoom signalingOptions: ptr={:p} type={} flag={:#x} mainMember={} "
             "pad={:02x} {:02x} {:02x} {:02x}",
             static_cast<const void*>(opt), opt->type, opt->flag, opt->mainMember,
             static_cast<unsigned>(opt->pad[0]), static_cast<unsigned>(opt->pad[1]),
             static_cast<unsigned>(opt->pad[2]), static_cast<unsigned>(opt->pad[3]));
    Libraries::Np::NpMatching2::LogRawBuffer("CreateJoinRoom signalingOptions raw", opt,
                                             sizeof(*opt), sizeof(*opt));
}

void LogCreateJoinRoomIntAttrArray(std::string_view label,
                                   const Libraries::Np::NpMatching2::OrbisNpMatching2IntAttr* attrs,
                                   u64 count) {
    const u64 limit = std::min<u64>(count, 8);
    for (u64 i = 0; attrs && i < limit; ++i) {
        LOG_INFO(Lib_NpMatching2, "{}[{}]: id={:#x} attr={:#x}", label, i, attrs[i].id,
                 attrs[i].attr);
    }
}

void LogCreateJoinRoomBinAttrArray(std::string_view label,
                                   const Libraries::Np::NpMatching2::OrbisNpMatching2BinAttr* attrs,
                                   u64 count) {
    const u64 limit = std::min<u64>(count, 8);
    for (u64 i = 0; attrs && i < limit; ++i) {
        const auto& attr = attrs[i];
        LOG_INFO(Lib_NpMatching2, "{}[{}]: id={:#x} data={:p} dataSize={}", label, i, attr.id,
                 static_cast<const void*>(attr.data), attr.dataSize);
        if (attr.data && attr.dataSize) {
            Libraries::Np::NpMatching2::LogRawBuffer("CreateJoinRoom bin attr payload", attr.data,
                                                     static_cast<size_t>(attr.dataSize), 0x40);
        }
    }
}

void DumpRoomDataInternalForCallback(
    const Libraries::Np::NpMatching2::OrbisNpMatching2RoomDataInternal* room) {
    if (!room) {
        LOG_INFO(Lib_NpMatching2, "Request payload roomData: (null)");
        return;
    }

    LOG_INFO(Lib_NpMatching2,
             "Request payload roomData: ptr={:p} roomId={:#x} serverId={} worldId={} lobbyId={} "
             "maxSlot={} publicSlots={} privateSlots={} openPublicSlots={} openPrivateSlots={} "
             "passwdSlotMask={:#x} joinedSlotMask={:#x} flags={:#x} roomGroup={:p} roomGroups={} "
             "roomBinAttrInternal={:p} roomBinAttrInternalNum={}",
             static_cast<const void*>(room), room->roomId, room->serverId, room->worldId,
             room->lobbyId, room->maxSlot, room->publicSlots, room->privateSlots,
             room->openPublicSlots, room->openPrivateSlots, room->passwdSlotMask,
             room->joinedSlotMask, room->flags, static_cast<const void*>(room->roomGroup),
             room->roomGroups, static_cast<const void*>(room->roomBinAttrInternal),
             room->roomBinAttrInternalNum);
}

void DumpRoomMemberDataInternalForCallback(
    const char* tag,
    const Libraries::Np::NpMatching2::OrbisNpMatching2RoomMemberDataInternal* member) {
    if (!member) {
        LOG_INFO(Lib_NpMatching2, "Request payload {}: (null)", tag);
        return;
    }

    LOG_INFO(Lib_NpMatching2,
             "Request payload {}: ptr={:p} next={:p} memberId={} npId='{}' "
             "teamId={} natType={} flagAttr={:#x} joinDate={:#x} roomGroup={:p} "
             "roomMemberBinAttrInternal={:p} roomMemberBinAttrInternalNum={}",
             tag, static_cast<const void*>(member), static_cast<const void*>(member->next),
             member->memberId, member->npId.handle.data, member->teamId, member->natType,
             member->flagAttr, member->joinDate, static_cast<const void*>(member->roomGroup),
             static_cast<const void*>(member->roomMemberBinAttrInternal),
             member->roomMemberBinAttrInternalNum);
}

void DumpCreateJoinRoomResponseForCallback(
    const Libraries::Np::NpMatching2::OrbisNpMatching2CreateJoinRoomResponse* response) {
    if (!response) {
        LOG_INFO(Lib_NpMatching2, "Request payload CreateJoinRoomResponse: (null)");
        return;
    }

    LOG_INFO(Lib_NpMatching2,
             "Request payload CreateJoinRoomResponse: ptr={:p} roomData={:p} members={:p} "
             "membersNum={} me={:p} owner={:p}",
             static_cast<const void*>(response), static_cast<const void*>(response->roomData),
             static_cast<const void*>(response->members.members), response->members.membersNum,
             static_cast<const void*>(response->members.me),
             static_cast<const void*>(response->members.owner));
    DumpRoomDataInternalForCallback(response->roomData);
    DumpRoomMemberDataInternalForCallback("me", response->members.me);
    DumpRoomMemberDataInternalForCallback("owner", response->members.owner);

    const auto* node = response->members.members;
    const u64 limit = std::min<u64>(response->members.membersNum, 8);
    for (u64 i = 0; node && i < limit; ++i) {
        char tag[32];
        std::snprintf(tag, sizeof(tag), "members[%" PRIu64 "]", i);
        DumpRoomMemberDataInternalForCallback(tag, node);
        node = node->next;
    }
}

struct CachedCreateJoinRoomBinAttr {
    Libraries::Np::NpMatching2::OrbisNpMatching2AttributeId id = 0;
    std::vector<u8> data;
};

struct CachedCreateJoinRoomGroupConfig {
    u32 slots = 0;
    bool has_label = false;
    Libraries::Np::NpMatching2::OrbisNpMatching2GroupLabel label{};
    bool has_password = false;
};

struct CachedCreateJoinRoomRequest {
    Libraries::Np::NpMatching2::OrbisNpMatching2RequestId req_id = 0;
    bool valid = false;
    u16 max_slot = 0;
    Libraries::Np::NpMatching2::OrbisNpMatching2TeamId team_id = 0;
    Libraries::Np::NpMatching2::OrbisNpMatching2Flags flags = 0;
    Libraries::Np::NpMatching2::OrbisNpMatching2WorldId world_id = 0;
    Libraries::Np::NpMatching2::OrbisNpMatching2LobbyId lobby_id = 0;
    bool has_room_password = false;
    u64 passwd_slot_mask = 0;
    bool has_join_group_label = false;
    Libraries::Np::NpMatching2::OrbisNpMatching2GroupLabel join_group_label{};
    std::vector<CachedCreateJoinRoomGroupConfig> room_groups;
    std::vector<CachedCreateJoinRoomBinAttr> internal_bin_attrs;
    std::vector<CachedCreateJoinRoomBinAttr> member_internal_bin_attrs;
};

CachedCreateJoinRoomRequest g_last_create_join_room_request;

std::vector<CachedCreateJoinRoomBinAttr> CopyCreateJoinRoomBinAttrs(
    const Libraries::Np::NpMatching2::OrbisNpMatching2BinAttr* attrs, u64 count) {
    std::vector<CachedCreateJoinRoomBinAttr> copied;
    if (!attrs || count == 0) {
        return copied;
    }

    copied.reserve(static_cast<size_t>(count));
    for (u64 i = 0; i < count; ++i) {
        CachedCreateJoinRoomBinAttr attr{};
        attr.id = attrs[i].id;
        if (attrs[i].data && attrs[i].dataSize != 0) {
            const auto size = static_cast<size_t>(attrs[i].dataSize);
            attr.data.assign(attrs[i].data, attrs[i].data + size);
        }
        copied.push_back(std::move(attr));
    }
    return copied;
}

} // anonymous namespace

namespace Libraries::Np::NpMatching2 {

// --- Logging / diagnostics helpers ---

void LogRawBuffer(const char* label, const void* ptr, size_t size, size_t max_size) {
    if (!ptr) {
        LOG_INFO(Lib_NpMatching2, "{}: (null)", label);
        return;
    }

    const auto* bytes = static_cast<const u8*>(ptr);
    const size_t dump_size = std::min(size, max_size);
    LOG_INFO(Lib_NpMatching2, "{}: ptr={:p} size={} dumping={} bytes", label, ptr, size, dump_size);

    for (size_t offset = 0; offset < dump_size; offset += 0x10) {
        char line[128];
        int written = std::snprintf(line, sizeof(line), "%s+0x%04zx:", label, offset);
        for (size_t i = 0; i < 0x10 && (offset + i) < dump_size && written > 0 &&
                           written < static_cast<int>(sizeof(line));
             ++i) {
            written +=
                std::snprintf(line + written, sizeof(line) - written, " %02x", bytes[offset + i]);
        }
        LOG_INFO(Lib_NpMatching2, "{}", line);
    }
}

void LogCreateJoinRoomWordView(const void* req_param, size_t size, size_t max_offset) {
    if (!req_param) {
        return;
    }

    const auto* bytes = static_cast<const u8*>(req_param);
    const size_t view_size = std::min(size, max_offset);

    for (size_t offset = 0; offset + sizeof(u64) <= view_size; offset += sizeof(u64)) {
        u64 qword = 0;
        std::memcpy(&qword, bytes + offset, sizeof(qword));
        const u32 low = static_cast<u32>(qword & 0xffffffffULL);
        const u32 high = static_cast<u32>(qword >> 32);
        LOG_INFO(
            Lib_NpMatching2,
            "CreateJoinRoom reqParam view: +0x{:02x} qword={:#018x} low32={:#010x} high32={:#010x}",
            static_cast<u32>(offset), qword, low, high);
    }
}

void LogCreateJoinRoomDecoded(const void* req) {
    if (!req) {
        return;
    }

    const auto* create_req = reinterpret_cast<const OrbisNpMatching2CreateJoinRoomRequest*>(req);

    LOG_INFO(Lib_NpMatching2,
             "CreateJoinRoom decoded: maxSlot={} teamId={} flags={:#x} worldId={} lobbyId={}",
             create_req->maxSlot, create_req->teamId, create_req->flags, create_req->worldId,
             create_req->lobbyId);
    LOG_INFO(Lib_NpMatching2,
             "CreateJoinRoom decoded: roomPasswd={:p} passwdSlotMask={:p} groupConfig={:p} "
             "groupConfigs={} joinGroupLabel={:p}",
             static_cast<const void*>(create_req->roomPasswd),
             static_cast<const void*>(create_req->passwdSlotMask),
             static_cast<const void*>(create_req->groupConfig), create_req->groupConfigs,
             static_cast<const void*>(create_req->joinGroupLabel));
    LOG_INFO(Lib_NpMatching2,
             "CreateJoinRoom decoded: allowedUser={:p} allowedUsers={} blockedUser={:p} "
             "blockedUsers={}",
             static_cast<const void*>(create_req->allowedUser), create_req->allowedUsers,
             static_cast<const void*>(create_req->blockedUser), create_req->blockedUsers);
    LOG_INFO(Lib_NpMatching2,
             "CreateJoinRoom decoded: internalBinAttr={:p} internalBinAttrs={} "
             "externalSearchIntAttr={:p} externalSearchIntAttrs={}",
             static_cast<const void*>(create_req->internalBinAttr), create_req->internalBinAttrs,
             static_cast<const void*>(create_req->externalSearchIntAttr),
             create_req->externalSearchIntAttrs);
    LOG_INFO(Lib_NpMatching2,
             "CreateJoinRoom decoded: externalSearchBinAttr={:p} externalSearchBinAttrs={} "
             "externalBinAttr={:p} externalBinAttrs={}",
             static_cast<const void*>(create_req->externalSearchBinAttr),
             create_req->externalSearchBinAttrs,
             static_cast<const void*>(create_req->externalBinAttr), create_req->externalBinAttrs);
    LOG_INFO(Lib_NpMatching2,
             "CreateJoinRoom decoded: memberInternalBinAttr={:p} memberInternalBinAttrs={} "
             "signalingParam={:p}",
             static_cast<const void*>(create_req->memberInternalBinAttr),
             create_req->memberInternalBinAttrs,
             static_cast<const void*>(create_req->signalingParam));

    if (create_req->roomPasswd) {
        LogRawBuffer("CreateJoinRoom roomPasswd", create_req->roomPasswd,
                     sizeof(*create_req->roomPasswd), sizeof(*create_req->roomPasswd));
    }
    LogCreateJoinRoomIntAttrArray("CreateJoinRoom externalSearchIntAttr",
                                  create_req->externalSearchIntAttr,
                                  create_req->externalSearchIntAttrs);
    LogCreateJoinRoomBinAttrArray("CreateJoinRoom internalBinAttr", create_req->internalBinAttr,
                                  create_req->internalBinAttrs);
    LogCreateJoinRoomBinAttrArray("CreateJoinRoom externalSearchBinAttr",
                                  create_req->externalSearchBinAttr,
                                  create_req->externalSearchBinAttrs);
    LogCreateJoinRoomBinAttrArray("CreateJoinRoom externalBinAttr", create_req->externalBinAttr,
                                  create_req->externalBinAttrs);
    LogCreateJoinRoomBinAttrArray("CreateJoinRoom memberInternalBinAttr",
                                  create_req->memberInternalBinAttr,
                                  create_req->memberInternalBinAttrs);
    LogCreateJoinRoomSignalingOptions(create_req->signalingParam);
}

// --- Request caching ---

void CacheCreateJoinRoomRequest(const OrbisNpMatching2CreateJoinRoomRequest& req,
                                OrbisNpMatching2RequestId req_id) {
    CachedCreateJoinRoomRequest cached{};
    cached.req_id = req_id;
    cached.valid = true;
    cached.max_slot = req.maxSlot;
    cached.team_id = req.teamId;
    cached.flags = req.flags;
    cached.world_id = req.worldId;
    cached.lobby_id = req.lobbyId;
    cached.has_room_password = req.roomPasswd != nullptr;
    cached.passwd_slot_mask = req.passwdSlotMask ? *req.passwdSlotMask : 0;
    cached.has_join_group_label = req.joinGroupLabel != nullptr;
    if (req.joinGroupLabel) {
        cached.join_group_label = *req.joinGroupLabel;
    }

    if (req.groupConfig && req.groupConfigs != 0) {
        cached.room_groups.reserve(static_cast<size_t>(req.groupConfigs));
        for (u64 i = 0; i < req.groupConfigs; ++i) {
            CachedCreateJoinRoomGroupConfig group{};
            group.slots = req.groupConfig[i].slots;
            group.has_label = req.groupConfig[i].hasLabel;
            group.label = req.groupConfig[i].label;
            group.has_password = req.groupConfig[i].hasPassword;
            cached.room_groups.push_back(group);
        }
    }

    cached.internal_bin_attrs =
        CopyCreateJoinRoomBinAttrs(req.internalBinAttr, req.internalBinAttrs);
    cached.member_internal_bin_attrs =
        CopyCreateJoinRoomBinAttrs(req.memberInternalBinAttr, req.memberInternalBinAttrs);

    g_last_create_join_room_request = std::move(cached);
}

// --- NpMatching2State member definitions ---

void NpMatching2State::FreeCallbackData() {
    delete last_create_join_response;
    last_create_join_response = nullptr;
    delete last_search_room_response;
    last_search_room_response = nullptr;
    delete last_ping_info_response;
    last_ping_info_response = nullptr;
    last_request_data = nullptr;
    delete last_room_data;
    last_room_data = nullptr;
    delete[] last_room_data_external;
    last_room_data_external = nullptr;
    delete[] last_room_owner_np_ids;
    last_room_owner_np_ids = nullptr;
    delete[] last_room_groups;
    last_room_groups = nullptr;
    delete[] last_room_bin_attr_internal;
    last_room_bin_attr_internal = nullptr;
    delete[] last_room_member_bin_attr_internal;
    last_room_member_bin_attr_internal = nullptr;
    delete[] last_member_data;
    last_member_data = nullptr;
    for (auto* buffer : last_request_bin_buffers) {
        delete[] buffer;
    }
    last_request_bin_buffers.clear();
    for (auto* p : callback_data_pointers) {
        std::free(p);
    }
    callback_data_pointers.clear();
    last_member_count = 0;
    last_room_data_external_count = 0;
    last_room_group_count = 0;
    last_room_bin_attr_internal_count = 0;
    last_room_member_bin_attr_internal_count = 0;
}

void NpMatching2State::StopPollThread() {
    ConfigureMmNotificationHandlers({});
    StopMmClient();

    dispatch_running = false;
    if (dispatch_thread != nullptr) {
        Kernel::posix_pthread_join(dispatch_thread, nullptr);
        dispatch_thread = nullptr;
    }
    {
        std::lock_guard<std::mutex> lock(event_queue_mutex);
        pending_events.clear();
    }
}

// --- Global state ---

NpMatching2State g_state;

// --- Event scheduling ---

void ScheduleEvent(PendingEvent ev) {
    auto now = std::chrono::steady_clock::now();
    auto delay_ms = std::chrono::duration_cast<std::chrono::milliseconds>(ev.fire_at - now).count();
    auto abs_ms =
        std::chrono::duration_cast<std::chrono::milliseconds>(now.time_since_epoch()).count();
    LOG_INFO(Lib_NpMatching2,
             "ScheduleEvent: type={} delay={}ms abs_t={}ms | "
             "ctx_event={:#x} req_event={:#x} sig_event={:#x} room_event={:#x} "
             "reqId={} memberId={} roomId={} connId={}",
             static_cast<int>(ev.type), delay_ms, abs_ms, ev.ctx_event, ev.req_event, ev.sig_event,
             ev.room_event, ev.req_id, ev.member_id, ev.room_id, ev.conn_id);
    std::lock_guard<std::mutex> lock(g_state.event_queue_mutex);
    g_state.pending_events.push_back(std::move(ev));
}

// Event dispatch thread — processes queued events from a single PS4 thread.
// Started once at ContextStart, polls every 50ms.
PS4_SYSV_ABI void* EventDispatchThreadFunc(void* /*arg*/) {
    LOG_INFO(Lib_NpMatching2, "EventDispatch thread started");

    while (g_state.dispatch_running) {
        std::this_thread::sleep_for(std::chrono::milliseconds(50));
        if (!g_state.dispatch_running)
            break;

        auto now = std::chrono::steady_clock::now();
        std::vector<PendingEvent> ready;

        {
            std::lock_guard<std::mutex> lock(g_state.event_queue_mutex);
            auto it = g_state.pending_events.begin();
            while (it != g_state.pending_events.end()) {
                if (it->fire_at <= now) {
                    ready.push_back(std::move(*it));
                    it = g_state.pending_events.erase(it);
                } else {
                    ++it;
                }
            }
        }

        if (ready.empty())
            continue;

        // Sort by type priority: CONTEXT < REQUEST < ROOM_EVENT < SIGNALING
        // Within same type, preserve insertion order (stable_sort).
        std::stable_sort(ready.begin(), ready.end(),
                         [](const PendingEvent& a, const PendingEvent& b) {
                             auto priority = [](PendingEvent::Type type) {
                                 switch (type) {
                                 case PendingEvent::CONTEXT_CB:
                                     return 0;
                                 case PendingEvent::REQUEST_CB:
                                     return 1;
                                 case PendingEvent::ROOM_EVENT_CB:
                                     return 2;
                                 case PendingEvent::LOBBY_EVENT_CB:
                                     return 2;
                                 case PendingEvent::ROOM_MESSAGE_CB:
                                     return 2;
                                 case PendingEvent::LOBBY_MESSAGE_CB:
                                     return 2;
                                 case PendingEvent::SIGNALING_CB:
                                     return 3;
                                 default:
                                     return 4;
                                 }
                             };
                             return priority(a.type) < priority(b.type);
                         });

        // Deduplicate REQUEST_CB events: only keep the first one per reqId+event.
        // This prevents double-firing which crashes the game's dispatch handler.
        {
            std::set<u64> seen_req;
            auto wit = ready.begin();
            for (auto rit = ready.begin(); rit != ready.end(); ++rit) {
                if (rit->type == PendingEvent::REQUEST_CB) {
                    u64 key = (static_cast<u64>(rit->req_id) << 16) | rit->req_event;
                    if (!seen_req.insert(key).second) {
                        LOG_WARNING(Lib_NpMatching2,
                                    "EventDispatch: removing duplicate REQUEST_CB "
                                    "event={:#x} reqId={} (ready.size={})",
                                    rit->req_event, rit->req_id, ready.size());
                        continue; // skip this duplicate
                    }
                }
                if (wit != rit)
                    *wit = std::move(*rit);
                ++wit;
            }
            ready.erase(wit, ready.end());
        }

        auto fire_abs_ms = std::chrono::duration_cast<std::chrono::milliseconds>(
                               std::chrono::steady_clock::now().time_since_epoch())
                               .count();
        LOG_INFO(Lib_NpMatching2, "EventDispatch: processing {} ready events at abs_t={}ms",
                 ready.size(), fire_abs_ms);

        for (auto& ev : ready) {
            switch (ev.type) {
            case PendingEvent::CONTEXT_CB:
                if (g_state.context_callback) {
                    LOG_INFO(Lib_NpMatching2,
                             "EventDispatch: context callback event={:#x} cause={} err={}\n"
                             "  INVOKE context_cb={:p} ctxId={} event={:#x} eventCause={} "
                             "errorCode={} arg={:p}",
                             ev.ctx_event, ev.ctx_event_cause, ev.error_code,
                             static_cast<void*>(g_state.context_callback), g_state.ctx.ctx_id,
                             ev.ctx_event, ev.ctx_event_cause, ev.error_code,
                             g_state.context_callback_arg);
                    g_state.context_callback(g_state.ctx.ctx_id, ev.ctx_event, ev.ctx_event_cause,
                                             ev.error_code, g_state.context_callback_arg);
                    LOG_INFO(Lib_NpMatching2, "EventDispatch: context callback returned");
                }
                break;

            case PendingEvent::REQUEST_CB:
                if (ev.request_cb) {
                    // Dedup guard: skip if we already fired this exact reqId+event.
                    // The game's RequestCallback_dispatch clears +0x458 after
                    // processing, so a duplicate would crash on the guard check
                    // (or worse, create duplicate ConnectionObjects).
                    if (ev.req_id == g_state.last_fired_req_id &&
                        ev.req_event == g_state.last_fired_req_event) {
                        LOG_WARNING(Lib_NpMatching2,
                                    "EventDispatch: SKIPPING duplicate request callback "
                                    "event={:#x} reqId={}",
                                    ev.req_event, ev.req_id);
                        break;
                    }
                    void* data = ev.request_data;
                    if (ev.req_event == ORBIS_NP_MATCHING2_REQUEST_EVENT_CREATE_JOIN_ROOM ||
                        ev.req_event == ORBIS_NP_MATCHING2_REQUEST_EVENT_JOIN_ROOM) {
                        DumpCreateJoinRoomResponseForCallback(
                            reinterpret_cast<const OrbisNpMatching2CreateJoinRoomResponse*>(data));
                    }
                    LOG_INFO(Lib_NpMatching2,
                             "EventDispatch: request callback event={:#x} reqId={} data={:p}\n"
                             "  INVOKE request_cb={:p} ctxId={} reqId={} event={:#x} "
                             "errorCode={} data={:p} arg={:p}",
                             ev.req_event, ev.req_id, data, static_cast<void*>(ev.request_cb),
                             g_state.ctx.ctx_id, ev.req_id, ev.req_event, ev.error_code, data,
                             ev.request_cb_arg);
                    g_state.last_fired_req_id = ev.req_id;
                    g_state.last_fired_req_event = ev.req_event;
                    ev.request_cb(g_state.ctx.ctx_id, ev.req_id, ev.req_event, ev.error_code, data,
                                  ev.request_cb_arg);
                    LOG_INFO(Lib_NpMatching2,
                             "EventDispatch: request callback returned (event={:#x} reqId={})",
                             ev.req_event, ev.req_id);
                }
                break;

            case PendingEvent::SIGNALING_CB:
                if (g_state.signaling_callback) {
                    LOG_INFO(Lib_NpMatching2,
                             "EventDispatch: signaling callback event={:#x} member={} conn={}\n"
                             "  INVOKE signaling_cb={:p} ctxId={} roomId={} memberId={} "
                             "event={:#x} connId={} arg={:p}",
                             ev.sig_event, ev.member_id, ev.conn_id,
                             static_cast<void*>(g_state.signaling_callback), g_state.ctx.ctx_id,
                             ev.room_id, ev.member_id, ev.sig_event, ev.conn_id,
                             g_state.signaling_callback_arg);
                    g_state.signaling_callback(g_state.ctx.ctx_id, ev.room_id, ev.member_id,
                                               ev.sig_event, 0, g_state.signaling_callback_arg);
                    LOG_INFO(Lib_NpMatching2,
                             "EventDispatch: signaling callback returned (event={:#x} member={})",
                             ev.sig_event, ev.member_id);
                }
                break;

            case PendingEvent::ROOM_EVENT_CB:
                if (g_state.room_event_callback) {
                    // Decode the RoomMemberUpdateInfo pointed to by ev.room_event_data
                    auto* upd =
                        reinterpret_cast<OrbisNpMatching2RoomMemberUpdateInfo*>(ev.room_event_data);
                    auto* mdp = upd ? upd->roomMemberDataInternal : nullptr;
                    LOG_INFO(Lib_NpMatching2,
                             "EventDispatch: room event callback event={:#x} room={} "
                             "cbArg={:p} eventData={:p}\n"
                             "  INVOKE room_event_cb={:p} ctxId={} roomId={} event={:#x} "
                             "data={:p} arg={:p}\n"
                             "  updateInfo: roomMemberData={:p} eventCause={}\n"
                             "  memberData: memberId={} npId='{}' teamId={} natType={} "
                             "flags={:#x}",
                             ev.room_event, ev.room_id, g_state.room_event_callback_arg,
                             ev.room_event_data, static_cast<void*>(g_state.room_event_callback),
                             g_state.ctx.ctx_id, ev.room_id, ev.room_event, ev.room_event_data,
                             g_state.room_event_callback_arg,
                             upd ? static_cast<void*>(upd->roomMemberDataInternal) : nullptr,
                             upd ? upd->eventCause : 0xff, mdp ? mdp->memberId : 0,
                             mdp ? mdp->npId.handle.data : "(null)", mdp ? mdp->teamId : 0,
                             mdp ? mdp->natType : 0, mdp ? mdp->flagAttr : 0);
                    g_state.room_event_callback(g_state.ctx.ctx_id, ev.room_id, ev.room_event,
                                                ev.room_event_data,
                                                g_state.room_event_callback_arg);
                    LOG_INFO(Lib_NpMatching2,
                             "EventDispatch: room event callback returned (event={:#x} room={})",
                             ev.room_event, ev.room_id);
                }
                break;

            case PendingEvent::LOBBY_EVENT_CB:
                if (g_state.lobby_event_callback) {
                    LOG_INFO(Lib_NpMatching2,
                             "EventDispatch: lobby event callback event={:#x} lobby={} data={:p}\n"
                             "  INVOKE lobby_event_cb={:p} ctxId={} lobbyId={} event={:#x} "
                             "data={:p} arg={:p}",
                             ev.lobby_event, ev.lobby_id, ev.lobby_event_data,
                             static_cast<void*>(g_state.lobby_event_callback), g_state.ctx.ctx_id,
                             ev.lobby_id, ev.lobby_event, ev.lobby_event_data,
                             g_state.lobby_event_callback_arg);
                    g_state.lobby_event_callback(g_state.ctx.ctx_id, ev.lobby_id, ev.lobby_event,
                                                 ev.lobby_event_data,
                                                 g_state.lobby_event_callback_arg);
                    LOG_INFO(Lib_NpMatching2,
                             "EventDispatch: lobby event callback returned (event={:#x} lobby={})",
                             ev.lobby_event, ev.lobby_id);
                }
                break;

            case PendingEvent::LOBBY_MESSAGE_CB:
                if (g_state.lobby_message_callback) {
                    LOG_INFO(Lib_NpMatching2,
                             "EventDispatch: lobby message callback event={:#x} lobby={} src={} "
                             "data={:p}\n"
                             "  INVOKE lobby_msg_cb={:p} ctxId={} lobbyId={} srcMemberId={} "
                             "event={:#x} data={:p} arg={:p}",
                             ev.msg_event, ev.lobby_id, ev.src_member_id, ev.message_data,
                             static_cast<void*>(g_state.lobby_message_callback), g_state.ctx.ctx_id,
                             ev.lobby_id, ev.src_member_id, ev.msg_event, ev.message_data,
                             g_state.lobby_message_callback_arg);
                    g_state.lobby_message_callback(g_state.ctx.ctx_id, ev.lobby_id,
                                                   ev.src_member_id, ev.msg_event, ev.message_data,
                                                   g_state.lobby_message_callback_arg);
                    LOG_INFO(
                        Lib_NpMatching2,
                        "EventDispatch: lobby message callback returned (event={:#x} lobby={})",
                        ev.msg_event, ev.lobby_id);
                }
                break;

            case PendingEvent::ROOM_MESSAGE_CB:
                if (g_state.room_message_callback) {
                    LOG_INFO(Lib_NpMatching2,
                             "EventDispatch: room message callback event={:#x} room={} src={} "
                             "data={:p}\n"
                             "  INVOKE room_msg_cb={:p} ctxId={} roomId={} srcMemberId={} "
                             "event={:#x} data={:p} arg={:p}",
                             ev.msg_event, ev.room_id, ev.src_member_id, ev.message_data,
                             static_cast<void*>(g_state.room_message_callback), g_state.ctx.ctx_id,
                             ev.room_id, ev.src_member_id, ev.msg_event, ev.message_data,
                             g_state.room_message_callback_arg);
                    g_state.room_message_callback(g_state.ctx.ctx_id, ev.room_id, ev.src_member_id,
                                                  ev.msg_event, ev.message_data,
                                                  g_state.room_message_callback_arg);
                    LOG_INFO(Lib_NpMatching2,
                             "EventDispatch: room message callback returned (event={:#x} room={})",
                             ev.msg_event, ev.room_id);
                }
                break;

            default:
                break;
            }
        }
    }

    LOG_INFO(Lib_NpMatching2, "EventDispatch thread stopped");
    return nullptr;
}

// --- Signaling event scheduling ---

void ScheduleSignalingEventForMemberId(OrbisNpMatching2RoomMemberId member_id,
                                       OrbisNpMatching2Event sig_event,
                                       std::chrono::steady_clock::time_point fire_at) {
    std::lock_guard<std::mutex> lock(g_state.mutex);

    const auto it = g_state.peers.find(member_id);
    if (it == g_state.peers.end()) {
        return;
    }

    PeerInfo& peer = it->second;
    peer.status = (sig_event == ORBIS_NP_MATCHING2_SIGNALING_EVENT_ESTABLISHED)
                      ? ORBIS_NP_MATCHING2_SIGNALING_CONN_STATUS_ACTIVE
                      : ORBIS_NP_MATCHING2_SIGNALING_CONN_STATUS_INACTIVE;
    peer.bootstrap_dead_sent = (sig_event == ORBIS_NP_MATCHING2_SIGNALING_EVENT_DEAD);

    PendingEvent ev{};
    ev.type = PendingEvent::SIGNALING_CB;
    ev.fire_at = fire_at;
    ev.room_id = g_state.ctx.room_id;
    ev.member_id = member_id;
    ev.sig_event = sig_event;
    ev.conn_id = static_cast<u32>(peer.conn_id);
    ScheduleEvent(std::move(ev));

    LOG_INFO(Lib_NpMatching2,
             "t={} Signaling hook: member={} conn={} event={:#x} status={} online_id='{}'", NowMs(),
             member_id, peer.conn_id, sig_event, peer.status, peer.online_id);
}

void ScheduleSignalingEventForMemberId(OrbisNpMatching2RoomMemberId member_id,
                                       OrbisNpMatching2Event sig_event) {
    ScheduleSignalingEventForMemberId(member_id, sig_event, std::chrono::steady_clock::now());
}

// --- Callback data construction ---

void* BuildRequestCallbackData(OrbisNpMatching2Event req_event, OrbisNpMatching2RequestId req_id,
                               OrbisNpMatching2RoomId room_id, OrbisNpMatching2ServerId server_id,
                               OrbisNpMatching2WorldId world_id, OrbisNpMatching2LobbyId lobby_id,
                               const std::vector<MemberInfo>& members,
                               OrbisNpMatching2RoomMemberId host_member_id,
                               OrbisNpMatching2RoomMemberId my_member_id, u32 max_slot,
                               OrbisNpMatching2Flags flag_attr) {
    std::lock_guard<std::mutex> lock(g_state.mutex);

    // Free previous data
    g_state.FreeCallbackData();

    int num_members = static_cast<int>(members.size());
    if (num_members == 0)
        num_members = 1; // At minimum, the local player

    const CachedCreateJoinRoomRequest* create_join_request = nullptr;
    if (req_event == ORBIS_NP_MATCHING2_REQUEST_EVENT_CREATE_JOIN_ROOM &&
        g_last_create_join_room_request.valid && g_last_create_join_room_request.req_id == req_id) {
        create_join_request = &g_last_create_join_room_request;
        LOG_INFO(Lib_NpMatching2,
                 "BuildRequestCallbackData(0x101): using cached request reqId={} worldId={} "
                 "lobbyId={} maxSlot={} flags={:#x} passwdSlotMask={:#x}",
                 req_id, create_join_request->world_id, create_join_request->lobby_id,
                 create_join_request->max_slot, create_join_request->flags,
                 create_join_request->passwd_slot_mask);
    } else if (req_event == ORBIS_NP_MATCHING2_REQUEST_EVENT_CREATE_JOIN_ROOM) {
        LOG_WARNING(Lib_NpMatching2,
                    "BuildRequestCallbackData(0x101): no cached request for reqId={}, falling "
                    "back to MM data",
                    req_id);
    }

    const auto effective_world_id = create_join_request ? create_join_request->world_id : world_id;
    const auto effective_lobby_id = create_join_request ? create_join_request->lobby_id : lobby_id;
    const auto effective_max_slot = create_join_request
                                        ? std::max<u16>(create_join_request->max_slot, 1)
                                        : static_cast<u16>(std::max<u32>(max_slot, 1));
    const auto effective_flags = create_join_request ? create_join_request->flags : flag_attr;

    // Allocate room data
    auto* room = new OrbisNpMatching2RoomDataInternal();
    std::memset(room, 0, sizeof(*room));
    room->serverId = server_id;
    room->worldId = effective_world_id;
    room->lobbyId = effective_lobby_id;
    room->roomId = room_id;
    room->maxSlot = effective_max_slot;
    room->publicSlots = effective_max_slot;
    room->openPublicSlots =
        static_cast<u16>(effective_max_slot > num_members ? effective_max_slot - num_members : 0);
    room->flags = effective_flags;
    if (create_join_request) {
        room->passwdSlotMask = create_join_request->passwd_slot_mask;
    }
    g_state.last_room_data = room;

    for (const auto& member : members) {
        if (member.member_id != 0) {
            room->joinedSlotMask |= (1ull << (member.member_id - 1));
        }
    }

    int assigned_group_index = -1;
    if (create_join_request && !create_join_request->room_groups.empty()) {
        const auto group_count = static_cast<int>(create_join_request->room_groups.size());
        g_state.last_room_groups = new OrbisNpMatching2RoomGroup[group_count]();
        g_state.last_room_group_count = group_count;
        room->roomGroup = g_state.last_room_groups;
        room->roomGroups = static_cast<u64>(group_count);

        for (int i = 0; i < group_count; ++i) {
            const auto& src = create_join_request->room_groups[i];
            auto& dst = g_state.last_room_groups[i];
            dst.id = static_cast<OrbisNpMatching2RoomGroupId>(i + 1);
            dst.hasPasswd = src.has_password;
            dst.hasLabel = src.has_label;
            dst.label = src.label;
            dst.slots = src.slots;
            dst.groupMembers = 0;

            if (assigned_group_index == -1) {
                if (create_join_request->has_join_group_label && src.has_label &&
                    std::memcmp(src.label.data, create_join_request->join_group_label.data,
                                sizeof(src.label.data)) == 0) {
                    assigned_group_index = i;
                } else if (!create_join_request->has_join_group_label && group_count == 1) {
                    assigned_group_index = i;
                }
            }
        }
    }

    if (create_join_request && !create_join_request->internal_bin_attrs.empty()) {
        const auto attr_count = static_cast<int>(create_join_request->internal_bin_attrs.size());
        g_state.last_room_bin_attr_internal = new OrbisNpMatching2RoomBinAttrInternal[attr_count]();
        g_state.last_room_bin_attr_internal_count = attr_count;
        room->roomBinAttrInternal = g_state.last_room_bin_attr_internal;
        room->roomBinAttrInternalNum = static_cast<u64>(attr_count);

        for (int i = 0; i < attr_count; ++i) {
            const auto& src = create_join_request->internal_bin_attrs[i];
            auto& dst = g_state.last_room_bin_attr_internal[i];
            dst.memberId = my_member_id;
            dst.binAttr.id = src.id;
            dst.binAttr.dataSize = static_cast<u64>(src.data.size());
            if (!src.data.empty()) {
                auto* buffer = new u8[src.data.size()];
                std::memcpy(buffer, src.data.data(), src.data.size());
                dst.binAttr.data = buffer;
                g_state.last_request_bin_buffers.push_back(buffer);
            }
        }
    }

    if (req_event == ORBIS_NP_MATCHING2_REQUEST_EVENT_GET_ROOM_DATA_INTERNAL) {
        g_state.last_request_data = room;
        LOG_INFO(Lib_NpMatching2,
                 "BuildRequestCallbackData(0x109): roomId={:#x} serverId={} worldId={} "
                 "lobbyId={} maxSlot={} flags={:#x} joinedSlotMask={:#x} room@{:p}",
                 room->roomId, room->serverId, room->worldId, room->lobbyId, room->maxSlot,
                 room->flags, room->joinedSlotMask, static_cast<void*>(room));
        return room;
    }

    if (req_event == ORBIS_NP_MATCHING2_REQUEST_EVENT_CREATE_JOIN_ROOM ||
        req_event == ORBIS_NP_MATCHING2_REQUEST_EVENT_JOIN_ROOM) {
        auto* member_arr = new OrbisNpMatching2RoomMemberDataInternal[num_members]();
        OrbisNpMatching2RoomMemberDataInternal* me_ptr = nullptr;
        OrbisNpMatching2RoomMemberDataInternal* owner_ptr = nullptr;

        // Count total member bin attrs: from cached request (for self on 0x101)
        // and from MemberInfo::bin_attrs (from server response blob on 0x102).
        int total_member_bin_attrs = 0;
        if (create_join_request) {
            total_member_bin_attrs +=
                static_cast<int>(create_join_request->member_internal_bin_attrs.size());
        }
        for (const auto& mi : members) {
            total_member_bin_attrs += static_cast<int>(mi.bin_attrs.size());
        }

        if (total_member_bin_attrs > 0) {
            g_state.last_room_member_bin_attr_internal =
                new OrbisNpMatching2RoomMemberBinAttrInternal[total_member_bin_attrs]();
            g_state.last_room_member_bin_attr_internal_count = total_member_bin_attrs;

            int slot = 0;
            // Fill from cached request (host's own member bin attrs for 0x101)
            if (create_join_request) {
                for (const auto& src : create_join_request->member_internal_bin_attrs) {
                    auto& dst = g_state.last_room_member_bin_attr_internal[slot++];
                    dst.binAttr.id = src.id;
                    dst.binAttr.dataSize = static_cast<u64>(src.data.size());
                    if (!src.data.empty()) {
                        auto* buffer = new u8[src.data.size()];
                        std::memcpy(buffer, src.data.data(), src.data.size());
                        dst.binAttr.data = buffer;
                        g_state.last_request_bin_buffers.push_back(buffer);
                    }
                }
            }
            // Fill from MemberInfo::bin_attrs (server response blob data)
            for (const auto& mi : members) {
                for (const auto& ba : mi.bin_attrs) {
                    auto& dst = g_state.last_room_member_bin_attr_internal[slot++];
                    dst.binAttr.id = ba.id;
                    dst.binAttr.dataSize = static_cast<u64>(ba.data.size());
                    if (!ba.data.empty()) {
                        auto* buffer = new u8[ba.data.size()];
                        std::memcpy(buffer, ba.data.data(), ba.data.size());
                        dst.binAttr.data = buffer;
                        g_state.last_request_bin_buffers.push_back(buffer);
                    }
                }
            }
        }

        // Track per-member offsets into the combined bin attr array
        // so each member's pointer can be set correctly.
        // cached_attr_count = number of attrs from create_join_request (for self on 0x101)
        int cached_attr_count =
            create_join_request
                ? static_cast<int>(create_join_request->member_internal_bin_attrs.size())
                : 0;

        for (int i = 0; i < num_members; i++) {
            std::memset(&member_arr[i], 0, sizeof(member_arr[i]));
            member_arr[i].next = (i + 1 < num_members) ? &member_arr[i + 1] : nullptr;

            if (i < static_cast<int>(members.size())) {
                member_arr[i].memberId = static_cast<u16>(members[i].member_id);
                std::strncpy(member_arr[i].npId.handle.data, members[i].online_id.c_str(),
                             ORBIS_NP_ONLINEID_MAX_LENGTH - 1);
            } else {
                member_arr[i].memberId = my_member_id;
            }

            if (member_arr[i].memberId == my_member_id) {
                if (create_join_request) {
                    member_arr[i].teamId = create_join_request->team_id;
                    if (assigned_group_index >= 0 && g_state.last_room_groups) {
                        member_arr[i].roomGroup = &g_state.last_room_groups[assigned_group_index];
                        g_state.last_room_groups[assigned_group_index].groupMembers += 1;
                    }
                    if (cached_attr_count > 0 && g_state.last_room_member_bin_attr_internal) {
                        member_arr[i].roomMemberBinAttrInternal =
                            g_state.last_room_member_bin_attr_internal;
                        member_arr[i].roomMemberBinAttrInternalNum =
                            static_cast<u64>(cached_attr_count);
                    }
                }
                me_ptr = &member_arr[i];
            }

            // Assign bin attrs from MemberInfo (server response blob) for non-self members,
            // or for self when there's no cached request (i.e. JoinRoom 0x102 case)
            if (i < static_cast<int>(members.size()) && !members[i].bin_attrs.empty() &&
                g_state.last_room_member_bin_attr_internal &&
                !(member_arr[i].memberId == my_member_id && cached_attr_count > 0)) {
                // Find the offset for this member's bin attrs in the combined array.
                // Cached attrs come first, then MemberInfo attrs in members[] order.
                int offset = cached_attr_count;
                for (int j = 0; j < i; ++j) {
                    if (j < static_cast<int>(members.size()))
                        offset += static_cast<int>(members[j].bin_attrs.size());
                }
                member_arr[i].roomMemberBinAttrInternal =
                    &g_state.last_room_member_bin_attr_internal[offset];
                member_arr[i].roomMemberBinAttrInternalNum =
                    static_cast<u64>(members[i].bin_attrs.size());
            }

            if (member_arr[i].memberId == host_member_id) {
                member_arr[i].flagAttr |= ORBIS_NP_MATCHING2_ROOM_MEMBER_FLAG_ATTR_OWNER;
                if (create_join_request && my_member_id == host_member_id) {
                    member_arr[i].teamId = create_join_request->team_id;
                }
                owner_ptr = &member_arr[i];
            }
        }

        auto* response = new OrbisNpMatching2CreateJoinRoomResponse();
        response->roomData = room;
        response->members.members = &member_arr[0];
        response->members.membersNum = static_cast<u64>(num_members);
        response->members.me = me_ptr;
        response->members.owner = owner_ptr ? owner_ptr : &member_arr[0];

        g_state.last_member_data = member_arr;
        g_state.last_member_count = num_members;
        g_state.last_create_join_response = response;
        g_state.last_request_data = response;

        LOG_INFO(Lib_NpMatching2,
                 "BuildRequestCallbackData({:#x}): roomData@{:p} members@{:p} membersNum={} "
                 "me@{:p} owner@{:p}",
                 req_event, static_cast<const void*>(response->roomData),
                 static_cast<const void*>(response->members.members), response->members.membersNum,
                 static_cast<const void*>(response->members.me),
                 static_cast<const void*>(response->members.owner));
        for (int i = 0; i < num_members; i++) {
            LOG_INFO(Lib_NpMatching2,
                     "  member[{}] @{:p}: next={:p} memberId={} npId='{}' teamId={} natType={} "
                     "flagAttr={:#x} joinDate={:#x} roomGroup={:p} binAttr={:p} binAttrNum={}",
                     i, (void*)&member_arr[i], static_cast<void*>(member_arr[i].next),
                     member_arr[i].memberId, member_arr[i].npId.handle.data, member_arr[i].teamId,
                     member_arr[i].natType, member_arr[i].flagAttr, member_arr[i].joinDate,
                     static_cast<void*>(member_arr[i].roomGroup),
                     static_cast<void*>(member_arr[i].roomMemberBinAttrInternal),
                     member_arr[i].roomMemberBinAttrInternalNum);
        }
        return response;
    }

    g_state.last_request_data = room;
    LOG_WARNING(Lib_NpMatching2,
                "BuildRequestCallbackData: no concrete response model for req_event={:#x}, "
                "returning roomData only @{:p}",
                req_event, static_cast<void*>(room));
    return room;
}

OrbisNpMatching2SearchRoomResponse* BuildSearchRoomResponseFromBinary(
    const std::vector<u8>& replyData) {
    NpMatching2::BinaryReader r(replyData);

    std::lock_guard<std::mutex> lock(g_state.mutex);
    g_state.FreeCallbackData();

    auto* response = new OrbisNpMatching2SearchRoomResponse();
    std::memset(response, 0, sizeof(*response));

    const u32 roomCount = r.GetU32LE();
    response->range.start = 0;
    response->range.total = roomCount;
    response->range.results = roomCount;

    if (roomCount == 0 || r.error()) {
        g_state.last_search_room_response = response;
        g_state.last_request_data = response;
        LOG_INFO(Lib_NpMatching2, "BuildSearchRoomResponseFromBinary: empty result set");
        return response;
    }

    auto* room_data = new OrbisNpMatching2RoomDataExternal[roomCount]();
    auto* owner_np_ids = new Libraries::Np::OrbisNpId[roomCount]();
    for (u32 i = 0; i < roomCount && !r.error(); ++i) {
        auto& room = room_data[i];
        room.next = (i + 1 < roomCount) ? &room_data[i + 1] : nullptr;
        room.maxSlot = r.GetU16LE();
        room.curMembers = r.GetU16LE();
        room.flags = static_cast<OrbisNpMatching2Flags>(r.GetU32LE());
        room.serverId = static_cast<OrbisNpMatching2ServerId>(r.GetU16LE());
        room.worldId = static_cast<OrbisNpMatching2WorldId>(r.GetU32LE());
        room.lobbyId = static_cast<OrbisNpMatching2LobbyId>(r.GetU64LE());
        room.roomId = static_cast<OrbisNpMatching2RoomId>(r.GetU64LE());
        room.passwdSlotMask = r.GetU64LE();
        room.joinedSlotMask = r.GetU64LE();
        room.publicSlots = r.GetU16LE();
        room.privateSlots = r.GetU16LE();
        room.openPublicSlots = r.GetU16LE();
        room.openPrivateSlots = r.GetU16LE();

        // Room groups
        u16 groupCount = r.GetU16LE();
        for (u16 g = 0; g < groupCount && !r.error(); ++g) {
            r.GetU16LE(); // groupId
            r.GetU8();    // hasPasswd
            r.GetU16LE(); // slots
            r.GetU16LE(); // groupMembers
        }
        // External search int attrs
        u16 searchIntCount = r.GetU16LE();
        OrbisNpMatching2IntAttr* searchIntArr = nullptr;
        if (searchIntCount > 0) {
            searchIntArr = static_cast<OrbisNpMatching2IntAttr*>(
                std::calloc(searchIntCount, sizeof(OrbisNpMatching2IntAttr)));
            g_state.callback_data_pointers.push_back(searchIntArr);
        }
        for (u16 j = 0; j < searchIntCount && !r.error(); ++j) {
            searchIntArr[j].id = r.GetU16LE();
            searchIntArr[j].attr = static_cast<u32>(r.GetU64LE());
        }
        // External search bin attrs
        u16 searchBinCount = r.GetU16LE();
        OrbisNpMatching2BinAttr* searchBinArr = nullptr;
        if (searchBinCount > 0) {
            searchBinArr = static_cast<OrbisNpMatching2BinAttr*>(
                std::calloc(searchBinCount, sizeof(OrbisNpMatching2BinAttr)));
            g_state.callback_data_pointers.push_back(searchBinArr);
        }
        for (u16 j = 0; j < searchBinCount && !r.error(); ++j) {
            searchBinArr[j].id = r.GetU16LE();
            u32 sz = r.GetU32LE();
            searchBinArr[j].dataSize = sz;
            if (sz > 0) {
                auto* buf = static_cast<u8*>(std::malloc(sz));
                g_state.callback_data_pointers.push_back(buf);
                for (u32 b = 0; b < sz && !r.error(); ++b)
                    buf[b] = r.GetU8();
                searchBinArr[j].data = buf;
            }
        }
        // External bin attrs
        u16 extBinCount = r.GetU16LE();
        OrbisNpMatching2BinAttr* extBinArr = nullptr;
        if (extBinCount > 0) {
            extBinArr = static_cast<OrbisNpMatching2BinAttr*>(
                std::calloc(extBinCount, sizeof(OrbisNpMatching2BinAttr)));
            g_state.callback_data_pointers.push_back(extBinArr);
        }
        for (u16 j = 0; j < extBinCount && !r.error(); ++j) {
            extBinArr[j].id = r.GetU16LE();
            u32 sz = r.GetU32LE();
            extBinArr[j].dataSize = sz;
            if (sz > 0) {
                auto* buf = static_cast<u8*>(std::malloc(sz));
                g_state.callback_data_pointers.push_back(buf);
                for (u32 b = 0; b < sz && !r.error(); ++b)
                    buf[b] = r.GetU8();
                extBinArr[j].data = buf;
            }
        }

        // Owner NpId — the game reads RoomDataExternal+0x40 as an OrbisNpId* and
        // immediately dereferences it with no null check in the SearchRoom (0x106) callback.
        // Must be populated for every room or the game will crash.
        auto& oid = owner_np_ids[i];
        for (int b = 0; b < 16; ++b)
            oid.handle.data[b] = static_cast<char>(r.GetU8());
        oid.handle.term = static_cast<s8>(r.GetU8());
        for (int b = 0; b < 3; ++b)
            oid.handle.dummy[b] = static_cast<s8>(r.GetU8());
        for (int b = 0; b < 8; ++b)
            oid.opt[b] = r.GetU8();
        for (int b = 0; b < 8; ++b)
            oid.reserved[b] = r.GetU8();
        room.ownerNpId = &owner_np_ids[i];

        room.roomGroup = nullptr;
        room.roomGroups = 0;
        room.externalSearchIntAttr = searchIntArr;
        room.externalSearchIntAttrs = searchIntCount;
        room.externalSearchBinAttr = searchBinArr;
        room.externalSearchBinAttrs = searchBinCount;
        room.externalBinAttr = extBinArr;
        room.externalBinAttrs = extBinCount;

        LOG_INFO(Lib_NpMatching2,
                 "BuildSearchRoomResponseFromBinary: room[{}] roomId={:#x} worldId={} lobbyId={} "
                 "members={} maxSlot={} flags={:#x}",
                 i, room.roomId, room.worldId, room.lobbyId, room.curMembers, room.maxSlot,
                 room.flags);
    }

    response->roomDataExt = room_data;
    g_state.last_room_data_external = room_data;
    g_state.last_room_owner_np_ids = owner_np_ids;
    g_state.last_room_data_external_count = static_cast<int>(roomCount);
    g_state.last_search_room_response = response;
    g_state.last_request_data = response;
    return response;
}

// --- Context ID counter ---

u16 g_next_ctx_id = 0;

// --- Static event data for room member join/leave ---

static OrbisNpMatching2RoomMemberDataInternal s_departed_member_data;
static OrbisNpMatching2RoomMemberUpdateInfo s_departed_update_info;

static OrbisNpMatching2RoomMemberDataInternal s_joined_member_data;
static OrbisNpMatching2RoomMemberUpdateInfo s_joined_update_info;
static std::vector<OrbisNpMatching2RoomMemberBinAttrInternal> s_joined_member_bin_attrs;
static std::vector<std::vector<u8>> s_joined_member_bin_buffers;

void ScheduleRoomEventMemberLeft(OrbisNpMatching2RoomMemberId departed_member_id, u8 reason,
                                 std::chrono::steady_clock::time_point fire_at) {
    if (!g_state.room_event_callback || g_state.ctx.room_id == 0) {
        LOG_WARNING(Lib_NpMatching2, "FireRoomEventMemberLeft: no room event callback or no room");
        return;
    }

    // Set up the departed member data
    std::memset(&s_departed_member_data, 0, sizeof(s_departed_member_data));
    s_departed_member_data.memberId = departed_member_id;
    std::memset(&s_departed_update_info, 0, sizeof(s_departed_update_info));
    s_departed_update_info.roomMemberDataInternal = &s_departed_member_data;
    s_departed_update_info.eventCause = reason;

    // Schedule room event via dispatch thread
    PendingEvent ev{};
    ev.type = PendingEvent::ROOM_EVENT_CB;
    ev.fire_at = fire_at;
    ev.room_id = g_state.ctx.room_id;
    ev.room_event = ORBIS_NP_MATCHING2_ROOM_EVENT_MEMBER_LEFT;
    ev.room_event_data = &s_departed_update_info;
    ScheduleEvent(std::move(ev));

    LOG_INFO(Lib_NpMatching2, "ScheduleRoomEventMemberLeft: member={} reason={} scheduled",
             departed_member_id, reason);
}

void ScheduleRoomEventMemberJoined(const MemberInfo& member,
                                   std::chrono::steady_clock::time_point fire_at) {
    if (!g_state.room_event_callback || g_state.ctx.room_id == 0) {
        LOG_WARNING(Lib_NpMatching2,
                    "ScheduleRoomEventMemberJoined: dropping — room_event_callback={:p} room_id={}",
                    static_cast<void*>(g_state.room_event_callback), g_state.ctx.room_id);
        return;
    }

    // Set up the joined member data (NpId at +0x10, memberId at +0x38)
    std::memset(&s_joined_member_data, 0, sizeof(s_joined_member_data));
    s_joined_member_data.memberId = static_cast<u16>(member.member_id);
    std::strncpy(s_joined_member_data.npId.handle.data, member.online_id.c_str(),
                 ORBIS_NP_ONLINEID_MAX_LENGTH - 1);
    if (member.member_id == g_state.ctx.my_member_id && g_state.ctx.is_room_owner) {
        s_joined_member_data.flagAttr |= ORBIS_NP_MATCHING2_ROOM_MEMBER_FLAG_ATTR_OWNER;
    }

    // Populate member bin attrs if present
    s_joined_member_bin_attrs.clear();
    s_joined_member_bin_buffers.clear();
    if (!member.bin_attrs.empty()) {
        s_joined_member_bin_attrs.resize(member.bin_attrs.size());
        s_joined_member_bin_buffers.resize(member.bin_attrs.size());
        for (size_t i = 0; i < member.bin_attrs.size(); ++i) {
            s_joined_member_bin_buffers[i] = member.bin_attrs[i].data;
            auto& ba = s_joined_member_bin_attrs[i];
            std::memset(&ba, 0, sizeof(ba));
            ba.binAttr.id = member.bin_attrs[i].id;
            ba.binAttr.data = s_joined_member_bin_buffers[i].data();
            ba.binAttr.dataSize = s_joined_member_bin_buffers[i].size();
        }
        s_joined_member_data.roomMemberBinAttrInternal = s_joined_member_bin_attrs.data();
        s_joined_member_data.roomMemberBinAttrInternalNum =
            static_cast<u32>(s_joined_member_bin_attrs.size());
    }

    std::memset(&s_joined_update_info, 0, sizeof(s_joined_update_info));
    s_joined_update_info.roomMemberDataInternal = &s_joined_member_data;
    s_joined_update_info.eventCause = 0; // joined normally

    // Schedule room event via dispatch thread
    PendingEvent ev{};
    ev.type = PendingEvent::ROOM_EVENT_CB;
    ev.fire_at = fire_at;
    ev.room_id = g_state.ctx.room_id;
    ev.room_event = ORBIS_NP_MATCHING2_ROOM_EVENT_MEMBER_JOINED;
    ev.room_event_data = &s_joined_update_info;
    ScheduleEvent(std::move(ev));

    LOG_INFO(Lib_NpMatching2, "ScheduleRoomEventMemberJoined: member={} online_id={} scheduled",
             member.member_id, member.online_id);
}

// --- Room request JSON building ---

std::string BuildCreateJoinRoomRequestJson(const OrbisNpMatching2CreateJoinRoomRequest& req,
                                           OrbisNpMatching2RequestId req_id) {
    std::string data = "{";
    data += "\"req_id\":" + std::to_string(req_id);
    data += ",\"max_slots\":" + std::to_string(std::max<u16>(req.maxSlot, 1));
    data += ",\"team_id\":" + std::to_string(req.teamId);
    data += ",\"world_id\":" + std::to_string(req.worldId);
    data += ",\"lobby_id\":" + std::to_string(req.lobbyId);
    data += ",\"flags\":" + std::to_string(req.flags);
    data += ",\"group_config_count\":" + std::to_string(req.groupConfigs);
    data += ",\"allowed_user_count\":" + std::to_string(req.allowedUsers);
    data += ",\"blocked_user_count\":" + std::to_string(req.blockedUsers);
    data += ",\"internal_bin_attr_count\":" + std::to_string(req.internalBinAttrs);
    data += ",\"external_search_int_attr_count\":" + std::to_string(req.externalSearchIntAttrs);
    data += ",\"external_search_bin_attr_count\":" + std::to_string(req.externalSearchBinAttrs);
    data += ",\"external_bin_attr_count\":" + std::to_string(req.externalBinAttrs);
    data += ",\"member_internal_bin_attr_count\":" + std::to_string(req.memberInternalBinAttrs);

    if (req.signalingParam) {
        data += ",\"signaling\":{";
        data += "\"type\":" + std::to_string(req.signalingParam->type);
        data += ",\"flag\":" + std::to_string(req.signalingParam->flag);
        data += ",\"main_member\":" + std::to_string(req.signalingParam->mainMember);
        data += "}";
    }

    data += ",\"join_group_label_present\":" + std::string(req.joinGroupLabel ? "true" : "false");
    data += ",\"room_password_present\":" + std::string(req.roomPasswd ? "true" : "false");
    data += "}";
    return data;
}

void LogCreateJoinRoomOutboundPayload(const OrbisNpMatching2CreateJoinRoomRequest& req,
                                      OrbisNpMatching2RequestId req_id,
                                      std::string_view request_json) {
    LOG_INFO(Lib_NpMatching2,
             "CreateJoinRoom outbound: reqId={} max_slots={} team_id={} world_id={} lobby_id={} "
             "flags={:#x}",
             req_id, std::max<u16>(req.maxSlot, 1), req.teamId, req.worldId, req.lobbyId,
             req.flags);
    LOG_INFO(Lib_NpMatching2,
             "CreateJoinRoom outbound: group_config_count={} allowed_user_count={} "
             "blocked_user_count={} internal_bin_attr_count={} external_search_int_attr_count={} "
             "external_search_bin_attr_count={} external_bin_attr_count={} "
             "member_internal_bin_attr_count={}",
             req.groupConfigs, req.allowedUsers, req.blockedUsers, req.internalBinAttrs,
             req.externalSearchIntAttrs, req.externalSearchBinAttrs, req.externalBinAttrs,
             req.memberInternalBinAttrs);
    LOG_INFO(Lib_NpMatching2,
             "CreateJoinRoom outbound: room_password_present={} join_group_label_present={} "
             "signaling_present={}",
             req.roomPasswd != nullptr, req.joinGroupLabel != nullptr,
             req.signalingParam != nullptr);
    if (req.signalingParam) {
        LOG_INFO(Lib_NpMatching2,
                 "CreateJoinRoom outbound: signaling.type={} signaling.flag={:#x} "
                 "signaling.main_member={}",
                 req.signalingParam->type, req.signalingParam->flag,
                 req.signalingParam->mainMember);
    }
    LOG_INFO(Lib_NpMatching2, "CreateJoinRoom outbound: cmd=CreateRoom payload={}", request_json);
}

// --- Async world info thread ---

PS4_SYSV_ABI void* GetWorldInfoListThreadFunc(void* arg) {
    auto* a = static_cast<AsyncGetWorldInfoArgs*>(arg);

    // Small delay to ensure the API call returns first
    std::this_thread::sleep_for(std::chrono::milliseconds(100));

    LOG_INFO(Lib_NpMatching2, "GetWorldInfoList async: firing callback with 1 world");

    // Construct world info response with one world (worldId=1).
    // Struct is 0x28 bytes; game reads u32 fields at +0x08, +0x18, +0x1c.
    // Stored in g_state so the game can safely reference the data after the callback returns.
    {
        std::lock_guard<std::mutex> lock(g_state.mutex);
        std::memset(g_state.world_info_worlds, 0, sizeof(g_state.world_info_worlds));
        auto& w0 = g_state.world_info_worlds[0];
        auto& w1 = g_state.world_info_worlds[1];
        w0.next = &w1;
        w0.worldId = 1;
        w0.numOfLobby = 1;
        w0.curNumOfTotalLobby = 1;
        w0.maxNumOfTotalLobby = 10;
        w0.curNumOfRoom = 0;
        w0.maxNumOfRoom = 10;
        w1.next = nullptr;
        w1.worldId = 2;
        w1.numOfLobby = 1;
        w1.curNumOfTotalLobby = 1;
        w1.maxNumOfTotalLobby = 10;
        w1.curNumOfRoom = 0;
        w1.maxNumOfRoom = 10;
        g_state.world_info_resp.world = &w0;
        g_state.world_info_resp.worldNum = 2;
    }

    LOG_INFO(Lib_NpMatching2,
             "GetWorldInfoList async: response worldNum={} world[0]: worldId={} "
             "curNumOfTotalLobby={} maxNumOfTotalLobby={} curNumOfRoom={} maxNumOfRoom={} "
             "world@{:p} resp@{:p} sizeof(World)={}",
             g_state.world_info_resp.worldNum, g_state.world_info_worlds[0].worldId,
             g_state.world_info_worlds[0].curNumOfTotalLobby,
             g_state.world_info_worlds[0].maxNumOfTotalLobby,
             g_state.world_info_worlds[0].curNumOfRoom, g_state.world_info_worlds[0].maxNumOfRoom,
             static_cast<void*>(g_state.world_info_resp.world), (void*)&g_state.world_info_resp,
             sizeof(g_state.world_info_worlds[0]));

    if (a->callback) {
        // Event code 2 = GetWorldInfoList response
        // Signature: callback(ctxId, reqId, event, errorCode, data, arg)
        LOG_INFO(Lib_NpMatching2,
                 "GetWorldInfoList async: INVOKE cb={:p} ctxId={} reqId={} event=2 "
                 "errorCode=0 data={:p} arg={:p}",
                 static_cast<void*>(a->callback), a->ctx_id, a->req_id,
                 (void*)&g_state.world_info_resp, a->callback_arg);
        a->callback(a->ctx_id, a->req_id, 2, 0, &g_state.world_info_resp, a->callback_arg);
        LOG_INFO(Lib_NpMatching2, "GetWorldInfoList async: callback returned");
    }

    delete a;
    return nullptr;
}

} // namespace Libraries::Np::NpMatching2
