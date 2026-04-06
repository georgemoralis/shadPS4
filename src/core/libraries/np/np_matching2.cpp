// SPDX-FileCopyrightText: Copyright 2025 shadPS4 Emulator Project
// SPDX-License-Identifier: GPL-2.0-or-later

#include <chrono>
#include <cstring>
#include <mutex>
#include <string>
#include <thread>
#include <vector>

#include "common/config.h"
#include "common/logging/log.h"
#include "core/libraries/error_codes.h"
#include "core/libraries/kernel/threads.h"
#include "core/libraries/libs.h"
#include "core/libraries/network/net.h"
#include "core/libraries/network/sockets.h"
#include "core/libraries/np/np_manager.h"
#include "core/libraries/np/np_matching2.h"
#include "core/libraries/np/np_matching2_error.h"
#include "core/libraries/np/np_matching2_internal.h"
#include "core/tls.h"

namespace {

long long NowMs() {
    return std::chrono::duration_cast<std::chrono::milliseconds>(
               std::chrono::steady_clock::now().time_since_epoch())
        .count();
}

// Convert dotted IP string to in_addr (network byte order)
u32 IpStringToAddr(const std::string& ip) {
    u32 a, b, c, d;
    if (sscanf(ip.c_str(), "%u.%u.%u.%u", &a, &b, &c, &d) != 4)
        return 0;
    return (a) | (b << 8) | (c << 16) | (d << 24);
}

} // anonymous namespace

namespace Libraries::Np::NpMatching2 {

// --- Core lifecycle ---

// sceNpMatching2Initialize validates the param struct then initialises the NP
// heap, net pool, SSL, HTTP, and signaling subsystems. In HLE we skip the
// native subsystem allocation but preserve all observable error semantics.
//
// Param struct layout (OrbisNpMatching2InitParam, 0x28 or 0x30 bytes):
//   +0x00 (u64): heap size override (0 = auto)
//   +0x08 (u64): passed through to signaling init (thread affinity / config)
//   +0x10 (s32): max connections (0 = default 700)
//   +0x14 (u32): pad
//   +0x18 (u64): thread stack size (0 = default 0x8000)
//   +0x20 (u64): struct size — must be 0x28 or 0x30
//   +0x28 (u64): SSL buffer size (only if size==0x30; 0=0x30000; must be<0x30000)
s32 PS4_SYSV_ABI sceNpMatching2Initialize(const OrbisNpMatching2InitParam* param) {
    LOG_INFO(Lib_NpMatching2, "called param={}", static_cast<const void*>(param));

    if (g_state.initialized) {
        LOG_ERROR(Lib_NpMatching2, "already initialized");
        return ORBIS_NP_MATCHING2_ERROR_ALREADY_INITIALIZED;
    }

    if (!param) {
        LOG_ERROR(Lib_NpMatching2, "null param");
        return ORBIS_NP_MATCHING2_ERROR_INVALID_ARGUMENT;
    }

    // Size field must be exactly 0x28 or 0x30.
    if (param->size != 0x28 && param->size != 0x30) {
        LOG_ERROR(Lib_NpMatching2, "invalid param size: {:#x}", param->size);
        return ORBIS_NP_MATCHING2_ERROR_INVALID_ARGUMENT;
    }

    // When the extended struct is present, sslBufSize must be < 0x30000 (if non-zero).
    if (param->size == 0x30 && param->sslBufSize != 0 && param->sslBufSize >= 0x30000) {
        LOG_ERROR(Lib_NpMatching2, "sslBufSize out of range: {:#x}", param->sslBufSize);
        return ORBIS_NP_MATCHING2_ERROR_INVALID_ARGUMENT;
    }

    // Log the meaningful fields; native subsystem allocation is skipped in HLE.
    const s32 maxConnections = param->maxConnections != 0 ? param->maxConnections : 700;
    const u64 stackSize = param->threadStackSize != 0 ? param->threadStackSize : 0x8000;
    LOG_INFO(Lib_NpMatching2, "maxConnections={} stackSize={:#x} heapSize={:#x}", maxConnections,
             stackSize, param->heapSize);

    g_state.signaling_port = Net::GetP2PConfiguredPort();
    g_state.signaling_addr = Net::GetP2PConfiguredAddr();
    g_state.initialized = true;
    return ORBIS_OK;
}

s32 PS4_SYSV_ABI sceNpMatching2Terminate() {
    LOG_INFO(Lib_NpMatching2, "called");
    g_state.StopPollThread();
    g_state.initialized = false;
    g_state.FreeCallbackData();
    g_state.peers.clear();
    // Reset context ID counter so re-initialisation starts from a clean state.
    g_next_ctx_id = 0;
    return ORBIS_OK;
}

s32 PS4_SYSV_ABI sceNpMatching2SetExtraInitParam(void* param) {
    LOG_INFO(Lib_NpMatching2, "called");
    if (!g_state.initialized) {
        return ORBIS_NP_MATCHING2_ERROR_NOT_INITIALIZED;
    }
    if (param) {
        auto* p = reinterpret_cast<OrbisNpMatching2ExtraInitParam*>(param);
        g_state.signaling_port = p->signalingPort;
        LOG_INFO(Lib_NpMatching2, "signaling port set to {}", g_state.signaling_port);
    }
    return ORBIS_OK;
}

// --- Context management ---

// sceNpMatching2CreateContextInternal is the actual allocating function.
// sceNpMatching2CreateContext (the public wrapper) validates the param struct
// size, extracts NpId* and service label into a smaller internal struct, then
// delegates here. Both are exported and may be called directly by the game.
//
// Internal param struct layout (16 bytes, built by the public wrapper):
//   +0x00 (u64): NpId* — pointer to the caller's OrbisNpId
//   +0x08 (u64): service label — u32 zero-extended to u64
s32 PS4_SYSV_ABI sceNpMatching2CreateContextInternal(const void* reqParam,
                                                     OrbisNpMatching2ContextId* ctxId) {
    if (!g_state.initialized) {
        LOG_ERROR(Lib_NpMatching2, "not initialized");
        return ORBIS_NP_MATCHING2_ERROR_NOT_INITIALIZED;
    }
    if (!reqParam || !ctxId) {
        LOG_ERROR(Lib_NpMatching2, "null param or ctxId");
        return ORBIS_NP_MATCHING2_ERROR_INVALID_ARGUMENT;
    }

    const auto* p = static_cast<const u64*>(reqParam);
    const auto* npIdPtr = reinterpret_cast<const OrbisNpId*>(p[0]);
    const u32 serviceLabel = static_cast<u32>(p[1]);

    // Allocate context ID, cycling 1-8 to match the 8-slot pool in the real impl.
    g_next_ctx_id = static_cast<u16>((g_next_ctx_id % 8) + 1);

    g_state.ctx.ctx_id = g_next_ctx_id;
    g_state.ctx.started = false;
    g_state.ctx.service_label = static_cast<u16>(serviceLabel);

    if (npIdPtr) {
        g_state.ctx.online_id = std::string(
            npIdPtr->handle.data, strnlen(npIdPtr->handle.data, ORBIS_NP_ONLINEID_MAX_LENGTH));
    }
    if (g_state.ctx.online_id.empty()) {
        g_state.ctx.online_id = "shadPS4_player";
    }

    *ctxId = g_state.ctx.ctx_id;

    LOG_INFO(Lib_NpMatching2, "context created: id={} online_id={} serviceLabel={:#x}",
             g_state.ctx.ctx_id, g_state.ctx.online_id, serviceLabel);
    return ORBIS_OK;
}

// Public wrapper around CreateContextInternal.
// Param struct layout (0x28 bytes, from RE of sceNpMatching2CreateContext):
//   +0x00 (u64): NpId* — pointer to OrbisNpId
//   +0x08 (u64): padding / unknown
//   +0x10 (u64): padding / unknown
//   +0x18 (u32): service label
//   +0x1C (u32): padding
//   +0x20 (u64): size — must equal 0x28
s32 PS4_SYSV_ABI sceNpMatching2CreateContext(const void* reqParam,
                                             OrbisNpMatching2ContextId* ctxId) {
    LOG_INFO(Lib_NpMatching2, "called reqParam={} ctxId={}", reqParam, static_cast<void*>(ctxId));

    if (!reqParam || !ctxId) {
        return ORBIS_NP_MATCHING2_ERROR_INVALID_ARGUMENT;
    }

    // Size field at byte offset 0x20 must equal 0x28.
    const auto* p = static_cast<const u64*>(reqParam);
    if (p[4] != 0x28) {
        LOG_ERROR(Lib_NpMatching2, "invalid param size: expected 0x28, got {:#x}", p[4]);
        return ORBIS_NP_MATCHING2_ERROR_INVALID_ARGUMENT;
    }

    // Extract NpId* (offset 0x00) and service label (offset 0x18) into the
    // smaller internal struct expected by CreateContextInternal.
    const u64 internal[2] = {
        p[0],                                                   // NpId*
        static_cast<u64>(*reinterpret_cast<const u32*>(p + 3)), // service label at +0x18
    };

    return sceNpMatching2CreateContextInternal(internal, ctxId);
}

// CreateContextA takes a compact account/user-id request and resolves it to
// an OrbisNpId before forwarding to the shared internal creation path.
// Layout observed from the title and libSceNpMatching2:
//   +0x00 (u32): user/account id
//   +0x04 (u32): service label
//   +0x08 (u64): size — must equal 0x10
s32 PS4_SYSV_ABI sceNpMatching2CreateContextA(const void* reqParam,
                                              OrbisNpMatching2ContextId* ctxId) {
    LOG_INFO(Lib_NpMatching2, "called (CreateContextA)");

    if (!reqParam || !ctxId) {
        return ORBIS_NP_MATCHING2_ERROR_INVALID_ARGUMENT;
    }

    const auto* raw = static_cast<const u8*>(reqParam);
    u32 userId;
    std::memcpy(&userId, raw + 0x00, sizeof(u32));
    u32 serviceLabel;
    std::memcpy(&serviceLabel, raw + 0x04, sizeof(u32));
    u64 size;
    std::memcpy(&size, raw + 0x08, sizeof(u64));

    LOG_INFO(Lib_NpMatching2, "CreateContextA: userId={} serviceLabel={} size={:#x}", userId,
             serviceLabel, size);

    if (size != 0x10) {
        LOG_ERROR(Lib_NpMatching2, "CreateContextA: invalid param size: expected 0x10, got {:#x}",
                  size);
        return ORBIS_NP_MATCHING2_ERROR_INVALID_ARGUMENT;
    }

    struct IntNpIdResult {
        OrbisNpId np_id;
        u32 metadata;
    } np_id_result{};

    const s32 ret = NpManager::sceNpManagerIntGetNpId(userId, &np_id_result);
    if (ret < 0) {
        LOG_ERROR(Lib_NpMatching2, "CreateContextA: failed to resolve userId {} to NpId: {:#x}",
                  userId, ret);
        return ret;
    }

    struct InternalCreateContextAParam {
        OrbisNpId* np_id;
        u32 service_label;
        u32 type;
    } internal = {
        .np_id = &np_id_result.np_id,
        .service_label = serviceLabel,
        .type = 2,
    };

    return sceNpMatching2CreateContextInternal(&internal, ctxId);
}

// ContextStartCallbackThread removed — context callback is now scheduled
// via the EventDispatchThread (PendingEvent::CONTEXT_CB at T+200ms).

s32 PS4_SYSV_ABI sceNpMatching2ContextStart(OrbisNpMatching2ContextId ctxId) {
    LOG_INFO(Lib_NpMatching2, "called ctxId={}", ctxId);

    g_state.ctx.started = true;
    g_state.ctx.server_id = 1;

    // Sync addr/port from config via p2p_sockets (which already includes config.h).
    // This keeps np_matching2 and the shared UDP transport in agreement without
    // requiring config.h here.
    g_state.signaling_port = Net::GetP2PConfiguredPort();
    g_state.signaling_addr = Net::GetP2PConfiguredAddr();

    u32 local_addr = IpStringToAddr(g_state.signaling_addr);
    LOG_INFO(Lib_NpMatching2, "ContextStart: signaling_addr='{}' ({:#x}) signaling_port={}",
             g_state.signaling_addr, local_addr, g_state.signaling_port);

    // Start the event dispatch thread (long-lived, processes all queued events).
    // Must be started before scheduling any events.
    if (!g_state.dispatch_running) {
        g_state.dispatch_running = true;
        int dret = Kernel::posix_pthread_create(&g_state.dispatch_thread, nullptr,
                                                EventDispatchThreadFunc, nullptr);
        if (dret != 0) {
            LOG_ERROR(Lib_NpMatching2, "Failed to create EventDispatch thread: {}", dret);
            g_state.dispatch_running = false;
        } else {
            LOG_INFO(Lib_NpMatching2, "EventDispatch thread created");
        }
    }

    // Schedule context callback via event queue (fires at T+200ms).
    // This notifies the caller that the context has started, allowing
    // subsequent callback registrations to proceed.
    {
        PendingEvent ev{};
        ev.type = PendingEvent::CONTEXT_CB;
        ev.fire_at = std::chrono::steady_clock::now() + std::chrono::milliseconds(200);
        ev.ctx_event = ORBIS_NP_MATCHING2_CONTEXT_EVENT_STARTED;
        ev.ctx_event_cause = 0;
        ev.error_code = 0;
        ScheduleEvent(std::move(ev));
        LOG_INFO(Lib_NpMatching2,
                 "context callback scheduled (event=CONTEXT_EVENT_STARTED, T+200ms)");
    }
    // Connect to MM server (persistent TCP, push notifications replace HTTP polling)
    ConfigureMmNotificationHandlers({
        .member_joined = OnMmMemberJoined,
        .member_left = OnMmMemberLeft,
        .room_destroyed = OnMmRoomDestroyed,
        .signaling_helper = OnMmSignalingHelper,
        .room_data_internal_updated = OnMmRoomDataInternalUpdated,
    });
    if (!IsMmClientRunning()) {
        StartMmClient(g_state.ctx.online_id, g_state.signaling_addr, g_state.signaling_port);
    }

    return ORBIS_OK;
}

s32 PS4_SYSV_ABI sceNpMatching2ContextStop(OrbisNpMatching2ContextId ctxId) {
    LOG_INFO(Lib_NpMatching2, "called ctxId={}", ctxId);

    if (g_state.context_callback) {
        PendingEvent ev{};
        ev.type = PendingEvent::CONTEXT_CB;
        ev.fire_at = std::chrono::steady_clock::now();
        ev.ctx_event = ORBIS_NP_MATCHING2_CONTEXT_EVENT_STARTED;
        ev.ctx_event_cause = 11;
        ev.error_code = 0;
        ScheduleEvent(std::move(ev));
    }

    g_state.StopPollThread();
    g_state.ctx.started = false;
    return ORBIS_OK;
}

s32 PS4_SYSV_ABI sceNpMatching2DestroyContext(OrbisNpMatching2ContextId ctxId) {
    LOG_INFO(Lib_NpMatching2, "called ctxId={}", ctxId);
    g_state.StopPollThread();
    g_state.ctx.started = false;
    g_state.ctx.ctx_id = 0;
    return ORBIS_OK;
}

s32 PS4_SYSV_ABI sceNpMatching2AbortContextStart(OrbisNpMatching2ContextId ctxId) {
    LOG_INFO(Lib_NpMatching2, "called ctxId={}", ctxId);
    return ORBIS_OK;
}

// --- Callback registration ---

s32 PS4_SYSV_ABI sceNpMatching2SetDefaultRequestOptParam(OrbisNpMatching2ContextId ctxId,
                                                         void* optParam) {
    LOG_INFO(Lib_NpMatching2, "called ctxId={}", ctxId);
    if (optParam) {
        auto* opt = reinterpret_cast<OrbisNpMatching2RequestOptParam*>(optParam);
        g_state.default_request_callback = opt->callback;
        g_state.default_request_callback_arg = opt->arg;
        LOG_INFO(Lib_NpMatching2,
                 "SetDefaultRequestOptParam: callback={:p} arg={:p} timeout={} appId={}",
                 static_cast<void*>(opt->callback), opt->arg, opt->timeout, opt->appId);
    }
    SendRegisterHandlers();
    return ORBIS_OK;
}

s32 PS4_SYSV_ABI sceNpMatching2RegisterContextCallback(void* cbFunc, void* cbFuncArg) {
    LOG_INFO(Lib_NpMatching2, "called cbFunc={} cbFuncArg={}", cbFunc, cbFuncArg);
    if (!g_state.initialized) {
        return ORBIS_NP_MATCHING2_ERROR_NOT_INITIALIZED;
    }
    g_state.context_callback = reinterpret_cast<OrbisNpMatching2ContextCallback>(cbFunc);
    g_state.context_callback_arg = cbFuncArg;
    SendRegisterHandlers();
    return ORBIS_OK;
}

s32 PS4_SYSV_ABI sceNpMatching2RegisterRoomEventCallback(OrbisNpMatching2ContextId ctxId,
                                                         void* cbFunc, void* cbFuncArg) {
    LOG_INFO(Lib_NpMatching2, "called ctxId={} cbFunc={:p} cbFuncArg={:p}", ctxId, cbFunc,
             cbFuncArg);
    g_state.room_event_callback = reinterpret_cast<OrbisNpMatching2RoomEventCallback>(cbFunc);
    g_state.room_event_callback_arg = cbFuncArg;
    SendRegisterHandlers();
    return ORBIS_OK;
}

s32 PS4_SYSV_ABI sceNpMatching2RegisterSignalingCallback(OrbisNpMatching2ContextId ctxId,
                                                         void* cbFunc, void* cbFuncArg) {
    LOG_INFO(Lib_NpMatching2, "called ctxId={} cbFunc={:p} cbFuncArg={:p}", ctxId, cbFunc,
             cbFuncArg);
    g_state.signaling_callback = reinterpret_cast<OrbisNpMatching2SignalingCallback>(cbFunc);
    g_state.signaling_callback_arg = cbFuncArg;
    SendRegisterHandlers();
    return ORBIS_OK;
}

s32 PS4_SYSV_ABI sceNpMatching2RegisterLobbyEventCallback(OrbisNpMatching2ContextId ctxId,
                                                          void* cbFunc, void* cbFuncArg) {
    LOG_INFO(Lib_NpMatching2, "called ctxId={}", ctxId);
    g_state.lobby_event_callback = reinterpret_cast<OrbisNpMatching2LobbyEventCallback>(cbFunc);
    g_state.lobby_event_callback_arg = cbFuncArg;
    SendRegisterHandlers();
    return ORBIS_OK;
}

s32 PS4_SYSV_ABI sceNpMatching2RegisterLobbyMessageCallback(OrbisNpMatching2ContextId ctxId,
                                                            void* cbFunc, void* cbFuncArg) {
    LOG_INFO(Lib_NpMatching2, "called ctxId={}", ctxId);
    g_state.lobby_message_callback = reinterpret_cast<OrbisNpMatching2LobbyMessageCallback>(cbFunc);
    g_state.lobby_message_callback_arg = cbFuncArg;
    SendRegisterHandlers();
    return ORBIS_OK;
}

s32 PS4_SYSV_ABI sceNpMatching2RegisterRoomMessageCallback(OrbisNpMatching2ContextId ctxId,
                                                           void* cbFunc, void* cbFuncArg) {
    LOG_INFO(Lib_NpMatching2, "called ctxId={}", ctxId);
    g_state.room_message_callback = reinterpret_cast<OrbisNpMatching2RoomMessageCallback>(cbFunc);
    g_state.room_message_callback_arg = cbFuncArg;
    SendRegisterHandlers();
    return ORBIS_OK;
}

s32 PS4_SYSV_ABI sceNpMatching2RegisterManualUdpSignalingCallback(OrbisNpMatching2ContextId ctxId,
                                                                  void* cbFunc, void* cbFuncArg) {
    LOG_INFO(Lib_NpMatching2, "(STUBBED) called ctxId={}", ctxId);
    return ORBIS_OK;
}

// --- Room operations ---

s32 PS4_SYSV_ABI sceNpMatching2CreateJoinRoom(OrbisNpMatching2ContextId ctxId, void* reqParam,
                                              void* optParam, s32* reqId) {
    LOG_INFO(Lib_NpMatching2, "called ctxId={}", ctxId);
    if (!g_state.initialized) {
        return ORBIS_NP_MATCHING2_ERROR_NOT_INITIALIZED;
    }
    if (!reqParam) {
        return ORBIS_NP_MATCHING2_ERROR_INVALID_ARGUMENT;
    }
    if (!g_state.ctx.started) {
        return ORBIS_NP_MATCHING2_ERROR_CONTEXT_NOT_STARTED;
    }

    LOG_INFO(Lib_NpMatching2, "CreateJoinRoom: reqParam={:p} optParam={:p}", reqParam, optParam);
    LogRawBuffer("CreateJoinRoom reqParam", reqParam, sizeof(OrbisNpMatching2CreateJoinRoomRequest),
                 sizeof(OrbisNpMatching2CreateJoinRoomRequest));
    LogCreateJoinRoomWordView(reqParam, sizeof(OrbisNpMatching2CreateJoinRoomRequest));
    LogCreateJoinRoomDecoded(reqParam);
    if (optParam) {
        LogRawBuffer("CreateJoinRoom optParam", optParam, sizeof(OrbisNpMatching2RequestOptParam),
                     sizeof(OrbisNpMatching2RequestOptParam));
    }

    // Extract callback from optParam
    OrbisNpMatching2RequestCallback callback = nullptr;
    void* callback_arg = nullptr;
    if (optParam) {
        auto* opt = reinterpret_cast<OrbisNpMatching2RequestOptParam*>(optParam);
        callback = opt->callback;
        callback_arg = opt->arg;
        LOG_INFO(Lib_NpMatching2,
                 "CreateJoinRoom: optParam callback={:p} arg={:p} timeout={} appId={}",
                 static_cast<void*>(opt->callback), opt->arg, opt->timeout, opt->appId);
    }
    if (!callback) {
        callback = g_state.default_request_callback;
        callback_arg = g_state.default_request_callback_arg;
    }

    g_state.per_request_callback = callback;
    g_state.per_request_callback_arg = callback_arg;
    SendRegisterHandlers();

    OrbisNpMatching2RequestId rid = g_state.next_request_id++;
    if (reqId) {
        *reqId = static_cast<s32>(rid);
    }

    LOG_INFO(Lib_NpMatching2, "CreateJoinRoom: reqId={} callback={:p} arg={:p}", rid,
             static_cast<void*>(callback), callback_arg);

    const auto* req = reinterpret_cast<const OrbisNpMatching2CreateJoinRoomRequest*>(reqParam);
    {
        std::lock_guard<std::mutex> lock(g_state.mutex);
        CacheCreateJoinRoomRequest(*req, rid);
    }
    LOG_INFO(Lib_NpMatching2,
             "CreateJoinRoom: cached request reqId={} worldId={} lobbyId={} maxSlot={} flags={:#x} "
             "passwdSlotMask={:#x} groupConfigs={} internalBinAttrs={} memberInternalBinAttrs={}",
             rid, req->worldId, req->lobbyId, req->maxSlot, req->flags,
             req->passwdSlotMask ? *req->passwdSlotMask : 0, req->groupConfigs,
             req->internalBinAttrs, req->memberInternalBinAttrs);
    BinaryWriter w;
    w.AppendU32LE(rid);                            // reqId
    w.AppendU16LE(std::max<u16>(req->maxSlot, 1)); // maxSlots
    w.AppendU16LE(req->teamId);                    // teamId
    w.AppendU16LE(req->worldId);                   // worldId
    w.AppendU16LE(req->lobbyId);                   // lobbyId
    w.AppendU32LE(static_cast<u32>(req->flags));   // flags
    w.AppendU16LE(req->groupConfigs);              // groupConfigCount
    w.AppendU16LE(req->allowedUsers);              // allowedUserCount
    w.AppendU16LE(req->blockedUsers);              // blockedUserCount
    w.AppendU16LE(req->internalBinAttrs);          // internalBinAttrCount
    w.AppendU16LE(req->externalSearchIntAttrs);    // externalSearchIntAttrCount
    for (u64 i = 0; i < req->externalSearchIntAttrs; ++i) {
        const auto& attr = req->externalSearchIntAttr[i];
        w.AppendU16LE(attr.id);
        w.AppendU32LE(attr.attr);
    }
    w.AppendU16LE(req->externalSearchBinAttrs); // externalSearchBinAttrCount
    for (u64 i = 0; i < req->externalSearchBinAttrs; ++i) {
        const auto& attr = req->externalSearchBinAttr[i];
        w.AppendU16LE(attr.id);
        w.AppendU32LE(static_cast<u32>(attr.dataSize));
        if (attr.data && attr.dataSize > 0)
            w.AppendBytes(attr.data, static_cast<size_t>(attr.dataSize));
    }
    w.AppendU16LE(req->externalBinAttrs); // externalBinAttrCount
    for (u64 i = 0; i < req->externalBinAttrs; ++i) {
        const auto& attr = req->externalBinAttr[i];
        w.AppendU16LE(attr.id);
        w.AppendU32LE(static_cast<u32>(attr.dataSize));
        if (attr.data && attr.dataSize > 0)
            w.AppendBytes(attr.data, static_cast<size_t>(attr.dataSize));
    }
    w.AppendU16LE(req->memberInternalBinAttrs); // memberInternalBinAttrCount
    // Send actual member bin attr data for the creating member
    for (u64 i = 0; i < req->memberInternalBinAttrs; ++i) {
        const auto& mattr = req->memberInternalBinAttr[i];
        w.AppendU16LE(mattr.id);
        w.AppendU32LE(static_cast<u32>(mattr.dataSize));
        if (mattr.data && mattr.dataSize > 0)
            w.AppendBytes(mattr.data, static_cast<size_t>(mattr.dataSize));
    }
    w.AppendU8(req->joinGroupLabel ? 1 : 0);                         // joinGroupLabelPresent
    w.AppendU8(req->roomPasswd ? 1 : 0);                             // roomPasswordPresent
    w.AppendU8(req->signalingParam ? req->signalingParam->type : 0); // signalingType
    w.AppendU8(req->signalingParam ? req->signalingParam->flag : 0); // signalingFlag
    w.AppendU16LE(req->signalingParam ? req->signalingParam->mainMember : 0); // signalingMainMember

    MmSendFireAndForget(ShadnetCommandType::CreateRoom, w);
    LOG_INFO(Lib_NpMatching2, "CreateJoinRoom: fire-and-forget sent reqId={}", rid);
    return ORBIS_OK;
}

s32 PS4_SYSV_ABI sceNpMatching2CreateJoinRoomA(OrbisNpMatching2ContextId ctxId, void* reqParam,
                                               void* optParam, s32* reqId) {
    LOG_INFO(Lib_NpMatching2, "called (forwarding to CreateJoinRoom)");
    return sceNpMatching2CreateJoinRoom(ctxId, reqParam, optParam, reqId);
}

s32 PS4_SYSV_ABI sceNpMatching2JoinRoom(OrbisNpMatching2ContextId ctxId, void* reqParam,
                                        void* optParam, s32* reqId) {
    LOG_INFO(Lib_NpMatching2, "called ctxId={}", ctxId);
    if (!g_state.initialized) {
        return ORBIS_NP_MATCHING2_ERROR_NOT_INITIALIZED;
    }
    if (!reqParam) {
        return ORBIS_NP_MATCHING2_ERROR_INVALID_ARGUMENT;
    }
    if (!g_state.ctx.started) {
        return ORBIS_NP_MATCHING2_ERROR_CONTEXT_NOT_STARTED;
    }

    OrbisNpMatching2RequestCallback callback = nullptr;
    void* callback_arg = nullptr;
    if (optParam) {
        auto* opt = reinterpret_cast<OrbisNpMatching2RequestOptParam*>(optParam);
        callback = opt->callback;
        callback_arg = opt->arg;
        LOG_INFO(Lib_NpMatching2, "JoinRoom: optParam callback={:p} arg={:p} timeout={} appId={}",
                 static_cast<void*>(opt->callback), opt->arg, opt->timeout, opt->appId);
    }
    if (!callback) {
        callback = g_state.default_request_callback;
        callback_arg = g_state.default_request_callback_arg;
    }

    g_state.per_request_callback = callback;
    g_state.per_request_callback_arg = callback_arg;
    SendRegisterHandlers();

    const auto* req = reinterpret_cast<const OrbisNpMatching2JoinRoomRequest*>(reqParam);
    const u64 target_room_id = req->roomId;
    LOG_INFO(Lib_NpMatching2, "JoinRoom: roomId={} passwordPtr={:p}", req->roomId,
             static_cast<void*>(req->roomPasswd));
    if (req->roomPasswd) {
        LOG_INFO(Lib_NpMatching2,
                 "JoinRoom: roomPasswd bytes: {:02x} {:02x} {:02x} {:02x} "
                 "{:02x} {:02x} {:02x} {:02x}",
                 req->roomPasswd->data[0], req->roomPasswd->data[1], req->roomPasswd->data[2],
                 req->roomPasswd->data[3], req->roomPasswd->data[4], req->roomPasswd->data[5],
                 req->roomPasswd->data[6], req->roomPasswd->data[7]);
    }

    OrbisNpMatching2RequestId rid = g_state.next_request_id++;
    if (reqId) {
        *reqId = static_cast<s32>(rid);
    }

    LOG_INFO(Lib_NpMatching2, "JoinRoom: reqId={} targetRoomId={}", rid, target_room_id);

    const u16 join_member_bin_attr_count = static_cast<u16>(req->roomMemberBinInternalAttrNum);
    LOG_INFO(Lib_NpMatching2, "JoinRoom: memberBinAttrCount={}", join_member_bin_attr_count);

    BinaryWriter joinW;
    joinW.AppendU64LE(target_room_id);             // roomId
    joinW.AppendU32LE(rid);                        // reqId
    joinW.AppendU16LE(0);                          // teamId
    joinW.AppendU32LE(0);                          // flags
    joinW.AppendU16LE(0);                          // blockedUserCount
    joinW.AppendU16LE(join_member_bin_attr_count); // roomMemberBinInternalAttrCount
    for (u16 i = 0; i < join_member_bin_attr_count; ++i) {
        const auto& attr = req->roomMemberBinInternalAttr[i];
        joinW.AppendU16LE(attr.id);
        joinW.AppendU32LE(static_cast<u32>(attr.dataSize));
        if (attr.data && attr.dataSize > 0)
            joinW.AppendBytes(attr.data, static_cast<size_t>(attr.dataSize));
    }
    joinW.AppendU8(0); // roomPasswordPresent
    joinW.AppendU8(0); // joinGroupLabelPresent
    MmSendFireAndForget(ShadnetCommandType::JoinRoom, joinW);
    LOG_INFO(Lib_NpMatching2, "JoinRoom: fire-and-forget sent reqId={}", rid);
    return ORBIS_OK;
}

s32 PS4_SYSV_ABI sceNpMatching2JoinRoomA(OrbisNpMatching2ContextId ctxId, void* reqParam,
                                         void* optParam, s32* reqId) {
    LOG_INFO(Lib_NpMatching2, "called (forwarding to JoinRoom)");
    return sceNpMatching2JoinRoom(ctxId, reqParam, optParam, reqId);
}

s32 PS4_SYSV_ABI sceNpMatching2LeaveRoom(OrbisNpMatching2ContextId ctxId, void* reqParam,
                                         void* optParam, s32* reqId) {
    LOG_INFO(Lib_NpMatching2, "called ctxId={}", ctxId);
    // --- Hex dump for cross-title RE ---
    if (reqParam) {
        auto* req = reinterpret_cast<OrbisNpMatching2LeaveRoomRequest*>(reqParam);
        LOG_INFO(Lib_NpMatching2,
                 "LeaveRoom: roomId={} optData.len={} optData bytes: "
                 "{:02x}{:02x}{:02x}{:02x}{:02x}{:02x}{:02x}{:02x}"
                 "{:02x}{:02x}{:02x}{:02x}{:02x}{:02x}{:02x}{:02x}",
                 req->roomId, req->optData.len, req->optData.data[0], req->optData.data[1],
                 req->optData.data[2], req->optData.data[3], req->optData.data[4],
                 req->optData.data[5], req->optData.data[6], req->optData.data[7],
                 req->optData.data[8], req->optData.data[9], req->optData.data[10],
                 req->optData.data[11], req->optData.data[12], req->optData.data[13],
                 req->optData.data[14], req->optData.data[15]);
    }
    if (!g_state.initialized) {
        return ORBIS_NP_MATCHING2_ERROR_NOT_INITIALIZED;
    }
    if (!g_state.ctx.started) {
        return ORBIS_NP_MATCHING2_ERROR_CONTEXT_NOT_STARTED;
    }

    OrbisNpMatching2RequestId rid = g_state.next_request_id++;
    if (reqId)
        *reqId = static_cast<s32>(rid);

    // Extract callback from optParam (same pattern as CreateJoinRoom/JoinRoom)
    OrbisNpMatching2RequestCallback callback = nullptr;
    void* callback_arg = nullptr;
    if (optParam) {
        auto* opt = reinterpret_cast<OrbisNpMatching2RequestOptParam*>(optParam);
        callback = opt->callback;
        callback_arg = opt->arg;
        LOG_INFO(Lib_NpMatching2, "LeaveRoom: optParam callback={} arg={}",
                 static_cast<void*>(opt->callback), opt->arg);
    }
    if (!callback) {
        callback = g_state.default_request_callback;
        callback_arg = g_state.default_request_callback_arg;
    }

    // Store callback for when server pushes RequestEvent(0x103)
    g_state.per_request_callback = callback;
    g_state.per_request_callback_arg = callback_arg;
    SendRegisterHandlers();

    LOG_INFO(Lib_NpMatching2, "t={} LeaveRoom: clearing session state, reqId={}", NowMs(), rid);

    // Notify server (fire-and-forget; server will push RequestEvent(0x103) back)
    if (g_state.ctx.room_id != 0) {
        BinaryWriter leaveW;
        leaveW.AppendU64LE(g_state.ctx.room_id); // roomId
        leaveW.AppendU32LE(rid);                 // reqId
        MmSendFireAndForget(ShadnetCommandType::LeaveRoom, leaveW);
    }

    // Clear HLE state immediately (game-side cleanup happens in the 0x103 callback)
    g_state.ctx.room_id = 0;
    g_state.ctx.is_room_owner = false;
    g_state.peers.clear();

    return ORBIS_OK;
}

s32 PS4_SYSV_ABI sceNpMatching2SearchRoom(OrbisNpMatching2ContextId ctxId, void* reqParam,
                                          void* optParam, s32* reqId) {
    LOG_INFO(Lib_NpMatching2, "called ctxId={} reqParam={:p} optParam={:p}", ctxId, reqParam,
             optParam);
    if (!g_state.initialized) {
        return ORBIS_NP_MATCHING2_ERROR_NOT_INITIALIZED;
    }
    if (!g_state.ctx.started) {
        return ORBIS_NP_MATCHING2_ERROR_CONTEXT_NOT_STARTED;
    }
    if (!reqParam) {
        return ORBIS_NP_MATCHING2_ERROR_INVALID_ARGUMENT;
    }

    const auto* req = reinterpret_cast<const OrbisNpMatching2SearchRoomRequest*>(reqParam);
    LOG_INFO(Lib_NpMatching2,
             "SearchRoom: option={} worldId={} lobbyId={} range.start={} range.max={} "
             "flagFilter={:#x} flagAttrs={:#x} intFilters={} binFilters={} attrs={}",
             req->option, req->worldId, req->lobbyId, req->rangeFilter.start, req->rangeFilter.max,
             req->flagFilter, req->flagAttrs, req->intFilters, req->binFilters, req->attrs);

    OrbisNpMatching2RequestCallback callback = nullptr;
    void* callback_arg = nullptr;
    if (optParam) {
        auto* opt = reinterpret_cast<OrbisNpMatching2RequestOptParam*>(optParam);
        callback = opt->callback;
        callback_arg = opt->arg;
        LOG_INFO(Lib_NpMatching2, "SearchRoom: optParam callback={:p} arg={:p} timeout={} appId={}",
                 static_cast<void*>(opt->callback), opt->arg, opt->timeout, opt->appId);
    }
    if (!callback) {
        callback = g_state.default_request_callback;
        callback_arg = g_state.default_request_callback_arg;
    }

    OrbisNpMatching2RequestId rid = g_state.next_request_id++;
    if (reqId) {
        *reqId = static_cast<s32>(rid);
    }

    BinaryWriter emptyW;
    std::vector<u8> roomListReply;
    auto roomListErr = MmRequest(ShadnetCommandType::GetRoomList, emptyW, roomListReply);
    if (roomListErr != ShadnetErrorType::NoError) {
        LOG_ERROR(Lib_NpMatching2, "SearchRoom: GetRoomList failed err={}",
                  static_cast<u8>(roomListErr));
        return ORBIS_NP_MATCHING2_ERROR_TIMEDOUT;
    }

    auto* response = BuildSearchRoomResponseFromBinary(roomListReply);

    // Client-side filtering: apply intFilter criteria to the room list.
    // The server returns all rooms, so we filter here.
    // Missing attrs are treated as value 0.
    if (response && response->roomDataExt && req->intFilter && req->intFilters > 0) {
        // Log filter criteria
        for (u64 f = 0; f < req->intFilters; ++f) {
            const auto& filt = req->intFilter[f];
            LOG_INFO(Lib_NpMatching2, "SearchRoom: intFilter[{}] op={} attrId={} attrVal={}", f,
                     static_cast<int>(filt.op), filt.attr.id, filt.attr.attr);
        }
        auto* head = response->roomDataExt;
        OrbisNpMatching2RoomDataExternal* prev = nullptr;
        auto* cur = head;
        u32 kept = 0;
        while (cur) {
            // Log room's attrs
            LOG_INFO(Lib_NpMatching2, "SearchRoom: room {:#x} has {} searchIntAttrs", cur->roomId,
                     cur->externalSearchIntAttrs);
            for (u64 a = 0; a < cur->externalSearchIntAttrs; ++a) {
                LOG_INFO(Lib_NpMatching2, "SearchRoom:   attr[{}] id={} val={}", a,
                         cur->externalSearchIntAttr[a].id, cur->externalSearchIntAttr[a].attr);
            }
            bool pass = true;
            for (u64 f = 0; f < req->intFilters && pass; ++f) {
                const auto& filt = req->intFilter[f];
                u32 roomVal = 0; // default to 0 if attr not found
                for (u64 a = 0; a < cur->externalSearchIntAttrs; ++a) {
                    if (cur->externalSearchIntAttr[a].id == filt.attr.id) {
                        roomVal = cur->externalSearchIntAttr[a].attr;
                        break;
                    }
                }
                u32 filtVal = filt.attr.attr;
                switch (filt.op) {
                case OrbisNpMatching2Operator::Eq:
                    pass = (roomVal == filtVal);
                    break;
                case OrbisNpMatching2Operator::Ne:
                    pass = (roomVal != filtVal);
                    break;
                case OrbisNpMatching2Operator::Lt:
                    pass = (roomVal < filtVal);
                    break;
                case OrbisNpMatching2Operator::Le:
                    pass = (roomVal <= filtVal);
                    break;
                case OrbisNpMatching2Operator::Gt:
                    pass = (roomVal > filtVal);
                    break;
                case OrbisNpMatching2Operator::Ge:
                    pass = (roomVal >= filtVal);
                    break;
                }
            }
            if (pass) {
                prev = cur;
                cur = cur->next;
                kept++;
            } else {
                if (prev)
                    prev->next = cur->next;
                else
                    head = cur->next;
                cur = prev ? prev->next : head;
            }
        }
        response->roomDataExt = head;
        response->range.results = kept;
        response->range.total = kept;
    }

    LOG_INFO(Lib_NpMatching2, "SearchRoom: built response reqId={} results={} data={:p}", rid,
             response ? response->range.results : 0, static_cast<void*>(response));

    if (callback && response) {
        PendingEvent ev{};
        ev.type = PendingEvent::REQUEST_CB;
        ev.fire_at = std::chrono::steady_clock::now();
        ev.req_id = rid;
        ev.req_event = ORBIS_NP_MATCHING2_REQUEST_EVENT_SEARCH_ROOM;
        ev.error_code = 0;
        ev.request_cb = callback;
        ev.request_cb_arg = callback_arg;
        ev.request_data = response;
        ScheduleEvent(std::move(ev));
    }

    return ORBIS_OK;
}

s32 PS4_SYSV_ABI sceNpMatching2KickoutRoomMember(OrbisNpMatching2ContextId ctxId, void* reqParam,
                                                 void* optParam, s32* reqId) {
    LOG_ERROR(Lib_NpMatching2, "(STUBBED) called ctxId={} reqParam={:p} optParam={:p}", ctxId,
              reqParam, optParam);
    if (reqId)
        *reqId = static_cast<s32>(g_state.next_request_id++);
    return ORBIS_OK;
}

s32 PS4_SYSV_ABI sceNpMatching2GrantRoomOwner(OrbisNpMatching2ContextId ctxId, void* reqParam,
                                              void* optParam, s32* reqId) {
    LOG_ERROR(Lib_NpMatching2, "(STUBBED) called ctxId={} reqParam={:p} optParam={:p}", ctxId,
              reqParam, optParam);
    if (reqId)
        *reqId = static_cast<s32>(g_state.next_request_id++);
    return ORBIS_OK;
}

s32 PS4_SYSV_ABI sceNpMatching2SendRoomMessage(OrbisNpMatching2ContextId ctxId, void* reqParam,
                                               void* optParam, s32* reqId) {
    LOG_ERROR(Lib_NpMatching2, "(STUBBED) called ctxId={} reqParam={:p} optParam={:p}", ctxId,
              reqParam, optParam);
    if (reqId)
        *reqId = static_cast<s32>(g_state.next_request_id++);
    return ORBIS_OK;
}

s32 PS4_SYSV_ABI sceNpMatching2SendRoomChatMessage(OrbisNpMatching2ContextId ctxId, void* reqParam,
                                                   void* optParam, s32* reqId) {
    LOG_ERROR(Lib_NpMatching2, "(STUBBED) called ctxId={} reqParam={:p} optParam={:p}", ctxId,
              reqParam, optParam);
    if (reqId)
        *reqId = static_cast<s32>(g_state.next_request_id++);
    return ORBIS_OK;
}

// --- Room data ---

s32 PS4_SYSV_ABI sceNpMatching2GetRoomDataInternal(OrbisNpMatching2ContextId ctxId, void* reqParam,
                                                   void* optParam, s32* reqId) {
    LOG_INFO(Lib_NpMatching2, "called ctxId={} reqParam={:p} optParam={:p}", ctxId, reqParam,
             optParam);
    if (!g_state.initialized) {
        return ORBIS_NP_MATCHING2_ERROR_NOT_INITIALIZED;
    }
    if (!g_state.ctx.started) {
        return ORBIS_NP_MATCHING2_ERROR_CONTEXT_NOT_STARTED;
    }

    OrbisNpMatching2RequestCallback callback = nullptr;
    void* callback_arg = nullptr;
    if (optParam) {
        auto* opt = reinterpret_cast<OrbisNpMatching2RequestOptParam*>(optParam);
        callback = opt->callback;
        callback_arg = opt->arg;
    }
    if (!callback) {
        callback = g_state.default_request_callback;
        callback_arg = g_state.default_request_callback_arg;
    }

    OrbisNpMatching2RequestId rid = g_state.next_request_id++;
    if (reqId)
        *reqId = static_cast<s32>(rid);

    // If we have room data, schedule a callback with it.
    // Otherwise fire with null data (game should handle gracefully).
    if (g_state.ctx.room_id != 0 && callback) {
        // Rebuild room data from current state so it's fresh
        std::vector<MemberInfo> members;
        MemberInfo self{};
        self.member_id = g_state.ctx.my_member_id;
        self.online_id = g_state.ctx.online_id;
        members.push_back(self);
        for (const auto& [mid, pi] : g_state.peers) {
            MemberInfo mi{};
            mi.member_id = mid;
            members.push_back(mi);
        }

        void* request_data = BuildRequestCallbackData(
            ORBIS_NP_MATCHING2_REQUEST_EVENT_GET_ROOM_DATA_INTERNAL, 0, g_state.ctx.room_id,
            g_state.ctx.server_id, g_state.ctx.world_id, g_state.ctx.lobby_id, members, 1,
            g_state.ctx.my_member_id, g_state.ctx.max_slot, g_state.ctx.flag_attr);

        PendingEvent ev{};
        ev.type = PendingEvent::REQUEST_CB;
        ev.fire_at = std::chrono::steady_clock::now() + std::chrono::milliseconds(100);
        ev.req_id = rid;
        ev.req_event = ORBIS_NP_MATCHING2_REQUEST_EVENT_GET_ROOM_DATA_INTERNAL;
        ev.error_code = 0;
        ev.request_cb = callback;
        ev.request_cb_arg = callback_arg;
        ev.request_data = request_data;
        ScheduleEvent(std::move(ev));

        LOG_INFO(Lib_NpMatching2, "GetRoomDataInternal: scheduled callback (event=0x109, reqId={})",
                 rid);
    } else {
        LOG_WARNING(Lib_NpMatching2, "GetRoomDataInternal: no room or no callback, skipping");
    }

    return ORBIS_OK;
}

s32 PS4_SYSV_ABI sceNpMatching2SetRoomDataInternal(OrbisNpMatching2ContextId ctxId,
                                                   const void* reqParam, const void* optParam,
                                                   s32* reqId) {
    LOG_INFO(Lib_NpMatching2, "called ctxId={} reqParam={:p} optParam={:p}", ctxId, reqParam,
             optParam);

    OrbisNpMatching2RequestCallback callback = nullptr;
    void* callback_arg = nullptr;
    if (optParam) {
        auto* opt = reinterpret_cast<const OrbisNpMatching2RequestOptParam*>(optParam);
        callback = opt->callback;
        callback_arg = opt->arg;
    }
    if (!callback) {
        callback = g_state.default_request_callback;
        callback_arg = g_state.default_request_callback_arg;
    }

    OrbisNpMatching2RequestId rid = g_state.next_request_id++;
    if (reqId)
        *reqId = static_cast<s32>(rid);

    const auto* req = reinterpret_cast<const OrbisNpMatching2SetRoomDataInternalRequest*>(reqParam);

    LOG_INFO(Lib_NpMatching2,
             "SetRoomDataInternal: rid={} roomId={} flagFilter={:#x} flagAttr={:#x} "
             "binAttrCount={} passwdSlotMask={}",
             rid, req->roomId, req->flagFilter, req->flagAttr, req->roomBinAttrInternalNum,
             req->passwordSlotMask ? *req->passwordSlotMask : 0);

    BinaryWriter w;
    w.AppendU32LE(rid);
    w.AppendU64LE(req->roomId);
    w.AppendU32LE(req->flagFilter);
    w.AppendU32LE(req->flagAttr);
    w.AppendU16LE(static_cast<u16>(req->roomBinAttrInternalNum));
    for (u32 i = 0; i < req->roomBinAttrInternalNum; ++i) {
        const auto& attr = req->roomBinAttrInternal[i];
        w.AppendU16LE(attr.id);
        w.AppendU32LE(static_cast<u32>(attr.dataSize));
        if (attr.data && attr.dataSize > 0)
            w.AppendBytes(attr.data, static_cast<size_t>(attr.dataSize));
    }
    const bool has_passwd_mask = (req->passwordSlotMask != nullptr);
    w.AppendU8(has_passwd_mask ? 1 : 0);
    if (has_passwd_mask)
        w.AppendU64LE(*req->passwordSlotMask);

    MmSendFireAndForget(ShadnetCommandType::SetRoomDataInternal, w);
    LOG_INFO(Lib_NpMatching2, "SetRoomDataInternal: sent reqId={}", rid);

    // Callback fires when server sends RequestEvent(0x0109) back to us.
    // Store so DispatchNotification can use it.
    g_state.per_request_callback = callback;
    g_state.per_request_callback_arg = callback_arg;

    return ORBIS_OK;
}

s32 PS4_SYSV_ABI sceNpMatching2SetRoomDataInternalExt() {
    LOG_ERROR(Lib_NpMatching2, "(STUBBED) called");
    return ORBIS_OK;
}

s32 PS4_SYSV_ABI sceNpMatching2SetRoomDataExternal(OrbisNpMatching2ContextId ctxId,
                                                   const void* reqParam, const void* optParam,
                                                   s32* reqId) {
    LOG_INFO(Lib_NpMatching2, "called ctxId={}", ctxId);

    OrbisNpMatching2RequestCallback callback = nullptr;
    void* callback_arg = nullptr;
    if (optParam) {
        auto* opt = reinterpret_cast<const OrbisNpMatching2RequestOptParam*>(optParam);
        callback = opt->callback;
        callback_arg = opt->arg;
    }
    if (!callback) {
        callback = g_state.default_request_callback;
        callback_arg = g_state.default_request_callback_arg;
    }

    OrbisNpMatching2RequestId rid = g_state.next_request_id++;
    if (reqId)
        *reqId = static_cast<s32>(rid);

    const auto* req = reinterpret_cast<const OrbisNpMatching2SetRoomDataExternalRequest*>(reqParam);

    LOG_INFO(Lib_NpMatching2,
             "SetRoomDataExternal: rid={} roomId={} searchIntAttrs={} searchBinAttrs={} "
             "binAttrs={}",
             rid, req->roomId, req->roomSearchableIntAttrExternalNum,
             req->roomSearchableBinAttrExternalNum, req->roomBinAttrExternalNum);

    BinaryWriter w;
    w.AppendU32LE(rid);
    w.AppendU64LE(req->roomId);

    // Searchable int attrs
    w.AppendU16LE(static_cast<u16>(req->roomSearchableIntAttrExternalNum));
    for (u32 i = 0; i < req->roomSearchableIntAttrExternalNum; ++i) {
        const auto& attr = req->roomSearchableIntAttrExternal[i];
        w.AppendU16LE(attr.id);
        w.AppendU32LE(attr.attr);
    }

    // Searchable bin attrs
    w.AppendU16LE(static_cast<u16>(req->roomSearchableBinAttrExternalNum));
    for (u32 i = 0; i < req->roomSearchableBinAttrExternalNum; ++i) {
        const auto& attr = req->roomSearchableBinAttrExternal[i];
        w.AppendU16LE(attr.id);
        w.AppendU32LE(static_cast<u32>(attr.dataSize));
        if (attr.data && attr.dataSize > 0)
            w.AppendBytes(attr.data, static_cast<size_t>(attr.dataSize));
    }

    // External bin attrs
    w.AppendU16LE(static_cast<u16>(req->roomBinAttrExternalNum));
    for (u32 i = 0; i < req->roomBinAttrExternalNum; ++i) {
        const auto& attr = req->roomBinAttrExternal[i];
        w.AppendU16LE(attr.id);
        w.AppendU32LE(static_cast<u32>(attr.dataSize));
        if (attr.data && attr.dataSize > 0)
            w.AppendBytes(attr.data, static_cast<size_t>(attr.dataSize));
    }

    MmSendFireAndForget(ShadnetCommandType::SetRoomDataExternal, w);

    g_state.per_request_callback = callback;
    g_state.per_request_callback_arg = callback_arg;

    return ORBIS_OK;
}

s32 PS4_SYSV_ABI sceNpMatching2GetRoomDataExternalList() {
    LOG_ERROR(Lib_NpMatching2, "(STUBBED) called");
    return ORBIS_OK;
}

s32 PS4_SYSV_ABI sceNpMatching2GetRoomMemberDataInternal() {
    LOG_ERROR(Lib_NpMatching2, "(STUBBED) called");
    return ORBIS_OK;
}

s32 PS4_SYSV_ABI sceNpMatching2SetRoomMemberDataInternal() {
    LOG_ERROR(Lib_NpMatching2, "(STUBBED) called");
    return ORBIS_OK;
}

s32 PS4_SYSV_ABI sceNpMatching2GetRoomMemberDataExternalList() {
    LOG_ERROR(Lib_NpMatching2, "(STUBBED) called");
    return ORBIS_OK;
}

s32 PS4_SYSV_ABI sceNpMatching2GetRoomMemberIdListLocal() {
    LOG_ERROR(Lib_NpMatching2, "(STUBBED) called");
    return ORBIS_OK;
}

s32 PS4_SYSV_ABI sceNpMatching2GetRoomJoinedSlotMaskLocal() {
    LOG_ERROR(Lib_NpMatching2, "(STUBBED) called");
    return ORBIS_OK;
}

s32 PS4_SYSV_ABI sceNpMatching2GetRoomPasswordLocal() {
    LOG_ERROR(Lib_NpMatching2, "(STUBBED) called");
    return ORBIS_OK;
}

// --- Lobby operations ---

// Async thread for JoinLobby callback
struct AsyncJoinLobbyArgs {
    OrbisNpMatching2ContextId ctx_id;
    OrbisNpMatching2RequestId req_id;
    OrbisNpMatching2RequestCallback callback;
    void* callback_arg;
};

static PS4_SYSV_ABI void* JoinLobbyThreadFunc(void* arg) {
    auto* a = static_cast<AsyncJoinLobbyArgs*>(arg);

    // Delay to ensure the API call returns first
    std::this_thread::sleep_for(std::chrono::milliseconds(200));

    LOG_INFO(Lib_NpMatching2, "JoinLobby async: firing request callback (no-op)");

    // Fire request callback only (effectively a no-op acknowledgment).
    // Do NOT fire context callback 0x6F03 — that event causes the game to destroy the context.
    if (a->callback) {
        a->callback(a->ctx_id, a->req_id, ORBIS_NP_MATCHING2_REQUEST_EVENT_JOIN_LOBBY, 0, nullptr,
                    a->callback_arg);
    }

    delete a;
    return nullptr;
}

s32 PS4_SYSV_ABI sceNpMatching2JoinLobby(OrbisNpMatching2ContextId ctxId, void* reqParam,
                                         void* optParam, s32* reqId) {
    LOG_INFO(Lib_NpMatching2, "called ctxId={} reqParam={:p} optParam={:p}", ctxId, reqParam,
             optParam);
    if (optParam) {
        auto* opt = reinterpret_cast<OrbisNpMatching2RequestOptParam*>(optParam);
        LOG_INFO(Lib_NpMatching2, "JoinLobby: optParam callback={:p} arg={:p} timeout={} appId={}",
                 static_cast<void*>(opt->callback), opt->arg, opt->timeout, opt->appId);
    }
    if (!g_state.initialized) {
        return ORBIS_NP_MATCHING2_ERROR_NOT_INITIALIZED;
    }
    if (!g_state.ctx.started) {
        return ORBIS_NP_MATCHING2_ERROR_CONTEXT_NOT_STARTED;
    }

    OrbisNpMatching2RequestCallback callback = nullptr;
    void* callback_arg = nullptr;
    if (optParam) {
        auto* opt = reinterpret_cast<OrbisNpMatching2RequestOptParam*>(optParam);
        callback = opt->callback;
        callback_arg = opt->arg;
    }
    if (!callback) {
        callback = g_state.default_request_callback;
        callback_arg = g_state.default_request_callback_arg;
    }

    OrbisNpMatching2RequestId rid = g_state.next_request_id++;
    if (reqId) {
        *reqId = static_cast<s32>(rid);
    }

    // Launch async PS4 thread for callback
    auto* args = new AsyncJoinLobbyArgs{ctxId, rid, callback, callback_arg};
    Kernel::PthreadT thread = nullptr;
    int ret = Kernel::posix_pthread_create(&thread, nullptr, JoinLobbyThreadFunc, args);
    if (ret != 0) {
        LOG_ERROR(Lib_NpMatching2, "JoinLobby: failed to create thread: {}", ret);
        delete args;
    }

    return ORBIS_OK;
}

s32 PS4_SYSV_ABI sceNpMatching2LeaveLobby() {
    LOG_ERROR(Lib_NpMatching2, "(STUBBED) called");
    return ORBIS_OK;
}

s32 PS4_SYSV_ABI sceNpMatching2GetLobbyInfoList() {
    LOG_ERROR(Lib_NpMatching2, "(STUBBED) called");
    return ORBIS_OK;
}

s32 PS4_SYSV_ABI sceNpMatching2GetLobbyMemberDataInternal() {
    LOG_ERROR(Lib_NpMatching2, "(STUBBED) called");
    return ORBIS_OK;
}

s32 PS4_SYSV_ABI sceNpMatching2GetLobbyMemberDataInternalList() {
    LOG_ERROR(Lib_NpMatching2, "(STUBBED) called");
    return ORBIS_OK;
}

s32 PS4_SYSV_ABI sceNpMatching2SetLobbyMemberDataInternal() {
    LOG_ERROR(Lib_NpMatching2, "(STUBBED) called");
    return ORBIS_OK;
}

s32 PS4_SYSV_ABI sceNpMatching2SendLobbyChatMessage() {
    LOG_ERROR(Lib_NpMatching2, "(STUBBED) called");
    return ORBIS_OK;
}

// --- Server/world info ---

s32 PS4_SYSV_ABI sceNpMatching2GetServerId(OrbisNpMatching2ContextId ctxId,
                                           OrbisNpMatching2ServerId* serverId) {
    LOG_INFO(Lib_NpMatching2, "called ctxId={}", ctxId);
    if (serverId) {
        *serverId = g_state.ctx.server_id;
    }
    return ORBIS_OK;
}

s32 PS4_SYSV_ABI sceNpMatching2GetWorldIdArrayForAllServers() {
    LOG_ERROR(Lib_NpMatching2, "(STUBBED) called");
    return ORBIS_OK;
}

s32 PS4_SYSV_ABI sceNpMatching2GetWorldInfoList(OrbisNpMatching2ContextId ctxId, void* reqParam,
                                                void* optParam, s32* reqId) {
    LOG_INFO(Lib_NpMatching2, "called ctxId={} reqParam={:p} optParam={:p}", ctxId, reqParam,
             optParam);
    if (optParam) {
        auto* opt = reinterpret_cast<OrbisNpMatching2RequestOptParam*>(optParam);
        LOG_INFO(Lib_NpMatching2,
                 "GetWorldInfoList: optParam callback={:p} arg={:p} timeout={} appId={}",
                 static_cast<void*>(opt->callback), opt->arg, opt->timeout, opt->appId);
    }
    if (!g_state.initialized) {
        return ORBIS_NP_MATCHING2_ERROR_NOT_INITIALIZED;
    }
    if (!g_state.ctx.started) {
        return ORBIS_NP_MATCHING2_ERROR_CONTEXT_NOT_STARTED;
    }

    OrbisNpMatching2RequestCallback callback = nullptr;
    void* callback_arg = nullptr;
    if (optParam) {
        auto* opt = reinterpret_cast<OrbisNpMatching2RequestOptParam*>(optParam);
        callback = opt->callback;
        callback_arg = opt->arg;
    }
    if (!callback) {
        callback = g_state.default_request_callback;
        callback_arg = g_state.default_request_callback_arg;
    }

    OrbisNpMatching2RequestId rid = g_state.next_request_id++;
    if (reqId) {
        *reqId = static_cast<s32>(rid);
    }

    // Launch async PS4 thread to fire callback with world data
    auto* args = new AsyncGetWorldInfoArgs{ctxId, rid, callback, callback_arg};
    Kernel::PthreadT thread = nullptr;
    int ret = Kernel::posix_pthread_create(&thread, nullptr, GetWorldInfoListThreadFunc, args);
    if (ret != 0) {
        LOG_ERROR(Lib_NpMatching2, "GetWorldInfoList: failed to create thread: {}", ret);
        delete args;
    }

    return ORBIS_OK;
}

// --- User info ---

s32 PS4_SYSV_ABI sceNpMatching2GetUserInfoList() {
    LOG_ERROR(Lib_NpMatching2, "(STUBBED) called");
    return ORBIS_OK;
}

s32 PS4_SYSV_ABI sceNpMatching2GetUserInfoListA() {
    LOG_ERROR(Lib_NpMatching2, "(STUBBED) called");
    return ORBIS_OK;
}

s32 PS4_SYSV_ABI sceNpMatching2SetUserInfo() {
    LOG_ERROR(Lib_NpMatching2, "(STUBBED) called");
    return ORBIS_OK;
}

// --- Signaling ---

s32 PS4_SYSV_ABI sceNpMatching2SignalingGetConnectionStatus(OrbisNpMatching2ContextId ctxId,
                                                            OrbisNpMatching2RoomId roomId,
                                                            OrbisNpMatching2RoomMemberId memberId,
                                                            s32* connStatus, void* peerAddr,
                                                            u16* peerPort) {
    LOG_INFO(Lib_NpMatching2, "called ctxId={} roomId={} memberId={}", ctxId, roomId, memberId);

    auto it = g_state.peers.find(memberId);

    if (it != g_state.peers.end()) {
        if (connStatus)
            *connStatus = it->second.status;
        if (peerAddr && it->second.status == ORBIS_NP_MATCHING2_SIGNALING_CONN_STATUS_ACTIVE) {
            std::memcpy(peerAddr, &it->second.addr, 4);
        }
        if (peerPort && it->second.status == ORBIS_NP_MATCHING2_SIGNALING_CONN_STATUS_ACTIVE) {
            *peerPort = it->second.port;
        }
        LOG_INFO(Lib_NpMatching2,
                 "SignalingGetConnectionStatus: member={} → status={} addr={:#x} port={} "
                 "peerAddr={:p} peerPort={:p}",
                 memberId, it->second.status, it->second.addr, ntohs(it->second.port), peerAddr,
                 static_cast<void*>(peerPort));
    } else {
        if (connStatus)
            *connStatus = ORBIS_NP_MATCHING2_SIGNALING_CONN_STATUS_INACTIVE;
        LOG_INFO(Lib_NpMatching2,
                 "SignalingGetConnectionStatus: member={} → INACTIVE (no peer entry)", memberId);
    }

    return ORBIS_OK;
}

s32 PS4_SYSV_ABI sceNpMatching2SignalingGetConnectionInfo(OrbisNpMatching2ContextId ctxId,
                                                          OrbisNpMatching2RoomId roomId,
                                                          OrbisNpMatching2RoomMemberId memberId,
                                                          u32 infoType, void* connInfo) {
    LOG_INFO(Lib_NpMatching2, "called ctxId={} roomId={} memberId={} infoType={}", ctxId, roomId,
             memberId, infoType);

    if (!connInfo)
        return ORBIS_OK;

    std::lock_guard<std::mutex> lock(g_state.mutex);
    auto it = g_state.peers.find(memberId);
    const bool active = it != g_state.peers.end() &&
                        it->second.status == ORBIS_NP_MATCHING2_SIGNALING_CONN_STATUS_ACTIVE;

    if (infoType == 1) {
        // infoType=1: first u32 is RTT in microseconds.
        // 0 means not connected; non-zero maps to a connection quality level.
        *reinterpret_cast<u32*>(connInfo) = active ? 1000u : 0u;
        LOG_INFO(Lib_NpMatching2,
                 "SignalingGetConnectionInfo: member={} infoType=1 rtt_us={} (active={})", memberId,
                 active ? 1000u : 0u, active);
    } else {
        auto* info = reinterpret_cast<OrbisNpMatching2SignalingConnectionInfo*>(connInfo);
        if (it != g_state.peers.end()) {
            info->address.addr = it->second.addr;
            info->address.port = it->second.port;
            LOG_INFO(Lib_NpMatching2,
                     "SignalingGetConnectionInfo: member={} infoType={} addr={:#x} port={}",
                     memberId, infoType, info->address.addr, ntohs(info->address.port));
        } else {
            info->address.addr = 0;
            info->address.port = 0;
            LOG_WARNING(Lib_NpMatching2, "SignalingGetConnectionInfo: member={} not in peer table",
                        memberId);
        }
    }

    return ORBIS_OK;
}

s32 PS4_SYSV_ABI sceNpMatching2SignalingGetConnectionInfoA() {
    LOG_ERROR(Lib_NpMatching2, "(STUBBED) called");
    return ORBIS_OK;
}

s32 PS4_SYSV_ABI sceNpMatching2SignalingEstablishConnection(OrbisNpMatching2ContextId ctxId,
                                                            OrbisNpMatching2RoomId roomId,
                                                            OrbisNpMatching2RoomMemberId memberId,
                                                            void* optParam, s32* reqId) {
    LOG_INFO(Lib_NpMatching2, "called ctxId={} roomId={} memberId={}", ctxId, roomId, memberId);
    (void)optParam;

    OrbisNpMatching2RequestId rid = g_state.next_request_id++;
    if (reqId)
        *reqId = static_cast<s32>(rid);

    auto it = g_state.peers.find(memberId);
    if (it == g_state.peers.end()) {
        LOG_WARNING(Lib_NpMatching2,
                    "SignalingEstablishConnection: no peer info for member={} roomId={}", memberId,
                    roomId);
        return ORBIS_OK;
    }

    it->second.status = ORBIS_NP_MATCHING2_SIGNALING_CONN_STATUS_PENDING;
    LOG_INFO(Lib_NpMatching2,
             "SignalingEstablishConnection: member={} roomId={} marked pending without signaling "
             "helper bridge",
             memberId, roomId);

    return ORBIS_OK;
}

s32 PS4_SYSV_ABI sceNpMatching2SignalingAbortConnection() {
    LOG_ERROR(Lib_NpMatching2, "(STUBBED) called");
    return ORBIS_OK;
}

s32 PS4_SYSV_ABI sceNpMatching2SignalingGetLocalNetInfo(OrbisNpMatching2ContextId ctxId,
                                                        void* info) {
    LOG_INFO(Lib_NpMatching2, "called ctxId={}", ctxId);
    if (!info) {
        return ORBIS_OK;
    }
    // Delegate to NpSignaling HLE which holds the local addr set by ContextStart.
    // Cast to OrbisNpSignalingNetInfo — the two structs are identical in layout.
    auto* net_info = reinterpret_cast<NpSignaling::OrbisNpSignalingNetInfo*>(info);
    return NpSignaling::sceNpSignalingGetLocalNetInfo(static_cast<s32>(ctxId), net_info);
}

s32 PS4_SYSV_ABI sceNpMatching2SignalingGetPeerNetInfo() {
    LOG_ERROR(Lib_NpMatching2, "(STUBBED) called");
    return ORBIS_OK;
}

s32 PS4_SYSV_ABI sceNpMatching2SignalingGetPeerNetInfoResult() {
    LOG_ERROR(Lib_NpMatching2, "(STUBBED) called");
    return ORBIS_OK;
}

s32 PS4_SYSV_ABI sceNpMatching2SignalingCancelPeerNetInfo() {
    LOG_ERROR(Lib_NpMatching2, "(STUBBED) called");
    return ORBIS_OK;
}

s32 PS4_SYSV_ABI sceNpMatching2SignalingSetPort() {
    LOG_ERROR(Lib_NpMatching2, "(STUBBED) called");
    return ORBIS_OK;
}

s32 PS4_SYSV_ABI sceNpMatching2SignalingGetPort() {
    LOG_ERROR(Lib_NpMatching2, "(STUBBED) called");
    return ORBIS_OK;
}

s32 PS4_SYSV_ABI sceNpMatching2SignalingGetPingInfo(OrbisNpMatching2ContextId ctxId,
                                                    const void* reqParam, const void* optParam,
                                                    s32* reqId) {
    LOG_INFO(Lib_NpMatching2, "called ctxId={} reqParam={:p} optParam={:p}", ctxId, reqParam,
             optParam);

    OrbisNpMatching2RequestCallback callback = nullptr;
    void* callback_arg = nullptr;
    if (optParam) {
        auto* opt = reinterpret_cast<const OrbisNpMatching2RequestOptParam*>(optParam);
        callback = opt->callback;
        callback_arg = opt->arg;
    }
    if (!callback) {
        callback = g_state.default_request_callback;
        callback_arg = g_state.default_request_callback_arg;
    }

    OrbisNpMatching2RequestId rid = g_state.next_request_id++;
    if (reqId)
        *reqId = static_cast<s32>(rid);

    // Allocate response and fill with current room state
    auto* resp = new OrbisNpMatching2SignalingGetPingInfoResponse{};
    resp->serverId = g_state.ctx.server_id;
    resp->worldId = g_state.ctx.world_id;
    resp->roomId = g_state.ctx.room_id;
    resp->pingUs = 1000; // 1ms — local/LAN estimate

    delete g_state.last_ping_info_response;
    g_state.last_ping_info_response = resp;

    if (callback) {
        PendingEvent ev{};
        ev.type = PendingEvent::REQUEST_CB;
        ev.fire_at = std::chrono::steady_clock::now();
        ev.req_id = rid;
        ev.req_event = ORBIS_NP_MATCHING2_REQUEST_EVENT_SIGNALING_GET_PING_INFO;
        ev.error_code = 0;
        ev.request_cb = callback;
        ev.request_cb_arg = callback_arg;
        ev.request_data = resp;
        ScheduleEvent(std::move(ev));
    }
    return ORBIS_OK;
}

s32 PS4_SYSV_ABI sceNpMatching2SignalingEnableManualUdpMode() {
    LOG_ERROR(Lib_NpMatching2, "(STUBBED) called");
    return ORBIS_OK;
}

s32 PS4_SYSV_ABI sceNpMatching2SetSignalingOptParam() {
    LOG_ERROR(Lib_NpMatching2, "(STUBBED) called");
    return ORBIS_OK;
}

s32 PS4_SYSV_ABI sceNpMatching2GetSignalingOptParamLocal() {
    LOG_ERROR(Lib_NpMatching2, "(STUBBED) called");
    return ORBIS_OK;
}

// --- Memory/misc ---

s32 PS4_SYSV_ABI sceNpMatching2GetMemoryInfo() {
    LOG_ERROR(Lib_NpMatching2, "(STUBBED) called");
    return ORBIS_OK;
}

s32 PS4_SYSV_ABI sceNpMatching2GetSslMemoryInfo() {
    LOG_ERROR(Lib_NpMatching2, "(STUBBED) called");
    return ORBIS_OK;
}

// --- RegisterLib ---

void RegisterLib(Core::Loader::SymbolsResolver* sym) {
    // Core lifecycle
    LIB_FUNCTION("10t3e5+JPnU", "libSceNpMatching2", 1, "libSceNpMatching2",
                 sceNpMatching2Initialize);
    LIB_FUNCTION("Mqp3lJ+sjy4", "libSceNpMatching2", 1, "libSceNpMatching2",
                 sceNpMatching2Terminate);
    LIB_FUNCTION("nHZpTF30wto", "libSceNpMatching2", 1, "libSceNpMatching2",
                 sceNpMatching2SetExtraInitParam);

    // Context management
    LIB_FUNCTION("YfmpW719rMo", "libSceNpMatching2", 1, "libSceNpMatching2",
                 sceNpMatching2CreateContext);
    LIB_FUNCTION("6xlf9+pa0GY", "libSceNpMatching2", 1, "libSceNpMatching2",
                 sceNpMatching2CreateContextInternal);
    LIB_FUNCTION("ajvzc8e2upo", "libSceNpMatching2", 1, "libSceNpMatching2",
                 sceNpMatching2CreateContextA);
    LIB_FUNCTION("7vjNQ6Z1op0", "libSceNpMatching2", 1, "libSceNpMatching2",
                 sceNpMatching2ContextStart);
    LIB_FUNCTION("-f6M4caNe8k", "libSceNpMatching2", 1, "libSceNpMatching2",
                 sceNpMatching2ContextStop);
    LIB_FUNCTION("Nz-ZE7ur32I", "libSceNpMatching2", 1, "libSceNpMatching2",
                 sceNpMatching2DestroyContext);
    LIB_FUNCTION("pFzhpCMlJXQ", "libSceNpMatching2", 1, "libSceNpMatching2",
                 sceNpMatching2AbortContextStart);

    // Callback registration
    LIB_FUNCTION("+8e7wXLmjds", "libSceNpMatching2", 1, "libSceNpMatching2",
                 sceNpMatching2SetDefaultRequestOptParam);
    LIB_FUNCTION("fQQfP87I7hs", "libSceNpMatching2", 1, "libSceNpMatching2",
                 sceNpMatching2RegisterContextCallback);
    LIB_FUNCTION("p+2EnxmaAMM", "libSceNpMatching2", 1, "libSceNpMatching2",
                 sceNpMatching2RegisterRoomEventCallback);
    LIB_FUNCTION("0UMeWRGnZKA", "libSceNpMatching2", 1, "libSceNpMatching2",
                 sceNpMatching2RegisterSignalingCallback);
    LIB_FUNCTION("4Nj7u5B5yCA", "libSceNpMatching2", 1, "libSceNpMatching2",
                 sceNpMatching2RegisterLobbyEventCallback);
    LIB_FUNCTION("DnPUsBAe8oI", "libSceNpMatching2", 1, "libSceNpMatching2",
                 sceNpMatching2RegisterLobbyMessageCallback);
    LIB_FUNCTION("uBESzz4CQws", "libSceNpMatching2", 1, "libSceNpMatching2",
                 sceNpMatching2RegisterRoomMessageCallback);
    LIB_FUNCTION("KT082n6I75E", "libSceNpMatching2", 1, "libSceNpMatching2",
                 sceNpMatching2RegisterManualUdpSignalingCallback);

    // Room operations
    LIB_FUNCTION("zCWZmXXN600", "libSceNpMatching2", 1, "libSceNpMatching2",
                 sceNpMatching2CreateJoinRoom);
    LIB_FUNCTION("V6KSpKv9XJE", "libSceNpMatching2", 1, "libSceNpMatching2",
                 sceNpMatching2CreateJoinRoomA);
    LIB_FUNCTION("CSIMDsVjs-g", "libSceNpMatching2", 1, "libSceNpMatching2",
                 sceNpMatching2JoinRoom);
    LIB_FUNCTION("gQ6cUriNpgs", "libSceNpMatching2", 1, "libSceNpMatching2",
                 sceNpMatching2JoinRoomA);
    LIB_FUNCTION("BD6kfx442Do", "libSceNpMatching2", 1, "libSceNpMatching2",
                 sceNpMatching2LeaveRoom);
    LIB_FUNCTION("VqZX7POg2Mk", "libSceNpMatching2", 1, "libSceNpMatching2",
                 sceNpMatching2SearchRoom);
    LIB_FUNCTION("AUVfU6byg3c", "libSceNpMatching2", 1, "libSceNpMatching2",
                 sceNpMatching2KickoutRoomMember);
    LIB_FUNCTION("NCP3bLGPt+o", "libSceNpMatching2", 1, "libSceNpMatching2",
                 sceNpMatching2GrantRoomOwner);
    LIB_FUNCTION("Iw2h0Jrrb5U", "libSceNpMatching2", 1, "libSceNpMatching2",
                 sceNpMatching2SendRoomMessage);
    LIB_FUNCTION("opDpl74pi2E", "libSceNpMatching2", 1, "libSceNpMatching2",
                 sceNpMatching2SendRoomChatMessage);

    // Room data
    LIB_FUNCTION("Jraxifmoet4", "libSceNpMatching2", 1, "libSceNpMatching2",
                 sceNpMatching2GetRoomDataInternal);
    LIB_FUNCTION("S9D8JSYIrjE", "libSceNpMatching2", 1, "libSceNpMatching2",
                 sceNpMatching2SetRoomDataInternal);
    LIB_FUNCTION("jMxxNNLh6ms", "libSceNpMatching2", 1, "libSceNpMatching2",
                 sceNpMatching2SetRoomDataInternalExt);
    LIB_FUNCTION("q7GK98-nYSE", "libSceNpMatching2", 1, "libSceNpMatching2",
                 sceNpMatching2SetRoomDataExternal);
    LIB_FUNCTION("26vWrPAWJfM", "libSceNpMatching2", 1, "libSceNpMatching2",
                 sceNpMatching2GetRoomDataExternalList);
    LIB_FUNCTION("5lhvOqheFBA", "libSceNpMatching2", 1, "libSceNpMatching2",
                 sceNpMatching2GetRoomMemberDataInternal);
    LIB_FUNCTION("HoqTrkS9c5Q", "libSceNpMatching2", 1, "libSceNpMatching2",
                 sceNpMatching2SetRoomMemberDataInternal);
    LIB_FUNCTION("dMQ+xGvTdqM", "libSceNpMatching2", 1, "libSceNpMatching2",
                 sceNpMatching2GetRoomMemberDataExternalList);
    LIB_FUNCTION("KC+GnHzrK2o", "libSceNpMatching2", 1, "libSceNpMatching2",
                 sceNpMatching2GetRoomMemberIdListLocal);
    LIB_FUNCTION("nddl5xnQQEY", "libSceNpMatching2", 1, "libSceNpMatching2",
                 sceNpMatching2GetRoomJoinedSlotMaskLocal);
    LIB_FUNCTION("vbtWT3lZBOM", "libSceNpMatching2", 1, "libSceNpMatching2",
                 sceNpMatching2GetRoomPasswordLocal);

    // Lobby operations
    LIB_FUNCTION("n5JmImxTiZU", "libSceNpMatching2", 1, "libSceNpMatching2",
                 sceNpMatching2JoinLobby);
    LIB_FUNCTION("BBbJ92uUdCg", "libSceNpMatching2", 1, "libSceNpMatching2",
                 sceNpMatching2LeaveLobby);
    LIB_FUNCTION("wyvlEgZ-55w", "libSceNpMatching2", 1, "libSceNpMatching2",
                 sceNpMatching2GetLobbyInfoList);
    LIB_FUNCTION("1JtbJ0kxm3E", "libSceNpMatching2", 1, "libSceNpMatching2",
                 sceNpMatching2GetLobbyMemberDataInternal);
    LIB_FUNCTION("1Z4Xxumgm+Y", "libSceNpMatching2", 1, "libSceNpMatching2",
                 sceNpMatching2GetLobbyMemberDataInternalList);
    LIB_FUNCTION("ir2CzSs9K-g", "libSceNpMatching2", 1, "libSceNpMatching2",
                 sceNpMatching2SetLobbyMemberDataInternal);
    LIB_FUNCTION("K+KtxhPsMZ4", "libSceNpMatching2", 1, "libSceNpMatching2",
                 sceNpMatching2SendLobbyChatMessage);

    // Server/world info
    LIB_FUNCTION("LhCPctIICxQ", "libSceNpMatching2", 1, "libSceNpMatching2",
                 sceNpMatching2GetServerId);
    LIB_FUNCTION("lagjVl+bHFI", "libSceNpMatching2", 1, "libSceNpMatching2",
                 sceNpMatching2GetWorldIdArrayForAllServers);
    LIB_FUNCTION("rJNPJqDCpiI", "libSceNpMatching2", 1, "libSceNpMatching2",
                 sceNpMatching2GetWorldInfoList);

    // User info
    LIB_FUNCTION("qeF-q5KDtAc", "libSceNpMatching2", 1, "libSceNpMatching2",
                 sceNpMatching2GetUserInfoList);
    LIB_FUNCTION("GyI2f9yDUXM", "libSceNpMatching2", 1, "libSceNpMatching2",
                 sceNpMatching2GetUserInfoListA);
    LIB_FUNCTION("meEjIdbjAA0", "libSceNpMatching2", 1, "libSceNpMatching2",
                 sceNpMatching2SetUserInfo);

    // Signaling
    LIB_FUNCTION("tHD5FPFXtu4", "libSceNpMatching2", 1, "libSceNpMatching2",
                 sceNpMatching2SignalingGetConnectionStatus);
    LIB_FUNCTION("twVupeaYYrk", "libSceNpMatching2", 1, "libSceNpMatching2",
                 sceNpMatching2SignalingGetConnectionInfo);
    LIB_FUNCTION("nNeC3F8-g+4", "libSceNpMatching2", 1, "libSceNpMatching2",
                 sceNpMatching2SignalingGetConnectionInfoA);
    LIB_FUNCTION("UcYuZkNhHI8", "libSceNpMatching2", 1, "libSceNpMatching2",
                 sceNpMatching2SignalingEstablishConnection);
    LIB_FUNCTION("eDxEHb9f7B8", "libSceNpMatching2", 1, "libSceNpMatching2",
                 sceNpMatching2SignalingAbortConnection);
    LIB_FUNCTION("380EWm2DrVg", "libSceNpMatching2", 1, "libSceNpMatching2",
                 sceNpMatching2SignalingGetLocalNetInfo);
    LIB_FUNCTION("8CqniKDzjvg", "libSceNpMatching2", 1, "libSceNpMatching2",
                 sceNpMatching2SignalingGetPeerNetInfo);
    LIB_FUNCTION("CTy4PBhpWDw", "libSceNpMatching2", 1, "libSceNpMatching2",
                 sceNpMatching2SignalingGetPeerNetInfoResult);
    LIB_FUNCTION("GNSN5849fjU", "libSceNpMatching2", 1, "libSceNpMatching2",
                 sceNpMatching2SignalingCancelPeerNetInfo);
    LIB_FUNCTION("wupHEf8WOhM", "libSceNpMatching2", 1, "libSceNpMatching2",
                 sceNpMatching2SignalingSetPort);
    LIB_FUNCTION("WkvclTMjNdI", "libSceNpMatching2", 1, "libSceNpMatching2",
                 sceNpMatching2SignalingGetPort);
    LIB_FUNCTION("wUmwXZHaX1w", "libSceNpMatching2", 1, "libSceNpMatching2",
                 sceNpMatching2SignalingGetPingInfo);
    LIB_FUNCTION("LKRatXLV85k", "libSceNpMatching2", 1, "libSceNpMatching2",
                 sceNpMatching2SignalingEnableManualUdpMode);
    LIB_FUNCTION("ES3UMUWWj9U", "libSceNpMatching2", 1, "libSceNpMatching2",
                 sceNpMatching2SetSignalingOptParam);
    LIB_FUNCTION("cgQhq3E0eGo", "libSceNpMatching2", 1, "libSceNpMatching2",
                 sceNpMatching2GetSignalingOptParamLocal);

    // Memory/misc
    LIB_FUNCTION("gpSAvdheZ0Q", "libSceNpMatching2", 1, "libSceNpMatching2",
                 sceNpMatching2GetMemoryInfo);
    LIB_FUNCTION("8btynvj0KNA", "libSceNpMatching2", 1, "libSceNpMatching2",
                 sceNpMatching2GetSslMemoryInfo);
}

} // namespace Libraries::Np::NpMatching2
