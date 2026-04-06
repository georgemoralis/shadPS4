// SPDX-FileCopyrightText: Copyright 2025 shadPS4 Emulator Project
// SPDX-License-Identifier: GPL-2.0-or-later

#include <chrono>
#include <cstring>
#include <thread>

#include "common/logging/log.h"
#include "core/libraries/error_codes.h"
#include "core/libraries/kernel/kernel.h"
#include "core/libraries/kernel/memory.h"
#include "core/libraries/network/net.h"
#include "core/libraries/network/sockets.h"
#include "core/libraries/np/np_error.h"
#include "core/libraries/np/np_matching2_mm.h"
#include "core/libraries/np/np_signaling_error.h"
#include "core/libraries/np/np_signaling_helpers.h"
#include "core/libraries/np/np_signaling_state.h"

namespace Libraries::Np::NpSignaling::Helpers {

namespace {
// These helpers back the staged initialize rewrite. They intentionally centralize
// the non-exported setup work that was previously being kept in np_signaling.cpp.
//
// Current status:
// - app-type helpers are decomp-shaped but still inferred because there is no
//   non-stub local sceNpIntGetAppType implementation to read from yet
// - heap helpers match the kernel flexible-memory part of FUN_0100c4b0 /
//   FUN_0100c5a0, but intentionally omit the libc mspace layer for now
// - IPMI helpers match the registration contract of FUN_0100bbd0 but do not
//   implement a real IPMI client/service path
// - runtime helpers are still placeholders until the thread/callout startup
//   chain is moved over here from np_signaling.cpp

// Inferred stand-in for sceNpIntGetAppType-backed state. This is deliberately
// local because the real app-type source is not implemented in the codebase yet.
constexpr s32 InferredNpAppType = 0;

// Helper-owned initialize state. These are local mirrors of the decomp's global
// setup flags/pointers while the rewrite is still being staged.
bool g_ipmi_handler_registered = false;
SignalingIpmiHandler g_ipmi_handler = nullptr;
bool g_signaling_heap_initialized = false;
void* g_signaling_heap_base = nullptr;
s64 g_signaling_heap_size = 0;
SignalingRuntimeHooks g_runtime_hooks{};
bool g_runtime_hooks_registered = false;
u64 g_echo_probe_word2 = 0;
u64 g_echo_probe_word3 = 0;
u64 g_echo_thread_handle = static_cast<u64>(-1);
s32 g_echo_thread_status = -1;

// Decomp shape: sceNpIntGetAppType(&local_type).
//
// Inference: this stays local until we have a real NP-int app-type source.
// Missing piece: the actual backing getter in Sony's NP stack.
s32 GetInferredAppType(s32* app_type) {
    if (app_type == nullptr) {
        return ORBIS_NP_INT_ERROR_INVALID_ARGUMENT;
    }

    *app_type = InferredNpAppType;
    return ORBIS_OK;
}

// Decomp shape: FUN_010021d0().
//
// Inference: modeled as an always-open gate for now because the real state
// source is still unknown.
s16 GetAppTypeStateGate() {
    return 0;
}

// Decomp shape: FUN_010021e0(0x245a).
//
// Inference: currently just a marker/log hook until the real side effect is
// identified.
void SetAppTypeMarker(u16 marker) {
    LOG_INFO(Lib_NpSignaling, "Helpers::SetAppTypeMarker marker={:#x}", marker);
}
} // namespace

// Staged bridge for the current HLE runtime pieces. This is intentionally a
// temporary adapter while the decomped startup chain is being moved out of
// sceNpSignalingInitialize and into this helper layer.
void SetRuntimeHooks(const SignalingRuntimeHooks& hooks) {
    LOG_INFO(Lib_NpSignaling, "Helpers::SetRuntimeHooks");
    g_runtime_hooks = hooks;
    g_runtime_hooks_registered = true;
}

// Maps to FUN_01007a20.
//
// Matching behavior:
// - fetch app type
// - if out pointer is non-null, write whether app type == 4
//
// Current inference:
// - app type comes from GetInferredAppType() because the real sceNpIntGetAppType
//   backing implementation is not available locally yet
s32 CheckInitializeAppType(u32* is_app_type_4) {
    LOG_INFO(Lib_NpSignaling, "Helpers::CheckInitializeAppType");

    s32 app_type = -1;
    const s32 rc = GetInferredAppType(&app_type);
    if (rc < 0) {
        return rc;
    }

    if (is_app_type_4 != nullptr) {
        *is_app_type_4 = app_type == 4 ? 1u : 0u;
    }

    return ORBIS_OK;
}

// Partial mapping of FUN_0100c4b0.
//
// Matching behavior:
// - reject zero-size / already-initialized cases locally
// - align requested size to 0x4000
// - map named flexible memory as "SceNpSignaling"
// - remember the resulting base/size
//
// Missing piece:
// - the real sceLibcMspaceCreate layer is intentionally not modeled here yet
s32 InitSignalingHeap(s64 pool_size) {
    LOG_INFO(Lib_NpSignaling, "Helpers::InitSignalingHeap pool_size={}", pool_size);

    if (pool_size == 0 || g_signaling_heap_initialized) {
        return ORBIS_NP_SIGNALING_INTERNAL_ERROR_ALLOCATOR;
    }

    const u64 alignment_mask = 0x3fff;
    const u64 remainder = static_cast<u64>(pool_size) & alignment_mask;
    const u64 aligned_size = remainder == 0 ? static_cast<u64>(pool_size)
                                            : static_cast<u64>(pool_size) + (0x4000 - remainder);

    void* heap_base = nullptr;
    const s32 rc = Libraries::Kernel::sceKernelMapNamedFlexibleMemory(&heap_base, aligned_size, 3,
                                                                      0, "SceNpSignaling");
    if (rc < 0) {
        return rc;
    }

    g_signaling_heap_initialized = true;
    g_signaling_heap_base = heap_base;
    g_signaling_heap_size = static_cast<s64>(aligned_size);
    return ORBIS_OK;
}

// Partial mapping of FUN_0100c5a0.
//
// Matching behavior:
// - only tears down if locally initialized
// - unmaps the flexible memory region and clears stored state
//
// Missing piece:
// - the real mspace empty/stats/destroy sequence is intentionally omitted
//   until libc-side support exists or is intentionally modeled locally
void ShutdownSignalingHeap() {
    LOG_INFO(Lib_NpSignaling, "Helpers::ShutdownSignalingHeap");

    if (!g_signaling_heap_initialized) {
        return;
    }

    g_signaling_heap_initialized = false;

    if (g_signaling_heap_base != nullptr) {
        Libraries::Kernel::sceKernelMunmap(g_signaling_heap_base,
                                           static_cast<u64>(g_signaling_heap_size));
        g_signaling_heap_base = nullptr;
        g_signaling_heap_size = 0;
    }
}

// Partial mapping of FUN_0100bbd0.
//
// Matching behavior:
// - null handler -> INVALID_ARGUMENT
// - store handler pointer on success
//
// Missing piece:
// - no real FUN_0100c740 / IPMI client registration path yet
s32 RegisterIpmiHandler(SignalingIpmiHandler handler) {
    LOG_INFO(Lib_NpSignaling, "Helpers::RegisterIpmiHandler handler={:p}",
             fmt::ptr(reinterpret_cast<void*>(handler)));
    if (handler == nullptr) {
        return ORBIS_NP_SIGNALING_INTERNAL_ERROR_INVALID_ARGUMENT;
    }

    g_ipmi_handler = handler;
    g_ipmi_handler_registered = true;
    return ORBIS_OK;
}

// Local reset point corresponding to the teardown side of the IPMI registration
// path. This is intentionally narrower than the real FUN_0100c7a0 path, which
// also talks to the IPMI service client.
void UnregisterIpmiHandler() {
    LOG_INFO(Lib_NpSignaling, "Helpers::UnregisterIpmiHandler");
    g_ipmi_handler = nullptr;
    g_ipmi_handler_registered = false;
}

// Decomp shape: FUN_010079b0.
//
// Matching control flow:
// - fetch app type
// - if app type == 5, consult a secondary gate
// - mark state with 0x245a
//
// Current inference:
// - app type is still local/inferred
// - the secondary gate and marker are placeholder stand-ins for still-unmapped
//   helper behavior
s32 CheckAppType() {
    LOG_INFO(Lib_NpSignaling, "Helpers::CheckAppType");

    s32 app_type = -1;
    const s32 rc = GetInferredAppType(&app_type);
    if (rc < 0) {
        return rc;
    }

    if (app_type == 5) {
        const s16 gate = GetAppTypeStateGate();
        if (gate != 0) {
            return ORBIS_OK;
        }
    }

    SetAppTypeMarker(0x245a);
    return ORBIS_OK;
}

// Planned home for FUN_010029a0-related startup work.
//
// Current staged behavior:
// - drives the existing HLE dispatch/receive/ping startup through registered
//   runtime hooks
//
// Missing pieces:
// - signaling socket initialization
// - lwmutex/callout setup
// - real main signaling thread creation
// - optional IPC recv thread setup
s32 StartMainRuntime(s32 thread_priority, s32 cpu_affinity_mask, s64 thread_stack_size) {
    LOG_INFO(Lib_NpSignaling,
             "Helpers::StartMainRuntime thread_priority={} cpu_affinity_mask={} "
             "thread_stack_size={}",
             thread_priority, cpu_affinity_mask, thread_stack_size);

    if (!g_runtime_hooks_registered) {
        return ORBIS_NP_SIGNALING_INTERNAL_ERROR_NOT_INITIALIZED;
    }

    if (g_runtime_hooks.start_dispatch != nullptr) {
        g_runtime_hooks.start_dispatch();
    }
    if (g_runtime_hooks.start_receive != nullptr) {
        g_runtime_hooks.start_receive();
    }
    if (g_runtime_hooks.start_ping != nullptr) {
        g_runtime_hooks.start_ping();
    }

    return ORBIS_OK;
}

// Planned home for FUN_01004f10-related echo startup.
//
// Current staged behavior:
// - creates the temporary ioctl socket
// - performs the currently-available local sceNetIoctl() stub call
// - resets the local echo-thread bookkeeping fields
//
// Still missing:
// - the real ioctl request/payload path because local sceNetIoctl is still a
//   zero-argument stub in this codebase
// - storing the decomp's returned trailing probe words
// - dedicated echo thread creation (FUN_010087f0)
// - echo thread teardown integration in ShutdownRuntime()
s32 StartEchoRuntime(s32 thread_priority, s32 cpu_affinity_mask) {
    LOG_INFO(Lib_NpSignaling, "Helpers::StartEchoRuntime thread_priority={} cpu_affinity_mask={}",
             thread_priority, cpu_affinity_mask);

    auto sock = Libraries::Net::sceNetSocket("SceNpSignalingIoctl", 2, 6, 0);
    if (sock < 0) {
        return sock;
    }

    const s32 ioctl_rc = Libraries::Net::sceNetIoctl();
    Libraries::Net::sceNetSocketClose(sock);

    g_echo_probe_word2 = 0;
    g_echo_probe_word3 = 0;
    g_echo_thread_handle = static_cast<u64>(-1);
    g_echo_thread_status = -1;
    return ioctl_rc;
}

// Planned home for FUN_010031a0-style staged shutdown.
//
// Current staged behavior:
// - stops the current HLE ping/receive/dispatch runtime threads through the
//   registered runtime hooks
// - clears the local echo probe/bookkeeping state populated by StartEchoRuntime()
//
// Still missing:
// - callout stop/term
// - socket/control-packet teardown
// - IPC cleanup sequencing
// - dedicated echo thread join/stop once that thread exists locally
void ShutdownRuntime() {
    LOG_INFO(Lib_NpSignaling, "Helpers::ShutdownRuntime");

    if (!g_runtime_hooks_registered) {
        g_echo_probe_word2 = 0;
        g_echo_probe_word3 = 0;
        g_echo_thread_handle = static_cast<u64>(-1);
        g_echo_thread_status = -1;
        return;
    }

    if (g_runtime_hooks.stop_ping != nullptr) {
        g_runtime_hooks.stop_ping();
    }
    if (g_runtime_hooks.stop_receive != nullptr) {
        g_runtime_hooks.stop_receive();
    }
    if (g_runtime_hooks.stop_dispatch != nullptr) {
        g_runtime_hooks.stop_dispatch();
    }

    g_echo_probe_word2 = 0;
    g_echo_probe_word3 = 0;
    g_echo_thread_handle = static_cast<u64>(-1);
    g_echo_thread_status = -1;
}

} // namespace Libraries::Np::NpSignaling::Helpers

namespace Libraries::Np::NpSignaling {

using Libraries::Net::sceNetNtohs;

// File-local thread state.
static Kernel::PthreadT g_dispatch_thread{};
static Kernel::PthreadT g_receive_thread{};
static bool g_receive_stop = false;
static Kernel::PthreadT g_ping_thread{};
static bool g_ping_stop = false;

static void HandleStunEcho(s32 ctx_id, const StunEcho& echo) {
    std::lock_guard<std::mutex> lock(g_mutex);
    const auto it = g_contexts.find(ctx_id);
    if (it == g_contexts.end() || !it->second.active) {
        return;
    }
    NpSignalingContext& ctx = it->second;
    ctx.ext_addr.store(echo.ext_ip);
    ctx.ext_port.store(echo.ext_port);

    LOG_DEBUG(Lib_NpSignaling, "STUN echo: ctxId={} ext_addr={:#x} ext_port={}", ctx_id,
              echo.ext_ip, sceNetNtohs(echo.ext_port));

    ctx.stun_cv.notify_all();
}

static s32 FindContextIdByTagLocked(u32 ctx_tag) {
    const u16 bound_port = static_cast<u16>(ctx_tag & 0xffff);
    for (const auto& [ctx_id, ctx] : g_contexts) {
        if (ctx.active && ctx.bound_port == bound_port) {
            return ctx_id;
        }
    }
    return 0;
}

static void HandleActivatePacket(u32 from_addr, u16 from_port, const ActivatePacket& pkt) {
    // This function is called by the receive thread when we receive an ActivatePacket
    // from a peer (Peer A).  Our role here is Peer B — we are the one being activated.
    //
    // IMPORTANT: We do NOT require a local connection object for Peer A to exist.
    // Peer A may have called ActivateConnection before we did, so our connection table
    // may not yet have an entry for them.  Under the old design this caused a fatal early
    // return; that is gone.  The handshake is now fully server-mediated:
    //
    //   1. Peer A registered an ActivationIntent with the server keyed on
    //      (Peer A's external IP, Peer A's ctx_tag).
    //   2. We received Peer A's ActivatePacket, which embeds ctx_tag in pkt.ctx_tag,
    //      and whose UDP source address gives us Peer A's external IP (from_addr).
    //   3. We send ActivationConfirm to the server with (online_id_me, from_addr, ctx_tag).
    //   4. Server looks up the key, finds Peer A's TCP client, and sends
    //      NpSignalingEvent{event:1} to Peer A — setting Peer A's connection ACTIVE.
    //   5. If no matching intent exists on the server, it silently drops the confirm.
    //
    // We still look for a local connection object (for logging), but we never bail on
    // its absence.

    s32 ctx_id = 0;
    s32 conn_id = 0; // 0 = no local connection object found (non-fatal)
    s32 conn_status = ORBIS_NP_SIGNALING_CONN_STATUS_INACTIVE;
    OrbisNpOnlineId my_online_id{};

    {
        std::lock_guard<std::mutex> lock(g_mutex);

        // Resolve our own context from the packet's ctx_tag.
        ctx_id = FindContextIdByTagLocked(pkt.ctx_tag);
        if (ctx_id == 0) {
            // ctx_tag did not match any bound port; fall back to the single active context
            // if there is exactly one.  This handles the case where the local port differs
            // from what the peer embedded (e.g. after a NAT remap).
            for (const auto& [cid, ctx] : g_contexts) {
                if (!ctx.active)
                    continue;
                if (ctx_id != 0) {
                    ctx_id = 0;
                    break;
                } // more than one — ambiguous, bail
                ctx_id = cid;
            }
            if (ctx_id == 0) {
                LOG_WARNING(Lib_NpSignaling,
                            "HandleActivatePacket: {:#x}:{} tag={:#x} — no matching context, "
                            "cannot identify ourselves, dropping",
                            from_addr, sceNetNtohs(from_port), pkt.ctx_tag);
                return;
            }
            LOG_WARNING(Lib_NpSignaling,
                        "HandleActivatePacket: {:#x}:{} tag={:#x} — fallback to ctxId={}",
                        from_addr, sceNetNtohs(from_port), pkt.ctx_tag, ctx_id);
        }

        // Fetch our own online_id from the context.  This is the peer_online_id the server
        // will report back to Peer A when it delivers NpSignalingEvent{event:1}.
        auto ctx_it = g_contexts.find(ctx_id);
        if (ctx_it != g_contexts.end())
            my_online_id = ctx_it->second.owner_online_id;

        // Opportunistic connection lookup — for logging only.
        // We match on (addr, port) rather than addr alone to avoid false matches when
        // multiple peers share the same host IP (e.g. local test sessions).
        for (auto& [cid, ci] : g_connections) {
            if (ci.ctx_id == ctx_id && ci.addr == from_addr && ci.port == from_port) {
                conn_id = cid;
                conn_status = ci.status;
                break;
            }
        }
    }

    if (conn_id == 0) {
        // No local connection object exists yet — this is the normal case when Peer A
        // activated before we called ActivateConnection on our side.  We still send the
        // ActivationConfirm so the server can fire event=1 to Peer A.
        LOG_INFO(Lib_NpSignaling,
                 "HandleActivatePacket: {:#x}:{} ctxId={} tag={:#x} — no local connection "
                 "object for this peer yet; sending ActivationConfirm regardless",
                 from_addr, sceNetNtohs(from_port), ctx_id, pkt.ctx_tag);
    } else {
        LOG_INFO(Lib_NpSignaling,
                 "HandleActivatePacket: {:#x}:{} ctxId={} connId={} status={} tag={:#x}", from_addr,
                 sceNetNtohs(from_port), ctx_id, conn_id, conn_status, pkt.ctx_tag);
    }

    // Send ActivationConfirm to the server over TCP (binary protocol).
    const u8* ip_bytes = reinterpret_cast<const u8*>(&from_addr);
    char initiator_ip_str[16]{};
    std::snprintf(initiator_ip_str, sizeof(initiator_ip_str), "%u.%u.%u.%u", ip_bytes[0],
                  ip_bytes[1], ip_bytes[2], ip_bytes[3]);

    const std::string me_id_str = OnlineIdFromOnlineId(my_online_id);

    LOG_INFO(
        Lib_NpSignaling,
        "HandleActivatePacket: sending TCP ActivationConfirm me='{}' initiator_ip={} ctx_tag={:#x}",
        me_id_str, initiator_ip_str, pkt.ctx_tag);

    if (!NpMatching2::SendActivationConfirm(me_id_str, initiator_ip_str, pkt.ctx_tag)) {
        LOG_WARNING(Lib_NpSignaling,
                    "HandleActivatePacket: server rejected ActivationConfirm for "
                    "initiator_ip={} ctx_tag={:#x} — dropping packet",
                    initiator_ip_str, pkt.ctx_tag);
        return;
    }

    LOG_INFO(Lib_NpSignaling,
             "HandleActivatePacket: ActivationConfirm accepted — Peer A will receive event=1");
}

static void ReceiveThreadMain() {
    static constexpr u32 kBufSize = 256;
    u8 buf[kBufSize];

    while (!g_receive_stop) {
        u32 from_addr = 0;
        u16 from_port = 0;

        const int rc = Net::P2PSignalingRecvFrom(buf, kBufSize, &from_addr, &from_port);
        if (rc <= 0) {
            std::this_thread::sleep_for(std::chrono::milliseconds(5));
            continue;
        }

        const auto nbytes = static_cast<size_t>(rc);

        // Distinguish StunEcho (6 bytes) from ActivatePacket (32 bytes).
        if (nbytes == sizeof(StunEcho)) {
            s32 ctx_id = 0;
            {
                std::lock_guard<std::mutex> lock(g_mutex);
                for (const auto& [cid, ctx] : g_contexts) {
                    if (ctx.active) {
                        ctx_id = cid;
                        break;
                    }
                }
            }
            if (ctx_id == 0) {
                continue;
            }

            StunEcho echo{};
            std::memcpy(&echo, buf, sizeof(echo));
            HandleStunEcho(ctx_id, echo);
        } else if (nbytes == sizeof(MutualActivated) && buf[0] == 0x03) {
            MutualActivated pkt{};
            std::memcpy(&pkt, buf, sizeof(pkt));
            char peer_id_buf[17]{};
            std::memcpy(peer_id_buf, pkt.online_id_peer, 16);
            const std::string peer_id(peer_id_buf);
            s32 ctx_id = 0;
            s32 conn_id = 0;
            {
                std::lock_guard<std::mutex> lock(g_mutex);
                for (const auto& [cid, ci] : g_connections) {
                    if (OnlineIdEqualsString(ci.online_id, peer_id)) {
                        ctx_id = ci.ctx_id;
                        conn_id = cid;
                        break;
                    }
                }
            }
            if (conn_id != 0) {
                LOG_INFO(Lib_NpSignaling, "MutualActivated: peer='{}' ctxId={} connId={}", peer_id,
                         ctx_id, conn_id);
                DeliverSignalingEventForCtx(ctx_id, conn_id,
                                            ORBIS_NP_SIGNALING_EVENT_MUTUAL_ACTIVATED, 0);
            } else {
                LOG_WARNING(Lib_NpSignaling, "MutualActivated: no connection for peer='{}'",
                            peer_id);
            }
        } else if (nbytes == sizeof(ActivatePacket)) {
            ActivatePacket pkt{};
            std::memcpy(&pkt, buf, sizeof(pkt));
            if (pkt.type != 5) {
                LOG_WARNING(Lib_NpSignaling,
                            "ReceiveThread: unexpected 32-byte packet from {:#x}:{} type={} "
                            "conn={} tag={:#x}",
                            from_addr, sceNetNtohs(from_port), pkt.type, pkt.conn_id, pkt.ctx_tag);
                continue;
            }
            HandleActivatePacket(from_addr, from_port, pkt);
        } else {
            LOG_WARNING(Lib_NpSignaling, "ReceiveThread: unexpected packet size={}", nbytes);
        }
    }
}

static PS4_SYSV_ABI void* ReceiveThreadFunc(void* /*arg*/) {
    ReceiveThreadMain();
    return nullptr;
}

static void StartReceiveThread() {
    g_receive_stop = false;
    if (!g_receive_thread) {
        Kernel::posix_pthread_create(&g_receive_thread, nullptr, ReceiveThreadFunc, nullptr);
    }
}

static void StopReceiveThread() {
    {
        std::lock_guard<std::mutex> lock(g_mutex);
        g_receive_stop = true;
    }
    if (g_receive_thread) {
        Kernel::posix_pthread_join(g_receive_thread, nullptr);
        g_receive_thread = {};
    }
}

static void PingThreadMain() {
    while (!g_ping_stop) {
        if (!Net::P2PTransportIsReady()) {
            std::this_thread::sleep_for(std::chrono::milliseconds(kSigRetryMs));
            continue;
        }

        struct CtxSnapshot {
            s32 ctx_id;
            OrbisNpOnlineId online_id{};
            bool resolved;
        };
        std::vector<CtxSnapshot> contexts;
        {
            std::lock_guard<std::mutex> lock(g_mutex);
            for (const auto& [ctx_id, ctx] : g_contexts) {
                if (ctx.active) {
                    contexts.push_back({ctx_id, ctx.owner_online_id, ctx.ext_addr.load() != 0});
                }
            }
        }

        const u32 server_addr = NpMatching2::GetMmServerAddr();
        const u16 server_port = NpMatching2::GetMmServerUdpPort();

        for (const auto& cs : contexts) {
            if (server_addr == 0 || server_port == 0) {
                break;
            }
            StunPing ping{};
            ping.cmd = 0x01;
            std::memcpy(ping.online_id, cs.online_id.data, ORBIS_NP_ONLINEID_MAX_LENGTH);

            Net::P2PSignalingSendTo(&ping, sizeof(ping), server_addr, server_port);
        }

        const bool any_unresolved = std::any_of(contexts.begin(), contexts.end(),
                                                [](const auto& cs) { return !cs.resolved; });
        const u32 sleep_ms = any_unresolved ? kSigRetryMs : kSigPingMs;
        std::this_thread::sleep_for(std::chrono::milliseconds(sleep_ms));
    }
}

static PS4_SYSV_ABI void* PingThreadFunc(void* /*arg*/) {
    PingThreadMain();
    return nullptr;
}

static void StartPingThread() {
    g_ping_stop = false;
    if (!g_ping_thread) {
        Kernel::posix_pthread_create(&g_ping_thread, nullptr, PingThreadFunc, nullptr);
    }
}

static void StopPingThread() {
    g_ping_stop = true;
    if (g_ping_thread) {
        Kernel::posix_pthread_join(g_ping_thread, nullptr);
        g_ping_thread = {};
    }
}

static void DispatchThreadMain() {
    for (;;) {
        QueuedDispatch dispatch;
        {
            std::unique_lock<std::mutex> lock(g_dispatch_mutex);
            for (;;) {
                if (g_dispatch_stop) {
                    return;
                }
                if (g_dispatch_queue.empty()) {
                    g_dispatch_cv.wait(lock,
                                       [] { return g_dispatch_stop || !g_dispatch_queue.empty(); });
                    continue;
                }
                const auto it = g_dispatch_queue.begin();
                const auto now = std::chrono::steady_clock::now();
                if (it->first > now) {
                    g_dispatch_cv.wait_until(lock, it->first);
                    continue;
                }
                dispatch = std::move(it->second);
                g_dispatch_queue.erase(it);
                break;
            }
        }

        if (dispatch.callback) {
            LOG_INFO(Lib_NpSignaling,
                     "t={} DeliverSignalingEvent: ctxId={} connId={} event={}({}) delay={}ms",
                     NowMs(), dispatch.ctx_id, dispatch.conn_id, dispatch.event_type,
                     SignalingEventName(dispatch.event_type), dispatch.delay_ms);
            LOG_ERROR(Lib_NpSignaling,
                      "INVOKE signaling_cb={} ctxId={} connId={} event={}({}) errorCode={} arg={}",
                      fmt::ptr(reinterpret_cast<void*>(dispatch.callback)), dispatch.ctx_id,
                      dispatch.conn_id, dispatch.event_type,
                      SignalingEventName(dispatch.event_type), dispatch.error_code,
                      fmt::ptr(dispatch.callback_arg));
            dispatch.callback(static_cast<u32>(dispatch.ctx_id), static_cast<u32>(dispatch.conn_id),
                              dispatch.event_type, dispatch.error_code, dispatch.callback_arg);
        }
    }
}

static PS4_SYSV_ABI void* DispatchThreadFunc(void* /*arg*/) {
    DispatchThreadMain();
    return nullptr;
}

static void StartDispatchThread() {
    std::lock_guard<std::mutex> lock(g_dispatch_mutex);
    g_dispatch_stop = false;
    if (!g_dispatch_thread) {
        Kernel::posix_pthread_create(&g_dispatch_thread, nullptr, DispatchThreadFunc, nullptr);
    }
}

static void StopDispatchThread() {
    {
        std::lock_guard<std::mutex> lock(g_dispatch_mutex);
        g_dispatch_stop = true;
        g_dispatch_queue.clear();
    }
    g_dispatch_cv.notify_all();
    if (g_dispatch_thread) {
        Kernel::posix_pthread_join(g_dispatch_thread, nullptr);
        g_dispatch_thread = {};
    }
}

// Staged stand-in for the FUN_0100a660 registration target used by the decomped
// initialize path. The real IPMI/client callback flow is not implemented yet,
// so this is currently just a logged non-null registration target.
static void SignalingIpmiHandlerStub(int* msg) {
    LOG_INFO(Lib_NpSignaling, "SignalingIpmiHandlerStub msg={:p}", fmt::ptr(msg));
}

void RegisterRuntimeHooks() {
    LOG_INFO(Lib_NpSignaling, "RegisterRuntimeHooks");
    Helpers::SetRuntimeHooks({
        .start_dispatch = StartDispatchThread,
        .start_receive = StartReceiveThread,
        .start_ping = StartPingThread,
        .stop_ping = StopPingThread,
        .stop_receive = StopReceiveThread,
        .stop_dispatch = StopDispatchThread,
    });
    Helpers::RegisterIpmiHandler(SignalingIpmiHandlerStub);
}

} // namespace Libraries::Np::NpSignaling
