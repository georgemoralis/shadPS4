// SPDX-FileCopyrightText: Copyright 2024-2026 shadPS4 Emulator Project
// SPDX-License-Identifier: GPL-2.0-or-later

#include <algorithm>
#include <atomic>
#include <cerrno>
#include <chrono>
#include <cstring>
#include <map>
#include <queue>
#include <array>
#include <string>
#include <vector>
#include <common/assert.h>
#include <common/config.h>
#include <common/logging/log.h>
#include "core/libraries/kernel/file_system.h"
#include "core/libraries/kernel/kernel.h"
#include "net.h"
#include "net_error.h"
#include "sockets.h"

#ifndef _WIN32
#include <sys/ioctl.h>
#include <unistd.h>
#endif

namespace Libraries::Net {

// PS4 P2P sockets multiplex multiple virtual ports (sin_vport) over a single UDP port.
// We emulate this with one real UDP socket (the "shared transport") bound to the signaling
// port. All P2PSocket instances share this socket. A 4-byte VPORT header is prepended to
// each outgoing packet:
//   [u16 src_vport (NBO)][u16 dst_vport (NBO)][payload...]
// On receive, we drain the shared socket and demultiplex packets into per-vport queues
// based on the dst_vport field.
static constexpr size_t VPORT_HEADER_SIZE = 4;
// Shared transport is configured non-blocking via FIONBIO, so MSG_DONTWAIT
// is optional. Keep it where available and use 0 on platforms that don't
// define it (e.g. WinSock).
#if defined(MSG_DONTWAIT)
static constexpr int P2P_RECV_NONBLOCK_FLAG = MSG_DONTWAIT;
#else
static constexpr int P2P_RECV_NONBLOCK_FLAG = 0;
#endif

#ifdef _WIN32
#define P2P_ERROR_CASE(errname)                                                                    \
    case (WSA##errname):                                                                           \
        *Libraries::Kernel::__Error() = ORBIS_NET_##errname;                                       \
        return -1;
#else
#define P2P_ERROR_CASE(errname)                                                                    \
    case (errname):                                                                                \
        *Libraries::Kernel::__Error() = ORBIS_NET_##errname;                                       \
        return -1;
#endif

static int ConvertReturnErrorCode(int retval) {
    if (retval < 0) {
#ifdef _WIN32
        switch (WSAGetLastError()) {
#else
        switch (errno) {
#endif
#ifndef _WIN32
            P2P_ERROR_CASE(EPERM)
            P2P_ERROR_CASE(ENOENT)
            P2P_ERROR_CASE(ENOMEM)
            P2P_ERROR_CASE(EEXIST)
            P2P_ERROR_CASE(ENODEV)
            P2P_ERROR_CASE(ENFILE)
            P2P_ERROR_CASE(ENOSPC)
            P2P_ERROR_CASE(EPIPE)
            P2P_ERROR_CASE(ECANCELED)
            P2P_ERROR_CASE(ENODATA)
#endif
            P2P_ERROR_CASE(EINTR)
            P2P_ERROR_CASE(EBADF)
            P2P_ERROR_CASE(EACCES)
            P2P_ERROR_CASE(EFAULT)
            P2P_ERROR_CASE(EINVAL)
            P2P_ERROR_CASE(EMFILE)
            P2P_ERROR_CASE(EWOULDBLOCK)
            P2P_ERROR_CASE(EINPROGRESS)
            P2P_ERROR_CASE(EALREADY)
            P2P_ERROR_CASE(ENOTSOCK)
            P2P_ERROR_CASE(EDESTADDRREQ)
            P2P_ERROR_CASE(EMSGSIZE)
            P2P_ERROR_CASE(EPROTOTYPE)
            P2P_ERROR_CASE(ENOPROTOOPT)
            P2P_ERROR_CASE(EPROTONOSUPPORT)
#if defined(__APPLE__) || defined(_WIN32)
            P2P_ERROR_CASE(EOPNOTSUPP)
#endif
            P2P_ERROR_CASE(EAFNOSUPPORT)
            P2P_ERROR_CASE(EADDRINUSE)
            P2P_ERROR_CASE(EADDRNOTAVAIL)
            P2P_ERROR_CASE(ENETDOWN)
            P2P_ERROR_CASE(ENETUNREACH)
            P2P_ERROR_CASE(ENETRESET)
            P2P_ERROR_CASE(ECONNABORTED)
            P2P_ERROR_CASE(ECONNRESET)
            P2P_ERROR_CASE(ENOBUFS)
            P2P_ERROR_CASE(EISCONN)
            P2P_ERROR_CASE(ENOTCONN)
            P2P_ERROR_CASE(ETIMEDOUT)
            P2P_ERROR_CASE(ECONNREFUSED)
            P2P_ERROR_CASE(ELOOP)
            P2P_ERROR_CASE(ENAMETOOLONG)
            P2P_ERROR_CASE(EHOSTUNREACH)
            P2P_ERROR_CASE(ENOTEMPTY)
        }
        *Libraries::Kernel::__Error() = ORBIS_NET_EINTERNAL;
        return -1;
    }
    return retval;
}

// ====== Shared P2P Transport ======
// One real UDP socket per process, shared by all P2PSocket instances.
// Packets are demultiplexed by the dst_vport field in the VPORT header.
namespace {

std::atomic<u32> g_p2p_bind_log_count{0};
std::atomic<u32> g_p2p_send_log_count{0};
std::atomic<u32> g_p2p_drain_log_count{0};
std::atomic<u32> g_p2p_recv_log_count{0};
std::atomic<bool> g_logged_shared_transport_ready{false};

struct BufferedPacket {
    std::vector<u8> payload;
    sockaddr_in from_addr;
    u16 src_vport;
};

// Signaling packets received on the shared transport that are NOT vport-
// multiplexed game data. Identified by StunEcho size (6 bytes) or the
// 32-byte ActivateConnection packet shape (first u32 == 5).
struct SignalingPacket {
    std::vector<u8> data;
    sockaddr_in from_addr;
};

static constexpr u32 kActivatePacketTypeP2P = 5;

struct SharedTransport {
    net_socket fd =
#ifdef _WIN32
        INVALID_SOCKET;
#else
        -1;
#endif
    u16 bound_port_nbo = 0;
    int refcount = 0;
    std::recursive_mutex mutex;
    std::map<u16, std::queue<BufferedPacket>> vport_queues;
    std::queue<SignalingPacket> signaling_queue;

    bool IsValid() const {
#ifdef _WIN32
        return fd != INVALID_SOCKET;
#else
        return fd >= 0;
#endif
    }

    // Drain all available packets from the shared socket into per-vport queues.
    // Must be called with mutex held.
    void Drain() {
        if (!IsValid())
            return;
        u8 buf[65536];
        for (;;) {
            sockaddr_in from{};
            socklen_t fromlen = sizeof(from);
            int n = ::recvfrom(fd, reinterpret_cast<char*>(buf), sizeof(buf),
                               P2P_RECV_NONBLOCK_FLAG,
                               reinterpret_cast<sockaddr*>(&from), &fromlen);
            if (n <= 0)
                break;

            // Check for signaling packets before vport demux.
            // ActivateConnection packet: 32 bytes, first 4 bytes == 5.
            // StunEcho: exactly 6 bytes (server -> client).
            const auto nbytes = static_cast<size_t>(n);
            bool is_signaling = false;
            if (nbytes == 6) {
                // StunEcho - always signaling.
                is_signaling = true;
            } else if (nbytes == 17 && buf[0] == 0x03) {
                // MutualActivated notification from server.
                is_signaling = true;
            } else if (nbytes == 32) {
                u32 packet_type = 0;
                memcpy(&packet_type, &buf[0], sizeof(packet_type));
                if (packet_type == kActivatePacketTypeP2P) {
                    is_signaling = true;
                }
            }

            if (is_signaling) {
                SignalingPacket sp;
                sp.data.assign(&buf[0], &buf[nbytes]);
                sp.from_addr = from;
                signaling_queue.push(std::move(sp));
                continue;
            }

            if (n < static_cast<int>(VPORT_HEADER_SIZE)) {
                LOG_WARNING(Lib_Net, "P2P: runt packet ({} bytes), dropping", n);
                continue;
            }

            u16 src_vp, dst_vp;
            memcpy(&src_vp, &buf[0], 2);
            memcpy(&dst_vp, &buf[2], 2);

            int plen = n - static_cast<int>(VPORT_HEADER_SIZE);

            if (dst_vp == 0) {
                LOG_WARNING(Lib_Net,
                            "P2P drain: dropping legacy vport 0 packet from {:#x}:{} len={}",
                            ntohl(from.sin_addr.s_addr), ntohs(from.sin_port), plen);
                continue;
            }

            u16 queue_vp = dst_vp;
            auto [queue_it, inserted] =
                vport_queues.try_emplace(queue_vp, std::queue<BufferedPacket>{});
            if (inserted) {
                LOG_WARNING(Lib_Net,
                            "P2P drain: buffering packet for currently-unbound dst_vp={}",
                            ntohs(dst_vp));
            }

            BufferedPacket pkt;
            pkt.payload.assign(&buf[VPORT_HEADER_SIZE], &buf[VPORT_HEADER_SIZE] + plen);
            pkt.from_addr = from;
            pkt.src_vport = src_vp;

            LOG_INFO(Lib_Net, "P2P drain: {} bytes from {:#x}:{} src_vp={} dst_vp={} queue_vp={}",
                     plen, ntohl(from.sin_addr.s_addr), ntohs(from.sin_port), ntohs(src_vp),
                     ntohs(dst_vp), ntohs(queue_vp));
            if (ntohs(src_vp) == 30 || ntohs(dst_vp) == 30 || ntohs(queue_vp) == 30) {
                LOG_ERROR(Lib_Net,
                          "P2P v30 drain: {} bytes from {:#x}:{} src_vp={} dst_vp={} queue_vp={} "
                          "queue_before={}",
                          plen, ntohl(from.sin_addr.s_addr), ntohs(from.sin_port),
                          ntohs(src_vp), ntohs(dst_vp), ntohs(queue_vp), queue_it->second.size());
            }
            auto drain_log_idx = g_p2p_drain_log_count.fetch_add(1, std::memory_order_relaxed);
            if (drain_log_idx < 16) {
                LOG_ERROR(Lib_Net,
                          "P2P drain trace: {} bytes from {:#x}:{} src_vp={} dst_vp={} "
                          "queue_vp={} queue_before={}",
                          plen, ntohl(from.sin_addr.s_addr), ntohs(from.sin_port),
                          ntohs(src_vp), ntohs(dst_vp), ntohs(queue_vp),
                          queue_it->second.size());
            }

            queue_it->second.push(std::move(pkt));
        }
    }
};

static SharedTransport s_transport;

// Initialize the shared transport if not already done.
// Must be called with s_transport.mutex held.
static bool EnsureTransport() {
    if (s_transport.IsValid())
        return true;

    net_socket fd = ::socket(AF_INET, SOCK_DGRAM, 0);
#ifdef _WIN32
    if (fd == INVALID_SOCKET) {
#else
    if (fd < 0) {
#endif
        LOG_ERROR(Lib_Net, "P2P: failed to create shared UDP socket");
        return false;
    }

    // Allow port reuse for restart scenarios
    int one = 1;
    ::setsockopt(fd, SOL_SOCKET, SO_REUSEADDR, reinterpret_cast<const char*>(&one), sizeof(one));
#ifdef SO_REUSEPORT
    ::setsockopt(fd, SOL_SOCKET, SO_REUSEPORT, reinterpret_cast<const char*>(&one), sizeof(one));
#endif

    const u16 sig_port = Config::GetSignalingPort();
    sockaddr_in bind_addr{};
    bind_addr.sin_family = AF_INET;
    bind_addr.sin_port = htons(sig_port);
    bind_addr.sin_addr.s_addr = INADDR_ANY;

    if (::bind(fd, reinterpret_cast<sockaddr*>(&bind_addr), sizeof(bind_addr)) < 0) {
        LOG_ERROR(Lib_Net, "P2P: failed to bind shared socket to port {}, trying ephemeral",
                  sig_port);
        bind_addr.sin_port = 0;
        if (::bind(fd, reinterpret_cast<sockaddr*>(&bind_addr), sizeof(bind_addr)) < 0) {
            LOG_ERROR(Lib_Net, "P2P: failed to bind shared socket to ephemeral port");
#ifdef _WIN32
            closesocket(fd);
#else
            ::close(fd);
#endif
            return false;
        }
    }

    // Read back actual bound port
    sockaddr_in actual{};
    socklen_t actual_len = sizeof(actual);
    ::getsockname(fd, reinterpret_cast<sockaddr*>(&actual), &actual_len);

    // Shared socket is always non-blocking — individual P2PSocket handles blocking semantics
    int nb = 1;
#ifdef _WIN32
    ioctlsocket(fd, FIONBIO, (u_long*)&nb);
#else
    ioctl(fd, FIONBIO, &nb);
#endif

    s_transport.fd = fd;
    s_transport.bound_port_nbo = actual.sin_port;

    LOG_INFO(Lib_Net, "P2P shared transport: bound to port {} (fd={})",
             ntohs(s_transport.bound_port_nbo), fd);
    if (!g_logged_shared_transport_ready.exchange(true, std::memory_order_relaxed)) {
        LOG_ERROR(Lib_Net, "P2P shared transport ready: port={} fd={}",
                  ntohs(s_transport.bound_port_nbo), fd);
    }
    return true;
}

} // anonymous namespace

// ====== P2PSocket implementation ======

P2PSocket::P2PSocket(int domain, int type, int protocol) : Socket(domain, type, protocol) {
    if (type == ORBIS_NET_SOCK_STREAM_P2P) {
        // STREAM_P2P: own a real TCP socket — no shared UDP transport involvement.
        is_stream_ = true;
        sock_ = ::socket(AF_INET, SOCK_STREAM, 0);
#ifdef _WIN32
        if (sock_ == INVALID_SOCKET) {
#else
        if (sock_ < 0) {
#endif
            LOG_ERROR(Lib_Net, "P2P STREAM_P2P: failed to create TCP socket");
            return;
        }
        int reuse = 1;
        ::setsockopt(sock_, SOL_SOCKET, SO_REUSEADDR, (char*)&reuse, sizeof(reuse));
        // P2P HACK: Hard-close (RST) instead of graceful FIN to avoid TIME_WAIT
        // holding the port on rapid reconnects. P2P is not fully implemented.
        struct linger lin = {1, 0};
        ::setsockopt(sock_, SOL_SOCKET, SO_LINGER, (const char*)&lin, sizeof(lin));
        // NOTE: listening socket stays blocking so accept() works normally.
        // Accepted/connected sockets are set non-blocking individually.
        LOG_INFO(Lib_Net, "P2P STREAM_P2P socket created (TCP fd={})", (int)sock_);
    } else {
        // DGRAM_P2P: reference the shared UDP transport.
        std::scoped_lock tlock{s_transport.mutex};
        if (EnsureTransport()) {
            sock_ = s_transport.fd;
            s_transport.refcount++;
            LOG_INFO(Lib_Net, "P2P socket handle created (shared fd={}, refcount={})", sock_,
                     s_transport.refcount);
        } else {
#ifdef _WIN32
            sock_ = INVALID_SOCKET;
#else
            sock_ = -1;
#endif
            LOG_ERROR(Lib_Net, "P2P socket handle creation failed (shared transport unavailable)");
        }
    }
}

// Internal constructor for accepted TCP connections.
P2PSocket::P2PSocket(net_socket accepted_fd, u16 peer_vport)
    : Socket(AF_INET, ORBIS_NET_SOCK_STREAM_P2P, 0) {
    is_stream_ = true;
    is_connected_ = true;
    sock_ = accepted_fd;
    bound_vport_ = peer_vport;
    // Accepted sockets inherit blocking mode — make non-blocking to match.
    int nb = 1;
#ifdef _WIN32
    ioctlsocket(sock_, FIONBIO, (u_long*)&nb);
#else
    ioctl(sock_, FIONBIO, &nb);
#endif
    // P2P HACK: Hard-close to avoid TIME_WAIT on reconnect. P2P is not fully implemented.
    struct linger lin = {1, 0};
    ::setsockopt(sock_, SOL_SOCKET, SO_LINGER, (const char*)&lin, sizeof(lin));
    LOG_INFO(Lib_Net, "P2P STREAM_P2P accepted socket (TCP fd={} peer_vport={})",
             (int)accepted_fd, ntohs(peer_vport));
}

bool P2PSocket::IsValid() const {
#ifdef _WIN32
    return sock_ != INVALID_SOCKET;
#else
    return sock_ >= 0;
#endif
}

int P2PSocket::Close() {
    std::scoped_lock lock{m_mutex};
    if (!IsValid())
        return 0;

    LOG_INFO(Lib_Net, "Closing P2P socket (vport={} stream={})", ntohs(bound_vport_), is_stream_);

    if (is_stream_) {
#ifdef _WIN32
        closesocket(sock_);
        sock_ = INVALID_SOCKET;
#else
        ::close(sock_);
        sock_ = -1;
#endif
        bound_vport_ = 0;
        return 0;
    }

    {
        std::scoped_lock tlock{s_transport.mutex};
        s_transport.vport_queues.erase(bound_vport_);
        s_transport.refcount--;
        if (s_transport.refcount <= 0 && s_transport.IsValid()) {
            LOG_INFO(Lib_Net, "P2P shared transport: closing (fd={})", s_transport.fd);
#ifdef _WIN32
            closesocket(s_transport.fd);
            s_transport.fd = INVALID_SOCKET;
#else
            ::close(s_transport.fd);
            s_transport.fd = -1;
#endif
            s_transport.bound_port_nbo = 0;
            s_transport.refcount = 0;
            s_transport.vport_queues.clear();
        }
    }

#ifdef _WIN32
    sock_ = INVALID_SOCKET;
#else
    sock_ = -1;
#endif
    bound_vport_ = 0;
    return 0;
}

int P2PSocket::Bind(const OrbisNetSockaddr* addr, u32 addrlen) {
    std::scoped_lock lock{m_mutex};

    if (!IsValid()) {
        *Libraries::Kernel::__Error() = ORBIS_NET_EBADF;
        return -1;
    }

    const auto* orbis_addr = reinterpret_cast<const OrbisNetSockaddrIn*>(addr);
    bound_vport_ = orbis_addr->sin_vport;

    if (is_stream_) {
        // P2P HACK: STREAM_P2P bind port selection. P2P is not fully implemented.
        // - STREAM_P2P_USE_SIGNALING_PORT=1: bind to the signaling port (same-machine testing).
        // - Set to 0 for external/LAN play (uses vport instead).
#define STREAM_P2P_USE_SIGNALING_PORT 1
#if STREAM_P2P_USE_SIGNALING_PORT
        uint16_t tcp_port = GetP2PBoundPort();
#else
        uint16_t tcp_port = ntohs(bound_vport_);
#endif
        sockaddr_in sa{};
        sa.sin_family = AF_INET;
        sa.sin_addr.s_addr = INADDR_ANY;
        sa.sin_port = htons(tcp_port);
        if (::bind(sock_, reinterpret_cast<sockaddr*>(&sa), sizeof(sa)) < 0) {
#ifdef _WIN32
            LOG_ERROR(Lib_Net, "P2P STREAM_P2P bind failed: port={} err={}", tcp_port, WSAGetLastError());
#else
            LOG_ERROR(Lib_Net, "P2P STREAM_P2P bind failed: port={} err={}", tcp_port, errno);
#endif
            *Libraries::Kernel::__Error() = ORBIS_NET_EADDRINUSE;
            return -1;
        }
        LOG_INFO(Lib_Net, "P2P STREAM_P2P bound: TCP port={} (signaling port, vport={})",
                 tcp_port, ntohs(bound_vport_));
        return 0;
    }

    LOG_INFO(Lib_Net, "P2P bind: game requested port={} vport={}, using shared signaling transport",
             ntohs(orbis_addr->sin_port), ntohs(bound_vport_));
    auto bind_log_idx = g_p2p_bind_log_count.fetch_add(1, std::memory_order_relaxed);
    if (bind_log_idx < 8) {
        LOG_ERROR(Lib_Net,
                  "P2P bind trace: requested_port={} requested_vport={} shared_port={} "
                  "socket_fd={}",
                  ntohs(orbis_addr->sin_port), ntohs(bound_vport_),
                  ntohs(s_transport.bound_port_nbo), sock_);
    }

    {
        std::scoped_lock tlock{s_transport.mutex};
        s_transport.vport_queues[bound_vport_]; // create receive queue for this vport
    }

    LOG_INFO(Lib_Net, "P2P socket bound: native port={} vport={}",
             ntohs(s_transport.bound_port_nbo), ntohs(bound_vport_));
    return 0;
}

int P2PSocket::SendPacket(const void* msg, u32 len, int flags, const OrbisNetSockaddr* to,
                          u32 tolen) {
    std::scoped_lock lock{m_mutex};
    if (!IsValid()) {
        *Libraries::Kernel::__Error() = ORBIS_NET_EBADF;
        return -1;
    }

    if (is_stream_) {
        // STREAM_P2P: TCP send (ignore destination address — connection is established)
        int res = ::send(sock_, reinterpret_cast<const char*>(msg), len, 0);
        if (res < 0) {
            *Libraries::Kernel::__Error() = ORBIS_NET_EWOULDBLOCK;
            return -1;
        }
        return res;
    }

    if (to == nullptr) {
        *Libraries::Kernel::__Error() = ORBIS_NET_EDESTADDRREQ;
        return -1;
    }

    const auto* orbis_to = reinterpret_cast<const OrbisNetSockaddrIn*>(to);

    OrbisNetSockaddrIn resolved_to = *orbis_to;
    if (orbis_to->sin_addr == 0 && orbis_to->sin_port == 0) {
        LOG_WARNING(Lib_Net,
                    "P2P sendto: refusing unresolved 0.0.0.0:0 destination without legacy probe fallback");
        *Libraries::Kernel::__Error() = ORBIS_NET_EDESTADDRREQ;
        return -1;
    }
    orbis_to = &resolved_to;

    // Build framed packet: [src_vport(2)][dst_vport(2)][payload(len)]
    std::vector<u8> framed(VPORT_HEADER_SIZE + len);
    memcpy(&framed[0], &bound_vport_, sizeof(u16));
    memcpy(&framed[2], &orbis_to->sin_vport, sizeof(u16));
    memcpy(&framed[VPORT_HEADER_SIZE], msg, len);

    // Build native destination (addr + port only; vport is in the header)
    sockaddr_in native_to{};
    native_to.sin_family = AF_INET;
    native_to.sin_port = orbis_to->sin_port;
    memcpy(&native_to.sin_addr, &orbis_to->sin_addr, 4);

    int native_flags = 0;
#ifndef _WIN32
    if (flags & ORBIS_NET_MSG_DONTWAIT)
        native_flags |= MSG_DONTWAIT;
#endif

    net_socket fd;
    {
        std::scoped_lock tlock{s_transport.mutex};
        fd = s_transport.fd;
    }

    LOG_INFO(Lib_Net,
             "P2P sendto: attempting {} bytes to addr={:#x} ({}.{}.{}.{}) port={} vport={} "
             "src_vport={} fd={} native_flags={:#x}",
             len, ntohl(orbis_to->sin_addr),
             (ntohl(orbis_to->sin_addr) >> 24) & 0xff,
             (ntohl(orbis_to->sin_addr) >> 16) & 0xff,
             (ntohl(orbis_to->sin_addr) >> 8) & 0xff,
             ntohl(orbis_to->sin_addr) & 0xff,
             ntohs(orbis_to->sin_port), ntohs(orbis_to->sin_vport),
             ntohs(bound_vport_), fd, native_flags);
    if (ntohs(bound_vport_) == 30 || ntohs(orbis_to->sin_vport) == 30) {
        LOG_ERROR(Lib_Net,
                  "P2P v30 send: len={} dst_addr={:#x} dst_port={} dst_vport={} src_vport={} "
                  "fd={}",
                  len, ntohl(orbis_to->sin_addr), ntohs(orbis_to->sin_port),
                  ntohs(orbis_to->sin_vport), ntohs(bound_vport_), fd);
    }
    auto send_log_idx = g_p2p_send_log_count.fetch_add(1, std::memory_order_relaxed);
    if (send_log_idx < 16) {
        LOG_ERROR(Lib_Net,
                  "P2P send trace: len={} dst_addr={:#x} dst_port={} dst_vport={} "
                  "src_vport={} fd={}",
                  len, ntohl(orbis_to->sin_addr), ntohs(orbis_to->sin_port),
                  ntohs(orbis_to->sin_vport), ntohs(bound_vport_), fd);
    }
    int res = ::sendto(fd, reinterpret_cast<const char*>(framed.data()), framed.size(),
                       native_flags, reinterpret_cast<sockaddr*>(&native_to), sizeof(native_to));
    if (res < 0) {
        int err = errno;
        LOG_ERROR(Lib_Net,
                  "P2P sendto FAILED: errno={} ({}) dst_addr={:#x} ({}.{}.{}.{}) "
                  "dst_port={} dst_vport={} src_vport={} len={} fd={}",
                  err, strerror(err),
                  ntohl(orbis_to->sin_addr),
                  (ntohl(orbis_to->sin_addr) >> 24) & 0xff,
                  (ntohl(orbis_to->sin_addr) >> 16) & 0xff,
                  (ntohl(orbis_to->sin_addr) >> 8) & 0xff,
                  ntohl(orbis_to->sin_addr) & 0xff,
                  ntohs(orbis_to->sin_port), ntohs(orbis_to->sin_vport),
                  ntohs(bound_vport_), len, fd);
        return ConvertReturnErrorCode(res);
    }

    // Return payload bytes sent (excluding VPORT header)
    int payload_sent = res - static_cast<int>(VPORT_HEADER_SIZE);
    if (payload_sent < 0)
        payload_sent = 0;

    LOG_INFO(Lib_Net, "P2P sendto OK: {} bytes to {:#x}:{} vport={}", payload_sent,
             ntohl(orbis_to->sin_addr), ntohs(orbis_to->sin_port), ntohs(orbis_to->sin_vport));
    if (ntohs(bound_vport_) == 30 || ntohs(orbis_to->sin_vport) == 30) {
        LOG_ERROR(Lib_Net,
                  "P2P v30 send OK: payload={} dst_addr={:#x} dst_port={} dst_vport={} "
                  "src_vport={}",
                  payload_sent, ntohl(orbis_to->sin_addr), ntohs(orbis_to->sin_port),
                  ntohs(orbis_to->sin_vport), ntohs(bound_vport_));
    }
    if (send_log_idx < 16) {
        LOG_ERROR(Lib_Net, "P2P send OK trace: payload={} dst_addr={:#x} dst_port={} dst_vport={}",
                  payload_sent, ntohl(orbis_to->sin_addr), ntohs(orbis_to->sin_port),
                  ntohs(orbis_to->sin_vport));
    }

    return payload_sent;
}

int P2PSocket::ReceivePacket(void* buf, u32 len, int flags, OrbisNetSockaddr* from,
                              u32* fromlen) {
    std::scoped_lock lock{receive_mutex};
    if (!IsValid()) {
        *Libraries::Kernel::__Error() = ORBIS_NET_EBADF;
        return -1;
    }

    if (is_stream_) {
        // STREAM_P2P: TCP recv — socket is non-blocking via FIONBIO.
        int tcp_flags = (flags & ORBIS_NET_MSG_PEEK) ? MSG_PEEK : 0;
        int res = ::recv(sock_, reinterpret_cast<char*>(buf), len, tcp_flags);
        if (res > 0) {
            return res;
        }

        // TCP had no data (EWOULDBLOCK) or connection died (0 / error).
        // P2P HACK: Fallback to DGRAM_P2P vport queues when STREAM has no data.
        // Both socket types should share one transport but P2P is not fully
        // implemented. Set STREAM_P2P_DGRAM_FALLBACK to 0 to disable.
#define STREAM_P2P_DGRAM_FALLBACK 1
#if STREAM_P2P_DGRAM_FALLBACK
        {
            std::scoped_lock tlock{s_transport.mutex};
            s_transport.Drain();
            for (auto& [vp, queue] : s_transport.vport_queues) {
                if (!queue.empty()) {
                    auto& pkt = queue.front();
                    int copy_len =
                        std::min(static_cast<int>(pkt.payload.size()), static_cast<int>(len));
                    memcpy(buf, pkt.payload.data(), copy_len);

                    if (from != nullptr) {
                        auto* orbis_from = reinterpret_cast<OrbisNetSockaddrIn*>(from);
                        memset(orbis_from, 0, sizeof(OrbisNetSockaddrIn));
                        orbis_from->sin_len = sizeof(OrbisNetSockaddrIn);
                        orbis_from->sin_family = AF_INET;
                        orbis_from->sin_port = pkt.from_addr.sin_port;
                        memcpy(&orbis_from->sin_addr, &pkt.from_addr.sin_addr, 4);
                        orbis_from->sin_vport = pkt.src_vport;
                        if (fromlen)
                            *fromlen = sizeof(OrbisNetSockaddrIn);
                    }

                    LOG_INFO(Lib_Net,
                             "P2P STREAM_P2P dgram fallback: {} bytes from vport queue vp={}",
                             copy_len, ntohs(vp));

                    if (!(flags & ORBIS_NET_MSG_PEEK)) {
                        queue.pop();
                    }
                    return copy_len;
                }
            }
        }
#endif // STREAM_P2P_DGRAM_FALLBACK

        // Nothing from TCP or DGRAM queues — report the original TCP result.
        if (res == 0) {
            *Libraries::Kernel::__Error() = ORBIS_NET_ECONNRESET;
            return -1;
        }
#ifdef _WIN32
        int err = WSAGetLastError();
        if (err == WSAEWOULDBLOCK)
            *Libraries::Kernel::__Error() = ORBIS_NET_EWOULDBLOCK;
        else
            *Libraries::Kernel::__Error() = ORBIS_NET_ECONNRESET;
#else
        if (errno == EAGAIN || errno == EWOULDBLOCK)
            *Libraries::Kernel::__Error() = ORBIS_NET_EWOULDBLOCK;
        else
            *Libraries::Kernel::__Error() = ORBIS_NET_ECONNRESET;
#endif
        return -1;
    }

    std::scoped_lock tlock{s_transport.mutex};

    // Drain any available packets from the shared socket into per-vport queues
    s_transport.Drain();

    // Check our vport queue
    auto it = s_transport.vport_queues.find(bound_vport_);
    if (it == s_transport.vport_queues.end() || it->second.empty()) {
        if (ntohs(bound_vport_) == 30) {
            LOG_ERROR(Lib_Net, "P2P v30 recv would block: queue_missing={} queue_size={}",
                      it == s_transport.vport_queues.end(),
                      it == s_transport.vport_queues.end() ? 0 : it->second.size());
        }
        *Libraries::Kernel::__Error() = ORBIS_NET_EWOULDBLOCK;
        return -1;
    }

    auto& pkt = it->second.front();

    // Copy payload to user buffer
    int copy_len = std::min(static_cast<int>(pkt.payload.size()), static_cast<int>(len));
    memcpy(buf, pkt.payload.data(), copy_len);

    // Fill in source address with vport from the VPORT header
    if (from != nullptr) {
        auto* orbis_from = reinterpret_cast<OrbisNetSockaddrIn*>(from);
        memset(orbis_from, 0, sizeof(OrbisNetSockaddrIn));
        orbis_from->sin_len = sizeof(OrbisNetSockaddrIn);
        orbis_from->sin_family = AF_INET;
        orbis_from->sin_port = pkt.from_addr.sin_port;
        memcpy(&orbis_from->sin_addr, &pkt.from_addr.sin_addr, 4);
        orbis_from->sin_vport = pkt.src_vport;

        if (fromlen) {
            *fromlen = sizeof(OrbisNetSockaddrIn);
        }
    }

    LOG_INFO(Lib_Net, "P2P recvfrom: {} bytes from {:#x}:{} src_vp={} dst_vp={}", copy_len,
             ntohl(pkt.from_addr.sin_addr.s_addr), ntohs(pkt.from_addr.sin_port),
             ntohs(pkt.src_vport), ntohs(bound_vport_));
    if (ntohs(pkt.src_vport) == 30 || ntohs(bound_vport_) == 30) {
        LOG_ERROR(Lib_Net,
                  "P2P v30 recv: {} bytes from {:#x}:{} src_vp={} dst_vp={} "
                  "peek={} remaining_before_pop={}",
                  copy_len, ntohl(pkt.from_addr.sin_addr.s_addr), ntohs(pkt.from_addr.sin_port),
                  ntohs(pkt.src_vport), ntohs(bound_vport_),
                  (flags & ORBIS_NET_MSG_PEEK) != 0, it->second.size());
    }
    auto recv_log_idx = g_p2p_recv_log_count.fetch_add(1, std::memory_order_relaxed);
    if (recv_log_idx < 32) {
        LOG_ERROR(Lib_Net,
                  "P2P recv trace: {} bytes from {:#x}:{} src_vp={} dst_vp={} "
                  "peek={} remaining_before_pop={}",
                  copy_len, ntohl(pkt.from_addr.sin_addr.s_addr), ntohs(pkt.from_addr.sin_port),
                  ntohs(pkt.src_vport), ntohs(bound_vport_),
                  (flags & ORBIS_NET_MSG_PEEK) != 0, it->second.size());
    }

    if (!(flags & ORBIS_NET_MSG_PEEK)) {
        it->second.pop();
    }

    return copy_len;
}

int P2PSocket::SetSocketOptions(int level, int optname, const void* optval, u32 optlen) {
    std::scoped_lock lock{m_mutex};

    if (level == ORBIS_NET_SOL_SOCKET) {
        switch (optname) {
        case ORBIS_NET_SO_NBIO: {
            if (optlen < sizeof(int)) {
                *Libraries::Kernel::__Error() = ORBIS_NET_EINVAL;
                return -1;
            }
            memcpy(&sockopt_so_nbio_, optval, sizeof(int));
            LOG_INFO(Lib_Net, "P2P SO_NBIO = {}", sockopt_so_nbio_);
            // Shared socket is always non-blocking; we handle blocking semantics per-socket
            return 0;
        }
        case ORBIS_NET_SO_BROADCAST: {
            if (!IsValid())
                return 0;
            std::scoped_lock tlock{s_transport.mutex};
            return ConvertReturnErrorCode(
                setsockopt(s_transport.fd, SOL_SOCKET, SO_BROADCAST, (const char*)optval, optlen));
        }
        case ORBIS_NET_SO_SNDBUF: {
            if (!IsValid())
                return 0;
            std::scoped_lock tlock{s_transport.mutex};
            return ConvertReturnErrorCode(
                setsockopt(s_transport.fd, SOL_SOCKET, SO_SNDBUF, (const char*)optval, optlen));
        }
        case ORBIS_NET_SO_RCVBUF: {
            if (!IsValid())
                return 0;
            std::scoped_lock tlock{s_transport.mutex};
            return ConvertReturnErrorCode(
                setsockopt(s_transport.fd, SOL_SOCKET, SO_RCVBUF, (const char*)optval, optlen));
        }
        case ORBIS_NET_SO_REUSEADDR: {
            // Shared socket already has SO_REUSEADDR set
            LOG_INFO(Lib_Net, "P2P SO_REUSEADDR (shared transport, no-op)");
            return 0;
        }
        case ORBIS_NET_SO_SNDTIMEO:
        case ORBIS_NET_SO_RCVTIMEO: {
            if (is_stream_ && IsValid()) {
                // Apply timeout directly to owned TCP socket
                int native_opt = (optname == ORBIS_NET_SO_SNDTIMEO) ? SO_SNDTIMEO : SO_RCVTIMEO;
                LOG_INFO(Lib_Net, "P2P SO_{} (STREAM_P2P TCP)",
                         (optname == ORBIS_NET_SO_SNDTIMEO) ? "SNDTIMEO" : "RCVTIMEO");
                return ConvertReturnErrorCode(
                    ::setsockopt(sock_, SOL_SOCKET, native_opt, (const char*)optval, optlen));
            }
            // Shared socket is always non-blocking; timeouts handled per-socket if needed
            LOG_INFO(Lib_Net, "P2P SO_{} (shared transport, stored locally)",
                     (optname == ORBIS_NET_SO_SNDTIMEO) ? "SNDTIMEO" : "RCVTIMEO");
            return 0;
        }
        default:
            LOG_WARNING(Lib_Net, "P2P setsockopt: unhandled SOL_SOCKET option {:#x}", optname);
            return 0;
        }
    }

    LOG_WARNING(Lib_Net, "P2P setsockopt: unhandled level={} optname={:#x}", level, optname);
    return 0;
}

int P2PSocket::GetSocketOptions(int level, int optname, void* optval, u32* optlen) {
    std::scoped_lock lock{m_mutex};

    if (level == ORBIS_NET_SOL_SOCKET) {
        switch (optname) {
        case ORBIS_NET_SO_NBIO: {
            if (*optlen < sizeof(int)) {
                *optlen = sizeof(int);
                *Libraries::Kernel::__Error() = ORBIS_NET_EFAULT;
                return -1;
            }
            *optlen = sizeof(int);
            *(int*)optval = sockopt_so_nbio_;
            return 0;
        }
        case ORBIS_NET_SO_ERROR: {
            if (!IsValid()) {
                *(int*)optval = 0;
                *optlen = sizeof(int);
                return 0;
            }
            std::scoped_lock tlock{s_transport.mutex};
            socklen_t optlen_temp = *optlen;
            auto retval = ConvertReturnErrorCode(
                getsockopt(s_transport.fd, SOL_SOCKET, SO_ERROR, (char*)optval, &optlen_temp));
            *optlen = optlen_temp;
            return retval;
        }
        case ORBIS_NET_SO_TYPE: {
            if (*optlen >= sizeof(int)) {
                *(int*)optval = socket_type;
                *optlen = sizeof(int);
            }
            return 0;
        }
        default:
            LOG_WARNING(Lib_Net, "P2P getsockopt: unhandled SOL_SOCKET option {:#x}", optname);
            if (*optlen >= sizeof(int)) {
                *(int*)optval = 0;
                *optlen = sizeof(int);
            }
            return 0;
        }
    }

    LOG_WARNING(Lib_Net, "P2P getsockopt: unhandled level={} optname={:#x}", level, optname);
    return 0;
}

int P2PSocket::GetSocketAddress(OrbisNetSockaddr* name, u32* namelen) {
    std::scoped_lock lock{m_mutex};
    if (!IsValid() || name == nullptr) {
        return 0;
    }

    // Return the shared transport's bound address with this socket's vport
    auto* orbis_addr = reinterpret_cast<OrbisNetSockaddrIn*>(name);
    memset(orbis_addr, 0, sizeof(OrbisNetSockaddrIn));
    orbis_addr->sin_len = sizeof(OrbisNetSockaddrIn);
    orbis_addr->sin_family = AF_INET;

    {
        std::scoped_lock tlock{s_transport.mutex};
        sockaddr_in native_addr{};
        socklen_t native_len = sizeof(native_addr);
        if (::getsockname(s_transport.fd, reinterpret_cast<sockaddr*>(&native_addr), &native_len) ==
            0) {
            orbis_addr->sin_port = native_addr.sin_port;
            memcpy(&orbis_addr->sin_addr, &native_addr.sin_addr, 4);
        }
    }

    orbis_addr->sin_vport = bound_vport_;

    if (namelen) {
        *namelen = sizeof(OrbisNetSockaddrIn);
    }

    return 0;
}

int P2PSocket::Connect(const OrbisNetSockaddr* addr, u32 namelen) {
    if (is_stream_) {
        const auto* orbis_addr = reinterpret_cast<const OrbisNetSockaddrIn*>(addr);
        // sin_vport carries the peer's P2P signaling port (what the host bound its TCP to).
        // sin_port is the host's game-level STREAM_P2P port — not the TCP listener port.
#if STREAM_P2P_USE_SIGNALING_PORT
        uint16_t tcp_port = ntohs(orbis_addr->sin_vport); // peer's signaling port
#else
        uint16_t tcp_port = ntohs(orbis_addr->sin_vport); // peer's vport (e.g. 3658)
        if (tcp_port == 0) tcp_port = ntohs(orbis_addr->sin_port);
#endif
        // Bind to our signaling port so the peer sees our known port, not an ephemeral one.
        {
            sockaddr_in local_sa{};
            local_sa.sin_family = AF_INET;
            local_sa.sin_addr.s_addr = INADDR_ANY;
            local_sa.sin_port = htons(GetP2PConfiguredPort());
            int reuse = 1;
            ::setsockopt(sock_, SOL_SOCKET, SO_REUSEADDR, (const char*)&reuse, sizeof(reuse));
            ::bind(sock_, reinterpret_cast<sockaddr*>(&local_sa), sizeof(local_sa));
        }

        sockaddr_in sa{};
        sa.sin_family = AF_INET;
        sa.sin_port = htons(tcp_port);
        std::memcpy(&sa.sin_addr, &orbis_addr->sin_addr, 4);
        LOG_INFO(Lib_Net, "P2P STREAM_P2P connect to {}:{} from port {} (vport={})",
                 ntohl(orbis_addr->sin_addr), tcp_port, GetP2PConfiguredPort(),
                 ntohs(orbis_addr->sin_vport));
        int res = ::connect(sock_, reinterpret_cast<sockaddr*>(&sa), sizeof(sa));
        if (res < 0) {
#ifdef _WIN32
            int err = WSAGetLastError();
            if (err == WSAEWOULDBLOCK || err == WSAEINPROGRESS) {
                *Libraries::Kernel::__Error() = ORBIS_NET_EINPROGRESS;
                return -1;
            }
            LOG_ERROR(Lib_Net, "P2P STREAM_P2P connect failed: err={}", err);
#else
            if (errno == EINPROGRESS) {
                *Libraries::Kernel::__Error() = ORBIS_NET_EINPROGRESS;
                return -1;
            }
            LOG_ERROR(Lib_Net, "P2P STREAM_P2P connect failed: err={}", errno);
#endif
            *Libraries::Kernel::__Error() = ORBIS_NET_ECONNREFUSED;
            return -1;
        }
        is_connected_ = true;
        // Set connected socket non-blocking so DGRAM fallback in recv works.
        {
            int nb = 1;
#ifdef _WIN32
            ioctlsocket(sock_, FIONBIO, (u_long*)&nb);
#else
            ioctl(sock_, FIONBIO, &nb);
#endif
        }
        return 0;
    }
    // P2P UDP sockets don't truly connect — connectionless datagram
    LOG_INFO(Lib_Net, "P2P Connect called (no-op for UDP P2P)");
    return 0;
}

int P2PSocket::Listen(int backlog) {
    if (is_stream_) {
        int res = ::listen(sock_, backlog > 0 ? backlog : 5);
        if (res < 0) {
#ifdef _WIN32
            LOG_ERROR(Lib_Net, "P2P STREAM_P2P listen failed: err={}", WSAGetLastError());
#else
            LOG_ERROR(Lib_Net, "P2P STREAM_P2P listen failed: err={}", errno);
#endif
            return -1;
        }
        LOG_INFO(Lib_Net, "P2P STREAM_P2P listening on TCP fd={} vport={}", (int)sock_,
                 ntohs(bound_vport_));
        return 0;
    }
    LOG_WARNING(Lib_Net, "P2P Listen called (not applicable for DGRAM)");
    return 0;
}

int P2PSocket::SendMessage(const OrbisNetMsghdr* msg, int flags) {
    LOG_ERROR(Lib_Net, "(STUBBED) P2P SendMessage called");
    *Libraries::Kernel::__Error() = ORBIS_NET_EAGAIN;
    return -1;
}

int P2PSocket::ReceiveMessage(OrbisNetMsghdr* msg, int flags) {
    LOG_ERROR(Lib_Net, "(STUBBED) P2P ReceiveMessage called");
    *Libraries::Kernel::__Error() = ORBIS_NET_EAGAIN;
    return -1;
}

SocketPtr P2PSocket::Accept(OrbisNetSockaddr* addr, u32* addrlen) {
    if (is_stream_) {
        sockaddr_in peer_addr{};
        socklen_t peer_len = sizeof(peer_addr);
        net_socket accepted = ::accept(sock_, reinterpret_cast<sockaddr*>(&peer_addr), &peer_len);
#ifdef _WIN32
        if (accepted == INVALID_SOCKET) {
            int err = WSAGetLastError();
            if (err == WSAEWOULDBLOCK) {
                *Libraries::Kernel::__Error() = ORBIS_NET_EWOULDBLOCK;
            } else {
                LOG_ERROR(Lib_Net, "P2P STREAM_P2P accept failed: err={}", err);
                *Libraries::Kernel::__Error() = ORBIS_NET_ECONNABORTED;
            }
            return nullptr;
        }
#else
        if (accepted < 0) {
            if (errno == EAGAIN || errno == EWOULDBLOCK) {
                *Libraries::Kernel::__Error() = ORBIS_NET_EWOULDBLOCK;
            } else {
                LOG_ERROR(Lib_Net, "P2P STREAM_P2P accept failed: err={}", errno);
                *Libraries::Kernel::__Error() = ORBIS_NET_ECONNABORTED;
            }
            return nullptr;
        }
#endif
        LOG_INFO(Lib_Net, "P2P STREAM_P2P accepted TCP connection from {}:{} fd={}",
                 ntohl(peer_addr.sin_addr.s_addr), ntohs(peer_addr.sin_port), (int)accepted);
        if (addr) {
            auto* orbis_from = reinterpret_cast<OrbisNetSockaddrIn*>(addr);
            std::memset(orbis_from, 0, sizeof(OrbisNetSockaddrIn));
            orbis_from->sin_len = sizeof(OrbisNetSockaddrIn);
            orbis_from->sin_family = AF_INET;
            orbis_from->sin_port = peer_addr.sin_port;
            std::memcpy(&orbis_from->sin_addr, &peer_addr.sin_addr, 4);
            orbis_from->sin_vport = peer_addr.sin_port; // peer's signaling port
            if (addrlen) *addrlen = sizeof(OrbisNetSockaddrIn);
        }
        return std::make_shared<P2PSocket>(accepted, bound_vport_);
    }
    LOG_ERROR(Lib_Net, "P2P Accept called (not applicable for DGRAM)");
    *Libraries::Kernel::__Error() = ORBIS_NET_EOPNOTSUPP;
    return nullptr;
}

int P2PSocket::GetPeerName(OrbisNetSockaddr* addr, u32* namelen) {
    LOG_WARNING(Lib_Net, "(STUBBED) P2P GetPeerName called");
    return 0;
}

int P2PSocket::fstat(Libraries::Kernel::OrbisKernelStat* sb) {
    if (sb) {
        sb->st_mode = 0000777u | 0140000u;
    }
    return 0;
}

bool P2PSocket::HasQueuedData() {
    std::scoped_lock tlock{s_transport.mutex};
    auto it = s_transport.vport_queues.find(bound_vport_);
    return it != s_transport.vport_queues.end() && !it->second.empty();
}

int P2PSocket::Abort(int flags) {
    // Flushing the per-vport receive queue on abort ensures that:
    //  1. sceNetEpollWait stops reporting data ready for this socket.
    //  2. Stale packets from the old P2P session don't bleed into a new one
    //     when the same vport is reused after reconnect.
    // We do NOT close the shared transport fd here — that is Close()'s job.
    if (!IsValid()) {
        return 0;
    }

    // P2P HACK: shutdown the TCP socket to unblock any thread blocked in
    // accept() or recv(). P2P is not fully implemented — sceNetSocketAbort
    // should handle this but we use socket shutdown as a workaround.
    if (is_stream_) {
        // On Windows, shutdown() on a listening socket does not interrupt a
        // blocked accept().  closesocket() is required.  We close the fd here
        // and mark it invalid so Close() becomes a no-op.
#ifdef _WIN32
        closesocket(sock_);
        sock_ = INVALID_SOCKET;
#else
        ::shutdown(sock_, SHUT_RDWR);
#endif
        LOG_INFO(Lib_Net, "P2PSocket::Abort: closed STREAM_P2P fd={} vport={} (flags={:#x})",
                 (int)sock_, ntohs(bound_vport_), flags);
    }

    {
        std::scoped_lock tlock{s_transport.mutex};
        auto it = s_transport.vport_queues.find(bound_vport_);
        if (it != s_transport.vport_queues.end() && !it->second.empty()) {
            LOG_INFO(Lib_Net, "P2PSocket::Abort: flushed {} stale packets from vport={} flags={:#x}",
                     it->second.size(), ntohs(bound_vport_), flags);
            it->second = {};
        }
    }

    LOG_INFO(Lib_Net, "P2PSocket::Abort: vport={} (flags={:#x})", ntohs(bound_vport_), flags);
    return 0;
}

void DrainP2PTransport() {
    std::scoped_lock lock{s_transport.mutex};
    s_transport.Drain();
}

u16 GetP2PBoundPort() {
    std::scoped_lock lock{s_transport.mutex};
    return ntohs(s_transport.bound_port_nbo);
}

u16 GetP2PConfiguredPort() {
    return Config::GetSignalingPort();
}

std::string GetP2PConfiguredAddr() {
    return Config::GetSignalingAddr();
}

int P2PSignalingSendTo(const void* data, u32 len, u32 dest_addr, u16 dest_port) {
    std::scoped_lock lock{s_transport.mutex};
    if (!s_transport.IsValid()) {
        return -1;
    }

    sockaddr_in dest{};
    dest.sin_family      = AF_INET;
    dest.sin_addr.s_addr = dest_addr; // already network byte order
    dest.sin_port        = dest_port; // already network byte order

    const int rc = ::sendto(s_transport.fd, reinterpret_cast<const char*>(data),
                            static_cast<int>(len), 0,
                            reinterpret_cast<const sockaddr*>(&dest), sizeof(dest));
    return ConvertReturnErrorCode(rc);
}

int P2PSignalingRecvFrom(void* buf, u32 len, u32* from_addr, u16* from_port) {
    std::scoped_lock lock{s_transport.mutex};

    // Drain first so any pending signaling packets are queued.
    s_transport.Drain();

    if (s_transport.signaling_queue.empty()) {
        return -1;
    }

    SignalingPacket& sp = s_transport.signaling_queue.front();
    const u32 copy_len = std::min(len, static_cast<u32>(sp.data.size()));
    std::memcpy(buf, sp.data.data(), copy_len);

    if (from_addr) {
        *from_addr = sp.from_addr.sin_addr.s_addr; // network byte order
    }
    if (from_port) {
        *from_port = sp.from_addr.sin_port; // network byte order
    }

    s_transport.signaling_queue.pop();
    return static_cast<int>(copy_len);
}

bool P2PTransportIsReady() {
    std::scoped_lock lock{s_transport.mutex};
    return s_transport.IsValid();
}


} // namespace Libraries::Net
