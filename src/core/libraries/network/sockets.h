// SPDX-FileCopyrightText: Copyright 2024-2026 shadPS4 Emulator Project
// SPDX-License-Identifier: GPL-2.0-or-later

#pragma once

#ifdef _WIN32
#define _WINSOCK_DEPRECATED_NO_WARNINGS
#include <Ws2tcpip.h>
#include <afunix.h>
#include <iphlpapi.h>
#include <mstcpip.h>
#include <winsock2.h>
using net_socket = SOCKET;
using socklen_t = int;
#ifndef LPFN_WSASENDMSG
typedef INT(PASCAL* LPFN_WSASENDMSG)(SOCKET s, LPWSAMSG lpMsg, DWORD dwFlags,
                                     LPDWORD lpNumberOfBytesSent, LPWSAOVERLAPPED lpOverlapped,
                                     LPWSAOVERLAPPED_COMPLETION_ROUTINE lpCompletionRoutine);
#endif
#ifndef WSAID_WSASENDMSG
static const GUID WSAID_WSASENDMSG = {
    0xa441e712, 0x754f, 0x43ca, {0x84, 0xa7, 0x0d, 0xee, 0x44, 0xcf, 0x60, 0x6d}};
#endif
#ifndef LPFN_WSARECVMSG
typedef INT(PASCAL* LPFN_WSARECVMSG)(SOCKET s, LPWSAMSG lpMsg, LPDWORD lpdwNumberOfBytesRecvd,
                                     LPWSAOVERLAPPED lpOverlapped,
                                     LPWSAOVERLAPPED_COMPLETION_ROUTINE lpCompletionRoutine);
#endif

#ifndef WSAID_WSARECVMSG
static const GUID WSAID_WSARECVMSG = {
    0xf689d7c8, 0x6f1f, 0x436b, {0x8a, 0x53, 0xe5, 0x4f, 0xe3, 0x51, 0xc3, 0x22}};
#endif
#else
#include <cerrno>
#include <arpa/inet.h>
#include <netdb.h>
#include <netinet/in.h>
#include <netinet/tcp.h>
#include <sys/ioctl.h>
#include <sys/socket.h>
#include <unistd.h>
using net_socket = int;
#endif
#include <chrono>
#include <map>
#include <memory>
#include <mutex>
#include "net.h"

namespace Libraries::Kernel {
struct OrbisKernelStat;
}

namespace Libraries::Net {

struct Socket;

using SocketPtr = std::shared_ptr<Socket>;

struct OrbisNetLinger {
    s32 l_onoff;
    s32 l_linger;
};
struct Socket {
    explicit Socket(int domain, int type, int protocol) : socket_type(type) {}
    virtual ~Socket() = default;
    virtual bool IsValid() const = 0;
    virtual int Close() = 0;
    virtual int SetSocketOptions(int level, int optname, const void* optval, u32 optlen) = 0;
    virtual int GetSocketOptions(int level, int optname, void* optval, u32* optlen) = 0;
    virtual int Bind(const OrbisNetSockaddr* addr, u32 addrlen) = 0;
    virtual int Listen(int backlog) = 0;
    virtual int SendMessage(const OrbisNetMsghdr* msg, int flags) = 0;
    virtual int SendPacket(const void* msg, u32 len, int flags, const OrbisNetSockaddr* to,
                           u32 tolen) = 0;
    virtual SocketPtr Accept(OrbisNetSockaddr* addr, u32* addrlen) = 0;
    virtual int ReceiveMessage(OrbisNetMsghdr* msg, int flags) = 0;
    virtual int ReceivePacket(void* buf, u32 len, int flags, OrbisNetSockaddr* from,
                              u32* fromlen) = 0;
    virtual int Connect(const OrbisNetSockaddr* addr, u32 namelen) = 0;
    virtual int GetSocketAddress(OrbisNetSockaddr* name, u32* namelen) = 0;
    virtual int GetPeerName(OrbisNetSockaddr* addr, u32* namelen) = 0;
    virtual int fstat(Libraries::Kernel::OrbisKernelStat* stat) = 0;
    virtual std::optional<net_socket> Native() = 0;
    virtual bool HasQueuedData() {
        return false;
    }
    // Abort a pending operation on this socket (sceNetSocketAbort).
    // P2P override clears the vport receive queue; others call OS shutdown.
    virtual int Abort(int flags) {
        return 0;
    }
    // Half-close the socket (sceNetShutdown).
    // P2P sockets are connectionless so this is a no-op for them.
    virtual int Shutdown(int how) {
        return 0;
    }
    std::mutex m_mutex;
    std::mutex receive_mutex;
    int socket_type;
};

struct PosixSocket : public Socket {
    net_socket sock;
    int sockopt_so_connecttimeo = 0;
    int sockopt_so_reuseport = 0;
    int sockopt_so_onesbcast = 0;
    int sockopt_so_usecrypto = 0;
    int sockopt_so_usesignature = 0;
    int sockopt_so_nbio = 0;
    int sockopt_ip_ttlchk = 0;
    int sockopt_ip_maxttl = 0;
    int sockopt_tcp_mss_to_advertise = 0;
    int socket_type;
    explicit PosixSocket(int domain, int type, int protocol)
        : Socket(domain, type, protocol), sock(socket(domain, type, protocol)) {
        socket_type = type;
    }
    explicit PosixSocket(net_socket sock) : Socket(0, 0, 0), sock(sock) {}
    bool IsValid() const override;
    int Close() override;
    int SetSocketOptions(int level, int optname, const void* optval, u32 optlen) override;
    int GetSocketOptions(int level, int optname, void* optval, u32* optlen) override;
    int Bind(const OrbisNetSockaddr* addr, u32 addrlen) override;
    int Listen(int backlog) override;
    int SendMessage(const OrbisNetMsghdr* msg, int flags) override;
    int SendPacket(const void* msg, u32 len, int flags, const OrbisNetSockaddr* to,
                   u32 tolen) override;
    int ReceiveMessage(OrbisNetMsghdr* msg, int flags) override;
    int ReceivePacket(void* buf, u32 len, int flags, OrbisNetSockaddr* from, u32* fromlen) override;
    SocketPtr Accept(OrbisNetSockaddr* addr, u32* addrlen) override;
    int Connect(const OrbisNetSockaddr* addr, u32 namelen) override;
    int GetSocketAddress(OrbisNetSockaddr* name, u32* namelen) override;
    int GetPeerName(OrbisNetSockaddr* addr, u32* namelen) override;
    int fstat(Libraries::Kernel::OrbisKernelStat* stat) override;
    int Shutdown(int how) override;
    std::optional<net_socket> Native() override {
        return sock;
    }
};

struct P2PSocket : public Socket {
    net_socket sock_;          // shared UDP transport fd (DGRAM) or owned TCP fd (STREAM)
    u16 bound_vport_{0};       // bound virtual port (network byte order)
    int sockopt_so_nbio_{0};   // non-blocking mode flag
    bool is_stream_{false};    // true = STREAM_P2P: owns a real TCP fd
    bool is_connected_{false}; // true = accepted/connected TCP stream socket

    explicit P2PSocket(int domain, int type, int protocol);
    // Internal: construct an already-accepted TCP stream socket
    explicit P2PSocket(net_socket accepted_fd, u16 peer_vport);
    bool IsValid() const override;
    int Close() override;
    int SetSocketOptions(int level, int optname, const void* optval, u32 optlen) override;
    int GetSocketOptions(int level, int optname, void* optval, u32* optlen) override;
    int Bind(const OrbisNetSockaddr* addr, u32 addrlen) override;
    int Listen(int backlog) override;
    int SendMessage(const OrbisNetMsghdr* msg, int flags) override;
    int SendPacket(const void* msg, u32 len, int flags, const OrbisNetSockaddr* to,
                   u32 tolen) override;
    int ReceiveMessage(OrbisNetMsghdr* msg, int flags) override;
    int ReceivePacket(void* buf, u32 len, int flags, OrbisNetSockaddr* from, u32* fromlen) override;
    SocketPtr Accept(OrbisNetSockaddr* addr, u32* addrlen) override;
    int Connect(const OrbisNetSockaddr* addr, u32 namelen) override;
    int GetSocketAddress(OrbisNetSockaddr* name, u32* namelen) override;
    int GetPeerName(OrbisNetSockaddr* addr, u32* namelen) override;
    int fstat(Libraries::Kernel::OrbisKernelStat* stat) override;
    bool HasQueuedData() override;
    int Abort(int flags) override;
    std::optional<net_socket> Native() override {
        if (IsValid())
            return sock_;
        return {};
    }
};

// Drain the shared P2P transport socket into per-vport queues.
// Call this before checking HasQueuedData() on P2P sockets.
void DrainP2PTransport();
// Returns the actual UDP port the shared P2P transport is bound to (host byte order).
// Returns 0 if the transport has not been initialized yet.
u16 GetP2PBoundPort();
// Returns the port the shared P2P transport is configured to bind to (from config).
// Safe to call before the transport is initialized.
u16 GetP2PConfiguredPort();
// Returns the signaling address configured for the P2P transport (from config).
std::string GetP2PConfiguredAddr();
// Returns true if the P2P shared transport socket is ready.
bool P2PTransportIsReady();
// Send raw signaling data through the P2P shared transport socket.
// dest_addr and dest_port must be in network byte order.
// Returns bytes sent, or -1 on failure.
int P2PSignalingSendTo(const void* data, u32 len, u32 dest_addr, u16 dest_port);
// Receive a signaling packet from the P2P shared transport.
// Drains the socket first and returns the next queued signaling packet.
// from_addr and from_port are filled in network byte order.
// Returns bytes copied, or -1 if no signaling packet is available.
int P2PSignalingRecvFrom(void* buf, u32 len, u32* from_addr, u16* from_port);

struct UnixSocket : public Socket {
    net_socket sock;
    int socket_type;
    explicit UnixSocket(int domain, int type, int protocol)
        : Socket(domain, type, protocol), sock(socket(domain, type, protocol)) {
        socket_type = type;
    }
    explicit UnixSocket(net_socket sock) : Socket(0, 0, 0), sock(sock) {}
    bool IsValid() const override;
    int Close() override;
    int SetSocketOptions(int level, int optname, const void* optval, u32 optlen) override;
    int GetSocketOptions(int level, int optname, void* optval, u32* optlen) override;
    int Bind(const OrbisNetSockaddr* addr, u32 addrlen) override;
    int Listen(int backlog) override;
    int SendMessage(const OrbisNetMsghdr* msg, int flags) override;
    int SendPacket(const void* msg, u32 len, int flags, const OrbisNetSockaddr* to,
                   u32 tolen) override;
    int ReceiveMessage(OrbisNetMsghdr* msg, int flags) override;
    int ReceivePacket(void* buf, u32 len, int flags, OrbisNetSockaddr* from, u32* fromlen) override;
    SocketPtr Accept(OrbisNetSockaddr* addr, u32* addrlen) override;
    int Connect(const OrbisNetSockaddr* addr, u32 namelen) override;
    int GetSocketAddress(OrbisNetSockaddr* name, u32* namelen) override;
    int GetPeerName(OrbisNetSockaddr* addr, u32* namelen) override;
    int fstat(Libraries::Kernel::OrbisKernelStat* stat) override;
    std::optional<net_socket> Native() override {
        return sock;
    }
};

} // namespace Libraries::Net
