// SPDX-FileCopyrightText: Copyright 2026 shadPS4 Emulator Project
// SPDX-License-Identifier: GPL-2.0-or-later

#include <algorithm>
#include <cstring>
#include <tuple>

#include "core/net/p2p_transport.h"

#ifndef _WIN32
#include <arpa/inet.h>
#include <netinet/in.h>
#endif

namespace Core::Net::P2P {

namespace {

auto Canonical(const Endpoint& e) {
    std::array<u8, 16> address{};
    u16 port = 0;
    u32 scope = 0;
    if (e.Family() == AF_INET) {
        const auto* in = reinterpret_cast<const sockaddr_in*>(&e.addr);
        std::memcpy(address.data(), &in->sin_addr, 4);
        port = in->sin_port;
    } else if (e.Family() == AF_INET6) {
        const auto* in6 = reinterpret_cast<const sockaddr_in6*>(&e.addr);
        std::memcpy(address.data(), &in6->sin6_addr, 16);
        port = in6->sin6_port;
        scope = in6->sin6_scope_id;
    }
    return std::tuple{e.Family(), address, port, scope};
}

} // namespace

Endpoint Endpoint::FromSockaddr(const sockaddr* sa, socklen_t len) {
    Endpoint e;
    std::memcpy(&e.addr, sa, std::min<size_t>(static_cast<size_t>(len), sizeof(e.addr)));
    return e;
}

Endpoint Endpoint::IPv4(const char* dotted, u16 port) {
    Endpoint e;
    auto* in = reinterpret_cast<sockaddr_in*>(&e.addr);
    in->sin_family = AF_INET;
    in->sin_port = htons(port);
    inet_pton(AF_INET, dotted, &in->sin_addr);
    return e;
}

socklen_t Endpoint::Length() const {
    return Family() == AF_INET6 ? sizeof(sockaddr_in6) : sizeof(sockaddr_in);
}

u16 Endpoint::Port() const {
    return ntohs(std::get<2>(Canonical(*this)));
}

bool Endpoint::operator<(const Endpoint& other) const {
    return Canonical(*this) < Canonical(other);
}

bool Endpoint::operator==(const Endpoint& other) const {
    return Canonical(*this) == Canonical(other);
}

} // namespace Core::Net::P2P
