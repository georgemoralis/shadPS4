// SPDX-FileCopyrightText: Copyright 2026 shadPS4 Emulator Project
// SPDX-License-Identifier: GPL-2.0-or-later

// The PS4 P2P wire format
//
//   Datagram:  [0xFF][flags|kind][src vport][dst vport][communication ID (4)?][payload]
//              vports are 1 byte each when both are <= 255 (COMPACT flag), else 2 bytes each,
//              network byte order.
//   Stream:    [0xFF][flags|kind][communication ID (4)?][TCP segment]
//   Signaling: a plain datagram from and to vport 0xFFFF.
//
//   flags: 0x80 valid, 0x40 compact vports, 0x20 communication ID present.
//   kind:  3 = datagram, 7 = stream; +1 encrypted, +2 signed (see p2p_crypto.h).

#pragma once

#include <map>
#include <mutex>
#include <optional>

#include "core/net/p2p_crypto.h"
#include "core/net/p2p_transport.h"

namespace Core::Net::P2P {

namespace Wire {
constexpr u8 Marker = 0xFF;
constexpr u8 FlagValid = 0x80;
constexpr u8 FlagCompactVports = 0x40;
constexpr u8 FlagCommunicationId = 0x20;
constexpr u8 KindDatagram = 3;
constexpr u8 KindStream = 7;
constexpr u16 SignalingVport = 0xFFFF;
} // namespace Wire

using CommunicationId = std::array<u8, 4>;

/// Keys for protected traffic and the title's communication ID.
struct KeyEntry {
    P2PKey key;
    u16 local_port = 0; // network byte order, as the ioctl carries it
    u8 flags = 0;
    u8 id = 0; // groups keys for removal
};

class Keyring {
public:
    /// Installs entry for peer, replacing any other key. When the same key value is
    /// already installed there, adds a reference instead (NP installs a key once per user).
    void AddPeerKey(const Endpoint& peer, const KeyEntry& entry);
    /// Drops one reference to the key installed for peer. It goes away with the last one.
    /// False when no key with that value and id is installed there.
    bool ReleasePeerKey(const Endpoint& peer, const P2PKey& key, u8 id);
    /// Removes every peer key installed with id.
    void RemoveKeysWithId(u8 id);
    void SetPeerKey(const Endpoint& peer, const P2PKey& key);
    void RemovePeerKey(const Endpoint& peer);
    void SetWildcardKey(std::optional<P2PKey> key);
    std::optional<P2PKey> GetWildcardKey() const;
    void SetDefaultKey(std::optional<P2PKey> key);
    std::optional<P2PKey> GetDefaultKey() const;
    std::optional<P2PKey> Find(const Endpoint& peer) const;
    std::optional<KeyEntry> FindPeerEntry(const Endpoint& peer) const;
    void SetCommunicationId(std::optional<CommunicationId> id);
    std::optional<CommunicationId> GetCommunicationId() const;

private:
    struct Installed {
        KeyEntry entry;
        int references = 1;
    };

    mutable std::mutex mutex;
    std::map<Endpoint, Installed> peer_keys;
    std::optional<P2PKey> wildcard_key;
    std::optional<P2PKey> default_key;
    std::optional<CommunicationId> communication_id;
};

} // namespace Core::Net::P2P
