// SPDX-FileCopyrightText: Copyright 2026 shadPS4 Emulator Project
// SPDX-License-Identifier: GPL-2.0-or-later

#include <cstring>

#include "core/net/p2p_codec.h"

namespace Core::Net::P2P {

using namespace Wire;

// Keyring
void Keyring::AddPeerKey(const Endpoint& peer, const KeyEntry& entry) {
    std::scoped_lock lock{mutex};
    const auto it = peer_keys.find(peer);
    if (it != peer_keys.end() && it->second.entry.key == entry.key) {
        ++it->second.references;
    } else {
        peer_keys[peer] = {entry, 1};
    }
}

bool Keyring::ReleasePeerKey(const Endpoint& peer, const P2PKey& key, u8 id) {
    std::scoped_lock lock{mutex};
    const auto it = peer_keys.find(peer);
    if (it == peer_keys.end() || !(it->second.entry.key == key) || it->second.entry.id != id) {
        return false;
    }
    if (--it->second.references == 0) {
        peer_keys.erase(it);
    }
    return true;
}

void Keyring::RemoveKeysWithId(u8 id) {
    std::scoped_lock lock{mutex};
    std::erase_if(peer_keys, [id](const auto& entry) { return entry.second.entry.id == id; });
}

void Keyring::SetPeerKey(const Endpoint& peer, const P2PKey& key) {
    std::scoped_lock lock{mutex};
    peer_keys[peer] = {KeyEntry{key}, 1};
}

void Keyring::RemovePeerKey(const Endpoint& peer) {
    std::scoped_lock lock{mutex};
    peer_keys.erase(peer);
}

void Keyring::SetWildcardKey(std::optional<P2PKey> key) {
    std::scoped_lock lock{mutex};
    wildcard_key = key;
}

std::optional<P2PKey> Keyring::GetWildcardKey() const {
    std::scoped_lock lock{mutex};
    return wildcard_key;
}

void Keyring::SetDefaultKey(std::optional<P2PKey> key) {
    std::scoped_lock lock{mutex};
    default_key = key;
}

std::optional<P2PKey> Keyring::GetDefaultKey() const {
    std::scoped_lock lock{mutex};
    return default_key;
}

std::optional<P2PKey> Keyring::Find(const Endpoint& peer) const {
    std::scoped_lock lock{mutex};
    if (const auto it = peer_keys.find(peer); it != peer_keys.end()) {
        return it->second.entry.key;
    }
    return wildcard_key ? wildcard_key : default_key;
}

std::optional<KeyEntry> Keyring::FindPeerEntry(const Endpoint& peer) const {
    std::scoped_lock lock{mutex};
    if (const auto it = peer_keys.find(peer); it != peer_keys.end()) {
        return it->second.entry;
    }
    return std::nullopt;
}

void Keyring::SetCommunicationId(std::optional<CommunicationId> id) {
    std::scoped_lock lock{mutex};
    communication_id = id;
}

std::optional<CommunicationId> Keyring::GetCommunicationId() const {
    std::scoped_lock lock{mutex};
    return communication_id;
}

} // namespace Core::Net::P2P
