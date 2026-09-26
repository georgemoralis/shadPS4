// SPDX-FileCopyrightText: Copyright 2026 shadPS4 Emulator Project
// SPDX-License-Identifier: GPL-2.0-or-later

#pragma once

#include <array>
#include <span>
#include <vector>

#include "common/types.h"

namespace Core::Net::P2P {

struct P2PKey {
    std::array<u8, 16> value{};
    bool operator==(const P2PKey&) const = default;
};

bool ProtectPayload(std::span<const u8> data, bool encrypt, bool sign, const P2PKey& key,
                    std::vector<u8>& output);

/// Fails on a short packet, a bad signature, or a cipher error.
bool UnprotectPayload(std::span<const u8> data, bool encrypt, bool sign, const P2PKey& key,
                      std::vector<u8>& output);

/// The 4-byte communication ID the NP libraries derive from a title's 16-byte value: the first
/// bytes of HMAC-SHA1 keyed with the value over an empty message.
std::array<u8, 4> DeriveCommunicationId(std::span<const u8, 16> value);

} // namespace Core::Net::P2P
