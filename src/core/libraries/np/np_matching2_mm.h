// SPDX-FileCopyrightText: Copyright 2025 shadPS4 Emulator Project
// SPDX-License-Identifier: GPL-2.0-or-later

#pragma once

#include <cstdint>
#include <string>
#include <string_view>
#include <vector>

#include "common/types.h"

namespace Libraries::Np::NpMatching2 {

// ── shadnet binary protocol constants ────────────────────────────────────────

static constexpr u32 SHADNET_HEADER_SIZE = 15;
static constexpr u32 SHADNET_MAX_PACKET_SIZE = 0x800000; // 8 MiB

enum class ShadnetPacketType : u8 {
    Request = 0,
    Reply = 1,
    Notification = 2,
    ServerInfo = 3,
};

// CommandType IDs matching shadnet server protocol.h
enum class ShadnetCommandType : u16 {
    Login = 0,
    Create = 2,
    RegisterHandlers = 12,
    CreateRoom = 13,
    JoinRoom = 14,
    LeaveRoom = 15,
    GetRoomList = 16,
    RequestSignalingInfos = 17,
    SignalingEstablished = 18,
    ActivationConfirm = 19,
    SetRoomDataInternal = 20,
    SetRoomDataExternal = 21,
};

// NotificationType IDs matching shadnet server protocol.h
enum class ShadnetNotificationType : u16 {
    RequestEvent = 9,
    MemberJoined = 10,
    MemberLeft = 11,
    SignalingHelper = 12,
    SignalingEvent = 13,
    NpSignalingEvent = 14,
    RoomDataInternalUpdated = 15,
};

// ErrorType codes matching shadnet server protocol.h
enum class ShadnetErrorType : u8 {
    NoError = 0,
    Malformed = 1,
    Invalid = 2,
    InvalidInput = 3,
    LoginError = 5,
    LoginAlreadyLoggedIn = 6,
    LoginInvalidPassword = 8,
    CreationExistingUsername = 11,
    RoomMissing = 14,
    RoomAlreadyJoined = 15,
    RoomFull = 16,
    Unauthorized = 23,
    NotFound = 26,
};

// ── Member info (kept for compatibility with existing code) ──────────────────

struct MemberBinAttr {
    u16 id = 0;
    std::vector<u8> data;
};

struct MemberInfo {
    int member_id = 0;
    std::string online_id;
    std::string addr;
    int port = 0;
    std::vector<MemberBinAttr> bin_attrs; // per-member bin attrs from server
};

// ── Binary buffer helpers ────────────────────────────────────────────────────

class BinaryWriter {
public:
    void AppendU8(u8 v) {
        buf_.push_back(v);
    }
    void AppendU16LE(u16 v) {
        buf_.push_back(static_cast<u8>(v & 0xFF));
        buf_.push_back(static_cast<u8>((v >> 8) & 0xFF));
    }
    void AppendU32LE(u32 v) {
        for (int i = 0; i < 4; ++i)
            buf_.push_back(static_cast<u8>((v >> (8 * i)) & 0xFF));
    }
    void AppendU64LE(u64 v) {
        for (int i = 0; i < 8; ++i)
            buf_.push_back(static_cast<u8>((v >> (8 * i)) & 0xFF));
    }
    void AppendCStr(std::string_view s) {
        buf_.insert(buf_.end(), s.begin(), s.end());
        buf_.push_back('\0');
    }
    void AppendBytes(const u8* data, size_t len) {
        buf_.insert(buf_.end(), data, data + len);
    }

    const std::vector<u8>& data() const {
        return buf_;
    }
    size_t size() const {
        return buf_.size();
    }

private:
    std::vector<u8> buf_;
};

class BinaryReader {
public:
    explicit BinaryReader(const std::vector<u8>& data) : data_(data), pos_(0) {}
    BinaryReader(const u8* data, size_t len) : data_(data, data + len), pos_(0) {}

    bool error() const {
        return error_;
    }
    size_t pos() const {
        return pos_;
    }
    size_t remaining() const {
        return error_ ? 0 : (data_.size() - pos_);
    }

    u8 GetU8() {
        if (pos_ + 1 > data_.size()) {
            error_ = true;
            return 0;
        }
        return data_[pos_++];
    }
    u16 GetU16LE() {
        if (pos_ + 2 > data_.size()) {
            error_ = true;
            return 0;
        }
        u16 v = static_cast<u16>(data_[pos_]) | (static_cast<u16>(data_[pos_ + 1]) << 8);
        pos_ += 2;
        return v;
    }
    u32 GetU32LE() {
        if (pos_ + 4 > data_.size()) {
            error_ = true;
            return 0;
        }
        u32 v = 0;
        for (int i = 0; i < 4; ++i)
            v |= static_cast<u32>(data_[pos_ + i]) << (8 * i);
        pos_ += 4;
        return v;
    }
    u64 GetU64LE() {
        if (pos_ + 8 > data_.size()) {
            error_ = true;
            return 0;
        }
        u64 v = 0;
        for (int i = 0; i < 8; ++i)
            v |= static_cast<u64>(data_[pos_ + i]) << (8 * i);
        pos_ += 8;
        return v;
    }
    std::string GetCStr() {
        std::string s;
        while (pos_ < data_.size() && data_[pos_] != '\0') {
            s += static_cast<char>(data_[pos_++]);
        }
        if (pos_ >= data_.size()) {
            error_ = true;
            return {};
        }
        pos_++; // consume null
        return s;
    }
    void skip(size_t n) {
        if (pos_ + n > data_.size()) {
            error_ = true;
            return;
        }
        pos_ += n;
    }
    const u8* GetRawPtr(size_t n) {
        if (pos_ + n > data_.size()) {
            error_ = true;
            return nullptr;
        }
        const u8* p = data_.data() + pos_;
        pos_ += n;
        return p;
    }

private:
    std::vector<u8> data_;
    size_t pos_;
    bool error_ = false;
};

// ── Public API ───────────────────────────────────────────────────────────────

// Notification handlers are now called with parsed binary data.
// MemberJoined/MemberLeft/SignalingHelper receive binary payloads via BinaryReader.
using MmBinaryHandler = void (*)(BinaryReader& reader);

struct MmNotificationHandlers {
    MmBinaryHandler member_joined = nullptr;
    MmBinaryHandler member_left = nullptr;
    MmBinaryHandler room_destroyed = nullptr;
    MmBinaryHandler signaling_helper = nullptr;
    MmBinaryHandler room_data_internal_updated = nullptr;
};

void ConfigureMmNotificationHandlers(MmNotificationHandlers handlers);
bool StartMmClient(std::string_view online_id, std::string_view signaling_addr, u16 signaling_port);
void StopMmClient();
bool IsMmClientRunning();

// Send a binary request and wait for the reply (up to 5 seconds).
// Returns the error type from the reply header. Reply payload (after error byte) is written to
// out_reply.
ShadnetErrorType MmRequest(ShadnetCommandType cmd, const BinaryWriter& payload,
                           std::vector<u8>& out_reply);

// Send a binary request without waiting for a reply.
void MmSendFireAndForget(ShadnetCommandType cmd, const BinaryWriter& payload);

void SendRegisterHandlers();

void OnMmMemberJoined(BinaryReader& reader);
void OnMmMemberLeft(BinaryReader& reader);
void OnMmRoomDestroyed(BinaryReader& reader);
void OnMmSignalingHelper(BinaryReader& reader);
void OnMmRoomDataInternalUpdated(BinaryReader& reader);

u32 GetMmServerAddr();
u16 GetMmServerUdpPort();

bool RequestSignalingInfos(std::string_view target_online_id, u32* out_addr, u16* out_port);

// Send ActivationConfirm as binary TCP command
bool SendActivationConfirm(std::string_view me_id, std::string_view initiator_ip, u32 ctx_tag);

} // namespace Libraries::Np::NpMatching2
