// SPDX-FileCopyrightText: Copyright 2026 shadPS4 Emulator Project
// SPDX-License-Identifier: GPL-2.0-or-later

#pragma once

#include "common/types.h"
#include "core/libraries/np/np_manager.h"
#include "core/libraries/np/np_matching2.h"
#include "core/libraries/rtc/rtc.h"

namespace Libraries::Np::NpMatching2 {

struct OrbisNpMatching2SignalingParam {
    int type;
    int flag;
    OrbisNpMatching2RoomMemberId mainMember;
    u8 pad[4];
};

struct OrbisNpMatching2InitParam {
    u64 heapSize;
    u64 signalingInitParam;
    s32 maxConnections;
    u32 pad;
    u64 threadStackSize;
    u64 size;
    u64 sslBufSize;
};

struct OrbisNpMatching2ExtraInitParam {
    u16 signalingPort;
    u8 pad[6];
};

struct OrbisNpMatching2SessionPassword {
    u8 data[8];
};

struct OrbisNpMatching2RoomPassword {
    u8 data[8];
};

struct OrbisNpMatching2GroupLabel {
    u8 data[8];
};

struct OrbisNpMatching2RoomGroupConfig {
    u32 slots;
    bool hasLabel;
    OrbisNpMatching2GroupLabel label;
    bool hasPassword;
    u8 pad[2];
};

struct OrbisNpMatching2BinAttr {
    OrbisNpMatching2AttributeId id;
    u8 pad[6];
    u8* data;
    u64 dataSize;
};

struct OrbisNpMatching2RoomBinAttrInternal {
    Libraries::Rtc::OrbisRtcTick lastUpdate;
    OrbisNpMatching2RoomMemberId memberId;
    u8 pad[6];
    OrbisNpMatching2BinAttr binAttr;
};

struct OrbisNpMatching2RoomMemberBinAttrInternal {
    Libraries::Rtc::OrbisRtcTick lastUpdate;
    OrbisNpMatching2BinAttr binAttr;
};

struct OrbisNpMatching2IntAttr {
    OrbisNpMatching2AttributeId id;
    u8 pad[2];
    u32 attr;
};

template <typename T>
struct OrbisNpMatching2CreateJoinRoomRequest_ {
    u16 maxSlot;
    OrbisNpMatching2TeamId teamId;
    u8 pad[5];
    OrbisNpMatching2Flags flags;
    OrbisNpMatching2WorldId worldId;
    OrbisNpMatching2LobbyId lobbyId;
    OrbisNpMatching2RoomPassword* roomPasswd;
    u64* passwdSlotMask;
    OrbisNpMatching2RoomGroupConfig* groupConfig;
    u64 groupConfigs;
    OrbisNpMatching2GroupLabel* joinGroupLabel;
    T* allowedUser;
    u64 allowedUsers;
    T* blockedUser;
    u64 blockedUsers;
    OrbisNpMatching2BinAttr* internalBinAttr;
    u64 internalBinAttrs;
    OrbisNpMatching2IntAttr* externalSearchIntAttr;
    u64 externalSearchIntAttrs;
    OrbisNpMatching2BinAttr* externalSearchBinAttr;
    u64 externalSearchBinAttrs;
    OrbisNpMatching2BinAttr* externalBinAttr;
    u64 externalBinAttrs;
    OrbisNpMatching2BinAttr* memberInternalBinAttr;
    u64 memberInternalBinAttrs;
    OrbisNpMatching2SignalingParam* signalingParam;

    int Validate() {
        return 0;
    }
};

using OrbisNpMatching2CreateJoinRoomRequest =
    OrbisNpMatching2CreateJoinRoomRequest_<Libraries::Np::OrbisNpOnlineId>;
using OrbisNpMatching2CreateJoinRoomRequestA =
    OrbisNpMatching2CreateJoinRoomRequest_<Libraries::Np::OrbisNpAccountId>;

static_assert(sizeof(OrbisNpMatching2CreateJoinRoomRequestA) == 184);

struct OrbisNpMatching2RoomGroup {
    OrbisNpMatching2RoomGroupId id;
    bool hasPasswd;
    bool hasLabel;
    u8 pad;
    OrbisNpMatching2GroupLabel label;
    u32 slots;
    u32 groupMembers;
};

struct OrbisNpMatching2RoomGroupInfo {
    OrbisNpMatching2RoomGroupId id;
    bool hasPasswd;
    u8 pad[2];
    u32 slots;
    u32 groupMembers;
};

struct OrbisNpMatching2RangeFilter {
    u32 start;
    u32 max;
};

enum class OrbisNpMatching2Operator : u8 { Eq = 1, Ne = 2, Lt = 3, Le = 4, Gt = 5, Ge = 6 };

struct OrbisNpMatching2IntFilter {
    OrbisNpMatching2Operator op;
    u8 pad[7];
    OrbisNpMatching2IntAttr attr;
};

struct OrbisNpMatching2BinFilter {
    OrbisNpMatching2Operator op;
    u8 pad[7];
    OrbisNpMatching2BinAttr attr;
};

struct OrbisNpMatching2SearchRoomRequest {
    int option;
    OrbisNpMatching2WorldId worldId;
    OrbisNpMatching2LobbyId lobbyId;
    OrbisNpMatching2RangeFilter rangeFilter;
    OrbisNpMatching2Flags flagFilter;
    OrbisNpMatching2Flags flagAttrs;
    OrbisNpMatching2IntFilter* intFilter;
    u64 intFilters;
    OrbisNpMatching2BinFilter* binFilter;
    u64 binFilters;
    OrbisNpMatching2AttributeId* attr;
    u64 attrs;

    int Validate() {
        return 0;
    }
};

struct OrbisNpMatching2RoomDataInternal {
    u16 publicSlots;
    u16 privateSlots;
    u16 openPublicSlots;
    u16 openPrivateSlots;
    u16 maxSlot;
    OrbisNpMatching2ServerId serverId;
    OrbisNpMatching2WorldId worldId;
    OrbisNpMatching2LobbyId lobbyId;
    OrbisNpMatching2RoomId roomId;
    u64 passwdSlotMask;
    u64 joinedSlotMask;
    const OrbisNpMatching2RoomGroup* roomGroup;
    u64 roomGroups;
    OrbisNpMatching2Flags flags;
    u8 pad[4];
    const OrbisNpMatching2RoomBinAttrInternal* roomBinAttrInternal;
    u64 roomBinAttrInternalNum;
};

static_assert(offsetof(OrbisNpMatching2RoomDataInternal, roomBinAttrInternal) == 0x48);

struct OrbisNpMatching2SetRoomDataInternalRequest {
    OrbisNpMatching2RoomId roomId;
    OrbisNpMatching2Flags flagFilter;
    OrbisNpMatching2Flags flagAttr;
    OrbisNpMatching2BinAttr* roomBinAttrInternal;
    u32 roomBinAttrInternalNum;
    OrbisNpMatching2RoomGroupConfig* passwordConfig;
    u32 passwordConfigNum;
    u64* passwordSlotMask;
    OrbisNpMatching2RoomMemberId* ownerPrivilegeRank;
    u32 ownerPrivilegeRankNum;
    u8 padding[4];
};

struct OrbisNpMatching2SetRoomDataExternalRequest {
    OrbisNpMatching2RoomId roomId;
    OrbisNpMatching2IntAttr* roomSearchableIntAttrExternal;
    u32 roomSearchableIntAttrExternalNum;
    u8 pad1[4];
    OrbisNpMatching2BinAttr* roomSearchableBinAttrExternal;
    u32 roomSearchableBinAttrExternalNum;
    u8 pad2[4];
    OrbisNpMatching2BinAttr* roomBinAttrExternal;
    u32 roomBinAttrExternalNum;
    u8 pad3[4];
};

struct OrbisNpMatching2RoomMemberDataInternal {
    OrbisNpMatching2RoomMemberDataInternal* next;          // +0x00
    u64 joinDate;                                          // +0x08
    Libraries::Np::OrbisNpId npId;                         // +0x10
    u8 pad[4];                                             // +0x34
    OrbisNpMatching2RoomMemberId memberId;                 // +0x38
    OrbisNpMatching2TeamId teamId;                         // +0x3A
    OrbisNpMatching2NatType natType;                       // +0x3B
    OrbisNpMatching2Flags flagAttr;                        // +0x3C
    OrbisNpMatching2RoomGroup* roomGroup;                  // +0x40
    OrbisNpMatching2RoomMemberBinAttrInternal* roomMemberBinAttrInternal; // +0x48
    u64 roomMemberBinAttrInternalNum;                      // +0x50
};
static_assert(sizeof(OrbisNpMatching2RoomMemberDataInternal) == 0x58);

struct OrbisNpMatching2RoomMemberDataInternalList {
    OrbisNpMatching2RoomMemberDataInternal* members;
    u64 membersNum;
    OrbisNpMatching2RoomMemberDataInternal* me;
    OrbisNpMatching2RoomMemberDataInternal* owner;
};

struct OrbisNpMatching2CreateJoinRoomResponse {
    const OrbisNpMatching2RoomDataInternal* roomData;
    OrbisNpMatching2RoomMemberDataInternalList members;
};

struct OrbisNpMatching2CreateJoinRoomResponseA {
    OrbisNpMatching2RoomDataInternal* roomData;
};

struct OrbisNpMatching2RoomDataExternalA {
    OrbisNpMatching2RoomDataExternalA* next;
    u16 maxSlot;
    u16 curMembers;
    OrbisNpMatching2Flags flags;
    OrbisNpMatching2ServerId serverId;
    u8 pad[2];
    OrbisNpMatching2WorldId worldId;
    OrbisNpMatching2LobbyId lobbyId;
    OrbisNpMatching2RoomId roomId;
    u64 passwdSlotMask;
    u64 joinedSlotMask;
    u16 publicSlots;
    u16 privateSlots;
    u16 openPublicSlots;
    u16 openPrivateSlots;
    Libraries::Np::OrbisNpPeerAddressA owner;
    Libraries::Np::OrbisNpOnlineId ownerOnlineId;
    OrbisNpMatching2RoomGroupInfo* roomGroup;
    u64 roomGroups;
    OrbisNpMatching2IntAttr* externalSearchIntAttr;
    u64 externalSearchIntAttrs;
    OrbisNpMatching2BinAttr* externalSearchBinAttr;
    u64 externalSearchBinAttrs;
    OrbisNpMatching2BinAttr* externalBinAttr;
    u64 externalBinAttrs;
};

struct OrbisNpMatching2RoomDataExternal {
    OrbisNpMatching2RoomDataExternal* next;
    u16 maxSlot;
    u16 curMembers;
    OrbisNpMatching2Flags flags;
    OrbisNpMatching2ServerId serverId;
    u8 pad[2];
    OrbisNpMatching2WorldId worldId;
    OrbisNpMatching2LobbyId lobbyId;
    OrbisNpMatching2RoomId roomId;
    u64 passwdSlotMask;
    u64 joinedSlotMask;
    u16 publicSlots;
    u16 privateSlots;
    u16 openPublicSlots;
    u16 openPrivateSlots;
    Libraries::Np::OrbisNpId* ownerNpId; // +0x40: room owner's NpId — game dereferences without null check in SearchRoom callback (0x106)
    OrbisNpMatching2RoomGroupInfo* roomGroup;
    u64 roomGroups;
    OrbisNpMatching2IntAttr* externalSearchIntAttr;
    u64 externalSearchIntAttrs;
    OrbisNpMatching2BinAttr* externalSearchBinAttr;
    u64 externalSearchBinAttrs;
    OrbisNpMatching2BinAttr* externalBinAttr;
    u64 externalBinAttrs;
};

static_assert(sizeof(OrbisNpMatching2RoomDataExternal) == 0x88);

struct OrbisNpMatching2Range {
    u32 start;
    u32 total;
    u32 results;
    u8 pad[4];
};

// 0x28 (40) bytes — confirmed via Ghidra: game iterates with stride 0x28
// and reads u32 fields at +0x08, +0x18, +0x1c per entry.
struct OrbisNpMatching2World {
    OrbisNpMatching2World* next;           // +0x00 [8] linked list
    OrbisNpMatching2WorldId worldId;       // +0x08 [4]
    u32 numOfLobby;                        // +0x0C [4] validity flag (u32 for alignment)
    u32 curNumOfTotalLobby;                // +0x10 [4]
    u32 maxNumOfTotalLobby;                // +0x14 [4]
    u32 curNumOfRoom;                      // +0x18 [4] ← game reads
    u32 maxNumOfRoom;                      // +0x1C [4] ← game reads
    u32 curNumOfTotalRoomMember;           // +0x20 [4]
    u32 numOfRoom;                         // +0x24 [4]
    u8 reserved[8];                        // +0x28 [8]
};

struct OrbisNpMatching2GetWorldInfoListResponse {
    OrbisNpMatching2World* world;
    u32 worldNum;
    u8 pad[4];
};

struct OrbisNpMatching2SearchRoomResponseA {
    OrbisNpMatching2Range range;
    OrbisNpMatching2RoomDataExternalA* roomDataExt;
};

struct OrbisNpMatching2SearchRoomResponse {
    OrbisNpMatching2Range range;
    OrbisNpMatching2RoomDataExternal* roomDataExt;
};

struct OrbisNpMatching2SignalingGetPingInfoRequest {
    OrbisNpMatching2RoomId roomId;
    u8 pad[16];

    int Validate() {
        return 0;
    }
};

struct OrbisNpMatching2SignalingGetPingInfoResponse {
    OrbisNpMatching2ServerId serverId;
    u8 pad[2];
    OrbisNpMatching2WorldId worldId;
    OrbisNpMatching2RoomId roomId;
    u32 pingUs;
    u8 reserved[20];
};

struct OrbisNpMatching2SignalingConnectionInfoAddr {
    u32 addr;
    u16 port;
    u8 pad[2];
};

struct OrbisNpMatching2SignalingConnectionInfo {
    OrbisNpMatching2SignalingConnectionInfoAddr address;
};

struct OrbisNpMatching2PresenceOptionData {
    u8 data[16];
    u64 len;
};

struct OrbisNpMatching2JoinRoomRequest {
    OrbisNpMatching2RoomId roomId;
    OrbisNpMatching2SessionPassword* roomPasswd;
    OrbisNpMatching2GroupLabel* joinGroupLabel;
    OrbisNpMatching2BinAttr* roomMemberBinInternalAttr;
    u64 roomMemberBinInternalAttrNum;
    OrbisNpMatching2PresenceOptionData optData;
    OrbisNpMatching2TeamId teamId;
    u8 pad[3];
    OrbisNpMatching2Flags flags;
    Libraries::Np::OrbisNpOnlineId* blockedUser;
    u64 blockedUsers;

    int Validate() {
        return 0;
    }
};

static_assert(sizeof(OrbisNpMatching2JoinRoomRequest) == 0x58);

struct OrbisNpMatching2LeaveRoomRequest {
    OrbisNpMatching2RoomId roomId;
    OrbisNpMatching2PresenceOptionData optData;

    int Validate() {
        return 0;
    }
};

struct OrbisNpMatching2LeaveRoomResponse {
    OrbisNpMatching2RoomId roomId;
};

struct OrbisNpMatching2RoomMemberUpdateInfo {
    OrbisNpMatching2RoomMemberDataInternal* roomMemberDataInternal;
    OrbisNpMatching2EventCause eventCause;
    u8 pad[7];
    OrbisNpMatching2PresenceOptionData optData;
};

struct OrbisNpMatching2RoomDataInternalUpdateInfo {
    OrbisNpMatching2RoomDataInternal* roomDataInternal;
    OrbisNpMatching2EventCause eventCause;
    u8 pad[7];
    OrbisNpMatching2PresenceOptionData optData;
};

using OrbisNpMatching2RequestCallback = PS4_SYSV_ABI void (*)(OrbisNpMatching2ContextId,
                                                              OrbisNpMatching2RequestId,
                                                              OrbisNpMatching2Event, int,
                                                              const void*, void*);
using OrbisNpMatching2RequestFn = PS4_SYSV_ABI void(OrbisNpMatching2ContextId,
                                                    OrbisNpMatching2RequestId,
                                                    OrbisNpMatching2Event, int, const void*,
                                                    void*);

struct OrbisNpMatching2RequestOptParam {
    OrbisNpMatching2RequestCallback callback;
    void* arg;
    u32 timeout;
    u16 appId;
    u8 dummy[2];
};

} // namespace Libraries::Np::NpMatching2
