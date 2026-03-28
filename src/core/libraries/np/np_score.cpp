// SPDX-FileCopyrightText: Copyright 2025 shadPS4 Emulator Project
// SPDX-License-Identifier: GPL-2.0-or-later

#include <atomic>
#include <cstring>
#include "common/logging/log.h"
#include "core/libraries/error_codes.h"
#include "core/libraries/libs.h"
#include "core/libraries/np/np_score.h"

namespace Libraries::Np::NpScore {

// Helper macro to format pointer safely
#define PTR(ptr) static_cast<const void*>(ptr)

// Monotonically increasing IDs for context and request handles.
// The game uses the return value of CreateNpTitleCtx/CreateRequest as an opaque
// handle for subsequent calls, so returning 0 (ORBIS_OK) would be mistaken for
// an error or an invalid handle by many titles.
static std::atomic<s32> s_ctxIdCounter{1};
static std::atomic<s32> s_reqIdCounter{1};

// Fake leaderboard constants ------------------------------------------------
// A plausible number of fake entries to populate per board query.
static constexpr u64 FAKE_ENTRY_COUNT = 3;
// Descending scores for ranks 1-3.
static constexpr OrbisNpScoreValue FAKE_SCORES[FAKE_ENTRY_COUNT] = {
    9'999'999LL,
    7'500'000LL,
    5'000'000LL,
};
// Fake online-ID strings written into RankDataA entries.
static constexpr const char* FAKE_ONLINE_IDS[FAKE_ENTRY_COUNT] = {
    "FakePlayer01",
    "FakePlayer02",
    "FakePlayer03",
};
// Arbitrary PS4 RTC tick value representing a past date
// (microseconds since the SCE epoch 2000-01-01 00:00:00 UTC).
static constexpr u64 FAKE_RTC_TICK = 0x00038D7E'A4C68000ULL;
// Fake account IDs (must be non-zero so the game treats them as valid).
static constexpr OrbisNpAccountId FAKE_ACCOUNT_IDS[FAKE_ENTRY_COUNT] = {
    0x0000'0001'0000'0001ULL,
    0x0000'0001'0000'0002ULL,
    0x0000'0001'0000'0003ULL,
};

// ---------------------------------------------------------------------------
// Internal helpers
// ---------------------------------------------------------------------------

// OrbisNpId layout (from PS4 SDK / np_types.h):
//   struct OrbisNpOnlineId { char data[16]; char term; char padding[3]; }; // 20 bytes
//   struct OrbisNpId { OrbisNpOnlineId handle; u8 opt[8]; u8 reserved[8]; }; // 36 bytes
// The online-ID string lives at byte offset 0 of OrbisNpId.
static constexpr u64 NP_ONLINE_ID_DATA_OFFSET = 0;
static constexpr u64 NP_ONLINE_ID_MAX_LEN = 16; // char data[16], last byte must be '\0'

static void WriteNpIdName(OrbisNpId& npId, const char* name) {
    char* dst = reinterpret_cast<char*>(&npId) + NP_ONLINE_ID_DATA_OFFSET;
    std::strncpy(dst, name, NP_ONLINE_ID_MAX_LEN - 1);
    dst[NP_ONLINE_ID_MAX_LEN - 1] = '\0';
}

// Fill up to FAKE_ENTRY_COUNT entries of OrbisNpScoreRankData (legacy / compat variant).
// If npIdArray is provided the caller's NpIds are copied back into the output entries so
// the game sees the real online-ID strings it originally passed in.
static void FillRankData(OrbisNpScoreRankData* rankArray, u64 arrayNum,
                         Rtc::OrbisRtcTick* lastSortDate,
                         OrbisNpScoreRankNumber* totalRecord,
                         const OrbisNpId* npIdArray = nullptr) {
    if (!rankArray || arrayNum == 0) {
        if (totalRecord)
            *totalRecord = 0;
        return;
    }
    const u64 fillCount = (arrayNum < FAKE_ENTRY_COUNT) ? arrayNum : FAKE_ENTRY_COUNT;
    for (u64 i = 0; i < fillCount; ++i) {
        OrbisNpScoreRankData& e = rankArray[i];
        std::memset(&e, 0, sizeof(e));
        if (npIdArray) {
            // Mirror the caller's NpId back so the game sees the real online ID string.
            e.npId = npIdArray[i];
        } else {
            // Generic fake name: "FakePlayer01", "FakePlayer02", ...
            // OrbisNpId starts with OrbisNpOnlineId whose first member is char data[16].
            WriteNpIdName(e.npId, FAKE_ONLINE_IDS[i]);
        }
        e.pcId = 0;
        e.serialRank = static_cast<OrbisNpScoreRankNumber>(i + 1);
        e.rank = static_cast<OrbisNpScoreRankNumber>(i + 1);
        e.highestRank = static_cast<OrbisNpScoreRankNumber>(i + 1);
        e.scoreValue = FAKE_SCORES[i];
        e.hasGameData = 0;
        e.recordDate.tick = FAKE_RTC_TICK;
    }
    if (totalRecord)
        *totalRecord = static_cast<OrbisNpScoreRankNumber>(fillCount);
    if (lastSortDate)
        lastSortDate->tick = FAKE_RTC_TICK;
}

// Fill up to FAKE_ENTRY_COUNT entries of OrbisNpScoreRankDataA (extended variant).
static void FillRankDataA(OrbisNpScoreRankDataA* rankArray, u64 arrayNum,
                          Rtc::OrbisRtcTick* lastSortDate,
                          OrbisNpScoreRankNumber* totalRecord) {
    if (!rankArray || arrayNum == 0) {
        if (totalRecord)
            *totalRecord = 0;
        return;
    }
    const u64 fillCount = (arrayNum < FAKE_ENTRY_COUNT) ? arrayNum : FAKE_ENTRY_COUNT;
    for (u64 i = 0; i < fillCount; ++i) {
        OrbisNpScoreRankDataA& e = rankArray[i];
        std::memset(&e, 0, sizeof(e));
        // OrbisNpOnlineId::data is a char[16] with a separate term byte.
        std::strncpy(e.onlineId.data, FAKE_ONLINE_IDS[i], sizeof(e.onlineId.data) - 1);
        e.onlineId.data[sizeof(e.onlineId.data) - 1] = '\0';
        e.pcId = 0;
        e.serialRank = static_cast<OrbisNpScoreRankNumber>(i + 1);
        e.rank = static_cast<OrbisNpScoreRankNumber>(i + 1);
        e.highestRank = static_cast<OrbisNpScoreRankNumber>(i + 1);
        e.scoreValue = FAKE_SCORES[i];
        e.hasGameData = 0;
        e.recordDate.tick = FAKE_RTC_TICK;
        e.accountId = FAKE_ACCOUNT_IDS[i];
    }
    if (totalRecord)
        *totalRecord = static_cast<OrbisNpScoreRankNumber>(fillCount);
    if (lastSortDate)
        lastSortDate->tick = FAKE_RTC_TICK;
}

// Fill up to FAKE_ENTRY_COUNT entries of OrbisNpScorePlayerRankDataA
// (used by GetRankingByAccountId / GetRankingByNpId-style queries where each
//  slot corresponds to a requested ID rather than a contiguous range).
static void FillPlayerRankDataA(OrbisNpScorePlayerRankDataA* rankArray, u64 arrayNum,
                                Rtc::OrbisRtcTick* lastSortDate,
                                OrbisNpScoreRankNumber* totalRecord) {
    if (!rankArray || arrayNum == 0) {
        if (totalRecord)
            *totalRecord = 0;
        return;
    }
    const u64 fillCount = (arrayNum < FAKE_ENTRY_COUNT) ? arrayNum : FAKE_ENTRY_COUNT;
    for (u64 i = 0; i < fillCount; ++i) {
        OrbisNpScorePlayerRankDataA& p = rankArray[i];
        std::memset(&p, 0, sizeof(p));
        p.hasData = 1;
        OrbisNpScoreRankDataA& e = p.rankData;
        std::strncpy(e.onlineId.data, FAKE_ONLINE_IDS[i], sizeof(e.onlineId.data) - 1);
        e.onlineId.data[sizeof(e.onlineId.data) - 1] = '\0';
        e.pcId = 0;
        e.serialRank = static_cast<OrbisNpScoreRankNumber>(i + 1);
        e.rank = static_cast<OrbisNpScoreRankNumber>(i + 1);
        e.highestRank = static_cast<OrbisNpScoreRankNumber>(i + 1);
        e.scoreValue = FAKE_SCORES[i];
        e.hasGameData = 0;
        e.recordDate.tick = FAKE_RTC_TICK;
        e.accountId = FAKE_ACCOUNT_IDS[i];
    }
    if (totalRecord)
        *totalRecord = static_cast<OrbisNpScoreRankNumber>(fillCount);
    if (lastSortDate)
        lastSortDate->tick = FAKE_RTC_TICK;
}

// Fill up to FAKE_ENTRY_COUNT entries of OrbisNpScoreRankDataForCrossSave.
static void FillRankDataForCrossSave(OrbisNpScoreRankDataForCrossSave* rankArray, u64 arrayNum,
                                     Rtc::OrbisRtcTick* lastSortDate,
                                     OrbisNpScoreRankNumber* totalRecord) {
    if (!rankArray || arrayNum == 0) {
        if (totalRecord)
            *totalRecord = 0;
        return;
    }
    const u64 fillCount = (arrayNum < FAKE_ENTRY_COUNT) ? arrayNum : FAKE_ENTRY_COUNT;
    for (u64 i = 0; i < fillCount; ++i) {
        OrbisNpScoreRankDataForCrossSave& e = rankArray[i];
        std::memset(&e, 0, sizeof(e));
        e.pcId = 0;
        e.serialRank = static_cast<OrbisNpScoreRankNumber>(i + 1);
        e.rank = static_cast<OrbisNpScoreRankNumber>(i + 1);
        e.highestRank = static_cast<OrbisNpScoreRankNumber>(i + 1);
        e.scoreValue = FAKE_SCORES[i];
        e.hasGameData = 0;
        e.recordDate.tick = FAKE_RTC_TICK;
        e.accountId = FAKE_ACCOUNT_IDS[i];
    }
    if (totalRecord)
        *totalRecord = static_cast<OrbisNpScoreRankNumber>(fillCount);
    if (lastSortDate)
        lastSortDate->tick = FAKE_RTC_TICK;
}

// ---------------------------------------------------------------------------
// API implementations
// ---------------------------------------------------------------------------

int PS4_SYSV_ABI sceNpScoreAbortRequest(s32 reqId) {
    LOG_INFO(Lib_NpScore, "(STUBBED) called reqId={}", reqId);
    return ORBIS_OK;
}

int PS4_SYSV_ABI sceNpScoreCensorComment(s32 reqId, const char* comment, void* option) {
    LOG_INFO(Lib_NpScore, "(STUBBED) called reqId={}, comment={}, option={}", reqId,
             comment ? comment : "null", PTR(option));
    return ORBIS_OK;
}

int PS4_SYSV_ABI sceNpScoreCensorCommentAsync(s32 reqId, const char* comment, void* option) {
    LOG_INFO(Lib_NpScore, "(STUBBED) called reqId={}, comment={}, option={}", reqId,
             comment ? comment : "null", PTR(option));
    return ORBIS_OK;
}

int PS4_SYSV_ABI sceNpScoreChangeModeForOtherSaveDataOwners() {
    LOG_INFO(Lib_NpScore, "(STUBBED) called");
    return ORBIS_OK;
}

int PS4_SYSV_ABI sceNpScoreCreateNpTitleCtx() {
    const s32 id = s_ctxIdCounter++;
    LOG_INFO(Lib_NpScore, "(FAKE) called -> ctxId={}", id);
    return id;
}

int PS4_SYSV_ABI sceNpScoreCreateNpTitleCtxA(OrbisNpServiceLabel npServiceLabel,
                                             UserService::OrbisUserServiceUserId selfId) {
    const s32 id = s_ctxIdCounter++;
    LOG_INFO(Lib_NpScore, "(FAKE) called npServiceLabel={}, selfId={} -> ctxId={}",
             static_cast<u32>(npServiceLabel), selfId, id);
    return id;
}

int PS4_SYSV_ABI sceNpScoreCreateRequest(s32 titleCtxId) {
    const s32 id = s_reqIdCounter++;
    LOG_INFO(Lib_NpScore, "(FAKE) called titleCtxId={} -> reqId={}", titleCtxId, id);
    return id;
}

int PS4_SYSV_ABI sceNpScoreCreateTitleCtx() {
    const s32 id = s_ctxIdCounter++;
    LOG_INFO(Lib_NpScore, "(FAKE) called -> ctxId={}", id);
    return id;
}

int PS4_SYSV_ABI sceNpScoreDeleteNpTitleCtx(s32 titleCtxId) {
    LOG_INFO(Lib_NpScore, "(STUBBED) called titleCtxId={}", titleCtxId);
    return ORBIS_OK;
}

int PS4_SYSV_ABI sceNpScoreDeleteRequest(s32 reqId) {
    LOG_INFO(Lib_NpScore, "(STUBBED) called reqId={}", reqId);
    return ORBIS_OK;
}

int PS4_SYSV_ABI sceNpScoreGetBoardInfo(s32 reqId, OrbisNpScoreBoardId boardId,
                                        OrbisNpScoreBoardInfo* boardInfo, void* option) {
    LOG_INFO(Lib_NpScore, "(STUBBED) called reqId={}, boardId={}, boardInfo={}, option={}", reqId,
             boardId, PTR(boardInfo), PTR(option));
    if (boardInfo) {
        std::memset(boardInfo, 0, sizeof(*boardInfo));
        boardInfo->rankLimit = 100;
        boardInfo->uploadNumLimit = 1;
    }
    return ORBIS_OK;
}

int PS4_SYSV_ABI sceNpScoreGetBoardInfoAsync(s32 reqId, OrbisNpScoreBoardId boardId,
                                             OrbisNpScoreBoardInfo* boardInfo, void* option) {
    LOG_INFO(Lib_NpScore, "(STUBBED) called reqId={}, boardId={}, boardInfo={}, option={}", reqId,
             boardId, PTR(boardInfo), PTR(option));
    if (boardInfo) {
        std::memset(boardInfo, 0, sizeof(*boardInfo));
        boardInfo->rankLimit = 100;
        boardInfo->uploadNumLimit = 1;
    }
    return ORBIS_OK;
}

int PS4_SYSV_ABI sceNpScoreGetFriendsRanking(s32 reqId, OrbisNpScoreBoardId boardId,
                                             s32 includeSelf, OrbisNpScoreRankData* rankArray,
                                             u64 rankArraySize, OrbisNpScoreComment* commentArray,
                                             u64 commentArraySize, OrbisNpScoreGameInfo* infoArray,
                                             u64 infoArraySize, u64 arrayNum,
                                             Rtc::OrbisRtcTick* lastSortDate,
                                             OrbisNpScoreRankNumber* totalRecord, void* option) {
    LOG_INFO(Lib_NpScore,
             "(FAKE) called reqId={}, boardId={}, includeSelf={}, arrayNum={}",
             reqId, boardId, includeSelf, arrayNum);
    FillRankData(rankArray, arrayNum, lastSortDate, totalRecord);
    return ORBIS_OK;
}

int PS4_SYSV_ABI sceNpScoreGetFriendsRankingA(s32 reqId, OrbisNpScoreBoardId boardId,
                                              s32 includeSelf, OrbisNpScoreRankDataA* rankArray,
                                              u64 rankArraySize, OrbisNpScoreComment* commentArray,
                                              u64 commentArraySize, OrbisNpScoreGameInfo* infoArray,
                                              u64 infoArraySize, u64 arrayNum,
                                              Rtc::OrbisRtcTick* lastSortDate,
                                              OrbisNpScoreRankNumber* totalRecord,
                                              OrbisNpScoreGetFriendRankingOptParam* option) {
    LOG_INFO(Lib_NpScore,
             "(FAKE) called reqId={}, boardId={}, includeSelf={}, arrayNum={}",
             reqId, boardId, includeSelf, arrayNum);
    FillRankDataA(rankArray, arrayNum, lastSortDate, totalRecord);
    return ORBIS_OK;
}

int PS4_SYSV_ABI sceNpScoreGetFriendsRankingAAsync(
    s32 reqId, OrbisNpScoreBoardId boardId, s32 includeSelf, OrbisNpScoreRankDataA* rankArray,
    u64 rankArraySize, OrbisNpScoreComment* commentArray, u64 commentArraySize,
    OrbisNpScoreGameInfo* infoArray, u64 infoArraySize, u64 arrayNum,
    Rtc::OrbisRtcTick* lastSortDate, OrbisNpScoreRankNumber* totalRecord,
    OrbisNpScoreGetFriendRankingOptParam* option) {
    LOG_INFO(Lib_NpScore,
             "(FAKE) called reqId={}, boardId={}, includeSelf={}, arrayNum={}",
             reqId, boardId, includeSelf, arrayNum);
    FillRankDataA(rankArray, arrayNum, lastSortDate, totalRecord);
    return ORBIS_OK;
}

int PS4_SYSV_ABI sceNpScoreGetFriendsRankingAsync() {
    LOG_INFO(Lib_NpScore, "(STUBBED) called");
    return ORBIS_OK;
}

int PS4_SYSV_ABI sceNpScoreGetFriendsRankingForCrossSave(
    s32 reqId, OrbisNpScoreBoardId boardId, s32 includeSelf,
    OrbisNpScoreRankDataForCrossSave* rankArray, u64 rankArraySize,
    OrbisNpScoreComment* commentArray, u64 commentArraySize, OrbisNpScoreGameInfo* infoArray,
    u64 infoArraySize, u64 arrayNum, Rtc::OrbisRtcTick* lastSortDate,
    OrbisNpScoreRankNumber* totalRecord, OrbisNpScoreGetFriendRankingOptParam* option) {
    LOG_INFO(Lib_NpScore,
             "(FAKE) called reqId={}, boardId={}, includeSelf={}, arrayNum={}",
             reqId, boardId, includeSelf, arrayNum);
    FillRankDataForCrossSave(rankArray, arrayNum, lastSortDate, totalRecord);
    return ORBIS_OK;
}

int PS4_SYSV_ABI sceNpScoreGetFriendsRankingForCrossSaveAsync(
    s32 reqId, OrbisNpScoreBoardId boardId, s32 includeSelf,
    OrbisNpScoreRankDataForCrossSave* rankArray, u64 rankArraySize,
    OrbisNpScoreComment* commentArray, u64 commentArraySize, OrbisNpScoreGameInfo* infoArray,
    u64 infoArraySize, u64 arrayNum, Rtc::OrbisRtcTick* lastSortDate,
    OrbisNpScoreRankNumber* totalRecord, OrbisNpScoreGetFriendRankingOptParam* option) {
    LOG_INFO(Lib_NpScore,
             "(FAKE) called reqId={}, boardId={}, includeSelf={}, arrayNum={}",
             reqId, boardId, includeSelf, arrayNum);
    FillRankDataForCrossSave(rankArray, arrayNum, lastSortDate, totalRecord);
    return ORBIS_OK;
}

int PS4_SYSV_ABI sceNpScoreGetGameData() {
    LOG_INFO(Lib_NpScore, "(STUBBED) called");
    return ORBIS_OK;
}

int PS4_SYSV_ABI sceNpScoreGetGameDataAsync() {
    LOG_INFO(Lib_NpScore, "(STUBBED) called");
    return ORBIS_OK;
}

int PS4_SYSV_ABI sceNpScoreGetGameDataByAccountId(s32 reqId, OrbisNpScoreBoardId boardId,
                                                  OrbisNpAccountId accountId, u64* totalSize,
                                                  u64 recvSize, void* data, void* option) {
    LOG_INFO(Lib_NpScore,
             "(STUBBED) called reqId={}, boardId={}, accountId={}, "
             "totalSize={}, recvSize={}, data={}, option={}",
             reqId, boardId, accountId, PTR(totalSize), recvSize, PTR(data), PTR(option));
    if (totalSize)
        *totalSize = 0;
    return ORBIS_OK;
}

int PS4_SYSV_ABI sceNpScoreGetGameDataByAccountIdAsync(s32 reqId, OrbisNpScoreBoardId boardId,
                                                       OrbisNpAccountId accountId, u64* totalSize,
                                                       u64 recvSize, void* data, void* option) {
    LOG_INFO(Lib_NpScore,
             "(STUBBED) called reqId={}, boardId={}, accountId={}, "
             "totalSize={}, recvSize={}, data={}, option={}",
             reqId, boardId, accountId, PTR(totalSize), recvSize, PTR(data), PTR(option));
    if (totalSize)
        *totalSize = 0;
    return ORBIS_OK;
}

int PS4_SYSV_ABI sceNpScoreGetRankingByAccountId(
    s32 reqId, OrbisNpScoreBoardId boardId, const OrbisNpAccountId* accountIdArray,
    u64 accountIdArraySize, OrbisNpScorePlayerRankDataA* rankArray, u64 rankArraySize,
    OrbisNpScoreComment* commentArray, u64 commentArraySize, OrbisNpScoreGameInfo* infoArray,
    u64 infoArraySize, u64 arrayNum, Rtc::OrbisRtcTick* lastSortDate,
    OrbisNpScoreRankNumber* totalRecord, void* option) {
    LOG_INFO(Lib_NpScore,
             "(FAKE) called reqId={}, boardId={}, accountIdArraySize={}, arrayNum={}",
             reqId, boardId, accountIdArraySize, arrayNum);
    FillPlayerRankDataA(rankArray, arrayNum, lastSortDate, totalRecord);
    return ORBIS_OK;
}

int PS4_SYSV_ABI sceNpScoreGetRankingByAccountIdAsync(
    s32 reqId, OrbisNpScoreBoardId boardId, const OrbisNpAccountId* accountIdArray,
    u64 accountIdArraySize, OrbisNpScorePlayerRankDataA* rankArray, u64 rankArraySize,
    OrbisNpScoreComment* commentArray, u64 commentArraySize, OrbisNpScoreGameInfo* infoArray,
    u64 infoArraySize, u64 arrayNum, Rtc::OrbisRtcTick* lastSortDate,
    OrbisNpScoreRankNumber* totalRecord, void* option) {
    LOG_INFO(Lib_NpScore,
             "(FAKE) called reqId={}, boardId={}, accountIdArraySize={}, arrayNum={}",
             reqId, boardId, accountIdArraySize, arrayNum);
    FillPlayerRankDataA(rankArray, arrayNum, lastSortDate, totalRecord);
    return ORBIS_OK;
}

int PS4_SYSV_ABI sceNpScoreGetRankingByAccountIdForCrossSave(
    s32 reqId, OrbisNpScoreBoardId boardId, const OrbisNpAccountId* accountIdArray,
    u64 accountIdArraySize, OrbisNpScorePlayerRankDataForCrossSave* rankArray, u64 rankArraySize,
    OrbisNpScoreComment* commentArray, u64 commentArraySize, OrbisNpScoreGameInfo* infoArray,
    u64 infoArraySize, u64 arrayNum, Rtc::OrbisRtcTick* lastSortDate,
    OrbisNpScoreRankNumber* totalRecord, void* option) {
    LOG_INFO(Lib_NpScore,
             "(STUBBED) called reqId={}, boardId={}, accountIdArraySize={}, arrayNum={}",
             reqId, boardId, accountIdArraySize, arrayNum);
    if (totalRecord)
        *totalRecord = 0;
    return ORBIS_OK;
}

int PS4_SYSV_ABI sceNpScoreGetRankingByAccountIdForCrossSaveAsync(
    s32 reqId, OrbisNpScoreBoardId boardId, const OrbisNpAccountId* accountIdArray,
    u64 accountIdArraySize, OrbisNpScorePlayerRankDataForCrossSave* rankArray, u64 rankArraySize,
    OrbisNpScoreComment* commentArray, u64 commentArraySize, OrbisNpScoreGameInfo* infoArray,
    u64 infoArraySize, u64 arrayNum, Rtc::OrbisRtcTick* lastSortDate,
    OrbisNpScoreRankNumber* totalRecord, void* option) {
    LOG_INFO(Lib_NpScore,
             "(STUBBED) called reqId={}, boardId={}, accountIdArraySize={}, arrayNum={}",
             reqId, boardId, accountIdArraySize, arrayNum);
    if (totalRecord)
        *totalRecord = 0;
    return ORBIS_OK;
}

int PS4_SYSV_ABI sceNpScoreGetRankingByAccountIdPcId(
    s32 reqId, OrbisNpScoreBoardId boardId, const OrbisNpScoreAccountIdPcId* idArray,
    u64 idArraySize, OrbisNpScorePlayerRankDataA* rankArray, u64 rankArraySize,
    OrbisNpScoreComment* commentArray, u64 commentArraySize, OrbisNpScoreGameInfo* infoArray,
    u64 infoArraySize, u64 arrayNum, Rtc::OrbisRtcTick* lastSortDate,
    OrbisNpScoreRankNumber* totalRecord, void* option) {
    LOG_INFO(Lib_NpScore,
             "(FAKE) called reqId={}, boardId={}, idArraySize={}, arrayNum={}",
             reqId, boardId, idArraySize, arrayNum);
    FillPlayerRankDataA(rankArray, arrayNum, lastSortDate, totalRecord);
    return ORBIS_OK;
}

int PS4_SYSV_ABI sceNpScoreGetRankingByAccountIdPcIdAsync(
    s32 reqId, OrbisNpScoreBoardId boardId, const OrbisNpScoreAccountIdPcId* idArray,
    u64 idArraySize, OrbisNpScorePlayerRankDataA* rankArray, u64 rankArraySize,
    OrbisNpScoreComment* commentArray, u64 commentArraySize, OrbisNpScoreGameInfo* infoArray,
    u64 infoArraySize, u64 arrayNum, Rtc::OrbisRtcTick* lastSortDate,
    OrbisNpScoreRankNumber* totalRecord, void* option) {
    LOG_INFO(Lib_NpScore,
             "(FAKE) called reqId={}, boardId={}, idArraySize={}, arrayNum={}",
             reqId, boardId, idArraySize, arrayNum);
    FillPlayerRankDataA(rankArray, arrayNum, lastSortDate, totalRecord);
    return ORBIS_OK;
}

int PS4_SYSV_ABI sceNpScoreGetRankingByAccountIdPcIdForCrossSave(
    s32 reqId, OrbisNpScoreBoardId boardId, const OrbisNpScoreAccountIdPcId* idArray,
    u64 idArraySize, OrbisNpScorePlayerRankDataForCrossSave* rankArray, u64 rankArraySize,
    OrbisNpScoreComment* commentArray, u64 commentArraySize, OrbisNpScoreGameInfo* infoArray,
    u64 infoArraySize, u64 arrayNum, Rtc::OrbisRtcTick* lastSortDate,
    OrbisNpScoreRankNumber* totalRecord, void* option) {
    LOG_INFO(Lib_NpScore,
             "(STUBBED) called reqId={}, boardId={}, idArraySize={}, arrayNum={}",
             reqId, boardId, idArraySize, arrayNum);
    if (totalRecord)
        *totalRecord = 0;
    return ORBIS_OK;
}

int PS4_SYSV_ABI sceNpScoreGetRankingByAccountIdPcIdForCrossSaveAsync(
    s32 reqId, OrbisNpScoreBoardId boardId, const OrbisNpScoreAccountIdPcId* idArray,
    u64 idArraySize, OrbisNpScorePlayerRankDataForCrossSave* rankArray, u64 rankArraySize,
    OrbisNpScoreComment* commentArray, u64 commentArraySize, OrbisNpScoreGameInfo* infoArray,
    u64 infoArraySize, u64 arrayNum, Rtc::OrbisRtcTick* lastSortDate,
    OrbisNpScoreRankNumber* totalRecord, void* option) {
    LOG_INFO(Lib_NpScore,
             "(STUBBED) called reqId={}, boardId={}, idArraySize={}, arrayNum={}",
             reqId, boardId, idArraySize, arrayNum);
    if (totalRecord)
        *totalRecord = 0;
    return ORBIS_OK;
}

int PS4_SYSV_ABI sceNpScoreGetRankingByNpId(s32 reqId, OrbisNpScoreBoardId boardId,
                                            const OrbisNpId* npIdArray, u64 npIdArraySize,
                                            OrbisNpScoreRankData* rankArray, u64 rankArraySize,
                                            OrbisNpScoreComment* commentArray,
                                            u64 commentArraySize, OrbisNpScoreGameInfo* infoArray,
                                            u64 infoArraySize, u64 arrayNum,
                                            Rtc::OrbisRtcTick* lastSortDate,
                                            OrbisNpScoreRankNumber* totalRecord, void* option) {
    LOG_INFO(Lib_NpScore,
             "(FAKE) called reqId={}, boardId={}, npIdArraySize={}, arrayNum={}",
             reqId, boardId, npIdArraySize, arrayNum);
    // Mirror the caller's NpIds back into the output entries so the game sees the
    // player's real online-ID string.  This is how a real server would respond:
    // the entry's npId always matches the one that was queried.
    FillRankData(rankArray, arrayNum, lastSortDate, totalRecord, npIdArray);
    return ORBIS_OK;
}

int PS4_SYSV_ABI sceNpScoreGetRankingByNpIdAsync(s32 reqId, OrbisNpScoreBoardId boardId,
                                                 const OrbisNpId* npIdArray, u64 npIdArraySize,
                                                 OrbisNpScoreRankData* rankArray, u64 rankArraySize,
                                                 OrbisNpScoreComment* commentArray,
                                                 u64 commentArraySize,
                                                 OrbisNpScoreGameInfo* infoArray, u64 infoArraySize,
                                                 u64 arrayNum, Rtc::OrbisRtcTick* lastSortDate,
                                                 OrbisNpScoreRankNumber* totalRecord, void* option) {
    LOG_INFO(Lib_NpScore,
             "(FAKE) called reqId={}, boardId={}, npIdArraySize={}, arrayNum={}",
             reqId, boardId, npIdArraySize, arrayNum);
    FillRankData(rankArray, arrayNum, lastSortDate, totalRecord, npIdArray);
    return ORBIS_OK;
}

int PS4_SYSV_ABI sceNpScoreGetRankingByNpIdPcId(s32 reqId, OrbisNpScoreBoardId boardId,
                                                const OrbisNpScoreNpIdPcId* idArray,
                                                u64 idArraySize, OrbisNpScoreRankData* rankArray,
                                                u64 rankArraySize,
                                                OrbisNpScoreComment* commentArray,
                                                u64 commentArraySize,
                                                OrbisNpScoreGameInfo* infoArray, u64 infoArraySize,
                                                u64 arrayNum, Rtc::OrbisRtcTick* lastSortDate,
                                                OrbisNpScoreRankNumber* totalRecord, void* option) {
    LOG_INFO(Lib_NpScore,
             "(FAKE) called reqId={}, boardId={}, idArraySize={}, arrayNum={}",
             reqId, boardId, idArraySize, arrayNum);
    FillRankData(rankArray, arrayNum, lastSortDate, totalRecord);
    return ORBIS_OK;
}

int PS4_SYSV_ABI sceNpScoreGetRankingByNpIdPcIdAsync(
    s32 reqId, OrbisNpScoreBoardId boardId, const OrbisNpScoreNpIdPcId* idArray, u64 idArraySize,
    OrbisNpScoreRankData* rankArray, u64 rankArraySize, OrbisNpScoreComment* commentArray,
    u64 commentArraySize, OrbisNpScoreGameInfo* infoArray, u64 infoArraySize, u64 arrayNum,
    Rtc::OrbisRtcTick* lastSortDate, OrbisNpScoreRankNumber* totalRecord, void* option) {
    LOG_INFO(Lib_NpScore,
             "(FAKE) called reqId={}, boardId={}, idArraySize={}, arrayNum={}",
             reqId, boardId, idArraySize, arrayNum);
    FillRankData(rankArray, arrayNum, lastSortDate, totalRecord);
    return ORBIS_OK;
}

int PS4_SYSV_ABI sceNpScoreGetRankingByRange(s32 reqId, OrbisNpScoreBoardId boardId,
                                             OrbisNpScoreRankNumber startSerialRank,
                                             OrbisNpScoreRankData* rankArray, u64 rankArraySize,
                                             OrbisNpScoreComment* commentArray,
                                             u64 commentArraySize, OrbisNpScoreGameInfo* infoArray,
                                             u64 infoArraySize, u64 arrayNum,
                                             Rtc::OrbisRtcTick* lastSortDate,
                                             OrbisNpScoreRankNumber* totalRecord, void* option) {
    LOG_INFO(Lib_NpScore,
             "(FAKE) called reqId={}, boardId={}, startSerialRank={}, arrayNum={}",
             reqId, boardId, startSerialRank, arrayNum);
    FillRankData(rankArray, arrayNum, lastSortDate, totalRecord);
    return ORBIS_OK;
}

int PS4_SYSV_ABI sceNpScoreGetRankingByRangeA(
    s32 reqId, OrbisNpScoreBoardId boardId, OrbisNpScoreRankNumber startSerialRank,
    OrbisNpScoreRankDataA* rankArray, u64 rankArraySize, OrbisNpScoreComment* commentArray,
    u64 commentArraySize, OrbisNpScoreGameInfo* infoArray, u64 infoArraySize, u64 arrayNum,
    Rtc::OrbisRtcTick* lastSortDate, OrbisNpScoreRankNumber* totalRecord, void* option) {
    LOG_INFO(Lib_NpScore,
             "(FAKE) called reqId={}, boardId={}, startSerialRank={}, arrayNum={}",
             reqId, boardId, startSerialRank, arrayNum);
    FillRankDataA(rankArray, arrayNum, lastSortDate, totalRecord);
    return ORBIS_OK;
}

int PS4_SYSV_ABI sceNpScoreGetRankingByRangeAAsync(
    s32 reqId, OrbisNpScoreBoardId boardId, OrbisNpScoreRankNumber startSerialRank,
    OrbisNpScoreRankDataA* rankArray, u64 rankArraySize, OrbisNpScoreComment* commentArray,
    u64 commentArraySize, OrbisNpScoreGameInfo* infoArray, u64 infoArraySize, u64 arrayNum,
    Rtc::OrbisRtcTick* lastSortDate, OrbisNpScoreRankNumber* totalRecord, void* option) {
    LOG_INFO(Lib_NpScore,
             "(FAKE) called reqId={}, boardId={}, startSerialRank={}, arrayNum={}",
             reqId, boardId, startSerialRank, arrayNum);
    FillRankDataA(rankArray, arrayNum, lastSortDate, totalRecord);
    return ORBIS_OK;
}

int PS4_SYSV_ABI sceNpScoreGetRankingByRangeAsync(
    s32 reqId, OrbisNpScoreBoardId boardId, OrbisNpScoreRankNumber startSerialRank,
    OrbisNpScoreRankData* rankArray, u64 rankArraySize, OrbisNpScoreComment* commentArray,
    u64 commentArraySize, OrbisNpScoreGameInfo* infoArray, u64 infoArraySize, u64 arrayNum,
    Rtc::OrbisRtcTick* lastSortDate, OrbisNpScoreRankNumber* totalRecord, void* option) {
    LOG_INFO(Lib_NpScore,
             "(FAKE) called reqId={}, boardId={}, startSerialRank={}, arrayNum={}",
             reqId, boardId, startSerialRank, arrayNum);
    FillRankData(rankArray, arrayNum, lastSortDate, totalRecord);
    return ORBIS_OK;
}

int PS4_SYSV_ABI sceNpScoreGetRankingByRangeForCrossSave(
    s32 reqId, OrbisNpScoreBoardId boardId, OrbisNpScoreRankNumber startSerialRank,
    OrbisNpScoreRankDataForCrossSave* rankArray, u64 rankArraySize,
    OrbisNpScoreComment* commentArray, u64 commentArraySize, OrbisNpScoreGameInfo* infoArray,
    u64 infoArraySize, u64 arrayNum, Rtc::OrbisRtcTick* lastSortDate,
    OrbisNpScoreRankNumber* totalRecord, void* option) {
    LOG_INFO(Lib_NpScore,
             "(FAKE) called reqId={}, boardId={}, startSerialRank={}, arrayNum={}",
             reqId, boardId, startSerialRank, arrayNum);
    FillRankDataForCrossSave(rankArray, arrayNum, lastSortDate, totalRecord);
    return ORBIS_OK;
}

int PS4_SYSV_ABI sceNpScoreGetRankingByRangeForCrossSaveAsync(
    s32 reqId, OrbisNpScoreBoardId boardId, OrbisNpScoreRankNumber startSerialRank,
    OrbisNpScoreRankDataForCrossSave* rankArray, u64 rankArraySize,
    OrbisNpScoreComment* commentArray, u64 commentArraySize, OrbisNpScoreGameInfo* infoArray,
    u64 infoArraySize, u64 arrayNum, Rtc::OrbisRtcTick* lastSortDate,
    OrbisNpScoreRankNumber* totalRecord, void* option) {
    LOG_INFO(Lib_NpScore,
             "(FAKE) called reqId={}, boardId={}, startSerialRank={}, arrayNum={}",
             reqId, boardId, startSerialRank, arrayNum);
    FillRankDataForCrossSave(rankArray, arrayNum, lastSortDate, totalRecord);
    return ORBIS_OK;
}

int PS4_SYSV_ABI sceNpScorePollAsync(s32 reqId, s32* result) {
    LOG_INFO(Lib_NpScore, "(STUBBED) called reqId={}, result={}", reqId, PTR(result));
    // Signal that the async operation completed successfully.
    if (result)
        *result = ORBIS_OK;
    return ORBIS_OK;
}

int PS4_SYSV_ABI sceNpScoreRecordGameData(s32 reqId, OrbisNpScoreBoardId boardId,
                                          OrbisNpScoreValue score, u64 totalSize, u64 sendSize,
                                          const void* data, void* option) {
    LOG_INFO(Lib_NpScore,
             "(STUBBED) called reqId={}, boardId={}, score={}, totalSize={}, "
             "sendSize={}, data={}, option={}",
             reqId, boardId, score, totalSize, sendSize, PTR(data), PTR(option));
    return ORBIS_OK;
}

int PS4_SYSV_ABI sceNpScoreRecordGameDataAsync(s32 reqId, OrbisNpScoreBoardId boardId,
                                               OrbisNpScoreValue score, u64 totalSize, u64 sendSize,
                                               const void* data, void* option) {
    LOG_INFO(Lib_NpScore,
             "(STUBBED) called reqId={}, boardId={}, score={}, "
             "totalSize={}, sendSize={}, data={}, option={}",
             reqId, boardId, score, totalSize, sendSize, PTR(data), PTR(option));
    return ORBIS_OK;
}

int PS4_SYSV_ABI sceNpScoreRecordScore(s32 reqId, OrbisNpScoreBoardId boardId,
                                       OrbisNpScoreValue score,
                                       const OrbisNpScoreComment* scoreComment,
                                       const OrbisNpScoreGameInfo* gameInfo,
                                       OrbisNpScoreRankNumber* tmpRank,
                                       const Rtc::OrbisRtcTick* compareDate, void* option) {
    LOG_INFO(Lib_NpScore,
             "(STUBBED) called reqId={}, boardId={}, score={}, scoreComment={}, "
             "gameInfo={}, tmpRank={}, compareDate={}, option={}",
             reqId, boardId, score, PTR(scoreComment), PTR(gameInfo), PTR(tmpRank),
             PTR(compareDate), PTR(option));
    if (tmpRank)
        *tmpRank = 1;
    return ORBIS_OK;
}

int PS4_SYSV_ABI sceNpScoreRecordScoreAsync(s32 reqId, OrbisNpScoreBoardId boardId,
                                            OrbisNpScoreValue score,
                                            const OrbisNpScoreComment* scoreComment,
                                            const OrbisNpScoreGameInfo* gameInfo,
                                            OrbisNpScoreRankNumber* tmpRank,
                                            const Rtc::OrbisRtcTick* compareDate, void* option) {
    LOG_INFO(Lib_NpScore,
             "(STUBBED) called reqId={}, boardId={}, score={}, "
             "scoreComment={}, gameInfo={}, tmpRank={}, compareDate={}, option={}",
             reqId, boardId, score, PTR(scoreComment), PTR(gameInfo), PTR(tmpRank),
             PTR(compareDate), PTR(option));
    if (tmpRank)
        *tmpRank = 1;
    return ORBIS_OK;
}

int PS4_SYSV_ABI sceNpScoreSanitizeComment(s32 reqId, const char* comment, char* sanitizedComment,
                                           void* option) {
    LOG_INFO(Lib_NpScore, "(STUBBED) called reqId={}, comment={}, sanitizedComment={}, option={}",
             reqId, comment ? comment : "null", PTR(sanitizedComment), PTR(option));
    // Pass comment through unchanged.
    if (sanitizedComment && comment)
        std::strncpy(sanitizedComment, comment, ORBIS_NP_SCORE_COMMENT_MAXLEN);
    return ORBIS_OK;
}

int PS4_SYSV_ABI sceNpScoreSanitizeCommentAsync(s32 reqId, const char* comment,
                                                char* sanitizedComment, void* option) {
    LOG_INFO(Lib_NpScore,
             "(STUBBED) called reqId={}, comment={}, sanitizedComment={}, "
             "option={}",
             reqId, comment ? comment : "null", PTR(sanitizedComment), PTR(option));
    if (sanitizedComment && comment)
        std::strncpy(sanitizedComment, comment, ORBIS_NP_SCORE_COMMENT_MAXLEN);
    return ORBIS_OK;
}

int PS4_SYSV_ABI sceNpScoreSetPlayerCharacterId(s32 ctxId, OrbisNpScorePcId pcId) {
    LOG_INFO(Lib_NpScore, "(STUBBED) called ctxId={}, pcId={}", ctxId, pcId);
    return ORBIS_OK;
}

int PS4_SYSV_ABI sceNpScoreSetThreadParam(s32 threadPriority, u64 cpuAffinityMask) {
    LOG_INFO(Lib_NpScore, "(STUBBED) called threadPriority={}, cpuAffinityMask={:#x}",
             threadPriority, cpuAffinityMask);
    return ORBIS_OK;
}

int PS4_SYSV_ABI sceNpScoreSetTimeout(s32 id, s32 resolveRetry, s32 resolveTimeout, s32 connTimeout,
                                      s32 sendTimeout, s32 recvTimeout) {
    LOG_INFO(Lib_NpScore,
             "(STUBBED) called id={}, resolveRetry={}, resolveTimeout={}, "
             "connTimeout={}, sendTimeout={}, recvTimeout={}",
             id, resolveRetry, resolveTimeout, connTimeout, sendTimeout, recvTimeout);
    return ORBIS_OK;
}

int PS4_SYSV_ABI sceNpScoreWaitAsync(s32 reqId, s32* result) {
    LOG_INFO(Lib_NpScore, "(STUBBED) sceNpScoreWaitAsync(reqId={}, result={})", reqId, PTR(result));
    if (result)
        *result = ORBIS_OK;
    return ORBIS_OK;
}

void RegisterLib(Core::Loader::SymbolsResolver* sym) {
    LIB_FUNCTION("1i7kmKbX6hk", "libSceNpScore", 1, "libSceNpScore", sceNpScoreAbortRequest);
    LIB_FUNCTION("2b3TI0mDYiI", "libSceNpScore", 1, "libSceNpScore", sceNpScoreCensorComment);
    LIB_FUNCTION("4eOvDyN-aZc", "libSceNpScore", 1, "libSceNpScore", sceNpScoreCensorCommentAsync);
    LIB_FUNCTION("dTXC+YcePtM", "libSceNpScore", 1, "libSceNpScore",
                 sceNpScoreChangeModeForOtherSaveDataOwners);
    LIB_FUNCTION("KnNA1TEgtBI", "libSceNpScore", 1, "libSceNpScore", sceNpScoreCreateNpTitleCtx);
    LIB_FUNCTION("GWnWQNXZH5M", "libSceNpScore", 1, "libSceNpScore", sceNpScoreCreateNpTitleCtxA);
    LIB_FUNCTION("gW8qyjYrUbk", "libSceNpScore", 1, "libSceNpScore", sceNpScoreCreateRequest);
    LIB_FUNCTION("qW9M0bQ-Zx0", "libSceNpScore", 1, "libSceNpScore", sceNpScoreCreateTitleCtx);
    LIB_FUNCTION("G0pE+RNCwfk", "libSceNpScore", 1, "libSceNpScore", sceNpScoreDeleteNpTitleCtx);
    LIB_FUNCTION("dK8-SgYf6r4", "libSceNpScore", 1, "libSceNpScore", sceNpScoreDeleteRequest);
    LIB_FUNCTION("LoVMVrijVOk", "libSceNpScore", 1, "libSceNpScore", sceNpScoreGetBoardInfo);
    LIB_FUNCTION("Q0Avi9kebsY", "libSceNpScore", 1, "libSceNpScore", sceNpScoreGetBoardInfoAsync);
    LIB_FUNCTION("8kuIzUw6utQ", "libSceNpScore", 1, "libSceNpScore", sceNpScoreGetFriendsRanking);
    LIB_FUNCTION("gMbOn+-6eXA", "libSceNpScore", 1, "libSceNpScore", sceNpScoreGetFriendsRankingA);
    LIB_FUNCTION("6-G9OxL5DKg", "libSceNpScore", 1, "libSceNpScore",
                 sceNpScoreGetFriendsRankingAAsync);
    LIB_FUNCTION("7SuMUlN7Q6I", "libSceNpScore", 1, "libSceNpScore",
                 sceNpScoreGetFriendsRankingAsync);
    LIB_FUNCTION("AgcxgceaH8k", "libSceNpScore", 1, "libSceNpScore",
                 sceNpScoreGetFriendsRankingForCrossSave);
    LIB_FUNCTION("m6F7sE1HQZU", "libSceNpScore", 1, "libSceNpScore",
                 sceNpScoreGetFriendsRankingForCrossSaveAsync);
    LIB_FUNCTION("zKoVok6FFEI", "libSceNpScore", 1, "libSceNpScore", sceNpScoreGetGameData);
    LIB_FUNCTION("JjOFRVPdQWc", "libSceNpScore", 1, "libSceNpScore", sceNpScoreGetGameDataAsync);
    LIB_FUNCTION("Lmtc9GljeUA", "libSceNpScore", 1, "libSceNpScore",
                 sceNpScoreGetGameDataByAccountId);
    LIB_FUNCTION("PP9jx8s0574", "libSceNpScore", 1, "libSceNpScore",
                 sceNpScoreGetGameDataByAccountIdAsync);
    LIB_FUNCTION("K9tlODTQx3c", "libSceNpScore", 1, "libSceNpScore",
                 sceNpScoreGetRankingByAccountId);
    LIB_FUNCTION("dRszNNyGWkw", "libSceNpScore", 1, "libSceNpScore",
                 sceNpScoreGetRankingByAccountIdAsync);
    LIB_FUNCTION("3Ybj4E1qNtY", "libSceNpScore", 1, "libSceNpScore",
                 sceNpScoreGetRankingByAccountIdForCrossSave);
    LIB_FUNCTION("Kc+3QK84AKM", "libSceNpScore", 1, "libSceNpScore",
                 sceNpScoreGetRankingByAccountIdForCrossSaveAsync);
    LIB_FUNCTION("wJPWycVGzrs", "libSceNpScore", 1, "libSceNpScore",
                 sceNpScoreGetRankingByAccountIdPcId);
    LIB_FUNCTION("bFVjDgxFapc", "libSceNpScore", 1, "libSceNpScore",
                 sceNpScoreGetRankingByAccountIdPcIdAsync);
    LIB_FUNCTION("oXjVieH6ZGQ", "libSceNpScore", 1, "libSceNpScore",
                 sceNpScoreGetRankingByAccountIdPcIdForCrossSave);
    LIB_FUNCTION("nXaF1Bxb-Nw", "libSceNpScore", 1, "libSceNpScore",
                 sceNpScoreGetRankingByAccountIdPcIdForCrossSaveAsync);
    LIB_FUNCTION("9mZEgoiEq6Y", "libSceNpScore", 1, "libSceNpScore", sceNpScoreGetRankingByNpId);
    LIB_FUNCTION("Rd27dqUFZV8", "libSceNpScore", 1, "libSceNpScore",
                 sceNpScoreGetRankingByNpIdAsync);
    LIB_FUNCTION("ETS-uM-vH9Q", "libSceNpScore", 1, "libSceNpScore",
                 sceNpScoreGetRankingByNpIdPcId);
    LIB_FUNCTION("FsouSN0ykN8", "libSceNpScore", 1, "libSceNpScore",
                 sceNpScoreGetRankingByNpIdPcIdAsync);
    LIB_FUNCTION("KBHxDjyk-jA", "libSceNpScore", 1, "libSceNpScore", sceNpScoreGetRankingByRange);
    LIB_FUNCTION("MA9vSt7JImY", "libSceNpScore", 1, "libSceNpScore", sceNpScoreGetRankingByRangeA);
    LIB_FUNCTION("y5ja7WI05rs", "libSceNpScore", 1, "libSceNpScore",
                 sceNpScoreGetRankingByRangeAAsync);
    LIB_FUNCTION("rShmqXHwoQE", "libSceNpScore", 1, "libSceNpScore",
                 sceNpScoreGetRankingByRangeAsync);
    LIB_FUNCTION("nRoYV2yeUuw", "libSceNpScore", 1, "libSceNpScore",
                 sceNpScoreGetRankingByRangeForCrossSave);
    LIB_FUNCTION("AZ4eAlGDy-Q", "libSceNpScore", 1, "libSceNpScore",
                 sceNpScoreGetRankingByRangeForCrossSaveAsync);
    LIB_FUNCTION("m1DfNRstkSQ", "libSceNpScore", 1, "libSceNpScore", sceNpScorePollAsync);
    LIB_FUNCTION("bcoVwcBjQ9E", "libSceNpScore", 1, "libSceNpScore", sceNpScoreRecordGameData);
    LIB_FUNCTION("1gL5PwYzrrw", "libSceNpScore", 1, "libSceNpScore", sceNpScoreRecordGameDataAsync);
    LIB_FUNCTION("zT0XBtgtOSI", "libSceNpScore", 1, "libSceNpScore", sceNpScoreRecordScore);
    LIB_FUNCTION("ANJssPz3mY0", "libSceNpScore", 1, "libSceNpScore", sceNpScoreRecordScoreAsync);
    LIB_FUNCTION("r4oAo9in0TA", "libSceNpScore", 1, "libSceNpScore", sceNpScoreSanitizeComment);
    LIB_FUNCTION("3UVqGJeDf30", "libSceNpScore", 1, "libSceNpScore",
                 sceNpScoreSanitizeCommentAsync);
    LIB_FUNCTION("bygbKdHmjn4", "libSceNpScore", 1, "libSceNpScore",
                 sceNpScoreSetPlayerCharacterId);
    LIB_FUNCTION("yxK68584JAU", "libSceNpScore", 1, "libSceNpScore", sceNpScoreSetThreadParam);
    LIB_FUNCTION("S3xZj35v8Z8", "libSceNpScore", 1, "libSceNpScore", sceNpScoreSetTimeout);
    LIB_FUNCTION("fqk8SC63p1U", "libSceNpScore", 1, "libSceNpScore", sceNpScoreWaitAsync);
    LIB_FUNCTION("KnNA1TEgtBI", "libSceNpScoreCompat", 1, "libSceNpScore",
                 sceNpScoreCreateNpTitleCtx);
    LIB_FUNCTION("8kuIzUw6utQ", "libSceNpScoreCompat", 1, "libSceNpScore",
                 sceNpScoreGetFriendsRanking);
    LIB_FUNCTION("7SuMUlN7Q6I", "libSceNpScoreCompat", 1, "libSceNpScore",
                 sceNpScoreGetFriendsRankingAsync);
    LIB_FUNCTION("zKoVok6FFEI", "libSceNpScoreCompat", 1, "libSceNpScore", sceNpScoreGetGameData);
    LIB_FUNCTION("JjOFRVPdQWc", "libSceNpScoreCompat", 1, "libSceNpScore",
                 sceNpScoreGetGameDataAsync);
    LIB_FUNCTION("9mZEgoiEq6Y", "libSceNpScoreCompat", 1, "libSceNpScore",
                 sceNpScoreGetRankingByNpId);
    LIB_FUNCTION("Rd27dqUFZV8", "libSceNpScoreCompat", 1, "libSceNpScore",
                 sceNpScoreGetRankingByNpIdAsync);
    LIB_FUNCTION("ETS-uM-vH9Q", "libSceNpScoreCompat", 1, "libSceNpScore",
                 sceNpScoreGetRankingByNpIdPcId);
    LIB_FUNCTION("FsouSN0ykN8", "libSceNpScoreCompat", 1, "libSceNpScore",
                 sceNpScoreGetRankingByNpIdPcIdAsync);
    LIB_FUNCTION("KBHxDjyk-jA", "libSceNpScoreCompat", 1, "libSceNpScore",
                 sceNpScoreGetRankingByRange);
    LIB_FUNCTION("rShmqXHwoQE", "libSceNpScoreCompat", 1, "libSceNpScore",
                 sceNpScoreGetRankingByRangeAsync);
};

} // namespace Libraries::Np::NpScore
