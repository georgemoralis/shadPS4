// SPDX-FileCopyrightText: Copyright 2025 shadPS4 Emulator Project
// SPDX-License-Identifier: GPL-2.0-or-later

#include <atomic>
#include <cstdio>
#include <cstring>
#include "common/logging/log.h"
#include "core/libraries/error_codes.h"
#include "core/libraries/libs.h"
#include "core/libraries/np/np_score.h"

namespace Libraries::Np::NpScore {

    // Add this after the includes
static OrbisNpScoreValue g_currentPlayerScore = 1000;

// Add this function to simulate score updates
static void UpdateCurrentPlayerScore(OrbisNpScoreValue score) {
    g_currentPlayerScore = score;
    LOG_INFO(Lib_NpScore, "Current player score updated to: {}", g_currentPlayerScore);
}

// Helper macro to format pointer safely
#define PTR(ptr) static_cast<const void*>(ptr)

// Monotonically increasing IDs for context and request handles.
static std::atomic<s32> s_ctxIdCounter{1};
static std::atomic<s32> s_reqIdCounter{1};

// Fake leaderboard constants
static constexpr u64 FAKE_ENTRY_COUNT = 10;
static constexpr OrbisNpScoreValue FAKE_SCORES[FAKE_ENTRY_COUNT] = {
    999LL, 100LL, 200LL, 500LL, 700LL,
    888LL, 777LL, 687LL,547LL, 687LL,
};
static constexpr const char* FAKE_ONLINE_IDS[FAKE_ENTRY_COUNT] = {
    "FakePlayer01", "FakePlayer02", "FakePlayer03", "FakePlayer04", "FakePlayer05",
    "FakePlayer06", "FakePlayer07", "FakePlayer08", "FakePlayer09", "FakePlayer10",
};
static constexpr u64 FAKE_RTC_TICK = 0x00038D7E'A4C68000ULL;
static constexpr OrbisNpAccountId FAKE_ACCOUNT_IDS[FAKE_ENTRY_COUNT] = {
    0x0000'0001'0000'0001ULL, 0x0000'0001'0000'0002ULL, 0x0000'0001'0000'0003ULL,
    0x0000'0001'0000'0004ULL, 0x0000'0001'0000'0005ULL, 0x0000'0001'0000'0006ULL,
    0x0000'0001'0000'0007ULL, 0x0000'0001'0000'0008ULL, 0x0000'0001'0000'0009ULL,
    0x0000'0001'0000'0010ULL,
};

// ---------------------------------------------------------------------------
// Debug helpers
// ---------------------------------------------------------------------------

static void DumpNpId(const OrbisNpId& npId, const char* label) {
    const char* onlineId = reinterpret_cast<const char*>(&npId);
    LOG_INFO(Lib_NpScore, "{} NpId: {} (first 16 chars: {})", label, PTR(&npId), onlineId);
}

static void DumpRankData(const OrbisNpScoreRankData& data, int index) {
    const char* onlineId = reinterpret_cast<const char*>(&data.npId);
    LOG_INFO(Lib_NpScore, "RankData[{}]: rank={}, score={}, pcId={}, onlineId={}", index, data.rank,
             data.scoreValue, data.pcId, onlineId);
}

static void DumpRankDataA(const OrbisNpScoreRankDataA& data, int index) {
    LOG_INFO(Lib_NpScore, "RankDataA[{}]: rank={}, score={}, pcId={}, onlineId={}, accountId={:#x}",
             index, data.rank, data.scoreValue, data.pcId, data.onlineId.data, data.accountId);
}

static void DumpGameInfo(const OrbisNpScoreGameInfo& info, int index) {
    LOG_INFO(Lib_NpScore, "GameInfo[{}]: infoSize={}, first few bytes: {:02x} {:02x} {:02x} {:02x}",
             index, info.infoSize, info.data[0], info.data[1], info.data[2], info.data[3]);
}

// ---------------------------------------------------------------------------
// Internal helpers
// ---------------------------------------------------------------------------

static constexpr u64 NP_ONLINE_ID_DATA_OFFSET = 0;
static constexpr u64 NP_ONLINE_ID_MAX_LEN = 16;

static void WriteNpIdName(OrbisNpId& npId, const char* name) {
    char* dst = reinterpret_cast<char*>(&npId) + NP_ONLINE_ID_DATA_OFFSET;
    std::strncpy(dst, name, NP_ONLINE_ID_MAX_LEN - 1);
    dst[NP_ONLINE_ID_MAX_LEN - 1] = '\0';
}

static OrbisNpScoreValue GetFakeScoreForBoard(OrbisNpScoreBoardId boardId, u64 rank) {
    // Make scores more varied and game-like
    switch (boardId) {
    case 0:
        // Board 0: High scores (1M to 10M)
        return 10000000 - (rank * 100000);
    case 1:
        // Board 1: Medium scores (500K to 5M)
        return 5000000 - (rank * 50000);
    default:
        return 1000000 - (rank * 10000);
    }
}

static void FillGameInfoData(OrbisNpScoreGameInfo* infoArray, u64 infoArraySize, u64 arrayNum) {
    if (!infoArray || infoArraySize == 0 || arrayNum == 0)
        return;

    const u64 fillCount = (arrayNum < infoArraySize) ? arrayNum : infoArraySize;
    for (u64 i = 0; i < fillCount; ++i) {
        OrbisNpScoreGameInfo& info = infoArray[i];
        // Put some dummy game data to show there's game-specific info
        info.infoSize = 8; // Small amount of game data
        // Fill with pattern data
        for (int j = 0; j < 8 && j < ORBIS_NP_SCORE_GAMEINFO_MAXSIZE; ++j) {
            info.data[j] = static_cast<u8>((i + j) & 0xFF);
        }
        // Zero out the rest
        for (int j = 8; j < ORBIS_NP_SCORE_GAMEINFO_MAXSIZE; ++j) {
            info.data[j] = 0;
        }
        DumpGameInfo(info, i);
    }
}

static void FillCommentData(OrbisNpScoreComment* commentArray, u64 commentArraySize, u64 arrayNum,
                            OrbisNpScoreBoardId boardId = 0) {
    if (!commentArray || commentArraySize == 0 || arrayNum == 0)
        return;

    const u64 fillCount = (arrayNum < commentArraySize) ? arrayNum : commentArraySize;
    const char* comments[] = {"First place! Amazing score!",
                              "Great run, almost first!",
                              "Top 3 performance!",
                              "Solid score!",
                              "Good job!",
                              "Nice try!",
                              "Keep practicing!",
                              "You can do better!",
                              "Almost there!",
                              "New player!"};

    for (u64 i = 0; i < fillCount; ++i) {
        std::strncpy(commentArray[i].utf8Comment, comments[i % 10], ORBIS_NP_SCORE_COMMENT_MAXLEN);
        commentArray[i].utf8Comment[ORBIS_NP_SCORE_COMMENT_MAXLEN] = '\0';
    }
}

static void FillRankData(OrbisNpScoreRankData* rankArray, u64 arrayNum,
                         Rtc::OrbisRtcTick* lastSortDate, OrbisNpScoreRankNumber* totalRecord,
                         const OrbisNpId* npIdArray = nullptr, OrbisNpScoreBoardId boardId = 0) {
    LOG_INFO(Lib_NpScore, "FillRankData: arrayNum={}, totalRecord ptr={}", arrayNum,
             PTR(totalRecord));

    if (!rankArray || arrayNum == 0) {
        if (totalRecord) {
            *totalRecord = 0;
            LOG_INFO(Lib_NpScore, "FillRankData: setting totalRecord to 0 (no data)");
        }
        return;
    }

    const u64 fillCount = (arrayNum < FAKE_ENTRY_COUNT) ? arrayNum : FAKE_ENTRY_COUNT;
    LOG_INFO(Lib_NpScore, "FillRankData: fillCount={}", fillCount);

    for (u64 i = 0; i < fillCount; ++i) {
        OrbisNpScoreRankData& e = rankArray[i];
        std::memset(&e, 0, sizeof(e));

        if (npIdArray) {
            e.npId = npIdArray[i];
        } else {
            char onlineId[17];
            snprintf(onlineId, sizeof(onlineId), "Player%03llu", i + 1);
            WriteNpIdName(e.npId, onlineId);
        }

        // Fill all fields with valid data
        e.pcId = static_cast<OrbisNpScorePcId>(i + 1000); // Use 1000+ for PC IDs
        e.serialRank = static_cast<OrbisNpScoreRankNumber>(i + 1);
        e.rank = static_cast<OrbisNpScoreRankNumber>(i + 1);
        e.highestRank = static_cast<OrbisNpScoreRankNumber>(i + 1);
        e.scoreValue = GetFakeScoreForBoard(boardId, i);
        e.hasGameData = 1; // Set to 1 to indicate there's game data
        e.recordDate.tick = FAKE_RTC_TICK;

        // Fill reserved fields with non-zero values to satisfy any validation
        std::memset(e.reserved, 0xFF, sizeof(e.reserved));

        DumpRankData(e, i);
    }

    if (totalRecord) {
        *totalRecord = static_cast<OrbisNpScoreRankNumber>(fillCount);
        LOG_INFO(Lib_NpScore, "FillRankData: setting totalRecord to {}", fillCount);
    }

    if (lastSortDate) {
        lastSortDate->tick = FAKE_RTC_TICK;
    }
}

static void FillRankDataA(OrbisNpScoreRankDataA* rankArray, u64 arrayNum,
                          Rtc::OrbisRtcTick* lastSortDate, OrbisNpScoreRankNumber* totalRecord,
                          OrbisNpScoreBoardId boardId = 0) {
    LOG_INFO(Lib_NpScore, "FillRankDataA: arrayNum={}, totalRecord ptr={}", arrayNum,
             PTR(totalRecord));

    if (!rankArray || arrayNum == 0) {
        if (totalRecord) {
            *totalRecord = 0;
            LOG_INFO(Lib_NpScore, "FillRankDataA: setting totalRecord to 0 (no data)");
        }
        return;
    }

    const u64 fillCount = (arrayNum < FAKE_ENTRY_COUNT) ? arrayNum : FAKE_ENTRY_COUNT;
    LOG_INFO(Lib_NpScore, "FillRankDataA: fillCount={}", fillCount);

    for (u64 i = 0; i < fillCount; ++i) {
        OrbisNpScoreRankDataA& e = rankArray[i];
        std::memset(&e, 0, sizeof(e));
        snprintf(e.onlineId.data, sizeof(e.onlineId.data), "Player%03llu", i + 1);
        e.onlineId.data[sizeof(e.onlineId.data) - 1] = '\0';
        e.pcId = static_cast<OrbisNpScorePcId>(i + 1000);
        e.serialRank = static_cast<OrbisNpScoreRankNumber>(i + 1);
        e.rank = static_cast<OrbisNpScoreRankNumber>(i + 1);
        e.highestRank = static_cast<OrbisNpScoreRankNumber>(i + 1);
        e.scoreValue = GetFakeScoreForBoard(boardId, i);
        e.hasGameData = 1;
        e.recordDate.tick = FAKE_RTC_TICK;
        e.accountId = FAKE_ACCOUNT_IDS[i];

        // Fill reserved fields with non-zero values
        std::memset(e.reserved0, 0xFF, sizeof(e.reserved0));
        std::memset(e.reserved, 0xFF, sizeof(e.reserved));

        DumpRankDataA(e, i);
    }

    if (totalRecord) {
        *totalRecord = static_cast<OrbisNpScoreRankNumber>(fillCount);
        LOG_INFO(Lib_NpScore, "FillRankDataA: setting totalRecord to {}", fillCount);
    }

    if (lastSortDate) {
        lastSortDate->tick = FAKE_RTC_TICK;
    }
}

static void FillPlayerRankDataA(OrbisNpScorePlayerRankDataA* rankArray, u64 arrayNum,
                                Rtc::OrbisRtcTick* lastSortDate,
                                OrbisNpScoreRankNumber* totalRecord,
                                OrbisNpScoreBoardId boardId = 0) {
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
        snprintf(e.onlineId.data, sizeof(e.onlineId.data), "Player%03llu", i + 1);
        e.onlineId.data[sizeof(e.onlineId.data) - 1] = '\0';
        e.pcId = static_cast<OrbisNpScorePcId>(i + 1000);
        e.serialRank = static_cast<OrbisNpScoreRankNumber>(i + 1);
        e.rank = static_cast<OrbisNpScoreRankNumber>(i + 1);
        e.highestRank = static_cast<OrbisNpScoreRankNumber>(i + 1);
        e.scoreValue = GetFakeScoreForBoard(boardId, i);
        e.hasGameData = 1;
        e.recordDate.tick = FAKE_RTC_TICK;
        e.accountId = FAKE_ACCOUNT_IDS[i];

        std::memset(e.reserved0, 0xFF, sizeof(e.reserved0));
        std::memset(e.reserved, 0xFF, sizeof(e.reserved));
    }

    if (totalRecord)
        *totalRecord = static_cast<OrbisNpScoreRankNumber>(fillCount);
    if (lastSortDate)
        lastSortDate->tick = FAKE_RTC_TICK;
}

static void FillRankDataForCrossSave(OrbisNpScoreRankDataForCrossSave* rankArray, u64 arrayNum,
                                     Rtc::OrbisRtcTick* lastSortDate,
                                     OrbisNpScoreRankNumber* totalRecord,
                                     OrbisNpScoreBoardId boardId = 0) {
    if (!rankArray || arrayNum == 0) {
        if (totalRecord)
            *totalRecord = 0;
        return;
    }

    const u64 fillCount = (arrayNum < FAKE_ENTRY_COUNT) ? arrayNum : FAKE_ENTRY_COUNT;
    for (u64 i = 0; i < fillCount; ++i) {
        OrbisNpScoreRankDataForCrossSave& e = rankArray[i];
        std::memset(&e, 0, sizeof(e));
        e.pcId = static_cast<OrbisNpScorePcId>(i + 1000);
        e.serialRank = static_cast<OrbisNpScoreRankNumber>(i + 1);
        e.rank = static_cast<OrbisNpScoreRankNumber>(i + 1);
        e.highestRank = static_cast<OrbisNpScoreRankNumber>(i + 1);
        e.scoreValue = GetFakeScoreForBoard(boardId, i);
        e.hasGameData = 1;
        e.recordDate.tick = FAKE_RTC_TICK;
        e.accountId = FAKE_ACCOUNT_IDS[i];

        std::memset(e.reserved, 0xFF, sizeof(e.reserved));
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
    LOG_INFO(Lib_NpScore, "called reqId={}, boardId={}, boardInfo={}, option={}", reqId, boardId,
             PTR(boardInfo), PTR(option));
    if (boardInfo) {
        std::memset(boardInfo, 0, sizeof(*boardInfo));
        boardInfo->rankLimit = 1000;
        boardInfo->updateMode = 1;
        boardInfo->sortMode = 1;
        boardInfo->uploadNumLimit = 10;
        boardInfo->uploadSizeLimit = 1024 * 1024;
        LOG_INFO(Lib_NpScore,
                 "BoardInfo: rankLimit={}, updateMode={}, sortMode={}, uploadNumLimit={}",
                 boardInfo->rankLimit, boardInfo->updateMode, boardInfo->sortMode,
                 boardInfo->uploadNumLimit);
    }
    return ORBIS_OK;
}

int PS4_SYSV_ABI sceNpScoreGetBoardInfoAsync(s32 reqId, OrbisNpScoreBoardId boardId,
                                             OrbisNpScoreBoardInfo* boardInfo, void* option) {
    LOG_INFO(Lib_NpScore, "called reqId={}, boardId={}, boardInfo={}, option={}", reqId, boardId,
             PTR(boardInfo), PTR(option));
    if (boardInfo) {
        std::memset(boardInfo, 0, sizeof(*boardInfo));
        boardInfo->rankLimit = 1000;
        boardInfo->updateMode = 1;
        boardInfo->sortMode = 1;
        boardInfo->uploadNumLimit = 10;
        boardInfo->uploadSizeLimit = 1024 * 1024;
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
             "(FAKE) called reqId={}, boardId={}, includeSelf={}, arrayNum={}, rankArraySize={}",
             reqId, boardId, includeSelf, arrayNum, rankArraySize);

    u64 fillCount = std::min(arrayNum, rankArraySize);
    fillCount = std::min(fillCount, FAKE_ENTRY_COUNT);

    LOG_INFO(Lib_NpScore, "Will fill {} entries", fillCount);

    // Fill the rank data
    if (rankArray && fillCount > 0) {
        for (u64 i = 0; i < fillCount; ++i) {
            OrbisNpScoreRankData& e = rankArray[i];
            std::memset(&e, 0, sizeof(e));

            char onlineId[17];
            snprintf(onlineId, sizeof(onlineId), "Player%03llu", i + 1);
            WriteNpIdName(e.npId, onlineId);

            e.pcId = static_cast<OrbisNpScorePcId>(i + 1000);
            e.serialRank = static_cast<OrbisNpScoreRankNumber>(i + 1);
            e.rank = static_cast<OrbisNpScoreRankNumber>(i + 1);
            e.highestRank = static_cast<OrbisNpScoreRankNumber>(i + 1);
            e.scoreValue = GetFakeScoreForBoard(boardId, i);
            e.hasGameData = 1;
            e.recordDate.tick = FAKE_RTC_TICK;

            std::memset(e.reserved, 0xFF, sizeof(e.reserved));

            DumpRankData(e, i);
        }
    }

    FillCommentData(commentArray, commentArraySize, fillCount, boardId);
    FillGameInfoData(infoArray, infoArraySize, fillCount);

    if (lastSortDate) {
        lastSortDate->tick = FAKE_RTC_TICK;
    }

    if (totalRecord)
        *totalRecord = static_cast<OrbisNpScoreRankNumber>(fillCount);

    LOG_INFO(Lib_NpScore, "Returning ORBIS_OK with totalRecord={}", fillCount);
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
             "(FAKE) called reqId={}, boardId={}, includeSelf={}, arrayNum={}, rankArraySize={}",
             reqId, boardId, includeSelf, arrayNum, rankArraySize);

    u64 fillCount = std::min(arrayNum, rankArraySize);
    fillCount = std::min(fillCount, FAKE_ENTRY_COUNT);

    FillRankDataA(rankArray, fillCount, lastSortDate, totalRecord, boardId);
    FillCommentData(commentArray, commentArraySize, fillCount, boardId);
    FillGameInfoData(infoArray, infoArraySize, fillCount);
    return ORBIS_OK;
}

int PS4_SYSV_ABI sceNpScoreGetFriendsRankingAAsync(
    s32 reqId, OrbisNpScoreBoardId boardId, s32 includeSelf, OrbisNpScoreRankDataA* rankArray,
    u64 rankArraySize, OrbisNpScoreComment* commentArray, u64 commentArraySize,
    OrbisNpScoreGameInfo* infoArray, u64 infoArraySize, u64 arrayNum,
    Rtc::OrbisRtcTick* lastSortDate, OrbisNpScoreRankNumber* totalRecord,
    OrbisNpScoreGetFriendRankingOptParam* option) {
    LOG_INFO(Lib_NpScore, "(FAKE) called reqId={}, boardId={}, includeSelf={}, arrayNum={}", reqId,
             boardId, includeSelf, arrayNum);

    u64 fillCount = std::min(arrayNum, rankArraySize);
    fillCount = std::min(fillCount, FAKE_ENTRY_COUNT);

    FillRankDataA(rankArray, fillCount, lastSortDate, totalRecord, boardId);
    FillCommentData(commentArray, commentArraySize, fillCount, boardId);
    FillGameInfoData(infoArray, infoArraySize, fillCount);
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
    LOG_INFO(Lib_NpScore, "(FAKE) called reqId={}, boardId={}, includeSelf={}, arrayNum={}", reqId,
             boardId, includeSelf, arrayNum);

    u64 fillCount = std::min(arrayNum, rankArraySize);
    fillCount = std::min(fillCount, FAKE_ENTRY_COUNT);

    FillRankDataForCrossSave(rankArray, fillCount, lastSortDate, totalRecord, boardId);
    FillCommentData(commentArray, commentArraySize, fillCount, boardId);
    FillGameInfoData(infoArray, infoArraySize, fillCount);
    return ORBIS_OK;
}

int PS4_SYSV_ABI sceNpScoreGetFriendsRankingForCrossSaveAsync(
    s32 reqId, OrbisNpScoreBoardId boardId, s32 includeSelf,
    OrbisNpScoreRankDataForCrossSave* rankArray, u64 rankArraySize,
    OrbisNpScoreComment* commentArray, u64 commentArraySize, OrbisNpScoreGameInfo* infoArray,
    u64 infoArraySize, u64 arrayNum, Rtc::OrbisRtcTick* lastSortDate,
    OrbisNpScoreRankNumber* totalRecord, OrbisNpScoreGetFriendRankingOptParam* option) {
    LOG_INFO(Lib_NpScore, "(FAKE) called reqId={}, boardId={}, includeSelf={}, arrayNum={}", reqId,
             boardId, includeSelf, arrayNum);

    u64 fillCount = std::min(arrayNum, rankArraySize);
    fillCount = std::min(fillCount, FAKE_ENTRY_COUNT);

    FillRankDataForCrossSave(rankArray, fillCount, lastSortDate, totalRecord, boardId);
    FillCommentData(commentArray, commentArraySize, fillCount, boardId);
    FillGameInfoData(infoArray, infoArraySize, fillCount);
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
    LOG_INFO(Lib_NpScore, "(FAKE) called reqId={}, boardId={}, accountIdArraySize={}, arrayNum={}",
             reqId, boardId, accountIdArraySize, arrayNum);

    u64 fillCount = std::min(arrayNum, rankArraySize);
    fillCount = std::min(fillCount, accountIdArraySize);
    fillCount = std::min(fillCount, FAKE_ENTRY_COUNT);

    FillPlayerRankDataA(rankArray, fillCount, lastSortDate, totalRecord, boardId);
    FillCommentData(commentArray, commentArraySize, fillCount, boardId);
    FillGameInfoData(infoArray, infoArraySize, fillCount);
    return ORBIS_OK;
}

int PS4_SYSV_ABI sceNpScoreGetRankingByAccountIdAsync(
    s32 reqId, OrbisNpScoreBoardId boardId, const OrbisNpAccountId* accountIdArray,
    u64 accountIdArraySize, OrbisNpScorePlayerRankDataA* rankArray, u64 rankArraySize,
    OrbisNpScoreComment* commentArray, u64 commentArraySize, OrbisNpScoreGameInfo* infoArray,
    u64 infoArraySize, u64 arrayNum, Rtc::OrbisRtcTick* lastSortDate,
    OrbisNpScoreRankNumber* totalRecord, void* option) {
    LOG_INFO(Lib_NpScore, "(FAKE) called reqId={}, boardId={}, accountIdArraySize={}, arrayNum={}",
             reqId, boardId, accountIdArraySize, arrayNum);

    u64 fillCount = std::min(arrayNum, rankArraySize);
    fillCount = std::min(fillCount, accountIdArraySize);
    fillCount = std::min(fillCount, FAKE_ENTRY_COUNT);

    FillPlayerRankDataA(rankArray, fillCount, lastSortDate, totalRecord, boardId);
    FillCommentData(commentArray, commentArraySize, fillCount, boardId);
    FillGameInfoData(infoArray, infoArraySize, fillCount);
    return ORBIS_OK;
}

int PS4_SYSV_ABI sceNpScoreGetRankingByAccountIdForCrossSave(
    s32 reqId, OrbisNpScoreBoardId boardId, const OrbisNpAccountId* accountIdArray,
    u64 accountIdArraySize, OrbisNpScorePlayerRankDataForCrossSave* rankArray, u64 rankArraySize,
    OrbisNpScoreComment* commentArray, u64 commentArraySize, OrbisNpScoreGameInfo* infoArray,
    u64 infoArraySize, u64 arrayNum, Rtc::OrbisRtcTick* lastSortDate,
    OrbisNpScoreRankNumber* totalRecord, void* option) {
    LOG_INFO(Lib_NpScore,
             "(STUBBED) called reqId={}, boardId={}, accountIdArraySize={}, arrayNum={}", reqId,
             boardId, accountIdArraySize, arrayNum);
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
             "(STUBBED) called reqId={}, boardId={}, accountIdArraySize={}, arrayNum={}", reqId,
             boardId, accountIdArraySize, arrayNum);
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
    LOG_INFO(Lib_NpScore, "(FAKE) called reqId={}, boardId={}, idArraySize={}, arrayNum={}", reqId,
             boardId, idArraySize, arrayNum);

    u64 fillCount = std::min(arrayNum, rankArraySize);
    fillCount = std::min(fillCount, idArraySize);
    fillCount = std::min(fillCount, FAKE_ENTRY_COUNT);

    FillPlayerRankDataA(rankArray, fillCount, lastSortDate, totalRecord, boardId);
    FillCommentData(commentArray, commentArraySize, fillCount, boardId);
    FillGameInfoData(infoArray, infoArraySize, fillCount);
    return ORBIS_OK;
}

int PS4_SYSV_ABI sceNpScoreGetRankingByAccountIdPcIdAsync(
    s32 reqId, OrbisNpScoreBoardId boardId, const OrbisNpScoreAccountIdPcId* idArray,
    u64 idArraySize, OrbisNpScorePlayerRankDataA* rankArray, u64 rankArraySize,
    OrbisNpScoreComment* commentArray, u64 commentArraySize, OrbisNpScoreGameInfo* infoArray,
    u64 infoArraySize, u64 arrayNum, Rtc::OrbisRtcTick* lastSortDate,
    OrbisNpScoreRankNumber* totalRecord, void* option) {
    LOG_INFO(Lib_NpScore, "(FAKE) called reqId={}, boardId={}, idArraySize={}, arrayNum={}", reqId,
             boardId, idArraySize, arrayNum);

    u64 fillCount = std::min(arrayNum, rankArraySize);
    fillCount = std::min(fillCount, idArraySize);
    fillCount = std::min(fillCount, FAKE_ENTRY_COUNT);

    FillPlayerRankDataA(rankArray, fillCount, lastSortDate, totalRecord, boardId);
    FillCommentData(commentArray, commentArraySize, fillCount, boardId);
    FillGameInfoData(infoArray, infoArraySize, fillCount);
    return ORBIS_OK;
}

int PS4_SYSV_ABI sceNpScoreGetRankingByAccountIdPcIdForCrossSave(
    s32 reqId, OrbisNpScoreBoardId boardId, const OrbisNpScoreAccountIdPcId* idArray,
    u64 idArraySize, OrbisNpScorePlayerRankDataForCrossSave* rankArray, u64 rankArraySize,
    OrbisNpScoreComment* commentArray, u64 commentArraySize, OrbisNpScoreGameInfo* infoArray,
    u64 infoArraySize, u64 arrayNum, Rtc::OrbisRtcTick* lastSortDate,
    OrbisNpScoreRankNumber* totalRecord, void* option) {
    LOG_INFO(Lib_NpScore, "(STUBBED) called reqId={}, boardId={}, idArraySize={}, arrayNum={}",
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
    LOG_INFO(Lib_NpScore, "(STUBBED) called reqId={}, boardId={}, idArraySize={}, arrayNum={}",
             reqId, boardId, idArraySize, arrayNum);
    if (totalRecord)
        *totalRecord = 0;
    return ORBIS_OK;
}

int PS4_SYSV_ABI sceNpScoreGetRankingByNpId(s32 reqId, OrbisNpScoreBoardId boardId,
                                            const OrbisNpId* npIdArray, u64 npIdArraySize,
                                            OrbisNpScoreRankData* rankArray, u64 rankArraySize,
                                            OrbisNpScoreComment* commentArray, u64 commentArraySize,
                                            OrbisNpScoreGameInfo* infoArray, u64 infoArraySize,
                                            u64 arrayNum, Rtc::OrbisRtcTick* lastSortDate,
                                            OrbisNpScoreRankNumber* totalRecord, void* option) {
    LOG_INFO(Lib_NpScore,
             "(FAKE) called reqId={}, boardId={}, npIdArraySize={}, arrayNum={}, rankArraySize={}",
             reqId, boardId, npIdArraySize, arrayNum, rankArraySize);

    if (npIdArray && npIdArraySize > 0) {
        for (u64 i = 0; i < std::min(npIdArraySize, (u64)5); ++i) {
            DumpNpId(npIdArray[i], "Requested");
        }
    }

    u64 fillCount = std::min(arrayNum, rankArraySize);
    fillCount = std::min(fillCount, npIdArraySize);
    fillCount = std::min(fillCount, FAKE_ENTRY_COUNT);

    LOG_INFO(Lib_NpScore, "Will fill {} entries", fillCount);

    if (rankArray && fillCount > 0) {
        for (u64 i = 0; i < fillCount; ++i) {
            OrbisNpScoreRankData& e = rankArray[i];
            std::memset(&e, 0, sizeof(e));

            // Mirror the requested NpId
            e.npId = npIdArray[i];

            // Check if this is the current player (shadPS4)
            const char* requestedId = reinterpret_cast<const char*>(&npIdArray[i]);
            bool isCurrentPlayer = (std::strncmp(requestedId, "shadPS4", 7) == 0);

            if (isCurrentPlayer && g_currentPlayerScore > 0) {
                // Return the player's actual score if we have one
                e.scoreValue = g_currentPlayerScore;
                e.rank = 1; // Assume they're rank 1 for now
                e.serialRank = 1;
                e.highestRank = 1;
                LOG_INFO(Lib_NpScore, "Returning current player score: {}", g_currentPlayerScore);
            } else {
                // Return a fake score
                e.scoreValue = GetFakeScoreForBoard(boardId, i);
                e.rank = static_cast<OrbisNpScoreRankNumber>(i + 1);
                e.serialRank = static_cast<OrbisNpScoreRankNumber>(i + 1);
                e.highestRank = static_cast<OrbisNpScoreRankNumber>(i + 1);
            }

            e.pcId = static_cast<OrbisNpScorePcId>(i + 1);
            e.hasGameData = 1;
            e.recordDate.tick = FAKE_RTC_TICK;

            // Fill reserved fields
            for (size_t j = 0; j < sizeof(e.reserved); ++j) {
                e.reserved[j] = static_cast<u8>(j & 0xFF);
            }

            DumpRankData(e, i);
        }
    }

    FillCommentData(commentArray, commentArraySize, fillCount, boardId);
    FillGameInfoData(infoArray, infoArraySize, fillCount);

    if (lastSortDate) {
        lastSortDate->tick = FAKE_RTC_TICK;
    }

    if (totalRecord)
        *totalRecord = static_cast<OrbisNpScoreRankNumber>(fillCount);

    LOG_INFO(Lib_NpScore, "Returning ORBIS_OK with totalRecord={}", fillCount);
    return ORBIS_OK;
}
int PS4_SYSV_ABI sceNpScoreGetRankingByNpIdAsync(
    s32 reqId, OrbisNpScoreBoardId boardId, const OrbisNpId* npIdArray, u64 npIdArraySize,
    OrbisNpScoreRankData* rankArray, u64 rankArraySize, OrbisNpScoreComment* commentArray,
    u64 commentArraySize, OrbisNpScoreGameInfo* infoArray, u64 infoArraySize, u64 arrayNum,
    Rtc::OrbisRtcTick* lastSortDate, OrbisNpScoreRankNumber* totalRecord, void* option) {
    LOG_INFO(Lib_NpScore, "(FAKE) called reqId={}, boardId={}, npIdArraySize={}, arrayNum={}",
             reqId, boardId, npIdArraySize, arrayNum);

    u64 fillCount = std::min(arrayNum, rankArraySize);
    fillCount = std::min(fillCount, npIdArraySize);
    fillCount = std::min(fillCount, FAKE_ENTRY_COUNT);

    FillRankData(rankArray, fillCount, lastSortDate, totalRecord, npIdArray, boardId);
    FillCommentData(commentArray, commentArraySize, fillCount, boardId);
    FillGameInfoData(infoArray, infoArraySize, fillCount);
    return ORBIS_OK;
}

int PS4_SYSV_ABI sceNpScoreGetRankingByNpIdPcId(
    s32 reqId, OrbisNpScoreBoardId boardId, const OrbisNpScoreNpIdPcId* idArray, u64 idArraySize,
    OrbisNpScoreRankData* rankArray, u64 rankArraySize, OrbisNpScoreComment* commentArray,
    u64 commentArraySize, OrbisNpScoreGameInfo* infoArray, u64 infoArraySize, u64 arrayNum,
    Rtc::OrbisRtcTick* lastSortDate, OrbisNpScoreRankNumber* totalRecord, void* option) {
    LOG_INFO(Lib_NpScore, "(FAKE) called reqId={}, boardId={}, idArraySize={}, arrayNum={}", reqId,
             boardId, idArraySize, arrayNum);

    u64 fillCount = std::min(arrayNum, rankArraySize);
    fillCount = std::min(fillCount, idArraySize);
    fillCount = std::min(fillCount, FAKE_ENTRY_COUNT);

    FillRankData(rankArray, fillCount, lastSortDate, totalRecord, nullptr, boardId);
    FillCommentData(commentArray, commentArraySize, fillCount, boardId);
    FillGameInfoData(infoArray, infoArraySize, fillCount);
    return ORBIS_OK;
}

int PS4_SYSV_ABI sceNpScoreGetRankingByNpIdPcIdAsync(
    s32 reqId, OrbisNpScoreBoardId boardId, const OrbisNpScoreNpIdPcId* idArray, u64 idArraySize,
    OrbisNpScoreRankData* rankArray, u64 rankArraySize, OrbisNpScoreComment* commentArray,
    u64 commentArraySize, OrbisNpScoreGameInfo* infoArray, u64 infoArraySize, u64 arrayNum,
    Rtc::OrbisRtcTick* lastSortDate, OrbisNpScoreRankNumber* totalRecord, void* option) {
    LOG_INFO(Lib_NpScore, "(FAKE) called reqId={}, boardId={}, idArraySize={}, arrayNum={}", reqId,
             boardId, idArraySize, arrayNum);

    u64 fillCount = std::min(arrayNum, rankArraySize);
    fillCount = std::min(fillCount, idArraySize);
    fillCount = std::min(fillCount, FAKE_ENTRY_COUNT);

    FillRankData(rankArray, fillCount, lastSortDate, totalRecord, nullptr, boardId);
    FillCommentData(commentArray, commentArraySize, fillCount, boardId);
    FillGameInfoData(infoArray, infoArraySize, fillCount);
    return ORBIS_OK;
}

int PS4_SYSV_ABI sceNpScoreGetRankingByRange(
    s32 reqId, OrbisNpScoreBoardId boardId, OrbisNpScoreRankNumber startSerialRank,
    OrbisNpScoreRankData* rankArray, u64 rankArraySize, OrbisNpScoreComment* commentArray,
    u64 commentArraySize, OrbisNpScoreGameInfo* infoArray, u64 infoArraySize, u64 arrayNum,
    Rtc::OrbisRtcTick* lastSortDate, OrbisNpScoreRankNumber* totalRecord, void* option) {
    LOG_INFO(
        Lib_NpScore,
        "(FAKE) called reqId={}, boardId={}, startSerialRank={}, arrayNum={}, rankArraySize={}",
        reqId, boardId, startSerialRank, arrayNum, rankArraySize);

    u64 fillCount = std::min(arrayNum, rankArraySize);
    fillCount = std::min(fillCount, FAKE_ENTRY_COUNT);

    LOG_INFO(Lib_NpScore, "Will fill {} entries", fillCount);

    if (rankArray && fillCount > 0) {
        for (u64 i = 0; i < fillCount; ++i) {
            OrbisNpScoreRankData& e = rankArray[i];
            std::memset(&e, 0, sizeof(e));

            char onlineId[17];
            snprintf(onlineId, sizeof(onlineId), "Rank%03llu", startSerialRank + i);
            WriteNpIdName(e.npId, onlineId);

            e.pcId = static_cast<OrbisNpScorePcId>(startSerialRank + i + 1000);
            e.serialRank = static_cast<OrbisNpScoreRankNumber>(startSerialRank + i);
            e.rank = static_cast<OrbisNpScoreRankNumber>(startSerialRank + i);
            e.highestRank = static_cast<OrbisNpScoreRankNumber>(startSerialRank + i);
            e.scoreValue =
                GetFakeScoreForBoard(boardId, (startSerialRank + i - 1) % FAKE_ENTRY_COUNT);
            e.hasGameData = 1;
            e.recordDate.tick = FAKE_RTC_TICK;

            std::memset(e.reserved, 0xFF, sizeof(e.reserved));

            DumpRankData(e, i);
        }
    }

    if (lastSortDate) {
        lastSortDate->tick = FAKE_RTC_TICK;
    }

    if (totalRecord)
        *totalRecord = static_cast<OrbisNpScoreRankNumber>(fillCount);

    FillCommentData(commentArray, commentArraySize, fillCount, boardId);
    FillGameInfoData(infoArray, infoArraySize, fillCount);

    LOG_INFO(Lib_NpScore, "Returning ORBIS_OK with totalRecord={}", fillCount);
    return ORBIS_OK;
}

int PS4_SYSV_ABI sceNpScoreGetRankingByRangeA(
    s32 reqId, OrbisNpScoreBoardId boardId, OrbisNpScoreRankNumber startSerialRank,
    OrbisNpScoreRankDataA* rankArray, u64 rankArraySize, OrbisNpScoreComment* commentArray,
    u64 commentArraySize, OrbisNpScoreGameInfo* infoArray, u64 infoArraySize, u64 arrayNum,
    Rtc::OrbisRtcTick* lastSortDate, OrbisNpScoreRankNumber* totalRecord, void* option) {
    LOG_INFO(Lib_NpScore, "(FAKE) called reqId={}, boardId={}, startSerialRank={}, arrayNum={}",
             reqId, boardId, startSerialRank, arrayNum);

    u64 fillCount = std::min(arrayNum, rankArraySize);
    fillCount = std::min(fillCount, FAKE_ENTRY_COUNT);

    FillRankDataA(rankArray, fillCount, lastSortDate, totalRecord, boardId);
    FillCommentData(commentArray, commentArraySize, fillCount, boardId);
    FillGameInfoData(infoArray, infoArraySize, fillCount);
    return ORBIS_OK;
}

int PS4_SYSV_ABI sceNpScoreGetRankingByRangeAAsync(
    s32 reqId, OrbisNpScoreBoardId boardId, OrbisNpScoreRankNumber startSerialRank,
    OrbisNpScoreRankDataA* rankArray, u64 rankArraySize, OrbisNpScoreComment* commentArray,
    u64 commentArraySize, OrbisNpScoreGameInfo* infoArray, u64 infoArraySize, u64 arrayNum,
    Rtc::OrbisRtcTick* lastSortDate, OrbisNpScoreRankNumber* totalRecord, void* option) {
    LOG_INFO(Lib_NpScore, "(FAKE) called reqId={}, boardId={}, startSerialRank={}, arrayNum={}",
             reqId, boardId, startSerialRank, arrayNum);

    u64 fillCount = std::min(arrayNum, rankArraySize);
    fillCount = std::min(fillCount, FAKE_ENTRY_COUNT);

    FillRankDataA(rankArray, fillCount, lastSortDate, totalRecord, boardId);
    FillCommentData(commentArray, commentArraySize, fillCount, boardId);
    FillGameInfoData(infoArray, infoArraySize, fillCount);
    return ORBIS_OK;
}

int PS4_SYSV_ABI sceNpScoreGetRankingByRangeAsync(
    s32 reqId, OrbisNpScoreBoardId boardId, OrbisNpScoreRankNumber startSerialRank,
    OrbisNpScoreRankData* rankArray, u64 rankArraySize, OrbisNpScoreComment* commentArray,
    u64 commentArraySize, OrbisNpScoreGameInfo* infoArray, u64 infoArraySize, u64 arrayNum,
    Rtc::OrbisRtcTick* lastSortDate, OrbisNpScoreRankNumber* totalRecord, void* option) {
    LOG_INFO(Lib_NpScore, "(FAKE) called reqId={}, boardId={}, startSerialRank={}, arrayNum={}",
             reqId, boardId, startSerialRank, arrayNum);

    u64 fillCount = std::min(arrayNum, rankArraySize);
    fillCount = std::min(fillCount, FAKE_ENTRY_COUNT);

    FillRankData(rankArray, fillCount, lastSortDate, totalRecord, nullptr, boardId);
    FillCommentData(commentArray, commentArraySize, fillCount, boardId);
    FillGameInfoData(infoArray, infoArraySize, fillCount);
    return ORBIS_OK;
}

int PS4_SYSV_ABI sceNpScoreGetRankingByRangeForCrossSave(
    s32 reqId, OrbisNpScoreBoardId boardId, OrbisNpScoreRankNumber startSerialRank,
    OrbisNpScoreRankDataForCrossSave* rankArray, u64 rankArraySize,
    OrbisNpScoreComment* commentArray, u64 commentArraySize, OrbisNpScoreGameInfo* infoArray,
    u64 infoArraySize, u64 arrayNum, Rtc::OrbisRtcTick* lastSortDate,
    OrbisNpScoreRankNumber* totalRecord, void* option) {
    LOG_INFO(Lib_NpScore, "(FAKE) called reqId={}, boardId={}, startSerialRank={}, arrayNum={}",
             reqId, boardId, startSerialRank, arrayNum);

    u64 fillCount = std::min(arrayNum, rankArraySize);
    fillCount = std::min(fillCount, FAKE_ENTRY_COUNT);

    FillRankDataForCrossSave(rankArray, fillCount, lastSortDate, totalRecord, boardId);
    FillCommentData(commentArray, commentArraySize, fillCount, boardId);
    FillGameInfoData(infoArray, infoArraySize, fillCount);
    return ORBIS_OK;
}

int PS4_SYSV_ABI sceNpScoreGetRankingByRangeForCrossSaveAsync(
    s32 reqId, OrbisNpScoreBoardId boardId, OrbisNpScoreRankNumber startSerialRank,
    OrbisNpScoreRankDataForCrossSave* rankArray, u64 rankArraySize,
    OrbisNpScoreComment* commentArray, u64 commentArraySize, OrbisNpScoreGameInfo* infoArray,
    u64 infoArraySize, u64 arrayNum, Rtc::OrbisRtcTick* lastSortDate,
    OrbisNpScoreRankNumber* totalRecord, void* option) {
    LOG_INFO(Lib_NpScore, "(FAKE) called reqId={}, boardId={}, startSerialRank={}, arrayNum={}",
             reqId, boardId, startSerialRank, arrayNum);

    u64 fillCount = std::min(arrayNum, rankArraySize);
    fillCount = std::min(fillCount, FAKE_ENTRY_COUNT);

    FillRankDataForCrossSave(rankArray, fillCount, lastSortDate, totalRecord, boardId);
    FillCommentData(commentArray, commentArraySize, fillCount, boardId);
    FillGameInfoData(infoArray, infoArraySize, fillCount);
    return ORBIS_OK;
}

int PS4_SYSV_ABI sceNpScorePollAsync(s32 reqId, s32* result) {
    LOG_INFO(Lib_NpScore, "called reqId={}, result={}", reqId, PTR(result));
    if (result) {
        *result = ORBIS_OK;
        LOG_INFO(Lib_NpScore, "Setting result to ORBIS_OK");
    }
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
             "called reqId={}, boardId={}, score={}, scoreComment={}, "
             "gameInfo={}, tmpRank={}, compareDate={}, option={}",
             reqId, boardId, score, PTR(scoreComment), PTR(gameInfo), PTR(tmpRank),
             PTR(compareDate), PTR(option));

    // Store the player's score
    UpdateCurrentPlayerScore(score);

    if (tmpRank) {
        *tmpRank = 1; // Assume rank 1 for now
        LOG_INFO(Lib_NpScore, "Setting tmpRank to 1");
    }
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
    if (sanitizedComment && comment)
        std::strncpy(sanitizedComment, comment, ORBIS_NP_SCORE_COMMENT_MAXLEN);
    return ORBIS_OK;
}

int PS4_SYSV_ABI sceNpScoreSanitizeCommentAsync(s32 reqId, const char* comment,
                                                char* sanitizedComment, void* option) {
    LOG_INFO(Lib_NpScore, "(STUBBED) called reqId={}, comment={}, sanitizedComment={}, option={}",
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
    LOG_INFO(Lib_NpScore, "sceNpScoreWaitAsync(reqId={}, result={})", reqId, PTR(result));
    if (result) {
        *result = ORBIS_OK;
        LOG_INFO(Lib_NpScore, "Setting result to ORBIS_OK");
    }
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