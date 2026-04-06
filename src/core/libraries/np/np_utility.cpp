// SPDX-FileCopyrightText: Copyright 2024-2026 shadPS4 Emulator Project
// SPDX-License-Identifier: GPL-2.0-or-later

#include <atomic>
#include <cstring>
#include <mutex>
#include <unordered_set>

#include "common/logging/log.h"
#include "core/libraries/error_codes.h"
#include "core/libraries/libs.h"
#include "core/libraries/np/np_utility.h"

namespace Libraries::Np::NpUtility {

s32 PS4_SYSV_ABI sceNpAppInfoIntAbortRequest() {
    LOG_ERROR(Lib_NpUtility, "(STUBBED) called");
    return ORBIS_OK;
}

s32 PS4_SYSV_ABI sceNpAppInfoIntCheckAvailability() {
    LOG_ERROR(Lib_NpUtility, "(STUBBED) called");
    return ORBIS_OK;
}

s32 PS4_SYSV_ABI sceNpAppInfoIntCheckAvailabilityA() {
    LOG_ERROR(Lib_NpUtility, "(STUBBED) called");
    return ORBIS_OK;
}

s32 PS4_SYSV_ABI sceNpAppInfoIntCheckAvailabilityAll() {
    LOG_ERROR(Lib_NpUtility, "(STUBBED) called");
    return ORBIS_OK;
}

s32 PS4_SYSV_ABI sceNpAppInfoIntCheckAvailabilityAllA() {
    LOG_ERROR(Lib_NpUtility, "(STUBBED) called");
    return ORBIS_OK;
}

s32 PS4_SYSV_ABI sceNpAppInfoIntCheckServiceAvailability() {
    LOG_ERROR(Lib_NpUtility, "(STUBBED) called");
    return ORBIS_OK;
}

s32 PS4_SYSV_ABI sceNpAppInfoIntCheckServiceAvailabilityA() {
    LOG_ERROR(Lib_NpUtility, "(STUBBED) called");
    return ORBIS_OK;
}

s32 PS4_SYSV_ABI sceNpAppInfoIntCheckServiceAvailabilityAll() {
    LOG_ERROR(Lib_NpUtility, "(STUBBED) called");
    return ORBIS_OK;
}

s32 PS4_SYSV_ABI sceNpAppInfoIntCheckServiceAvailabilityAllA() {
    LOG_ERROR(Lib_NpUtility, "(STUBBED) called");
    return ORBIS_OK;
}

s32 PS4_SYSV_ABI sceNpAppInfoIntCreateRequest() {
    LOG_ERROR(Lib_NpUtility, "(STUBBED) called");
    return ORBIS_OK;
}

s32 PS4_SYSV_ABI sceNpAppInfoIntDestroyRequest() {
    LOG_ERROR(Lib_NpUtility, "(STUBBED) called");
    return ORBIS_OK;
}

s32 PS4_SYSV_ABI sceNpAppInfoIntFinalize() {
    LOG_ERROR(Lib_NpUtility, "(STUBBED) called");
    return ORBIS_OK;
}

s32 PS4_SYSV_ABI sceNpAppInfoIntInitialize() {
    LOG_ERROR(Lib_NpUtility, "(STUBBED) called");
    return ORBIS_OK;
}

s32 PS4_SYSV_ABI sceNpAppLaunchLink2IntAbortRequest() {
    LOG_ERROR(Lib_NpUtility, "(STUBBED) called");
    return ORBIS_OK;
}

s32 PS4_SYSV_ABI sceNpAppLaunchLink2IntCreateRequest() {
    LOG_ERROR(Lib_NpUtility, "(STUBBED) called");
    return ORBIS_OK;
}

s32 PS4_SYSV_ABI sceNpAppLaunchLink2IntDestroyRequest() {
    LOG_ERROR(Lib_NpUtility, "(STUBBED) called");
    return ORBIS_OK;
}

s32 PS4_SYSV_ABI sceNpAppLaunchLink2IntFinalize() {
    LOG_ERROR(Lib_NpUtility, "(STUBBED) called");
    return ORBIS_OK;
}

s32 PS4_SYSV_ABI sceNpAppLaunchLink2IntGetCompatibleTitleIdList() {
    LOG_ERROR(Lib_NpUtility, "(STUBBED) called");
    return ORBIS_OK;
}

s32 PS4_SYSV_ABI sceNpAppLaunchLink2IntGetCompatibleTitleIdNum() {
    LOG_ERROR(Lib_NpUtility, "(STUBBED) called");
    return ORBIS_OK;
}

s32 PS4_SYSV_ABI sceNpAppLaunchLink2IntInitialize() {
    LOG_ERROR(Lib_NpUtility, "(STUBBED) called");
    return ORBIS_OK;
}

s32 PS4_SYSV_ABI sceNpAppLaunchLinkIntAbortRequest() {
    LOG_ERROR(Lib_NpUtility, "(STUBBED) called");
    return ORBIS_OK;
}

s32 PS4_SYSV_ABI sceNpAppLaunchLinkIntCreateRequest() {
    LOG_ERROR(Lib_NpUtility, "(STUBBED) called");
    return ORBIS_OK;
}

s32 PS4_SYSV_ABI sceNpAppLaunchLinkIntDestroyRequest() {
    LOG_ERROR(Lib_NpUtility, "(STUBBED) called");
    return ORBIS_OK;
}

s32 PS4_SYSV_ABI sceNpAppLaunchLinkIntFinalize() {
    LOG_ERROR(Lib_NpUtility, "(STUBBED) called");
    return ORBIS_OK;
}

s32 PS4_SYSV_ABI sceNpAppLaunchLinkIntGetCompatibleTitleIdList() {
    LOG_ERROR(Lib_NpUtility, "(STUBBED) called");
    return ORBIS_OK;
}

s32 PS4_SYSV_ABI sceNpAppLaunchLinkIntGetCompatibleTitleIdNum() {
    LOG_ERROR(Lib_NpUtility, "(STUBBED) called");
    return ORBIS_OK;
}

s32 PS4_SYSV_ABI sceNpAppLaunchLinkIntInitialize() {
    LOG_ERROR(Lib_NpUtility, "(STUBBED) called");
    return ORBIS_OK;
}

s32 PS4_SYSV_ABI sceNpBandwidthTestAbort() {
    LOG_ERROR(Lib_NpUtility, "(STUBBED) called");
    return ORBIS_OK;
}

s32 PS4_SYSV_ABI sceNpBandwidthTestDownloadOnlyInitStart() {
    LOG_ERROR(Lib_NpUtility, "(STUBBED) called");
    return ORBIS_OK;
}

s32 PS4_SYSV_ABI sceNpBandwidthTestGetStatus() {
    LOG_ERROR(Lib_NpUtility, "(STUBBED) called");
    return ORBIS_OK;
}

s32 PS4_SYSV_ABI sceNpBandwidthTestInitStart() {
    LOG_ERROR(Lib_NpUtility, "(STUBBED) called");
    return ORBIS_OK;
}

s32 PS4_SYSV_ABI sceNpBandwidthTestInitStartDownload() {
    LOG_ERROR(Lib_NpUtility, "(STUBBED) called");
    return ORBIS_OK;
}

s32 PS4_SYSV_ABI sceNpBandwidthTestInitStartUpload() {
    LOG_ERROR(Lib_NpUtility, "(STUBBED) called");
    return ORBIS_OK;
}

s32 PS4_SYSV_ABI sceNpBandwidthTestShutdown() {
    LOG_ERROR(Lib_NpUtility, "(STUBBED) called");
    return ORBIS_OK;
}

s32 PS4_SYSV_ABI sceNpBandwidthTestShutdownWithDetailedInfo() {
    LOG_ERROR(Lib_NpUtility, "(STUBBED) called");
    return ORBIS_OK;
}

s32 PS4_SYSV_ABI sceNpBandwidthTestUploadOnlyInitStart() {
    LOG_ERROR(Lib_NpUtility, "(STUBBED) called");
    return ORBIS_OK;
}

// --- sceNpLookup* ---
// The connection state machine calls these in states 0xD/0xE to resolve peer NpIds.
// Flow: state 0xD calls CreateAsyncRequest + NpId (starts async lookup)
//       state 0xE calls PollAsync (checks completion) → creates SigDataRef → sets ConnObj+0x70
//       state 0xE calls DeleteRequest (cleanup)

static std::atomic<s32> s_next_lookup_handle{1};
static std::unordered_set<s32> s_completed_lookups;
static std::mutex s_lookup_mutex;

s32 PS4_SYSV_ABI sceNpLookupAbortRequest(s32 requestHandle) {
    LOG_DEBUG(Lib_NpUtility, "handle={}", requestHandle);
    return ORBIS_OK;
}

s32 PS4_SYSV_ABI sceNpLookupCreateAsyncRequest(s32 titleCtxId, void* param) {
    // Returns a request handle stored at ConnObj+0xF0. Must be positive.
    s32 handle = s_next_lookup_handle++;
    LOG_DEBUG(Lib_NpUtility, "titleCtxId={} param={} -> handle={}", titleCtxId, param, handle);
    return handle;
}

s32 PS4_SYSV_ABI sceNpLookupCreateRequest(s32 titleCtxId, void* param) {
    s32 handle = s_next_lookup_handle++;
    LOG_DEBUG(Lib_NpUtility, "titleCtxId={} -> handle={}", titleCtxId, handle);
    return handle;
}

s32 PS4_SYSV_ABI sceNpLookupCreateTitleCtx(s32 titleId, void* npId, void* param) {
    // Called during NP init. Result stored at NP object +0x7C.
    // Passed to sceNpLookupCreateAsyncRequest as context.
    LOG_DEBUG(Lib_NpUtility, "titleId={} npId={} param={}", titleId, npId, param);
    return 1;
}

s32 PS4_SYSV_ABI sceNpLookupCreateTitleCtxA() {
    LOG_ERROR(Lib_NpUtility, "(STUBBED) called");
    return ORBIS_OK;
}

s32 PS4_SYSV_ABI sceNpLookupDeleteRequest(s32 requestHandle) {
    LOG_DEBUG(Lib_NpUtility, "handle={}", requestHandle);
    {
        std::lock_guard lock(s_lookup_mutex);
        s_completed_lookups.erase(requestHandle);
    }
    return ORBIS_OK;
}

s32 PS4_SYSV_ABI sceNpLookupDeleteTitleCtx(s32 titleCtxId) {
    LOG_DEBUG(Lib_NpUtility, "titleCtxId={}", titleCtxId);
    return ORBIS_OK;
}

s32 PS4_SYSV_ABI sceNpLookupNetAbortRequest() {
    LOG_ERROR(Lib_NpUtility, "(STUBBED) called");
    return ORBIS_OK;
}

s32 PS4_SYSV_ABI sceNpLookupNetCensorComment() {
    LOG_ERROR(Lib_NpUtility, "(STUBBED) called");
    return ORBIS_OK;
}

s32 PS4_SYSV_ABI sceNpLookupNetConvertJidToNpId() {
    LOG_ERROR(Lib_NpUtility, "(STUBBED) called");
    return ORBIS_OK;
}

s32 PS4_SYSV_ABI sceNpLookupNetConvertNpIdToJid() {
    LOG_ERROR(Lib_NpUtility, "(STUBBED) called");
    return ORBIS_OK;
}

s32 PS4_SYSV_ABI sceNpLookupNetCreateRequest() {
    LOG_ERROR(Lib_NpUtility, "(STUBBED) called");
    return ORBIS_OK;
}

s32 PS4_SYSV_ABI sceNpLookupNetCreateTitleCtx() {
    LOG_ERROR(Lib_NpUtility, "(STUBBED) called");
    return ORBIS_OK;
}

s32 PS4_SYSV_ABI sceNpLookupNetDeleteRequest() {
    LOG_ERROR(Lib_NpUtility, "(STUBBED) called");
    return ORBIS_OK;
}

s32 PS4_SYSV_ABI sceNpLookupNetDeleteTitleCtx() {
    LOG_ERROR(Lib_NpUtility, "(STUBBED) called");
    return ORBIS_OK;
}

s32 PS4_SYSV_ABI sceNpLookupNetInit() {
    LOG_ERROR(Lib_NpUtility, "(STUBBED) called");
    return ORBIS_OK;
}

s32 PS4_SYSV_ABI sceNpLookupNetInitWithFunctionPointer() {
    LOG_ERROR(Lib_NpUtility, "(STUBBED) called");
    return ORBIS_OK;
}

s32 PS4_SYSV_ABI sceNpLookupNetInitWithMemoryPool() {
    LOG_ERROR(Lib_NpUtility, "(STUBBED) called");
    return ORBIS_OK;
}

s32 PS4_SYSV_ABI sceNpLookupNetIsInit() {
    LOG_ERROR(Lib_NpUtility, "(STUBBED) called");
    return ORBIS_OK;
}

s32 PS4_SYSV_ABI sceNpLookupNetNpId() {
    LOG_ERROR(Lib_NpUtility, "(STUBBED) called");
    return ORBIS_OK;
}

s32 PS4_SYSV_ABI sceNpLookupNetSanitizeComment() {
    LOG_ERROR(Lib_NpUtility, "(STUBBED) called");
    return ORBIS_OK;
}

s32 PS4_SYSV_ABI sceNpLookupNetSetTimeout() {
    LOG_ERROR(Lib_NpUtility, "(STUBBED) called");
    return ORBIS_OK;
}

s32 PS4_SYSV_ABI sceNpLookupNetTerm() {
    LOG_ERROR(Lib_NpUtility, "(STUBBED) called");
    return ORBIS_OK;
}

s32 PS4_SYSV_ABI sceNpLookupNpId(s32 requestHandle, const char* onlineIdStr, void* npIdOut,
                                  s32 option) {
    // NpId layout (36 bytes): +0x00 online_id[16], +0x10 padding[4], +0x14 opt[8], +0x1C reserved[8]
    LOG_DEBUG(Lib_NpUtility, "handle={} onlineId='{}' option={}", requestHandle,
              onlineIdStr ? onlineIdStr : "null", option);
    if (npIdOut) {
        std::memset(npIdOut, 0, 36);
        if (onlineIdStr) {
            std::strncpy(static_cast<char*>(npIdOut), onlineIdStr, 15);
        }
    }
    {
        std::lock_guard lock(s_lookup_mutex);
        s_completed_lookups.insert(requestHandle);
    }
    return ORBIS_OK;
}

s32 PS4_SYSV_ABI sceNpLookupPollAsync(s32 requestHandle, s32* result) {
    // Return 0 = completed, 1 = pending. *result >= 0 = success.
    LOG_DEBUG(Lib_NpUtility, "handle={}", requestHandle);
    if (result) {
        *result = 0;
    }
    return 0; // completed
}

s32 PS4_SYSV_ABI sceNpLookupSetTimeout(s32 requestHandle, s32 timeout) {
    LOG_DEBUG(Lib_NpUtility, "handle={} timeout={}", requestHandle, timeout);
    return ORBIS_OK;
}

s32 PS4_SYSV_ABI sceNpLookupWaitAsync(s32 requestHandle, s32* result) {
    LOG_DEBUG(Lib_NpUtility, "handle={}", requestHandle);
    if (result) {
        *result = 0;
    }
    return ORBIS_OK;
}

s32 PS4_SYSV_ABI sceNpServiceChecker2IntAbortRequest() {
    LOG_ERROR(Lib_NpUtility, "(STUBBED) called");
    return ORBIS_OK;
}

s32 PS4_SYSV_ABI sceNpServiceChecker2IntCreateRequest() {
    LOG_ERROR(Lib_NpUtility, "(STUBBED) called");
    return ORBIS_OK;
}

s32 PS4_SYSV_ABI sceNpServiceChecker2IntDestroyRequest() {
    LOG_ERROR(Lib_NpUtility, "(STUBBED) called");
    return ORBIS_OK;
}

s32 PS4_SYSV_ABI sceNpServiceChecker2IntFinalize() {
    LOG_ERROR(Lib_NpUtility, "(STUBBED) called");
    return ORBIS_OK;
}

s32 PS4_SYSV_ABI sceNpServiceChecker2IntGetServiceAvailability() {
    LOG_ERROR(Lib_NpUtility, "(STUBBED) called");
    return ORBIS_OK;
}

s32 PS4_SYSV_ABI sceNpServiceChecker2IntGetServiceAvailabilityA() {
    LOG_ERROR(Lib_NpUtility, "(STUBBED) called");
    return ORBIS_OK;
}

s32 PS4_SYSV_ABI sceNpServiceChecker2IntInitialize() {
    LOG_ERROR(Lib_NpUtility, "(STUBBED) called");
    return ORBIS_OK;
}

s32 PS4_SYSV_ABI sceNpServiceChecker2IntIsSetServiceType() {
    LOG_ERROR(Lib_NpUtility, "(STUBBED) called");
    return ORBIS_OK;
}

s32 PS4_SYSV_ABI sceNpServiceCheckerIntAbortRequest() {
    LOG_ERROR(Lib_NpUtility, "(STUBBED) called");
    return ORBIS_OK;
}

s32 PS4_SYSV_ABI sceNpServiceCheckerIntCreateRequest() {
    LOG_ERROR(Lib_NpUtility, "(STUBBED) called");
    return ORBIS_OK;
}

s32 PS4_SYSV_ABI sceNpServiceCheckerIntDestroyRequest() {
    LOG_ERROR(Lib_NpUtility, "(STUBBED) called");
    return ORBIS_OK;
}

s32 PS4_SYSV_ABI sceNpServiceCheckerIntFinalize() {
    LOG_ERROR(Lib_NpUtility, "(STUBBED) called");
    return ORBIS_OK;
}

s32 PS4_SYSV_ABI sceNpServiceCheckerIntGetAvailability() {
    LOG_ERROR(Lib_NpUtility, "(STUBBED) called");
    return ORBIS_OK;
}

s32 PS4_SYSV_ABI sceNpServiceCheckerIntGetAvailabilityList() {
    LOG_ERROR(Lib_NpUtility, "(STUBBED) called");
    return ORBIS_OK;
}

s32 PS4_SYSV_ABI sceNpServiceCheckerIntInitialize() {
    LOG_ERROR(Lib_NpUtility, "(STUBBED) called");
    return ORBIS_OK;
}

s32 PS4_SYSV_ABI sceNpServiceCheckerIntIsCached() {
    LOG_ERROR(Lib_NpUtility, "(STUBBED) called");
    return ORBIS_OK;
}

s32 PS4_SYSV_ABI sceNpTitleMetadataIntAbortRequest() {
    LOG_ERROR(Lib_NpUtility, "(STUBBED) called");
    return ORBIS_OK;
}

s32 PS4_SYSV_ABI sceNpTitleMetadataIntCreateRequest() {
    LOG_ERROR(Lib_NpUtility, "(STUBBED) called");
    return ORBIS_OK;
}

s32 PS4_SYSV_ABI sceNpTitleMetadataIntDeleteRequest() {
    LOG_ERROR(Lib_NpUtility, "(STUBBED) called");
    return ORBIS_OK;
}

s32 PS4_SYSV_ABI sceNpTitleMetadataIntGetInfo() {
    LOG_ERROR(Lib_NpUtility, "(STUBBED) called");
    return ORBIS_OK;
}

s32 PS4_SYSV_ABI sceNpTitleMetadataIntGetNpTitleId() {
    LOG_ERROR(Lib_NpUtility, "(STUBBED) called");
    return ORBIS_OK;
}

s32 PS4_SYSV_ABI sceNpUtilityInit() {
    LOG_ERROR(Lib_NpUtility, "(STUBBED) called");
    return ORBIS_OK;
}

s32 PS4_SYSV_ABI sceNpUtilityTerm() {
    LOG_ERROR(Lib_NpUtility, "(STUBBED) called");
    return ORBIS_OK;
}

s32 PS4_SYSV_ABI sceNpWordFilterAbortRequest() {
    LOG_ERROR(Lib_NpUtility, "(STUBBED) called");
    return ORBIS_OK;
}

s32 PS4_SYSV_ABI sceNpWordFilterCensorComment() {
    LOG_ERROR(Lib_NpUtility, "(STUBBED) called");
    return ORBIS_OK;
}

s32 PS4_SYSV_ABI sceNpWordFilterCreateAsyncRequest() {
    LOG_ERROR(Lib_NpUtility, "(STUBBED) called");
    return ORBIS_OK;
}

s32 PS4_SYSV_ABI sceNpWordFilterCreateRequest() {
    LOG_ERROR(Lib_NpUtility, "(STUBBED) called");
    return ORBIS_OK;
}

s32 PS4_SYSV_ABI sceNpWordFilterCreateTitleCtx() {
    LOG_ERROR(Lib_NpUtility, "(STUBBED) called");
    return ORBIS_OK;
}

s32 PS4_SYSV_ABI sceNpWordFilterCreateTitleCtxA() {
    LOG_ERROR(Lib_NpUtility, "(STUBBED) called");
    return ORBIS_OK;
}

s32 PS4_SYSV_ABI sceNpWordFilterDeleteRequest() {
    LOG_ERROR(Lib_NpUtility, "(STUBBED) called");
    return ORBIS_OK;
}

s32 PS4_SYSV_ABI sceNpWordFilterDeleteTitleCtx() {
    LOG_ERROR(Lib_NpUtility, "(STUBBED) called");
    return ORBIS_OK;
}

s32 PS4_SYSV_ABI sceNpWordFilterPollAsync() {
    LOG_ERROR(Lib_NpUtility, "(STUBBED) called");
    return ORBIS_OK;
}

s32 PS4_SYSV_ABI sceNpWordFilterSanitizeComment() {
    LOG_ERROR(Lib_NpUtility, "(STUBBED) called");
    return ORBIS_OK;
}

s32 PS4_SYSV_ABI sceNpWordFilterSetTimeout() {
    LOG_ERROR(Lib_NpUtility, "(STUBBED) called");
    return ORBIS_OK;
}

s32 PS4_SYSV_ABI sceNpWordFilterWaitAsync() {
    LOG_ERROR(Lib_NpUtility, "(STUBBED) called");
    return ORBIS_OK;
}

void RegisterLib(Core::Loader::SymbolsResolver* sym) {
    LIB_FUNCTION("Y797Sw9-jqY", "libSceNpUtility", 1, "libSceNpUtility",
                 sceNpAppInfoIntAbortRequest);
    LIB_FUNCTION("UUhI+IUMrcE", "libSceNpUtility", 1, "libSceNpUtility",
                 sceNpAppInfoIntCheckAvailability);
    LIB_FUNCTION("ASonnwltwEk", "libSceNpUtility", 1, "libSceNpUtility",
                 sceNpAppInfoIntCheckAvailabilityA);
    LIB_FUNCTION("jXx0+2Wd1q8", "libSceNpUtility", 1, "libSceNpUtility",
                 sceNpAppInfoIntCheckAvailabilityAll);
    LIB_FUNCTION("f1OwQ7jdqn0", "libSceNpUtility", 1, "libSceNpUtility",
                 sceNpAppInfoIntCheckAvailabilityAllA);
    LIB_FUNCTION("1mfDBl40Dms", "libSceNpUtility", 1, "libSceNpUtility",
                 sceNpAppInfoIntCheckServiceAvailability);
    LIB_FUNCTION("XAmDowAQhFs", "libSceNpUtility", 1, "libSceNpUtility",
                 sceNpAppInfoIntCheckServiceAvailabilityA);
    LIB_FUNCTION("BaihFa8LBw0", "libSceNpUtility", 1, "libSceNpUtility",
                 sceNpAppInfoIntCheckServiceAvailabilityAll);
    LIB_FUNCTION("JcqdKidhuK0", "libSceNpUtility", 1, "libSceNpUtility",
                 sceNpAppInfoIntCheckServiceAvailabilityAllA);
    LIB_FUNCTION("cXpyESo49ko", "libSceNpUtility", 1, "libSceNpUtility",
                 sceNpAppInfoIntCreateRequest);
    LIB_FUNCTION("pRgpBtHx8P4", "libSceNpUtility", 1, "libSceNpUtility",
                 sceNpAppInfoIntDestroyRequest);
    LIB_FUNCTION("s9+zoKE8cBA", "libSceNpUtility", 1, "libSceNpUtility", sceNpAppInfoIntFinalize);
    LIB_FUNCTION("l6Dl+2zlua0", "libSceNpUtility", 1, "libSceNpUtility",
                 sceNpAppInfoIntInitialize);
    LIB_FUNCTION("OHCO6MMFvdQ", "libSceNpUtility", 1, "libSceNpUtility",
                 sceNpAppLaunchLink2IntAbortRequest);
    LIB_FUNCTION("B6IXdHGBL-g", "libSceNpUtility", 1, "libSceNpUtility",
                 sceNpAppLaunchLink2IntCreateRequest);
    LIB_FUNCTION("0H0JBpVp03o", "libSceNpUtility", 1, "libSceNpUtility",
                 sceNpAppLaunchLink2IntDestroyRequest);
    LIB_FUNCTION("FWonlDV6d5k", "libSceNpUtility", 1, "libSceNpUtility",
                 sceNpAppLaunchLink2IntFinalize);
    LIB_FUNCTION("PdYx470F6B8", "libSceNpUtility", 1, "libSceNpUtility",
                 sceNpAppLaunchLink2IntGetCompatibleTitleIdList);
    LIB_FUNCTION("tesM6ViaX6M", "libSceNpUtility", 1, "libSceNpUtility",
                 sceNpAppLaunchLink2IntGetCompatibleTitleIdNum);
    LIB_FUNCTION("DK6xpBP1gxw", "libSceNpUtility", 1, "libSceNpUtility",
                 sceNpAppLaunchLink2IntInitialize);
    LIB_FUNCTION("AQV4A8YFx44", "libSceNpUtility", 1, "libSceNpUtility",
                 sceNpAppLaunchLinkIntAbortRequest);
    LIB_FUNCTION("9YhwG4DhwtU", "libSceNpUtility", 1, "libSceNpUtility",
                 sceNpAppLaunchLinkIntCreateRequest);
    LIB_FUNCTION("-8Wn4YKZLMM", "libSceNpUtility", 1, "libSceNpUtility",
                 sceNpAppLaunchLinkIntDestroyRequest);
    LIB_FUNCTION("TnQqJsyek5o", "libSceNpUtility", 1, "libSceNpUtility",
                 sceNpAppLaunchLinkIntFinalize);
    LIB_FUNCTION("GB7Fhk5SUaA", "libSceNpUtility", 1, "libSceNpUtility",
                 sceNpAppLaunchLinkIntGetCompatibleTitleIdList);
    LIB_FUNCTION("X4elOoiAtB4", "libSceNpUtility", 1, "libSceNpUtility",
                 sceNpAppLaunchLinkIntGetCompatibleTitleIdNum);
    LIB_FUNCTION("1F4yweQoqgg", "libSceNpUtility", 1, "libSceNpUtility",
                 sceNpAppLaunchLinkIntInitialize);
    LIB_FUNCTION("kvdMF48mB3Y", "libSceNpUtility", 1, "libSceNpUtility", sceNpBandwidthTestAbort);
    LIB_FUNCTION("DWWW02MbKdk", "libSceNpUtility", 1, "libSceNpUtility",
                 sceNpBandwidthTestDownloadOnlyInitStart);
    LIB_FUNCTION("BYIZGKm6bO4", "libSceNpUtility", 1, "libSceNpUtility",
                 sceNpBandwidthTestGetStatus);
    LIB_FUNCTION("jktww3yJXnc", "libSceNpUtility", 1, "libSceNpUtility",
                 sceNpBandwidthTestInitStart);
    LIB_FUNCTION("hqzi1IHdQQQ", "libSceNpUtility", 1, "libSceNpUtility",
                 sceNpBandwidthTestInitStartDownload);
    LIB_FUNCTION("mA0zsbqm+kA", "libSceNpUtility", 1, "libSceNpUtility",
                 sceNpBandwidthTestInitStartUpload);
    LIB_FUNCTION("pLr1fEQS1z8", "libSceNpUtility", 1, "libSceNpUtility",
                 sceNpBandwidthTestShutdown);
    LIB_FUNCTION("tyArYWj+1QE", "libSceNpUtility", 1, "libSceNpUtility",
                 sceNpBandwidthTestShutdownWithDetailedInfo);
    LIB_FUNCTION("oXOyqxO8dX8", "libSceNpUtility", 1, "libSceNpUtility",
                 sceNpBandwidthTestUploadOnlyInitStart);
    LIB_FUNCTION("eYz4v5Uek9U", "libSceNpUtility", 1, "libSceNpUtility", sceNpLookupAbortRequest);
    LIB_FUNCTION("JA4+sS39GMs", "libSceNpUtility", 1, "libSceNpUtility",
                 sceNpLookupCreateAsyncRequest);
    LIB_FUNCTION("iQr9UxPHUFs", "libSceNpUtility", 1, "libSceNpUtility", sceNpLookupCreateRequest);
    LIB_FUNCTION("8533Q+LU7EQ", "libSceNpUtility", 1, "libSceNpUtility",
                 sceNpLookupCreateTitleCtx);
    LIB_FUNCTION("vT9xhqPO6+0", "libSceNpUtility", 1, "libSceNpUtility",
                 sceNpLookupCreateTitleCtxA);
    LIB_FUNCTION("wLaxchvEEnk", "libSceNpUtility", 1, "libSceNpUtility", sceNpLookupDeleteRequest);
    LIB_FUNCTION("mtqDK9zkoIE", "libSceNpUtility", 1, "libSceNpUtility",
                 sceNpLookupDeleteTitleCtx);
    LIB_FUNCTION("1O96muPzhgU", "libSceNpUtility", 1, "libSceNpUtility",
                 sceNpLookupNetAbortRequest);
    LIB_FUNCTION("N0iF180VjGk", "libSceNpUtility", 1, "libSceNpUtility",
                 sceNpLookupNetCensorComment);
    LIB_FUNCTION("UI5t6Rx6s5I", "libSceNpUtility", 1, "libSceNpUtility",
                 sceNpLookupNetConvertJidToNpId);
    LIB_FUNCTION("ieROYX4vspk", "libSceNpUtility", 1, "libSceNpUtility",
                 sceNpLookupNetConvertNpIdToJid);
    LIB_FUNCTION("KUIRsku7EPk", "libSceNpUtility", 1, "libSceNpUtility",
                 sceNpLookupNetCreateRequest);
    LIB_FUNCTION("8DPEdJh9RkE", "libSceNpUtility", 1, "libSceNpUtility",
                 sceNpLookupNetCreateTitleCtx);
    LIB_FUNCTION("HL-venrRcnQ", "libSceNpUtility", 1, "libSceNpUtility",
                 sceNpLookupNetDeleteRequest);
    LIB_FUNCTION("dxpUx7z9StY", "libSceNpUtility", 1, "libSceNpUtility",
                 sceNpLookupNetDeleteTitleCtx);
    LIB_FUNCTION("zVZE+fAhgFY", "libSceNpUtility", 1, "libSceNpUtility", sceNpLookupNetInit);
    LIB_FUNCTION("DiUk6-mq--0", "libSceNpUtility", 1, "libSceNpUtility",
                 sceNpLookupNetInitWithFunctionPointer);
    LIB_FUNCTION("cpnwZeVIq8E", "libSceNpUtility", 1, "libSceNpUtility",
                 sceNpLookupNetInitWithMemoryPool);
    LIB_FUNCTION("ZXlTj9RRCFo", "libSceNpUtility", 1, "libSceNpUtility", sceNpLookupNetIsInit);
    LIB_FUNCTION("2nEVmFiV6OI", "libSceNpUtility", 1, "libSceNpUtility", sceNpLookupNetNpId);
    LIB_FUNCTION("jJH2P7KA4XU", "libSceNpUtility", 1, "libSceNpUtility",
                 sceNpLookupNetSanitizeComment);
    LIB_FUNCTION("NWtf77WCXJs", "libSceNpUtility", 1, "libSceNpUtility", sceNpLookupNetSetTimeout);
    LIB_FUNCTION("Dbd5BY0QjG0", "libSceNpUtility", 1, "libSceNpUtility", sceNpLookupNetTerm);
    LIB_FUNCTION("T6tnM1Uti4g", "libSceNpUtility", 1, "libSceNpUtility", sceNpLookupNpId);
    LIB_FUNCTION("V4EVrruHuy8", "libSceNpUtility", 1, "libSceNpUtility", sceNpLookupPollAsync);
    LIB_FUNCTION("0MV72WO7V34", "libSceNpUtility", 1, "libSceNpUtility", sceNpLookupSetTimeout);
    LIB_FUNCTION("YX9dAus6baE", "libSceNpUtility", 1, "libSceNpUtility", sceNpLookupWaitAsync);
    LIB_FUNCTION("Kq+ftR9LHlE", "libSceNpUtility", 1, "libSceNpUtility",
                 sceNpServiceChecker2IntAbortRequest);
    LIB_FUNCTION("IG1Kd+k6U3s", "libSceNpUtility", 1, "libSceNpUtility",
                 sceNpServiceChecker2IntCreateRequest);
    LIB_FUNCTION("hBsBswrAiGM", "libSceNpUtility", 1, "libSceNpUtility",
                 sceNpServiceChecker2IntDestroyRequest);
    LIB_FUNCTION("cvZrmlSlwn8", "libSceNpUtility", 1, "libSceNpUtility",
                 sceNpServiceChecker2IntFinalize);
    LIB_FUNCTION("aUgLCb3pSOo", "libSceNpUtility", 1, "libSceNpUtility",
                 sceNpServiceChecker2IntGetServiceAvailability);
    LIB_FUNCTION("Yp2yK5YXb78", "libSceNpUtility", 1, "libSceNpUtility",
                 sceNpServiceChecker2IntGetServiceAvailabilityA);
    LIB_FUNCTION("-Afi-JoRZ-U", "libSceNpUtility", 1, "libSceNpUtility",
                 sceNpServiceChecker2IntInitialize);
    LIB_FUNCTION("ukBq62OPAYA", "libSceNpUtility", 1, "libSceNpUtility",
                 sceNpServiceChecker2IntIsSetServiceType);
    LIB_FUNCTION("waeEzwwYfZY", "libSceNpUtility", 1, "libSceNpUtility",
                 sceNpServiceCheckerIntAbortRequest);
    LIB_FUNCTION("YLXt-vGw4Kg", "libSceNpUtility", 1, "libSceNpUtility",
                 sceNpServiceCheckerIntCreateRequest);
    LIB_FUNCTION("85ZWdzWYgas", "libSceNpUtility", 1, "libSceNpUtility",
                 sceNpServiceCheckerIntDestroyRequest);
    LIB_FUNCTION("LSQ3xApEoxY", "libSceNpUtility", 1, "libSceNpUtility",
                 sceNpServiceCheckerIntFinalize);
    LIB_FUNCTION("wIX00Brskoc", "libSceNpUtility", 1, "libSceNpUtility",
                 sceNpServiceCheckerIntGetAvailability);
    LIB_FUNCTION("MjOFdwXYRKY", "libSceNpUtility", 1, "libSceNpUtility",
                 sceNpServiceCheckerIntGetAvailabilityList);
    LIB_FUNCTION("rT9Yk55JGho", "libSceNpUtility", 1, "libSceNpUtility",
                 sceNpServiceCheckerIntInitialize);
    LIB_FUNCTION("az7fl9snOqw", "libSceNpUtility", 1, "libSceNpUtility",
                 sceNpServiceCheckerIntIsCached);
    LIB_FUNCTION("rei4kjOSiyc", "libSceNpUtility", 1, "libSceNpUtility",
                 sceNpTitleMetadataIntAbortRequest);
    LIB_FUNCTION("A1XQslLAA-Y", "libSceNpUtility", 1, "libSceNpUtility",
                 sceNpTitleMetadataIntCreateRequest);
    LIB_FUNCTION("tynva-9jrtI", "libSceNpUtility", 1, "libSceNpUtility",
                 sceNpTitleMetadataIntDeleteRequest);
    LIB_FUNCTION("-McDhX8tnWE", "libSceNpUtility", 1, "libSceNpUtility",
                 sceNpTitleMetadataIntGetInfo);
    LIB_FUNCTION("LXHkrCV453o", "libSceNpUtility", 1, "libSceNpUtility",
                 sceNpTitleMetadataIntGetNpTitleId);
    LIB_FUNCTION("W6iWw8aUQtA", "libSceNpUtility", 1, "libSceNpUtility", sceNpUtilityInit);
    LIB_FUNCTION("M5Jyo9TKYPI", "libSceNpUtility", 1, "libSceNpUtility", sceNpUtilityTerm);
    LIB_FUNCTION("rAOOqDAxBIk", "libSceNpUtility", 1, "libSceNpUtility",
                 sceNpWordFilterAbortRequest);
    LIB_FUNCTION("1dMndqL-QgE", "libSceNpUtility", 1, "libSceNpUtility",
                 sceNpWordFilterCensorComment);
    LIB_FUNCTION("IEB+vgVoQbw", "libSceNpUtility", 1, "libSceNpUtility",
                 sceNpWordFilterCreateAsyncRequest);
    LIB_FUNCTION("iCq5xW5KQW4", "libSceNpUtility", 1, "libSceNpUtility",
                 sceNpWordFilterCreateRequest);
    LIB_FUNCTION("r9BgI0PfJZg", "libSceNpUtility", 1, "libSceNpUtility",
                 sceNpWordFilterCreateTitleCtx);
    LIB_FUNCTION("6p9jvljuvsw", "libSceNpUtility", 1, "libSceNpUtility",
                 sceNpWordFilterCreateTitleCtxA);
    LIB_FUNCTION("PYFS1H70bDs", "libSceNpUtility", 1, "libSceNpUtility",
                 sceNpWordFilterDeleteRequest);
    LIB_FUNCTION("t0P5z5yuFPA", "libSceNpUtility", 1, "libSceNpUtility",
                 sceNpWordFilterDeleteTitleCtx);
    LIB_FUNCTION("ur5SShyG0dk", "libSceNpUtility", 1, "libSceNpUtility", sceNpWordFilterPollAsync);
    LIB_FUNCTION("Jj4mkpFO2gE", "libSceNpUtility", 1, "libSceNpUtility",
                 sceNpWordFilterSanitizeComment);
    LIB_FUNCTION("Fa4dVWgmffk", "libSceNpUtility", 1, "libSceNpUtility",
                 sceNpWordFilterSetTimeout);
    LIB_FUNCTION("87ivWj5yKzg", "libSceNpUtility", 1, "libSceNpUtility", sceNpWordFilterWaitAsync);
    LIB_FUNCTION("8533Q+LU7EQ", "libSceNpUtilityCompat", 1, "libSceNpUtility",
                 sceNpLookupCreateTitleCtx);
    LIB_FUNCTION("T6tnM1Uti4g", "libSceNpUtilityCompat", 1, "libSceNpUtility", sceNpLookupNpId);
    LIB_FUNCTION("r9BgI0PfJZg", "libSceNpUtilityCompat", 1, "libSceNpUtility",
                 sceNpWordFilterCreateTitleCtx);
};

} // namespace Libraries::Np::NpUtility