// SPDX-FileCopyrightText: Copyright 2024-2026 shadPS4 Emulator Project
// SPDX-License-Identifier: GPL-2.0-or-later

#include <magic_enum/magic_enum.hpp>
#include "common/logging/log.h"
#include "core/libraries/audio/audio_out.h"
#include "core/libraries/audio/audioout_error.h"
#include "core/libraries/error_codes.h"
#include "core/libraries/libs.h"

namespace Libraries::AudioOut {

s32 PS4_SYSV_ABI sceAudioOutDeviceIdOpen() {
    LOG_ERROR(Lib_AudioOut, "(STUBBED) called");
    return ORBIS_OK;
}

s32 PS4_SYSV_ABI sceAudioDeviceControlGet() {
    LOG_ERROR(Lib_AudioOut, "(STUBBED) called");
    return ORBIS_OK;
}

s32 PS4_SYSV_ABI sceAudioDeviceControlSet() {
    LOG_ERROR(Lib_AudioOut, "(STUBBED) called");
    return ORBIS_OK;
}

s32 PS4_SYSV_ABI sceAudioOutA3dControl() {
    LOG_ERROR(Lib_AudioOut, "(STUBBED) called");
    return ORBIS_OK;
}

s32 PS4_SYSV_ABI sceAudioOutA3dExit() {
    LOG_ERROR(Lib_AudioOut, "(STUBBED) called");
    return ORBIS_OK;
}

s32 PS4_SYSV_ABI sceAudioOutA3dInit() {
    LOG_ERROR(Lib_AudioOut, "(STUBBED) called");
    return ORBIS_OK;
}

s32 PS4_SYSV_ABI sceAudioOutAttachToApplicationByPid() {
    LOG_ERROR(Lib_AudioOut, "(STUBBED) called");
    return ORBIS_OK;
}

s32 PS4_SYSV_ABI sceAudioOutChangeAppModuleState() {
    LOG_ERROR(Lib_AudioOut, "(STUBBED) called");
    return ORBIS_OK;
}

s32 PS4_SYSV_ABI sceAudioOutClose(s32 handle) {
    LOG_ERROR(Lib_AudioOut, "(STUBBED) called, handle={}", handle);
    return ORBIS_OK;
}

s32 PS4_SYSV_ABI sceAudioOutDetachFromApplicationByPid() {
    LOG_ERROR(Lib_AudioOut, "(STUBBED) called");
    return ORBIS_OK;
}

s32 PS4_SYSV_ABI sceAudioOutExConfigureOutputMode() {
    LOG_ERROR(Lib_AudioOut, "(STUBBED) called");
    return ORBIS_OK;
}

s32 PS4_SYSV_ABI sceAudioOutExGetSystemInfo() {
    LOG_ERROR(Lib_AudioOut, "(STUBBED) called");
    return ORBIS_OK;
}

s32 PS4_SYSV_ABI sceAudioOutExPtClose() {
    LOG_ERROR(Lib_AudioOut, "(STUBBED) called");
    return ORBIS_OK;
}

s32 PS4_SYSV_ABI sceAudioOutExPtGetLastOutputTime() {
    LOG_ERROR(Lib_AudioOut, "(STUBBED) called");
    return ORBIS_OK;
}

s32 PS4_SYSV_ABI sceAudioOutExPtOpen() {
    LOG_ERROR(Lib_AudioOut, "(STUBBED) called");
    return ORBIS_OK;
}

s32 PS4_SYSV_ABI sceAudioOutExSystemInfoIsSupportedAudioOutExMode() {
    LOG_ERROR(Lib_AudioOut, "(STUBBED) called");
    return ORBIS_OK;
}

s32 PS4_SYSV_ABI sceAudioOutGetFocusEnablePid() {
    LOG_ERROR(Lib_AudioOut, "(STUBBED) called");
    return ORBIS_OK;
}

s32 PS4_SYSV_ABI sceAudioOutGetHandleStatusInfo() {
    LOG_ERROR(Lib_AudioOut, "(STUBBED) called");
    return ORBIS_OK;
}

s32 PS4_SYSV_ABI sceAudioOutGetInfo() {
    LOG_ERROR(Lib_AudioOut, "(STUBBED) called");
    return ORBIS_OK;
}

s32 PS4_SYSV_ABI sceAudioOutGetInfoOpenNum() {
    LOG_ERROR(Lib_AudioOut, "(STUBBED) called");
    return ORBIS_OK;
}

s32 PS4_SYSV_ABI sceAudioOutGetLastOutputTime(s32 handle, u64* output_time) {
    LOG_ERROR(Lib_AudioOut, "(STUBBED) called, handle={}, output_time={}", handle,
              fmt::ptr(output_time));
    return ORBIS_OK;
}

s32 PS4_SYSV_ABI sceAudioOutGetPortState(s32 handle, OrbisAudioOutPortState* state) {
    if (state) {
        LOG_ERROR(Lib_AudioOut,
                  "(STUBBED) called, handle={}, state={}, output={}, channel={}, volume={}, "
                  "rerouteCounter={}, flag={}",
                  handle, fmt::ptr(state), state->output, state->channel, state->volume,
                  state->rerouteCounter, state->flag);
    } else {
        LOG_ERROR(Lib_AudioOut, "(STUBBED) called, handle={}, state=nullptr", handle);
    }
    return ORBIS_OK;
}

s32 PS4_SYSV_ABI sceAudioOutGetSimulatedBusUsableStatusByBusType() {
    LOG_ERROR(Lib_AudioOut, "(STUBBED) called");
    return ORBIS_OK;
}

s32 PS4_SYSV_ABI sceAudioOutGetSimulatedHandleStatusInfo() {
    LOG_ERROR(Lib_AudioOut, "(STUBBED) called");
    return ORBIS_OK;
}

s32 PS4_SYSV_ABI sceAudioOutGetSimulatedHandleStatusInfo2() {
    LOG_ERROR(Lib_AudioOut, "(STUBBED) called");
    return ORBIS_OK;
}

s32 PS4_SYSV_ABI sceAudioOutGetSparkVss() {
    LOG_ERROR(Lib_AudioOut, "(STUBBED) called");
    return ORBIS_OK;
}

s32 PS4_SYSV_ABI sceAudioOutGetSystemState() {
    LOG_ERROR(Lib_AudioOut, "(STUBBED) called");
    return ORBIS_OK;
}

s32 PS4_SYSV_ABI sceAudioOutInit() {
    LOG_ERROR(Lib_AudioOut, "(STUBBED) called");
    return ORBIS_OK;
}

s32 PS4_SYSV_ABI sceAudioOutInitIpmiGetSession() {
    LOG_ERROR(Lib_AudioOut, "(STUBBED) called");
    return ORBIS_OK;
}

s32 PS4_SYSV_ABI sceAudioOutMasteringGetState() {
    LOG_ERROR(Lib_AudioOut, "(STUBBED) called");
    return ORBIS_OK;
}

s32 PS4_SYSV_ABI sceAudioOutMasteringInit() {
    LOG_ERROR(Lib_AudioOut, "(STUBBED) called");
    return ORBIS_OK;
}

s32 PS4_SYSV_ABI sceAudioOutMasteringSetParam() {
    LOG_ERROR(Lib_AudioOut, "(STUBBED) called");
    return ORBIS_OK;
}

s32 PS4_SYSV_ABI sceAudioOutMasteringTerm() {
    LOG_ERROR(Lib_AudioOut, "(STUBBED) called");
    return ORBIS_OK;
}

s32 PS4_SYSV_ABI sceAudioOutMbusInit() {
    LOG_ERROR(Lib_AudioOut, "(STUBBED) called");
    return ORBIS_OK;
}

s32 PS4_SYSV_ABI sceAudioOutOpen(UserService::OrbisUserServiceUserId user_id,
                                 OrbisAudioOutPort port_type, s32 index, u32 length,
                                 u32 sample_rate,
                                 OrbisAudioOutParamExtendedInformation param_type) {
    LOG_ERROR(Lib_AudioOut,
              "(STUBBED) called, user_id={}, port_type={}({}), index={}, length={}, "
              "sample_rate={}, data_format={}({}), attributes={}({})",
              user_id, magic_enum::enum_name(port_type), static_cast<u32>(port_type), index, length,
              sample_rate, magic_enum::enum_name(param_type.data_format.Value()),
              static_cast<u32>(param_type.data_format.Value()),
              magic_enum::enum_name(param_type.attributes.Value()),
              static_cast<u32>(param_type.attributes.Value()));

    /* Initialization checks */
    // if (_lazy_init == 0 || g_audioout_interface == nullptr) {
    //     return ORBIS_AUDIO_OUT_ERROR_NOT_INIT;
    // }

    /* len must be nonzero, ≤2048, and a multiple of 256 */
    if (length == 0 || length > 2048 || (length & 0xFF) != 0) {
        return ORBIS_AUDIO_OUT_ERROR_INVALID_SIZE;
    }
    u32 param_raw = 0;
    std::memcpy(&param_raw, &param_type, sizeof(param_raw));

    /* param format restrictions */
    if (port_type != OrbisAudioOutPort::Personal && (param_raw & 0x00020000) != 0) {
        return ORBIS_AUDIO_OUT_ERROR_INVALID_FORMAT;
    }

    if (port_type != OrbisAudioOutPort::Main && (param_raw & 0x70000000) != 0) {
        return ORBIS_AUDIO_OUT_ERROR_INVALID_FORMAT;
    }

    if ((param_raw & 0x8FFcff00) != 0) {
        return ORBIS_AUDIO_OUT_ERROR_INVALID_FORMAT;
    }

    /*
     * Validate port type + sample rate
     */
    u32 utype = static_cast<u32>(port_type);

    switch (utype) {
    case static_cast<u32>(OrbisAudioOutPort::Main):
    case static_cast<u32>(OrbisAudioOutPort::Bgm):
    case static_cast<u32>(OrbisAudioOutPort::Voice):
    case static_cast<u32>(OrbisAudioOutPort::Personal):
    case static_cast<u32>(OrbisAudioOutPort::PadSpk):
    case static_cast<u32>(OrbisAudioOutPort::Aux):
    case static_cast<u32>(OrbisAudioOutPort::Audio3d):
    case static_cast<u32>(OrbisAudioOutPort::Unk1):
        if (sample_rate != 48000) {
            return ORBIS_AUDIO_OUT_ERROR_INVALID_SAMPLE_FREQ;
        }
        break;
    default:
        return ORBIS_AUDIO_OUT_ERROR_INVALID_PORT_TYPE;
    }

    /* clear high bit of port type */
    utype &= 0x7FFFFFFF;

    /*
     * Open backend port
     * (stubbed for now)
     */
    // mtx_lock(&g_port_lock);
    s32 port_index = 1; /* temporary stub */
    // s32 port_index = _out_open(user_id, utype, length, param_type);
    // mtx_unlock(&g_port_lock);

    

    if (port_index < 0) {
        return port_index;
    }

    /*
     * Build PS4 audio handle
     *  bits 31..29 : 0b001
     *  bits 28..16 : port type
     *  bits 15..0  : port index
     */
    s32 handle =
        static_cast<s32>((utype << 16) | (static_cast<u32>(port_index) & 0xFFFF) | 0x20000000);

    return handle;
}

s32 PS4_SYSV_ABI sceAudioOutOpenEx() {
    LOG_ERROR(Lib_AudioOut, "(STUBBED) called");
    return ORBIS_OK;
}

s32 PS4_SYSV_ABI sceAudioOutOutput(s32 handle, void* ptr) {
    LOG_ERROR(Lib_AudioOut, "(STUBBED) called, handle={}, ptr={}", handle, fmt::ptr(ptr));
    return ORBIS_OK;
}

s32 PS4_SYSV_ABI sceAudioOutOutputs(OrbisAudioOutOutputParam* param, u32 num) {
    if (param) {
        LOG_ERROR(Lib_AudioOut, "(STUBBED) called, param={}, num={}", fmt::ptr(param), num);
        for (u32 i = 0; i < num; i++) {
            LOG_ERROR(Lib_AudioOut, "  [{}] handle={}, ptr={}", i, param[i].handle,
                      fmt::ptr(param[i].ptr));
        }
    } else {
        LOG_ERROR(Lib_AudioOut, "(STUBBED) called, param=nullptr, num={}", num);
    }
    return ORBIS_OK;
}

s32 PS4_SYSV_ABI sceAudioOutPtClose() {
    LOG_ERROR(Lib_AudioOut, "(STUBBED) called");
    return ORBIS_OK;
}

s32 PS4_SYSV_ABI sceAudioOutPtGetLastOutputTime() {
    LOG_ERROR(Lib_AudioOut, "(STUBBED) called");
    return ORBIS_OK;
}

s32 PS4_SYSV_ABI sceAudioOutPtOpen() {
    LOG_ERROR(Lib_AudioOut, "(STUBBED) called");
    return ORBIS_OK;
}

s32 PS4_SYSV_ABI sceAudioOutSetConnections() {
    LOG_ERROR(Lib_AudioOut, "(STUBBED) called");
    return ORBIS_OK;
}

s32 PS4_SYSV_ABI sceAudioOutSetConnectionsForUser() {
    LOG_ERROR(Lib_AudioOut, "(STUBBED) called");
    return ORBIS_OK;
}

s32 PS4_SYSV_ABI sceAudioOutSetDevConnection() {
    LOG_ERROR(Lib_AudioOut, "(STUBBED) called");
    return ORBIS_OK;
}

s32 PS4_SYSV_ABI sceAudioOutSetHeadphoneOutMode() {
    LOG_ERROR(Lib_AudioOut, "(STUBBED) called");
    return ORBIS_OK;
}

s32 PS4_SYSV_ABI sceAudioOutSetJediJackVolume() {
    LOG_ERROR(Lib_AudioOut, "(STUBBED) called");
    return ORBIS_OK;
}

s32 PS4_SYSV_ABI sceAudioOutSetJediSpkVolume() {
    LOG_ERROR(Lib_AudioOut, "(STUBBED) called");
    return ORBIS_OK;
}

s32 PS4_SYSV_ABI sceAudioOutSetMainOutput() {
    LOG_ERROR(Lib_AudioOut, "(STUBBED) called");
    return ORBIS_OK;
}

s32 PS4_SYSV_ABI sceAudioOutSetMixLevelPadSpk() {
    LOG_ERROR(Lib_AudioOut, "(STUBBED) called");
    return ORBIS_OK;
}

s32 PS4_SYSV_ABI sceAudioOutSetMorpheusParam() {
    LOG_ERROR(Lib_AudioOut, "(STUBBED) called");
    return ORBIS_OK;
}

s32 PS4_SYSV_ABI sceAudioOutSetMorpheusWorkingMode() {
    LOG_ERROR(Lib_AudioOut, "(STUBBED) called");
    return ORBIS_OK;
}

s32 PS4_SYSV_ABI sceAudioOutSetPortConnections() {
    LOG_ERROR(Lib_AudioOut, "(STUBBED) called");
    return ORBIS_OK;
}

s32 PS4_SYSV_ABI sceAudioOutSetPortStatuses() {
    LOG_ERROR(Lib_AudioOut, "(STUBBED) called");
    return ORBIS_OK;
}

s32 PS4_SYSV_ABI sceAudioOutSetRecMode() {
    LOG_ERROR(Lib_AudioOut, "(STUBBED) called");
    return ORBIS_OK;
}

s32 PS4_SYSV_ABI sceAudioOutSetSparkParam() {
    LOG_ERROR(Lib_AudioOut, "(STUBBED) called");
    return ORBIS_OK;
}

s32 PS4_SYSV_ABI sceAudioOutSetUsbVolume() {
    LOG_ERROR(Lib_AudioOut, "(STUBBED) called");
    return ORBIS_OK;
}

s32 PS4_SYSV_ABI sceAudioOutSetVolume() {
    LOG_ERROR(Lib_AudioOut, "(STUBBED) called");
    return ORBIS_OK;
}

s32 PS4_SYSV_ABI sceAudioOutSetVolumeDown() {
    LOG_ERROR(Lib_AudioOut, "(STUBBED) called");
    return ORBIS_OK;
}

s32 PS4_SYSV_ABI sceAudioOutStartAuxBroadcast() {
    LOG_ERROR(Lib_AudioOut, "(STUBBED) called");
    return ORBIS_OK;
}

s32 PS4_SYSV_ABI sceAudioOutStartSharePlay() {
    LOG_ERROR(Lib_AudioOut, "(STUBBED) called");
    return ORBIS_OK;
}

s32 PS4_SYSV_ABI sceAudioOutStopAuxBroadcast() {
    LOG_ERROR(Lib_AudioOut, "(STUBBED) called");
    return ORBIS_OK;
}

s32 PS4_SYSV_ABI sceAudioOutStopSharePlay() {
    LOG_ERROR(Lib_AudioOut, "(STUBBED) called");
    return ORBIS_OK;
}

s32 PS4_SYSV_ABI sceAudioOutSuspendResume() {
    LOG_ERROR(Lib_AudioOut, "(STUBBED) called");
    return ORBIS_OK;
}

s32 PS4_SYSV_ABI sceAudioOutSysConfigureOutputMode() {
    LOG_ERROR(Lib_AudioOut, "(STUBBED) called");
    return ORBIS_OK;
}

s32 PS4_SYSV_ABI sceAudioOutSysGetHdmiMonitorInfo() {
    LOG_ERROR(Lib_AudioOut, "(STUBBED) called");
    return ORBIS_OK;
}

s32 PS4_SYSV_ABI sceAudioOutSysGetSystemInfo() {
    LOG_ERROR(Lib_AudioOut, "(STUBBED) called");
    return ORBIS_OK;
}

s32 PS4_SYSV_ABI sceAudioOutSysHdmiMonitorInfoIsSupportedAudioOutMode() {
    LOG_ERROR(Lib_AudioOut, "(STUBBED) called");
    return ORBIS_OK;
}

s32 PS4_SYSV_ABI sceAudioOutSystemControlGet() {
    LOG_ERROR(Lib_AudioOut, "(STUBBED) called");
    return ORBIS_OK;
}

s32 PS4_SYSV_ABI sceAudioOutSystemControlSet() {
    LOG_ERROR(Lib_AudioOut, "(STUBBED) called");
    return ORBIS_OK;
}

s32 PS4_SYSV_ABI sceAudioOutSparkControlSetEqCoef() {
    LOG_ERROR(Lib_AudioOut, "(STUBBED) called");
    return ORBIS_OK;
}

s32 PS4_SYSV_ABI sceAudioOutSetSystemDebugState() {
    LOG_ERROR(Lib_AudioOut, "(STUBBED) called");
    return ORBIS_OK;
}

void AdjustVol() {
    LOG_ERROR(Lib_AudioOut, "(STUBBED) called");
}

void RegisterLib(Core::Loader::SymbolsResolver* sym) {
    LIB_FUNCTION("cx2dYFbzIAg", "libSceAudioOutDeviceService", 1, "libSceAudioOut",
                 sceAudioOutDeviceIdOpen);
    LIB_FUNCTION("tKumjQSzhys", "libSceAudioDeviceControl", 1, "libSceAudioOut",
                 sceAudioDeviceControlGet);
    LIB_FUNCTION("5ChfcHOf3SM", "libSceAudioDeviceControl", 1, "libSceAudioOut",
                 sceAudioDeviceControlSet);
    LIB_FUNCTION("Iz9X7ISldhs", "libSceAudioOut", 1, "libSceAudioOut", sceAudioOutA3dControl);
    LIB_FUNCTION("9RVIoocOVAo", "libSceAudioOut", 1, "libSceAudioOut", sceAudioOutA3dExit);
    LIB_FUNCTION("n7KgxE8rOuE", "libSceAudioOut", 1, "libSceAudioOut", sceAudioOutA3dInit);
    LIB_FUNCTION("WBAO6-n0-4M", "libSceAudioOut", 1, "libSceAudioOut",
                 sceAudioOutAttachToApplicationByPid);
    LIB_FUNCTION("O3FM2WXIJaI", "libSceAudioOut", 1, "libSceAudioOut",
                 sceAudioOutChangeAppModuleState);
    LIB_FUNCTION("s1--uE9mBFw", "libSceAudioOut", 1, "libSceAudioOut", sceAudioOutClose);
    LIB_FUNCTION("ol4LbeTG8mc", "libSceAudioOut", 1, "libSceAudioOut",
                 sceAudioOutDetachFromApplicationByPid);
    LIB_FUNCTION("r1V9IFEE+Ts", "libSceAudioOut", 1, "libSceAudioOut",
                 sceAudioOutExConfigureOutputMode);
    LIB_FUNCTION("wZakRQsWGos", "libSceAudioOut", 1, "libSceAudioOut", sceAudioOutExGetSystemInfo);
    LIB_FUNCTION("xjjhT5uw08o", "libSceAudioOut", 1, "libSceAudioOut", sceAudioOutExPtClose);
    LIB_FUNCTION("DsST7TNsyfo", "libSceAudioOut", 1, "libSceAudioOut",
                 sceAudioOutExPtGetLastOutputTime);
    LIB_FUNCTION("4UlW3CSuCa4", "libSceAudioOut", 1, "libSceAudioOut", sceAudioOutExPtOpen);
    LIB_FUNCTION("Xcj8VTtnZw0", "libSceAudioOut", 1, "libSceAudioOut",
                 sceAudioOutExSystemInfoIsSupportedAudioOutExMode);
    LIB_FUNCTION("I3Fwcmkg5Po", "libSceAudioOut", 1, "libSceAudioOut",
                 sceAudioOutGetFocusEnablePid);
    LIB_FUNCTION("Y3lXfCFEWFY", "libSceAudioOut", 1, "libSceAudioOut",
                 sceAudioOutGetHandleStatusInfo);
    LIB_FUNCTION("-00OAutAw+c", "libSceAudioOut", 1, "libSceAudioOut", sceAudioOutGetInfo);
    LIB_FUNCTION("RqmKxBqB8B4", "libSceAudioOut", 1, "libSceAudioOut", sceAudioOutGetInfoOpenNum);
    LIB_FUNCTION("Ptlts326pds", "libSceAudioOut", 1, "libSceAudioOut",
                 sceAudioOutGetLastOutputTime);
    LIB_FUNCTION("GrQ9s4IrNaQ", "libSceAudioOut", 1, "libSceAudioOut", sceAudioOutGetPortState);
    LIB_FUNCTION("c7mVozxJkPU", "libSceAudioOut", 1, "libSceAudioOut",
                 sceAudioOutGetSimulatedBusUsableStatusByBusType);
    LIB_FUNCTION("pWmS7LajYlo", "libSceAudioOut", 1, "libSceAudioOut",
                 sceAudioOutGetSimulatedHandleStatusInfo);
    LIB_FUNCTION("oPLghhAWgMM", "libSceAudioOut", 1, "libSceAudioOut",
                 sceAudioOutGetSimulatedHandleStatusInfo2);
    LIB_FUNCTION("5+r7JYHpkXg", "libSceAudioOut", 1, "libSceAudioOut", sceAudioOutGetSparkVss);
    LIB_FUNCTION("R5hemoKKID8", "libSceAudioOut", 1, "libSceAudioOut", sceAudioOutGetSystemState);
    LIB_FUNCTION("JfEPXVxhFqA", "libSceAudioOut", 1, "libSceAudioOut", sceAudioOutInit);
    LIB_FUNCTION("n16Kdoxnvl0", "libSceAudioOut", 1, "libSceAudioOut",
                 sceAudioOutInitIpmiGetSession);
    LIB_FUNCTION("r+qKw+ueD+Q", "libSceAudioOut", 1, "libSceAudioOut",
                 sceAudioOutMasteringGetState);
    LIB_FUNCTION("xX4RLegarbg", "libSceAudioOut", 1, "libSceAudioOut", sceAudioOutMasteringInit);
    LIB_FUNCTION("4055yaUg3EY", "libSceAudioOut", 1, "libSceAudioOut",
                 sceAudioOutMasteringSetParam);
    LIB_FUNCTION("RVWtUgoif5o", "libSceAudioOut", 1, "libSceAudioOut", sceAudioOutMasteringTerm);
    LIB_FUNCTION("-LXhcGARw3k", "libSceAudioOut", 1, "libSceAudioOut", sceAudioOutMbusInit);
    LIB_FUNCTION("ekNvsT22rsY", "libSceAudioOut", 1, "libSceAudioOut", sceAudioOutOpen);
    LIB_FUNCTION("qLpSK75lXI4", "libSceAudioOut", 1, "libSceAudioOut", sceAudioOutOpenEx);
    LIB_FUNCTION("QOQtbeDqsT4", "libSceAudioOut", 1, "libSceAudioOut", sceAudioOutOutput);
    LIB_FUNCTION("w3PdaSTSwGE", "libSceAudioOut", 1, "libSceAudioOut", sceAudioOutOutputs);
    LIB_FUNCTION("MapHTgeogbk", "libSceAudioOut", 1, "libSceAudioOut", sceAudioOutPtClose);
    LIB_FUNCTION("YZaq+UKbriQ", "libSceAudioOut", 1, "libSceAudioOut",
                 sceAudioOutPtGetLastOutputTime);
    LIB_FUNCTION("xyT8IUCL3CI", "libSceAudioOut", 1, "libSceAudioOut", sceAudioOutPtOpen);
    LIB_FUNCTION("o4OLQQqqA90", "libSceAudioOut", 1, "libSceAudioOut", sceAudioOutSetConnections);
    LIB_FUNCTION("QHq2ylFOZ0k", "libSceAudioOut", 1, "libSceAudioOut",
                 sceAudioOutSetConnectionsForUser);
    LIB_FUNCTION("r9KGqGpwTpg", "libSceAudioOut", 1, "libSceAudioOut", sceAudioOutSetDevConnection);
    LIB_FUNCTION("08MKi2E-RcE", "libSceAudioOut", 1, "libSceAudioOut",
                 sceAudioOutSetHeadphoneOutMode);
    LIB_FUNCTION("18IVGrIQDU4", "libSceAudioOut", 1, "libSceAudioOut",
                 sceAudioOutSetJediJackVolume);
    LIB_FUNCTION("h0o+D4YYr1k", "libSceAudioOut", 1, "libSceAudioOut", sceAudioOutSetJediSpkVolume);
    LIB_FUNCTION("KI9cl22to7E", "libSceAudioOut", 1, "libSceAudioOut", sceAudioOutSetMainOutput);
    LIB_FUNCTION("wVwPU50pS1c", "libSceAudioOut", 1, "libSceAudioOut",
                 sceAudioOutSetMixLevelPadSpk);
    LIB_FUNCTION("eeRsbeGYe20", "libSceAudioOut", 1, "libSceAudioOut", sceAudioOutSetMorpheusParam);
    LIB_FUNCTION("IZrItPnflBM", "libSceAudioOut", 1, "libSceAudioOut",
                 sceAudioOutSetMorpheusWorkingMode);
    LIB_FUNCTION("Gy0ReOgXW00", "libSceAudioOut", 1, "libSceAudioOut",
                 sceAudioOutSetPortConnections);
    LIB_FUNCTION("oRBFflIrCg0", "libSceAudioOut", 1, "libSceAudioOut", sceAudioOutSetPortStatuses);
    LIB_FUNCTION("ae-IVPMSWjU", "libSceAudioOut", 1, "libSceAudioOut", sceAudioOutSetRecMode);
    LIB_FUNCTION("d3WL2uPE1eE", "libSceAudioOut", 1, "libSceAudioOut", sceAudioOutSetSparkParam);
    LIB_FUNCTION("X7Cfsiujm8Y", "libSceAudioOut", 1, "libSceAudioOut", sceAudioOutSetUsbVolume);
    LIB_FUNCTION("b+uAV89IlxE", "libSceAudioOut", 1, "libSceAudioOut", sceAudioOutSetVolume);
    LIB_FUNCTION("rho9DH-0ehs", "libSceAudioOut", 1, "libSceAudioOut", sceAudioOutSetVolumeDown);
    LIB_FUNCTION("I91P0HAPpjw", "libSceAudioOut", 1, "libSceAudioOut",
                 sceAudioOutStartAuxBroadcast);
    LIB_FUNCTION("uo+eoPzdQ-s", "libSceAudioOut", 1, "libSceAudioOut", sceAudioOutStartSharePlay);
    LIB_FUNCTION("AImiaYFrKdc", "libSceAudioOut", 1, "libSceAudioOut", sceAudioOutStopAuxBroadcast);
    LIB_FUNCTION("teCyKKZPjME", "libSceAudioOut", 1, "libSceAudioOut", sceAudioOutStopSharePlay);
    LIB_FUNCTION("95bdtHdNUic", "libSceAudioOut", 1, "libSceAudioOut", sceAudioOutSuspendResume);
    LIB_FUNCTION("oRJZnXxok-M", "libSceAudioOut", 1, "libSceAudioOut",
                 sceAudioOutSysConfigureOutputMode);
    LIB_FUNCTION("Tf9-yOJwF-A", "libSceAudioOut", 1, "libSceAudioOut",
                 sceAudioOutSysGetHdmiMonitorInfo);
    LIB_FUNCTION("y2-hP-KoTMI", "libSceAudioOut", 1, "libSceAudioOut", sceAudioOutSysGetSystemInfo);
    LIB_FUNCTION("YV+bnMvMfYg", "libSceAudioOut", 1, "libSceAudioOut",
                 sceAudioOutSysHdmiMonitorInfoIsSupportedAudioOutMode);
    LIB_FUNCTION("JEHhANREcLs", "libSceAudioOut", 1, "libSceAudioOut", sceAudioOutSystemControlGet);
    LIB_FUNCTION("9CHWVv6r3Dg", "libSceAudioOut", 1, "libSceAudioOut", sceAudioOutSystemControlSet);
    LIB_FUNCTION("Mt7JB3lOyJk", "libSceAudioOutSparkControl", 1, "libSceAudioOut",
                 sceAudioOutSparkControlSetEqCoef);
    LIB_FUNCTION("7UsdDOEvjlk", "libSceDbgAudioOut", 1, "libSceAudioOut",
                 sceAudioOutSetSystemDebugState);
};

} // namespace Libraries::AudioOut