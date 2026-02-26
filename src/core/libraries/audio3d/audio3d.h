// SPDX-FileCopyrightText: Copyright 2025-2026 shadPS4 Emulator Project
// SPDX-License-Identifier: GPL-2.0-or-later

#pragma once

#include <algorithm>
#include <mutex>
#include <optional>
#include <vector>
#include <queue>

#include "common/types.h"
#include "core/libraries/audio/audioout.h"

namespace Core::Loader {
class SymbolsResolver;
}

namespace Libraries::Audio3d {

constexpr int ORBIS_AUDIO3D_OBJECT_INVALID = 0xFFFFFFFF;

enum class OrbisAudio3dRate : u32 {
    ORBIS_AUDIO3D_RATE_48000 = 0,
};

enum class OrbisAudio3dBufferMode : u32 {
    ORBIS_AUDIO3D_BUFFER_NO_ADVANCE = 0,
    ORBIS_AUDIO3D_BUFFER_ADVANCE_NO_PUSH = 1,
    ORBIS_AUDIO3D_BUFFER_ADVANCE_AND_PUSH = 2,
};

struct OrbisAudio3dOpenParameters {
    u64 size_this;
    u32 granularity;
    OrbisAudio3dRate rate;
    u32 max_objects;
    u32 queue_depth;
    OrbisAudio3dBufferMode buffer_mode;
    int : 32;
    u32 num_beds;
};

enum class OrbisAudio3dFormat : u32 {
    ORBIS_AUDIO3D_FORMAT_S16 = 0,
    ORBIS_AUDIO3D_FORMAT_FLOAT = 1,
};

enum class OrbisAudio3dOutputRoute : u32 {
    ORBIS_AUDIO3D_OUTPUT_BOTH = 0,
    ORBIS_AUDIO3D_OUTPUT_HMU_ONLY = 1,
    ORBIS_AUDIO3D_OUTPUT_TV_ONLY = 2,
};

enum class OrbisAudio3dBlocking : u32 {
    ORBIS_AUDIO3D_BLOCKING_ASYNC = 0,
    ORBIS_AUDIO3D_BLOCKING_SYNC = 1,
};

// World-space position (right-handed: +X right, +Y up, listener faces -Z).
// Per SDK docs: position affects perceived direction only, not volume.
// Distance attenuation must be applied by the game via the GAIN attribute.
struct OrbisAudio3dPosition {
    float fX;
    float fY;
    float fZ;
};

// Per SDK defines table: when set to any value other than NONE, the values of
// priority, position, gain, and spread attributes are all ignored and the object
// is routed dry to the output (no spatialization). Used for UI audio and music.
enum class OrbisAudio3dPassthrough : u32 {
    ORBIS_AUDIO3D_PASSTHROUGH_NONE = 0,
    ORBIS_AUDIO3D_PASSTHROUGH_LEFT = 1,
    ORBIS_AUDIO3D_PASSTHROUGH_RIGHT = 2,
    ORBIS_AUDIO3D_PASSTHROUGH_STEREO = 3,
};

struct OrbisAudio3dPcm {
    OrbisAudio3dFormat format;
    void* sample_buffer;
    u32 num_samples;
};

enum class OrbisAudio3dAttributeId : u32 {
    ORBIS_AUDIO3D_ATTRIBUTE_PCM = 1,
    ORBIS_AUDIO3D_ATTRIBUTE_POSITION = 2,
    ORBIS_AUDIO3D_ATTRIBUTE_GAIN = 3,
    ORBIS_AUDIO3D_ATTRIBUTE_SPREAD = 4,
    ORBIS_AUDIO3D_ATTRIBUTE_PRIORITY = 5,
    ORBIS_AUDIO3D_ATTRIBUTE_PASSTHROUGH = 6,
    ORBIS_AUDIO3D_ATTRIBUTE_AMBISONICS = 7,
    ORBIS_AUDIO3D_ATTRIBUTE_APPLICATION_SPECIFIC = 8,
    ORBIS_AUDIO3D_ATTRIBUTE_RESET_STATE = 9, // Resets all attributes and internal state
    ORBIS_AUDIO3D_ATTRIBUTE_RESTRICTED = 10,
    ORBIS_AUDIO3D_ATTRIBUTE_OUTPUT_ROUTE = 11,
};

using OrbisAudio3dPortId = u32;
using OrbisAudio3dObjectId = u32;

// B-format ambisonics channel assignment in ACN (Ambisonics Channel Number) ordering
// with SN3D normalization. Each Audio3D object carrying ambisonics PCM stores one channel.
// When set to any value other than NONE, position/spread/priority are all ignored per SDK docs.
enum class OrbisAudio3dAmbisonics : u32 {
    ORBIS_AUDIO3D_AMBISONICS_NONE = 0,
    // First order (4 channels)
    ORBIS_AUDIO3D_AMBISONICS_ACN_0 = 1, // W — omnidirectional
    ORBIS_AUDIO3D_AMBISONICS_ACN_1 = 2, // Y — left/right  (+Y = left in listener frame)
    ORBIS_AUDIO3D_AMBISONICS_ACN_2 = 3, // Z — up/down
    ORBIS_AUDIO3D_AMBISONICS_ACN_3 = 4, // X — front/back (+X = right in listener frame)
    // Second order (9 channels total, ACN 4-8)
    ORBIS_AUDIO3D_AMBISONICS_ACN_4 = 5,
    ORBIS_AUDIO3D_AMBISONICS_ACN_5 = 6,
    ORBIS_AUDIO3D_AMBISONICS_ACN_6 = 7,
    ORBIS_AUDIO3D_AMBISONICS_ACN_7 = 8,
    ORBIS_AUDIO3D_AMBISONICS_ACN_8 = 9,
    // Third order (16 channels total, ACN 9-15)
    ORBIS_AUDIO3D_AMBISONICS_ACN_9 = 10,
    ORBIS_AUDIO3D_AMBISONICS_ACN_10 = 11,
    ORBIS_AUDIO3D_AMBISONICS_ACN_11 = 12,
    ORBIS_AUDIO3D_AMBISONICS_ACN_12 = 13,
    ORBIS_AUDIO3D_AMBISONICS_ACN_13 = 14,
    ORBIS_AUDIO3D_AMBISONICS_ACN_14 = 15,
    ORBIS_AUDIO3D_AMBISONICS_ACN_15 = 16,
};

// Sentinel values: ((SceAudio3dPortId)-1) per SDK defines table.
// Guaranteed never returned by PortOpen on success.
constexpr int ORBIS_AUDIO3D_PORT_INVALID = 0xFFFFFFFF;

// Port-level attribute IDs (used with sceAudio3dPortSetAttribute).
enum class OrbisAudio3dPortAttributeId : u32 {
    ORBIS_AUDIO3D_PORT_ATTRIBUTE_LATE_REVERB_LEVEL = 1,           // float [0, 1]
    ORBIS_AUDIO3D_PORT_ATTRIBUTE_DOWNMIX_SPREAD_RADIUS = 2,       // float, metres
    ORBIS_AUDIO3D_PORT_ATTRIBUTE_DOWNMIX_SPREAD_HEIGHT_AWARE = 3, // bool
};

struct OrbisAudio3dAttribute {
    OrbisAudio3dAttributeId attribute_id;
    int : 32;
    void* value;
    u64 value_size;
};

struct AudioData {
    u8* sample_buffer;
    u32 num_samples;
    u32 num_channels{1}; // channels in sample_buffer
    OrbisAudio3dFormat format{
        OrbisAudio3dFormat::ORBIS_AUDIO3D_FORMAT_S16}; // format of sample_buffer
};

// Persistent per-object state: attributes set via sceAudio3dObjectSetAttribute survive
// across multiple render frames until explicitly changed or the object is unreserved.
struct ObjectState {
    std::deque<AudioData> pcm_queue;
    // Raw persistent attributes keyed by attribute ID (position, gain, spread, priority…).
    // Stored as byte blobs so we can forward them once spatialization is implemented.
    std::unordered_map<u32, std::vector<u8>> persistent_attributes;
    // Set when PCM is submitted this frame; cleared by PortAdvance/PortFlush.
    // Prevents RESET_STATE from being called after PCM in the same frame (NOT_READY).
    bool pcm_submitted_this_frame{false};
};

// Schroeder reverb (Freeverb topology): 8 parallel comb filters feeding 4 series allpass filters,
// one network per stereo channel. Provides a simple but convincing late reverberation tail.
// All delay lengths are prime-ish and slightly offset between L/R to decorrelate channels.
// Sample rate assumed: 48000 Hz (matching ORBIS_AUDIO3D_RATE_48000).
struct SchroederReverb {
    // ----- Comb filter -----
    // y[n] = x[n - delay] + feedback * y[n - delay]   (feedback comb / IIR comb)
    struct CombFilter {
        std::vector<float> buf;
        u32 pos{0};
        float feedback{0.0f};
        float store{0.0f}; // damping low-pass state

        void init(u32 delay_samples, float fb) {
            buf.assign(delay_samples, 0.0f);
            feedback = fb;
            pos = 0;
            store = 0.0f;
        }

        float process(float input, float damp) {
            float output = buf[pos];
            store = output * (1.0f - damp) + store * damp; // one-pole LPF in feedback path
            buf[pos] = input + store * feedback;
            pos = (pos + 1 < static_cast<u32>(buf.size())) ? pos + 1 : 0;
            return output;
        }
    };

    // ----- Allpass filter -----
    // y[n] = -x[n] + x[n - delay] + g * y[n - delay]   (Schroeder allpass)
    struct AllpassFilter {
        std::vector<float> buf;
        u32 pos{0};

        void init(u32 delay_samples) {
            buf.assign(delay_samples, 0.0f);
            pos = 0;
        }

        float process(float input) {
            constexpr float kGain = 0.5f;
            float bufout = buf[pos];
            buf[pos] = input + bufout * kGain;
            pos = (pos + 1 < static_cast<u32>(buf.size())) ? pos + 1 : 0;
            return bufout - input;
        }
    };

    // Freeverb uses 8 comb + 4 allpass per channel at 44100 Hz; we scale to 48000 Hz.
    // Delay lengths (in samples at 48k) — L channel, R channel offset by +23 samples.
    static constexpr u32 kNumCombs = 8;
    static constexpr u32 kNumAllpass = 4;
    static constexpr u32 kStereoSpread = 23;

    static constexpr u32 kCombDelays[kNumCombs] = {1116, 1188, 1277, 1356, 1422, 1491, 1557, 1617};
    static constexpr u32 kAllpassDelays[kNumAllpass] = {556, 441, 341, 225};

    CombFilter combs_l[kNumCombs], combs_r[kNumCombs];
    AllpassFilter aps_l[kNumAllpass], aps_r[kNumAllpass];

    float level{0.0f};     // LATE_REVERB_LEVEL [0, 1] — 0 means reverb is bypassed entirely
    float feedback{0.84f}; // room size proxy
    float damp{0.2f};      // high-frequency damping in comb feedback path

    void init() {
        for (u32 i = 0; i < kNumCombs; i++) {
            combs_l[i].init(kCombDelays[i], feedback);
            combs_r[i].init(kCombDelays[i] + kStereoSpread, feedback);
        }
        for (u32 i = 0; i < kNumAllpass; i++) {
            aps_l[i].init(kAllpassDelays[i]);
            aps_r[i].init(kAllpassDelays[i] + kStereoSpread);
        }
        level = 0.0f;
    }

    // Process one stereo sample pair. Returns wet signal; caller mixes dry + wet.
    void processSample(float in_l, float in_r, float& out_l, float& out_r) {
        // Sum mono input into both comb networks (standard Freeverb input routing).
        const float mono = (in_l + in_r) * 0.5f;

        float sum_l = 0.0f, sum_r = 0.0f;
        for (u32 i = 0; i < kNumCombs; i++) {
            sum_l += combs_l[i].process(mono, damp);
            sum_r += combs_r[i].process(mono, damp);
        }

        // Series allpass diffusion.
        for (u32 i = 0; i < kNumAllpass; i++) {
            sum_l = aps_l[i].process(sum_l);
            sum_r = aps_r[i].process(sum_r);
        }

        out_l = sum_l * level;
        out_r = sum_r * level;
    }
};

struct Port {
    // Guards all mutable fields accessed from multiple threads.
    // On real hardware the Audio3d library is internally thread-safe: games are documented
    // to call ObjectReserve/Unreserve from one thread while rendering audio from another.
    // recursive_mutex allows PortFlush to call PortAdvance internally without deadlocking.
    mutable std::recursive_mutex mutex;
    OrbisAudio3dOpenParameters parameters{};
    // Internal handle used by the advance/push model (sceAudio3dPortPush).
    // Opened lazily on the first sceAudio3dPortPush call.
    s32 audio_out_handle{-1};
    // Handles explicitly opened by the game via sceAudio3dAudioOutOpen.
    // The game owns their lifetime and closes them with sceAudio3dAudioOutClose.
    std::vector<s32> audioout_handles;
    // Reserved objects and their state.
    std::unordered_map<OrbisAudio3dObjectId, ObjectState> objects;
    // Monotonically increasing counter for generating unique object IDs within this port.
    // Kept here (not as a global static) so IDs reset when the port is closed and reopened,
    // and so that it's protected by port.mutex automatically.
    OrbisAudio3dObjectId next_object_id{0};
    // Bed audio queue (from sceAudio3dBedWrite).
    std::deque<AudioData> bed_queue;
    // Mixed stereo frames ready to be consumed by sceAudio3dPortPush.
    // Supports queue_depth > 1 (double/triple buffering).
    std::deque<AudioData> mixed_queue;
    // Late reverb applied to the final mix. Level controlled by sceAudio3dPortSetAttribute
    // with LATE_REVERB_LEVEL. Bypassed entirely when level == 0.
    SchroederReverb reverb;
};

struct Audio3dState {
    std::unordered_map<OrbisAudio3dPortId, Port> ports;
};

s32 PS4_SYSV_ABI sceAudio3dAudioOutClose(s32 handle);
s32 PS4_SYSV_ABI sceAudio3dAudioOutOpen(OrbisAudio3dPortId port_id,
                                        Libraries::UserService::OrbisUserServiceUserId user_id,
                                        s32 type, s32 index, u32 len, u32 freq,
                                        AudioOut::OrbisAudioOutParamExtendedInformation param);
s32 PS4_SYSV_ABI sceAudio3dAudioOutOutput(s32 handle, void* ptr);
s32 PS4_SYSV_ABI sceAudio3dAudioOutOutputs(AudioOut::OrbisAudioOutOutputParam* param, u32 num);
s32 PS4_SYSV_ABI sceAudio3dBedWrite(OrbisAudio3dPortId port_id, u32 num_channels,
                                    OrbisAudio3dFormat format, void* buffer, u32 num_samples);
s32 PS4_SYSV_ABI sceAudio3dBedWrite2(OrbisAudio3dPortId port_id, u32 num_channels,
                                     OrbisAudio3dFormat format, void* buffer, u32 num_samples,
                                     OrbisAudio3dOutputRoute output_route, bool restricted);
s32 PS4_SYSV_ABI sceAudio3dCreateSpeakerArray();
s32 PS4_SYSV_ABI sceAudio3dDeleteSpeakerArray();
s32 PS4_SYSV_ABI sceAudio3dGetDefaultOpenParameters(OrbisAudio3dOpenParameters* params);
s32 PS4_SYSV_ABI sceAudio3dGetSpeakerArrayMemorySize();
s32 PS4_SYSV_ABI sceAudio3dGetSpeakerArrayMixCoefficients();
s32 PS4_SYSV_ABI sceAudio3dGetSpeakerArrayMixCoefficients2();
s32 PS4_SYSV_ABI sceAudio3dInitialize(s64 reserved);
s32 PS4_SYSV_ABI sceAudio3dObjectReserve(OrbisAudio3dPortId port_id,
                                         OrbisAudio3dObjectId* object_id);
s32 PS4_SYSV_ABI sceAudio3dObjectSetAttribute(OrbisAudio3dPortId port_id,
                                              OrbisAudio3dObjectId object_id,
                                              OrbisAudio3dAttributeId attribute_id,
                                              const void* attribute, u64 attribute_size);
s32 PS4_SYSV_ABI sceAudio3dObjectSetAttributes(OrbisAudio3dPortId port_id,
                                               OrbisAudio3dObjectId object_id, u64 num_attributes,
                                               const OrbisAudio3dAttribute* attribute_array);
s32 PS4_SYSV_ABI sceAudio3dObjectUnreserve(OrbisAudio3dPortId port_id,
                                           OrbisAudio3dObjectId object_id);
s32 PS4_SYSV_ABI sceAudio3dPortAdvance(OrbisAudio3dPortId port_id);
s32 PS4_SYSV_ABI sceAudio3dPortClose(OrbisAudio3dPortId port_id);
s32 PS4_SYSV_ABI sceAudio3dPortCreate();
s32 PS4_SYSV_ABI sceAudio3dPortDestroy();
s32 PS4_SYSV_ABI sceAudio3dPortFlush(OrbisAudio3dPortId port_id);
s32 PS4_SYSV_ABI sceAudio3dPortFreeState();
s32 PS4_SYSV_ABI sceAudio3dPortGetAttributesSupported();
s32 PS4_SYSV_ABI sceAudio3dPortGetList();
s32 PS4_SYSV_ABI sceAudio3dPortGetParameters();
s32 PS4_SYSV_ABI sceAudio3dPortGetQueueLevel(OrbisAudio3dPortId port_id, u32* queue_level,
                                             u32* queue_available);
s32 PS4_SYSV_ABI sceAudio3dPortGetState();
s32 PS4_SYSV_ABI sceAudio3dPortGetStatus();
s32 PS4_SYSV_ABI sceAudio3dPortOpen(Libraries::UserService::OrbisUserServiceUserId user_id,
                                    const OrbisAudio3dOpenParameters* parameters,
                                    OrbisAudio3dPortId* port_id);
s32 PS4_SYSV_ABI sceAudio3dPortPush(OrbisAudio3dPortId port_id, OrbisAudio3dBlocking blocking);
s32 PS4_SYSV_ABI sceAudio3dPortQueryDebug();
s32 PS4_SYSV_ABI sceAudio3dPortSetAttribute(OrbisAudio3dPortId port_id,
                                            OrbisAudio3dAttributeId attribute_id, void* attribute,
                                            u64 attribute_size);
s32 PS4_SYSV_ABI sceAudio3dReportRegisterHandler();
s32 PS4_SYSV_ABI sceAudio3dReportUnregisterHandler();
s32 PS4_SYSV_ABI sceAudio3dSetGpuRenderer();
s32 PS4_SYSV_ABI sceAudio3dStrError();
s32 PS4_SYSV_ABI sceAudio3dTerminate();

void RegisterLib(Core::Loader::SymbolsResolver* sym);
} // namespace Libraries::Audio3d
