// SPDX-FileCopyrightText: Copyright 2025 shadPS4 Emulator Project
// SPDX-License-Identifier: GPL-2.0-or-later

#include <algorithm>
#include <cmath>
#include <cstdlib>
#include <cstring>
#include <vector>
#include <magic_enum/magic_enum.hpp>

#include "common/assert.h"
#include "common/logging/log.h"
#include "core/libraries/audio/audioout.h"
#include "core/libraries/audio/audioout_error.h"
#include "core/libraries/audio3d/audio3d.h"
#include "core/libraries/audio3d/audio3d_error.h"
#include "core/libraries/error_codes.h"
#include "core/libraries/libs.h"

namespace Libraries::Audio3d {

static constexpr u32 AUDIO3D_SAMPLE_RATE = 48000;

static constexpr AudioOut::OrbisAudioOutParamFormat AUDIO3D_OUTPUT_FORMAT =
    AudioOut::OrbisAudioOutParamFormat::S16Stereo;
static constexpr u32 AUDIO3D_OUTPUT_NUM_CHANNELS = 2;

static std::unique_ptr<Audio3dState> state;

s32 PS4_SYSV_ABI sceAudio3dAudioOutClose(const s32 handle) {
    LOG_INFO(Lib_Audio3d, "called, handle = {}", handle);

    // Remove from any port that was tracking this handle.
    if (state) {
        for (auto& [port_id, port] : state->ports) {
            std::scoped_lock lock{port.mutex};
            auto& handles = port.audioout_handles;
            handles.erase(std::remove(handles.begin(), handles.end(), handle), handles.end());
        }
    }

    return AudioOut::sceAudioOutClose(handle);
}

s32 PS4_SYSV_ABI sceAudio3dAudioOutOpen(
    const OrbisAudio3dPortId port_id, const Libraries::UserService::OrbisUserServiceUserId user_id,
    s32 type, const s32 index, const u32 len, const u32 freq,
    const AudioOut::OrbisAudioOutParamExtendedInformation param) {
    LOG_INFO(Lib_Audio3d,
             "called, port_id = {}, user_id = {}, type = {}, index = {}, len = {}, freq = {}",
             port_id, user_id, type, index, len, freq);

    if (!state->ports.contains(port_id)) {
        LOG_ERROR(Lib_Audio3d, "!state->ports.contains(port_id)");
        return ORBIS_AUDIO3D_ERROR_INVALID_PORT;
    }

    std::scoped_lock lock{state->ports[port_id].mutex};
    if (len != state->ports[port_id].parameters.granularity) {
        LOG_ERROR(Lib_Audio3d, "len != state->ports[port_id].parameters.granularity");
        return ORBIS_AUDIO3D_ERROR_INVALID_PARAMETER;
    }

    const s32 handle = sceAudioOutOpen(user_id, static_cast<AudioOut::OrbisAudioOutPort>(type),
                                       index, len, freq, param);
    if (handle < 0) {
        return handle;
    }

    // Track this handle in the port so sceAudio3dPortFlush can use it for sync.
    state->ports[port_id].audioout_handles.push_back(handle);
    return handle;
}

s32 PS4_SYSV_ABI sceAudio3dAudioOutOutput(const s32 handle, void* ptr) {
    LOG_DEBUG(Lib_Audio3d, "called, handle = {}, ptr = {}", handle, ptr);

    if (!ptr) {
        LOG_ERROR(Lib_Audio3d, "!ptr");
        return ORBIS_AUDIO3D_ERROR_INVALID_PARAMETER;
    }

    if (handle < 0 || (handle & 0xFFFF) > 25) {
        LOG_ERROR(Lib_Audio3d, "handle < 0 || (handle & 0xFFFF) > 25");
        return ORBIS_AUDIO3D_ERROR_INVALID_PORT;
    }

    return AudioOut::sceAudioOutOutput(handle, ptr);
}

s32 PS4_SYSV_ABI sceAudio3dAudioOutOutputs(AudioOut::OrbisAudioOutOutputParam* param,
                                           const u32 num) {
    LOG_DEBUG(Lib_Audio3d, "called, param = {}, num = {}", static_cast<void*>(param), num);

    if (!param || !num) {
        LOG_ERROR(Lib_Audio3d, "!param || !num");
        return ORBIS_AUDIO3D_ERROR_INVALID_PARAMETER;
    }

    return AudioOut::sceAudioOutOutputs(param, num);
}

// Store a raw copy of the PCM data in the queue. Format conversion to stereo S16 happens
// later at mix time in PortAdvance, so that float ambisonics channels are summed in floating
// point before being quantised — avoiding precision loss from 16 concurrent S16 conversions.
// Computes left/right gain factors for stereo panning from a 3D position and spread.
//
// The PS4 SDK documents that POSITION does not affect volume — distance attenuation
// must be applied by the game via the GAIN attribute. POSITION only affects the
// perceived direction (azimuth) of the sound, and SPREAD controls stereo width.
//
// Implementation:
//   1. Compute azimuth from listener to source in the horizontal plane (atan2 of X/Z).
//   2. Compute distance and derive an implicit distance spread addition.
//   3. Combine game-provided spread with distance spread (clamped to 2π).
//   4. Apply equal-power (sin/cos) stereo panning from the azimuth, clamped to ±90°.
//   5. Apply rear attenuation for sounds behind the listener (|azimuth| > 90°).
//   6. Blend toward center by the effective spread.
//
// Elevation (fY) is intentionally ignored — we output stereo, not binaural HRTF,
// so vertical positioning cannot be faithfully reproduced.
//
// Rear attenuation rationale:
//   The human pinna (outer ear) filters sounds arriving from behind, making them
//   perceptibly quieter and duller than frontal sounds. Without HRTF filtering we
//   cannot reproduce the timbral change, but we can approximate the loudness drop.
//   The attenuation ramps from 0 dB at ±90° (side) to -6 dB at ±180° (behind).
//
// Distance-dependent spread rationale:
//   A sound source very close to the listener (< 1 m) has a sharp, precise stereo
//   position. As distance increases, inter-aural time and level differences shrink,
//   making the source progressively harder to localize — it perceptually widens and
//   drifts toward center. We model this as an additive spread term that starts at 0
//   for sources within kSpreadNearDistance and reaches 2π (fully centered) at
//   kSpreadFarDistance. The ramp uses a square-root curve so the widening is
//   perceptually more uniform (loudness perception is roughly logarithmic, so a
//   sqrt ramp in linear distance gives a more even subjective rate of change).
//   These thresholds are tunable and represent typical in-game world scales.
static void SpatialPan(const OrbisAudio3dPosition& pos, const float spread, float& out_left,
                       float& out_right) {
    constexpr float kHalfPi = 1.5707963267948966f;
    constexpr float k2Pi = 6.2831853071795865f;

    // Distance from listener origin in world units (Y included for full 3D distance).
    const float dist = std::sqrt(pos.fX * pos.fX + pos.fY * pos.fY + pos.fZ * pos.fZ);

    // Distance-dependent spread: sounds far from the listener widen toward center.
    // kSpreadNearDistance — within this radius the implicit spread addition is 0.
    // kSpreadFarDistance  — at this distance and beyond the implicit spread is 2π (fully centered).
    // sqrt ramp gives a more perceptually uniform rate of widening than a linear ramp.
    constexpr float kSpreadNearDistance = 1.0f; // metres (or game units)
    constexpr float kSpreadFarDistance = 50.0f; // metres (or game units)
    float distance_spread = 0.0f;
    if (dist > kSpreadNearDistance) {
        const float ramp = std::min(
            (dist - kSpreadNearDistance) / (kSpreadFarDistance - kSpreadNearDistance), 1.0f);
        distance_spread = k2Pi * std::sqrt(ramp);
    }

    // Effective spread is the sum of game-set spread and distance spread, clamped to 2π.
    // The game's explicit spread value still takes effect at close range.
    const float effective_spread = std::min(spread + distance_spread, k2Pi);

    // Azimuth: angle in the horizontal plane from listener's forward axis (-Z).
    // atan2(X, -Z) gives 0 straight ahead, +π/2 hard right, -π/2 hard left,
    // ±π directly behind.
    const float azimuth = std::atan2(pos.fX, -pos.fZ);
    const float abs_azimuth = std::abs(azimuth);

    // Equal-power pan: clamp azimuth to ±90° for the pan law.
    // Sounds beyond 90° stay hard-panned to the nearer side.
    const float pan_angle = std::clamp(azimuth, -kHalfPi, kHalfPi);
    const float t = (pan_angle + kHalfPi) * 0.5f; // remap [-π/2, π/2] → [0, π/2]
    const float pan_left = std::cos(t);           // 1.0 at hard left, 0.0 at hard right
    const float pan_right = std::sin(t);          // 0.0 at hard left, 1.0 at hard right

    // Rear attenuation: ramps from 1.0 at ±90° (side) to 0.5 at ±180° (behind).
    float rear_gain = 1.0f;
    if (abs_azimuth > kHalfPi) {
        const float rear_t = (abs_azimuth - kHalfPi) / kHalfPi; // [0, 1]
        rear_gain = 1.0f - 0.5f * rear_t;                       // [1.0, 0.5]
    }

    // Spread blends between the panned signal and a centered (equal L/R) signal.
    // effective_spread=0 → pure pan; effective_spread=2π → fully centered.
    constexpr float kCenter = 0.7071067811865476f; // 1/√2 — equal-power center
    const float spread_t = std::clamp(effective_spread / k2Pi, 0.0f, 1.0f);
    out_left = (pan_left + (kCenter - pan_left) * spread_t) * rear_gain;
    out_right = (pan_right + (kCenter - pan_right) * spread_t) * rear_gain;
}

// Validates type-specific constraints on object attribute values.
// Returns ORBIS_OK if valid, ORBIS_AUDIO3D_ERROR_INVALID_PARAMETER otherwise.
// Per SDK defines table: GAIN must be a positive float; SPREAD must be in [0, 2*PI].
static s32 ValidateObjectAttributeValue(const OrbisAudio3dAttributeId attribute_id,
                                        const void* value, const u64 size) {
    if (!value || !size) {
        return ORBIS_AUDIO3D_ERROR_INVALID_PARAMETER;
    }
    switch (attribute_id) {
    case OrbisAudio3dAttributeId::ORBIS_AUDIO3D_ATTRIBUTE_GAIN: {
        if (size < sizeof(float)) {
            return ORBIS_AUDIO3D_ERROR_INVALID_PARAMETER;
        }
        float gain;
        std::memcpy(&gain, value, sizeof(float));
        if (gain < 0.0f) {
            LOG_ERROR(Lib_Audio3d, "GAIN must be a positive float, got {}", gain);
            return ORBIS_AUDIO3D_ERROR_INVALID_PARAMETER;
        }
        break;
    }
    case OrbisAudio3dAttributeId::ORBIS_AUDIO3D_ATTRIBUTE_SPREAD: {
        if (size < sizeof(float)) {
            return ORBIS_AUDIO3D_ERROR_INVALID_PARAMETER;
        }
        float spread;
        std::memcpy(&spread, value, sizeof(float));
        constexpr float k2Pi = 6.28318530717958647692f;
        if (spread < 0.0f || spread > k2Pi) {
            LOG_ERROR(Lib_Audio3d, "SPREAD must be in [0, 2*PI], got {}", spread);
            return ORBIS_AUDIO3D_ERROR_INVALID_PARAMETER;
        }
        break;
    }
    default:
        break;
    }
    return ORBIS_OK;
}

// Per SDK docs: if the game submits fewer samples than granularity, the remainder is padded
// with silence. If it submits too many, the extra samples are dropped.
static s32 ConvertAndEnqueue(std::deque<AudioData>& queue, const OrbisAudio3dPcm& pcm,
                             const u32 num_channels, const u32 granularity) {
    if (!pcm.sample_buffer || !pcm.num_samples) {
        return ORBIS_AUDIO3D_ERROR_INVALID_PARAMETER;
    }

    const u32 bytes_per_sample =
        (pcm.format == OrbisAudio3dFormat::ORBIS_AUDIO3D_FORMAT_S16) ? sizeof(s16) : sizeof(float);

    // Always allocate exactly granularity samples (zeroed = silence for padding).
    const u32 dst_bytes = granularity * num_channels * bytes_per_sample;
    u8* copy = static_cast<u8*>(std::calloc(1, dst_bytes));
    if (!copy) {
        return ORBIS_AUDIO3D_ERROR_OUT_OF_MEMORY;
    }

    // Copy min(provided, granularity) samples — extra are dropped, shortage stays zero.
    const u32 samples_to_copy = std::min(pcm.num_samples, granularity);
    std::memcpy(copy, pcm.sample_buffer, samples_to_copy * num_channels * bytes_per_sample);

    queue.emplace_back(AudioData{
        .sample_buffer = copy,
        .num_samples = granularity,
        .num_channels = num_channels,
        .format = pcm.format,
    });
    return ORBIS_OK;
}

s32 PS4_SYSV_ABI sceAudio3dBedWrite(const OrbisAudio3dPortId port_id, const u32 num_channels,
                                    const OrbisAudio3dFormat format, void* buffer,
                                    const u32 num_samples) {
    return sceAudio3dBedWrite2(port_id, num_channels, format, buffer, num_samples,
                               OrbisAudio3dOutputRoute::ORBIS_AUDIO3D_OUTPUT_BOTH, false);
}

s32 PS4_SYSV_ABI sceAudio3dBedWrite2(const OrbisAudio3dPortId port_id, const u32 num_channels,
                                     const OrbisAudio3dFormat format, void* buffer,
                                     const u32 num_samples,
                                     const OrbisAudio3dOutputRoute output_route,
                                     const bool restricted) {
    LOG_DEBUG(
        Lib_Audio3d,
        "called, port_id = {}, num_channels = {}, format = {}, num_samples = {}, output_route "
        "= {}, restricted = {}",
        port_id, num_channels, magic_enum::enum_name(format), num_samples,
        magic_enum::enum_name(output_route), restricted);

    if (!state->ports.contains(port_id)) {
        LOG_ERROR(Lib_Audio3d, "!state->ports.contains(port_id)");
        return ORBIS_AUDIO3D_ERROR_INVALID_PORT;
    }

    if (output_route != OrbisAudio3dOutputRoute::ORBIS_AUDIO3D_OUTPUT_BOTH &&
        output_route != OrbisAudio3dOutputRoute::ORBIS_AUDIO3D_OUTPUT_HMU_ONLY &&
        output_route != OrbisAudio3dOutputRoute::ORBIS_AUDIO3D_OUTPUT_TV_ONLY) {
        LOG_ERROR(Lib_Audio3d, "invalid output_route {}", static_cast<u32>(output_route));
        return ORBIS_AUDIO3D_ERROR_INVALID_PARAMETER;
    }

    if (format > OrbisAudio3dFormat::ORBIS_AUDIO3D_FORMAT_FLOAT) {
        LOG_ERROR(Lib_Audio3d, "format > ORBIS_AUDIO3D_FORMAT_FLOAT");
        return ORBIS_AUDIO3D_ERROR_INVALID_PARAMETER;
    }

    if (num_channels != 2 && num_channels != 6 && num_channels != 8) {
        LOG_ERROR(Lib_Audio3d, "num_channels must be 2, 6, or 8");
        return ORBIS_AUDIO3D_ERROR_INVALID_PARAMETER;
    }

    if (!buffer || !num_samples) {
        LOG_ERROR(Lib_Audio3d, "!buffer || !num_samples");
        return ORBIS_AUDIO3D_ERROR_INVALID_PARAMETER;
    }

    if (format == OrbisAudio3dFormat::ORBIS_AUDIO3D_FORMAT_FLOAT) {
        if ((reinterpret_cast<uintptr_t>(buffer) & 3) != 0) {
            LOG_ERROR(Lib_Audio3d, "buffer & 3 != 0");
            return ORBIS_AUDIO3D_ERROR_INVALID_PARAMETER;
        }
    } else if (format == OrbisAudio3dFormat::ORBIS_AUDIO3D_FORMAT_S16) {
        if ((reinterpret_cast<uintptr_t>(buffer) & 1) != 0) {
            LOG_ERROR(Lib_Audio3d, "buffer & 1 != 0");
            return ORBIS_AUDIO3D_ERROR_INVALID_PARAMETER;
        }
    }

    std::scoped_lock lock{state->ports[port_id].mutex};

    // Per SDK docs: returns NOT_READY if all buffers are full.
    // The bed queue depth matches the port's queue_depth (same as the mixed_queue).
    const auto& port = state->ports[port_id];
    if (port.bed_queue.size() >= port.parameters.queue_depth) {
        return ORBIS_AUDIO3D_ERROR_NOT_READY;
    }

    return ConvertAndEnqueue(state->ports[port_id].bed_queue,
                             OrbisAudio3dPcm{
                                 .format = format,
                                 .sample_buffer = buffer,
                                 .num_samples = num_samples,
                             },
                             num_channels, state->ports[port_id].parameters.granularity);
}

s32 PS4_SYSV_ABI sceAudio3dCreateSpeakerArray() {
    LOG_ERROR(Lib_Audio3d, "(STUBBED) called");
    return ORBIS_OK;
}

s32 PS4_SYSV_ABI sceAudio3dDeleteSpeakerArray() {
    LOG_ERROR(Lib_Audio3d, "(STUBBED) called");
    return ORBIS_OK;
}

s32 PS4_SYSV_ABI sceAudio3dGetDefaultOpenParameters(OrbisAudio3dOpenParameters* params) {
    LOG_DEBUG(Lib_Audio3d, "called");
    if (params) {
        auto default_params = OrbisAudio3dOpenParameters{
            .size_this = 0x20,
            .granularity = 0x100,
            .rate = OrbisAudio3dRate::ORBIS_AUDIO3D_RATE_48000,
            .max_objects = 512,
            .queue_depth = 2,
            .buffer_mode = OrbisAudio3dBufferMode::ORBIS_AUDIO3D_BUFFER_ADVANCE_AND_PUSH,
        };
        memcpy(params, &default_params, 0x20);
    }
    return ORBIS_OK;
}

s32 PS4_SYSV_ABI sceAudio3dGetSpeakerArrayMemorySize() {
    LOG_ERROR(Lib_Audio3d, "(STUBBED) called");
    return ORBIS_OK;
}

s32 PS4_SYSV_ABI sceAudio3dGetSpeakerArrayMixCoefficients() {
    LOG_ERROR(Lib_Audio3d, "(STUBBED) called");
    return ORBIS_OK;
}

s32 PS4_SYSV_ABI sceAudio3dGetSpeakerArrayMixCoefficients2() {
    LOG_ERROR(Lib_Audio3d, "(STUBBED) called");
    return ORBIS_OK;
}

s32 PS4_SYSV_ABI sceAudio3dInitialize(const s64 reserved) {
    LOG_INFO(Lib_Audio3d, "called, reserved = {}", reserved);

    if (reserved != 0) {
        LOG_ERROR(Lib_Audio3d, "reserved != 0");
        return ORBIS_AUDIO3D_ERROR_INVALID_PARAMETER;
    }

    if (state) {
        LOG_ERROR(Lib_Audio3d, "already initialized");
        return ORBIS_AUDIO3D_ERROR_NOT_READY;
    }

    state = std::make_unique<Audio3dState>();

    if (const auto init_ret = AudioOut::sceAudioOutInit();
        init_ret < 0 && init_ret != ORBIS_AUDIO_OUT_ERROR_ALREADY_INIT) {
        return init_ret;
    }

    return ORBIS_OK;
}

s32 PS4_SYSV_ABI sceAudio3dObjectReserve(const OrbisAudio3dPortId port_id,
                                         OrbisAudio3dObjectId* object_id) {
    LOG_INFO(Lib_Audio3d, "called, port_id = {}, object_id = {}", port_id,
             static_cast<void*>(object_id));

    if (!object_id) {
        LOG_ERROR(Lib_Audio3d, "!object_id");
        return ORBIS_AUDIO3D_ERROR_INVALID_PARAMETER;
    }

    // Per SDK docs: if the operation fails, SCE_AUDIO3D_OBJECT_INVALID must be returned.
    *object_id = ORBIS_AUDIO3D_OBJECT_INVALID;

    if (!state->ports.contains(port_id)) {
        LOG_ERROR(Lib_Audio3d, "!state->ports.contains(port_id)");
        return ORBIS_AUDIO3D_ERROR_INVALID_PORT;
    }

    auto& port = state->ports[port_id];
    std::scoped_lock lock{port.mutex};

    // Enforce the max_objects limit set at PortOpen time.
    if (port.objects.size() >= port.parameters.max_objects) {
        LOG_ERROR(Lib_Audio3d, "port has no available objects (max_objects = {})",
                  port.parameters.max_objects);
        return ORBIS_AUDIO3D_ERROR_OUT_OF_RESOURCES;
    }

    // ID counter lives in the Port so it resets when the port is closed and reopened,
    // and is automatically protected by port.mutex. Skip 0 and OBJECT_INVALID (0xFFFFFFFF).
    do {
        ++port.next_object_id;
    } while (port.next_object_id == 0 || port.next_object_id == ORBIS_AUDIO3D_OBJECT_INVALID ||
             port.objects.contains(port.next_object_id));

    *object_id = port.next_object_id;
    port.objects.emplace(*object_id, ObjectState{});
    LOG_INFO(Lib_Audio3d, "reserved object_id = {}", *object_id);

    return ORBIS_OK;
}

s32 PS4_SYSV_ABI sceAudio3dObjectSetAttribute(const OrbisAudio3dPortId port_id,
                                              const OrbisAudio3dObjectId object_id,
                                              const OrbisAudio3dAttributeId attribute_id,
                                              const void* attribute, const u64 attribute_size) {
    LOG_DEBUG(Lib_Audio3d, "called, port_id = {}, object_id = {}, attribute_id = {:#x}, size = {}",
              port_id, object_id, static_cast<u32>(attribute_id), attribute_size);

    if (!state->ports.contains(port_id)) {
        LOG_ERROR(Lib_Audio3d, "!state->ports.contains(port_id)");
        return ORBIS_AUDIO3D_ERROR_INVALID_PORT;
    }

    auto& port = state->ports[port_id];
    std::scoped_lock lock{port.mutex};
    if (!port.objects.contains(object_id)) {
        LOG_ERROR(Lib_Audio3d, "object_id {} not reserved", object_id);
        return ORBIS_AUDIO3D_ERROR_INVALID_OBJECT;
    }

    auto& obj = port.objects[object_id];

    // RESET_STATE clears all attributes and queued PCM; it takes no value.
    if (attribute_id == OrbisAudio3dAttributeId::ORBIS_AUDIO3D_ATTRIBUTE_RESET_STATE) {
        // Per SDK docs: "the state of an object cannot be reset after a PCM attribute has
        // been set" in the same frame (prior to the next PortAdvance/PortFlush).
        if (obj.pcm_submitted_this_frame) {
            LOG_ERROR(Lib_Audio3d, "RESET_STATE called after PCM in same frame (NOT_READY)");
            return ORBIS_AUDIO3D_ERROR_NOT_READY;
        }
        for (auto& data : obj.pcm_queue) {
            std::free(data.sample_buffer);
        }
        obj.pcm_queue.clear();
        obj.persistent_attributes.clear();
        LOG_DEBUG(Lib_Audio3d, "RESET_STATE for object {}", object_id);
        return ORBIS_OK;
    }

    if (!attribute || !attribute_size) {
        LOG_ERROR(Lib_Audio3d, "!attribute || !attribute_size");
        return ORBIS_AUDIO3D_ERROR_INVALID_PARAMETER;
    }

    if (attribute_id == OrbisAudio3dAttributeId::ORBIS_AUDIO3D_ATTRIBUTE_PCM) {
        if (attribute_size < sizeof(OrbisAudio3dPcm)) {
            LOG_ERROR(Lib_Audio3d, "PCM attribute value_size too small");
            return ORBIS_AUDIO3D_ERROR_INVALID_PARAMETER;
        }
        const auto* pcm = static_cast<const OrbisAudio3dPcm*>(attribute);

        // Alignment check: S16 must be 2-byte aligned, float must be 4-byte aligned.
        const uintptr_t buf_addr = reinterpret_cast<uintptr_t>(pcm->sample_buffer);
        if (pcm->format == OrbisAudio3dFormat::ORBIS_AUDIO3D_FORMAT_S16) {
            if (buf_addr & 1) {
                LOG_ERROR(Lib_Audio3d, "S16 sample buffer not 2-byte aligned");
                return ORBIS_AUDIO3D_ERROR_INVALID_PARAMETER;
            }
        } else if (pcm->format == OrbisAudio3dFormat::ORBIS_AUDIO3D_FORMAT_FLOAT) {
            if (buf_addr & 3) {
                LOG_ERROR(Lib_Audio3d, "float sample buffer not 4-byte aligned");
                return ORBIS_AUDIO3D_ERROR_INVALID_PARAMETER;
            }
        }

        // NOT_READY if the object's PCM queue is already full.
        if (obj.pcm_queue.size() >= port.parameters.queue_depth) {
            return ORBIS_AUDIO3D_ERROR_NOT_READY;
        }

        if (const auto ret = ConvertAndEnqueue(obj.pcm_queue, *pcm, 1, port.parameters.granularity);
            ret != ORBIS_OK) {
            return ret;
        }
        obj.pcm_submitted_this_frame = true;
        return ORBIS_OK;
    }

    // Validate type-specific value constraints (GAIN >= 0, SPREAD in [0, 2*PI], etc.)
    if (const auto ret = ValidateObjectAttributeValue(attribute_id, attribute, attribute_size);
        ret != ORBIS_OK) {
        return ret;
    }

    // Store the attribute persistently on the object. It survives across frames
    // until changed again or the object is unreserved.
    const auto* src = static_cast<const u8*>(attribute);
    obj.persistent_attributes[static_cast<u32>(attribute_id)].assign(src, src + attribute_size);

    return ORBIS_OK;
}

s32 PS4_SYSV_ABI sceAudio3dObjectSetAttributes(const OrbisAudio3dPortId port_id,
                                               OrbisAudio3dObjectId object_id,
                                               const u64 num_attributes,
                                               const OrbisAudio3dAttribute* attribute_array) {
    LOG_DEBUG(Lib_Audio3d,
              "called, port_id = {}, object_id = {}, num_attributes = {}, attribute_array = {}",
              port_id, object_id, num_attributes, fmt::ptr(attribute_array));

    if (!state->ports.contains(port_id)) {
        LOG_ERROR(Lib_Audio3d, "!state->ports.contains(port_id)");
        return ORBIS_AUDIO3D_ERROR_INVALID_PORT;
    }

    if (!num_attributes || !attribute_array) {
        LOG_ERROR(Lib_Audio3d, "!num_attributes || !attribute_array");
        return ORBIS_AUDIO3D_ERROR_INVALID_PARAMETER;
    }

    // Per SDK docs: passing the same attribute ID more than once returns INVALID_PARAMETER.
    // Attribute IDs are 1-11, so a u32 bitmask covers all of them.
    {
        u32 seen_mask = 0;
        for (u64 i = 0; i < num_attributes; i++) {
            const u32 id = static_cast<u32>(attribute_array[i].attribute_id);
            const u32 bit = (id < 32) ? (u32{1} << id) : 0;
            if (bit && (seen_mask & bit)) {
                LOG_ERROR(Lib_Audio3d, "duplicate attribute_id {:#x}", id);
                return ORBIS_AUDIO3D_ERROR_INVALID_PARAMETER;
            }
            seen_mask |= bit;
        }
    }

    auto& port = state->ports[port_id];
    std::scoped_lock lock{port.mutex};
    if (!port.objects.contains(object_id)) {
        // On real hardware, the race window between ObjectUnreserve (main thread) and
        // ObjectSetAttributes (audio thread) is a silent no-op, not a hard error.
        // Returning INVALID_OBJECT causes the game to reach code paths marked unreachable
        // on real HW. Log at debug level and return success.
        LOG_DEBUG(Lib_Audio3d, "object_id {} not reserved (race with Unreserve?), no-op",
                  object_id);
        return ORBIS_OK;
    }

    auto& obj = port.objects[object_id];

    // Per SDK docs: processing order is always RESET_STATE first, PCM last, regardless of
    // array order. Non-PCM non-RESET attributes are applied in between (pass 2).

    // Pass 1: RESET_STATE
    for (u64 i = 0; i < num_attributes; i++) {
        if (attribute_array[i].attribute_id !=
            OrbisAudio3dAttributeId::ORBIS_AUDIO3D_ATTRIBUTE_RESET_STATE) {
            continue;
        }
        // Per SDK docs: RESET_STATE returns NOT_READY if PCM was already submitted this
        // frame (prior to the next PortAdvance/PortFlush).
        if (obj.pcm_submitted_this_frame) {
            LOG_ERROR(Lib_Audio3d, "RESET_STATE called after PCM in same frame (NOT_READY)");
            return ORBIS_AUDIO3D_ERROR_NOT_READY;
        }
        for (auto& data : obj.pcm_queue) {
            std::free(data.sample_buffer);
        }
        obj.pcm_queue.clear();
        obj.persistent_attributes.clear();
        obj.pcm_submitted_this_frame = false;
        LOG_DEBUG(Lib_Audio3d, "RESET_STATE for object {}", object_id);
        break; // Duplicate check above ensures at most one RESET_STATE.
    }

    // Pass 2: all non-PCM, non-RESET_STATE attributes (POSITION, GAIN, SPREAD, etc.)
    // Stored persistently — survives across frames until changed or object unreserved.
    for (u64 i = 0; i < num_attributes; i++) {
        const auto& attr = attribute_array[i];
        if (attr.attribute_id == OrbisAudio3dAttributeId::ORBIS_AUDIO3D_ATTRIBUTE_RESET_STATE ||
            attr.attribute_id == OrbisAudio3dAttributeId::ORBIS_AUDIO3D_ATTRIBUTE_PCM) {
            continue;
        }
        if (attr.value && attr.value_size > 0) {
            // Validate type-specific constraints (GAIN >= 0, SPREAD in [0, 2*PI], etc.)
            if (const auto ret =
                    ValidateObjectAttributeValue(attr.attribute_id, attr.value, attr.value_size);
                ret != ORBIS_OK) {
                return ret;
            }
            const auto* src = static_cast<const u8*>(attr.value);
            obj.persistent_attributes[static_cast<u32>(attr.attribute_id)].assign(
                src, src + attr.value_size);
            LOG_DEBUG(Lib_Audio3d, "Stored attribute {:#x} for object {}",
                      static_cast<u32>(attr.attribute_id), object_id);
        }
    }

    // Pass 3: PCM (always last per SDK docs).
    for (u64 i = 0; i < num_attributes; i++) {
        const auto& attr = attribute_array[i];
        if (attr.attribute_id != OrbisAudio3dAttributeId::ORBIS_AUDIO3D_ATTRIBUTE_PCM) {
            continue;
        }
        if (attr.value_size < sizeof(OrbisAudio3dPcm)) {
            LOG_ERROR(Lib_Audio3d, "PCM attribute value_size too small");
            return ORBIS_AUDIO3D_ERROR_INVALID_PARAMETER;
        }
        const auto* pcm = static_cast<const OrbisAudio3dPcm*>(attr.value);

        // Alignment check: S16 must be 2-byte aligned, float must be 4-byte aligned.
        const uintptr_t buf_addr = reinterpret_cast<uintptr_t>(pcm->sample_buffer);
        if (pcm->format == OrbisAudio3dFormat::ORBIS_AUDIO3D_FORMAT_S16) {
            if (buf_addr & 1) {
                LOG_ERROR(Lib_Audio3d, "S16 sample buffer not 2-byte aligned");
                return ORBIS_AUDIO3D_ERROR_INVALID_PARAMETER;
            }
        } else if (pcm->format == OrbisAudio3dFormat::ORBIS_AUDIO3D_FORMAT_FLOAT) {
            if (buf_addr & 3) {
                LOG_ERROR(Lib_Audio3d, "float sample buffer not 4-byte aligned");
                return ORBIS_AUDIO3D_ERROR_INVALID_PARAMETER;
            }
        }

        // NOT_READY if the object's PCM queue is already full.
        if (obj.pcm_queue.size() >= port.parameters.queue_depth) {
            return ORBIS_AUDIO3D_ERROR_NOT_READY;
        }

        if (const auto ret = ConvertAndEnqueue(obj.pcm_queue, *pcm, 1, port.parameters.granularity);
            ret != ORBIS_OK) {
            return ret;
        }
        obj.pcm_submitted_this_frame = true;
        break; // Duplicate check above ensures at most one PCM attribute.
    }

    return ORBIS_OK;
}

s32 PS4_SYSV_ABI sceAudio3dObjectUnreserve(const OrbisAudio3dPortId port_id,
                                           const OrbisAudio3dObjectId object_id) {
    LOG_DEBUG(Lib_Audio3d, "called, port_id = {}, object_id = {}", port_id, object_id);

    if (!state->ports.contains(port_id)) {
        LOG_ERROR(Lib_Audio3d, "!state->ports.contains(port_id)");
        return ORBIS_AUDIO3D_ERROR_INVALID_PORT;
    }

    auto& port = state->ports[port_id];
    std::scoped_lock lock{port.mutex};

    if (!port.objects.contains(object_id)) {
        LOG_ERROR(Lib_Audio3d, "object_id not reserved");
        return ORBIS_AUDIO3D_ERROR_INVALID_OBJECT;
    }

    // Free any queued PCM audio for this object.
    for (auto& data : port.objects[object_id].pcm_queue) {
        std::free(data.sample_buffer);
    }

    port.objects.erase(object_id);
    return ORBIS_OK;
}

s32 PS4_SYSV_ABI sceAudio3dPortAdvance(const OrbisAudio3dPortId port_id) {
    LOG_DEBUG(Lib_Audio3d, "called, port_id = {}", port_id);

    if (!state->ports.contains(port_id)) {
        LOG_ERROR(Lib_Audio3d, "!state->ports.contains(port_id)");
        return ORBIS_AUDIO3D_ERROR_INVALID_PORT;
    }

    auto& port = state->ports[port_id];

    if (port.parameters.buffer_mode == OrbisAudio3dBufferMode::ORBIS_AUDIO3D_BUFFER_NO_ADVANCE) {
        LOG_ERROR(Lib_Audio3d, "port doesn't have advance capability");
        return ORBIS_AUDIO3D_ERROR_NOT_SUPPORTED;
    }

    if (port.mixed_queue.size() >= port.parameters.queue_depth) {
        LOG_WARNING(Lib_Audio3d, "mixed queue full (depth={}), dropping advance",
                    port.parameters.queue_depth);
        return ORBIS_AUDIO3D_ERROR_NOT_READY;
    }

    const u32 granularity = port.parameters.granularity;
    const u32 out_samples = granularity * AUDIO3D_OUTPUT_NUM_CHANNELS;

    // ---- FLOAT MIX BUFFER ----
    float* mix_float = static_cast<float*>(std::calloc(out_samples, sizeof(float)));

    if (!mix_float)
        return ORBIS_AUDIO3D_ERROR_OUT_OF_MEMORY;

    auto mix_in = [&](std::deque<AudioData>& queue, const float gain_l, const float gain_r) {
        if (queue.empty())
            return;

        // Per SDK docs: default gain is 0.0 — objects with no GAIN set are silent.
        // Fast path: skip the conversion work entirely for silent objects.
        if (gain_l == 0.0f && gain_r == 0.0f) {
            AudioData data = queue.front();
            queue.pop_front();
            std::free(data.sample_buffer);
            return;
        }

        AudioData data = queue.front();
        queue.pop_front();

        const u32 frames = std::min(granularity, data.num_samples);
        const u32 channels = data.num_channels;

        if (data.format == OrbisAudio3dFormat::ORBIS_AUDIO3D_FORMAT_S16) {
            const s16* src = reinterpret_cast<const s16*>(data.sample_buffer);

            for (u32 i = 0; i < frames; i++) {
                float left = 0.0f;
                float right = 0.0f;

                if (channels == 1) {
                    float v = src[i] / 32768.0f;
                    left = v;
                    right = v;
                } else {
                    left = src[i * channels + 0] / 32768.0f;
                    right = src[i * channels + 1] / 32768.0f;
                }

                mix_float[i * 2 + 0] += left * gain_l;
                mix_float[i * 2 + 1] += right * gain_r;
            }
        } else { // FLOAT input
            const float* src = reinterpret_cast<const float*>(data.sample_buffer);

            for (u32 i = 0; i < frames; i++) {
                float left = 0.0f;
                float right = 0.0f;

                if (channels == 1) {
                    left = src[i];
                    right = src[i];
                } else {
                    left = src[i * channels + 0];
                    right = src[i * channels + 1];
                }

                mix_float[i * 2 + 0] += left * gain_l;
                mix_float[i * 2 + 1] += right * gain_r;
            }
        }

        std::free(data.sample_buffer);
    };

    // Bed is mixed at full gain to both channels — no per-bed position or gain attribute.
    mix_in(port.bed_queue, 1.0f, 1.0f);

    // ---- PRIORITY-BASED OBJECT CULLING ----
    // Per SDK docs: PRIORITY is a u32 where 0 = highest priority. When more objects have
    // PCM queued this frame than max_objects allows, the lowest-priority ones are dropped
    // (their PCM frame is consumed and discarded without mixing). Objects with no PCM
    // queued this frame do not consume a slot — culling only applies to active objects.
    const auto priority_key =
        static_cast<u32>(OrbisAudio3dAttributeId::ORBIS_AUDIO3D_ATTRIBUTE_PRIORITY);
    const auto gain_key = static_cast<u32>(OrbisAudio3dAttributeId::ORBIS_AUDIO3D_ATTRIBUTE_GAIN);
    const auto pos_key =
        static_cast<u32>(OrbisAudio3dAttributeId::ORBIS_AUDIO3D_ATTRIBUTE_POSITION);
    const auto spread_key =
        static_cast<u32>(OrbisAudio3dAttributeId::ORBIS_AUDIO3D_ATTRIBUTE_SPREAD);
    const auto passthrough_key =
        static_cast<u32>(OrbisAudio3dAttributeId::ORBIS_AUDIO3D_ATTRIBUTE_PASSTHROUGH);

    struct ObjPriority {
        u32 priority;
        OrbisAudio3dObjectId obj_id;
    };

    std::vector<ObjPriority> active_objects;
    active_objects.reserve(port.objects.size());

    for (const auto& [obj_id, obj] : port.objects) {
        if (obj.pcm_queue.empty())
            continue;

        // Per SDK docs: PASSTHROUGH objects bypass priority culling — they are always
        // mixed dry and do not consume a slot from the max_objects budget.
        OrbisAudio3dPassthrough passthrough =
            OrbisAudio3dPassthrough::ORBIS_AUDIO3D_PASSTHROUGH_NONE;
        if (obj.persistent_attributes.contains(passthrough_key)) {
            const auto& blob = obj.persistent_attributes.at(passthrough_key);
            if (blob.size() >= sizeof(u32)) {
                u32 raw = 0;
                std::memcpy(&raw, blob.data(), sizeof(u32));
                passthrough = static_cast<OrbisAudio3dPassthrough>(raw);
            }
        }
        if (passthrough != OrbisAudio3dPassthrough::ORBIS_AUDIO3D_PASSTHROUGH_NONE)
            continue; // Mixed in the passthrough pass below.

        u32 priority = 0xFFFFFFFFu; // Default: lowest possible if PRIORITY not set.
        if (obj.persistent_attributes.contains(priority_key)) {
            const auto& blob = obj.persistent_attributes.at(priority_key);
            if (blob.size() >= sizeof(u32)) {
                std::memcpy(&priority, blob.data(), sizeof(u32));
            }
        }
        active_objects.push_back({priority, obj_id});
    }

    // Stable sort ascending — priority 0 (highest) first.
    std::stable_sort(
        active_objects.begin(), active_objects.end(),
        [](const ObjPriority& a, const ObjPriority& b) { return a.priority < b.priority; });

    const u32 mix_limit = port.parameters.max_objects;
    const u32 active_count = static_cast<u32>(active_objects.size());

    if (active_count > mix_limit) {
        LOG_DEBUG(Lib_Audio3d, "culling {} low-priority objects ({} active, limit {})",
                  active_count - mix_limit, active_count, mix_limit);
    }

    // ---- SPATIALISED MIX ----
    // Per SDK docs:
    //   GAIN     — scales output volume (default 0.0 = silent).
    //   POSITION — sets 3D direction; does not affect volume, only stereo pan.
    //   SPREAD   — widens the stereo image toward center (0 = fully panned, 2π = centered).
    for (u32 slot = 0; slot < active_count; slot++) {
        const OrbisAudio3dObjectId obj_id = active_objects[slot].obj_id;
        auto& obj = port.objects[obj_id];

        if (slot >= mix_limit) {
            // Culled: consume the PCM frame without mixing so the queue stays in sync.
            AudioData dropped = obj.pcm_queue.front();
            obj.pcm_queue.pop_front();
            std::free(dropped.sample_buffer);
            obj.pcm_submitted_this_frame = false;
            continue;
        }

        // Read GAIN (default 0.0 — silent until explicitly set).
        float gain = 0.0f;
        if (obj.persistent_attributes.contains(gain_key)) {
            const auto& blob = obj.persistent_attributes.at(gain_key);
            if (blob.size() >= sizeof(float)) {
                std::memcpy(&gain, blob.data(), sizeof(float));
            }
        }

        // Read POSITION and SPREAD to compute per-channel gain factors via SpatialPan.
        float gain_l = gain;
        float gain_r = gain;
        if (gain > 0.0f && obj.persistent_attributes.contains(pos_key)) {
            const auto& pos_blob = obj.persistent_attributes.at(pos_key);
            if (pos_blob.size() >= sizeof(OrbisAudio3dPosition)) {
                OrbisAudio3dPosition pos{};
                std::memcpy(&pos, pos_blob.data(), sizeof(OrbisAudio3dPosition));

                float spread = 0.0f;
                if (obj.persistent_attributes.contains(spread_key)) {
                    const auto& sp_blob = obj.persistent_attributes.at(spread_key);
                    if (sp_blob.size() >= sizeof(float)) {
                        std::memcpy(&spread, sp_blob.data(), sizeof(float));
                    }
                }

                float pan_l, pan_r;
                SpatialPan(pos, spread, pan_l, pan_r);
                gain_l = gain * pan_l;
                gain_r = gain * pan_r;
            }
        }

        mix_in(obj.pcm_queue, gain_l, gain_r);
        obj.pcm_submitted_this_frame = false;
    }

    // Objects with no PCM this frame still need their flag cleared.
    for (auto& [obj_id, obj] : port.objects) {
        if (obj.pcm_queue.empty()) {
            obj.pcm_submitted_this_frame = false;
        }
    }

    // ---- PASSTHROUGH OBJECTS ----
    // Per SDK docs: when PASSTHROUGH != NONE, position/gain/spread/priority are all ignored.
    // The PCM is mixed dry at unity gain, routed according to the passthrough mode:
    //   LEFT   — mono signal to left channel only.
    //   RIGHT  — mono signal to right channel only.
    //   STEREO — dry stereo (or mono duplicated to both channels) at unity gain.
    for (auto& [obj_id, obj] : port.objects) {
        if (obj.pcm_queue.empty())
            continue;
        if (!obj.persistent_attributes.contains(passthrough_key))
            continue;

        const auto& blob = obj.persistent_attributes.at(passthrough_key);
        if (blob.size() < sizeof(u32))
            continue;

        u32 raw = 0;
        std::memcpy(&raw, blob.data(), sizeof(u32));
        const auto passthrough = static_cast<OrbisAudio3dPassthrough>(raw);
        if (passthrough == OrbisAudio3dPassthrough::ORBIS_AUDIO3D_PASSTHROUGH_NONE)
            continue;

        float gain_l = 0.0f;
        float gain_r = 0.0f;
        switch (passthrough) {
        case OrbisAudio3dPassthrough::ORBIS_AUDIO3D_PASSTHROUGH_LEFT:
            gain_l = 1.0f;
            gain_r = 0.0f;
            break;
        case OrbisAudio3dPassthrough::ORBIS_AUDIO3D_PASSTHROUGH_RIGHT:
            gain_l = 0.0f;
            gain_r = 1.0f;
            break;
        case OrbisAudio3dPassthrough::ORBIS_AUDIO3D_PASSTHROUGH_STEREO:
        default:
            gain_l = 1.0f;
            gain_r = 1.0f;
            break;
        }

        mix_in(obj.pcm_queue, gain_l, gain_r);
        obj.pcm_submitted_this_frame = false;
    }

    // ---- FINAL FLOAT → S16 CONVERSION ----
    s16* mix_s16 = static_cast<s16*>(std::malloc(out_samples * sizeof(s16)));

    if (!mix_s16) {
        std::free(mix_float);
        return ORBIS_AUDIO3D_ERROR_OUT_OF_MEMORY;
    }

    for (u32 i = 0; i < out_samples; i++) {
        float v = std::clamp(mix_float[i], -1.0f, 1.0f);
        mix_s16[i] = static_cast<s16>(v * 32767.0f);
    }

    std::free(mix_float);

    port.mixed_queue.push_back(AudioData{.sample_buffer = reinterpret_cast<u8*>(mix_s16),
                                         .num_samples = granularity,
                                         .num_channels = AUDIO3D_OUTPUT_NUM_CHANNELS,
                                         .format = OrbisAudio3dFormat::ORBIS_AUDIO3D_FORMAT_S16});

    return ORBIS_OK;
}
s32 PS4_SYSV_ABI sceAudio3dPortClose(const OrbisAudio3dPortId port_id) {
    LOG_INFO(Lib_Audio3d, "called, port_id = {}", port_id);

    if (!state->ports.contains(port_id)) {
        LOG_ERROR(Lib_Audio3d, "!state->ports.contains(port_id)");
        return ORBIS_AUDIO3D_ERROR_INVALID_PORT;
    }

    auto& port = state->ports[port_id];
    {
        std::scoped_lock lock{port.mutex};

        // Close the internal push-model audio handle if it was opened.
        if (port.audio_out_handle >= 0) {
            AudioOut::sceAudioOutClose(port.audio_out_handle);
            port.audio_out_handle = -1;
        }

        // Per SDK docs: "the call to sceAudio3dPortClose() will close all associated AudioOut
        // ports that are still open." Close any game-managed handles the game forgot to close.
        for (const s32 handle : port.audioout_handles) {
            AudioOut::sceAudioOutClose(handle);
        }
        port.audioout_handles.clear();

        // Free all pending mixed frames.
        for (auto& data : port.mixed_queue) {
            std::free(data.sample_buffer);
        }

        // Free all queued bed audio.
        for (auto& data : port.bed_queue) {
            std::free(data.sample_buffer);
        }

        // Free all queued object PCM audio.
        for (auto& [obj_id, obj] : port.objects) {
            for (auto& data : obj.pcm_queue) {
                std::free(data.sample_buffer);
            }
        }
    } // lock released here — mutex must not be held when erase destroys the Port

    state->ports.erase(port_id);
    return ORBIS_OK;
}

s32 PS4_SYSV_ABI sceAudio3dPortCreate() {
    LOG_ERROR(Lib_Audio3d, "(STUBBED) called");
    return ORBIS_OK;
}

s32 PS4_SYSV_ABI sceAudio3dPortDestroy() {
    LOG_ERROR(Lib_Audio3d, "(STUBBED) called");
    return ORBIS_OK;
}

s32 PS4_SYSV_ABI sceAudio3dPortFlush(const OrbisAudio3dPortId port_id) {
    LOG_DEBUG(Lib_Audio3d, "called, port_id = {}", port_id);

    if (!state->ports.contains(port_id)) {
        LOG_ERROR(Lib_Audio3d, "!state->ports.contains(port_id)");
        return ORBIS_AUDIO3D_ERROR_INVALID_PORT;
    }

    auto& port = state->ports[port_id];
    std::scoped_lock lock{port.mutex};

    // PortFlush has three usage modes depending on SDK version and port configuration
    // (per SDK docs Table 8 / Buffering Strategies):
    //
    // Mode A — Standalone (SDK 1.0, deprecated, buffer_mode = ANY):
    //   PortFlush alone replaces Advance+Push. It must mix all queued audio and output
    //   it, blocking until the device consumes the buffer. No game-managed AudioOut.
    //
    // Mode B — Advance + Flush (SDK 2.0, deprecated, buffer_mode = ADVANCE_NO_PUSH):
    //   PortAdvance already mixed audio into mixed_queue. PortFlush just outputs it.
    //   Still no game-managed AudioOut in this mode.
    //
    // Mode C — Combined AudioOut mode (SDK 2.0+):
    //   Game has AudioOut handles from sceAudio3dAudioOutOpen. PortFlush synchronizes
    //   those AudioOut ports by blocking until their pending buffers are consumed.

    if (!port.audioout_handles.empty()) {
        // Mode C: synchronize all associated game-managed AudioOut ports.
        // Per SDK docs: "In addition to synchronizing the objects and the bed passed to
        // the Audio3d port, these calls also synchronize all associated AudioOut ports."
        for (const s32 handle : port.audioout_handles) {
            const s32 ret = AudioOut::sceAudioOutOutput(handle, nullptr);
            if (ret < 0) {
                return ret;
            }
        }
        return ORBIS_OK;
    }

    // Mode A or B: no game-managed AudioOut handles. Mix if needed (Mode A), then output.

    // If mixed_queue is empty, the game hasn't called PortAdvance — we're in Mode A
    // (standalone flush) and must do the mix step ourselves.
    if (port.mixed_queue.empty()) {
        // Only mix if there's actually something to mix.
        if (!port.bed_queue.empty() ||
            std::any_of(port.objects.begin(), port.objects.end(),
                        [](const auto& kv) { return !kv.second.pcm_queue.empty(); })) {
            const s32 ret = sceAudio3dPortAdvance(port_id);
            if (ret != ORBIS_OK && ret != ORBIS_AUDIO3D_ERROR_NOT_READY) {
                return ret;
            }
        }
    }

    if (port.mixed_queue.empty()) {
        // Nothing to output.
        return ORBIS_OK;
    }

    // Lazily open the internal audio output handle.
    if (port.audio_out_handle < 0) {
        AudioOut::OrbisAudioOutParamExtendedInformation ext_info{};
        ext_info.data_format.Assign(AUDIO3D_OUTPUT_FORMAT);
        port.audio_out_handle =
            AudioOut::sceAudioOutOpen(0xFF, AudioOut::OrbisAudioOutPort::Audio3d, 0,
                                      port.parameters.granularity, AUDIO3D_SAMPLE_RATE, ext_info);
        if (port.audio_out_handle < 0) {
            return port.audio_out_handle;
        }
    }

    // Drain all queued mixed frames, blocking on each until consumed.
    while (!port.mixed_queue.empty()) {
        AudioData frame = port.mixed_queue.front();
        port.mixed_queue.pop_front();
        const s32 ret = AudioOut::sceAudioOutOutput(port.audio_out_handle, frame.sample_buffer);
        std::free(frame.sample_buffer);
        if (ret < 0) {
            return ret;
        }
    }

    return ORBIS_OK;
}

s32 PS4_SYSV_ABI sceAudio3dPortFreeState() {
    LOG_ERROR(Lib_Audio3d, "(STUBBED) called");
    return ORBIS_OK;
}

s32 PS4_SYSV_ABI sceAudio3dPortGetAttributesSupported() {
    LOG_ERROR(Lib_Audio3d, "(STUBBED) called");
    return ORBIS_OK;
}

s32 PS4_SYSV_ABI sceAudio3dPortGetList() {
    LOG_ERROR(Lib_Audio3d, "(STUBBED) called");
    return ORBIS_OK;
}

s32 PS4_SYSV_ABI sceAudio3dPortGetParameters() {
    LOG_ERROR(Lib_Audio3d, "(STUBBED) called");
    return ORBIS_OK;
}

s32 PS4_SYSV_ABI sceAudio3dPortGetQueueLevel(const OrbisAudio3dPortId port_id, u32* queue_level,
                                             u32* queue_available) {
    LOG_DEBUG(Lib_Audio3d, "called, port_id = {}, queue_level = {}, queue_available = {}", port_id,
              static_cast<void*>(queue_level), static_cast<void*>(queue_available));

    if (!state->ports.contains(port_id)) {
        LOG_ERROR(Lib_Audio3d, "!state->ports.contains(port_id)");
        return ORBIS_AUDIO3D_ERROR_INVALID_PORT;
    }

    if (!queue_level && !queue_available) {
        return ORBIS_AUDIO3D_ERROR_INVALID_PARAMETER;
    }

    const auto& port = state->ports[port_id];
    std::scoped_lock lock{port.mutex};
    const size_t size = port.mixed_queue.size();

    if (queue_level) {
        *queue_level = static_cast<u32>(size);
    }

    if (queue_available) {
        const u32 depth = port.parameters.queue_depth;
        *queue_available = (size < depth) ? static_cast<u32>(depth - size) : 0u;
    }

    return ORBIS_OK;
}

s32 PS4_SYSV_ABI sceAudio3dPortGetState() {
    LOG_ERROR(Lib_Audio3d, "(STUBBED) called");
    return ORBIS_OK;
}

s32 PS4_SYSV_ABI sceAudio3dPortGetStatus() {
    LOG_ERROR(Lib_Audio3d, "(STUBBED) called");
    return ORBIS_OK;
}

s32 PS4_SYSV_ABI sceAudio3dPortOpen(const Libraries::UserService::OrbisUserServiceUserId user_id,
                                    const OrbisAudio3dOpenParameters* parameters,
                                    OrbisAudio3dPortId* port_id) {
    LOG_INFO(Lib_Audio3d, "called, user_id = {}, parameters = {}, id = {}", user_id,
             static_cast<const void*>(parameters), static_cast<void*>(port_id));

    // RE: only SCE_USER_SERVICE_USER_ID_SYSTEM (0xFF) is accepted. All other user IDs,
    // or null pId/pParameters, jump straight to the INVALID_PARAMETER path.
    if (user_id != 0xFF || !port_id || !parameters) {
        if (port_id) {
            *port_id = ORBIS_AUDIO3D_PORT_INVALID;
        }
        return ORBIS_AUDIO3D_ERROR_INVALID_PARAMETER;
    }

    // Per SDK docs: on failure, *port_id receives ORBIS_AUDIO3D_PORT_INVALID.
    *port_id = ORBIS_AUDIO3D_PORT_INVALID;

    if (!state) {
        LOG_ERROR(Lib_Audio3d, "!initialized");
        return ORBIS_AUDIO3D_ERROR_NOT_READY;
    }

    // RE: size_this encodes the SDK version of the struct. The switch is effectively
    // (size_this - 0x10) >> 3, selecting which fields exist in the caller's struct.
    // Older SDK sizes get hardcoded defaults for fields that didn't exist yet.
    //   case 0: size=0x10 — granularity+rate only; max_objects=512, queue_depth=2,
    //                        num_beds=2, buffer_mode=NO_ADVANCE
    //   case 1: size=0x18 — adds max_objects+queue_depth; num_beds=2, buffer_mode=ADVANCE_NO_PUSH
    //   case 2: size=0x20 — adds buffer_mode; num_beds=2
    //   case 3: size=0x28 — full struct including num_beds
    u32 max_objects;
    u32 queue_depth;
    u32 num_beds;
    u32 buffer_mode_raw;

    const u64 version = (parameters->size_this - 0x10) >> 3;
    switch (version) {
    case 0:
        max_objects = 512;
        queue_depth = 2;
        num_beds = 2;
        buffer_mode_raw = static_cast<u32>(OrbisAudio3dBufferMode::ORBIS_AUDIO3D_BUFFER_NO_ADVANCE);
        break;
    case 1:
        max_objects = parameters->max_objects;
        queue_depth = parameters->queue_depth;
        num_beds = 2;
        buffer_mode_raw =
            static_cast<u32>(OrbisAudio3dBufferMode::ORBIS_AUDIO3D_BUFFER_ADVANCE_NO_PUSH);
        break;
    case 2:
        max_objects = parameters->max_objects;
        queue_depth = parameters->queue_depth;
        num_beds = 2;
        buffer_mode_raw = static_cast<u32>(parameters->buffer_mode);
        break;
    case 3:
        max_objects = parameters->max_objects;
        queue_depth = parameters->queue_depth;
        num_beds = parameters->num_beds;
        buffer_mode_raw = static_cast<u32>(parameters->buffer_mode);
        break;
    default:
        LOG_ERROR(Lib_Audio3d, "unrecognised size_this {:#x}", parameters->size_this);
        return ORBIS_AUDIO3D_ERROR_INVALID_PARAMETER;
    }

    const u32 granularity = parameters->granularity;
    const u32 rate = static_cast<u32>(parameters->rate);

    // RE: full validation chain as seen in the binary.
    // rate must be 0 (48000 Hz), granularity >= 256 and a multiple of 256.
    // Queue depth limits per granularity, num_beds must be 2 or 3, buffer_mode <= 2.
    bool invalid = (rate != 0) || (granularity < 0x100) || ((granularity & 0xFF) != 0) ||
                   (max_objects == 0) || (queue_depth == 0) ||
                   (granularity == 0x100 && queue_depth > 0x40) ||
                   (granularity == 0x200 && queue_depth > 0x1F) ||
                   (granularity == 0x300 && queue_depth > 0x14) ||
                   (queue_depth > 0xF && granularity > 0x3FF) ||
                   ((num_beds & 0xFFFFFFFEu) != 2) || // only 2 or 3 are valid
                   (buffer_mode_raw > 2);

    if (invalid) {
        LOG_ERROR(Lib_Audio3d,
                  "invalid parameters: gran={}, rate={}, max_obj={}, depth={}, beds={}, mode={}",
                  granularity, rate, max_objects, queue_depth, num_beds, buffer_mode_raw);
        return ORBIS_AUDIO3D_ERROR_INVALID_PARAMETER;
    }

    // RE: max_objects silently capped at 512 with a warning print.
    if (max_objects > 512) {
        LOG_WARNING(Lib_Audio3d, "[Audio3D] The uiMaxObjects is limit to 512");
        max_objects = 512;
    }

    // RE: up to 4 ports (indices 0–3), scanned for the first free slot.
    OrbisAudio3dPortId id = ORBIS_AUDIO3D_PORT_INVALID;
    for (OrbisAudio3dPortId candidate = 0; candidate < 4; candidate++) {
        if (!state->ports.contains(candidate)) {
            id = candidate;
            break;
        }
    }
    if (id == static_cast<OrbisAudio3dPortId>(ORBIS_AUDIO3D_PORT_INVALID)) {
        LOG_ERROR(Lib_Audio3d, "no free port slots");
        return ORBIS_AUDIO3D_ERROR_OUT_OF_RESOURCES;
    }

    *port_id = id;
    auto& port = state->ports[id];

    // Copy the fields we've resolved (potentially with version-defaulted values).
    port.parameters.size_this = parameters->size_this;
    port.parameters.granularity = granularity;
    port.parameters.rate = parameters->rate;
    port.parameters.max_objects = max_objects;
    port.parameters.queue_depth = queue_depth;
    port.parameters.buffer_mode = static_cast<OrbisAudio3dBufferMode>(buffer_mode_raw);
    port.parameters.num_beds = num_beds;

    return ORBIS_OK;
}

s32 PS4_SYSV_ABI sceAudio3dPortPush(const OrbisAudio3dPortId port_id,
                                    const OrbisAudio3dBlocking blocking) {
    LOG_DEBUG(Lib_Audio3d, "called, port_id = {}, blocking = {}", port_id,
              magic_enum::enum_name(blocking));

    if (!state->ports.contains(port_id)) {
        LOG_ERROR(Lib_Audio3d, "!state->ports.contains(port_id)");
        return ORBIS_AUDIO3D_ERROR_INVALID_PORT;
    }

    auto& port = state->ports[port_id];

    if (port.parameters.buffer_mode !=
        OrbisAudio3dBufferMode::ORBIS_AUDIO3D_BUFFER_ADVANCE_AND_PUSH) {
        LOG_ERROR(Lib_Audio3d, "port doesn't have push capability");
        return ORBIS_AUDIO3D_ERROR_NOT_SUPPORTED;
    }

    const u32 depth = port.parameters.queue_depth;

    // Lazily open output handle
    if (port.audio_out_handle < 0) {
        AudioOut::OrbisAudioOutParamExtendedInformation ext_info{};
        ext_info.data_format.Assign(AUDIO3D_OUTPUT_FORMAT);

        port.audio_out_handle =
            AudioOut::sceAudioOutOpen(0xFF, AudioOut::OrbisAudioOutPort::Audio3d, 0,
                                      port.parameters.granularity, AUDIO3D_SAMPLE_RATE, ext_info);

        if (port.audio_out_handle < 0)
            return port.audio_out_handle;
    }

    // Function that submits exactly one frame (if available)
    auto submit_one_frame = [&](bool& submitted) -> s32 {
        AudioData frame;

        {
            std::scoped_lock lock{port.mutex};

            if (port.mixed_queue.empty()) {
                submitted = false;
                return ORBIS_OK;
            }

            frame = port.mixed_queue.front();
            port.mixed_queue.pop_front();
        }

        // OUTSIDE LOCK — important!
        const s32 ret = AudioOut::sceAudioOutOutput(port.audio_out_handle, frame.sample_buffer);

        std::free(frame.sample_buffer);

        if (ret < 0)
            return ret;

        submitted = true;
        return ORBIS_OK;
    };

    // First quick check: if not full, return immediately
    {
        std::scoped_lock lock{port.mutex};
        if (port.mixed_queue.size() < depth) {
            return ORBIS_OK;
        }
    }

    // Submit one frame to free space
    bool submitted = false;
    s32 ret = submit_one_frame(submitted);
    if (ret < 0)
        return ret;

    if (!submitted)
        return ORBIS_OK;

    // ASYNC: free exactly one slot and return
    if (blocking == OrbisAudio3dBlocking::ORBIS_AUDIO3D_BLOCKING_ASYNC) {
        return ORBIS_OK;
    }

    // SYNC: ensure at least one slot is free
    // (drain until size < depth)
    while (true) {
        {
            std::scoped_lock lock{port.mutex};
            if (port.mixed_queue.size() < depth)
                break;
        }

        bool drained = false;
        ret = submit_one_frame(drained);
        if (ret < 0)
            return ret;

        if (!drained)
            break;
    }

    return ORBIS_OK;
}
s32 PS4_SYSV_ABI sceAudio3dPortQueryDebug() {
    LOG_ERROR(Lib_Audio3d, "(STUBBED) called");
    return ORBIS_OK;
}

s32 PS4_SYSV_ABI sceAudio3dPortSetAttribute(const OrbisAudio3dPortId port_id,
                                            const OrbisAudio3dAttributeId attribute_id,
                                            void* attribute, const u64 attribute_size) {
    LOG_INFO(Lib_Audio3d,
             "called, port_id = {}, attribute_id = {}, attribute = {}, attribute_size = {}",
             port_id, static_cast<u32>(attribute_id), attribute, attribute_size);

    if (!state->ports.contains(port_id)) {
        LOG_ERROR(Lib_Audio3d, "!state->ports.contains(port_id)");
        return ORBIS_AUDIO3D_ERROR_INVALID_PORT;
    }

    if (!attribute) {
        LOG_ERROR(Lib_Audio3d, "!attribute");
        return ORBIS_AUDIO3D_ERROR_INVALID_PARAMETER;
    }

    // TODO

    return ORBIS_OK;
}

s32 PS4_SYSV_ABI sceAudio3dReportRegisterHandler() {
    LOG_ERROR(Lib_Audio3d, "(STUBBED) called");
    return ORBIS_OK;
}

s32 PS4_SYSV_ABI sceAudio3dReportUnregisterHandler() {
    LOG_ERROR(Lib_Audio3d, "(STUBBED) called");
    return ORBIS_OK;
}

s32 PS4_SYSV_ABI sceAudio3dSetGpuRenderer() {
    LOG_ERROR(Lib_Audio3d, "(STUBBED) called");
    return ORBIS_OK;
}

s32 PS4_SYSV_ABI sceAudio3dStrError() {
    LOG_ERROR(Lib_Audio3d, "(STUBBED) called");
    return ORBIS_OK;
}

s32 PS4_SYSV_ABI sceAudio3dTerminate() {
    LOG_INFO(Lib_Audio3d, "called");
    if (!state) {
        return ORBIS_AUDIO3D_ERROR_NOT_READY;
    }

    // Close all open ports cleanly.
    std::vector<OrbisAudio3dPortId> port_ids;
    for (const auto& [id, _] : state->ports) {
        port_ids.push_back(id);
    }
    for (const auto id : port_ids) {
        sceAudio3dPortClose(id);
    }

    state.reset();
    return ORBIS_OK;
}

void RegisterLib(Core::Loader::SymbolsResolver* sym) {
    LIB_FUNCTION("pZlOm1aF3aA", "libSceAudio3d", 1, "libSceAudio3d", sceAudio3dAudioOutClose);
    LIB_FUNCTION("ucEsi62soTo", "libSceAudio3d", 1, "libSceAudio3d", sceAudio3dAudioOutOpen);
    LIB_FUNCTION("7NYEzJ9SJbM", "libSceAudio3d", 1, "libSceAudio3d", sceAudio3dAudioOutOutput);
    LIB_FUNCTION("HbxYY27lK6E", "libSceAudio3d", 1, "libSceAudio3d", sceAudio3dAudioOutOutputs);
    LIB_FUNCTION("9tEwE0GV0qo", "libSceAudio3d", 1, "libSceAudio3d", sceAudio3dBedWrite);
    LIB_FUNCTION("xH4Q9UILL3o", "libSceAudio3d", 1, "libSceAudio3d", sceAudio3dBedWrite2);
    LIB_FUNCTION("lvWMW6vEqFU", "libSceAudio3d", 1, "libSceAudio3d", sceAudio3dCreateSpeakerArray);
    LIB_FUNCTION("8hm6YdoQgwg", "libSceAudio3d", 1, "libSceAudio3d", sceAudio3dDeleteSpeakerArray);
    LIB_FUNCTION("Im+jOoa5WAI", "libSceAudio3d", 1, "libSceAudio3d",
                 sceAudio3dGetDefaultOpenParameters);
    LIB_FUNCTION("kEqqyDkmgdI", "libSceAudio3d", 1, "libSceAudio3d",
                 sceAudio3dGetSpeakerArrayMemorySize);
    LIB_FUNCTION("-R1DukFq7Dk", "libSceAudio3d", 1, "libSceAudio3d",
                 sceAudio3dGetSpeakerArrayMixCoefficients);
    LIB_FUNCTION("-Re+pCWvwjQ", "libSceAudio3d", 1, "libSceAudio3d",
                 sceAudio3dGetSpeakerArrayMixCoefficients2);
    LIB_FUNCTION("UmCvjSmuZIw", "libSceAudio3d", 1, "libSceAudio3d", sceAudio3dInitialize);
    LIB_FUNCTION("jO2tec4dJ2M", "libSceAudio3d", 1, "libSceAudio3d", sceAudio3dObjectReserve);
    LIB_FUNCTION("V1FBFpNIAzk", "libSceAudio3d", 1, "libSceAudio3d", sceAudio3dObjectSetAttribute);
    LIB_FUNCTION("4uyHN9q4ZeU", "libSceAudio3d", 1, "libSceAudio3d", sceAudio3dObjectSetAttributes);
    LIB_FUNCTION("1HXxo-+1qCw", "libSceAudio3d", 1, "libSceAudio3d", sceAudio3dObjectUnreserve);
    LIB_FUNCTION("lw0qrdSjZt8", "libSceAudio3d", 1, "libSceAudio3d", sceAudio3dPortAdvance);
    LIB_FUNCTION("OyVqOeVNtSk", "libSceAudio3d", 1, "libSceAudio3d", sceAudio3dPortClose);
    LIB_FUNCTION("UHFOgVNz0kk", "libSceAudio3d", 1, "libSceAudio3d", sceAudio3dPortCreate);
    LIB_FUNCTION("Mw9mRQtWepY", "libSceAudio3d", 1, "libSceAudio3d", sceAudio3dPortDestroy);
    LIB_FUNCTION("ZOGrxWLgQzE", "libSceAudio3d", 1, "libSceAudio3d", sceAudio3dPortFlush);
    LIB_FUNCTION("uJ0VhGcxCTQ", "libSceAudio3d", 1, "libSceAudio3d", sceAudio3dPortFreeState);
    LIB_FUNCTION("9ZA23Ia46Po", "libSceAudio3d", 1, "libSceAudio3d",
                 sceAudio3dPortGetAttributesSupported);
    LIB_FUNCTION("SEggctIeTcI", "libSceAudio3d", 1, "libSceAudio3d", sceAudio3dPortGetList);
    LIB_FUNCTION("flPcUaXVXcw", "libSceAudio3d", 1, "libSceAudio3d", sceAudio3dPortGetParameters);
    LIB_FUNCTION("YaaDbDwKpFM", "libSceAudio3d", 1, "libSceAudio3d", sceAudio3dPortGetQueueLevel);
    LIB_FUNCTION("CKHlRW2E9dA", "libSceAudio3d", 1, "libSceAudio3d", sceAudio3dPortGetState);
    LIB_FUNCTION("iRX6GJs9tvE", "libSceAudio3d", 1, "libSceAudio3d", sceAudio3dPortGetStatus);
    LIB_FUNCTION("XeDDK0xJWQA", "libSceAudio3d", 1, "libSceAudio3d", sceAudio3dPortOpen);
    LIB_FUNCTION("VEVhZ9qd4ZY", "libSceAudio3d", 1, "libSceAudio3d", sceAudio3dPortPush);
    LIB_FUNCTION("-pzYDZozm+M", "libSceAudio3d", 1, "libSceAudio3d", sceAudio3dPortQueryDebug);
    LIB_FUNCTION("Yq9bfUQ0uJg", "libSceAudio3d", 1, "libSceAudio3d", sceAudio3dPortSetAttribute);
    LIB_FUNCTION("QfNXBrKZeI0", "libSceAudio3d", 1, "libSceAudio3d",
                 sceAudio3dReportRegisterHandler);
    LIB_FUNCTION("psv2gbihC1A", "libSceAudio3d", 1, "libSceAudio3d",
                 sceAudio3dReportUnregisterHandler);
    LIB_FUNCTION("yEYXcbAGK14", "libSceAudio3d", 1, "libSceAudio3d", sceAudio3dSetGpuRenderer);
    LIB_FUNCTION("Aacl5qkRU6U", "libSceAudio3d", 1, "libSceAudio3d", sceAudio3dStrError);
    LIB_FUNCTION("WW1TS2iz5yc", "libSceAudio3d", 1, "libSceAudio3d", sceAudio3dTerminate);
};

} // namespace Libraries::Audio3d
