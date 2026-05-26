// SPDX-FileCopyrightText: Copyright 2024 shadPS4 Emulator Project
// SPDX-License-Identifier: GPL-2.0-or-later

#include "core/libraries/avplayer/avplayer_common.h"
#include "core/libraries/avplayer/avplayer_error.h"
#include "core/libraries/avplayer/avplayer_impl.h"
#include "core/tls.h"

#ifdef SHADPS4_USES_RUNTIME
#include <cstdlib>
#include "core/cpu_runtime/runtime.h"
#endif

namespace Libraries::AvPlayer {

#ifdef SHADPS4_USES_RUNTIME
namespace {

// Helper: invoke a guest callback. Dual-context pattern, same as
// pthread-cleanup (see pthread_clean.cpp:posix_pthread_cleanup_pop):
//   - If we're mid-JIT (e.g. AvPlayer wrapper called from an HLE shim
//     that's running guest code through the JIT), reuse the caller's
//     GuestState so the callback runs on the guest's stack.
//   - If we're post-JIT (e.g. AvPlayer worker thread that started on
//     a host start_routine), allocate a fresh stack via malloc.
//
// Returns the callback's RAX value (typically the return value).
//
// The wrappers in this file all dispatch to one of these helpers.
u64 InvokeGuestCallback(u64 guest_fn, u64 a0 = 0, u64 a1 = 0,
                        u64 a2 = 0, u64 a3 = 0) {
    Core::Runtime::GuestState* caller_state =
        Core::Runtime::Runtime::CurrentGuestState();
    if (caller_state != nullptr) {
        return Core::Runtime::Runtime::Instance().CallGuestSimpleOnCallerStack(
            *caller_state, guest_fn, a0, a1, a2, a3);
    }
    // Post-JIT path: AvPlayer worker thread invoking a guest callback.
    // Allocate a per-call stack; 256 KB matches the size used by other
    // post-JIT callback sites (pthread cleanup, ThreadDtors).
    constexpr u64 kCallbackStackSize = 256 * 1024;
    void* guest_stack = std::malloc(kCallbackStackSize);
    if (guest_stack == nullptr) {
        return 0;  // Best-effort; AvPlayer code generally doesn't check.
    }
    void* guest_stack_top = static_cast<u8*>(guest_stack) + kCallbackStackSize;
    const u64 result = Core::Runtime::Runtime::Instance().CallGuestSimple(
        guest_fn, guest_stack_top, a0, a1, a2, a3);
    std::free(guest_stack);
    return result;
}

}  // namespace
#endif  // SHADPS4_USES_RUNTIME

void* PS4_SYSV_ABI AvPlayer::Allocate(void* handle, u32 alignment, u32 size) {
    const auto* const self = reinterpret_cast<AvPlayer*>(handle);
    const auto allocate = self->m_init_data_original.memory_replacement.allocate;
    const auto ptr = self->m_init_data_original.memory_replacement.object_ptr;
#ifdef SHADPS4_USES_RUNTIME
    return reinterpret_cast<void*>(InvokeGuestCallback(
        reinterpret_cast<u64>(allocate),
        reinterpret_cast<u64>(ptr),
        static_cast<u64>(alignment),
        static_cast<u64>(size)));
#else
    return allocate(ptr, alignment, size);
#endif
}

void PS4_SYSV_ABI AvPlayer::Deallocate(void* handle, void* memory) {
    const auto* const self = reinterpret_cast<AvPlayer*>(handle);
    const auto deallocate = self->m_init_data_original.memory_replacement.deallocate;
    const auto ptr = self->m_init_data_original.memory_replacement.object_ptr;
#ifdef SHADPS4_USES_RUNTIME
    InvokeGuestCallback(
        reinterpret_cast<u64>(deallocate),
        reinterpret_cast<u64>(ptr),
        reinterpret_cast<u64>(memory));
#else
    return deallocate(ptr, memory);
#endif
}

void* PS4_SYSV_ABI AvPlayer::AllocateTexture(void* handle, u32 alignment, u32 size) {
    const auto* const self = reinterpret_cast<AvPlayer*>(handle);
    const auto allocate = self->m_init_data_original.memory_replacement.allocate_texture;
    const auto ptr = self->m_init_data_original.memory_replacement.object_ptr;
#ifdef SHADPS4_USES_RUNTIME
    return reinterpret_cast<void*>(InvokeGuestCallback(
        reinterpret_cast<u64>(allocate),
        reinterpret_cast<u64>(ptr),
        static_cast<u64>(alignment),
        static_cast<u64>(size)));
#else
    return allocate(ptr, alignment, size);
#endif
}

void PS4_SYSV_ABI AvPlayer::DeallocateTexture(void* handle, void* memory) {
    const auto* const self = reinterpret_cast<AvPlayer*>(handle);
    const auto deallocate = self->m_init_data_original.memory_replacement.deallocate_texture;
    const auto ptr = self->m_init_data_original.memory_replacement.object_ptr;
#ifdef SHADPS4_USES_RUNTIME
    InvokeGuestCallback(
        reinterpret_cast<u64>(deallocate),
        reinterpret_cast<u64>(ptr),
        reinterpret_cast<u64>(memory));
#else
    return deallocate(ptr, memory);
#endif
}

int PS4_SYSV_ABI AvPlayer::OpenFile(void* handle, const char* filename) {
    auto const self = reinterpret_cast<AvPlayer*>(handle);
    std::lock_guard guard(self->m_file_io_mutex);

    const auto open = self->m_init_data_original.file_replacement.open;
    const auto ptr = self->m_init_data_original.file_replacement.object_ptr;
#ifdef SHADPS4_USES_RUNTIME
    return static_cast<int>(InvokeGuestCallback(
        reinterpret_cast<u64>(open),
        reinterpret_cast<u64>(ptr),
        reinterpret_cast<u64>(filename)));
#else
    return open(ptr, filename);
#endif
}

int PS4_SYSV_ABI AvPlayer::CloseFile(void* handle) {
    auto const self = reinterpret_cast<AvPlayer*>(handle);
    std::lock_guard guard(self->m_file_io_mutex);

    const auto close = self->m_init_data_original.file_replacement.close;
    const auto ptr = self->m_init_data_original.file_replacement.object_ptr;
#ifdef SHADPS4_USES_RUNTIME
    return static_cast<int>(InvokeGuestCallback(
        reinterpret_cast<u64>(close),
        reinterpret_cast<u64>(ptr)));
#else
    return close(ptr);
#endif
}

int PS4_SYSV_ABI AvPlayer::ReadOffsetFile(void* handle, u8* buffer, u64 position, u32 length) {
    auto const self = reinterpret_cast<AvPlayer*>(handle);
    std::lock_guard guard(self->m_file_io_mutex);

    const auto read_offset = self->m_init_data_original.file_replacement.read_offset;
    const auto ptr = self->m_init_data_original.file_replacement.object_ptr;
#ifdef SHADPS4_USES_RUNTIME
    return static_cast<int>(InvokeGuestCallback(
        reinterpret_cast<u64>(read_offset),
        reinterpret_cast<u64>(ptr),
        reinterpret_cast<u64>(buffer),
        position,
        static_cast<u64>(length)));
#else
    return read_offset(ptr, buffer, position, length);
#endif
}

u64 PS4_SYSV_ABI AvPlayer::SizeFile(void* handle) {
    auto const self = reinterpret_cast<AvPlayer*>(handle);
    std::lock_guard guard(self->m_file_io_mutex);

    const auto size = self->m_init_data_original.file_replacement.size;
    const auto ptr = self->m_init_data_original.file_replacement.object_ptr;
#ifdef SHADPS4_USES_RUNTIME
    return InvokeGuestCallback(
        reinterpret_cast<u64>(size),
        reinterpret_cast<u64>(ptr));
#else
    return size(ptr);
#endif
}

AvPlayerInitData AvPlayer::StubInitData(const AvPlayerInitData& data) {
    AvPlayerInitData result = data;
    result.memory_replacement.object_ptr = this;
    result.memory_replacement.allocate = &AvPlayer::Allocate;
    result.memory_replacement.deallocate = &AvPlayer::Deallocate;
    result.memory_replacement.allocate_texture = &AvPlayer::AllocateTexture;
    result.memory_replacement.deallocate_texture = &AvPlayer::DeallocateTexture;
    if (data.file_replacement.open == nullptr || data.file_replacement.close == nullptr ||
        data.file_replacement.read_offset == nullptr || data.file_replacement.size == nullptr) {
        result.file_replacement = {};
    } else {
        result.file_replacement.object_ptr = this;
        result.file_replacement.open = &AvPlayer::OpenFile;
        result.file_replacement.close = &AvPlayer::CloseFile;
        result.file_replacement.read_offset = &AvPlayer::ReadOffsetFile;
        result.file_replacement.size = &AvPlayer::SizeFile;
    }
    return result;
}

AvPlayer::AvPlayer(const AvPlayerInitData& data)
    : m_init_data(StubInitData(data)), m_init_data_original(data),
      m_state(std::make_unique<AvPlayerState>(m_init_data)) {}

s32 AvPlayer::PostInit(const AvPlayerPostInitData& data) {
    m_state->PostInit(data);
    return ORBIS_OK;
}

s32 AvPlayer::AddSource(std::string_view path) {
    return AddSourceEx(path, AvPlayerSourceType::Unknown);
}

s32 AvPlayer::AddSourceEx(std::string_view path, AvPlayerSourceType source_type) {
    if (source_type == AvPlayerSourceType::Unknown) {
        source_type = GetSourceType(path);
    }
    if (source_type == AvPlayerSourceType::Hls) {
        LOG_ERROR(Lib_AvPlayer, "HTTP Live Streaming is not implemented");
        return ORBIS_AVPLAYER_ERROR_NOT_SUPPORTED;
    }
    if (!m_state->AddSource(path, GetSourceType(path))) {
        return ORBIS_AVPLAYER_ERROR_OPERATION_FAILED;
    }
    return ORBIS_OK;
}

s32 AvPlayer::GetStreamCount() {
    if (m_state == nullptr) {
        return ORBIS_AVPLAYER_ERROR_OPERATION_FAILED;
    }
    const auto res = m_state->GetStreamCount();
    if (AVPLAYER_IS_ERROR(res)) {
        return ORBIS_AVPLAYER_ERROR_OPERATION_FAILED;
    }
    return res;
}

s32 AvPlayer::GetStreamInfo(u32 stream_index, AvPlayerStreamInfo& info) {
    if (!m_state->GetStreamInfo(stream_index, info)) {
        return ORBIS_AVPLAYER_ERROR_OPERATION_FAILED;
    }
    return ORBIS_OK;
}

s32 AvPlayer::EnableStream(u32 stream_index) {
    if (m_state == nullptr) {
        return ORBIS_AVPLAYER_ERROR_OPERATION_FAILED;
    }
    if (!m_state->EnableStream(stream_index)) {
        return ORBIS_AVPLAYER_ERROR_OPERATION_FAILED;
    }
    return ORBIS_OK;
}

s32 AvPlayer::Start() {
    if (m_state == nullptr || !m_state->Start()) {
        return ORBIS_AVPLAYER_ERROR_OPERATION_FAILED;
    }
    return ORBIS_OK;
}

s32 AvPlayer::Pause() {
    if (m_state == nullptr || !m_state->Pause()) {
        return ORBIS_AVPLAYER_ERROR_OPERATION_FAILED;
    }
    return ORBIS_OK;
}

s32 AvPlayer::Resume() {
    if (m_state == nullptr || !m_state->Resume()) {
        return ORBIS_AVPLAYER_ERROR_OPERATION_FAILED;
    }
    return ORBIS_OK;
}

s32 AvPlayer::SetAvSyncMode(AvPlayerAvSyncMode sync_mode) {
    if (m_state == nullptr) {
        return ORBIS_AVPLAYER_ERROR_OPERATION_FAILED;
    }
    m_state->SetAvSyncMode(sync_mode);
    return ORBIS_OK;
}

bool AvPlayer::GetVideoData(AvPlayerFrameInfo& video_info) {
    if (m_state == nullptr) {
        return false;
    }
    return m_state->GetVideoData(video_info);
}

bool AvPlayer::GetVideoData(AvPlayerFrameInfoEx& video_info) {
    if (m_state == nullptr) {
        return false;
    }
    return m_state->GetVideoData(video_info);
}

bool AvPlayer::GetAudioData(AvPlayerFrameInfo& audio_info) {
    if (m_state == nullptr) {
        return false;
    }
    return m_state->GetAudioData(audio_info);
}

bool AvPlayer::IsActive() {
    if (m_state == nullptr) {
        return false;
    }
    return m_state->IsActive();
}

u64 AvPlayer::CurrentTime() {
    if (m_state == nullptr) {
        return 0;
    }
    return m_state->CurrentTime();
}

s32 AvPlayer::Stop() {
    if (m_state == nullptr || !m_state->Stop()) {
        return ORBIS_AVPLAYER_ERROR_OPERATION_FAILED;
    }
    return ORBIS_OK;
}

bool AvPlayer::SetLooping(bool is_looping) {
    if (m_state == nullptr) {
        return false;
    }
    return m_state->SetLooping(is_looping);
}

} // namespace Libraries::AvPlayer
