// SPDX-FileCopyrightText: Copyright 2024 shadPS4 Emulator Project
// SPDX-License-Identifier: GPL-2.0-or-later

#include "core/libraries/kernel/threads/pthread.h"
#include "core/libraries/libs.h"

#ifdef SHADPS4_USES_RUNTIME
#include <cstdlib>
#include "core/cpu_runtime/runtime.h"
#endif

namespace Libraries::Kernel {

void PS4_SYSV_ABI __pthread_cleanup_push_imp(PthreadCleanupFunc routine, void* arg,
                                             PthreadCleanup* newbuf) {
    newbuf->routine = routine;
    newbuf->routine_arg = arg;
    newbuf->onheap = 0;
    g_curthread->cleanup.push_front(newbuf);
}

void PS4_SYSV_ABI posix_pthread_cleanup_push(PthreadCleanupFunc routine, void* arg) {
    Pthread* curthread = g_curthread;
    auto* newbuf = new (std::nothrow) PthreadCleanup{};
    if (newbuf == nullptr) {
        return;
    }

    newbuf->routine = routine;
    newbuf->routine_arg = arg;
    newbuf->onheap = 1;
    curthread->cleanup.push_front(newbuf);
}

void PS4_SYSV_ABI posix_pthread_cleanup_pop(int execute) {
    Pthread* curthread = g_curthread;
    if (!curthread->cleanup.empty()) {
        PthreadCleanup* old = curthread->cleanup.front();
        curthread->cleanup.pop_front();
        if (execute) {
#ifdef SHADPS4_USES_RUNTIME
            // Dual-context invocation:
            //   - Mid-JIT (guest called pthread_cleanup_pop): caller has
            //     active GuestState. Reuse the caller's stack so the
            //     cleanup routine sees the calling guest's frame chain.
            //   - Post-JIT (called from posix_pthread_exit's cleanup
            //     loop after start_routine returned): no caller state.
            //     Allocate a fresh stack, same pattern as ThreadDtors.
            Core::Runtime::GuestState* caller_state =
                Core::Runtime::Runtime::CurrentGuestState();
            if (caller_state != nullptr) {
                Core::Runtime::Runtime::Instance().CallGuestSimpleOnCallerStack(
                    *caller_state,
                    reinterpret_cast<u64>(old->routine),
                    reinterpret_cast<u64>(old->routine_arg));
            } else {
                constexpr u64 kCleanupStackSize = 256 * 1024;
                void* guest_stack = std::malloc(kCleanupStackSize);
                if (guest_stack != nullptr) {
                    void* guest_stack_top =
                        static_cast<u8*>(guest_stack) + kCleanupStackSize;
                    Core::Runtime::Runtime::Instance().CallGuestSimple(
                        reinterpret_cast<u64>(old->routine),
                        guest_stack_top,
                        reinterpret_cast<u64>(old->routine_arg));
                    std::free(guest_stack);
                } else {
                    LOG_ERROR(Lib_Kernel,
                              "pthread cleanup: failed to allocate guest stack");
                }
            }
#else
            old->routine(old->routine_arg);
#endif
        }
        if (old->onheap) {
            delete old;
        }
    }
}

void RegisterPthreadClean(Core::Loader::SymbolsResolver* sym) {
    // Posix
    LIB_FUNCTION("4ZeZWcMsAV0", "libScePosix", 1, "libkernel", posix_pthread_cleanup_push);
    LIB_FUNCTION("RVxb0Ssa5t0", "libScePosix", 1, "libkernel", posix_pthread_cleanup_pop);

    // Posix-Kernel
    LIB_FUNCTION("1xvtUVx1-Sg", "libkernel", 1, "libkernel", __pthread_cleanup_push_imp);
    LIB_FUNCTION("iWsFlYMf3Kw", "libkernel", 1, "libkernel", posix_pthread_cleanup_pop);
}

} // namespace Libraries::Kernel
