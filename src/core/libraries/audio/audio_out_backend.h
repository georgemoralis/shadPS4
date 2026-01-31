// SPDX-FileCopyrightText: Copyright 2026 shadPS4 Emulator Project
// SPDX-License-Identifier: GPL-2.0-or-later
#include "common/types.h"

struct AudioOutHandle {
    u32 f_userId;
    u32 f_type;
    u32 f_len;
    struct {
        u32 channels;
        bool is_float;
        bool is_std;
        bool is_restricted;
        bool is_mix_to_main;
    } f_param;

    // Virtual function pointers
    bool (*Open)(struct AudioOutHandle* self, const char* device_id);
    void (*SetVolume)(struct AudioOutHandle* self, int channel, int vol);
    void (*SetMixLevelPadSpk)(struct AudioOutHandle* self, int mixLevel);
    uint64_t (*GetLastOutputTime)(struct AudioOutHandle* self);
    int (*Output)(struct AudioOutHandle* self, void* ptr);
    void (*Destroy)(struct AudioOutHandle* self);
} AudioOutHandle;
