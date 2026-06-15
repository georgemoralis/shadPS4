// SPDX-FileCopyrightText: Copyright 2026 shadPS4 Emulator Project
// SPDX-License-Identifier: GPL-2.0-or-later
#pragma once
#include <cstddef>
#include "common/types.h"

namespace Core::Runtime {

enum GprIndex : int {
    kRax=0,kRcx=1,kRdx=2,kRbx=3,kRsp=4,kRbp=5,kRsi=6,kRdi=7,
    kR8=8,kR9=9,kR10=10,kR11=11,kR12=12,kR13=13,kR14=14,kR15=15,
};

enum class ExitReason : u32 { BlockEnd=0, HostCall=1, UnsupportedInstruction=2 };

// Shared x86_64 guest register file. r13 points here during block execution.
// Offsets are baked into emitted code, so additions go at the END only.
struct alignas(16) GuestState {
    u64 gpr[16];
    u64 rip;
    u64 rflags;
    u64 fs_base;
    u64 gs_base;
    u32 exit_reason;
    u32 pad_;
    u64 xmm[16][2];   // guest XMM spill (gateway spill deferred; layout reserved now)
};

namespace Offsets {
inline constexpr u32 Gpr        = static_cast<u32>(offsetof(GuestState, gpr));
inline constexpr u32 Rip        = static_cast<u32>(offsetof(GuestState, rip));
inline constexpr u32 Rflags     = static_cast<u32>(offsetof(GuestState, rflags));
inline constexpr u32 FsBase     = static_cast<u32>(offsetof(GuestState, fs_base));
inline constexpr u32 GsBase     = static_cast<u32>(offsetof(GuestState, gs_base));
inline constexpr u32 ExitReason_= static_cast<u32>(offsetof(GuestState, exit_reason));
inline constexpr u32 GprSlot(int i){ return Gpr + static_cast<u32>(i)*8; }
}  // namespace Offsets

// copy-backend-specific constants (the load-op-store backend ignores these).
namespace copyjit {
inline constexpr u64 kFlagsMask = 0x0CD5;  // CF PF AF ZF SF OF DF
inline constexpr int kPinnedGpr[12] = { kRax,kRcx,kRdx,kRbx,kRbp,kRsi,kRdi,kR8,kR9,kR10,kR11,kR12 };
inline constexpr bool IsVirtualizedGuestGpr(int i){ return i==kRsp||i==kR13||i==kR14||i==kR15; }
}  // namespace copyjit

}  // namespace Core::Runtime
