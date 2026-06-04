// SPDX-FileCopyrightText: Copyright 2024 shadPS4 Emulator Project
// SPDX-License-Identifier: GPL-2.0-or-later

#pragma once

#include <string_view>

#include "common/types.h"
#include "core/cpu_runtime/guest_state.h"

// ============================================================================
// RuntimeDiagnostics — opt-in JIT corruption-investigation instrumentation
// ============================================================================
//
// This subsystem holds the register-corruption tracers, GPR snapshot ring,
// block-RIP ring, and targeted block dumps that were used to root-cause guest
// register corruption (the Sonic Mania 0x800274c09 / 0x414c40040 arc). They
// are pure observers: they read GuestState at block boundaries and log, but
// never alter execution. (The stuck-RIP watchdog, which *does* alter execution
// by forcing a fatal exit, deliberately stays in the dispatcher — it is a real
// fault condition, not investigation code.)
//
// The entire subsystem compiles out unless SHADPS4_RUNTIME_DIAGNOSTICS is
// defined. When it is not defined, every entry point below is an inline no-op,
// so the dispatcher's calls vanish at compile time and the hot path carries
// zero cost. Enable it for a targeted debugging build:
//
//   cmake -DSHADPS4_RUNTIME_DIAGNOSTICS=ON ...
//
// (or add -DSHADPS4_RUNTIME_DIAGNOSTICS to the compile flags). The dispatcher
// calls these in fixed places; the granular surface keeps each concern (record
// snapshot / detect corruption / inspect HLE return / pre-block dump) separable
// and individually greppable rather than fused into one opaque hook.

namespace Core::Runtime {

class Runtime;  // owns BlockCache/CodeCache; diagnostics borrow it read-only

namespace Diagnostics {

#ifdef SHADPS4_RUNTIME_DIAGNOSTICS

// Emit a one-time beacon proving the diagnostics-enabled runtime is the binary
// actually executing. Call on the first dispatcher entry per thread.
void AnnounceOnce(u64 first_rip);

// Record the full GPR file for this block boundary into the snapshot ring and
// push cur_rip onto the block-RIP ring. Must be called once per dispatcher
// entry, BEFORE CheckRegisterCorruption (which consumes what this records).
void RecordBlockBoundary(GuestState* state, u64 cur_rip);

// Inspect every GPR for a transition into the "impossible"/unmapped address
// gap. On first sighting of a corrupt register, log the producing block (via a
// backward walk of the snapshot ring), dump that block's emitted host code,
// and dump the block-RIP ring so the loop cycle is visible. Reads state
// recorded by RecordBlockBoundary; pass the same cur_rip.
void CheckRegisterCorruption(Runtime* rt, GuestState* state, u64 cur_rip);

// After an HLE bridge call returns, check whether the host stub handed back a
// value in the unmapped gap via RAX — a likely bad-pointer source. Fires once.
void CheckHleReturn(std::string_view name, u64 host_fn, u64 rax);

// Snapshot of the SysV callee-saved GPRs, captured by the HLE bridge just
// before the host call and passed to BridgeCheckCalleeSaved just after.
struct BridgeCalleeSaved {
    u64 rbx, rbp, r12, r13, r14, r15;
};

// Verify the host HLE callee preserved the SysV callee-saved registers
// (rbx/rbp/r12-r15) in GuestState. Any divergence is a real runtime/ABI bug
// (host callee, guest-callback reentrancy, or GuestState mutation) and is
// logged with the exact register, before/after values, and HLE name. Capped.
void BridgeCheckCalleeSaved(std::string_view name, u64 host_fn,
                            const BridgeCalleeSaved& before, GuestState* state);

// Before an HLE bridge call, log the stub name and its six integer-arg registers
// (SysV: rdi/rsi/rdx/rcx/r8/r9) at LOG_ERROR level so they survive in release-
// style logging. This is the active hunt for a stub that receives an output-
// struct pointer it then fails to populate. Gated to a budget so it can't flood.
void LogHleCall(std::string_view name, u64 host_fn, u64 guest_return_addr,
                u64 rdi, u64 rsi, u64 rdx, u64 rcx, u64 r8, u64 r9);

// If the next block to enter is a specific block under investigation, dump all
// guest GPRs at this safe point (one block before the fault handler sees them).
// Fires once for the hard-coded target RIP.
void MaybeDumpPreBlock(GuestState* state);

// Right after a block is compiled, if its guest RIP is one under investigation,
// hex-dump the freshly emitted host bytes (length = emitted_size, capped). Used
// to capture the exact lifted sequence for offline disassembly. Fires once per
// RIP of interest.
void OnBlockCompiled(u64 guest_rip, const void* host_ptr, u64 emitted_size);

// Dump the rolling HLE-call ring and the recent block-RIP ring immediately,
// regardless of fault address. Intended to be called from the signal handler
// at the moment of an unhandled fault, so we capture the HLE-call window that
// preceded ANY crash (works for mid-block faults where the block-entry hooks
// never fire on the exact fault RIP). Safe to call from the handler: reads
// thread-local rings only, no allocation.
void DumpHleRingNow();

#else  // !SHADPS4_RUNTIME_DIAGNOSTICS — all entry points compile to nothing.

inline void AnnounceOnce(u64) {}
inline void RecordBlockBoundary(GuestState*, u64) {}
inline void CheckRegisterCorruption(Runtime*, GuestState*, u64) {}
inline void CheckHleReturn(std::string_view, u64, u64) {}
struct BridgeCalleeSaved {
    u64 rbx, rbp, r12, r13, r14, r15;
};
inline void BridgeCheckCalleeSaved(std::string_view, u64,
                                   const BridgeCalleeSaved&, GuestState*) {}
inline void LogHleCall(std::string_view, u64, u64,
                       u64, u64, u64, u64, u64, u64) {}
inline void MaybeDumpPreBlock(GuestState*) {}
inline void OnBlockCompiled(u64, const void*, u64) {}
inline void DumpHleRingNow() {}

#endif  // SHADPS4_RUNTIME_DIAGNOSTICS

}  // namespace Diagnostics
}  // namespace Core::Runtime
