// SPDX-FileCopyrightText: Copyright 2024-2026 shadPS4 Emulator Project
// SPDX-License-Identifier: GPL-2.0-or-later

#include "common/arch.h"
#include "common/assert.h"
#include "common/decoder.h"
#include "common/signal_context.h"
#include "core/cpu_runtime/runtime.h"
#include "core/libraries/kernel/threads/exception.h"
#include "core/signals.h"

// ZydisMnemonicGetString (used to name the faulting guest instruction
// in the crash report) lives here; needed on every x86_64 host.
#ifdef ARCH_X86_64
#include <Zydis/Zydis.h>
#endif

#ifdef _WIN32
#include <windows.h>
#else
#include <csignal>
#include <pthread.h>
#ifdef ARCH_X86_64
#include <Zydis/Formatter.h>
#endif
#endif

#ifndef _WIN32
namespace Libraries::Kernel {
void SigactionHandler(int native_signum, siginfo_t* inf, ucontext_t* raw_context);
extern std::array<OrbisKernelExceptionHandler, 32> Handlers;
} // namespace Libraries::Kernel
#endif

namespace Core {

#if defined(_WIN32)

// Windows invokes a vectored exception handler on whatever stack is current at
// fault time. Under the JIT that is a guest stack (1MB pthread stacks with a
// single guard page, or smaller malloc'd callback stacks); a fault while the
// guest stack is deep leaves no room for the handler's own C++ frames, so the
// handler itself faults writing its locals off the end of the mapped stack.
// POSIX avoids this with SA_ONSTACK; Windows has no equivalent, so we allocate
// a dedicated stack per thread (EnsureVehStack) and switch onto it via the
// CallOnStack asm trampoline (stack.S) before running the real handler body.
extern "C" long CallOnStack(void* stack_top, void* pexp, void* impl);

// Hot lookup for the VEH: TRIVIAL thread_local only. The handler must never
// touch a thread_local with a non-trivial constructor/destructor -- MSVC's
// lazy first-touch registration for those is not async-safe, and a fault
// context is the wrong place to run it. The owning object below is touched
// exclusively from EnsureVehStack (a normal, non-fault context).
static thread_local void* g_veh_stack_top = nullptr;

// Owns the per-thread exception stack and returns it to the OS at thread
// exit. Without this, every thread that ever entered the JIT leaked its
// 512 KiB of COMMITTED stack -- unbounded for guests that churn short-lived
// threads (job systems, audio workers): a thousand create/join cycles was
// half a gigabyte of commit gone for good.
//
// Teardown ordering is the load-bearing detail: later TLS destructors, CRT
// teardown, or detach hooks can still fault on this thread AFTER this
// destructor runs, and the VEH fires for those. So we DISARM FIRST (null
// the trivial hot pointer -- the handler then falls back to the current
// stack, which is the shallow host stack at thread exit, exactly the
// no-dedicated-stack path threads outside the JIT use) and only then
// release the memory. A fault landing between the two steps runs on the
// current stack; a fault during VirtualFree itself likewise. At no point
// can the handler switch onto freed memory.
struct VehStackOwner {
    u8* base = nullptr;
    ~VehStackOwner() {
        g_veh_stack_top = nullptr; // disarm BEFORE free (see above)
        if (base != nullptr) {
            VirtualFree(base, 0, MEM_RELEASE);
            base = nullptr;
        }
    }
};
static thread_local VehStackOwner g_veh_stack_owner;

// The real handler body. Runs on the dedicated exception stack (or, for threads
// that never entered the JIT, on the current stack).
static LONG SignalHandlerImpl(EXCEPTION_POINTERS* pExp) noexcept {
    const auto* signals = Signals::Instance();
    DWORD code = 0;
    PVOID address = nullptr;

    if (pExp != nullptr && pExp->ExceptionRecord != nullptr) {
        code = pExp->ExceptionRecord->ExceptionCode;
        address = pExp->ExceptionRecord->ExceptionAddress;
    }

    bool handled = false;
    switch (code) {
    case EXCEPTION_ACCESS_VIOLATION:
        handled = signals->DispatchAccessViolation(
            pExp, reinterpret_cast<void*>(pExp->ExceptionRecord->ExceptionInformation[1]));
        break;
    case EXCEPTION_ILLEGAL_INSTRUCTION:
        handled = signals->DispatchIllegalInstruction(pExp);
        break;
    case DBG_PRINTEXCEPTION_C:
    case DBG_PRINTEXCEPTION_WIDE_C:
        // Used by OutputDebugString functions.
        return EXCEPTION_CONTINUE_EXECUTION;
    default:
        break;
    }

    if (handled) {
        return EXCEPTION_CONTINUE_EXECUTION;
    }

    // Enrich the unhandled-fault report with CPU-runtime context. If
    // this thread was executing guest code under the JIT, the raw host
    // fault address alone is hard to triage — DescribeFaultContext pins
    // it to the guest RIP that was executing, says whether the fault
    // was inside JIT code, and snapshots the guest GPRs so we can see
    // which register carried the bad pointer. Async-signal-safe.
    //
    // For an access violation, ExceptionInformation[1] is the faulting
    // DATA address (distinct from `address`, the code/instruction
    // address). Logging both is essential: "executing at <code> while
    // dereferencing <data>" — and the data address is what we match
    // against the GPR snapshot to find the offending register.
    const void* data_addr = nullptr;
    if (code == EXCEPTION_ACCESS_VIOLATION && pExp != nullptr &&
        pExp->ExceptionRecord != nullptr &&
        pExp->ExceptionRecord->NumberParameters >= 2) {
        data_addr = reinterpret_cast<const void*>(
            pExp->ExceptionRecord->ExceptionInformation[1]);
    }

#ifdef SHADPS4_USES_RUNTIME
    const auto fctx = Core::Runtime::DescribeFaultContext(address);
    if (fctx.in_runtime) {
        const char* mnem =
#ifdef ARCH_X86_64
            fctx.faulting_insn_decoded
                ? ZydisMnemonicGetString(static_cast<ZydisMnemonic>(fctx.faulting_mnemonic))
                : "<undecoded>";
#else
            "<n/a>";
#endif
        LOG_CRITICAL(Debug,
                     "Unhandled Exception code {:#x} | [diag-v2 hostpc+rawbytes] "
                     "code_addr={} data_addr={} | "
                     "CPU-runtime: guest_rip={:#x} faulting_insn={} (len={}) site={} exit_reason={}",
                     code, address, data_addr, fctx.guest_rip, mnem,
                     fctx.faulting_insn_length,
                     fctx.in_jit_code ? "in-JIT-code" : "not-in-JIT-code (HLE or bad-ptr deref)",
                     fctx.guest_exit_reason);
        if (fctx.have_gprs) {
            // Dump the guest GPRs so the bad-pointer register is
            // identifiable by matching against data_addr.
            LOG_CRITICAL(Debug,
                "  guest GPRs: rax={:#x} rcx={:#x} rdx={:#x} rbx={:#x} "
                "rsp={:#x} rbp={:#x} rsi={:#x} rdi={:#x}",
                fctx.guest_gpr[0], fctx.guest_gpr[1], fctx.guest_gpr[2], fctx.guest_gpr[3],
                fctx.guest_gpr[4], fctx.guest_gpr[5], fctx.guest_gpr[6], fctx.guest_gpr[7]);
            LOG_CRITICAL(Debug,
                "              r8={:#x} r9={:#x} r10={:#x} r11={:#x} "
                "r12={:#x} r13={:#x} r14={:#x} r15={:#x}",
                fctx.guest_gpr[8], fctx.guest_gpr[9], fctx.guest_gpr[10], fctx.guest_gpr[11],
                fctx.guest_gpr[12], fctx.guest_gpr[13], fctx.guest_gpr[14], fctx.guest_gpr[15]);
        }
        // Report the faulting access shape: load vs store and which base
        // register fed the effective address. This tells us whether the
        // bad pointer was being read FROM or used as a write destination.
        if (fctx.have_mem_operand) {
            const char* reg =
#ifdef ARCH_X86_64
                ZydisRegisterGetString(static_cast<ZydisRegister>(fctx.mem_base_reg));
#else
                "<n/a>";
#endif
            LOG_CRITICAL(Debug, "  fault access: {} via base={} (read={} write={})",
                         fctx.mem_is_write ? "WRITE/store" : "READ/load",
                         reg ? reg : "<none>", fctx.mem_is_read, fctx.mem_is_write);
        }
        // Always dump the fault-PC byte window AND the preceding bytes, as
        // hex. The faulting instruction is usually just the consumer of a
        // bad pointer; the instructions before it computed it, so the
        // preceding window is where the actual bug (e.g. a stray bit set
        // in address math) is visible. Disassemble offline with:
        //   echo <bytes> | llvm-mc --disassemble -triple=x86_64
        {
            const char* digits = "0123456789abcdef";
            auto dump = [&](const u8* b, int n, const char* label) {
                if (n <= 0) return;
                char hex[48 * 3 + 1];
                if (n > 48) n = 48;
                for (int i = 0; i < n; ++i) {
                    hex[i * 3 + 0] = digits[(b[i] >> 4) & 0xF];
                    hex[i * 3 + 1] = digits[b[i] & 0xF];
                    hex[i * 3 + 2] = ' ';
                }
                hex[n * 3] = '\0';
                LOG_CRITICAL(Debug, "  {}: {}", label, hex);
            };
            if (fctx.pre_byte_count > 0)
                dump(fctx.pre_fault_bytes, fctx.pre_byte_count, "host bytes BEFORE fault PC");
            if (fctx.raw_byte_count > 0)
                dump(fctx.faulting_raw_bytes, fctx.raw_byte_count, "host bytes AT fault PC");
        }
    } else {
        LOG_CRITICAL(Debug, "Unhandled Exception code {:#x} at {} | not in CPU runtime",
                     code, address);
    }
#else
    LOG_CRITICAL(Debug, "Unhandled Exception code {:#x} at {}", code, address);
#endif
    Common::Log::Flush();

    return EXCEPTION_CONTINUE_SEARCH;
}

// Allocate this thread's dedicated exception stack, once. Called from the JIT
// entry (Runtime::Run) in a NON-fault context where the stack is healthy, so
// the allocation never runs on an already-exhausted stack in a fault context
// -- and, equally, so the non-trivial g_veh_stack_owner thread_local gets its
// lazy first-touch initialization here rather than inside the VEH. Idempotent
// per thread; the owner's destructor returns the stack at thread exit.
void EnsureVehStack() {
    if (g_veh_stack_top != nullptr) {
        return;
    }
    // 512 KiB is generous for the handler (decoder + logging + ring dumps);
    // committed up front so no guard-page growth is needed in a fault context.
    constexpr SIZE_T kVehStackSize = 512 * 1024;
    u8* base = static_cast<u8*>(
        VirtualAlloc(nullptr, kVehStackSize, MEM_COMMIT | MEM_RESERVE, PAGE_READWRITE));
    if (base != nullptr) {
        g_veh_stack_owner.base = base;
        g_veh_stack_top = base + kVehStackSize;
    }
}

// Registered VEH. Thin: switch to this thread's dedicated exception stack (if
// allocated) and run the real handler there, so a deep/near-exhausted guest
// stack at fault time can't starve the handler. Threads that never entered the
// JIT have no dedicated stack and fall back to the current stack (a normal
// large host thread stack), which is fine.
static LONG WINAPI SignalHandler(EXCEPTION_POINTERS* pExp) noexcept {
    void* const top = g_veh_stack_top;
    if (top == nullptr) {
        return SignalHandlerImpl(pExp);
    }
    return CallOnStack(top, pExp, reinterpret_cast<void*>(&SignalHandlerImpl));
}

#else

static std::string DisassembleInstruction(void* code_address) {
    char buffer[256] = "<unable to decode>";

#ifdef ARCH_X86_64
    ZydisDecodedInstruction instruction;
    ZydisDecodedOperand operands[ZYDIS_MAX_OPERAND_COUNT];
    const auto status =
        Common::Decoder::Instance()->decodeInstruction(instruction, operands, code_address);
    if (ZYAN_SUCCESS(status)) {
        ZydisFormatter formatter;
        ZydisFormatterInit(&formatter, ZYDIS_FORMATTER_STYLE_INTEL);
        ZydisFormatterFormatInstruction(&formatter, &instruction, operands,
                                        instruction.operand_count_visible, buffer, sizeof(buffer),
                                        reinterpret_cast<u64>(code_address), ZYAN_NULL);
    }
#endif

    return buffer;
}

void SignalHandler(int sig, siginfo_t* info, void* raw_context) {
    const auto* signals = Signals::Instance();

    auto* code_address = Common::GetRip(raw_context);

    switch (sig) {
    case SIGSEGV:
    case SIGBUS: {
        const bool is_write = Common::IsWriteError(raw_context);
        if (!signals->DispatchAccessViolation(raw_context, info->si_addr)) {
            // If the guest has installed a custom signal handler, and the access violation didn't
            // come from HLE memory tracking, pass the signal on
            if (Libraries::Kernel::Handlers[Libraries::Kernel::NativeToOrbisSignal(sig)]) {
                Libraries::Kernel::SigactionHandler(sig, info,
                                                    reinterpret_cast<ucontext_t*>(raw_context));
                return;
            }
#ifdef SHADPS4_USES_RUNTIME
            const auto fctx = Core::Runtime::DescribeFaultContext(code_address);
            UNREACHABLE_MSG("Unhandled access violation at code address {}: {} address {} | "
                            "CPU-runtime: {} guest_rip={:#x} site={}",
                            fmt::ptr(code_address), is_write ? "Write to" : "Read from",
                            fmt::ptr(info->si_addr),
                            fctx.in_runtime ? "yes" : "no", fctx.guest_rip,
                            fctx.in_jit_code ? "in-JIT-code" : "not-in-JIT-code");
#else
            UNREACHABLE_MSG("Unhandled access violation at code address {}: {} address {}",
                            fmt::ptr(code_address), is_write ? "Write to" : "Read from",
                            fmt::ptr(info->si_addr));
#endif
        }
        break;
    }
    case SIGILL:
        if (!signals->DispatchIllegalInstruction(raw_context)) {
            if (Libraries::Kernel::Handlers[Libraries::Kernel::NativeToOrbisSignal(sig)]) {
                Libraries::Kernel::SigactionHandler(sig, info,
                                                    reinterpret_cast<ucontext_t*>(raw_context));
                return;
            }
#ifdef SHADPS4_USES_RUNTIME
            const auto fctx = Core::Runtime::DescribeFaultContext(code_address);
            UNREACHABLE_MSG("Unhandled illegal instruction at code address {}: {} | "
                            "CPU-runtime: {} guest_rip={:#x} site={}",
                            fmt::ptr(code_address), DisassembleInstruction(code_address),
                            fctx.in_runtime ? "yes" : "no", fctx.guest_rip,
                            fctx.in_jit_code ? "in-JIT-code" : "not-in-JIT-code");
#else
            UNREACHABLE_MSG("Unhandled illegal instruction at code address {}: {}",
                            fmt::ptr(code_address), DisassembleInstruction(code_address));
#endif
        }
        break;
    default:
        if (sig == SIGSLEEP) {
            // Sleep thread until signal is received again
            sigset_t sigset;
            sigemptyset(&sigset);
            sigaddset(&sigset, SIGSLEEP);
            sigwait(&sigset, &sig);
        }
        break;
    }
}

#endif

SignalDispatch::SignalDispatch() {
#if defined(_WIN32)
    ASSERT_MSG(handle = AddVectoredExceptionHandler(0, SignalHandler),
               "Failed to register exception handler.");
#else
    struct sigaction action{};
    action.sa_sigaction = SignalHandler;
    action.sa_flags = SA_SIGINFO | SA_ONSTACK;
    sigemptyset(&action.sa_mask);

    ASSERT_MSG(sigaction(SIGSEGV, &action, nullptr) == 0 &&
                   sigaction(SIGBUS, &action, nullptr) == 0,
               "Failed to register access violation signal handler.");
    ASSERT_MSG(sigaction(SIGILL, &action, nullptr) == 0,
               "Failed to register illegal instruction signal handler.");
    ASSERT_MSG(sigaction(SIGSLEEP, &action, nullptr) == 0,
               "Failed to register sleep signal handler.");
#endif
}

SignalDispatch::~SignalDispatch() {
#if defined(_WIN32)
    ASSERT_MSG(RemoveVectoredExceptionHandler(handle), "Failed to remove exception handler.");
#else
    struct sigaction action{};
    action.sa_handler = SIG_DFL;
    action.sa_flags = 0;
    sigemptyset(&action.sa_mask);

    ASSERT_MSG(sigaction(SIGSEGV, &action, nullptr) == 0 &&
                   sigaction(SIGBUS, &action, nullptr) == 0,
               "Failed to remove access violation signal handler.");
    ASSERT_MSG(sigaction(SIGILL, &action, nullptr) == 0,
               "Failed to remove illegal instruction signal handler.");
#endif
}

bool SignalDispatch::DispatchAccessViolation(void* context, void* fault_address) const {
    for (const auto& [handler, _] : access_violation_handlers) {
        if (handler(context, fault_address)) {
            return true;
        }
    }
    return false;
}

bool SignalDispatch::DispatchIllegalInstruction(void* context) const {
    for (const auto& [handler, _] : illegal_instruction_handlers) {
        if (handler(context)) {
            return true;
        }
    }
    return false;
}

} // namespace Core