// SPDX-FileCopyrightText: Copyright 2026 shadPS4 Emulator Project
// SPDX-License-Identifier: GPL-2.0-or-later
//
// runtime_copy backend: x86_64 guest -> x86_64 host via verbatim block-copy plus
// a small mangler for the instructions that touch reserved machinery. Owns the
// gateway (call/ret transport) and the HLE bridge.

#include <new>
#define XBYAK_NO_OP_NAMES
#include <xbyak/xbyak.h>
#include <Zydis/Zydis.h>

#include "common/types.h"
#include "core/cpu_runtime/code_cache.h"
#include "core/cpu_runtime/copy_gate.h"
#include "core/cpu_runtime/cpu_backend.h"
#include "core/cpu_runtime/guest_state.h"

namespace Core::Runtime {
namespace {

using namespace Xbyak::util;
namespace cj = ::Core::Runtime::copyjit;

constexpr u64 BLOCK_HOST_CAP = 16384;
constexpr u64 BLOCK_GUEST_CAP = 4096;

const Xbyak::Reg64* HostReg(int idx) {
    static const Xbyak::Reg64* t[16] = {&rax,&rcx,&rdx,&rbx,nullptr,&rbp,&rsi,&rdi,
                                         &r8,&r9,&r10,&r11,&r12,nullptr,nullptr,nullptr};
    return t[idx];
}
int IdxOf(ZydisRegister r) {
    return static_cast<int>(ZydisRegisterGetLargestEnclosing(ZYDIS_MACHINE_MODE_LONG_64, r)) -
           static_cast<int>(ZYDIS_REGISTER_RAX);
}
constexpr u32 RspSlot() { return Offsets::GprSlot(kRsp); }

bool EmitHostJcc(Xbyak::CodeGenerator& c, ZydisMnemonic m, const Xbyak::Label& t) {
    switch (m) {
        case ZYDIS_MNEMONIC_JZ:c.je(t);return true;   case ZYDIS_MNEMONIC_JNZ:c.jne(t);return true;
        case ZYDIS_MNEMONIC_JL:c.jl(t);return true;   case ZYDIS_MNEMONIC_JLE:c.jle(t);return true;
        case ZYDIS_MNEMONIC_JNL:c.jge(t);return true; case ZYDIS_MNEMONIC_JNLE:c.jg(t);return true;
        case ZYDIS_MNEMONIC_JB:c.jb(t);return true;   case ZYDIS_MNEMONIC_JBE:c.jbe(t);return true;
        case ZYDIS_MNEMONIC_JNB:c.jae(t);return true; case ZYDIS_MNEMONIC_JNBE:c.ja(t);return true;
        case ZYDIS_MNEMONIC_JS:c.js(t);return true;   case ZYDIS_MNEMONIC_JNS:c.jns(t);return true;
        case ZYDIS_MNEMONIC_JO:c.jo(t);return true;   case ZYDIS_MNEMONIC_JNO:c.jno(t);return true;
        case ZYDIS_MNEMONIC_JP:c.jp(t);return true;   case ZYDIS_MNEMONIC_JNP:c.jnp(t);return true;
        default:return false;
    }
}
void StoreRipExit(Xbyak::CodeGenerator& c, const Xbyak::Reg64& rip_src, ExitReason r) {
    c.mov(qword[r13+Offsets::Rip], rip_src);
    c.mov(dword[r13+Offsets::ExitReason_], static_cast<u32>(r));
    c.ret();
}
void EmitExitImm(Xbyak::CodeGenerator& c, u64 rip, ExitReason r) {
    c.mov(r14, rip); StoreRipExit(c, r14, r);
}

// ---- stack manglers (inline; return false -> caller emits Unsupported) ----
bool EmitPush(Xbyak::CodeGenerator& c, const ZydisDecodedOperand* ops) {
    const auto& s = ops[0];
    c.mov(r14, qword[r13+RspSlot()]);
    if (s.type==ZYDIS_OPERAND_TYPE_REGISTER) {
        if (s.size!=64) return false;
        int gi=IdxOf(s.reg.value);
        if (gi==kRsp){ c.mov(r15,r14); c.lea(r14,ptr[r14-8]); c.mov(qword[r13+RspSlot()],r14); c.mov(qword[r14],r15); return true; }
        if (const auto* hr=HostReg(gi)){ c.lea(r14,ptr[r14-8]); c.mov(qword[r13+RspSlot()],r14); c.mov(qword[r14],*hr); return true; }
        c.mov(r15, qword[r13+Offsets::GprSlot(gi)]); c.lea(r14,ptr[r14-8]); c.mov(qword[r13+RspSlot()],r14); c.mov(qword[r14],r15); return true;
    }
    if (s.type==ZYDIS_OPERAND_TYPE_IMMEDIATE){ c.lea(r14,ptr[r14-8]); c.mov(qword[r13+RspSlot()],r14); c.mov(qword[r14],(u64)s.imm.value.s); return true; }
    return false;
}
bool EmitPop(Xbyak::CodeGenerator& c, const ZydisDecodedOperand* ops) {
    const auto& d = ops[0];
    if (d.type!=ZYDIS_OPERAND_TYPE_REGISTER || d.size!=64) return false;
    int gi=IdxOf(d.reg.value);
    c.mov(r14, qword[r13+RspSlot()]); c.mov(r15, qword[r14]);
    if (gi==kRsp){ c.mov(qword[r13+RspSlot()],r15); return true; }
    c.lea(r14,ptr[r14+8]); c.mov(qword[r13+RspSlot()],r14);
    if (const auto* hr=HostReg(gi)) c.mov(*hr,r15); else c.mov(qword[r13+Offsets::GprSlot(gi)],r15);
    return true;
}
bool EmitLeave(Xbyak::CodeGenerator& c) {
    c.mov(r14,rbp); c.mov(rbp,qword[r14]); c.lea(r14,ptr[r14+8]); c.mov(qword[r13+RspSlot()],r14); return true;
}
// ---- rip-relative + fs/gs manglers (re-encode via Zydis encoder) ----
bool ReEncodeToR14(const ZydisDecodedInstruction& insn, const ZydisDecodedOperand* ops,
                   bool strip_segment, Xbyak::CodeGenerator& c) {
    ZydisEncoderRequest req;
    if (!ZYAN_SUCCESS(ZydisEncoderDecodedInstructionToEncoderRequest(&insn,ops,insn.operand_count_visible,&req)))
        return false;
    if (strip_segment)
        req.prefixes &= ~(ZYDIS_ATTRIB_HAS_SEGMENT_FS|ZYDIS_ATTRIB_HAS_SEGMENT_GS|ZYDIS_ATTRIB_HAS_SEGMENT);
    for (ZyanU8 i=0;i<req.operand_count;i++){
        auto& o=req.operands[i];
        if (o.type==ZYDIS_OPERAND_TYPE_MEMORY){ o.mem.base=ZYDIS_REGISTER_R14; o.mem.index=ZYDIS_REGISTER_NONE; o.mem.scale=0; o.mem.displacement=0; }
    }
    ZyanU8 buf[ZYDIS_MAX_INSTRUCTION_LENGTH]; ZyanUSize n=sizeof buf;
    if (!ZYAN_SUCCESS(ZydisEncoderEncodeInstruction(&req,buf,&n))) return false;
    c.db(buf,(size_t)n); return true;
}
bool EmitRipRel(Xbyak::CodeGenerator& c, const ZydisDecodedInstruction& insn,
                const ZydisDecodedOperand* ops, u64 rip) {
    int memidx=-1;
    for (ZyanU8 i=0;i<insn.operand_count;i++){
        const auto& o=ops[i];
        if (o.type==ZYDIS_OPERAND_TYPE_REGISTER && cj::detail::IsReservedEnclosing(o.reg.value)) return false;
        if (o.type==ZYDIS_OPERAND_TYPE_MEMORY){
            if (o.mem.base==ZYDIS_REGISTER_RIP||o.mem.base==ZYDIS_REGISTER_EIP) memidx=i;
            else if (cj::detail::IsReservedEnclosing(o.mem.base)||cj::detail::IsReservedEnclosing(o.mem.index)) return false;
            if (o.mem.segment==ZYDIS_REGISTER_FS||o.mem.segment==ZYDIS_REGISTER_GS) return false;
        }
    }
    if (memidx<0) return false;
    ZyanU64 abs=0;
    if (!ZYAN_SUCCESS(ZydisCalcAbsoluteAddress(&insn,&ops[memidx],rip,&abs))) return false;
    c.mov(r14,(u64)abs);
    return ReEncodeToR14(insn,ops,/*strip_segment=*/false,c);
}
bool EmitSegFsGs(Xbyak::CodeGenerator& c, const ZydisDecodedInstruction& insn,
                 const ZydisDecodedOperand* ops) {
    int memidx=-1;
    for (ZyanU8 i=0;i<insn.operand_count;i++){
        const auto& o=ops[i];
        if (o.type==ZYDIS_OPERAND_TYPE_REGISTER && cj::detail::IsReservedEnclosing(o.reg.value)) return false;
        if (o.type==ZYDIS_OPERAND_TYPE_MEMORY && (o.mem.segment==ZYDIS_REGISTER_FS||o.mem.segment==ZYDIS_REGISTER_GS)) memidx=i;
    }
    if (memidx<0) return false;
    const auto& m=ops[memidx].mem;
    if (m.base==ZYDIS_REGISTER_RIP||m.base==ZYDIS_REGISTER_EIP) return false;
    c.mov(r14, qword[r13 + (m.segment==ZYDIS_REGISTER_FS?Offsets::FsBase:Offsets::GsBase)]);
    if (m.disp.value!=0) c.lea(r14, ptr[r14+(int)m.disp.value]);
    if (m.base!=ZYDIS_REGISTER_NONE){ int gi=IdxOf(m.base);
        if (const auto* hr=HostReg(gi)) c.lea(r14,ptr[r14+*hr]); else { c.mov(r15,qword[r13+Offsets::GprSlot(gi)]); c.lea(r14,ptr[r14+r15]); } }
    if (m.index!=ZYDIS_REGISTER_NONE){ int gi=IdxOf(m.index);
        if (const auto* hr=HostReg(gi)) c.lea(r14,ptr[r14+(*hr)*(int)m.scale]); else { c.mov(r15,qword[r13+Offsets::GprSlot(gi)]); c.lea(r14,ptr[r14+r15*(int)m.scale]); } }
    return ReEncodeToR14(insn,ops,/*strip_segment=*/true,c);
}

// ---- generic virtualized-register / RIP-relative mangler ----
// Handles non-stack, non-control-flow instructions that name any virtualized GPR
// (RSP/R13/R14/R15) as a register operand or SIB base/index, and/or use a
// RIP-relative memory operand. Each need binds a scratch host reg (r14 then r15):
//   * virtualized reg  -> preload its GuestState slot (if read), substitute, and
//                         store the slot back (if written);
//   * RIP-relative mem -> materialize the absolute target (next_rip + disp) into
//                         the scratch and rewrite the operand to [scratch].
// The instruction then runs verbatim (flags match the guest). At most two scratch
// needs total. Bails (return false -> UnsupportedInstruction) on: >2 scratch
// needs, sub-64-bit virt operands, or FS/GS memory combined with a reserved reg.
bool EmitVirtReg(Xbyak::CodeGenerator& c, const ZydisDecodedInstruction& insn,
                 const ZydisDecodedOperand* ops, u64 rip) {
    constexpr ZyanU8 kRead  = ZYDIS_OPERAND_ACTION_READ  | ZYDIS_OPERAND_ACTION_CONDREAD;
    constexpr ZyanU8 kWrite = ZYDIS_OPERAND_ACTION_WRITE | ZYDIS_OPERAND_ACTION_CONDWRITE;
    auto encl = [](ZydisRegister r) -> int { return r==ZYDIS_REGISTER_NONE ? -1 : IdxOf(r); };
    const u64 next_rip = rip + insn.length;

    struct Bind { int gi; bool rd, wr, is_rip; u64 abs; };
    Bind binds[2]; int nb=0; bool found=false;
    const Xbyak::Reg64* scratch[2]   = {&r14,&r15};
    const ZydisRegister scratch_z[2] = {ZYDIS_REGISTER_R14, ZYDIS_REGISTER_R15};
    auto bindGpr = [&](int gi,bool r,bool w)->int{
        for(int k=0;k<nb;k++) if(!binds[k].is_rip && binds[k].gi==gi){ binds[k].rd|=r; binds[k].wr|=w; return k; }
        if(nb>=2) return -1;
        binds[nb]={gi,r,w,false,0}; return nb++;
    };
    auto bindRip = [&](u64 abs)->int{
        for(int k=0;k<nb;k++) if(binds[k].is_rip) return k;
        if(nb>=2) return -1;
        binds[nb]={-1,true,false,true,abs}; return nb++;
    };

    for (ZyanU8 i=0;i<insn.operand_count_visible;i++){
        const auto& op = ops[i];
        if (op.type==ZYDIS_OPERAND_TYPE_REGISTER){
            if (cj::detail::IsReservedEnclosing(op.reg.value)){
                if (ZydisRegisterGetWidth(ZYDIS_MACHINE_MODE_LONG_64, op.reg.value)!=64) return false;
                if (bindGpr(encl(op.reg.value), (op.actions&kRead)!=0, (op.actions&kWrite)!=0)<0) return false;
                found=true;
            }
        } else if (op.type==ZYDIS_OPERAND_TYPE_MEMORY){
            if (op.mem.base==ZYDIS_REGISTER_RIP || op.mem.base==ZYDIS_REGISTER_EIP){
                if (bindRip(next_rip + (u64)op.mem.disp.value)<0) return false;
                found=true;
            } else {
                if (op.mem.segment==ZYDIS_REGISTER_FS || op.mem.segment==ZYDIS_REGISTER_GS) return false;
                if (cj::detail::IsReservedEnclosing(op.mem.base)){  if(bindGpr(encl(op.mem.base), true,false)<0)return false; found=true; }
                if (cj::detail::IsReservedEnclosing(op.mem.index)){ if(bindGpr(encl(op.mem.index),true,false)<0)return false; found=true; }
            }
        }
    }
    if (!found) return false;  // reserved reg only in a hidden/implicit operand (e.g. pushfq) -> defer

    for (int k=0;k<nb;k++){
        if (binds[k].is_rip)   c.mov(*scratch[k], binds[k].abs);
        else if (binds[k].rd)  c.mov(*scratch[k], qword[r13+Offsets::GprSlot(binds[k].gi)]);
    }

    ZydisEncoderRequest req;
    if (!ZYAN_SUCCESS(ZydisEncoderDecodedInstructionToEncoderRequest(&insn,ops,insn.operand_count_visible,&req)))
        return false;
    auto substReg = [&](ZydisRegister& reg){
        if (reg==ZYDIS_REGISTER_NONE) return;
        const int gi=IdxOf(reg);
        for (int k=0;k<nb;k++) if(!binds[k].is_rip && binds[k].gi==gi){ reg=scratch_z[k]; return; }
    };
    int rip_k=-1; for(int k=0;k<nb;k++) if(binds[k].is_rip) rip_k=k;
    for (ZyanU8 j=0;j<req.operand_count;j++){
        auto& o=req.operands[j];
        if (o.type==ZYDIS_OPERAND_TYPE_REGISTER) substReg(o.reg.value);
        else if (o.type==ZYDIS_OPERAND_TYPE_MEMORY){
            if ((o.mem.base==ZYDIS_REGISTER_RIP || o.mem.base==ZYDIS_REGISTER_EIP) && rip_k>=0){
                o.mem.base=scratch_z[rip_k]; o.mem.index=ZYDIS_REGISTER_NONE; o.mem.scale=0; o.mem.displacement=0;
            } else { substReg(o.mem.base); substReg(o.mem.index); }
        }
    }
    ZyanU8 buf[ZYDIS_MAX_INSTRUCTION_LENGTH]; ZyanUSize n=sizeof buf;
    if (!ZYAN_SUCCESS(ZydisEncoderEncodeInstruction(&req,buf,&n))) return false;
    c.db(buf,(size_t)n);

    for (int k=0;k<nb;k++) if(!binds[k].is_rip && binds[k].wr)
        c.mov(qword[r13+Offsets::GprSlot(binds[k].gi)], *scratch[k]);
    return true;
}

// ---- gateway: dispatcher CALLs a block; block RETs after terminator ----
// The gateway is the host->guest boundary, so its ENTRY follows the HOST C ABI:
// params (state, block) arrive in rcx/rdx on Windows (MS x64) and rdi/rsi on
// SysV. MS x64 additionally treats rdi/rsi as callee-saved, and the gateway
// loads guest values into them, so on Windows we must save/restore them too.
struct GatewayGen : Xbyak::CodeGenerator {
    GatewayGen() {
#ifdef _WIN32
        push(rbx);push(rbp);push(rdi);push(rsi);push(r12);push(r13);push(r14);push(r15);
        mov(r13,rcx); mov(r14,rdx);            // MS x64: 1st=rcx (state), 2nd=rdx (block)
#else
        push(rbx);push(rbp);push(r12);push(r13);push(r14);push(r15);
        mov(r13,rdi); mov(r14,rsi);            // SysV: 1st=rdi (state), 2nd=rsi (block)
#endif
        mov(rax,ptr[r13+Offsets::Rflags]); and_(rax,cj::kFlagsMask); push(rax); popfq();
        for (int gi:cj::kPinnedGpr) mov(*HostReg(gi),ptr[r13+Offsets::GprSlot(gi)]);
        lea(rsp,ptr[rsp-8]); call(r14); lea(rsp,ptr[rsp+8]);
        for (int gi:cj::kPinnedGpr) mov(ptr[r13+Offsets::GprSlot(gi)],*HostReg(gi));
        pushfq(); pop(rax); and_(rax,cj::kFlagsMask); mov(ptr[r13+Offsets::Rflags],rax);
        db(0xFC);  // CLD
#ifdef _WIN32
        pop(r15);pop(r14);pop(r13);pop(r12);pop(rsi);pop(rdi);pop(rbp);pop(rbx); ret();
#else
        pop(r15);pop(r14);pop(r13);pop(r12);pop(rbp);pop(rbx); ret();
#endif
    }
};
// ---- HLE bridge: invoked when rip is a host function (integer/pointer args) ----
// Bridge entry also follows the host C ABI for its single `state` param. The
// HLE host function it calls is PS4_SYSV_ABI, so its arg regs (rdi/rsi/rdx/
// rcx/r8/r9) and 16-byte call alignment are SysV on every host -- only the
// gateway/bridge *entry* differs by host OS.
struct BridgeGen : Xbyak::CodeGenerator {
    BridgeGen() {
#ifdef _WIN32
        push(rbp);push(rbx);push(rdi);push(rsi);push(r12);push(r13);push(r14);push(r15);
        mov(r13,rcx);                          // MS x64: state in rcx
#else
        push(rbp);push(rbx);push(r12);push(r13);push(r14);push(r15);
        mov(r13,rdi);                          // SysV: state in rdi
#endif
        mov(rax,qword[r13+Offsets::GprSlot(kRsp)]); mov(r12,qword[rax]);  // pop guest return addr
        lea(rax,ptr[rax+8]); mov(qword[r13+Offsets::GprSlot(kRsp)],rax);
        mov(r15,qword[r13+Offsets::Rip]);                                 // host fn addr
        mov(rdi,qword[r13+Offsets::GprSlot(kRdi)]); mov(rsi,qword[r13+Offsets::GprSlot(kRsi)]);
        mov(rdx,qword[r13+Offsets::GprSlot(kRdx)]); mov(rcx,qword[r13+Offsets::GprSlot(kRcx)]);
        mov(r8, qword[r13+Offsets::GprSlot(kR8)]);  mov(r9, qword[r13+Offsets::GprSlot(kR9)]);
        sub(rsp,8); call(r15); add(rsp,8);
        mov(qword[r13+Offsets::GprSlot(kRax)],rax); mov(qword[r13+Offsets::GprSlot(kRdx)],rdx);
        mov(qword[r13+Offsets::Rip],r12);
        mov(dword[r13+Offsets::ExitReason_],(u32)ExitReason::BlockEnd);
#ifdef _WIN32
        pop(r15);pop(r14);pop(r13);pop(r12);pop(rsi);pop(rdi);pop(rbx);pop(rbp); ret();
#else
        pop(r15);pop(r14);pop(r13);pop(r12);pop(rbx);pop(rbp); ret();
#endif
    }
};

class X64CopyBackend final : public Backend {
public:
    explicit X64CopyBackend(CodeCache& cc) : cc_(cc) {
        ZydisDecoderInit(&dec_, ZYDIS_MACHINE_MODE_LONG_64, ZYDIS_STACK_WIDTH_64);
    }

    void* CompileBlock(u64 guest_rip) override try {
        u8* buf = cc_.Allocate(BLOCK_HOST_CAP);
        if (!buf) return nullptr;
        Xbyak::CodeGenerator c{BLOCK_HOST_CAP, buf};
        u64 rip = guest_rip; const u64 cap = guest_rip + BLOCK_GUEST_CAP; bool term = false;
        while (rip < cap) {
            ZydisDecodedInstruction insn; ZydisDecodedOperand ops[ZYDIS_MAX_OPERAND_COUNT];
            if (!ZYAN_SUCCESS(ZydisDecoderDecodeFull(&dec_, reinterpret_cast<const void*>(rip), 15, &insn, ops))) {
                EmitExitImm(c, rip, ExitReason::UnsupportedInstruction); term = true; break; }
            const u64 next_rip = rip + insn.length;
            const cj::CopyVerdict v = cj::ClassifyDecoded(insn, ops, nullptr);
            if (v == cj::CopyVerdict::kAccept) { c.db(reinterpret_cast<const u8*>(rip), insn.length); rip = next_rip; continue; }
            if (v == cj::CopyVerdict::kControlFlow) { EmitTerminator(c, insn, ops, rip, next_rip); term = true; break; }
            bool handled = false;
            switch (insn.mnemonic) {
                case ZYDIS_MNEMONIC_PUSH:  handled = EmitPush(c, ops); break;
                case ZYDIS_MNEMONIC_POP:   handled = EmitPop(c, ops); break;
                case ZYDIS_MNEMONIC_LEAVE: handled = EmitLeave(c); break;
                default: break;
            }
            if (!handled && (v == cj::CopyVerdict::kReservedReg || v == cj::CopyVerdict::kRipRelative))
                handled = EmitVirtReg(c, insn, ops, rip);
            if (!handled && v == cj::CopyVerdict::kRipRelative) handled = EmitRipRel(c, insn, ops, rip);
            if (!handled && v == cj::CopyVerdict::kSegmentFsGs) handled = EmitSegFsGs(c, insn, ops);
            if (handled) { rip = next_rip; continue; }
            EmitExitImm(c, rip, ExitReason::UnsupportedInstruction); term = true; break;
        }
        if (!term) EmitExitImm(c, rip, ExitReason::BlockEnd);
        cc_.ReturnTail(buf, BLOCK_HOST_CAP, c.getSize());
        ++blocks_compiled_;
        return buf;
    } catch (const std::exception&) { return nullptr; }

    u64 BlockReservationSize() const noexcept override { return BLOCK_HOST_CAP; }
    EnterBlockFn GetEnterBlock() const noexcept override { return gateway_.getCode<EnterBlockFn>(); }
    HleBridgeFn  GetHleBridge() const noexcept override { return bridge_.getCode<HleBridgeFn>(); }
    CpuBackend   Kind() const noexcept override { return CpuBackend::RuntimeCopy; }

private:
    void EmitTerminator(Xbyak::CodeGenerator& c, const ZydisDecodedInstruction& insn,
                        const ZydisDecodedOperand* ops, u64 rip, u64 next_rip) {
        const auto cat = insn.meta.category;
        auto rel = [&]{ ZyanU64 t=0; ZydisCalcAbsoluteAddress(&insn,&ops[0],rip,&t); return (u64)t; };
        if (cat==ZYDIS_CATEGORY_UNCOND_BR){
            if (ops[0].type==ZYDIS_OPERAND_TYPE_IMMEDIATE) EmitExitImm(c, rel(), ExitReason::BlockEnd);
            else if (ops[0].type==ZYDIS_OPERAND_TYPE_REGISTER){ int gi=IdxOf(ops[0].reg.value);
                if (const auto* hr=HostReg(gi)) StoreRipExit(c,*hr,ExitReason::BlockEnd);
                else { c.mov(r14,qword[r13+Offsets::GprSlot(gi)]); StoreRipExit(c,r14,ExitReason::BlockEnd); } }
            else EmitExitImm(c, rip, ExitReason::UnsupportedInstruction);
            return;
        }
        if (cat==ZYDIS_CATEGORY_COND_BR){
            Xbyak::Label taken,done;
            if (!EmitHostJcc(c,insn.mnemonic,taken)){ EmitExitImm(c,rip,ExitReason::UnsupportedInstruction); return; }
            const u64 tgt=rel();
            c.mov(r14,next_rip); c.jmp(done); c.L(taken); c.mov(r14,tgt); c.L(done);
            StoreRipExit(c,r14,ExitReason::BlockEnd); return;
        }
        if (cat==ZYDIS_CATEGORY_CALL){
            c.mov(r14,qword[r13+RspSlot()]); c.lea(r14,ptr[r14-8]); c.mov(qword[r13+RspSlot()],r14);
            c.mov(r15,next_rip); c.mov(qword[r14],r15);
            if (ops[0].type==ZYDIS_OPERAND_TYPE_IMMEDIATE) c.mov(r15,rel());
            else if (ops[0].type==ZYDIS_OPERAND_TYPE_REGISTER){ int gi=IdxOf(ops[0].reg.value);
                if (gi==kRsp){ EmitExitImm(c,rip,ExitReason::UnsupportedInstruction); return; }
                if (const auto* hr=HostReg(gi)) c.mov(r15,*hr); else c.mov(r15,qword[r13+Offsets::GprSlot(gi)]); }
            else { EmitExitImm(c,rip,ExitReason::UnsupportedInstruction); return; }
            StoreRipExit(c,r15,ExitReason::BlockEnd); return;
        }
        if (cat==ZYDIS_CATEGORY_RET){
            c.mov(r14,qword[r13+RspSlot()]); c.mov(r15,qword[r14]);
            u64 inc=8;
            if (insn.operand_count_visible>=1 && ops[0].type==ZYDIS_OPERAND_TYPE_IMMEDIATE) inc+=ops[0].imm.value.u;
            c.lea(r14,ptr[r14+inc]); c.mov(qword[r13+RspSlot()],r14);
            StoreRipExit(c,r15,ExitReason::BlockEnd); return;
        }
        EmitExitImm(c, rip, ExitReason::UnsupportedInstruction);
    }

    CodeCache& cc_;
    ZydisDecoder dec_;
    GatewayGen gateway_;
    BridgeGen bridge_;
    u64 blocks_compiled_ = 0;
};

}  // namespace

#if defined(SHADPS4_CPU_BACKEND_RUNTIME_COPY)
std::unique_ptr<Backend> CreateActiveJitBackend(CodeCache& code_cache) {
    return std::make_unique<X64CopyBackend>(code_cache);
}
#endif

}  // namespace Core::Runtime
