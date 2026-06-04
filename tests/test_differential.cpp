// SPDX-FileCopyrightText: Copyright 2026 shadPS4 Emulator Project
// SPDX-License-Identifier: GPL-2.0-or-later
//
// Differential tests: every snippet is executed BOTH through the JIT and
// natively on the host CPU (the "normal" shadPS4 model, since guest and host
// are both x86-64). Their resulting GPRs, status flags, and memory effects
// must agree. See diff_oracle.h for scope and the comparison contract.

#include <array>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <memory>
#include <random>
#include <vector>

#include <gtest/gtest.h>

#include "common/arch.h"

// The native side of the oracle emits and executes x86-64 host machine code, so
// the differential is meaningful only when the host is x86-64. On the arm64 job
// this file still compiles (the loop in CMakeLists adds it unconditionally), but
// reduces to a single skipped test rather than breaking the build.
#ifdef ARCH_X86_64

#ifdef _WIN32
#include <windows.h>
#else
#include <sys/mman.h>
#endif

#include "diff_oracle.h"

namespace {

using diff::Context;
using diff::kStatusMask;

// ---- guest arena: RW, identity-mapped (guest addr == host addr) ------------
// Layout:  [0, kCodeSz)        guest code (snippet + BSR terminator; JIT only)
//          [kDataOff, ...)     data scratch + stack (both runs; save/restored)
//          stack top = kStackTopOff (grows down)
class Arena {
public:
    static constexpr size_t kSize = 64 * 1024;
    static constexpr size_t kCodeSz = 4096;
    static constexpr size_t kDataOff = 8 * 1024;       // data scratch
    static constexpr size_t kStackTopOff = 48 * 1024;  // stack grows down from here
    static constexpr size_t kVolatileOff = kDataOff;   // region that may change

    Arena() {
#ifdef _WIN32
        base_ = static_cast<u8*>(
            ::VirtualAlloc(nullptr, kSize, MEM_COMMIT | MEM_RESERVE, PAGE_READWRITE));
#else
        void* p = ::mmap(nullptr, kSize, PROT_READ | PROT_WRITE,
                         MAP_PRIVATE | MAP_ANON, -1, 0);
        base_ = (p == MAP_FAILED) ? nullptr : static_cast<u8*>(p);
#endif
    }
    ~Arena() {
        if (!base_) return;
#ifdef _WIN32
        ::VirtualFree(base_, 0, MEM_RELEASE);
#else
        ::munmap(base_, kSize);
#endif
    }
    Arena(const Arena&) = delete;
    Arena& operator=(const Arena&) = delete;

    bool valid() const { return base_ != nullptr; }
    u8* code() const { return base_; }
    u64 code_addr() const { return reinterpret_cast<u64>(base_); }
    u64 data_addr() const { return reinterpret_cast<u64>(base_ + kDataOff); }
    u64 stack_top() const { return reinterpret_cast<u64>(base_ + kStackTopOff); }

    // Save/restore/compare the volatile (data + stack) region so the JIT and
    // native runs each start from identical memory and we can diff their writes.
    void save() { std::memcpy(snap_.data(), base_ + kVolatileOff, snap_.size()); }
    void restore() { std::memcpy(base_ + kVolatileOff, snap_.data(), snap_.size()); }
    std::vector<u8> capture() const {
        return std::vector<u8>(base_ + kVolatileOff, base_ + kSize);
    }
    void clear_volatile() { std::memset(base_ + kVolatileOff, 0, kSize - kVolatileOff); }

private:
    u8* base_ = nullptr;
    std::array<u8, kSize - kVolatileOff> snap_{};
};

class DiffTest : public ::testing::Test {
protected:
    void SetUp() override {
        ASSERT_TRUE(arena.valid()) << "arena alloc failed";
        // Zero the data+stack region once. Done here (not in Expect) so a test
        // may seed guest memory before calling Expect without it being wiped.
        arena.clear_volatile();
    }

    // Run `snippet` both ways from input `in`; assert GPRs/flags/memory agree.
    void Expect(const char* name, const std::vector<u8>& snippet, Context in,
                u16 gpr_mask = 0xFFFF, u64 flag_mask = kStatusMask) {
        in.gpr[4] = arena.stack_top(); // both runs share this guest stack

        // The JIT caches compiled blocks by guest address, and every snippet is
        // staged at the same guest address. A *different* snippet would hit the
        // previous one's cached block, so recompile on a fresh Runtime whenever
        // the snippet bytes change. This is cheap: it fires only on a snippet
        // change (e.g. once per op in the fuzz loop), never per input.
        if (snippet != last_snippet_) {
            rt_ = std::make_unique<Core::Runtime::Runtime>();
            last_snippet_ = snippet;
        }

        arena.save(); // pristine memory image (whatever the test seeded)

        auto jit = diff::RunJit(*rt_, arena.code(), arena.code_addr(), snippet.data(),
                                snippet.size(), in);
        auto memJ = arena.capture();

        arena.restore(); // back to pristine before the native run

        Context nat;
        diff::RunNative(snippet.data(), snippet.size(), in, nat);
        auto memN = arena.capture();

        // The JIT must have lifted the WHOLE snippet (exit at the BSR terminator,
        // not at some earlier unsupported instruction).
        EXPECT_EQ(jit.exit_reason,
                  static_cast<u32>(Core::Runtime::ExitReason::UnsupportedInstruction))
            << name << ": JIT did not reach the BSR terminator";
        EXPECT_EQ(jit.rip, arena.code_addr() + snippet.size())
            << name << ": JIT bailed mid-snippet (an instruction is unsupported?)";

        const std::string regs = diff::Compare(jit.out, nat, gpr_mask, flag_mask);
        EXPECT_TRUE(regs.empty()) << name << ": register/flag mismatch\n" << regs;
        EXPECT_EQ(memJ, memN) << name << ": memory effects differ between JIT and native";
    }

    Arena arena;
    std::unique_ptr<Core::Runtime::Runtime> rt_ =
        std::make_unique<Core::Runtime::Runtime>();
    std::vector<u8> last_snippet_; // triggers a JIT reset when the snippet changes
};

// Convenience: status-flag subsets for per-op masking (the x86 spec leaves some
// flags undefined for some ops; comparing those would be a false mismatch).
constexpr u64 F_ALL = kStatusMask;
constexpr u64 F_NO_OF = diff::CF | diff::PF | diff::ZF | diff::SF; // OF undefined
constexpr u64 F_CF_OF = diff::CF | diff::OF;                       // mul/imul defined set

Context inGpr(std::initializer_list<std::pair<int, u64>> regs, u64 flags = 0) {
    Context c;
    c.rflags = flags;
    for (auto& [i, v] : regs) c.gpr[i] = v;
    return c;
}

// ============================================================================
// Curated matrix — representative, byte-verified encodings.
// ============================================================================

TEST_F(DiffTest, Add_Reg)   { Expect("add rax,rcx", {0x48,0x01,0xC8}, inGpr({{0,0x7fffffffffffffffULL},{1,1}})); }
TEST_F(DiffTest, Sub_Reg)   { Expect("sub rax,rcx", {0x48,0x29,0xC8}, inGpr({{0,3},{1,5}})); }
TEST_F(DiffTest, And_Reg)   { Expect("and rax,rcx", {0x48,0x21,0xC8}, inGpr({{0,0xF0F0},{1,0xFF00}})); }
TEST_F(DiffTest, Or_Reg)    { Expect("or rax,rcx",  {0x48,0x09,0xC8}, inGpr({{0,0x0F0F},{1,0x00FF}})); }
TEST_F(DiffTest, Xor_Reg)   { Expect("xor rax,rcx", {0x48,0x31,0xC8}, inGpr({{0,0xAAAA},{1,0xAAAA}})); }
TEST_F(DiffTest, Cmp_Reg)   { Expect("cmp rax,rcx", {0x48,0x39,0xC8}, inGpr({{0,5},{1,5}})); }
TEST_F(DiffTest, Test_Reg)  { Expect("test rax,rcx",{0x48,0x85,0xC8}, inGpr({{0,0xFF},{1,0x0F}})); }
TEST_F(DiffTest, Inc_Reg)   { Expect("inc rax",     {0x48,0xFF,0xC0}, inGpr({{0,0x7fffffffffffffffULL}})); } // CF preserved
TEST_F(DiffTest, Dec_Reg)   { Expect("dec rax",     {0x48,0xFF,0xC8}, inGpr({{0,0}})); }
TEST_F(DiffTest, Neg_Reg)   { Expect("neg rax",     {0x48,0xF7,0xD8}, inGpr({{0,5}})); }
TEST_F(DiffTest, Not_Reg)   { Expect("not rax",     {0x48,0xF7,0xD0}, inGpr({{0,0x1234}})); } // no flags
TEST_F(DiffTest, Adc_Reg)   { Expect("adc rax,rcx (CF=1)", {0x48,0x11,0xC8}, inGpr({{0,1},{1,2}}, diff::CF)); }
TEST_F(DiffTest, Sbb_Reg)   { Expect("sbb rax,rcx (CF=1)", {0x48,0x19,0xC8}, inGpr({{0,10},{1,3}}, diff::CF)); }
TEST_F(DiffTest, AddImm32)  { Expect("add rax,0x100", {0x48,0x05,0x00,0x01,0x00,0x00}, inGpr({{0,0xFFFFFFFFFF00ULL}})); }

TEST_F(DiffTest, Shl1)      { Expect("shl rax,1", {0x48,0xD1,0xE0}, inGpr({{0,0x4000000000000000ULL}})); }
TEST_F(DiffTest, Shr1)      { Expect("shr rax,1", {0x48,0xD1,0xE8}, inGpr({{0,0x1}})); }
TEST_F(DiffTest, Sar1)      { Expect("sar rax,1", {0x48,0xD1,0xF8}, inGpr({{0,0x8000000000000000ULL}})); }
TEST_F(DiffTest, ShlImm)    { Expect("shl rax,4", {0x48,0xC1,0xE0,0x04}, inGpr({{0,0x1111}}), 0xFFFF, F_NO_OF); }
TEST_F(DiffTest, Rol1)      { Expect("rol rax,1", {0x48,0xD1,0xC0}, inGpr({{0,0x8000000000000001ULL}}), 0xFFFF, diff::CF|diff::OF); }
TEST_F(DiffTest, Ror1)      { Expect("ror rax,1", {0x48,0xD1,0xC8}, inGpr({{0,0x1}}), 0xFFFF, diff::CF|diff::OF); }

TEST_F(DiffTest, ImulReg)   { Expect("imul rax,rcx", {0x48,0x0F,0xAF,0xC1}, inGpr({{0,7},{1,9}}), 0xFFFF, F_CF_OF); }
TEST_F(DiffTest, Lea)       { Expect("lea rax,[rcx+rdx*4+8]", {0x48,0x8D,0x44,0x91,0x08}, inGpr({{1,0x1000},{2,3}})); }
TEST_F(DiffTest, Movzx)     { Expect("movzx rax,cl", {0x48,0x0F,0xB6,0xC1}, inGpr({{1,0xFF}})); }
TEST_F(DiffTest, Movsx)     { Expect("movsx rax,cl", {0x48,0x0F,0xBE,0xC1}, inGpr({{1,0x80}})); }

// CMOV family — these caught a real scratch-clobber bug (CMOVLE/CMOVG) per the
// project journal; differential coverage guards against regression.
TEST_F(DiffTest, CmovleTaken)    { Expect("cmp rax,rcx; cmovle rdx,rbx", {0x48,0x39,0xC8,0x48,0x0F,0x4E,0xD3}, inGpr({{0,1},{1,2},{2,0xDEAD},{3,0xBEEF}})); }
TEST_F(DiffTest, CmovleNotTaken) { Expect("cmp rax,rcx; cmovle rdx,rbx", {0x48,0x39,0xC8,0x48,0x0F,0x4E,0xD3}, inGpr({{0,5},{1,2},{2,0xDEAD},{3,0xBEEF}})); }
TEST_F(DiffTest, CmovgTaken)     { Expect("cmp rax,rcx; cmovg rdx,rbx",  {0x48,0x39,0xC8,0x48,0x0F,0x4F,0xD3}, inGpr({{0,9},{1,2},{2,0xDEAD},{3,0xBEEF}})); }

// SETcc — writes a byte from flags.
TEST_F(DiffTest, Setz)  { Expect("cmp rax,rcx; setz al", {0x48,0x39,0xC8,0x0F,0x94,0xC0}, inGpr({{0,4},{1,4}})); }
TEST_F(DiffTest, Setl)  { Expect("cmp rax,rcx; setl al", {0x48,0x39,0xC8,0x0F,0x9C,0xC0}, inGpr({{0,1},{1,9}})); }

// Memory: load/store via register base (data pointer in rsi).
TEST_F(DiffTest, MemLoad)  {
    *reinterpret_cast<u64*>(arena.data_addr()) = 0x1122334455667788ULL;
    Expect("mov rdx,[rsi]", {0x48,0x8B,0x16}, inGpr({{6, arena.data_addr()}}));
}
TEST_F(DiffTest, MemStore) {
    Expect("mov [rsi],rdx", {0x48,0x89,0x16}, inGpr({{6, arena.data_addr()},{2,0xABCDEF0123456789ULL}}));
}
TEST_F(DiffTest, MemAddToMem) {
    *reinterpret_cast<u64*>(arena.data_addr()) = 100;
    Expect("add [rsi],rdx", {0x48,0x01,0x16}, inGpr({{6, arena.data_addr()},{2,23}}));
}

// Stack: push/pop hit guest memory; rsp must round-trip and the value match.
TEST_F(DiffTest, PushPop) { Expect("push rax; pop rcx", {0x50,0x59}, inGpr({{0,0xCAFEF00DBEEF1234ULL}})); }

// ============================================================================
// Operand widths (8 / 16 / 32 / 64). The 32-bit forms ZERO-EXTEND the result
// into the full 64-bit register; the 8/16-bit forms PRESERVE the upper bits.
// Inputs carry high bits so both behaviours are actually exercised.
// ============================================================================
TEST_F(DiffTest, Add8)  { Expect("add al,cl",   {0x00,0xC8},      inGpr({{0,0xFFFFFFFFFFFFFF80ULL},{1,0xAAAAAAAAAAAAAA90ULL}})); }
TEST_F(DiffTest, Add16) { Expect("add ax,cx",   {0x66,0x01,0xC8}, inGpr({{0,0xFFFFFFFFFFFF8000ULL},{1,0xAAAAAAAAAAAA9000ULL}})); }
TEST_F(DiffTest, Add32) { Expect("add eax,ecx", {0x01,0xC8},      inGpr({{0,0xFFFFFFFF80000000ULL},{1,0x11111111A0000000ULL}})); } // zero-extends
TEST_F(DiffTest, Sub32) { Expect("sub eax,ecx", {0x29,0xC8},      inGpr({{0,0xDEADBEEF00000003ULL},{1,0xCAFE000000000005ULL}})); }
TEST_F(DiffTest, And32) { Expect("and eax,ecx", {0x21,0xC8},      inGpr({{0,0xFFFFFFFFF0F0F0F0ULL},{1,0x00000000FF00FF00ULL}})); }
TEST_F(DiffTest, Xor32) { Expect("xor eax,ecx", {0x31,0xC8},      inGpr({{0,0xABCDEF12AAAA5555ULL},{1,0x000000005555AAAAULL}})); }
TEST_F(DiffTest, Inc32) { Expect("inc ecx",     {0xFF,0xC1},      inGpr({{1,0xFFFFFFFFFFFFFFFFULL}})); } // -> 0 (zero-ext), CF preserved
TEST_F(DiffTest, Dec16) { Expect("dec cx",      {0x66,0xFF,0xC9}, inGpr({{1,0xAAAAAAAAAAAA0000ULL}})); }
TEST_F(DiffTest, Neg8)  { Expect("neg cl",      {0xF6,0xD9},      inGpr({{1,0xFFFFFFFFFFFFFF05ULL}})); }
TEST_F(DiffTest, Imul32){ Expect("imul eax,ecx",{0x0F,0xAF,0xC1}, inGpr({{0,0x12345678ULL},{1,0x1000ULL}}), 0xFFFF, F_CF_OF); }
TEST_F(DiffTest, Mov32) { Expect("mov eax,ecx", {0x89,0xC8},      inGpr({{0,0xFFFFFFFFFFFFFFFFULL},{1,0x00000000DEADBEEFULL}})); } // zero-extends rax

// ============================================================================
// CMOVcc — full 16-condition matrix, each across several flag seeds (cmov reads
// flags, does not write them). cmovcc rdx,rbx with rdx != rbx so a (non-)move
// is observable. This family caught a real scratch-clobber bug per the journal.
// ============================================================================
TEST_F(DiffTest, CmovccMatrix) {
    const std::vector<u64> seeds = {0, diff::CF, diff::ZF, diff::SF, diff::OF, diff::PF,
                                    diff::SF | diff::OF, diff::ZF | diff::SF, kStatusMask};
    for (u8 cc = 0x40; cc <= 0x4F; ++cc) {
        const std::vector<u8> snip = {0x48, 0x0F, cc, 0xD3}; // cmovcc rdx, rbx
        for (u64 f : seeds) {
            char nm[40];
            std::snprintf(nm, sizeof nm, "cmov(0x%02x) flags=0x%03llx", cc,
                          (unsigned long long)f);
            Expect(nm, snip, inGpr({{2, 0xAAAAAAAAAAAAAAAAULL}, {3, 0xBBBBBBBBBBBBBBBBULL}}, f));
            if (::testing::Test::HasFatalFailure()) return;
        }
    }
}

// ============================================================================
// SETcc — full 16-condition matrix. setcc al writes the low byte only; the
// upper bits of rax must be preserved (input rax has them set).
// ============================================================================
TEST_F(DiffTest, SetccMatrix) {
    const std::vector<u64> seeds = {0, diff::CF, diff::ZF, diff::SF, diff::OF, diff::PF,
                                    diff::SF | diff::OF, diff::CF | diff::ZF, kStatusMask};
    for (u8 cc = 0x90; cc <= 0x9F; ++cc) {
        const std::vector<u8> snip = {0x0F, cc, 0xC0}; // setcc al
        for (u64 f : seeds) {
            char nm[40];
            std::snprintf(nm, sizeof nm, "setcc(0x%02x) flags=0x%03llx", cc,
                          (unsigned long long)f);
            Expect(nm, snip, inGpr({{0, 0xFFFFFFFFFFFFFF00ULL}}, f));
            if (::testing::Test::HasFatalFailure()) return;
        }
    }
}

// ============================================================================
// Shifts / rotates by immediate, including the count==0 edge (flags untouched)
// and count>1 (OF undefined -> excluded). By-CL forms are fuzzed below.
// ============================================================================
TEST_F(DiffTest, ShlImm0)  { Expect("shl rax,0",  {0x48,0xC1,0xE0,0x00}, inGpr({{0,0x1234}}, diff::CF|diff::OF)); } // count 0: flags preserved
TEST_F(DiffTest, ShrImm3)  { Expect("shr rax,3",  {0x48,0xC1,0xE8,0x03}, inGpr({{0,0xFF}}), 0xFFFF, F_NO_OF); }
TEST_F(DiffTest, SarImm5)  { Expect("sar rax,5",  {0x48,0xC1,0xF8,0x05}, inGpr({{0,0x8000000000000000ULL}}), 0xFFFF, F_NO_OF); }
TEST_F(DiffTest, RolImm7)  { Expect("rol rax,7",  {0x48,0xC1,0xC0,0x07}, inGpr({{0,0x8100000000000081ULL}}), 0xFFFF, diff::CF); } // count!=1: OF undef
TEST_F(DiffTest, Shr32Imm) { Expect("shr eax,1",  {0xD1,0xE8},           inGpr({{0,0xFFFFFFFF00000003ULL}})); } // zero-extends, count 1

// ============================================================================
// MUL / IMUL one-operand (rdx:rax = rax * src). CF/OF defined; rest undefined.
// ============================================================================
TEST_F(DiffTest, Mul64)  { Expect("mul rcx",   {0x48,0xF7,0xE1}, inGpr({{0,0x100000001ULL},{1,0x100000001ULL}}), 0xFFFF, F_CF_OF); }
TEST_F(DiffTest, Imul64) { Expect("imul rcx",  {0x48,0xF7,0xE9}, inGpr({{0,(u64)-3},{1,7}}),                    0xFFFF, F_CF_OF); }

// DIV / IDIV — inputs constrained so they never fault (#DE): rdx=0 and a
// non-zero divisor (and a non-negative dividend below 2^63 for idiv) guarantee
// no divide-by-zero and no quotient overflow. All flags are undefined here.
TEST_F(DiffTest, Div64)  { Expect("div rcx",   {0x48,0xF7,0xF1}, inGpr({{0,0xDEADBEEFCAFEULL},{2,0},{1,0x1000}}), 0xFFFF, 0); }
TEST_F(DiffTest, Idiv64) { Expect("idiv rcx",  {0x48,0xF7,0xF9}, inGpr({{0,0x0123456789ABCDEULL},{2,0},{1,0x100}}), 0xFFFF, 0); }

// ============================================================================
// Sign-extend accumulator family (implicit operands, no flags).
// ============================================================================
TEST_F(DiffTest, Cdqe) { Expect("cdqe", {0x48,0x98}, inGpr({{0,0x00000000FFFFFFFFULL}})); } // eax(-1) -> rax(-1)
TEST_F(DiffTest, Cqo)  { Expect("cqo",  {0x48,0x99}, inGpr({{0,0x8000000000000000ULL}})); } // rax<0 -> rdx=all ones
TEST_F(DiffTest, Cdq)  { Expect("cdq",  {0x99},      inGpr({{0,0x00000000FF000000ULL}})); }

// ============================================================================
// BT — bit test sets CF from the selected bit; other flags undefined.
// ============================================================================
TEST_F(DiffTest, BtImm) { Expect("bt rax,63", {0x48,0x0F,0xBA,0xE0,0x3F}, inGpr({{0,0x8000000000000000ULL}}), 0xFFFF, diff::CF); }
TEST_F(DiffTest, BtReg) { Expect("bt rax,rcx", {0x48,0x0F,0xA3,0xC8},     inGpr({{0,0x4},{1,2}}),              0xFFFF, diff::CF); }

// ============================================================================
// XCHG / XADD / CMPXCHG — register exchange and read-modify-write with flags.
// CMPXCHG exercises the implicit RAX operand (a classic lifter pitfall).
// ============================================================================
TEST_F(DiffTest, Xchg)        { Expect("xchg rax,rcx", {0x48,0x87,0xC8}, inGpr({{0,0x1111},{1,0x2222}})); }
TEST_F(DiffTest, Xadd)        { Expect("xadd rcx,rax", {0x48,0x0F,0xC1,0xC1}, inGpr({{0,5},{1,7}})); }
TEST_F(DiffTest, CmpxchgEq)   { Expect("cmpxchg rcx,rdx (rax==rcx)", {0x48,0x0F,0xB1,0xD1}, inGpr({{0,9},{1,9},{2,0x42}})); }
TEST_F(DiffTest, CmpxchgNeq)  { Expect("cmpxchg rcx,rdx (rax!=rcx)", {0x48,0x0F,0xB1,0xD1}, inGpr({{0,9},{1,8},{2,0x42}})); }

// ============================================================================
// POPCNT / LZCNT — compare the GPR result only (flag semantics are specialised
// and not the focus here).
// ============================================================================
TEST_F(DiffTest, Popcnt) { Expect("popcnt rax,rcx", {0xF3,0x48,0x0F,0xB8,0xC1}, inGpr({{1,0xF0F0F0F0F0F0F0F0ULL}}), 0xFFFF, 0); }
TEST_F(DiffTest, Lzcnt)  { Expect("lzcnt rax,rcx",  {0xF3,0x48,0x0F,0xBD,0xC1}, inGpr({{1,0x0000FF0000000000ULL}}), 0xFFFF, 0); }

// ============================================================================
// Multi-instruction basic blocks — exercise chaining, dependency, and flag
// propagation across several lifted instructions (each ends in a full-flag op).
// ============================================================================
TEST_F(DiffTest, Block_AddSubAnd) {
    Expect("add rax,rcx; sub rax,rdx; and rax,rbx",
           {0x48,0x01,0xC8, 0x48,0x29,0xD0, 0x48,0x21,0xD8},
           inGpr({{0,0xF00D},{1,0x1234},{2,0x99},{3,0xFFFF}}));
}
TEST_F(DiffTest, Block_MovShlAddNeg) {
    Expect("mov rax,rcx; shl rax,3; add rax,rdx; neg rax",
           {0x48,0x89,0xC8, 0x48,0xC1,0xE0,0x03, 0x48,0x01,0xD0, 0x48,0xF7,0xD8},
           inGpr({{1,0x21},{2,0x7}}));
}
TEST_F(DiffTest, Block_ImulXorSub) {
    Expect("imul rax,rcx; xor rax,rdx; sub rax,rbx",
           {0x48,0x0F,0xAF,0xC1, 0x48,0x31,0xD0, 0x48,0x29,0xD8},
           inGpr({{0,0x101},{1,0x3},{2,0xAAAA},{3,0x55}}));
}
TEST_F(DiffTest, Block_MemRoundTrip) {
    *reinterpret_cast<u64*>(arena.data_addr()) = 0x1000;
    Expect("mov rdx,[rsi]; add rdx,rax; mov [rsi],rdx",
           {0x48,0x8B,0x16, 0x48,0x01,0xC2, 0x48,0x89,0x16},
           inGpr({{6, arena.data_addr()}, {0, 0x37}}));
}

// ============================================================================
// Extended registers r8..r15 — exercises REX.R / REX.B decoding, a common
// lifter blind spot (the wrong bit silently aliases to a low register).
// ============================================================================
TEST_F(DiffTest, AddR8R9)   { Expect("add r8,r9",   {0x4D,0x01,0xC8}, inGpr({{8,0x7fffffffffffffffULL},{9,1}})); }
TEST_F(DiffTest, SubR15R12) { Expect("sub r15,r12", {0x4D,0x29,0xE7}, inGpr({{15,3},{12,5}})); }
TEST_F(DiffTest, XorR10R11) { Expect("xor r10,r11", {0x4D,0x31,0xDA}, inGpr({{10,0xF0F0},{11,0x0FF0}})); }
TEST_F(DiffTest, ShlR13)    { Expect("shl r13,cl",  {0x49,0xD3,0xE5}, inGpr({{13,0x1},{1,8}}), 0xFFFF, F_NO_OF); }
TEST_F(DiffTest, ImulR8R9)  { Expect("imul r8,r9",  {0x4D,0x0F,0xAF,0xC1}, inGpr({{8,7},{9,9}}), 0xFFFF, F_CF_OF); }
TEST_F(DiffTest, MovR10Mem) {
    *reinterpret_cast<u64*>(arena.data_addr()) = 0xC0FFEEULL;
    Expect("mov r10,[r11]", {0x4D,0x8B,0x13}, inGpr({{11, arena.data_addr()}}));
}

// ============================================================================
// Addressing modes — base, base+disp, base+index*scale(+disp). SIB encodings
// are where address-generation bugs concentrate.
// ============================================================================
TEST_F(DiffTest, LeaIndexScale8) { Expect("lea rax,[rcx+rdx*8]", {0x48,0x8D,0x04,0xD1}, inGpr({{1,0x1000},{2,3}})); }
TEST_F(DiffTest, LeaDispOnly)    { Expect("lea rax,[rcx+0x40]",  {0x48,0x8D,0x41,0x40}, inGpr({{1,0x2000}})); }
TEST_F(DiffTest, LeaNegDisp)     { Expect("lea rax,[rcx-8]",     {0x48,0x8D,0x41,0xF8}, inGpr({{1,0x2000}})); }
TEST_F(DiffTest, MemLoadSib) {
    const u64 base = arena.data_addr();
    *reinterpret_cast<u64*>(base + 3 * 2 + 0x100) = 0x9999ULL;
    // mov rax,[rcx+rdx*2+0x100]
    Expect("mov rax,[rcx+rdx*2+0x100]", {0x48,0x8B,0x84,0x51,0x00,0x01,0x00,0x00},
           inGpr({{1, base}, {2, 3}}));
}
TEST_F(DiffTest, MemStoreSibDisp8) {
    const u64 base = arena.data_addr();
    // mov [rcx+rdx*4+0x10], rax
    Expect("mov [rcx+rdx*4+0x10],rax", {0x48,0x89,0x44,0x91,0x10},
           inGpr({{1, base}, {2, 2}, {0, 0xDEADBEEFCAFEULL}}));
}

// ============================================================================
// Immediate operand forms — imm8 sign-extended (0x83), imm32 (0x81), and the
// sign-extended mov-imm32 (0xC7). Sign-extension of imm8/imm32 into 64 bits is
// a frequent off-by-a-bug.
// ============================================================================
TEST_F(DiffTest, AddImm8Sext)  { Expect("add rax,-1",      {0x48,0x83,0xC0,0xFF}, inGpr({{0,0x100}})); }
TEST_F(DiffTest, AndImm8)      { Expect("and rax,0x0F",    {0x48,0x83,0xE0,0x0F}, inGpr({{0,0xFF}})); }
TEST_F(DiffTest, CmpImm8)      { Expect("cmp rax,5",       {0x48,0x83,0xF8,0x05}, inGpr({{0,5}})); }
TEST_F(DiffTest, SubImm32)     { Expect("sub rax,0x12345", {0x48,0x81,0xE8,0x45,0x23,0x01,0x00}, inGpr({{0,0x20000}})); }
TEST_F(DiffTest, MovImm32Sext) { Expect("mov rax,-100",    {0x48,0xC7,0xC0,0x9C,0xFF,0xFF,0xFF}, inGpr({{0,0}})); }
TEST_F(DiffTest, TestImm32)    { Expect("test rax,0xFF00", {0x48,0xF7,0xC0,0x00,0xFF,0x00,0x00}, inGpr({{0,0xAB00}})); }

// ============================================================================
// Memory-operand widths + read-modify-write. Stores must touch exactly the
// operand width (a too-wide store corrupts neighbours; caught by the memory
// diff). RMW ops fold load+op+store and must set flags from the result.
// ============================================================================
TEST_F(DiffTest, MemStoreByte) {
    *reinterpret_cast<u64*>(arena.data_addr()) = 0xFFFFFFFFFFFFFFFFULL;
    Expect("mov byte [rsi],cl", {0x88,0x0E}, inGpr({{6, arena.data_addr()}, {1, 0xAB}}));
}
TEST_F(DiffTest, MemStoreWord) {
    *reinterpret_cast<u64*>(arena.data_addr()) = 0xFFFFFFFFFFFFFFFFULL;
    Expect("mov word [rsi],cx", {0x66,0x89,0x0E}, inGpr({{6, arena.data_addr()}, {1, 0x1234}}));
}
TEST_F(DiffTest, MemStoreDword) {
    *reinterpret_cast<u64*>(arena.data_addr()) = 0xFFFFFFFFFFFFFFFFULL;
    Expect("mov dword [rsi],ecx", {0x89,0x0E}, inGpr({{6, arena.data_addr()}, {1, 0x11223344}}));
}
TEST_F(DiffTest, MemLoadByteZx) {
    *reinterpret_cast<u8*>(arena.data_addr()) = 0xFF;
    Expect("movzx rax,byte [rsi]", {0x48,0x0F,0xB6,0x06}, inGpr({{6, arena.data_addr()}}));
}
TEST_F(DiffTest, MemIncQword) {
    *reinterpret_cast<u64*>(arena.data_addr()) = 0x7fffffffffffffffULL;
    Expect("inc qword [rsi]", {0x48,0xFF,0x06}, inGpr({{6, arena.data_addr()}}));
}
TEST_F(DiffTest, MemNegQword) {
    *reinterpret_cast<u64*>(arena.data_addr()) = 5;
    Expect("neg qword [rsi]", {0x48,0xF7,0x1E}, inGpr({{6, arena.data_addr()}}));
}
TEST_F(DiffTest, MemNegDword) {
    *reinterpret_cast<u64*>(arena.data_addr()) = 0xFFFFFFFF00000007ULL;
    Expect("neg dword [rsi]", {0xF7,0x1E}, inGpr({{6, arena.data_addr()}}));
}
TEST_F(DiffTest, MemNotQword) {
    *reinterpret_cast<u64*>(arena.data_addr()) = 0x0F0F0F0F0F0F0F0FULL;
    Expect("not qword [rsi]", {0x48,0xF7,0x16}, inGpr({{6, arena.data_addr()}}));
}
TEST_F(DiffTest, MemNotDword) {
    *reinterpret_cast<u64*>(arena.data_addr()) = 0xAAAAAAAAAAAAAAAAULL;
    Expect("not dword [rsi]", {0xF7,0x16}, inGpr({{6, arena.data_addr()}}));
}
TEST_F(DiffTest, MemShlQword) {
    *reinterpret_cast<u64*>(arena.data_addr()) = 0x1;
    Expect("shl qword [rsi],1", {0x48,0xD1,0x26}, inGpr({{6, arena.data_addr()}}));
}

// ============================================================================
// Randomized fuzz: for each snippet, many random GPR inputs.
// ============================================================================
struct FuzzOp { const char* name; std::vector<u8> bytes; u16 gpr_mask; u64 flag_mask; };

TEST_F(DiffTest, Fuzz_Arithmetic) {
    const std::vector<FuzzOp> ops = {
        {"add rax,rcx",  {0x48,0x01,0xC8}, 0xFFFF, F_ALL},
        {"sub rax,rcx",  {0x48,0x29,0xC8}, 0xFFFF, F_ALL},
        {"and rax,rcx",  {0x48,0x21,0xC8}, 0xFFFF, F_ALL},
        {"or rax,rcx",   {0x48,0x09,0xC8}, 0xFFFF, F_ALL},
        {"xor rax,rcx",  {0x48,0x31,0xC8}, 0xFFFF, F_ALL},
        {"cmp rax,rcx",  {0x48,0x39,0xC8}, 0xFFFF, F_ALL},
        {"neg rax",      {0x48,0xF7,0xD8}, 0xFFFF, F_ALL},
        {"inc rax",      {0x48,0xFF,0xC0}, 0xFFFF, F_ALL},
        {"sar rax,1",    {0x48,0xD1,0xF8}, 0xFFFF, F_ALL},
        {"imul rax,rcx", {0x48,0x0F,0xAF,0xC1}, 0xFFFF, F_CF_OF},
    };
    std::mt19937_64 rng(0xC0FFEE); // fixed seed -> reproducible
    for (const auto& op : ops) {
        for (int iter = 0; iter < 200; ++iter) {
            Context in;
            in.gpr[0] = rng(); in.gpr[1] = rng();
            in.rflags = rng() & kStatusMask;
            Expect(op.name, op.bytes, in, op.gpr_mask, op.flag_mask);
            if (::testing::Test::HasFatalFailure()) return;
        }
    }
}

// Flag-consuming ops (ADC/SBB read CF; CMOVcc/SETcc read the condition) driven
// with RANDOM input flags so both taken and not-taken paths are exercised.
TEST_F(DiffTest, Fuzz_FlagSensitive) {
    const std::vector<FuzzOp> ops = {
        {"adc rax,rcx",   {0x48,0x11,0xC8},      0xFFFF, F_ALL},
        {"sbb rax,rcx",   {0x48,0x19,0xC8},      0xFFFF, F_ALL},
        {"cmovz rdx,rbx", {0x48,0x0F,0x44,0xD3}, 0xFFFF, F_ALL}, // flags preserved
        {"cmovs rdx,rbx", {0x48,0x0F,0x48,0xD3}, 0xFFFF, F_ALL},
        {"cmovl rdx,rbx", {0x48,0x0F,0x4C,0xD3}, 0xFFFF, F_ALL},
        {"setz al",       {0x0F,0x94,0xC0},      0xFFFF, F_ALL},
        {"setl al",       {0x0F,0x9C,0xC0},      0xFFFF, F_ALL},
    };
    std::mt19937_64 rng(0x5EED1234);
    for (const auto& op : ops) {
        for (int iter = 0; iter < 200; ++iter) {
            Context in;
            for (int i = 0; i < 4; ++i) in.gpr[i] = rng();
            in.rflags = rng() & kStatusMask;
            Expect(op.name, op.bytes, in, op.gpr_mask, op.flag_mask);
            if (::testing::Test::HasFatalFailure()) return;
        }
    }
}

// Non-64-bit widths: 32-bit results must zero-extend; 8/16-bit must preserve
// the upper bits. Random inputs carry high bits to exercise both.
TEST_F(DiffTest, Fuzz_Widths) {
    const std::vector<FuzzOp> ops = {
        {"add eax,ecx",  {0x01,0xC8},      0xFFFF, F_ALL},
        {"sub eax,ecx",  {0x29,0xC8},      0xFFFF, F_ALL},
        {"and eax,ecx",  {0x21,0xC8},      0xFFFF, F_ALL},
        {"imul eax,ecx", {0x0F,0xAF,0xC1}, 0xFFFF, F_CF_OF},
        {"add ax,cx",    {0x66,0x01,0xC8}, 0xFFFF, F_ALL},
        {"add al,cl",    {0x00,0xC8},      0xFFFF, F_ALL},
        {"mov eax,ecx",  {0x89,0xC8},      0xFFFF, F_ALL}, // mov: flags preserved
    };
    std::mt19937_64 rng(0xA11CE5EED);
    for (const auto& op : ops) {
        for (int iter = 0; iter < 200; ++iter) {
            Context in;
            in.gpr[0] = rng(); in.gpr[1] = rng();
            in.rflags = rng() & kStatusMask;
            Expect(op.name, op.bytes, in, op.gpr_mask, op.flag_mask);
            if (::testing::Test::HasFatalFailure()) return;
        }
    }
}

// Shifts/rotates by CL: random count (incl. 0 and >63, masked to 6 bits). OF is
// defined only for count==1, so it is excluded (F_NO_OF); the preserved flags
// stay comparable across count==0 too.
TEST_F(DiffTest, Fuzz_ShiftsByCL) {
    const std::vector<FuzzOp> ops = {
        {"shl rax,cl", {0x48,0xD3,0xE0}, 0xFFFF, F_NO_OF},
        {"shr rax,cl", {0x48,0xD3,0xE8}, 0xFFFF, F_NO_OF},
        {"sar rax,cl", {0x48,0xD3,0xF8}, 0xFFFF, F_NO_OF},
        {"rol rax,cl", {0x48,0xD3,0xC0}, 0xFFFF, F_NO_OF},
        {"ror rax,cl", {0x48,0xD3,0xC8}, 0xFFFF, F_NO_OF},
    };
    std::mt19937_64 rng(0xC1C1C1C1);
    for (const auto& op : ops) {
        for (int iter = 0; iter < 200; ++iter) {
            Context in;
            in.gpr[0] = rng();          // value
            in.gpr[1] = rng() & 0xFF;   // cl = shift count (low byte of rcx)
            in.rflags = rng() & kStatusMask;
            Expect(op.name, op.bytes, in, op.gpr_mask, op.flag_mask);
            if (::testing::Test::HasFatalFailure()) return;
        }
    }
}

// DIV / IDIV with inputs constrained to never fault (rdx=0, non-zero divisor,
// non-negative dividend below 2^63 for idiv). All flags undefined -> mask 0.
TEST_F(DiffTest, Fuzz_Division) {
    std::mt19937_64 rng(0xD1D1D3);
    for (int iter = 0; iter < 200; ++iter) {
        Context in;
        in.gpr[0] = rng();                  // dividend low (rax)
        in.gpr[2] = 0;                      // rdx = 0 (high half) -> no overflow
        in.gpr[1] = rng() | 1ULL;           // rcx divisor, guaranteed non-zero
        Expect("div rcx", {0x48,0xF7,0xF1}, in, 0xFFFF, 0);
        if (::testing::Test::HasFatalFailure()) return;
    }
    for (int iter = 0; iter < 200; ++iter) {
        Context in;
        in.gpr[0] = rng() & 0x7FFFFFFFFFFFFFFFULL; // dividend in [0, 2^63) -> no idiv overflow
        in.gpr[2] = 0;
        in.gpr[1] = rng() | 1ULL;
        Expect("idiv rcx", {0x48,0xF7,0xF9}, in, 0xFFFF, 0);
        if (::testing::Test::HasFatalFailure()) return;
    }
}

// Curated boundary values: zero, signed/unsigned extremes, power-of-two edges,
// alternating bit patterns. Random64 almost never lands on these, yet they are
// exactly where ZF/CF/OF/SF computation breaks (exact overflow, result==0,
// neg(INT_MIN), etc.). Cross-product over the pool, both carry-in states.
static const u64 kCorner[] = {
    0, 1, 2, 0x7F, 0x80, 0xFF, 0x100,
    0x7FFF, 0x8000, 0xFFFF, 0x10000,
    0x7FFFFFFF, 0x80000000ULL, 0xFFFFFFFFULL, 0x100000000ULL,
    0x7FFFFFFFFFFFFFFFULL, 0x8000000000000000ULL, 0xFFFFFFFFFFFFFFFFULL,
    0x5555555555555555ULL, 0xAAAAAAAAAAAAAAAAULL,
};

TEST_F(DiffTest, Corner_Arithmetic) {
    const std::vector<FuzzOp> ops = {
        {"add rax,rcx",  {0x48,0x01,0xC8},      0xFFFF, F_ALL},
        {"sub rax,rcx",  {0x48,0x29,0xC8},      0xFFFF, F_ALL},
        {"adc rax,rcx",  {0x48,0x11,0xC8},      0xFFFF, F_ALL},
        {"sbb rax,rcx",  {0x48,0x19,0xC8},      0xFFFF, F_ALL},
        {"cmp rax,rcx",  {0x48,0x39,0xC8},      0xFFFF, F_ALL},
        {"and rax,rcx",  {0x48,0x21,0xC8},      0xFFFF, F_ALL},
        {"neg rax",      {0x48,0xF7,0xD8},      0xFFFF, F_ALL},
        {"inc rax",      {0x48,0xFF,0xC0},      0xFFFF, F_ALL},
        {"imul rax,rcx", {0x48,0x0F,0xAF,0xC1}, 0xFFFF, F_CF_OF},
    };
    for (const auto& op : ops) {
        for (u64 a : kCorner)
            for (u64 b : kCorner)
                for (u64 cf : {u64{0}, diff::CF}) { // carry-in for adc/sbb
                    Expect(op.name, op.bytes, inGpr({{0, a}, {1, b}}, cf),
                           op.gpr_mask, op.flag_mask);
                    if (::testing::Test::HasFatalFailure()) return;
                }
    }
}

// Shift/rotate count edges: 64-bit shifts mask the count to 6 bits, so 64->0
// (no-op, flags preserved) and 65->1; values straddle the sign bit.
TEST_F(DiffTest, Corner_ShiftCounts) {
    const u8 counts[] = {0, 1, 2, 7, 8, 15, 16, 31, 32, 33, 63, 64, 65, 127, 200, 255};
    const u64 vals[] = {0x1ULL, 0x8000000000000000ULL, 0xFFFFFFFFFFFFFFFFULL,
                        0x123456789ABCDEF0ULL};
    const std::vector<FuzzOp> ops = {
        {"shl rax,cl", {0x48,0xD3,0xE0}, 0xFFFF, F_NO_OF},
        {"shr rax,cl", {0x48,0xD3,0xE8}, 0xFFFF, F_NO_OF},
        {"sar rax,cl", {0x48,0xD3,0xF8}, 0xFFFF, F_NO_OF},
        {"rol rax,cl", {0x48,0xD3,0xC0}, 0xFFFF, F_NO_OF},
        {"ror rax,cl", {0x48,0xD3,0xC8}, 0xFFFF, F_NO_OF},
    };
    for (const auto& op : ops)
        for (u64 v : vals)
            for (u8 c : counts) {
                Expect(op.name, op.bytes, inGpr({{0, v}, {1, c}}), op.gpr_mask, op.flag_mask);
                if (::testing::Test::HasFatalFailure()) return;
            }
}

// Extended-register fuzz: r8..r15 operands to shake out REX.R/REX.B handling.
TEST_F(DiffTest, Fuzz_ExtendedRegs) {
    const std::vector<FuzzOp> ops = {
        {"add r8,r9",    {0x4D,0x01,0xC8},      0xFFFF, F_ALL},
        {"sub r10,r11",  {0x4D,0x29,0xDA},      0xFFFF, F_ALL},
        {"and r12,r13",  {0x4D,0x21,0xEC},      0xFFFF, F_ALL},
        {"imul r14,r15", {0x4D,0x0F,0xAF,0xF7}, 0xFFFF, F_CF_OF},
    };
    std::mt19937_64 rng(0xE87E7DED);
    for (const auto& op : ops) {
        for (int iter = 0; iter < 200; ++iter) {
            Context in;
            for (int i = 8; i < 16; ++i) in.gpr[i] = rng();
            in.rflags = rng() & kStatusMask;
            Expect(op.name, op.bytes, in, op.gpr_mask, op.flag_mask);
            if (::testing::Test::HasFatalFailure()) return;
        }
    }
}

} // namespace

#else // !ARCH_X86_64

TEST(DiffTest, SkippedOnNonX86Host) {
    GTEST_SKIP() << "JIT-vs-native differential requires an x86-64 host "
                    "(the native reference executes guest x86-64 bytes directly).";
}

#endif // ARCH_X86_64
