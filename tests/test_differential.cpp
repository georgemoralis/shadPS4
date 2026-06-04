// SPDX-FileCopyrightText: Copyright 2026 shadPS4 Emulator Project
// SPDX-License-Identifier: GPL-2.0-or-later
//
// Differential tests: every snippet is executed BOTH through the JIT and
// natively on the host CPU (the "normal" shadPS4 model, since guest and host
// are both x86-64). Their resulting GPRs, status flags, and memory effects
// must agree. See diff_oracle.h for scope and the comparison contract.

#include <array>
#include <cstdint>
#include <cstring>
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
    void SetUp() override { ASSERT_TRUE(arena.valid()) << "arena alloc failed"; }

    // Run `snippet` both ways from input `in`; assert GPRs/flags/memory agree.
    void Expect(const char* name, const std::vector<u8>& snippet, Context in,
                u16 gpr_mask = 0xFFFF, u64 flag_mask = kStatusMask) {
        in.gpr[4] = arena.stack_top(); // both runs share this guest stack

        arena.clear_volatile();
        arena.save(); // pristine memory image

        auto jit = diff::RunJit(rt, arena.code(), arena.code_addr(), snippet.data(),
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
    Core::Runtime::Runtime rt;
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

} // namespace

#else // !ARCH_X86_64

TEST(DiffTest, SkippedOnNonX86Host) {
    GTEST_SKIP() << "JIT-vs-native differential requires an x86-64 host "
                    "(the native reference executes guest x86-64 bytes directly).";
}

#endif // ARCH_X86_64
