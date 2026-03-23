// SPDX-FileCopyrightText: Copyright 2025-2026 shadPS4 Emulator Project
// SPDX-License-Identifier: GPL-2.0-or-later

// =============================================================================
// tests/test_emulator_settings.cpp
//
// Google Test suite for EmulatorSettingsImpl (src/core/emulator_settings.h/.cpp)
// =============================================================================

#include <algorithm>
#include <chrono>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <memory>
#include <string>

#include <gtest/gtest.h>
#include <nlohmann/json.hpp>

#include "common/path_util.h"
#include "common/scm_rev.h"
#include "core/emulator_settings.h"
#include "core/emulator_state.h"

namespace fs = std::filesystem;
using json = nlohmann::json;

// ---------------------------------------------------------------------------
// TempDir — RAII temporary directory, unique per construction
// ---------------------------------------------------------------------------
class TempDir {
public:
    TempDir() {
        auto ns = std::chrono::steady_clock::now().time_since_epoch().count();
        path_ = fs::temp_directory_path() /
                ("shadps4_test_" + std::to_string(ns) + "_" +
                 std::to_string(reinterpret_cast<uintptr_t>(this)));
        fs::create_directories(path_);
    }
    ~TempDir() {
        std::error_code ec;
        fs::remove_all(path_, ec);
    }
    const fs::path& path() const { return path_; }

private:
    fs::path path_;
};

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------
static void WriteJson(const fs::path& p, const json& j) {
    std::ofstream out(p);
    ASSERT_TRUE(out.is_open()) << "Cannot write: " << p;
    out << std::setw(2) << j;
}

static json ReadJson(const fs::path& p) {
    std::ifstream in(p);
    EXPECT_TRUE(in.is_open()) << "Cannot read: " << p;
    json j;
    in >> j;
    return j;
}

// ---------------------------------------------------------------------------
// Test fixture
// Each test gets a fresh TempDir, fresh singletons, and all PathType entries
// redirected into that directory so no real user data is touched.
// ---------------------------------------------------------------------------
class EmulatorSettingsTest : public ::testing::Test {
protected:
    void SetUp() override {
        tmp_ = std::make_unique<TempDir>();
        const fs::path root = tmp_->path();

        // Pre-create every sub-directory that SetUserPath validates with
        // is_directory() before accepting the new path.
        using PT = Common::FS::PathType;
        const struct {
            PT type;
            const char* sub;
        } dirs[] = {
            {PT::UserDir,       ""},
            {PT::LogDir,        "log"},
            {PT::ScreenshotsDir,"screenshots"},
            {PT::ShaderDir,     "shader"},
            {PT::GameDataDir,   "data"},
            {PT::TempDataDir,   "temp"},
            {PT::SysModuleDir,  "sys_modules"},
            {PT::DownloadDir,   "download"},
            {PT::CapturesDir,   "captures"},
            {PT::CheatsDir,     "cheats"},
            {PT::PatchesDir,    "patches"},
            {PT::MetaDataDir,   "game_data"},
            {PT::CustomTrophy,  "custom_trophy"},
            {PT::CustomConfigs, "custom_configs"},
            {PT::CacheDir,      "cache"},
            {PT::FontsDir,      "fonts"},
            {PT::HomeDir,       "home"},
        };
        for (const auto& d : dirs) {
            fs::path p = d.sub[0] ? (root / d.sub) : root;
            fs::create_directories(p);
            Common::FS::SetUserPath(d.type, p);
        }

        emu_state_ = std::make_shared<EmulatorState>();
        EmulatorState::SetInstance(emu_state_);

        settings_ = std::make_shared<EmulatorSettingsImpl>();
        EmulatorSettingsImpl::SetInstance(settings_);
    }

    void TearDown() override {
        EmulatorSettingsImpl::SetInstance(nullptr);
        EmulatorState::SetInstance(nullptr);
        settings_.reset();
        emu_state_.reset();
        tmp_.reset();
    }

    fs::path ConfigJson() const {
        return tmp_->path() / "config.json";
    }
    fs::path GameConfig(const std::string& serial) const {
        return tmp_->path() / "custom_configs" / (serial + ".json");
    }

    std::unique_ptr<TempDir>              tmp_;
    std::shared_ptr<EmulatorSettingsImpl> settings_;
    std::shared_ptr<EmulatorState>        emu_state_;
};

// =============================================================================
// 1. Setting<T> — pure unit tests, no filesystem
// =============================================================================

TEST(SettingTest, DefaultCtorZeroInitialises) {
    Setting<int> s;
    EXPECT_EQ(s.value, 0);
    EXPECT_EQ(s.default_value, 0);
    EXPECT_FALSE(s.game_specific_value.has_value());
}

TEST(SettingTest, ValueCtorSetsBothValueAndDefault) {
    Setting<int> s{42};
    EXPECT_EQ(s.value, 42);
    EXPECT_EQ(s.default_value, 42);
}

TEST(SettingTest, GetDefault_PrefersGameSpecificOverBase) {
    Setting<int> s{10};
    s.value = 20;
    s.game_specific_value = 99;
    EXPECT_EQ(s.get(ConfigMode::Default), 99);
}

TEST(SettingTest, GetDefault_FallsBackToBaseWhenNoOverride) {
    Setting<int> s{10};
    s.value = 20;
    EXPECT_EQ(s.get(ConfigMode::Default), 20);
}

TEST(SettingTest, GetGlobal_IgnoresGameSpecific) {
    Setting<int> s{10};
    s.value = 20;
    s.game_specific_value = 99;
    EXPECT_EQ(s.get(ConfigMode::Global), 20);
}

TEST(SettingTest, GetClean_AlwaysReturnsFactoryDefault) {
    Setting<int> s{10};
    s.value = 20;
    s.game_specific_value = 99;
    EXPECT_EQ(s.get(ConfigMode::Clean), 10);
}

TEST(SettingTest, SetWritesToBaseOnly) {
    Setting<int> s{0};
    s.game_specific_value = 55;
    s.set(77);
    EXPECT_EQ(s.value, 77);
    EXPECT_EQ(s.game_specific_value.value(), 55); // override untouched
}

TEST(SettingTest, ResetGameSpecificClearsOverride) {
    Setting<int> s{0};
    s.game_specific_value = 55;
    s.reset_game_specific();
    EXPECT_FALSE(s.game_specific_value.has_value());
    // base and default must be intact
    EXPECT_EQ(s.value, 0);
    EXPECT_EQ(s.default_value, 0);
}

TEST(SettingTest, BoolSetting_AllModes) {
    Setting<bool> s{false};
    s.value = true;
    s.game_specific_value = false;
    EXPECT_FALSE(s.get(ConfigMode::Default));
    EXPECT_TRUE(s.get(ConfigMode::Global));
    EXPECT_FALSE(s.get(ConfigMode::Clean));
}

TEST(SettingTest, StringSetting_AllModes) {
    Setting<std::string> s{"hello"};
    s.value = "world";
    s.game_specific_value = "override";
    EXPECT_EQ(s.get(ConfigMode::Default), "override");
    EXPECT_EQ(s.get(ConfigMode::Global),  "world");
    EXPECT_EQ(s.get(ConfigMode::Clean),   "hello");
}

TEST(SettingTest, NoGameSpecific_DefaultAndGlobalAgree) {
    Setting<int> s{7};
    s.value = 7;
    EXPECT_EQ(s.get(ConfigMode::Default), s.get(ConfigMode::Global));
}

// =============================================================================
// 2. Singleton
// =============================================================================

TEST_F(EmulatorSettingsTest, GetInstance_ReturnsSamePointerRepeatedly) {
    auto a = EmulatorSettingsImpl::GetInstance();
    auto b = EmulatorSettingsImpl::GetInstance();
    EXPECT_EQ(a.get(), b.get());
}

TEST_F(EmulatorSettingsTest, SetInstance_ReplacesGlobal) {
    auto replacement = std::make_shared<EmulatorSettingsImpl>();
    EmulatorSettingsImpl::SetInstance(replacement);
    EXPECT_EQ(EmulatorSettingsImpl::GetInstance().get(), replacement.get());
}

TEST_F(EmulatorSettingsTest, Macro_DereferecesSingleton) {
    EXPECT_NO_THROW(EmulatorSettings.GetConfigMode());
}

// =============================================================================
// 3. SetDefaultValues
// =============================================================================

TEST_F(EmulatorSettingsTest, SetDefaultValues_ResetsAllGroupsToFactory) {
    settings_->SetNeo(true);
    settings_->SetWindowWidth(3840u);
    settings_->SetGpuId(2);
    settings_->SetDebugDump(true);
    settings_->SetCursorState(HideCursorState::Always);

    settings_->SetDefaultValues();

    EXPECT_FALSE(settings_->IsNeo());
    EXPECT_EQ(settings_->GetWindowWidth(), 1280u);
    EXPECT_EQ(settings_->GetGpuId(), -1);
    EXPECT_FALSE(settings_->IsDebugDump());
    EXPECT_EQ(settings_->GetCursorState(), static_cast<int>(HideCursorState::Idle));
}

TEST_F(EmulatorSettingsTest, SetDefaultValues_ClearsGameSpecificOverrides) {
    json game;
    game["General"]["neo_mode"] = true;
    WriteJson(GameConfig("CUSA00001"), game);
    settings_->Load("CUSA00001");

    settings_->SetDefaultValues();
    settings_->SetConfigMode(ConfigMode::Default);

    EXPECT_FALSE(settings_->IsNeo());
}

// =============================================================================
// 4. ConfigMode propagation
// =============================================================================

TEST_F(EmulatorSettingsTest, ConfigMode_SetAndGet_RoundTrips) {
    settings_->SetConfigMode(ConfigMode::Clean);
    EXPECT_EQ(settings_->GetConfigMode(), ConfigMode::Clean);
    settings_->SetConfigMode(ConfigMode::Global);
    EXPECT_EQ(settings_->GetConfigMode(), ConfigMode::Global);
    settings_->SetConfigMode(ConfigMode::Default);
    EXPECT_EQ(settings_->GetConfigMode(), ConfigMode::Default);
}

TEST_F(EmulatorSettingsTest, ConfigMode_Clean_ReturnFactoryDefaults) {
    settings_->SetWindowWidth(3840u);
    json game;
    game["GPU"]["window_width"] = 2560;
    WriteJson(GameConfig("CUSA00001"), game);
    settings_->Load("CUSA00001");

    settings_->SetConfigMode(ConfigMode::Clean);
    EXPECT_EQ(settings_->GetWindowWidth(), 1280u); // factory default
}

TEST_F(EmulatorSettingsTest, ConfigMode_Global_IgnoresGameSpecific) {
    settings_->SetNeo(false);
    json game;
    game["General"]["neo_mode"] = true;
    WriteJson(GameConfig("CUSA00001"), game);
    settings_->Load("CUSA00001");

    settings_->SetConfigMode(ConfigMode::Global);
    EXPECT_FALSE(settings_->IsNeo());
}

TEST_F(EmulatorSettingsTest, ConfigMode_Default_ResolvesGameSpecificWhenPresent) {
    settings_->SetNeo(false);
    json game;
    game["General"]["neo_mode"] = true;
    WriteJson(GameConfig("CUSA00001"), game);
    settings_->Load("CUSA00001");

    settings_->SetConfigMode(ConfigMode::Default);
    EXPECT_TRUE(settings_->IsNeo());
}

// =============================================================================
// 5. Save / Load — global config.json
// =============================================================================

TEST_F(EmulatorSettingsTest, Save_CreatesConfigJson) {
    ASSERT_TRUE(settings_->Save());
    EXPECT_TRUE(fs::exists(ConfigJson()));
}

TEST_F(EmulatorSettingsTest, Save_WritesAllExpectedSections) {
    ASSERT_TRUE(settings_->Save());
    json j = ReadJson(ConfigJson());
    for (const char* section : {"General", "Debug", "Input", "Audio", "GPU", "Vulkan"})
        EXPECT_TRUE(j.contains(section)) << "Missing section: " << section;
}

TEST_F(EmulatorSettingsTest, Load_ReturnsTrueForExistingFile) {
    settings_->Save();
    auto fresh = std::make_shared<EmulatorSettingsImpl>();
    EmulatorSettingsImpl::SetInstance(fresh);
    EXPECT_TRUE(fresh->Load());
}

TEST_F(EmulatorSettingsTest, RoundTrip_BoolGeneral) {
    settings_->SetNeo(true);
    settings_->Save();

    auto f = std::make_shared<EmulatorSettingsImpl>();
    EmulatorSettingsImpl::SetInstance(f);
    f->Load();
    EXPECT_TRUE(f->IsNeo());
}

TEST_F(EmulatorSettingsTest, RoundTrip_UintGPU) {
    settings_->SetWindowWidth(2560u);
    settings_->SetWindowHeight(1440u);
    settings_->Save();

    auto f = std::make_shared<EmulatorSettingsImpl>();
    EmulatorSettingsImpl::SetInstance(f);
    f->Load();
    EXPECT_EQ(f->GetWindowWidth(),  2560u);
    EXPECT_EQ(f->GetWindowHeight(), 1440u);
}

TEST_F(EmulatorSettingsTest, RoundTrip_StringGeneral) {
    settings_->SetLogType("async");
    settings_->Save();

    auto f = std::make_shared<EmulatorSettingsImpl>();
    EmulatorSettingsImpl::SetInstance(f);
    f->Load();
    EXPECT_EQ(f->GetLogType(), "async");
}

TEST_F(EmulatorSettingsTest, RoundTrip_AllGroups) {
    settings_->SetNeo(true);
    settings_->SetDebugDump(true);
    settings_->SetWindowWidth(1920u);
    settings_->SetGpuId(1);
    settings_->SetCursorState(HideCursorState::Always);
    settings_->SetAudioBackend(AudioBackend::OpenAL);
    settings_->Save();

    auto f = std::make_shared<EmulatorSettingsImpl>();
    EmulatorSettingsImpl::SetInstance(f);
    f->Load();
    EXPECT_TRUE(f->IsNeo());
    EXPECT_TRUE(f->IsDebugDump());
    EXPECT_EQ(f->GetWindowWidth(), 1920u);
    EXPECT_EQ(f->GetGpuId(), 1);
    EXPECT_EQ(f->GetCursorState(), static_cast<int>(HideCursorState::Always));
    EXPECT_EQ(f->GetAudioBackend(), static_cast<u32>(AudioBackend::OpenAL));
}

TEST_F(EmulatorSettingsTest, RoundTrip_VblankFrequency) {
    settings_->SetVblankFrequency(120u);
    settings_->Save();

    auto f = std::make_shared<EmulatorSettingsImpl>();
    EmulatorSettingsImpl::SetInstance(f);
    f->Load();
    EXPECT_EQ(f->GetVblankFrequency(), 120u);
}

TEST_F(EmulatorSettingsTest, RoundTrip_TrophyNotificationDuration) {
    settings_->SetTrophyNotificationDuration(3.5);
    settings_->Save();

    auto f = std::make_shared<EmulatorSettingsImpl>();
    EmulatorSettingsImpl::SetInstance(f);
    f->Load();
    EXPECT_DOUBLE_EQ(f->GetTrophyNotificationDuration(), 3.5);
}

TEST_F(EmulatorSettingsTest, Load_MissingFile_CreatesDefaultsOnDisk) {
    ASSERT_FALSE(fs::exists(ConfigJson()));
    settings_->Load();
    EXPECT_TRUE(fs::exists(ConfigJson()));
    EXPECT_FALSE(settings_->IsNeo()); // defaults
}

TEST_F(EmulatorSettingsTest, Load_MissingSectionDoesNotZeroOtherSections) {
    settings_->SetNeo(true);
    settings_->Save();
    json j = ReadJson(ConfigJson());
    j.erase("GPU");
    WriteJson(ConfigJson(), j);

    auto f = std::make_shared<EmulatorSettingsImpl>();
    EmulatorSettingsImpl::SetInstance(f);
    f->Load();

    EXPECT_TRUE(f->IsNeo());            // General survived
    EXPECT_EQ(f->GetWindowWidth(), 1280u); // GPU fell back to default
}

TEST_F(EmulatorSettingsTest, Load_PreservesUnknownKeysOnResave) {
    settings_->Save();
    json j = ReadJson(ConfigJson());
    j["General"]["future_feature_xyz"] = "preserved";
    WriteJson(ConfigJson(), j);

    // A fresh load + save (triggered by version mismatch) must keep the key
    auto f = std::make_shared<EmulatorSettingsImpl>();
    EmulatorSettingsImpl::SetInstance(f);
    f->Load();
    f->Save();

    json after = ReadJson(ConfigJson());
    EXPECT_EQ(after["General"]["future_feature_xyz"], "preserved");
}

TEST_F(EmulatorSettingsTest, Load_UnknownTopLevelSectionPreserved) {
    settings_->Save();
    json j = ReadJson(ConfigJson());
    j["FutureSection"]["key"] = 42;
    WriteJson(ConfigJson(), j);

    settings_->SetNeo(true);
    settings_->Save(); // merge path

    json after = ReadJson(ConfigJson());
    EXPECT_TRUE(after.contains("FutureSection"));
    EXPECT_EQ(after["FutureSection"]["key"], 42);
}

TEST_F(EmulatorSettingsTest, Load_CorruptJson_DoesNotCrash) {
    {
        std::ofstream out(ConfigJson());
        out << "{NOT VALID JSON!!!";
    }
    EXPECT_NO_THROW(settings_->Load());
}

TEST_F(EmulatorSettingsTest, Load_EmptyJsonObject_DoesNotCrash) {
    WriteJson(ConfigJson(), json::object());
    EXPECT_NO_THROW(settings_->Load());
}

// =============================================================================
// 6. VblankFrequency clamping
// =============================================================================

TEST_F(EmulatorSettingsTest, VblankFrequency_ClampedToMinimum60) {
    settings_->SetVblankFrequency(30u);
    EXPECT_EQ(settings_->GetVblankFrequency(), 60u);
}

TEST_F(EmulatorSettingsTest, VblankFrequency_ExactlyAt60_NotClamped) {
    settings_->SetVblankFrequency(60u);
    EXPECT_EQ(settings_->GetVblankFrequency(), 60u);
}

TEST_F(EmulatorSettingsTest, VblankFrequency_Above60_Preserved) {
    settings_->SetVblankFrequency(240u);
    EXPECT_EQ(settings_->GetVblankFrequency(), 240u);
}

// =============================================================================
// 7. Save / Load — per-game config (serial)
// =============================================================================

TEST_F(EmulatorSettingsTest, SaveSerial_CreatesPerGameFile) {
    ASSERT_TRUE(settings_->Save("CUSA01234"));
    EXPECT_TRUE(fs::exists(GameConfig("CUSA01234")));
}

TEST_F(EmulatorSettingsTest, LoadSerial_ReturnsFalseWhenFileAbsent) {
    EXPECT_FALSE(settings_->Load("CUSA99999"));
}

TEST_F(EmulatorSettingsTest, LoadSerial_AppliesOverrideToGameSpecificValue) {
    settings_->SetNeo(false);
    json game;
    game["General"]["neo_mode"] = true;
    WriteJson(GameConfig("CUSA01234"), game);

    ASSERT_TRUE(settings_->Load("CUSA01234"));
    settings_->SetConfigMode(ConfigMode::Default);
    EXPECT_TRUE(settings_->IsNeo());
}

TEST_F(EmulatorSettingsTest, LoadSerial_BaseValueUntouched) {
    settings_->SetWindowWidth(1280u);
    json game;
    game["GPU"]["window_width"] = 3840;
    WriteJson(GameConfig("CUSA01234"), game);
    settings_->Load("CUSA01234");

    settings_->SetConfigMode(ConfigMode::Global);
    EXPECT_EQ(settings_->GetWindowWidth(), 1280u);
}

TEST_F(EmulatorSettingsTest, LoadSerial_OverridesMultipleGroups) {
    settings_->SetNeo(false);
    settings_->SetWindowWidth(1280u);
    settings_->SetDebugDump(false);

    json game;
    game["General"]["neo_mode"]  = true;
    game["GPU"]["window_width"]  = 3840;
    game["Debug"]["debug_dump"]  = true;
    WriteJson(GameConfig("CUSA01234"), game);
    settings_->Load("CUSA01234");

    settings_->SetConfigMode(ConfigMode::Default);
    EXPECT_TRUE(settings_->IsNeo());
    EXPECT_EQ(settings_->GetWindowWidth(), 3840u);
    EXPECT_TRUE(settings_->IsDebugDump());
}

TEST_F(EmulatorSettingsTest, LoadSerial_UnrecognisedKeyIgnored) {
    json game;
    game["GPU"]["key_that_does_not_exist_xyz"] = 999;
    WriteJson(GameConfig("CUSA01234"), game);
    EXPECT_NO_THROW(settings_->Load("CUSA01234"));
}

TEST_F(EmulatorSettingsTest, LoadSerial_TypeMismatch_DoesNotCrash) {
    json game;
    game["GPU"]["window_width"] = "not_a_number";
    WriteJson(GameConfig("CUSA01234"), game);
    EXPECT_NO_THROW(settings_->Load("CUSA01234"));
    // base unchanged
    settings_->SetConfigMode(ConfigMode::Global);
    EXPECT_EQ(settings_->GetWindowWidth(), 1280u);
}

TEST_F(EmulatorSettingsTest, LoadSerial_CorruptFile_DoesNotCrash) {
    {
        std::ofstream out(GameConfig("CUSA01234"));
        out << "{{{{totally broken";
    }
    EXPECT_NO_THROW(settings_->Load("CUSA01234"));
}

TEST_F(EmulatorSettingsTest, SaveSerial_WritesGameSpecificValueWhenOverrideLoaded) {
    settings_->SetWindowWidth(1280u);
    json game;
    game["GPU"]["window_width"] = 3840;
    WriteJson(GameConfig("CUSA01234"), game);
    settings_->Load("CUSA01234");

    settings_->Save("CUSA01234");

    json saved = ReadJson(GameConfig("CUSA01234"));
    EXPECT_EQ(saved["GPU"]["window_width"].get<unsigned>(), 3840u);
}

TEST_F(EmulatorSettingsTest, SaveSerial_WritesBaseValueWhenNoOverrideSet) {
    settings_->SetWindowWidth(2560u);
    settings_->Save("CUSA01234");

    json saved = ReadJson(GameConfig("CUSA01234"));
    EXPECT_EQ(saved["GPU"]["window_width"].get<unsigned>(), 2560u);
}

TEST_F(EmulatorSettingsTest, MultipleSerials_DoNotInterfere) {
    json g1;
    g1["General"]["neo_mode"]  = true;
    g1["GPU"]["window_width"]  = 3840;
    WriteJson(GameConfig("CUSA00001"), g1);

    json g2;
    g2["General"]["neo_mode"]  = false;
    g2["GPU"]["window_width"]  = 1920;
    WriteJson(GameConfig("CUSA00002"), g2);

    {
        auto s = std::make_shared<EmulatorSettingsImpl>();
        EmulatorSettingsImpl::SetInstance(s);
        s->Load();
        s->Load("CUSA00001");
        s->SetConfigMode(ConfigMode::Default);
        EXPECT_TRUE(s->IsNeo());
        EXPECT_EQ(s->GetWindowWidth(), 3840u);
    }
    {
        auto s = std::make_shared<EmulatorSettingsImpl>();
        EmulatorSettingsImpl::SetInstance(s);
        s->Load();
        s->Load("CUSA00002");
        s->SetConfigMode(ConfigMode::Default);
        EXPECT_FALSE(s->IsNeo());
        EXPECT_EQ(s->GetWindowWidth(), 1920u);
    }
}

// =============================================================================
// 8. ClearGameSpecificOverrides
// =============================================================================

TEST_F(EmulatorSettingsTest, ClearGameSpecificOverrides_RemovesAllGroups) {
    json game;
    game["General"]["neo_mode"]     = true;
    game["GPU"]["window_width"]     = 3840;
    game["Debug"]["debug_dump"]     = true;
    game["Input"]["cursor_state"]   = 2;
    game["Audio"]["audio_backend"]  = 1;
    game["Vulkan"]["gpu_id"]        = 2;
    WriteJson(GameConfig("CUSA01234"), game);
    settings_->Load("CUSA01234");

    settings_->ClearGameSpecificOverrides();
    settings_->SetConfigMode(ConfigMode::Default);

    EXPECT_FALSE(settings_->IsNeo());
    EXPECT_EQ(settings_->GetWindowWidth(), 1280u);
    EXPECT_FALSE(settings_->IsDebugDump());
    EXPECT_EQ(settings_->GetCursorState(), static_cast<int>(HideCursorState::Idle));
    EXPECT_EQ(settings_->GetGpuId(), -1);
}

TEST_F(EmulatorSettingsTest, ClearGameSpecificOverrides_DoesNotTouchBaseValues) {
    settings_->SetWindowWidth(1920u);
    json game;
    game["GPU"]["window_width"] = 3840;
    WriteJson(GameConfig("CUSA01234"), game);
    settings_->Load("CUSA01234");

    settings_->ClearGameSpecificOverrides();

    settings_->SetConfigMode(ConfigMode::Global);
    EXPECT_EQ(settings_->GetWindowWidth(), 1920u);
}

TEST_F(EmulatorSettingsTest, ClearGameSpecificOverrides_NoopWhenNothingLoaded) {
    EXPECT_NO_THROW(settings_->ClearGameSpecificOverrides());
}

// =============================================================================
// 9. ResetGameSpecificValue
// =============================================================================

TEST_F(EmulatorSettingsTest, ResetGameSpecificValue_ClearsNamedKey) {
    settings_->SetWindowWidth(1280u);
    json game;
    game["GPU"]["window_width"] = 3840;
    WriteJson(GameConfig("CUSA01234"), game);
    settings_->Load("CUSA01234");

    settings_->SetConfigMode(ConfigMode::Default);
    ASSERT_EQ(settings_->GetWindowWidth(), 3840u);

    settings_->ResetGameSpecificValue("window_width");
    EXPECT_EQ(settings_->GetWindowWidth(), 1280u);
}

TEST_F(EmulatorSettingsTest, ResetGameSpecificValue_OnlyAffectsTargetKey) {
    json game;
    game["GPU"]["window_width"] = 3840;
    game["General"]["neo_mode"] = true;
    WriteJson(GameConfig("CUSA01234"), game);
    settings_->Load("CUSA01234");

    settings_->ResetGameSpecificValue("window_width");
    settings_->SetConfigMode(ConfigMode::Default);

    EXPECT_EQ(settings_->GetWindowWidth(), 1280u); // cleared
    EXPECT_TRUE(settings_->IsNeo());                // still set
}

TEST_F(EmulatorSettingsTest, ResetGameSpecificValue_UnknownKeyNoOp) {
    EXPECT_NO_THROW(settings_->ResetGameSpecificValue("does_not_exist_xyz"));
}

// =============================================================================
// 10. GameInstallDir management
// =============================================================================

TEST_F(EmulatorSettingsTest, AddGameInstallDir_AddsEnabled) {
    fs::path dir = tmp_->path() / "games";
    fs::create_directories(dir);
    EXPECT_TRUE(settings_->AddGameInstallDir(dir));
    ASSERT_EQ(settings_->GetGameInstallDirs().size(), 1u);
    EXPECT_EQ(settings_->GetGameInstallDirs()[0], dir);
}

TEST_F(EmulatorSettingsTest, AddGameInstallDir_RejectsDuplicate) {
    fs::path dir = tmp_->path() / "games";
    fs::create_directories(dir);
    settings_->AddGameInstallDir(dir);
    EXPECT_FALSE(settings_->AddGameInstallDir(dir));
    EXPECT_EQ(settings_->GetGameInstallDirs().size(), 1u);
}

TEST_F(EmulatorSettingsTest, RemoveGameInstallDir_RemovesEntry) {
    fs::path dir = tmp_->path() / "games";
    fs::create_directories(dir);
    settings_->AddGameInstallDir(dir);
    settings_->RemoveGameInstallDir(dir);
    EXPECT_TRUE(settings_->GetGameInstallDirs().empty());
}

TEST_F(EmulatorSettingsTest, RemoveGameInstallDir_NoopForMissing) {
    EXPECT_NO_THROW(settings_->RemoveGameInstallDir("/nonexistent/path"));
}

TEST_F(EmulatorSettingsTest, SetGameInstallDirEnabled_DisablesDir) {
    fs::path dir = tmp_->path() / "games";
    fs::create_directories(dir);
    settings_->AddGameInstallDir(dir, true);
    settings_->SetGameInstallDirEnabled(dir, false);
    EXPECT_TRUE(settings_->GetGameInstallDirs().empty());
}

TEST_F(EmulatorSettingsTest, SetGameInstallDirEnabled_ReEnablesDir) {
    fs::path dir = tmp_->path() / "games";
    fs::create_directories(dir);
    settings_->AddGameInstallDir(dir, false);
    ASSERT_TRUE(settings_->GetGameInstallDirs().empty());
    settings_->SetGameInstallDirEnabled(dir, true);
    EXPECT_EQ(settings_->GetGameInstallDirs().size(), 1u);
}

TEST_F(EmulatorSettingsTest, SetAllGameInstallDirs_ReplacesExistingList) {
    fs::path d1 = tmp_->path() / "g1";
    fs::path d2 = tmp_->path() / "g2";
    fs::create_directories(d1);
    fs::create_directories(d2);
    settings_->AddGameInstallDir(d1);

    settings_->SetAllGameInstallDirs({{d2, true}});
    ASSERT_EQ(settings_->GetGameInstallDirs().size(), 1u);
    EXPECT_EQ(settings_->GetGameInstallDirs()[0], d2);
}

TEST_F(EmulatorSettingsTest, GameInstallDirs_FullRoundTrip_WithEnabledFlags) {
    fs::path d1 = tmp_->path() / "g1";
    fs::path d2 = tmp_->path() / "g2";
    fs::create_directories(d1);
    fs::create_directories(d2);
    settings_->AddGameInstallDir(d1, true);
    settings_->AddGameInstallDir(d2, false);
    settings_->Save();

    auto f = std::make_shared<EmulatorSettingsImpl>();
    EmulatorSettingsImpl::SetInstance(f);
    f->Load();

    const auto& all = f->GetAllGameInstallDirs();
    ASSERT_EQ(all.size(), 2u);
    EXPECT_EQ(all[0].path, d1);
    EXPECT_TRUE(all[0].enabled);
    EXPECT_EQ(all[1].path, d2);
    EXPECT_FALSE(all[1].enabled);
}

TEST_F(EmulatorSettingsTest, GetGameInstallDirsEnabled_ReflectsState) {
    fs::path d1 = tmp_->path() / "g1";
    fs::path d2 = tmp_->path() / "g2";
    fs::create_directories(d1);
    fs::create_directories(d2);
    settings_->AddGameInstallDir(d1, true);
    settings_->AddGameInstallDir(d2, false);

    auto enabled = settings_->GetGameInstallDirsEnabled();
    ASSERT_EQ(enabled.size(), 2u);
    EXPECT_TRUE(enabled[0]);
    EXPECT_FALSE(enabled[1]);
}

// =============================================================================
// 11. GetAllOverrideableKeys
// =============================================================================

TEST_F(EmulatorSettingsTest, GetAllOverrideableKeys_IsNonEmpty) {
    EXPECT_FALSE(settings_->GetAllOverrideableKeys().empty());
}

TEST_F(EmulatorSettingsTest, GetAllOverrideableKeys_ContainsRepresentativeKeys) {
    auto keys = settings_->GetAllOverrideableKeys();
    auto has  = [&](const char* k) {
        return std::find(keys.begin(), keys.end(), k) != keys.end();
    };
    // General
    EXPECT_TRUE(has("neo_mode"));
    EXPECT_TRUE(has("volume_slider"));
    // GPU
    EXPECT_TRUE(has("window_width"));
    EXPECT_TRUE(has("null_gpu"));
    EXPECT_TRUE(has("vblank_frequency"));
    // Vulkan
    EXPECT_TRUE(has("gpu_id"));
    EXPECT_TRUE(has("pipeline_cache_enabled"));
    // Debug
    EXPECT_TRUE(has("debug_dump"));
    EXPECT_TRUE(has("log_enabled"));
    // Input
    EXPECT_TRUE(has("cursor_state"));
    // Audio
    EXPECT_TRUE(has("audio_backend"));
}

TEST_F(EmulatorSettingsTest, GetAllOverrideableKeys_NoDuplicates) {
    auto keys = settings_->GetAllOverrideableKeys();
    std::vector<std::string> sorted = keys;
    std::sort(sorted.begin(), sorted.end());
    auto it = std::unique(sorted.begin(), sorted.end());
    EXPECT_EQ(it, sorted.end()) << "Duplicate key found in overrideable keys list";
}

// =============================================================================
// 12. Per-group GetOverrideableFields
// =============================================================================

TEST_F(EmulatorSettingsTest, GetGeneralOverrideableFields_NonEmpty) {
    EXPECT_FALSE(settings_->GetGeneralOverrideableFields().empty());
}

TEST_F(EmulatorSettingsTest, GetGPUOverrideableFields_ContainsWindowAndFullscreen) {
    auto fields = settings_->GetGPUOverrideableFields();
    auto has = [&](const char* k) {
        return std::any_of(fields.begin(), fields.end(),
            [k](const OverrideItem& f){ return std::string(f.key) == k; });
    };
    EXPECT_TRUE(has("window_width"));
    EXPECT_TRUE(has("window_height"));
    EXPECT_TRUE(has("full_screen"));
    EXPECT_TRUE(has("vblank_frequency"));
}

TEST_F(EmulatorSettingsTest, GetVulkanOverrideableFields_ContainsGpuId) {
    auto fields = settings_->GetVulkanOverrideableFields();
    bool found = std::any_of(fields.begin(), fields.end(),
        [](const OverrideItem& f){ return std::string(f.key) == "gpu_id"; });
    EXPECT_TRUE(found);
}

// =============================================================================
// 13. EmulatorState integration
// =============================================================================

TEST_F(EmulatorSettingsTest, EmulatorState_GameRunning_DefaultFalse) {
    EXPECT_FALSE(emu_state_->IsGameRunning());
}

TEST_F(EmulatorSettingsTest, EmulatorState_GameRunningRoundTrip) {
    emu_state_->SetGameRunning(true);
    EXPECT_TRUE(emu_state_->IsGameRunning());
    emu_state_->SetGameRunning(false);
    EXPECT_FALSE(emu_state_->IsGameRunning());
}

TEST_F(EmulatorSettingsTest, EmulatorState_AutoPatchesEnabledByDefault) {
    EXPECT_TRUE(emu_state_->IsAutoPatchesLoadEnabled());
}

TEST_F(EmulatorSettingsTest, EmulatorState_GetInstanceLazilyCreates) {
    EmulatorState::SetInstance(nullptr);
    EXPECT_NE(EmulatorState::GetInstance(), nullptr);
}

TEST_F(EmulatorSettingsTest, LoadSerial_SetsGameSpecificConfigUsedFlag) {
    json game;
    game["General"]["neo_mode"] = true;
    WriteJson(GameConfig("CUSA01234"), game);

    settings_->Load("CUSA01234");
    EXPECT_TRUE(EmulatorState::GetInstance()->IsGameSpecifigConfigUsed());
}

TEST_F(EmulatorSettingsTest, LoadSerial_FileAbsent_ConfigUsedFlagNotSet) {
    settings_->Load("CUSA99999");
    EXPECT_FALSE(EmulatorState::GetInstance()->IsGameSpecifigConfigUsed());
}

// =============================================================================
// 14. Spot-check every getter/setter pair (one per group)
// =============================================================================

// --- General ---
TEST_F(EmulatorSettingsTest, Setting_VolumeSlider) {
    settings_->SetVolumeSlider(75);
    EXPECT_EQ(settings_->GetVolumeSlider(), 75);
}
TEST_F(EmulatorSettingsTest, Setting_Neo) {
    settings_->SetNeo(true);
    EXPECT_TRUE(settings_->IsNeo());
}
TEST_F(EmulatorSettingsTest, Setting_DevKit) {
    settings_->SetDevKit(true);
    EXPECT_TRUE(settings_->IsDevKit());
}
TEST_F(EmulatorSettingsTest, Setting_PSNSignedIn) {
    settings_->SetPSNSignedIn(true);
    EXPECT_TRUE(settings_->IsPSNSignedIn());
}
TEST_F(EmulatorSettingsTest, Setting_TrophyPopupDisabled) {
    settings_->SetTrophyPopupDisabled(true);
    EXPECT_TRUE(settings_->IsTrophyPopupDisabled());
}
TEST_F(EmulatorSettingsTest, Setting_TrophyNotificationDuration) {
    settings_->SetTrophyNotificationDuration(3.5);
    EXPECT_DOUBLE_EQ(settings_->GetTrophyNotificationDuration(), 3.5);
}
TEST_F(EmulatorSettingsTest, Setting_TrophyNotificationSide) {
    settings_->SetTrophyNotificationSide("left");
    EXPECT_EQ(settings_->GetTrophyNotificationSide(), "left");
}
TEST_F(EmulatorSettingsTest, Setting_ShowSplash) {
    settings_->SetShowSplash(true);
    EXPECT_TRUE(settings_->IsShowSplash());
}
TEST_F(EmulatorSettingsTest, Setting_IdenticalLogGrouped) {
    settings_->SetIdenticalLogGrouped(false);
    EXPECT_FALSE(settings_->IsIdenticalLogGrouped());
}
TEST_F(EmulatorSettingsTest, Setting_LogType) {
    settings_->SetLogType("async");
    EXPECT_EQ(settings_->GetLogType(), "async");
}
TEST_F(EmulatorSettingsTest, Setting_LogFilter) {
    settings_->SetLogFilter("EmuSettings:Debug");
    EXPECT_EQ(settings_->GetLogFilter(), "EmuSettings:Debug");
}
TEST_F(EmulatorSettingsTest, Setting_ConnectedToNetwork) {
    settings_->SetConnectedToNetwork(true);
    EXPECT_TRUE(settings_->IsConnectedToNetwork());
}
TEST_F(EmulatorSettingsTest, Setting_DiscordRPCEnabled) {
    settings_->SetDiscordRPCEnabled(true);
    EXPECT_TRUE(settings_->IsDiscordRPCEnabled());
}
TEST_F(EmulatorSettingsTest, Setting_ShowFpsCounter) {
    settings_->SetShowFpsCounter(true);
    EXPECT_TRUE(settings_->IsShowFpsCounter());
}
TEST_F(EmulatorSettingsTest, Setting_ConsoleLanguage) {
    settings_->SetConsoleLanguage(6);
    EXPECT_EQ(settings_->GetConsoleLanguage(), 6);
}

// --- GPU ---
TEST_F(EmulatorSettingsTest, Setting_WindowDimensions) {
    settings_->SetWindowWidth(2560u);
    settings_->SetWindowHeight(1440u);
    EXPECT_EQ(settings_->GetWindowWidth(),  2560u);
    EXPECT_EQ(settings_->GetWindowHeight(), 1440u);
}
TEST_F(EmulatorSettingsTest, Setting_InternalScreenDimensions) {
    settings_->SetInternalScreenWidth(1920u);
    settings_->SetInternalScreenHeight(1080u);
    EXPECT_EQ(settings_->GetInternalScreenWidth(),  1920u);
    EXPECT_EQ(settings_->GetInternalScreenHeight(), 1080u);
}
/*TEST_F(EmulatorSettingsTest, Setting_NullGpu) {
    settings_->SetNullGpu(true);
    EXPECT_TRUE(settings_->IsNullGPU());
}*/
TEST_F(EmulatorSettingsTest, Setting_CopyGpuBuffers) {
    settings_->SetCopyGpuBuffers(true);
    EXPECT_TRUE(settings_->IsCopyGpuBuffers());
}
TEST_F(EmulatorSettingsTest, Setting_ReadbacksMode) {
    settings_->SetReadbacksMode(static_cast<u32>(GpuReadbacksMode::Relaxed));
    EXPECT_EQ(settings_->GetReadbacksMode(), static_cast<u32>(GpuReadbacksMode::Relaxed));
}
TEST_F(EmulatorSettingsTest, Setting_ReadbackLinearImages) {
    settings_->SetReadbackLinearImagesEnabled(true);
    EXPECT_TRUE(settings_->IsReadbackLinearImagesEnabled());
}
TEST_F(EmulatorSettingsTest, Setting_DirectMemoryAccess) {
    settings_->SetDirectMemoryAccessEnabled(true);
    EXPECT_TRUE(settings_->IsDirectMemoryAccessEnabled());
}
TEST_F(EmulatorSettingsTest, Setting_DumpShaders) {
    settings_->SetDumpShaders(true);
    EXPECT_TRUE(settings_->IsDumpShaders());
}
TEST_F(EmulatorSettingsTest, Setting_FullScreen) {
    settings_->SetFullScreen(true);
    EXPECT_TRUE(settings_->IsFullScreen());
}
TEST_F(EmulatorSettingsTest, Setting_FullScreenMode) {
    settings_->SetFullScreenMode("Borderless");
    EXPECT_EQ(settings_->GetFullScreenMode(), "Borderless");
}
TEST_F(EmulatorSettingsTest, Setting_PresentMode) {
    settings_->SetPresentMode("Fifo");
    EXPECT_EQ(settings_->GetPresentMode(), "Fifo");
}
TEST_F(EmulatorSettingsTest, Setting_HdrAllowed) {
    settings_->SetHdrAllowed(true);
    EXPECT_TRUE(settings_->IsHdrAllowed());
}
TEST_F(EmulatorSettingsTest, Setting_FsrEnabled) {
    settings_->SetFsrEnabled(true);
    EXPECT_TRUE(settings_->IsFsrEnabled());
}
TEST_F(EmulatorSettingsTest, Setting_RcasEnabled) {
    settings_->SetRcasEnabled(false);
    EXPECT_FALSE(settings_->IsRcasEnabled());
}
TEST_F(EmulatorSettingsTest, Setting_RcasAttenuation) {
    settings_->SetRcasAttenuation(500);
    EXPECT_EQ(settings_->GetRcasAttenuation(), 500);
}

// --- Debug ---
TEST_F(EmulatorSettingsTest, Setting_SeparateLoggingEnabled) {
    settings_->SetSeparateLoggingEnabled(true);
    EXPECT_TRUE(settings_->IsSeparateLoggingEnabled());
}
TEST_F(EmulatorSettingsTest, Setting_DebugDump) {
    settings_->SetDebugDump(true);
    EXPECT_TRUE(settings_->IsDebugDump());
}
TEST_F(EmulatorSettingsTest, Setting_ShaderCollect) {
    settings_->SetShaderCollect(true);
    EXPECT_TRUE(settings_->IsShaderCollect());
}
TEST_F(EmulatorSettingsTest, Setting_LogEnabled) {
    settings_->SetLogEnabled(false);
    EXPECT_FALSE(settings_->IsLogEnabled());
}

// --- Input ---
TEST_F(EmulatorSettingsTest, Setting_CursorState) {
    settings_->SetCursorState(HideCursorState::Always);
    EXPECT_EQ(settings_->GetCursorState(), static_cast<int>(HideCursorState::Always));
}
TEST_F(EmulatorSettingsTest, Setting_CursorHideTimeout) {
    settings_->SetCursorHideTimeout(10);
    EXPECT_EQ(settings_->GetCursorHideTimeout(), 10);
}
TEST_F(EmulatorSettingsTest, Setting_UsbDeviceBackend) {
    settings_->SetUsbDeviceBackend(static_cast<int>(UsbBackendType::SkylandersPortal));
    EXPECT_EQ(settings_->GetUsbDeviceBackend(),
              static_cast<int>(UsbBackendType::SkylandersPortal));
}
TEST_F(EmulatorSettingsTest, Setting_MotionControlsEnabled) {
    settings_->SetMotionControlsEnabled(false);
    EXPECT_FALSE(settings_->IsMotionControlsEnabled());
}
TEST_F(EmulatorSettingsTest, Setting_BackgroundControllerInput) {
    settings_->SetBackgroundControllerInput(true);
    EXPECT_TRUE(settings_->IsBackgroundControllerInput());
}
TEST_F(EmulatorSettingsTest, Setting_UseUnifiedInputConfig) {
    settings_->SetUseUnifiedInputConfig(false);
    EXPECT_FALSE(settings_->IsUseUnifiedInputConfig());
}

// --- Audio ---
TEST_F(EmulatorSettingsTest, Setting_AudioBackend) {
    settings_->SetAudioBackend(static_cast<u32>(AudioBackend::OpenAL));
    EXPECT_EQ(settings_->GetAudioBackend(), static_cast<u32>(AudioBackend::OpenAL));
}
TEST_F(EmulatorSettingsTest, Setting_SDLMicDevice) {
    settings_->SetSDLMicDevice("My Mic");
    EXPECT_EQ(settings_->GetSDLMicDevice(), "My Mic");
}
TEST_F(EmulatorSettingsTest, Setting_SDLMainOutputDevice) {
    settings_->SetSDLMainOutputDevice("My Headphones");
    EXPECT_EQ(settings_->GetSDLMainOutputDevice(), "My Headphones");
}
TEST_F(EmulatorSettingsTest, Setting_SDLPadSpkOutputDevice) {
    settings_->SetSDLPadSpkOutputDevice("Controller Speaker");
    EXPECT_EQ(settings_->GetSDLPadSpkOutputDevice(), "Controller Speaker");
}
TEST_F(EmulatorSettingsTest, Setting_OpenALMicDevice) {
    settings_->SetOpenALMicDevice("OpenAL Mic");
    EXPECT_EQ(settings_->GetOpenALMicDevice(), "OpenAL Mic");
}
TEST_F(EmulatorSettingsTest, Setting_OpenALMainOutputDevice) {
    settings_->SetOpenALMainOutputDevice("OpenAL Out");
    EXPECT_EQ(settings_->GetOpenALMainOutputDevice(), "OpenAL Out");
}

// --- Vulkan ---
TEST_F(EmulatorSettingsTest, Setting_GpuId) {
    settings_->SetGpuId(2);
    EXPECT_EQ(settings_->GetGpuId(), 2);
}
TEST_F(EmulatorSettingsTest, Setting_VkValidationEnabled) {
    settings_->SetVkValidationEnabled(true);
    EXPECT_TRUE(settings_->IsVkValidationEnabled());
}
TEST_F(EmulatorSettingsTest, Setting_VkValidationCoreEnabled) {
    settings_->SetVkValidationCoreEnabled(false);
    EXPECT_FALSE(settings_->IsVkValidationCoreEnabled());
}
TEST_F(EmulatorSettingsTest, Setting_VkValidationSyncEnabled) {
    settings_->SetVkValidationSyncEnabled(true);
    EXPECT_TRUE(settings_->IsVkValidationSyncEnabled());
}
TEST_F(EmulatorSettingsTest, Setting_VkValidationGpuEnabled) {
    settings_->SetVkValidationGpuEnabled(true);
    EXPECT_TRUE(settings_->IsVkValidationGpuEnabled());
}
TEST_F(EmulatorSettingsTest, Setting_VkCrashDiagnosticEnabled) {
    settings_->SetVkCrashDiagnosticEnabled(true);
    EXPECT_TRUE(settings_->IsVkCrashDiagnosticEnabled());
}
TEST_F(EmulatorSettingsTest, Setting_VkHostMarkersEnabled) {
    settings_->SetVkHostMarkersEnabled(true);
    EXPECT_TRUE(settings_->IsVkHostMarkersEnabled());
}
TEST_F(EmulatorSettingsTest, Setting_VkGuestMarkersEnabled) {
    settings_->SetVkGuestMarkersEnabled(true);
    EXPECT_TRUE(settings_->IsVkGuestMarkersEnabled());
}
TEST_F(EmulatorSettingsTest, Setting_RenderdocEnabled) {
    settings_->SetRenderdocEnabled(true);
    EXPECT_TRUE(settings_->IsRenderdocEnabled());
}
TEST_F(EmulatorSettingsTest, Setting_PipelineCacheEnabled) {
    settings_->SetPipelineCacheEnabled(true);
    EXPECT_TRUE(settings_->IsPipelineCacheEnabled());
}
TEST_F(EmulatorSettingsTest, Setting_PipelineCacheArchived) {
    settings_->SetPipelineCacheArchived(true);
    EXPECT_TRUE(settings_->IsPipelineCacheArchived());
}

// --- Path accessors ---
TEST_F(EmulatorSettingsTest, GetHomeDir_ReturnsCustomWhenSet) {
    fs::path dir = tmp_->path() / "custom_home";
    fs::create_directories(dir);
    settings_->SetHomeDir(dir);
    EXPECT_EQ(settings_->GetHomeDir(), dir);
}
TEST_F(EmulatorSettingsTest, GetSysModulesDir_FallsBackToPathUtilWhenEmpty) {
    // default_value is empty; GetSysModulesDir falls back to GetUserPath(SysModuleDir)
    auto result = settings_->GetSysModulesDir();
    EXPECT_FALSE(result.empty());
}
TEST_F(EmulatorSettingsTest, GetFontsDir_FallsBackToPathUtilWhenEmpty) {
    auto result = settings_->GetFontsDir();
    EXPECT_FALSE(result.empty());
}

// =============================================================================
// 15. Regression: Bug 2 — version-mismatch Save() must not clobber settings
// =============================================================================

TEST_F(EmulatorSettingsTest, Regression_VersionMismatch_PreservesSettings) {
    settings_->SetNeo(true);
    settings_->SetWindowWidth(2560u);
    settings_->Save();

    // Force a stale version string so the mismatch branch fires
    json j = ReadJson(ConfigJson());
    j["Debug"]["config_version"] = "old_hash_0000";
    WriteJson(ConfigJson(), j);

    auto f = std::make_shared<EmulatorSettingsImpl>();
    EmulatorSettingsImpl::SetInstance(f);
    f->Load(); // triggers version-bump Save() internally

    EXPECT_TRUE(f->IsNeo());
    EXPECT_EQ(f->GetWindowWidth(), 2560u);
}

// =============================================================================
// 16. Regression: Bug 3 — Load("") called twice must be idempotent
// =============================================================================

TEST_F(EmulatorSettingsTest, Regression_DoubleGlobalLoad_IsIdempotent) {
    settings_->SetNeo(true);
    settings_->SetWindowWidth(2560u);
    settings_->Save();

    auto f = std::make_shared<EmulatorSettingsImpl>();
    EmulatorSettingsImpl::SetInstance(f);
    f->Load(""); // first — loads from disk
    f->Load(""); // second — must not reset anything

    EXPECT_TRUE(f->IsNeo());
    EXPECT_EQ(f->GetWindowWidth(), 2560u);
}

// =============================================================================
// 17. Regression: Bug 4 — SetGameSpecifigConfigUsed flag correctness
// =============================================================================

TEST_F(EmulatorSettingsTest, Regression_ConfigUsedFlag_TrueWhenFileExists) {
    json game;
    game["General"]["neo_mode"] = true;
    WriteJson(GameConfig("CUSA01234"), game);
    settings_->Load("CUSA01234");
    EXPECT_TRUE(EmulatorState::GetInstance()->IsGameSpecifigConfigUsed());
}

TEST_F(EmulatorSettingsTest, Regression_ConfigUsedFlag_FalseWhenFileAbsent) {
    settings_->Load("CUSA99999");
    EXPECT_FALSE(EmulatorState::GetInstance()->IsGameSpecifigConfigUsed());
}

// =============================================================================
// 18. Regression: Bug 5 — destructor must not save when Load() was never called
// =============================================================================

TEST_F(EmulatorSettingsTest, Regression_Destructor_NoSaveIfLoadNeverCalled) {
    settings_->SetNeo(true);
    settings_->Save();
    auto t0 = fs::last_write_time(ConfigJson());

    {
        // Create and immediately destroy without calling Load()
        auto untouched = std::make_shared<EmulatorSettingsImpl>();
        // destructor fires here
    }

    auto t1 = fs::last_write_time(ConfigJson());
    EXPECT_EQ(t0, t1) << "Destructor wrote config.json without a prior Load()";
}

TEST_F(EmulatorSettingsTest, Regression_Destructor_SavesAfterSuccessfulLoad) {
    settings_->SetNeo(true);
    settings_->Save();

    {
        auto s = std::make_shared<EmulatorSettingsImpl>();
        EmulatorSettingsImpl::SetInstance(s);
        s->Load();
        s->SetWindowWidth(2560u); // mutate after successful load
        // destructor should write this change
    }

    auto verify = std::make_shared<EmulatorSettingsImpl>();
    EmulatorSettingsImpl::SetInstance(verify);
    verify->Load();
    EXPECT_EQ(verify->GetWindowWidth(), 2560u);
}
