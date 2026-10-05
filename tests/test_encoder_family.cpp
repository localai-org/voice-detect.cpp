// Encoder identity accessors (voicedetect_capi_encoder_arch / _name / _family).
// No model files needed: the GGUFs are synthesized in memory, so this runs in CI.
//
//  * the family string has exactly the documented format
//  * path, memory and prefixed (bundle) loads give identical strings
//  * a bundle's own header keys never leak into a component
//  * missing keys, a wrong-typed name and odd characters
//  * NULL context, pointer lifetime and concurrent reads
#include "gguf_bundle.hpp"

#include "voicedetect_capi.h"

#include <atomic>
#include <cstdio>
#include <cstdlib>
#include <random>
#include <string>
#include <thread>

using namespace vdtest;

static int g_fail = 0;
#define CHECK(c) do { if (!(c)) { std::fprintf(stderr, "FAIL %s:%d: %s\n", __FILE__, __LINE__, #c); ++g_fail; } } while (0)
#define CHECK_STR(got, want) do { const char* g_ = (got); const std::string w_ = (want); \
    if (!g_ || w_ != g_) { std::fprintf(stderr, "FAIL %s:%d: got '%s' want '%s'\n", __FILE__, __LINE__, g_ ? g_ : "(null)", w_.c_str()); ++g_fail; } } while (0)

static std::string write_tmp(const std::vector<uint8_t>& b) {
    std::string p = std::string("vd_test_encoder_family_") + std::to_string((long)std::random_device()()) + ".gguf";
    FILE* f = std::fopen(p.c_str(), "wb");
    if (f) { std::fwrite(b.data(), 1, b.size(), f); std::fclose(f); }
    return p;
}

static void set_identity(Tiny& t, const char* arch_key, const char* name) {
    gguf_set_val_str(t.g, "general.architecture", "voicedetect");
    if (name) gguf_set_val_str(t.g, "general.name", name);
    if (arch_key) gguf_set_val_str(t.g, "voicedetect.arch", arch_key);  // replaces the value
}

struct Loaded {
    voicedetect_ctx* path = nullptr;
    voicedetect_ctx* mem = nullptr;
    voicedetect_ctx* bundled = nullptr;
    ~Loaded() {
        voicedetect_capi_free(path);
        voicedetect_capi_free(mem);
        voicedetect_capi_free(bundled);
    }
};

// Load `t` through the three paths. The bundle holds an unrelated component and a
// header with its own general.* keys, next to `t` under "voice.".
static void load_all(Tiny& t, Loaded& L) {
    Tiny other(7, "other");
    set_identity(other, "ecapa_tdnn", "someone/else");
    gguf_context* hdr = gguf_init_empty();
    gguf_set_val_str(hdr, "general.architecture", "bundle-kind");
    gguf_set_val_str(hdr, "general.name", "the-bundle");
    auto solo = make_bundle({{"", t.g, t.ctx}});
    auto bundle = make_bundle({{"", hdr, nullptr}, {"other.", other.g, other.ctx}, {"voice.", t.g, t.ctx}});
    gguf_free(hdr);

    const std::string path = write_tmp(solo);
    L.path = voicedetect_capi_load(path.c_str());
    std::remove(path.c_str());
    L.mem = voicedetect_capi_load_from_memory(solo.data(), solo.size());
    L.bundled = voicedetect_capi_load_from_memory_prefixed(bundle.data(), bundle.size(), "voice.");
}

static void test_format_and_equivalence() {
    Tiny t(1, "stem");
    set_identity(t, nullptr, "speechbrain/spkrec-ecapa-voxceleb");
    Loaded L;
    load_all(t, L);
    CHECK(L.path && L.mem && L.bundled);
    for (voicedetect_ctx* c : {L.path, L.mem, L.bundled}) {
        if (!c) continue;
        CHECK_STR(voicedetect_capi_encoder_arch(c), "wespeaker_resnet34");
        CHECK_STR(voicedetect_capi_encoder_name(c), "speechbrain/spkrec-ecapa-voxceleb");
        CHECK_STR(voicedetect_capi_encoder_family(c),
                  "voicedetect:wespeaker_resnet34:speechbrain/spkrec-ecapa-voxceleb:256");
    }
    // Same pointer on every call, and the same for repeated reads.
    if (L.path) {
        const char* a = voicedetect_capi_encoder_family(L.path);
        const char* b = voicedetect_capi_encoder_family(L.path);
        CHECK(a == b);
        CHECK(voicedetect_capi_encoder_arch(L.path) == voicedetect_capi_encoder_arch(L.path));
    }
}

static void test_missing_and_odd_keys() {
    {   // no general.name: empty field, colons stay
        Tiny t(1, "stem");
        set_identity(t, nullptr, nullptr);
        Loaded L;
        load_all(t, L);
        for (voicedetect_ctx* c : {L.path, L.mem, L.bundled}) {
            CHECK(c != nullptr);
            if (!c) continue;
            CHECK_STR(voicedetect_capi_encoder_name(c), "");
            CHECK_STR(voicedetect_capi_encoder_family(c), "voicedetect:wespeaker_resnet34::256");
        }
    }
    {   // general.architecture other than voicedetect (or absent): no family, still loads
        Tiny t(1, "stem");
        gguf_set_val_str(t.g, "general.name", "x");
        Loaded L;   // general.architecture absent
        load_all(t, L);
        for (voicedetect_ctx* c : {L.path, L.mem, L.bundled}) {
            CHECK(c != nullptr);
            if (!c) continue;
            CHECK_STR(voicedetect_capi_encoder_family(c), "");
            CHECK_STR(voicedetect_capi_encoder_name(c), "x");
            CHECK_STR(voicedetect_capi_encoder_arch(c), "wespeaker_resnet34");
        }
        Tiny u(1, "stem");
        gguf_set_val_str(u.g, "general.architecture", "parakeet");
        Loaded M;
        load_all(u, M);
        CHECK(M.path && M.mem && M.bundled);
        if (M.path) CHECK_STR(voicedetect_capi_encoder_family(M.path), "");
    }
    {   // a wrong-typed general.name must not make a loadable model fail
        Tiny t(1, "stem");
        gguf_set_val_str(t.g, "general.architecture", "voicedetect");
        gguf_set_val_u32(t.g, "general.name", 5);
        Loaded L;
        load_all(t, L);
        for (voicedetect_ctx* c : {L.path, L.mem, L.bundled}) {
            CHECK(c != nullptr);
            if (!c) continue;
            CHECK_STR(voicedetect_capi_encoder_name(c), "");
            CHECK_STR(voicedetect_capi_encoder_family(c), "voicedetect:wespeaker_resnet34::256");
        }
    }
    {   // values are copied as they are: colons, spaces and UTF-8 are not escaped
        Tiny t(1, "stem");
        set_identity(t, nullptr, "a:b c/\xC3\xA9");
        Loaded L;
        load_all(t, L);
        for (voicedetect_ctx* c : {L.path, L.mem, L.bundled}) {
            CHECK(c != nullptr);
            if (!c) continue;
            CHECK_STR(voicedetect_capi_encoder_family(c), "voicedetect:wespeaker_resnet34:a:b c/\xC3\xA9:256");
        }
    }
}

static void test_prefixed_reads_own_keys() {
    // Two components with different identities in one bundle: each prefix reports
    // its own, and a prefix never picks up the bundle header or the sibling.
    Tiny a(1, "stem"), b(2, "blk");
    set_identity(a, "ecapa_tdnn", "model-a");
    set_identity(b, "campplus", "model-b");
    gguf_context* hdr = gguf_init_empty();
    gguf_set_val_str(hdr, "general.architecture", "bundle-kind");
    gguf_set_val_str(hdr, "general.name", "the-bundle");
    auto bundle = make_bundle({{"", hdr, nullptr}, {"a.", a.g, a.ctx}, {"b.", b.g, b.ctx}});
    gguf_free(hdr);
    voicedetect_ctx* ca = voicedetect_capi_load_from_memory_prefixed(bundle.data(), bundle.size(), "a.");
    voicedetect_ctx* cb = voicedetect_capi_load_from_memory_prefixed(bundle.data(), bundle.size(), "b.");
    CHECK(ca && cb);
    if (ca) CHECK_STR(voicedetect_capi_encoder_family(ca), "voicedetect:ecapa_tdnn:model-a:256");
    if (cb) CHECK_STR(voicedetect_capi_encoder_family(cb), "voicedetect:campplus:model-b:256");
    voicedetect_capi_free(ca);
    voicedetect_capi_free(cb);
}

static void test_null_and_threads() {
    CHECK(voicedetect_capi_encoder_arch(nullptr) == nullptr);
    CHECK(voicedetect_capi_encoder_name(nullptr) == nullptr);
    CHECK(voicedetect_capi_encoder_family(nullptr) == nullptr);

    Tiny t(1, "stem");
    set_identity(t, nullptr, "n");
    auto solo = make_bundle({{"", t.g, t.ctx}});
    voicedetect_ctx* c = voicedetect_capi_load_from_memory(solo.data(), solo.size());
    CHECK(c != nullptr);
    if (!c) return;
    std::atomic<int> bad{0};
    std::vector<std::thread> th;
    for (int i = 0; i < 8; ++i)
        th.emplace_back([&] {
            for (int k = 0; k < 2000; ++k) {
                const char* f = voicedetect_capi_encoder_family(c);
                const char* a = voicedetect_capi_encoder_arch(c);
                const char* n = voicedetect_capi_encoder_name(c);
                if (!f || !a || !n || std::string(f) != "voicedetect:wespeaker_resnet34:n:256") ++bad;
            }
        });
    for (auto& x : th) x.join();
    CHECK(bad == 0);
    voicedetect_capi_free(c);
}

int main() {
    test_format_and_equivalence();
    test_missing_and_odd_keys();
    test_prefixed_reads_own_keys();
    test_null_and_threads();
    std::fprintf(stderr, g_fail ? "FAILED (%d)\n" : "PASS\n", g_fail);
    return g_fail ? 1 : 0;
}
