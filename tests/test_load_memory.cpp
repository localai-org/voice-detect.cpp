// Loading a model from a memory buffer (no model files needed: the GGUFs are
// synthesized in memory, so this runs in CI).
//
//  * same tensors and config through the path, memory and prefixed loaders
//  * the bundle's other component is not loaded
//  * the buffer is not referenced after the call (poisoned, freed)
//  * NULL, empty, truncated (every length) and corrupt buffers are rejected
//    with a message, never a crash (run under ASan/UBSan to prove no OOB read)
//  * concurrent loads
#include "gguf_bundle.hpp"

#include "model_loader.hpp"
#include "model.hpp"
#include "voicedetect_capi.h"

#include <algorithm>
#include <atomic>
#include <cstdio>
#include <cstdlib>
#include <random>
#include <thread>

using namespace vdtest;

static int g_fail = 0;
#define CHECK(c) do { if (!(c)) { std::fprintf(stderr, "FAIL %s:%d: %s\n", __FILE__, __LINE__, #c); ++g_fail; } } while (0)

static const char* kNames[] = {"stem.w", "stem.b", "stem.q"};

// Compare the three tensors of a loader with those of a Tiny.
static bool same_tensors(const vd::ModelLoader& L, const Tiny& t, const char* tag) {
    for (const char* suffix : {".w", ".b", ".q"}) {
        const std::string nm = std::string(tag) + suffix;
        ggml_tensor* a = L.tensor(nm);
        ggml_tensor* b = ggml_get_tensor(t.ctx, nm.c_str());
        if (!a || !b || a->type != b->type || ggml_nbytes(a) != ggml_nbytes(b)) return false;
        for (int d = 0; d < GGML_MAX_DIMS; ++d) if (a->ne[d] != b->ne[d]) return false;
        if (std::memcmp(a->data, b->data, ggml_nbytes(a)) != 0) return false;
    }
    return true;
}

static bool config_ok(const vd::ModelLoader& L, const char* tag) {
    const auto& c = L.config();
    return c.arch == "wespeaker_resnet34" && c.embedding_dim == 256 && c.resnet.conv1_weight.size() == 1 &&
           c.resnet.conv1_weight[0] == std::string(tag) + ".w" && c.resnet.stem_bias == std::string(tag) + ".b" &&
           c.resnet.stride.size() == 1 && L.tensor(c.resnet.conv1_weight[0]) != nullptr;
}

static std::string write_tmp(const std::vector<uint8_t>& b) {
    std::string p = std::string("vd_test_load_memory_") + std::to_string((long)std::random_device()()) + ".gguf";
    FILE* f = std::fopen(p.c_str(), "wb");
    if (f) { std::fwrite(b.data(), 1, b.size(), f); std::fclose(f); }
    return p;
}

static void test_equivalence() {
    Tiny voice(1, "stem"), other(2, "other");
    auto solo = make_bundle({{"", voice.g, voice.ctx}});
    auto bundle = make_bundle({{"other.", other.g, other.ctx}, {"voice.", voice.g, voice.ctx}});

    // Path loader (reference)
    std::string path = write_tmp(solo);
    vd::ModelLoader P;
    CHECK(P.load(path));
    std::remove(path.c_str());
    CHECK(same_tensors(P, voice, "stem") && config_ok(P, "stem"));

    // Memory loader; buffer is poisoned and freed before anything is compared.
    {
        auto* heap = new std::vector<uint8_t>(solo);
        vd::ModelLoader M;
        CHECK(M.load_from_memory(heap->data(), heap->size()));
        std::memset(heap->data(), 0xAB, heap->size());
        delete heap;
        CHECK(same_tensors(M, voice, "stem") && config_ok(M, "stem"));
        CHECK(M.error().empty());
        CHECK(!M.load_from_memory(solo.data(), solo.size()));   // second load refused
    }
    // Prefixed loader on the whole bundle.
    {
        auto* heap = new std::vector<uint8_t>(bundle);
        vd::ModelLoader B;
        CHECK(B.load_from_memory(heap->data(), heap->size(), "voice."));
        std::memset(heap->data(), 0xCD, heap->size());
        delete heap;
        CHECK(same_tensors(B, voice, "stem") && config_ok(B, "stem"));
        CHECK(B.tensor("other.stem.w") == nullptr && B.tensor("voice.stem.w") == nullptr);
        CHECK(B.tensor("other.w") == nullptr);
        // the other component loads on its own from the same bundle
        vd::ModelLoader O;
        CHECK(O.load_from_memory(bundle.data(), bundle.size(), "other."));
        CHECK(same_tensors(O, other, "other") && config_ok(O, "other"));
        CHECK(!same_tensors(O, voice, "stem"));
    }
    // Empty prefix == standalone
    {
        vd::ModelLoader E;
        CHECK(E.load_from_memory(solo.data(), solo.size(), ""));
        CHECK(same_tensors(E, voice, "stem"));
    }
    // Weights can be realized on the CPU backend after a prefixed load (the
    // tensors are laid out in one contiguous ctx buffer).
    // (covered by the embedding test with real models)
    (void)kNames;
}

static void test_bad_inputs() {
    Tiny voice(1, "stem"), other(2, "other");
    auto solo = make_bundle({{"", voice.g, voice.ctx}});
    auto bundle = make_bundle({{"other.", other.g, other.ctx}, {"voice.", voice.g, voice.ctx}});
    std::string msg;
    {
        vd::ModelLoader L;
        CHECK(!L.load_from_memory(nullptr, 100)); CHECK(!L.error().empty());
        CHECK(!L.load_from_memory(solo.data(), 0)); CHECK(!L.error().empty());
        CHECK(!L.load_from_memory(solo.data(), 1));
        CHECK(!L.load_from_memory(nullptr, 0, "voice."));
        CHECK(!L.load_from_memory(solo.data(), 0, "voice."));
    }
    // wrong / partial / unknown prefix
    for (const char* pre : {"nope.", "voic", "voice", "v", "voice.stem.", "other.stem.", "VOICE."}) {
        vd::ModelLoader L;
        bool ok = L.load_from_memory(bundle.data(), bundle.size(), pre);
        CHECK(!ok);
        CHECK(!L.error().empty());
    }
    { vd::ModelLoader L; CHECK(!L.load_from_memory(solo.data(), solo.size(), "voice.")); }   // solo has no prefix
    { vd::ModelLoader L; CHECK(!L.load_from_memory(bundle.data(), bundle.size())); }          // bundle is not a model unprefixed

    // wrong key type must be an error, not an abort in ggml
    {
        Tiny bad(3, "stem", /*bad_type=*/true);
        auto b = make_bundle({{"", bad.g, bad.ctx}});
        vd::ModelLoader L;
        CHECK(!L.load_from_memory(b.data(), b.size()));
        CHECK(L.error().find("unexpected type") != std::string::npos);
        auto bb = make_bundle({{"voice.", bad.g, bad.ctx}});
        vd::ModelLoader L2;
        CHECK(!L2.load_from_memory(bb.data(), bb.size(), "voice."));
        CHECK(L2.error().find("unexpected type") != std::string::npos);
    }

    // Truncation at EVERY length, both loaders. `voice.` is the last component,
    // so any cut removes bytes the prefixed load needs.
    // Only the alignment padding after the last tensor may be missing.
    auto first_ok = [&](const std::vector<uint8_t>& full, const char* prefix) {
        size_t first = full.size() + 1;
        for (size_t len = 0; len <= full.size(); ++len) {
            std::vector<uint8_t> cut(full.begin(), full.begin() + len);   // exact-size heap block: ASan sees any overread
            vd::ModelLoader L;
            bool ok = prefix ? L.load_from_memory(cut.data(), cut.size(), prefix)
                             : L.load_from_memory(cut.data(), cut.size());
            if (!ok) CHECK(!L.error().empty());
            if (ok && first > len) first = len;
            if (!ok) CHECK(first > len);   // once a length loads, every longer one does too
        }
        return first;
    };
    const size_t fs = first_ok(solo, nullptr), fb = first_ok(bundle, "voice.");
    CHECK(fs <= solo.size() && solo.size() - fs < gguf_get_alignment(voice.g));
    CHECK(fb <= bundle.size() && bundle.size() - fb < gguf_get_alignment(voice.g));
    // Extra trailing bytes are harmless.
    { auto big = solo; big.resize(big.size() + 100, 0); vd::ModelLoader L; CHECK(L.load_from_memory(big.data(), big.size())); }

    // Targeted corruption of the bundle header.
    auto patch64 = [&](size_t off, uint64_t v) { auto b = bundle; std::memcpy(b.data() + off, &v, 8); return b; };
    std::vector<std::vector<uint8_t>> bad;
    { auto b = bundle; b[0] = 'X'; bad.push_back(b); }                      // magic
    { auto b = bundle; b[4] = 99; bad.push_back(b); }                       // version
    bad.push_back(patch64(8, ~0ull));                                       // n_tensors huge
    bad.push_back(patch64(8, 1ull << 40));
    bad.push_back(patch64(16, ~0ull));                                      // n_kv huge
    bad.push_back(patch64(16, 1ull << 40));
    // tensor-info offset out of range: find the info record of "voice.stem.w"
    {
        const std::string nm = "voice.stem.w";
        auto it = std::search(bundle.begin(), bundle.end(), nm.begin(), nm.end());
        CHECK(it != bundle.end());
        size_t p = (it - bundle.begin()) + nm.size();
        uint32_t nd; std::memcpy(&nd, &bundle[p], 4);
        size_t off_pos = p + 4 + 8 * nd + 4;
        bad.push_back(patch64(off_pos, 1ull << 40));
        bad.push_back(patch64(off_pos, ~0ull));
        bad.push_back(patch64(off_pos, bundle.size()));
        bad.push_back(patch64(off_pos, bundle.size() - 8));
        // dimension grows: size no longer fits the data
        bad.push_back(patch64(p + 4, 1ull << 40));
        bad.push_back(patch64(p + 4, ~0ull));
        // type id out of range
        { auto b = bundle; uint32_t t = 9999; std::memcpy(&b[p + 4 + 8 * nd], &t, 4); bad.push_back(b); }
    }
    for (size_t i = 0; i < bad.size(); ++i) {
        std::vector<uint8_t> b = bad[i];
        vd::ModelLoader L1, L2;
        bool a = L1.load_from_memory(b.data(), b.size(), "voice.");
        bool c = L2.load_from_memory(b.data(), b.size());
        if (a || c) std::fprintf(stderr, "corrupt case %zu accepted (prefixed=%d plain=%d)\n", i, a, c);
        CHECK(!a);
        CHECK(!c);
        CHECK((!a && !L1.error().empty()));
    }

    // A metadata key with an EMPTY name makes ggml's reader abort the process;
    // the loader must refuse it first. Build the buffer by hand.
    {
        auto b = solo;
        // first key is "voicedetect.arch" after the 24-byte header: len(8)+name
        uint64_t zero = 0;
        std::memcpy(b.data() + 24, &zero, 8);   // key length 0
        vd::ModelLoader L1, L2;
        CHECK(!L1.load_from_memory(b.data(), b.size()));
        CHECK(!L2.load_from_memory(b.data(), b.size(), "voice."));
        CHECK(L1.error().find("empty name") != std::string::npos);
    }
    // Random byte flips across the header/table region and the data: must never
    // crash; success is allowed only when the result is a valid model.
    std::mt19937 rng(1234);
    const size_t hdr = std::min<size_t>(bundle.size(), 1500);
    int accepted = 0;
    for (int it = 0; it < 6000; ++it) {
        std::vector<uint8_t> b = bundle;
        int flips = 1 + rng() % 4;
        for (int f = 0; f < flips; ++f) b[rng() % hdr] ^= (uint8_t)(1u << (rng() % 8)) | (uint8_t)(rng() % 255);
        vd::ModelLoader L;
        if (L.load_from_memory(b.data(), b.size(), "voice.")) {
            ++accepted;
            CHECK(L.config().embedding_dim > 0 || L.config().analyze_present);
        }
        vd::ModelLoader L2;
        (void)L2.load_from_memory(b.data(), b.size());
    }
    std::fprintf(stderr, "fuzz: %d of 6000 mutated buffers still loaded as valid models\n", accepted);
}

static void test_capi() {
    Tiny voice(1, "stem"), other(2, "other");
    auto solo = make_bundle({{"", voice.g, voice.ctx}});
    auto bundle = make_bundle({{"other.", other.g, other.ctx}, {"voice.", voice.g, voice.ctx}});
    CHECK(voicedetect_capi_load_from_memory(nullptr, 10) == nullptr);
    CHECK(std::strlen(voicedetect_capi_last_load_error()) > 0);
    CHECK(voicedetect_capi_load_from_memory(solo.data(), 0) == nullptr);
    CHECK(voicedetect_capi_load_from_memory(solo.data(), solo.size() / 2) == nullptr);
    CHECK(voicedetect_capi_load_from_memory_prefixed(bundle.data(), bundle.size(), "nope.") == nullptr);
    CHECK(std::string(voicedetect_capi_last_load_error()).find("nope.") != std::string::npos);
    CHECK(voicedetect_capi_load("/nonexistent/model.gguf") == nullptr);
    CHECK(std::strlen(voicedetect_capi_last_load_error()) > 0);
    CHECK(voicedetect_capi_load(nullptr) == nullptr);
    // These synthetic models load but cannot embed (no real weights); the
    // context must still be valid, report errors cleanly, and free.
    voicedetect_ctx* c = voicedetect_capi_load_from_memory(solo.data(), solo.size());
    CHECK(c != nullptr);
    CHECK(std::strlen(voicedetect_capi_last_load_error()) == 0);
    voicedetect_capi_free(c);
    c = voicedetect_capi_load_from_memory_prefixed(bundle.data(), bundle.size(), "voice.");
    CHECK(c != nullptr);
    voicedetect_capi_free(c);
    c = voicedetect_capi_load_from_memory_prefixed(solo.data(), solo.size(), nullptr);  // NULL prefix == plain
    CHECK(c != nullptr);
    voicedetect_capi_free(c);
    voicedetect_capi_free(nullptr);
}

static void test_threads() {
    Tiny voice(1, "stem"), other(2, "other");
    auto bundle = make_bundle({{"other.", other.g, other.ctx}, {"voice.", voice.g, voice.ctx}});
    auto solo = make_bundle({{"", voice.g, voice.ctx}});
    std::atomic<int> bad{0};
    std::vector<std::thread> th;
    for (int t = 0; t < 8; ++t)
        th.emplace_back([&, t] {
            for (int i = 0; i < 30; ++i) {
                vd::ModelLoader L;
                // each thread also loads from its own private copy of the buffer
                std::vector<uint8_t> mine = (t & 1) ? bundle : solo;
                bool ok = (t & 1) ? L.load_from_memory(mine.data(), mine.size(), "voice.")
                                  : L.load_from_memory(mine.data(), mine.size());
                std::fill(mine.begin(), mine.end(), 0);
                if (!ok || !same_tensors(L, voice, "stem") || !config_ok(L, "stem")) ++bad;
                // a failing load on the same thread must not disturb the others
                voicedetect_ctx* x = voicedetect_capi_load_from_memory(solo.data(), 7);
                if (x) ++bad;
            }
        });
    for (auto& x : th) x.join();
    CHECK(bad == 0);
}

int main() {
    test_equivalence();
    test_bad_inputs();
    test_capi();
    test_threads();
    std::fprintf(stderr, g_fail ? "FAILED (%d)\n" : "PASS\n", g_fail);
    return g_fail ? 1 : 0;
}
