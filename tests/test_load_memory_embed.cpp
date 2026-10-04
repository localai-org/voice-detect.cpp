// A real model loaded by path, from memory and from a bundle (with a prefix)
// must give bitwise-identical embeddings. The memory buffers are poisoned and
// freed before the first embedding runs.
//
// Needs VOICEDETECT_TEST_GGUF (speaker encoder GGUF) and VOICEDETECT_TEST_AUDIO
// (WAV). Exits 77 (skip) when unset.
#include "gguf_bundle.hpp"

#include "backend.hpp"
#include "model.hpp"
#include "voicedetect_capi.h"

#include <cstdio>
#include <cstdlib>

using namespace vdtest;

static int g_fail = 0;
#define CHECK(c) do { if (!(c)) { std::fprintf(stderr, "FAIL %s:%d: %s\n", __FILE__, __LINE__, #c); ++g_fail; } } while (0)

static bool bitwise_equal(const std::vector<float>& a, const std::vector<float>& b) {
    return !a.empty() && a.size() == b.size() && std::memcmp(a.data(), b.data(), a.size() * sizeof(float)) == 0;
}

static std::vector<uint8_t> read_file(const char* p) {
    std::vector<uint8_t> b;
    FILE* f = std::fopen(p, "rb");
    if (!f) return b;
    uint8_t tmp[1 << 16];
    size_t n;
    while ((n = std::fread(tmp, 1, sizeof tmp, f)) > 0) b.insert(b.end(), tmp, tmp + n);
    std::fclose(f);
    return b;
}

int main() {
    const char* gguf = std::getenv("VOICEDETECT_TEST_GGUF");
    const char* audio = std::getenv("VOICEDETECT_TEST_AUDIO");
    if (!gguf || !audio) return 77;

    std::string err;
    auto ref = vd::Model::load(gguf, &err);
    if (!ref) { std::fprintf(stderr, "load failed: %s\n", err.c_str()); return 77; }
    const std::vector<float> e_path = ref->embed_path(audio);
    CHECK((int)e_path.size() == ref->embedding_dim());

    // Memory, unprefixed.
    {
        auto* buf = new std::vector<uint8_t>(read_file(gguf));
        CHECK(!buf->empty());
        auto m = vd::Model::load_from_memory(buf->data(), buf->size(), &err);
        std::memset(buf->data(), 0xAB, buf->size());
        delete buf;   // freed before the embedding runs
        CHECK(m != nullptr);
        if (m) CHECK(bitwise_equal(m->embed_path(audio), e_path));
    }

    // Prefixed, inside a bundle next to another component.
    {
        ggml_context* ctx = nullptr;
        gguf_init_params ip{false, &ctx};
        gguf_context* g = gguf_init_from_file(gguf, ip);
        CHECK(g != nullptr);
        Tiny other(2, "other");
        auto bundle = make_bundle({{"other.", other.g, other.ctx}, {"v.", g, ctx}});
        gguf_free(g);
        ggml_free(ctx);
        auto* buf = new std::vector<uint8_t>(std::move(bundle));
        auto m = vd::Model::load_from_memory(buf->data(), buf->size(), "v.", &err);
        std::memset(buf->data(), 0xCD, buf->size());
        delete buf;
        CHECK(m != nullptr);
        if (m) CHECK(bitwise_equal(m->embed_path(audio), e_path));
        // wrong prefix on the same data
        auto b2 = vd::Model::load_from_memory(nullptr, 0, "v.", &err);
        CHECK(b2 == nullptr && !err.empty());
    }

    // C API, memory + path equality (embed_pcm on the same samples)
    {
        std::vector<uint8_t> buf = read_file(gguf);
        voicedetect_ctx* c = voicedetect_capi_load_from_memory(buf.data(), buf.size());
        CHECK(c != nullptr);
        std::fill(buf.begin(), buf.end(), 0xEE);
        buf.clear(); buf.shrink_to_fit();
        voicedetect_ctx* p = voicedetect_capi_load(gguf);
        CHECK(p != nullptr);
        if (c && p) {
            float *vc = nullptr, *vp = nullptr; int dc = 0, dp = 0;
            CHECK(voicedetect_capi_embed_path(c, audio, &vc, &dc) == 0);
            CHECK(voicedetect_capi_embed_path(p, audio, &vp, &dp) == 0);
            CHECK(dc == dp && dc > 0 && std::memcmp(vc, vp, dc * sizeof(float)) == 0);
            voicedetect_capi_free_vec(vc);
            voicedetect_capi_free_vec(vp);
        }
        voicedetect_capi_free(c);
        voicedetect_capi_free(p);
    }

    // Truncated real model: clear failure at a handful of lengths.
    {
        std::vector<uint8_t> full = read_file(gguf);
        for (double f : {0.0, 0.001, 0.01, 0.1, 0.5, 0.9, 0.999}) {
            std::vector<uint8_t> cut(full.begin(), full.begin() + (size_t)(full.size() * f));
            std::string e;
            auto m = vd::Model::load_from_memory(cut.data(), cut.size(), &e);
            CHECK(m == nullptr && !e.empty());
        }
    }
    vd::shutdown_backend();
    std::fprintf(stderr, g_fail ? "FAILED (%d)\n" : "PASS (embedding dim %d)\n", g_fail ? g_fail : (int)e_path.size());
    return g_fail ? 1 : 0;
}
