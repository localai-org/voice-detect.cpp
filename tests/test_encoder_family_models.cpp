// Encoder family of real speaker encoders, through every load path.
//
// VOICEDETECT_TEST_GGUF_LIST is a list of GGUF files separated by ':' (for
// example the ECAPA-TDNN, CAM++, WeSpeaker ResNet34 and ERes2Net f32 models).
// For each file the family from the path loader, the memory loader and the
// prefixed loader (the file inside a bundle next to a second model) must be
// identical and must equal the string the documented formula gives for the keys
// read straight from the file header. Exits 77 (skip) when the variable is
// unset. Each family is printed, so another implementation of the same formula
// (parakeet.cpp) can be compared with it.
#include "gguf_bundle.hpp"

#include "voicedetect_capi.h"

#include <cstdio>
#include <cstdlib>
#include <string>

using namespace vdtest;

static int g_fail = 0;
#define CHECK(c) do { if (!(c)) { std::fprintf(stderr, "FAIL %s:%d: %s\n", __FILE__, __LINE__, #c); ++g_fail; } } while (0)

static std::vector<uint8_t> read_file(const std::string& p) {
    std::vector<uint8_t> b;
    FILE* f = std::fopen(p.c_str(), "rb");
    if (!f) return b;
    uint8_t tmp[1 << 16];
    size_t n;
    while ((n = std::fread(tmp, 1, sizeof tmp, f)) > 0) b.insert(b.end(), tmp, tmp + n);
    std::fclose(f);
    return b;
}

static std::string key_str(gguf_context* g, const char* k) {
    const int64_t id = gguf_find_key(g, k);
    return (id >= 0 && gguf_get_kv_type(g, id) == GGUF_TYPE_STRING) ? gguf_get_val_str(g, id) : "";
}

// The documented formula, computed from the header of the file itself.
static std::string expected_family(const std::string& path, std::string* arch, std::string* name) {
    gguf_init_params ip{true, nullptr};
    gguf_context* g = gguf_init_from_file(path.c_str(), ip);
    if (!g) return "";
    *arch = key_str(g, "voicedetect.arch");
    *name = key_str(g, "general.name");
    std::string out;
    if (key_str(g, "general.architecture") == "voicedetect") {
        const int64_t id = gguf_find_key(g, "voicedetect.embedding_dim");
        const uint32_t dim = (id >= 0 && gguf_get_kv_type(g, id) == GGUF_TYPE_UINT32) ? gguf_get_val_u32(g, id) : 0;
        out = "voicedetect:" + *arch + ":" + *name + ":" + (dim ? std::to_string(dim) : "");
    }
    gguf_free(g);
    return out;
}

static void check_file(const std::string& path) {
    std::string arch, name;
    const std::string want = expected_family(path, &arch, &name);
    CHECK(want.rfind("voicedetect:", 0) == 0);
    std::printf("%s\n", want.c_str());

    voicedetect_ctx* p = voicedetect_capi_load(path.c_str());
    CHECK(p != nullptr);
    std::vector<uint8_t> bytes = read_file(path);
    voicedetect_ctx* m = voicedetect_capi_load_from_memory(bytes.data(), bytes.size());
    CHECK(m != nullptr);

    // Prefixed: the same file inside a bundle, next to a synthetic second model
    // that has another identity. The prefix is short ("v.") because a ggml
    // tensor name, prefix included, is limited to 63 bytes.
    ggml_context* ctx = nullptr;
    gguf_init_params ip{false, &ctx};
    gguf_context* g = gguf_init_from_file(path.c_str(), ip);
    CHECK(g != nullptr);
    Tiny other(2, "other");
    gguf_set_val_str(other.g, "general.architecture", "voicedetect");
    gguf_set_val_str(other.g, "general.name", "other-model");
    auto bundle = make_bundle({{"other.", other.g, other.ctx}, {"v.", g, ctx}});
    gguf_free(g);
    ggml_free(ctx);
    voicedetect_ctx* b = voicedetect_capi_load_from_memory_prefixed(bundle.data(), bundle.size(), "v.");
    CHECK(b != nullptr);

    for (voicedetect_ctx* c : {p, m, b}) {
        if (!c) continue;
        const char* f = voicedetect_capi_encoder_family(c);
        CHECK(f && want == f);
        const char* a = voicedetect_capi_encoder_arch(c);
        const char* n = voicedetect_capi_encoder_name(c);
        CHECK(a && arch == a);
        CHECK(n && name == n);
        // The last field is the embedding size the context reports.
        CHECK(f && std::string(f).size() > 1 &&
              std::string(f).substr(std::string(f).rfind(':') + 1) == std::to_string(voicedetect_capi_embedding_dim(c)));
    }
    voicedetect_capi_free(p);
    voicedetect_capi_free(m);
    voicedetect_capi_free(b);
}

int main() {
    const char* list = std::getenv("VOICEDETECT_TEST_GGUF_LIST");
    if (!list || !*list) return 77;
    std::string s = list;
    size_t pos = 0;
    int n = 0;
    while (pos <= s.size()) {
        size_t e = s.find(':', pos);
        if (e == std::string::npos) e = s.size();
        if (e > pos) { check_file(s.substr(pos, e - pos)); ++n; }
        pos = e + 1;
    }
    CHECK(n > 0);
    std::fprintf(stderr, g_fail ? "FAILED (%d)\n" : "PASS (%d models)\n", g_fail ? g_fail : n);
    return g_fail ? 1 : 0;
}
