// voicedetect_capi_embedding_dim: NULL ctx is -1; with VOICEDETECT_TEST_GGUF set,
// it matches the length embed_pcm returns. Skips (77) without a model.
#include "voicedetect_capi.h"

#include <cstdio>
#include <cstdlib>
#include <vector>

int main() {
    if (voicedetect_capi_embedding_dim(nullptr) != -1) {
        std::fprintf(stderr, "FAIL: NULL ctx should give -1\n");
        return 1;
    }
    const char* path = std::getenv("VOICEDETECT_TEST_GGUF");
    if (!path) return 77;
    voicedetect_ctx* ctx = voicedetect_capi_load(path);
    if (!ctx) { std::fprintf(stderr, "FAIL: load %s\n", path); return 1; }
    const int dim = voicedetect_capi_embedding_dim(ctx);
    std::vector<float> pcm(32000);
    for (size_t i = 0; i < pcm.size(); ++i) pcm[i] = 0.1f * (float)((i * 37) % 200 - 100) / 100.0f;
    float* v = nullptr;
    int n = 0;
    const int rc = voicedetect_capi_embed_pcm(ctx, pcm.data(), (int)pcm.size(), 16000, &v, &n);
    voicedetect_capi_free_vec(v);
    voicedetect_capi_free(ctx);
    if (rc != 0 || dim <= 0 || dim != n) {
        std::fprintf(stderr, "FAIL: rc=%d dim=%d embed len=%d\n", rc, dim, n);
        return 1;
    }
    std::printf("embedding_dim OK (%d)\n", dim);
    return 0;
}
