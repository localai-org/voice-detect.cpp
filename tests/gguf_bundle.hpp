// Test helpers: build GGUF files and bundles entirely in memory (no files).
#pragma once
#include "ggml.h"
#include "gguf.h"

#include <cstdint>
#include <cstring>
#include <memory>
#include <string>
#include <vector>

namespace vdtest {

// One model to place in a bundle: a GGUF context plus the ggml context that
// owns its tensor data (as returned by gguf_init_from_*(no_alloc=false)).
struct Source {
    std::string prefix;   // "" = standalone
    gguf_context* g;
    ggml_context* ctx;
};

// Copy one key of `g` into `out` as `name`. Covers the types voice-detect uses.
inline void copy_kv(gguf_context* out, const gguf_context* g, int64_t i, const std::string& name) {
    const char* k = name.c_str();
    switch (gguf_get_kv_type(g, i)) {
        case GGUF_TYPE_UINT32:  gguf_set_val_u32(out, k, gguf_get_val_u32(g, i)); break;
        case GGUF_TYPE_INT32:   gguf_set_val_i32(out, k, gguf_get_val_i32(g, i)); break;
        case GGUF_TYPE_FLOAT32: gguf_set_val_f32(out, k, gguf_get_val_f32(g, i)); break;
        case GGUF_TYPE_BOOL:    gguf_set_val_bool(out, k, gguf_get_val_bool(g, i)); break;
        case GGUF_TYPE_STRING:  gguf_set_val_str(out, k, gguf_get_val_str(g, i)); break;
        case GGUF_TYPE_UINT8:   gguf_set_val_u8(out, k, gguf_get_val_u8(g, i)); break;
        case GGUF_TYPE_UINT64:  gguf_set_val_u64(out, k, gguf_get_val_u64(g, i)); break;
        case GGUF_TYPE_ARRAY: {
            const gguf_type at = gguf_get_arr_type(g, i);
            const size_t n = gguf_get_arr_n(g, i);
            if (at == GGUF_TYPE_STRING) {
                std::vector<const char*> v(n);
                for (size_t j = 0; j < n; ++j) v[j] = gguf_get_arr_str(g, i, j);
                gguf_set_arr_str(out, k, v.data(), n);
            } else if (at != GGUF_TYPE_ARRAY) {
                gguf_set_arr_data(out, k, at, gguf_get_arr_data(g, i), n);
            }
            break;
        }
        default: break;
    }
}

// Serialize `g` to bytes. `data_of(name)` returns the tensor bytes.
template <class F>
std::vector<uint8_t> serialize(gguf_context* g, F data_of) {
    const size_t meta = gguf_get_meta_size(g);
    std::vector<uint8_t> out(meta);
    gguf_get_meta_data(g, out.data());
    const int64_t nt = gguf_get_n_tensors(g);
    for (int64_t i = 0; i < nt; ++i) {
        const size_t off = meta + gguf_get_tensor_offset(g, i);
        const size_t sz = gguf_get_tensor_size(g, i);
        if (out.size() < off + sz) out.resize(off + sz, 0);
        std::memcpy(out.data() + off, data_of(gguf_get_tensor_name(g, i)), sz);
    }
    // The data section is a whole number of alignment units (ggml's file reader
    // checks the size of the last tensor with padding).
    const size_t al = gguf_get_alignment(g);
    out.resize((out.size() + al - 1) / al * al, 0);
    return out;
}

// Build one GGUF holding every source, each under its prefix (a bundle), or a
// plain GGUF when there is one source with an empty prefix.
inline std::vector<uint8_t> make_bundle(const std::vector<Source>& srcs) {
    gguf_context* out = gguf_init_empty();
    ggml_init_params ip{ggml_tensor_overhead() * 4096 + 65536, nullptr, true};
    ggml_context* tctx = ggml_init(ip);
    std::vector<std::pair<std::string, const void*>> data;
    for (const Source& s : srcs) {
        for (int64_t i = 0; i < gguf_get_n_kv(s.g); ++i) {
            const std::string k = gguf_get_key(s.g, i);
            if (k == "general.alignment") continue;
            copy_kv(out, s.g, i, s.prefix + k);
        }
        for (int64_t i = 0; i < gguf_get_n_tensors(s.g); ++i) {
            const char* nm = gguf_get_tensor_name(s.g, i);
            ggml_tensor* src = ggml_get_tensor(s.ctx, nm);
            ggml_tensor* t = ggml_new_tensor(tctx, src->type, GGML_MAX_DIMS, src->ne);
            const std::string full = s.prefix + nm;
            ggml_set_name(t, full.c_str());
            gguf_add_tensor(out, t);
            data.emplace_back(full, src->data);
        }
    }
    auto bytes = serialize(out, [&](const char* nm) -> const void* {
        for (auto& d : data) if (d.first == nm) return d.second;
        return nullptr;
    });
    gguf_free(out);
    ggml_free(tctx);
    return bytes;
}

// Tiny synthetic speaker model (arch wespeaker_resnet34 manifest, so the tensor
// names live in string-array metadata). Not runnable; enough to load.
struct Tiny {
    gguf_context* g = nullptr;
    ggml_context* ctx = nullptr;
    ~Tiny() { if (g) gguf_free(g); if (ctx) ggml_free(ctx); }
    Tiny(const Tiny&) = delete;
    Tiny& operator=(const Tiny&) = delete;
    // `seed` varies the weights; `name_tag` varies the tensor names.
    Tiny(uint32_t seed, const char* name_tag, bool bad_type = false) {
        ggml_init_params ip{1 << 20, nullptr, false};
        ctx = ggml_init(ip);
        g = gguf_init_empty();
        gguf_set_val_str(g, "voicedetect.arch", "wespeaker_resnet34");
        if (bad_type) gguf_set_val_str(g, "voicedetect.embedding_dim", "256");  // wrong type
        else gguf_set_val_u32(g, "voicedetect.embedding_dim", 256);
        gguf_set_val_bool(g, "voicedetect.l2_normalize", true);
        gguf_set_val_f32(g, "voicedetect.resnet.var_eps", 1e-8f);
        const std::string w = std::string(name_tag) + ".w", b = std::string(name_tag) + ".b";
        gguf_set_val_str(g, "voicedetect.resnet.stem_weight", w.c_str());
        gguf_set_val_str(g, "voicedetect.resnet.stem_bias", b.c_str());
        const char* ws[] = {w.c_str()};
        const char* bs[] = {b.c_str()};
        gguf_set_arr_str(g, "voicedetect.resnet.conv1_weight", ws, 1);
        gguf_set_arr_str(g, "voicedetect.resnet.conv1_bias", bs, 1);
        int32_t st[] = {1};
        gguf_set_arr_data(g, "voicedetect.resnet.stride", GGUF_TYPE_INT32, st, 1);
        ggml_tensor* tw = ggml_new_tensor_4d(ctx, GGML_TYPE_F32, 3, 3, 2, 5);
        ggml_tensor* tb = ggml_new_tensor_1d(ctx, GGML_TYPE_F16, 5);
        ggml_tensor* tq = ggml_new_tensor_2d(ctx, GGML_TYPE_Q8_0, 64, 3);
        ggml_set_name(tw, w.c_str());
        ggml_set_name(tb, b.c_str());
        const std::string q = std::string(name_tag) + ".q";
        ggml_set_name(tq, q.c_str());
        uint32_t x = seed * 2654435761u + 1;
        for (ggml_tensor* t : {tw, tb, tq}) {
            uint8_t* p = (uint8_t*)t->data;
            for (size_t i = 0; i < ggml_nbytes(t); ++i) { x = x * 1664525u + 1013904223u; p[i] = (uint8_t)(x >> 24); }
            gguf_add_tensor(g, t);
        }
    }
};

}  // namespace vdtest
