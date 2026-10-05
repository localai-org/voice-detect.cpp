#include "model_loader.hpp"
#include "common.hpp"
#include "backend.hpp"
#include "gguf_check.hpp"
#include "ggml.h"
#include "ggml-backend.h"
#include "ggml-alloc.h"
#include "gguf.h"
#include <cstring>
#include <cstdint>
#include <vector>
#include <utility>
namespace vd {

// A GGUF plus an optional key prefix. Every key is looked up as prefix+key, so
// a model stored inside a larger GGUF (a bundle) reads like a standalone file.
struct Kv {
    gguf_context* g;
    std::string   prefix;
    mutable std::string err;   // first type mismatch; ggml would abort on one
    // Index of key `k` if present with scalar type `want`; -1 if absent. A key of
    // another type records an error instead of reaching ggml's asserting getters.
    int64_t find(const char* k, gguf_type want) const {
        int64_t id = gguf_find_key(g, (prefix + k).c_str());
        if(id < 0) return -1;
        if(gguf_get_kv_type(g, id) != want){
            if(err.empty()) err = std::string("GGUF key '") + prefix + k + "' has an unexpected type";
            return -1;
        }
        return id;
    }
    // Array key whose elements have type `elem`.
    int64_t find_arr(const char* k, gguf_type elem) const {
        int64_t id = find(k, GGUF_TYPE_ARRAY);
        if(id < 0) return -1;
        if(gguf_get_arr_type(g, id) != elem){
            if(err.empty()) err = std::string("GGUF key '") + prefix + k + "' has an unexpected element type";
            return -1;
        }
        return id;
    }
};
static uint32_t kv_u32(const Kv& kv, const char* k, uint32_t d=0){
    int64_t id = kv.find(k, GGUF_TYPE_UINT32); return id<0 ? d : gguf_get_val_u32(kv.g,id);
}
static float kv_f32(const Kv& kv, const char* k, float d=0){
    int64_t id = kv.find(k, GGUF_TYPE_FLOAT32); return id<0 ? d : gguf_get_val_f32(kv.g,id);
}
static bool kv_bool(const Kv& kv, const char* k, bool d=false){
    int64_t id = kv.find(k, GGUF_TYPE_BOOL); return id<0 ? d : gguf_get_val_bool(kv.g,id);
}
static std::string kv_str(const Kv& kv, const char* k, const char* d=""){
    int64_t id = kv.find(k, GGUF_TYPE_STRING); return id<0 ? std::string(d) : std::string(gguf_get_val_str(kv.g,id));
}
// A string key that is absent or of another type reads as "". Unlike kv_str it
// never records a load error: the identity keys are descriptive, and a model
// that loaded before must still load.
static std::string kv_str_soft(const Kv& kv, const char* k){
    int64_t id = gguf_find_key(kv.g, (kv.prefix + k).c_str());
    if(id < 0 || gguf_get_kv_type(kv.g, id) != GGUF_TYPE_STRING) return std::string();
    return std::string(gguf_get_val_str(kv.g, id));
}
static std::vector<std::string> kv_str_arr(const Kv& kv, const char* k){
    std::vector<std::string> out;
    int64_t id = kv.find_arr(k, GGUF_TYPE_STRING);
    if(id>=0){
        size_t n = gguf_get_arr_n(kv.g,id);
        out.resize(n);
        for(size_t i=0;i<n;++i) out[i] = gguf_get_arr_str(kv.g,id,i);
    }
    return out;
}
static std::vector<int32_t> kv_i32_arr(const Kv& kv, const char* k){
    std::vector<int32_t> out;
    int64_t id = kv.find_arr(k, GGUF_TYPE_INT32);
    if(id>=0){
        size_t n = gguf_get_arr_n(kv.g,id);
        const int32_t* d = (const int32_t*)gguf_get_arr_data(kv.g,id);
        out.assign(d, d+n);
    }
    return out;
}

ModelLoader::~ModelLoader(){
    // Free the weight buffer BEFORE the ctxs. CPU path: a from_ptr buffer that
    // does NOT own its memory (ctx_ does). Device path: OWNS the device buffer.
    // ggml_backend_buffer_free handles both.
    if(weights_buf_){
        // Purge any persistent graph-cache entries that reference these weights
        // BEFORE the buffer is freed, so a later model reallocating this address
        // cannot false-hit a stale cached graph (multi-model hosting safety).
        invalidate_graph_cache_for_weights(weights_buf_);
        ggml_backend_buffer_free(weights_buf_);
    }
    if(device_ctx_) ggml_free(device_ctx_);
    if(gguf_) gguf_free(gguf_); if(ctx_) ggml_free(ctx_);
}

bool ModelLoader::realize_weights(ggml_backend_t backend){
    if(weights_buf_) return true;                       // idempotent
    if(!backend || !ctx_){ VD_LOG("realize_weights: null backend/ctx"); return false; }

    // Backend-agnostic CPU check: ggml_backend_is_cpu() lives in the ggml-cpu
    // module (a dlopen'd MODULE under GGML_BACKEND_DL, not linkable here), so
    // query the device type through the generic backend API instead.
    ggml_backend_dev_t dev = ggml_backend_get_device(backend);
    const bool is_cpu = dev && ggml_backend_dev_type(dev) == GGML_BACKEND_DEVICE_TYPE_CPU;
    if (is_cpu) {
        // Fast path: borrow the host ctx memory directly (no copy). The GGUF is
        // loaded with no_alloc=false, so every tensor's data lives in one
        // contiguous ctx mem_buffer; wrap that exact memory as a CPU backend
        // buffer and point every tensor's ->buffer at it, so graphs can reference
        // the loader tensors DIRECTLY as leaves.
        void*  base = ggml_get_mem_buffer(ctx_);
        size_t size = ggml_get_mem_size(ctx_);
        weights_buf_ = ggml_backend_cpu_buffer_from_ptr(base, size);
        if(!weights_buf_){ VD_LOG("realize_weights: buffer_from_ptr failed"); return false; }
        for(auto& kv : tensors_) kv.second->buffer = weights_buf_;
        return true;
    }

    // Device path (CUDA/Metal/Vulkan/...): mirror every weight into a no_alloc
    // ctx, allocate THAT on the backend, upload each tensor's bytes from the host
    // source, and repoint the name->tensor map at the device tensors. ctx_ stays
    // alive as the host source.
    const size_t n = tensors_.size();
    struct ggml_init_params dp = {
        /*.mem_size  =*/ ggml_tensor_overhead() * (n + 8),
        /*.mem_buffer=*/ nullptr,
        /*.no_alloc  =*/ true,
    };
    device_ctx_ = ggml_init(dp);
    if(!device_ctx_){ VD_LOG("realize_weights: device ctx init failed"); return false; }

    std::vector<std::pair<ggml_tensor*, const void*>> ups; ups.reserve(n);
    std::unordered_map<std::string, ggml_tensor*> devmap; devmap.reserve(n);
    for (auto& kv : tensors_) {
        ggml_tensor* s = kv.second;
        ggml_tensor* d = ggml_new_tensor(device_ctx_, s->type, GGML_MAX_DIMS, s->ne);
        ggml_set_name(d, kv.first.c_str());
        devmap.emplace(kv.first, d);
        ups.emplace_back(d, s->data);
    }
    weights_buf_ = ggml_backend_alloc_ctx_tensors(device_ctx_, backend);
    if(!weights_buf_){ VD_LOG("realize_weights: alloc_ctx_tensors failed"); return false; }
    for (auto& pr : ups)
        ggml_backend_tensor_set(pr.first, pr.second, 0, ggml_nbytes(pr.first));
    tensors_.swap(devmap);
    return true;
}

void ModelLoader::reset_(){
    if(gguf_){ gguf_free(gguf_); gguf_ = nullptr; }
    if(ctx_){ ggml_free(ctx_); ctx_ = nullptr; }
    tensors_.clear();
    cfg_ = VoiceDetectConfig();
}

bool ModelLoader::fail_(const std::string& msg){
    error_ = msg;
    VD_LOG("%s", msg.c_str());
    reset_();
    return false;
}

bool ModelLoader::load(const std::string& path){
    if(gguf_ || ctx_) return fail_("model already loaded");
    error_.clear();
    struct gguf_init_params p{ /*no_alloc*/false, /*ctx*/&ctx_ };
    gguf_ = gguf_init_from_file(path.c_str(), p);
    if(!gguf_) return fail_("gguf open failed: " + path);
    return finish_(Kv{gguf_, ""});
}

bool ModelLoader::load_from_memory(const void* data, size_t size){
    return load_from_memory(data, size, std::string());
}

bool ModelLoader::load_from_memory(const void* data, size_t size, const std::string& prefix){
    if(gguf_ || ctx_) return fail_("model already loaded");
    error_.clear();
    if(!data || size == 0) return fail_("empty GGUF buffer");

    // Parse the header and tensor table only (no_alloc), validate every tensor
    // against the buffer size, then copy the selected tensors (all of them for
    // an empty prefix) into a context of our own. ggml's own no_alloc=false
    // reader is not used: it sizes its allocation from the tensor table before
    // checking it against the buffer, so a hostile table can make it abort.
    {
        std::string perr;
        if(!gguf_precheck(data, size, &perr)) return fail_("invalid GGUF buffer: " + perr);
    }
    ggml_context* meta = nullptr;
    struct gguf_init_params mp{ /*no_alloc*/true, /*ctx*/&meta };
    gguf_ = gguf_init_from_buffer(data, size, mp);
    if(!gguf_){ if(meta) ggml_free(meta); return fail_("invalid or truncated GGUF buffer"); }

    struct MetaGuard { ggml_context* c; ~MetaGuard(){ if(c) ggml_free(c); } } guard{meta};
    const size_t data_off = gguf_get_data_offset(gguf_);
    if(data_off > size) return fail_("GGUF buffer is truncated (tensor table exceeds buffer)");
    const size_t avail = size - data_off;
    const int64_t nt = gguf_get_n_tensors(gguf_);

    std::vector<int64_t> ids;
    size_t total = 0;
    for(int64_t i=0;i<nt;++i){
        const char* nm = gguf_get_tensor_name(gguf_, i);
        if(std::strncmp(nm, prefix.c_str(), prefix.size()) != 0) continue;
        const size_t off = gguf_get_tensor_offset(gguf_, i);
        const size_t nb  = gguf_get_tensor_size(gguf_, i);
        if(off > avail || nb > avail - off)
            return fail_(std::string("GGUF buffer is truncated or corrupt (tensor '") + nm + "' is out of range)");
        const size_t padded = GGML_PAD(nb, GGML_MEM_ALIGN);
        if(padded < nb || total > SIZE_MAX - padded) return fail_("GGUF tensor sizes overflow");
        total += padded;
        ids.push_back(i);
    }
    if(ids.empty()) return fail_("no tensors with prefix '" + prefix + "' in GGUF buffer");
    // Distinct tensors never share bytes in a valid file; a larger sum means a corrupt table.
    if(total > avail + ids.size()*GGML_MEM_ALIGN) return fail_("GGUF tensor table is corrupt (tensors overlap)");

    struct ggml_init_params ip{
        /*mem_size*/ total + (ids.size()+1) * ggml_tensor_overhead() + GGML_MEM_ALIGN,
        /*mem_buffer*/ nullptr, /*no_alloc*/ false };
    ctx_ = ggml_init(ip);
    if(!ctx_) return fail_("out of memory");
    const uint8_t* base = static_cast<const uint8_t*>(data) + data_off;
    for(int64_t i : ids){
        const char* nm = gguf_get_tensor_name(gguf_, i);
        ggml_tensor* src = ggml_get_tensor(meta, nm);
        if(!src) return fail_(std::string("GGUF tensor '") + nm + "' has no descriptor");
        ggml_tensor* t = ggml_new_tensor(ctx_, src->type, GGML_MAX_DIMS, src->ne);
        if(!t || !t->data) return fail_("out of memory");
        const size_t nb = ggml_nbytes(t);
        if(nb != gguf_get_tensor_size(gguf_, i)) return fail_(std::string("GGUF tensor '") + nm + "' has inconsistent size");
        ggml_set_name(t, nm + prefix.size());
        std::memcpy(t->data, base + gguf_get_tensor_offset(gguf_, i), nb);
    }
    return finish_(Kv{gguf_, prefix}, /*ctx_names=*/true);
}

bool ModelLoader::finish_(const Kv& kv, bool ctx_names){
    cfg_.arch          = kv_str(kv, "voicedetect.arch");
    cfg_.embedding_dim = kv_u32(kv, "voicedetect.embedding_dim");
    cfg_.l2_normalize  = kv_bool(kv, "voicedetect.l2_normalize", true);
    // Encoder identity (see VoiceDetectConfig::family). Keys come from the same
    // prefix as every other key, so a bundle component and the standalone file
    // it was made from give the same string.
    cfg_.name = kv_str_soft(kv, "general.name");
    if(kv_str_soft(kv, "general.architecture") == "voicedetect"){
        cfg_.family = "voicedetect:" + kv_str_soft(kv, "voicedetect.arch") + ":" + cfg_.name + ":" +
                      (cfg_.embedding_dim > 0 ? std::to_string(cfg_.embedding_dim) : std::string());
    } else {
        cfg_.family.clear();
    }
    // FBank front end
    cfg_.sample_rate   = kv_u32(kv, "voicedetect.fbank.sample_rate", 16000);
    cfg_.n_mels        = kv_u32(kv, "voicedetect.fbank.n_mels", 80);
    cfg_.n_fft         = kv_u32(kv, "voicedetect.fbank.n_fft", 512);
    cfg_.win_length    = kv_u32(kv, "voicedetect.fbank.win_length", 400);
    cfg_.hop_length    = kv_u32(kv, "voicedetect.fbank.hop_length", 160);
    cfg_.preemph       = kv_f32(kv, "voicedetect.fbank.preemph", 0.97f);
    cfg_.fbank_low_freq  = kv_f32(kv, "voicedetect.fbank.low_freq", 20.0f);
    cfg_.fbank_high_freq = kv_f32(kv, "voicedetect.fbank.high_freq", 0.0f);
    cfg_.fbank_use_energy = kv_bool(kv, "voicedetect.fbank.use_energy", false);
    cfg_.fbank_cmn        = kv_bool(kv, "voicedetect.fbank.cmn", true);
    cfg_.fbank_window     = kv_str(kv, "voicedetect.fbank.window", "povey");
    // WeSpeaker ResNet34 block manifest (only present for that arch).
    if(cfg_.arch == "wespeaker_resnet34"){
        VoiceDetectConfig::ResNetConfig& r = cfg_.resnet;
        r.stem_weight  = kv_str(kv, "voicedetect.resnet.stem_weight");
        r.stem_bias    = kv_str(kv, "voicedetect.resnet.stem_bias");
        r.conv1_weight = kv_str_arr(kv, "voicedetect.resnet.conv1_weight");
        r.conv1_bias   = kv_str_arr(kv, "voicedetect.resnet.conv1_bias");
        r.conv2_weight = kv_str_arr(kv, "voicedetect.resnet.conv2_weight");
        r.conv2_bias   = kv_str_arr(kv, "voicedetect.resnet.conv2_bias");
        r.down_weight  = kv_str_arr(kv, "voicedetect.resnet.down_weight");
        r.down_bias    = kv_str_arr(kv, "voicedetect.resnet.down_bias");
        r.stride       = kv_i32_arr(kv, "voicedetect.resnet.stride");
        r.seg_weight   = kv_str(kv, "voicedetect.resnet.seg_weight");
        r.seg_bias     = kv_str(kv, "voicedetect.resnet.seg_bias");
        r.mean_vec     = kv_str(kv, "voicedetect.resnet.mean_vec");
        r.var_eps      = kv_f32(kv, "voicedetect.resnet.var_eps", 1e-8f);
    }
    // 3D-Speaker ERes2Net manifest (only present for that arch).
    if(cfg_.arch == "eres2net"){
        VoiceDetectConfig::ERes2NetConfig& e = cfg_.eres2net;
        e.conv_weight = kv_str_arr(kv, "voicedetect.eres2net.conv_weight");
        e.conv_bias   = kv_str_arr(kv, "voicedetect.eres2net.conv_bias");
        e.conv_stride = kv_i32_arr(kv, "voicedetect.eres2net.conv_stride");
        e.num_blocks  = kv_i32_arr(kv, "voicedetect.eres2net.num_blocks");
        e.m_channels  = kv_u32(kv, "voicedetect.eres2net.m_channels");
        e.scale       = kv_u32(kv, "voicedetect.eres2net.scale", 2);
        e.seg_weight  = kv_str(kv, "voicedetect.eres2net.seg_weight");
        e.seg_bias    = kv_str(kv, "voicedetect.eres2net.seg_bias");
        e.relu_clamp  = kv_f32(kv, "voicedetect.eres2net.relu_clamp", 20.0f);
        e.var_eps     = kv_f32(kv, "voicedetect.eres2net.var_eps", 1e-8f);
    }
    // 3D-Speaker CAM++ manifest (only present for that arch).
    if(cfg_.arch == "campplus"){
        VoiceDetectConfig::CamPPlusConfig& cp = cfg_.campplus;
        cp.conv_weight = kv_str_arr(kv, "voicedetect.campplus.conv_weight");
        cp.conv_bias   = kv_str_arr(kv, "voicedetect.campplus.conv_bias");
        cp.bn_prefix   = kv_str_arr(kv, "voicedetect.campplus.bn_prefix");
        cp.emb_bn_mean = kv_str(kv, "voicedetect.campplus.emb_bn_mean");
        cp.emb_bn_var  = kv_str(kv, "voicedetect.campplus.emb_bn_var");
        cp.bn_eps      = kv_f32(kv, "voicedetect.campplus.bn_eps", 1e-5f);
    }

    // Analyze heads (phased; absent -> present=false, engine skips analyze).
    cfg_.analyze_present = kv_bool(kv, "voicedetect.analyze.present", false);
    if(cfg_.analyze_present){
        cfg_.emotion_labels = kv_str_arr(kv, "voicedetect.analyze.emotion_labels");
        cfg_.gender_labels  = kv_str_arr(kv, "voicedetect.analyze.gender_labels");
    }
    // wav2vec2 analyze config (only present for the analyze GGUF). All dims/kernels/
    // strides live in KV so the C++ analyze graph carries no magic numbers. Shared
    // by the emotion (base) and age/gender (large-robust) analyze archs.
    if(cfg_.arch == "wav2vec2_emotion" || cfg_.arch == "wav2vec2_age_gender"){
        VoiceDetectConfig::W2V2Config& w = cfg_.w2v2;
        w.hidden_size     = kv_u32(kv, "voicedetect.w2v2.hidden_size");
        w.n_layers        = kv_u32(kv, "voicedetect.w2v2.n_layers");
        w.n_heads         = kv_u32(kv, "voicedetect.w2v2.n_heads");
        w.ff_dim          = kv_u32(kv, "voicedetect.w2v2.ff_dim");
        w.num_conv_layers = kv_u32(kv, "voicedetect.w2v2.num_conv_layers");
        w.conv_dims       = kv_i32_arr(kv, "voicedetect.w2v2.conv_dims");
        w.conv_kernels    = kv_i32_arr(kv, "voicedetect.w2v2.conv_kernels");
        w.conv_strides    = kv_i32_arr(kv, "voicedetect.w2v2.conv_strides");
        w.feat_extract_norm       = kv_str(kv, "voicedetect.w2v2.feat_extract_norm");
        w.conv_bias               = kv_bool(kv, "voicedetect.w2v2.conv_bias", false);
        w.do_normalize            = kv_bool(kv, "voicedetect.w2v2.do_normalize", false);
        w.feat_extract_activation = kv_str(kv, "voicedetect.w2v2.feat_extract_activation");
        w.hidden_act              = kv_str(kv, "voicedetect.w2v2.hidden_act");
        w.do_stable_layer_norm    = kv_bool(kv, "voicedetect.w2v2.do_stable_layer_norm", false);
        w.layer_norm_eps          = kv_f32(kv, "voicedetect.w2v2.layer_norm_eps", 1e-5f);
        w.num_conv_pos_embeddings = kv_u32(kv, "voicedetect.w2v2.num_conv_pos_embeddings");
        w.num_conv_pos_embedding_groups =
            kv_u32(kv, "voicedetect.w2v2.num_conv_pos_embedding_groups");
        w.pos_conv_weight_norm_dim = kv_u32(kv, "voicedetect.w2v2.pos_conv_weight_norm_dim", 2);
        w.use_weighted_layer_sum   = kv_bool(kv, "voicedetect.w2v2.use_weighted_layer_sum", false);
        w.classifier_proj_size     = kv_u32(kv, "voicedetect.w2v2.classifier_proj_size");
    }
    if(!kv.err.empty()) return fail_(kv.err);
    // tensors (verbatim names). A prefixed load already stripped the prefix from
    // the names in ctx_, so it walks ctx_ instead of the (prefixed) GGUF table.
    if(ctx_names){
        for(ggml_tensor* t = ggml_get_first_tensor(ctx_); t; t = ggml_get_next_tensor(ctx_, t))
            tensors_[t->name] = t;
    } else {
        const int64_t nt = gguf_get_n_tensors(gguf_);
        for(int64_t i=0;i<nt;++i){ const char* nm = gguf_get_tensor_name(gguf_,i);
            ggml_tensor* t = ggml_get_tensor(ctx_, nm); if(t) tensors_[nm]=t; }
    }
    // The analyze GGUF carries no embedding head (no voicedetect.embedding_dim);
    // it is valid iff the analyze schema is present.
    if(!(cfg_.embedding_dim>0 || cfg_.analyze_present))
        return fail_("GGUF is not a voicedetect model (no voicedetect.embedding_dim)");
    // The standalone GGUF context is only needed for the keys read above.
    gguf_free(gguf_); gguf_ = nullptr;
    return true;
}

ggml_tensor* ModelLoader::tensor(const std::string& n) const {
    auto it = tensors_.find(n); return it==tensors_.end()? nullptr : it->second;
}

} // namespace vd
