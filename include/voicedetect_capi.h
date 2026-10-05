#ifndef VOICEDETECT_CAPI_H
#define VOICEDETECT_CAPI_H

#include <stddef.h>

#ifdef __cplusplus
extern "C" {
#endif

// Flat C-API for voice-detect.cpp - designed for dlopen / cgo / purego (LocalAI).
//
// All functions are extern "C" and never let a C++ exception cross the
// boundary. A speaker-recognition model is loaded ONCE into an opaque
// `voicedetect_ctx` and reused across embed/verify/analyze calls. Returned
// strings are malloc'd UTF-8 owned by the caller and must be released with
// voicedetect_capi_free_string; returned float vectors with
// voicedetect_capi_free_vec.
//
// Pipeline: decode/resample audio -> 16 kHz mono -> 80-dim Kaldi-compatible
// FBank features -> speaker encoder -> L2-normalized embedding. `verify`
// compares two clips by cosine distance against a threshold; `analyze` runs the
// (phased) age/gender/emotion heads.

// Opaque speaker-recognition context (wraps a loaded model + last-error buffer).
typedef struct voicedetect_ctx voicedetect_ctx;

// ABI version of this header/implementation. Bump on any breaking change to the
// function signatures or semantics below. Additive changes (new functions, such
// as the load_from_memory family) are fine without a bump; LocalAI checks this integer for compatibility.
//
// v1: initial flat C-API surface - load/free/last_error/free_string,
//     embed_path / embed_pcm (+ free_vec), verify_paths, analyze_path_json.
int voicedetect_capi_abi_version(void);

// Load a GGUF speaker-recognition model. Returns an owning context, or NULL on
// failure (bad/missing GGUF). The returned context must be released with
// voicedetect_capi_free.
voicedetect_ctx* voicedetect_capi_load(const char* gguf_path);

// Load a model from a GGUF held in memory (no file, no temporary file, no file
// descriptor; works the same on every platform). `data` points to the complete
// GGUF file of `size` bytes.
//
// OWNERSHIP: the loader copies the tensor data into its own memory during the
// call. `data` is only read inside the call: the caller may free or overwrite it
// as soon as the function returns, on success and on failure. While loading, the
// buffer and the model are in memory together (peak = buffer + model).
//
// A NULL or empty buffer, a truncated buffer or a corrupt buffer gives NULL and
// a message in voicedetect_capi_last_load_error(); no byte outside
// [data, data+size) is read. Returns an owning context, released with
// voicedetect_capi_free. Calls on different threads are independent.
voicedetect_ctx* voicedetect_capi_load_from_memory(const void* data, size_t size);

// As voicedetect_capi_load_from_memory, for a model that is one component of a
// larger GGUF (a bundle): every metadata key and every tensor name of the model
// is stored as `prefix` + name (for example "voice."). Pass the whole bundle and
// the prefix: no standalone copy of the component is needed, and only the
// tensors under the prefix are copied. Names stored as string values inside the
// model's metadata (tensor manifests) are not prefixed. Same ownership and error
// rules as above. A prefix that matches no tensor is an error. A NULL or empty
// `prefix` behaves like voicedetect_capi_load_from_memory.
voicedetect_ctx* voicedetect_capi_load_from_memory_prefixed(const void* data, size_t size,
                                                            const char* prefix);

// Reason for the last failed voicedetect_capi_load* call made on the CALLING
// THREAD, or "" if the last such call succeeded. The pointer stays valid until
// the next load call on the same thread. Never NULL.
const char* voicedetect_capi_last_load_error(void);

// Free a context obtained from voicedetect_capi_load. Safe on NULL.
void voicedetect_capi_free(voicedetect_ctx* ctx);

// Human-readable description of the last error on `ctx`, or "" if none. The
// returned pointer is owned by the context and valid until the next call on it
// (or until voicedetect_capi_free). Returns "" if `ctx` is NULL.
const char* voicedetect_capi_last_error(voicedetect_ctx* ctx);

// Free a string previously returned by voicedetect_capi_analyze_path_json (or
// any other char*-returning entry point). Safe on NULL.
void voicedetect_capi_free_string(char* s);

// Free a float vector previously returned via voicedetect_capi_embed_* out_vec.
// Safe on NULL.
void voicedetect_capi_free_vec(float* v);

// Compute the L2-normalized speaker embedding for a WAV file. On success returns
// 0 and sets `*out_vec` to a malloc'd array of `*out_dim` floats (release with
// voicedetect_capi_free_vec). On error returns nonzero, leaves `*out_vec` NULL /
// `*out_dim` 0, and sets the context's last error (see
// voicedetect_capi_last_error).
int voicedetect_capi_embed_path(voicedetect_ctx* ctx, const char* wav_path,
                                float** out_vec, int* out_dim);

// Like voicedetect_capi_embed_path but from in-memory mono float PCM
// (`pcm`, length `n_samples`). If `sample_rate != 16000` the audio is linearly
// resampled to 16 kHz first. Same ownership/error contract as embed_path.
int voicedetect_capi_embed_pcm(voicedetect_ctx* ctx, const float* pcm,
                               int n_samples, int sample_rate,
                               float** out_vec, int* out_dim);

// Verify whether two clips are the same speaker. Embeds `a` and `b`, computes
// the cosine distance (1 - cosine_similarity) between their L2-normalized
// embeddings, writes it to `*out_distance`, and sets `*out_verified` to 1 iff
// the distance is <= `threshold` (same speaker), else 0. On success returns 0;
// on error returns nonzero, leaves the out params unchanged, and sets the
// context's last error.
int voicedetect_capi_verify_paths(voicedetect_ctx* ctx, const char* a,
                                  const char* b, float threshold,
                                  float* out_distance, int* out_verified);

// Analyze a WAV file with the (phased) age/gender/emotion heads, returning a
// malloc'd UTF-8 JSON document (free with voicedetect_capi_free_string). The
// shape is:
//
//   {"age":42.0,
//    "gender":{"label":"female","female":0.88,"male":0.12},
//    "emotion":{"label":"neutral","scores":{"neutral":0.7, ...}}}
//
// On error returns NULL and sets the context's last error.
char* voicedetect_capi_analyze_path_json(voicedetect_ctx* ctx,
                                         const char* wav_path);

// Size of the L2-normalized embedding this model produces (192 to 512), 0 for a
// model with no speaker embedding (the age/gender/emotion analyze heads), -1 for
// a NULL ctx. Additive: no ABI version bump.
int voicedetect_capi_embedding_dim(voicedetect_ctx* ctx);

// Identity of the loaded encoder, read from the GGUF header the loader already
// parsed. For a model loaded with a prefix the keys are the prefixed ones, so a
// component of a bundle reports the same values as the standalone file it was
// made from. All three work for every load path (path, memory, prefixed memory).
// Each returns NULL for a NULL ctx. The returned pointer is owned by the context,
// never changes, and stays valid until voicedetect_capi_free; do not free it.
// Reading it from several threads at once is safe. Additive: no ABI version bump.
//
// voicedetect_capi_encoder_arch: the `voicedetect.arch` value, for example
//   "ecapa_tdnn", "campplus", "wespeaker_resnet34", "eres2net". "" if the key
//   is absent.
const char* voicedetect_capi_encoder_arch(const voicedetect_ctx* ctx);

// voicedetect_capi_encoder_name: the `general.name` value (the checkpoint the
//   converter was run on, for example "speechbrain/spkrec-ecapa-voxceleb"). ""
//   if the key is absent.
const char* voicedetect_capi_encoder_name(const voicedetect_ctx* ctx);

// voicedetect_capi_encoder_family: a string that names the embedding space, for
//   a registry of voices to record next to its embeddings and compare before it
//   matches a new embedding. Equal embedding sizes do not mean the same space
//   (ECAPA-TDNN and CAM++ both give 192 values). The format is exactly
//
//     voicedetect:<voicedetect.arch>:<general.name>:<voicedetect.embedding_dim>
//
//   for example "voicedetect:ecapa_tdnn:speechbrain/spkrec-ecapa-voxceleb:192".
//   A missing key gives an empty field (the colons stay), and no escaping is
//   done: the values are copied as they are. The embedding_dim is written in
//   decimal, and is an empty field when it is 0 (an analyze model). If
//   `general.architecture` is not "voicedetect" the string is "" (not a
//   voicedetect model). Another quantization of the same encoder has the same
//   family. parakeet.cpp defines its speaker-registry encoder fingerprint with
//   the same formula (speaker_encoder_family); the two MUST stay identical, so
//   change neither without the other.
const char* voicedetect_capi_encoder_family(const voicedetect_ctx* ctx);

#ifdef __cplusplus
} // extern "C"
#endif

#endif // VOICEDETECT_CAPI_H
