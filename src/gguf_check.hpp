#pragma once
#include <cstddef>
#include <string>

namespace vd {

// Cheap structural check of a GGUF held in memory, run BEFORE the buffer is
// handed to ggml's reader. ggml validates sizes and bounds, but it still aborts
// the whole process (GGML_ASSERT) on a few well-formed-looking inputs, such as a
// metadata key with an empty name. A model buffer comes from the caller, so such
// input must become an error instead.
//
// Walks the header, the metadata key/value section and the tensor table with
// bounds checks on every read. Returns false and sets `err` on: a bad magic or
// version, an empty or oversized key, an unknown value type, a nested array, a
// tensor with an empty name, a bad dimension count or an unknown type id, or
// anything that runs past the end of the buffer. Tensor offsets and sizes are
// not checked here: the loaders check them against the buffer size.
bool gguf_precheck(const void* data, size_t size, std::string* err);

}  // namespace vd
