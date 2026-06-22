# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Build Commands

```bash
cmake -B build -DBUILD_TESTS=ON
cmake --build build
ctest --test-dir build --output-on-failure
```

No platform-specific CMake files exist here, and none should be added — see "Platform
Independence" below.

## Architecture Overview

`campello_llm` is scoped specifically to **weight-only LLM formats**: safetensors and gguf — files
with no embedded graph topology, where the architecture (how tensors wire into a decoder block)
must be supplied by code that already knows it. Models distributed as ONNX/TFLite carry their own
topology and skip this project entirely, going through `campello_nn`'s own importer
(`importOnnxFromFile`/`importTfliteFromFile`) instead. This is the layer that justifies
`campello_llm` existing as a separate project from `campello_nn`: the architecture knowledge a
graph-format file would have carried for free.

### Platform Independence

`campello_llm` only ever calls `campello_nn`'s public `GraphBuilder`/`Context` API. All
platform-specific backend selection (CPU/MPSGraph/DirectML/...) lives inside `campello_nn`'s
`Context::create()`, so the same source here builds and runs identically on every platform
`campello_nn` supports. **Do not add per-platform `.cmake` files to this repo** — if a platform
needs different behavior, that belongs in `campello_nn`, not here.

### Dependency on campello_nn

Fetched via `FetchContent` in the top-level `CMakeLists.txt`, pinned to a commit SHA (no tags exist
yet upstream — move to a real tag once `campello_nn` cuts one). `campello_nn`'s own
`CMakeLists.txt` declares an `option(BUILD_TESTS ...)` with the same name as ours; the top-level
`CMakeLists.txt` forces it off around the `FetchContent_MakeAvailable(campello_nn)` call so a
`-DBUILD_TESTS=ON` meant for `campello_llm`'s own tests doesn't also build (and run, via `ctest`)
`campello_nn`'s universal test suite — that suite's fixture paths resolve against
`CMAKE_SOURCE_DIR`, which becomes *this* project's source tree once it's pulled in as a
subdirectory, so its tests fail for reasons that have nothing to do with `campello_llm`. Don't
remove that guard without re-checking that case.

### Weights-File Reading (Phase 1)

`include/campello_llm/weights_file.hpp` defines the shared, deliberately small abstraction: a
`WeightDType` enum (the union of safetensors/gguf element types — wider than
`campello_nn::DataType`, since mapping/rejecting unsupported dtypes is the Phase 3 architecture
registry's job, not this layer's) plus an abstract `WeightsFile` exposing only `tensors()`/
`find(name)` — `{name -> dtype/shape/bytes}`, nothing format-specific. `SafetensorsFile`
(`safetensors_reader.hpp`) and `GgufFile` (`gguf_reader.hpp`) both implement it; each keeps its own
format-specific extras off the shared interface (safetensors' `__metadata__` string map via
`SafetensorsFile::metadata()`; gguf's key/value metadata — architecture name, hyperparameters, and
for self-contained models, the embedded `tokenizer.ggml.*` vocab/merges — via
`GgufFile::metadata()`/`findMetadata()`, returned as a generic `GgufValue`, not a separate
tokenizer-specific API).

Both readers follow the `loadXFromMemory`/`loadXFromFile` split (`loadXFromFile` just reads the
file into a buffer and calls `loadXFromMemory`, same as `campello_nn`'s ONNX importer). Each
returned `WeightsFile` owns one buffer holding every tensor's bytes; `TensorInfo::data` pointers
are zero-copy views into that single buffer (no per-tensor copies) — `loadXFromMemory` copies its
input into that buffer rather than aliasing the caller's pointer, so the caller's buffer (e.g. an
Android `AAssetManager` read) can be freed immediately after the call returns.

- **safetensors**: 8-byte little-endian header length, then a JSON header (tensor name ->
  `{dtype, shape, data_offsets}`, plus an optional `__metadata__` string map), then raw tensor
  bytes. The JSON header is parsed by a small hand-rolled recursive-descent parser
  (`src/safetensors/json_value.{hpp,cpp}`, internal) rather than a JSON dependency — same
  "hand-roll the minimal reader for a known, stable format" call `proto_reader.hpp` made for ONNX's
  protobuf in `campello_nn`. Dtype strings verified against the real `safetensors` Rust crate's
  `Dtype` enum (fetched from its GitHub source) before trusting them; sub-byte (F4/F6_*), exotic
  float8 (F8_*), and complex (C64) types have no campello_llm representation yet and throw rather
  than guess a conversion.
- **gguf**: magic + version, a key/value metadata block, a tensor info table, then tensor data
  (offsets relative to an aligned data-section start — default alignment 32, overridable by a
  `general.alignment` metadata key). All field layouts/enum values (`GGUFValueType`,
  `GGMLQuantizationType`, magic number, default alignment) were verified against the real `gguf`
  Python package's `gguf/constants.py` and `gguf/gguf_reader.py`, not assumed from memory. Only
  non-block-quantized tensor types (F32/F16/BF16/F64/I8/I16/I32/I64) are supported — block-quantized
  formats (Q4_0, Q8_0, the `_K`/`IQ*` variants, ...) throw rather than guess a dequantization
  scheme; add one only once a real quantized model needs it. **Gotcha confirmed against a real
  generated file, not assumed:** gguf's on-disk tensor dims are the *reverse* of a row-major numpy
  shape (ggml lists the fastest-varying dimension first) — this reader stores dims exactly as read
  from the file, with no un-reversing, since shape-order convention is the architecture registry's
  concern (Phase 3), not this reader-level type. safetensors does **not** reverse — its JSON
  `shape` matches the numpy shape exactly. Don't assume the two formats agree on this.

Test fixtures (`tests/fixtures/test_tensors.{safetensors,gguf}`) are synthetic, generated by the
real `safetensors`/`gguf` Python packages (`tests/fixtures/generate_test_safetensors.py`/
`generate_test_gguf.py`, regenerate with `python3 <script>.py`) — every expected byte/shape value
hardcoded in `tests/test_safetensors_reader.cpp`/`test_gguf_reader.cpp` was independently confirmed
against the actual generated file (raw hex dump for safetensors, `python3 -m
gguf.scripts.gguf_dump` for gguf) before being written into the test, not assumed.

### First Target Model (decision recorded here, not yet implemented)

**TinyLlama-1.1B, safetensors, first.** LLaMA-style architecture (RMSNorm, RoPE, SwiGLU),
HuggingFace `tokenizer.json` BPE tokenizer — `campello_nn` already has `rmsNorm`/`rotaryEmbedding`
wired (composed from existing primitives, see `campello_nn`'s `CLAUDE.md`/`TODO.md`) for exactly
this shape. A small GGUF-quantized build of the same architecture is the second target, once the
architecture-registry/tokenizer plumbing built for the safetensors path already works — gguf's
embedded vocab/merges should reuse the same BPE encode/decode logic rather than a second
implementation. GPT-style architecture (`layerNorm`, `gelu`, no RoPE) comes after both, specifically
to validate the architecture registry generalizes instead of being accidentally LLaMA-shaped (see
`TODO.md` Phase 3).

### Conventions Inherited from campello_nn

- C++20, `systems::leal::campello_llm` namespace (matches `campello_nn`'s
  `systems::leal::campello_nn` and the sibling `campello_gpu`/`campello_image` convention).
- Public headers under `include/campello_llm/`; implementation under `src/`. Whether public types
  follow `campello_nn`'s handle-based (`void*` pimpl) pattern is still open — decide once `Model`'s
  actual shape is known (Phase 3/4).
- `loadFromMemory`/`loadFromFile` split for anything that reads a file (weights, tokenizer config),
  matching `campello_nn`'s `importOnnxFromMemory`/`importOnnxFromFile` convention — needed for
  Android's `AAssetManager`, which has no real filesystem path.
- Any binary format field (gguf key-value entries, safetensors header fields, tokenizer.json
  structure) gets verified against a real generated file before being trusted, never assumed from
  memory — the same discipline `campello_nn`'s ONNX/TFLite importers followed for protobuf/
  FlatBuffers field numbers.
- `STATIC` library (matches `campello_nn`'s own `add_library(${PROJECT_NAME} STATIC ...)`).
- Version number has exactly one source of truth: `project(campello_llm VERSION x.y.z)` in the
  top-level `CMakeLists.txt`. `configure_file(src/campello_llm_config.h.in campello_llm_config.h)`
  substitutes `campello_llm_VERSION_MAJOR/MINOR/PATCH` (set automatically by `project()`) into a
  generated header, included only by `src/version.cpp` — same pattern as `campello_gpu`'s
  `campello_gpu_config.h.in`. Never hardcode a version number anywhere else.

## Current State

Phase 0 (scaffolding) and Phase 1 (weights-file parsing — safetensors + gguf readers, see above)
are done. No tokenizer, architecture wiring, or generation logic exists yet — see `TODO.md` for the
full phased plan, starting at Phase 2 (tokenizer).
