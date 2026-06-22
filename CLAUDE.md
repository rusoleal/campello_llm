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

Phase 0 (scaffolding) only: a `campello_llm` static library with a placeholder `version()` and one
smoke test (`tests/test_smoke.cpp`) proving the `campello_nn` dependency actually links and runs
(`Context::create({DeviceType::Cpu})`). No weights-file parsing, tokenizer, architecture wiring, or
generation logic exists yet — see `TODO.md` for the full phased plan, starting at Phase 1
(safetensors/gguf reading).
