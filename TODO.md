# campello_llm — TODO

Roadmap derived from `campello_nn`'s `NN_ARCHITECTURE.md` §4 (the `campello_llm` design) and §5
(model import/graph caching, which this project depends on but does not own). `campello_llm` is
scoped specifically to **weight-only LLM formats** (safetensors, gguf — files with no embedded
graph topology, where the architecture must be supplied by code that already knows it). Models
distributed as ONNX/TFLite skip this project entirely and go through `campello_nn`'s own importer
instead.

`campello_llm` is **platform-independent**: all platform-specific backend selection (CPU/MPSGraph/
DirectML/...) lives inside `campello_nn`'s `Context::create()`. This project only ever calls
`campello_nn`'s public `GraphBuilder`/`Context` API, so the same source builds and runs identically
on every platform `campello_nn` supports — no per-platform `.cmake` files needed here.

Phases are ordered so each is buildable and testable before the next depends on it: scaffolding →
read weight files → tokenize → wire a graph for a known architecture → tie it together with
KV-cache → generate tokens → polish/document.

---

## Phase 0 — Project Scaffolding

- [x] `CMakeLists.txt` (single, no per-platform files) — `FetchContent` for `campello_nn`
      (`https://github.com/rusoleal/campello_nn.git`, pinned commit — no tags exist upstream yet)
      and GoogleTest, mirroring `campello_nn`'s own FetchContent conventions
- [x] Directory layout: `include/campello_llm/`, `src/`, `tests/` — mirror `campello_nn`'s
      `inc`/`src`/`tests` split. Handle-based public-API convention deferred to Phase 3 once
      `Model`'s actual shape is known
- [x] C++20, GoogleTest + `ctest` wired (`tests/CMakeLists.txt`)
- [x] CI workflow (`.github/workflows/ci.yml`) building + testing on macOS/Linux/Windows
- [x] Root `CLAUDE.md`: architecture summary (weight-only-format scope, why this layer exists
      separately from `campello_nn`, the platform-independence note above), build commands,
      conventions — same role as `campello_nn`'s `CLAUDE.md`
- [x] Decide minimum viable first real model to target end-to-end (see "Open Questions") — **decided:
      TinyLlama-1.1B safetensors first, then a small GGUF-quantized model second**

---

## Phase 1 — Weights-File Parsing

Goal: read safetensors/gguf files into an architecture-agnostic `{tensor name → dtype/shape/bytes}`
view, with zero knowledge yet of what architecture the tensors belong to.

- [ ] safetensors reader: parse the JSON header (tensor name → dtype/shape/byte-offset), expose
      raw tensor bytes by name without copying the whole file into memory if avoidable
- [ ] gguf reader: parse the key-value metadata block (architecture name, hyperparameters), the
      tensor info table, and the embedded tokenizer vocab/merges — verify every field read against
      a real `.gguf` file (same "verify against a real generated file before trusting it" standard
      `campello_nn`'s ONNX/TFLite importers used for protobuf/FlatBuffers field numbers)
- [ ] Shared `WeightsFile`-style abstraction so Phase 3's architecture registry doesn't care which
      format a given model came from (don't over-build this — only abstract what both formats
      actually share: name→bytes lookup; gguf's extra metadata/tokenizer stays format-specific)
- [ ] Memory/File split for both readers (`loadFromMemory`/`loadFromFile`), matching
      `campello_nn`'s `importOnnxFromMemory`/`importOnnxFromFile` convention, for the same Android
      `AAssetManager` reason
- [ ] Test fixtures: a small synthetic safetensors file (generate via a Python script using the
      real `safetensors` package, same pattern as `campello_nn`'s
      `generate_conv_add_relu_onnx.py`) and a small real or synthetic `.gguf` file
- [ ] Tests: read back every tensor's name/shape/dtype/bytes from both fixtures and confirm they
      match what was written

---

## Phase 2 — Tokenizer

Goal: text ↔ token ids, for whichever tokenizer format the chosen first model (see Open Questions)
actually ships.

- [ ] Tokenizer support for one format first (likely HuggingFace `tokenizer.json` BPE, since it's
      the most common safetensors-adjacent format; gguf models carry vocab/merges embedded
      already, read by Phase 1's gguf reader, so the same BPE encode/decode logic should be
      reusable against either source)
- [ ] Encode: text → token ids (BPE merge algorithm)
- [ ] Decode: token ids → text
- [ ] Special tokens (BOS/EOS/PAD/UNK) and basic chat template handling (enough to format a
      single-turn prompt correctly for the chosen first model)
- [ ] Tests: encode/decode round-trip, and encode output checked against a known-good reference
      tokenization for a handful of real strings (generated once via Python's `transformers`
      or `tokenizers` library and checked into a fixture file)

---

## Phase 3 — Architecture Registry / Graph Wiring

Goal: turn "a `WeightsFile` plus a known architecture name" into a compiled `campello_nn::Graph`,
using `GraphBuilder` calls with weights bound via `constant()`. This is the layer that justifies
`campello_llm` existing separately from `campello_nn` — the architecture knowledge a graph-format
file would have carried for free.

- [ ] `GenerationConfig` struct (maxTokens, temperature, topP, topK)
- [ ] LLaMA-style architecture wiring: embedding (`gather`) → N decoder layers (`rmsNorm` →
      `rotaryEmbedding`-applied attention with causal masking via a precomputed additive mask
      constant + `softmax` → `rmsNorm` → SwiGLU MLP composed from `sigmoid`+`mul` for SiLU) →
      final `rmsNorm` → output projection (`matmul`/`gemm`). All the ops this needs
      (`rmsNorm`/`rotaryEmbedding`) already exist in `campello_nn` — see its `TODO.md` Phase 5
      "Op-set prep" section for exactly how each is implemented and on which backends
- [ ] Decide: one graph covering both prefill (full-sequence) and decode (single-token + KV-cache
      read/write), or two separately compiled graphs with different input shapes — see Open
      Questions
- [ ] GPT-style architecture wiring as a second case (`layerNorm` instead of `rmsNorm`, `gelu`
      instead of SwiGLU, no RoPE) — deliberately doing a second, structurally different
      architecture early validates the registry abstraction generalizes instead of accidentally
      being LLaMA-shaped
- [ ] Architecture registry: dispatch by the architecture name read from `config.json` (safetensors
      case) or the `general.architecture` gguf metadata key, to the matching wiring function above
- [ ] Tests: build a tiny synthetic model (small hidden size, 1-2 layers, hand-picked weights) for
      each architecture and check graph output against an independently hand/numpy-computed
      reference — same pattern as `campello_nn`'s `test_transformer_block.cpp`

---

## Phase 4 — `Model::load()` and KV-Cache

Goal: tie weights + tokenizer + architecture wiring + a `campello_nn::Context` together into the
loadable `Model` from the architecture doc, plus the stateful bookkeeping a WebNN-shaped graph
doesn't model on its own.

- [ ] `Model::load(context, path)` (and a `loadFromMemory`-style variant for the Android-asset
      case) — detects format (safetensors vs gguf) and architecture, builds or loads the graph
- [ ] Support loading a pre-cached graph via `campello_nn`'s graph caching
      (`loadGraphFromFile`/`loadGraphFromMemory`) instead of rebuilding one live every time —
      `Model::load()` should check for a cached sibling file first and fall back to building fresh
- [ ] KV-cache: explicit `Tensor`s fed back as inputs each decode step and read back out; decide
      pre-allocate-to-`maxTokens` vs. dynamic growth (see Open Questions)
- [ ] Tests: decoding token-by-token through the KV-cache produces the same logits as a single
      one-shot prefill over the equivalent full sequence (the standard correctness check for any
      KV-cache implementation)

---

## Phase 5 — Generation Loop

Goal: the actual `generate()` from the architecture doc.

- [ ] Prefill: single graph dispatch over the full prompt
- [ ] Decode loop: one graph dispatch per token, feeding KV-cache deltas from Phase 4
- [ ] Sampling on CPU after reading back logits: temperature scaling, top-k, top-p; greedy/argmax
      as a baseline deterministic mode (needed for the determinism test below)
- [ ] Streaming: `generate()`'s `onToken` callback invoked per generated token
- [ ] Stop conditions: max tokens, EOS token, stop sequences
- [ ] Tests: deterministic generation test (temperature=0/greedy) against a known-good reference
      output for the chosen small first model — the first real end-to-end test of the whole stack

---

## Phase 6 — Testing, Benchmarking, Examples

- [ ] Cross-backend conformance: same model, same prompt, run via `campello_nn`'s available
      backends on a given platform, assert outputs agree within per-dtype tolerance (mirrors
      `campello_nn`'s own Phase 6 plan)
- [ ] Performance benchmarks: prefill throughput, decode tokens/sec, per backend
- [ ] Example: minimal CLI chat/completion demo using a small real open-weights model
- [ ] Example: graph-caching demo (load a model once, cache its compiled graph, reload the cache
      on a second run) to demonstrate the startup-cost savings `campello_nn`'s graph caching exists
      for
- [ ] Public headers finalized under `include/campello_llm/`

---

## Phase 7 — Documentation & Packaging

- [ ] API reference docs (`Model`, `GenerationConfig`)
- [ ] Build/integration instructions (depends on `campello_nn`'s build for the target platform)
- [ ] CONTRIBUTING notes on adding a new architecture to the registry (what wiring code must
      supply, how to add a conformance/determinism test for it)
- [ ] Versioning/compatibility notes for any cached-graph files this project produces (inherits
      `campello_nn`'s own graph-cache versioning, see that project's `TODO.md` Phase 4c)

---

## Open Questions

- [x] **Minimum viable first real model** — **decided:** TinyLlama-1.1B (safetensors) first, then a
      small GGUF-quantized build of the same architecture second, once the safetensors-path
      architecture registry/tokenizer plumbing already works. GPT-style architecture (Phase 3) comes
      after both, specifically to validate the registry generalizes beyond LLaMA-shaped models.
- [x] **Tokenizer format priority** — **decided:** HuggingFace `tokenizer.json` BPE first (matches
      TinyLlama's safetensors release); gguf's embedded vocab/merges should reuse the same BPE
      encode/decode logic rather than a second implementation.
- [ ] **One shared prefill/decode graph vs. two separately compiled graphs** — a single graph with
      dynamic sequence length is more flexible but may be harder to express cleanly with
      `campello_nn`'s current shape-inference-at-build-time model; two fixed-shape graphs (one for
      the N-token prompt, one for 1-token decode steps) is more mechanical but means rebuilding/
      recompiling if `maxTokens` assumptions change. Decide in Phase 3.
- [ ] **KV-cache growth strategy** — pre-allocate `Tensor`s sized to `GenerationConfig::maxTokens`
      (simple, wastes memory for short generations) vs. dynamic growth (more complex, needs a
      reallocate-and-copy or chunked strategy since `campello_nn` `Tensor`s aren't resizable in
      place). Decide in Phase 4.
- [ ] **DirectML `RmsNorm` gap** — `campello_nn`'s DirectML backend doesn't implement `RmsNorm` yet
      (throws clearly rather than guessing; see that project's `TODO.md` Phase 5 "Op-set prep"
      note on `DML_MEAN_VARIANCE_NORMALIZATION1_OPERATOR_DESC`). LLaMA-style wiring (Phase 3) won't
      run on Windows/DirectML until that's filled in on the `campello_nn` side — track as a
      cross-repo dependency, not something to work around here.
- [ ] Library type (`STATIC` vs `SHARED`) — follow whatever `campello_nn` settles on for
      consistency, revisit if Windows symbol-export conventions matter.
