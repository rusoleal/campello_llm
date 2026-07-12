# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [0.3.0] - 2026-07-12

### Added
- Real Jinja2-like chat template engine (`formatChatPrompt()`/`defaultChatTemplate()`,
  `include/campello_llm/chat_template.hpp` + `src/chat_template/`): for-loops (`{% for message in
  messages %}` with `loop.first`/`loop.last`/`loop.index`), `if`/`elif`/`else`, string/member-access
  expressions, `==`/`!=`/`and`/`or`/`not`/`+` — replaces the earlier single-turn-only
  `formatSingleTurnChatPrompt()`, so multi-turn conversations render each model's own
  `tokenizer_config.json` chat template instead of one hardcoded literal shape.
- Full GGML block-quantized type coverage: `WeightDType` and dequantization now cover `Q4_1`,
  `Q5_0`, `Q5_1`, `Q8_1`, `Q2_K`, `Q3_K`, `Q4_K`, `Q5_K`, `Q6_K`, `Q8_K` on top of the existing
  `Q4_0`/`Q8_0` — real GGUF exports (e.g. Llama 3.1's `Q4_K_M`) use the K-quant formats, not just
  the legacy ones.
- `loadTokenizerFromGgufFile()` now handles `tokenizer.ggml.model = "gpt2"` (byte-level BPE), not
  just the original `"llama"` (SentencePiece) case — needed for Llama 3.1+, which ships a
  tiktoken-derived tokenizer rather than the older SentencePiece one; a GGUF reporting `"gpt2"`
  previously threw `"only tokenizer.ggml.model='llama' is supported"` unconditionally.
- `Model` public accessors: `maxSequenceLength()`, `vocabSize()`, `numLayers()`,
  `architectureName()`.
- `buildLlamaDecodeGraph()` now builds its GQA attention block with `campello_nn`'s new
  `GraphBuilder::gqaMatMul()` when running on `DeviceType::Cpu`/`GpuGeneric` (checked via the new
  `Context::deviceType()`), instead of unconditionally physically replicating K/V up to the full
  query head count via `slice()`+`concat()`. Found and fixed while chasing `GpuGeneric`'s memory
  usage on real `llama3.1_8b` inference far above a comparable llama.cpp/Ollama process for the
  same GGUF file — see `campello_nn` v0.5.0's changelog for the measured before/after. MPSGraph/
  DirectML are unaffected (that op isn't implemented for them, so they keep the old path).

### Changed
- **KV-cache/decode-graph capacity now starts small and grows on demand**, rather than always
  being pre-allocated at `maxSequenceLength`. `Model::generate()` (no longer `const`) starts at a
  256-token capacity and doubles it (rebuilding — or loading from its own on-disk cache — the
  compiled decode graph at each new tier) only as an actual conversation needs more room, up to
  `maxSequenceLength`. This supersedes the "decided in Phase 4: pre-allocate, sized to
  maxSequenceLength" call recorded in `TODO.md`'s Open Questions (now updated) — short
  conversations no longer pay the memory or `Model::load()`-time cost of a
  `maxSequenceLength`-sized KV-cache/graph they'll never fill.
- `Model::load()` now dispatches directly to a `.gguf` file path (`loadFromGgufFile()`) or a
  HuggingFace-layout directory (`loadFromSafetensorsDirectory()`) — the gguf dispatch gap the
  0.2.0-era `CLAUDE.md`/`TODO.md` explicitly called out as not yet wired is closed.
- Bumped `campello_nn` dependency from `v0.4.0` to `v0.5.0`.

## [0.2.0] - 2026-07-04

### Added
- Full GGUF model loading support: `Model::load()` now accepts a `.gguf` file path.
- GGUF Q4_0 and Q8_0 block-quantized tensor dequantization to Float32.
- GGUF metadata-based LLaMA config reader (`loadLlamaConfigFromGgufFile`).
- GGUF embedded tokenizer reader (`loadTokenizerFromGgufFile`) for SentencePiece-style BPE.
- GGUF-to-HuggingFace weight-name adapter (`GgufWeightsAdapter`) so the existing LLaMA graph builder consumes llama.cpp-named tensors.

### Changed
- Bumped `campello_nn` dependency from `v0.3.0` to `v0.4.0`.

### Fixed
- Ignored generated graph-cache files (`*.cache`) so they are no longer accidentally committed.

## [0.1.0] - 2026-06-24

### Added
- Initial release of `campello_llm`.
- CMake-based build with `FetchContent` for `campello_nn` and GoogleTest.
- Safetensors and GGUF weights-file readers.
- HuggingFace `tokenizer.json` BPE tokenizer.
- Architecture registry with LLaMA and GPT graph wiring.
- `Model::load()` / `Model::generate()` with KV-cache and graph caching.
- Cross-platform CI for macOS, Linux, and Windows.
