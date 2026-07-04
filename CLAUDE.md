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

Fetched via `FetchContent` in the top-level `CMakeLists.txt`, pinned to a version tag.
`campello_nn`'s own `CMakeLists.txt` declares an `option(BUILD_TESTS ...)` with the same name as
ours; the top-level `CMakeLists.txt` forces it off around the `FetchContent_MakeAvailable(campello_nn)`
call so a `-DBUILD_TESTS=ON` meant for `campello_llm`'s own tests doesn't also build (and run, via
`ctest`) `campello_nn`'s universal test suite — that suite's fixture paths resolve against
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
  (`src/json/json_value.{hpp,cpp}`, internal — shared with the Phase 2 tokenizer reader below)
  rather than a JSON dependency — same "hand-roll the minimal reader for a known, stable format"
  call `proto_reader.hpp` made for ONNX's protobuf in `campello_nn`. Dtype strings verified against
  the real `safetensors` Rust crate's `Dtype` enum (fetched from its GitHub source) before trusting
  them; sub-byte (F4/F6_*), exotic float8 (F8_*), and complex (C64) types have no campello_llm
  representation yet and throw rather than guess a conversion.
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

### Tokenizer (Phase 2)

`include/campello_llm/tokenizer.hpp`'s `Tokenizer` reads a HuggingFace "fast" `tokenizer.json`,
scoped specifically to the SentencePiece-style byte-fallback BPE shape LLaMA/TinyLlama-family
models ship — **not** a general tokenizer.json interpreter. It requires `model.type == "BPE"` with
`byte_fallback`, and matches exactly one of two shapes for each of `normalizer`/`decoder`/
`post_processor`: `null`, or the specific `Sequence` shape verified against TinyLlama's real
tokenizer.json (`Sequence[Prepend("▁"), Replace(" "->"▁")]` for the normalizer;
`Sequence[Replace("▁"->" "), ByteFallback, Fuse, Strip(" ",1,0)]` for the decoder;
`TemplateProcessing`'s `single` template for the post-processor). `pre_tokenizer` must be `null`
(the whole normalized text is one BPE "word" — no `ByteLevel`-style pre-splitting, which is a
structurally different algorithm GPT-2/Llama-3 use instead). Every one of these shapes, plus the
BPE merge/byte-fallback algorithm itself, was verified against the real `huggingface/tokenizers`
Rust source (`models/bpe/{model,word}.rs`, `decoders/{byte_fallback,strip}.rs`,
`normalizers/prepend.rs`, fetched from GitHub) before being trusted — not assumed from memory.
Anything else throws rather than silently mistokenizing.

- **Encode**: literal occurrences of any `added_tokens` string (e.g. `</s>`) are recognized and
  emitted as that token's id directly, scanning left-to-right with longest-content-first matching,
  *regardless* of `addSpecialTokens` — confirmed this is real behavior (not a bug) by checking the
  real tokenizer's output on `"a</s>b"` with `add_special_tokens=False`. `addSpecialTokens` only
  controls whether the post-processor's template token (`Tokenizer::bosId()`) gets prepended.
  Plain-text segments between added-token matches are normalized (if a normalizer is configured)
  and BPE-tokenized independently — `Tokenizer::bpeTokenizeWord()` splits per-Unicode-character
  (not per-byte), looks each character up in vocab directly, and falls back to per-byte
  `<0xXX>`-format vocab entries (`byte_fallback`) for characters with no direct vocab entry (CJK/
  emoji in TinyLlama's case), then repeatedly merges the lowest-rank adjacent pair (by index in
  `model.merges`) until none remain — logically equivalent to the real implementation's
  priority-queue approach, just a simpler full-rescan-per-merge (words here are a single chat
  message, not a training corpus, so this is in no way performance-critical).
- **Decode**: applies the verified decoder chain (space-marker replace, byte-fallback
  reassembly — with the real "each invalid byte becomes its own U+FFFD" fallback, not a single
  replacement for the whole run — fuse, then strip exactly one leading space, undoing the
  normalizer's artificial leading `"▁"`).
- **Chat prompt**: `formatSingleTurnChatPrompt(role, content, eosToken)` renders the literal
  `"<|{role}|>\n{content}{eosToken}\n<|assistant|>\n"` shape — confirmed against the real
  `chat_template` (a Jinja2 string in `tokenizer_config.json`) actually rendered with `jinja2`
  using HuggingFace's default `Environment(trim_blocks=True, lstrip_blocks=True,
  keep_trailing_newline=True)`, not guessed from reading the Jinja source. This is **not** a Jinja2
  interpreter — a different model's `chat_template` with a different literal shape needs a new
  formatting function, not a config flag here.
- **EOS/PAD**: not derivable from `tokenizer.json` alone (no field for it — only the post-processor
  implies a "BOS-like" token). `include/campello_llm/tokenizer_config.hpp`'s
  `loadSpecialTokenStringsFromFile()` reads `tokenizer_config.json`'s `bos_token`/`eos_token`/
  `unk_token`/`pad_token` (plain string or `{"content": "..."}` object) as literal text; resolve to
  an id via `Tokenizer::tokenToId()`. Kept deliberately separate from `Tokenizer` rather than
  merged in, since it's a different file with a different (much simpler) shape.
- **Not yet done**: a `GgufFile` -> `Tokenizer` adapter for gguf's embedded
  `tokenizer.ggml.tokens`/`merges` metadata (Phase 1's gguf reader already exposes these generically
  via `GgufFile::metadata()`, but nothing builds a `Tokenizer` from them yet) — deferred until the
  second, gguf-format target model is actually wired up (see `TODO.md` Open Questions).

Test fixtures are the **real** `tokenizer.json`/`tokenizer_config.json` from
`TinyLlama/TinyLlama-1.1B-Chat-v1.0` (Apache-2.0, see `tests/fixtures/NOTICE.md`) — not synthetic,
since the whole point is validating against the model this project actually targets. Every expected
token id list / decoded string in `tests/test_tokenizer.cpp` was independently confirmed against the
real `tokenizers` Python package (`pip install tokenizers`; `Tokenizer.from_file(path).encode()`/
`.decode()`) before being hardcoded — including CJK, emoji/byte-fallback, accented Latin text, and
the literal-added-token-recognition edge case above.

### Architecture Registry / Graph Wiring (Phase 3)

`include/campello_llm/architecture.hpp` turns `{a WeightsFile, hyperparameters}` into a compiled
`campello_nn::Graph` — the layer that justifies `campello_llm` existing separately from
`campello_nn`. `ArchitectureGraphResult` mirrors `campello_nn::OnnxImportResult`'s shape
(`{graph, inputs, outputs}`).

- **Fixed-shape, prefill-only graphs.** `buildLlamaGraph()`/`buildGptGraph()` each take a fixed
  `seqLen` and build a full-sequence causal-LM forward pass (batch size always 1, no KV-cache).
  This isn't a simplification of choice: `campello_nn::GraphBuilder::input()` bakes an exact
  `TensorDescriptor` shape into the graph at build time (no dynamic/`-1` dimension exists), so "one
  graph for both prefill and decode" isn't expressible at all — confirmed by reading
  `GraphBuilder::matmul()`'s/`reshape()`'s shape-inference code, not assumed. Both still exist and
  are tested as-is (`buildGptGraph()` has no decode-graph counterpart since GPT isn't a target
  model — see "First Target Model" below); `Model` itself no longer uses `buildLlamaGraph()` (see
  Phase 4's `buildLlamaDecodeGraph()` below).
- **`buildLlamaDecodeGraph()`** (`src/architecture/llama_architecture.cpp`, Phase 4): the real
  KV-cache graph, `seqLen=1` fixed, with explicit per-layer K/V cache tensors as inputs/outputs
  (`campello_nn` graphs are stateless — see `NN_ARCHITECTURE.md` §4 in `campello_nn` — so the cache
  is plain `Tensor`s the caller feeds back in and reads back out each call, not internal graph
  state). **Deliberately the only LLaMA graph `Model` builds** — one graph, not a batched-prefill
  graph plus a separate decode graph, specifically because every weight bound via `constant()` gets
  its own copy baked into whichever graph(s) it's bound to; a second graph would double resident
  weight memory (~4GB → ~8GB of decoded Float32 for TinyLlama) for the sake of one fast batched
  prefill dispatch. The cost: prefill is `promptLength` sequential small dispatches through this
  same graph instead of one batched dispatch — see `Model::generate()` below. RoPE's `cos`/`sin` and
  the causal mask can no longer be build-time constants the way `buildLlamaGraph()`'s are (the
  absolute position / valid-cache-length varies every call), so they're inputs instead:
  `rope_cos`/`rope_sin` (`[1, headDim]`, computed host-side per call by
  `weight_loading.hpp`'s `ropeCosSinForPosition()`) and `attn_mask` (`[1, maxSequenceLength + 1]`,
  additive — `0` for cache slots already filled plus the trailing slot for the current token, `-1e9`
  for not-yet-filled slots). Per layer, the freshly rotated K / projected V for just the current
  token (`llamaKeyCacheOutputName(i)`/`llamaValueCacheOutputName(i)`, shape
  `[numKeyValueHeads, 1, headDim]`) is `concat()`-ed onto that layer's incoming
  `[numKeyValueHeads, maxSequenceLength, headDim]` cache input before `repeatKvHeads()`/attention,
  giving every call a `[maxSequenceLength + 1]`-length keys/values axis (cache + current token);
  those same pre-concat K/V become the graph's per-layer outputs, for the caller to fold into its
  own persistent cache buffer at the right slot before the next call. Verified against
  `buildLlamaGraph()`'s own numpy-checked output: `LlamaArchitecture.
  DecodeGraphMatchesBatchedPrefillPerPosition` drives the decode graph one token at a time and
  checks each step's logits against the same hardcoded `expected[]` rows
  `MatchesNumpyReferenceForwardPass` already validates — the standard KV-cache correctness check
  (decode token-by-token == one-shot prefill over the equivalent sequence).
- **`buildLlamaGraph()`** (`src/architecture/llama_architecture.cpp`): embedding `gather` → N
  layers (`rmsNorm` → GQA attention with RoPE + causal mask → residual → `rmsNorm` → SwiGLU MLP →
  residual) → final `rmsNorm` → `lm_head` `matmul`. Two real gotchas worth knowing before touching
  this code:
  - **Weight layout.** HF LLaMA checkpoints store `nn.Linear` weights as `[out, in]`; `campello_nn`'s
    `matmul()` expects its second operand as `[in, out]` (confirmed against campello_nn's own
    `tests/universal/test_transformer_block.cpp`). `constantLinearWeightTransposed()`
    (`src/architecture/weight_loading.cpp`) transposes host-side, once, at graph-build time — not a
    graph node.
  - **GQA repeat-kv.** TinyLlama's real `config.json` has `numKeyValueHeads=4` vs.
    `numAttentionHeads=32`. `matmul()` requires exactly-matching batch dimensions (no broadcasting),
    so K/V must be physically repeated up to the full head count before the batched attention
    matmul — `repeatKvHeads()` does this via `slice()`+`concat()`, replicating each KV head
    contiguously (`[kv0,kv0,...,kv1,kv1,...]`), matching PyTorch's standard `repeat_kv` layout.
  - Only Float32 weights are supported (`constantWeight()`/`constantLinearWeightTransposed()` throw
    otherwise) — **real TinyLlama checkpoints are BF16** (`torch_dtype: bfloat16` in its
    `config.json`), so loading one for real needs a BF16->F32 (or ->F16) conversion step that
    doesn't exist yet. Tracked as Phase 4 work, not done here; Phase 3's own tests use Float32
    synthetic fixtures specifically to sidestep this gap for now.
- **`buildGptGraph()`** (`src/architecture/gpt_architecture.cpp`): the deliberately
  structurally-different second case — `layerNorm` (not `rmsNorm`), exact erf-based `gelu` (not
  SwiGLU; **note:** real GPT-2 uses the tanh-approximate `"gelu_new"`, which `campello_nn`'s `gelu()`
  op doesn't implement and there's no `tanh()` op to compose one from — a known small precision gap
  for real GPT-2 weights, not just a synthetic-test simplification), learned absolute positional
  embeddings (not RoPE), plain MHA (not GQA), and `gemm()` (matmul+bias in one op) for every
  projection since GPT-2's `Conv1D` layers all have biases. Two more real gotchas, both confirmed
  against the actual `openai-community/gpt2` checkpoint's safetensors header rather than assumed:
  - **Weight layout is the opposite of LLaMA's.** GPT-2's `Conv1D` weights are stored `[in, out]`
    already (`h.0.attn.c_attn.weight` is `[768, 2304]` = `[hiddenSize, 3*hiddenSize]`) — bound
    directly via `constantWeight()`, no transpose, unlike every LLaMA linear weight.
  - **Tied embeddings, no separate `lm_head.weight`.** The real checkpoint has no such tensor at
    all; `wte.weight` is reused transposed for the output projection.
- **Architecture registry** (`buildGraphForArchitecture()`,
  `src/architecture/architecture_registry.cpp`): dispatches on an architecture-name string
  (`"llama"`/`"gpt2"`) to the matching function above, taking an `ArchitectureConfig`
  (`std::variant<LlamaConfig, GptConfig>`) — throws if the name is unknown or if `config` holds the
  wrong alternative for the resolved architecture. **Scope boundary:** this dispatches on an
  *already-parsed* config struct; turning `config.json`'s `model_type` (safetensors) or gguf's
  `general.architecture` metadata key into a `LlamaConfig`/`GptConfig` is Phase 4's `Model::load()`
  job, not this registry's.

Test fixtures (`tests/fixtures/generate_{llama,gpt}_test_fixture.py`) are synthetic — tiny hidden
sizes, 2 layers, fixed-seed random Float32 weights — specifically so they sidestep the BF16-weight
gap above while still exercising every real architectural mechanism (GQA, RoPE, tied embeddings,
the opposite weight-layout conventions). Each script independently reimplements the *entire*
forward pass from scratch in plain numpy (deliberately not importing or calling any campello_llm
code) to produce the expected logits hardcoded into `test_llama_architecture.cpp`/
`test_gpt_architecture.cpp` — both matched the real `campello_nn` graph dispatch output to within
`1e-3` on the first actual run, which is reasonably strong evidence the wiring is correct and not
just plausible-looking.

### `Model::load()` / `generate()` — Phases 4-5

`include/campello_llm/model.hpp`'s `Model` ties everything above into something you can actually
load and talk to.

- **BF16/F16 weight decode** (`weight_loading.cpp`'s `decodeWeightToFloat32()`): real LLaMA
  checkpoints are BF16 (confirmed: TinyLlama's `config.json` says `"torch_dtype": "bfloat16"`).
  BF16->F32 is a 16-bit left-shift (BF16 is just an F32's truncated top half); F16->F32 reuses
  `campello_nn::decodeFloat16()` rather than re-implementing half-precision decode. Verified
  end-to-end (not just unit-testing the bit math) against a hand-rolled-binary BF16 safetensors
  fixture (`tests/fixtures/generate_llama_bf16_test_fixture.py` — neither numpy nor the installed
  `safetensors` Python binding can *write* BF16, so that script writes the safetensors binary
  format by hand, the same format Phase 1 already parses).
- **`loadLlamaConfigFromFile()`** (`architecture.hpp`/`llama_config_reader.cpp`): the `config.json`
  reader Phase 3's registry deferred to Phase 4 — `num_key_value_heads` defaults to
  `num_attention_heads` (plain MHA) and `rope_theta` defaults to `10000.0` when absent, matching
  `transformers`' own defaults for configs that predate those fields. Tested against TinyLlama's
  real `config.json`, not just a synthetic one.
- **`Model::load(context, directory, maxSequenceLength)`** reads the standard HF directory layout
  (`config.json`, `tokenizer.json`, `tokenizer_config.json`, `model.safetensors` — confirmed against
  the real `TinyLlama/TinyLlama-1.1B-Chat-v1.0` repo's file listing) and calls
  `buildLlamaDecodeGraph()` **once**, at a fixed `maxSequenceLength` (the KV-cache's capacity, i.e.
  the hard cap on prompt-tokens-plus-generated-tokens — same meaning as before, just enforced via
  cache size instead of a padded sequence length now). The multi-gigabyte `WeightsFile` is scoped to
  die right after that call (`constant()` copies bytes immediately — see Phase 3 notes — so nothing
  needs it afterward). **Only the safetensors+LLaMA path is wired** — no gguf dispatch through the
  architecture registry, no `loadFromMemory`/Android-asset variant. `tie_word_embeddings`,
  `pretraining_tp`, and other LLaMA `config.json` fields beyond what `LlamaConfig` already has
  aren't read.
- **Graph caching is implemented** — `Model::load()` writes/reads a sibling
  `campello_llm_decode_graph.<maxSequenceLength>.cache` file in the model directory via
  `campello_nn::saveGraphToFile`/`loadGraphFromFile`, keyed by `maxSequenceLength` since that's
  baked into the decode graph's per-layer cache-tensor shapes (a different `maxSequenceLength` is a
  structurally different graph, not just a different runtime parameter — see `graphCachePath()` in
  `src/model.cpp`). Freshness is a cheap mtime check (`isGraphCacheFresh()`): the cache is only
  trusted if it's at least as new as both `model.safetensors` and `config.json`, the same heuristic
  build-artifact caches like `make`/`ccache` use rather than a content hash. A missing/stale cache,
  or one that fails `loadGraphFromFile()`'s own magic/version check (corrupt or from an incompatible
  `campello_nn` version), falls back to building fresh via `buildLlamaDecodeGraph()` rather than
  failing `Model::load()`; writing a fresh cache back is similarly best-effort (a read-only model
  directory shouldn't fail loading, it just skips caching for next time). Round-tripped and the
  corrupt-cache fallback both covered by `Model.GraphCacheRoundTripProducesIdenticalOutput`/
  `Model.CorruptGraphCacheFallsBackToBuildingFresh` in `tests/test_model.cpp`.
- **`Model::generate()`** (`src/model.cpp`) — real KV-cache, host-managed. Allocates one
  zero-initialized `[numKeyValueHeads, maxSequenceLength, headDim]` buffer per layer per K/V (zeroed
  so not-yet-filled slots are deterministic, harmless once `attn_mask` masks them out regardless —
  not garbage memory). Two phases through the same `dispatchStep(tokenId, position)` helper (one
  `buildLlamaDecodeGraph()` dispatch each call, writing that position's `rope_cos`/`rope_sin`/
  `attn_mask`/cache-in tensors and folding the K/V outputs into the host cache buffers at slot
  `position` before returning):
  - **Phase A (prefill):** one `dispatchStep()` call per prompt token (positions `0` ..
    `promptLength-1`), filling the cache; the last call's logits seed phase B's first sample.
  - **Phase B (decode):** sample from the previous call's logits, append/stream/check stop
    conditions exactly as before, then `dispatchStep()` once for the newly sampled token to extend
    the cache and obtain the next position's logits.

  Total cost: `O(1)` of the heavy per-token matmuls (one token's worth of work each call, not the
  whole sequence's, unlike the old brute-force re-dispatch) plus the unavoidable
  `O(position)` attention-over-the-cache cost each call — `O(n)` amortized per generated token,
  `O(n²)` total across a full generation (inherent to attention itself, not a shortcut). The
  tradeoff for using *one* graph for both prefill and decode (deliberate — see
  `buildLlamaDecodeGraph()` above) is that prefill is `promptLength` sequential small dispatches
  rather than a single batched one; for short chat-turn prompts this is still far cheaper than the
  old per-generated-token full-`maxSequenceLength` recompute. Sampling (`sampleNextToken()`): greedy
  argmax when `temperature <= 0`, else temperature scaling + top-k + top-p nucleus filtering +
  `std::discrete_distribution`, seeded from `std::random_device` (no reproducibility-across-runs
  guarantee for non-greedy sampling — only greedy is tested for determinism, per `TODO.md` Phase 5's
  own test plan). Streaming (`onToken`) decodes the *entire* generated-so-far id list fresh every
  step and emits only the new suffix, rather than decoding/emitting per-token — necessary because a
  byte-fallback multi-byte UTF-8 character can span several token ids that only decode to real text
  once all of them exist.
- **`formatSingleTurnChatPrompt()`** (Phase 2, `tokenizer.hpp`) plus a small CLI
  (`examples/cli_chat/`, `option(BUILD_EXAMPLES ...)`, mirrors `campello_nn`'s own example
  CMake/option pattern **including its `BUILD_TESTS`-style name collision** — `BUILD_EXAMPLES` is
  forced off around `FetchContent_MakeAvailable(campello_nn)` too, for the same reason as
  `BUILD_TESTS`) is the actual minimum interactive example: load a model directory, read a line,
  format it as a chat prompt, stream the reply.

Test fixtures: `tests/fixtures/tiny_llama_model/` (`generate_tiny_llama_model_fixture.py`) is a
*complete* tiny synthetic HF-layout checkpoint directory (config.json + tokenizer.json +
tokenizer_config.json + model.safetensors, all consistent with each other) — built via the real
`tokenizers` Python library's direct `models.BPE(vocab=..., merges=...)` construction (not trained),
deliberately without a normalizer/decoder/byte_fallback (those are already covered against the real
TinyLlama tokenizer in `test_tokenizer.cpp`) so it stays small and fast. Building it surfaced a real
format gotcha worth remembering: that library's *current* version (0.22.2) serializes
`model.merges` as 2-element `["left","right"]` arrays, not the single `"left right"` string
TinyLlama's real (older-`tokenizers`-version-produced) tokenizer.json uses — both are valid,
`Tokenizer`'s merges parser now accepts either rather than guessing one was wrong.

### First Target Model

**TinyLlama-1.1B, safetensors.** LLaMA architecture (RMSNorm, RoPE, SwiGLU, GQA) —
`buildLlamaDecodeGraph()` plus the BF16 decode above now implement this shape completely with a
real KV-cache; `Model::load()`/`generate()` can load and run TinyLlama's actual checkpoint (see
`TODO.md`'s Phase 6 note on the real end-to-end attempt — and its CPU-performance caveat: even with
the KV-cache eliminating the old `O(maxSequenceLength²)` blowup, a *single* token's forward pass
through all 22 layers is itself slow on campello_nn's current reference CPU backend, no
BLAS/SIMD-optimized GEMM — for how well "run" holds up in practice). A small GGUF-quantized build of
the same architecture is the second target, once gguf
dispatch is wired into `Model::load()` — gguf's embedded vocab/merges should reuse the tokenizer's
existing BPE encode/decode logic rather than a second implementation. GPT-style architecture
(`buildGptGraph()`) is implemented and tested but isn't a target model on its own — it exists
specifically to validate the architecture registry generalizes instead of being accidentally
LLaMA-shaped (see `TODO.md` Phase 3).

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

Phases 0-3 (scaffolding, weights-file parsing, tokenizer, architecture registry/graph wiring) are
done. Phases 4-5 (`Model::load()`/`generate()`) are done, including a real KV-cache
(`buildLlamaDecodeGraph()` — see above) and graph caching (see above) — the one deliberate
remaining scope cut is no gguf dispatch. BF16/F16 weight conversion is done, so TinyLlama's real
checkpoint loads. See `TODO.md` for what's still open, then Phase 6 (cross-backend conformance,
benchmarks, graph-caching demo).
