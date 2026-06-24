#pragma once

#include <cstdint>
#include <memory>
#include <string>
#include <unordered_map>
#include <variant>
#include <vector>

#include <campello_nn/context.hpp>
#include <campello_nn/descriptors/tensor_descriptor.hpp>
#include <campello_nn/graph.hpp>

#include <campello_llm/weights_file.hpp>

namespace systems::leal::campello_llm
{

    /**
     * @brief A compiled architecture graph plus what's needed to actually use it —
     * same shape as `campello_nn::OnnxImportResult`/`GraphCacheResult`.
     */
    struct ArchitectureGraphResult
    {
        std::shared_ptr<systems::leal::campello_nn::Graph> graph;
        std::unordered_map<std::string, systems::leal::campello_nn::TensorDescriptor> inputs;
        std::unordered_map<std::string, systems::leal::campello_nn::TensorDescriptor> outputs;

        /**
         * @brief The same IR `graph` above was compiled from, in `GraphBuilder::serialize()`'s
         * on-disk form (`campello_nn::GraphBuilder::deserialize()`/`loadGraphFromMemory()`/
         * `loadGraphFromFile()`-compatible bytes, magic/version-checked on reload — see
         * `campello_nn`'s `graph_cache.hpp`). `Model::load()` uses this to write a graph-cache
         * file (Phase 4); callers that don't care about caching can ignore it.
         */
        std::vector<std::uint8_t> serializedGraph;
    };

    /**
     * @brief Hyperparameters for a LLaMA-family decoder-only model (TinyLlama, Llama
     * 2/3, Mistral, ...) — the field names match HuggingFace `config.json` keys
     * directly (`hidden_size` -> `hiddenSize`, etc.), populated 1:1 by
     * `loadLlamaConfigFromFile()`/`loadLlamaConfigFromMemory()` below.
     */
    struct LlamaConfig
    {
        std::int64_t vocabSize;
        std::int64_t hiddenSize;
        std::int64_t numLayers;
        std::int64_t numAttentionHeads;
        std::int64_t numKeyValueHeads; // == numAttentionHeads for plain MHA; < for GQA
        std::int64_t intermediateSize;
        float rmsNormEps;
        float ropeTheta;
    };

    /**
     * @brief Parses a HuggingFace `config.json` already loaded into memory into a
     * `LlamaConfig`. `num_key_value_heads` defaults to `num_attention_heads` (plain
     * MHA) if absent, matching transformers' own default for configs predating GQA;
     * `rope_theta` defaults to `10000.0` (transformers' own default) if absent.
     * @throws std::runtime_error if `model_type != "llama"`, or a required field
     * (everything else) is missing.
     */
    LlamaConfig loadLlamaConfigFromMemory(const void *data, std::size_t size);

    /**
     * @brief Reads `path` into memory and calls `loadLlamaConfigFromMemory()`.
     */
    LlamaConfig loadLlamaConfigFromFile(const std::string &path);

    /**
     * @brief Hyperparameters for a GPT-2-family decoder-only model — deliberately
     * structurally different from `LlamaConfig` (LayerNorm not RMSNorm, GELU not
     * SwiGLU, learned absolute positional embeddings not RoPE, plain MHA not GQA,
     * and every linear layer has a bias) to validate the architecture registry
     * generalizes beyond LLaMA-shaped models.
     */
    struct GptConfig
    {
        std::int64_t vocabSize;
        std::int64_t hiddenSize;
        std::int64_t numLayers;
        std::int64_t numAttentionHeads;
        std::int64_t intermediateSize;
        float layerNormEps;
        std::int64_t maxPositionEmbeddings; // size of the learned positional embedding table
    };

    /**
     * @brief Builds a fixed-`seqLen`, full-sequence ("prefill-shaped") causal-LM
     * forward pass graph for a LLaMA-family model: embedding `gather` -> N decoder
     * layers (`rmsNorm` -> GQA attention with RoPE + an additive causal mask ->
     * residual -> `rmsNorm` -> SwiGLU MLP -> residual) -> final `rmsNorm` -> `lm_head`
     * projection. No KV-cache (Phase 4); batch size is always 1.
     *
     * Expects weights named per HuggingFace's standard LLaMA checkpoint layout
     * (`model.embed_tokens.weight`, `model.layers.{i}.self_attn.{q,k,v,o}_proj.weight`,
     * `model.layers.{i}.{input,post_attention}_layernorm.weight`,
     * `model.layers.{i}.mlp.{gate,up,down}_proj.weight`, `model.norm.weight`,
     * `lm_head.weight`) as Float32 tensors — see `CLAUDE.md` for why only Float32 is
     * supported so far (real LLaMA checkpoints ship BF16; conversion is deferred).
     *
     * @param seqLen Fixed input sequence length this graph is compiled for.
     * @throws std::runtime_error on a missing/wrong-dtype/wrong-shape weight tensor.
     */
    ArchitectureGraphResult buildLlamaGraph(std::shared_ptr<systems::leal::campello_nn::Context> context,
                                             const WeightsFile &weights, const LlamaConfig &config,
                                             std::int64_t seqLen);

    /** @brief Per-layer KV-cache tensor names `buildLlamaDecodeGraph()` and `Model::generate()` agree on. */
    std::string llamaKeyCacheInputName(std::int64_t layer);
    std::string llamaValueCacheInputName(std::int64_t layer);
    std::string llamaKeyCacheOutputName(std::int64_t layer);
    std::string llamaValueCacheOutputName(std::int64_t layer);

    /**
     * @brief Builds a fixed-`seqLen=1` LLaMA decode-step graph with an explicit
     * KV-cache, per the real Phase 4 design (see `CLAUDE.md`) — as opposed to
     * `buildLlamaGraph()`'s fixed-full-sequence, no-cache prefill shape. One compiled
     * graph serves both prefill (dispatched once per prompt token, to fill the
     * cache) and decode (dispatched once per generated token) — there is no separate
     * batched-prefill graph, specifically so weights are only ever bound via
     * `constant()` once (a second graph would double resident weight memory, ~4GB ->
     * ~8GB for TinyLlama's decoded Float32 weights).
     *
     * Inputs: `"input_ids"` (Int32 `[1]`), `"rope_cos"`/`"rope_sin"` (Float32
     * `[1, headDim]`, computed by the caller for the absolute position being
     * processed — unlike `buildLlamaGraph()`'s baked-in RoPE table, these vary call to
     * call and so must be runtime inputs, not constants), `"attn_mask"` (Float32
     * `[1, maxSequenceLength + 1]`, additive — `0` for cache slots already filled by a
     * prior call plus the trailing slot for the current token, `-1e9` for not-yet-filled
     * slots; the caller recomputes this every call since which slots are valid changes
     * each step), and per layer `llamaKeyCacheInputName(i)`/`llamaValueCacheInputName(i)`
     * (Float32 `[numKeyValueHeads, maxSequenceLength, headDim]` — the cache accumulated
     * by every prior call, *not including* the current token).
     *
     * Outputs: `"logits"` (Float32 `[1, vocabSize]`) and per layer
     * `llamaKeyCacheOutputName(i)`/`llamaValueCacheOutputName(i)` (Float32
     * `[numKeyValueHeads, 1, headDim]` — the freshly rotated K / projected V for just
     * the current token, pre-repeat-kv). The caller is responsible for folding these
     * into its own persistent cache buffers at the right slot before the next call —
     * this graph never reads or writes "the cache" as a single stateful object, only
     * the explicit Tensors bound at dispatch time (`campello_nn` graphs are stateless).
     *
     * @param maxSequenceLength The cache's capacity (same value `Model::load()`'s
     * caller picks) — the hard cap on prompt-tokens-plus-generated-tokens.
     * @throws std::runtime_error on a missing/wrong-dtype/wrong-shape weight tensor.
     */
    ArchitectureGraphResult buildLlamaDecodeGraph(std::shared_ptr<systems::leal::campello_nn::Context> context,
                                                   const WeightsFile &weights, const LlamaConfig &config,
                                                   std::int64_t maxSequenceLength);

    /**
     * @brief Builds a fixed-`seqLen` GPT-2-family forward pass graph: embedding
     * `gather` + learned positional embedding -> N decoder layers (`layerNorm` ->
     * MHA with causal mask and biased q/k/v/output projections -> residual ->
     * `layerNorm` -> GELU MLP with biases -> residual) -> final `layerNorm` ->
     * `lm_head` projection.
     *
     * Expects weights named per HuggingFace's standard GPT-2 checkpoint layout
     * (`wte.weight`, `wpe.weight`, `h.{i}.ln_1.{weight,bias}`,
     * `h.{i}.attn.c_attn.{weight,bias}` (fused qkv), `h.{i}.attn.c_proj.{weight,bias}`,
     * `h.{i}.ln_2.{weight,bias}`, `h.{i}.mlp.c_fc.{weight,bias}`,
     * `h.{i}.mlp.c_proj.{weight,bias}`, `ln_f.{weight,bias}`) as Float32 tensors.
     * **No separate `lm_head.weight`** — GPT-2 ties its output projection to
     * `wte.weight` (confirmed against the real `openai-community/gpt2` checkpoint,
     * which has no `lm_head.weight` tensor at all), reused here transposed.
     * `attn.c_attn`/`c_proj`/`mlp.c_fc`/`mlp.c_proj` weights are stored `[in, out]`
     * (HF GPT-2's `Conv1D`, confirmed against that same checkpoint) — the *opposite*
     * convention from LLaMA's `nn.Linear` `[out, in]`, so unlike `buildLlamaGraph()`
     * these are bound directly with no host-side transpose.
     *
     * @throws std::runtime_error on a missing/wrong-dtype/wrong-shape weight tensor.
     */
    ArchitectureGraphResult buildGptGraph(std::shared_ptr<systems::leal::campello_nn::Context> context,
                                           const WeightsFile &weights, const GptConfig &config, std::int64_t seqLen);

    /**
     * @brief The set of hyperparameter structs the registry can dispatch to. Phase 4's
     * `Model::load()` is responsible for parsing `config.json`/gguf metadata into one
     * of these — this registry only does the name -> wiring-function dispatch.
     */
    using ArchitectureConfig = std::variant<LlamaConfig, GptConfig>;

    /**
     * @brief Dispatches to `buildLlamaGraph()`/`buildGptGraph()` by `architectureName`
     * (`"llama"` or `"gpt2"`, matching HuggingFace `config.json`'s `model_type` /
     * gguf's `general.architecture` convention).
     * @throws std::runtime_error if `architectureName` is unknown, or `config` doesn't
     * hold the variant alternative the resolved architecture expects.
     */
    ArchitectureGraphResult buildGraphForArchitecture(const std::string &architectureName,
                                                       std::shared_ptr<systems::leal::campello_nn::Context> context,
                                                       const WeightsFile &weights, const ArchitectureConfig &config,
                                                       std::int64_t seqLen);

} // namespace systems::leal::campello_llm
