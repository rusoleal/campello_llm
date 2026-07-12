#pragma once

#include <cstdint>
#include <functional>
#include <memory>
#include <string>

#include <campello_nn/context.hpp>
#include <campello_nn/tensor.hpp>

#include <campello_llm/architecture.hpp>
#include <campello_llm/generation_config.hpp>
#include <campello_llm/tokenizer.hpp>

namespace systems::leal::campello_llm
{

    /**
     * @brief A loaded LLaMA-family model, ready to `generate()` from. Ties together
     * `loadLlamaConfigFromFile()`, `loadTokenizerFromFile()`,
     * `loadSpecialTokenStringsFromFile()`, `loadSafetensorsFromFile()`, and
     * `buildLlamaDecodeGraph()` into the single loadable object the architecture doc
     * describes. GPT-style models aren't wired up here (`buildGptGraph()` exists only
     * to validate the architecture registry — see `CLAUDE.md`); add a parallel
     * `Model` (or a `architectureName` parameter) if a real GPT-2-family target model
     * is ever needed.
     *
     * **Real KV-cache (see `CLAUDE.md`).** `Model::load()` compiles a single
     * fixed-`seqLen=1` decode graph with explicit KV-cache tensors as inputs/outputs
     * (`buildLlamaDecodeGraph()`) — there is no separate batched-prefill graph, so
     * weights are only ever bound once. `generate()` dispatches that graph once per
     * prompt token (filling the cache) and once per generated token, folding each
     * call's K/V output into a host-held cache buffer for the next call. Total work
     * is `O(maxSequenceLength)` for the heavy per-token matmuls (one token's worth
     * each call, not the whole sequence's) plus the unavoidable `O(maxSequenceLength)`
     * per-step attention-over-the-cache cost — i.e. `O(n)` amortized per token instead
     * of the brute-force re-dispatch's `O(maxSequenceLength)` per token.
     */
    class Model
    {
    public:
        /**
         * @brief Loads a model from either:
         *   - a directory containing `config.json`, `tokenizer.json`,
         *     `tokenizer_config.json`, and `model.safetensors` — the standard
         *     HuggingFace LLaMA checkpoint layout (e.g. `TinyLlama/TinyLlama-1.1B-Chat-v1.0`), or
         *   - a `.gguf` file containing embedded LLaMA config, tokenizer, and weights.
         *
         * @param maxSequenceLength Fixed size of the compiled graph — the hard cap on
         * prompt-tokens-plus-generated-tokens for any `generate()` call against the
         * returned `Model`. Larger values cost proportionally more compute per
         * generated token (see the class-level note on the lack of a KV-cache).
         * @throws std::runtime_error on a missing/malformed file, or an
         * unsupported/wrong-dtype/wrong-shape weight tensor (see `buildLlamaGraph()`).
         */
        static std::unique_ptr<Model> load(std::shared_ptr<systems::leal::campello_nn::Context> context,
                                            const std::string &directory, std::int64_t maxSequenceLength);

        /**
         * @brief Tokenizes `prompt` (with the post-processor's BOS, if any),
         * generates up to `config.maxTokens` further tokens (greedy argmax if
         * `config.temperature <= 0`, else temperature/top-k/top-p sampling), and
         * decodes the generated portion back to text.
         *
         * Stops at end-of-sequence (`tokenizer_config.json`'s `eos_token`, if it
         * resolves to a real vocab id), `config.maxTokens`, hitting
         * `maxSequenceLength`, or any of `config.stopSequences`.
         *
         * @param onToken If set, invoked once per generated token with the
         * incremental decoded text since the last call (not necessarily one
         * character/token per call — byte-fallback tokens spanning a multi-byte
         * UTF-8 character only decode to real text once the whole character's bytes
         * have been generated).
         * @throws std::runtime_error if the tokenized prompt (plus its BOS) already
         * reaches `maxSequenceLength`, leaving no room to generate anything.
         *
         * Not `const`: the KV-cache capacity backing this call starts small and grows
         * (doubling, capped at `maxSequenceLength`) as the conversation needs more
         * room, rebuilding the compiled decode graph on each doubling — see
         * `currentCapacity_`/`rebuildGraph_`. That capacity is a `Model`-lifetime
         * high-water mark, so later calls reuse whatever a previous call already grew
         * into instead of starting over.
         */
        std::string generate(const std::string &prompt, const GenerationConfig &config,
                              const std::function<void(const std::string &)> &onToken = nullptr);

        const Tokenizer &tokenizer() const { return *tokenizer_; }

        std::int64_t maxSequenceLength() const { return maxSequenceLength_; }
        std::int64_t vocabSize() const { return vocabSize_; }
        std::int64_t numLayers() const { return numLayers_; }
        const std::string &architectureName() const { return architectureName_; }

    private:
        Model() = default;

        static std::unique_ptr<Model> loadFromSafetensorsDirectory(
            std::shared_ptr<systems::leal::campello_nn::Context> context, const std::string &directory,
            std::int64_t maxSequenceLength);
        static std::unique_ptr<Model> loadFromGgufFile(
            std::shared_ptr<systems::leal::campello_nn::Context> context, const std::string &path,
            std::int64_t maxSequenceLength);

        std::shared_ptr<systems::leal::campello_nn::Context> context_;
        std::unique_ptr<Tokenizer> tokenizer_;
        ArchitectureGraphResult graph_;
        std::int64_t maxSequenceLength_ = 0;
        std::int64_t vocabSize_ = 0;
        std::int32_t eosId_ = -1;
        std::int64_t numLayers_ = 0;
        std::int64_t numKeyValueHeads_ = 0;
        std::int64_t headDim_ = 0;
        float ropeTheta_ = 10000.0f;
        std::string architectureName_ = "llama";

        // KV-cache/decode-graph capacity currently backing `graph_` (<= maxSequenceLength_).
        // Starts at a small initial bucket and only ever grows (doubling), so a Model
        // that never sees a long conversation never pays for a maxSequenceLength_-sized
        // KV-cache. See `generate()`'s KV-cache-growth lambda.
        std::int64_t currentCapacity_ = 0;

        // Rebuilds (or loads from the on-disk decode-graph cache) the compiled decode
        // graph at a new capacity, re-reading weights from their original file rather
        // than retaining a parsed copy on `Model` for this purpose alone -- the graph
        // already holds its own copy of every weight it binds as a constant, so keeping
        // a second parsed copy around just to make regrowth cheap would double the
        // resident weight memory. Set once at load time (captures the model's path and
        // `LlamaConfig`, not the parsed weights file).
        std::function<ArchitectureGraphResult(std::int64_t)> rebuildGraph_;
    };

} // namespace systems::leal::campello_llm
