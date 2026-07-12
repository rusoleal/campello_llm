#include <campello_llm/model.hpp>

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <filesystem>
#include <numeric>
#include <random>
#include <stdexcept>
#include <string>
#include <unordered_map>
#include <vector>

#include <campello_llm/gguf_reader.hpp>
#include <campello_llm/safetensors_reader.hpp>
#include <campello_llm/tokenizer_config.hpp>
#include <campello_nn/graph_cache.hpp>

#include "architecture/weight_loading.hpp"
#include "gguf/gguf_model_loader.hpp"

namespace cnn = systems::leal::campello_nn;
namespace fs = std::filesystem;

namespace
{
    std::string joinPath(const std::string &directory, const char *fileName)
    {
        if (!directory.empty() && directory.back() == '/')
        {
            return directory + fileName;
        }
        return directory + "/" + fileName;
    }

    // Sibling cache file, keyed by maxSequenceLength since that's baked into the
    // decode graph's cache-tensor shapes -- a different maxSequenceLength is a
    // structurally different graph, not just a different runtime parameter.
    std::string graphCachePath(const std::string &directory, std::int64_t maxSequenceLength)
    {
        return joinPath(directory, ("campello_llm_decode_graph." + std::to_string(maxSequenceLength) + ".cache").c_str());
    }

    // A cache is only trusted if it's at least as new as both the weights and the
    // config it was presumably built from -- cheap mtime check, not a content hash,
    // matching how build-artifact caches (make, ccache) usually decide freshness.
    bool isGraphCacheFresh(const std::string &cachePath, const std::string &weightsPath, const std::string &configPath)
    {
        std::error_code ec;
        if (!fs::exists(cachePath, ec) || ec)
        {
            return false;
        }
        fs::file_time_type cacheTime = fs::last_write_time(cachePath, ec);
        if (ec)
        {
            return false;
        }
        for (const std::string &sourcePath : {weightsPath, configPath})
        {
            fs::file_time_type sourceTime = fs::last_write_time(sourcePath, ec);
            if (ec || sourceTime > cacheTime)
            {
                return false;
            }
        }
        return true;
    }

    // Greedy argmax (temperature <= 0) or temperature/top-k/top-p sampling, on CPU,
    // straight off the freshly-read-back logits row -- per TODO.md Phase 5.
    std::int32_t sampleNextToken(const float *logits, std::int64_t vocabSize,
                                  const systems::leal::campello_llm::GenerationConfig &config, std::mt19937 &rng)
    {
        if (config.temperature <= 0.0f)
        {
            std::int64_t best = 0;
            for (std::int64_t i = 1; i < vocabSize; ++i)
            {
                if (logits[i] > logits[best])
                {
                    best = i;
                }
            }
            return static_cast<std::int32_t>(best);
        }

        std::vector<std::int64_t> order(static_cast<std::size_t>(vocabSize));
        std::iota(order.begin(), order.end(), 0);
        std::sort(order.begin(), order.end(),
                  [&](std::int64_t a, std::int64_t b) { return logits[a] > logits[b]; });

        std::size_t keep = order.size();
        if (config.topK > 0 && config.topK < order.size())
        {
            keep = config.topK;
        }

        // Softmax (numerically stable: subtract the max, which is logits[order[0]])
        // over the kept candidates only, in descending-probability order.
        std::vector<double> probs(keep);
        double maxLogit = logits[order[0]] / config.temperature;
        double sum = 0.0;
        for (std::size_t i = 0; i < keep; ++i)
        {
            probs[i] = std::exp(logits[order[i]] / config.temperature - maxLogit);
            sum += probs[i];
        }
        for (double &p : probs)
        {
            p /= sum;
        }

        if (config.topP < 1.0f)
        {
            double cumulative = 0.0;
            std::size_t cutoff = probs.size();
            for (std::size_t i = 0; i < probs.size(); ++i)
            {
                cumulative += probs[i];
                if (cumulative >= config.topP)
                {
                    cutoff = i + 1;
                    break;
                }
            }
            probs.resize(cutoff);
            double renorm = std::accumulate(probs.begin(), probs.end(), 0.0);
            for (double &p : probs)
            {
                p /= renorm;
            }
        }

        std::discrete_distribution<std::size_t> dist(probs.begin(), probs.end());
        return static_cast<std::int32_t>(order[dist(rng)]);
    }
} // namespace

namespace systems::leal::campello_llm
{

    namespace
    {
        std::int32_t eosTokenIdFromTokenizerConfig(const Tokenizer &tokenizer,
                                                    const SpecialTokenStrings &specialTokens)
        {
            if (!specialTokens.eosToken.empty())
            {
                if (auto id = tokenizer.tokenToId(specialTokens.eosToken))
                {
                    return *id;
                }
            }
            return -1;
        }

        std::int32_t eosTokenIdFromGguf(const Tokenizer &tokenizer, const GgufFile &file)
        {
            const GgufValue *eosId = file.findMetadata("tokenizer.ggml.eos_token_id");
            if (eosId != nullptr && eosId->type == GgufValueType::Int32)
            {
                std::int32_t id = eosId->asInt32();
                if (id >= 0 && static_cast<std::size_t>(id) < tokenizer.vocabSize())
                {
                    return id;
                }
            }
            return -1;
        }

        std::string ggufGraphCachePath(const std::string &ggufPath, std::int64_t maxSequenceLength)
        {
            return ggufPath + ".campello_llm_decode_graph." + std::to_string(maxSequenceLength) + ".cache";
        }

        // The initial KV-cache/decode-graph capacity a freshly loaded Model starts at,
        // before generate() grows it (doubling) toward maxSequenceLength as needed.
        // Keeps `Model::load()` cheap and avoids paying for a maxSequenceLength-sized
        // KV-cache for conversations that never get that long.
        constexpr std::int64_t kInitialKvCacheCapacity = 256;

        std::int64_t initialCapacityFor(std::int64_t maxSequenceLength)
        {
            return std::min<std::int64_t>(kInitialKvCacheCapacity, maxSequenceLength);
        }

        // Builds (or loads from the on-disk decode-graph cache) the compiled decode
        // graph for a GGUF model at `capacity`. Always re-reads/re-parses the GGUF file
        // rather than reusing a retained `GgufFile` -- buildLlamaDecodeGraph() copies
        // every weight it binds into the graph's own constants, so keeping a second
        // parsed copy around just to make later regrowth cheap would double the
        // resident weight memory. This only runs when growing past a capacity that
        // hasn't been built before (or its cache file is stale/missing).
        ArchitectureGraphResult buildOrLoadGgufDecodeGraph(std::shared_ptr<cnn::Context> context,
                                                            const std::string &path, const LlamaConfig &config,
                                                            std::int64_t capacity)
        {
            std::string cachePath = ggufGraphCachePath(path, capacity);
            if (isGraphCacheFresh(cachePath, path, path))
            {
                try
                {
                    cnn::GraphCacheResult cached = cnn::loadGraphFromFile(context, cachePath);
                    ArchitectureGraphResult result;
                    result.graph = cached.graph;
                    result.inputs = std::move(cached.inputs);
                    result.outputs = std::move(cached.outputs);
                    return result;
                }
                catch (const std::exception &)
                {
                    // Corrupt/truncated/incompatible-version cache: fall back to building fresh.
                }
            }

            auto gguf = loadGgufFromFile(path);
            GgufWeightsAdapter weights(*gguf);
            ArchitectureGraphResult result = buildLlamaDecodeGraph(context, weights, config, capacity);
            try
            {
                cnn::saveGraphToFile(result.serializedGraph, cachePath);
            }
            catch (const std::exception &)
            {
                // Best-effort: skip caching on read-only directories.
            }
            // `serializedGraph` is only useful for the on-disk cache write above --
            // ends up stored on Model::graph_ (a member with the Model's own
            // lifetime) otherwise, retaining a full second copy of every weight byte
            // (roughly the size of the model file itself) for as long as the model
            // stays loaded, for no further purpose.
            result.serializedGraph.clear();
            result.serializedGraph.shrink_to_fit();
            return result;
        }

        // Same as buildOrLoadGgufDecodeGraph(), for the HuggingFace safetensors layout.
        ArchitectureGraphResult buildOrLoadSafetensorsDecodeGraph(std::shared_ptr<cnn::Context> context,
                                                                   const std::string &directory,
                                                                   const LlamaConfig &config, std::int64_t capacity)
        {
            std::string weightsPath = joinPath(directory, "model.safetensors");
            std::string configPath = joinPath(directory, "config.json");
            std::string cachePath = graphCachePath(directory, capacity);
            if (isGraphCacheFresh(cachePath, weightsPath, configPath))
            {
                try
                {
                    cnn::GraphCacheResult cached = cnn::loadGraphFromFile(context, cachePath);
                    ArchitectureGraphResult result;
                    result.graph = cached.graph;
                    result.inputs = std::move(cached.inputs);
                    result.outputs = std::move(cached.outputs);
                    return result;
                }
                catch (const std::exception &)
                {
                    // Corrupt cache: fall back to building fresh.
                }
            }

            auto weights = loadSafetensorsFromFile(weightsPath);
            ArchitectureGraphResult result = buildLlamaDecodeGraph(context, *weights, config, capacity);
            try
            {
                cnn::saveGraphToFile(result.serializedGraph, cachePath);
            }
            catch (const std::exception &)
            {
                // Best-effort: skip caching on read-only directories.
            }
            // See buildOrLoadGgufDecodeGraph()'s comment: drop the now-unneeded second
            // copy of every weight byte before this result becomes Model::graph_.
            result.serializedGraph.clear();
            result.serializedGraph.shrink_to_fit();
            return result;
        }
    } // namespace

    std::unique_ptr<Model> Model::loadFromSafetensorsDirectory(std::shared_ptr<cnn::Context> context,
                                                                const std::string &directory,
                                                                std::int64_t maxSequenceLength)
    {
        LlamaConfig config = loadLlamaConfigFromFile(joinPath(directory, "config.json"));
        auto tokenizer = loadTokenizerFromFile(joinPath(directory, "tokenizer.json"));
        SpecialTokenStrings specialTokens =
            loadSpecialTokenStringsFromFile(joinPath(directory, "tokenizer_config.json"));

        auto model = std::unique_ptr<Model>(new Model());
        model->context_ = context;
        model->tokenizer_ = std::move(tokenizer);
        model->maxSequenceLength_ = maxSequenceLength;
        model->vocabSize_ = config.vocabSize;
        model->numLayers_ = config.numLayers;
        model->numKeyValueHeads_ = config.numKeyValueHeads;
        model->headDim_ = config.hiddenSize / config.numAttentionHeads;
        model->ropeTheta_ = config.ropeTheta;
        model->architectureName_ = "llama";
        model->eosId_ = eosTokenIdFromTokenizerConfig(*model->tokenizer_, specialTokens);

        model->currentCapacity_ = initialCapacityFor(maxSequenceLength);
        model->graph_ = buildOrLoadSafetensorsDecodeGraph(context, directory, config, model->currentCapacity_);
        model->rebuildGraph_ = [context, directory, config](std::int64_t capacity) {
            return buildOrLoadSafetensorsDecodeGraph(context, directory, config, capacity);
        };

        return model;
    }

    std::unique_ptr<Model> Model::loadFromGgufFile(std::shared_ptr<cnn::Context> context,
                                                    const std::string &path,
                                                    std::int64_t maxSequenceLength)
    {
        auto gguf = loadGgufFromFile(path);
        LlamaConfig config = loadLlamaConfigFromGgufFile(*gguf);
        auto tokenizer = loadTokenizerFromGgufFile(*gguf);

        auto model = std::unique_ptr<Model>(new Model());
        model->context_ = context;
        model->tokenizer_ = std::move(tokenizer);
        model->maxSequenceLength_ = maxSequenceLength;
        model->vocabSize_ = config.vocabSize;
        model->numLayers_ = config.numLayers;
        model->numKeyValueHeads_ = config.numKeyValueHeads;
        model->headDim_ = config.hiddenSize / config.numAttentionHeads;
        model->ropeTheta_ = config.ropeTheta;
        model->architectureName_ = "llama";
        model->eosId_ = eosTokenIdFromGguf(*model->tokenizer_, *gguf);

        // Reuse the already-parsed `gguf` for the initial graph build (no need to
        // re-read the file we just read above); only later regrowth, via
        // rebuildGraph_, re-reads from disk -- see buildOrLoadGgufDecodeGraph()'s doc
        // comment for why it doesn't retain a parsed copy for that purpose instead.
        model->currentCapacity_ = initialCapacityFor(maxSequenceLength);
        std::string initialCachePath = ggufGraphCachePath(path, model->currentCapacity_);
        bool loadedFromCache = false;
        if (isGraphCacheFresh(initialCachePath, path, path))
        {
            try
            {
                cnn::GraphCacheResult cached = cnn::loadGraphFromFile(context, initialCachePath);
                model->graph_.graph = cached.graph;
                model->graph_.inputs = std::move(cached.inputs);
                model->graph_.outputs = std::move(cached.outputs);
                loadedFromCache = true;
            }
            catch (const std::exception &)
            {
                // Corrupt cache: fall back to building fresh.
            }
        }
        if (!loadedFromCache)
        {
            GgufWeightsAdapter weights(*gguf);
            model->graph_ = buildLlamaDecodeGraph(context, weights, config, model->currentCapacity_);
            try
            {
                cnn::saveGraphToFile(model->graph_.serializedGraph, initialCachePath);
            }
            catch (const std::exception &)
            {
                // Best-effort: skip caching on read-only directories.
            }
            // See buildOrLoadGgufDecodeGraph()'s comment: drop the now-unneeded second
            // copy of every weight byte before it sits on model->graph_ indefinitely.
            model->graph_.serializedGraph.clear();
            model->graph_.serializedGraph.shrink_to_fit();
        }
        model->rebuildGraph_ = [context, path, config](std::int64_t capacity) {
            return buildOrLoadGgufDecodeGraph(context, path, config, capacity);
        };

        return model;
    }

    std::unique_ptr<Model> Model::load(std::shared_ptr<cnn::Context> context, const std::string &directory,
                                        std::int64_t maxSequenceLength)
    {
        if (maxSequenceLength <= 0)
        {
            throw std::runtime_error("campello_llm: maxSequenceLength must be positive");
        }

        std::error_code ec;
        std::filesystem::file_status status = std::filesystem::status(directory, ec);
        if (ec || !std::filesystem::exists(status))
        {
            throw std::runtime_error("campello_llm: model path does not exist: '" + directory + "'");
        }

        if (std::filesystem::is_regular_file(status))
        {
            if (directory.size() >= 5 && directory.compare(directory.size() - 5, 5, ".gguf") == 0)
            {
                return loadFromGgufFile(context, directory, maxSequenceLength);
            }
            throw std::runtime_error("campello_llm: unsupported model file format: '" + directory + "'");
        }

        return loadFromSafetensorsDirectory(context, directory, maxSequenceLength);
    }

    std::string Model::generate(const std::string &prompt, const GenerationConfig &config,
                                 const std::function<void(const std::string &)> &onToken)
    {
        std::vector<std::int32_t> allIds = tokenizer_->encode(prompt, true);
        if (static_cast<std::int64_t>(allIds.size()) >= maxSequenceLength_)
        {
            throw std::runtime_error("campello_llm: prompt (plus BOS) already reaches maxSequenceLength, "
                                      "leaving no room to generate");
        }
        std::size_t promptLength = allIds.size();

        // KV-cache/tensor capacity backing this call. Starts at currentCapacity_ (a
        // Model-lifetime high-water mark -- whatever an earlier generate() call, or
        // Model::load() itself, already grew into) and doubles as needed, capped at
        // maxSequenceLength_, instead of always paying for a maxSequenceLength_-sized
        // KV-cache. `0` is a not-yet-allocated sentinel, forcing reallocateTo()'s
        // first call to treat it as a fresh allocation rather than a regrowth.
        std::int64_t capacity = 0;
        std::vector<std::vector<float>> kCache(static_cast<std::size_t>(numLayers_));
        std::vector<std::vector<float>> vCache(static_cast<std::size_t>(numLayers_));
        std::vector<std::shared_ptr<cnn::Tensor>> kCacheInTensor(static_cast<std::size_t>(numLayers_));
        std::vector<std::shared_ptr<cnn::Tensor>> vCacheInTensor(static_cast<std::size_t>(numLayers_));
        std::shared_ptr<cnn::Tensor> attnMaskTensor;
        std::vector<float> maskBuffer;

        // (Re)allocates the KV-cache host buffers/device tensors and attn_mask tensor
        // at `newCapacity`, copying forward any existing cache contents (zero-padding
        // the newly added slots -- attn_mask masks them out regardless, same as the
        // original zero-init). Rebuilds (or on-disk-cache-loads) the compiled decode
        // graph too, whenever `newCapacity` differs from the Model's persisted
        // currentCapacity_.
        auto reallocateTo = [&](std::int64_t newCapacity) {
            std::int64_t oldCapacity = capacity;

            if (newCapacity != currentCapacity_)
            {
                graph_ = rebuildGraph_(newCapacity);
                currentCapacity_ = newCapacity;
            }

            std::size_t newElemsPerLayer = static_cast<std::size_t>(numKeyValueHeads_ * newCapacity * headDim_);
            for (std::int64_t layer = 0; layer < numLayers_; ++layer)
            {
                std::size_t li = static_cast<std::size_t>(layer);
                std::vector<float> newK(newElemsPerLayer, 0.0f);
                std::vector<float> newV(newElemsPerLayer, 0.0f);
                if (oldCapacity > 0)
                {
                    for (std::int64_t h = 0; h < numKeyValueHeads_; ++h)
                    {
                        std::copy_n(kCache[li].begin() + static_cast<std::ptrdiff_t>(h * oldCapacity * headDim_),
                                    static_cast<std::size_t>(oldCapacity * headDim_),
                                    newK.begin() + static_cast<std::ptrdiff_t>(h * newCapacity * headDim_));
                        std::copy_n(vCache[li].begin() + static_cast<std::ptrdiff_t>(h * oldCapacity * headDim_),
                                    static_cast<std::size_t>(oldCapacity * headDim_),
                                    newV.begin() + static_cast<std::ptrdiff_t>(h * newCapacity * headDim_));
                    }
                }
                kCache[li] = std::move(newK);
                vCache[li] = std::move(newV);

                kCacheInTensor[li] = context_->createTensor(
                    {cnn::DataType::Float32, {numKeyValueHeads_, newCapacity, headDim_}, false, true});
                vCacheInTensor[li] = context_->createTensor(
                    {cnn::DataType::Float32, {numKeyValueHeads_, newCapacity, headDim_}, false, true});
            }

            attnMaskTensor = context_->createTensor({cnn::DataType::Float32, {1, newCapacity + 1}, false, true});
            maskBuffer.assign(static_cast<std::size_t>(newCapacity + 1), 0.0f);
            capacity = newCapacity;
        };

        // Grows capacity (doubling from whatever it currently is, capped at
        // maxSequenceLength_) until cache slot `position` fits, reallocating only when
        // it actually needs to -- called before every dispatchStep(), so the very
        // first call (capacity == 0) always reallocates at currentCapacity_, and every
        // later call is a no-op unless `position` has walked past the current capacity.
        auto ensureCapacityFor = [&](std::int64_t position) {
            if (capacity > position)
            {
                return;
            }
            std::int64_t newCapacity = capacity > 0 ? capacity : currentCapacity_;
            while (newCapacity <= position && newCapacity < maxSequenceLength_)
            {
                newCapacity *= 2;
            }
            reallocateTo(std::min(newCapacity, maxSequenceLength_));
        };

        auto inputIdsTensor = context_->createTensor({cnn::DataType::Int32, {1}, false, true});
        auto ropeCosTensor = context_->createTensor({cnn::DataType::Float32, {1, headDim_}, false, true});
        auto ropeSinTensor = context_->createTensor({cnn::DataType::Float32, {1, headDim_}, false, true});
        auto logitsTensor = context_->createTensor({cnn::DataType::Float32, {1, vocabSize_}, true, false});

        std::vector<std::shared_ptr<cnn::Tensor>> kNewTensor(static_cast<std::size_t>(numLayers_));
        std::vector<std::shared_ptr<cnn::Tensor>> vNewTensor(static_cast<std::size_t>(numLayers_));
        for (std::int64_t layer = 0; layer < numLayers_; ++layer)
        {
            kNewTensor[static_cast<std::size_t>(layer)] = context_->createTensor(
                {cnn::DataType::Float32, {numKeyValueHeads_, 1, headDim_}, true, false});
            vNewTensor[static_cast<std::size_t>(layer)] = context_->createTensor(
                {cnn::DataType::Float32, {numKeyValueHeads_, 1, headDim_}, true, false});
        }

        std::vector<float> logitsBuffer(static_cast<std::size_t>(vocabSize_));
        std::vector<float> newKvBuffer(static_cast<std::size_t>(numKeyValueHeads_ * headDim_));

        // Dispatches the decode graph for `tokenId` at absolute sequence `position`,
        // refreshing logitsBuffer with that position's predictions and folding the
        // graph's k_new/v_new outputs into kCache/vCache at slot `position` for the
        // next call -- used for every prompt token (filling the cache) and every
        // generated token alike, since there is only the one seqLen=1 graph (see
        // CLAUDE.md / architecture.hpp's buildLlamaDecodeGraph()). Callers must call
        // ensureCapacityFor(position) first so `capacity`/the cache tensors already
        // fit `position`.
        auto dispatchStep = [&](std::int32_t tokenId, std::int64_t position) {
            inputIdsTensor->write(&tokenId, sizeof(tokenId));

            auto [cosRow, sinRow] = internal::ropeCosSinForPosition(position, headDim_, ropeTheta_);
            ropeCosTensor->write(cosRow.data(), cosRow.size() * sizeof(float));
            ropeSinTensor->write(sinRow.data(), sinRow.size() * sizeof(float));

            for (std::int64_t i = 0; i < capacity; ++i)
            {
                maskBuffer[static_cast<std::size_t>(i)] = (i < position) ? 0.0f : -1e9f;
            }
            maskBuffer[static_cast<std::size_t>(capacity)] = 0.0f; // current token always attends to itself
            attnMaskTensor->write(maskBuffer.data(), maskBuffer.size() * sizeof(float));

            std::unordered_map<std::string, std::shared_ptr<cnn::Tensor>> inputs = {
                {"input_ids", inputIdsTensor},
                {"rope_cos", ropeCosTensor},
                {"rope_sin", ropeSinTensor},
                {"attn_mask", attnMaskTensor},
            };
            std::unordered_map<std::string, std::shared_ptr<cnn::Tensor>> outputs = {{"logits", logitsTensor}};
            for (std::int64_t layer = 0; layer < numLayers_; ++layer)
            {
                std::size_t li = static_cast<std::size_t>(layer);
                kCacheInTensor[li]->write(kCache[li].data(), kCache[li].size() * sizeof(float));
                vCacheInTensor[li]->write(vCache[li].data(), vCache[li].size() * sizeof(float));
                inputs[llamaKeyCacheInputName(layer)] = kCacheInTensor[li];
                inputs[llamaValueCacheInputName(layer)] = vCacheInTensor[li];
                outputs[llamaKeyCacheOutputName(layer)] = kNewTensor[li];
                outputs[llamaValueCacheOutputName(layer)] = vNewTensor[li];
            }

            auto fence = context_->dispatch(*graph_.graph, inputs, outputs);
            fence->wait();

            logitsTensor->read(logitsBuffer.data(), logitsBuffer.size() * sizeof(float));
            for (std::int64_t layer = 0; layer < numLayers_; ++layer)
            {
                std::size_t li = static_cast<std::size_t>(layer);
                kNewTensor[li]->read(newKvBuffer.data(), newKvBuffer.size() * sizeof(float));
                for (std::int64_t h = 0; h < numKeyValueHeads_; ++h)
                {
                    std::copy_n(newKvBuffer.begin() + static_cast<std::ptrdiff_t>(h * headDim_),
                                static_cast<std::size_t>(headDim_),
                                kCache[li].begin() +
                                    static_cast<std::ptrdiff_t>(h * capacity * headDim_ + position * headDim_));
                }
                vNewTensor[li]->read(newKvBuffer.data(), newKvBuffer.size() * sizeof(float));
                for (std::int64_t h = 0; h < numKeyValueHeads_; ++h)
                {
                    std::copy_n(newKvBuffer.begin() + static_cast<std::ptrdiff_t>(h * headDim_),
                                static_cast<std::size_t>(headDim_),
                                vCache[li].begin() +
                                    static_cast<std::ptrdiff_t>(h * capacity * headDim_ + position * headDim_));
                }
            }
        };

        // Phase A: prefill -- one dispatch per prompt token, filling the cache.
        // logitsBuffer ends up holding the prediction for the token right after the
        // prompt, ready for phase B's first sample.
        for (std::int64_t pos = 0; pos < static_cast<std::int64_t>(promptLength); ++pos)
        {
            ensureCapacityFor(pos);
            dispatchStep(allIds[static_cast<std::size_t>(pos)], pos);
        }

        // Phase B: decode -- sample from the previous dispatch's logits, then (if
        // continuing) dispatch once for the newly sampled token to extend the cache
        // and obtain the next position's logits.
        std::mt19937 rng(std::random_device{}());
        std::string emittedSoFar;
        std::int64_t nextPos = static_cast<std::int64_t>(promptLength);

        for (std::uint32_t generated = 0; generated < config.maxTokens; ++generated)
        {
            std::int32_t nextId = sampleNextToken(logitsBuffer.data(), vocabSize_, config, rng);
            allIds.push_back(nextId);

            std::vector<std::int32_t> generatedSoFar(allIds.begin() + static_cast<std::ptrdiff_t>(promptLength),
                                                        allIds.end());
            std::string decodedSoFar = tokenizer_->decode(generatedSoFar);

            if (onToken && decodedSoFar.size() > emittedSoFar.size())
            {
                onToken(decodedSoFar.substr(emittedSoFar.size()));
            }
            emittedSoFar = decodedSoFar;

            bool stop = (eosId_ >= 0 && nextId == eosId_);
            for (const std::string &stopSeq : config.stopSequences)
            {
                if (!stopSeq.empty() && decodedSoFar.size() >= stopSeq.size() &&
                    decodedSoFar.compare(decodedSoFar.size() - stopSeq.size(), stopSeq.size(), stopSeq) == 0)
                {
                    stop = true;
                    break;
                }
            }
            if (stop || static_cast<std::int64_t>(allIds.size()) >= maxSequenceLength_)
            {
                break;
            }

            ensureCapacityFor(nextPos);
            dispatchStep(nextId, nextPos);
            ++nextPos;
        }

        std::vector<std::int32_t> generatedIds(allIds.begin() + static_cast<std::ptrdiff_t>(promptLength),
                                                 allIds.end());
        return tokenizer_->decode(generatedIds);
    }

} // namespace systems::leal::campello_llm
