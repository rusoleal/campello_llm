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

#include <campello_llm/safetensors_reader.hpp>
#include <campello_llm/tokenizer_config.hpp>
#include <campello_nn/graph_cache.hpp>

#include "architecture/weight_loading.hpp"

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

    std::unique_ptr<Model> Model::load(std::shared_ptr<cnn::Context> context, const std::string &directory,
                                        std::int64_t maxSequenceLength)
    {
        if (maxSequenceLength <= 0)
        {
            throw std::runtime_error("campello_llm: maxSequenceLength must be positive");
        }

        LlamaConfig config = loadLlamaConfigFromFile(joinPath(directory, "config.json"));
        auto tokenizer = loadTokenizerFromFile(joinPath(directory, "tokenizer.json"));
        SpecialTokenStrings specialTokens = loadSpecialTokenStringsFromFile(joinPath(directory, "tokenizer_config.json"));

        auto model = std::unique_ptr<Model>(new Model());
        model->context_ = context;
        model->tokenizer_ = std::move(tokenizer);
        model->maxSequenceLength_ = maxSequenceLength;
        model->vocabSize_ = config.vocabSize;
        model->numLayers_ = config.numLayers;
        model->numKeyValueHeads_ = config.numKeyValueHeads;
        model->headDim_ = config.hiddenSize / config.numAttentionHeads;
        model->ropeTheta_ = config.ropeTheta;

        if (!specialTokens.eosToken.empty())
        {
            if (auto id = model->tokenizer_->tokenToId(specialTokens.eosToken))
            {
                model->eosId_ = *id;
            }
        }

        std::string weightsPath = joinPath(directory, "model.safetensors");
        std::string configPath = joinPath(directory, "config.json");
        std::string cachePath = graphCachePath(directory, maxSequenceLength);

        bool loadedFromCache = false;
        if (isGraphCacheFresh(cachePath, weightsPath, configPath))
        {
            try
            {
                cnn::GraphCacheResult cached = cnn::loadGraphFromFile(context, cachePath);
                model->graph_.graph = cached.graph;
                model->graph_.inputs = std::move(cached.inputs);
                model->graph_.outputs = std::move(cached.outputs);
                loadedFromCache = true;
            }
            catch (const std::exception &)
            {
                // Corrupt/truncated/incompatible-version cache (campello_nn's
                // loadGraphFromFile() magic/version-checks the bytes) -- fall back to
                // building fresh below rather than failing Model::load() over a stale
                // or damaged cache file.
            }
        }

        if (!loadedFromCache)
        {
            // Weights only need to live long enough to bind every constant() into the
            // graph -- buildLlamaDecodeGraph() copies every tensor's bytes immediately
            // (see GraphBuilder::constant()), so the multi-gigabyte WeightsFile buffer
            // can be freed right after this block instead of held for the Model's
            // whole lifetime.
            auto weights = loadSafetensorsFromFile(weightsPath);
            model->graph_ = buildLlamaDecodeGraph(context, *weights, config, maxSequenceLength);

            try
            {
                cnn::saveGraphToFile(model->graph_.serializedGraph, cachePath);
            }
            catch (const std::exception &)
            {
                // Best-effort: a read-only model directory shouldn't fail loading,
                // just skip caching for next time.
            }
        }

        return model;
    }

    std::string Model::generate(const std::string &prompt, const GenerationConfig &config,
                                 const std::function<void(const std::string &)> &onToken) const
    {
        std::vector<std::int32_t> allIds = tokenizer_->encode(prompt, true);
        if (static_cast<std::int64_t>(allIds.size()) >= maxSequenceLength_)
        {
            throw std::runtime_error("campello_llm: prompt (plus BOS) already reaches maxSequenceLength, "
                                      "leaving no room to generate");
        }
        std::size_t promptLength = allIds.size();

        // Host-held KV-cache: one [numKeyValueHeads, maxSequenceLength, headDim]
        // buffer per layer, per K/V. Zero-initialized so cache slots not yet filled
        // contribute a deterministic (and harmless, since attn_mask masks them out
        // regardless) value rather than uninitialized garbage.
        std::size_t cacheElemsPerLayer =
            static_cast<std::size_t>(numKeyValueHeads_ * maxSequenceLength_ * headDim_);
        std::vector<std::vector<float>> kCache(static_cast<std::size_t>(numLayers_),
                                                std::vector<float>(cacheElemsPerLayer, 0.0f));
        std::vector<std::vector<float>> vCache(static_cast<std::size_t>(numLayers_),
                                                std::vector<float>(cacheElemsPerLayer, 0.0f));

        auto inputIdsTensor = context_->createTensor({cnn::DataType::Int32, {1}, false, true});
        auto ropeCosTensor = context_->createTensor({cnn::DataType::Float32, {1, headDim_}, false, true});
        auto ropeSinTensor = context_->createTensor({cnn::DataType::Float32, {1, headDim_}, false, true});
        auto attnMaskTensor =
            context_->createTensor({cnn::DataType::Float32, {1, maxSequenceLength_ + 1}, false, true});
        auto logitsTensor = context_->createTensor({cnn::DataType::Float32, {1, vocabSize_}, true, false});

        std::vector<std::shared_ptr<cnn::Tensor>> kCacheInTensor(static_cast<std::size_t>(numLayers_));
        std::vector<std::shared_ptr<cnn::Tensor>> vCacheInTensor(static_cast<std::size_t>(numLayers_));
        std::vector<std::shared_ptr<cnn::Tensor>> kNewTensor(static_cast<std::size_t>(numLayers_));
        std::vector<std::shared_ptr<cnn::Tensor>> vNewTensor(static_cast<std::size_t>(numLayers_));
        for (std::int64_t layer = 0; layer < numLayers_; ++layer)
        {
            kCacheInTensor[static_cast<std::size_t>(layer)] = context_->createTensor(
                {cnn::DataType::Float32, {numKeyValueHeads_, maxSequenceLength_, headDim_}, false, true});
            vCacheInTensor[static_cast<std::size_t>(layer)] = context_->createTensor(
                {cnn::DataType::Float32, {numKeyValueHeads_, maxSequenceLength_, headDim_}, false, true});
            kNewTensor[static_cast<std::size_t>(layer)] = context_->createTensor(
                {cnn::DataType::Float32, {numKeyValueHeads_, 1, headDim_}, true, false});
            vNewTensor[static_cast<std::size_t>(layer)] = context_->createTensor(
                {cnn::DataType::Float32, {numKeyValueHeads_, 1, headDim_}, true, false});
        }

        std::vector<float> logitsBuffer(static_cast<std::size_t>(vocabSize_));
        std::vector<float> newKvBuffer(static_cast<std::size_t>(numKeyValueHeads_ * headDim_));
        std::vector<float> maskBuffer(static_cast<std::size_t>(maxSequenceLength_ + 1));

        // Dispatches the decode graph for `tokenId` at absolute sequence `position`,
        // refreshing logitsBuffer with that position's predictions and folding the
        // graph's k_new/v_new outputs into kCache/vCache at slot `position` for the
        // next call -- used for every prompt token (filling the cache) and every
        // generated token alike, since there is only the one seqLen=1 graph (see
        // CLAUDE.md / architecture.hpp's buildLlamaDecodeGraph()).
        auto dispatchStep = [&](std::int32_t tokenId, std::int64_t position) {
            inputIdsTensor->write(&tokenId, sizeof(tokenId));

            auto [cosRow, sinRow] = internal::ropeCosSinForPosition(position, headDim_, ropeTheta_);
            ropeCosTensor->write(cosRow.data(), cosRow.size() * sizeof(float));
            ropeSinTensor->write(sinRow.data(), sinRow.size() * sizeof(float));

            for (std::int64_t i = 0; i < maxSequenceLength_; ++i)
            {
                maskBuffer[static_cast<std::size_t>(i)] = (i < position) ? 0.0f : -1e9f;
            }
            maskBuffer[static_cast<std::size_t>(maxSequenceLength_)] = 0.0f; // current token always attends to itself
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
                                    static_cast<std::ptrdiff_t>(h * maxSequenceLength_ * headDim_ + position * headDim_));
                }
                vNewTensor[li]->read(newKvBuffer.data(), newKvBuffer.size() * sizeof(float));
                for (std::int64_t h = 0; h < numKeyValueHeads_; ++h)
                {
                    std::copy_n(newKvBuffer.begin() + static_cast<std::ptrdiff_t>(h * headDim_),
                                static_cast<std::size_t>(headDim_),
                                vCache[li].begin() +
                                    static_cast<std::ptrdiff_t>(h * maxSequenceLength_ * headDim_ + position * headDim_));
                }
            }
        };

        // Phase A: prefill -- one dispatch per prompt token, filling the cache.
        // logitsBuffer ends up holding the prediction for the token right after the
        // prompt, ready for phase B's first sample.
        for (std::int64_t pos = 0; pos < static_cast<std::int64_t>(promptLength); ++pos)
        {
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

            dispatchStep(nextId, nextPos);
            ++nextPos;
        }

        std::vector<std::int32_t> generatedIds(allIds.begin() + static_cast<std::ptrdiff_t>(promptLength),
                                                 allIds.end());
        return tokenizer_->decode(generatedIds);
    }

} // namespace systems::leal::campello_llm
