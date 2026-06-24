#include <gtest/gtest.h>

#include <cmath>
#include <cstdint>
#include <string>
#include <unordered_map>
#include <vector>

#include <campello_llm/architecture.hpp>
#include <campello_llm/safetensors_reader.hpp>
#include <campello_nn/context.hpp>

using namespace systems::leal::campello_llm;
namespace cnn = systems::leal::campello_nn;

namespace
{
    std::string fixturePath(const std::string &name)
    {
        return std::string(CAMPELLO_LLM_TEST_FIXTURES_DIR) + "/" + name;
    }

    // Mirrors tests/fixtures/generate_llama_test_fixture.py's tiny config exactly.
    LlamaConfig tinyLlamaTestConfig()
    {
        LlamaConfig config;
        config.vocabSize = 6;
        config.hiddenSize = 8;
        config.numLayers = 2;
        config.numAttentionHeads = 4;
        config.numKeyValueHeads = 2;
        config.intermediateSize = 10;
        config.rmsNormEps = 1e-5f;
        config.ropeTheta = 10000.0f;
        return config;
    }

    // Same GPT-NeoX/LLaMA "rotate-half" RoPE formula buildLlamaGraph()'s baked-in
    // table uses (verified there against the real huggingface/tokenizers-adjacent
    // implementations -- see CLAUDE.md), computed here for a single absolute
    // position to drive buildLlamaDecodeGraph()'s runtime rope_cos/rope_sin inputs.
    std::pair<std::vector<float>, std::vector<float>> ropeCosSinForPosition(std::int64_t position,
                                                                             std::int64_t headDim, float theta)
    {
        std::int64_t half = headDim / 2;
        std::vector<float> cosRow(static_cast<std::size_t>(headDim));
        std::vector<float> sinRow(static_cast<std::size_t>(headDim));
        for (std::int64_t i = 0; i < half; ++i)
        {
            float freq = std::pow(theta, -static_cast<float>(2 * i) / static_cast<float>(headDim));
            float angle = static_cast<float>(position) * freq;
            cosRow[static_cast<std::size_t>(i)] = cosRow[static_cast<std::size_t>(i + half)] = std::cos(angle);
            sinRow[static_cast<std::size_t>(i)] = sinRow[static_cast<std::size_t>(i + half)] = std::sin(angle);
        }
        return {cosRow, sinRow};
    }
} // namespace

// Expected logits independently computed by a from-scratch numpy reference
// implementation (tests/fixtures/generate_llama_test_fixture.py) -- not derived
// from this project's own code, per TODO.md Phase 3's testing standard.
TEST(LlamaArchitecture, MatchesNumpyReferenceForwardPass)
{
    constexpr std::int64_t seqLen = 3;
    constexpr std::int64_t vocabSize = 6;

    auto weights = loadSafetensorsFromFile(fixturePath("llama_test_weights.safetensors"));
    auto context = cnn::Context::create({cnn::DeviceType::Cpu});

    ArchitectureGraphResult result = buildLlamaGraph(context, *weights, tinyLlamaTestConfig(), seqLen);

    auto inputIds = context->createTensor({cnn::DataType::Int32, {seqLen}, false, true});
    auto logits = context->createTensor({cnn::DataType::Float32, {seqLen, vocabSize}, true, false});

    std::int32_t ids[3] = {1, 4, 2};
    inputIds->write(ids, sizeof(ids));

    auto fence = context->dispatch(*result.graph, {{"input_ids", inputIds}}, {{"logits", logits}});
    fence->wait();

    float actual[seqLen * vocabSize];
    logits->read(actual, sizeof(actual));

    float expected[] = {0.432659f,  -0.665152f, 0.119578f,  -0.733880f, -0.601274f, 0.357684f,
                         0.142916f,  -0.451302f, 0.009399f,  -0.163286f, -0.234247f, 0.726573f,
                         -0.044811f, -0.615250f, -0.159470f, -0.534479f, -0.347264f, 0.524364f};

    for (std::size_t i = 0; i < seqLen * vocabSize; ++i)
    {
        EXPECT_NEAR(actual[i], expected[i], 1e-3f) << "mismatch at flat index " << i;
    }
}

// Real LLaMA/TinyLlama checkpoints ship BF16 (confirmed against TinyLlama's real
// config.json: "torch_dtype": "bfloat16"), not Float32. This fixture
// (tests/fixtures/generate_llama_bf16_test_fixture.py) is the exact same tiny
// config, with every weight rounded to BF16 precision and stored as actual BF16
// bytes -- and the expected logits were computed from those *same BF16-rounded*
// weights, so this is an apples-to-apples check of the BF16->F32 decode path,
// not just a tolerance-widened version of the F32 test above.
TEST(LlamaArchitecture, DecodesBf16Weights)
{
    constexpr std::int64_t seqLen = 3;
    constexpr std::int64_t vocabSize = 6;

    auto weights = loadSafetensorsFromFile(fixturePath("llama_test_weights_bf16.safetensors"));
    auto context = cnn::Context::create({cnn::DeviceType::Cpu});

    ArchitectureGraphResult result = buildLlamaGraph(context, *weights, tinyLlamaTestConfig(), seqLen);

    auto inputIds = context->createTensor({cnn::DataType::Int32, {seqLen}, false, true});
    auto logits = context->createTensor({cnn::DataType::Float32, {seqLen, vocabSize}, true, false});

    std::int32_t ids[3] = {1, 4, 2};
    inputIds->write(ids, sizeof(ids));

    auto fence = context->dispatch(*result.graph, {{"input_ids", inputIds}}, {{"logits", logits}});
    fence->wait();

    float actual[seqLen * vocabSize];
    logits->read(actual, sizeof(actual));

    float expected[] = {0.432094f,  -0.660537f, 0.118302f,  -0.729922f, -0.594396f, 0.354659f,
                         0.144202f,  -0.451774f, 0.006725f,  -0.166201f, -0.232141f, 0.718220f,
                         -0.043655f, -0.610142f, -0.160226f, -0.530805f, -0.340267f, 0.519917f};

    for (std::size_t i = 0; i < seqLen * vocabSize; ++i)
    {
        EXPECT_NEAR(actual[i], expected[i], 1e-3f) << "mismatch at flat index " << i;
    }
}

TEST(LlamaArchitecture, ThrowsOnMissingWeightTensor)
{
    auto weights = loadSafetensorsFromFile(fixturePath("llama_test_weights.safetensors"));
    auto context = cnn::Context::create({cnn::DeviceType::Cpu});

    LlamaConfig config = tinyLlamaTestConfig();
    config.numLayers = 99; // far beyond what the fixture actually has weights for

    EXPECT_THROW(buildLlamaGraph(context, *weights, config, 3), std::runtime_error);
}

// The standard KV-cache correctness check (TODO.md Phase 4): decoding token by
// token through buildLlamaDecodeGraph()'s explicit cache must produce the exact
// same per-position logits as buildLlamaGraph()'s one-shot batched prefill over
// the equivalent full sequence -- both exercised here against the same fixture/
// ids MatchesNumpyReferenceForwardPass already validated, so `expected` is the
// same numpy-derived array, just compared one row at a time.
TEST(LlamaArchitecture, DecodeGraphMatchesBatchedPrefillPerPosition)
{
    constexpr std::int64_t maxSequenceLength = 3; // exactly fits the 3-token sequence below
    constexpr std::int64_t vocabSize = 6;
    constexpr std::int64_t numKeyValueHeads = 2;
    constexpr std::int64_t headDim = 2; // hiddenSize(8) / numAttentionHeads(4)

    auto weights = loadSafetensorsFromFile(fixturePath("llama_test_weights.safetensors"));
    auto context = cnn::Context::create({cnn::DeviceType::Cpu});
    LlamaConfig config = tinyLlamaTestConfig();

    ArchitectureGraphResult decodeGraph = buildLlamaDecodeGraph(context, *weights, config, maxSequenceLength);

    std::int32_t ids[3] = {1, 4, 2};
    float expected[] = {0.432659f,  -0.665152f, 0.119578f,  -0.733880f, -0.601274f, 0.357684f,
                         0.142916f,  -0.451302f, 0.009399f,  -0.163286f, -0.234247f, 0.726573f,
                         -0.044811f, -0.615250f, -0.159470f, -0.534479f, -0.347264f, 0.524364f};

    std::vector<std::vector<float>> kCache(static_cast<std::size_t>(config.numLayers),
                                            std::vector<float>(numKeyValueHeads * maxSequenceLength * headDim, 0.0f));
    std::vector<std::vector<float>> vCache(static_cast<std::size_t>(config.numLayers),
                                            std::vector<float>(numKeyValueHeads * maxSequenceLength * headDim, 0.0f));

    auto inputIdsTensor = context->createTensor({cnn::DataType::Int32, {1}, false, true});
    auto ropeCosTensor = context->createTensor({cnn::DataType::Float32, {1, headDim}, false, true});
    auto ropeSinTensor = context->createTensor({cnn::DataType::Float32, {1, headDim}, false, true});
    auto attnMaskTensor = context->createTensor({cnn::DataType::Float32, {1, maxSequenceLength + 1}, false, true});
    auto logitsTensor = context->createTensor({cnn::DataType::Float32, {1, vocabSize}, true, false});

    std::vector<std::shared_ptr<cnn::Tensor>> kCacheInTensor(static_cast<std::size_t>(config.numLayers));
    std::vector<std::shared_ptr<cnn::Tensor>> vCacheInTensor(static_cast<std::size_t>(config.numLayers));
    std::vector<std::shared_ptr<cnn::Tensor>> kNewTensor(static_cast<std::size_t>(config.numLayers));
    std::vector<std::shared_ptr<cnn::Tensor>> vNewTensor(static_cast<std::size_t>(config.numLayers));
    for (std::int64_t layer = 0; layer < config.numLayers; ++layer)
    {
        std::size_t li = static_cast<std::size_t>(layer);
        kCacheInTensor[li] = context->createTensor(
            {cnn::DataType::Float32, {numKeyValueHeads, maxSequenceLength, headDim}, false, true});
        vCacheInTensor[li] = context->createTensor(
            {cnn::DataType::Float32, {numKeyValueHeads, maxSequenceLength, headDim}, false, true});
        kNewTensor[li] = context->createTensor({cnn::DataType::Float32, {numKeyValueHeads, 1, headDim}, true, false});
        vNewTensor[li] = context->createTensor({cnn::DataType::Float32, {numKeyValueHeads, 1, headDim}, true, false});
    }

    std::vector<float> newKvBuffer(static_cast<std::size_t>(numKeyValueHeads * headDim));
    std::vector<float> maskBuffer(static_cast<std::size_t>(maxSequenceLength + 1));

    for (std::int64_t pos = 0; pos < maxSequenceLength; ++pos)
    {
        inputIdsTensor->write(&ids[pos], sizeof(ids[pos]));

        auto [cosRow, sinRow] = ropeCosSinForPosition(pos, headDim, config.ropeTheta);
        ropeCosTensor->write(cosRow.data(), cosRow.size() * sizeof(float));
        ropeSinTensor->write(sinRow.data(), sinRow.size() * sizeof(float));

        for (std::int64_t i = 0; i < maxSequenceLength; ++i)
        {
            maskBuffer[static_cast<std::size_t>(i)] = (i < pos) ? 0.0f : -1e9f;
        }
        maskBuffer[static_cast<std::size_t>(maxSequenceLength)] = 0.0f;
        attnMaskTensor->write(maskBuffer.data(), maskBuffer.size() * sizeof(float));

        std::unordered_map<std::string, std::shared_ptr<cnn::Tensor>> inputs = {
            {"input_ids", inputIdsTensor},
            {"rope_cos", ropeCosTensor},
            {"rope_sin", ropeSinTensor},
            {"attn_mask", attnMaskTensor},
        };
        std::unordered_map<std::string, std::shared_ptr<cnn::Tensor>> outputs = {{"logits", logitsTensor}};
        for (std::int64_t layer = 0; layer < config.numLayers; ++layer)
        {
            std::size_t li = static_cast<std::size_t>(layer);
            kCacheInTensor[li]->write(kCache[li].data(), kCache[li].size() * sizeof(float));
            vCacheInTensor[li]->write(vCache[li].data(), vCache[li].size() * sizeof(float));
            inputs[llamaKeyCacheInputName(layer)] = kCacheInTensor[li];
            inputs[llamaValueCacheInputName(layer)] = vCacheInTensor[li];
            outputs[llamaKeyCacheOutputName(layer)] = kNewTensor[li];
            outputs[llamaValueCacheOutputName(layer)] = vNewTensor[li];
        }

        auto fence = context->dispatch(*decodeGraph.graph, inputs, outputs);
        fence->wait();

        float logitsRow[vocabSize];
        logitsTensor->read(logitsRow, sizeof(logitsRow));
        for (std::int64_t i = 0; i < vocabSize; ++i)
        {
            EXPECT_NEAR(logitsRow[i], expected[pos * vocabSize + i], 1e-3f)
                << "mismatch at position " << pos << ", vocab index " << i;
        }

        for (std::int64_t layer = 0; layer < config.numLayers; ++layer)
        {
            std::size_t li = static_cast<std::size_t>(layer);
            kNewTensor[li]->read(newKvBuffer.data(), newKvBuffer.size() * sizeof(float));
            for (std::int64_t h = 0; h < numKeyValueHeads; ++h)
            {
                std::copy_n(newKvBuffer.begin() + static_cast<std::ptrdiff_t>(h * headDim),
                            static_cast<std::size_t>(headDim),
                            kCache[li].begin() +
                                static_cast<std::ptrdiff_t>(h * maxSequenceLength * headDim + pos * headDim));
            }
            vNewTensor[li]->read(newKvBuffer.data(), newKvBuffer.size() * sizeof(float));
            for (std::int64_t h = 0; h < numKeyValueHeads; ++h)
            {
                std::copy_n(newKvBuffer.begin() + static_cast<std::ptrdiff_t>(h * headDim),
                            static_cast<std::size_t>(headDim),
                            vCache[li].begin() +
                                static_cast<std::ptrdiff_t>(h * maxSequenceLength * headDim + pos * headDim));
            }
        }
    }
}
