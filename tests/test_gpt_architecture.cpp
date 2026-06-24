#include <gtest/gtest.h>

#include <cstdint>
#include <string>
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

    // Mirrors tests/fixtures/generate_gpt_test_fixture.py's tiny config exactly.
    GptConfig tinyGptTestConfig()
    {
        GptConfig config;
        config.vocabSize = 7;
        config.hiddenSize = 8;
        config.numLayers = 2;
        config.numAttentionHeads = 2;
        config.intermediateSize = 12;
        config.layerNormEps = 1e-5f;
        config.maxPositionEmbeddings = 4;
        return config;
    }
} // namespace

// Expected logits independently computed by a from-scratch numpy reference
// implementation (tests/fixtures/generate_gpt_test_fixture.py) -- not derived
// from this project's own code, per TODO.md Phase 3's testing standard. This is
// a structurally different architecture from LLaMA (LayerNorm not RMSNorm, exact
// GELU not SwiGLU, learned positional embeddings not RoPE, plain MHA not GQA,
// [in,out] Conv1D weights not [out,in] nn.Linear, tied lm_head) -- deliberately,
// to validate the registry generalizes rather than being accidentally
// LLaMA-shaped.
TEST(GptArchitecture, MatchesNumpyReferenceForwardPass)
{
    constexpr std::int64_t seqLen = 3;
    constexpr std::int64_t vocabSize = 7;

    auto weights = loadSafetensorsFromFile(fixturePath("gpt_test_weights.safetensors"));
    auto context = cnn::Context::create({cnn::DeviceType::Cpu});

    ArchitectureGraphResult result = buildGptGraph(context, *weights, tinyGptTestConfig(), seqLen);

    auto inputIds = context->createTensor({cnn::DataType::Int32, {seqLen}, false, true});
    auto logits = context->createTensor({cnn::DataType::Float32, {seqLen, vocabSize}, true, false});

    std::int32_t ids[3] = {2, 5, 0};
    inputIds->write(ids, sizeof(ids));

    auto fence = context->dispatch(*result.graph, {{"input_ids", inputIds}}, {{"logits", logits}});
    fence->wait();

    float actual[seqLen * vocabSize];
    logits->read(actual, sizeof(actual));

    float expected[] = {0.289978f,  -0.215047f, 0.142526f, -0.246236f, 0.652247f,  -0.379452f, 1.226883f,
                         0.409712f,  -0.293554f, -0.214945f, -0.098689f, -0.319543f, 0.272893f,  0.586046f,
                         0.614507f,  -0.061498f, 0.362973f, -0.390258f, 0.481074f,  -0.284297f, 1.301201f};

    for (std::size_t i = 0; i < seqLen * vocabSize; ++i)
    {
        EXPECT_NEAR(actual[i], expected[i], 1e-3f) << "mismatch at flat index " << i;
    }
}

TEST(GptArchitecture, ThrowsOnMissingWeightTensor)
{
    auto weights = loadSafetensorsFromFile(fixturePath("gpt_test_weights.safetensors"));
    auto context = cnn::Context::create({cnn::DeviceType::Cpu});

    GptConfig config = tinyGptTestConfig();
    config.numLayers = 99;

    EXPECT_THROW(buildGptGraph(context, *weights, config, 3), std::runtime_error);
}
