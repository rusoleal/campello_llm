#include <gtest/gtest.h>

#include <string>

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
} // namespace

TEST(ArchitectureRegistry, DispatchesLlamaByName)
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

    auto weights = loadSafetensorsFromFile(fixturePath("llama_test_weights.safetensors"));
    auto context = cnn::Context::create({cnn::DeviceType::Cpu});

    ArchitectureGraphResult result = buildGraphForArchitecture("llama", context, *weights, ArchitectureConfig(config), 3);
    ASSERT_NE(result.graph, nullptr);
    EXPECT_NE(result.outputs.find("logits"), result.outputs.end());
}

TEST(ArchitectureRegistry, DispatchesGptByName)
{
    GptConfig config;
    config.vocabSize = 7;
    config.hiddenSize = 8;
    config.numLayers = 2;
    config.numAttentionHeads = 2;
    config.intermediateSize = 12;
    config.layerNormEps = 1e-5f;
    config.maxPositionEmbeddings = 4;

    auto weights = loadSafetensorsFromFile(fixturePath("gpt_test_weights.safetensors"));
    auto context = cnn::Context::create({cnn::DeviceType::Cpu});

    ArchitectureGraphResult result = buildGraphForArchitecture("gpt2", context, *weights, ArchitectureConfig(config), 3);
    ASSERT_NE(result.graph, nullptr);
    EXPECT_NE(result.outputs.find("logits"), result.outputs.end());
}

TEST(ArchitectureRegistry, UnknownArchitectureNameThrows)
{
    LlamaConfig config{};
    auto weights = loadSafetensorsFromFile(fixturePath("llama_test_weights.safetensors"));
    auto context = cnn::Context::create({cnn::DeviceType::Cpu});

    EXPECT_THROW(buildGraphForArchitecture("not-a-real-architecture", context, *weights, ArchitectureConfig(config), 3),
                 std::runtime_error);
}

TEST(ArchitectureRegistry, MismatchedConfigVariantThrows)
{
    // "llama" requires a LlamaConfig -- passing a GptConfig must throw, not
    // silently misinterpret the hyperparameters.
    GptConfig config{};
    auto weights = loadSafetensorsFromFile(fixturePath("llama_test_weights.safetensors"));
    auto context = cnn::Context::create({cnn::DeviceType::Cpu});

    EXPECT_THROW(buildGraphForArchitecture("llama", context, *weights, ArchitectureConfig(config), 3),
                 std::runtime_error);
}
