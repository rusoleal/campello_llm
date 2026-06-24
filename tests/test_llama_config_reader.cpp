#include <gtest/gtest.h>

#include <string>

#include <campello_llm/architecture.hpp>

using namespace systems::leal::campello_llm;

namespace
{
    std::string fixturePath(const std::string &name)
    {
        return std::string(CAMPELLO_LLM_TEST_FIXTURES_DIR) + "/" + name;
    }
} // namespace

// tests/fixtures/tinyllama_config.json is the real config.json from
// TinyLlama/TinyLlama-1.1B-Chat-v1.0 (Apache-2.0, see tests/fixtures/NOTICE.md).
TEST(LlamaConfigReader, ReadsRealTinyLlamaConfig)
{
    LlamaConfig config = loadLlamaConfigFromFile(fixturePath("tinyllama_config.json"));
    EXPECT_EQ(config.vocabSize, 32000);
    EXPECT_EQ(config.hiddenSize, 2048);
    EXPECT_EQ(config.numLayers, 22);
    EXPECT_EQ(config.numAttentionHeads, 32);
    EXPECT_EQ(config.numKeyValueHeads, 4);
    EXPECT_EQ(config.intermediateSize, 5632);
    EXPECT_FLOAT_EQ(config.rmsNormEps, 1e-5f);
    EXPECT_FLOAT_EQ(config.ropeTheta, 10000.0f);
}

TEST(LlamaConfigReader, DefaultsNumKeyValueHeadsToNumAttentionHeadsWhenAbsent)
{
    std::string json = R"({"model_type":"llama","vocab_size":1,"hidden_size":2,"num_hidden_layers":1,
                            "num_attention_heads":4,"intermediate_size":8,"rms_norm_eps":1e-5})";
    LlamaConfig config = loadLlamaConfigFromMemory(json.data(), json.size());
    EXPECT_EQ(config.numKeyValueHeads, 4);
    EXPECT_FLOAT_EQ(config.ropeTheta, 10000.0f);
}

TEST(LlamaConfigReader, WrongModelTypeThrows)
{
    std::string json = R"({"model_type":"gpt2","vocab_size":1,"hidden_size":2,"num_hidden_layers":1,
                            "num_attention_heads":4,"intermediate_size":8,"rms_norm_eps":1e-5})";
    EXPECT_THROW(loadLlamaConfigFromMemory(json.data(), json.size()), std::runtime_error);
}

TEST(LlamaConfigReader, MissingRequiredFieldThrows)
{
    std::string json = R"({"model_type":"llama","vocab_size":1})";
    EXPECT_THROW(loadLlamaConfigFromMemory(json.data(), json.size()), std::runtime_error);
}

TEST(LlamaConfigReader, MissingFileThrows)
{
    EXPECT_THROW(loadLlamaConfigFromFile(fixturePath("does_not_exist_config.json")), std::runtime_error);
}
