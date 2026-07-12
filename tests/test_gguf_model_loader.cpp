#include <gtest/gtest.h>

#include <campello_llm/gguf_reader.hpp>

#include "../src/gguf/gguf_model_loader.hpp"

#include <cstdint>
#include <cstring>
#include <string>
#include <vector>

using namespace systems::leal::campello_llm;

namespace
{
    template <typename T>
    void appendLE(std::vector<std::uint8_t> &buf, T value)
    {
        std::uint8_t bytes[sizeof(T)];
        std::memcpy(bytes, &value, sizeof(T));
        buf.insert(buf.end(), bytes, bytes + sizeof(T));
    }

    void appendGgufString(std::vector<std::uint8_t> &buf, const std::string &s)
    {
        appendLE<std::uint64_t>(buf, s.size());
        buf.insert(buf.end(), s.begin(), s.end());
    }

    void appendMetadataString(std::vector<std::uint8_t> &buf, const std::string &key, const std::string &value)
    {
        appendGgufString(buf, key);
        appendLE<std::uint32_t>(buf, static_cast<std::uint32_t>(GgufValueType::String));
        appendGgufString(buf, value);
    }

    void appendMetadataUInt32(std::vector<std::uint8_t> &buf, const std::string &key, std::uint32_t value)
    {
        appendGgufString(buf, key);
        appendLE<std::uint32_t>(buf, static_cast<std::uint32_t>(GgufValueType::UInt32));
        appendLE<std::uint32_t>(buf, value);
    }

    void appendMetadataFloat32(std::vector<std::uint8_t> &buf, const std::string &key, float value)
    {
        appendGgufString(buf, key);
        appendLE<std::uint32_t>(buf, static_cast<std::uint32_t>(GgufValueType::Float32));
        appendLE<float>(buf, value);
    }

    void appendMetadataStringArray(std::vector<std::uint8_t> &buf, const std::string &key,
                                   const std::vector<std::string> &values)
    {
        appendGgufString(buf, key);
        appendLE<std::uint32_t>(buf, static_cast<std::uint32_t>(GgufValueType::Array));
        appendLE<std::uint32_t>(buf, static_cast<std::uint32_t>(GgufValueType::String));
        appendLE<std::uint64_t>(buf, values.size());
        for (const std::string &v : values)
        {
            appendGgufString(buf, v);
        }
    }

    void appendTensorInfo(std::vector<std::uint8_t> &buf, const std::string &name,
                          const std::vector<std::uint64_t> &dims, std::uint32_t ggmlType,
                          std::uint64_t relativeOffset)
    {
        appendGgufString(buf, name);
        appendLE<std::uint32_t>(buf, static_cast<std::uint32_t>(dims.size()));
        for (std::uint64_t d : dims)
        {
            appendLE<std::uint64_t>(buf, d);
        }
        appendLE<std::uint32_t>(buf, ggmlType);
        appendLE<std::uint64_t>(buf, relativeOffset);
    }

    std::size_t alignedOffset(std::size_t offset, std::size_t alignment)
    {
        std::size_t remainder = offset % alignment;
        return remainder == 0 ? offset : offset + (alignment - remainder);
    }

    std::unique_ptr<GgufFile> buildMinimalLlamaGguf()
    {
        std::vector<std::uint8_t> buf;
        appendLE<std::uint32_t>(buf, 0x46554747); // magic "GGUF"
        appendLE<std::uint32_t>(buf, 3);          // version
        appendLE<std::uint64_t>(buf, 1);          // tensor_count
        appendLE<std::uint64_t>(buf, 9);          // kv_count

        appendMetadataString(buf, "general.architecture", "llama");
        appendMetadataUInt32(buf, "llama.block_count", 2);
        appendMetadataUInt32(buf, "llama.embedding_length", 64);
        appendMetadataUInt32(buf, "llama.feed_forward_length", 128);
        appendMetadataUInt32(buf, "llama.attention.head_count", 4);
        appendMetadataUInt32(buf, "llama.attention.head_count_kv", 2);
        appendMetadataFloat32(buf, "llama.attention.layer_norm_rms_epsilon", 1e-5f);
        appendMetadataFloat32(buf, "llama.rope.freq_base", 10000.0f);
        appendMetadataUInt32(buf, "llama.vocab_size", 8);

        appendTensorInfo(buf, "token_embd.weight", {64, 8}, 0, 0); // GGUF dims reversed: [hidden, vocab]

        std::size_t headerEnd = buf.size();
        std::size_t dataStart = alignedOffset(headerEnd, 32);
        buf.resize(dataStart + 64 * 8 * sizeof(float), 0);

        return loadGgufFromMemory(buf.data(), buf.size());
    }

    std::unique_ptr<GgufFile> buildGgufWithTokenizer()
    {
        std::vector<std::uint8_t> buf;
        appendLE<std::uint32_t>(buf, 0x46554747);
        appendLE<std::uint32_t>(buf, 3);
        appendLE<std::uint64_t>(buf, 0); // no tensors
        appendLE<std::uint64_t>(buf, 6); // kv_count

        appendMetadataString(buf, "general.architecture", "llama");
        appendMetadataStringArray(
            buf, "tokenizer.ggml.tokens",
            {"<s>", "h", "e", "l", "o", "hello", "world", "</s>", "<0x41>", "\xE2\x96\x81", "he", "hel", "hell"});
        appendMetadataStringArray(buf, "tokenizer.ggml.merges", {"h e", "he l", "hel l", "hell o"});
        appendMetadataUInt32(buf, "tokenizer.ggml.bos_token_id", 0);
        appendMetadataUInt32(buf, "tokenizer.ggml.eos_token_id", 3);
        appendMetadataUInt32(buf, "tokenizer.ggml.unknown_token_id", 0);

        std::size_t headerEnd = buf.size();
        std::size_t dataStart = alignedOffset(headerEnd, 32);
        buf.resize(dataStart, 0);

        return loadGgufFromMemory(buf.data(), buf.size());
    }
} // namespace

TEST(GgufModelLoader, ReadsLlamaConfig)
{
    auto file = buildMinimalLlamaGguf();
    LlamaConfig config = loadLlamaConfigFromGgufFile(*file);
    EXPECT_EQ(config.numLayers, 2);
    EXPECT_EQ(config.hiddenSize, 64);
    EXPECT_EQ(config.intermediateSize, 128);
    EXPECT_EQ(config.numAttentionHeads, 4);
    EXPECT_EQ(config.numKeyValueHeads, 2);
    EXPECT_FLOAT_EQ(config.rmsNormEps, 1e-5f);
    EXPECT_FLOAT_EQ(config.ropeTheta, 10000.0f);
    EXPECT_EQ(config.vocabSize, 8);
}

TEST(GgufModelLoader, AdapterMapsAndReversesShapes)
{
    auto file = buildMinimalLlamaGguf();
    GgufWeightsAdapter adapter(*file);

    const TensorInfo *t = adapter.find("model.embed_tokens.weight");
    ASSERT_NE(t, nullptr);
    EXPECT_EQ(t->dtype, WeightDType::F32);
    EXPECT_EQ(t->shape, (std::vector<std::int64_t>{8, 64}));
    EXPECT_EQ(t->byteLength, 64u * 8u * sizeof(float));
}

TEST(GgufModelLoader, AdapterFindMissingReturnsNull)
{
    auto file = buildMinimalLlamaGguf();
    GgufWeightsAdapter adapter(*file);
    EXPECT_EQ(adapter.find("model.layers.0.self_attn.q_proj.weight"), nullptr);
}

TEST(GgufModelLoader, TokenizerFromGgufMetadata)
{
    auto file = buildGgufWithTokenizer();
    auto tokenizer = loadTokenizerFromGgufFile(*file);

    EXPECT_EQ(tokenizer->bosId(), 0);
    EXPECT_EQ(tokenizer->unkId(), 0);
    EXPECT_TRUE(tokenizer->tokenToId("hello").has_value());
    EXPECT_EQ(*tokenizer->tokenToId("hello"), 5);

    // Encoding "hello" with BOS should produce [0, 9 (▁), 5 (hello)] for this
    // tiny SentencePiece-style BPE table (h+e -> he -> hel -> hell -> hello).
    std::vector<std::int32_t> ids = tokenizer->encode("hello", true);
    ASSERT_EQ(ids.size(), 3u);
    EXPECT_EQ(ids[0], 0);
    EXPECT_EQ(ids[1], 9);
    EXPECT_EQ(ids[2], 5);
}
