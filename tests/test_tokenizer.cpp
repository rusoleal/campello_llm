#include <gtest/gtest.h>

#include <cstdint>
#include <fstream>
#include <memory>
#include <string>
#include <vector>

#include <campello_llm/tokenizer.hpp>
#include <campello_llm/tokenizer_config.hpp>

using namespace systems::leal::campello_llm;

namespace
{
    std::string fixturePath(const std::string &name)
    {
        return std::string(CAMPELLO_LLM_TEST_FIXTURES_DIR) + "/" + name;
    }

    std::unique_ptr<Tokenizer> loadTinyLlamaTokenizer()
    {
        return loadTokenizerFromFile(fixturePath("tinyllama_tokenizer.json"));
    }
} // namespace

// tests/fixtures/tinyllama_tokenizer{,_config}.json are the real files from
// TinyLlama/TinyLlama-1.1B-Chat-v1.0 (Apache-2.0, see tests/fixtures/NOTICE.md).
// Every expected token id list / decoded string below was independently confirmed
// against the real `tokenizers` Python package (`Tokenizer.from_file(...).encode()`/
// `.decode()`) before being hardcoded here, not assumed.

TEST(Tokenizer, VocabSize)
{
    auto tok = loadTinyLlamaTokenizer();
    EXPECT_EQ(tok->vocabSize(), 32000u);
}

TEST(Tokenizer, SpecialTokenIds)
{
    auto tok = loadTinyLlamaTokenizer();
    EXPECT_EQ(tok->bosId(), 1);
    EXPECT_EQ(tok->unkId(), 0);
    EXPECT_EQ(tok->tokenToId("</s>"), 2);
}

TEST(Tokenizer, EncodeHelloWorld)
{
    auto tok = loadTinyLlamaTokenizer();
    EXPECT_EQ(tok->encode("Hello, world!"), (std::vector<std::int32_t>{1, 15043, 29892, 3186, 29991}));
}

TEST(Tokenizer, EncodeLongerSentence)
{
    auto tok = loadTinyLlamaTokenizer();
    EXPECT_EQ(tok->encode("The quick brown fox jumps over the lazy dog."),
              (std::vector<std::int32_t>{1, 450, 4996, 17354, 1701, 29916, 432, 17204, 975, 278, 17366, 11203, 29889}));
}

TEST(Tokenizer, EncodeUnderscoreIdentifier)
{
    auto tok = loadTinyLlamaTokenizer();
    EXPECT_EQ(tok->encode("campello_llm"), (std::vector<std::int32_t>{1, 4242, 3156, 29918, 645, 29885}));
}

TEST(Tokenizer, EncodeAccentedLatinText)
{
    auto tok = loadTinyLlamaTokenizer();
    EXPECT_EQ(tok->encode("café"), (std::vector<std::int32_t>{1, 274, 28059}));
    EXPECT_EQ(tok->encode("naïve façade"), (std::vector<std::int32_t>{1, 1055, 30085, 345, 2258, 30019, 1943}));
}

TEST(Tokenizer, EncodeCjkUsesByteFallbackBoundaryMarker)
{
    auto tok = loadTinyLlamaTokenizer();
    EXPECT_EQ(tok->encode("你好"), (std::vector<std::int32_t>{1, 29871, 30919, 31076}));
}

TEST(Tokenizer, EncodeEmojiFallsBackToRawBytes)
{
    auto tok = loadTinyLlamaTokenizer();
    // U+1F600 has no direct vocab entry: byte_fallback emits its 4 UTF-8 bytes as
    // four separate <0xXX> tokens (none of which happen to have a merge rule here).
    EXPECT_EQ(tok->encode("\xF0\x9F\x98\x80", false), (std::vector<std::int32_t>{29871, 243, 162, 155, 131}));
}

TEST(Tokenizer, EncodeEmptyString)
{
    auto tok = loadTinyLlamaTokenizer();
    EXPECT_EQ(tok->encode("", false), (std::vector<std::int32_t>{}));
    EXPECT_EQ(tok->encode(""), (std::vector<std::int32_t>{1}));
}

TEST(Tokenizer, EncodeWithoutSpecialTokens)
{
    auto tok = loadTinyLlamaTokenizer();
    EXPECT_EQ(tok->encode("Hello, world!", false), (std::vector<std::int32_t>{15043, 29892, 3186, 29991}));
}

TEST(Tokenizer, EncodeRecognizesLiteralAddedTokenRegardlessOfAddSpecialTokens)
{
    auto tok = loadTinyLlamaTokenizer();
    // "</s>" (id 2) is recognized as a literal added token even with
    // addSpecialTokens=false -- only the post-processor's auto-added BOS is gated
    // by that flag, not added_tokens recognition.
    EXPECT_EQ(tok->encode("a</s>b", false), (std::vector<std::int32_t>{263, 2, 289}));
}

TEST(Tokenizer, EncodeMultipleSpaces)
{
    auto tok = loadTinyLlamaTokenizer();
    EXPECT_EQ(tok->encode("Multiple   spaces   between   words"),
              (std::vector<std::int32_t>{1, 26905, 259, 8162, 259, 1546, 259, 3838}));
}

TEST(Tokenizer, DecodeRoundTrip)
{
    auto tok = loadTinyLlamaTokenizer();
    for (const std::string &text : {
             "Hello, world!",
             "The quick brown fox jumps over the lazy dog.",
             "campello_llm",
             "café",
             "naïve façade",
             "你好",
             "   leading and trailing spaces   ",
             "line1\nline2\ttab",
         })
    {
        auto ids = tok->encode(text);
        EXPECT_EQ(tok->decode(ids), text) << "round-trip failed for " << text;
    }
}

TEST(Tokenizer, DecodeEmptyIds)
{
    auto tok = loadTinyLlamaTokenizer();
    EXPECT_EQ(tok->decode({}), "");
}

TEST(Tokenizer, ChatPromptFormattingMatchesRealJinjaRender)
{
    // Verified against the real chat_template rendered with Jinja2 using
    // HuggingFace's default Environment settings (trim_blocks/lstrip_blocks=True,
    // keep_trailing_newline=True) -- see CLAUDE.md.
    std::string prompt = formatSingleTurnChatPrompt("user", "Hello there", "</s>");
    EXPECT_EQ(prompt, "<|user|>\nHello there</s>\n<|assistant|>\n");

    auto tok = loadTinyLlamaTokenizer();
    auto ids = tok->encode(prompt);
    EXPECT_EQ(ids, (std::vector<std::int32_t>{1, 529, 29989, 1792, 29989, 29958, 13, 10994, 727, 2, 29871, 13, 29966,
                                               29989, 465, 22137, 29989, 29958, 13}));
    EXPECT_EQ(tok->decode(ids), "<|user|>\nHello there \n<|assistant|>\n");
    EXPECT_EQ(tok->decode(ids, false), "<s> <|user|>\nHello there</s> \n<|assistant|>\n");
}

TEST(Tokenizer, LoadFromMemoryMatchesLoadFromFile)
{
    std::string path = fixturePath("tinyllama_tokenizer.json");
    std::ifstream stream(path, std::ios::binary | std::ios::ate);
    ASSERT_TRUE(stream.is_open());
    std::streamsize size = stream.tellg();
    stream.seekg(0, std::ios::beg);
    std::vector<char> buffer(static_cast<std::size_t>(size));
    ASSERT_TRUE(stream.read(buffer.data(), size));

    auto tok = loadTokenizerFromMemory(buffer.data(), buffer.size());
    EXPECT_EQ(tok->encode("Hello, world!"), (std::vector<std::int32_t>{1, 15043, 29892, 3186, 29991}));
}

TEST(Tokenizer, MissingFileThrows)
{
    EXPECT_THROW(loadTokenizerFromFile(fixturePath("does_not_exist_tokenizer.json")), std::runtime_error);
}

TEST(TokenizerConfig, ReadsSpecialTokenStrings)
{
    SpecialTokenStrings tokens = loadSpecialTokenStringsFromFile(fixturePath("tinyllama_tokenizer_config.json"));
    EXPECT_EQ(tokens.bosToken, "<s>");
    EXPECT_EQ(tokens.eosToken, "</s>");
    EXPECT_EQ(tokens.unkToken, "<unk>");
    EXPECT_EQ(tokens.padToken, "</s>");
}

TEST(TokenizerConfig, SpecialTokenStringsResolveToIdsViaTokenizer)
{
    auto tok = loadTinyLlamaTokenizer();
    SpecialTokenStrings tokens = loadSpecialTokenStringsFromFile(fixturePath("tinyllama_tokenizer_config.json"));
    EXPECT_EQ(tok->tokenToId(tokens.bosToken), 1);
    EXPECT_EQ(tok->tokenToId(tokens.eosToken), 2);
    EXPECT_EQ(tok->tokenToId(tokens.unkToken), 0);
}

namespace
{
    // Smallest possible valid tokenizer.json: a 2-token vocab, one merge, no
    // normalizer/pre_tokenizer/decoder/post_processor/added_tokens -- used to probe
    // the "throw rather than guess" paths without needing a second large fixture.
    std::string minimalValidTokenizerJson()
    {
        return R"({"model":{"type":"BPE","vocab":{"a":0,"b":1,"ab":2},"merges":["a b"]}})";
    }
} // namespace

TEST(Tokenizer, MinimalTokenizerWithNoOptionalSectionsWorks)
{
    std::string json = minimalValidTokenizerJson();
    auto tok = loadTokenizerFromMemory(json.data(), json.size());
    EXPECT_EQ(tok->vocabSize(), 3u);
    EXPECT_EQ(tok->bosId(), -1);
    EXPECT_EQ(tok->encode("ab", false), (std::vector<std::int32_t>{2}));
    EXPECT_EQ(tok->decode({2}), "ab");
}

TEST(Tokenizer, MergesAsTwoElementArraysAlsoSupported)
{
    // Newer `tokenizers` versions (confirmed against the real library, 0.22.2)
    // serialize each merge as ["left","right"] instead of the older single
    // "left right" string TinyLlama's real tokenizer.json (and
    // minimalValidTokenizerJson() above) uses -- both are real, valid shapes.
    std::string json = R"({"model":{"type":"BPE","vocab":{"a":0,"b":1,"ab":2},"merges":[["a","b"]]}})";
    auto tok = loadTokenizerFromMemory(json.data(), json.size());
    EXPECT_EQ(tok->encode("ab", false), (std::vector<std::int32_t>{2}));
}

TEST(Tokenizer, UnsupportedModelTypeThrows)
{
    std::string json = R"({"model":{"type":"Unigram","vocab":{},"merges":[]}})";
    EXPECT_THROW(loadTokenizerFromMemory(json.data(), json.size()), std::runtime_error);
}

TEST(Tokenizer, UnsupportedPreTokenizerThrows)
{
    std::string json = R"({"model":{"type":"BPE","vocab":{"a":0},"merges":[]},
                            "pre_tokenizer":{"type":"ByteLevel"}})";
    EXPECT_THROW(loadTokenizerFromMemory(json.data(), json.size()), std::runtime_error);
}

TEST(Tokenizer, UnsupportedNormalizerShapeThrows)
{
    std::string json = R"({"model":{"type":"BPE","vocab":{"a":0},"merges":[]},
                            "normalizer":{"type":"Lowercase"}})";
    EXPECT_THROW(loadTokenizerFromMemory(json.data(), json.size()), std::runtime_error);
}

TEST(Tokenizer, MergesReferencingMissingVocabEntryThrows)
{
    std::string json = R"({"model":{"type":"BPE","vocab":{"a":0,"b":1},"merges":["a b"]}})";
    EXPECT_THROW(loadTokenizerFromMemory(json.data(), json.size()), std::runtime_error);
}
