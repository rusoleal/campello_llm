#include "gguf_model_loader.hpp"

#include <campello_nn/float16.hpp>

#include <algorithm>
#include <cctype>
#include <cstdlib>
#include <cstring>
#include <regex>
#include <stdexcept>

namespace systems::leal::campello_llm
{

    namespace
    {
        // ------------------------------------------------------------------
        // Metadata helpers
        // ------------------------------------------------------------------

        std::int64_t metadataInt64(const GgufFile &file, const std::string &key)
        {
            const GgufValue *v = file.findMetadata(key);
            if (v == nullptr)
            {
                throw std::runtime_error("campello_llm: missing required gguf metadata key '" + key + "'");
            }
            switch (v->type)
            {
            case GgufValueType::UInt8:
                return static_cast<std::int64_t>(v->asUInt8());
            case GgufValueType::Int8:
                return static_cast<std::int64_t>(v->asInt8());
            case GgufValueType::UInt16:
                return static_cast<std::int64_t>(v->asUInt16());
            case GgufValueType::Int16:
                return static_cast<std::int64_t>(v->asInt16());
            case GgufValueType::UInt32:
                return static_cast<std::int64_t>(v->asUInt32());
            case GgufValueType::Int32:
                return static_cast<std::int64_t>(v->asInt32());
            case GgufValueType::UInt64:
                return static_cast<std::int64_t>(v->asUInt64());
            case GgufValueType::Int64:
                return v->asInt64();
            case GgufValueType::Float32:
                return static_cast<std::int64_t>(v->asFloat32());
            case GgufValueType::Float64:
                return static_cast<std::int64_t>(v->asFloat64());
            default:
                throw std::runtime_error("campello_llm: gguf metadata key '" + key + "' is not numeric");
            }
        }

        std::int64_t metadataInt64Or(const GgufFile &file, const std::string &key, std::int64_t defaultValue)
        {
            if (file.findMetadata(key) == nullptr)
            {
                return defaultValue;
            }
            return metadataInt64(file, key);
        }

        float metadataFloat32Or(const GgufFile &file, const std::string &key, float defaultValue)
        {
            const GgufValue *v = file.findMetadata(key);
            if (v == nullptr)
            {
                return defaultValue;
            }
            switch (v->type)
            {
            case GgufValueType::UInt8:
                return static_cast<float>(v->asUInt8());
            case GgufValueType::Int8:
                return static_cast<float>(v->asInt8());
            case GgufValueType::UInt16:
                return static_cast<float>(v->asUInt16());
            case GgufValueType::Int16:
                return static_cast<float>(v->asInt16());
            case GgufValueType::UInt32:
                return static_cast<float>(v->asUInt32());
            case GgufValueType::Int32:
                return static_cast<float>(v->asInt32());
            case GgufValueType::UInt64:
                return static_cast<float>(v->asUInt64());
            case GgufValueType::Int64:
                return static_cast<float>(v->asInt64());
            case GgufValueType::Float32:
                return v->asFloat32();
            case GgufValueType::Float64:
                return static_cast<float>(v->asFloat64());
            default:
                return defaultValue;
            }
        }

        const std::vector<GgufValue> &metadataStringArray(const GgufFile &file, const std::string &key)
        {
            const GgufValue *v = file.findMetadata(key);
            if (v == nullptr)
            {
                throw std::runtime_error("campello_llm: missing required gguf metadata key '" + key + "'");
            }
            if (v->type != GgufValueType::Array || v->arrayElementType != GgufValueType::String)
            {
                throw std::runtime_error("campello_llm: gguf metadata key '" + key + "' is not a string array");
            }
            return v->asArray();
        }

        const std::vector<GgufValue> *metadataStringArrayOptional(const GgufFile &file, const std::string &key)
        {
            const GgufValue *v = file.findMetadata(key);
            if (v == nullptr)
            {
                return nullptr;
            }
            if (v->type != GgufValueType::Array || v->arrayElementType != GgufValueType::String)
            {
                throw std::runtime_error("campello_llm: gguf metadata key '" + key + "' is not a string array");
            }
            return &v->asArray();
        }

        const std::vector<GgufValue> *metadataIntArrayOptional(const GgufFile &file, const std::string &key)
        {
            const GgufValue *v = file.findMetadata(key);
            if (v == nullptr)
            {
                return nullptr;
            }
            if (v->type != GgufValueType::Array ||
                (v->arrayElementType != GgufValueType::Int32 && v->arrayElementType != GgufValueType::UInt32))
            {
                throw std::runtime_error("campello_llm: gguf metadata key '" + key + "' is not an int32 array");
            }
            return &v->asArray();
        }

        // ------------------------------------------------------------------
        // Name mapping: llama.cpp (GGUF) -> HuggingFace
        // ------------------------------------------------------------------

        bool startsWith(const std::string &s, const std::string &prefix)
        {
            return s.size() >= prefix.size() && s.compare(0, prefix.size(), prefix) == 0;
        }

        bool parseLayerPrefix(const std::string &name, const std::string &prefix, std::string &suffix,
                              std::int64_t &layer)
        {
            if (!startsWith(name, prefix))
            {
                return false;
            }
            std::string rest = name.substr(prefix.size());
            std::size_t dot = rest.find('.');
            if (dot == std::string::npos)
            {
                return false;
            }
            std::string numStr = rest.substr(0, dot);
            char *end = nullptr;
            long value = std::strtol(numStr.c_str(), &end, 10);
            if (end != numStr.c_str() + numStr.size() || value < 0)
            {
                return false;
            }
            layer = static_cast<std::int64_t>(value);
            suffix = rest.substr(dot + 1);
            return true;
        }

        std::optional<std::string> mapGgufNameToHf(const std::string &ggufName)
        {
            if (ggufName == "token_embd.weight")
            {
                return "model.embed_tokens.weight";
            }
            if (ggufName == "output_norm.weight")
            {
                return "model.norm.weight";
            }
            if (ggufName == "output.weight")
            {
                return "lm_head.weight";
            }

            std::int64_t layer;
            std::string suffix;
            if (parseLayerPrefix(ggufName, "blk.", suffix, layer))
            {
                const std::string layerPrefix = "model.layers." + std::to_string(layer) + ".";
                if (suffix == "attn_norm.weight")
                    return layerPrefix + "input_layernorm.weight";
                if (suffix == "ffn_norm.weight")
                    return layerPrefix + "post_attention_layernorm.weight";
                if (suffix == "attn_q.weight")
                    return layerPrefix + "self_attn.q_proj.weight";
                if (suffix == "attn_k.weight")
                    return layerPrefix + "self_attn.k_proj.weight";
                if (suffix == "attn_v.weight")
                    return layerPrefix + "self_attn.v_proj.weight";
                if (suffix == "attn_output.weight")
                    return layerPrefix + "self_attn.o_proj.weight";
                if (suffix == "ffn_gate.weight")
                    return layerPrefix + "mlp.gate_proj.weight";
                if (suffix == "ffn_up.weight")
                    return layerPrefix + "mlp.up_proj.weight";
                if (suffix == "ffn_down.weight")
                    return layerPrefix + "mlp.down_proj.weight";
            }

            return std::nullopt;
        }

        std::vector<std::int64_t> reverseShape(const std::vector<std::int64_t> &shape)
        {
            std::vector<std::int64_t> reversed = shape;
            std::reverse(reversed.begin(), reversed.end());
            return reversed;
        }

        // ------------------------------------------------------------------
        // Tokenizer helpers
        // ------------------------------------------------------------------

        constexpr const char *kSpaceMarker = "\xE2\x96\x81"; // U+2581

        std::uint64_t pairKey(std::int32_t left, std::int32_t right)
        {
            return (static_cast<std::uint64_t>(static_cast<std::uint32_t>(left)) << 32) |
                   static_cast<std::uint32_t>(right);
        }

        bool isByteFallbackToken(const std::string &token)
        {
            if (token.size() != 6 || token[0] != '<' || token[1] != '0' || token[2] != 'x' || token[5] != '>')
            {
                return false;
            }
            auto hexVal = [](char c) -> int {
                if (c >= '0' && c <= '9')
                    return c - '0';
                if (c >= 'A' && c <= 'F')
                    return c - 'A' + 10;
                if (c >= 'a' && c <= 'f')
                    return c - 'a' + 10;
                return -1;
            };
            return hexVal(token[3]) >= 0 && hexVal(token[4]) >= 0;
        }

    } // namespace

    // ------------------------------------------------------------------
    // GgufWeightsAdapter
    // ------------------------------------------------------------------

    GgufWeightsAdapter::GgufWeightsAdapter(const GgufFile &gguf) : gguf_(gguf)
    {
        for (const TensorInfo &t : gguf.tensors())
        {
            std::optional<std::string> hfName = mapGgufNameToHf(t.name);
            if (!hfName)
            {
                continue;
            }

            TensorInfo adapted;
            adapted.name = *hfName;
            adapted.dtype = t.dtype;
            adapted.shape = reverseShape(t.shape);
            adapted.data = t.data;
            adapted.byteLength = t.byteLength;

            nameToIndex_.emplace(adapted.name, adaptedTensors_.size());
            adaptedTensors_.push_back(std::move(adapted));
        }
    }

    const std::vector<TensorInfo> &GgufWeightsAdapter::tensors() const
    {
        return adaptedTensors_;
    }

    const TensorInfo *GgufWeightsAdapter::find(const std::string &name) const
    {
        auto it = nameToIndex_.find(name);
        if (it == nameToIndex_.end())
        {
            return nullptr;
        }
        return &adaptedTensors_[it->second];
    }

    // ------------------------------------------------------------------
    // Config reader
    // ------------------------------------------------------------------

    LlamaConfig loadLlamaConfigFromGgufFile(const GgufFile &file)
    {
        const GgufValue *arch = file.findMetadata("general.architecture");
        if (arch == nullptr || arch->type != GgufValueType::String || arch->asString() != "llama")
        {
            throw std::runtime_error("campello_llm: gguf general.architecture is not 'llama'");
        }

        LlamaConfig config;
        config.numLayers = metadataInt64(file, "llama.block_count");
        config.hiddenSize = metadataInt64(file, "llama.embedding_length");
        config.intermediateSize = metadataInt64(file, "llama.feed_forward_length");
        config.numAttentionHeads = metadataInt64(file, "llama.attention.head_count");
        config.numKeyValueHeads =
            metadataInt64Or(file, "llama.attention.head_count_kv", config.numAttentionHeads);
        config.rmsNormEps = metadataFloat32Or(file, "llama.attention.layer_norm_rms_epsilon", 1e-5f);
        config.ropeTheta = metadataFloat32Or(file, "llama.rope.freq_base", 10000.0f);

        if (const GgufValue *vocabSize = file.findMetadata("llama.vocab_size"))
        {
            config.vocabSize = metadataInt64(file, "llama.vocab_size");
        }
        else
        {
            const TensorInfo *embed = file.find("token_embd.weight");
            if (embed == nullptr || embed->shape.size() < 2)
            {
                throw std::runtime_error("campello_llm: gguf missing llama.vocab_size and token_embd.weight");
            }
            config.vocabSize = embed->shape.back();
        }

        return config;
    }

    // ------------------------------------------------------------------
    // Tokenizer loader
    // ------------------------------------------------------------------

    std::unique_ptr<Tokenizer> loadTokenizerFromGgufFile(const GgufFile &file)
    {
        const GgufValue *model = file.findMetadata("tokenizer.ggml.model");
        const std::string modelName = (model != nullptr && model->type == GgufValueType::String)
                                          ? model->asString()
                                          : std::string("llama");

        const std::vector<GgufValue> &tokens = metadataStringArray(file, "tokenizer.ggml.tokens");
        const std::vector<GgufValue> *merges = metadataStringArrayOptional(file, "tokenizer.ggml.merges");
        const std::vector<GgufValue> *tokenTypes = metadataIntArrayOptional(file, "tokenizer.ggml.token_type");

        auto tokenizer = std::make_unique<Tokenizer>();

        tokenizer->idToToken_.reserve(tokens.size());
        for (std::size_t i = 0; i < tokens.size(); ++i)
        {
            tokenizer->idToToken_.push_back(tokens[i].asString());
            tokenizer->tokenToId_[tokens[i].asString()] = static_cast<std::int32_t>(i);
        }

        if (merges != nullptr)
        {
            for (std::size_t rank = 0; rank < merges->size(); ++rank)
            {
                const std::string &entry = (*merges)[rank].asString();
                std::size_t sep = entry.find(' ');
                if (sep == std::string::npos)
                {
                    throw std::runtime_error("campello_llm: malformed gguf merge entry '" + entry + "'");
                }
                std::string left = entry.substr(0, sep);
                std::string right = entry.substr(sep + 1);

                auto leftIt = tokenizer->tokenToId_.find(left);
                auto rightIt = tokenizer->tokenToId_.find(right);
                auto mergedIt = tokenizer->tokenToId_.find(left + right);
                if (leftIt == tokenizer->tokenToId_.end() || rightIt == tokenizer->tokenToId_.end() ||
                    mergedIt == tokenizer->tokenToId_.end())
                {
                    throw std::runtime_error("campello_llm: gguf merge entry '" + entry +
                                              "' references a token missing from vocab");
                }
                tokenizer->mergeRank_[pairKey(leftIt->second, rightIt->second)] = {
                    static_cast<std::int32_t>(rank), mergedIt->second};
            }
        }

        for (const std::string &token : tokenizer->idToToken_)
        {
            if (isByteFallbackToken(token))
            {
                tokenizer->byteFallback_ = true;
                break;
            }
        }

        tokenizer->bosId_ = static_cast<std::int32_t>(metadataInt64Or(file, "tokenizer.ggml.bos_token_id", -1));
        tokenizer->unkId_ = static_cast<std::int32_t>(metadataInt64Or(file, "tokenizer.ggml.unknown_token_id", -1));

        if (tokenTypes != nullptr)
        {
            for (std::size_t i = 0; i < tokenTypes->size() && i < tokens.size(); ++i)
            {
                std::int32_t type = static_cast<std::int32_t>((*tokenTypes)[i].asInt32());
                if (type == 3 || type == 2) // control or unknown
                {
                    bool special = (type == 3);
                    tokenizer->addedTokens_.push_back(
                        {static_cast<std::int32_t>(i), tokens[i].asString(), special});
                }
            }
            std::sort(tokenizer->addedTokens_.begin(), tokenizer->addedTokens_.end(),
                      [](const Tokenizer::AddedToken &a, const Tokenizer::AddedToken &b) {
                          return a.content.size() > b.content.size();
                      });
        }

        if (modelName == "gpt2")
        {
            tokenizer->model_ = TokenizerModel::Gpt2ByteLevel;
            tokenizer->hasNormalizer_ = false;
            tokenizer->hasDecoder_ = false;
            tokenizer->byteFallback_ = false;
            // GPT-2 byte-level BPE does not use an automatic post-processor BOS; the
            // chat template / caller is responsible for adding special tokens.
            tokenizer->bosId_ = -1;

            buildGpt2ByteToUnicode(tokenizer->byteToUnicode_);
            for (int b = 0; b < 256; ++b)
            {
                tokenizer->unicodeToByte_[tokenizer->byteToUnicode_[static_cast<std::uint8_t>(b)]] =
                    static_cast<std::uint8_t>(b);
            }
        }
        else
        {
            // SentencePiece-style normalizer/decoder semantics for LLaMA-family GGUF.
            tokenizer->model_ = TokenizerModel::LlamaSentencePiece;
            tokenizer->hasNormalizer_ = true;
            tokenizer->hasDecoder_ = true;
        }

        return tokenizer;
    }

} // namespace systems::leal::campello_llm
