#include <campello_llm/architecture.hpp>

#include <fstream>
#include <stdexcept>

#include "../json/json_value.hpp"

using namespace systems::leal::campello_llm;
using namespace systems::leal::campello_llm::internal;

namespace
{
    const JsonValue &requireField(const JsonValue &root, const char *key)
    {
        const JsonValue *v = root.find(key);
        if (v == nullptr || v->type != JsonType::Number)
        {
            throw std::runtime_error(std::string("campello_llm: config.json missing numeric field '") + key + "'");
        }
        return *v;
    }
} // namespace

namespace systems::leal::campello_llm
{

    LlamaConfig loadLlamaConfigFromMemory(const void *data, std::size_t size)
    {
        JsonValue root = parseJson(static_cast<const char *>(data), size);
        if (root.type != JsonType::Object)
        {
            throw std::runtime_error("campello_llm: config.json is not a JSON object");
        }

        const JsonValue *modelType = root.find("model_type");
        if (modelType == nullptr || modelType->type != JsonType::String || modelType->stringValue != "llama")
        {
            throw std::runtime_error("campello_llm: config.json model_type is not 'llama'");
        }

        LlamaConfig config;
        config.vocabSize = static_cast<std::int64_t>(requireField(root, "vocab_size").numberValue);
        config.hiddenSize = static_cast<std::int64_t>(requireField(root, "hidden_size").numberValue);
        config.numLayers = static_cast<std::int64_t>(requireField(root, "num_hidden_layers").numberValue);
        config.numAttentionHeads = static_cast<std::int64_t>(requireField(root, "num_attention_heads").numberValue);
        config.intermediateSize = static_cast<std::int64_t>(requireField(root, "intermediate_size").numberValue);
        config.rmsNormEps = static_cast<float>(requireField(root, "rms_norm_eps").numberValue);

        const JsonValue *numKvHeads = root.find("num_key_value_heads");
        config.numKeyValueHeads = (numKvHeads != nullptr && numKvHeads->type == JsonType::Number)
                                       ? static_cast<std::int64_t>(numKvHeads->numberValue)
                                       : config.numAttentionHeads;

        const JsonValue *ropeTheta = root.find("rope_theta");
        config.ropeTheta =
            (ropeTheta != nullptr && ropeTheta->type == JsonType::Number) ? static_cast<float>(ropeTheta->numberValue) : 10000.0f;

        return config;
    }

    LlamaConfig loadLlamaConfigFromFile(const std::string &path)
    {
        std::ifstream stream(path, std::ios::binary | std::ios::ate);
        if (!stream)
        {
            throw std::runtime_error("campello_llm: cannot open '" + path + "'");
        }
        std::streamsize size = stream.tellg();
        if (size < 0)
        {
            throw std::runtime_error("campello_llm: cannot determine size of '" + path + "'");
        }
        stream.seekg(0, std::ios::beg);

        std::vector<char> buffer(static_cast<std::size_t>(size));
        if (size > 0 && !stream.read(buffer.data(), size))
        {
            throw std::runtime_error("campello_llm: failed to read '" + path + "'");
        }

        return loadLlamaConfigFromMemory(buffer.data(), buffer.size());
    }

} // namespace systems::leal::campello_llm
