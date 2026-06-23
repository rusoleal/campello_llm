#include <campello_llm/tokenizer_config.hpp>

#include <fstream>
#include <stdexcept>

#include "../json/json_value.hpp"

using namespace systems::leal::campello_llm;
using namespace systems::leal::campello_llm::internal;

namespace
{
    // `bos_token`/`eos_token`/etc. are usually a plain string, but some configs store
    // them as an object (the same shape as an `added_tokens_decoder` entry) with a
    // `"content"` field carrying the actual text — support both, default to empty
    // for anything else rather than throwing (this is auxiliary metadata, not a
    // format campello_llm strictly depends on existing).
    std::string extractTokenText(const JsonValue *value)
    {
        if (value == nullptr)
        {
            return "";
        }
        if (value->type == JsonType::String)
        {
            return value->stringValue;
        }
        if (value->type == JsonType::Object)
        {
            const JsonValue *content = value->find("content");
            if (content != nullptr && content->type == JsonType::String)
            {
                return content->stringValue;
            }
        }
        return "";
    }

} // namespace

namespace systems::leal::campello_llm
{

    SpecialTokenStrings loadSpecialTokenStringsFromMemory(const void *data, std::size_t size)
    {
        JsonValue root = parseJson(static_cast<const char *>(data), size);
        if (root.type != JsonType::Object)
        {
            throw std::runtime_error("campello_llm: tokenizer_config.json is not a JSON object");
        }

        SpecialTokenStrings result;
        result.bosToken = extractTokenText(root.find("bos_token"));
        result.eosToken = extractTokenText(root.find("eos_token"));
        result.unkToken = extractTokenText(root.find("unk_token"));
        result.padToken = extractTokenText(root.find("pad_token"));
        return result;
    }

    SpecialTokenStrings loadSpecialTokenStringsFromFile(const std::string &path)
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

        return loadSpecialTokenStringsFromMemory(buffer.data(), buffer.size());
    }

} // namespace systems::leal::campello_llm
