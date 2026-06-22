#include <campello_llm/safetensors_reader.hpp>

#include <cstring>
#include <fstream>
#include <stdexcept>

#include "json_value.hpp"

using namespace systems::leal::campello_llm;
using namespace systems::leal::campello_llm::internal;

namespace
{
    WeightDType dtypeFromString(const std::string &s)
    {
        if (s == "F32")
            return WeightDType::F32;
        if (s == "F16")
            return WeightDType::F16;
        if (s == "BF16")
            return WeightDType::BF16;
        if (s == "F64")
            return WeightDType::F64;
        if (s == "I8")
            return WeightDType::I8;
        if (s == "I16")
            return WeightDType::I16;
        if (s == "I32")
            return WeightDType::I32;
        if (s == "I64")
            return WeightDType::I64;
        if (s == "U8")
            return WeightDType::U8;
        if (s == "U16")
            return WeightDType::U16;
        if (s == "U32")
            return WeightDType::U32;
        if (s == "U64")
            return WeightDType::U64;
        if (s == "BOOL")
            return WeightDType::Bool;
        // Sub-byte (F4/F6_*) and exotic float8 (F8_*) / complex (C64) types exist in
        // the safetensors spec but have no campello_llm representation yet — throw
        // rather than guess a conversion.
        throw std::runtime_error("campello_llm: unsupported safetensors dtype '" + s + "'");
    }

    std::int64_t expectInt(const JsonValue &v)
    {
        if (v.type != JsonType::Number)
        {
            throw std::runtime_error("campello_llm: expected a number in safetensors header");
        }
        return static_cast<std::int64_t>(v.numberValue);
    }

} // namespace

namespace systems::leal::campello_llm
{

    const std::vector<TensorInfo> &SafetensorsFile::tensors() const
    {
        return tensors_;
    }

    const TensorInfo *SafetensorsFile::find(const std::string &name) const
    {
        auto it = nameToIndex_.find(name);
        if (it == nameToIndex_.end())
        {
            return nullptr;
        }
        return &tensors_[it->second];
    }

    const std::unordered_map<std::string, std::string> &SafetensorsFile::metadata() const
    {
        return metadata_;
    }

    std::unique_ptr<SafetensorsFile> loadSafetensorsFromMemory(const void *data, std::size_t size)
    {
        if (size < 8)
        {
            throw std::runtime_error("campello_llm: safetensors file too small for header length");
        }
        const auto *bytes = static_cast<const std::uint8_t *>(data);

        std::uint64_t headerLen;
        std::memcpy(&headerLen, bytes, sizeof(headerLen));
        // Safetensors header length is always little-endian, regardless of host
        // byte order (the format's only fixed-width binary field).
        static_assert(sizeof(headerLen) == 8);

        if (headerLen > size - 8)
        {
            throw std::runtime_error("campello_llm: safetensors header length exceeds file size");
        }

        JsonValue header = parseJson(reinterpret_cast<const char *>(bytes + 8), headerLen);
        if (header.type != JsonType::Object)
        {
            throw std::runtime_error("campello_llm: safetensors header is not a JSON object");
        }

        std::size_t dataStart = 8 + static_cast<std::size_t>(headerLen);
        std::size_t dataSize = size - dataStart;

        auto file = std::make_unique<SafetensorsFile>();
        file->buffer_.assign(bytes + dataStart, bytes + dataStart + dataSize);

        for (const auto &entry : header.objectValue)
        {
            const std::string &name = entry.first;
            const JsonValue &value = entry.second;

            if (name == "__metadata__")
            {
                if (value.type != JsonType::Object)
                {
                    throw std::runtime_error("campello_llm: safetensors __metadata__ is not a JSON object");
                }
                for (const auto &metaEntry : value.objectValue)
                {
                    if (metaEntry.second.type != JsonType::String)
                    {
                        throw std::runtime_error("campello_llm: safetensors __metadata__ value is not a string");
                    }
                    file->metadata_.emplace(metaEntry.first, metaEntry.second.stringValue);
                }
                continue;
            }

            const JsonValue *dtypeVal = value.find("dtype");
            const JsonValue *shapeVal = value.find("shape");
            const JsonValue *offsetsVal = value.find("data_offsets");
            if (dtypeVal == nullptr || dtypeVal->type != JsonType::String ||
                shapeVal == nullptr || shapeVal->type != JsonType::Array ||
                offsetsVal == nullptr || offsetsVal->type != JsonType::Array || offsetsVal->arrayValue.size() != 2)
            {
                throw std::runtime_error("campello_llm: malformed safetensors tensor entry '" + name + "'");
            }

            TensorInfo info;
            info.name = name;
            info.dtype = dtypeFromString(dtypeVal->stringValue);
            info.shape.reserve(shapeVal->arrayValue.size());
            for (const JsonValue &dim : shapeVal->arrayValue)
            {
                info.shape.push_back(expectInt(dim));
            }

            std::int64_t begin = expectInt(offsetsVal->arrayValue[0]);
            std::int64_t end = expectInt(offsetsVal->arrayValue[1]);
            if (begin < 0 || end < begin || static_cast<std::size_t>(end) > dataSize)
            {
                throw std::runtime_error("campello_llm: safetensors data_offsets out of range for tensor '" + name + "'");
            }
            info.data = file->buffer_.data() + begin;
            info.byteLength = static_cast<std::size_t>(end - begin);

            file->nameToIndex_.emplace(name, file->tensors_.size());
            file->tensors_.push_back(std::move(info));
        }

        return file;
    }

    std::unique_ptr<SafetensorsFile> loadSafetensorsFromFile(const std::string &path)
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

        return loadSafetensorsFromMemory(buffer.data(), buffer.size());
    }

} // namespace systems::leal::campello_llm
