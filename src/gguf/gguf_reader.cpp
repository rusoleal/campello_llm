#include <campello_llm/gguf_reader.hpp>

#include <cstring>
#include <fstream>
#include <stdexcept>
#include <type_traits>

using namespace systems::leal::campello_llm;

namespace
{
    // gguf spec constants, verified against the real `gguf` Python package
    // (gguf/constants.py) before being trusted — same discipline campello_nn's
    // ONNX/TFLite importers used for protobuf/FlatBuffers field numbers.
    constexpr std::uint32_t kGgufMagic = 0x46554747; // bytes "GGUF" read as little-endian u32
    constexpr std::uint32_t kGgufDefaultAlignment = 32;

    enum class GgmlQuantizationType : std::uint32_t
    {
        F32 = 0,
        F16 = 1,
        Q4_0 = 2,
        Q4_1 = 3,
        Q5_0 = 6,
        Q5_1 = 7,
        Q8_0 = 8,
        Q8_1 = 9,
        Q2_K = 10,
        Q3_K = 11,
        Q4_K = 12,
        Q5_K = 13,
        Q6_K = 14,
        Q8_K = 15,
        I8 = 24,
        I16 = 25,
        I32 = 26,
        I64 = 27,
        F64 = 28,
        BF16 = 30,
    };

    // Bounds-checked little-endian binary cursor over an in-memory gguf file —
    // same "every read is bounds-checked, throw rather than guess" discipline as
    // campello_nn's onnx `proto_reader.hpp`.
    class Reader
    {
    public:
        Reader(const std::uint8_t *data, std::size_t size) : data_(data), size_(size) {}

        std::size_t pos() const { return pos_; }

        template <typename T>
        T read()
        {
            static_assert(std::is_trivially_copyable_v<T>);
            require(sizeof(T));
            T value;
            std::memcpy(&value, data_ + pos_, sizeof(T));
            pos_ += sizeof(T);
            return value;
        }

        std::string readString()
        {
            std::uint64_t len = read<std::uint64_t>();
            require(len);
            std::string s(reinterpret_cast<const char *>(data_ + pos_), len);
            pos_ += len;
            return s;
        }

        const std::uint8_t *ptr(std::size_t offset) const
        {
            if (offset > size_)
            {
                throw std::runtime_error("campello_llm: gguf offset out of range");
            }
            return data_ + offset;
        }

        std::size_t size() const { return size_; }

        void seek(std::size_t offset)
        {
            if (offset > size_)
            {
                throw std::runtime_error("campello_llm: gguf seek out of range");
            }
            pos_ = offset;
        }

    private:
        void require(std::uint64_t n) const
        {
            if (n > size_ - pos_)
            {
                throw std::runtime_error("campello_llm: unexpected end of gguf input");
            }
        }

        const std::uint8_t *data_;
        std::size_t size_;
        std::size_t pos_ = 0;
    };

    GgufValue readValue(Reader &r, GgufValueType type);

    GgufValue readValue(Reader &r, GgufValueType type)
    {
        GgufValue v;
        v.type = type;
        switch (type)
        {
        case GgufValueType::UInt8:
            v.value = r.read<std::uint8_t>();
            return v;
        case GgufValueType::Int8:
            v.value = r.read<std::int8_t>();
            return v;
        case GgufValueType::UInt16:
            v.value = r.read<std::uint16_t>();
            return v;
        case GgufValueType::Int16:
            v.value = r.read<std::int16_t>();
            return v;
        case GgufValueType::UInt32:
            v.value = r.read<std::uint32_t>();
            return v;
        case GgufValueType::Int32:
            v.value = r.read<std::int32_t>();
            return v;
        case GgufValueType::Float32:
            v.value = r.read<float>();
            return v;
        case GgufValueType::Bool:
            v.value = r.read<std::uint8_t>() != 0;
            return v;
        case GgufValueType::String:
            v.value = r.readString();
            return v;
        case GgufValueType::UInt64:
            v.value = r.read<std::uint64_t>();
            return v;
        case GgufValueType::Int64:
            v.value = r.read<std::int64_t>();
            return v;
        case GgufValueType::Float64:
            v.value = r.read<double>();
            return v;
        case GgufValueType::Array:
        {
            auto elemType = static_cast<GgufValueType>(r.read<std::uint32_t>());
            std::uint64_t count = r.read<std::uint64_t>();
            v.arrayElementType = elemType;
            std::vector<GgufValue> elements;
            elements.reserve(count);
            for (std::uint64_t i = 0; i < count; ++i)
            {
                elements.push_back(readValue(r, elemType));
            }
            v.value = std::move(elements);
            return v;
        }
        }
        throw std::runtime_error("campello_llm: unknown gguf value type");
    }

    WeightDType dtypeFromGgml(GgmlQuantizationType t)
    {
        switch (t)
        {
        case GgmlQuantizationType::F32:
            return WeightDType::F32;
        case GgmlQuantizationType::F16:
            return WeightDType::F16;
        case GgmlQuantizationType::BF16:
            return WeightDType::BF16;
        case GgmlQuantizationType::F64:
            return WeightDType::F64;
        case GgmlQuantizationType::I8:
            return WeightDType::I8;
        case GgmlQuantizationType::I16:
            return WeightDType::I16;
        case GgmlQuantizationType::I32:
            return WeightDType::I32;
        case GgmlQuantizationType::I64:
            return WeightDType::I64;
        default:
            // Block-quantized types (Q4_0, Q8_0, the _K variants, ...) have no
            // campello_llm dequantization path yet — throw rather than guess one.
            throw std::runtime_error("campello_llm: unsupported (block-quantized) gguf tensor type");
        }
    }

} // namespace

namespace systems::leal::campello_llm
{

    const std::vector<TensorInfo> &GgufFile::tensors() const
    {
        return tensors_;
    }

    const TensorInfo *GgufFile::find(const std::string &name) const
    {
        auto it = nameToIndex_.find(name);
        if (it == nameToIndex_.end())
        {
            return nullptr;
        }
        return &tensors_[it->second];
    }

    const std::unordered_map<std::string, GgufValue> &GgufFile::metadata() const
    {
        return metadata_;
    }

    const GgufValue *GgufFile::findMetadata(const std::string &key) const
    {
        auto it = metadata_.find(key);
        if (it == metadata_.end())
        {
            return nullptr;
        }
        return &it->second;
    }

    std::unique_ptr<GgufFile> loadGgufFromMemory(const void *data, std::size_t size)
    {
        const auto *bytes = static_cast<const std::uint8_t *>(data);
        Reader r(bytes, size);

        if (r.read<std::uint32_t>() != kGgufMagic)
        {
            throw std::runtime_error("campello_llm: gguf magic invalid");
        }
        std::uint32_t version = r.read<std::uint32_t>();
        if (version != 2 && version != 3)
        {
            throw std::runtime_error("campello_llm: unsupported gguf version " + std::to_string(version));
        }
        std::uint64_t tensorCount = r.read<std::uint64_t>();
        std::uint64_t kvCount = r.read<std::uint64_t>();

        auto file = std::make_unique<GgufFile>();

        for (std::uint64_t i = 0; i < kvCount; ++i)
        {
            std::string key = r.readString();
            auto type = static_cast<GgufValueType>(r.read<std::uint32_t>());
            GgufValue value = readValue(r, type);
            file->metadata_.emplace(std::move(key), std::move(value));
        }

        std::uint32_t alignment = kGgufDefaultAlignment;
        if (const GgufValue *alignVal = file->findMetadata("general.alignment"))
        {
            if (alignVal->type != GgufValueType::UInt32)
            {
                throw std::runtime_error("campello_llm: gguf general.alignment has unexpected type");
            }
            alignment = alignVal->asUInt32();
            if (alignment == 0 || (alignment & (alignment - 1)) != 0)
            {
                throw std::runtime_error("campello_llm: gguf general.alignment must be a non-zero power of two");
            }
        }

        struct PendingTensor
        {
            std::string name;
            std::vector<std::int64_t> shape;
            GgmlQuantizationType ggmlType;
            std::uint64_t relativeOffset;
        };
        std::vector<PendingTensor> pending;
        pending.reserve(tensorCount);

        for (std::uint64_t i = 0; i < tensorCount; ++i)
        {
            PendingTensor t;
            t.name = r.readString();
            std::uint32_t nDims = r.read<std::uint32_t>();
            t.shape.reserve(nDims);
            for (std::uint32_t d = 0; d < nDims; ++d)
            {
                t.shape.push_back(static_cast<std::int64_t>(r.read<std::uint64_t>()));
            }
            t.ggmlType = static_cast<GgmlQuantizationType>(r.read<std::uint32_t>());
            t.relativeOffset = r.read<std::uint64_t>();
            pending.push_back(std::move(t));
        }

        std::size_t headerEnd = r.pos();
        std::size_t padding = headerEnd % alignment;
        std::size_t dataStart = padding == 0 ? headerEnd : headerEnd + (alignment - padding);
        if (dataStart > size)
        {
            throw std::runtime_error("campello_llm: gguf tensor data offset exceeds file size");
        }

        file->buffer_.assign(bytes + dataStart, bytes + size);

        for (const PendingTensor &t : pending)
        {
            TensorInfo info;
            info.name = t.name;
            info.dtype = dtypeFromGgml(t.ggmlType);
            info.shape = t.shape;

            std::int64_t nElements = 1;
            for (std::int64_t dim : t.shape)
            {
                nElements *= dim;
            }
            info.byteLength = static_cast<std::size_t>(nElements) * weightDTypeSize(info.dtype);

            if (t.relativeOffset > file->buffer_.size() || info.byteLength > file->buffer_.size() - t.relativeOffset)
            {
                throw std::runtime_error("campello_llm: gguf tensor '" + t.name + "' data extends past end of file");
            }
            info.data = file->buffer_.data() + t.relativeOffset;

            file->nameToIndex_.emplace(t.name, file->tensors_.size());
            file->tensors_.push_back(std::move(info));
        }

        return file;
    }

    std::unique_ptr<GgufFile> loadGgufFromFile(const std::string &path)
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

        return loadGgufFromMemory(buffer.data(), buffer.size());
    }

} // namespace systems::leal::campello_llm
