#include <campello_llm/gguf_reader.hpp>

#include <cstring>
#include <fstream>
#include <stdexcept>
#include <type_traits>

#if defined(_WIN32)
#include <windows.h>
#else
#include <fcntl.h>
#include <sys/mman.h>
#include <sys/stat.h>
#include <unistd.h>
#endif

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

    template <typename R>
    GgufValue readValue(R &r, GgufValueType type)
    {
        GgufValue v;
        v.type = type;
        switch (type)
        {
        case GgufValueType::UInt8:
            v.value = r.template read<std::uint8_t>();
            return v;
        case GgufValueType::Int8:
            v.value = r.template read<std::int8_t>();
            return v;
        case GgufValueType::UInt16:
            v.value = r.template read<std::uint16_t>();
            return v;
        case GgufValueType::Int16:
            v.value = r.template read<std::int16_t>();
            return v;
        case GgufValueType::UInt32:
            v.value = r.template read<std::uint32_t>();
            return v;
        case GgufValueType::Int32:
            v.value = r.template read<std::int32_t>();
            return v;
        case GgufValueType::Float32:
            v.value = r.template read<float>();
            return v;
        case GgufValueType::Bool:
            v.value = r.template read<std::uint8_t>() != 0;
            return v;
        case GgufValueType::String:
            v.value = r.readString();
            return v;
        case GgufValueType::UInt64:
            v.value = r.template read<std::uint64_t>();
            return v;
        case GgufValueType::Int64:
            v.value = r.template read<std::int64_t>();
            return v;
        case GgufValueType::Float64:
            v.value = r.template read<double>();
            return v;
        case GgufValueType::Array:
        {
            auto elemType = static_cast<GgufValueType>(r.template read<std::uint32_t>());
            std::uint64_t count = r.template read<std::uint64_t>();
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

    // Stream-based reader for metadata-only inspection. Avoids loading tensor data.
    class FileReader
    {
    public:
        FileReader(std::ifstream &stream, std::size_t fileSize) : stream_(stream), fileSize_(fileSize) {}

        std::size_t pos() const { return pos_; }
        std::size_t size() const { return fileSize_; }

        template <typename T>
        T read()
        {
            static_assert(std::is_trivially_copyable_v<T>);
            require(sizeof(T));
            T value{};
            if (!stream_.read(reinterpret_cast<char *>(&value), sizeof(T)))
            {
                throw std::runtime_error("campello_llm: failed to read gguf file");
            }
            pos_ += sizeof(T);
            return value;
        }

        std::string readString()
        {
            std::uint64_t len = read<std::uint64_t>();
            require(len);
            std::string s;
            s.resize(len);
            if (len > 0 && !stream_.read(s.data(), static_cast<std::streamsize>(len)))
            {
                throw std::runtime_error("campello_llm: failed to read gguf string");
            }
            pos_ += len;
            return s;
        }

    private:
        void require(std::uint64_t n) const
        {
            if (n > fileSize_ - pos_)
            {
                throw std::runtime_error("campello_llm: unexpected end of gguf input");
            }
        }

        std::ifstream &stream_;
        std::size_t fileSize_;
        std::size_t pos_ = 0;
    };

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
        case GgmlQuantizationType::Q4_0:
            return WeightDType::Q4_0;
        case GgmlQuantizationType::Q4_1:
            return WeightDType::Q4_1;
        case GgmlQuantizationType::Q5_0:
            return WeightDType::Q5_0;
        case GgmlQuantizationType::Q5_1:
            return WeightDType::Q5_1;
        case GgmlQuantizationType::Q8_0:
            return WeightDType::Q8_0;
        case GgmlQuantizationType::Q8_1:
            return WeightDType::Q8_1;
        case GgmlQuantizationType::Q2_K:
            return WeightDType::Q2_K;
        case GgmlQuantizationType::Q3_K:
            return WeightDType::Q3_K;
        case GgmlQuantizationType::Q4_K:
            return WeightDType::Q4_K;
        case GgmlQuantizationType::Q5_K:
            return WeightDType::Q5_K;
        case GgmlQuantizationType::Q6_K:
            return WeightDType::Q6_K;
        case GgmlQuantizationType::Q8_K:
            return WeightDType::Q8_K;
        default:
            // IQ* and future block-quantized types are not supported yet.
            throw std::runtime_error("campello_llm: unsupported GGUF tensor type " +
                                     std::to_string(static_cast<std::uint32_t>(t)) +
                                     " for tensor (block-quantized)");
        }
    }

    // Byte length of a GGML tensor given its quantization type and element count.
    // For block-quantized types this is the block size times the number of blocks,
    // not the element count times a per-element size.
    std::size_t ggmlTensorByteLength(GgmlQuantizationType type, std::int64_t nElements)
    {
        switch (type)
        {
        case GgmlQuantizationType::F32:
            return static_cast<std::size_t>(nElements) * 4;
        case GgmlQuantizationType::F16:
        case GgmlQuantizationType::BF16:
        case GgmlQuantizationType::I16:
            return static_cast<std::size_t>(nElements) * 2;
        case GgmlQuantizationType::F64:
        case GgmlQuantizationType::I64:
            return static_cast<std::size_t>(nElements) * 8;
        case GgmlQuantizationType::I32:
            return static_cast<std::size_t>(nElements) * 4;
        case GgmlQuantizationType::I8:
            return static_cast<std::size_t>(nElements);
        case GgmlQuantizationType::Q4_0:
        {
            if (nElements % 32 != 0)
            {
                throw std::runtime_error("campello_llm: Q4_0 tensor element count is not a multiple of 32");
            }
            return static_cast<std::size_t>(nElements / 32) * 18;
        }
        case GgmlQuantizationType::Q8_0:
        {
            if (nElements % 32 != 0)
            {
                throw std::runtime_error("campello_llm: Q8_0 tensor element count is not a multiple of 32");
            }
            return static_cast<std::size_t>(nElements / 32) * 34;
        }
        case GgmlQuantizationType::Q4_1:
        {
            if (nElements % 32 != 0)
            {
                throw std::runtime_error("campello_llm: Q4_1 tensor element count is not a multiple of 32");
            }
            return static_cast<std::size_t>(nElements / 32) * 20;
        }
        case GgmlQuantizationType::Q5_0:
        {
            if (nElements % 32 != 0)
            {
                throw std::runtime_error("campello_llm: Q5_0 tensor element count is not a multiple of 32");
            }
            return static_cast<std::size_t>(nElements / 32) * 22;
        }
        case GgmlQuantizationType::Q5_1:
        {
            if (nElements % 32 != 0)
            {
                throw std::runtime_error("campello_llm: Q5_1 tensor element count is not a multiple of 32");
            }
            return static_cast<std::size_t>(nElements / 32) * 24;
        }
        case GgmlQuantizationType::Q8_1:
        {
            if (nElements % 32 != 0)
            {
                throw std::runtime_error("campello_llm: Q8_1 tensor element count is not a multiple of 32");
            }
            return static_cast<std::size_t>(nElements / 32) * 36;
        }
        case GgmlQuantizationType::Q2_K:
        {
            if (nElements % 256 != 0)
            {
                throw std::runtime_error("campello_llm: Q2_K tensor element count is not a multiple of 256");
            }
            return static_cast<std::size_t>(nElements / 256) * 84;
        }
        case GgmlQuantizationType::Q3_K:
        {
            if (nElements % 256 != 0)
            {
                throw std::runtime_error("campello_llm: Q3_K tensor element count is not a multiple of 256");
            }
            return static_cast<std::size_t>(nElements / 256) * 110;
        }
        case GgmlQuantizationType::Q4_K:
        {
            if (nElements % 256 != 0)
            {
                throw std::runtime_error("campello_llm: Q4_K tensor element count is not a multiple of 256");
            }
            return static_cast<std::size_t>(nElements / 256) * 144;
        }
        case GgmlQuantizationType::Q5_K:
        {
            if (nElements % 256 != 0)
            {
                throw std::runtime_error("campello_llm: Q5_K tensor element count is not a multiple of 256");
            }
            return static_cast<std::size_t>(nElements / 256) * 176;
        }
        case GgmlQuantizationType::Q6_K:
        {
            if (nElements % 256 != 0)
            {
                throw std::runtime_error("campello_llm: Q6_K tensor element count is not a multiple of 256");
            }
            return static_cast<std::size_t>(nElements / 256) * 210;
        }
        case GgmlQuantizationType::Q8_K:
        {
            if (nElements % 256 != 0)
            {
                throw std::runtime_error("campello_llm: Q8_K tensor element count is not a multiple of 256");
            }
            return static_cast<std::size_t>(nElements / 256) * 292;
        }
        default:
            throw std::runtime_error("campello_llm: unsupported gguf tensor type for byte-length calculation");
        }
    }

    struct PendingTensor
    {
        std::string name;
        std::vector<std::int64_t> shape;
        GgmlQuantizationType ggmlType;
        std::uint64_t relativeOffset;
    };

    // Everything parseGgufHeader() extracts before the tensor *data* region --
    // returned by value (rather than written into a GgufFile's private members
    // directly) so this helper doesn't need friend access; the two public loader
    // functions below (which are friends) assign these into the GgufFile they own.
    struct ParsedGgufHeader
    {
        std::unordered_map<std::string, GgufValue> metadata;
        std::vector<PendingTensor> pending;
        std::size_t dataStart = 0;
    };

    // Parses the gguf magic/version/metadata/tensor-descriptor header out of
    // `bytes`/`size`. Shared by loadGgufFromMemory() (which then copies the tensor
    // data region that starts at the returned dataStart) and loadGgufFromFile()
    // (which maps it instead) -- everything up to dataStart is identical either way.
    ParsedGgufHeader parseGgufHeader(const std::uint8_t *bytes, std::size_t size)
    {
        Reader r(bytes, size);
        ParsedGgufHeader header;

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

        for (std::uint64_t i = 0; i < kvCount; ++i)
        {
            std::string key = r.readString();
            auto type = static_cast<GgufValueType>(r.read<std::uint32_t>());
            GgufValue value = readValue(r, type);
            header.metadata.emplace(std::move(key), std::move(value));
        }

        std::uint32_t alignment = kGgufDefaultAlignment;
        auto alignIt = header.metadata.find("general.alignment");
        if (alignIt != header.metadata.end())
        {
            const GgufValue &alignVal = alignIt->second;
            if (alignVal.type != GgufValueType::UInt32)
            {
                throw std::runtime_error("campello_llm: gguf general.alignment has unexpected type");
            }
            alignment = alignVal.asUInt32();
            if (alignment == 0 || (alignment & (alignment - 1)) != 0)
            {
                throw std::runtime_error("campello_llm: gguf general.alignment must be a non-zero power of two");
            }
        }

        header.pending.reserve(tensorCount);
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
            header.pending.push_back(std::move(t));
        }

        std::size_t headerEnd = r.pos();
        std::size_t padding = headerEnd % alignment;
        header.dataStart = padding == 0 ? headerEnd : headerEnd + (alignment - padding);
        if (header.dataStart > size)
        {
            throw std::runtime_error("campello_llm: gguf tensor data offset exceeds file size");
        }
        return header;
    }

    // Builds the TensorInfo list from `pending`, with each entry's `data` computed
    // as `dataBase + relativeOffset`. `dataBase` must stay valid for the owning
    // GgufFile's whole lifetime -- either file->buffer_.data() (the copying path)
    // or the mmap base (loadGgufFromFile()'s path). Returned by value for the same
    // friend-access reason as ParsedGgufHeader above.
    std::vector<TensorInfo> populateTensors(const std::vector<PendingTensor> &pending, const std::uint8_t *dataBase,
                                            std::size_t dataSize)
    {
        std::vector<TensorInfo> tensors;
        tensors.reserve(pending.size());
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
            info.byteLength = ggmlTensorByteLength(t.ggmlType, nElements);

            if (t.relativeOffset > dataSize || info.byteLength > dataSize - t.relativeOffset)
            {
                throw std::runtime_error("campello_llm: gguf tensor '" + t.name + "' data extends past end of file");
            }
            info.data = dataBase + t.relativeOffset;
            tensors.push_back(std::move(info));
        }
        return tensors;
    }

    // Read-only whole-file memory mapping, returned as a type-erased
    // std::shared_ptr<void> (see GgufFile::mappedFile_) so callers don't need any
    // platform-specific mapping type -- `out` carries the pointer/size, and the
    // shared_ptr's custom deleter unmaps when the last reference (the owning
    // GgufFile) is destroyed.
    struct FileMapping
    {
        const std::uint8_t *data = nullptr;
        std::size_t size = 0;
    };

#if defined(_WIN32)
    std::shared_ptr<void> mapFileReadOnly(const std::string &path, FileMapping &out)
    {
        HANDLE file = CreateFileA(path.c_str(), GENERIC_READ, FILE_SHARE_READ, nullptr, OPEN_EXISTING,
                                   FILE_ATTRIBUTE_NORMAL, nullptr);
        if (file == INVALID_HANDLE_VALUE)
        {
            throw std::runtime_error("campello_llm: cannot open '" + path + "'");
        }

        LARGE_INTEGER fileSize;
        if (!GetFileSizeEx(file, &fileSize))
        {
            CloseHandle(file);
            throw std::runtime_error("campello_llm: cannot determine size of '" + path + "'");
        }
        std::size_t size = static_cast<std::size_t>(fileSize.QuadPart);

        if (size == 0)
        {
            CloseHandle(file);
            out.data = nullptr;
            out.size = 0;
            return nullptr;
        }

        HANDLE mapping = CreateFileMappingA(file, nullptr, PAGE_READONLY, 0, 0, nullptr);
        CloseHandle(file); // the mapping keeps the underlying file alive
        if (!mapping)
        {
            throw std::runtime_error("campello_llm: CreateFileMapping failed for '" + path + "'");
        }
        void *base = MapViewOfFile(mapping, FILE_MAP_READ, 0, 0, size);
        if (!base)
        {
            CloseHandle(mapping);
            throw std::runtime_error("campello_llm: MapViewOfFile failed for '" + path + "'");
        }

        out.data = static_cast<const std::uint8_t *>(base);
        out.size = size;
        return std::shared_ptr<void>(base, [mapping](void *p) {
            UnmapViewOfFile(p);
            CloseHandle(mapping);
        });
    }
#else
    std::shared_ptr<void> mapFileReadOnly(const std::string &path, FileMapping &out)
    {
        int fd = open(path.c_str(), O_RDONLY);
        if (fd < 0)
        {
            throw std::runtime_error("campello_llm: cannot open '" + path + "'");
        }

        struct stat st{};
        if (fstat(fd, &st) != 0)
        {
            close(fd);
            throw std::runtime_error("campello_llm: cannot determine size of '" + path + "'");
        }
        std::size_t size = static_cast<std::size_t>(st.st_size);

        if (size == 0)
        {
            close(fd);
            out.data = nullptr;
            out.size = 0;
            return nullptr;
        }

        void *base = mmap(nullptr, size, PROT_READ, MAP_PRIVATE, fd, 0);
        close(fd); // the mapping keeps the file's contents accessible; fd isn't needed afterward
        if (base == MAP_FAILED)
        {
            throw std::runtime_error("campello_llm: mmap failed for '" + path + "'");
        }

        out.data = static_cast<const std::uint8_t *>(base);
        out.size = size;
        return std::shared_ptr<void>(base, [size](void *p) { munmap(p, size); });
    }
#endif

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
        ParsedGgufHeader header = parseGgufHeader(bytes, size);

        auto file = std::make_unique<GgufFile>();
        file->metadata_ = std::move(header.metadata);
        file->buffer_.assign(bytes + header.dataStart, bytes + size);
        file->tensors_ = populateTensors(header.pending, file->buffer_.data(), file->buffer_.size());
        for (std::size_t i = 0; i < file->tensors_.size(); ++i)
        {
            file->nameToIndex_.emplace(file->tensors_[i].name, i);
        }

        return file;
    }

    std::unique_ptr<GgufFile> loadGgufFromFile(const std::string &path)
    {
        FileMapping mapping;
        std::shared_ptr<void> mappedHandle = mapFileReadOnly(path, mapping);
        ParsedGgufHeader header = parseGgufHeader(mapping.data, mapping.size);

        auto file = std::make_unique<GgufFile>();
        file->metadata_ = std::move(header.metadata);
        // No copy: every TensorInfo::data below points straight into the mapping,
        // which mappedFile_ keeps alive for as long as this GgufFile exists (see
        // its doc comment in gguf_reader.hpp).
        file->mappedFile_ = mappedHandle;
        file->tensors_ =
            populateTensors(header.pending, mapping.data + header.dataStart, mapping.size - header.dataStart);
        for (std::size_t i = 0; i < file->tensors_.size(); ++i)
        {
            file->nameToIndex_.emplace(file->tensors_[i].name, i);
        }

        return file;
    }

    std::unique_ptr<GgufFileInfo> loadGgufInfoFromFile(const std::string &path)
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
        const std::size_t fileSize = static_cast<std::size_t>(size);

        FileReader r(stream, fileSize);

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

        auto file = std::make_unique<GgufFileInfo>();

        for (std::uint64_t i = 0; i < kvCount; ++i)
        {
            std::string key = r.readString();
            auto type = static_cast<GgufValueType>(r.read<std::uint32_t>());
            GgufValue value = readValue(r, type);
            file->metadata.emplace(std::move(key), std::move(value));
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
        if (dataStart > fileSize)
        {
            throw std::runtime_error("campello_llm: gguf tensor data offset exceeds file size");
        }

        for (const PendingTensor &t : pending)
        {
            GgufTensorInfo info;
            info.name = t.name;
            info.dtype = dtypeFromGgml(t.ggmlType);
            info.shape = t.shape;

            std::int64_t nElements = 1;
            for (std::int64_t dim : t.shape)
            {
                nElements *= dim;
            }
            info.byteLength = ggmlTensorByteLength(t.ggmlType, nElements);

            if (t.relativeOffset > fileSize - dataStart || info.byteLength > fileSize - dataStart - t.relativeOffset)
            {
                throw std::runtime_error("campello_llm: gguf tensor '" + t.name + "' data extends past end of file");
            }

            file->tensors.push_back(std::move(info));
        }

        return file;
    }

} // namespace systems::leal::campello_llm
