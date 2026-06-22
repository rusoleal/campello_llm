#pragma once

#include <cstdint>
#include <memory>
#include <string>
#include <unordered_map>
#include <variant>
#include <vector>

#include <campello_llm/weights_file.hpp>

namespace systems::leal::campello_llm
{

    /**
     * @brief gguf key/value metadata value type (`GGUFValueType` in the gguf spec).
     */
    enum class GgufValueType
    {
        UInt8,
        Int8,
        UInt16,
        Int16,
        UInt32,
        Int32,
        Float32,
        Bool,
        String,
        Array,
        UInt64,
        Int64,
        Float64,
    };

    struct GgufValue;

    /**
     * @brief One gguf metadata value. Scalars and strings hold their value directly;
     * `Array` holds a homogeneous (per the gguf spec) vector of `GgufValue` of
     * `arrayElementType`.
     *
     * Hyperparameters needed by the Phase 3 architecture registry (e.g.
     * `llama.block_count`, `llama.embedding_length`) are read through this generic
     * value rather than a bespoke struct per architecture — there's no fixed schema
     * gguf itself enforces beyond key naming convention.
     */
    struct GgufValue
    {
        GgufValueType type;
        GgufValueType arrayElementType = GgufValueType::UInt8; // meaningful only when type == Array
        std::variant<
            std::uint8_t, std::int8_t, std::uint16_t, std::int16_t,
            std::uint32_t, std::int32_t, std::uint64_t, std::int64_t,
            float, double, bool, std::string, std::vector<GgufValue>>
            value;

        std::uint8_t asUInt8() const { return std::get<std::uint8_t>(value); }
        std::int8_t asInt8() const { return std::get<std::int8_t>(value); }
        std::uint16_t asUInt16() const { return std::get<std::uint16_t>(value); }
        std::int16_t asInt16() const { return std::get<std::int16_t>(value); }
        std::uint32_t asUInt32() const { return std::get<std::uint32_t>(value); }
        std::int32_t asInt32() const { return std::get<std::int32_t>(value); }
        std::uint64_t asUInt64() const { return std::get<std::uint64_t>(value); }
        std::int64_t asInt64() const { return std::get<std::int64_t>(value); }
        float asFloat32() const { return std::get<float>(value); }
        double asFloat64() const { return std::get<double>(value); }
        bool asBool() const { return std::get<bool>(value); }
        const std::string &asString() const { return std::get<std::string>(value); }
        const std::vector<GgufValue> &asArray() const { return std::get<std::vector<GgufValue>>(value); }
    };

    /**
     * @brief A parsed `.gguf` file: magic + version, a key/value metadata block
     * (architecture name, hyperparameters, and — for models that embed their own
     * tokenizer — `tokenizer.ggml.*` vocab/merges entries, exposed here as ordinary
     * metadata, not a separate API), a tensor info table, and the tensor data itself.
     *
     * Only non-block-quantized tensor types (F32/F16/BF16/F64/I8/I16/I32/I64) are
     * supported for now — quantized block formats (Q4_0, Q8_0, the `_K`/`IQ*`
     * variants, ...) throw rather than guess at a dequantization scheme; revisit once
     * a real quantized model needs one (see `TODO.md`).
     */
    class GgufFile : public WeightsFile
    {
    public:
        const std::vector<TensorInfo> &tensors() const override;

        const TensorInfo *find(const std::string &name) const override;

        const std::unordered_map<std::string, GgufValue> &metadata() const;

        const GgufValue *findMetadata(const std::string &key) const;

    private:
        friend std::unique_ptr<GgufFile> loadGgufFromMemory(const void *data, std::size_t size);

        std::vector<std::uint8_t> buffer_;
        std::vector<TensorInfo> tensors_;
        std::unordered_map<std::string, std::size_t> nameToIndex_;
        std::unordered_map<std::string, GgufValue> metadata_;
    };

    /**
     * @brief Parses a `.gguf` file already loaded into memory.
     *
     * Copies `data`/`size` into the returned `GgufFile`'s own buffer (callers are free
     * to discard `data` afterwards) — see Android `AAssetManager` rationale in
     * `CLAUDE.md`.
     */
    std::unique_ptr<GgufFile> loadGgufFromMemory(const void *data, std::size_t size);

    /**
     * @brief Reads `path` into memory and calls `loadGgufFromMemory()`.
     */
    std::unique_ptr<GgufFile> loadGgufFromFile(const std::string &path);

} // namespace systems::leal::campello_llm
