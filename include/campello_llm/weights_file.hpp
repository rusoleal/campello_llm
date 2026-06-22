#pragma once

#include <cstddef>
#include <cstdint>
#include <stdexcept>
#include <string>
#include <vector>

namespace systems::leal::campello_llm
{

    /**
     * @brief Element type of a tensor as stored in a weights file.
     *
     * This is deliberately its own enum, not `campello_nn::DataType` — weights files
     * (safetensors, gguf) carry a wider set of element types (Int16/Int64/Float64/
     * BFloat16/Bool) than campello_nn's graph IR represents. Mapping down to
     * `campello_nn::DataType` (and rejecting what can't be mapped) is the architecture
     * registry's job (Phase 3), not this reader-level type.
     */
    enum class WeightDType
    {
        F32,
        F16,
        BF16,
        F64,
        I8,
        I16,
        I32,
        I64,
        U8,
        U16,
        U32,
        U64,
        Bool,
    };

    /**
     * @brief Byte size of a single element of `dtype`.
     */
    constexpr std::size_t weightDTypeSize(WeightDType dtype)
    {
        switch (dtype)
        {
        case WeightDType::F32:
            return 4;
        case WeightDType::F16:
            return 2;
        case WeightDType::BF16:
            return 2;
        case WeightDType::F64:
            return 8;
        case WeightDType::I8:
            return 1;
        case WeightDType::I16:
            return 2;
        case WeightDType::I32:
            return 4;
        case WeightDType::I64:
            return 8;
        case WeightDType::U8:
            return 1;
        case WeightDType::U16:
            return 2;
        case WeightDType::U32:
            return 4;
        case WeightDType::U64:
            return 8;
        case WeightDType::Bool:
            return 1;
        }
        throw std::runtime_error("campello_llm: unhandled WeightDType in weightDTypeSize()");
    }

    /**
     * @brief One tensor's name/dtype/shape plus a zero-copy view into its owning
     * `WeightsFile`'s backing buffer.
     *
     * `data`/`byteLength` point into memory owned by the `WeightsFile` that produced
     * this `TensorInfo` — valid only as long as that `WeightsFile` is alive.
     */
    struct TensorInfo
    {
        std::string name;
        WeightDType dtype;
        std::vector<std::int64_t> shape;
        const std::uint8_t *data = nullptr;
        std::size_t byteLength = 0;
    };

    /**
     * @brief Architecture-agnostic {name -> dtype/shape/bytes} view over a parsed
     * weights file.
     *
     * Implemented by `SafetensorsFile` and `GgufFile`. This is the only thing the
     * Phase 3 architecture registry should depend on to read tensor weights —
     * format-specific data (gguf's key/value metadata, embedded tokenizer vocab) stays
     * on the concrete subclass rather than being folded in here.
     */
    class WeightsFile
    {
    public:
        virtual ~WeightsFile() = default;

        virtual const std::vector<TensorInfo> &tensors() const = 0;

        virtual const TensorInfo *find(const std::string &name) const = 0;
    };

} // namespace systems::leal::campello_llm
