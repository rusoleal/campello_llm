#include "weight_loading.hpp"

#include <cmath>
#include <cstring>
#include <stdexcept>

#include <campello_nn/float16.hpp>

namespace systems::leal::campello_llm::internal
{
    namespace
    {
        const TensorInfo &findWeight(const WeightsFile &weights, const std::string &name)
        {
            const TensorInfo *t = weights.find(name);
            if (t == nullptr)
            {
                throw std::runtime_error("campello_llm: missing weight tensor '" + name + "'");
            }
            return *t;
        }

        void checkShape(const std::string &name, const std::vector<std::int64_t> &actual,
                         const std::vector<std::int64_t> &expected)
        {
            if (actual != expected)
            {
                throw std::runtime_error("campello_llm: weight '" + name + "' has an unexpected shape");
            }
        }

        float decodeBFloat16(std::uint16_t bits)
        {
            // BF16 is simply an IEEE-754 float32's top 16 bits (same exponent range as
            // F32, truncated mantissa) -- left-shift back into position.
            std::uint32_t f32Bits = static_cast<std::uint32_t>(bits) << 16;
            float value;
            std::memcpy(&value, &f32Bits, sizeof(value));
            return value;
        }

        // Real LLaMA/GPT checkpoints commonly ship BF16 (confirmed against
        // TinyLlama's real config.json: `"torch_dtype": "bfloat16"`) or F16 weights,
        // not Float32 -- decode whichever was actually stored into a Float32 buffer
        // host-side, once, at graph-build time. Anything else (Int8 quantized
        // weights, etc.) throws rather than guess a dequantization scheme.
        std::vector<float> decodeWeightToFloat32(const std::string &name, const TensorInfo &t)
        {
            std::size_t count = t.byteLength / weightDTypeSize(t.dtype);
            std::vector<float> out(count);
            switch (t.dtype)
            {
            case WeightDType::F32:
                std::memcpy(out.data(), t.data, t.byteLength);
                break;
            case WeightDType::F16:
            {
                const auto *src = reinterpret_cast<const std::uint16_t *>(t.data);
                for (std::size_t i = 0; i < count; ++i)
                {
                    out[i] = cnn::decodeFloat16(src[i]);
                }
                break;
            }
            case WeightDType::BF16:
            {
                const auto *src = reinterpret_cast<const std::uint16_t *>(t.data);
                for (std::size_t i = 0; i < count; ++i)
                {
                    out[i] = decodeBFloat16(src[i]);
                }
                break;
            }
            default:
                throw std::runtime_error("campello_llm: weight '" + name +
                                          "' has an unsupported dtype (only F32/F16/BF16 are supported)");
            }
            return out;
        }
    } // namespace

    cnn::Operand constantWeight(cnn::GraphBuilder &builder, const WeightsFile &weights, const std::string &name,
                                 const std::vector<std::int64_t> &expectedShape)
    {
        const TensorInfo &t = findWeight(weights, name);
        checkShape(name, t.shape, expectedShape);
        std::vector<float> decoded = decodeWeightToFloat32(name, t);
        return builder.constant({cnn::DataType::Float32, expectedShape, false, false}, decoded.data(),
                                 decoded.size() * sizeof(float));
    }

    cnn::Operand constantLinearWeightTransposed(cnn::GraphBuilder &builder, const WeightsFile &weights,
                                                 const std::string &name, std::int64_t outDim, std::int64_t inDim)
    {
        const TensorInfo &t = findWeight(weights, name);
        checkShape(name, t.shape, {outDim, inDim});
        std::vector<float> decoded = decodeWeightToFloat32(name, t);

        std::vector<float> transposed(static_cast<std::size_t>(outDim * inDim));
        for (std::int64_t row = 0; row < outDim; ++row)
        {
            for (std::int64_t col = 0; col < inDim; ++col)
            {
                transposed[static_cast<std::size_t>(col * outDim + row)] =
                    decoded[static_cast<std::size_t>(row * inDim + col)];
            }
        }
        return builder.constant({cnn::DataType::Float32, {inDim, outDim}, false, false}, transposed.data(),
                                 transposed.size() * sizeof(float));
    }

    cnn::Operand constantScalarF32(cnn::GraphBuilder &builder, float value)
    {
        return builder.constant({cnn::DataType::Float32, {1}, false, false}, &value, sizeof(value));
    }

    cnn::Operand constantCausalMask(cnn::GraphBuilder &builder, std::int64_t seqLen)
    {
        std::vector<float> mask(static_cast<std::size_t>(seqLen * seqLen));
        for (std::int64_t row = 0; row < seqLen; ++row)
        {
            for (std::int64_t col = 0; col < seqLen; ++col)
            {
                mask[static_cast<std::size_t>(row * seqLen + col)] = col <= row ? 0.0f : -1e9f;
            }
        }
        return builder.constant({cnn::DataType::Float32, {seqLen, seqLen}, false, false}, mask.data(),
                                 mask.size() * sizeof(float));
    }

    std::pair<std::vector<float>, std::vector<float>> ropeCosSinForPosition(std::int64_t position,
                                                                             std::int64_t headDim, float theta)
    {
        std::int64_t half = headDim / 2;
        std::vector<float> cosRow(static_cast<std::size_t>(headDim));
        std::vector<float> sinRow(static_cast<std::size_t>(headDim));
        for (std::int64_t i = 0; i < half; ++i)
        {
            float freq = std::pow(theta, -static_cast<float>(2 * i) / static_cast<float>(headDim));
            float angle = static_cast<float>(position) * freq;
            float c = std::cos(angle);
            float s = std::sin(angle);
            cosRow[static_cast<std::size_t>(i)] = c;
            cosRow[static_cast<std::size_t>(i + half)] = c;
            sinRow[static_cast<std::size_t>(i)] = s;
            sinRow[static_cast<std::size_t>(i + half)] = s;
        }
        return {cosRow, sinRow};
    }

    std::pair<cnn::Operand, cnn::Operand> constantRope(cnn::GraphBuilder &builder, std::int64_t seqLen,
                                                        std::int64_t headDim, float theta)
    {
        std::vector<float> cosTable(static_cast<std::size_t>(seqLen * headDim));
        std::vector<float> sinTable(static_cast<std::size_t>(seqLen * headDim));
        for (std::int64_t pos = 0; pos < seqLen; ++pos)
        {
            auto [cosRow, sinRow] = ropeCosSinForPosition(pos, headDim, theta);
            std::size_t base = static_cast<std::size_t>(pos * headDim);
            std::copy(cosRow.begin(), cosRow.end(), cosTable.begin() + static_cast<std::ptrdiff_t>(base));
            std::copy(sinRow.begin(), sinRow.end(), sinTable.begin() + static_cast<std::ptrdiff_t>(base));
        }
        cnn::Operand cosOp = builder.constant({cnn::DataType::Float32, {seqLen, headDim}, false, false},
                                               cosTable.data(), cosTable.size() * sizeof(float));
        cnn::Operand sinOp = builder.constant({cnn::DataType::Float32, {seqLen, headDim}, false, false},
                                               sinTable.data(), sinTable.size() * sizeof(float));
        return {cosOp, sinOp};
    }

    cnn::Operand repeatKvHeads(cnn::GraphBuilder &builder, cnn::Operand kv, std::int64_t numKvHeads,
                               std::int64_t groupSize, std::int64_t seqLen, std::int64_t headDim)
    {
        if (numKvHeads == 1 && groupSize == 1)
        {
            return kv;
        }
        std::vector<cnn::Operand> headBlocks;
        headBlocks.reserve(static_cast<std::size_t>(numKvHeads));
        for (std::int64_t g = 0; g < numKvHeads; ++g)
        {
            cnn::Operand headSlice = builder.slice(kv, {g, 0, 0}, {1, seqLen, headDim});
            if (groupSize == 1)
            {
                headBlocks.push_back(headSlice);
            }
            else
            {
                std::vector<cnn::Operand> repeated(static_cast<std::size_t>(groupSize), headSlice);
                headBlocks.push_back(builder.concat(repeated, 0));
            }
        }
        return headBlocks.size() == 1 ? headBlocks[0] : builder.concat(headBlocks, 0);
    }

} // namespace systems::leal::campello_llm::internal
