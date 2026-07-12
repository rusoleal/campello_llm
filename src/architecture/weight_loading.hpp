#pragma once

#include <cstdint>
#include <string>
#include <utility>
#include <vector>

#include <campello_nn/graph_builder.hpp>

#include <campello_llm/weights_file.hpp>

namespace systems::leal::campello_llm::internal
{

    namespace cnn = systems::leal::campello_nn;

    /**
     * @brief Fetches a weight tensor named `name` with exactly `expectedShape`,
     * decodes it to Float32 if stored as F16/BF16 (real LLaMA/GPT checkpoints
     * commonly ship BF16), and binds it as a graph constant verbatim (no transpose)
     * — for tensors whose on-disk layout already matches what the consuming op
     * expects (norm weights, embedding tables, biases).
     * @throws std::runtime_error if missing, an unsupported dtype (anything other
     * than F32/F16/BF16 — e.g. an Int8-quantized weight, which has no
     * dequantization path here yet), or shape-mismatched.
     */
    cnn::Operand constantWeight(cnn::GraphBuilder &builder, const WeightsFile &weights, const std::string &name,
                                 const std::vector<std::int64_t> &expectedShape);

    /**
     * @brief Fetches an `[outDim, inDim]` PyTorch `nn.Linear`-style weight (decoding
     * F16/BF16 to Float32 first, same as `constantWeight()`), transposes it
     * host-side to `[inDim, outDim]` (`campello_nn::GraphBuilder::matmul` expects its
     * second operand as `[in, out]`, confirmed against
     * `tests/universal/test_transformer_block.cpp` in campello_nn), and binds the
     * transposed bytes as a graph constant. The decode+transpose happens once, at
     * graph-build time, on a temporary host buffer — not a graph node.
     * @throws std::runtime_error if missing, an unsupported dtype, or shape-mismatched.
     */
    cnn::Operand constantLinearWeightTransposed(cnn::GraphBuilder &builder, const WeightsFile &weights,
                                                 const std::string &name, std::int64_t outDim, std::int64_t inDim);

    /**
     * @brief Description of a loaded linear weight, which may be either a plain
     * Float32 constant (legacy path) or a raw GGML block-quantized constant.
     */
    struct LinearWeight
    {
        cnn::Operand operand;
        bool isQuantized = false;
        std::int32_t ggmlQuantType = 0; // GGML quantization type enum value (e.g. 12 for Q4_K)
    };

    /**
     * @brief Loads an `[outDim, inDim]` PyTorch `nn.Linear`-style weight for use with
     * `applyLinear()`. If the weight is F32/F16/BF16 it is decoded and transposed
     * host-side to `[inDim, outDim]` and `isQuantized` is false. If it is a supported
     * GGML block-quantized type (Q4_0, Q8_0, Q4_K, etc.), the raw quantized bytes are
     * bound as an Int8 constant and `isQuantized` is true with the GGML type stored
     * in `ggmlQuantType`.
     * @throws std::runtime_error if missing, an unsupported dtype, or shape-mismatched.
     */
    LinearWeight loadLinearWeight(cnn::GraphBuilder &builder, const WeightsFile &weights,
                                   const std::string &name, std::int64_t outDim, std::int64_t inDim);

    /**
     * @brief Applies a `LinearWeight` to `input` (shape [..., M, inDim]), returning
     * [..., M, outDim]. Uses `matmul()` for Float32 weights and `ggmlQuantizedMatmul()`
     * for raw GGML quantized weights.
     */
    cnn::Operand applyLinear(cnn::GraphBuilder &builder, cnn::Operand input, const LinearWeight &weight,
                              std::int64_t outDim, std::int64_t inDim);

    /**
     * @brief A rank-1 `[1]` Float32 constant holding `value` (for scale factors etc.,
     * relying on `mul()`/`add()`'s NumPy-style broadcasting).
     */
    cnn::Operand constantScalarF32(cnn::GraphBuilder &builder, float value);

    /**
     * @brief A `[seqLen, seqLen]` additive causal mask: `0` on/below the diagonal,
     * `-1e9` above it (a large finite negative number rather than `-inf`, to stay
     * well-defined across every backend/dtype, not just the CPU reference backend).
     */
    cnn::Operand constantCausalMask(cnn::GraphBuilder &builder, std::int64_t seqLen);

    /**
     * @brief `{cos, sin}`, each a `[seqLen, headDim]` Float32 constant for the
     * GPT-NeoX/LLaMA "rotate-half" RoPE convention `campello_nn::GraphBuilder::
     * rotaryEmbedding()` implements — broadcasts against a `[..., seqLen, headDim]`
     * operand via `mul()`/`add()`'s NumPy-style broadcasting (no need to expand to
     * the consuming operand's full rank).
     */
    std::pair<cnn::Operand, cnn::Operand> constantRope(cnn::GraphBuilder &builder, std::int64_t seqLen,
                                                        std::int64_t headDim, float theta);

    /**
     * @brief Plain host-side (no `GraphBuilder` involved) `{cos, sin}` vectors, each
     * length `headDim`, for a single absolute position — the same GPT-NeoX/LLaMA
     * "rotate-half" RoPE formula `constantRope()` bakes into a build-time constant
     * table, computed here for one runtime-supplied position instead. Used to feed
     * the decode graph's `rope_cos`/`rope_sin` *inputs* (see `buildLlamaDecodeGraph()`
     * in `architecture.hpp`), since a KV-cache decode step's absolute position varies
     * call to call and can't be baked into the graph at build time the way the
     * fixed-`seqLen` prefill graph's `cosOp`/`sinOp` constants are.
     */
    std::pair<std::vector<float>, std::vector<float>> ropeCosSinForPosition(std::int64_t position,
                                                                             std::int64_t headDim, float theta);

    /**
     * @brief Repeats `kv` (shape `[numKvHeads, seqLen, headDim]`) along axis 0 so each
     * KV head is immediately followed by `groupSize - 1` copies of itself — the
     * standard GQA `repeat_kv` layout (query heads `[g*groupSize, (g+1)*groupSize)`
     * all attend to KV head `g`) — via `slice()` + `concat()`, since `matmul()`
     * requires its batch dimensions to match exactly (no implicit broadcasting).
     */
    cnn::Operand repeatKvHeads(cnn::GraphBuilder &builder, cnn::Operand kv, std::int64_t numKvHeads,
                               std::int64_t groupSize, std::int64_t seqLen, std::int64_t headDim);

} // namespace systems::leal::campello_llm::internal
