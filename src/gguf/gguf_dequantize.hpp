#pragma once

#include <campello_llm/weights_file.hpp>

#include <cstdint>
#include <vector>

namespace systems::leal::campello_llm
{

    /**
     * @brief Dequantizes a GGUF tensor to a host Float32 array.
     *
     * Supports the non-quantized types F32/F16/BF16 (passthrough/decode) and the
     * GGML block-quantized types Q4_0, Q4_1, Q5_0, Q5_1, Q8_0, Q8_1, Q2_K,
     * Q3_K, Q4_K, Q5_K, Q6_K and Q8_K. The caller is responsible for ensuring
     * `t.byteLength` matches `t.dtype` and `t.shape`.
     *
     * @throws std::runtime_error for unsupported dtypes or malformed block sizes.
     */
    std::vector<float> dequantizeGgufTensorToFloat32(const TensorInfo &t);

} // namespace systems::leal::campello_llm
