#include "gguf_dequantize.hpp"

#include <campello_nn/float16.hpp>

#include <cmath>
#include <cstring>
#include <stdexcept>

namespace systems::leal::campello_llm
{

    namespace
    {
        namespace cnn = systems::leal::campello_nn;

        float decodeBFloat16(std::uint16_t bits)
        {
            std::uint32_t f32Bits = static_cast<std::uint32_t>(bits) << 16;
            float value;
            std::memcpy(&value, &f32Bits, sizeof(value));
            return value;
        }

        float decodeFloat16Scale(std::uint16_t bits)
        {
            return cnn::decodeFloat16(bits);
        }

        std::vector<float> decodeF32(const TensorInfo &t)
        {
            std::size_t count = t.byteLength / sizeof(float);
            std::vector<float> out(count);
            std::memcpy(out.data(), t.data, t.byteLength);
            return out;
        }

        std::vector<float> decodeF16(const TensorInfo &t)
        {
            std::size_t count = t.byteLength / sizeof(std::uint16_t);
            std::vector<float> out(count);
            const auto *src = reinterpret_cast<const std::uint16_t *>(t.data);
            for (std::size_t i = 0; i < count; ++i)
            {
                out[i] = cnn::decodeFloat16(src[i]);
            }
            return out;
        }

        std::vector<float> decodeBF16(const TensorInfo &t)
        {
            std::size_t count = t.byteLength / sizeof(std::uint16_t);
            std::vector<float> out(count);
            const auto *src = reinterpret_cast<const std::uint16_t *>(t.data);
            for (std::size_t i = 0; i < count; ++i)
            {
                out[i] = decodeBFloat16(src[i]);
            }
            return out;
        }

        std::int64_t tensorElementCount(const TensorInfo &t)
        {
            std::int64_t count = 1;
            for (std::int64_t dim : t.shape)
            {
                count *= dim;
            }
            return count;
        }

        // Q4_0: 32 elements per block. Each block is:
        //   2 bytes: scale `d` (fp16)
        //   16 bytes: 32 nibbles (qs[0..15]), where qs[j] stores:
        //     low nibble  -> element j
        //     high nibble -> element j + 16
        // Dequant: y = (nibble - 8) * d
        std::vector<float> dequantizeQ4_0(const TensorInfo &t)
        {
            constexpr std::size_t kBlockElements = 32;
            constexpr std::size_t kBlockBytes = 2 + 16;

            std::int64_t nElements = tensorElementCount(t);
            if (nElements % static_cast<std::int64_t>(kBlockElements) != 0)
            {
                throw std::runtime_error("campello_llm: Q4_0 tensor element count is not a multiple of 32");
            }
            std::size_t nBlocks = static_cast<std::size_t>(nElements / static_cast<std::int64_t>(kBlockElements));
            std::size_t expectedBytes = nBlocks * kBlockBytes;
            if (t.byteLength != expectedBytes)
            {
                throw std::runtime_error("campello_llm: Q4_0 tensor byte length does not match element count");
            }

            std::vector<float> out(static_cast<std::size_t>(nElements));
            const std::uint8_t *ptr = t.data;
            std::size_t outIdx = 0;
            for (std::size_t b = 0; b < nBlocks; ++b)
            {
                std::uint16_t scaleBits;
                std::memcpy(&scaleBits, ptr, sizeof(scaleBits));
                ptr += sizeof(scaleBits);
                float d = decodeFloat16Scale(scaleBits);
                for (std::size_t j = 0; j < 16; ++j)
                {
                    std::uint8_t q = ptr[j];
                    out[outIdx + j] = (static_cast<float>(static_cast<std::int8_t>(q & 0x0F) - 8)) * d;
                    out[outIdx + j + 16] = (static_cast<float>(static_cast<std::int8_t>(q >> 4) - 8)) * d;
                }
                ptr += 16;
                outIdx += kBlockElements;
            }
            return out;
        }

        // Q8_0: 32 elements per block. Each block is:
        //   2 bytes: scale `d` (fp16)
        //   32 bytes: int8 values qs[0..31]
        // Dequant: y = qs * d
        std::vector<float> dequantizeQ8_0(const TensorInfo &t)
        {
            constexpr std::size_t kBlockElements = 32;
            constexpr std::size_t kBlockBytes = 2 + 32;

            std::int64_t nElements = tensorElementCount(t);
            if (nElements % static_cast<std::int64_t>(kBlockElements) != 0)
            {
                throw std::runtime_error("campello_llm: Q8_0 tensor element count is not a multiple of 32");
            }
            std::size_t nBlocks = static_cast<std::size_t>(nElements / static_cast<std::int64_t>(kBlockElements));
            std::size_t expectedBytes = nBlocks * kBlockBytes;
            if (t.byteLength != expectedBytes)
            {
                throw std::runtime_error("campello_llm: Q8_0 tensor byte length does not match element count");
            }

            std::vector<float> out(static_cast<std::size_t>(nElements));
            const std::uint8_t *ptr = t.data;
            std::size_t outIdx = 0;
            for (std::size_t b = 0; b < nBlocks; ++b)
            {
                std::uint16_t scaleBits;
                std::memcpy(&scaleBits, ptr, sizeof(scaleBits));
                ptr += sizeof(scaleBits);
                float d = decodeFloat16Scale(scaleBits);
                for (std::size_t j = 0; j < kBlockElements; ++j)
                {
                    out[outIdx + j] = static_cast<float>(static_cast<std::int8_t>(ptr[j])) * d;
                }
                ptr += kBlockElements;
                outIdx += kBlockElements;
            }
            return out;
        }

        float decodeFloat32Scale(const std::uint8_t *ptr)
        {
            float d;
            std::memcpy(&d, ptr, sizeof(d));
            return d;
        }

        // Q4_1: 32 elements per block. Bytes: d(2), m(2), qs[16].
        // Dequant: y = qs * d + m
        std::vector<float> dequantizeQ4_1(const TensorInfo &t)
        {
            constexpr std::size_t kBlockElements = 32;
            constexpr std::size_t kBlockBytes = 2 + 2 + 16;

            std::int64_t nElements = tensorElementCount(t);
            if (nElements % static_cast<std::int64_t>(kBlockElements) != 0)
            {
                throw std::runtime_error("campello_llm: Q4_1 tensor element count is not a multiple of 32");
            }
            std::size_t nBlocks = static_cast<std::size_t>(nElements / static_cast<std::int64_t>(kBlockElements));
            if (t.byteLength != nBlocks * kBlockBytes)
            {
                throw std::runtime_error("campello_llm: Q4_1 tensor byte length does not match element count");
            }

            std::vector<float> out(static_cast<std::size_t>(nElements));
            const std::uint8_t *ptr = t.data;
            std::size_t outIdx = 0;
            for (std::size_t b = 0; b < nBlocks; ++b)
            {
                std::uint16_t scaleBits, minBits;
                std::memcpy(&scaleBits, ptr, sizeof(scaleBits));
                std::memcpy(&minBits, ptr + 2, sizeof(minBits));
                ptr += 4;
                float d = decodeFloat16Scale(scaleBits);
                float m = decodeFloat16Scale(minBits);
                for (std::size_t j = 0; j < 16; ++j)
                {
                    std::uint8_t q = ptr[j];
                    out[outIdx + j] = static_cast<float>(q & 0x0F) * d + m;
                    out[outIdx + j + 16] = static_cast<float>(q >> 4) * d + m;
                }
                ptr += 16;
                outIdx += kBlockElements;
            }
            return out;
        }

        // Q5_0: 32 elements per block. Bytes: d(2), qh(4), qs(16).
        // Dequant: y = ((qs | qh_bit) - 16) * d
        std::vector<float> dequantizeQ5_0(const TensorInfo &t)
        {
            constexpr std::size_t kBlockElements = 32;
            constexpr std::size_t kBlockBytes = 2 + 4 + 16;

            std::int64_t nElements = tensorElementCount(t);
            if (nElements % static_cast<std::int64_t>(kBlockElements) != 0)
            {
                throw std::runtime_error("campello_llm: Q5_0 tensor element count is not a multiple of 32");
            }
            std::size_t nBlocks = static_cast<std::size_t>(nElements / static_cast<std::int64_t>(kBlockElements));
            if (t.byteLength != nBlocks * kBlockBytes)
            {
                throw std::runtime_error("campello_llm: Q5_0 tensor byte length does not match element count");
            }

            std::vector<float> out(static_cast<std::size_t>(nElements));
            const std::uint8_t *ptr = t.data;
            std::size_t outIdx = 0;
            for (std::size_t b = 0; b < nBlocks; ++b)
            {
                std::uint16_t scaleBits;
                std::memcpy(&scaleBits, ptr, sizeof(scaleBits));
                ptr += 2;
                float d = decodeFloat16Scale(scaleBits);
                std::uint32_t qh;
                std::memcpy(&qh, ptr, sizeof(qh));
                ptr += 4;
                for (std::size_t j = 0; j < 16; ++j)
                {
                    std::uint8_t xh0 = ((qh >> (j + 0)) << 4) & 0x10;
                    std::uint8_t xh1 = ((qh >> (j + 12))) & 0x10;
                    std::int32_t x0 = static_cast<std::int32_t>((ptr[j] & 0x0F) | xh0) - 16;
                    std::int32_t x1 = static_cast<std::int32_t>((ptr[j] >> 4) | xh1) - 16;
                    out[outIdx + j] = static_cast<float>(x0) * d;
                    out[outIdx + j + 16] = static_cast<float>(x1) * d;
                }
                ptr += 16;
                outIdx += kBlockElements;
            }
            return out;
        }

        // Q5_1: 32 elements per block. Bytes: d(2), m(2), qh(4), qs(16).
        // Dequant: y = (qs | qh_bit) * d + m
        std::vector<float> dequantizeQ5_1(const TensorInfo &t)
        {
            constexpr std::size_t kBlockElements = 32;
            constexpr std::size_t kBlockBytes = 2 + 2 + 4 + 16;

            std::int64_t nElements = tensorElementCount(t);
            if (nElements % static_cast<std::int64_t>(kBlockElements) != 0)
            {
                throw std::runtime_error("campello_llm: Q5_1 tensor element count is not a multiple of 32");
            }
            std::size_t nBlocks = static_cast<std::size_t>(nElements / static_cast<std::int64_t>(kBlockElements));
            if (t.byteLength != nBlocks * kBlockBytes)
            {
                throw std::runtime_error("campello_llm: Q5_1 tensor byte length does not match element count");
            }

            std::vector<float> out(static_cast<std::size_t>(nElements));
            const std::uint8_t *ptr = t.data;
            std::size_t outIdx = 0;
            for (std::size_t b = 0; b < nBlocks; ++b)
            {
                std::uint16_t scaleBits, minBits;
                std::memcpy(&scaleBits, ptr, sizeof(scaleBits));
                std::memcpy(&minBits, ptr + 2, sizeof(minBits));
                ptr += 4;
                float d = decodeFloat16Scale(scaleBits);
                float m = decodeFloat16Scale(minBits);
                std::uint32_t qh;
                std::memcpy(&qh, ptr, sizeof(qh));
                ptr += 4;
                for (std::size_t j = 0; j < 16; ++j)
                {
                    std::uint8_t xh0 = ((qh >> (j + 0)) << 4) & 0x10;
                    std::uint8_t xh1 = ((qh >> (j + 12))) & 0x10;
                    std::int32_t x0 = static_cast<std::int32_t>((ptr[j] & 0x0F) | xh0);
                    std::int32_t x1 = static_cast<std::int32_t>((ptr[j] >> 4) | xh1);
                    out[outIdx + j] = static_cast<float>(x0) * d + m;
                    out[outIdx + j + 16] = static_cast<float>(x1) * d + m;
                }
                ptr += 16;
                outIdx += kBlockElements;
            }
            return out;
        }

        // Q8_1: 32 elements per block. Bytes: d(2), s(2), qs[32].
        // Dequant: y = qs * d (s is redundant for dequantization).
        std::vector<float> dequantizeQ8_1(const TensorInfo &t)
        {
            constexpr std::size_t kBlockElements = 32;
            constexpr std::size_t kBlockBytes = 2 + 2 + 32;

            std::int64_t nElements = tensorElementCount(t);
            if (nElements % static_cast<std::int64_t>(kBlockElements) != 0)
            {
                throw std::runtime_error("campello_llm: Q8_1 tensor element count is not a multiple of 32");
            }
            std::size_t nBlocks = static_cast<std::size_t>(nElements / static_cast<std::int64_t>(kBlockElements));
            if (t.byteLength != nBlocks * kBlockBytes)
            {
                throw std::runtime_error("campello_llm: Q8_1 tensor byte length does not match element count");
            }

            std::vector<float> out(static_cast<std::size_t>(nElements));
            const std::uint8_t *ptr = t.data;
            std::size_t outIdx = 0;
            for (std::size_t b = 0; b < nBlocks; ++b)
            {
                std::uint16_t scaleBits;
                std::memcpy(&scaleBits, ptr, sizeof(scaleBits));
                ptr += 4; // skip d and s
                float d = decodeFloat16Scale(scaleBits);
                for (std::size_t j = 0; j < kBlockElements; ++j)
                {
                    out[outIdx + j] = static_cast<float>(static_cast<std::int8_t>(ptr[j])) * d;
                }
                ptr += kBlockElements;
                outIdx += kBlockElements;
            }
            return out;
        }

        // K-quant helpers -----------------------------------------------------

        // Decodes the 6-bit scale and min used by Q4_K and Q5_K.
        inline void getScaleMinK4(int j, const std::uint8_t *q, std::uint8_t *d, std::uint8_t *m)
        {
            if (j < 4)
            {
                *d = q[j] & 63;
                *m = q[j + 4] & 63;
            }
            else
            {
                *d = (q[j + 4] & 0x0F) | ((q[j - 4] >> 6) << 4);
                *m = (q[j + 4] >> 4) | ((q[j - 0] >> 6) << 4);
            }
        }

        // Q2_K: 256 elements per super-block. Layout:
        //   scales[16], qs[64], d(2), dmin(2)
        // Dequant: y = d * (sc&0xF) * ((qs >> shift)&3) - min * (sc>>4)
        std::vector<float> dequantizeQ2_K(const TensorInfo &t)
        {
            constexpr std::size_t kBlockElements = 256;
            constexpr std::size_t kBlockBytes = 16 + 64 + 2 + 2;

            std::int64_t nElements = tensorElementCount(t);
            if (nElements % static_cast<std::int64_t>(kBlockElements) != 0)
            {
                throw std::runtime_error("campello_llm: Q2_K tensor element count is not a multiple of 256");
            }
            std::size_t nBlocks = static_cast<std::size_t>(nElements / static_cast<std::int64_t>(kBlockElements));
            if (t.byteLength != nBlocks * kBlockBytes)
            {
                throw std::runtime_error("campello_llm: Q2_K tensor byte length does not match element count");
            }

            std::vector<float> out(static_cast<std::size_t>(nElements));
            const std::uint8_t *ptr = t.data;
            std::size_t outIdx = 0;
            for (std::size_t b = 0; b < nBlocks; ++b)
            {
                const std::uint8_t *scales = ptr;
                ptr += 16;
                const std::uint8_t *qs = ptr;
                ptr += 64;
                std::uint16_t dBits, minBits;
                std::memcpy(&dBits, ptr, sizeof(dBits));
                std::memcpy(&minBits, ptr + 2, sizeof(minBits));
                ptr += 4;
                float d = decodeFloat16Scale(dBits);
                float min = decodeFloat16Scale(minBits);

                int is = 0;
                for (int n = 0; n < static_cast<int>(kBlockElements); n += 128)
                {
                    int shift = 0;
                    for (int j = 0; j < 4; ++j)
                    {
                        std::uint8_t sc = scales[is++];
                        float dl = d * static_cast<float>(sc & 0x0F);
                        float ml = min * static_cast<float>(sc >> 4);
                        for (int l = 0; l < 16; ++l)
                        {
                            out[outIdx++] = dl * static_cast<float>((qs[l] >> shift) & 3) - ml;
                        }
                        sc = scales[is++];
                        dl = d * static_cast<float>(sc & 0x0F);
                        ml = min * static_cast<float>(sc >> 4);
                        for (int l = 0; l < 16; ++l)
                        {
                            out[outIdx++] = dl * static_cast<float>((qs[l + 16] >> shift) & 3) - ml;
                        }
                        shift += 2;
                    }
                    qs += 32;
                }
            }
            return out;
        }

        // Q3_K: 256 elements per super-block. Layout:
        //   hmask[32], qs[64], scales[12], d(2)
        std::vector<float> dequantizeQ3_K(const TensorInfo &t)
        {
            constexpr std::size_t kBlockElements = 256;
            constexpr std::size_t kBlockBytes = 32 + 64 + 12 + 2;

            std::int64_t nElements = tensorElementCount(t);
            if (nElements % static_cast<std::int64_t>(kBlockElements) != 0)
            {
                throw std::runtime_error("campello_llm: Q3_K tensor element count is not a multiple of 256");
            }
            std::size_t nBlocks = static_cast<std::size_t>(nElements / static_cast<std::int64_t>(kBlockElements));
            if (t.byteLength != nBlocks * kBlockBytes)
            {
                throw std::runtime_error("campello_llm: Q3_K tensor byte length does not match element count");
            }

            std::vector<float> out(static_cast<std::size_t>(nElements));
            const std::uint8_t *ptr = t.data;
            std::size_t outIdx = 0;
            for (std::size_t b = 0; b < nBlocks; ++b)
            {
                const std::uint8_t *hmask = ptr;
                ptr += 32;
                const std::uint8_t *qs = ptr;
                ptr += 64;
                const std::uint8_t *scalesIn = ptr;
                ptr += 12;
                std::uint16_t dBits;
                std::memcpy(&dBits, ptr, sizeof(dBits));
                ptr += 2;
                float dAll = decodeFloat16Scale(dBits);

                std::uint32_t aux[4];
                std::memcpy(aux, scalesIn, 12);
                std::uint32_t tmp = aux[2];
                aux[2] = ((aux[0] >> 4) & 0x0F0F0F0Fu) | (((tmp >> 4) & 0x03030303u) << 4);
                aux[3] = ((aux[1] >> 4) & 0x0F0F0F0Fu) | (((tmp >> 6) & 0x03030303u) << 4);
                aux[0] = (aux[0] & 0x0F0F0F0Fu) | (((tmp >> 0) & 0x03030303u) << 4);
                aux[1] = (aux[1] & 0x0F0F0F0Fu) | (((tmp >> 2) & 0x03030303u) << 4);
                const auto *scales = reinterpret_cast<const std::int8_t *>(aux);

                std::uint8_t m = 1;
                int is = 0;
                for (int n = 0; n < static_cast<int>(kBlockElements); n += 128)
                {
                    int shift = 0;
                    for (int j = 0; j < 4; ++j)
                    {
                        float dl = dAll * static_cast<float>(scales[is++] - 32);
                        for (int l = 0; l < 16; ++l)
                        {
                            int q = static_cast<int>((qs[l] >> shift) & 3);
                            q -= (hmask[l] & m) ? 0 : 4;
                            out[outIdx++] = dl * static_cast<float>(q);
                        }
                        dl = dAll * static_cast<float>(scales[is++] - 32);
                        for (int l = 0; l < 16; ++l)
                        {
                            int q = static_cast<int>((qs[l + 16] >> shift) & 3);
                            q -= (hmask[l + 16] & m) ? 0 : 4;
                            out[outIdx++] = dl * static_cast<float>(q);
                        }
                        shift += 2;
                        m <<= 1;
                    }
                    qs += 32;
                }
            }
            return out;
        }

        // Q4_K: 256 elements per super-block. Layout:
        //   d(2), dmin(2), scales[12], qs[128]
        std::vector<float> dequantizeQ4_K(const TensorInfo &t)
        {
            constexpr std::size_t kBlockElements = 256;
            constexpr std::size_t kBlockBytes = 2 + 2 + 12 + 128;

            std::int64_t nElements = tensorElementCount(t);
            if (nElements % static_cast<std::int64_t>(kBlockElements) != 0)
            {
                throw std::runtime_error("campello_llm: Q4_K tensor element count is not a multiple of 256");
            }
            std::size_t nBlocks = static_cast<std::size_t>(nElements / static_cast<std::int64_t>(kBlockElements));
            if (t.byteLength != nBlocks * kBlockBytes)
            {
                throw std::runtime_error("campello_llm: Q4_K tensor byte length does not match element count");
            }

            std::vector<float> out(static_cast<std::size_t>(nElements));
            const std::uint8_t *ptr = t.data;
            std::size_t outIdx = 0;
            for (std::size_t b = 0; b < nBlocks; ++b)
            {
                std::uint16_t dBits, minBits;
                std::memcpy(&dBits, ptr, sizeof(dBits));
                std::memcpy(&minBits, ptr + 2, sizeof(minBits));
                ptr += 4;
                float d = decodeFloat16Scale(dBits);
                float min = decodeFloat16Scale(minBits);
                const std::uint8_t *scales = ptr;
                ptr += 12;
                const std::uint8_t *q = ptr;
                ptr += 128;

                int is = 0;
                std::uint8_t sc, m;
                for (int j = 0; j < static_cast<int>(kBlockElements); j += 64)
                {
                    getScaleMinK4(is + 0, scales, &sc, &m);
                    float d1 = d * static_cast<float>(sc);
                    float m1 = min * static_cast<float>(m);
                    getScaleMinK4(is + 1, scales, &sc, &m);
                    float d2 = d * static_cast<float>(sc);
                    float m2 = min * static_cast<float>(m);
                    for (int l = 0; l < 32; ++l)
                    {
                        out[outIdx++] = d1 * static_cast<float>(q[l] & 0x0F) - m1;
                    }
                    for (int l = 0; l < 32; ++l)
                    {
                        out[outIdx++] = d2 * static_cast<float>(q[l] >> 4) - m2;
                    }
                    q += 32;
                    is += 2;
                }
            }
            return out;
        }

        // Q5_K: 256 elements per super-block. Layout:
        //   d(2), dmin(2), scales[12], qh[32], qs[128]
        std::vector<float> dequantizeQ5_K(const TensorInfo &t)
        {
            constexpr std::size_t kBlockElements = 256;
            constexpr std::size_t kBlockBytes = 2 + 2 + 12 + 32 + 128;

            std::int64_t nElements = tensorElementCount(t);
            if (nElements % static_cast<std::int64_t>(kBlockElements) != 0)
            {
                throw std::runtime_error("campello_llm: Q5_K tensor element count is not a multiple of 256");
            }
            std::size_t nBlocks = static_cast<std::size_t>(nElements / static_cast<std::int64_t>(kBlockElements));
            if (t.byteLength != nBlocks * kBlockBytes)
            {
                throw std::runtime_error("campello_llm: Q5_K tensor byte length does not match element count");
            }

            std::vector<float> out(static_cast<std::size_t>(nElements));
            const std::uint8_t *ptr = t.data;
            std::size_t outIdx = 0;
            for (std::size_t b = 0; b < nBlocks; ++b)
            {
                std::uint16_t dBits, minBits;
                std::memcpy(&dBits, ptr, sizeof(dBits));
                std::memcpy(&minBits, ptr + 2, sizeof(minBits));
                ptr += 4;
                float d = decodeFloat16Scale(dBits);
                float min = decodeFloat16Scale(minBits);
                const std::uint8_t *scales = ptr;
                ptr += 12;
                const std::uint8_t *qh = ptr;
                ptr += 32;
                const std::uint8_t *ql = ptr;
                ptr += 128;

                int is = 0;
                std::uint8_t sc, m;
                std::uint8_t u1 = 1, u2 = 2;
                for (int j = 0; j < static_cast<int>(kBlockElements); j += 64)
                {
                    getScaleMinK4(is + 0, scales, &sc, &m);
                    float d1 = d * static_cast<float>(sc);
                    float m1 = min * static_cast<float>(m);
                    getScaleMinK4(is + 1, scales, &sc, &m);
                    float d2 = d * static_cast<float>(sc);
                    float m2 = min * static_cast<float>(m);
                    for (int l = 0; l < 32; ++l)
                    {
                        std::uint8_t v = (ql[l] & 0x0F) + ((qh[l] & u1) ? 16 : 0);
                        out[outIdx++] = d1 * static_cast<float>(v) - m1;
                    }
                    for (int l = 0; l < 32; ++l)
                    {
                        std::uint8_t v = (ql[l] >> 4) + ((qh[l] & u2) ? 16 : 0);
                        out[outIdx++] = d2 * static_cast<float>(v) - m2;
                    }
                    ql += 32;
                    is += 2;
                    u1 <<= 2;
                    u2 <<= 2;
                }
            }
            return out;
        }

        // Q6_K: 256 elements per super-block. Layout:
        //   ql[128], qh[64], scales[16], d(2)
        std::vector<float> dequantizeQ6_K(const TensorInfo &t)
        {
            constexpr std::size_t kBlockElements = 256;
            constexpr std::size_t kBlockBytes = 128 + 64 + 16 + 2;

            std::int64_t nElements = tensorElementCount(t);
            if (nElements % static_cast<std::int64_t>(kBlockElements) != 0)
            {
                throw std::runtime_error("campello_llm: Q6_K tensor element count is not a multiple of 256");
            }
            std::size_t nBlocks = static_cast<std::size_t>(nElements / static_cast<std::int64_t>(kBlockElements));
            if (t.byteLength != nBlocks * kBlockBytes)
            {
                throw std::runtime_error("campello_llm: Q6_K tensor byte length does not match element count");
            }

            std::vector<float> out(static_cast<std::size_t>(nElements));
            const std::uint8_t *ptr = t.data;
            std::size_t outIdx = 0;
            for (std::size_t b = 0; b < nBlocks; ++b)
            {
                const std::uint8_t *ql = ptr;
                ptr += 128;
                const std::uint8_t *qh = ptr;
                ptr += 64;
                const auto *scales = reinterpret_cast<const std::int8_t *>(ptr);
                ptr += 16;
                std::uint16_t dBits;
                std::memcpy(&dBits, ptr, sizeof(dBits));
                ptr += 2;
                float d = decodeFloat16Scale(dBits);

                for (int n = 0; n < static_cast<int>(kBlockElements); n += 128)
                {
                    for (int l = 0; l < 32; ++l)
                    {
                        int is = l / 16;
                        std::int8_t q1 = static_cast<std::int8_t>((ql[l + 0] & 0x0F) | (((qh[l] >> 0) & 3) << 4)) - 32;
                        std::int8_t q2 = static_cast<std::int8_t>((ql[l + 32] & 0x0F) | (((qh[l] >> 2) & 3) << 4)) - 32;
                        std::int8_t q3 = static_cast<std::int8_t>((ql[l + 0] >> 4) | (((qh[l] >> 4) & 3) << 4)) - 32;
                        std::int8_t q4 = static_cast<std::int8_t>((ql[l + 32] >> 4) | (((qh[l] >> 6) & 3) << 4)) - 32;
                        out[outIdx + l + 0] = d * static_cast<float>(scales[is + 0]) * static_cast<float>(q1);
                        out[outIdx + l + 32] = d * static_cast<float>(scales[is + 2]) * static_cast<float>(q2);
                        out[outIdx + l + 64] = d * static_cast<float>(scales[is + 4]) * static_cast<float>(q3);
                        out[outIdx + l + 96] = d * static_cast<float>(scales[is + 6]) * static_cast<float>(q4);
                    }
                    outIdx += 128;
                    ql += 64;
                    qh += 32;
                    scales += 8;
                }
            }
            return out;
        }

        // Q8_K: 256 elements per super-block. Layout:
        //   d(4), qs[256], bsums[16]*2
        std::vector<float> dequantizeQ8_K(const TensorInfo &t)
        {
            constexpr std::size_t kBlockElements = 256;
            constexpr std::size_t kBlockBytes = 4 + 256 + 32;

            std::int64_t nElements = tensorElementCount(t);
            if (nElements % static_cast<std::int64_t>(kBlockElements) != 0)
            {
                throw std::runtime_error("campello_llm: Q8_K tensor element count is not a multiple of 256");
            }
            std::size_t nBlocks = static_cast<std::size_t>(nElements / static_cast<std::int64_t>(kBlockElements));
            if (t.byteLength != nBlocks * kBlockBytes)
            {
                throw std::runtime_error("campello_llm: Q8_K tensor byte length does not match element count");
            }

            std::vector<float> out(static_cast<std::size_t>(nElements));
            const std::uint8_t *ptr = t.data;
            std::size_t outIdx = 0;
            for (std::size_t b = 0; b < nBlocks; ++b)
            {
                float d = decodeFloat32Scale(ptr);
                ptr += 4;
                const auto *qs = reinterpret_cast<const std::int8_t *>(ptr);
                ptr += 256;
                ptr += 32; // skip bsums
                for (std::size_t j = 0; j < kBlockElements; ++j)
                {
                    out[outIdx + j] = d * static_cast<float>(qs[j]);
                }
                outIdx += kBlockElements;
            }
            return out;
        }
    } // namespace

    std::vector<float> dequantizeGgufTensorToFloat32(const TensorInfo &t)
    {
        switch (t.dtype)
        {
        case WeightDType::F32:
            return decodeF32(t);
        case WeightDType::F16:
            return decodeF16(t);
        case WeightDType::BF16:
            return decodeBF16(t);
        case WeightDType::Q4_0:
            return dequantizeQ4_0(t);
        case WeightDType::Q4_1:
            return dequantizeQ4_1(t);
        case WeightDType::Q5_0:
            return dequantizeQ5_0(t);
        case WeightDType::Q5_1:
            return dequantizeQ5_1(t);
        case WeightDType::Q8_0:
            return dequantizeQ8_0(t);
        case WeightDType::Q8_1:
            return dequantizeQ8_1(t);
        case WeightDType::Q2_K:
            return dequantizeQ2_K(t);
        case WeightDType::Q3_K:
            return dequantizeQ3_K(t);
        case WeightDType::Q4_K:
            return dequantizeQ4_K(t);
        case WeightDType::Q5_K:
            return dequantizeQ5_K(t);
        case WeightDType::Q6_K:
            return dequantizeQ6_K(t);
        case WeightDType::Q8_K:
            return dequantizeQ8_K(t);
        default:
            throw std::runtime_error("campello_llm: unsupported GGUF tensor dtype for dequantization");
        }
    }

} // namespace systems::leal::campello_llm
