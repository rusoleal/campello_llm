#include <gtest/gtest.h>

#include <campello_nn/float16.hpp>

#include "../src/gguf/gguf_dequantize.hpp"
#include <campello_llm/weights_file.hpp>

#include <cstdint>
#include <cstring>
#include <vector>

using namespace systems::leal::campello_llm;
namespace cnn = systems::leal::campello_nn;

namespace
{
    void appendLE16(std::vector<std::uint8_t> &buf, std::uint16_t value)
    {
        buf.push_back(static_cast<std::uint8_t>(value & 0xFF));
        buf.push_back(static_cast<std::uint8_t>((value >> 8) & 0xFF));
    }

    void appendLE32(std::vector<std::uint8_t> &buf, std::uint32_t value)
    {
        buf.push_back(static_cast<std::uint8_t>(value & 0xFF));
        buf.push_back(static_cast<std::uint8_t>((value >> 8) & 0xFF));
        buf.push_back(static_cast<std::uint8_t>((value >> 16) & 0xFF));
        buf.push_back(static_cast<std::uint8_t>((value >> 24) & 0xFF));
    }

    void appendFloat(std::vector<std::uint8_t> &buf, float value)
    {
        static_assert(sizeof(float) == sizeof(std::uint32_t));
        std::uint32_t bits;
        std::memcpy(&bits, &value, sizeof(bits));
        appendLE32(buf, bits);
    }

    TensorInfo makeTensor(WeightDType dtype, const std::vector<std::int64_t> &shape,
                          const std::vector<std::uint8_t> &bytes)
    {
        TensorInfo t;
        t.dtype = dtype;
        t.shape = shape;
        // TensorInfo.data is a non-owning pointer; tests must keep `bytes` alive.
        t.data = bytes.data();
        t.byteLength = bytes.size();
        return t;
    }
} // namespace

TEST(GgufDequantize, Q8_0Block)
{
    // One Q8_0 block of 32 elements, scale = 2.0, qs[i] = i - 16.
    std::vector<std::uint8_t> bytes;
    appendLE16(bytes, cnn::encodeFloat16(2.0f));
    for (std::int32_t i = 0; i < 32; ++i)
    {
        bytes.push_back(static_cast<std::uint8_t>(i - 16));
    }

    TensorInfo t = makeTensor(WeightDType::Q8_0, {32}, bytes);
    std::vector<float> out = dequantizeGgufTensorToFloat32(t);
    ASSERT_EQ(out.size(), 32u);
    for (std::int32_t i = 0; i < 32; ++i)
    {
        EXPECT_FLOAT_EQ(out[static_cast<std::size_t>(i)], static_cast<float>(i - 16) * 2.0f);
    }
}

TEST(GgufDequantize, Q4_0Block)
{
    // One Q4_0 block of 32 elements, scale = 1.5.
    // qs[j] stores element j in its low nibble and element j+16 in its high nibble.
    // Set low nibble = j, high nibble = 15 - j.
    // Expected: y[j]      = (j - 8) * 1.5
    //           y[j + 16] = ((15 - j) - 8) * 1.5 = (7 - j) * 1.5
    std::vector<std::uint8_t> bytes;
    appendLE16(bytes, cnn::encodeFloat16(1.5f));
    for (std::int32_t j = 0; j < 16; ++j)
    {
        std::uint8_t low = static_cast<std::uint8_t>(j);
        std::uint8_t high = static_cast<std::uint8_t>(15 - j);
        bytes.push_back(low | (high << 4));
    }

    TensorInfo t = makeTensor(WeightDType::Q4_0, {32}, bytes);
    std::vector<float> out = dequantizeGgufTensorToFloat32(t);
    ASSERT_EQ(out.size(), 32u);
    for (std::int32_t j = 0; j < 16; ++j)
    {
        EXPECT_FLOAT_EQ(out[static_cast<std::size_t>(j)], static_cast<float>(j - 8) * 1.5f);
        EXPECT_FLOAT_EQ(out[static_cast<std::size_t>(j + 16)], static_cast<float>(7 - j) * 1.5f);
    }
}

TEST(GgufDequantize, F16RoundTrip)
{
    std::vector<std::uint8_t> bytes;
    appendLE16(bytes, cnn::encodeFloat16(1.0f));
    appendLE16(bytes, cnn::encodeFloat16(-2.0f));
    appendLE16(bytes, cnn::encodeFloat16(0.0f));

    TensorInfo t = makeTensor(WeightDType::F16, {3}, bytes);
    std::vector<float> out = dequantizeGgufTensorToFloat32(t);
    ASSERT_EQ(out.size(), 3u);
    EXPECT_FLOAT_EQ(out[0], 1.0f);
    EXPECT_FLOAT_EQ(out[1], -2.0f);
    EXPECT_FLOAT_EQ(out[2], 0.0f);
}

TEST(GgufDequantize, Q8_0TwoBlocks)
{
    // Two blocks: first scale 1.0 all zeros, second scale -1.0 all ones.
    std::vector<std::uint8_t> bytes;
    appendLE16(bytes, cnn::encodeFloat16(1.0f));
    for (std::int32_t i = 0; i < 32; ++i)
    {
        bytes.push_back(0);
    }
    appendLE16(bytes, cnn::encodeFloat16(-1.0f));
    for (std::int32_t i = 0; i < 32; ++i)
    {
        bytes.push_back(1);
    }

    TensorInfo t = makeTensor(WeightDType::Q8_0, {64}, bytes);
    std::vector<float> out = dequantizeGgufTensorToFloat32(t);
    ASSERT_EQ(out.size(), 64u);
    for (std::size_t i = 0; i < 32; ++i)
    {
        EXPECT_FLOAT_EQ(out[i], 0.0f);
    }
    for (std::size_t i = 32; i < 64; ++i)
    {
        EXPECT_FLOAT_EQ(out[i], -1.0f);
    }
}

TEST(GgufDequantize, Q4_1Block)
{
    // One Q4_1 block: d = 2.0, m = 1.0, qs nibbles low=j, high=15-j.
    std::vector<std::uint8_t> bytes;
    appendLE16(bytes, cnn::encodeFloat16(2.0f));
    appendLE16(bytes, cnn::encodeFloat16(1.0f));
    for (std::int32_t j = 0; j < 16; ++j)
    {
        std::uint8_t low = static_cast<std::uint8_t>(j);
        std::uint8_t high = static_cast<std::uint8_t>(15 - j);
        bytes.push_back(low | (high << 4));
    }

    TensorInfo t = makeTensor(WeightDType::Q4_1, {32}, bytes);
    std::vector<float> out = dequantizeGgufTensorToFloat32(t);
    ASSERT_EQ(out.size(), 32u);
    for (std::int32_t j = 0; j < 16; ++j)
    {
        EXPECT_FLOAT_EQ(out[static_cast<std::size_t>(j)], static_cast<float>(j) * 2.0f + 1.0f);
        EXPECT_FLOAT_EQ(out[static_cast<std::size_t>(j + 16)],
                        static_cast<float>(15 - j) * 2.0f + 1.0f);
    }
}

TEST(GgufDequantize, Q5_0Block)
{
    // One Q5_0 block with all zero quants and no high bits -> every value is -16 * d.
    std::vector<std::uint8_t> bytes;
    appendLE16(bytes, cnn::encodeFloat16(2.0f));
    for (int i = 0; i < 4; ++i)
        bytes.push_back(0);
    for (int i = 0; i < 16; ++i)
        bytes.push_back(0);

    TensorInfo t = makeTensor(WeightDType::Q5_0, {32}, bytes);
    std::vector<float> out = dequantizeGgufTensorToFloat32(t);
    ASSERT_EQ(out.size(), 32u);
    for (float v : out)
    {
        EXPECT_FLOAT_EQ(v, -32.0f);
    }
}

TEST(GgufDequantize, Q5_1Block)
{
    // One Q5_1 block with all zero quants and no high bits -> every value is m.
    std::vector<std::uint8_t> bytes;
    appendLE16(bytes, cnn::encodeFloat16(2.0f));
    appendLE16(bytes, cnn::encodeFloat16(5.0f));
    for (int i = 0; i < 4; ++i)
        bytes.push_back(0);
    for (int i = 0; i < 16; ++i)
        bytes.push_back(0);

    TensorInfo t = makeTensor(WeightDType::Q5_1, {32}, bytes);
    std::vector<float> out = dequantizeGgufTensorToFloat32(t);
    ASSERT_EQ(out.size(), 32u);
    for (float v : out)
    {
        EXPECT_FLOAT_EQ(v, 5.0f);
    }
}

TEST(GgufDequantize, Q8_1Block)
{
    // One Q8_1 block: d = 3.0, qs[i] = i - 16.
    std::vector<std::uint8_t> bytes;
    appendLE16(bytes, cnn::encodeFloat16(3.0f));
    appendLE16(bytes, cnn::encodeFloat16(0.0f)); // s is unused
    for (std::int32_t i = 0; i < 32; ++i)
    {
        bytes.push_back(static_cast<std::uint8_t>(i - 16));
    }

    TensorInfo t = makeTensor(WeightDType::Q8_1, {32}, bytes);
    std::vector<float> out = dequantizeGgufTensorToFloat32(t);
    ASSERT_EQ(out.size(), 32u);
    for (std::int32_t i = 0; i < 32; ++i)
    {
        EXPECT_FLOAT_EQ(out[static_cast<std::size_t>(i)], static_cast<float>(i - 16) * 3.0f);
    }
}

TEST(GgufDequantize, Q2_KBlock)
{
    // One Q2_K super-block (256 elements). d=1, min=0, all scales=0x01, all qs=0x55
    // so every 2-bit quant is 1.
    std::vector<std::uint8_t> bytes;
    for (int i = 0; i < 16; ++i)
        bytes.push_back(0x01);
    for (int i = 0; i < 64; ++i)
        bytes.push_back(0x55);
    appendLE16(bytes, cnn::encodeFloat16(1.0f));
    appendLE16(bytes, cnn::encodeFloat16(0.0f));

    TensorInfo t = makeTensor(WeightDType::Q2_K, {256}, bytes);
    std::vector<float> out = dequantizeGgufTensorToFloat32(t);
    ASSERT_EQ(out.size(), 256u);
    for (float v : out)
    {
        EXPECT_FLOAT_EQ(v, 1.0f);
    }
}

TEST(GgufDequantize, Q3_KZeros)
{
    // One Q3_K super-block with d=0 so every output is zero regardless of quants.
    std::vector<std::uint8_t> bytes;
    for (int i = 0; i < 32; ++i)
        bytes.push_back(0);
    for (int i = 0; i < 64; ++i)
        bytes.push_back(0);
    for (int i = 0; i < 12; ++i)
        bytes.push_back(0);
    appendLE16(bytes, cnn::encodeFloat16(0.0f));

    TensorInfo t = makeTensor(WeightDType::Q3_K, {256}, bytes);
    std::vector<float> out = dequantizeGgufTensorToFloat32(t);
    ASSERT_EQ(out.size(), 256u);
    for (float v : out)
    {
        EXPECT_FLOAT_EQ(v, 0.0f);
    }
}

TEST(GgufDequantize, Q4_KBlock)
{
    // One Q4_K super-block (256 elements). d=1, min=1, all sub-block scales/mins=1.
    // qs bytes are 0x12 -> low nibble 2, high nibble 1. First half of each 64-group = 2*1-1=1,
    // second half = 1*1-1=0.
    std::vector<std::uint8_t> bytes;
    appendLE16(bytes, cnn::encodeFloat16(1.0f));
    appendLE16(bytes, cnn::encodeFloat16(1.0f));
    // scales[0..3]=1, scales[4..7]=1, scales[8..11]=0x11 -> every getScaleMinK4 returns 1.
    for (int i = 0; i < 4; ++i)
        bytes.push_back(0x01);
    for (int i = 0; i < 4; ++i)
        bytes.push_back(0x01);
    for (int i = 0; i < 4; ++i)
        bytes.push_back(0x11);
    for (int i = 0; i < 128; ++i)
        bytes.push_back(0x12);

    TensorInfo t = makeTensor(WeightDType::Q4_K, {256}, bytes);
    std::vector<float> out = dequantizeGgufTensorToFloat32(t);
    ASSERT_EQ(out.size(), 256u);
    for (std::size_t i = 0; i < 256; ++i)
    {
        float expected = ((i % 64) < 32) ? 1.0f : 0.0f;
        EXPECT_FLOAT_EQ(out[i], expected);
    }
}

TEST(GgufDequantize, Q5_KBlock)
{
    // One Q5_K super-block. d=1, min=0, all scales=1, no high bits.
    // qs bytes 0x12 -> first half values 2, second half values 1.
    std::vector<std::uint8_t> bytes;
    appendLE16(bytes, cnn::encodeFloat16(1.0f));
    appendLE16(bytes, cnn::encodeFloat16(0.0f));
    for (int i = 0; i < 4; ++i)
        bytes.push_back(0x01);
    for (int i = 0; i < 4; ++i)
        bytes.push_back(0x01);
    for (int i = 0; i < 4; ++i)
        bytes.push_back(0x11);
    for (int i = 0; i < 32; ++i)
        bytes.push_back(0);
    for (int i = 0; i < 128; ++i)
        bytes.push_back(0x12);

    TensorInfo t = makeTensor(WeightDType::Q5_K, {256}, bytes);
    std::vector<float> out = dequantizeGgufTensorToFloat32(t);
    ASSERT_EQ(out.size(), 256u);
    for (std::size_t i = 0; i < 256; ++i)
    {
        float expected = ((i % 64) < 32) ? 2.0f : 1.0f;
        EXPECT_FLOAT_EQ(out[i], expected);
    }
}

TEST(GgufDequantize, Q6_KBlock)
{
    // One Q6_K super-block. d=1, all scales=1, all low/high bits zero -> quant = -32.
    std::vector<std::uint8_t> bytes;
    for (int i = 0; i < 128; ++i)
        bytes.push_back(0);
    for (int i = 0; i < 64; ++i)
        bytes.push_back(0);
    for (int i = 0; i < 16; ++i)
        bytes.push_back(0x01);
    appendLE16(bytes, cnn::encodeFloat16(1.0f));

    TensorInfo t = makeTensor(WeightDType::Q6_K, {256}, bytes);
    std::vector<float> out = dequantizeGgufTensorToFloat32(t);
    ASSERT_EQ(out.size(), 256u);
    for (float v : out)
    {
        EXPECT_FLOAT_EQ(v, -32.0f);
    }
}

TEST(GgufDequantize, Q8_KBlock)
{
    // One Q8_K super-block. d = 2.5, qs[i] = i - 128.
    std::vector<std::uint8_t> bytes;
    appendFloat(bytes, 2.5f);
    for (std::int32_t i = 0; i < 256; ++i)
    {
        bytes.push_back(static_cast<std::uint8_t>(i - 128));
    }
    for (int i = 0; i < 32; ++i)
        bytes.push_back(0); // bsums

    TensorInfo t = makeTensor(WeightDType::Q8_K, {256}, bytes);
    std::vector<float> out = dequantizeGgufTensorToFloat32(t);
    ASSERT_EQ(out.size(), 256u);
    for (std::int32_t i = 0; i < 256; ++i)
    {
        EXPECT_FLOAT_EQ(out[static_cast<std::size_t>(i)], static_cast<float>(i - 128) * 2.5f);
    }
}
