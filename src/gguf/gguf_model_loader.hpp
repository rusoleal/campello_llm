#pragma once

#include <campello_llm/architecture.hpp>
#include <campello_llm/gguf_reader.hpp>
#include <campello_llm/tokenizer.hpp>
#include <campello_llm/weights_file.hpp>

#include <cstdint>
#include <memory>
#include <string>
#include <unordered_map>
#include <vector>

namespace systems::leal::campello_llm
{

    /**
     * @brief Adapts a `GgufFile`'s tensor names and shapes to the HuggingFace layout
     * that `buildLlamaDecodeGraph()` expects.
     *
     * GGUF stores weight matrices with reversed dims (ggml convention: fastest-varying
     * first) and uses llama.cpp tensor names (`token_embd.weight`, `blk.N.attn_q.weight`,
     * ...). This adapter maps those to HF names (`model.embed_tokens.weight`,
     * `model.layers.N.self_attn.q_proj.weight`, ...) and reverses the shape dims so the
     * flat data layout is interpreted as row-major `[out, in]` / `[vocab, hidden]`.
     *
     * Quantized tensors (Q4_0, Q8_0) are exposed with their original dtype; the
     * architecture's weight-loading helpers dequantize them to Float32 when binding
     * constants.
     */
    class GgufWeightsAdapter : public WeightsFile
    {
    public:
        explicit GgufWeightsAdapter(const GgufFile &gguf);

        const std::vector<TensorInfo> &tensors() const override;
        const TensorInfo *find(const std::string &name) const override;

    private:
        const GgufFile &gguf_;
        std::vector<TensorInfo> adaptedTensors_;
        std::unordered_map<std::string, std::size_t> nameToIndex_;
    };

    /**
     * @brief Reads a `LlamaConfig` from the GGUF metadata block.
     */
    LlamaConfig loadLlamaConfigFromGgufFile(const GgufFile &file);

    /**
     * @brief Reads the embedded tokenizer from the GGUF metadata block.
     */
    std::unique_ptr<Tokenizer> loadTokenizerFromGgufFile(const GgufFile &file);

} // namespace systems::leal::campello_llm
