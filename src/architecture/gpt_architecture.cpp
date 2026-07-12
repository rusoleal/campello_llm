#include <campello_llm/architecture.hpp>

#include <cmath>
#include <cstdint>
#include <stdexcept>
#include <string>
#include <unordered_map>
#include <vector>

#include <campello_nn/graph_builder.hpp>

#include "weight_loading.hpp"

namespace cnn = systems::leal::campello_nn;
using namespace systems::leal::campello_llm;
using namespace systems::leal::campello_llm::internal;

namespace
{
    std::string layerWeight(std::int64_t i, const char *suffix)
    {
        return "h." + std::to_string(i) + "." + suffix;
    }
} // namespace

namespace systems::leal::campello_llm
{

    ArchitectureGraphResult buildGptGraph(std::shared_ptr<cnn::Context> context, const WeightsFile &weights,
                                           const GptConfig &config, std::int64_t seqLen)
    {
        if (config.hiddenSize % config.numAttentionHeads != 0)
        {
            throw std::runtime_error("campello_llm: hiddenSize must be a multiple of numAttentionHeads");
        }
        std::int64_t headDim = config.hiddenSize / config.numAttentionHeads;
        std::int64_t hidden3 = config.hiddenSize; // alias to keep lines below readable

        cnn::GraphBuilder builder(context);

        cnn::Operand causalMask = constantCausalMask(builder, seqLen);
        cnn::Operand attnScale = constantScalarF32(builder, 1.0f / std::sqrt(static_cast<float>(headDim)));

        std::vector<std::int32_t> positionIdsData(static_cast<std::size_t>(seqLen));
        for (std::int64_t i = 0; i < seqLen; ++i)
        {
            positionIdsData[static_cast<std::size_t>(i)] = static_cast<std::int32_t>(i);
        }
        cnn::Operand positionIds = builder.constant({cnn::DataType::Int32, {seqLen}, false, false},
                                                      positionIdsData.data(),
                                                      positionIdsData.size() * sizeof(std::int32_t));

        cnn::Operand inputIds = builder.input("input_ids", {cnn::DataType::Int32, {seqLen}});
        cnn::Operand wte = constantWeight(builder, weights, "wte.weight", {config.vocabSize, config.hiddenSize});
        cnn::Operand wpe =
            constantWeight(builder, weights, "wpe.weight", {config.maxPositionEmbeddings, config.hiddenSize});

        cnn::Operand tokenEmbed = builder.gather(wte, inputIds, 0);     // [seqLen, hiddenSize]
        cnn::Operand positionEmbed = builder.gather(wpe, positionIds, 0); // [seqLen, hiddenSize]
        cnn::Operand hidden = builder.add(tokenEmbed, positionEmbed);

        for (std::int64_t layer = 0; layer < config.numLayers; ++layer)
        {
            cnn::Operand ln1Weight = constantWeight(builder, weights, layerWeight(layer, "ln_1.weight"), {hidden3});
            cnn::Operand ln1Bias = constantWeight(builder, weights, layerWeight(layer, "ln_1.bias"), {hidden3});
            cnn::Operand normed = builder.layerNorm(hidden, ln1Weight, ln1Bias, config.layerNormEps);

            // GPT-2's Conv1D weights are [in, out] already -- no transpose needed,
            // unlike LLaMA's nn.Linear [out, in] (see architecture.hpp).
            cnn::Operand cAttnWeight =
                constantWeight(builder, weights, layerWeight(layer, "attn.c_attn.weight"), {hidden3, 3 * hidden3});
            cnn::Operand cAttnBias =
                constantWeight(builder, weights, layerWeight(layer, "attn.c_attn.bias"), {3 * hidden3});
            cnn::Operand qkv = builder.gemm(normed, cAttnWeight, cAttnBias, 1.0f, 1.0f); // [seqLen, 3*hidden]

            cnn::Operand q = builder.slice(qkv, {0, 0}, {seqLen, hidden3});
            cnn::Operand k = builder.slice(qkv, {0, hidden3}, {seqLen, hidden3});
            cnn::Operand v = builder.slice(qkv, {0, 2 * hidden3}, {seqLen, hidden3});

            q = builder.transpose(builder.reshape(q, {seqLen, config.numAttentionHeads, headDim}), {1, 0, 2});
            k = builder.transpose(builder.reshape(k, {seqLen, config.numAttentionHeads, headDim}), {1, 0, 2});
            v = builder.transpose(builder.reshape(v, {seqLen, config.numAttentionHeads, headDim}), {1, 0, 2});
            // q, k, v: [numAttentionHeads, seqLen, headDim] -- no RoPE, plain MHA (no GQA).

            cnn::Operand kT = builder.transpose(k, {0, 2, 1});
            cnn::Operand scores = builder.mul(builder.matmul(q, kT), attnScale);
            scores = builder.add(scores, causalMask);
            cnn::Operand probs = builder.softmax(scores, -1);
            cnn::Operand ctx = builder.matmul(probs, v); // [numAttentionHeads, seqLen, headDim]

            ctx = builder.reshape(builder.transpose(ctx, {1, 0, 2}), {seqLen, hidden3});

            cnn::Operand cProjWeight =
                constantWeight(builder, weights, layerWeight(layer, "attn.c_proj.weight"), {hidden3, hidden3});
            cnn::Operand cProjBias = constantWeight(builder, weights, layerWeight(layer, "attn.c_proj.bias"), {hidden3});
            cnn::Operand attnOut = builder.gemm(ctx, cProjWeight, cProjBias, 1.0f, 1.0f);
            hidden = builder.add(hidden, attnOut); // residual 1

            cnn::Operand ln2Weight = constantWeight(builder, weights, layerWeight(layer, "ln_2.weight"), {hidden3});
            cnn::Operand ln2Bias = constantWeight(builder, weights, layerWeight(layer, "ln_2.bias"), {hidden3});
            cnn::Operand normed2 = builder.layerNorm(hidden, ln2Weight, ln2Bias, config.layerNormEps);

            cnn::Operand cFcWeight = constantWeight(builder, weights, layerWeight(layer, "mlp.c_fc.weight"),
                                                     {hidden3, config.intermediateSize});
            cnn::Operand cFcBias =
                constantWeight(builder, weights, layerWeight(layer, "mlp.c_fc.bias"), {config.intermediateSize});
            cnn::Operand ff = builder.gemm(normed2, cFcWeight, cFcBias, 1.0f, 1.0f);
            ff = builder.gelu(ff);

            cnn::Operand cProjMlpWeight = constantWeight(builder, weights, layerWeight(layer, "mlp.c_proj.weight"),
                                                          {config.intermediateSize, hidden3});
            cnn::Operand cProjMlpBias = constantWeight(builder, weights, layerWeight(layer, "mlp.c_proj.bias"), {hidden3});
            cnn::Operand ffOut = builder.gemm(ff, cProjMlpWeight, cProjMlpBias, 1.0f, 1.0f);
            hidden = builder.add(hidden, ffOut); // residual 2
        }

        cnn::Operand lnFWeight = constantWeight(builder, weights, "ln_f.weight", {hidden3});
        cnn::Operand lnFBias = constantWeight(builder, weights, "ln_f.bias", {hidden3});
        hidden = builder.layerNorm(hidden, lnFWeight, lnFBias, config.layerNormEps);

        LinearWeight lmHeadWeight =
            loadLinearWeight(builder, weights, "wte.weight", config.vocabSize, config.hiddenSize);
        cnn::Operand logits = applyLinear(builder, hidden, lmHeadWeight, config.vocabSize, config.hiddenSize); // [seqLen, vocabSize]

        std::unordered_map<std::string, cnn::Operand> outputOperands = {{"logits", logits}};
        ArchitectureGraphResult result;
        // serialize() before build() -- GraphBuilder::build() moves (rather than
        // copies) the builder's IR into the compiled graph, so it must run last,
        // after anything else that still needs the builder's IR intact (see
        // campello_nn's Backend::compileGraph() doc comment).
        result.serializedGraph = builder.serialize(outputOperands);
        result.graph = builder.build(outputOperands);
        result.inputs["input_ids"] = {cnn::DataType::Int32, {seqLen}, false, true};
        result.outputs["logits"] = {cnn::DataType::Float32, {seqLen, config.vocabSize}, true, false};
        return result;
    }

} // namespace systems::leal::campello_llm
