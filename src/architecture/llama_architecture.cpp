#include <campello_llm/architecture.hpp>

#include <cmath>
#include <stdexcept>
#include <string>
#include <unordered_map>

#include <campello_nn/graph_builder.hpp>

#include "weight_loading.hpp"

namespace cnn = systems::leal::campello_nn;
using namespace systems::leal::campello_llm;
using namespace systems::leal::campello_llm::internal;

namespace
{
    std::string layerWeight(std::int64_t i, const char *suffix)
    {
        return "model.layers." + std::to_string(i) + "." + suffix;
    }
} // namespace

namespace systems::leal::campello_llm
{
    std::string llamaKeyCacheInputName(std::int64_t layer) { return "k_cache_in_" + std::to_string(layer); }
    std::string llamaValueCacheInputName(std::int64_t layer) { return "v_cache_in_" + std::to_string(layer); }
    std::string llamaKeyCacheOutputName(std::int64_t layer) { return "k_new_" + std::to_string(layer); }
    std::string llamaValueCacheOutputName(std::int64_t layer) { return "v_new_" + std::to_string(layer); }
} // namespace systems::leal::campello_llm

namespace systems::leal::campello_llm
{

    ArchitectureGraphResult buildLlamaGraph(std::shared_ptr<cnn::Context> context, const WeightsFile &weights,
                                             const LlamaConfig &config, std::int64_t seqLen)
    {
        if (config.numAttentionHeads % config.numKeyValueHeads != 0)
        {
            throw std::runtime_error("campello_llm: numAttentionHeads must be a multiple of numKeyValueHeads");
        }
        std::int64_t headDim = config.hiddenSize / config.numAttentionHeads;
        if (headDim * config.numAttentionHeads != config.hiddenSize)
        {
            throw std::runtime_error("campello_llm: hiddenSize must be a multiple of numAttentionHeads");
        }
        std::int64_t groupSize = config.numAttentionHeads / config.numKeyValueHeads;
        std::int64_t kvDim = config.numKeyValueHeads * headDim;

        cnn::GraphBuilder builder(context);

        auto [cosOp, sinOp] = constantRope(builder, seqLen, headDim, config.ropeTheta);
        cnn::Operand causalMask = constantCausalMask(builder, seqLen);
        cnn::Operand attnScale = constantScalarF32(builder, 1.0f / std::sqrt(static_cast<float>(headDim)));

        cnn::Operand inputIds = builder.input("input_ids", {cnn::DataType::Int32, {seqLen}});
        cnn::Operand embedTokens =
            constantWeight(builder, weights, "model.embed_tokens.weight", {config.vocabSize, config.hiddenSize});
        cnn::Operand hidden = builder.gather(embedTokens, inputIds, 0); // [seqLen, hiddenSize]

        for (std::int64_t layer = 0; layer < config.numLayers; ++layer)
        {
            cnn::Operand attnNormWeight =
                constantWeight(builder, weights, layerWeight(layer, "input_layernorm.weight"), {config.hiddenSize});
            cnn::Operand normed = builder.rmsNorm(hidden, attnNormWeight, config.rmsNormEps);

            cnn::Operand qWeight = constantLinearWeightTransposed(
                builder, weights, layerWeight(layer, "self_attn.q_proj.weight"), config.hiddenSize, config.hiddenSize);
            cnn::Operand kWeight = constantLinearWeightTransposed(
                builder, weights, layerWeight(layer, "self_attn.k_proj.weight"), kvDim, config.hiddenSize);
            cnn::Operand vWeight = constantLinearWeightTransposed(
                builder, weights, layerWeight(layer, "self_attn.v_proj.weight"), kvDim, config.hiddenSize);

            cnn::Operand q = builder.matmul(normed, qWeight); // [seqLen, hiddenSize]
            cnn::Operand k = builder.matmul(normed, kWeight); // [seqLen, kvDim]
            cnn::Operand v = builder.matmul(normed, vWeight); // [seqLen, kvDim]

            q = builder.transpose(builder.reshape(q, {seqLen, config.numAttentionHeads, headDim}), {1, 0, 2});
            k = builder.transpose(builder.reshape(k, {seqLen, config.numKeyValueHeads, headDim}), {1, 0, 2});
            v = builder.transpose(builder.reshape(v, {seqLen, config.numKeyValueHeads, headDim}), {1, 0, 2});

            q = builder.rotaryEmbedding(q, cosOp, sinOp); // [numAttentionHeads, seqLen, headDim]
            k = builder.rotaryEmbedding(k, cosOp, sinOp); // [numKeyValueHeads, seqLen, headDim]

            k = repeatKvHeads(builder, k, config.numKeyValueHeads, groupSize, seqLen, headDim);
            v = repeatKvHeads(builder, v, config.numKeyValueHeads, groupSize, seqLen, headDim);
            // k, v are now [numAttentionHeads, seqLen, headDim]

            cnn::Operand kT = builder.transpose(k, {0, 2, 1}); // [numAttentionHeads, headDim, seqLen]
            cnn::Operand scores = builder.matmul(q, kT);       // [numAttentionHeads, seqLen, seqLen]
            scores = builder.mul(scores, attnScale);
            scores = builder.add(scores, causalMask);
            cnn::Operand probs = builder.softmax(scores, -1);
            cnn::Operand ctx = builder.matmul(probs, v); // [numAttentionHeads, seqLen, headDim]

            ctx = builder.transpose(ctx, {1, 0, 2});                              // [seqLen, numAttentionHeads, headDim]
            ctx = builder.reshape(ctx, {seqLen, config.hiddenSize});              // [seqLen, hiddenSize]

            cnn::Operand oWeight = constantLinearWeightTransposed(
                builder, weights, layerWeight(layer, "self_attn.o_proj.weight"), config.hiddenSize, config.hiddenSize);
            cnn::Operand attnOut = builder.matmul(ctx, oWeight);
            hidden = builder.add(hidden, attnOut); // residual 1

            cnn::Operand ffnNormWeight = constantWeight(
                builder, weights, layerWeight(layer, "post_attention_layernorm.weight"), {config.hiddenSize});
            cnn::Operand normed2 = builder.rmsNorm(hidden, ffnNormWeight, config.rmsNormEps);

            cnn::Operand gateWeight = constantLinearWeightTransposed(
                builder, weights, layerWeight(layer, "mlp.gate_proj.weight"), config.intermediateSize, config.hiddenSize);
            cnn::Operand upWeight = constantLinearWeightTransposed(
                builder, weights, layerWeight(layer, "mlp.up_proj.weight"), config.intermediateSize, config.hiddenSize);
            cnn::Operand downWeight = constantLinearWeightTransposed(
                builder, weights, layerWeight(layer, "mlp.down_proj.weight"), config.hiddenSize, config.intermediateSize);

            cnn::Operand gate = builder.matmul(normed2, gateWeight);
            cnn::Operand up = builder.matmul(normed2, upWeight);
            cnn::Operand silu = builder.mul(gate, builder.sigmoid(gate)); // SiLU(gate) = gate * sigmoid(gate)
            cnn::Operand ffnHidden = builder.mul(silu, up);
            cnn::Operand down = builder.matmul(ffnHidden, downWeight);

            hidden = builder.add(hidden, down); // residual 2
        }

        cnn::Operand finalNormWeight = constantWeight(builder, weights, "model.norm.weight", {config.hiddenSize});
        hidden = builder.rmsNorm(hidden, finalNormWeight, config.rmsNormEps);

        cnn::Operand lmHeadWeight =
            constantLinearWeightTransposed(builder, weights, "lm_head.weight", config.vocabSize, config.hiddenSize);
        cnn::Operand logits = builder.matmul(hidden, lmHeadWeight); // [seqLen, vocabSize]

        std::unordered_map<std::string, cnn::Operand> outputOperands = {{"logits", logits}};
        ArchitectureGraphResult result;
        result.graph = builder.build(outputOperands);
        result.serializedGraph = builder.serialize(outputOperands);
        result.inputs["input_ids"] = {cnn::DataType::Int32, {seqLen}, false, true};
        result.outputs["logits"] = {cnn::DataType::Float32, {seqLen, config.vocabSize}, true, false};
        return result;
    }

    ArchitectureGraphResult buildLlamaDecodeGraph(std::shared_ptr<cnn::Context> context, const WeightsFile &weights,
                                                   const LlamaConfig &config, std::int64_t maxSequenceLength)
    {
        if (config.numAttentionHeads % config.numKeyValueHeads != 0)
        {
            throw std::runtime_error("campello_llm: numAttentionHeads must be a multiple of numKeyValueHeads");
        }
        std::int64_t headDim = config.hiddenSize / config.numAttentionHeads;
        if (headDim * config.numAttentionHeads != config.hiddenSize)
        {
            throw std::runtime_error("campello_llm: hiddenSize must be a multiple of numAttentionHeads");
        }
        std::int64_t groupSize = config.numAttentionHeads / config.numKeyValueHeads;
        std::int64_t cacheAndCurrentLen = maxSequenceLength + 1;

        cnn::GraphBuilder builder(context);

        cnn::Operand inputIds = builder.input("input_ids", {cnn::DataType::Int32, {1}});
        cnn::Operand ropeCos = builder.input("rope_cos", {cnn::DataType::Float32, {1, headDim}});
        cnn::Operand ropeSin = builder.input("rope_sin", {cnn::DataType::Float32, {1, headDim}});
        cnn::Operand attnMask = builder.input("attn_mask", {cnn::DataType::Float32, {1, cacheAndCurrentLen}});
        cnn::Operand attnScale = constantScalarF32(builder, 1.0f / std::sqrt(static_cast<float>(headDim)));

        cnn::Operand embedTokens =
            constantWeight(builder, weights, "model.embed_tokens.weight", {config.vocabSize, config.hiddenSize});
        cnn::Operand hidden = builder.gather(embedTokens, inputIds, 0); // [1, hiddenSize]

        std::unordered_map<std::string, cnn::Operand> outputs;

        for (std::int64_t layer = 0; layer < config.numLayers; ++layer)
        {
            cnn::Operand attnNormWeight =
                constantWeight(builder, weights, layerWeight(layer, "input_layernorm.weight"), {config.hiddenSize});
            cnn::Operand normed = builder.rmsNorm(hidden, attnNormWeight, config.rmsNormEps);

            cnn::Operand qWeight = constantLinearWeightTransposed(
                builder, weights, layerWeight(layer, "self_attn.q_proj.weight"), config.hiddenSize, config.hiddenSize);
            cnn::Operand kWeight = constantLinearWeightTransposed(
                builder, weights, layerWeight(layer, "self_attn.k_proj.weight"),
                config.numKeyValueHeads * headDim, config.hiddenSize);
            cnn::Operand vWeight = constantLinearWeightTransposed(
                builder, weights, layerWeight(layer, "self_attn.v_proj.weight"),
                config.numKeyValueHeads * headDim, config.hiddenSize);

            cnn::Operand q = builder.matmul(normed, qWeight); // [1, hiddenSize]
            cnn::Operand kNew = builder.matmul(normed, kWeight); // [1, kvDim]
            cnn::Operand vNew = builder.matmul(normed, vWeight); // [1, kvDim]

            q = builder.transpose(builder.reshape(q, {1, config.numAttentionHeads, headDim}), {1, 0, 2});
            kNew = builder.transpose(builder.reshape(kNew, {1, config.numKeyValueHeads, headDim}), {1, 0, 2});
            vNew = builder.transpose(builder.reshape(vNew, {1, config.numKeyValueHeads, headDim}), {1, 0, 2});

            q = builder.rotaryEmbedding(q, ropeCos, ropeSin);       // [numAttentionHeads, 1, headDim]
            kNew = builder.rotaryEmbedding(kNew, ropeCos, ropeSin); // [numKeyValueHeads, 1, headDim]
            // kNew/vNew (pre-cache-concat) are exactly this layer's cache outputs -- the
            // freshly rotated K / projected V for just the current token.
            outputs[llamaKeyCacheOutputName(layer)] = kNew;
            outputs[llamaValueCacheOutputName(layer)] = vNew;

            cnn::Operand kCacheIn = builder.input(llamaKeyCacheInputName(layer),
                                                   {cnn::DataType::Float32, {config.numKeyValueHeads, maxSequenceLength, headDim}});
            cnn::Operand vCacheIn = builder.input(llamaValueCacheInputName(layer),
                                                   {cnn::DataType::Float32, {config.numKeyValueHeads, maxSequenceLength, headDim}});

            cnn::Operand kFull = builder.concat({kCacheIn, kNew}, 1); // [numKeyValueHeads, maxSequenceLength+1, headDim]
            cnn::Operand vFull = builder.concat({vCacheIn, vNew}, 1);

            kFull = repeatKvHeads(builder, kFull, config.numKeyValueHeads, groupSize, cacheAndCurrentLen, headDim);
            vFull = repeatKvHeads(builder, vFull, config.numKeyValueHeads, groupSize, cacheAndCurrentLen, headDim);
            // kFull, vFull are now [numAttentionHeads, maxSequenceLength+1, headDim]

            cnn::Operand kT = builder.transpose(kFull, {0, 2, 1}); // [numAttentionHeads, headDim, maxSequenceLength+1]
            cnn::Operand scores = builder.matmul(q, kT);           // [numAttentionHeads, 1, maxSequenceLength+1]
            scores = builder.mul(scores, attnScale);
            scores = builder.add(scores, attnMask);
            cnn::Operand probs = builder.softmax(scores, -1);
            cnn::Operand ctx = builder.matmul(probs, vFull); // [numAttentionHeads, 1, headDim]

            ctx = builder.transpose(ctx, {1, 0, 2});                 // [1, numAttentionHeads, headDim]
            ctx = builder.reshape(ctx, {1, config.hiddenSize});      // [1, hiddenSize]

            cnn::Operand oWeight = constantLinearWeightTransposed(
                builder, weights, layerWeight(layer, "self_attn.o_proj.weight"), config.hiddenSize, config.hiddenSize);
            cnn::Operand attnOut = builder.matmul(ctx, oWeight);
            hidden = builder.add(hidden, attnOut); // residual 1

            cnn::Operand ffnNormWeight = constantWeight(
                builder, weights, layerWeight(layer, "post_attention_layernorm.weight"), {config.hiddenSize});
            cnn::Operand normed2 = builder.rmsNorm(hidden, ffnNormWeight, config.rmsNormEps);

            cnn::Operand gateWeight = constantLinearWeightTransposed(
                builder, weights, layerWeight(layer, "mlp.gate_proj.weight"), config.intermediateSize, config.hiddenSize);
            cnn::Operand upWeight = constantLinearWeightTransposed(
                builder, weights, layerWeight(layer, "mlp.up_proj.weight"), config.intermediateSize, config.hiddenSize);
            cnn::Operand downWeight = constantLinearWeightTransposed(
                builder, weights, layerWeight(layer, "mlp.down_proj.weight"), config.hiddenSize, config.intermediateSize);

            cnn::Operand gate = builder.matmul(normed2, gateWeight);
            cnn::Operand up = builder.matmul(normed2, upWeight);
            cnn::Operand silu = builder.mul(gate, builder.sigmoid(gate)); // SiLU(gate) = gate * sigmoid(gate)
            cnn::Operand ffnHidden = builder.mul(silu, up);
            cnn::Operand down = builder.matmul(ffnHidden, downWeight);

            hidden = builder.add(hidden, down); // residual 2
        }

        cnn::Operand finalNormWeight = constantWeight(builder, weights, "model.norm.weight", {config.hiddenSize});
        hidden = builder.rmsNorm(hidden, finalNormWeight, config.rmsNormEps);

        cnn::Operand lmHeadWeight =
            constantLinearWeightTransposed(builder, weights, "lm_head.weight", config.vocabSize, config.hiddenSize);
        cnn::Operand logits = builder.matmul(hidden, lmHeadWeight); // [1, vocabSize]
        outputs["logits"] = logits;

        ArchitectureGraphResult result;
        result.graph = builder.build(outputs);
        result.serializedGraph = builder.serialize(outputs);
        result.inputs["input_ids"] = {cnn::DataType::Int32, {1}, false, true};
        result.inputs["rope_cos"] = {cnn::DataType::Float32, {1, headDim}, false, true};
        result.inputs["rope_sin"] = {cnn::DataType::Float32, {1, headDim}, false, true};
        result.inputs["attn_mask"] = {cnn::DataType::Float32, {1, cacheAndCurrentLen}, false, true};
        result.outputs["logits"] = {cnn::DataType::Float32, {1, config.vocabSize}, true, false};
        for (std::int64_t layer = 0; layer < config.numLayers; ++layer)
        {
            cnn::TensorDescriptor cacheDesc = {cnn::DataType::Float32,
                                                {config.numKeyValueHeads, maxSequenceLength, headDim}, false, true};
            result.inputs[llamaKeyCacheInputName(layer)] = cacheDesc;
            result.inputs[llamaValueCacheInputName(layer)] = cacheDesc;
            cnn::TensorDescriptor newDesc = {cnn::DataType::Float32, {config.numKeyValueHeads, 1, headDim}, true, false};
            result.outputs[llamaKeyCacheOutputName(layer)] = newDesc;
            result.outputs[llamaValueCacheOutputName(layer)] = newDesc;
        }
        return result;
    }

} // namespace systems::leal::campello_llm
