#!/usr/bin/env python3
"""Generates tests/fixtures/gpt_test_weights.safetensors -- a tiny, synthetic
GPT-2-shaped model (LayerNorm, GELU, learned positional embeddings, tied
lm_head, Conv1D [in,out] weight layout) with fixed-seed random Float32 weights
-- and independently computes the expected forward-pass logits in plain numpy,
printed as a C++-literal array to paste into test_gpt_architecture.cpp.

This script IS the "independently hand/numpy-computed reference" TODO.md's Phase 3
test plan calls for -- it reimplements the forward pass from scratch in numpy,
deliberately not sharing any code with src/architecture/gpt_architecture.cpp.

Regenerate with: python3 generate_gpt_test_fixture.py (requires `pip install
safetensors numpy`). Only rerun if the tiny config below changes.
"""

import numpy as np
from safetensors.numpy import save_file

VOCAB_SIZE = 7
HIDDEN_SIZE = 8
NUM_LAYERS = 2
NUM_ATTENTION_HEADS = 2
HEAD_DIM = HIDDEN_SIZE // NUM_ATTENTION_HEADS
INTERMEDIATE_SIZE = 12
LAYER_NORM_EPS = 1e-5
MAX_POSITION_EMBEDDINGS = 4
SEQ_LEN = 3
INPUT_IDS = np.array([2, 5, 0], dtype=np.int64)

rng = np.random.default_rng(7)


def randw(*shape):
    return rng.normal(scale=0.2, size=shape).astype(np.float32)


def layer_norm(x, weight, bias, eps):
    mean = np.mean(x.astype(np.float64), axis=-1, keepdims=True)
    var = np.mean((x.astype(np.float64) - mean) ** 2, axis=-1, keepdims=True)
    normed = (x - mean) / np.sqrt(var + eps)
    return (normed * weight + bias).astype(np.float32)


def gelu_exact(x):
    from math import erf
    vec_erf = np.vectorize(erf)
    return (0.5 * x * (1.0 + vec_erf(x * 0.70710678118654752))).astype(np.float32)


def softmax(x, axis):
    x = x - np.max(x, axis=axis, keepdims=True)
    e = np.exp(x)
    return (e / np.sum(e, axis=axis, keepdims=True)).astype(np.float32)


tensors = {}
tensors["wte.weight"] = randw(VOCAB_SIZE, HIDDEN_SIZE)
tensors["wpe.weight"] = randw(MAX_POSITION_EMBEDDINGS, HIDDEN_SIZE)
for i in range(NUM_LAYERS):
    p = f"h.{i}."
    tensors[p + "ln_1.weight"] = np.ones(HIDDEN_SIZE, dtype=np.float32) + randw(HIDDEN_SIZE) * 0.1
    tensors[p + "ln_1.bias"] = randw(HIDDEN_SIZE) * 0.1
    tensors[p + "attn.c_attn.weight"] = randw(HIDDEN_SIZE, 3 * HIDDEN_SIZE)  # [in, out] Conv1D
    tensors[p + "attn.c_attn.bias"] = randw(3 * HIDDEN_SIZE)
    tensors[p + "attn.c_proj.weight"] = randw(HIDDEN_SIZE, HIDDEN_SIZE)
    tensors[p + "attn.c_proj.bias"] = randw(HIDDEN_SIZE)
    tensors[p + "ln_2.weight"] = np.ones(HIDDEN_SIZE, dtype=np.float32) + randw(HIDDEN_SIZE) * 0.1
    tensors[p + "ln_2.bias"] = randw(HIDDEN_SIZE) * 0.1
    tensors[p + "mlp.c_fc.weight"] = randw(HIDDEN_SIZE, INTERMEDIATE_SIZE)
    tensors[p + "mlp.c_fc.bias"] = randw(INTERMEDIATE_SIZE)
    tensors[p + "mlp.c_proj.weight"] = randw(INTERMEDIATE_SIZE, HIDDEN_SIZE)
    tensors[p + "mlp.c_proj.bias"] = randw(HIDDEN_SIZE)
tensors["ln_f.weight"] = np.ones(HIDDEN_SIZE, dtype=np.float32) + randw(HIDDEN_SIZE) * 0.1
tensors["ln_f.bias"] = randw(HIDDEN_SIZE) * 0.1

save_file(tensors, "gpt_test_weights.safetensors")
print("wrote gpt_test_weights.safetensors")

# ---------------------------------------------------------------------------
# Independent numpy reference forward pass.
# ---------------------------------------------------------------------------
causal_mask = np.where(np.tril(np.ones((SEQ_LEN, SEQ_LEN))) > 0, 0.0, -1e9).astype(np.float32)
position_ids = np.arange(SEQ_LEN)

hidden = (tensors["wte.weight"][INPUT_IDS] + tensors["wpe.weight"][position_ids]).astype(np.float32)

for i in range(NUM_LAYERS):
    p = f"h.{i}."
    normed = layer_norm(hidden, tensors[p + "ln_1.weight"], tensors[p + "ln_1.bias"], LAYER_NORM_EPS)

    qkv = normed @ tensors[p + "attn.c_attn.weight"] + tensors[p + "attn.c_attn.bias"]  # [seq, 3*hidden]
    q = qkv[:, :HIDDEN_SIZE]
    k = qkv[:, HIDDEN_SIZE : 2 * HIDDEN_SIZE]
    v = qkv[:, 2 * HIDDEN_SIZE :]

    q = q.reshape(SEQ_LEN, NUM_ATTENTION_HEADS, HEAD_DIM).transpose(1, 0, 2)
    k = k.reshape(SEQ_LEN, NUM_ATTENTION_HEADS, HEAD_DIM).transpose(1, 0, 2)
    v = v.reshape(SEQ_LEN, NUM_ATTENTION_HEADS, HEAD_DIM).transpose(1, 0, 2)

    scores = (q @ k.transpose(0, 2, 1)) / np.sqrt(HEAD_DIM).astype(np.float32)
    scores = scores + causal_mask
    probs = softmax(scores, axis=-1)
    ctx = probs @ v  # [H, seq, hd]

    ctx = ctx.transpose(1, 0, 2).reshape(SEQ_LEN, HIDDEN_SIZE)
    attn_out = ctx @ tensors[p + "attn.c_proj.weight"] + tensors[p + "attn.c_proj.bias"]
    hidden = (hidden + attn_out).astype(np.float32)

    normed2 = layer_norm(hidden, tensors[p + "ln_2.weight"], tensors[p + "ln_2.bias"], LAYER_NORM_EPS)
    ff = normed2 @ tensors[p + "mlp.c_fc.weight"] + tensors[p + "mlp.c_fc.bias"]
    ff = gelu_exact(ff)
    ff_out = ff @ tensors[p + "mlp.c_proj.weight"] + tensors[p + "mlp.c_proj.bias"]
    hidden = (hidden + ff_out).astype(np.float32)

hidden = layer_norm(hidden, tensors["ln_f.weight"], tensors["ln_f.bias"], LAYER_NORM_EPS)
logits = hidden @ tensors["wte.weight"].T  # tied lm_head

print("expected logits, row-major [seqLen, vocabSize]:")
flat = ", ".join(f"{v:.6f}f" for v in logits.flatten())
print("{" + flat + "}")
