#!/usr/bin/env python3
"""Generates tests/fixtures/llama_test_weights.safetensors -- a tiny, synthetic
LLaMA-shaped model (GQA, RoPE, SwiGLU, RMSNorm) with fixed-seed random Float32
weights -- and independently computes the expected forward-pass logits in plain
numpy, printed as a C++-literal array to paste into test_llama_architecture.cpp.

This script IS the "independently hand/numpy-computed reference" TODO.md's Phase 3
test plan calls for: it reimplements the forward pass from scratch in numpy,
deliberately not sharing any code with src/architecture/llama_architecture.cpp.

Regenerate with: python3 generate_llama_test_fixture.py (requires `pip install
safetensors numpy`). Only rerun if the tiny config below changes -- the printed
expected-logits array must be re-pasted into the C++ test if so.
"""

import numpy as np
from safetensors.numpy import save_file

# ---------------------------------------------------------------------------
# Tiny synthetic config -- small enough to eyeball, but exercises every real
# LLaMA-specific mechanism: GQA (numKeyValueHeads < numAttentionHeads), RoPE,
# SwiGLU, RMSNorm, and 2 stacked layers (residual accumulation).
# ---------------------------------------------------------------------------
VOCAB_SIZE = 6
HIDDEN_SIZE = 8
NUM_LAYERS = 2
NUM_ATTENTION_HEADS = 4
NUM_KEY_VALUE_HEADS = 2
HEAD_DIM = HIDDEN_SIZE // NUM_ATTENTION_HEADS
INTERMEDIATE_SIZE = 10
RMS_NORM_EPS = 1e-5
ROPE_THETA = 10000.0
SEQ_LEN = 3
INPUT_IDS = np.array([1, 4, 2], dtype=np.int64)

rng = np.random.default_rng(42)


def randw(*shape):
    return rng.normal(scale=0.2, size=shape).astype(np.float32)


def rms_norm(x, weight, eps):
    variance = np.mean(x.astype(np.float64) ** 2, axis=-1, keepdims=True)
    normed = x / np.sqrt(variance + eps).astype(np.float32)
    return (normed * weight).astype(np.float32)


def silu(x):
    return (x / (1.0 + np.exp(-x))).astype(np.float32)


def softmax(x, axis):
    x = x - np.max(x, axis=axis, keepdims=True)
    e = np.exp(x)
    return (e / np.sum(e, axis=axis, keepdims=True)).astype(np.float32)


def rotate_half(x):
    half = x.shape[-1] // 2
    return np.concatenate([-x[..., half:], x[..., :half]], axis=-1)


def build_rope(seq_len, head_dim, theta):
    half = head_dim // 2
    freqs = theta ** (-(2.0 * np.arange(half)) / head_dim)
    positions = np.arange(seq_len)
    angles = np.outer(positions, freqs)  # [seq_len, half]
    cos = np.concatenate([np.cos(angles), np.cos(angles)], axis=-1).astype(np.float32)
    sin = np.concatenate([np.sin(angles), np.sin(angles)], axis=-1).astype(np.float32)
    return cos, sin  # [seq_len, head_dim]


tensors = {}
tensors["model.embed_tokens.weight"] = randw(VOCAB_SIZE, HIDDEN_SIZE)
for i in range(NUM_LAYERS):
    p = f"model.layers.{i}."
    tensors[p + "input_layernorm.weight"] = np.ones(HIDDEN_SIZE, dtype=np.float32) + randw(HIDDEN_SIZE) * 0.1
    tensors[p + "self_attn.q_proj.weight"] = randw(HIDDEN_SIZE, HIDDEN_SIZE)
    tensors[p + "self_attn.k_proj.weight"] = randw(NUM_KEY_VALUE_HEADS * HEAD_DIM, HIDDEN_SIZE)
    tensors[p + "self_attn.v_proj.weight"] = randw(NUM_KEY_VALUE_HEADS * HEAD_DIM, HIDDEN_SIZE)
    tensors[p + "self_attn.o_proj.weight"] = randw(HIDDEN_SIZE, HIDDEN_SIZE)
    tensors[p + "post_attention_layernorm.weight"] = np.ones(HIDDEN_SIZE, dtype=np.float32) + randw(HIDDEN_SIZE) * 0.1
    tensors[p + "mlp.gate_proj.weight"] = randw(INTERMEDIATE_SIZE, HIDDEN_SIZE)
    tensors[p + "mlp.up_proj.weight"] = randw(INTERMEDIATE_SIZE, HIDDEN_SIZE)
    tensors[p + "mlp.down_proj.weight"] = randw(HIDDEN_SIZE, INTERMEDIATE_SIZE)
tensors["model.norm.weight"] = np.ones(HIDDEN_SIZE, dtype=np.float32) + randw(HIDDEN_SIZE) * 0.1
tensors["lm_head.weight"] = randw(VOCAB_SIZE, HIDDEN_SIZE)

save_file(tensors, "llama_test_weights.safetensors")
print("wrote llama_test_weights.safetensors")

# ---------------------------------------------------------------------------
# Independent numpy reference forward pass.
# ---------------------------------------------------------------------------
group_size = NUM_ATTENTION_HEADS // NUM_KEY_VALUE_HEADS
cos, sin = build_rope(SEQ_LEN, HEAD_DIM, ROPE_THETA)
causal_mask = np.where(np.tril(np.ones((SEQ_LEN, SEQ_LEN))) > 0, 0.0, -1e9).astype(np.float32)

hidden = tensors["model.embed_tokens.weight"][INPUT_IDS]  # [seq, hidden]

for i in range(NUM_LAYERS):
    p = f"model.layers.{i}."
    normed = rms_norm(hidden, tensors[p + "input_layernorm.weight"], RMS_NORM_EPS)

    q = normed @ tensors[p + "self_attn.q_proj.weight"].T  # [seq, hidden]
    k = normed @ tensors[p + "self_attn.k_proj.weight"].T  # [seq, kvDim]
    v = normed @ tensors[p + "self_attn.v_proj.weight"].T

    q = q.reshape(SEQ_LEN, NUM_ATTENTION_HEADS, HEAD_DIM).transpose(1, 0, 2)  # [H, seq, hd]
    k = k.reshape(SEQ_LEN, NUM_KEY_VALUE_HEADS, HEAD_DIM).transpose(1, 0, 2)  # [Hkv, seq, hd]
    v = v.reshape(SEQ_LEN, NUM_KEY_VALUE_HEADS, HEAD_DIM).transpose(1, 0, 2)

    q = q * cos + rotate_half(q) * sin
    k = k * cos + rotate_half(k) * sin

    k = np.repeat(k, group_size, axis=0)  # [H, seq, hd]
    v = np.repeat(v, group_size, axis=0)

    scores = (q @ k.transpose(0, 2, 1)) / np.sqrt(HEAD_DIM).astype(np.float32)  # [H, seq, seq]
    scores = scores + causal_mask
    probs = softmax(scores, axis=-1)
    ctx = probs @ v  # [H, seq, hd]

    ctx = ctx.transpose(1, 0, 2).reshape(SEQ_LEN, HIDDEN_SIZE)
    attn_out = ctx @ tensors[p + "self_attn.o_proj.weight"].T
    hidden = (hidden + attn_out).astype(np.float32)

    normed2 = rms_norm(hidden, tensors[p + "post_attention_layernorm.weight"], RMS_NORM_EPS)
    gate = normed2 @ tensors[p + "mlp.gate_proj.weight"].T
    up = normed2 @ tensors[p + "mlp.up_proj.weight"].T
    ffn_hidden = silu(gate) * up
    down = ffn_hidden @ tensors[p + "mlp.down_proj.weight"].T
    hidden = (hidden + down).astype(np.float32)

hidden = rms_norm(hidden, tensors["model.norm.weight"], RMS_NORM_EPS)
logits = hidden @ tensors["lm_head.weight"].T  # [seq, vocab]

print("expected logits, row-major [seqLen, vocabSize]:")
flat = ", ".join(f"{v:.6f}f" for v in logits.flatten())
print("{" + flat + "}")
