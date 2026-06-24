#!/usr/bin/env python3
"""Generates tests/fixtures/llama_test_weights_bf16.safetensors -- the exact same
tiny synthetic LLaMA config as generate_llama_test_fixture.py, but with every
weight actually stored as BF16 (real LLaMA/TinyLlama checkpoints ship BF16, per
TinyLlama's own config.json `"torch_dtype": "bfloat16"`) -- to test
constantWeight()/constantLinearWeightTransposed()'s BF16->F32 decode path
end-to-end, not just unit-test the bit manipulation in isolation.

Every weight is rounded to BF16 precision *before* computing the numpy reference
logits too, so the comparison is apples-to-apples (BF16-rounded weights in, BF16-
rounded weights out) rather than comparing against unrounded F32 logits.

Regenerate with: python3 generate_llama_bf16_test_fixture.py (requires `pip
install safetensors numpy`).
"""

import json
import struct

import numpy as np
from safetensors import safe_open

SRC = "llama_test_weights.safetensors"
OUT = "llama_test_weights_bf16.safetensors"

NUM_LAYERS = 2
SEQ_LEN = 3
INPUT_IDS = np.array([1, 4, 2], dtype=np.int64)
RMS_NORM_EPS = 1e-5
ROPE_THETA = 10000.0
HIDDEN_SIZE = 8
NUM_ATTENTION_HEADS = 4
NUM_KEY_VALUE_HEADS = 2
HEAD_DIM = HIDDEN_SIZE // NUM_ATTENTION_HEADS


def round_to_bf16(x: np.ndarray) -> np.ndarray:
    """Truncates every float32 element to BF16 precision (zero the low 16 mantissa
    bits), staying in a float32 array -- exactly what a real BF16 round-trip loses."""
    bits = x.astype(np.float32).view(np.uint32)
    bits = bits & np.uint32(0xFFFF0000)
    return bits.view(np.float32)


def to_bf16_bytes(x: np.ndarray) -> bytes:
    bits = x.astype(np.float32).view(np.uint32)
    bf16 = (bits >> 16).astype(np.uint16)
    return bf16.tobytes()


tensors = {}
rounded = {}
with safe_open(SRC, framework="numpy") as f:
    for key in f.keys():
        arr = f.get_tensor(key)
        rounded[key] = round_to_bf16(arr)
        tensors[key] = arr.shape

# Neither numpy (no native bfloat16) nor the installed `safetensors` Python
# binding's serialize_file() (raises "dtype BF16 is not covered") can write BF16
# tensors -- write the file by hand instead, using the exact binary format
# Phase 1's safetensors_reader.cpp already parses (verified against the real
# package's own header output, see CLAUDE.md): 8-byte LE header length, then the
# JSON header, then raw tensor bytes back-to-back in header order.
header = {}
body = bytearray()
for key, shape in tensors.items():
    data = to_bf16_bytes(rounded[key])
    start = len(body)
    body += data
    header[key] = {"dtype": "BF16", "shape": list(shape), "data_offsets": [start, start + len(data)]}

header_bytes = json.dumps(header).encode("utf-8")
with open(OUT, "wb") as f:
    f.write(struct.pack("<Q", len(header_bytes)))
    f.write(header_bytes)
    f.write(body)
print(f"wrote {OUT}")


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
    angles = np.outer(positions, freqs)
    cos = np.concatenate([np.cos(angles), np.cos(angles)], axis=-1).astype(np.float32)
    sin = np.concatenate([np.sin(angles), np.sin(angles)], axis=-1).astype(np.float32)
    return cos, sin


group_size = NUM_ATTENTION_HEADS // NUM_KEY_VALUE_HEADS
cos, sin = build_rope(SEQ_LEN, HEAD_DIM, ROPE_THETA)
causal_mask = np.where(np.tril(np.ones((SEQ_LEN, SEQ_LEN))) > 0, 0.0, -1e9).astype(np.float32)

hidden = rounded["model.embed_tokens.weight"][INPUT_IDS]

for i in range(NUM_LAYERS):
    p = f"model.layers.{i}."
    normed = rms_norm(hidden, rounded[p + "input_layernorm.weight"], RMS_NORM_EPS)

    q = normed @ rounded[p + "self_attn.q_proj.weight"].T
    k = normed @ rounded[p + "self_attn.k_proj.weight"].T
    v = normed @ rounded[p + "self_attn.v_proj.weight"].T

    q = q.reshape(SEQ_LEN, NUM_ATTENTION_HEADS, HEAD_DIM).transpose(1, 0, 2)
    k = k.reshape(SEQ_LEN, NUM_KEY_VALUE_HEADS, HEAD_DIM).transpose(1, 0, 2)
    v = v.reshape(SEQ_LEN, NUM_KEY_VALUE_HEADS, HEAD_DIM).transpose(1, 0, 2)

    q = q * cos + rotate_half(q) * sin
    k = k * cos + rotate_half(k) * sin

    k = np.repeat(k, group_size, axis=0)
    v = np.repeat(v, group_size, axis=0)

    scores = (q @ k.transpose(0, 2, 1)) / np.sqrt(HEAD_DIM).astype(np.float32)
    scores = scores + causal_mask
    probs = softmax(scores, axis=-1)
    ctx = probs @ v

    ctx = ctx.transpose(1, 0, 2).reshape(SEQ_LEN, HIDDEN_SIZE)
    attn_out = ctx @ rounded[p + "self_attn.o_proj.weight"].T
    hidden = (hidden + attn_out).astype(np.float32)

    normed2 = rms_norm(hidden, rounded[p + "post_attention_layernorm.weight"], RMS_NORM_EPS)
    gate = normed2 @ rounded[p + "mlp.gate_proj.weight"].T
    up = normed2 @ rounded[p + "mlp.up_proj.weight"].T
    ffn_hidden = silu(gate) * up
    down = ffn_hidden @ rounded[p + "mlp.down_proj.weight"].T
    hidden = (hidden + down).astype(np.float32)

hidden = rms_norm(hidden, rounded["model.norm.weight"], RMS_NORM_EPS)
logits = hidden @ rounded["lm_head.weight"].T

print("expected logits (bf16-rounded weights), row-major [seqLen, vocabSize]:")
flat = ", ".join(f"{v:.6f}f" for v in logits.flatten())
print("{" + flat + "}")
