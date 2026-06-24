#!/usr/bin/env python3
"""Generates tests/fixtures/tiny_llama_model/ -- a complete, minimal HuggingFace
LLaMA checkpoint directory (config.json, tokenizer.json, tokenizer_config.json,
model.safetensors) for test_model.cpp's Model::load()/generate() integration
tests. Unlike generate_llama_test_fixture.py (which only exercises the graph
wiring against a numpy reference), this exercises the *whole* Model pipeline:
config.json parsing, tokenizer.json parsing, and end-to-end generate().

The tokenizer is intentionally simple (no normalizer/byte_fallback/decoder --
covered already by the real-TinyLlama-tokenizer tests in test_tokenizer.cpp) --
just enough vocabulary to round-trip a couple of short test prompts.

Regenerate with: python3 generate_tiny_llama_model_fixture.py (requires
`pip install tokenizers safetensors numpy`).
"""

import json
import os

import numpy as np
from safetensors.numpy import save_file
from tokenizers import AddedToken, Tokenizer, models, processors

OUT_DIR = "tiny_llama_model"
os.makedirs(OUT_DIR, exist_ok=True)

VOCAB_SIZE = 9
HIDDEN_SIZE = 8
NUM_LAYERS = 1
NUM_ATTENTION_HEADS = 4
NUM_KEY_VALUE_HEADS = 2
INTERMEDIATE_SIZE = 10
RMS_NORM_EPS = 1e-5
ROPE_THETA = 10000.0

# ---------------------------------------------------------------------------
# Tokenizer: vocab/merges chosen only to round-trip "hi"/"hi!" -- no
# normalizer/byte_fallback/decoder (those are exercised against the real
# TinyLlama tokenizer in test_tokenizer.cpp already).
# ---------------------------------------------------------------------------
vocab = {"<unk>": 0, "<s>": 1, "</s>": 2, "h": 3, "i": 4, "hi": 5, "a": 6, "y": 7, "!": 8}
merges = [("h", "i")]
assert len(vocab) == VOCAB_SIZE

bpe = models.BPE(vocab=vocab, merges=merges, unk_token="<unk>", byte_fallback=False, fuse_unk=False)
tok = Tokenizer(bpe)
tok.add_special_tokens([
    AddedToken("<unk>", special=True),
    AddedToken("<s>", special=True),
    AddedToken("</s>", special=True),
])
tok.post_processor = processors.TemplateProcessing(single="<s> $A", special_tokens=[("<s>", 1)])
tok.save(os.path.join(OUT_DIR, "tokenizer.json"))

with open(os.path.join(OUT_DIR, "tokenizer_config.json"), "w") as f:
    json.dump({"bos_token": "<s>", "eos_token": "</s>", "unk_token": "<unk>", "pad_token": "</s>"}, f)

# ---------------------------------------------------------------------------
# config.json
# ---------------------------------------------------------------------------
with open(os.path.join(OUT_DIR, "config.json"), "w") as f:
    json.dump(
        {
            "model_type": "llama",
            "vocab_size": VOCAB_SIZE,
            "hidden_size": HIDDEN_SIZE,
            "num_hidden_layers": NUM_LAYERS,
            "num_attention_heads": NUM_ATTENTION_HEADS,
            "num_key_value_heads": NUM_KEY_VALUE_HEADS,
            "intermediate_size": INTERMEDIATE_SIZE,
            "rms_norm_eps": RMS_NORM_EPS,
            "rope_theta": ROPE_THETA,
        },
        f,
    )

# ---------------------------------------------------------------------------
# model.safetensors -- same fixed-seed random-weight approach as
# generate_llama_test_fixture.py, just resized to this tokenizer's VOCAB_SIZE.
# ---------------------------------------------------------------------------
rng = np.random.default_rng(123)


def randw(*shape):
    return rng.normal(scale=0.2, size=shape).astype(np.float32)


tensors = {}
tensors["model.embed_tokens.weight"] = randw(VOCAB_SIZE, HIDDEN_SIZE)
for i in range(NUM_LAYERS):
    p = f"model.layers.{i}."
    tensors[p + "input_layernorm.weight"] = np.ones(HIDDEN_SIZE, dtype=np.float32)
    tensors[p + "self_attn.q_proj.weight"] = randw(HIDDEN_SIZE, HIDDEN_SIZE)
    tensors[p + "self_attn.k_proj.weight"] = randw(NUM_KEY_VALUE_HEADS * (HIDDEN_SIZE // NUM_ATTENTION_HEADS), HIDDEN_SIZE)
    tensors[p + "self_attn.v_proj.weight"] = randw(NUM_KEY_VALUE_HEADS * (HIDDEN_SIZE // NUM_ATTENTION_HEADS), HIDDEN_SIZE)
    tensors[p + "self_attn.o_proj.weight"] = randw(HIDDEN_SIZE, HIDDEN_SIZE)
    tensors[p + "post_attention_layernorm.weight"] = np.ones(HIDDEN_SIZE, dtype=np.float32)
    tensors[p + "mlp.gate_proj.weight"] = randw(INTERMEDIATE_SIZE, HIDDEN_SIZE)
    tensors[p + "mlp.up_proj.weight"] = randw(INTERMEDIATE_SIZE, HIDDEN_SIZE)
    tensors[p + "mlp.down_proj.weight"] = randw(HIDDEN_SIZE, INTERMEDIATE_SIZE)
tensors["model.norm.weight"] = np.ones(HIDDEN_SIZE, dtype=np.float32)
tensors["lm_head.weight"] = randw(VOCAB_SIZE, HIDDEN_SIZE)

save_file(tensors, os.path.join(OUT_DIR, "model.safetensors"))
print(f"wrote {OUT_DIR}/{{tokenizer.json,tokenizer_config.json,config.json,model.safetensors}}")
