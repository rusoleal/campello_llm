#!/usr/bin/env python3
"""Generates tests/fixtures/tiny_gguf_model.gguf — a minimal loadable LLaMA
model used by test_model.cpp to exercise Model::load() on a .gguf file.
Regenerate with: python3 generate_tiny_gguf_model.py (requires `pip install gguf numpy`)."""

import numpy as np
from gguf import GGUFWriter

OUT_PATH = "tiny_gguf_model.gguf"

VOCAB = ["<unk>", "<s>", "</s>", "h", "i", "▁", "<0x41>"]
MERGES = []


def add_f32_tensor(writer, name, shape):
    """Adds a deterministic F32 tensor with the given (HF-style) shape."""
    weight = np.zeros(shape, dtype=np.float32)
    # Deterministic non-zero values so matmuls don't collapse to zero.
    it = np.nditer(weight, flags=["multi_index"], op_flags=["writeonly"])
    while not it.finished:
        idx = it.multi_index
        # A simple hash of the indices into a small float.
        value = float(hash((name, idx)) % 1000) / 1000.0 - 0.5
        it[0] = value
        it.iternext()
    writer.add_tensor(name, weight)


def main():
    writer = GGUFWriter(OUT_PATH, arch="llama")

    # Hyperparameters (same tiny shape as tests/fixtures/tiny_llama_model).
    writer.add_uint32("llama.block_count", 1)
    writer.add_uint32("llama.embedding_length", 8)
    writer.add_uint32("llama.feed_forward_length", 10)
    writer.add_uint32("llama.attention.head_count", 4)
    writer.add_uint32("llama.attention.head_count_kv", 2)
    writer.add_float32("llama.attention.layer_norm_rms_epsilon", 1e-5)
    writer.add_float32("llama.rope.freq_base", 10000.0)
    writer.add_uint32("llama.vocab_size", len(VOCAB))
    writer.add_uint32("llama.context_length", 128)

    # Embedded SentencePiece-style tokenizer.
    writer.add_string("tokenizer.ggml.model", "llama")
    writer.add_array("tokenizer.ggml.tokens", VOCAB)
    writer.add_array("tokenizer.ggml.merges", MERGES)
    writer.add_uint32("tokenizer.ggml.bos_token_id", 1)
    writer.add_uint32("tokenizer.ggml.eos_token_id", 2)
    writer.add_uint32("tokenizer.ggml.unknown_token_id", 0)
    writer.add_array("tokenizer.ggml.token_type", [2, 3, 3, 1, 1, 1, 1])

    # Weight tensors using llama.cpp naming. Provide HF-style shapes here:
    # GGUF itself stores dims reversed, and the adapter reverses them back.
    add_f32_tensor(writer, "token_embd.weight", (len(VOCAB), 8))  # HF [vocab, hidden]
    add_f32_tensor(writer, "output_norm.weight", (8,))
    add_f32_tensor(writer, "output.weight", (len(VOCAB), 8))

    add_f32_tensor(writer, "blk.0.attn_norm.weight", (8,))
    add_f32_tensor(writer, "blk.0.attn_q.weight", (8, 8))
    add_f32_tensor(writer, "blk.0.attn_k.weight", (4, 8))  # kv_dim = num_kv_heads * head_dim = 2 * 4
    add_f32_tensor(writer, "blk.0.attn_v.weight", (4, 8))
    add_f32_tensor(writer, "blk.0.attn_output.weight", (8, 8))
    add_f32_tensor(writer, "blk.0.ffn_norm.weight", (8,))
    add_f32_tensor(writer, "blk.0.ffn_gate.weight", (10, 8))
    add_f32_tensor(writer, "blk.0.ffn_up.weight", (10, 8))
    add_f32_tensor(writer, "blk.0.ffn_down.weight", (8, 10))

    writer.write_header_to_file()
    writer.write_kv_data_to_file()
    writer.write_tensors_to_file()
    writer.close()

    print(f"wrote {OUT_PATH}")


if __name__ == "__main__":
    main()
