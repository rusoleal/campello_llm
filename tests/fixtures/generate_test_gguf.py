#!/usr/bin/env python3
"""Generates tests/fixtures/test_tensors.gguf — a small synthetic gguf file used by
test_gguf_reader.cpp. Regenerate with: python3 generate_test_gguf.py
(requires `pip install gguf numpy`)."""

import numpy as np
from gguf import GGUFWriter

OUT_PATH = "test_tensors.gguf"


def main():
    writer = GGUFWriter(OUT_PATH, arch="campello_llm_test")

    # A handful of metadata value shapes the reader needs to round-trip:
    # scalar string/uint32/float32/bool, and a string array (the same shape
    # tokenizer.ggml.tokens uses for a real embedded vocab).
    writer.add_string("general.name", "campello_llm test model")
    writer.add_uint32("campello_llm_test.block_count", 2)
    writer.add_float32("campello_llm_test.attention.scale", 0.5)
    writer.add_bool("campello_llm_test.flag", True)
    writer.add_array("campello_llm_test.tokens", ["<s>", "hello", "world"])

    weight = np.array([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]], dtype=np.float32)
    bias = np.array([0.5, -0.5], dtype=np.float16)
    ids = np.array([1, 2, 3, 4], dtype=np.int32)

    writer.add_tensor("weight", weight)
    writer.add_tensor("bias", bias)
    writer.add_tensor("ids", ids)

    writer.write_header_to_file()
    writer.write_kv_data_to_file()
    writer.write_tensors_to_file()
    writer.close()

    print(f"wrote {OUT_PATH}")


if __name__ == "__main__":
    main()
