#!/usr/bin/env python3
"""Generates tests/fixtures/test_tensors.safetensors — a small synthetic safetensors
file used by test_safetensors_reader.cpp. Regenerate with:
python3 generate_test_safetensors.py (requires `pip install safetensors numpy`)."""

import numpy as np
from safetensors.numpy import save_file

OUT_PATH = "test_tensors.safetensors"


def main():
    tensors = {
        "weight": np.array([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]], dtype=np.float32),
        "bias": np.array([0.5, -0.5], dtype=np.float16),
        "ids": np.array([1, 2, 3, 4], dtype=np.int32),
        "flags": np.array([1, 0, 1], dtype=np.uint8),
    }
    save_file(tensors, OUT_PATH, metadata={"format": "campello_llm_test"})
    print(f"wrote {OUT_PATH}")


if __name__ == "__main__":
    main()
