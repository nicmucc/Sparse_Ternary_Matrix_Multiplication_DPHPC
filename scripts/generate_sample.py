"""
Temporary file that generates one test sample in data/example_001
Testing parameters are temporarily hardcoded

The scripts generates: 
  - metadata.json file with sample metadata
  - X.bin, W.bin with the matrices weights
  - bias.bin with the bias vector data
  - alpha.bin for the PReLU
  - expected.json with the expected solution (this should be turned into .bin as well)

Usage: python3 scripts/generate_sample.py
"""

import json
import random
import struct
from pathlib import Path


def main():
    out = Path("data/sample_001")
    seed = 42
    rng = random.Random(seed)
    M, K, N = 2, 3, 2

    X = [float(rng.randint(-3, 3)) for _ in range(M * K)]
    W = rng.choices((-1, 0, 1), weights=(1, 2, 1), k=K * N)
    bias = [0.5, -0.5]
    alpha = [0.25]

    linear = [
        [
            sum(X[i * K + k] * W[k * N + j] for k in range(K))
            + bias[j]
            for j in range(N)
        ]
        for i in range(M)
    ]
    prelu = [
        [v if v >= 0 else alpha[0] * v for v in row]
        for row in linear
    ]

    metadata = {
        "case_id": out.name,
        "format_version": 1,
        "generator_version": 1,
        "seed": seed,
        "M": M,
        "K": K,
        "N": N,
        "layout": "row_major",
        "byte_order": "little",
        "dtypes": {
            "X": "float32",
            "W": "int8",
            "bias": "float32",
            "alpha": "float32",
        },
        "sparsity_pattern": "independent",
        "target_zero_fraction": 0.5,
        "actual_zero_fraction": W.count(0) / len(W),
        "activation": "prelu",
        "alpha_mode": "shared",
    }

    out.mkdir(parents=True)

    for name, fmt, values in [
        ("X", "f", X),
        ("W", "b", W),
        ("bias", "f", bias),
        ("alpha", "f", alpha),
    ]:
        (out / f"{name}.bin").write_bytes(
            struct.pack(f"<{len(values)}{fmt}", *values)
        )

    for name, contents in [
        ("metadata", metadata),
        ("expected", {"linear": linear, "prelu": prelu}),
    ]:
        (out / f"{name}.json").write_text(
            json.dumps(contents, indent=2) + "\n",
            encoding="utf-8",
        )

    print(f"Generated {out.resolve()}")


if __name__ == "__main__":
    main()
