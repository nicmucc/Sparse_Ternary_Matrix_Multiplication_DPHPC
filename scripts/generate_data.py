"""
Generate the original small test sample in data/sample_001.

Usage:
    python3 outputs/generate_sample.py
    python3 outputs/generate_sample.py --config config.json
    python3 outputs/generate_sample.py --out data/sample_001 --overwrite

Config files override selected DEFAULT_CONFIG fields.
Expected outputs are little-endian float32, stored in row-major order.
"""

import argparse
import json
import math
import platform
import random
import shutil
import tempfile
from pathlib import Path

import numpy as np


DEFAULT_CONFIG = {
    "M": 2,
    "K": 3,
    "N": 2,
    "seed": 42,
    "x_min": -3,
    "x_max": 3,
    "bias": [0.5, -0.5],
    "alpha": [0.25],
    "target_zero_fraction": 0.5,
    "zero_count_mode": "independent",
    "chunk_rows": 1024,
    "rtol": 1e-5,
    "atol": 1e-6,
}


def load_config(path, seed):
    config = DEFAULT_CONFIG.copy()
    if path is not None:
        overrides = json.loads(path.read_text(encoding="utf-8"))
        if not isinstance(overrides, dict):
            raise ValueError("Config must be a JSON object")
        unknown = overrides.keys() - config.keys()
        if unknown:
            raise ValueError(f"Unknown config fields: {sorted(unknown)}")
        config.update(overrides)
    if seed is not None:
        config["seed"] = seed
    validate_config(config)
    return config


def validate_config(config):
    for key in ("M", "K", "N", "chunk_rows"):
        if type(config[key]) is not int or config[key] <= 0:
            raise ValueError(f"{key} must be a positive integer")

    for key in ("seed", "x_min", "x_max"):
        if type(config[key]) is not int:
            raise ValueError(f"{key} must be an integer")
    if config["x_min"] > config["x_max"]:
        raise ValueError("x_min must not exceed x_max")

    for key in ("target_zero_fraction", "rtol", "atol"):
        value = config[key]
        if type(value) not in (int, float) or not math.isfinite(value):
            raise ValueError(f"{key} must be a finite number")
    if not 0 <= config["target_zero_fraction"] <= 1:
        raise ValueError("target_zero_fraction must be between 0 and 1")
    if config["rtol"] < 0 or config["atol"] < 0:
        raise ValueError("Tolerances must be nonnegative")
    if config["zero_count_mode"] not in ("independent", "exact"):
        raise ValueError("zero_count_mode must be independent or exact")

    for key, length in (("bias", config["N"]), ("alpha", 1)):
        values = config[key]
        if not isinstance(values, list) or len(values) != length:
            raise ValueError(f"{key} must contain {length} values")
        if any(
            type(value) not in (int, float)
            or not math.isfinite(value)
            or abs(value) > np.finfo(np.float32).max
            for value in values
        ):
            raise ValueError(f"{key} must contain finite float32 values")

    if max(abs(config["x_min"]), abs(config["x_max"])) > np.finfo(
        np.float32
    ).max:
        raise ValueError("X bounds must be representable as finite float32")


def generate_inputs(config):
    rng = random.Random(config["seed"])
    M, K, N = (config[key] for key in ("M", "K", "N"))

    X = np.fromiter(
        (
            rng.randint(config["x_min"], config["x_max"])
            for _ in range(M * K)
        ),
        dtype="<f4",
        count=M * K,
    ).reshape(M, K)

    count = K * N
    fraction = config["target_zero_fraction"]
    if config["zero_count_mode"] == "independent":
        negative_cutoff = (1 - fraction) / 2
        zero_cutoff = negative_cutoff + fraction

        def draw_weight():
            value = rng.random()
            return -1 if value < negative_cutoff else (
                0 if value < zero_cutoff else 1
            )

        W = np.fromiter(
            (draw_weight() for _ in range(count)),
            dtype=np.int8,
            count=count,
        )
    else:
        zero_count = int(math.floor(count * fraction + 0.5))
        W = np.empty(count, dtype=np.int8)
        W[:zero_count] = 0
        W[zero_count:] = np.fromiter(
            (rng.choice((-1, 1)) for _ in range(count - zero_count)),
            dtype=np.int8,
            count=count - zero_count,
        )
        rng.shuffle(W)

    return {
        "X": X,
        "W": W.reshape(K, N),
        "bias": np.asarray(config["bias"], dtype="<f4"),
        "alpha": np.asarray(config["alpha"], dtype="<f4"),
    }


def compute_expected(X, W, bias, alpha):
    # Use serialized input precision with float64 reference arithmetic.
    linear = np.einsum("ik,kj->ij", X, W, dtype=np.float64)
    linear += bias.astype(np.float64)
    prelu = np.where(linear >= 0, linear, float(alpha[0]) * linear)
    if not np.isfinite(linear).all() or not np.isfinite(prelu).all():
        raise ValueError("Reference computation produced nonfinite values")
    limit = np.finfo(np.float32).max
    if np.any(np.abs(linear) > limit) or np.any(np.abs(prelu) > limit):
        raise ValueError("Expected outputs exceed finite float32 range")
    return linear.astype("<f4"), prelu.astype("<f4")


def write_json(path, contents):
    path.write_text(
        json.dumps(contents, indent=2, allow_nan=False) + "\n",
        encoding="utf-8",
    )


def write_sample(directory, case_id, config, inputs, debug_json):
    chunk_rows = config["chunk_rows"]
    files = {}

    for name, values in inputs.items():
        flat = values.reshape(-1)
        chunk_elements = chunk_rows * (
            values.shape[1] if values.ndim == 2 else 1
        )
        with (directory / f"{name}.bin").open("wb") as stream:
            for start in range(0, flat.size, chunk_elements):
                stream.write(flat[start:start + chunk_elements].tobytes())
        files[name] = {
            "path": f"{name}.bin",
            "shape": list(values.shape),
            "dtype": "int8" if name == "W" else "float32",
            "bytes": values.nbytes,
        }

    debug = {"linear": [], "prelu": []} if debug_json else None
    with (
        (directory / "expected_linear.bin").open("wb") as linear_file,
        (directory / "expected_prelu.bin").open("wb") as prelu_file,
    ):
        for start in range(0, config["M"], chunk_rows):
            linear, prelu = compute_expected(
                inputs["X"][start:start + chunk_rows],
                inputs["W"],
                inputs["bias"],
                inputs["alpha"],
            )
            linear_file.write(linear.tobytes())
            prelu_file.write(prelu.tobytes())
            if debug is not None:
                debug["linear"].extend(linear.tolist())
                debug["prelu"].extend(prelu.tolist())

    for name in ("expected_linear", "expected_prelu"):
        files[name] = {
            "path": f"{name}.bin",
            "shape": [config["M"], config["N"]],
            "dtype": "float32",
            "bytes": config["M"] * config["N"] * 4,
        }

    metadata = {
        "case_id": case_id,
        "format_version": 2,
        "generator_version": 2,
        "config": config,
        "layout": "row_major",
        "byte_order": "little",
        "files": files,
        "weight_values": [-1, 0, 1],
        "sparsity_pattern": "independent"
        if config["zero_count_mode"] == "independent"
        else "uniform_fixed_zero_count",
        "actual_zero_fraction": 1 - (
            np.count_nonzero(inputs["W"]) / inputs["W"].size
        ),
        "activation": "prelu",
        "alpha_mode": "shared",
        "reference": {
            "input_precision": "serialized_dtypes",
            "accumulation_dtype": "float64",
            "bias_and_activation_dtype": "float64",
            "output_rounding": "float32_after_each_reference_output",
            "comparison": "abs(actual - expected) <= atol + rtol * abs(expected)",
            "rtol": config["rtol"],
            "atol": config["atol"],
        },
        "environment": {
            "python": platform.python_version(),
            "numpy": np.__version__,
            "rng": "python.random.Random",
        },
    }
    if debug is not None:
        write_json(directory / "expected.json", debug)
        metadata["debug_expected"] = "expected.json"
    write_json(directory / "metadata.json", metadata)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path)
    parser.add_argument("--out", type=Path, default=Path("data/sample_001"))
    parser.add_argument("--seed", type=int)
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--debug-json", action="store_true")
    args = parser.parse_args()

    config = load_config(args.config, args.seed)
    out = args.out.absolute()
    if out.is_symlink() or (out.exists() and not out.is_dir()):
        raise ValueError(f"Output must be a regular directory: {out}")
    if out.exists() and not args.overwrite:
        raise FileExistsError(f"{out} already exists; use --overwrite")

    inputs = generate_inputs(config)
    out.parent.mkdir(parents=True, exist_ok=True)
    temporary = Path(tempfile.mkdtemp(prefix=f".{out.name}.", dir=out.parent))
    try:
        write_sample(temporary, out.name, config, inputs, args.debug_json)
        if out.exists():
            if not args.overwrite:
                raise FileExistsError(f"{out} already exists")
            shutil.rmtree(out)
        temporary.rename(out)
    finally:
        if temporary.exists():
            shutil.rmtree(temporary)

    print(f"Generated {out}")


if __name__ == "__main__":
    main()
