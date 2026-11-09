"""Read the same saved benchmark datasets used by the C++ executable."""

from dataclasses import dataclass
import json
from pathlib import Path

import numpy as np
import torch


@dataclass(frozen=True)
class BenchmarkDataset:
    path: Path
    case_id: str
    M: int
    K: int
    N: int
    seed: int
    target_sparsity: float
    atol: float
    rtol: float
    X: torch.Tensor
    W: torch.Tensor


def _read(path: Path, dtype: str, shape: tuple[int, ...]) -> torch.Tensor:
    expected = int(np.prod(shape))
    values = np.fromfile(path, dtype=dtype)
    if values.size != expected:
        raise ValueError(
            f"{path} contains {values.size} elements; expected {expected}"
        )
    return torch.from_numpy(values.reshape(shape))


def load_dataset(directory: str | Path) -> BenchmarkDataset:
    directory = Path(directory)
    metadata_path = directory / "metadata.json"
    try:
        metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    except FileNotFoundError as error:
        raise ValueError(f"Cannot open {metadata_path}") from error

    if metadata.get("layout") != "row_major":
        raise ValueError("Only row-major datasets are supported")
    if metadata.get("byte_order") != "little":
        raise ValueError("Only little-endian datasets are supported")

    config = metadata["config"]
    reference = metadata["reference"]
    M, K, N = (int(config[name]) for name in ("M", "K", "N"))
    if min(M, K, N) <= 0:
        raise ValueError("Dataset dimensions must be positive")

    
    # Read aux to share numpy arrays's storage with torch tensors
    X = _read(directory / "X.bin", "<f4", (M, K))               # Little-endian float32
    W = _read(directory / "W.bin", "i1", (K, N))                # Unsigned 8-bit integer (int8)
    if not torch.isfinite(X).all():
        raise ValueError("X contains non-finite values")
    if not torch.all((W == -1) | (W == 0) | (W == 1)):
        raise ValueError("W contains values outside {-1, 0, 1}")

    return BenchmarkDataset(
        path=directory,
        case_id=str(metadata.get("case_id", directory.name)),
        M=M,
        K=K,
        N=N,
        seed=int(config["seed"]),
        target_sparsity=float(config["target_zero_fraction"]),
        atol=float(reference["atol"]),
        rtol=float(reference["rtol"]),
        X=X,
        W=W,
    )
