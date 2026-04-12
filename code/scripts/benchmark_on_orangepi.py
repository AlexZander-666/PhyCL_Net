import argparse
import json
import os
import platform
import sys
import time
from pathlib import Path
from typing import Dict, Optional, Sequence

import numpy as np
import torch


def _read_first_existing(paths: Sequence[str]) -> Optional[str]:
    for raw_path in paths:
        path = Path(raw_path)
        if path.exists():
            try:
                return path.read_text(encoding="utf-8", errors="ignore").strip("\x00\r\n ")
            except OSError:
                continue
    return None


def detect_board_model() -> str:
    return _read_first_existing(
        (
            "/proc/device-tree/model",
            "/sys/firmware/devicetree/base/model",
        )
    ) or platform.node() or "unknown-board"


def detect_peak_rss_mb() -> Optional[float]:
    try:
        import resource

        usage = resource.getrusage(resource.RUSAGE_SELF)
        rss_value = float(usage.ru_maxrss)
        if sys.platform == "darwin":
            return rss_value / (1024.0 * 1024.0)
        return rss_value / 1024.0
    except Exception:
        return None


def _load_input_samples(npz_path: Optional[Path], input_shape: Sequence[int]) -> Dict[str, object]:
    batch, channels, length = [int(v) for v in input_shape]
    if npz_path is None:
        tensor = np.random.randn(batch, channels, length).astype(np.float32)
        return {
            "source": "fixed",
            "samples": tensor,
            "batch_size": batch,
            "sample_count": batch,
        }

    data = np.load(npz_path, allow_pickle=True)
    if "x" not in data:
        raise KeyError(f"{npz_path} does not contain an 'x' array.")

    samples = np.asarray(data["x"], dtype=np.float32)
    if samples.ndim == 2:
        samples = np.expand_dims(samples, axis=0)
    if samples.ndim != 3:
        raise ValueError(f"Expected x with shape [N, C, L], got {samples.shape}")
    if int(samples.shape[1]) != channels or int(samples.shape[2]) != length:
        raise ValueError(f"Input shape mismatch: expected CxL={channels}x{length}, got {samples.shape[1]}x{samples.shape[2]}")

    return {
        "source": "npz",
        "samples": samples,
        "batch_size": 1,
        "sample_count": int(samples.shape[0]),
    }


def _prepare_tensor(sample_bank: np.ndarray, index: int, device: torch.device) -> torch.Tensor:
    if sample_bank.ndim == 3:
        sample = sample_bank[index % sample_bank.shape[0]]
        sample = np.expand_dims(sample, axis=0)
    else:
        sample = sample_bank
    return torch.from_numpy(np.ascontiguousarray(sample)).to(device)


def run_benchmark(
    model_path: Path,
    out_json: Path,
    input_shape: Sequence[int] = (1, 3, 512),
    warmup: int = 50,
    repeats: int = 200,
    runtime_backend: str = "torchscript",
    execution_mode: str = "CPU",
    board_model: Optional[str] = None,
    npz_path: Optional[Path] = None,
    device: str = "cpu",
) -> Dict:
    model_path = Path(model_path)
    out_json = Path(out_json)
    out_json.parent.mkdir(parents=True, exist_ok=True)

    target_device = torch.device(str(device).lower())
    model = torch.jit.load(str(model_path), map_location=target_device).eval()

    input_bank = _load_input_samples(Path(npz_path) if npz_path is not None else None, input_shape)
    samples = input_bank["samples"]

    latencies_ms = []
    with torch.no_grad():
        for idx in range(int(warmup)):
            batch = _prepare_tensor(samples, idx, target_device)
            _ = model(batch)
            if target_device.type == "cuda":
                torch.cuda.synchronize()

        for idx in range(int(repeats)):
            batch = _prepare_tensor(samples, idx, target_device)
            start = time.perf_counter()
            _ = model(batch)
            if target_device.type == "cuda":
                torch.cuda.synchronize()
            end = time.perf_counter()
            latencies_ms.append((end - start) * 1000.0)

    latency_array = np.asarray(latencies_ms, dtype=np.float64)
    summary = {
        "board": {
            "model": board_model or detect_board_model(),
            "hostname": platform.node(),
        },
        "environment": {
            "os": platform.platform(),
            "python": platform.python_version(),
            "torch": torch.__version__,
            "cpu_count": os.cpu_count(),
            "device": str(target_device),
        },
        "runtime": {
            "backend": runtime_backend,
            "device": str(target_device),
        },
        "model_path": str(model_path.resolve()),
        "model_file_size_bytes": model_path.stat().st_size,
        "input_shape": [int(v) for v in input_shape],
        "input_source": input_bank["source"],
        "sample_count": int(input_bank["sample_count"]),
        "warmup_count": int(warmup),
        "repeat_count": int(repeats),
        "batch_size": int(input_bank["batch_size"]),
        "execution_mode": execution_mode,
        "latency_ms": {
            "p50": float(np.percentile(latency_array, 50)),
            "p95": float(np.percentile(latency_array, 95)),
            "mean": float(np.mean(latency_array)),
            "min": float(np.min(latency_array)),
            "max": float(np.max(latency_array)),
        },
        "memory": {
            "peak_rss_mb": detect_peak_rss_mb(),
        },
    }

    with open(out_json, "w", encoding="utf-8") as handle:
        json.dump(summary, handle, indent=2)
    return summary


def make_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run a TorchScript latency benchmark on Orange Pi or another edge board.")
    parser.add_argument("--model-path", required=True, help="Path to the exported TorchScript model")
    parser.add_argument("--out-json", required=True, help="Where to write the benchmark summary JSON")
    parser.add_argument("--input-shape", type=int, nargs=3, default=[1, 3, 512], metavar=("B", "C", "L"))
    parser.add_argument("--warmup", type=int, default=50)
    parser.add_argument("--repeats", type=int, default=200)
    parser.add_argument("--runtime-backend", default="torchscript")
    parser.add_argument("--execution-mode", default="CPU")
    parser.add_argument("--board-model", default=None)
    parser.add_argument("--npz-path", default=None, help="Optional NPZ file with prepared windows under key x")
    parser.add_argument("--device", default="cpu")
    return parser


def main() -> int:
    parser = make_parser()
    args = parser.parse_args()
    summary = run_benchmark(
        model_path=Path(args.model_path),
        out_json=Path(args.out_json),
        input_shape=args.input_shape,
        warmup=args.warmup,
        repeats=args.repeats,
        runtime_backend=args.runtime_backend,
        execution_mode=args.execution_mode,
        board_model=args.board_model,
        npz_path=Path(args.npz_path) if args.npz_path else None,
        device=args.device,
    )
    print(json.dumps(summary, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
