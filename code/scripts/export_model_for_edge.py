import argparse
import json
import os
import sys
from pathlib import Path
from typing import Dict, Optional, Sequence, Tuple

import numpy as np
import torch
import torch.nn as nn


REPO_ROOT = Path(__file__).resolve().parents[2]
CODE_DIR = REPO_ROOT / "code"

if str(CODE_DIR) not in sys.path:
    sys.path.insert(0, str(CODE_DIR))

import phycl_net_experiments as experiments


RUNTIME_ONLY_KEYS = {
    "checkpoint",
    "out_dir",
    "device",
    "input_shape",
    "prepared_npz",
    "sample_count",
    "model_path",
    "manifest_path",
}


class EdgeInferenceWrapper(nn.Module):
    """Reduce the training backbone output to logits for edge benchmarking."""

    def __init__(self, backbone: nn.Module):
        super().__init__()
        self.backbone = backbone

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        output = self.backbone(x)
        if isinstance(output, tuple):
            return output[0]
        return output


def merge_config_from_checkpoint(runtime_config: Dict, checkpoint: object) -> Dict:
    merged = dict(runtime_config)
    if not isinstance(checkpoint, dict):
        return merged

    checkpoint_config = checkpoint.get("config")
    if not isinstance(checkpoint_config, dict):
        return merged

    for key, value in checkpoint_config.items():
        if key in RUNTIME_ONLY_KEYS:
            continue
        merged[key] = value

    if "model_key" in merged:
        merged["model"] = merged["model_key"]
    return merged


def build_runtime_config(
    model_key: str,
    ablation: Optional[str] = None,
    channels_used: str = "accel3",
    num_classes: int = 2,
    channels: int = 64,
    n_blocks: int = 4,
    freq_method: str = "fft",
    fusion_variant: str = "enhanced",
    fusion_kernel_sizes: Sequence[int] = (3, 5, 7),
    num_bands: int = 4,
    band_edges: Optional[Sequence[float]] = None,
    adaptive_bands: bool = True,
    faa_axis_attn: bool = True,
    sample_rate: float = 50.0,
    proj_dim: int = 128,
) -> Dict:
    internal_model, effective_ablation_spec, model_name = experiments.resolve_requested_model(model_key, ablation)
    config = {
        "model_key": experiments.canonicalize_public_model_key(model_key),
        "model": experiments.canonicalize_public_model_key(model_key),
        "model_name": model_name,
        "_model_impl": internal_model,
        "channels_used": channels_used,
        "num_classes": num_classes,
        "channels": channels,
        "n_blocks": n_blocks,
        "freq_method": freq_method,
        "fusion_variant": fusion_variant,
        "fusion_kernel_sizes": tuple(fusion_kernel_sizes),
        "num_bands": num_bands,
        "band_edges": tuple(band_edges) if band_edges else None,
        "adaptive_bands": adaptive_bands,
        "faa_axis_attn": faa_axis_attn,
        "sample_rate": sample_rate,
        "proj_dim": proj_dim,
        "ablation": experiments.parse_ablation_config(effective_ablation_spec),
    }
    return config


def _resolve_input_channels(config: Dict) -> int:
    channels_used = config.get("channels_used", "accel3")
    return experiments._channels_from_mode(channels_used)


def _extract_state_dict(checkpoint: object) -> Dict[str, torch.Tensor]:
    if isinstance(checkpoint, dict):
        state_dict = checkpoint.get("model_state")
        if state_dict is None:
            state_dict = checkpoint.get("model_state_dict", checkpoint.get("state_dict", checkpoint))
    else:
        state_dict = checkpoint
    return {key.replace("module.", ""): value for key, value in state_dict.items()}


def _load_model_from_checkpoint(checkpoint_path: Path, runtime_config: Dict, device: torch.device) -> Tuple[nn.Module, Dict]:
    checkpoint = experiments.torch_load_full(str(checkpoint_path), map_location=device)
    merged_config = merge_config_from_checkpoint(runtime_config, checkpoint)
    in_channels = _resolve_input_channels(merged_config)
    model = experiments.build_model_from_config(merged_config, in_channels=in_channels).to(device)
    model.load_state_dict(_extract_state_dict(checkpoint), strict=True)
    model.eval()
    return model, merged_config


def _export_sample_npz(prepared_npz: Path, out_path: Path, sample_count: int) -> int:
    source = np.load(prepared_npz, allow_pickle=True)
    if "x" not in source:
        raise KeyError(f"{prepared_npz} does not contain an 'x' array.")

    x = source["x"]
    if x.ndim != 3:
        raise ValueError(f"Expected x with shape [N, C, L], got {x.shape}")

    actual_count = min(int(sample_count), int(x.shape[0]))
    payload = {"x": x[:actual_count].astype(np.float32, copy=False)}
    if "y" in source:
        payload["y"] = source["y"][:actual_count]
    if "subjects" in source:
        payload["subjects"] = source["subjects"][:actual_count]
    elif "subject" in source:
        payload["subjects"] = source["subject"][:actual_count]
    if "sources" in source:
        payload["sources"] = source["sources"][:actual_count]
    np.savez(out_path, **payload)
    return actual_count


def export_edge_bundle(
    checkpoint_path: Path,
    out_dir: Path,
    model_key: str,
    input_shape: Sequence[int] = (1, 3, 512),
    device: str = "cpu",
    prepared_npz: Optional[Path] = None,
    sample_count: int = 0,
    ablation: Optional[str] = None,
    channels_used: str = "accel3",
    num_classes: int = 2,
    channels: int = 64,
    n_blocks: int = 4,
    freq_method: str = "fft",
    fusion_variant: str = "enhanced",
    fusion_kernel_sizes: Sequence[int] = (3, 5, 7),
    num_bands: int = 4,
    band_edges: Optional[Sequence[float]] = None,
    adaptive_bands: bool = True,
    faa_axis_attn: bool = True,
    sample_rate: float = 50.0,
    proj_dim: int = 128,
) -> Dict:
    checkpoint_path = Path(checkpoint_path)
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    runtime_config = build_runtime_config(
        model_key=model_key,
        ablation=ablation,
        channels_used=channels_used,
        num_classes=num_classes,
        channels=channels,
        n_blocks=n_blocks,
        freq_method=freq_method,
        fusion_variant=fusion_variant,
        fusion_kernel_sizes=fusion_kernel_sizes,
        num_bands=num_bands,
        band_edges=band_edges,
        adaptive_bands=adaptive_bands,
        faa_axis_attn=faa_axis_attn,
        sample_rate=sample_rate,
        proj_dim=proj_dim,
    )

    target_device = torch.device(str(device).lower())
    model, merged_config = _load_model_from_checkpoint(checkpoint_path, runtime_config, target_device)
    wrapper = EdgeInferenceWrapper(model).to(target_device).eval()

    example_input = torch.randn(tuple(int(v) for v in input_shape), device=target_device)
    with torch.no_grad():
        scripted = torch.jit.trace(wrapper, example_input, strict=False)
        scripted = torch.jit.freeze(scripted.eval())

    model_path = out_dir / "phycl_edge_model.ts"
    scripted.save(str(model_path))

    exported_sample_count = 0
    sample_npz_path: Optional[Path] = None
    if prepared_npz is not None:
        sample_npz_path = out_dir / "edge_benchmark_windows.npz"
        exported_sample_count = _export_sample_npz(Path(prepared_npz), sample_npz_path, sample_count)

    summary = {
        "checkpoint": str(checkpoint_path.resolve()),
        "model": merged_config["model"],
        "runtime": "torchscript",
        "input_shape": [int(v) for v in input_shape],
        "model_path": str(model_path.resolve()),
        "model_file_size_bytes": model_path.stat().st_size,
        "manifest_path": str((out_dir / "edge_export_manifest.json").resolve()),
        "sample_npz_path": str(sample_npz_path.resolve()) if sample_npz_path else None,
        "sample_count": exported_sample_count,
        "config": experiments.build_public_config_snapshot(merged_config),
    }

    with open(summary["manifest_path"], "w", encoding="utf-8") as handle:
        json.dump(summary, handle, indent=2)
    return summary


def make_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Export a reviewer-facing PhyCL-Net checkpoint into an edge benchmark bundle.")
    parser.add_argument("--checkpoint", required=True, help="Path to the source checkpoint (.pth)")
    parser.add_argument("--model", default="phycl_full", help="Reviewer-facing model key matching the checkpoint")
    parser.add_argument("--out-dir", required=True, help="Directory to write the exported bundle")
    parser.add_argument("--input-shape", type=int, nargs=3, default=[1, 3, 512], metavar=("B", "C", "L"))
    parser.add_argument("--device", default="cpu", help="Export device, normally cpu")
    parser.add_argument("--prepared-npz", default=None, help="Optional prepared dataset NPZ used to export real windows")
    parser.add_argument("--sample-count", type=int, default=0, help="How many prepared windows to copy into the bundle")
    parser.add_argument("--ablation", type=str, default=None)
    parser.add_argument("--channels-used", type=str, default="accel3")
    parser.add_argument("--num-classes", type=int, default=2)
    parser.add_argument("--channels", type=int, default=64)
    parser.add_argument("--n-blocks", type=int, default=4)
    parser.add_argument("--freq-method", type=str, default="fft")
    parser.add_argument("--fusion-variant", type=str, default="enhanced")
    parser.add_argument("--fusion-kernel-sizes", type=int, nargs="+", default=[3, 5, 7])
    parser.add_argument("--num-bands", type=int, default=4)
    parser.add_argument("--band-edges", type=float, nargs="+", default=None)
    parser.add_argument("--adaptive-bands", action="store_true")
    parser.add_argument("--no-adaptive-bands", action="store_false", dest="adaptive_bands")
    parser.add_argument("--disable-faa-axis-attn", action="store_false", dest="faa_axis_attn")
    parser.add_argument("--sample-rate", type=float, default=50.0)
    parser.add_argument("--proj-dim", type=int, default=128)
    parser.set_defaults(adaptive_bands=True, faa_axis_attn=True)
    return parser


def main() -> int:
    parser = make_parser()
    args = parser.parse_args()
    experiments.ensure_dir(args.out_dir)
    experiments.setup_logging(args.out_dir)

    summary = export_edge_bundle(
        checkpoint_path=Path(args.checkpoint),
        out_dir=Path(args.out_dir),
        model_key=str(args.model).lower(),
        input_shape=args.input_shape,
        device=args.device,
        prepared_npz=Path(args.prepared_npz) if args.prepared_npz else None,
        sample_count=args.sample_count,
        ablation=args.ablation,
        channels_used=args.channels_used,
        num_classes=args.num_classes,
        channels=args.channels,
        n_blocks=args.n_blocks,
        freq_method=args.freq_method,
        fusion_variant=args.fusion_variant,
        fusion_kernel_sizes=args.fusion_kernel_sizes,
        num_bands=args.num_bands,
        band_edges=args.band_edges,
        adaptive_bands=args.adaptive_bands,
        faa_axis_attn=args.faa_axis_attn,
        sample_rate=args.sample_rate,
        proj_dim=args.proj_dim,
    )
    print(json.dumps(summary, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
