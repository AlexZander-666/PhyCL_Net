import argparse
import json
import os
import sys
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import torch


REPO_ROOT = Path(__file__).resolve().parents[2]
CODE_DIR = REPO_ROOT / "code"

if str(CODE_DIR) not in sys.path:
    sys.path.insert(0, str(CODE_DIR))

import phycl_net_experiments as experiments


SUPPORTED_DATASETS = ("sisfall", "mobiact", "unimib", "kfall")
RUNTIME_ONLY_KEYS = {
    "checkpoint",
    "data_root",
    "out_dir",
    "device",
    "target_datasets",
    "targets",
    "base_dataset",
    "seed",
}


def resolve_target_datasets(base_dataset: str, requested: Optional[Sequence[str]] = None) -> List[str]:
    base = str(base_dataset).lower()
    if base not in SUPPORTED_DATASETS:
        raise ValueError(f"Unsupported base dataset: {base_dataset}")

    source = requested if requested else SUPPORTED_DATASETS
    targets: List[str] = []
    seen = set()
    for name in source:
        dataset = str(name).lower()
        if dataset not in SUPPORTED_DATASETS:
            raise ValueError(f"Unsupported target dataset: {name}")
        if dataset == base or dataset in seen:
            continue
        seen.add(dataset)
        targets.append(dataset)
    return targets


def build_config(args: argparse.Namespace) -> dict:
    internal_model, effective_ablation_spec, model_name = experiments.resolve_requested_model(args.model, args.ablation)
    config = vars(args).copy()
    config["dataset"] = args.base_dataset
    config["model_key"] = experiments.canonicalize_public_model_key(args.model)
    config["model_name"] = model_name
    config["model"] = config["model_key"]
    config["_model_impl"] = internal_model
    config["band_edges"] = tuple(config["band_edges"]) if config.get("band_edges") else None
    config["fusion_kernel_sizes"] = tuple(config.get("fusion_kernel_sizes", (3, 5, 7)))
    config["ablation"] = experiments.parse_ablation_config(effective_ablation_spec)
    config["target_datasets"] = resolve_target_datasets(args.base_dataset, args.targets)
    return config


def merge_config_from_checkpoint(runtime_config: dict, checkpoint: object) -> dict:
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


def resolve_input_channels(config: dict) -> int:
    if config["dataset"] == "sisfall":
        return experiments._channels_from_mode(config.get("channels_used", "accel3"))
    return int(config.get("in_channels", 3))


def load_model_for_cross_eval(config: dict, checkpoint_path: str, device: torch.device) -> Tuple[torch.nn.Module, int, int, Dict]:
    checkpoint = experiments.torch_load_full(checkpoint_path, map_location=device)
    merged_config = merge_config_from_checkpoint(config, checkpoint)
    in_channels = resolve_input_channels(merged_config)
    model = experiments.build_model_from_config(merged_config, in_channels).to(device)

    if isinstance(checkpoint, dict):
        state_dict = checkpoint.get("model_state")
        if state_dict is None:
            state_dict = checkpoint.get("model_state_dict", checkpoint.get("state_dict", checkpoint))
        seed = int(checkpoint.get("seed", merged_config.get("seed", 42)))
    else:
        state_dict = checkpoint
        seed = int(merged_config.get("seed", 42))

    state_dict = {k.replace("module.", ""): v for k, v in state_dict.items()}
    model.load_state_dict(state_dict, strict=True)
    model.eval()
    return model, seed, in_channels, merged_config


def make_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Evaluate a trained checkpoint across external fall datasets.")
    parser.add_argument("--checkpoint", required=True, help="Path to checkpoint produced by phycl_net_experiments.py")
    parser.add_argument("--data-root", required=True, help="Root directory containing SisFall and/or NPZ datasets")
    parser.add_argument("--out-dir", required=True, help="Directory for cross-dataset evaluation artifacts")
    parser.add_argument("--base-dataset", default="sisfall", choices=SUPPORTED_DATASETS)
    parser.add_argument("--targets", nargs="*", default=None, help="Optional subset of target datasets")
    parser.add_argument("--model", default="phycl", help="Reviewer-facing model key matching the checkpoint")
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--num-classes", type=int, default=2)
    parser.add_argument("--in-channels", type=int, default=3)
    parser.add_argument("--window-size", type=int, default=512)
    parser.add_argument("--stride", type=int, default=256)
    parser.add_argument("--channels-used", type=str, default="accel3")
    parser.add_argument("--channels", type=int, default=64)
    parser.add_argument("--n-blocks", type=int, default=4)
    parser.add_argument("--kernel-sizes", type=int, nargs="+", default=[7, 15, 31, 63])
    parser.add_argument("--freq-method", type=str, default="fft", choices=["fft", "fft_attn", "stft", "cwt", "adaptive_fft"])
    parser.add_argument("--fusion-variant", type=str, default="enhanced", choices=["baseline", "enhanced"])
    parser.add_argument("--fusion-kernel-sizes", type=int, nargs="+", default=[3, 5, 7])
    parser.add_argument("--num-bands", type=int, default=4)
    parser.add_argument("--band-edges", type=float, nargs="+", default=None)
    parser.add_argument("--adaptive-bands", action="store_true")
    parser.add_argument("--no-adaptive-bands", action="store_false", dest="adaptive_bands")
    parser.add_argument("--attn-time", type=str, default="none")
    parser.add_argument("--attn-freq", type=str, default="none")
    parser.add_argument("--attn-fuse", type=str, default="none")
    parser.add_argument("--attn-lite", type=str, default="none")
    parser.add_argument("--disable-faa-axis-attn", action="store_false", dest="faa_axis_attn")
    parser.add_argument("--sample-rate", type=float, default=50.0)
    parser.add_argument("--proj-dim", type=int, default=128)
    parser.add_argument("--rocket-kernels", type=int, default=256)
    parser.add_argument("--rocket-kernel-size", type=int, default=9)
    parser.add_argument("--lstm-hidden", type=int, default=128)
    parser.add_argument("--ablation", type=str, default=None)
    parser.add_argument("--seed", type=int, default=42, help="Fallback seed if checkpoint metadata does not contain one")
    parser.set_defaults(adaptive_bands=True, faa_axis_attn=True)
    return parser


def main() -> int:
    parser = make_parser()
    args = parser.parse_args()
    args.model = str(args.model).lower()
    if args.model not in experiments.ALL_MODEL_KEYS:
        parser.error("Unknown --model value. Public keys are: " + ", ".join(experiments.PUBLIC_MODEL_KEYS))

    experiments.ensure_dir(args.out_dir)
    experiments.setup_logging(args.out_dir)
    experiments.ensure_metric_dependencies(False)

    config = build_config(args)
    if not config["target_datasets"]:
        raise ValueError("No target datasets remain after filtering the base dataset.")

    requested_device = str(args.device).lower()
    if requested_device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("Requested CUDA device but torch.cuda.is_available() is False.")
    device = torch.device(requested_device)

    model, seed, in_channels, config = load_model_for_cross_eval(config, args.checkpoint, device)
    experiments.save_complete_experiment_config(config, args.out_dir)
    results = experiments.evaluate_cross_datasets(
        model=model,
        config=config,
        device=device,
        seed=seed,
        base_dataset=config["dataset"],
        in_channels=in_channels,
        target_datasets=config["target_datasets"],
    )

    summary = {
        "checkpoint": os.path.abspath(args.checkpoint),
        "model": config["model_key"],
        "base_dataset": config["dataset"],
        "target_datasets": config["target_datasets"],
        "seed": seed,
        "device": str(device),
        "results": results,
    }
    with open(os.path.join(args.out_dir, "cross_eval_summary.json"), "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)
    print(json.dumps(summary, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
