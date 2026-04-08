import argparse
import os
import sys
from typing import Tuple

import torch
from fvcore.nn import FlopCountAnalysis


def calculate_complexity(model: torch.nn.Module, input_shape: Tuple[int, ...], device: str = "cpu") -> Tuple[float, int]:
    """
    Calculates FLOPs (MACs) and parameters for a PyTorch model using fvcore.

    Notes:
        - fvcore's `FlopCountAnalysis.total()` corresponds to MAC-like counts for conv/linear kernels;
          many papers report this as "FLOPs". If you want the "2*MACs" convention, report 2x.

    Args:
        model: PyTorch model
        input_shape: Dummy input tensor shape (e.g., (B, C, L) for Conv1d models)
        device: 'cpu' or 'cuda'

    Returns:
        (macs, params) where macs is a float count and params is an int.
    """
    if device != "cpu" and device != "cuda":
        raise ValueError("device must be 'cpu' or 'cuda'")

    model = model.to(device)
    model.eval()
    dummy_input = torch.randn(*input_shape, device=device)

    with torch.no_grad():
        flops = FlopCountAnalysis(model, dummy_input)

    macs = float(flops.total())
    params = int(sum(p.numel() for p in model.parameters() if p.requires_grad))
    return macs, params


def _add_repo_code_to_path() -> None:
    repo_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
    code_dir = os.path.join(repo_root, "code")
    if code_dir not in sys.path:
        sys.path.insert(0, code_dir)


class _LogitsOnly(torch.nn.Module):
    def __init__(self, model: torch.nn.Module):
        super().__init__()
        self.model = model

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        logits, _, _ = self.model(x)
        return logits


def main() -> int:
    parser = argparse.ArgumentParser(description="Compute FLOPs (MACs) and params for the reviewer-facing PhyCL-Net CPU protocol.")
    parser.add_argument("--device", type=str, default="cpu", help="Device to run on (default: cpu).")
    parser.add_argument("--window-size", type=int, default=512, help="Input length L for (B, C, L).")
    parser.add_argument("--in-channels", type=int, default=3, help="Input channels (default: 3).")
    parser.add_argument("--num-classes", type=int, default=2, help="Classifier classes (default: 2).")
    parser.add_argument(
        "--ablation-mspa",
        action="store_true",
        help="Enable MSPA (default: disabled for the time-domain PhyCL-Net configuration).",
    )
    args = parser.parse_args()

    # Requirement from the paper run: stay on CPU to avoid GPU OOM / contention.
    device = "cpu" if args.device.lower() != "cuda" else "cuda"
    if device != "cpu":
        raise RuntimeError("This script is intended to be run on CPU. Use --device cpu.")

    _add_repo_code_to_path()
    from models.phycl_net import PhyCLNet

    ablation_mspa = bool(args.ablation_mspa)
    ablation = {"mspa": ablation_mspa, "dks": True, "faa": True}
    model = PhyCLNet(in_channels=args.in_channels, num_classes=args.num_classes, ablation=ablation)

    # Wrap to ensure traceable tensor output for fvcore.
    model_for_flops = _LogitsOnly(model)

    input_shape = (1, args.in_channels, args.window_size)
    macs, params = calculate_complexity(model_for_flops, input_shape=input_shape, device=device)

    gmacs = macs / 1e9
    gflops = 2.0 * gmacs
    mparams = params / 1e6

    print(f"Model: PhyCL-Net (ablation_mspa={ablation_mspa})")
    print(f"Device: {device}")
    print(f"Input: {input_shape} (B, C, L)")
    print(f"GMACs: {gmacs:.6f}")
    print(f"GFLOPs: {gflops:.6f}")
    print(f"MParams: {mparams:.6f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

