from pathlib import Path
import importlib.util
import sys

import numpy as np
import torch


REPO_ROOT = Path(__file__).resolve().parents[1]
CODE_DIR = REPO_ROOT / "code"
SCRIPTS_DIR = CODE_DIR / "scripts"

if str(CODE_DIR) not in sys.path:
    sys.path.insert(0, str(CODE_DIR))


def _load_script_module(module_name: str, path: Path):
    spec = importlib.util.spec_from_file_location(module_name, path)
    module = importlib.util.module_from_spec(spec)
    assert spec is not None
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def test_export_model_for_edge_writes_torchscript_bundle_and_sample_npz(tmp_path):
    experiments = importlib.import_module("phycl_net_experiments")
    script_path = SCRIPTS_DIR / "export_model_for_edge.py"
    assert script_path.exists()
    export_module = _load_script_module("export_model_for_edge", script_path)

    config = {
        "model": "phycl_full",
        "model_key": "phycl_full",
        "_model_impl": "phycl_core",
        "num_classes": 2,
        "channels_used": "accel3",
        "window_size": 512,
        "proj_dim": 128,
        "freq_method": "fft",
        "sample_rate": 50.0,
        "fusion_variant": "enhanced",
        "fusion_kernel_sizes": [3, 5, 7],
        "num_bands": 4,
        "adaptive_bands": True,
        "faa_axis_attn": True,
        "ablation": experiments.parse_ablation_config("full"),
    }
    model = experiments.build_model_from_config(config, in_channels=3)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=1)

    checkpoint = experiments.build_checkpoint_state(
        config=config,
        seed=42,
        epoch=1,
        model=model,
        optimizer=optimizer,
        scheduler=scheduler,
        scaler=None,
        best_f1=0.9,
    )
    checkpoint_path = tmp_path / "mock_checkpoint.pth"
    torch.save(checkpoint, checkpoint_path)

    prepared_npz = tmp_path / "prepared_windows.npz"
    np.savez(
        prepared_npz,
        x=np.random.randn(4, 3, 512).astype(np.float32),
        y=np.array([0, 1, 0, 1], dtype=np.int64),
        subject=np.array(["s1", "s2", "s3", "s4"]),
    )

    summary = export_module.export_edge_bundle(
        checkpoint_path=checkpoint_path,
        out_dir=tmp_path / "edge_export",
        model_key="phycl_full",
        input_shape=(1, 3, 512),
        prepared_npz=prepared_npz,
        sample_count=2,
    )

    model_path = Path(summary["model_path"])
    manifest_path = Path(summary["manifest_path"])
    sample_npz_path = Path(summary["sample_npz_path"])

    assert model_path.exists()
    assert manifest_path.exists()
    assert sample_npz_path.exists()
    assert summary["runtime"] == "torchscript"
    assert summary["input_shape"] == [1, 3, 512]
    assert summary["sample_count"] == 2

    exported_samples = np.load(sample_npz_path)
    assert exported_samples["x"].shape == (2, 3, 512)
    assert exported_samples["y"].tolist() == [0, 1]


def test_benchmark_on_orangepi_emits_required_latency_json_keys(tmp_path):
    script_path = SCRIPTS_DIR / "benchmark_on_orangepi.py"
    assert script_path.exists()
    benchmark_module = _load_script_module("benchmark_on_orangepi", script_path)

    class TinyEdgeModel(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.conv = torch.nn.Conv1d(3, 8, kernel_size=3, padding=1)
            self.pool = torch.nn.AdaptiveAvgPool1d(1)
            self.fc = torch.nn.Linear(8, 2)

        def forward(self, x):
            x = torch.relu(self.conv(x))
            x = self.pool(x).squeeze(-1)
            return self.fc(x)

    scripted_model_path = tmp_path / "tiny_edge_model.ts"
    scripted = torch.jit.trace(TinyEdgeModel().eval(), torch.randn(1, 3, 512))
    scripted.save(str(scripted_model_path))

    real_windows_path = tmp_path / "real_windows.npz"
    np.savez(
        real_windows_path,
        x=np.random.randn(3, 3, 512).astype(np.float32),
        y=np.array([0, 1, 0], dtype=np.int64),
    )

    fixed_json = tmp_path / "fixed_benchmark.json"
    fixed_summary = benchmark_module.run_benchmark(
        model_path=scripted_model_path,
        out_json=fixed_json,
        input_shape=(1, 3, 512),
        warmup=2,
        repeats=5,
        runtime_backend="torchscript",
        execution_mode="CPU",
        board_model="Unit Test Board",
        npz_path=None,
    )

    real_json = tmp_path / "real_benchmark.json"
    real_summary = benchmark_module.run_benchmark(
        model_path=scripted_model_path,
        out_json=real_json,
        input_shape=(1, 3, 512),
        warmup=1,
        repeats=3,
        runtime_backend="torchscript",
        execution_mode="CPU",
        board_model="Unit Test Board",
        npz_path=real_windows_path,
    )

    for summary, expected_source in ((fixed_summary, "fixed"), (real_summary, "npz")):
        assert summary["board"]["model"] == "Unit Test Board"
        assert summary["runtime"]["backend"] == "torchscript"
        assert summary["input_shape"] == [1, 3, 512]
        assert isinstance(summary["warmup_count"], int)
        assert isinstance(summary["repeat_count"], int)
        assert summary["batch_size"] == 1
        assert summary["execution_mode"] == "CPU"
        assert summary["latency_ms"]["p50"] >= 0.0
        assert summary["latency_ms"]["p95"] >= summary["latency_ms"]["p50"]
        assert summary["input_source"] == expected_source

    assert fixed_json.exists()
    assert real_json.exists()
