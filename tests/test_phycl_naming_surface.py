from pathlib import Path
import importlib
import sys


REPO_ROOT = Path(__file__).resolve().parents[1]
CODE_DIR = REPO_ROOT / "code"

if str(CODE_DIR) not in sys.path:
    sys.path.insert(0, str(CODE_DIR))


def test_manuscript_facing_entrypoint_and_modules_exist():
    assert (CODE_DIR / "phycl_net_experiments.py").exists()
    assert (CODE_DIR / "models" / "phycl_net.py").exists()
    assert (CODE_DIR / "scripts" / "run_baseline_comparison.py").exists()
    assert (CODE_DIR / "scripts" / "evaluate_noise_robustness.py").exists()
    assert (REPO_ROOT / "scripts" / "profile_phycl_complexity.py").exists()
    assert not (CODE_DIR / "DMC_Net_experiments.py").exists()
    assert not (CODE_DIR / "models" / "ams_net_v2.py").exists()


def test_manuscript_facing_model_and_loss_names_are_primary():
    from models import PhyCLNet
    from models.phycl_net import PhyCLBlock, CrossGatedFusion
    from losses import PhyCLNetLoss

    assert PhyCLNet is not None
    assert PhyCLBlock is not None
    assert CrossGatedFusion is not None
    assert PhyCLNetLoss is not None


def test_legacy_public_aliases_are_no_longer_exposed():
    models = importlib.import_module("models")
    losses = importlib.import_module("losses")
    experiments = importlib.import_module("phycl_net_experiments")

    assert not hasattr(models, "AMSNetV2")
    assert not hasattr(losses, "AMSNetLoss")
    assert "dmc" not in experiments.PUBLIC_MODEL_KEYS
    assert "liteams" not in experiments.PUBLIC_MODEL_KEYS
    assert "dual_branch_baseline" in experiments.PUBLIC_MODEL_KEYS
    assert "compact_comparison_baseline" in experiments.PUBLIC_MODEL_KEYS
    assert experiments.MODEL_ALIASES["dmc"][0] == "dual_branch_baseline"
    assert experiments.MODEL_ALIASES["liteams"][0] == "compact_comparison_baseline"
    assert experiments.resolve_requested_model("dmc", None)[0] == "dual_branch_baseline"
    assert experiments.resolve_requested_model("liteams", None)[0] == "compact_comparison_baseline"


def test_internal_baseline_language_is_cleaned_up():
    source = (CODE_DIR / "phycl_net_experiments.py").read_text(encoding="utf-8")
    forbidden_terms = [
        "ImprovedDMCBlock",
        "legacy alias",
        "PhyCL-inspired baseline",
        "lightweight PhyCL baseline",
    ]

    for term in forbidden_terms:
        assert term not in source


def test_run_metadata_uses_reviewer_facing_model_keys():
    source = (CODE_DIR / "phycl_net_experiments.py").read_text(encoding="utf-8")

    assert "config['requested_model']" not in source
    assert "config['model_internal']" not in source
    assert "config['model_display_name']" not in source
    assert "(implementation=" not in source
    assert "config['model_key']" in source
    assert "config['model_name']" in source


def test_output_artifact_names_use_reviewer_facing_terms():
    noise_source = (CODE_DIR / "scripts" / "evaluate_noise_robustness.py").read_text(encoding="utf-8")
    baseline_source = (CODE_DIR / "scripts" / "run_baseline_comparison.py").read_text(encoding="utf-8")

    assert "baseline_accuracy" not in noise_source
    assert "baseline_f1" not in noise_source
    assert "Baseline F1 (" not in noise_source
    assert "clean_accuracy" in noise_source
    assert "clean_f1" in noise_source
    assert "Reference F1 (" in noise_source
    assert "_baseline.pth" not in baseline_source
    assert "_checkpoint.pth" in baseline_source


def test_reviewer_facing_docs_match_current_artifact_names():
    readme = (REPO_ROOT / "README.md").read_text(encoding="utf-8")
    repro = (REPO_ROOT / "docs" / "REPRODUCIBILITY.md").read_text(encoding="utf-8")
    manifest = (REPO_ROOT / "docs" / "REPRODUCIBILITY_MANIFEST.json").read_text(encoding="utf-8")
    combined = "\n".join([readme, repro, manifest])

    assert "LSTM_baseline.pth" not in combined
    assert "ResNet_baseline.pth" not in combined
    assert "baseline_accuracy" not in combined
    assert "baseline_f1" not in combined
    assert "lstm_checkpoint.pth" in combined
    assert "resnet_checkpoint.pth" in combined
    assert "noise_robustness_curve.png" in combined
    assert "clean_accuracy" in combined
    assert "clean_f1" in combined


def test_readme_opening_matches_reviewer_scope_tone():
    readme = (REPO_ROOT / "README.md").read_text(encoding="utf-8")

    assert "reviewer-facing reproducibility repository" not in readme
    assert "reviewer-facing code and protocol package" in readme
    assert "not a mirror of the full local workspace" in readme
    assert "not a second copy of the journal submission package" in readme


def test_readme_sections_use_hard_boundary_wording():
    readme = (REPO_ROOT / "README.md").read_text(encoding="utf-8")

    assert "Minimal supporting scripts for baseline comparison, CPU complexity measurement, and noise robustness evaluation" not in readme
    assert "Reproducibility notes and reviewer-facing documentation" not in readme
    assert "The retained reviewer-facing support scripts:" in readme
    assert "The minimal reviewer-facing documentation:" in readme
    assert "The authoritative run protocol is documented in `docs/REPRODUCIBILITY.md`." not in readme
    assert "If a manuscript statement and a legacy script comment disagree" not in readme
    assert "The canonical run protocol is defined in `docs/REPRODUCIBILITY.md`." in readme
    assert "If any README text, script comment, or local note conflicts with the manuscript-facing commands" in readme
