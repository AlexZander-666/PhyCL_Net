"""Statistical edge cases for the portable saved-evidence checker."""
import importlib.util
from pathlib import Path

import pytest

path = Path(__file__).resolve().parents[1] / 'scripts/verify_reviewer_evidence.py'
spec = importlib.util.spec_from_file_location('reviewer_evidence', path)
evidence = importlib.util.module_from_spec(spec)
spec.loader.exec_module(evidence)


def test_roc_does_not_split_tied_scores_to_improve_sensitivity():
    rows = [{'y_true': 1, 'y_prob': .8}, {'y_true': 0, 'y_prob': .8}]
    assert evidence.constrained_roc(rows) == {
        'tpr_at_fpr_le_1_pct': 0.0, 'tpr_at_fpr_le_5_pct': 0.0,
        'fpr_at_tpr_ge_95_pct': 100.0,
    }


def test_roc_includes_exact_constraint_boundary_without_interpolation():
    rows = ([{'y_true': 1, 'y_prob': .9}] * 19 +
            [{'y_true': 0, 'y_prob': .9}] +
            [{'y_true': 0, 'y_prob': .5}] * 99 +
            [{'y_true': 1, 'y_prob': .1}])
    result = evidence.constrained_roc(rows)
    assert result['tpr_at_fpr_le_1_pct'] == 95.0
    assert result['fpr_at_tpr_ge_95_pct'] == 1.0


def test_roc_requires_both_classes():
    with pytest.raises(ValueError, match='both classes'):
        evidence.constrained_roc([{'y_true': 1, 'y_prob': .8}])


def test_macro_f1_is_mean_of_both_classes_not_positive_f1():
    rows = ([{'y_true': 0, 'y_pred': 0}] * 90 +
            [{'y_true': 1, 'y_pred': 0}] * 10)
    assert evidence.classification(rows)['macro_f1'] == pytest.approx(90 / 190)


def test_source_integrity_detects_tampering(tmp_path, monkeypatch):
    import json

    (tmp_path / 'artifacts/manifests').mkdir(parents=True)
    file = tmp_path / 'input.csv'
    file.write_text('original', encoding='utf-8')
    manifest = {'entries': [{'path': 'input.csv', 'sha256': evidence.sha256(file)}]}
    (tmp_path / 'artifacts/manifests/revision_source_manifest.json').write_text(json.dumps(manifest), encoding='utf-8')
    monkeypatch.setattr(evidence, 'ROOT', tmp_path)
    assert evidence.verify_sources() == 1
    file.write_text('modified', encoding='utf-8')
    with pytest.raises(ValueError, match='digest mismatch'):
        evidence.verify_sources()


def test_faa_grid_rejects_duplicate_or_missing_pairs(monkeypatch):
    rows = evidence.read_csv(evidence.EVIDENCE / 'report_transcriptions/classification/paired_macro_f1.csv')
    rows[-1] = dict(rows[0])
    monkeypatch.setattr(evidence, 'read_csv', lambda _: rows)
    with pytest.raises(ValueError, match='Incomplete or duplicate'):
        evidence.paired_faa()
