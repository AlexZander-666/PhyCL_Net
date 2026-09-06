"""Recalculate staged evidence using Python's standard library only.

This checks saved numbers and file integrity; it does not train, infer, or
validate manuscript-to-checkpoint identity. No network or private data required.
"""
import argparse
import csv
import hashlib
import itertools
import json
import math
from pathlib import Path
import statistics

ROOT = Path(__file__).resolve().parents[1]
EVIDENCE = ROOT / 'artifacts/revision_20260906'
SUBJECTS = {'SA01', 'SA02', 'SA04', 'SA05', 'SA06', 'SA09',
            'SA10', 'SA11', 'SA17', 'SA18', 'SA19', 'SA21'}


def require(condition, message):
    if not condition:
        raise ValueError(message)


def read_csv(path):
    with path.open(encoding='utf-8-sig', newline='') as stream:
        return list(csv.DictReader(stream))


def sha256(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def classification(rows):
    tn = sum(r['y_true'] == 0 and r['y_pred'] == 0 for r in rows)
    tp = sum(r['y_true'] == 1 and r['y_pred'] == 1 for r in rows)
    fp = sum(r['y_true'] == 0 and r['y_pred'] == 1 for r in rows)
    fn = sum(r['y_true'] == 1 and r['y_pred'] == 0 for r in rows)
    return dict(accuracy=(tp + tn) / len(rows),
                macro_f1=(2 * tp / (2 * tp + fp + fn) +
                          2 * tn / (2 * tn + fp + fn)) / 2,
                sensitivity=tp / (tp + fn), specificity=tn / (tn + fp))


def constrained_roc(rows):
    """All observed threshold points, tied scores together, no interpolation."""
    positive = sum(r['y_true'] for r in rows)
    negative = len(rows) - positive
    require(positive > 0 and negative > 0, 'ROC requires both classes')
    tp = fp = 0
    tpr_at_1 = tpr_at_5 = 0.0
    fpr_at_95 = 1.0
    ordered = sorted(rows, key=lambda r: r['y_prob'], reverse=True)
    for _, group in itertools.groupby(ordered, key=lambda r: r['y_prob']):
        for row in group:
            tp += row['y_true']
            fp += 1 - row['y_true']
        # Integer constraints avoid boundary errors at exact percentages.
        if fp * 100 <= negative:
            tpr_at_1 = max(tpr_at_1, tp / positive)
        if fp * 100 <= 5 * negative:
            tpr_at_5 = max(tpr_at_5, tp / positive)
        if tp * 100 >= 95 * positive:
            fpr_at_95 = min(fpr_at_95, fp / negative)
    return dict(tpr_at_fpr_le_1_pct=100 * tpr_at_1,
                tpr_at_fpr_le_5_pct=100 * tpr_at_5,
                fpr_at_tpr_ge_95_pct=100 * fpr_at_95)


def verify_sources():
    manifest = json.loads((ROOT / 'artifacts/manifests/revision_source_manifest.json').read_text(encoding='utf-8'))
    for entry in manifest['entries']:
        path = (ROOT / entry['path']).resolve()
        require(path.is_relative_to(ROOT), 'Source path escapes repository')
        require(sha256(path) == entry['sha256'], f'Source digest mismatch: {entry["path"]}')
    return len(manifest['entries'])


def paired_faa():
    rows = read_csv(EVIDENCE / 'report_transcriptions/classification/paired_macro_f1.csv')
    expected = {(f'SA{i:02}', seed) for i in range(1, 13) for seed in [42, 123, 456, 789, 1024]}
    keys = [(r['subject'], int(r['seed'])) for r in rows]
    require(len(keys) == 60 and set(keys) == expected, 'Incomplete or duplicate FAA paired grid')
    values = {}
    for row in rows:
        faa = float(row['faa_macro_f1_pct'])
        no_faa = float(row['no_faa_macro_f1_pct'])
        require(0 <= faa <= 100 and 0 <= no_faa <= 100, 'F1 outside percentage range')
        delta = faa - no_faa
        require(abs(delta - float(row['recomputed_delta_pp'])) < 1e-10, 'FAA delta mismatch')
        require(abs(delta - float(row['reported_delta_pp'])) < 1e-10, 'Reported FAA delta mismatch')
        values[(row['subject'], int(row['seed']))] = (faa, no_faa, delta)
    arm_rows = read_csv(EVIDENCE / 'report_transcriptions/classification/all_subject_seed_metrics.csv')
    arm_keys = [(r['arm'], r['subject'], int(r['seed'])) for r in arm_rows]
    require(len(arm_keys) == 120 and set(arm_keys) == {(arm, subject, seed) for arm in ['faa', 'no_faa'] for subject, seed in expected},
            'Incomplete or duplicate 120-cell arm grid')
    for row in arm_rows:
        require(float(row['threshold']) == .5, 'Supplementary decision threshold changed')
        paired = values[(row['subject'], int(row['seed']))][0 if row['arm'] == 'faa' else 1]
        require(float(row['macro_f1_pct']) == paired, 'Paired and arm grid differ')
    result = {'status': 'DESCRIPTIVE_RECALCULATION', 'subjects': 12, 'seeds_per_arm': 5, 'pairs': 60,
              'sd_unit': 'sample SD across subject means after averaging five seeds'}
    by_subject = {}
    by_seed = {}
    for subject in sorted({k[0] for k in values}):
        by_subject[subject] = [statistics.mean(v[i] for k, v in values.items() if k[0] == subject) for i in range(3)]
    for seed in sorted({k[1] for k in values}):
        by_seed[str(seed)] = [statistics.mean(v[i] for k, v in values.items() if k[1] == seed) for i in range(3)]
    for i, name in enumerate(['faa_pct', 'no_faa_pct', 'delta_pp']):
        result[name] = {'mean': statistics.mean(v[i] for v in by_subject.values()),
                        'subject_sd': statistics.stdev(v[i] for v in by_subject.values())}
    result['subject_means_faa_no_faa_delta'] = by_subject
    result['seed_means_faa_no_faa_delta'] = by_seed
    result['positive_cells'] = sum(v[2] > 0 for v in values.values())
    result['positive_subject_means'] = sum(v[2] > 0 for v in by_subject.values())
    result['positive_seed_means'] = sum(v[2] > 0 for v in by_seed.values())
    expected_rounded = [(97.73, .49), (97.10, .59), (.63, .12)]
    for name, (mean, sd) in zip(['faa_pct', 'no_faa_pct', 'delta_pp'], expected_rounded):
        require(round(result[name]['mean'], 2) == mean and round(result[name]['subject_sd'], 2) == sd,
                f'Manuscript paired FAA value mismatch: {name}')
    return result


def historical_runs():
    results = {}
    total_rows = 0
    for folder in sorted((EVIDENCE / 'historical_runs').iterdir()):
        if not folder.is_dir():
            continue
        seeds = []
        for seed in [42, 123]:
            rows = read_csv(folder / f'predictions_seed{seed}.csv')
            require(len(rows) == 21678, f'Unexpected row count: {folder.name}/{seed}')
            for i, row in enumerate(rows):
                require(int(row['source_row']) == i, 'Prediction rows reordered or missing')
                for key in ['y_true', 'y_pred', 'fold']:
                    row[key] = int(row[key])
                row['y_prob'] = float(row['y_prob'])
                require(row['y_true'] in [0, 1] and row['y_pred'] in [0, 1], 'Nonbinary labels')
                require(math.isfinite(row['y_prob']) and 0 <= row['y_prob'] <= 1, 'Invalid score')
            require({r['subject'] for r in rows} == SUBJECTS, 'Historical subject contract mismatch')
            require(sum(r['y_true'] for r in rows) == 9000, 'Historical label count mismatch')
            folds = json.loads((folder / f'folds_seed{seed}.json').read_text(encoding='utf-8'))['folds']
            require(len(folds) == 12 and len({f['test_subject'] for f in folds}) == 12, 'Incomplete folds')
            computed = []
            for fold in folds:
                selected = [r for r in rows if r['subject'] == fold['test_subject']]
                require(len(selected) == fold['n_test'], 'Fold count mismatch')
                require({r['fold'] for r in selected} == {fold['fold']}, 'Fold identity mismatch')
                metrics = classification(selected)
                for key, value in metrics.items():
                    require(abs(value - fold['metrics'][key]) < 1e-9, f'Fold metric mismatch: {folder.name}/{seed}/{key}')
                computed.append(metrics)
            item = {'seed': seed, 'windows': len(rows),
                    **{key + '_pct': 100 * statistics.mean(f[key] for f in computed) for key in computed[0]},
                    **constrained_roc(rows)}
            seeds.append(item)
            total_rows += len(rows)
        results[folder.name] = {'status': 'HISTORICAL_PREDICTION_RECALCULATION', 'seeds': seeds,
                                'mean': {k: statistics.mean(s[k] for s in seeds) for k in seeds[0] if k not in ['seed', 'windows']}}
    # Published baseline rows: fold means within seed, then equal mean of two seeds.
    expected = {'phycl_full': (98.04, 97.98), 'inceptiontime': (97.91, 97.85),
                'tcn': (97.13, 97.04), 'transformer': (95.48, 95.34),
                'resnet': (95.13, 94.98), 'lstm': (95.02, 94.86)}
    for model, (acc, f1) in expected.items():
        require(round(results[model]['mean']['accuracy_pct'], 2) == acc, f'Baseline accuracy mismatch: {model}')
        require(round(results[model]['mean']['macro_f1_pct'], 2) == f1, f'Baseline F1 mismatch: {model}')
    expected_roc = {'phycl_full': (96.02, 99.28, .82), 'inceptiontime': (95.52, 99.11, .90),
                    'tcn': (93.38, 98.12, 1.38), 'transformer': (82.98, 95.86, 4.19)}
    for model, target in expected_roc.items():
        actual = results[model]['mean']
        require(tuple(round(actual[key], 2) for key in ['tpr_at_fpr_le_1_pct', 'tpr_at_fpr_le_5_pct', 'fpr_at_tpr_ge_95_pct']) == target,
                f'Baseline ROC mismatch: {model}')
    return {'rows_checked': total_rows, 'runs': results}


def hardware():
    report = EVIDENCE / 'report_transcriptions/hardware'
    result = {'status': 'SAVED_RECORD_AND_REPORT_ARITHMETIC_ONLY'}
    for platform, name, count, windows in [('raspberry_pi_zero_2_w', 'runtime_intervals.csv', 12, 67500),
                                          ('apollo4_blue_plus', 'board_2_runtime_intervals.csv', 6, 202500)]:
        rows = read_csv(report / platform / name)
        require(len(rows) == count and int(rows[-1]['cumulative_windows']) == windows, 'Hardware interval mismatch')
        require(all(int(r['deadline_violations']) == 0 for r in rows), 'Reported deadline violations changed')
        result[platform] = {'intervals': count, 'reported_cumulative_windows': windows, 'reported_deadline_violations': 0}
    sessions = read_csv(report / 'apollo4_blue_plus/board_sessions.csv')
    require(len(sessions) == 9 and all(not r['arm'] for r in sessions), 'Board/session arm attribution changed')
    result['apollo_sessions'] = {key: [min(float(r[key]) for r in sessions), max(float(r[key]) for r in sessions)]
                                 for key in ['p95_latency_ms', 'mean_power_mw', 'peak_sram_kb']}
    for name, target in [('fixed', (622.78, 669.91, 255.25)), ('real_windows', (619.19, 670.92, 255.95))]:
        data = json.loads((ROOT / f'artifacts/staging/orangepi/orangepi_cpu_{name}_benchmark.json').read_text(encoding='utf-8'))
        require(data['input_shape'] == [1, 3, 512] and data['warmup_count'] == 50 and data['repeat_count'] == 200, 'Orange Pi protocol mismatch')
        actual = (data['latency_ms']['p50'], data['latency_ms']['p95'], data['memory']['peak_rss_mb'])
        require(tuple(round(v, 2) for v in actual) == target, 'Orange Pi manuscript value mismatch')
        result['orangepi_' + name] = {'p50_ms': actual[0], 'p95_ms': actual[1], 'peak_rss_mb': actual[2]}
    result['desktop_reported_p50_reduction_pct'] = 100 * (184.31 - 125.99) / 184.31
    metrics = read_csv(report / 'apollo4_blue_plus/reported_metrics.csv')
    for label, key in [('整板平均功率', 'power'), ('每窗能量（增量）', 'energy')]:
        arms = {r['arm']: float(r['numeric_value']) for r in metrics if r['metric_reported'] == label}
        require(set(arms) == {'faa', 'no_faa'}, 'Missing Apollo comparison arm')
        result[f'apollo_reported_{key}_increase_pct'] = 100 * (arms['faa'] - arms['no_faa']) / arms['no_faa']
    return result


def verify_checksums():
    path = ROOT / 'artifacts/manifests/reviewer_package.sha256'
    count = 0
    for line in path.read_text(encoding='utf-8').splitlines():
        expected, name = line.split('  ', 1)
        target = (ROOT / name).resolve()
        require(target.is_relative_to(ROOT), 'Checksum path escapes repository')
        require(sha256(target) == expected, f'Package digest mismatch: {name}')
        count += 1
    return count


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--write-summary', type=Path, help='Write new deterministic descriptive summary')
    args = parser.parse_args()
    report = {'verification_scope': 'file integrity and saved-number recalculation; no new experiments',
              'source_files_checked': verify_sources(), 'paired_faa': paired_faa(),
              'historical_predictions': historical_runs(), 'hardware': hardware()}
    # Stable precision across Python versions; never truncate prediction input.
    def rounded(value):
        if isinstance(value, float):
            return round(value, 10)
        if isinstance(value, dict):
            return {k: rounded(v) for k, v in value.items()}
        if isinstance(value, list):
            return [rounded(v) for v in value]
        return value
    report = rounded(report)
    if args.write_summary:
        require(not args.write_summary.exists(), 'Use a new output path; existing evidence will not be overwritten')
        args.write_summary.parent.mkdir(parents=True, exist_ok=True)
        args.write_summary.write_text(json.dumps(report, ensure_ascii=False, indent=2, allow_nan=False) + '\n', encoding='utf-8')
        print('Saved descriptive recalculation:', args.write_summary)
    else:
        expected = json.loads((EVIDENCE / 'recalculated_summary.json').read_text(encoding='utf-8'))
        require(report == expected, 'Recalculated summary differs from committed summary')
        count = verify_checksums()
        print(f'PASS: {count} package digests; {report["source_files_checked"]} source-traced files; '
              f'60 FAA pairs; {report["historical_predictions"]["rows_checked"]} historical prediction rows; '
              'saved hardware arithmetic. No training, inference, or hardware rerun.')


if __name__ == '__main__':
    main()
