# Revision evidence

- `report_transcriptions/`: selected source CSVs copied byte-for-byte from the manuscript's evidence supplement; includes full FAA grid and both hardware arms. Its `readme_time` and checklist fields are source annotations, not authenticated execution records.
- `historical_runs/`: complete prediction exports for six main comparison baselines and three historical ablations, with original argument names retained. Each model has seeds 42 and 123, 12 subjects and 21,678 prediction rows per seed. No seeds, errors or unfavorable cells were filtered out.
- `recalculated_summary.json`: deterministic standard-library analysis. Classification uses fold means then seed means; ROC uses per-seed pooled scores, then seed means; FAA SD uses subject means after five-seed averaging.

Prediction columns: `source_row` is the zero-based row in the original JSON prediction arrays; `subject` is a public SisFall subject ID; `fold` is the original fold index; `y_true`/`y_pred` are binary ADL=0/fall=1; `y_prob` is the stored fall probability, exported without rounding. These are model outputs, not raw sensor recordings.

`folds_seed*.json` retains recorded fold metrics, replacing nonfinite NaN with JSON null. Historical zero-valued detection-latency fields must not be read as measured hardware delay. Historical source YAML `args` mappings were extracted to JSON; unrelated machine dependency inventories were omitted. `sample_rate` is not proof of input resampling.

See [source transformations](../manifests/revision_source_manifest.json), [reviewer guide](../../docs/REVIEWER_GUIDE.md), and [evidence limits](../../docs/EVIDENCE_BOUNDARIES.md). Run `python scripts/verify_reviewer_evidence.py` from the repository root.
