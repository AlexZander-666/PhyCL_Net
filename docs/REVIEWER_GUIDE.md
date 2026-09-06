# Reviewer evidence guide — 6 September 2026

## Material Passport

- Target: the author-supplied 2026-09-06 `main.tex`, title **Physics-Guided Contrastive Learning for Wearable Fall Detection: Toward Low-Latency On-Device Inference**.
- Source identity: [source manifest](../artifacts/manifests/revision_source_manifest.json); published experimental inputs and their transformations have SHA-256 records. Manuscript files remain local.
- Verification status: **ANALYZED**. Saved numerical evidence has been recalculated; no experiments were rerun.
- Units: classification/ROC values in the generated summary are percentages, paired differences are percentage points, time is milliseconds, power is mW. Stored prediction probabilities use 0–1.
- Authority: the paper remains in the separate submission package. This guide describes what the available files allow a reviewer to inspect; it does not amend the paper or certify all of its scientific claims.

## Paper-to-evidence map

Table labels below refer to the clean manuscript supplied through the journal submission process. The complete manuscript is not included in this repository.

| Paper location | Repository evidence | Interpretation |
| --- | --- | --- |
| Methodology; Figure 1 | [Code correspondence](MANUSCRIPT_CODE_MAPPING.md); Figure 1 in the separate submission | Trace PDK/FAA/global gates; documented implementation differences remain |
| `tab:train_setup`; Dataset and Preprocessing | [Specification and historical protocol](REPRODUCIBILITY.md), [recorded arguments](../artifacts/revision_20260906/historical_runs/) | Paper specification and actual archived settings are separate |
| `tab:main_results` | [Six baseline bundles](../artifacts/staging/sisfall/), [complete score exports](../artifacts/revision_20260906/historical_runs/), [recalculation](../artifacts/revision_20260906/recalculated_summary.json) | Six baseline rows reproduce from equal fold means then equal seed means; main five-seed row remains paper-reported |
| `tab:binary_confusion` | Table in the separate submission | 98.41/1.59 and 2.07/97.93 are complementary mean conditional rates, not pooled counts |
| `tab:complexity` | [CPU profiling utility](../scripts/profile_phycl_complexity.py); reported values in the separate submission | Reported 184.31→125.99 ms gives 31.6423%, rounded to 31.6%; no new CPU benchmark |
| `tab:cross_dataset` | [Scope notes](EVIDENCE_BOUNDARIES.md#transfer-and-noise) | Supplied six-row table preserved; current staged mixed-data experiment does not reproduce it |
| `tab:orangepi_cpu` | [Orange Pi JSON/CSV](../artifacts/staging/orangepi/) | Saved p50/p95/RSS match paper rounding; exported model identity remains unresolved |
| `tab:prototype` | [Transcribed hardware records](../artifacts/revision_20260906/report_transcriptions/hardware/) | Both FAA arms, full reported interval/session coverage; not raw device logs |
| `tab:safety` | [ROC recalculation](../artifacts/revision_20260906/recalculated_summary.json), [algorithm](../scripts/verify_reviewer_evidence.py) | Four historical baseline rows reproduce; PhyCL scores for its reported row are absent |
| `tab:ablation`, `tab:detailed_ablation` | [Historical no-DKS](../artifacts/revision_20260906/historical_runs/historical_no_dks/), [no-TFCL](../artifacts/revision_20260906/historical_runs/historical_no_tfcl/), [no-MSPA](../artifacts/revision_20260906/historical_runs/historical_no_mspa/) | Preserve original configurations; no-MSPA is not relabeled as no-FAA |
| `tab:paired_faa` | [All cells](../artifacts/revision_20260906/report_transcriptions/classification/all_subject_seed_metrics.csv), [paired cells](../artifacts/revision_20260906/report_transcriptions/classification/paired_macro_f1.csv) | All 120 arm cells / 60 pairs; descriptive statistics agree with revised paper |
| Robustness and Practical Margins | [Historical noise table](../artifacts/staging/noise/noise_robustness_results.csv) | Eight SNR levels, five repetitions; final model/protocol linkage not established |

## Statistical inspection

Run `python scripts/verify_reviewer_evidence.py` from the repository root. The reproducible calculation:

1. Verifies all source-transformation and package hashes.
2. Checks the FAA grid for every subject/seed pair, recalculates differences, averages five seeds per subject, then computes the sample SD over 12 subjects. Subject and seed summaries describe the same grid.
3. Uses **all** 21,678 saved predictions for each of 18 historical run/seed files, including errors. Recalculates accuracy, macro-F1, sensitivity and specificity for each of 12 held-out subjects and compares these with saved fold metrics. Averages folds within seed, then the two seeds equally.
4. Pools historical scores within each seed, groups all tied scores together, and evaluates every observed threshold. Computes maximum TPR with FPR≤1%/5%, and minimum FPR with TPR≥95%; no interpolation. Then averages the seed-level ROC summaries.
5. Checks Orange Pi record values and the reported Pi/Apollo interval/session arithmetic.

For a separate output, use `python scripts/verify_reviewer_evidence.py --write-summary outputs/reviewer_check.json` with a new filename. It will not overwrite an existing output.

No new significance tests or confidence intervals are derived here. The main paper's five-seed CI and Wilcoxon p=0.47 cannot be reconstructed from unrelated ablation runs. Repeated noise draws, hardware sessions and seeds are not additional independent subjects. A nonsignificant result does not establish equivalence; a favorable descriptive FAA difference does not establish causality or universal superiority.

Statistical interpretation coverage: **11/11 checks considered**, limited to these supplied records:

| Check | Assessment within this package |
| --- | --- |
| Simpson's paradox | FAA cell, subject-mean and seed-mean differences are all positive; no sign reversal in these summaries. Other subgroup effects are unassessed. |
| Ecological fallacy | Subject/report summaries are not used to infer individual clinical benefit. |
| Berkson/selection bias | Complete selected grids are retained; source cohort selection remains outside what these records can validate. |
| Collider bias | No adjustment model is fitted here; original selection mechanisms cannot be reconstructed. |
| Base-rate neglect | Benchmark class counts are explicit; empirical ROC does not predict clinical alarm burden or positive predictive value. |
| Regression to the mean | No pre/post intervention effect is estimated; no population improvement claim is made. |
| Survivorship bias | No rows are omitted from selected source prediction arrays; missing runs and their completion history remain unknown. |
| Look-elsewhere effect | No new significance test or favorable-seed selection; the full historical search space is not recoverable. |
| Researcher degrees of freedom | Estimator, thresholds, tie rules and transformations are explicit; no retrospective preregistration claim. |
| Correlation versus causation | Descriptive FAA differences and differing historical configurations do not establish a causal component effect. |
| Reverse causality | No human causal/temporal mechanism is inferred from classification or hardware summaries. |

These checks describe interpretation scope; they do not certify the underlying study as free of bias.

## Hardware

- Orange Pi AI Pro 20T 24G: existing TorchScript CPU records, shape 1×3×512, warmup 50, repeats 200. Fixed input p50/p95 is 622.78/669.91 ms; prepared windows 619.19/670.92 ms. These files do not include the export/checkpoint hash needed to establish which architecture was measured.
- Raspberry Pi Zero 2 W: the source report contains both-arm metrics and 12 two-hour summaries ending at 67,500 windows. Interval arm is unspecified.
- Apollo4 Blue Plus: both-arm metrics, six 12-hour summaries ending at 202,500 windows, and nine board/session summaries. Board 2 intervals identify FAA; the nine-session table does not identify the arm.
- Source column labels such as `mean_e2e_latency_ms` or Chinese “end-to-end” are preserved as source text; the revised paper interprets them as reported processing boundaries, not validated fall-onset-to-alert delay.
- Reported mean power increases from 26.7 to 28.4 mW (6.3670%); reported per-window energy increases from 0.035 to 0.039 mJ (11.4286%). These are different measures. No battery endurance or blanket cost-limit pass is inferred.
- `readme_time` and original pass/checklist fields in copied report CSVs are source annotations, not verified experiment timestamps or this package's certification. See [source consistency notes](../artifacts/revision_20260906/report_transcriptions/notes/consistency_notes.md).

## Selection and provenance

All seeds and all subjects in each selected comparison are retained, including unfavorable predictions and added hardware cost. Files were selected by correspondence to the revised paper and review questions, not by the sign of a result. The independent eight-person report, unadopted frozen-threshold tables, old 34-class confusion image, unused fusion figure, private notes, generated transcript logs and duplicate manuscript backups are outside this revision's inspection surface.

Historical mixed-data transfer files were already public and remain available with a local scope notice; they are not offered as proof of the paper's six-row transfer table. Complete raw acquisitions and model weights remain local, consistent with the paper's data-availability statement.
