# PhyCL-Net

**Physics-Guided Contrastive Learning for Wearable Fall Detection: Toward Low-Latency On-Device Inference**

Code and selected evidence accompanying the author-supplied **6 September 2026 revision**. Start with the [reviewer evidence guide](docs/REVIEWER_GUIDE.md), [revised paper](paper/revised_20260906/main.pdf), and [point-by-point response](paper/revised_20260906/response_to_reviewers.pdf). The supplied manuscript files are preserved byte for byte.

| Inspect | Entry | Evidence available |
| --- | --- | --- |
| Revised manuscript and changes | [Paper files](paper/revised_20260906/README.md) | Clean PDF, marked PDF, clean TeX/BibTeX, active figures and response |
| Architecture and protocol | [Implementation correspondence](docs/MANUSCRIPT_CODE_MAPPING.md) | PDK, FAA, fusion, MSPA switch, training and preprocessing comparison |
| Main-table comparison baselines | [Saved prediction analysis](artifacts/revision_20260906/recalculated_summary.json) | Six baselines, two seeds each; all 12 folds and scores |
| FAA supplementary analysis | [Complete paired grid](artifacts/revision_20260906/report_transcriptions/classification/paired_macro_f1.csv) | All 60 pairs; subject- and seed-level descriptive recalculation |
| Hardware | [Evidence guide](docs/REVIEWER_GUIDE.md#hardware) | Orange Pi JSON records; Pi/Apollo transcribed metrics, intervals and sessions |
| Reviewer comments | [Response mapping](docs/paper/REVIEWER_RESPONSE_MAPPING.md) | Reviewer 1, Reviewer 2 and all five Reviewer 3 comments |
| Provenance and integrity | [Source manifest](artifacts/manifests/revision_source_manifest.json) | Source hashes, exact transformations and complete prediction exports |
| Validation performed | [Validation record](docs/PACKAGE_VALIDATION.md) | Saved-number checks, 23 tests, source identity and PDF readability |

## Check the evidence without training

Python 3.10 or newer; standard library only, no GPU, private paths or downloads:

```bash
python scripts/verify_reviewer_evidence.py
```

The script verifies package hashes, all 60 FAA pairs, 390,204 historical prediction rows, 216 fold metric records, six baseline classification rows, four baseline ROC rows, and saved hardware arithmetic. It checks saved evidence; it does not perform new training, model inference or device measurements.

The paired FAA report gives **97.73 ± 0.49%** versus **97.10 ± 0.59%** Macro-F1. SD is across 12 subject means after averaging five seeds, and the mean paired difference is **+0.63 ± 0.12 percentage points**. This SA01–SA12 report is distinct from the main experiment's subject set.

## Code and reproduction

`code/phycl_net_experiments.py` is the existing training/evaluation entrypoint. `phycl` disables the explicit MSPA feature branch; `phycl_full` enables it. PDK routing still uses two rFFT-derived descriptors. See the [reproduction guide](docs/REPRODUCIBILITY.md) before running the historical implementation or interpreting its outputs.

The main paper reports five-seed accuracy of 98.21 ± 0.10% and Macro-F1 of 98.16 ± 0.10%. The corresponding complete five-seed run/checkpoint lineage is **not present in this package**. The historical loader and projection-head implementation also differ in specified ways from the revised methods. [Evidence boundaries](docs/EVIDENCE_BOUNDARIES.md) identify these differences, the transfer-table definition questions and hardware attribution limits. Supplied manuscript values are retained as reported, without substituting a different cohort or historical ablation.

Datasets, checkpoints, unpublished raw participant recordings, firmware, credentials and internal editing histories are excluded. Public dataset acquisition and local input requirements are documented in [REPRODUCIBILITY.md](docs/REPRODUCIBILITY.md). No license grant for third-party datasets or journal assets is implied by their citation.

This update publishes repository files on `main`; it does not assert that a separately named GitHub Release exists.
