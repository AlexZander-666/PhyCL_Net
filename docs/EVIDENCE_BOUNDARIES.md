# Evidence boundaries

The revised manuscript remains the author's supplied record. This companion describes available verification, with no fabricated runs or replacement values.

## Main experiment and historical configurations

- The main five-seed PhyCL result (98.21% accuracy, 98.16% Macro-F1), its SD/CI, Figure 5 points, main ROC scores and reported Wilcoxon p=0.47 lack a complete matching run/checkpoint lineage in the available package. They remain **manuscript-reported**.
- All six comparison-baseline mean accuracy/Macro-F1 rows and four baseline ROC rows can be recalculated from the published historical scores. These runs use the original 12 subjects and native 200Hz preprocessing; numerical agreement does not verify the paper's stated 50Hz filtering pipeline.
- `historical_no_mspa` records `mspa:False` with FAA active and a different training setup. Its approximately 98.24% accuracy coincides with the paper's no-FAA row, but this is not proof of that row's identity. The data are published under their original configuration, with no no-FAA relabeling.
- Historical no-DKS and no-TFCL records remain configuration comparisons. Their differing architecture/training settings prevent assigning a matched causal component effect. The archived no-TFCL ROC recalculates to approximately 95.7167% TPR@FPR≤1%, not the table's 95.45%.
- The retained model differs from the paper's schematic in FAA smoothing input, projection-head count, preprocessing and the dual-view contrastive training objective; see [the exact map](MANUSCRIPT_CODE_MAPPING.md). Packaging success does not close those differences.

## Supplementary FAA and hardware

- All 60 FAA pairs are report-transcribed Macro-F1 cells from SA01–SA12, not raw probability records. Their complete descriptive arithmetic is reproducible. They do not replace the main cohort or establish 120 authenticated training executions, confidence intervals, new significance, or raw checkpoint availability.
- Orange Pi files substantiate the saved benchmark's reported numbers. They do not identify a source checkpoint hash or a matched export manifest. Existing historical export commands use `phycl_full`; the board results cannot be assigned to the no-MSPA main model from the filename alone.
- Pi/Apollo files are report transcriptions. Original device timing logs, power waveforms, firmware/model hashes and conversion-accuracy tests are absent. Source “pass” labels and assigned timestamps are not independently verified.
- Different hardware timing boundaries cannot establish a matched speedup. Mean power, per-window energy, device workload duration, SRAM/RSS and battery lifetime remain distinct quantities.

## Transfer and noise

- The six-row 6-channel/length-200 cross-dataset table is preserved in the paper. The current repository's mixed SisFall/MobiFall archive is a different experiment. It is explicitly labeled as such in its directory.
- Five of the six supplied transfer rows report Macro-F1 above accuracy. For a common binary confusion matrix (or common-weight average), macro-F1 cannot exceed accuracy. The F1 definition, aggregation, six-channel construction and unit of “5-shot” require author/source confirmation. This package does not guess corrected numbers or relabel five epochs as five shots.
- Existing noise artifacts are historical supplementary inspection data. Their exact final-model/checkpoint/protocol identity is not supplied. Noise-repeat SD measures repeat variability for that setup, not cross-subject or retraining reliability. At 30dB, the saved accuracy SD is 0.1541%, so it is not strictly below 0.15%.

## What would close the remaining gaps

Matching main-run configs and exact source revision; five-seed fold scores and checkpoints; preprocessing/window identities; contrastive/head and FAA-order confirmation; transfer support/query splits and metric definitions; Orange Pi export hashes; and Pi/Apollo firmware, per-window timing and power records. These are evidence requirements, not work claimed completed by this repository update.

The supplied paper uses anonymous author fields and a template DOI/volume. `10.55056/jec.ARTNUM` is a placeholder, not an assigned article DOI. No citation metadata has been invented.
