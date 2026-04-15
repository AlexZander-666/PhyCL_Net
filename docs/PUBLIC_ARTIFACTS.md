# Public Artifacts

## Repository Content
- Reviewer-facing code and protocol docs live in this repository.
- `experiments/PhyCL_Net/` is the only public repository boundary for this release package.
- Raw directories under `SCI666/outputs/` are never published directly; every public artifact is first normalized under `artifacts/staging/`.

## Release Assets
- `orangepi-phase5-evidence`
- `noise-robustness-evidence`
- `sisfall-results-pack`
- `cross-dataset-results-pack`

## Publication Status
- `orangepi-phase5-evidence` is release-ready reviewer evidence for the CPU-only Orange Pi Phase 5 benchmark.
- `noise-robustness-evidence` is release-ready supplemental robustness evidence.
- `sisfall-results-pack` contains normalized bundles for `phycl_full`, `lstm`, `resnet`, `tcn`, `transformer`, and `inceptiontime`.
- The `phycl` SisFall bundle is currently blocked because `summary_results.json` reports `n=1` while three `loso_results_seed*.json` files are visible in the source directory.
- `cross-dataset-results-pack` is supportive-only until a clean result pack matching the published supplementary table is identified or regenerated.

## Non-Public Content
- raw datasets
- raw training workspace outputs
- manuscript build trees
- protected files and source-authoritative reference trees
