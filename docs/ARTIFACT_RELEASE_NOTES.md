# Artifact Release Notes

## Release Name
- `reviewer-artifacts-v1`

## Reviewer-Facing Public Artifacts
This release accompanies the public `PhyCL_Net` reviewer-facing repository surface.

### Included
- Orange Pi AI Pro 20T 24G CPU-only benchmark evidence.
- Supplemental noise robustness evidence.
- Normalized SisFall result bundles for `phycl_full`, `lstm`, `resnet`, `tcn`, `transformer`, and `inceptiontime`.
- Normalized cross-dataset result bundle published as supportive-only supplemental evidence.

### Not Included
- The blocked `phycl` SisFall summary bundle, because `summary_results.json` reports `n=1` while three `loso_results_seed*.json` files are visible in the source directory.
- Raw datasets.
- Full raw training workspace outputs.
- Bulk checkpoints and temporary logs.
- Protected files and source-authoritative trees such as `paper/important.md` and everything under `paper2/`.

### Integrity
- SHA256 checksums are provided in `artifacts/manifests/artifact_checksums.sha256`.
- Every published artifact is staged under `artifacts/staging/` before packaging.

## Known Gaps
- Some legacy `SCI666` result summaries still require normalization before publication.
- The `phycl` SisFall summary requires regeneration or provenance cleanup before it can be released as final backing evidence for the manuscript's main SisFall claim.
- Cross-dataset release evidence remains supportive-only until exact manuscript-table provenance is frozen.
