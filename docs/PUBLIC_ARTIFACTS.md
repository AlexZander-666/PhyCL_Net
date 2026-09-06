# Public artifacts

The public surface is the repository root. Actual tracked files, not proposed release names, define availability.

| Directory | Content | Status |
| --- | --- | --- |
| `artifacts/revision_20260906/report_transcriptions/` | Complete selected FAA grid and Pi/Apollo CSV records | Report-transcribed; not raw training/device logs |
| `artifacts/revision_20260906/historical_runs/` | Nine historical configurations × two seeds, all predictions and folds | Complete selected prediction exports; old protocol retained |
| `artifacts/revision_20260906/recalculated_summary.json` | Descriptive and historical-prediction arithmetic | Reproducible with the standard-library checker |
| `artifacts/staging/sisfall/` | Previously published normalized six-baseline records | Historical summaries; retained unchanged |
| `artifacts/staging/orangepi/` | Previously published benchmark records | Record arithmetic agrees; model identity unresolved |
| `artifacts/staging/noise/` | Previously published AWGN CSV and plots | Historical support; final model identity unresolved |
| `artifacts/staging/cross_dataset/` | Previously published mixed-data experiment | Different protocol; not the six-row manuscript transfer table |

The [source manifest](../artifacts/manifests/revision_source_manifest.json) records copy/extraction transformations and source hashes. [Package SHA-256](../artifacts/manifests/reviewer_package.sha256) covers the current tracked surface except itself. [Artifact checksums](../artifacts/manifests/artifact_checksums.sha256) cover staged and revised evidence. Hashes establish file identity and transfer integrity, not experiment authenticity.

Complete manuscripts and journal submission assets, raw acquisition datasets, checkpoints, device firmware, private paths/credentials, author correspondence unrelated to the supplied review response, internal audit drafts and duplicate editing histories are excluded. Existing public historical numerical files are preserved, including nonmatching transfer evidence with an explicit notice. No new GitHub Release or dataset mirror is implied.
