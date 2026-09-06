# Package validation — 6 September 2026

This is an engineering and saved-evidence validation record, not a new scientific experiment.

| Check | Result and scope |
| --- | --- |
| Source identity | 55 source-traced experimental files checked against original source SHA-256. Copied CSV bytes preserved; complete historical prediction exports retain all rows and probability precision. |
| Saved numerical evidence | `python scripts/verify_reviewer_evidence.py` passes: 120 FAA arm cells / 60 pairs; 390,204 historical predictions and 216 fold records; six baseline classification rows; four baseline ROC rows; saved hardware arithmetic. |
| Unit and existing interface tests | `python -m pytest -q tests --disable-warnings --basetemp outputs/package_pytest_final`: 23 passed. The suite includes synthetic export/benchmark interface tests, not real device measurements or manuscript accuracy evaluation. |
| Trace warnings | 28 warnings from existing TorchScript tracing/deprecation behavior. A passing fixed-input interface test does not guarantee generalization of a trace to arbitrary input shapes. No model implementation change is made here. |
| Navigation | Relative Markdown links in published documents resolve inside the repository; source-note references to omitted larger-package files are explained locally. |
| Git and portability | Package checksum manifest covers the tracked publication surface except itself; text checkout policy is explicit and exact experimental source copies disable newline conversion. Copied report whitespace is preserved and excluded from whitespace rewriting; generated CSV CRLF is allowed explicitly. `git diff --check` uses those declared attributes. |

No manuscript-scale training, real-checkpoint accuracy evaluation, new cohort collection, hardware measurement, battery test or deployed threshold calibration was performed. See [remaining evidence boundaries](EVIDENCE_BOUNDARIES.md) before making a reproduction claim.

To independently repeat the saved-number and hash checks from a fresh checkout:

```bash
python scripts/verify_reviewer_evidence.py
```

The new checker uses only Python's standard library. Model-side tests require the existing project dependencies and pytest. Validation of a source SHA identifies bytes; it does not authenticate the original experiment's execution history.

## Manuscript withdrawal

The full manuscript directory is excluded from the current public tree. Its former PDF checks are not current-package checks. The withdrawal updates source counts, manifests, checksums and navigation while retaining all experimental evidence. Earlier Git history is not erased by this follow-up commit.
