# Package validation — 6 September 2026

This is an engineering and saved-evidence validation record, not a new scientific experiment.

| Check | Result and scope |
| --- | --- |
| Source identity | 73 source-traced files checked against original source SHA-256. Supplied paper and copied CSV bytes preserved; complete historical prediction exports retain all rows and probability precision. |
| Saved numerical evidence | `python scripts/verify_reviewer_evidence.py` passes: 120 FAA arm cells / 60 pairs; 390,204 historical predictions and 216 fold records; six baseline classification rows; four baseline ROC rows; saved hardware arithmetic. |
| Unit and existing interface tests | `python -m pytest -q tests --disable-warnings --basetemp outputs/package_pytest_final`: 23 passed. The suite includes synthetic export/benchmark interface tests, not real device measurements or manuscript accuracy evaluation. |
| Trace warnings | 28 warnings from existing TorchScript tracing/deprecation behavior. A passing fixed-input interface test does not guarantee generalization of a trace to arbitrary input shapes. No model implementation change is made here. |
| PDF readability | Supplied clean paper: 20 pages; marked paper: 24 pages; response: 3 pages; seven active figure PDFs: one page each. All open unencrypted. Expected title/reviewer text is extractable. |
| PDF qualification | The reader reported duplicate PDF dictionary keys in supplied assets. Files are preserved as supplied; no regenerated PDF, layout repair, or visual redesign is claimed. |
| Navigation | Relative Markdown links in published documents resolve inside the repository; source-note references to omitted larger-package files are explained locally. |
| Git and portability | Package checksum manifest covers the tracked publication surface except itself; text checkout policy is explicit and exact manuscript/source copies disable newline conversion. Copied manuscript/report whitespace is preserved and excluded from whitespace rewriting; generated CSV CRLF is allowed explicitly. `git diff --check` uses those declared attributes. |

No manuscript-scale training, real-checkpoint accuracy evaluation, new cohort collection, hardware measurement, battery test or deployed threshold calibration was performed. See [remaining evidence boundaries](EVIDENCE_BOUNDARIES.md) before making a reproduction claim.

To independently repeat the saved-number and hash checks from a fresh checkout:

```bash
python scripts/verify_reviewer_evidence.py
```

The new checker uses only Python's standard library. Model-side tests require the existing project dependencies and pytest. Validation of a source SHA identifies bytes; it does not authenticate the original experiment's execution history.
