# Journal Submission Snapshot

This directory mirrors the current journal-submission source from the workspace-level `paper/` folder.

Included here are only the files required to inspect or rebuild the submission version:

- `main.tex`
- `main.bib`
- `acnsart.cls`
- `Makefile`
- `main.xmpdata`
- `pdfa.xmpi`
- `jec-logo.jpg`
- `figures/`

This snapshot is kept separate from `paper/arXiv/` so that the reviewer-facing repository preserves both the arXiv materials and the current submission manuscript without mixing their build systems.

To regenerate the journal PDF locally, run `make` inside this directory.
