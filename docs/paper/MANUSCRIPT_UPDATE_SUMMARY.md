# PhyCL-Net Manuscript Notes

- Manuscript source: `paper/arXiv/main.tex`
- Figures: `paper/arXiv/figures/`
- Experimental log (source of truth): `docs/experiments/1.md`

## Build (LaTeX)
```bash
cd paper/arXiv
pdflatex main.tex
bibtex main
pdflatex main.tex
pdflatex main.tex
```

## Data Integrity
- If you update any reported number in `paper/arXiv/main.tex`, update `docs/experiments/1.md` first and keep the extraction traceable.
