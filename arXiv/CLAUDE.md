# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Repository Context

This is the **arXiv manuscript directory** for the PhyCL-Net fall detection research paper. The parent directory (D:\SCI666\) contains the full machine learning training codebase, while this subdirectory focuses on the LaTeX manuscript preparation and submission to arXiv.

**Paper Title**: PhyCL-Net: Physics-Inspired Contrastive Lightweight Network for Wearable Fall Detection

**Target Venue**: arXiv preprint (based on NeurIPS-style template)

## Document Structure

- **[main.tex](main.tex)** - Main manuscript file containing the full paper content
- **[references.bib](references.bib)** - BibTeX bibliography file with all citations
- **[arxiv.sty](arxiv.sty)** - Custom LaTeX style file (NeurIPS-based aesthetic)
- **[template.tex](template.tex)** - Original template example (reference only)
- **figures/** - All paper figures (PDF format, named fig1_*.pdf through fig8_*.pdf)
- **[1.md](1.md)** - Comprehensive experimental data summary (Chinese) with verified results

## Building the Paper

### Compile LaTeX to PDF

```bash
# Standard LaTeX compilation workflow
pdflatex main.tex
bibtex main
pdflatex main.tex
pdflatex main.tex
```

**Note**: Run pdflatex three times (once before bibtex, twice after) to resolve all cross-references and citations.

### For arXiv Submission

When preparing for arXiv upload, you need to create a self-contained .tex file:

```bash
# Generate .bbl file
pdflatex main.tex
bibtex main

# Manual steps:
# 1. Copy content from main.bbl into main.tex at \begin{thebibliography}
# 2. Comment out \bibliography{references} line
# 3. Upload main.tex + figures/ + arxiv.sty to arXiv
```

See [README.md](README.md) for detailed arXiv submission instructions.

## Key Paper Content

### Core Contribution

PhyCL-Net achieves **98.20% accuracy** on fall detection with only **1.049M parameters** by removing the computationally expensive Multi-Scale Spectral Pyramid Analysis (MSPA) module while retaining physics-aware components (PDK and FAA).

### Critical Trade-off (Central Narrative)

**Removing MSPA causes**:
- TPR@FPR=1% drops from 96.02% → 93.29% (minor sensitivity loss at strict threshold)

**But yields**:
- 31.5% latency reduction (184ms → 126ms)
- 36.7% parameter reduction (1.66M → 1.05M)
- Lowest FPR@TPR=95% (0.72%) among all models

This is a **pragmatic engineering trade-off** optimized for edge wearable deployment.

### Key Experimental Results (Verified from 1.md)

All numerical values below are **verified from original experiment logs** - do not modify these without re-running experiments:

**Main Results (LOSO, 12 folds, 5 seeds)**:
- PhyCL-Net: 98.20% Acc, 98.15% Macro-F1, 1.049M params
- MSPA-FAA-PDK baseline: 98.04% Acc, 97.98% Macro-F1, 1.657M params

**Safety Metrics**:
- PhyCL-Net TPR@FPR=1%: 93.29%
- PhyCL-Net FPR@TPR=95%: 0.72% (lowest among all models)

**Efficiency**:
- PhyCL-Net latency: 125.99ms (p50), 141.43ms (p95)
- Baseline latency: 184.31ms (p50), 203.27ms (p95)

## Figure Files

All figures follow the naming convention `figN_description.pdf`:

1. **fig1_accuracy_vs_params.pdf** - Accuracy vs model complexity scatter plot
2. **fig2_radar_comparison.pdf** - Multi-metric radar chart (top 3 models)
3. **fig3_tsne_comparison.pdf** - t-SNE embedding visualization
4. **fig4_latency_reduction.pdf** - CPU inference latency bar chart
5. **fig5_safety_metrics.pdf** - Safety-critical metrics comparison
6. **fig6_noise_robustness.pdf** - Noise robustness curves (SNR analysis)
7. **fig7_attention_heatmap.pdf** - FAA attention visualization
8. **fig8_confusion_matrix.pdf** - 34-class confusion matrix

## Data Integrity Protocol

**CRITICAL**: All experimental numbers in [main.tex](main.tex) are verified against original experiment logs documented in [1.md](1.md).

**Before modifying any numerical results**:
1. Check if the value exists in `1.md` with verification status (✅)
2. If not verified, re-run the experiment and update `1.md` first
3. Never use LLM to generate or modify experimental numbers
4. Always trace values back to original experiment output files

## Common Tasks

### Update a figure

```bash
# Replace existing figure (ensure same filename)
cp /path/to/new/fig1_accuracy_vs_params.pdf figures/

# Recompile to verify
pdflatex main.tex
```

### Add a new citation

1. Add BibTeX entry to [references.bib](references.bib)
2. Cite in text using `\cite{key}` or `\citep{key}` (natbib)
3. Recompile with bibtex workflow (see above)

### Check paper statistics

```bash
# Word count (approximate, excludes LaTeX commands)
detex main.tex | wc -w

# Figure count
ls figures/*.pdf | wc -l

# Citation count
grep -o '\\cite' main.tex | wc -l
```

### Validate LaTeX syntax

```bash
# Check for common errors
pdflatex -interaction=nonstopmode main.tex 2>&1 | grep -i "error\|warning"

# Check for undefined references
grep "undefined" main.log
```

## Important LaTeX Conventions

### Paper Structure
- Abstract must emphasize the accuracy-efficiency trade-off
- Methodology section describes PDK, FAA, and CGFU modules
- Results section reports LOSO validation with 5 random seeds
- Discussion section transparently addresses limitations (young adults only, simulated falls, single sensor placement)

### Statistical Reporting
- Always report 95% confidence intervals using t-distribution
- Use bootstrap resampling (10,000 iterations) for CIs
- Report both mean and (lower-upper) format: `98.20 (96.81--99.59)`
- Use paired statistical tests across folds (not window-level)

### Notation Conventions
- Bold for vectors: `\mathbf{X}`
- Calligraphic for sets: `\mathcal{D}`
- Time-domain features: `\mathbf{X}_{phy}`
- Frequency-domain features: `\mathbf{X}_{freq}`

## Limitations to Acknowledge

The paper transparently addresses these critical limitations:

1. **Population mismatch**: Only young adults (19-30 years), not elderly (65+)
2. **Simulated falls**: Lab-controlled, not real-world accidental falls
3. **Single sensor**: Waist-mounted accelerometer only
4. **Noise sensitivity**: Accuracy drops to 60.87% at SNR=15dB (trade-off of removing MSPA)
5. **Window overlap**: 50% overlap creates correlated samples (mitigated by fold-level aggregation)

## Related Documentation

- **Parent directory [CLAUDE.md](../CLAUDE.md)**: Machine learning training codebase guidance
- **[1.md](1.md)**: Complete experimental results summary (Chinese) with data integrity verification
- **[MANUSCRIPT_UPDATE_SUMMARY.md](../MANUSCRIPT_UPDATE_SUMMARY.md)**: Recent manuscript updates and revisions

## Style Guidelines

- Use `\citep{}` for parenthetical citations, `\citet{}` for textual citations
- Use `\ref{}` for section/figure/table references with `\cref{}` for automatic "Figure X" formatting
- Maintain consistent terminology: "PhyCL-Net" (not "PhyCLNet"), "MSPA-FAA-PDK" (not "Full model")
- All experimental claims must include confidence intervals
- Use `~` non-breaking space before citations and references: `Figure~\ref{fig:accuracy}`

## Package Dependencies

Required LaTeX packages (already in [main.tex](main.tex)):
- arxiv (custom style)
- hyperref (cross-references)
- natbib (bibliography)
- graphicx (figures)
- amsmath, amssymb, amsfonts (math)
- booktabs (tables)
- algorithm, algorithmic (pseudocode)
- cleveref (automatic reference formatting)

No additional packages should be added without verifying arXiv compatibility.
