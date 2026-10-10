# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Repository scope

This repository is a **versioned data-modeling project** on EDM (electrical discharge machining) process parameter optimization, with iterative snapshots in:
- `ver1/`
- `ver2/`
- `ver3/`
- `verFinal/` (final deliverable baseline)

Root-level data and problem documents live beside these folders, especially:
- `202601B题数据.xlsx` (primary modeling dataset)

## Common commands

### Python environment check
Run in `verFinal/`:

```bash
python requirement.py
```

This prints installed versions expected by the project (`pandas`, `numpy`, `matplotlib`, `scikit-learn`, `scipy`).

### Generate all analysis figures and metrics (final version)
Run in `verFinal/`:

```bash
python code_vFinal.py
```

Notes:
- This script performs end-to-end analysis and writes outputs to `verFinal/outputs/`.
- `DATA_FILE` is hardcoded to an absolute path in `verFinal/code_vFinal.py`, so if you move the repo or run on another machine, update that constant first.

### Generate the flowchart figure only
Run in `verFinal/`:

```bash
python "flow_chart(CHN).py"
```

Writes `outputs/Fig12_Flowchart.png`.

### Build the final paper PDF
Run in `verFinal/` (XeLaTeX is recommended for `ctex`):

```bash
xelatex paper_B_vFinal.tex
bibtex paper_B_vFinal
xelatex paper_B_vFinal.tex
xelatex paper_B_vFinal.tex
```

If `latexmk` is installed, a one-command alternative is:

```bash
latexmk -xelatex paper_B_vFinal.tex
```

## Testing/linting status

- No `pytest`/unit-test suite is configured in this repository.
- No dedicated lint/format config (`pyproject.toml`, `ruff`, `flake8`, etc.) is present.
- Therefore, there is no “single test” command available; verification is script execution + output artifact checks.

## High-level architecture

### 1) Versioned workflow model
- Each `ver*` directory is a self-contained snapshot pairing:
  - one main Python analysis script (`code_ver1.py`, `code_ver2.py`, `code_ver3.py`, `code_vFinal.py`),
  - one LaTeX manuscript (`paper_B*.tex`),
  - generated artifacts (`outputs/*.png`, compiled `.pdf/.aux/.log`).
- The project evolved by refining methodology across versions, with `verFinal/` as the operational target for edits unless explicitly asked otherwise.

### 2) End-to-end pipeline in `verFinal/code_vFinal.py`
The script is monolithic and executes top-to-bottom to produce figures and printed metrics:
1. Load Excel dataset (`DFlow`, `DGap`, 16 pulse features).
2. Perform statistical/feature-structure analysis (correlations, clustering, PCA/LDA views).
3. Build ML models (Random Forest, SVM, k-NN, LDA, Logistic Regression) under `Pipeline` to avoid leakage.
4. Evaluate via stratified CV, confusion matrices, classifier-chain logic, and comparative plots.
5. Export a full figure set into `outputs/` (`Fig1`–`Fig14` plus comparison figures).

### 3) Manuscript coupling
- `verFinal/paper_B_vFinal.tex` references generated figures directly from `verFinal/outputs/`.
- Regenerating figures before LaTeX build is important to keep paper content and visuals aligned.

### 4) Separate flowchart renderer
- `verFinal/flow_chart(CHN).py` is an independent drawing script for the modeling flowchart figure (Chinese/English typography handling), instead of being fully embedded in the main analysis run.

## Practical editing guidance

- Treat `verFinal/` as the primary implementation target unless task scope explicitly requests historical versions.
- When adjusting modeling logic, check both:
  - `verFinal/code_vFinal.py` (analysis + figure generation)
  - `verFinal/paper_B_vFinal.tex` (narrative, reported metrics, figure references)
- Keep output filenames stable unless you also update LaTeX references.