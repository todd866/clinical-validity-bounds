# The Dimensional Validity Bound

**A Structural Limit on Clinical AI in Multimorbidity**

> **Target Journal:** JAMIA (Journal of the American Medical Informatics Association)
> **Status:** Preparing for submission
> **Data:** MIMIC-IV (N=425,216)

## Overview

This repository demonstrates that patient complexity imposes fundamental validity bounds on clinical AI systems. When patient state dimensionality exceeds algorithmic tractability, no amount of additional data or model refinement can restore validity.

### Key Findings (MIMIC-IV)

| Multimorbidity | N | D_eff | AUC |
|----------------|---|-------|-----|
| Low | 159,211 | 44.3 | 0.867 |
| **Moderate** | **128,979** | **37.5** | **0.835** |
| High | 137,026 | 29.5 | 0.859 |

- **Loss of complexity:** D_eff *decreases* with multimorbidity (Cohen et al. 2022 complexity hypothesis)
- **U-shaped AUC:** Worst performance in *moderate* stratum—the "zone of maximum entropy"
- **Diagnostic cascades:** P(false positive) exceeds 50% with >14 tests at 5% FPR
- **Dimensional validity bound:** Coupling fails when r = D_obs/D_pat < 0.3

### Theoretical Contribution

The dimensional validity bound is motivated by Fano's inequality: when observer capacity C < patient state entropy H(X), inference error has a non-vanishing lower bound. The r ≈ 0.3 threshold is simulation-supported.

## Repository Structure

```
├── manuscript.tex/pdf      # JAMIA paper
├── cover_letter.tex/pdf    # Submission cover letter
├── references.bib          # Bibliography
├── code/
│   ├── mimic_analysis_local.py      # Main MIMIC-IV analysis
│   ├── simulations.py               # Simulation studies (Figs 1-2)
│   └── create_validity_bound_figure.py  # Figure 3
├── data/mimic/             # MIMIC-IV CSVs (PhysioNet access required)
├── figures/                # Manuscript figures
├── results/                # Analysis outputs
└── archive/                # Old code/figures
```

## Reproducing Results

### MIMIC-IV Analysis

```bash
cd code
python mimic_analysis_local.py
```

Outputs: `results/classifier_results.csv`, `results/patient_analysis.csv`

### Simulations & Figures

```bash
python simulations.py                    # Figures 1-2
python create_validity_bound_figure.py   # Figure 3
```

## Reproducibility Notes for Reviewers

| Manuscript Section | Script | Output |
|--------------------|--------|--------|
| §4.1 Diagnostic Cascades | `simulations.py` | Fig 1, `diagnostic_cascade.csv` |
| §4.2 Dimensional Matching | `simulations.py` | Fig 2, `dimensional_matching.csv` |
| §4.5 MIMIC-IV Validation | `mimic_analysis_local.py` | Tables 2-3, `classifier_results.csv` |
| Fig 3 (Validity Bound) | `create_validity_bound_figure.py` | `fig3_validity_bound.png` |

## Requirements

Python 3.10+, NumPy, pandas, scikit-learn, matplotlib

## Key Concepts

### Effective Dimensionality (D_eff)

Participation ratio measuring dimensional spread:

```
D_eff = (Σλᵢ)² / Σλᵢ²
```

### The "Zone of Maximum Entropy"

- **Low multimorbidity:** High D_eff, strong healthy-state priors → Bayesian works
- **High multimorbidity:** Low D_eff, collapsed to disease attractors → Bayesian works (tragically)
- **Moderate multimorbidity:** Transitional zone, flat priors, noisy likelihoods → posterior instability

### Three Failure Modes

1. **Projection Error:** Lossy compression loses clinical distinctions
2. **Noise Amplification:** Unmeasured dimensions increase prediction variance
3. **Hypothesis Space Explosion:** Exponential growth defeats prior specification

## Citation

```bibtex
@article{todd2025dimensional,
  author  = {Todd, Ian},
  title   = {The Dimensional Validity Bound: Structural Limits of
             Clinical AI Evaluation in Multimorbidity},
  journal = {Journal of the American Medical Informatics Association},
  year    = {2025},
  note    = {Under review}
}
```

## Related Work

- Todd I. (2025). The limits of falsifiability. *BioSystems*, 258, 105608.
- Berisha V. et al. (2021). Digital medicine and the curse of dimensionality. *npj Digital Medicine*, 4, 153.
- Cohen A.A. et al. (2022). A complex systems approach to aging biology. *Nature Aging*, 2, 580-591.

## License

MIT License

## Contact

Ian Todd - itod2305@uni.sydney.edu.au
