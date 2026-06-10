# 🌿 SPADE

### Statistical Platform for Agronomic Data Evaluation

> An open-source, locally-running Python dashboard for standardised nitrogen use efficiency (NUE) analysis in controlled agricultural pot experiments.

---

[![Python](https://img.shields.io/badge/Python-3.9%2B-blue?logo=python&logoColor=white)](https://python.org)
[![Streamlit](https://img.shields.io/badge/Built%20with-Streamlit-FF4B4B?logo=streamlit&logoColor=white)](https://streamlit.io)
[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)
[![Platform](https://img.shields.io/badge/Platform-Windows%20%7C%20macOS%20%7C%20Linux-lightgrey)]()
[![Offline](https://img.shields.io/badge/Runs-Offline-success)]()

---

## What is SPADE?

SPADE addresses a gap in agricultural research tooling: no existing open-source tool integrates structured data entry, multi-index NUE computation, factorial ANOVA with correct compact letter display (CLD), outlier detection, and publication-ready figure and report export within a single accessible application.

SPADE was developed at the **Dhaka University Nanotechnology Centre (DUNTC)**, University of Dhaka, as part of an MSc research project comparing nano-urea versus conventional granular urea on wheat (*BARI Gom 33*) under a Completely Randomised Design (CRD).

---

## Key Features

### Statistical Analysis
- **One-Way ANOVA** with Tukey HSD post-hoc and compact letter display (CLD) using the Piepho (2004) maximal clique algorithm — the only correct method for non-monotonic significance patterns
- **Two-Way Factorial ANOVA** (N rate × foliar type) with Type III SS for unbalanced designs, interaction plots, and cell means
- **Effect sizes** (η² and ω²) reported alongside every ANOVA — addresses a systematic gap in agricultural publications
- **Kruskal-Wallis + Dunn's test** as a non-parametric fallback for n=3 or discrete parameters
- **Residual Q-Q plots** for normality assessment (more informative than Shapiro-Wilk at n=3)

### Nitrogen Use Efficiency
- **All five NUE indices** computed simultaneously from the same data entry:
  - PFP-N (Partial Factor Productivity)
  - AE-N (Agronomic Efficiency)
  - RE-N (Recovery Efficiency)
  - PE-N (Physiological Efficiency)
  - NHI (Nitrogen Harvest Index)
  - Grain protein (N% × 5.7, wheat-specific)
- **Nitrogen balance** (N applied − crop N uptake − ΔSoil N)
- **Multi-nutrient uptake** (N, P, K, S, Ca, B) from tissue concentrations

### Data Management
- **30 agronomic parameters** across Growth & Biometrics, Harvest & Quality, and Soil Analysis
- **Persistent Excel storage** with named sheets: Raw Data, Pre-Experiment Soil, NUE Indices, Bio Yield & HI, N Balance, Data Flags
- **Pre-experiment baseline soil** data entry and pre/post comparison
- **Editable treatment descriptions** — propagate to all tables, figures, and report
- **Data Quality Flags** audit trail (Verified / Suspect / Excluded)

### Outlier Detection
- **Dixon's Q test** (designed for n=3–10) with plain-language explanations
- **Grubbs' test** as secondary check
- **Full scan mode** across all 30 parameters simultaneously
- **Monte Carlo power analysis** — Dixon's Q has ~28% power for a 2σ outlier at n=3 (documented in UI)

### Visualisation & Export
- **Bar charts** with Okabe-Ito colourblind-safe palette, CLD letters, SD/SE error bars, significance brackets (*, **, ***)
- **Strip plots** showing individual replicate values (honest for n=3)
- **Scatter plots** with Pearson r and regression line
- **Radar charts** for multi-parameter, multi-treatment comparison
- **PNG and JPG export** at 2× resolution with content-aware table dimensions
- **Per-parameter CSV export** for easy sharing with collaborators

### Reporting
- **Word document (.docx) report** with formatted ANOVA tables, treatment means, NUE indices, N balance, and statistical notes — ready for thesis or manuscript submission
- **Significance summary prose** auto-generated per parameter — copy-paste ready sentences for your Results section
- **Plain-text report** covering all parameters with entered data

---

## Installation

### Requirements
- Python 3.9 or higher
- No internet connection required after installation

### Steps

**1. Clone the repository**
```bash
git clone https://github.com/sajjadurrahmanzadid-hue/SPADE.git
cd spade
```

**2. Install dependencies**
```bash
pip install -r requirements.txt
```

> **Note on kaleido (for PNG/JPG export):** On some systems `pip install kaleido` may fail. If so, try `pip install kaleido==0.2.1`. PNG/JPG export is optional — all other features work without it.

**3. Run SPADE**
```bash
streamlit run wheat_dashboard.py
```
### Run the validation suite

```bash
python validate_spade.py
```

This cross-checks all statistical routines against independent references.
Exit code 0 = all checks passed.

**4. Open in browser**

SPADE opens automatically at `http://localhost:8501`. If it does not, navigate there manually.

---

## Quick Start

### With sample data
A sample dataset is included in `sample_data/sample_wheat_data.xlsx`. Copy it to the same folder as `wheat_dashboard.py` and rename it `wheat_data.xlsx`, then run SPADE.

### With your own data
1. Run SPADE
2. Navigate to the **📋 Data Entry** tab
3. Enter your replicate values directly in the editable tables
4. Click **💾 Save** — data is stored in `wheat_data.xlsx` in the same folder
5. Proceed to any analysis tab

### Adapting to a different experiment
SPADE is designed for CRD experiments with up to 8 treatments and 3–6 replicates. To adapt it:
- Edit treatment descriptions directly in **Data Entry → Edit Treatment Descriptions**
- Enter your N rate per treatment via the T_INFO structure in `wheat_dashboard.py` (lines 55–90)
- Adjust the pot area in the sidebar for your N balance calculation

---

## Data Format

SPADE saves data to `wheat_data.xlsx` with the following sheets:

| Sheet | Contents |
|-------|----------|
| Raw Data | 24-row master dataset (8 treatments × 3 replicates, 30 parameters) |
| Pre-Experiment Soil | 3 composite baseline soil samples |
| NUE Indices | Computed five-index NUE per treatment |
| Bio Yield & HI | Biological yield and harvest index |
| N Balance | Nitrogen balance components per treatment |
| Data Flags | Audit trail for quality-flagged observations |

If you have existing data in a flat Excel file (one sheet, columns matching the parameter names), SPADE will load it automatically on first run and convert to the multi-sheet format on the next save.

---
## Repository Structure

| File | Description |
|------|-------------|
| `wheat_dashboard.py` | Main Streamlit application — run this |
| `spade_stats.py` | Pure statistical core (no Streamlit dependency) — importable and testable independently |
| `validate_spade.py` | Automated validation harness — 17 checks against independent references |
| `requirements.txt` | Python dependencies |
| `USER_GUIDE.md` | Detailed usage guide with tab-by-tab walkthrough |
| `CONTRIBUTING.md` | Contribution guidelines |
| `sample_data/` | Sample dataset for demonstration |

## Tabs Overview

| Tab | Contents |
|-----|----------|
| 📋 Data Entry | Editable parameter tables, treatment description editor, pre-experiment soil, data flags |
| 📊 One-Way ANOVA & Tukey | F-test, Tukey HSD, CLD, Q-Q plot, Kruskal-Wallis, Dunn's test, CSV export |
| 📊 Two-Way ANOVA | Factorial N rate × foliar type, Type III SS, interaction plot |
| ⚗️ NUE & R:S Ratio | Five NUE indices, grain protein, root:shoot ratio with ANOVA |
| 🌱 Nutrient Dynamics | Multi-nutrient uptake, biological yield, N balance, pre/post soil comparison |
| 🔍 Outlier Tests | Dixon's Q + Grubbs per parameter, full scan, manual flags |
| 📈 Figures | Bar chart, strip plot, scatter/correlation, radar chart — all exportable |
| 📄 Report Draft | .docx report download, significance summary prose, plain-text draft |

---

## Statistical Methods

| Method | Implementation |
|--------|---------------|
| Tukey HSD CLD | Piepho (2004) maximal clique algorithm |
| Two-way ANOVA | statsmodels OLS + anova_lm, Type II or III SS |
| Effect sizes | η² (eta-squared) and ω² (omega-squared, bias-corrected) |
| Outlier detection | Dixon's Q test + Grubbs' test |
| Non-parametric | Kruskal-Wallis + Dunn's test (Bonferroni correction) |
| Normality check | Residual Q-Q plot (16 df for 8×3 design) |
| NUE indices | Per Baligar et al. (2001) |

### Key references
- Piepho, H.P. (2004). *JCGS*, 13(2), 456–466 — CLD algorithm
- Langsrud, Ø. (2003). *Statistics and Computing*, 13, 163–167 — Type III SS
- Dixon, W.J. (1953). *Biometrics*, 9(1), 74–89 — outlier test
- Baligar, V.C. et al. (2001). *Comm. Soil Sci. Plant Anal.*, 32(7–8), 921–950 — NUE indices

---

## Case Study

SPADE was developed using a pot experiment evaluating nano-urea versus conventional granular urea on wheat (*BARI Gom 33*) at the Dhaka University Nanotechnology Centre:

- **Design:** CRD, 8 treatments × 3 replicates (24 pots, 10 kg soil each)
- **Treatments:** Factorial structure of N rate (0, 50, 75, 100% RDN = 0–175 kg N ha⁻¹) × foliar type (water spray, nano urea, granulated urea spray)
- **Funder:** Dhaka University Nanotechnology Centre (DUNTC), Bangladesh

---

## Citation

If you use SPADE in your research, please cite:

```
Rahman, S. (2025). SPADE: Statistical Platform for Agronomic Data Evaluation.
Dhaka University Nanotechnology Centre (DUNTC), University of Dhaka.
GitHub: https://github.com/sajjadurrahmanzadid-hue/SPADE.git
```

A methods paper describing SPADE's analytical framework is in preparation, targeting *Computers and Electronics in Agriculture*.

---

## Contributing

Contributions are welcome. See [CONTRIBUTING.md](CONTRIBUTING.md) for guidelines.

Planned improvements include RCBD support, mixed-effects models, dose-response curve fitting, and cloud deployment.

---

## License

MIT License — see [LICENSE](LICENSE) for details.

---

## Contact

**Sajjadur Rahman**  
MSc Candidate, Department of Soil, Water and Environmental Science  
University of Dhaka | Dhaka University Nanotechnology Centre (DUNTC)  
National Science and Technology Fellow, Ministry of Science and Technology, Bangladesh  
🌐 [sajjadur-rahman.com](https://sajjadur-rahman.com)
