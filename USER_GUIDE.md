# SPADE User Guide

**Statistical Platform for Agronomic Data Evaluation**  
Version 1.0 | Dhaka University Nanotechnology Centre (DUNTC)

---

## Contents

1. [Getting Started](#1-getting-started)
2. [Data Entry](#2-data-entry)
3. [One-Way ANOVA & Tukey HSD](#3-one-way-anova--tukey-hsd)
4. [Two-Way ANOVA](#4-two-way-anova)
5. [NUE Indices & R:S Ratio](#5-nue-indices--rs-ratio)
6. [Nutrient Dynamics](#6-nutrient-dynamics)
7. [Outlier Tests](#7-outlier-tests)
8. [Figures](#8-figures)
9. [Report Draft](#9-report-draft)
10. [Exporting Data](#10-exporting-data)
11. [Adapting SPADE to Your Experiment](#11-adapting-spade-to-your-experiment)
12. [Statistical Notes](#12-statistical-notes)
13. [Troubleshooting](#13-troubleshooting)

---

## 1. Getting Started

### Installation

```bash
pip install -r requirements.txt
streamlit run wheat_dashboard.py
```

SPADE opens at `http://localhost:8501`. Data is auto-saved to `wheat_data.xlsx` in the same folder.

### First run with existing data

If you have a flat Excel file (single sheet, columns matching SPADE parameter names), place it in the same folder as `wheat_dashboard.py` named `wheat_data.xlsx`. SPADE will load it automatically and convert to multi-sheet format on next save.

### Sidebar controls

| Control | Description |
|---------|-------------|
| Pot area (m²) | Used for N balance calculation. Default 0.04 m² (25 cm diameter pot). |
| 💾 Save | Writes all data to `wheat_data.xlsx` with named sheets |
| 🔄 Reload | Reloads data from disk (also reloads pre-experiment soil) |
| Data completeness | Progress bar showing filled cells per group |

---

## 2. Data Entry

### Parameter groups

SPADE manages **30 parameters** across three groups:

**Growth & Biometrics (12 parameters)**  
Plant height, tiller count, SPAD at tillering, SPAD at heading, LAI, stem diameter, spike length, root length, shoot fresh weight, root fresh weight, shoot dry weight, root dry weight

**Harvest & Quality (18 parameters)**  
Grains/spike, spikelets/spike, 1000-grain weight, grain yield, straw yield, grain N%, straw N%, grain B, straw B, grain P, straw P, grain K, straw K, grain S, straw S, grain Ca, straw Ca

**Soil Analysis (10 parameters)**  
pH, EC, organic carbon, total N, residual inorganic N, available P, exchangeable K, available S, exchangeable Ca, hot-water soluble B

> **Tip:** Hover over any column header to see a tooltip with the measurement method, units, and typical ranges for wheat.

### Editing treatment descriptions

Open **Edit Treatment Descriptions** to customise the label for each treatment. Changes propagate immediately to all tables, figures, tooltips, and the generated report.

### Pre-experiment baseline soil

Open **Pre-Experiment Baseline Soil** to enter the three composite soil samples taken before the experiment. Required for:
- Pre/post soil comparison table
- N balance calculation (ΔSoil N component)

### Data Quality Flags

Open **Data Quality Flags & Audit Trail** to flag individual replicate values. Choose a parameter group and parameter, then set each cell's status:

| Status | Meaning |
|--------|---------|
| ✓ Verified | Value confirmed against field/lab record |
| ⚠ Suspect — review | Unusual value that needs checking |
| ❌ Excluded from analysis | Value excluded; reason noted |

Flags are saved to the **Data Flags** sheet in `wheat_data.xlsx` and appear in the Outlier Tests tab.

---

## 3. One-Way ANOVA & Tukey HSD

### Workflow

1. Select a parameter from the dropdown
2. ANOVA runs automatically on all 8 treatment means
3. Review F-statistic, p-value, LSD(0.05), CV%, η², and ω²
4. Tukey HSD pairwise comparisons table shows which pairs are significantly different
5. Compact Letter Display (CLD) summarises groupings — treatments sharing a letter are not significantly different

### Effect size interpretation

| ω² value | Interpretation |
|----------|---------------|
| < 0.01 | Negligible |
| 0.01 – 0.06 | Small |
| 0.06 – 0.14 | Medium |
| ≥ 0.14 | Large |

### Q-Q plot

The residual Q-Q plot (below the ANOVA section) shows whether ANOVA residuals follow a normal distribution. Points near the dashed line indicate approximate normality. At n=3 per group, formal Shapiro-Wilk testing has very low power — visual inspection of the Q-Q plot is more informative.

### Kruskal-Wallis

If the Q-Q plot shows a clear departure from normality, or for discrete parameters (tiller count), use the Kruskal-Wallis non-parametric test at the bottom of the tab. Dunn's post-hoc test with Bonferroni correction runs automatically if Kruskal-Wallis is significant.

### CSV export

Below the CLD table, a download button exports the parameter's per-replicate data (R1, R2, R3), means, SD, SE, and CLD letters as a CSV file for sharing with collaborators.

---

## 4. Two-Way ANOVA

Decomposes the treatment effect into:
- **Factor A:** N rate (N0, N1, N2, N3 — low to high)
- **Factor B:** Foliar type (control, water spray, nano urea, granulated urea spray)

### Balanced vs Full mode

| Mode | Treatments used | When to use |
|------|----------------|-------------|
| Balanced (T2–T7) | 6 treatments in 3×2 factorial | Primary factorial analysis |
| Full (all 8) | All 8, unbalanced | Exploratory, Type III SS |

### Interaction

If the interaction (N rate × foliar type) is significant: interpret main effects cautiously and focus on the interaction plot and cell means table. A significant interaction means the effect of foliar type depends on the N rate applied.

---

## 5. NUE Indices & R:S Ratio

### NUE index requirements

| Index | Required inputs |
|-------|----------------|
| PFP-N | Grain yield + N applied |
| AE-N | Grain yield + T1 grain yield + N applied |
| RE-N | Grain yield + grain N% + straw yield + straw N% + N applied |
| PE-N | Same as RE-N |
| NHI | Grain yield + grain N% + straw yield + straw N% |
| Grain protein | Grain N% only |

SPADE computes whichever indices are calculable from entered data, flagging unavailable indices.

### R:S Ratio

Root:Shoot ratio uses dry weights (rootDW, shootDW) when available — the scientifically correct basis. Falls back to fresh weights if dry weights have not been entered, with an explicit note in the results table.

---

## 6. Nutrient Dynamics

### Multi-nutrient uptake

Uptake = yield × tissue concentration. Units:

| Nutrient | Unit |
|----------|------|
| N | g/pot |
| P, K, S, Ca | mg/pot |
| B | μg/pot |

Requires tissue concentrations (grainP, strawP, grainK, etc.) in the Data Entry tab.

### Biological yield and harvest index

HI = grain yield / (grain + straw + root dry weight) × 100

Using all three components is the correct formula. When rootDW is not entered, falls back to grain + straw only with an explicit note.

### Nitrogen balance

N balance = N applied (g/pot) − total crop N uptake − ΔSoil N

- **N applied:** N rate (kg/ha) × pot area (m²) set in sidebar
- **Crop N uptake:** grain N + straw N
- **ΔSoil N:** post-harvest total soil N − pre-experiment baseline soil N

Positive balance = unaccounted N (losses to atmosphere/leaching).

### Pre/post soil comparison

Requires pre-experiment baseline soil data entered in Data Entry. Shows pre-experiment mean, post-harvest treatment mean, and % change for all 10 soil parameters.

---

## 7. Outlier Tests

### Dixon's Q test (recommended for n=3–10)

Tests whether the lowest or highest replicate value is statistically unlikely. Critical value Q=0.970 at α=0.05 for n=3. The result includes a plain-language comment explaining which replicate is suspect, by how much, and in which direction.

### Grubbs' test

Tests whether the most extreme value deviates more than expected from the group mean. Better for n>6; included as a secondary check.

### ⚠️ Important caveat

At n=3, both tests have **very low statistical power**. Dixon's Q detects only ~28% of 2σ outliers and ~84% of 3σ outliers. A flagged value should always be verified against your field diary before exclusion — the test identifies candidates for review, not values to automatically remove.

### Full scan

The **Run full outlier scan** button tests all 30 parameters simultaneously and returns a summary table of every flagged value.

---

## 8. Figures

### Chart types

| Type | Best for |
|------|----------|
| Bar chart (mean ± SD/SE) | Standard publication figures |
| Strip plot | Showing actual replicates, honest for n=3 |
| Scatter / Correlation | Bivariate relationships between parameters |
| 🕸 Radar chart | Multi-parameter comparison across treatments |

### Significance brackets

In bar chart mode, choose:
- **None** — no brackets
- **vs Control (T1)** — brackets only for treatments significantly different from T1
- **All significant pairs** — all Tukey HSD significant pairs

### Radar chart normalisation

- **% of max treatment** — each parameter scaled to its highest treatment mean
- **% of T1 (control)** — each parameter scaled relative to the unfertilised control

---

## 9. Report Draft

### .docx report

Click **⬇ Download .docx report** to get a formatted Word document containing:
- Experiment details table
- Treatment descriptions
- ANOVA table + means table per parameter (all parameters with data)
- NUE indices table
- Multi-nutrient uptake table (if data available)
- N balance table (if data available)
- Field events documentation
- Statistical notes

### Significance summary

Select a parameter and the text area below shows two copy-paste sentences:  
*"[Parameter] was significantly affected by treatment (F(df₁,df₂) = x.xxx, p = 0.xxxx ***, ω² = 0.xxx [large effect]). The highest mean was recorded in T3 (x.xx ± x.xx SD), which was significantly superior to T1 (p = 0.001) and T6 (p = 0.023)."*

---

## 10. Exporting Data

| Export | Location | Format |
|--------|----------|--------|
| Full dataset | Sidebar → 💾 Save | .xlsx (6 sheets) |
| Per-parameter results | ANOVA tab → CSV button | .csv |
| Figure (PNG) | Figures tab | .png (2× resolution) |
| Figure (JPG) | Figures tab | .jpg (2× resolution) |
| ANOVA table | ANOVA tab → JPG button | .jpg |
| Treatment means table | ANOVA tab → JPG button | .jpg |
| Tukey HSD table | ANOVA tab → JPG button | .jpg |
| Results report | Report tab → .docx button | .docx |
| Data flags | Sidebar → Save | .xlsx (Data Flags sheet) |

---

## 11. Adapting SPADE to Your Experiment

SPADE is built for the DUNTC nano-urea wheat CRD experiment but can be adapted:

**Changing treatment descriptions:** Use the UI — no code editing required.

**Changing N rates:** Edit the TREATMENTS list in `wheat_dashboard.py` (lines 55–90). Each entry has an `"n"` key for the N rate in kg/ha.

**Different number of treatments or replicates:** Significant code changes are required. Currently fixed at 8 treatments × 3 replicates (24 rows).

**Different crop protein conversion factor:** Change `5.7` in the `nue_dataframe()` function to the appropriate crop-specific factor (e.g. 6.25 for a generic factor, 5.83 for wheat alternative, 6.38 for rice).

---

## 12. Statistical Notes

### Why maximal clique CLD?

The naive CLD algorithm (linear sweep) produces incorrect letter assignments for non-monotonic significance patterns — a situation common in agricultural experiments where intermediate doses overlap in significance with both high and low doses. The Piepho (2004) maximal clique method is mathematically correct for all cases. SPADE's implementation is equivalent to R's `multcomp::cld()`.

### Type III vs Type II SS

SPADE uses Type III SS by default in the two-way ANOVA (when interaction is explicitly tested). For datasets where no interaction is expected, Type II SS is more appropriate per Langsrud (2003) and is available by selecting "Balanced" mode and noting the small NS interaction.

### Effect sizes vs p-values

At n=3 per treatment, a non-significant ANOVA (p>0.05) can coincide with a medium or large effect size (ω²≥0.06) simply because statistical power is low. Report both — the p-value and the ω² — to give readers a complete picture.

---

## 13. Troubleshooting

| Problem | Solution |
|---------|----------|
| `kaleido` not found | `pip install kaleido==0.2.1`. PNG/JPG export is optional. |
| Data not saving | Ensure `wheat_dashboard.py` folder is writable |
| `KeyError: 'shootDW'` | Restart Streamlit fully (Ctrl+C, then rerun) to clear stale session cache |
| Double-input in tables | This is fixed in SPADE v1.0. If it recurs, reload the page |
| Q-Q plot empty | Enter data for at least 4 observations total |
| Radar chart empty | Select ≥3 parameters and ≥2 treatments |
| `python-docx` error in .docx report | `pip install python-docx` |

---

*SPADE is developed at the Dhaka University Nanotechnology Centre (DUNTC), University of Dhaka, Bangladesh.*
