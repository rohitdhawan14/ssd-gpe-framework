# Software Sustainability Debt and the GPE Framework

[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)

> **Reproducibility artifact for:**  
> *Software Sustainability Debt and the Governance-Process-Execution Framework: Evidence from 513,000 Real-World CI/CD Workflows*  
> Rohit Dhawan, Minav Suresh Patel, Ankush Dhar  
> IEEE Sustainable Technology Conference (Sustech) 2026

This repository provides the complete pipeline for reproducing all empirical results, tables, and figures from our paper analyzing software sustainability debt through 513,000 GitHub Actions workflows across three European electricity grids (GB, DE, FR).

---

## 📋 Table of Contents

- [Quick Start](#-quick-start)
- [Repository Structure](#-repository-structure)
- [Installation](#-installation)
- [Data Requirements](#-data-requirements)
- [Reproduction Steps](#-reproduction-steps)
  - [Step 1: Preprocessing](#step-1-preprocessing)
  - [Step 2: Base Simulation](#step-2-base-simulation)
  - [Step 3: Pairwise Portfolios](#step-3-pairwise-portfolios)
  - [Step 4: Sensitivity Analysis](#step-4-sensitivity-analysis)
  - [Step 5: Generate Plots](#step-5-generate-plots)
- [Output Reference](#-output-reference)
- [Parameters Reference](#-parameters-reference)
- [Citation](#-citation)
- [License](#-license)
- [Contact](#-contact)

---

## 🚀 Quick Start

```bash
# Clone repository
git clone https://github.com/YOUR-USERNAME/ssd-gpe-framework.git
cd ssd-gpe-framework

# Install dependencies
pip install -r requirements.txt

```

**Expected runtime:** ~30 minutes on a modern laptop (depends on dataset size)

---

## 📁 Repository Structure

```
ssd-gpe-framework/
├── README.md                          # This file
├── requirements.txt                   # Python dependencies
├── LICENSE                            # MIT License
│
├── scripts/
│   ├── preprocess_gha_multi.py       # Step 1: Data preprocessing
│   ├── simulate_case_studyN_regions.py # Step 2: Main simulator
│   ├── run_pairwise_portfolios.py    # Step 3: Pairwise analysis
│   ├── sweep_tri_region_sensitivity.py # Step 4: Parameter sweeps
│   └── plot_sensitivity.py           # Step 5: Plotting
│
├── data/                              # Place raw data here (not in repo)
│   ├── runs.json.gz                  # GitHub Actions logs (see Data Requirements)
│   └── raw_entsoe/                   # ENTSO-E data (see Data Requirements)
│       ├── entsoe_GB_raw.csv
│       ├── entsoe_DE_raw.csv
│       └── entsoe_FR_raw.csv
│
├── outputs/                           # Generated outputs
│   ├── prep_out/                     # Preprocessed hourly CSVs
│   ├── out_case/                     # Base simulation results
│   ├── out_pairs/                    # Pairwise portfolio results
|   ├── out_sens_tri/                 # sensitivity analysis + plots
│
└── docs/
    ├── PARAMETERS.md                  # Detailed parameter documentation
    └── OUTPUTS.md                     # Output file format reference
```

---

## 🔧 Installation

### Requirements

- **Python:** 3.10 or higher
- **Operating System:** Linux, macOS, or Windows with WSL
- **Memory:** 4GB RAM minimum (8GB recommended)
- **Disk Space:** ~5GB for data and outputs

### Install Dependencies

```bash
pip install -r requirements.txt
```

**Dependencies:**
- `numpy>=1.26.4` - Numerical computations
- `pandas>=2.2.2` - Data manipulation
- `matplotlib>=3.8.4` - Plotting
- `python-dateutil>=2.9.0` - Date/time handling
- `pytz>=2024.1` - Timezone conversions
- `tqdm>=4.66.4` - Progress bars

### Verify Installation

```bash
python3 -c "import pandas, numpy, matplotlib; print('✓ All dependencies installed')"
```

---

## 📊 Data Requirements

This artifact requires two data sources that are **not included** in this repository due to size and licensing:

### 1. GitHub Actions Workflow Logs

**What you need:** `runs.json.gz` (~2GB compressed)

**Source:** [GitHub Actions 2023 Dataset (D2K Lab)](https://github.com/D2KLab/gha-dataset)

**Download:**
```bash
# Option A: Direct download
wget https://github.com/D2KLab/gha-dataset/releases/download/v2023/runs.json.gz -O data/runs.json.gz

# Option B: Using gh CLI
gh release download v2023 --repo D2KLab/gha-dataset --pattern "runs.json.gz" --dir data/
```

**Format:** JSONL (newline-delimited JSON), one workflow run per line

**Required fields:**
- `created_at`: ISO 8601 timestamp
- `run_attempt`: Integer (we use first attempts only)
- `event`: Trigger type (push, pull_request, etc.)

### 2. ENTSO-E Grid Data

**What you need:** Hourly generation mix or carbon intensity for GB, DE, FR (2023)

**Source:** [ENTSO-E Transparency Platform](https://transparency.entsoe.eu/)

**Data type:** Either:
- **(A) Pre-computed CI:** Columns `hour_utc, ci_kg_per_kwh`
- **(B) Generation mix:** Columns `MTU, [technology]_MW` (script will compute CI)

**How to obtain:**
1. Visit [ENTSO-E Transparency Platform](https://transparency.entsoe.eu/)
2. Navigate to: Generation → Actual Generation per Production Type
3. Select regions: GB (Great Britain), DE (Germany), FR (France)
4. Date range: 2023-01-01 to 2023-12-31
5. Download as CSV
6. Place in `data/raw_entsoe/`

**Alternative:** Pre-processed files available upon request (see Contact section)

---

## 🔄 Reproduction Steps

### Step 1: Preprocessing

**Script:** `preprocess_gha_multi.py`

Converts raw GitHub Actions logs and ENTSO-E data into aligned hourly time series.

```bash
python3 scripts/preprocess_gha_multi.py \
  --runs-json data/runs.json.gz \
  --region GB data/raw_entsoe/entsoe_GB_raw.csv \
  --region DE data/raw_entsoe/entsoe_DE_raw.csv \
  --region FR data/raw_entsoe/entsoe_FR_raw.csv \
  --year 2023 \
  --first-attempt-only \
  --outdir outputs/prep_out
```

**Outputs:**
- `outputs/prep_out/gha_runs_2023_hourly.csv` - Hourly workflow counts
- `outputs/prep_out/entsoe_GB_2023_hourly_ci.csv` - GB carbon intensity
- `outputs/prep_out/entsoe_DE_2023_hourly_ci.csv` - DE carbon intensity
- `outputs/prep_out/entsoe_FR_2023_hourly_ci.csv` - FR carbon intensity

**Runtime:** ~5 minutes

---

### Step 2: Base Simulation

**Script:** `simulate_case_studyN_regions.py`

Runs the core GPE simulation across all scenarios (Unmanaged, Opportunistic, GPE, Geo-GPE).

```bash
python3 scripts/simulate_case_studyN_regions.py \
  --pushes outputs/prep_out/gha_runs_2023_hourly.csv \
  --region GB outputs/prep_out/entsoe_GB_2023_hourly_ci.csv \
  --region DE outputs/prep_out/entsoe_DE_2023_hourly_ci.csv \
  --region FR outputs/prep_out/entsoe_FR_2023_hourly_ci.csv \
  --outdir outputs/out_case \
  --window-hours 6 \
  --phi 0.5 \
  --cap-mult 2.0 \
  --geo-tau 0.15 \
  --opportunistic-phi 0.25 \
  --energy-per-push-kwh 0.1
```

**Key Parameters:**
- `--window-hours 6`: Maximum 6-hour deferral window (D)
- `--phi 0.5`: Up to 50% of workloads deferrable (φ)
- `--cap-mult 2.0`: Capacity capped at 2× median throughput (κ)
- `--geo-tau 0.15`: Minimum 15% CI improvement for geo-migration (τ)

**Outputs:**
- `outputs/out_case/summary_caseN.csv` - Normalized results (Table II & III from paper)
- `outputs/out_case/raw_totals_caseN.csv` - Absolute emissions

**Runtime:** ~2 minutes

**What this produces:**
- Single-region results (GB, DE, FR) for Table II
- Tri-region portfolio results for Table III

---

### Step 3: Pairwise Portfolios

**Script:** `run_pairwise_portfolios.py`

Generates all two-region combinations (GB-DE, GB-FR, DE-FR).

```bash
python3 scripts/run_pairwise_portfolios.py \
  --sim-path scripts/simulate_case_studyN_regions.py \
  --pushes outputs/prep_out/gha_runs_2023_hourly.csv \
  --region GB outputs/prep_out/entsoe_GB_2023_hourly_ci.csv \
  --region DE outputs/prep_out/entsoe_DE_2023_hourly_ci.csv \
  --region FR outputs/prep_out/entsoe_FR_2023_hourly_ci.csv \
  --outdir outputs/out_pairs \
  --window-hours 6 --phi 0.5 --cap-mult 2.0 --geo-tau 0.15
```

**Outputs:**
```
outputs/out_pairs/
├── GB_DE/summary_caseN.csv
├── GB_FR/summary_caseN.csv
└── DE_FR/summary_caseN.csv
```

**Runtime:** ~5 minutes

**What this produces:**
- Two-region portfolio results for Table III

---

### Step 4: Sensitivity Analysis

**Script:** `sweep_tri_region_sensitivity.py`

Performs parameter sweeps for all four GPE guardrails (D, φ, κ, τ).

```bash
python3 scripts/sweep_tri_region_sensitivity.py \
  --pushes outputs/prep_out/gha_runs_2023_hourly.csv \
  --region GB outputs/prep_out/entsoe_GB_2023_hourly_ci.csv \
  --region DE outputs/prep_out/entsoe_DE_2023_hourly_ci.csv \
  --region FR outputs/prep_out/entsoe_FR_2023_hourly_ci.csv \
  --outdir outputs/out_sens_tri \
  --sim-path scripts/simulate_case_studyN_regions.py
```

**Parameter Ranges:**
- **D (deferral window):** 2, 4, 6, 8, 10, 12 hours
- **φ (deferrable share):** 0.3, 0.4, 0.5, 0.6
- **κ (capacity cap):** 1.5, 2.0, 2.5
- **τ (geo threshold):** 0.10, 0.15, 0.20

**Outputs:**
```
outputs/out_sens_tri/
├── d_2.csv, d_4.csv, ..., d_12.csv
├── phi_0.3.csv, phi_0.4.csv, ..., phi_0.6.csv
├── cap_1.5.csv, cap_2.0.csv, cap_2.5.csv
└── tau_0.10.csv, tau_0.15.csv, tau_0.20.csv
```

**Runtime:** ~15 minutes (4 sweeps × multiple parameter values)

**What this produces:**
- Data for Figure 1 (sensitivity analysis)

---

### Step 5: Generate Plots

**Script:** `plot_sensitivity.py`

Creates publication-quality sensitivity plots from sweep results.

```bash
# Generate plots for GPE (policy-constrained) scenario
python3 scripts/plot_sensitivity.py outputs/out_sens_tri --scenario pol

# Alternative scenarios:
python3 scripts/plot_sensitivity.py outputs/out_sens_tri --scenario geo  # Geo-GPE
python3 scripts/plot_sensitivity.py outputs/out_sens_tri --scenario opt  # Opportunistic
```

**Outputs:**
```
outputs/out_sens_tri/
├── sens_window_hours.pdf    # D sensitivity (Figure 1a)
├── sens_phi.pdf             # φ sensitivity (Figure 1b)
├── sens_cap_mult.pdf        # κ sensitivity (Figure 1c)
└── sens_geo_tau.pdf         # τ sensitivity (Figure 1d)
```

**Runtime:** <1 minute

**What this produces:**
- Figure 1 from paper (parameter sensitivity analysis)

---

## 📈 Output Reference

### Key Output Files

#### `summary_caseN.csv`
Main results file with normalized emissions and reductions.

| Column | Description | Example |
|--------|-------------|---------|
| `setting` | Scenario name | `Unmanaged`, `GPE`, `Geo-GPE` |
| `region` | Region or portfolio | `GB`, `DE`, `FR`, `Portfolio` |
| `emissions_kg` | Absolute CO₂e emissions (kg) | 15626.3 |
| `index_normalized` | Normalized index (Unmanaged=100) | 98.97 |
| `reduction_vs_unmanaged_%` | Percentage reduction | 1.03 |

**Example row:**
```csv
setting,region,emissions_kg,index_normalized,reduction_vs_unmanaged_%
GPE,GB,15626.3,98.97,1.03
Geo-GPE,Portfolio,8751.2,67.6,32.4
```

#### Interpretation Guide

**Single-Region Results (Table II):**
- **Unmanaged (100.0):** Baseline - no sustainability governance
- **Opportunistic (94-96):** Informal shifting to cleaner hours (~4-6% reduction)
- **GPE (99.0-99.1):** Governed scheduling with SLA guarantees (~1% reduction)

**Cross-Region Portfolio Results (Table III):**
- **GB-DE:** 10.7% governed reduction
- **GB-FR:** 29.9% governed reduction
- **DE-FR:** 30.6% governed reduction
- **GB-DE-FR:** 32.4% governed reduction

---

## ⚙️ Parameters Reference

### GPE Guardrail Parameters

| Symbol | Flag | Description | Paper Value | Range |
|:------:|------|-------------|:-----------:|-------|
| **φ** | `--phi` | Deferrable workload fraction | 0.5 | 0.0-1.0 |
| **D** | `--window-hours` | Maximum deferral window (hours) | 6 | 1-24 |
| **κ** | `--cap-mult` | Capacity cap (× median throughput) | 2.0 | 1.0-5.0 |
| **τ** | `--geo-tau` | Geographic threshold (minimum CI improvement) | 0.15 | 0.0-1.0 |

### Additional Parameters

| Flag | Description | Default | Notes |
|------|-------------|:-------:|-------|
| `--energy-per-push-kwh` | Energy per workflow run (kWh) | 0.1 | Based on prior studies [3, 14] |
| `--opportunistic-phi` | Opportunistic deferral fraction | 0.25 | For baseline comparison |
| `--first-attempt-only` | Filter to first workflow attempts | - | Recommended for GHA data |

**Note:** Normalized results (Unmanaged=100) are independent of `--energy-per-push-kwh`. The absolute value affects total emissions but not relative comparisons.

---

## 🎯 Validating Results

### Expected Outcomes

After running the full pipeline, verify your results match the paper:

**Table II (Single-Region, GPE scenario):**
- GB: ~99.0 normalized (1.0% reduction)
- DE: ~99.1 normalized (0.9% reduction)
- FR: ~99.0 normalized (1.0% reduction)

**Table III (Portfolio, Geo-GPE scenario):**
- GB-DE-FR: ~67.6 normalized (32.4% reduction)

**Figure 1 (Sensitivity):**
- Diminishing returns beyond D≈6h, φ≈0.5
- Capacity cap κ has moderate impact
- Geographic threshold τ strongly affects cross-region gains

**Tolerance:** ±0.5% due to floating-point precision and minor implementation differences

### Troubleshooting

**Common Issues:**

1. **"No data for 2023" error**
   - Verify date ranges in ENTSO-E data
   - Check `--year 2023` flag in preprocessing

2. **"Mismatched hours" warning**
   - Normal - indicates missing data hours
   - Should be <1% of total hours for valid results

3. **"No CSV found" in sensitivity plots**
   - Ensure sweep script completed successfully
   - Check folder naming: `d_6`, `phi_0.5` (underscore, not dash)

4. **Large memory usage**
   - Expected with 513K workflows × 8760 hours
   - Reduce `--year` range if needed for testing

---

**Related Publications:**
- Dataset documentation: [D2K Lab GitHub Actions 2023](https://github.com/D2KLab/gha-dataset)

---

**Data licenses:**
- GitHub Actions data: [D2K Lab terms](https://github.com/D2KLab/gha-dataset#license)
- ENTSO-E data: [ENTSO-E Terms of Service](https://transparency.entsoe.eu/tos/)

---

## 👥 Contact

**Authors:**
- **Rohit Dhawan** - rohitdhwn14@ieee.org

**Questions about:**
- **Paper:** Email author
- **Reproducibility:** Open a [GitHub Issue](https://github.com/YOUR-USERNAME/ssd-gpe-framework/issues)
- **Data access:** See Data Requirements section or contact authors
- **Collaboration:** Welcome! Please reach out via email

---

## 🙏 Acknowledgments

We thank:
- **ENTSO-E** for providing public grid transparency data
- **D2K Lab** for curating the GitHub Actions 2023 dataset

---