# Extreme Heat & Regional Mortality in Italy (2012–2022)

**How has extreme heat affected regional mortality in Italy, and did the 2022 heatwave produce disproportionately higher impacts in socially vulnerable regions?**

---

## Overview

This project investigates the relationship between extreme heat exposure and regional mortality across Italy's 20 NUTS-2 regions from 2012 to 2022. It focuses on the severe 2022 European heatwave and tests whether socially vulnerable regions experienced disproportionately higher heat-related mortality.

The analysis centers on three hypotheses:

| ID | Hypothesis | Test |
|---|---|---|
| **H1** | Extreme heat is associated with significant increases in regional mortality | Panel regression coefficient on heat exposure |
| **H2** | Higher social vulnerability amplifies heat-mortality impacts | Interaction term: Heat × RSVI |
| **H3** | The 2022 heatwave widened inequality between high- and low-vulnerability regions | Three-way interaction: Heat × RSVI × Year2022 |

---

## Project Structure

```
Project/
├── README.md
├── pyproject.toml                  # Python dependencies (managed by uv)
├── Makefile                        # Pipeline entry points
│
├── config/
│   └── config.yaml                 # All project parameters
│
├── src/
│   ├── __init__.py
│   │
│   ├── data/                       # Data acquisition & processing
│   │   ├── __init__.py
│   │   ├── download_era5.py        # ERA5-Land data via CDS API
│   │   ├── process_mortality.py    # ISTAT/Eurostat mortality processing
│   │   ├── process_climate.py      # Spatial aggregation of gridded climate
│   │   ├── process_socioeconomic.py# Socioeconomic indicator processing
│   │   └── nuts2_boundaries.py     # NUTS-2 geometry utilities
│   │
│   ├── features/                   # Feature engineering
│   │   ├── __init__.py
│   │   ├── heatwave.py             # Heatwave detection & metrics
│   │   ├── temperature.py          # Apparent temp, WBGT, anomalies
│   │   ├── rsvi.py                 # Regional Social Vulnerability Index
│   │   └── mortality_rates.py      # Age-standardized mortality rates
│   │
│   ├── analysis/                   # Statistical analysis
│   │   ├── __init__.py
│   │   ├── panel_dataset.py        # Assemble the merged panel
│   │   ├── eda.py                  # Exploratory data analysis
│   │   ├── panel_regression.py     # Fixed-effects panel models (H1–H3)
│   │   └── diagnostics.py          # Model diagnostics & robustness checks
│   │
│   ├── visualization/              # Plots & maps
│   │   ├── __init__.py
│   │   ├── maps.py                 # Choropleth & spatial maps
│   │   ├── timeseries.py           # Time series & trend plots
│   │   ├── case_studies.py         # Paired regional comparison plots
│   │   └── regression_plots.py     # Coefficient plots, interaction effects
│   │
│   └── utils/                      # Shared utilities
│       ├── __init__.py
│       ├── config.py               # Configuration loader
│       ├── constants.py            # Project-wide constants
│       └── io.py                   # Data I/O helpers
│
├── scripts/                        # CLI entry points
│   ├── run_pipeline.py             # Full pipeline orchestrator
│   ├── download_data.py            # Data download script
│   └── run_analysis.py             # Analysis-only script
│
├── notebooks/                      # Exploratory Jupyter notebooks
│   └── .gitkeep
│
├── data/
│   ├── raw/                        # Untouched downloaded data
│   │   ├── mortality/
│   │   ├── climate/
│   │   ├── socioeconomic/
│   │   └── boundaries/
│   ├── interim/                    # Intermediate processed data
│   └── processed/                  # Final analysis-ready datasets
│
├── outputs/
│   ├── figures/                    # Generated plots & maps
│   ├── tables/                     # Regression tables, summary stats
│   └── reports/                    # Generated reports
│
└── tests/                          # Unit tests
    ├── __init__.py
    ├── test_heatwave.py
    ├── test_rsvi.py
    └── test_panel_regression.py
```

---

## Data Sources

### Mortality (Dependent Variable)
| Source | Variables | Resolution |
|---|---|---|
| **ISTAT Daily Mortality** | All-cause deaths by age, sex | Municipal → NUTS-2, daily |
| Eurostat NUTS-2 | Weekly/annual deaths by age | NUTS-2 regional |
| SiSMG / ISS | Summer excess mortality (%) | 34 cities |

### Temperature / Heat Exposure (Independent Variable)
| Source | Variables | Resolution |
|---|---|---|
| **ERA5-Land (Copernicus)** | Daily T_max, T_mean, T_min, dewpoint | ~9 km grid |
| E-OBS (ECA&D) | Daily T_max, T_min, T_mean | ~10 km (robustness check) |

### Social Vulnerability (Moderators)
| Domain | Source | Variables |
|---|---|---|
| Age structure | ISTAT I.Stat | % population aged 65+, 75+, 80+ |
| Income / deprivation | ISTAT, Eurostat | Poverty rate, GDP per capita |
| Urban density | ISTAT territory | Population density, urbanization rate |

---

## Methodology

### Heatwave Definition
A heatwave day occurs when daily T_max exceeds the **region-specific 90th percentile** of June–September T_max over the 2012–2022 baseline, **for ≥3 consecutive days**.

### Regional Social Vulnerability Index (RSVI)
Constructed following the CDC/ATSDR SVI methodology:
1. Percentile-rank each indicator across regions (per year)
2. Group into thematic sub-indices (Demographic, Economic, Urban)
3. Average sub-index percentiles → composite RSVI score (0–1)

### Fixed-Effects Panel Regression

**Primary model (H1 + H2):**
```
Mortality_rt = β₁·Heat_rt + β₂·(Heat_rt × RSVI_rt) + β₃·COVID_rt + αᵣ + γₜ + εᵣₜ
```

**Extended model (H3):**
```
Mortality_rt = β₁·Heat_rt + β₂·(Heat_rt × RSVI_rt) + β₃·(Heat_rt × RSVI_rt × D2022) + controls + αᵣ + γₜ + εᵣₜ
```

Where `αᵣ` = region fixed effects, `γₜ` = year fixed effects.

### Case Study Comparisons
| Pair | High Vulnerability | Low Vulnerability | Rationale |
|---|---|---|---|
| Economic divide | Calabria | Lombardia | North-south gradient |
| Urban vs. rural | Lazio | Trentino-Alto Adige | Urban heat island effect |
| Climate control | Puglia | Emilia-Romagna | Similar heat, different economy |

---

## Setup

### Prerequisites
- Python 3.11+
- [uv](https://docs.astral.sh/uv/) package manager
- CDS API key (for ERA5-Land data) — [register here](https://cds.climate.copernicus.eu/)

### Installation
```bash
# Clone & enter the project
cd Project/

# Install dependencies with uv
uv sync

# Verify installation
uv run python -c "import src; print('Setup complete')"
```

### Configuration
Edit `config/config.yaml` to set:
- Study period (default: 2012–2022)
- Heatwave threshold percentile (default: 90th)
- NUTS-2 region list
- Data paths
- CDS API credentials

---

## Usage

### Full Pipeline
```bash
# Run the complete pipeline (download → process → analyze → visualize)
uv run python scripts/run_pipeline.py

# Or using Make targets
make download    # Download raw data
make process     # Process & engineer features
make analyze     # Run panel regressions
make visualize   # Generate all figures & tables
make all         # Run everything
```

### Individual Steps
```bash
# Download ERA5-Land data
uv run python scripts/download_data.py --source era5

# Run analysis only (assumes processed data exists)
uv run python scripts/run_analysis.py

# Run specific analysis
uv run python -m src.analysis.eda
uv run python -m src.analysis.panel_regression
```

---

## Key Outputs

| Output | Location | Description |
|---|---|---|
| Panel dataset | `data/processed/panel_dataset.csv` | Merged region-year dataset |
| Summary statistics | `outputs/tables/summary_stats.csv` | Descriptive stats by region |
| Regression results | `outputs/tables/regression_results.csv` | H1, H2, H3 model estimates |
| Mortality maps | `outputs/figures/mortality_map_*.png` | Choropleth maps by year |
| RSVI maps | `outputs/figures/rsvi_map_*.png` | Vulnerability index maps |
| Heatwave maps | `outputs/figures/heatwave_map_*.png` | Heatwave exposure maps |
| Case study plots | `outputs/figures/case_study_*.png` | Paired regional comparisons |
| Coefficient plots | `outputs/figures/coeff_plot_*.png` | Regression coefficient visualizations |

---

## License

This project is for academic purposes (CLMT5202, Columbia University).
