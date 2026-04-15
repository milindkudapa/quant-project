# Walkthrough — Project Setup

## What was built

A complete Python project for analyzing extreme heat and regional mortality in Italy (2012–2022), structured as a modular data science pipeline managed by `uv`.

## Project Structure (25+ files)

### Core Setup
| File | Purpose |
|---|---|
| [README.md](file:///Users/milind/Desktop/Academics/Semester%202/CLMT5202/Project/README.md) | Comprehensive project documentation (no member names per request) |
| [pyproject.toml](file:///Users/milind/Desktop/Academics/Semester%202/CLMT5202/Project/pyproject.toml) | 30+ dependencies managed by uv (pandas, xarray, geopandas, linearmodels, etc.) |
| [config.yaml](file:///Users/milind/Desktop/Academics/Semester%202/CLMT5202/Project/config/config.yaml) | Central config: study period, heatwave thresholds, RSVI sub-indices, regression specs, NUTS-2 codes |
| [Makefile](file:///Users/milind/Desktop/Academics/Semester%202/CLMT5202/Project/Makefile) | Make targets for each pipeline stage |

### Data Layer (`src/data/`)
| Module | Purpose |
|---|---|
| [download_era5.py](file:///Users/milind/Desktop/Academics/Semester%202/CLMT5202/Project/src/data/download_era5.py) | CDS API download of ERA5-Land (T2m, dewpoint) for Italy bounding box |
| [process_climate.py](file:///Users/milind/Desktop/Academics/Semester%202/CLMT5202/Project/src/data/process_climate.py) | Spatial aggregation of gridded data → NUTS-2 daily Tmax/Tmin/Tmean |
| [process_mortality.py](file:///Users/milind/Desktop/Academics/Semester%202/CLMT5202/Project/src/data/process_mortality.py) | ISTAT/Eurostat mortality processing → NUTS-2 monthly/seasonal |
| [process_socioeconomic.py](file:///Users/milind/Desktop/Academics/Semester%202/CLMT5202/Project/src/data/process_socioeconomic.py) | Age structure, economic, urban density indicator processing |
| [nuts2_boundaries.py](file:///Users/milind/Desktop/Academics/Semester%202/CLMT5202/Project/src/data/nuts2_boundaries.py) | Download & load Eurostat NUTS-2 shapefiles for Italy |

### Feature Engineering (`src/features/`)
| Module | Purpose |
|---|---|
| [heatwave.py](file:///Users/milind/Desktop/Academics/Semester%202/CLMT5202/Project/src/features/heatwave.py) | 90th percentile thresholds, ≥3 consecutive day detection, seasonal metrics |
| [temperature.py](file:///Users/milind/Desktop/Academics/Semester%202/CLMT5202/Project/src/features/temperature.py) | Relative humidity, apparent temperature (heat index), WBGT approximation |
| [rsvi.py](file:///Users/milind/Desktop/Academics/Semester%202/CLMT5202/Project/src/features/rsvi.py) | CDC/ATSDR SVI methodology → percentile rank → sub-indices → composite 0–1 |
| [mortality_rates.py](file:///Users/milind/Desktop/Academics/Semester%202/CLMT5202/Project/src/features/mortality_rates.py) | Age-standardized rates (European Standard Pop), excess mortality |

### Analysis (`src/analysis/`)
| Module | Purpose |
|---|---|
| [panel_dataset.py](file:///Users/milind/Desktop/Academics/Semester%202/CLMT5202/Project/src/analysis/panel_dataset.py) | Merge all data → panel with interaction terms (Heat×RSVI, Heat×RSVI×D2022) |
| [eda.py](file:///Users/milind/Desktop/Academics/Semester%202/CLMT5202/Project/src/analysis/eda.py) | Summary stats, correlations, time series, scatter plots |
| [panel_regression.py](file:///Users/milind/Desktop/Academics/Semester%202/CLMT5202/Project/src/analysis/panel_regression.py) | H1/H2/H3 PanelOLS models with entity+time FE, clustered SEs |
| [diagnostics.py](file:///Users/milind/Desktop/Academics/Semester%202/CLMT5202/Project/src/analysis/diagnostics.py) | VIF, residuals, Hausman test, COVID sensitivity analysis |

### Visualization (`src/visualization/`)
| Module | Purpose |
|---|---|
| [maps.py](file:///Users/milind/Desktop/Academics/Semester%202/CLMT5202/Project/src/visualization/maps.py) | NUTS-2 choropleth maps (single-year and multi-year grids) |
| [timeseries.py](file:///Users/milind/Desktop/Academics/Semester%202/CLMT5202/Project/src/visualization/timeseries.py) | National trend panels and regional comparison plots |
| [case_studies.py](file:///Users/milind/Desktop/Academics/Semester%202/CLMT5202/Project/src/visualization/case_studies.py) | Paired regional comparisons (Calabria/Lombardia, etc.) |
| [regression_plots.py](file:///Users/milind/Desktop/Academics/Semester%202/CLMT5202/Project/src/visualization/regression_plots.py) | Coefficient plots and interaction (marginal effect) plots |

### Pipeline & CLI
| Module | Purpose |
|---|---|
| [run_pipeline.py](file:///Users/milind/Desktop/Academics/Semester%202/CLMT5202/Project/scripts/run_pipeline.py) | 14-step orchestrator with Click CLI (`--step`, `--from-step`, `--list-steps`) |
| [download_data.py](file:///Users/milind/Desktop/Academics/Semester%202/CLMT5202/Project/scripts/download_data.py) | Data download CLI |
| [run_analysis.py](file:///Users/milind/Desktop/Academics/Semester%202/CLMT5202/Project/scripts/run_analysis.py) | Analysis-only CLI |

## Verification

- **10/10 unit tests passing** (heatwave detection, RSVI construction, panel regression)
- **Package imports verified** — all modules load correctly
- **Pipeline CLI verified** — 14 steps recognized
- **uv environment working** — Python 3.11 with all 30+ dependencies installed

## Next Steps

To start using the project:

1. **Download raw data** — place mortality CSVs in `data/raw/mortality/`, socioeconomic data in `data/raw/socioeconomic/`
2. **Set CDS API key** — `export CDS_API_KEY=your_key` for ERA5-Land download
3. **Run the pipeline** — `uv run python scripts/run_pipeline.py`
