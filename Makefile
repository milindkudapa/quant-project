.PHONY: all download process features analyze visualize clean test help

# Default target
all: download process features analyze visualize

help:  ## Show this help
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) | sort | awk 'BEGIN {FS = ":.*?## "}; {printf "\033[36m%-20s\033[0m %s\n", $$1, $$2}'

# ─── Data Download ───────────────────────────────────────────────────
download: download-boundaries download-era5  ## Download all raw data

download-boundaries:  ## Download NUTS-2 boundary shapefiles
	uv run python -c "from src.data.nuts2_boundaries import setup_boundaries; setup_boundaries()"

download-era5:  ## Download ERA5-Land climate data
	uv run python scripts/download_data.py --source era5

# ─── Data Processing ────────────────────────────────────────────────
process: process-climate process-mortality process-socioeconomic  ## Process all raw data

process-climate:  ## Process ERA5-Land → regional daily climate
	uv run python -m src.data.process_climate

process-mortality:  ## Process ISTAT/Eurostat mortality data
	uv run python -m src.data.process_mortality

process-socioeconomic:  ## Process socioeconomic indicators
	uv run python -m src.data.process_socioeconomic

# ─── Feature Engineering ────────────────────────────────────────────
features: features-heatwave features-rsvi  ## Build all features

features-heatwave:  ## Detect heatwaves and compute metrics
	uv run python -m src.features.heatwave

features-rsvi:  ## Construct Regional Social Vulnerability Index
	uv run python -m src.features.rsvi

# ─── Analysis ───────────────────────────────────────────────────────
analyze: panel eda regression diagnostics  ## Run full analysis

panel:  ## Assemble panel dataset
	uv run python -m src.analysis.panel_dataset

eda:  ## Run exploratory data analysis
	uv run python -m src.analysis.eda

regression:  ## Run panel regression models (H1, H2, H3)
	uv run python -m src.analysis.panel_regression

diagnostics:  ## Run model diagnostics
	uv run python -m src.analysis.diagnostics

# ─── Visualization ──────────────────────────────────────────────────
visualize: maps case-studies  ## Generate all visualizations

maps:  ## Generate choropleth maps
	uv run python -m src.visualization.maps

case-studies:  ## Generate case study comparison plots
	uv run python -m src.visualization.case_studies

# ─── Pipeline ───────────────────────────────────────────────────────
pipeline:  ## Run the full pipeline end-to-end
	uv run python scripts/run_pipeline.py

pipeline-from-%:  ## Run pipeline from a specific step (e.g., make pipeline-from-run_eda)
	uv run python scripts/run_pipeline.py --from-step $*

# ─── Testing ────────────────────────────────────────────────────────
test:  ## Run unit tests
	uv run python -m pytest tests/ -v

# ─── Utilities ──────────────────────────────────────────────────────
clean:  ## Remove generated outputs (keep raw data)
	rm -rf data/interim/*
	rm -rf data/processed/*
	rm -rf outputs/figures/*
	rm -rf outputs/tables/*
	rm -rf outputs/reports/*
	@echo "Cleaned interim, processed, and output directories"

clean-all: clean  ## Remove everything including raw data
	rm -rf data/raw/climate/*.nc
	rm -rf data/raw/boundaries/*
	@echo "Cleaned all data directories"
