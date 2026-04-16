"""
Full pipeline orchestrator.

Runs the complete analysis pipeline:
1. Download data (boundaries, ERA5)
2. Process raw data (climate, mortality, socioeconomic)
3. Engineer features (heatwave, temperature, RSVI, mortality rates)
4. Assemble panel dataset
5. Run EDA
6. Run panel regressions
7. Run diagnostics
8. Generate visualizations (maps, time series, case studies)

Usage
-----
    uv run python scripts/run_pipeline.py [--step STEP] [--from STEP]
"""

from __future__ import annotations

import sys
import time

import click
from loguru import logger

from src.utils.config import load_config


# Configure logging
logger.remove()
logger.add(
    sys.stderr,
    format="<green>{time:HH:mm:ss}</green> | <level>{level:<8}</level> | <cyan>{name}</cyan> - {message}",
    level="INFO",
)
logger.add(
    "outputs/reports/pipeline.log",
    format="{time:YYYY-MM-DD HH:mm:ss} | {level:<8} | {name} - {message}",
    level="DEBUG",
    rotation="10 MB",
)


PIPELINE_STEPS = [
    "download_boundaries",
    "download_era5",
    "process_climate",
    "process_mortality",
    "process_socioeconomic",
    "build_heatwave_features",
    "build_rsvi",
    "assemble_panel",
    "run_eda",
    "run_regression",
    "run_diagnostics",
    "generate_maps",
    "generate_timeseries",
    "generate_case_studies",
]


def run_step(step_name: str, cfg: dict) -> None:
    """Run a single pipeline step.

    Parameters
    ----------
    step_name : str
        Name of the step to run.
    cfg : dict
        Configuration dictionary.
    """
    logger.info(f"{'='*60}")
    logger.info(f"Running step: {step_name}")
    logger.info(f"{'='*60}")

    start = time.time()

    if step_name == "download_boundaries":
        from src.data.nuts2_boundaries import setup_boundaries
        setup_boundaries(cfg)

    elif step_name == "download_era5":
        source = cfg["temperature"].get("primary_dataset", "era5_land")
        if source == "earthmover":
            logger.info("Using Earthmover (AWS Zarr) for climate data...")
            from src.data.download_earthmover import download_all_earthmover
            download_all_earthmover(cfg)
        else:
            logger.info("Using CDS API for climate data...")
            from src.data.download_era5 import download_all_era5
            download_all_era5(cfg)

    elif step_name == "process_climate":
        from src.data.process_climate import process_climate_data
        process_climate_data(cfg)

    elif step_name == "process_mortality":
        from src.data.process_mortality import process_mortality_data
        process_mortality_data(cfg)

    elif step_name == "process_socioeconomic":
        from src.data.process_socioeconomic import process_socioeconomic_data
        process_socioeconomic_data(cfg)

    elif step_name == "build_heatwave_features":
        from src.features.heatwave import build_heatwave_features
        build_heatwave_features(cfg)

    elif step_name == "build_rsvi":
        from src.features.rsvi import build_rsvi
        build_rsvi(cfg)

    elif step_name == "assemble_panel":
        from src.analysis.panel_dataset import build_panel_dataset
        build_panel_dataset(cfg)

    elif step_name == "run_eda":
        from src.analysis.eda import run_eda
        run_eda(cfg)

    elif step_name == "run_regression":
        from src.analysis.panel_regression import run_all_models
        run_all_models(cfg)

    elif step_name == "run_diagnostics":
        from src.analysis.diagnostics import run_diagnostics
        run_diagnostics(cfg)

    elif step_name == "generate_maps":
        from src.visualization.maps import generate_all_maps
        generate_all_maps(cfg)

    elif step_name == "generate_timeseries":
        from src.visualization.timeseries import plot_national_trends
        from src.utils.io import load_dataframe
        from src.utils.config import get_path
        panel = load_dataframe(get_path(cfg, "processed_data") / "panel_dataset.csv")
        fig_dir = get_path(cfg, "figures")
        fig_dir.mkdir(parents=True, exist_ok=True)
        plot_national_trends(panel, output_path=fig_dir / "national_trends.png")

    elif step_name == "generate_case_studies":
        from src.visualization.case_studies import generate_all_case_studies
        generate_all_case_studies(cfg)

    else:
        raise ValueError(f"Unknown pipeline step: {step_name}")

    elapsed = time.time() - start
    logger.success(f"Step '{step_name}' completed in {elapsed:.1f}s")


@click.command()
@click.option("--step", default=None, help="Run a single step by name")
@click.option("--from-step", "from_step", default=None, help="Run from a specific step onwards")
@click.option("--config", default=None, help="Path to config file")
@click.option("--list-steps", is_flag=True, help="List all pipeline steps")
def main(step: str | None, from_step: str | None, config: str | None, list_steps: bool):
    """Run the heat-mortality analysis pipeline."""
    if list_steps:
        click.echo("Available pipeline steps:")
        for i, s in enumerate(PIPELINE_STEPS, 1):
            click.echo(f"  {i:2d}. {s}")
        return

    cfg = load_config(config)

    if step:
        if step not in PIPELINE_STEPS:
            click.echo(f"Unknown step: {step}")
            click.echo(f"Available: {', '.join(PIPELINE_STEPS)}")
            sys.exit(1)
        run_step(step, cfg)
        return

    # Determine starting step
    steps_to_run = PIPELINE_STEPS
    if from_step:
        if from_step not in PIPELINE_STEPS:
            click.echo(f"Unknown step: {from_step}")
            sys.exit(1)
        idx = PIPELINE_STEPS.index(from_step)
        steps_to_run = PIPELINE_STEPS[idx:]

    logger.info(f"Running {len(steps_to_run)} pipeline steps")
    total_start = time.time()

    for step_name in steps_to_run:
        try:
            run_step(step_name, cfg)
        except Exception as e:
            logger.error(f"Step '{step_name}' failed: {e}")
            logger.info("Pipeline halted. Fix the issue and re-run with --from-step")
            sys.exit(1)

    total_elapsed = time.time() - total_start
    logger.success(f"Pipeline complete! Total time: {total_elapsed:.1f}s")


if __name__ == "__main__":
    main()
