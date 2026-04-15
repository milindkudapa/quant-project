"""
Analysis-only script (assumes processed data exists).

Usage
-----
    uv run python scripts/run_analysis.py
    uv run python scripts/run_analysis.py --skip-eda
"""

from __future__ import annotations

import click
from loguru import logger

from src.utils.config import load_config


@click.command()
@click.option("--skip-eda", is_flag=True, help="Skip exploratory data analysis")
@click.option("--skip-viz", is_flag=True, help="Skip visualization generation")
@click.option("--config", default=None, help="Path to config file")
def main(skip_eda: bool, skip_viz: bool, config: str | None):
    """Run the analysis pipeline (assumes data is processed)."""
    cfg = load_config(config)

    # Step 1: Assemble panel
    logger.info("Assembling panel dataset...")
    from src.analysis.panel_dataset import build_panel_dataset
    build_panel_dataset(cfg)

    # Step 2: EDA
    if not skip_eda:
        logger.info("Running exploratory data analysis...")
        from src.analysis.eda import run_eda
        run_eda(cfg)

    # Step 3: Panel regressions
    logger.info("Running panel regressions...")
    from src.analysis.panel_regression import run_all_models
    results, comparison = run_all_models(cfg)

    # Step 4: Diagnostics
    logger.info("Running model diagnostics...")
    from src.analysis.diagnostics import run_diagnostics
    run_diagnostics(cfg)

    # Step 5: Visualizations
    if not skip_viz:
        logger.info("Generating visualizations...")
        from src.visualization.maps import generate_all_maps
        from src.visualization.case_studies import generate_all_case_studies

        try:
            generate_all_maps(cfg)
        except Exception as e:
            logger.warning(f"Map generation failed: {e}")

        generate_all_case_studies(cfg)

    logger.success("Analysis complete! Check outputs/ for results.")


if __name__ == "__main__":
    main()
