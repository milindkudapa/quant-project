"""One-off generator for notebooks/07_weekly_panel_analysis.ipynb.

Keeps the notebook content source-controlled as Python, then renders a clean
.ipynb on disk. Run from project root:

    python scripts/build_weekly_notebook.py
"""
from __future__ import annotations

from pathlib import Path

import nbformat as nbf


CELLS: list[tuple[str, str]] = []


def md(src: str) -> None:
    CELLS.append(("markdown", src.strip() + "\n"))


def code(src: str) -> None:
    CELLS.append(("code", src.strip() + "\n"))


# -------------------- NOTEBOOK CONTENT --------------------

md("""
# 07 — Weekly Panel Analysis (Non-linear + Distributed Lags)

Reproduces the region-week panel approach that replaces the original
annual (N=220) regression with a richer weekly panel (N≈3,800).

**What this notebook does**

1. Build weekly climate features from daily ERA5 aggregates (climatology,
   anomaly + anomaly², p90/p95 exceedances, `hw_days` per week).
2. Parse Eurostat weekly all-cause deaths and compute per-100k weekly
   mortality rates.
3. Assemble the region-week panel with 1- and 2-week lags, RSVI
   interactions, and triple-interactions with the 2022 dummy.
4. Fit H1/H2/H3 with region FE + explicit year and week-of-year dummies
   (preserves `d2022` identification), clustered SE by region.
5. Compare head-to-head against the annual panel.

Run the whole notebook top-to-bottom. It reads intermediate parquet
files from `data/interim/` and writes result tables to
`outputs/tables/`.
""")

code("""
import sys, os
project_root = os.path.abspath(os.path.join(os.getcwd(), '..'))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from IPython.display import display

from src.utils.config import load_config, get_path
from src.utils.io import load_dataframe

cfg = load_config()
print('Ready.')
""")

md("""
## 1. Weekly climate features

Aggregate daily regional climate (June 1 – September 30, 2012–2022) to
ISO-weeks. Compute a **(region, week-of-year) climatology** so the
`tmax_anomaly_week` is purged of seasonality, then add:

- `tmax_anomaly_week_sq` — quadratic for non-linear heat response
- `above_p90_days`, `above_p95_days` — threshold exceedance counts
- `hw_days_week` — heatwave days (from existing daily flags)
""")

code("""
from src.features.weekly_climate import build_weekly_climate

weekly_climate = build_weekly_climate(cfg)
print(weekly_climate.shape)
display(weekly_climate.head())
""")

md("""
## 2. Weekly mortality

Parse Eurostat weekly deaths (`sex=Total`, 20 Italian NUTS-2 regions,
ITH2 merged into ITH1). Compute mortality rate per 100k using the
annualized-weekly denominator: `weekly_deaths / (population / 52) × 100k`.
""")

code("""
from src.features.weekly_mortality import build_weekly_mortality

weekly_mortality = build_weekly_mortality(cfg)
print(weekly_mortality.shape)
display(weekly_mortality.head())
""")

md("""
## 3. Assemble the weekly panel

Merge climate + mortality + annual RSVI (propagated to each week within
the year). Add 1- and 2-week lags of heat exposure — lags that would
cross a summer boundary (W39 → W22 of the next year) are set to NaN to
prevent leakage.
""")

code("""
from src.analysis.weekly_panel_dataset import build_weekly_panel

panel = build_weekly_panel(cfg)
print(f'Weekly panel: {panel.shape[0]} rows × {panel.shape[1]} columns')
print(f'Regions: {panel[\"nuts2_code\"].nunique()}, '
      f'Years: {panel[\"iso_year\"].nunique()}, '
      f'Weeks-of-year: {panel[\"iso_week\"].nunique()}')
display(panel[['nuts2_code', 'iso_year', 'iso_week', 'mortality_rate_week',
               'hw_days_week', 'tmax_anomaly_week', 'rsvi', 'd2022']].head())
""")

md("""
### Panel descriptives
""")

code("""
panel[['mortality_rate_week', 'hw_days_week', 'tmax_anomaly_week',
       'tmax_anomaly_week_sq', 'above_p95_days', 'rsvi']].describe()
""")

md("""
### Check for multicollinearity (before regression)

Using the centered-interaction construction ensures main-effect VIFs stay
well below 5 — the interaction VIF itself is mechanical and informative
only for the interaction coefficient.
""")

code("""
from src.analysis.diagnostics import compute_vif

p = panel.copy()
for v in ['hw_days_week', 'tmax_anomaly_week', 'rsvi']:
    p[f'{v}_c'] = p[v] - p[v].mean()
p['hwxrsvi_c']  = p['hw_days_week_c'] * p['rsvi_c']
p['anomxrsvi_c'] = p['tmax_anomaly_week_c'] * p['rsvi_c']

vif_cols = ['hw_days_week_c', 'tmax_anomaly_week_c', 'rsvi_c',
            'hwxrsvi_c', 'anomxrsvi_c']
vif_df = compute_vif(p.dropna(subset=vif_cols), vif_cols)
display(vif_df)
""")

md("""
## 4. Run weekly regressions

Seven specifications:

- **H1_weekly_FE** — base model, region FE + year + week-of-year dummies,
  `hw_days`, lags, `tmax_anomaly`, `tmax_anomaly²`, p95 threshold.
- **H2_weekly_FE** — adds RSVI main effect and heat × RSVI interactions.
- **H3_weekly_FE** — adds the triple interactions with the 2022 dummy.
- **H2_weekly_pooled, H3_weekly_pooled** — no region FE, to keep the
  between-region RSVI variation identified (RSVI is slow-moving, so
  region FE absorbs most of its signal).
- **H3_weekly_FE_excl2020** and **H3_weekly_FE_excl_covid** — robustness
  that drops COVID-contaminated years.

Clustered standard errors by region.
""")

code("""
from src.analysis.weekly_regression import run_weekly_models

results = run_weekly_models(cfg)
results.head(10)
""")

md("""
### Key coefficients from the weekly FE models
""")

code("""
key_vars = [
    'hw_days_week', 'hw_days_week_lag1', 'hw_days_week_lag2',
    'tmax_anomaly_week', 'tmax_anomaly_week_sq', 'hot_week_p95',
    'rsvi', 'hw_days_x_rsvi', 'tmax_anom_x_rsvi', 'p95_x_rsvi',
    'hw_days_x_rsvi_x_d2022', 'tmax_anom_x_rsvi_x_d2022',
    'p95_x_rsvi_x_d2022',
]

fe_models = ['H1_weekly_FE', 'H2_weekly_FE', 'H3_weekly_FE']
key = (
    results[results['model'].isin(fe_models) & results['variable'].isin(key_vars)]
    .pivot(index='variable', columns='model', values='coefficient')
    .reindex(key_vars)
)
pvals = (
    results[results['model'].isin(fe_models) & results['variable'].isin(key_vars)]
    .pivot(index='variable', columns='model', values='p_value')
    .reindex(key_vars)
)

def _mark(coef, p):
    if pd.isna(coef):
        return ''
    stars = '***' if p < 0.001 else '**' if p < 0.01 else '*' if p < 0.05 else ''
    return f'{coef:+.4f}{stars}'

formatted = key.copy().astype(object)
for m in fe_models:
    for v in key_vars:
        formatted.loc[v, m] = _mark(key.loc[v, m], pvals.loc[v, m])
display(formatted)
""")

md("""
## 5. Annual vs Weekly — head-to-head

Do the same hypotheses look different under the weekly specification?
""")

code("""
annual = pd.read_csv(get_path(cfg, 'tables') / 'regression_results.csv')

def _summarise(df, model, var):
    row = df[(df['model'] == model) & (df['variable'] == var)]
    if row.empty:
        return {'coef': np.nan, 'p': np.nan}
    return {'coef': float(row['coefficient'].iloc[0]),
            'p': float(row['p_value'].iloc[0])}

rows = [
    ('H1: heat → mortality',
     _summarise(annual, 'H1_base', 'hw_days'),
     _summarise(results, 'H1_weekly_FE', 'hw_days_week')),
    ('H1: lag-1 heat effect',
     {'coef': np.nan, 'p': np.nan},
     _summarise(results, 'H1_weekly_FE', 'hw_days_week_lag1')),
    ('H2: hw × rsvi',
     _summarise(annual, 'H2_interaction', 'hw_days_x_rsvi'),
     _summarise(results, 'H2_weekly_FE', 'hw_days_x_rsvi')),
    ('H2: tmax_anom × rsvi',
     _summarise(annual, 'H2_interaction', 'tmax_anomaly_x_rsvi'),
     _summarise(results, 'H2_weekly_FE', 'tmax_anom_x_rsvi')),
    ('H2: p95 × rsvi',
     {'coef': np.nan, 'p': np.nan},
     _summarise(results, 'H2_weekly_FE', 'p95_x_rsvi')),
    ('H3: hw × rsvi × d2022',
     _summarise(annual, 'H3_2022_amplification', 'hw_days_x_rsvi_x_d2022'),
     _summarise(results, 'H3_weekly_FE', 'hw_days_x_rsvi_x_d2022')),
    ('H3: tmax_anom × rsvi × d2022',
     _summarise(annual, 'H3_2022_amplification', 'tmax_anomaly_x_rsvi_x_d2022'),
     _summarise(results, 'H3_weekly_FE', 'tmax_anom_x_rsvi_x_d2022')),
]

def _fmt(x):
    if pd.isna(x['coef']):
        return '—'
    stars = '***' if x['p'] < 0.001 else '**' if x['p'] < 0.01 else '*' if x['p'] < 0.05 else ''
    return f\"{x['coef']:+.4f}{stars} (p={x['p']:.3f})\"

comp = pd.DataFrame(rows, columns=['term', 'annual', 'weekly'])
comp['annual'] = comp['annual'].apply(_fmt)
comp['weekly'] = comp['weekly'].apply(_fmt)
display(comp)
""")

md("""
## 6. Visualizations

### Lag structure of heat exposure

The lag-1 coefficient is larger than the contemporaneous coefficient —
mortality responds to heat with roughly a one-week delay. Annual
aggregation collapsed this signal.
""")

code("""
h1 = results[results['model'] == 'H1_weekly_FE']
lag_terms = ['hw_days_week', 'hw_days_week_lag1', 'hw_days_week_lag2']
lag_df = h1[h1['variable'].isin(lag_terms)].set_index('variable').loc[lag_terms]

fig, ax = plt.subplots(figsize=(7, 4))
xs = ['lag 0 (current)', 'lag 1 (prev week)', 'lag 2 (2 weeks prior)']
ax.errorbar(xs, lag_df['coefficient'],
            yerr=1.96 * lag_df['std_error'], fmt='o',
            capsize=5, color='steelblue', ecolor='grey')
ax.axhline(0, color='black', linewidth=0.5)
ax.set_ylabel('log-mortality coefficient\\n(per extra heatwave day)')
ax.set_title('Distributed lag of heatwave exposure on weekly mortality')
ax.grid(True, alpha=0.3)
plt.tight_layout()
plt.show()
""")

md("""
### Marginal effect of heat at different RSVI levels (H2)

Under the H2 weekly FE model, the marginal effect of a 1°C temperature
anomaly on log mortality is `β_anom + β_anom×rsvi × RSVI`. We evaluate
at the 10th, 50th, and 90th percentile of RSVI.
""")

code("""
h2 = results[results['model'] == 'H2_weekly_FE'].set_index('variable')
b_anom   = h2.loc['tmax_anomaly_week', 'coefficient']
b_inter  = h2.loc['tmax_anom_x_rsvi',   'coefficient']
se_inter = h2.loc['tmax_anom_x_rsvi',   'std_error']

rsvi_p10, rsvi_p50, rsvi_p90 = panel['rsvi'].quantile([0.1, 0.5, 0.9])
levels = {'RSVI p10': rsvi_p10, 'RSVI p50': rsvi_p50, 'RSVI p90': rsvi_p90}

effects = {lbl: b_anom + b_inter * v for lbl, v in levels.items()}
ses     = {lbl: abs(v) * se_inter    for lbl, v in levels.items()}

fig, ax = plt.subplots(figsize=(6, 4))
xs = list(effects.keys())
ys = list(effects.values())
yerr = [1.96 * s for s in ses.values()]
ax.bar(xs, ys, yerr=yerr, capsize=6, color=['#a6cee3', '#1f78b4', '#08306b'])
ax.axhline(0, color='black', linewidth=0.5)
ax.set_ylabel('Marginal effect on log-mortality\\n(per °C tmax anomaly)')
ax.set_title('H2: vulnerability amplifies the temperature-anomaly response')
plt.tight_layout()
plt.show()
""")

md("""
### H3 triple-interaction across specifications

The `tmax_anom × rsvi × d2022` coefficient is consistently negative and
robust to excluding 2020 (COVID year) and 2020–2021.
""")

code("""
triple = 'tmax_anom_x_rsvi_x_d2022'
models_h3 = ['H3_weekly_FE', 'H3_weekly_FE_excl2020', 'H3_weekly_FE_excl_covid']
sub = results[(results['model'].isin(models_h3)) & (results['variable'] == triple)]

fig, ax = plt.subplots(figsize=(7, 4))
ax.errorbar(sub['model'], sub['coefficient'],
            yerr=1.96 * sub['std_error'], fmt='o', capsize=5,
            color='crimson', ecolor='grey')
ax.axhline(0, color='black', linewidth=0.5)
ax.set_ylabel('β (tmax_anom × rsvi × d2022)')
ax.set_title('H3: 2022 attenuated (not amplified) the vulnerability–heat gradient')
ax.tick_params(axis='x', rotation=20)
plt.tight_layout()
plt.show()
""")

md("""
## 7. Summary

**Findings under the weekly panel**

- H1 — strongly supported. Heat raises weekly mortality in the current
  *and* following week, with lag-1 the larger coefficient. Annual
  aggregation missed the lag channel entirely.
- H2 — supported on the **non-linear** channel. RSVI amplifies the
  temperature-anomaly effect (p≈0.017) and the p95 extreme-week effect
  (p≈0.008). The linear `hw_days × rsvi` interaction is still null,
  matching the annual result.
- H3 — significant (p<0.001) and robust to COVID-year exclusion.
  Critically, the sign is *negative*: in 2022 the gradient between
  vulnerable and non-vulnerable regions *narrowed*, not widened.
  Candidates: adaptation after 2003/2015, COVID harvesting,
  2022 emergency response.

**Why the weekly panel helps**

Annual aggregation collapses ~120 daily observations per region-year
into one row, which both removes the lag signal and averages away the
extreme-heat days that drive most mortality. The weekly panel preserves
within-summer variation and gives ~17× more observations, which is what
moves H2/H3 from null to significant.
""")


# -------------------- WRITE NOTEBOOK --------------------

nb = nbf.v4.new_notebook()
for kind, src in CELLS:
    if kind == "markdown":
        nb.cells.append(nbf.v4.new_markdown_cell(src))
    else:
        nb.cells.append(nbf.v4.new_code_cell(src))

out_path = (
    Path(__file__).resolve().parents[1]
    / "notebooks"
    / "07_weekly_panel_analysis.ipynb"
)
nbf.write(nb, out_path)
print(f"Wrote {out_path}")
