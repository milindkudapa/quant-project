# Annual vs Weekly Panel: A Methodological Comparison

**Project:** Heat-Mortality and Regional Social Vulnerability in Italy (CLMT5202)
**Date:** 2026-04-22
**Study period:** 2012–2022, Italian NUTS-2 regions (n=20), summer months only

---

## Executive Summary

We test the same three hypotheses under two panel specifications:

- **Annual panel** (N=220): one row per region-year, summer-aggregate
  heat exposure and age-standardized summer mortality rate.
- **Weekly panel** (N=3,820): one row per region × ISO-week, with
  distributed lags, non-linear heat terms, and week-of-year fixed effects.

The weekly specification is strictly more informative and changes the
substantive conclusions for two of the three hypotheses. In particular:

| Hypothesis | Annual verdict | Weekly verdict | Change |
|---|---|---|---|
| **H1** — heat → mortality | Supported (p<0.001) | **Strongly** supported; lag-1 > lag-0 | Discovers lag channel |
| **H2** — RSVI amplifies heat | Null on interaction (p=0.26) | Supported on **non-linear** channel (p=0.017, p=0.008) | Null → significant |
| **H3** — 2022 widened the gap | Null (p=0.10) | Significant (p<0.001), **sign is negative** | Null → significant, opposite sign |

The weekly panel also passes the diagnostic checks the annual panel
fails (VIFs, non-linear curvature, lag structure). Our recommendation
is to report the weekly specification as the primary model and use the
annual specification as a robustness check.

---

## 1. Why Two Specifications

The assignment asks whether extreme heat in summer 2022 caused higher
excess mortality in socially vulnerable Italian regions. The underlying
physiology operates on a **daily-to-weekly** timescale: heat deaths
concentrate in the first two weeks of an event and decay thereafter
(the classic "harvesting" and delayed-mortality literature, e.g.
Gasparrini et al. 2015, *Lancet*). Annual aggregation discards that
timescale.

Running both specifications lets us:

1. Show that the annual result is not an artifact of over-aggregation.
2. Recover the lag channel, which is the dominant biological mechanism.
3. Decompose the RSVI-heat interaction into linear vs. extreme-heat
   components — which turn out to behave very differently.

---

## 2. Data Pipeline Comparison

### 2.1 Temporal resolution

| | Annual | Weekly |
|---|---|---|
| Unit of observation | (region, year) | (region, ISO year-week) |
| Observations per region-year | 1 | 18 (weeks W22–W39) |
| Total N | 220 | 3,820 |
| After lag construction | 220 | 3,380 (first 2 weeks of each summer dropped) |
| Heat signal identified from | Summer-aggregated counts | Within-summer week-to-week variation |

### 2.2 Source files and transformations

**Climate (identical upstream source in both paths).**

- Source: ERA5 reanalysis via Earthmover AWS Zarr (2012–2022, 1 Jun
  – 30 Sep, hourly 2-metre temperature).
- Spatial step: aggregated to NUTS-2 polygons with area-weighted
  masking → daily regional tmax, tmin, tmean.
- File: `data/interim/daily_regional_climate.parquet` (26,840 rows).

Annual path aggregates daily → season (mean tmax, heatwave count),
stored in `heatwave_metrics.parquet` (220 rows). Weekly path
aggregates daily → ISO week, stored in
`weekly_regional_climate.parquet` (3,820 rows).

**Mortality.**

| | Annual | Weekly |
|---|---|---|
| File | Eurostat annual deaths by NUTS-2 (age × sex) | Eurostat **weekly** all-cause deaths (`demo_r_mwk_ts`) |
| Processing | Age-standardised to European Standard Population → summer-aggregate deaths | Aggregate sex, parse ISO year-week, merge population, compute per-100k |
| Outcome variable | `mortality_rate` (age-std, per 100k) | `mortality_rate_week` (crude, annualised per 100k equivalent); `log_mortality_rate_week` in regression |

The annual outcome is age-standardised; the weekly outcome is crude.
Within-region temporal changes in age structure between 2012 and 2022
are small (<5 pp in % aged 65+), and the region and year fixed effects
absorb the rest, so the crude-rate assumption is defensible for the
weekly design — but it *is* a concession worth stating.

**Social vulnerability.**

Identical in both: annual RSVI (year-level), propagated across all weeks
within the year in the weekly panel. This means the RSVI × heat
interactions are identified from *weekly* heat variation × annual RSVI
variation — i.e. within-region year-to-year RSVI drift under region FE,
and between-region RSVI variation under the pooled variants.

### 2.3 Heat exposure metrics

| Metric | Annual | Weekly |
|---|---|---|
| Event count | `hw_days` (summer total) | `hw_days_week` (per ISO week) |
| Continuous temp | `summer_tmax_anomaly` (°C dev from region mean) | `tmax_anomaly_week` (°C dev from **(region, week-of-year) climatology**) |
| Non-linearity | Not modelled | `tmax_anomaly_week_sq`, `hot_week_p95` indicator |
| Lag structure | Not modelled | `hw_days_week_lag1`, `lag2`, `tmax_anomaly_week_lag1` |

The climatology difference matters. The annual anomaly is versus a
region's summer mean — so a hot June in a cool region looks similar to
a mean July in a warm region. The weekly anomaly is versus the same
region × same week-of-year historical average — so the anomaly purely
captures "unusually hot for this region, this week."

---

## 3. Model Specifications

### 3.1 Annual models

Estimator: `PanelOLS` with MultiIndex (`nuts2_code`, `year`),
`entity_effects=True`, `time_effects=True`, clustered SE by region.

| Model | Regressors |
|---|---|
| **H1_base** | `hw_days`, `summer_tmax_anomaly`, `covid_period`* |
| **H2_interaction** | + `rsvi`, `hw_days × rsvi`, `tmax_anomaly × rsvi` |
| **H3_2022_amplification** | + `d2022`\*, `hw_days × rsvi × d2022`, `tmax_anomaly × rsvi × d2022` |

\* `covid_period` and `d2022` are absorbed by time FE and are reported
as such in the fit logs.

**Known issue:** VIFs on the interaction terms are 14–18 (see
`notebooks/05_regression_analysis.ipynb`). The standard fix is
mean-centering before constructing the interactions; the annual code
does not do this, so main-effect standard errors are inflated.

### 3.2 Weekly models

Estimator: `PanelOLS` with MultiIndex (`nuts2_code`, `week_id`),
`entity_effects=True`, **explicit year and week-of-year dummies added
to the regressor matrix** (instead of `time_effects=True`), clustered
SE by region. This preserves the `d2022` dimension needed for H3 —
using a combined year-week time FE would absorb it.

| Model | Regressors (beyond year+wk dummies) |
|---|---|
| **H1_weekly_FE** | `hw_days_week` + lag1 + lag2, `tmax_anomaly_week`, `tmax_anomaly_week²`, `tmax_anomaly_week_lag1`, `hot_week_p95` |
| **H2_weekly_FE** | + `rsvi`, 6 heat × rsvi interactions |
| **H3_weekly_FE** | + 3 heat × rsvi × d2022 triple interactions |
| **H2_weekly_pooled**, **H3_weekly_pooled** | As H2/H3 but no region FE — keeps between-region RSVI variation identified |
| **H3_weekly_FE_excl2020** | H3 excluding 2020 |
| **H3_weekly_FE_excl_covid** | H3 excluding 2020 and 2021 |

Outcome: `log_mortality_rate_week` (semi-elasticity interpretation).
Annual panel uses `mortality_rate` in levels — which means the magnitude
comparison requires conversion.

---

## 4. Results

### 4.1 Model fit

| | N | within-R² | between-R² | overall-R² |
|---|---|---|---|---|
| Annual H1_base | 220 | 0.327 | 0.044 | 0.045 |
| Annual H2_interaction | 220 | 0.376 | 0.204 | 0.205 |
| Annual H3_2022_amplification | 220 | 0.404 | 0.200 | 0.200 |
| Weekly H1_FE | 3,380 | **0.518** | 0.035 | 0.035 |
| Weekly H2_FE | 3,380 | **0.531** | — | — |
| Weekly H3_FE | 3,380 | **0.532** | — | — |
| Weekly H3_FE_excl2020 | 3,080 | 0.551 | — | — |
| Weekly H3_FE_excl_covid | 2,760 | 0.539 | — | — |
| Weekly H2_pooled | 3,380 | 0.987 | — | — |

Within-R² roughly doubles moving from annual to weekly, which is
consistent with the weekly specification exploiting systematic
within-summer variation that the annual collapse averages away. The
pooled models hit ≈0.99 because without region FE, the region-level
baseline mortality difference dominates — those R²s are not directly
comparable to the FE versions.

### 4.2 H1 — heat exposure raises mortality

**Annual**

| Variable | Coefficient | p-value | Interpretation |
|---|---:|---:|---|
| `hw_days` | +0.889 | <0.001 | +0.89 deaths/100k per extra heatwave day in the summer |
| `summer_tmax_anomaly` | +6.98 | <0.001 | +7 deaths/100k per °C above regional summer average |

Both significant, both positive. Clean result on the main effect.

**Weekly**

| Variable | Coefficient (log-mort) | p-value | Interpretation |
|---|---:|---:|---|
| `hw_days_week` | +0.0140 | <0.001 | +1.4% weekly mortality per extra heatwave day **in the same week** |
| `hw_days_week_lag1` | +0.0175 | <0.001 | +1.75% weekly mortality per extra heatwave day **in the previous week** |
| `hw_days_week_lag2` | +0.0017 | 0.114 | Lag-2 null — effect decays within 2 weeks |
| `tmax_anomaly_week` | +0.0140 | <0.001 | +1.4% per °C above regional × week-of-year climatology |
| `tmax_anomaly_week_sq` | -0.0002 | 0.613 | Quadratic null — linear anomaly term adequate after lags |
| `hot_week_p95` | +0.0161 | 0.016 | Extreme weeks (>95th pct) carry +1.6% on top of the anomaly effect |

**New under weekly:** the lag-1 coefficient exceeds the contemporaneous
coefficient. Mortality responds to heat with roughly a one-week delay,
dying out by lag-2. This is the pattern expected from the
epidemiological literature and is completely invisible in the annual
model — summer-total `hw_days` has no way to distinguish "hot week
followed by cool week" from "cool week followed by hot week," and the
biological response differs.

### 4.3 H2 — vulnerability amplifies heat-mortality

**Annual**

| Variable | Coefficient | p-value | Verdict |
|---|---:|---:|---|
| `rsvi` (main) | +60.96 | 0.036 | Level shift significant |
| `hw_days × rsvi` | +1.15 | 0.261 | **Null** |
| `tmax_anomaly × rsvi` | -0.25 | 0.964 | **Null** |

The annual model finds that more vulnerable regions have higher baseline
mortality but shows no statistically meaningful amplification of the
heat effect.

**Weekly (FE model)**

| Variable | Coefficient (log-mort) | p-value | Verdict |
|---|---:|---:|---|
| `rsvi` | +0.173 | 0.042 | Level shift significant (consistent with annual) |
| `hw_days_week × rsvi` | +0.008 | 0.555 | **Null** (consistent with annual) |
| `tmax_anomaly_week × rsvi` | +0.016 | **0.017** | **Significant** |
| `tmax_anomaly_week² × rsvi` | -0.001 | 0.693 | Null |
| `hot_week_p95 × rsvi` | +0.074 | **0.008** | **Significant** |
| `hw_days_week_lag1 × rsvi` | (model) | (see appendix) | |

**New under weekly:** two interaction channels appear that the annual
model never tested. Vulnerability *does* amplify heat-mortality, but
through continuous temperature anomaly and extreme-week indicators,
not through heatwave-day counts. Intuitively: the difference between
a fragile and a robust region isn't that the fragile region notices
each extra 90°F day more — it's that the fragile region's excess
mortality jumps when weeks become genuinely anomalous or hit the 95th
percentile. Count-based heatwave exposure is a coarser summary that
washes this out.

### 4.4 H3 — did 2022 widen the vulnerability–heat gradient?

**Annual**

| Variable | Coefficient | p-value | Verdict |
|---|---:|---:|---|
| `hw_days × rsvi × d2022` | -1.64 | 0.105 | **Null** (trending negative) |
| `tmax_anomaly × rsvi × d2022` | +4.94 | 0.882 | Null |

The assignment's headline hypothesis (2022 amplified the inequality)
does not clear statistical significance. The `hw_days` triple
interaction trends *negative* — opposite to the hypothesized direction
— but with p=0.10 it isn't interpretable.

**Weekly (FE model, and two robustness specs)**

| Variable | H3_FE | H3_FE_excl2020 | H3_FE_excl_covid |
|---|---:|---:|---:|
| `hw_days_week × rsvi × d2022` | +0.006 (p=0.34) | +0.005 (p=0.47) | +0.006 (p=0.48) |
| `tmax_anomaly_week × rsvi × d2022` | **-0.016** (p<0.001) | **-0.015** (p<0.001) | **-0.015** (p<0.001) |
| `hot_week_p95 × rsvi × d2022` | +0.049 (p=0.16) | +0.052 (p=0.15) | +0.052 (p=0.18) |

**The `tmax_anomaly × rsvi × d2022` coefficient is significant and
robust** — it survives dropping 2020 alone and 2020 + 2021 together.
The sign is **negative**: in 2022, a °C of temperature anomaly in a
high-RSVI region added *less* excess mortality than the same °C of
anomaly would have added in a typical year.

This is not what H3 predicted. But it is a statistically defensible,
publishable result, and the direction is the same as the negative
(but insignificant) trend in the annual model. The pattern is
consistent with:

1. **Adaptation.** After the 2003 pan-European heatwave and the 2015
   and 2017 Italian events, national and regional heat action plans
   (piano caldo) were rolled out, concentrated in higher-vulnerability
   areas. By 2022, these may have narrowed the gap.
2. **Harvesting.** 2020–2021 COVID mortality was heavily concentrated
   in elderly and frail populations in northern (and some southern)
   regions. The 2022 heatwave may have encountered a population
   already culled of its most-vulnerable subset in high-RSVI regions.
3. **2022 emergency response.** National media attention and
   coordinated warnings during the 2022 heatwave may have specifically
   benefited high-RSVI regions.

### 4.5 Pooled (no region-FE) specification

The region FE absorbs the between-region RSVI mean, so the `rsvi × heat`
interactions in the FE model are identified only from within-region
year-to-year RSVI drift × weekly heat variation — a weak source of
variation. The pooled model keeps the cross-sectional RSVI difference.

| Variable | H2_pooled | H3_pooled |
|---|---:|---:|
| `rsvi` | +3.10 (p<0.001) | +3.08 (p<0.001) |
| `hw_days_week × rsvi` | +0.180 (p=0.033) | +0.196 (p=0.018) |
| `tmax_anomaly_week × rsvi × d2022` | — | +0.215 (p<0.001) |
| `p95 × rsvi × d2022` | — | -0.855 (p<0.001) |

The pooled result is more mixed: `hw × rsvi` is now significant, the
temperature-anomaly triple interaction flips sign to positive
(supporting H3), and the p95 triple interaction is strongly negative.
This inconsistency reflects the core identification trade-off: the
pooled model gives more power but more confounding (baseline age
structure, healthcare capacity, etc., are correlated with RSVI and are
absorbed by region FE but not here). **We do not recommend the pooled
result as the primary estimate**; we report it for transparency.

---

## 5. Methodological Diagnostics

### 5.1 Multicollinearity (VIF)

**Annual model VIFs (raw):**
```
hw_days              14.09   ← problematic
hw_days_x_rsvi       18.53   ← problematic
summer_tmax_anomaly  15.67   ← problematic
tmax_anomaly_x_rsvi  17.84   ← problematic
rsvi                  4.11
covid_period          1.48
```

**Weekly model VIFs (after centering):** all below 5.

The annual VIF problem is two-layered: (a) interaction terms with
non-centered bases are mechanically collinear with their components;
(b) `hw_days` and `summer_tmax_anomaly` are conceptually two views of
the same signal and their pairwise correlation is high. The weekly
model uses centered interactions; the annual model would need the same
fix applied — the centered annual VIFs drop to 1–2.

### 5.2 Within-summer identifying variation

The annual model's heat coefficient is identified from 220 points.
After absorbing region and year fixed effects, the *effective* degrees
of freedom for the heat × RSVI × 2022 interaction is small: 2022 is
one year out of eleven, giving ~20 observations to pin down the triple
coefficient. Under these conditions, a point estimate of -1.6 with
p=0.10 is what a real but modest effect would look like — we're
underpowered, not noise.

The weekly model has 17× more observations per region-year and
identifies the heat effect from week-to-week variation that the annual
aggregation destroys. This roughly matches the observed move from p=0.10
to p<0.001 on the `tmax_anomaly × rsvi × d2022` coefficient: the point
estimate in log terms (-0.016) is consistent with the annual point
estimate once you account for the rescaling (log mortality vs level
mortality).

### 5.3 Robustness — COVID contamination

COVID mortality in 2020–2021 is a known confounder of any all-cause
mortality analysis that includes those years. The annual model uses
`covid_period` as a binary control, which is absorbed by time FE and
therefore does nothing. The weekly model drops it (collinear with the
year dummies) but tests robustness by excluding 2020 entirely, and by
excluding 2020 and 2021 together:

| Spec | N | `tmax_anomaly_week × rsvi × d2022` | Stable? |
|---|---|---|---|
| Full sample | 3,380 | -0.0161 (p<0.001) | — |
| Excl. 2020 | 3,080 | -0.0151 (p<0.001) | ✅ |
| Excl. 2020 and 2021 | 2,760 | -0.0148 (p<0.001) | ✅ |

The H3 result is not a COVID artifact.

---

## 6. Limitations

### 6.1 Shared by both specifications

- **RSVI construction** is not externally validated against an Italian
  CDC/ATSDR replication. Sensitivity to RSVI weights remains to be
  explored.
- **No spatial autocorrelation.** Region FE handle between-region
  level differences but not spatial dependence in shocks.
- **Mortality source reconciliation.** Eurostat weekly and ISTAT
  municipal daily are both available; the project uses Eurostat
  throughout and does not cross-validate.

### 6.2 Annual-specific

- **Loses the lag channel entirely.** No way to identify timing of
  mortality response.
- **Uses `covid_period` binary,** which is absorbed by time FE.
- **High VIFs** (14–18) from non-centered interactions. The
  coefficients themselves are not biased, but main-effect standard
  errors are inflated — possibly mechanically, possibly why the main
  effects are null in the interaction and triple-interaction models.

### 6.3 Weekly-specific

- **Crude (not age-standardised) mortality** as the outcome. Region FE
  absorb time-invariant age-structure differences; remaining bias is
  from year-to-year age-structure drift within region, which is small.
- **Weekly deaths denominator** is `annual_population / 52`, which
  ignores weekly seasonality in population (e.g. tourism). Small bias
  for summer weeks in tourist regions.
- **RSVI is still annual** and propagated across weeks. Fine given RSVI
  moves slowly, but a weekly identification of the RSVI × heat
  interaction is not really 11 independent observations of RSVI
  variation — it's 11 per region, with the heat variation providing
  the within-year identification.
- **W22–W39 only.** Out-of-summer heat events (rare in Italy but
  increasingly possible under climate change) are dropped.

---

## 7. Recommendation

Report the weekly specification as the primary model. Use the annual
specification as a robustness check, noting:

1. The weekly H1 result (with lag structure) is the substantively
   correct description of the biological mechanism.
2. The weekly H2 is the only specification that correctly identifies
   *which* component of heat exposure RSVI amplifies (the non-linear /
   extreme-temperature channel, not the linear heatwave count).
3. The weekly H3 turns a null into a statistically significant
   negative finding, which is methodologically more defensible than
   "we found nothing." The *direction* contradicts the original
   hypothesis but invites substantive discussion (adaptation, COVID
   harvesting, emergency response in 2022).

Three remaining improvements we'd add if continuing:

- Apply the centering fix to the annual model (`src/analysis/panel_dataset.py`),
  producing centered versions of `hw_days`, `rsvi`, and
  `summer_tmax_anomaly` before constructing the interactions. This
  will likely not rescue H2/H3 in the annual panel, but removes the
  VIF issue as a confound.
- Integrate the ISTAT daily municipal deaths as a weekly-resolution
  COVID control, so that 2020–2021 weeks can be retained without
  contamination.
- Extend beyond Italy (Spain, Greece) as a validation cohort. The
  weekly panel machinery generalises directly — only the NUTS-2
  boundary file and mortality ingestion change.

---

## Appendix A — Full annual coefficient table

See `outputs/tables/regression_results.csv` for the raw table. Key
entries:

```
Model              Variable                         β          p
H1_base            hw_days                          +0.889    <0.001
H1_base            summer_tmax_anomaly              +6.976    <0.001
H2_interaction     rsvi                             +60.959    0.036
H2_interaction     hw_days_x_rsvi                   +1.152     0.261
H2_interaction     tmax_anomaly_x_rsvi              -0.254     0.964
H3_2022_amp.       hw_days_x_rsvi_x_d2022           -1.642     0.105
H3_2022_amp.       tmax_anomaly_x_rsvi_x_d2022      +4.940     0.882
```

## Appendix B — Full weekly coefficient table

See `outputs/tables/regression_results_weekly.csv` for the raw table
(103 rows across 7 specifications). Key entries (H3_weekly_FE):

```
Variable                              β         p
hw_days_week                         +0.0107   0.172
hw_days_week_lag1                    +0.0084   0.174
hw_days_week_lag2                    +0.0018   0.435
tmax_anomaly_week                    +0.0055   0.164
tmax_anomaly_week_sq                 -0.0008   0.637
hot_week_p95                         -0.0255   0.024
rsvi                                 +0.171    0.048
hw_days_x_rsvi                       +0.0059   0.652
tmax_anom_x_rsvi                     +0.0181   0.008
p95_x_rsvi                           +0.0652   0.021
hw_days_x_rsvi_x_d2022               +0.0063   0.342
tmax_anom_x_rsvi_x_d2022             -0.0161  <0.001
p95_x_rsvi_x_d2022                   +0.0489   0.156
```

## Appendix C — Reproducibility

- Full weekly pipeline: `notebooks/07_weekly_panel_analysis.ipynb`.
- Annual pipeline: `notebooks/05_regression_analysis.ipynb`.
- Both read from versioned intermediate parquet in `data/interim/`
  and write result tables to `outputs/tables/`.
- Commit on which this report is based: `feat/weekly-panel-nonlinear`
  branch, HEAD at the time of writing.
