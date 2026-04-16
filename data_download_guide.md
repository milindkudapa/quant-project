# Data Download Guide

Step-by-step instructions for downloading all data needed for the project.

**Total estimated time:** ~1–2 hours (mostly waiting for ERA5 downloads)

---

## Quick Reference

| Data | Source | Free? | Download Method | Save To |
|---|---|---|---|---|
| NUTS-2 Boundaries | Eurostat GISCO | ✅ Yes | Automated (script) | `data/raw/boundaries/` |
| ERA5 Climate (Recommended) | Earthmover (AWS) | ✅ Yes | Automated (script/API) | `data/raw/climate/` |
| ERA5-Land (Alternative) | Copernicus CDS | ✅ Yes (registration) | Automated (API) | `data/raw/climate/` |
| Mortality Data | Eurostat | ✅ Yes | Manual download | `data/raw/mortality/` |
| Age Structure | ISTAT I.Stat | ✅ Yes | Manual download | `data/raw/socioeconomic/` |
| GDP per Capita | Eurostat | ✅ Yes | Manual download | `data/raw/socioeconomic/` |
| Poverty Rates | ISTAT | ✅ Yes | Manual download | `data/raw/socioeconomic/` |
| Population Density | Eurostat | ✅ Yes | Manual download | `data/raw/socioeconomic/` |

---

## 1. NUTS-2 Boundaries (Automated)

**Source:** Eurostat GISCO  
**What:** Shapefile of Italian NUTS-2 region polygons (2021 classification)

### Steps

This is fully automated. Just run:

```bash
uv run python -c "from src.data.nuts2_boundaries import setup_boundaries; setup_boundaries()"
```

Or use Notebook 01, which does this for you.

**Verify:**
```bash
ls data/raw/boundaries/
# Should see .shp, .shx, .dbf, .prj files
```

---

## 2. ERA5 Climate Data (Recommended — Fast & Simple)

**Source:** Earthmover (AWS Zarr)  
**What:** Hourly 2m temperature and dewpoint for Italy, June–September, 2012–2022.  
**Benefit:** No account registration needed. Downloading 11 years takes **~5 minutes** instead of hours.

### Steps

This is automated via a dedicated script that streams the data directly from AWS and saves only the Italy subset locally.

**Download all years:**
```bash
uv run python scripts/download_data.py --source earthmover
```

**Verify:**
```bash
ls -lh data/raw/climate/
# Should see: era5_earthmover_italy_2012.nc through era5_earthmover_italy_2022.nc
```

---

## 3. ERA5-Land Climate Data (Alternative — CDS API)

**Source:** Copernicus Climate Data Store (CDS)  
**What:** Hourly 2m temperature and dewpoint for Italy, June–September, 2012–2022  
**Size:** ~500 MB–1 GB per year, ~5–10 GB total
**Note:** Use this only if the Earthmover source is unavailable. Requires registration.

### Step 3.1 — Register for CDS Account

1. Go to **https://cds.climate.copernicus.eu/**
2. Click **"Register"** (top right)
3. Fill in the registration form — use your Columbia email
4. Verify your email and log in

### Step 3.2 — Accept the ERA5-Land Licence

1. Go to **https://cds.climate.copernicus.eu/datasets/reanalysis-era5-land**
2. Scroll down to the **"Terms of use"** section
3. Click **"Accept"** on the licence agreement
4. ⚠️ You MUST accept this or the API downloads will fail

### Step 3.3 — Get Your API Key

1. Log into CDS and go to your **profile page**: https://cds.climate.copernicus.eu/profile
2. Scroll down to the **"Personal Access Token"** section
3. Click **"Generate Token"** if you don't have one
4. Copy the full token string

### Step 3.4 — Configure the API Key

**Option A (recommended):** Set as environment variable:
```bash
export CDS_API_KEY="your-token-here"
```

Add this to your `~/.zshrc` so it persists:
```bash
echo 'export CDS_API_KEY="your-token-here"' >> ~/.zshrc
source ~/.zshrc
```

**Option B:** Create a `~/.cdsapirc` file:
```bash
cat > ~/.cdsapirc << EOF
url: https://cds.climate.copernicus.eu/api
key: your-token-here
EOF
```

**Option C:** Add directly to `config/config.yaml`:
```yaml
cds_api:
  url: "https://cds.climate.copernicus.eu/api"
  key: "your-token-here"
```

### Step 3.5 — Download the Data

**Download all years:**
```bash
uv run python scripts/download_data.py --source era5
```

⏱ **Each year takes ~15–30 minutes.** The full download (11 years) may take 3–6 hours. You can leave it running overnight.

**Verify:**
```bash
ls -lh data/raw/climate/
# Should see: era5land_italy_2012.nc through era5land_italy_2022.nc
```

---

## 3. Mortality Data (Manual Download)

**Source:** Eurostat  
**What:** Weekly deaths by NUTS-2 region, age group, and sex  
**Dataset code:** `demo_r_mwk2_ts`

### Steps

1. Go to: **https://ec.europa.eu/eurostat/databrowser/view/demo_r_mwk2_ts/default/table**

2. Configure the filters:
   - **GEO:** Select all Italian NUTS-2 regions (ITC1 through ITG2)
     - Click on "GEO" → search "IT" → check all the NUTS-2 codes
   - **TIME:** 2012W01 through 2022W52
   - **AGE:** Select all age groups, OR select: `TOTAL`, `Y_LT65`, `Y65-69`, `Y70-74`, `Y75-79`, `Y80-84`, `Y_GE85`
   - **SEX:** T (total), M (male), F (female)

3. Click **"Download"** (top right) → **"Full dataset (CSV)"**

4. Save the file to: `data/raw/mortality/eurostat_weekly_deaths.csv`

### Alternative: ISTAT Daily Mortality

If you need daily data (more granular):

1. Go to: **https://www.istat.it/en/archivio/240401**
   - This is the ISTAT mortality dataset used during COVID monitoring
2. Download the CSV file(s)
3. Save to: `data/raw/mortality/`

### Alternative: Eurostat Annual Deaths

For simpler annual data:

1. Go to: **https://ec.europa.eu/eurostat/databrowser/view/demo_r_magec/default/table**
2. Filters:
   - **GEO:** All Italian NUTS-2
   - **TIME:** 2012–2022
   - **AGE:** All groups
   - **SEX:** Total
3. Download CSV → save to `data/raw/mortality/eurostat_annual_deaths.csv`

---

## 4. Age Structure Data (Manual Download)

**Source:** Eurostat  
**What:** Population by age group per NUTS-2 region  
**Dataset code:** `demo_r_pjangrp3`

### Steps

1. Go to: **https://ec.europa.eu/eurostat/databrowser/view/demo_r_pjangrp3/default/table**

2. Configure filters:
   - **GEO:** All Italian NUTS-2 regions (ITC1–ITG2)
   - **TIME:** 2012–2022
   - **AGE:** `TOTAL`, `Y65-69`, `Y70-74`, `Y75-79`, `Y80-84`, `Y_GE85`
   - **SEX:** T (total)

3. Download CSV → save to: `data/raw/socioeconomic/age_structure.csv`

### Computing the Required Variables

After download, you'll compute:
- **pct_pop_65plus** = (pop 65-69 + 70-74 + 75-79 + 80-84 + 85+) / total × 100
- **pct_pop_75plus** = (pop 75-79 + 80-84 + 85+) / total × 100
- **pct_pop_80plus** = (pop 80-84 + 85+) / total × 100

The processing module handles this if columns are named correctly.

---

## 5. GDP per Capita (Manual Download)

**Source:** Eurostat  
**What:** GDP per capita (EUR) by NUTS-2 region  
**Dataset code:** `nama_10r_2gdp`

### Steps

1. Go to: **https://ec.europa.eu/eurostat/databrowser/view/nama_10r_2gdp/default/table**

2. Configure filters:
   - **GEO:** All Italian NUTS-2 regions
   - **TIME:** 2012–2022
   - **UNIT:** EUR_HAB (Euro per inhabitant)

3. Download CSV → save to: `data/raw/socioeconomic/gdp_per_capita.csv`

---

## 6. Poverty Rates (Manual Download)

**Source:** ISTAT  
**What:** Absolute poverty rate by region  

### Steps

1. Go to: **https://dati.istat.it/**

2. Search for: **"povertà assoluta regione"** (absolute poverty by region)

3. Or navigate to:
   - **Theme:** Living conditions and poverty
   - **Dataset:** Absolute poverty - incidence by region

4. Configure:
   - **Territory:** All regions
   - **Years:** 2012–2022
   - **Indicator:** Incidence of absolute poverty (%)

5. Download as CSV → save to: `data/raw/socioeconomic/poverty_rates.csv`

### Alternative: Eurostat SILC Data

1. Go to: **https://ec.europa.eu/eurostat/databrowser/view/ilc_li41/default/table**
2. Filters: Italian NUTS-2, 2012–2022
3. Download → `data/raw/socioeconomic/poverty_eurostat.csv`

---

## 7. Population Density (Manual Download)

**Source:** Eurostat  
**What:** Population density (persons per km²) by NUTS-2  
**Dataset code:** `demo_r_d3dens`

### Steps

1. Go to: **https://ec.europa.eu/eurostat/databrowser/view/demo_r_d3dens/default/table**

2. Configure filters:
   - **GEO:** All Italian NUTS-2 regions
   - **TIME:** 2012–2022

3. Download CSV → save to: `data/raw/socioeconomic/population_density.csv`

---

## 8. (Optional) Disposable Income

**Source:** Eurostat  
**What:** Disposable income per capita by NUTS-2  
**Dataset code:** `nama_10r_2hhinc`

### Steps

1. Go to: **https://ec.europa.eu/eurostat/databrowser/view/nama_10r_2hhinc/default/table**

2. Filters:
   - **GEO:** Italian NUTS-2
   - **TIME:** 2012–2022
   - **UNIT:** EUR_HAB
   - **NA_ITEM:** B6N (net disposable income)

3. Download → `data/raw/socioeconomic/disposable_income.csv`

---

## 9. (Optional) COVID-19 Deaths

For controlling pandemic confounding in 2020–2022.

**Source:** Our World in Data, or ISS (Istituto Superiore di Sanità)

1. **OWID:** https://github.com/owid/covid-19-data/tree/master/public/data
   - Download `owid-covid-data.csv`
   - Filter to Italy
   - Save to: `data/raw/mortality/covid_deaths_italy.csv`

2. **ISS:** https://www.epicentro.iss.it/coronavirus/sars-cov-2-dashboard
   - Download regional COVID death data
   - Save to: `data/raw/mortality/`

---

## Post-Download Checklist

After downloading all data, your directory should look like:

```
data/raw/
├── boundaries/
│   ├── *.shp (+ .shx, .dbf, .prj files)
│
├── climate/
│   ├── era5land_italy_2012.nc
│   ├── era5land_italy_2013.nc
│   ├── ...
│   └── era5land_italy_2022.nc
│
├── mortality/
│   ├── eurostat_weekly_deaths.csv    (or equivalent)
│   └── covid_deaths_italy.csv        (optional)
│
└── socioeconomic/
    ├── age_structure.csv
    ├── gdp_per_capita.csv
    ├── poverty_rates.csv
    └── population_density.csv
```

### Verify with this command:

```bash
find data/raw -type f -not -name '.gitkeep' | sort
```

### Then proceed to processing:

```bash
# Run Notebook 02, or:
uv run python scripts/run_pipeline.py --from-step process_climate
```

---

## Troubleshooting

### ERA5 download fails
- Check your CDS API key is valid: `echo $CDS_API_KEY`
- Verify you accepted the ERA5-Land licence on the CDS website
- Check CDS status: https://cds.climate.copernicus.eu/live/queue

### Eurostat download is slow
- Use the "full dataset" CSV option (faster than interactive table)
- Alternatively, use the Eurostat bulk download: https://ec.europa.eu/eurostat/data/bulkdownload

### ISTAT website is in Italian
- Use Google Translate, or look for the English language toggle (top right)
- The English version: https://www.istat.it/en/

### File format issues
- Eurostat CSVs use tab (`\t`) separation — check the delimiter
- ISTAT files may use `;` as delimiter and `,` as decimal separator
- The processing modules handle common formats automatically
