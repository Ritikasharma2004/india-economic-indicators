# India Economic Indicators — Analysis and Forecasting

Analysis of 11 World Bank macroeconomic indicators for India, 1960–2025, with a
walk-forward forecasting study of annual GDP growth.

**Headline result:** no machine learning model beats a trivial baseline. The
long-run mean forecast (RMSE 2.97) outperforms Ridge, Random Forest and
Gradient Boosting on both feature sets. That finding, and why it happens, is
the point of the project — see [reports/findings.md](reports/findings.md).

---

## What this project does

| Stage | Script | Output |
|---|---|---|
| Ingest | `src/extract.py` | Raw JSON per indicator from the World Bank API |
| Clean | `src/transform.py` | Tidy panel + data quality report |
| Explore | `notebooks/01_exploratory_analysis.ipynb` | Full EDA walkthrough with outputs |
| Explore | `src/eda.py` | 5 figures + EDA findings |
| Model | `src/forecast.py` | Walk-forward scoreboard, 5 models × 2 feature sets |
| SQL | `src/load_db.py`, `src/run_sql.py` | SQLite database + 8 analysis queries |
| Dashboard | `dashboard/app.py` | Interactive Streamlit app |

## Key findings

1. **Post-1991 growth is higher *and* steadier** — 6.14% vs 4.23%, with
   volatility falling from 3.36 to 2.80.
2. **India moved from agriculture to services, skipping industry** —
   agriculture fell 40.9% → 17.6% of GDP; almost all of that share went to
   services, while industry stayed flat.
3. **Trade openness tracks growth** — the most open third of post-1991 years
   grew 1.67pp faster than the least open third.
4. **Inflation does not trade off cleanly against growth** — high-inflation
   years averaged higher growth than moderate-inflation years. Correlation:
   −0.14.
5. **The mean baseline beats every model.** Adding ten macro indicators made
   forecasts worse, not better. The exploratory analysis predicted this before
   any model was fitted: GDP growth is stationary (ADF statistic −7.65,
   p < 0.0001) with a **lag-1 autocorrelation of 0.026**, so there is no serial
   dependence for lag features to capture.

## Tech stack

**Python** (pandas, NumPy, scikit-learn, statsmodels) · **SQL** (SQLite,
portable to PostgreSQL) · **Streamlit** and **Plotly** (interactive dashboard) ·
**matplotlib**/**seaborn** (static figures) · **pytest** (13 tests, including
leakage checks)

## Quick start

```bash
pip install -r requirements.txt
```

Run the full pipeline:

```bash
python src/extract.py && python src/transform.py && python src/eda.py && python src/forecast.py && python src/load_db.py && python src/run_sql.py
```

Launch the dashboard:

```bash
streamlit run dashboard/app.py
```

Run the tests:

```bash
python -m pytest tests/ -q
```

## Project structure

```
india-economic-indicators/
├── data/
│   ├── raw/                  API responses, unmodified, with pull date
│   ├── processed/            analysis tables + model predictions
│   └── india_economy.db      SQLite database
├── src/
│   ├── extract.py            World Bank API ingestion
│   ├── transform.py          cleaning, merging, quality report
│   ├── eda.py                figures + EDA findings
│   ├── forecast.py           walk-forward model comparison
│   ├── load_db.py            build SQLite database
│   └── run_sql.py            execute SQL, save results
├── notebooks/
│   └── 01_exploratory_analysis.ipynb   EDA walkthrough, outputs included
├── sql/analysis.sql          8 analysis queries (CTEs, window functions)
├── dashboard/app.py          Streamlit dashboard
├── reports/
│   ├── findings.md           executive summary
│   ├── eda_findings.md       exploratory analysis
│   ├── data_quality.md       coverage and missingness
│   ├── sql_results.md        query output
│   └── figures/              5 PNG charts
└── tests/test_pipeline.py    13 tests
```

## Notes on method

**Why the analysis window starts at 1991.** Indicator coverage is uneven before
then — unemployment is only reported from 1991, FDI from 1970, national
accounts from 1960. Starting at 1991 keeps every indicator present rather than
letting models silently drop rows.

**Why features are lagged.** The 2024 inflation figure is not known when the
2024 GDP figure is being forecast. Using it would leak information the
forecaster would not have had. Every feature is lagged by at least one year,
and `tests/test_pipeline.py` verifies this rather than assuming it.

**Why walk-forward validation.** A random train/test split on time series
trains on the future to predict the past. Validation here trains on all years
up to *t−1*, predicts year *t*, then steps forward.

## Limitations

- Annual frequency gives a small sample (62 usable years univariate, 32
  multivariate). Baselines are reported alongside every model for this reason.
- India rebased its GDP series to 2011–12 prices; long-run comparisons cross
  that definitional break.
- The most recent year is a provisional estimate, revised in later releases.
- Every relationship reported is correlational. Nothing here supports a causal
  claim.

## Data source

World Bank World Development Indicators, accessed via the public API
(`api.worldbank.org/v2`). No API key required. Data is licensed under
[CC BY 4.0](https://datacatalog.worldbank.org/public-licenses).

---

Built by **Ritika Sharma** — [LinkedIn](https://www.linkedin.com/in/ritikasharma04/) ·
[GitHub](https://github.com/Ritikasharma2004)
