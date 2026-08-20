# India Economic Indicators — Findings

**Question.** What has driven India's growth since 1960, and can annual GDP
growth be forecast one year ahead from published macro indicators?

**Data.** World Bank World Development Indicators API, 11 indicators for India,
1960–2025, 683 observations. Pulled live from the API; raw JSON retained.

---

## 1. Liberalisation changed both the level and the reliability of growth

| Era | Years | Mean growth | Volatility (sd) |
|---|---|---|---|
| 1961–1990 | 30 | 4.23% | 3.36 |
| 1991–2025 | 35 | 6.14% | 2.80 |

Growth after 1991 is **1.91 percentage points higher on average**, and it is
also **less volatile** — the standard deviation falls from 3.36 to 2.80.

This second half is the part usually left out. The pre-liberalisation economy
did not merely grow more slowly; it grew less predictably. Decade-level numbers
make the point sharply: the 1970s averaged 2.93% growth with volatility of
3.94, while the 2010s averaged 6.64% with volatility of 1.40. Measured as
growth per unit of risk, the 2010s scored 4.74 against the 1970s' 0.74 — a
six-fold improvement in the stability of growth.

## 2. India moved from agriculture to services, skipping industry

| Decade | Agriculture | Industry | Services |
|---|---|---|---|
| 1960s | 40.9% | 21.2% | 36.9% |
| 1990s | 25.7% | 27.3% | 38.5% |
| 2020s | 17.6% | 25.5% | 48.0% |

Agriculture's share of GDP fell by 23 percentage points across the period.
Almost all of that share moved to services, not to industry — the industry
share is essentially flat, peaking at 29.2% in the 2000s and slipping back
since.

This is the opposite of the path taken by most industrialising economies,
where manufacturing absorbs labour leaving agriculture before services expand.
It matters for policy because services growth absorbs far fewer low-skilled
workers per unit of output than manufacturing does.

## 3. Trade openness tracks higher growth

| Openness (exports + imports, % of GDP) | Years | Avg growth |
|---|---|---|
| Under 30% | 12 | 5.39% |
| 30–45% | 9 | 5.70% |
| Over 45% | 14 | 7.06% |

Post-1991 years in the most open third of the sample grew **1.67 percentage
points faster** than the least open third.

The direction of causation is not established here. Openness rises when global
demand is strong, and strong global demand also lifts Indian growth directly,
so part of this gap is common cause rather than effect. The association is
worth reporting; a causal claim would need an instrument this dataset does not
contain.

## 4. Inflation shows no clean relationship with growth

| Inflation band | Years | Avg growth |
|---|---|---|
| Low (under 5%) | 22 | 5.67% |
| Moderate (5–10%) | 27 | 4.72% |
| High (10%+) | 16 | 5.59% |

There is no monotonic pattern. High-inflation years averaged *higher* growth
than moderate-inflation years. The correlation between the two series across
1991–2025 is −0.14, which is negligible.

This is a negative result and it is reported as one. The intuition that
inflation and growth trade off cleanly does not survive contact with 65 years
of Indian annual data.

## 5. No forecasting model beats a trivial baseline

Models were validated **walk-forward** — train on all years up to *t−1*,
predict year *t*, step forward — with every feature lagged by at least one year
so no model sees information unavailable at forecast time.

| Feature set | Model | RMSE | MAE |
|---|---|---|---|
| Univariate | **Mean (baseline)** | **2.97** | 2.30 |
| Univariate | Ridge | 3.19 | 2.51 |
| Univariate | Random Forest | 3.28 | 2.59 |
| Univariate | Gradient Boosting | 3.54 | 2.67 |
| Univariate | Naive (baseline) | 3.70 | 2.31 |
| Multivariate | Mean (baseline) | 3.91 | 2.30 |
| Multivariate | Random Forest | 4.03 | 2.59 |
| Multivariate | Ridge | 7.42 | 5.45 |

**The long-run average wins.** Predicting "growth will be about 6%" every year
beats every machine learning model tried, on both feature sets. Adding ten
macro indicators made forecasts *worse*, not better — the multivariate models
are uniformly less accurate than their univariate counterparts.

Three reasons, all properties of the data rather than the models:

1. **The sample is small.** The univariate setup has 62 usable years; the
   multivariate one has 32. Tree ensembles need far more to learn stable
   structure, and Ridge with 31 features over 32 observations is fitting noise
   — its RMSE of 7.42 is worse than guessing.
2. **Growth is stationary white noise around a constant mean.** An Augmented
   Dickey-Fuller test rejects non-stationarity decisively (statistic −7.65,
   p < 0.0001), and the **lag-1 autocorrelation is 0.026** — effectively zero.
   Last year's growth carries almost no information about this year's. This is
   the single most important number in the project: it means lag features
   cannot work, because there is no serial dependence for them to capture.
   Shocks — 1965, 1979, 2020 — are severe but do not persist, which is also
   why the naive carry-forward forecast is the *worst* baseline while the mean
   is the best.
3. **The predictors are weak.** The strongest same-year correlation with growth
   across 1991–2025 is −0.24 (trade balance). Lagging these variables, as
   honest forecasting requires, weakens them further.

The exploratory analysis reached this conclusion *before* any model was fitted
(see [notebooks/01_exploratory_analysis.ipynb](../notebooks/01_exploratory_analysis.ipynb),
section 10). The modelling step confirmed a prediction the data had already
made, which is why the baseline result is a finding rather than a failed
experiment.

### What would actually improve this

- **Quarterly data** instead of annual would roughly quadruple the sample.
- **Genuine leading indicators** — PMI, index of industrial production, credit
  growth, monsoon rainfall — carry signal that annual national accounts do not.

Neither is available from this API. Reporting a machine learning model as the
winner here would have required either dropping the baseline or using unlagged
features, and both would be misleading.

---

## Limitations

- **Annual frequency, small sample.** 65 observations is enough to describe
  trends, thin for modelling. This is why baselines are reported alongside
  every model.
- **Definitional breaks.** India rebased its GDP series to 2011–12 prices;
  long-run comparisons cross that break.
- **Provisional recent years.** The most recent value is an estimate and gets
  revised in later releases.
- **Association, not causation.** Every relationship reported here is
  correlational. Nothing in this dataset supports a causal claim.

## Reproducing this analysis

```bash
pip install -r requirements.txt
python src/extract.py         # pull from the World Bank API
python src/transform.py       # clean, merge, quality report
python src/eda.py             # figures and EDA findings
python src/forecast.py        # walk-forward model comparison
python src/load_db.py         # build SQLite database
python src/run_sql.py         # run SQL analysis
streamlit run dashboard/app.py
```
