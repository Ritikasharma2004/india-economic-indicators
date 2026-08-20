# Data Quality Report

Source: World Bank World Development Indicators API (country = IND).
Pulled on: 2026-08-20

## Coverage by indicator

| indicator                 |   first_year |   last_year |   observations |
|:--------------------------|-------------:|------------:|---------------:|
| unemployment_pct          |         1991 |        2025 |             35 |
| fdi_inflow_pct_gdp        |         1970 |        2025 |             56 |
| gdp_per_capita_growth_pct |         1961 |        2025 |             65 |
| gdp_growth_pct            |         1961 |        2025 |             65 |
| agriculture_pct_gdp       |         1960 |        2025 |             66 |
| imports_pct_gdp           |         1960 |        2025 |             66 |
| industry_pct_gdp          |         1960 |        2025 |             66 |
| exports_pct_gdp           |         1960 |        2025 |             66 |
| inflation_cpi_pct         |         1960 |        2025 |             66 |
| services_pct_gdp          |         1960 |        2025 |             66 |

## Why the analysis starts at 1991

Coverage is uneven before 1991. Unemployment is only reported from 1991 and FDI from 1970, while the national-accounts indicators go back to 1960. Using the full 1960-2025 range would mean every model silently drops rows wherever unemployment is absent, so the shared analysis window starts at 1991 where all 11 indicators are present.

## Missing values inside the analysis window (1991-2025)

|                           |   missing_values |
|:--------------------------|-----------------:|
| agriculture_pct_gdp       |                0 |
| exports_pct_gdp           |                0 |
| fdi_inflow_pct_gdp        |                0 |
| gdp_growth_pct            |                0 |
| gdp_per_capita_growth_pct |                0 |
| imports_pct_gdp           |                0 |
| industry_pct_gdp          |                0 |
| inflation_cpi_pct         |                0 |
| services_pct_gdp          |                0 |
| unemployment_pct          |                0 |
| trade_balance_pct_gdp     |                0 |

## Structural checks

- Duplicate (indicator, year) pairs: 0
- Years in analysis window: 35
- Indicators: 11

## Known limitations

- Annual frequency gives a small sample. The 1991-2025 window is 35 observations, which is enough for trend description but thin for machine learning. Model results are reported against a naive baseline for this reason.
- The most recent year is an estimate and is revised in later releases.
- Indicator definitions changed over time (notably the 2011-12 GDP series rebasing), so long-run comparisons carry a definitional break.