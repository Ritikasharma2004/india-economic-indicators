"""
Load the processed indicators into a SQLite database so the analysis can be
run in SQL as well as pandas.

SQLite is used because it needs no server and the .db file can be committed
alongside the project, but the queries in sql/ are standard SQL and run on
PostgreSQL or MySQL with no changes.
"""

import sqlite3
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
PROCESSED_DIR = ROOT / "data" / "processed"
DB_PATH = ROOT / "data" / "india_economy.db"

SCHEMA = """
DROP TABLE IF EXISTS indicators;
CREATE TABLE indicators (
    year          INTEGER NOT NULL,
    indicator     TEXT    NOT NULL,
    value         REAL    NOT NULL,
    PRIMARY KEY (year, indicator)
);

DROP TABLE IF EXISTS indicator_meta;
CREATE TABLE indicator_meta (
    indicator     TEXT PRIMARY KEY,
    display_name  TEXT NOT NULL,
    unit          TEXT NOT NULL,
    category      TEXT NOT NULL
);

CREATE INDEX idx_indicators_year ON indicators(year);
CREATE INDEX idx_indicators_name ON indicators(indicator);
"""

META = [
    ("gdp_growth_pct", "GDP growth", "annual %", "growth"),
    ("gdp_per_capita_growth_pct", "GDP per capita growth", "annual %", "growth"),
    ("inflation_cpi_pct", "Inflation (CPI)", "annual %", "prices"),
    ("unemployment_pct", "Unemployment", "% of labour force", "labour"),
    ("exports_pct_gdp", "Exports", "% of GDP", "trade"),
    ("imports_pct_gdp", "Imports", "% of GDP", "trade"),
    ("trade_balance_pct_gdp", "Trade balance", "% of GDP", "trade"),
    ("fdi_inflow_pct_gdp", "FDI net inflows", "% of GDP", "investment"),
    ("agriculture_pct_gdp", "Agriculture value added", "% of GDP", "structure"),
    ("industry_pct_gdp", "Industry value added", "% of GDP", "structure"),
    ("services_pct_gdp", "Services value added", "% of GDP", "structure"),
]


def main() -> None:
    long_df = pd.read_csv(PROCESSED_DIR / "india_indicators_long.csv")

    with sqlite3.connect(DB_PATH) as conn:
        conn.executescript(SCHEMA)
        long_df.to_sql("indicators", conn, if_exists="append", index=False)
        pd.DataFrame(
            META, columns=["indicator", "display_name", "unit", "category"]
        ).to_sql("indicator_meta", conn, if_exists="append", index=False)

        rows = conn.execute("SELECT COUNT(*) FROM indicators").fetchone()[0]
        span = conn.execute("SELECT MIN(year), MAX(year) FROM indicators").fetchone()

    print(f"Loaded {rows} rows into {DB_PATH.name} covering {span[0]}-{span[1]}")


if __name__ == "__main__":
    main()
