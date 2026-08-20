"""
Turn the raw World Bank JSON into one tidy analysis table, and write a data
quality report alongside it.

Output:
    data/processed/india_indicators.csv   one row per year, one column per indicator
    data/processed/india_indicators_long.csv   tidy long format, one row per year-indicator
    reports/data_quality.md               coverage and missingness summary
"""

import json
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
RAW_DIR = ROOT / "data" / "raw"
PROCESSED_DIR = ROOT / "data" / "processed"
REPORTS_DIR = ROOT / "reports"

# Indicators are reported with a long lag in their first years and the most
# recent year is often an estimate; we keep everything and flag it instead of
# silently trimming, so the analysis can decide what to use.
ANALYSIS_START_YEAR = 1991  # liberalisation reforms - the era every indicator covers


def load_raw() -> pd.DataFrame:
    """Read every raw JSON file into a single long dataframe."""
    frames = []

    for path in sorted(RAW_DIR.glob("*.json")):
        payload = json.loads(path.read_text(encoding="utf-8"))
        records = payload["records"]

        frame = pd.DataFrame(
            [
                {
                    "year": int(r["date"]),
                    "indicator": path.stem,
                    "value": r["value"],
                }
                for r in records
            ]
        )
        frame["indicator_code"] = payload["indicator_code"]
        frame["pulled_on"] = payload["pulled_on"]
        frames.append(frame)

    return pd.concat(frames, ignore_index=True)


def build_quality_report(long_df: pd.DataFrame, wide_df: pd.DataFrame) -> str:
    """Summarise coverage, missingness and duplicates as markdown."""
    coverage = (
        long_df.dropna(subset=["value"])
        .groupby("indicator")["year"]
        .agg(first_year="min", last_year="max", observations="count")
        .sort_values("observations")
    )

    analysis_window = wide_df.loc[wide_df.index >= ANALYSIS_START_YEAR]
    missing_in_window = analysis_window.isna().sum()

    duplicates = long_df.duplicated(subset=["indicator", "year"]).sum()

    lines = [
        "# Data Quality Report",
        "",
        f"Source: World Bank World Development Indicators API (country = IND).",
        f"Pulled on: {long_df['pulled_on'].iloc[0]}",
        "",
        "## Coverage by indicator",
        "",
        coverage.to_markdown(),
        "",
        "## Why the analysis starts at 1991",
        "",
        "Coverage is uneven before 1991. Unemployment is only reported from 1991 "
        "and FDI from 1970, while the national-accounts indicators go back to 1960. "
        "Using the full 1960-2025 range would mean every model silently drops rows "
        "wherever unemployment is absent, so the shared analysis window starts at "
        f"{ANALYSIS_START_YEAR} where all {wide_df.shape[1]} indicators are present.",
        "",
        "## Missing values inside the analysis window "
        f"({ANALYSIS_START_YEAR}-{int(wide_df.index.max())})",
        "",
        missing_in_window.to_frame("missing_values").to_markdown(),
        "",
        "## Structural checks",
        "",
        f"- Duplicate (indicator, year) pairs: {duplicates}",
        f"- Years in analysis window: {len(analysis_window)}",
        f"- Indicators: {wide_df.shape[1]}",
        "",
        "## Known limitations",
        "",
        "- Annual frequency gives a small sample. The 1991-2025 window is 35 "
        "observations, which is enough for trend description but thin for "
        "machine learning. Model results are reported against a naive baseline "
        "for this reason.",
        "- The most recent year is an estimate and is revised in later releases.",
        "- Indicator definitions changed over time (notably the 2011-12 GDP "
        "series rebasing), so long-run comparisons carry a definitional break.",
    ]
    return "\n".join(lines)


def main() -> None:
    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)

    long_df = load_raw()

    wide_df = (
        long_df.pivot_table(index="year", columns="indicator", values="value")
        .sort_index()
    )
    wide_df.columns.name = None

    # Trade balance is not published as its own indicator but falls straight out
    # of exports and imports, and it is the figure the trade analysis needs.
    wide_df["trade_balance_pct_gdp"] = (
        wide_df["exports_pct_gdp"] - wide_df["imports_pct_gdp"]
    )

    wide_df.to_csv(PROCESSED_DIR / "india_indicators.csv")

    tidy = (
        wide_df.reset_index()
        .melt(id_vars="year", var_name="indicator", value_name="value")
        .dropna(subset=["value"])
        .sort_values(["indicator", "year"])
    )
    tidy.to_csv(PROCESSED_DIR / "india_indicators_long.csv", index=False)

    report = build_quality_report(long_df, wide_df)
    (REPORTS_DIR / "data_quality.md").write_text(report, encoding="utf-8")

    print(f"Wide table : {wide_df.shape[0]} years x {wide_df.shape[1]} indicators")
    print(f"Long table : {len(tidy)} rows")
    print(f"Years      : {int(wide_df.index.min())}-{int(wide_df.index.max())}")
    print(f"\nWrote {PROCESSED_DIR / 'india_indicators.csv'}")
    print(f"Wrote {REPORTS_DIR / 'data_quality.md'}")


if __name__ == "__main__":
    main()
