"""
Tests for the indicator pipeline.

The tests that matter here are the leakage checks. A forecasting project that
accidentally lets a model see the year it is predicting will report excellent
scores and be worthless, and that failure is silent unless it is tested for.
"""

import sys
from pathlib import Path

import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

import forecast  # noqa: E402

PROCESSED = ROOT / "data" / "processed"


@pytest.fixture(scope="module")
def wide() -> pd.DataFrame:
    return pd.read_csv(PROCESSED / "india_indicators.csv", index_col="year")


@pytest.fixture(scope="module")
def long() -> pd.DataFrame:
    return pd.read_csv(PROCESSED / "india_indicators_long.csv")


# --- data integrity -------------------------------------------------------

def test_no_duplicate_year_indicator_pairs(long):
    assert not long.duplicated(subset=["year", "indicator"]).any()


def test_years_are_contiguous(wide):
    years = sorted(wide.index)
    assert years == list(range(years[0], years[-1] + 1))


def test_no_null_values_in_long_form(long):
    assert long["value"].notna().all()


def test_trade_balance_is_exports_minus_imports(wide):
    expected = wide["exports_pct_gdp"] - wide["imports_pct_gdp"]
    pd.testing.assert_series_equal(
        wide["trade_balance_pct_gdp"], expected, check_names=False
    )


def test_sector_shares_are_plausible(wide):
    """Value-added shares exclude taxes less subsidies, so they sum near but
    not exactly to 100."""
    total = wide[
        ["agriculture_pct_gdp", "industry_pct_gdp", "services_pct_gdp"]
    ].dropna().sum(axis=1)
    assert total.between(85, 105).all()


def test_growth_values_are_in_a_sane_range(wide):
    growth = wide["gdp_growth_pct"].dropna()
    assert growth.between(-15, 15).all()


# --- leakage -------------------------------------------------------------

def test_lag_features_use_only_past_values(wide):
    """A lag-1 feature for year Y must equal the raw value from year Y-1."""
    df = wide[["gdp_growth_pct"]].dropna()
    features = forecast.build_features(df, ["gdp_growth_pct"])

    for year in features.index[:10]:
        for lag in forecast.LAGS:
            assert features.loc[year, f"gdp_growth_pct_lag{lag}"] == pytest.approx(
                df.loc[year - lag, "gdp_growth_pct"]
            )


def test_no_feature_equals_the_target(wide):
    """No column may be an unlagged copy of what is being predicted."""
    df = wide[["gdp_growth_pct"]].dropna()
    features = forecast.build_features(df, ["gdp_growth_pct"])
    target = features[forecast.TARGET]

    for column in features.drop(columns=[forecast.TARGET]).columns:
        assert not features[column].equals(target), f"{column} leaks the target"


def test_rolling_feature_excludes_current_year(wide):
    """The 3-year rolling mean must be built from years strictly before Y."""
    df = wide[["gdp_growth_pct"]].dropna()
    features = forecast.build_features(df, ["gdp_growth_pct"])

    year = features.index[5]
    expected = df.loc[[year - 3, year - 2, year - 1], "gdp_growth_pct"].mean()
    assert features.loc[year, "gdp_growth_pct_roll3"] == pytest.approx(expected)


def test_walk_forward_predicts_each_year_once(wide):
    df = wide[["gdp_growth_pct"]].dropna()
    features = forecast.build_features(df, ["gdp_growth_pct"])
    results = forecast.walk_forward(features, "naive")

    assert not results["year"].duplicated().any()
    assert len(results) == len(features) - forecast.MIN_TRAIN_YEARS
    assert results["year"].is_monotonic_increasing


def test_naive_baseline_carries_previous_actual_forward(wide):
    """The naive forecast for year Y must be the actual value from Y-1."""
    df = wide[["gdp_growth_pct"]].dropna()
    features = forecast.build_features(df, ["gdp_growth_pct"])
    results = forecast.walk_forward(features, "naive").set_index("year")

    for year in results.index[:5]:
        assert results.loc[year, "predicted"] == pytest.approx(
            features.loc[year - 1, forecast.TARGET]
        )


# --- scoring -------------------------------------------------------------

def test_scoreboard_covers_every_model_and_feature_set():
    scoreboard = pd.read_csv(PROCESSED / "model_scoreboard.csv")
    expected_models = {"naive", "mean", "ridge", "random_forest", "gradient_boosting"}

    for feature_set in ["univariate", "multivariate"]:
        subset = scoreboard[scoreboard["feature_set"] == feature_set]
        assert set(subset["model"]) == expected_models


def test_reported_rmse_matches_the_saved_predictions():
    """Guards against the scoreboard drifting out of sync with the predictions."""
    scoreboard = pd.read_csv(PROCESSED / "model_scoreboard.csv")

    for _, row in scoreboard.iterrows():
        path = PROCESSED / f"predictions_{row['feature_set']}_{row['model']}.csv"
        preds = pd.read_csv(path)
        rmse = ((preds["predicted"] - preds["actual"]) ** 2).mean() ** 0.5
        assert rmse == pytest.approx(row["rmse"], rel=1e-6)
