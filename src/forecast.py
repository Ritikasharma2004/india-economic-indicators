"""
Forecast India's annual GDP growth one year ahead, and check whether any model
actually beats a naive baseline.

Two feature sets are compared:
    univariate    - lagged GDP growth only, usable from 1962 (63 observations)
    multivariate  - lagged GDP growth plus nine other lagged indicators,
                    usable from 1992 (34 observations)

Every feature is lagged by at least one year. The 2024 value of inflation is
not known when the 2024 GDP figure is being forecast, so using it would leak
information the forecaster would not have had.

Validation is walk-forward: train on all years up to t-1, predict year t, step
forward. This respects time order, unlike a random train/test split.
"""

from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_absolute_error, mean_squared_error

ROOT = Path(__file__).resolve().parents[1]
PROCESSED_DIR = ROOT / "data" / "processed"
REPORTS_DIR = ROOT / "reports"
ARTIFACTS_DIR = ROOT / "data" / "processed"

TARGET = "gdp_growth_pct"
LAGS = (1, 2, 3)
MIN_TRAIN_YEARS = 20
RANDOM_STATE = 42


def build_features(df: pd.DataFrame, columns: list[str]) -> pd.DataFrame:
    """Create lagged features for the given columns plus the unlagged target."""
    out = pd.DataFrame(index=df.index)
    out[TARGET] = df[TARGET]

    for col in columns:
        for lag in LAGS:
            out[f"{col}_lag{lag}"] = df[col].shift(lag)

    # A three-year rolling mean of past growth carries the medium-term trend
    # that single lags miss.
    out[f"{TARGET}_roll3"] = df[TARGET].shift(1).rolling(3).mean()

    return out.dropna()


def walk_forward(features: pd.DataFrame, model_name: str) -> pd.DataFrame:
    """Run expanding-window one-step-ahead validation, returning per-year results."""
    X = features.drop(columns=[TARGET])
    y = features[TARGET]
    years = features.index.to_numpy()

    rows = []
    for i in range(MIN_TRAIN_YEARS, len(features)):
        X_train, y_train = X.iloc[:i], y.iloc[:i]
        X_test, y_test = X.iloc[i : i + 1], y.iloc[i]

        if model_name == "naive":
            # Carry last year's actual growth forward. This is the number to beat.
            pred = y_train.iloc[-1]
        elif model_name == "mean":
            # Long-run average growth, the other trivial forecast.
            pred = y_train.mean()
        else:
            model = make_model(model_name)
            model.fit(X_train, y_train)
            pred = float(model.predict(X_test)[0])

        rows.append({"year": int(years[i]), "actual": y_test, "predicted": pred})

    return pd.DataFrame(rows)


def make_model(name: str):
    if name == "ridge":
        return Ridge(alpha=1.0, random_state=None)
    if name == "random_forest":
        return RandomForestRegressor(
            n_estimators=300, max_depth=4, random_state=RANDOM_STATE
        )
    if name == "gradient_boosting":
        return GradientBoostingRegressor(
            n_estimators=200, max_depth=2, learning_rate=0.05, random_state=RANDOM_STATE
        )
    raise ValueError(f"unknown model: {name}")


def score(results: pd.DataFrame) -> dict:
    rmse = float(np.sqrt(mean_squared_error(results["actual"], results["predicted"])))
    mae = float(mean_absolute_error(results["actual"], results["predicted"]))

    # Directional accuracy: did the forecast get the direction of change right?
    # The naive forecast predicts no change at all, so its direction is
    # undefined rather than wrong - reporting it as 0% would misread the model.
    actual_change = results["actual"].diff()
    pred_change = results["predicted"] - results["actual"].shift(1)
    both = pd.concat([actual_change, pred_change], axis=1).dropna()
    both = both[both.iloc[:, 1] != 0]
    direction = (
        float((np.sign(both.iloc[:, 0]) == np.sign(both.iloc[:, 1])).mean())
        if len(both)
        else float("nan")
    )

    return {"rmse": rmse, "mae": mae, "directional_accuracy": direction}


def evaluate(df: pd.DataFrame, feature_cols: list[str], label: str) -> pd.DataFrame:
    features = build_features(df, feature_cols)
    print(f"\n{label}: {len(features)} usable years "
          f"({int(features.index.min())}-{int(features.index.max())}), "
          f"{features.shape[1] - 1} features, "
          f"{len(features) - MIN_TRAIN_YEARS} test points")

    rows = []
    for model_name in ["naive", "mean", "ridge", "random_forest", "gradient_boosting"]:
        results = walk_forward(features, model_name)
        metrics = score(results)
        metrics["model"] = model_name
        metrics["feature_set"] = label
        rows.append(metrics)
        results.to_csv(
            ARTIFACTS_DIR / f"predictions_{label}_{model_name}.csv", index=False
        )

    return pd.DataFrame(rows)[
        ["feature_set", "model", "rmse", "mae", "directional_accuracy"]
    ].sort_values("rmse")


def main() -> None:
    df = pd.read_csv(PROCESSED_DIR / "india_indicators.csv", index_col="year")

    # Univariate: GDP growth has the longest history, so it gets the most data.
    uni = evaluate(df[[TARGET]].dropna(), [TARGET], "univariate")
    print(uni.to_string(index=False))

    # Multivariate: every indicator, restricted to years where all are present.
    multi_cols = [c for c in df.columns if c != "trade_balance_pct_gdp"]
    multi_df = df[multi_cols].dropna()
    multi = evaluate(multi_df, multi_cols, "multivariate")
    print(multi.to_string(index=False))

    scoreboard = pd.concat([uni, multi], ignore_index=True)
    scoreboard.to_csv(PROCESSED_DIR / "model_scoreboard.csv", index=False)

    print("\n" + "=" * 68)
    best = scoreboard.loc[scoreboard["rmse"].idxmin()]
    naive_uni = scoreboard[
        (scoreboard["model"] == "naive") & (scoreboard["feature_set"] == "univariate")
    ].iloc[0]
    print(f"Best overall     : {best['model']} ({best['feature_set']}), RMSE {best['rmse']:.3f}")
    print(f"Naive baseline   : RMSE {naive_uni['rmse']:.3f}")
    improvement = (naive_uni["rmse"] - best["rmse"]) / naive_uni["rmse"] * 100
    print(f"Improvement      : {improvement:+.1f}% vs naive")
    print("=" * 68)


if __name__ == "__main__":
    main()
