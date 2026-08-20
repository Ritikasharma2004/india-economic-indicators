"""
Exploratory data analysis on the processed indicator panel.

Produces reports/figures/*.png and reports/eda_findings.md. Every number quoted
in the findings file is computed here rather than typed in by hand, so the
report cannot drift away from the data.
"""

from pathlib import Path

import matplotlib

matplotlib.use("Agg")  # no display needed; write straight to file

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

ROOT = Path(__file__).resolve().parents[1]
PROCESSED_DIR = ROOT / "data" / "processed"
FIG_DIR = ROOT / "reports" / "figures"
REPORT_PATH = ROOT / "reports" / "eda_findings.md"

TARGET = "gdp_growth_pct"

# Events worth marking on the time series. These are annotations for the
# reader, not inputs to any model.
EVENTS = {
    1965: "Indo-Pak war\n+ drought",
    1979: "Second oil shock",
    1991: "Balance of payments\ncrisis / liberalisation",
    2008: "Global financial\ncrisis",
    2020: "COVID-19",
}

sns.set_theme(style="whitegrid", palette="deep")
plt.rcParams["figure.dpi"] = 130
plt.rcParams["savefig.bbox"] = "tight"


def fig_growth_timeline(df: pd.DataFrame) -> None:
    """GDP growth over the full period with major shocks annotated."""
    growth = df[TARGET].dropna()

    fig, ax = plt.subplots(figsize=(11, 5))
    ax.plot(growth.index, growth.values, color="#2166ac", linewidth=1.8,
            label="Annual growth")
    ax.plot(
        growth.index,
        growth.rolling(5, center=True).mean(),
        color="#b2182b",
        linewidth=2.2,
        linestyle="--",
        label="5-year centred mean",
    )
    ax.axhline(0, color="#444", linewidth=0.9)
    ax.fill_between(growth.index, 0, growth.values, where=growth.values < 0,
                    color="#b2182b", alpha=0.15)

    for year, label in EVENTS.items():
        if year not in growth.index:
            continue
        value = growth.loc[year]
        # Shock years sit near the bottom of the axis, so their labels go above
        # the point instead of below where they would fall off the chart.
        offset = 3.6 if value < 0 else -4.5
        ax.annotate(
            label,
            xy=(year, value),
            xytext=(year, value + offset),
            ha="center", va="center", fontsize=7.5, color="#333",
            arrowprops=dict(arrowstyle="->", color="#888", linewidth=0.8),
        )

    ax.set_title("India GDP growth, 1961-2025", fontsize=13, weight="bold")
    ax.set_xlabel("Year")
    ax.set_ylabel("Annual growth (%)")
    ax.legend(frameon=False)
    fig.savefig(FIG_DIR / "01_growth_timeline.png")
    plt.close(fig)


def fig_growth_distribution(df: pd.DataFrame) -> None:
    """Distribution of annual growth, pre- and post-liberalisation."""
    growth = df[TARGET].dropna()
    pre = growth[growth.index < 1991]
    post = growth[growth.index >= 1991]

    fig, axes = plt.subplots(1, 2, figsize=(11, 4))

    sns.histplot(growth, bins=18, kde=True, ax=axes[0], color="#2166ac")
    axes[0].axvline(growth.mean(), color="#b2182b", linestyle="--",
                    label=f"mean {growth.mean():.2f}%")
    axes[0].axvline(growth.median(), color="#1a9850", linestyle=":",
                    label=f"median {growth.median():.2f}%")
    axes[0].set_title("Distribution of annual growth")
    axes[0].set_xlabel("Growth (%)")
    axes[0].legend(frameon=False, fontsize=8)

    box_df = pd.DataFrame({
        "growth": pd.concat([pre, post]),
        "era": ["1961-1990"] * len(pre) + ["1991-2025"] * len(post),
    })
    sns.boxplot(data=box_df, x="era", y="growth", ax=axes[1],
                palette=["#92c5de", "#2166ac"], hue="era", legend=False)
    sns.stripplot(data=box_df, x="era", y="growth", ax=axes[1],
                  color="#333", size=3, alpha=0.5)
    axes[1].set_title("Growth before and after liberalisation")
    axes[1].set_xlabel("")
    axes[1].set_ylabel("Growth (%)")

    fig.savefig(FIG_DIR / "02_growth_distribution.png")
    plt.close(fig)


def fig_correlation(df: pd.DataFrame) -> pd.DataFrame:
    """Correlation heatmap across all indicators; returns the matrix."""
    window = df.loc[df.index >= 1991].dropna(axis=1, how="all")
    corr = window.corr()

    fig, ax = plt.subplots(figsize=(9.5, 7.5))
    mask = np.triu(np.ones_like(corr, dtype=bool), k=1)
    sns.heatmap(
        corr, mask=mask, annot=True, fmt=".2f", cmap="RdBu_r", center=0,
        vmin=-1, vmax=1, square=True, linewidths=0.5,
        cbar_kws={"shrink": 0.7, "label": "Pearson r"}, ax=ax,
        annot_kws={"size": 7},
    )
    ax.set_title("Indicator correlations, 1991-2025", fontsize=12, weight="bold")
    fig.savefig(FIG_DIR / "03_correlation_heatmap.png")
    plt.close(fig)
    return corr


def fig_structure(df: pd.DataFrame) -> None:
    """Sector shares of GDP as a stacked area chart."""
    sectors = df[
        ["agriculture_pct_gdp", "industry_pct_gdp", "services_pct_gdp"]
    ].dropna()

    fig, ax = plt.subplots(figsize=(11, 4.8))
    ax.stackplot(
        sectors.index,
        sectors["agriculture_pct_gdp"],
        sectors["industry_pct_gdp"],
        sectors["services_pct_gdp"],
        labels=["Agriculture", "Industry", "Services"],
        colors=["#8c6d31", "#6b6ecf", "#31a354"],
        alpha=0.85,
    )
    ax.set_title("Sector share of GDP, 1960-2025", fontsize=13, weight="bold")
    ax.set_xlabel("Year")
    ax.set_ylabel("Share of GDP (%)")
    ax.set_xlim(sectors.index.min(), sectors.index.max())
    ax.legend(loc="upper left", frameon=True, framealpha=0.9)
    fig.savefig(FIG_DIR / "04_sector_structure.png")
    plt.close(fig)


def fig_volatility(df: pd.DataFrame) -> None:
    """Rolling 10-year volatility of growth."""
    growth = df[TARGET].dropna()
    vol = growth.rolling(10).std()

    fig, ax = plt.subplots(figsize=(11, 4))
    ax.plot(vol.index, vol.values, color="#b2182b", linewidth=2)
    ax.fill_between(vol.index, 0, vol.values, color="#b2182b", alpha=0.15)
    ax.set_title("Rolling 10-year volatility of GDP growth", fontsize=12,
                 weight="bold")
    ax.set_xlabel("Year")
    ax.set_ylabel("Standard deviation (pp)")
    fig.savefig(FIG_DIR / "05_rolling_volatility.png")
    plt.close(fig)


def detect_outliers(df: pd.DataFrame) -> pd.DataFrame:
    """Flag growth years outside 1.5x IQR, the years any model will struggle on."""
    growth = df[TARGET].dropna()
    q1, q3 = growth.quantile([0.25, 0.75])
    iqr = q3 - q1
    low, high = q1 - 1.5 * iqr, q3 + 1.5 * iqr

    flagged = growth[(growth < low) | (growth > high)]
    return pd.DataFrame({
        "year": flagged.index,
        "growth_pct": flagged.round(2).values,
        "direction": np.where(flagged > high, "unusually high", "unusually low"),
    }).sort_values("year")


def main() -> None:
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    df = pd.read_csv(PROCESSED_DIR / "india_indicators.csv", index_col="year")

    fig_growth_timeline(df)
    fig_growth_distribution(df)
    corr = fig_correlation(df)
    fig_structure(df)
    fig_volatility(df)
    print(f"Wrote 5 figures to {FIG_DIR}")

    growth = df[TARGET].dropna()
    pre = growth[growth.index < 1991]
    post = growth[growth.index >= 1991]
    outliers = detect_outliers(df)

    summary = df.describe().T[["count", "mean", "std", "min", "50%", "max"]].round(2)
    summary.columns = ["count", "mean", "std", "min", "median", "max"]

    # Strongest correlations with growth, ignoring growth against itself and
    # the per-capita series which is definitionally almost the same number.
    growth_corr = (
        corr[TARGET]
        .drop([TARGET, "gdp_per_capita_growth_pct"], errors="ignore")
        .sort_values(key=abs, ascending=False)
    )

    agri = df["agriculture_pct_gdp"].dropna()
    serv = df["services_pct_gdp"].dropna()
    spread_word = "narrower" if post.std() < pre.std() else "wider"

    lines = [
        "# Exploratory Data Analysis",
        "",
        "All figures are generated by `src/eda.py` and saved to "
        "`reports/figures/`.",
        "",
        "## 1. Summary statistics",
        "",
        summary.to_markdown(),
        "",
        "## 2. Growth over time",
        "",
        "![Growth timeline](figures/01_growth_timeline.png)",
        "",
        f"Mean annual growth across {len(growth)} years is "
        f"**{growth.mean():.2f}%** with a standard deviation of "
        f"{growth.std():.2f} percentage points. The series ranges from "
        f"{growth.min():.2f}% ({int(growth.idxmin())}) to "
        f"{growth.max():.2f}% ({int(growth.idxmax())}).",
        "",
        "## 3. The 1991 break",
        "",
        "![Growth distribution](figures/02_growth_distribution.png)",
        "",
        "| Era | Years | Mean growth | Volatility |",
        "|---|---|---|---|",
        f"| 1961-1990 | {len(pre)} | {pre.mean():.2f}% | {pre.std():.2f} |",
        f"| 1991-2025 | {len(post)} | {post.mean():.2f}% | {post.std():.2f} |",
        "",
        f"Average growth after liberalisation is "
        f"**{post.mean() - pre.mean():+.2f} percentage points** higher, and the "
        f"spread is {spread_word} ({post.std():.2f} vs {pre.std():.2f}). The "
        "pre-1991 economy did not just grow more slowly, it grew less "
        "predictably.",
        "",
        "## 4. What moves with growth",
        "",
        "![Correlation heatmap](figures/03_correlation_heatmap.png)",
        "",
        "Correlation with GDP growth, 1991-2025:",
        "",
        growth_corr.round(3).to_frame("correlation").to_markdown(),
        "",
        "These are contemporaneous correlations, not predictive ones. A "
        "variable that moves with growth in the same year is not necessarily "
        "able to forecast it, which is why `src/forecast.py` uses lagged "
        "values only.",
        "",
        "## 5. Structural change",
        "",
        "![Sector structure](figures/04_sector_structure.png)",
        "",
        f"The agriculture share of GDP fell from {agri.iloc[0]:.1f}% in "
        f"{int(agri.index[0])} to {agri.iloc[-1]:.1f}% in "
        f"{int(agri.index[-1])}, while services rose from {serv.iloc[0]:.1f}% "
        f"to {serv.iloc[-1]:.1f}%. The industry share is broadly flat across "
        "the whole period, so the shift has been agriculture to services "
        "rather than the agriculture-to-industry path of most industrialising "
        "economies.",
        "",
        "## 6. Volatility",
        "",
        "![Rolling volatility](figures/05_rolling_volatility.png)",
        "",
        "## 7. Outlier years",
        "",
        outliers.to_markdown(index=False),
        "",
        "These years are shocks, not data errors, so they are kept in the "
        "dataset. They are the main reason a naive carry-forward forecast "
        "performs badly: it propagates a one-off shock into the prediction for "
        "the following year.",
    ]

    REPORT_PATH.write_text("\n".join(lines), encoding="utf-8")
    print(f"Wrote {REPORT_PATH}")
    print(f"\nOutlier years detected: {len(outliers)}")
    print(outliers.to_string(index=False))
    print("\nTop correlations with growth:")
    print(growth_corr.head(4).round(3).to_string())


if __name__ == "__main__":
    main()
