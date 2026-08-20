"""
Streamlit dashboard for the India economic indicators analysis.

Run with:
    streamlit run dashboard/app.py
"""

from pathlib import Path

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

ROOT = Path(__file__).resolve().parents[1]
PROCESSED_DIR = ROOT / "data" / "processed"

TARGET = "gdp_growth_pct"

LABELS = {
    "gdp_growth_pct": "GDP growth (%)",
    "gdp_per_capita_growth_pct": "GDP per capita growth (%)",
    "inflation_cpi_pct": "Inflation, CPI (%)",
    "unemployment_pct": "Unemployment (%)",
    "exports_pct_gdp": "Exports (% of GDP)",
    "imports_pct_gdp": "Imports (% of GDP)",
    "trade_balance_pct_gdp": "Trade balance (% of GDP)",
    "fdi_inflow_pct_gdp": "FDI net inflows (% of GDP)",
    "agriculture_pct_gdp": "Agriculture (% of GDP)",
    "industry_pct_gdp": "Industry (% of GDP)",
    "services_pct_gdp": "Services (% of GDP)",
}

st.set_page_config(
    page_title="India Economic Indicators",
    page_icon="IN",
    layout="wide",
)


@st.cache_data
def load_wide() -> pd.DataFrame:
    return pd.read_csv(PROCESSED_DIR / "india_indicators.csv", index_col="year")


@st.cache_data
def load_scoreboard() -> pd.DataFrame:
    return pd.read_csv(PROCESSED_DIR / "model_scoreboard.csv")


@st.cache_data
def load_predictions() -> pd.DataFrame:
    frames = []
    for path in sorted(PROCESSED_DIR.glob("predictions_*.csv")):
        feature_set, model = path.stem.replace("predictions_", "").split("_", 1)
        frame = pd.read_csv(path)
        frame["feature_set"] = feature_set
        frame["model"] = model
        frames.append(frame)
    return pd.concat(frames, ignore_index=True)


df = load_wide()
scoreboard = load_scoreboard()
predictions = load_predictions()

st.title("India Economic Indicators, 1960-2025")
st.caption(
    "Source: World Bank World Development Indicators API. "
    "Built by Ritika Sharma."
)

with st.sidebar:
    st.header("Filters")
    year_min, year_max = int(df.index.min()), int(df.index.max())
    year_range = st.slider(
        "Year range", year_min, year_max, (1991, year_max), step=1
    )
    st.markdown("---")
    st.caption(
        "The default window starts at 1991 because that is the first year "
        "every indicator is reported. See the Methodology tab."
    )

window = df.loc[year_range[0] : year_range[1]]

tab_overview, tab_explore, tab_structure, tab_forecast, tab_method = st.tabs(
    ["Overview", "Indicator explorer", "Economic structure", "Forecasting", "Methodology"]
)


with tab_overview:
    growth = window[TARGET].dropna()

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Average growth", f"{growth.mean():.2f}%")
    c2.metric("Volatility (sd)", f"{growth.std():.2f} pp")
    c3.metric(
        "Best year", f"{growth.max():.2f}%", help=f"Year {int(growth.idxmax())}"
    )
    c4.metric(
        "Worst year", f"{growth.min():.2f}%", help=f"Year {int(growth.idxmin())}"
    )

    fig = go.Figure()
    fig.add_trace(
        go.Scatter(
            x=growth.index, y=growth.values, name="Annual growth",
            line=dict(color="#2166ac", width=2),
        )
    )
    fig.add_trace(
        go.Scatter(
            x=growth.index, y=growth.rolling(5, center=True).mean(),
            name="5-year centred mean",
            line=dict(color="#b2182b", width=2.5, dash="dash"),
        )
    )
    fig.add_hline(y=0, line_color="#666", line_width=1)
    fig.update_layout(
        title="GDP growth over time",
        xaxis_title="Year", yaxis_title="Annual growth (%)",
        hovermode="x unified", height=440,
    )
    st.plotly_chart(fig, width="stretch")

    st.subheader("Average growth by decade")
    decade = (
        df[TARGET].dropna().to_frame()
        .assign(decade=lambda d: (d.index // 10) * 10)
        .groupby("decade")[TARGET]
        .agg(["mean", "std", "count"])
        .round(2)
    )
    decade.columns = ["Average growth (%)", "Volatility (pp)", "Years"]
    bar = px.bar(
        decade.reset_index(), x="decade", y="Average growth (%)",
        error_y="Volatility (pp)", color="Average growth (%)",
        color_continuous_scale="Blues", height=380,
    )
    bar.update_layout(xaxis_title="Decade", coloraxis_showscale=False)
    st.plotly_chart(bar, width="stretch")
    st.caption(
        "Error bars show the standard deviation within each decade. The 2020s "
        "average is dragged down and widened by the COVID contraction and rebound."
    )


with tab_explore:
    st.subheader("Compare indicators")
    chosen = st.multiselect(
        "Indicators",
        options=[c for c in df.columns if c in LABELS],
        default=[TARGET, "inflation_cpi_pct"],
        format_func=lambda c: LABELS.get(c, c),
    )

    if not chosen:
        st.info("Pick at least one indicator to plot.")
    else:
        tidy = (
            window[chosen].reset_index()
            .melt(id_vars="year", var_name="indicator", value_name="value")
            .dropna(subset=["value"])
        )
        tidy["indicator"] = tidy["indicator"].map(LABELS).fillna(tidy["indicator"])
        line = px.line(
            tidy, x="year", y="value", color="indicator", height=460,
            markers=len(window) <= 40,
        )
        line.update_layout(
            xaxis_title="Year", yaxis_title="Value",
            hovermode="x unified", legend_title="",
        )
        st.plotly_chart(line, width="stretch")

        if len(chosen) >= 2:
            st.subheader("Correlation")
            corr = window[chosen].corr().round(2)
            corr.index = [LABELS.get(c, c) for c in corr.index]
            corr.columns = [LABELS.get(c, c) for c in corr.columns]
            heat = px.imshow(
                corr, text_auto=True, color_continuous_scale="RdBu_r",
                zmin=-1, zmax=1, height=420, aspect="auto",
            )
            st.plotly_chart(heat, width="stretch")
            st.caption(
                "Same-year correlation only. A high value here does not mean "
                "one indicator can forecast the other."
            )


with tab_structure:
    sectors = window[
        ["agriculture_pct_gdp", "industry_pct_gdp", "services_pct_gdp"]
    ].dropna()

    area = go.Figure()
    for col, colour, name in [
        ("agriculture_pct_gdp", "#8c6d31", "Agriculture"),
        ("industry_pct_gdp", "#6b6ecf", "Industry"),
        ("services_pct_gdp", "#31a354", "Services"),
    ]:
        area.add_trace(
            go.Scatter(
                x=sectors.index, y=sectors[col], name=name, stackgroup="one",
                line=dict(width=0.5, color=colour), fillcolor=colour,
            )
        )
    area.update_layout(
        title="Sector share of GDP", xaxis_title="Year",
        yaxis_title="Share of GDP (%)", hovermode="x unified", height=460,
    )
    st.plotly_chart(area, width="stretch")

    if len(sectors) > 1:
        first, last = sectors.iloc[0], sectors.iloc[-1]
        change = pd.DataFrame({
            "Sector": ["Agriculture", "Industry", "Services"],
            f"{int(sectors.index[0])}": [
                first["agriculture_pct_gdp"], first["industry_pct_gdp"],
                first["services_pct_gdp"],
            ],
            f"{int(sectors.index[-1])}": [
                last["agriculture_pct_gdp"], last["industry_pct_gdp"],
                last["services_pct_gdp"],
            ],
        })
        change["Change (pp)"] = (
            change[f"{int(sectors.index[-1])}"] - change[f"{int(sectors.index[0])}"]
        )
        st.dataframe(
            change.round(1), width="stretch", hide_index=True
        )


with tab_forecast:
    st.subheader("Can any model beat a naive forecast?")
    st.markdown(
        "Models are validated **walk-forward**: train on every year up to "
        "*t-1*, predict year *t*, step forward. All features are lagged, so no "
        "model sees information that would not have been available at the time."
    )

    display = scoreboard.copy()
    display["model"] = display["model"].str.replace("_", " ").str.title()
    display["feature_set"] = display["feature_set"].str.title()
    display = display.rename(columns={
        "feature_set": "Feature set", "model": "Model", "rmse": "RMSE",
        "mae": "MAE", "directional_accuracy": "Directional accuracy",
    }).round(3)

    st.dataframe(
        display.sort_values("RMSE"), width="stretch", hide_index=True
    )

    best = scoreboard.loc[scoreboard["rmse"].idxmin()]
    st.warning(
        f"The lowest error belongs to **{best['model'].replace('_', ' ')}** "
        f"(RMSE {best['rmse']:.2f}) - a trivial baseline, not a machine "
        "learning model. See the Methodology tab for why."
    )

    fs = st.selectbox(
        "Feature set", sorted(predictions["feature_set"].unique())
    )
    subset = predictions[predictions["feature_set"] == fs]

    fig = go.Figure()
    actual = subset[subset["model"] == "naive"][["year", "actual"]].drop_duplicates()
    fig.add_trace(
        go.Scatter(
            x=actual["year"], y=actual["actual"], name="Actual",
            line=dict(color="#111", width=3),
        )
    )
    for model in sorted(subset["model"].unique()):
        rows = subset[subset["model"] == model]
        fig.add_trace(
            go.Scatter(
                x=rows["year"], y=rows["predicted"],
                name=model.replace("_", " ").title(),
                line=dict(width=1.6, dash="dot"),
            )
        )
    fig.update_layout(
        title=f"Predicted vs actual growth ({fs})", xaxis_title="Year",
        yaxis_title="GDP growth (%)", hovermode="x unified", height=460,
    )
    st.plotly_chart(fig, width="stretch")


with tab_method:
    st.subheader("Method and limitations")
    st.markdown(
        """
**Data.** World Bank World Development Indicators API, country code `IND`,
eleven indicators. Raw JSON responses are stored in `data/raw/` so any figure
can be traced back to the API response it came from.

**Why the window starts at 1991.** Coverage is uneven before then. Unemployment
is only reported from 1991 and FDI from 1970, while national-accounts series go
back to 1960. Starting at 1991 keeps every indicator present rather than
letting models silently drop rows.

**Why no model beats the baseline.** Three reasons, and all of them are
properties of the data rather than of the models:

1. **The sample is small.** Even the univariate setup has 62 usable years.
   Tree-based models need far more than that to learn a stable pattern.
2. **Growth is stationary white noise around a constant mean.** An Augmented
   Dickey-Fuller test rejects non-stationarity decisively (statistic -7.65,
   p < 0.0001), and the lag-1 autocorrelation is **0.026** - effectively zero.
   Last year's growth says almost nothing about this year's, so lag features
   have no serial dependence to capture. Shocks such as 1965, 1979 and 2020
   are sharp but do not persist, which is why predicting the long-run average
   beats carrying last year forward.
3. **Same-year correlations are weak.** The strongest correlation with growth
   in the 1991-2025 window is about 0.24 in absolute terms. Lagging those
   variables weakens the relationship further.

**What would actually improve this.** Quarterly rather than annual data would
multiply the sample size roughly fourfold. Genuine leading indicators - PMI,
IIP, credit growth, monsoon rainfall - carry signal that annual national
accounts do not. Neither is available from this API, which is why the honest
answer here is that the baseline wins.

**A note on reporting.** It would have been easy to report only the best
machine learning model and leave the baseline out. The baseline is included
precisely because it wins. A forecasting project that cannot beat the mean
should say so.
        """
    )
