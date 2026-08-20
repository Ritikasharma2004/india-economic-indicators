"""
Extract India's economic indicators from the World Bank WDI API.

Raw JSON responses are saved unchanged to data/raw/ so that every later step
can be traced back to exactly what the API returned on the day it was pulled.
"""

import json
import time
from datetime import date
from pathlib import Path

import pandas as pd
import requests

BASE_URL = "https://api.worldbank.org/v2"
COUNTRY = "IND"
ROOT = Path(__file__).resolve().parents[1]
RAW_DIR = ROOT / "data" / "raw"

# World Bank indicator codes -> short column names used across the project.
INDICATORS = {
    "NY.GDP.MKTP.KD.ZG": "gdp_growth_pct",
    "NY.GDP.PCAP.KD.ZG": "gdp_per_capita_growth_pct",
    "FP.CPI.TOTL.ZG": "inflation_cpi_pct",
    "SL.UEM.TOTL.ZS": "unemployment_pct",
    "NE.EXP.GNFS.ZS": "exports_pct_gdp",
    "NE.IMP.GNFS.ZS": "imports_pct_gdp",
    "BX.KLT.DINV.WD.GD.ZS": "fdi_inflow_pct_gdp",
    "NV.AGR.TOTL.ZS": "agriculture_pct_gdp",
    "NV.IND.TOTL.ZS": "industry_pct_gdp",
    "NV.SRV.TOTL.ZS": "services_pct_gdp",
}


def fetch_indicator(code: str, retries: int = 3) -> list[dict]:
    """Return the raw record list for one indicator, retrying on transient errors."""
    url = f"{BASE_URL}/country/{COUNTRY}/indicator/{code}"
    params = {"format": "json", "per_page": 500}

    for attempt in range(1, retries + 1):
        try:
            response = requests.get(url, params=params, timeout=30)
            response.raise_for_status()
            payload = response.json()
        except (requests.RequestException, ValueError) as exc:
            if attempt == retries:
                raise RuntimeError(f"{code}: failed after {retries} attempts") from exc
            time.sleep(2 * attempt)
            continue

        # The API answers with [metadata, records]; an error answers with a
        # single-element list carrying a "message" key.
        if not isinstance(payload, list) or len(payload) < 2:
            raise RuntimeError(f"{code}: unexpected API response {payload!r}")

        return payload[1] or []

    return []


def main() -> None:
    RAW_DIR.mkdir(parents=True, exist_ok=True)
    pulled_on = date.today().isoformat()
    summary = []

    for code, name in INDICATORS.items():
        records = fetch_indicator(code)

        raw_path = RAW_DIR / f"{name}.json"
        raw_path.write_text(
            json.dumps(
                {"indicator_code": code, "pulled_on": pulled_on, "records": records},
                indent=2,
            ),
            encoding="utf-8",
        )

        values = [r for r in records if r.get("value") is not None]
        years = sorted(int(r["date"]) for r in values)
        summary.append(
            {
                "indicator_code": code,
                "column_name": name,
                "records_returned": len(records),
                "non_null_values": len(values),
                "first_year": years[0] if years else None,
                "last_year": years[-1] if years else None,
            }
        )
        print(f"{name:28s} {len(values):3d} values  {years[0] if years else '-'}-{years[-1] if years else '-'}")

    manifest = pd.DataFrame(summary)
    manifest["pulled_on"] = pulled_on
    manifest.to_csv(RAW_DIR / "_manifest.csv", index=False)
    print(f"\nSaved {len(summary)} raw files to {RAW_DIR}")


if __name__ == "__main__":
    main()
