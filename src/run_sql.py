"""
Execute every query in sql/analysis.sql and write the results to
reports/sql_results.md, so the SQL output is reviewable without a database
client.
"""

import re
import sqlite3
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
DB_PATH = ROOT / "data" / "india_economy.db"
SQL_PATH = ROOT / "sql" / "analysis.sql"
OUT_PATH = ROOT / "reports" / "sql_results.md"


def split_queries(text: str) -> list[tuple[str, str]]:
    """Split the file into (title, sql) pairs on the '-- Qn.' comment markers."""
    blocks = re.split(r"\n(?=-- Q\d+\.)", text)
    queries = []

    for block in blocks:
        if not block.strip().startswith("-- Q"):
            continue
        lines = block.strip().splitlines()
        title = lines[0].lstrip("- ").strip()
        sql = "\n".join(line for line in lines if not line.strip().startswith("--"))
        if sql.strip():
            queries.append((title, sql.strip()))

    return queries


def main() -> None:
    queries = split_queries(SQL_PATH.read_text(encoding="utf-8"))
    out = ["# SQL Analysis Results", "", f"Source: `sql/analysis.sql` against `{DB_PATH.name}`", ""]

    with sqlite3.connect(DB_PATH) as conn:
        for title, sql in queries:
            df = pd.read_sql_query(sql, conn)
            out += [f"## {title}", "", df.to_markdown(index=False), ""]
            print(f"{title}  ->  {len(df)} rows")

    OUT_PATH.write_text("\n".join(out), encoding="utf-8")
    print(f"\nWrote {OUT_PATH}")


if __name__ == "__main__":
    main()
