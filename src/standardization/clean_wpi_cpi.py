"""
AgriVault – Clean WPI + CPI (Silver Layer)
============================================
Reads WPI and CPI Excel files from S3, standardizes them to long format,
writes to standardized/wpi/ and standardized/cpi/.

WPI Excel structure (Sheet3):
    Wide format — rows are commodity hierarchies, columns are month-year labels
    e.g. "Apr-23", "May-23", ...
    Key cols: Level, Commodity Name, Commodity Code, Commodity Weight, <month-year cols>

CPI Excel structure (Sheet: CPI Data):
    Already in long format:
    base_year, series, year, month, state, sector, division, group, class,
    sub_class, item, code, index, inflation, imputation

Strategy:
    WPI: melt wide → long, parse month-year string → date, keep Commodity Name + index value
    CPI: filter to Food-related divisions relevant for agri price forecasting, pivot/select

Output:
    standardized/wpi/   — columns: date, commodity_name, commodity_code, wpi_index, weight
    standardized/cpi/   — columns: date, state, sector, item, cpi_index, inflation

Note: Both Excel files are read locally (since reading Excel from S3 via Spark is non-trivial),
      then converted to Parquet locally and uploaded. This is acceptable since they are
      relatively small reference tables (WPI: <1MB, CPI: 101MB but loaded with pandas).

Run
---
    python -m src.standardization.clean_wpi_cpi
"""

from __future__ import annotations

import io
import logging
from pathlib import Path

import boto3
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
import yaml

log = logging.getLogger(__name__)

PROJECT_ROOT = Path(__file__).resolve().parents[2]
CONFIG_PATH = PROJECT_ROOT / "configs" / "aws_config.yaml"
RAW_ROOT = PROJECT_ROOT / "data" / "raw" / "cpi_wpi"
STD_ROOT = PROJECT_ROOT / "data" / "standardized"


def load_config() -> dict:
    with open(CONFIG_PATH) as f:
        return yaml.safe_load(f)


def make_s3(cfg: dict):
    session = boto3.Session(
        profile_name=cfg["aws"].get("profile"),
        region_name=cfg["aws"].get("region", "ap-south-1"),
    )
    return session.client("s3"), cfg["s3"]["bucket"]


def bucket_key(cfg: dict, layer: str, path: str) -> str:
    prefix = cfg["s3"]["prefixes"].get(layer, f"{layer}/").rstrip("/")
    return f"{prefix}/{path}"


# ---------------------------------------------------------------------------
# WPI Cleaning
# ---------------------------------------------------------------------------

def parse_wpi_month(label: str) -> pd.Timestamp | None:
    """Parse 'Apr-23' → 2023-04-01, 'Jan-24' → 2024-01-01."""
    try:
        return pd.to_datetime(label, format="%b-%y")
    except Exception:
        return None


def clean_wpi(path: Path) -> pd.DataFrame:
    """
    Read WPI Excel, melt wide → long.
    Outputs: date, commodity_name, commodity_code, level, weight, wpi_index
    """
    log.info("Reading WPI: %s", path)
    df = pd.read_excel(path, sheet_name=0)

    id_cols = ["Level", "Commodity Name", "Commodity Code", "Commodity Weight"]
    id_cols_present = [c for c in id_cols if c in df.columns]
    value_cols = [c for c in df.columns if c not in id_cols_present]

    # Filter out non-month columns (some Excels have notes columns)
    month_cols = [c for c in value_cols if parse_wpi_month(str(c)) is not None]

    melted = df[id_cols_present + month_cols].melt(
        id_vars=id_cols_present,
        value_vars=month_cols,
        var_name="month_label",
        value_name="wpi_index",
    )

    melted["date"] = melted["month_label"].apply(
        lambda x: parse_wpi_month(str(x))
    )
    melted = melted.dropna(subset=["date", "wpi_index"])
    melted["wpi_index"] = pd.to_numeric(melted["wpi_index"], errors="coerce")
    melted = melted.dropna(subset=["wpi_index"])

    # Rename columns
    rename = {
        "Level": "level",
        "Commodity Name": "commodity_name",
        "Commodity Code": "commodity_code",
        "Commodity Weight": "weight",
    }
    melted = melted.rename(columns=rename)
    melted["commodity_name"] = melted["commodity_name"].str.strip().str.upper()

    result = melted[["date", "commodity_name", "commodity_code", "level", "weight", "wpi_index"]].copy()
    result = result.sort_values(["commodity_name", "date"]).reset_index(drop=True)

    log.info("WPI clean rows: %d | date range: %s → %s",
             len(result), result["date"].min().date(), result["date"].max().date())
    return result


# ---------------------------------------------------------------------------
# CPI Cleaning
# ---------------------------------------------------------------------------

def clean_cpi(path: Path, chunksize: int = 100_000) -> pd.DataFrame:
    """
    Read CPI Excel in chunks (101 MB file).
    Outputs: date, state, sector, division, group, item, cpi_index, inflation

    Filters to food-relevant rows only (Food and Beverages division +
    Fuel and Light — proxy for energy cost passed into farm gate prices).
    """
    log.info("Reading CPI: %s (this may take 30-60s for 101 MB file)", path)

    # pandas read_excel doesn't support chunking — load all at once
    # but select only relevant columns first
    keep_cols = [
        "year", "month", "state", "sector", "division",
        "group", "item", "code", "index", "inflation",
    ]
    df = pd.read_excel(path, sheet_name="CPI Data", usecols=keep_cols)

    log.info("CPI raw rows: %d", len(df))

    # ── Parse date ─────────────────────────────────────────────────────────
    # year=2025, month="December" → 2025-12-01
    df["date"] = pd.to_datetime(
        df["year"].astype(str) + "-" + df["month"],
        format="%Y-%B",
        errors="coerce",
    )
    df = df.dropna(subset=["date"])

    # ── Cast numerics ──────────────────────────────────────────────────────
    df["cpi_index"]  = pd.to_numeric(df["index"],     errors="coerce")
    df["inflation"]  = pd.to_numeric(df["inflation"],  errors="coerce")
    df = df.drop(columns=["index"])

    # ── Normalise text ─────────────────────────────────────────────────────
    for col in ("state", "sector", "division", "group", "item"):
        df[col] = df[col].str.strip().str.upper()

    # ── Filter: food-related + fuel divisions (relevant for agri) ─────────
    food_divisions = {"FOOD AND BEVERAGES", "FUEL AND LIGHT", "MISCELLANEOUS"}
    df = df[df["division"].isin(food_divisions)]

    # ── Dedup ──────────────────────────────────────────────────────────────
    df = df.drop_duplicates(subset=["date", "state", "sector", "item", "code"])

    result = df[["date", "state", "sector", "division", "group", "item",
                 "code", "cpi_index", "inflation"]].copy()
    result = result.sort_values(["date", "state", "item"]).reset_index(drop=True)

    log.info("CPI clean rows: %d | date range: %s → %s",
             len(result), result["date"].min().date(), result["date"].max().date())
    return result


# ---------------------------------------------------------------------------
# Write to S3 as Parquet
# ---------------------------------------------------------------------------

def write_parquet_to_s3(df: pd.DataFrame, s3, bucket: str, key: str) -> None:
    """Serialize DataFrame as Parquet and upload to S3."""
    table = pa.Table.from_pandas(df, preserve_index=False)
    buf = io.BytesIO()
    pq.write_table(table, buf, compression="snappy")
    buf.seek(0)
    s3.put_object(Bucket=bucket, Key=key, Body=buf.getvalue())
    log.info("Written %d rows → s3://%s/%s", len(df), bucket, key)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s  %(levelname)-8s  %(message)s",
        datefmt="%H:%M:%S",
    )

    cfg = load_config()
    s3, bucket = make_s3(cfg)

    # ── WPI ────────────────────────────────────────────────────────────────
    wpi_path = RAW_ROOT / "wpi" / "wpi_monthly.xlsx"
    wpi_df = clean_wpi(wpi_path)
    wpi_key = bucket_key(cfg, "standardized", "wpi/wpi_monthly.parquet")
    write_parquet_to_s3(wpi_df, s3, bucket, wpi_key)

    # Also save locally
    local_wpi = STD_ROOT / "wpi" / "wpi_monthly.parquet"
    local_wpi.parent.mkdir(parents=True, exist_ok=True)
    wpi_df.to_parquet(local_wpi, index=False)
    log.info("WPI saved locally: %s", local_wpi)

    # ── CPI ────────────────────────────────────────────────────────────────
    cpi_path = RAW_ROOT / "cpi" / "cpi_monthly.xlsx"
    cpi_df = clean_cpi(cpi_path)
    cpi_key = bucket_key(cfg, "standardized", "cpi/cpi_monthly.parquet")
    write_parquet_to_s3(cpi_df, s3, bucket, cpi_key)

    # Also save locally
    local_cpi = STD_ROOT / "cpi" / "cpi_monthly.parquet"
    local_cpi.parent.mkdir(parents=True, exist_ok=True)
    cpi_df.to_parquet(local_cpi, index=False)
    log.info("CPI saved locally: %s", local_cpi)

    log.info("✓ WPI + CPI standardized and written to S3")


if __name__ == "__main__":
    main()
