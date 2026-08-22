"""
AgriVault – Fetch Multi-Year MODIS NDVI from NASA (via Google Earth Engine)
============================================================================
Pulls MOD13A2 (16-day, 1 km) NDVI composites for all mandi locations.

Two output files per run:
  1. ndvi_modis_baseline.csv   — 4-year baseline (2021-2024), one row per mandi × DOY
     Columns: mandi_id, market_code, mandi_name, district, state, latitude,
              longitude, doy, ndvi_mean, ndvi_std, n_years
  2. ndvi_modis_current.csv    — 2025 observations + z-score anomaly
     Columns: mandi_id, market_code, mandi_name, district, state, latitude,
              longitude, date, doy, ndvi_raw, ndvi_anomaly (z-score)

MODIS NDVI convention:
  - Raw values are int16 scaled by ×0.0001 -> NDVI in [-0.2, 1.0]
  - Quality filter: pixel_reliability ∈ {0 (good), 1 (marginal)}

Run
---
    python -m src.ingestion.fetch_modis_ndvi
    python -m src.ingestion.fetch_modis_ndvi --limit 10      # first 10 mandis
    python -m src.ingestion.fetch_modis_ndvi --batch-size 150

Prerequisites:
    pip install earthengine-api pandas pyyaml
    earthengine authenticate
"""

from __future__ import annotations

import argparse
import logging
import sys
import time
from pathlib import Path

import ee
import numpy as np
import pandas as pd
import yaml

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
PROJECT_ROOT = Path(__file__).resolve().parents[2]
CONFIG_FILE = PROJECT_ROOT / "configs" / "gee_config.yaml"
MANDI_FILE = PROJECT_ROOT / "data" / "reference" / "mandi_locations.csv"
OUTPUT_DIR = PROJECT_ROOT / "data" / "raw" / "ndvi_modis"

log = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------
def load_config() -> dict:
    with open(CONFIG_FILE, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def initialize_gee(project_id: str) -> None:
    ee.Initialize(project=project_id)
    log.info("GEE initialized for project %s", project_id)


# ---------------------------------------------------------------------------
# MODIS helpers
# ---------------------------------------------------------------------------
def modis_ndvi_scale(image: ee.Image) -> ee.Image:
    """Scale raw MOD13A2 DN values to physical NDVI (×0.0001)."""
    ndvi = image.select("NDVI").multiply(0.0001).rename("ndvi")
    return ndvi.copyProperties(image, ["system:time_start"])


def modis_qa_mask(image: ee.Image) -> ee.Image:
    """Mask to good / marginal pixels using the 'SummaryQA' band.

    MODIS Collection 6.1 (MOD13A2) SummaryQA values:
         0  = good
         1  = marginal
         2  = snow/ice
         3  = cloud
    """
    qa = image.select("SummaryQA")
    mask = qa.lte(1)  # keep good (0) and marginal (1)
    return image.updateMask(mask)


def build_mandi_fc(mandis_df: pd.DataFrame, buffer_m: float) -> ee.FeatureCollection:
    """Build a FeatureCollection of buffered mandi points."""
    features = []
    for row in mandis_df.itertuples(index=False):
        point = ee.Geometry.Point([float(row.longitude), float(row.latitude)])
        region = point.buffer(buffer_m)
        features.append(
            ee.Feature(
                region,
                {
                    "mandi_id": row.mandi_id,
                    "market_code": str(row.market_code),
                    "mandi_name": row.mandi_name,
                    "district": row.district,
                    "state": row.state,
                    "latitude": float(row.latitude),
                    "longitude": float(row.longitude),
                },
            )
        )
    return ee.FeatureCollection(features)


# ---------------------------------------------------------------------------
# Batch extraction
# ---------------------------------------------------------------------------
def _extract_month(
    mandi_fc: ee.FeatureCollection,
    collection: ee.ImageCollection,
    start: str,
    end: str,
    scale: int,
    tag: str,
) -> list[dict]:
    """Extract one month's median NDVI composite over mandi regions."""
    filtered = (
        collection
        .filterDate(start, end)
        .filterBounds(mandi_fc.geometry())
        .map(modis_qa_mask)
        .map(modis_ndvi_scale)
    )

    scene_count = filtered.size().getInfo()
    if scene_count == 0:
        log.debug("  %s %s: 0 scenes — skipping", tag, start[:7])
        return []

    # 16-day composites — take the median across the month
    composite = filtered.median()

    reduced = composite.reduceRegions(
        collection=mandi_fc,
        reducer=ee.Reducer.mean(),
        scale=scale,
        tileScale=4,
    )

    result = reduced.getInfo()
    rows = []
    for feat in result["features"]:
        props = feat["properties"]
        val = props.get("mean")
        if val is None or np.isnan(val):
            continue
        rows.append(
            {
                "mandi_id": props["mandi_id"],
                "market_code": props["market_code"],
                "mandi_name": props["mandi_name"],
                "district": props["district"],
                "state": props["state"],
                "latitude": props["latitude"],
                "longitude": props["longitude"],
                "date": start,
                "ndvi": round(val, 6),
                "scene_count": scene_count,
            }
        )
    return rows


def fetch_period(
    mandis_df: pd.DataFrame,
    collection: ee.ImageCollection,
    start_date: str,
    end_date: str,
    scale: int,
    buffer_m: int,
    batch_size: int,
    tag: str,
) -> pd.DataFrame:
    """Fetch NDVI for every mandi across a date range, month by month."""
    all_rows: list[dict] = []
    current = pd.Timestamp(start_date)
    end = pd.Timestamp(end_date)
    total_months = (end.year - current.year) * 12 + (end.month - current.month)
    month_idx = 0

    while current < end:
        month_start = current.strftime("%Y-%m-%d")
        # next month first day
        next_month = (current + pd.offsets.MonthBegin(1))
        if next_month > end:
            next_month = end
        month_end = next_month.strftime("%Y-%m-%d")

        month_idx += 1
        log.info(
            "[%s] Month %d/%d  %s  (mandis=%d)",
            tag, month_idx, total_months, current.strftime("%Y-%m"),
            len(mandis_df),
        )

        for batch_start in range(0, len(mandis_df), batch_size):
            batch_end = min(batch_start + batch_size, len(mandis_df))
            batch_df = mandis_df.iloc[batch_start:batch_end]

            mandi_fc = build_mandi_fc(batch_df, buffer_m)

            try:
                rows = _extract_month(
                    mandi_fc, collection, month_start, month_end, scale, tag,
                )
                all_rows.extend(rows)
                log.info(
                    "    Batch %d–%d: %d rows, scenes=%s",
                    batch_start + 1, batch_end,
                    len(rows),
                    rows[0]["scene_count"] if rows else 0,
                )
            except Exception as exc:
                log.error(
                    "    Batch %d–%d ERROR: %s: %s",
                    batch_start + 1, batch_end,
                    type(exc).__name__, exc,
                )
                # GEE rate-limit back-off
                time.sleep(5)

        current = next_month

    if not all_rows:
        return pd.DataFrame()

    df = pd.DataFrame(all_rows)
    df["date"] = pd.to_datetime(df["date"])
    return df.sort_values(["mandi_id", "date"]).reset_index(drop=True)


# ---------------------------------------------------------------------------
# Anomaly computation
# ---------------------------------------------------------------------------
def compute_anomalies(baseline: pd.DataFrame, current: pd.DataFrame) -> pd.DataFrame:
    """Compute NDVI z-score anomaly for each current observation.

    anomaly = (ndvi_current - ndvi_baseline_mean) / ndvi_baseline_std

    where baseline_mean and baseline_std are per-mandi, per-DOY computed
    from the multi-year baseline.
    """
    # Add day-of-year to both
    baseline = baseline.copy()
    baseline["doy"] = baseline["date"].dt.dayofyear
    current = current.copy()
    current["doy"] = current["date"].dt.dayofyear

    # Per-mandi, per-DOY baseline stats
    stats = (
        baseline.groupby(["mandi_id", "doy"])["ndvi"]
        .agg(["mean", "std", "count"])
        .rename(columns={"mean": "ndvi_baseline_mean", "std": "ndvi_baseline_std", "count": "n_baseline_years"})
        .reset_index()
    )

    # Merge onto current observations
    merged = current.merge(stats, on=["mandi_id", "doy"], how="left")

    # Compute z-score
    merged["ndvi_anomaly"] = (
        (merged["ndvi"] - merged["ndvi_baseline_mean"])
        / merged["ndvi_baseline_std"].clip(lower=1e-6)
    )

    # Flag where we couldn't compute anomaly (no baseline for that DOY)
    merged["has_anomaly"] = merged["ndvi_baseline_mean"].notna().astype(int)

    return merged


def build_baseline_table(raw_baseline: pd.DataFrame) -> pd.DataFrame:
    """Aggregate raw baseline into per-mandi, per-DOY statistics."""
    raw_baseline = raw_baseline.copy()
    raw_baseline["doy"] = raw_baseline["date"].dt.dayofyear

    # Per-mandi DOY stats across all years
    meta_cols = [
        "mandi_id", "market_code", "mandi_name", "district",
        "state", "latitude", "longitude",
    ]

    stats = (
        raw_baseline.groupby(["mandi_id", "doy"])["ndvi"]
        .agg(["mean", "std", "count"])
        .rename(columns={"mean": "ndvi_mean", "std": "ndvi_std", "count": "n_years"})
        .reset_index()
    )

    # Join back metadata (take first occurrence per mandi)
    meta = raw_baseline[meta_cols].drop_duplicates("mandi_id")
    stats = stats.merge(meta, on="mandi_id", how="left")

    # Reorder columns
    cols = meta_cols + ["doy", "ndvi_mean", "ndvi_std", "n_years"]
    return stats[[c for c in cols if c in stats.columns]]


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser(
        description="AgriVault MODIS NDVI multi-year extraction",
    )
    parser.add_argument("--limit", type=int, default=None,
                        help="Process only first N mandis (for testing)")
    parser.add_argument("--batch-size", type=int, default=250,
                        help="Number of mandis per GEE request")
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s  %(levelname)-8s  %(message)s",
        datefmt="%H:%M:%S",
    )

    cfg = load_config()
    modis_cfg = cfg["modis_ndvi"]
    project_id = cfg["project_id"]

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    if not MANDI_FILE.exists():
        sys.exit(f"Mandi file not found: {MANDI_FILE}")

    initialize_gee(project_id)

    mandis = pd.read_csv(MANDI_FILE)
    if args.limit is not None:
        mandis = mandis.head(args.limit)

    collection = ee.ImageCollection(modis_cfg["dataset"])

    # ------------------------------------------------------------------
    print()
    print("=" * 70)
    print("  AgriVault MODIS NDVI Multi-Year Extraction (NASA/GEE)")
    print("=" * 70)
    print(f"  Mandis          : {len(mandis):,}")
    print(f"  Dataset         : {modis_cfg['dataset']}")
    print(f"  Baseline period : {modis_cfg['baseline_start']} -> {modis_cfg['baseline_end']}")
    print(f"  Current year    : {modis_cfg['current_start']} -> {modis_cfg['current_end']}")
    print(f"  Buffer          : {modis_cfg['buffer_meters']} m")
    print(f"  Resolution      : {modis_cfg['scale_meters']} m")
    print(f"  Batch size      : {args.batch_size}")
    print("=" * 70)

    # ── Step 1: Fetch baseline (2021–2024) ────────────────────────────
    log.info("PHASE 1/3 — Fetching baseline NDVI (%s -> %s) …",
             modis_cfg["baseline_start"], modis_cfg["baseline_end"])

    baseline_raw = fetch_period(
        mandis_df=mandis,
        collection=collection,
        start_date=modis_cfg["baseline_start"],
        end_date=modis_cfg["baseline_end"],
        scale=modis_cfg["scale_meters"],
        buffer_m=modis_cfg["buffer_meters"],
        batch_size=args.batch_size,
        tag="baseline",
    )

    if baseline_raw.empty:
        sys.exit("ERROR: No baseline NDVI data retrieved. Check GEE auth and mandi coverage.")

    baseline_raw_path = OUTPUT_DIR / "ndvi_modis_baseline_raw.csv"
    baseline_raw.to_csv(baseline_raw_path, index=False)
    log.info("Baseline raw saved: %s  (%d rows)", baseline_raw_path, len(baseline_raw))

    # ── Step 2: Build baseline statistics ──────────────────────────────
    log.info("PHASE 2/3 — Computing per-mandi, per-DOY baseline stats …")
    baseline_stats = build_baseline_table(baseline_raw)

    baseline_path = OUTPUT_DIR / "ndvi_modis_baseline.csv"
    baseline_stats.to_csv(baseline_path, index=False)
    log.info("Baseline stats saved: %s  (%d mandi-DOY rows)", baseline_path, len(baseline_stats))

    # ── Step 3: Fetch current year (2025) ─────────────────────────────
    log.info("PHASE 3/3 — Fetching current NDVI (%s -> %s) …",
             modis_cfg["current_start"], modis_cfg["current_end"])

    current_raw = fetch_period(
        mandis_df=mandis,
        collection=collection,
        start_date=modis_cfg["current_start"],
        end_date=modis_cfg["current_end"],
        scale=modis_cfg["scale_meters"],
        buffer_m=modis_cfg["buffer_meters"],
        batch_size=args.batch_size,
        tag="current",
    )

    if current_raw.empty:
        sys.exit("ERROR: No current NDVI data retrieved.")

    # ── Step 4: Compute anomalies ─────────────────────────────────────
    log.info("Computing NDVI z-score anomalies …")
    current_with_anomaly = compute_anomalies(baseline_raw, current_raw)

    current_path = OUTPUT_DIR / "ndvi_modis_current.csv"
    current_with_anomaly.to_csv(current_path, index=False)
    log.info("Current + anomaly saved: %s  (%d rows)", current_path, len(current_with_anomaly))

    # ── Summary ───────────────────────────────────────────────────────
    n_mandis = current_with_anomaly["mandi_id"].nunique()
    pct_with_anomaly = current_with_anomaly["has_anomaly"].mean() * 100
    print()
    print("=" * 70)
    print("  MODIS NDVI EXTRACTION COMPLETE")
    print("=" * 70)
    print(f"  Baseline file    : {baseline_path}")
    print(f"  Current file     : {current_path}")
    print(f"  Mandis covered   : {n_mandis:,}")
    print(f"  Baseline rows    : {len(baseline_raw):,}")
    print(f"  Current rows     : {len(current_with_anomaly):,}")
    print(f"  Anomaly coverage : {pct_with_anomaly:.1f}% of obs have baseline")
    print(f"  Date range       : {current_with_anomaly['date'].min().date()} -> "
          f"{current_with_anomaly['date'].max().date()}")
    if "ndvi_anomaly" in current_with_anomaly.columns:
        valid = current_with_anomaly["ndvi_anomaly"].dropna()
        if len(valid) > 0:
            print(f"  Anomaly range    : {valid.min():.3f} -> {valid.max():.3f}")
            print(f"  Anomaly mean     : {valid.mean():.3f}")
    print("=" * 70)


if __name__ == "__main__":
    main()
