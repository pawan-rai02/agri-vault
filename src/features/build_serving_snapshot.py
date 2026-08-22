"""
AgriVault - Build Serving Snapshot (Feature Store for Live API)
================================================================
Runs after the daily Gold price_features build.  Takes the latest row per
(mandi_id, commodity) and writes a small, fast-loading snapshot for the
live prediction API to read - one row per active mandi-commodity pair.

This is completely different from the Gold price_features table, which has
one row per historical date; this table has exactly one row per active
mandi-commodity pair: "here's the feature vector as of today."

Input  : s3://agrivault-lake-pawan/features/price_features/  (Gold layer)
Output : s3://agrivault-lake-pawan/features/serving_snapshot/latest.parquet

Run
---
    python -m src.features.build_serving_snapshot

Schedule with Windows Task Scheduler to run daily after the Gold pipeline.
"""

from __future__ import annotations

import logging

import pandas as pd

from src.storage.s3_client import S3Client

log = logging.getLogger(__name__)

# S3 key for the serving snapshot
_SNAPSHOT_KEY = "features/serving_snapshot/latest.parquet"


def build_snapshot(s3: S3Client | None = None) -> pd.DataFrame:
    """Build the serving snapshot from Gold price_features.

    Steps
    -----
    1. Read the full Gold price_features table from S3.
    2. Sort by date descending so the most recent row is first per group.
    3. Group by (mandi_id, commodity) and take the first row (latest date).
    4. Drop target columns that leak future info (target_price_*).
    5. Write the result as a small Parquet to S3.

    Returns the snapshot DataFrame for inspection / testing.
    """
    if s3 is None:
        s3 = S3Client()

    # -- 1. Read Gold price_features --------------------------------------
    log.info("Loading Gold price_features from S3...")
    gold = s3.read_parquet_s3("features/price_features/")
    log.info("  - %d rows, %d cols", len(gold), len(gold.columns))

    if gold.empty:
        raise ValueError("Gold price_features is empty - nothing to snapshot")

    # -- 2. Sort by date so we can take the latest per group --------------
    gold = gold.sort_values("date", ascending=False)

    # -- 3. Take latest row per (mandi_id, commodity) ---------------------
    # groupby().head(1) is faster than groupby().tail(1) on desc-sorted data
    snapshot = (
        gold
        .groupby(["mandi_id", "commodity"], as_index=False)
        .head(1)
        .copy()
    )
    snapshot = snapshot.sort_values(["mandi_id", "commodity"]).reset_index(drop=True)

    n_mandis = snapshot["mandi_id"].nunique()
    n_commodities = snapshot["commodity"].nunique()
    n_pairs = len(snapshot)
    log.info(
        "Snapshot: %d rows (%d unique mandis × %d unique commodities)",
        n_pairs, n_mandis, n_commodities,
    )

    # -- 4. Drop target columns (future-leaking) -------------------------
    target_cols = [c for c in snapshot.columns if c.startswith("target_price_")]
    if target_cols:
        snapshot = snapshot.drop(columns=target_cols)
        log.info("Dropped target columns: %s", target_cols)

    # -- 5. Write to S3 --------------------------------------------------
    s3.write_parquet_s3(snapshot, _SNAPSHOT_KEY)
    log.info(
        "✓ Serving snapshot written: s3://%s/%s (%d rows, %d cols)",
        s3.bucket, _SNAPSHOT_KEY, len(snapshot), len(snapshot.columns),
    )

    return snapshot


def load_snapshot(s3: S3Client | None = None) -> pd.DataFrame:
    """Load the serving snapshot from S3 into a DataFrame.

    Called by the Flask app at startup to get the latest feature vectors
    for live prediction.
    """
    if s3 is None:
        s3 = S3Client()

    log.info("Loading serving snapshot from s3://%s/%s", s3.bucket, _SNAPSHOT_KEY)
    df = s3.read_parquet(_SNAPSHOT_KEY)
    log.info("  - %d rows, %d cols", len(df), len(df.columns))
    return df


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s  %(levelname)-8s  %(message)s",
        datefmt="%H:%M:%S",
    )

    snapshot = build_snapshot()

    print("\n" + "=" * 60)
    print("Serving snapshot built successfully")
    print("=" * 60)
    print(f"\nRows    : {len(snapshot):,}")
    print(f"Columns : {len(snapshot.columns)}")
    print(f"Commodities: {snapshot['commodity'].nunique()}")
    print(f"States   : {snapshot['state'].nunique() if 'state' in snapshot.columns else 'N/A'}")
    print(f"\nSample columns: {list(snapshot.columns[:15])}")
    print(f"\nSample data:")
    print(snapshot.head(5).to_string(index=False))
