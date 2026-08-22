"""
AgriVault – Fetch Historical APMC Market Prices from Agmarknet (2021–2025)
=============================================================================
Pulls daily modal prices, arrivals, and market info from the Agmarknet portal
(data.gov.in / Government of India) for all states, major commodities, and
mandis in the existing reference file.

Data sources (tried in order):
  1. Manual CSV files in data/raw/apmc/manual/ -- highest reliability
  2. data.gov.in API v3 (CKAN-style) -- structured, paginated
  3. Agmarknet CSV export endpoint -- fallback for older records

Manual download instructions:
  1. Go to https://agmarknet.gov.in
  2. Click 'Price & Arrivals' > 'Commodity-Wise Daily Report'
  3. Select state + commodity + date range (2021-01-01 to 2025-12-31)
  4. Click 'Get Report' then 'Download CSV'
  5. Save files to data/raw/apmc/manual/ (one per state-commodity combo)
  6. Run: python -m src.ingestion.fetch_apmc_history --manual-only

Output
------
    data/raw/apmc/apmc_market_prices_2021_2025.csv
    Columns: state_name, state_code, district_name, district_code,
             market_center, market_code, commodity_type, commodity, variety,
             origin, min_price, max_price, modal_price, report_date,
             arrivals_tonnes, arrivals_unit, price_unit, latitude, longitude

Run
---
    python -m src.ingestion.fetch_apmc_history
    python -m src.ingestion.fetch_apmc_history --start 2021-01-01 --end 2023-12-31
    python -m src.ingestion.fetch_apmc_history --states KARNATAKA MAHARASHTRA

Prerequisites:
    pip install requests pandas tqdm pyyaml
"""

from __future__ import annotations

import argparse
import csv
import io
import logging
import sys
import time
from datetime import date, timedelta
from pathlib import Path

import pandas as pd
import requests
import yaml
from tqdm import tqdm

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
PROJECT_ROOT = Path(__file__).resolve().parents[2]
CONFIG_FILE = PROJECT_ROOT / "configs" / "aws_config.yaml"
MANDI_FILE = PROJECT_ROOT / "data" / "reference" / "mandi_locations.csv"
OUTPUT_DIR = PROJECT_ROOT / "data" / "raw" / "apmc"

log = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Agmarknet / data.gov.in API configuration
# ---------------------------------------------------------------------------

# data.gov.in API endpoint for Agmarknet daily prices
# Dataset: "Daily Market Prices of Various Commodities in APMC Markets"
# This dataset is updated daily and contains historical data from ~2019 onwards.
DATAGOV_API = "https://data.gov.in/backend/dmspublic/v1"
DATAGOV_RESOURCE_ID = "8d032d97-810d-495e-a480-8fcbf2d35b43"  # Agmarknet daily

# Agmarknet direct API (JSON responses)
AGMARKNET_API = "https://agmarknet.gov.in/api/v1"

# Rate-limiting: be respectful to government APIs
REQUEST_DELAY_S = 0.5
MAX_RETRIES = 3
RETRY_BACKOFF = 5  # seconds

HEADERS = {
    "User-Agent": "AgriVault/1.0 (research; ndvi-anomaly-pipeline)",
    "Accept": "application/json, text/csv, */*",
}

# Major commodities tracked by AgriVault
MAJOR_COMMODITIES = [
    "Rice", "Wheat", "Maize", "Bajra", "Jowar", "Ragi",
    "Soyabean", "Groundnut", "Sunflower", "Mustard",
    "Cotton", "Sugarcane",
    "Onion", "Potato", "Tomato",
    "Banana", "Mango",
    "Chilli", "Turmeric", "Ginger",
    "Arhar", "Moong", "Urad",
]


# ---------------------------------------------------------------------------
# API interaction helpers
# ---------------------------------------------------------------------------
def _api_get(url: str, params: dict | None = None, timeout: int = 60) -> dict | list | None:
    """GET with retries and back-off."""
    for attempt in range(1, MAX_RETRIES + 1):
        try:
            resp = requests.get(url, params=params, headers=HEADERS, timeout=timeout)
            if resp.status_code == 429:
                wait = RETRY_BACKOFF * attempt
                log.warning("Rate-limited (429). Waiting %ds …", wait)
                time.sleep(wait)
                continue
            resp.raise_for_status()
            return resp.json()
        except requests.exceptions.RequestException as exc:
            log.warning("Request failed (attempt %d/%d): %s", attempt, MAX_RETRIES, exc)
            if attempt < MAX_RETRIES:
                time.sleep(RETRY_BACKOFF * attempt)
    return None


def _date_range_chunks(start: date, end: date, chunk_days: int = 90):
    """Yield (chunk_start, chunk_end) tuples of `chunk_days` each."""
    current = start
    while current <= end:
        chunk_end = min(current + timedelta(days=chunk_days - 1), end)
        yield current, chunk_end
        current = chunk_end + timedelta(days=1)


# ---------------------------------------------------------------------------
# data.gov.in resource download
# ---------------------------------------------------------------------------
def fetch_from_datagov(
    start: date, end: date, states: list[str] | None = None,
) -> pd.DataFrame:
    """Try fetching from data.gov.in CKAN-style API.

    data.gov.in has two patterns:
      A) Resource download API -- single CSV for the full dataset
      B) Search/filter API -- parameterized queries

    We attempt the filtered API first (smaller payloads), then fall back to
    bulk resource download.
    """
    all_frames: list[pd.DataFrame] = []

    # ── Strategy A: Filtered search via data.gov.in ────────────────────
    log.info("Trying data.gov.in filtered API …")
    for chunk_start, chunk_end in _date_range_chunks(start, end, chunk_days=90):
        params = {
            "filters[report_date]": f"{chunk_start.isoformat()}|{chunk_end.isoformat()}",
            "page": 1,
            "size": 10000,
        }
        if states:
            # data.gov.in uses comma-separated state filter
            params["filters[state_name]"] = "|".join(s.upper() for s in states)

        url = f"{DATAGOV_API}/resources/{DATAGOV_RESOURCE_ID}/data"
        data = _api_get(url, params=params)

        if data and isinstance(data, dict):
            records = data.get("data", data.get("records", []))
            if records:
                df = pd.DataFrame(records)
                all_frames.append(df)
                log.info("  data.gov.in chunk %s -> %s: %d rows",
                         chunk_start, chunk_end, len(df))

        time.sleep(REQUEST_DELAY_S)

    if all_frames:
        return pd.concat(all_frames, ignore_index=True)

    # ── Strategy B: Agmarknet search API ───────────────────────────────
    log.info("Falling back to Agmarknet search API …")
    return _fetch_agmarknet_search(start, end, states)


def _fetch_agmarknet_search(
    start: date, end: date, states: list[str] | None = None,
) -> pd.DataFrame:
    """Fetch from Agmarknet's own search endpoint (JSON).

    Agmarknet provides an API at:
        https://agmarknet.gov.in/api/market/search
    which returns daily prices filtered by state, commodity, and date.
    """
    all_rows: list[dict] = []

    # Get unique states from mandi reference if not provided
    if states is None:
        mandis = pd.read_csv(MANDI_FILE)
        states = mandis["state"].str.title().unique().tolist()

    # Query each (state, commodity) combination
    commodities = MAJOR_COMMODITIES

    total_combos = len(states) * len(commodities)
    log.info("Agmarknet search: %d states × %d commodities = %d queries",
             len(states), len(commodities), total_combos)

    progress = tqdm(total=total_combos, desc="Agmarknet queries", unit="query")

    for state in sorted(states):
        for commodity in sorted(commodities):
            progress.set_postfix_str(f"{state[:15]}/{commodity[:12]}")

            # Agmarknet search endpoint
            search_url = f"{AGMARKNET_API}/commodityprice"
            params = {
                "State": state.upper(),
                "Commodity": commodity.upper(),
                "StartDate": start.strftime("%d-%b-%Y"),
                "EndDate": end.strftime("%d-%b-%Y"),
            }

            data = _api_get(search_url, params=params, timeout=30)

            if data and isinstance(data, dict):
                records = data.get("data", data.get("records", []))
                if isinstance(records, list):
                    for rec in records:
                        rec["state_name"] = state.upper()
                        rec["commodity"] = commodity
                    all_rows.extend(records)
            elif data and isinstance(data, list):
                for rec in data:
                    if isinstance(rec, dict):
                        rec["state_name"] = state.upper()
                        rec["commodity"] = commodity
                all_rows.extend(data)

            progress.update(1)
            time.sleep(REQUEST_DELAY_S)

    progress.close()

    if not all_rows:
        return pd.DataFrame()

    return pd.DataFrame(all_rows)


# ---------------------------------------------------------------------------
# Agmarknet CSV bulk export (last resort)
# ---------------------------------------------------------------------------
def _fetch_agmarknet_csv_bulk(
    start: date, end: date, states: list[str] | None = None,
) -> pd.DataFrame:
    """Download Agmarknet daily price CSVs by date range.

    Agmarknet provides CSV exports at URLs like:
        https://agmarknet.gov.in/MarketL1/GetMarketL1DataCSV
    This requires form POST with specific parameters.
    """
    all_frames: list[pd.DataFrame] = []

    if states is None:
        mandis = pd.read_csv(MANDI_FILE)
        states = mandis["state"].str.title().unique().tolist()

    for state in sorted(states):
        for chunk_start, chunk_end in _date_range_chunks(start, end, chunk_days=30):
            log.info("CSV bulk: %s  %s -> %s", state, chunk_start, chunk_end)

            try:
                form_data = {
                    "State": state.upper(),
                    "StartDate": chunk_start.strftime("%d-%b-%Y"),
                    "EndDate": chunk_end.strftime("%d-%b-%Y"),
                }
                resp = requests.post(
                    f"{AGMARKNET_API}/GetMarketL1DataCSV",
                    data=form_data,
                    headers=HEADERS,
                    timeout=60,
                )
                if resp.status_code == 200 and resp.text.strip():
                    df = pd.read_csv(io.StringIO(resp.text))
                    if len(df) > 0:
                        all_frames.append(df)
                        log.info("  CSV chunk: %d rows", len(df))

            except Exception as exc:
                log.warning("  CSV chunk failed: %s", exc)

            time.sleep(REQUEST_DELAY_S)

    return pd.concat(all_frames, ignore_index=True) if all_frames else pd.DataFrame()


# ---------------------------------------------------------------------------
# Normalise raw records to AgriVault APMC schema
# ---------------------------------------------------------------------------
def normalise_apmc(raw: pd.DataFrame) -> pd.DataFrame:
    """Map raw Agmarknet columns to the schema used by clean_apmc.py."""
    df = raw.copy()

    # Column name normalisation (handle various Agmarknet formats)
    col_map = {}
    for col in df.columns:
        cl = col.lower().strip()
        if cl in ("state_name", "state", "statename"):
            col_map[col] = "state_name"
        elif cl in ("district_name", "district", "districtname"):
            col_map[col] = "district_name"
        elif cl in ("market_center", "market", "market_name", "marketname",
                     "market_center_name"):
            col_map[col] = "market_center"
        elif cl in ("market_code",):
            col_map[col] = "market_code"
        elif cl in ("commodity_type",):
            col_map[col] = "commodity_type"
        elif cl in ("commodity", "commodity_name", "commodityname"):
            col_map[col] = "commodity"
        elif cl in ("variety", "variety_name"):
            col_map[col] = "variety"
        elif cl in ("origin",):
            col_map[col] = "origin"
        elif cl in ("min_price", "minimum_price", "min",
                     "min price (rs/quintal)"):
            col_map[col] = "min_price"
        elif cl in ("max_price", "maximum_price", "max",
                     "max price (rs/quintal)"):
            col_map[col] = "max_price"
        elif cl in ("modal_price", "model_price", "prices",
                     "modal price (rs/quintal)"):
            col_map[col] = "modal_price"
        elif cl in ("report_date", "date", "arrival_date", "price_date",
                     "report date"):
            col_map[col] = "report_date"
        elif cl in ("arrivals_tonnes", "arrival", "arrivals", "arrival_tonnes",
                     "arrivals(tonnes)"):
            col_map[col] = "arrivals_tonnes"
        elif cl in ("state_code",):
            col_map[col] = "state_code"
        elif cl in ("district_code",):
            col_map[col] = "district_code"
        elif cl in ("commodity_code",):
            col_map[col] = "commodity_code"

    df = df.rename(columns=col_map)

    # Ensure required columns exist
    required = ["state_name", "district_name", "market_center",
                "commodity", "modal_price", "report_date"]
    for col in required:
        if col not in df.columns:
            log.warning("Missing column '%s' -- filling with empty", col)
            df[col] = ""

    # Standardise state names
    df["state_name"] = df["state_name"].astype(str).str.strip().str.upper()
    df["district_name"] = df["district_name"].astype(str).str.strip().str.upper()
    df["market_center"] = df["market_center"].astype(str).str.strip()
    df["commodity"] = df["commodity"].astype(str).str.strip().str.upper()

    # Parse prices to numeric
    for col in ("min_price", "max_price", "modal_price", "arrivals_tonnes"):
        if col in df.columns:
            df[col] = pd.to_numeric(
                df[col].astype(str).str.replace(",", ""), errors="coerce"
            )

    # Parse dates -- try multiple formats
    df["report_date"] = pd.to_datetime(
        df["report_date"].astype(str).str.strip(),
        dayfirst=True, errors="coerce",
    )
    df["report_date"] = df["report_date"].dt.strftime("%Y-%m-%d")

    # Drop rows with no date or price
    df = df[df["report_date"].notna() & df["modal_price"].notna()]
    df = df[df["modal_price"] > 0]

    return df


# ---------------------------------------------------------------------------
# Enrich with mandi metadata (lat/lon, market_code)
# ---------------------------------------------------------------------------
def enrich_with_mandi_meta(df: pd.DataFrame) -> pd.DataFrame:
    """Join mandi reference to add lat/lon and market_code where possible."""
    if not MANDI_FILE.exists():
        log.warning("mandi_locations.csv not found -- skipping enrichment")
        return df

    mandis = pd.read_csv(MANDI_FILE)
    mandis["mandi_name_upper"] = mandis["mandi_name"].str.upper().str.strip()
    mandis["state_upper"] = mandis["state"].str.upper().str.strip()
    mandis["district_upper"] = mandis["district"].str.upper().str.strip()

    # Build a lookup keyed by (state, mandi_name)
    lookup_df = mandis.drop_duplicates(["state_upper", "mandi_name_upper"], keep="first")
    lookup = lookup_df.set_index(["state_upper", "mandi_name_upper"])[
        ["latitude", "longitude", "market_code", "mandi_id"]
    ].to_dict("index")

    # Also build district-level fallback (deduplicate first)
    district_lookup = (
        mandis.drop_duplicates(["state_upper", "district_upper"], keep="first")
        .set_index(["state_upper", "district_upper"])[["latitude", "longitude"]]
        .to_dict("index")
    )

    def _lookup_latlon(row):
        key = (str(row.get("state_name", "")).upper().strip(),
               str(row.get("market_center", "")).upper().strip())
        if key in lookup:
            m = lookup[key]
            return pd.Series({
                "latitude": m["latitude"],
                "longitude": m["longitude"],
                "market_code": m.get("market_code", ""),
                "mandi_id": m.get("mandi_id", ""),
            })
        # District fallback
        dkey = (str(row.get("state_name", "")).upper().strip(),
                str(row.get("district_name", "")).upper().strip())
        if dkey in district_lookup:
            m = district_lookup[dkey]
            return pd.Series({
                "latitude": m["latitude"],
                "longitude": m["longitude"],
                "market_code": "",
                "mandi_id": "",
            })
        return pd.Series({
            "latitude": None, "longitude": None,
            "market_code": "", "mandi_id": "",
        })

    enriched = df.join(df.apply(_lookup_latlon, axis=1))
    n_matched = enriched["mandi_id"].notna().sum()
    log.info("Mandi meta enrichment: %d / %d rows matched to mandi_id",
             n_matched, len(enriched))

    return enriched


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def load_manual_csvs(manual_dir: Path) -> pd.DataFrame:
    """Load manually downloaded Agmarknet CSV files.

    Expects files in data/raw/apmc/manual/ with any naming convention.
    Common Agmarknet CSV formats have columns like:
        State, District, Market, Commodity, Variety, Arrivals (Tonnes),
        Min Price, Max Price, Modal Price, Report Date
    """
    if not manual_dir.exists():
        return pd.DataFrame()

    csv_files = list(manual_dir.glob("*.csv")) + list(manual_dir.glob("*.CSV"))
    if not csv_files:
        return pd.DataFrame()

    log.info("Found %d manual CSV files in %s", len(csv_files), manual_dir)
    frames = []
    for f in sorted(csv_files):
        try:
            df = pd.read_csv(f, encoding="utf-8")
            log.info("  %s: %d rows, %d cols", f.name, len(df), len(df.columns))
            frames.append(df)
        except Exception as exc:
            log.warning("  %s: FAILED to read: %s", f.name, exc)

    if not frames:
        return pd.DataFrame()

    return pd.concat(frames, ignore_index=True)


def main():
    parser = argparse.ArgumentParser(
        description="AgriVault APMC historical data fetcher (Agmarknet / data.gov.in)",
    )
    parser.add_argument("--start", type=str, default="2021-01-01",
                        help="Start date (YYYY-MM-DD)")
    parser.add_argument("--end", type=str, default="2025-12-31",
                        help="End date (YYYY-MM-DD)")
    parser.add_argument("--states", nargs="*", default=None,
                        help="Limit to specific states (space-separated)")
    parser.add_argument("--commodities", nargs="*", default=None,
                        help="Limit to specific commodities")
    parser.add_argument("--manual-only", action="store_true",
                        help="Only process manual CSVs (skip API)")
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s  %(levelname)-8s  %(message)s",
        datefmt="%H:%M:%S",
    )

    start = date.fromisoformat(args.start)
    end = date.fromisoformat(args.end)
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    if args.commodities:
        global MAJOR_COMMODITIES
        MAJOR_COMMODITIES = args.commodities

    print()
    print("=" * 70)
    print("  AgriVault APMC Historical Data Fetcher")
    print("=" * 70)
    print(f"  Date range : {start} -> {end}")
    print(f"  States     : {args.states or 'ALL'}")
    print(f"  Commodities: {len(MAJOR_COMMODITIES)} tracked")
    print("=" * 70)

    # ── Try manual CSVs first ────────────────────────────────────────
    manual_dir = OUTPUT_DIR / "manual"
    raw = load_manual_csvs(manual_dir)

    if not raw.empty:
        log.info("Loaded %d rows from manual CSVs", len(raw))
    elif args.manual_only:
        sys.exit(
            f"ERROR: No manual CSVs found in {manual_dir}\n"
            "Download CSVs from https://agmarknet.gov.in and save them there.\n"
            "See script header for detailed instructions."
        )
    else:
        # ── Fetch from data.gov.in / Agmarknet ──────────────────────
        raw = fetch_from_datagov(start, end, states=args.states)

        if raw.empty:
            log.warning("Primary API returned no data. Trying CSV bulk export...")
            raw = _fetch_agmarknet_csv_bulk(start, end, states=args.states)

    if raw.empty:
        print()
        print("ERROR: No data retrieved from any source.")
        print()
        print("The Agmarknet 2.0 portal requires browser-based access.")
        print("To get historical APMC data:")
        print()
        print("  1. Go to https://agmarknet.gov.in")
        print("  2. Click 'Price & Arrivals' > 'Commodity-Wise Daily Report'")
        print("  3. Select state + commodity + date range (2021-2025)")
        print("  4. Click 'Get Report' then 'Download CSV'")
        print(f"  5. Save files to: {manual_dir}/")
        print(f"  6. Re-run: py -m src.ingestion.fetch_apmc_history --manual-only")
        print()
        print("Alternatively, the existing 2025 data can still be used:")
        print(f"  data/raw/apmc/apmc_market_prices.csv")
        sys.exit(1)

    log.info("Raw records fetched: %d", len(raw))
    log.info("Columns: %s", list(raw.columns))

    # ── Normalise to AgriVault schema ──────────────────────────────────
    log.info("Normalising to AgriVault APMC schema …")
    cleaned = normalise_apmc(raw)
    log.info("After normalisation: %d rows", len(cleaned))

    # ── Enrich with mandi metadata ─────────────────────────────────────
    cleaned = enrich_with_mandi_meta(cleaned)

    # ── Deduplicate ────────────────────────────────────────────────────
    dedup_keys = ["report_date", "state_name", "district_name",
                  "market_center", "commodity"]
    existing_dedup = [c for c in dedup_keys if c in cleaned.columns]
    before = len(cleaned)
    cleaned = cleaned.drop_duplicates(existing_dedup)
    log.info("Dedup: %d -> %d rows", before, len(cleaned))

    # ── Write output ───────────────────────────────────────────────────
    output_file = OUTPUT_DIR / f"apmc_market_prices_{start.year}_{end.year}.csv"
    cleaned.to_csv(output_file, index=False)
    log.info("Saved: %s", output_file)

    # ── Summary ────────────────────────────────────────────────────────
    n_states = cleaned["state_name"].nunique() if "state_name" in cleaned.columns else 0
    n_commodities = cleaned["commodity"].nunique() if "commodity" in cleaned.columns else 0
    n_dates = cleaned["report_date"].nunique() if "report_date" in cleaned.columns else 0
    print()
    print("=" * 70)
    print("  APMC HISTORICAL FETCH COMPLETE")
    print("=" * 70)
    print(f"  Output file : {output_file}")
    print(f"  Total rows  : {len(cleaned):,}")
    print(f"  States      : {n_states}")
    print(f"  Commodities : {n_commodities}")
    print(f"  Date span   : {n_dates} unique dates")
    if "report_date" in cleaned.columns:
        valid_dates = pd.to_datetime(cleaned["report_date"], errors="coerce")
        print(f"  First date  : {valid_dates.min().date()}")
        print(f"  Last date   : {valid_dates.max().date()}")
    print("=" * 70)


if __name__ == "__main__":
    main()
