"""
AgriVault – Location Resolver
==============================
Maps user input (state, district, market name) to a known mandi_id.

Resolution strategy (in order):
    1. Exact match on mandi_name + state (case-insensitive)
    2. District + state match (returns first mandi in that district)
    3. State-only match (returns first mandi in that state)
    4. Nearest mandi by lat/lon (geo fallback)

Always returns which strategy was used — important for user trust
("this prediction uses the nearest available mandi, 12km away").

Run tests
---------
    python -m pytest tests/test_location_resolver.py -v

Usage
-----
    from src.serving.location_resolver import resolve_mandi

    mandi_id, resolution = resolve_mandi(state="MAHARASHTRA", district="PUNE")
"""

from __future__ import annotations

import logging
from math import atan2, cos, radians, sin, sqrt

import pandas as pd

log = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

_EARTH_RADIUS_KM = 6371.0

# Module-level cache for the mandi locations CSV
_mandi_locs: pd.DataFrame | None = None


def _load_mandi_locs() -> pd.DataFrame:
    """Load mandi_locations.csv from the data/reference directory.

    Cached after first load — the CSV is ~3k rows, trivially small.
    """
    global _mandi_locs
    if _mandi_locs is not None:
        return _mandi_locs

    csv_path = (
        # __file__ = src/serving/location_resolver.py
        # parents[2] = project root
        __import__("pathlib").Path(__file__).resolve().parents[2]
        / "data"
        / "reference"
        / "mandi_locations.csv"
    )

    if not csv_path.exists():
        raise FileNotFoundError(
            f"Mandi locations CSV not found at {csv_path}. "
            "Run: python -m src.reference.build_mandi_locations"
        )

    _mandi_locs = pd.read_csv(csv_path)
    # Normalize string columns for case-insensitive matching
    for col in ("mandi_name", "district", "state"):
        if col in _mandi_locs.columns:
            _mandi_locs[col] = _mandi_locs[col].astype(str).str.strip()

    log.info("Loaded mandi locations: %d rows", len(_mandi_locs))
    return _mandi_locs


# ---------------------------------------------------------------------------
# Haversine distance
# ---------------------------------------------------------------------------

def haversine_km(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    """Great-circle distance between two points in kilometres."""
    lat1, lon1, lat2, lon2 = map(radians, [lat1, lon1, lat2, lon2])
    dlat = lat2 - lat1
    dlon = lon2 - lon1
    a = sin(dlat / 2) ** 2 + cos(lat1) * cos(lat2) * sin(dlon / 2) ** 2
    return 2 * _EARTH_RADIUS_KM * atan2(sqrt(a), sqrt(1 - a))


# ---------------------------------------------------------------------------
# Core resolver
# ---------------------------------------------------------------------------

def resolve_mandi(
    state: str,
    district: str | None = None,
    market: str | None = None,
) -> tuple[str | None, str]:
    """Resolve user input to a mandi_id.

    Parameters
    ----------
    state : str
        State name (case-insensitive, e.g. "MAHARASHTRA" or "Maharashtra").
    district : str, optional
        District name (case-insensitive).
    market : str, optional
        Market / mandi name (case-insensitive).

    Returns
    -------
    mandi_id : str or None
        The resolved mandi_id, or None if no match found.
    resolution_type : str
        How the mandi was resolved:
        - "exact_match"           : market name + state matched exactly
        - "district_fallback"     : matched on district + state
        - "state_fallback"        : matched on state only (first mandi in state)
        - "not_found"             : no match at all
    """
    df = _load_mandi_locs()
    state_upper = state.strip().upper()

    # ── 1. Exact match on market name + state ────────────────────────────
    if market:
        market_upper = market.strip().upper().replace("-", " ")
        candidates = df[
            (df["state"].str.upper() == state_upper)
            & (df["mandi_name"].str.upper() == market_upper)
        ]
        if not candidates.empty:
            return candidates.iloc[0]["mandi_id"], "exact_match"

    # ── 2. District + state fallback ─────────────────────────────────────
    if district:
        district_upper = district.strip().upper().replace("-", " ")
        candidates = df[
            (df["state"].str.upper() == state_upper)
            & (df["district"].str.upper() == district_upper)
        ]
        if not candidates.empty:
            return candidates.iloc[0]["mandi_id"], "district_fallback"

    # ── 3. State-only fallback ───────────────────────────────────────────
    state_candidates = df[df["state"].str.upper() == state_upper]
    if not state_candidates.empty:
        return state_candidates.iloc[0]["mandi_id"], "state_fallback"

    # ── 4. Not found ─────────────────────────────────────────────────────
    return None, "not_found"


# ---------------------------------------------------------------------------
# Geo fallback
# ---------------------------------------------------------------------------

def nearest_mandi_with_data(
    lat: float,
    lon: float,
    available_mandi_ids: list[str] | set[str],
    top_n: int = 1,
) -> pd.DataFrame:
    """Find the nearest mandi(s) with data by haversine distance.

    Parameters
    ----------
    lat, lon : float
        Reference point (e.g. the user's location or the unmatched mandi).
    available_mandi_ids : list or set
        Mandi IDs that have data in the serving snapshot.
    top_n : int
        Number of nearest mandis to return.

    Returns
    -------
    pd.DataFrame
        Top-N mandis sorted by distance, with an added 'distance_km' column.
    """
    df = _load_mandi_locs()
    subset = df[df["mandi_id"].isin(available_mandi_ids)].copy()

    if subset.empty:
        return pd.DataFrame()

    subset["distance_km"] = subset.apply(
        lambda row: haversine_km(lat, lon, row["latitude"], row["longitude"]),
        axis=1,
    )
    return subset.sort_values("distance_km").head(top_n)


# ---------------------------------------------------------------------------
# CLI convenience
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s  %(levelname)-8s  %(message)s",
        datefmt="%H:%M:%S",
    )

    # Quick smoke test
    test_cases = [
        ("Maharashtra", "Pune", "Pune"),
        ("KARNATAKA", "Bangalore", None),
        ("MAHARASHTRA", None, "NONEXISTENT_MARKET"),
        ("INVALID_STATE", None, None),
    ]

    print("=" * 70)
    print("Location Resolver — Smoke Test")
    print("=" * 70)

    for state, district, market in test_cases:
        mandi_id, resolution = resolve_mandi(state, district, market)
        print(f"\n  Input: state={state!r}, district={district!r}, market={market!r}")
        print(f"  → mandi_id={mandi_id}, resolution={resolution}")
