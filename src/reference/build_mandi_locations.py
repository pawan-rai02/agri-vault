import pandas as pd
import re
from pathlib import Path


# ============================================================
# Configuration
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parents[2]

INPUT_FILE = (
    PROJECT_ROOT
    / "data"
    / "raw"
    / "apmc"
    / "apmc_market_prices.csv"
)

OUTPUT_FILE = (
    PROJECT_ROOT
    / "data"
    / "reference"
    / "mandi_locations.csv"
)

CHUNK_SIZE = 200_000

COLUMNS = [
    "market_code",
    "market_center",
    "district_name",
    "state_name",
    "latitude",
    "longitude",
]


# ============================================================
# Helpers
# ============================================================

def clean_identifier(value):
    """
    Convert a text value into a stable identifier component.

    Example:
        'Dharashiv' -> 'DHARASHIV'
        'Osmanabad District' -> 'OSMANABAD_DISTRICT'
    """
    value = str(value).strip().upper()
    value = re.sub(r"[^A-Z0-9]+", "_", value)
    return value.strip("_")


# ============================================================
# Main pipeline
# ============================================================

def build_mandi_locations():
    print("=" * 60)
    print("AgriVault - Building Mandi Location Master")
    print("=" * 60)

    if not INPUT_FILE.exists():
        raise FileNotFoundError(
            f"APMC input file not found:\n{INPUT_FILE}"
        )

    print(f"\nInput : {INPUT_FILE}")
    print(f"Output: {OUTPUT_FILE}")

    # --------------------------------------------------------
    # Read APMC data in chunks
    # --------------------------------------------------------

    unique_parts = []

    print("\nReading APMC data in chunks...")

    for chunk_number, chunk in enumerate(
        pd.read_csv(
            INPUT_FILE,
            usecols=COLUMNS,
            chunksize=CHUNK_SIZE,
        ),
        start=1,
    ):
        chunk = chunk.drop_duplicates()

        unique_parts.append(chunk)

        print(
            f"  Processed chunk {chunk_number}: "
            f"{len(chunk):,} unique rows"
        )

    # --------------------------------------------------------
    # Combine chunk results
    # --------------------------------------------------------

    df = pd.concat(
        unique_parts,
        ignore_index=True,
    )

    print(
        f"\nRows after chunk-level deduplication: "
        f"{len(df):,}"
    )

    # --------------------------------------------------------
    # Check metadata consistency
    #
    # A market_code may theoretically be reused for multiple
    # physical mandis. Therefore market_code alone is NOT used
    # as the mandi identifier.
    # --------------------------------------------------------

    print("\nChecking market-code consistency...")

    metadata_columns = [
        "market_center",
        "district_name",
        "state_name",
        "latitude",
        "longitude",
    ]

    inconsistent_codes = []

    grouped = df.groupby("market_code", dropna=False)

    for market_code, group in grouped:
        if any(
            group[column].nunique(dropna=False) > 1
            for column in metadata_columns
        ):
            inconsistent_codes.append(market_code)

    print(
        f"Unique market codes: "
        f"{df['market_code'].nunique():,}"
    )

    print(
        f"Market codes with inconsistent metadata: "
        f"{len(inconsistent_codes):,}"
    )

    if inconsistent_codes:
        print("\nThese market codes map to multiple physical locations:")

        for code in inconsistent_codes[:20]:
            print(f"\nMarket code: {code}")

            print(
                df[df["market_code"] == code][
                    [
                        "market_code",
                        "market_center",
                        "district_name",
                        "state_name",
                        "latitude",
                        "longitude",
                    ]
                ].to_string(index=False)
            )

        if len(inconsistent_codes) > 20:
            print(
                f"\n...and "
                f"{len(inconsistent_codes) - 20} more."
            )

    # --------------------------------------------------------
    # One row per physical mandi/location
    # --------------------------------------------------------

    location_columns = [
        "market_code",
        "market_center",
        "district_name",
        "state_name",
        "latitude",
        "longitude",
    ]

    df = df.drop_duplicates(
        subset=location_columns
    ).copy()

    print(
        f"\nUnique physical mandi locations: "
        f"{len(df):,}"
    )

    # --------------------------------------------------------
    # Validate coordinates
    # --------------------------------------------------------

    missing_latitude = df["latitude"].isna().sum()
    missing_longitude = df["longitude"].isna().sum()

    print(
        f"Missing latitude : {missing_latitude:,}"
    )

    print(
        f"Missing longitude: {missing_longitude:,}"
    )

    if missing_latitude > 0 or missing_longitude > 0:
        raise ValueError(
            "Some mandi locations are missing coordinates."
        )

    invalid_coordinates = df[
        (df["latitude"] < 6)
        | (df["latitude"] > 38)
        | (df["longitude"] < 68)
        | (df["longitude"] > 98)
    ]

    print(
        f"Suspicious coordinates: "
        f"{len(invalid_coordinates):,}"
    )

    if not invalid_coordinates.empty:
        print(
            invalid_coordinates.to_string(index=False)
        )

        raise ValueError(
            "Invalid/suspicious coordinates detected."
        )

    # --------------------------------------------------------
    # Build stable mandi ID
    #
    # market_code is NOT sufficient because the source data
    # contains at least one reused code (e.g. 1765).
    # --------------------------------------------------------

    df["mandi_id"] = (
        df["state_name"].map(clean_identifier)
        + "_"
        + df["district_name"].map(clean_identifier)
        + "_"
        + df["market_code"].astype(str)
        + "_"
        + df["market_center"].map(clean_identifier)
    )

    # --------------------------------------------------------
    # Rename columns
    # --------------------------------------------------------

    df = df.rename(
        columns={
            "market_center": "mandi_name",
            "district_name": "district",
            "state_name": "state",
        }
    )

    # --------------------------------------------------------
    # Select final schema
    # --------------------------------------------------------

    df = df[
        [
            "mandi_id",
            "market_code",
            "mandi_name",
            "district",
            "state",
            "latitude",
            "longitude",
        ]
    ]

    # --------------------------------------------------------
    # Final validations
    # --------------------------------------------------------

    if df["mandi_id"].duplicated().any():
        duplicates = df[
            df["mandi_id"].duplicated(keep=False)
        ]

        print("\nDuplicate mandi IDs detected:")
        print(duplicates.to_string(index=False))

        raise ValueError(
            "mandi_id is not unique."
        )

    if df["mandi_id"].isna().any():
        raise ValueError(
            "Some mandi IDs are missing."
        )

    # --------------------------------------------------------
    # Sort and write
    # --------------------------------------------------------

    df = df.sort_values(
        by=["state", "district", "mandi_name"]
    ).reset_index(drop=True)

    OUTPUT_FILE.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    df.to_csv(
        OUTPUT_FILE,
        index=False,
    )

    # --------------------------------------------------------
    # Summary
    # --------------------------------------------------------

    print("\n" + "=" * 60)
    print("Mandi master created successfully")
    print("=" * 60)

    print(f"\nOutput file:")
    print(OUTPUT_FILE)

    print(
        f"\nTotal physical mandis: "
        f"{len(df):,}"
    )

    print(
        f"Unique market codes: "
        f"{df['market_code'].nunique():,}"
    )

    print(
        f"States: "
        f"{df['state'].nunique():,}"
    )

    print(
        f"Districts: "
        f"{df['district'].nunique():,}"
    )

    print("\nFinal schema:")
    print(df.dtypes)

    print("\nSample:")
    print(
        df.head(10).to_string(index=False)
    )

    print("\nDone.")


if __name__ == "__main__":
    build_mandi_locations()
