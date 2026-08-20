import argparse
from pathlib import Path

import ee
import pandas as pd
import yaml


PROJECT_ROOT = Path(__file__).resolve().parents[2]

CONFIG_FILE = PROJECT_ROOT / "configs" / "gee_config.yaml"
MANDI_FILE = PROJECT_ROOT / "data" / "reference" / "mandi_locations.csv"
OUTPUT_DIR = PROJECT_ROOT / "data" / "raw" / "ndvi"


def load_config():
    with open(CONFIG_FILE, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def initialize_gee(project_id):
    ee.Initialize(project=project_id)
    print(f"GEE initialized: {project_id}")


def mask_clouds(image):
    """
    Mask clouds and cloud shadows using Sentinel-2 SCL.

    SCL:
        3  = cloud shadow
        8  = medium probability cloud
        9  = high probability cloud
        10 = cirrus
        11 = snow/ice
    """

    scl = image.select("SCL")

    mask = (
        scl.neq(3)
        .And(scl.neq(8))
        .And(scl.neq(9))
        .And(scl.neq(10))
        .And(scl.neq(11))
    )

    return image.updateMask(mask)


def add_ndvi(image):
    image = mask_clouds(image)

    ndvi = image.normalizedDifference(
        ["B8", "B4"]
    ).rename("ndvi")

    return ndvi.copyProperties(
        image,
        ["system:time_start"]
    )


def build_mandi_features(mandi_df, buffer_meters):
    features = []

    for row in mandi_df.itertuples(index=False):

        point = ee.Geometry.Point(
            [
                float(row.longitude),
                float(row.latitude),
            ]
        )

        region = point.buffer(
            buffer_meters
        )

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


def process_batch(
    mandi_features,
    collection,
    start_date,
    end_date,
    cloud_percentage,
    scale_meters,
):
    """
    Build ONE monthly median NDVI composite and reduce it
    over a batch of mandi regions.
    """

    images = (
        collection
        .filterDate(
            start_date,
            end_date,
        )
        .filterBounds(
            mandi_features.geometry()
        )
        .filter(
            ee.Filter.lte(
                "CLOUDY_PIXEL_PERCENTAGE",
                cloud_percentage,
            )
        )
        .map(mask_clouds)
    )

    scene_count = images.size().getInfo()

    if scene_count == 0:
        return []

    # Monthly median composite.
    ndvi = (
        images
        .map(
            lambda image: image.normalizedDifference(
                ["B8", "B4"]
            ).rename("ndvi")
        )
        .median()
    )

    reduced = ndvi.reduceRegions(
        collection=mandi_features,
        reducer=ee.Reducer.mean(),
        scale=scale_meters,
        tileScale=4,
    )

    result = reduced.getInfo()

    rows = []

    observation_date = start_date

    for feature in result["features"]:

        properties = feature["properties"]

        value = properties.get("mean")

        if value is None:
            continue

        rows.append(
            {
                "mandi_id": properties.get("mandi_id"),
                "market_code": properties.get("market_code"),
                "mandi_name": properties.get("mandi_name"),
                "district": properties.get("district"),
                "state": properties.get("state"),
                "latitude": properties.get("latitude"),
                "longitude": properties.get("longitude"),
                "date": observation_date,
                "ndvi": value,
            }
        )

    return rows, scene_count


def main():

    parser = argparse.ArgumentParser(
        description="AgriVault monthly Sentinel-2 NDVI extraction"
    )

    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Process only first N mandis.",
    )

    parser.add_argument(
        "--batch-size",
        type=int,
        default=250,
        help="Number of mandis per Earth Engine request.",
    )

    args = parser.parse_args()

    config = load_config()

    project_id = config["project_id"]
    dataset = config["dataset"]

    start_date = pd.Timestamp(
        config["start_date"]
    )

    end_date = pd.Timestamp(
        config["end_date"]
    )

    cloud_percentage = config[
        "cloud_percentage"
    ]

    buffer_meters = config[
        "buffer_meters"
    ]

    scale_meters = config[
        "scale_meters"
    ]

    OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    if not MANDI_FILE.exists():
        raise FileNotFoundError(
            f"Mandi file not found:\n{MANDI_FILE}"
        )

    initialize_gee(
        project_id
    )

    mandis = pd.read_csv(
        MANDI_FILE
    )

    if args.limit is not None:
        mandis = mandis.head(
            args.limit
        )

    print()
    print("=" * 65)
    print(
        "AgriVault Sentinel-2 Monthly NDVI"
    )
    print("=" * 65)
    print(
        f"Mandis          : {len(mandis):,}"
    )
    print(
        f"Batch size      : {args.batch_size:,}"
    )
    print(
        f"Start date      : "
        f"{start_date.date()}"
    )
    print(
        f"End date        : "
        f"{end_date.date()}"
    )
    print(
        f"Cloud threshold : "
        f"{cloud_percentage}%"
    )
    print(
        f"Buffer          : "
        f"{buffer_meters} m"
    )
    print(
        f"Resolution      : "
        f"{scale_meters} m"
    )
    print(
        f"Dataset         : "
        f"{dataset}"
    )
    print("=" * 65)

    collection = ee.ImageCollection(
        dataset
    )

    all_results = []

    current = start_date

    while current < end_date:

        month_end = (
            current
            + pd.offsets.MonthBegin(1)
        )

        if month_end > end_date:
            month_end = end_date

        month_start_str = (
            current.strftime("%Y-%m-%d")
        )

        month_end_str = (
            month_end.strftime("%Y-%m-%d")
        )

        print()
        print(
            f"MONTH: {current.strftime('%Y-%m')}"
        )

        month_results = []

        for batch_start in range(
            0,
            len(mandis),
            args.batch_size,
        ):

            batch_end = min(
                batch_start
                + args.batch_size,
                len(mandis),
            )

            batch_df = mandis.iloc[
                batch_start:batch_end
            ]

            print(
                f"  Batch "
                f"{batch_start + 1}-"
                f"{batch_end} / "
                f"{len(mandis)}"
            )

            try:

                mandi_features = (
                    build_mandi_features(
                        batch_df,
                        buffer_meters,
                    )
                )

                result = process_batch(
                    mandi_features=mandi_features,
                    collection=collection,
                    start_date=month_start_str,
                    end_date=month_end_str,
                    cloud_percentage=cloud_percentage,
                    scale_meters=scale_meters,
                )

                rows, scene_count = result

                month_results.extend(
                    rows
                )

                print(
                    f"    Scenes: "
                    f"{scene_count}"
                    f" | Valid NDVI: "
                    f"{len(rows):,}"
                )

            except Exception as e:

                print(
                    f"    ERROR: "
                    f"{type(e).__name__}: {e}"
                )

        all_results.extend(
            month_results
        )

        print(
            f"  Month total: "
            f"{len(month_results):,}"
        )

        current = month_end

    if not all_results:
        raise RuntimeError(
            "No NDVI observations retrieved."
        )

    df = pd.DataFrame(
        all_results
    )

    df["date"] = pd.to_datetime(
        df["date"]
    )

    df = df.sort_values(
        [
            "mandi_id",
            "date",
        ]
    )

    output_file = (
        OUTPUT_DIR
        / "ndvi_sentinel2_2025.csv"
    )

    df.to_csv(
        output_file,
        index=False,
    )

    print()
    print("=" * 65)
    print(
        "NDVI EXTRACTION COMPLETE"
    )
    print("=" * 65)
    print(
        f"Output       : "
        f"{output_file}"
    )
    print(
        f"Mandis       : "
        f"{df['mandi_id'].nunique():,}"
    )
    print(
        f"Observations : "
        f"{len(df):,}"
    )
    print(
        f"Date range   : "
        f"{df['date'].min().date()} -> "
        f"{df['date'].max().date()}"
    )
    print(
        f"NDVI range   : "
        f"{df['ndvi'].min():.4f} -> "
        f"{df['ndvi'].max():.4f}"
    )
    print(
        f"Mean NDVI    : "
        f"{df['ndvi'].mean():.4f}"
    )
    print("=" * 65)


if __name__ == "__main__":
    main()
