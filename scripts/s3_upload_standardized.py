"""
AgriVault – Upload Standardized Data to S3
============================================
After PySpark cleaning, uploads everything under data/standardized/
to s3://agrivault-lake-pawan/standardized/<rel_path>

Same idempotency logic as s3_upload_raw.py.

Usage
-----
    python scripts/s3_upload_standardized.py
    python scripts/s3_upload_standardized.py --force
    python scripts/s3_upload_standardized.py --dry-run
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

import boto3
import yaml
from botocore.exceptions import ClientError
from tqdm import tqdm

PROJECT_ROOT = Path(__file__).resolve().parents[1]
CONFIG_PATH = PROJECT_ROOT / "configs" / "aws_config.yaml"
STANDARDIZED_ROOT = PROJECT_ROOT / "data" / "standardized"

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger(__name__)


def load_config() -> dict:
    with open(CONFIG_PATH) as f:
        return yaml.safe_load(f)


def make_s3_client(cfg: dict):
    session = boto3.Session(
        profile_name=cfg["aws"].get("profile"),
        region_name=cfg["aws"].get("region", "ap-south-1"),
    )
    return session.client("s3"), cfg["s3"]["bucket"]


def remote_size(s3, bucket: str, key: str) -> int | None:
    try:
        resp = s3.head_object(Bucket=bucket, Key=key)
        return resp["ContentLength"]
    except ClientError as e:
        if e.response["Error"]["Code"] in ("404", "NoSuchKey"):
            return None
        raise


def upload_file(s3, bucket, local_path, s3_key, force=False, dry_run=False) -> str:
    if dry_run:
        log.info("[DRY-RUN] %s → s3://%s/%s", local_path.name, bucket, s3_key)
        return "dry_run"

    if not force:
        existing = remote_size(s3, bucket, s3_key)
        if existing is not None and existing == local_path.stat().st_size:
            return "skipped"

    s3.upload_file(str(local_path), bucket, s3_key)
    return "uploaded"


def main():
    parser = argparse.ArgumentParser(description="Upload standardized AgriVault data to S3")
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument(
        "--dataset",
        choices=["apmc", "weather", "wdra", "cpi", "wpi", "ndvi", "loans"],
        help="Upload only one dataset (default: all)",
    )
    args = parser.parse_args()

    cfg = load_config()
    s3, bucket = make_s3_client(cfg)
    std_prefix = cfg["s3"]["prefixes"].get("standardized", "standardized").rstrip("/")

    root = STANDARDIZED_ROOT
    if args.dataset:
        root = STANDARDIZED_ROOT / args.dataset
        if not root.exists():
            log.error("Dataset directory not found: %s", root)
            sys.exit(1)

    all_files = [(p, f"{std_prefix}/{p.relative_to(STANDARDIZED_ROOT).as_posix()}")
                 for p in sorted(root.rglob("*")) if p.is_file()]

    if not all_files:
        log.warning("No files found under %s", root)
        sys.exit(0)

    log.info("Bucket : s3://%s/%s/", bucket, std_prefix)
    log.info("Files  : %d", len(all_files))

    counts = {"uploaded": 0, "skipped": 0, "dry_run": 0, "error": 0}

    with tqdm(all_files, unit="file", desc="Uploading standardized") as bar:
        for local_path, s3_key in bar:
            bar.set_postfix_str(local_path.name[:40])
            try:
                result = upload_file(s3, bucket, local_path, s3_key,
                                     force=args.force, dry_run=args.dry_run)
                counts[result] += 1
            except Exception as exc:
                log.error("FAILED %s: %s", s3_key, exc)
                counts["error"] += 1

    log.info("Uploaded=%d  Skipped=%d  Errors=%d",
             counts["uploaded"], counts["skipped"], counts["error"])
    if counts["error"] > 0:
        sys.exit(1)


if __name__ == "__main__":
    main()
