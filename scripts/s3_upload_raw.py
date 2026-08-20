"""
AgriVault – Upload Raw Data to S3
==================================
Uploads everything under data/raw/ and data/reference/ to the S3 bucket.

    s3://agrivault-lake-pawan/raw/<rel_path>
    s3://agrivault-lake-pawan/reference/<rel_path>

Idempotent: skips files already present on S3 with the same size.
Credentials: uses the 'agrivault' AWS CLI profile (no keys in code).

Usage
-----
    python scripts/s3_upload_raw.py
    python scripts/s3_upload_raw.py --force          # re-upload even if exists
    python scripts/s3_upload_raw.py --dry-run        # print what would be uploaded
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

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
PROJECT_ROOT = Path(__file__).resolve().parents[1]
CONFIG_PATH = PROJECT_ROOT / "configs" / "aws_config.yaml"

RAW_ROOT = PROJECT_ROOT / "data" / "raw"
REFERENCE_ROOT = PROJECT_ROOT / "data" / "reference"

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

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
    """Return the size of an existing S3 object, or None if it doesn't exist."""
    try:
        resp = s3.head_object(Bucket=bucket, Key=key)
        return resp["ContentLength"]
    except ClientError as e:
        if e.response["Error"]["Code"] in ("404", "NoSuchKey"):
            return None
        raise


def upload_file(
    s3,
    bucket: str,
    local_path: Path,
    s3_key: str,
    force: bool = False,
    dry_run: bool = False,
) -> str:
    """
    Upload a single file to S3.

    Returns one of: 'uploaded', 'skipped', 'dry_run'
    """
    if dry_run:
        log.info("[DRY-RUN] Would upload %s → s3://%s/%s", local_path.name, bucket, s3_key)
        return "dry_run"

    local_size = local_path.stat().st_size

    if not force:
        existing = remote_size(s3, bucket, s3_key)
        if existing is not None and existing == local_size:
            log.debug("SKIP (same size) %s", s3_key)
            return "skipped"

    s3.upload_file(str(local_path), bucket, s3_key)
    return "uploaded"


def collect_files(root: Path, prefix: str) -> list[tuple[Path, str]]:
    """Return [(local_path, s3_key), ...] for every file under root."""
    pairs = []
    for path in sorted(root.rglob("*")):
        if path.is_file():
            rel = path.relative_to(root)
            key = f"{prefix}/{rel.as_posix()}"
            pairs.append((path, key))
    return pairs


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="Upload raw AgriVault data to S3")
    parser.add_argument("--force", action="store_true", help="Re-upload even if file already exists on S3")
    parser.add_argument("--dry-run", action="store_true", help="Print what would be uploaded, don't actually upload")
    args = parser.parse_args()

    cfg = load_config()
    s3, bucket = make_s3_client(cfg)
    prefixes = cfg["s3"]["prefixes"]

    raw_prefix = prefixes.get("raw", "raw").rstrip("/")
    ref_prefix = prefixes.get("reference", "reference").rstrip("/")

    log.info("Bucket : s3://%s", bucket)
    log.info("Profile: %s", cfg["aws"].get("profile"))
    log.info("Region : %s", cfg["aws"].get("region"))
    log.info("")

    # Collect all files from raw/ and reference/
    all_files: list[tuple[Path, str]] = []

    if RAW_ROOT.exists():
        all_files += collect_files(RAW_ROOT, raw_prefix)
    else:
        log.warning("data/raw/ not found — nothing to upload from raw layer")

    if REFERENCE_ROOT.exists():
        all_files += collect_files(REFERENCE_ROOT, ref_prefix)
    else:
        log.warning("data/reference/ not found — skipping reference layer")

    if not all_files:
        log.error("No files found to upload.")
        sys.exit(1)

    log.info("Files to process: %d", len(all_files))

    # Upload with progress bar
    counts = {"uploaded": 0, "skipped": 0, "dry_run": 0, "error": 0}

    with tqdm(all_files, unit="file", desc="Uploading") as bar:
        for local_path, s3_key in bar:
            bar.set_postfix_str(local_path.name[:40])
            try:
                result = upload_file(
                    s3, bucket, local_path, s3_key,
                    force=args.force,
                    dry_run=args.dry_run,
                )
                counts[result] += 1
            except Exception as exc:
                log.error("FAILED %s: %s", s3_key, exc)
                counts["error"] += 1

    log.info("")
    log.info("=" * 55)
    log.info("Upload summary")
    log.info("  Uploaded : %d", counts["uploaded"])
    log.info("  Skipped  : %d (already on S3, same size)", counts["skipped"])
    log.info("  Dry-run  : %d", counts["dry_run"])
    log.info("  Errors   : %d", counts["error"])
    log.info("=" * 55)

    if counts["error"] > 0:
        sys.exit(1)


if __name__ == "__main__":
    main()
