"""
AgriVault – S3 Storage Client
==============================
Wraps boto3 for all S3 read/write operations used by the pipeline.
Credentials are loaded from the named AWS CLI profile defined in
configs/aws_config.yaml (never hard-coded here).

Usage
-----
    from src.storage.s3_client import S3Client

    s3 = S3Client()                        # uses config file
    s3.upload("data/raw/apmc/foo.csv", "raw/apmc/foo.csv")
    df = s3.read_csv("raw/apmc/foo.csv")
    s3.download("raw/apmc/foo.csv", "data/raw/apmc/foo.csv")
    keys = s3.list_keys("raw/apmc/")

    # Read all parquet files under a prefix (with Hive partition extraction)
    df = s3.read_parquet_s3("standardized/apmc/")

    # Write a DataFrame as parquet
    s3.write_parquet_s3(df, "features/price_features/state=MAHARASHTRA/data.parquet")
"""

from __future__ import annotations

import io
import logging
import re
from pathlib import Path
from typing import Iterator

import boto3
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
import yaml
from botocore.exceptions import ClientError

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Config loading
# ---------------------------------------------------------------------------

# Locate config relative to this file (src/storage/ → project root/configs/)
_CONFIG_PATH = Path(__file__).resolve().parents[2] / "configs" / "aws_config.yaml"


def load_config(path: Path | None = None) -> dict:
    """Load the AWS/S3 configuration from yaml."""
    with open(path or _CONFIG_PATH) as f:
        return yaml.safe_load(f)


class S3Client:
    """Thin, opinionated wrapper around boto3 S3 for AgriVault."""

    def __init__(self, config_path: Path | None = None):
        cfg = load_config(config_path or _CONFIG_PATH)
        aws_cfg = cfg["aws"]
        s3_cfg = cfg["s3"]

        session = boto3.Session(
            profile_name=aws_cfg.get("profile"),
            region_name=aws_cfg.get("region", "ap-south-1"),
        )
        self._s3 = session.client("s3")
        self.bucket: str = s3_cfg["bucket"]
        self.prefixes: dict[str, str] = s3_cfg.get("prefixes", {})
        self.sse: str | None = s3_cfg.get("sse")

        logger.info(
            "S3Client initialised | bucket=%s profile=%s region=%s",
            self.bucket,
            aws_cfg.get("profile"),
            aws_cfg.get("region"),
        )

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _put_kwargs(self) -> dict:
        """Extra kwargs for put_object / upload_file when SSE is set."""
        return {"ServerSideEncryption": self.sse} if self.sse else {}

    def key(self, layer: str, relative_path: str) -> str:
        """Build a full S3 key: layer prefix + relative path.

        Example:
            s3.key("raw", "apmc/2024-01.csv")
            # → "raw/apmc/2024-01.csv"
        """
        prefix = self.prefixes.get(layer, f"{layer}/").rstrip("/")
        return f"{prefix}/{relative_path}"

    # ------------------------------------------------------------------
    # Core operations
    # ------------------------------------------------------------------

    def upload(self, local_path: str | Path, s3_key: str) -> None:
        """Upload a local file to S3."""
        local_path = Path(local_path)
        logger.info("Uploading %s → s3://%s/%s", local_path, self.bucket, s3_key)
        extra = self._put_kwargs()
        self._s3.upload_file(
            str(local_path),
            self.bucket,
            s3_key,
            ExtraArgs=extra if extra else None,
        )

    def download(self, s3_key: str, local_path: str | Path) -> None:
        """Download an S3 object to a local file."""
        local_path = Path(local_path)
        local_path.parent.mkdir(parents=True, exist_ok=True)
        logger.info("Downloading s3://%s/%s → %s", self.bucket, s3_key, local_path)
        self._s3.download_file(self.bucket, s3_key, str(local_path))

    def read_bytes(self, s3_key: str) -> bytes:
        """Read an S3 object into memory as bytes."""
        resp = self._s3.get_object(Bucket=self.bucket, Key=s3_key)
        return resp["Body"].read()

    def write_bytes(self, data: bytes, s3_key: str) -> None:
        """Write bytes directly to an S3 object."""
        kwargs = {"Bucket": self.bucket, "Key": s3_key, "Body": data}
        kwargs.update(self._put_kwargs())
        self._s3.put_object(**kwargs)

    # ------------------------------------------------------------------
    # Pandas helpers
    # ------------------------------------------------------------------

    def read_csv(self, s3_key: str, **kwargs) -> pd.DataFrame:
        """Read a CSV from S3 directly into a DataFrame."""
        data = self.read_bytes(s3_key)
        return pd.read_csv(io.BytesIO(data), **kwargs)

    def write_csv(self, df: pd.DataFrame, s3_key: str, index: bool = False) -> None:
        """Write a DataFrame as CSV to S3."""
        buf = io.BytesIO()
        df.to_csv(buf, index=index)
        self.write_bytes(buf.getvalue(), s3_key)

    def read_parquet(self, s3_key: str, **kwargs) -> pd.DataFrame:
        """Read a single Parquet file from S3 into a DataFrame."""
        data = self.read_bytes(s3_key)
        return pd.read_parquet(io.BytesIO(data), **kwargs)

    def write_parquet(self, df: pd.DataFrame, s3_key: str) -> None:
        """Write a DataFrame as Parquet to S3."""
        buf = io.BytesIO()
        df.to_parquet(buf, index=False)
        self.write_bytes(buf.getvalue(), s3_key)

    # ------------------------------------------------------------------
    # Parquet directory reading (multi-file)
    # ------------------------------------------------------------------

    def list_parquet_keys(self, prefix: str) -> list[str]:
        """List all .parquet object keys under a prefix.

        Uses paginator to handle >1000 objects.
        """
        keys: list[str] = []
        paginator = self._s3.get_paginator("list_objects_v2")
        for page in paginator.paginate(Bucket=self.bucket, Prefix=prefix):
            for obj in page.get("Contents", []):
                if obj["Key"].endswith(".parquet"):
                    keys.append(obj["Key"])
        return keys

    def read_parquet_s3(
        self,
        prefix: str,
        columns: list[str] | None = None,
        extract_hive_partitions: bool = True,
    ) -> pd.DataFrame:
        """Read all parquet files under a prefix into one DataFrame.

        Parameters
        ----------
        prefix : str
            S3 prefix (e.g. "standardized/apmc/").
        columns : list[str] | None
            Optional list of column names to read (pushed down to PyArrow).
        extract_hive_partitions : bool
            If True, extract Hive-style partition values from S3 key paths
            (e.g. ``state=MAHARASHTRA/``) and add them as columns.

        Returns
        -------
        pd.DataFrame
            Concatenated DataFrame from all parquet files under the prefix.
        """
        keys = self.list_parquet_keys(prefix)
        if not keys:
            raise FileNotFoundError(f"No parquet at s3://{self.bucket}/{prefix}")

        logger.info(
            "Reading %d parquet files from s3://%s/%s",
            len(keys),
            self.bucket,
            prefix,
        )

        frames: list[pd.DataFrame] = []
        for key in keys:
            buf = io.BytesIO()
            self._s3.download_fileobj(self.bucket, key, buf)
            buf.seek(0)
            table = pq.read_table(buf, columns=columns)
            part_df = table.to_pandas()

            if extract_hive_partitions:
                for match in re.finditer(r"([^/=]+)=([^/]+)/", key):
                    col_name, col_val = match.group(1), match.group(2)
                    if col_name not in part_df.columns:
                        part_df[col_name] = col_val

            frames.append(part_df)

        df = pd.concat(frames, ignore_index=True)
        logger.info("  → %d rows, %d cols", len(df), len(df.columns))
        return df

    def write_parquet_s3(self, df: pd.DataFrame, s3_key: str) -> None:
        """Write a DataFrame as Snappy-compressed Parquet to S3."""
        table = pa.Table.from_pandas(df, preserve_index=False)
        buf = io.BytesIO()
        pq.write_table(table, buf, compression="snappy")
        buf.seek(0)
        kwargs = {"Bucket": self.bucket, "Key": s3_key, "Body": buf.getvalue()}
        kwargs.update(self._put_kwargs())
        self._s3.put_object(**kwargs)
        logger.info("Written %d rows → s3://%s/%s", len(df), self.bucket, s3_key)

    # ------------------------------------------------------------------
    # Listing
    # ------------------------------------------------------------------

    def list_keys(self, prefix: str = "") -> list[str]:
        """Return all object keys under a given prefix."""
        keys: list[str] = []
        paginator = self._s3.get_paginator("list_objects_v2")
        for page in paginator.paginate(Bucket=self.bucket, Prefix=prefix):
            for obj in page.get("Contents", []):
                keys.append(obj["Key"])
        return keys

    def exists(self, s3_key: str) -> bool:
        """Return True if an S3 object exists."""
        try:
            self._s3.head_object(Bucket=self.bucket, Key=s3_key)
            return True
        except ClientError as e:
            if e.response["Error"]["Code"] == "404":
                return False
            raise

    def delete(self, s3_key: str) -> None:
        """Delete an S3 object."""
        logger.warning("Deleting s3://%s/%s", self.bucket, s3_key)
        self._s3.delete_object(Bucket=self.bucket, Key=s3_key)
