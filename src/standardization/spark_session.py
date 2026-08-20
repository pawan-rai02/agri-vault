"""
AgriVault – Central PySpark Session Factory
=============================================
All standardization and feature scripts import get_spark() from here.

S3A configuration uses the AWS named profile set in configs/aws_config.yaml
(read from ~/.aws/credentials — no credentials hard-coded here).

The hadoop-aws + aws-java-sdk-bundle JARs are downloaded automatically by
Maven on first run (requires internet access on first use, cached after).
"""

from __future__ import annotations

import os
from pathlib import Path

import yaml
from pyspark.sql import SparkSession

_CONFIG_PATH = Path(__file__).resolve().parents[2] / "configs" / "aws_config.yaml"


def _load_aws_config() -> dict:
    with open(_CONFIG_PATH) as f:
        return yaml.safe_load(f)


def get_spark(app_name: str = "agrivault", local_cores: int = 4) -> SparkSession:
    """
    Build and return a SparkSession configured for S3A access.

    Parameters
    ----------
    app_name : str
        Name shown in the Spark UI.
    local_cores : int
        Number of local threads (local[N] master).

    Notes
    -----
    - Credentials come from ~/.aws/credentials via DefaultAWSCredentialsProviderChain.
    - The profile is set via AWS_PROFILE env var so Spark's Java SDK picks it up.
    - hadoop-aws 3.3.4 works with Spark 3.5.x.
    """
    cfg = _load_aws_config()
    profile = cfg["aws"].get("profile")
    region = cfg["aws"].get("region", "ap-south-1")

    # ── Windows: set HADOOP_HOME so PySpark finds winutils.exe ────────────
    import platform
    if platform.system() == "Windows":
        hadoop_home = os.environ.get("HADOOP_HOME", r"C:\hadoop")
        os.environ["HADOOP_HOME"] = hadoop_home
        os.environ["hadoop.home.dir"] = hadoop_home

    # Expose profile to the underlying AWS Java SDK
    if profile:
        os.environ["AWS_PROFILE"] = profile

    spark = (
        SparkSession.builder
        .appName(app_name)
        .master(f"local[{local_cores}]")
        # -----------------------------------------------------------
        # Hadoop-AWS + AWS Java SDK (auto-downloaded by Maven resolver)
        # Spark 4.x ships with Hadoop 3.4.x
        # -----------------------------------------------------------
        .config(
            "spark.jars.packages",
            "org.apache.hadoop:hadoop-aws:3.4.1,"
            "com.amazonaws:aws-java-sdk-bundle:1.12.780",
        )
        # -----------------------------------------------------------
        # S3A filesystem settings
        # -----------------------------------------------------------
        .config(
            "spark.hadoop.fs.s3a.impl",
            "org.apache.hadoop.fs.s3a.S3AFileSystem",
        )
        .config(
            "spark.hadoop.fs.s3a.aws.credentials.provider",
            "com.amazonaws.auth.profile.ProfileCredentialsProvider",
        )
        .config("spark.hadoop.fs.s3a.endpoint", f"s3.{region}.amazonaws.com")
        .config("spark.hadoop.fs.s3a.path.style.access", "false")
        # ── Use in-memory buffer for S3A writes (avoids Windows NativeIO
        #    DiskBlockFactory error caused by hadoop.dll version mismatch) ──
        .config("spark.hadoop.fs.s3a.fast.upload", "true")
        .config("spark.hadoop.fs.s3a.fast.upload.buffer", "bytebuffer")
        .config("spark.hadoop.fs.s3a.multipart.size", "67108864")    # 64 MB chunks
        .config("spark.hadoop.fs.s3a.multipart.threshold", "67108864")
        # -----------------------------------------------------------
        # Performance tuning for local mode
        # -----------------------------------------------------------
        .config("spark.driver.memory", "4g")
        .config("spark.sql.shuffle.partitions", "8")
        .config("spark.sql.parquet.compression.codec", "snappy")
        # ── Spark 4.0 changed default ANSI mode to ON, which makes
        # to_date() throw on bad values instead of returning NULL.
        # We handle nulls explicitly in cleaners, so disable ANSI.
        .config("spark.sql.ansi.enabled", "false")
        .config("spark.sql.ansi.enforceReservedKeywords", "false")
        # Suppress verbose INFO logs from AWS SDK
        .config("spark.driver.extraJavaOptions",
                "-Dlog4j.logger.com.amazonaws=WARN "
                "-Dlog4j.logger.org.apache.hadoop.fs.s3a=WARN")
        .getOrCreate()
    )

    spark.sparkContext.setLogLevel("WARN")
    return spark


def bucket_uri(cfg: dict, layer: str, path: str = "") -> str:
    """
    Build a full s3a:// URI from config.

    Example
    -------
        bucket_uri(cfg, "raw", "apmc/apmc_market_prices.csv")
        # → "s3a://agrivault-lake-pawan/raw/apmc/apmc_market_prices.csv"
    """
    bucket = cfg["s3"]["bucket"]
    prefix = cfg["s3"]["prefixes"].get(layer, f"{layer}/").rstrip("/")
    path = path.lstrip("/")
    return f"s3a://{bucket}/{prefix}/{path}" if path else f"s3a://{bucket}/{prefix}/"


def load_config() -> dict:
    with open(_CONFIG_PATH) as f:
        return yaml.safe_load(f)
