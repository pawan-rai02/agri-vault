"""
AgriVault – WSGI Entry Point
==============================
Production entry point for Gunicorn (or any WSGI server).

Usage
-----
    # Production (Gunicorn)
    gunicorn wsgi:app --bind 0.0.0.0:8000 --workers 4 --timeout 120

    # Development
    python wsgi.py
"""

from __future__ import annotations

import logging
import os

from dotenv import load_dotenv

# Load .env before importing the app (env vars must be set first)
load_dotenv()

from src.api.app import app  # noqa: E402


def _configure_logging() -> None:
    """Configure structured logging for production."""
    level = os.environ.get("AGRIVAULT_LOG_LEVEL", "INFO").upper()
    log_format = (
        "%(asctime)s  %(levelname)-8s  [%(name)s]  %(message)s"
    )
    logging.basicConfig(
        level=getattr(logging, level, logging.INFO),
        format=log_format,
        datefmt="%Y-%m-%d %H:%M:%S",
    )
    # Quiet noisy libraries
    logging.getLogger("boto3").setLevel(logging.WARNING)
    logging.getLogger("botocore").setLevel(logging.WARNING)
    logging.getLogger("urllib3").setLevel(logging.WARNING)


_configure_logging()


if __name__ == "__main__":
    host = os.environ.get("AGRIVAULT_HOST", "127.0.0.1")
    port = int(os.environ.get("AGRIVAULT_PORT", "5000"))
    debug = os.environ.get("AGRIVAULT_DEBUG", "false").lower() == "true"

    app.run(host=host, port=port, debug=debug)
