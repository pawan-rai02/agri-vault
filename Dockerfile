# =============================================================================
# AgriVault – Dockerfile
# =============================================================================
# Multi-stage build for a lean production image.
#
# Base image includes Python 3.12 + Java 17 (required by PySpark).
# Maven JARs (hadoop-aws, aws-java-sdk-bundle) are downloaded on the
# first SparkSession creation and cached in the image layer.
#
# Build
# -----
#     docker build -t agrivault .
#
# Run (API / Dashboard)
# ---------------------
#     docker run --env-file .env -p 8000:8000 agrivault
#     docker run --env-file .env -p 8000:8000 agrivault pipeline clean-apmc
#
# Compose
# -------
#     docker compose up --build
# =============================================================================

# ── Stage 1: Base with system dependencies ──────────────────────────────────
FROM python:3.12-slim AS base

# Avoid interactive prompts during package installation
ENV DEBIAN_FRONTEND=noninteractive

# Install Java 17 (required by PySpark) + common utils
RUN apt-get update && apt-get install -y --no-install-recommends \
        default-jdk-headless \
        curl \
        procps \
    && rm -rf /var/lib/apt/lists/*

ENV JAVA_HOME=/usr/lib/jvm/java-17-openjdk-amd64
ENV PATH="${JAVA_HOME}/bin:${PATH}"

# Verify Java is available
RUN java -version

# ── Stage 2: Python dependencies ───────────────────────────────────────────
FROM base AS deps

WORKDIR /app

# Copy only requirements first (Docker layer caching — deps change rarely)
COPY requirements.txt .

RUN pip install --no-cache-dir --upgrade pip \
    && pip install --no-cache-dir -r requirements.txt

# ── Stage 3: Production image ──────────────────────────────────────────────
FROM deps AS production

WORKDIR /app

# Copy application code
COPY configs/    configs/
COPY src/        src/
COPY wsgi.py     .
COPY scripts/    scripts/

# Create non-root user for security
RUN groupadd -r agrivault && useradd -r -g agrivault agrivault \
    && chown -R agrivault:agrivault /app
USER agrivault

# Expose the Gunicorn port
EXPOSE 8000

# Default: run the API with Gunicorn
#   --bind 0.0.0.0:8000   listen on all interfaces
#   --workers 2            2 worker processes (adjust for CPU cores)
#   --timeout 120          allow slow S3 downloads
#   --access-logfile -     log access to stdout
CMD ["gunicorn", "wsgi:app", \
     "--bind", "0.0.0.0:8000", \
     "--workers", "2", \
     "--timeout", "120", \
     "--access-logfile", "-", \
     "--error-logfile", "-"]
