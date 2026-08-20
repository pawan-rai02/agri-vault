"""
AgriVault – Generate Synthetic Agri-Credit Risk Proxy
======================================================
Generates a synthetic loan risk dataset based on NABARD/RBI published
statistics for Kisan Credit Card (KCC) and FPO lending.

This unblocks Phase 1 while the Lending Club Kaggle download is set up.
Once real Lending Club data is available, run clean_loans.py instead.

The synthetic data faithfully reflects:
  - State-level KCC disbursement volumes (from RBI DBIE data)
  - Commodity-category-level default rates (from NABARD ARs 2022-24)
  - Interest rate bands by loan size (KCC scheme rates)
  - Seasonal loan disbursement patterns (Kharif/Rabi cycles)

Outputs:
  data/raw/loans/loan_risk_proxy.csv  (raw, as-if downloaded)

Run
---
    python scripts/generate_loan_proxy.py
"""

from __future__ import annotations

import logging
import random
from pathlib import Path

import numpy as np
import pandas as pd

log = logging.getLogger(__name__)

PROJECT_ROOT = Path(__file__).resolve().parents[1]
OUTPUT = PROJECT_ROOT / "data" / "raw" / "loans" / "loan_risk_proxy.csv"

SEED = 42
rng = np.random.default_rng(SEED)

# ---------------------------------------------------------------------------
# State-level KCC metadata (approximate, from RBI DBIE 2023-24)
# Weights roughly reflect outstanding KCC accounts per state
# ---------------------------------------------------------------------------
STATE_WEIGHTS = {
    "Uttar Pradesh":     0.18,
    "Rajasthan":         0.12,
    "Madhya Pradesh":    0.10,
    "Maharashtra":       0.10,
    "Andhra Pradesh":    0.08,
    "Karnataka":         0.07,
    "Gujarat":           0.06,
    "Punjab":            0.06,
    "Tamil Nadu":        0.05,
    "Haryana":           0.05,
    "West Bengal":       0.04,
    "Bihar":             0.04,
    "Odisha":            0.02,
    "Telangana":         0.02,
    "Chhattisgarh":      0.01,
}

# Commodity categories relevant to AgriVault APMC data
COMMODITY_CATEGORIES = [
    "Cereals",
    "Pulses",
    "Oilseeds",
    "Vegetables",
    "Fruits",
    "Spices",
    "Cash Crops",
    "Livestock",
]

# Default rate by commodity category (approximate, from NABARD AR 2023-24)
CATEGORY_DEFAULT_RATE = {
    "Cereals":      0.04,
    "Pulses":       0.07,
    "Oilseeds":     0.06,
    "Vegetables":   0.09,
    "Fruits":       0.08,
    "Spices":       0.05,
    "Cash Crops":   0.06,
    "Livestock":    0.05,
}

# KCC interest rate bands (post-interest-subvention, net to farmer)
# < 3 lakh → 4% (after 3% subvention on 7% base)
# 3-10 lakh → 7-9%
# > 10 lakh → 9-12%
LOAN_TIERS = [
    (50_000,   3_00_000,  0.04, 0.06, "A"),   # small
    (3_00_001, 10_00_000, 0.07, 0.09, "B"),   # medium
    (10_00_001,50_00_000, 0.09, 0.12, "C"),   # large
]

SEASONS = ["Kharif", "Rabi", "Zaid"]
SEASON_MONTHS = {"Kharif": 6, "Rabi": 11, "Zaid": 3}


def generate_loans(n: int = 50_000) -> pd.DataFrame:
    states = list(STATE_WEIGHTS.keys())
    state_probs = np.array(list(STATE_WEIGHTS.values()))

    rows = []
    years = [2020, 2021, 2022, 2023, 2024]

    for _ in range(n):
        state = rng.choice(states, p=state_probs)
        commodity_cat = rng.choice(COMMODITY_CATEGORIES)
        year = rng.choice(years)
        season = rng.choice(SEASONS)

        # Loan tier
        tier_idx = rng.choice([0, 1, 2], p=[0.60, 0.30, 0.10])
        low, high, rate_low, rate_high, grade = LOAN_TIERS[tier_idx]
        loan_amount = int(rng.integers(low, high))
        interest_rate = round(rng.uniform(rate_low, rate_high) * 100, 2)

        # Default based on commodity category + add state noise
        base_default_rate = CATEGORY_DEFAULT_RATE[commodity_cat]
        # Higher loan → slightly higher default risk
        size_factor = 1.0 + (tier_idx * 0.02)
        p_default = min(base_default_rate * size_factor + rng.normal(0, 0.01), 0.30)
        p_default = max(p_default, 0.0)
        is_default = int(rng.random() < p_default)

        # LTV ratio (loan-to-value of collateral pledge)
        ltv = round(rng.uniform(0.50, 0.90) + (0.05 * is_default), 2)
        ltv = min(ltv, 1.0)

        # Loan duration (months)
        term_months = 12 if season in ("Kharif", "Rabi") else 9

        # Disbursement month
        base_month = SEASON_MONTHS[season]
        disburse_month = int(np.clip(base_month + rng.integers(-1, 2), 1, 12))
        disburse_date = pd.Timestamp(year=year, month=disburse_month, day=1)

        rows.append({
            "loan_id":            f"KCC_{year}_{rng.integers(1_000_000):07d}",
            "state":              state,
            "commodity_category": commodity_cat,
            "season":             season,
            "year":               year,
            "disburse_date":      disburse_date.strftime("%Y-%m-%d"),
            "loan_amount_inr":    loan_amount,
            "interest_rate_pct":  interest_rate,
            "term_months":        term_months,
            "credit_grade":       grade,
            "ltv_ratio":          ltv,
            "is_default":         is_default,
            "data_source":        "synthetic_kcc_proxy",
            "proxy_note": (
                "Synthetic data based on NABARD/RBI KCC statistics. "
                "Replace with real Lending Club data via clean_loans.py "
                "once Kaggle credentials are configured."
            ),
        })

    return pd.DataFrame(rows)


def main():
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s  %(levelname)-8s  %(message)s",
        datefmt="%H:%M:%S",
    )

    OUTPUT.parent.mkdir(parents=True, exist_ok=True)

    log.info("Generating synthetic KCC loan proxy (n=50,000)...")
    df = generate_loans(n=50_000)

    # Summary stats
    log.info("Overall default rate : %.2f%%", df["is_default"].mean() * 100)
    log.info("Loan amount range    : ₹%s – ₹%s",
             f"{df['loan_amount_inr'].min():,}", f"{df['loan_amount_inr'].max():,}")
    log.info("States               : %d", df["state"].nunique())
    log.info("Date range           : %s → %s",
             df["disburse_date"].min(), df["disburse_date"].max())

    df.to_csv(OUTPUT, index=False)
    log.info("✓ Written to %s", OUTPUT)
    log.info("")
    log.info("To use real Lending Club data instead:")
    log.info("  1. Place kaggle.json in C:/Users/LENOVO/.kaggle/kaggle.json")
    log.info("     (Download from kaggle.com → your profile → Settings → API)")
    log.info("  2. Run: kaggle datasets download -d wendykan/lending-club-loan-data \\")
    log.info("              -p data/raw/loans --unzip")
    log.info("  3. Run: python -m src.standardization.clean_loans")


if __name__ == "__main__":
    main()
