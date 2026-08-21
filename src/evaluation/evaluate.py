"""
AgriVault – Model Evaluation Report
====================================
Evaluates trained Quantile GBM models against a naive persistence baseline
and produces a comprehensive report with:

1. Pinball loss comparison (model vs naive) per commodity × horizon
2. RMSE / MAPE at median forecast
3. Prediction interval coverage (80% PI should capture ~80%)
4. Risk decision precision/recall (against synthetic default labels)

The naive baseline is a "persistence forecast" — the last known price
at the time of prediction is used as the forecast for all horizons.
This is the simplest reasonable benchmark for time-series forecasting.

Run
---
    python -m src.evaluation.evaluate
    python -m src.evaluation.evaluate --no-s3
    python -m src.evaluation.evaluate --commodity ONION
"""

from __future__ import annotations

import argparse
import logging
from pathlib import Path
from typing import Dict, List

import numpy as np
import pandas as pd

from src.models.quantile_gbm.gradient_boosted_trees import QuantileGBM
from src.models.quantile_gbm.loss import pinball_loss

log = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

QUANTILES = [0.10, 0.50, 0.90]
HORIZONS = [7, 15, 30]

FEATURE_COLS = [
    "price_lag_1d", "price_lag_7d", "price_lag_14d", "price_lag_30d",
    "price_mean_7d", "price_std_7d",
    "price_mean_14d", "price_std_14d",
    "price_mean_30d", "price_std_30d",
    "price_momentum_7d",
    "arrivals_tonnes", "arrivals_mean_7d",
    "temp_mean_7d", "precip_sum_7d", "humidity_mean_7d",
    "ndvi", "ndvi_delta_30d",
    "food_cpi_index", "food_wpi_index",
    "day_of_week", "day_of_month", "month", "is_weekend",
]


# ---------------------------------------------------------------------------
# Naive persistence baseline
# ---------------------------------------------------------------------------

def naive_persistence_forecast(
    y_train: np.ndarray,
    y_test: np.ndarray,
    horizon: int,
) -> Dict[str, np.ndarray]:
    """
    Naive persistence baseline: forecast = last known training price.

    For each test sample, the forecast is the most recent observed price
    before the test period. This is the simplest possible benchmark.

    For quantile forecasts, the naive baseline predicts the same value
    for all quantiles (point forecast → degenerate interval).

    Returns
    -------
    dict with keys 'q10', 'q50', 'q90' → np.ndarray of predictions
    """
    # Use the last training price as the forecast for all test samples
    last_price = y_train[-1] if len(y_train) > 0 else np.nan

    n_test = len(y_test)
    forecast = np.full(n_test, last_price, dtype=np.float64)

    return {
        0.10: forecast,
        0.50: forecast,
        0.90: forecast,
    }


# ---------------------------------------------------------------------------
# Evaluation metrics
# ---------------------------------------------------------------------------

def compute_metrics(
    y_true: np.ndarray,
    y_pred_q: Dict[float, np.ndarray],
    label: str,
) -> dict:
    """
    Compute evaluation metrics for a set of quantile predictions.

    Returns a dict of metric_name → value.
    """
    metrics = {"method": label}

    # Pinball loss per quantile
    for q in QUANTILES:
        preds = y_pred_q.get(q)
        if preds is not None:
            metrics[f"pinball_q{int(q*100):02d}"] = pinball_loss(y_true, preds, q)

    # RMSE at median
    p50 = y_pred_q.get(0.50)
    if p50 is not None:
        metrics["rmse"] = float(np.sqrt(np.mean((y_true - p50) ** 2)))
        metrics["mape"] = float(
            np.mean(np.abs((y_true - p50) / np.clip(np.abs(y_true), 1e-6, None))) * 100
        )

    # 80% PI coverage (q10–q90)
    q10 = y_pred_q.get(0.10)
    q90 = y_pred_q.get(0.90)
    if q10 is not None and q90 is not None:
        in_80 = np.mean((y_true >= q10) & (y_true <= q90)) * 100
        metrics["coverage_80pct"] = in_80
        # Interval width (relative to median price)
        median_price = np.median(np.abs(y_true))
        if median_price > 0:
            avg_width = np.mean(q90 - q10)
            metrics["avg_interval_width"] = avg_width
            metrics["relative_interval_width"] = avg_width / median_price * 100

    return metrics


# ---------------------------------------------------------------------------
# Risk decision evaluation
# ---------------------------------------------------------------------------

def evaluate_risk_decisions(
    df: pd.DataFrame,
    price_cv_col: str = "price_cv",
    forecast_uncertainty_col: str = "forecast_uncertainty",
) -> dict:
    """
    Evaluate risk decision quality using synthetic default labels.

    Since we don't have real default labels, we create proxy labels:
    - High price volatility (CV > median) + high forecast uncertainty
      → synthetic "high risk" label (1)
    - Otherwise → synthetic "low risk" label (0)

    Then compute precision/recall/F1 for the risk model's decisions
    (APPROVE/CONDITIONAL/REJECT) against these proxy labels.
    """
    from src.models.risk_ltv_model import RiskLTVModel

    # Create synthetic risk labels based on price characteristics
    median_cv = df[price_cv_col].median()
    median_fu = df[forecast_uncertainty_col].median() if forecast_uncertainty_col in df.columns else 0.3

    # "True risk" = high CV AND high forecast uncertainty
    df = df.copy()
    df["synthetic_high_risk"] = (
        (df[price_cv_col] > median_cv) &
        (df[forecast_uncertainty_col] > median_fu)
    ).astype(int)

    # Apply risk model
    model = RiskLTVModel()
    scored = model.score(df)

    # Map decisions to binary: REJECT → 1 (high risk), APPROVE/CONDITIONAL → 0
    scored["predicted_high_risk"] = (scored["decision"] == "REJECT").astype(int)

    # Compute confusion matrix components
    tp = ((scored["predicted_high_risk"] == 1) & (scored["synthetic_high_risk"] == 1)).sum()
    fp = ((scored["predicted_high_risk"] == 1) & (scored["synthetic_high_risk"] == 0)).sum()
    fn = ((scored["predicted_high_risk"] == 0) & (scored["synthetic_high_risk"] == 1)).sum()
    tn = ((scored["predicted_high_risk"] == 0) & (scored["synthetic_high_risk"] == 0)).sum()

    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0
    accuracy = (tp + tn) / len(scored) if len(scored) > 0 else 0.0

    # Decision distribution
    decision_dist = scored["decision"].value_counts().to_dict()
    avg_risk = scored["risk_score"].mean()
    avg_ltv = scored["recommended_ltv"].mean()

    return {
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "accuracy": accuracy,
        "tp": int(tp),
        "fp": int(fp),
        "fn": int(fn),
        "tn": int(tn),
        "decision_distribution": decision_dist,
        "avg_risk_score": avg_risk,
        "avg_recommended_ltv": avg_ltv,
        "n_samples": len(scored),
    }


# ---------------------------------------------------------------------------
# Main evaluation
# ---------------------------------------------------------------------------

def evaluate_commodity(
    df: pd.DataFrame,
    commodity: str,
    feature_cols: List[str],
) -> List[dict]:
    """
    Evaluate all horizons for one commodity.
    Returns a list of metric dicts (one per horizon).
    """
    sub = df[df["commodity"] == commodity].sort_values("date").copy()

    results = []
    for horizon in HORIZONS:
        target_col = f"target_price_{horizon}d"
        if target_col not in sub.columns:
            log.warning("Target %s missing for %s — skipping", target_col, commodity)
            continue

        # Drop NaN targets and features
        available_feats = [c for c in feature_cols if c in sub.columns]
        clean = sub.dropna(subset=[target_col] + available_feats)

        if len(clean) < 200:
            log.warning(
                "Too few rows (%d) for %s @%dd — skipping",
                len(clean), commodity, horizon,
            )
            continue

        # Time-based split: 70% train, 15% val, 15% test
        n = len(clean)
        dates = clean["date"].sort_values()
        train_cut = dates.iloc[int(n * 0.70)]
        val_cut = dates.iloc[int(n * 0.85)]

        train = clean[clean["date"] <= train_cut]
        val = clean[(clean["date"] > train_cut) & (clean["date"] <= val_cut)]
        test = clean[clean["date"] > val_cut]

        X_tr = train[available_feats].values.astype(np.float64)
        y_tr = train[target_col].values.astype(np.float64)
        X_vl = val[available_feats].values.astype(np.float64)
        y_vl = val[target_col].values.astype(np.float64)
        X_te = test[available_feats].values.astype(np.float64)
        y_te = test[target_col].values.astype(np.float64)

        log.info(
            "%s @%dd  train=%d  val=%d  test=%d",
            commodity, horizon, len(X_tr), len(X_vl), len(X_te),
        )

        # --- Train model ---
        model = QuantileGBM(
            quantiles=QUANTILES,
            n_estimators=200,
            learning_rate=0.05,
            max_depth=4,
            min_samples_leaf=10,
            subsample=0.8,
            random_state=42,
        )
        model.fit(X_tr, y_tr, X_val=X_vl, y_val=y_vl)

        # --- Model predictions ---
        model_preds = model.predict(X_te)
        model_metrics = compute_metrics(y_te, model_preds, label="quantile_gbm")
        model_metrics["commodity"] = commodity
        model_metrics["horizon_d"] = horizon
        model_metrics["n_train"] = len(X_tr) + len(X_vl)
        model_metrics["n_test"] = len(X_te)

        # --- Naive baseline ---
        naive_preds = naive_persistence_forecast(y_tr, y_te, horizon)
        naive_metrics = compute_metrics(y_te, naive_preds, label="naive_persistence")
        naive_metrics["commodity"] = commodity
        naive_metrics["horizon_d"] = horizon
        naive_metrics["n_train"] = len(X_tr) + len(X_vl)
        naive_metrics["n_test"] = len(X_te)

        # --- Lift (model improvement over naive) ---
        lift = {}
        for key in ["pinball_q50", "rmse", "mape"]:
            if key in model_metrics and key in naive_metrics and naive_metrics[key] > 0:
                if key == "rmse" or key == "mape":
                    # Lower is better → positive lift means improvement
                    lift[f"{key}_lift_pct"] = (
                        (naive_metrics[key] - model_metrics[key]) / naive_metrics[key] * 100
                    )
                else:
                    # Pinball loss: lower is better
                    lift[f"{key}_lift_pct"] = (
                        (naive_metrics[key] - model_metrics[key]) / naive_metrics[key] * 100
                    )
        model_metrics.update(lift)

        results.append(model_metrics)
        results.append(naive_metrics)

        log.info(
            "  Model:  RMSE=%.2f  MAPE=%.1f%%  pinball_q50=%.4f",
            model_metrics["rmse"], model_metrics["mape"],
            model_metrics.get("pinball_q50", 0),
        )
        log.info(
            "  Naive:  RMSE=%.2f  MAPE=%.1f%%  pinball_q50=%.4f",
            naive_metrics["rmse"], naive_metrics["mape"],
            naive_metrics.get("pinball_q50", 0),
        )
        if "rmse_lift_pct" in lift:
            log.info("  Lift:   RMSE %.1f%%  MAPE %.1f%%",
                     lift.get("rmse_lift_pct", 0), lift.get("mape_lift_pct", 0))

    return results


def main():
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s  %(levelname)-8s  %(message)s",
        datefmt="%H:%M:%S",
    )

    parser = argparse.ArgumentParser(description="AgriVault model evaluation report")
    parser.add_argument("--commodity", default=None, help="Single commodity to evaluate")
    parser.add_argument("--top-n", type=int, default=10, help="Top-N commodities by volume")
    parser.add_argument("--no-s3", action="store_true", help="Use local feature files")
    args = parser.parse_args()

    # ── Load features ─────────────────────────────────────────────────────
    if args.no_s3:
        local_path = Path(__file__).resolve().parents[2] / "data" / "features" / "price_features"
        if not local_path.exists() or not list(local_path.rglob("*.parquet")):
            log.error("No local feature files found at %s", local_path)
            log.error("Run build_price_features.py first, or remove --no-s3 flag")
            return
        dfs = [pd.read_parquet(f) for f in local_path.rglob("*.parquet")]
        df = pd.concat(dfs, ignore_index=True)
        log.info("Loaded %d rows from local features", len(df))
    else:
        from src.storage.s3_client import S3Client
        s3 = S3Client()
        key_prefix = s3.key("features", "price_features/")
        df = s3.read_parquet_s3(key_prefix)
        log.info("Loaded %d rows from S3", len(df))

    df["date"] = pd.to_datetime(df["date"])

    # ── Select commodities ────────────────────────────────────────────────
    if args.commodity:
        commodities = [args.commodity.upper()]
    else:
        top = (
            df.groupby("commodity")["modal_price"]
            .count()
            .sort_values(ascending=False)
            .head(args.top_n)
            .index.tolist()
        )
        commodities = top
    log.info("Evaluating commodities: %s", commodities)

    # ── Evaluate each commodity ───────────────────────────────────────────
    all_results = []
    for commodity in commodities:
        log.info("\n=== Evaluating %s ===", commodity)
        results = evaluate_commodity(df, commodity, FEATURE_COLS)
        all_results.extend(results)

    if not all_results:
        log.error("No evaluation results produced")
        return

    # ── Summary table ─────────────────────────────────────────────────────
    summary = pd.DataFrame(all_results)

    # Pivot to show model vs naive side by side
    model_rows = summary[summary["method"] == "quantile_gbm"].copy()
    naive_rows = summary[summary["method"] == "naive_persistence"].copy()

    # Merge for comparison
    merge_cols = ["commodity", "horizon_d"]
    compare_cols = ["pinball_q10", "pinball_q50", "pinball_q90", "rmse", "mape"]
    merged = model_rows[merge_cols + compare_cols].merge(
        naive_rows[merge_cols + compare_cols],
        on=merge_cols,
        suffixes=("_model", "_naive"),
    )

    # Compute lift
    for col in compare_cols:
        model_col = f"{col}_model"
        naive_col = f"{col}_naive"
        if model_col in merged.columns and naive_col in merged.columns:
            merged[f"{col}_lift_pct"] = (
                (merged[naive_col] - merged[model_col])
                / merged[naive_col].clip(lower=1e-9) * 100
            )

    # Coverage (model only)
    coverage = model_rows[merge_cols + ["coverage_80pct", "avg_interval_width", "relative_interval_width"]]
    merged = merged.merge(coverage, on=merge_cols, how="left")

    print("\n" + "=" * 80)
    print("AGRIVAULT MODEL EVALUATION REPORT")
    print("=" * 80)
    print(f"\nCommodities evaluated: {', '.join(commodities)}")
    print(f"Horizons: {HORIZONS}")
    print(f"Quantiles: {QUANTILES}")

    print("\n--- Model vs Naive Baseline (lower is better) ---\n")
    display_cols = merge_cols.copy()
    for col in compare_cols:
        display_cols.extend([f"{col}_model", f"{col}_naive", f"{col}_lift_pct"])
    display_cols.extend(["coverage_80pct", "relative_interval_width"])

    available_display = [c for c in display_cols if c in merged.columns]
    print(merged[available_display].to_string(index=False, float_format="%.4f"))

    # Lift summary
    print("\n--- Lift Summary (positive = model outperforms naive) ---\n")
    lift_cols = [c for c in merged.columns if c.endswith("_lift_pct")]
    if lift_cols:
        lift_summary = merged[merge_cols + lift_cols].copy()
        print(lift_summary.to_string(index=False, float_format="%.1f"))

    # Coverage analysis
    print("\n--- Prediction Interval Coverage ---\n")
    for _, row in merged.iterrows():
        commodity = row["commodity"]
        horizon = row["horizon_d"]
        coverage = row.get("coverage_80pct", 0)
        status = "✓ OK" if 75 <= coverage <= 85 else "⚠ CALIBRATION NEEDED"
        print(f"  {commodity} @{horizon}d: {coverage:.1f}% coverage (target: 80%) {status}")

    # Save report
    out_dir = Path(__file__).resolve().parents[2] / "data" / "evaluation"
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / "evaluation_report.csv"
    merged.to_csv(out_path, index=False)
    log.info("Report saved to %s", out_path)

    # Also save the full summary
    full_path = out_dir / "evaluation_full.csv"
    summary.to_csv(full_path, index=False)
    log.info("Full metrics saved to %s", full_path)

    print(f"\n✓ Evaluation complete. Reports saved to {out_dir}/")


if __name__ == "__main__":
    main()
