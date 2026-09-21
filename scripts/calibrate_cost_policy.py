"""Calibrate a frozen experiment score and retune its three-way cost policy.

The mature validation period is split chronologically: the earlier half fits
the calibrator and the later half selects the calibration method and policy.
No test labels are used until final evaluation.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
import time

import joblib
import numpy as np
import pandas as pd
from sklearn.isotonic import IsotonicRegression
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import brier_score_loss, log_loss

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from fraud_pipeline.evaluate import evaluate_predictions
from fraud_pipeline.kaggle_data import mature_labels
from fraud_pipeline.risk import RiskAggregator, RiskConfig


def safe(value):
    if isinstance(value, dict):
        return {k: safe(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [safe(v) for v in value]
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating, float)):
        return float(value) if np.isfinite(value) else None
    return value


def exact_cost_thresholds(
    score: np.ndarray,
    label: np.ndarray,
    *,
    false_negative_cost: float,
    false_positive_cost: float,
    review_cost: float,
    review_catch_rate: float,
) -> tuple[float, float, float]:
    """Find exact ordered Low/Medium/High boundaries in O(n log n)."""
    score = np.asarray(score, dtype=float)
    label = np.asarray(label, dtype=int)
    if len(score) == 0 or len(score) != len(label) or not np.isfinite(score).all():
        raise ValueError("Scores and labels must be finite, aligned and non-empty")

    order = np.argsort(score, kind="stable")
    s, y = score[order], label[order]
    low_cost = y * false_negative_cost
    medium_cost = review_cost + y * (1.0 - review_catch_rate) * false_negative_cost
    high_cost = (1 - y) * false_positive_cost
    low_prefix = np.r_[0.0, np.cumsum(low_cost)]
    medium_prefix = np.r_[0.0, np.cumsum(medium_cost)]
    high_prefix = np.r_[0.0, np.cumsum(high_cost)]

    # Only split before a new distinct score, because tied scores cannot be
    # assigned to different bands by deterministic thresholds.
    boundaries = np.r_[0, np.flatnonzero(np.diff(s) > 0) + 1, len(s)]
    best_loss = np.inf
    best_low_count = best_high_count = 0
    best_delta = np.inf
    best_delta_at = 0
    high_total = high_prefix[-1]
    for high_count in boundaries:
        delta = low_prefix[high_count] - medium_prefix[high_count]
        if delta < best_delta:
            best_delta = delta
            best_delta_at = int(high_count)
        loss = best_delta + medium_prefix[high_count] + high_total - high_prefix[high_count]
        if loss < best_loss:
            best_loss = float(loss)
            best_low_count = best_delta_at
            best_high_count = int(high_count)

    def threshold_at(count: int) -> float:
        return float(s[count]) if count < len(s) else float(np.nextafter(s[-1], np.inf))

    return threshold_at(best_low_count), threshold_at(best_high_count), best_loss


def assign_bands(score: np.ndarray, low: float, high: float) -> np.ndarray:
    return np.where(score < low, "low", np.where(score < high, "medium", "high"))


def cost_derived_thresholds(settings: dict) -> tuple[float, float]:
    fn = float(settings["false_negative_cost"])
    fp = float(settings["false_positive_cost"])
    review = float(settings["manual_review_cost"])
    catch = float(settings["review_catch_rate"])
    low = review / (catch * fn)
    high = (fp - review) / (fp + (1.0 - catch) * fn)
    return float(np.clip(low, 0, 1)), float(np.clip(high, 0, 1))


def calibration_metrics(y: np.ndarray, probability: np.ndarray) -> dict[str, float]:
    p = np.clip(np.asarray(probability, dtype=float), 1e-12, 1 - 1e-12)
    return {
        "brier_score": float(brier_score_loss(y, p)),
        "log_loss": float(log_loss(y, p, labels=[0, 1])),
        "mean_probability": float(p.mean()),
        "fraud_prevalence": float(np.mean(y)),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source-dir", default="artifacts/kaggle_cost_unconstrained")
    parser.add_argument("--raw-run-dir", default="artifacts/kaggle")
    parser.add_argument("--config", default="configs/kaggle_cost_unconstrained.json")
    parser.add_argument("--experiment", default="D")
    parser.add_argument("--output-dir", default="artifacts/kaggle_calibrated")
    args = parser.parse_args()

    started = time.perf_counter()
    source = Path(args.source_dir) / args.experiment
    output = Path(args.output_dir) / args.experiment
    output.mkdir(parents=True, exist_ok=True)
    settings = json.loads(Path(args.config).read_text(encoding="utf-8"))
    raw_manifest = json.loads((Path(args.raw_run_dir) / "run_manifest.json").read_text(encoding="utf-8"))

    validation = pd.read_parquet(source / "predictions_validation.parquet")
    test = pd.read_parquet(source / "predictions_test.parquet")
    cutoff = pd.Timestamp(raw_manifest["validation_end_exclusive"])
    delay = raw_manifest["config"]["graph"]["label_maturity"]
    validation = validation.loc[mature_labels(validation, cutoff, delay)].sort_values(
        ["timestamp", "transaction_id"], kind="stable"
    ).reset_index(drop=True)
    split = len(validation) // 2
    calibration, policy = validation.iloc[:split], validation.iloc[split:]

    x_cal = calibration[["final_score"]].to_numpy(dtype=float)
    y_cal = calibration.label.to_numpy(dtype=int)
    x_policy = policy[["final_score"]].to_numpy(dtype=float)
    y_policy = policy.label.to_numpy(dtype=int)
    x_test = test[["final_score"]].to_numpy(dtype=float)
    y_test = test.label.to_numpy(dtype=int)

    platt = LogisticRegression(C=1e6, solver="lbfgs", max_iter=1000).fit(x_cal, y_cal)
    isotonic = IsotonicRegression(y_min=0.0, y_max=1.0, out_of_bounds="clip").fit(x_cal[:, 0], y_cal)
    calibrators = {
        "platt": (platt, lambda model, x: model.predict_proba(x)[:, 1]),
        "isotonic": (isotonic, lambda model, x: model.predict(x[:, 0])),
    }
    candidates = {}
    for name, (model, predict) in calibrators.items():
        policy_probability = predict(model, x_policy)
        candidates[name] = calibration_metrics(y_policy, policy_probability)

    # Select calibration quality without looking at test labels.
    selected_method = min(candidates, key=lambda name: candidates[name]["brier_score"])
    selected_model, predict_probability = calibrators[selected_method]
    policy_probability = predict_probability(selected_model, x_policy)
    test_probability = predict_probability(selected_model, x_test)

    costs = {
        "false_negative_cost": float(settings["false_negative_cost"]),
        "false_positive_cost": float(settings["false_positive_cost"]),
        "review_cost": float(settings["manual_review_cost"]),
        "review_catch_rate": float(settings["review_catch_rate"]),
    }
    empirical_low, empirical_high, empirical_loss = exact_cost_thresholds(
        policy_probability, y_policy, **costs
    )
    derived_low, derived_high = cost_derived_thresholds(settings)

    risk = RiskAggregator(RiskConfig(
        weight_supervised=1.0,
        weight_anomaly=0.0,
        weight_graph=0.0,
        c_fn=costs["false_negative_cost"],
        c_fp=costs["false_positive_cost"],
        c_review=costs["review_cost"],
        review_catch_rate=costs["review_catch_rate"],
        fn_cost_mode=str(settings["fn_cost_mode"]),
    ))

    policies = {}
    for policy_name, (low, high) in {
        "empirical_validation_optimum": (empirical_low, empirical_high),
        "cost_derived_calibrated": (derived_low, derived_high),
    }.items():
        policy_bands = assign_bands(policy_probability, low, high)
        policy_loss = risk.expected_loss(policy_bands, y_policy, policy.amount.to_numpy())
        test_bands = assign_bands(test_probability, low, high)
        scored = test[["transaction_id", "account_id", "timestamp", "amount", "label"]].copy()
        scored["uncalibrated_score"] = test.final_score.to_numpy()
        scored["final_score"] = test_probability
        scored["risk_band"] = test_bands
        metrics = evaluate_predictions(scored, "label", risk).metrics
        metrics.update({
            "low_rate": float(np.mean(test_bands == "low")),
            "medium_rate": float(np.mean(test_bands == "medium")),
            "high_rate": float(np.mean(test_bands == "high")),
        })
        scored.to_parquet(output / f"predictions_test_{policy_name}.parquet", index=False)
        policies[policy_name] = {
            "thresholds": {"low": low, "high": high},
            "policy_validation_loss": policy_loss,
            "test": metrics,
        }

    selected_policy = min(policies, key=lambda name: policies[name]["policy_validation_loss"])
    result = {
        "experiment": args.experiment,
        "chronological_split": {
            "mature_validation_rows": len(validation),
            "calibrator_fit_rows": len(calibration),
            "policy_selection_rows": len(policy),
            "calibrator_fit_end": str(calibration.timestamp.iloc[-1]),
            "policy_selection_start": str(policy.timestamp.iloc[0]),
        },
        "costs": costs,
        "calibration_candidates_on_policy_validation": candidates,
        "selected_calibration": selected_method,
        "selected_calibration_test": calibration_metrics(y_test, test_probability),
        "selected_policy": selected_policy,
        "policies": policies,
        "elapsed_seconds": time.perf_counter() - started,
        "note": "Ensemble and graph weights were frozen from the source run; no test labels selected calibration or thresholds.",
    }
    (output / "metrics.json").write_text(
        json.dumps(safe(result), indent=2, allow_nan=False), encoding="utf-8"
    )
    joblib.dump(selected_model, output / f"{selected_method}_calibrator.joblib")

    selected = policies[selected_policy]
    report = [
        "# Calibrated D cost policy",
        "",
        f"Selected calibration: {selected_method} (lowest Brier score on the later mature validation half).",
        f"Selected policy: {selected_policy} (lowest economic loss on that same policy-selection half).",
        f"Thresholds: low={selected['thresholds']['low']:.8f}, high={selected['thresholds']['high']:.8f}.",
        f"Test rates: Low={selected['test']['low_rate']:.6f}, Medium={selected['test']['medium_rate']:.6f}, High={selected['test']['high_rate']:.6f}.",
        f"Test economic loss: {selected['test']['total_economic_loss']:.2f}.",
        "",
        "Calibration changes probability interpretation, not rank ordering. Scores remain weakly ranked if ROC-AUC/AP do not improve.",
    ]
    (output / "evaluation.md").write_text("\n".join(report) + "\n", encoding="utf-8")
    print(json.dumps(safe(result), indent=2, allow_nan=False), flush=True)


if __name__ == "__main__":
    main()
