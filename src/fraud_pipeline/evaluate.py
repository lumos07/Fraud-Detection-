from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from sklearn.metrics import precision_score, recall_score, f1_score

from fraud_pipeline.risk import RiskAggregator


@dataclass
class EvaluationResult:
    metrics: dict[str, float]
    frame: pd.DataFrame


def evaluate_predictions(
    scored_frame: pd.DataFrame,
    label_col: str,
    risk_aggregator: RiskAggregator,
    timestamp_col: str = "timestamp",
) -> EvaluationResult:
    if label_col not in scored_frame.columns:
        raise ValueError(f"Missing label column for evaluation: {label_col}")

    out = scored_frame.copy()
    y_col = out[label_col]
    if isinstance(y_col, pd.DataFrame):
        y_col = y_col.iloc[:, 0]
    y_true = pd.to_numeric(y_col, errors="coerce").fillna(0).astype(int).to_numpy()
    risk_band = out["risk_band"].to_numpy()

    flagged = (risk_band != "low").astype(int)
    high_blocked = (risk_band == "high").astype(int)

    tp = int(((flagged == 1) & (y_true == 1)).sum())
    fp = int(((flagged == 1) & (y_true == 0)).sum())
    fn = int(((flagged == 0) & (y_true == 1)).sum())
    tn = int(((flagged == 0) & (y_true == 0)).sum())

    precision = float(precision_score(y_true, flagged, zero_division=0))
    recall = float(recall_score(y_true, flagged, zero_division=0))
    f1 = float(f1_score(y_true, flagged, zero_division=0))
    loss = float(risk_aggregator.expected_loss(risk_band, y_true))

    metrics = {
        "n_samples": float(len(out)),
        "tp_flagged": float(tp),
        "fp_flagged": float(fp),
        "fn_low_band": float(fn),
        "tn_low_band": float(tn),
        "precision_flagged": precision,
        "recall_flagged": recall,
        "f1_flagged": f1,
        "high_block_rate": float(high_blocked.mean()),
        "total_economic_loss": loss,
        "avg_economic_loss_per_tx": float(loss / max(len(out), 1)),
    }

    if timestamp_col in out.columns and "detected_at" in out.columns:
        ts = pd.to_datetime(out[timestamp_col], utc=True, errors="coerce")
        detected = pd.to_datetime(out["detected_at"], utc=True, errors="coerce")
        latency = (detected - ts).dt.total_seconds() / 3600.0
        metrics["detection_latency_hours"] = float(latency.dropna().mean())
    else:
        metrics["detection_latency_hours"] = float("nan")

    return EvaluationResult(metrics=metrics, frame=out)


def network_disruption_rate(
    flagged_accounts: set[str],
    component_map: dict[str, int],
) -> float:
    if not component_map:
        return 0.0
    account_count = len(component_map)
    impacted = sum(1 for a in flagged_accounts if a in component_map)
    return float(impacted / max(account_count, 1))
