from __future__ import annotations

import argparse
import copy
import json
from pathlib import Path
import sys
from typing import Any

import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, average_precision_score, roc_auc_score

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from fraud_pipeline.config import PipelineConfig, load_config
from fraud_pipeline.data import load_transactions, standardize_schema, temporal_split
from fraud_pipeline.evaluate import evaluate_predictions
from fraud_pipeline.pipeline import FraudPipeline


EXPERIMENTS = {
    "A": {"name": "Supervised only", "anomaly": False, "graph": False},
    "B": {"name": "Supervised + anomaly", "anomaly": True, "graph": False},
    "C": {"name": "Supervised + graph", "anomaly": False, "graph": True},
    "D": {"name": "Supervised + anomaly + graph", "anomaly": True, "graph": True},
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run the A-D fraud-pipeline ablation study")
    parser.add_argument("--input", default="data/transactions.csv")
    parser.add_argument("--config", default="configs/default.json")
    parser.add_argument("--train-end", default="2025-06-30")
    parser.add_argument("--val-end", default="2025-07-31")
    parser.add_argument("--output-dir", default="artifacts/ablation")
    parser.add_argument("--save-models", action="store_true")
    parser.add_argument(
        "--anomaly-backend", choices=("pca", "torch", "auto"), default="pca",
        help="Reconstruction-error backend; PCA is deterministic and fast for ablation runs",
    )
    return parser.parse_args()


def _json_safe(value: Any) -> Any:
    if isinstance(value, dict):
        return {key: _json_safe(item) for key, item in value.items()}
    if isinstance(value, list):
        return [_json_safe(item) for item in value]
    if isinstance(value, (np.floating, float)):
        return None if not np.isfinite(value) else float(value)
    if isinstance(value, (np.integer, int)):
        return int(value)
    return value


def _extended_metrics(scored: pd.DataFrame, label_col: str, base: dict[str, float]) -> dict[str, Any]:
    metrics: dict[str, Any] = dict(base)
    y_true = pd.to_numeric(scored[label_col], errors="coerce").fillna(0).astype(int).to_numpy()
    score = scored["final_score"].to_numpy(dtype=float)
    flagged = (scored["risk_band"].to_numpy() == "high").astype(int)
    tn = int(((flagged == 0) & (y_true == 0)).sum())
    fp = int(((flagged == 1) & (y_true == 0)).sum())
    metrics["accuracy_flagged"] = float(accuracy_score(y_true, flagged))
    metrics["specificity_flagged"] = float(tn / max(tn + fp, 1))
    metrics["flagged_rate"] = float(flagged.mean())
    metrics["fraud_prevalence"] = float(y_true.mean())
    metrics["roc_auc"] = float(roc_auc_score(y_true, score)) if len(np.unique(y_true)) > 1 else None
    metrics["average_precision"] = (
        float(average_precision_score(y_true, score)) if len(np.unique(y_true)) > 1 else None
    )
    return _json_safe(metrics)


def _report(summary: pd.DataFrame, split_counts: dict[str, int], anomaly_backend: str) -> str:
    test = summary[summary["split"] == "test"].copy().sort_values("experiment")
    best_f1 = test.loc[test["f1_flagged"].idxmax()]
    best_ap = test.loc[test["average_precision"].idxmax()]
    best_loss = test.loc[test["total_economic_loss"].idxmin()]
    baseline = test[test["experiment"] == "A"].iloc[0]
    graph_only = test[test["experiment"] == "C"].iloc[0]
    full_model = test[test["experiment"] == "D"].iloc[0]
    loss_reduction = 100.0 * (
        baseline["total_economic_loss"] - best_loss["total_economic_loss"]
    ) / baseline["total_economic_loss"]
    display_cols = [
        "experiment", "components", "roc_auc", "average_precision", "precision_flagged",
        "recall_flagged", "f1_flagged", "specificity_flagged", "flagged_rate",
        "total_economic_loss", "avg_economic_loss_per_tx",
    ]
    display = test[display_cols].copy()
    headers = list(display.columns)
    table_lines = [
        "| " + " | ".join(headers) + " |",
        "| " + " | ".join(["---"] * len(headers)) + " |",
    ]
    for _, row in display.iterrows():
        cells = []
        for column in headers:
            value = row[column]
            cells.append(f"{value:.4f}" if isinstance(value, (float, np.floating)) else str(value))
        table_lines.append("| " + " | ".join(cells) + " |")
    table = "\n".join(table_lines)
    return f"""# Fraud Detection Ablation Study

## Design

- Temporal split: {split_counts['train']} train, {split_counts['validation']} validation, {split_counts['test']} test transactions.
- All experiments use the same data, random seed, supervised learner, cost model, and validation-based threshold search.
- Disabled components are neither trained nor used. Graph features are excluded from the supervised feature matrix when graph is disabled.
- Anomaly reconstruction backend: {anomaly_backend} (combined 50/50 with Isolation Forest when enabled).
- A: supervised only; B: supervised + anomaly; C: supervised + graph; D: supervised + anomaly + graph.

## Test-set results

{table}

## Evaluation

- Best flagged F1: experiment {best_f1['experiment']} ({best_f1['components']}) at {best_f1['f1_flagged']:.4f}.
- Best ranking quality (average precision): experiment {best_ap['experiment']} ({best_ap['components']}) at {best_ap['average_precision']:.4f}.
- Lowest modeled economic loss: experiment {best_loss['experiment']} ({best_loss['components']}) at {best_loss['total_economic_loss']:.2f}, a {loss_reduction:.1f}% reduction from A.
- Compared with C, D changed average precision from {graph_only['average_precision']:.4f} to {full_model['average_precision']:.4f}, flagged F1 from {graph_only['f1_flagged']:.4f} to {full_model['f1_flagged']:.4f}, and modeled loss from {graph_only['total_economic_loss']:.2f} to {full_model['total_economic_loss']:.2f}.
- Validation metrics are included in `summary_metrics.csv` and `metrics.json`; interpret test results as a single synthetic-data evaluation rather than a confidence interval.
"""


def main() -> None:
    args = parse_args()
    base_cfg = load_config(args.config)
    if base_cfg.raw.get("data", {}).get("profile") == "kaggle":
        raise ValueError("Use scripts/run_kaggle_study.py for cached point-in-time Kaggle ablations")
    cols = base_cfg.columns
    df = standardize_schema(
        load_transactions(args.input), timestamp_col=cols["timestamp"], label_col=cols["label"]
    )
    splits = temporal_split(df, args.train_end, args.val_end, timestamp_col=cols["timestamp"])
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    all_metrics: dict[str, Any] = {}
    rows: list[dict[str, Any]] = []
    for experiment, definition in EXPERIMENTS.items():
        print(f"Running {experiment}: {definition['name']}", flush=True)
        raw = copy.deepcopy(base_cfg.raw)
        raw["components"] = {
            "supervised": True,
            "anomaly": definition["anomaly"],
            "graph": definition["graph"],
        }
        raw.setdefault("anomaly", {}).setdefault("autoencoder", {})["backend"] = args.anomaly_backend
        pipeline = FraudPipeline(PipelineConfig(raw=raw))
        fit = pipeline.fit(splits.train, splits.validation)
        history = pd.concat([splits.train, splits.validation], ignore_index=True)
        test_scored = pipeline.predict(splits.test, history_df=history)

        experiment_dir = out_dir / experiment
        experiment_dir.mkdir(parents=True, exist_ok=True)
        fit.validation_scored.to_csv(experiment_dir / "predictions_validation.csv", index=False)
        test_scored.to_csv(experiment_dir / "predictions_test.csv", index=False)
        pipeline.dump_metadata(experiment_dir / "metadata.json")
        if args.save_models:
            pipeline.save(experiment_dir / "pipeline.joblib")

        split_metrics: dict[str, Any] = {}
        for split_name, scored in (("validation", fit.validation_scored), ("test", test_scored)):
            evaluated = evaluate_predictions(
                scored, label_col=cols["label"], risk_aggregator=pipeline.risk_aggregator,
                timestamp_col=cols["timestamp"],
            )
            metrics = _extended_metrics(scored, cols["label"], evaluated.metrics)
            split_metrics[split_name] = metrics
            rows.append({
                "experiment": experiment,
                "components": definition["name"],
                "split": split_name,
                **metrics,
                "low_threshold": pipeline.risk_aggregator.low_threshold,
                "high_threshold": pipeline.risk_aggregator.high_threshold,
            })

        all_metrics[experiment] = {
            "components": definition,
            "thresholds": {
                "low": pipeline.risk_aggregator.low_threshold,
                "high": pipeline.risk_aggregator.high_threshold,
            },
            **split_metrics,
        }
        with (experiment_dir / "metrics.json").open("w", encoding="utf-8") as handle:
            json.dump(_json_safe(all_metrics[experiment]), handle, indent=2)

    summary = pd.DataFrame(rows)
    summary.to_csv(out_dir / "summary_metrics.csv", index=False)
    with (out_dir / "metrics.json").open("w", encoding="utf-8") as handle:
        json.dump(_json_safe(all_metrics), handle, indent=2)
    split_counts = {
        "train": len(splits.train), "validation": len(splits.validation), "test": len(splits.test)
    }
    (out_dir / "evaluation.md").write_text(
        _report(summary, split_counts, args.anomaly_backend), encoding="utf-8"
    )
    print(f"Saved ablation metrics and evaluation to {out_dir}")


if __name__ == "__main__":
    main()
