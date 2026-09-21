"""Retune graph/ensemble weights and thresholds from cached component scores."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
import time

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from fraud_pipeline.evaluate import evaluate_predictions
from fraud_pipeline.kaggle_data import mature_labels
from fraud_pipeline.risk import RiskAggregator, RiskConfig


EXPERIMENTS = {
    "A": (1.0, 0.0, 0.0),
    "B": (0.5 / 0.7, 0.2 / 0.7, 0.0),
    "C": (0.5 / 0.8, 0.0, 0.3 / 0.8),
    "D": (0.5, 0.2, 0.3),
}
INITIAL_GRAPH = {"community": 0.4, "interaction": 0.4, "pagerank": 0.2}


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


def write_json(path: Path, payload) -> None:
    path.write_text(json.dumps(safe(payload), indent=2, allow_nan=False), encoding="utf-8")


def graph_candidates(divisions: int):
    candidates = [INITIAL_GRAPH]
    for a in range(divisions + 1):
        for b in range(divisions + 1 - a):
            item = {"community": a / divisions, "interaction": b / divisions,
                    "pagerank": (divisions - a - b) / divisions}
            if item not in candidates:
                candidates.append(item)
    return candidates


def graph_score(frame: pd.DataFrame, weights: dict[str, float]) -> np.ndarray:
    c = frame["community_risk_normalized"].to_numpy(dtype=float)
    e = frame["fraud_neighbor_exposure_normalized"].to_numpy(dtype=float)
    p = frame["pagerank_normalized"].to_numpy(dtype=float)
    return weights["community"] * c + weights["interaction"] * c * e + weights["pagerank"] * p


def config(settings, weights, catch_rate):
    return RiskConfig(
        weight_supervised=weights[0], weight_anomaly=weights[1], weight_graph=weights[2],
        c_fn=float(settings["false_negative_cost"]), c_fp=float(settings["false_positive_cost"]),
        c_review=float(settings["manual_review_cost"]), review_catch_rate=float(catch_rate),
        low_points=int(settings["threshold_grid_points"]), high_points=int(settings["threshold_grid_points"]),
        tune_weights=True, weight_grid_step=float(settings["ensemble_weight_grid_step"]),
        max_review_rate=None, fn_cost_mode=str(settings["fn_cost_mode"]), full_threshold_range=True,
    )


def tune(validation, mature_mask, settings, component_weights, catch_rate, tune_graph):
    candidates = graph_candidates(int(settings["graph_weight_grid_divisions"])) if tune_graph else [INITIAL_GRAPH]
    best = None
    y = validation.loc[mature_mask, "label"].astype(int)
    amounts = validation.loc[mature_mask, "amount"].to_numpy()
    for graph_weights in candidates:
        scores = validation[["s_supervised", "s_anomaly", "s_graph"]].copy()
        if tune_graph:
            scores["s_graph"] = graph_score(validation, graph_weights)
        aggregator = RiskAggregator(config(settings, component_weights, catch_rate)).fit(scores.loc[mature_mask], y, amounts)
        bands = aggregator.predict(scores.loc[mature_mask]).risk_band.to_numpy()
        loss = aggregator.expected_loss(bands, y.to_numpy(), amounts)
        if best is None or loss < best[0]:
            best = (loss, graph_weights.copy(), aggregator)
    return best


def attach(frame, aggregator, graph_weights, use_graph):
    scores = frame[["s_supervised", "s_anomaly", "s_graph"]].copy()
    if use_graph:
        scores["s_graph"] = graph_score(frame, graph_weights)
    risk = aggregator.predict(scores)
    result = frame[["transaction_id", "account_id", "timestamp", "amount", "label",
                    "s_supervised", "s_anomaly", "s_graph"]].copy().reset_index(drop=True)
    if use_graph:
        result["s_graph"] = scores["s_graph"].to_numpy()
    return pd.concat([result, risk.reset_index(drop=True)], axis=1)


def baseline_losses(frame, aggregator):
    y = frame.label.astype(int).to_numpy()
    amounts = frame.amount.to_numpy()
    return {name: aggregator.expected_loss(np.full(len(frame), band, dtype=object), y, amounts)
            for name, band in [("allow_all", "low"), ("review_all", "medium"), ("block_all", "high")]}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--source-dir", default="artifacts/kaggle")
    parser.add_argument("--config", default="configs/kaggle_cost_unconstrained.json")
    parser.add_argument("--output-dir", default="artifacts/kaggle_cost_unconstrained")
    args = parser.parse_args()
    source, output = Path(args.source_dir), Path(args.output_dir)
    settings = json.loads(Path(args.config).read_text())
    output.mkdir(parents=True, exist_ok=True)
    manifest = json.loads((source / "run_manifest.json").read_text())
    cutoff = pd.Timestamp(manifest["validation_end_exclusive"])
    started = time.perf_counter()
    results, summary = {}, []
    for name, component_weights in EXPERIMENTS.items():
        validation = pd.read_parquet(source / name / "predictions_validation.parquet")
        test = pd.read_parquet(source / name / "predictions_test.parquet")
        mature = mature_labels(validation, cutoff, manifest["config"]["graph"]["label_maturity"])
        loss, graph_weights, aggregator = tune(validation, mature, settings, component_weights, settings["review_catch_rate"], name in {"C", "D"})
        folder = output / name
        folder.mkdir(exist_ok=True)
        split_results = {}
        for split, raw in [("validation", validation), ("test", test)]:
            scored = attach(raw, aggregator, graph_weights, name in {"C", "D"})
            metrics = evaluate_predictions(scored, "label", aggregator).metrics
            metrics["low_rate"] = float((scored.risk_band == "low").mean())
            metrics["medium_rate"] = float((scored.risk_band == "medium").mean())
            metrics["high_rate"] = float((scored.risk_band == "high").mean())
            split_results[split] = metrics
            scored.to_parquet(folder / f"predictions_{split}.parquet", index=False)
            summary.append({"experiment": name, "split": split, **metrics})
        result = {
            "component_weights": {"supervised": aggregator.config.weight_supervised,
                                  "anomaly": aggregator.config.weight_anomaly, "graph": aggregator.config.weight_graph},
            "graph_weights": graph_weights if name in {"C", "D"} else None,
            "thresholds": {"low": aggregator.low_threshold, "high": aggregator.high_threshold},
            "mature_validation_loss": loss,
            "test_baselines": baseline_losses(test, aggregator),
            **split_results,
        }
        results[name] = result
        write_json(folder / "metrics.json", result)
        print(f"{name}: thresholds={result['thresholds']} rates L/M/H={result['test']['low_rate']:.3f}/{result['test']['medium_rate']:.3f}/{result['test']['high_rate']:.3f} F1={result['test']['f1_flagged']:.4f} loss={result['test']['total_economic_loss']:.2f}", flush=True)

    # Review-effectiveness sensitivity for the complete D system.
    validation = pd.read_parquet(source / "D" / "predictions_validation.parquet")
    test = pd.read_parquet(source / "D" / "predictions_test.parquet")
    mature = mature_labels(validation, cutoff, manifest["config"]["graph"]["label_maturity"])
    sensitivity = []
    for catch_rate in settings["review_catch_rate_sensitivity"]:
        loss, graph_weights, aggregator = tune(validation, mature, settings, EXPERIMENTS["D"], catch_rate, True)
        scored = attach(test, aggregator, graph_weights, True)
        metrics = evaluate_predictions(scored, "label", aggregator).metrics
        sensitivity.append({"review_catch_rate": catch_rate, "thresholds": {"low": aggregator.low_threshold, "high": aggregator.high_threshold},
                            "component_weights": {"supervised": aggregator.config.weight_supervised, "anomaly": aggregator.config.weight_anomaly, "graph": aggregator.config.weight_graph},
                            "graph_weights": graph_weights, "mature_validation_loss": loss,
                            "test_low_rate": float((scored.risk_band == "low").mean()),
                            "test_review_rate": float((scored.risk_band == "medium").mean()),
                            "test_high_rate": float((scored.risk_band == "high").mean()), **metrics})
    write_json(output / "review_catch_sensitivity.json", sensitivity)
    write_json(output / "metrics.json", results)
    write_json(output / "run_manifest.json", {"source_run": str(source.resolve()), "source_model_code_sha256": manifest.get("model_code_sha256"),
                                               "validation_cutoff": cutoff.isoformat(), "mature_validation_rows": int(mature.sum()),
                                               "settings": settings, "elapsed_seconds": time.perf_counter() - started})
    pd.DataFrame(summary).to_csv(output / "summary_metrics.csv", index=False)
    lines = ["# Unconstrained cost-policy retuning", "", "Models and point-in-time features are frozen from the full 5M-row run. Only graph/ensemble weights and thresholds were retuned on mature validation scores.", "",
             f"Costs: FN={settings['false_negative_cost']}, FP={settings['false_positive_cost']} (training mean transaction amount), review={settings['manual_review_cost']}, catch rate={settings['review_catch_rate']}. No review or block capacity constraint was applied.", "",
             "| Experiment | Low | Medium | High | Precision | Recall | F1 | AP | Test loss |", "|---|---:|---:|---:|---:|---:|---:|---:|---:|"]
    for name in EXPERIMENTS:
        m = results[name]["test"]
        lines.append(f"| {name} | {m['low_rate']:.4f} | {m['medium_rate']:.4f} | {m['high_rate']:.4f} | {m['precision_flagged']:.4f} | {m['recall_flagged']:.4f} | {m['f1_flagged']:.4f} | {m['average_precision']:.4f} | {m['total_economic_loss']:.2f} |")
    lines += ["", "High only is a positive prediction; Medium is review. Costs are hypotheses, not values learned from the Kaggle labels. Test data was not used to choose weights or thresholds."]
    (output / "evaluation.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"Completed in {time.perf_counter() - started:.1f}s: {output}", flush=True)


if __name__ == "__main__":
    main()
