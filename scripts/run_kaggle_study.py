"""Reusable chronological A-D, internal ablation and robustness experiment runner."""
from __future__ import annotations

import argparse
import copy
from dataclasses import asdict
import hashlib
import importlib.metadata
import inspect
import json
from pathlib import Path
import sys
import time

import numpy as np
import pandas as pd
import polars as pl

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from fraud_pipeline.config import PipelineConfig, load_config
from fraud_pipeline.evaluate import evaluate_predictions
from fraud_pipeline.kaggle_data import adapt_transactions, mature_labels
from fraud_pipeline.kaggle_pipeline import KagglePipeline
from fraud_pipeline.risk import RiskAggregator, RiskConfig
from fraud_pipeline.temporal_features import build_temporal_features
from fraud_pipeline.temporal_graph import temporal_graph_features


def safe(value):
    if isinstance(value, dict): return {k: safe(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)): return [safe(v) for v in value]
    if isinstance(value, (np.floating, float)): return float(value) if np.isfinite(value) else None
    if isinstance(value, np.integer): return int(value)
    return value


def write_json(path, payload):
    Path(path).write_text(json.dumps(safe(payload), default=str, indent=2, allow_nan=False))


def args_parser():
    p = argparse.ArgumentParser()
    p.add_argument("--input", default="data/kaggle/transactions.parquet")
    p.add_argument("--config", default="configs/kaggle.json")
    p.add_argument("--output-dir", default="artifacts/kaggle")
    p.add_argument("--sample-every", type=int, default=1, help="1 uses all rows; >1 is a clearly labelled systematic smoke sample")
    p.add_argument("--experiments", nargs="+", default=["A", "B", "C", "D", "graph_pagerank", "graph_community", "graph_exposure", "graph_community_exposure", "anomaly_if", "anomaly_ae"])
    p.add_argument("--robustness", action="store_true")
    p.add_argument("--resume", action="store_true", help="Reuse verified feature cache, not prior fitted models")
    return p.parse_args()


def stress_cases(test, legitimate_train, seed):
    rng = np.random.default_rng(seed)
    drift = test.copy()
    drift["amount"] *= 1.3
    yield "temporal_drift", drift
    del drift
    mimic = test.copy()
    fraud = mimic.label.eq(1)
    mimic.loc[fraud, "amount"] = rng.choice(legitimate_train.amount.to_numpy(), int(fraud.sum()))
    yield "amount_mimicry", mimic
    behavior = mimic.copy()
    for col in ["transaction_type", "device_used", "location", "merchant_category"]:
        if col in behavior:
            behavior[col] = behavior[col].astype(str)
            behavior.loc[fraud, col] = rng.choice(legitimate_train[col].astype(str).to_numpy(), int(fraud.sum()))
    yield "behavior_mimicry", behavior
    del behavior, mimic
    threshold = legitimate_train.amount.quantile(.99)
    to_split = test.label.eq(1) & (test.amount >= threshold)
    pieces = []
    for i in range(4):
        piece = test.loc[to_split].copy()
        piece["amount"] /= 4
        piece["transaction_id"] = piece.transaction_id + f"_split_{i}"
        pieces.append(piece)
    yield "transaction_splitting", pd.concat([test.loc[~to_split]] + pieces, ignore_index=True).sort_values("timestamp")


def robustness(model, test, history, legitimate_train, feature_model, cache, folder):
    results = {}
    for name, frame in stress_cases(test, legitimate_train, model.config.seed):
        if name in cache:
            features = pd.read_parquet(cache[name])
        else:
            print(f"Preparing robustness features: {name}", flush=True)
            features, _, _ = feature_model.engineer(frame, history)
            path = folder / f"features_robustness_{name}.parquet"
            features.to_parquet(path, index=False)
            cache[name] = path
        scored = model._attach_risk(features, model._score_components(features))
        results[name] = evaluate_predictions(scored, "label", model.risk_aggregator).metrics
    return results


def main():
    args = args_parser()
    if args.sample_every < 1: raise ValueError("sample-every must be positive")
    cfg = load_config(args.config)
    out = Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=True)
    started = time.perf_counter()
    lazy = pl.scan_parquet(args.input)
    total_rows = lazy.select(pl.len()).collect().item()
    if args.sample_every > 1:
        lazy = lazy.gather_every(args.sample_every)
    data = adapt_transactions(lazy.collect().to_pandas(), require_labels=True)
    # Boundaries on actual timestamps keep equal-time events together.
    train_end = data.timestamp.iloc[int(len(data) * .6)]
    val_end = data.timestamp.iloc[int(len(data) * .8)]
    train = data[data.timestamp < train_end].copy()
    validation = data[(data.timestamp >= train_end) & (data.timestamp < val_end)].copy()
    test = data[data.timestamp >= val_end].copy()
    cfg.raw["validation_label_cutoff"] = val_end.isoformat()
    if any(part.empty for part in [train, validation, test]):
        raise ValueError("Chronological split has an empty partition; use more data or a smaller sampling interval")
    source_dir = Path(__file__).resolve().parents[1] / "src" / "fraud_pipeline"
    model_code_hash = hashlib.sha256(b"".join(p.read_bytes() for p in sorted(source_dir.glob("*.py"))) + Path(__file__).read_bytes()).hexdigest()
    manifest = {"source_rows": total_rows, "used_rows": len(data), "sample_every": args.sample_every, "model_code_sha256": model_code_hash,
                "sampled_run": args.sample_every > 1, "train_rows": len(train), "validation_rows": len(validation), "test_rows": len(test),
                "train_end_exclusive": train_end.isoformat(), "validation_end_exclusive": val_end.isoformat(),
                "fraud_counts": {k: int(v.label.sum()) for k, v in [("train", train), ("validation", validation), ("test", test)]},
                "config": cfg.raw, "packages": {p: importlib.metadata.version(p) for p in ["pandas", "polars", "igraph", "scikit-learn", "lightgbm", "xgboost", "torch"]}}
    write_json(out / "run_manifest.json", manifest)
    print(f"Loaded {len(data):,} rows (sample_every={args.sample_every}); chronological split {len(train):,}/{len(validation):,}/{len(test):,}", flush=True)
    feature_code = b"".join((source_dir / file).read_bytes() for file in ["kaggle_data.py", "temporal_features.py", "temporal_graph.py"])
    signature = hashlib.sha256(feature_code + (json.dumps(cfg.raw, sort_keys=True) + str(Path(args.input).resolve()) + str(Path(args.input).stat().st_mtime_ns) + str(args.sample_every)).encode()).hexdigest()
    cache_manifest = out / "feature_manifest.json"
    if args.resume and cache_manifest.exists():
        cached = json.loads(cache_manifest.read_text())
        if cached["signature"] != signature:
            raise ValueError("Feature cache configuration/input mismatch; rerun without --resume")
        numeric, categorical = cached["numeric"], cached["categorical"]
        frames = {part: pd.read_parquet(out / f"features_{part}.parquet") for part in ["train", "validation", "test"]}
    else:
        print("Generating strict historical behavioral features", flush=True)
        engineered = build_temporal_features(data)
        numeric, categorical = engineered.numeric_cols, engineered.categorical_cols
        frames = {}
        indexed = engineered.frame.set_index("transaction_id", drop=False)
        for part, target in [("train", train), ("validation", validation), ("test", test)]:
            indexed.loc[target.transaction_id].reset_index(drop=True).to_parquet(out / f"behavioral_{part}.parquet", index=False)
        del engineered, indexed
        audits = {}
        for part, target, history in [("train", train, train), ("validation", validation, train), ("test", test, pd.concat([train, validation], ignore_index=True))]:
            print(f"Generating {part} graph snapshots", flush=True)
            base = pd.read_parquet(out / f"behavioral_{part}.parquet")
            graph, audit = temporal_graph_features(history, target.reset_index(drop=True), cfg.graph)
            frames[part] = pd.concat([base, graph], axis=1)
            frames[part].to_parquet(out / f"features_{part}.parquet", index=False)
            audits[part] = audit
        write_json(out / "snapshot_audit.json", audits)
        write_json(cache_manifest, {"signature": signature, "numeric": numeric, "categorical": categorical})
    print(f"Feature preparation completed in {time.perf_counter() - started:.1f}s", flush=True)
    model_cache, anomaly_cache = {}, None
    robustness_cache = {}
    stress_signature = hashlib.sha256((signature + inspect.getsource(stress_cases)).encode()).hexdigest()
    stress_manifest = out / "robustness_feature_manifest.json"
    if args.resume and stress_manifest.exists():
        prior_stress = json.loads(stress_manifest.read_text())
        if prior_stress["signature"] == stress_signature:
            robustness_cache = {k: Path(v) for k, v in prior_stress["files"].items() if Path(v).exists()}
    robustness_feature_model = KagglePipeline(cfg)
    summary, report_results, robustness_results = [], {}, {}
    for name in args.experiments:
        raw = copy.deepcopy(cfg.raw)
        if name in "ABCD" and len(name) == 1:
            graph_on, anomaly_on = name in "CD", name in "BD"
        elif name.startswith("graph_"):
            graph_on, anomaly_on = True, True
            raw["graph"]["ablation"] = name.removeprefix("graph_")
            raw["graph"]["tune_weights"] = False
        elif name in ["anomaly_if", "anomaly_ae"]:
            graph_on, anomaly_on = True, True
            raw["anomaly"]["weights"] = {"isolation_forest": float(name == "anomaly_if"), "autoencoder": float(name == "anomaly_ae")}
        else:
            raise ValueError(f"Unknown experiment {name}")
        raw["components"] = {"supervised": True, "anomaly": anomaly_on, "graph": graph_on}
        model = KagglePipeline(PipelineConfig(raw))
        model.supervised_model = model_cache.get(graph_on)
        if anomaly_on and anomaly_cache:
            model.isolation_model, model.autoencoder_model, model.normalizers = anomaly_cache
        run_started = time.perf_counter()
        print(f"Fitting {name}", flush=True)
        fit = model.fit_frames(frames["train"], frames["validation"], numeric, categorical, train)
        model_cache[graph_on] = model.supervised_model
        if anomaly_on and model.isolation_model and model.autoencoder_model:
            anomaly_cache = (model.isolation_model, model.autoencoder_model, model.normalizers)
        folder = out / name
        folder.mkdir(exist_ok=True)
        metrics = {"positive_prediction": "high only", "thresholds": {"low": model.risk_aggregator.low_threshold, "high": model.risk_aggregator.high_threshold}}
        for split, scored in [("validation", fit.validation_scored), ("test", model._attach_risk(frames["test"], model._score_components(frames["test"])) )]:
            metrics[split] = evaluate_predictions(scored, "label", model.risk_aggregator).metrics
            summary.append({"experiment": name, "split": split, **metrics[split]})
            columns = [c for c in scored if c in ["transaction_id", "account_id", "timestamp", "amount", "label", "final_score", "risk_band", "s_supervised", "s_anomaly", "s_graph", "community_risk_normalized", "fraud_neighbor_exposure_normalized", "pagerank_normalized", "interaction", "autoencoder_raw", "autoencoder_percentile", "isolation_forest_percentile"]]
            scored[columns].to_parquet(folder / f"predictions_{split}.parquet", index=False)
        metrics["fit_and_score_seconds"] = time.perf_counter() - run_started
        write_json(folder / "metrics.json", metrics)
        model.dump_metadata(folder / "metadata.json")
        write_json(folder / "config.json", raw)
        report_results[name] = metrics
        if name == "D":
            # Save only training history. Inference callers provide additional observed history.
            model.save(folder / "pipeline.joblib")
            sensitivity = []
            val = fit.validation_scored
            mask = mature_labels(val, val_end, cfg.graph.get("label_maturity", "7d"))
            for fn in [1000., 10000., 604000.]:
                for fp in [10., 75., 200.]:
                    rc = copy.deepcopy(model.risk_aggregator.config)
                    rc.c_fn, rc.c_fp, rc.tune_weights = fn, fp, False
                    agg = RiskAggregator(rc)
                    agg.fit(val.loc[mask, ["s_supervised", "s_anomaly", "s_graph"]], val.loc[mask, "label"].astype(int), val.loc[mask, "amount"].to_numpy())
                    bands = agg.predict(val.loc[mask, ["s_supervised", "s_anomaly", "s_graph"]]).risk_band.to_numpy()
                    sensitivity.append({"c_fn": fn, "c_fp": fp, "low": agg.low_threshold, "high": agg.high_threshold, "validation_loss": agg.expected_loss(bands, val.loc[mask, "label"].to_numpy(), val.loc[mask, "amount"].to_numpy())})
            write_json(out / "cost_sensitivity.json", sensitivity)
        if args.robustness and name in "ABCD":
            print(f"Robustness replay for {name}", flush=True)
            robustness_results[name] = robustness(model, test, pd.concat([train, validation], ignore_index=True), train[train.label == 0], robustness_feature_model, robustness_cache, out)
            write_json(stress_manifest, {"signature": stress_signature, "files": {k: str(v.resolve()) for k, v in robustness_cache.items()}})
            write_json(out / "robustness.json", robustness_results)
        pd.DataFrame(summary).to_csv(out / "summary_metrics.csv", index=False)
        write_json(out / "metrics.json", report_results)
        print(f"{name}: test AP={metrics['test']['average_precision']:.4f}, F1={metrics['test']['f1_flagged']:.4f}, loss={metrics['test']['total_economic_loss']:.2f}", flush=True)
    lines = ["# Kaggle fraud study", "", f"Rows used: {len(data):,} of {total_rows:,}; systematic sampling interval: {args.sample_every}.", "", "Precision/recall/F1 use high risk only. Review costs include medium risk. This dataset is synthetic.", "", "| Experiment | AP | Precision | Recall | F1 | Economic loss |", "|---|---:|---:|---:|---:|---:|"]
    for name, result in report_results.items():
        m = result["test"]
        lines.append(f"| {name} | {m['average_precision']:.4f} | {m['precision_flagged']:.4f} | {m['recall_flagged']:.4f} | {m['f1_flagged']:.4f} | {m['total_economic_loss']:.2f} |")
    lines += ["", "Graph and ensemble weights were selected on matured validation labels only. All graph snapshots precede their scoring periods. Anomaly models see behavioral numeric features only. Labels are attributed to senders, not automatically to receivers. The 7-day label delay is an assumption because the source has no observed confirmation times. Test labels never update test graph features; only supplied training/validation history may contribute matured labels.", "", "Internal graph ablations change the independent graph score, retaining the same compact supervised graph subset. Model variants reuse identically fitted components where inputs/configuration match. These are single chronological-split results, not confidence intervals."]
    (out / "evaluation.md").write_text("\n".join(lines) + "\n")
    print(f"Finished in {time.perf_counter() - started:.1f}s; results: {out}", flush=True)


if __name__ == "__main__":
    main()
