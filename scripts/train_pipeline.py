from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from fraud_pipeline.config import load_config
from fraud_pipeline.data import load_transactions, standardize_schema, temporal_split
from fraud_pipeline.evaluate import evaluate_predictions
from fraud_pipeline.pipeline import FraudPipeline


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train adaptive fraud pipeline")
    parser.add_argument("--input", required=True, help="Path to input csv/parquet")
    parser.add_argument("--config", default="configs/default.json", help="Config path (.json/.yaml)")
    parser.add_argument("--train-end", required=True, help="Train split end date (YYYY-MM-DD)")
    parser.add_argument("--val-end", required=True, help="Validation split end date (YYYY-MM-DD)")
    parser.add_argument("--output-dir", default="artifacts", help="Directory for artifacts")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    cfg = load_config(args.config)
    cols = cfg.columns

    raw = load_transactions(args.input)
    df = standardize_schema(
        raw,
        timestamp_col=cols["timestamp"],
        label_col=cols["label"],
    )
    splits = temporal_split(
        df,
        train_end=args.train_end,
        validation_end=args.val_end,
        timestamp_col=cols["timestamp"],
    )

    pipeline = FraudPipeline(cfg)
    fit_artifacts = pipeline.fit(splits.train, splits.validation)

    val_eval = evaluate_predictions(
        fit_artifacts.validation_scored,
        label_col=cols["label"],
        risk_aggregator=pipeline.risk_aggregator,
        timestamp_col=cols["timestamp"],
    )

    history_for_test = pd.concat([splits.train, splits.validation], ignore_index=True)
    test_scored = pipeline.predict(splits.test, history_df=history_for_test)
    test_eval = evaluate_predictions(
        test_scored,
        label_col=cols["label"],
        risk_aggregator=pipeline.risk_aggregator,
        timestamp_col=cols["timestamp"],
    )

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    model_path = out_dir / "pipeline.joblib"
    pipeline.save(model_path)
    pipeline.dump_metadata(out_dir / "metadata.json")

    fit_artifacts.validation_scored.to_csv(out_dir / "predictions_validation.csv", index=False)
    test_scored.to_csv(out_dir / "predictions_test.csv", index=False)

    metrics = {
        "validation": val_eval.metrics,
        "test": test_eval.metrics,
        "thresholds": {
            "low": pipeline.risk_aggregator.low_threshold,
            "high": pipeline.risk_aggregator.high_threshold,
        },
    }
    with (out_dir / "metrics.json").open("w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2)

    print(json.dumps(metrics, indent=2))
    print(f"Saved model to: {model_path}")


if __name__ == "__main__":
    main()
