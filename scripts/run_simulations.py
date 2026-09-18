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
from fraud_pipeline.simulation import (
    run_incentive_ab_test,
    simulate_concept_drift,
    simulate_mimicry_attack,
    simulate_transaction_splitting,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run drift/adversarial simulations")
    parser.add_argument("--input", required=True, help="Path to input csv/parquet")
    parser.add_argument("--pipeline", required=True, help="Path to trained pipeline.joblib")
    parser.add_argument("--config", default="configs/default.json", help="Config path (.json/.yaml)")
    parser.add_argument("--train-end", required=True, help="Train split end date (YYYY-MM-DD)")
    parser.add_argument("--val-end", required=True, help="Validation split end date (YYYY-MM-DD)")
    parser.add_argument("--output", default="artifacts/simulation_results.json", help="Output json path")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    cfg = load_config(args.config)
    cols = cfg.columns

    raw = load_transactions(args.input)
    df = standardize_schema(raw, timestamp_col=cols["timestamp"], label_col=cols["label"])
    splits = temporal_split(df, args.train_end, args.val_end, timestamp_col=cols["timestamp"])

    pipeline = FraudPipeline.load(args.pipeline)
    history = pd.concat([splits.train, splits.validation], ignore_index=True)

    base_scored = pipeline.predict(splits.test, history_df=history)
    base_eval = evaluate_predictions(
        base_scored,
        label_col=cols["label"],
        risk_aggregator=pipeline.risk_aggregator,
        timestamp_col=cols["timestamp"],
    )

    drift_df = simulate_concept_drift(splits.test)
    drift_scored = pipeline.predict(drift_df, history_df=history)
    drift_eval = evaluate_predictions(
        drift_scored,
        label_col=cols["label"],
        risk_aggregator=pipeline.risk_aggregator,
        timestamp_col=cols["timestamp"],
    )

    mimic_df = simulate_mimicry_attack(splits.test, label_col=cols["label"])
    mimic_scored = pipeline.predict(mimic_df, history_df=history)
    mimic_eval = evaluate_predictions(
        mimic_scored,
        label_col=cols["label"],
        risk_aggregator=pipeline.risk_aggregator,
        timestamp_col=cols["timestamp"],
    )

    split_df = simulate_transaction_splitting(splits.test, label_col=cols["label"])
    split_scored = pipeline.predict(split_df, history_df=history)
    split_eval = evaluate_predictions(
        split_scored,
        label_col=cols["label"],
        risk_aggregator=pipeline.risk_aggregator,
        timestamp_col=cols["timestamp"],
    )

    ab = run_incentive_ab_test(base_scored, label_col=cols["label"], band_col="risk_band")

    payload = {
        "baseline": base_eval.metrics,
        "concept_drift": drift_eval.metrics,
        "mimicry_attack": mimic_eval.metrics,
        "transaction_splitting": split_eval.metrics,
        "ab_test_medium_risk": {
            "control_fraud_rate": ab.control_fraud_rate,
            "treatment_fraud_rate": ab.treatment_fraud_rate,
            "control_dropout": ab.control_dropout,
            "treatment_dropout": ab.treatment_dropout,
            "effect_fraud_rate": ab.effect_fraud_rate,
            "effect_dropout": ab.effect_dropout,
        },
    }

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)

    print(json.dumps(payload, indent=2))


if __name__ == "__main__":
    main()
