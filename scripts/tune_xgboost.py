from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
import time

import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.model_selection import GridSearchCV, TimeSeriesSplit
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder
from xgboost import XGBClassifier

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from fraud_pipeline.config import load_config
from fraud_pipeline.data import load_transactions, standardize_schema, temporal_split
from fraud_pipeline.features import build_tabular_features, fit_feature_spec
from fraud_pipeline.graph_features import AccountGraphBuilder, GraphFeatureConfig
from fraud_pipeline.pipeline import FraudPipeline


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Time-series GridSearchCV for XGBoost in experiment D")
    parser.add_argument("--input", default="data/transactions.csv")
    parser.add_argument("--config", default="configs/xgboost_d.json")
    parser.add_argument("--train-end", default="2025-06-30")
    parser.add_argument("--val-end", default="2025-07-31")
    parser.add_argument("--output-dir", default="artifacts/xgboost_gridsearch")
    parser.add_argument("--cv-splits", type=int, default=3)
    parser.add_argument("--jobs", type=int, default=-1)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    cfg = load_config(args.config)
    cols = cfg.columns
    data = standardize_schema(
        load_transactions(args.input), timestamp_col=cols["timestamp"], label_col=cols["label"]
    )
    splits = temporal_split(data, args.train_end, args.val_end, timestamp_col=cols["timestamp"])

    spec = fit_feature_spec(
        splits.train,
        timestamp_col=cols["timestamp"],
        label_col=cols["label"],
        account_col=cols["account_id"],
    )
    tabular = build_tabular_features(splits.train, spec)
    feature_pipeline = FraudPipeline(cfg)
    feature_pipeline.feature_spec = spec
    feature_pipeline.graph_builder = AccountGraphBuilder(
        GraphFeatureConfig(
            account_col=cols["account_id"],
            label_col=cols["label"],
            max_attribute_group_size=int(cfg.graph.get("max_attribute_group_size", 50)),
            use_louvain=bool(cfg.graph.get("use_louvain", True)),
        )
    ).fit(splits.train)
    frame, feature_cols = feature_pipeline._compose_feature_frame(tabular, splits.train)
    frame = frame.sort_values(cols["timestamp"]).reset_index(drop=True)
    y = frame[cols["label"]].astype(int)

    numeric_cols = feature_cols["numeric"]
    categorical_cols = feature_cols["categorical"]
    preprocessor = ColumnTransformer(
        transformers=[
            ("num", SimpleImputer(strategy="median"), numeric_cols),
            (
                "cat",
                Pipeline(
                    steps=[
                        ("imputer", SimpleImputer(strategy="most_frequent")),
                        ("onehot", OneHotEncoder(handle_unknown="ignore")),
                    ]
                ),
                categorical_cols,
            ),
        ],
        remainder="drop",
    )
    negative = int((y == 0).sum())
    positive = int((y == 1).sum())
    estimator = Pipeline(
        steps=[
            ("preprocessor", preprocessor),
            (
                "model",
                XGBClassifier(
                    objective="binary:logistic",
                    eval_metric="aucpr",
                    random_state=cfg.seed,
                    n_jobs=1,
                    scale_pos_weight=float(negative / max(positive, 1)),
                ),
            ),
        ]
    )
    param_grid = {
        "model__n_estimators": [200, 400, 700],
        "model__max_depth": [3, 5],
        "model__learning_rate": [0.03, 0.07],
        "model__min_child_weight": [1, 5],
        "model__subsample": [0.8, 1.0],
        "model__colsample_bytree": [0.8],
        "model__reg_lambda": [1.0, 5.0],
    }
    cv = TimeSeriesSplit(n_splits=args.cv_splits)
    search = GridSearchCV(
        estimator=estimator,
        param_grid=param_grid,
        scoring="average_precision",
        cv=cv,
        n_jobs=args.jobs,
        refit=True,
        return_train_score=True,
        error_score="raise",
        verbose=1,
    )
    started = time.perf_counter()
    search.fit(frame, y)
    elapsed = time.perf_counter() - started

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    results = pd.DataFrame(search.cv_results_).sort_values("rank_test_score")
    results.to_csv(out_dir / "cv_results.csv", index=False)
    best_params = {key.removeprefix("model__"): value for key, value in search.best_params_.items()}
    payload = {
        "scoring": "average_precision",
        "cv": {"type": "TimeSeriesSplit", "n_splits": args.cv_splits},
        "candidate_count": int(len(results)),
        "fit_count": int(len(results) * args.cv_splits),
        "elapsed_seconds": elapsed,
        "best_cv_average_precision": float(search.best_score_),
        "best_params": best_params,
        "training_rows": int(len(frame)),
        "positive_rows": positive,
        "negative_rows": negative,
    }
    (out_dir / "best_params.json").write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(json.dumps(payload, indent=2))


if __name__ == "__main__":
    main()
