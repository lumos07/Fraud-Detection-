"""Vectorized behavioral features. Every historical window excludes timestamp t."""
from __future__ import annotations

import numpy as np
import pandas as pd
import polars as pl

from fraud_pipeline.features import FeatureOutput
from fraud_pipeline.kaggle_data import CATEGORICAL


def build_temporal_features(df: pd.DataFrame) -> FeatureOutput:
    source = df.sort_values(["timestamp", "transaction_id"], kind="stable").reset_index(drop=True)
    required = ["transaction_id", "timestamp", "account_id", "counterparty_account_id", "amount"]
    data = pl.from_pandas(source[required + [c for c in CATEGORICAL if c in source]])
    data = data.with_row_index("_row").with_columns(pl.col("timestamp").cast(pl.Datetime("us", "UTC")))
    numeric: dict[str, np.ndarray] = {
        "amount": source.amount.to_numpy(),
        "hour": source.timestamp.dt.hour.to_numpy(),
        "is_weekend": (source.timestamp.dt.weekday >= 5).astype(int).to_numpy(),
    }

    def rolling(keys: list[str], period: str, expressions: list[pl.Expr]) -> None:
        ordered = data.sort(keys + ["timestamp", "_row"])
        result = ordered.rolling("timestamp", period=period, group_by=keys, closed="left").agg(expressions)
        # Rolling produces one output per input, including duplicate timestamps.
        rows = ordered["_row"].to_numpy()
        for name in result.columns:
            if name in keys or name == "timestamp":
                continue
            arr = np.zeros(len(source), dtype=np.float32)
            arr[rows] = result[name].fill_null(0).to_numpy()
            numeric[name] = arr

    for window, suffix in [("1h", "1h"), ("1d", "24h"), ("7d", "7d")]:
        expressions = [pl.len().alias(f"count_tx_last_{suffix}")]
        if suffix == "24h":
            expressions.append(pl.col("amount").mean().alias("avg_amt_1d"))
        if suffix == "7d":
            expressions += [pl.col("amount").std().alias("std_amt_7d"), pl.col("counterparty_account_id").n_unique().alias("unique_receivers_7d")]
        rolling(["account_id"], window, expressions)
    rolling(["account_id"], "30d", [pl.col("amount").mean().alias("mean_amount_30d"), pl.col("amount").median().alias("median_amount_30d")])
    rolling(["counterparty_account_id"], "1d", [pl.len().alias("receiver_frequency_24h")])

    def prior_summary(keys: list[str], prefix: str) -> None:
        events = data.group_by(keys + ["timestamp"]).agg(pl.len().alias("_count")).sort(keys + ["timestamp"])
        events = events.with_columns(pl.col("_count").cum_sum().over(keys).alias(prefix + "_count"), pl.col("timestamp").alias("_previous"))
        joined = data.select("_row", "timestamp", *keys).sort("timestamp").join_asof(
            events.drop("_count").sort("timestamp"), on="timestamp", by=keys,
            strategy="backward", allow_exact_matches=False, check_sortedness=False,
        ).sort("_row")
        numeric[prefix + "_count"] = joined[prefix + "_count"].fill_null(0).to_numpy()
        numeric[prefix + "_seconds_since"] = joined.select((pl.col("timestamp") - pl.col("_previous")).dt.total_seconds().fill_null(-1))["timestamp"].to_numpy()

    prior_summary(["account_id"], "sender_lifetime")
    prior_summary(["account_id", "counterparty_account_id"], "interaction")
    numeric["new_receiver_indicator"] = (numeric["interaction_count"] == 0).astype(float)
    for col in ["transaction_type", "location", "device_used"]:
        if col not in source:
            continue
        prior_summary(["account_id", col], col)
        denom = numeric["sender_lifetime_count"]
        numeric[col + "_deviation"] = np.where(denom > 0, 1.0 - numeric[col + "_count"] / np.maximum(denom, 1), 0.0)
        del numeric[col + "_count"], numeric[col + "_seconds_since"]
    for kind in ["mean", "median"]:
        historical = numeric[kind + "_amount_30d"]
        numeric["amount_over_" + kind + "_30d"] = np.divide(numeric["amount"], historical, out=np.zeros(len(source)), where=historical > 0)
    out = source.copy()
    for col, values in numeric.items():
        out[col] = np.nan_to_num(values, nan=0, posinf=0, neginf=0).astype(np.float32)
    return FeatureOutput(out, list(numeric), [c for c in CATEGORICAL if c in source], ["transaction_id", "account_id", "timestamp", "label"])
