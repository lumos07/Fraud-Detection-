from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

import numpy as np
import pandas as pd


@dataclass
class FeatureSpec:
    large_tx_threshold: float
    timestamp_col: str
    label_col: str
    account_col: str


@dataclass
class FeatureOutput:
    frame: pd.DataFrame
    numeric_cols: list[str]
    categorical_cols: list[str]
    id_cols: list[str]


def fit_feature_spec(
    train_df: pd.DataFrame,
    timestamp_col: str = "timestamp",
    label_col: str = "label",
    account_col: str = "account_id",
    large_tx_quantile: float = 0.95,
) -> FeatureSpec:
    threshold = float(train_df["amount"].quantile(large_tx_quantile))
    return FeatureSpec(
        large_tx_threshold=threshold,
        timestamp_col=timestamp_col,
        label_col=label_col,
        account_col=account_col,
    )


def _safe_div(numerator: pd.Series, denominator: pd.Series | float) -> pd.Series:
    if not isinstance(denominator, pd.Series):
        denominator = pd.Series(denominator, index=numerator.index)
    den = denominator.replace(0, np.nan)
    return (numerator / den).replace([np.inf, -np.inf], np.nan).fillna(0.0)


def _rolling_group_metric(
    df: pd.DataFrame,
    group_col: str,
    timestamp_col: str,
    value_col: str,
    window: str,
    agg: str,
) -> pd.Series:
    work = df[[group_col, timestamp_col, value_col]].copy()
    work["_row_id"] = np.arange(len(work))
    work = work.sort_values([group_col, timestamp_col, "_row_id"]).reset_index(drop=True)
    result = np.full(len(df), np.nan, dtype=float)

    grouped = work.groupby(group_col, sort=False)
    for _, group in grouped:
        rolled_values = (
            group.set_index(timestamp_col)[value_col]
            .rolling(window)
            .agg(agg)
            .to_numpy(dtype=float)
        )
        row_ids = group["_row_id"].to_numpy(dtype=int)
        result[row_ids] = rolled_values

    return pd.Series(result, index=np.arange(len(df)), dtype=float)


def _historical_device_score(
    df: pd.DataFrame,
    device_col: str,
    account_col: str,
    timestamp_col: str,
) -> pd.Series:
    """Score device sharing using information available at each event time.

    A row sees accounts observed for its device at strictly earlier timestamps,
    plus its own account. Rows at the same timestamp do not see one another, so
    the result is independent of their input ordering.
    """
    work = df[[device_col, account_col, timestamp_col]].copy()
    work["_row_id"] = np.arange(len(work))
    work = work.sort_values([timestamp_col, "_row_id"]).reset_index(drop=True)
    result = np.ones(len(df), dtype=float)
    accounts_by_device: dict[str, set[str]] = {}

    for _, timestamp_group in work.groupby(timestamp_col, sort=False):
        # Calculate every score before updating state for this timestamp.
        for device_value, account_value, _, row_id_value in timestamp_group.itertuples(
            index=False, name=None
        ):
            device = str(device_value)
            account = str(account_value)
            row_id = int(row_id_value)
            prior_accounts = accounts_by_device.get(device, set())
            result[row_id] = 1.0 / float(len(prior_accounts | {account}))

        for device_value, account_value, _, _ in timestamp_group.itertuples(
            index=False, name=None
        ):
            device = str(device_value)
            account = str(account_value)
            accounts_by_device.setdefault(device, set()).add(account)

    return pd.Series(result, index=df.index, dtype=float)


def build_tabular_features(df: pd.DataFrame, spec: FeatureSpec) -> FeatureOutput:
    work = df.copy()
    ts_col = spec.timestamp_col
    label_col = spec.label_col
    account_col = spec.account_col

    work = work.sort_values([account_col, ts_col]).reset_index(drop=True)

    work["hour"] = work[ts_col].dt.hour.astype(int)
    work["is_weekend"] = (work[ts_col].dt.weekday >= 5).astype(int)
    work["transactions_count_lifetime"] = work.groupby(account_col).cumcount() + 1

    if "account_created_at" in work.columns:
        work["account_age_days"] = (
            (work[ts_col] - work["account_created_at"]).dt.total_seconds() / 86400.0
        ).clip(lower=0).fillna(0.0)
    else:
        first_seen = work.groupby(account_col)[ts_col].transform("min")
        work["account_age_days"] = (
            (work[ts_col] - first_seen).dt.total_seconds() / 86400.0
        ).clip(lower=0).fillna(0.0)

    work["median_amount_30d"] = _rolling_group_metric(
        work, account_col, ts_col, "amount", "30D", "median"
    )
    work["avg_amt_1d"] = _rolling_group_metric(
        work, account_col, ts_col, "amount", "1D", "mean"
    )
    work["std_amt_7d"] = _rolling_group_metric(
        work, account_col, ts_col, "amount", "7D", "std"
    ).fillna(0.0)
    work["avg_amt_30d"] = _rolling_group_metric(
        work, account_col, ts_col, "amount", "30D", "mean"
    )
    work["avg_amt_30d_prev"] = work.groupby(account_col)["avg_amt_30d"].shift(1)
    work["pct_change_avg_amt_30d"] = _safe_div(
        work["avg_amt_30d"] - work["avg_amt_30d_prev"], work["avg_amt_30d_prev"]
    )

    work["count_tx_last_1h"] = _rolling_group_metric(
        work, account_col, ts_col, "amount", "1h", "count"
    )
    work["count_tx_last_24h"] = _rolling_group_metric(
        work, account_col, ts_col, "amount", "1D", "count"
    )
    work["count_tx_last_7d"] = _rolling_group_metric(
        work, account_col, ts_col, "amount", "7D", "count"
    )

    work["amount_over_median_30d"] = _safe_div(work["amount"], work["median_amount_30d"])

    if "balance" in work.columns:
        work["avg_balance"] = (
            work.groupby(account_col)["balance"]
            .transform(lambda s: s.ffill().expanding().mean())
            .fillna(0.0)
        )
    else:
        work["avg_balance"] = work["avg_amt_30d"]
    work["amount_over_avg_balance"] = _safe_div(work["amount"], work["avg_balance"])

    work["is_large_tx"] = (work["amount"] > spec.large_tx_threshold).astype(int)
    work["large_tx_count_30d"] = _rolling_group_metric(
        work, account_col, ts_col, "is_large_tx", "30D", "sum"
    )
    work["tx_count_30d"] = _rolling_group_metric(
        work, account_col, ts_col, "amount", "30D", "count"
    )
    work["ratio_large_tx_count"] = _safe_div(work["large_tx_count_30d"], work["tx_count_30d"])

    if "device_id" in work.columns:
        work["device_score"] = _historical_device_score(
            work,
            device_col="device_id",
            account_col=account_col,
            timestamp_col=ts_col,
        )
    else:
        work["device_score"] = 1.0

    if "merchant_category" not in work.columns:
        work["merchant_category"] = "unknown"
    if "channel" not in work.columns:
        work["channel"] = "unknown"

    work["merchant_category"] = work["merchant_category"].fillna("unknown").astype(str)
    work["channel"] = work["channel"].fillna("unknown").astype(str)

    drop_helpers: Iterable[str] = [
        "avg_amt_30d_prev",
        "is_large_tx",
    ]
    work = work.drop(columns=[c for c in drop_helpers if c in work.columns])
    work = work.replace([np.inf, -np.inf], np.nan)
    num_cols = work.select_dtypes(include=[np.number]).columns.tolist()
    if num_cols:
        work[num_cols] = work[num_cols].fillna(0.0)
    obj_cols = work.select_dtypes(include=["object"]).columns.tolist()
    if obj_cols:
        work[obj_cols] = work[obj_cols].fillna("unknown")

    numeric_cols = [
        "amount",
        "amount_over_median_30d",
        "amount_over_avg_balance",
        "hour",
        "is_weekend",
        "device_score",
        "count_tx_last_1h",
        "count_tx_last_24h",
        "count_tx_last_7d",
        "transactions_count_lifetime",
        "account_age_days",
        "avg_amt_1d",
        "std_amt_7d",
        "pct_change_avg_amt_30d",
        "ratio_large_tx_count",
    ]
    numeric_cols = [c for c in numeric_cols if c in work.columns]
    categorical_cols = [c for c in ["merchant_category", "channel"] if c in work.columns]

    id_cols = [c for c in ["transaction_id", account_col, ts_col, label_col] if c in work.columns]
    return FeatureOutput(
        frame=work,
        numeric_cols=numeric_cols,
        categorical_cols=categorical_cols,
        id_cols=id_cols,
    )
