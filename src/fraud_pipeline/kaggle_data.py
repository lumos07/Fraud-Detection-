"""Explicit schema adapter for Kaggle's financial transactions (version 1)."""
from __future__ import annotations

import pandas as pd

ALIASES = {
    "sender_account": "account_id",
    "receiver_account": "counterparty_account_id",
    "is_fraud": "label",
    "device_hash": "device_id",
}
CATEGORICAL = ["transaction_type", "merchant_category", "location", "device_used", "payment_channel"]
REQUIRED = ["transaction_id", "timestamp", "account_id", "counterparty_account_id", "amount"]


def adapt_transactions(df: pd.DataFrame, *, require_labels: bool = False) -> pd.DataFrame:
    out = df.copy()
    for source, target in ALIASES.items():
        if source in out:
            if target in out:
                raise ValueError(f"Ambiguous schema: both {source} and {target} were supplied")
            out = out.rename(columns={source: target})
    missing = set(REQUIRED) - set(out.columns)
    if missing:
        raise ValueError(f"Missing required transaction fields: {sorted(missing)}")
    for col in ["transaction_id", "account_id", "counterparty_account_id"]:
        if out[col].isna().any() or out[col].astype(str).str.strip().eq("").any():
            raise ValueError(f"Missing identifiers in {col}")
        out[col] = out[col].astype(str)
    if not out.transaction_id.is_unique:
        raise ValueError("transaction_id must be unique")
    out["timestamp"] = pd.to_datetime(out.timestamp, utc=True, format="mixed", errors="raise")
    if out.timestamp.isna().any():
        raise ValueError("Missing timestamps")
    out["amount"] = pd.to_numeric(out.amount, errors="raise")
    if out.amount.isna().any() or not out.amount.between(0, float("inf"), inclusive="left").all():
        raise ValueError("Amounts must be finite and nonnegative")
    if "label" not in out:
        if require_labels:
            raise ValueError("Training/evaluation requires is_fraud (or label); labels are never invented")
        out["label"] = float("nan")
    else:
        raw = out.label.astype("string").str.lower()
        labels = raw.replace({"true": "1", "false": "0"})
        out["label"] = pd.to_numeric(labels, errors="raise").astype(float)
        if not out.label.dropna().isin([0, 1]).all() or (require_labels and out.label.isna().any()):
            raise ValueError("Labels must be binary and known for training/evaluation")
    if "label_available_at" in out:
        out["label_available_at"] = pd.to_datetime(out.label_available_at, utc=True, format="mixed")
        if (out.label_available_at < out.timestamp).any():
            raise ValueError("A label cannot be known before its transaction")
    # Never use fraud_type or the dataset's opaque pre-engineered risk scores.
    keep = REQUIRED + ["label"] + [c for c in CATEGORICAL + ["device_id", "merchant_id", "label_available_at"] if c in out]
    out = out[keep].sort_values(["timestamp", "transaction_id"], kind="stable").reset_index(drop=True)
    for col in CATEGORICAL:
        if col in out:
            out[col] = out[col].fillna("unknown").astype("category")
    return out


def mature_labels(df: pd.DataFrame, cutoff: pd.Timestamp, delay: str) -> pd.Series:
    available = df["label_available_at"] if "label_available_at" in df else df.timestamp + pd.Timedelta(delay)
    return df.label.notna() & available.notna() & (available < cutoff) & (df.timestamp < cutoff)
