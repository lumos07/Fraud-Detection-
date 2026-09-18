from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import pandas as pd


@dataclass
class DatasetSplits:
    train: pd.DataFrame
    validation: pd.DataFrame
    test: pd.DataFrame


def load_transactions(path: str | Path) -> pd.DataFrame:
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Input data not found: {path}")

    if path.suffix.lower() in {".parquet", ".pq"}:
        df = pd.read_parquet(path)
    elif path.suffix.lower() in {".csv"}:
        df = pd.read_csv(path)
    else:
        raise ValueError(f"Unsupported file format: {path.suffix}")
    return df


def standardize_schema(
    df: pd.DataFrame,
    timestamp_col: str = "timestamp",
    label_col: str = "label",
) -> pd.DataFrame:
    out = df.copy()
    if timestamp_col not in out.columns:
        raise ValueError(f"Missing timestamp column: {timestamp_col}")

    raw_timestamp = out[timestamp_col].copy()
    out[timestamp_col] = pd.to_datetime(out[timestamp_col], utc=True, errors="coerce")
    if out[timestamp_col].isna().any():
        # Mixed datetime precision (e.g., some rows with nanoseconds) needs format="mixed".
        out[timestamp_col] = pd.to_datetime(
            raw_timestamp.astype(str), utc=True, errors="coerce", format="mixed"
        )
    if out[timestamp_col].isna().any():
        bad = int(out[timestamp_col].isna().sum())
        raise ValueError(f"Found {bad} invalid timestamps in {timestamp_col}")

    if "amount" not in out.columns:
        raise ValueError("Missing required column: amount")
    out["amount"] = pd.to_numeric(out["amount"], errors="coerce").fillna(0.0)

    if label_col not in out.columns:
        out[label_col] = 0
    out[label_col] = pd.to_numeric(out[label_col], errors="coerce").fillna(0).astype(int)

    if "account_created_at" in out.columns:
        raw_created = out["account_created_at"].copy()
        out["account_created_at"] = pd.to_datetime(
            out["account_created_at"], utc=True, errors="coerce"
        )
        if out["account_created_at"].isna().any():
            out["account_created_at"] = pd.to_datetime(
                raw_created.astype(str),
                utc=True,
                errors="coerce",
                format="mixed",
            )

    if "balance" in out.columns:
        out["balance"] = pd.to_numeric(out["balance"], errors="coerce")

    for col in ["account_id", "transaction_id", "counterparty_account_id", "device_id", "ip_address", "phone", "email"]:
        if col in out.columns:
            out[col] = out[col].astype(str)

    out = out.sort_values(timestamp_col).reset_index(drop=True)
    return out


def temporal_split(
    df: pd.DataFrame,
    train_end: str,
    validation_end: str,
    timestamp_col: str = "timestamp",
) -> DatasetSplits:
    train_end_ts = pd.Timestamp(train_end, tz="UTC")
    val_end_ts = pd.Timestamp(validation_end, tz="UTC")
    if val_end_ts <= train_end_ts:
        raise ValueError("validation_end must be strictly after train_end")

    train = df[df[timestamp_col] <= train_end_ts].copy()
    validation = df[(df[timestamp_col] > train_end_ts) & (df[timestamp_col] <= val_end_ts)].copy()
    test = df[df[timestamp_col] > val_end_ts].copy()

    if train.empty or validation.empty or test.empty:
        raise ValueError(
            "Temporal split produced an empty partition. "
            "Adjust split boundaries or verify timestamp data."
        )
    return DatasetSplits(train=train, validation=validation, test=test)
