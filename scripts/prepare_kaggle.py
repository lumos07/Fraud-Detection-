"""Convert the verified Kaggle CSV archive to chronologically sorted Parquet."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import zipfile

import polars as pl


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--archive", default="data/kaggle/financial-transactions-v1.zip")
    parser.add_argument("--output", default="data/kaggle/transactions.parquet")
    args = parser.parse_args()
    archive, output = Path(args.archive), Path(args.output)
    filename = "financial_fraud_detection_dataset.csv"
    with zipfile.ZipFile(archive) as z:
        if filename not in z.namelist():
            raise ValueError(f"Expected {filename} in the Kaggle archive")
        csv_path = archive.parent / filename
        if csv_path.exists() and csv_path.stat().st_size != z.getinfo(filename).file_size:
            raise ValueError(f"Existing extracted CSV has the wrong size: {csv_path}. Check the incomplete file before retrying.")
        # Only this verified member is extracted, never arbitrary archive paths.
        if not csv_path.exists():
            z.extract(filename, archive.parent)
    source = pl.scan_csv(csv_path, try_parse_dates=False)
    columns = source.collect_schema().names()
    required = ["transaction_id", "timestamp", "sender_account", "receiver_account", "amount", "is_fraud"]
    if not set(required).issubset(columns):
        raise ValueError(f"Missing fields: {set(required) - set(columns)}")
    optional = [c for c in ["transaction_type", "merchant_category", "location", "device_used", "device_hash", "merchant_id", "payment_channel", "label_available_at"] if c in columns]
    source = source.select(required + optional).with_columns(pl.col("timestamp").str.to_datetime(time_zone="UTC"))
    output.parent.mkdir(parents=True, exist_ok=True)
    source.sort(["timestamp", "transaction_id"]).sink_parquet(output, compression="zstd")
    summary = pl.scan_parquet(output).select(pl.len().alias("rows"), pl.col("is_fraud").sum().alias("frauds"), pl.col("timestamp").min().alias("min_time"), pl.col("timestamp").max().alias("max_time"), pl.col("transaction_id").n_unique().alias("unique_transactions")).collect().to_dicts()[0]
    if summary["rows"] != summary["unique_transactions"]:
        raise ValueError("Duplicate transaction IDs")
    with archive.open("rb") as f:
        digest = hashlib.file_digest(f, "sha256").hexdigest()
    manifest = {"source": "https://www.kaggle.com/datasets/aryan208/financial-transactions-dataset-for-fraud-detection", "version": 1, "synthetic": True, "archive_sha256": digest, "columns": columns, "selected_columns": required + optional, "timezone_assumption": "naive timestamps treated as UTC", **summary}
    output.with_suffix(".manifest.json").write_text(json.dumps(manifest, default=str, indent=2))
    print(json.dumps(manifest, default=str, indent=2))


if __name__ == "__main__":
    main()
