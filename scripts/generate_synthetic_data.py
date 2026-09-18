from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate synthetic fraud transactions")
    parser.add_argument("--output", default="data/transactions.parquet")
    parser.add_argument("--rows", type=int, default=50000)
    parser.add_argument("--fraud-rate", type=float, default=0.01)
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    rng = np.random.default_rng(args.seed)

    n = args.rows
    account_count = max(500, n // 20)
    accounts = [f"A{i:06d}" for i in range(account_count)]
    devices = [f"D{i:05d}" for i in range(max(200, account_count // 2))]
    ips = [f"10.0.{i // 256}.{i % 256}" for i in range(max(500, account_count // 3))]

    timestamps = pd.date_range(
        "2025-01-01", "2025-08-31", periods=n, tz="UTC"
    )
    base_amount = rng.lognormal(mean=3.2, sigma=1.0, size=n)
    labels = (rng.random(n) < args.fraud_rate).astype(int)

    # Make fraud values more extreme.
    base_amount = base_amount * (1.0 + labels * rng.uniform(2.0, 8.0, size=n))

    account_ids = rng.choice(accounts, size=n, replace=True)
    counterparty = rng.choice(accounts, size=n, replace=True)
    merchant = rng.choice(
        ["retail", "electronics", "travel", "crypto", "gaming", "utilities"],
        size=n,
        p=[0.25, 0.18, 0.12, 0.08, 0.15, 0.22],
    )
    channel = rng.choice(["card", "web", "transfer", "wire"], size=n, p=[0.45, 0.25, 0.2, 0.1])

    device_id = rng.choice(devices, size=n, replace=True)
    ip_address = rng.choice(ips, size=n, replace=True)
    balance = rng.lognormal(mean=7.2, sigma=0.8, size=n)

    # Fraud clusters share devices/ips to create graph patterns.
    fraud_idx = np.where(labels == 1)[0]
    if len(fraud_idx) > 0:
        shared_devices = rng.choice(devices, size=max(5, len(fraud_idx) // 100), replace=False)
        shared_ips = rng.choice(ips, size=max(10, len(fraud_idx) // 50), replace=False)
        device_id[fraud_idx] = rng.choice(shared_devices, size=len(fraud_idx), replace=True)
        ip_address[fraud_idx] = rng.choice(shared_ips, size=len(fraud_idx), replace=True)
        channel[fraud_idx] = rng.choice(["wire", "transfer", "web"], size=len(fraud_idx), p=[0.4, 0.4, 0.2])

    account_created_at = pd.to_datetime(timestamps) - pd.to_timedelta(
        rng.integers(10, 3650, size=n), unit="D"
    )

    df = pd.DataFrame(
        {
            "transaction_id": [f"T{i:09d}" for i in range(n)],
            "account_id": account_ids,
            "counterparty_account_id": counterparty,
            "timestamp": timestamps,
            "amount": base_amount.round(2),
            "merchant_category": merchant,
            "channel": channel,
            "device_id": device_id,
            "ip_address": ip_address,
            "phone": [f"+1-555-{x:04d}" for x in rng.integers(0, 10000, size=n)],
            "email": [f"user{x}@example.com" for x in rng.integers(0, 100000, size=n)],
            "balance": balance.round(2),
            "account_created_at": account_created_at,
            "label": labels,
        }
    )

    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    if output.suffix.lower() == ".csv":
        df.to_csv(output, index=False)
    else:
        df.to_parquet(output, index=False)
    print(f"Wrote {len(df)} rows to {output}")


if __name__ == "__main__":
    main()
