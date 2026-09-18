from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd


def simulate_concept_drift(
    df: pd.DataFrame,
    channel_col: str = "channel",
    cross_border_label: str = "cross_border",
    amount_multiplier: float = 1.3,
    cross_border_boost: float = 2.0,
    random_state: int = 42,
) -> pd.DataFrame:
    rng = np.random.default_rng(random_state)
    out = df.copy()

    if "amount" in out.columns:
        out["amount"] = out["amount"] * amount_multiplier

    if cross_border_label in out.columns:
        mask = out[cross_border_label].astype(int) == 1
        extra = rng.random(len(out)) < (cross_border_boost - 1.0) * mask.astype(float).clip(0, 1)
        out.loc[extra, "amount"] = out.loc[extra, "amount"] * 1.1
    elif channel_col in out.columns:
        channels = out[channel_col].astype(str)
        mask = channels.str.contains("international|wire|cross", case=False, regex=True)
        out.loc[mask, "amount"] = out.loc[mask, "amount"] * cross_border_boost

    return out


def simulate_mimicry_attack(
    df: pd.DataFrame,
    label_col: str = "label",
    random_state: int = 42,
) -> pd.DataFrame:
    rng = np.random.default_rng(random_state)
    out = df.copy()
    if label_col not in out.columns:
        return out

    fraud = out[out[label_col] == 1].copy()
    legit = out[out[label_col] == 0].copy()
    if fraud.empty or legit.empty:
        return out

    legit_amount_mean = float(legit["amount"].mean())
    legit_amount_std = float(max(legit["amount"].std(), 1e-6))

    idx = fraud.index.to_numpy()
    out.loc[idx, "amount"] = rng.normal(legit_amount_mean, legit_amount_std, size=len(idx)).clip(min=1.0)

    if "hour" in out.columns:
        legit_hour_dist = legit["hour"].value_counts(normalize=True).sort_index()
        sampled_hours = rng.choice(
            legit_hour_dist.index.to_numpy(),
            size=len(idx),
            p=legit_hour_dist.to_numpy(),
        )
        out.loc[idx, "hour"] = sampled_hours
    return out


def simulate_transaction_splitting(
    df: pd.DataFrame,
    label_col: str = "label",
    split_factor: int = 4,
    quantile_threshold: float = 0.99,
) -> pd.DataFrame:
    if split_factor < 2:
        return df.copy()

    out = df.copy()
    if "amount" not in out.columns or label_col not in out.columns:
        return out

    threshold = float(out["amount"].quantile(quantile_threshold))
    candidates = out[(out[label_col] == 1) & (out["amount"] >= threshold)]
    if candidates.empty:
        return out

    pieces = []
    for row in candidates.itertuples(index=False):
        row_dict = row._asdict()
        amount = float(row_dict["amount"])
        for i in range(split_factor):
            new_row = row_dict.copy()
            new_row["amount"] = amount / split_factor
            if "transaction_id" in new_row:
                new_row["transaction_id"] = f"{new_row['transaction_id']}_split_{i+1}"
            pieces.append(new_row)

    split_df = pd.DataFrame(pieces)
    out = out.drop(index=candidates.index)
    out = pd.concat([out, split_df], ignore_index=True)
    return out


@dataclass
class ABResult:
    control_fraud_rate: float
    treatment_fraud_rate: float
    control_dropout: float
    treatment_dropout: float
    effect_fraud_rate: float
    effect_dropout: float


def run_incentive_ab_test(
    scored_df: pd.DataFrame,
    label_col: str = "label",
    band_col: str = "risk_band",
    random_state: int = 42,
) -> ABResult:
    rng = np.random.default_rng(random_state)
    data = scored_df.copy()
    if band_col not in data.columns or label_col not in data.columns:
        raise ValueError("scored_df must include risk_band and label columns")

    medium = data[data[band_col] == "medium"].copy()
    if medium.empty and "final_score" in data.columns:
        lo = float(data["final_score"].quantile(0.4))
        hi = float(data["final_score"].quantile(0.8))
        medium = data[(data["final_score"] >= lo) & (data["final_score"] <= hi)].copy()
    if medium.empty:
        return ABResult(0.0, 0.0, 0.0, 0.0, 0.0, 0.0)

    assign = rng.integers(0, 2, size=len(medium))
    medium["group"] = np.where(assign == 0, "control", "treatment")

    # Treatment assumption: more fraud blocked, slightly more user friction.
    medium["fraud_success"] = medium[label_col].astype(int)
    treatment_mask = medium["group"] == "treatment"
    treated_fraud_idx = medium[treatment_mask & (medium[label_col] == 1)].index
    reduce = int(len(treated_fraud_idx) * 0.25)
    if reduce > 0:
        blocked_idx = rng.choice(treated_fraud_idx.to_numpy(), size=reduce, replace=False)
        medium.loc[blocked_idx, "fraud_success"] = 0

    medium["dropout"] = 0
    treatment_any = medium[treatment_mask].index
    if len(treatment_any) > 0:
        drop_n = int(len(treatment_any) * 0.03)
        if drop_n > 0:
            drop_idx = rng.choice(treatment_any.to_numpy(), size=drop_n, replace=False)
            medium.loc[drop_idx, "dropout"] = 1

    control = medium[medium["group"] == "control"]
    treatment = medium[medium["group"] == "treatment"]

    c_fraud = float(control["fraud_success"].mean()) if not control.empty else 0.0
    t_fraud = float(treatment["fraud_success"].mean()) if not treatment.empty else 0.0
    c_dropout = float(control["dropout"].mean()) if not control.empty else 0.0
    t_dropout = float(treatment["dropout"].mean()) if not treatment.empty else 0.0

    return ABResult(
        control_fraud_rate=c_fraud,
        treatment_fraud_rate=t_fraud,
        control_dropout=c_dropout,
        treatment_dropout=t_dropout,
        effect_fraud_rate=t_fraud - c_fraud,
        effect_dropout=t_dropout - c_dropout,
    )
