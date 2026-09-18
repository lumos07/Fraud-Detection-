from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd


@dataclass
class RiskConfig:
    weight_supervised: float = 0.5
    weight_anomaly: float = 0.2
    weight_graph: float = 0.3
    c_fn: float = 604000.0
    c_fp: float = 75.0
    c_review: float = 8.0
    review_catch_rate: float = 0.7
    low_threshold: float = 0.2
    high_threshold: float = 0.7
    low_points: int = 25
    high_points: int = 25


class RiskAggregator:
    def __init__(self, config: RiskConfig):
        self.config = config
        self.scale_params: dict[str, tuple[float, float]] = {}
        self.low_threshold = config.low_threshold
        self.high_threshold = config.high_threshold

    def fit(
        self,
        score_frame: pd.DataFrame,
        y_true: pd.Series,
    ) -> "RiskAggregator":
        needed = ["s_supervised", "s_anomaly", "s_graph"]
        missing = [c for c in needed if c not in score_frame.columns]
        if missing:
            raise ValueError(f"Missing score columns: {missing}")

        scaled = self._scale_frame(score_frame, fit=True)
        final_score = self._weighted_sum(scaled)
        low, high = self._tune_thresholds(final_score, y_true.to_numpy(dtype=int))
        self.low_threshold = float(low)
        self.high_threshold = float(high)
        return self

    def predict(
        self,
        score_frame: pd.DataFrame,
    ) -> pd.DataFrame:
        scaled = self._scale_frame(score_frame, fit=False)
        final_score = self._weighted_sum(scaled)
        bands = np.where(
            final_score < self.low_threshold,
            "low",
            np.where(final_score < self.high_threshold, "medium", "high"),
        )
        return pd.DataFrame(
            {
                "s_supervised_norm": scaled["s_supervised"],
                "s_anomaly_norm": scaled["s_anomaly"],
                "s_graph_norm": scaled["s_graph"],
                "final_score": final_score,
                "risk_band": bands,
            }
        )

    def expected_loss(self, bands: np.ndarray, y_true: np.ndarray) -> float:
        c_fn = self.config.c_fn
        c_fp = self.config.c_fp
        c_review = self.config.c_review
        review_catch_rate = self.config.review_catch_rate

        y = y_true.astype(int)
        loss = np.zeros_like(y, dtype=float)

        low_mask = bands == "low"
        medium_mask = bands == "medium"
        high_mask = bands == "high"

        # Low band: missed fraud pays full FN cost.
        loss[low_mask & (y == 1)] = c_fn

        # Medium band: review cost always paid; fraud can still slip.
        loss[medium_mask & (y == 0)] = c_review
        loss[medium_mask & (y == 1)] = c_review + c_fn * (1.0 - review_catch_rate)

        # High band: block cost for legitimate users.
        loss[high_mask & (y == 0)] = c_fp
        return float(loss.sum())

    def _scale_frame(self, frame: pd.DataFrame, fit: bool) -> pd.DataFrame:
        out = pd.DataFrame(index=frame.index)
        for col in ["s_supervised", "s_anomaly", "s_graph"]:
            values = frame[col].to_numpy(dtype=float)
            if fit:
                lo = float(np.nanpercentile(values, 1))
                hi = float(np.nanpercentile(values, 99))
                if hi <= lo:
                    hi = lo + 1e-6
                self.scale_params[col] = (lo, hi)
            lo, hi = self.scale_params[col]
            scaled = (values - lo) / (hi - lo)
            out[col] = np.clip(scaled, 0.0, 1.0)
        return out

    def _weighted_sum(self, scaled_frame: pd.DataFrame) -> np.ndarray:
        w = self.config
        total = (
            w.weight_supervised * scaled_frame["s_supervised"].to_numpy()
            + w.weight_anomaly * scaled_frame["s_anomaly"].to_numpy()
            + w.weight_graph * scaled_frame["s_graph"].to_numpy()
        )
        denom = w.weight_supervised + w.weight_anomaly + w.weight_graph
        if denom <= 0:
            denom = 1.0
        return total / denom

    def _tune_thresholds(self, final_score: np.ndarray, y_true: np.ndarray) -> tuple[float, float]:
        low_grid = np.linspace(0.05, 0.65, int(self.config.low_points))
        high_grid = np.linspace(0.35, 0.95, int(self.config.high_points))
        best = (self.low_threshold, self.high_threshold)
        best_loss = np.inf

        for low in low_grid:
            for high in high_grid:
                if high <= low:
                    continue
                bands = np.where(
                    final_score < low,
                    "low",
                    np.where(final_score < high, "medium", "high"),
                )
                loss = self.expected_loss(bands, y_true)
                if loss < best_loss:
                    best_loss = loss
                    best = (float(low), float(high))
        return best
