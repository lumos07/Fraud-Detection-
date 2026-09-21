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
    tune_weights: bool = False
    weight_grid_step: float = 0.25
    max_review_rate: float | None = None
    fn_cost_mode: str = "constant"
    amount_loss_multiplier: float = 1.0
    full_threshold_range: bool = False

    def __post_init__(self):
        weights = np.array([self.weight_supervised, self.weight_anomaly, self.weight_graph])
        if not np.isfinite(weights).all() or (weights < 0).any() or weights.sum() <= 0:
            raise ValueError("Risk weights must be finite, nonnegative and have a positive sum")
        costs = [self.c_fn, self.c_fp, self.c_review, self.amount_loss_multiplier]
        if not all(np.isfinite(c) and c >= 0 for c in costs):
            raise ValueError("Cost assumptions must be finite and nonnegative")
        if not 0 <= self.review_catch_rate <= 1:
            raise ValueError("review_catch_rate must be between 0 and 1")
        if self.max_review_rate is not None and not 0 <= self.max_review_rate <= 1:
            raise ValueError("max_review_rate must be between 0 and 1")
        if min(self.low_points, self.high_points) < 2:
            raise ValueError("Threshold grids need at least two points")


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
        amounts: np.ndarray | None = None,
    ) -> "RiskAggregator":
        needed = ["s_supervised", "s_anomaly", "s_graph"]
        missing = [c for c in needed if c not in score_frame.columns]
        if missing:
            raise ValueError(f"Missing score columns: {missing}")

        scaled = self._scale_frame(score_frame, fit=True)
        final_score = self._weighted_sum(scaled)
        y = y_true.to_numpy(dtype=int)
        low, high = self._tune_thresholds(final_score, y, amounts)
        if self.config.tune_weights:
            step = self.config.weight_grid_step
            divisions = int(round(1 / step)) if step > 0 else 0
            if divisions < 1 or not np.isclose(divisions * step, 1):
                raise ValueError("weight_grid_step must divide 1 exactly")
            initial = np.array([self.config.weight_supervised, self.config.weight_anomaly, self.config.weight_graph])
            active = initial > 0
            bands = np.where(final_score < low, "low", np.where(final_score < high, "medium", "high"))
            best_loss = self.expected_loss(bands, y, amounts)
            best_weights = initial.copy()
            matrix = scaled[["s_supervised", "s_anomaly", "s_graph"]].to_numpy()
            for a in range(divisions + 1):
                for b in range(divisions + 1 - a):
                    weights = np.array([a, b, divisions - a - b], dtype=float) / divisions
                    if np.any(weights[~active] > 0):
                        continue
                    scores = matrix @ weights
                    lo, hi = self._tune_thresholds(scores, y, amounts)
                    bands = np.where(scores < lo, "low", np.where(scores < hi, "medium", "high"))
                    cost = self.expected_loss(bands, y, amounts)
                    if cost < best_loss:
                        best_loss, low, high, best_weights = cost, lo, hi, weights
            self.config.weight_supervised, self.config.weight_anomaly, self.config.weight_graph = map(float, best_weights)
        self.low_threshold = float(low)
        self.high_threshold = float(high)
        self.config.low_threshold = self.low_threshold
        self.config.high_threshold = self.high_threshold
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

    def expected_loss(self, bands: np.ndarray, y_true: np.ndarray, amounts: np.ndarray | None = None) -> float:
        c_fn = self._fn_costs(len(y_true), amounts)
        c_fp = self.config.c_fp
        c_review = self.config.c_review
        review_catch_rate = self.config.review_catch_rate

        y = y_true.astype(int)
        loss = np.zeros_like(y, dtype=float)

        low_mask = bands == "low"
        medium_mask = bands == "medium"
        high_mask = bands == "high"

        # Low band: missed fraud pays full FN cost.
        loss[low_mask & (y == 1)] = c_fn[low_mask & (y == 1)]

        # Medium band: review cost always paid; fraud can still slip.
        loss[medium_mask & (y == 0)] = c_review
        loss[medium_mask & (y == 1)] = c_review + c_fn[medium_mask & (y == 1)] * (1.0 - review_catch_rate)

        # High band: block cost for legitimate users.
        loss[high_mask & (y == 0)] = c_fp
        return float(loss.sum())

    def _fn_costs(self, n: int, amounts: np.ndarray | None) -> np.ndarray:
        if self.config.fn_cost_mode == "constant":
            return np.full(n, self.config.c_fn, dtype=float)
        if self.config.fn_cost_mode != "amount" or amounts is None:
            raise ValueError("fn_cost_mode=amount requires transaction amounts")
        values = np.asarray(amounts, dtype=float)
        if len(values) != n or not np.isfinite(values).all() or (values < 0).any():
            raise ValueError("Invalid amounts for loss calculation")
        return values * self.config.amount_loss_multiplier

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

    def _tune_thresholds(self, final_score: np.ndarray, y_true: np.ndarray, amounts: np.ndarray | None = None) -> tuple[float, float]:
        # Prefix sums replace an O(rows * low_points * high_points) scan.
        low_grid = np.linspace(0 if self.config.full_threshold_range else 0.05, 1 if self.config.full_threshold_range else 0.65, int(self.config.low_points))
        high_grid = np.linspace(0 if self.config.full_threshold_range else 0.35, 1 if self.config.full_threshold_range else 0.95, int(self.config.high_points))
        if self.config.full_threshold_range:
            # Scores are clipped to [0, 1]. Sentinels make the three constant
            # policies explicit candidates: all High (0,0), all Medium
            # (0, >1), and all Low (>1, >1). Without the all-Low endpoint,
            # score==1 rows were forced into review even when allow-all won.
            low_grid = np.r_[low_grid, 1.0 + 1e-6]
            high_grid = np.r_[high_grid, 1.0 + 2e-6]
        order = np.argsort(final_score, kind="stable")
        scores, labels = final_score[order], y_true[order]
        fraud_cost = np.r_[0.0, np.cumsum(self._fn_costs(len(labels), amounts)[order] * labels)]
        legitimate = np.r_[0, np.cumsum(labels == 0)]
        best = (self.low_threshold, self.high_threshold)
        best_loss = np.inf

        for low in low_grid:
            for high in high_grid:
                # A real gap is required between bands. Floating-point grid
                # values such as 0.42499999999999993 and 0.425 must not create
                # an effectively empty medium-risk band.
                all_block_endpoint = self.config.full_threshold_range and low == 0 and high == 0
                if high - low < 1e-9 and not all_block_endpoint:
                    continue
                left, right = np.searchsorted(scores, [low, high], side="left")
                review_rate = (right - left) / max(len(scores), 1)
                if self.config.max_review_rate is not None and review_rate > self.config.max_review_rate:
                    continue
                loss = fraud_cost[left] + (fraud_cost[right] - fraud_cost[left]) * (1 - self.config.review_catch_rate)
                loss += (right - left) * self.config.c_review + (legitimate[-1] - legitimate[right]) * self.config.c_fp
                if loss < best_loss:
                    best_loss = loss
                    best = (float(low), float(high))
        if not np.isfinite(best_loss):
            raise ValueError("No threshold pair satisfies the manual-review capacity")
        return best
