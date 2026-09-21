from __future__ import annotations

import numpy as np


class PercentileNormalizer:
    """Empirical training-score CDF: separately normalize each anomaly detector."""

    def fit(self, scores: np.ndarray) -> "PercentileNormalizer":
        values = np.asarray(scores, dtype=float)
        if len(values) == 0 or not np.isfinite(values).all():
            raise ValueError("Scores must be nonempty and finite")
        self.reference = np.sort(values)
        return self

    def transform(self, scores: np.ndarray) -> np.ndarray:
        if self.reference[0] == self.reference[-1]:
            return (np.asarray(scores) > self.reference[-1]).astype(float)
        return np.searchsorted(self.reference, scores, side="right") / len(self.reference)


def simplex_weights(values: dict[str, float]) -> dict[str, float]:
    weights = {k: float(v) for k, v in values.items()}
    if not weights or not all(np.isfinite(v) and v >= 0 for v in weights.values()) or not np.isclose(sum(weights.values()), 1.0):
        raise ValueError("Weights must be finite, nonnegative, and sum to one")
    return weights
