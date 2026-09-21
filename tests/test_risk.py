from __future__ import annotations

import unittest

import numpy as np

from fraud_pipeline.risk import RiskAggregator, RiskConfig


class ThresholdGridTests(unittest.TestCase):
    def test_tuned_thresholds_have_a_real_medium_band(self) -> None:
        aggregator = RiskAggregator(RiskConfig())
        scores = np.array([0.1, 0.2, 0.4, 0.425, 0.6, 0.9])
        labels = np.array([0, 0, 0, 1, 1, 1])

        low, high = aggregator._tune_thresholds(scores, labels)

        self.assertGreaterEqual(high - low, 1e-9)


if __name__ == "__main__":
    unittest.main()
