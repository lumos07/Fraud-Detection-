from __future__ import annotations

import unittest

import pandas as pd

from fraud_pipeline.config import PipelineConfig
from fraud_pipeline.features import build_tabular_features, fit_feature_spec
from fraud_pipeline.graph_features import AccountGraphBuilder, GraphFeatureConfig
from fraud_pipeline.pipeline import FraudPipeline


class HistoricalDeviceScoreTests(unittest.TestCase):
    def test_device_score_uses_only_strictly_earlier_timestamps(self) -> None:
        frame = pd.DataFrame(
            {
                "transaction_id": ["t1", "t2", "t3", "t4", "t5", "t6"],
                "account_id": ["A", "A", "B", "D", "E", "F"],
                "timestamp": pd.to_datetime(
                    [
                        "2025-01-01T10:00:00Z",
                        "2025-01-01T11:00:00Z",
                        "2025-01-01T12:00:00Z",
                        "2025-01-01T13:00:00Z",
                        "2025-01-01T13:00:00Z",
                        "2025-01-01T14:00:00Z",
                    ],
                    utc=True,
                ),
                "amount": [10.0] * 6,
                "device_id": ["D1", "D1", "D1", "D2", "D2", "D2"],
                "label": [0] * 6,
            }
        )

        spec = fit_feature_spec(frame)
        scored = build_tabular_features(frame, spec).frame.set_index("transaction_id")

        self.assertEqual(scored.loc["t1", "device_score"], 1.0)
        self.assertEqual(scored.loc["t2", "device_score"], 1.0)
        self.assertEqual(scored.loc["t3", "device_score"], 0.5)
        # Same-time rows cannot observe each other.
        self.assertEqual(scored.loc["t4", "device_score"], 1.0)
        self.assertEqual(scored.loc["t5", "device_score"], 1.0)
        self.assertAlmostEqual(scored.loc["t6", "device_score"], 1.0 / 3.0)

    def test_graph_features_align_by_transaction_after_account_sort(self) -> None:
        frame = pd.DataFrame(
            {
                "transaction_id": ["t1", "t2", "t3"],
                "account_id": ["B", "A", "C"],
                "counterparty_account_id": ["C", "B", "A"],
                "timestamp": pd.to_datetime(
                    ["2025-01-01T10:00:00Z", "2025-01-01T11:00:00Z", "2025-01-01T12:00:00Z"],
                    utc=True,
                ),
                "amount": [10.0, 20.0, 30.0],
                "label": [0, 0, 1],
            }
        )
        spec = fit_feature_spec(frame)
        features = build_tabular_features(frame, spec)
        pipeline = FraudPipeline(PipelineConfig(raw={"components": {"graph": True}}))
        pipeline.graph_builder = AccountGraphBuilder(GraphFeatureConfig()).fit(frame)

        composed, _ = pipeline._compose_feature_frame(features, frame)
        expected = pipeline.graph_builder.account_features_["graph_degree"]

        for row in composed.itertuples(index=False):
            self.assertEqual(row.graph_degree, expected.loc[row.account_id])


if __name__ == "__main__":
    unittest.main()
