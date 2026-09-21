from __future__ import annotations

import unittest
import tempfile
from pathlib import Path
from unittest.mock import patch

import numpy as np
import pandas as pd

from fraud_pipeline.kaggle_data import adapt_transactions, mature_labels
from fraud_pipeline.temporal_features import build_temporal_features
from fraud_pipeline.temporal_graph import build_snapshot, temporal_graph_features
from fraud_pipeline.kaggle_pipeline import KagglePipeline
from fraud_pipeline.config import PipelineConfig
from fraud_pipeline.risk import RiskAggregator, RiskConfig
from fraud_pipeline.evaluate import evaluate_predictions
from fraud_pipeline.pipeline import FraudPipeline


def fixture():
    return adapt_transactions(pd.DataFrame({
        "transaction_id": ["t1", "t2", "t3", "t4"],
        "sender_account": ["A", "A", "B", "C"], "receiver_account": ["B", "C", "D", "A"],
        "timestamp": ["2023-01-01", "2023-01-02", "2023-01-02", "2023-01-10"],
        "amount": [10., 20., 999., 100.], "device_hash": ["d1", "d1", "d1", "d2"],
        "device_used": ["mobile"] * 4, "location": ["city"] * 4,
        "merchant_category": ["food"] * 4, "is_fraud": [True, False, False, False],
        "transaction_type": ["transfer"] * 4,
    }), require_labels=True)


class TemporalMigrationTests(unittest.TestCase):
    def test_boolean_labels_and_no_fabricated_identity(self):
        df = fixture()
        self.assertEqual(df.label.tolist(), [1, 0, 0, 0])
        raw = df.drop(columns=["label", "device_id"])
        self.assertTrue(adapt_transactions(raw).label.isna().all())
        self.assertNotIn("device_id", adapt_transactions(raw))
        with self.assertRaises(ValueError):
            adapt_transactions(raw, require_labels=True)

    def test_future_invariance_of_behavioral_features(self):
        df = fixture()
        first = build_temporal_features(df.iloc[:3])
        full = build_temporal_features(df)
        pd.testing.assert_frame_equal(first.frame[first.numeric_cols], full.frame.iloc[:3][full.numeric_cols])
        self.assertEqual(first.frame.iloc[0].sender_lifetime_count, 0)
        self.assertEqual(first.frame.iloc[1].mean_amount_30d, 10)
        self.assertEqual(first.frame.iloc[1].interaction_count, 0)

    def test_same_timestamp_does_not_change_history(self):
        df = fixture()
        df.loc[2, "account_id"] = "A"
        features = build_temporal_features(df).frame.set_index("transaction_id")
        self.assertEqual(features.loc["t2", "sender_lifetime_count"], 1)
        self.assertEqual(features.loc["t3", "sender_lifetime_count"], 1)
        self.assertEqual(features.loc["t2", "mean_amount_30d"], 10)
        self.assertEqual(features.loc["t3", "mean_amount_30d"], 10)

    def test_snapshot_exposure_and_label_delay(self):
        df = fixture()
        cfg = {"label_maturity": "3d"}
        early = build_snapshot(df, pd.Timestamp("2023-01-03", tz="UTC"), cfg)
        late = build_snapshot(df, pd.Timestamp("2023-01-05", tz="UTC"), cfg)
        self.assertEqual(early.features.fraud_neighbor_exposure.sum(), 0)
        self.assertEqual(late.features.loc["B", "fraud_neighbor_exposure"], 0.5)
        self.assertEqual(late.features.loc["C", "fraud_neighbor_exposure"], 0.5)
        self.assertEqual(late.features.loc["D", "fraud_neighbor_exposure"], 0)
        self.assertEqual(late.metadata["edges"], 3)
        self.assertIn("device_id", late.metadata["attribute_relations"])
        self.assertNotIn("device_used", late.metadata["attribute_relations"])

    def test_explicit_confirmation_times_override_assumed_delay(self):
        df = fixture()
        df["label_available_at"] = pd.to_datetime(["2023-02-01", "2023-01-05", None, None], utc=True)
        mask = mature_labels(df, pd.Timestamp("2023-01-05", tz="UTC"), "0D")
        self.assertFalse(mask.any())
        mask = mature_labels(df, pd.Timestamp("2023-01-06", tz="UTC"), "0D")
        self.assertEqual(mask.tolist(), [False, True, False, False])

    def test_attribute_nodes_do_not_dilute_community_account_ratio(self):
        df = fixture().iloc[:3].copy()
        df["account_id"] = ["A", "A", "B"]
        df["counterparty_account_id"] = ["B", "C", "C"]
        snapshot = build_snapshot(df, pd.Timestamp("2023-01-08", tz="UTC"), {"label_maturity": "1D"})
        np.testing.assert_allclose(snapshot.features.community_risk, [1 / 3] * 3)

    def test_evaluation_rejects_unknown_outcomes(self):
        with self.assertRaises(ValueError):
            evaluate_predictions(pd.DataFrame({"label": [np.nan], "risk_band": ["low"]}), "label", RiskAggregator(RiskConfig()))

    def test_future_labels_do_not_change_pagerank_or_snapshot(self):
        df = fixture()
        cutoff = pd.Timestamp("2023-01-03", tz="UTC")
        a = build_snapshot(df, cutoff, {"label_maturity": "3d"})
        changed = df.copy()
        changed.loc[3, ["label", "amount"]] = [1, 1e9]
        changed.loc[3, "counterparty_account_id"] = "NEW"
        b = build_snapshot(changed, cutoff, {"label_maturity": "3d"})
        pd.testing.assert_frame_equal(a.features, b.features)
        changed["label"] = 1 - changed.label
        c = build_snapshot(changed, cutoff, {"label_maturity": "0d"})
        np.testing.assert_allclose(a.features.graph_pagerank, c.features.graph_pagerank)

    def test_target_labels_ignored_and_graph_formula(self):
        df = fixture()
        history, target = df.iloc[:2], df.iloc[2:].copy()
        target["label"] = 1
        a, _ = temporal_graph_features(history, target, {"snapshot_frequency": "1d", "label_maturity": "0d"})
        target["label"] = 0
        b, _ = temporal_graph_features(history, target, {"snapshot_frequency": "1d", "label_maturity": "0d"})
        pd.testing.assert_frame_equal(a, b)
        pipeline = KagglePipeline(PipelineConfig({}))
        expected = 0.4 * a.community_risk_normalized + 0.4 * a.interaction + 0.2 * a.pagerank_normalized
        np.testing.assert_allclose(pipeline._compute_graph_score(a), expected)

    def test_prefix_cost_search_matches_brute_force_and_capacity(self):
        rng = np.random.default_rng(42)
        scores, y, amounts = rng.random(50), rng.integers(0, 2, 50), rng.uniform(1, 100, 50)
        agg = RiskAggregator(RiskConfig(fn_cost_mode="amount", max_review_rate=0.2))
        low, high = agg._tune_thresholds(scores, y, amounts)
        selected = np.where(scores < low, "low", np.where(scores < high, "medium", "high"))
        cost = agg.expected_loss(selected, y, amounts)
        brute = []
        for lo in np.linspace(.05, .65, 25):
            for hi in np.linspace(.35, .95, 25):
                bands = np.where(scores < lo, "low", np.where(scores < hi, "medium", "high"))
                if hi - lo >= 1e-9 and np.mean(bands == "medium") <= 0.2:
                    brute.append(agg.expected_loss(bands, y, amounts))
        self.assertAlmostEqual(cost, min(brute))

    def test_medium_is_not_a_positive_prediction(self):
        scored = pd.DataFrame({"label": [1, 1, 0, 0], "risk_band": ["medium", "high", "medium", "high"], "final_score": [.5, .9, .4, .8]})
        metrics = evaluate_predictions(scored, "label", RiskAggregator(RiskConfig())).metrics
        self.assertEqual(metrics["recall_flagged"], .5)
        self.assertEqual(metrics["fp_flagged"], 1)
        self.assertEqual(metrics["fn_not_high"], 1)

    def test_all_block_endpoint_is_available_for_extreme_fn_cost(self):
        agg = RiskAggregator(RiskConfig(full_threshold_range=True, c_fn=1e6, c_fp=1))
        low, high = agg._tune_thresholds(np.array([0., .5, 1.]), np.array([1, 0, 1]))
        self.assertEqual((low, high), (0., 0.))

    def test_all_allow_endpoint_is_available_for_extreme_fp_cost(self):
        agg = RiskAggregator(RiskConfig(full_threshold_range=True, c_fn=1, c_fp=1e6, c_review=1e6))
        low, high = agg._tune_thresholds(np.array([0., .5, 1.]), np.array([0, 0, 0]))
        self.assertGreater(low, 1.)
        self.assertGreater(high, low)

    def test_all_review_endpoint_is_available_when_review_is_cheapest(self):
        agg = RiskAggregator(RiskConfig(full_threshold_range=True, c_fn=1e6, c_fp=1e6, c_review=0, review_catch_rate=1))
        low, high = agg._tune_thresholds(np.array([0., .5, 1.]), np.array([0, 0, 0]))
        self.assertEqual(low, 0.)
        self.assertGreater(high, 1.)

    def test_snapshot_cache_invalidates_changed_history(self):
        df = fixture()
        cache = {}
        cfg = {"snapshot_frequency": "1D", "label_maturity": "0D"}
        a, audit = temporal_graph_features(df.iloc[:3], df.iloc[3:], cfg, cache)
        b, audit = temporal_graph_features(df.iloc[:3], df.iloc[3:], cfg, cache)
        self.assertTrue(audit[0]["cache_hit"])
        pd.testing.assert_frame_equal(a, b)
        changed = df.iloc[:3].copy()
        changed["label"] = 0
        _, audit = temporal_graph_features(changed, df.iloc[3:], cfg, cache)
        self.assertFalse(audit[0]["cache_hit"])

    def test_training_save_load_and_api(self):
        from fraud_pipeline import service
        rows = []
        for i in range(300):
            rows.append({"transaction_id": str(i), "sender_account": f"A{i % 10}", "receiver_account": f"A{(i + 1) % 10}",
                         "timestamp": pd.Timestamp("2023-01-01", tz="UTC") + pd.Timedelta(hours=i * 8),
                         "amount": 10. + (i % 7) * 30., "is_fraud": i % 7 == 0,
                         "transaction_type": "transfer", "device_used": "mobile", "location": "city", "merchant_category": "food"})
        data = adapt_transactions(pd.DataFrame(rows), require_labels=True)
        config = PipelineConfig({"supervised": {"params": {"n_estimators": 5, "n_jobs": 1, "verbosity": -1}},
                                 "anomaly": {"autoencoder": {"epochs": 1, "batch_size": 64}, "isolation_forest": {"n_estimators": 5}},
                                 "graph": {"label_maturity": "1D", "snapshot_frequency": "10D"},
                                 "risk": {"threshold_grid": {"low_points": 5, "high_points": 5}}})
        model = KagglePipeline(config)
        model.fit(data.iloc[:180], data.iloc[180:240])
        target = data.iloc[240:243].copy()
        a = model.predict(target)
        target["label"] = 1 - target.label
        b = model.predict(target)
        np.testing.assert_allclose(a.final_score, b.final_score)
        self.assertIn("fraud_neighbor_exposure_normalized", model.numeric_cols)
        self.assertNotIn("community_risk_normalized", model.numeric_cols)
        self.assertNotIn("fraud_neighbor_exposure_normalized", model.anomaly_numeric_cols)
        with tempfile.TemporaryDirectory() as folder:
            model.save(Path(folder) / "model.joblib")
            loaded = FraudPipeline.load(Path(folder) / "model.joblib")
            np.testing.assert_allclose(a.final_score, loaded.predict(target).final_score)
        request = service.AnalystScoreRequest(transactions=[{"transaction_id": "request", "sender_account": "A0", "receiver_account": "A1", "timestamp": "2023-04-15", "amount": 100}], include_explanations=True)
        previous = service.PIPELINE
        try:
            service.PIPELINE = loaded
            reply = service.score_analyst(request)
        finally:
            service.PIPELINE = previous
        self.assertEqual(reply["n"], 1)
        self.assertIn("fraud_neighbor_exposure_normalized", reply["scores"][0]["analyst_payload"]["graph"])

    def test_lightgbm_to_xgboost_fallback(self):
        from fraud_pipeline import models
        if models.xgb is None:
            self.skipTest("XGBoost not installed")
        frame = pd.DataFrame({"amount": np.arange(100, dtype=np.float32)})
        y = pd.Series(np.arange(100) % 3 == 0).astype(int)
        estimator = models.SupervisedModel("lightgbm", {"n_estimators": 3, "n_jobs": 1, "num_leaves": 31, "verbosity": -1}, 2)
        with patch.object(models, "lgb", None):
            estimator.fit(frame.iloc[:80], y.iloc[:80], frame.iloc[80:], y.iloc[80:], ["amount"], [])
        self.assertIsInstance(estimator.model, models.xgb.XGBClassifier)
        self.assertEqual(len(estimator.predict_proba(frame.iloc[80:])), 20)


if __name__ == "__main__":
    unittest.main()
