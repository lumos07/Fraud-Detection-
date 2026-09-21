"""Kaggle adaptation reusing the existing estimators, risk optimizer and API model."""
from __future__ import annotations

import copy
from dataclasses import asdict

import numpy as np
import pandas as pd

from fraud_pipeline.features import FeatureSpec
from fraud_pipeline.kaggle_data import CATEGORICAL, adapt_transactions, mature_labels
from fraud_pipeline.models import AutoencoderScorer, IsolationForestScorer, SupervisedModel
from fraud_pipeline.pipeline import FitArtifacts, FraudPipeline
from fraud_pipeline.risk import RiskAggregator, RiskConfig
from fraud_pipeline.score_scaling import PercentileNormalizer, simplex_weights
from fraud_pipeline.temporal_features import build_temporal_features
from fraud_pipeline.temporal_graph import GRAPH_COLUMNS, SUPERVISED_GRAPH_COLUMNS, build_snapshot, temporal_graph_features


class KagglePipeline(FraudPipeline):
    def __init__(self, config):
        super().__init__(config)
        self.anomaly_numeric_cols: list[str] = []
        self.normalizers: dict[str, PercentileNormalizer] = {}
        self.graph_weights = simplex_weights(config.graph.get("risk_weights", {"community": 0.4, "interaction": 0.4, "pagerank": 0.2}))
        self.anomaly_weights = simplex_weights(config.anomaly.get("weights", {"isolation_forest": 0.5, "autoencoder": 0.5}))
        if set(self.graph_weights) != {"community", "interaction", "pagerank"}:
            raise ValueError("Graph risk weights must be community, interaction and pagerank only")
        if set(self.anomaly_weights) != {"isolation_forest", "autoencoder"}:
            raise ValueError("Anomaly weights must specify isolation_forest and autoencoder")
        if int(config.graph.get("weight_grid_divisions", 4)) < 1:
            raise ValueError("Graph weight_grid_divisions must be positive")
        self.history_: pd.DataFrame | None = None
        self.snapshot_audit: list[dict] = []
        self.graph_snapshot = None
        self._snapshot_cache: dict = {}
        self.fit_metadata: dict = {}
        from fraud_pipeline.models import torch
        if torch is not None:
            # A single intra-op thread avoids cross-runtime OpenMP barriers on
            # macOS when LightGBM and PyTorch ship different libomp builds.
            torch.set_num_threads(int(config.anomaly.get("torch_threads", 1)))

    def engineer(self, target: pd.DataFrame, history: pd.DataFrame | None = None) -> tuple[pd.DataFrame, list[str], list[str]]:
        target = adapt_transactions(target)
        history = adapt_transactions(history) if history is not None and len(history) else target.iloc[:0].copy()
        if not set(history.transaction_id).isdisjoint(target.transaction_id):
            history = history[~history.transaction_id.isin(target.transaction_id)]
        combined = pd.concat([history, target], ignore_index=True).sort_values(["timestamp", "transaction_id"])
        for col in CATEGORICAL:
            if col in combined:
                combined[col] = combined[col].astype("string").fillna("unknown")
        output = build_temporal_features(combined)
        base = output.frame.set_index("transaction_id", drop=False).loc[target.transaction_id].reset_index(drop=True)
        if self.graph_enabled:
            graph, audit = temporal_graph_features(history, target, self.config.graph, self._snapshot_cache)
            base = pd.concat([base, graph], axis=1)
            self.snapshot_audit = (self.snapshot_audit + audit)[-100:]
        return base, output.numeric_cols, output.categorical_cols

    def training_features(self, train: pd.DataFrame) -> tuple[pd.DataFrame, list[str], list[str]]:
        output = build_temporal_features(train)
        base = output.frame
        if self.graph_enabled:
            graph, audit = temporal_graph_features(train, base, self.config.graph)
            base = pd.concat([base.reset_index(drop=True), graph], axis=1)
            self.snapshot_audit.extend(audit)
        return base, output.numeric_cols, output.categorical_cols

    def fit(self, train_df: pd.DataFrame, validation_df: pd.DataFrame) -> FitArtifacts:
        train = adapt_transactions(train_df, require_labels=True)
        validation = adapt_transactions(validation_df, require_labels=True)
        if train.timestamp.max() >= validation.timestamp.min():
            raise ValueError("Training must strictly precede validation")
        train_frame, numeric, categorical = self.training_features(train)
        validation_frame, _, _ = self.engineer(validation, train)
        return self.fit_frames(train_frame, validation_frame, numeric, categorical, train)

    def fit_frames(self, train_frame, validation_frame, numeric, categorical, raw_train) -> FitArtifacts:
        """Accept reusable, historically generated frames for component ablations."""
        self.anomaly_numeric_cols = list(numeric)
        self.graph_cols = list(GRAPH_COLUMNS) if self.graph_enabled else []
        self.numeric_cols = list(numeric) + (SUPERVISED_GRAPH_COLUMNS if self.graph_enabled else [])
        self.categorical_cols = list(categorical)
        self.feature_spec = FeatureSpec(0.0, "timestamp", "label", "account_id")
        # Delayed labels must be available before the next deployment period.
        train_cutoff = validation_frame.timestamp.min()
        delay = self.config.graph.get("label_maturity", "7d")
        train_mask = mature_labels(train_frame, train_cutoff, delay)
        validation_cutoff = pd.Timestamp(self.config.raw.get("validation_label_cutoff", validation_frame.timestamp.max()))
        if validation_cutoff.tzinfo is None:
            validation_cutoff = validation_cutoff.tz_localize("UTC")
        val_mask = mature_labels(validation_frame, validation_cutoff, delay)
        tr, va = train_frame.loc[train_mask], validation_frame.loc[val_mask]
        if tr.label.nunique() != 2 or va.label.nunique() != 2:
            raise ValueError("Mature training and calibration data must each contain both classes; use longer periods")
        cfg = self.config.supervised
        if self.supervised_model is None:
            self.supervised_model = SupervisedModel(str(cfg.get("algorithm", "lightgbm")), dict(cfg.get("params", {})), int(cfg.get("early_stopping_rounds", 50)))
            self.supervised_model.fit(tr, tr.label.astype(int), va, va.label.astype(int), self.numeric_cols, self.categorical_cols, return_training_scores=False)
        elif self.supervised_model.numeric_cols != self.numeric_cols or self.supervised_model.categorical_cols != self.categorical_cols:
            raise ValueError("Cached supervised estimator has different input columns")
        if self.anomaly_enabled:
            # Deterministic sample caps reconstruction-model training memory/work.
            sample_size = min(len(train_frame), int(self.config.anomaly.get("max_training_rows", 200000)))
            sample = train_frame.sample(n=sample_size, random_state=self.config.seed)[self.anomaly_numeric_cols]
            if self.anomaly_weights["isolation_forest"] > 0 and self.isolation_model is None:
                settings = self.config.anomaly.get("isolation_forest", {})
                self.isolation_model = IsolationForestScorer(int(settings.get("n_estimators", 200)), float(settings.get("contamination", 0.005)), self.config.seed).fit(sample)
                self.normalizers["isolation_forest"] = PercentileNormalizer().fit(self.isolation_model.score(sample))
            if self.anomaly_weights["autoencoder"] > 0 and self.autoencoder_model is None:
                settings = self.config.anomaly.get("autoencoder", {})
                self.autoencoder_model = AutoencoderScorer(
                    latent_dim=int(settings.get("latent_dim", 8)), hidden_dims=settings.get("hidden_dims", [32, 16, 8]),
                    epochs=int(settings.get("epochs", 10)), batch_size=int(settings.get("batch_size", 1024)),
                    learning_rate=float(settings.get("learning_rate", 0.001)), random_state=self.config.seed,
                    backend=str(settings.get("backend", "torch")),
                ).fit(sample)
                self.normalizers["autoencoder"] = PercentileNormalizer().fit(self.autoencoder_model.score(sample))
        validation_scores = self._score_components(validation_frame)
        risk_config = self._risk_config()
        self.risk_aggregator = RiskAggregator(risk_config)
        self.risk_aggregator.fit(validation_scores.loc[val_mask], va.label.astype(int), va.amount.to_numpy())
        if self.graph_enabled and self.config.graph.get("tune_weights", False):
            self._tune_graph_weights(validation_frame, validation_scores, val_mask)
            validation_scores = self._score_components(validation_frame)
        self.history_ = raw_train.copy()
        self.fit_metadata = {
            "supervised_training_rows": len(tr), "mature_calibration_rows": len(va),
            "positive_training_rows": int(tr.label.sum()), "label_maturity": delay,
            "supervised_backend": type(self.supervised_model.model).__name__,
            "anomaly_backend": self.autoencoder_model.backend if self.autoencoder_model else None,
            "anomaly_weights": self.anomaly_weights,
            "anomaly_training_rows": min(len(train_frame), int(self.config.anomaly.get("max_training_rows", 200000))) if self.anomaly_enabled else 0,
            "class_weight_negative_positive_ratio": float((tr.label == 0).sum() / (tr.label == 1).sum()),
            "graph_weights": self.graph_weights, "risk_config": asdict(self.risk_aggregator.config),
            "anomaly_columns": self.anomaly_numeric_cols, "supervised_graph_columns": SUPERVISED_GRAPH_COLUMNS if self.graph_enabled else [],
            "normalization": "community ratio; exposure/(exposure+scale); PageRank/(PageRank+scale/accounts)",
        }
        # Avoid retaining/scoring millions of training rows merely for reporting.
        return FitArtifacts(self._attach_risk(validation_frame, validation_scores), pd.DataFrame())

    def _tune_graph_weights(self, frame, scores, val_mask):
        y, amounts = frame.loc[val_mask, "label"].astype(int), frame.loc[val_mask, "amount"].to_numpy()
        risk = self.risk_aggregator
        bands = risk.predict(scores.loc[val_mask]).risk_band.to_numpy()
        best_cost = risk.expected_loss(bands, y.to_numpy(), amounts)
        best_weights, best_risk = self.graph_weights.copy(), risk
        divisions = int(self.config.graph.get("weight_grid_divisions", 4))
        for a in range(divisions + 1):
            for b in range(divisions + 1 - a):
                candidate = {"community": a / divisions, "interaction": b / divisions, "pagerank": (divisions - a - b) / divisions}
                self.graph_weights = candidate
                trial = scores.loc[val_mask].copy()
                trial["s_graph"] = self._compute_graph_score(frame.loc[val_mask])
                agg = RiskAggregator(copy.deepcopy(self._risk_config()))
                agg.fit(trial, y, amounts)
                cost = agg.expected_loss(agg.predict(trial).risk_band.to_numpy(), y.to_numpy(), amounts)
                if cost < best_cost:
                    best_cost, best_weights, best_risk = cost, candidate.copy(), agg
        self.graph_weights, self.risk_aggregator = best_weights, best_risk

    def _risk_config(self) -> RiskConfig:
        cfg = super()._risk_config()
        weights = np.array([cfg.weight_supervised, cfg.weight_anomaly, cfg.weight_graph])
        if not np.isfinite(weights).all() or (weights < 0).any() or weights.sum() <= 0:
            raise ValueError("Active ensemble weights must be nonnegative with positive sum")
        cfg.weight_supervised, cfg.weight_anomaly, cfg.weight_graph = map(float, weights / weights.sum())
        settings = self.config.risk
        cfg.tune_weights = bool(settings.get("tune_weights", False))
        cfg.weight_grid_step = float(settings.get("weight_grid_step", 0.5))
        cfg.max_review_rate = settings.get("max_review_rate")
        cfg.fn_cost_mode = settings.get("fn_cost_mode", "constant")
        cfg.amount_loss_multiplier = float(settings.get("amount_loss_multiplier", 1.0))
        cfg.full_threshold_range = True
        cfg.__post_init__()
        return cfg

    def _compute_graph_score(self, frame):
        mode = self.config.graph.get("ablation", "all")
        c, e, p = [frame[col].to_numpy() for col in ["community_risk_normalized", "fraud_neighbor_exposure_normalized", "pagerank_normalized"]]
        if mode == "pagerank": return p
        if mode == "community": return c
        if mode == "exposure": return e
        if mode == "community_exposure": return (c + e) / 2
        if mode != "all": raise ValueError(f"Unknown graph ablation {mode}")
        w = self.graph_weights
        return w["community"] * c + w["interaction"] * c * e + w["pagerank"] * p

    def _score_components(self, frame):
        scores = pd.DataFrame(index=frame.index)
        scores["s_supervised"] = np.concatenate([self.supervised_model.predict_proba(frame.iloc[start:start + 65536]) for start in range(0, len(frame), 65536)])
        scores["s_anomaly"] = np.zeros(len(frame))
        if self.anomaly_enabled:
            for name, model in [("isolation_forest", self.isolation_model), ("autoencoder", self.autoencoder_model)]:
                if model is None or self.anomaly_weights[name] == 0:
                    continue
                raw = np.concatenate([model.score(frame.iloc[start:start + 65536][self.anomaly_numeric_cols]) for start in range(0, len(frame), 65536)])
                normalized = self.normalizers[name].transform(raw)
                scores[name + "_raw"] = raw
                scores[name + "_percentile"] = normalized
                scores["s_anomaly"] += self.anomaly_weights[name] * normalized
        scores["s_graph"] = self._compute_graph_score(frame) if self.graph_enabled else 0.0
        return scores

    def _check_fitted(self):
        if self.supervised_model is None or self.risk_aggregator is None:
            raise RuntimeError("Pipeline is not fit yet")

    def predict(self, df, history_df=None):
        self._check_fitted()
        target = adapt_transactions(df)
        # Never infer fraud history from the labels on target scoring requests.
        history = history_df if history_df is not None else self.history_
        frame, _, _ = self.engineer(target, history)
        return self._attach_risk(frame, self._score_components(frame))

    def dump_metadata(self, output_path):
        import json
        from pathlib import Path
        payload = {"components": self.config.components, "numeric_cols": self.numeric_cols, "categorical_cols": self.categorical_cols,
                   "risk_thresholds": {"low": self.risk_aggregator.low_threshold, "high": self.risk_aggregator.high_threshold}, **self.fit_metadata}
        Path(output_path).write_text(json.dumps(payload, indent=2), encoding="utf-8")

    def __getstate__(self):
        state = self.__dict__.copy()
        # Persist models/history, not mutable request-time snapshot caches.
        state["_snapshot_cache"] = {}
        return state

    def analyst_payload(self, scored_df, history_df=None, top_k=5, neighbor_limit=10, include_explanations=True, include_graph_neighbors=True, include_recent_history=True):
        explanations = self.supervised_model.explain(scored_df, top_k) if include_explanations else [[] for _ in range(len(scored_df))]
        history = None
        if include_recent_history:
            raw = history_df if history_df is not None else self.history_
            if raw is not None:
                history = pd.concat([adapt_transactions(raw), adapt_transactions(scored_df)], ignore_index=True).drop_duplicates("transaction_id")
        payload = []
        for i, (_, row) in enumerate(scored_df.iterrows()):
            recent = None
            if history is not None:
                past = history.loc[(history.account_id == row.account_id) & (history.timestamp < row.timestamp)]
                day = past.loc[past.timestamp >= row.timestamp - pd.Timedelta(days=1)]
                recent = {"prior_sender_transactions": len(past), "transactions_24h": len(day), "amount_total_24h": float(day.amount.sum()),
                          "last_transaction_at": past.timestamp.max().isoformat() if len(past) else None}
            payload.append({"top_features": explanations[i], "graph": {c: float(row[c]) for c in GRAPH_COLUMNS if c in row},
                            "anomaly": {c: float(row[c]) for c in ["isolation_forest_percentile", "autoencoder_percentile", "autoencoder_raw"] if c in row},
                            "graph_neighbors": [], "graph_neighbor_status": "Not expanded in compact Kaggle payload" if include_graph_neighbors else "Disabled",
                            "recent_history": recent})
        return payload
