from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any
import pickle

import numpy as np
import pandas as pd

from fraud_pipeline.config import PipelineConfig
from fraud_pipeline.data import standardize_schema
from fraud_pipeline.evaluate import EvaluationResult, evaluate_predictions
from fraud_pipeline.features import FeatureOutput, FeatureSpec, build_tabular_features, fit_feature_spec
from fraud_pipeline.graph_features import AccountGraphBuilder, GraphFeatureConfig, link_prediction_features
from fraud_pipeline.models import AutoencoderScorer, IsolationForestScorer, SupervisedModel
from fraud_pipeline.risk import RiskAggregator, RiskConfig


@dataclass
class FitArtifacts:
    validation_scored: pd.DataFrame
    train_scored: pd.DataFrame


class FraudPipeline:
    def __init__(self, config: PipelineConfig):
        self.config = config
        self.feature_spec: FeatureSpec | None = None
        self.graph_builder: AccountGraphBuilder | None = None
        self.supervised_model: SupervisedModel | None = None
        self.isolation_model: IsolationForestScorer | None = None
        self.autoencoder_model: AutoencoderScorer | None = None
        self.risk_aggregator: RiskAggregator | None = None
        self.numeric_cols: list[str] = []
        self.categorical_cols: list[str] = []
        self.graph_cols: list[str] = []

    @property
    def timestamp_col(self) -> str:
        return self.config.columns["timestamp"]

    @property
    def label_col(self) -> str:
        return self.config.columns["label"]

    @property
    def account_col(self) -> str:
        return self.config.columns["account_id"]

    @property
    def anomaly_enabled(self) -> bool:
        return self.config.components["anomaly"]

    @property
    def graph_enabled(self) -> bool:
        return self.config.components["graph"]

    def fit(self, train_df: pd.DataFrame, validation_df: pd.DataFrame) -> FitArtifacts:
        self.feature_spec = fit_feature_spec(
            train_df,
            timestamp_col=self.timestamp_col,
            label_col=self.label_col,
            account_col=self.account_col,
        )

        train_features = build_tabular_features(train_df, self.feature_spec)
        validation_features = self._build_with_history(train_df, validation_df, self.feature_spec)

        if self.graph_enabled:
            self.graph_builder = AccountGraphBuilder(
                GraphFeatureConfig(
                    account_col=self.account_col,
                    label_col=self.label_col,
                    max_attribute_group_size=int(self.config.graph.get("max_attribute_group_size", 50)),
                    use_louvain=bool(self.config.graph.get("use_louvain", True)),
                )
            )
            self.graph_builder.fit(train_df)

        train_frame, feature_cols = self._compose_feature_frame(train_features, train_df)
        validation_frame, _ = self._compose_feature_frame(validation_features, validation_df)
        self.numeric_cols = feature_cols["numeric"]
        self.categorical_cols = feature_cols["categorical"]
        self.graph_cols = feature_cols["graph"]

        y_train = train_frame[self.label_col].astype(int)
        y_validation = validation_frame[self.label_col].astype(int)

        supervised_cfg = self.config.supervised
        self.supervised_model = SupervisedModel(
            algorithm=str(supervised_cfg.get("algorithm", "lightgbm")),
            params=dict(supervised_cfg.get("params", {})),
            early_stopping_rounds=int(supervised_cfg.get("early_stopping_rounds", 100)),
        )
        self.supervised_model.fit(
            train_df=train_frame,
            y_train=y_train,
            validation_df=validation_frame,
            y_validation=y_validation,
            numeric_cols=self.numeric_cols,
            categorical_cols=self.categorical_cols,
        )

        if self.anomaly_enabled:
            anomaly_cfg = self.config.anomaly
            if_cfg = anomaly_cfg.get("isolation_forest", {})
            self.isolation_model = IsolationForestScorer(
                n_estimators=int(if_cfg.get("n_estimators", 200)),
                contamination=float(if_cfg.get("contamination", 0.005)),
                random_state=int(if_cfg.get("random_state", self.config.seed)),
            )
            self.isolation_model.fit(train_frame[self.numeric_cols])

            ae_cfg = anomaly_cfg.get("autoencoder", {})
            self.autoencoder_model = AutoencoderScorer(
                latent_dim=int(ae_cfg.get("latent_dim", 8)),
                hidden_dims=[int(x) for x in ae_cfg.get("hidden_dims", [64, 32, 8, 32, 64])],
                epochs=int(ae_cfg.get("epochs", 30)),
                batch_size=int(ae_cfg.get("batch_size", 1024)),
                learning_rate=float(ae_cfg.get("learning_rate", 1e-3)),
                random_state=int(self.config.seed),
                backend=str(ae_cfg.get("backend", "auto")),
            )
            self.autoencoder_model.fit(train_frame[self.numeric_cols])

        train_scores = self._score_components(train_frame)
        validation_scores = self._score_components(validation_frame)

        self.risk_aggregator = RiskAggregator(self._risk_config())
        self.risk_aggregator.fit(
            pd.DataFrame(
                {
                    "s_supervised": validation_scores["s_supervised"],
                    "s_anomaly": validation_scores["s_anomaly"],
                    "s_graph": validation_scores["s_graph"],
                }
            ),
            y_true=y_validation,
        )

        train_scored = self._attach_risk(train_frame, train_scores)
        validation_scored = self._attach_risk(validation_frame, validation_scores)
        return FitArtifacts(validation_scored=validation_scored, train_scored=train_scored)

    def predict(self, df: pd.DataFrame, history_df: pd.DataFrame | None = None) -> pd.DataFrame:
        self._check_fitted()
        data_df = standardize_schema(
            df,
            timestamp_col=self.timestamp_col,
            label_col=self.label_col,
        )
        if history_df is not None and len(history_df) > 0:
            history_std = standardize_schema(
                history_df,
                timestamp_col=self.timestamp_col,
                label_col=self.label_col,
            )
            feature_output = self._build_with_history(history_std, data_df, self.feature_spec)
            source = data_df
        else:
            feature_output = build_tabular_features(data_df, self.feature_spec)
            source = data_df

        frame, _ = self._compose_feature_frame(feature_output, source)
        comp_scores = self._score_components(frame)
        return self._attach_risk(frame, comp_scores)

    def evaluate(self, df: pd.DataFrame, history_df: pd.DataFrame | None = None) -> EvaluationResult:
        scored = self.predict(df, history_df=history_df)
        return evaluate_predictions(
            scored_frame=scored,
            label_col=self.label_col,
            risk_aggregator=self.risk_aggregator,
            timestamp_col=self.timestamp_col,
        )

    def analyst_payload(
        self,
        scored_df: pd.DataFrame,
        history_df: pd.DataFrame | None = None,
        top_k: int = 5,
        neighbor_limit: int = 10,
        include_explanations: bool = True,
        include_graph_neighbors: bool = True,
        include_recent_history: bool = True,
    ) -> list[dict[str, Any]]:
        self._check_fitted()
        if self.supervised_model is None:
            raise RuntimeError("Supervised model is not available.")

        frame = scored_df.copy()
        if include_explanations:
            explanations = self.supervised_model.explain(frame, top_k=top_k)
        else:
            explanations = [[] for _ in range(len(frame))]

        if self.account_col in frame.columns:
            accounts = frame[self.account_col].astype(str).tolist()
        else:
            accounts = ["unknown"] * len(frame)

        if self.timestamp_col in frame.columns:
            timestamps = pd.to_datetime(frame[self.timestamp_col], utc=True, errors="coerce", format="mixed")
        else:
            timestamps = pd.Series([pd.Timestamp.utcnow()] * len(frame))

        if history_df is not None and len(history_df) > 0:
            history = standardize_schema(
                history_df,
                timestamp_col=self.timestamp_col,
                label_col=self.label_col,
            )
        else:
            history = frame.copy()
            if self.timestamp_col in history.columns:
                history[self.timestamp_col] = pd.to_datetime(
                    history[self.timestamp_col], utc=True, errors="coerce", format="mixed"
                )

        payload = []
        for i, account_id in enumerate(accounts):
            graph_neighbors = []
            recent_history = None
            if include_graph_neighbors:
                graph_neighbors = self._graph_neighbors(account_id, limit=neighbor_limit)
            if include_recent_history:
                recent_history = self._history_summary(
                    account_id=account_id,
                    event_ts=timestamps.iloc[i] if i < len(timestamps) else pd.Timestamp.utcnow(),
                    history_df=history,
                )
            payload.append(
                {
                    "top_features": explanations[i] if i < len(explanations) else [],
                    "graph_neighbors": graph_neighbors,
                    "recent_history": recent_history,
                }
            )
        return payload

    def save(self, path: str | Path) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("wb") as f:
            pickle.dump(self, f)

    @staticmethod
    def load(path: str | Path) -> "FraudPipeline":
        with Path(path).open("rb") as f:
            pipeline = pickle.load(f)
        if not isinstance(pipeline, FraudPipeline):
            raise TypeError("Loaded object is not FraudPipeline")
        return pipeline

    def dump_metadata(self, output_path: str | Path) -> None:
        self._check_fitted()
        payload = {
            "timestamp_col": self.timestamp_col,
            "label_col": self.label_col,
            "account_col": self.account_col,
            "numeric_cols": self.numeric_cols,
            "categorical_cols": self.categorical_cols,
            "graph_cols": self.graph_cols,
            "components": self.config.components,
            "risk_thresholds": {
                "low": self.risk_aggregator.low_threshold,
                "high": self.risk_aggregator.high_threshold,
            },
        }
        with Path(output_path).open("w", encoding="utf-8") as f:
            json.dump(payload, f, indent=2)

    def _compose_feature_frame(
        self,
        feature_output: FeatureOutput,
        source_df: pd.DataFrame,
    ) -> tuple[pd.DataFrame, dict[str, list[str]]]:
        base = feature_output.frame.copy()
        base = base.reset_index(drop=True)
        graph_cols: list[str] = []
        if self.graph_enabled:
            if self.graph_builder is None:
                raise RuntimeError("Graph builder must be initialized before composing graph features.")
            transaction_col = self.config.columns.get("transaction_id", "transaction_id")
            if transaction_col not in base.columns or transaction_col not in source_df.columns:
                raise ValueError(
                    f"{transaction_col} is required to align graph features with engineered rows"
                )
            if not base[transaction_col].is_unique or not source_df[transaction_col].is_unique:
                raise ValueError(f"{transaction_col} must be unique for graph feature alignment")
            source_by_transaction = source_df.set_index(transaction_col, drop=False)
            missing = set(base[transaction_col]) - set(source_by_transaction.index)
            if missing:
                raise ValueError(
                    f"Could not align {len(missing)} engineered rows to source transactions"
                )
            aligned_source = source_by_transaction.loc[base[transaction_col]].reset_index(drop=True)

            graph_feats = self.graph_builder.transform(aligned_source).reset_index(drop=True)
            link_feats = link_prediction_features(
                aligned_source,
                self.graph_builder,
                account_col=self.account_col,
                counterparty_col=self.config.columns.get("counterparty_account_id", "counterparty_account_id"),
            ).reset_index(drop=True)
            graph_cols = list(graph_feats.columns) + list(link_feats.columns)
            full = pd.concat([base, graph_feats, link_feats], axis=1)
        else:
            full = base
        full = full.replace([np.inf, -np.inf], np.nan)

        numeric_cols = list(feature_output.numeric_cols) + graph_cols
        categorical_cols = list(feature_output.categorical_cols)
        for col in numeric_cols:
            if col in full.columns:
                full[col] = pd.to_numeric(full[col], errors="coerce").fillna(0.0)
        for col in categorical_cols:
            if col in full.columns:
                full[col] = full[col].fillna("unknown").astype(str)
        return full, {"numeric": numeric_cols, "categorical": categorical_cols, "graph": graph_cols}

    def _score_components(self, frame: pd.DataFrame) -> pd.DataFrame:
        if self.supervised_model is None:
            raise RuntimeError("Supervised model is not initialized.")
        s_supervised = self.supervised_model.predict_proba(frame)

        if self.anomaly_enabled:
            if self.isolation_model is None or self.autoencoder_model is None:
                raise RuntimeError("Anomaly models are not initialized.")
            isolation_raw = self.isolation_model.score(frame[self.numeric_cols])
            autoencoder_raw = self.autoencoder_model.score(frame[self.numeric_cols])
            s_anomaly = 0.5 * isolation_raw + 0.5 * autoencoder_raw
        else:
            s_anomaly = np.zeros(len(frame), dtype=float)

        s_graph = self._compute_graph_score(frame) if self.graph_enabled else np.zeros(len(frame), dtype=float)
        return pd.DataFrame(
            {
                "s_supervised": s_supervised,
                "s_anomaly": s_anomaly,
                "s_graph": s_graph,
            },
            index=frame.index,
        )

    def _attach_risk(self, frame: pd.DataFrame, comp_scores: pd.DataFrame) -> pd.DataFrame:
        risk = self.risk_aggregator.predict(comp_scores)
        out = frame.copy()
        out = pd.concat([out.reset_index(drop=True), comp_scores.reset_index(drop=True), risk.reset_index(drop=True)], axis=1)
        return out

    def _compute_graph_score(self, frame: pd.DataFrame) -> np.ndarray:
        components = []
        if "community_risk" in frame.columns:
            components.append(frame["community_risk"].to_numpy(dtype=float))
        if "graph_pagerank" in frame.columns:
            components.append(frame["graph_pagerank"].to_numpy(dtype=float))
        if "graph_weighted_degree" in frame.columns:
            components.append(np.log1p(frame["graph_weighted_degree"].to_numpy(dtype=float)))
        if "common_neighbors" in frame.columns:
            components.append(frame["common_neighbors"].to_numpy(dtype=float))
        if not components:
            return np.zeros(len(frame), dtype=float)
        stack = np.vstack(components)
        return stack.mean(axis=0)

    def _build_with_history(
        self,
        history_df: pd.DataFrame,
        target_df: pd.DataFrame,
        spec: FeatureSpec,
    ) -> FeatureOutput:
        history = history_df.copy()
        target = target_df.copy()
        history["_is_target"] = 0
        target["_is_target"] = 1

        concat = pd.concat([history, target], ignore_index=True)
        engineered = build_tabular_features(concat, spec)
        frame = engineered.frame[engineered.frame["_is_target"] == 1].copy()
        frame = frame.drop(columns=["_is_target"])
        return FeatureOutput(
            frame=frame,
            numeric_cols=engineered.numeric_cols,
            categorical_cols=engineered.categorical_cols,
            id_cols=engineered.id_cols,
        )

    def _risk_config(self) -> RiskConfig:
        risk = self.config.risk
        weights = risk.get("weights", {})
        costs = risk.get("costs", {})
        initial = risk.get("initial_thresholds", {})
        grid = risk.get("threshold_grid", {})
        return RiskConfig(
            weight_supervised=float(weights.get("supervised", 0.5)),
            weight_anomaly=float(weights.get("anomaly", 0.2)) if self.anomaly_enabled else 0.0,
            weight_graph=float(weights.get("graph", 0.3)) if self.graph_enabled else 0.0,
            c_fn=float(costs.get("c_fn", 604000.0)),
            c_fp=float(costs.get("c_fp", 75.0)),
            c_review=float(costs.get("c_review", 8.0)),
            review_catch_rate=float(costs.get("review_catch_rate", 0.7)),
            low_threshold=float(initial.get("low", 0.2)),
            high_threshold=float(initial.get("high", 0.7)),
            low_points=int(grid.get("low_points", 25)),
            high_points=int(grid.get("high_points", 25)),
        )

    def _check_fitted(self) -> None:
        if (
            self.feature_spec is None
            or self.supervised_model is None
            or self.risk_aggregator is None
            or (self.graph_enabled and self.graph_builder is None)
            or (self.anomaly_enabled and (self.isolation_model is None or self.autoencoder_model is None))
        ):
            raise RuntimeError("Pipeline is not fit yet.")

    def _graph_neighbors(self, account_id: str, limit: int = 10) -> list[dict[str, float | str]]:
        if self.graph_builder is None or self.graph_builder.graph is None:
            return []
        g = self.graph_builder.graph
        if account_id not in g:
            return []

        neighbors: list[dict[str, float | str]] = []
        for nbr, attrs in g[account_id].items():
            neighbors.append(
                {
                    "account_id": str(nbr),
                    "weight": float(attrs.get("weight", 0.0)),
                    "tx_count": float(attrs.get("tx_count", 0.0)),
                    "shared_attrs": float(attrs.get("shared_attrs", 0.0)),
                }
            )
        neighbors.sort(key=lambda x: float(x.get("weight", 0.0)), reverse=True)
        return neighbors[:limit]

    def _history_summary(
        self,
        account_id: str,
        event_ts: pd.Timestamp,
        history_df: pd.DataFrame,
    ) -> dict[str, float | int | None]:
        if (
            self.account_col not in history_df.columns
            or self.timestamp_col not in history_df.columns
            or "amount" not in history_df.columns
        ):
            return {
                "tx_count_30d": 0,
                "avg_amount_30d": 0.0,
                "max_amount_30d": 0.0,
                "lifetime_tx_count": 0,
                "lifetime_fraud_count": None,
            }

        account_hist = history_df[history_df[self.account_col].astype(str) == str(account_id)].copy()
        if account_hist.empty:
            return {
                "tx_count_30d": 0,
                "avg_amount_30d": 0.0,
                "max_amount_30d": 0.0,
                "lifetime_tx_count": 0,
                "lifetime_fraud_count": 0 if self.label_col in history_df.columns else None,
            }

        ts = pd.to_datetime(account_hist[self.timestamp_col], utc=True, errors="coerce", format="mixed")
        account_hist = account_hist.assign(_ts=ts)
        account_hist = account_hist[account_hist["_ts"].notna()]
        if account_hist.empty:
            return {
                "tx_count_30d": 0,
                "avg_amount_30d": 0.0,
                "max_amount_30d": 0.0,
                "lifetime_tx_count": 0,
                "lifetime_fraud_count": 0 if self.label_col in history_df.columns else None,
            }

        prior = account_hist[account_hist["_ts"] <= event_ts]
        win_start = event_ts - pd.Timedelta(days=30)
        recent = prior[prior["_ts"] >= win_start]
        lifetime_fraud_count = None
        if self.label_col in prior.columns:
            lifetime_fraud_count = int(pd.to_numeric(prior[self.label_col], errors="coerce").fillna(0).sum())

        recent_amount = pd.to_numeric(recent["amount"], errors="coerce").fillna(0.0)
        avg_amount_30d = float(recent_amount.mean()) if len(recent_amount) > 0 else 0.0
        max_amount_30d = float(recent_amount.max()) if len(recent_amount) > 0 else 0.0

        return {
            "tx_count_30d": int(len(recent)),
            "avg_amount_30d": avg_amount_30d,
            "max_amount_30d": max_amount_30d,
            "lifetime_tx_count": int(len(prior)),
            "lifetime_fraud_count": lifetime_fraud_count,
        }
