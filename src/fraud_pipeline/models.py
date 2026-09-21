from __future__ import annotations

from dataclasses import dataclass
from typing import Any
import warnings

import numpy as np
import pandas as pd
from scipy import sparse
from sklearn.compose import ColumnTransformer
from sklearn.decomposition import PCA
from sklearn.ensemble import GradientBoostingClassifier, IsolationForest
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

try:
    # pyrefly: ignore [missing-import]
    import lightgbm as lgb
except Exception:  # pragma: no cover
    lgb = None

try:
    # pyrefly: ignore [missing-import]
    import xgboost as xgb
except Exception:  # pragma: no cover
    xgb = None

try:
    # pyrefly: ignore [missing-import]
    import shap
except Exception:  # pragma: no cover
    shap = None

try:
    import torch
    # pyrefly: ignore [missing-import]
    import torch.nn as nn
    import torch.optim as optim
    from torch.utils.data import DataLoader, TensorDataset
except Exception:  # pragma: no cover
    torch = None
    nn = None
    optim = None
    DataLoader = None
    TensorDataset = None


@dataclass
class SupervisedResult:
    train_scores: np.ndarray
    validation_scores: np.ndarray


class SupervisedModel:
    def __init__(
        self,
        algorithm: str = "lightgbm",
        params: dict[str, Any] | None = None,
        early_stopping_rounds: int = 100,
    ) -> None:
        self.algorithm = algorithm.lower()
        self.params = params or {}
        self.early_stopping_rounds = early_stopping_rounds

        self.preprocessor: ColumnTransformer | None = None
        self.model: Any | None = None
        self.numeric_cols: list[str] = []
        self.categorical_cols: list[str] = []
        self.transformed_feature_names: list[str] = []

    def fit(
        self,
        train_df: pd.DataFrame,
        y_train: pd.Series,
        validation_df: pd.DataFrame,
        y_validation: pd.Series,
        numeric_cols: list[str],
        categorical_cols: list[str],
        return_training_scores: bool = True,
    ) -> SupervisedResult:
        self.numeric_cols = list(numeric_cols)
        self.categorical_cols = list(categorical_cols)

        num_pipe = Pipeline(
            steps=[
                ("imputer", SimpleImputer(strategy="median")),
            ]
        )
        cat_pipe = Pipeline(
            steps=[
                ("imputer", SimpleImputer(strategy="most_frequent")),
                ("onehot", OneHotEncoder(handle_unknown="ignore", dtype=np.float32)),
            ]
        )
        self.preprocessor = ColumnTransformer(
            transformers=[
                ("num", num_pipe, self.numeric_cols),
                ("cat", cat_pipe, self.categorical_cols),
            ],
            remainder="drop",
            sparse_threshold=1.0,
        )

        x_train = self.preprocessor.fit_transform(train_df)
        x_validation = self.preprocessor.transform(validation_df)
        try:
            self.transformed_feature_names = list(self.preprocessor.get_feature_names_out())
        except Exception:
            self.transformed_feature_names = [f"f_{i}" for i in range(int(x_train.shape[1]))]

        x_train = self._format_matrix_for_model(x_train)
        x_validation = self._format_matrix_for_model(x_validation)
        self.model = self._build_model(y_train)
        self._fit_model(x_train, y_train.to_numpy(), x_validation, y_validation.to_numpy())

        train_scores = self.predict_proba(train_df) if return_training_scores else np.empty(0)
        validation_scores = self.predict_proba(validation_df)
        return SupervisedResult(train_scores=train_scores, validation_scores=validation_scores)

    def _build_model(self, y_train: pd.Series) -> Any:
        negative = int((y_train == 0).sum())
        positive = int((y_train == 1).sum())
        pos_weight = float(negative / max(positive, 1))

        if self.algorithm == "lightgbm" and lgb is not None:
            params = {
                "learning_rate": 0.05,
                "n_estimators": 1000,
                "max_depth": 8,
                "num_leaves": 63,
                "subsample": 0.8,
                "colsample_bytree": 0.8,
                "reg_lambda": 1.0,
                "random_state": 42,
                "n_jobs": -1,
                "verbosity": -1,
            }
            params.update(self.params)
            params.setdefault("scale_pos_weight", pos_weight)
            return lgb.LGBMClassifier(**params)

        use_xgb_fallback = self.algorithm == "lightgbm" and lgb is None and xgb is not None
        if (self.algorithm in {"xgboost", "xgb"} or use_xgb_fallback) and xgb is not None:
            if use_xgb_fallback:
                warnings.warn("LightGBM is unavailable; using XGBoost with compatible parameters", RuntimeWarning)
            params = {
                "learning_rate": 0.05,
                "n_estimators": 800,
                "max_depth": 8,
                "subsample": 0.8,
                "colsample_bytree": 0.8,
                "reg_lambda": 1.0,
                "random_state": 42,
                "n_jobs": -1,
                "objective": "binary:logistic",
                "eval_metric": "auc",
            }
            params.update(self.params)
            params.pop("num_leaves", None)
            if params.get("verbosity", 1) < 0:
                params["verbosity"] = 0
            params.setdefault("scale_pos_weight", pos_weight)
            params.setdefault("early_stopping_rounds", self.early_stopping_rounds)
            return xgb.XGBClassifier(**params)

        params = {
            "learning_rate": 0.05,
            "n_estimators": 300,
            "max_depth": 4,
            "random_state": 42,
        }
        params.update({k: v for k, v in self.params.items() if k in params})
        return GradientBoostingClassifier(**params)

    def _fit_model(
        self,
        x_train: Any,
        y_train: np.ndarray,
        x_validation: Any,
        y_validation: np.ndarray,
    ) -> None:
        if self.model is None:
            raise RuntimeError("Model has not been initialized.")

        if lgb is not None and isinstance(self.model, lgb.LGBMClassifier):
            self.model.fit(
                x_train,
                y_train,
                eval_set=[(x_validation, y_validation)],
                eval_metric="auc",
                callbacks=[lgb.early_stopping(self.early_stopping_rounds, verbose=False)],
            )
            return

        if xgb is not None and isinstance(self.model, xgb.XGBClassifier):
            self.model.fit(
                x_train,
                y_train,
                eval_set=[(x_validation, y_validation)],
                verbose=False,
            )
            return

        positive = max(int((y_train == 1).sum()), 1)
        weights = np.where(y_train == 1, (y_train == 0).sum() / positive, 1.0)
        self.model.fit(x_train, y_train, sample_weight=weights)

    def predict_proba(self, data: pd.DataFrame) -> np.ndarray:
        if self.model is None or self.preprocessor is None:
            raise RuntimeError("Supervised model must be fit before prediction.")
        x = self.preprocessor.transform(data)
        x = self._format_matrix_for_model(x)
        if hasattr(self.model, "predict_proba"):
            return self.model.predict_proba(x)[:, 1]

        raw = self.model.decision_function(x)
        return 1.0 / (1.0 + np.exp(-raw))

    def explain(self, data: pd.DataFrame, top_k: int = 5) -> list[list[dict[str, float | str]]]:
        if self.model is None or self.preprocessor is None:
            raise RuntimeError("Supervised model must be fit before explanation.")
        if top_k < 1:
            top_k = 1

        x = self.preprocessor.transform(data)
        x = self._format_matrix_for_model(x)
        feature_names = self._feature_names()
        contributions = self._feature_contributions(x)

        max_k = min(top_k, contributions.shape[1])
        results: list[list[dict[str, float | str]]] = []
        for row in contributions:
            idx = np.argsort(np.abs(row))[::-1][:max_k]
            explanation = [
                {
                    "feature": str(feature_names[i]),
                    "contribution": float(row[i]),
                }
                for i in idx
            ]
            results.append(explanation)
        return results

    def _feature_names(self) -> np.ndarray:
        if self.preprocessor is None:
            return np.array([])
        if self.transformed_feature_names:
            return np.asarray(self.transformed_feature_names, dtype=object)
        n = len(self.numeric_cols) + len(self.categorical_cols)
        return np.asarray([f"f_{i}" for i in range(n)], dtype=object)

    def _feature_contributions(self, x: Any) -> np.ndarray:
        contrib = self._shap_contributions(x)
        if contrib is not None:
            return contrib
        return self._fallback_contributions(x)

    def _shap_contributions(self, x: Any) -> np.ndarray | None:
        # Native tree SHAP avoids compatibility failures between independently
        # updated SHAP and boosting-library releases. The final column is bias.
        try:
            if lgb is not None and isinstance(self.model, lgb.LGBMClassifier):
                values = self.model.predict(x, pred_contrib=True)
                values = values.toarray() if sparse.issparse(values) else np.asarray(values)
                return values[:, :-1]
            if xgb is not None and isinstance(self.model, xgb.XGBClassifier):
                values = self.model.get_booster().predict(xgb.DMatrix(x), pred_contribs=True)
                return np.asarray(values)[:, :-1]
        except Exception:
            pass
        if shap is None or self.model is None:
            return None
        try:
            explainer = shap.TreeExplainer(self.model)
            with warnings.catch_warnings():
                warnings.filterwarnings(
                    "ignore",
                    message="LightGBM binary classifier with TreeExplainer shap values output has changed*",
                    category=UserWarning,
                )
                values = explainer.shap_values(x)
            if isinstance(values, list):
                arr = np.asarray(values[1] if len(values) > 1 else values[0], dtype=float)
            else:
                arr = np.asarray(values, dtype=float)
            if arr.ndim == 3:
                # multiclass-like shape: [n, features, classes]
                if arr.shape[2] > 1:
                    arr = arr[:, :, 1]
                else:
                    arr = arr[:, :, 0]
            if arr.ndim != 2:
                return None
            return arr
        except Exception:
            return None

    def _fallback_contributions(self, x: Any) -> np.ndarray:
        if sparse.issparse(x):
            dense = x.toarray()
        else:
            dense = np.asarray(x, dtype=float)

        if dense.ndim == 1:
            dense = dense.reshape(1, -1)

        baseline = np.nanmean(dense, axis=0, keepdims=True)
        centered = dense - baseline

        importances = None
        if hasattr(self.model, "feature_importances_"):
            importances = np.asarray(getattr(self.model, "feature_importances_"), dtype=float)
        elif hasattr(self.model, "coef_"):
            coef = np.asarray(getattr(self.model, "coef_"), dtype=float)
            importances = np.abs(coef).ravel()

        if importances is None or importances.size == 0:
            importances = np.ones(centered.shape[1], dtype=float)
        if importances.size != centered.shape[1]:
            importances = np.resize(importances, centered.shape[1])

        return centered * importances.reshape(1, -1)

    def _format_matrix_for_model(self, x: Any) -> Any:
        use_named_frame = False
        if self.model is not None and lgb is not None and isinstance(self.model, lgb.LGBMClassifier):
            use_named_frame = True
        elif self.model is None and self.algorithm == "lightgbm" and lgb is not None:
            use_named_frame = True

        if use_named_frame and not sparse.issparse(x):
            return _to_feature_frame(x, self.transformed_feature_names)
        return x


class IsolationForestScorer:
    def __init__(
        self,
        n_estimators: int = 200,
        contamination: float = 0.005,
        random_state: int = 42,
    ) -> None:
        self.scaler = StandardScaler()
        self.model = IsolationForest(
            n_estimators=n_estimators,
            contamination=contamination,
            random_state=random_state,
            n_jobs=-1,
        )

    def fit(self, x: pd.DataFrame | np.ndarray) -> "IsolationForestScorer":
        arr = self.scaler.fit_transform(_to_numpy(x))
        self.model.fit(arr)
        return self

    def score(self, x: pd.DataFrame | np.ndarray) -> np.ndarray:
        arr = self.scaler.transform(_to_numpy(x))
        return -self.model.score_samples(arr)


class _TorchAutoencoder(nn.Module):  # pragma: no cover
    def __init__(self, input_dim: int, hidden_dims: list[int]):
        super().__init__()
        if hidden_dims[0] != input_dim:
            hidden_dims = [input_dim] + hidden_dims
        encoder = []
        for in_d, out_d in zip(hidden_dims[:-1], hidden_dims[1:]):
            encoder.append(nn.Linear(in_d, out_d))
            encoder.append(nn.ReLU())
        decoder = []
        rev = list(reversed(hidden_dims))
        for in_d, out_d in zip(rev[:-1], rev[1:]):
            decoder.append(nn.Linear(in_d, out_d))
            if out_d != input_dim:
                decoder.append(nn.ReLU())
        self.encoder = nn.Sequential(*encoder[:-1] if encoder else [])
        self.decoder = nn.Sequential(*decoder)

    def forward(self, x):
        z = self.encoder(x)
        recon = self.decoder(z)
        return recon


class AutoencoderScorer:
    def __init__(
        self,
        latent_dim: int = 8,
        hidden_dims: list[int] | None = None,
        epochs: int = 30,
        batch_size: int = 1024,
        learning_rate: float = 1e-3,
        random_state: int = 42,
        backend: str = "auto",
    ) -> None:
        self.latent_dim = latent_dim
        self.hidden_dims = hidden_dims or [64, 32, 8, 32, 64]
        self.epochs = epochs
        self.batch_size = batch_size
        self.learning_rate = learning_rate
        self.random_state = random_state

        backend = backend.lower()
        if backend not in {"auto", "torch", "pca"}:
            raise ValueError("Autoencoder backend must be one of: auto, torch, pca")
        if backend == "torch" and torch is None:
            raise ModuleNotFoundError("Torch backend requested, but torch is not installed")

        self.scaler = StandardScaler()
        self.backend = ("torch" if torch is not None else "pca") if backend == "auto" else backend
        self.model = None

    def fit(self, x: pd.DataFrame | np.ndarray) -> "AutoencoderScorer":
        arr = self.scaler.fit_transform(_to_numpy(x).astype(np.float32))
        if arr.shape[1] < 2:
            self.backend = "pca"

        if self.backend == "torch":
            self.model = self._fit_torch(arr)
        else:
            n_components = int(max(1, min(self.latent_dim, arr.shape[1] - 1)))
            self.model = PCA(n_components=n_components, random_state=self.random_state)
            self.model.fit(arr)
        return self

    def score(self, x: pd.DataFrame | np.ndarray) -> np.ndarray:
        arr = self.scaler.transform(_to_numpy(x).astype(np.float32))
        if self.backend == "torch":
            model: _TorchAutoencoder = self.model
            model.eval()
            with torch.no_grad():
                tensor = torch.from_numpy(arr)
                recon = model(tensor).cpu().numpy()
            return np.mean((arr - recon) ** 2, axis=1)

        model: PCA = self.model
        projected = model.transform(arr)
        recon = model.inverse_transform(projected)
        return np.mean((arr - recon) ** 2, axis=1)

    def _fit_torch(self, arr: np.ndarray):  # pragma: no cover
        torch.manual_seed(self.random_state)
        input_dim = arr.shape[1]

        hidden_dims = [h for h in self.hidden_dims if h > 0]
        if not hidden_dims:
            hidden_dims = [max(self.latent_dim, 4)]
        if hidden_dims[-1] != self.latent_dim:
            hidden_dims[-1] = self.latent_dim

        model = _TorchAutoencoder(input_dim=input_dim, hidden_dims=hidden_dims)
        optimizer = optim.Adam(model.parameters(), lr=self.learning_rate)
        criterion = nn.MSELoss()

        dataset = TensorDataset(torch.from_numpy(arr), torch.from_numpy(arr))
        loader = DataLoader(dataset, batch_size=self.batch_size, shuffle=True)

        model.train()
        for _ in range(self.epochs):
            for x_batch, y_batch in loader:
                optimizer.zero_grad()
                recon = model(x_batch)
                loss = criterion(recon, y_batch)
                loss.backward()
                optimizer.step()
        return model


def _to_numpy(x: pd.DataFrame | np.ndarray) -> np.ndarray:
    if isinstance(x, pd.DataFrame):
        return x.to_numpy(dtype=np.float32)
    return np.asarray(x, dtype=np.float32)


def _to_feature_frame(x: Any, feature_names: list[str]) -> pd.DataFrame:
    if isinstance(x, pd.DataFrame):
        return x
    if sparse.issparse(x):
        return pd.DataFrame.sparse.from_spmatrix(x, columns=feature_names)
    arr = np.asarray(x)
    if arr.ndim == 1:
        arr = arr.reshape(-1, 1)
    return pd.DataFrame(arr, columns=feature_names[: arr.shape[1]])
