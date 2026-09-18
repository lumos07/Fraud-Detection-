from __future__ import annotations

import os
from pathlib import Path
from typing import Any

import pandas as pd
from fastapi import FastAPI, HTTPException
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, ConfigDict, Field

from fraud_pipeline.pipeline import FraudPipeline


class ScoreRequest(BaseModel):
    transactions: list[dict[str, Any]]
    history: list[dict[str, Any]] | None = None
    include_analyst_payload: bool = False
    top_k: int = 5
    neighbor_limit: int = 10
    low_latency_mode: bool = False
    include_graph_neighbors: bool = True
    include_recent_history: bool = True


class AnalystTransaction(BaseModel):
    model_config = ConfigDict(extra="forbid")

    transaction_id: str
    account_id: str
    timestamp: str
    amount: float = Field(ge=0)
    counterparty_account_id: str | None = None
    merchant_category: str | None = None
    channel: str | None = None
    device_id: str | None = None
    ip_address: str | None = None
    phone: str | None = None
    email: str | None = None
    balance: float | None = None
    account_created_at: str | None = None
    label: int | None = None


class AnalystScoreRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    transactions: list[AnalystTransaction]
    history: list[AnalystTransaction] | None = None
    top_k: int = Field(default=5, ge=1, le=20)
    neighbor_limit: int = Field(default=10, ge=1, le=50)
    include_explanations: bool = True
    include_graph_neighbors: bool = True
    include_recent_history: bool = True
    low_latency_mode: bool = False


app = FastAPI(title="Fraud Risk API", version="0.1.0")
BASE_DIR = Path(__file__).resolve().parent
WEB_DIR = BASE_DIR / "web"
ARTIFACTS_DIR = Path(os.environ.get("ARTIFACTS_DIR", "artifacts"))

app.mount(
    "/artifacts",
    StaticFiles(directory=str(ARTIFACTS_DIR), check_dir=False),
    name="artifacts",
)

PIPELINE: FraudPipeline | None = None


@app.on_event("startup")
def startup_event() -> None:
    global PIPELINE
    model_path = os.environ.get("MODEL_PATH", "artifacts/pipeline.joblib")
    if not os.path.exists(model_path):
        PIPELINE = None
        return
    PIPELINE = FraudPipeline.load(model_path)


@app.get("/health")
def health() -> dict[str, Any]:
    return {"status": "ok", "model_loaded": PIPELINE is not None}


@app.get("/", include_in_schema=False)
def root() -> dict[str, Any]:
    return {"message": "Fraud Risk API is running.", "dashboard": "/dashboard", "health": "/health"}


@app.get("/dashboard", response_class=HTMLResponse, include_in_schema=False)
def dashboard() -> HTMLResponse:
    page_path = WEB_DIR / "dashboard.html"
    if not page_path.exists():
        return HTMLResponse(
            "<h1>Dashboard not found</h1><p>Create src/fraud_pipeline/web/dashboard.html.</p>",
            status_code=404,
        )
    return HTMLResponse(page_path.read_text(encoding="utf-8"))


@app.get("/dashboard/data", include_in_schema=False)
def dashboard_data() -> dict[str, Any]:
    data = {
        "health": {"model_loaded": PIPELINE is not None},
        "metrics": _load_json_file(ARTIFACTS_DIR / "metrics.json"),
        "simulation_results": _load_json_file(ARTIFACTS_DIR / "simulation_results.json"),
        "metadata": _load_json_file(ARTIFACTS_DIR / "metadata.json"),
        "validation_preview": _load_prediction_preview(ARTIFACTS_DIR / "predictions_validation.csv", limit=15),
        "test_preview": _load_prediction_preview(ARTIFACTS_DIR / "predictions_test.csv", limit=15),
        "validation_risk_bands": _load_risk_band_counts(ARTIFACTS_DIR / "predictions_validation.csv"),
        "test_risk_bands": _load_risk_band_counts(ARTIFACTS_DIR / "predictions_test.csv"),
        "artifacts": _list_artifacts(ARTIFACTS_DIR),
    }
    return _sanitize_json(data)


def _require_pipeline() -> FraudPipeline:
    if PIPELINE is None:
        raise HTTPException(status_code=503, detail="Model not loaded. Set MODEL_PATH.")
    return PIPELINE


@app.post("/score")
def score(request: ScoreRequest) -> dict[str, Any]:
    pipeline = _require_pipeline()
    if not request.transactions:
        raise HTTPException(status_code=400, detail="No transactions provided.")
    if request.top_k < 1:
        raise HTTPException(status_code=400, detail="top_k must be >= 1")
    if request.neighbor_limit < 1:
        raise HTTPException(status_code=400, detail="neighbor_limit must be >= 1")

    df = pd.DataFrame(request.transactions)
    history_df = pd.DataFrame(request.history) if request.history else None
    scored = pipeline.predict(df, history_df=history_df)

    if request.include_analyst_payload:
        payload = pipeline.analyst_payload(
            scored_df=scored,
            history_df=history_df,
            top_k=request.top_k,
            neighbor_limit=request.neighbor_limit,
            include_explanations=(not request.low_latency_mode),
            include_graph_neighbors=request.include_graph_neighbors,
            include_recent_history=request.include_recent_history,
        )
        scored["analyst_payload"] = [
            p if band == "medium" else None
            for p, band in zip(payload, scored["risk_band"].tolist())
        ]

    cols = ["final_score", "risk_band", "s_supervised", "s_anomaly", "s_graph"]
    if request.include_analyst_payload:
        cols.append("analyst_payload")
    cols = [c for c in cols if c in scored.columns]
    records = scored[cols].to_dict(orient="records")
    return {"n": len(records), "scores": records}


@app.post("/score/analyst")
def score_analyst(request: AnalystScoreRequest) -> dict[str, Any]:
    pipeline = _require_pipeline()
    if not request.transactions:
        raise HTTPException(status_code=400, detail="No transactions provided.")

    tx_records = [item.model_dump() for item in request.transactions]
    history_records = [item.model_dump() for item in request.history] if request.history else None

    df = pd.DataFrame(tx_records)
    history_df = pd.DataFrame(history_records) if history_records else None
    scored = pipeline.predict(df, history_df=history_df)

    payload = pipeline.analyst_payload(
        scored_df=scored,
        history_df=history_df,
        top_k=request.top_k,
        neighbor_limit=request.neighbor_limit,
        include_explanations=(request.include_explanations and not request.low_latency_mode),
        include_graph_neighbors=request.include_graph_neighbors,
        include_recent_history=request.include_recent_history,
    )
    scored["analyst_payload"] = payload

    cols = [
        "transaction_id",
        "account_id",
        "final_score",
        "risk_band",
        "s_supervised",
        "s_anomaly",
        "s_graph",
        "analyst_payload",
    ]
    cols = [c for c in cols if c in scored.columns]
    return {
        "n": int(len(scored)),
        "low_latency_mode": bool(request.low_latency_mode),
        "scores": scored[cols].to_dict(orient="records"),
    }


def _load_json_file(path: Path) -> dict[str, Any] | None:
    if not path.exists():
        return None
    try:
        import json

        with path.open("r", encoding="utf-8") as f:
            return json.load(f)
    except Exception:
        return None


def _load_prediction_preview(path: Path, limit: int = 15) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    try:
        frame = pd.read_csv(path, nrows=limit)
    except Exception:
        return []

    preferred = [
        "transaction_id",
        "account_id",
        "timestamp",
        "amount",
        "label",
        "final_score",
        "risk_band",
        "s_supervised",
        "s_anomaly",
        "s_graph",
    ]
    cols = [c for c in preferred if c in frame.columns]
    if not cols:
        cols = frame.columns[: min(10, len(frame.columns))].tolist()
    preview = frame[cols].copy()
    return preview.to_dict(orient="records")


def _load_risk_band_counts(path: Path) -> dict[str, int]:
    if not path.exists():
        return {}
    try:
        frame = pd.read_csv(path, usecols=["risk_band"])
    except Exception:
        return {}
    if "risk_band" not in frame.columns:
        return {}
    counts = frame["risk_band"].fillna("unknown").value_counts().to_dict()
    return {str(k): int(v) for k, v in counts.items()}


def _list_artifacts(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    items = []
    for child in sorted(path.glob("*")):
        if not child.is_file():
            continue
        stat = child.stat()
        items.append(
            {
                "name": child.name,
                "size_bytes": int(stat.st_size),
                "modified_utc": pd.Timestamp(stat.st_mtime, unit="s", tz="UTC").isoformat(),
                "link": f"/artifacts/{child.name}",
            }
        )
    return items


def _sanitize_json(value: Any) -> Any:
    if isinstance(value, dict):
        return {k: _sanitize_json(v) for k, v in value.items()}
    if isinstance(value, list):
        return [_sanitize_json(v) for v in value]
    if isinstance(value, float):
        if pd.isna(value) or value in {float("inf"), float("-inf")}:
            return None
        return value
    return value
