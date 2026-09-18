# Failure of AI-Based Early Warning Systems in Financial Fraud

Operational starter implementation for an adaptive hybrid fraud pipeline with:

- Temporal data splits (leakage-safe)
- Tabular behavioral features
- Graph-derived risk features
- Supervised + anomaly models
- Cost-aware score fusion and risk bands
- Drift/adversarial simulations
- FastAPI scoring service

## 1) Quick Start

```bash
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
```

## 2) Expected Input Schema

Input data can be `.csv` or `.parquet` with at least:

- `transaction_id`
- `account_id`
- `timestamp` (ISO datetime)
- `amount`
- `label` (`0/1`)

Recommended optional columns:

- `counterparty_account_id`
- `merchant_category`
- `channel`
- `device_id`
- `ip_address`
- `phone`
- `email`
- `balance`
- `account_created_at`

## 3) Train End-to-End

```bash
python scripts/train_pipeline.py ^
  --input data/transactions.parquet ^
  --config configs/default.json ^
  --train-end 2025-06-30 ^
  --val-end 2025-07-31 ^
  --output-dir artifacts
```

Outputs:

- `artifacts/pipeline.joblib`
- `artifacts/metrics.json`
- `artifacts/predictions_validation.csv`
- `artifacts/predictions_test.csv`

## 3.1) Train On Your Real Dataset

If you have your own transaction export, place it as `.csv` or `.parquet` and run:

```bash
python scripts/train_pipeline.py ^
  --input data/your_transactions.parquet ^
  --config configs/default.json ^
  --train-end 2025-06-30 ^
  --val-end 2025-07-31 ^
  --output-dir artifacts_real
```

Minimum required columns:

- `transaction_id`
- `account_id`
- `timestamp`
- `amount`
- `label` (`0/1`)

Optional columns can be missing (`device_id`, `channel`, `merchant_category`, etc.); safe defaults are applied.

## 4) Run Drift/Adversarial Simulations

```bash
python scripts/run_simulations.py ^
  --input data/transactions.parquet ^
  --pipeline artifacts/pipeline.joblib ^
  --train-end 2025-06-30 ^
  --val-end 2025-07-31 ^
  --output artifacts/simulation_results.json
```

## 5) Serve Risk API

```bash
set MODEL_PATH=artifacts/pipeline.joblib
python scripts/serve_api.py --host 0.0.0.0 --port 8000
```

Health check:

```bash
curl http://localhost:8000/health
```

Open dashboard:

```bash
http://localhost:8000/dashboard
```

Scoring with analyst payload for medium-risk transactions:

```bash
curl -X POST http://localhost:8000/score ^
  -H "Content-Type: application/json" ^
  -d "{\"transactions\":[{\"transaction_id\":\"T1\",\"account_id\":\"A1\",\"timestamp\":\"2025-08-01T10:00:00Z\",\"amount\":1200,\"merchant_category\":\"electronics\",\"channel\":\"web\",\"device_id\":\"D1\",\"ip_address\":\"10.0.0.1\"}],\"include_analyst_payload\":true,\"top_k\":5}"
```

Dedicated analyst endpoint with strict schema and low-latency mode:

```bash
curl -X POST http://localhost:8000/score/analyst ^
  -H "Content-Type: application/json" ^
  -d "{\"transactions\":[{\"transaction_id\":\"T1\",\"account_id\":\"A1\",\"timestamp\":\"2025-08-01T10:00:00Z\",\"amount\":1200,\"merchant_category\":\"electronics\",\"channel\":\"web\"}],\"top_k\":5,\"neighbor_limit\":10,\"low_latency_mode\":true}"
```

`low_latency_mode=true` disables SHAP feature explanations and returns faster analyst payloads.

## 6) Project Layout

```text
configs/
scripts/
src/fraud_pipeline/
```

## 7) Notes

- Main optimization target is economic loss:
  - `Expected_Loss = C_FN * FN + C_FP * FP`
- Risk bands:
  - Low: `< low_threshold`
  - Medium: `[low_threshold, high_threshold)`
  - High: `>= high_threshold`
- If `lightgbm` is unavailable, supervised training falls back to `xgboost`, then to sklearn GBM.
