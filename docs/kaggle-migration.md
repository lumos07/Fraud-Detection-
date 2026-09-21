# Kaggle migration

## Source and reproducibility

Source: [Financial Transactions Dataset for Fraud Detection](https://www.kaggle.com/datasets/aryan208/financial-transactions-dataset-for-fraud-detection), version 1.
The data card describes synthetic transactions. The verified archive contains
5,000,000 unique transactions and 179,553 fraud labels (3.59106%). The timestamp
range is January 2023 to January 2024. Naive timestamps are assumed UTC.
`prepare_kaggle.py` records the archive SHA-256, actual schema, counts, and time range
in `data/kaggle/transactions.manifest.json`. The study records package versions,
configuration, boundaries, sampling, and fraud counts in `run_manifest.json`.
Raw data, caches, and fitted models are excluded from Git.

## Schema

| Source | Internal field | Use |
|---|---|---|
| `sender_account` | `account_id` | Sender identity |
| `receiver_account` | `counterparty_account_id` | Receiver identity |
| `is_fraud` | `label` | Binary outcome, not an input feature |
| `device_hash` | `device_id` | Exact device identity when present |
| `timestamp` | `timestamp` | Chronological ordering |
| `transaction_id`, `amount` | unchanged | Identity and amount |
| `transaction_type`, `merchant_category`, `location`, `device_used`, `payment_channel` | unchanged | Categorical behavior |

Missing required identities, duplicate transaction IDs, invalid timestamps/amounts,
and missing/non-binary training labels fail validation. Inference labels may be
unknown; they are never filled with legitimate outcomes. No invented device,
merchant, email, phone, balance, or account-creation identifiers are used.
`device_used` (e.g. mobile) is a device **type**, not a unique device. The actual
archive has `device_hash` but no exact merchant ID. If `merchant_id` is supplied,
it replaces category nodes in that dataset. `fraud_type`, `ip_address`, and opaque
precomputed scores are excluded from model inputs; their historical provenance
is not established.

## Temporal contract

The study splits chronologically at approximately 60% / 80% of events, keeping
equal timestamps in one partition. Every behavioral window is `[t-window, t)`;
lifetime counts and previous-event lookups are also strictly before `t`.
Same-timestamp events do not see each other. Earlier unlabeled validation/test
transactions can inform later behavior: this models sequential scoring, not an
independent frozen batch. Deterministic features may be generated over the full
sorted stream because each row uses only its strict prefix; scalers, imputers,
encoders, anomaly models, and supervised fitting remain training-only.

Graphs use fixed **30-day**, epoch-anchored snapshots (not calendar months), with
90 days of transaction topology. An event in a snapshot period sees only edges
before that period's start. This deliberately trades freshness for bounded work.
Previously mature fraud knowledge is retained across the edge-window expiry.
`label_available_at`, when supplied, governs maturity; unknown availability remains
unknown. Otherwise `label_maturity` assumes a 7-day confirmation delay. The source
does not contain confirmation timestamps, so this delay is a scenario assumption.
Both transaction and label-availability times must be strictly before the cutoff.

Training graph rows can use earlier mature training outcomes. Validation graphs
use outcomes from training history only; test graphs use supplied training and
validation history only. Target-period labels are masked, including after they
would hypothetically mature. This conservative replay prevents evaluation labels
from silently becoming features. Training excludes labels not mature before
validation starts; early stopping, normalization of ensemble components, and
weight/threshold selection use only validation labels mature before test starts.
Reported validation metrics also include the immature tail, but validation is not
an unbiased holdout because it is used for selection. Test outcomes are evaluation-only.

## Graph construction and scoring

Actual sender–receiver transfers form an undirected graph, aggregated per pair:
count, total/mean amount, first and last transfer time. Self-transfers do not create
neighbor edges. Account-to-attribute edges connect shared devices, locations, and
merchants/categories as requested. Typed attribute nodes avoid account cliques;
only values shared by at least two distinct senders are retained. These attributes
describe observed sender context, not an invented relationship to the receiver.
Attribute edge strength is `attribute_weight / sqrt(number_of_sharing_accounts)`.
Broad location/category nodes are similarity signals, not proof of fraud.

Leiden communities (two iterations by default) use transaction intensity `log1p(count)` plus attribute edges.
Louvain remains selectable with `graph.community_algorithm="louvain"`; the first
full-data profile found Louvain to be the dominant CPU bottleneck. Bounded Leiden
iterations are a computational budget, not a guarantee of an optimal partition.
Community risk is **known fraudulent accounts / total accounts** in the community;
attribute nodes are excluded from both counts. A fraud transaction marks its sender
as known fraud after maturity, not its receiver automatically.

PageRank and direct-neighbor exposure use the transaction graph **only**. Shared
attributes affect communities, not PageRank/exposure. PageRank uses transaction
counts, never fraud labels or amount-times-label weights. Exposure is
`sum(1 / degree(u))` over previously known fraudulent transaction neighbors `u`.
We also calculate fraud-neighbor count/fraction, common neighbors, common fraudulent
neighbors, and fraudulent-neighbor Jaccard overlap. No exact betweenness is computed.

Normalization is causal and explicit:

- `community_risk_normalized = community_risk` (already a ratio).
- `fraud_neighbor_exposure_normalized = exposure / (exposure + exposure_scale)`.
- `pagerank_normalized = pagerank / (pagerank + pagerank_scale / account_count)`.
- Unknown accounts receive zero graph features.

The corrected score is exactly:

```text
interaction = community_risk_normalized * fraud_neighbor_exposure_normalized
graph_risk = w1 * community_risk_normalized
           + w3 * interaction
           + w4 * pagerank_normalized
w1 + w3 + w4 = 1
```

There is **no standalone exposure term** in the combined graph score. The compact
supervised graph subset is `sender_degree`, `receiver_degree`,
`number_of_known_fraud_neighbors`, and **`fraud_neighbor_exposure_normalized`**.
The behavioral `new_receiver_indicator` also remains supervised. PageRank/community
risk are not supervised inputs, although components are still statistically correlated.

## Models, tuning, and evaluation

LightGBM is the default for this profile, with class weighting derived from mature
training class counts. XGBoost is selectable in `supervised.algorithm` and is the
fallback if LightGBM is unavailable. Class-weighted scores and the final normalized
ensemble are **risk scores, not calibrated fraud probabilities**. No new probability
calibrator is implied by percentile normalization.

Isolation Forest and a neural autoencoder see numeric behavioral features only,
not graph/label-derived features. Both fit on a deterministic, configurable sample
of at most 100,000 training transactions. Each score is mapped to its training-score
empirical percentile before configurable anomaly fusion. Raw/percentile scores are
preserved for explanations. This sample cap applies to anomaly fitting, not to the
supervised model or test evaluation.

Graph weights and ensemble simplex weights are searched on mature validation data,
jointly with Low/Medium/High thresholds. The default grid is deliberately coarse
(`graph.weight_grid_divisions=2`, `risk.weight_grid_step=0.5`, 21 threshold points);
initial weights are also candidates. This is validation search, **not random-fold
GridSearchCV** and not a claim of globally optimal weights. Identical supervised and
anomaly components are reused across ablations. Internal graph ablations vary the
standalone graph score; their supervised compact graph inputs are held constant.

Loss assumptions are configurable: low fraud pays FN cost, medium pays review cost
plus expected missed-fraud loss after review, and blocked legitimate transactions
pay FP cost. `fn_cost_mode="amount"` uses `amount * amount_loss_multiplier` instead
of constant FN cost. `max_review_rate` constrains validation review volume, not
guaranteed future volume. `cost_sensitivity.json` retunes validation thresholds over
FN/FP assumptions. A large FN cost can rationally choose to block most transactions;
inspect block rate and cost assumptions rather than presenting that as good ranking.
The search includes the explicit `(low=0, high=0)` all-block endpoint; empty
Low/Medium bands at this endpoint are intentional, not a floating-point accident.
It also includes sentinel thresholds above 1 for exact all-review and all-allow
baselines, so the optimizer never selects a dominated policy merely because rows
at normalized score 1 cannot enter the Low band.

Classification precision/recall/F1/FP/FN count **only high** as positive. Medium still
contributes review costs. Reported ranking metrics include ROC-AUC and Average
Precision; AP, not accuracy, is the primary imbalance-aware ranking summary.
Robustness replays simulate amount drift, legitimate-amount mimicry, behavior
mimicry, and splitting large fraudulent transactions. Legitimate reference
distributions come only from training, never test. These are counterfactual stress
tests, not demonstrated real adversarial behavior; they do not tune the model.
Splitting changes the transaction count, so constant per-transaction FN costs are
not conserved; amount-based loss is preferable when comparing conserved exposure.

## Runtime and API

Behavior uses Polars; graph operations use scipy sparse matrices and igraph.
Snapshot counts, nodes, edges, attribute memberships, timings, and cache hits are
audited. Sparse one-hot encoding, Float32 features, chunked scoring, anomaly fitting
caps, reusable features, and component reuse limit memory/work. Feature caches are
invalidated by input path/mtime, configuration, and feature-code hashes. Full-data
feature preparation still needs memory proportional to row count: this is an
offline/batch implementation, not an out-of-core streaming feature store.
PyTorch defaults to one intra-op thread to avoid a confirmed OpenMP barrier deadlock
with the locally installed LightGBM/PyTorch runtimes on macOS. This is configurable;
do not assume increasing it improves throughput on that environment.

The existing FastAPI service loads the D artifact with `MODEL_PATH`. It accepts
Kaggle aliases on `/score` and `/score/analyst`, and returns transaction IDs so sorted
results are unambiguous. Analyst payloads expose supervised explanations and graph/
anomaly components. Graph snapshots are cached for identical historical state and
cutoff; changed history invalidates the cache. API behavioral features still replay
supplied history, so do not claim millisecond online latency at five-million-row
scale. Production serving needs a persistent historical feature store and scheduled
snapshot refresh. Default saved history is training-only; callers supply additional
observed history explicitly, including only genuinely observable labels.

Run regression checks with `PYTHONPATH=src .venv/bin/python -m unittest discover -s tests -v`.
Tests cover same-time isolation, future-feature invariance, maturity, PageRank label
independence, target-label masking, normalized graph formula, cache invalidation,
high-only metrics, economic-loss optimization, serialization, and API scoring.
