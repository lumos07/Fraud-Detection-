# Fraud Detection Ablation Study

## Design

- Temporal split: 2975 train, 512 validation, 513 test transactions.
- All experiments use the same data, random seed, supervised learner, cost model, and validation-based threshold search.
- Disabled components are neither trained nor used. Graph features are excluded from the supervised feature matrix when graph is disabled.
- Anomaly reconstruction backend: pca (combined 50/50 with Isolation Forest when enabled).
- A: supervised only; B: supervised + anomaly; C: supervised + graph; D: supervised + anomaly + graph.

## Test-set results

| experiment | components | roc_auc | average_precision | precision_flagged | recall_flagged | f1_flagged | specificity_flagged | flagged_rate | total_economic_loss | avg_economic_loss_per_tx |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| A | Supervised only | 0.9939 | 0.5984 | 0.3478 | 1.0000 | 0.5161 | 0.9703 | 0.0448 | 1197.0000 | 2.3333 |
| B | Supervised + anomaly | 0.9955 | 0.7601 | 0.3333 | 1.0000 | 0.5000 | 0.9683 | 0.0468 | 1360.0000 | 2.6511 |
| C | Supervised + graph | 0.9903 | 0.5235 | 0.3182 | 0.8750 | 0.4667 | 0.9703 | 0.0429 | 182925.0000 | 356.5789 |
| D | Supervised + anomaly + graph | 0.9901 | 0.5697 | 0.3077 | 1.0000 | 0.4706 | 0.9644 | 0.0507 | 4702.0000 | 9.1657 |

## Evaluation

- Best flagged F1: experiment A (Supervised only) at 0.5161.
- Best ranking quality (average precision): experiment B (Supervised + anomaly) at 0.7601.
- Lowest modeled economic loss: experiment A (Supervised only) at 1197.00, a 0.0% reduction from A.
- Compared with C, D changed average precision from 0.5235 to 0.5697, flagged F1 from 0.4667 to 0.4706, and modeled loss from 182925.00 to 4702.00.
- Validation metrics are included in `summary_metrics.csv` and `metrics.json`; interpret test results as a single synthetic-data evaluation rather than a confidence interval.
