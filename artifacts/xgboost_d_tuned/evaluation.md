# GridSearchCV-Tuned XGBoost Experiment D

## Search design

- Search method: GridSearchCV
- Cross-validation: three-fold forward-chaining TimeSeriesSplit
- Objective: average precision
- Candidates: 96
- Total fits: 288
- Training rows: 2,975, including 62 fraud rows
- Best CV average precision: 0.6515
- Search time: 49.2 seconds, excluding feature construction

## Best parameters

```json
{
  "colsample_bytree": 0.8,
  "learning_rate": 0.03,
  "max_depth": 3,
  "min_child_weight": 1,
  "n_estimators": 400,
  "reg_lambda": 1.0,
  "subsample": 1.0
}
```

The final model used validation early stopping and stopped at iteration 270.

## Corrected test comparison

| Metric | Untuned XGBoost D | Tuned XGBoost D |
| --- | ---: | ---: |
| Supervised ROC AUC | 0.9946 | 0.9960 |
| Supervised average precision | 0.7406 | 0.8464 |
| Final ROC AUC | 0.9921 | 0.9891 |
| Final average precision | 0.6334 | 0.4834 |
| Flagged precision | 0.0415 | 0.1250 |
| Flagged recall | 1.0000 | 1.0000 |
| Flagged F1 | 0.0796 | 0.2222 |
| False positives | 185 | 56 |
| Economic loss | 2,686 | 3,463 |

The search substantially improved XGBoost's supervised ranking and thresholded precision/F1.
The final blended score did not improve because anomaly and graph scores interact with the
supervised score during percentile normalization and weighting. Economic loss rose because
45 legitimate test transactions entered the high/block band, compared with 18 for the
untuned model, even though total false positives fell.

## Leakage limitation

The CV folds are chronological, but the current graph features are precomputed from the full
training-period graph. Consequently, this is not yet a fully point-in-time-safe graph CV.
After incremental historical graph features are implemented, this search should be rerun before
using the selected parameters as a production conclusion.
