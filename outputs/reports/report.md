# Results Report

## Headline metrics

| Metric | Definition | Result | Target | Baseline | Status |
|---|---|---|---|---|---|
| High-risk capture (Recall) | L4/L5 recall vs `is_delayed15` (test) | **0.829** (P=0.586, F1=0.687) | ≥ 0.80 | rule 0.431 | PASS |
| Constraint satisfaction | satisfied / total constraints (CP-SAT) | **0.998** (cap.viol 59→7) | ≥ 0.85 | greedy 0.997 | PASS |
| High-risk delay reduction | mean Δdelay/flight (CP-SAT) | **3.04 min** (17.3→14.2) | ≥ 1.0 min | greedy 2.82 | PASS |
| Prediction error (MAE) | test-set MAE, minutes | **3.74** | lower is better | RF 3.98 / mean 6.47 | PASS |

## Prediction detail (held-out test split)

| Model | MAE | RMSE |
|---|---|---|
| lightgbm | 3.745 | 4.686 |
| random_forest | 3.979 | 4.897 |
| mean_baseline | 6.466 | 7.942 |

| Model | Precision | Recall | F1 | PR-AUC | ROC-AUC |
|---|---|---|---|---|---|
| lightgbm | 0.743 | 0.610 | 0.670 | 0.774 | 0.868 |
| random_forest | 0.725 | 0.537 | 0.617 | 0.726 | 0.854 |
| rule_baseline | 0.828 | 0.431 | 0.567 | n/a | n/a |

## New-tail generalization (GroupKFold on tail_id)

- MAE: 4.653 ± 0.226
- PR-AUC: 0.769 ± 0.034

## Top regressor features (gain)

- `wsev_x_peak`: 0.234
- `prev_leg_delay`: 0.206
- `prevdelay_x_slack`: 0.124
- `weather_severity`: 0.103
- `cong_x_peak`: 0.102
- `airport_congestion`: 0.099
- `turnaround_slack`: 0.036
- `origin`: 0.018

## Scheduling: CP-SAT vs greedy vs before

| Solver | High-risk mean delay | Δ reduction | Satisfaction | Capacity violations |
|---|---|---|---|---|
| before | 17.27 | - | 0.985 | 59 |
| cpsat | 14.23 | 3.04 | 0.998 | 7 |
| greedy | 14.46 | 2.82 | 0.997 | 10 |
