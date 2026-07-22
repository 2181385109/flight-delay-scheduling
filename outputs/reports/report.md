# Results Report

## Headline metrics

| Metric | Definition | Result | Target | Baseline | Status |
|---|---|---|---|---|---|
| High-risk capture (Recall) | L4/L5 recall vs `is_delayed15` (test) | **0.664** (P=0.344, F1=0.453) | ≥ 0.80 | rule 0.161 | MISS — reaching 0.80 needs 0.58 of flights flagged |
| Constraint satisfaction | satisfied / total constraints (CP-SAT) | **0.997** (cap.viol 77→5) | ≥ 0.85 | greedy 0.981 | PASS |
| High-risk delay reduction | mean Δ predicted delay/flight (CP-SAT) | **0.48 min** (11.7→11.2; all-flights total predicted delay −4.07%) | ≥ 1.0 min | greedy 0.45 | MISS |
| Prediction error (MAE) | test-set MAE, minutes | **12.62** | lower is better | median 13.82 / RF 19.00 / mean 19.90 | PASS |

## Prediction detail (held-out test split)

| Model | MAE | RMSE |
|---|---|---|
| lightgbm | 12.622 | 33.517 |
| random_forest | 19.002 | 32.471 |
| median_baseline | 13.821 | 37.834 |
| mean_baseline | 19.900 | 35.222 |

| Model | Precision | Recall | F1 | PR-AUC | ROC-AUC |
|---|---|---|---|---|---|
| lightgbm | 0.665 | 0.229 | 0.340 | 0.486 | 0.727 |
| random_forest | 0.687 | 0.198 | 0.308 | 0.482 | 0.741 |
| rule_baseline | 0.648 | 0.161 | 0.258 | n/a | n/a |

## New-tail generalization (GroupKFold on tail_id)

- MAE: 12.881 ± 0.117
- PR-AUC: 0.498 ± 0.014

## Top regressor features (gain)

- `dest`: 0.178
- `aircraft_type`: 0.108
- `airport_delay_state`: 0.078
- `tail_delay_rate`: 0.073
- `carrier_delay_state`: 0.071
- `origin_hour_delay_rate`: 0.046
- `airport_recent_flights`: 0.039
- `humid`: 0.038

## Scheduling: CP-SAT vs greedy vs before

Re-timing one real operating day (**2013-03-14**, 981 flights). Baselines: the original as-flown schedule (`before`) and a greedy local search. Delay figures are reductions in *predicted* delay.

| Solver | High-risk mean delay (min) | Δ/flight | Total delay (all, min) | Total Δ | Satisfaction | Capacity violations |
|---|---|---|---|---|---|---|
| before | 11.67 | - | 5852.8 | - | 0.947 | 77 |
| cpsat | 11.19 | 0.48 | 5614.6 | −4.07% | 0.997 | 5 |
| greedy | 11.21 | 0.45 | 5630.3 | −3.80% | 0.981 | 28 |
