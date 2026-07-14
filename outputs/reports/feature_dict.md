# Feature dictionary

7 core + 3 interaction features (plus native categoricals). Every feature uses planning-time information only; target-derived statistics are fit on the training split to prevent leakage.

| Feature | Kind | Definition & motivation |
|---|---|---|
| `weather_severity` | core | Normalized weighted composite of low visibility, wind, precipitation and thunder in [0,1]. Primary meteorological driver of delay. |
| `dep_hour_sin / dep_hour_cos` | core | Cyclical (sin/cos) encoding of the scheduled departure hour so 23:00 and 00:00 are adjacent. |
| `is_peak` | core | 1 if the departure hour is in a configured peak window (morning/evening). Interacts non-linearly with weather and congestion. |
| `time_bucket` | core (categorical) | Coarse part-of-day label (early/morning_peak/midday/evening_peak/night). |
| `airport_congestion` | core | Scheduled departures at the origin within +/-30 min, min-max normalized with train-only bounds. Planning information only -> leakage-free. |
| `carrier_ontime_rate` | core | Historical on-time rate of the carrier, fit on the TRAIN split only (per-fold in CV) with a global-mean fallback -> no target leakage. |
| `distance / sched_duration` | core | Great-circle-style leg distance (km) and scheduled block time (min). |
| `day_of_week / is_weekend / is_holiday` | core (calendar) | Calendar effects on demand and delay propagation. |
| `prev_leg_delay` | core | Departure delay of the previous leg of the same tail (0 for the first leg). The main delay-propagation signal. |
| `wsev_x_peak` | interaction | weather_severity x is_peak: bad weather hurts far more during peaks. |
| `cong_x_peak` | interaction | airport_congestion x is_peak: congestion bites hardest at peak times. |
| `prevdelay_x_slack` | interaction | prev_leg_delay x turnaround_slack: slack (scheduled turnaround minus the minimum) absorbs inherited delay. |
| `airline / origin / dest / aircraft_type` | categorical | Handled natively by LightGBM (one-hot for the RandomForest baseline). |
