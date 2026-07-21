# Feature dictionary

Features are grouped by **leakage policy**, which is the thing that matters most on real data:

* **planning** — known well ahead of departure (timetable, forecast, calendar).
* **network state** — realized delays of flights that already departed, strictly lagged by the prediction horizon (default 60 min), i.e. exactly what an operator knows one hour out.
* **target encoding** — aggregates the label, therefore fit on TRAINING rows only (re-fit inside each CV fold) and smoothed toward the global mean.

| Feature | Kind | Definition & motivation |
|---|---|---|
| `weather_severity` | planning | Normalized composite of low visibility, wind and precipitation in [0,1]. |
| `vis / wind / precip / temp / humid / wind_gust / pressure` | planning | Raw hourly weather observed at the origin airport. Temperature/humidity/pressure matter for winter icing and de-icing delays. |
| `dep_hour_sin / dep_hour_cos / is_peak / time_bucket` | planning | Cyclical departure-hour encoding plus peak-window flag and part-of-day. |
| `airport_congestion` | planning | Scheduled departures at the origin within +/-30 min, min-max normalized with train-only bounds. Uses the timetable only, never realized delays. |
| `distance / sched_duration` | planning | Leg distance (km) and scheduled block time. |
| `day_of_week / is_weekend / is_holiday` | planning | Calendar effects, using the real US federal holiday list for the data year. |
| `turnaround_slack / leg_index` | planning | Scheduled turnaround minus the minimum, and how many legs the aircraft has already flown that day. |
| `airport_delay_state` | network state | Mean realized departure delay at this airport over a 3-hour window ending one hour before scheduled departure. The strongest real signal: a backed-up airport stays backed up. Strictly causal. |
| `airport_recent_flights` | network state | How many departures fed that window -- distinguishes a quiet airport from a busy one with the same mean delay. |
| `carrier_delay_state` | network state | Same lagged statistic for the carrier's whole network, capturing airline-wide disruption. |
| `prev_leg_delay` | network state | Departure delay of the same aircraft's previous leg that day (0 for the first leg) -- direct delay propagation. |
| `carrier_delay_rate / route_delay_rate / tail_delay_rate / origin_hour_delay_rate` | target encoding | Historical P(delay>15) by carrier, route, aircraft and origin-hour. Fit on TRAIN rows only (re-fit per CV fold) and smoothed toward the global mean so rare keys cannot memorise their own label. |
| `wsev_x_peak / cong_x_peak / prevdelay_x_slack` | interaction | Weather and congestion bite hardest at peak; slack absorbs inherited delay. |
| `airline / origin / dest / aircraft_type` | categorical | Handled natively by LightGBM (one-hot for the RandomForest baseline). |
