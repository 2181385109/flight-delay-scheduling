"""Feature engineering for real flight data -- leakage-safe by construction.

Three families of features, each with an explicit leakage policy:

1. **Planning-time features** (weather forecast, schedule, distance, calendar).
   Known arbitrarily far ahead; no leakage risk.

2. **Network-state features** (``airport_delay_state``, ``carrier_delay_state``).
   These summarise *realized* delays of flights that have **already departed**,
   and are the strongest real-world signal -- a backed-up airport stays backed
   up. They are strictly causal: a flight scheduled at ``t`` may only see
   departures before ``t - prediction_horizon_min``. That horizon (default 60
   min) is the operational assumption of the whole model: *we predict one hour
   out, knowing what has happened so far.*

3. **Historical target encodings** (carrier / route / tail / origin-hour delay
   rates). These aggregate the label, so they are fit on **training rows only**
   (re-fit inside every CV fold) and smoothed toward the global mean so rare
   keys cannot memorise their own outcome.

The public entry point is :func:`build_features`.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from flightopt.config import Config

# ---------------------------------------------------------------------------
# Feature name groups.
# ---------------------------------------------------------------------------
PLANNING_FEATURES: list[str] = [
    "weather_severity", "vis", "wind", "precip", "temp", "humid", "wind_gust", "pressure",
    "dep_hour_sin", "dep_hour_cos", "is_peak",
    "airport_congestion", "distance", "sched_duration",
    "is_weekend", "is_holiday", "turnaround_slack", "leg_index",
]
NETWORK_FEATURES: list[str] = [
    "airport_delay_state", "airport_recent_flights", "carrier_delay_state", "prev_leg_delay",
]
ENCODED_FEATURES: list[str] = [
    "carrier_delay_rate", "route_delay_rate", "tail_delay_rate", "origin_hour_delay_rate",
]
INTERACTION_FEATURES: list[str] = [
    "wsev_x_peak", "cong_x_peak", "prevdelay_x_slack",
]
CATEGORICAL_FEATURES: list[str] = [
    "airline", "origin", "dest", "aircraft_type", "time_bucket", "day_of_week",
]
NUMERIC_FEATURES: list[str] = PLANNING_FEATURES + NETWORK_FEATURES + ENCODED_FEATURES
FEATURE_COLUMNS: list[str] = NUMERIC_FEATURES + INTERACTION_FEATURES + CATEGORICAL_FEATURES

REG_TARGET = "dep_delay"
CLF_TARGET = "is_delayed15"

#: Features that change when a flight is moved to a different slot (used by the
#: scheduler to re-score candidate offsets).
OFFSET_DEPENDENT: list[str] = [
    "dep_hour_sin", "dep_hour_cos", "is_peak", "time_bucket",
    "airport_congestion", "origin_hour_delay_rate", "wsev_x_peak", "cong_x_peak",
]


# ---------------------------------------------------------------------------
# Pure transforms (shared with the loader and the scheduler).
# ---------------------------------------------------------------------------
def weather_severity_from_raw(vis, wind, precip, thunder, weather_cfg, weights) -> np.ndarray:
    """Composite weather severity in ``[0, 1]`` (higher = worse)."""
    vis_min, vis_max = float(weather_cfg["vis_min"]), float(weather_cfg["vis_max"])
    low_vis = np.clip((vis_max - np.asarray(vis, float)) / (vis_max - vis_min), 0.0, 1.0)
    wind_n = np.clip(np.asarray(wind, float) / float(weather_cfg["wind_max"]), 0.0, 1.0)
    precip_n = np.clip(np.asarray(precip, float) / float(weather_cfg["precip_max"]), 0.0, 1.0)
    w = np.array([weights["low_vis"], weights["wind"], weights["precip"], weights["thunder"]])
    w = w / w.sum()
    sev = w[0] * low_vis + w[1] * wind_n + w[2] * precip_n + w[3] * np.asarray(thunder, float)
    return np.clip(sev, 0.0, 1.0)


def is_peak_hour(hour: np.ndarray, peak_windows: list[list[int]]) -> np.ndarray:
    hour = np.asarray(hour)
    peak = np.zeros(hour.shape, dtype=int)
    for start, end in peak_windows:
        peak |= ((hour >= start) & (hour < end)).astype(int)
    return peak


def cyclical_hour(hour: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    rad = 2.0 * np.pi * np.asarray(hour, float) / 24.0
    return np.sin(rad), np.cos(rad)


def time_bucket(hour: np.ndarray, buckets: dict[str, list[int]]) -> np.ndarray:
    hour = np.asarray(hour)
    labels = np.array(["night"] * len(hour), dtype=object)
    for name, (start, end) in buckets.items():
        labels[(hour >= start) & (hour < end)] = name
    return labels


def compute_airport_congestion(origin, dep_minute, window_min: int) -> np.ndarray:
    """Scheduled departures at the same origin within +/- ``window_min``.

    Planning information only (uses scheduled times, never realized delays).
    """
    df = pd.DataFrame({"o": np.asarray(origin), "m": np.asarray(dep_minute, float)})
    counts = np.zeros(len(df))
    for _, idx in df.groupby("o").groups.items():
        idx = np.asarray(list(idx))
        m = df.loc[idx, "m"].to_numpy()
        order = np.argsort(m, kind="stable")
        ms = m[order]
        lo = np.searchsorted(ms, ms - window_min, side="left")
        hi = np.searchsorted(ms, ms + window_min, side="right")
        counts[idx[order]] = (hi - lo).astype(float)
    return counts


def lagged_group_state(
    keys: np.ndarray,
    minutes: np.ndarray,
    values: np.ndarray,
    horizon_min: int,
    window_min: int,
) -> tuple[np.ndarray, np.ndarray]:
    """Mean and count of ``values`` over same-key rows that occurred in
    ``[t - horizon - window, t - horizon)``.

    Strictly causal: nothing at or after ``t - horizon`` can influence a row, so
    this is exactly the information available at the prediction horizon.
    """
    n = len(minutes)
    out_mean = np.zeros(n)
    out_cnt = np.zeros(n)
    df = pd.DataFrame({"k": np.asarray(keys), "t": np.asarray(minutes, float)})
    vals = np.asarray(values, float)
    for _, idx in df.groupby("k").groups.items():
        idx = np.asarray(list(idx))
        t = df.loc[idx, "t"].to_numpy()
        order = np.argsort(t, kind="stable")
        ts, vs, ids = t[order], vals[idx][order], idx[order]
        csum = np.concatenate([[0.0], np.cumsum(vs)])
        hi = np.searchsorted(ts, ts - horizon_min, side="left")
        lo = np.searchsorted(ts, ts - horizon_min - window_min, side="left")
        cnt = (hi - lo).astype(float)
        total = csum[hi] - csum[lo]
        out_mean[ids] = np.where(cnt > 0, total / np.maximum(cnt, 1.0), 0.0)
        out_cnt[ids] = cnt
    return out_mean, out_cnt


def _fit_target_encoding(
    df: pd.DataFrame, keys: list[str], target: str, prior: float = 20.0
) -> tuple[dict, float]:
    """Smoothed target encoding: rare keys shrink toward the global mean."""
    global_mean = float(df[target].mean())
    g = df.groupby(keys, observed=True)[target].agg(["mean", "count"])
    smoothed = (g["mean"] * g["count"] + global_mean * prior) / (g["count"] + prior)
    return smoothed.to_dict(), global_mean


def _apply_target_encoding(df: pd.DataFrame, keys: list[str], mapping: dict, default: float):
    if len(keys) == 1:
        idx = df[keys[0]]
    else:
        idx = pd.MultiIndex.from_frame(df[keys])
    return pd.Series(idx.map(mapping), index=df.index).astype(float).fillna(default)


# ---------------------------------------------------------------------------
# Full feature builder.
# ---------------------------------------------------------------------------
def build_features(
    df: pd.DataFrame,
    cfg: Config,
    fit_stats: dict | None = None,
) -> tuple[pd.DataFrame, pd.Series | None, pd.Series | None, dict]:
    """Build the model matrix from unified-schema flight records.

    ``fit_stats=None`` fits the leakage-sensitive statistics on ``df`` (training
    mode) and returns them; passing stored stats applies them without ever
    touching ``df``'s labels (validation / test / serving mode).
    """
    df = df.reset_index(drop=True)
    fitting = fit_stats is None
    stats: dict = {} if fitting else dict(fit_stats)
    fcfg, dcfg = cfg.features, cfg.data

    dep = pd.to_datetime(df["sched_dep"])
    arr = pd.to_datetime(df["sched_arr"])
    hour = dep.dt.hour.to_numpy()
    abs_min = (dep - dep.dt.normalize().min()).dt.total_seconds().to_numpy() / 60.0
    day_min = (dep.dt.hour * 60 + dep.dt.minute).to_numpy()

    out = pd.DataFrame(index=df.index)

    # --- 1) planning-time -------------------------------------------------
    out["weather_severity"] = df["weather_severity"].astype(float)
    for c in ("vis", "wind", "precip", "temp", "humid", "wind_gust", "pressure"):
        out[c] = df[c].astype(float)
    sin_h, cos_h = cyclical_hour(hour)
    out["dep_hour_sin"], out["dep_hour_cos"] = sin_h, cos_h
    is_peak = is_peak_hour(hour, fcfg.peak_windows)
    out["is_peak"] = is_peak
    out["time_bucket"] = time_bucket(hour, fcfg.time_buckets)

    raw_cong = compute_airport_congestion(df["origin"], day_min, fcfg.congestion_window_min)
    if fitting:
        stats["congestion_min"] = float(raw_cong.min())
        stats["congestion_max"] = float(raw_cong.max())
    denom = max(stats["congestion_max"] - stats["congestion_min"], 1e-9)
    out["airport_congestion"] = np.clip((raw_cong - stats["congestion_min"]) / denom, 0.0, 1.0)

    out["distance"] = df["distance"].astype(float)
    out["sched_duration"] = (arr - dep).dt.total_seconds().to_numpy() / 60.0
    dow = dep.dt.dayofweek
    out["day_of_week"] = dow.astype(int)
    out["is_weekend"] = (dow >= 5).astype(int)
    holidays = set(pd.to_datetime(fcfg.holidays).date) if fcfg.holidays else set()
    out["is_holiday"] = dep.dt.date.map(lambda d: int(d in holidays)).astype(int)
    out["turnaround_slack"] = (
        df["sched_turnaround_min"].astype(float) - df["min_turnaround"].astype(float)
    ).fillna(0.0)
    out["leg_index"] = df["leg_index"].astype(int)

    # --- 2) network state (strictly lagged by the prediction horizon) -----
    delay = df[REG_TARGET].astype(float).to_numpy() if REG_TARGET in df else np.zeros(len(df))
    ap_mean, ap_cnt = lagged_group_state(
        df["origin"].to_numpy(), abs_min, delay,
        dcfg.prediction_horizon_min, dcfg.network_window_min,
    )
    ca_mean, _ = lagged_group_state(
        df["airline"].to_numpy(), abs_min, delay,
        dcfg.prediction_horizon_min, dcfg.network_window_min,
    )
    out["airport_delay_state"] = ap_mean
    out["airport_recent_flights"] = ap_cnt
    out["carrier_delay_state"] = ca_mean
    out["prev_leg_delay"] = df["prev_leg_delay"].astype(float)

    # --- 3) historical target encodings (train-only) ----------------------
    enc_specs = {
        "carrier_delay_rate": ["airline"],
        "route_delay_rate": ["origin", "dest"],
        "tail_delay_rate": ["tail_id"],
        "origin_hour_delay_rate": ["origin", "_hour"],
    }
    work = df.copy()
    work["_hour"] = hour
    if fitting:
        if CLF_TARGET in work.columns:
            stats["encodings"] = {
                name: _fit_target_encoding(work, keys, CLF_TARGET)
                for name, keys in enc_specs.items()
            }
        else:
            stats["encodings"] = {name: ({}, 0.0) for name in enc_specs}
    for name, keys in enc_specs.items():
        mapping, default = stats["encodings"][name]
        out[name] = _apply_target_encoding(work, keys, mapping, default)

    # --- interactions -----------------------------------------------------
    out["wsev_x_peak"] = out["weather_severity"] * out["is_peak"]
    out["cong_x_peak"] = out["airport_congestion"] * out["is_peak"]
    out["prevdelay_x_slack"] = out["prev_leg_delay"] * out["turnaround_slack"]

    # --- categoricals -----------------------------------------------------
    for c in ("airline", "origin", "dest", "aircraft_type"):
        out[c] = df[c].astype(str)
    if fitting:
        stats["categories"] = {
            c: sorted(out[c].astype(str).unique().tolist())
            for c in ("airline", "origin", "dest", "aircraft_type", "time_bucket")
        }
        stats["categories"]["day_of_week"] = list(range(7))
    for c, cats in stats["categories"].items():
        out[c] = pd.Categorical(out[c], categories=cats)

    X = out[FEATURE_COLUMNS]
    y_reg = df[REG_TARGET].astype(float) if REG_TARGET in df.columns else None
    y_clf = df[CLF_TARGET].astype(int) if CLF_TARGET in df.columns else None
    return X, y_reg, y_clf, stats


def one_hot_for_baseline(X: pd.DataFrame) -> pd.DataFrame:
    """One-hot encode categoricals for the RandomForest baseline.

    ``tail_id``-scale cardinality is avoided: only the low-cardinality
    categoricals are expanded.
    """
    return pd.get_dummies(X, columns=CATEGORICAL_FEATURES, dummy_na=False)


# Human-readable definition + motivation for each feature (the "feature dict").
FEATURE_DICT: list[tuple[str, str, str]] = [
    ("weather_severity", "planning",
     "Normalized composite of low visibility, wind and precipitation in [0,1]."),
    ("vis / wind / precip / temp / humid / wind_gust / pressure", "planning",
     "Raw hourly weather observed at the origin airport. Temperature/humidity/"
     "pressure matter for winter icing and de-icing delays."),
    ("dep_hour_sin / dep_hour_cos / is_peak / time_bucket", "planning",
     "Cyclical departure-hour encoding plus peak-window flag and part-of-day."),
    ("airport_congestion", "planning",
     "Scheduled departures at the origin within +/-30 min, min-max normalized "
     "with train-only bounds. Uses the timetable only, never realized delays."),
    ("distance / sched_duration", "planning", "Leg distance (km) and scheduled block time."),
    ("day_of_week / is_weekend / is_holiday", "planning",
     "Calendar effects, using the real US federal holiday list for the data year."),
    ("turnaround_slack / leg_index", "planning",
     "Scheduled turnaround minus the minimum, and how many legs the aircraft has "
     "already flown that day."),
    ("airport_delay_state", "network state",
     "Mean realized departure delay at this airport over a 3-hour window ending "
     "one hour before scheduled departure. The strongest real signal: a "
     "backed-up airport stays backed up. Strictly causal."),
    ("airport_recent_flights", "network state",
     "How many departures fed that window -- distinguishes a quiet airport from "
     "a busy one with the same mean delay."),
    ("carrier_delay_state", "network state",
     "Same lagged statistic for the carrier's whole network, capturing "
     "airline-wide disruption."),
    ("prev_leg_delay", "network state",
     "Departure delay of the same aircraft's previous leg that day (0 for the "
     "first leg) -- direct delay propagation."),
    ("carrier_delay_rate / route_delay_rate / tail_delay_rate / origin_hour_delay_rate",
     "target encoding",
     "Historical P(delay>15) by carrier, route, aircraft and origin-hour. Fit on "
     "TRAIN rows only (re-fit per CV fold) and smoothed toward the global mean "
     "so rare keys cannot memorise their own label."),
    ("wsev_x_peak / cong_x_peak / prevdelay_x_slack", "interaction",
     "Weather and congestion bite hardest at peak; slack absorbs inherited delay."),
    ("airline / origin / dest / aircraft_type", "categorical",
     "Handled natively by LightGBM (one-hot for the RandomForest baseline)."),
]


def write_feature_dict(path) -> None:
    """Write ``feature_dict.md`` documenting every feature and its leakage policy."""
    lines = [
        "# Feature dictionary",
        "",
        "Features are grouped by **leakage policy**, which is the thing that "
        "matters most on real data:",
        "",
        "* **planning** — known well ahead of departure (timetable, forecast, calendar).",
        "* **network state** — realized delays of flights that already departed, "
        "strictly lagged by the prediction horizon (default 60 min), i.e. exactly "
        "what an operator knows one hour out.",
        "* **target encoding** — aggregates the label, therefore fit on TRAINING "
        "rows only (re-fit inside each CV fold) and smoothed toward the global mean.",
        "",
        "| Feature | Kind | Definition & motivation |",
        "|---|---|---|",
    ]
    for name, kind, desc in FEATURE_DICT:
        lines.append(f"| `{name}` | {kind} | {desc} |")
    lines.append("")
    with open(path, "w", encoding="utf-8") as fh:
        fh.write("\n".join(lines))
