"""Feature engineering: 7 core + 3 interaction features, leakage-safe.

The public entry point is :func:`build_features`.  A handful of *pure*
transforms (weather severity, congestion, peak flag, cyclical time) are exposed
individually so that both the synthetic data generator (label construction) and
the scheduler (re-scoring flights at shifted slots) reuse **exactly the same**
definitions -- guaranteeing consistency across the predict -> schedule loop.

Leakage policy
--------------
* ``carrier_ontime_rate`` is a target-derived statistic.  It is fit on the
  training rows only and stored in ``stats``; validation/test rows reuse the
  stored mapping (with a global-mean fallback for unseen carriers).
* ``airport_congestion`` is min-max normalized with train-only bounds.
* Only planning-time information (schedule, distance, weather forecast) enters
  the features -- never the realized ``dep_delay``.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd

# ---------------------------------------------------------------------------
# Feature name groups (referenced by predict / viz).
# ---------------------------------------------------------------------------
NUMERIC_FEATURES: list[str] = [
    "weather_severity",
    "dep_hour_sin",
    "dep_hour_cos",
    "is_peak",
    "airport_congestion",
    "carrier_ontime_rate",
    "distance",
    "sched_duration",
    "is_weekend",
    "is_holiday",
    "prev_leg_delay",
    "turnaround_slack",
]
INTERACTION_FEATURES: list[str] = [
    "wsev_x_peak",
    "cong_x_peak",
    "prevdelay_x_slack",
]
CATEGORICAL_FEATURES: list[str] = [
    "airline",
    "origin",
    "dest",
    "aircraft_type",
    "time_bucket",
    "day_of_week",
]
FEATURE_COLUMNS: list[str] = NUMERIC_FEATURES + INTERACTION_FEATURES + CATEGORICAL_FEATURES

REG_TARGET = "dep_delay"
CLF_TARGET = "is_delayed15"


# ---------------------------------------------------------------------------
# Pure transforms (shared with synth.py and schedule.py).
# ---------------------------------------------------------------------------
def weather_severity_from_raw(
    vis: np.ndarray,
    wind: np.ndarray,
    precip: np.ndarray,
    thunder: np.ndarray,
    weather_cfg: dict[str, float],
    weights: dict[str, float],
) -> np.ndarray:
    """Composite weather severity in ``[0, 1]`` (higher = worse)."""
    vis_min = float(weather_cfg["vis_min"])
    vis_max = float(weather_cfg["vis_max"])
    wind_max = float(weather_cfg["wind_max"])
    precip_max = float(weather_cfg["precip_max"])

    low_vis = np.clip((vis_max - np.asarray(vis, float)) / (vis_max - vis_min), 0.0, 1.0)
    wind_n = np.clip(np.asarray(wind, float) / wind_max, 0.0, 1.0)
    precip_n = np.clip(np.asarray(precip, float) / precip_max, 0.0, 1.0)
    thunder_n = np.asarray(thunder, float)

    w = np.array(
        [weights["low_vis"], weights["wind"], weights["precip"], weights["thunder"]],
        dtype=float,
    )
    w = w / w.sum()
    severity = w[0] * low_vis + w[1] * wind_n + w[2] * precip_n + w[3] * thunder_n
    return np.clip(severity, 0.0, 1.0)


def is_peak_hour(hour: np.ndarray, peak_windows: list[list[int]]) -> np.ndarray:
    """Return 1 where ``hour`` falls in any ``[start, end)`` peak window."""
    hour = np.asarray(hour)
    peak = np.zeros(hour.shape, dtype=int)
    for start, end in peak_windows:
        peak |= ((hour >= start) & (hour < end)).astype(int)
    return peak


def cyclical_hour(hour: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Cyclical (sin, cos) encoding of the departure hour."""
    radians = 2.0 * np.pi * np.asarray(hour, float) / 24.0
    return np.sin(radians), np.cos(radians)


def time_bucket(hour: np.ndarray, buckets: dict[str, list[int]]) -> np.ndarray:
    """Map each hour to a coarse bucket label (config-driven)."""
    hour = np.asarray(hour)
    labels = np.array(["night"] * len(hour), dtype=object)
    # Order matters only for overlapping ranges; config ranges are disjoint.
    for name, (start, end) in buckets.items():
        labels[(hour >= start) & (hour < end)] = name
    return labels


def compute_airport_congestion(
    origin: pd.Series,
    dep_minute: np.ndarray,
    window_min: int,
) -> np.ndarray:
    """Scheduled departures at the same origin within +/- ``window_min`` minutes.

    Uses only planning information (scheduled departure times), so it is
    leakage-free.  Returns the raw count including the flight itself.
    """
    df = pd.DataFrame({"origin": np.asarray(origin), "m": np.asarray(dep_minute, float)})
    counts = np.zeros(len(df), dtype=float)
    for _, idx in df.groupby("origin").groups.items():
        idx = np.asarray(list(idx))
        minutes = df.loc[idx, "m"].to_numpy()
        order = np.argsort(minutes)
        sorted_m = minutes[order]
        lo = np.searchsorted(sorted_m, sorted_m - window_min, side="left")
        hi = np.searchsorted(sorted_m, sorted_m + window_min, side="right")
        counts[idx[order]] = (hi - lo).astype(float)
    return counts


# ---------------------------------------------------------------------------
# Full feature builder.
# ---------------------------------------------------------------------------
def build_features(
    df: pd.DataFrame,
    *,
    weather_cfg: dict[str, float],
    weather_weights: dict[str, float],
    features_cfg: Any,
    fit_stats: dict | None = None,
) -> tuple[pd.DataFrame, pd.Series | None, pd.Series | None, dict]:
    """Build the model feature matrix from a unified-schema DataFrame.

    Parameters
    ----------
    df:
        Flight records following the schema produced by ``synth.generate``.
    weather_cfg, weather_weights:
        Sub-configs needed to recompute weather severity.
    features_cfg:
        ``FeaturesConfig`` instance (peak windows, holidays, buckets, window).
    fit_stats:
        When ``None`` the target-derived / normalization statistics are fit on
        ``df`` (train mode) and returned.  When provided, the stored statistics
        are applied (validation / test mode) with no peeking at ``df`` labels.

    Returns
    -------
    (X, y_reg, y_clf, stats)
        ``y_reg`` / ``y_clf`` are ``None`` if the target columns are absent.
    """
    df = df.copy()
    fitting = fit_stats is None
    stats: dict = {} if fitting else dict(fit_stats)

    dep = pd.to_datetime(df["sched_dep"])
    hour = dep.dt.hour.to_numpy()
    dep_minute = (dep.dt.hour * 60 + dep.dt.minute).to_numpy()

    out = pd.DataFrame(index=df.index)

    # 1) Weather severity ----------------------------------------------------
    out["weather_severity"] = weather_severity_from_raw(
        df["vis"].to_numpy(),
        df["wind"].to_numpy(),
        df["precip"].to_numpy(),
        df["thunder"].to_numpy(),
        weather_cfg,
        weather_weights,
    )

    # 2) Time encoding (cyclical + peak + bucket) ----------------------------
    sin_h, cos_h = cyclical_hour(hour)
    out["dep_hour_sin"] = sin_h
    out["dep_hour_cos"] = cos_h
    is_peak = is_peak_hour(hour, features_cfg.peak_windows)
    out["is_peak"] = is_peak
    out["time_bucket"] = time_bucket(hour, features_cfg.time_buckets)

    # 3) Airport congestion (min-max normalized with train bounds) -----------
    raw_cong = compute_airport_congestion(
        df["origin"], dep_minute, features_cfg.congestion_window_min
    )
    if fitting:
        c_min, c_max = float(raw_cong.min()), float(raw_cong.max())
        stats["congestion_min"] = c_min
        stats["congestion_max"] = c_max
    else:
        c_min = stats["congestion_min"]
        c_max = stats["congestion_max"]
    denom = max(c_max - c_min, 1e-9)
    out["airport_congestion"] = np.clip((raw_cong - c_min) / denom, 0.0, 1.0)

    # 4) Carrier on-time rate (target-derived; train-only) -------------------
    if fitting:
        if CLF_TARGET in df.columns:
            ontime = 1.0 - df.groupby("airline")[CLF_TARGET].mean()
            global_rate = float(1.0 - df[CLF_TARGET].mean())
            stats["carrier_ontime_rate"] = ontime.to_dict()
            stats["carrier_ontime_global"] = global_rate
        else:  # no labels -> neutral prior
            stats["carrier_ontime_rate"] = {}
            stats["carrier_ontime_global"] = 0.5
    mapping = stats.get("carrier_ontime_rate", {})
    global_rate = stats.get("carrier_ontime_global", 0.5)
    out["carrier_ontime_rate"] = (
        df["airline"].map(mapping).fillna(global_rate).astype(float)
    )

    # 5) Distance + scheduled duration ---------------------------------------
    arr = pd.to_datetime(df["sched_arr"])
    out["distance"] = df["distance"].astype(float)
    out["sched_duration"] = (arr - dep).dt.total_seconds().to_numpy() / 60.0

    # 6) Calendar ------------------------------------------------------------
    dow = dep.dt.dayofweek
    out["day_of_week"] = dow.astype(int)
    out["is_weekend"] = (dow >= 5).astype(int)
    holidays = set(pd.to_datetime(features_cfg.holidays).date) if features_cfg.holidays else set()
    out["is_holiday"] = dep.dt.date.map(lambda d: int(d in holidays)).astype(int)

    # 7) Delay propagation + turnaround slack --------------------------------
    out["prev_leg_delay"] = df["prev_leg_delay"].astype(float)
    sched_turn = df.get("sched_turnaround_min")
    if sched_turn is None:
        slack = pd.Series(0.0, index=df.index)
    else:
        slack = (sched_turn.astype(float) - df["min_turnaround"].astype(float)).fillna(0.0)
    out["turnaround_slack"] = slack.to_numpy()

    # Interaction features ---------------------------------------------------
    out["wsev_x_peak"] = out["weather_severity"] * out["is_peak"]
    out["cong_x_peak"] = out["airport_congestion"] * out["is_peak"]
    out["prevdelay_x_slack"] = out["prev_leg_delay"] * out["turnaround_slack"]

    # Categorical passthrough ------------------------------------------------
    out["airline"] = df["airline"].astype(str)
    out["origin"] = df["origin"].astype(str)
    out["dest"] = df["dest"].astype(str)
    out["aircraft_type"] = df["aircraft_type"].astype(str)

    # Consistent category dtypes (train defines the category universe) -------
    if fitting:
        stats["categories"] = {
            col: sorted(out[col].astype(str).unique().tolist())
            for col in ("airline", "origin", "dest", "aircraft_type", "time_bucket")
        }
        stats["categories"]["day_of_week"] = list(range(7))
    for col, cats in stats["categories"].items():
        out[col] = pd.Categorical(out[col], categories=cats)

    X = out[FEATURE_COLUMNS]

    y_reg = df[REG_TARGET].astype(float) if REG_TARGET in df.columns else None
    y_clf = df[CLF_TARGET].astype(int) if CLF_TARGET in df.columns else None

    return X, y_reg, y_clf, stats


def one_hot_for_baseline(X: pd.DataFrame) -> pd.DataFrame:
    """One-hot encode categorical columns for the RandomForest baseline."""
    return pd.get_dummies(X, columns=CATEGORICAL_FEATURES, dummy_na=False)


# Human-readable definition + motivation for each feature (the "feature dict").
FEATURE_DICT: list[tuple[str, str, str]] = [
    ("weather_severity", "core",
     "Normalized weighted composite of low visibility, wind, precipitation and "
     "thunder in [0,1]. Primary meteorological driver of delay."),
    ("dep_hour_sin / dep_hour_cos", "core",
     "Cyclical (sin/cos) encoding of the scheduled departure hour so 23:00 and "
     "00:00 are adjacent."),
    ("is_peak", "core",
     "1 if the departure hour is in a configured peak window (morning/evening). "
     "Interacts non-linearly with weather and congestion."),
    ("time_bucket", "core (categorical)",
     "Coarse part-of-day label (early/morning_peak/midday/evening_peak/night)."),
    ("airport_congestion", "core",
     "Scheduled departures at the origin within +/-30 min, min-max normalized "
     "with train-only bounds. Planning information only -> leakage-free."),
    ("carrier_ontime_rate", "core",
     "Historical on-time rate of the carrier, fit on the TRAIN split only "
     "(per-fold in CV) with a global-mean fallback -> no target leakage."),
    ("distance / sched_duration", "core",
     "Great-circle-style leg distance (km) and scheduled block time (min)."),
    ("day_of_week / is_weekend / is_holiday", "core (calendar)",
     "Calendar effects on demand and delay propagation."),
    ("prev_leg_delay", "core",
     "Departure delay of the previous leg of the same tail (0 for the first "
     "leg). The main delay-propagation signal."),
    ("wsev_x_peak", "interaction",
     "weather_severity x is_peak: bad weather hurts far more during peaks."),
    ("cong_x_peak", "interaction",
     "airport_congestion x is_peak: congestion bites hardest at peak times."),
    ("prevdelay_x_slack", "interaction",
     "prev_leg_delay x turnaround_slack: slack (scheduled turnaround minus the "
     "minimum) absorbs inherited delay."),
    ("airline / origin / dest / aircraft_type", "categorical",
     "Handled natively by LightGBM (one-hot for the RandomForest baseline)."),
]


def write_feature_dict(path) -> None:
    """Write ``feature_dict.md`` documenting every feature's definition/motivation."""
    lines = [
        "# Feature dictionary",
        "",
        "7 core + 3 interaction features (plus native categoricals). Every feature "
        "uses planning-time information only; target-derived statistics are fit on "
        "the training split to prevent leakage.",
        "",
        "| Feature | Kind | Definition & motivation |",
        "|---|---|---|",
    ]
    for name, kind, desc in FEATURE_DICT:
        lines.append(f"| `{name}` | {kind} | {desc} |")
    lines.append("")
    with open(path, "w", encoding="utf-8") as fh:
        fh.write("\n".join(lines))
