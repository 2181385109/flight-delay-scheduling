"""Synthetic flight generator (primary data source).

Generates ~2000 flight records with a *deliberately injected causal structure*
so that the engineered features are genuinely predictive of the label and the
three headline metrics are reproducible.  Key properties:

* Aircraft (``tail_id``) fly 2-4 chained legs -> supports **delay propagation**.
* Per-airport, per-hour weather evolves as a smooth AR(1) field.
* ``dep_delay`` follows the documented additive causal model (weather,
  congestion, peak, a non-linear weather*peak interaction, propagation, noise).
* Fully deterministic given ``config.yaml``'s ``seed``.

The generator reuses the *same* weather-severity / congestion / peak transforms
as :mod:`flightopt.features`, so labels and features never drift apart.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from flightopt.config import Config
from flightopt.features import (
    compute_airport_congestion,
    is_peak_hour,
    weather_severity_from_raw,
)

TAXI_MINUTES = 25.0  # fixed taxi-out + climb + descent + taxi-in allowance


def _airport_distance_matrix(rng: np.random.Generator, n: int, cfg: Config) -> np.ndarray:
    """Symmetric distance matrix (km) from random 2-D airport coordinates."""
    coords = rng.uniform(0.0, 1.0, size=(n, 2))
    diff = coords[:, None, :] - coords[None, :, :]
    dist = np.sqrt((diff**2).sum(axis=-1))
    dn = (dist - dist.min()) / (dist.max() - dist.min() + 1e-9)
    return cfg.synth.distance_km_min + dn * (cfg.synth.distance_km_max - cfg.synth.distance_km_min)


def _weather_field(rng: np.random.Generator, cfg: Config, n_hours: int) -> dict[str, np.ndarray]:
    """AR(1)-smooth weather per airport over ``n_hours`` hours.

    Returns arrays shaped ``(n_airports, n_hours)`` for vis/wind/precip/thunder.
    """
    w = cfg.synth.weather
    n_ap = cfg.synth.n_airports
    s = float(w["smoothing"])

    vis = np.empty((n_ap, n_hours))
    wind = np.empty((n_ap, n_hours))
    precip = np.empty((n_ap, n_hours))
    thunder = np.zeros((n_ap, n_hours), dtype=int)

    for a in range(n_ap):
        # Each airport has its own baseline climate.
        vis_mean = rng.uniform(w["vis_min"] + 3.0, w["vis_max"])
        wind_mean = rng.uniform(3.0, w["wind_max"] * 0.55)
        precip_mean = rng.uniform(0.0, w["precip_max"] * 0.35)

        v = vis_mean
        wd = wind_mean
        p = precip_mean
        for h in range(n_hours):
            v = vis_mean + s * (v - vis_mean) + (1 - s) * rng.normal(0, 2.5)
            wd = wind_mean + s * (wd - wind_mean) + (1 - s) * rng.normal(0, 6.0)
            p = precip_mean + s * (p - precip_mean) + (1 - s) * rng.normal(0, 2.0)
            vis[a, h] = np.clip(v, w["vis_min"], w["vis_max"])
            wind[a, h] = np.clip(wd, 0.0, w["wind_max"])
            precip[a, h] = max(0.0, p)
            # Thunder is more likely under heavy precipitation.
            prob = w["thunder_prob"] + 0.6 * min(precip[a, h] / w["precip_max"], 1.0)
            thunder[a, h] = int(rng.random() < min(prob, 0.9))
    precip = np.clip(precip, 0.0, w["precip_max"])
    return {"vis": vis, "wind": wind, "precip": precip, "thunder": thunder}


def _sample_departure_minute(rng: np.random.Generator, cfg: Config) -> int:
    """Sample a first-leg departure minute biased toward morning/evening peaks."""
    start_h = cfg.synth.day_start_hour
    end_h = cfg.synth.day_end_hour
    # Mixture of two gaussians (morning ~8:30, evening ~18:00) + uniform base.
    if rng.random() < 0.7:
        center = 8.5 if rng.random() < 0.5 else 18.0
        hour = rng.normal(center, 1.6)
    else:
        hour = rng.uniform(start_h, end_h)
    hour = float(np.clip(hour, start_h, end_h - 1))
    minute = hour * 60 + rng.uniform(0, 60)
    minute = int(round(minute / 5.0) * 5)  # snap to 5-minute grid
    return int(np.clip(minute, start_h * 60, (end_h - 1) * 60 + 55))


def generate(cfg: Config) -> pd.DataFrame:
    """Generate the synthetic flight table (deterministic given ``cfg.seed``)."""
    rng = np.random.default_rng(cfg.seed)
    s = cfg.synth

    airports = [f"AP{i:02d}" for i in range(s.n_airports)]
    carriers = [f"CA{i}" for i in range(s.n_carriers)]
    types = [f"AT{i}" for i in range(s.n_aircraft_types)]
    type_min_turn = rng.integers(s.turnaround_min, s.turnaround_max + 1, size=s.n_aircraft_types)
    type_speed = rng.uniform(s.speed_kmh_min, s.speed_kmh_max, size=s.n_aircraft_types)

    dist_km = _airport_distance_matrix(rng, s.n_airports, cfg)
    n_hours = 28
    weather = _weather_field(rng, cfg, n_hours)

    # --- Pass 1: build tail chains / schedules ------------------------------
    rows: list[dict] = []
    tail_seq = 0
    while len(rows) < s.n_flights:
        carrier = carriers[rng.integers(s.n_carriers)]
        ti = int(rng.integers(s.n_aircraft_types))
        atype = types[ti]
        min_turn = int(type_min_turn[ti])
        speed = float(type_speed[ti])
        tail_id = f"{carrier}-{tail_seq:04d}"
        n_legs = int(rng.integers(s.legs_min, s.legs_max + 1))

        origin_idx = int(rng.integers(s.n_airports))
        dep_min = _sample_departure_minute(rng, cfg)
        prev_arr = None
        for leg in range(n_legs):
            if leg > 0:
                slack = int(rng.integers(s.turnaround_slack_min, s.turnaround_slack_max + 1))
                dep_min = int(round((prev_arr + min_turn + slack) / 5.0) * 5)
                # Never schedule a departure past midnight (keeps the curfew
                # window free); drop optional legs once the operating day ends,
                # but let mandatory (< legs_min) legs run to late evening.
                if dep_min >= 24 * 60 or (leg >= s.legs_min and dep_min > s.day_end_hour * 60):
                    break
            dest_idx = int(rng.choice([j for j in range(s.n_airports) if j != origin_idx]))
            distance = float(dist_km[origin_idx, dest_idx])
            duration = distance / speed * 60.0 + TAXI_MINUTES
            arr_min = int(round(dep_min + duration))
            sched_turn = float(dep_min - prev_arr) if prev_arr is not None else np.nan

            rows.append(
                {
                    "tail_id": tail_id,
                    "leg_index": leg,
                    "airline": carrier,
                    "origin": airports[origin_idx],
                    "dest": airports[dest_idx],
                    "aircraft_type": atype,
                    "origin_idx": origin_idx,
                    "distance": distance,
                    "min_turnaround": min_turn,
                    "sched_turnaround_min": sched_turn,
                    "dep_min": dep_min,
                    "arr_min": arr_min,
                }
            )
            origin_idx = dest_idx
            prev_arr = arr_min
        tail_seq += 1

    rows = rows[: s.n_flights]
    df = pd.DataFrame(rows)
    df.insert(0, "flight_id", [f"FL{i:06d}" for i in range(len(df))])

    # --- Datetimes ----------------------------------------------------------
    base = pd.Timestamp(s.base_date)
    df["sched_dep"] = base + pd.to_timedelta(df["dep_min"], unit="m")
    df["sched_arr"] = base + pd.to_timedelta(df["arr_min"], unit="m")
    dep_hour = (df["dep_min"] // 60).to_numpy()

    # --- Weather sampled at (origin, departure hour) ------------------------
    oi = df["origin_idx"].to_numpy()
    hh = np.clip(dep_hour, 0, n_hours - 1)
    df["vis"] = weather["vis"][oi, hh]
    df["wind"] = weather["wind"][oi, hh]
    df["precip"] = weather["precip"][oi, hh]
    df["thunder"] = weather["thunder"][oi, hh]

    # --- Causal quantities --------------------------------------------------
    wsev = weather_severity_from_raw(
        df["vis"].to_numpy(),
        df["wind"].to_numpy(),
        df["precip"].to_numpy(),
        df["thunder"].to_numpy(),
        s.weather,
        s.weather_severity_weights,
    )
    peak = is_peak_hour(dep_hour, cfg.features.peak_windows)
    raw_cong = compute_airport_congestion(
        df["origin"], df["dep_min"].to_numpy(), cfg.features.congestion_window_min
    )
    cong = (raw_cong - raw_cong.min()) / (raw_cong.max() - raw_cong.min() + 1e-9)

    df["weather_severity"] = wsev
    df["congestion"] = cong

    # --- Pass 2: sequential label with delay propagation --------------------
    c = s.coeffs
    noise = rng.normal(0.0, c["sigma"], size=len(df))
    order = df.sort_values(["tail_id", "leg_index"]).index.to_numpy()
    dep_delay = np.zeros(len(df))
    prev_leg_delay = np.zeros(len(df))
    last_delay: dict[str, float] = {}
    pos = {idx: i for i, idx in enumerate(df.index)}
    for idx in order:
        i = pos[idx]
        tail = df.at[idx, "tail_id"]
        leg = df.at[idx, "leg_index"]
        prev = 0.0 if leg == 0 else last_delay.get(tail, 0.0)
        prev_leg_delay[i] = prev
        base_delay = (
            c["a0"]
            + c["a1"] * wsev[i]
            + c["a2"] * cong[i]
            + c["a3"] * peak[i]
            + c["a4"] * wsev[i] * peak[i]
            + c["a5"] * prev
            + noise[i]
        )
        d = max(0.0, float(base_delay))
        dep_delay[i] = d
        last_delay[tail] = d

    df["prev_leg_delay"] = prev_leg_delay
    df["dep_delay"] = dep_delay
    df["is_delayed15"] = (df["dep_delay"] > 15.0).astype(int)

    # First legs have no preceding turnaround: default to min_turnaround (slack 0).
    df["sched_turnaround_min"] = df["sched_turnaround_min"].fillna(
        df["min_turnaround"].astype(float)
    )

    # --- Final schema ordering ----------------------------------------------
    schema = [
        "flight_id",
        "tail_id",
        "leg_index",
        "airline",
        "origin",
        "dest",
        "aircraft_type",
        "sched_dep",
        "sched_arr",
        "distance",
        "min_turnaround",
        "sched_turnaround_min",
        "prev_leg_delay",
        "vis",
        "wind",
        "precip",
        "thunder",
        "weather_severity",
        "congestion",
        "dep_delay",
        "is_delayed15",
    ]
    return df[schema].reset_index(drop=True)


def save(df: pd.DataFrame, cfg: Config) -> None:
    cfg.paths.ensure()
    df.to_parquet(cfg.paths.flights_parquet, index=False)


def load_or_generate(cfg: Config, *, force: bool = False) -> pd.DataFrame:
    """Load cached flights or generate them if missing (or ``force``)."""
    path = cfg.paths.flights_parquet
    if path.exists() and not force:
        return pd.read_parquet(path)
    df = generate(cfg)
    save(df, cfg)
    return df


def summarize(df: pd.DataFrame) -> dict:
    """Quick descriptive stats used by the CLI / tests."""
    return {
        "n_flights": int(len(df)),
        "n_tails": int(df["tail_id"].nunique()),
        "n_airports": int(pd.concat([df["origin"], df["dest"]]).nunique()),
        "positive_rate": float(df["is_delayed15"].mean()),
        "mean_dep_delay": float(df["dep_delay"].mean()),
        "median_dep_delay": float(df["dep_delay"].median()),
        "corr_weather": float(df["weather_severity"].corr(df["dep_delay"])),
        "corr_congestion": float(df["congestion"].corr(df["dep_delay"])),
        "corr_prev_leg": float(df["prev_leg_delay"].corr(df["dep_delay"])),
    }
