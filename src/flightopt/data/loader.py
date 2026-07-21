"""Real flight-data loading.

Primary source: **nycflights13** -- every 2013 departure from the three New York
airports (EWR/JFK/LGA), taken from the US DOT/BTS on-time database and joined
with the real hourly weather observations recorded at those airports.

``load_flights(cfg)`` is the canonical entry point used by the whole pipeline
(it caches to ``data/processed/flights.parquet``).  A generic CSV adapter
(:func:`load_public`) is also provided so other real on-time datasets can be
mapped onto the same unified schema.

Documented real-data handling
-----------------------------
* ``dep_delay`` is clipped at 0 -- an early departure is not a delay. Extreme
  delays are **kept** (no capping); the long tail is part of the real problem.
* Wind above ``cfg.data.wind_max_mph_valid`` is a known sensor error in this
  dataset (max 1048 mph) and is treated as missing.
* Units are converted to the ranges the feature layer expects: visibility
  miles->km, wind mph->knots, precipitation inches->mm, distance miles->km.
* ``thunder`` is not recorded in this dataset and is set to 0.
* ``min_turnaround`` is not in the data; an operational default is assumed.
* Leg order and delay propagation are reconstructed per (tail, calendar day).
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd

from flightopt.config import Config

#: Unified schema every loader must produce.
SCHEMA: list[str] = [
    "flight_id", "tail_id", "leg_index", "airline", "origin", "dest", "aircraft_type",
    "sched_dep", "sched_arr", "distance", "min_turnaround", "sched_turnaround_min",
    "prev_leg_delay", "vis", "wind", "precip", "thunder", "temp", "humid",
    "wind_gust", "pressure", "weather_severity", "dep_delay", "is_delayed15",
]


def _finalize(out: pd.DataFrame, cfg: Config) -> pd.DataFrame:
    """Shared post-processing: leg chaining, propagation, severity, schema."""
    from flightopt.features import weather_severity_from_raw

    out = out.sort_values(["tail_id", "sched_dep"]).reset_index(drop=True)
    day_key = out["sched_dep"].dt.date
    grp = out.groupby(["tail_id", day_key])
    out["leg_index"] = grp.cumcount()
    out["prev_leg_delay"] = grp["dep_delay"].shift(1).fillna(0.0)
    prev_arr = grp["sched_arr"].shift(1)
    out["sched_turnaround_min"] = (
        (out["sched_dep"] - prev_arr).dt.total_seconds() / 60.0
    ).fillna(out["min_turnaround"].astype(float))

    wcfg = cfg.data.weather
    out["weather_severity"] = weather_severity_from_raw(
        out["vis"].to_numpy(), out["wind"].to_numpy(), out["precip"].to_numpy(),
        out["thunder"].to_numpy(), wcfg, cfg.data.weather_severity_weights,
    )
    out["is_delayed15"] = (out["dep_delay"] > 15).astype(int)
    out = out.reset_index(drop=True)
    out.insert(0, "flight_id", [f"FL{i:07d}" for i in range(len(out))])
    return out[SCHEMA]


def load_nycflights13(cfg: Config, months: tuple[int, ...] | None = None,
                      year: int | None = None) -> pd.DataFrame:
    """Load real 2013 NYC departures + real hourly weather into the schema."""
    import nycflights13 as nyc

    year = year if year is not None else cfg.data.year
    months = tuple(months) if months is not None else tuple(cfg.data.months)

    f = nyc.flights
    f = f[(f["year"] == year) & (f["month"].isin(list(months)))].copy()
    f = f.dropna(subset=["dep_delay", "sched_dep_time", "sched_arr_time", "tailnum"])
    f = f.reset_index(drop=True)

    # Real hourly weather, joined on the LOCAL calendar/hour columns present in
    # both tables (`time_hour` is tz-aware UTC; using it would shift the
    # hour-of-day by the UTC offset and corrupt the peak-hour features).
    wcols = ["visib", "wind_speed", "precip", "temp", "humid", "wind_gust", "pressure"]
    keys = ["origin", "year", "month", "day", "hour"]
    w = nyc.weather.copy()[keys + wcols].drop_duplicates(subset=keys)
    f = f.merge(w, on=keys, how="left")

    planes = nyc.planes[["tailnum", "model"]].drop_duplicates(subset=["tailnum"])
    f = f.merge(planes, on="tailnum", how="left")

    # Timestamps from local wall-clock fields.
    date_base = pd.to_datetime(dict(year=f["year"], month=f["month"], day=f["day"]))
    sched_dep = (
        date_base
        + pd.to_timedelta(f["hour"].astype(int), unit="h")
        + pd.to_timedelta(f["minute"].astype(int), unit="m")
    )
    sat = f["sched_arr_time"].astype(int)
    sched_arr = (
        date_base + pd.to_timedelta(sat // 100, unit="h") + pd.to_timedelta(sat % 100, unit="m")
    )
    sched_arr = sched_arr.where(sched_arr >= sched_dep, sched_arr + pd.Timedelta(days=1))

    # Unit conversion + sensor-error cleaning.
    wcfg = cfg.data.weather
    wind_raw = f["wind_speed"].where(f["wind_speed"] <= cfg.data.wind_max_mph_valid)
    gust_raw = f["wind_gust"].where(f["wind_gust"] <= cfg.data.wind_max_mph_valid)
    vis_km = (f["visib"].astype(float) * 1.60934).fillna(wcfg["vis_max"])
    wind_kt = (wind_raw.astype(float) * 0.868976)
    wind_kt = wind_kt.fillna(wind_kt.median())
    gust_kt = (gust_raw.astype(float) * 0.868976).fillna(0.0)
    precip_mm = (f["precip"].astype(float) * 25.4).fillna(0.0)

    out = pd.DataFrame(
        {
            "tail_id": f["tailnum"].astype(str),
            "airline": f["carrier"].astype(str),
            "origin": f["origin"].astype(str),
            "dest": f["dest"].astype(str),
            "aircraft_type": f["model"].fillna("UNKNOWN").astype(str),
            "sched_dep": sched_dep,
            "sched_arr": sched_arr,
            "distance": f["distance"].astype(float) * 1.60934,
            "dep_delay": f["dep_delay"].astype(float).clip(lower=0.0),
            "vis": vis_km.clip(wcfg["vis_min"], wcfg["vis_max"]),
            "wind": wind_kt.clip(0.0, wcfg["wind_max"]),
            "precip": precip_mm.clip(0.0, wcfg["precip_max"]),
            "thunder": 0,  # not recorded in this dataset
            "temp": f["temp"].astype(float),
            "humid": f["humid"].astype(float),
            "wind_gust": gust_kt,
            "pressure": f["pressure"].astype(float),
            "min_turnaround": int(cfg.data.default_min_turnaround),
        }
    )
    for c in ("temp", "humid", "pressure"):
        out[c] = out[c].fillna(out[c].median())
    return _finalize(out, cfg)


# ---------------------------------------------------------------------------
# Generic CSV adapter (other real on-time datasets, e.g. BTS / Kaggle exports).
# ---------------------------------------------------------------------------
_ALIASES: dict[str, list[str]] = {
    "airline": ["airline", "AIRLINE", "OP_CARRIER", "carrier", "Reporting_Airline"],
    "origin": ["origin", "ORIGIN", "ORIGIN_AIRPORT", "Origin"],
    "dest": ["dest", "DEST", "DESTINATION_AIRPORT", "Dest"],
    "tail_id": ["tail_id", "TAIL_NUMBER", "TAIL_NUM", "Tail_Number", "tailnum"],
    "aircraft_type": ["aircraft_type", "AIRCRAFT_TYPE", "TYPE", "model"],
    "distance": ["distance", "DISTANCE", "Distance"],
    "dep_delay": ["dep_delay", "DEPARTURE_DELAY", "DEP_DELAY", "DepDelay"],
    "sched_dep": ["sched_dep", "SCHEDULED_DEPARTURE", "CRS_DEP_TIME"],
    "sched_arr": ["sched_arr", "SCHEDULED_ARRIVAL", "CRS_ARR_TIME"],
    "sched_elapsed": ["SCHEDULED_TIME", "CRS_ELAPSED_TIME", "sched_duration"],
    "date": ["FL_DATE", "date", "flight_date"],
}


def _first_present(df: pd.DataFrame, names: list[str]) -> str | None:
    return next((n for n in names if n in df.columns), None)


def load_public(csv_path: str | Path, cfg: Config) -> pd.DataFrame:
    """Map a generic public on-time CSV onto the unified schema (best effort).

    Weather columns absent from such exports are filled with benign values, so
    the weather features are effectively inert -- state that when reporting.
    """
    raw = pd.read_csv(csv_path)
    cols = {k: _first_present(raw, v) for k, v in _ALIASES.items()}
    wcfg = cfg.data.weather

    sched_dep = pd.to_datetime(raw[cols["sched_dep"]], errors="coerce") if cols["sched_dep"] else None
    if sched_dep is None or sched_dep.isna().mean() > 0.5:
        hhmm = pd.to_numeric(raw[cols["sched_dep"]], errors="coerce").fillna(0).astype(int)
        day = (
            pd.to_datetime(raw[cols["date"]], errors="coerce")
            if cols["date"]
            else pd.Series(pd.Timestamp(f"{cfg.data.year}-01-01"), index=raw.index)
        )
        sched_dep = day.dt.normalize() + pd.to_timedelta(
            (hhmm // 100) * 60 + (hhmm % 100), unit="m"
        )

    elapsed = (
        pd.to_numeric(raw[cols["sched_elapsed"]], errors="coerce")
        if cols["sched_elapsed"]
        else pd.Series(90.0, index=raw.index)
    )
    out = pd.DataFrame(
        {
            "tail_id": (raw[cols["tail_id"]].astype(str) if cols["tail_id"] else "UNKNOWN"),
            "airline": (raw[cols["airline"]].astype(str) if cols["airline"] else "NA"),
            "origin": (raw[cols["origin"]].astype(str) if cols["origin"] else "NA"),
            "dest": (raw[cols["dest"]].astype(str) if cols["dest"] else "NA"),
            "aircraft_type": (
                raw[cols["aircraft_type"]].astype(str) if cols["aircraft_type"] else "UNKNOWN"
            ),
            "sched_dep": sched_dep,
            "sched_arr": sched_dep + pd.to_timedelta(elapsed.fillna(90), unit="m"),
            "distance": (
                pd.to_numeric(raw[cols["distance"]], errors="coerce").fillna(800.0)
                if cols["distance"]
                else pd.Series(800.0, index=raw.index)
            ),
            "dep_delay": (
                pd.to_numeric(raw[cols["dep_delay"]], errors="coerce").fillna(0).clip(lower=0)
                if cols["dep_delay"]
                else pd.Series(0.0, index=raw.index)
            ),
            "vis": wcfg["vis_max"], "wind": 0.0, "precip": 0.0, "thunder": 0,
            "temp": 50.0, "humid": 60.0, "wind_gust": 0.0, "pressure": 1013.0,
            "min_turnaround": int(cfg.data.default_min_turnaround),
        }
    )
    out = out.dropna(subset=["sched_dep"]).reset_index(drop=True)
    return _finalize(out, cfg)


def find_raw_csv(cfg: Config) -> Path | None:
    """First CSV directly inside ``data/raw`` (ignoring sub-directories)."""
    if not cfg.paths.data_raw.exists():
        return None
    return next(iter(sorted(cfg.paths.data_raw.glob("*.csv"))), None)


def load_flights(cfg: Config, *, force: bool = False) -> pd.DataFrame:
    """Canonical loader: cached real flights for the whole pipeline.

    Uses ``data/raw/*.csv`` when present, otherwise the configured real source.
    Re-run with ``force=True`` (``flightopt fetch-data``) after changing
    ``data.months`` / ``data.year``.
    """
    cache = cfg.paths.flights_parquet
    if cache.exists() and not force:
        return pd.read_parquet(cache)

    csv = find_raw_csv(cfg)
    if csv is not None:
        df = load_public(csv, cfg)
    elif cfg.data.source == "nycflights13":
        df = load_nycflights13(cfg)
    else:
        raise ValueError(f"Unknown data source: {cfg.data.source!r}")

    cfg.paths.ensure()
    df.to_parquet(cache, index=False)
    return df


def summarize(df: pd.DataFrame) -> dict:
    """Descriptive stats used by the CLI and tests."""
    return {
        "n_flights": int(len(df)),
        "n_tails": int(df["tail_id"].nunique()),
        "n_airports": int(pd.concat([df["origin"], df["dest"]]).nunique()),
        "n_carriers": int(df["airline"].nunique()),
        "positive_rate": float(df["is_delayed15"].mean()),
        "mean_dep_delay": float(df["dep_delay"].mean()),
        "median_dep_delay": float(df["dep_delay"].median()),
        "max_dep_delay": float(df["dep_delay"].max()),
        "date_min": str(df["sched_dep"].min()),
        "date_max": str(df["sched_dep"].max()),
        "corr_weather": float(df["weather_severity"].corr(df["dep_delay"])),
        "corr_prev_leg": float(df["prev_leg_delay"].corr(df["dep_delay"])),
    }
