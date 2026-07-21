"""Public / real flight-data loader (optional).

Adapts a common public schema (e.g. the Kaggle "Flight Delays" / US BTS
on-time dataset) to the unified schema produced by :mod:`flightopt.data.synth`,
so the exact same feature / predict / schedule / evaluate pipeline runs on real
data with no other change.

Usage
-----
Drop a CSV into ``data/raw/`` and call :func:`load_or_default` (the CLI's
``gen-data`` still defaults to the synthetic generator; wire this in explicitly
when you have real data).  Missing weather columns are filled with benign
defaults; ``tail_id`` and leg ordering are approximated when absent.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from flightopt.config import Config

# Column aliases: unified_name -> list of accepted source names.
_ALIASES: dict[str, list[str]] = {
    "airline": ["airline", "AIRLINE", "OP_CARRIER", "carrier", "Reporting_Airline"],
    "origin": ["origin", "ORIGIN", "ORIGIN_AIRPORT", "Origin"],
    "dest": ["dest", "DEST", "DESTINATION_AIRPORT", "Dest"],
    "tail_id": ["tail_id", "TAIL_NUMBER", "TAIL_NUM", "Tail_Number"],
    "aircraft_type": ["aircraft_type", "AIRCRAFT_TYPE", "TYPE"],
    "distance": ["distance", "DISTANCE", "Distance"],
    "dep_delay": ["dep_delay", "DEPARTURE_DELAY", "DEP_DELAY", "DepDelay"],
    "sched_dep": ["sched_dep", "SCHEDULED_DEPARTURE", "CRS_DEP_TIME", "sched_departure"],
    "sched_arr": ["sched_arr", "SCHEDULED_ARRIVAL", "CRS_ARR_TIME"],
    "sched_elapsed": ["SCHEDULED_TIME", "CRS_ELAPSED_TIME", "sched_duration"],
    "date": ["FL_DATE", "date", "flight_date"],
    "year": ["YEAR", "year"],
    "month": ["MONTH", "month"],
    "day": ["DAY", "day"],
}


def _first_present(df: pd.DataFrame, names: list[str]) -> str | None:
    for n in names:
        if n in df.columns:
            return n
    return None


def _resolve(df: pd.DataFrame) -> dict[str, str | None]:
    return {k: _first_present(df, v) for k, v in _ALIASES.items()}


def _parse_datetime(df: pd.DataFrame, cols: dict, cfg: Config) -> pd.Series:
    """Best-effort scheduled-departure timestamp from assorted public formats."""
    base = pd.Timestamp(cfg.synth.base_date)
    # Case 1: an explicit parseable datetime column.
    if cols["sched_dep"] is not None:
        raw = df[cols["sched_dep"]]
        parsed = pd.to_datetime(raw, errors="coerce")
        if parsed.notna().mean() > 0.5:
            return parsed
        # HHMM integer format (e.g. 1435) combined with a date if available.
        hhmm = pd.to_numeric(raw, errors="coerce").fillna(0).astype(int)
        minutes = (hhmm // 100) * 60 + (hhmm % 100)
        if cols["date"] is not None:
            day = pd.to_datetime(df[cols["date"]], errors="coerce").fillna(base)
        elif cols["year"] and cols["month"] and cols["day"]:
            day = pd.to_datetime(
                dict(year=df[cols["year"]], month=df[cols["month"]], day=df[cols["day"]]),
                errors="coerce",
            ).fillna(base)
        else:
            day = pd.Series(base, index=df.index)
        return day.dt.normalize() + pd.to_timedelta(minutes, unit="m")
    return pd.Series(base, index=df.index)


def load_public(csv_path: str | Path, cfg: Config) -> pd.DataFrame:
    """Load a public flight CSV and coerce it to the unified schema."""
    raw = pd.read_csv(csv_path)
    cols = _resolve(raw)
    n = len(raw)
    out = pd.DataFrame(index=range(n))

    out["airline"] = raw[cols["airline"]].astype(str) if cols["airline"] else "NA"
    out["origin"] = raw[cols["origin"]].astype(str) if cols["origin"] else "NA"
    out["dest"] = raw[cols["dest"]].astype(str) if cols["dest"] else "NA"
    out["aircraft_type"] = (
        raw[cols["aircraft_type"]].astype(str) if cols["aircraft_type"] else "UNKNOWN"
    )
    out["distance"] = (
        pd.to_numeric(raw[cols["distance"]], errors="coerce") if cols["distance"] else np.nan
    )

    out["sched_dep"] = _parse_datetime(raw, cols, cfg)
    if cols["sched_arr"] is not None:
        arr = pd.to_datetime(raw[cols["sched_arr"]], errors="coerce")
    else:
        arr = pd.Series(pd.NaT, index=out.index)
    if cols["sched_elapsed"] is not None:
        elapsed = pd.to_numeric(raw[cols["sched_elapsed"]], errors="coerce")
    else:
        elapsed = out["distance"] / 12.0 + 25.0  # ~720 km/h + taxi fallback
    arr = arr.fillna(out["sched_dep"] + pd.to_timedelta(elapsed.fillna(90), unit="m"))
    out["sched_arr"] = arr

    out["dep_delay"] = (
        pd.to_numeric(raw[cols["dep_delay"]], errors="coerce").clip(lower=0)
        if cols["dep_delay"]
        else 0.0
    )
    out["is_delayed15"] = (out["dep_delay"] > 15).astype(int)

    # tail_id: use the source column, else approximate one per (airline, aircraft).
    out["tail_id"] = (
        raw[cols["tail_id"]].astype(str) if cols["tail_id"] else out["airline"].astype(str)
    )
    out["tail_id"] = out["tail_id"].replace({"nan": None}).fillna(out["airline"])

    # Leg ordering + propagation approximated from the tail time-series.
    out = out.sort_values(["tail_id", "sched_dep"]).reset_index(drop=True)
    out["leg_index"] = out.groupby("tail_id").cumcount()
    out["prev_leg_delay"] = out.groupby("tail_id")["dep_delay"].shift(1).fillna(0.0)
    prev_arr = out.groupby("tail_id")["sched_arr"].shift(1)
    out["sched_turnaround_min"] = (
        (out["sched_dep"] - prev_arr).dt.total_seconds() / 60.0
    )

    # Defaults for fields absent from public data.
    out["min_turnaround"] = cfg.synth.turnaround_min
    out["distance"] = out["distance"].fillna(out["distance"].median() if out["distance"].notna().any() else 800.0)
    w = cfg.synth.weather
    out["vis"] = w["vis_max"]     # clear skies -> weather_severity ~ 0
    out["wind"] = 0.0
    out["precip"] = 0.0
    out["thunder"] = 0
    out["weather_severity"] = 0.0
    out["congestion"] = 0.0
    out["flight_id"] = [f"FL{i:06d}" for i in range(len(out))]

    schema = [
        "flight_id", "tail_id", "leg_index", "airline", "origin", "dest", "aircraft_type",
        "sched_dep", "sched_arr", "distance", "min_turnaround", "sched_turnaround_min",
        "prev_leg_delay", "vis", "wind", "precip", "thunder", "weather_severity",
        "congestion", "dep_delay", "is_delayed15",
    ]
    return out[schema]


def load_nycflights13(
    cfg: Config,
    months: tuple[int, ...] = (1,),
    year: int = 2013,
    wind_max_mph: float = 100.0,
) -> pd.DataFrame:
    """Load the **real** nycflights13 dataset into the unified schema.

    2013 departures from the three New York airports (EWR/JFK/LGA), sourced from
    the US DOT/BTS on-time database, joined with the real hourly weather
    observations at those airports.

    Real-data handling (documented, so the benchmark stays honest):

    * ``dep_delay`` is clipped at 0 (early departures are not "delay"), matching
      the synthetic convention so the two runs are comparable. Extreme delays
      are **kept** (no capping) -- the tail is part of the real difficulty.
    * Wind speed above ``wind_max_mph`` is a known sensor error in this dataset
      (max 1048 mph) and is treated as missing.
    * Units are converted to the ranges the feature layer expects: visibility
      miles->km, wind mph->knots, precipitation inches->mm, distance miles->km.
    * ``thunder`` is **not available** in this dataset and is set to 0.
    * ``min_turnaround`` is not in the data; a constant operational default is
      assumed. Leg order / propagation is reconstructed per (tail, date).
    """
    import nycflights13 as nyc

    from flightopt.features import compute_airport_congestion, weather_severity_from_raw

    f = nyc.flights
    f = f[(f["year"] == year) & (f["month"].isin(list(months)))].copy()
    f = f.dropna(subset=["dep_delay", "sched_dep_time", "sched_arr_time", "tailnum"])
    f = f.reset_index(drop=True)

    # --- real hourly weather at the origin airport ------------------------
    # Join on the LOCAL calendar/hour columns present in both tables: the
    # `time_hour` column is tz-aware UTC, and using it would shift the
    # hour-of-day by the UTC offset and corrupt the peak-hour features.
    w = nyc.weather.copy()
    keys = ["origin", "year", "month", "day", "hour"]
    w = w[keys + ["visib", "wind_speed", "precip"]].drop_duplicates(subset=keys)
    f = f.merge(w, on=keys, how="left")

    # --- aircraft type from the planes table ------------------------------
    planes = nyc.planes[["tailnum", "model"]].drop_duplicates(subset=["tailnum"])
    f = f.merge(planes, on="tailnum", how="left")

    # --- timestamps (local wall-clock, timezone-free) ---------------------
    date_base = pd.to_datetime(dict(year=f["year"], month=f["month"], day=f["day"]))
    sched_dep = (
        date_base
        + pd.to_timedelta(f["hour"].astype(int), unit="h")
        + pd.to_timedelta(f["minute"].astype(int), unit="m")
    )
    sat = f["sched_arr_time"].astype(int)
    sched_arr = (
        date_base
        + pd.to_timedelta(sat // 100, unit="h")
        + pd.to_timedelta(sat % 100, unit="m")
    )
    # Overnight arrivals roll into the next day.
    sched_arr = sched_arr.where(sched_arr >= sched_dep, sched_arr + pd.Timedelta(days=1))

    # --- unit conversion + sensor-error cleaning --------------------------
    wind = f["wind_speed"].where(f["wind_speed"] <= wind_max_mph)
    vis_km = f["visib"].astype(float) * 1.60934
    wind_kt = wind.astype(float) * 0.868976
    precip_mm = f["precip"].astype(float) * 25.4
    wcfg = cfg.synth.weather
    vis_km = vis_km.fillna(wcfg["vis_max"])
    wind_kt = wind_kt.fillna(wind_kt.median())
    precip_mm = precip_mm.fillna(0.0)

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
            "thunder": 0,
        }
    )
    out["is_delayed15"] = (out["dep_delay"] > 15).astype(int)
    out["min_turnaround"] = int(cfg.synth.turnaround_min)

    # --- leg order + delay propagation, reconstructed per (tail, date) ----
    out = out.sort_values(["tail_id", "sched_dep"]).reset_index(drop=True)
    day_key = out["sched_dep"].dt.date
    grp = out.groupby(["tail_id", day_key])
    out["leg_index"] = grp.cumcount()
    out["prev_leg_delay"] = grp["dep_delay"].shift(1).fillna(0.0)
    prev_arr = grp["sched_arr"].shift(1)
    out["sched_turnaround_min"] = (
        (out["sched_dep"] - prev_arr).dt.total_seconds() / 60.0
    ).fillna(out["min_turnaround"].astype(float))

    # --- derived planning-time quantities (same formulas as the generator) -
    out["weather_severity"] = weather_severity_from_raw(
        out["vis"].to_numpy(), out["wind"].to_numpy(), out["precip"].to_numpy(),
        out["thunder"].to_numpy(), wcfg, cfg.synth.weather_severity_weights,
    )
    dep_min = (
        (out["sched_dep"] - out["sched_dep"].dt.normalize().min()).dt.total_seconds() // 60
    ).to_numpy()
    raw_cong = compute_airport_congestion(
        out["origin"], dep_min, cfg.features.congestion_window_min
    )
    out["congestion"] = (raw_cong - raw_cong.min()) / (raw_cong.max() - raw_cong.min() + 1e-9)

    out = out.reset_index(drop=True)
    out.insert(0, "flight_id", [f"FL{i:06d}" for i in range(len(out))])
    schema = [
        "flight_id", "tail_id", "leg_index", "airline", "origin", "dest", "aircraft_type",
        "sched_dep", "sched_arr", "distance", "min_turnaround", "sched_turnaround_min",
        "prev_leg_delay", "vis", "wind", "precip", "thunder", "weather_severity",
        "congestion", "dep_delay", "is_delayed15",
    ]
    return out[schema]


def find_raw_csv(cfg: Config) -> Path | None:
    """Return the first CSV in ``data/raw`` (ignoring the .gitkeep), else None."""
    if not cfg.paths.data_raw.exists():
        return None
    csvs = sorted(cfg.paths.data_raw.glob("*.csv"))
    return csvs[0] if csvs else None


def load_or_default(cfg: Config) -> pd.DataFrame:
    """Load a real CSV from ``data/raw`` if present; otherwise synthesize."""
    from flightopt.data import synth

    csv = find_raw_csv(cfg)
    if csv is not None:
        return load_public(csv, cfg)
    return synth.load_or_generate(cfg)
