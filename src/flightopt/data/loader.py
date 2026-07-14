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
