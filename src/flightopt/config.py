"""Load ``config.yaml`` into a strongly-typed configuration object.

Dataclasses mirror the YAML structure so the rest of the codebase reads typed
attributes (``cfg.data.months``) instead of dict lookups.  Unknown keys in
the YAML are ignored gracefully, and the raw dict remains available via
``cfg.raw`` for anything not promoted to a field.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml

from flightopt.paths import ProjectPaths, build_paths, find_repo_root


@dataclass
class DataConfig:
    """Real flight-data source configuration."""

    source: str = "nycflights13"
    year: int = 2013
    months: list[int] = field(default_factory=lambda: [1, 2, 3])
    # Network-state features may only use information observable this many
    # minutes before scheduled departure (the assumed prediction horizon).
    prediction_horizon_min: int = 60
    network_window_min: int = 180
    # Not present in the source data; an operational default is assumed.
    default_min_turnaround: int = 30
    # Wind above this (mph) is a known sensor error in nycflights13 (max 1048).
    wind_max_mph_valid: float = 100.0
    weather: dict[str, float] = field(default_factory=dict)
    weather_severity_weights: dict[str, float] = field(default_factory=dict)


@dataclass
class FeaturesConfig:
    congestion_window_min: int = 30
    peak_windows: list[list[int]] = field(default_factory=lambda: [[7, 10], [17, 20]])
    holidays: list[str] = field(default_factory=list)
    time_buckets: dict[str, list[int]] = field(default_factory=dict)


@dataclass
class PredictConfig:
    test_size: float = 0.20
    group_kfold_splits: int = 5
    optuna_trials: int = 30
    optuna_timeout_s: int = 180
    early_stopping_rounds: int = 50
    rf_n_estimators: int = 300
    rule_baseline: dict[str, float] = field(default_factory=dict)
    search_space: dict[str, Any] = field(default_factory=dict)


@dataclass
class RiskConfig:
    source: str = "delay"
    quantiles: list[float] = field(default_factory=lambda: [0.2, 0.4, 0.6, 0.8])
    n_levels: int = 5
    high_risk_levels: list[int] = field(default_factory=lambda: [4, 5])


@dataclass
class ScheduleConfig:
    slot_minutes: int = 5
    max_offset_min: int = 30
    window_minutes: int = 15
    runway_capacity: int = 5
    curfew_hours: list[int] = field(default_factory=lambda: [0, 5])
    weight_high_risk: float = 5.0
    weight_normal: float = 1.0
    capacity_penalty: float = 8.0
    # Secondary penalty on the *number* of over-capacity windows. Without it the
    # optimum is degenerate: the same total excess can be spread over a
    # different number of windows at identical cost, so the reported violation
    # count would vary between runs.
    capacity_window_penalty: float = 2.0
    only_high_risk: bool = True
    solver_time_limit_s: int = 25
    greedy_max_passes: int = 6
    # CP-SAT's parallel portfolio is non-deterministic even with a fixed seed,
    # so the default is a single worker: reproducibility beats a second or two.
    solver_workers: int = 1
    # Which operating day to re-time (ISO date). ``None`` picks the busiest day
    # in the loaded window. Scheduling is inherently a single-day problem.
    day: str | None = None

    @property
    def offsets(self) -> list[int]:
        """Allowed offsets in minutes, e.g. [-30, -25, ..., 25, 30]."""
        k = self.max_offset_min
        step = self.slot_minutes
        return list(range(-k, k + 1, step))


@dataclass
class MetricsConfig:
    high_risk_recall_target: float = 0.80
    constraint_satisfaction_target: float = 0.85
    delay_reduction_target: float = 1.0


@dataclass
class Config:
    seed: int
    paths: ProjectPaths
    data: DataConfig
    features: FeaturesConfig
    predict: PredictConfig
    risk: RiskConfig
    schedule: ScheduleConfig
    metrics: MetricsConfig
    raw: dict[str, Any] = field(default_factory=dict)


def _filter_kwargs(cls: type, data: dict[str, Any]) -> dict[str, Any]:
    """Keep only keys that are declared fields of ``cls``."""
    valid = {f.name for f in cls.__dataclass_fields__.values()}  # type: ignore[attr-defined]
    return {k: v for k, v in (data or {}).items() if k in valid}


def default_config_path() -> Path:
    return find_repo_root() / "config.yaml"


def load_config(
    path: str | Path | None = None,
    *,
    overrides: dict[str, Any] | None = None,
) -> Config:
    """Load and parse ``config.yaml`` into a :class:`Config`.

    Parameters
    ----------
    path:
        Explicit path to a YAML config; defaults to ``<repo>/config.yaml``.
    overrides:
        Optional shallow-per-section overrides applied after loading, e.g.
        ``{"predict": {"optuna_trials": 5}}`` (used by tests / CI smoke runs).
    """
    cfg_path = Path(path) if path else default_config_path()
    with open(cfg_path, "r", encoding="utf-8") as fh:
        raw: dict[str, Any] = yaml.safe_load(fh) or {}

    if overrides:
        for section, values in overrides.items():
            if isinstance(values, dict) and isinstance(raw.get(section), dict):
                raw[section] = {**raw[section], **values}
            else:
                raw[section] = values

    paths = build_paths(raw.get("paths", {}), root_override=raw.get("paths", {}).get("root"))

    return Config(
        seed=int(raw.get("seed", 42)),
        paths=paths,
        data=DataConfig(**_filter_kwargs(DataConfig, raw.get("data", {}))),
        features=FeaturesConfig(**_filter_kwargs(FeaturesConfig, raw.get("features", {}))),
        predict=PredictConfig(**_filter_kwargs(PredictConfig, raw.get("predict", {}))),
        risk=RiskConfig(**_filter_kwargs(RiskConfig, raw.get("risk", {}))),
        schedule=ScheduleConfig(**_filter_kwargs(ScheduleConfig, raw.get("schedule", {}))),
        metrics=MetricsConfig(**_filter_kwargs(MetricsConfig, raw.get("metrics", {}))),
        raw=raw,
    )
