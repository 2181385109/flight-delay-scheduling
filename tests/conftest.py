"""Shared fixtures: a small, fast end-to-end pipeline on REAL data.

Tests run on a contiguous 5-day slice of real January-2013 departures. A
contiguous slice (rather than a random sample) is essential: the network-state
features depend on the temporal ordering of flights.
"""

from __future__ import annotations

import pandas as pd
import pytest

from flightopt.config import load_config

# Small + fast overrides so the whole suite runs in well under a minute.
SMALL = {
    "data": {"months": [1]},
    "predict": {"optuna_trials": 4, "group_kfold_splits": 3, "optuna_timeout_s": 120,
                "rf_n_estimators": 80},
    "schedule": {"solver_time_limit_s": 6, "only_high_risk": False, "runway_capacity": 8},
}

SKIP_DAYS = 6    # skip the New-Year holiday period at the start of January
SLICE_DAYS = 10


@pytest.fixture(scope="session")
def cfg(tmp_path_factory):
    root = tmp_path_factory.mktemp("flightopt")
    return load_config(overrides={**SMALL, "paths": {"root": str(root)}})


@pytest.fixture(scope="session")
def flights(cfg):
    """A real, contiguous mid-January slice, cached where the stages expect it.

    Starts a few days in so the lagged network-state features have history, and
    avoids the atypical New-Year holiday traffic.
    """
    from flightopt.data import loader

    df = loader.load_nycflights13(cfg, months=(1,))
    start = df["sched_dep"].min() + pd.Timedelta(days=SKIP_DAYS)
    end = start + pd.Timedelta(days=SLICE_DAYS)
    df = df[(df["sched_dep"] >= start) & (df["sched_dep"] < end)].reset_index(drop=True)
    cfg.paths.ensure()
    df.to_parquet(cfg.paths.flights_parquet, index=False)
    return df


@pytest.fixture(scope="session")
def pipeline(cfg, flights):
    """Run train -> grade -> schedule -> evaluate once; artifacts land in temp root."""
    from flightopt import evaluate, predict, risk, schedule

    predict.run_training(cfg, flights)
    risk.run_grading(cfg)
    schedule.run_scheduling(cfg, solver="cpsat")
    evaluate.run_evaluate(cfg)
    return cfg
