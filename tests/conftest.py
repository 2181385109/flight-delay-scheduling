"""Shared fixtures: a small, fast end-to-end pipeline in an isolated temp root."""

from __future__ import annotations

import pytest

from flightopt.config import load_config

# Small + fast overrides so the whole suite runs in seconds.
SMALL = {
    "synth": {"n_flights": 600},
    "predict": {"optuna_trials": 4, "group_kfold_splits": 3, "optuna_timeout_s": 120},
    "schedule": {"solver_time_limit_s": 6, "only_high_risk": False},
}


@pytest.fixture(scope="session")
def cfg(tmp_path_factory):
    root = tmp_path_factory.mktemp("flightopt")
    return load_config(overrides={**SMALL, "paths": {"root": str(root)}})


@pytest.fixture(scope="session")
def flights(cfg):
    from flightopt.data import synth

    return synth.load_or_generate(cfg, force=True)


@pytest.fixture(scope="session")
def pipeline(cfg, flights):
    """Run train -> grade -> schedule -> evaluate once; artifacts land in temp root."""
    from flightopt import evaluate, predict, risk, schedule

    predict.run_training(cfg, flights)
    risk.run_grading(cfg)
    schedule.run_scheduling(cfg, solver="cpsat")
    evaluate.run_evaluate(cfg)
    return cfg
