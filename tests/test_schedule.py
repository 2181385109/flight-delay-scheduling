"""Scheduling: CP-SAT respects hard constraints, never worsens delay; greedy runs."""

from __future__ import annotations

import numpy as np
import pandas as pd

from flightopt.data import loader
from flightopt.predict import ModelBundle
from flightopt.schedule import (
    build_problem,
    constraint_report,
    delay_report,
    optimize_cpsat,
    optimize_greedy,
)


def _problem(cfg):
    flights = loader.load_flights(cfg)
    graded = pd.read_parquet(cfg.paths.graded_parquet)
    bundle = ModelBundle.load(cfg)
    return build_problem(cfg, flights, graded, bundle)


def test_cpsat_satisfies_hard_constraints(pipeline):
    cfg = pipeline
    p = _problem(cfg)
    off, _trace, _meta = optimize_cpsat(p, cfg)
    rep = constraint_report(p, off, cfg)
    assert rep["turnaround_rate"] == 1.0
    assert rep["curfew_rate"] == 1.0
    assert int(np.abs(off).max()) <= cfg.schedule.max_offset_min


def test_cpsat_does_not_increase_delay(pipeline):
    cfg = pipeline
    p = _problem(cfg)
    off, _t, _m = optimize_cpsat(p, cfg)
    before = delay_report(p, np.zeros(p.n, dtype=int))
    after = delay_report(p, off)
    assert after["weighted_total_delay"] <= before["weighted_total_delay"] + 1e-6
    assert after["high_risk_delay_reduction"] >= 0.0


def test_greedy_runs_and_is_feasible(pipeline):
    cfg = pipeline
    p = _problem(cfg)
    off, trace, _meta = optimize_greedy(p, cfg)
    assert len(off) == p.n
    assert set(np.unique(off)).issubset(set(p.offsets))
    assert len(trace) >= 1
    rep = constraint_report(p, off, cfg)
    assert rep["turnaround_rate"] == 1.0
    assert rep["curfew_rate"] == 1.0


def test_offsets_respect_curfew(pipeline):
    cfg = pipeline
    p = _problem(cfg)
    # Every flagged-feasible offset must keep the departure out of curfew
    # (except the grandfathered original slot at offset 0).
    from flightopt.schedule import _in_curfew

    off, _t, _m = optimize_cpsat(p, cfg)
    new_min = p.dep_min + off
    assert not _in_curfew(new_min, cfg.schedule.curfew_hours).any()
