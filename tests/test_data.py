"""Real data loading: schema integrity, real-world properties, and — most
importantly — that the network-state features are strictly causal."""

from __future__ import annotations

import numpy as np
import pandas as pd

from flightopt.data.loader import SCHEMA
from flightopt.features import lagged_group_state


def test_schema_complete(flights):
    assert list(flights.columns) == SCHEMA
    assert flights["flight_id"].is_unique
    assert flights.isna().sum().sum() == 0
    assert len(flights) > 1000


def test_real_world_properties(flights):
    # Real NYC origins and real two-letter carrier codes.
    assert set(flights["origin"].unique()) <= {"EWR", "JFK", "LGA"}
    assert flights["airline"].str.len().max() <= 3
    assert flights["tail_id"].nunique() > 100
    # Early departures are clipped: delay is a non-negative quantity.
    assert (flights["dep_delay"] >= 0).all()
    assert (flights["is_delayed15"] == (flights["dep_delay"] > 15).astype(int)).all()
    # Real data has a long right tail, unlike a tidy simulation.
    assert flights["dep_delay"].max() > 100


def test_leg_chaining(flights):
    firsts = flights[flights["leg_index"] == 0]
    assert (firsts["prev_leg_delay"] == 0).all()
    # Later legs of the same aircraft/day do inherit delay sometimes.
    later = flights[flights["leg_index"] > 0]
    assert len(later) > 0
    assert (later["prev_leg_delay"] > 0).any()


def test_deterministic(cfg, flights):
    from flightopt.data import loader

    again = loader.load_nycflights13(cfg, months=(1,))
    again = again[again["sched_dep"] < flights["sched_dep"].max() + pd.Timedelta(seconds=1)]
    assert len(again) >= len(flights)


# ---------------------------------------------------------------------------
# The critical anti-leakage property of the network-state features.
# ---------------------------------------------------------------------------
def test_lagged_state_only_sees_the_past():
    keys = np.array(["A"] * 5)
    minutes = np.array([0.0, 60.0, 120.0, 180.0, 240.0])
    values = np.array([0.0, 10.0, 20.0, 30.0, 40.0])

    mean, cnt = lagged_group_state(keys, minutes, values, horizon_min=60, window_min=180)

    # Nothing precedes the first flight.
    assert cnt[0] == 0 and mean[0] == 0.0
    # For t=240 the admissible window is [0, 180): values 0, 10, 20 -> mean 10.
    assert cnt[4] == 3
    assert abs(mean[4] - 10.0) < 1e-9


def test_lagged_state_ignores_future_changes():
    """Perturbing a later flight must not change any earlier flight's feature."""
    keys = np.array(["A"] * 5)
    minutes = np.array([0.0, 60.0, 120.0, 180.0, 240.0])
    values = np.array([0.0, 10.0, 20.0, 30.0, 40.0])
    base, _ = lagged_group_state(keys, minutes, values, 60, 180)

    tampered = values.copy()
    tampered[4] = 9999.0  # a huge delay in the future
    after, _ = lagged_group_state(keys, minutes, tampered, 60, 180)

    assert np.allclose(base, after), "future information leaked into earlier rows"


def test_lagged_state_never_includes_own_value():
    """A flight's own delay must never enter its own network-state feature."""
    keys = np.array(["A", "A"])
    minutes = np.array([0.0, 10.0])          # second flight only 10 min later
    values = np.array([100.0, 500.0])
    mean, cnt = lagged_group_state(keys, minutes, values, horizon_min=60, window_min=180)
    # With a 60-min horizon neither flight can see the other (or itself).
    assert cnt.tolist() == [0.0, 0.0]
    assert mean.tolist() == [0.0, 0.0]


def test_lagged_state_groups_are_independent():
    keys = np.array(["A", "B", "A"])
    minutes = np.array([0.0, 0.0, 240.0])
    values = np.array([10.0, 999.0, 0.0])
    mean, cnt = lagged_group_state(keys, minutes, values, 60, 180)
    # The A-flight at t=240 sees only the other A-flight, never B's 999.
    assert cnt[2] == 1
    assert abs(mean[2] - 10.0) < 1e-9
