"""Data generator: schema, positive-rate band, reproducibility, injected signal."""

from __future__ import annotations

SCHEMA = {
    "flight_id", "tail_id", "leg_index", "airline", "origin", "dest", "aircraft_type",
    "sched_dep", "sched_arr", "distance", "min_turnaround", "sched_turnaround_min",
    "prev_leg_delay", "vis", "wind", "precip", "thunder", "weather_severity",
    "congestion", "dep_delay", "is_delayed15",
}


def test_schema_complete(flights):
    assert SCHEMA.issubset(set(flights.columns))
    assert len(flights) == 600
    assert flights["flight_id"].is_unique
    assert flights.isna().sum().sum() == 0


def test_positive_rate_in_band(cfg):
    # The 20-35% band is a property of the delivered 2000-flight config
    # (congestion density scales with flight count), so validate at full size.
    from flightopt.config import load_config
    from flightopt.data import synth

    full = load_config(overrides={"paths": {"root": str(cfg.paths.root)}})
    rate = synth.generate(full)["is_delayed15"].mean()
    assert 0.20 <= rate <= 0.35, f"positive rate {rate:.3f} outside [0.20, 0.35]"


def test_reproducible(cfg):
    from flightopt.data import synth

    a = synth.generate(cfg)
    b = synth.generate(cfg)
    assert a.equals(b)


def test_injected_correlations(flights):
    # Each documented driver must be positively (and non-trivially) correlated.
    assert flights["weather_severity"].corr(flights["dep_delay"]) > 0.1
    assert flights["congestion"].corr(flights["dep_delay"]) > 0.1
    assert flights["prev_leg_delay"].corr(flights["dep_delay"]) > 0.1


def test_no_curfew_departures(flights):
    hour = flights["sched_dep"].dt.hour
    assert ((hour >= 0) & (hour < 5)).sum() == 0


def test_leg_chaining(flights):
    legs = flights.groupby("tail_id").size()
    assert 2.0 <= legs.mean() <= 4.0
    # First leg of every tail has zero inherited delay.
    firsts = flights[flights["leg_index"] == 0]
    assert (firsts["prev_leg_delay"] == 0).all()
