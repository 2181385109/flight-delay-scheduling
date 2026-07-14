"""Feature engineering: shape, no leakage, correct interaction values."""

from __future__ import annotations

import numpy as np

from flightopt.features import (
    CATEGORICAL_FEATURES,
    FEATURE_COLUMNS,
    INTERACTION_FEATURES,
    NUMERIC_FEATURES,
    build_features,
)


def _build(cfg, df, fit_stats=None):
    return build_features(
        df,
        weather_cfg=cfg.synth.weather,
        weather_weights=cfg.synth.weather_severity_weights,
        features_cfg=cfg.features,
        fit_stats=fit_stats,
    )


def test_feature_count():
    assert len(FEATURE_COLUMNS) == 21
    assert len(NUMERIC_FEATURES) == 12
    assert len(INTERACTION_FEATURES) == 3
    assert len(CATEGORICAL_FEATURES) == 6


def test_no_missing_numeric(cfg, flights):
    X, y_reg, y_clf, _ = _build(cfg, flights)
    assert list(X.columns) == FEATURE_COLUMNS
    assert X[NUMERIC_FEATURES + INTERACTION_FEATURES].isna().sum().sum() == 0
    assert len(y_reg) == len(flights) and len(y_clf) == len(flights)


def test_interaction_values(cfg, flights):
    X, _, _, _ = _build(cfg, flights)
    assert np.allclose(X["wsev_x_peak"], X["weather_severity"] * X["is_peak"])
    assert np.allclose(X["cong_x_peak"], X["airport_congestion"] * X["is_peak"])
    assert np.allclose(X["prevdelay_x_slack"], X["prev_leg_delay"] * X["turnaround_slack"])


def test_carrier_rate_is_leakage_safe(cfg, flights):
    train = flights.iloc[:400]
    test = flights.iloc[400:]
    _, _, _, stats = _build(cfg, train)
    X_test, _, _, _ = _build(cfg, test, fit_stats=stats)
    # Test rows must reuse the TRAIN mapping (never their own labels).
    mapping = stats["carrier_ontime_rate"]
    g = stats["carrier_ontime_global"]
    expected = test["airline"].map(mapping).fillna(g).to_numpy()
    assert np.allclose(X_test["carrier_ontime_rate"].to_numpy(), expected)


def test_congestion_bounds(cfg, flights):
    X, _, _, _ = _build(cfg, flights)
    assert X["airport_congestion"].between(0.0, 1.0).all()
    assert X["weather_severity"].between(0.0, 1.0).all()


def test_no_targets_returns_none(cfg, flights):
    X, y_reg, y_clf, _ = _build(cfg, flights.drop(columns=["dep_delay", "is_delayed15"]))
    assert y_reg is None and y_clf is None
    assert len(X) == len(flights)
