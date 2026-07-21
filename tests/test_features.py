"""Feature engineering: shape, no leakage, correct interaction values."""

from __future__ import annotations

import numpy as np

from flightopt.features import (
    CATEGORICAL_FEATURES,
    ENCODED_FEATURES,
    FEATURE_COLUMNS,
    INTERACTION_FEATURES,
    NETWORK_FEATURES,
    NUMERIC_FEATURES,
    build_features,
)


def test_feature_groups_are_disjoint_and_complete():
    assert len(set(FEATURE_COLUMNS)) == len(FEATURE_COLUMNS)  # no duplicates
    assert set(FEATURE_COLUMNS) == set(
        NUMERIC_FEATURES + INTERACTION_FEATURES + CATEGORICAL_FEATURES
    )
    assert set(NETWORK_FEATURES) <= set(NUMERIC_FEATURES)
    assert set(ENCODED_FEATURES) <= set(NUMERIC_FEATURES)


def test_no_missing_values(cfg, flights):
    X, y_reg, y_clf, _ = build_features(flights, cfg)
    assert list(X.columns) == FEATURE_COLUMNS
    assert X[NUMERIC_FEATURES + INTERACTION_FEATURES].isna().sum().sum() == 0
    assert len(y_reg) == len(flights) and len(y_clf) == len(flights)


def test_interaction_values(cfg, flights):
    X, _, _, _ = build_features(flights, cfg)
    assert np.allclose(X["wsev_x_peak"], X["weather_severity"] * X["is_peak"])
    assert np.allclose(X["cong_x_peak"], X["airport_congestion"] * X["is_peak"])
    assert np.allclose(X["prevdelay_x_slack"], X["prev_leg_delay"] * X["turnaround_slack"])


def test_target_encodings_are_fit_on_train_only(cfg, flights):
    """Test rows must reuse the TRAIN mapping and never see their own labels."""
    cut = int(len(flights) * 0.7)
    train, test = flights.iloc[:cut], flights.iloc[cut:].reset_index(drop=True)
    _, _, _, stats = build_features(train, cfg)
    X_test, _, _, _ = build_features(test, cfg, stats)

    mapping, default = stats["encodings"]["carrier_delay_rate"]
    expected = test["airline"].map(mapping).astype(float).fillna(default).to_numpy()
    assert np.allclose(X_test["carrier_delay_rate"].to_numpy(), expected)

    # Flipping the test labels must not move a single encoded value.
    flipped = test.copy()
    flipped["is_delayed15"] = 1 - flipped["is_delayed15"]
    X_flipped, _, _, _ = build_features(flipped, cfg, stats)
    for col in ("carrier_delay_rate", "route_delay_rate", "origin_hour_delay_rate"):
        assert np.allclose(X_test[col].to_numpy(), X_flipped[col].to_numpy())


def test_bounded_features(cfg, flights):
    X, _, _, _ = build_features(flights, cfg)
    assert X["airport_congestion"].between(0.0, 1.0).all()
    assert X["weather_severity"].between(0.0, 1.0).all()
    for col in ("carrier_delay_rate", "route_delay_rate", "tail_delay_rate"):
        assert X[col].between(0.0, 1.0).all()
    assert (X["airport_recent_flights"] >= 0).all()


def test_no_targets_returns_none(cfg, flights):
    X, y_reg, y_clf, _ = build_features(
        flights.drop(columns=["dep_delay", "is_delayed15"]), cfg
    )
    assert y_reg is None and y_clf is None
    assert len(X) == len(flights)
