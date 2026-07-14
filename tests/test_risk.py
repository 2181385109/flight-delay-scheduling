"""Risk grading: monotonic levels, train-derived cut-points, capture beats rule."""

from __future__ import annotations

import json

import numpy as np

from flightopt import risk


def test_grade_is_monotonic():
    cut = np.array([1.0, 2.0, 3.0, 4.0])
    values = np.array([0.5, 1.0, 1.5, 2.5, 3.5, 4.5])
    levels = risk.grade(values, cut)
    assert list(levels) == [1, 2, 2, 3, 4, 5]
    # Non-decreasing on sorted input.
    assert (np.diff(risk.grade(np.sort(values), cut)) >= 0).all()


def test_cutpoints_from_training_quantiles():
    rng = np.random.default_rng(0)
    train = rng.normal(10, 3, 500)
    cut = risk.fit_quantiles(train, [0.2, 0.4, 0.6, 0.8])
    assert np.allclose(cut, np.quantile(train, [0.2, 0.4, 0.6, 0.8]))
    assert (np.diff(cut) > 0).all()


def test_high_risk_proportion(pipeline):
    import pandas as pd

    cfg = pipeline
    graded = pd.read_parquet(cfg.paths.graded_parquet)
    flagged = graded["high_risk"].mean()
    assert 0.25 <= flagged <= 0.55  # ~top 40% by construction


def test_capture_beats_rule(pipeline):
    cfg = pipeline
    m = json.loads((cfg.paths.reports / "risk_metrics.json").read_text(encoding="utf-8"))
    model_recall = m["capture_test"]["high_risk_recall"]
    rule_recall = m["rule_baseline_test"]["high_risk_recall"]
    assert model_recall > rule_recall
    assert model_recall >= 0.60  # small-sample floor; full run targets >= 0.80


def test_capture_metrics_math():
    y_true = np.array([1, 1, 0, 0, 1])
    levels = np.array([5, 3, 4, 1, 5])  # high risk = {4,5}
    out = risk.capture_metrics(y_true, levels, [4, 5])
    # Positives at idx 0,1,4 -> flagged {0,4} -> recall 2/3.
    assert abs(out["high_risk_recall"] - 2 / 3) < 1e-9
