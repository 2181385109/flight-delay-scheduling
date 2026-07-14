"""Evaluation: metric helpers, headline table, and end-to-end summary structure."""

from __future__ import annotations

from flightopt import evaluate


def test_ge_lt_helpers():
    assert evaluate._ge(0.9, 0.8) is True
    assert evaluate._ge(0.7, 0.8) is False
    assert evaluate._ge(None, 0.8) is None
    assert evaluate._lt(1.0, 2.0) is True
    assert evaluate._lt(3.0, 2.0) is False


def test_headline_table_renders():
    metrics = {
        "headline": {
            "high_risk_capture_recall": {
                "value": 0.84, "precision": 0.6, "f1": 0.7, "target": 0.8,
                "baseline_rule_recall": 0.4, "pass": True,
            },
            "prediction_error": {
                "lightgbm_mae": 3.2, "random_forest_mae": 3.6,
                "mean_baseline_mae": 6.0, "beats_baselines": True,
            },
        }
    }
    table = evaluate.headline_table(metrics)
    assert "High-risk capture" in table
    assert "PASS" in table
    assert "Prediction error" in table


def test_summarize_structure(pipeline):
    cfg = pipeline
    m = evaluate.summarize(cfg)
    h = m["headline"]
    for key in (
        "high_risk_capture_recall",
        "constraint_satisfaction_rate",
        "high_risk_delay_reduction_min",
        "prediction_error",
    ):
        assert key in h
    assert cfg.paths.metrics_json.exists()
    # Constraint satisfaction should clear the configured target in the run.
    assert h["constraint_satisfaction_rate"]["value"] >= cfg.metrics.constraint_satisfaction_target
