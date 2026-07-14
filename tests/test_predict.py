"""Prediction: pipeline trains, beats baselines, correct output shapes."""

from __future__ import annotations

import json

import pandas as pd


def test_pipeline_artifacts_exist(pipeline):
    cfg = pipeline
    assert (cfg.paths.models / "bundle.pkl").exists()
    assert cfg.paths.predictions_parquet.exists()
    assert (cfg.paths.reports / "predict_metrics.json").exists()
    assert cfg.paths.feature_dict_md.exists()


def test_beats_baselines(pipeline):
    cfg = pipeline
    m = json.loads((cfg.paths.reports / "predict_metrics.json").read_text(encoding="utf-8"))
    reg, clf = m["regression"], m["classification"]
    # LightGBM regression beats the mean baseline and is competitive with RF.
    assert reg["lightgbm"]["mae"] < reg["mean_baseline"]["mae"]
    assert reg["lightgbm"]["mae"] <= reg["random_forest"]["mae"] * 1.10
    # Classifier shows real discriminative signal and beats the no-skill PR
    # baseline (= positive prevalence). Thresholds are robust on the small
    # smoke config; the full run reports ROC-AUC ~0.87.
    lg = clf["lightgbm"]
    cm = lg["confusion_matrix"]
    prevalence = (cm["tp"] + cm["fn"]) / sum(cm.values())
    assert lg["roc_auc"] > 0.6
    assert lg["pr_auc"] > prevalence


def test_predictions_frame(pipeline):
    cfg = pipeline
    preds = pd.read_parquet(cfg.paths.predictions_parquet)
    for col in ("flight_id", "pred_delay", "pred_proba", "split", "is_delayed15"):
        assert col in preds.columns
    assert set(preds["split"].unique()) <= {"train", "test"}
    assert preds["pred_proba"].between(0, 1).all()


def test_predict_output_dims(pipeline, flights):
    from flightopt.predict import ModelBundle, predict

    cfg = pipeline
    bundle = ModelBundle.load(cfg)
    out = predict(bundle, cfg, flights)
    assert len(out["delay"]) == len(flights)
    assert len(out["risk_proba"]) == len(flights)
