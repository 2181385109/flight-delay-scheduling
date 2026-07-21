"""Quantile-based 5-level risk grading.

Cut-points are fit on the **training** predictions (20/40/60/80 percentiles by
default) and reused verbatim on new data, so the grading is distribution
adaptive yet leakage-free.  L4/L5 are treated as high risk.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.metrics import f1_score, precision_score, recall_score

from flightopt.config import Config


def fit_quantiles(values: np.ndarray, quantiles: list[float]) -> np.ndarray:
    """Return the cut-points (one per quantile) fit on ``values``."""
    return np.quantile(np.asarray(values, float), quantiles)


def grade(values: np.ndarray, cutpoints: np.ndarray) -> np.ndarray:
    """Assign integer risk levels ``1..len(cutpoints)+1`` via the cut-points."""
    return np.digitize(np.asarray(values, float), np.asarray(cutpoints), right=False) + 1


def is_high_risk(levels: np.ndarray, high_risk_levels: list[int]) -> np.ndarray:
    return np.isin(np.asarray(levels), np.asarray(high_risk_levels))


def capture_metrics(
    y_true_clf: np.ndarray,
    levels: np.ndarray,
    high_risk_levels: list[int],
) -> dict:
    """High-risk capture statistics vs. the binary delay label.

    Treats "flagged as high risk (L4/L5)" as the positive prediction for
    ``is_delayed15`` and reports recall (capture rate), precision and F1.
    """
    y_true = np.asarray(y_true_clf).astype(int)
    y_pred = is_high_risk(levels, high_risk_levels).astype(int)
    return {
        "high_risk_recall": float(recall_score(y_true, y_pred, zero_division=0)),
        "high_risk_precision": float(precision_score(y_true, y_pred, zero_division=0)),
        "high_risk_f1": float(f1_score(y_true, y_pred, zero_division=0)),
        "flagged_rate": float(y_pred.mean()),
        "positive_rate": float(y_true.mean()),
    }


def grade_frame(
    cfg: Config,
    predictions: pd.DataFrame,
    *,
    score_col: str | None = None,
) -> tuple[pd.DataFrame, np.ndarray]:
    """Grade a predictions frame; cut-points come from the training split.

    ``predictions`` must contain ``split`` ("train"/"test") and the score column
    (``pred_delay`` for ``risk.source == 'delay'`` or ``pred_proba`` otherwise).
    Returns the frame with an added ``risk_level`` / ``high_risk`` column and the
    fitted cut-points.
    """
    col = score_col or ("pred_delay" if cfg.risk.source == "delay" else "pred_proba")
    out = predictions.copy()
    train_scores = out.loc[out["split"] == "train", col].to_numpy()
    if train_scores.size == 0:  # fall back to all rows if no split marker
        train_scores = out[col].to_numpy()
    cutpoints = fit_quantiles(train_scores, cfg.risk.quantiles)
    out["risk_level"] = grade(out[col].to_numpy(), cutpoints)
    out["high_risk"] = is_high_risk(out["risk_level"].to_numpy(), cfg.risk.high_risk_levels)
    return out, cutpoints


def rule_baseline_high_risk(df: pd.DataFrame, cfg: Config) -> np.ndarray:
    """Naive single-rule baseline: high risk if bad weather OR inherited delay."""
    rb = cfg.predict.rule_baseline
    wsev = df["weather_severity"].to_numpy()
    prev = df["prev_leg_delay"].to_numpy()
    return (
        (wsev > rb["weather_severity_threshold"]) | (prev > rb["prev_leg_delay_threshold"])
    ).astype(int)


def run_grading(cfg: Config) -> dict:
    """CLI entry: grade the out-of-fold predictions, persist ``graded.parquet``
    and the risk-grading metrics (model vs rule baseline)."""
    import json

    from flightopt.data import loader

    cfg.paths.ensure()
    preds = pd.read_parquet(cfg.paths.predictions_parquet)
    graded, cutpoints = grade_frame(cfg, preds)

    flights = loader.load_flights(cfg)
    aux = flights.set_index("flight_id")
    graded = graded.merge(
        aux[["weather_severity", "prev_leg_delay"]], left_on="flight_id", right_index=True
    )

    test = graded[graded["split"] == "test"]
    hl = cfg.risk.high_risk_levels

    # Model grading capture (held-out test = headline, plus full operation).
    cap_test = capture_metrics(test["is_delayed15"].to_numpy(), test["risk_level"].to_numpy(), hl)
    cap_all = capture_metrics(
        graded["is_delayed15"].to_numpy(), graded["risk_level"].to_numpy(), hl
    )

    # Rule baseline on the same test rows.
    rule_hr = rule_baseline_high_risk(test, cfg)
    rule = {
        "high_risk_recall": float(recall_score(test["is_delayed15"], rule_hr, zero_division=0)),
        "high_risk_precision": float(precision_score(test["is_delayed15"], rule_hr, zero_division=0)),
        "high_risk_f1": float(f1_score(test["is_delayed15"], rule_hr, zero_division=0)),
        "flagged_rate": float(np.mean(rule_hr)),
    }

    # Operating-point analysis. Flagging the top 40% (L4/L5) is a *design
    # choice*, not a law: report what share of flights would have to be flagged
    # to reach the target recall, so a MISS is actionable rather than opaque.
    score_col = "pred_delay" if cfg.risk.source == "delay" else "pred_proba"
    scores = test[score_col].to_numpy()
    labels = test["is_delayed15"].to_numpy().astype(int)
    target = float(cfg.metrics.high_risk_recall_target)
    total_pos = int(labels.sum())
    if total_pos > 0:
        order = np.argsort(-scores, kind="stable")
        recalls = np.cumsum(labels[order]) / total_pos
        idx = int(np.searchsorted(recalls, target))
        flag_for_target = min((idx + 1) / len(labels), 1.0) if idx < len(labels) else 1.0
    else:
        flag_for_target = float("nan")
    operating_point = {
        "target_recall": target,
        "flag_rate_for_target_recall": float(flag_for_target),
        "current_flag_rate": float(cap_test["flagged_rate"]),
        "current_recall": float(cap_test["high_risk_recall"]),
    }

    # Proba-based grading reported as an alternative.
    graded_proba, _ = grade_frame(cfg, preds, score_col="pred_proba")
    test_proba = graded_proba[graded_proba["split"] == "test"]
    cap_proba_test = capture_metrics(
        test_proba["is_delayed15"].to_numpy(), test_proba["risk_level"].to_numpy(), hl
    )

    graded.to_parquet(cfg.paths.graded_parquet, index=False)
    level_counts = graded["risk_level"].value_counts().sort_index().to_dict()
    metrics = {
        "source": cfg.risk.source,
        "cutpoints": [float(c) for c in cutpoints],
        "capture_test": cap_test,
        "capture_all": cap_all,
        "capture_proba_test": cap_proba_test,
        "operating_point": operating_point,
        "rule_baseline_test": rule,
        "level_counts": {int(k): int(v) for k, v in level_counts.items()},
    }
    with open(cfg.paths.reports / "risk_metrics.json", "w", encoding="utf-8") as fh:
        json.dump(metrics, fh, indent=2)
    return metrics
