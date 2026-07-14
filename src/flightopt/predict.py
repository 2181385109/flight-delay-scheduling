"""Prediction: LightGBM double-head (regression + classification) vs baselines.

Pipeline
--------
1. Time-series split (leading 80% train / trailing 20% test).
2. Optuna (TPE) hyper-parameter search with **GroupKFold(tail_id)** CV; the
   leakage-sensitive feature statistics are re-fit inside every fold.
3. Final LightGBM models fit on the whole training split (early stopping on a
   trailing slice of *train* -- never the test set).
4. Baselines: RandomForest (one-hot) and a naive rule / mean predictor.
5. Persist a model bundle + full-dataset predictions + metrics.

The regressor predicts ``dep_delay`` (minutes); the classifier predicts
``P(dep_delay > 15)``.  Downstream, risk grading ranks on the regressor output.
"""

from __future__ import annotations

import json
import warnings
from dataclasses import dataclass, field
from typing import Any

import joblib
import numpy as np
import optuna
import pandas as pd
from lightgbm import LGBMClassifier, LGBMRegressor, early_stopping, log_evaluation
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.metrics import (
    average_precision_score,
    confusion_matrix,
    f1_score,
    mean_absolute_error,
    mean_squared_error,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import GroupKFold

from flightopt.config import Config
from flightopt.features import (
    CATEGORICAL_FEATURES,
    FEATURE_COLUMNS,
    build_features,
    one_hot_for_baseline,
    write_feature_dict,
)

optuna.logging.set_verbosity(optuna.logging.WARNING)
warnings.filterwarnings("ignore", category=UserWarning)


# ---------------------------------------------------------------------------
# Model bundle (serialized to outputs/models/bundle.pkl).
# ---------------------------------------------------------------------------
@dataclass
class ModelBundle:
    lgbm_reg: Any
    lgbm_clf: Any
    rf_reg: Any
    rf_clf: Any
    stats: dict
    feature_columns: list[str]
    rf_columns: list[str]
    best_params_reg: dict
    best_params_clf: dict
    weather_cfg: dict = field(default_factory=dict)
    weather_weights: dict = field(default_factory=dict)

    def save(self, cfg: Config) -> None:
        cfg.paths.ensure()
        joblib.dump(self, cfg.paths.models / "bundle.pkl")

    @staticmethod
    def load(cfg: Config) -> "ModelBundle":
        return joblib.load(cfg.paths.models / "bundle.pkl")


# ---------------------------------------------------------------------------
# Feature helpers bound to a config.
# ---------------------------------------------------------------------------
def _features(cfg: Config, df: pd.DataFrame, fit_stats: dict | None):
    return build_features(
        df,
        weather_cfg=cfg.synth.weather,
        weather_weights=cfg.synth.weather_severity_weights,
        features_cfg=cfg.features,
        fit_stats=fit_stats,
    )


def time_split(df: pd.DataFrame, test_size: float) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Leading (1-test_size) train / trailing test, ordered by scheduled dep."""
    ordered = df.sort_values("sched_dep").reset_index(drop=True)
    cut = int(len(ordered) * (1.0 - test_size))
    return ordered.iloc[:cut].copy(), ordered.iloc[cut:].copy()


def _lgbm_params(trial: optuna.Trial, ss: dict) -> dict:
    return {
        "num_leaves": trial.suggest_int("num_leaves", *ss["num_leaves"]),
        "max_depth": trial.suggest_int("max_depth", *ss["max_depth"]),
        "learning_rate": trial.suggest_float("learning_rate", *ss["learning_rate"], log=True),
        "n_estimators": trial.suggest_int("n_estimators", *ss["n_estimators"]),
        "min_child_samples": trial.suggest_int("min_child_samples", *ss["min_child_samples"]),
        "subsample": trial.suggest_float("subsample", *ss["subsample"]),
        "colsample_bytree": trial.suggest_float("colsample_bytree", *ss["colsample_bytree"]),
        "reg_alpha": trial.suggest_float("reg_alpha", *ss["reg_alpha"], log=True),
        "reg_lambda": trial.suggest_float("reg_lambda", *ss["reg_lambda"], log=True),
    }


def _base_lgbm_kwargs(cfg: Config) -> dict:
    return {
        "random_state": cfg.seed,
        "n_jobs": -1,
        "verbosity": -1,
        "subsample_freq": 1,
    }


def _fit_lgbm(model, X_tr, y_tr, X_val, y_val, eval_metric: str, rounds: int):
    model.fit(
        X_tr,
        y_tr,
        eval_set=[(X_val, y_val)],
        eval_metric=eval_metric,
        callbacks=[early_stopping(rounds, verbose=False), log_evaluation(0)],
    )
    return model


def _fit_reg_clf(cfg: Config, X_tr, yr_tr, yc_tr, best_reg: dict, best_clf: dict):
    """Fit the LightGBM regressor + classifier, early-stopping on an inner slice
    of ``X_tr`` (never the evaluation fold)."""
    rounds = cfg.predict.early_stopping_rounds
    cut = int(len(X_tr) * 0.85)
    if cut < 1 or cut >= len(X_tr):  # too small to hold out a slice
        Xi = Xv = X_tr
        yri, yrv, yci, ycv = yr_tr, yr_tr, yc_tr, yc_tr
    else:
        Xi, Xv = X_tr.iloc[:cut], X_tr.iloc[cut:]
        yri, yrv = yr_tr.iloc[:cut], yr_tr.iloc[cut:]
        yci, ycv = yc_tr.iloc[:cut], yc_tr.iloc[cut:]
    reg = LGBMRegressor(**{**_base_lgbm_kwargs(cfg), **best_reg})
    _fit_lgbm(reg, Xi, yri, Xv, yrv, "l1", rounds)
    clf = LGBMClassifier(**{**_base_lgbm_kwargs(cfg), **best_clf})
    _fit_lgbm(clf, Xi, yci, Xv, ycv, "binary_logloss", rounds)
    return reg, clf


def _cv_folds(train_df: pd.DataFrame, n_splits: int):
    n_splits = min(n_splits, train_df["tail_id"].nunique())
    gkf = GroupKFold(n_splits=max(2, n_splits))
    groups = train_df["tail_id"].to_numpy()
    return list(gkf.split(train_df, groups=groups)), groups


def _optuna_search(
    cfg: Config,
    train_df: pd.DataFrame,
    task: str,  # "reg" | "clf"
) -> dict:
    """Bayesian (TPE) search maximizing PR-AUC (clf) or minimizing MAE (reg)."""
    ss = cfg.predict.search_space
    folds, _ = _cv_folds(train_df, cfg.predict.group_kfold_splits)
    rounds = cfg.predict.early_stopping_rounds

    # Pre-build per-fold features once (independent of hyper-params).
    prepared = []
    for tr_idx, va_idx in folds:
        f_tr = train_df.iloc[tr_idx]
        f_va = train_df.iloc[va_idx]
        X_tr, yr_tr, yc_tr, stats = _features(cfg, f_tr, None)
        X_va, yr_va, yc_va, _ = _features(cfg, f_va, stats)
        prepared.append((X_tr, yr_tr, yc_tr, X_va, yr_va, yc_va))

    def objective(trial: optuna.Trial) -> float:
        params = {**_base_lgbm_kwargs(cfg), **_lgbm_params(trial, ss)}
        scores = []
        for X_tr, yr_tr, yc_tr, X_va, yr_va, yc_va in prepared:
            if task == "reg":
                model = LGBMRegressor(**params)
                _fit_lgbm(model, X_tr, yr_tr, X_va, yr_va, "l1", rounds)
                scores.append(mean_absolute_error(yr_va, model.predict(X_va)))
            else:
                model = LGBMClassifier(**params)
                _fit_lgbm(model, X_tr, yc_tr, X_va, yc_va, "binary_logloss", rounds)
                proba = model.predict_proba(X_va)[:, 1]
                scores.append(average_precision_score(yc_va, proba))
        return float(np.mean(scores))

    direction = "minimize" if task == "reg" else "maximize"
    sampler = optuna.samplers.TPESampler(seed=cfg.seed)
    study = optuna.create_study(direction=direction, sampler=sampler)
    study.optimize(
        objective,
        n_trials=cfg.predict.optuna_trials,
        timeout=cfg.predict.optuna_timeout_s,
        show_progress_bar=False,
    )
    return study.best_params


def _reg_metrics(y_true, y_pred) -> dict:
    return {
        "mae": float(mean_absolute_error(y_true, y_pred)),
        "rmse": float(np.sqrt(mean_squared_error(y_true, y_pred))),
    }


def _clf_metrics(y_true, proba, threshold: float = 0.5) -> dict:
    y_pred = (np.asarray(proba) >= threshold).astype(int)
    y_true = np.asarray(y_true).astype(int)
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()
    return {
        "precision": float(precision_score(y_true, y_pred, zero_division=0)),
        "recall": float(recall_score(y_true, y_pred, zero_division=0)),
        "f1": float(f1_score(y_true, y_pred, zero_division=0)),
        "pr_auc": float(average_precision_score(y_true, proba)),
        "roc_auc": float(roc_auc_score(y_true, proba)) if len(set(y_true)) > 1 else float("nan"),
        "confusion_matrix": {"tn": int(tn), "fp": int(fp), "fn": int(fn), "tp": int(tp)},
    }


def run_training(cfg: Config, df: pd.DataFrame) -> dict:
    """Train all models, persist artifacts, and return a metrics summary."""
    cfg.paths.ensure()
    df = df.reset_index(drop=True)
    train_df, test_df = time_split(df, cfg.predict.test_size)

    # --- Hyper-parameter search (on the training split) -------------------
    best_reg = _optuna_search(cfg, train_df, "reg")
    best_clf = _optuna_search(cfg, train_df, "clf")

    # --- Final features (fit on train, applied to test) -------------------
    X_tr, yr_tr, yc_tr, stats = _features(cfg, train_df, None)
    X_te, yr_te, yc_te, _ = _features(cfg, test_df, stats)

    # Final LightGBM models (early stopping on an inner slice of train).
    lgbm_reg, lgbm_clf = _fit_reg_clf(cfg, X_tr, yr_tr, yc_tr, best_reg, best_clf)

    # --- Baselines ---------------------------------------------------------
    Xoh_tr = one_hot_for_baseline(X_tr)
    Xoh_te = one_hot_for_baseline(X_te).reindex(columns=Xoh_tr.columns, fill_value=0)
    rf_reg = RandomForestRegressor(n_estimators=300, random_state=cfg.seed, n_jobs=-1)
    rf_reg.fit(Xoh_tr, yr_tr)
    rf_clf = RandomForestClassifier(n_estimators=300, random_state=cfg.seed, n_jobs=-1)
    rf_clf.fit(Xoh_tr, yc_tr)

    # --- Test-set predictions ---------------------------------------------
    lgbm_reg_pred = lgbm_reg.predict(X_te)
    lgbm_clf_proba = lgbm_clf.predict_proba(X_te)[:, 1]
    rf_reg_pred = rf_reg.predict(Xoh_te)
    rf_clf_proba = rf_clf.predict_proba(Xoh_te)[:, 1]

    mean_pred = np.full(len(yr_te), float(yr_tr.mean()))
    rb = cfg.predict.rule_baseline
    rule_pred_clf = (
        (test_df["weather_severity"].to_numpy() > rb["weather_severity_threshold"])
        | (test_df["prev_leg_delay"].to_numpy() > rb["prev_leg_delay_threshold"])
    ).astype(int)

    # Out-of-fold predictions over the whole dataset (honest, distribution-
    # representative) + GroupKFold(tail_id) generalization metrics in one pass.
    oof_delay, oof_proba, cv_metrics = _oof_and_cv(cfg, df, best_reg, best_clf)

    metrics = {
        "regression": {
            "lightgbm": _reg_metrics(yr_te, lgbm_reg_pred),
            "random_forest": _reg_metrics(yr_te, rf_reg_pred),
            "mean_baseline": _reg_metrics(yr_te, mean_pred),
        },
        "classification": {
            "lightgbm": _clf_metrics(yc_te, lgbm_clf_proba),
            "random_forest": _clf_metrics(yc_te, rf_clf_proba),
            "rule_baseline": {
                "precision": float(precision_score(yc_te, rule_pred_clf, zero_division=0)),
                "recall": float(recall_score(yc_te, rule_pred_clf, zero_division=0)),
                "f1": float(f1_score(yc_te, rule_pred_clf, zero_division=0)),
            },
        },
        "cv_generalization": cv_metrics,
        "n_train": int(len(train_df)),
        "n_test": int(len(test_df)),
        "best_params_reg": best_reg,
        "best_params_clf": best_clf,
    }

    # --- Persist out-of-fold predictions (canonical for grading/scheduling) -
    all_pred = pd.DataFrame(
        {
            "flight_id": df["flight_id"].to_numpy(),
            "tail_id": df["tail_id"].to_numpy(),
            "dep_delay": df["dep_delay"].to_numpy(),
            "is_delayed15": df["is_delayed15"].to_numpy(),
            "pred_delay": oof_delay,
            "pred_proba": oof_proba,
        }
    )
    test_ids = set(test_df["flight_id"])
    all_pred["split"] = np.where(all_pred["flight_id"].isin(test_ids), "test", "train")
    all_pred.to_parquet(cfg.paths.predictions_parquet, index=False)

    # --- Feature importance (gain) ----------------------------------------
    importance = _feature_importance(lgbm_reg, lgbm_clf)
    metrics["feature_importance"] = importance

    # --- Persist ----------------------------------------------------------
    bundle = ModelBundle(
        lgbm_reg=lgbm_reg,
        lgbm_clf=lgbm_clf,
        rf_reg=rf_reg,
        rf_clf=rf_clf,
        stats=stats,
        feature_columns=FEATURE_COLUMNS,
        rf_columns=list(Xoh_tr.columns),
        best_params_reg=best_reg,
        best_params_clf=best_clf,
        weather_cfg=dict(cfg.synth.weather),
        weather_weights=dict(cfg.synth.weather_severity_weights),
    )
    bundle.save(cfg)
    write_feature_dict(cfg.paths.feature_dict_md)
    with open(cfg.paths.reports / "predict_metrics.json", "w", encoding="utf-8") as fh:
        json.dump(metrics, fh, indent=2)
    return metrics


def _oof_and_cv(
    cfg: Config, df: pd.DataFrame, best_reg: dict, best_clf: dict
) -> tuple[np.ndarray, np.ndarray, dict]:
    """One GroupKFold(tail_id) pass producing both out-of-fold predictions for
    every flight and the new-tail generalization metrics (MAE / PR-AUC)."""
    folds, _ = _cv_folds(df, cfg.predict.group_kfold_splits)
    n = len(df)
    oof_delay = np.zeros(n)
    oof_proba = np.zeros(n)
    maes, praucs = [], []
    for tr_idx, va_idx in folds:
        f_tr, f_va = df.iloc[tr_idx], df.iloc[va_idx]
        X_tr, yr_tr, yc_tr, stats = _features(cfg, f_tr, None)
        X_va, yr_va, yc_va, _ = _features(cfg, f_va, stats)
        reg, clf = _fit_reg_clf(cfg, X_tr, yr_tr, yc_tr, best_reg, best_clf)
        d = reg.predict(X_va)
        p = clf.predict_proba(X_va)[:, 1]
        oof_delay[va_idx] = d
        oof_proba[va_idx] = p
        maes.append(mean_absolute_error(yr_va, d))
        praucs.append(average_precision_score(yc_va, p))
    return (
        oof_delay,
        oof_proba,
        {
            "groupkfold_mae_mean": float(np.mean(maes)),
            "groupkfold_mae_std": float(np.std(maes)),
            "groupkfold_pr_auc_mean": float(np.mean(praucs)),
            "groupkfold_pr_auc_std": float(np.std(praucs)),
        },
    )


def _feature_importance(lgbm_reg, lgbm_clf) -> dict:
    def _imp(model):
        vals = model.booster_.feature_importance(importance_type="gain")
        names = model.booster_.feature_name()
        total = float(vals.sum()) or 1.0
        pairs = sorted(zip(names, vals), key=lambda kv: kv[1], reverse=True)
        return {n: float(v) / total for n, v in pairs}

    return {"regressor_gain": _imp(lgbm_reg), "classifier_gain": _imp(lgbm_clf)}


def predict(bundle: ModelBundle, cfg: Config, df: pd.DataFrame) -> dict:
    """Predict delay + risk probability for arbitrary flights using a bundle."""
    X, _, _, _ = build_features(
        df,
        weather_cfg=bundle.weather_cfg or cfg.synth.weather,
        weather_weights=bundle.weather_weights or cfg.synth.weather_severity_weights,
        features_cfg=cfg.features,
        fit_stats=bundle.stats,
    )
    return {
        "delay": bundle.lgbm_reg.predict(X),
        "risk_proba": bundle.lgbm_clf.predict_proba(X)[:, 1],
    }


# Categorical feature names are handled natively by LightGBM via pandas
# ``category`` dtype (see features.build_features); exported for reference.
__all__ = ["ModelBundle", "run_training", "predict", "time_split", "CATEGORICAL_FEATURES"]
