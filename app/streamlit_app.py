"""Streamlit control panel for the flight-delay-scheduling system.

Run with:  ``streamlit run app/streamlit_app.py``

The sidebar lets you (re)generate data, dial weather intensity / runway
capacity, pick the solver, and run the full predict -> grade -> schedule loop.
The main area shows the headline metrics, risk distribution, a prediction
sample, and every figure produced by the pipeline.
"""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
import streamlit as st

from flightopt.config import load_config
from flightopt.data import synth
from flightopt.evaluate import run_evaluate
from flightopt.predict import run_training
from flightopt.risk import run_grading
from flightopt.schedule import run_scheduling
from flightopt.viz import run_viz

st.set_page_config(page_title="Flight Delay & Scheduling", page_icon="✈️", layout="wide")


def _cfg(trials: int, capacity: int, weather_mult: float, solver: str):
    return load_config(
        overrides={
            "synth": {"coeffs": {
                "a0": -10.0, "a1": 21.0 * weather_mult, "a2": 15.0, "a3": 5.0,
                "a4": 12.0 * weather_mult, "a5": 0.70, "sigma": 4.5,
            }},
            "predict": {"optuna_trials": trials},
            "schedule": {"runway_capacity": capacity},
        }
    )


def _load_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8")) if path.exists() else {}


st.title("✈️ Flight Delay Prediction & Intelligent Scheduling")
st.caption("A decoupled predict → grade → schedule closed loop "
           "(LightGBM · quantile risk grading · OR-Tools CP-SAT).")

with st.sidebar:
    st.header("Controls")
    trials = st.slider("Optuna trials", 5, 60, 15, help="Fewer = faster")
    capacity = st.slider("Runway capacity (per airport / 15-min)", 2, 10, 5)
    weather_mult = st.slider("Weather intensity ×", 0.5, 2.0, 1.0, 0.1)
    solver = st.selectbox("Scheduler", ["cpsat", "greedy"])
    run = st.button("▶ Run full pipeline", type="primary", use_container_width=True)

cfg = _cfg(trials, capacity, weather_mult, solver)

if run:
    prog = st.progress(0.0, text="Generating data…")
    df = synth.load_or_generate(cfg, force=True)
    prog.progress(0.2, text="Training models…")
    run_training(cfg, df)
    prog.progress(0.55, text="Grading risk…")
    run_grading(cfg)
    prog.progress(0.7, text="Optimizing schedule…")
    run_scheduling(cfg, solver=solver)
    prog.progress(0.85, text="Evaluating…")
    run_evaluate(cfg)
    prog.progress(0.95, text="Rendering figures…")
    run_viz(cfg)
    prog.progress(1.0, text="Done")
    st.success("Pipeline complete.")

metrics = _load_json(cfg.paths.metrics_json)
head = metrics.get("headline", {})

if not head:
    st.info("No results yet — set your controls and click **Run full pipeline** "
            "(or run `python -m flightopt run-all` in a terminal).")
else:
    st.subheader("Headline metrics")
    c1, c2, c3, c4 = st.columns(4)
    cap = head.get("high_risk_capture_recall", {})
    sat = head.get("constraint_satisfaction_rate", {})
    red = head.get("high_risk_delay_reduction_min", {})
    err = head.get("prediction_error", {})
    if cap:
        c1.metric("High-risk capture (Recall)", f"{cap.get('value', 0):.3f}",
                  f"target ≥ {cap.get('target')}")
    if sat:
        c2.metric("Constraint satisfaction", f"{sat.get('value', 0):.3f}",
                  f"greedy {sat.get('baseline_greedy', 0):.3f}")
    if red:
        c3.metric("High-risk delay reduction", f"{red.get('value', 0):.2f} min",
                  f"target ≥ {red.get('target')}")
    if err:
        c4.metric("Prediction MAE (LightGBM)", f"{err.get('lightgbm_mae', 0):.2f}",
                  f"RF {err.get('random_forest_mae', 0):.2f}", delta_color="inverse")

    left, right = st.columns(2)
    with left:
        if cfg.paths.graded_parquet.exists():
            graded = pd.read_parquet(cfg.paths.graded_parquet)
            st.markdown("**Risk-level distribution**")
            st.bar_chart(graded["risk_level"].value_counts().sort_index())
        fig_dir = cfg.paths.figures
        for name, cap_txt in [
            ("schedule_convergence.png", "Optimization convergence"),
            ("feature_importance.png", "Feature importance (gain)"),
        ]:
            p = fig_dir / name
            if p.exists():
                st.image(str(p), caption=cap_txt, use_container_width=True)
    with right:
        for name, cap_txt in [
            ("gantt.gif", "Slot re-assignment (animated)"),
            ("schedule_before_after.png", "High-risk delay before/after"),
            ("shap_summary.png", "SHAP feature impact"),
        ]:
            p = cfg.paths.figures / name
            if p.exists():
                st.image(str(p), caption=cap_txt, use_container_width=True)

    if cfg.paths.schedule_parquet.exists():
        st.subheader("Optimized schedule (flights that moved)")
        sched = pd.read_parquet(cfg.paths.schedule_parquet)
        moved = sched[sched["offset_min"] != 0].copy()
        moved["Δdelay"] = (moved["pred_delay_before"] - moved["pred_delay_after"]).round(2)
        st.dataframe(
            moved[["flight_id", "tail_id", "origin", "risk_level", "high_risk",
                   "sched_dep", "offset_min", "new_dep", "Δdelay"]]
            .sort_values("Δdelay", ascending=False)
            .head(50),
            use_container_width=True,
        )
