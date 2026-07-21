"""Visualization: static figures, SHAP summary, and the Gantt animation.

All figures are written to ``outputs/figures``.  Small PNGs are committed; the
animated GIF is gitignored (regenerate with ``flightopt viz``).  Matplotlib runs
on the non-interactive ``Agg`` backend so everything works headless / in CI.
"""

from __future__ import annotations

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from matplotlib.animation import FuncAnimation, PillowWriter  # noqa: E402
from sklearn.metrics import (  # noqa: E402
    ConfusionMatrixDisplay,
    confusion_matrix,
    precision_recall_curve,
)

from flightopt.config import Config  # noqa: E402

_HR_COLOR = "#d1495b"
_NORMAL_COLOR = "#4472c4"


def _save(fig, path) -> None:
    fig.tight_layout()
    fig.savefig(path, dpi=120, bbox_inches="tight")
    plt.close(fig)


def plot_delay_distribution(flights: pd.DataFrame, path) -> None:
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.hist(flights["dep_delay"], bins=40, color=_NORMAL_COLOR, alpha=0.85)
    ax.axvline(15, color=_HR_COLOR, ls="--", label="15-min delay threshold")
    ax.set(xlabel="Departure delay (min)", ylabel="Flights", title="Departure-delay distribution")
    ax.legend()
    _save(fig, path)


def plot_feature_importance(gain: dict, path, title: str) -> None:
    items = list(gain.items())[:15][::-1]
    names = [k for k, _ in items]
    vals = [v for _, v in items]
    fig, ax = plt.subplots(figsize=(7, 5))
    ax.barh(names, vals, color=_NORMAL_COLOR)
    ax.set(xlabel="Normalized gain", title=title)
    _save(fig, path)


def plot_risk_levels(graded: pd.DataFrame, cfg: Config, path) -> None:
    counts = graded["risk_level"].value_counts().sort_index()
    pos_rate = graded.groupby("risk_level")["is_delayed15"].mean().reindex(counts.index)
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 4))
    colors = [_HR_COLOR if lvl in cfg.risk.high_risk_levels else _NORMAL_COLOR for lvl in counts.index]
    ax1.bar([f"L{i}" for i in counts.index], counts.values, color=colors)
    ax1.set(title="Flights per risk level", ylabel="Flights")
    ax2.bar([f"L{i}" for i in pos_rate.index], pos_rate.values, color=colors)
    ax2.set(title="Actual delayed>15 rate by level", ylabel="P(is_delayed15)")
    _save(fig, path)


def plot_classification_eval(preds: pd.DataFrame, path) -> None:
    test = preds[preds["split"] == "test"]
    y, proba = test["is_delayed15"].to_numpy(), test["pred_proba"].to_numpy()
    y_pred = (proba >= 0.5).astype(int)
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 4))
    cm = confusion_matrix(y, y_pred, labels=[0, 1])
    ConfusionMatrixDisplay(cm, display_labels=["on-time", "delayed>15"]).plot(
        ax=ax1, colorbar=False, cmap="Blues"
    )
    ax1.set_title("Confusion matrix (test)")
    prec, rec, _ = precision_recall_curve(y, proba)
    ax2.plot(rec, prec, color=_NORMAL_COLOR)
    ax2.set(xlabel="Recall", ylabel="Precision", title="Precision-Recall curve (test)")
    ax2.grid(alpha=0.3)
    _save(fig, path)


def plot_schedule_before_after(schedule_metrics: dict, path) -> None:
    comp = schedule_metrics.get("comparison", {})
    labels, before, after = [], [], []
    ref = comp.get("cpsat") or comp.get("greedy")
    base = ref["before"]["high_risk_mean_delay"] if ref else 0.0
    for name in ("cpsat", "greedy"):
        if name in comp:
            labels.append(name)
            before.append(base)
            after.append(comp[name]["after"]["high_risk_mean_delay"])
    x = np.arange(len(labels))
    w = 0.35
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.bar(x - w / 2, before, w, label="before", color="#b0b0b0")
    ax.bar(x + w / 2, after, w, label="after", color=_HR_COLOR)
    ax.set_xticks(x, labels)
    ax.set(ylabel="High-risk mean predicted delay (min)", title="High-risk delay: before vs after")
    for i, (b, a) in enumerate(zip(before, after)):
        ax.text(i + w / 2, a, f"-{b - a:.2f}", ha="center", va="bottom", fontsize=9)
    ax.legend()
    _save(fig, path)


def plot_convergence(trace: dict, path) -> None:
    fig, ax1 = plt.subplots(figsize=(8, 4.5))
    ax2 = ax1.twinx()
    for solver, style in (("cpsat", "-"), ("greedy", "--")):
        tr = trace.get(solver)
        if not tr:
            continue
        steps = [t["step"] for t in tr]
        delay = [t["weighted_total_delay"] for t in tr]
        sat = [t["satisfaction_rate"] for t in tr]
        ax1.plot(steps, delay, style, color=_NORMAL_COLOR, label=f"{solver} weighted delay")
        ax2.plot(steps, sat, style, color=_HR_COLOR, label=f"{solver} satisfaction")
    ax1.set(xlabel="Solver step", ylabel="Weighted total delay (min)")
    ax2.set_ylabel("Constraint satisfaction rate")
    ax1.legend(loc="upper right", fontsize=8)
    ax1.set_title("Optimization convergence")
    _save(fig, path)


def plot_shap_summary(cfg: Config, path) -> bool:
    """SHAP bar summary for the classifier; returns False if SHAP is unavailable."""
    try:
        import shap

        from flightopt.data import loader
        from flightopt.features import build_features
        from flightopt.predict import ModelBundle

        bundle = ModelBundle.load(cfg)
        flights = loader.load_flights(cfg)
        X, _, _, _ = build_features(flights, cfg, bundle.stats)
        Xs = X.sample(n=min(400, len(X)), random_state=cfg.seed)
        explainer = shap.TreeExplainer(bundle.lgbm_clf)
        sv = explainer.shap_values(Xs)
        if isinstance(sv, list):
            sv = sv[1] if len(sv) > 1 else sv[0]
        fig = plt.figure(figsize=(7, 5))
        shap.summary_plot(sv, Xs, plot_type="bar", show=False, max_display=15)
        fig.suptitle("SHAP feature impact (classifier)", y=1.02)
        _save(fig, path)
        return True
    except Exception as exc:  # pragma: no cover - SHAP is best-effort
        print(f"[viz] SHAP summary skipped: {exc}")
        return False


def gantt_animation(schedule_df: pd.DataFrame, trace: dict, cfg: Config, gif_path, png_path) -> None:
    """Two-panel animation: sample flights sliding to optimized slots (top) +
    the convergence trajectory drawn progressively (bottom).  Also saves a
    static PNG of the final state (committed; the GIF is gitignored)."""
    df = schedule_df.copy()
    df["dep_min"] = pd.to_datetime(df["sched_dep"]).dt.hour * 60 + pd.to_datetime(df["sched_dep"]).dt.minute
    # Sample: the high-risk flights that actually moved, plus a few normal ones.
    movers = df[(df["offset_min"] != 0)].reindex()
    hr = movers[movers["high_risk"]].sort_values("dep_min").head(22)
    nm = movers[~movers["high_risk"]].sort_values("dep_min").head(6)
    sample = pd.concat([hr, nm]).reset_index(drop=True)
    if sample.empty:
        sample = df.sort_values("dep_min").head(20).reset_index(drop=True)
    y = np.arange(len(sample))
    dep0 = sample["dep_min"].to_numpy() / 60.0
    off = sample["offset_min"].to_numpy() / 60.0
    colors = [_HR_COLOR if h else _NORMAL_COLOR for h in sample["high_risk"]]

    tr = trace.get("cpsat") or trace.get("greedy") or []
    steps = [t["step"] for t in tr] or [0]
    delay = [t["weighted_total_delay"] for t in tr] or [0]
    sat = [t["satisfaction_rate"] for t in tr] or [1]

    n_frames = 28
    fig, (ax_g, ax_c) = plt.subplots(2, 1, figsize=(9, 8), gridspec_kw={"height_ratios": [3, 2]})
    bars = ax_g.barh(y, 0.35, left=dep0, color=colors, edgecolor="black", linewidth=0.4)
    ax_g.scatter(dep0, y, marker="|", color="gray", s=200, label="original slot")
    ax_g.set(
        xlabel="Departure time (hour of day)",
        ylabel="Flight (sample)",
        title="Slot re-assignment (red = high risk)",
    )
    ax_g.set_yticks(y, sample["flight_id"], fontsize=6)
    ax_c.set(xlabel="Solver step", ylabel="Weighted delay (min)")
    ax_c2 = ax_c.twinx()
    ax_c2.set_ylabel("Satisfaction")
    (line_d,) = ax_c.plot([], [], color=_NORMAL_COLOR, label="weighted delay")
    (line_s,) = ax_c2.plot([], [], color=_HR_COLOR, label="satisfaction")
    ax_c.set_xlim(min(steps), max(steps) + 0.01)
    ax_c.set_ylim(min(delay) * 0.98, max(delay) * 1.02)
    ax_c2.set_ylim(min(sat) * 0.99, 1.001)

    def update(frame):
        prog = frame / (n_frames - 1)
        for bar, d0, o in zip(bars, dep0, off):
            bar.set_x(d0 + o * prog)
        k = max(1, int(len(steps) * prog))
        line_d.set_data(steps[:k], delay[:k])
        line_s.set_data(steps[:k], sat[:k])
        return [*bars, line_d, line_s]

    anim = FuncAnimation(fig, update, frames=n_frames, blit=False, interval=120)
    try:
        anim.save(gif_path, writer=PillowWriter(fps=8))
    except Exception as exc:  # pragma: no cover
        print(f"[viz] GIF export skipped: {exc}")
    update(n_frames - 1)  # render final state for the static PNG
    _save(fig, png_path)


def gantt_interactive(schedule_df: pd.DataFrame, path) -> None:
    """Interactive Plotly timeline (HTML) of a sample of re-scheduled flights."""
    try:
        import plotly.express as px

        df = schedule_df.copy()
        movers = df[df["offset_min"] != 0].sort_values("sched_dep").head(40)
        if movers.empty:
            movers = df.sort_values("sched_dep").head(40)
        dur = pd.to_timedelta(20, unit="m")
        rows = []
        for _, r in movers.iterrows():
            rows.append({"flight": r["flight_id"], "start": r["sched_dep"], "end": r["sched_dep"] + dur,
                         "state": "original", "risk": f"L{r['risk_level']}"})
            rows.append({"flight": r["flight_id"], "start": r["new_dep"], "end": r["new_dep"] + dur,
                         "state": "optimized", "risk": f"L{r['risk_level']}"})
        tl = pd.DataFrame(rows)
        fig = px.timeline(
            tl, x_start="start", x_end="end", y="flight", color="state",
            title="Original vs optimized departure slots (sample)",
        )
        fig.update_yaxes(autorange="reversed")
        fig.write_html(str(path), include_plotlyjs="cdn")
    except Exception as exc:  # pragma: no cover
        print(f"[viz] interactive gantt skipped: {exc}")


def run_viz(cfg: Config) -> list[str]:
    """Produce every figure from the persisted artifacts; returns file paths."""
    import json

    from flightopt.data import loader

    cfg.paths.ensure()
    fig_dir = cfg.paths.figures
    produced: list[str] = []

    flights = loader.load_flights(cfg)
    plot_delay_distribution(flights, fig_dir / "delay_distribution.png")
    produced.append("delay_distribution.png")

    pm_path = cfg.paths.reports / "predict_metrics.json"
    if pm_path.exists():
        pm = json.loads(pm_path.read_text(encoding="utf-8"))
        gain = pm.get("feature_importance", {}).get("regressor_gain", {})
        if gain:
            plot_feature_importance(gain, fig_dir / "feature_importance.png",
                                    "Regressor feature importance (gain)")
            produced.append("feature_importance.png")

    if cfg.paths.predictions_parquet.exists():
        preds = pd.read_parquet(cfg.paths.predictions_parquet)
        plot_classification_eval(preds, fig_dir / "classification_eval.png")
        produced.append("classification_eval.png")

    if cfg.paths.graded_parquet.exists():
        graded = pd.read_parquet(cfg.paths.graded_parquet)
        plot_risk_levels(graded, cfg, fig_dir / "risk_levels.png")
        produced.append("risk_levels.png")

    if plot_shap_summary(cfg, fig_dir / "shap_summary.png"):
        produced.append("shap_summary.png")

    sm_path = cfg.paths.reports / "schedule_metrics.json"
    if sm_path.exists():
        sm = json.loads(sm_path.read_text(encoding="utf-8"))
        plot_schedule_before_after(sm, fig_dir / "schedule_before_after.png")
        produced.append("schedule_before_after.png")
        trace = sm.get("trace", {})
        plot_convergence(trace, fig_dir / "schedule_convergence.png")
        produced.append("schedule_convergence.png")
        if cfg.paths.schedule_parquet.exists():
            schedule_df = pd.read_parquet(cfg.paths.schedule_parquet)
            gantt_animation(schedule_df, trace, cfg, fig_dir / "gantt.gif",
                            fig_dir / "gantt_final.png")
            produced += ["gantt.gif", "gantt_final.png"]
            gantt_interactive(schedule_df, fig_dir / "gantt_interactive.html")
            produced.append("gantt_interactive.html")

    return produced
