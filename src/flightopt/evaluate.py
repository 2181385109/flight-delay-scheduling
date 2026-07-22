"""Evaluation: aggregate stage metrics, emit ``metrics.json`` + ``report.md``,
and write the headline table back into ``README.md``.

Every headline metric carries an explicit definition, a baseline comparison and
a pass/fail flag against the configured target.
"""

from __future__ import annotations

import json
from pathlib import Path

from flightopt.config import Config

_README_START = "<!-- METRICS:START -->"
_README_END = "<!-- METRICS:END -->"


def _load_json(path: Path) -> dict:
    if path.exists():
        with open(path, "r", encoding="utf-8") as fh:
            return json.load(fh)
    return {}


def summarize(cfg: Config) -> dict:
    """Assemble the unified metrics dict from the per-stage report files."""
    pm = _load_json(cfg.paths.reports / "predict_metrics.json")
    rm = _load_json(cfg.paths.reports / "risk_metrics.json")
    sm = _load_json(cfg.paths.reports / "schedule_metrics.json")

    tgt = cfg.metrics
    headline: dict = {}

    # 1) High-risk capture rate (recall of L4/L5 vs the delay label).
    cap = rm.get("capture_test", {})
    rule = rm.get("rule_baseline_test", {})
    if cap:
        headline["high_risk_capture_recall"] = {
            "definition": "Recall of the high-risk (L4/L5) grade vs is_delayed15 on the held-out test split.",
            "value": cap.get("high_risk_recall"),
            "precision": cap.get("high_risk_precision"),
            "f1": cap.get("high_risk_f1"),
            "target": tgt.high_risk_recall_target,
            "baseline_rule_recall": rule.get("high_risk_recall"),
            "pass": _ge(cap.get("high_risk_recall"), tgt.high_risk_recall_target),
            "operating_point": rm.get("operating_point", {}),
        }

    # 2) Scheduling constraint satisfaction (CP-SAT vs greedy).
    comp = sm.get("comparison", {})
    cpsat = comp.get("cpsat", {})
    greedy = comp.get("greedy", {})
    if cpsat:
        after = cpsat.get("after", {})
        g_after = greedy.get("after", {})
        headline["constraint_satisfaction_rate"] = {
            "definition": "Satisfied constraints / total (turnaround + curfew + runway-capacity windows) after CP-SAT.",
            "value": after.get("satisfaction_rate"),
            "capacity_rate": after.get("capacity_rate"),
            "capacity_violations_before": cpsat.get("before", {}).get("capacity_violations"),
            "capacity_violations_after": after.get("capacity_violations"),
            "target": tgt.constraint_satisfaction_target,
            "baseline_greedy": g_after.get("satisfaction_rate"),
            "pass": _ge(after.get("satisfaction_rate"), tgt.constraint_satisfaction_target),
        }
        # 3) High-risk delay reduction (CP-SAT vs before, vs greedy).
        before = cpsat.get("before", {})
        tot_before = before.get("total_delay")
        tot_after = after.get("total_delay")
        tot_pct = (
            (tot_before - tot_after) / tot_before * 100.0
            if tot_before and tot_after is not None
            else None
        )
        headline["high_risk_delay_reduction_min"] = {
            "definition": "Mean predicted-delay reduction per high-risk flight (offset 0 -> optimized offset).",
            "value": after.get("high_risk_delay_reduction"),
            "mean_delay_before": after.get("high_risk_mean_delay_before"),
            "mean_delay_after": after.get("high_risk_mean_delay"),
            "target": tgt.delay_reduction_target,
            "baseline_greedy": g_after.get("high_risk_delay_reduction"),
            "pass": _ge(after.get("high_risk_delay_reduction"), tgt.delay_reduction_target),
            # All-flights total predicted delay reduction (reduction in *predicted*
            # delay, not measured real-world delay).
            "total_delay_before": tot_before,
            "total_delay_after": tot_after,
            "total_delay_reduction_pct": tot_pct,
            "operating_day": sm.get("operating_day"),
            "n_flights_day": sm.get("n_flights"),
        }

    # 4) Prediction error (auxiliary): LightGBM vs RF vs mean baseline.
    reg = pm.get("regression", {})
    clf = pm.get("classification", {})
    if reg:
        headline["prediction_error"] = {
            "definition": "Regression MAE / RMSE on the held-out test split (minutes).",
            "lightgbm_mae": reg.get("lightgbm", {}).get("mae"),
            "lightgbm_rmse": reg.get("lightgbm", {}).get("rmse"),
            "random_forest_mae": reg.get("random_forest", {}).get("mae"),
            "mean_baseline_mae": reg.get("mean_baseline", {}).get("mae"),
            "median_baseline_mae": reg.get("median_baseline", {}).get("mae"),
            "beats_baselines": _lt(
                reg.get("lightgbm", {}).get("mae"),
                min(
                    reg.get("random_forest", {}).get("mae", float("inf")),
                    reg.get("mean_baseline", {}).get("mae", float("inf")),
                    reg.get("median_baseline", {}).get("mae", float("inf")),
                ),
            ),
        }
        if clf:
            headline["classification_quality"] = {
                "definition": "Binary P(delay>15) quality on the test split.",
                "lightgbm_pr_auc": clf.get("lightgbm", {}).get("pr_auc"),
                "lightgbm_roc_auc": clf.get("lightgbm", {}).get("roc_auc"),
                "random_forest_pr_auc": clf.get("random_forest", {}).get("pr_auc"),
            }

    return {
        "headline": headline,
        "prediction": pm,
        "risk": rm,
        "schedule": sm,
    }


def _ge(value, target) -> bool | None:
    return None if value is None else bool(value >= target)


def _lt(value, target) -> bool | None:
    return None if value is None else bool(value < target)


def _fmt(x, nd: int = 3) -> str:
    if x is None:
        return "n/a"
    if isinstance(x, bool):
        return "PASS" if x else "MISS"
    if isinstance(x, (int, float)):
        return f"{x:.{nd}f}"
    return str(x)


def headline_table(metrics: dict) -> str:
    """Markdown table of the four headline metrics (for README + report)."""
    h = metrics.get("headline", {})
    rows = [
        "| Metric | Definition | Result | Target | Baseline | Status |",
        "|---|---|---|---|---|---|",
    ]
    cap = h.get("high_risk_capture_recall")
    if cap:
        op = cap.get("operating_point") or {}
        note = ""
        if op.get("flag_rate_for_target_recall") is not None and not cap.get("pass", True):
            note = (f" — reaching {_fmt(op.get('target_recall'),2)} needs "
                    f"{_fmt(op.get('flag_rate_for_target_recall'),2)} of flights flagged")
        rows.append(
            f"| High-risk capture (Recall) | L4/L5 recall vs `is_delayed15` (test) | "
            f"**{_fmt(cap['value'])}** (P={_fmt(cap['precision'])}, F1={_fmt(cap['f1'])}) | "
            f"≥ {_fmt(cap['target'],2)} | rule {_fmt(cap['baseline_rule_recall'])} | "
            f"{_fmt(cap['pass'])}{note} |"
        )
    sat = h.get("constraint_satisfaction_rate")
    if sat:
        rows.append(
            f"| Constraint satisfaction | satisfied / total constraints (CP-SAT) | "
            f"**{_fmt(sat['value'])}** (cap.viol {sat['capacity_violations_before']}→{sat['capacity_violations_after']}) | "
            f"≥ {_fmt(sat['target'],2)} | greedy {_fmt(sat['baseline_greedy'])} | {_fmt(sat['pass'])} |"
        )
    red = h.get("high_risk_delay_reduction_min")
    if red:
        tot = red.get("total_delay_reduction_pct")
        tot_note = (f"; all-flights total predicted delay −{_fmt(tot,2)}%"
                    if tot is not None else "")
        rows.append(
            f"| High-risk delay reduction | mean Δ predicted delay/flight (CP-SAT) | "
            f"**{_fmt(red['value'],2)} min** ({_fmt(red['mean_delay_before'],1)}→"
            f"{_fmt(red['mean_delay_after'],1)}{tot_note}) | "
            f"≥ {_fmt(red['target'],1)} min | greedy {_fmt(red['baseline_greedy'],2)} | {_fmt(red['pass'])} |"
        )
    err = h.get("prediction_error")
    if err:
        rows.append(
            f"| Prediction error (MAE) | test-set MAE, minutes | "
            f"**{_fmt(err['lightgbm_mae'],2)}** | lower is better | "
            f"median {_fmt(err.get('median_baseline_mae'),2)} / RF {_fmt(err['random_forest_mae'],2)} "
            f"/ mean {_fmt(err['mean_baseline_mae'],2)} | "
            f"{_fmt(err['beats_baselines'])} |"
        )
    return "\n".join(rows)


def write_report(cfg: Config, metrics: dict) -> None:
    """Human-readable ``report.md`` with the headline table + detail."""
    pm = metrics.get("prediction", {})
    lines = [
        "# Results Report",
        "",
        "## Headline metrics",
        "",
        headline_table(metrics),
        "",
        "## Prediction detail (held-out test split)",
        "",
        "| Model | MAE | RMSE |",
        "|---|---|---|",
    ]
    reg = pm.get("regression", {})
    for name in ("lightgbm", "random_forest", "median_baseline", "mean_baseline"):
        m = reg.get(name, {})
        lines.append(f"| {name} | {_fmt(m.get('mae'),3)} | {_fmt(m.get('rmse'),3)} |")
    lines += ["", "| Model | Precision | Recall | F1 | PR-AUC | ROC-AUC |", "|---|---|---|---|---|---|"]
    clf = pm.get("classification", {})
    for name in ("lightgbm", "random_forest", "rule_baseline"):
        m = clf.get(name, {})
        lines.append(
            f"| {name} | {_fmt(m.get('precision'))} | {_fmt(m.get('recall'))} | "
            f"{_fmt(m.get('f1'))} | {_fmt(m.get('pr_auc'))} | {_fmt(m.get('roc_auc'))} |"
        )
    cv = pm.get("cv_generalization", {})
    if cv:
        lines += [
            "",
            "## New-tail generalization (GroupKFold on tail_id)",
            "",
            f"- MAE: {_fmt(cv.get('groupkfold_mae_mean'),3)} ± {_fmt(cv.get('groupkfold_mae_std'),3)}",
            f"- PR-AUC: {_fmt(cv.get('groupkfold_pr_auc_mean'),3)} ± {_fmt(cv.get('groupkfold_pr_auc_std'),3)}",
        ]
    fi = pm.get("feature_importance", {}).get("regressor_gain", {})
    if fi:
        top = list(fi.items())[:8]
        lines += ["", "## Top regressor features (gain)", ""]
        lines += [f"- `{name}`: {_fmt(val)}" for name, val in top]

    schedule_block = metrics.get("schedule", {})
    sched = schedule_block.get("comparison", {})
    if sched:
        day = schedule_block.get("operating_day")
        n_day = schedule_block.get("n_flights")
        lines += [
            "",
            "## Scheduling: CP-SAT vs greedy vs before",
            "",
            f"Re-timing one real operating day (**{day}**, {n_day} flights). "
            "Baselines: the original as-flown schedule (`before`) and a greedy "
            "local search. Delay figures are reductions in *predicted* delay.",
            "",
            "| Solver | High-risk mean delay (min) | Δ/flight | Total delay (all, min) | Total Δ | Satisfaction | Capacity violations |",
            "|---|---|---|---|---|---|---|",
        ]
        before = (sched.get("cpsat") or sched.get("greedy") or {}).get("before", {})
        tot_b = before.get("total_delay")
        lines.append(
            f"| before | {_fmt(before.get('high_risk_mean_delay'),2)} | - | "
            f"{_fmt(tot_b,1)} | - | {_fmt(before.get('satisfaction_rate'))} | "
            f"{before.get('capacity_violations')} |"
        )
        for name in ("cpsat", "greedy"):
            s = sched.get(name)
            if s:
                a = s["after"]
                tot_a = a.get("total_delay")
                pct = ((tot_b - tot_a) / tot_b * 100.0) if tot_b and tot_a is not None else None
                lines.append(
                    f"| {name} | {_fmt(a.get('high_risk_mean_delay'),2)} | "
                    f"{_fmt(a.get('high_risk_delay_reduction'),2)} | {_fmt(tot_a,1)} | "
                    f"−{_fmt(pct,2)}% | {_fmt(a.get('satisfaction_rate'))} | "
                    f"{a.get('capacity_violations')} |"
                )
    lines.append("")
    with open(cfg.paths.report_md, "w", encoding="utf-8") as fh:
        fh.write("\n".join(lines))


def update_readme(cfg: Config, metrics: dict) -> bool:
    """Replace the README placeholder block with the live headline table."""
    readme = cfg.paths.root / "README.md"
    if not readme.exists():
        return False
    text = readme.read_text(encoding="utf-8")
    if _README_START not in text or _README_END not in text:
        return False
    table = headline_table(metrics)
    block = f"{_README_START}\n\n{table}\n\n_Auto-generated by `flightopt evaluate`._\n\n{_README_END}"
    pre = text.split(_README_START)[0]
    post = text.split(_README_END)[1]
    readme.write_text(pre + block + post, encoding="utf-8")
    return True


def run_evaluate(cfg: Config) -> dict:
    cfg.paths.ensure()
    metrics = summarize(cfg)
    with open(cfg.paths.metrics_json, "w", encoding="utf-8") as fh:
        json.dump(metrics, fh, indent=2)
    write_report(cfg, metrics)
    update_readme(cfg, metrics)
    return metrics
