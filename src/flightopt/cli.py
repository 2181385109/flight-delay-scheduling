"""Command-line interface (Typer).

    python -m flightopt gen-data
    python -m flightopt train [--trials N]
    python -m flightopt grade
    python -m flightopt schedule [--solver cpsat|greedy]
    python -m flightopt evaluate
    python -m flightopt viz
    python -m flightopt run-all [--trials N]
"""

from __future__ import annotations

import time

import typer

from flightopt.config import load_config

app = typer.Typer(
    add_completion=False,
    help="Flight delay prediction & intelligent scheduling optimization.",
    no_args_is_help=True,
)


def _load(config: str | None, seed: int | None = None, overrides: dict | None = None):
    ov = dict(overrides or {})
    if seed is not None:
        ov["seed"] = seed
    return load_config(config, overrides=ov or None)


@app.command("gen-data")
def gen_data(
    config: str = typer.Option(None, help="Path to config.yaml"),
    seed: int = typer.Option(None, help="Override the global seed"),
    force: bool = typer.Option(True, help="Regenerate even if cached"),
):
    """Generate the synthetic flight dataset."""
    from flightopt.data import synth

    cfg = _load(config, seed)
    t0 = time.time()
    df = synth.load_or_generate(cfg, force=force)
    s = synth.summarize(df)
    typer.echo(f"Generated {s['n_flights']} flights / {s['n_tails']} tails "
               f"-> {cfg.paths.flights_parquet}")
    typer.echo(f"  positive rate (is_delayed15): {s['positive_rate']:.3f}")
    typer.echo(f"  corr(weather/congestion/prev_leg vs delay): "
               f"{s['corr_weather']:.2f} / {s['corr_congestion']:.2f} / {s['corr_prev_leg']:.2f}")
    typer.echo(f"  done in {time.time() - t0:.2f}s")


@app.command()
def train(
    config: str = typer.Option(None, help="Path to config.yaml"),
    seed: int = typer.Option(None, help="Override the global seed"),
    trials: int = typer.Option(None, help="Override Optuna trials"),
):
    """Train the LightGBM double-head (+ baselines) with Optuna tuning."""
    from flightopt.data import synth
    from flightopt.predict import run_training

    ov = {"predict": {"optuna_trials": trials}} if trials else None
    cfg = _load(config, seed, ov)
    t0 = time.time()
    df = synth.load_or_generate(cfg)
    m = run_training(cfg, df)
    reg = m["regression"]
    clf = m["classification"]
    typer.echo(f"Trained in {time.time() - t0:.1f}s")
    typer.echo(f"  REG  MAE  lgbm={reg['lightgbm']['mae']:.3f}  "
               f"rf={reg['random_forest']['mae']:.3f}  mean={reg['mean_baseline']['mae']:.3f}")
    typer.echo(f"  CLF  PR-AUC lgbm={clf['lightgbm']['pr_auc']:.3f}  "
               f"rf={clf['random_forest']['pr_auc']:.3f}")
    typer.echo(f"  models -> {cfg.paths.models / 'bundle.pkl'}")


@app.command()
def grade(
    config: str = typer.Option(None, help="Path to config.yaml"),
    seed: int = typer.Option(None, help="Override the global seed"),
):
    """Grade flights into 5 quantile risk levels (L4/L5 = high risk)."""
    from flightopt.risk import run_grading

    cfg = _load(config, seed)
    m = run_grading(cfg)
    cap = m["capture_test"]
    typer.echo(f"Risk grading (source={m['source']}), cutpoints={[round(c, 2) for c in m['cutpoints']]}")
    typer.echo(f"  high-risk capture (test): recall={cap['high_risk_recall']:.3f} "
               f"precision={cap['high_risk_precision']:.3f} f1={cap['high_risk_f1']:.3f}")
    typer.echo(f"  rule baseline recall (test): {m['rule_baseline_test']['high_risk_recall']:.3f}")


@app.command()
def schedule(
    config: str = typer.Option(None, help="Path to config.yaml"),
    seed: int = typer.Option(None, help="Override the global seed"),
    solver: str = typer.Option("cpsat", help="cpsat | greedy"),
):
    """Optimize departure slots (CP-SAT primary, greedy baseline for contrast)."""
    from flightopt.schedule import run_scheduling

    cfg = _load(config, seed)
    t0 = time.time()
    res = run_scheduling(cfg, solver=solver)
    typer.echo(f"Scheduling ({solver}) done in {time.time() - t0:.1f}s")
    for name, m in res["comparison"].items():
        a = m["after"]
        typer.echo(f"  [{name}] high-risk delay {a['high_risk_mean_delay_before']:.2f}"
                   f"->{a['high_risk_mean_delay']:.2f} (-{a['high_risk_delay_reduction']:.2f} min) "
                   f"| satisfaction {a['satisfaction_rate']:.3f} "
                   f"| capacity violations {m['before']['capacity_violations']}->{a['capacity_violations']}")


@app.command()
def evaluate(
    config: str = typer.Option(None, help="Path to config.yaml"),
    seed: int = typer.Option(None, help="Override the global seed"),
):
    """Aggregate metrics, write report.md/metrics.json, update README."""
    from flightopt.evaluate import headline_table, run_evaluate

    cfg = _load(config, seed)
    m = run_evaluate(cfg)
    typer.echo(headline_table(m))
    typer.echo(f"\nmetrics -> {cfg.paths.metrics_json}")
    typer.echo(f"report  -> {cfg.paths.report_md}")


@app.command()
def viz(
    config: str = typer.Option(None, help="Path to config.yaml"),
    seed: int = typer.Option(None, help="Override the global seed"),
):
    """Render static figures, SHAP summary, and the Gantt animation."""
    from flightopt.viz import run_viz

    cfg = _load(config, seed)
    t0 = time.time()
    files = run_viz(cfg)
    typer.echo(f"Rendered {len(files)} figures in {time.time() - t0:.1f}s -> {cfg.paths.figures}")
    for f in files:
        typer.echo(f"  {f}")


@app.command("run-all")
def run_all(
    config: str = typer.Option(None, help="Path to config.yaml"),
    seed: int = typer.Option(None, help="Override the global seed"),
    trials: int = typer.Option(None, help="Override Optuna trials"),
    solver: str = typer.Option("cpsat", help="Scheduling solver"),
):
    """Run the full pipeline: gen-data -> train -> grade -> schedule -> evaluate -> viz."""
    from flightopt.data import synth
    from flightopt.evaluate import headline_table, run_evaluate
    from flightopt.predict import run_training
    from flightopt.risk import run_grading
    from flightopt.schedule import run_scheduling
    from flightopt.viz import run_viz

    ov = {"predict": {"optuna_trials": trials}} if trials else None
    cfg = _load(config, seed, ov)
    t0 = time.time()

    typer.echo("==> [1/6] generating data")
    df = synth.load_or_generate(cfg, force=True)
    typer.echo("==> [2/6] training models")
    run_training(cfg, df)
    typer.echo("==> [3/6] grading risk")
    run_grading(cfg)
    typer.echo("==> [4/6] scheduling optimization")
    run_scheduling(cfg, solver=solver)
    typer.echo("==> [5/6] evaluating")
    metrics = run_evaluate(cfg)
    typer.echo("==> [6/6] rendering figures")
    run_viz(cfg)

    typer.echo("\n" + headline_table(metrics))
    typer.echo(f"\nrun-all complete in {time.time() - t0:.1f}s. "
               f"Artifacts in {cfg.paths.outputs}")


if __name__ == "__main__":
    app()
