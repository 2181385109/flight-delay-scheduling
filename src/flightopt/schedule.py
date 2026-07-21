"""Scheduling optimization: OR-Tools CP-SAT vs a greedy baseline.

Problem (ground-delay / slot assignment)
----------------------------------------
Each flight may take a departure-slot **offset** ``o in {-K, ..., +K}`` (5-min
grid).  The predicted delay of a flight at a candidate offset,
``D[f][o]``, is re-scored with the trained LightGBM regressor: shifting a flight
changes its departure hour (peak flag, cyclical time) and, crucially, the
airport congestion it experiences (evaluated against the *static* schedule of
the other flights).  This turns the coupled problem into a clean **assignment**
whose objective is ``min sum_f w_f * D[f][o_f]`` (high-risk flights carry a
larger weight ``w_f``).

Constraints
-----------
* **Turnaround** (hard): consecutive legs of a tail keep
  ``dep_next + o_next >= arr_prev + o_prev + min_turnaround``.
* **Curfew** (hard): no departure inside the curfew window.
* **Window** (hard): ``|o_f| <= K`` (encoded in the offset domain).
* **Runway capacity** (soft): departures per airport per time-window ``<= C``;
  excess is penalized so the satisfaction rate is reported honestly.

Solvers: **CP-SAT** (primary, feasibility-guaranteed on the hard constraints)
and a **greedy** local search baseline.
"""

from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass

import numpy as np
import pandas as pd
from ortools.sat.python import cp_model

from flightopt.config import Config
from flightopt.features import (
    build_features,
    cyclical_hour,
    is_peak_hour,
    time_bucket,
)
from flightopt.predict import ModelBundle

_SCALE = 100  # float minutes -> integer centi-minutes for CP-SAT
_EPS = 1e-6


# ---------------------------------------------------------------------------
# Per-flight metadata used by the solvers / evaluators.
# ---------------------------------------------------------------------------
@dataclass
class ScheduleProblem:
    flight_id: np.ndarray
    tail_id: np.ndarray
    leg_index: np.ndarray
    origin: np.ndarray
    dep_min: np.ndarray          # absolute scheduled departure minute
    arr_min: np.ndarray          # absolute scheduled arrival minute
    min_turnaround: np.ndarray
    high_risk: np.ndarray        # bool
    risk_level: np.ndarray
    weight: np.ndarray           # objective weight per flight
    D: np.ndarray                # (n_flights, n_offsets) predicted delay minutes
    offsets: list[int]
    feasible: np.ndarray         # (n_flights, n_offsets) bool (curfew feasibility)
    optimized: np.ndarray        # bool: flight participates in optimization
    # (prev_idx, next_idx, effective_min_turnaround) for consecutive same-day legs
    tail_pairs: list[tuple[int, int, int]]

    @property
    def n(self) -> int:
        return len(self.flight_id)

    @property
    def zero_idx(self) -> int:
        return self.offsets.index(0)


def _in_curfew(minute: np.ndarray, curfew: list[int]) -> np.ndarray:
    hour = (np.asarray(minute) // 60) % 24
    c0, c1 = curfew
    if c0 <= c1:
        return (hour >= c0) & (hour < c1)
    return (hour >= c0) | (hour < c1)


def _congestion_counts(
    origin: np.ndarray,
    query_min: np.ndarray,
    per_origin_sorted: dict[str, np.ndarray],
    window: int,
) -> np.ndarray:
    counts = np.zeros(len(query_min), dtype=float)
    for ap, arr in per_origin_sorted.items():
        mask = origin == ap
        if not mask.any():
            continue
        q = query_min[mask]
        lo = np.searchsorted(arr, q - window, side="left")
        hi = np.searchsorted(arr, q + window, side="right")
        counts[mask] = (hi - lo).astype(float)
    return counts


def build_problem(
    cfg: Config,
    flights: pd.DataFrame,
    graded: pd.DataFrame,
    bundle: ModelBundle,
) -> ScheduleProblem:
    """Assemble the delay lookup table + metadata for the solvers."""
    flights = flights.reset_index(drop=True)
    g = graded.set_index("flight_id")
    high_risk = flights["flight_id"].map(g["high_risk"]).fillna(False).to_numpy().astype(bool)
    risk_level = flights["flight_id"].map(g["risk_level"]).fillna(1).to_numpy().astype(int)

    dep = pd.to_datetime(flights["sched_dep"])
    arr = pd.to_datetime(flights["sched_arr"])
    day0 = dep.dt.normalize().min()
    dep_min = ((dep - day0).dt.total_seconds() // 60).to_numpy().astype(int)
    arr_min = ((arr - day0).dt.total_seconds() // 60).to_numpy().astype(int)
    origin = flights["origin"].to_numpy()

    # Base features (offset 0) and the columns that shift with the offset.
    stats = bundle.stats
    X0, _, _, _ = build_features(flights, cfg, stats)
    wsev = X0["weather_severity"].to_numpy()
    per_origin_sorted = {
        ap: np.sort(dep_min[origin == ap]) for ap in np.unique(origin)
    }
    window = cfg.features.congestion_window_min
    c_min, c_max = stats["congestion_min"], stats["congestion_max"]
    denom = max(c_max - c_min, 1e-9)
    bucket_cats = stats["categories"]["time_bucket"]

    offsets = cfg.schedule.offsets
    n, K = len(flights), len(offsets)
    feasible = np.ones((n, K), dtype=bool)
    frames = []
    for j, o in enumerate(offsets):
        new_min = dep_min + o
        new_hour = (new_min // 60) % 24
        sin_h, cos_h = cyclical_hour(new_hour)
        peak = is_peak_hour(new_hour, cfg.features.peak_windows)
        raw = _congestion_counts(origin, new_min, per_origin_sorted, window)
        cong = np.clip((raw - c_min) / denom, 0.0, 1.0)

        Xo = X0.copy()
        Xo["dep_hour_sin"] = sin_h
        Xo["dep_hour_cos"] = cos_h
        Xo["is_peak"] = peak
        Xo["airport_congestion"] = cong
        Xo["wsev_x_peak"] = wsev * peak
        Xo["cong_x_peak"] = cong * peak
        Xo["time_bucket"] = pd.Categorical(
            time_bucket(new_hour, cfg.features.time_buckets), categories=bucket_cats
        )
        # The origin x hour historical rate also moves with the slot.
        oh_map, oh_def = stats["encodings"]["origin_hour_delay_rate"]
        Xo["origin_hour_delay_rate"] = [
            oh_map.get((ap, int(h)), oh_def) for ap, h in zip(origin, new_hour)
        ]
        frames.append(Xo)
        feasible[:, j] = ~_in_curfew(new_min, cfg.schedule.curfew_hours)
    # The original schedule is always a valid input (offset 0 grandfathered).
    feasible[:, offsets.index(0)] = True

    big = pd.concat(frames, ignore_index=True)
    preds = np.clip(bundle.lgbm_reg.predict(big), 0.0, None)
    D = preds.reshape(K, n).T  # (n_flights, n_offsets)

    # Weights + which flights are optimized.
    weight = np.where(high_risk, cfg.schedule.weight_high_risk, cfg.schedule.weight_normal)
    optimized = high_risk.copy() if cfg.schedule.only_high_risk else np.ones(n, dtype=bool)

    # Consecutive same-day leg pairs per tail (turnaround constraints).
    #
    # Real schedules sometimes plan a turnaround shorter than our assumed
    # minimum, so demanding `min_turnaround` outright would make the model
    # infeasible on real data. The operationally correct constraint for
    # *re-timing an existing schedule* is: never shorten a turnaround below
    # what was already planned, and never below the minimum -- whichever is
    # smaller. This holds trivially at offset 0, so the original schedule is
    # always a feasible starting point.
    min_turn = flights["min_turnaround"].to_numpy().astype(int)
    tail_pairs: list[tuple[int, int, int]] = []
    tmp = flights[["tail_id"]].copy()
    tmp["_day"] = dep.dt.date
    tmp["_t"] = dep_min
    for _, grp in tmp.sort_values(["tail_id", "_day", "_t"]).groupby(
        ["tail_id", "_day"], sort=False
    ):
        idx = grp.index.to_numpy()
        for a, b in zip(idx[:-1], idx[1:]):
            a, b = int(a), int(b)
            # NOTE: the planned gap can be negative in real data (the next
            # departure precedes the previous arrival). It is *not* clamped at
            # zero, so the constraint holds with equality at offset 0 and the
            # original schedule stays feasible for every solver.
            planned_gap = int(dep_min[b] - arr_min[a])
            tail_pairs.append((a, b, int(min(min_turn[b], planned_gap))))

    return ScheduleProblem(
        flight_id=flights["flight_id"].to_numpy(),
        tail_id=flights["tail_id"].to_numpy(),
        leg_index=flights["leg_index"].to_numpy(),
        origin=origin,
        dep_min=dep_min,
        arr_min=arr_min,
        min_turnaround=flights["min_turnaround"].to_numpy().astype(int),
        high_risk=high_risk,
        risk_level=risk_level,
        weight=weight,
        D=D,
        offsets=offsets,
        feasible=feasible,
        optimized=optimized,
        tail_pairs=tail_pairs,
    )


# ---------------------------------------------------------------------------
# Evaluation of a candidate offset assignment.
# ---------------------------------------------------------------------------
def constraint_report(p: ScheduleProblem, off: np.ndarray, cfg: Config) -> dict:
    """Constraint-satisfaction breakdown for an offset vector (minutes)."""
    window = cfg.schedule.window_minutes
    cap = cfg.schedule.runway_capacity

    # Turnaround (hard).
    turn_total = len(p.tail_pairs)
    turn_ok = 0
    for a, b, eff_turn in p.tail_pairs:
        if (p.dep_min[b] + off[b]) >= (p.arr_min[a] + off[a]) + eff_turn - _EPS:
            turn_ok += 1

    # Curfew (hard).
    new_min = p.dep_min + off
    cur_total = p.n
    cur_ok = int((~_in_curfew(new_min, cfg.schedule.curfew_hours)).sum())

    # Capacity (soft) per (airport, window bin).
    counts: dict[tuple, int] = defaultdict(int)
    for i in range(p.n):
        counts[(p.origin[i], (new_min[i]) // window)] += 1
    cap_total = len(counts)
    cap_ok = sum(1 for c in counts.values() if c <= cap)
    cap_excess = sum(max(0, c - cap) for c in counts.values())

    total = turn_total + cur_total + cap_total
    ok = turn_ok + cur_ok + cap_ok
    return {
        "satisfaction_rate": ok / total if total else 1.0,
        "turnaround_rate": turn_ok / turn_total if turn_total else 1.0,
        "curfew_rate": cur_ok / cur_total if cur_total else 1.0,
        "capacity_rate": cap_ok / cap_total if cap_total else 1.0,
        "capacity_windows": cap_total,
        "capacity_violations": cap_total - cap_ok,
        "capacity_excess_departures": int(cap_excess),
    }


def delay_report(p: ScheduleProblem, off: np.ndarray) -> dict:
    """Delay statistics for an offset vector."""
    idx = np.array([p.offsets.index(int(o)) for o in off])
    chosen = p.D[np.arange(p.n), idx]
    base = p.D[:, p.zero_idx]
    hr = p.high_risk
    return {
        "total_delay": float(chosen.sum()),
        "weighted_total_delay": float((p.weight * chosen).sum()),
        "mean_delay": float(chosen.mean()),
        "high_risk_mean_delay": float(chosen[hr].mean()) if hr.any() else 0.0,
        "high_risk_mean_delay_before": float(base[hr].mean()) if hr.any() else 0.0,
        "high_risk_delay_reduction": float((base[hr] - chosen[hr]).mean()) if hr.any() else 0.0,
        "n_high_risk": int(hr.sum()),
    }


def _evaluate(p: ScheduleProblem, off: np.ndarray, cfg: Config) -> dict:
    return {**delay_report(p, off), **constraint_report(p, off, cfg)}


def _trace_entry(step: int, ev: dict, cfg: Config) -> dict:
    """One convergence point.

    ``objective`` is what the solver actually minimizes (weighted delay *plus*
    the capacity penalties). Plotting the delay component alone is misleading:
    it can rise while the solver trades a little delay for far fewer breached
    capacity windows.
    """
    objective = (
        ev["weighted_total_delay"]
        + cfg.schedule.capacity_penalty * ev["capacity_excess_departures"]
        + cfg.schedule.capacity_window_penalty * ev["capacity_violations"]
    )
    return {
        "step": step,
        "objective": float(objective),
        "weighted_total_delay": ev["weighted_total_delay"],
        "high_risk_mean_delay": ev["high_risk_mean_delay"],
        "satisfaction_rate": ev["satisfaction_rate"],
    }


# ---------------------------------------------------------------------------
# CP-SAT solver.
# ---------------------------------------------------------------------------
class _TraceCallback(cp_model.CpSolverSolutionCallback):
    """Record the improving-solution trajectory (objective) during search."""

    def __init__(self, off_vars: dict[int, cp_model.IntVar]):
        super().__init__()
        self._off_vars = off_vars
        self.snapshots: list[dict[int, int]] = []
        self.objectives: list[float] = []

    def on_solution_callback(self) -> None:
        self.objectives.append(self.ObjectiveValue() / _SCALE)
        if len(self.snapshots) < 60:
            self.snapshots.append({f: int(self.Value(v)) for f, v in self._off_vars.items()})


def optimize_cpsat(p: ScheduleProblem, cfg: Config) -> tuple[np.ndarray, list[dict], dict]:
    window = cfg.schedule.window_minutes
    cap = cfg.schedule.runway_capacity
    model = cp_model.CpModel()

    x: dict[tuple[int, int], cp_model.IntVar] = {}
    off_vars: dict[int, cp_model.IntVar] = {}
    for i in range(p.n):
        if not p.optimized[i]:
            continue
        feas = [o for j, o in enumerate(p.offsets) if p.feasible[i, j]]
        bvars = [model.NewBoolVar(f"x_{i}_{o}") for o in feas]
        model.AddExactlyOne(bvars)
        for o, bv in zip(feas, bvars):
            x[(i, o)] = bv
        ov = model.NewIntVar(min(feas), max(feas), f"off_{i}")
        model.Add(ov == sum(o * bv for o, bv in zip(feas, bvars)))
        off_vars[i] = ov

    def off_expr(i: int):
        return off_vars[i] if i in off_vars else 0

    # Turnaround (hard).
    for a, b, eff_turn in p.tail_pairs:
        rhs = int(p.arr_min[a] + eff_turn - p.dep_min[b])
        model.Add(off_expr(b) - off_expr(a) >= rhs)

    # Capacity (soft).
    bins: dict[tuple, list] = defaultdict(list)
    fixed_count: dict[tuple, int] = defaultdict(int)
    for i in range(p.n):
        if p.optimized[i]:
            for j, o in enumerate(p.offsets):
                if p.feasible[i, j]:
                    bins[(p.origin[i], (p.dep_min[i] + o) // window)].append(x[(i, o)])
        else:
            fixed_count[(p.origin[i], p.dep_min[i] // window)] += 1

    excess_terms = []
    over_terms = []
    all_bins = sorted(set(bins) | set(fixed_count))  # sorted => stable model build
    for b in all_bins:
        members = bins.get(b, [])
        base = fixed_count.get(b, 0)
        if not members:
            continue  # purely fixed bin: constant, not optimizable
        exc = model.NewIntVar(0, len(members) + base, f"exc_{b[0]}_{b[1]}")
        model.Add(exc >= sum(members) + base - cap)
        excess_terms.append(exc)
        # over <=> the window breaches capacity at all. Penalising this as well
        # as the excess makes the reported violation count part of the objective
        # instead of an arbitrary choice between equally-optimal solutions.
        over = model.NewBoolVar(f"over_{b[0]}_{b[1]}")
        # One-sided big-M indicator: exc >= 1 forces over = 1, while the
        # positive penalty in the objective drives over back to 0 when exc = 0.
        # Much cheaper for the solver than a fully reified pair of constraints.
        model.Add(exc <= (len(members) + base) * over)
        over_terms.append(over)

    # Objective.
    obj = []
    for i in range(p.n):
        if not p.optimized[i]:
            continue
        for j, o in enumerate(p.offsets):
            if p.feasible[i, j]:
                coef = int(round(p.weight[i] * p.D[i, j] * _SCALE))
                obj.append(coef * x[(i, o)])
    pen = int(round(cfg.schedule.capacity_penalty * _SCALE))
    obj.extend(pen * e for e in excess_terms)
    wpen = int(round(cfg.schedule.capacity_window_penalty * _SCALE))
    obj.extend(wpen * o for o in over_terms)
    model.Minimize(sum(obj))

    solver = cp_model.CpSolver()
    solver.parameters.max_time_in_seconds = float(cfg.schedule.solver_time_limit_s)
    # A single worker keeps the search deterministic (the parallel portfolio is
    # not reproducible even with a fixed seed).
    solver.parameters.num_search_workers = int(cfg.schedule.solver_workers)
    solver.parameters.random_seed = cfg.seed
    cb = _TraceCallback(off_vars)
    status = solver.Solve(model, cb)

    off = np.zeros(p.n, dtype=int)
    if status in (cp_model.OPTIMAL, cp_model.FEASIBLE):
        for i, v in off_vars.items():
            off[i] = int(solver.Value(v))
    else:
        # Falling back to the original schedule would otherwise look like a
        # legitimate "no change" result, so make the failure loud.
        print(
            f"[schedule] WARNING: CP-SAT returned {solver.StatusName(status)} for "
            f"{p.n} flights within {cfg.schedule.solver_time_limit_s}s; keeping the "
            f"original schedule. Reduce the problem size or raise the time limit."
        )

    trace = _snapshots_to_trace(p, cb.snapshots, cfg)
    meta = {
        "status": solver.StatusName(status),
        "solve_time_s": float(solver.WallTime()),
        "objective": float(solver.ObjectiveValue() / _SCALE) if status else float("nan"),
        "num_solutions": len(cb.objectives),
    }
    return off, trace, meta


def _snapshots_to_trace(p: ScheduleProblem, snapshots, cfg: Config) -> list[dict]:
    trace = []
    for k, snap in enumerate(snapshots):
        off = np.zeros(p.n, dtype=int)
        for i, o in snap.items():
            off[i] = o
        trace.append(_trace_entry(k, _evaluate(p, off, cfg), cfg))
    return trace


# ---------------------------------------------------------------------------
# Greedy baseline (restores the original route, then local search).
# ---------------------------------------------------------------------------
def optimize_greedy(p: ScheduleProblem, cfg: Config) -> tuple[np.ndarray, list[dict], dict]:
    window = cfg.schedule.window_minutes
    cap = cfg.schedule.runway_capacity
    pen = cfg.schedule.capacity_penalty
    off = np.zeros(p.n, dtype=int)

    # Live per-bin counts (all flights, starting at offset 0).
    counts: dict[tuple, int] = defaultdict(int)
    for i in range(p.n):
        counts[(p.origin[i], p.dep_min[i] // window)] += 1

    def bin_excess(cnt: int) -> int:
        return max(0, cnt - cap)

    # Optimize highest-risk / highest-delay flights first.
    order = sorted(
        [i for i in range(p.n) if p.optimized[i]],
        key=lambda i: (p.high_risk[i], p.D[i, p.zero_idx]),
        reverse=True,
    )

    prev_of = {b: (a, t) for a, b, t in p.tail_pairs}
    next_of = {a: (b, t) for a, b, t in p.tail_pairs}

    trace = []
    for _pass in range(cfg.schedule.greedy_max_passes):
        improved = False
        for i in order:
            cur = off[i]
            cur_bin = (p.origin[i], (p.dep_min[i] + cur) // window)
            best_o, best_cost = cur, None
            for j, o in enumerate(p.offsets):
                if not p.feasible[i, j]:
                    continue
                # Turnaround feasibility against current neighbor offsets.
                ok = True
                if i in prev_of:
                    a, eff_prev = prev_of[i]
                    if (p.dep_min[i] + o) < (p.arr_min[a] + off[a]) + eff_prev:
                        ok = False
                if ok and i in next_of:
                    b, eff_next = next_of[i]
                    if (p.dep_min[b] + off[b]) < (p.arr_min[i] + o) + eff_next:
                        ok = False
                if not ok:
                    continue
                new_bin = (p.origin[i], (p.dep_min[i] + o) // window)
                # Marginal capacity change if i moves cur_bin -> new_bin.
                if new_bin == cur_bin:
                    dcap = 0
                else:
                    before = bin_excess(counts[cur_bin]) + bin_excess(counts[new_bin])
                    after = bin_excess(counts[cur_bin] - 1) + bin_excess(counts[new_bin] + 1)
                    dcap = after - before
                cost = p.weight[i] * p.D[i, j] + pen * dcap
                if best_cost is None or cost < best_cost - _EPS:
                    best_cost, best_o = cost, o
            if best_o != cur:
                counts[cur_bin] -= 1
                counts[(p.origin[i], (p.dep_min[i] + best_o) // window)] += 1
                off[i] = best_o
                improved = True
        trace.append(_trace_entry(_pass, _evaluate(p, off, cfg), cfg))
        if not improved:
            break

    return off, trace, {"passes": len(trace)}


# ---------------------------------------------------------------------------
# Orchestration + persistence.
# ---------------------------------------------------------------------------
def optimize(
    cfg: Config,
    flights: pd.DataFrame,
    graded: pd.DataFrame,
    bundle: ModelBundle,
    solver: str = "cpsat",
) -> tuple[pd.DataFrame, list[dict], dict]:
    """Run the requested solver and return (schedule_df, trace, metrics)."""
    p = build_problem(cfg, flights, graded, bundle)
    if solver == "greedy":
        off, trace, solver_meta = optimize_greedy(p, cfg)
    else:
        off, trace, solver_meta = optimize_cpsat(p, cfg)

    base_off = np.zeros(p.n, dtype=int)
    metrics = {
        "solver": solver,
        "solver_meta": solver_meta,
        "before": _evaluate(p, base_off, cfg),
        "after": _evaluate(p, off, cfg),
    }
    idx = np.array([p.offsets.index(int(o)) for o in off])
    day0 = pd.to_datetime(flights["sched_dep"]).dt.normalize().min()
    schedule_df = pd.DataFrame(
        {
            "flight_id": p.flight_id,
            "tail_id": p.tail_id,
            "leg_index": p.leg_index,
            "origin": p.origin,
            "risk_level": p.risk_level,
            "high_risk": p.high_risk,
            "sched_dep": pd.to_datetime(flights["sched_dep"]).to_numpy(),
            "offset_min": off,
            "new_dep": day0 + pd.to_timedelta(p.dep_min + off, unit="m"),
            "pred_delay_before": p.D[:, p.zero_idx],
            "pred_delay_after": p.D[np.arange(p.n), idx],
        }
    )
    return schedule_df, trace, metrics


def select_operating_day(flights: pd.DataFrame, cfg: Config):
    """Restrict the schedule to a single operating day (config or busiest)."""
    days = pd.to_datetime(flights["sched_dep"]).dt.date
    if cfg.schedule.day:
        target = pd.Timestamp(cfg.schedule.day).date()
    else:
        target = days.value_counts().idxmax()
    subset = flights[days == target].reset_index(drop=True)
    return subset, target


def run_scheduling(cfg: Config, solver: str = "cpsat") -> dict:
    """CLI entry: load inputs, run the chosen solver + the greedy baseline for
    comparison, persist the schedule, and return the metrics."""
    from flightopt.data import loader

    cfg.paths.ensure()
    flights = loader.load_flights(cfg)
    graded = pd.read_parquet(cfg.paths.graded_parquet)
    bundle = ModelBundle.load(cfg)

    # Re-timing is a single-day problem: capacity windows, curfew and turnaround
    # all live inside one operating day, and solving a whole quarter at once is
    # both meaningless operationally and intractable for CP-SAT.
    flights, day = select_operating_day(flights, cfg)
    graded = graded[graded["flight_id"].isin(set(flights["flight_id"]))]

    schedule_df, trace, metrics = optimize(cfg, flights, graded, bundle, solver=solver)
    schedule_df.to_parquet(cfg.paths.schedule_parquet, index=False)

    # Always compute the greedy baseline for the head-to-head comparison.
    comparison = {solver: metrics}
    if solver != "greedy":
        _, greedy_trace, greedy_metrics = optimize(cfg, flights, graded, bundle, solver="greedy")
        comparison["greedy"] = greedy_metrics
        trace = {"cpsat": trace, "greedy": greedy_trace}
    else:
        trace = {"greedy": trace}

    result = {
        "primary_solver": solver,
        "operating_day": str(day),
        "n_flights": int(len(flights)),
        "comparison": comparison,
        "trace": trace,
    }
    import json

    with open(cfg.paths.reports / "schedule_metrics.json", "w", encoding="utf-8") as fh:
        json.dump(result, fh, indent=2, default=float)
    pd.to_pickle(trace, cfg.paths.data_processed / "schedule_trace.pkl")
    return result
