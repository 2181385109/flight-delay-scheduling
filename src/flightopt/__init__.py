"""flightopt: flight delay prediction and intelligent scheduling optimization.

A decoupled predict -> grade -> schedule closed loop:

* ``data``     real flight-data loading (nycflights13 / generic CSV adapter).
* ``features`` planning + network-state + target-encoded features, leakage-safe.
* ``predict``  LightGBM double-head (regression + classification) vs baselines.
* ``risk``     quantile 5-level risk grading (L4/L5 = high risk).
* ``schedule`` OR-Tools CP-SAT slot optimization vs greedy baseline.
* ``evaluate`` metric aggregation + baseline comparison.
* ``viz``      static figures + Gantt animation.
"""

__version__ = "0.1.0"
