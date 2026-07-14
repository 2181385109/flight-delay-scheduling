"""flightopt: flight delay prediction and intelligent scheduling optimization.

A decoupled predict -> grade -> schedule closed loop:

* ``data``     synthetic flight generator (primary source) + public loader.
* ``features`` 7 core + 3 interaction features, leakage-safe.
* ``predict``  LightGBM double-head (regression + classification) vs baselines.
* ``risk``     quantile 5-level risk grading (L4/L5 = high risk).
* ``schedule`` OR-Tools CP-SAT slot optimization vs greedy baseline.
* ``evaluate`` metric aggregation + baseline comparison.
* ``viz``      static figures + Gantt animation.
"""

__version__ = "0.1.0"
