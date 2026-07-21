"""Cross-platform path management built entirely on :class:`pathlib.Path`.

The repository root is auto-detected (the first parent that contains
``pyproject.toml``) but can be overridden through ``config.yaml``'s
``paths.root``.  Every other directory is resolved relative to the root, so the
project behaves identically on Windows and POSIX systems.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path


def find_repo_root(start: Path | None = None) -> Path:
    """Return the nearest ancestor directory that contains ``pyproject.toml``.

    Falls back to two levels above this file (``src/flightopt`` -> repo root)
    when no marker is found (e.g. when installed as a wheel).
    """
    here = (start or Path(__file__)).resolve()
    for parent in [here, *here.parents]:
        if (parent / "pyproject.toml").is_file():
            return parent
    return Path(__file__).resolve().parents[2]


@dataclass(frozen=True)
class ProjectPaths:
    """Resolved absolute paths for every project directory."""

    root: Path
    data_raw: Path
    data_processed: Path
    outputs: Path
    figures: Path
    models: Path
    reports: Path

    @property
    def flights_parquet(self) -> Path:
        return self.data_processed / "flights.parquet"

    @property
    def predictions_parquet(self) -> Path:
        return self.data_processed / "predictions.parquet"

    @property
    def test_predictions_parquet(self) -> Path:
        """Final-model predictions on the held-out test split (matches report.md)."""
        return self.data_processed / "test_predictions.parquet"

    @property
    def graded_parquet(self) -> Path:
        return self.data_processed / "graded.parquet"

    @property
    def schedule_parquet(self) -> Path:
        return self.data_processed / "schedule.parquet"

    @property
    def metrics_json(self) -> Path:
        return self.reports / "metrics.json"

    @property
    def report_md(self) -> Path:
        return self.reports / "report.md"

    @property
    def feature_dict_md(self) -> Path:
        return self.reports / "feature_dict.md"

    def ensure(self) -> "ProjectPaths":
        """Create every managed directory if it does not yet exist."""
        for d in (
            self.data_raw,
            self.data_processed,
            self.outputs,
            self.figures,
            self.models,
            self.reports,
        ):
            d.mkdir(parents=True, exist_ok=True)
        return self


def build_paths(path_cfg: dict, root_override: str | Path | None = None) -> ProjectPaths:
    """Construct :class:`ProjectPaths` from the ``paths`` config section."""
    root_value = root_override if root_override is not None else path_cfg.get("root")
    root = Path(root_value).resolve() if root_value else find_repo_root()

    def _resolve(key: str, default: str) -> Path:
        rel = Path(path_cfg.get(key, default))
        return rel if rel.is_absolute() else (root / rel)

    return ProjectPaths(
        root=root,
        data_raw=_resolve("data_raw", "data/raw"),
        data_processed=_resolve("data_processed", "data/processed"),
        outputs=_resolve("outputs", "outputs"),
        figures=_resolve("figures", "outputs/figures"),
        models=_resolve("models", "outputs/models"),
        reports=_resolve("reports", "outputs/reports"),
    )
