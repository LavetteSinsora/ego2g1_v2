"""The dashboard-facing contracts (docs/deploy_refactor_plan.md §5):
telemetry.py owns the page's data shape (one declared dataclass instead of
three hand-copied dicts), overlay.py owns the perception overlay drawing
(one renderer fed the RECORDED `percept` shape, so live and replay cannot
diverge)."""

from .telemetry import TelemetrySnapshot, executor_row_groups, relation_panel  # noqa: F401
