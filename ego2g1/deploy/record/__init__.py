"""The recording subsystem: the event-stream schema (schema.py) that both the
writer (recorder.py, currently one level up pending the layout move) and every
replay reader share. See docs/deploy_refactor_plan.md §4."""

from .schema import (  # noqa: F401
    EVENT_KINDS,
    SCHEMA_VERSION,
    STRATEGY_PARAM_KEYS,
    EventSpec,
    build_meta,
)
