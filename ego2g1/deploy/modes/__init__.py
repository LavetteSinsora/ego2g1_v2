"""The mode registry (docs/deploy_refactor_plan.md §2): one DeployMode object
per policy family. Importing this package registers the built-in three;
adding a family means one new module here + a `base.register(...)` call."""

from . import joint, relation_eef, relative_eef  # noqa: F401  (registration)
from .base import (  # noqa: F401
    MODES,
    DeployMode,
    ProprioModeBase,
    get,
    register,
    resolve,
    resolve_action_mode,
)
