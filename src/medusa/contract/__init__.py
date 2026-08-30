"""The twin contract: the interface a generated `twin.py` must satisfy, and its checker."""

from medusa.contract.checker import (
    CheckResult,
    check_twin_object,
    check_twin_source,
)
from medusa.contract.interface import (
    PARAM_TRIPLE_LEN,
    POPULATION_TASK,
    SPATIAL_TASK,
    STRUCTURED_TASK,
    Observations,
    Task,
)

__all__ = [
    "Observations",
    "Task",
    "POPULATION_TASK",
    "STRUCTURED_TASK",
    "SPATIAL_TASK",
    "PARAM_TRIPLE_LEN",
    "CheckResult",
    "check_twin_source",
    "check_twin_object",
]
