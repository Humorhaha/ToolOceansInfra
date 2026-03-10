from __future__ import annotations
from dataclasses import dataclass, field, replace
from typing import Any
import uuid


@dataclass
class RunContext:
    run_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    episode_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    step_id: str | None = None
    trace_id: str | None = None
    timeout_seconds: float | None = None
    experiment_metadata: dict[str, Any] = field(default_factory=dict)

    def step(self, step_id: str) -> RunContext:
        return replace(self, step_id=step_id)
