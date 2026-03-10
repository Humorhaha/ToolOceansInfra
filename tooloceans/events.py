from __future__ import annotations
from dataclasses import dataclass, field
from enum import Enum
from typing import Any
import time
import uuid


class EventType(str, Enum):
    EPISODE_STARTED = "episode.started"
    EPISODE_ENDED   = "episode.ended"
    STEP_STARTED    = "step.started"
    STEP_ENDED      = "step.ended"
    TOOL_REQUESTED  = "tool.requested"
    TOOL_COMPLETED  = "tool.completed"
    TOOL_FAILED     = "tool.failed"
    OBSERVATION     = "observation.emitted"
    REWARD          = "reward.attached"


@dataclass
class Event:
    type: EventType
    run_id: str
    episode_id: str
    event_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    step_id: str | None = None
    timestamp: float = field(default_factory=time.time)
    payload: dict[str, Any] = field(default_factory=dict)
    trace_id: str | None = None
