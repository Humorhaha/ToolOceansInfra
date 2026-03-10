from __future__ import annotations
from dataclasses import dataclass, field
from typing import Any


@dataclass
class ToolCall:
    tool_name: str
    arguments: dict[str, Any]
    call_id: str


@dataclass
class ToolError:
    code: str
    message: str
    detail: dict[str, Any] = field(default_factory=dict)
    retryable: bool = False


@dataclass
class ToolResult:
    call_id: str
    output: Any = None
    error: ToolError | None = None
    duration_ms: float | None = None


@dataclass
class Step:
    step_id: str
    tool_calls: list[ToolCall] = field(default_factory=list)
    tool_results: list[ToolResult] = field(default_factory=list)
    observation: Any = None
    reward: float | None = None
    # action: Action | None = None  # reserved


@dataclass
class Episode:
    episode_id: str
    run_id: str
    steps: list[Step] = field(default_factory=list)
    metadata: dict[str, Any] = field(default_factory=dict)
    terminal_reward: float | None = None

    def as_rl_trajectory(self) -> list[dict]:
        return [
            {"action": s.tool_calls, "observation": s.observation, "reward": s.reward}
            for s in self.steps
        ]
