from __future__ import annotations
from dataclasses import dataclass, field
from typing import Any, Protocol
from .context import RunContext
from .trajectory import ToolCall


@dataclass
class PolicyDecision:
    actions: list[ToolCall]
    done: bool = False
    metadata: dict[str, Any] = field(default_factory=dict)


class Policy(Protocol):
    async def decide(self, observation: Any, ctx: RunContext) -> PolicyDecision: ...
