from __future__ import annotations
from dataclasses import dataclass, field
from typing import Any, Protocol, runtime_checkable
from .context import RunContext


@dataclass
class ToolCapability:
    has_side_effects: bool = True
    is_deterministic: bool = False
    is_idempotent: bool = False
    is_pure_read: bool = False
    supports_mock: bool = False
    timeout_hint_seconds: float | None = None
    max_concurrency: int | None = None


@dataclass
class ToolSpec:
    name: str
    version: str
    input_schema: dict[str, Any]
    output_schema: dict[str, Any]
    capability: ToolCapability = field(default_factory=ToolCapability)


@runtime_checkable
class ToolHandler(Protocol):
    async def __call__(self, arguments: dict[str, Any], ctx: RunContext) -> Any: ...


class ToolRegistry(Protocol):
    def register(self, spec: ToolSpec, handler: ToolHandler) -> None: ...
    def get(self, name: str, version: str | None = None) -> tuple[ToolSpec, ToolHandler]: ...
    def list(self) -> list[ToolSpec]: ...
