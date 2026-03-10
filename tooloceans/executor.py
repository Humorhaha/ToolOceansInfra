from __future__ import annotations
from typing import Protocol
from .context import RunContext
from .trajectory import ToolCall, ToolResult


class Executor(Protocol):
    async def execute(self, call: ToolCall, ctx: RunContext) -> ToolResult: ...
