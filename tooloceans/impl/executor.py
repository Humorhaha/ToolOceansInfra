from __future__ import annotations
import asyncio
import time
import uuid
from ..context import RunContext
from ..trajectory import ToolCall, ToolResult, ToolError
from ..registry import ToolRegistry
from ..bus import EventBus
from ..events import Event, EventType


class AsyncExecutor:
    def __init__(self, registry: ToolRegistry, bus: EventBus) -> None:
        self._registry = registry
        self._bus = bus

    async def execute(self, call: ToolCall, ctx: RunContext) -> ToolResult:
        await self._bus.emit(Event(
            type=EventType.TOOL_REQUESTED,
            run_id=ctx.run_id,
            episode_id=ctx.episode_id,
            step_id=ctx.step_id,
            trace_id=ctx.trace_id,
            payload={"tool_name": call.tool_name, "call_id": call.call_id, "arguments": call.arguments},
        ))

        spec, handler = self._registry.get(call.tool_name)
        timeout = ctx.timeout_seconds or spec.capability.timeout_hint_seconds

        t0 = time.monotonic()
        try:
            if timeout:
                output = await asyncio.wait_for(
                    handler(call.arguments, ctx), timeout=timeout
                )
            else:
                output = await handler(call.arguments, ctx)
            duration_ms = (time.monotonic() - t0) * 1000
            result = ToolResult(call_id=call.call_id, output=output, duration_ms=duration_ms)
            await self._bus.emit(Event(
                type=EventType.TOOL_COMPLETED,
                run_id=ctx.run_id,
                episode_id=ctx.episode_id,
                step_id=ctx.step_id,
                trace_id=ctx.trace_id,
                payload={"call_id": call.call_id, "duration_ms": duration_ms, "output": output},
            ))
        except asyncio.TimeoutError:
            duration_ms = (time.monotonic() - t0) * 1000
            result = ToolResult(
                call_id=call.call_id,
                error=ToolError(code="timeout", message=f"Exceeded {timeout}s", retryable=True),
                duration_ms=duration_ms,
            )
            await self._bus.emit(Event(
                type=EventType.TOOL_FAILED,
                run_id=ctx.run_id,
                episode_id=ctx.episode_id,
                step_id=ctx.step_id,
                trace_id=ctx.trace_id,
                payload={"call_id": call.call_id, "error_code": "timeout"},
            ))
        except Exception as exc:
            duration_ms = (time.monotonic() - t0) * 1000
            result = ToolResult(
                call_id=call.call_id,
                error=ToolError(code="execution_error", message=str(exc), retryable=False),
                duration_ms=duration_ms,
            )
            await self._bus.emit(Event(
                type=EventType.TOOL_FAILED,
                run_id=ctx.run_id,
                episode_id=ctx.episode_id,
                step_id=ctx.step_id,
                trace_id=ctx.trace_id,
                payload={"call_id": call.call_id, "error_code": "execution_error"},
            ))
        return result
