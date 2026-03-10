from __future__ import annotations
import uuid
from typing import Any
from .context import RunContext
from .trajectory import Episode, Step, ToolCall, ToolResult
from .events import Event, EventType
from .bus import EventBus
from .executor import Executor
from .storage import ColdStore
from .hooks import PostEpisodeHook


class EpisodeManager:
    def __init__(
        self,
        executor: Executor,
        bus: EventBus,
        cold_store: ColdStore,
        hooks: list[PostEpisodeHook] | None = None,
    ) -> None:
        self._executor = executor
        self._bus = bus
        self._cold_store = cold_store
        self._hooks: list[PostEpisodeHook] = hooks or []

    async def run_episode(
        self,
        ctx: RunContext,
        steps_input: list[list[ToolCall]],
        metadata: dict[str, Any] | None = None,
    ) -> Episode:
        episode = Episode(
            episode_id=ctx.episode_id,
            run_id=ctx.run_id,
            metadata=metadata or {},
        )

        await self._bus.emit(Event(
            type=EventType.EPISODE_STARTED,
            run_id=ctx.run_id,
            episode_id=ctx.episode_id,
            trace_id=ctx.trace_id,
        ))

        for calls in steps_input:
            step_id = str(uuid.uuid4())
            step_ctx = ctx.step(step_id)

            await self._bus.emit(Event(
                type=EventType.STEP_STARTED,
                run_id=ctx.run_id,
                episode_id=ctx.episode_id,
                step_id=step_id,
                trace_id=ctx.trace_id,
            ))

            results: list[ToolResult] = []
            for call in calls:
                result = await self._executor.execute(call, step_ctx)
                results.append(result)

            # observation: list of outputs (or errors) in order
            observation = [
                r.output if r.error is None else {"error": r.error.code}
                for r in results
            ]

            await self._bus.emit(Event(
                type=EventType.OBSERVATION,
                run_id=ctx.run_id,
                episode_id=ctx.episode_id,
                step_id=step_id,
                trace_id=ctx.trace_id,
                payload={"observation": observation},
            ))

            step = Step(
                step_id=step_id,
                tool_calls=calls,
                tool_results=results,
                observation=observation,
            )
            episode.steps.append(step)

            await self._bus.emit(Event(
                type=EventType.STEP_ENDED,
                run_id=ctx.run_id,
                episode_id=ctx.episode_id,
                step_id=step_id,
                trace_id=ctx.trace_id,
            ))

        await self._bus.emit(Event(
            type=EventType.EPISODE_ENDED,
            run_id=ctx.run_id,
            episode_id=ctx.episode_id,
            trace_id=ctx.trace_id,
        ))

        for hook in self._hooks:
            episode = await hook.on_episode_end(episode)

        await self._cold_store.save_episode(episode)
        return episode
