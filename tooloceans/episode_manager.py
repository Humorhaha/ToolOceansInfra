from __future__ import annotations
import uuid
import asyncio
from typing import Any
from .context import RunContext
from .trajectory import Episode, Step, ToolCall, ToolResult
from .events import Event, EventType
from .bus import EventBus
from .executor import Executor
from .storage import ColdStore
from .hooks import PostEpisodeHook, StepRewardFn
from .agent import Policy


class EpisodeManager:
    def __init__(
        self,
        executor: Executor,
        bus: EventBus,
        cold_store: ColdStore,
        hooks: list[PostEpisodeHook] | None = None,
        step_reward_fn: StepRewardFn | None = None,
    ) -> None:
        self._executor = executor
        self._bus = bus
        self._cold_store = cold_store
        self._hooks: list[PostEpisodeHook] = hooks or []
        self._step_reward_fn = step_reward_fn

    async def _score_and_emit_reward(
        self, step: Step, ctx: RunContext
    ) -> None:
        if self._step_reward_fn is None:
            return
        step.reward = await self._step_reward_fn.score(step, ctx)
        await self._bus.emit(Event(
            type=EventType.REWARD,
            run_id=ctx.run_id,
            episode_id=ctx.episode_id,
            step_id=ctx.step_id,
            trace_id=ctx.trace_id,
            payload={"reward": step.reward},
        ))

    # Plan 模式 / offline RL
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
            results = await asyncio.gather(
                *(self._executor.execute(call, step_ctx) for call in calls)
            )

            observation = [
                r.output if r.error is None else {"error": r.error.code}
                for r in results
            ]

            step = Step(
                step_id=step_id,
                tool_calls=calls,
                tool_results=results,
                observation=observation,
            )
            await self._score_and_emit_reward(step, step_ctx)

            await self._bus.emit(Event(
                type=EventType.OBSERVATION,
                run_id=ctx.run_id,
                episode_id=ctx.episode_id,
                step_id=step_id,
                trace_id=ctx.trace_id,
                payload={"observation": observation, "reward": step.reward},
            ))

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

    #ReAct模式 / online RL
    async def run_episode_dynamic(
        self,
        ctx: RunContext,
        policy: Policy,
        max_steps: int,
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

        observation: Any = None
        for _ in range(max_steps):
            decision = await policy.decide(observation, ctx)
            if decision.done or not decision.actions:
                break

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

            results = await asyncio.gather(
                *(self._executor.execute(call, step_ctx) for call in decision.actions)
            )

            observation = [
                r.output if r.error is None else {"error": r.error.code}
                for r in results
            ]

            step = Step(
                step_id=step_id,
                tool_calls=decision.actions,
                tool_results=results,
                observation=observation,
            )
            await self._score_and_emit_reward(step, step_ctx)

            await self._bus.emit(Event(
                type=EventType.OBSERVATION,
                run_id=ctx.run_id,
                episode_id=ctx.episode_id,
                step_id=step_id,
                trace_id=ctx.trace_id,
                payload={"observation": observation, "reward": step.reward},
            ))

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
