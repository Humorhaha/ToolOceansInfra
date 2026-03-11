from __future__ import annotations
import asyncio
from dataclasses import dataclass, field
from enum import Enum
from typing import Any
from ..dataset import (
    DatasetBuilder,
    EpisodeDatasetSample,
    EpisodeStepSample,
    OnlineDatasetBuilder,
    RLTransition,
    ToolAction,
    TransitionDatasetBuilder,
)
from ..events import Event, EventType
from ..trajectory import Episode


class EpisodeDatasetBuilder(DatasetBuilder[EpisodeDatasetSample]):
    """Compatibility builder that preserves the legacy episode-shaped dataset."""

    def add_episode(self, episode: Episode) -> None:
        steps: list[EpisodeStepSample] = []
        for i, step in enumerate(episode.steps):
            next_obs = episode.steps[i + 1].observation if i + 1 < len(episode.steps) else None
            steps.append({
                "observation": step.observation,
                "actions": self._build_actions(step.tool_calls),
                "reward": step.reward,
                "next_observation": next_obs,
            })
        self._append_item({
            "episode_id": episode.episode_id,
            "steps": steps,
            "terminal_reward": episode.terminal_reward,
        })


class OfflineTransitionDatasetBuilder(TransitionDatasetBuilder):
    """Primary offline RL entrypoint producing canonical RL transitions.

    `Episode` does not persist `trace_id`, so offline transitions always expose
    `trace_id=None`.
    """

    def add_episode(self, episode: Episode) -> None:
        for i, step in enumerate(episode.steps):
            next_obs = episode.steps[i + 1].observation if i + 1 < len(episode.steps) else None
            self._append_item(self._build_transition(
                run_id=episode.run_id,
                episode_id=episode.episode_id,
                step_id=step.step_id,
                trace_id=None,
                observation=step.observation,
                actions=self._build_actions(step.tool_calls),
                reward=step.reward,
                next_observation=next_obs,
                done=i + 1 == len(episode.steps),
            ))


@dataclass
class _PendingStep:
    run_id: str
    episode_id: str
    step_id: str
    trace_id: str | None
    actions: list[ToolAction] = field(default_factory=list)
    observation: Any = None
    reward: float | None = None


class QueueOverflowPolicy(str, Enum):
    RAISE = "raise"
    DROP_NEWEST = "drop_newest"


class _InlineOnlineTransitionDatasetBuilder(OnlineDatasetBuilder):
    """Inline transition materializer used behind the queue-backed online builder."""

    def __init__(self) -> None:
        super().__init__()
        self._open_steps: dict[str, _PendingStep] = {}
        self._completed_steps: dict[str, _PendingStep] = {}

    async def handle(self, event: Event) -> None:
        if event.type == EventType.EPISODE_STARTED:
            self._open_steps.pop(event.episode_id, None)
            self._completed_steps.pop(event.episode_id, None)
            return

        if event.type == EventType.STEP_STARTED:
            if event.step_id is None:
                return
            self._open_steps[event.episode_id] = _PendingStep(
                run_id=event.run_id,
                episode_id=event.episode_id,
                step_id=event.step_id,
                trace_id=event.trace_id,
            )
            return

        if event.type == EventType.TOOL_REQUESTED:
            step = self._open_steps.get(event.episode_id)
            if step is None or step.step_id != event.step_id:
                return
            step.actions.append({
                "tool_name": event.payload["tool_name"],
                "arguments": event.payload["arguments"],
                "call_id": event.payload["call_id"],
            })
            return

        if event.type == EventType.OBSERVATION:
            step = self._open_steps.get(event.episode_id)
            if step is None or step.step_id != event.step_id:
                return

            step.observation = event.payload.get("observation")
            step.reward = event.payload.get("reward")

            previous_step = self._completed_steps.get(event.episode_id)
            if previous_step is not None:
                self._append_item(self._to_transition(
                    previous_step,
                    next_observation=step.observation,
                    done=False,
                ))

            self._completed_steps[event.episode_id] = step
            del self._open_steps[event.episode_id]
            return

        if event.type == EventType.EPISODE_ENDED:
            previous_step = self._completed_steps.pop(event.episode_id, None)
            if previous_step is not None:
                self._append_item(self._to_transition(
                    previous_step,
                    next_observation=None,
                    done=True,
                ))
            self._open_steps.pop(event.episode_id, None)

    def _to_transition(
        self,
        step: _PendingStep,
        next_observation: Any,
        done: bool,
    ) -> RLTransition:
        return self._build_transition(
            run_id=step.run_id,
            episode_id=step.episode_id,
            step_id=step.step_id,
            trace_id=step.trace_id,
            observation=step.observation,
            actions=step.actions,
            reward=step.reward,
            next_observation=next_observation,
            done=done,
        )


class OnlineTransitionDatasetBuilder(OnlineDatasetBuilder):
    """Online RL entrypoint producing canonical RL transitions from events.

    `handle()` enqueues the event and returns after local queue ingestion.
    `build()` returns only transitions already processed by the background worker.
    `drain()` waits until the queue is fully processed.
    `aclose()` drains pending events and stops the current worker task. A later
    `handle()` call will lazily start a new worker.
    """

    def __init__(
        self,
        processor: OnlineDatasetBuilder | None = None,
        max_queue_size: int | None = None,
        overflow_policy: QueueOverflowPolicy | str = QueueOverflowPolicy.RAISE,
    ) -> None:
        super().__init__()
        if max_queue_size is not None and max_queue_size <= 0:
            raise ValueError("max_queue_size must be positive when provided")
        self._processor = processor or _InlineOnlineTransitionDatasetBuilder()
        self._overflow_policy = QueueOverflowPolicy(overflow_policy)
        self._queue: asyncio.Queue[Event | object] = asyncio.Queue(maxsize=max_queue_size or 0)
        self._worker_task: asyncio.Task[None] | None = None
        self._worker_error: BaseException | None = None
        self._dropped_events = 0
        self._stop_sentinel = object()

    async def handle(self, event: Event) -> None:
        self._raise_worker_error()
        self._ensure_worker()
        if self._queue.maxsize > 0 and self._queue.full():
            if self._overflow_policy == QueueOverflowPolicy.RAISE:
                raise RuntimeError(
                    f"OnlineTransitionDatasetBuilder queue is full (max_queue_size={self._queue.maxsize})"
                )
            if self._overflow_policy == QueueOverflowPolicy.DROP_NEWEST:
                self._dropped_events += 1
                return
        self._queue.put_nowait(event)

    def build(self) -> list[RLTransition]:
        self._raise_worker_error()
        return self._processor.build()

    @property
    def dropped_events(self) -> int:
        return self._dropped_events

    @property
    def max_queue_size(self) -> int | None:
        return self._queue.maxsize or None

    @property
    def overflow_policy(self) -> QueueOverflowPolicy:
        return self._overflow_policy

    async def drain(self) -> None:
        if self._worker_task is None:
            return
        await self._queue.join()
        self._raise_worker_error()

    async def aclose(self) -> None:
        if self._worker_task is None:
            return
        await self._queue.join()
        self._queue.put_nowait(self._stop_sentinel)
        await self._worker_task
        self._worker_task = None
        self._raise_worker_error()

    def _ensure_worker(self) -> None:
        if self._worker_task is None:
            self._worker_task = asyncio.create_task(self._run_worker())

    async def _run_worker(self) -> None:
        try:
            while True:
                item = await self._queue.get()
                try:
                    if item is self._stop_sentinel:
                        return
                    await self._processor.handle(item)
                finally:
                    self._queue.task_done()
        except BaseException as exc:
            self._worker_error = exc
            while not self._queue.empty():
                self._queue.get_nowait()
                self._queue.task_done()

    def _raise_worker_error(self) -> None:
        if self._worker_error is None:
            return
        raise RuntimeError("Background online dataset worker failed") from self._worker_error
