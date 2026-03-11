import asyncio
import uuid

import pytest

from tooloceans.context import RunContext
from tooloceans.dataset import OnlineDatasetBuilder
from tooloceans.episode_manager import EpisodeManager
from tooloceans.events import Event, EventType
from tooloceans.impl.bus import InProcessEventBus
from tooloceans.impl.cold_store import LocalFileColdStore
from tooloceans.impl.dataset import OnlineTransitionDatasetBuilder, QueueOverflowPolicy
from tooloceans.impl.executor import AsyncExecutor
from tooloceans.impl.registry import InMemoryToolRegistry
from tooloceans.registry import ToolSpec
from tooloceans.trajectory import Step, ToolCall


def _make_call(name: str, arguments: dict) -> ToolCall:
    return ToolCall(tool_name=name, arguments=arguments, call_id=str(uuid.uuid4()))


class IndexedRewardFn:
    async def score(self, step: Step, ctx) -> float:
        return float(step.tool_calls[0].arguments["value"])


class SlowProcessor(OnlineDatasetBuilder):
    def __init__(self) -> None:
        super().__init__()
        self.started = asyncio.Event()
        self.release = asyncio.Event()

    async def handle(self, event: Event) -> None:
        self.started.set()
        await self.release.wait()
        self._append_item({"event_type": event.type.value})


@pytest.mark.asyncio
async def test_online_builder_emits_transitions_from_episode_events(tmp_path):
    registry = InMemoryToolRegistry()

    async def echo_handler(args, ctx):
        return {"echo": args["value"]}

    registry.register(
        ToolSpec(name="echo", version="1", input_schema={}, output_schema={}),
        echo_handler,
    )

    bus = InProcessEventBus()
    online = OnlineTransitionDatasetBuilder()
    await bus.subscribe(online)

    executor = AsyncExecutor(registry, bus)
    manager = EpisodeManager(
        executor,
        bus,
        LocalFileColdStore(tmp_path),
        step_reward_fn=IndexedRewardFn(),
    )

    ctx = RunContext(trace_id="trace-online")
    calls = [
        [_make_call("echo", {"value": 1})],
        [_make_call("echo", {"value": 2})],
    ]

    try:
        await manager.run_episode(ctx, calls)
        await online.drain()

        transitions = online.build()
        assert len(transitions) == 2

        assert transitions[0]["episode_id"] == ctx.episode_id
        assert transitions[0]["trace_id"] == "trace-online"
        assert transitions[0]["observation"] == [{"echo": 1}]
        assert transitions[0]["actions"][0]["tool_name"] == "echo"
        assert transitions[0]["reward"] == 1.0
        assert transitions[0]["next_observation"] == [{"echo": 2}]
        assert transitions[0]["done"] is False

        assert transitions[1]["observation"] == [{"echo": 2}]
        assert transitions[1]["reward"] == 2.0
        assert transitions[1]["next_observation"] is None
        assert transitions[1]["done"] is True
    finally:
        await online.aclose()


@pytest.mark.asyncio
async def test_online_builder_handles_single_step_episode(tmp_path):
    registry = InMemoryToolRegistry()

    async def echo_handler(args, ctx):
        return {"echo": args["value"]}

    registry.register(
        ToolSpec(name="echo", version="1", input_schema={}, output_schema={}),
        echo_handler,
    )

    bus = InProcessEventBus()
    online = OnlineTransitionDatasetBuilder()
    await bus.subscribe(online)

    executor = AsyncExecutor(registry, bus)
    manager = EpisodeManager(executor, bus, LocalFileColdStore(tmp_path))

    ctx = RunContext()
    try:
        await manager.run_episode(ctx, [[_make_call("echo", {"value": 7})]])
        await online.drain()

        transitions = online.build()
        assert len(transitions) == 1
        assert transitions[0]["observation"] == [{"echo": 7}]
        assert transitions[0]["actions"][0]["arguments"] == {"value": 7}
        assert transitions[0]["reward"] is None
        assert transitions[0]["next_observation"] is None
        assert transitions[0]["done"] is True
    finally:
        await online.aclose()


@pytest.mark.asyncio
async def test_online_builder_queues_events_without_blocking_bus():
    bus = InProcessEventBus()
    processor = SlowProcessor()
    online = OnlineTransitionDatasetBuilder(processor=processor)
    await bus.subscribe(online)

    event = Event(
        type=EventType.EPISODE_STARTED,
        run_id="run-1",
        episode_id="episode-1",
    )

    try:
        await asyncio.wait_for(bus.emit(event), timeout=0.05)
        await asyncio.wait_for(processor.started.wait(), timeout=0.05)

        processor.release.set()
        await online.drain()

        items = online.build()
        assert items == [{"event_type": EventType.EPISODE_STARTED.value}]
    finally:
        processor.release.set()
        await online.aclose()


@pytest.mark.asyncio
async def test_online_builder_build_returns_processed_snapshot_only():
    bus = InProcessEventBus()
    processor = SlowProcessor()
    online = OnlineTransitionDatasetBuilder(processor=processor)
    await bus.subscribe(online)

    try:
        await bus.emit(Event(type=EventType.EPISODE_STARTED, run_id="run-1", episode_id="episode-1"))
        await asyncio.wait_for(processor.started.wait(), timeout=0.05)

        assert online.build() == []

        processor.release.set()
        await online.drain()

        assert online.build() == [{"event_type": EventType.EPISODE_STARTED.value}]
    finally:
        processor.release.set()
        await online.aclose()


@pytest.mark.asyncio
async def test_online_builder_raises_when_bounded_queue_is_full():
    bus = InProcessEventBus()
    processor = SlowProcessor()
    online = OnlineTransitionDatasetBuilder(
        processor=processor,
        max_queue_size=1,
        overflow_policy=QueueOverflowPolicy.RAISE,
    )
    await bus.subscribe(online)

    try:
        await bus.emit(Event(type=EventType.EPISODE_STARTED, run_id="run-1", episode_id="episode-1"))
        await asyncio.wait_for(processor.started.wait(), timeout=0.05)

        await bus.emit(Event(type=EventType.STEP_STARTED, run_id="run-1", episode_id="episode-1", step_id="step-1"))

        with pytest.raises(RuntimeError, match="queue is full"):
            await bus.emit(Event(type=EventType.STEP_ENDED, run_id="run-1", episode_id="episode-1", step_id="step-1"))
    finally:
        processor.release.set()
        await online.aclose()


@pytest.mark.asyncio
async def test_online_builder_drops_newest_when_configured_to_drop():
    bus = InProcessEventBus()
    processor = SlowProcessor()
    online = OnlineTransitionDatasetBuilder(
        processor=processor,
        max_queue_size=1,
        overflow_policy=QueueOverflowPolicy.DROP_NEWEST,
    )
    await bus.subscribe(online)

    try:
        await bus.emit(Event(type=EventType.EPISODE_STARTED, run_id="run-1", episode_id="episode-1"))
        await asyncio.wait_for(processor.started.wait(), timeout=0.05)

        await bus.emit(Event(type=EventType.STEP_STARTED, run_id="run-1", episode_id="episode-1", step_id="step-1"))
        await bus.emit(Event(type=EventType.STEP_ENDED, run_id="run-1", episode_id="episode-1", step_id="step-1"))

        processor.release.set()
        await online.drain()

        items = online.build()
        assert items == [
            {"event_type": EventType.EPISODE_STARTED.value},
            {"event_type": EventType.STEP_STARTED.value},
        ]
        assert online.dropped_events == 1
    finally:
        processor.release.set()
        await online.aclose()


def test_online_builder_exposes_queue_configuration():
    online = OnlineTransitionDatasetBuilder(max_queue_size=8, overflow_policy=QueueOverflowPolicy.DROP_NEWEST)
    assert online.max_queue_size == 8
    assert online.overflow_policy == QueueOverflowPolicy.DROP_NEWEST
