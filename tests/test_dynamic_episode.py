import pytest
import uuid
from tooloceans.context import RunContext
from tooloceans.agent import PolicyDecision
from tooloceans.trajectory import ToolCall
from tooloceans.events import EventType
from tooloceans.impl.registry import InMemoryToolRegistry
from tooloceans.impl.executor import AsyncExecutor
from tooloceans.impl.bus import InProcessEventBus
from tooloceans.impl.hot_store import InMemoryHotStore
from tooloceans.impl.cold_store import LocalFileColdStore
from tooloceans.episode_manager import EpisodeManager
from tooloceans.registry import ToolSpec


def _make_call():
    return ToolCall(tool_name="echo", arguments={"v": 1}, call_id=str(uuid.uuid4()))


def build_stack(tmp_path):
    registry = InMemoryToolRegistry()
    spec = ToolSpec(name="echo", version="1", input_schema={}, output_schema={})
    async def echo_handler(args, ctx):
        return args
    registry.register(spec, echo_handler)

    bus = InProcessEventBus()
    hot = InMemoryHotStore()
    import asyncio
    asyncio.get_event_loop().run_until_complete(bus.subscribe(hot))

    executor = AsyncExecutor(registry, bus)
    cold = LocalFileColdStore(tmp_path)
    manager = EpisodeManager(executor, bus, cold)
    return manager, bus, hot


@pytest.mark.asyncio
async def test_normal_termination_done_true(tmp_path):
    registry = InMemoryToolRegistry()
    spec = ToolSpec(name="echo", version="1", input_schema={}, output_schema={})
    async def echo_handler(args, ctx): return args
    registry.register(spec, echo_handler)
    bus = InProcessEventBus()
    hot = InMemoryHotStore()
    await bus.subscribe(hot)
    executor = AsyncExecutor(registry, bus)
    manager = EpisodeManager(executor, bus, LocalFileColdStore(tmp_path))

    call_count = 0
    class P:
        async def decide(self, obs, ctx):
            nonlocal call_count
            call_count += 1
            if call_count <= 2:
                return PolicyDecision(actions=[_make_call()], done=False)
            return PolicyDecision(actions=[], done=True)

    ctx = RunContext()
    episode = await manager.run_episode_dynamic(ctx, P(), max_steps=10)
    assert len(episode.steps) == 2


@pytest.mark.asyncio
async def test_max_steps_termination(tmp_path):
    registry = InMemoryToolRegistry()
    spec = ToolSpec(name="echo", version="1", input_schema={}, output_schema={})
    async def echo_handler(args, ctx): return args
    registry.register(spec, echo_handler)
    bus = InProcessEventBus()
    hot = InMemoryHotStore()
    await bus.subscribe(hot)
    executor = AsyncExecutor(registry, bus)
    manager = EpisodeManager(executor, bus, LocalFileColdStore(tmp_path))

    class P:
        async def decide(self, obs, ctx):
            return PolicyDecision(actions=[_make_call()], done=False)

    ctx = RunContext()
    episode = await manager.run_episode_dynamic(ctx, P(), max_steps=3)
    assert len(episode.steps) == 3


@pytest.mark.asyncio
async def test_immediate_termination(tmp_path):
    registry = InMemoryToolRegistry()
    spec = ToolSpec(name="echo", version="1", input_schema={}, output_schema={})
    async def echo_handler(args, ctx): return args
    registry.register(spec, echo_handler)
    bus = InProcessEventBus()
    hot = InMemoryHotStore()
    await bus.subscribe(hot)
    executor = AsyncExecutor(registry, bus)
    manager = EpisodeManager(executor, bus, LocalFileColdStore(tmp_path))

    class P:
        async def decide(self, obs, ctx):
            return PolicyDecision(actions=[], done=True)

    ctx = RunContext()
    episode = await manager.run_episode_dynamic(ctx, P(), max_steps=10)
    assert len(episode.steps) == 0


@pytest.mark.asyncio
async def test_observation_threading(tmp_path):
    registry = InMemoryToolRegistry()
    spec = ToolSpec(name="echo", version="1", input_schema={}, output_schema={})
    async def echo_handler(args, ctx): return {"echoed": args["v"]}
    registry.register(spec, echo_handler)
    bus = InProcessEventBus()
    hot = InMemoryHotStore()
    await bus.subscribe(hot)
    executor = AsyncExecutor(registry, bus)
    manager = EpisodeManager(executor, bus, LocalFileColdStore(tmp_path))

    received_obs = []
    call_count = 0

    class P:
        async def decide(self, obs, ctx):
            nonlocal call_count
            received_obs.append(obs)
            call_count += 1
            if call_count <= 2:
                return PolicyDecision(actions=[ToolCall(tool_name="echo", arguments={"v": call_count}, call_id=str(uuid.uuid4()))], done=False)
            return PolicyDecision(actions=[], done=True)

    ctx = RunContext()
    episode = await manager.run_episode_dynamic(ctx, P(), max_steps=10)

    assert received_obs[0] is None
    assert received_obs[1] == [{"echoed": 1}]
    assert received_obs[2] == [{"echoed": 2}]


@pytest.mark.asyncio
async def test_event_sequence(tmp_path):
    registry = InMemoryToolRegistry()
    spec = ToolSpec(name="echo", version="1", input_schema={}, output_schema={})
    async def echo_handler(args, ctx): return args
    registry.register(spec, echo_handler)
    bus = InProcessEventBus()
    hot = InMemoryHotStore()
    await bus.subscribe(hot)
    executor = AsyncExecutor(registry, bus)
    manager = EpisodeManager(executor, bus, LocalFileColdStore(tmp_path))

    call_count = 0
    class P:
        async def decide(self, obs, ctx):
            nonlocal call_count
            call_count += 1
            if call_count <= 2:
                return PolicyDecision(actions=[_make_call()], done=False)
            return PolicyDecision(actions=[], done=True)

    ctx = RunContext()
    episode = await manager.run_episode_dynamic(ctx, P(), max_steps=10)

    types = [e.type for e in hot.events_for_episode(ctx.episode_id)]
    # filter to the structural events only
    structural = [t for t in types if t in (
        EventType.EPISODE_STARTED, EventType.STEP_STARTED,
        EventType.STEP_ENDED, EventType.EPISODE_ENDED,
    )]
    assert structural == [
        EventType.EPISODE_STARTED,
        EventType.STEP_STARTED, EventType.STEP_ENDED,
        EventType.STEP_STARTED, EventType.STEP_ENDED,
        EventType.EPISODE_ENDED,
    ]
