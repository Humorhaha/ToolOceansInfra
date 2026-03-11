import pytest
import uuid
from tooloceans.context import RunContext
from tooloceans.agent import PolicyDecision
from tooloceans.trajectory import ToolCall, Step
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


def build_stack(tmp_path, step_reward_fn=None):
    registry = InMemoryToolRegistry()
    spec = ToolSpec(name="echo", version="1", input_schema={}, output_schema={})
    async def echo_handler(args, ctx): return args
    registry.register(spec, echo_handler)
    bus = InProcessEventBus()
    hot = InMemoryHotStore()
    executor = AsyncExecutor(registry, bus)
    cold = LocalFileColdStore(tmp_path)
    manager = EpisodeManager(executor, bus, cold, step_reward_fn=step_reward_fn)
    return manager, bus, hot


class ConstantRewardFn:
    async def score(self, step: Step, ctx) -> float:
        return 1.0


@pytest.mark.asyncio
async def test_step_rewards_populated(tmp_path):
    manager, bus, hot = build_stack(tmp_path, step_reward_fn=ConstantRewardFn())
    await bus.subscribe(hot)

    ctx = RunContext()
    calls = [[_make_call()], [_make_call()]]
    episode = await manager.run_episode(ctx, calls)

    assert all(s.reward == 1.0 for s in episode.steps)


@pytest.mark.asyncio
async def test_reward_attached_events_emitted(tmp_path):
    manager, bus, hot = build_stack(tmp_path, step_reward_fn=ConstantRewardFn())
    await bus.subscribe(hot)

    ctx = RunContext()
    calls = [[_make_call()], [_make_call()]]
    await manager.run_episode(ctx, calls)

    reward_events = [
        e for e in hot.events_for_episode(ctx.episode_id)
        if e.type == EventType.REWARD
    ]
    assert len(reward_events) == 2
    assert all(e.payload["reward"] == 1.0 for e in reward_events)


@pytest.mark.asyncio
async def test_no_reward_fn_leaves_reward_none(tmp_path):
    manager, bus, hot = build_stack(tmp_path)
    await bus.subscribe(hot)

    ctx = RunContext()
    episode = await manager.run_episode(ctx, [[_make_call()]])

    assert episode.steps[0].reward is None
    reward_events = [
        e for e in hot.events_for_episode(ctx.episode_id)
        if e.type == EventType.REWARD
    ]
    assert len(reward_events) == 0


@pytest.mark.asyncio
async def test_dynamic_episode_rewards(tmp_path):
    manager, bus, hot = build_stack(tmp_path, step_reward_fn=ConstantRewardFn())
    await bus.subscribe(hot)

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
    assert all(s.reward == 1.0 for s in episode.steps)
    reward_events = [
        e for e in hot.events_for_episode(ctx.episode_id)
        if e.type == EventType.REWARD
    ]
    assert len(reward_events) == 2
