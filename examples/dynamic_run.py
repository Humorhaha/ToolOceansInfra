import asyncio
import uuid
from tooloceans.context import RunContext
from tooloceans.agent import PolicyDecision
from tooloceans.trajectory import ToolCall
from tooloceans.registry import ToolSpec
from tooloceans.impl.registry import InMemoryToolRegistry
from tooloceans.impl.executor import AsyncExecutor
from tooloceans.impl.bus import InProcessEventBus
from tooloceans.impl.hot_store import InMemoryHotStore
from tooloceans.impl.cold_store import LocalFileColdStore
from tooloceans.episode_manager import EpisodeManager


async def main():
    registry = InMemoryToolRegistry()
    spec = ToolSpec(name="add", version="1", input_schema={}, output_schema={})
    async def add_handler(args, ctx):
        return {"result": args["a"] + args["b"]}
    registry.register(spec, add_handler)

    bus = InProcessEventBus()
    hot = InMemoryHotStore()
    await bus.subscribe(hot)

    executor = AsyncExecutor(registry, bus)
    cold = LocalFileColdStore("/tmp/tooloceans_example")
    manager = EpisodeManager(executor, bus, cold)

    step_count = 0

    class SimplePolicy:
        async def decide(self, obs, ctx):
            nonlocal step_count
            if step_count >= 3:
                return PolicyDecision(actions=[], done=True)
            call = ToolCall(
                tool_name="add",
                arguments={"a": step_count, "b": step_count + 1},
                call_id=str(uuid.uuid4()),
            )
            step_count += 1
            return PolicyDecision(actions=[call], done=False)

    ctx = RunContext()
    episode = await manager.run_episode_dynamic(ctx, SimplePolicy(), max_steps=10)

    print(f"episode_id: {episode.episode_id}")
    print(f"steps: {len(episode.steps)}")
    for i, step in enumerate(episode.steps):
        print(f"  step {i}: observation={step.observation}")

    print("hot-store event types:")
    for e in hot.events_for_episode(ctx.episode_id):
        print(f"  {e.type}")


asyncio.run(main())
