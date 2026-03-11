"""
Example: subscribe an online transition builder to the event bus and
collect RL-ready transitions as an episode executes.
"""
import asyncio
import uuid

from tooloceans.context import RunContext
from tooloceans.episode_manager import EpisodeManager
from tooloceans.impl.bus import InProcessEventBus
from tooloceans.impl.cold_store import LocalFileColdStore
from tooloceans.impl.dataset import OnlineTransitionDatasetBuilder, QueueOverflowPolicy
from tooloceans.impl.executor import AsyncExecutor
from tooloceans.impl.registry import InMemoryToolRegistry
from tooloceans.registry import ToolSpec
from tooloceans.trajectory import ToolCall


async def echo(arguments: dict, ctx: RunContext) -> dict:
    return {"echo": arguments["message"]}


async def main() -> None:
    registry = InMemoryToolRegistry()
    registry.register(
        ToolSpec(name="echo", version="1.0", input_schema={}, output_schema={}),
        echo,
    )

    bus = InProcessEventBus()
    online = OnlineTransitionDatasetBuilder(
        max_queue_size=1024,
        overflow_policy=QueueOverflowPolicy.RAISE,
    )
    await bus.subscribe(online)

    executor = AsyncExecutor(registry, bus)
    cold_store = LocalFileColdStore("/tmp/tooloceans_online")
    manager = EpisodeManager(executor, bus, cold_store)

    ctx = RunContext()
    steps_input = [
        [ToolCall(tool_name="echo", arguments={"message": "hello"}, call_id=str(uuid.uuid4()))],
        [ToolCall(tool_name="echo", arguments={"message": "world"}, call_id=str(uuid.uuid4()))],
    ]

    try:
        await manager.run_episode(ctx, steps_input)
        # build() only reflects transitions already processed by the worker.
        await online.drain()

        for transition in online.build():
            print(transition)
    finally:
        await online.aclose()


if __name__ == "__main__":
    asyncio.run(main())
