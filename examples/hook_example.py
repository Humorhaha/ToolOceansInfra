"""
Example: use PostEpisodeHook to enrich an Episode before it is persisted.
"""
import asyncio
import uuid

from tooloceans.context import RunContext
from tooloceans.episode_manager import EpisodeManager
from tooloceans.hooks import PostEpisodeHook
from tooloceans.impl.bus import InProcessEventBus
from tooloceans.impl.cold_store import LocalFileColdStore
from tooloceans.impl.executor import AsyncExecutor
from tooloceans.impl.registry import InMemoryToolRegistry
from tooloceans.registry import ToolSpec
from tooloceans.trajectory import Episode, ToolCall


async def add(arguments: dict, ctx: RunContext) -> dict:
    return {"result": arguments["a"] + arguments["b"]}


async def fail(arguments: dict, ctx: RunContext) -> dict:
    raise RuntimeError("demo failure")


class ScoreEpisodeHook(PostEpisodeHook):
    async def on_episode_end(self, episode: Episode) -> Episode:
        total_calls = 0
        total_errors = 0

        for step in episode.steps:
            total_calls += len(step.tool_results)
            total_errors += sum(1 for result in step.tool_results if result.error is not None)

        episode.metadata["stats"] = {
            "total_calls": total_calls,
            "total_errors": total_errors,
        }
        episode.terminal_reward = float(total_calls - total_errors)
        return episode


async def main() -> None:
    registry = InMemoryToolRegistry()
    registry.register(
        ToolSpec(name="add", version="1.0", input_schema={}, output_schema={}),
        add,
    )
    registry.register(
        ToolSpec(name="fail", version="1.0", input_schema={}, output_schema={}),
        fail,
    )

    bus = InProcessEventBus()
    cold_store = LocalFileColdStore("/tmp/tooloceans_episodes")
    executor = AsyncExecutor(registry, bus)
    hook = ScoreEpisodeHook()
    manager = EpisodeManager(executor, bus, cold_store, hooks=[hook])

    ctx = RunContext()
    steps_input = [
        [
            ToolCall(
                tool_name="add",
                arguments={"a": 1, "b": 2},
                call_id=str(uuid.uuid4()),
            ),
            ToolCall(
                tool_name="fail",
                arguments={},
                call_id=str(uuid.uuid4()),
            ),
        ]
    ]

    episode = await manager.run_episode(ctx, steps_input, metadata={"task": "hook-demo"})

    print("episode_id:", episode.episode_id)
    print("terminal_reward:", episode.terminal_reward)
    print("metadata.stats:", episode.metadata["stats"])
    print("observation:", episode.steps[0].observation)


if __name__ == "__main__":
    asyncio.run(main())
