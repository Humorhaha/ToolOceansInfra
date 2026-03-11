"""
Basic end-to-end example: register two tools, run a 2-step episode,
inspect the trajectory and hot-store events.
"""
import asyncio
import uuid
from tooloceans.context import RunContext
from tooloceans.trajectory import ToolCall
from tooloceans.registry import ToolSpec, ToolCapability
from tooloceans.episode_manager import EpisodeManager
from tooloceans.impl.registry import InMemoryToolRegistry
from tooloceans.impl.executor import AsyncExecutor
from tooloceans.impl.bus import InProcessEventBus
from tooloceans.impl.hot_store import InMemoryHotStore
from tooloceans.impl.cold_store import LocalFileColdStore


# --- tool handlers ---

async def add(arguments: dict, ctx: RunContext) -> dict:
    return {"result": arguments["a"] + arguments["b"]}


async def echo(arguments: dict, ctx: RunContext) -> dict:
    return {"echo": arguments["message"]}


async def main() -> None:
    # wiring
    registry = InMemoryToolRegistry()
    registry.register(
        ToolSpec(
            name="add", version="1.0",
            input_schema={"a": "number", "b": "number"},
            output_schema={"result": "number"},
            capability=ToolCapability(has_side_effects=False, is_deterministic=True, is_pure_read=True),
        ),
        add,
    )
    registry.register(
        ToolSpec(
            name="echo", version="1.0",
            input_schema={"message": "string"},
            output_schema={"echo": "string"},
            capability=ToolCapability(has_side_effects=False, is_deterministic=True),
        ),
        echo,
    )

    bus = InProcessEventBus()
    hot_store = InMemoryHotStore()
    await bus.subscribe(hot_store)

    cold_store = LocalFileColdStore("/tmp/tooloceans_episodes")
    executor = AsyncExecutor(registry, bus)
    manager = EpisodeManager(executor, bus, cold_store)

    ctx = RunContext(experiment_metadata={"experiment": "basic_run"})

    # step 0: two parallel tool calls
    # step 1: single echo call
    steps_input = [
        [
            ToolCall(tool_name="add", arguments={"a": 1, "b": 2}, call_id=str(uuid.uuid4())),
            ToolCall(tool_name="add", arguments={"a": 10, "b": 20}, call_id=str(uuid.uuid4())),
        ],
        [
            ToolCall(tool_name="echo", arguments={"message": "hello"}, call_id=str(uuid.uuid4())),
        ],
    ]

    episode = await manager.run_episode(ctx, steps_input, metadata={"task": "demo"})

    print(f"episode_id : {episode.episode_id}")
    print(f"steps      : {len(episode.steps)}")
    for i, step in enumerate(episode.steps):
        print(f"  step {i}: calls={len(step.tool_calls)}  observation={step.observation}")

    print(f"\nhot-store events ({len(hot_store.events())}):")
    for e in hot_store.events():
        print(f"  {e.type.value}")

    traj = episode.as_rl_trajectory()
    print(f"\nRL trajectory steps: {len(traj)}")


if __name__ == "__main__":
    asyncio.run(main())
