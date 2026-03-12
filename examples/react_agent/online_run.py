"""
Online RL data collection loop.

Runs episodes continuously, materializing (s, a, r, s') transitions in real time
via OnlineTransitionDatasetBuilder. After each episode the transition buffer is
drained and printed — replace the print with your training update call.
"""
import asyncio
import argparse
import json
from tooloceans.context import RunContext
from tooloceans.impl.executor import AsyncExecutor
from tooloceans.impl.bus import InProcessEventBus
from tooloceans.impl.cold_store import LocalFileColdStore
from tooloceans.impl.dataset import OnlineTransitionDatasetBuilder
from tooloceans.episode_manager import EpisodeManager
from tools import build_registry
from policy import OllamaPolicy
from reward import LLMRewardFn, FinalAnswerRewardHook

QUESTIONS = [
    "What will the weather be like in Shanghai over the next 3 days?",
    "Is it safe to hike in Tokyo this weekend?",
    "Will it rain in Denver tomorrow?",
]


async def training_step(transitions: list[dict], episode_idx: int) -> None:
    """Placeholder: replace with actual gradient update."""
    print(f"\n[train] episode {episode_idx}: {len(transitions)} transitions")
    for t in transitions:
        print(
            f"  step={t['step_id'][:8]} "
            f"actions={[a['tool_name'] for a in t['actions']]} "
            f"reward={t['reward']} done={t['done']}"
        )


async def run_online(
    questions: list[str],
    model: str,
    base_url: str,
    max_steps: int,
    n_episodes: int,
) -> None:
    registry = build_registry()

    # one shared online builder across all episodes
    online_builder = OnlineTransitionDatasetBuilder()

    for ep_idx in range(n_episodes):
        question = questions[ep_idx % len(questions)]
        print(f"\n=== Episode {ep_idx} | Q: {question} ===")

        bus = InProcessEventBus()
        await bus.subscribe(online_builder)

        manager = EpisodeManager(
            executor=AsyncExecutor(registry, bus),
            bus=bus,
            cold_store=LocalFileColdStore("/Users/apple/ToolOceansInfra/examples/react_agent"),
            step_reward_fn=LLMRewardFn(question, model, base_url),
            hooks=[FinalAnswerRewardHook(question, model, base_url)],
        )

        policy = OllamaPolicy(model=model, base_url=base_url)
        policy._messages.append({"role": "user", "content": question})

        ctx = RunContext()
        episode = await manager.run_episode_dynamic(ctx, policy, max_steps=max_steps)

        # wait for all events from this episode to be processed
        await online_builder.drain()

        # take only transitions from this episode
        all_transitions = online_builder.build()
        episode_transitions = [
            t for t in all_transitions if t["episode_id"] == episode.episode_id
        ]

        print(f"terminal_reward={episode.terminal_reward}")
        await training_step(episode_transitions, ep_idx)

    await online_builder.aclose()
    print(f"\nTotal transitions collected: {len(online_builder.build())}")


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Online RL data collection loop.")
    parser.add_argument("--model", default="qwen2.5:32b")
    parser.add_argument("--base-url", default="http://localhost:8080")
    parser.add_argument("--max-steps", type=int, default=10)
    parser.add_argument("--n-episodes", type=int, default=3)
    return parser


if __name__ == "__main__":
    args = _build_parser().parse_args()
    asyncio.run(run_online(
        questions=QUESTIONS,
        model=args.model,
        base_url=args.base_url,
        max_steps=args.max_steps,
        n_episodes=args.n_episodes,
    ))
