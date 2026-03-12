import asyncio
import argparse
import json
import httpx
from tooloceans.context import RunContext
from tooloceans.trajectory import Step
from tooloceans.impl.executor import AsyncExecutor
from tooloceans.impl.bus import InProcessEventBus
from tooloceans.impl.hot_store import InMemoryHotStore
from tooloceans.impl.cold_store import LocalFileColdStore
from tooloceans.episode_manager import EpisodeManager
from tools import build_registry
from policy import OllamaPolicy, SYSTEM_PROMPT


JUDGE_PROMPT = """\
You are a strict evaluator. Given a weather question and an agent's final answer, rate the answer quality.

Question: {question}
Final Answer: {answer}

Reply with a single float between 0.0 and 1.0.
- 1.0: answer is accurate, specific, and directly addresses the question
- 0.5: answer is partially correct or vague
- 0.0: answer is wrong, missing, or refuses to answer

Reply with ONLY the number, nothing else."""


class LLMRewardFn:
    def __init__(self, question: str, model: str, base_url: str) -> None:
        self._question = question
        self._model = model
        self._base_url = base_url

    async def score(self, step: Step, ctx) -> float:
        final_answer = step.metadata.get("final_answer", "")
        if not final_answer:
            return 0.0
        prompt = JUDGE_PROMPT.format(question=self._question, answer=final_answer)
        async with httpx.AsyncClient(timeout=30) as client:
            resp = await client.post(
                f"{self._base_url}/api/generate",
                json={"model": self._model, "prompt": prompt, "stream": False},
            )
            resp.raise_for_status()
            text = resp.json()["response"].strip()
        try:
            return max(0.0, min(1.0, float(text)))
        except ValueError:
            return 0.0


def to_sft_sample(episode, question: str) -> dict:
    """Reconstruct the full ReAct conversation as a single (prompt, response) pair."""
    turns = []
    for step in episode.steps:
        thought = step.metadata.get("thought", "")
        call = step.tool_calls[0] if step.tool_calls else None
        obs = step.observation[0] if step.observation else {}
        if thought:
            turns.append(f"Thought: {thought}")
        if call:
            turns.append(f"Action: {call.tool_name}")
            turns.append(f"Input: {json.dumps(call.arguments)}")
        turns.append(f"Observation: {json.dumps(obs)}")

    final = episode.metadata.get("final_step", {})
    if final.get("thought"):
        turns.append(f"Thought: {final['thought']}")
    if final.get("final_answer"):
        turns.append(f"Final Answer: {final['final_answer']}")

    return {
        "episode_id": episode.episode_id,
        "prompt": f"{SYSTEM_PROMPT}\n\nUser: {question}",
        "response": "\n".join(turns),
        "terminal_reward": episode.terminal_reward,
    }


class FinalAnswerRewardHook:
    """Score the final_answer from episode.metadata and set terminal_reward."""

    def __init__(self, question: str, model: str, base_url: str) -> None:
        self._question = question
        self._model = model
        self._base_url = base_url

    async def on_episode_end(self, episode) -> object:
        final = episode.metadata.get("final_step", {})
        answer = final.get("final_answer", "")
        if not answer:
            episode.terminal_reward = 0.0
            return episode
        prompt = JUDGE_PROMPT.format(question=self._question, answer=answer)
        async with httpx.AsyncClient(timeout=30) as client:
            resp = await client.post(
                f"{self._base_url}/api/generate",
                json={"model": self._model, "prompt": prompt, "stream": False},
            )
            resp.raise_for_status()
            text = resp.json()["response"].strip()
        try:
            episode.terminal_reward = max(0.0, min(1.0, float(text)))
        except ValueError:
            episode.terminal_reward = 0.0
        return episode


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Run the weather-focused ReAct agent example.",
    )
    parser.add_argument(
        "--question",
        nargs="?",
        default="What will the weather be like in Shanghai, Tokyo, Denver over the next 3, 5, 9 days, and what should I watch out for?",
        help="Weather question for the agent to answer.",
    )
    parser.add_argument(
        "--model",
        default="qwen2.5:32b",
        help="Ollama model name.",
    )
    parser.add_argument(
        "--base-url",
        default="http://localhost:8080",
        help="Ollama base URL.",
    )
    parser.add_argument(
        "--max-steps",
        type=int,
        default=10,
        help="Maximum number of agent steps.",
    )
    return parser


async def main(
    question: str,
    model: str = "qwen2.5:32b",
    base_url: str = "http://localhost:8080",
    max_steps: int = 10,
) -> None:
    registry = build_registry()
    bus = InProcessEventBus()
    hot = InMemoryHotStore()
    await bus.subscribe(hot)

    manager = EpisodeManager(
        executor=AsyncExecutor(registry, bus),
        bus=bus,
        cold_store=LocalFileColdStore("/Users/apple/ToolOceansInfra/examples/react_agent"),
        step_reward_fn=LLMRewardFn(question, model, base_url),
        hooks=[FinalAnswerRewardHook(question, model, base_url)],
    )

    print(f"Weather Question: {question}\n")

    policy = OllamaPolicy(model=model, base_url=base_url)
    policy._messages.append({"role": "user", "content": question})

    ctx = RunContext()
    episode = await manager.run_episode_dynamic(ctx, policy, max_steps=max_steps)

    print(f"\n--- Episode {episode.episode_id} ---")
    print(f"Steps: {len(episode.steps)}")
    for i, step in enumerate(episode.steps):
        calls = [f"{c.tool_name}({c.arguments})" for c in step.tool_calls]
        print(f"  step {i}: {calls} | reward={step.reward}")

    final = episode.metadata.get("final_step", {})
    print(f"\nFinal Answer: {final.get('final_answer', '(none)')}")
    print(f"Terminal Reward: {episode.terminal_reward}")

    sft = to_sft_sample(episode, question)
    sft_path = f"/Users/apple/ToolOceansInfra/examples/react_agent/{episode.episode_id}_sft.json"
    with open(sft_path, "w") as f:
        json.dump(sft, f, ensure_ascii=False, indent=2)
    print(f"\nSFT sample saved to {sft_path}")


if __name__ == "__main__":
    args = _build_parser().parse_args()
    asyncio.run(
        main(
            question=args.question,
            model=args.model,
            base_url=args.base_url,
            max_steps=args.max_steps,
        )
    )
