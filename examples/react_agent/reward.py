from __future__ import annotations
import json
import httpx
from tooloceans.trajectory import Step
from policy import SYSTEM_PROMPT

JUDGE_PROMPT = """\
You are a strict evaluator. Given a weather question and an agent's final answer, rate the answer quality.

Question: {question}
Final Answer: {answer}

Reply with a single float between 0.0 and 1.0.
- 1.0: answer is accurate, specific, and directly addresses the question
- 0.5: answer is partially correct or vague
- 0.0: answer is wrong, missing, or refuses to answer

Reply with ONLY the number, nothing else."""


async def _llm_score(question: str, answer: str, model: str, base_url: str) -> float:
    if not answer:
        return 0.0
    prompt = JUDGE_PROMPT.format(question=question, answer=answer)
    async with httpx.AsyncClient(timeout=30) as client:
        resp = await client.post(
            f"{base_url}/api/generate",
            json={"model": model, "prompt": prompt, "stream": False},
        )
        resp.raise_for_status()
        text = resp.json()["response"].strip()
    try:
        return max(0.0, min(1.0, float(text)))
    except ValueError:
        return 0.0


class LLMRewardFn:
    def __init__(self, question: str, model: str, base_url: str) -> None:
        self._question = question
        self._model = model
        self._base_url = base_url

    async def score(self, step: Step, ctx) -> float:
        return await _llm_score(
            self._question,
            step.metadata.get("final_answer", ""),
            self._model,
            self._base_url,
        )


class FinalAnswerRewardHook:
    def __init__(self, question: str, model: str, base_url: str) -> None:
        self._question = question
        self._model = model
        self._base_url = base_url

    async def on_episode_end(self, episode) -> object:
        final = episode.metadata.get("final_step", {})
        episode.terminal_reward = await _llm_score(
            self._question,
            final.get("final_answer", ""),
            self._model,
            self._base_url,
        )
        return episode


def to_sft_sample(episode, question: str) -> dict:
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
