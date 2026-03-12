from __future__ import annotations
import json
import re
import uuid
import httpx
from typing import Any
from tooloceans.agent import PolicyDecision
from tooloceans.context import RunContext
from tooloceans.trajectory import ToolCall

SYSTEM_PROMPT = """\
You are a specialized ReAct weather analyst. Your job is to answer weather and forecast questions with clear, practical conclusions.

Available tools:
- resolve_location: find the most relevant place match. Input: {"query": "<location>"}
- get_current_weather: fetch the current weather for coordinates. Input: {"latitude": <number>, "longitude": <number>}
- get_weather_forecast: fetch the next 1-7 days of forecast for coordinates. Input: {"latitude": <number>, "longitude": <number>, "days": <integer>}
- assess_weather_risk: summarize forecast risk from a forecast object. Input: {"forecast": <forecast object>}

Recommended workflow:
1. Resolve the location first unless coordinates are already known.
2. Use current weather when the user asks about "now", "today", or current conditions.
3. Use get_weather_forecast for prediction questions.
4. Use assess_weather_risk when the user needs planning advice, trip advice, or a concise risk summary.
5. Finish with a direct answer that cites the key weather signals.

Rules:
- Prefer the smallest number of tool calls needed for a correct answer.
- Do not invent coordinates or weather values.
- If a tool returns an error, recover if possible; otherwise explain the limitation in the final answer.
- If the user asks for advice, ground it in the tool results.

Each response must follow EXACTLY one of these two formats:

Format 1 (use a tool):
Thought: <your reasoning>
Action: <tool_name>
Input: <json arguments>

Format 2 (final answer):
Thought: <your reasoning>
Final Answer: <your answer>

Do not add any other text."""

_ACTION_RE = re.compile(r"Action:\s*(\w+)", re.IGNORECASE)
_INPUT_RE = re.compile(r"Input:\s*(\{.*?\})", re.DOTALL)
_FINAL_RE = re.compile(r"Final Answer:\s*(.*)", re.IGNORECASE | re.DOTALL)
_THOUGHT_RE = re.compile(r"Thought:\s*(.*?)(?=\nAction:|\nFinal Answer:|$)", re.DOTALL)


class OllamaPolicy:
    def __init__(self, model: str = "qwen2.5:32b", base_url: str = "http://localhost:8080") -> None:
        self._model = model
        self._base_url = base_url
        self._messages: list[dict] = [{"role": "system", "content": SYSTEM_PROMPT}]

    async def decide(self, observation: Any, ctx: RunContext) -> PolicyDecision:
        if observation is not None:
            obs_text = "\n".join(
                f"Observation: {json.dumps(o)}" for o in observation
            )
            self._messages.append({"role": "user", "content": obs_text})

        response_text = await self._call_ollama()
        self._messages.append({"role": "assistant", "content": response_text})

        print(f"\n[LLM]\n{response_text}")

        thought = m.group(1).strip() if (m := _THOUGHT_RE.search(response_text)) else ""

        final_match = _FINAL_RE.search(response_text)
        if final_match:
            return PolicyDecision(
                actions=[],
                done=True,
                metadata={"thought": thought, "final_answer": final_match.group(1).strip(), "llm_response": response_text},
            )

        action_match = _ACTION_RE.search(response_text)
        input_match = _INPUT_RE.search(response_text)

        if not action_match or not input_match:
            return PolicyDecision(actions=[], done=True, metadata={"thought": thought, "llm_response": response_text})

        tool_name = action_match.group(1).strip()
        try:
            arguments = json.loads(input_match.group(1))
        except json.JSONDecodeError:
            return PolicyDecision(actions=[], done=True, metadata={"thought": thought, "llm_response": response_text})

        return PolicyDecision(
            actions=[ToolCall(tool_name=tool_name, arguments=arguments, call_id=str(uuid.uuid4()))],
            done=False,
            metadata={"thought": thought, "llm_response": response_text},
        )

    async def _call_ollama(self) -> str:
        prompt = "\n\n".join(
            f"{message['role'].upper()}: {message['content']}"
            for message in self._messages
        )
        async with httpx.AsyncClient(timeout=60) as client:
            resp = await client.post(
                f"{self._base_url}/api/generate",
                json={"model": self._model, "prompt": prompt, "stream": False},
            )
            resp.raise_for_status()
            payload = resp.json()
            return payload["response"]
