from __future__ import annotations
import json
import dataclasses
from pathlib import Path
from ..trajectory import Episode, Step, ToolCall, ToolResult, ToolError


def _to_dict(obj: object) -> object:
    if dataclasses.is_dataclass(obj) and not isinstance(obj, type):
        return {k: _to_dict(v) for k, v in dataclasses.asdict(obj).items()}
    if isinstance(obj, list):
        return [_to_dict(i) for i in obj]
    return obj


class LocalFileColdStore:
    def __init__(self, path: str | Path) -> None:
        self._path = Path(path)
        self._path.mkdir(parents=True, exist_ok=True)

    async def save_episode(self, episode: Episode) -> None:
        file = self._path / f"{episode.episode_id}.jsonl"
        with file.open("w") as f:
            f.write(json.dumps(_to_dict(episode)) + "\n")

    async def load_episode(self, episode_id: str) -> Episode | None:
        file = self._path / f"{episode_id}.jsonl"
        if not file.exists():
            return None
        with file.open() as f:
            data = json.loads(f.readline())
        # minimal reconstruction
        steps = [
            Step(
                step_id=s["step_id"],
                tool_calls=[ToolCall(**tc) for tc in s["tool_calls"]],
                tool_results=[
                    ToolResult(
                        call_id=r["call_id"],
                        output=r["output"],
                        error=ToolError(**r["error"]) if r.get("error") else None,
                        duration_ms=r["duration_ms"],
                    )
                    for r in s["tool_results"]
                ],
                observation=s["observation"],
                reward=s["reward"],
                metadata=s.get("metadata", {}),
            )
            for s in data["steps"]
        ]
        return Episode(
            episode_id=data["episode_id"],
            run_id=data["run_id"],
            steps=steps,
            metadata=data["metadata"],
            terminal_reward=data["terminal_reward"],
        )
