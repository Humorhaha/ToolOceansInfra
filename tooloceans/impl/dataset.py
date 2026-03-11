from __future__ import annotations
from typing import Any
from ..trajectory import Episode


class EpisodeDatasetBuilder:
    def __init__(self) -> None:
        self._episodes: list[Episode] = []

    def add_episode(self, episode: Episode) -> None:
        self._episodes.append(episode)

    # offline episode collection
    def build(self) -> list[dict[str, Any]]:
        result = []
        for ep in self._episodes:
            steps = []
            for i, step in enumerate(ep.steps):
                next_obs = ep.steps[i + 1].observation if i + 1 < len(ep.steps) else None
                steps.append({
                    "observation": step.observation,
                    "actions": [
                        {"tool_name": c.tool_name, "arguments": c.arguments, "call_id": c.call_id}
                        for c in step.tool_calls
                    ],
                    "reward": step.reward,
                    "next_observation": next_obs,
                })
            result.append({
                "episode_id": ep.episode_id,
                "steps": steps,
                "terminal_reward": ep.terminal_reward,
            })
        return result
