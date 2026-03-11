from __future__ import annotations
from typing import Protocol
from .trajectory import Episode, Step
from .context import RunContext


class PostEpisodeHook(Protocol):
    async def on_episode_end(self, episode: Episode) -> Episode: ...


class StepRewardFn(Protocol):
    async def score(self, step: Step, ctx: RunContext) -> float: ...
