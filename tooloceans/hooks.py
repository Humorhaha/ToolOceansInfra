from __future__ import annotations
from typing import Protocol
from .trajectory import Episode


class PostEpisodeHook(Protocol):
    async def on_episode_end(self, episode: Episode) -> Episode: ...
