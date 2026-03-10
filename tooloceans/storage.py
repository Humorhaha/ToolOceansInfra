from __future__ import annotations
from typing import Protocol
from .events import Event
from .trajectory import Episode


class HotStore(Protocol):
    async def emit(self, event: Event) -> None: ...


class ColdStore(Protocol):
    async def save_episode(self, episode: Episode) -> None: ...
    async def load_episode(self, episode_id: str) -> Episode | None: ...
