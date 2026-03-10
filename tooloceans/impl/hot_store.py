from __future__ import annotations
from ..events import Event


class InMemoryHotStore:
    def __init__(self) -> None:
        self._events: list[Event] = []

    async def handle(self, event: Event) -> None:
        self._events.append(event)

    async def emit(self, event: Event) -> None:
        self._events.append(event)

    def events(self) -> list[Event]:
        return list(self._events)

    def events_for_episode(self, episode_id: str) -> list[Event]:
        return [e for e in self._events if e.episode_id == episode_id]
