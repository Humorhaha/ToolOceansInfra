from __future__ import annotations
from typing import Protocol
from .events import Event


class EventHandler(Protocol):
    async def handle(self, event: Event) -> None: ...


class EventBus(Protocol):
    async def emit(self, event: Event) -> None: ...
    async def subscribe(self, handler: EventHandler) -> None: ...
