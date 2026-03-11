from __future__ import annotations
from ..bus import EventHandler
from ..events import Event


class InProcessEventBus:
    def __init__(self) -> None:
        self._handlers: list[EventHandler] = []

    async def emit(self, event: Event) -> None:
        import asyncio
        await asyncio.gather(*(h.handle(event) for h in self._handlers))

    async def subscribe(self, handler: EventHandler) -> None:
        self._handlers.append(handler)
