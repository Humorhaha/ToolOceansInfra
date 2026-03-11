from __future__ import annotations
from typing import Any, Protocol
from .trajectory import Episode


class DatasetBuilder(Protocol):
    def add_episode(self, episode: Episode) -> None: ...
    def build(self) -> list[dict[str, Any]]: ...
