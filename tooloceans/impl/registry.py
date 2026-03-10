from __future__ import annotations
from typing import Any
from ..registry import ToolSpec, ToolHandler
from ..context import RunContext


class InMemoryToolRegistry:
    def __init__(self) -> None:
        # key: (name, version)
        self._store: dict[tuple[str, str], tuple[ToolSpec, ToolHandler]] = {}

    def register(self, spec: ToolSpec, handler: ToolHandler) -> None:
        self._store[(spec.name, spec.version)] = (spec, handler)

    def get(self, name: str, version: str | None = None) -> tuple[ToolSpec, ToolHandler]:
        if version is not None:
            return self._store[(name, version)]
        # latest = highest version string among matches
        matches = [(k, v) for k, v in self._store.items() if k[0] == name]
        if not matches:
            raise KeyError(f"Tool not found: {name}")
        matches.sort(key=lambda x: x[0][1])
        return matches[-1][1]

    def list(self) -> list[ToolSpec]:
        return [spec for spec, _ in self._store.values()]
