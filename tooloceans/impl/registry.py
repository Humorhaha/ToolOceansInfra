from __future__ import annotations
import importlib
import inspect
import types
from typing import Any
from ..registry import ToolSpec, ToolHandler, ToolCapability
from ..context import RunContext


def tool(
    *,
    name: str | None = None,
    version: str = "1",
    description: str = "",
    input_schema: dict[str, Any] | None = None,
    output_schema: dict[str, Any] | None = None,
    capability: ToolCapability | None = None,
) -> Any:
    """Decorator that marks an async function as a registered tool.

    Usage::

        @tool(name="get_weather", description="Fetch current weather.")
        async def get_weather(args, ctx): ...

    The decorated function gains a ``_tool_spec`` attribute. Pass the module
    (or any object) to ``InMemoryToolRegistry.register_module()`` to bulk-register.
    """
    def decorator(fn: Any) -> Any:
        spec = ToolSpec(
            name=name or fn.__name__,
            version=version,
            description=description or (inspect.getdoc(fn) or ""),
            input_schema=input_schema or {},
            output_schema=output_schema or {},
            capability=capability or ToolCapability(),
        )
        fn._tool_spec = spec
        return fn
    return decorator


class InMemoryToolRegistry:
    def __init__(self) -> None:
        # key: (name, version)
        self._store: dict[tuple[str, str], tuple[ToolSpec, ToolHandler]] = {}

    def register(self, spec: ToolSpec, handler: ToolHandler) -> None:
        self._store[(spec.name, spec.version)] = (spec, handler)

    def register_module(self, module: types.ModuleType | str) -> None:
        """Register all @tool-decorated functions found in *module*.

        *module* may be a module object or a dotted import path string.
        """
        if isinstance(module, str):
            module = importlib.import_module(module)
        for _, obj in inspect.getmembers(module, inspect.isfunction):
            spec = getattr(obj, "_tool_spec", None)
            if spec is not None:
                self.register(spec, obj)

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
