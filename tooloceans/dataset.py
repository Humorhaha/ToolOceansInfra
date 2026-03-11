from __future__ import annotations
from abc import ABC, abstractmethod
from typing import Any, Generic, Iterable, TypeVar, TypedDict
from .events import Event
from .trajectory import Episode, ToolCall


class ToolAction(TypedDict):
    tool_name: str
    arguments: dict[str, Any]
    call_id: str


class RLTransition(TypedDict):
    """Canonical internal RL transition contract used by offline and online builders."""

    run_id: str | None
    episode_id: str
    step_id: str | None
    trace_id: str | None
    observation: Any
    actions: list[ToolAction]
    reward: float | None
    next_observation: Any
    done: bool


class EpisodeStepSample(TypedDict):
    observation: Any
    actions: list[ToolAction]
    reward: float | None
    next_observation: Any


class EpisodeDatasetSample(TypedDict):
    episode_id: str
    steps: list[EpisodeStepSample]
    terminal_reward: float | None


DatasetItemT = TypeVar("DatasetItemT")


class DatasetBuilder(ABC, Generic[DatasetItemT]):
    """Base builder for in-memory dataset samples."""

    def __init__(self) -> None:
        self._items: list[DatasetItemT] = []

    def build(self) -> list[DatasetItemT]:
        """Return a snapshot of samples already materialized by this builder."""
        return list(self._items)

    def _append_item(self, item: DatasetItemT) -> None:
        self._items.append(item)

    def _build_actions(self, tool_calls: Iterable[ToolCall]) -> list[ToolAction]:
        return [
            {
                "tool_name": call.tool_name,
                "arguments": call.arguments,
                "call_id": call.call_id,
            }
            for call in tool_calls
        ]

    @abstractmethod
    def add_episode(self, episode: Episode) -> None:
        raise NotImplementedError


class TransitionDatasetBuilder(DatasetBuilder[RLTransition], ABC):
    """Base class for builders that emit the canonical RL transition schema."""

    def _build_transition(
        self,
        *,
        run_id: str | None,
        episode_id: str,
        step_id: str | None,
        trace_id: str | None,
        observation: Any,
        actions: list[ToolAction],
        reward: float | None,
        next_observation: Any,
        done: bool,
    ) -> RLTransition:
        return {
            "run_id": run_id,
            "episode_id": episode_id,
            "step_id": step_id,
            "trace_id": trace_id,
            "observation": observation,
            "actions": list(actions),
            "reward": reward,
            "next_observation": next_observation,
            "done": done,
        }


class OnlineDatasetBuilder(TransitionDatasetBuilder, ABC):
    """Base class for event-driven builders that materialize canonical RL transitions."""

    def add_episode(self, episode: Episode) -> None:
        raise NotImplementedError("OnlineDatasetBuilder does not ingest full episodes")

    @abstractmethod
    async def handle(self, event: Event) -> None:
        raise NotImplementedError
