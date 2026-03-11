import uuid

from tooloceans.impl.dataset import EpisodeDatasetBuilder, OfflineTransitionDatasetBuilder
from tooloceans.trajectory import Episode, Step, ToolCall, ToolResult


def _make_call(value: int) -> ToolCall:
    return ToolCall(
        tool_name="echo",
        arguments={"value": value},
        call_id=str(uuid.uuid4()),
    )


def test_offline_builder_inherits_base_and_builds_episode_samples():
    builder = EpisodeDatasetBuilder()
    episode = Episode(
        episode_id="episode-1",
        run_id="run-1",
        steps=[
            Step(
                step_id="step-1",
                tool_calls=[_make_call(1)],
                tool_results=[ToolResult(call_id="r1", output={"echo": 1})],
                observation=[{"echo": 1}],
                reward=1.0,
            ),
            Step(
                step_id="step-2",
                tool_calls=[_make_call(2)],
                tool_results=[ToolResult(call_id="r2", output={"echo": 2})],
                observation=[{"echo": 2}],
                reward=2.0,
            ),
        ],
        terminal_reward=3.0,
    )

    builder.add_episode(episode)
    dataset = builder.build()

    assert len(dataset) == 1
    assert dataset[0]["episode_id"] == "episode-1"
    assert dataset[0]["steps"][0]["observation"] == [{"echo": 1}]
    assert dataset[0]["steps"][0]["next_observation"] == [{"echo": 2}]
    assert dataset[0]["steps"][1]["next_observation"] is None
    assert dataset[0]["terminal_reward"] == 3.0


def test_offline_transition_builder_emits_flat_transitions():
    builder = OfflineTransitionDatasetBuilder()
    episode = Episode(
        episode_id="episode-1",
        run_id="run-1",
        steps=[
            Step(
                step_id="step-1",
                tool_calls=[_make_call(1)],
                tool_results=[ToolResult(call_id="r1", output={"echo": 1})],
                observation=[{"echo": 1}],
                reward=1.0,
            ),
            Step(
                step_id="step-2",
                tool_calls=[_make_call(2)],
                tool_results=[ToolResult(call_id="r2", output={"echo": 2})],
                observation=[{"echo": 2}],
                reward=2.0,
            ),
        ],
        terminal_reward=3.0,
    )

    builder.add_episode(episode)
    dataset = builder.build()

    assert len(dataset) == 2
    assert dataset[0]["run_id"] == "run-1"
    assert dataset[0]["episode_id"] == "episode-1"
    assert dataset[0]["step_id"] == "step-1"
    assert dataset[0]["trace_id"] is None
    assert dataset[0]["observation"] == [{"echo": 1}]
    assert dataset[0]["actions"][0]["arguments"] == {"value": 1}
    assert dataset[0]["reward"] == 1.0
    assert dataset[0]["next_observation"] == [{"echo": 2}]
    assert dataset[0]["done"] is False

    assert dataset[1]["step_id"] == "step-2"
    assert dataset[1]["next_observation"] is None
    assert dataset[1]["done"] is True
