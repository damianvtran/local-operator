"""Discriminating contract tests for the benchmark-neutral evaluation wire format."""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import pytest
from pydantic import ValidationError

from local_operator.evaluation.protocol import (
    MAX_BATCH_SIZE,
    ActionBatch,
    ArtifactRef,
    AskUserAction,
    ClickAction,
    DoubleClickAction,
    FinishAction,
    FrameGeometry,
    FrameRef,
    FrameSize,
    KeyAction,
    Observation,
    ObservationEnvelope,
    PixelPoint,
    ScrollAction,
    TypeAction,
    WaitAction,
    parse_envelope,
)

REPO = Path(__file__).resolve().parents[3]
DIGEST = "0123456789abcdef" * 4


def _artifact() -> ArtifactRef:
    return ArtifactRef(sha256=DIGEST, media_type="image/png", byte_count=1234)


def _frame(*, frame_id: str = "frame-1") -> FrameRef:
    return FrameRef(
        frame_id=frame_id,
        artifact=_artifact(),
        geometry=FrameGeometry(
            native=FrameSize(width=1920, height=1080),
            model_visible=FrameSize(width=1280, height=720),
        ),
    )


def _observation() -> Observation:
    return Observation(
        task_id="task-1",
        episode_id="episode-1",
        sequence=7,
        observation_id="observation-7",
        text="A settings window is visible.",
        frames=(_frame(),),
        metadata={"benchmark": "neutral", "attempt": 1, "nested": {"ready": True}},
    )


def _batch(*actions: object, observation_id: str = "observation-7") -> ActionBatch:
    return ActionBatch.model_validate(
        {
            "task_id": "task-1",
            "episode_id": "episode-1",
            "observation_id": observation_id,
            "actions": list(actions),
        }
    )


def test_artifact_ref_is_content_only_and_rejects_transport_fields() -> None:
    artifact = _artifact()
    assert artifact.model_dump() == {
        "sha256": DIGEST,
        "media_type": "image/png",
        "byte_count": 1234,
    }
    with pytest.raises(ValidationError):
        ArtifactRef.model_validate(
            {
                **artifact.model_dump(),
                "path": "../../secret",
            }
        )
    with pytest.raises(ValidationError):
        ArtifactRef(sha256="not-a-digest", media_type="image/png", byte_count=1)
    with pytest.raises(ValidationError):
        ArtifactRef(sha256=DIGEST, media_type="not a media type", byte_count=1)


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("width", 0),
        ("width", -1),
        ("width", 1.5),
        ("height", float("nan")),
        ("height", float("inf")),
    ],
)
def test_frame_size_rejects_nonpositive_noninteger_or_nonfinite_dimensions(
    field: str, value: object
) -> None:
    payload = {"width": 100, "height": 100, field: value}
    with pytest.raises(ValidationError):
        FrameSize.model_validate(payload)


def test_geometry_conversion_has_explicit_floor_and_clamp_policy() -> None:
    geometry = FrameGeometry(
        native=FrameSize(width=1920, height=1080),
        model_visible=FrameSize(width=1280, height=720),
    )
    assert geometry.model_to_native(1, 1) == PixelPoint(x=1, y=1)
    assert geometry.model_to_native(1279, 719) == PixelPoint(x=1918, y=1078)
    assert geometry.model_to_native(-2, 9999) == PixelPoint(x=0, y=1079)
    assert geometry.native_to_model(1919, 1079) == PixelPoint(x=1279, y=719)
    for invalid in (float("nan"), float("inf"), float("-inf")):
        with pytest.raises(ValueError, match="finite"):
            geometry.model_to_native(invalid, 1)


def test_observation_is_strict_ordered_and_forbids_extras() -> None:
    observation = _observation()
    assert [frame.frame_id for frame in observation.frames] == ["frame-1"]
    with pytest.raises(ValidationError):
        Observation.model_validate({**observation.model_dump(), "sequence": "7"})
    with pytest.raises(ValidationError):
        Observation.model_validate({**observation.model_dump(), "unknown": True})
    with pytest.raises(ValidationError, match="unique"):
        Observation.model_validate(
            {**observation.model_dump(), "frames": [observation.frames[0], observation.frames[0]]}
        )


@pytest.mark.parametrize("value", [float("nan"), float("inf"), float("-inf")])
def test_observation_metadata_rejects_nonfinite_numbers_at_any_depth(value: float) -> None:
    with pytest.raises(ValidationError, match="non-finite"):
        Observation.model_validate(
            {**_observation().model_dump(), "metadata": {"nested": [1, {"bad": value}]}}
        )


def test_action_union_is_closed_and_has_no_command_escape_hatches() -> None:
    base = {
        "task_id": "task-1",
        "episode_id": "episode-1",
        "observation_id": "observation-7",
    }
    with pytest.raises(ValidationError):
        ActionBatch.model_validate(
            {
                **base,
                "actions": [{"kind": "shell", "observation_id": "observation-7", "command": "id"}],
            }
        )
    with pytest.raises(ValidationError):
        ActionBatch.model_validate(
            {
                **base,
                "actions": [
                    {
                        "kind": "click",
                        "observation_id": "observation-7",
                        "frame_id": "frame-1",
                        "x": 2,
                        "y": 3,
                        "pyautogui": "click(2, 3)",
                    }
                ],
            }
        )


def test_each_action_binds_to_batch_observation() -> None:
    with pytest.raises(ValidationError, match="different observation_id"):
        _batch(TypeAction(observation_id="stale", text="hello"))


def test_batch_validate_for_rejects_stale_identity_and_bad_frame_coordinates() -> None:
    observation = _observation()
    with pytest.raises(ValueError, match="current task, episode, and observation"):
        _batch(
            WaitAction(observation_id="stale", duration_ms=1), observation_id="stale"
        ).validate_for(observation)
    with pytest.raises(ValueError, match="unknown frame_id"):
        _batch(
            ClickAction(
                observation_id=observation.observation_id,
                frame_id="old-frame",
                x=1,
                y=1,
            )
        ).validate_for(observation)
    with pytest.raises(ValueError, match="outside model-visible"):
        _batch(
            ClickAction(
                observation_id=observation.observation_id,
                frame_id="frame-1",
                x=1280,
                y=719,
            )
        ).validate_for(observation)


def test_terminal_actions_are_isolated_and_empty_batches_are_invalid() -> None:
    finish = FinishAction(observation_id="observation-7", status="done", reason="Task completed")
    ask = AskUserAction(
        observation_id="observation-7",
        request_id="request-1",
        question="Which account should I use?",
    )
    for terminal in (finish, ask):
        assert _batch(terminal).actions == (terminal,)
        with pytest.raises(ValidationError, match="only action"):
            _batch(WaitAction(observation_id="observation-7", duration_ms=1), terminal)
        with pytest.raises(ValidationError, match="only action"):
            _batch(terminal, WaitAction(observation_id="observation-7", duration_ms=1))
    with pytest.raises(ValidationError):
        _batch()


def test_batch_size_status_mouse_and_numeric_bounds_are_closed() -> None:
    with pytest.raises(ValidationError):
        _batch(*[WaitAction(observation_id="observation-7", duration_ms=1)] * (MAX_BATCH_SIZE + 1))
    with pytest.raises(ValidationError):
        FinishAction(observation_id="observation-7", status="maybe", reason="uncertain")  # type: ignore[arg-type]
    with pytest.raises(ValidationError):
        ClickAction(
            observation_id="observation-7", frame_id="frame-1", x=1, y=2, button="primary"  # type: ignore[arg-type]
        )
    with pytest.raises(ValidationError):
        ScrollAction(
            observation_id="observation-7",
            frame_id="frame-1",
            x=1,
            y=2,
            delta_y=0,
        )
    with pytest.raises(ValidationError):
        WaitAction(observation_id="observation-7", duration_ms=float("nan"))  # type: ignore[arg-type]


def test_key_validation_and_historical_double_click_are_explicit() -> None:
    assert KeyAction(observation_id="observation-7", keys=("CTRL", "a")).keys == ("CTRL", "a")
    assert (
        DoubleClickAction(observation_id="observation-7", frame_id="frame-1", x=10, y=20).kind
        == "double_click"
    )
    with pytest.raises(ValidationError, match="unknown key"):
        KeyAction(observation_id="observation-7", keys=("NOT_A_KEY",))


def test_batch_preserves_action_order_across_canonical_round_trip() -> None:
    batch = _batch(
        ClickAction(observation_id="observation-7", frame_id="frame-1", x=3, y=4),
        TypeAction(observation_id="observation-7", text="hello"),
        KeyAction(observation_id="observation-7", keys=("ENTER",)),
    )
    decoded = ActionBatch.from_canonical_json(batch.to_canonical_json())
    assert [action.kind for action in decoded.actions] == ["click", "type", "key"]
    decoded.validate_for(_observation())


def test_canonical_observation_fixture_is_stable_and_requires_exact_encoding() -> None:
    envelope = ObservationEnvelope(observation=_observation())
    expected = (
        '{"kind":"observation","observation":{"episode_id":"episode-1","frames":'
        '[{"artifact":{"byte_count":1234,"media_type":"image/png","sha256":"'
        + DIGEST
        + '"},"frame_id":"frame-1","geometry":{"model_visible":{"height":720,'
        '"origin":"top_left","unit":"pixel","width":1280},"native":{"height":1080,'
        '"origin":"top_left","unit":"pixel","width":1920}}}],"metadata":{"attempt":1,'
        '"benchmark":"neutral","nested":{"ready":true}},"observation_id":"observation-7",'
        '"sequence":7,"task_id":"task-1","text":"A settings window is visible."},'
        '"protocol_version":"1.0"}'
    ).encode()
    assert envelope.to_canonical_json() == expected
    assert ObservationEnvelope.from_canonical_json(expected) == envelope
    assert parse_envelope(expected) == envelope
    noncanonical = json.dumps(json.loads(expected), ensure_ascii=False).encode()
    with pytest.raises(ValueError, match="not canonical"):
        ObservationEnvelope.from_canonical_json(noncanonical)


@pytest.mark.parametrize("module", ["local_operator.cli", "local_operator.session_factory"])
def test_startup_imports_do_not_load_evaluation(module: str) -> None:
    imported = _fresh_import_modules(module)
    assert not {name for name in imported if name.startswith("local_operator.evaluation")}


def test_evaluation_package_import_is_inert() -> None:
    imported = _fresh_import_modules("local_operator.evaluation")
    forbidden_prefixes = (
        "PIL",
        "boto3",
        "gymnasium",
        "osworld",
        "OSWorld",
        "local_operator.evaluation.protocol",
        "local_operator.evaluation.adapter",
    )
    assert not {
        name
        for name in imported
        if any(name == prefix or name.startswith(prefix + ".") for prefix in forbidden_prefixes)
    }


def _fresh_import_modules(module: str) -> set[str]:
    probe = (
        "import importlib,json,sys;"
        "importlib.import_module(sys.argv[1]);"
        "print(json.dumps(sorted(sys.modules)))"
    )
    completed = subprocess.run(
        [sys.executable, "-c", probe, module],
        capture_output=True,
        text=True,
        cwd=REPO,
        check=False,
    )
    assert completed.returncode == 0, completed.stderr[-3000:]
    return set(json.loads(completed.stdout.strip().splitlines()[-1]))
