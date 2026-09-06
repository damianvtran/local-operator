"""The compiler independently gates loss and emits one explicit paste statement."""

from __future__ import annotations

from typing import Any

import pytest
from lop_osworld_v2_adapter import actions
from lop_osworld_v2_adapter.adapter import OSWorldV2Adapter

from local_operator.evaluation.action_surface import ActionAdmissionError
from local_operator.evaluation.protocol import (
    ActionBatch,
    ClickAction,
    PasteTextAction,
    TypeAction,
)
from tests.unit.evaluation.adapters.osworld.test_actions import _geo


def test_whole_batch_rejected_before_any_compilation(monkeypatch: pytest.MonkeyPatch) -> None:
    batch = ActionBatch(
        protocol_version="1.0",
        task_id="task",
        episode_id="episode",
        observation_id="obs",
        actions=(
            ClickAction(observation_id="obs", frame_id="screen", x=1, y=1),
            TypeAction(observation_id="obs", text="café 東京🙂"),
        ),
    )
    called: list[Any] = []
    monkeypatch.setattr(actions, "compile_action", lambda *args: called.append(args))
    with pytest.raises(ActionAdmissionError, match="use paste_text"):
        actions.compile_batch(batch, _geo())
    assert not called


@pytest.mark.parametrize(
    "keys,expected",
    [
        (("CTRL", "v"), ("ctrl", "v")),
        (("CTRL", "SHIFT", "v"), ("ctrl", "shift", "v")),
        (("SHIFT", "INSERT"), ("shift", "insert")),
        (("META", "v"), ("win", "v")),
    ],
)
def test_paste_uses_native_key_mapping_without_default_chord(
    monkeypatch: pytest.MonkeyPatch,
    keys: tuple[str, ...],
    expected: tuple[str, ...],
) -> None:
    captured: list[Any] = []
    monkeypatch.setattr(
        actions,
        "paste_text_source",
        lambda text, chord: captured.append((text, tuple(chord))) or "paste",
    )
    action = PasteTextAction(
        observation_id="obs", text="東京🙂", keys=keys, clipboard_policy="overwrite"
    )
    assert actions.compile_action(action) == "paste"
    assert captured == [("東京🙂", expected)]


def test_adapter_metadata_matches_compiler_admission() -> None:
    assert OSWorldV2Adapter().metadata.capabilities.action_surface() == actions.ACTION_SURFACE


def test_native_type_control_and_small_transport_are_unchanged() -> None:
    assert actions.compile_action(TypeAction(observation_id="obs", text="a\t\n\r")) == (
        "pyautogui.typewrite('a\\t\\n\\r')"
    )
