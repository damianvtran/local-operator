"""Negotiated admission is stricter than historic action-data parsing."""

from __future__ import annotations

import hashlib
import json

import pytest
from pydantic import ValidationError

from local_operator.evaluation.action_surface import ActionAdmissionError, ActionSurface
from local_operator.evaluation.adapters.api import AdapterCapabilities, AdapterSelector
from local_operator.evaluation.evidence.models import canonical_bytes, canonical_digest
from local_operator.evaluation.protocol import (
    ActionBatch,
    KeyAction,
    PasteTextAction,
    TypeAction,
)
from local_operator.evaluation.runner.provider_client import build_system_prompt


def batch(action: object) -> ActionBatch:
    return ActionBatch.model_validate(
        {
            "protocol_version": "1.0",
            "task_id": "task",
            "episode_id": "episode",
            "observation_id": "obs",
            "actions": [action],
        }
    )


def paste(**changes: object) -> PasteTextAction:
    return PasteTextAction.model_validate(
        {
            "observation_id": "obs",
            "text": "café 東京🙂",
            "keys": ["ctrl", "v"],
            "clipboard_policy": "overwrite",
            **changes,
        }
    )


@pytest.mark.parametrize("text", [" ", "\t\n\r", "e\u0301", "東京🙂", "🙂" * 100_000])
def test_unicode_and_whitespace_are_paste_data(text: str) -> None:
    action = paste(text=text)
    assert action.text == text
    assert action.keys == ("CTRL", "v")
    assert ActionBatch.from_canonical_json(batch(action).to_canonical_json()) == batch(action)


@pytest.mark.parametrize("text", ["", "x" * 100_001, "\0", "\x1b", "\x7f", "\x85", "\ud800"])
def test_invalid_paste_rejected_before_dispatch(text: str) -> None:
    with pytest.raises(ValidationError):
        paste(text=text)


@pytest.mark.parametrize("field", ["text", "keys", "clipboard_policy", "observation_id"])
def test_paste_has_no_implicit_policy_or_chord(field: str) -> None:
    payload = paste().model_dump(mode="json")
    del payload[field]
    with pytest.raises(ValidationError):
        PasteTextAction.model_validate(payload)


@pytest.mark.parametrize("keys", [[], ["control", "v"], ["CTRL", "ctrl"], ["a"] * 9, "ctrl+v"])
def test_paste_uses_the_same_key_validator(keys: object) -> None:
    for model, extra in [
        (KeyAction, {}),
        (PasteTextAction, {"text": "x", "clipboard_policy": "overwrite"}),
    ]:
        with pytest.raises(ValidationError):
            model.model_validate({"observation_id": "obs", "keys": keys, **extra})


def test_no_restore_policy() -> None:
    with pytest.raises(ValidationError):
        paste(clipboard_policy="restore")


def test_old_type_and_key_bytes_are_frozen() -> None:
    # Captured with the 743b014f frozen baseline interpreter/source, not derived
    # from the implementation under test. Adding paste must not rehash old input.
    old = [
        (
            TypeAction(observation_id="obs", text="café 東京🙂\t\n"),
            "5a469102f4362cb1af551110f4e2c560d6c785829e2a2a7b96922bdf8cccecfd",
        ),
        (
            KeyAction(observation_id="obs", keys=("ctrl", "v")),
            "d0d97bafcca4f751006d1a2f7928b4c0eb0b5858a2ee32c9d3808dea557fe689",
        ),
    ]
    for action, expected in old:
        encoded = batch(action).to_canonical_json()
        assert hashlib.sha256(encoded).hexdigest() == expected
        assert ActionBatch.from_canonical_json(encoded).to_canonical_json() == encoded


def test_negotiation_controls_admission_prompt_and_schema_identity() -> None:
    legacy = AdapterCapabilities(routes=("computer",), ask_user=True, scoring=True).action_surface()
    enabled = AdapterCapabilities(
        routes=("computer",), ask_user=False, scoring=True, paste_text=True, type_text_mode="ascii"
    ).action_surface()
    assert '"paste_text"' not in build_system_prompt(legacy)
    assert '"paste_text"' in build_system_prompt(enabled)
    assert '"ask_user"' not in build_system_prompt(enabled)
    assert '"keys": [str, ...]' in build_system_prompt(enabled)
    assert '"clipboard_policy": overwrite' in build_system_prompt(enabled)
    with pytest.raises(ActionAdmissionError):
        legacy.validate_batch(batch(paste()))
    enabled.validate_batch(batch(paste()))
    unicode_type = batch(TypeAction(observation_id="obs", text="東京"))
    legacy.validate_batch(unicode_type)
    with pytest.raises(ActionAdmissionError, match="use paste_text"):
        enabled.validate_batch(unicode_type)
    assert canonical_digest("runner-tool-schema-v1", enabled.schema()) != canonical_digest(
        "runner-tool-schema-v1", legacy.schema()
    )
    assert json.loads(canonical_bytes(enabled.schema())) == enabled.schema()


def test_old_rpc_refused_before_allocation() -> None:
    with pytest.raises(ValidationError, match="1.6"):
        AdapterSelector.model_validate({"schema_version": "1.4"})


def test_ascii_native_controls_are_not_reinterpreted_as_paste() -> None:
    ActionSurface(type_text_mode="ascii").validate_batch(
        batch(TypeAction(observation_id="obs", text="x\t\n\r"))
    )
