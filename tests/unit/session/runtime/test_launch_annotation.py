"""The wake-path errand carriage for the input-mode annotation (input-mode-v1).

The wake path is a DIFFERENT route than the live one: a cold session's first
prompt travels ``MobileDaemon`` → ``ContinuationCommand`` → ``continue_command``
→ ``PromptErrand`` → ``launch._deliver`` → ``AttachClient``, and the annotation
must ride it exactly as it rides the live path — one rule, one place (the
client drops it for an owner that did not advertise the capability). The
``_fields`` case below is the id-attach rebuild (``slots=True`` dataclasses
rebuilt from ``__slots__``), where a forgotten field would silently drop the
annotation on a retried prompt.
"""

from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest

from local_operator.session.runtime import launch
from local_operator.session.runtime.launch import PromptErrand, SteerErrand


class _FakeAttach:
    created: list["_FakeAttach"] = []

    def __init__(self, *_args: Any, **_kwargs: Any) -> None:
        self.calls: list[tuple[str, dict[str, Any]]] = []
        _FakeAttach.created.append(self)

    async def connect(self, record: Any, session_id: str) -> None:
        self.connected_to = (record, session_id)

    async def request_ack_with_duplicate(self, op: str, **fields: Any) -> tuple[str, bool]:
        self.calls.append((op, fields))
        return "prompt sent", False

    def close(self) -> None:
        self.closed = True


@pytest.fixture()
def fake_attach(monkeypatch: pytest.MonkeyPatch) -> type[_FakeAttach]:
    _FakeAttach.created = []
    monkeypatch.setattr("local_operator.mobile.attach_client.AttachClient", _FakeAttach)
    return _FakeAttach


@pytest.mark.asyncio
async def test_a_prompt_errand_carries_the_annotation_to_the_wire(
    fake_attach: type[_FakeAttach],
) -> None:
    detail, duplicate = await launch._deliver(
        object(),
        "s",
        PromptErrand(text="hi", input_mode="mixed", input_path="provider_stt_radient"),
    )

    assert (detail, duplicate) == ("prompt sent", False)
    op, fields = fake_attach.created[-1].calls[-1]
    assert op == "prompt"
    assert fields["input_mode"] == "mixed"
    assert fields["input_path"] == "provider_stt_radient"


@pytest.mark.asyncio
async def test_a_legacy_errand_sends_no_annotation(fake_attach: type[_FakeAttach]) -> None:
    """The pre-feature errand's frame is byte-identical to today's."""
    await launch._deliver(object(), "s", PromptErrand(text="hi"))

    op, fields = fake_attach.created[-1].calls[-1]
    assert op == "prompt"
    assert "input_mode" not in fields
    assert "input_path" not in fields


@pytest.mark.asyncio
async def test_a_steer_errand_carries_the_annotation_too(
    fake_attach: type[_FakeAttach],
) -> None:
    await launch._deliver(
        object(),
        "s",
        SteerErrand(text="more", input_mode="dictated", input_path="provider_stt_elevenlabs"),
    )

    op, fields = fake_attach.created[-1].calls[-1]
    assert op == "steer"
    assert fields["input_mode"] == "dictated"
    assert fields["input_path"] == "provider_stt_elevenlabs"


def test_the_id_attach_rebuild_keeps_the_annotation() -> None:
    """``_fields()`` rebuilds from ``__slots__``; a missed slot drops the field."""
    errand = PromptErrand(text="x", command_id="c", input_mode="typed", input_path="")
    fields = launch._fields(errand)

    assert fields["input_mode"] == "typed"
    assert fields["input_path"] == ""
    assert type(errand)(**fields) == errand


@pytest.mark.asyncio
async def test_continue_command_threads_the_annotation_into_the_errand(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The wake path builds the errand from the command's own fields."""
    from local_operator.mobile import attach_client as mobile_attach
    from local_operator.mobile.types import ContinuationCommand

    captured: dict[str, Any] = {}

    async def fake_engage(*args: Any, **kwargs: Any) -> Any:
        captured["errand"] = args[2]
        return SimpleNamespace(detail="admitted")

    monkeypatch.setattr(launch, "engage_runtime", fake_engage)
    # The engage fake returns without a runtime, so the dial finds no record;
    # the errand has already been captured, which is this test's subject.
    monkeypatch.setattr(mobile_attach, "find_runtime_record", lambda *a, **k: (None, None))

    command = ContinuationCommand.from_json(
        {
            "command_id": "12345678-1234-4678-9234-567812345678",
            "session_id": "s1",
            "text": "hello",
            "input_mode": "dictated",
            "input_path": "provider_stt_radient",
        }
    )
    with pytest.raises(TimeoutError):
        await mobile_attach.continue_command(tmp_path, command)

    errand = captured["errand"]
    assert isinstance(errand, PromptErrand)
    assert errand.text == "hello"
    assert errand.input_mode == "dictated"
    assert errand.input_path == "provider_stt_radient"
