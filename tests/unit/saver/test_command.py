"""Command-surface tests: persistence, availability and exec/spec wiring."""
from __future__ import annotations

from dataclasses import dataclass
from types import SimpleNamespace

import pytest

from local_operator.saver import command


@dataclass
class Result:
    kind: str
    text: str
    style: str


class Transcript:
    def __init__(self) -> None:
        self.rows: list[dict] = []

    async def append_custom(self, kind, details):
        self.rows.append({"custom_type": kind, "details": details})

    def latest_custom(self, kind):
        return next((row for row in reversed(self.rows) if row["custom_type"] == kind), None)


def session(mode="opt_in"):
    return SimpleNamespace(
        routing_settings={"radient": {"saver": {"mode": mode}}},
        radient_saver_choice=None,
        _transcript=Transcript(),
        refresh_frontend_state=lambda: None,
    )


@pytest.mark.asyncio
async def test_command_toggle_status_inherit_and_journal_restore(monkeypatch):
    monkeypatch.setattr(command, "signed_in", lambda: True)
    current = session()
    assert saver(current, "status").text.startswith("Radient Saver: off")
    assert saver(current, "on").text.startswith("Radient Saver is on")
    await __import__("asyncio").sleep(0)
    assert current.radient_saver_choice is True
    assert saver(current, "status").text.startswith("Radient Saver: on")
    assert saver(current, "").text.startswith("Radient Saver is off")
    await __import__("asyncio").sleep(0)
    assert saver(current, "inherit").text.startswith("Radient Saver now follows")
    await __import__("asyncio").sleep(0)
    assert current.radient_saver_choice is None
    await command.set_saver_choice(current, True)
    restored = session()
    restored._transcript = current._transcript
    command.restore_saver_choice(restored)
    assert restored.radient_saver_choice is True


def saver(current, arg):
    return command.saver_slash(current, arg, Result)


def test_signed_out_does_not_mutate(monkeypatch):
    monkeypatch.setattr(command, "signed_in", lambda: False)
    current = session()
    result = saver(current, "on")
    assert "/login radient" in result.text
    assert current._transcript.rows == []
    assert current.radient_saver_choice is None


@pytest.mark.asyncio
async def test_global_gate_explains_why_enable_cannot_run(monkeypatch):
    monkeypatch.setattr(command, "signed_in", lambda: True)
    current = session(mode="off")
    assert "not available" in saver(current, "on").text
    await __import__("asyncio").sleep(0)
    assert current.radient_saver_choice is True


def test_exec_and_sdk_share_the_same_tri_state():
    from local_operator.session.spec import SessionSpec

    assert SessionSpec(radient_saver=True).to_runner_args().saver == "on"
    assert SessionSpec(radient_saver=False).to_runner_args().saver == "off"
    assert SessionSpec().to_runner_args().saver == "inherit"
