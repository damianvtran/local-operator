"""The serving host's `/radient` dispatch: only an explicit value may write.

Round 4 of review found the two hosts disagreeing about a whitespace argument:
`rest or "status"` is a TRUTHINESS test, so `/radient saver␣␣` reported in the
app host and WROTE Saver ON here — this host forwards `body.args` verbatim
(`desktop_sessions.py`), so it never strips. The table below is the guard: it
drives the real `ServingSessionHandle._slash_result` against a real
`saver.command.saver_slash` and asserts the JOURNAL outcome for every input,
because the intermediate string is what let the previous version pass.
"""

from __future__ import annotations

import asyncio
from dataclasses import dataclass
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest

from local_operator.saver import command as saver_command
from local_operator.saver.gate import Availability
from local_operator.session.runtime.serving import ServingSessionHandle
from local_operator.session.transcript import Transcript


@dataclass
class SlashResult:
    kind: str
    text: str = ""
    style: str = "info"


def _session(tmp_path: Path) -> Any:
    return SimpleNamespace(
        routing_settings={"radient": {"saver": {"mode": "opt_in"}}},
        radient_saver_choice=None,
        _transcript=Transcript(tmp_path / "journal"),
        _saver_availability=None,
        refresh_frontend_state=lambda: None,
    )


def _journal(tmp_path: Path) -> list[bool | None]:
    return [
        entry.payload["details"]["enabled"]
        for entry in Transcript(tmp_path / "journal").entries()
        if entry.payload.get("custom_type") == "saver_choice_v1"
        or entry.payload.get("custom_type") == saver_command.SAVER_CUSTOM_TYPE
    ]


def _serve(session: Any, command: str, args: str) -> SlashResult:
    fake = SimpleNamespace(_session=session, _notify=lambda: None)
    return asyncio.run(
        ServingSessionHandle._slash_result(
            fake, command, args, SlashResult  # type: ignore[arg-type]
        )
    )


@pytest.fixture(autouse=True)
def _signed_in(monkeypatch: pytest.MonkeyPatch) -> None:
    async def available(*_args: Any, **_kwargs: Any) -> Availability:
        return Availability(signed_in=True, positive_balance=True)

    monkeypatch.setattr(saver_command, "signed_in", lambda: True)
    monkeypatch.setattr(saver_command, "read_availability", available)
    monkeypatch.setattr(saver_command, "refresh_availability", available)


@pytest.mark.parametrize(
    ("command", "args", "expected_rows"),
    [
        pytest.param("radient", "", [], id="radient-bare"),
        pytest.param("radient", " ", [], id="radient-space"),
        pytest.param("radient", "  \t ", [], id="radient-mixed-whitespace"),
        pytest.param("radient", "saver", [], id="radient-saver-bare"),
        pytest.param("radient", "saver ", [], id="radient-saver-trailing-space"),
        pytest.param("radient", "saver  ", [], id="radient-saver-two-spaces"),
        pytest.param("radient", "saver \t", [], id="radient-saver-space-tab"),
        pytest.param("radient", "saver --", [], id="radient-saver-dashes"),
        pytest.param("radient", "saver status", [], id="radient-saver-status"),
        pytest.param("radient", "saver on", [True], id="radient-saver-on"),
        pytest.param("radient", "saver off", [False], id="radient-saver-off"),
        pytest.param("radient", "saver inherit", [None], id="radient-saver-inherit"),
        # The alias keeps its documented bare toggle: typing the whole word IS
        # the statement of intent. Whitespace after it is NOT that statement.
        pytest.param("saver", "", [True], id="saver-bare-toggles"),
        pytest.param("saver", "  ", [], id="saver-whitespace-does-not-toggle"),
    ],
)
def test_the_serving_host_writes_only_for_an_explicit_value(
    tmp_path: Path, command: str, args: str, expected_rows: list[bool | None]
) -> None:
    session = _session(tmp_path)
    _serve(session, command, args)
    assert _journal(tmp_path) == expected_rows, (command, repr(args), _journal(tmp_path))


@pytest.mark.parametrize(
    ("args", "forwarded"),
    [
        pytest.param("saver", "status", id="saver-bare"),
        pytest.param("saver ", "status", id="saver-trailing-space"),
        pytest.param("saver  ", "status", id="saver-two-spaces"),
        pytest.param("saver \t", "status", id="saver-space-tab"),
        pytest.param("saver on", "on", id="saver-on"),
        pytest.param("saver off", "off", id="saver-off"),
    ],
)
def test_the_serving_host_normalises_before_the_handler_sees_it(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, args: str, forwarded: str
) -> None:
    """The host half of the fix, asserted where it happens.

    The handler also refuses to treat whitespace as a toggle (so the rule holds
    for any caller), which means the outcome table alone cannot tell WHICH layer
    caught it — removing this host's strip still passed every outcome case. This
    test watches the argument the host forwards, so the normalisation here is
    covered directly.
    """
    seen: list[str] = []

    async def spy(_session: Any, arg: str, _result: Any) -> SlashResult:
        seen.append(arg)
        return SlashResult(kind="notice", text="stub")

    monkeypatch.setattr(saver_command, "saver_slash", spy)
    _serve(_session(tmp_path), "radient", args)
    assert seen == [forwarded], (args, seen)
