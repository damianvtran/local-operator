"""Command-surface tests: the writer, the one resolver and a real resume.

Every test here was rewritten after review round 1, because three of them passed
with the behaviour deleted: the mock transcript answered the way the buggy
`restore_saver_choice` expected, and the gate test ASSERTED the refused-write
row (``/saver on`` under mode ``off`` journaled ``enabled: True`` while the
notice said "not available"). The shapes below make the fake impossible:

* the resume drives a REAL ``Transcript`` on disk, so a row has to survive a
  reopen to restore — deleting the restore fails the test;
* ``refusing`` states are asserted for the ABSENCE of a row, not only for the
  notice text.
"""
from __future__ import annotations

import asyncio
import json
from dataclasses import dataclass
from pathlib import Path
from types import SimpleNamespace

import pytest

from local_operator.saver import command
from local_operator.saver.gate import (
    Availability,
    configured_mode,
    resolve_saver,
    resolve_saver_setting,
)
from local_operator.session.transcript import Transcript


@dataclass
class Result:
    kind: str
    text: str
    style: str


def session(tmp_path: Path, mode: str = "opt_in", *, settings: dict | None = None):
    """A session-shaped object over a real journal directory."""
    routing = settings if settings is not None else {"radient": {"saver": {"mode": mode}}}
    return SimpleNamespace(
        routing_settings=routing,
        radient_saver_choice=None,
        _transcript=Transcript(tmp_path),
        _saver_availability=None,
        # Stamped the way a real session stamps it: the badge reads a cache,
        # never the credential store (see ``cached_signed_in``).
        _saver_signed_in=True,
        refresh_frontend_state=lambda: None,
    )


def rows(tmp_path: Path) -> list[dict]:
    """The custom rows actually on disk, read back through the real reader."""
    return [
        entry.payload
        for entry in Transcript(tmp_path).entries()
        if entry.payload.get("custom_type") == command.SAVER_CUSTOM_TYPE
    ]


@pytest.fixture(autouse=True)
def positive_gate(monkeypatch):
    """Signed in with credit, unless a test says otherwise."""
    monkeypatch.setattr(command, "signed_in", lambda: True)

    async def read() -> Availability | None:
        return Availability(signed_in=True, positive_balance=True)

    monkeypatch.setattr(command, "read_availability", read)


def run(current, arg: str) -> Result:
    return asyncio.run(command.saver_slash(current, arg, Result))


def test_toggle_status_and_inherit_write_one_row_each(tmp_path):
    current = session(tmp_path)
    assert run(current, "status").text.startswith("Radient Saver: off")
    assert run(current, "on").text.startswith("Radient Saver is on")
    assert current.radient_saver_choice is True
    assert run(current, "status").text.startswith("Radient Saver: on")
    assert run(current, "").text.startswith("Radient Saver is off")
    assert current.radient_saver_choice is False
    assert run(current, "inherit").text.startswith("Radient Saver now follows")
    assert current.radient_saver_choice is None
    assert [row["details"]["enabled"] for row in rows(tmp_path)] == [True, False, None]


def test_a_resume_restores_from_a_real_journal_row(tmp_path):
    """QA Q2: the old test passed because its fake returned the row itself.

    Two real transcripts over one directory in different processes' order: the
    writer appends, a FRESH reader is built from disk, and the restore is the
    only thing that can put the value back.
    """
    writer = session(tmp_path)
    assert run(writer, "on").text.startswith("Radient Saver is on")
    assert rows(tmp_path)[0]["details"] == {"enabled": True}

    resumed = SimpleNamespace(radient_saver_choice=None, _transcript=Transcript(tmp_path))
    command.restore_saver_choice(resumed)
    assert resumed.radient_saver_choice is True

    # And the `None` arm is a real durable choice, not "absent".
    assert run(writer, "inherit").text.startswith("Radient Saver now follows")
    resumed_none = SimpleNamespace(radient_saver_choice=True, _transcript=Transcript(tmp_path))
    command.restore_saver_choice(resumed_none)
    assert resumed_none.radient_saver_choice is None


def test_restore_is_deletion_sensitive():
    """The guard itself: a transcript whose details carry no key restores nothing."""
    empty = SimpleNamespace(
        radient_saver_choice="untouched",
        _transcript=SimpleNamespace(latest_custom=lambda _kind: {"other": 1}),
    )
    command.restore_saver_choice(empty)
    assert empty.radient_saver_choice == "untouched"


def test_signed_out_refuses_without_writing(tmp_path, monkeypatch):
    monkeypatch.setattr(command, "signed_in", lambda: False)
    current = session(tmp_path)
    result = run(current, "on")
    assert "/login radient" in result.text
    assert current.radient_saver_choice is None
    assert rows(tmp_path) == []


def test_mode_off_refuses_and_writes_nothing(tmp_path):
    """Review M4: the old test asserted the bad row (`enabled: True`)."""
    current = session(tmp_path, mode="off")
    result = run(current, "on")
    assert "not available" in result.text
    assert current.radient_saver_choice is None
    assert rows(tmp_path) == []


def test_empty_balance_refuses_with_no_credit(tmp_path, monkeypatch):
    """Review M3: `no_credit` was unreachable from the command."""
    async def empty() -> Availability:
        return Availability(signed_in=True, positive_balance=False)

    monkeypatch.setattr(command, "read_availability", empty)
    current = session(tmp_path)
    result = run(current, "on")
    assert "balance is empty" in result.text
    assert rows(tmp_path) == []
    assert current.radient_saver_choice is None


def test_unreadable_balance_refuses_and_says_so(tmp_path, monkeypatch):
    async def unknown() -> None:
        return None

    monkeypatch.setattr(command, "read_availability", unknown)
    current = session(tmp_path)
    result = run(current, "on")
    assert "could not check your Radient balance" in result.text
    assert rows(tmp_path) == []
    assert current.radient_saver_choice is None


def test_off_is_writable_even_when_the_gate_is_shut(tmp_path):
    """An off/inherit write only ever reduces what leaves the machine."""
    current = session(tmp_path, mode="off")
    current.radient_saver_choice = True
    assert run(current, "off").text.startswith("Radient Saver is off")
    assert current.radient_saver_choice is False
    assert [row["details"]["enabled"] for row in rows(tmp_path)] == [False]


def test_signed_out_can_still_read_and_clear(tmp_path, monkeypatch):
    """UX U6: the sign-out guard used to deny the one view and the one word."""
    current = session(tmp_path)
    current.radient_saver_choice = True
    monkeypatch.setattr(command, "signed_in", lambda: False)

    async def signed_out_read() -> Availability:
        return Availability(signed_in=False, positive_balance=False)

    monkeypatch.setattr(command, "read_availability", signed_out_read)
    assert run(current, "status").text.startswith("Radient Saver: off")
    assert run(current, "inherit").text.startswith("Radient Saver now follows")
    assert current.radient_saver_choice is None


def test_the_command_and_the_turn_path_cannot_disagree(tmp_path):
    """Review M1 / QA Q3: one resolver, so both readers name the same inputs.

    The pin is seeded where the LIVE code seeds it (``_saver_agent_pin_value``,
    ``harness/subagent.py``) and the parent value where the stream carries it.
    A reader that goes back to ``_saver_agent_pin`` fails here, which is the
    regression the round-1 finding filed.
    """
    current = session(tmp_path)
    current._saver_agent_pin_value = True
    assert command.saver_state(current).on is True
    assert command.saver_state(current).source == "agent"

    current._saver_agent_pin_value = None
    stream = SimpleNamespace(_saver_parent_value=True)
    current._stream_fn = stream
    assert command.saver_state(current).source == "parent"

    # The turn path's tuple is the same tuple.
    assert command.saver_choices(current) == (None, None, True)

    # An explicit opt-out still wins over both, so the bare toggle inverts the
    # right thing (QA Q3's second half).
    current.radient_saver_choice = False
    assert command.saver_state(current).on is False
    assert command.saver_state(current).source == "session"


def test_the_session_state_never_disagrees_with_the_request_resolver(tmp_path):
    """The two answers are separate ON PURPOSE; with credit they must agree."""
    positive = Availability(signed_in=True, positive_balance=True)
    settings = {"radient": {"saver": {"mode": "opt_in"}}}
    for choice in (None, True, False):
        for pin in (None, True, False):
            for parent in (None, True, False):
                session_state = resolve_saver_setting(
                    settings,
                    signed_in=True,
                    session_choice=choice,
                    agent_pin=pin,
                    parent_value=parent,
                )
                request = resolve_saver(
                    settings,
                    positive,
                    session_choice=choice,
                    agent_pin=pin,
                    parent_value=parent,
                )
                assert session_state == request, (choice, pin, parent)


def test_exec_and_sdk_share_the_same_tri_state():
    from local_operator.session.spec import SessionSpec

    assert SessionSpec(radient_saver=True).to_runner_args().saver == "on"
    assert SessionSpec(radient_saver=False).to_runner_args().saver == "off"
    assert SessionSpec().to_runner_args().saver == "inherit"


def test_exec_startup_refuses_and_reports(tmp_path, monkeypatch, capsys):
    """``--saver on`` obeys the same writer as the slash command (review M4)."""
    from local_operator.exec_startup import apply_saver_startup

    async def signed_out() -> Availability:
        return Availability(signed_in=False, positive_balance=False)

    monkeypatch.setattr(command, "read_availability", signed_out)
    monkeypatch.setattr(command, "signed_in", lambda: False)
    current = session(tmp_path)
    asyncio.run(apply_saver_startup(current, SimpleNamespace(saver="on")))
    assert current.radient_saver_choice is None
    assert rows(tmp_path) == []
    assert "/login radient" in capsys.readouterr().err


def test_exec_startup_says_which_refusal_it_was(tmp_path, monkeypatch, capsys):
    """A blip is not an empty account: the copy must not say "add credit"."""
    from local_operator.exec_startup import apply_saver_startup

    async def unknown() -> None:
        return None

    monkeypatch.setattr(command, "read_availability", unknown)
    current = session(tmp_path)
    asyncio.run(apply_saver_startup(current, SimpleNamespace(saver="on")))
    assert current.radient_saver_choice is None
    assert "could not check your Radient balance" in capsys.readouterr().err


def test_exec_startup_writes_before_the_first_turn(tmp_path):
    """Review M2: the write is awaited, so nothing can read the old state."""
    from local_operator.exec_startup import apply_saver_startup

    current = session(tmp_path)
    asyncio.run(apply_saver_startup(current, SimpleNamespace(saver="on")))
    assert current.radient_saver_choice is True
    assert rows(tmp_path) == [{"custom_type": command.SAVER_CUSTOM_TYPE, "details": {"enabled": True}}]


def test_saver_argv_keeps_a_prompt_that_follows_the_flag():
    """Review minor: ``lop exec --saver "do X"`` lost its prompt."""
    from local_operator.exec_startup import fold_saver_argv

    # Every occurrence becomes explicit, so argparse can never read the prompt
    # as the flag's value (the defect this fold exists for).
    assert fold_saver_argv(["exec", "--saver", "do X"]) == ["exec", "--saver=on", "do X"]
    assert fold_saver_argv(["exec", "--saver", "on", "do X"]) == [
        "exec",
        "--saver=on",
        "do X",
    ]
    assert fold_saver_argv(["exec", "--saver", "inherit"]) == ["exec", "--saver=inherit"]
    assert fold_saver_argv(["exec", "--saver", "=wat"]) == ["exec", "--saver=on", "=wat"]

    # And the parsed result says so: the prompt survives, the flag is ON.
    from local_operator.cli import build_cli_parser

    parsed = build_cli_parser().parse_args(fold_saver_argv(["exec", "--saver", "do X"]))
    assert (parsed.command, parsed.saver) == ("do X", "on")
    parsed = build_cli_parser().parse_args(
        fold_saver_argv(["exec", "--saver=off", "do X"])
    )
    assert (parsed.command, parsed.saver) == ("do X", "off")


def test_mode_reading_is_shared_with_the_gate(tmp_path):
    """A settings shape the command reads is the one the gate reads."""
    current = session(tmp_path, settings={"radient": {"saver": {"mode": "opt_in"}}})
    assert configured_mode(command.session_settings(current)) == "opt_in"


def test_no_stray_state_field_survives_in_the_tree():
    """Review m1/M1: the dead names must not come back.

    ``_saver_agent_pin`` (nothing assigned it) and ``_saver_session_choice``
    (nothing set it) each had a live twin, which is how two readers came to
    answer one question differently. The live names are
    ``_saver_agent_pin_value`` and ``radient_saver_choice``.
    """
    import re

    root = Path(__file__).resolve().parents[3]
    # Dotted, so the module that DOCUMENTS the removed names is not a hit.
    dead = re.compile(r"\._saver_agent_pin(?![_a-z])|\._saver_session_choice")
    offenders = [
        f"{path.relative_to(root)}"
        for path in (root / "local_operator").rglob("*.py")
        if dead.search(path.read_text(encoding="utf-8"))
    ]
    assert offenders == []


def test_the_ledger_row_and_the_catalogue_agree(tmp_path):
    """The row's digest is the en catalogue's, so a partial edit cannot ship."""
    import hashlib

    root = Path(__file__).resolve().parents[3]
    digest = hashlib.sha256(
        (root / "local_operator/i18n/catalogues/en/wire.radient_saver.json").read_bytes()
    ).hexdigest()
    ledger = json.loads((root / "i18n/ledger.json").read_text(encoding="utf-8"))
    rows_for_namespace = [
        entry for entry in ledger["entries"] if entry["namespace"] == "wire.radient_saver"
    ]
    assert rows_for_namespace
    assert {entry["source_sha256"] for entry in rows_for_namespace} == {digest}
