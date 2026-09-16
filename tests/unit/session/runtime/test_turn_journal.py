"""The runtime's own account of a death: a turn journal and a boot record.

WHY THESE TESTS EXIST. On 2026-09-15 a fleet outage killed 36 session runtimes
at 19:41 and every one of them recorded the SAME anonymous verdict, because no
artifact on disk was authored by the process that died. The tests below pin the
three things that fix that, and they are deliberately arranged so that each one
fails if the responsible half is removed:

* a killed runtime leaves an OPEN row naming the turn that was in flight — a
  REAL process, SIGKILLed by exact pid, read back from disk by another process;
* the classifier PREFERS the named cause that evidence supports (an escalated
  sweep, an install-window tear, a crash) and falls back to the legacy rungs
  unchanged when the evidence is absent;
* the interruption reaches the RESUMED session's own context exactly once, and
  re-runs nothing.

The inertness half — that none of this moved ``_should_exit`` or a grace
constant — is pinned in ``test_residency_guard.py``.
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path
from typing import Any

import pytest

from local_operator import update
from local_operator.session.attention import _classify_orphaned_run
from local_operator.session.runtime import journal, registry

#: A child that opens a real journal row, records a real tool boundary, writes
#: its boot record, and then blocks. The test SIGKILLs it: nothing in this
#: module's subject runs on the child's exit path, which is the point — the
#: artifact has to stand on its own after a process that recorded nothing at
#: exit.
_CHILD_SCRIPT = """
import sys, time
from pathlib import Path
from local_operator.session.runtime import journal
from local_operator.update import installed_build

directory = Path(sys.argv[1])
session_id = sys.argv[2]
build = installed_build()
journal.write_boot_record(session_id, build)
writer = journal.TurnJournal(directory, session_id, build)
writer.open_turn(command_id="cmd-1")
writer.note_boundary("bash")
print("ready", flush=True)
time.sleep(60)
"""


def _isolated_env(home: Path) -> dict[str, str]:
    """A child environment with every inherited harness variable stripped.

    ``CMUX_*`` because an inherited workspace id lets a test rename the
    operator's real cmux workspaces; ``LOP_*`` because an inherited
    ``LOP_RUNTIME_*``/``LOP_MOBILE_CHILD_*`` pins a spawned runtime's session and
    provider, so a cell would exercise the operator's own session instead of its
    own. Both families are stripped rather than a few names, so a new one cannot
    quietly re-enter.
    """
    env = {k: v for k, v in os.environ.items() if not k.startswith(("CMUX_", "LOP_"))}
    env["HOME"] = str(home)
    env["LOCAL_OPERATOR_CONFIG_DIR"] = str(home / ".local-operator")
    return env


def _child_config_root(tmp_path: Path) -> Path:
    """The root the spawned child writes its records under.

    Read from the SAME environment the child is spawned with rather than from
    this process's ambient config: the two are different roots by design (that
    is what the isolation is for), and a test that read the ambient one would
    assert about a directory nothing ever wrote to.
    """
    return Path(_isolated_env(tmp_path)["LOCAL_OPERATOR_CONFIG_DIR"])


def _session_directory(root: Path, session_id: str) -> Path:
    """``<root>/sessions/<id>``, the layout ``attention._config_root_for`` reads."""
    directory = root / "sessions" / session_id
    directory.mkdir(parents=True, exist_ok=True)
    return directory


def _spawn_mid_turn(directory: Path, session_id: str, tmp_path: Path) -> subprocess.Popen[bytes]:
    """A real process parked with an open journal row.

    Returns the live child; the caller kills it. Nothing here sends a signal to
    anything but this exact pid — a pattern kill on a box holding dozens of live
    agent sessions is the hazard the whole design exists to remove.
    """
    child = subprocess.Popen(
        [sys.executable, "-c", _CHILD_SCRIPT, str(directory), session_id],
        env=_isolated_env(tmp_path),
        stdin=subprocess.DEVNULL,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        start_new_session=True,
    )
    assert child.stdout is not None
    line = child.stdout.readline()
    assert line.strip() == b"ready", (line, child.stderr.read() if child.stderr else b"")
    return child


def _kill(child: subprocess.Popen[bytes]) -> int:
    """SIGKILL one pid this test spawned, and reap it."""
    pid = child.pid
    child.kill()
    child.wait(timeout=10)
    return pid


def _pid_from_child() -> int:
    """A pid that has certainly exited — taken from a child we spawned.

    From the kernel rather than invented: writing a fabricated pid into a row
    would make "the owner is gone" a guess about which numbers a test picked,
    and PID reuse would turn it into a silently live owner.
    """
    child = subprocess.Popen(
        [sys.executable, "-c", "pass"], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL
    )
    child.wait(timeout=30)
    return child.pid


# ---------------------------------------------------------------------------
# The row, as the runtime writes it
# ---------------------------------------------------------------------------


def test_a_turn_opens_a_row_and_closes_it(tmp_path: Path) -> None:
    """The row covers the turn: open at the start, closed with a cause at the end."""
    directory = _session_directory(tmp_path, "sess-open-close")
    build = update.BuildStamp(version="0.55.9", source_ref="abc1234def")
    writer = journal.TurnJournal(directory, "sess-open-close", build)

    writer.open_turn(command_id="cmd-7")
    opened = registry.read_turn_journal(directory)
    assert opened is not None
    assert opened["open"] is True
    assert opened["ended_at"] is None
    assert opened["session_id"] == "sess-open-close"
    assert opened["pid"] == os.getpid()
    assert opened["turn_seq"] == 1
    assert opened["command_id"] == "cmd-7"
    # The generation/install root and the build the process loaded, so a
    # successor can say WHAT was running and not merely that something was.
    assert opened["build"] == {"version": "0.55.9", "source_ref": "abc1234def"}
    assert opened["install_root"] == journal._install_root()

    writer.note_boundary("bash")
    bounded = registry.read_turn_journal(directory)
    assert bounded is not None and bounded["last_boundary"] == "bash"

    writer.close_turn("completed")
    closed = registry.read_turn_journal(directory)
    assert closed is not None
    assert closed["open"] is False
    assert closed["ended_at"] is not None
    assert closed["end_cause"] == "completed"

    # A second turn is a NEW row, not an edit of the closed one.
    writer.open_turn()
    second = registry.read_turn_journal(directory)
    assert second is not None
    assert second["turn_seq"] == 2
    assert second["open"] is True


def test_the_row_survives_a_killed_runtime_and_names_the_turn(tmp_path: Path) -> None:
    """THE CASE STUDY'S SHAPE: a real process, SIGKILLed mid-turn.

    Asserted from a DIFFERENT process than the one that wrote it, because that
    is the position every reader of this artifact is actually in — a successor
    runtime, or an operator's shell the next morning.
    """
    session_id = "sess-killed-midturn"
    directory = _session_directory(tmp_path, session_id)
    child = _spawn_mid_turn(directory, session_id, tmp_path)
    killed_pid = _kill(child)

    row = journal.TurnJournalRow.from_json(registry.read_turn_journal(directory))
    assert row is not None
    assert row.open is True, "a killed runtime's row must still say the turn was in flight"
    assert row.pid == killed_pid
    assert row.session_id == session_id
    assert row.turn_seq == 1
    assert row.last_boundary == "bash"
    assert row.end_cause == ""
    assert row.build.get("version") == update.installed_build().version

    # And it reads as a DEATH, not as a live turn: the liveness test is what
    # keeps the other direction honest.
    assert journal.open_row_after_death(directory) is not None

    # The boot record outlives the process too, and names the parent chain the
    # incident investigation could not reconstruct from the corpse.
    record = journal.read_boot_record(killed_pid, root=_child_config_root(tmp_path))
    assert record is not None
    assert record.session_id == session_id
    assert record.kind == journal.BOOT_RECORD_KIND
    assert record.parent_pid == os.getpid(), "the spawned child's parent is this test process"
    assert record.install_root


def test_a_clean_exit_withdraws_the_boot_record(tmp_path: Path) -> None:
    """A surviving record means "stopped without running its own exit ordering"."""
    session_id = "sess-clean-exit"
    directory = _session_directory(tmp_path, session_id)
    build = update.BuildStamp(version="1.0.0", source_ref="aaa")

    record_path = journal.write_boot_record(session_id, build, root=tmp_path)
    assert record_path.parent.name == "host"
    assert journal.read_boot_record(os.getpid(), root=tmp_path) is not None

    journal.clear_boot_record(root=tmp_path)
    assert journal.read_boot_record(os.getpid(), root=tmp_path) is None
    # The row-side artefact is untouched by the record's withdrawal.
    assert registry.read_turn_journal(directory) is None


def test_the_boot_record_is_readable_by_a_successor_for_another_pid(tmp_path: Path) -> None:
    """The missing link in ``app -> backend -> runtime -> runtimes``."""
    session_id = "sess-successor"
    directory = _session_directory(tmp_path, session_id)
    child = _spawn_mid_turn(directory, session_id, tmp_path)
    killed_pid = _kill(child)

    root = _child_config_root(tmp_path)
    record = journal.read_boot_record(killed_pid, root=root)
    assert record is not None, "the successor has to be able to see that this pid existed"
    assert record.kind == journal.BOOT_RECORD_KIND
    assert record.build_stamp() is not None
    # ``recorded_boot_build`` is the durable second opinion the import classifier
    # falls back to when a torn install makes the live stamp unreadable.
    assert journal.recorded_boot_build(killed_pid, root=root) is not None


# ---------------------------------------------------------------------------
# The classifier's preference order
# ---------------------------------------------------------------------------


def _current_build_fields() -> dict[str, str]:
    """The build this process is running, as a row would record it."""
    stamp = update.installed_build()
    return {"version": stamp.version, "source_ref": stamp.source_ref}


def _write_open_row(
    directory: Path, pid: int, *, build: dict[str, str] | None = None, **fields: Any
) -> journal.TurnJournalRow:
    """An open row whose owner is gone, written the way the runtime writes it.

    The build defaults to the RUNNING install unless a test overrides it: the
    classifier's tear rung compares the row's build against the install on disk,
    so a fabricated build would make every cell read as an install-window tear
    unless the test that means to exercise that says so explicitly.
    """
    payload: dict[str, Any] = {
        "session_id": directory.name,
        "pid": pid,
        "parent_pid": 1,
        "turn_seq": 4,
        "command_id": "cmd-4",
        "started_at": 1000.0,
        "ended_at": None,
        "open": True,
        "end_cause": "",
        "exit_cause": "",
        "still_open_at_exit": False,
        "last_boundary": "bash",
        "build": _current_build_fields() if build is None else build,
        "install_root": "/tmp/install",
        "updated_at": 1001.0,
    }
    payload.update(fields)
    registry.write_turn_journal(directory, payload)
    row = journal.TurnJournalRow.from_json(payload)
    assert row is not None
    return row


def test_the_classifier_names_a_crash_from_the_runtimes_own_row(tmp_path: Path) -> None:
    """No signal recorded and no install movement: the crash, now evidenced."""
    directory = _session_directory(tmp_path, "sess-crash")
    _write_open_row(directory, _pid_from_child())

    kind, cause, reason = _classify_orphaned_run(directory)
    assert (kind, cause) == ("error", "runtime-killed")
    assert "turn 4 in flight" in reason
    assert "last boundary bash" in reason
    # The pid has to be there: the 19:41 investigation's first question was
    # "which process", and the legacy sentence never carried one.
    written = json.loads((directory / registry.TURN_JOURNAL_NAME).read_text())
    assert f"pid {written['pid']}" in reason


def test_the_classifier_names_the_escalated_sweep(tmp_path: Path) -> None:
    """A row that recorded a signal is a stop sweep that REACHED its target.

    This is the shape the fleet could not name: the runtime was asked to leave,
    wrote the signal down, and was killed before its turn ended.
    """
    directory = _session_directory(tmp_path, "sess-signalled")
    _write_open_row(directory, _pid_from_child(), exit_cause="SIGTERM", still_open_at_exit=True)

    kind, cause, reason = _classify_orphaned_run(directory)
    assert (kind, cause) == ("error", "runtime-shutdown")
    assert "SIGTERM" in reason


def test_the_classifier_names_the_install_window_tear(tmp_path: Path, monkeypatch) -> None:
    """An install that moved under the runtime names the tear, not the crash."""
    directory = _session_directory(tmp_path, "sess-torn")
    _write_open_row(
        directory, _pid_from_child(), build={"version": "0.55.9", "source_ref": "beef1234"}
    )
    monkeypatch.setattr(
        update, "installed_build", lambda *_a, **_k: update.BuildStamp("0.56.0", "cafe9999")
    )

    kind, cause, reason = _classify_orphaned_run(directory)
    assert (kind, cause) == ("error", "install-mid-update")
    assert "0.55.9@beef123" in reason and "0.56.0@cafe999" in reason


def test_the_classifier_keeps_the_legacy_paths_without_a_row(tmp_path: Path) -> None:
    """No evidence must behave exactly as it did before this change.

    Two arms, and both are the ones the existing taxonomy was built on: a dead
    record with no journal row still reads ``runtime-killed`` with the record's
    own detail, and a session with nothing at all still reads
    ``CUT_OFF_UNKNOWN`` rather than a guess.
    """
    directory = _session_directory(tmp_path, "sess-legacy")
    assert _classify_orphaned_run(directory) == (
        "error",
        "",
        "the turn was cut off and the cause could not be determined",
    )

    dead = _pid_from_child()
    root = tmp_path
    registry.publish(
        registry.SessionRecord(
            pid=dead,
            kind="daemon",
            session_id="sess-legacy",
            conversation_name="legacy",
            cwd=str(tmp_path),
            model_label="mock",
            control_port=0,
            control_key="k" * 8,
        ),
        root,
    )
    kind, cause, reason = _classify_orphaned_run(directory)
    assert (kind, cause) == ("error", "runtime-killed")
    assert f"pid {dead}" in reason


def test_a_live_owner_with_an_open_row_is_not_a_death(tmp_path: Path) -> None:
    """A turn in flight in a LIVE runtime must never read as an interruption."""
    directory = _session_directory(tmp_path, "sess-live")
    writer = journal.TurnJournal(directory, "sess-live", update.BuildStamp("1.0.0", "aaa"))
    writer.open_turn(command_id="cmd-live")

    assert journal.open_row_after_death(directory) is None
    # And the row itself is there, saying the turn is in flight.
    row = journal.TurnJournalRow.from_json(registry.read_turn_journal(directory))
    assert row is not None and row.open is True


def test_the_death_verdict_never_falls_back_to_no_cause(tmp_path: Path) -> None:
    """``CUT_OFF_UNKNOWN`` is unreachable FROM a row, which is the whole point."""
    directory = _session_directory(tmp_path, "sess-verdict")
    row = _write_open_row(directory, _pid_from_child())
    kind, cause, _reason = journal.death_verdict(row)
    assert kind == "error"
    assert cause in {"runtime-killed", "install-mid-update", "runtime-shutdown"}


# ---------------------------------------------------------------------------
# The interruption reaches the resumed session, once, and re-runs nothing
# ---------------------------------------------------------------------------


def _stream_recording(calls: list[Any]) -> Any:
    """A stream double that records a call and yields nothing.

    An async generator, because that is what ``stream_fn`` requires: a plain
    callable returning ``None`` would type-check as a mistake and would raise
    the first time a turn actually ran — which is precisely the turn this suite
    asserts does NOT happen.
    """

    async def stream(request: Any, *args: Any, **kwargs: Any) -> Any:
        calls.append(request)
        return
        yield  # pragma: no cover — the ``yield`` is what makes this a generator

    return stream


def _boot_session(directory: Path, calls: list[Any]) -> Any:
    """A REAL successor session over the killed session's directory."""
    from local_operator.harness.types import ModelSpec
    from local_operator.session.session import Session
    from local_operator.session.transcript import Transcript

    return Session(
        model=ModelSpec(provider="test", model_id="mock"),
        stream_fn=_stream_recording(calls),
        tools=[],
        transcript=Transcript(directory),
        system_blocks_provider=lambda *_args: [],
    )


def _incidents(directory: Path) -> list[Any]:
    """Every ``session_incident`` row, however it was written."""
    from local_operator.session.transcript import Transcript

    return [
        entry
        for entry in Transcript(directory).entries()
        if entry.payload.get("custom_type") == "session_incident"
    ]


@pytest.mark.asyncio
async def test_the_interruption_reaches_the_resumed_session_once(tmp_path: Path) -> None:
    """The trust half: an agent must be able to tell a cut-off from a completion
    in its OWN state, and must be told exactly once."""
    session_id = "sess-resumed"
    directory = _session_directory(tmp_path, session_id)
    # A durable row of work, so the resumed session has real history to read.
    (directory / "transcript.jsonl").write_text(
        '{"id": "seed", "ts": 1, "type": "message", "payload": {"kind": "message", '
        '"role": "user", "content": [{"type": "text", "text": "start the deploy"}]}}\n',
        encoding="utf-8",
    )
    child = _spawn_mid_turn(directory, session_id, tmp_path)
    killed_pid = _kill(child)

    calls: list[Any] = []
    session = _boot_session(directory, calls)
    try:
        restored = session._restored_cut_off
        assert restored is not None, "the boot must carry the interruption into the session"
        kind, cause, reason, token = restored
        assert kind == "error"
        assert cause == "runtime-killed"
        assert f"pid {killed_pid}" in reason
        assert token == f"turn-journal:{session_id}:{killed_pid}:1"

        # The narration is the EXISTING once-per-run path, so it must land once.
        await session.async_init()
        incidents = _incidents(directory)
        assert len(incidents) == 1, [
            (row.payload.get("details") or {}).get("text") for row in incidents
        ]
        text = str((incidents[0].payload.get("details") or {}).get("raw") or "")
        assert "never ended" in text or "disappeared without exiting cleanly" in text

        # It is in the model's OWN context, through the production converter.
        rendered = _rendered_text(session)
        assert "[session incident]" in rendered
        assert "in flight" in rendered

        # AND IT RERAN NOTHING: no prompt, no provider call, no turn.
        assert calls == [], "the interrupted turn must never be replayed"
        assert session._restored_cut_off is None
    finally:
        await session.dispose()

    # A second open of the same session narrates nothing new.
    calls_again: list[Any] = []
    second = _boot_session(directory, calls_again)
    try:
        await second.async_init()
        assert len(_incidents(directory)) == 1, "re-opening re-narrated the interruption"
        assert calls_again == []
    finally:
        await second.dispose()


def _rendered_text(session: Any) -> str:
    """The next turn's model-visible text, through the production converter."""
    from local_operator.session.session import _default_convert_to_llm

    parts: list[str] = []
    for message in _default_convert_to_llm(list(session._context.messages)):
        for block in message.content:
            text = getattr(block, "text", None)
            if text:
                parts.append(str(text))
    return "\n".join(parts)
