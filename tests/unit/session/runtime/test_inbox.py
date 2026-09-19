"""The cold-session inbox: spooling, ordering, and what a crash costs.

The guarantee this file pins down is an ORDERING one, and it is structural
rather than timed: ``process.py`` drains the spool after the session exists and
before the control socket listens, so a message written while the session was
cold is delivered ahead of anything a socket client could send. The test for
that lives with the drain (``test_drain_precedes_the_socket``); the rest here
cover the file format, concurrency, and the crash contract.
"""

from __future__ import annotations

import json
import os
import threading
from pathlib import Path

import pytest

from local_operator.harness.types import StreamEndEvent
from local_operator.session.protocol import RuntimeLocality
from local_operator.session.runtime.inbox import (
    MAX_INBOX_ROWS,
    SOURCE_PEER,
    SOURCE_USER,
    InboxLine,
    append_inbox,
    drain_inbox,
    inbox_path,
    peek_inbox,
    withdraw_inbox,
)
from local_operator.session.transcript import TRANSCRIPT_FILENAME


def _line(
    text: str,
    *,
    source: str = "",
    command_id: str = "",
    wake: bool = False,
) -> InboxLine:
    return InboxLine(
        text=text,
        sender={"pid": 1, "conversation_name": "peer"},
        source=source,
        command_id=command_id,
        wake=wake,
    )


def test_append_then_drain_preserves_write_order(tmp_path: Path) -> None:
    for index in range(5):
        assert append_inbox(tmp_path, _line(f"note {index}")) is True

    drained = drain_inbox(tmp_path)
    assert [line.text for line in drained] == [f"note {i}" for i in range(5)]
    assert drained[0].sender["conversation_name"] == "peer"
    # Consumed: a second open must not re-deliver them.
    assert drain_inbox(tmp_path) == []


def test_peek_does_not_consume(tmp_path: Path) -> None:
    """The cold viewer reads the spool without stealing the runtime's work."""
    append_inbox(tmp_path, _line("pending"))

    assert [line.text for line in peek_inbox(tmp_path)] == ["pending"]
    assert [line.text for line in peek_inbox(tmp_path)] == ["pending"]
    assert [line.text for line in drain_inbox(tmp_path)] == ["pending"]


def test_missing_and_empty_spools_are_not_errors(tmp_path: Path) -> None:
    assert drain_inbox(tmp_path) == []
    assert peek_inbox(tmp_path) == []
    inbox_path(tmp_path).write_text("", encoding="utf-8")
    assert drain_inbox(tmp_path) == []


def test_a_torn_final_line_is_skipped_not_fatal(tmp_path: Path) -> None:
    """A writer killed mid-write must not make the whole spool unreadable."""
    append_inbox(tmp_path, _line("intact"))
    with open(inbox_path(tmp_path), "ab") as handle:
        handle.write(b'{"text": "half a row')  # no newline, no closing brace

    assert [line.text for line in drain_inbox(tmp_path)] == ["intact"]


def test_concurrent_writers_never_interleave_within_a_line(tmp_path: Path) -> None:
    """O_APPEND + one write() per row is what keeps rows whole.

    Two peers writing to the same cold session in the same instant is the
    contended case; the requirement is not that they be ordered against each
    other but that neither row is corrupted by the other.
    """
    bodies = [f"peer-{index}-{'x' * 200}" for index in range(20)]

    def write(text: str) -> None:
        append_inbox(tmp_path, _line(text))

    threads = [threading.Thread(target=write, args=(body,)) for body in bodies]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join()

    raw = inbox_path(tmp_path).read_bytes().decode()
    # Every line parses: no row was cut in half by another writer.
    parsed = [json.loads(row) for row in raw.splitlines() if row.strip()]
    assert len(parsed) == len(bodies)
    assert {row["text"] for row in parsed} == set(bodies)


def test_the_spool_refuses_to_grow_without_bound(tmp_path: Path) -> None:
    """A runaway producer must not create a file the next open cannot replay."""
    for index in range(MAX_INBOX_ROWS):
        assert append_inbox(tmp_path, _line(f"row {index}")) is True
    assert append_inbox(tmp_path, _line("one too many")) is False

    assert len(drain_inbox(tmp_path)) == MAX_INBOX_ROWS


def test_a_crash_between_read_and_delivery_redelivers_rather_than_drops(
    tmp_path: Path,
) -> None:
    """The crash contract is at-least-once, and this is the direction chosen.

    A runtime killed after ``drain_inbox`` returned but before it delivered
    loses those messages; one killed before the truncate re-delivers them. The
    second is the recoverable failure, so the implementation truncates AFTER
    reading. Simulated by reading the spool without going through the drain,
    which is exactly the state a process killed mid-drain leaves behind.
    """
    append_inbox(tmp_path, _line("must survive"))

    # A crash before the truncate: the file is still whole on disk.
    contents = inbox_path(tmp_path).read_bytes()
    assert b"must survive" in contents

    # The next runtime to open the session drains it successfully.
    assert [line.text for line in drain_inbox(tmp_path)] == ["must survive"]


def test_the_spool_is_owner_only(tmp_path: Path) -> None:
    """It holds message bodies, so it gets the same 0600 the records get."""
    append_inbox(tmp_path, _line("private"))
    assert (os.stat(inbox_path(tmp_path)).st_mode & 0o777) == 0o600


def test_the_inbox_counts_as_content_for_retention(tmp_path: Path) -> None:
    """A spooled message must survive the junk reap.

    ``retention`` treats any file that is not a declared sidecar as content, so
    this is a guard on ``inbox.jsonl`` NOT being added to ``_SIDECAR_NAMES``
    later: doing so would let the sweep delete a session directory holding a
    message the user has never seen.
    """
    from local_operator.session.retention import _SIDECAR_NAMES
    from local_operator.session.runtime.inbox import INBOX_NAME

    assert INBOX_NAME not in _SIDECAR_NAMES


def test_the_drain_is_wired_before_the_socket_starts_listening() -> None:
    """THE ordering guarantee, asserted against the source that provides it.

    ``process.amain`` must drain the spool BEFORE ``RuntimeServer`` begins
    listening. That ordering is the entire reason a spooled message cannot be
    interleaved with an errand a client sends: while the drain runs there is no
    socket to send one on.

    Asserted structurally — on the order of the two statements in the source —
    because there is no runtime observable that distinguishes "drained first"
    from "drained fast enough", and a timing test would be measuring luck. If
    someone moves the drain below the server start, the ordering silently
    becomes a race and every functional test still passes.

    PARSED, NOT SUBSTRING-MATCHED. This used to locate the listener with
    ``source.index("start_in_process()")``, and when the daemon's serving plane
    moved to ``start()`` the only occurrence of that string left in ``amain`` was
    inside the COMMENT explaining the move — so the assertion kept passing
    against prose, would have failed only if somebody reworded the paragraph,
    and could reject correct code placed between the two. A paragraph is not a
    statement; this resolves the calls.
    """
    import ast
    import inspect
    import textwrap

    from local_operator.session.runtime import process

    tree = ast.parse(textwrap.dedent(inspect.getsource(process.amain)))

    def own_body(node: ast.AST) -> list[ast.AST]:
        """Every node in ``amain``'s OWN body — nested scopes excluded.

        A nested ``def`` is not part of the statement order this assertion is
        about, so a call inside one must not be able to satisfy it (review round
        2, NIT-2).
        """
        out: list[ast.AST] = []
        stack = list(ast.iter_child_nodes(node))
        while stack:
            child = stack.pop()
            out.append(child)
            if isinstance(child, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef, ast.Lambda)):
                continue
            stack.extend(ast.iter_child_nodes(child))
        return out

    def runtime_name() -> str:
        """The local the runtime is bound to — ``runtime`` today, whatever after.

        Resolved from the construction rather than hard-coded, so renaming the
        local cannot fail a valid ordering (review round 2, NIT-2).
        """
        for node in own_body(tree.body[0]):
            if (
                isinstance(node, ast.Assign)
                and len(node.targets) == 1
                and isinstance(node.targets[0], ast.Name)
                and isinstance(node.value, ast.Call)
                and isinstance(node.value.func, ast.Name)
                and node.value.func.id == "RuntimeServer"
            ):
                return node.targets[0].id
        raise AssertionError("amain no longer constructs a RuntimeServer")

    def call_line(
        *, function: str | None = None, method: str | None = None, receiver: str | None = None
    ) -> int:
        """The earliest line calling ``function(...)`` or ``<receiver>.method(...)``."""
        found: list[int] = []
        for node in own_body(tree.body[0]):
            if not isinstance(node, ast.Call):
                continue
            called = node.func
            if function is not None and isinstance(called, ast.Name) and called.id == function:
                found.append(node.lineno)
            if (
                method is not None
                and isinstance(called, ast.Attribute)
                and called.attr == method
                and isinstance(called.value, ast.Name)
                and called.value.id == receiver
            ):
                found.append(node.lineno)
        assert found, f"amain no longer calls {function or method}"
        return min(found)

    drain_at = call_line(function="_drain_inbox_into")
    listen_at = call_line(method="start", receiver=runtime_name())
    assert drain_at < listen_at, (
        "the inbox drain must run before the control socket listens; "
        "moving it after turns the delivery guarantee into a race"
    )


def test_a_spool_for_an_unengaged_session_is_preserved_for_the_first_turn(
    tmp_path: Path,
) -> None:
    """THE NEW RULE, boot-drain half: an unengaged session does NOT consume its
    spool.

    The spool can hold rows written by a sender on an OLDER build (one whose
    record read is absent-as-``True``) or from before any record existed, and
    the boot drain runs before the socket listens — so draining here would put
    a peer row at the HEAD of a conversation its owner has never typed in,
    which is the reported symptom. The rows stay for
    ``Session._drain_spooled_peer_inbox``, which runs inside the first real
    turn, once the session IS engaged: nothing is lost, and nothing opens the
    history.

    "No durable history" is the whole test: a session directory with a spool
    and no transcript file at all (the fresh ``/new`` this gate exists for).
    """
    import asyncio

    from local_operator.session.runtime.process import _drain_inbox_into

    session_dir = tmp_path / "sessions" / "freshsess"
    session_dir.mkdir(parents=True)
    assert append_inbox(session_dir, _line("held until you start"))

    class _Transcript:
        directory = session_dir

    class _Session:
        transcript = _Transcript()

    class _Handle:
        def __init__(self) -> None:
            self._session = _Session()
            self.received: list[tuple[str, str, bool]] = []

        async def receive_peer_message(self, text, *, mode, wake, sender=None):
            self.received.append((text, mode, wake))
            return "ok"

    handle = _Handle()
    assert asyncio.run(_drain_inbox_into(handle)) == 0
    assert handle.received == []
    # NOT consumed: the rows are still there for the first real turn.
    assert [line.text for line in peek_inbox(session_dir)] == ["held until you start"]


def test_a_spool_for_an_engaged_session_drains_on_start(tmp_path: Path) -> None:
    """The unchanged half: a session with durable history drains at boot exactly
    as it always did — including the handover case, where a draining runtime
    spooled a message for the successor that is about to start here."""
    import asyncio

    from local_operator.session.runtime.process import _drain_inbox_into

    session_dir = tmp_path / "sessions" / "usedsess"
    session_dir.mkdir(parents=True)
    (session_dir / TRANSCRIPT_FILENAME).write_text(
        json.dumps(
            {
                "id": "h1",
                "ts": 1,
                "type": "message",
                "payload": {"kind": "message", "role": "user", "content": []},
            }
        )
        + "\n",
        encoding="utf-8",
    )
    assert append_inbox(session_dir, _line("hello again"))

    class _Transcript:
        directory = session_dir

    class _Session:
        transcript = _Transcript()

    class _Handle:
        def __init__(self) -> None:
            self._session = _Session()
            self.received: list[tuple[str, str, bool]] = []

        async def receive_peer_message(self, text, *, mode, wake, sender=None):
            self.received.append((text, mode, wake))
            return "ok"

    handle = _Handle()
    delivered = asyncio.run(_drain_inbox_into(handle))

    assert delivered == 1
    # Drained as a quiet mailbox note (never a wake), matching the cold path.
    assert handle.received == [("hello again", "mailbox", False)]
    # And the spool is consumed, not peeked.
    assert drain_inbox(session_dir) == []


def test_the_drain_reads_a_property_the_session_exposes(tmp_path: Path) -> None:
    """Round 2 (U5): the drain read ``session.transcript`` while ``Session``
    only exposed ``_transcript``, so it bailed before calling ``drain_inbox``
    and a cold session's spooled mail silently never arrived.

    Asserted against a REAL session rather than the source text: the failure
    was a missing attribute, and the only test that can catch it is one that
    hands the drain a genuine ``Session`` and a non-empty spool.
    """
    import asyncio
    from typing import Any

    from local_operator.session.runtime.inbox import InboxLine, append_inbox
    from local_operator.session.runtime.process import _drain_inbox_into
    from local_operator.session.transcript import Transcript

    class _Handle:
        def __init__(self, session: Any) -> None:
            self._session = session
            self.received: list[str] = []

        async def receive_peer_message(self, text, *, mode, wake, sender=None):
            self.received.append(text)
            return "ok"

    transcript = Transcript(tmp_path / "sessions" / "inboxsess01")
    transcript.directory.mkdir(parents=True, exist_ok=True)

    class _Session:
        # Runtime role (SessionProtocol). This fake stands in for an OWNER:
        # it carries no attached runtime, which is what the absent legacy
        # `is_remote` meant.
        owns_runtime = True
        outcome_is_synchronous = True
        runtime_locality: RuntimeLocality = "this-process"

        # ``SessionProtocol.credential_op``: the REAL verb table against a
        # memory-only store (the ``test_app_pilot.FakeSession`` pattern), so a
        # credential probe of this double answers the way the owner session it
        # stands in for does instead of silently refusing — a double that
        # swallows the verb is how #891 passed review on an unreachable path.
        @property
        def variables(self) -> Any:
            store = getattr(self, "_variables", None)
            if store is None:
                from local_operator.variables import VariableStore

                store = self._variables = VariableStore(cwd="/tmp", env={})
            return store

        async def credential_op(
            self, action: str, key: str = "", value: str = ""
        ) -> dict[str, Any]:
            from local_operator.session.credential_ops import run_credential_verb

            return await run_credential_verb(
                self.variables, getattr(self, "journal_credential_change", None), action, key, value
            )

    # A bare stand-in would re-create the defect's blind spot; the point is
    # that the PRODUCTION attribute name resolves. Use the real Session's
    # property by construction: build a real session is heavy here, so assert
    # the property exists on the class and mirror its contract.
    from local_operator.session.session import Session

    assert isinstance(Session.transcript, property), (
        "Session must expose `transcript` as a property; the inbox drain and "
        "the gate-timeout writer both resolve it by that name"
    )

    append_inbox(
        transcript.directory,
        InboxLine(text="a quiet note", sender={"name": "peer"}),
    )
    # The session has run a turn, so it is a recipient at boot: the drain is
    # what serves it. Written to disk because that is where the gate reads the
    # engagement signal from.
    (transcript.directory / TRANSCRIPT_FILENAME).write_text(
        json.dumps(
            {
                "id": "h1",
                "ts": 1,
                "type": "message",
                "payload": {"kind": "message", "role": "user", "content": []},
            }
        )
        + "\n",
        encoding="utf-8",
    )
    handle = _Handle(_Session())
    handle._session = type("S", (), {"transcript": transcript})()
    delivered = asyncio.run(_drain_inbox_into(handle))
    assert delivered == 1
    assert handle.received == ["a quiet note"]
    assert not (transcript.directory / "inbox.jsonl").read_bytes()


@pytest.mark.asyncio
async def test_a_twice_spooled_owner_row_steers_once_inside_the_first_turn(tmp_path: Path) -> None:
    """The mid-turn arm's own repeat, on a REAL session (agent review round 1, R3).

    ``process._drain_inbox_into`` answers a repeated owner row from the durable
    index, but the mid-turn twin steers instead, and a queued steer reaches the
    index only when the correction is drained at a later tool boundary — so two
    rows carrying one ``command_id`` in ONE batch would steer the user's text
    twice. The batch-local seen-set closes exactly that window; this cell drives
    the real ``Session._drain_spooled_peer_inbox`` over a real spool and asserts
    one queued correction rather than two.
    """
    from tests.unit.session.test_session import ScriptedStream, make_session

    session_dir = tmp_path / "sess"
    session_dir.mkdir(parents=True, exist_ok=True)
    (session_dir / TRANSCRIPT_FILENAME).write_text(
        json.dumps(
            {
                "id": "h1",
                "ts": 1,
                "type": "message",
                "payload": {"kind": "message", "role": "user", "content": []},
            }
        )
        + "\n",
        encoding="utf-8",
    )
    for _ in range(2):
        assert append_inbox(
            session_dir,
            _line("deploy the fix", source=SOURCE_USER, command_id="p" * 8, wake=True),
        )

    session = make_session(tmp_path, ScriptedStream([[StreamEndEvent(stop_reason="stop")]]))
    await session._drain_spooled_peer_inbox()

    queued = session.queued_steering()
    assert len(queued) == 1, [getattr(item, "id", "") for item in queued]


def test_a_recall_marker_withholds_its_row_from_both_readers(tmp_path: Path) -> None:
    """The recall is an append-only marker, and both readers honour it.

    Rewriting the spool instead — the obvious shape — cannot be made safe here:
    ``append_inbox`` proceeds unlocked when it cannot take the lock, so a
    concurrent append is either cut by an in-place truncate or orphaned by a
    staged replace (measured: 14 of 42 acked rows lost with the staged shape, 1
    of 241 and 8 of 488 with the truncate, in the PR thread). So the withdrawal
    appends ``SOURCE_RECALL`` and nothing is rewritten; the marker and the row it
    names both leave with the batch.
    """
    assert append_inbox(
        tmp_path, _line("deploy the fix", source=SOURCE_USER, command_id="p" * 8, wake=True)
    )
    assert append_inbox(tmp_path, _line("fyi from a peer"))

    assert withdraw_inbox(tmp_path, "p" * 8) is True

    # READERS: the recalled row is not deliverable, and neither is the marker.
    assert [line.text for line in peek_inbox(tmp_path)] == ["fyi from a peer"]
    assert [line.text for line in drain_inbox(tmp_path)] == ["fyi from a peer"]
    # CONSUMED TOGETHER: the batch took the marker with it, so nothing is left
    # asserting a recall whose message can no longer arrive.
    assert peek_inbox(tmp_path) == []


def test_a_recall_of_a_row_that_is_gone_is_refused(tmp_path: Path) -> None:
    """A drained row cannot be recalled, and the caller is told so."""
    assert append_inbox(tmp_path, _line("deploy the fix", source=SOURCE_USER, command_id="p" * 8))
    assert drain_inbox(tmp_path) != []

    assert withdraw_inbox(tmp_path, "p" * 8) is False
    assert withdraw_inbox(tmp_path, "") is False


def test_a_recall_never_touches_a_peer_row_with_the_same_id(tmp_path: Path) -> None:
    """Only the OWNER's rows carry a recallable identity."""
    assert append_inbox(
        tmp_path,
        InboxLine(
            text="peer words",
            sender={"pid": 2},
            source=SOURCE_PEER,
            command_id="p" * 8,
        ),
    )

    assert withdraw_inbox(tmp_path, "p" * 8) is False
    assert [line.text for line in drain_inbox(tmp_path)] == ["peer words"]


def test_a_marker_appended_while_the_drain_reads_still_withholds_its_row(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """QA Q-1 (round 3): the recall must be honoured by the batch being drained.

    The receipt says the message was taken back, so the drain that is *reading*
    the batch while the marker lands must not deliver it. ``_parse`` is hooked to
    append the marker on its first call, which is exactly that interleave made
    deterministic: without the drain's late re-read the row is in the list it had
    already built, and the user is told a recall worked while their message runs.
    Measured by QA at 4/120 trials naturally, 0/120 after this re-read.
    """
    import local_operator.session.runtime.inbox as inbox_mod

    assert append_inbox(tmp_path, _line("deploy the fix", source=SOURCE_USER, command_id="q" * 8))

    real_parse = inbox_mod._parse
    calls = {"n": 0}

    def parse_then_recall(raw: bytes):
        calls["n"] += 1
        if calls["n"] == 1:
            # The concurrent recall: its marker is written after this batch's
            # first read and before the decision.
            assert withdraw_inbox(tmp_path, "q" * 8) is True
        return real_parse(raw)

    monkeypatch.setattr(inbox_mod, "_parse", parse_then_recall)
    delivered = drain_inbox(tmp_path)

    assert delivered == [], [line.text for line in delivered]
    assert peek_inbox(tmp_path) == []


def test_a_recall_that_loses_the_race_answers_so(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The verify is what makes "taken back" true — and what refuses to say it.

    When the drain's batch has the row by the time the marker lands, the append
    is too late and the answer must be ``False`` — the caller renders "the next
    runtime already has that message", which is then the truth. ``_read_all`` is
    hooked so the VERIFY sees the batch already consumed, which is the interleave
    the lock cannot always exclude (a third holder).
    """
    import local_operator.session.runtime.inbox as inbox_mod

    assert append_inbox(tmp_path, _line("deploy the fix", source=SOURCE_USER, command_id="w" * 8))

    real_read_all = inbox_mod._read_all
    state = {"calls": 0}

    def read_all(fd: int) -> bytes:
        state["calls"] += 1
        if state["calls"] == 2:
            # Between the peek and the verify, a drain consumed the batch.
            os.ftruncate(fd, 0)
        return real_read_all(fd)

    monkeypatch.setattr(inbox_mod, "_read_all", read_all)

    assert withdraw_inbox(tmp_path, "w" * 8) is False
