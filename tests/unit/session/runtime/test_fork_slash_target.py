"""The routed ``fork`` word's CUT POINT: the seam the desktop route rides.

``POST /v1/desktop/sessions/{id}/fork`` reaches the serving runtime through
``route_shared_slash("fork", …)``, and the seam carries only STRINGS — so the
cut point rides the args slot as a JSON object, the way ``/checkpoints_warm``,
``/wake`` and ``/monitor`` carry theirs from the same plane. Nothing else pins
that encoding: the route tests fake its far side, and an encoder that drifted
from this decoder would fail at runtime with every unit green.

A REAL session and a REAL ``ServingSessionHandle`` are used, so the assertions
are about the artifact a fork actually leaves on disk, not about a call shape.
"""

from __future__ import annotations

import asyncio
import json
from pathlib import Path

import pytest

from local_operator.harness.types import Message
from local_operator.session.runtime.serving import (
    ServingSessionHandle,
    _fork_entry_target,
)
from local_operator.session.session import Session
from local_operator.session.transcript import ENTRY_MESSAGE, Transcript
from tests.e2e.harness import ScriptedStream
from tests.unit.session.test_session import MODEL

pytestmark = pytest.mark.asyncio

PARENT_ID = "forkparent01"


def _session(store: Path) -> Session:
    """A session whose transcript sits at ``<store>/sessions/<id>``.

    That depth is load-bearing rather than cosmetic: ``Transcript.fork_snapshot``
    sites the clone at ``directory.parent.parent``, which is the store only at
    the real layout's nesting.
    """
    return Session(
        model=MODEL,
        stream_fn=ScriptedStream([]),
        tools=[],
        transcript=Transcript(store / "sessions" / PARENT_ID),
        system_blocks_provider=lambda: ["stable", "env"],
    )


def _handle(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> tuple[Session, ServingSessionHandle, Path]:
    """A session whose store IS the process's ``config_dir()``.

    The two arms resolve a clone's destination differently — the historical
    no-target arm through ``config_dir()``, a named cut through the session's
    own transcript — so the test sows them where production keeps them equal:
    the real layout is exactly ``<config_dir>/sessions/<id>``.
    """
    store = tmp_path / "cfg"
    monkeypatch.setenv("HOME", str(tmp_path / "home"))
    monkeypatch.setenv("LOCAL_OPERATOR_CONFIG_DIR", str(store))
    session = _session(store)
    return (
        session,
        ServingSessionHandle(session, asyncio.get_running_loop(), cwd=str(tmp_path)),
        store,
    )


async def _seed(session: Session) -> tuple[Message, Message]:
    first = Message.user("keep me")
    second = Message.assistant("kept answer")
    await session._transcript.append_messages([first, second])
    return first, second


def _message_ids(session: Session) -> list[str]:
    """The conversation rows this session has committed, oldest first."""
    return [row.id for row in session._transcript.entries() if row.type == ENTRY_MESSAGE]


async def _parent_bytes(session: Session) -> bytes:
    """The parent's bytes AFTER its own model-selection row is durable.

    ``Session.fork_snapshot`` persists that row first — the admission path
    every fork shares, pre-existing and not this seam's — so the invariant
    asserted here is that the FORK adds nothing else to the parent.
    """
    await session._ensure_selected_model()
    return session._transcript.path.read_bytes()


async def test_a_named_cut_point_forks_the_prefix_and_truncates_the_child(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    session, handle, store = _handle(tmp_path, monkeypatch)
    first, cut_at = await _seed(session)
    dropped = Message.assistant("after the cut")
    await session._transcript.append_message(dropped)
    before = await _parent_bytes(session)

    result = await handle.run_slash_authoritative("fork", json.dumps({"entry_id": cut_at.id}))

    assert result["kind"] == "block"
    assert result["data"]["type"] == "forked"
    # The row the copy actually stopped at rides the answer: equal to the named
    # entry here, and earlier when a cut lands at-or-before an unfinished batch.
    assert result["data"]["cut_entry_id"] == cut_at.id
    child_id = result["data"]["session_id"]
    child = Transcript(store / "sessions" / child_id)
    assert [item.id for item in child.build_llm_history()] == [first.id, cut_at.id]
    # The row is not merely absent from the model's view: it was not copied.
    assert dropped.id.encode() not in child.path.read_bytes()
    assert session._transcript.path.read_bytes() == before


async def test_the_absent_target_is_still_the_whole_conversation_fork(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """``""`` is the desktop route's old call, character for character."""
    session, handle, store = _handle(tmp_path, monkeypatch)
    await _seed(session)
    before = await _parent_bytes(session)

    result = await handle.run_slash_authoritative("fork", "")

    child = Transcript(store / "sessions" / result["data"]["session_id"])
    assert child.path.read_bytes() == before
    assert session._transcript.path.read_bytes() == before


@pytest.mark.parametrize(
    "args",
    [
        "fork from here",  # a typed boot prompt's text, not a payload
        json.dumps("fork from here"),  # JSON, but not an object
        json.dumps(["abc"]),
        "{not json",
        "  ",
    ],
)
async def test_a_payload_that_is_not_an_object_means_no_cut(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, args: str
) -> None:
    """The legacy call, and a typed fork's trailing text: today's whole fork.

    ``/fork <text>`` reaches this seam only from the mesh CLI (the TUI's own
    handler is frontend-local), and its trailing text is a boot prompt — so an
    argument that is not a JSON object must keep meaning exactly what it meant
    before this feature existed.
    """
    assert _fork_entry_target(args) is None
    session, handle, store = _handle(tmp_path, monkeypatch)
    await _seed(session)
    before = await _parent_bytes(session)

    result = await handle.run_slash_authoritative("fork", args)

    child = Transcript(store / "sessions" / result["data"]["session_id"])
    assert child.path.read_bytes() == before


@pytest.mark.parametrize(
    "args",
    [
        json.dumps({"entry_id": ""}),
        json.dumps({"entry_id": "   "}),
        json.dumps({"entry_id": 7}),
        json.dumps({"entry_id": "abc", "message": "hi"}),  # a widened shape
        json.dumps({"cut": "abc"}),  # a key this runtime does not know
    ],
)
async def test_a_malformed_cut_payload_refuses_rather_than_forking_everything(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, args: str
) -> None:
    """Fail CLOSED: an object that is not exactly the cut payload refuses.

    Falling through here would fork the WHOLE conversation — more history than
    the caller asked for, and the failure shape nothing downstream can notice.
    The refusal creates nothing, so a corrected retry is the same as a first
    attempt.
    """
    with pytest.raises(ValueError, match="was not understood"):
        _fork_entry_target(args)
    session, handle, store = _handle(tmp_path, monkeypatch)
    await _seed(session)
    before = await _parent_bytes(session)

    with pytest.raises(ValueError, match="was not understood"):
        await handle.run_slash_authoritative("fork", args)

    assert session._transcript.path.read_bytes() == before
    assert [entry.name for entry in (store / "sessions").iterdir()] == [PARENT_ID]


async def test_a_foreign_entry_id_refuses_rather_than_forking_everything(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    session, handle, store = _handle(tmp_path, monkeypatch)
    await _seed(session)
    before = await _parent_bytes(session)

    with pytest.raises(ValueError, match="not part of this conversation"):
        await handle.run_slash_authoritative(
            "fork", json.dumps({"entry_id": "ffffffffffffffffffffffffffffffff"})
        )

    assert session._transcript.path.read_bytes() == before
    assert [entry.name for entry in (store / "sessions").iterdir()] == [PARENT_ID]


async def test_both_arms_land_the_clone_beside_its_parent(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """ONE placement rule: a clone lands in the session's OWN store.

    The two arms used to disagree — a named cut resolved the store from the
    session's transcript, a ``next_safe`` fork read ``config_dir()``. They agree
    in production (the real layout is exactly ``<config_dir>/sessions/<id>``),
    which is why the split was invisible; for a session served out of ANOTHER
    store the old no-target arm could not even find its own parent (its clone
    was sited in a store that does not hold the transcript), so this pins the
    reconciliation rather than a preference.
    """
    elsewhere = tmp_path / "elsewhere"
    (elsewhere / "sessions").mkdir(parents=True)
    monkeypatch.setenv("HOME", str(tmp_path / "home"))
    monkeypatch.setenv("LOCAL_OPERATOR_CONFIG_DIR", str(elsewhere))
    session = _session(tmp_path / "store")
    handle = ServingSessionHandle(session, asyncio.get_running_loop(), cwd=str(tmp_path))
    await _seed(session)

    named = await handle.run_slash_authoritative(
        "fork", json.dumps({"entry_id": _message_ids(session)[-1]})
    )
    whole = await handle.run_slash_authoritative("fork", "")

    for result in (named, whole):
        child = result["data"]["session_id"]
        assert (tmp_path / "store" / "sessions" / child).is_dir()
    assert list((elsewhere / "sessions").iterdir()) == []


async def test_a_named_cut_leaves_a_pending_next_safe_request_alone(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The two fork actions are independent, and neither silently eats the other.

    A ``next_safe`` request is deferred until a turn boundary, so a user can hold
    one while pointing at a message. Replacing it would drop a request they made
    (``request_fork`` only ever replaces a rival BOUNDARY request, and says so);
    refusing the cut would put a "try again later" in front of the gesture a user
    makes while a fork is already waiting. So: the cut runs now, the pending
    request still fires at its boundary.
    """
    session, handle, store = _handle(tmp_path, monkeypatch)
    await _seed(session)
    completed: list[str] = []
    replaced = session.request_fork(
        tmp_path, message="", on_complete=lambda fork_id, error: completed.append(fork_id)
    )
    assert replaced is False, "no boundary request was pending yet"
    assert session.has_pending_fork()

    result = await handle.run_slash_authoritative(
        "fork", json.dumps({"entry_id": _message_ids(session)[-1]})
    )

    assert result["kind"] == "block"
    assert session.has_pending_fork(), "the cut replaced a request the user made"
    assert completed == [], "the cut ran no deferred clone of its own"
    assert (store / "sessions" / result["data"]["session_id"]).is_dir()
    assert session.cancel_fork()
