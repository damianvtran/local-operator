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
from local_operator.session.transcript import Transcript
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
        json.dumps({"entry_id": ""}),
        json.dumps({"entry_id": 7}),
        json.dumps({"entry_id": "abc", "message": "hi"}),  # a widened shape
        json.dumps(["abc"]),
        json.dumps({"cut": "abc"}),
        "{not json",
        "  ",
    ],
)
async def test_only_the_exact_payload_names_a_cut(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, args: str
) -> None:
    """Anything that is not exactly ``{"entry_id": <str>}`` means "no cut".

    The strictness is the guard, not decoration: the same slot carries a typed
    fork's trailing text, and a payload a future client widens must refuse to
    cut rather than cut somewhere the caller never named.
    """
    assert _fork_entry_target(args) is None
    session, handle, store = _handle(tmp_path, monkeypatch)
    await _seed(session)
    before = await _parent_bytes(session)

    result = await handle.run_slash_authoritative("fork", args)

    child = Transcript(store / "sessions" / result["data"]["session_id"])
    assert child.path.read_bytes() == before


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
