"""The runtime's ``checkpoints_warm`` word: the seam the desktop route rides.

``sessions.checkpoints.warm`` reaches the serving runtime through this internal
shared-slash word (the ``desktop_mcp`` precedent) because provider errands must
run where the session's credentials and errand tier live. Nothing else pins that
word: the route tests fake its far side, and a rename on one end of the wire
would otherwise fail silently at runtime with every unit green.

A REAL session and a REAL ``ServingSessionHandle`` are used; the only doubles
are the stream (never driven) and the session's ``complete_once`` (the provider
call this test must not make).
"""

from __future__ import annotations

import asyncio
import json
from pathlib import Path

import pytest

from local_operator.session import transcript_index as ti
from local_operator.session.runtime.serving import ServingSessionHandle
from local_operator.session.session import Session
from tests.e2e.harness import ScriptedStream
from tests.unit.session.test_checkpoint_naming import _settle
from tests.unit.session.test_session import make_session

pytestmark = pytest.mark.asyncio


def _seed_index(cfg: Path, sid: str, *, user: str, answer: str) -> None:
    checkpoints = [
        ti.Checkpoint(id="u1", kind=ti.KIND_USER, turn=1, ts=1.0, seq=0, text=user, outcome=None),
        ti.Checkpoint(
            id="a1",
            kind=ti.KIND_COMPLETION,
            turn=1,
            ts=2.0,
            seq=2,
            text=answer,
            outcome="complete",
        ),
    ]
    index = ti.TranscriptIndex(
        checkpoints=checkpoints,
        messages=[
            ti.MessageDoc(id="u1", ts=1.0, role="user", text=user, injected=False, seq=0),
            ti.MessageDoc(id="a1", ts=2.0, role="assistant", text=answer, injected=False, seq=2),
        ],
        sig={"size": 2, "mtime": 1.0, "last_id": "a1"},
        coverage={"first_id": "u1", "last_id": "a1", "complete": True},
        naming={"prompt_version": ti.NAMING_PROMPT_VERSION, "items": {}},
        scan=ti.ScanState(rows=2, offset=0, window_offset=0, window_rows=0),
    )
    ti.write_index(cfg, sid, index)


def _handle(tmp_path: Path) -> tuple[Session, ServingSessionHandle, Path]:
    home = tmp_path / "home"
    home.mkdir()
    cfg = tmp_path / "cfg"
    session = make_session(tmp_path, ScriptedStream([]))
    handle = ServingSessionHandle(session, asyncio.get_running_loop(), cwd=str(tmp_path))
    return session, handle, cfg


async def test_the_word_admits_work_and_answers_the_receipt(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("HOME", str(tmp_path / "home"))
    monkeypatch.setenv("LOCAL_OPERATOR_CONFIG_DIR", str(tmp_path / "cfg"))
    session, handle, cfg = _handle(tmp_path)
    sid = session.session_id
    _seed_index(cfg, sid, user="Fix the flaky test", answer="Pinned the race.")
    calls: list[str] = []

    async def complete_once(system: str, prompt: str) -> str:
        calls.append(prompt)
        return "<name>Fix flaky test</name><summary>Pinned the race.</summary>"

    session.complete_once = complete_once  # type: ignore[method-assign]

    result = await handle.run_slash_authoritative(
        "checkpoints_warm", json.dumps({"ids": ["a1"], "limit": None})
    )

    assert result["kind"] == "block"
    assert result["data"] == {"accepted": ["a1"], "pending": ["a1"]}
    await _settle(cfg, sid)
    index = ti.read_index(cfg, sid)
    assert index is not None
    assert index.naming["items"]["u1"]["name"] == "Fix flaky test"
    assert len(calls) == 1


async def test_no_errand_seam_answers_naming_unavailable(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("HOME", str(tmp_path / "home"))
    monkeypatch.setenv("LOCAL_OPERATOR_CONFIG_DIR", str(tmp_path / "cfg"))
    session, handle, cfg = _handle(tmp_path)
    _seed_index(cfg, session.session_id, user="One", answer="A1")
    session.complete_once = None  # type: ignore[method-assign]

    result = await handle.run_slash_authoritative("checkpoints_warm", "{}")

    assert result["kind"] == "error"
    assert result["data"] == {"code": "naming_unavailable"}


async def test_malformed_args_are_tolerated_and_without_a_cache_answer_empty(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("HOME", str(tmp_path / "home"))
    monkeypatch.setenv("LOCAL_OPERATOR_CONFIG_DIR", str(tmp_path / "cfg"))
    session, handle, cfg = _handle(tmp_path)

    async def complete_once(system: str, prompt: str) -> str:  # pragma: no cover
        raise AssertionError("nothing is nameable without an index")

    session.complete_once = complete_once  # type: ignore[method-assign]

    result = await handle.run_slash_authoritative("checkpoints_warm", "not json at all")

    assert result["kind"] == "block"
    assert result["data"] == {"accepted": [], "pending": []}
    await _settle(cfg, session.session_id)
