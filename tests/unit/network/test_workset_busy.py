"""A move waits for the session tree to be quiet before it judges the working set (§8.5).

The design RELIES on existing predicates rather than adding a second check: the
runtime's ``is_busy`` is true while ``running_subagents() > 0``, the direct move
refuses busy at the retire and ``--queue``'s driver waits on the same predicate, and
the owner re-runs the working-set gate on the quiescent tree (P3) — for a queued move,
in the resume prepare. These cells pin the two seams that make that true, so a later
change to either cannot silently let a move judge (or, with the transfer half,
snapshot) a tree a subagent is still writing:

* ``is_busy`` counts live subagents;
* a queued move keeps the user's consent ON THE RECORD and the driver's resume invite
  carries it, so the resume prepare re-judges the then-current tree with the digest
  the user actually confirmed (not with none, which would refuse; not with a widened
  scope, which would be silent).
"""

from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest

from local_operator.network import mobility
from local_operator.network import move_queue as mq
from tests.unit.session.runtime.test_serving import make_handle

SID = "9f3ac1e0b7d2"
CONSENT = {"version": 1, "confirm": "sha256:" + "b" * 64, "without": False}


def test_a_live_subagent_keeps_the_session_busy() -> None:
    handle, session = make_handle()
    assert handle.is_busy() is False
    session.running_children = 1

    assert handle.is_busy() is True
    session.running_children = 0
    assert handle.is_busy() is False


def test_a_queued_moves_consent_rides_the_record_into_the_resume_invite(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    record, created = mq.enqueue(
        tmp_path,
        SID,
        to_device="d_b",
        to_name="device-b",
        request_id="rq-1",
        requested_by="d_b",
        working_set=CONSENT,
    )
    assert created and record["working_set"] == CONSENT
    assert (mq.read_record(tmp_path, SID) or {})["working_set"] == CONSENT
    # The safe point was reached (the runtime claimed the pause): the driver copies.
    mq.transition(tmp_path, SID, mq.QUEUE_PHASE_FINISHING, writer="relay")
    mq.transition(tmp_path, SID, mq.QUEUE_PHASE_PAUSED, writer="runtime")
    frames: list[dict[str, Any]] = []

    def ask(self: Any, frame: dict[str, Any], **_kw: Any) -> dict[str, Any]:
        frames.append(frame)
        return {"result": "refused", "code": "busy", "message": "stop here"}

    monkeypatch.setattr(mobility.LinkTransport, "ask", ask)
    monkeypatch.setattr(mobility, "_resolve_typed_peer", lambda *_a: ("d_b", "device-b"))
    monkeypatch.setattr(mobility, "_link_for_move", lambda *_a: SimpleNamespace(device_id="d_b"))
    monkeypatch.setattr(mobility, "_push_move_definitions", lambda *_a: None)
    monkeypatch.setattr(mobility, "reconcile", lambda *_a, **_k: [])
    server = SimpleNamespace(root=tmp_path)

    mq._drive_copy(server, SID)  # type: ignore[arg-type]  # noqa: SLF001

    invite = next(frame for frame in frames if frame.get("phase") == "invite")
    assert invite["resume_queued"] is True
    assert invite["working_set"] == CONSENT


def test_a_record_written_before_the_gate_reads_as_no_consent(tmp_path: Path) -> None:
    """An older relay's record has no key: the resume then refuses an unconfirmed
    working set rather than moving it silently."""
    record, _ = mq.enqueue(
        tmp_path, SID, to_device="d_b", to_name="device-b", request_id="rq", requested_by="d_b"
    )
    assert record["working_set"] is None
    assert mobility._consent_frame(record["working_set"])["confirm"] is None  # noqa: SLF001
