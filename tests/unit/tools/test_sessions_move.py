"""A session queues a move of its OWN working directory and nothing else moves.

Slice 1 is the durable intent plus the tool surface: the call returns
``queued``, the effective cwd is untouched, and every other address is refused
with a technical code. The lifecycle (retire, successor boot, confirmation) is
slice 2 and is deliberately absent here.
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any

import pytest

from local_operator.harness.types import ToolContext
from local_operator.session import cwd_move
from local_operator.tools.builtin import _sessions_tier, execute_sessions

OWN = "own0000000001"


def _ctx(
    tmp_path: Path,
    *,
    session_id: str = OWN,
    job_id: str | None = None,
) -> ToolContext:
    return ToolContext(
        cwd=str(tmp_path),
        session_id=session_id,
        session_dir=str(tmp_path / "sessions" / OWN),
        job_id=job_id,
    )


async def _move(context: ToolContext, **args: object):
    return await execute_sessions("t", {"op": "move", **args}, None, None, context)


def _code(result) -> str:
    assert result.is_error, result.text
    return str((result.details or {}).get("code") or "")


def test_move_is_a_write_tier_op() -> None:
    assert _sessions_tier({"op": "move"}) == "write"


@pytest.mark.asyncio
async def test_a_queued_move_returns_queued_and_never_changes_the_cwd(tmp_path: Path) -> None:
    destination = tmp_path / "elsewhere"
    destination.mkdir()
    context = _ctx(tmp_path)
    result = await _move(context, cwd=str(destination))
    assert not result.is_error, result.text
    details = result.details or {}
    assert details["stage"] == cwd_move.STAGE_QUEUED
    assert details["session_id"] == OWN
    assert details["requested_cwd"] == str(destination.resolve())
    assert details["effective_cwd"] == str(tmp_path), "the caller's cwd is untouched"
    assert details["request_id"]
    assert context.cwd == str(tmp_path), "ToolContext.cwd is never mutated in place"
    stored = cwd_move.read_move_intent(tmp_path, OWN)
    assert stored is not None, "the intent is durable"
    assert stored.stage == cwd_move.STAGE_QUEUED
    assert stored.destination_cwd == str(destination.resolve())


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "args, context_over, code",
    [
        ({"session": "ffffffffffff"}, {}, "sessions_move_foreign_session"),
        ({"pid": os.getpid() + 1}, {}, "sessions_move_pid_not_self"),
        ({"peer": "cloud-node"}, {}, "sessions_move_peer_unsupported"),
        ({"target": "other"}, {}, "sessions_move_target_unsupported"),
        ({"cwd": "x"}, {"session_id": ""}, "sessions_move_no_owner_identity"),
        ({"cwd": "x"}, {"job_id": "job-1"}, "sessions_move_child_has_no_runtime"),
    ],
    ids=["foreign", "pid", "peer", "target", "no-identity", "child"],
)
async def test_every_other_address_is_refused_by_code(
    tmp_path: Path, args: dict[str, Any], context_over: dict[str, Any], code: str
) -> None:
    (tmp_path / "x").mkdir()
    result = await _move(_ctx(tmp_path, **context_over), **args)
    assert _code(result) == code, result.text
    assert cwd_move.read_move_intent(tmp_path, OWN) is None, "a refusal writes no intent"


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "make_destination, code",
    [
        (lambda root: root / "missing", "sessions_move_destination_missing"),
        (
            lambda root: (root / "a-file").write_text("x") and root / "a-file",
            "sessions_move_destination_not_a_directory",
        ),
    ],
    ids=["missing", "not-a-directory"],
)
async def test_an_unusable_destination_is_refused(
    tmp_path: Path, make_destination, code: str
) -> None:
    result = await _move(_ctx(tmp_path), cwd=str(make_destination(tmp_path)))
    assert _code(result) == code, result.text


@pytest.mark.asyncio
async def test_a_relative_destination_resolves_against_the_session_cwd(tmp_path: Path) -> None:
    (tmp_path / "inside").mkdir()
    context = _ctx(tmp_path)
    result = await _move(context, cwd="inside")
    assert not result.is_error, result.text
    assert (result.details or {})["requested_cwd"] == str((tmp_path / "inside").resolve())


def test_commit_and_reconcile_settle_the_same_intent(tmp_path: Path) -> None:
    destination = tmp_path / "dest"
    destination.mkdir()
    (tmp_path / "sessions" / OWN).mkdir(parents=True)
    intent = cwd_move.enqueue_self_move(
        tmp_path, OWN, requested_cwd=str(destination), effective_cwd=str(tmp_path), owner_pid=1
    )
    # Reconciling while still in the source leaves the request queued: an
    # unobserved move is not a failed one.
    reconciled = cwd_move.reconcile_move(tmp_path, OWN, effective_cwd=str(tmp_path))
    assert reconciled is not None and reconciled.stage == "queued"
    committed = cwd_move.commit_move(tmp_path, OWN, intent.request_id, effective_cwd=str(tmp_path))
    assert committed.stage == cwd_move.STAGE_COMMITTED
    # A committed intent is settled: asking again is a NEW request, not a
    # resurrection of the old one, and committing the old id again is refused.
    with pytest.raises(cwd_move.CwdMoveRefused) as refused:
        cwd_move.commit_move(tmp_path, OWN, intent.request_id, effective_cwd=str(tmp_path))
    assert refused.value.code == "sessions_move_request_not_queued"


def test_a_second_destination_supersedes_a_queued_one(tmp_path: Path) -> None:
    (tmp_path / "sessions" / OWN).mkdir(parents=True)
    first, second = tmp_path / "one", tmp_path / "two"
    first.mkdir()
    second.mkdir()
    a = cwd_move.enqueue_self_move(
        tmp_path, OWN, requested_cwd=str(first), effective_cwd=str(tmp_path), owner_pid=1
    )
    b = cwd_move.enqueue_self_move(
        tmp_path, OWN, requested_cwd=str(second), effective_cwd=str(tmp_path), owner_pid=1
    )
    assert a.request_id != b.request_id
    assert b.stage == cwd_move.STAGE_QUEUED
    with pytest.raises(cwd_move.CwdMoveRefused) as refused:
        cwd_move.commit_move(tmp_path, OWN, a.request_id, effective_cwd=str(tmp_path))
    assert refused.value.code == "sessions_move_request_unknown"
