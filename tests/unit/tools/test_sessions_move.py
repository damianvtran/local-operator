"""A session queues a move of its OWN working directory and nothing else moves.

Slice 1 is the durable intent plus the tool surface: the call returns
``queued``, the effective cwd is untouched, and every other address is refused
with a technical code. The lifecycle (retire, successor boot, confirmation) is
slice 2 and is deliberately absent here.
"""

from __future__ import annotations

import time
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
        ({"cwd": "x"}, {"session_id": ""}, "sessions_move_no_owner_identity"),
        ({"cwd": "x"}, {"job_id": "job-1"}, "sessions_move_child_has_no_runtime"),
    ],
    ids=["no-identity", "child"],
)
async def test_a_runtime_that_cannot_move_itself_is_refused_by_code(
    tmp_path: Path, args: dict[str, Any], context_over: dict[str, Any], code: str
) -> None:
    """The two states that refuse a move outright.

    The ADDRESS refusals that used to sit beside these are gone by design: `move`
    no longer accepts an address at all, so `session`/`pid`/`target`/`peer` are
    refused by the shared wrong-field refusal before this function runs — see
    `test_a_move_takes_only_cwd`, which pins that at the tool's own edge.
    """
    (tmp_path / "x").mkdir()
    result = await _move(_ctx(tmp_path, **context_over), **args)
    assert _code(result) == code, result.text
    assert cwd_move.read_move_intent(tmp_path, OWN) is None, "a refusal writes no intent"


@pytest.mark.asyncio
@pytest.mark.parametrize("stray", ["session", "pid", "target", "peer"])
async def test_a_move_takes_only_cwd(tmp_path: Path, stray: str) -> None:
    """NO ADDRESS IS ACCEPTED, so none is advertised (UX round 1, U6).

    A move acts on the runtime running the call; the schema admitted four address
    fields that could only ever be refused. They are refused here — with the
    shared wrong-field sentence naming `cwd` — rather than by a `move`-specific
    code, and nothing is written.
    """
    (tmp_path / "x").mkdir()
    value: Any = 4242 if stray == "pid" else "elsewhere"
    result = await _move(_ctx(tmp_path), cwd=str(tmp_path / "x"), **{stray: value})
    assert result.is_error, result.text
    assert "`move` takes: cwd." in result.text, result.text
    assert cwd_move.read_move_intent(tmp_path, OWN) is None


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


def test_reconcile_settles_the_intent_it_observes(tmp_path: Path) -> None:
    """One commit primitive: the intent's successor, observed where it boots.

    ``commit_move`` used to exist beside this and had no production caller, while
    repeating the non-realpath comparison the reconcile path had already been
    fixed for (agent review round 1, m2). It is gone: a move commits by being
    OBSERVED, in this function, and there is no second way to say so.
    """
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
    # Observed at the destination: committed, with the request's identity kept.
    committed = cwd_move.reconcile_move(tmp_path, OWN, effective_cwd=str(destination))
    assert committed is not None and committed.stage == cwd_move.STAGE_COMMITTED
    assert committed.request_id == intent.request_id
    # Settled is settled: a later observation in the source never resurrects it.
    again = cwd_move.reconcile_move(tmp_path, OWN, effective_cwd=str(tmp_path))
    assert again is not None and again.stage == cwd_move.STAGE_COMMITTED


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
    stored = cwd_move.read_move_intent(tmp_path, OWN)
    assert stored is not None and stored.request_id == b.request_id
    assert stored.stage == cwd_move.STAGE_QUEUED


@pytest.mark.asyncio
async def test_moving_back_to_the_current_directory_cancels_the_queued_move(
    tmp_path: Path,
) -> None:
    """THE ONLY WAY TO WITHDRAW A MOVE (UX round 1, U4).

    Before this, a queued move could not be undone except by queueing a third
    directory or deleting the marker by hand, and a session that could not
    satisfy the intent kept closing itself on every open. ``move`` back to where
    the session already is now MEANS cancel, and the refusal that used to fire
    first (``destination_is_current``) is reserved for the case with nothing to
    cancel.
    """
    destination = tmp_path / "elsewhere"
    destination.mkdir()
    context = _ctx(tmp_path)
    queued = await _move(context, cwd=str(destination))
    assert not queued.is_error, queued.text

    back = await _move(context, cwd=".")
    assert not back.is_error, back.text
    assert back.text.startswith("cancelled:")
    assert (back.details or {})["stage"] == cwd_move.STAGE_SUPERSEDED
    stored = cwd_move.read_move_intent(tmp_path, OWN)
    assert stored is not None and stored.stage == cwd_move.STAGE_SUPERSEDED

    # With NOTHING queued the same call is the refusal it always was, and it now
    # says what to do instead.
    again = await _move(context, cwd=".")
    assert again.is_error
    assert _code(again) == "sessions_move_destination_is_current"
    assert "no move is queued" in again.text
    assert "cancels a queued move" in again.text


@pytest.mark.asyncio
async def test_the_result_names_the_destination_first_and_the_request_last(
    tmp_path: Path,
) -> None:
    """WHAT THE MODEL IS TOLD (UX round 1, U3 and U8).

    ``queued: 4a579df8…`` led with the least useful fact, gave no destination and
    did not say that the cwd was still the source — so a model could reasonably
    treat "queued" as done and start working in a directory nothing was running
    in.
    """
    destination = tmp_path / "elsewhere"
    destination.mkdir()
    result = await _move(_ctx(tmp_path), cwd=str(destination))
    assert not result.is_error, result.text
    text = result.text
    assert text.startswith("queued: move this session from ")
    assert str(destination.resolve()) in text
    assert "still" in text and "until then" in text
    assert text.index(str(destination.resolve())) < text.index(
        str((result.details or {})["request_id"])
    ), "the request id is the last thing, not the first"
    details = result.details or {}
    assert details["destination_cwd"] == str(destination.resolve())
    # THE WAKE IS REPORTED HONESTLY. This test process hosts no runtime for the
    # session, so the honest answer is "nothing is watching it yet" — the value
    # production gets depends on which process the call runs in.
    assert details["watcher_armed"] is False
    assert "the next time this session opens" in text


@pytest.mark.asyncio
async def test_every_refusal_carries_a_remedy(tmp_path: Path) -> None:
    """A CODE IS NOT AN INSTRUCTION (UX round 1, U5).

    The sibling refusals in this tool are full sentences that name the way out;
    ``move``'s were bare codes, and the fields that would have helped lived in
    ``details``, which the model never sees.
    """
    result = await _move(_ctx(tmp_path), cwd=str(tmp_path / "nope"))
    assert result.is_error
    assert result.text.startswith("refused (sessions_move_destination_missing):")
    assert "does not exist" in result.text


def test_info_reports_a_queued_move_and_why_it_is_held(tmp_path: Path) -> None:
    """`sessions(info)` SAYS IT (UX round 1, U3).

    Between the call that queues a move and the boundary that acts on it there is
    a real state — "still here, going there" — and before this nothing on the
    session's own row said so. The hold a runtime records (B2) rides the same
    line, so an intent that is deliberately not being acted on cannot be mistaken
    for one that simply has not happened yet.
    """
    from local_operator.tools.builtin import (
        _sessions_info_body,
        _sessions_pending_move_fields,
    )

    destination = tmp_path / "elsewhere"
    destination.mkdir()
    context = _ctx(tmp_path)
    queued = cwd_move.enqueue_self_move(
        tmp_path, OWN, requested_cwd=str(destination), effective_cwd=str(tmp_path), owner_pid=1
    )
    fields = _sessions_pending_move_fields(context, OWN, is_own=True)
    assert fields["move_stage"] == cwd_move.STAGE_QUEUED
    assert fields["move_destination"] == queued.destination_cwd
    body = _sessions_info_body({"session_id": OWN, "state": "ready"}, fields, context, is_own=True)
    assert "is queued" in body and queued.destination_cwd in body
    assert "still" in body, "the cwd in force is named, not implied"
    # A hold is reported as a hold.
    cwd_move.mark_blocked(tmp_path, OWN, "no_capable_viewer")
    held = _sessions_pending_move_fields(context, OWN, is_own=True)
    held_body = _sessions_info_body(
        {"session_id": OWN, "state": "ready"}, held, context, is_own=True
    )
    # ...in WORDS with a next step, never a bare code (UX round 2, U10), and
    # never as "at the next idle point".
    assert "queued but held" in held_body
    assert "nothing attached can follow it" in held_body and "cwd='.'" in held_body
    assert "no_capable_viewer" not in held_body and "next idle point" not in held_body
    # A CANCEL reads as one (UX round 2, U14), and a lapsed request is reported as
    # expired by info ITSELF, not only once a runtime ticks (agent review m-e).
    cwd_move.enqueue_self_move(
        tmp_path, OWN, requested_cwd=str(tmp_path), effective_cwd=str(tmp_path), owner_pid=1
    )
    cancelled = _sessions_info_body(
        {"session_id": OWN, "state": "ready"},
        _sessions_pending_move_fields(context, OWN, is_own=True),
        context,
        is_own=True,
    )
    assert "was cancelled" in cancelled and "superseded" not in cancelled
    cwd_move.enqueue_self_move(
        tmp_path, OWN, requested_cwd=str(destination), effective_cwd=str(tmp_path), owner_pid=1
    )
    cwd_move.mark_blocked(
        tmp_path, OWN, "no_capable_viewer", now=time.time() - cwd_move.QUEUED_MOVE_TTL_S - 1
    )
    lapsed = _sessions_pending_move_fields(context, OWN, is_own=True)
    assert lapsed["move_stage"] == cwd_move.STAGE_EXPIRED
    assert "expired without happening" in _sessions_info_body(
        {"session_id": OWN, "state": "ready"}, lapsed, context, is_own=True
    )
    # Another session's marker is not this caller's to report.
    assert _sessions_pending_move_fields(context, "other", is_own=False) == {}


def test_the_approval_card_names_the_directory_the_move_really_uses(tmp_path: Path) -> None:
    """R2-M2: symlinks are followed BEFORE ``..``, as ``Path.resolve`` does.

    cwd ``a/link`` -> ``x/y`` and ``cwd='../dest'`` with BOTH ``a/dest`` and
    ``x/dest`` existing: the card said ``a/dest`` while the op moved to
    ``x/dest``, so the operator approved one existing directory and got another.
    """
    from local_operator.tools.builtin import _describe_sessions_approval

    (tmp_path / "a" / "dest").mkdir(parents=True)
    (tmp_path / "x" / "y").mkdir(parents=True)
    (tmp_path / "x" / "dest").mkdir()
    link = tmp_path / "a" / "link"
    link.symlink_to(tmp_path / "x" / "y")
    used = cwd_move.resolve_destination("../dest", effective_cwd=str(link))
    assert used == str((tmp_path / "x" / "dest").resolve())
    card = _describe_sessions_approval({"op": "move", "cwd": "../dest"}, str(link))
    assert card.startswith(f"move to {used} after this turn (now in "), card
    assert str(tmp_path.resolve() / "a" / "dest") not in card


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("readiness", "recorded", "promised"),
    [
        ("ready", "", True),
        ("no_capable_viewer", "", False),
        ("host_cannot_restart", "", False),
        # Q3 / m-b: a REPEAT call while a hold is already on the marker.
        ("ready", "no_capable_viewer", False),
        ("", "no_capable_viewer", False),
    ],
    ids=["can-act", "nobody-can-follow", "host-cannot-restart", "repeat-ready", "repeat-unhosted"],
)
async def test_the_result_promises_the_idle_point_only_when_the_runtime_can_act(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    readiness: str,
    recorded: str,
    promised: bool,
) -> None:
    """U9's real defect, m-b and Q3: "next idle point" is said only when it is true.

    ``notify_move_queued`` reports what the hosting runtime can do; a predicted
    hold, or one already recorded on the marker, is reported as a hold with its
    reason and next step instead.
    """
    import local_operator.session.runtime.server as server_module

    monkeypatch.setattr(server_module, "notify_move_queued", lambda *_: readiness)
    destination = tmp_path / "elsewhere"
    destination.mkdir()
    if recorded:
        cwd_move.enqueue_self_move(
            tmp_path, OWN, requested_cwd=str(destination), effective_cwd=str(tmp_path), owner_pid=1
        )
        cwd_move.mark_blocked(tmp_path, OWN, recorded)
    result = await _move(_ctx(tmp_path), cwd=str(destination))
    assert not result.is_error, result.text
    assert ("next idle point" in result.text) is promised, result.text
    if not promised:
        assert result.text.startswith("queued but held: "), result.text
        assert "cwd='.'" in result.text, "the way out is named"
        assert (result.details or {})["held"] in {readiness, recorded}


@pytest.mark.asyncio
async def test_both_directories_in_the_result_are_spelled_one_way(tmp_path: Path) -> None:
    """U13: the source was printed as given and the destination realpath-ed."""
    real = tmp_path / "src"
    real.mkdir()
    link = tmp_path / "src-link"
    link.symlink_to(real)
    destination = tmp_path / "dest"
    destination.mkdir()
    result = await _move(_ctx(link), cwd="../dest")
    assert not result.is_error, result.text
    assert str(link) not in result.text, result.text
    assert f"from {real.resolve()} to {destination.resolve()}" in result.text


def test_the_tool_row_tells_a_move_from_a_cancel() -> None:
    """U11: the pending row said only ``move`` for both."""
    from local_operator.harness.rows import sessions_row_summary

    assert sessions_row_summary({"op": "move", "cwd": "~/work/dest"}) == "move · ~/work/dest"
    assert sessions_row_summary({"op": "move", "cwd": "."}) == "move · ."
    assert sessions_row_summary({"op": "move"}) == "move · ?"
