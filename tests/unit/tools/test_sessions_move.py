"""A session queues a move of its OWN working directory and nothing else moves.

Slice 1 is the durable intent plus the tool surface: the call returns
``queued``, the effective cwd is untouched, and every other address is refused
with a technical code. The lifecycle (retire, successor boot, confirmation) is
slice 2 and is deliberately absent here.
"""

from __future__ import annotations

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
    # A TRANSIENT hold is reported as a hold -- in WORDS with a next step, never
    # as a bare code (UX round 2, U10) and never as "at the next idle point".
    cwd_move.mark_blocked(tmp_path, OWN, "destination_unavailable")
    held = _sessions_pending_move_fields(context, OWN, is_own=True)
    held_body = _sessions_info_body(
        {"session_id": OWN, "state": "ready"}, held, context, is_own=True
    )
    assert "queued but held" in held_body
    assert "is gone" in held_body and "cwd='.'" in held_body
    assert "destination_unavailable" not in held_body and "next idle point" not in held_body
    # A CANCEL reads as one (UX round 2, U14).
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
    # A RECORDED HANDOFF says the session is stopping rather than still running:
    # the one stage a caller must not misread as "it is still here".
    handoff = cwd_move.enqueue_self_move(
        tmp_path, OWN, requested_cwd=str(destination), effective_cwd=str(tmp_path), owner_pid=1
    )
    cwd_move.accept_handoff(tmp_path, OWN, handoff)
    handed = _sessions_pending_move_fields(context, OWN, is_own=True)
    assert handed["move_stage"] == cwd_move.STAGE_HANDOFF
    assert "is recorded" in _sessions_info_body(
        {"session_id": OWN, "state": "ready"}, handed, context, is_own=True
    )
    # A marker an OLDER build aged out is still reported, once, in words.
    expired = cwd_move._with_stage(handoff, cwd_move.STAGE_EXPIRED)
    cwd_move._write(tmp_path, expired)
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
    assert card.startswith(f"move to {used} (now in "), card
    assert str(tmp_path.resolve() / "a" / "dest") not in card


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("readiness", "recorded", "shape"),
    [
        ("ready", "", "idle-point"),
        # Nobody can follow: the runtime writes the destination and stops, so the
        # reply promises a restart at the NEXT ENGAGE, not at an idle point.
        ("handoff", "", "handoff"),
        # A REPEAT call while a transient hold is on the marker (Q3 / m-b), with
        # or without a runtime in this process.
        ("ready", "destination_unavailable", "held"),
        ("", "destination_unavailable", "held"),
    ],
    ids=["viewer-follows", "nobody-can-follow", "repeat-ready", "repeat-unhosted"],
)
async def test_the_result_says_which_way_the_session_will_move(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    readiness: str,
    recorded: str,
    shape: str,
) -> None:
    """U9's real defect, m-b and Q3: the promise must match what will happen.

    ``notify_move_queued`` reports what the hosting runtime will do -- follow the
    move with a viewer, hand the destination to the next engage, or (refused
    before this point) never act at all. The reply is worded from that answer, so
    "at the next idle point" is said only when a successor really follows, and the
    handoff case says the session STOPS and starts elsewhere when it is next used.
    A transient hold already recorded on the marker is reported as a hold with its
    reason and the way out.
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
    if shape == "idle-point":
        assert "next idle point" in result.text, result.text
        assert (result.details or {})["handoff"] is False
    elif shape == "handoff":
        assert "next idle point" not in result.text, result.text
        assert result.text.startswith("handing off: "), "the shape leads, not a third clause"
        # THE STOP IS BOUNDED BY IDLENESS, NOT BY THE TURN (design review D3): a
        # parked gate or a background job defers it past the turn end.
        assert "next point of idleness" in result.text, result.text
        assert "stops when this turn ends" not in result.text, result.text
        assert (result.details or {})["handoff"] is True, "the shape is machine-readable too"
    else:
        assert result.text.startswith("queued but held: "), result.text
        assert "cwd='.'" in result.text, "the way out is named"
        assert (result.details or {})["held"] == recorded


@pytest.mark.asyncio
async def test_a_host_that_can_never_restart_is_refused_at_the_call(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A request that could only ever sit there is not recorded at all.

    ``host_cannot_restart`` used to be a hold: the tool queued the move, the
    runtime recorded why it could not act, and thirty minutes later the request
    lapsed. With the TTL gone such a request would sit ``queued`` for ever, so the
    call refuses it -- AND DOES SO BEFORE RECORDING ANYTHING (agent review NIT-3):
    a marker published first would be a request a runtime in another process could
    already be acting on when the refusal is returned.
    """
    import local_operator.session.runtime.server as server_module

    monkeypatch.setattr(
        server_module,
        "move_host_refusal",
        lambda *_: "host_cannot_restart",
    )
    destination = tmp_path / "elsewhere"
    destination.mkdir()
    result = await _move(_ctx(tmp_path), cwd=str(destination))
    assert _code(result) == "host_cannot_restart", result.text
    assert "cannot restart it in another directory" in result.text, result.text
    assert cwd_move.read_move_intent(tmp_path, OWN) is None, "nothing was recorded"


@pytest.mark.asyncio
async def test_a_destination_outside_the_confinement_is_refused_and_withdrawn(
    tmp_path: Path,
) -> None:
    """THE BOUNDARY THE TOOLS ENFORCE IS THE BOUNDARY THE MOVE RESPECTS.

    ``_sessions_move`` consulted ``context.job_id`` but never
    ``context.confinement_root``, so a confined session could queue a move outside
    its world root -- and every tool in the relocated session would then refuse the
    directory it had been moved into.

    THE REFUSAL SPEAKS IN THE BOUNDARY'S OWN WORDS (design review D2): the check
    already computes "this session is confined to <root>, and its working directory
    (<path>) is outside it", which is the one fact an agent needs to pick a
    directory it may use; a static remedy withheld it. Nothing is recorded either --
    the check runs before the enqueue (agent review NIT-3).
    """
    inside = tmp_path / "world"
    outside = tmp_path / "outside"
    inside.mkdir()
    outside.mkdir()
    context = ToolContext(
        cwd=str(inside),
        session_id=OWN,
        session_dir=str(tmp_path / "sessions" / OWN),
        confinement_root=str(inside),
    )
    result = await _move(context, cwd=str(outside))
    assert _code(result) == "sessions_move_outside_confinement", result.text
    assert "confined" in result.text, result.text
    # The root and the offending path, both named: that is what makes the refusal
    # actionable without provoking a second tool's error.
    assert str(inside.resolve()) in result.text, result.text
    assert str(outside.resolve()) in result.text, result.text
    assert cwd_move.read_move_intent(tmp_path, OWN) is None, "nothing was recorded"
    # INSIDE the boundary the same call is the ordinary queued move.
    inner = inside / "sub"
    inner.mkdir()
    allowed = await _move(context, cwd=str(inner))
    assert not allowed.is_error, allowed.text
    assert (allowed.details or {})["stage"] == cwd_move.STAGE_QUEUED


def test_info_renders_a_recorded_handoff(tmp_path: Path) -> None:
    """``info`` names the handoff stage in words a caller can plan around."""
    from local_operator.tools.builtin import _sessions_move_state_sentence

    destination, source = tmp_path / "dest", tmp_path / "src"
    destination.mkdir()
    source.mkdir()
    text = _sessions_move_state_sentence(
        {
            "move_stage": cwd_move.STAGE_HANDOFF,
            "move_destination": str(destination),
            "move_source": str(source),
        }
    )
    assert "is recorded" in text, text
    assert "once it is idle it stops" in text, "idleness bounds the stop, not the turn"
    assert "the next thing to start it" in text, text
    assert str(destination) in text and str(source) in text


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


def test_the_help_reference_describes_what_move_really_does(tmp_path: Path) -> None:
    """THE CONTRACT A MODEL READS BEFORE IT CALLS `move` (design review D1).

    It used to say the op NEEDS a viewer, that with none the move is HELD, and
    that a hold EXPIRES — three statements this change makes false. It also
    omitted the two refusals the op now has. A model that plans around a held move
    that lapses plans around a request that never lapses and never holds.
    """
    from local_operator.tools.builtin import (
        _SESSIONS_OP_DOCS,
        render_sessions_reference,
    )

    rendered = render_sessions_reference()
    move_doc = _SESSIONS_OP_DOCS["move"]
    text = move_doc.summary + " " + " ".join(move_doc.refusals)
    assert "It needs a viewer" not in text, "a viewer is not a precondition"
    assert "and expires" not in text.lower(), "nothing expires any more"
    assert "nothing expires" in text.lower(), "and the reference says so"
    assert "with none, the move is held" not in text.lower()
    assert "the next engage — a message, a wake, or `lop exec` — starts it there" in text, text
    for code in ("host_cannot_restart", "sessions_move_outside_confinement"):
        assert code in text, f"{code} is a refusal of this op"
    assert "`move`" in rendered


def test_the_approval_card_names_the_stop_when_nothing_can_follow(tmp_path: Path) -> None:
    """The card IS the approval, so the viewer-less consequence is on it.

    With nothing attached the session stops first; a card promising only a restart
    approved a restart the operator may not get (agent review round 1, NIT-1).
    """
    from local_operator.tools.builtin import _describe_sessions_approval

    source = tmp_path / "src"
    destination = tmp_path / "dest"
    source.mkdir()
    destination.mkdir()
    card = _describe_sessions_approval({"op": "move", "cwd": "../dest"}, str(source))
    assert "restarts there" in card and "stopping first if nothing is attached" in card, card


def test_a_repeat_request_does_not_demote_a_recorded_handoff(tmp_path: Path) -> None:
    """m2 / Q2: the same destination is the same request, at ANY unobserved stage.

    At ``handoff`` the identical request used to fall through to the supersede
    branch, withdraw the destination the engage guard was honouring and mint a
    fresh ``queued`` request whose source is the caller's cwd — so the next engage
    booted back in the SOURCE.
    """
    (tmp_path / "sessions" / OWN).mkdir(parents=True)
    source, destination = tmp_path / "src", tmp_path / "dest"
    source.mkdir()
    destination.mkdir()
    first = cwd_move.enqueue_self_move(
        tmp_path, OWN, requested_cwd=str(destination), effective_cwd=str(source), owner_pid=1
    )
    handoff = cwd_move.accept_handoff(tmp_path, OWN, first)
    assert handoff is not None and handoff.stage == cwd_move.STAGE_HANDOFF
    again = cwd_move.enqueue_self_move(
        tmp_path, OWN, requested_cwd=str(destination), effective_cwd=str(destination), owner_pid=1
    )
    assert again.request_id == handoff.request_id, "a retry is one request"
    assert again.stage == cwd_move.STAGE_HANDOFF, "and it is still the recorded truth"
    stored = cwd_move.read_move_intent(tmp_path, OWN)
    assert stored is not None and stored.destination_cwd == str(destination.resolve())
    assert stored.stage == cwd_move.STAGE_HANDOFF
