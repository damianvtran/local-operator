"""§10's in-flight revision at the SESSION layer (design doc, #1936).

WHY THIS FILE EXISTS, and what each cell is for. The queue's own cells
(``tests/unit/asks/test_queue.py``) pin the fold and the write; these pin the
two things that only exist above it:

* **THE SECRET HOP'S ORDER** — ``Session.revise_ask`` stores a pasted secret
  before the queue sees the map, exactly as ``Session.respond_ask`` does, so the
  refusal must be consulted FIRST. Agent review round 1, MAJOR 2: the first
  version hopped first and then asked, which meant a revision refused as
  delivered still wrote the value into the session's store and announced it to
  later turns — a durable effect from a path whose contract is that the refusal
  IS the effect. The positive control (an admissible revision DOES hop) is in
  the same cell, because "never called" and "never called because the branch is
  dead" are different claims.

* **THE WINDOW'S REACHABILITY** — ``queue.revise`` accepts while the
  ``ask-response-<ask_id>`` row is not DURABLE, and durable means the transcript's
  append resolved — not that reconcile handed the message to a delivery path, and
  certainly not that the answering op's ACK returned (amended 2026-10-04: the ACK
  used to close it via ``_handed``, which is why a person could not land a
  revision in a live session; #1936). The delivery turn's first append — or a
  mid-turn steer's boundary append — is what closes it, and until then the row
  that lands RE-RESOLVES from the fold, so a revision accepted in the gap is what
  the model reads. Both arms are pinned here rather than argued in prose.

The session is the real one (``make_session``), the queue is the real queue, and
the op is driven through ``ServingSessionHandle`` — the seam every wire client
crosses. Only ``asyncio`` and the isolated config root are test fixtures.
"""

from __future__ import annotations

import asyncio
import time
from pathlib import Path
from typing import Any

import pytest

from local_operator.asks import render, store
from local_operator.harness.types import CustomMessage, StreamEndEvent, StreamTextDelta
from local_operator.session.runtime.serving import ServingSessionHandle
from tests.unit.session.test_session import make_session, wait_for

#: A value that must never reach the log, the store or a later turn.
SENTINEL = "sk-live-do-not-persist"


def _questions(count: int = 1, *, secret: str = "") -> list[dict[str, Any]]:
    return [
        {
            "id": secret or f"q{index}",
            "question": "Which one?",
            "options": [] if secret else [{"label": "yes"}, {"label": "no"}],
            "multi": False,
            "secret": bool(secret),
            "persist": False,
            "recommended": None,
        }
        for index in range(count)
    ]


@pytest.fixture
def isolated_config(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    """Home, config and cwd out of the way of the developer's real ones."""
    monkeypatch.setattr(Path, "home", lambda: tmp_path)
    monkeypatch.setenv("LOCAL_OPERATOR_CONFIG_DIR", str(tmp_path))
    monkeypatch.chdir(tmp_path)
    return tmp_path


def _stream() -> Any:
    """A stream that ends at once: a delivery turn costs one empty request."""

    def stream(request: Any, signal: Any) -> Any:
        async def gen() -> Any:
            yield StreamEndEvent(stop_reason="stop")

        return gen()

    return stream


def _ask_session(tmp_path: Path) -> tuple[Any, Any]:
    """A real session with the queued arm live, plus its queue and a first ask."""

    async def handler(questions: Any) -> Any:  # the host hook: only its presence matters
        return None

    session = make_session(tmp_path, _stream())
    session.set_ask_handler(handler)
    queue = session.ask_queue()
    assert queue is not None, "the queued arm must be live for these cells"
    return session, queue


def _enqueue(queue: Any, questions: list[dict[str, Any]]) -> str:
    outcome = queue.enqueue(questions, None)
    assert outcome["ok"] is True, outcome
    return str(outcome["details"]["ask_id"])


async def _run_delivery_turns(session: Any) -> None:
    """Run the session's spawned delivery turn(s) to completion, oldest first.

    A delivery turn is opened through ``_spawn_background``; the response row
    becomes durable at that turn's first append, so a cell that wants the row
    CONSUMED must let the task RUN rather than assert across it. Bounded and
    looped because a turn can spawn follow-ups; draining rather than sleeping
    keeps the ordering a statement about the code, not the scheduler
    (AGENTS.md, timing section).
    """
    for _ in range(10):
        pending = [task for task in list(session._background_tasks) if not task.done()]
        if not pending:
            return
        await asyncio.gather(*pending, return_exceptions=True)
    raise AssertionError("delivery turns never settled")


def _secret_and_plain(
    secret_id: str = "API_KEY", plain_id: str = "q1", *, text: str = ""
) -> list[dict[str, Any]]:
    """One SECRET question and one ordinary one, in that order.

    The order is the point: the hop walks the questions, so a secret cell in the
    map is the thing that used to make the hop return a cell for the OTHER
    question too (see the partial-map cell below). ``text`` keeps two asks in one
    test from tripping the "do not re-ask" guard while the first is still open.
    """
    return [
        {
            "id": secret_id,
            "question": f"Paste the key{text}",
            "options": [],
            "multi": False,
            "secret": True,
            "persist": False,
            "recommended": None,
        },
        {
            "id": plain_id,
            "question": f"Which one?{text}",
            "options": [{"label": "yes"}, {"label": "no"}],
            "multi": False,
            "secret": False,
            "persist": False,
            "recommended": None,
        },
    ]


def test_the_ask_op_family_carries_the_session_loop_hop() -> None:
    """Every ask op the dispatch reaches must be hopped onto the session's loop.

    ``ServingSessionHandle``'s docstring states the contract — *"every ``async
    def`` that touches session state carries ``@_on_session_loop``, so its WHOLE
    BODY runs on the session's loop"* — and agent review round 1 (BLOCKER 1)
    found ``ask_revise`` alone without it: on the thread-hosted plane its fold
    read and its log append would run on the registrant's thread, concurrently
    with ``reconcile``, which is the accepted-and-then-dropped interleaving §10
    forbids by name.

    ENUMERATED rather than derived, because the point is that a FIFTH op cannot
    repeat it: add one to ``_dispatch``'s queued-ask arm and it belongs here too.
    ``test_daemon_serving_plane.py`` drives the real thread-hosted plane for the
    two mutating ``def``s; this is the cheap half of the same contract.
    """
    family = ("ask_answer", "ask_respond", "ask_revise", "ask_decline", "ask_dismiss")
    unhopped = [
        name
        for name in family
        if getattr(getattr(ServingSessionHandle, name), "__wrapped__", None) is None
    ]
    assert not unhopped, (
        f"{unhopped} carry no @_on_session_loop: their bodies would run on the "
        "caller's thread, not the session's loop"
    )


@pytest.mark.asyncio
async def test_a_refused_revision_never_stores_or_announces_its_secret(
    isolated_config: Path, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """MAJOR 2: the queue decides before the secret hop, so a refusal costs nothing.

    The store is REAL (a refused store would be its own failure mode), and the hop
    is counted by delegating to it: the claim is about whether the credential was
    written and journaled, not about what a return value said.
    """
    from local_operator.variables import VariableStore

    session, queue = _ask_session(tmp_path)
    session._variables = VariableStore(cwd=str(tmp_path))  # noqa: SLF001 — the store the hop writes

    calls: list[Any] = []
    real_apply = render.apply_secret_answers

    def spy(*args: Any, **kwargs: Any) -> Any:
        calls.append(args[1])
        return real_apply(*args, **kwargs)

    monkeypatch.setattr(render, "apply_secret_answers", spy)

    # -- the refused arm: answered AND consumed, so the row is DURABLE and
    # closes the window before revise_ask is even called. The awaited reconcile
    # HANDS the row to the delivery path; the idle-spawned turn's first append
    # is what makes it durable, so the turn is run to completion here — the
    # refusal below is then a statement about the consumed row, not scheduler
    # order (consumption bound, amended 2026-10-04).
    delivered_ask = _enqueue(queue, _questions(secret="API_KEY"))
    assert session.respond_ask(delivered_ask, {"API_KEY": ["API_KEY"]})["ok"] is True
    await session.reconcile_asks()
    await _run_delivery_turns(session)
    assert store.response_row_id(delivered_ask) in queue.present_row_ids()
    calls.clear()
    refused = session.revise_ask(delivered_ask, {"API_KEY": [SENTINEL]}, by="desktop")
    assert refused["ok"] is False
    assert refused["error"] == render.REVISED_ALREADY_DELIVERED
    assert calls == [], "the hop ran for a revision the queue was always going to refuse"
    assert SENTINEL not in store.asks_log_path(queue.session_dir).read_text()

    # -- the positive control: an admissible revision DOES hop, so the assertion
    # above is about the order and not about a branch nothing reaches.
    open_ask = _enqueue(queue, _questions(secret="API_KEY"))
    assert session.respond_ask(open_ask, {"API_KEY": ["API_KEY"]})["ok"] is True
    calls.clear()
    accepted = session.revise_ask(open_ask, {"API_KEY": [SENTINEL]}, by="desktop")
    assert accepted["ok"] is True and accepted["revised"] is True
    assert len(calls) == 1
    # The value was stored, and the LOG still holds the key name only.
    log = store.asks_log_path(queue.session_dir).read_text()
    assert SENTINEL not in log
    record = queue.find(open_ask)
    assert record is not None and record["answers"] == {"API_KEY": ["API_KEY"]}


@pytest.mark.asyncio
async def test_a_partial_map_is_refused_even_when_the_ask_carries_a_secret(
    isolated_config: Path, tmp_path: Path
) -> None:
    """PRE-EXISTING defect found en route, identical in ``respond_ask`` (round 2).

    REPRODUCTION, kept as a test: with a SECRET question in the ask, the hop
    returned a cell for EVERY question id — ``[]`` for the ones the caller never
    supplied — so ``_whole_ask_cells``' completeness check saw a full map and an
    omitted question was recorded as "no answer" instead of being refused. Asked
    for here because the contract is already open in this PR: the hop now returns
    ONLY the cells the caller supplied, so the check sees the true key set.

    BOTH ops are pinned, and each gets a FRESH session, because the defect shipped
    in the shared hop and ``respond_ask`` is where it shipped: the queued revision
    path must not inherit it either. The store is real, so the assertion about the
    refusal is about the op and not about a missing credential backend.
    """
    from local_operator.variables import VariableStore

    for op in ("respond_ask", "revise_ask"):
        home = tmp_path / op
        home.mkdir(parents=True, exist_ok=True)
        session, queue = _ask_session(home)
        session._variables = VariableStore(cwd=str(home))  # noqa: SLF001 — the store the hop writes
        secret_id = f"{op.upper()}_API_KEY"
        ask_id = _enqueue(queue, _secret_and_plain(secret_id=secret_id, text=f" ({op})"))

        # The caller supplies ONLY the secret cell and omits the other question.
        outcome = getattr(session, op)(ask_id, {secret_id: [secret_id.upper()]})

        assert outcome["ok"] is False, (op, outcome)
        assert "'q1'" in outcome["error"], (op, outcome)
        assert "empty list" in outcome["error"], (op, outcome)
        # The refusal is the whole effect: no answer was recorded for THIS ask, so
        # the omitted question cannot be read back as a deliberate "no answer".
        kinds = [
            event["kind"]
            for event in store.read_events(queue.session_dir)
            if event["ask_id"] == ask_id
        ]
        assert kinds == [store.EVENT_QUEUED], (op, kinds)


@pytest.mark.asyncio
async def test_a_complete_map_with_a_secret_question_still_records_every_cell(
    isolated_config: Path, tmp_path: Path
) -> None:
    """THE FILTER'S OTHER SIDE: a COMPLETE map still records every cell.

    The cell above proves an omitted question stays omitted; this pins the arm that
    must not move — the secret cell recorded as its KEY NAME (never the pasted
    bytes) and the ordinary cell as the value chosen.

    A behavioural guard rather than a teeth-bearing one, and worth saying so:
    ``Session.respond_ask`` merges the hop's output OVER the caller's own map, so a
    filter that dropped a SUPPLIED cell would not change the record — and a dropped
    secret cell would still be caught downstream by ``_guard_secret_cells``. The
    filter's whole job is to withhold the cells nobody supplied, which is the cell
    above; this one pins the post-fix shape of the complete path so a later change
    to the hop has to face both arms at once.
    """
    from local_operator.variables import VariableStore

    session, queue = _ask_session(tmp_path)
    session._variables = VariableStore(cwd=str(tmp_path))  # noqa: SLF001 — the store the hop writes
    ask_id = _enqueue(queue, _secret_and_plain())

    outcome = session.respond_ask(ask_id, {"API_KEY": [SENTINEL], "q1": ["yes"]})

    assert outcome["ok"] is True, outcome
    record = queue.find(ask_id)
    assert record is not None
    assert record["answers"]["q1"] == ["yes"], record["answers"]
    assert SENTINEL not in record["answers"]["API_KEY"], record["answers"]
    assert SENTINEL not in store.asks_log_path(queue.session_dir).read_text()


@pytest.mark.asyncio
async def test_a_supplied_empty_cell_survives_the_hop_filter(
    isolated_config: Path, tmp_path: Path
) -> None:
    """NIT 1 (round 3): a deliberate "no answer" keeps its key through the filter.

    The filter discriminates on the caller's KEY SET, not on cell emptiness — an
    explicit ``[]`` is how "no answer" is said (§2.4) and must survive, while an
    omission has no key to survive on. Written as ``if cell`` instead of
    ``if qid in answers``, the filter would drop the supplied ``[]`` and turn a
    legitimate skip into the partial-map refusal.

    THE ADAPTER'S OWN OUTPUT IS THE ONLY PLACE THIS IS VISIBLE, and that is why
    the arm is written at that level: ``Session.respond_ask`` merges the hop's map
    OVER the caller's own, so a dropped ``[]`` would still be present from the
    caller's copy and a session-level cell could not discriminate — which is
    exactly why nothing in the suite redded against that mutation. The second half
    pins the whole-path behaviour the adapter assertion protects.
    """
    from local_operator.variables import VariableStore

    session, queue = _ask_session(tmp_path)
    session._variables = VariableStore(cwd=str(tmp_path))  # noqa: SLF001 — the store the hop writes

    # (a) the adapter, directly: the supplied [] comes back WITH its key.
    hopped = render.apply_secret_answers(
        _secret_and_plain(),
        {"API_KEY": ["API_KEY"], "q1": []},
        variables=session.variables,
    )
    assert "q1" in hopped, hopped
    assert hopped["q1"] == [], hopped

    # (b) the whole path: the skip is accepted, and it is recorded as [].
    ask_id = _enqueue(queue, _secret_and_plain())
    outcome = session.respond_ask(ask_id, {"API_KEY": ["API_KEY"], "q1": []})
    assert outcome["ok"] is True, outcome
    record = queue.find(ask_id)
    assert record is not None
    assert record["answers"]["q1"] == [], record["answers"]


@pytest.mark.asyncio
async def test_a_revision_is_accepted_through_the_op_while_delivery_has_not_run(
    isolated_config: Path, tmp_path: Path
) -> None:
    """THE WINDOW, at the layer a wire client actually crosses.

    The owner surface answers IN PROCESS — ``Session.respond_ask`` is
    synchronous and only SCHEDULES its reconcile (``_settled`` → ``_kick``) — so
    between that answer and the delivery path's append there is no durable row,
    and a revision arriving over the handle (the desktop/relay seam) is accepted
    and supersedes.

    THE ACK CLOSES NOTHING (amended 2026-10-04): the handle's own
    ``reconcile_asks`` hands the row to the delivery path and SPAWNS the
    delivery turn — which this cell then runs, because the row it appends must
    be the REVISED one (every append re-resolves from the fold) and the window
    must only shut there. The asserts walk that boundary explicitly: open before
    the op, still open right after the ACK, shut after the append.
    """
    session, queue = _ask_session(tmp_path)
    ask_id = _enqueue(queue, _questions(2))
    loop = asyncio.get_running_loop()
    handle = ServingSessionHandle(session, loop, cwd=str(tmp_path))

    assert session.respond_ask(ask_id, {"q0": ["no"], "q1": ["maybe"]}, by="terminal")["ok"]
    row_id = store.response_row_id(ask_id)
    assert row_id not in queue.present_row_ids(), "the window must still be open here"

    outcome = await handle.ask_revise(ask_id, {"q0": ["yes"], "q1": ["maybe"]}, by="desktop")
    assert outcome == "revised"

    # The op's awaited reconcile hands the row off and spawns the delivery turn;
    # the ACK is not the close — the row is still not durable here.
    assert row_id not in queue.present_row_ids(), "the ACK closes nothing"
    await _run_delivery_turns(session)

    # The append is the close, and what it landed is the REVISION — re-resolved
    # from the fold, into the transcript entry the model reads.
    assert row_id in queue.present_row_ids()
    entry = next(e for e in session._transcript.entries() if e.id == row_id)
    assert entry.payload["details"]["answers"] == {"q0": ["yes"], "q1": ["maybe"]}
    record = queue.find(ask_id)
    assert record is not None
    assert record["answers"] == {"q0": ["yes"], "q1": ["maybe"]}
    assert record["answered_at"] and record["answered_by"] == {"surface": "terminal"}
    assert [event["kind"] for event in store.read_events(queue.session_dir)] == [
        store.EVENT_QUEUED,
        store.EVENT_ANSWERED,
        store.EVENT_REVISED,
    ]

    # …and the window is now shut for the next one, in the revision path's own
    # sentence — the one the surface renders rather than a paraphrase.
    with pytest.raises(ValueError, match="already delivered"):
        await handle.ask_revise(ask_id, {"q0": ["maybe"], "q1": ["maybe"]}, by="desktop")


# ---------------------------------------------------------------------------
# the consumption bound, session layer (amended 2026-10-04)
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_a_revision_is_carried_into_the_row_the_delivery_turn_appends(
    isolated_config: Path, tmp_path: Path
) -> None:
    """THE IDLE CARRY: the message handed at reconcile is a PREVIEW, and the row
    the delivery turn appends is rebuilt from the fold at the append.

    Holding ``_turn_lock`` keeps the spawned delivery turn parked, so the state
    is exact rather than racy: answered, handed, nothing durable — the revision
    accepted here is the one a real user sends in the seconds after the answer,
    and it MUST be what the model reads (the append re-resolves; without that,
    the window would accept-and-drop). Then the lock releases, the turn runs,
    and the transcript entry is asserted — and the post-append revision refused.
    """
    session, queue = _ask_session(tmp_path)
    ask_id = _enqueue(queue, _questions(2))
    loop = asyncio.get_running_loop()
    handle = ServingSessionHandle(session, loop, cwd=str(tmp_path))

    await session._turn_lock.acquire()
    try:
        answered = await handle.ask_respond(ask_id, {"q0": ["no"], "q1": ["maybe"]}, by="terminal")
        assert answered == "answered"
        row_id = store.response_row_id(ask_id)
        assert row_id not in queue.present_row_ids(), "hand-off consumed nothing"
        outcome = await handle.ask_revise(ask_id, {"q0": ["yes"], "q1": ["maybe"]}, by="desktop")
        assert outcome == "revised"
        assert row_id not in queue.present_row_ids(), "still nothing durable"
    finally:
        session._turn_lock.release()

    await _run_delivery_turns(session)

    assert row_id in queue.present_row_ids()
    entry = next(e for e in session._transcript.entries() if e.id == row_id)
    assert entry.payload["details"]["answers"] == {"q0": ["yes"], "q1": ["maybe"]}
    record = queue.find(ask_id)
    assert record is not None and record["answers"] == {"q0": ["yes"], "q1": ["maybe"]}
    with pytest.raises(ValueError, match="already delivered"):
        await handle.ask_revise(ask_id, {"q0": ["maybe"], "q1": ["maybe"]}, by="desktop")


@pytest.mark.asyncio
async def test_a_mid_turn_revision_is_carried_by_the_boundary_append(
    isolated_config: Path, tmp_path: Path
) -> None:
    """THE STEERING CARRY: an answer during a running turn parks on the steering
    queue, and the boundary append carries a revision accepted while the turn is
    still mid-flight — the exact moment a mis-tap is cheapest to fix (#1936).

    The turn is held inside its provider stream until the test releases it, so
    `mid-turn` is a fact of the run, not a clock race; the waits are on events
    the code already publishes (AGENTS.md, timing section). The revision arrives
    after the hand-off is visible on the steering queue — i.e. strictly later
    than anything the old ACK-bound code would have refused.
    """
    released = asyncio.Event()

    def stream(request: Any, signal: Any) -> Any:
        async def gen() -> Any:
            yield StreamTextDelta(delta="working")
            await released.wait()
            yield StreamEndEvent(stop_reason="stop")

        return gen()

    async def handler(questions: Any) -> Any:
        return None

    session = make_session(tmp_path, stream)
    session.set_ask_handler(handler)
    queue = session.ask_queue()
    assert queue is not None, "the queued arm must be live for this cell"
    ask_id = _enqueue(queue, _questions(2))
    loop = asyncio.get_running_loop()
    handle = ServingSessionHandle(session, loop, cwd=str(tmp_path))

    turn = asyncio.create_task(session.prompt("do the work"))
    await wait_for(lambda: session.is_streaming)

    # The answer lands while the turn is mid-flight: the delivery goes to the
    # steering queue (the session is busy), visible as that queue going non-empty.
    assert session.respond_ask(ask_id, {"q0": ["no"], "q1": ["maybe"]}, by="terminal")["ok"]
    await wait_for(lambda: not session._steering_queue.empty())
    row_id = store.response_row_id(ask_id)
    assert row_id not in queue.present_row_ids(), "the boundary has not been crossed"

    # The revision arrives AFTER that hand-off was visible, still mid-turn: the
    # window is the append, so it is accepted (the old bound refused exactly here).
    outcome = await handle.ask_revise(ask_id, {"q0": ["yes"], "q1": ["maybe"]}, by="desktop")
    assert outcome == "revised"

    # Cross the boundary: the drain appends the row (re-resolved from the fold)
    # and injects it into the same turn's context.
    released.set()
    await turn

    assert row_id in queue.present_row_ids()
    entry = next(e for e in session._transcript.entries() if e.id == row_id)
    assert entry.payload["details"]["answers"] == {"q0": ["yes"], "q1": ["maybe"]}
    injected = [m for m in session._context.messages if getattr(m, "id", None) == row_id]
    assert injected, "the drained steer must reach live context"
    assert isinstance(injected[0], CustomMessage)
    assert injected[0].details["answers"] == {"q0": ["yes"], "q1": ["maybe"]}

    with pytest.raises(ValueError, match="already delivered"):
        await handle.ask_revise(ask_id, {"q0": ["no"], "q1": ["maybe"]}, by="desktop")


def test_a_stopped_sessions_revision_lands_through_the_boot_reconcile(
    isolated_config: Path, tmp_path: Path
) -> None:
    """THE KEPT CASE: an ask answered while the session was STOPPED.

    With no running loop the answer and the revision are log writes and nothing
    else — ``_kick`` cannot schedule, and the next runtime's boot reconcile is
    the backstop (design §2.2) — so the window is the log alone and that boot is
    what delivers. The row it appends must carry the revision. The +60 s is a
    CLOCK shift rather than a sleep: the property is "elapsed time does not
    close the window", and the fold is the clock's only reader.
    """
    session, queue = _ask_session(tmp_path)
    ask_id = _enqueue(queue, _questions(2))
    accepted = session.respond_ask(ask_id, {"q0": ["no"], "q1": ["maybe"]}, by="terminal")
    assert accepted["ok"] is True

    shifted = int(time.time() * 1000) + 60_000
    queue._now = lambda: shifted
    revised = session.revise_ask(ask_id, {"q0": ["yes"], "q1": ["maybe"]}, by="desktop")
    assert revised["ok"] is True and revised["revised"] is True

    # A stopped session has no loop for the schedules its writes imply (the wake
    # arm/retire, the reconcile kicks). CPython's policy still hands plain
    # ``ensure_future`` a fresh, NEVER-STARTED loop, so those tasks sit on it
    # unrun — which IS the stopped state — and the boot below must run on the
    # one loop that actually executes: settle the strays first (cancel, let
    # their own loop process the cancels, close it), so neither the delivery
    # drain nor ``dispose`` ever meets a foreign-loop task.
    stray_loops = {task.get_loop() for task in session._background_tasks if not task.done()}
    for task in list(session._background_tasks):
        task.cancel()
    for stray in stray_loops:
        if not stray.is_running() and not stray.is_closed():
            stray.run_until_complete(asyncio.sleep(0))
            stray.close()

    async def boot() -> None:
        await session.reconcile_asks(load_time=True)
        await _run_delivery_turns(session)
        await session.dispose()

    asyncio.run(boot())

    row_id = store.response_row_id(ask_id)
    assert row_id in queue.present_row_ids()
    entry = next(e for e in session._transcript.entries() if e.id == row_id)
    assert entry.payload["details"]["answers"] == {"q0": ["yes"], "q1": ["maybe"]}


@pytest.mark.asyncio
async def test_the_wire_flag_stays_false_until_the_append_lands(
    isolated_config: Path, tmp_path: Path
) -> None:
    """THE WIRE READING: ``delivered`` is the consumption flag now.

    A surface gating the change affordance on ``delivered:false`` must see the
    flag true only once the answer is committed to the conversation — exactly
    the window — so this cell records every publication the queue pushes and
    checks the two ends of the arc: false after the ACK, true after the append.
    """
    session, queue = _ask_session(tmp_path)
    loop = asyncio.get_running_loop()
    handle = ServingSessionHandle(session, loop, cwd=str(tmp_path))
    # AFTER the handle, not before: constructing the handle installs its OWN
    # ask-state sink on the session (the repaint path for attached surfaces),
    # and the last installer wins — this cell's readings need the raw
    # publications, so the recorder must be what the session carries.
    published: list[list[dict[str, Any]]] = []
    session.set_ask_state_sink(lambda rows, count: published.append(list(rows or [])))
    ask_id = _enqueue(queue, _questions())

    assert await handle.ask_respond(ask_id, {"q0": ["yes"]}, by="desktop") == "answered"

    def delivered_now() -> Any:
        rows = [row for row in published[-1] if row["ask_id"] == ask_id]
        assert rows, published[-1]
        return rows[0]["delivered"]

    row_id = store.response_row_id(ask_id)
    assert row_id not in queue.present_row_ids()
    assert delivered_now() is False, "the ACK must not read as consumption"

    await _run_delivery_turns(session)

    assert row_id in queue.present_row_ids()
    assert delivered_now() is True, "the append is where the flag flips"
