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
  ``ask-response-<ask_id>`` row is absent, and ``AskQueue.reconcile`` marks that
  row present (in ``_handed``) BEFORE it hands the batch on. So an ANSWERING
  path that awaits its own reconcile closes the window at its own ACK, while an
  in-process owner answer — which only SCHEDULES the reconcile (``_settled`` →
  ``_kick``) — leaves it open until the loop runs that task. That contrast is
  the interesting, load-bearing fact for the surface lanes, so both arms are
  pinned here rather than argued in prose.

The session is the real one (``make_session``), the queue is the real queue, and
the op is driven through ``ServingSessionHandle`` — the seam every wire client
crosses. Only ``asyncio`` and the isolated config root are test fixtures.
"""

from __future__ import annotations

import asyncio
from pathlib import Path
from typing import Any

import pytest

from local_operator.asks import render, store
from local_operator.harness.types import StreamEndEvent
from local_operator.session.runtime.serving import ServingSessionHandle
from tests.unit.session.test_session import make_session

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

    # -- the refused arm: answered AND delivered, so the row exists and closes the
    # window before revise_ask is even called.
    delivered_ask = _enqueue(queue, _questions(secret="API_KEY"))
    assert session.respond_ask(delivered_ask, {"API_KEY": ["API_KEY"]})["ok"] is True
    await session.reconcile_asks()
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
async def test_a_revision_is_accepted_through_the_op_while_delivery_has_not_run(
    isolated_config: Path, tmp_path: Path
) -> None:
    """THE WINDOW, at the layer a wire client actually crosses.

    The owner surface answers IN PROCESS — ``Session.respond_ask`` is
    synchronous and only SCHEDULES its reconcile (``_settled`` → ``_kick``) — so
    between that answer and the loop running the scheduled task there is no
    response row, and a revision arriving over the handle (the desktop/relay
    seam) is accepted and supersedes. The handle's own ``reconcile_asks`` then
    delivers the row carrying the REVISED map.

    This ordering is not a fixture artefact: no await stands between the answer
    and the op (the answer path only SCHEDULED its reconcile, and the handle runs
    inline for a caller already on the session's loop), and the asserts below
    show the window shutting exactly where §10 says it shuts — on the row.
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

    # The op awaited its own reconcile, and what it delivered is the REVISION.
    assert row_id in queue.present_row_ids()
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
