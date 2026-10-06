"""End-to-end: the durable ask queue over a REAL session and a REAL transcript.

Why this file exists
--------------------

``asks/`` has unit coverage for the fold, the caps and the text, and that
coverage would all stay green if the queue were never CONSTRUCTED — the failure
class the design's §5 invariant lives or dies on. So these cells drive assembled
sessions: the real ``ask`` tool reached through the real tool inventory, writing
a real ``asks.jsonl`` beside a real ``transcript.jsonl``, with the responses
injected as real turns.

Three properties are asserted here and nowhere else:

* **The kill-switch arm is the old behaviour.** With ``NONBLOCKING_ASK`` False —
  the kill switch, since the queue became the shipped default on 2026-10-03 —
  the tool awaits the host hook exactly as it did before the queue existed and
  NOTHING is written to an ask log. This is the arm ``LOP_ASK_NONBLOCKING=0``
  restores for an operator, so it stays covered rather than retired.
* **The log survives the runtime.** A second session over the same directory
  delivers what the first one owed, exactly once, because the transcript row is
  the delivery marker rather than a process-local boolean.
* **Nothing secret reaches disk.** A sentinel credential value is grepped for
  across the ask log, the derived index and the transcript.

The provider is scripted for the reason ``test_ask_tool_e2e`` gives: this stage's
failure signal is "the ask was never delivered", and live model latency inside
its bound would only make that signal slower to read.
"""

from __future__ import annotations

import asyncio
from pathlib import Path
from typing import Any

import pytest

from local_operator.asks import policy, store
from local_operator.session.session import Session
from tests.e2e.harness import (
    ScriptedStream,
    build_session,
    dispose_quietly,
    text_turn,
    tool_call_turn,
)
from tests.e2e.watchdog import bounded

pytestmark = pytest.mark.e2e

BOUND_S = 90.0

#: How long a cell waits for a row the runtime OWES — a response/timeout row after
#: ``respond_ask`` / ``withdraw_ask`` / ``reconcile_asks``, delivered by a spawned
#: turn (and, for the boot cells, by the boot reconcile).
#:
#: CALIBRATION, from CI junits as of 2026-10-06: three healthy samples of this file's
#: waits read 0.100 s (macos-latest, run 37411607304), 0.107 s and 0.082 s
#: (ubuntu-latest / macos-latest, run 37404837806), and the one known failure read
#: >=30.421 s — i.e. it hit its own 30 s deadline — in run 37411607304
#: (``test_an_answer_racing_a_withdrawal_wins_in_both_orders``, the Direction-2 wait).
#: The job duration that run was normal (505 s against 512 s on the prior green run),
#: so this is not a slow runner.
#:
#: 60 s is ~600x the healthy samples and 2x the observed floor of the stall, and it is
#: deliberately BELOW the cell's stage bound (``BOUND_S``, 90 s): a wait that does time
#: out must fail with THIS wait's state block rather than let the outer watchdog end
#: the stage with a stack dump and no account of what the runtime owed.
#:
#: WHAT THIS STOPS CATCHING: a stall between 31 s and 60 s now PASSES where it used to
#: red. Accepted, because this wait is an EVENT BACKSTOP and not a timing assertion
#: (AGENTS.md, "wait on the event, never on the clock"): what it asserts is that the
#: row eventually arrives, and 60 s still refuses to call a stuck delivery green. A
#: stall at or beyond 60 s still fails — and now fails with the state block below
#: rather than with a bare deadline.
DELIVERY_WAIT_S = 60.0
SENTINEL = "sk-live-QA-SENTINEL-9f31"


def _ask_args(question: str, *, timeout: int | str | None = None) -> dict[str, Any]:
    args: dict[str, Any] = {
        "questions": [
            {"id": "q0", "question": question, "options": [{"label": "yes"}, {"label": "no"}]}
        ]
    }
    if timeout is not None:
        args["timeout"] = timeout
    return args


def _secret_args(key: str = "API_KEY") -> dict[str, Any]:
    return {"questions": [{"id": key, "question": "Paste the key", "secret": True}]}


async def _poll_until(predicate, timeout_s: float) -> bool:
    """Poll ``predicate`` on a 20 ms cadence, bounded; ``True`` iff it became true.

    The one loop under both waits below. The queued path delivers from a spawned
    turn, so a test cannot await the delivery handle directly; polling on the
    OBSERVABLE (a transcript row, a provider request) is what keeps the assertion
    about the effect rather than about a task object. The predicate is re-checked
    once AFTER the deadline, because "it arrived just as we gave up" must not be
    reported as "it never arrived".
    """
    deadline = asyncio.get_running_loop().time() + timeout_s
    while asyncio.get_running_loop().time() < deadline:
        if predicate():
            return True
        await asyncio.sleep(0.02)
    return predicate()


async def _wait_until(predicate, *, timeout_s: float = 30.0, what: str = "condition") -> None:
    """Poll ``predicate`` until true, bounded and LOUD about what never happened.

    The plain backstop, for waits that are NOT about a row the runtime owes (see
    :func:`_wait_for_delivery` for the ones that are).
    """
    if not await _poll_until(predicate, timeout_s):
        raise AssertionError(f"timed out after {timeout_s}s waiting for {what}")


async def _wait_for_delivery(session: Session, predicate, *, what: str) -> None:
    """Wait for a row the runtime OWES, and on timeout say what the runtime held.

    THE CLASS THIS IS FOR: a predicate waiting for a response/timeout row after
    ``respond_ask`` / ``withdraw_ask`` / ``reconcile_asks``, where the row is
    produced by a spawned delivery turn. It uses :data:`DELIVERY_WAIT_S` and, on
    timeout, appends :func:`_delivery_state` — so a stall arrives with evidence
    instead of a bare deadline.

    The file's OTHER waits were audited against this class and classified: the
    waits whose predicate is a DOWNSTREAM provider request (the model's turn
    reading the report, not the row the runtime owes) are a different class and
    keep the plain :func:`_wait_until` backstop. The per-site verdicts are in the
    PR; the split is stated here so a later reader does not have to re-derive it.
    """
    started = asyncio.get_running_loop().time()
    if not await _poll_until(predicate, DELIVERY_WAIT_S):
        elapsed = asyncio.get_running_loop().time() - started
        raise AssertionError(
            f"timed out after {DELIVERY_WAIT_S}s waiting for {what}\n"
            + _delivery_state(session, predicate, elapsed=elapsed)
        )


def _delivery_task_names(limit: int = 8) -> list[str]:
    """Names of pending tasks on the delivery paths, bounded.

    ``ask`` is a substring of asyncio's own default task name (``Task-7``), so those
    default names are dropped explicitly: a line listing every unnamed task would be
    noise, and this line exists to show a DELIVERY task still pending — the stall
    signature — or the absence of one.
    """
    names: set[str] = set()
    for task in asyncio.all_tasks():
        if task is asyncio.current_task():
            continue
        name = task.get_name()
        if name.startswith("Task-") and name[len("Task-") :].isdigit():
            continue
        if any(token in name for token in ("deliver", "prompt", "ask")):
            names.add(name)
    return sorted(names)[:limit]


def _delivery_state(session: Session, predicate, *, elapsed: float) -> str:
    """A BOUNDED snapshot of what the runtime owed when a delivery wait timed out.

    Every part is bounded so the block cannot itself become the failure, and each
    line answers a question the bare deadline could not: did the row arrive just as
    the wait gave up; did the ask FOLD without delivering; is the transcript's tail
    still moving; and is a delivery task still pending. That last one is the stall
    SIGNATURE the ask-gate lane asked for — an event-loop stall leaves the delivery
    task alive and unfinished, where a dropped turn leaves it gone.

    Read-only, and deliberately outside the assertion it annotates: a diagnostic
    that raised would replace the real failure with its own.
    """

    def _safe(read, default):
        try:
            return read()
        except Exception:  # noqa: BLE001 -- a diagnostic must never replace the failure
            return default

    lines = ["--- delivery state at timeout ---"]
    lines.append(f"elapsed: {elapsed:.1f}s (backstop {DELIVERY_WAIT_S}s)")
    # 1. The predicate, re-checked: the wait can give up between polls, and "it
    #    arrived as we gave up" (a too-tight backstop) is not a stall.
    lines.append(f"predicate at re-check: {bool(_safe(predicate, False))}")
    # 2. The fold, for the ids this session's log holds (last 4, bounded).
    ids = _safe(lambda: _ask_ids(session.transcript.directory)[-4:], [])
    fold = _safe(
        lambda: [(ask_id, session.ask_queue().find(ask_id)) for ask_id in ids],
        [],
    )
    lines.append(
        "asks (status, delivered): "
        + (
            "; ".join(
                f"{ask_id}={row.get('status') if row else None},"
                f"{row.get('delivered') if row else None}"
                for ask_id, row in fold
            )
            or "(none)"
        )
    )
    # 3. The transcript tail (bounded, ids and kinds only — never message text).
    tail = _safe(lambda: session.transcript.entries()[-8:], [])
    lines.append(
        "transcript tail: " + ("; ".join(f"{entry.id}:{entry.type}" for entry in tail) or "(empty)")
    )
    # 4. Pending tasks on the delivery paths (bounded), the stall signature.
    lines.append(
        "pending tasks (deliver|prompt|ask): "
        + (", ".join(_safe(_delivery_task_names, [])) or "(none)")
    )
    return "\n".join(lines)


def _ask_ids(directory: Path) -> list[str]:
    return store.ask_ids(store.read_events(directory))


@pytest.fixture(autouse=True)
def _the_blocking_arm_is_selected_explicitly(monkeypatch: pytest.MonkeyPatch) -> None:
    """Start every cell on the KILL-SWITCH arm, and say why out loud.

    These cells pin BOTH arms of one feature. The queued one is the shipped
    default (2026-10-03), so the arm a cell means is no longer implied by the
    process default and must be stated — this fixture states it for the file,
    and the queued cells override it by name (``monkeypatch.setattr(... True)``).
    Autouse rather than per-cell so a blocking cell added later cannot forget to
    select the arm it is named for and quietly test the default instead.
    """
    monkeypatch.setattr(policy, "NONBLOCKING_ASK", False)


# ---------------------------------------------------------------------------
# The flag-off invariant
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_with_the_kill_switch_the_tool_still_awaits_and_writes_no_log(
    headless_tui_env: Path,
) -> None:
    """The kill switch is real: with ``NONBLOCKING_ASK`` off the model's ``ask``
    call blocks on the host hook exactly as it did before the queue shipped,
    and no ask log exists afterwards."""
    answered: list[str] = []

    async def hook(questions: list[Any]) -> dict[str, list[str]] | None:
        answered.append(questions[0].question)
        return {questions[0].id: ["yes"]}

    stream = ScriptedStream(
        [
            tool_call_turn(
                text="Asking.",
                tool_name="ask",
                tool_call_id="ask-1",
                arguments=_ask_args("Still blocking?"),
            ),
            text_turn("done"),
        ]
    )
    session = _session(headless_tui_env, "flag-off", stream)
    session.set_ask_handler(hook)
    try:
        with bounded(BOUND_S, "flag-off ask still blocks"):
            await session.prompt("go")
            assert answered == ["Still blocking?"], "the hook was not awaited"
            assert not store.asks_log_path(session.transcript.directory).exists()
            assert store.ask_ids(store.read_events(session.transcript.directory)) == []
            # And the AGENT-SIDE SETTLE is not mounted either: with no queue
            # there is no log that could hold a `withdrawn` row, so the tool
            # must be absent from the provider array (design §12's kill switch —
            # an absence, not a tool that can only error).
            advertised = {tool.name for tool in stream.requests[-1].tools}
            assert "ask_withdraw" not in advertised, sorted(advertised)
    finally:
        await dispose_quietly(session)


# ---------------------------------------------------------------------------
# Queued, accumulating, answered out of order
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_three_asks_accumulate_while_the_agent_works_and_answer_out_of_order(
    headless_tui_env: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(policy, "NONBLOCKING_ASK", True)
    hook_calls: list[str] = []

    async def hook(questions: list[Any]) -> dict[str, list[str]] | None:
        hook_calls.append(questions[0].question)
        return None

    turns: list[list[Any]] = []
    for index in range(3):
        turns.append(
            tool_call_turn(
                text=f"queuing {index}",
                tool_name="ask",
                tool_call_id=f"ask-{index}",
                arguments=_ask_args(f"question {index}"),
            )
        )
        turns.append(text_turn("continuing with other work"))
    # Three response turns: one per answer, each a separate paid turn.
    turns.extend([text_turn("noted a"), text_turn("noted b"), text_turn("noted c")])
    stream = ScriptedStream(turns)
    session = _session(headless_tui_env, "accumulate", stream)
    session.set_ask_handler(hook)
    try:
        with bounded(BOUND_S, "three queued asks answered out of order"):
            for index in range(3):
                await session.prompt(f"ask {index}")
            assert hook_calls == [], "the queued path must not await a human"
            ask_ids = _ask_ids(session.transcript.directory)
            assert len(ask_ids) == 3, ask_ids
            assert len(session.ask_queue().open_records()) == 3

            # Out of order: the last ask first.
            for ask_id in (ask_ids[2], ask_ids[0], ask_ids[1]):
                outcome = session.respond_ask(ask_id, {"q0": ["yes"]}, by="terminal")
                assert outcome["ok"] is True, outcome
                await session.reconcile_asks()

            await _wait_for_delivery(
                session,
                lambda: all(
                    session.transcript.has_entry(store.response_row_id(ask_id))
                    for ask_id in ask_ids
                ),
                what="one ask_response row per ask",
            )
            rows = [entry for entry in session.transcript.entries() if entry.id]
            for ask_id in ask_ids:
                assert sum(1 for e in rows if e.id == store.response_row_id(ask_id)) == 1
                assert not session.transcript.has_entry(store.timeout_row_id(ask_id))
    finally:
        await dispose_quietly(session)


@pytest.mark.asyncio
async def test_a_multi_question_ask_is_answered_atomically(
    headless_tui_env: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(policy, "NONBLOCKING_ASK", True)

    async def hook(questions: list[Any]) -> dict[str, list[str]] | None:
        raise AssertionError("the queued path must not call the host hook")

    stream = ScriptedStream(
        [
            tool_call_turn(
                text="two questions",
                tool_name="ask",
                tool_call_id="ask-multi",
                arguments={
                    "questions": [
                        {
                            "id": "q0",
                            "question": "first?",
                            "options": [{"label": "a"}, {"label": "b"}],
                        },
                        {
                            "id": "q1",
                            "question": "second?",
                            "options": [{"label": "c"}, {"label": "d"}],
                        },
                    ]
                },
            ),
            text_turn("working"),
            text_turn("both answers in hand"),
        ]
    )
    session = _session(headless_tui_env, "multi", stream)
    session.set_ask_handler(hook)
    try:
        with bounded(BOUND_S, "a multi-question ask answered atomically"):
            await session.prompt("go")
            (ask_id,) = _ask_ids(session.transcript.directory)
            outcome = session.respond_ask(ask_id, {"q0": ["a"], "q1": ["c"]}, by="terminal")
            assert outcome["ok"] is True
            await session.reconcile_asks()
            await _wait_for_delivery(
                session,
                lambda: session.transcript.has_entry(store.response_row_id(ask_id)),
                what="the response row",
            )
            record = session.ask_queue().find(ask_id)
            assert record["answers"] == {"q0": ["a"], "q1": ["c"]}
            # The model read both, from ONE row.
            last = stream.requests[-1]
            text = "\n".join((message.text or "") for message in last.messages)
            assert "first?" in text and "second?" in text
    finally:
        await dispose_quietly(session)


# ---------------------------------------------------------------------------
# The deadline
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_a_deadline_delivers_a_notice_into_the_model_context(
    headless_tui_env: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A normal (non-urgent) timeout: one ``ask_timeout`` row, delivered even
    though the session was idle, and the model's next request carries it."""
    monkeypatch.setattr(policy, "NONBLOCKING_ASK", True)
    stream = ScriptedStream(
        [
            tool_call_turn(
                text="asking",
                tool_name="ask",
                tool_call_id="ask-t",
                arguments=_ask_args("expires soon?", timeout=600),
            ),
            text_turn("carrying on"),
            text_turn("the deadline passed, so I proceeded"),
        ]
    )
    session = _session(headless_tui_env, "timeout", stream)
    session.set_ask_handler(_never_answers)
    try:
        with bounded(BOUND_S, "a deadline notice reaching the model"):
            await session.prompt("go")
            (ask_id,) = _ask_ids(session.transcript.directory)
            queue = session.ask_queue()
            # Inject the clock rather than sleeping two minutes (the floor).
            # The origin is read BEFORE the clock is replaced: a lambda that
            # called back into the queue would recurse.
            started_at = int(queue.find(ask_id)["created_at"])
            queue._now = lambda: started_at + 700_000
            await session.reconcile_asks()
            await _wait_for_delivery(
                session,
                lambda: session.transcript.has_entry(store.timeout_row_id(ask_id)),
                what="the ask_timeout row",
            )
            await _wait_until(
                lambda: any(
                    "[Ask timed out]" in (message.text or "")
                    for request in stream.requests
                    for message in request.messages
                ),
                what="the notice in a provider request",
            )
            # The ask is still answerable, and a late answer is attributed.
            assert queue.find(ask_id)["status"] == store.STATUS_TIMED_OUT
            outcome = session.respond_ask(ask_id, {"q0": ["yes"]}, by="phone")
            assert outcome["ok"] is True, outcome
            await session.reconcile_asks()
            await _wait_for_delivery(
                session,
                lambda: session.transcript.has_entry(store.response_row_id(ask_id)),
                what="the late response row",
            )
            record = queue.find(ask_id)
            assert record["status"] == store.STATUS_LATE
    finally:
        await dispose_quietly(session)


async def _never_answers(questions: list[Any]) -> dict[str, list[str]] | None:
    raise AssertionError("the queued path must not call the host hook")


# ---------------------------------------------------------------------------
# The agent-side settle (design §12): withdraw + chat-answer attribution
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_a_moot_withdraw_through_the_tool_settles_the_ask_and_injects_nothing(
    headless_tui_env: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Design §12's moot half over the assembled runtime, driven through the
    REAL tool loop: the receipt's ask is withdrawn by a second tool call, the
    fold settles it, the wire's outstanding count drops by itself — and NO row
    is ever injected for it (symmetric with `dismissed`)."""
    monkeypatch.setattr(policy, "NONBLOCKING_ASK", True)
    stream = ScriptedStream(
        [
            tool_call_turn(
                text="asking",
                tool_name="ask",
                tool_call_id="ask-w",
                arguments=_ask_args("still needed?"),
            ),
            text_turn("working"),
        ]
    )
    session = _session(headless_tui_env, "withdraw-moot", stream)
    session.set_ask_handler(_never_answers)
    try:
        with bounded(BOUND_S, "a moot withdraw through the tool loop"):
            await session.prompt("go")
            (ask_id,) = _ask_ids(session.transcript.directory)
            # The tool is advertised exactly where the queue is, asserted on the
            # provider's own array (the #868 lesson: the gate and the wire can
            # disagree even when both halves are individually right).
            advertised = {tool.name for tool in stream.requests[-1].tools}
            assert "ask_withdraw" in advertised, sorted(advertised)
            # The second turn withdraws the ask the FIRST turn queued.
            stream.turns.append(
                tool_call_turn(
                    text="closing it",
                    tool_name="ask_withdraw",
                    tool_call_id="withdraw-1",
                    arguments={"ask_id": ask_id, "reason": "moot"},
                )
            )
            stream.turns.append(text_turn("noted"))
            await session.prompt("actually, never mind")
            record = session.ask_queue().find(ask_id)
            assert record["status"] == store.STATUS_WITHDRAWN
            assert record["delivered"] is False
            # Nothing is injected, now or on the next reconcile — the ask has
            # no expected row at all.
            await session.reconcile_asks()
            assert not session.transcript.has_entry(store.response_row_id(ask_id))
            assert not session.transcript.has_entry(store.timeout_row_id(ask_id))
            # The wire the bar, the sidebar and every list read: settled, and
            # the outstanding count is zero.
            from local_operator.session.frontend_state import ask_wire

            rows, outstanding = ask_wire(session)
            assert rows is not None
            assert outstanding == 0
            assert [row["status"] for row in rows] == [store.STATUS_WITHDRAWN]
            # The receipt reached the model in its own words, not a paraphrase.
            follow_up = "\n".join((message.text or "") for message in stream.requests[-1].messages)
            assert "withdrawn" in follow_up and "nothing will be delivered" in follow_up
    finally:
        await dispose_quietly(session)


@pytest.mark.asyncio
async def test_answered_in_chat_records_the_users_words_and_delivers_the_response(
    headless_tui_env: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Design §12's chat half over the assembled runtime: the user replies in
    the transcript, the model records their words via the tool, and BOTH the
    `answered` fold and the standard response row land — the response arriving
    as its own paid turn, exactly as a card answer would cost."""
    monkeypatch.setattr(policy, "NONBLOCKING_ASK", True)
    stream = ScriptedStream(
        [
            tool_call_turn(
                text="asking",
                tool_name="ask",
                tool_call_id="ask-c",
                arguments=_ask_args("deploy or roll back?"),
            ),
            text_turn("working"),
        ]
    )
    session = _session(headless_tui_env, "withdraw-chat", stream)
    session.set_ask_handler(_never_answers)
    try:
        with bounded(BOUND_S, "a chat answer recorded through the tool loop"):
            await session.prompt("go")
            (ask_id,) = _ask_ids(session.transcript.directory)
            stream.turns.append(
                tool_call_turn(
                    text="recording the reply",
                    tool_name="ask_withdraw",
                    tool_call_id="withdraw-1",
                    arguments={
                        "ask_id": ask_id,
                        "reason": "answered_in_chat",
                        "answers": {"q0": ["roll back, keep the audit log"]},
                    },
                )
            )
            stream.turns.append(text_turn("recorded"))
            # Spare turns: the response row this settle records buys its own
            # delivery turn, and a short tape is how a cell starts lying.
            stream.turns.append(text_turn("noted"))
            stream.turns.append(text_turn("noted again"))
            await session.prompt("roll back, keep the audit log")
            record = session.ask_queue().find(ask_id)
            assert record["status"] == store.STATUS_ANSWERED
            assert record["answers"] == {"q0": ["roll back, keep the audit log"]}
            assert record["answered_by"] == {"surface": "chat"}
            # The standard response row delivers through decide-reconcile, and
            # the model reads it: the report's own framing, not the user's turn.
            await _wait_for_delivery(
                session,
                lambda: session.transcript.has_entry(store.response_row_id(ask_id)),
                what="the chat answer's response row",
            )
            await _wait_until(
                lambda: any(
                    "The user answered:" in (message.text or "")
                    for request in stream.requests
                    for message in request.messages
                ),
                what="the response report in a provider request",
            )
            # The wire settles: answered and delivered, nothing outstanding.
            from local_operator.session.frontend_state import ask_wire

            rows, outstanding = ask_wire(session)
            assert rows is not None
            assert outstanding == 0
            assert rows[-1]["status"] == store.STATUS_ANSWERED
            assert rows[-1]["delivered"] is True
    finally:
        await dispose_quietly(session)


@pytest.mark.asyncio
async def test_an_answer_racing_a_withdrawal_wins_in_both_orders(
    headless_tui_env: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Contract (a) over the real runtime, from both directions:

    * the answer lands first — the racing withdrawal is REFUSED and writes no
      ``withdrawn`` row, so the ask folds `answered` and its response delivers;
    * the withdrawal lands first — a stale answer path is refused with the
      STATE'S sentence — while an answered row that still lands (a write that
      crossed the guard, appended directly as the cross-process gap would)
      STILL WINS: the fold says `answered` and the response delivers.
    """
    monkeypatch.setattr(policy, "NONBLOCKING_ASK", True)
    stream = ScriptedStream(
        [
            tool_call_turn(
                text="first",
                tool_name="ask",
                tool_call_id="ask-a",
                arguments=_ask_args("first question"),
            ),
            tool_call_turn(
                text="second",
                tool_name="ask",
                tool_call_id="ask-b",
                arguments=_ask_args("second question"),
            ),
            text_turn("both queued"),
            # Spares: each delivered response buys its own turn.
            text_turn("noted"),
            text_turn("noted"),
            text_turn("noted"),
        ]
    )
    session = _session(headless_tui_env, "race", stream)
    session.set_ask_handler(_never_answers)
    try:
        with bounded(BOUND_S, "an answer racing a withdrawal"):
            await session.prompt("ask both")
            first, second = _ask_ids(session.transcript.directory)

            # Direction 1: answer first — the withdrawal loses and writes NOTHING.
            assert session.respond_ask(first, {"q0": ["yes"]}, by="terminal")["ok"] is True
            refused = session.withdraw_ask(first, reason="moot")
            assert refused["ok"] is False
            assert "already has the user's answer" in refused["error"]
            kinds = [event["kind"] for event in store.read_events(session.transcript.directory)]
            assert kinds.count(store.EVENT_WITHDRAWN) == 0
            await session.reconcile_asks()
            await _wait_for_delivery(
                session,
                lambda: session.transcript.has_entry(store.response_row_id(first)),
                what="the winning answer's response row",
            )
            assert session.ask_queue().find(first)["status"] == store.STATUS_ANSWERED

            # Direction 2: the withdrawal lands first...
            assert session.withdraw_ask(second, reason="moot")["ok"] is True
            stale = session.respond_ask(second, {"q0": ["a real answer"]})
            assert stale["ok"] is False
            assert stale["error"] == (
                "the agent withdrew this question — if you have an answer, send it "
                "as a chat message."
            )
            # ...and an answered row that crossed the guard anyway still WINS at
            # the fold (contract (a)'s belt-and-braces half; appended directly
            # because no real path exists in-process to defeat its own check).
            store.append_event(
                session.transcript.directory,
                {
                    "v": store.EVENT_SCHEMA,
                    "kind": store.EVENT_ANSWERED,
                    "ask_id": second,
                    "at": store.now_ms(),
                    "by": {"surface": "phone"},
                    "answers": {"q0": ["late to the race"]},
                },
            )
            assert session.ask_queue().find(second)["status"] == store.STATUS_ANSWERED
            await session.reconcile_asks()
            await _wait_for_delivery(
                session,
                lambda: session.transcript.has_entry(store.response_row_id(second)),
                what="the racing answer's response row",
            )
    finally:
        await dispose_quietly(session)


# ---------------------------------------------------------------------------
# Restart durability
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_an_answer_recorded_before_the_runtime_died_is_delivered_once_on_boot(
    headless_tui_env: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Answer-then-kill: the log holds the answer, the transcript holds no row,
    and the NEXT runtime's boot reconcile delivers it — exactly once."""
    monkeypatch.setattr(policy, "NONBLOCKING_ASK", True)
    directory = headless_tui_env / "sessions" / "restart"
    first_stream = ScriptedStream(
        [
            tool_call_turn(
                text="asking",
                tool_name="ask",
                tool_call_id="ask-r",
                arguments=_ask_args("survive the kill?"),
            ),
            text_turn("working"),
        ]
    )
    first = _session(headless_tui_env, "restart", first_stream)
    first.set_ask_handler(_never_answers)
    try:
        with bounded(BOUND_S, "answer survives a runtime death"):
            await first.prompt("go")
            (ask_id,) = _ask_ids(directory)
            # The answer is recorded, then the runtime dies before delivering.
            session_queue = first.ask_queue()
            assert session_queue.respond(ask_id, {"q0": ["yes"]}, by="phone")["ok"] is True
            assert not first.transcript.has_entry(store.response_row_id(ask_id))
    finally:
        await dispose_quietly(first)

    second_stream = ScriptedStream([text_turn("I have the answer now")])
    second = _session(headless_tui_env, "restart", second_stream)
    second.set_ask_handler(_never_answers)
    try:
        with bounded(BOUND_S, "boot reconcile delivers once"):
            await second.reconcile_asks()
            await _wait_for_delivery(
                second,
                lambda: second.transcript.has_entry(store.response_row_id(ask_id)),
                what="the delivered response row",
            )
            # Delivering again must write nothing new.
            await second.reconcile_asks()
            rows = [e for e in second.transcript.entries() if e.id == store.response_row_id(ask_id)]
            assert len(rows) == 1
            record = second.ask_queue().find(ask_id)
            assert record["status"] == store.STATUS_ANSWERED
            assert record["delivered"] is True
    finally:
        await dispose_quietly(second)


@pytest.mark.asyncio
async def test_an_unanswered_ask_outlives_the_runtime_and_stays_open(
    headless_tui_env: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """SIGKILL-equivalent: the next runtime re-reads the log and the ask is still
    there, still open, with its deadline untouched."""
    monkeypatch.setattr(policy, "NONBLOCKING_ASK", True)
    directory = headless_tui_env / "sessions" / "survive"
    stream = ScriptedStream(
        [
            tool_call_turn(
                text="asking",
                tool_name="ask",
                tool_call_id="ask-s",
                arguments=_ask_args("still there?", timeout=1800),
            ),
            text_turn("working"),
        ]
    )
    session = _session(headless_tui_env, "survive", stream)
    session.set_ask_handler(_never_answers)
    try:
        with bounded(BOUND_S, "an open ask outlives its runtime"):
            await session.prompt("go")
            (ask_id,) = _ask_ids(directory)
    finally:
        await dispose_quietly(session)

    revived = _session(headless_tui_env, "survive", ScriptedStream([]))
    revived.set_ask_handler(_never_answers)
    try:
        with bounded(BOUND_S, "the ask is re-read from disk"):
            record = revived.ask_queue().find(ask_id)
            assert record is not None
            assert record["status"] == store.STATUS_OPEN
            assert record["timeout_s"] == 1800
            # Boot reconcile writes nothing for an ask nobody has answered.
            await revived.reconcile_asks()
            assert revived.transcript.has_entry(store.timeout_row_id(ask_id)) is False
    finally:
        await dispose_quietly(revived)


# ---------------------------------------------------------------------------
# Secrets
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_a_secret_answer_never_reaches_disk(
    headless_tui_env: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The sentinel is grepped for across every artefact the queue writes, and
    the transcript names the KEY instead — which is the whole contract."""
    monkeypatch.setattr(policy, "NONBLOCKING_ASK", True)
    directory = headless_tui_env / "sessions" / "secret"
    stream = ScriptedStream(
        [
            tool_call_turn(
                text="need the key",
                tool_name="ask",
                tool_call_id="ask-k",
                arguments=_secret_args("API_KEY"),
            ),
            text_turn("working"),
            text_turn("key in hand"),
        ]
    )
    session = _session(headless_tui_env, "secret", stream)
    session.set_ask_handler(_never_answers)
    try:
        with bounded(BOUND_S, "a secret answer stays out of the log"):
            await session.prompt("go")
            (ask_id,) = _ask_ids(directory)
            outcome = session.respond_ask(ask_id, {"API_KEY": [SENTINEL]}, by="terminal")
            assert outcome["ok"] is True, outcome
            await session.reconcile_asks()
            await _wait_for_delivery(
                session,
                lambda: session.transcript.has_entry(store.response_row_id(ask_id)),
                what="the secret response row",
            )

            log = store.asks_log_path(directory).read_text()
            index = (store.entry_path(headless_tui_env, "secret")).read_text()
            transcript = (directory / "transcript.jsonl").read_text()
            for name, blob in (("asks.jsonl", log), ("index", index), ("transcript", transcript)):
                assert SENTINEL not in blob, f"the secret value reached {name}"
            # And the model was told the KEY NAME, not that nothing happened.
            assert "API_KEY" in log
            assert "API_KEY" in transcript
            record = session.ask_queue().find(ask_id)
            assert record["answers"] == {"API_KEY": ["API_KEY"]}
    finally:
        await dispose_quietly(session)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _session(config_dir: Path, name: str, stream: ScriptedStream) -> Session:
    return build_session(config_dir / "sessions" / name, stream, cwd=config_dir)


# ---------------------------------------------------------------------------
# The FLEET read (design §4/§11): what the TUI's fleet scope reads
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_the_aggregate_read_the_fleet_scope_uses_reports_a_real_sessions_ask(
    headless_tui_env: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The cross-session INDEX is the only source a fleet view has, so it has to
    carry a real session's ask — with the two facts a row for another
    conversation needs (``session_id``, ``cwd``) — and then carry its
    SETTLEMENT, because the TUI's halves and the sidebar's mark are computed
    from exactly this read and no wire."""
    monkeypatch.setattr(policy, "NONBLOCKING_ASK", True)
    directory = headless_tui_env / "sessions" / "fleet-read"
    stream = ScriptedStream(
        [
            tool_call_turn(
                text="asking",
                tool_name="ask",
                tool_call_id="ask-f",
                arguments=_ask_args("did the fleet see it?"),
            ),
            text_turn("working"),
        ]
    )
    session = _session(headless_tui_env, "fleet-read", stream)
    session.set_ask_handler(_never_answers)
    try:
        with bounded(BOUND_S, "the aggregate read sees a real session's ask"):
            await session.prompt("go")
            (ask_id,) = _ask_ids(directory)
            rows = store.index_asks(headless_tui_env)
            row = next(r for r in rows if r["ask_id"] == ask_id)
            assert row["session_id"] == "fleet-read"
            assert row["cwd"] == str(headless_tui_env)
            assert store.is_outstanding(row["status"])
            # The MARK predicate (open ∪ timed_out) agrees with the row.
            assert len(store.outstanding_asks(rows)) == 1
            # And the answer settles it, so the halves and the mark both move.
            assert session.ask_queue().respond(ask_id, {"q0": ["yes"]}, by="terminal")["ok"] is True
            await session.reconcile_asks()
            settled = next(r for r in store.index_asks(headless_tui_env) if r["ask_id"] == ask_id)
            assert settled["status"] == store.STATUS_ANSWERED
            # The DELIVERY is the transcript row, and it is asserted there rather
            # than on the index's `delivered` hint: the index is derived and is
            # rewritten on the owner's next publish, so it may lag one write —
            # which is exactly why the SESSION scope reads the wire first and the
            # fleet scope (no wire to read) is the one that falls back to it.
            await _wait_for_delivery(
                session,
                lambda: session.transcript.has_entry(store.response_row_id(ask_id)),
                what="the delivered response row for the fleet read",
            )
            assert store.outstanding_asks(store.index_asks(headless_tui_env)) == []
    finally:
        await dispose_quietly(session)


@pytest.mark.asyncio
async def test_a_fleet_scope_sees_a_stopped_sessions_ask_and_can_answer_it_by_id(
    headless_tui_env: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A row whose runtime is GONE is still readable and still answerable — the
    property the fleet scope is built on, asserted through the store the TUI
    reads: the index keeps the row after the runtime is disposed, and the ask
    log still accepts the answer, which the next runtime's reconcile delivers."""
    monkeypatch.setattr(policy, "NONBLOCKING_ASK", True)
    directory = headless_tui_env / "sessions" / "fleet-cold"
    stream = ScriptedStream(
        [
            tool_call_turn(
                text="asking",
                tool_name="ask",
                tool_call_id="ask-c",
                arguments=_ask_args("answer me later?"),
            ),
            text_turn("working"),
        ]
    )
    session = _session(headless_tui_env, "fleet-cold", stream)
    session.set_ask_handler(_never_answers)
    with bounded(BOUND_S, "a stopped session's ask stays in the fleet read"):
        await session.prompt("go")
        (ask_id,) = _ask_ids(directory)
    await dispose_quietly(session)

    # The runtime is disposed; the row the fleet scope would draw is still there,
    # with nothing live behind it.
    rows = store.index_asks(headless_tui_env)
    assert [r["ask_id"] for r in rows] == [ask_id]
    assert rows[0]["session_id"] == "fleet-cold"

    # Answering by ask_id settles it at the LOG (the TUI's fleet path sends the
    # op through an engaged runtime; the durability underneath is this).
    assert (
        store.append_event(
            directory,
            {
                "kind": store.EVENT_ANSWERED,
                "v": store.EVENT_SCHEMA,
                "ask_id": ask_id,
                "at": store.now_ms(),
                "answers": {"q0": ["yes"]},
                "by": {"surface": "terminal"},
            },
        )
        is True
    )
    revived = _session(headless_tui_env, "fleet-cold", ScriptedStream([text_turn("got it")]))
    revived.set_ask_handler(_never_answers)
    try:
        with bounded(BOUND_S, "the cold answer is delivered on boot"):
            await revived.reconcile_asks()
            await _wait_for_delivery(
                revived,
                lambda: revived.transcript.has_entry(store.response_row_id(ask_id)),
                what="the delivered response row for a cold fleet answer",
            )
    finally:
        await dispose_quietly(revived)
