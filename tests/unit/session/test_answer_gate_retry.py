"""One bounded retry on the answer gate, inside the caller's own budget.

``AttachedSession.answer_gate`` is the only implementation of the ``answer_gate``
protocol method, so the desktop route, the mobile relay and any attached viewer
inherit its behaviour. Two things are pinned here:

* the retry itself — ONE more attempt under the same id when the owner did not
  acknowledge, with the gate's OWN state deciding whether that attempt's refusal
  means "the owner settled it" (a receipt) or "the owner is not answering" (a
  transport failure); and
* the BUDGET — when the caller passes a deadline, both attempts share it, so the
  answer route cannot outlive the renderer's per-op deadline and deliver a
  disposition to nobody (agent review round 1, MAJOR-1).

The fake client stands in for the socket at its boundary (``connected`` plus the
op, recording the envelope it was given); everything above it — the pending-gate
identity check, the budget arithmetic, the receipt vocabulary — is the real code.
"""

from __future__ import annotations

import asyncio
import time
from pathlib import Path

import pytest

from local_operator.mobile.attach_client import ACK_TIMEOUT_S, OwnerAckTimeout
from local_operator.session.attached import (
    _ANSWER_ATTEMPT_FLOOR_S,
    _ANSWER_DEADLINE_MARGIN_S,
    _ANSWER_RETRY_RESERVE_S,
    AttachedSession,
)


def _no_takeover() -> None:
    raise AssertionError("a viewer must never become the runtime")


class _Gate:
    """The facade's view of one pending gate (``PendingRequest``'s three fields)."""

    def __init__(self, request_id: str = "gate-1", kind: str = "approval") -> None:
        self.request_id = request_id
        self.kind = kind
        self.question_index = None


class _Store:
    """Just enough of ``FrontendStateStore`` for ``pending_gate`` to answer."""

    def __init__(self, gate: _Gate | None) -> None:
        self.gate = gate
        self.state = None

    @property
    def pending_gate(self) -> _Gate | None:
        return self.gate


class _Client:
    """A client whose op results are scripted, in order, per call.

    ``budgets`` records the acknowledgement envelope each attempt was given, which
    is how the budget tests read the arithmetic without waiting 15 s for it.
    """

    def __init__(self, results: list[object]) -> None:
        self.connected = True
        self.results = list(results)
        self.calls: list[tuple[str, object]] = []
        self.budgets: list[float] = []

    async def approval_answer(
        self, request_id: str, approved: bool, *, deadline_s: float = ACK_TIMEOUT_S
    ) -> str:
        self.calls.append((request_id, approved))
        self.budgets.append(deadline_s)
        result = self.results.pop(0)
        if isinstance(result, BaseException):
            raise result
        return str(result)


def _viewer(tmp_path: Path, store: _Store | None, client: _Client | None) -> AttachedSession:
    viewer = AttachedSession(config_dir=tmp_path, session_id="s1", takeover_factory=_no_takeover)
    # The collaborators are replaced with scripted stand-ins, the convention the
    # other facade tests use (``test_liveness_reader.py``, ``test_desktop_sessions.py``).
    viewer._frontend_store = store  # type: ignore[assignment]
    viewer._client = client  # type: ignore[assignment]
    return viewer


@pytest.mark.asyncio
async def test_a_lost_ack_is_retried_once_and_the_answer_lands(tmp_path: Path) -> None:
    """The retry is what makes the first tap's outcome visible to the operator."""
    client = _Client([OwnerAckTimeout("no ack"), "approved"])
    viewer = _viewer(tmp_path, _Store(_Gate()), client)

    receipt = await viewer.answer_gate("gate-1", approved=True)

    assert receipt == "approved"
    assert client.calls == [("gate-1", True), ("gate-1", True)], "not exactly one retry"
    # No deadline from this caller: each attempt keeps the client's own envelope.
    assert client.budgets == [ACK_TIMEOUT_S, ACK_TIMEOUT_S]


@pytest.mark.asyncio
async def test_a_retry_that_finds_the_gate_gone_reports_settled(tmp_path: Path) -> None:
    """The first attempt SETTLED it; the lost ack is not the operator's problem.

    The second refusal is modelled the way the owner sends it (an error frame the
    client raises for): by then the gate is gone from the facade's state, which is
    the fact the retry reads rather than the exception's class.
    """
    store = _Store(_Gate())

    class _SettlingClient(_Client):
        async def approval_answer(
            self, request_id: str, approved: bool, *, deadline_s: float = ACK_TIMEOUT_S
        ) -> str:
            self.calls.append((request_id, approved))
            self.budgets.append(deadline_s)
            if len(self.calls) == 1:
                # The ack is lost while the gate is still pending here, which is
                # what buys the retry its one attempt.
                raise OwnerAckTimeout("no ack")
            # The settle, arriving on the same pump the lost ack would have used,
            # together with the refusal an owner without an idempotent settle
            # sends for the repeat.
            store.gate = None
            raise RuntimeError("that question was already answered")

    client = _SettlingClient([])
    viewer = _viewer(tmp_path, store, client)

    receipt = await viewer.answer_gate("gate-1", approved=True)

    assert receipt == "approved"
    assert len(client.calls) == 2


@pytest.mark.asyncio
async def test_an_owner_that_never_answers_still_fails(tmp_path: Path) -> None:
    """A retry is not a substitute for a reachable owner: two timeouts still raise.

    The typed error is what the route ladder turns into the RETRYABLE refusal, so
    raising it (rather than inventing a receipt) is the whole point.
    """
    client = _Client([OwnerAckTimeout("no ack"), OwnerAckTimeout("no ack again")])
    viewer = _viewer(tmp_path, _Store(_Gate()), client)

    with pytest.raises(OwnerAckTimeout):
        await viewer.answer_gate("gate-1", approved=True)

    assert len(client.calls) == 2, "the retry is bounded to one"


@pytest.mark.asyncio
async def test_the_two_attempts_share_the_callers_budget(tmp_path: Path) -> None:
    """Neither attempt may claim a fresh envelope when the caller has a window.

    This is the arithmetic behind MAJOR-1: two 15 s envelopes plus the control
    attach is ~33 s against a 20 s client deadline, so the refusal the retry
    exists to deliver arrived after the client had stopped listening.
    """
    client = _Client([OwnerAckTimeout("no ack"), OwnerAckTimeout("no ack again")])
    viewer = _viewer(tmp_path, _Store(_Gate()), client)

    with pytest.raises(OwnerAckTimeout):
        await viewer.answer_gate("gate-1", approved=True, deadline=time.monotonic() + 4.0)

    first, second = client.budgets
    assert first < ACK_TIMEOUT_S, "the first attempt did not yield to the budget"
    assert first <= 4.0 - _ANSWER_RETRY_RESERVE_S, first
    assert second < ACK_TIMEOUT_S
    assert second <= 4.0, second


@pytest.mark.asyncio
async def test_a_shortened_retry_still_catches_the_lost_ack(tmp_path: Path) -> None:
    """The retry keeps its job under a budget: it is shorter, not skipped.

    The owner here answers the repeat at once — its settle is idempotent and it
    still has the value — which is exactly why a two-second reserve is enough for
    the common case even though the first attempt gets the rest of the window.
    """
    client = _Client([OwnerAckTimeout("no ack"), "approved"])
    viewer = _viewer(tmp_path, _Store(_Gate()), client)

    receipt = await viewer.answer_gate("gate-1", approved=True, deadline=time.monotonic() + 4.0)

    assert receipt == "approved"
    assert len(client.calls) == 2
    # Neither attempt may exceed the window; the first is the one that yields.
    first, second = client.budgets
    assert first <= 4.0 - _ANSWER_RETRY_RESERVE_S, first
    assert 0 < second <= 4.0, second


@pytest.mark.asyncio
async def test_no_retry_once_the_budget_is_spent(tmp_path: Path) -> None:
    """Too little left to be worth issuing: the first attempt's timeout travels.

    Spending the last of the window on a doomed second envelope is what pushes the
    response past the caller's deadline, where the disposition is lost entirely.
    """
    client = _Client([OwnerAckTimeout("no ack")])
    viewer = _viewer(tmp_path, _Store(_Gate()), client)
    # Below the retry's floor once the margin is taken out.
    deadline = time.monotonic() + _ANSWER_DEADLINE_MARGIN_S + 0.5

    with pytest.raises(OwnerAckTimeout):
        await viewer.answer_gate("gate-1", approved=True, deadline=deadline)

    assert len(client.calls) == 1, "a spent budget must not buy a second attempt"


@pytest.mark.asyncio
async def test_the_whole_call_returns_inside_the_deadline(tmp_path: Path) -> None:
    """The end of the argument: a real clock, a real overrun, no overrun of the budget.

    The client above answers instantly, so it can only prove the ARITHMETIC. This
    one actually spends its envelope before timing out, twice, which is the shape
    the route meets — and the assertion is on elapsed wall time, because "the
    backend answered before the caller gave up" is a fact about the clock and
    nothing else.
    """
    budget = 1.0

    class _SlowClient(_Client):
        async def approval_answer(
            self, request_id: str, approved: bool, *, deadline_s: float = ACK_TIMEOUT_S
        ) -> str:
            self.calls.append((request_id, approved))
            self.budgets.append(deadline_s)
            await asyncio.sleep(deadline_s)
            raise OwnerAckTimeout(f"no ack within {deadline_s:g}s")

    client = _SlowClient([])
    viewer = _viewer(tmp_path, _Store(_Gate()), client)

    started = time.monotonic()
    with pytest.raises(OwnerAckTimeout):
        await viewer.answer_gate("gate-1", approved=True, deadline=started + budget)
    elapsed = time.monotonic() - started

    assert elapsed <= budget + 0.3, f"the call outlived its own budget by {elapsed - budget:.2f}s"
    assert len(client.calls) <= 2
    assert min(client.budgets) >= _ANSWER_ATTEMPT_FLOOR_S
