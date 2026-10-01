"""One bounded retry on the answer gate, and the typed refusal behind it.

The two halves of the desktop answer path's lost-acknowledgement handling, tested
where they live rather than through the wire:

* ``AttachedSession.answer_gate`` re-issues ONCE under the same id when the owner
  did not acknowledge, and reads the gate's OWN state to decide whether the
  retry's refusal means "the owner settled it" (a receipt) or "the owner is not
  answering" (a transport failure). This is the only implementation of the
  ``answer_gate`` protocol method, so the desktop route, the mobile relay and any
  attached viewer all inherit it.
* ``AttachedSession.attach_existing`` refuses TYPED when a control acquire ends
  unsynced, instead of reporting a retained dial as a served call and leaving the
  caller to trip over ``frontend_state`` one line later.

The fake client stands in for the socket at its boundary (``connected`` plus the
two ops); everything above it — the pending-gate identity check, the retry
budget, the receipt vocabulary, the acquire's own branches — is the real code.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from local_operator.mobile.attach_client import OwnerAckTimeout
from local_operator.session.attached import AttachedSession, RuntimeUnresponsiveError


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
    """A client whose op results are scripted, in order, per call."""

    def __init__(self, results: list[object]) -> None:
        self.connected = True
        self.results = list(results)
        self.calls: list[tuple[str, object]] = []

    async def approval_answer(self, request_id: str, approved: bool) -> str:
        self.calls.append((request_id, approved))
        result = self.results.pop(0)
        if isinstance(result, BaseException):
            raise result
        return str(result)


def _viewer(tmp_path: Path, store: _Store | None, client: _Client | None) -> AttachedSession:
    viewer = AttachedSession(config_dir=tmp_path, session_id="s1", takeover_factory=_no_takeover)
    # The two collaborators are replaced with scripted stand-ins, the convention
    # the other facade tests use (``test_liveness_reader.py``, ``test_desktop_sessions.py``).
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


@pytest.mark.asyncio
async def test_a_retry_that_finds_the_gate_gone_reports_settled(tmp_path: Path) -> None:
    """The first attempt SETTLED it; the lost ack is not the operator's problem.

    The second refusal is modelled the way the owner sends it (an error frame the
    client raises for): by then the gate is gone from the facade's state, which is
    the fact the retry reads rather than the exception's class.
    """
    store = _Store(_Gate())

    class _SettlingClient(_Client):
        async def approval_answer(self, request_id: str, approved: bool) -> str:
            self.calls.append((request_id, approved))
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
async def test_a_control_acquire_in_the_retained_unsynced_window_refuses_typed(
    tmp_path: Path,
) -> None:
    """The read's retained dial is not a served control call.

    The window: an authenticated socket is up (``owner_reachable``), its canonical
    state has NOT landed (``_frontend_store is None``) and there is no landing task
    to adopt. Before, this arm returned success and the caller read
    ``frontend_state`` one line later, whose ``RuntimeError`` the ladder answered
    as a generic unreachable refusal.
    """
    client = _Client([])
    viewer = _viewer(tmp_path, None, client)
    viewer._socketed_unsynced = True

    with pytest.raises(RuntimeUnresponsiveError):
        await viewer.attach_existing(control_budget=0.05)

    assert viewer.owner_reachable, "the window under test really was the retained one"
