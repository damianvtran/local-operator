"""``AttachedSession.interrupt`` — the awaiting twin of the fire-and-forget abort.

The desktop route needs one thing the existing facade op cannot give it: the
RUNTIME's own sentence describing what actually settled. ``abort`` throws it away
by design (its callers have no request left to answer into), so a second entry
point exists rather than changing the first.

Both halves of that contract are pinned here, because both are decisions a
reader has to be able to trust: the control frame is the existing ``abort`` op
and the receipt is returned VERBATIM, and a facade with no attached client
RAISES instead of resolving a no-op — "nothing to interrupt" and "the owner went
away" are different answers, and only one of them is safe to paint as success.
"""

from __future__ import annotations

import pytest

from local_operator.session.attached import AttachedSession


@pytest.fixture
def cold_session(tmp_path):
    async def _build(cwd: str = "/tmp") -> AttachedSession:
        return await AttachedSession.cold(
            "session-under-test",
            config_dir=tmp_path,
            cwd=cwd,
            takeover_factory=lambda *_a, **_k: None,
        )

    return _build


class FakeClient:
    """The attach client's interrupt-relevant surface, and nothing else."""

    def __init__(self, receipt: str = "stopping this turn", error: BaseException | None = None):
        self.receipt = receipt
        self.error = error
        #: The control op each call dialled, so the mapping this facade
        #: documents (interrupt -> the runtime's ``abort``) is asserted rather
        #: than assumed.
        self.ops: list[str] = []
        # ``is_cold`` reads this, together with ``_ready_for_events``.
        self.connected = True

    async def abort(self) -> str:
        self.ops.append("abort")
        if self.error is not None:
            raise self.error
        return self.receipt


def _bind(session: AttachedSession, client: object) -> None:
    """Make ``session`` look bound, the way ``is_cold`` actually reads it."""
    session._client = client  # type: ignore[assignment]
    session._ready_for_events = True


@pytest.mark.asyncio
async def test_an_interrupt_returns_the_owners_receipt_verbatim(cold_session) -> None:
    """The receipt is the OWNER's, not the follower's.

    It counts what actually settled and names what refused to die; a facade that
    re-worded or truncated it would be guessing at exactly the number it is
    refusing to parse, which is why the whole second entry point exists.
    """
    session = await cold_session("/tmp")
    client = FakeClient(receipt="stopped 1 subagent; 2 background bash jobs untouched")
    _bind(session, client)

    receipt = await session.interrupt()

    assert receipt == "stopped 1 subagent; 2 background bash jobs untouched"
    assert client.ops == ["abort"], "interrupt must dial the runtime's existing abort op"


@pytest.mark.asyncio
async def test_an_interrupt_with_no_client_raises_rather_than_answering(cold_session) -> None:
    """A detached facade must not answer ``idle``-shaped silence.

    The route maps a genuinely COLD session to ``idle`` before it gets here, so
    reaching this branch means the client vanished mid-flight (or the pod is a
    reduced host) — an event the user needs reported, not absorbed. Returning an
    empty sentence would paint a press that reached nobody as one that found
    nothing to stop.
    """
    session = await cold_session("/tmp")
    assert session.is_cold

    with pytest.raises(ConnectionError) as raised:
        await session.interrupt()

    assert "not attached" in str(raised.value)


@pytest.mark.asyncio
async def test_a_dead_socket_is_not_an_idle_answer_either(cold_session) -> None:
    """``connected`` False is the same answer as no client at all.

    ``is_cold`` reads both, so a facade whose socket dropped mid-turn must not
    take the other branch merely because a client object is still assigned.
    """
    session = await cold_session("/tmp")
    client = FakeClient()
    client.connected = False
    _bind(session, client)

    with pytest.raises(ConnectionError):
        await session.interrupt()

    assert client.ops == [], "a dead client was dialled anyway"
