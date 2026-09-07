"""``RemoteSession.set_working_directory`` — the three honest outcomes of `/move`.

The cwd is fixed when a runtime is spawned (``LOP_MOBILE_CHILD_CWD``), so there
are exactly two ways to honour a change and one situation where neither is
safe. What these tests pin is that the facade never ends up in the state the
feature exists to prevent: its ``_cwd`` saying one thing while its runtime
works in another.

The RETIRING route is the load-bearing detail. A move must not leave by the
``stopping`` route, because that latches ``_deliberate_stop`` and parks the
viewer in the stopped state — the right answer for a session the user ENDED and
the wrong one for a move, after which the conversation continues.
"""

from __future__ import annotations

from types import SimpleNamespace

import pytest

from local_operator.session.remote import RemoteSession


@pytest.fixture
def cold_session(tmp_path):
    async def _build(cwd: str = "/tmp") -> RemoteSession:
        return await RemoteSession.cold(
            "session-under-test",
            config_dir=tmp_path,
            cwd=cwd,
            takeover_factory=lambda *_a, **_k: None,
        )

    return _build


class FakeClient:
    """The attach client's move-relevant surface, and nothing else."""

    def __init__(self, answer: str = "retiring", error: Exception | None = None) -> None:
        self.answer = answer
        self.error = error
        self.ops: list[str] = []
        #: ``is_cold`` reads this — a bound facade is one whose client is up.
        self.connected = True

    async def retire_now(self) -> str:
        self.ops.append("retire_now")
        if self.error is not None:
            raise self.error
        return self.answer


def _bind(session: RemoteSession, client: object) -> None:
    """Make ``session`` look bound, the way ``is_cold`` actually reads it."""
    session._client = client  # type: ignore[assignment]
    session._ready_for_events = True


@pytest.mark.asyncio
async def test_a_cold_session_just_changes_the_directory_it_will_spawn_with(
    cold_session,
) -> None:
    """The "at the start of a session" case the feature was asked for, and the
    common one: `lop` opens cold. It is a field assignment and costs nothing."""
    session = await cold_session("/tmp")
    assert session.is_cold
    assert await session.set_working_directory("/usr") == "cold"
    assert session._cwd == "/usr"


@pytest.mark.asyncio
async def test_a_bound_idle_session_retires_its_runtime_and_rebinds(cold_session) -> None:
    session = await cold_session("/tmp")
    client = FakeClient()
    _bind(session, client)
    session.owner_idle = lambda: True  # type: ignore[method-assign]
    assert await session.set_working_directory("/usr") == "rebound"
    assert client.ops == ["retire_now"]
    assert session._cwd == "/usr"


@pytest.mark.asyncio
async def test_a_move_does_NOT_park_the_viewer_in_the_stopped_state(cold_session) -> None:
    """The reason `/move` uses `retire_now` rather than `stop`. Left latched,
    the next prompt would route to the stopped notice for a conversation that
    is deliberately still alive — and the flag could not be cleared here
    anyway, because the disconnect that sets it arrives after this returns."""
    session = await cold_session("/tmp")
    _bind(session, FakeClient())
    session.owner_idle = lambda: True  # type: ignore[method-assign]
    await session.set_working_directory("/usr")
    assert session._deliberate_stop is False


@pytest.mark.asyncio
async def test_a_busy_session_is_REFUSED_and_does_not_move(cold_session) -> None:
    """Retiring mid-turn would abort a model call the user is paying for, and
    "apply it next turn" is exactly the divergence AGENTS.md names for
    `/reload`: the band showing one directory while the running turn's tools
    resolve against another."""
    session = await cold_session("/tmp")
    client = FakeClient()
    _bind(session, client)
    session.owner_idle = lambda: False  # type: ignore[method-assign]
    with pytest.raises(RuntimeError, match="working right now"):
        await session.set_working_directory("/usr")
    assert session._cwd == "/tmp"
    assert client.ops == []


@pytest.mark.asyncio
async def test_a_runtime_that_keeps_itself_rolls_the_directory_back(cold_session) -> None:
    """Work can arrive between this viewer's idle read and the runtime's own
    re-check. The runtime is the authority; its reason is the receipt, and
    nothing moved so nothing is recorded as moved."""
    session = await cold_session("/tmp")
    _bind(session, FakeClient(answer="kept: busy"))
    session.owner_idle = lambda: True  # type: ignore[method-assign]
    with pytest.raises(RuntimeError, match="busy"):
        await session.set_working_directory("/usr")
    assert session._cwd == "/tmp"


@pytest.mark.asyncio
async def test_a_refused_op_rolls_the_directory_back(cold_session) -> None:
    session = await cold_session("/tmp")
    _bind(session, FakeClient(error=RuntimeError("unknown op: 'retire_now'")))
    session.owner_idle = lambda: True  # type: ignore[method-assign]
    with pytest.raises(RuntimeError, match="could not move"):
        await session.set_working_directory("/usr")
    assert session._cwd == "/tmp"


@pytest.mark.asyncio
async def test_a_runtime_too_old_to_know_the_op_refuses_cleanly(cold_session) -> None:
    """Rather than moving anyway and leaving the runtime in the old directory."""
    session = await cold_session("/tmp")
    _bind(session, SimpleNamespace(connected=True))
    session.owner_idle = lambda: True  # type: ignore[method-assign]
    with pytest.raises(RuntimeError, match="too old"):
        await session.set_working_directory("/usr")
    assert session._cwd == "/tmp"
