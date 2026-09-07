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

import asyncio
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
async def test_a_move_during_the_mount_engage_spawns_in_the_NEW_directory(
    cold_session, monkeypatch
) -> None:
    """The blocker this feature shipped with, and its primary use case.

    ``is_cold`` means "no SYNCHRONISED runtime is attached", NOT "no runtime
    exists". The TUI engages eagerly at mount, so by the time a user can type
    ``/move`` the engage has usually already read ``_cwd`` and handed it to
    ``engage_runtime`` — a 1-3 s spawn. A move that only assigned the field
    landed after that read: the runtime came up in the OLD directory while the
    receipt and the band both said the new one, permanently and silently.

    Asserts what the runtime is SPAWNED WITH, which is the property the feature
    promises. The original cold test asserted the return value and ``_cwd``,
    both of which were true throughout the bug — which is why 92 green tests
    did not catch it.
    """
    from local_operator.session.runtime import launch as launch_mod

    spawns: list[str] = []
    engage_started = asyncio.Event()

    async def slow_engage(session_id, cwd, work, *, config_dir, deadline_s=30.0):
        engage_started.set()
        await asyncio.sleep(0.05)  # the spawn window the move must not lose
        spawns.append(str(cwd))

    monkeypatch.setattr(launch_mod, "engage_runtime", slow_engage)

    session = await cold_session("/tmp")
    task = asyncio.create_task(session._ensure_bound())
    await engage_started.wait()  # the engage has read _cwd and is in flight

    assert session.is_cold, "the state a user types /move in at boot"
    await asyncio.wait_for(session.set_working_directory("/usr"), timeout=10)
    try:
        await asyncio.wait_for(task, timeout=10)
    except Exception:  # noqa: BLE001 — the bind's own outcome is not under test
        pass

    assert spawns, "no runtime was ever engaged"
    assert spawns[-1] == "/usr", (
        f"the surviving runtime was spawned at {spawns[-1]!r}, not the moved-to "
        "directory — the band and the receipt would say /usr while every tool "
        "call, the system prompt and skill discovery used the old path"
    )
    assert session._cwd == "/usr"


@pytest.mark.asyncio
async def test_a_move_while_a_live_runtime_is_resyncing_still_retires_it(
    cold_session,
) -> None:
    """``_ready_for_events`` is False during a history refresh while the runtime
    keeps serving, so an ``is_cold`` test routed a LIVE runtime into the
    free-field-assignment branch: no retire, nothing rebound, and the user told
    it worked (review MAJOR-1; QA Q2 is the same hole during a rebind).

    Liveness of the socket is the honest term, and this pins it.
    """
    session = await cold_session("/tmp")
    client = FakeClient()
    _bind(session, client)
    session._ready_for_events = False  # mid-resync: cold by the old predicate
    session.owner_idle = lambda: True  # type: ignore[method-assign]

    assert session.is_cold, "the predicate that used to decide the branch"
    outcome = await session.set_working_directory("/usr")

    assert outcome == "rebound", "a live runtime must be retired, not skipped"
    assert client.ops == ["retire_now"]


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
async def test_a_version_skewed_runtime_gets_the_vetted_sentence(cold_session) -> None:
    """A runtime older than this build answers ``unknown op: 'retire_now'``.

    This is the REACHABLE skew path: the viewer always carries this build's
    ``AttachClient``, so the ``callable`` guard below can never fire in
    production and the user was shown a raw wire internal instead of the
    sentence written for them (review MINOR-1 / QA Q4).
    """
    session = await cold_session("/tmp")
    _bind(session, FakeClient(error=RuntimeError("unknown op: 'retire_now'")))
    session.owner_idle = lambda: True  # type: ignore[method-assign]
    with pytest.raises(RuntimeError, match="too old to be moved"):
        await session.set_working_directory("/usr")
    assert session._cwd == "/tmp"


@pytest.mark.asyncio
async def test_any_other_transport_failure_rolls_the_directory_back(cold_session) -> None:
    """A failure that is NOT version skew still refuses and restores the cwd."""
    session = await cold_session("/tmp")
    _bind(session, FakeClient(error=ConnectionError("socket closed")))
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


@pytest.mark.asyncio
@pytest.mark.parametrize("bound", [False, True])
async def test_a_move_repoints_an_armed_wake_at_the_new_directory(
    cold_session, tmp_path, bound
) -> None:
    """An armed wake spawns an UNATTENDED runtime from the index's own ``cwd``.

    Nothing rewrote that entry, so a session moved with a wake armed fired it
    in the old directory (review MAJOR-2). A wake is the one mechanism designed
    to run without the user present to notice, so the divergence survived until
    the next manual prompt. Checked on both paths: the bound path self-heals
    only when its successor opens, which can be long after a wake is due.
    """
    from local_operator.wakes.store import read_index, write_entry

    session = await cold_session("/tmp")
    write_entry(
        tmp_path,
        session.session_id,
        cwd="/tmp",
        schedules=[{"id": "w1", "due_at_ms": 1, "note": "check the build"}],
    )
    if bound:
        _bind(session, FakeClient())
        session.owner_idle = lambda: True  # type: ignore[method-assign]

    await session.set_working_directory("/usr")

    entry = read_index(tmp_path)[session.session_id]
    assert entry["cwd"] == "/usr", (
        "the wake index still names the old directory, so the supervisor would "
        "spawn the unattended runtime there"
    )
    assert [s["id"] for s in entry["schedules"]] == ["w1"], "the schedule must survive"
