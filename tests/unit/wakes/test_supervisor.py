"""The wake supervisor: what it fires, what it refuses to fire, when it retires.

The property under test throughout is that the supervisor STARTS runtimes and
never delivers wakes. A session fires its own overdue wakes on load
(``WakeScheduler.load`` re-arms them to ``now + LOAD_GRACE_MS``), so the
supervisor's whole contribution is making a runtime exist — and any attempt to
also deliver would double-fire every wake it touched.
"""

from __future__ import annotations

import time
from pathlib import Path

import pytest

from local_operator.wakes.store import write_entry
from local_operator.wakes.supervisor import fire_due_wakes, serve

NOW_MS = int(time.time() * 1000)


def _schedule(due_ms: int, wake_id: str = "w1") -> dict[str, object]:
    return {"id": wake_id, "message": "check the deploy", "next_due_at": due_ms, "created_at": 1}


@pytest.fixture
def engagements(monkeypatch):  # noqa: ANN201
    """Record every engage the supervisor makes, without starting a process."""
    calls: list[dict[str, object]] = []

    async def fake_engage(session_id, cwd, work, *, config_dir, deadline_s=30.0):  # noqa: ANN001
        calls.append({"session_id": session_id, "cwd": cwd, "work": work})
        return None

    monkeypatch.setattr("local_operator.session.runtime.launch.engage_runtime", fake_engage)
    return calls


@pytest.fixture
def no_live_runtimes(monkeypatch):  # noqa: ANN201
    async def _none(config_dir, session_id):  # noqa: ANN001
        return False

    monkeypatch.setattr("local_operator.wakes.supervisor._has_live_runtime", _none)


@pytest.mark.asyncio
async def test_a_due_wake_on_a_cold_session_starts_a_runtime(
    tmp_path: Path, engagements, no_live_runtimes
) -> None:
    write_entry(tmp_path, "sessioncold1", cwd=str(tmp_path), schedules=[_schedule(NOW_MS - 5_000)])

    fired = await fire_due_wakes(tmp_path, now_ms=NOW_MS)

    assert fired == 1
    assert [call["session_id"] for call in engagements] == ["sessioncold1"]
    assert engagements[0]["cwd"] == str(tmp_path)


@pytest.mark.asyncio
async def test_the_errand_delivers_nothing(tmp_path: Path, engagements, no_live_runtimes) -> None:
    """The correction that supersedes the spec's ``wake_fire`` op.

    A ``WakeErrand`` carries no text and no message: the session's own
    scheduler delivers the occurrence when it loads. An errand that also
    delivered would append every wake twice.
    """
    from local_operator.session.runtime.launch import WakeErrand

    write_entry(tmp_path, "sessioncold1", cwd=str(tmp_path), schedules=[_schedule(NOW_MS - 1_000)])
    await fire_due_wakes(tmp_path, now_ms=NOW_MS)

    work = engagements[0]["work"]
    assert isinstance(work, WakeErrand)
    assert not hasattr(work, "text"), "a wake errand must carry no message to deliver"


@pytest.mark.asyncio
async def test_a_live_session_fires_its_own_wakes(tmp_path: Path, engagements, monkeypatch) -> None:
    """The no-live-record rule.

    A session with a live record is already running and its scheduler owns its
    wakes. Engaging there would be a second opinion about a schedule the live
    session is actively advancing.
    """

    async def _always_live(config_dir, session_id):  # noqa: ANN001
        return True

    monkeypatch.setattr("local_operator.wakes.supervisor._has_live_runtime", _always_live)
    write_entry(tmp_path, "sessionlive1", cwd=str(tmp_path), schedules=[_schedule(NOW_MS - 5_000)])

    fired = await fire_due_wakes(tmp_path, now_ms=NOW_MS)

    assert fired == 0
    assert engagements == []


@pytest.mark.asyncio
async def test_a_future_wake_is_not_fired_early(
    tmp_path: Path, engagements, no_live_runtimes
) -> None:
    write_entry(
        tmp_path, "sessionlater", cwd=str(tmp_path), schedules=[_schedule(NOW_MS + 600_000)]
    )

    assert await fire_due_wakes(tmp_path, now_ms=NOW_MS) == 0
    assert engagements == []


@pytest.mark.asyncio
async def test_a_dormant_session_is_left_alone(
    tmp_path: Path, engagements, no_live_runtimes
) -> None:
    """A /stop stamps ``stopped_at``; its wakes stay armed but must not fire.

    Firing here would resurrect a session the kill switch deliberately ended —
    PR 3's contract is that reopening the session is what re-arms it.
    """
    write_entry(
        tmp_path,
        "sessionstopd",
        cwd=str(tmp_path),
        schedules=[_schedule(NOW_MS - 5_000)],
        preserve={"stopped_at": NOW_MS - 10_000},
    )

    assert await fire_due_wakes(tmp_path, now_ms=NOW_MS) == 0
    assert engagements == []


@pytest.mark.asyncio
async def test_a_long_overdue_wake_is_left_to_the_session_catchup(
    tmp_path: Path, engagements, no_live_runtimes
) -> None:
    """Nothing is gained by racing to start a runtime for a week-old wake.

    The session's resume catch-up handles arbitrarily-old overdue schedules
    when the user next opens it, and that is the surface where a stale reminder
    belongs.
    """
    ancient = NOW_MS - int((8 * 24 * 3600) * 1000)
    write_entry(tmp_path, "sessionstale", cwd=str(tmp_path), schedules=[_schedule(ancient)])

    assert await fire_due_wakes(tmp_path, now_ms=NOW_MS) == 0


@pytest.mark.asyncio
async def test_one_session_failing_does_not_stop_the_others(
    tmp_path: Path, no_live_runtimes, monkeypatch
) -> None:
    """A sweep is not all-or-nothing: the schedule is untouched, so it retries."""
    seen: list[str] = []

    async def flaky(session_id, cwd, work, *, config_dir, deadline_s=30.0):  # noqa: ANN001
        seen.append(session_id)
        if session_id == "sessionbadaa":
            raise TimeoutError("no runtime")
        return None

    monkeypatch.setattr("local_operator.session.runtime.launch.engage_runtime", flaky)
    write_entry(tmp_path, "sessionbadaa", cwd=str(tmp_path), schedules=[_schedule(NOW_MS - 9_000)])
    write_entry(tmp_path, "sessiongood1", cwd=str(tmp_path), schedules=[_schedule(NOW_MS - 8_000)])

    fired = await fire_due_wakes(tmp_path, now_ms=NOW_MS)

    assert set(seen) == {"sessionbadaa", "sessiongood1"}
    assert fired == 1, "the healthy session's wake still fired"


@pytest.mark.asyncio
async def test_the_supervisor_retires_when_the_index_empties(tmp_path: Path) -> None:
    """Exit 0, so ``KeepAlive: {SuccessfulExit: False}`` leaves it down.

    This is the whole reason the supervisor is not an always-on cost: a machine
    with no wakes left runs no supervisor at all.
    """
    assert await serve(tmp_path) == 0


@pytest.mark.asyncio
async def test_a_malformed_entry_costs_one_session_not_the_sweep(
    tmp_path: Path, engagements, no_live_runtimes
) -> None:
    """The index is written by other processes; one bad file must not be fatal."""
    write_entry(tmp_path, "sessiongood2", cwd=str(tmp_path), schedules=[_schedule(NOW_MS - 3_000)])
    from local_operator.wakes.store import entry_path

    bad = entry_path(tmp_path, "sessionbroken")
    bad.parent.mkdir(parents=True, exist_ok=True)
    bad.write_text("{not json", encoding="utf-8")

    fired = await fire_due_wakes(tmp_path, now_ms=NOW_MS)

    assert fired == 1
    assert [call["session_id"] for call in engagements] == ["sessiongood2"]
