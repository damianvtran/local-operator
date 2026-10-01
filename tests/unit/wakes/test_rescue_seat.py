"""Unit tests for the rescue pass's seat in the wake supervisor.

The seat owns the cadence, the engagement and the verification; the pass itself
is tested in ``tests/unit/session/runtime/test_rescue.py``. Everything here runs
the seat's own seams — the census is a spy, the engage and the record read are
replaced — so a seat bug cannot be hidden by a working pass, or the other way
round. No real runtime is spawned and no clock is read for a bound.
"""

from __future__ import annotations

import asyncio
import time
from pathlib import Path

import pytest

from local_operator.session.runtime import rescue
from local_operator.wakes import supervisor as sup


def _report(*, engage: int = 1) -> rescue.RescueReport:
    report = rescue.RescueReport()
    for index in range(engage):
        report.to_engage.append(
            rescue.RescueDecision(
                session_id=f"dead{index:02d}",
                verdict="fire",
                tag="disposed",
                pid=4700000 + index,
                started_at=1.0,
                cwd="/Users/damian",
                death_at_ms=1_800_000_000_000,
            )
        )
    return report


def test_the_first_pass_is_due_and_the_next_follows_the_interval(tmp_path: Path) -> None:
    seat = sup._RescueSweep(tmp_path)
    assert seat.due()
    assert seat.seconds_until() == 0.0

    seat.next_at = time.monotonic() + 10.0
    now = seat.next_at
    assert not seat.due(now=now - 5)
    assert seat.seconds_until(now=now - 5) == pytest.approx(5.0)
    assert seat.due(now=now)


def test_kick_runs_one_pass_per_interval(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    calls: list[dict[str, object]] = []

    async def scenario() -> None:
        seat = sup._RescueSweep(tmp_path)
        monkeypatch.setattr(
            rescue, "rescue_scan", lambda *a, **k: calls.append(k) or _report(engage=0)
        )
        seat.kick()
        await asyncio.sleep(0.05)
        assert len(calls) == 1
        # Not due again: the next pass is armed from HERE, not from the finished
        # one, so a slow census cannot make the loop spin.
        seat.kick()
        await asyncio.sleep(0.05)
        assert len(calls) == 1
        await seat.shutdown()

    asyncio.run(scenario())


def test_an_engage_is_verified_against_a_new_record(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The seat engages through the supervisor's own path and records the outcome."""
    engaged: list[tuple[str, str]] = []
    noted: list[tuple[str, str, int | None]] = []

    class _Record:
        pid = 4_700_099  # a DIFFERENT pid from the dead run

    async def fake_engage(session_id, cwd, errand, *, config_dir, deadline_s):  # noqa: ANN001
        engaged.append((session_id, cwd))

    monkeypatch.setattr(rescue, "rescue_scan", lambda *a, **k: _report(engage=1))
    monkeypatch.setattr(rescue, "read_ledger", lambda *a, **k: {"count": 0})
    monkeypatch.setattr(
        rescue,
        "note_rescue_attempt",
        lambda *a, **k: noted.append((a[1], k["outcome"], k.get("engaged_pid"))),
    )
    monkeypatch.setattr("local_operator.session.runtime.launch.engage_runtime", fake_engage)
    monkeypatch.setattr(
        "local_operator.mobile.attach_client.find_runtime_record",
        lambda *a, **k: (_Record(), 4_700_099),
    )
    monkeypatch.setattr(sup, "RESCUE_START_STAGGER_S", 0.0)

    async def scenario() -> None:
        seat = sup._RescueSweep(tmp_path)
        seat.kick()
        for _ in range(50):
            await asyncio.sleep(0.02)
            if seat.in_flight == 0 and engaged:
                break
        await seat.shutdown()

    asyncio.run(scenario())
    assert engaged == [("dead00", "/Users/damian")]
    assert noted == [("dead00", "verified", 4_700_099)]


def test_shutdown_drops_an_in_flight_pass(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    released = asyncio.Event()

    def slow_scan(*args: object, **kwargs: object) -> rescue.RescueReport:
        # A census that never returns: shutdown must not wait for it.
        raise AssertionError("the pass should have been cancelled before it ran")

    monkeypatch.setattr(rescue, "rescue_scan", slow_scan)

    async def scenario() -> None:
        seat = sup._RescueSweep(tmp_path)
        seat.next_at = None
        seat.kick()
        await seat.shutdown()
        released.set()

    asyncio.run(scenario())
    assert released.is_set()
