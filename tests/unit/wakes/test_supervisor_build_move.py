"""The wake supervisor moves itself onto a newer build — only once it owns nothing.

WHY (update-parity item 4, 2026-11-18): ``lop update`` defers the wakes-unit
restart while wake-started conversations run, and nothing re-checked once they
ended — on node2 one supervisor ran 0.102.2 for five days. ``_BuildMove`` re-execs
the same pid through the stable shim when, and only when, a newer settled build
is on disk, no seat is busy and ``count_owned`` proves zero. Every gate here is a
"do not move" on an unknown answer; these tests pin each one.
"""

from __future__ import annotations

import asyncio
import contextlib
import os
import sys
from pathlib import Path
from typing import Any

import pytest

from local_operator.update import BuildStamp
from local_operator.wakes import supervisor
from local_operator.wakes.supervisor import MOVE_RETRY_S, _BuildMove

posix_only = pytest.mark.skipif(os.name != "posix", reason="an in-place exec is POSIX-only")

OLD = BuildStamp("0.102.2", "a" * 40)
NEW = BuildStamp("0.106.0", "b" * 40)


class _Clock:
    def __init__(self) -> None:
        self.now = 1000.0

    def __call__(self) -> float:
        return self.now


class _Seat:
    """A seat with the two shapes ``_seat_busy`` reads."""

    def __init__(self, in_flight: int = 0) -> None:
        self.in_flight = in_flight
        self._task = None


def _mover(
    tmp_path: Path,
    *,
    changed: Any = lambda boot: NEW,
    census: Any = lambda root: 0,
    image: Any = None,
    boot: Any = OLD,
    clock: _Clock | None = None,
    execs: list[Any] | None = None,
    exec_fn: Any = None,
) -> _BuildMove:
    shim = tmp_path / "python3"
    if not shim.exists():
        shim.write_text("#!/bin/sh\n")
        shim.chmod(0o755)
    calls = execs if execs is not None else []
    return _BuildMove(
        tmp_path,
        boot=boot,
        sample_boot=False,
        changed=changed,
        census=census,
        image=image or (lambda: shim),
        exec_fn=exec_fn or (lambda img, argv: calls.append((img, argv))),
        argv_tail=["--flag"],
        clock=clock or _Clock(),
    )


@posix_only
def test_an_idle_supervisor_execs_onto_a_newer_build_through_the_shim(tmp_path: Path) -> None:
    execs: list[Any] = []
    roots: list[Any] = []
    mover = _mover(tmp_path, execs=execs, census=lambda root: roots.append(root) or 0)

    mover.attempt((_Seat(), _Seat()))

    assert execs == [
        (
            tmp_path / "python3",
            [str(tmp_path / "python3"), "-m", "local_operator.wakes.supervisor", "--flag"],
        )
    ]
    # Scoped to THIS store: the root its own stop sweep reaches.
    assert roots == [tmp_path]


@posix_only
@pytest.mark.parametrize(
    ("label", "kwargs", "seats"),
    [
        ("a wake-started conversation runs", {"census": lambda root: 1}, ()),
        ("the census is unknown", {"census": lambda root: None}, ()),
        ("the census raises", {"census": lambda root: 1 / 0}, ()),
        ("an engagement is in flight", {}, (_Seat(in_flight=1),)),
        ("the build stamp is unreadable", {"changed": lambda boot: None}, ()),
        ("there is no boot stamp", {"boot": None}, ()),
        ("the shim is unresolvable", {"image": lambda: None}, ()),
    ],
)
def test_no_exec_when_any_gate_is_unproven(
    tmp_path: Path, label: str, kwargs: dict[str, Any], seats: tuple[Any, ...]
) -> None:
    execs: list[Any] = []
    mover = _mover(tmp_path, execs=execs, **kwargs)

    assert mover.attempt(seats) is False
    assert execs == [], label


@posix_only
def test_a_task_seat_still_running_blocks_the_move(tmp_path: Path) -> None:
    class _Pending:
        def done(self) -> bool:
            return False

    seat = _Seat()
    seat._task = _Pending()  # type: ignore[assignment]
    execs: list[Any] = []
    _mover(tmp_path, execs=execs).attempt((seat,))
    assert execs == []


@posix_only
def test_an_older_or_unreadable_pointer_never_moves_through_the_real_build_check(
    tmp_path: Path, monkeypatch
) -> None:
    """The real ``buildwatch.build_changed``: backwards and unreadable are both "no"."""
    from local_operator import buildwatch, update

    monkeypatch.setattr(buildwatch, "_settle_elapsed", lambda: True)
    execs: list[Any] = []

    monkeypatch.setattr(update, "disk_build", lambda root=None: BuildStamp("0.100.0", "c" * 40))
    _BuildMove(
        tmp_path,
        boot=OLD,
        sample_boot=False,
        census=lambda root: 0,
        image=lambda: Path(sys.executable),
        exec_fn=lambda img, argv: execs.append(argv),
        argv_tail=[],
    ).attempt(())
    assert execs == [], "an older pointer walked the supervisor backwards"

    def unreadable(root=None):
        raise OSError("stamp unreadable")

    monkeypatch.setattr(update, "disk_build", unreadable)
    mover = _BuildMove(
        tmp_path,
        boot=OLD,
        sample_boot=False,
        census=lambda root: 0,
        image=lambda: Path(sys.executable),
        exec_fn=lambda img, argv: execs.append(argv),
        argv_tail=[],
    )
    mover.attempt(())
    assert execs == []

    # Control: the same real check with a NEWER stamp does move.
    monkeypatch.setattr(update, "disk_build", lambda root=None: NEW)
    mover.attempt(())
    assert len(execs) == 1


@posix_only
def test_a_refused_move_retries_after_the_interval_and_then_follows(tmp_path: Path) -> None:
    clock = _Clock()
    owned = [1]
    execs: list[Any] = []
    mover = _mover(tmp_path, clock=clock, execs=execs, census=lambda root: owned[0])

    mover.attempt(())
    assert execs == [] and mover.due() is False, "a refusal must not re-census every slice"
    owned[0] = 0
    clock.now += MOVE_RETRY_S - 1
    mover.attempt(())
    assert execs == []
    clock.now += 1
    assert mover.due() is True
    mover.attempt(())
    assert len(execs) == 1, "the move follows the last conversation's end"


@posix_only
def test_a_failed_exec_restores_fd_inheritance_and_keeps_serving(
    tmp_path: Path, monkeypatch
) -> None:
    read_fd, write_fd = os.pipe()
    os.set_inheritable(read_fd, True)
    seen: dict[str, Any] = {}

    def failing_execv(path: str, argv: list[str]) -> None:
        seen["inheritable_during_exec"] = os.get_inheritable(read_fd)
        seen["argv"] = argv
        raise OSError("exec format error")

    monkeypatch.setattr(supervisor.os, "execv", failing_execv)
    clock = _Clock()
    try:
        mover = _mover(tmp_path, clock=clock, exec_fn=supervisor._exec_supervisor)
        assert mover.attempt(()) is False
        assert seen["inheritable_during_exec"] is False, "a stray fd would leak into the successor"
        assert os.get_inheritable(read_fd) is True, "the fd was not restored after the failed exec"
        assert os.get_inheritable(write_fd) is False, "a non-inheritable fd was made inheritable"
        assert mover.due() is False, "the failure must back off, not spin"
        clock.now += MOVE_RETRY_S
        assert mover.due() is True, "a failed exec is retried on a later loop"
    finally:
        os.close(read_fd)
        os.close(write_fd)


def _arm_a_wake_an_hour_out(config_dir: Path) -> int:
    """One real, fireable wake an hour out, so the loop sleeps instead of retiring."""
    import time

    from local_operator.resume import TRANSCRIPT_NAME
    from local_operator.wakes.store import write_entry

    session = config_dir / "sessions" / "sessionmove1"
    session.mkdir(parents=True, exist_ok=True)
    (session / TRANSCRIPT_NAME).write_text("", encoding="utf-8")
    due = int(time.time() * 1000) + 3_600_000
    write_entry(
        config_dir,
        "sessionmove1",
        cwd=str(config_dir),
        schedules=[{"id": "w1", "message": "later", "next_due_at": due, "created_at": 1}],
    )
    return due


@posix_only
@pytest.mark.asyncio
async def test_serve_keeps_running_when_the_exec_raises(tmp_path: Path, monkeypatch) -> None:
    """The loop survives a failed move and keeps attempting on later passes."""
    import time

    monkeypatch.setattr(supervisor, "SLICE_S", 0.01)
    monkeypatch.setattr(supervisor, "MOVE_RETRY_S", 0.0)
    monkeypatch.setattr(supervisor, "_install_stop_handlers", lambda loop, on_stop: [])
    # The periodic passes fork ps/lsof against the machine; this test is about
    # the loop surviving a failed exec, so they are inert here.
    for seat in (
        supervisor._ResidencySweep,
        supervisor._MachineMemorySweep,
        supervisor._MemoryMaintenanceSeat,
        supervisor._RescueSweep,
        supervisor._TriggerSweep,
    ):
        monkeypatch.setattr(seat, "kick", lambda self: None)
    _arm_a_wake_an_hour_out(tmp_path)
    attempts: list[int] = []

    def boom(img: Path, argv: list[str]) -> None:
        attempts.append(1)
        raise OSError("exec failed")

    mover = _mover(tmp_path, exec_fn=boom, clock=time.monotonic)  # type: ignore[arg-type]
    task = asyncio.create_task(supervisor.serve(tmp_path, move=mover))
    for _ in range(2000):
        if len(attempts) >= 2:
            break
        await asyncio.sleep(0.01)
    assert len(attempts) >= 2, "the loop stopped after the first failed exec"
    assert not task.done()
    task.cancel()
    with contextlib.suppress(asyncio.CancelledError):
        await asyncio.wait_for(task, timeout=30)


@pytest.mark.asyncio
async def test_a_due_move_ends_the_sleep_early(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.setattr(supervisor, "SLICE_S", 0.01)
    due = _arm_a_wake_an_hour_out(tmp_path)
    calls: list[int] = []

    def interrupt() -> bool:
        calls.append(1)
        return len(calls) >= 3

    await asyncio.wait_for(
        supervisor._sleep_in_slices(tmp_path, 3600.0, due, interrupt=interrupt), timeout=10
    )
    assert len(calls) == 3


def test_only_the_unit_entry_moves_and_never_with_once(tmp_path: Path, monkeypatch) -> None:
    """``lop wake serve`` (a CLI process) gets no mover; ``--once`` neither."""
    import inspect

    assert inspect.signature(supervisor.serve).parameters["move"].default is None
    seen: list[Any] = []

    async def fake_serve(root: Path, *, once: bool = False, move: Any = None) -> int:
        seen.append((once, move))
        return 0

    monkeypatch.setattr(supervisor, "serve", fake_serve)
    monkeypatch.setattr("local_operator.paths.config_dir", lambda: tmp_path)
    monkeypatch.setattr("local_operator.buildwatch.boot_build", lambda: OLD)
    monkeypatch.setattr("local_operator.helpers.setup_cross_platform_environment", lambda: None)
    assert supervisor.main(["--once"]) == 0
    assert supervisor.main([]) == 0
    assert seen[0] == (True, None)
    assert isinstance(seen[1][1], _BuildMove) and seen[1][1].boot == OLD
