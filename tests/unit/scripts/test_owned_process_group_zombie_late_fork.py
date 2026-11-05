"""A corpse-only group is certified empty only after a POST-SAMPLE census agrees.

The hole (architect spike on main 30e864e06d38): `refresh` takes censuses 1 and 2,
THEN samples. A member that forks a child after census 2 and then dies reads
``Z`` at sample time, so every member looks exited and the old code reached
OBSERVED_EMPTY -- with members (10, 21) while pid 30 ran. Neither census could
have seen 30. The fix takes census 3 after a corpse-only sample
(`OwnedGroup._late_members`) and treats any new pid, or any member that is no
longer the exited process sampled, as a reason to re-run the cycle (shared
`_SAMPLE_CYCLES` cap) or refuse.

A second door: on Linux a process whose MAIN thread exited while other threads
run also reads ``Z`` (state is the leader's, fs/proc/array.c:491) and can still
fork, so "exited" also requires ``num_threads <= 1`` (field 20).

The synthetic tests reuse `World` (real `_proc_samples` over intercepted stat
rows). The two real-kernel tests at the bottom run only on Linux (CI): each
builds the shape with real processes and expects anything but OBSERVED_EMPTY.
"""

from __future__ import annotations

import ctypes
import os
import signal
import sys
import threading
import time
from pathlib import Path

import pytest

from local_operator import owned_process_group as owned
from tests.unit.scripts.test_owned_process_group_departed import LEADER, World, bbed

MEMBER = 21
LATE = 30


def zworld(monkeypatch, script, live=()):
    """Censuses list {LEADER, 21}; 21 is a corpse at sample time (birth "201")."""
    world = World(monkeypatch, [LEADER, MEMBER], [LEADER, MEMBER], live=live, script=script)
    world.kernel.rows[MEMBER] = (LEADER, LEADER, LEADER, "201")
    world.kernel.corpses.add(MEMBER)
    return world


def _on_call(world, n, action):
    """Run ``action`` right after census call number ``n`` (1-based) returns."""
    base = world.census

    def census(pgid, deadline):
        out = base(pgid, deadline)
        if world.calls == n:
            action()
        return out

    world.cap.census = census


# Discriminating: each FAILS on main's product files -------------------------


def test_zombie_member_late_fork_is_not_certified_empty(monkeypatch):
    # Censuses 1 and 2 see {L, 21}; 21 is a zombie at sample time; 30 was forked
    # after census 2, so every LATER listing shows it.
    world = zworld(monkeypatch, [[LEADER, MEMBER], [LEADER, MEMBER]], live=(LATE,))
    world.first = world.second = [LEADER, MEMBER, LATE]
    verdict = world.refresh(terminal=True)
    assert verdict.status != "OBSERVED_EMPTY", verdict
    assert verdict.status == "VERIFIED" and LATE in verdict.members, verdict
    assert world.cap.probe["sample_retries"][0]["late"] == [LATE]


def test_leader_forks_after_census_two_then_dies(monkeypatch):
    # The common shape: members == {L}; L forks 30 after census 2 and exits.
    world = World(
        monkeypatch, [LEADER, LATE], [LEADER, LATE], live=(LATE,), script=[[LEADER], [LEADER]]
    )
    verdict = world.refresh(terminal=True)
    assert verdict.status == "VERIFIED" and LATE in verdict.members, verdict
    # Cycle 0: C1, C2, C3 (late 30); cycle 1: C1, C2 with a live member -> no C3.
    assert world.calls == 5


def test_reused_member_pid_hosting_a_live_late_child_is_caught(monkeypatch):
    # Census 3 lists only {L, 21} -- a subset of the members -- but pid 21 now
    # hosts a LIVE process with another birth: the subset proves nothing alone.
    world = zworld(monkeypatch, [])

    def reuse():
        world.kernel.rows[MEMBER] = (LEADER, LEADER, LEADER, "999")
        world.kernel.corpses.discard(MEMBER)

    _on_call(world, 3, reuse)
    verdict = world.refresh(terminal=True)
    assert verdict.status == "VERIFIED" and MEMBER in verdict.members, verdict
    assert world.cap.probe["sample_retries"][0]["late"] == [MEMBER]


def test_threaded_zombie_member_is_not_corpse_only(monkeypatch):
    # Linux: a member whose main thread exited while another thread runs reads "Z"
    # with num_threads == 2. It can still fork, so it is live, not a corpse.
    world = zworld(monkeypatch, [])
    plain = world.read

    def read(path):
        row = plain(path)
        if f"/{MEMBER}/" in str(path):
            parts = row.split(" ")
            # parts[0] pid, parts[1] "(synthetic)", then the tail: tail[17] is
            # num_threads.
            parts[2 + 17] = "2"
            row = " ".join(parts)
        return row

    world.read = read
    verdict = world.refresh(terminal=True)
    assert verdict.status != "OBSERVED_EMPTY", verdict
    assert MEMBER in verdict.members, verdict
    # It is decided as live at the sample: no census 3 is needed.
    assert world.calls == 2
    # The record says WHY a `zombie: true` member counted as live (S3).
    assert world.cap.probe["observed"][str(MEMBER)]["before"] == {
        "birth": "201",
        "zombie": True,
        "threads": 2,
    }


def test_single_threaded_rows_record_no_thread_count(monkeypatch):
    # S3 keeps the record bounded: an ordinary row is exactly what it was.
    world = zworld(monkeypatch, [])
    verdict = world.refresh(terminal=True)
    assert verdict.status == "OBSERVED_EMPTY", verdict
    assert world.cap.probe["observed"][str(MEMBER)]["before"] == {"birth": "201", "zombie": True}


def test_member_reborn_as_another_corpse_is_late(monkeypatch):
    # R2: census 3 lists only {L, 21} and 21 still reads a single-threaded "Z" --
    # but with a NEW birth. A different process died at that number after the
    # sample (it could have forked first), so the sampled corpse no longer
    # describes it: the birth comparison alone makes it late.
    world = zworld(monkeypatch, [])

    def reborn():
        world.kernel.rows[MEMBER] = (LEADER, LEADER, LEADER, "777")

    _on_call(world, 3, reborn)
    verdict = world.refresh(terminal=True)
    assert world.cap.probe["sample_retries"][0]["late"] == [MEMBER], world.cap.probe
    # Cycle 1 samples the reborn corpse fresh and its census 3 agrees.
    assert verdict.status == "OBSERVED_EMPTY" and world.calls == 6, (verdict, world.calls)


def test_a_discarded_cycle_publishes_no_vanished(monkeypatch):
    # R1/S1: cycle 0 prunes 20/21 as departed, is corpse-only, and its census 3
    # shows late pid 30, so it is DISCARDED. Cycle 1 decides on {L, 30}; the
    # departures of the discarded cycle must not ride along beside it.
    world = bbed(
        monkeypatch,
        first=[LEADER, 20, 21],
        second=[LEADER],
        live=(LATE,),
        script=[[LEADER, 20, 21], [LEADER], [LEADER, LATE], [LEADER, LATE], [LEADER, LATE]],
    )
    verdict = world.refresh(terminal=True)
    assert verdict.status == "VERIFIED" and verdict.members == (LEADER, LATE), verdict
    assert world.cap.probe["sample_retries"][0]["late"] == [LATE]
    assert "vanished" not in world.cap.probe, world.cap.probe["vanished"]


# Refusals: the guard never turns doubt into emptiness -----------------------


def test_unparseable_thread_count_omits_the_row_and_refuses(monkeypatch):
    # `threads=None` would read as single-threaded; the reader omits the row, so
    # the member is unresolved rather than silently exited.
    world = zworld(monkeypatch, [])
    plain = world.read

    def read(path):
        row = plain(path)
        if f"/{MEMBER}/" in str(path):
            parts = row.split(" ")
            parts[2 + 17] = "x"
            row = " ".join(parts)
        return row

    world.read = read
    verdict = world.refresh(terminal=True)
    assert verdict.status == "REFUSED", verdict


def test_resample_enoent_while_censuses_list_the_member_refuses(monkeypatch):
    # After census 3, 21's stat read fails ENOENT while every census still lists
    # it: an in-both ENOENT member is unresolved, so the run refuses.
    world = zworld(monkeypatch, [])
    _on_call(world, 3, lambda: world.errors.__setitem__(MEMBER, FileNotFoundError(2, "reaped")))
    verdict = world.refresh(terminal=True)
    assert verdict.status == "REFUSED", verdict


def test_late_change_persisting_every_cycle_refuses(monkeypatch):
    world = zworld(monkeypatch, [[LEADER, MEMBER], [LEADER, MEMBER], [LEADER, MEMBER, LATE]] * 3)
    world.kernel.rows[LATE] = (LEADER, LEADER, LEADER, "300")
    world.kernel.corpses.add(LATE)
    verdict = world.refresh(terminal=True)
    assert verdict.status == "REFUSED", verdict
    assert "changed after the corpse-only sample" in verdict.reason, verdict
    assert world.calls == 9
    assert [row["cycle"] for row in world.cap.probe["sample_retries"]] == [0, 1]
    failure = world.cap.probe["sample_failure"]
    assert failure["phase"] == "post-sample-census" and failure["late"] == [LATE]


def test_census_three_failure_refuses_not_empty(monkeypatch):
    world = zworld(monkeypatch, [])
    base = world.census

    def census(pgid, deadline):
        if world.calls == 2:
            world.calls += 1
            raise owned.Unknown("strict /proc census deadline")
        return base(pgid, deadline)

    world.cap.census = census
    verdict = world.refresh(terminal=True)
    assert verdict.status == "REFUSED", verdict
    assert verdict.refusal_class == "unproven", verdict


def _foreign_leader(world):
    """The leader's number now hosts a readable process with ANOTHER birth."""
    world.kernel.rows[LEADER] = (*world.kernel.rows[LEADER][:3], "777")


def test_census_three_failure_with_a_reaped_leader_is_pin_lost(monkeypatch):
    # R3/Q1: census 3 raises and the witness re-reads as a foreign birth. Loss
    # outranks the unreadable census: PinLost, and the capability is revoked.
    world = zworld(monkeypatch, [])
    base = world.census

    def census(pgid, deadline):
        if world.calls == 2:
            world.calls += 1
            _foreign_leader(world)
            raise owned.Unknown("strict /proc census deadline")
        return base(pgid, deadline)

    world.cap.census = census
    verdict = world.refresh(terminal=True)
    assert verdict.status == "REFUSED", verdict
    assert verdict.reason == "PinLost:reservation lost during the post-sample census", verdict
    assert verdict.refusal_class == "pin_lost", verdict
    assert not world.cap.accepted_terminal(verdict)


def _late_every_cycle(monkeypatch):
    world = zworld(monkeypatch, [[LEADER, MEMBER], [LEADER, MEMBER], [LEADER, MEMBER, LATE]] * 3)
    world.kernel.rows[LATE] = (LEADER, LEADER, LEADER, "300")
    world.kernel.corpses.add(LATE)
    return world


def test_late_refusal_with_a_reaped_leader_is_pin_lost(monkeypatch):
    # S2: the last cycle's census 3 still shows a late pid, and by then the
    # leader's number hosts a foreign birth. The refusal is loss, not unproven.
    world = _late_every_cycle(monkeypatch)
    _on_call(world, 9, lambda: _foreign_leader(world))
    verdict = world.refresh(terminal=True)
    assert verdict.reason == "PinLost:reservation lost after the corpse-only sample", verdict
    assert verdict.refusal_class == "pin_lost", verdict


def test_late_refusal_with_an_unreadable_leader_stays_unknown(monkeypatch):
    # S2 stays fail-closed: an unreadable witness is doubt, never loss.
    world = _late_every_cycle(monkeypatch)
    _on_call(world, 9, lambda: world.errors.__setitem__(LEADER, PermissionError(13, "synthetic")))
    verdict = world.refresh(terminal=True)
    assert verdict.reason == "Unknown:group changed after the corpse-only sample", verdict
    assert verdict.refusal_class == "unproven", verdict


def test_budget_guard_counts_census_three(monkeypatch):
    world = zworld(monkeypatch, [[LEADER, MEMBER], [LEADER, MEMBER], [LEADER, MEMBER, LATE]])
    world.kernel.rows[LATE] = (LEADER, LEADER, LEADER, "300")
    base = world.census

    def slow(pgid, deadline):
        world.kernel.now += 1.0
        return base(pgid, deadline)

    world.cap.census = slow
    verdict = world.refresh(terminal=True)  # cycle 0 takes ~3 s of 5: no rerun fits
    assert verdict.status == "REFUSED" and world.calls == 3, (verdict, world.calls)


# Controls: what must stay cheap and empty ------------------------------------


def test_zombie_only_group_without_child_is_observed_empty(monkeypatch):
    world = zworld(monkeypatch, [])
    verdict = world.refresh(terminal=True)
    assert verdict.status == "OBSERVED_EMPTY", verdict
    assert world.cap.accepted_terminal(verdict)
    assert world.calls == 3


def test_leader_only_corpse_costs_exactly_one_extra_census(monkeypatch):
    world = World(monkeypatch, [LEADER], [LEADER])
    verdict = world.refresh(terminal=True)
    assert verdict.status == "OBSERVED_EMPTY" and world.cap.accepted_terminal(verdict), verdict
    assert world.calls == 3


def test_reaped_member_between_sample_and_census_three_is_fine(monkeypatch):
    world = zworld(monkeypatch, [[LEADER, MEMBER], [LEADER, MEMBER], [LEADER]])
    verdict = world.refresh(terminal=True)
    assert verdict.status == "OBSERVED_EMPTY", verdict


def test_live_member_path_takes_no_third_census(monkeypatch):
    world = World(monkeypatch, [LEADER, 22], [LEADER, 22], live=(22,))
    verdict = world.refresh()
    assert verdict.status == "VERIFIED" and world.calls == 2


# Real kernel (Linux only: CI) ---------------------------------------------------

linux_only = pytest.mark.skipif(
    not sys.platform.startswith("linux"), reason="real /proc state and thread counts"
)

_PR_SET_CHILD_SUBREAPER = 36


def _stat(pid):
    """(state, num_threads) from the real /proc/<pid>/stat, or None once gone."""
    try:
        tail = Path(f"/proc/{pid}/stat").read_text().rpartition(")")[2].split()
    except OSError:
        return None
    return tail[0], int(tail[17])


def _until(predicate, what, bound=10.0):
    end = time.monotonic() + bound
    while time.monotonic() < end:
        if predicate():
            return
        time.sleep(0.01)
    raise AssertionError("timed out waiting for " + what)


def _capture(leader):
    return owned.OwnedGroup.capture(
        leader,
        parent_pid=os.getpid(),
        signal_group=lambda *args: None,
        deadline=time.monotonic() + 5.0,
    )


def _reap(pids):
    """SIGKILL then reap exactly the pids this test made, in order; ignore the gone."""
    for pid in pids:
        if pid is None:
            continue
        try:
            os.kill(pid, signal.SIGKILL)
        except ProcessLookupError:
            pass
        try:
            os.waitpid(pid, 0)
        except ChildProcessError:
            pass


@linux_only
def test_real_kernel_member_forks_after_census_two_then_dies():
    """Leader exits; member M forks sleeper S after census 2 and exits.

    We are a CHILD SUBREAPER for the duration, so the orphaned M re-parents to
    this process and stays an unreaped ZOMBIE at sample time -- without it init
    would reap M at once and the shape would refuse on ENOENT instead, a test
    that main also passes. S re-parents to us too, so we can reap it.
    """
    libc = ctypes.CDLL(None, use_errno=True)
    assert libc.prctl(_PR_SET_CHILD_SUBREAPER, 1, 0, 0, 0) == 0, ctypes.get_errno()
    ready_r, ready_w = os.pipe()  # leader -> us: "M forked, pid follows"
    go_r, go_w = os.pipe()  # us -> leader: "exit now"
    fork_r, fork_w = os.pipe()  # us -> M: "fork S and exit"
    late_r, late_w = os.pipe()  # M -> us: S's pid
    leader = member = late = None
    try:
        leader = os.fork()
        if leader == 0:  # pragma: no cover - child never returns
            try:
                os.setsid()
                child = os.fork()
                if child == 0:
                    os.read(fork_r, 1)
                    sleeper = os.fork()
                    if sleeper == 0:
                        time.sleep(60)
                        os._exit(0)
                    os.write(late_w, str(sleeper).encode() + b"\n")
                    os._exit(0)
                os.write(ready_w, str(child).encode() + b"\n")
                os.read(go_r, 1)
            finally:
                os._exit(0)
        member = int(os.read(ready_r, 32).decode())
        cap = _capture(leader)
        os.write(go_w, b"x")
        _until(lambda: (_stat(leader) or ("?",))[0] == "Z", "the leader corpse")
        base = cap.census
        calls = 0

        def census(pgid, deadline):
            nonlocal calls, late
            out = base(pgid, deadline)
            calls += 1
            if calls == 2:
                # After census 2 and BEFORE the sample: M forks S and dies.
                os.write(fork_w, b"x")
                late = int(os.read(late_r, 32).decode())
                _until(lambda: (_stat(member) or ("?",))[0] == "Z", "the member corpse")
            return out

        cap.census = census
        verdict = cap.refresh(deadline=time.monotonic() + 5.0, terminal=True)
        assert late is not None and _stat(late) is not None, "the sleeper never ran"
        assert verdict.status != "OBSERVED_EMPTY", (verdict, cap.probe.get("members"))
    finally:
        _reap([late, member, leader])
        libc.prctl(_PR_SET_CHILD_SUBREAPER, 0, 0, 0, 0)
        for fd in (ready_r, ready_w, go_r, go_w, fork_r, fork_w, late_r, late_w):
            os.close(fd)


@linux_only
def test_real_kernel_main_thread_exit_reads_z_with_two_threads_and_is_not_empty():
    """The leader's MAIN thread calls pthread_exit while a second thread lives."""
    ready_r, ready_w = os.pipe()
    go_r, go_w = os.pipe()
    end_r, end_w = os.pipe()
    leader = None
    try:
        leader = os.fork()
        if leader == 0:  # pragma: no cover - child never returns
            try:
                os.setsid()

                def survivor():
                    os.read(end_r, 1)
                    os._exit(0)

                threading.Thread(target=survivor, daemon=True).start()
                os.write(ready_w, b"r")
                os.read(go_r, 1)
                # ctypes releases the GIL around the call, so the survivor keeps
                # running after the main thread is gone.
                ctypes.CDLL(None).pthread_exit(None)
            finally:
                os._exit(0)
        os.read(ready_r, 1)
        cap = _capture(leader)
        os.write(go_w, b"x")
        _until(lambda: _stat(leader) == ("Z", 2), "a Z leader with two threads")
        assert _stat(leader) == ("Z", 2)
        verdict = cap.refresh(deadline=time.monotonic() + 5.0, terminal=True)
        assert verdict.status != "OBSERVED_EMPTY", verdict
    finally:
        os.write(end_w, b"x")
        _reap([leader])
        for fd in (ready_r, ready_w, go_r, go_w, end_r, end_w):
            os.close(fd)
