"""Orphaned-process-group reaper: reap ONLY genuine waste, never a live runner.

The reaper kills a bash process group iff the ``lop`` process that spawned it is
provably dead — the one leak the in-process ``_kill()`` paths cannot cover,
because a hard-SIGKILLed owner runs no cleanup and ``start_new_session`` already
stripped the group's SIGHUP. The central property this suite defends is the
inverse of retention's "nothing is ever deleted": here nothing is ever KILLED
unless owner-death is proven, because a 10h ML trainer and a 10h stuck pyright
are identical by every signal EXCEPT owner liveness.

Wherever possible identity is injected (a fake pid + start token + config dir)
rather than spawning real processes, mirroring retention's pid-injection style;
the few cases that need a real killable group spawn a short ``sleep`` in its own
session so the group-leader/pgid-reuse gate can be exercised for real.
"""

from __future__ import annotations

import json
import os
import signal
import subprocess
import time

from local_operator.tools import group_reaper
from local_operator.tools.group_reaper import (
    PROC_GROUPS_DIRNAME,
    ReaperSweepResult,
    kill_own_groups,
    register_group,
    sweep_orphan_groups,
    unregister_group,
)


# --------------------------------------------------------------------------- #
# Helpers
# --------------------------------------------------------------------------- #
def _groups_dir(config_dir):
    return config_dir / PROC_GROUPS_DIRNAME


def _write_ledger(config_dir, owner_pid, owner_start, entries):
    """Fabricate a ledger for ``(owner_pid, owner_start)`` with raw entry dicts.

    Bypasses ``register_group`` so a test can point a ledger at any pid/token/
    pgid combination it wants (dead owner, wrong token, recycled pgid) without a
    real process behind every field.
    """
    path = group_reaper._ledger_path(config_dir, owner_pid, owner_start)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as handle:
        for entry in entries:
            handle.write(json.dumps(entry) + "\n")
    return path


def _spawn_group(cmd="sleep 600"):
    """Spawn ``cmd`` in its own session/group, returning (Popen, pgid).

    ``start_new_session=True`` mirrors exactly how ``execute_bash`` spawns, so
    the group-leader-start (pgid-reuse) gate is exercised against a real leader.
    """
    proc = subprocess.Popen(
        ["/bin/sh", "-c", cmd],
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
        start_new_session=True,
    )
    # Give sh a beat to exec the workload in place so getpgid is stable.
    for _ in range(50):
        try:
            pgid = os.getpgid(proc.pid)
        except ProcessLookupError:
            time.sleep(0.01)
            continue
        return proc, pgid
    raise AssertionError("group never became visible")


def _alive(pgid):
    try:
        os.killpg(pgid, 0)
        return True
    except ProcessLookupError:
        return False
    except PermissionError:
        # The pgid is held by a process this user cannot signal. For a group we
        # spawned ourselves that can only mean the pgid was RECYCLED onto an
        # unrelated process after our group died — i.e. our group is gone. This
        # is the pgid-reuse race the reaper itself defends against; for a
        # liveness probe it means "not our group any more".
        return False


def _reaped(proc, pgid):
    """Deterministic proof a spawned group was killed: its leader process exits.

    ``os.killpg(pgid, 0)`` alone is racy in-test because the OS recycles the
    freed pgid onto an unrelated process within milliseconds. The child we
    spawned is the ground truth: SIGKILL to the group kills the leader, so
    ``proc.wait`` returns and ``returncode`` is the negative kill signal.
    """
    try:
        proc.wait(timeout=2)
    except subprocess.TimeoutExpired:
        return False
    return proc.returncode is not None and proc.returncode != 0


def _reap_group(pgid):
    # Best-effort cleanup only. After a successful reap the pgid may already be
    # gone (ProcessLookupError) or recycled onto a group this user cannot signal
    # (PermissionError) — both mean "our group is already dead", so tolerate.
    try:
        os.killpg(pgid, signal.SIGKILL)
    except (ProcessLookupError, PermissionError):
        pass


# --------------------------------------------------------------------------- #
# register / unregister
# --------------------------------------------------------------------------- #
def test_register_writes_owner_keyed_ledger(tmp_path):
    """A registered group lands in ONE file keyed by the owner's identity."""
    pid = os.getpid()
    register_group(4242, "pyright .", config_dir=tmp_path, owner_pid=pid)
    files = list(_groups_dir(tmp_path).glob("*.jsonl"))
    assert len(files) == 1
    assert files[0].name.startswith(f"{pid}-")
    entry = json.loads(files[0].read_text().splitlines()[0])
    assert entry["pgid"] == 4242
    assert entry["owner_pid"] == pid
    assert entry["cmd"] == "pyright ."
    assert "ts" in entry  # written for diagnostics


def test_ts_is_never_a_decision_input():
    """The ``ts`` field must be write-only: no branch in the module reads it.

    Proven structurally rather than behaviourally — the design bans age/idle as
    inputs, and the cheapest durable guarantee is that ``ts`` is never compared.
    """
    src = __import__("pathlib").Path(group_reaper.__file__).read_text(encoding="utf-8")
    # The only permitted occurrences: the doc comment, the write, and _now's
    # docstring. Any comparison (ts <, ts >, entry["ts"]) would be a new line.
    read_sites = [
        ln
        for ln in src.splitlines()
        if "ts" in ln
        and ('["ts"]' in ln or "'ts'" in ln)
        and '"ts":' not in ln  # exclude the write
    ]
    assert read_sites == [], f"ts read as a decision input: {read_sites}"


def test_register_appends_multiple_groups_one_file(tmp_path):
    """Parallel bash calls from one owner append lines to a single ledger."""
    pid = os.getpid()
    register_group(11, "a", config_dir=tmp_path, owner_pid=pid)
    register_group(22, "b", config_dir=tmp_path, owner_pid=pid)
    files = list(_groups_dir(tmp_path).glob("*.jsonl"))
    assert len(files) == 1
    lines = files[0].read_text().splitlines()
    assert {json.loads(x)["pgid"] for x in lines} == {11, 22}


def test_unregister_drops_one_line_keeps_others(tmp_path):
    """Clean group death removes exactly its line, leaving siblings intact."""
    pid = os.getpid()
    register_group(11, "a", config_dir=tmp_path, owner_pid=pid)
    register_group(22, "b", config_dir=tmp_path, owner_pid=pid)
    unregister_group(11, config_dir=tmp_path, owner_pid=pid)
    files = list(_groups_dir(tmp_path).glob("*.jsonl"))
    lines = files[0].read_text().splitlines()
    assert {json.loads(x)["pgid"] for x in lines} == {22}


def test_unregister_last_line_unlinks_ledger(tmp_path):
    """Removing the final line drops the whole owner ledger file."""
    pid = os.getpid()
    register_group(11, "a", config_dir=tmp_path, owner_pid=pid)
    unregister_group(11, config_dir=tmp_path, owner_pid=pid)
    assert list(_groups_dir(tmp_path).glob("*.jsonl")) == []


def test_double_unregister_is_harmless(tmp_path):
    """A second unregister (or one for a missing pgid) never raises."""
    pid = os.getpid()
    register_group(11, "a", config_dir=tmp_path, owner_pid=pid)
    unregister_group(11, config_dir=tmp_path, owner_pid=pid)
    unregister_group(11, config_dir=tmp_path, owner_pid=pid)  # ledger already gone
    unregister_group(999, config_dir=tmp_path, owner_pid=pid)  # no such pgid


def test_register_best_effort_swallows_write_failure(tmp_path, monkeypatch):
    """A ledger that cannot be written must not raise into the caller."""

    def boom(*a, **k):
        raise OSError("disk full")

    monkeypatch.setattr("builtins.open", boom)
    # Must not raise.
    register_group(11, "a", config_dir=tmp_path, owner_pid=os.getpid())


# --------------------------------------------------------------------------- #
# sweep — owner truth table
# --------------------------------------------------------------------------- #
def test_owner_dead_reaps(tmp_path, monkeypatch):
    """Row 1: owner pid gone -> reap. Uses a fabricated dead owner + live group."""
    proc, pgid = _spawn_group()
    try:
        leader_start = group_reaper._owner_start_token(pgid)
        # A dead owner pid: spawn+reap a throwaway child to get a surely-dead id.
        dead = subprocess.Popen(["/bin/sh", "-c", "true"])
        dead.wait()
        _write_ledger(
            tmp_path,
            dead.pid,
            "any-start-token",
            [
                {
                    "pgid": pgid,
                    "owner_pid": dead.pid,
                    "owner_start": "any-start-token",
                    "grp_leader_start": leader_start,
                    "cmd": "sleep 600",
                    "ts": 1.0,
                }
            ],
        )
        assert _alive(pgid)
        result = sweep_orphan_groups(tmp_path)
        assert result.reaped_groups == 1
        # Group is gone and the ledger unlinked.
        assert _reaped(proc, pgid)
        assert list(_groups_dir(tmp_path).glob("*.jsonl")) == []
    finally:
        _reap_group(pgid)
        proc.wait(timeout=2)


def test_owner_alive_matching_token_never_reaps(tmp_path):
    """Row 2 — the LIVE-TRAINER GUARANTEE: owner alive + token match -> never reap."""
    proc, pgid = _spawn_group()
    try:
        my_pid = os.getpid()
        my_start = group_reaper._owner_start_token(my_pid)
        assert my_start is not None  # our own live process always has a token
        register_group(pgid, "sleep 600", config_dir=tmp_path, owner_pid=my_pid)
        # Sweep from a DIFFERENT self_pid so the "skip my own pid" shortcut is
        # not what protects it — the owner-liveness check must be what does.
        result = sweep_orphan_groups(tmp_path, self_pid=my_pid + 10_000_000)
        assert result.reaped_groups == 0
        assert result.skipped_live_owners == 1
        assert _alive(pgid)  # untouched
        # Ledger intact.
        path = group_reaper._ledger_path(tmp_path, my_pid, my_start)
        assert path.exists()
    finally:
        _reap_group(pgid)
        proc.wait(timeout=2)


def test_owner_alive_differing_token_reaps(tmp_path):
    """Row 3: pid alive but start token differs (pid reuse) -> reap."""
    proc, pgid = _spawn_group()
    try:
        leader_start = group_reaper._owner_start_token(pgid)
        live_pid = os.getpid()  # alive...
        _write_ledger(
            tmp_path,
            live_pid,
            "DELIBERATELY-WRONG-START-TOKEN",  # ...but wrong identity
            [
                {
                    "pgid": pgid,
                    "owner_pid": live_pid,
                    "owner_start": "DELIBERATELY-WRONG-START-TOKEN",
                    "grp_leader_start": leader_start,
                    "cmd": "sleep 600",
                    "ts": 1.0,
                }
            ],
        )
        result = sweep_orphan_groups(tmp_path, self_pid=live_pid + 10_000_000)
        assert result.reaped_groups == 1
        assert _reaped(proc, pgid)
    finally:
        _reap_group(pgid)
        proc.wait(timeout=2)


def test_owner_alive_unreadable_token_never_reaps(tmp_path, monkeypatch):
    """Row 4: pid alive, current token unreadable -> ambiguous -> never reap."""
    proc, pgid = _spawn_group()
    try:
        live_pid = os.getpid()
        real_token = group_reaper._owner_start_token
        leader_start = real_token(pgid)

        def token(pid):
            # Owner token unreadable; leave the leader token readable so the
            # ambiguity is exclusively on the OWNER side.
            if pid == live_pid:
                return None
            return real_token(pid)

        _write_ledger(
            tmp_path,
            live_pid,
            "stored-token",
            [
                {
                    "pgid": pgid,
                    "owner_pid": live_pid,
                    "owner_start": "stored-token",
                    "grp_leader_start": leader_start,
                    "cmd": "sleep 600",
                    "ts": 1.0,
                }
            ],
        )
        monkeypatch.setattr(group_reaper, "_owner_start_token", token)
        result = sweep_orphan_groups(tmp_path, self_pid=live_pid + 10_000_000)
        assert result.reaped_groups == 0
        assert result.skipped_live_owners == 1
        assert _alive(pgid)  # untouched
    finally:
        _reap_group(pgid)
        proc.wait(timeout=2)


# --------------------------------------------------------------------------- #
# sweep — victim (pgid-reuse) gate
# --------------------------------------------------------------------------- #
def test_victim_leader_gone_drops_no_kill(tmp_path):
    """Dead owner, but the group already died: drop the line, kill nothing."""
    dead = subprocess.Popen(["/bin/sh", "-c", "true"])
    dead.wait()
    # A pgid that is surely gone (the just-reaped child's own pid space).
    gone_pgid = dead.pid
    _write_ledger(
        tmp_path,
        dead.pid,
        "start",
        [
            {
                "pgid": gone_pgid,
                "owner_pid": dead.pid,
                "owner_start": "start",
                "grp_leader_start": "whatever",
                "cmd": "sleep 600",
                "ts": 1.0,
            }
        ],
    )
    result = sweep_orphan_groups(tmp_path)
    assert result.reaped_groups == 0
    assert list(_groups_dir(tmp_path).glob("*.jsonl")) == []  # ledger dropped


def test_victim_leader_recycled_different_start_drops_no_kill(tmp_path):
    """Dead owner, pgid now held by a DIFFERENT live group: never kill it."""
    innocent, innocent_pgid = _spawn_group()
    try:
        dead = subprocess.Popen(["/bin/sh", "-c", "true"])
        dead.wait()
        _write_ledger(
            tmp_path,
            dead.pid,
            "start",
            [
                {
                    "pgid": innocent_pgid,  # pgid points at an innocent live group
                    "owner_pid": dead.pid,
                    "owner_start": "start",
                    # ...but the recorded leader start does NOT match the
                    # innocent group's real leader start.
                    "grp_leader_start": "STALE-LEADER-START-DOES-NOT-MATCH",
                    "cmd": "sleep 600",
                    "ts": 1.0,
                }
            ],
        )
        result = sweep_orphan_groups(tmp_path)
        assert result.reaped_groups == 0
        assert _alive(innocent_pgid)  # innocent survives
        assert list(_groups_dir(tmp_path).glob("*.jsonl")) == []
    finally:
        _reap_group(innocent_pgid)
        innocent.wait(timeout=2)


def test_victim_leader_alive_and_matches_kills(tmp_path):
    """Dead owner + leader alive + start token matches: this is our orphan -> kill."""
    proc, pgid = _spawn_group()
    try:
        leader_start = group_reaper._owner_start_token(pgid)
        dead = subprocess.Popen(["/bin/sh", "-c", "true"])
        dead.wait()
        _write_ledger(
            tmp_path,
            dead.pid,
            "start",
            [
                {
                    "pgid": pgid,
                    "owner_pid": dead.pid,
                    "owner_start": "start",
                    "grp_leader_start": leader_start,
                    "cmd": "sleep 600",
                    "ts": 1.0,
                }
            ],
        )
        result = sweep_orphan_groups(tmp_path)
        assert result.reaped_groups == 1
        assert _reaped(proc, pgid)
    finally:
        _reap_group(pgid)
        proc.wait(timeout=2)


# --------------------------------------------------------------------------- #
# robustness
# --------------------------------------------------------------------------- #
def test_torn_ledger_line_is_skipped(tmp_path):
    """A corrupt/torn JSONL line is skipped; valid lines still processed."""
    proc, pgid = _spawn_group()
    try:
        leader_start = group_reaper._owner_start_token(pgid)
        dead = subprocess.Popen(["/bin/sh", "-c", "true"])
        dead.wait()
        path = group_reaper._ledger_path(tmp_path, dead.pid, "start")
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w", encoding="utf-8") as handle:
            handle.write("{ this is not valid json\n")  # torn line
            handle.write(
                json.dumps(
                    {
                        "pgid": pgid,
                        "owner_pid": dead.pid,
                        "owner_start": "start",
                        "grp_leader_start": leader_start,
                        "cmd": "sleep 600",
                        "ts": 1.0,
                    }
                )
                + "\n"
            )
        result = sweep_orphan_groups(tmp_path)
        assert result.reaped_groups == 1  # the good line still acted on
        assert _reaped(proc, pgid)
    finally:
        _reap_group(pgid)
        proc.wait(timeout=2)


def test_double_sweep_is_idempotent(tmp_path):
    """Two sweeps of the same dead-owner ledger: second finds nothing, no raise."""
    proc, pgid = _spawn_group()
    try:
        leader_start = group_reaper._owner_start_token(pgid)
        dead = subprocess.Popen(["/bin/sh", "-c", "true"])
        dead.wait()
        _write_ledger(
            tmp_path,
            dead.pid,
            "start",
            [
                {
                    "pgid": pgid,
                    "owner_pid": dead.pid,
                    "owner_start": "start",
                    "grp_leader_start": leader_start,
                    "cmd": "sleep 600",
                    "ts": 1.0,
                }
            ],
        )
        first = sweep_orphan_groups(tmp_path)
        second = sweep_orphan_groups(tmp_path)
        assert first.reaped_groups == 1
        assert second.reaped_groups == 0
    finally:
        _reap_group(pgid)
        proc.wait(timeout=2)


def test_sweep_missing_dir_is_noop(tmp_path):
    """A config dir with no proc-groups directory sweeps cleanly to zero."""
    result = sweep_orphan_groups(tmp_path)
    assert result == ReaperSweepResult()


# --------------------------------------------------------------------------- #
# soft death
# --------------------------------------------------------------------------- #
def test_kill_own_groups_reaps_self_and_spares_sibling(tmp_path):
    """Soft death kills THIS owner's group and unlinks its ledger; a sibling
    owner's live group survives untouched (the scope proof)."""
    mine, my_pgid = _spawn_group()
    sibling, sib_pgid = _spawn_group()
    try:
        my_pid = os.getpid()
        # My own ledger, via the real register path.
        register_group(my_pgid, "sleep 600", config_dir=tmp_path, owner_pid=my_pid)
        # A sibling owner's ledger (a different, live pid) with a live group.
        sib_leader = group_reaper._owner_start_token(sib_pgid)
        _write_ledger(
            tmp_path,
            my_pid + 7_000_000,  # a different owner id
            group_reaper._owner_start_token(my_pid) or "x",
            [
                {
                    "pgid": sib_pgid,
                    "owner_pid": my_pid + 7_000_000,
                    "owner_start": "sibling",
                    "grp_leader_start": sib_leader,
                    "cmd": "sleep 600",
                    "ts": 1.0,
                }
            ],
        )
        kill_own_groups(config_dir=tmp_path, owner_pid=my_pid)
        assert _reaped(mine, my_pgid)  # my group reaped
        assert _alive(sib_pgid)  # sibling untouched
        # My ledger gone; the sibling's remains.
        my_token = group_reaper._owner_start_token(my_pid)
        assert my_token is not None
        my_path = group_reaper._ledger_path(tmp_path, my_pid, my_token)
        assert not my_path.exists()
        assert len(list(_groups_dir(tmp_path).glob("*.jsonl"))) == 1
    finally:
        _reap_group(my_pgid)
        _reap_group(sib_pgid)
        mine.wait(timeout=2)
        sibling.wait(timeout=2)


def test_kill_own_groups_idempotent_when_ledger_absent(tmp_path):
    """A second soft-death call (ledger already unlinked) is a clean no-op."""
    kill_own_groups(config_dir=tmp_path, owner_pid=os.getpid())


# --------------------------------------------------------------------------- #
# platform gate
# --------------------------------------------------------------------------- #
def test_win32_is_a_total_noop(tmp_path, monkeypatch):
    """On Windows nothing is registered or reaped: the leak is POSIX-specific."""
    monkeypatch.setattr(group_reaper, "_REAPING_IS_SUPPORTED", False)
    register_group(11, "a", config_dir=tmp_path, owner_pid=os.getpid())
    assert not _groups_dir(tmp_path).exists()
    result = sweep_orphan_groups(tmp_path)
    assert result == ReaperSweepResult()
    kill_own_groups(config_dir=tmp_path, owner_pid=os.getpid())  # no raise
