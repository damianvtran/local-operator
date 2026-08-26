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
# lock-husk cleanup (R4)
# --------------------------------------------------------------------------- #
def test_sweep_removes_dead_owner_lock_spares_live_owner(tmp_path):
    """A dead owner's ``.lock`` is reaped with its ledger; a live owner's is not.

    Two owners each have a ledger AND its ``.lock`` sidecar. The dead owner's
    pair must both be gone after the sweep (the husk cleanup the reaper's whole
    purpose demands); the LIVE owner's ledger and lock must both survive
    untouched — the live-trainer guarantee extended to the lock file.
    """
    # DEAD owner: fabricate a ledger keyed on a surely-dead pid, plus its lock.
    dead = subprocess.Popen(["/bin/sh", "-c", "true"])
    dead.wait()
    dead_ledger = _write_ledger(
        tmp_path,
        dead.pid,
        "dead-start",
        [
            {
                "pgid": 999_999,  # a pgid with no live leader: dropped, no kill
                "owner_pid": dead.pid,
                "owner_start": "dead-start",
                "grp_leader_start": None,
                "cmd": "sleep 600",
                "ts": 1.0,
            }
        ],
    )
    dead_lock = group_reaper._lock_path(dead_ledger)
    dead_lock.write_text("")

    # LIVE owner: our own pid, registered through the real path so the sweep's
    # owner-liveness check (not a pid shortcut) is what protects it.
    my_pid = os.getpid()
    my_start = group_reaper._self_start_token(my_pid)
    assert my_start is not None
    register_group(123456, "sleep 600", config_dir=tmp_path, owner_pid=my_pid)
    live_ledger = group_reaper._ledger_path(tmp_path, my_pid, my_start)
    live_lock = group_reaper._lock_path(live_ledger)
    live_lock.write_text("")

    # Sweep from a different self_pid so the "skip my own pid" shortcut is not
    # what spares the live owner — its liveness must be.
    sweep_orphan_groups(tmp_path, self_pid=my_pid + 10_000_000)

    assert not dead_ledger.exists()
    assert not dead_lock.exists()  # dead owner's husk reaped
    assert live_ledger.exists()
    assert live_lock.exists()  # live owner's lock untouched


def test_sweep_removes_orphan_lock_of_dead_owner(tmp_path):
    """A ``.lock`` with no ledger (clean-exit husk) whose pid is dead is reaped.

    The common accumulation path: a graceful shutdown compacts the ledger away
    but leaves the lock sidecar behind. There is no ledger to key on, so this is
    reaped by the orphan-lock pass on the pid-gone signal alone.
    """
    dead = subprocess.Popen(["/bin/sh", "-c", "true"])
    dead.wait()
    groups = _groups_dir(tmp_path)
    groups.mkdir(parents=True, exist_ok=True)
    orphan = groups / f"{dead.pid}-dead-start.jsonl.lock"
    orphan.write_text("")

    sweep_orphan_groups(tmp_path)

    assert not orphan.exists()


def test_sweep_spares_orphan_lock_of_live_owner(tmp_path):
    """A ledger-less ``.lock`` whose pid is still alive is NEVER removed."""
    groups = _groups_dir(tmp_path)
    groups.mkdir(parents=True, exist_ok=True)
    # Our own pid is unambiguously alive.
    orphan = groups / f"{os.getpid()}-live-start.jsonl.lock"
    orphan.write_text("")

    sweep_orphan_groups(tmp_path)

    assert orphan.exists()  # possibly-live owner's lock left in place


def test_sweep_leaves_unparseable_lock_name(tmp_path):
    """A ``.lock`` whose leading segment is not an integer is left untouched."""
    groups = _groups_dir(tmp_path)
    groups.mkdir(parents=True, exist_ok=True)
    weird = groups / "not-a-pid.jsonl.lock"
    weird.write_text("")

    sweep_orphan_groups(tmp_path)

    assert weird.exists()


def test_sweep_never_treats_lock_as_ledger(tmp_path):
    """The ledger glob (``*.jsonl``) must not pick up a ``.lock`` as a ledger."""
    groups = _groups_dir(tmp_path)
    groups.mkdir(parents=True, exist_ok=True)
    # A lone lock husk for a DEAD owner and nothing else: scanned_ledgers must be
    # zero (no ledger scanned) even though the lock is reaped.
    dead = subprocess.Popen(["/bin/sh", "-c", "true"])
    dead.wait()
    (groups / f"{dead.pid}-x.jsonl.lock").write_text("")

    result = sweep_orphan_groups(tmp_path)

    assert result.scanned_ledgers == 0  # the lock was never counted as a ledger


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


# --------------------------------------------------------------------------- #
# R2: the owner's own start token is derived once, not on every hot-path call.
# --------------------------------------------------------------------------- #
def test_self_start_token_is_cached_across_calls(monkeypatch):
    """A repeated self lookup forks ``ps`` once; the value is memoized."""
    # Reset the module memo so this test sees a cold cache regardless of order.
    monkeypatch.setattr(group_reaper, "_SELF_START_TOKEN", None)
    calls: list[int] = []
    real = group_reaper._owner_start_token

    def counting(pid):
        calls.append(pid)
        return real(pid)

    monkeypatch.setattr(group_reaper, "_owner_start_token", counting)

    me = os.getpid()
    first = group_reaper._self_start_token(me)
    second = group_reaper._self_start_token(me)
    third = group_reaper._self_start_token(me)

    assert first == second == third
    # Exactly one underlying ``ps`` derivation for three self lookups.
    assert calls == [me]


def test_self_start_token_probes_foreign_pid_live(monkeypatch):
    """A pid that is not ours is always probed live, never served from the memo."""
    monkeypatch.setattr(group_reaper, "_SELF_START_TOKEN", "SELF-CACHED")
    seen: list[int] = []

    def probe(pid):
        seen.append(pid)
        return f"token-{pid}"

    monkeypatch.setattr(group_reaper, "_owner_start_token", probe)

    other = os.getpid() + 1
    assert group_reaper._self_start_token(other) == f"token-{other}"
    assert seen == [other]  # foreign pid hit the live probe, not the memo


def test_self_start_token_failure_is_not_cached(monkeypatch):
    """A transient ``ps`` failure must not poison the memo with a sticky None."""
    monkeypatch.setattr(group_reaper, "_SELF_START_TOKEN", None)
    outcomes = iter([None, "recovered-token"])
    monkeypatch.setattr(group_reaper, "_owner_start_token", lambda pid: next(outcomes))

    me = os.getpid()
    assert group_reaper._self_start_token(me) is None  # first probe failed
    assert group_reaper._self_start_token(me) == "recovered-token"  # retried, not stuck


# --------------------------------------------------------------------------- #
# R3: a concurrent register append is not lost by unregister's rewrite.
# --------------------------------------------------------------------------- #
def test_unregister_preserves_concurrent_register(tmp_path, monkeypatch):
    """A register that lands mid-rewrite survives: the exclusive lock serialises them.

    Simulates the race deterministically by appending a new group in the middle
    of unregister's read-modify-write (patched onto the read step). Without the
    lock ordering the appended line would be truncated away; with it, unregister
    is guaranteed to observe the append (it holds the file exclusive, so the
    append inside the patch is the test's own controlled interleave) and both
    survivors remain.
    """
    pid = os.getpid()
    register_group(11, "keep-a", config_dir=tmp_path, owner_pid=pid)
    register_group(22, "drop-me", config_dir=tmp_path, owner_pid=pid)

    token = group_reaper._self_start_token(pid)
    assert token is not None
    path = group_reaper._ledger_path(tmp_path, pid, token)

    real_read_text = type(path).read_text
    injected = {"done": False}

    def read_then_append(self, *args, **kwargs):
        # First read (the unregister's own): simulate a parallel register landing
        # right here by appending a fresh line before the filter/rewrite runs.
        text = real_read_text(self, *args, **kwargs)
        if self == path and not injected["done"]:
            injected["done"] = True
            with open(path, "a", encoding="utf-8") as handle:
                handle.write(json.dumps({"pgid": 33, "owner_pid": pid}) + "\n")
            # Re-read so the rewrite sees the appended line (models the lock
            # forcing the append to be visible before the truncating write).
            text = real_read_text(self, *args, **kwargs)
        return text

    monkeypatch.setattr(type(path), "read_text", read_then_append)
    unregister_group(22, config_dir=tmp_path, owner_pid=pid)

    lines = path.read_text().splitlines()
    pgids = {json.loads(x)["pgid"] for x in lines}
    assert pgids == {11, 33}  # the concurrently-registered 33 was not lost


def test_ledger_lock_degrades_when_unavailable(tmp_path, monkeypatch):
    """If the advisory lock cannot be taken, unregister still runs (best-effort)."""
    import builtins

    real_import = builtins.__import__

    def no_fcntl(name, *args, **kwargs):
        if name == "fcntl":
            raise ImportError("no fcntl here")
        return real_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", no_fcntl)

    pid = os.getpid()
    register_group(44, "a", config_dir=tmp_path, owner_pid=pid)
    register_group(55, "b", config_dir=tmp_path, owner_pid=pid)
    unregister_group(44, config_dir=tmp_path, owner_pid=pid)

    token = group_reaper._self_start_token(pid)
    assert token is not None
    path = group_reaper._ledger_path(tmp_path, pid, token)
    pgids = {json.loads(x)["pgid"] for x in path.read_text().splitlines()}
    assert pgids == {55}  # compaction still happened without the lock
