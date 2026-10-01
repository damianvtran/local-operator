"""Unit tests for the machine memory pass.

Everything here drives the pass through its INJECTABLE seams (``runner``,
``pids_probe``, ``kill``, ``total_mb``) — no test forks a real ``ps``, spawns a
real memory hog, or signals a real pid. The pass's footsteps are deliberately
kept off the platform-dependent footprint arm: the readings below are carried by
the RSS column of the injected ``ps`` table, which both ``darwin`` and ``linux``
answer the same way, so these tests pin the DECISION (what is summed, what is
ranked, what is ended, what is withheld) rather than re-testing
``mobile.resources``'s own probes (that module's suite owns them).

Fake pids are chosen ABOVE the largest pid any supported platform can allocate
(Linux ``pid_max`` can reach 4,194,304; macOS caps at 99,998), so a live test
host cannot accidentally answer a default footprint probe with a REAL number
for a fake pid: a fake that collided with a live pid would silently rewrite the
arithmetic these tests exist to pin.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest

from local_operator.session.runtime import machine_memory as mm

_ROOT = 9900001


@pytest.fixture(autouse=True)
def _no_retry_pause(monkeypatch: pytest.MonkeyPatch) -> None:
    """The retry pause is real time between real forks; a fake runner needs none.

    Fake pids also "exist" by default: the all-gone fast path asks the kernel
    (``procstate.pid_alive``), and a fixture pid above any allocatable pid would
    otherwise read as gone and turn every retry cell into a different test. The
    cells about a gone pid say so themselves.
    """
    from local_operator import procstate

    monkeypatch.setattr(mm, "PASS_PROBE_RETRY_PAUSE_S", 0.0)
    monkeypatch.setattr(procstate, "pid_alive", lambda pid: True)


class _Killer:
    """Records the fragments a pass asks to end; always reports delivery."""

    def __init__(self) -> None:
        self.fragments: list[mm.memory_guard.Fragment] = []

    def __call__(self, fragment: mm.memory_guard.Fragment) -> bool:
        self.fragments.append(fragment)
        return True

    @property
    def pids(self) -> list[int]:
        return [fragment.pid for fragment in self.fragments]


def _no_kill(fragment: mm.memory_guard.Fragment) -> bool:
    return False


def _rss_line(pid: int, mib: int) -> str:
    """One ``ps -o pid=,rss=`` row, in KiB, for ``mib`` MiB."""
    return f"{pid} {mib * 1024}"


def _table(rows: list[tuple[int, int] | tuple[int, int, int]]) -> str:
    """A ``ps -axo pid=,ppid=,pgid=`` table from ``(pid, ppid[, pgid])`` rows.

    ``pgid`` defaults to the row's own pid — every fixture process leads its own
    group, the shape a spawned command tree has (``start_new_session``), so the
    pre-signal re-check compares against exactly the row the table showed.
    """
    lines = []
    for row in rows:
        pid, ppid = row[0], row[1]
        pgid = row[2] if len(row) > 2 else pid
        lines.append(f"{pid} {ppid} {pgid}")
    return "\n".join(lines)


def _fake_runner(
    *,
    topology: str | None = None,
    rss: dict[int, int] | None = None,
    drift: bool = False,
    drift_pids: set[int] | None = None,
    recheck_unreadable: bool = False,
) -> mm.Runner:
    """A runner keyed on argv shape: topology, the re-check rows, the RSS batch.

    A keyed fake rather than a sequence: a fake coupled to call ORDER is the
    brittleness that hides a broken pass the day a tick grows a second read.
    ``drift``/``drift_pids`` make the RE-CHECK answer a changed ``pgid`` for
    every row / for named rows — the recycled-pid shape the pre-signal withhold
    exists for (batched across the WHOLE fragment, round 2, R2-1).
    """
    rows: dict[int, tuple[int, int]] = {}
    if topology:
        for line in topology.splitlines():
            parts = line.split()
            if len(parts) >= 2:
                rows[int(parts[0])] = (
                    int(parts[1]),
                    int(parts[2]) if len(parts) >= 3 else 0,
                )

    def run(argv: list[str]) -> tuple[int, str]:
        if argv[:3] == ["ps", "-axo", "pid=,ppid=,pgid="]:
            return (0, topology) if topology is not None else (1, "")
        if argv[:3] == ["ps", "-o", "pid=,ppid=,pgid="]:
            if recheck_unreadable:
                return 1, ""
            lines = []
            for token in argv[-1].split(","):
                try:
                    pid = int(token)
                except ValueError:
                    return 1, ""
                row = rows.get(pid)
                if row is None:
                    # A missing row is a change; omit it and let the caller's
                    # comparison refuse (never invent an answer).
                    continue
                ppid, pgid = row
                if drift or (drift_pids is not None and pid in drift_pids):
                    pgid += 1
                lines.append(f"{pid} {ppid} {pgid}")
            return 0, "\n".join(lines)
        if argv[:3] == ["ps", "-o", "pid=,rss="]:
            if rss is None:
                return 1, ""
            lines = [_rss_line(pid, mib) for pid, mib in sorted(rss.items())]
            return 0, "\n".join(lines)
        # Any other probe (a `top` dump, say) is "no data", never a failure.
        return 1, ""

    return run


def _pass(**overrides: Any) -> mm.MemoryPassReport:
    """Run the pass against a 1,000 MB machine with the given fakes.

    ``Any``, not ``object``: the overrides are unpacked straight into the
    pass's typed keyword parameters, and ``object`` turns every one of them
    into a reportArgumentType error without adding real safety to a test
    helper whose overrides ARE the fakes (pyright 1.1.414).
    """
    kwargs: dict[str, Any] = {
        "runner": _fake_runner(topology=_table([(_ROOT, 1)]), rss={_ROOT: 50}),
        "pids_probe": lambda config_dir: [_ROOT],
        "total_mb": 1000,
    }
    kwargs.update(overrides)
    return mm.machine_memory_pass(Path("/nonexistent-config-root"), **kwargs)


# ---------------------------------------------------------------------------
# Reading the table
# ---------------------------------------------------------------------------


def test_parse_process_table_skips_noise_and_nonpositive_pids() -> None:
    table = "1 0 1\nnot a row\n\n-5 1 1\n42 7 7 extra\n"
    assert mm.parse_process_table(table) == [(1, 0, 1), (42, 7, 7)]


# ---------------------------------------------------------------------------
# The decision, per state
# ---------------------------------------------------------------------------


def test_ok_when_the_fleet_is_inside_the_warn_line() -> None:
    report = _pass()
    assert report.state == "ok"
    assert report.fleet_mb == 50
    assert report.runtimes == 1


def test_warn_names_the_largest_fragments_and_kills_nothing() -> None:
    killer = _Killer()
    report = _pass(
        runner=_fake_runner(
            topology=_table([(_ROOT, 1), (9900002, _ROOT)]),
            rss={_ROOT: 100, 9900002: 700},
        ),
        kill=killer,
    )
    assert report.state == "warn"
    assert report.fleet_mb == 800
    assert [fragment.pid for fragment in report.top][:1] == [9900002]
    assert killer.fragments == []
    assert report.killed is None


def test_act_ends_the_largest_fragment_at_or_above_the_floor() -> None:
    killer = _Killer()
    report = _pass(
        runner=_fake_runner(
            topology=_table(
                [
                    (_ROOT, 1),
                    (9900002, _ROOT),
                    (9900003, 9900002),
                    (9900004, _ROOT),
                ]
            ),
            rss={_ROOT: 50, 9900002: 500, 9900003: 600, 9900004: 300},
        ),
        kill=killer,
    )
    assert report.state == "act"
    assert report.fleet_mb == 1450
    # 9900002's fragment is 500 + 600 = 1100 MB, the largest; 9900003 alone
    # (600) and 9900004 (300) are smaller.
    assert killer.pids == [9900002]
    assert report.killed is not None and report.killed.pid == 9900002
    assert report.killed.mb == 1100
    # The stop receives the exact subtree the sum counted, not a bare pid.
    assert report.killed.pids == (9900002, 9900003)


def test_act_with_no_fragment_over_the_floor_warns_only() -> None:
    killer = _Killer()
    report = _pass(
        runner=_fake_runner(
            topology=_table([(_ROOT, 1), (9900002, _ROOT), (9900003, _ROOT), (9900004, _ROOT)]),
            rss={_ROOT: 100, 9900002: 400, 9900003: 400, 9900004: 400},
        ),
        kill=killer,
    )
    assert report.state == "act"
    assert killer.fragments == []
    assert "no single fragment reaches" in report.reason


def test_kill_is_withheld_when_the_seat_says_so() -> None:
    killer = _Killer()
    report = _pass(
        runner=_fake_runner(
            topology=_table([(_ROOT, 1), (9900002, _ROOT), (9900003, 9900002)]),
            rss={_ROOT: 50, 9900002: 500, 9900003: 600},
        ),
        kill=killer,
        kill_allowed=False,
    )
    assert report.state == "act"
    assert killer.fragments == []
    assert report.kill_withheld is True
    assert "withheld" in report.reason
    assert "within the cooldown" in report.summary()


def test_apply_false_measures_and_never_kills() -> None:
    killer = _Killer()
    report = _pass(
        runner=_fake_runner(
            topology=_table([(_ROOT, 1), (9900002, _ROOT), (9900003, 9900002)]),
            rss={_ROOT: 50, 9900002: 500, 9900003: 600},
        ),
        kill=killer,
        apply=False,
    )
    assert report.state == "act"
    assert killer.fragments == []
    assert "does not apply" in report.reason


# ---------------------------------------------------------------------------
# The fail-closed rungs
# ---------------------------------------------------------------------------


def test_unreadable_table_is_unknown_and_never_kills() -> None:
    killer = _Killer()
    report = _pass(runner=_fake_runner(topology=None), kill=killer)
    assert report.state == "unknown"
    assert killer.fragments == []


def test_unmeasurable_host_is_unknown_and_never_kills(monkeypatch) -> None:
    # The production shape: ``total_mb`` is left None and the verdict measures
    # the HOST — a host probe that cannot answer must not authorise a stop,
    # however large a fragment is sitting in front of the pass.
    monkeypatch.setattr(mm.memory_guard, "_total_memory_mb", lambda: None)
    killer = _Killer()
    report = _pass(
        runner=_fake_runner(
            topology=_table([(_ROOT, 1), (9900002, _ROOT), (9900003, 9900002)]),
            rss={_ROOT: 50, 9900002: 500, 9900003: 600},
        ),
        kill=killer,
        total_mb=None,
    )
    assert report.state == "unknown"
    assert killer.fragments == []


def test_total_mb_zero_is_unknown_too(monkeypatch) -> None:
    killer = _Killer()
    report = _pass(
        runner=_fake_runner(
            topology=_table([(_ROOT, 1), (9900002, _ROOT), (9900003, 9900002)]),
            rss={_ROOT: 50, 9900002: 500, 9900003: 600},
        ),
        kill=killer,
        total_mb=0,
    )
    assert report.state == "unknown"
    assert killer.fragments == []


def test_a_changed_descendant_row_withholds_the_whole_stop() -> None:
    """R2-1: every pid the ranking summed is a signal target and is re-checked.

    The root stands; the DESCENDANT has drifted. The stop must be withheld —
    the walk signals descendants too, and a recycled pid that leads a group
    would take that whole group.
    """
    killer = _Killer()
    report = _pass(
        runner=_fake_runner(
            topology=_table([(_ROOT, 1), (9900002, _ROOT), (9900003, 9900002)]),
            rss={_ROOT: 50, 9900002: 500, 9900003: 600},
            drift_pids={9900003},
        ),
        kill=killer,
    )
    assert report.state == "act"
    assert killer.fragments == []
    assert report.kill_withheld is True
    assert "9900003" in report.reason
    assert "the fragment changed before the signal" in report.summary()


def test_a_changed_candidate_row_withholds_the_stop() -> None:
    # The candidate's row answers with a different pgid at re-check time: the
    # snapshot is stale, so the stop is withheld, not delivered.
    killer = _Killer()
    report = _pass(
        runner=_fake_runner(
            topology=_table([(_ROOT, 1), (9900002, _ROOT), (9900003, 9900002)]),
            rss={_ROOT: 50, 9900002: 500, 9900003: 600},
            drift=True,
        ),
        kill=killer,
    )
    assert report.state == "act"
    assert killer.fragments == []
    assert report.kill_withheld is True
    assert "changed before the signal" in report.reason


def test_an_unreadable_recheck_states_that_cause() -> None:
    """R3-2: 'could not be re-read' must not read as 'changed'.

    A ``ps`` that will not answer is a different operator destination from a
    snapshot that moved; the summary must say which one happened.
    """
    killer = _Killer()
    report = _pass(
        runner=_fake_runner(
            topology=_table([(_ROOT, 1), (9900002, _ROOT), (9900003, 9900002)]),
            rss={_ROOT: 50, 9900002: 500, 9900003: 600},
            recheck_unreadable=True,
        ),
        kill=killer,
    )
    assert report.state == "act"
    assert killer.fragments == []
    assert report.kill_withheld is True
    assert "could not be re-read" in report.summary()
    assert "changed before the signal" not in report.summary()


def test_no_live_runtimes_is_empty_and_never_kills() -> None:
    killer = _Killer()
    report = _pass(pids_probe=lambda config_dir: [], kill=killer)
    assert report.state == "empty"
    assert killer.fragments == []


def test_unmeasured_processes_are_counted_not_guessed() -> None:
    # 9900002 has a table row but no RSS row: nothing can be read for it, and
    # the pass must say so rather than treat it as free.
    report = _pass(
        runner=_fake_runner(topology=_table([(_ROOT, 1), (9900002, _ROOT)]), rss={_ROOT: 50}),
    )
    assert report.unmeasured == 1
    assert report.measured == 1
    assert report.fleet_mb == 50


def test_a_kill_that_delivers_nothing_is_not_recorded_as_one() -> None:
    report = _pass(
        runner=_fake_runner(
            topology=_table([(_ROOT, 1), (9900002, _ROOT), (9900003, 9900002)]),
            rss={_ROOT: 50, 9900002: 500, 9900003: 600},
        ),
        kill=_no_kill,
    )
    assert report.state == "act"
    assert report.killed is None


# ---------------------------------------------------------------------------
# 2026-09-30: the pass's reads survive pressure; the stop still needs identity
# ---------------------------------------------------------------------------

_FRAGMENT_TOPOLOGY = _table([(_ROOT, 1), (9900002, _ROOT), (9900003, 9900002)])
_FRAGMENT_RSS = {_ROOT: 50, 9900002: 500, 9900003: 600}


def _identity_probe(rows: dict[int, tuple[int, int]]) -> mm.IdentityProbe:
    """A fork-free identity reader over a fixed table; a missing pid is gone."""
    return lambda pid: rows.get(pid)


def _flaky(inner: mm.Runner, *, match: list[str], failures: int) -> tuple[mm.Runner, list[int]]:
    """``inner`` with the first ``failures`` calls of one argv shape failing."""
    calls: list[int] = []

    def run(argv: list[str]) -> tuple[int, str]:
        if argv[: len(match)] == match:
            calls.append(1)
            if len(calls) <= failures:
                return 1, ""
        return inner(argv)

    return run, calls


def test_the_table_read_is_retried_before_the_pass_gives_up() -> None:
    inner = _fake_runner(topology=_FRAGMENT_TOPOLOGY, rss=_FRAGMENT_RSS)
    runner, calls = _flaky(inner, match=["ps", "-axo", "pid=,ppid=,pgid="], failures=2)
    report = _pass(runner=runner, kill=_Killer())
    assert report.state == "act" and len(calls) == 3


def test_a_table_that_never_answers_is_still_unknown_and_never_kills() -> None:
    killer = _Killer()
    inner = _fake_runner(topology=_FRAGMENT_TOPOLOGY, rss=_FRAGMENT_RSS)
    runner, calls = _flaky(inner, match=["ps", "-axo", "pid=,ppid=,pgid="], failures=99)
    report = _pass(runner=runner, kill=killer)
    assert report.state == "unknown" and killer.fragments == []
    assert len(calls) == mm.PASS_PROBE_ATTEMPTS


def test_the_recheck_read_is_retried_and_a_late_answer_lets_the_stop_proceed() -> None:
    killer = _Killer()
    inner = _fake_runner(topology=_FRAGMENT_TOPOLOGY, rss=_FRAGMENT_RSS)
    runner, calls = _flaky(inner, match=["ps", "-o", "pid=,ppid=,pgid="], failures=2)
    report = _pass(runner=runner, kill=killer, identity_probe=_identity_probe({}))
    assert killer.pids == [9900002] and report.killed is not None
    assert len(calls) == 3  # the fork-free fallback was NOT needed


def test_ps_unreadable_but_fork_free_identity_holds_the_stop_proceeds() -> None:
    """(e) The re-check ps never answers; every row is confirmed by syscall."""
    killer = _Killer()
    rows = {_ROOT: (1, _ROOT), 9900002: (_ROOT, 9900002), 9900003: (9900002, 9900003)}
    report = _pass(
        runner=_fake_runner(
            topology=_FRAGMENT_TOPOLOGY, rss=_FRAGMENT_RSS, recheck_unreadable=True
        ),
        kill=killer,
        identity_probe=_identity_probe(rows),
    )
    assert killer.pids == [9900002]
    assert report.killed is not None and report.kill_withheld is False


def test_ps_unreadable_and_the_identity_changed_withholds_the_stop() -> None:
    killer = _Killer()
    rows = {_ROOT: (1, _ROOT), 9900002: (_ROOT, 9900002), 9900003: (9900002, 4242)}  # pgid moved
    report = _pass(
        runner=_fake_runner(
            topology=_FRAGMENT_TOPOLOGY, rss=_FRAGMENT_RSS, recheck_unreadable=True
        ),
        kill=killer,
        identity_probe=_identity_probe(rows),
    )
    assert killer.fragments == [] and report.kill_withheld is True
    assert report.withheld_cause == "changed" and "9900003" in report.reason


def test_ps_unreadable_and_a_pid_is_gone_or_unreadable_withholds_the_stop() -> None:
    killer = _Killer()
    rows = {_ROOT: (1, _ROOT), 9900002: (_ROOT, 9900002)}  # 9900003 has vanished
    report = _pass(
        runner=_fake_runner(
            topology=_FRAGMENT_TOPOLOGY, rss=_FRAGMENT_RSS, recheck_unreadable=True
        ),
        kill=killer,
        identity_probe=_identity_probe(rows),
    )
    assert killer.fragments == [] and report.kill_withheld is True
    assert report.withheld_cause == "unreadable"
    assert "could not be re-read" in report.summary()


def test_a_raising_identity_probe_withholds_rather_than_kills() -> None:
    def boom(pid: int) -> tuple[int, int] | None:
        raise OSError("libproc")

    killer = _Killer()
    report = _pass(
        runner=_fake_runner(
            topology=_FRAGMENT_TOPOLOGY, rss=_FRAGMENT_RSS, recheck_unreadable=True
        ),
        kill=killer,
        identity_probe=boom,
    )
    assert killer.fragments == [] and report.withheld_cause == "unreadable"


def test_the_default_identity_probe_confirms_a_real_process_and_refuses_a_fake_one() -> None:
    import os

    assert mm._default_identity_probe(os.getpid()) == (os.getppid(), os.getpgid(0))
    assert mm._default_identity_probe(9900001) is None  # above any allocatable pid


def test_a_runtime_root_is_never_a_candidate_even_when_the_recheck_is_dead() -> None:
    """No new kill authority: the fallback only ever confirms a fragment the
    ranking chose, and the ranking never chooses a root."""
    killer = _Killer()
    _pass(
        runner=_fake_runner(
            topology=_table([(_ROOT, 1)]), rss={_ROOT: 900}, recheck_unreadable=True
        ),
        kill=killer,
        identity_probe=_identity_probe({_ROOT: (1, _ROOT)}),
    )
    assert killer.fragments == []


def test_a_candidate_in_cooldown_is_named_and_withheld_and_another_is_not() -> None:
    killer = _Killer()
    held: list[frozenset[tuple[str, int]]] = []

    def in_cooldown(lineage: frozenset[tuple[str, int]]) -> bool:
        held.append(lineage)
        return True

    report = _pass(
        runner=_fake_runner(topology=_FRAGMENT_TOPOLOGY, rss=_FRAGMENT_RSS),
        kill=killer,
        in_cooldown=in_cooldown,
    )
    assert killer.fragments == [] and len(held) == 1 and ("pid", 9900002) in held[0]
    assert report.withheld_cause == "cooldown" and "pid 9900002" in report.reason
    killed = _pass(
        runner=_fake_runner(topology=_FRAGMENT_TOPOLOGY, rss=_FRAGMENT_RSS),
        kill=killer,
        in_cooldown=lambda lineage: False,
        identity_probe=_identity_probe({}),
    )
    assert killed.killed is not None
    assert ("pid", 9900002) in killed.killed_lineage


def test_a_runtimes_own_pid_and_group_are_never_lineage_keys() -> None:
    """R2/Q2: every command of one session is a child of the runtime, and half of
    them share its process group. Keying on either made the cooldown per-SESSION,
    so a second unrelated runaway under the same runtime was withheld for 10 min."""
    rows = {_ROOT: (1, _ROOT), 9900002: (_ROOT, _ROOT), 9900003: (_ROOT, 9900003)}
    shares_runtime_group = mm.memory_guard.Fragment(pid=9900002, mb=1, ppid=_ROOT, pgid=_ROOT)
    own_group = mm.memory_guard.Fragment(pid=9900003, mb=1, ppid=_ROOT, pgid=9900003)
    first = mm.lineage_keys(shares_runtime_group, rows, [_ROOT])
    second = mm.lineage_keys(own_group, rows, [_ROOT])
    assert first == frozenset({("pid", 9900002)})
    assert second == frozenset({("pid", 9900003), ("pgid", 9900003)})
    assert not (first & second)


def test_a_fragment_under_a_non_runtime_parent_keeps_that_parent_as_a_key() -> None:
    """A respawning supervisor (``timeout``, a runner) is NOT a runtime, so its
    children stay kin through it — that is the regrowth the cooldown is for."""
    rows = {_ROOT: (1, _ROOT), 9900005: (_ROOT, 9900005), 9900006: (9900005, 9900005)}
    fragment = mm.memory_guard.Fragment(pid=9900006, mb=1, ppid=9900005, pgid=9900005)
    keys = mm.lineage_keys(fragment, rows, [_ROOT])
    assert ("ppid", 9900005) in keys and ("pgid", 9900005) in keys


# -- owner notification ------------------------------------------------------


def _notified_pass(**overrides: Any) -> tuple[mm.MemoryPassReport, list[mm.KillEvent]]:
    events: list[mm.KillEvent] = []
    report = _pass(
        runner=_fake_runner(topology=_FRAGMENT_TOPOLOGY, rss=_FRAGMENT_RSS),
        identity_probe=_identity_probe({}),
        notify=lambda config_dir, event: events.append(event),
        **overrides,
    )
    return report, events


def test_a_delivered_stop_tells_the_owning_runtime_with_the_numbers() -> None:
    report, events = _notified_pass(kill=_Killer())
    assert report.killed is not None and len(events) == 1
    event = events[0]
    assert event.owner_runtime_pid == _ROOT
    assert event.fragment.pid == 9900002
    assert event.fragment.mb == 1100 and event.act_mb == 850 and event.total_mb == 1000
    text = mm.owner_notice_text(event)
    assert "your process group (pid 9900002" in text
    assert "was ended by the memory guard" in text and "1.1 GB footprint" in text


def test_a_withheld_or_undelivered_stop_tells_nobody() -> None:
    _, events = _notified_pass(kill=_no_kill)
    assert events == []
    _, events = _notified_pass(kill=_Killer(), apply=False)
    assert events == []
    _, events = _notified_pass(kill=_Killer(), kill_allowed=False)
    assert events == []


def test_a_notifier_that_raises_never_undoes_the_stop() -> None:
    def boom(config_dir: Path, event: mm.KillEvent) -> None:
        raise RuntimeError("socket")

    report = _pass(
        runner=_fake_runner(topology=_FRAGMENT_TOPOLOGY, rss=_FRAGMENT_RSS),
        kill=_Killer(),
        identity_probe=_identity_probe({}),
        notify=boom,
    )
    assert report.killed is not None


def test_the_kill_is_logged_with_who_what_and_how_big(caplog: pytest.LogCaptureFixture) -> None:
    caplog.set_level("WARNING", logger=mm.logger.name)
    _notified_pass(kill=_Killer())
    line = next(r.getMessage() for r in caplog.records if "machine memory kill:" in r.getMessage())
    for needle in (
        f"owner_runtime_pid={_ROOT}",
        "fragment_pid=9900002",
        "footprint_mb=1100",
        "act_mb=850",
        "total_mb=1000",
        "measured=3",
        "unmeasured=0",
    ):
        assert needle in line, (needle, line)


def _event(owner: int | None = _ROOT) -> mm.KillEvent:
    fragment = mm.memory_guard.Fragment(pid=9900002, mb=4096, pids=(9900002, 9900003))
    return mm.KillEvent(
        fragment=fragment,
        owner_runtime_pid=owner,
        fleet_mb=900,
        act_mb=850,
        total_mb=1000,
        measured=3,
        unmeasured=0,
        cause="test",
    )


def test_the_notice_is_spooled_to_the_owning_session_when_the_dial_fails(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The session inbox is the transport that survives a runtime that will not
    answer: the row lands under ``sessions/<id>/`` and the next drain delivers it."""
    from local_operator.session.runtime import inbox, registry

    record = type("R", (), {"pid": _ROOT, "session_id": "abc123abc123"})()
    monkeypatch.setattr(registry, "scan", lambda *a, **k: [(record, "live")])

    async def refuse(*args: Any, **kwargs: Any) -> str:
        raise ConnectionRefusedError("no listener")

    monkeypatch.setattr("local_operator.mobile.peer_client.send_peer_message", refuse)
    assert mm.notify_owner_of_kill(tmp_path, _event()) == "spooled"
    lines = inbox.peek_inbox(tmp_path / "sessions" / "abc123abc123")
    assert len(lines) == 1
    assert "your process group (pid 9900002, 2 processes, 4.0 GB footprint)" in lines[0].text
    assert lines[0].wake is False and lines[0].source == inbox.SOURCE_PEER


def test_a_live_owner_is_dialled_and_nothing_is_spooled(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from local_operator.session.runtime import registry

    record = type("R", (), {"pid": _ROOT, "session_id": "abc123abc123"})()
    monkeypatch.setattr(registry, "scan", lambda *a, **k: [(record, "live")])
    sent: list[dict[str, Any]] = []

    async def accept(rec: Any, **kwargs: Any) -> str:
        sent.append(kwargs)
        return "delivered"

    monkeypatch.setattr("local_operator.mobile.peer_client.send_peer_message", accept)
    assert mm.notify_owner_of_kill(tmp_path, _event()) == "dialled"
    assert len(sent) == 1 and sent[0]["wake"] is False and sent[0]["mode"] == "mailbox"
    assert not (tmp_path / "sessions").exists()


def test_no_owning_runtime_means_the_owner_is_not_told_and_nothing_raises(
    tmp_path: Path,
) -> None:
    assert mm.notify_owner_of_kill(tmp_path, _event(owner=None)) == "not_told"
    assert json.dumps(mm.owner_notice_text(_event()))  # renders without a registry


def test_a_spooled_notice_is_logged_as_queued_never_as_told(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
) -> None:
    """R5: a spool is read at the next runtime OPEN, so a live-but-unresponsive
    runtime has NOT been told; the log must not say it was."""
    from local_operator.session.runtime import registry

    record = type("R", (), {"pid": _ROOT, "session_id": "abc123abc123"})()
    monkeypatch.setattr(registry, "scan", lambda *a, **k: [(record, "live")])

    async def refuse(*args: Any, **kwargs: Any) -> str:
        raise TimeoutError("no ack")

    monkeypatch.setattr("local_operator.mobile.peer_client.send_peer_message", refuse)
    caplog.set_level("INFO", logger=mm.logger.name)
    assert mm.notify_owner_of_kill(tmp_path, _event()) == "spooled"
    text = " ".join(r.getMessage() for r in caplog.records)
    assert "QUEUED, not delivered" in text and "told session" not in text


def test_all_pids_gone_is_one_ps_call_and_reads_as_changed_not_unreadable(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """R7: ``ps -p`` exits 1 with empty output when every pid has gone. That is an
    answer: not retried (3 forks), and the cause is "changed", not "unreadable"."""
    from local_operator import procstate

    monkeypatch.setattr(procstate, "pid_alive", lambda pid: False)
    inner = _fake_runner(topology=_FRAGMENT_TOPOLOGY, rss=_FRAGMENT_RSS, recheck_unreadable=True)
    runner, calls = _flaky(inner, match=["ps", "-o", "pid=,ppid=,pgid="], failures=99)
    killer = _Killer()
    report = _pass(runner=runner, kill=killer, identity_probe=_identity_probe({}))
    assert len(calls) == 1
    assert killer.fragments == [] and report.withheld_cause == "changed"
    assert "gone" in report.reason and "could not be re-read" not in report.reason


def test_the_retry_budget_bounds_the_whole_read_not_one_attempt(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """R4: no new attempt starts once the budget is spent."""
    clock = [0.0]
    monkeypatch.setattr(mm.time, "monotonic", lambda: clock[0])
    monkeypatch.setattr(mm.time, "sleep", lambda s: clock.__setitem__(0, clock[0] + s))
    monkeypatch.setattr(mm, "PASS_PROBE_RETRY_PAUSE_S", 0.5)
    calls: list[int] = []

    def slow_fail(argv: list[str]) -> tuple[int, str]:
        calls.append(1)
        clock[0] += mm.PASS_READ_BUDGET_S  # one attempt eats the whole budget
        return 1, ""

    assert mm._run_with_retry(slow_fail, ["ps"]) == (1, "")
    assert len(calls) == 1


def test_the_default_identity_probe_never_signals_a_process() -> None:
    """R1: ``os.kill(pid, 0)`` terminates the process on Windows, and the xplat probe
    refuses it. The identity probe must not contain the call."""
    import inspect

    source = inspect.getsource(mm._default_identity_probe)
    assert "os.kill" not in source.replace("``os.kill(pid, 0)``", "")
