"""Unit tests for the machine memory pass.

Everything here drives the pass through its INJECTABLE seams (``runner``,
``pids_probe``, ``kill``, ``total_mb``) — no test forks a real ``ps``, spawns a
real memory hog, or signals a real pid. The pass's footsteps are deliberately
kept off the platform-dependent footprint arm: the readings below are carried by
the RSS column of the injected ``ps`` table, which both ``darwin`` and ``linux``
answer the same way, so these tests pin the DECISION (what is summed, what is
ranked, what is ended) rather than re-testing ``mobile.resources``'s own probes
(that module's suite owns them).

Fake pids are chosen far above any real allocation (990001+) so that a default
footprint probe on a live test host cannot accidentally return a REAL number
for them: a fake that collided with a live pid would silently rewrite the
arithmetic these tests exist to pin.
"""

from __future__ import annotations

from pathlib import Path

from local_operator.session.runtime import machine_memory as mm

_ROOT = 990001


class _Killer:
    """Records the pids a pass asks to end; always reports delivery."""

    def __init__(self) -> None:
        self.pids: list[int] = []

    def __call__(self, pid: int) -> bool:
        self.pids.append(pid)
        return True


def _no_kill(pid: int) -> bool:
    return False


def _rss_line(pid: int, mib: int) -> str:
    """One ``ps -o pid=,rss=`` row, in KiB, for ``mib`` MiB."""
    return f"{pid} {mib * 1024}"


def _table(rows: list[tuple[int, int]]) -> str:
    """A ``ps -axo pid=,ppid=`` table from ``[(pid, ppid)]``."""
    return "\n".join(f"{pid} {ppid}" for pid, ppid in rows)


def _fake_runner(
    *,
    topology: str | None = None,
    rss: dict[int, int] | None = None,
) -> mm.Runner:
    """A runner keyed on argv shape: the topology read, then the RSS batch.

    A keyed fake rather than a sequence: the pass makes exactly one topology
    read and one RSS read per ``session_resource_usage`` call today, but a fake
    coupled to call ORDER is the brittleness that hides a broken pass the day a
    tick grows a second read.
    """

    def run(argv: list[str]) -> tuple[int, str]:
        if argv[:3] == ["ps", "-axo", "pid=,ppid="]:
            return (0, topology) if topology is not None else (1, "")
        if argv[:3] == ["ps", "-o", "pid=,rss="]:
            if rss is None:
                return 1, ""
            lines = [_rss_line(pid, mib) for pid, mib in sorted(rss.items())]
            return 0, "\n".join(lines)
        # Any other probe (a `top` dump, say) is "no data", never a failure.
        return 1, ""

    return run


def _pass(**overrides: object) -> mm.MemoryPassReport:
    """Run the pass against a 1,000 MB machine with the given fakes."""
    kwargs: dict[str, object] = {
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
    table = "1 0\nnot a row\n\n-5 1\n42 7 extra\n"
    assert mm.parse_process_table(table) == [(1, 0), (42, 7)]


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
            topology=_table([(_ROOT, 1), (990002, _ROOT)]),
            rss={_ROOT: 100, 990002: 700},
        ),
        kill=killer,
    )
    assert report.state == "warn"
    assert report.fleet_mb == 800
    assert [fragment.pid for fragment in report.top][:1] == [990002]
    assert killer.pids == []
    assert report.killed is None


def test_act_ends_the_largest_fragment_at_or_above_the_floor() -> None:
    killer = _Killer()
    report = _pass(
        runner=_fake_runner(
            topology=_table(
                [
                    (_ROOT, 1),
                    (990002, _ROOT),
                    (990003, 990002),
                    (990004, _ROOT),
                ]
            ),
            rss={_ROOT: 50, 990002: 500, 990003: 600, 990004: 300},
        ),
        kill=killer,
    )
    assert report.state == "act"
    assert report.fleet_mb == 1450
    # 990002's fragment is 500 + 600 = 1100 MB, the largest; 990003 alone (600)
    # and 990004 (300) are smaller.
    assert killer.pids == [990002]
    assert report.killed is not None and report.killed.pid == 990002
    assert report.killed.mb == 1100


def test_act_with_no_fragment_over_the_floor_warns_only() -> None:
    killer = _Killer()
    report = _pass(
        runner=_fake_runner(
            topology=_table(
                [(_ROOT, 1), (990002, _ROOT), (990003, _ROOT), (990004, _ROOT)]
            ),
            rss={_ROOT: 100, 990002: 400, 990003: 400, 990004: 400},
        ),
        kill=killer,
    )
    assert report.state == "act"
    assert killer.pids == []
    assert "no single fragment reaches" in report.reason


def test_kill_is_withheld_when_the_seat_says_so() -> None:
    killer = _Killer()
    report = _pass(
        runner=_fake_runner(
            topology=_table([(_ROOT, 1), (990002, _ROOT), (990003, 990002)]),
            rss={_ROOT: 50, 990002: 500, 990003: 600},
        ),
        kill=killer,
        kill_allowed=False,
    )
    assert report.state == "act"
    assert killer.pids == []
    assert report.kill_withheld is True
    assert "withheld" in report.reason


def test_apply_false_measures_and_never_kills() -> None:
    killer = _Killer()
    report = _pass(
        runner=_fake_runner(
            topology=_table([(_ROOT, 1), (990002, _ROOT), (990003, 990002)]),
            rss={_ROOT: 50, 990002: 500, 990003: 600},
        ),
        kill=killer,
        apply=False,
    )
    assert report.state == "act"
    assert killer.pids == []
    assert "does not apply" in report.reason


# ---------------------------------------------------------------------------
# The fail-closed rungs
# ---------------------------------------------------------------------------


def test_unreadable_table_is_unknown_and_never_kills() -> None:
    killer = _Killer()
    report = _pass(runner=_fake_runner(topology=None), kill=killer)
    assert report.state == "unknown"
    assert killer.pids == []


def test_no_live_runtimes_is_empty_and_never_kills() -> None:
    killer = _Killer()
    report = _pass(pids_probe=lambda config_dir: [], kill=killer)
    assert report.state == "empty"
    assert killer.pids == []


def test_unmeasured_processes_are_counted_not_guessed() -> None:
    # 990002 has a table row but no RSS row: nothing can be read for it, and the
    # pass must say so rather than treat it as free.
    report = _pass(
        runner=_fake_runner(topology=_table([(_ROOT, 1), (990002, _ROOT)]), rss={_ROOT: 50}),
    )
    assert report.unmeasured == 1
    assert report.measured == 1
    assert report.fleet_mb == 50


def test_a_kill_that_delivers_nothing_is_not_recorded_as_one() -> None:
    report = _pass(
        runner=_fake_runner(
            topology=_table([(_ROOT, 1), (990002, _ROOT), (990003, 990002)]),
            rss={_ROOT: 50, 990002: 500, 990003: 600},
        ),
        kill=_no_kill,
    )
    assert report.state == "act"
    assert report.killed is None
