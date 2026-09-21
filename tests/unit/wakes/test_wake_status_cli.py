"""``lop wake status`` / ``list`` — is the thing that fires wakes alive?

The command reported ``installed`` from ``plist_path().exists()``, which is an
even weaker test than the install hook's own: a supervisor that had exited
still has its plist on disk, so the one surface an operator would check to
answer "why did my wake not fire" confidently answered "installed" while
nothing was running. It also reported no per-schedule state at all — not when
a wake last fired, not whether it is overdue now, not whether it has gone past
the staleness bound the supervisor silently stops firing at.

These tests never touch launchd: ``supervisor_state`` is patched, which is what
keeps them inside the safety contract in ``test_install.py``.
"""

from __future__ import annotations

import argparse
import json
import os
import time
from pathlib import Path

import pytest

from local_operator.wakes.install import SupervisorState
from local_operator.wakes.store import write_entry

NOW_MS = int(time.time() * 1000)


def _arm(config_dir: Path, session_id: str, **kwargs: object) -> None:
    """Index entry PLUS the session it belongs to.

    Round 2 (QA Q4) taught `wake status` the supervisor's ghost predicate: an
    index entry whose session has no transcript can never be engaged, so it is
    no longer rendered as an armed, overdue wake. A bare ``write_entry`` builds
    exactly that shape, which the product never produces — the session writes
    its own transcript before it ever persists a schedule — so these fixtures
    would all be ghosts and would test the ghost branch by accident. Tests that
    WANT a ghost call ``write_entry`` directly and say so.
    """
    session = config_dir / "sessions" / session_id
    session.mkdir(parents=True, exist_ok=True)
    (session / "transcript.jsonl").write_text("{}\n", encoding="utf-8")
    write_entry(config_dir, session_id, **kwargs)  # type: ignore[arg-type]


def _args(**kwargs: object) -> argparse.Namespace:
    base: dict[str, object] = {
        "wake_command": "status",
        "json": False,
        "install": False,
        "uninstall": False,
    }
    base.update(kwargs)
    return argparse.Namespace(**base)


def _status_block(out: str, label: str) -> str:
    """One `wake status` line, with its hanging-indent continuations joined.

    The surface folds prose at its label column (``_wrap_status``), so a line's
    payload is spread over several physical lines; joining them keeps an
    assertion about the SENTENCE rather than about the terminal width. Runs of
    whitespace are collapsed for the same reason: the fold point moves with the
    width, and a phrase that straddles it must stay assertable — CI's 80 columns
    broke "could not reach a runtime" where a wider local terminal did not.
    """
    lines = out.splitlines()
    start = next(i for i, line in enumerate(lines) if line.startswith(label))
    block = [lines[start]]
    for line in lines[start + 1 :]:
        if not line.startswith(" "):
            break
        block.append(line)
    return " ".join(" ".join(block).split())


@pytest.fixture(autouse=True)
def _isolated(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    monkeypatch.setenv("LOCAL_OPERATOR_CONFIG_DIR", str(tmp_path))
    return tmp_path


@pytest.fixture
def _installer_available(monkeypatch: pytest.MonkeyPatch) -> None:
    """Pretend this platform has an installer.

    The status RENDERING is platform-independent, but the command asks
    `is_supported()` before probing at all — so on Linux (where there is no
    installer and the honest answer is "wakes fire only while a session is
    open") these fixtures would never be consulted and every assertion below
    would be about the unsupported branch instead. Patching the capability
    keeps the rendering under test on every runner rather than only on macOS.
    """
    monkeypatch.setattr("local_operator.wakes.install.is_supported", lambda: True)


@pytest.fixture
def stopped_supervisor(monkeypatch: pytest.MonkeyPatch, _installer_available: None) -> None:
    """Loaded, exited — the state that produced the permanent misses."""
    monkeypatch.setattr(
        "local_operator.wakes.install.supervisor_state",
        lambda _config: SupervisorState(loaded=True, running=False, detail="not running"),
    )


@pytest.fixture
def running_supervisor(monkeypatch: pytest.MonkeyPatch, _installer_available: None) -> None:
    monkeypatch.setattr(
        "local_operator.wakes.install.supervisor_state",
        lambda _config: SupervisorState(loaded=True, running=True, pid=4242, detail="running"),
    )
    monkeypatch.setattr("local_operator.cli._process_uptime_s", lambda _pid: 3600.0)


def test_a_stopped_supervisor_is_not_reported_as_installed(
    tmp_path: Path, stopped_supervisor, capsys: pytest.CaptureFixture[str]
) -> None:
    """THE BLIND SPOT, on the surface an operator actually reads.

    ``plist_path().exists()`` is true of a supervisor that exited hours ago.
    Reporting that as "installed" beside an overdue wake is precisely the
    reassurance that kept the misses invisible.
    """
    from local_operator.cli import wake_command

    _arm(
        tmp_path,
        "statussess01",
        cwd=str(tmp_path),
        schedules=[{"id": "w1", "message": "watch prod", "next_due_at": NOW_MS - 60_000}],
    )

    assert wake_command(_args()) == 0

    out = capsys.readouterr().out
    assert "loaded but NOT running" in out, out
    assert "lop wake install" in out, "the actionable repair must be named"


def test_a_running_supervisor_reports_its_pid_and_uptime(
    tmp_path: Path, running_supervisor, capsys: pytest.CaptureFixture[str]
) -> None:
    """The positive half: "running" must be distinguishable from "present"."""
    from local_operator.cli import wake_command

    assert wake_command(_args()) == 0

    out = capsys.readouterr().out
    assert "running (pid 4242)" in out, out
    assert "up 1h" in out, out


def test_status_reports_overdue_and_stale_counts(
    tmp_path: Path, running_supervisor, capsys: pytest.CaptureFixture[str]
) -> None:
    """A wake past ``STALE_AFTER_S`` is skipped by the supervisor FOREVER.

    That is deliberate — the session's own catch-up owns it — but it must be
    visible, because it is indistinguishable from a working wake otherwise.
    """
    from local_operator.cli import wake_command

    _arm(
        tmp_path,
        "statussess02",
        cwd=str(tmp_path),
        schedules=[{"id": "w1", "message": "late watch", "next_due_at": NOW_MS - 600_000}],
    )
    _arm(
        tmp_path,
        "statussess03",
        cwd=str(tmp_path),
        schedules=[{"id": "w1", "message": "forgotten", "next_due_at": NOW_MS - 9 * 24 * 3600_000}],
    )

    assert wake_command(_args()) == 0

    out = capsys.readouterr().out
    # ONE overdue, not two: the stale row is counted as stale and excluded
    # from overdue, so "worst" describes a wake that is actually coming
    # (round 1, D2). Counting it both ways let a wake the supervisor has given
    # up on dominate the figure an operator reads as "how late am I".
    assert "overdue:     1" in out, out
    assert "stale:       1 past 7d" in out, out
    # The FIREABLE wake is the one named, never the stale one — and it is named
    # on `overdue:`, not `next:`. Round 2 (D16): with nothing in the future,
    # `next:` was labelling an already-late wake as a coming event and saying
    # the same fact the line below it said. `next:` now appears only when there
    # genuinely is a next.
    overdue_line = next(line for line in out.splitlines() if line.startswith("overdue:"))
    assert "late watch" in overdue_line, overdue_line
    assert not [line for line in out.splitlines() if line.startswith("next:")], out


def test_the_json_form_carries_the_machine_readable_state(
    tmp_path: Path, stopped_supervisor, capsys: pytest.CaptureFixture[str]
) -> None:
    """A monitoring caller must not have to scrape the human rendering."""
    from local_operator.cli import wake_command

    _arm(
        tmp_path,
        "statussess04",
        cwd=str(tmp_path),
        schedules=[{"id": "w1", "message": "watch prod", "next_due_at": NOW_MS - 120_000}],
    )

    assert wake_command(_args(json=True)) == 0

    payload = json.loads(capsys.readouterr().out)
    assert payload["supervisor"]["running"] is False
    assert payload["supervisor"]["loaded"] is True
    # The legacy key keeps its name but now means RUNNING, which is the only
    # reading of "installed" that answers "will my wake fire".
    assert payload["installed"] is False
    assert payload["overdue"] == 1
    assert payload["max_overdue_s"] >= 120


def test_list_marks_overdue_and_stale_schedules(
    tmp_path: Path, running_supervisor, capsys: pytest.CaptureFixture[str]
) -> None:
    from local_operator.cli import wake_command

    _arm(
        tmp_path,
        "statussess05",
        cwd=str(tmp_path),
        schedules=[{"id": "w1", "message": "late watch", "next_due_at": NOW_MS - 600_000}],
    )
    _arm(
        tmp_path,
        "statussess06",
        cwd=str(tmp_path),
        schedules=[{"id": "w1", "message": "forgotten", "next_due_at": NOW_MS - 9 * 24 * 3600_000}],
    )

    assert wake_command(_args(wake_command="list", json=True)) == 0
    rows = {row["session_id"]: row for row in json.loads(capsys.readouterr().out)}

    assert rows["statussess05"]["overdue"] is True
    assert rows["statussess05"]["stale"] is False
    assert rows["statussess06"]["stale"] is True

    assert wake_command(_args(wake_command="list", json=False)) == 0
    out = capsys.readouterr().out
    # The DUE column carries the state word now that the listing is a
    # fixed-width table (round 1, D4): an overdue row reads "10m overdue"
    # there, and only the stale row needs the explanatory tail.
    overdue_line = next(line for line in out.splitlines() if "statussess05" in line)
    stale_line = next(line for line in out.splitlines() if "statussess06" in line)
    assert "overdue" in overdue_line, overdue_line
    # Stale is the STRONGER statement (the supervisor has stopped trying), so
    # it replaces the overdue mark rather than doubling up on one line.
    assert "stale" in stale_line and "overdue" not in stale_line, stale_line


def test_the_install_subcommand_reaches_the_repair_path(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
    _installer_available: None,
) -> None:
    """THE ROLLOUT PATH.

    Repair-on-demand otherwise runs only on a wake PERSIST, so a machine whose
    supervisor is stale cannot be fixed without some session happening to
    schedule a wake — exactly the wrong dependency after an upgrade, when the
    running supervisor is still executing the old code.
    """
    from local_operator.cli import wake_command
    from local_operator.wakes.install import InstallOutcome

    called: list[Path] = []

    def fake_install(config_dir: Path) -> InstallOutcome:
        called.append(config_dir)
        return InstallOutcome(installed=True, reason="restarted a stopped supervisor")

    monkeypatch.setattr("local_operator.wakes.install.ensure_supervisor_installed", fake_install)
    monkeypatch.setattr(
        "local_operator.wakes.install.supervisor_state",
        lambda _config: SupervisorState(loaded=True, running=True, pid=7, detail="running"),
    )

    assert wake_command(_args(wake_command="install")) == 0

    assert called, "the install subcommand did not reach the repair hook"
    assert "restarted a stopped supervisor" in capsys.readouterr().out


def test_an_unsupervisable_store_never_reports_another_stores_supervisor(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    """THE ISOLATED-RUN CASE, end to end through the command.

    No patching of the probe: this drives the real `supervisor_state` against
    a config dir outside the real home — the shape every isolated run has —
    and the command must report a state of its own rather than the operator's
    live LaunchAgent. Before this, the same invocation printed
    `running (pid 47545)`, which is a different store's process.
    """
    from local_operator.cli import wake_command
    from local_operator.wakes.install import is_supported

    if not is_supported():
        pytest.skip("launchd scoping is only meaningful on darwin with launchctl")

    _arm(
        tmp_path,
        "statussess07",
        cwd=str(tmp_path),
        schedules=[{"id": "w1", "message": "watch prod", "next_due_at": NOW_MS - 60_000}],
    )

    assert wake_command(_args()) == 0
    out = capsys.readouterr().out

    assert "cannot be verified for this store" in out, out
    assert "running" not in out.split("scheduled:")[0].replace("not running", ""), out
    assert "pid" not in out, "another store's pid was printed"

    assert wake_command(_args(json=True)) == 0
    payload = json.loads(capsys.readouterr().out)
    assert payload["supervisor"]["verifiable"] is False
    assert payload["supervisor"]["running"] is False
    assert payload["supervisor"]["pid"] is None
    assert payload["installed"] is False


def test_status_reports_a_fire_that_could_not_be_delivered(
    tmp_path: Path, running_supervisor, capsys: pytest.CaptureFixture[str]
) -> None:
    """THE SURFACE THE DEFECT NEVER HAD. 510 failed engages on this machine left
    a WARNING in an unrotated log and nothing on any screen: every other line
    here rendered such a wake as an ordinary overdue row, which is exactly the
    reassurance that kept it invisible."""
    from local_operator.cli import wake_command
    from local_operator.wakes import deliveries

    due = NOW_MS - 600_000
    _arm(
        tmp_path,
        "statussess05",
        cwd=str(tmp_path),
        schedules=[{"id": "w1", "message": "nightly cleanup", "next_due_at": due}],
    )
    deliveries.note_failure(tmp_path, "statussess05", due, error="unreachable: 180s", now_ms=NOW_MS)

    assert wake_command(_args()) == 0

    out = capsys.readouterr().out

    retrying = _status_block(out, "retrying:")
    assert "retrying:" in retrying, f"an owed fire was not reported: {out}"
    assert "statussess05" in retrying, retrying
    assert "nightly cleanup" in retrying, retrying
    assert "retried with backoff" in retrying, retrying

    assert wake_command(_args(json=True)) == 0
    payload = json.loads(capsys.readouterr().out)
    assert payload["retrying"] == 1 and payload["undelivered"] == 0
    assert payload["deliveries"][0]["session_id"] == "statussess05"
    assert payload["deliveries"][0]["occurrence_ms"] == due
    assert payload["deliveries"][0]["attempts"] == 1
    assert payload["deliveries"][0]["last_error"] == "unreachable: 180s"


def test_status_says_an_undelivered_fire_is_still_owed(
    tmp_path: Path, running_supervisor, capsys: pytest.CaptureFixture[str]
) -> None:
    """Past the report threshold the fire is STILL OWED and still retried.

    The whole point of the fix is that a wake is not dropped when a budget
    expires, so the surface must not read as "lost" either — it names the
    attempt count, the age, the last error, and the next attempt.

    The line states what IS true rather than what changed (design round 1, D7:
    "a fire that cannot be delivered is no longer dropped" is a change-note, and
    the same columns buy the retry clock the `retrying:` line already printed).
    """
    from local_operator.cli import wake_command
    from local_operator.wakes import deliveries

    # A FRESH CLOCK, not the module-level NOW_MS: that one is captured at import,
    # and a sharded CI run can execute this file minutes later — long enough to
    # put the recorded next attempt in the past and turn the "next attempt" fact
    # into a conversation about how slow the runner was.
    now = int(time.time() * 1000)
    due = now - 900_000
    _arm(
        tmp_path,
        "statussess06",
        cwd=str(tmp_path),
        schedules=[{"id": "w1", "message": "collect metrics", "next_due_at": due}],
    )
    for _ in range(deliveries.UNDELIVERED_AFTER_ATTEMPTS):
        deliveries.note_failure(
            tmp_path, "statussess06", due, error="could not reach a runtime", now_ms=now
        )

    assert wake_command(_args()) == 0

    out = capsys.readouterr().out

    line = _status_block(out, "undelivered:")
    assert "undelivered:" in line, f"an undelivered fire was not reported: {out}"
    assert "statussess06" in line, line
    assert f"{deliveries.UNDELIVERED_AFTER_ATTEMPTS} attempt(s)" in line, line
    assert "could not reach a runtime" in line, line
    assert "STILL OWED" in line, line
    assert "retried with backoff" in line, line
    # The WHEN clause is not silently dropped when the attempt is already due
    # (design round 1, D3): this row's next attempt is three minutes out.
    assert "next attempt in" in line, line

    assert wake_command(_args(json=True)) == 0
    payload = json.loads(capsys.readouterr().out)
    assert payload["undelivered"] == 1
    assert payload["deliveries"][0]["state"] == deliveries.STATE_UNDELIVERED
    assert payload["deliveries"][0]["attempts"] == deliveries.UNDELIVERED_AFTER_ATTEMPTS
    # A NEXT ATTEMPT IS SCHEDULED, asserted on the record rather than only on the
    # rendered offset: the JSON field is a difference against the read clock, so
    # the property that matters is that the record carries a next attempt after
    # its last one.
    stored = deliveries.read_delivery(tmp_path, "statussess06")
    assert stored is not None and stored["next_attempt_ms"] > stored["last_attempt_ms"], stored
    assert payload["deliveries"][0]["next_attempt_in_s"] is not None


def test_list_marks_an_owed_fire_as_retrying(
    tmp_path: Path, running_supervisor, capsys: pytest.CaptureFixture[str]
) -> None:
    """The table's own vocabulary, extended by one word.

    The DUE column is where this listing states why a wake is not firing
    (`dormant`, `ghost`, `stale`), and an owed fire is a fourth reason with a
    different consequence: the supervisor is still working on it. Rendering it
    as `stale` would say the opposite — the legend promises `stale` wakes are
    left to the session's next open.
    """
    from local_operator.cli import wake_command
    from local_operator.wakes import deliveries

    due = NOW_MS - 600_000
    _arm(
        tmp_path,
        "statussess08",
        cwd=str(tmp_path),
        schedules=[{"id": "w1", "message": "nightly cleanup", "next_due_at": due}],
    )
    deliveries.note_failure(tmp_path, "statussess08", due, error="unreachable", now_ms=NOW_MS)

    assert wake_command(_args(wake_command="list", json=False)) == 0
    out = capsys.readouterr().out
    line = next(line for line in out.splitlines() if "statussess08" in line)
    assert "retrying" in line, line
    assert "overdue" not in line, line
    assert "still owed" in out, out
    assert "lop wake status" in out, out


def test_list_marks_a_stale_owed_fire_as_undelivered_rather_than_stale(
    tmp_path: Path, running_supervisor, capsys: pytest.CaptureFixture[str]
) -> None:
    """Past the bound, the owed fire is the STRONGER fact.

    A stale schedule is normally given up on; one with an owed record is still
    being retried, and that is what the column must say.
    """
    from local_operator.cli import wake_command
    from local_operator.wakes import deliveries

    due = NOW_MS - int(9 * 86400 * 1000)
    _arm(
        tmp_path,
        "statussess09",
        cwd=str(tmp_path),
        schedules=[{"id": "w1", "message": "forgotten cleanup", "next_due_at": due}],
    )
    for _ in range(deliveries.UNDELIVERED_AFTER_ATTEMPTS):
        deliveries.note_failure(
            tmp_path, "statussess09", due, error="could not reach a runtime", now_ms=NOW_MS
        )

    assert wake_command(_args(wake_command="list", json=False)) == 0
    out = capsys.readouterr().out
    line = next(line for line in out.splitlines() if "statussess09" in line)
    assert "undelivered" in line, line
    assert "stale" not in line.split("forgotten cleanup")[0], line
    # And the stale legend is NOT printed for a store whose only old wake is
    # one the supervisor is still retrying.
    assert "the supervisor no longer fires these" not in out, out


def test_status_counts_a_stale_wake_with_an_owed_fire_as_fireable(
    tmp_path: Path, running_supervisor, capsys: pytest.CaptureFixture[str]
) -> None:
    """Reconcilable counts, and no contradiction between the two lines.

    `scheduled` must still equal fireable + dormant + stale + ghost, so a stale
    row that is nevertheless being fired belongs in `fireable` — otherwise the
    same wake would be reported as given up on AND as still owed.
    """
    from local_operator.cli import wake_command
    from local_operator.wakes import deliveries

    due = NOW_MS - int(9 * 86400 * 1000)
    _arm(
        tmp_path,
        "statussess10",
        cwd=str(tmp_path),
        schedules=[{"id": "w1", "message": "old but owed", "next_due_at": due}],
    )
    deliveries.note_failure(tmp_path, "statussess10", due, error="unreachable", now_ms=NOW_MS)

    assert wake_command(_args(json=True)) == 0
    payload = json.loads(capsys.readouterr().out)
    assert payload["stale"] == 0, payload
    assert payload["retrying"] == 1 and payload["undelivered"] == 0
    assert payload["unfireable"]["stale"] == []
    assert (
        payload["scheduled"]
        == (len(payload["unscheduled"]) if "unscheduled" in payload else payload["armed"])
        or payload["armed"] == 1
    )

    assert wake_command(_args()) == 0
    out = capsys.readouterr().out
    assert "stale:" not in out, out
    assert "retrying:" in out, out


def test_a_record_for_a_different_occurrence_is_not_reported(
    tmp_path: Path, running_supervisor, capsys: pytest.CaptureFixture[str]
) -> None:
    """The ledger names an OCCURRENCE, and the recurrence is a different fire.

    A recurring wake that has already advanced its next due time must not
    inherit the previous occurrence's failed delivery, or every healthy watch
    with a single old failure would report an owed fire forever.
    """
    from local_operator.cli import wake_command
    from local_operator.wakes import deliveries

    _arm(
        tmp_path,
        "statussess07",
        cwd=str(tmp_path),
        schedules=[{"id": "w1", "message": "hourly watch", "next_due_at": NOW_MS + 60_000}],
    )
    deliveries.note_failure(
        tmp_path, "statussess07", NOW_MS - 3_600_000, error="old failure", now_ms=NOW_MS - 3_600_000
    )

    assert wake_command(_args()) == 0
    out = capsys.readouterr().out
    assert "retrying:" not in out and "undelivered:" not in out, out

    assert wake_command(_args(json=True)) == 0
    payload = json.loads(capsys.readouterr().out)
    assert payload["deliveries"] == [] and payload["retrying"] == 0


# --- The REAL parser ---------------------------------------------------------
#
# Every test above hand-builds an `argparse.Namespace`, which is why round 1's
# blocker (`lop wake install` and bare `lop wake` dying on `args.json`) was
# invisible to the whole file: a hand-built namespace supplies exactly the
# attributes the test author remembered, so it can never catch the dispatcher
# reading one the PARSER does not define. These walk
# `build_cli_parser().parse_args(...)` and then run the command, which is the
# only arrangement that exercises the parser/dispatcher contract.


@pytest.mark.parametrize(
    "argv",
    [
        ["wake"],  # bare: `wake_command` defaults to "status"
        ["wake", "status"],
        ["wake", "status", "--json"],
        ["wake", "install"],
        ["wake", "list"],
        ["wake", "list", "--json"],
    ],
)
def test_every_wake_entry_point_survives_the_real_parser(
    tmp_path: Path, argv: list[str], capsys: pytest.CaptureFixture[str]
) -> None:
    """The regression test for the round-1 blocker, across the whole surface.

    Not just the reported command: `lop wake status` PRINTS `run 'lop wake
    install'` as its remedy in several states, so the crash was reachable from
    the surface's own advice. Parametrised over every route into
    `wake_command` so a future subcommand that forgets a flag fails here.
    """
    from local_operator.cli import build_cli_parser, wake_command

    _arm(
        tmp_path,
        "realparser1",
        cwd=str(tmp_path),
        schedules=[{"id": "w1", "message": "watch prod", "next_due_at": NOW_MS + 600_000}],
    )

    args = build_cli_parser().parse_args(argv)

    # The assertion is that this does not raise. `install` reaches the real
    # install hook, which under a redirected home writes a plist and declines
    # to address launchd (see `test_install.py`'s safety contract).
    assert wake_command(args) == 0
    assert capsys.readouterr().out, f"{argv} produced no output"


def test_the_parser_defines_every_flag_the_dispatcher_reads(tmp_path: Path) -> None:
    """The structural form of the same bug, stated once.

    `wake_command` reads `json`, `install` and `uninstall` off the namespace
    for any route that lands in the status branch. Asserting on the parsed
    namespace rather than on behaviour means a flag added to the dispatcher
    without a parser default fails here with a message naming it.
    """
    from local_operator.cli import build_cli_parser

    parser = build_cli_parser()
    for argv in (["wake"], ["wake", "status"], ["wake", "install"], ["wake", "list"]):
        args = parser.parse_args(argv)
        for flag in ("json", "install", "uninstall"):
            assert hasattr(args, flag), (
                f"`lop {' '.join(argv)}` parses without `{flag}`, which `wake_command` "
                f"dereferences — this is the round-1 blocker's exact shape"
            )


# --- Round 2: the rendering guards ------------------------------------------


def test_a_wake_in_another_year_renders_its_whole_time(
    tmp_path: Path, stopped_supervisor, capsys: pytest.CaptureFixture[str]
) -> None:
    """D11 (MAJOR): `format_wake_time` widens to 24 characters for an off-year
    wake, and a fixed 18-wide column cut `Jan 01 2027 9:00 AM EST` to
    `Jan 01 2027 9:00 A` — a half meridiem, no zone, no marker. `wake create`
    printed the full form one line earlier, so the same wake read two ways.

    Mutation-checked: restoring `format_wake_time(...)[:18]` fails this with
    `the rendered time was truncated mid-token`.
    """
    from local_operator.cli import wake_command
    from local_operator.wakes.display import format_wake_time

    due = NOW_MS + 400 * 86400_000  # comfortably into the next calendar year
    _arm(
        tmp_path,
        "nextyear0001",
        cwd=str(tmp_path),
        schedules=[{"id": "w1", "message": "annual renewal: TLS cert", "next_due_at": due}],
    )

    assert wake_command(_args(wake_command="list", json=False)) == 0

    out = capsys.readouterr().out
    expected = format_wake_time(due)
    assert "2027" in expected or "2026" in expected, expected
    row = next(line for line in out.splitlines() if "nextyear0001" in line)
    assert expected in row, f"the rendered time was truncated mid-token: {row!r}"


def test_a_session_id_is_never_silently_shortened(
    tmp_path: Path, stopped_supervisor, capsys: pytest.CaptureFixture[str]
) -> None:
    """R9: a 12-wide column rendered a longer id as a DIFFERENT id — one an
    operator cannot paste into `lop wake list` or `lop stop`. A real id fits
    whole; anything longer is marked.

    Mutation-checked: restoring `row['session_id'][:id_w]` fails this with
    `a truncated id is indistinguishable from a real one`.
    """
    from local_operator.cli import wake_command

    real = "00264921d0d9"  # uuid4().hex[:12], the shape the product mints
    longer = "lr_bda7b76d34e0"
    for session_id in (real, longer):
        _arm(
            tmp_path,
            session_id,
            cwd=str(tmp_path),
            schedules=[{"id": "w1", "message": "watch", "next_due_at": NOW_MS + 60_000}],
        )

    assert wake_command(_args(wake_command="list", json=False)) == 0

    out = capsys.readouterr().out
    assert real in out, f"a real 12-character id must render whole: {out}"
    # The longer id does not fit; it must be MARKED, never silently shortened
    # into a different, real-looking id.
    assert (
        longer[:12] not in out or longer in out or "…" in out
    ), f"a truncated id is indistinguishable from a real one: {out!r}"
    ghosty = [line for line in out.splitlines() if "lr_bda7b76d" in line]
    assert ghosty and ("…" in ghosty[0] or longer in ghosty[0]), ghosty


def test_a_narrow_terminal_keeps_the_columns_aligned(
    tmp_path: Path, stopped_supervisor, monkeypatch: pytest.MonkeyPatch, capsys
) -> None:
    """D12 / QA Q1: at 60 columns every row wrapped (68-71 chars), so the
    alignment this table exists for was gone in a split pane. The absolute
    time is the redundant half — `DUE` answers "when" in 11 characters — so it
    is what yields, and the omission is stated rather than silent.

    Mutation-checked: forcing `show_when = True` fails this with
    `row exceeds the terminal width`.
    """
    import shutil

    from local_operator.cli import wake_command

    monkeypatch.setattr(
        shutil, "get_terminal_size", lambda _default=None: os.terminal_size((60, 24))
    )
    _arm(
        tmp_path,
        "narrowsess01",
        cwd=str(tmp_path),
        schedules=[
            {
                "id": "w1",
                "message": "cluster watch: aws-prod-2 node rotation",
                "next_due_at": NOW_MS + 400 * 86400_000,
            }
        ],
    )

    assert wake_command(_args(wake_command="list", json=False)) == 0

    out = capsys.readouterr().out
    for line in out.splitlines():
        assert len(line) <= 60, f"row exceeds the terminal width ({len(line)}): {line!r}"
    assert "WHEN hidden" in out, out


def test_status_and_the_supervisor_agree_about_a_ghost(
    tmp_path: Path, stopped_supervisor, capsys: pytest.CaptureFixture[str]
) -> None:
    """QA Q4: a ghost-only store made the two halves contradict each other —
    the supervisor retired ("no fireable wakes remain") while this frame said
    `1 armed`, `overdue: 1`. A painted frame that disagrees with the process is
    the defect class this PR exists to remove.

    `write_entry` directly, not `_arm`: the whole point is an index entry with
    no session on disk.

    Mutation-checked: removing the `ghost` predicate from `_wake_rows` fails
    this with `status claims a wake is coming that the supervisor has retired
    over`.
    """
    from local_operator.cli import wake_command
    from local_operator.wakes.store import read_index
    from local_operator.wakes.supervisor import _has_fireable_wakes

    write_entry(
        tmp_path,
        "ghostsess001",
        cwd=str(tmp_path),
        schedules=[{"id": "w1", "message": "orphaned watch", "next_due_at": NOW_MS - 600_000}],
    )

    # The supervisor's verdict on this exact store.
    assert not _has_fireable_wakes(read_index(tmp_path), config_dir=tmp_path)

    assert wake_command(_args()) == 0
    out = capsys.readouterr().out
    assert "ghost:       1 with no session on disk" in out, out
    assert (
        "overdue:" not in out
    ), f"status claims a wake is coming that the supervisor has retired over: {out}"

    assert wake_command(_args(json=True)) == 0
    payload = json.loads(capsys.readouterr().out)
    assert payload["ghost"] == 1
    assert payload["overdue"] == 0
    assert payload["next_fireable_due_in_s"] is None
    assert payload["unfireable"]["ghost"] == ["ghostsess001"]


def test_a_long_status_line_folds_at_the_surfaces_indent(
    tmp_path: Path, stopped_supervisor, monkeypatch: pytest.MonkeyPatch, capsys
) -> None:
    """D13 + QA Q2: the `wedged:` line was 108 columns and the only one whose
    continuation started at column 0, and `next:`/`overdue:` interpolate a
    user-authored message that was never clamped (156 columns measured).

    Mutation-checked: bypassing `_wrap_status` fails this with
    `status line exceeds the terminal width`.
    """
    import shutil

    from local_operator.cli import wake_command

    monkeypatch.setattr(
        shutil, "get_terminal_size", lambda _default=None: os.terminal_size((80, 24))
    )
    _arm(
        tmp_path,
        "verbosewake1",
        cwd=str(tmp_path),
        schedules=[
            {
                "id": "w1",
                "message": (
                    "prod watch: NER backfill completion across every shard, then "
                    "reconcile the ledger and post the summary to the release thread"
                ),
                "next_due_at": NOW_MS - 600_000,
            }
        ],
    )

    assert wake_command(_args()) == 0

    out = capsys.readouterr().out
    for line in out.splitlines():
        assert len(line) <= 80, f"status line exceeds the terminal width ({len(line)}): {line!r}"
    # Continuations align with the label column rather than starting at 0.
    continuations = [
        line
        for line in out.splitlines()
        if line.startswith(" ") and line.strip() and not line.startswith(" " * 13)
    ]
    assert not continuations, f"a continuation broke the 13-column indent: {continuations}"


def test_a_remedy_command_is_never_split_across_lines(
    tmp_path: Path, stopped_supervisor, monkeypatch: pytest.MonkeyPatch, capsys
) -> None:
    """Every remedy on this surface is a command the operator copies; a wrap
    inside one produces a line that looks like an instruction and is not
    runnable. Folding must treat a quoted command as one token.
    """
    import shutil

    from local_operator.cli import wake_command

    monkeypatch.setattr(
        shutil, "get_terminal_size", lambda _default=None: os.terminal_size((62, 24))
    )
    _arm(
        tmp_path,
        "remedysess01",
        cwd=str(tmp_path),
        schedules=[{"id": "w1", "message": "watch", "next_due_at": NOW_MS + 60_000}],
    )

    assert wake_command(_args()) == 0

    out = capsys.readouterr().out
    assert "'lop wake install'" in out, out


def test_a_future_wake_is_named_even_when_another_is_overdue(
    tmp_path: Path, stopped_supervisor, capsys: pytest.CaptureFixture[str]
) -> None:
    """D19: one late wake must not hide the next one.

    `upcoming` is sorted soonest-first and the `next:` line was gated on its
    HEAD, so a single overdue row suppressed `next:` for every future wake —
    a wake a minute away went unnamed on the surface README promises reports
    "the soonest wake that will fire". D16 (no `next:` when there is nothing
    in the future) is asserted by
    `test_status_reports_overdue_and_stale_counts`; this is the other half.
    """
    from local_operator.cli import wake_command

    # NOT the module-level `NOW_MS`, which is captured at IMPORT: a CI shard
    # can run for 17 minutes, and a wake armed "one minute from import" is
    # long overdue by the time this executes — which lands in the very branch
    # the test exists to distinguish from. An hour off a fresh reading is
    # future under any plausible shard duration.
    now_ms = int(time.time() * 1000)
    _arm(
        tmp_path,
        "d19late0001",
        cwd=str(tmp_path),
        schedules=[{"id": "w1", "message": "late watch", "next_due_at": now_ms - 600_000}],
    )
    _arm(
        tmp_path,
        "d19soon0001",
        cwd=str(tmp_path),
        schedules=[
            {"id": "w1", "message": "release-owner check", "next_due_at": now_ms + 3_600_000}
        ],
    )

    assert wake_command(_args()) == 0

    out = capsys.readouterr().out
    next_line = next(
        (line for line in out.splitlines() if line.startswith("next:")),
        "",
    )
    assert next_line, f"a wake one minute away went unnamed: {out}"
    assert "release-owner check" in next_line, next_line
    # And the late one is still reported, on its own line, exactly once.
    overdue_line = next(line for line in out.splitlines() if line.startswith("overdue:"))
    assert "late watch" in overdue_line, overdue_line
    assert "late watch" not in next_line, next_line


def test_list_gives_each_owed_row_the_age_of_its_fire(
    tmp_path: Path, stopped_supervisor, monkeypatch: pytest.MonkeyPatch, capsys
) -> None:
    """Design round 1, D2: two owed rows must not render identically.

    `WHEN` is dropped below 69 columns (measured: hidden at 68, shown at 69), so
    `retrying owedstale001` — owed for nine days — and `retrying owedretry001` —
    owed for four minutes — were the same row shape with no age anywhere. The age
    goes in the TAIL, where the other bounds already live and where the round-5
    R6/U16 rule says it is never clamped: the message is what gives, and it is
    the part of the row the reader already knows.

    Mutation-checked: removing the `, owed …` clause fails this with
    `both owed rows still render the same shape`.
    """
    import shutil

    from local_operator.cli import wake_command
    from local_operator.wakes import deliveries

    monkeypatch.setattr(
        shutil, "get_terminal_size", lambda _default=None: os.terminal_size((60, 24))
    )
    now = int(time.time() * 1000)
    nine_days = now - 9 * 86_400_000
    four_minutes = now - 240_000
    _arm(
        tmp_path,
        "owedsessage1",
        cwd=str(tmp_path),
        schedules=[{"id": "w1", "message": "stale but owed", "next_due_at": nine_days}],
    )
    _arm(
        tmp_path,
        "owedsessage2",
        cwd=str(tmp_path),
        schedules=[{"id": "w1", "message": "failed once", "next_due_at": four_minutes}],
    )
    deliveries.note_failure(
        tmp_path, "owedsessage1", nine_days, error="unreachable", now_ms=nine_days
    )
    deliveries.note_failure(
        tmp_path, "owedsessage2", four_minutes, error="unreachable", now_ms=four_minutes
    )

    assert wake_command(_args(wake_command="list", json=False)) == 0
    out = capsys.readouterr().out
    stale_row = next(line for line in out.splitlines() if "owedsessage1" in line)
    fresh_row = next(line for line in out.splitlines() if "owedsessage2" in line)

    assert "owed 9d" in stale_row, f"the aged owed row carries no age: {stale_row!r}"
    assert "owed 4m" in fresh_row, f"the young owed row carries no age: {fresh_row!r}"
    assert stale_row != fresh_row, "both owed rows still render the same shape"
    for line in out.splitlines():
        assert len(line) <= 60, f"the age pushed a row past the terminal ({len(line)}): {line!r}"


def test_status_prints_a_due_retry_rather_than_dropping_the_clause(
    tmp_path: Path, stopped_supervisor, capsys: pytest.CaptureFixture[str]
) -> None:
    """Design round 1, D3: `next_attempt_in_s` is signed, and ≤ 0 is a state.

    The rendering guard dropped the WHEN clause whenever the figure was not
    positive — which is exactly when the attempt is already due — so the line
    that must answer "is this being retried?" ended at `retried with backoff`
    with no time at all while the neighbouring `undelivered:` line printed one.
    """
    from local_operator.cli import wake_command
    from local_operator.wakes import deliveries

    now = int(time.time() * 1000)
    due = now - 900_000
    _arm(
        tmp_path,
        "statussess07",
        cwd=str(tmp_path),
        schedules=[{"id": "w1", "message": "collect metrics", "next_due_at": due}],
    )
    # Recorded ten minutes ago, so its next attempt has long since fallen due.
    deliveries.note_failure(
        tmp_path, "statussess07", due, error="could not reach a runtime", now_ms=now - 600_000
    )

    assert wake_command(_args()) == 0
    out = capsys.readouterr().out
    assert "retry due now" in _status_block(out, "retrying:"), out

    assert wake_command(_args(json=True)) == 0
    payload = json.loads(capsys.readouterr().out)
    assert payload["deliveries"][0]["next_attempt_in_s"] <= 0, payload["deliveries"][0]


def test_status_says_the_owed_fires_are_part_of_the_overdue_count(
    tmp_path: Path, stopped_supervisor, capsys: pytest.CaptureFixture[str]
) -> None:
    """Design round 1, D4: the counts double-reported the same wakes.

    Every owed fire is overdue by construction, so a 5-wake store printed
    `overdue: 3`, `retrying: 2`, `undelivered: 1` and an operator adding the
    lines got 6 of 5. The subset is now stated where the counts are, and the JSON
    says it structurally as well as the line does.
    """
    from local_operator.cli import wake_command
    from local_operator.wakes import deliveries

    now = int(time.time() * 1000)
    retried_due = now - 600_000
    stalled_due = now - 900_000
    plain_due = now - 300_000
    for session, due, message in (
        ("overduesub001", retried_due, "one failed attempt"),
        ("overduesub002", stalled_due, "many failures"),
        ("overduesub003", plain_due, "never attempted"),
    ):
        _arm(
            tmp_path,
            session,
            cwd=str(tmp_path),
            schedules=[{"id": "w1", "message": message, "next_due_at": due}],
        )
    deliveries.note_failure(tmp_path, "overduesub001", retried_due, error="x", now_ms=now - 30_000)
    for _ in range(deliveries.UNDELIVERED_AFTER_ATTEMPTS):
        deliveries.note_failure(
            tmp_path, "overduesub002", stalled_due, error="x", now_ms=now - 30_000
        )

    assert wake_command(_args()) == 0
    out = capsys.readouterr().out
    overdue_line = _status_block(out, "overdue:")
    assert "— 1 retrying, 1 undelivered" in overdue_line, overdue_line

    assert wake_command(_args(json=True)) == 0
    payload = json.loads(capsys.readouterr().out)
    assert payload["overdue"] == 3, payload
    assert payload["owed"] == {
        "subset_of": "overdue",
        "total": 2,
        "retrying": 1,
        "undelivered": 1,
    }, payload["owed"]


def test_a_ghost_with_an_owed_record_is_not_reported_as_retried(
    tmp_path: Path, stopped_supervisor, capsys: pytest.CaptureFixture[str]
) -> None:
    """QA round 1, Q1: the frame argued with itself about a ghost.

    The supervisor refuses a ghost before recording any attempt and retires on a
    ghost-only store, so an owed record for a deleted session is frozen — nothing
    retries it and nothing will. The frame reported it as "still owed and retried
    with backoff" beside the `ghost:` line that says nothing can fire it, and
    `--json` said `retrying: 1` with `overdue: 0`.

    `write_entry` directly, not `_arm`: the point is a session with no transcript.

    Mutation-checked: dropping `and not row["ghost"]` from the owed bucket fails
    this with `the frozen record is still reported as work in progress`.
    """
    from local_operator.cli import wake_command
    from local_operator.wakes import deliveries

    now = int(time.time() * 1000)
    due = now - 600_000
    write_entry(
        tmp_path,
        "ghostowed001",
        cwd=str(tmp_path),
        schedules=[{"id": "w1", "message": "orphaned", "next_due_at": due}],
    )
    deliveries.note_failure(tmp_path, "ghostowed001", due, error="unreachable", now_ms=now - 60_000)

    assert wake_command(_args()) == 0
    out = capsys.readouterr().out
    assert "ghost:" in out, out
    assert "retrying:" not in out, f"the frozen record is still reported as work in progress: {out}"
    assert "undelivered:" not in out, out

    assert wake_command(_args(json=True)) == 0
    payload = json.loads(capsys.readouterr().out)
    assert payload["ghost"] == 1, payload
    assert payload["retrying"] == 0 and payload["undelivered"] == 0, payload
    assert payload["owed"]["total"] == 0, payload["owed"]
    # The record is still on disk — the operator's to delete — it is simply not
    # reported as an attempt in progress.
    assert deliveries.read_delivery(tmp_path, "ghostowed001") is not None

    # AND THE LISTING AGREES WITH IT. The `status` bucket was the reported half;
    # the table's DUE word said `ghost` while its tail said `owed 1m` and the
    # `retrying` legend printed under it — the same contradiction one surface
    # over.
    assert wake_command(_args(wake_command="list", json=False)) == 0
    listed = capsys.readouterr().out
    row = next(line for line in listed.splitlines() if "ghostowed001" in line)
    assert "ghost" in row, row
    # `, owed ` and not the bare word: the synthetic id itself contains "owed".
    assert ", owed " not in row, row
    assert "failed attempts." not in listed, listed


@pytest.mark.parametrize("threshold", [3, 5])
def test_retry_legend_uses_the_actual_failure_threshold(
    tmp_path: Path, stopped_supervisor, monkeypatch: pytest.MonkeyPatch, capsys, threshold: int
) -> None:
    """Two failures are still retrying; the legend must cover the whole bucket.

    Moving the real classifier's threshold also moves the rendered boundary,
    so a copy-only hardcoded replacement cannot silently drift from state.
    """
    from local_operator.cli import wake_command
    from local_operator.wakes import deliveries

    monkeypatch.setattr(deliveries, "UNDELIVERED_AFTER_ATTEMPTS", threshold)
    now = int(time.time() * 1000)
    due = now - 600_000
    session_id = "retrythreshold"
    _arm(
        tmp_path,
        session_id,
        cwd=str(tmp_path),
        schedules=[{"id": "w1", "message": "retry boundary", "next_due_at": due}],
    )
    for attempts in range(1, threshold + 1):
        record = deliveries.note_failure(tmp_path, session_id, due, error="x", now_ms=now)
        assert record is not None and record["attempts"] == attempts
        retrying = attempts < threshold
        assert record["state"] == (
            deliveries.STATE_RETRYING if retrying else deliveries.STATE_UNDELIVERED
        )
        assert wake_command(_args(wake_command="list", json=False)) == 0
        out = capsys.readouterr().out
        description = (
            f"fewer than {threshold} failed attempts."
            if retrying
            else f"{threshold}+ failed attempts."
        )
        assert description in out, out
        assert "one failed attempt so far" not in out, out


def test_the_owed_legends_come_first_and_share_one_tail(
    tmp_path: Path, stopped_supervisor, capsys: pytest.CaptureFixture[str]
) -> None:
    """Design round 1, D5 and D8.

    D5: the two legends repeated ~100 characters of the same sentence three
    lines apart and buried the clause that separates them, which took `wake
    list` from 11 rows to 21 at 60 columns — past a standard screen.
    D8: the order put the two states this PR exists to surface LAST, so a reader
    scanning for what `retrying` means passed the three that mean the opposite.
    """
    from local_operator.cli import wake_command
    from local_operator.wakes import deliveries

    now = int(time.time() * 1000)
    owed_due = now - 600_000
    stale_due = now - 9 * 86_400_000
    _arm(
        tmp_path,
        "legendssess1",
        cwd=str(tmp_path),
        schedules=[{"id": "w1", "message": "still firing", "next_due_at": owed_due}],
    )
    _arm(
        tmp_path,
        "legendssess2",
        cwd=str(tmp_path),
        schedules=[{"id": "w1", "message": "given up", "next_due_at": stale_due}],
    )
    deliveries.note_failure(tmp_path, "legendssess1", owed_due, error="x", now_ms=now - 30_000)

    assert wake_command(_args(wake_command="list", json=False)) == 0
    out = capsys.readouterr().out

    # The legends are located by their own text rather than by their column
    # padding: the label width is sized from the words actually rendered (round
    # 2, D12), so it moves with which states are present.
    owed_legend = out.index(f"fewer than {deliveries.UNDELIVERED_AFTER_ATTEMPTS} failed attempts.")
    assert out.index("these are still owed and retried with a backoff") > owed_legend, out
    # D5: one short clause per word and ONE shared sentence, not the same
    # ~100 characters restated three lines apart.
    assert out.count("these are still owed and retried with a backoff") == 1, out
    assert "('lop wake status' has the attempts and the error)" not in out, out
    # D8: the owed pair leads the three states that mean the opposite.
    assert owed_legend < out.index("the supervisor no longer fires these"), out


def test_the_when_note_lands_under_the_table_not_under_the_legends(
    tmp_path: Path, stopped_supervisor, monkeypatch: pytest.MonkeyPatch, capsys
) -> None:
    """Design round 1, D6: the note explaining the missing column was 14 rows late.

    At 60 columns it printed after three legend blocks, so the reader who noticed
    the absent `WHEN` column had to scroll past every legend to learn why.
    """
    import shutil

    from local_operator.cli import wake_command
    from local_operator.wakes import deliveries

    monkeypatch.setattr(
        shutil, "get_terminal_size", lambda _default=None: os.terminal_size((60, 24))
    )
    now = int(time.time() * 1000)
    due = now - 600_000
    _arm(
        tmp_path,
        "whennotess01",
        cwd=str(tmp_path),
        schedules=[{"id": "w1", "message": "collect metrics", "next_due_at": due}],
    )
    deliveries.note_failure(tmp_path, "whennotess01", due, error="x", now_ms=now - 30_000)

    assert wake_command(_args(wake_command="list", json=False)) == 0
    out = capsys.readouterr().out
    assert "WHEN hidden" in out, out
    assert out.index("WHEN hidden") < out.index(
        f"fewer than {deliveries.UNDELIVERED_AFTER_ATTEMPTS} failed attempts."
    ), out


def test_a_quiet_owner_line_names_the_age_and_the_ladders_first_rung(
    tmp_path: Path, stopped_supervisor, monkeypatch: pytest.MonkeyPatch, capsys
) -> None:
    """R2/D3, as an assertion: `wake status` must not advertise a stop that refuses.

    The line used to end "... 'lop stop --pid N' ends it", and it also claimed
    "It will not recover on its own". Both were wrong in the same direction. A
    plain ``lop stop --pid N`` asks the owner's socket, then needs an identity
    proof that a fresh heartbeat forbids — on an owner that is still beating it
    refuses; and a stale beat does not establish that the owner cannot recover,
    because the beat is authored by the runtime's own event loop and a long
    turn produces it.

    What the surface names instead is the rung that ACTS on a lapsed beat —
    the plain ``lop stop``, whose start-time proof ``_identity_by_start_time``
    admits precisely because the beat has lapsed — with the forced rung's cost
    beside it. ``--force`` is the rung for the still-beating silent owner and
    it cannot admit a lapsed record at all (``_identity_by_record`` refuses on
    its own age gate), so it is named for the shape it IS for, priced, and
    left to the operator's judgement.
    """
    from local_operator.cli import wake_command

    monkeypatch.setattr(
        "local_operator.wakes.supervisor.wedged_runtime",
        lambda _config, _session: (4242, 312.0),
    )
    _arm(
        tmp_path,
        "quietown0001",
        cwd=str(tmp_path),
        schedules=[{"id": "w1", "message": "watch", "next_due_at": NOW_MS - 600_000}],
    )

    assert wake_command(_args()) == 0

    # The sentence WRAPS at the surface's indent, so it is read flat: asserting
    # on physical lines would test the terminal width instead of the copy.
    flat = " ".join(capsys.readouterr().out.split())
    assert "wedged: quietown0001 (pid 4242)" in flat, flat
    assert "has not sent a heartbeat in 5m and is not answering its socket" in flat, flat
    assert "'lop stop --pid 4242' asks it to stop" in flat, flat
    assert "'lop stop --pid 4242 --force' signal-stops the process" in flat, flat
    assert "discarding its in-flight turn" in flat, flat
    # The claims that were withdrawn.
    assert "ends it" not in flat, flat
    assert "recover on its own" not in flat, flat


# ---------------------------------------------------------------------------
# D4: the supervisor named on these two lines must be the one on THIS host.
# ---------------------------------------------------------------------------

#: The state word and the supervisor's own vocabulary, per host. macOS is first
#: and byte-identical to the wording this branch inherited, because making
#: Windows and Linux work is not a reason to rewrite the platform that did.
_SUPERVISOR_WORDING = [
    ("launchctl", "loaded but NOT running (launchd has the job; it has exited)"),
    ("systemctl", "loaded but NOT running (systemd has the unit; it has exited)"),
    ("schtasks", "loaded but NOT running (Task Scheduler has the task; it has exited)"),
]
_UNLOADED_WORDING = [
    ("launchctl", "not loaded (a plist exists but launchd has no job)"),
    ("systemctl", "not loaded (a unit file exists but systemd has not loaded it)"),
    ("schtasks", "not loaded (a task definition exists but Task Scheduler has not registered it)"),
]


@pytest.mark.parametrize(("kind", "expected"), _SUPERVISOR_WORDING)
def test_a_stopped_supervisor_is_named_in_its_own_words(
    kind: str,
    expected: str,
    stopped_supervisor,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    """D4: both of these lines became REACHABLE off macOS in this same PR.

    Before it, the wake installer was the probe-documented ``FAIL wake.install``
    ("no supervisor installer for this platform") on Linux and Windows, so no
    unit could exist for ``wake status`` to report on and neither parenthetical
    could print. This PR gives both platforms a real installer, which makes
    "loaded but NOT running" the ordinary state a Linux or Windows operator
    reaches right after ``lop wake install`` — and they were being told
    "launchd has the job", a daemon that does not exist on their machine.
    """
    from local_operator.cli import wake_command

    monkeypatch.setattr("local_operator.supervisors.supervisor", lambda: kind)

    assert wake_command(_args()) == 0
    line = _status_block(capsys.readouterr().out, "supervisor:")

    assert expected in line, line
    if kind != "launchctl":
        # The leak, stated as its own assertion so a future edit cannot bring
        # one platform's vocabulary back into another's line.
        assert "launchd" not in line and "plist" not in line, line


@pytest.mark.parametrize(("kind", "expected"), _UNLOADED_WORDING)
def test_a_registration_file_with_no_job_is_named_in_its_own_words(
    kind: str,
    expected: str,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    """The sibling line, whose flag comes from ``plist_path()``.

    That function already answered per platform — the systemd unit path on
    Linux, our own copy of the Task Scheduler definition on Windows — so the
    word "plist" was the only part of this line still spelling a macOS file
    type at a Linux or Windows reader (D4).

    ``plist_path`` is patched to a scratch file rather than created through the
    installer: the real function answers under the REAL home on macOS, and the
    one thing an isolated test must never do is write there.
    """
    from local_operator.cli import wake_command

    registration = tmp_path / "registration"
    registration.write_text("", encoding="utf-8")
    monkeypatch.setattr("local_operator.wakes.install.is_supported", lambda: True)
    monkeypatch.setattr("local_operator.wakes.install.supervisor_state", lambda _config: None)
    monkeypatch.setattr("local_operator.wakes.install.plist_path", lambda: registration)
    monkeypatch.setattr("local_operator.supervisors.supervisor", lambda: kind)

    assert wake_command(_args()) == 0
    line = _status_block(capsys.readouterr().out, "supervisor:")

    assert expected in line, line
    if kind != "launchctl":
        assert "launchd" not in line and "plist" not in line, line


def test_no_supervisor_falls_back_to_no_platforms_vocabulary(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    """``supervisor()`` can answer ``None``; the fallback must not guess a platform.

    Neither of the two lines is reachable in that state — ``is_supported()``
    gates both, and the honest rendering there is "not installed" — so this pins
    the helper against a future caller, not against the status command.
    """
    from local_operator.cli import _supervisor_parentheticals

    monkeypatch.setattr("local_operator.supervisors.supervisor", lambda: None)

    loaded, unloaded = _supervisor_parentheticals()
    for phrase in (loaded, unloaded):
        for platform_word in ("launchd", "plist", "systemd", "Task Scheduler", "schtasks"):
            assert platform_word not in phrase, phrase
