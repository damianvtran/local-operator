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
