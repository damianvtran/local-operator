"""Guest disk reclamation: the threshold, the fail-soft paths, the evidence.

Nothing here touches the network or AWS. ``guest_disk`` takes its guest-command
runner and its clock as arguments precisely so the whole module is exercisable
against a stub, and the provider integration is covered separately in
``test_aws_provider.py`` against botocore's Stubber.

WHAT THESE TESTS ARE PROTECTING. The measured failure is that the released AMI
ships ~93% full and its own snapd fills the rest on a clock: root going 93% ->
100% used / 0 bytes free at t+383s, first ``ObservationPhaseError`` at t+424s,
across 7 of 8 runs and both instance types. ``pgrep -af ffmpeg`` showed NO
ffmpeg, which is why the screen-recorder theory in the 0.46.11 notes is wrong
and why these tests pin snapd's cache and auto-refresh specifically.

Each failure mode is exercised on its OWN, because "fails soft" is a claim about
every mode independently: a runner that raises, a step whose command exits
non-zero, an unparseable measurement, and a budget that runs out mid-pass must
each leave a usable report rather than an exception.

The clock is virtual (a list-backed counter), so no assertion depends on wall
time (AGENTS.md "Timing, flakes").
"""

from __future__ import annotations

import json
import subprocess
from pathlib import Path
from typing import Any, Sequence

import pytest
from lop_osworld_v2_adapter import guest_disk
from lop_osworld_v2_adapter.guest_disk import (
    COMMAND_TIMEOUT_S,
    RECLAIM_BELOW_FREE_BYTES,
    TOTAL_BUDGET_S,
    CommandResult,
    prepare_guest_disk,
)

# A guest below the threshold (the measured 2.2 GB) and one comfortably above.
TIGHT_FREE = 2_200_000_000
ROOMY_FREE = 40 * 1024**3


class _Guest:
    """A scripted guest control server.

    ``script`` maps a substring of the posted command to the reply. The first
    match wins, so a test names only the step it is about; everything else gets
    ``default``. Every command is recorded, which is what lets a test assert
    that a step ran WITHOUT asserting on a brittle verbatim string.
    """

    def __init__(
        self,
        *,
        free_bytes: int = TIGHT_FREE,
        script: dict[str, Any] | None = None,
        default: Any = None,
    ) -> None:
        self.commands: list[str] = []
        self.timeouts: list[float] = []
        self._script = script or {}
        self._free = free_bytes
        self._default = default if default is not None else CommandResult(0, "", "")

    def __call__(self, command: Sequence[str], timeout: float) -> CommandResult:
        script = " ".join(command)
        self.commands.append(script)
        self.timeouts.append(timeout)
        for needle, response in self._script.items():
            if needle in script:
                if isinstance(response, Exception):
                    raise response
                if isinstance(response, CommandResult):
                    return response
                # Anything else is a callable computing the reply from the
                # guest's current state -- used to model a cache clear that
                # actually frees space between the two measurements.
                result = response(self)
                assert isinstance(result, CommandResult)
                return result
        # A raising DEFAULT models a guest that answers nothing at all, so it
        # has to pre-empt the convenience replies below -- otherwise an
        # "unreachable guest" test would still get its measurements answered.
        if isinstance(self._default, Exception):
            raise self._default
        if "--output=avail" in script:
            return CommandResult(0, f"{self._free}\n", "")
        if "--output=size" in script:
            return CommandResult(0, "31138512896\n107374182400\n", "")
        return self._default

    def ran(self, needle: str) -> bool:
        return any(needle in command for command in self.commands)


class _Clock:
    """A virtual monotonic clock that advances only when a test says so."""

    def __init__(self, step: float = 0.0) -> None:
        self.now = 0.0
        self._step = step

    def __call__(self) -> float:
        value = self.now
        self.now += self._step
        return value


def _prepare(guest: _Guest, clock: Any = None) -> guest_disk.GuestDiskReport:
    return prepare_guest_disk(guest, client_password="pw", clock=clock or _Clock())


def _status(report: guest_disk.GuestDiskReport, name: str) -> str | None:
    for step in report.steps:
        if step.name == name:
            return step.status
    return None


# ----------------------------------------------------------------------
# The reclamation itself
# ----------------------------------------------------------------------


def test_a_tight_guest_aborts_downloads_holds_snap_refresh_and_clears_the_cache() -> None:
    """The three measured consumers, in the order that makes them stick.

    Aborting FIRST because it is immediate, while ``snap refresh --hold`` is a
    hook change the CLI waits on and that queues behind an in-flight refresh's
    own hooks -- long enough to outlast the per-command ceiling. Holding second
    stops a NEW auto-refresh (snapd's 20-minute retry delay covers the gap).
    Clearing last means nothing is still writing into the directory being
    emptied.
    """

    guest = _Guest(free_bytes=TIGHT_FREE)
    report = _prepare(guest)

    assert report.reclamation_attempted is True
    assert report.reason == "below-threshold"
    assert guest.ran("snap refresh --hold=forever")
    assert guest.ran("snap abort")
    assert guest.ran("/var/lib/snapd/cache")

    abort = next(i for i, c in enumerate(guest.commands) if "snap abort" in c)
    hold = next(i for i, c in enumerate(guest.commands) if "--hold=forever" in c)
    clear = next(i for i, c in enumerate(guest.commands) if "/var/lib/snapd/cache" in c)
    assert abort < hold < clear


def test_a_roomy_guest_is_measured_and_left_alone() -> None:
    """Above the threshold nothing is touched, but the measurement still happens.

    The threshold sits above snapd's largest measured appetite (a 9.7 GB cache),
    so a guest with this much free can absorb a full auto-refresh; running the
    hygiene there would be a write to a guest that does not need one.
    """

    guest = _Guest(free_bytes=ROOMY_FREE)
    report = _prepare(guest)

    assert report.reclamation_attempted is False
    assert report.reason == "above-threshold"
    assert report.free_bytes_before == ROOMY_FREE
    assert not guest.ran("snap refresh")
    assert not guest.ran("/var/lib/snapd/cache")


def test_the_threshold_is_the_boundary_between_the_two_behaviours() -> None:
    """Exactly at the threshold is 'enough': the guard is ``>=``.

    Pinned because an off-by-one here is invisible in production -- it only
    changes behaviour for a guest sitting on the exact byte count.
    """

    assert _prepare(_Guest(free_bytes=RECLAIM_BELOW_FREE_BYTES)).reclamation_attempted is False
    assert _prepare(_Guest(free_bytes=RECLAIM_BELOW_FREE_BYTES - 1)).reclamation_attempted is True


def test_the_snapd_cache_is_emptied_but_no_snap_is_ever_removed() -> None:
    """The line between housekeeping and changing the benchmark.

    Clearing a download cache costs a re-download. Uninstalling a snap would
    remove an application a task may legitimately need, which would change what
    the benchmark measures -- so no command may ever do it.
    """

    guest = _Guest(free_bytes=TIGHT_FREE)
    _prepare(guest)

    for command in guest.commands:
        assert "snap remove" not in command
        assert "apt-get remove" not in command
        assert "apt-get purge" not in command
    # The cache CONTENTS, never the directory itself: snapd expects it to exist.
    clear = next(c for c in guest.commands if "/var/lib/snapd/cache" in c)
    assert "find /var/lib/snapd/cache -mindepth 1 -delete" in clear
    # And no glob anywhere near it: a ``/*`` is expanded by whichever shell
    # holds it, and the OUTER shell is unprivileged (see the real-shell test
    # below for what that did).
    assert "*" not in clear


def _privileged_scripts(guest: _Guest) -> list[str]:
    """The ``bash -c`` scripts that reach ``sudo``, as the guest received them."""

    return [c.split("bash -c ", 1)[1] for c in guest.commands if "sudo -S" in c]


def test_every_privileged_step_is_one_sudo_running_one_inner_shell() -> None:
    """The shape that keeps the work on the privileged side of the boundary.

    The guest's control server is NOT root (upstream's server runs ``sudo -S``
    for everything, setup.py:661), so anything the outer shell does itself --
    expanding a glob, feeding ``xargs`` -- happens without privilege. Two real
    defects had that shape: ``rm -rf -- /var/lib/snapd/cache/*`` expanded to
    nothing against a ``drwx------ root:root`` directory and exited 0, and
    ``xargs -r -n1 echo 'pw' | sudo -S snap abort`` parsed as ``xargs echo``
    PIPED INTO one id-less ``snap abort``. Each step is therefore exactly one
    ``echo <pw> | sudo -S bash -c '<fragment>'`` with the password appearing
    once, and nothing else on the outer command line.
    """

    guest = _Guest(free_bytes=TIGHT_FREE)
    _prepare(guest)
    scripts = _privileged_scripts(guest)

    assert len(scripts) == 3  # abort, hold, clear
    for script in scripts:
        assert script.count("sudo") == 1
        assert script.count("'pw'") == 1
        assert script.startswith("echo 'pw' | sudo -S bash -c '")
        assert "xargs" not in script
        # No PATHNAME glob (a ``case`` pattern is matched, never expanded).
        assert "/*" not in script


def _install_fake_sudo(bin_dir: Path, *, password: str, marker: Path) -> None:
    """A ``sudo -S`` that regains access the way root would.

    It reads the password from stdin like the real one, refuses anything else,
    then runs the command with ``marker``'s parent tree made readable -- the
    model for "root can see what the unprivileged shell cannot". Whatever the
    command's stdin was is consumed by the password read, exactly as sudo's.
    """

    script = f"""#!/bin/bash
[ "$1" = "-S" ] && shift
IFS= read -r pw
if [ "$pw" != {guest_disk._shell_quote(password)} ]; then
  echo 'sudo: 1 incorrect password attempt' >&2; exit 1
fi
chmod -R u+rwx {guest_disk._shell_quote(str(marker))}
"$@"
rc=$?
chmod 000 {guest_disk._shell_quote(str(marker))} 2>/dev/null
exit $rc
"""
    (bin_dir / "sudo").write_text(script)
    (bin_dir / "sudo").chmod(0o755)


def test_the_cache_is_emptied_through_a_real_shell_when_only_root_can_read_it(
    tmp_path: Path,
) -> None:
    """B1, replayed against a real ``bash``: the delete must run as root.

    The cache directory is ``chmod 000`` -- the ``drwx------ root:root`` that
    snapd creates, as seen from a user who is not root. The fake ``sudo``
    restores access only for the command it runs. A glob expanded outside it
    matches nothing, ``rm -rf`` of the literal name exits 0, and the bytes
    stay; ``find -mindepth 1 -delete`` INSIDE it empties the directory.
    """

    cache = tmp_path / "var/lib/snapd/cache"
    (cache / "nested").mkdir(parents=True)
    (cache / "blob-1").write_bytes(b"x" * 4096)
    (cache / "nested" / "inner").write_bytes(b"y" * 4096)
    bin_dir = tmp_path / "bin"
    bin_dir.mkdir()
    _install_fake_sudo(bin_dir, password="pw", marker=cache)

    guest = _Guest(free_bytes=TIGHT_FREE)
    _prepare(guest)
    clear = next(s for s in _privileged_scripts(guest) if "/var/lib/snapd/cache" in s)
    # The production path is absolute; the fixture relocates it under tmp_path.
    clear = clear.replace("/var/lib/snapd/cache", str(cache))

    cache.chmod(0o000)
    try:
        completed = subprocess.run(
            ["bash", "-c", clear],
            capture_output=True,
            text=True,
            env={"PATH": f"{bin_dir}:/usr/bin:/bin"},
            check=False,
        )
    finally:
        cache.chmod(0o700)

    assert completed.returncode == 0, completed.stdout
    assert cache.is_dir(), "the directory itself must survive"
    assert list(cache.iterdir()) == [], "bytes must actually be gone"


def test_every_in_flight_refresh_is_aborted_by_id_through_a_real_shell(
    tmp_path: Path,
) -> None:
    """M1, replayed against a real ``bash`` and a ``snap`` with the real CLI rules.

    ``snap abort`` REQUIRES a change id ("please provide change ID or type
    with --last"), and only ``Doing`` rows whose summary is an auto-refresh or
    its pre-download are candidates: a seeding hook that happens to be running
    is not the benchmark's to abort, and a COMPLETED auto-refresh must not be
    touched at all (aborting it is an error in real snapd, and the intent it
    would express is a revert). The fake ``snap`` records every call it
    receives, so an id-less abort, an abort of the wrong change, or an abort
    running as the wrong user is visible rather than masked by ``|| true``.
    """

    bin_dir = tmp_path / "bin"
    bin_dir.mkdir()
    calls = tmp_path / "snap-calls"
    guard = tmp_path / "root-only"
    guard.mkdir()
    _install_fake_sudo(bin_dir, password="pw", marker=guard)
    (bin_dir / "snap").write_text(f"""#!/bin/bash
echo "$*" >> {guest_disk._shell_quote(str(calls))}
case "$1" in
  changes)
    printf 'ID   Status  Spawn  Ready  Summary\\n'
    printf '10   Done    -      -      Initialize system state\\n'
    printf '11   Done    -      -      Auto-refresh 3 snaps\\n'
    printf '12   Doing   -      -      Auto-refresh 9 snaps\\n'
    printf '13   Doing   -      -      Pre-download "novnc" for auto-refresh\\n'
    printf '14   Doing   -      -      Run configure hook of "seed-thing" snap\\n'
    printf '15   Doing   -      -      Hold auto-refreshes for all snaps\\n'
    ;;
  abort)
    [ -r {guest_disk._shell_quote(str(guard))} ] || {{ echo 'error: access denied' >&2; exit 1; }}
    [ -n "$2" ] || {{ echo 'error: please provide change ID or type with --last' >&2; exit 1; }}
    ;;
esac
""")
    (bin_dir / "snap").chmod(0o755)

    guest = _Guest(free_bytes=TIGHT_FREE)
    _prepare(guest)
    abort = next(s for s in _privileged_scripts(guest) if "snap abort" in s)

    guard.chmod(0o000)
    try:
        completed = subprocess.run(
            ["bash", "-c", abort],
            capture_output=True,
            text=True,
            env={"PATH": f"{bin_dir}:/usr/bin:/bin"},
            check=False,
        )
    finally:
        guard.chmod(0o700)

    assert completed.returncode == 0, completed.stdout
    assert calls.read_text().splitlines() == ["changes", "abort 12", "abort 13"]


def test_a_failed_abort_is_recorded_as_a_failed_step_rather_than_masked() -> None:
    """``|| true`` used to turn an abort that did nothing into ``ok``."""

    guest = _Guest(
        free_bytes=TIGHT_FREE,
        script={"snap abort": CommandResult(1, "", "error: cannot abort change 12")},
    )
    report = _prepare(guest)

    assert _status(report, "abort-in-flight-snap-changes") == "failed"
    abort = next(s for s in _privileged_scripts(guest) if "snap abort" in s)
    assert "|| true" not in abort


def test_an_old_snapd_without_hold_falls_back_to_the_refresh_hold_setting() -> None:
    """``snap refresh --hold`` needs snapd 2.58+; older guests get the setting.

    The fallback is CONDITIONAL on the first failing, so a modern guest is not
    written to twice for the same effect.
    """

    guest = _Guest(
        free_bytes=TIGHT_FREE,
        script={"--hold=forever": CommandResult(1, "", "unknown flag `hold'")},
    )
    report = _prepare(guest)

    assert _status(report, "hold-snap-auto-refresh") == "failed"
    assert _status(report, "hold-snap-auto-refresh-fallback") == "ok"
    assert guest.ran("snap set system refresh.hold=")


def test_a_modern_snapd_does_not_run_the_fallback() -> None:
    guest = _Guest(free_bytes=TIGHT_FREE)
    report = _prepare(guest)

    assert _status(report, "hold-snap-auto-refresh") == "ok"
    assert _status(report, "hold-snap-auto-refresh-fallback") is None
    assert not guest.ran("refresh.hold=")


@pytest.mark.parametrize(
    "password",
    [
        "osworld-public-evaluation",
        "pw'; touch /tmp/pwned; echo '",
        'pw"$(id)`id`\\',
        "pw with spaces",
    ],
)
def test_the_client_password_survives_a_real_shell_as_exactly_one_word(
    password: str, tmp_path: Path
) -> None:
    """A password with shell metacharacters must not be able to escape the quote.

    ``sudo -S`` needs the password on stdin and the guest endpoint has no stdin,
    so it arrives through a pipeline (upstream's own pattern, setup.py:609).
    That makes quoting a correctness requirement rather than a style choice.

    Verified against a REAL ``bash``, not by matching the quoted string: the
    property that matters is what a shell does with it, and a string assertion
    would pass just as happily on a quoting scheme that a shell mis-parses.
    The command's echo half is replayed with the ``sudo`` half replaced, so the
    injection would land here if it landed anywhere.
    """

    guest = _Guest(free_bytes=TIGHT_FREE)
    prepare_guest_disk(guest, client_password=password, clock=_Clock())
    hold = next(c for c in guest.commands if "--hold=forever" in c)

    # The generated script is ``bash -c <script>``; take the script and cut the
    # pipeline at the pipe, leaving exactly the ``echo <quoted-password>`` the
    # module built.
    script = hold.split("bash -c ", 1)[1]
    echo_half = script.split(" | sudo -S ", 1)[0]
    canary = tmp_path / "pwned"
    completed = subprocess.run(
        ["bash", "-c", echo_half],
        capture_output=True,
        text=True,
        cwd=tmp_path,
        check=False,
    )

    assert completed.returncode == 0
    # Exactly the password, nothing expanded, nothing split off, nothing run.
    assert completed.stdout == password + "\n"
    assert completed.stderr == ""
    assert not canary.exists()


# ----------------------------------------------------------------------
# Fail-soft: each mode on its own
# ----------------------------------------------------------------------


def test_an_unreachable_guest_returns_a_report_instead_of_raising() -> None:
    """The whole fail-soft contract in one case: nothing propagates.

    An episode that would have worked must not be destroyed by a hygiene step,
    so a control server that answers nothing produces a report saying so.
    """

    guest = _Guest(default=ConnectionError("connection refused"))
    report = _prepare(guest)

    assert report.free_bytes_before is None
    # Unmeasurable means reclaim ANYWAY: the protection must not go missing
    # exactly when the guest is least healthy. Asserted on the ATTEMPTED steps,
    # not only on the flag -- a flag alone would still be set by an
    # implementation that skipped the work.
    assert report.reclamation_attempted is True
    assert report.reason == "unmeasured"
    assert guest.ran("snap refresh --hold=forever")
    assert guest.ran("/var/lib/snapd/cache")
    assert all(step.status == "unreachable" for step in report.steps)
    # The exception TYPE is recorded; its message is not, because a transport
    # error echoes the URL it was given.
    assert all(step.detail == "ConnectionError" for step in report.steps)


def test_a_denied_sudo_is_recorded_and_the_remaining_steps_still_run() -> None:
    """One failing step must not abort the pass: the others may still help."""

    guest = _Guest(
        free_bytes=TIGHT_FREE,
        script={"snap refresh": CommandResult(1, "", "sudo: a password is required")},
    )
    report = _prepare(guest)

    assert _status(report, "hold-snap-auto-refresh") == "failed"
    assert _status(report, "clear-snapd-cache") == "ok"
    assert guest.ran("/var/lib/snapd/cache")


def test_a_missing_binary_is_recorded_as_a_failed_step_not_an_error() -> None:
    guest = _Guest(
        free_bytes=TIGHT_FREE,
        script={"snap changes": CommandResult(127, "", "snap: command not found")},
    )
    report = _prepare(guest)

    assert _status(report, "abort-in-flight-snap-changes") == "failed"
    detail = next(s.detail for s in report.steps if s.name == "abort-in-flight-snap-changes")
    assert "not found" in detail


def test_an_unparseable_measurement_reads_as_unknown_rather_than_zero() -> None:
    """A garbled ``df`` must never be read as "0 bytes free".

    Inventing a number would be worse than reporting none: a later reader would
    take it as measured fact.
    """

    guest = _Guest(script={"--output=avail": CommandResult(0, "Filesystem\n", "")})
    report = _prepare(guest)

    assert report.free_bytes_before is None
    assert report.reason == "unmeasured"
    # An unmeasurable guest is still reclaimed: skipping would make the
    # protection absent exactly when the guest is least healthy, and this guest
    # ANSWERS -- only its measurement is unparseable.
    assert report.reclamation_attempted is True
    assert guest.ran("snap refresh --hold=forever")


def test_a_transport_failure_on_one_step_alone_does_not_stop_the_others() -> None:
    guest = _Guest(
        free_bytes=TIGHT_FREE,
        script={"snap abort": TimeoutError("read timed out")},
    )
    report = _prepare(guest)

    assert _status(report, "abort-in-flight-snap-changes") == "unreachable"
    assert _status(report, "clear-snapd-cache") == "ok"


def test_an_exhausted_budget_skips_the_remaining_steps_rather_than_hanging() -> None:
    """A wedged guest must not eat the reset budget.

    Each command already has its own timeout; the whole-pass budget is what
    bounds a guest that answers every call slowly rather than not at all.
    """

    clock = _Clock(step=TOTAL_BUDGET_S)
    guest = _Guest(free_bytes=TIGHT_FREE)
    report = _prepare(guest, clock=clock)

    skipped = [step for step in report.steps if step.status == "skipped"]
    assert skipped, "a spent budget must skip, not run"
    assert all(step.detail == "budget exhausted" for step in skipped)


def test_a_command_timeout_never_exceeds_the_remaining_budget() -> None:
    """The per-command ceiling is the smaller of its own and what is left."""

    guest = _Guest(free_bytes=TIGHT_FREE)
    _prepare(guest)
    assert guest.timeouts
    assert max(guest.timeouts) <= COMMAND_TIMEOUT_S
    assert all(timeout > 0 for timeout in guest.timeouts)


# ----------------------------------------------------------------------
# Observability
# ----------------------------------------------------------------------


def test_the_report_records_free_space_before_and_after_reclamation() -> None:
    """ "The guest had N bytes free at the start" is the point of the whole step.

    It is the fact a later "environment returned no screenshot frame" is read
    against, and it must not depend on anyone having probed the guest by hand.
    """

    freed = {"value": TIGHT_FREE}

    def measure(guest: _Guest) -> CommandResult:
        return CommandResult(0, f"{freed['value']}\n", "")

    def clear(guest: _Guest) -> CommandResult:
        freed["value"] = 12_000_000_000
        return CommandResult(0, "", "")

    guest = _Guest(
        script={"--output=avail": measure, "/var/lib/snapd/cache": clear},
    )
    report = _prepare(guest)

    assert report.free_bytes_before == TIGHT_FREE
    assert report.free_bytes_after == 12_000_000_000
    assert report.free_bytes_after > report.free_bytes_before


def test_the_report_records_the_partition_and_whole_disk_geometry() -> None:
    """A 29.5G filesystem inside a 100 GiB disk is the AWS_ROOT_VOLUME_SIZE story.

    Reported read-only: nothing here repartitions (``growpart`` is absent from
    the AMI and an ``sfdisk`` rewrite can fail HARD, which a hygiene step must
    never do), but the two numbers are what tell the next reader whether the
    volume override bought anything.
    """

    report = _prepare(_Guest(free_bytes=TIGHT_FREE))

    assert report.filesystem_bytes == 31138512896
    assert report.disk_bytes == 107374182400
    assert report.disk_bytes > report.filesystem_bytes


@pytest.mark.parametrize("returncode", [0, 127])
def test_a_partial_geometry_answer_still_reports_the_filesystem_size(returncode: int) -> None:
    """An image without ``lsblk`` must still yield the first number.

    Parametrised over the exit status because a missing tool is exactly the
    case where the shell reports failure (127) AFTER ``df`` has already
    printed: the number that was measured must not be discarded over a tool
    that was not there.
    """

    guest = _Guest(
        free_bytes=TIGHT_FREE,
        script={"--output=size": CommandResult(returncode, "31138512896\n", "lsblk: not found")},
    )
    report = _prepare(guest)

    assert report.filesystem_bytes == 31138512896
    assert report.disk_bytes is None


def test_the_geometry_script_exits_zero_through_a_real_shell_without_lsblk(
    tmp_path: Path,
) -> None:
    """The block-device half is best-effort, so the step itself reports ``ok``.

    Run against a real ``bash`` with a PATH holding ``df`` and nothing else:
    the ``&&`` chain used to leak its 127 into the step status, which the
    report then showed as a FAILED measurement beside a perfectly good
    filesystem figure.
    """

    bin_dir = tmp_path / "bin"
    bin_dir.mkdir()
    (bin_dir / "df").write_text("#!/bin/bash\nprintf '  1B-blocks\\n31138512896\\n'\n")
    (bin_dir / "df").chmod(0o755)
    for tool in ("tail", "head"):
        (bin_dir / tool).symlink_to(f"/usr/bin/{tool}")

    completed = subprocess.run(
        ["/bin/bash", "-c", guest_disk._GEOMETRY_SCRIPT],
        capture_output=True,
        text=True,
        env={"PATH": str(bin_dir)},
        check=False,
    )

    assert completed.returncode == 0, completed.stderr
    assert completed.stdout == "31138512896\n"


def test_the_report_serialises_to_portable_json_with_every_step() -> None:
    """The report is written to a file, so it has to round-trip as JSON."""

    report = _prepare(_Guest(free_bytes=TIGHT_FREE))
    decoded = json.loads(report.to_json_bytes())

    assert decoded["free_bytes_before"] == TIGHT_FREE
    assert decoded["threshold_bytes"] == RECLAIM_BELOW_FREE_BYTES
    # ATTEMPTED, never "achieved": whether it worked is the steps plus the two
    # measurements, which cannot claim something that did not happen.
    assert decoded["reclamation_attempted"] is True
    assert "reclaimed" not in decoded
    assert decoded["reason"] == "below-threshold"
    names = [step["name"] for step in decoded["steps"]]
    assert "measure-free-before" in names
    assert "clear-snapd-cache" in names
    assert "measure-free-after" in names
    for step in decoded["steps"]:
        assert step["status"] in {"ok", "failed", "unreachable", "skipped"}


def test_the_report_never_carries_the_client_password() -> None:
    """The commands are not recorded, and that is why.

    Every privileged step runs through ``echo <password> | sudo -S``, so a
    verbatim argv in the report would put the password in a file on the
    operator's disk for no diagnostic gain.
    """

    marker = "marker-client-password"
    guest = _Guest(
        free_bytes=TIGHT_FREE,
        script={"snap refresh": CommandResult(1, "", "sudo: 1 incorrect password attempt")},
    )
    report = prepare_guest_disk(guest, client_password=marker, clock=_Clock())

    assert marker not in report.to_json_bytes().decode()
    assert marker in " ".join(guest.commands), "the password must reach the guest"


@pytest.mark.parametrize("free", [TIGHT_FREE, ROOMY_FREE, None])
def test_a_report_is_produced_on_every_path(free: int | None) -> None:
    """Above, below, and unmeasurable all produce a usable report.

    There is no path on which the episode's evidence loses the disk facts.
    """

    guest = _Guest(default=ConnectionError("refused")) if free is None else _Guest(free_bytes=free)
    report = _prepare(guest)

    assert report.threshold_bytes == RECLAIM_BELOW_FREE_BYTES
    assert report.reason in {"above-threshold", "below-threshold", "unmeasured"}
    assert report.steps
    json.loads(report.to_json_bytes())
