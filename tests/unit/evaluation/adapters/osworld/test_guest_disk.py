"""Guest disk reclamation: the threshold, the fail-soft paths, the evidence.

Nothing here touches the network or AWS. ``guest_disk`` takes its guest-command
runner and its clock as arguments precisely so the whole module is exercisable
against a stub, and the provider integration is covered separately in
``test_aws_provider.py`` against botocore's Stubber.

WHAT THESE TESTS ARE PROTECTING. The measured failure is that the released AMI
ships ~93% full and its own snapd fills the rest on a clock: root going 93% ->
100% used / 0 bytes free at t+383s, first ``ObservationPhaseError`` at t+424s,
across 7 of 8 runs and both instance types. ``pgrep -af ffmpeg`` showed NO
ffmpeg, which is why the screen-recorder theory in the 0.46.11 notes is wrong,
and the directories are what the live guest disproved next: these tests pin
snapd's ``*.partial`` download scratch and its auto-refresh, not the cache
directory an earlier revision named (4096 bytes on this image).

Each failure mode is exercised on its OWN, because "fails soft" is a claim about
every mode independently: a runner that raises, a step whose command exits
non-zero, an unparseable measurement, and a budget that runs out mid-pass must
each leave a usable report rather than an exception.

The clock is virtual (a list-backed counter), so no assertion depends on wall
time (AGENTS.md "Timing, flakes").
"""

from __future__ import annotations

import json
import re
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
    assert guest.ran("/var/lib/snapd/snaps")

    abort = next(i for i, c in enumerate(guest.commands) if "snap abort" in c)
    hold = next(i for i, c in enumerate(guest.commands) if "--hold=forever" in c)
    clear = next(i for i, c in enumerate(guest.commands) if "/var/lib/snapd/snaps" in c)
    assert abort < hold < clear


def test_a_roomy_guest_is_measured_and_left_alone() -> None:
    """Above the threshold nothing is touched, but the measurement still happens.

    The threshold sits above snapd's largest measured appetite (the ~10 GB of
    delta downloads in the revision directory), so a guest with this much free
    can absorb a full auto-refresh; running the hygiene there would be a write to
    a guest that does not need one.
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


def test_the_download_scratch_is_emptied_but_no_snap_is_ever_removed() -> None:
    """The directory that fills, and the line that must not be crossed.

    Clearing a download costs a re-download. Uninstalling a snap would remove an
    application a task may legitimately need, which would change what the
    benchmark measures -- so no command may ever do it, and that includes the
    ``.snap`` files sitting in the same directory as the incomplete downloads: a
    ``.snap`` file there is an INSTALLED revision, mounted through a loop device.
    """

    guest = _Guest(free_bytes=TIGHT_FREE)
    _prepare(guest)

    for command in guest.commands:
        assert "snap remove" not in command
        assert "apt-get remove" not in command
        assert "apt-get purge" not in command
        assert "rm -rf" not in command
    # The cache CONTENTS, never the directory itself: snapd expects it to exist.
    clear = next(c for c in guest.commands if "/var/lib/snapd/snaps" in c)
    assert "find /var/lib/snapd/cache -mindepth 1 -delete" in clear
    # The measured consumer: incomplete downloads, matched by SUFFIX so an
    # installed revision cannot be caught by the same command.
    assert "rm -f -- *.partial" in clear
    assert "*.snap" not in clear
    # And no glob left for the OUTER shell to expand (see the real-shell test
    # below for what that did): the only one in play is the suffix the
    # PRIVILEGED shell matches, inside the quoted fragment.
    assert "*" not in _unquoted(clear)
    assert "*.partial" in _unquoted(clear.replace("'", ""))


def _unquoted(shell: str) -> str:
    """A generated script with every single-quoted span removed.

    What is left is what the OUTER shell would see, and therefore what it would
    expand: a glob surviving in here is the defect that once made a cache clear
    delete nothing while reporting success.
    """

    return re.sub(r"'[^']*'", "", shell)


def _privileged_scripts(guest: _Guest) -> list[str]:
    """The ``bash -c`` scripts that reach ``sudo``, as the guest received them."""

    return [c.split("bash -c ", 1)[1] for c in guest.commands if "sudo -S" in c]


def test_every_privileged_step_is_one_privileged_shell_reached_by_the_ladder() -> None:
    """The shape that keeps the work on the privileged side of the boundary.

    The guest's control server is NOT root (upstream's server runs ``sudo -S``
    for everything, setup.py:661), so anything the outer shell does itself --
    expanding a glob, feeding ``xargs`` -- happens without privilege. Two real
    defects had that shape: ``rm -rf -- /var/lib/snapd/cache/*`` expanded to
    nothing against a ``drwx------ root:root`` directory and exited 0, and
    ``xargs -r -n1 echo 'pw' | sudo -S snap abort`` parsed as ``xargs echo``
    PIPED INTO one id-less ``snap abort``. So: no ``xargs``, no glob outside a
    quoted fragment, and one candidate per ``sudo`` -- the agentless rung with
    no password at all, then exactly one ``sudo -S`` per candidate, each the
    stdin of exactly that sudo.
    """

    guest = _Guest(free_bytes=TIGHT_FREE)
    _prepare(guest)
    scripts = _privileged_scripts(guest)

    assert len(scripts) == 3  # abort, hold, clear
    for script in scripts:
        assert script.count("sudo -n bash -c") == 1
        assert script.count("| sudo -S bash -c") == 1
        assert script.count("'pw'") == 1
        assert "xargs" not in script
        # No PATHNAME glob survives outside a quoted fragment, so the outer
        # shell has nothing to expand (the abort fragment's ``case`` patterns
        # are matched, never expanded, and the clear fragment's suffix glob is
        # expanded by the privileged shell).
        assert "*" not in _unquoted(script)


def _install_fake_sudo(
    bin_dir: Path, *, password: str, marker: Path, nopasswd: bool = False
) -> None:
    """A ``sudo`` that regains access the way root would, in both modes.

    ``-n`` -- the escalation ladder's first rung -- is always refused, which is
    what the real one does when no credential is cached and is the whole reason
    the rung exists. ``-S`` reads the password from stdin like the real one,
    refuses anything else, then runs the command with ``marker``'s parent tree
    made readable -- the model for "root can see what the unprivileged shell
    cannot". Whatever the command's stdin was is consumed by the password read,
    exactly as sudo's.
    """

    # The options both ladders use: ``-n`` (refused unless ``nopasswd``, as on
    # a guest with no NOPASSWD rule and no cached credential), ``-S`` (read
    # the password from stdin), ``-v`` (validate only, run nothing), ``-k``
    # (ignore cached credentials -- this stub caches none) and ``-p ''``.
    script = f"""#!/bin/bash
nonint=0; validate=0
while [ $# -gt 0 ]; do
  case "$1" in
    -n) nonint=1; shift;; -S|-k) shift;; -v) validate=1; shift;;
    -p) shift 2;; --) shift; break;; *) break;;
  esac
done
if [ $nonint = 1 ] && [ {"1" if nopasswd else "0"} = 0 ]; then
  echo 'sudo: a password is required' >&2; exit 1
fi
if [ $nonint = 0 ]; then
  IFS= read -r pw
  if [ "$pw" != {guest_disk._shell_quote(password)} ]; then
    echo 'sudo: 1 incorrect password attempt' >&2; exit 1
  fi
fi
[ $validate = 1 ] && exit 0
chmod -R u+rwx {guest_disk._shell_quote(str(marker))}
"$@"
rc=$?
chmod 000 {guest_disk._shell_quote(str(marker))} 2>/dev/null
exit $rc
"""
    (bin_dir / "sudo").write_text(script)
    (bin_dir / "sudo").chmod(0o755)


def _bash_script_of(step_command: str) -> str:
    """The outer ``bash -c`` script of a generated step command."""

    return step_command.split("bash -c ", 1)[1]


def _run_in_fake_guest(script: str, bin_dir: Path) -> subprocess.CompletedProcess[str]:
    """Run a generated script against a real ``bash`` and a fake ``sudo``."""

    return subprocess.run(
        ["bash", "-c", script],
        capture_output=True,
        text=True,
        env={"PATH": f"{bin_dir}:/usr/bin:/bin"},
        check=False,
    )


def test_the_download_scratch_is_emptied_through_a_real_shell_when_only_root_can_read_it(
    tmp_path: Path,
) -> None:
    """B1, replayed against a real ``bash``: the delete must run as root.

    The cache directory is ``chmod 000`` -- the ``drwx------ root:root`` that
    snapd creates, as seen from a user who is not root. The fake ``sudo``
    restores access only for the command it runs. A glob expanded outside it
    matches nothing, ``rm -rf`` of the literal name exits 0, and the bytes
    stay; the fragment does its work INSIDE the privileged shell.

    THE SECOND HALF IS THE ONE THAT RECLAIMS ANYTHING ON THIS IMAGE, and it is
    asserted on a real filesystem rather than on the shape of a string: the
    ``*.partial`` download goes, and an installed revision -- a mounted snap, an
    application the benchmark may need -- stays, as does a file that is neither.
    """

    cache = tmp_path / "var/lib/snapd/cache"
    (cache / "nested").mkdir(parents=True)
    (cache / "blob-1").write_bytes(b"x" * 4096)
    (cache / "nested" / "inner").write_bytes(b"y" * 4096)
    snaps = tmp_path / "var/lib/snapd/snaps"
    snaps.mkdir(parents=True)
    (snaps / "kf6-core24_64.snap").write_bytes(b"installed revision")
    (snaps / "audacity_1239.snap.xdelta3-1212-to-1239.partial").write_bytes(b"z" * 4096)
    (snaps / "kf6-core24_64.snap.xdelta3-36-to-64.partial").write_bytes(b"z" * 4096)
    bin_dir = tmp_path / "bin"
    bin_dir.mkdir()
    _install_fake_sudo(bin_dir, password="pw", marker=cache)

    guest = _Guest(free_bytes=TIGHT_FREE)
    _prepare(guest)
    clear = next(s for s in _privileged_scripts(guest) if "/var/lib/snapd/snaps" in s)
    # The production paths are absolute; the fixture relocates them under tmp_path.
    clear = clear.replace("/var/lib/snapd/cache", str(cache)).replace(
        "/var/lib/snapd/snaps", str(snaps)
    )

    cache.chmod(0o000)
    try:
        completed = _run_in_fake_guest(clear, bin_dir)
    finally:
        cache.chmod(0o700)

    assert completed.returncode == 0, completed.stdout
    assert cache.is_dir(), "the directory itself must survive"
    assert list(cache.iterdir()) == [], "the cache bytes must actually be gone"
    assert sorted(p.name for p in snaps.iterdir()) == [
        "kf6-core24_64.snap"
    ], "the incomplete downloads go; the installed revision stays"


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
    That makes quoting a correctness requirement rather than a style choice, and
    the ladder multiplies the number of places a candidate enters the script.

    Verified against a REAL ``bash``, not by matching the quoted string: the
    property that matters is what a shell does with it, and a string assertion
    would pass just as happily on a quoting scheme that a shell mis-parses.
    The command's ``printf`` half of the SUPPLIED tier -- the first attempt that
    carries a password at all -- is replayed with the ``sudo`` half replaced, so
    the injection would land here if it landed anywhere.
    """

    guest = _Guest(free_bytes=TIGHT_FREE)
    prepare_guest_disk(guest, client_password=password, clock=_Clock())
    hold = next(c for c in guest.commands if "--hold=forever" in c)

    # The generated script is ``bash -c <script>``; cut it at the first
    # `` | sudo -S `` and keep the last ``printf`` before it, which is the whole
    # of what the module hands that sudo as stdin.
    script = _bash_script_of(hold)
    before_pipe = script.split(" | sudo -S ", 1)[0]
    printf_half = before_pipe[before_pipe.rindex("printf ") :]
    canary = tmp_path / "pwned"
    completed = subprocess.run(
        ["bash", "-c", printf_half],
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
# The escalation ladder: which candidate opens the guest
# ----------------------------------------------------------------------


def test_the_ladder_tries_sudo_n_before_any_password_is_used() -> None:
    """Order, reach and the agentless rung, pinned.

    ``sudo -n`` first: it is the only attempt that cannot fail for the wrong
    reason, and a live sudo timestamp makes it free. Then exactly one candidate
    per remaining rung, each the stdin of exactly one ``sudo -S``. The
    agentless rung carries NO password -- nothing reads stdin there, and piping
    a credential into a command whose whole point is that it used none is how a
    password ends up somewhere nobody expected it.
    """

    guest = _Guest(free_bytes=TIGHT_FREE)
    _prepare(guest)
    hold = next(c for c in guest.commands if "--hold=forever" in c)

    assert hold.count("sudo -n bash -c") == 1
    assert hold.count("| sudo -S bash -c") == 1
    assert hold.count("'pw'") == 1
    assert hold.index("sudo -n bash -c") < hold.index("| sudo -S bash -c")
    assert "| sudo -n" not in hold


def test_a_documented_upstream_default_unlocks_upstreams_own_candidates() -> None:
    """The gate, and the measured case it exists for.

    The campaign passed upstream's older development default, which this image
    rejects, while the value this project's runbook documents is the one that
    works. Both are upstream's OWN documented defaults, so the ladder reaches
    the second one -- the same discipline, the same values and the same order as
    upstream's ``expand_guest_volume``.
    """

    first, second = guest_disk._UPSTREAM_DEFAULT_PASSWORDS
    guest = _Guest(free_bytes=TIGHT_FREE)
    prepare_guest_disk(guest, client_password=second, clock=_Clock())
    hold = next(c for c in guest.commands if "--hold=forever" in c)

    assert hold.count(f"'{second}'") == 1
    assert hold.count(f"'{first}'") == 1
    assert hold.count("| sudo -S bash -c") == 2


def test_a_value_the_operator_chose_is_not_second_guessed() -> None:
    """The gate's other half: a deliberate value unlocks nothing.

    There is no rescue to perform here -- if this value is rejected, the steps
    fail and the caller refuses the episode with the knob named. Succeeding
    instead on a credential nobody supplied is not a rescue; it is a silent
    substitution, and it leaves the report's escalation chain as the only record
    of which password actually opened the guest.
    """

    guest = _Guest(free_bytes=TIGHT_FREE)
    prepare_guest_disk(guest, client_password="a-value-of-my-own", clock=_Clock())
    hold = next(c for c in guest.commands if "--hold=forever" in c)

    for default in guest_disk._UPSTREAM_DEFAULT_PASSWORDS:
        assert default not in hold
    assert hold.count("'a-value-of-my-own'") == 1
    assert hold.count("| sudo -S bash -c") == 1


def test_the_ladder_reclaims_a_guest_that_rejects_the_supplied_value(
    tmp_path: Path,
) -> None:
    """The four-episode failure replayed through a real shell, and how it reads.

    The fake ``sudo`` accepts only the OTHER documented default -- the image's
    own -- and refuses ``-n`` and everything else, which is the measured
    situation. What matters beyond the bytes reclaimed is the last line of the
    script: the chain says which rungs were refused and which one opened the
    guest, so a stale infra value is visible in the evidence instead of silently
    corrected.
    """

    first, second = guest_disk._UPSTREAM_DEFAULT_PASSWORDS
    bin_dir = tmp_path / "bin"
    bin_dir.mkdir()
    guard = tmp_path / "guard"
    guard.mkdir()
    _install_fake_sudo(bin_dir, password=first, marker=guard)

    completed = _run_in_fake_guest(guest_disk.escalation_script("echo RECLAIMED", second), bin_dir)

    assert completed.returncode == 0, completed.stdout
    assert "RECLAIMED" in completed.stdout
    # The chain, in order: ``sudo -n`` was refused, the operator's value was
    # refused, and the image's own documented default opened the guest.
    assert "escalation=agentless>supplied>upstream-default" in completed.stdout
    assert second not in completed.stdout, "the tier chain is recorded, never a value"


def test_a_step_no_candidate_can_open_fails_with_its_tier_recorded(tmp_path: Path) -> None:
    """A dead ladder is a FAILED step, not a silent one.

    ``rc`` is sudo's, so the caller sees the failure; ``out`` is the last
    attempt's output, so the auth failure is in the step's detail; and the chain
    names every rung that was in play, so the reader knows the operator's value
    was among the candidates that were refused.
    """

    bin_dir = tmp_path / "bin"
    bin_dir.mkdir()
    guard = tmp_path / "guard"
    guard.mkdir()
    _install_fake_sudo(bin_dir, password="none-of-them", marker=guard)

    completed = _run_in_fake_guest(
        guest_disk.escalation_script("echo RECLAIMED", "password"), bin_dir
    )

    assert completed.returncode == 1
    assert "RECLAIMED" not in completed.stdout
    # Every rung in play, in the order tried, ending on the last one.
    assert "escalation=agentless>supplied>upstream-default" in completed.stdout
    assert "sudo: 1 incorrect password attempt" in completed.stdout


def test_the_ladder_records_the_rungs_it_tried_and_no_values() -> None:
    """The step's detail names the chain of rungs, never a password.

    A reader of ``guest-preparation.json`` has to be able to tell "the operator's
    value was refused and a documented default opened the guest" from "the
    operator's value worked" -- and, when a step failed, to tell which rungs were
    in play before it did. The chain says where it stopped; the collected output
    says why.
    """

    first, second = guest_disk._UPSTREAM_DEFAULT_PASSWORDS
    script = guest_disk.escalation_script("echo INNER", second)

    # The first attempted rung is assigned literally, and every later one
    # EXTENDS the chain -- so the label can never name a rung that was not
    # reached.
    assert "tiers='agentless'" in script
    assert 'tiers="$tiers>supplied"' in script
    assert 'tiers="$tiers>upstream-default"' in script
    # Each candidate appears exactly once, as the stdin of exactly one sudo.
    assert script.count(f"'{second}'") == 1
    assert script.count(f"'{first}'") == 1
    assert script.count("| sudo -S bash -c") == 2
    assert "tier=" not in script, "the single-tier label is gone"


# ----------------------------------------------------------------------
# Blocking steps: what the caller refuses an episode on
# ----------------------------------------------------------------------


def test_a_reclamation_whose_steps_failed_is_reported_as_blocking() -> None:
    """THE MEASURED FAILURE, at the level the caller acts on.

    Every privileged step comes back ``sudo: no password was provided`` because
    this image rejects the value the campaign passed. The report recorded it and
    the caller walked on, so the episode ran into a guest whose root filesystem
    then filled -- it died minutes later and the operator saw a transport error.
    ``blocking_steps`` is what turns that into a preparation failure.
    """

    guest = _Guest(
        free_bytes=TIGHT_FREE,
        script={"sudo -S": CommandResult(1, "", "sudo: no password was provided")},
    )
    report = _prepare(guest)

    assert [step.name for step in report.blocking_steps()] == [
        "abort-in-flight-snap-changes",
        "hold-snap-auto-refresh",
        "clear-snapd-download-scratch",
    ]
    assert json.loads(report.to_json_bytes())["blocking_steps"] == [
        "abort-in-flight-snap-changes",
        "hold-snap-auto-refresh",
        "clear-snapd-download-scratch",
    ]


def test_a_healthy_but_still_tight_guest_is_not_blocking() -> None:
    """The warn-not-abort decision, pinned against the measurement.

    A successful reclamation leaves the guest at ~2.2 GB free -- measured on the
    run that completed, scored, and served 200 on all 50 screenshots -- while the
    threshold is 12 GiB, so "still below the threshold afterwards" is the NORMAL
    completed state and cannot be the error condition: refusing on it would
    refuse every healthy episode while protecting nothing. What protects the run
    is that the hold and the clear LANDED.
    """

    report = _prepare(_Guest(free_bytes=TIGHT_FREE))

    assert report.free_bytes_after == TIGHT_FREE
    assert report.free_bytes_after < RECLAIM_BELOW_FREE_BYTES
    assert report.blocking_steps() == ()


def test_an_above_threshold_guest_blocks_nothing() -> None:
    """No reclamation was attempted, so there is nothing that failed to land."""

    report = _prepare(_Guest(free_bytes=ROOMY_FREE))

    assert report.reclamation_attempted is False
    assert report.blocking_steps() == ()


def test_a_pass_that_ran_out_of_budget_is_blocking() -> None:
    """Skipped is not done: a step the pass never reached has not landed.

    The budget exists so a wedged guest cannot eat the reset window, and it
    leaves protection ABSENT -- which is exactly the state the caller must refuse
    rather than read as "the steps were fine".
    """

    report = _prepare(_Guest(free_bytes=TIGHT_FREE), clock=_Clock(step=TOTAL_BUDGET_S))
    blocking = report.blocking_steps()

    assert blocking, "an exhausted pass must not read as a prepared guest"
    assert {step.status for step in blocking} == {"skipped"}


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
    assert guest.ran("/var/lib/snapd/snaps")
    assert all(step.status == "unreachable" for step in report.steps)
    # The exception TYPE is recorded; its message is not, because a transport
    # error echoes the URL it was given.
    assert all(step.detail == "ConnectionError" for step in report.steps)
    # A guest nothing could be done to is NOT prepared, and saying so is the
    # caller's cue to refuse the episode: the measured alternative is a guest
    # whose root filesystem fills a few hundred seconds later.
    assert [step.name for step in report.blocking_steps()] == [
        "abort-in-flight-snap-changes",
        "hold-snap-auto-refresh",
        "clear-snapd-download-scratch",
    ]


def test_a_denied_sudo_is_recorded_and_the_remaining_steps_still_run() -> None:
    """One failing step must not abort the PASS: the others may still help.

    And a failed hold whose fallback succeeded is not a failure to report to the
    caller: the two are the same hold by two snapd-version spellings.
    """

    guest = _Guest(
        free_bytes=TIGHT_FREE,
        script={"snap refresh": CommandResult(1, "", "sudo: a password is required")},
    )
    report = _prepare(guest)

    assert _status(report, "hold-snap-auto-refresh") == "failed"
    assert _status(report, "hold-snap-auto-refresh-fallback") == "ok"
    assert _status(report, "clear-snapd-download-scratch") == "ok"
    assert guest.ran("/var/lib/snapd/snaps")
    assert report.blocking_steps() == ()


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
    assert _status(report, "clear-snapd-download-scratch") == "ok"


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
    assert "clear-snapd-download-scratch" in names
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


# ----------------------------------------------------------------------
# The snapd hold, retried while the control server cannot answer it
# ----------------------------------------------------------------------
#
# The measured case is t33v-t054-20261112-090503: ``abort`` ok, then
# ``hold-snap-auto-refresh`` AND its fallback ``unreachable / HTTPError``, then
# ``clear`` ok -- the control server was up on both sides, so the hold was lost
# to a non-2xx answer rather than a dead guest, and one transient answer
# refused a whole episode. These tests pin that a transient answer is re-sent
# on a bounded cadence, and that a hold which never lands still blocks.


def _server_error(status: int = 500) -> Exception:
    """What ``requests`` raises for a non-2xx answer: the measured ``HTTPError``."""

    import requests

    response = requests.Response()
    response.status_code = status
    return requests.HTTPError("500 Server Error for url: http://203.0.113.9/x", response=response)


class _Flaky:
    """Raise ``error`` for the first ``failures`` calls, then answer ``reply``."""

    def __init__(self, failures: int, error: Exception, reply: CommandResult) -> None:
        self.calls = 0
        self._failures = failures
        self._error = error
        self._reply = reply

    def __call__(self, _guest: _Guest) -> CommandResult:
        self.calls += 1
        if self.calls <= self._failures:
            raise self._error
        return self._reply


class _SleepingClock(_Clock):
    """A virtual clock whose ``sleep`` advances it, recording every wait."""

    def __init__(self, step: float = 0.0) -> None:
        super().__init__(step)
        self.slept: list[float] = []

    def sleep(self, seconds: float) -> None:
        self.slept.append(seconds)
        self.now += seconds


def _prepare_with(
    guest: _Guest, clock: _SleepingClock | None = None, **kwargs: Any
) -> guest_disk.GuestDiskReport:
    clock = clock or _SleepingClock()
    return prepare_guest_disk(guest, client_password="pw", clock=clock, sleep=clock.sleep, **kwargs)


def _step(report: guest_disk.GuestDiskReport, name: str) -> guest_disk.StepOutcome:
    matches = [step for step in report.steps if step.name == name]
    assert len(matches) == 1, f"{name}: {[s.name for s in report.steps]}"
    return matches[0]


def test_the_measured_t33v_hold_failure_is_retried_and_no_longer_refuses() -> None:
    """RED on the old code: one 500 on the hold refused the episode.

    The fallback is not needed once the retried hold lands, and the record
    says the hold took two sends -- a retried step must never read as clean.
    """

    flaky = _Flaky(1, _server_error(), CommandResult(0, "", ""))
    guest = _Guest(free_bytes=TIGHT_FREE, script={"snap refresh --hold": flaky})
    clock = _SleepingClock()

    report = _prepare_with(guest, clock)

    hold = _step(report, "hold-snap-auto-refresh")
    assert hold.status == "ok"
    assert hold.attempts == 2
    assert flaky.calls == 2
    assert report.blocking_steps() == ()
    assert not guest.ran("refresh.hold="), "the fallback is for a hold that did not land"
    assert clock.slept == [guest_disk.DEFAULT_RETRY_DELAYS_S[0]]
    assert _status(report, "clear-snapd-download-scratch") == "ok"


def test_a_hold_retried_on_the_injected_cadence_records_the_status_code() -> None:
    """The waits are the caller's cadence; the record names the HTTP status.

    ``HTTPError`` alone could not separate "the server answered 500" from
    "nothing answered", which is the question the t33v record left open. The
    status is value-free; the message (it echoes the URL) never lands.
    """

    flaky = _Flaky(3, _server_error(503), CommandResult(1, "", "still failing"))
    guest = _Guest(free_bytes=TIGHT_FREE, script={"snap refresh --hold": flaky})
    clock = _SleepingClock()

    report = _prepare_with(guest, clock, retry_delays=(0.5, 0.5, 1.0, 1.0, 2.0))

    hold = _step(report, "hold-snap-auto-refresh")
    # The fourth send answered with an exit status: a real answer, recorded as
    # ``failed`` -- and the fallback then ran, as for an old snapd.
    assert (hold.status, hold.attempts, hold.returncode) == ("failed", 4, 1)
    assert clock.slept[:3] == [0.5, 0.5, 1.0]
    assert guest.ran("refresh.hold=")
    unreachable = _Guest(free_bytes=TIGHT_FREE, script={"snap refresh --hold": _server_error(503)})
    report = _prepare_with(unreachable)
    hold = _step(report, "hold-snap-auto-refresh")
    assert hold.detail == "HTTPError status=503"
    assert "203.0.113.9" not in report.to_json_bytes().decode()


def test_a_hold_that_never_answers_is_bounded_and_still_blocks() -> None:
    """Retrying is not dropping: an unanswered hold stays a blocker.

    Both the hold and its fallback are re-sent at most ``HOLD_RETRY_ATTEMPTS``
    times, and the clear still runs afterwards -- a retry loop that starved
    the step that actually frees the space would be worse than no retry.
    """

    guest = _Guest(
        free_bytes=TIGHT_FREE,
        script={"snap refresh --hold": _server_error(), "refresh.hold=": _server_error()},
    )

    report = _prepare_with(guest)

    hold = _step(report, "hold-snap-auto-refresh")
    fallback = _step(report, "hold-snap-auto-refresh-fallback")
    assert (hold.status, hold.attempts) == ("unreachable", guest_disk.HOLD_RETRY_ATTEMPTS)
    assert (fallback.status, fallback.attempts) == (
        "unreachable",
        guest_disk.HOLD_RETRY_ATTEMPTS,
    )
    assert [step.name for step in report.blocking_steps()] == ["hold-snap-auto-refresh"]
    assert _status(report, "clear-snapd-download-scratch") == "ok"


def test_hold_retries_stop_before_they_could_starve_the_clear() -> None:
    """Each send costing a full command timeout ends the retrying early.

    The measured hold failure spent ~60 s per send (a guest-side timeout
    answered 500). Each send here costs a full ``COMMAND_TIMEOUT_S`` of virtual
    time, so the retries must stop while the budget can still fund BOTH the
    fallback and the clear -- the clear is what returns the space, and a
    retry that skipped it would trade a recorded hold failure for a guest
    that is never reclaimed at all.
    """

    clock = _SleepingClock()

    def slow_500(_guest: _Guest) -> CommandResult:
        clock.now += COMMAND_TIMEOUT_S
        raise _server_error()

    guest = _Guest(
        free_bytes=TIGHT_FREE,
        script={"snap refresh --hold": slow_500, "refresh.hold=": slow_500},
    )

    report = _prepare_with(guest, clock)

    hold = _step(report, "hold-snap-auto-refresh")
    fallback = _step(report, "hold-snap-auto-refresh-fallback")
    assert hold.status == fallback.status == "unreachable"
    # 270 s budget: hold 0-60, retry 60-120 (~210 s left funds it plus the
    # fallback and clear), then fallback 120-180 with no retry (~90 s left
    # funds only the clear), then the clear.
    assert (hold.attempts, fallback.attempts) == (2, 1)
    assert _status(report, "clear-snapd-download-scratch") == "ok"
    assert [s.name for s in report.blocking_steps()] == ["hold-snap-auto-refresh"]


def test_a_hold_that_answered_with_an_exit_status_is_not_retried() -> None:
    """An old snapd's unknown flag is an answer, so it goes to the fallback."""

    guest = _Guest(
        free_bytes=TIGHT_FREE,
        script={"snap refresh --hold": CommandResult(64, "", "unknown flag `hold'")},
    )
    clock = _SleepingClock()

    report = _prepare_with(guest, clock)

    assert _step(report, "hold-snap-auto-refresh").attempts == 1
    assert clock.slept == []
    assert _status(report, "hold-snap-auto-refresh-fallback") == "ok"
    assert report.blocking_steps() == ()


# ----------------------------------------------------------------------
# Growing the root filesystem into the volume -- upstream's volume_size gate
# ----------------------------------------------------------------------

# The campaign geometry: the AMI's filesystem inside a 120 GiB volume.
_FS_BYTES = 30_993_747_968
_VOLUME_120 = 128_849_018_880


def _growable_guest(*, grown: int = _VOLUME_120 - 600_000_000, **kwargs: Any) -> _Guest:
    """A guest whose geometry reads 29 GiB-in-120 GiB until the grow step runs."""

    state = {"grown": False}

    def geometry(_guest: _Guest) -> CommandResult:
        size = grown if state["grown"] else _FS_BYTES
        return CommandResult(0, f"{size}\n{_VOLUME_120}\n", "")

    def grow(_guest: _Guest) -> CommandResult:
        state["grown"] = True
        return CommandResult(0, "escalation=supplied\ngrow=grown fs=ext4 resize_rc=0", "")

    script: dict[str, Any] = {"--output=size": geometry, "growpart": grow}
    script.update(kwargs.pop("script", {}))
    kwargs.setdefault("free_bytes", TIGHT_FREE)
    return _Guest(script=script, **kwargs)


def test_no_volume_size_requested_means_no_grow_exactly_as_upstream() -> None:
    """``volume_size=None`` upstream never calls ``expand_guest_volume``."""

    guest = _growable_guest()

    report = _prepare_with(guest)

    assert not guest.ran("growpart")
    assert guest_disk.GROW_STEP_NAME not in [step.name for step in report.steps]
    payload = json.loads(report.to_json_bytes())
    assert payload["root_grow_requested"] is False
    assert payload["filesystem_bytes_after"] is None


def test_a_requested_volume_size_grows_the_filesystem_after_the_reclamation() -> None:
    """RED on the old code: the 120 GiB volume kept a 29 GiB filesystem.

    The grow runs AFTER abort/hold/clear (they stop the fill; apt must not
    delay them) and is measured again, so the record proves what it achieved.
    """

    guest = _growable_guest()

    report = _prepare_with(guest, requested_volume_gb=120)

    names = [step.name for step in report.steps]
    assert names.index(guest_disk.GROW_STEP_NAME) > names.index("clear-snapd-download-scratch")
    assert names.index("measure-geometry-after") > names.index(guest_disk.GROW_STEP_NAME)
    assert _status(report, guest_disk.GROW_STEP_NAME) == "ok"
    assert report.filesystem_bytes == _FS_BYTES
    assert report.filesystem_bytes_after == _VOLUME_120 - 600_000_000
    assert report.blocking_steps() == ()
    payload = json.loads(report.to_json_bytes())
    assert payload["root_grow_requested"] is True
    # Privileged through the same ladder as every other step.
    grow_command = next(command for command in guest.commands if "growpart" in command)
    assert "sudo -n bash -c" in grow_command


def test_the_grow_step_runs_on_its_own_ceiling_outside_the_reclamation_budget() -> None:
    """A reclamation that used its whole budget must not starve the grow."""

    guest = _growable_guest()
    clock = _SleepingClock(step=TOTAL_BUDGET_S / 6)

    report = _prepare_with(guest, clock, requested_volume_gb=120)

    assert _status(report, guest_disk.GROW_STEP_NAME) == "ok"
    grow_index = next(i for i, command in enumerate(guest.commands) if "growpart" in command)
    assert COMMAND_TIMEOUT_S < guest.timeouts[grow_index] <= guest_disk.GROW_TIMEOUT_S


def test_a_filesystem_that_already_spans_the_volume_is_not_grown_again() -> None:
    """Idempotent before anything is sent: nothing to grow is a skip, recorded."""

    guest = _Guest(
        free_bytes=TIGHT_FREE,
        script={"--output=size": CommandResult(0, f"{_VOLUME_120 - 10**8}\n{_VOLUME_120}\n", "")},
    )

    report = _prepare_with(guest, requested_volume_gb=120)

    assert not guest.ran("growpart")
    grow = _step(report, guest_disk.GROW_STEP_NAME)
    assert (grow.status, grow.attempts) == ("skipped", 0)
    assert report.blocking_steps() == ()


def test_a_roomy_guest_is_still_grown_when_a_volume_size_was_requested() -> None:
    """Upstream grows on the request, not on free space; so does this."""

    guest = _growable_guest(free_bytes=ROOMY_FREE)

    report = _prepare_with(guest, requested_volume_gb=120)

    assert report.reclamation_attempted is False
    assert not guest.ran("snap refresh")
    assert _status(report, guest_disk.GROW_STEP_NAME) == "ok"
    assert _status(report, "measure-free-after") == "ok"


@pytest.mark.parametrize(
    "reply",
    [
        CommandResult(4, "escalation=supplied\ngrow=no-growpart install=failed", ""),
        ConnectionError("refused"),
    ],
)
def test_a_failed_grow_is_recorded_but_never_refuses_the_episode(reply: Any) -> None:
    """Advisory: upstream tolerates a failed growpart, and a held guest runs.

    The gap stays visible -- the step's own status plus the after-geometry
    still reading the AMI's filesystem size.
    """

    guest = _growable_guest(script={"growpart": reply})

    report = _prepare_with(guest, requested_volume_gb=120)

    assert _status(report, guest_disk.GROW_STEP_NAME) in {"failed", "unreachable"}
    assert report.blocking_steps() == ()
    assert report.filesystem_bytes_after == _FS_BYTES


def _fake_root_guest(tmp_path: Path, *, has_growpart: bool, growpart_says: str = "") -> Path:
    """A PATH with stub disk tools, logging each call to ``calls.log``."""

    import shutil

    bin_dir = tmp_path / "bin"
    bin_dir.mkdir()
    log = tmp_path / "calls.log"
    for tool in ("readlink", "head", "cat", "chmod"):
        found = shutil.which(tool, path="/usr/bin:/bin")
        assert found, tool
        (bin_dir / tool).symlink_to(found)

    def stub(name: str, body: str) -> None:
        (bin_dir / name).write_text(f'#!/bin/bash\necho "{name} $*" >> {log}\n{body}\n')
        (bin_dir / name).chmod(0o755)

    stub(
        "findmnt",
        'case "$*" in *SOURCE*) echo /dev/nvme0n1p1;; *FSTYPE*) echo ext4;; esac',
    )
    stub("lsblk", 'case "$*" in *PKNAME*) echo nvme0n1;; *PARTN*) echo 1;; esac')
    stub("resize2fs", "exit 0")
    growpart_body = f'echo "{growpart_says}"; exit 1' if "NOCHANGE" in growpart_says else "exit 0"
    if has_growpart:
        stub("growpart", growpart_body)
    # ``apt-get install`` makes ``growpart`` appear, as the real package does.
    stub(
        "apt-get",
        'case "$*" in *install*cloud-guest-utils*) '
        f'printf "#!/bin/bash\\necho growpart \\$* >> {log}\\nexit 0\\n" > {bin_dir}/growpart; '
        f"chmod 755 {bin_dir}/growpart;; esac",
    )
    return bin_dir


def _run_grow(bin_dir: Path) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        ["/bin/bash", "-c", guest_disk._GROW_ROOT_FRAGMENT],
        capture_output=True,
        text=True,
        env={"PATH": str(bin_dir)},
        check=False,
    )


def test_the_grow_fragment_grows_partition_then_filesystem_through_a_real_shell(
    tmp_path: Path,
) -> None:
    """Upstream's discovery and resize, executed: disk, partition, device."""

    bin_dir = _fake_root_guest(tmp_path, has_growpart=True)

    completed = _run_grow(bin_dir)

    assert completed.returncode == 0, completed.stdout + completed.stderr
    assert "grow=grown fs=ext4 resize_rc=0 growpart_installed=0" in completed.stdout
    calls = (tmp_path / "calls.log").read_text().splitlines()
    assert "growpart /dev/nvme0n1 1" in calls
    assert "resize2fs /dev/nvme0n1p1" in calls
    assert calls.index("growpart /dev/nvme0n1 1") < calls.index("resize2fs /dev/nvme0n1p1")
    assert not any(call.startswith("apt-get") for call in calls)


def test_the_grow_fragment_installs_growpart_the_way_upstream_does(tmp_path: Path) -> None:
    """The AMI ships no ``growpart``; ``cloud-guest-utils`` is upstream's fix."""

    bin_dir = _fake_root_guest(tmp_path, has_growpart=False)

    completed = _run_grow(bin_dir)

    assert completed.returncode == 0, completed.stdout + completed.stderr
    assert "growpart_installed=1" in completed.stdout
    calls = (tmp_path / "calls.log").read_text()
    assert "cloud-guest-utils" in calls
    assert "DPkg::Lock::Timeout=60" in calls
    assert "growpart /dev/nvme0n1 1" in calls


def test_the_grow_fragment_is_idempotent_on_an_already_grown_partition(tmp_path: Path) -> None:
    """``growpart`` NOCHANGE is success, so a second run changes nothing."""

    bin_dir = _fake_root_guest(
        tmp_path,
        has_growpart=True,
        growpart_says="NOCHANGE: partition 1 is size 251656159. it cannot be grown",
    )

    completed = _run_grow(bin_dir)

    assert completed.returncode == 0, completed.stdout + completed.stderr
    assert "grow=nochange fs=ext4 resize_rc=0" in completed.stdout


def test_the_grow_fragment_reports_which_link_broke(tmp_path: Path) -> None:
    """No ``growpart`` and an install that did not provide one: exit 4, named."""

    bin_dir = _fake_root_guest(tmp_path, has_growpart=False)
    (bin_dir / "apt-get").write_text("#!/bin/bash\nexit 100\n")

    completed = _run_grow(bin_dir)

    assert completed.returncode == 4
    assert "grow=no-growpart install=failed" in completed.stdout


def test_the_grow_fragment_validates_the_filesystem_before_touching_the_partition(
    tmp_path: Path,
) -> None:
    """N1: an unsupported fs exits 6 with neither apt nor ``growpart`` run.

    v2.1 validates the resize handler first; RED on the previous fragment,
    which ran ``growpart`` and only then found no resize arm for the fstype.
    """

    bin_dir = _fake_root_guest(tmp_path, has_growpart=True)
    (bin_dir / "findmnt").write_text(
        '#!/bin/bash\ncase "$*" in *SOURCE*) echo /dev/nvme0n1p1;; *FSTYPE*) echo vfat;; esac\n'
    )

    completed = _run_grow(bin_dir)

    assert completed.returncode == 6
    assert "grow=unsupported-fs fs=vfat" in completed.stdout
    log = tmp_path / "calls.log"
    calls = log.read_text() if log.exists() else ""
    assert "growpart" not in calls
    assert "apt-get" not in calls


def test_the_grow_fragment_refuses_when_the_resize_tool_is_missing(tmp_path: Path) -> None:
    """v2.1's ``tool_available "$RESIZE_TOOL"`` check, also before the partition."""

    bin_dir = _fake_root_guest(tmp_path, has_growpart=True)
    (bin_dir / "resize2fs").unlink()

    completed = _run_grow(bin_dir)

    assert completed.returncode == 6
    assert "grow=no-resize-tool tool=resize2fs" in completed.stdout
    log = tmp_path / "calls.log"
    assert "growpart" not in (log.read_text() if log.exists() else "")


# ----------------------------------------------------------------------
# The grow's escalation: authenticate first, run the fragment once
# ----------------------------------------------------------------------


def _counting_fragment(counter: Path, *, rc: int, says: str) -> str:
    """A fragment that records each run, then exits like a failed grow."""

    return f"echo run >> {guest_disk._shell_quote(str(counter))}; echo {says}; exit {rc}"


def _fake_sudo_guest(tmp_path: Path, *, password: str, nopasswd: bool = False) -> Path:
    bin_dir = tmp_path / "bin"
    bin_dir.mkdir()
    guard = tmp_path / "guard"
    guard.mkdir()
    _install_fake_sudo(bin_dir, password=password, marker=guard, nopasswd=nopasswd)
    return bin_dir


def test_a_failed_grow_is_not_replayed_or_reported_as_an_auth_failure(tmp_path: Path) -> None:
    """F3/Q2, reproduced: the supplied value is an upstream default.

    That opens the ladder's third rung, and the shared ladder re-ran a failed
    grow under it -- whose ``sudo: 1 incorrect password attempt`` then replaced
    the grow's own reason in the record. Authenticating first runs the fragment
    once, under the rung that validated, and keeps ITS output.
    """

    first, second = guest_disk._UPSTREAM_DEFAULT_PASSWORDS
    bin_dir = _fake_sudo_guest(tmp_path, password=second)
    counter = tmp_path / "runs"
    fragment = _counting_fragment(counter, rc=4, says="grow=no-growpart install=failed")

    completed = _run_in_fake_guest(
        guest_disk.authenticated_escalation_script(fragment, second), bin_dir
    )

    assert completed.returncode == 4
    assert counter.read_text().splitlines() == ["run"]
    assert "grow=no-growpart install=failed" in completed.stdout
    assert "incorrect password" not in completed.stdout
    assert completed.stdout.splitlines()[0] == "escalation=agentless>supplied"
    assert second not in completed.stdout and first not in completed.stdout

    # RED control: the shared ladder, same guest and fragment, masks the reason.
    counter.unlink()
    shared = _run_in_fake_guest(guest_disk.escalation_script(fragment, second), bin_dir)
    assert "grow=no-growpart" not in shared.stdout
    assert "incorrect password" in shared.stdout


def test_a_failed_grow_runs_once_on_a_nopasswd_guest(tmp_path: Path) -> None:
    """F3(b): the shared ladder ran a failed ``growpart`` twice here."""

    bin_dir = _fake_sudo_guest(tmp_path, password="unused", nopasswd=True)
    counter = tmp_path / "runs"
    fragment = _counting_fragment(counter, rc=5, says="grow=growpart-failed rc=2")

    completed = _run_in_fake_guest(
        guest_disk.authenticated_escalation_script(fragment, "password"), bin_dir
    )

    assert completed.returncode == 5
    assert counter.read_text().splitlines() == ["run"]
    assert completed.stdout.splitlines()[0] == "escalation=agentless"
    assert "grow=growpart-failed rc=2" in completed.stdout


def test_the_grow_escalation_falls_through_to_a_default_that_validates(tmp_path: Path) -> None:
    """The ladder's reach is unchanged: the measured stale-default case still opens."""

    first, second = guest_disk._UPSTREAM_DEFAULT_PASSWORDS
    bin_dir = _fake_sudo_guest(tmp_path, password=first)
    counter = tmp_path / "runs"

    completed = _run_in_fake_guest(
        guest_disk.authenticated_escalation_script(
            _counting_fragment(counter, rc=0, says="grow=grown"), second
        ),
        bin_dir,
    )

    assert completed.returncode == 0
    assert counter.read_text().splitlines() == ["run"]
    assert completed.stdout.splitlines()[0] == "escalation=agentless>supplied>upstream-default"


def test_the_grow_never_runs_when_no_candidate_authenticates(tmp_path: Path) -> None:
    """An auth failure reads as one, and nothing privileged was attempted."""

    bin_dir = _fake_sudo_guest(tmp_path, password="none-of-them")
    counter = tmp_path / "runs"

    completed = _run_in_fake_guest(
        guest_disk.authenticated_escalation_script(
            _counting_fragment(counter, rc=0, says="grow=grown"), "password"
        ),
        bin_dir,
    )

    assert completed.returncode == 1
    assert not counter.exists()
    assert "no candidate authenticated; fragment not run" in completed.stdout
    assert "escalation=agentless>supplied>upstream-default" in completed.stdout


def test_each_grow_candidate_appears_once_and_only_on_stdin() -> None:
    first, second = guest_disk._UPSTREAM_DEFAULT_PASSWORDS
    script = guest_disk.authenticated_escalation_script("echo INNER", second)

    assert script.count(f"'{second}'") == 1
    assert script.count(f"'{first}'") == 1
    # Validation only ever pipes into ``sudo -S ... -v``; the one run re-feeds
    # the validated value with ``-k`` so sudo, not the fragment, consumes it.
    assert script.count("| sudo -S -p '' -v") == 2
    assert script.count("| sudo -S -k -p '' bash -c") == 1
    assert "| sudo -n" not in script


def test_the_grow_step_uses_the_auth_first_escalation() -> None:
    guest = _growable_guest()

    _prepare_with(guest, requested_volume_gb=120)

    grow_command = next(command for command in guest.commands if "growpart" in command)
    assert "sudo -n -v" in grow_command
    assert "fragment not run" in grow_command
    # The reclamation steps keep the shared ladder (pre-existing behaviour).
    hold = next(command for command in guest.commands if "--hold=forever" in command)
    assert "sudo -n -v" not in hold


# ----------------------------------------------------------------------
# What the record says about the grow: source, target, duration, skip
# ----------------------------------------------------------------------


def test_the_record_carries_the_requested_size_its_source_and_the_v21_target() -> None:
    """F1(c)/F2: which size, from where, and did the fs reach 95 % of it."""

    guest = _growable_guest()

    report = _prepare_with(guest, requested_volume_gb=120, volume_source="override")

    payload = json.loads(report.to_json_bytes())
    assert payload["requested_volume_gb"] == 120
    assert payload["volume_size_source"] == "override"
    # 128,249,018,880 B >= 0.95 x 120 GiB (122,406,567,936 B).
    assert payload["grow_target_met"] is True


def test_a_short_grow_is_identifiable_by_grow_target_met() -> None:
    """v2.1 raises under 95 %; this records False and still runs the episode."""

    guest = _growable_guest(grown=_FS_BYTES + 10 * 1024**3)

    report = _prepare_with(guest, requested_volume_gb=120, volume_source="task")

    payload = json.loads(report.to_json_bytes())
    assert payload["grow_target_met"] is False
    assert payload["volume_size_source"] == "task"
    assert report.blocking_steps() == ()


def test_grow_target_met_is_none_when_nothing_was_requested() -> None:
    payload = json.loads(_prepare_with(_growable_guest()).to_json_bytes())

    assert payload["requested_volume_gb"] is None
    assert payload["volume_size_source"] is None
    assert payload["grow_target_met"] is None


def test_an_already_spanning_filesystem_is_judged_as_it_stood() -> None:
    """No grow needed: the 95 % test reads the starting filesystem size."""

    guest = _Guest(
        free_bytes=TIGHT_FREE,
        script={"--output=size": CommandResult(0, f"{_VOLUME_120 - 10**8}\n{_VOLUME_120}\n", "")},
    )

    report = _prepare_with(guest, requested_volume_gb=120, volume_source="task")

    assert report.grow_target_met is True


def test_the_grow_step_records_its_duration_and_no_other_step_does() -> None:
    """F4: the 180 s cap is unmeasured; the record is what will measure it."""

    guest = _growable_guest()

    payload = json.loads(
        _prepare_with(guest, _SleepingClock(step=1.0), requested_volume_gb=120).to_json_bytes()
    )

    steps = {step["name"]: step for step in payload["steps"]}
    assert steps[guest_disk.GROW_STEP_NAME]["duration_s"] > 0
    assert all(
        "duration_s" not in step for name, step in steps.items() if name != "grow-root-filesystem"
    )


def test_the_grow_is_skipped_when_the_episode_will_be_refused() -> None:
    """F5/Q3: a blocked reclamation is refused; growing first only delays it."""

    guest = _growable_guest(
        script={
            "snap refresh --hold": ConnectionError("refused"),
            "refresh.hold=": ConnectionError("refused"),
        }
    )

    report = _prepare_with(guest, requested_volume_gb=120, volume_source="task")

    assert report.blocking_steps() != ()
    assert not guest.ran("growpart")
    grow = _step(report, guest_disk.GROW_STEP_NAME)
    assert (grow.status, grow.attempts) == ("skipped", 0)
    assert "reclamation blocked" in grow.detail
