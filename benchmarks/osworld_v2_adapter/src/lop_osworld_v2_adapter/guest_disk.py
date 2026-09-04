"""Reclaim the guest's root filesystem before the episode's first observation.

WHY THIS EXISTS -- and the wrong answer that came first, so nobody re-derives it.

Every OSWorld episode died at roughly the same WALL-CLOCK time: 7 of 8 runs
first failed in a 424-466s window, at 16-32 steps, on both ``t3.xlarge`` and
``m5.xlarge``. Probing the guest's own control server showed the root
filesystem going 93% used / 2.2 GB free (stable from t+54s to t+342s) -> 95% at
t+363s -> **100% used, 0 bytes free at t+383s**, with the first
``ObservationPhaseError: environment returned no screenshot frame`` at t+424s. A
disk at 0 bytes cannot write a screenshot, so the observation failure is the
symptom and the full disk is the cause.

**THE FIRST DIAGNOSIS BLAMED OSWORLD'S x11grab SCREEN RECORDER. THAT WAS WRONG.**
``pgrep -af ffmpeg`` on the failing guest showed **no ffmpeg process at all**;
the only match was the probe's own ``pgrep`` command line, which is what made
the theory look confirmed from outside. The measured consumer is **snapd**:

* ``/var`` is 15G of a 29G disk, and ``/var/lib/snapd/cache`` **alone is 9.7 GB**
* ``snap changes`` shows ``Auto-refresh 9 snaps`` and ``Pre-download novnc``,
  both fired at boot
* the AMI ships ~93% full, so a few GB of snap downloads exhausts it

That is exactly why the failure looked like a clock rather than a workload:
snapd's auto-refresh starts at boot and downloads at its own pace, entirely
independent of what the agent is doing. It is also why ``AWS_INSTANCE_TYPE``
changed nothing, and why ``AWS_ROOT_VOLUME_SIZE`` (0.46.11) helped but did not
fix it -- a 100 GiB volume moved the first failure from t+424s to t+1936s, yet
the root PARTITION stays 29.5G with ~70 GiB unallocated, because the AMI carries
no ``growpart`` and ``apt-get install cloud-guest-utils`` cannot run on a disk
with no free space to download into.

WHAT THIS MODULE DOES, AND THE LINE IT WILL NOT CROSS. This is guest
ENVIRONMENT PREPARATION, not benchmark semantics. It clears a package manager's
**download scratch** and stops that download from restarting; it does not
uninstall an application, change a task, alter scoring, or touch anything the
model observes. ``/var/lib/snapd/cache`` is documented as "the working cache,
used to minimise download size and speed up refreshes" -- deleting it costs a
re-download and nothing else. Nothing here uninstalls a snap or stops
``snapd`` itself, because a task may legitimately launch a snap-packaged
application and removing one would change the benchmark.

FAIL SOFT, ALWAYS. Every step is best-effort and every failure mode -- a missing
binary, a denied sudo, an unreachable control server, a wedged guest -- is
recorded and stepped over. An episode that would otherwise have succeeded must
never be destroyed by a hygiene step, so ``prepare_guest_disk`` raises nothing
and the caller ignores nothing: it writes the report either way.

CONDITIONAL, AND WHY THE THRESHOLD IS WHERE IT IS. Free space is measured on
every episode (that measurement is the point -- see "observability" below), but
the reclamation only runs when the guest has less than
``RECLAIM_BELOW_FREE_BYTES`` free. The threshold is set ABOVE the largest
measured consumer: snapd's cache was 9.7 GB, so a guest with more than 12 GiB
free can absorb snapd's entire measured appetite and still have room for the
episode's own writes, and touching it would be housekeeping nobody needs. Below
that, the guest is on the trajectory the measurements above describe.

OBSERVABILITY. "The guest had N MB free at the start" is the single fact needed
to interpret a later environment failure, so the report is written to the
episode's own cache root as ``guest-preparation.json`` -- beside the artifact
root, under the durable run root, in the same episode-owned directory the
adapter already routes upstream's writes to. It deliberately does NOT ride on
the observation: ``AckResult`` carries no fields, ``ObservationPayload`` carries
no metadata, and ``Observation.metadata`` feeds ``observation_content_id``, so
putting guest disk state there would make a content-addressed observation id a
function of the guest's filesystem rather than of what the model saw.

WHAT IS DELIBERATELY NOT DONE: growing the partition. See
``docs/benchmarks/osworld_2/README.md`` for the evidence; briefly, ``growpart``
is absent, the in-place ``sfdisk`` alternative rewrites the root partition table
and a wrong start sector destroys the guest -- a hygiene step that can fail HARD
is precisely what this module must not contain -- and once snapd is held and its
cache cleared the 29.5G partition has ample room for an episode. The disk vs
partition geometry is REPORTED instead, read-only, because that is the number
telling the next reader whether ``AWS_ROOT_VOLUME_SIZE`` bought anything.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Callable, Literal, Sequence

# Reclaim only below this much free space. Set above the largest measured
# consumer (``/var/lib/snapd/cache`` at 9.7 GB) plus room for the episode's own
# writes, so a guest that can already absorb a full auto-refresh is left alone.
RECLAIM_BELOW_FREE_BYTES = 12 * 1024**3

# Per-command and whole-preparation ceilings. The runner allows 900s for the
# whole ``reset_start`` (scripts/run_episode.py), most of which is instance
# readiness and upstream's own task setup, so hygiene gets a small fixed slice
# and a wedged guest cannot eat the reset budget: once the budget is spent the
# remaining steps are recorded as skipped rather than attempted.
COMMAND_TIMEOUT_S = 60.0
TOTAL_BUDGET_S = 180.0

# The snapd cache is pure download scratch (snapcraft "Data locations": "the
# working cache ... used to minimise download size and speed-up refreshes").
# The contents are removed, never the directory: snapd recreates files in it but
# does not recreate the directory itself on every path.
_SNAPD_CACHE = "/var/lib/snapd/cache"

# A hold far enough out that no episode can outlive it. Used only as the
# fallback for snapd older than 2.58 (which has no ``snap refresh --hold``);
# newer snapd caps ``refresh.hold`` at 90 days, which is still four orders of
# magnitude longer than the 2-hour lease, so the cap is harmless here.
_FALLBACK_HOLD_UNTIL = "2100-01-01T00:00:00Z"

StepStatus = Literal["ok", "failed", "unreachable", "skipped"]


@dataclass(frozen=True)
class CommandResult:
    """One command's result as the guest's control server reported it."""

    returncode: int
    stdout: str
    stderr: str


#: Runs ONE command in the guest and returns its result, or raises if the guest
#: could not be reached at all. Injected so the whole module is testable against
#: a stub and the AWS provider owns the HTTP details.
GuestCommand = Callable[[Sequence[str], float], CommandResult]

#: Monotonic clock, injected so tests need no wall time (AGENTS.md "Timing").
Clock = Callable[[], float]


@dataclass(frozen=True)
class StepOutcome:
    """What one preparation step did, in terms a later reader can act on.

    ``detail`` is bounded and carries the guest's own stderr/stdout tail. The
    COMMAND is never recorded: holding and clearing run through
    ``echo <password> | sudo -S`` (upstream's own pattern, setup.py:609), so the
    verbatim argv would put the client password in a file on the operator's
    disk for no diagnostic gain.
    """

    name: str
    status: StepStatus
    returncode: int | None = None
    detail: str = ""

    def to_json(self) -> dict[str, object]:
        return {
            "name": self.name,
            "status": self.status,
            "returncode": self.returncode,
            "detail": self.detail,
        }


@dataclass(frozen=True)
class GuestDiskReport:
    """Free space before and after, the geometry, and every step's outcome.

    ``free_bytes_before`` is the load-bearing field: an episode that later fails
    with "environment returned no screenshot frame" is read completely
    differently depending on whether the guest started with 2 GB or 12 GB free,
    and that question must not depend on anyone having probed by hand.
    """

    free_bytes_before: int | None
    free_bytes_after: int | None
    filesystem_bytes: int | None
    disk_bytes: int | None
    threshold_bytes: int
    # ATTEMPTED, not achieved. A report that said "reclaimed" while every step
    # came back unreachable would be a false statement sealed beside the
    # episode's evidence; whether it WORKED is the steps plus the two free-space
    # measurements, which cannot claim something that did not happen.
    reclamation_attempted: bool
    reason: str
    steps: tuple[StepOutcome, ...] = ()

    def to_json(self) -> dict[str, object]:
        return {
            "free_bytes_before": self.free_bytes_before,
            "free_bytes_after": self.free_bytes_after,
            "filesystem_bytes": self.filesystem_bytes,
            # The whole block device. When it exceeds ``filesystem_bytes`` the
            # root partition was never grown into an AWS_ROOT_VOLUME_SIZE
            # override -- the measured 29.5G-in-100GiB case.
            "disk_bytes": self.disk_bytes,
            "threshold_bytes": self.threshold_bytes,
            "reclamation_attempted": self.reclamation_attempted,
            "reason": self.reason,
            "steps": [step.to_json() for step in self.steps],
        }

    def to_json_bytes(self) -> bytes:
        return json.dumps(self.to_json(), indent=2, sort_keys=True).encode("utf-8")


def _bash(script: str) -> list[str]:
    # ``shell: false`` on the guest endpoint means the server execs argv
    # directly, so a pipeline has to be an explicit ``bash -c``. This is the
    # same shape upstream's own SetupController uses (setup.py:609).
    return ["bash", "-c", script]


def _first_int(text: str) -> int | None:
    """The first bare integer in a command's output, or None.

    ``df --output`` prints a header line; a busy guest may prepend a warning.
    Scanning for the first integer token is what makes the parse survive both
    without a format assumption that a different coreutils would break.
    """

    for line in text.splitlines():
        token = line.strip()
        if token.isdigit():
            return int(token)
    return None


class _Session:
    """One preparation pass: runs steps, records them, and honours the budget."""

    def __init__(self, run: GuestCommand, clock: Clock) -> None:
        self._run = run
        self._clock = clock
        self._deadline = clock() + TOTAL_BUDGET_S
        self.steps: list[StepOutcome] = []

    def step(self, name: str, script: str) -> CommandResult | None:
        """Run one step, record it, and return its result or None on failure.

        Never raises: a step that cannot run is a recorded fact, not an
        episode-ending error. That is the whole fail-soft contract.
        """

        remaining = self._deadline - self._clock()
        if remaining <= 0:
            self.steps.append(StepOutcome(name=name, status="skipped", detail="budget exhausted"))
            return None
        try:
            result = self._run(_bash(script), min(COMMAND_TIMEOUT_S, remaining))
        except Exception as error:
            # A transport failure (control server down, timeout, malformed
            # response) is indistinguishable from a guest that never answered,
            # and both mean the same thing to a reader: the guest was not
            # prepared. The TYPE is recorded; the message is not, because a
            # requests exception echoes the URL and query it was given.
            self.steps.append(
                StepOutcome(name=name, status="unreachable", detail=type(error).__name__)
            )
            return None
        detail = (result.stderr or result.stdout or "").strip()[:200]
        self.steps.append(
            StepOutcome(
                name=name,
                status="ok" if result.returncode == 0 else "failed",
                returncode=result.returncode,
                detail=detail,
            )
        )
        return result


def _measure_free_bytes(session: _Session, name: str) -> int | None:
    result = session.step(name, "df -B1 --output=avail / | tail -1")
    if result is None or result.returncode != 0:
        return None
    return _first_int(result.stdout)


def prepare_guest_disk(
    run: GuestCommand,
    *,
    client_password: str,
    clock: Clock,
) -> GuestDiskReport:
    """Measure the guest's root filesystem and, if it is tight, reclaim it.

    Returns a report in every case, including the cases where nothing could be
    measured or nothing could be run. It raises nothing by construction: see the
    module docstring for why an unreachable guest must not cost an episode that
    would otherwise have worked.
    """

    session = _Session(run, clock)
    free_before = _measure_free_bytes(session, "measure-free-before")
    filesystem_bytes = None
    disk_bytes = None

    size_result = session.step("measure-geometry", _GEOMETRY_SCRIPT)
    if size_result is not None and size_result.returncode == 0:
        numbers = [int(line) for line in size_result.stdout.split() if line.strip().isdigit()]
        # The script prints filesystem size then whole-disk size, so a partial
        # answer (an image without lsblk) still yields the first number.
        if numbers:
            filesystem_bytes = numbers[0]
        if len(numbers) > 1:
            disk_bytes = numbers[1]

    if free_before is not None and free_before >= RECLAIM_BELOW_FREE_BYTES:
        return GuestDiskReport(
            free_bytes_before=free_before,
            free_bytes_after=free_before,
            filesystem_bytes=filesystem_bytes,
            disk_bytes=disk_bytes,
            threshold_bytes=RECLAIM_BELOW_FREE_BYTES,
            reclamation_attempted=False,
            reason="above-threshold",
            steps=tuple(session.steps),
        )

    # An unmeasurable guest is reclaimed anyway. The steps are safe and
    # reversible; the failure they prevent destroys a paid episode. Choosing to
    # skip on a failed measurement would make the protection absent exactly when
    # the guest is least healthy.
    reason = "below-threshold" if free_before is not None else "unmeasured"
    _reclaim(session, client_password)
    free_after = _measure_free_bytes(session, "measure-free-after")
    return GuestDiskReport(
        free_bytes_before=free_before,
        free_bytes_after=free_after,
        filesystem_bytes=filesystem_bytes,
        disk_bytes=disk_bytes,
        threshold_bytes=RECLAIM_BELOW_FREE_BYTES,
        reclamation_attempted=True,
        reason=reason,
        steps=tuple(session.steps),
    )


# Filesystem size, then the size of the whole block device the root filesystem
# sits on. Read-only: nothing here alters the partition table (see the module
# docstring for why growing it is deliberately out of scope). Each half is
# independently allowed to fail, so an image without ``lsblk`` still reports the
# filesystem size rather than nothing.
_GEOMETRY_SCRIPT = (
    "df -B1 --output=size / | tail -1; "
    'source=$(findmnt -no SOURCE / 2>/dev/null) && parent=$(lsblk -no PKNAME "$source" '
    '2>/dev/null | head -1) && lsblk -bdno SIZE "/dev/$parent" 2>/dev/null'
)


def _reclaim(session: _Session, client_password: str) -> None:
    """Stop snapd re-filling the disk, then delete what it already downloaded.

    ORDER IS LOAD-BEARING. Holding first stops a NEW auto-refresh from starting;
    aborting second stops the one that ``snap changes`` showed already running
    at boot -- a hold does not touch a change that is already in flight;
    clearing last means the abort has stopped writing before the delete runs.
    Clearing first would race a live download and reclaim nothing.
    """

    sudo = f"echo {_shell_quote(client_password)} | sudo -S"
    # ``snap refresh --hold`` needs snapd 2.58+. On anything older it exits
    # non-zero with an unknown-flag error, which is recorded and then covered by
    # the ``refresh.hold`` fallback below -- the pre-2.58 way of saying the same
    # thing. Running the fallback unconditionally would be a second write to the
    # same setting on every modern guest, so it is conditional on the first
    # failing.
    held = session.step("hold-snap-auto-refresh", f"{sudo} snap refresh --hold=forever 2>&1")
    if held is None or held.returncode != 0:
        session.step(
            "hold-snap-auto-refresh-fallback",
            f"{sudo} snap set system refresh.hold={_FALLBACK_HOLD_UNTIL} 2>&1",
        )

    # ``snap changes`` lists in-flight changes as ``Doing``; the boot-time
    # "Auto-refresh 9 snaps" is one of them. Aborting cancels the download and
    # leaves every INSTALLED snap exactly as it was -- it is not an uninstall,
    # which is the line this module does not cross. No ``Doing`` row means the
    # awk prints nothing and xargs runs nothing, so the step is a clean no-op.
    session.step(
        "abort-in-flight-snap-changes",
        f"{sudo} snap changes 2>/dev/null | awk '$2==\"Doing\"{{print $1}}' "
        f"| xargs -r -n1 {sudo} snap abort 2>&1 || true",
    )

    # The contents, not the directory: snapd expects the directory to exist.
    session.step("clear-snapd-cache", f'{sudo} rm -rf -- "{_SNAPD_CACHE}"/* 2>&1')


def _shell_quote(value: str) -> str:
    """POSIX single-quote one value for the ``bash -c`` script.

    The client password is operator-supplied infra and reaches the guest through
    a shell pipeline (the only way to feed ``sudo -S`` through an endpoint with
    no stdin). Quoting it here means a password containing a space, a quote, or
    a ``$`` cannot terminate the command or expand into something else.
    """

    return "'" + value.replace("'", "'\"'\"'") + "'"
