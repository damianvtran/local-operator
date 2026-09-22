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

* ``/var`` is 15G of a 29G disk, and what fills it is snapd's **delta
  downloads**. ``/var/lib/snapd/snaps`` measured 7.9 GB -> 9.9 GB -> 10.6 GB in
  about 50 s while free space went 2.2 GB -> 0.2 GB -> **0 bytes**, and the
  files growing there were ``<name>_<rev>.snap.xdelta3-<old>-to-<new>.partial``
  (``kf6-core24_64`` 209 MB -> 1.17 GB, ``audacity_1239`` 128 MB -> 399 MB):
  snapd's downloader writes a pending refresh revision -- a delta, most of the
  time -- beside the installed ones and renames it when the download completes.
* ``/var/lib/snapd/cache``, the directory this module used to clear, measured
  **4096 bytes** on this image. It IS documented as snapd's download cache
  ("the working cache ... used to minimise download size and speed-up
  refreshes"), which is exactly why it looks like the right target and is not:
  this snapd streams refresh downloads, deltas included, straight into the
  ``snaps`` directory as ``*.partial`` files, and leaves the cache directory
  empty. Clearing it reclaimed nothing, and four paid episodes died on the
  trajectory it was supposed to interrupt.
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
model observes. Both scratch locations cost a re-download and nothing else:
the cache directory's contents, and the ``*.partial`` files -- a download that
never completed -- beside the installed revisions. The installed ``.snap``
revisions themselves are NEVER touched, and nothing here uninstalls a snap or
stops ``snapd`` itself, because a task may legitimately launch a snap-packaged
application and removing one would change the benchmark. (A ``.snap`` file in
that directory is an installed revision, mounted through a loop device; there
is no version of deleting it that is housekeeping.)

FAIL SOFT ON THE EPISODE, LOUD ABOUT AN UNPREPARED GUEST. ``prepare_guest_disk``
raises nothing: a missing binary, a denied sudo, an unreachable control server
and a wedged guest are each recorded as a step outcome and stepped over. What
the CALLER may not do is walk into the guest anyway -- that is what the four
dead episodes were. Every privileged step came back ``sudo: no password was
provided`` / ``sudo: 1 incorrect password attempt`` because the
``OSWORLD_CLIENT_PASSWORD`` the campaign passed was rejected by this image; the
report recorded it, the adapter read past it, and ~400 s later the episode died
on an opaque transport error with the root filesystem at 0 bytes free. So the
report is still written in EVERY case, and ``blocking_steps`` names the
reclamation steps that did not land so the caller can refuse the episode at
preparation time -- before any model spend. See ``providers/aws.py``
``_prepare_guest_disk`` for that refusal and its diagnostic.

CONDITIONAL, AND WHY THE THRESHOLD IS WHERE IT IS. Free space is measured on
every episode (that measurement is the point -- see "observability" below), but
the reclamation only runs when the guest has less than
``RECLAIM_BELOW_FREE_BYTES`` free. The threshold is set ABOVE the largest
measured consumer: the delta-download residue reached ~9.9 GB, so a guest with
more than 12 GiB free can absorb snapd's entire measured appetite and still
have room for the episode's own writes, and touching it would be housekeeping
nobody needs. Below that, the guest is on the trajectory the measurements above
describe.

NOTHING HERE DECIDES HEALTH FROM FREE SPACE ALONE, and that is deliberate: a
guest that reclaimed perfectly still sits at ~2.2 GB free (measured on the run
with the correct password, which completed, was scored, and served 200 on all
50 screenshots), well under the 12 GiB threshold. "Still below the threshold
after reclamation" is the NORMAL successful state, so it must never be an
error; what decides the outcome is whether the protective steps LANDED. That
is what ``blocking_steps`` reports.

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
download scratch cleared the 29.5G partition has room for a full episode
(measured: ~2.2 GB free after a successful hold+clear, on the run that completed
and scored). The disk vs partition geometry is REPORTED instead, read-only,
because that pair is what shows ``AWS_ROOT_VOLUME_SIZE`` changing nothing at all:
30,993,747,968 bytes of filesystem inside both a 40 GiB and a 120 GiB volume.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Callable, Literal, Sequence

# Reclaim only below this much free space. Set above the largest measured
# consumer (the ~9.9 GB of snapd delta downloads) plus room for the episode's
# own writes, so a guest that can already absorb a full auto-refresh is left
# alone.
RECLAIM_BELOW_FREE_BYTES = 12 * 1024**3

# Per-command and whole-preparation ceilings. The runner allows 900s for the
# whole ``reset_start`` (scripts/run_episode.py), most of which is instance
# readiness and upstream's own task setup, so hygiene gets a small fixed slice
# and a wedged guest cannot eat the reset budget: once the budget is spent the
# remaining steps are recorded as skipped rather than attempted.
COMMAND_TIMEOUT_S = 60.0
TOTAL_BUDGET_S = 180.0

# Two directions of the same download, and only one of them is where this
# image puts the bytes.
#
# ``_SNAPD_CACHE`` is snapcraft's documented download cache ("Data locations":
# "the working cache ... used to minimise download size and speed-up
# refreshes"). Its CONTENTS are removed, never the directory: snapd recreates
# files in it but does not recreate the directory itself on every path. Clearing
# it is still correct hygiene -- it is download scratch by definition -- but on
# this image it measured 4096 bytes, so on its own it reclaims nothing.
#
# ``_SNAPD_REVISIONS`` is what actually fills: snapd's downloader writes a
# pending refresh revision into it as ``<name>_<rev>.snap`` (a full download) or
# ``<name>_<rev>.snap.xdelta3-<old>-to-<new>.partial`` (a delta, the common
# case), and renames it on completion. Only the ``*.partial`` files are touched:
# every other ``.snap`` in there is an INSTALLED revision, and deleting one
# would remove an application from the benchmark rather than reclaim space.
_SNAPD_CACHE = "/var/lib/snapd/cache"
_SNAPD_REVISIONS = "/var/lib/snapd/snaps"

# The suffix snapd gives an incomplete download, matched as a GLOB inside the
# privileged shell (see ``_clear_download_scratch_fragment``): a literal string
# here would be expanded by whichever shell held it unquoted, which is the
# defect the fragments below exist to avoid, and a hard-coded name list would
# go stale the moment a different snap refreshes.
_SNAPD_PARTIAL_GLOB = "*.partial"

# Upstream's OWN documented development defaults, in upstream's own order
# (``desktop_env/providers/volume.py`` ``_expand_linux_guest_volume``:
# ``for candidate in "$PASSWORD" "<the OSWorld image default>" "password"``).
# They are here so the escalation ladder below has the same reach as upstream's
# and no invented value enters it: the first is the password OSWorld 2.0 images
# ship with and the value this project's runbook documents, the second upstream
# keeps for its older development images. NO other password may be added.
_UPSTREAM_DEFAULT_PASSWORDS: tuple[str, ...] = (
    "osworld-public-evaluation",
    "password",
)

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

    ``detail`` is bounded and carries the guest's own output, which leads with the
    escalation CHAIN that was tried (``escalation=agentless>supplied`` means
    ``sudo -n`` was refused and the supplied value was accepted) so a stale infra
    value is visible in the evidence. The COMMAND is never recorded: escalation
    runs through ``printf <candidate> | sudo -S`` (upstream's own pattern,
    setup.py:609), so the verbatim argv would put a password -- the operator's or
    upstream's default -- in a file on the operator's disk for no diagnostic
    gain.
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
            # Derived, not stored: the steps are the record, and this is the
            # answer the caller refused the episode on -- so an operator reading
            # the file after a refusal does not have to know the rule.
            "blocking_steps": [step.name for step in self.blocking_steps()],
            "reason": self.reason,
            "steps": [step.to_json() for step in self.steps],
        }

    def to_json_bytes(self) -> bytes:
        return json.dumps(self.to_json(), indent=2, sort_keys=True).encode("utf-8")

    def blocking_steps(self) -> tuple[StepOutcome, ...]:
        """Reclamation steps that did NOT land, in the order they ran.

        THE QUESTION THIS ANSWERS, precisely. Not "is the guest short of
        space" -- a guest that reclaimed perfectly still sits at ~2.2 GB free,
        far under the 12 GiB threshold, and that is the state every COMPLETED
        run measured. And not "did the reclamation run" -- the
        ``reclamation_attempted`` flag says that. It is "are the steps that stop
        the fill still standing": a guest whose hold and clear landed is
        protected even at 2.2 GB free, while a guest whose escalation failed is
        on the trajectory that killed four paid episodes however much space it
        happens to have right now. So a report is blocking when one of those
        steps is not ``ok``, which includes ``skipped`` (the pass ran out of
        budget before reaching it) and ``unreachable`` (the control server never
        answered, so nothing was done either).

        The one EXCEPTION is the covered pair: an old snapd rejects
        ``snap refresh --hold`` as an unknown flag and the ``refresh.hold``
        setting is the same hold by another name, so a failed hold with a
        succeeding fallback is not a blocker. Both failing is.

        Measurement and geometry steps are deliberately NOT consulted: a
        garbled ``df`` after a successful reclamation is a gap in the evidence,
        not a reason to throw away an episode that would have run.
        """

        if not self.reclamation_attempted:
            return ()
        by_name = {step.name: step for step in self.steps}
        blockers: list[StepOutcome] = []
        for step in self.steps:
            if step.name not in _RECLAMATION_STEP_NAMES or step.status == "ok":
                continue
            covering = _COVERING_STEPS.get(step.name)
            if covering is not None and by_name.get(covering, None) is not None:
                if by_name[covering].status == "ok":
                    continue
            blockers.append(step)
        return tuple(blockers)


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
        exception thrown out of the preparation pass. What that fact then MEANS
        is the caller's decision -- ``blocking_steps`` decides it here, and
        ``providers/aws.py`` refuses an episode whose reclamation did not land.
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
    # Parsed whenever a number is present, NOT only on exit 0: the script
    # prints the filesystem size first and the whole-disk size second, so a
    # partial answer (an image without ``lsblk``, a ``df`` that printed and
    # then something else failed) still yields what was measured. Gating on the
    # exit code discarded a real ``df`` figure over a missing block-device
    # tool, which is observability lost for no safety gained.
    if size_result is not None:
        numbers = [int(line) for line in size_result.stdout.split() if line.strip().isdigit()]
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
# docstring for why growing it is deliberately out of scope). The block-device
# half is best-effort (``|| true``): without it a guest lacking ``findmnt`` or
# ``lsblk`` exits 127 from the ``&&`` chain and the ``df`` figure that WAS
# printed arrives under a non-zero status. Its own stderr is dropped so the
# step's detail is the numbers, not a tool's usage text.
_GEOMETRY_SCRIPT = (
    "df -B1 --output=size / | tail -1; "
    '{ source=$(findmnt -no SOURCE /) && parent=$(lsblk -no PKNAME "$source" | head -1) '
    '&& lsblk -bdno SIZE "/dev/$parent"; } 2>/dev/null || true'
)

# Everything privileged runs INSIDE ``sudo -S bash -c '<fragment>'``, and the
# fragments below are what that inner shell executes. Two hard-won rules:
#
# * Nothing that depends on privilege may happen in the outer shell. The guest's
#   control server runs as an unprivileged user, so a glob like
#   ``"/var/lib/snapd/cache"/*`` expanded OUT there matches nothing -- the
#   directory is ``drwx------ root:root`` -- and ``rm -rf`` of the literal,
#   non-existent name exits 0. That was a reclaim that deleted nothing and
#   reported ``ok``. ``find -mindepth 1 -delete`` inside the privileged shell
#   has no glob to expand anywhere.
# * Nothing may be exec'd through ``xargs`` (or any other splitter) with the
#   password pipeline as its command: ``xargs ... echo 'pw' | sudo -S snap abort``
#   parses as ``xargs echo 'pw'`` PIPED INTO one ``sudo -S snap abort`` -- the
#   password line becomes ``pw <id>`` (rejected), abort receives no id, and the
#   exec'd ``/bin/echo pw`` is visible in ``ps``. The loop lives inside the
#   privileged shell instead, and the fragments avoid single quotes so the one
#   quoting layer (``_shell_quote``) stays readable.

# Only the two change kinds that fill the disk are aborted (snapd's own
# summaries are ``Auto-refresh ...`` and ``Pre-download ... for auto-refresh``),
# matched CASE-SENSITIVELY so this module's own ``Hold auto-refreshes for all
# snaps`` change is never a candidate, and so a seeding change or a hook that
# happens to be ``Doing`` is left alone. ``snap changes`` failing (no snapd,
# denied) is surfaced as the step's exit status rather than read as "nothing
# in flight"; a failed abort of any one change fails the step, with the
# remaining ids still attempted.
_ABORT_REFRESH_FRAGMENT = (
    "changes=$(snap changes) || exit $?; rc=0; "
    "while read -r id status rest; do "
    'case "$status $rest" in "Doing "*Auto-refresh*|"Doing "*Pre-download*) '
    'snap abort "$id" || rc=1;; esac; '
    'done <<<"$changes"; exit $rc'
)

# The download scratch this image actually fills, and the directory that LOOKS
# like it. ``/var/lib/snapd/cache`` is cleared because that is what it is for --
# snapd's documented working cache -- and because on another image it may be the
# one holding gigabytes. It measured 4096 bytes here. The bytes are in
# ``/var/lib/snapd/snaps``, as incomplete downloads: see ``_SNAPD_REVISIONS``.
#
# A PATHNAME GLOB IS CORRECT HERE and only because of where it runs. This
# fragment is the argument of ``sudo -S bash -c '...'``, so the shell that
# expands ``*.partial`` is the PRIVILEGED one. The same glob written in the
# OUTER shell is the defect this module already paid for: an unprivileged shell
# cannot read a ``drwx------ root:root`` directory, so ``rm -rf --
# <dir>/*`` matched nothing, ``rm`` of the literal name exited 0, and the step
# reported success while deleting nothing. ``rm -f`` also exits 0 when the glob
# matches nothing, which is right: an empty scratch directory is not a failure.
_CLEAR_DOWNLOAD_SCRATCH_FRAGMENT = (
    f"rc=0; "
    f"find {_SNAPD_CACHE} -mindepth 1 -delete || rc=1; "
    f"cd {_SNAPD_REVISIONS} && rm -f -- {_SNAPD_PARTIAL_GLOB} || rc=1; "
    f"exit $rc"
)

#: The steps whose failure means the guest is still on the trajectory the
#: measurements describe: snapd downloading into a filesystem that has no room
#: for it. ``blocking_steps`` reports them and ``providers/aws.py`` refuses the
#: episode on any of them. The measurements are the steps' own names -- a step
#: renamed here without being renamed there would silently stop blocking.
_RECLAMATION_STEP_NAMES = frozenset(
    {
        "abort-in-flight-snap-changes",
        "hold-snap-auto-refresh",
        "clear-snapd-download-scratch",
    }
)

#: A failed step that a later step covers. ``snap refresh --hold`` needs snapd
#: 2.58+; on anything older it fails with an unknown-flag error and the
#: ``refresh.hold`` setting is the pre-2.58 way of saying the same thing, so the
#: pair achieves the hold and only BOTH failing means the hold did not land.
_COVERING_STEPS = {"hold-snap-auto-refresh": "hold-snap-auto-refresh-fallback"}


def _reclaim(session: _Session, client_password: str) -> None:
    """Stop snapd filling the disk, then delete what it already downloaded.

    ORDER IS LOAD-BEARING: abort, then hold, then clear.

    Abort FIRST because it is the only step that is immediate. ``snap abort``
    flips the change's state and returns. ``snap refresh --hold`` is a
    ``configure core`` hook change that the CLI WAITS on (``cmd_snap_op.go``
    ``holdRefreshes``), and snapd runs one hook task per snap at a time
    (``hookmgr.go`` ``snapIsRunningHook``), so while the boot-time auto-refresh
    is still ``Doing`` a core/snapd hook of its own the hold queues behind it --
    long enough to outlast the per-command ceiling and be recorded as
    unreachable, after which the fallback joins the same queue. With nothing in
    flight, the hold completes at once. Snapd cannot launch a fresh auto-refresh
    in the gap: it enforces a 20-minute ``refreshRetryDelay`` between launches
    (``autorefresh.go``), so the hold lands long before a retry is considered.

    Hold SECOND so no NEW refresh starts once the in-flight one is gone.

    Clear LAST so nothing is still writing into the directory being emptied;
    clearing first would race a live download and reclaim nothing.

    What abort does to installed snaps, stated accurately: a change that had
    already finished refreshing some of its snaps has those tasks UNDONE, which
    returns each such snap to the revision the AMI shipped -- the benchmark's
    own baseline, and the state every other episode starts from. It is not an
    uninstall, which is the line this module does not cross.

    EVERY STEP RUNS THROUGH ``escalation_script``, so each is one privileged
    ``bash -c`` reached by the candidate ladder rather than by the operator's
    password alone: see that function for why, and for what a wrong value used
    to cost.
    """

    def privileged(fragment: str) -> str:
        return escalation_script(fragment, client_password)

    session.step("abort-in-flight-snap-changes", privileged(_ABORT_REFRESH_FRAGMENT))

    # ``snap refresh --hold`` needs snapd 2.58+. On anything older it exits
    # non-zero with an unknown-flag error, which is recorded and then covered by
    # the ``refresh.hold`` fallback below -- the pre-2.58 way of saying the same
    # thing. Running the fallback unconditionally would be a second write to the
    # same setting on every modern guest, so it is conditional on the first
    # failing.
    held = session.step("hold-snap-auto-refresh", privileged("snap refresh --hold=forever"))
    if held is None or held.returncode != 0:
        session.step(
            "hold-snap-auto-refresh-fallback",
            privileged(f"snap set system refresh.hold={_FALLBACK_HOLD_UNTIL}"),
        )

    # The contents, not the directory: snapd expects the directory to exist.
    # ``-mindepth 1`` is what keeps the directory; ``-delete`` implies
    # depth-first so nested entries go before their parents. The second half --
    # the incomplete downloads in the revision directory, which is where this
    # image's ~10 GB of growth actually is -- is why the step is no longer named
    # after the cache alone: the cache is download scratch that was EMPTY, and a
    # reader who trusts the old name goes looking in the wrong directory.
    session.step("clear-snapd-download-scratch", privileged(_CLEAR_DOWNLOAD_SCRATCH_FRAGMENT))


def _fallback_passwords(client_password: str) -> tuple[str, ...]:
    """Upstream's documented defaults to try after the supplied value, or none.

    THE LADDER'S THIRD RUNG IS GATED, and the gate is the whole point of this
    function. It opens only when the value the operator supplied is ITSELF one
    of upstream's documented defaults -- which is both the measured case (the
    campaign passed upstream's older development default, which this image
    rejects, while the value this project's runbook documents is the one that
    works) and the case the runbook's own command produces. A value the operator
    chose deliberately is not second-guessed: if it is rejected, the steps fail
    and ``blocking_steps`` refuses the episode with a diagnostic naming
    ``OSWORLD_CLIENT_PASSWORD``, rather than the harness quietly succeeding with
    a credential nobody supplied. Both paths are safe; only one of them tells
    the truth about which password opened the guest.

    Nothing here is invented: the candidates ARE upstream's list, and a value
    absent from it can never be tried.
    """

    if client_password not in _UPSTREAM_DEFAULT_PASSWORDS:
        return ()
    return tuple(
        candidate for candidate in _UPSTREAM_DEFAULT_PASSWORDS if candidate != client_password
    )


def escalation_script(fragment: str, client_password: str) -> str:
    """One privileged step, run through the same candidate ladder upstream uses.

    WHY A LADDER. ``sudo -S`` is fed the operator's ``OSWORLD_CLIENT_PASSWORD``,
    and a stale or wrong value is not a request error -- sudo simply rejects it,
    the step exits non-zero, and this module used to record that and carry on
    into a guest that dies later. Measured: ``sudo: no password was provided`` /
    ``sudo: 1 incorrect password attempt`` on every privileged step, four paid
    episodes lost to the disk filling behind them. Upstream's own
    ``expand_guest_volume`` (vendored ``desktop_env/providers/volume.py``) meets
    the same class with a candidate SEQUENCE -- ``sudo -n``, then the supplied
    password, then known development defaults -- and this is that same
    discipline, in the same order, with the same values and nothing else.

    WHY ``sudo -n`` FIRST. It is the only candidate that cannot fail for the
    wrong reason: if the guest has a live sudo timestamp it costs one fork and
    no credential is used at all. It is also the one attempt whose stderr would
    otherwise be noise (``sudo: a password is required``), which is why the
    ladder discards each failed attempt's output instead of concatenating them.

    THE RECORDED OUTPUT. Everything the last attempt produced is collected into
    ONE output, prefixed by ``escalation=<chain>``: the rungs that were tried, in
    order, so ``guest-preparation.json`` says WHICH password opened the guest --
    ``escalation=agentless>supplied`` means ``sudo -n`` was refused and the
    operator's value worked -- without ever saying what any password was. On a
    step that SUCCEEDED the chain ends on the rung that authenticated; on one
    that failed it ends on the last rung tried and the collected output says why
    (``sudo: 1 incorrect password attempt`` is auth; anything else is the
    fragment's own failure). ``escalation=none`` means nothing was attempted at
    all, which is its own answer.

    Each candidate appears exactly once in the script, as the stdin of exactly
    one ``sudo -S``; nothing is ever exec'd with a password in its argv
    (``printf`` is a shell builtin).
    """

    quoted_fragment = _shell_quote(fragment)
    #: Tier labels are fixed vocabulary, never a value: they land in the report.
    attempts: list[tuple[str, str]] = [("agentless", "")]
    if client_password:
        attempts.append(("supplied", client_password))
    attempts.extend(("upstream-default", value) for value in _fallback_passwords(client_password))

    lines = ["rc=1", "tiers=none", "out=''"]
    for index, (tier, password) in enumerate(attempts):
        # One ``bash -c`` per attempt: ``sudo -n`` takes no password at all, and
        # every other tier feeds exactly one candidate to exactly one
        # ``sudo -S``. The password is NEVER piped into the agentless attempt --
        # nothing needs to read stdin there, and a pipe would put a credential
        # on a command line whose whole point is that it used none.
        if password:
            invocation = (
                f"printf '%s\\n' {_shell_quote(password)} | sudo -S bash -c {quoted_fragment}"
            )
        else:
            invocation = f"sudo -n bash -c {quoted_fragment}"
        # The FIRST attempted rung replaces the ``none`` placeholder; later ones
        # extend the chain, so the label can never claim a rung that was not
        # reached.
        record = f"tiers={_shell_quote(tier)}" if index == 0 else f'tiers="$tiers>{tier}"'
        lines.append(f'if [ "$rc" -ne 0 ]; then {record}; out=$({invocation} 2>&1); rc=$?; fi')
    lines.append("printf 'escalation=%s\\n' \"$tiers\"")
    lines.append("printf '%s\\n' \"$out\"")
    lines.append("exit $rc")
    return "; ".join(lines)


def _shell_quote(value: str) -> str:
    """POSIX single-quote one value for the ``bash -c`` script.

    The client password is operator-supplied infra and reaches the guest through
    a shell pipeline (the only way to feed ``sudo -S`` through an endpoint with
    no stdin). Quoting it here means a password containing a space, a quote, or
    a ``$`` cannot terminate the command or expand into something else.
    """

    return "'" + value.replace("'", "'\"'\"'") + "'"
