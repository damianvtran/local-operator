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

GROWING THE ROOT FILESYSTEM -- ONLY WHEN A VOLUME SIZE WAS REQUESTED, AS
UPSTREAM DOES. An earlier revision refused to grow the partition at all, and
the record it left shows what that cost: all 84 ``guest-preparation.json``
records of the 2026-11 campaign read ``disk_bytes`` 128,849,018,880 (the 120
GiB ``AWS_ROOT_VOLUME_SIZE`` volume) around ``filesystem_bytes`` 30,993,747,968,
so the 120 GiB was INERT: paid for and unreachable. Upstream's contract, as of
the pinned release (OSWorld-V2.1 @ 3d778a3,
``desktop_env/providers/volume.py``), is explicit: ``DesktopEnv(volume_size=N)``
both sizes the EBS volume AND runs ``expand_guest_volume`` (``growpart`` +
``resize2fs``/``xfs_growfs``/``btrfs``, installing ``cloud-guest-utils`` first
when ``growpart`` is missing), while ``volume_size=None`` leaves the AMI's
29.5G filesystem inside upstream's 40 GiB default volume untouched.

WHICH SIZE IS "REQUESTED" follows upstream's precedence exactly: a task's own
``volume_size`` first, and the run-level default (upstream's ``--volume_size``,
our ``AWS_ROOT_VOLUME_SIZE``) only for a task that declares none
(``provisioning._resolve_root_volume``). THE FAITHFUL CONFIGURATION SETS NO
OVERRIDE: upstream's documented run passes no ``--volume_size``, so 18 of
v2.1's 108 tasks are grown to their declared size and the other 90 are not.
An override grants the non-declaring tasks disk upstream's documented run
does not; the report records ``volume_size_source`` (``task`` | ``override`` |
null) per episode and the manifest stamps ``aws_root_volume_size_override``,
so such a run is identifiable and must be disclosed as a deviation.

The objections the earlier revision recorded are answered rather than ignored:
``growpart`` IS absent from the AMI, which is why the step installs it the way
upstream does -- and at preparation time the guest has ~2.2 GB free (measured on
every record above), so ``apt-get`` can run; the "cannot install onto a full
disk" observation was made at 0 bytes free, late in a dying episode. ``sfdisk``
is still never called directly: ``growpart`` is the standard, upstream-chosen
tool, and it refuses (``NOCHANGE``) rather than rewriting a partition that cannot
grow.

ADVISORY HERE, STRICT UPSTREAM -- AND WHY THAT IS NOT AN ADVANTAGE. v2.1's
script runs under ``set -euo pipefail`` (an ``apt-get`` or ``growpart`` failure
fails ``DesktopEnv`` construction), and ``verify_linux_guest_volume`` RAISES
when the mounted filesystem is under 95 % of the requested size. This step
never refuses the episode over a failed or short grow: the only difference that
makes is an agent running with LESS disk than upstream would have given it,
never more, and a guest whose snapd hold landed runs a full episode at 2.2 GB
free. Refusing would trade a recorded gap for a lost paid episode. The gap is
recorded instead, against v2.1's own test: ``grow_target_met`` is the 95 %
check (``filesystem_bytes_after`` -- or ``filesystem_bytes`` when no grow was
needed -- against the requested GiB), so a short-grown episode is identifiable
from its bundle. The grow is SKIPPED when the reclamation already failed
(``blocking_steps`` non-empty): the provider refuses that episode anyway.
"""

from __future__ import annotations

import dataclasses
import json
import time
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
#
# 270 s, up from 180 s, to fund the hold's transient-failure retries below: the
# measured failure (t33v-t054) spent ~60 s on ``hold-snap-auto-refresh`` and
# ~60 s on its fallback before ``clear-snapd-download-scratch`` ran, so 180 s
# left no room for a single retry of the hold without starving the clear. Four
# full sends (hold, its retry, fallback, clear) plus the abort, the
# measurements and the cadence's waits need a little over 240 s; 270 s funds
# them with margin. A guest whose hold lands at once spends what it always did.
COMMAND_TIMEOUT_S = 60.0
TOTAL_BUDGET_S = 270.0

# Retrying the snapd hold when the control server could not answer it.
#
# THE MEASURED CASE. Two of 84 campaign records show ``hold-snap-auto-refresh``
# as ``unreachable`` / ``HTTPError`` while the steps on either side of it
# (``abort-in-flight-snap-changes`` before, ``clear-snapd-download-scratch``
# after) came back ``ok`` -- so the control server was up, and the error was
# the server's own non-2xx answer, not a dead guest. The server answers 500 from
# its ``except Exception`` arm, ``subprocess.TimeoutExpired`` among them, and
# ``snap refresh --hold`` is a ``configure`` hook change the CLI WAITS on behind
# whatever hook the just-aborted auto-refresh is still undoing (see
# ``_reclaim``) -- the most likely reading, though the record kept only the
# exception type. In one of the two the fallback then landed; in the other
# (t33v-t054-20261112-090503) both did not, and the episode was refused.
#
# WHY A RETRY IS SAFE HERE when the provider's own transport policy forbids
# replaying a sent request: holding refreshes is IDEMPOTENT. A hold that did
# land on the guest after its HTTP answer was lost is simply re-asserted, which
# is the state the step exists to reach. Nothing else is retried: the abort and
# the clear are not where the failures were, and a COMPLETED step that exited
# non-zero (an old snapd's unknown flag) is an answer, not a transient, so it
# goes straight to the fallback as before.
#
# TWO BOUNDS, for the reason ``providers/base.py`` gives for its transport
# retries: an attempt count bounds nothing when each attempt can spend a full
# command timeout, and a window alone bounds nothing on a guest that refuses in
# milliseconds. A retry also never starts unless the session can still fund
# the retry itself AND every step after it -- the fallback and the clear for
# the hold, the clear for the fallback -- because the clear is the step that
# actually returns the space. Worst case, every send spending its whole 60 s:
# hold 0-60 s, one retry 60-120 s, fallback 120-180 s, clear 180-240 s.
HOLD_RETRY_ATTEMPTS = 4  # TOTAL sends: the first attempt plus three retries
HOLD_RETRY_WINDOW_S = 90.0

# The waits between those attempts, by default. The provider passes its own
# recovery-probe cadence (``providers/aws.py`` ``_RECOVERY_PROBE_DELAYS_S``:
# 0.5, 0.5, 1, 1, then 2 s) so a guest that is briefly not answering is
# re-tried on the same schedule as every other recovery in the adapter; this
# default is that schedule's head, for callers (and tests) that inject none.
DEFAULT_RETRY_DELAYS_S: tuple[float, ...] = (0.5, 0.5, 1.0)

# The grow step's own ceiling, OUTSIDE ``TOTAL_BUDGET_S``: it runs after the
# protective steps and must neither starve them nor be starved by them.
# ``apt-get update`` plus one small package install is the long pole (tens of
# seconds on a fresh guest); ``growpart`` and an online ``resize2fs`` of ~90 GiB
# are seconds. Upstream allows 600 s for the same script, but upstream's reset
# budget is not shared with ours: worst case here is 270 s + 180 s of
# preparation inside the runner's 900 s ``reset_start``, and the common case
# (hold lands first time, grow in well under a minute) is unchanged from before.
# UNMEASURED: 180 s is an estimate, not a fitted bound -- no grown guest on this
# AMI had been timed when it was set. The grow step records its own
# ``duration_s`` in ``guest-preparation.json``, so the first live grown episodes
# supply the distribution to re-fit it against ("measure before you cap").
GROW_TIMEOUT_S = 180.0

# v2.1's ``verify_linux_guest_volume`` threshold: the mounted filesystem must be
# at least this fraction of the requested GiB ("Allow filesystem metadata and
# the small boot partitions in provider images"). Upstream raises below it; this
# module records ``grow_target_met`` instead (see the module docstring).
GROW_TARGET_FRACTION = 0.95

# A filesystem within this much of its block device is already grown. The
# AMI's own partition table, alignment and an ESP/BIOS-boot partition account
# for well under a GiB; the gap the grow step exists for is ~90 GiB (or ~11 GiB
# at upstream's 40 GiB default), so 2 GiB separates the two without touching
# either.
GROWN_WITHIN_BYTES = 2 * 1024**3

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
    # How many times the command was SENT. 1 for every ordinary step, 0 for a
    # skipped one, and up to ``HOLD_RETRY_ATTEMPTS`` for the retried hold -- so
    # a hold that landed on its third send reads differently from one that
    # landed at once, and a reader never mistakes a retried step for a clean one.
    attempts: int = 1
    # Wall seconds the step took, set only on steps whose ceiling is still an
    # estimate (the grow step; see ``GROW_TIMEOUT_S``) and omitted from the JSON
    # otherwise, so every existing record keeps its shape.
    duration_s: float | None = None

    def to_json(self) -> dict[str, object]:
        payload: dict[str, object] = {
            "name": self.name,
            "status": self.status,
            "returncode": self.returncode,
            "detail": self.detail,
            "attempts": self.attempts,
        }
        if self.duration_s is not None:
            payload["duration_s"] = round(self.duration_s, 3)
        return payload


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
    # Whether a volume size was requested, i.e. whether upstream would have
    # grown this guest (see the module docstring), and the filesystem size once
    # the grow step ran. ``filesystem_bytes_after`` is None when growth was not
    # requested, was skipped as unnecessary, or could not be re-measured; set,
    # it is the number that proves the grow landed (``disk_bytes`` within
    # ``GROWN_WITHIN_BYTES``) or shows the fidelity gap it left.
    root_grow_requested: bool = False
    filesystem_bytes_after: int | None = None
    # The size that was requested and WHERE it came from (``task`` -- the task
    # file's own ``volume_size``, upstream-faithful -- or ``override``, the
    # ``AWS_ROOT_VOLUME_SIZE`` default filling a task that declares none, a
    # disclosed deviation), and v2.1's 95 % check against it: True/False once a
    # filesystem size is known, None when nothing was requested or nothing
    # could be measured.
    requested_volume_gb: int | None = None
    volume_size_source: str | None = None
    grow_target_met: bool | None = None

    def to_json(self) -> dict[str, object]:
        return {
            "free_bytes_before": self.free_bytes_before,
            "free_bytes_after": self.free_bytes_after,
            "filesystem_bytes": self.filesystem_bytes,
            # The whole block device. When it exceeds ``filesystem_bytes`` the
            # root partition was never grown into an AWS_ROOT_VOLUME_SIZE
            # override -- the measured 29.5G-in-100GiB case.
            "disk_bytes": self.disk_bytes,
            "root_grow_requested": self.root_grow_requested,
            "requested_volume_gb": self.requested_volume_gb,
            "volume_size_source": self.volume_size_source,
            "filesystem_bytes_after": self.filesystem_bytes_after,
            "grow_target_met": self.grow_target_met,
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

        return _blocking_steps(self.steps, reclamation_attempted=self.reclamation_attempted)


def _blocking_steps(
    steps: Sequence[StepOutcome], *, reclamation_attempted: bool
) -> tuple[StepOutcome, ...]:
    """``GuestDiskReport.blocking_steps`` over a step list still being built.

    A function rather than only the method because ``prepare_guest_disk``
    needs the same answer BEFORE the report exists, to skip a grow on an
    episode the provider is about to refuse; one rule, two callers.
    """

    if not reclamation_attempted:
        return ()
    by_name = {step.name: step for step in steps}
    blockers: list[StepOutcome] = []
    for step in steps:
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

    def __init__(
        self,
        run: GuestCommand,
        clock: Clock,
        sleep: Callable[[float], None] = time.sleep,
    ) -> None:
        self._run = run
        self._clock = clock
        self._sleep = sleep
        self._deadline = clock() + TOTAL_BUDGET_S
        self.steps: list[StepOutcome] = []

    def extend(self, seconds: float) -> None:
        """Fund a step that has its own ceiling outside ``TOTAL_BUDGET_S``.

        Measured from NOW, not added to the old deadline: a reclamation that
        finished early must not hand its unused seconds to the grow step, and
        one that ran out must not leave the grow step with nothing.
        """

        self._deadline = self._clock() + seconds

    def step(
        self,
        name: str,
        script: str,
        *,
        timeout_s: float = COMMAND_TIMEOUT_S,
        timed: bool = False,
    ) -> CommandResult | None:
        """Run one step, record it, and return its result or None on failure.

        Never raises: a step that cannot run is a recorded fact, not an
        exception thrown out of the preparation pass. What that fact then MEANS
        is the caller's decision -- ``blocking_steps`` decides it here, and
        ``providers/aws.py`` refuses an episode whose reclamation did not land.
        """

        # The clock is read only for a timed step: an untimed one must consume
        # exactly the clock reads it always did, or every budget test shifts.
        started = self._clock() if timed else 0.0
        outcome, result = self._attempt(name, script, timeout_s)
        if timed:
            outcome = dataclasses.replace(outcome, duration_s=self._clock() - started)
        self.steps.append(outcome)
        return result

    def step_retrying_unreachable(
        self, name: str, script: str, *, delays: Sequence[float], reserve_s: float
    ) -> CommandResult | None:
        """``step``, re-sent while the control server could not answer it.

        ONLY for an idempotent fragment (see ``HOLD_RETRY_ATTEMPTS`` for why
        the hold qualifies). Only ``unreachable`` is retried: ``failed`` is the
        guest's own answer and ``skipped`` means the budget is gone. ONE step is
        recorded whatever happens -- its final outcome, with ``attempts`` saying
        how many sends it took -- because ``blocking_steps`` and the covering
        fallback key on the step's name, and a retried step written once per
        send would make an early failure shadow the send that landed.

        ``reserve_s`` is what the steps AFTER this one need; a retry starts
        only when the budget still covers one more full send plus that.
        """

        first_failure: float | None = None
        attempt = 0
        while True:
            attempt += 1
            outcome, result = self._attempt(name, script, COMMAND_TIMEOUT_S)
            if outcome.status != "unreachable" or attempt >= HOLD_RETRY_ATTEMPTS:
                break
            now = self._clock()
            if first_failure is None:
                first_failure = now
            # Indexed by the send that just failed, holding on the last entry:
            # the cadence is a schedule of waits, not a cap on how many fit.
            delay = delays[min(attempt - 1, len(delays) - 1)] if delays else 0.0
            if now + delay - first_failure > HOLD_RETRY_WINDOW_S:
                break
            if self._deadline - (now + delay) < COMMAND_TIMEOUT_S + reserve_s:
                break
            self._sleep(delay)
        # A skipped final round sent nothing, so it does not count as a send.
        sends = attempt if outcome.status != "skipped" else attempt - 1
        self.steps.append(dataclasses.replace(outcome, attempts=sends))
        return result

    def _attempt(
        self, name: str, script: str, timeout_s: float
    ) -> tuple[StepOutcome, CommandResult | None]:
        remaining = self._deadline - self._clock()
        if remaining <= 0:
            return (
                StepOutcome(name=name, status="skipped", detail="budget exhausted", attempts=0),
                None,
            )
        try:
            result = self._run(_bash(script), min(timeout_s, remaining))
        except Exception as error:
            # A transport failure (control server down, timeout, malformed
            # response) is indistinguishable from a guest that never answered,
            # and both mean the same thing to a reader: the guest was not
            # prepared. The TYPE is recorded; the message is not, because a
            # requests exception echoes the URL and query it was given.
            return (
                StepOutcome(name=name, status="unreachable", detail=_describe_error(error)),
                None,
            )
        detail = (result.stderr or result.stdout or "").strip()[:200]
        return (
            StepOutcome(
                name=name,
                status="ok" if result.returncode == 0 else "failed",
                returncode=result.returncode,
                detail=detail,
            ),
            result,
        )


def _describe_error(error: BaseException) -> str:
    """The exception's type, plus the HTTP status when the server sent one.

    The status is the one value-free fact that separates "the control server
    answered 500" (its exception envelope: a guest-side ``TimeoutExpired``, for
    one) from "nothing answered at all" -- exactly what the t33v-t054 record
    could not say, because it kept only ``HTTPError``. The message stays out for
    the reason the caller gives: it echoes the URL, and the guest's own 500
    message would echo the command, whose escalation ladder carries a password.
    """

    name = type(error).__name__
    status = getattr(getattr(error, "response", None), "status_code", None)
    if type(status) is int:
        return f"{name} status={status}"
    return name


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
    requested_volume_gb: int | None = None,
    volume_source: str | None = None,
    sleep: Callable[[float], None] = time.sleep,
    retry_delays: Sequence[float] = DEFAULT_RETRY_DELAYS_S,
) -> GuestDiskReport:
    """Measure the guest's root filesystem, reclaim it if tight, grow it if asked.

    Returns a report in every case, including the cases where nothing could be
    measured or nothing could be run. It raises nothing by construction: see the
    module docstring for why an unreachable guest must not cost an episode that
    would otherwise have worked.

    ``requested_volume_gb`` is upstream's own gate -- "was a volume size
    requested" -- and the caller passes exactly the resolved size (a task's own
    ``volume_size`` first, then the override; ``volume_source`` says which).
    None means no grow; see the module docstring for why growing on any other
    condition would hand the agent a disk upstream would not. ``sleep`` and
    ``retry_delays`` pace the hold's retries (injected so the
    provider uses its own recovery cadence and tests use no wall time).
    """

    session = _Session(run, clock, sleep)
    free_before = _measure_free_bytes(session, "measure-free-before")
    filesystem_bytes, disk_bytes = _measure_geometry(session, "measure-geometry")

    if free_before is not None and free_before >= RECLAIM_BELOW_FREE_BYTES:
        reclamation_attempted = False
        reason = "above-threshold"
    else:
        # An unmeasurable guest is reclaimed anyway. The steps are safe and
        # reversible; the failure they prevent destroys a paid episode. Choosing
        # to skip on a failed measurement would make the protection absent
        # exactly when the guest is least healthy.
        reclamation_attempted = True
        reason = "below-threshold" if free_before is not None else "unmeasured"
        _reclaim(session, client_password, retry_delays)

    # AFTER the reclamation, never before: the hold and clear are what stop the
    # fill, and they must not wait behind an ``apt-get update``. Growing first
    # would also race snapd's downloads for the very space apt needs.
    filesystem_after: int | None = None
    grew = False
    grow_root = requested_volume_gb is not None
    if grow_root and _blocking_steps(session.steps, reclamation_attempted=reclamation_attempted):
        # The provider refuses this episode on ``blocking_steps`` the moment
        # this returns, so a grow (apt + growpart + resize, up to
        # ``GROW_TIMEOUT_S``) would only delay a refusal. Recorded, not silent.
        session.steps.append(
            StepOutcome(
                name=GROW_STEP_NAME,
                status="skipped",
                detail="reclamation blocked; episode will be refused",
                attempts=0,
            )
        )
    elif grow_root:
        grew = _grow_root_filesystem(session, client_password, filesystem_bytes, disk_bytes)
        if grew:
            filesystem_after, _ = _measure_geometry(session, "measure-geometry-after")

    if reclamation_attempted or grew:
        free_after = _measure_free_bytes(session, "measure-free-after")
    else:
        free_after = free_before
    return GuestDiskReport(
        free_bytes_before=free_before,
        free_bytes_after=free_after,
        filesystem_bytes=filesystem_bytes,
        disk_bytes=disk_bytes,
        threshold_bytes=RECLAIM_BELOW_FREE_BYTES,
        reclamation_attempted=reclamation_attempted,
        reason=reason,
        steps=tuple(session.steps),
        root_grow_requested=grow_root,
        filesystem_bytes_after=filesystem_after,
        requested_volume_gb=requested_volume_gb,
        volume_size_source=volume_source if grow_root else None,
        grow_target_met=_grow_target_met(
            requested_volume_gb, filesystem_after if grew else filesystem_bytes
        ),
    )


def _grow_target_met(requested_volume_gb: int | None, filesystem_bytes: int | None) -> bool | None:
    """v2.1's ``verify_linux_guest_volume`` test, recorded instead of raised.

    ``filesystem_bytes`` is the size AFTER the grow when one ran, else the
    size measured at the start (a filesystem that already spanned the volume
    needed no grow and is judged as it stood). Same measure as upstream:
    ``df --output=size`` is statvfs ``f_blocks * f_frsize``, the number upstream's
    probe compares.
    """

    if requested_volume_gb is None or filesystem_bytes is None:
        return None
    return filesystem_bytes >= int(requested_volume_gb * 1024**3 * GROW_TARGET_FRACTION)


def _measure_geometry(session: _Session, name: str) -> tuple[int | None, int | None]:
    """(filesystem bytes, whole-disk bytes), each None when not measured.

    Parsed whenever a number is present, NOT only on exit 0: the script prints
    the filesystem size first and the whole-disk size second, so a partial
    answer (an image without ``lsblk``, a ``df`` that printed and then something
    else failed) still yields what was measured. Gating on the exit code
    discarded a real ``df`` figure over a missing block-device tool, which is
    observability lost for no safety gained.
    """

    result = session.step(name, _GEOMETRY_SCRIPT)
    if result is None:
        return None, None
    numbers = [int(line) for line in result.stdout.split() if line.strip().isdigit()]
    filesystem_bytes = numbers[0] if numbers else None
    disk_bytes = numbers[1] if len(numbers) > 1 else None
    return filesystem_bytes, disk_bytes


def _grow_root_filesystem(
    session: _Session,
    client_password: str,
    filesystem_bytes: int | None,
    disk_bytes: int | None,
) -> bool:
    """Grow the root partition and filesystem into the volume; True if it ran.

    IDEMPOTENT by two independent checks. Here: a filesystem already within
    ``GROWN_WITHIN_BYTES`` of its disk is recorded as ``skipped`` and nothing is
    sent. In the guest: ``growpart`` answers ``NOCHANGE`` for a partition that
    cannot grow, which the fragment reads as success, and ``resize2fs`` on a
    filesystem that already fills its partition is a no-op. A geometry that
    could not be measured is grown anyway: upstream runs its script without
    measuring at all, and both guest-side checks still hold.

    BOUNDED by ``GROW_TIMEOUT_S``, funded outside the reclamation budget (see
    that constant). ADVISORY: never in ``_RECLAMATION_STEP_NAMES``, so a failed
    grow is recorded, not refused -- the module docstring has the reasoning.
    """

    if (
        filesystem_bytes is not None
        and disk_bytes is not None
        and disk_bytes - filesystem_bytes <= GROWN_WITHIN_BYTES
    ):
        session.steps.append(
            StepOutcome(
                name=GROW_STEP_NAME,
                status="skipped",
                detail="filesystem already spans the volume",
                attempts=0,
            )
        )
        return False
    session.extend(GROW_TIMEOUT_S + COMMAND_TIMEOUT_S)
    # ``authenticated_escalation_script``, not the shared ladder: the grow must
    # run ONCE (v2.1 "Do not replay partition writes") and a grow failure must
    # never be re-tried under the next password and then reported as that
    # password's auth failure. See that function.
    session.step(
        GROW_STEP_NAME,
        authenticated_escalation_script(_GROW_ROOT_FRAGMENT, client_password),
        timeout_s=GROW_TIMEOUT_S,
        timed=True,
    )
    return True


# Filesystem size, then the size of the whole block device the root filesystem
# sits on. Read-only: growing is the separate, gated ``grow-root-filesystem``
# step (``_GROW_ROOT_FRAGMENT``), and this script runs again after it as
# ``measure-geometry-after`` to show what the grow achieved. The block-device
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

#: The grow step's name in ``guest-preparation.json``. Deliberately NOT in
#: ``_RECLAMATION_STEP_NAMES``: it is advisory (see the module docstring).
GROW_STEP_NAME = "grow-root-filesystem"

# Upstream's ``_expand_linux_guest_volume`` (vendored
# ``desktop_env/providers/volume.py``), as one privileged fragment. The same
# discovery (``findmnt`` for the root source and fstype, ``lsblk PKNAME`` for
# the disk, ``/sys/class/block/<dev>/partition`` then ``lsblk PARTN`` for the
# partition number), the same install of ``cloud-guest-utils`` when ``growpart``
# is missing, and the same per-fstype resize. Differences, each on purpose:
#
# * It runs INSIDE the escalation ladder's ``sudo bash -c``, so there is no
#   ``sudo_cmd`` helper and the password is never embedded in the script body.
# * A failure EXITS NON-ZERO with a one-line ``grow=<reason>`` instead of
#   upstream's ``|| true``: the step is advisory either way, but the record must
#   say which link broke, or a fidelity gap is invisible.
# * ``growpart``'s ``NOCHANGE`` (the partition already fills the disk) is
#   success, which is what makes the fragment safe to run twice.
# * apt waits up to 60 s for the dpkg lock: a fresh Ubuntu guest runs
#   ``unattended-upgrades`` at boot, and upstream's bare ``apt-get`` would fail
#   on the lock instead of waiting it out. Its output is discarded so the
#   step's 200-character detail is the summary line, not apt's progress.
# * ``parted`` is not installed and ``partprobe`` not run: ``growpart`` informs
#   the kernel itself (``partx --update`` / ``resizepart``), and upstream treats
#   ``partprobe`` as optional (``|| true``).
# * No ``[ -b ]`` test: ``lsblk`` returning a parent disk is the same evidence
#   (it fails for anything that is not a partition of a block device) and keeps
#   the fragment exercisable against stubs.
# * The fstype and its resize tool are validated FIRST, before apt or
#   ``growpart`` touch anything -- v2.1's order: an unsupported filesystem must
#   not leave a partition table rewritten for a resize that can never follow.
_GROW_ROOT_FRAGMENT = (
    "src=$(findmnt -no SOURCE /) || { echo grow=unsupported root=unknown; exit 3; }; "
    "fstype=$(findmnt -no FSTYPE /); "
    'case "$fstype" in ext2|ext3|ext4) tool=resize2fs;; xfs) tool=xfs_growfs;; '
    'btrfs) tool=btrfs;; *) echo "grow=unsupported-fs fs=$fstype"; exit 6;; esac; '
    'command -v "$tool" >/dev/null 2>&1 || { echo "grow=no-resize-tool tool=$tool"; exit 6; }; '
    'dev=$(readlink -f -- "$src" 2>/dev/null || printf %s "$src"); '
    "name=${dev##*/}; "
    'disk=$(lsblk -ndo PKNAME "$dev" 2>/dev/null | head -n 1); '
    'part=$(cat "/sys/class/block/$name/partition" 2>/dev/null); '
    '[ -n "$part" ] || part=$(lsblk -ndo PARTN "$dev" 2>/dev/null | head -n 1); '
    'if [ -z "$disk" ] || [ -z "$part" ]; then echo "grow=unsupported dev=$dev"; exit 3; fi; '
    "installed=0; "
    "if ! command -v growpart >/dev/null 2>&1; then "
    "apt-get -qq -o DPkg::Lock::Timeout=60 update >/dev/null 2>&1; "
    "DEBIAN_FRONTEND=noninteractive apt-get install -y -qq -o DPkg::Lock::Timeout=60 "
    "--no-install-recommends cloud-guest-utils >/dev/null 2>&1; "
    "command -v growpart >/dev/null 2>&1 || { echo grow=no-growpart install=failed; exit 4; }; "
    "installed=1; fi; "
    'out=$(growpart "/dev/$disk" "$part" 2>&1); gprc=$?; '
    'case "$out" in *NOCHANGE*) state=nochange;; '
    "*) if [ $gprc -ne 0 ]; then echo grow=growpart-failed rc=$gprc; exit 5; fi; "
    "state=grown;; esac; "
    "udevadm settle >/dev/null 2>&1 || true; "
    # The fstype was validated above, so this ``case`` has no fall-through arm.
    'case "$fstype" in '
    'ext2|ext3|ext4) resize2fs "$dev" >/dev/null 2>&1;; '
    "xfs) xfs_growfs / >/dev/null 2>&1;; "
    "btrfs) btrfs filesystem resize max / >/dev/null 2>&1;; esac; "
    "rs=$?; "
    'echo "grow=$state fs=$fstype resize_rc=$rs growpart_installed=$installed"; '
    "exit $rs"
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


def _reclaim(session: _Session, client_password: str, retry_delays: Sequence[float]) -> None:
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
    #
    # Both are re-sent while the control server cannot answer them (see
    # ``HOLD_RETRY_ATTEMPTS`` for the measured case and why a hold may be
    # re-sent when nothing else may). The fallback still runs if the hold is
    # unreachable after its retries: it is the same hold by another name, and a
    # step that never landed is never silently dropped -- it stays in the record
    # and in ``blocking_steps``.
    held = session.step_retrying_unreachable(
        "hold-snap-auto-refresh",
        privileged("snap refresh --hold=forever"),
        delays=retry_delays,
        # The fallback and the clear may both still have to run.
        reserve_s=2 * COMMAND_TIMEOUT_S,
    )
    if held is None or held.returncode != 0:
        session.step_retrying_unreachable(
            "hold-snap-auto-refresh-fallback",
            privileged(f"snap set system refresh.hold={_FALLBACK_HOLD_UNTIL}"),
            delays=retry_delays,
            # Only the clear is left after the fallback.
            reserve_s=COMMAND_TIMEOUT_S,
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
    # THE PAYLOAD STAYS LAST. ``out`` holds one rung's combined stdout+stderr and
    # is printed BELOW, after the metadata line, as the ladder's final writes.
    # ``guest_supervisor.parse_output`` takes the LAST captured object that
    # matches the caller's expected shape, and that ordering is load-bearing: a
    # command's own ``print`` is its final statement, so its payload is the last
    # payload-shaped object in the capture, while sudo's prompt and any guest
    # warning land BEFORE it. Anything written after the payload -- a second
    # ``printf``, an echoed diagnostic, or an explicit stdout flush followed by a
    # later write -- would silently invert that ordering and let a shadow object
    # be read as the result, which is the "no restart" blindness this parser
    # exists to prevent. Keep this print last.
    lines.append("printf 'escalation=%s\\n' \"$tiers\"")
    lines.append("printf '%s\\n' \"$out\"")
    lines.append("exit $rc")
    return "; ".join(lines)


def authenticated_escalation_script(fragment: str, client_password: str) -> str:
    """Authenticate first, then run the fragment ONCE under the rung that worked.

    WHY NOT ``escalation_script``. That ladder moves to the next rung on ANY
    non-zero exit, so a fragment that fails for its own reasons is re-run under
    every remaining candidate, and the record keeps only the last rung's
    output. For the idempotent reclamation fragments that is tolerable and is
    left as it is (pre-existing behaviour; every campaign record reads
    ``escalation=agentless>supplied``, so it has never fired there). For the
    grow it is wrong twice over, both reproduced in review: a failed
    ``growpart`` was REPLAYED (twice on a NOPASSWD guest), which v2.1 forbids
    ("Do not replay partition writes", ``attempts=1``), and when the supplied
    value is one of upstream's defaults the grow's own ``grow=<reason>`` was
    overwritten by the next default's ``sudo: 1 incorrect password attempt``,
    so the record blamed a credential for a disk failure.

    This is v2.1's own ``sudo_cmd`` discipline ("Authenticate separately: a
    command failure is not a password failure"): ``sudo -n -v``, then each
    candidate through ``sudo -S -p '' -v``, same candidates and order as the
    ladder; the first that VALIDATES is the one the fragment runs under, once.
    The run re-feeds that candidate with ``-k`` rather than trusting the
    timestamp ``-v`` left: without a tty sudo keys the ticket on the parent
    PID, which a command substitution may or may not change (the hazard
    upstream's own comment warns about), and ``-k`` also guarantees the
    password line is consumed by sudo, never left on the fragment's stdin.

    THE RECORDED OUTPUT keeps the ladder's shape: ``escalation=<chain>`` (the
    rungs tried, ending on the one that authenticated), then the fragment's own
    output. When no rung authenticates the fragment never runs and the output
    says so -- an auth failure that cannot be mistaken for a grow failure, or
    the reverse. Each candidate appears once in the script, assigned to a shell
    variable and fed to ``sudo -S`` on stdin only.
    """

    quoted_fragment = _shell_quote(fragment)
    #: Tier labels are fixed vocabulary, never a value: they land in the report.
    candidates: list[tuple[str, str]] = []
    if client_password:
        candidates.append(("supplied", client_password))
    candidates.extend(("upstream-default", value) for value in _fallback_passwords(client_password))

    lines = [
        "rc=1",
        "out=''",
        "pw=''",
        "auth=''",
        "tiers='agentless'",
        "if sudo -n -v >/dev/null 2>&1; then auth=agentless; fi",
    ]
    for index, (tier, password) in enumerate(candidates):
        var = f"cand{index}"
        lines.append(
            f'if [ -z "$auth" ]; then tiers="$tiers>{tier}"; {var}={_shell_quote(password)}; '
            f"if printf '%s\\n' \"${var}\" | sudo -S -p '' -v >/dev/null 2>&1; "
            f'then auth={tier}; pw="${var}"; fi; fi'
        )
    # One invocation of the fragment, under whichever rung validated. With no
    # password (agentless) nothing at all is piped; with one, ``-k`` makes sudo
    # read it afresh (see the docstring).
    lines.append(
        'if [ -z "$auth" ]; then '
        "out='sudo: no candidate authenticated; fragment not run'; rc=1; "
        'elif [ -z "$pw" ]; then '
        f"out=$(sudo -n bash -c {quoted_fragment} 2>&1); rc=$?; "
        "else "
        f"out=$(printf '%s\\n' \"$pw\" | sudo -S -k -p '' bash -c {quoted_fragment} 2>&1); rc=$?; "
        "fi"
    )
    # Payload last, for the same reason as in ``escalation_script``.
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
