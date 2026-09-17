"""Shared LaunchAgent plist helpers: addressability, staleness, and the repair.

WHY THIS MODULE EXISTS
----------------------
Three of the four installers (``mobile``, ``tunnels``, ``browser_bridge``) wrote
their plist at install time and never looked at it again, so a plist written by
an older build kept running that older build's interpreter for as long as the
job existed. ``wakes`` is the one that already repaired itself, "idempotent by
CONTENT, not by existence"; this module is that idea lifted out of it so the
other three can share the parts that are the same — and, more importantly, so
the parts that are dangerous are written once:

- **The addressability guard.** ``launchctl`` has no notion of a sandbox: it
  always addresses the calling user's live session, whatever ``Path.home()``
  has been redirected to. A test that patches ``home`` to a tmpdir — the
  ordinary way to test an installer — would otherwise rewrite and restart the
  operator's REAL daemon. ``wakes/install.py`` documents the incident that put
  this guard in (a live unit pointed at a pytest tmpdir). The test is
  IDENTITY, not location: the plist path must be the one the passwd home
  produces, because a containment test fails open whenever a redirected home
  lands inside the real one (``TMPDIR`` under ``$HOME`` is not exotic) — see
  :func:`is_own_plist`.

- **The config-dir guard.** A plist that records a config dir (``wakes``,
  ``tunnels``) must not be rewritten to point at a store that is outside the
  real home: that is how a sandbox run plants a supervised unit pointed at a
  directory that vanishes when the sandbox ends.

- **Bootout + bootstrap, never ``kickstart -k``, after a rewrite.** MEASURED on
  macOS with a scratch label, ``Program`` = the branded hardlink and
  ``ProgramArguments[0]`` = a label: after rewriting the plist on disk,
  ``launchctl kickstart -k`` returned 0, started a NEW pid, and kept running
  the PREVIOUS argv — its marker file was never written, the old one still was.
  launchd restarts from the in-memory job definition, so a re-write followed by
  a kickstart silently repairs nothing. ``bootout`` + ``bootstrap`` on the same
  scratch label did pick up the new argv. ``wakes``'s kickstart repair is
  correct for the different case it handles (the plist is right and the job is
  stopped) and is left alone.

- **A ``bootout`` + ``bootstrap`` pair is a RACE, so it is written once.**
  Measured on macOS 2026-09-17: issued back to back, the bootstrap fails with
  ``Bootstrap failed: 5: Input/output error`` on 8 attempts out of 8 (3 of 4
  with a ~50 ms gap) and succeeds at ~500 ms, because launchd is still tearing
  the previous job down. The bootout has already succeeded at that point, so
  the failure leaves the daemon DOWN. All four installers open-coded this pair;
  :func:`reload_job` is now the only copy — bootout tolerating an absent job,
  a bounded wait for the label to be released, bounded bootstrap retries, and
  a check that the job is registered before anything claims success.

WHY THE REPAIR MUST RUN IN A NEW PROCESS
----------------------------------------
``lop update`` upgrading the wheel does not change the code its own process has
already imported. A repair run in-process would render the OLD plist shape and
therefore rewrite nothing — the pre-fix behaviour with more code. Everything in
this module is executed by a child started from the NEW wheel; see
``update.refresh_daemons_after_upgrade``.
"""

from __future__ import annotations

import logging
import os
import plistlib
import subprocess
import time
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import Literal

logger = logging.getLogger(__name__)

#: Where a user-level LaunchAgent lives. Both halves are needed by
#: :func:`is_own_plist`, which builds the expected path from the passwd home.
_PLIST_DIR = ("Library", "LaunchAgents")

PlistRefreshKind = Literal[
    "unsupported", "not-installed", "not-addressable", "current", "repaired", "failed"
]

ReloadOutcome = Literal["reloaded", "not-addressable", "failed"]

#: Timeout for ONE ``launchctl`` invocation. Every call this module makes is
#: answered in milliseconds; the bound exists only so a wedged ``launchctl``
#: cannot hang an install or the daemons-refresh child of ``lop update``.
_LAUNCHCTL_TIMEOUT_S = 20.0

#: How long to wait for launchd to actually RELEASE a label after ``bootout``.
#:
#: MEASURED on macOS (2026-09-17, ``gui/501``, scratch label): bootout returns
#: before the job is gone. A bootout-then-bootstrap pair issued back to back
#: failed with ``Bootstrap failed: 5: Input/output error`` on **8 attempts out
#: of 8**; with a ~50 ms gap it still failed **3 times out of 4**; it succeeded
#: at ~500 ms and again at 5 s, on the same plist that never changed. So the
#: EIO is the teardown race, not a broken plist — which is why the fix is to
#: wait for the label and then STILL retry (see the bootstrap budget below),
#: rather than to treat the first failure as final.
_LABEL_RELEASE_DEADLINE_S = 2.5

#: Poll spacing for that wait. The label usually stops resolving immediately on
#: an idle machine, so this is the granularity of an early exit, not a delay.
_LABEL_RELEASE_POLL_S = 0.05

#: Bootstrap retry budget for ONE reload, and the first backoff between tries.
#: Sized from the measurement above (~500 ms was enough at the worst spacing)
#: with an order of magnitude of headroom, while leaving the four-daemon
#: refresh child comfortably inside ``update._DAEMON_REFRESH_TIMEOUT_S`` even
#: on a machine where every launchctl call burns its whole budget.
_BOOTSTRAP_DEADLINE_S = 6.0
_BOOTSTRAP_BACKOFF_S = 0.1
_BOOTSTRAP_BACKOFF_CAP_S = 1.0


@dataclass(frozen=True)
class JobReload:
    """What one supervised job's ``bootout`` + ``bootstrap`` reload did.

    A ``launchctl`` pair run back to back is a race on macOS: the bootstrap
    lands while launchd is still tearing the previous job down and fails with
    ``Bootstrap failed: 5: Input/output error``, which leaves the daemon DOWN
    and, in the silent case, says nothing at all. Every installer used to issue
    exactly that pair inline (four copies), so every installer had the defect.

    ``detail`` is launchd's OWN stderr, never a paraphrase: it is what
    :func:`reload_failure` turns into the sentence that names the recovery
    command. ``attempts`` is how many bootstraps it took, so a machine that
    needed the retries is visible in a log rather than invisible.

    Never an exception: the only callers that matter are a ``lop update``
    refresh that has already succeeded, and an installer whose failure is a
    message rather than a traceback.
    """

    label: str
    outcome: ReloadOutcome
    detail: str = ""
    attempts: int = 0

    @property
    def ok(self) -> bool:
        return self.outcome == "reloaded"

    def as_refresh_failure(self, *, name: str, path: Path, recovery: str) -> PlistRefresh:
        """The ``lop update`` outcome for a reload that did not happen.

        The two failure shapes are NOT the same and must not be reported as
        one: ``not-addressable`` never reached launchd at all — the guard
        (:func:`is_own_plist`) refused before anything was booted out — so it
        gets the same silent non-event the per-daemon guards report, whereas a
        genuine failure DID boot the job out and goes through
        :func:`reload_failure`, whose sentence says the daemon is STOPPED and
        names the command that brings it back.
        """
        if self.outcome == "not-addressable":
            return PlistRefresh(name=name, kind="not-addressable")
        return reload_failure(name, path, recovery, self.detail)


@dataclass(frozen=True)
class PlistRefresh:
    """What a per-daemon repair did. Never an exception — see ``update.py``.

    ``name`` is the daemon's human name ("mobile", "browser bridge", "tunnel",
    "wakes supervisor") because this is what the upgrade summary prints; the
    caller owns the vocabulary, this type only carries it.

    ``unsupported`` is the honest answer on a platform with no LaunchAgent to
    repair (Linux, where the unit is re-read on every start) and on a machine
    with no ``launchctl``; it is deliberately silent, because a platform that
    cannot have this problem must not read as a failure on every upgrade.
    """

    name: str
    kind: PlistRefreshKind
    detail: str = ""

    def summary(self) -> str:
        """The upgrade-summary line, or ``""`` when there is nothing to say.

        Only a repair speaks. A daemon that is absent, unaddressable from here
        or already current is the normal state of a machine and printing a line
        for it would turn every upgrade into a list of non-events — the same
        reason the mobile refresh stays silent when it skips.
        """
        if self.kind == "repaired":
            return f"{self.name} daemon: refreshed a stale LaunchAgent and restarted it"
        return ""

    def warning(self) -> str:
        """The line for the upgrade summary's stderr, or ``""``.

        A failure DETAIL is a whole sentence rather than a token: the reload
        failures name the command that restores a stopped daemon, and that
        sentence is the point of the line, so it is printed as it stands.
        """
        if self.kind == "failed":
            return f"warning: {self.name} daemon was not refreshed: {self.detail}"
        return ""


def reload_failure(name: str, path: Path, recovery: str, error: str) -> PlistRefresh:
    """Outcome for a repair that rewrote the plist but could not reload the job.

    THE ONE FAILURE IN THE LADDER THAT LEAVES THE MACHINE CHANGED **AND** THE
    SERVICE DOWN: ``bootout`` succeeded, so the daemon the operator had is no
    longer running, and a bare ``launchctl`` error reads like a stale-file
    problem rather than a stopped service. The ``recovery`` command named here
    is that daemon's own installer (``lop mobile install``, ``lop browser
    install``, ``lop tunnel install``, ``lop wake install``), which rewrites
    the same plist and loads it again. One helper so all four installers cannot
    drift in how they say this.

    ``error`` is :attr:`JobReload.detail`, which carries launchd's own stderr
    rather than a summary of it: the sentence is only useful if the reader can
    see what launchd objected to.
    """
    return PlistRefresh(
        name=name,
        kind="failed",
        detail=(
            f"rewrote {path} but launchctl could not load it: {error} "
            f"— the daemon is now STOPPED; run `{recovery}` to reinstall it"
        ),
    )


def _text(value: object) -> str:
    """``str`` from whatever a ``launchctl`` wrapper captured.

    The helpers in this repo capture with ``text=True``, but that is a choice
    each caller makes and ``subprocess`` returns bytes without it — including
    from a test stub standing in for one. One adapter here is cheaper than a
    shape every reload caller has to remember.
    """
    if isinstance(value, bytes):
        return value.decode(errors="replace")
    return value if isinstance(value, str) else ""


def _launchctl_message(result: object) -> str:
    """launchd's own words for a failed call, NEVER discarded.

    ``stderr`` first (that is where launchd explains itself), then stdout, then
    the exit status — a bare "it failed" is the failure mode this whole module
    exists to delete, so there is always something to print.
    """
    message = (
        _text(getattr(result, "stderr", "")).strip() or _text(getattr(result, "stdout", "")).strip()
    )
    if message:
        return message
    return f"launchctl exited {getattr(result, 'returncode', '?')}"


def _launchctl(*args: str) -> subprocess.CompletedProcess[str]:
    """This module's own ``launchctl`` invocation.

    The default :func:`reload_job` runner. The installers each have a copy of
    this one-liner and pass their own, so a test that already stubs a module's
    helper intercepts every call the reload makes — no test-only seam, and no
    module silently reaching the real ``launchctl`` because a stub stopped
    covering it.
    """
    return subprocess.run(  # noqa: S603 — fixed argv, no shell
        ["launchctl", *args], capture_output=True, text=True, timeout=_LAUNCHCTL_TIMEOUT_S
    )


def _monotonic() -> float:
    """Indirection over ``time.monotonic`` so a test can exhaust a deadline."""
    return time.monotonic()


def _sleep(seconds: float) -> None:
    """Indirection over ``time.sleep``, for the same reason."""
    time.sleep(seconds)


def job_domain() -> str:
    """The per-user launchd domain every LaunchAgent in this module lives in.

    ``gui/<uid>`` is derived from the uid and NOT from ``$HOME``, which is the
    whole reason :func:`is_own_plist` is an identity test: launchd addresses
    the calling user's live session however the process has redirected its
    home. Supplying this domain explicitly is also what keeps the reload's
    probes addressable — ``launchctl print <plist path>`` is not a thing.
    """
    return f"gui/{os.getuid()}"


def _registered(target: str, run: Callable[..., object]) -> bool:
    """Whether launchd currently resolves ``<domain>/<label>``.

    ``print`` and only ``print``: it answers about the label launchd knows,
    and is what the reload waits on and verifies with. A job that has exited
    still prints (a real trap this codebase has already been bitten by — see
    ``wakes/install.supervisor_state``), which is exactly right here, where
    the question is "is the job registered", not "is it running".
    """
    return getattr(run("print", target), "returncode", 1) == 0


def _wait_for_label_release(
    target: str, run: Callable[..., object], *, deadline_s: float = _LABEL_RELEASE_DEADLINE_S
) -> bool:
    """Wait, bounded, for ``bootout`` to actually release the label.

    Returns whether the label went away within the deadline. The first probe
    is immediate, so a job that was never loaded (the ordinary "bootout an
    absent job" case) costs one call and no sleep. A label STILL resolving at
    the deadline is not an error here: the bootstrap below retries on its own
    budget and reports launchd's own words if it truly cannot load, which is
    the honest answer rather than a timeout invented by this module.
    """
    limit = _monotonic() + deadline_s
    while True:
        if not _registered(target, run):
            return True
        if _monotonic() >= limit:
            return False
        _sleep(_LABEL_RELEASE_POLL_S)


def reload_job(
    *,
    label: str,
    path: Path,
    runner: Callable[..., object] | None = None,
) -> JobReload:
    """Reload one LaunchAgent: ``bootout``, wait for release, ``bootstrap``.

    THE ONE PLACE THIS SEQUENCE IS WRITTEN. All four installers used to open
    code it inline and all four had the same defect (measured 2026-09-17):
    ``bootout`` then ``bootstrap`` back to back fails with ``Bootstrap
    failed: 5: Input/output error`` because launchd is still tearing the old
    job down, and since the bootout has ALREADY succeeded the failure leaves
    the daemon DOWN — silently, in the paths that discarded the result. Field
    cost: the post-upgrade plist refresh rewrote the tunnel plist, failed to
    load it back, and the operator's phone showed Cloudflare Error 1033 for
    ~7 hours while the connector was simply not running.

    The shape, and why each part is here:

    1. ``bootout`` by ``<domain>/<label>``, result IGNORED — an absent job is
       the ordinary case (a first install, a hand bootout, a reboot with the
       agent removed) and must not read as a failure.
    2. Wait, bounded, until ``print <domain>/<label>`` stops resolving, rather
       than assuming bootout is instantaneous. Addressed by domain and label,
       never by plist path: the path form answers a different question.
    3. ``bootstrap`` with bounded retries and backoff until a deadline, so the
       transient EIO above ends the attempt only when it is real.
    4. Verify the job is registered, and return launchd's own stderr either
       way so a failure can be reported as :func:`reload_failure`'s sentence
       naming the recovery.

    ``bootout`` + ``bootstrap`` rather than ``kickstart -k`` because the
    callers reload after REWRITING the plist: a kickstart restarts the job
    from launchd's in-memory definition and keeps running the old argv (see the
    module docstring's measurement). Kickstart repairs for the stopped-job
    case, where the plist is already correct, are deliberate and stay.

    Refuses before touching launchd unless ``path`` is the plist the real
    passwd home produces for ``label`` (:func:`is_own_plist`): a redirected
    ``HOME`` would otherwise reload a REAL unit from a sandbox. This is a
    precondition rather than a convenience, so a caller that forgot the guard
    cannot reach launchd through this function.

    ``runner`` is the caller's own ``launchctl`` helper — see
    :func:`_launchctl` for why it is threaded through instead of hard-coded.
    """
    if not is_own_plist(path, label):
        return JobReload(
            label=label,
            outcome="not-addressable",
            detail=f"{path} is not the LaunchAgent the real home owns for {label}",
        )
    run = runner if runner is not None else _launchctl
    domain = job_domain()
    target = f"{domain}/{label}"

    # 1. Tolerate an absent job: `launchctl bootout` on one exits non-zero, and
    # that is not a failure to report.
    run("bootout", target)
    # 2. Wait for launchd to let go of the label (bounded; not fatal if not).
    _wait_for_label_release(target, run)

    # 3. Bootstrap until the deadline. Every attempt is retried, not only one
    # that looks transient: launchd's messages vary by version, and a deadline
    # is a bound this module can defend where a message match is a guess. The
    # cost of being wrong is 6 s on a permanently broken plist, in exchange for
    # never again reading a teardown race as a finished reload.
    limit = _monotonic() + _BOOTSTRAP_DEADLINE_S
    backoff = _BOOTSTRAP_BACKOFF_S
    attempts = 0
    while True:
        attempts += 1
        result = run("bootstrap", domain, str(path))
        if getattr(result, "returncode", 1) == 0:
            break
        failure = _launchctl_message(result)
        if _monotonic() >= limit:
            return JobReload(
                label=label,
                outcome="failed",
                detail=f"bootstrap failed after {attempts} attempts: {failure}",
                attempts=attempts,
            )
        logger.debug("launchctl bootstrap of %s failed (%s); retrying", target, failure)
        _sleep(backoff)
        backoff = min(backoff * 2, _BOOTSTRAP_BACKOFF_CAP_S)

    # 4. A zero exit is launchd's claim, not evidence: `bootstrap` has been
    # observed to return 0 for a job it then does not resolve. The caller
    # claims success to the operator, so the claim is checked.
    if not _registered(target, run):
        return JobReload(
            label=label,
            outcome="failed",
            detail=(
                f"bootstrap reported success but {target} is not registered"
                f" ({_launchctl_message(result)})"
            ),
            attempts=attempts,
        )
    return JobReload(label=label, outcome="reloaded", attempts=attempts)


def recorded_install_prefix(data: object) -> Path | None:
    """The install prefix the plist's interpreter lives in, or ``None``.

    Reads the interpreter path from either plist shape: the branded ``Program``
    (``<prefix>/bin/Local Operator``) or, on a plist written before that key
    existed, ``ProgramArguments[0]`` when it is a path rather than a label.
    Used to answer "is this the SAME installation?" — see
    :func:`update.daemons_refresh_command`, where the repair may change how a
    daemon is NAMED but must never change WHICH INSTALL it runs.

    ``None`` means "cannot tell", which callers must treat as no objection: a
    plist this code cannot read is not evidence of a different install.
    """
    if not isinstance(data, dict):
        return None
    program = data.get("Program")
    candidates: list[str] = []
    if isinstance(program, str):
        candidates.append(program)
    argv = data.get("ProgramArguments")
    if isinstance(argv, list) and argv and isinstance(argv[0], str):
        candidates.append(argv[0])
    for candidate in candidates:
        if not candidate.startswith("/"):
            # A label, not a path: the branded shape carries the image in
            # ``Program``, and anything else here is not a candidate prefix.
            continue
        prefix = Path(candidate).parent.parent
        return prefix
    return None


def real_home() -> Path | None:
    """The uid's passwd home, or ``None`` when it cannot be read.

    NOT ``Path.home()``, and that difference is the whole guard: ``Path.home()``
    reads ``$HOME``, so an isolated run would compare its own redirected home
    against itself and conclude it is the real one.
    """
    import pwd

    try:
        return Path(pwd.getpwuid(os.getuid()).pw_dir).resolve()
    except (KeyError, OSError):
        return None


def is_own_plist(path: Path, label: str) -> bool:
    """Whether ``launchctl``'s answer about ``path`` would be about THIS run.

    True only when ``path`` is the plist path the real passwd home produces for
    ``label``. A redirected ``HOME`` therefore refuses: the file half of an
    installer stays fully testable, while the half that reaches outside the
    process declines to run. Unreadable passwd entry degrades to False, which
    is the safe direction — the cost of a wrongly-skipped repair is a stale
    plist; the cost of a wrongly-executed one is the operator's live daemon.
    """
    home = real_home()
    if home is None:
        return False
    try:
        expected = (home.joinpath(*_PLIST_DIR) / f"{label}.plist").resolve()
        return Path(path).resolve() == expected
    except (OSError, ValueError):
        return False


def config_lives_in_real_home(config_dir: Path) -> bool:
    """Whether a unit supervising ``config_dir`` would outlive this process.

    CONTAINMENT here, not the identity test :func:`is_own_plist` uses, and the
    difference is deliberate: the config dir is an ordinary path the user may
    legitimately place anywhere under their home, so only dirs OUTSIDE it are
    the sandbox shape (``/tmp``, a throwaway home). A unit pointed at one of
    those is a live launchd job watching a store that is deleted when the
    sandbox ends.
    """
    home = real_home()
    if home is None:
        return False
    try:
        return Path(config_dir).resolve().is_relative_to(home)
    except (OSError, ValueError):
        return False


def load(path: Path) -> dict[str, object] | None:
    """The parsed plist, or ``None`` for absent/corrupt/unreadable.

    One answer for all three on purpose: every caller wants "can I read what is
    there?", and an unreadable plist is a stale one.
    """
    try:
        parsed = plistlib.loads(Path(path).read_bytes())
    except Exception:  # noqa: BLE001 — a corrupt plist is "no evidence", not a crash
        return None
    return parsed if isinstance(parsed, dict) else None


def arg_value(plist: dict[str, object] | None, flag: str) -> str | None:
    """The value following ``flag`` in ``ProgramArguments``, if present.

    Reads argv for its ARGUMENTS, not for an interpreter: the plist's element 0
    is a label in the current shape and an interpreter path in the pre-branding
    one, which is exactly why nothing may parse it positionally (see
    ``procname.launchd_job``). ``--flag=value`` is accepted as well as
    ``--flag value``.
    """
    if not plist:
        return None
    argv = plist.get("ProgramArguments")
    if not isinstance(argv, list):
        return None
    for index, item in enumerate(argv):
        if not isinstance(item, str):
            continue
        if item == flag:
            following = argv[index + 1] if index + 1 < len(argv) else None
            return following if isinstance(following, str) else None
        if item.startswith(f"{flag}="):
            return item.partition("=")[2] or None
    return None


def int_arg(plist: dict[str, object] | None, flag: str, default: int) -> int:
    """``arg_value`` as an int, falling back to ``default``.

    A repair must never CHANGE a setting: a daemon installed on a non-default
    port is repaired on THAT port, which is only knowable from the plist being
    replaced. Anything unparseable keeps the default rather than inventing one.
    """
    raw = arg_value(plist, flag)
    if raw is None:
        return default
    try:
        return int(raw)
    except ValueError:
        return default


def config_dir_from_plist(plist: dict[str, object] | None) -> Path | None:
    """The config dir a plist's own ``EnvironmentVariables`` record, if any.

    Read from the file being repaired rather than from this process's ambient
    environment: the repair's job is to bring a unit up to date IN PLACE, not to
    migrate it to whatever store happens to be current here.
    """
    from local_operator.paths import CONFIG_DIR_ENV

    if not plist:
        return None
    environment = plist.get("EnvironmentVariables")
    if not isinstance(environment, dict):
        return None
    value = environment.get(CONFIG_DIR_ENV)
    return Path(value) if isinstance(value, str) and value else None


def rewrite_if_stale(*, name: str, path: Path, rendered: dict[str, object]) -> PlistRefresh:
    """Rewrite ``path`` when it does not already say what ``rendered`` says.

    The caller has ALREADY established that it may act on this path
    (:func:`is_own_plist`) and that the store it names is durable
    (:func:`config_lives_in_real_home`); this function only compares content and
    writes. It never restarts anything — the per-daemon module owns that,
    because the restart command differs per unit.

    Never raises: writing is done in the caller's try/except too, but a
    permission error on ``~/Library/LaunchAgents`` must not escape from here
    either, since the only caller that matters is an upgrade that has already
    succeeded.
    """
    if not path.exists():
        return PlistRefresh(name=name, kind="not-installed")
    current = load(path)
    if current == rendered:
        return PlistRefresh(name=name, kind="current")
    try:
        # ``write_bytes`` over an existing file keeps its mode, which is the
        # mode the installer that wrote it chose (0600 for the tunnel plist).
        # Nothing here chmods: a repair must not change permissions it was not
        # asked about.
        path.write_bytes(plistlib.dumps(rendered))
    except Exception as exc:  # noqa: BLE001 — a stale plist is not a failure
        logger.debug("could not rewrite %s", path, exc_info=True)
        return PlistRefresh(name=name, kind="failed", detail=f"could not rewrite {path}: {exc}")
    return PlistRefresh(name=name, kind="repaired")
