"""Install-on-demand for the wake supervisor — the chokepoint, not yet the
installer.

There is exactly one writer of schedule state, ``Session._persist_wake_schedules``,
so there is exactly one place to ask "does something outside this process
now need to exist to fire these?" That question is this function. It is
called after every non-empty persist, idempotently and best-effort: the
persist has already succeeded by the time it runs, and nothing it does (or
fails to do) may change that.

The supervisor it installs is :mod:`local_operator.wakes.supervisor`, run as
a LaunchAgent on macOS, a ``systemd --user`` service on Linux and a Task
Scheduler task on Windows, and shaped after ``local_operator.mobile.install``.
``KeepAlive: {SuccessfulExit: False}`` is the load-bearing key on macOS: the
supervisor exits 0 when the wake index empties, and that setting is what lets
a FINISHED supervisor stay down while a CRASHED one restarts. The same
distinction is spelled ``Restart=on-failure`` in the systemd unit and is
carried by ``RestartOnFailure`` plus the periodic trigger on Windows. The next
persist calls this hook again and brings it back.

Install-on-demand rather than install-at-setup, because the cost only makes
sense once there is something to supervise: a user who never schedules a wake
never gets the process.

**"No installer for this platform" and "wakes only fire while a session is
open" used to be the answer on Linux and Windows, and that is the whole point
of ``lop wake``.** Nothing ran when no TUI was open, so a scheduled wake fired
only when a human next opened that session — silently, with no error anywhere.
All three platforms now install a supervisor; :func:`is_supported` answers
``False`` only where there is genuinely no user-level supervisor to install
into (see :mod:`local_operator.supervisors`).

**"Installed" now means RUNNING, and that change fixed a permanent miss.**
The hook used to answer "is it installed?" with ``launchctl print`` returning
0, which is true of a job launchd merely *knows about*. A job that has exited
still prints, still returns 0, and reports ``state = not running``, measured
on a scratch label:

.. code-block:: text

    $ launchctl print gui/501/com.local-operator.waketest ; echo $?
    state = not running
    runs = 1
    0

Combined with self-retirement that is a permanent hole: the supervisor exits
0 on an empty index, the next persist writes an index entry and asks to
install, the hook says "already installed" because the dead job still prints,
and nothing is running to fire the wake. ``KeepAlive`` does not help — a
successful exit is exactly what it is configured not to restart. The wake
then surfaces only when a human next opens the session, with no log line
anywhere. :func:`supervisor_state` is the fix: it parses the print output for
a live pid/state rather than trusting the exit code, and
:func:`ensure_supervisor_installed` REPAIRS what it finds (``kickstart`` when
loaded-but-stopped, ``bootstrap`` when absent).
"""

from __future__ import annotations

import logging
import os
import plistlib
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path

from local_operator import launchd, procname, supervisors
from local_operator.paths import CONFIG_DIR_ENV
from local_operator.paths import config_dir as ambient_config_dir

logger = logging.getLogger(__name__)

#: LaunchAgent label, matching ``mobile.install``'s spelling so the two
#: supervised units read as siblings in ``launchctl list``.
LABEL = "com.local-operator.wakes"

#: Linux user unit and timer. Deliberately NOT suffixed per config root, for
#: the same reason ``LABEL`` is not: the unit RECORDS the store it supervises in
#: ``EnvironmentVariables``/``Environment=``, so a second store retargets the
#: one supervised unit rather than shadowing it — exactly how the plist behaves.
SYSTEMD_UNIT = "local-operator-wakes.service"
SYSTEMD_TIMER = "local-operator-wakes.timer"

#: Task Scheduler task name (Windows), same reasoning.
TASK_NAME = "Local Operator wake supervisor"

#: Reported when the platform has no installer this hook knows.
UNSUPPORTED_REASON = "no supervisor installer for this platform"


def plist_path() -> Path:
    """Where THIS platform's supervisor registration lives.

    Named for the launchd plist because that was the only platform it
    addressed; what it means is "the registration file", and the three callers
    that matter (this installer, ``uninstall``, and the ``lop wake status``
    lines that report the path) all mean that too. On Windows Task Scheduler
    keeps its registration in its own store — the registry — so the answer there
    is the definition file this installer wrote, which exists exactly when we
    registered a task.
    """
    kind = supervisors.supervisor()
    if kind == supervisors.SYSTEMCTL:
        return supervisors.systemd_unit_path(SYSTEMD_UNIT)
    if kind == supervisors.SCHTASKS:
        return task_record_path(ambient_config_dir())
    return Path.home() / "Library" / "LaunchAgents" / f"{LABEL}.plist"


def systemd_timer_path() -> Path:
    return supervisors.systemd_unit_path(SYSTEMD_TIMER)


def task_record_path(config_dir: Path) -> Path:
    """Our own copy of the task definition registered for ``config_dir``.

    Task Scheduler stores its copy in the registry and ``schtasks /Query`` is
    the authoritative question, so this is a RECORD: it makes "is anything
    registered for this store, and with which command?" answerable without a
    subprocess, and it is what ``plist_path()`` answers with on Windows.
    """
    return config_dir / "supervisor" / "wakes-task.xml"


def log_path(config_dir: Path) -> Path:
    return config_dir / "logs" / "wake-supervisor.log"


#: How often launchd re-runs the supervisor on its own (seconds).
#:
#: This is a bounded self-heal, and it deliberately SOFTENS the invariant the
#: rest of this module documents — "a machine with no wakes left runs no
#: supervisor at all" is now "a machine with no wakes left runs the supervisor
#: briefly every 15 minutes, where it reads an empty index and exits 0 again".
#: That reversal is bought knowingly, because the alternative is the permanent
#: miss above: every other repair path in this file needs SOMETHING to call it
#: (a persist, a ``lop wake status``), and the failure mode is precisely the
#: one where nothing is left running to make that call. ``StartInterval`` is
#: the only mechanism that does not depend on a live process.
#:
#: Safe, and measured rather than assumed (scratch label, 20 s job, 5 s
#: interval): launchd started runs at +0 s, +25 s and +51 s — it does NOT
#: start a second instance while the job runs, it waits for the current one to
#: exit. And it DOES re-run a job that exited 0, which ``KeepAlive:
#: {SuccessfulExit: False}`` alone would not.
#:
#: 900 s because it must be well under the shortest realistic wake cadence
#: (the live machine's tightest recurring wake is 20 min) so a resurrected
#: supervisor still catches the next occurrence, while an idle wake-free
#: machine pays only ~96 short-lived reads of an empty directory a day.
SELF_HEAL_INTERVAL_S = 900


def is_supported() -> bool:
    """Whether this machine has a user supervisor this hook can install into.

    Binary-guarded, not platform-guarded (see :mod:`local_operator.supervisors`):
    a Linux without systemd is a real machine with no user manager, and a
    ``False`` here is what makes ``lop wake status`` say "wakes fire only while
    a session is open" instead of pretending something supervises them.
    """
    return supervisors.supervisor() is not None


def render_systemd(config_dir: Path) -> str:
    """The Linux user unit — the ``KeepAlive`` contract, in systemd's words.

    Three keys are load-bearing and each replaces one launchd setting:

    * ``Restart=on-failure`` — the ``KeepAlive{SuccessfulExit: false}``
      analogue, and NOT ``Restart=always``: the supervisor exits 0 when the
      index empties, and that exit has to STICK or an empty-index machine runs
      a supervisor forever.
    * ``Environment=LOCAL_OPERATOR_CONFIG_DIR=`` — the plist's
      ``EnvironmentVariables``: the store is part of the contract, so a second
      profile (or a test) supervises its own store.
    * the separate ``.timer`` (:func:`render_systemd_timer`) — the
      ``StartInterval`` self-heal, which is the one repair that needs no live
      caller.
    """
    image = procname.supervised_image() or Path(sys.executable)
    return supervisors.render_systemd_unit(
        description="Local Operator wake supervisor",
        exec_start=f"{image} -m local_operator.wakes.supervisor",
        post_lines=[
            f"Environment=LOCAL_OPERATOR_CONFIG_DIR={config_dir}",
            *supervisors.output_redirect_lines(log_path(config_dir)),
        ],
    )


def render_systemd_timer() -> str:
    """The ``.timer`` that re-runs the supervisor every ``SELF_HEAL_INTERVAL_S``.

    ``OnUnitInactiveSec`` and not ``OnCalendar``: the semantic being reproduced
    is launchd's ``StartInterval``, which (measured) does NOT start a second
    instance while the job runs — it waits for the current one to exit.
    ``systemd`` behaves the same way for an ``OnUnitInactiveSec`` timer, and the
    timer is ``WantedBy=timers.target`` so enabling it is enough. The store lives
    in the SERVICE unit, not here: the timer only decides when to start it.
    """
    return supervisors.render_systemd_timer(
        description="Local Operator wake supervisor self-heal",
        unit=SYSTEMD_UNIT,
        interval_seconds=SELF_HEAL_INTERVAL_S,
    )


def render_task_xml(config_dir: Path) -> str:
    """The Windows task: logon start, restart-on-failure, 15-minute self-heal.

    ``interval_minutes`` is the ``StartInterval``/``OnUnitInactiveSec`` analogue
    and is the reason the wake supervisor ports to Windows at all: a wake armed
    on a machine where nothing is running needs a mechanism that does not depend
    on a live caller.
    """
    image = procname.supervised_image() or Path(sys.executable)
    return supervisors.render_task_xml(
        description="Local Operator wake supervisor (fire due wakes)",
        image=str(image),
        argv=["-m", "local_operator.wakes.supervisor"],
        environment={CONFIG_DIR_ENV: str(config_dir)},
        log=log_path(config_dir),
        interval_minutes=max(1, SELF_HEAL_INTERVAL_S // 60),
        user_id=supervisors.current_user_id(),
    )


def _launchd_is_addressable() -> bool:
    """Whether this process may bootstrap into the REAL user's launchd domain.

    ``launchctl`` has no notion of a sandbox: it always addresses the calling
    user's live session, whatever ``Path.home()`` has been redirected to. A
    test that patches ``home`` to a tmpdir — the ordinary way to test an
    installer — would therefore write a harmless plist and then bootstrap a
    REAL supervised unit into the developer's session, pointed at a pytest
    tmpdir that is deleted moments later. That happened during development:
    ``launchctl print gui/501/com.local-operator.wakes`` showed a live unit
    whose plist path was under ``/private/var/folders/…/pytest-of-damian/``.

    So the plist is written wherever ``plist_path()`` says, but launchd is
    only ADDRESSED when that path is the one the real passwd home produces.
    The file half of the installer stays fully testable; the half that reaches
    outside the process refuses to run under a redirected home.

    IDENTITY, NOT LOCATION. This asked ``is_relative_to(real_home)`` until
    round 1 (R4) showed it fails OPEN whenever a redirected home lands inside
    the real one — `TMPDIR` set under ``$HOME`` is not exotic (it is how you
    avoid ``/var/folders`` cleanup races), and it makes pytest's ``tmp_path``,
    and therefore a patched ``Path.home()``, satisfy a containment test. That
    re-arms precisely the incident above, and no test using ``tmp_path`` could
    catch it because ``tmp_path`` follows ``TMPDIR`` too.

    Comparing against the path BUILT from the passwd entry closes it: a
    redirected home produces a different path wherever it points, so the only
    way to satisfy this is to genuinely be the real home.
    """
    # The identity test itself lives in :mod:`local_operator.launchd`, because
    # the other three daemon installers now need exactly it and a guard whose
    # reasoning is this sharp should exist once. The reasoning above stays here:
    # it records the incident that put the guard in.
    return launchd.is_own_plist(plist_path(), LABEL)


def _config_lives_in_real_home(config_dir: Path) -> bool:
    """Whether the supervised unit would point at a store that outlives us.

    A unit supervising a config dir under ``/tmp`` or a sandbox home watches
    a store that is deleted when the sandbox ends — a live launchd unit with
    a corpse for a config. Containment under the passwd home is the right
    test HERE (not the identity test ``_launchd_is_addressable`` uses) because
    the config dir is an ordinary path the user may legitimately place
    anywhere under their home; only dirs OUTSIDE it are the sandbox shape.
    """
    # Shared with the other three installers; see :mod:`local_operator.launchd`.
    return launchd.config_lives_in_real_home(config_dir)


def render_plist(config_dir: Path) -> dict[str, object]:
    """The whole supervised-unit plan in one pure function.

    Every consumer (install, tests) reads this same rendering, so what the
    tests assert is what launchd is handed.
    """
    return {
        "Label": LABEL,
        # ``launchd_job`` rather than ``launchd_program``: ``Program`` carries
        # the branded interpreter image and ``ProgramArguments[0]`` this
        # supervisor's role label, so the four supervised daemons stop reading
        # as one indistinguishable row. macOS names this background item by the
        # basename of ``ProgramArguments[0]``, so a bare ``sys.executable`` is
        # what makes installing a supervised unit notify 'python3 is running in
        # the background'. The trade-off that shape accepts is recorded in
        # ``procname.launchd_job``; with no link to plant this is byte-for-byte
        # the plist this function wrote before. On a machine with the generation
        # layout ``Program`` is the stable shim instead (see
        # ``procname.supervised_image``) and the label is unchanged.
        **procname.launchd_job(
            "local_operator.wakes.supervisor",
            label=procname.branded_argv0(procname.LABEL_WAKES),
        ),
        "RunAtLoad": True,
        # SELF-RETIREMENT, and the reason this key is not optional: the
        # supervisor exits 0 when the index empties. Keying restarts on
        # unsuccessful exit only is what makes that exit STICK, so a machine
        # with no wakes left runs no supervisor at all. A plain KeepAlive:true
        # would restart it forever against an empty index.
        "KeepAlive": {"SuccessfulExit": False},
        # The bounded self-heal, and the one repair that needs no live caller.
        # See SELF_HEAL_INTERVAL_S for why this softens the self-retirement
        # invariant directly above, and for the measurement showing launchd
        # will not start a second instance while one is running.
        "StartInterval": SELF_HEAL_INTERVAL_S,
        "StandardOutPath": str(log_path(config_dir)),
        "StandardErrorPath": str(log_path(config_dir)),
        # The config dir is part of the contract: the supervisor reads the
        # index under it, and a test or a second profile must be able to run
        # its own supervisor against its own store.
        "EnvironmentVariables": {"LOCAL_OPERATOR_CONFIG_DIR": str(config_dir)},
    }


def _domain() -> str:
    return f"gui/{os.getuid()}"


def _launchctl(*args: str) -> subprocess.CompletedProcess[str]:
    return subprocess.run(  # noqa: S603 — fixed argv, no shell
        ["launchctl", *args], capture_output=True, text=True, timeout=15
    )


@dataclass(frozen=True)
class SupervisorState:
    """What launchd actually has, as opposed to what the plist says.

    ``loaded`` is "launchd knows this label"; ``running`` is "a process is
    alive behind it". The two disagree exactly in the case this module exists
    to fix, and that is why they are separate fields rather than one boolean.
    ``pid`` is present only when running; ``detail`` carries the state word
    for the status surface.

    ``verifiable`` is a THIRD state, and it is not a technicality: launchd has
    no sandbox, so a probe run against an isolated store still addresses the
    real user's domain and answers about the real user's supervisor. Reporting
    that as this store's supervisor is the same class of lie this module
    exists to remove — it tells someone their wakes are supervised while the
    running process watches a different store entirely. When ``verifiable`` is
    False, ``loaded``/``running``/``pid`` say nothing about the store that was
    asked about, and a caller must not render them.
    """

    loaded: bool
    running: bool
    pid: int | None = None
    detail: str = ""
    verifiable: bool = True


#: ``state = <word>`` values that POSITIVELY mean "not running". Deliberately
#: an allowlist of stopped states rather than "anything that is not
#: 'running'": launchd's vocabulary is not ours to enumerate exhaustively, and
#: the cost of the two errors is asymmetric. Reading an unknown state as
#: stopped would kickstart a healthy supervisor (and, worse, do it on the
#: persist path, every persist, forever); reading it as running only means a
#: repair we would have liked does not happen, which is the status quo this
#: change improves on. So the parse FAILS SAFE: unparseable output is treated
#: as running.
_STOPPED_STATES = frozenset({"not running", "exited", "stopped", "waiting"})


def _parse_supervisor_state(stdout: str) -> tuple[bool, int | None, str]:
    """``(running, pid, detail)`` from one ``launchctl print`` body.

    Pure so the parse is testable against recorded launchctl output without a
    subprocess; see ``tests/unit/wakes/test_install.py``.
    """
    pid: int | None = None
    state = ""
    for raw in stdout.splitlines():
        line = raw.strip()
        # `pid = 47545` is the strongest signal launchd gives: it is printed
        # only while a process is alive behind the label.
        if line.startswith("pid =") and pid is None:
            value = line.partition("=")[2].strip()
            if value.isdigit():
                pid = int(value)
        elif line.startswith("state =") and not state:
            state = line.partition("=")[2].strip().lower()
    if pid is not None:
        return True, pid, state or "running"
    if state in _STOPPED_STATES:
        return False, None, state
    # No pid AND no state we recognise as stopped: fail safe (see
    # _STOPPED_STATES). `detail` still carries whatever was read so a status
    # reader is not told a confident lie.
    return True, None, state


def supervisor_state(config_dir: Path) -> SupervisorState:
    """Whether a supervisor process is actually alive FOR ``config_dir``.

    ``config_dir`` is REQUIRED rather than implied, because the label is
    global while the store is not. Both installer guards are applied here, at
    the probe itself, so no caller can accidentally report the operator's live
    supervisor as the supervisor of a sandboxed store. That exact wrong answer
    was produced during validation: an isolated run printed
    ``running (pid 47545)`` — the real LaunchAgent, watching the real store,
    while the store under test was in a tmpdir.

    ONE ``launchctl print``, parsed \u2014 not a print plus a second probe. This
    runs on the persist path (every wake write goes through
    :func:`ensure_supervisor_installed`), so a second subprocess here would be
    a cost paid by every scheduling operation on the machine.
    """
    kind = supervisors.supervisor()
    if kind is None:
        return SupervisorState(
            loaded=False, running=False, verifiable=False, detail=UNSUPPORTED_REASON
        )
    if kind == supervisors.SYSTEMCTL:
        return _systemd_supervisor_state(config_dir)
    if kind == supervisors.SCHTASKS:
        return _task_supervisor_state(config_dir)
    if not _launchd_is_addressable() or not _config_lives_in_real_home(config_dir):
        # Same two guards the installer uses to decide whether it may ACT;
        # asking is subject to them for the same reason, because the answer
        # would be about someone else's store.
        return SupervisorState(
            loaded=False,
            running=False,
            verifiable=False,
            detail="this store is outside the real home; launchd cannot supervise it",
        )
    result = _launchctl("print", f"{_domain()}/{LABEL}")
    if result.returncode != 0:
        # Not loaded at all. This is the only honest "no" launchctl gives by
        # exit code; a loaded-but-dead job also returns 0 (module docstring).
        return SupervisorState(loaded=False, running=False, detail="not loaded")
    running, pid, detail = _parse_supervisor_state(result.stdout)
    return SupervisorState(loaded=True, running=running, pid=pid, detail=detail)


#: ActiveState words that POSITIVELY mean "not serving", for the same
#: asymmetric-cost reason as ``_STOPPED_STATES`` above, plus one MEASURED case:
#: ``activating`` with ``auto-restart`` is what systemd shows while it retries a
#: CRASHING unit (observed on a real systemd 255 crash loop), and a unit systemd
#: is still retrying is not a supervisor that can fire a wake. A `start` on such
#: a unit is a no-op — systemd merges the job into the pending one — so treating
#: it as stopped costs nothing and stops a crash loop reading as "running".
_SYSTEMD_STOPPED_STATES = frozenset({"inactive", "failed", "deactivating", "activating"})


def _parse_systemd_state(stdout: str) -> SupervisorState:
    """``SupervisorState`` from one ``systemctl --user show`` body.

    Pure so the parse is testable against recorded systemctl output without a
    subprocess. ``LoadState=not-found`` is the only honest "systemd has never
    heard of this unit"; everything else it prints describes a unit it knows,
    and a unit whose process exited 0 (the supervisor's self-retirement) is
    exactly the ``loaded but not running`` state the installer has to repair.
    """
    fields: dict[str, str] = {}
    for raw in stdout.splitlines():
        key, _, value = raw.partition("=")
        if key.strip():
            fields[key.strip()] = value.strip()
    active = fields.get("ActiveState", "").lower()
    sub = fields.get("SubState", "")
    main = fields.get("MainPID", "0")
    pid = int(main) if main.isdigit() and main != "0" else None
    detail = f"{active}/{sub}" if sub else active
    if fields.get("LoadState", "") == "not-found":
        return SupervisorState(loaded=False, running=False, detail="not loaded")
    if active in _SYSTEMD_STOPPED_STATES:
        return SupervisorState(loaded=True, running=False, detail=detail)
    if active == "active" and pid is not None:
        return SupervisorState(loaded=True, running=True, pid=pid, detail=detail)
    # Unknown vocabulary: fail safe (see _SYSTEMD_STOPPED_STATES).
    return SupervisorState(loaded=True, running=True, pid=pid, detail=detail or "unknown")


def _systemd_supervisor_state(config_dir: Path) -> SupervisorState:
    """The Linux probe: one ``systemctl --user show``.

    A user manager that cannot be reached at all (no D-Bus, a plain SSH login
    without lingering) is ``verifiable=False`` and NOT "not loaded" — the
    difference matters, because the status surface renders the first as "cannot
    be verified for this store" and the second as "nothing is installed", and
    only one of those is actionable by installing something.
    """
    if not supervisors.systemd_unit_is_addressable(SYSTEMD_UNIT) or not _config_lives_in_real_home(
        config_dir
    ):
        return SupervisorState(
            loaded=False,
            running=False,
            verifiable=False,
            detail="this store is outside the real home; the user manager cannot supervise it",
        )
    shown = supervisors.systemctl_user(
        "show", "--property=LoadState,ActiveState,SubState,MainPID", SYSTEMD_UNIT
    )
    if shown.returncode != 0:
        text = (shown.stderr or shown.stdout or "").strip()
        if supervisors.is_bus_failure(text):
            return SupervisorState(
                loaded=False,
                running=False,
                verifiable=False,
                detail=supervisors.translate_systemctl_error(text),
            )
        return SupervisorState(loaded=False, running=False, detail="not loaded")
    return _parse_systemd_state(shown.stdout)


def _task_supervisor_state(config_dir: Path) -> SupervisorState:
    """The Windows probe: one ``schtasks /Query``.

    ``running`` comes from Task Scheduler's own status word. No pid is available
    through ``schtasks``, so ``pid`` stays ``None`` — reporting a guess would be
    worse than reporting nothing, and the field is optional by design.
    """
    if not supervisors.task_scheduler_is_addressable(config_dir):
        return SupervisorState(
            loaded=False,
            running=False,
            verifiable=False,
            detail="this store is outside the real profile; Task Scheduler cannot supervise it",
        )
    registered, running, detail = supervisors.task_state(TASK_NAME)
    if not registered:
        return SupervisorState(loaded=False, running=False, detail=detail or "not loaded")
    return SupervisorState(loaded=True, running=running, pid=None, detail=detail or "ready")


@dataclass(frozen=True)
class InstallOutcome:
    """What the hook did. ``installed`` is "a supervisor is now in place"
    (freshly installed OR already present); ``reason`` explains a False."""

    installed: bool
    reason: str = ""


#: Retained so a caller pinned to the stub's vocabulary still resolves. The
#: hook no longer reports it — kept because it is part of the published
#: surface this module shipped with, and removing a name is a separate
#: decision from filling in the implementation behind it.
NOT_AVAILABLE_REASON = "supervisor not yet available"


def ensure_supervisor_installed(config_dir: Path) -> InstallOutcome:
    """Make sure the wake supervisor is installed for ``config_dir``.

    Contract (binding on the real implementation, not just the stub):

    - **Idempotent.** Called after every persist; an already-installed
      supervisor is a cheap check, never a reinstall.
    - **Never raises.** The caller is the wake persist path; an installer
      failure is logged and reported through the outcome, never propagated.
      The persist has already succeeded and must stay succeeded.
    - **Best-effort.** A platform with no user-level supervisor at all (a
      Linux without systemd, see :mod:`local_operator.supervisors`) reports
      ``installed=False`` and the session carries on firing its own wakes
      in-process. That is the only platform where a wake can sit unfired, and
      the reason it is reported rather than passed over in silence.
    """
    kind = supervisors.supervisor()
    if kind is None:
        return InstallOutcome(installed=False, reason=UNSUPPORTED_REASON)
    if kind == supervisors.SYSTEMCTL:
        return _ensure_systemd_installed(config_dir)
    if kind == supervisors.SCHTASKS:
        return _ensure_task_installed(config_dir)
    try:
        wanted = render_plist(config_dir)
        path = plist_path()
        path.parent.mkdir(parents=True, exist_ok=True)
        log_path(config_dir).parent.mkdir(parents=True, exist_ok=True)

        # Idempotent by CONTENT, not by existence. A plist from an older
        # release names a different interpreter or config dir, and treating
        # "a file is there" as "installed" would leave that stale unit running
        # forever — the wakes would fire against the wrong store.
        current = None
        if path.exists():
            try:
                current = plistlib.loads(path.read_bytes())
            except Exception:  # noqa: BLE001 — an unreadable plist is a stale one
                current = None
        addressable = _launchd_is_addressable()
        if current == wanted and not addressable:
            # A redirected home: the plist is the only half we own here, and
            # it already matches. Probing launchd would address the REAL
            # domain from a sandbox (see `_launchd_is_addressable`).
            #
            # It must NOT claim "already installed" (round 1, Q1). Nothing
            # supervises this store — the first call in the same store says
            # "plist written; launchd not addressable from here" and
            # `wake status` says "cannot be verified for this store", so
            # answering in the old two-valued vocabulary made three surfaces
            # disagree about one store. `installed=False` for the same reason:
            # this PR's whole thesis is that "installed" stops meaning "a file
            # exists".
            return InstallOutcome(
                installed=False,
                reason="plist already written; launchd not addressable from here",
            )
        if current == wanted and addressable:
            state = supervisor_state(config_dir)
            if state.running:
                return InstallOutcome(installed=True, reason="already installed")
            if state.loaded:
                # THE REPAIR THAT CLOSES THE PERMANENT MISS. The plist is
                # right and launchd knows the label, but the process exited
                # (self-retirement on an empty index, or a crash KeepAlive
                # declined to restart). The old code read this as "already
                # installed" and left the armed wake with nothing to fire it.
                #
                # `kickstart -k` rather than bootout+bootstrap: it is the
                # narrower operation (restart this job) and does not briefly
                # unregister the label. The `-k` is what makes it a repair
                # rather than a no-op if the state read were ever wrong.
                #
                # Safe to run mid-engage, and that is load-bearing: a runtime
                # this supervisor spawned is DETACHED (`start_new_session=True`,
                # `session/runtime/launch.py`), so killing and restarting the
                # supervisor cannot kill a candidate mid-construction. The
                # restarted supervisor re-reads the index and finds the wake
                # still armed — at worst it re-engages and the lease
                # arbitration hands it the existing runtime.
                result = _launchctl("kickstart", "-k", f"{_domain()}/{LABEL}")
                if result.returncode != 0:
                    return InstallOutcome(
                        installed=False,
                        reason=(
                            "supervisor was loaded but not running and could not be "
                            f"restarted: {result.stderr.strip() or result.returncode}"
                        ),
                    )
                logger.info("wake supervisor was loaded but not running; restarted it")
                return InstallOutcome(installed=True, reason="restarted a stopped supervisor")
            # Loaded=False with a matching plist: launchd forgot the label (a
            # reboot with the agent removed, a hand `bootout`). Falls through
            # to the write+bootstrap below, which is the correct repair.

        if addressable and not _config_lives_in_real_home(config_dir):
            # The guard used to cover the launchctl call but not the WRITE,
            # and the write is the half that escapes: with the real HOME and
            # a redirected config dir (a test, a sandbox, an agent's isolated
            # store) `plist_path()` is the REAL `~/Library/LaunchAgents`, so
            # a sandbox run planted a supervised unit in the operator's live
            # launchd domain, pointed at a store that vanishes with the
            # sandbox (round 2). The real domain supervises only setups whose
            # config lives under the real home; anything else gets the same
            # answer a redirected home gets — file half skipped, no address.
            return InstallOutcome(
                installed=False,
                reason=(
                    "config dir is outside the real home; " "not writing into the real LaunchAgents"
                ),
            )

        path.write_bytes(plistlib.dumps(wanted))
        if not addressable:
            # A redirected home (a test, a sandbox): the plist is written and
            # verifiable, but loading it would install a real unit into the
            # developer's own launchd session. See `_launchd_is_addressable`.
            return InstallOutcome(
                installed=False, reason="plist written; launchd not addressable from here"
            )
        # bootout first so a reinstall replaces a loaded stale unit; a missing
        # unit makes this a no-op, which is why `reload_job` ignores that
        # result. The rest of the pair — waiting for launchd to release the
        # label, retrying past the measured teardown race, and checking the job
        # is registered — is the whole reason this is not written inline. See
        # :mod:`local_operator.launchd`.
        reloaded = launchd.reload_job(label=LABEL, path=path, runner=_launchctl)
        if not reloaded.ok:
            return InstallOutcome(
                installed=False,
                reason=f"launchctl could not load the supervisor: {reloaded.detail}",
            )
        return InstallOutcome(installed=True, reason="installed")
    except Exception as exc:  # noqa: BLE001 — NEVER raises: the persist already won
        logger.debug("wake supervisor install failed", exc_info=True)
        return InstallOutcome(installed=False, reason=f"install failed: {exc}")


def _ensure_systemd_installed(config_dir: Path) -> InstallOutcome:
    """The Linux arm of :func:`ensure_supervisor_installed`.

    Same three properties as the launchd arm, in the same order, because they
    are the contract rather than a launchd detail: idempotent by CONTENT (a unit
    from an older build names a different interpreter or store), the file half
    runs under a redirected home while the manager is only addressed when the
    unit is the one the real home owns, and a loaded-but-dead unit is RESTARTED
    rather than counted as installed. The systemd arm adds one thing the plist
    has no analogue for: the ``.timer`` that re-runs the supervisor every
    ``SELF_HEAL_INTERVAL_S``, which is what makes a wake armed on an otherwise
    idle machine fire.
    """
    try:
        wanted = render_systemd(config_dir)
        unit = supervisors.systemd_unit_path(SYSTEMD_UNIT)
        timer = supervisors.systemd_unit_path(SYSTEMD_TIMER)
        unit.parent.mkdir(parents=True, exist_ok=True)
        log_path(config_dir).parent.mkdir(parents=True, exist_ok=True)
        addressable = supervisors.systemd_unit_is_addressable(
            SYSTEMD_UNIT
        ) and _config_lives_in_real_home(config_dir)
        current = None
        if unit.exists():
            try:
                current = unit.read_text(encoding="utf-8")
            except (OSError, UnicodeDecodeError):
                current = None
        if current == wanted and not addressable:
            return InstallOutcome(
                installed=False,
                reason="unit already written; the user manager is not addressable from here",
            )
        if current == wanted and addressable:
            state = _systemd_supervisor_state(config_dir)
            if state.running:
                return InstallOutcome(installed=True, reason="already installed")
            if state.loaded:
                # The same repair the launchd arm makes with `kickstart -k`: the
                # unit is right and systemd knows it, but the process exited
                # (self-retirement on an empty index, or a crash). `start` on a
                # known unit is the narrow operation; the timer would repair this
                # within SELF_HEAL_INTERVAL_S anyway, but the wake is due now.
                started = supervisors.systemctl_user("start", SYSTEMD_UNIT)
                if started.returncode != 0:
                    return InstallOutcome(
                        installed=False,
                        reason=(
                            "supervisor was loaded but not running and could not be "
                            f"restarted: {supervisors.translate_systemctl_error(started.stderr)}"
                        ),
                    )
                logger.info("wake supervisor was loaded but not running; started it")
                return InstallOutcome(installed=True, reason="restarted a stopped supervisor")
        if addressable and not _config_lives_in_real_home(config_dir):
            # The write half is the one that escapes: with the real HOME and a
            # redirected config dir, `systemd_unit_path()` is the REAL
            # `~/.config/systemd/user`, so a sandbox run would plant a unit in
            # the operator's live user manager pointed at a store that vanishes
            # with the sandbox. Same refusal, same reason, as the plist half.
            return InstallOutcome(
                installed=False,
                reason=(
                    "config dir is outside the real home; "
                    "not writing into the real systemd user directory"
                ),
            )
        unit.write_text(wanted, encoding="utf-8")
        timer.write_text(render_systemd_timer(), encoding="utf-8")
        if not addressable:
            return InstallOutcome(
                installed=False,
                reason="unit written; the user manager is not addressable from here",
            )
        # Lingering BEFORE enable --now: without a user manager the enable fails
        # with the bus error, and enabling linger is what spawns one. Best-effort,
        # so a refusal does not fail the install.
        supervisors.enable_linger()
        supervisors.systemctl_user("daemon-reload")
        # THREE calls, each doing one thing, none of them a duplicate of another:
        #
        # 1. enable the service — the `RunAtLoad` half, so it starts at login.
        # 2. enable --now the TIMER — the `StartInterval` half: it re-runs the
        #    service once it is no longer active, which is the self-heal that
        #    needs no live caller. Enabling a timer does NOT enable the unit it
        #    activates, which is why both are enabled.
        # 3. start the service — because `enable --now <timer>` only schedules
        #    the first fire (in SELF_HEAL_INTERVAL_S), and the wake that
        #    triggered this install is due NOW. macOS gets this from
        #    `bootstrap` running the job immediately; without step 3 a wake armed
        #    on a freshly installed Linux would sit for a quarter of an hour.
        supervisors.systemctl_user("enable", SYSTEMD_UNIT)
        enabled = supervisors.systemctl_user("enable", "--now", SYSTEMD_TIMER)
        if enabled.returncode:
            return InstallOutcome(
                installed=False,
                reason=(
                    "systemctl could not load the supervisor: "
                    f"{supervisors.translate_systemctl_error(enabled.stderr)}"
                ),
            )
        started = supervisors.systemctl_user("start", SYSTEMD_UNIT)
        if started.returncode:
            return InstallOutcome(
                installed=False,
                reason=(
                    "the supervisor unit was registered but could not be started: "
                    f"{supervisors.translate_systemctl_error(started.stderr)}"
                ),
            )
        return InstallOutcome(installed=True, reason="installed")
    except Exception as exc:  # noqa: BLE001 — NEVER raises: the persist already won
        logger.debug("wake supervisor systemd install failed", exc_info=True)
        return InstallOutcome(installed=False, reason=f"install failed: {exc}")


def _ensure_task_installed(config_dir: Path) -> InstallOutcome:
    """The Windows arm: register the task, or restart a stopped one.

    Same contract as the other two arms, and one honest difference recorded here
    rather than in a footnote: Task Scheduler publishes no pid, so "restarted a
    stopped supervisor" is decided by the task's own status word plus the
    trigger, not by an observed process.
    """
    try:
        wanted = render_task_xml(config_dir)
        record = task_record_path(config_dir)
        if not supervisors.task_scheduler_is_addressable(config_dir):
            return InstallOutcome(
                installed=False,
                reason=(
                    "config dir is outside the real profile; "
                    "not registering a scheduled task for it"
                ),
            )
        current = None
        if record.exists():
            try:
                current = record.read_text(encoding="utf-8")
            except (OSError, UnicodeDecodeError):
                current = None
        if current == wanted:
            registered, running, detail = supervisors.task_state(TASK_NAME)
            if registered and running:
                return InstallOutcome(installed=True, reason="already installed")
            if registered:
                started = supervisors.schtasks(*supervisors.task_run_args(TASK_NAME))
                if started.returncode != 0:
                    return InstallOutcome(
                        installed=False,
                        reason=(
                            "supervisor task was registered but not running and could not be "
                            f"started: {((started.stderr or started.stdout) or '').strip()[:200]}"
                        ),
                    )
                logger.info("wake supervisor task was registered but not running; started it")
                return InstallOutcome(installed=True, reason="restarted a stopped supervisor")
            logger.debug("wake supervisor task is gone (%s); registering it again", detail)
        record.parent.mkdir(parents=True, exist_ok=True)
        record.write_text(wanted, encoding="utf-8")
        ok, detail = supervisors.create_task(TASK_NAME, wanted)
        if not ok:
            return InstallOutcome(
                installed=False, reason=f"schtasks could not register the supervisor: {detail}"
            )
        started = supervisors.schtasks(*supervisors.task_run_args(TASK_NAME))
        if started.returncode != 0:
            return InstallOutcome(
                installed=False,
                reason=(
                    "the supervisor task was registered but could not be started: "
                    f"{((started.stderr or started.stdout) or '').strip()[:200]}"
                ),
            )
        return InstallOutcome(installed=True, reason="installed")
    except Exception as exc:  # noqa: BLE001 — NEVER raises: the persist already won
        logger.debug("wake supervisor task install failed", exc_info=True)
        return InstallOutcome(installed=False, reason=f"install failed: {exc}")


def refresh_plist_if_stale() -> launchd.PlistRefresh:
    """Rewrite the supervisor's plist when an older build wrote it.

    ``ensure_supervisor_installed`` already repairs by CONTENT, on every wake
    persist, so this is the one daemon whose staleness was never permanent —
    but that repair is only reached by a session that WRITES a wake. A machine
    that stops scheduling keeps running whatever plist was there last, so the
    upgrade path repairs it here too, on the same terms.

    Never raises, and the same two guards in the same order as the installer:
    the plist must be the one the real passwd home produces, and the store it
    records must live under the real home.
    """
    name = "wakes supervisor"
    try:
        kind = supervisors.supervisor()
        if kind != supervisors.LAUNCHCTL:
            # NOT "the unit is re-read on every start": systemd caches a unit's
            # definition until `daemon-reload`, and it is this installer's own
            # `daemon-reload` at install time that refreshes it. What makes the
            # omission harmless here is narrower, and true: the systemd unit and
            # the Windows task both name `procname.supervised_image()` — the shim
            # that resolves the pointer at exec — so there is no stale
            # INTERPRETER PATH for a rewrite to fix. A refresh for the other two
            # platforms needs its own decision (the upgrade path that calls this
            # only walks LaunchAgents), not a plist function stretched to cover
            # them.
            return launchd.PlistRefresh(name=name, kind="unsupported")
        path = plist_path()
        if not _launchd_is_addressable():
            return launchd.PlistRefresh(name=name, kind="not-addressable")
        # The store the plist already names, not this process's ambient one: a
        # repair brings a unit up to date in place instead of migrating it. The
        # ambient dir is only the fallback for a plist that records none.
        store = launchd.config_dir_from_plist(launchd.load(path)) or ambient_config_dir()
        if not _config_lives_in_real_home(store):
            return launchd.PlistRefresh(name=name, kind="not-addressable")
        outcome = launchd.rewrite_if_stale(name=name, path=path, rendered=render_plist(store))
        if outcome.kind != "repaired":
            return outcome
        # bootout + bootstrap through the shared helper, unlike the
        # `kickstart -k` a few lines up in ``ensure_supervisor_installed``: that
        # repair restarts a STOPPED job whose plist is already correct, while
        # this one has just CHANGED the plist, and launchd restarts a kickstarted
        # job from its in-memory definition — measured, it keeps running the old
        # argv. See :mod:`local_operator.launchd`.
        reloaded = launchd.reload_job(label=LABEL, path=path, runner=_launchctl)
        if not reloaded.ok:
            # Names the recovery, because the job is DOWN at this point: see
            # `launchd.reload_failure`.
            return reloaded.as_refresh_failure(name=name, path=path, recovery="lop wake install")
        return outcome
    except Exception as exc:  # noqa: BLE001 — a repair must never fail an upgrade
        return launchd.PlistRefresh(name=name, kind="failed", detail=str(exc))


def _is_loaded(config_dir: Path) -> bool:
    """Whether launchd currently has the unit supervising ``config_dir``.

    **Loaded is not running, and the install path must not confuse the two.**
    This answers only "does launchd know the label", which a job that exited
    0 still satisfies \u2014 the blind spot that let an armed wake sit with no
    live supervisor. :func:`supervisor_state` is what the installer and the
    status surface use; this is kept because it is part of the published
    surface of this module and it remains the correct answer to its own,
    narrower question.
    """
    return supervisor_state(config_dir).loaded


def uninstall() -> InstallOutcome:
    """Remove the supervisor. Used by ``lop wake status --uninstall`` and tests."""
    kind = supervisors.supervisor()
    if kind is None:
        return InstallOutcome(installed=False, reason=UNSUPPORTED_REASON)
    if kind == supervisors.SYSTEMCTL:
        unit = supervisors.systemd_unit_path(SYSTEMD_UNIT)
        timer = supervisors.systemd_unit_path(SYSTEMD_TIMER)
        if supervisors.systemd_unit_is_addressable(SYSTEMD_UNIT):
            # The timer first: disabling the service alone leaves a timer that
            # starts it again, which is the opposite of an uninstall.
            supervisors.systemctl_user("disable", "--now", SYSTEMD_TIMER)
            supervisors.systemctl_user("disable", "--now", SYSTEMD_UNIT)
        try:
            unit.unlink(missing_ok=True)
            timer.unlink(missing_ok=True)
        except OSError as exc:
            return InstallOutcome(installed=False, reason=f"could not remove the unit: {exc}")
        return InstallOutcome(installed=False, reason="uninstalled")
    if kind == supervisors.SCHTASKS:
        ok, detail = supervisors.delete_task(TASK_NAME)
        if not ok:
            return InstallOutcome(installed=False, reason=f"could not remove the task: {detail}")
        task_record_path(ambient_config_dir()).unlink(missing_ok=True)
        return InstallOutcome(installed=False, reason="uninstalled")
    path = plist_path()
    if _launchd_is_addressable():
        _launchctl("bootout", _domain(), str(path))
    try:
        path.unlink(missing_ok=True)
    except OSError as exc:
        return InstallOutcome(installed=False, reason=f"could not remove the plist: {exc}")
    return InstallOutcome(installed=False, reason="uninstalled")
