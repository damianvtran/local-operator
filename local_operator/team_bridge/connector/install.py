"""Service unit for the connector (H4a render, D2a lifecycle, cross-platform).

Mechanics are the wake supervisor's family (`wakes/install.py`), under a NEW
dedicated label so the two daemons never share a plist/unit identity:

* macOS: a LaunchAgent plist with the supervisor's lifecycle contract
  (``RunAtLoad``; the connector is ALWAYS-ON, so ``KeepAlive: true`` — unlike
  the self-retiring wake supervisor, an empty poll window is its normal state
  and not an exit condition).
* Linux: a ``systemd --user`` unit with ``Restart=always``, enabled so the user
  manager starts it at login (and at boot when lingering is on).
* Windows: a Task Scheduler task — logon trigger, restart on failure, the
  current user, least privilege — registered with ``schtasks /Create /XML``.

WHY THIS MODULE INSTALLS (D2a, a reversal of the H4a "render only" contract,
by operator directive). Rendering alone meant nothing in the product ever put
the connector in place: the operator's own LaunchAgent was placed by hand, and
an enrolled machine nobody hand-configured (a teammate's) had NO receiver at
all — every message to it sat at ``feed_seq=0``. The first cut installed on
macOS only and answered ``unsupported`` elsewhere; the receiver is now placed
on all three platforms under ONE contract (operator directive: "team bridge
should be cross-platform and should work on Mac, Windows, Linux"):

* **Event-driven, never polled.** :func:`sync_connector_service` runs at the
  moments the wanted state can change — a local enrolment commit, the one-click
  Turn off, an explicit ``team_bridge.enabled`` write — and once at every
  surface's start (``lop serve`` lifespan, the desktop's app-start status
  boundary, the TUI's startup, the mobile daemon) as the catch-up for machines
  enrolled before this existed. No loop, no per-session poller; a no-op call is
  one file read and one supervisor query.
* **One function, one guard set.** Every surface converges on
  :func:`sync_connector_service`; ``lop team-bridge receiver install|remove``
  is the same call (``remove`` forces the not-wanted answer through the same
  locked body). There is no second installer and no second supervisor layer:
  the platform plumbing is ``launchd`` and ``supervisors``, as for every other
  daemon here.
* **User scope, no admin.** A LaunchAgent in ``~/Library/LaunchAgents``, a unit
  in ``~/.config/systemd/user``, a task registered without ``/RU`` (it belongs
  to the creating user, ``LeastPrivilege``). Never a system daemon, a Windows
  Service or an elevation prompt.
* **Idempotent.** The registration is rewritten only when its CONTENT differs
  from the render, and the supervisor is (re)started only when it does not
  already hold the job. Two connectors are impossible twice over: one name per
  user, and the process's own machine-wide :class:`~.lock.ConnectorLock`.
* **Never for an isolated store.** One name per user means exactly one store
  may own it: the real user's DEFAULT config root
  (``launchd.shared_label_refusal``, the 2026-10-08 incident class — the same
  rule the wake supervisor applies on all three platforms). A test run, a QA
  scratch root or any non-default ``LOCAL_OPERATOR_CONFIG_DIR`` is refused
  before anything is written, and a redirected ``HOME`` never reaches the real
  supervisor (``launchd.is_own_plist``, ``supervisors.systemd_unit_is_addressable``,
  ``supervisors.task_scheduler_is_addressable``). The guards are called through
  their MODULE attributes, so a test that opens one patches the same name
  production reads. On Windows the profile comes from the operating system's
  token-backed folder API, not USERPROFILE; redirected HOME or USERPROFILE
  cannot authorize access to the real task.
* **No user manager is not "unsupported".** A Linux box whose systemd has no
  user instance for this login (a container, WSL without systemd, SSH without
  lingering) answers :data:`CODE_NO_USER_MANAGER` with the remedy in the
  detail, and a Linux/Windows host with no supervisor binary at all answers
  :data:`CODE_NO_SUPERVISOR` naming the foreground fallback
  (``python -m local_operator.team_bridge.connector --config-dir <dir>``).
* **Lingering, decided:** the Linux install ATTEMPTS ``loginctl enable-linger``
  (best effort, exactly as the wake supervisor does), because a receiver that
  stops when the user logs out of the box defeats "respond while nobody is at
  the machine". Whether it took is READ BACK (``supervisors.linger_enabled``)
  and reported, never assumed: polkit commonly allows a user to linger
  themselves, but not on every distro, and a refused linger still leaves a
  receiver that runs while the user is logged in.
* **Survives updates.** The unit names the STABLE interpreter shim
  (``procname.supervised_image``), never a versioned generation, so a flip or a
  prune cannot leave the supervisor pointing at a deleted tree.
"""

from __future__ import annotations

import hashlib
import logging
import os
import plistlib
import subprocess
import sys
import threading
import time
from collections.abc import Iterator
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path

from local_operator import launchd, procname, supervisors

logger = logging.getLogger(__name__)

#: The connector's own identity — a NEW label per the manager decision; the
#: wake supervisor's ``com.local-operator.wakes`` is never reused or altered.
LABEL = "com.local-operator.bridge-connector"
TASK_NAME = "Local Operator bridge connector"
SYSTEMD_UNIT = "local-operator-bridge-connector.service"

#: The module the unit runs; one spelling for the render and the systemd/task argv.
MODULE = "local_operator.team_bridge.connector"


def _argv(config_dir: Path) -> list[str]:
    """What the Linux/Windows renders run: this interpreter, this package, this dir."""

    return [sys.executable, "-m", MODULE, "--config-dir", str(config_dir)]


def log_path(config_dir: Path) -> Path:
    """The connector's own rotating log (written by the process itself).

    Under ``<config_dir>/logs`` like the wake supervisor's log, so ``config_dir``
    alone relocates it for tests and second profiles.
    """

    return config_dir / "logs" / "bridge-connector.log"


def stderr_path(config_dir: Path) -> Path:
    """What launchd points the connector's stdout/stderr at.

    DELIBERATELY NOT :func:`log_path`. The process rotates its log by renaming
    it, and a launchd ``StandardErrorPath`` fd is never reopened, so sharing one
    file would strand launchd's stream in the rotated backup (the failure
    ``local_operator.paths.runtime_log_path`` documents for ``mobile.log``). This
    file only receives what escapes the logging handler: an interpreter crash
    traceback or a stray ``print``, so it stays small without rotation.
    """

    return config_dir / "logs" / "bridge-connector.stderr.log"


def plist_path() -> Path:
    """Where launchd reads the connector's LaunchAgent under THIS process's home.

    ``Path.home()`` (via ``launchd.plist_path_for``), so a redirected home
    writes into its own tree; whether launchd may be ADDRESSED for it is
    ``launchd.is_own_plist``'s separate answer.
    """

    return launchd.plist_path_for(LABEL)


def render_plist(config_dir: Path) -> dict[str, object]:
    """The macOS job plan — pure, so tests assert exactly what launchd reads."""

    return {
        "Label": LABEL,
        # The wake supervisor's shape (``launchd_job``): ``Program`` is the
        # STABLE shim on a generation-layout machine, so an update or a prune
        # never orphans this unit, and ``ProgramArguments[0]`` is the role label
        # macOS shows in Login Items instead of a bare "python3". Where no
        # branded image exists it degrades to ``[sys.executable, ...]``.
        **procname.launchd_job(
            MODULE,
            "--config-dir",
            str(config_dir),
            label=procname.branded_argv0(procname.LABEL_BRIDGE_CONNECTOR),
        ),
        "RunAtLoad": True,
        # ALWAYS-ON: restarted regardless of exit status. The wake
        # supervisor's `SuccessfulExit: False` is its self-retirement
        # contract; the connector has no empty-index exit, so it keeps the
        # plain dictionary form.
        "KeepAlive": True,
        # Belt and braces for whatever still exits (a lock loser, a crash):
        # launchd's default 10 s throttle relaunches an always-on job six
        # times a minute. The connector no longer exits on no-owner or on a
        # refusal (it idles in-process), so this only paces real failures, at
        # the loop's own backoff cap (loop.BACKOFF_MAX_S).
        "ThrottleInterval": 60,
        "EnvironmentVariables": {"LOCAL_OPERATOR_CONNECTOR": "1"},
        # Only what escapes the in-process rotating handler lands here
        # (``__main__._configure_logging`` is the operative diagnostic); launchd
        # creates the file with the umask's mode and never rotates it.
        "StandardOutPath": str(stderr_path(config_dir)),
        "StandardErrorPath": str(stderr_path(config_dir)),
    }


def _image() -> str:
    """The interpreter a Linux/Windows unit names: the STABLE shim, else this one.

    The same answer the macOS plist gets through ``procname.launchd_job`` and the
    wake supervisor's units use: a unit is re-executed after updates and prunes,
    so it must never name a generation's own venv when a stable path exists.
    """

    return str(procname.supervised_image() or Path(sys.executable))


def render_systemd(config_dir: Path) -> str:
    """The Linux user unit — the same always-on contract, in systemd's words.

    ``Restart=always`` (not the shared default ``on-failure``): the receiver is
    always-on, and a clean exit (a lock loser, a stop that did not come from
    systemd) must still come back — launchd's ``KeepAlive: true``. ``RestartSec``
    is the plist's ``ThrottleInterval``. Both values are QUOTED in systemd's own
    grammar (``supervisors.quoted``): an interpreter or store under a path with a
    space otherwise makes the unit unstartable or truncates the store. The output
    lines follow the systemd version (``append:`` needs 240+); the receiver's
    real diagnostic is its own rotating :func:`log_path` either way, so on an
    older systemd escaped output goes to the journal (``journalctl --user -u``).
    """

    return supervisors.render_systemd_unit(
        description=TASK_NAME,
        exec_start=(
            f"{supervisors.quoted(_image())} -m {MODULE} "
            f"--config-dir {supervisors.quoted(str(config_dir))}"
        ),
        restart="always",
        restart_sec=60,
        # Detached session runtimes share the receiver's cgroup. Stopping the
        # receiver must not kill the chief of staff's already-admitted turn.
        kill_mode="process",
        post_lines=[
            f"Environment={supervisors.quoted('LOCAL_OPERATOR_CONNECTOR=1')}",
            *supervisors.output_redirect_lines(stderr_path(config_dir)),
        ],
    )


def render_task_xml(config_dir: Path) -> str:
    """The Windows task: run at logon, restarted by Task Scheduler, hidden.

    The shared renderer (``supervisors.render_task_xml``): a ``LogonTrigger`` for
    the current user, ``LeastPrivilege``, ``InteractiveToken`` (no stored
    password, no elevation), ``RestartOnFailure`` at the schema's maximum count,
    ``IgnoreNew`` so a second instance never starts beside a running one, and no
    execution time limit. With no environment and no log redirect the action is
    the interpreter itself rather than ``cmd.exe``, so no console host is
    involved; the image is ``pythonw.exe`` when one sits beside the interpreter,
    which is what keeps a console window from flashing at every logon — the
    receiver logs to its own rotating :func:`log_path`, so it needs no console.
    """

    return supervisors.render_task_xml(
        description=TASK_NAME,
        image=_windowless(_image()),
        argv=["-m", MODULE, "--config-dir", str(config_dir)],
        user_id=supervisors.current_user_id(),
        hidden=True,
    )


def _windowless(image: str) -> str:
    """``pythonw.exe`` beside ``python.exe`` when it exists, else ``image``.

    ``python.exe`` is a console-subsystem binary, so Task Scheduler gives it a
    console window for an interactive logon task; ``pythonw.exe`` is the same
    interpreter built for the GUI subsystem and opens none. Only swapped when the
    sibling really exists (a venv carries both on Windows; a POSIX host never
    does), so a render on macOS/Linux — tests included — is unchanged.
    """

    path = Path(image)
    if path.name.lower() == "python.exe":
        candidate = path.with_name("pythonw.exe")
        if candidate.exists():
            return str(candidate)
    return image


def task_name() -> str:
    """User-qualified task identity in Task Scheduler's machine-wide namespace.

    No Windows installer shipped before this one, so no legacy task needs a
    global-name migration. Hash the OS profile, not environment USERNAME, so
    two users never replace each other's receiver definition.
    """
    home = supervisors.windows_profile_home() if _is_windows() else Path.home()
    if home is None:
        raise ValueError("the Windows profile cannot be verified")
    suffix = hashlib.sha256(str(home.resolve()).casefold().encode()).hexdigest()[:12]
    return f"{TASK_NAME}-{suffix}"


def task_record_path(config_dir: Path) -> Path:
    """Our own copy of the task definition registered for ``config_dir``.

    Task Scheduler keeps its copy in its own store and ``schtasks /Query`` is the
    authoritative question; this RECORD makes "did this product register the
    task, and with which command?" answerable without a subprocess and gives the
    content comparison something to compare (the wake supervisor's shape).
    """

    return Path(config_dir) / "supervisor" / "bridge-connector-task.xml"


def registration_path(config_dir: Path) -> Path | None:
    """Where THIS platform's registration lives, or ``None`` with no supervisor."""

    kind = supervisors.supervisor()
    if kind == supervisors.LAUNCHCTL:
        return plist_path()
    if kind == supervisors.SYSTEMCTL:
        return supervisors.systemd_unit_path(SYSTEMD_UNIT)
    if kind == supervisors.SCHTASKS:
        return task_record_path(Path(config_dir))
    return None


def write_plist(config_dir: Path, target_dir: Path) -> Path:
    """Render to ``target_dir`` ONLY — the inspection/test helper, never an install."""

    target_dir = Path(target_dir)
    target_dir.mkdir(parents=True, exist_ok=True)
    path = target_dir / f"{LABEL}.plist"
    path.write_bytes(plistlib.dumps(render_plist(config_dir)))
    return path


# -- D2a: the lifecycle (install / start / remove / report) --------------------
#
# Every outcome is a CODE, not prose, because several readers branch on it (the
# commit hook's log line, the app-start catch-up, the CLI, and tests) and none
# of them may parse a sentence. ``detail`` is for the log and the CLI only.

#: Registration written (or kept) and the supervisor now holds the job — this
#: call did work.
CODE_INSTALLED = "installed"
#: Already in place and held by the supervisor: the idempotent no-op every
#: repeat call takes.
CODE_CURRENT = "current"
#: The registration was already correct and the supervisor had forgotten (or
#: stopped) the job, so this call only re-registered/started it. Distinct from
#: ``installed`` because NOTHING was written — the log line must not claim an
#: install that did not happen.
CODE_REGISTERED = "registered"
#: Stopped + unregistered after the wanted state turned off.
CODE_REMOVED = "removed"
#: Not wanted and nothing installed: the cheap answer on a machine never enrolled.
CODE_ABSENT = "absent"
#: RETAINED for readers pinned to the first cut's vocabulary; no path returns it
#: any more. Its two honest successors are :data:`CODE_NO_SUPERVISOR` (no user
#: service supervisor binary at all) and :data:`CODE_NO_USER_MANAGER` (systemd is
#: installed but no user instance is reachable).
CODE_UNSUPPORTED = "unsupported"
#: No user-level service supervisor on this host (no ``launchctl``, no
#: ``systemctl``, no ``schtasks``): nothing is touched, and the detail names the
#: foreground command that runs the same receiver.
CODE_NO_SUPERVISOR = "no_supervisor"
#: Linux with ``systemctl`` but no reachable user manager — a container or WSL
#: without systemd as PID 1, or an SSH login without lingering. NOT
#: "unsupported": the detail says what to do (enable lingering, or run the
#: receiver in the foreground).
CODE_NO_USER_MANAGER = "no_user_manager"
#: The store is not the real user's default one (a test, a QA root, a second
#: ``LOCAL_OPERATOR_CONFIG_DIR``): the per-user unit is never handed to it.
CODE_ISOLATED = "isolated"
#: A source checkout with no stable runtime to name: refuses to point the
#: user's unit at a worktree venv (``update._repair_refusal``'s first question).
CODE_SOURCE_CHECKOUT = "source_checkout"
#: The file half ran under a redirected ``HOME``; the supervisor was not addressed.
CODE_NOT_ADDRESSABLE = "not_addressable"
#: The store or config could not be read, so the wanted state is unknown and
#: nothing was changed — never a removal reasoned from a failed read.
CODE_UNKNOWN = "unknown"
CODE_FAILED = "failed"

#: What ``GET /v1/desktop/team-bridge/status`` reports as ``connector`` (and the
#: CLI's ``status``). ``loaded``: the supervisor holds the job (launchd label
#: registered; systemd unit active; the Windows task running or registered with
#: an unreadable localized status). ``installed``: the registration exists but
#: the supervisor is not running it (or may not be asked). ``missing``: nothing
#: is registered. Off these three the state is one of the CODE_* refusals above
#: (``no_supervisor``, ``no_user_manager``, ``isolated``, ``unknown``).
STATE_LOADED = "loaded"
STATE_INSTALLED = "installed"
STATE_MISSING = "missing"


@dataclass(frozen=True)
class ServiceOutcome:
    """What :func:`sync_connector_service` did, as a code plus a log detail."""

    code: str
    detail: str = ""


def _launchctl(*args: str) -> subprocess.CompletedProcess[str]:
    """THE macOS test seam: every launchd call in this module goes through here.

    The wake supervisor's shape and budget (``wakes/install._launchctl``); the
    shared ``launchd.reload_job``/``bootout_job`` receive it as their runner.
    """

    return subprocess.run(  # noqa: S603 — fixed argv, no shell
        ["launchctl", *args], capture_output=True, text=True, timeout=15
    )


def _systemctl(*args: str) -> subprocess.CompletedProcess[str]:
    """THE Linux test seam: every ``systemctl --user`` call goes through here.

    A module attribute (not a direct ``supervisors.systemctl_user`` call) for the
    reason ``_launchctl`` is one: the unit tests' package-wide guard
    (``tests/unit/conftest.py``) replaces it with a hard failure, so no test can
    reach a developer's real user manager through the connector.
    """

    return supervisors.systemctl_user(*args, timeout=30)


def _schtasks(*args: str) -> subprocess.CompletedProcess[str]:
    """THE Windows test seam: every ``schtasks`` call goes through here.

    Passed as ``runner=`` to ``supervisors.create_task``/``delete_task``/
    ``task_state`` so those shared helpers address Task Scheduler through it too.
    """

    return supervisors.schtasks(*args, timeout=30)


def _enable_linger() -> bool:
    """Best-effort ``loginctl enable-linger`` (seam; see the module docstring)."""

    return supervisors.enable_linger()


def _linger_enabled() -> bool:
    """Whether lingering is on for this user, READ rather than assumed (seam)."""

    return supervisors.linger_enabled()


def foreground_command(config_dir: Path) -> str:
    """The command that runs this receiver without a supervisor.

    The honest fallback on a host with no user service manager: the same module
    the unit would run, in a terminal (or under whatever process manager the
    host does have). Quoted only where a path needs it.
    """

    root = str(config_dir)
    if " " in root:
        root = f'"{root}"'
    return f"python -m {MODULE} --config-dir {root}"


def _source_checkout_refusal() -> str | None:
    """Why this process must not WRITE the user's unit, or ``None``.

    With the stable shim present the render names it, and any install may write
    (the shim runs whatever ``current`` is). Without it the render names THIS
    interpreter, which is only safe from an installed distribution: an editable
    worktree venv serving the default store would otherwise repoint the user's
    connector at a tree that is deleted with the worktree — the incident
    ``update._repair_refusal`` exists for. A non-generation install that IS
    current (a desktop-managed venv) is allowed, and stays correct because the
    app-start catch-up re-renders from the running install on every serve start.
    """

    if procname.supervised_image() is not None:
        return None
    from local_operator import update

    if update.install_kind() in (update.InstallKind.EDITABLE, update.InstallKind.UNKNOWN):
        return "a source checkout with no stable runtime must not install the user's connector"
    return None


def _wanted(config_dir: Path) -> bool | None:
    """Should a connector run for ``config_dir``? ``None`` when it cannot tell.

    Yes exactly when this store holds ONE committed owner and
    ``team_bridge.enabled`` is not explicitly false — the same two facts the
    connector's own fences read, so the unit exists whenever the process would
    have work. An absent ``enabled`` key still wants it: the explicit enable
    commits before the surface writes the key, and a running connector with the
    key unset idles behind its own ``bridge_off`` fence at no cost.

    Read-only on both sides: a machine that never enrolled has no state
    database, and this must not create one at every app start.
    """

    from local_operator.config import read_config_values
    from local_operator.team_bridge.settings import TeamBridgeSettings
    from local_operator.team_bridge.store import Store

    try:
        owner = Store(config_dir).active_owner_readonly()
    except Exception:  # noqa: BLE001 — an unreadable store is "cannot tell"
        logger.debug("bridge connector: owner unreadable", exc_info=True)
        return None
    if owner is None:
        return False
    try:
        values = read_config_values(config_dir)
    except Exception:  # noqa: BLE001
        values = None
    if values is None:
        return None
    local = TeamBridgeSettings.from_values(values)
    return not (local.enabled_is_set and not local.enabled)


def _registered(path: Path) -> bool | None:
    """Does launchd hold the label? ``None`` when it may not be asked.

    REGISTERED, not running, on purpose: with ``KeepAlive: true`` launchd owns
    restarting the process, so the only repair this module makes is
    "bootstrap what launchd does not know". ``None`` under a redirected home:
    ``launchctl`` would answer about the OPERATOR's job (``is_own_plist``).
    """

    if not launchd.is_own_plist(path, LABEL):
        return None
    result = _launchctl("print", f"{launchd.job_domain()}/{LABEL}")
    return result.returncode == 0


#: One sync at a time, per process. Two triggers can reach this at the same
#: moment — the serve lifespan's catch-up and an APP_START status boundary, or
#: a commit hook and a settings edit — and both would otherwise see "no plist"
#: and each run a full bootout+bootstrap. QA measured four concurrent first
#: installs producing 19-34 launchctl calls and FOUR reload cycles, and a
#: bootout that lands after another thread's bootstrap leaves the label
#: UNLOADED while both callers report success: a silent outage, not a wasted
#: round trip. Serialising makes the losers observe the winner's result
#: (``current``) instead of repeating it.
_SYNC_LOCK = threading.Lock()

#: How long a second caller waits for the one in flight before giving up. Sized
#: above the whole launchctl budget (a reload's release + bootstrap deadlines),
#: because waiting is exactly what a caller wants here: the answer it needs is
#: the one being computed.
_SYNC_LOCK_TIMEOUT_S = 30.0


#: How long the CROSS-PROCESS half waits for another process's sync. The same
#: budget as the in-process half, and for the same reason: the answer a waiter
#: needs is the one being computed. Polled at ``_SYNC_FILE_POLL_S`` because the
#: portable primitive is a NON-blocking attempt (``msvcrt.locking``'s blocking
#: mode retries only ten times, one second apart, and then raises).
_SYNC_FILE_LOCK_TIMEOUT_S = _SYNC_LOCK_TIMEOUT_S
_SYNC_FILE_POLL_S = 0.05


@contextmanager
def _sync_exclusive(config_dir: Path) -> Iterator[None]:
    """Serialize surface starts and opt-outs across threads AND processes.

    Both locks must be held. Falling back to unlocked work on contention would
    let a delayed install undo a completed Turn off. Failure is therefore a
    failed sync (the caller reports it), never permission to mutate a unit.
    The portable wake primitive uses flock on POSIX and msvcrt on Windows.
    """
    from local_operator.procstate import O_BINARY
    from local_operator.wakes.lock import _try_lock, _unlock

    if not _SYNC_LOCK.acquire(timeout=_SYNC_LOCK_TIMEOUT_S):
        raise TimeoutError("receiver service sync is busy")
    fd: int | None = None
    locked = False
    try:
        path = Path(config_dir) / "team_bridge" / "connector-service.lock"
        path.parent.mkdir(parents=True, exist_ok=True)
        fd = os.open(path, O_BINARY | os.O_CREAT | os.O_RDWR, 0o600)
        deadline = time.monotonic() + _SYNC_FILE_LOCK_TIMEOUT_S
        while not (locked := _try_lock(fd)):
            if time.monotonic() >= deadline:
                raise TimeoutError("receiver service sync file lock is busy")
            time.sleep(_SYNC_FILE_POLL_S)
        yield
    finally:
        try:
            if fd is not None:
                try:
                    if locked:
                        _unlock(fd)
                finally:
                    os.close(fd)
        finally:
            _SYNC_LOCK.release()


def _install(config_dir: Path) -> ServiceOutcome:
    """Place and start the receiver on THIS platform's supervisor (locked caller)."""

    kind = supervisors.supervisor()
    if kind == supervisors.SYSTEMCTL:
        return _install_systemd(config_dir)
    if kind == supervisors.SCHTASKS:
        return _install_task(config_dir)
    return _install_launchd(config_dir)


def _remove(config_dir: Path) -> ServiceOutcome:
    """Stop and unregister the receiver on THIS platform's supervisor (locked caller)."""

    kind = supervisors.supervisor()
    if kind == supervisors.SYSTEMCTL:
        return _remove_systemd()
    if kind == supervisors.SCHTASKS:
        return _remove_task(config_dir)
    return _remove_launchd()


def _install_launchd(config_dir: Path) -> ServiceOutcome:
    path = plist_path()
    wanted = render_plist(config_dir)
    current = launchd.load(path)
    addressable = launchd.is_own_plist(path, LABEL)
    register_only = False
    if current == wanted:
        if not addressable:
            return ServiceOutcome(CODE_NOT_ADDRESSABLE, "plist current; launchd not addressed")
        if _registered(path):
            return ServiceOutcome(CODE_CURRENT)
        # A current plist launchd has forgotten (a reboot before the agent was
        # ever loaded, a hand bootout): fall through to the bootstrap, which
        # re-registers the SAME bytes — so the answer is ``registered``, not
        # ``installed`` (QA Q-6: the log claimed an install nothing wrote).
        register_only = True
    else:
        refusal = _source_checkout_refusal()
        if refusal is not None:
            return ServiceOutcome(CODE_SOURCE_CHECKOUT, refusal)
        path.parent.mkdir(parents=True, exist_ok=True)
        stderr_path(config_dir).parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(plistlib.dumps(wanted))
        if not addressable:
            return ServiceOutcome(CODE_NOT_ADDRESSABLE, "plist written; launchd not addressed")
    # bootout + bootstrap through the shared helper even for a first install:
    # a rewritten plist must replace the job launchd holds in memory (a
    # kickstart would keep the old argv), and an absent job makes the bootout
    # a no-op. See ``launchd.reload_job`` for the measured teardown race.
    reloaded = launchd.reload_job(label=LABEL, path=path, runner=_launchctl)
    if not reloaded.ok:
        return ServiceOutcome(
            CODE_FAILED, f"launchctl could not load the connector: {reloaded.detail}"
        )
    return ServiceOutcome(CODE_REGISTERED if register_only else CODE_INSTALLED)


def _remove_launchd() -> ServiceOutcome:
    path = plist_path()
    if not path.exists():
        # The cheap answer for every never-enrolled machine at app start: no
        # file, no ``launchctl``. A label loaded from some other path is not
        # this module's to unload.
        return ServiceOutcome(CODE_ABSENT)
    if launchd.is_own_plist(path, LABEL):
        launchd.bootout_job(label=LABEL, path=path, runner=_launchctl)
        if _registered(path):
            return ServiceOutcome(CODE_FAILED, "launchd still holds the receiver; not removed")
    path.unlink(missing_ok=True)
    return ServiceOutcome(CODE_REMOVED)


# -- Linux: systemd --user ----------------------------------------------------


def _systemd_text(result: subprocess.CompletedProcess[str]) -> str:
    return ((result.stderr or "") + (result.stdout or "")).strip()


def _no_user_manager(text: str, config_dir: Path) -> ServiceOutcome:
    """The distinct answer for "systemd is here, its user instance is not".

    The remedy is the one users never find on their own (``loginctl
    enable-linger``) plus the foreground fallback for a host where that cannot
    work at all (a container or WSL whose PID 1 is not systemd).
    """

    return ServiceOutcome(
        CODE_NO_USER_MANAGER,
        (
            f"{supervisors.translate_systemctl_error(text)}\n"
            "If this host does not run systemd as its init (a container, WSL without "
            "systemd), run the receiver in the foreground instead: "
            f"{foreground_command(config_dir)}"
        ).strip(),
    )


def _systemd_active() -> tuple[bool | None, str]:
    """``(active, failure_text)`` from one ``systemctl --user is-active``.

    ``None`` with the failure text when the user manager cannot be reached at
    all — the caller turns that into :data:`CODE_NO_USER_MANAGER` rather than a
    misleading "not running".
    """

    result = _systemctl("is-active", SYSTEMD_UNIT)
    text = _systemd_text(result)
    if result.returncode == 0:
        return True, ""
    if supervisors.is_bus_failure(text):
        return None, text
    return False, text


def _linger_note() -> str:
    """The lingering half of every Linux install outcome, READ, not assumed."""

    if _linger_enabled():
        return "lingering is on: the receiver also runs while you are logged out"
    return (
        "lingering is off: the receiver runs only while you are logged in; to keep it "
        f"running after logout and at boot run: loginctl enable-linger "
        f"{os.environ.get('USER') or '$USER'}"
    )


def _install_systemd(config_dir: Path) -> ServiceOutcome:
    """Reconcile file, manager definition, login enablement and running job.

    The receipt is written only after reload/enable/start all succeed. A unit
    file alone cannot prove systemd loaded those bytes: a failed reload leaves
    an old active process, and a later call must retry instead of declaring it
    current just because disk bytes match.
    """
    unit = supervisors.systemd_unit_path(SYSTEMD_UNIT)
    receipt = Path(config_dir) / "supervisor" / "bridge-connector-systemd.applied"
    wanted = render_systemd(config_dir)
    current = unit.read_text(encoding="utf-8") if unit.exists() else None
    applied = receipt.read_text(encoding="utf-8") if receipt.exists() else None
    addressable = supervisors.systemd_unit_is_addressable(SYSTEMD_UNIT)
    if current == wanted and addressable and applied == wanted:
        active, text = _systemd_active()
        if active is None:
            return _no_user_manager(text, config_dir)
        enabled = _systemctl("is-enabled", SYSTEMD_UNIT)
        if active and enabled.returncode == 0:
            return ServiceOutcome(CODE_CURRENT)
    if current != wanted:
        unit.parent.mkdir(parents=True, exist_ok=True)
        stderr_path(config_dir).parent.mkdir(parents=True, exist_ok=True)
        unit.write_text(wanted, encoding="utf-8")
    if not addressable:
        return ServiceOutcome(CODE_NOT_ADDRESSABLE, "unit written; systemd not addressed")
    _enable_linger()
    reloaded = _systemctl("daemon-reload")
    if reloaded.returncode:
        text = _systemd_text(reloaded)
        if supervisors.is_bus_failure(text):
            return _no_user_manager(text, config_dir)
        return ServiceOutcome(CODE_FAILED, f"systemctl daemon-reload failed: {text}")
    failed = _systemd_enable_and_start()
    if failed is not None:
        if failed.code == CODE_NO_USER_MANAGER:
            return _no_user_manager(failed.detail, config_dir)
        return failed
    receipt.parent.mkdir(parents=True, exist_ok=True)
    if applied != wanted:
        receipt.write_text(wanted, encoding="utf-8")
    return ServiceOutcome(CODE_REGISTERED if current == wanted else CODE_INSTALLED, _linger_note())


def _systemd_enable_and_start() -> ServiceOutcome | None:
    """``enable`` (start at login) then ``restart`` (run THIS render now).

    ``restart`` rather than ``start``: a rewritten unit must replace the process
    systemd is running with the old argv (the plist arm's bootout+bootstrap for
    the same reason); on a stopped unit ``restart`` simply starts it. ``None``
    on success, else the failure as an outcome.
    """

    enabled = _systemctl("enable", SYSTEMD_UNIT)
    if enabled.returncode != 0:
        text = _systemd_text(enabled)
        if supervisors.is_bus_failure(text):
            return ServiceOutcome(CODE_NO_USER_MANAGER, text)
        return ServiceOutcome(
            CODE_FAILED,
            "systemctl could not enable the connector: "
            + supervisors.translate_systemctl_error(text),
        )
    started = _systemctl("restart", SYSTEMD_UNIT)
    if started.returncode != 0:
        text = _systemd_text(started)
        if supervisors.is_bus_failure(text):
            return ServiceOutcome(CODE_NO_USER_MANAGER, text)
        return ServiceOutcome(
            CODE_FAILED,
            "systemctl could not start the connector: "
            + supervisors.translate_systemctl_error(text),
        )
    return None


def _remove_systemd() -> ServiceOutcome:
    unit = supervisors.systemd_unit_path(SYSTEMD_UNIT)
    if not unit.exists():
        # Mirrors the plist arm: no file, no ``systemctl``. A unit enabled from
        # some other path is not this module's to disable.
        return ServiceOutcome(CODE_ABSENT)
    if supervisors.systemd_unit_is_addressable(SYSTEMD_UNIT):
        # Do not erase the only registration record when stopping failed:
        # an unreachable bus is not proof that no receiver is running.
        stopped = _systemctl("disable", "--now", SYSTEMD_UNIT)
        if stopped.returncode != 0:
            return ServiceOutcome(CODE_FAILED, _systemd_text(stopped))
    unit.unlink(missing_ok=True)
    if supervisors.systemd_unit_is_addressable(SYSTEMD_UNIT):
        _systemctl("daemon-reload")
    return ServiceOutcome(CODE_REMOVED)


# -- Windows: Task Scheduler ----------------------------------------------------


def _install_task(config_dir: Path) -> ServiceOutcome:
    """The Windows arm: register the task, or start a registered-but-idle one.

    Same contract as the other two arms. Task Scheduler publishes no pid and
    LOCALIZES its status word, so "running" is believed only on the English
    word ``supervisors.task_state`` positively recognises; an unreadable status
    is treated like a stopped one here (``/Run`` on a running ``IgnoreNew`` task
    is harmless), never as "current".
    """

    if not supervisors.task_scheduler_is_addressable(Path(config_dir)):
        return ServiceOutcome(
            CODE_NOT_ADDRESSABLE,
            "the config dir is outside the real profile; no scheduled task registered",
        )
    if _is_windows() and Path(_windowless(_image())).name.lower() != "pythonw.exe":
        return ServiceOutcome(
            CODE_FAILED,
            "pythonw.exe is required for a receiver with no console window; "
            "install a full Windows Python distribution and reinstall Local Operator",
        )
    wanted = render_task_xml(config_dir)
    record = task_record_path(config_dir)
    current: str | None = None
    if record.exists():
        try:
            current = record.read_text(encoding="utf-8")
        except (OSError, UnicodeDecodeError):
            current = None
    if current == wanted:
        registered, running, _detail = supervisors.task_state(task_name(), runner=_schtasks)
        if registered and running is True:
            return ServiceOutcome(CODE_CURRENT)
        if registered:
            started = _schtasks(*supervisors.task_run_args(task_name()))
            if started.returncode != 0:
                return ServiceOutcome(
                    CODE_FAILED,
                    "the connector task is registered but could not be started: "
                    f"{((started.stderr or started.stdout) or '').strip()[:200]}",
                )
            return ServiceOutcome(CODE_REGISTERED)
        # The record says we registered it and Task Scheduler no longer has it
        # (removed by hand): register the same bytes again below.
    else:
        refusal = _source_checkout_refusal()
        if refusal is not None:
            return ServiceOutcome(CODE_SOURCE_CHECKOUT, refusal)
    if record.exists():
        stopped = _stop_task()
        if stopped is not None:
            return stopped
    log_path(config_dir).parent.mkdir(parents=True, exist_ok=True)
    ok, detail = supervisors.create_task(task_name(), wanted, runner=_schtasks)
    if not ok:
        return ServiceOutcome(CODE_FAILED, f"schtasks could not register the connector: {detail}")
    # The record is written only once Task Scheduler accepted the definition, so
    # a failed /Create is retried by the next sync instead of reading "current".
    record.parent.mkdir(parents=True, exist_ok=True)
    record.write_text(wanted, encoding="utf-8")
    # Existing instances were stopped BEFORE replacing their definition, so a
    # failed stop can never be mistaken for a successful upgrade.
    started = _schtasks(*supervisors.task_run_args(task_name()))
    if started.returncode != 0:
        return ServiceOutcome(
            CODE_FAILED,
            "the connector task was registered but could not be started: "
            f"{((started.stderr or started.stdout) or '').strip()[:200]}",
        )
    return ServiceOutcome(CODE_REGISTERED if current == wanted else CODE_INSTALLED)


def _stop_task() -> ServiceOutcome | None:
    """End the running action before deleting/replacing its registration.

    Task deletion does not terminate the action. Keep the registration record
    on a failed End so the next attempt can still find and stop this receiver.
    A positively idle or missing task needs no termination.
    """
    registered, running, detail = supervisors.task_state(task_name(), runner=_schtasks)
    if registered and running is False:
        return None
    if not registered and "cannot find" in detail.lower():
        return None
    ended = _schtasks(*supervisors.task_end_args(task_name()))
    if ended.returncode:
        return ServiceOutcome(
            CODE_FAILED, (ended.stderr or ended.stdout or "task stop failed").strip()
        )
    return None


def _remove_task(config_dir: Path) -> ServiceOutcome:
    record = task_record_path(config_dir)
    if not record.exists():
        # The record exists exactly when this product registered the task; a
        # task of the same name registered by hand is not this module's.
        return ServiceOutcome(CODE_ABSENT)
    if supervisors.task_scheduler_is_addressable(Path(config_dir)):
        # ``/End`` before ``/Delete``: deleting the definition does not stop the
        # program it started (the wake installer's measured lesson).
        stopped = _stop_task()
        if stopped is not None:
            return stopped
        ok, detail = supervisors.delete_task(task_name(), runner=_schtasks)
        if not ok:
            return ServiceOutcome(CODE_FAILED, f"schtasks could not remove the connector: {detail}")
    record.unlink(missing_ok=True)
    return ServiceOutcome(CODE_REMOVED)


def _store_exists(config_dir: Path) -> bool:
    """Whether this store was ever initialised — never creates it.

    The one fact that decides whether :func:`sync_connector_service` may answer
    ``absent`` without the lock (review R3-1): a store that exists means
    enrolment happened, so an in-flight install may be racing this call.
    """

    from local_operator.team_bridge.store import state_db_path

    return state_db_path(config_dir).exists()


def _platform_refusal(config_dir: Path) -> ServiceOutcome | None:
    """The guards every entry point runs FIRST, in order, or ``None`` to proceed.

    1. a user service supervisor must exist (else :data:`CODE_NO_SUPERVISOR`,
       naming the foreground command);
    2. the per-user name must belong to THIS store — the real user's default
       root, from the real home (else :data:`CODE_ISOLATED`). On every platform,
       because the name is shared on every platform.

    Through the MODULE attributes so a test that opens one guard patches the
    name production reads (diagnosis R3-3).
    """

    if supervisors.supervisor() is None:
        return ServiceOutcome(
            CODE_NO_SUPERVISOR,
            supervisors.no_supervisor_error(foreground_command(Path(config_dir))),
        )
    refusal = _ownership_refusal(Path(config_dir))
    if refusal is not None:
        return ServiceOutcome(CODE_ISOLATED, refusal)
    return None


def _is_windows() -> bool:
    """The platform question, as a seam (``procstate.is_windows``).

    A seam rather than ``os.name`` inline so the Windows identity rule is
    testable on POSIX: patching ``os.name`` itself makes ``pathlib`` build
    ``WindowsPath`` objects, which cannot be instantiated off Windows.
    """

    from local_operator.procstate import is_windows

    return is_windows()


def _ownership_refusal(config_dir: Path) -> str | None:
    """``launchd.shared_label_refusal``, with Windows' home spelled correctly.

    The shared rule (one name per user ⇒ only the default root) is platform-
    neutral, but ``launchd.real_home`` reads the passwd database, which Windows
    does not have — so on Windows the rule would refuse EVERY store, the default
    included, and the receiver could never be placed there. On Windows the real
    home is ``supervisors.windows_profile_home()`` (the Windows profile-folder API); the
    comparison is the same identity test, independent of redirected variables.
    """

    if not _is_windows():
        return launchd.shared_label_refusal(config_dir)
    home = supervisors.windows_profile_home()
    if home is None:
        return (
            "the real user's profile cannot be verified on this host; the shared task is left alone"
        )
    from local_operator.paths import DEFAULT_CONFIG_DIRNAME

    try:
        expected = (home / DEFAULT_CONFIG_DIRNAME).resolve()
        root = Path(config_dir).resolve()
        if Path.home().resolve() != home.resolve():
            return "this process's profile is redirected; the shared task is left alone"
        if os.environ.get("HOME") and Path(os.environ["HOME"]).resolve() != home.resolve():
            return "this process's HOME is redirected; the shared task is left alone"
    except (OSError, ValueError):
        return f"the config dir {config_dir} cannot be resolved; the shared task is left alone"
    if root != expected:
        return (
            f"the config dir {root} is not the default one the shared task serves "
            f"({expected}); not repointing the shared task at it"
        )
    return None


def _reconcile(config_dir: Path, *, force_off: bool) -> ServiceOutcome:
    """THE one locked body behind ``sync_connector_service`` and ``remove_connector_service``."""

    try:
        refused = _platform_refusal(config_dir)
        if refused is not None:
            if refused.code == CODE_ISOLATED:
                logger.info("bridge connector: %s", refused.detail)
            return refused
        # CHEAP PRE-CHECK, deliberately OUTSIDE the lock and side-effect free.
        # WHICH MACHINES MAY TAKE THIS SHORT CUT, and why it is exactly those
        # (review R3-1): the lock is what makes "absent" a CONSISTENT answer, so
        # skipping it is only sound when there is nothing in flight to race — a
        # store that does not exist yet, i.e. a machine that never enrolled.
        # Returning ``absent`` on a merely-not-wanted machine was wrong: a Turn
        # off could answer ``absent`` while a catch-up held the lock mid-install
        # and then wrote the plist, leaving an opted-out machine running a
        # connector. An ENROLLED machine therefore always takes the lock, even
        # when the answer is "remove it"; only a machine with no store at all
        # skips it, which is also what keeps R2-5's property (no
        # ``<config>/team_bridge`` directory and no lock file on a machine that
        # has neither) — the check below creates nothing.
        wanted = False if force_off else _wanted(Path(config_dir))
        if wanted is None:
            return ServiceOutcome(CODE_UNKNOWN, "enrolment or settings unreadable; left as is")
        registration = registration_path(Path(config_dir))
        if (
            not wanted
            and (registration is None or not registration.exists())
            and not _store_exists(Path(config_dir))
        ):
            return ServiceOutcome(CODE_ABSENT)
        source_refusal = _source_checkout_refusal()
        if source_refusal is not None:
            return ServiceOutcome(CODE_SOURCE_CHECKOUT, source_refusal)
        with _sync_exclusive(Path(config_dir)):
            # DECIDE AGAIN, INSIDE THE LOCK (review R2-1). The pre-check above
            # is only a filter: a Turn off that lands while this call waits for
            # the lock must WIN, or the catch-up installs a connector for an
            # opted-out machine and leaves it loaded until the next serve start
            # — the mirror of the outage the lock was added to close. "Losers
            # observe the winner's result" only holds when both callers want
            # the same state, so the state is re-read under the lock.
            wanted = False if force_off else _wanted(Path(config_dir))
            if wanted is None:
                return ServiceOutcome(CODE_UNKNOWN, "enrolment became unreadable; left as is")
            outcome = _install(Path(config_dir)) if wanted else _remove(Path(config_dir))
    except Exception as exc:  # noqa: BLE001 — NEVER raises: the caller already won
        logger.debug("bridge connector service sync failed", exc_info=True)
        return ServiceOutcome(CODE_FAILED, str(exc))
    if outcome.code in (
        CODE_INSTALLED,
        CODE_REGISTERED,
        CODE_REMOVED,
        CODE_FAILED,
        CODE_SOURCE_CHECKOUT,
        CODE_NO_USER_MANAGER,
    ):
        logger.info("bridge connector service: %s %s", outcome.code, outcome.detail)
    return outcome


def sync_connector_service(config_dir: Path) -> ServiceOutcome:
    """Make the connector's service unit match what ``config_dir`` wants.

    THE ONE ENTRY POINT, called at the events that can change the answer (a
    local enrolment commit, the one-click Turn off, an explicit
    ``team_bridge.enabled`` write) and once at every surface's start as the
    catch-up for machines enrolled before this existed: the ``lop serve``
    lifespan (the desktop backend and a backend-only install), the desktop's
    app-start status boundary, the TUI's startup and the mobile daemon. ``lop
    team-bridge receiver install`` is this same call. Idempotent and cheap on
    repeat: a content-equal registration the supervisor holds is one read and
    one supervisor query; a never-enrolled machine is one missing-file check.

    NEVER RAISES: every caller has already done its real work (the commit, the
    opt-out, the setting write, the server boot) and must not fail because a
    service could not be placed. Guards, in order: a supervisor must exist,
    then the per-user name's ownership (an isolated store is refused before any
    read or write), then the wanted state — decided again inside the lock.
    """

    return _reconcile(Path(config_dir), force_off=False)


def remove_connector_service(config_dir: Path) -> ServiceOutcome:
    """Stop and unregister the receiver now, through the SAME guards and lock.

    ``lop team-bridge receiver remove``. It forces the not-wanted answer without
    changing the wanted STATE: enrolment and ``team_bridge.enabled`` are left
    exactly as they are, so the next surface start (or ``install``) would place
    the receiver again. Turning the feature OFF is the durable answer, and the
    CLI says so.
    """

    return _reconcile(Path(config_dir), force_off=True)


def connector_state(config_dir: Path) -> str:
    """``loaded`` / ``installed`` / ``missing`` for the status route, or why not.

    Truthful per platform (see the ``STATE_*`` comment). Off those three it is
    :data:`CODE_NO_SUPERVISOR` (no user service supervisor here),
    :data:`CODE_NO_USER_MANAGER` (Linux, unit written but no user manager to
    ask), :data:`CODE_ISOLATED` for a store the per-user unit never serves (a
    registration at the shared path describes the DEFAULT store's connector,
    not this one's) or :data:`CODE_UNKNOWN`. Never raises, never writes.
    """

    try:
        refused = _platform_refusal(Path(config_dir))
        if refused is not None:
            return refused.code
        kind = supervisors.supervisor()
        if kind == supervisors.SYSTEMCTL:
            unit = supervisors.systemd_unit_path(SYSTEMD_UNIT)
            if not unit.exists():
                return STATE_MISSING
            if not supervisors.systemd_unit_is_addressable(SYSTEMD_UNIT):
                return STATE_INSTALLED
            active, _text = _systemd_active()
            if active is None:
                return CODE_NO_USER_MANAGER
            return STATE_LOADED if active else STATE_INSTALLED
        if kind == supervisors.SCHTASKS:
            if not task_record_path(Path(config_dir)).exists():
                return STATE_MISSING
            if not supervisors.task_scheduler_is_addressable(Path(config_dir)):
                return STATE_INSTALLED
            registered, running, _detail = supervisors.task_state(task_name(), runner=_schtasks)
            if not registered:
                return STATE_MISSING
            # A localized/unknown status is not evidence of a running process.
            if running is None:
                return CODE_UNKNOWN
            return STATE_LOADED if running else STATE_INSTALLED
        path = plist_path()
        if not path.exists():
            return STATE_MISSING
        return STATE_LOADED if _registered(path) else STATE_INSTALLED
    except Exception:  # noqa: BLE001 — a status field never fails the status read
        logger.debug("bridge connector state unreadable", exc_info=True)
        return CODE_UNKNOWN


def supervisor_name() -> str:
    """The supervisor and the name this receiver is registered under, for status."""

    kind = supervisors.supervisor()
    if kind == supervisors.LAUNCHCTL:
        return f"launchd (LaunchAgent {LABEL})"
    if kind == supervisors.SYSTEMCTL:
        return f"systemd --user (unit {SYSTEMD_UNIT})"
    if kind == supervisors.SCHTASKS:
        return f"Task Scheduler (task {task_name()!r})"
    return "none"


def journal_hint() -> str | None:
    """Where a Linux unit's escaped output went when systemd cannot append to a file."""

    if (
        supervisors.supervisor() == supervisors.SYSTEMCTL
        and not supervisors.systemd_writes_log_file()
    ):
        return f"journalctl --user -u {SYSTEMD_UNIT}"
    return None


# -- the one catch-up every surface runs at start ------------------------------


async def catch_up_at_surface_start(config_dir: Path, *, surface: str) -> ServiceOutcome:
    """THE surface-start catch-up: :func:`sync_connector_service` off the loop.

    WHY ONE HELPER. The operator's directive is that the receiver works through
    EVERY surface — the desktop app, the TUI, a backend-only ``lop serve``, the
    mobile daemon — with one function and one guard set. Each of those surfaces
    calls exactly this at its start (``server/app.py`` lifespan, ``tui/app.py``
    ``on_mount``, ``mobile/service.py`` ``amain``) and the desktop's app-start
    status boundary calls the same :func:`sync_connector_service`
    (``client.StatusDiscipline.observe_boundary``), so there is no surface whose
    start can place, skip or remove the receiver by a rule of its own.

    ``asyncio.to_thread`` because the supervisor calls are synchronous
    subprocesses with multi-second budgets: inline they would stall the
    surface's event loop (QA measured 0.97 s on one launchctl disable). NEVER
    RAISES — a surface boots whether or not the receiver could be placed; the
    outcome is logged at DEBUG with the surface name so a reader can tell which
    start placed it (``install`` / ``registered`` / ``removed`` already log at
    INFO inside the sync).
    """

    import asyncio

    try:
        outcome = await asyncio.to_thread(sync_connector_service, Path(config_dir))
    except Exception as exc:  # noqa: BLE001 — a surface start never fails for this
        logger.debug("bridge connector catch-up at %s start failed", surface, exc_info=True)
        return ServiceOutcome(CODE_FAILED, str(exc))
    logger.debug("bridge connector catch-up at %s start: %s", surface, outcome.code)
    return outcome
