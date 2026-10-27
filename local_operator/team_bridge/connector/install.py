"""Service unit for the connector (H4a render, D2a lifecycle).

Mechanics are the wake supervisor's family (`wakes/install.py`), under a NEW
dedicated label so the two daemons never share a plist/unit identity:

* macOS: a LaunchAgent plist with the supervisor's lifecycle contract
  (``RunAtLoad``; the connector is ALWAYS-ON, so ``KeepAlive: true`` — unlike
  the self-retiring wake supervisor, an empty poll window is its normal state
  and not an exit condition).
* Linux: a ``systemd --user`` unit with ``Restart=always``.
* Windows: a Task Scheduler XML whose trigger runs at logon and repeats.

WHY THIS MODULE NOW INSTALLS (D2a, a reversal of the H4a "render only"
contract, by operator directive). Rendering alone meant nothing in the product
ever put the connector in place: the operator's own LaunchAgent was placed by
hand, and an enrolled machine nobody hand-configured (a teammate's) had NO
receiver at all — every message to it sat at ``feed_seq=0``. So the product
now installs and starts it, under these constraints:

* **Event-driven, never polled.** :func:`sync_connector_service` runs at the
  moments the wanted state can change — a local enrolment commit, the one-click
  Turn off, an explicit ``team_bridge.enabled`` write — and once at app/serve
  start as the catch-up for machines enrolled before this existed. No loop, no
  per-session poller; a no-op call is one file read and one ``launchctl print``.
* **macOS only, user scope.** A login-only LaunchAgent in
  ``~/Library/LaunchAgents``: no system daemon, no admin rights. Linux and
  Windows answer :data:`CODE_UNSUPPORTED` and touch nothing (their renders stay
  render-only until each gets its own identity-guarded installer).
* **Idempotent.** The plist is rewritten only when its CONTENT differs from the
  render, and launchd is bootstrapped only when it does not already hold the
  label. Two connectors are impossible twice over: one label per user, and the
  process's own machine-wide :class:`~.lock.ConnectorLock`.
* **Never for an isolated store.** One label per user means exactly one store
  may own it: the real user's DEFAULT config root
  (``launchd.shared_label_refusal``, the 2026-10-08 incident class). A test
  run, a QA scratch root or any non-default ``LOCAL_OPERATOR_CONFIG_DIR`` is
  refused before anything is written, and a redirected ``HOME`` never reaches
  ``launchctl`` (``launchd.is_own_plist``). Both are called through the
  ``launchd`` MODULE attribute, so a test that opens one guard patches the
  same name production reads.
* **Survives updates.** The plist names the STABLE interpreter shim
  (``procname.launchd_job`` → ``procname.supervised_image``), never a versioned
  generation, so a flip or a prune cannot leave launchd pointing at a deleted
  tree.
"""

from __future__ import annotations

import logging
import plistlib
import subprocess
import sys
import threading
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


def render_systemd(config_dir: Path) -> str:
    """The Linux user unit — the same always-on contract, in systemd's words."""

    argv = " ".join(_argv(config_dir))
    return (
        "[Unit]\n"
        f"Description={TASK_NAME}\n"
        "\n"
        "[Service]\n"
        f"ExecStart={argv}\n"
        "Restart=always\n"
        "RestartSec=5\n"
        "\n"
        "[Install]\n"
        "WantedBy=default.target\n"
    )


def render_task_xml(config_dir: Path) -> str:
    """The Windows task: run at logon, restarted by Task Scheduler."""

    argv = " ".join(_argv(config_dir))
    return (
        '<?xml version="1.0" encoding="UTF-16"?>\n'
        '<Task version="1.2" xmlns="http://schemas.microsoft.com/windows/2004/02/mit/task">\n'
        f"  <Description>{TASK_NAME}</Description>\n"
        "  <Triggers><LogonTrigger><Enabled>true</Enabled></LogonTrigger></Triggers>\n"
        "  <Settings><RestartOnFailure><Interval>PT1M</Interval><Count>999</Count>"
        "</RestartOnFailure></Settings>\n"
        "  <Actions><Exec><Command>{}</Command></Exec></Actions>\n"
        "</Task>\n"
    ).format(argv)


def write_plist(config_dir: Path, target_dir: Path) -> Path:
    """Render to ``target_dir`` ONLY — the inspection/test helper, never an install."""

    target_dir = Path(target_dir)
    target_dir.mkdir(parents=True, exist_ok=True)
    path = target_dir / f"{LABEL}.plist"
    path.write_bytes(plistlib.dumps(render_plist(config_dir)))
    return path


# -- D2a: the lifecycle (install / start / remove / report) --------------------
#
# Every outcome is a CODE, not prose, because three readers branch on it (the
# commit hook's log line, the app-start catch-up, and tests) and none of them
# may parse a sentence. ``detail`` is for the log only.

#: Plist written (or kept) and launchd now holds the job — this call did work.
CODE_INSTALLED = "installed"
#: Already in place and loaded: the idempotent no-op every repeat call takes.
CODE_CURRENT = "current"
#: The plist was already correct and launchd had forgotten the label, so this
#: call only re-registered it. Distinct from ``installed`` because NOTHING was
#: written — the log line must not claim an install that did not happen.
CODE_REGISTERED = "registered"
#: Bootout + unlink after the wanted state turned off.
CODE_REMOVED = "removed"
#: Not wanted and nothing installed: the cheap answer on a machine never enrolled.
CODE_ABSENT = "absent"
#: Linux, Windows, or a macOS without ``launchctl``: nothing is touched.
CODE_UNSUPPORTED = "unsupported"
#: The store is not the real user's default one (a test, a QA root, a second
#: ``LOCAL_OPERATOR_CONFIG_DIR``): the per-user label is never handed to it.
CODE_ISOLATED = "isolated"
#: A source checkout with no stable runtime to name: refuses to point the
#: user's unit at a worktree venv (``update._repair_refusal``'s first question).
CODE_SOURCE_CHECKOUT = "source_checkout"
#: The plist half ran under a redirected ``HOME``; launchd was not addressed.
CODE_NOT_ADDRESSABLE = "not_addressable"
#: The store or config could not be read, so the wanted state is unknown and
#: nothing was changed — never a removal reasoned from a failed read.
CODE_UNKNOWN = "unknown"
CODE_FAILED = "failed"

#: What ``GET /v1/desktop/team-bridge/status`` reports as ``connector``.
STATE_LOADED = "loaded"
STATE_INSTALLED = "installed"
STATE_MISSING = "missing"


@dataclass(frozen=True)
class ServiceOutcome:
    """What :func:`sync_connector_service` did, as a code plus a log detail."""

    code: str
    detail: str = ""


def _launchctl(*args: str) -> subprocess.CompletedProcess[str]:
    """THE test seam: every launchd call in this module goes through here.

    The wake supervisor's shape and budget (``wakes/install._launchctl``); the
    shared ``launchd.reload_job``/``bootout_job`` receive it as their runner.
    """

    return subprocess.run(  # noqa: S603 — fixed argv, no shell
        ["launchctl", *args], capture_output=True, text=True, timeout=15
    )


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


@contextmanager
def _sync_exclusive(config_dir: Path) -> Iterator[None]:
    """Hold the process lock AND, where the platform has it, a file lock.

    TWO LOCKS, because the process lock cannot cover the server + a TUI session
    syncing their own app starts at the same moment. The file lock is a
    best-effort flock beside the store (``<config>/team_bridge``): POSIX only,
    and ``fcntl`` is imported here rather than at module scope so importing this
    module stays portable. A failure to take it is never fatal — losing the
    cross-process half degrades to the pre-existing in-process guarantee.
    """

    acquired = _SYNC_LOCK.acquire(timeout=_SYNC_LOCK_TIMEOUT_S)
    handle = None
    try:
        if not acquired:
            logger.debug("bridge connector: another sync holds the lock; proceeding serially")
        try:
            import fcntl

            lock_path = Path(config_dir) / "team_bridge" / "connector-service.lock"
            lock_path.parent.mkdir(parents=True, exist_ok=True)
            handle = open(lock_path, "a+")  # noqa: SIM115 — closed in the finally
            fcntl.flock(handle.fileno(), fcntl.LOCK_EX)
        except Exception:  # noqa: BLE001 — the file half is best-effort
            logger.debug("bridge connector: no file lock available", exc_info=True)
            if handle is not None:
                handle.close()
                handle = None
        yield
    finally:
        if handle is not None:
            try:
                import fcntl

                fcntl.flock(handle.fileno(), fcntl.LOCK_UN)
            except Exception:  # noqa: BLE001
                pass
            handle.close()
        if acquired:
            _SYNC_LOCK.release()


def _install(config_dir: Path) -> ServiceOutcome:
    path = plist_path()
    wanted = render_plist(config_dir)
    current = launchd.load(path)
    addressable = launchd.is_own_plist(path, LABEL)
    _register_only = False
    if current == wanted:
        if not addressable:
            return ServiceOutcome(CODE_NOT_ADDRESSABLE, "plist current; launchd not addressed")
        if _registered(path):
            return ServiceOutcome(CODE_CURRENT)
        # A current plist launchd has forgotten (a reboot before the agent was
        # ever loaded, a hand bootout): fall through to the bootstrap, which
        # re-registers the SAME bytes — so the answer is ``registered``, not
        # ``installed`` (QA Q-6: the log claimed an install nothing wrote).
        _register_only = True
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
    return ServiceOutcome(CODE_REGISTERED if _register_only else CODE_INSTALLED)


def _remove() -> ServiceOutcome:
    path = plist_path()
    if not path.exists():
        # The cheap answer for every never-enrolled machine at app start: no
        # file, no ``launchctl``. A label loaded from some other path is not
        # this module's to unload.
        return ServiceOutcome(CODE_ABSENT)
    if launchd.is_own_plist(path, LABEL):
        launchd.bootout_job(label=LABEL, path=path, runner=_launchctl)
    path.unlink(missing_ok=True)
    return ServiceOutcome(CODE_REMOVED)


def sync_connector_service(config_dir: Path) -> ServiceOutcome:
    """Make the connector's LaunchAgent match what ``config_dir`` wants.

    THE ONE ENTRY POINT, called at the events that can change the answer (a
    local enrolment commit, the one-click Turn off, an explicit
    ``team_bridge.enabled`` write) and once at app/serve start as the catch-up
    for machines enrolled before this existed. Idempotent and cheap on repeat:
    a content-equal plist that launchd holds is one read and one
    ``launchctl print``; a never-enrolled machine is one missing-file check.

    NEVER RAISES: every caller has already done its real work (the commit, the
    opt-out, the setting write, the server boot) and must not fail because a
    service could not be placed. Guards, in order: platform, then the
    per-user label's ownership (an isolated store is refused before any read
    or write), then the wanted state.
    """

    try:
        if supervisors.supervisor() != supervisors.LAUNCHCTL:
            return ServiceOutcome(CODE_UNSUPPORTED, "no launchd on this platform")
        # Through the MODULE attribute so a test that opens this guard patches
        # the name production reads (diagnosis R3-3).
        refusal = launchd.shared_label_refusal(Path(config_dir))
        if refusal is not None:
            logger.info("bridge connector: %s", refusal)
            return ServiceOutcome(CODE_ISOLATED, refusal)
        wanted = _wanted(Path(config_dir))
        if wanted is None:
            return ServiceOutcome(CODE_UNKNOWN, "enrolment or settings unreadable; left as is")
        with _sync_exclusive(Path(config_dir)):
            outcome = _install(Path(config_dir)) if wanted else _remove()
    except Exception as exc:  # noqa: BLE001 — NEVER raises: the caller already won
        logger.debug("bridge connector service sync failed", exc_info=True)
        return ServiceOutcome(CODE_FAILED, str(exc))
    if outcome.code in (
        CODE_INSTALLED,
        CODE_REGISTERED,
        CODE_REMOVED,
        CODE_FAILED,
        CODE_SOURCE_CHECKOUT,
    ):
        logger.info("bridge connector service: %s %s", outcome.code, outcome.detail)
    return outcome


def connector_state(config_dir: Path) -> str:
    """``loaded`` / ``installed`` / ``missing`` for the status route, or why not.

    ``installed`` is "the plist is there but launchd does not (or may not be
    asked whether it) holds it". :data:`CODE_UNSUPPORTED` off macOS and
    :data:`CODE_ISOLATED` for a store the per-user unit never serves, because a
    plist at the shared path describes the DEFAULT store's connector, not this
    one's. Never raises.
    """

    try:
        if supervisors.supervisor() != supervisors.LAUNCHCTL:
            return CODE_UNSUPPORTED
        if launchd.shared_label_refusal(Path(config_dir)) is not None:
            return CODE_ISOLATED
        path = plist_path()
        if not path.exists():
            return STATE_MISSING
        return STATE_LOADED if _registered(path) else STATE_INSTALLED
    except Exception:  # noqa: BLE001 — a status field never fails the status read
        logger.debug("bridge connector state unreadable", exc_info=True)
        return CODE_UNKNOWN
