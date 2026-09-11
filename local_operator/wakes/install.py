"""Install-on-demand for the wake supervisor — the chokepoint, not yet the
installer.

There is exactly one writer of schedule state, ``Session._persist_wake_schedules``,
so there is exactly one place to ask "does something outside this process
now need to exist to fire these?" That question is this function. It is
called after every non-empty persist, idempotently and best-effort: the
persist has already succeeded by the time it runs, and nothing it does (or
fails to do) may change that.

The supervisor it installs is :mod:`local_operator.wakes.supervisor`, run as
a LaunchAgent on macOS and shaped after ``local_operator.mobile.install``.
``KeepAlive: {SuccessfulExit: False}`` is the load-bearing key: the
supervisor exits 0 when the wake index empties, and that setting is what lets
a FINISHED supervisor stay down while a CRASHED one restarts. The next
persist calls this hook again and brings it back.

Install-on-demand rather than install-at-setup, because the cost only makes
sense once there is something to supervise: a user who never schedules a wake
never gets the process. Linux has no installer here yet and reports
``installed=False``; a session there keeps firing its own wakes in-process,
which is what happened everywhere before this existed.

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
import shutil
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path

from local_operator import procname

logger = logging.getLogger(__name__)

#: LaunchAgent label, matching ``mobile.install``'s spelling so the two
#: supervised units read as siblings in ``launchctl list``.
LABEL = "com.local-operator.wakes"

#: Reported when the platform has no installer this hook knows.
UNSUPPORTED_REASON = "no supervisor installer for this platform"


def plist_path() -> Path:
    return Path.home() / "Library" / "LaunchAgents" / f"{LABEL}.plist"


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
    return sys.platform == "darwin" and shutil.which("launchctl") is not None


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
    import pwd

    try:
        real_home = Path(pwd.getpwuid(os.getuid()).pw_dir).resolve()
    except (KeyError, OSError):
        return False
    try:
        expected = (real_home / "Library" / "LaunchAgents" / f"{LABEL}.plist").resolve()
        return plist_path().resolve() == expected
    except (OSError, ValueError):
        return False


def _config_lives_in_real_home(config_dir: Path) -> bool:
    """Whether the supervised unit would point at a store that outlives us.

    A unit supervising a config dir under ``/tmp`` or a sandbox home watches
    a store that is deleted when the sandbox ends — a live launchd unit with
    a corpse for a config. Containment under the passwd home is the right
    test HERE (not the identity test ``_launchd_is_addressable`` uses) because
    the config dir is an ordinary path the user may legitimately place
    anywhere under their home; only dirs OUTSIDE it are the sandbox shape.
    """
    import pwd

    try:
        real_home = Path(pwd.getpwuid(os.getuid()).pw_dir).resolve()
    except (KeyError, OSError):
        return False
    try:
        return config_dir.resolve().is_relative_to(real_home)
    except (OSError, ValueError):
        return False


def render_plist(config_dir: Path) -> dict[str, object]:
    """The whole supervised-unit plan in one pure function.

    Every consumer (install, tests) reads this same rendering, so what the
    tests assert is what launchd is handed.
    """
    return {
        "Label": LABEL,
        # Branded interpreter image when one can be planted: macOS names this
        # background item by the basename of ProgramArguments[0], so a bare
        # `sys.executable` is what makes installing a supervised unit notify
        # 'python3 is running in the background'. Falls back to sys.executable.
        "ProgramArguments": procname.launchd_program("local_operator.wakes.supervisor"),
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
    if not is_supported():
        return SupervisorState(
            loaded=False, running=False, verifiable=False, detail=UNSUPPORTED_REASON
        )
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
    - **Best-effort.** A platform with no installer (Linux without a service
      manager the hook knows) reports ``installed=False`` and the session
      carries on firing its own wakes in-process.
    """
    if not is_supported():
        return InstallOutcome(installed=False, reason=UNSUPPORTED_REASON)
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
            return InstallOutcome(installed=True, reason="already installed")
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
        # unit makes this a no-op, which is why its result is ignored.
        _launchctl("bootout", _domain(), str(path))
        result = _launchctl("bootstrap", _domain(), str(path))
        if result.returncode != 0:
            return InstallOutcome(
                installed=False,
                reason=f"launchctl bootstrap failed: {result.stderr.strip() or result.returncode}",
            )
        return InstallOutcome(installed=True, reason="installed")
    except Exception as exc:  # noqa: BLE001 — NEVER raises: the persist already won
        logger.debug("wake supervisor install failed", exc_info=True)
        return InstallOutcome(installed=False, reason=f"install failed: {exc}")


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
    if not is_supported():
        return InstallOutcome(installed=False, reason=UNSUPPORTED_REASON)
    path = plist_path()
    if _launchd_is_addressable():
        _launchctl("bootout", _domain(), str(path))
    try:
        path.unlink(missing_ok=True)
    except OSError as exc:
        return InstallOutcome(installed=False, reason=f"could not remove the plist: {exc}")
    return InstallOutcome(installed=False, reason="uninstalled")
