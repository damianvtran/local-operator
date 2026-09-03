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
        "ProgramArguments": [sys.executable, "-m", "local_operator.wakes.supervisor"],
        "RunAtLoad": True,
        # SELF-RETIREMENT, and the reason this key is not optional: the
        # supervisor exits 0 when the index empties. Keying restarts on
        # unsuccessful exit only is what makes that exit STICK, so a machine
        # with no wakes left runs no supervisor at all. A plain KeepAlive:true
        # would restart it forever against an empty index.
        "KeepAlive": {"SuccessfulExit": False},
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
        if current == wanted and (not addressable or _is_loaded()):
            return InstallOutcome(installed=True, reason="already installed")

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


def _is_loaded() -> bool:
    """Whether launchd currently has the unit.

    Checked alongside the plist's content because the two can disagree: a
    machine that rebooted with the agent removed, or a `bootout` run by hand,
    leaves the file on disk with nothing running behind it.
    """
    return _launchctl("print", f"{_domain()}/{LABEL}").returncode == 0


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
