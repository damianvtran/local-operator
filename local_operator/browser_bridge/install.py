"""Install and supervise the browser bridge on macOS or Linux.

The unit re-enters this interpreter's package rather than pinning a checkout;
updating the installed Local Operator package therefore updates the daemon on
its next restart without rewriting supervisor configuration.
"""

from __future__ import annotations

import hashlib
import json
import os
import plistlib
import re
import shutil
import subprocess
import sys
import time
import urllib.error
import urllib.request
from pathlib import Path
from typing import Any

from local_operator import procname
from local_operator.browser_bridge import state as state_store
from local_operator.browser_bridge.daemon import (
    DEFAULT_PORT,
    pairing_status,
    reset_pairing,
)
from local_operator.paths import config_dir, log_dir

#: The supervisor name for the DEFAULT config root. Every already-installed
#: user's daemon is registered under exactly this label, so it must never
#: change: a rename would orphan those daemons, leaving a running process the
#: CLI can no longer stop, start or uninstall.
LABEL = "com.local-operator.browser"
SYSTEMD_UNIT = "local-operator-browser.service"

#: The config root the default label belongs to. Compared against the resolved
#: root to decide whether this run is the default install or an isolated one.
_DEFAULT_CONFIG_DIRNAME = ".local-operator"


def _passwd_home() -> Path:
    """The uid's home from the passwd database, ignoring ``$HOME``.

    The supervisor namespace is keyed by UID (launchd's ``gui/<uid>``, systemd's
    ``--user`` instance) and does not move when ``$HOME`` is redirected, so
    anything reasoning about "the default install" has to anchor here rather
    than on :meth:`Path.home`, which reads ``$HOME``.
    """
    try:
        import pwd

        return Path(pwd.getpwuid(os.getuid()).pw_dir)
    except (ImportError, KeyError, OSError):  # pragma: no cover - no pwd on win32
        return Path.home()


def _default_config_root() -> Path:
    """The config root that owns the DEFAULT supervisor name.

    Anchored on the uid's passwd home rather than on ``Path.home()``, which
    reads ``$HOME``. That distinction is the entire fix: the launchd domain is
    ``gui/<uid>`` and the systemd ``--user`` instance is likewise per-uid, so
    the namespace a label collides in is keyed by UID and does NOT move when
    ``$HOME`` is redirected. Deriving the "is this the default install?" test
    from ``Path.home()`` would make an isolated ``HOME=/tmp/... lop browser
    install`` compare its own root against its own home, conclude it IS the
    default, and reuse the real daemon's label — which is the collision this
    exists to prevent.
    """
    return _passwd_home() / _DEFAULT_CONFIG_DIRNAME


def _root_suffix() -> str:
    """``""`` for the default config root, else ``.<8-hex-of-root>``.

    Why the supervisor name has to depend on the config root: the label is a
    GLOBAL name in the user's launchd domain (``gui/<uid>``) and in the systemd
    ``--user`` instance. ``plist_path()`` varies with ``$HOME``, but launchd
    resolves a plist to the ``Label`` INSIDE it — so ``bootout`` with a plist
    at a different path evicts the incumbent registered under the same label.
    Verified directly with a throwaway service: bootstrapping from one path and
    booting out via a second path carrying the same ``Label`` removed it.

    The consequence was that an isolated run — the exact isolation AGENTS.md
    tells agents to use, ``HOME=/tmp/... lop browser install`` — evicted the
    operator's live daemon and replaced it with its own. That happened on this
    machine.

    Deriving the suffix from the config root rather than refusing a non-default
    root keeps that isolation working (concurrent isolated daemons no longer
    collide) instead of blocking it. The digest is of the RESOLVED, symlink-free
    path so two spellings of one root produce one label.
    """
    try:
        root = config_dir().expanduser().resolve()
        default_root = _default_config_root().expanduser().resolve()
    except OSError:  # pragma: no cover - resolve() on an unreadable parent
        root = config_dir().expanduser()
        default_root = _default_config_root().expanduser()
    if root == default_root:
        return ""
    digest = hashlib.sha256(str(root).encode("utf-8")).hexdigest()[:8]
    return f".{digest}"


def label() -> str:
    """Launchd label for this config root; byte-identical to :data:`LABEL` by default."""
    return f"{LABEL}{_root_suffix()}"


def systemd_unit() -> str:
    """systemd unit name for this config root; the default root keeps the plain name."""
    suffix = _root_suffix()
    if not suffix:
        return SYSTEMD_UNIT
    return f"local-operator-browser{suffix}.service"


def plist_path() -> Path:
    return Path.home() / "Library" / "LaunchAgents" / f"{label()}.plist"


def systemd_path() -> Path:
    return Path.home() / ".config" / "systemd" / "user" / systemd_unit()


def log_path() -> Path:
    return log_dir() / "browser-bridge.log"


def render_plist(port: int = DEFAULT_PORT) -> dict[str, object]:
    return {
        # Per-config-root label (this PR) over #752's branded interpreter
        # image: the two are orthogonal — one decides WHICH daemon launchd is
        # told about, the other decides what the user sees it called.
        "Label": label(),
        # Branded interpreter image when one can be planted: macOS names this
        # background item by the basename of ProgramArguments[0], so a bare
        # `sys.executable` is what made installing the bridge notify that
        # 'python3 is running in the background'. Falls back to sys.executable.
        "ProgramArguments": procname.launchd_program(
            "local_operator.browser_bridge.daemon",
            "--port",
            str(port),
        ),
        "RunAtLoad": True,
        "KeepAlive": {"SuccessfulExit": False},
        "StandardOutPath": str(log_path()),
        "StandardErrorPath": str(log_path()),
        "ProcessType": "Interactive",
    }


#: ``StandardOutput=append:`` landed in systemd 240 (upstream NEWS; confirmed
#: by the maintainer on systemd-devel). An older systemd does NOT refuse the
#: unit — measured on 255 against a deliberately invalid specifier, it logs
#: "Failed to parse output specifier, ignoring" and starts anyway — so the
#: real cost of emitting it blindly is subtler than a hard failure: the
#: directive is silently dropped, the output goes to the journal, and the
#: product would still be pointing users at a file nothing writes. That is the
#: exact defect being fixed, so the version gate is what keeps the emitted unit
#: and :func:`log_location` telling the same story on every systemd.
MIN_SYSTEMD_APPEND_VERSION = 240


class _Detect:
    """Sentinel type: "detect the version" as distinct from "it is unknown"."""


#: Default for ``render_systemd(version=...)``. See the note at its use site.
_DETECT = _Detect()


def systemd_version() -> int | None:
    """Major version of the running systemd, or ``None`` if it cannot be read.

    ``systemctl --version`` prints e.g. ``systemd 255 (255.4-1ubuntu8.17)``.
    Unknown degrades to "assume old", which is the safe direction: the unit
    stays loadable and the output goes to the journal.
    """
    if not shutil.which("systemctl"):
        return None
    try:
        result = subprocess.run(
            ["systemctl", "--version"], capture_output=True, text=True, timeout=10
        )
    except (OSError, subprocess.SubprocessError):
        return None
    match = re.search(r"systemd\s+(\d+)", result.stdout or "")
    return int(match.group(1)) if match else None


def render_systemd(port: int = DEFAULT_PORT, *, version: int | None | _Detect = _DETECT) -> str:
    """The user unit.

    ``StandardOutput``/``StandardError`` are the reason this takes a version.
    systemd's default sends daemon output to the JOURNAL, while three product
    surfaces — the install failure message, ``lop browser logs`` and the ``log:``
    line in ``lop browser status`` — all pointed at :func:`log_path`, a file
    nothing ever wrote. ``install()`` creates that file's parent directory, so
    a Linux user following those instructions found an empty directory, which
    reads as "the daemon produced no output" rather than "look somewhere else".
    Reproduced on real systemd 255: ``tail`` of that path reported
    ``No such file or directory`` while the output sat in the journal.
    """
    command = f"{sys.executable} -m local_operator.browser_bridge.daemon --port {port}"
    # ``_DETECT`` and not ``None`` as the default: ``None`` is a MEANINGFUL
    # version value here ("systemd is present but its version could not be
    # read"), so overloading it to also mean "caller did not pass one" would
    # make that branch unreachable — and untestable — on any machine where
    # detection happens to succeed. Caught by CI, whose Linux runners detect a
    # modern systemd and so silently took the redirect path.
    resolved = systemd_version() if isinstance(version, _Detect) else version
    redirect = ""
    if resolved is not None and resolved >= MIN_SYSTEMD_APPEND_VERSION:
        redirect = f"StandardOutput=append:{log_path()}\nStandardError=append:{log_path()}\n"
    return f"""[Unit]
Description=Local Operator browser bridge

[Service]
ExecStart={command}
Restart=on-failure
RestartSec=5
{redirect}
[Install]
WantedBy=default.target
"""


def logs_command(lines: int = 100, *, follow: bool = False) -> list[str]:
    """The command that actually shows this platform's daemon output.

    Keeping this in one place is what stops the three surfaces disagreeing:
    on a systemd too old for ``append:`` the log file genuinely does not exist,
    and the honest answer is ``journalctl``, not a ``tail`` that cannot open it.
    """
    if _supervisor() == "systemctl" and not _log_file_is_written():
        command = ["journalctl", "--user", "-u", systemd_unit(), "-n", str(lines)]
        if follow:
            command.append("-f")
        return command
    command = ["tail", "-n", str(lines)]
    if follow:
        command.append("-f")
    command.append(str(log_path()))
    return command


def _log_file_is_written() -> bool:
    """Whether this platform's supervisor redirects the daemon's output to a file."""
    if sys.platform == "darwin":
        return True
    version = systemd_version()
    return version is not None and version >= MIN_SYSTEMD_APPEND_VERSION


def log_location() -> str:
    """Human-readable location of the daemon's output, for status and errors."""
    if _log_file_is_written():
        return str(log_path())
    if _supervisor() == "systemctl":
        return f"journalctl --user -u {systemd_unit()}"
    return str(log_path())


def _domain() -> str:
    return f"gui/{os.getuid()}"


def _launchctl(*args: str) -> subprocess.CompletedProcess[str]:
    return subprocess.run(["launchctl", *args], capture_output=True, text=True, timeout=15)


#: What to tell a user whose platform has no user-level service supervisor.
#: One string shared by install/uninstall/start/stop/restart so the entry
#: points cannot drift into describing the same machine differently.
NO_SUPERVISOR_ERROR = (
    "no supported user service supervisor found (launchctl on macOS, "
    "systemctl --user on Linux); run `lop browser serve` in the foreground"
)


def _supervisor() -> str | None:
    """``"launchctl"``, ``"systemctl"``, or ``None`` when neither is usable.

    Guarding on the BINARY rather than on ``sys.platform`` is the whole point.
    ``subprocess.run(..., check=False)`` suppresses a non-zero exit status but
    NOT ``FileNotFoundError`` when the executable is absent, so every caller
    that skipped this check raised an uncaught traceback out of the CLI on a
    Linux without systemd (Devuan, Alpine, Void, OpenRC, many containers, WSL2
    without systemd) and on win32, which fell into the same branch. Reproduced
    before this fix: ``start``, ``stop``, ``restart`` and ``uninstall`` all
    raised ``FileNotFoundError: 'systemctl'``.
    """
    if sys.platform == "darwin" and shutil.which("launchctl"):
        return "launchctl"
    if sys.platform.startswith("linux") and shutil.which("systemctl"):
        return "systemctl"
    return None


def _systemctl_user(*args: str) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        ["systemctl", "--user", *args], capture_output=True, text=True, timeout=30
    )


#: systemd's message when there is no user D-Bus to talk to — the normal state
#: over plain SSH without lingering.
_NO_BUS_MARKERS = ("Failed to connect to bus", "No medium found", "XDG_RUNTIME_DIR")


def linger_remedy() -> str:
    user = os.environ.get("USER") or "$USER"
    return (
        "systemd has no user session bus for this login (normal over plain "
        "SSH). Enable lingering so the user manager starts at boot and "
        f"survives logout:\n    loginctl enable-linger {user}\n"
        "then re-run `lop browser install`."
    )


def _translate_systemctl_error(stderr: str) -> str:
    """Name the remedy for the one systemctl failure users actually hit.

    The raw stderr was surfaced verbatim — truthful, but it left the user to
    discover ``loginctl enable-linger`` on their own, and nothing in the
    product named it.
    """
    text = (stderr or "").strip()
    if any(marker in text for marker in _NO_BUS_MARKERS):
        return f"{text[:200]}\n\n{linger_remedy()}" if text else linger_remedy()
    return text[:300]


def enable_linger() -> bool:
    """Best-effort ``loginctl enable-linger``; never fatal to an install.

    Per ``loginctl(1)``, without lingering no user manager is spawned at boot
    and it is torn down when the last session ends — so ``WantedBy=default.target``
    plus ``enable --now`` does NOT deliver "survives restarts" the way macOS's
    ``RunAtLoad`` does. Confirmed on systemd 255: with lingering disabled,
    ``systemctl --user`` cannot reach the user manager at all ("User ID is not
    logged in or lingering"). A graphical desktop login is close enough in
    practice; SSH and headless hosts are where the daemon dies and stays dead.

    Non-fatal because it can legitimately be refused (no polkit agent, a locked
    down host), and an install that otherwise succeeded must not be reported as
    failed over it.
    """
    binary = shutil.which("loginctl")
    if not binary:
        return False
    user = os.environ.get("USER")
    try:
        result = subprocess.run(
            [binary, "enable-linger", *([user] if user else [])],
            capture_output=True,
            text=True,
            timeout=15,
        )
    except (OSError, subprocess.SubprocessError):
        return False
    return result.returncode == 0


def health(port: int = DEFAULT_PORT, timeout: float = 3.0) -> dict[str, Any] | None:
    try:
        with urllib.request.urlopen(f"http://127.0.0.1:{port}/health", timeout=timeout) as response:
            value = json.loads(response.read().decode())
            return value if isinstance(value, dict) else None
    except Exception:  # noqa: BLE001 - a probe answers absent rather than raising
        return None


def stale_heartbeat_age(root: Path | None = None) -> float | None:
    """Heartbeat age when the discovery file is stale but the daemon is alive.

    ``None`` whenever there is nothing worth reporting (no daemon, or a fresh
    heartbeat). This is the diagnostic that makes the incident's central
    contradiction visible: ``status`` reads the live socket, every session
    reads this file, and when the file stops being refreshed the two disagree
    silently — status says "connected" while every session falls back to cmux.
    """
    try:
        status_value, current = state_store.liveness(root)
    except Exception:  # noqa: BLE001 - a diagnostic never raises
        return None
    if status_value is not state_store.Liveness.STALE or current is None:
        return None
    return state_store.heartbeat_age(current)


def repair(port: int | None = None, root: Path | None = None) -> dict[str, Any]:
    """Reconcile the daemon's advertised state against reality.

    The user's remedy in the incident was "there isn't one": the bridge
    advertised a tab that no longer existed and a heartbeat that had stopped,
    and the only lever anyone had was killing a healthy daemon. This asks the
    daemon to prune what reality does not back and to republish a fresh
    heartbeat, then reports what it cleaned.

    Safe while sessions are live: the daemon's ``/repair`` closes no tab,
    cancels no in-flight command, and touches no pairing.
    """
    current = state_store.read(root)
    resolved_port = port or (current.port if current else DEFAULT_PORT)
    steps: list[str] = []
    if current is None:
        return {
            "ok": False,
            "steps": steps,
            "error": "no bridge daemon state found; 'lop browser install' starts it.",
        }
    age = state_store.heartbeat_age(current)
    steps.append(f"daemon pid {current.pid} on port {resolved_port}, heartbeat {age:.0f}s old")
    if not state_store.pid_alive(current.pid):
        return {
            "ok": False,
            "steps": steps,
            "error": (
                f"daemon pid {current.pid} is not running; run 'lop browser restart' "
                "to start a fresh one."
            ),
        }
    try:
        request = urllib.request.Request(
            f"http://127.0.0.1:{resolved_port}/repair", data=b"", method="POST"
        )
        with urllib.request.urlopen(request, timeout=5.0) as response:
            payload: Any = json.loads(response.read().decode())
    except urllib.error.HTTPError as error:
        if error.code == 404:
            # The daemon answers, but predates /repair. Restarting is the right
            # advice and the reason matters: this is exactly the long-running
            # daemon whose heartbeat died, so it is the one most likely to be
            # running older code than the CLI asking it to repair itself.
            return {
                "ok": False,
                "steps": steps,
                "error": (
                    f"the daemon on port {resolved_port} is running a build with no "
                    "/repair endpoint (it predates this fix). Run 'lop browser restart' "
                    "to load the current build."
                ),
            }
        return {
            "ok": False,
            "steps": steps,
            "error": f"daemon returned HTTP {error.code} on port {resolved_port}.",
        }
    except Exception as error:  # noqa: BLE001 - report, do not raise, at a CLI edge
        return {
            "ok": False,
            "steps": steps,
            "error": (
                f"daemon is not answering on port {resolved_port} ({error}); "
                "run 'lop browser restart'."
            ),
        }
    cleared = payload.get("cleared_tabs") or []
    if cleared:
        for url in cleared:
            steps.append(f"cleared phantom driven tab: {url}")
    else:
        steps.append("no phantom driven tabs to clear")
    steps.append(f"driven tabs now: {payload.get('driven_tabs', 0)}")
    if payload.get("heartbeat_republished"):
        fresh = state_store.read(root)
        refreshed = state_store.heartbeat_age(fresh) if fresh else None
        steps.append(
            "heartbeat republished"
            + (f" (now {refreshed:.0f}s old)" if refreshed is not None else "")
        )
    else:
        steps.append("heartbeat could NOT be republished — check disk space and permissions")
    return {"ok": True, "steps": steps, "error": ""}


def install(port: int = DEFAULT_PORT, *, dry_run: bool = False) -> dict[str, object]:
    steps: list[str] = []
    log_path().parent.mkdir(parents=True, exist_ok=True)
    # An isolated root that finds an older build's shared-name registration in
    # its own HOME would otherwise leave it running and unaddressable.
    orphan = legacy_registration()
    if orphan is not None:
        steps.append(
            f"note: {orphan} was written by an older build under the shared "
            "supervisor name and is no longer managed by this config root; "
            "remove it by hand if it is still running"
        )
    supervisor = _supervisor()
    if supervisor == "launchctl":
        plist_path().parent.mkdir(parents=True, exist_ok=True)
        if not dry_run:
            plist_path().write_bytes(plistlib.dumps(render_plist(port)))
        steps.append(f"wrote {plist_path()}")
        if not dry_run:
            _launchctl("bootout", _domain(), str(plist_path()))
            loaded = _launchctl("bootstrap", _domain(), str(plist_path()))
            if loaded.returncode:
                return {"ok": False, "steps": steps, "error": loaded.stderr.strip()[:300]}
            steps.append(f"loaded the LaunchAgent ({label()})")
    elif supervisor == "systemctl":
        systemd_path().parent.mkdir(parents=True, exist_ok=True)
        if not dry_run:
            systemd_path().write_text(render_systemd(port), encoding="utf-8")
        steps.append(f"wrote {systemd_path()}")
        if not dry_run:
            # Lingering BEFORE enable --now: without a user manager the enable
            # itself fails with the bus error, and enabling linger is what
            # spawns one. Best-effort, so a refusal does not fail the install.
            if enable_linger():
                steps.append("enabled lingering so the daemon survives logout and reboot")
            else:
                steps.append(
                    "could not enable lingering; the daemon may not survive logout "
                    f"(run: loginctl enable-linger {os.environ.get('USER', '$USER')})"
                )
            _systemctl_user("daemon-reload")
            loaded = _systemctl_user("enable", "--now", systemd_unit())
            if loaded.returncode:
                return {
                    "ok": False,
                    "steps": steps,
                    "error": _translate_systemctl_error(loaded.stderr),
                }
            steps.append(f"enabled the systemd user service ({systemd_unit()})")
    else:
        return {"ok": False, "steps": steps, "error": NO_SUPERVISOR_ERROR}
    if dry_run:
        return {"ok": True, "steps": [*steps, "dry run: skipped load and verification"]}
    deadline = time.time() + 20
    while time.time() < deadline:
        if health(port) is not None:
            return {"ok": True, "steps": [*steps, "health check passed"]}
        time.sleep(0.5)
    return {
        "ok": False,
        "steps": steps,
        # Naming the journal where that is where the output actually went: the
        # old message sent Linux users to a file nothing writes.
        "error": f"daemon did not become healthy; see {log_location()}",
    }


def _own_registration_exists() -> bool:
    """Whether this config root has a supervisor file under its OWN name."""
    if sys.platform == "darwin":
        return plist_path().exists()
    if sys.platform.startswith("linux"):
        return systemd_path().exists()
    return False


def _legacy_paths() -> list[Path]:
    """Every place a pre-per-root build could have written this user's registration.

    BOTH homes, and that is the correction this needed. ``Path.home()`` reads
    ``$HOME``, but the released build ran under whatever ``$HOME`` was at the
    time — normally the passwd home. An agent or a service running with a
    redirected ``$HOME`` therefore looked in a directory the legacy install was
    never written to, found nothing, and reported the orphan as absent. The two
    coincide in the common case, so the list is de-duplicated rather than
    assumed distinct.
    """
    homes: list[Path] = []
    for home in (Path.home(), _passwd_home()):
        if home not in homes:
            homes.append(home)
    if sys.platform == "darwin":
        return [home / "Library" / "LaunchAgents" / f"{LABEL}.plist" for home in homes]
    if sys.platform.startswith("linux"):
        return [home / ".config" / "systemd" / "user" / SYSTEMD_UNIT for home in homes]
    return []


def legacy_registration() -> Path | None:
    """A registration written by a pre-per-root build that this root now orphans.

    Only ever non-``None`` when this root's supervisor name is SUFFIXED: the
    default root's name is unchanged, so an existing install stays exactly
    where it was and is adopted, not orphaned. Two things produce a suffix and
    both are ordinary user situations rather than just test isolation — a
    ``LOCAL_OPERATOR_CONFIG_DIR`` (a documented setting), and a ``$HOME`` that
    differs from the passwd home.

    Such a user upgrading from a released build has a registration under the
    SHARED default name that this build no longer addresses. Left unresolved
    that produced a false success: ``uninstall`` reported "no LaunchAgent was
    installed" and exited 0 while the daemon kept running — the same "claimed
    success it did not achieve" shape :func:`uninstall` was fixed to remove,
    reintroduced on a different path. So every entry point that manages a
    supervisor — install, uninstall, start/stop/restart, status — resolves it
    the same way, on both platforms.
    """
    if not _root_suffix():
        return None
    for legacy in _legacy_paths():
        if legacy.exists():
            return legacy
    return None


def uninstall(*, purge: bool = False, dry_run: bool = False) -> dict[str, object]:
    """Remove this config root's supervisor registration.

    Reports what it actually achieved. It used to append "removed the systemd
    user service" and return ``ok: True`` unconditionally, so the CLI printed
    success and exited 0 even when ``systemctl`` was absent or the disable
    failed — reproduced with a stub ``systemctl`` exiting 1, which still
    yielded ``{'ok': True}``.
    """
    steps: list[str] = []
    ok = True
    supervisor = _supervisor()
    # A registration this root inherits from a pre-per-root build is part of
    # what "uninstall" means to the user: leaving it behind while reporting
    # success is the false-success shape this function exists to avoid. It is
    # removed under the LEGACY name, which is the name it was registered with.
    orphan = legacy_registration()
    if supervisor == "launchctl":
        if not dry_run:
            existed = plist_path().exists()
            _launchctl("bootout", _domain(), str(plist_path()))
            plist_path().unlink(missing_ok=True)
            if orphan is not None:
                _launchctl("bootout", _domain(), str(orphan))
                orphan.unlink(missing_ok=True)
                steps.append(f"removed the LaunchAgent an older build left at {orphan}")
            steps.append(
                "removed the LaunchAgent"
                if existed
                else (
                    "no LaunchAgent was installed under this config root's name"
                    if orphan is not None
                    else "no LaunchAgent was installed"
                )
            )
        else:
            steps.append("removed the LaunchAgent")
    elif supervisor == "systemctl":
        if not dry_run:
            existed = systemd_path().exists()
            disabled = _systemctl_user("disable", "--now", systemd_unit())
            systemd_path().unlink(missing_ok=True)
            if orphan is not None:
                # Disable by UNIT NAME: systemd addresses units by name, and the
                # legacy unit is the one an older build enabled.
                _systemctl_user("disable", "--now", SYSTEMD_UNIT)
                orphan.unlink(missing_ok=True)
                steps.append(f"removed the systemd unit an older build left at {orphan}")
            _systemctl_user("daemon-reload")
            if disabled.returncode and existed:
                ok = False
                steps.append(
                    f"unit file removed, but `systemctl --user disable` failed: "
                    f"{_translate_systemctl_error(disabled.stderr)}"
                )
            else:
                steps.append(
                    "removed the systemd user service"
                    if existed
                    else (
                        "no systemd user service was installed under this config " "root's name"
                        if orphan is not None
                        else "no systemd user service was installed"
                    )
                )
        else:
            steps.append("removed the systemd user service")
    elif not purge:
        # Nothing to unregister and nothing else asked for: say so rather than
        # claiming a removal that never happened.
        return {"ok": False, "steps": steps, "error": NO_SUPERVISOR_ERROR}
    if purge:
        if not dry_run:
            reset_pairing()
            state_store.remove()
        steps.append("deleted pairing and bridge state")
    # `ok` and not a literal: the disable branch above sets it False, and
    # returning True there is exactly the "claimed success it did not achieve"
    # bug this function was fixed for.
    return {"ok": ok, "steps": steps}


def service_action(action: str) -> dict[str, object]:
    supervisor = _supervisor()
    if supervisor is None:
        # Returned, never raised. Without this guard all three of start/stop/
        # restart raised FileNotFoundError out of the CLI on a systemd-less
        # Linux and on win32, which fell into the same branch.
        return {"ok": False, "error": NO_SUPERVISOR_ERROR}
    # A root whose own registration does not exist but which INHERITS one from a
    # pre-per-root build must act on the inherited name, or start/stop/restart
    # address a service that was never registered: launchctl answers "no such
    # service" while the real daemon keeps running. Only when this root has no
    # install of its own — an existing per-root install always wins.
    orphan = legacy_registration()
    adopted = orphan is not None and not _own_registration_exists()
    if supervisor == "launchctl":
        target = f"{_domain()}/{LABEL if adopted else label()}"
        plist = orphan if adopted and orphan is not None else plist_path()
        if action in ("start", "restart") and plist.exists():
            if _launchctl("print", target).returncode:
                loaded = _launchctl("bootstrap", _domain(), str(plist))
                if loaded.returncode:
                    return {"ok": False, "error": loaded.stderr.strip()[:300]}
        args = {
            "start": ("kickstart", target),
            "stop": ("kill", "SIGTERM", target),
            "restart": ("kickstart", "-k", target),
        }[action]
        result = _launchctl(*args)
        return {"ok": result.returncode == 0, "error": result.stderr.strip()[:300]}
    # Linux gains the recovery macOS already had: a unit file on disk that the
    # user manager has not read yet (installed by an older run, or written
    # before the manager started) fails start/restart with "not found" until
    # something reloads it. Reload once, then retry, rather than making the
    # user discover `systemctl --user daemon-reload` themselves.
    unit = SYSTEMD_UNIT if adopted else systemd_unit()
    result = _systemctl_user(action, unit)
    if result.returncode and action in ("start", "restart") and systemd_path().exists():
        _systemctl_user("daemon-reload")
        result = _systemctl_user(action, unit)
    return {"ok": result.returncode == 0, "error": _translate_systemctl_error(result.stderr)}


def status(port: int | None = None) -> dict[str, object]:
    current = state_store.read()
    resolved_port = port or (current.port if current else DEFAULT_PORT)
    probe = health(resolved_port)
    pairing = pairing_status()
    # An inherited registration counts as installed: reporting "installed: no"
    # while a pre-per-root build's daemon is running is the same false answer
    # `uninstall` was fixed for, and it is what sends a user to reinstall on
    # top of a daemon they already have.
    orphan = legacy_registration()
    return {
        "installed": _own_registration_exists() or orphan is not None,
        "healthy": probe is not None,
        "health": probe,
        "port": resolved_port,
        "state": current.model_dump(mode="json", exclude={"session_key"}) if current else None,
        "paired": pairing["paired"],
        "extension_id": pairing["extension_id"],
        "pending_code": pairing["pending_code"],
        "pending_expires_at": pairing["pending_expires_at"],
        # The location the output is ACTUALLY readable from, which on a systemd
        # too old for `append:` is the journal, not a file that never exists.
        "log": log_location(),
        # Which config root's supervisor registration this is reporting on.
        # Without it two isolated daemons produce identical status output and
        # there is no way to tell which instance you are talking to.
        "supervisor": label() if sys.platform == "darwin" else systemd_unit(),
        "config_root": str(config_dir()),
        # Named, not merely folded into `installed`: this is the one thing that
        # explains why a daemon is running under a name the CLI would not
        # otherwise mention, and it tells the user which file to act on.
        "legacy_registration": str(orphan) if orphan is not None else None,
    }
