"""Install and supervise the browser bridge on macOS, Linux or Windows.

The unit re-enters this interpreter's package rather than pinning a checkout;
updating the installed Local Operator package therefore updates the daemon on
its next restart without rewriting supervisor configuration.

Three supervisors, one per platform — a LaunchAgent plist on macOS, a
``systemd --user`` unit on Linux, a Task Scheduler task on Windows — discovered
and rendered by :mod:`local_operator.supervisors`. This module used to be the
ONLY daemon whose supervisor knowledge was platform-correct, which is why the
shared half moved there rather than being copied three more times.
"""

from __future__ import annotations

import hashlib
import json
import os
import plistlib
import re
import subprocess
import sys
import time
import urllib.error
import urllib.request
from contextlib import suppress
from pathlib import Path
from typing import Any

from local_operator import launchd, procname, procstate, supervisors
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

#: Whether this process is on Windows, read ONCE as a module constant.
#:
#: A named constant rather than an inline ``os.name`` read, for the reason
#: every other platform branch in this tree gives (``secrets.keys``,
#: ``group_reaper``, ``procstate``): a test can only flip an inline read by
#: patching ``os.name`` PROCESS-WIDE, and ``pathlib`` picks its flavour from
#: ``os.name`` at call time — so the next ``Path(...)`` anywhere in the test
#: process becomes a ``WindowsPath`` and refuses to exist on the host running
#: the test. Patching this name flips the branch and nothing else.
_IS_WINDOWS = os.name == "nt"

#: The host that can run the Windows log command. ``powershell.exe`` ships with
#: every supported Windows; ``pwsh`` (PowerShell 7) is an optional install, so
#: naming it would reintroduce the very defect the Windows ``logs_command``
#: branch fixes — a command that is not on the machine.
_WINDOWS_POWERSHELL = "powershell"


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
    """The LaunchAgent for this config root and port.

    ``launchd_job`` rather than ``launchd_program``: ``Program`` carries the
    branded image (what Activity Monitor reads) and ``ProgramArguments[0]``
    carries this daemon's role label (what ``ps`` reads), so the bridge stops
    reading as the same bare row as the mobile daemon and the tunnel. macOS
    names a background item by the basename of ``ProgramArguments[0]``, so the
    old bare ``sys.executable`` is what made installing the bridge notify that
    'python3 is running in the background'. The trade-off that shape accepts is
    recorded in ``procname.launchd_job``; with no link to plant, this is
    byte-for-byte the plist this function wrote before. On a machine with the
    generation layout ``Program`` is the stable shim instead (see
    ``procname.supervised_image``) and the role label is unchanged.
    """
    return {
        # Per-config-root label (this PR) over #752's branded interpreter
        # image: the two are orthogonal — one decides WHICH daemon launchd is
        # told about, the other decides what the user sees it called.
        "Label": label(),
        **procname.launchd_job(
            "local_operator.browser_bridge.daemon",
            "--port",
            str(port),
            label=procname.branded_argv0(procname.LABEL_BROWSER, port=port),
        ),
        "RunAtLoad": True,
        "KeepAlive": {"SuccessfulExit": False},
        "StandardOutPath": str(log_path()),
        "StandardErrorPath": str(log_path()),
        "ProcessType": "Interactive",
    }


def refresh_plist_if_stale() -> launchd.PlistRefresh:
    """Rewrite this daemon's LaunchAgent when an older build wrote it.

    Same gap as the mobile and tunnel daemons, and nothing repaired it either:
    ``lop-update`` never touched the bridge plist, so a daemon installed before
    branding keeps its bare ``python3`` image until someone reinstalls.

    Guarded like the others: only the plist the real passwd home produces is
    eligible (``launchctl`` always addresses the real user's session whatever
    ``HOME`` says), and the port comes off the plist being replaced, so a daemon
    on a non-default port stays there. There is deliberately no config-dir guard:
    a non-default config ROOT does not share this plist at all — it has its own
    label and its own path (:func:`label`, :func:`plist_path`), so a sandbox
    either finds no file or finds one whose identity check fails.

    Never raises. A platform without ``launchctl`` is reported unsupported: the
    systemd unit is re-read on every start and has no image name to go stale.
    """
    name = "browser bridge"
    try:
        if _supervisor() != "launchctl":
            return launchd.PlistRefresh(name=name, kind="unsupported")
        path = plist_path()
        if not launchd.is_own_plist(path, label()):
            return launchd.PlistRefresh(name=name, kind="not-addressable")
        # The port comes off the plist being REPLACED: a repair must not move a
        # daemon someone installed on a non-default port back to the default.
        port = launchd.int_arg(launchd.load(path), "--port", DEFAULT_PORT)
        outcome = launchd.rewrite_if_stale(name=name, path=path, rendered=render_plist(port))
        if outcome.kind != "repaired":
            return outcome
        # bootout + bootstrap through the shared helper, NOT kickstart -k:
        # measured — a kickstart after a rewrite restarts the job from launchd's
        # in-memory definition and keeps running the old argv. See
        # :mod:`local_operator.launchd`.
        reloaded = launchd.reload_job(label=label(), path=path, runner=_launchctl)
        if not reloaded.ok:
            # Names the recovery, because the job is DOWN at this point: see
            # `launchd.reload_failure`.
            return reloaded.as_refresh_failure(name=name, path=path, recovery="lop browser install")
        return outcome
    except Exception as exc:  # noqa: BLE001 — a repair must never fail an upgrade
        return launchd.PlistRefresh(name=name, kind="failed", detail=str(exc))


#: ``StandardOutput=append:`` landed in systemd 240 (upstream NEWS; confirmed
#: by the maintainer on systemd-devel). An older systemd does NOT refuse the
#: unit — measured on 255 against a deliberately invalid specifier, it logs
#: "Failed to parse output specifier, ignoring" and starts anyway — so the
#: real cost of emitting it blindly is subtler than a hard failure: the
#: directive is silently dropped, the output goes to the journal, and the
#: product would still be pointing users at a file nothing writes. That is the
#: exact defect being fixed, so the version gate is what keeps the emitted unit
#: and :func:`log_location` telling the same story on every systemd.
MIN_SYSTEMD_APPEND_VERSION = supervisors.MIN_SYSTEMD_APPEND_VERSION


class _Detect:
    """Sentinel type: "detect the version" as distinct from "it is unknown"."""


#: Default for ``render_systemd(version=...)``. See the note at its use site.
_DETECT = _Detect()


def systemd_version() -> int | None:
    """Major version of the running systemd, or ``None`` if it cannot be read.

    ``systemctl --version`` prints e.g. ``systemd 255 (255.4-1ubuntu8.17)``.
    Unknown degrades to "assume old", which is the safe direction: the unit
    stays loadable and the output goes to the journal.

    Delegates to :mod:`local_operator.supervisors`, and stays a module-level
    function here because this module's own tests patch it to exercise the
    ``append:`` version gate.
    """
    return supervisors.systemd_version()


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
    # A unit is re-executed on every restart, so its image must be a path that
    # survives a generation flip and a prune: ``procname.supervised_image``
    # answers with the stable shim when this machine has the layout, and ``None``
    # (either branded image or bare interpreter) when it does not.
    image = procname.supervised_image() or Path(sys.executable)
    command = f"{image} -m local_operator.browser_bridge.daemon --port {port}"
    # ``_DETECT`` and not ``None`` as the default: ``None`` is a MEANINGFUL
    # version value here ("systemd is present but its version could not be
    # read"), so overloading it to also mean "caller did not pass one" would
    # make that branch unreachable — and untestable — on any machine where
    # detection happens to succeed. Caught by CI, whose Linux runners detect a
    # modern systemd and so silently took the redirect path.
    resolved = systemd_version() if isinstance(version, _Detect) else version
    post_lines: list[str] = []
    if resolved is not None and resolved >= MIN_SYSTEMD_APPEND_VERSION:
        post_lines = [
            f"StandardOutput=append:{log_path()}",
            f"StandardError=append:{log_path()}",
        ]
    # One renderer for all four daemons (see supervisors.render_systemd_unit);
    # this call is the browser's parameterisation of it and its output is
    # byte-for-byte what this function wrote before.
    return supervisors.render_systemd_unit(
        description="Local Operator browser bridge",
        exec_start=command,
        post_lines=post_lines,
    )


def _logs_use_journal() -> bool:
    """Whether this platform's daemon output goes to the journal, not a file.

    One home for the decision because TWO callers ask it: :func:`logs_command`
    (which argv to run) and :func:`logs_read_the_log_file` (whether a missing
    file means anything). Kept private — callers ask their own question, not
    this one.
    """
    return _supervisor() == "systemctl" and not _log_file_is_written()


def logs_read_the_log_file() -> bool:
    """Whether this platform's ``logs_command`` reads :func:`log_path` itself.

    The CLI asks this to tell "the daemon has not run under a supervisor here"
    (no file, and the command WOULD read one) apart from "the command reads
    somewhere else entirely" (the journal, where an absent file says nothing).
    It used to key on ``command_line[0] == "tail"``, which identified the case
    only while every such platform ran ``tail`` — the Windows arm now runs
    PowerShell's ``Get-Content`` against a file the scheduled task really
    redirects into, so the literal name of one binary stopped being the
    question. The question was always "does this command read the log file?",
    and it is asked that way now.
    """
    return not _logs_use_journal()


def logs_command(lines: int = 100, *, follow: bool = False) -> list[str]:
    """The command that actually shows this platform's daemon output.

    Keeping this in one place is what stops the three surfaces disagreeing:
    on a systemd too old for ``append:`` the log file genuinely does not exist,
    and the honest answer is ``journalctl``, not a ``tail`` that cannot open it.

    WINDOWS has no ``tail`` AT ALL (audit C9), so the command this used to
    return on every non-systemd platform named an executable no machine there
    has — and because ``lop browser logs`` RUNS this argv rather than printing
    advice, the Windows surface failed with "cannot run `tail`" instead of
    showing the log. The platform's equivalent is PowerShell's
    ``Get-Content -Tail n [-Wait] <path>``, and the file it reads DOES exist
    there: the scheduled task's action redirects the daemon's stdout/stderr
    into :func:`log_path` (see ``supervisors.render_task_xml``).
    """
    if _logs_use_journal():
        command = ["journalctl", "--user", "-u", systemd_unit(), "-n", str(lines)]
        if follow:
            command.append("-f")
        return command
    if _IS_WINDOWS:
        # ``Get-Content`` is a CMDLET, not an executable: ``CreateProcess``
        # cannot start it, so the argv has to name the host that can run it.
        # Handing back the bare cmdlet would move the failure one layer down
        # (``subprocess.call`` -> FileNotFoundError) rather than fix it.
        # ``-NoProfile`` keeps the operator's profile out of a diagnostic.
        cmdlet = f"Get-Content -Tail {int(lines)}"
        if follow:
            cmdlet += " -Wait"
        # Single-quoted because PowerShell is the language here; a path holding
        # an apostrophe is escaped the way PowerShell escapes one.
        cmdlet += " '" + str(log_path()).replace("'", "''") + "'"
        return [_WINDOWS_POWERSHELL, "-NoProfile", "-Command", cmdlet]
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
    """``gui/<uid>`` — the launchd domain every caller feeds to ``launchctl``.

    The platform guard lives INSIDE this function rather than at its call
    sites, each of which is behind ``if kind == supervisors.LAUNCHCTL``. Two
    reasons: ``os.getuid`` does not exist off POSIX, so this was an
    ``AttributeError`` waiting for any caller whose arm was not checked; and a
    guard that lives at the call site is invisible — to a reader, and to the
    static scan that grades this branch — which cannot see that the ARM is
    unreachable. Here the function is safe to call anywhere on its own merits.
    """
    if procstate.is_windows():
        raise RuntimeError("launchd domains exist only on macOS")
    return f"gui/{os.getuid()}"


def _launchctl(*args: str) -> subprocess.CompletedProcess[str]:
    return subprocess.run(["launchctl", *args], capture_output=True, text=True, timeout=15)


#: What to tell a user whose platform has no user-level service supervisor.
#: One string shared by install/uninstall/start/stop/restart so the entry
#: points cannot drift into describing the same machine differently. Composed
#: by :mod:`local_operator.supervisors` (which lists every supervisor that
#: WOULD work, on every platform) with this daemon's own foreground command.
NO_SUPERVISOR_ERROR = supervisors.no_supervisor_error("lop browser serve")

#: Task Scheduler task name (Windows). Per-config-root like ``label()`` and for
#: the same reason: the user's task folder is a global namespace, so two roots
#: sharing one name would evict each other.
TASK_NAME = "Local Operator browser"


def task_name() -> str:
    """The Windows task name for this config root; the plain name for the default."""
    suffix = _root_suffix()
    return TASK_NAME if not suffix else f"{TASK_NAME}{suffix}"


def _registration_name() -> str:
    """How the supervisor answering here names this daemon's registration.

    The ``status`` payload's ``supervisor`` field, and the one place the three
    spellings are chosen. Keyed on :func:`_supervisor` — the capability — rather
    than on ``sys.platform``, which is the convention every other branch in this
    module already follows: the platform-only version named a launchd label on a
    darwin without ``launchctl`` and a systemd unit on a Linux without systemd,
    i.e. a registration that cannot exist.
    """
    kind = _supervisor()
    if kind == supervisors.LAUNCHCTL:
        return label()
    if kind == supervisors.SYSTEMCTL:
        return systemd_unit()
    if kind == supervisors.SCHTASKS:
        return task_name()
    return "none"


def task_record_path() -> Path:
    """Our own copy of the registered task (Task Scheduler keeps the original)."""
    return config_dir() / "browser-bridge-task.xml"


def _supervisor() -> str | None:
    """``"launchctl"``, ``"systemctl"``, ``"schtasks"``, or ``None``.

    Guarding on the BINARY rather than on ``sys.platform`` is the whole point.
    ``subprocess.run(..., check=False)`` suppresses a non-zero exit status but
    NOT ``FileNotFoundError`` when the executable is absent, so every caller
    that skipped this check raised an uncaught traceback out of the CLI on a
    Linux without systemd (Devuan, Alpine, Void, OpenRC, many containers, WSL2
    without systemd) and on win32, which fell into the same branch. Reproduced
    before this fix: ``start``, ``stop``, ``restart`` and ``uninstall`` all
    raised ``FileNotFoundError: 'systemctl'``.

    The discovery itself lives in :func:`local_operator.supervisors.supervisor`
    now that three other daemons need the same answer; this stays as the
    module-level seam the tests patch.
    """
    return supervisors.supervisor()


def _systemctl_user(*args: str) -> subprocess.CompletedProcess[str]:
    return supervisors.systemctl_user(*args)


def linger_remedy() -> str:
    """The shared "no user manager" remedy, plus THIS daemon's recovery command."""
    return supervisors.linger_remedy() + "then re-run `lop browser install`."


def _translate_systemctl_error(stderr: str) -> str:
    """Name the remedy for the one systemctl failure users actually hit.

    The raw stderr was surfaced verbatim — truthful, but it left the user to
    discover ``loginctl enable-linger`` on their own, and nothing in the
    product named it. ``limit=300`` is this module's historical truncation,
    kept so the messages this daemon prints do not change length.
    """
    return supervisors.translate_systemctl_error(stderr, remedy=linger_remedy(), limit=300)


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
    return supervisors.enable_linger()


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


def pin_driver(target: str, port: int | None = None, root: Path | None = None) -> dict[str, Any]:
    """Ask the daemon to make one authorised extension THE driver.

    The escape hatch from the incumbency rule (design §8.2): with two installs
    connected, the one already driving keeps the wheel, and reconnecting the
    other cannot take it — so without this command the operator's only lever is
    quitting a browser.

    Carries the discovery file's session key, exactly as the session leg does:
    moving the wheel to another browser is the same authority a session already
    holds, and the key is what keeps that decision off the loopback interface
    for any other local user.
    """
    current = state_store.read(root)
    resolved_port = port or (current.port if current else DEFAULT_PORT)
    if current is None or not state_store.pid_alive(current.pid):
        return {
            "ok": False,
            "error": "no running bridge daemon; run 'lop browser install' first.",
        }
    body = json.dumps({"target": target}).encode()
    request = urllib.request.Request(
        f"http://127.0.0.1:{resolved_port}/driver",
        data=body,
        method="POST",
        headers={"X-Bridge-Key": current.session_key, "Content-Type": "application/json"},
    )
    try:
        with urllib.request.urlopen(request, timeout=5.0) as response:
            payload: Any = json.loads(response.read().decode())
    except urllib.error.HTTPError as error:
        if error.code == 404:
            # TWO different 404s arrive here, and conflating them would send the
            # operator to restart a daemon that is perfectly current. The daemon's
            # own miss answers with a JSON `unknown_extension` body naming the ids
            # that ARE connected; a daemon with no /driver route at all answers
            # Starlette's plain-text 404.
            body_text = ""
            with suppress(Exception):
                body_text = error.read().decode()
            named = _ids_from_payload(body_text)
            if named or "unknown_extension" in body_text:
                # "no SINGLE" rather than "no" (copy review C1): the daemon
                # refuses an AMBIGUOUS target as well as an unmatched one, and
                # saying "nothing matched" about a target two installs matched
                # sends the user to re-check an id that is not the problem. Same
                # formulation `pair --revoke` already uses for the same refusal.
                #
                # `matches` lets the CALLER say which of the two happened (copy
                # review C8); absent on a daemon that predates the field, which is
                # why the sentence above stays as the default.
                return {
                    "ok": False,
                    "error": f"no single connected extension matches '{target}'.",
                    "authorized_extension_ids": named,
                    "matches": _int_from_payload(body_text, "matches"),
                }
            # An older daemon answers /health but has no /driver. Saying so beats
            # reporting a generic failure the user would read as "my id is wrong".
            return {
                "ok": False,
                "error": (
                    f"the daemon on port {resolved_port} predates this command. Run "
                    "'lop browser restart' to load the current build."
                ),
            }
        body_text = ""
        with suppress(Exception):
            body_text = error.read().decode()
        # A daemon sentence written FOR the reader beats a JSON dump around it
        # (copy review C5): `not_paired`/`not_connected`/`not_driving` all carry a
        # `message` that says what to do, and it arrived as the middle of a blob.
        # Parsed, not matched: a body without one falls through to the dump, which
        # is still better than swallowing an answer nobody predicted.
        message = _message_from_payload(body_text)
        if message:
            return {
                "ok": False,
                "error": message,
                "authorized_extension_ids": _ids_from_payload(body_text),
            }
        return {
            "ok": False,
            "error": f"daemon returned HTTP {error.code} on port {resolved_port}: {body_text}",
            "authorized_extension_ids": _ids_from_payload(body_text),
        }
    except Exception as error:  # noqa: BLE001 - report, do not raise, at a CLI edge
        return {
            "ok": False,
            "error": (
                f"daemon is not answering on port {resolved_port} ({error}); "
                "run 'lop browser restart'."
            ),
        }
    return {
        "ok": bool(payload.get("ok")),
        "driver_extension_id": payload.get("driver_extension_id", ""),
    }


def _message_from_payload(body_text: str) -> str:
    """The human sentence a failed /driver response carries, or "" if it has none.

    Only the daemon's own `message` field is used: it is written for the operator
    and is the actionable half of an otherwise machine-shaped body. Anything
    unexpected (no field, wrong type, unparseable) returns "" so the caller keeps
    printing the raw response rather than an invented error.
    """
    with suppress(Exception):
        parsed = json.loads(body_text)
        if isinstance(parsed, dict):
            message = parsed.get("message")
            if isinstance(message, str):
                return message.strip()
    return ""


def _int_from_payload(body_text: str, key: str) -> int | None:
    """One integer field from a failed response body, or None if it is not there.

    Best-effort like `_ids_from_payload`: a missing or non-integer field means the
    caller keeps its default wording rather than reporting a guess.
    """
    with suppress(Exception):
        parsed = json.loads(body_text)
        if isinstance(parsed, dict):
            value = parsed.get(key)
            if isinstance(value, bool):  # bool is an int subclass; not a count
                return None
            if isinstance(value, int):
                return value
    return None


def _ids_from_payload(body_text: str) -> list[str]:
    """The authorised ids a failed /driver response names, or [] if it names none.

    Best-effort: this only feeds the CLI's "did you mean" list, so an
    unparseable body must not turn one error into a different one.
    """
    with suppress(Exception):
        parsed = json.loads(body_text)
        listed = parsed.get("authorized_extension_ids")
        if isinstance(listed, list):
            return [str(item) for item in listed]
    return []


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


def _plist_is_current(path: Path, wanted: dict[str, object]) -> bool:
    """Whether the LaunchAgent on disk already says what ``wanted`` says.

    Content rather than existence: a plist from an older build names a different
    interpreter and port and must still be replaced. An unreadable or
    unparseable file answers ``False``, the rewrite direction.
    """
    try:
        return plistlib.loads(path.read_bytes()) == wanted
    except (OSError, ValueError):
        return False


def _systemd_unit_is_current(path: Path, rendered: str) -> bool:
    """The systemd twin of :func:`_plist_is_current`, text rather than plist."""
    try:
        return path.read_text(encoding="utf-8") == rendered
    except (OSError, ValueError):
        return False


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
        # WRITE AND RELOAD ONLY WHEN SOMETHING WOULD CHANGE, and what that claim
        # is and is not (review round 1, R-3): an install that would change
        # nothing no longer emits the two signals an EDR reads as "Persistence:
        # launchd job / plist file modification" (MITRE T1543.001) — an
        # identical-bytes rewrite followed by a bootout/bootstrap. It does NOT
        # explain the 2026-09-19 incident's plist modification, which came from
        # the `[daemons] refresh` child and a genuinely stale plist; see the
        # longer note in `mobile/install.py`.
        wanted = render_plist(port)
        current = _plist_is_current(plist_path(), wanted)
        if not dry_run and not current:
            plist_path().write_bytes(plistlib.dumps(wanted))
        steps.append(f"wrote {plist_path()}" if not current else "LaunchAgent already current")
        if not dry_run:
            # A DEAD OR WEDGED JOB IS STILL REPAIRED. The reload is skipped only
            # when the file is current, launchd holds a live pid for this label
            # AND the bridge is answering — the same liveness-plus-serving
            # predicate the mobile installer's twin uses. Liveness alone was not
            # enough: a job with a pid that no longer answers was left in place
            # and the install then failed its own health check below, which the
            # old unconditional reload used to repair (round 1, R-5/Q-1). A
            # health probe is safe to add HERE because `job_running` is asked
            # first: launchd can only hold a pid for OUR label, so a leftover
            # foreground daemon on the port cannot fake it.
            reload_needed = True
            healthy = (
                current
                and launchd.job_running(label=label(), path=plist_path(), run=_launchctl)
                and health(port) is not None
            )
            if healthy:
                reload_needed = False
                steps.append("LaunchAgent already current and running; left it loaded")
            elif current and launchd.kickstart(label=label(), path=plist_path(), run=_launchctl):
                reload_needed = False
                steps.append("restarted the loaded LaunchAgent (its file was already current)")
            if reload_needed:
                # The shared reload rather than an inline pair: it tolerates an
                # absent job, waits for launchd to release the label, retries past
                # the measured teardown race, and only then reports the load. The
                # old shape returned launchd's raw stderr as the whole error and, on
                # success, said "loaded the LaunchAgent" without checking. See
                # :mod:`local_operator.launchd`.
                reloaded = launchd.reload_job(label=label(), path=plist_path(), runner=_launchctl)
                if not reloaded.ok:
                    return {"ok": False, "steps": steps, "error": reloaded.detail[:300]}
                steps.append(f"loaded the LaunchAgent ({label()})")
    elif supervisor == "systemctl":
        systemd_path().parent.mkdir(parents=True, exist_ok=True)
        if not dry_run:
            # Same idea, minus the launchd half: an unchanged unit file is not
            # written again. The `daemon-reload` and `enable --now` below stay,
            # because they are the LOAD rather than the write and neither
            # restarts a unit that is already running.
            rendered = render_systemd(port)
            if not _systemd_unit_is_current(systemd_path(), rendered):
                systemd_path().write_text(rendered, encoding="utf-8")
                steps.append(f"wrote {systemd_path()}")
            else:
                steps.append(f"unit file already current ({systemd_path()})")
        else:
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
    elif supervisor == "schtasks":
        # Task Scheduler cannot redirect a task's stdout, so the log the other
        # two platforms get from launchd/systemd is provided by the task's own
        # command line (see supervisors.render_task_xml); log_path() stays the
        # one place every log surface points at on every platform.
        log_path().parent.mkdir(parents=True, exist_ok=True)
        if not dry_run:
            xml = render_task_xml(port)
            # The record's parent may not exist yet on a first install (it is the
            # config root itself), and an install must not fail on that.
            task_record_path().parent.mkdir(parents=True, exist_ok=True)
            task_record_path().write_text(xml, encoding="utf-8")
            ok, detail = supervisors.create_task(task_name(), xml)
            if not ok:
                return {
                    "ok": False,
                    "steps": steps,
                    "error": f"schtasks could not register the task: {detail}",
                }
            started = supervisors.schtasks(*supervisors.task_run_args(task_name()))
            if started.returncode:
                return {
                    "ok": False,
                    "steps": steps,
                    "error": (
                        "the task was registered but could not be started: "
                        f"{((started.stderr or started.stdout) or '').strip()[:300]}"
                    ),
                }
        steps.append(f"registered the scheduled task ({task_name()})")
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
    """Whether this config root has a supervisor registration under its OWN name."""
    supervisor = _supervisor()
    if supervisor == "launchctl":
        return plist_path().exists()
    if supervisor == "systemctl":
        return systemd_path().exists()
    if supervisor == "schtasks":
        # The task's own store, not our record file: the record is evidence an
        # install ran, and the question here is whether the task is registered.
        registered, _running, _detail = supervisors.task_state(task_name())
        return registered
    return False


def _legacy_path() -> Path | None:
    """Where a pre-per-root build running under THIS ``$HOME`` wrote its registration.

    ``Path.home()`` only, and the distinction from :func:`_default_config_root`
    is the whole point rather than an inconsistency:

    * :func:`_default_config_root` asks a UID-keyed question — "does this root
      own the default supervisor NAME?" The namespace it guards is launchd's
      ``gui/<uid>`` and systemd's ``--user`` instance, neither of which moves
      when ``$HOME`` does, so it must anchor on the passwd home.
    * This asks a HOME-keyed question — "did MY predecessor write this file?"
      A registration under a *different* ``$HOME`` belongs to a different run,
      and adopting it is not an upgrade path; it is one root claiming another
      root's supervisor.

    Searching the passwd home here as well looked like it served a second
    upgrade trigger, and instead recreated the incident this module exists to
    prevent: under the isolation AGENTS.md prescribes
    (``HOME=/tmp/... LOCAL_OPERATOR_CONFIG_DIR=...``) the lookup resolved the
    OPERATOR's live plist, and because such a root has no install of its own it
    would be adopted — then booted out, unlinked, and SIGTERM'd.

    Location alone is NOT ownership, which is why this is only the first of
    three tests; see :func:`legacy_registration`.
    """
    home = Path.home()
    if sys.platform == "darwin":
        return home / "Library" / "LaunchAgents" / f"{LABEL}.plist"
    if sys.platform.startswith("linux"):
        return home / ".config" / "systemd" / "user" / SYSTEMD_UNIT
    return None


def _canonical(path: str | Path) -> str:
    """A path in the one spelling ownership comparisons can use.

    Ownership is decided by comparing two recorded paths, so the comparison has
    to survive symlinks (``/tmp`` → ``/private/tmp`` on macOS) and ``..``/``.``
    segments. ``resolve()`` without ``strict`` also canonicalises a path whose
    final component no longer exists, which is the normal case for a log file
    that was never written.
    """
    try:
        return str(Path(path).expanduser().resolve())
    except OSError:  # pragma: no cover - unreadable parent
        return str(Path(path).expanduser())


def _recorded_log_path(registration: Path) -> str | None:
    """The log destination a supervisor file records, or ``None`` when it records none.

    This is the ownership evidence, and it exists because every build derives
    the daemon's stdout/stderr destination from :func:`log_path`, which honours
    ``LOCAL_OPERATOR_CONFIG_DIR`` unconditionally. A registration written under
    ``LOCAL_OPERATOR_CONFIG_DIR=/x`` therefore records ``/x/logs/...`` while the
    default root records the platform log dir — verified against the RELEASED
    writer at ``v0.51.14``, not merely against current code.

    ``None`` means the file carries no evidence, and callers must read that as
    "not provably mine" rather than "mine". That case is real and common on
    Linux: the released ``render_systemd`` emits ``ExecStart`` only, with no
    ``StandardOutput=``, so a legacy unit from any build before this one is
    journal-only and unattributable. Such a unit is never auto-adopted.
    """
    if sys.platform == "darwin":
        try:
            with registration.open("rb") as handle:
                parsed = plistlib.load(handle)
        except Exception:  # noqa: BLE001 - a corrupt plist is "no evidence", not a crash
            return None
        value = parsed.get("StandardOutPath") if isinstance(parsed, dict) else None
        return value if isinstance(value, str) and value else None
    try:
        text = registration.read_text(encoding="utf-8", errors="replace")
    except OSError:
        return None
    # `StandardOutput=append:<path>` is the only place a unit records this, and
    # this build emits it only on systemd >= 240; older systemd logs to the
    # journal, so the absence of this line is expected rather than anomalous.
    match = re.search(r"^StandardOutput=append:(.+)$", text, re.MULTILINE)
    return match.group(1).strip() if match else None


def _default_root_dir_exists() -> bool:
    """HEURISTIC: does the default config root's DIRECTORY exist?

    Deliberately named for what it measures. Directory existence proves neither
    that a default-root daemon is running nor that the default root wrote the
    registration under scrutiny — it is a conservative refusal signal, not
    ownership. It is kept because the cost of the two errors is wildly
    asymmetric: wrongly refusing leaves a stale file for the user to delete by
    hand, while wrongly claiming deletes a live LaunchAgent along with its
    ``RunAtLoad``/``KeepAlive``, so the bridge never returns after a reboot.

    What it legitimately blocks, and this is a real cost rather than a
    theoretical one: a user who has BOTH a populated ``~/.local-operator`` and a
    ``LOCAL_OPERATOR_CONFIG_DIR`` root whose own legacy registration is the one
    on disk cannot have it auto-removed, even when the evidence says it is
    theirs. They get :func:`legacy_ambiguity`'s message naming the file instead
    of a silent no-op. Positive evidence alone would serve that user; this
    refusal is the price of not trusting evidence that a pre-``append:`` Linux
    unit cannot supply at all.
    """
    return _default_config_root().exists()


def legacy_ambiguity() -> str | None:
    """Why a legacy registration that IS present was not claimed by this root.

    Ambiguity must reach the user as an actionable statement naming the file,
    never as a silent ``None`` that the caller then reports as "no service
    installed" — that is the false-success shape this module keeps re-learning.
    ``None`` here means there is genuinely nothing to report.
    """
    if not _root_suffix():
        return None
    candidate = _legacy_path()
    if candidate is None or not candidate.exists():
        return None
    if _own_registration_exists():
        return (
            f"{candidate} was left by an older build, but this config root has its "
            "own registration, which takes precedence. Left untouched."
        )
    recorded = _recorded_log_path(candidate)
    if recorded is None:
        noun = "unit" if sys.platform.startswith("linux") else "plist"
        return (
            f"{candidate} records no log destination, so nothing proves it belongs "
            f"to this config root (a pre-0.51 systemd {noun} logs to the journal and "
            "carries no such evidence). Left untouched — remove it by hand if it is "
            "yours."
        )
    if _canonical(recorded) != _canonical(log_path()):
        return (
            f"{candidate} was written by a different config root: its logs go to "
            f"{recorded}, this root's to {log_path()}. Left untouched."
        )
    if _default_root_dir_exists():
        return (
            f"{candidate} looks like this root's, but the default config root "
            f"{_default_config_root()} still exists and may own it. Left untouched. "
            "Inspect the named supervisor registration and confirm ownership before "
            "stopping or removing it. Keep all configuration and session data."
        )
    return None


def legacy_registration() -> Path | None:
    """A registration written by a pre-per-root build that this root now orphans.

    Only ever non-``None`` when this root's supervisor name is SUFFIXED: the
    default root's name is unchanged, so an existing install stays exactly
    where it was and is adopted, not orphaned. Two things produce a suffix and
    the usual cause being a ``LOCAL_OPERATOR_CONFIG_DIR``, which is a
    documented setting.

    Such a user upgrading from a released build has a registration under the
    SHARED default name that this build no longer addresses. Left unresolved
    that produced a false success: ``uninstall`` reported "no LaunchAgent was
    installed" and exited 0 while the daemon kept running — the same "claimed
    success it did not achieve" shape :func:`uninstall` was fixed to remove,
    reintroduced on a different path. So every entry point that manages a
    supervisor — install, uninstall, start/stop/restart, status — resolves it
    the same way, on both platforms.

    Claiming one requires POSITIVE, canonicalised evidence that it is this
    root's, never mere presence at the expected location. All of:

    1. discovery is scoped to this ``$HOME`` (:func:`_legacy_path`);
    2. this root has no registration of its own, which always takes precedence;
    3. the file records a log destination equal to this root's, and the default
       root's directory is absent (:func:`_default_root_dir_exists`, a labelled
       heuristic — see its docstring for the upgrade it legitimately blocks).

    Anything else returns ``None`` and is reported through
    :func:`legacy_ambiguity`, because the alternative is deleting a supervisor
    belonging to another root. Presence alone let ``uninstall`` from any
    non-default root remove the DEFAULT root's live LaunchAgent while reporting
    success — data loss behind a success message, which is the shape this
    module exists to prevent.
    """
    if not _root_suffix():
        return None
    legacy = _legacy_path()
    if legacy is None or not legacy.exists():
        return None
    if _own_registration_exists():
        return None
    recorded = _recorded_log_path(legacy)
    if recorded is None or _canonical(recorded) != _canonical(log_path()):
        return None
    if _default_root_dir_exists():
        return None
    return legacy


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
    # Unclaimable-but-present must not be reported as "nothing was installed".
    # The user asked this root to manage a supervisor and it declined; saying so
    # is the difference between an answer and a false success.
    ambiguity = legacy_ambiguity()
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
    elif supervisor == "schtasks":
        if not dry_run:
            deleted, detail = supervisors.delete_task(task_name())
            task_record_path().unlink(missing_ok=True)
            if not deleted:
                ok = False
                steps.append(f"could not remove the scheduled task: {detail}")
            else:
                steps.append(f"removed the scheduled task ({task_name()})")
        else:
            steps.append("removed the scheduled task")
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
    result: dict[str, object] = {"ok": ok, "steps": steps}
    if ambiguity is not None:
        # Not an error — the removal this root COULD do succeeded — but the user
        # must be told what was found and left, with the path, so "uninstalled"
        # never silently means "and something of yours is still registered".
        result["warning"] = ambiguity
    return result


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
    if supervisor == "schtasks":
        name = task_name()
        if action in ("stop", "restart"):
            # `/End` on a task that is not running exits non-zero, which for a
            # stop is the state the caller asked for; only an unregistered task
            # is a real failure, and its stderr says which it was.
            ended = supervisors.schtasks(*supervisors.task_end_args(name))
            if ended.returncode:
                detail = ((ended.stderr or ended.stdout) or "").strip()
                if "running" not in detail.lower():
                    return {"ok": False, "error": detail[:300]}
        if action in ("start", "restart"):
            started = supervisors.schtasks(*supervisors.task_run_args(name))
            return {
                "ok": started.returncode == 0,
                "error": ((started.stderr or started.stdout) or "").strip()[:300],
            }
        return {"ok": True, "error": ""}
    result = _systemctl_user(action, unit)
    if result.returncode and action in ("start", "restart") and systemd_path().exists():
        _systemctl_user("daemon-reload")
        result = _systemctl_user(action, unit)
    return {"ok": result.returncode == 0, "error": _translate_systemctl_error(result.stderr)}


def render_task_xml(port: int = DEFAULT_PORT) -> str:
    """The Windows Task Scheduler task for this daemon.

    No ``environment``: this daemon's plist and unit record nothing either — the
    config root travels in the task NAME (see :func:`task_name`) and in the log
    path — so the Windows arm matches the other two rather than inventing a
    third contract. The log IS set: Task Scheduler has no stdout redirection, so
    without it a Windows user has no daemon output at all while every log
    surface still points at ``log_path()``.
    """
    image = procname.supervised_image() or Path(sys.executable)
    return supervisors.render_task_xml(
        description="Local Operator browser bridge",
        image=str(image),
        argv=["-m", "local_operator.browser_bridge.daemon", "--port", str(port)],
        log=log_path(),
        user_id=supervisors.current_user_id(),
    )


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
        "identities": pairing["identities"],
        "pending_code": pairing["pending_code"],
        "pending": pairing["pending"],
        "pending_expires_at": pairing["pending_expires_at"],
        # The location the output is ACTUALLY readable from, which on a systemd
        # too old for `append:` is the journal, not a file that never exists.
        "log": log_location(),
        # Which supervisor REGISTRATION this is reporting on, spelled the way
        # that supervisor names it: two isolated daemons otherwise produce
        # identical status output with no way to tell which instance you are
        # talking to. Chosen by the CAPABILITY (``_supervisor()``), not by
        # ``sys.platform``: a darwin without ``launchctl`` has no label and a
        # systemd-less Linux has no unit, and naming one anyway reported a
        # registration that cannot exist. ``none`` is the honest answer for a
        # host with no supervisor at all — which is a state this module's other
        # branches already handle rather than assume away.
        "supervisor": _registration_name(),
        "config_root": str(config_dir()),
        # Named, not merely folded into `installed`: this is the one thing that
        # explains why a daemon is running under a name the CLI would not
        # otherwise mention, and it tells the user which file to act on.
        "legacy_registration": str(orphan) if orphan is not None else None,
        # A present-but-unclaimable registration is the case a user most needs
        # named: it explains a daemon this root can neither see nor manage.
        "legacy_ambiguity": legacy_ambiguity(),
    }
