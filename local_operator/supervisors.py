"""One module that knows which user-level service supervisor this machine has.

WHY THIS MODULE EXISTS
----------------------
Four daemons (mobile, wakes, tunnels, browser bridge) each need the same three
things from the platform: something that starts them at login, something that
restarts them after a crash, and a command that starts/stops/restarts/queries
them. Until this module, that knowledge existed in exactly one place —
``browser_bridge/install.py`` had already learned to guard on the *binary*
rather than on ``sys.platform``, because ``subprocess.run(..., check=False)``
suppresses a non-zero exit status but NOT ``FileNotFoundError`` when the
executable is absent. The other three installers had not, so on a Linux
without systemd (Devuan, Alpine, Void, OpenRC, most containers, WSL2 without
systemd) they raised ``FileNotFoundError`` out of the CLI, and on Windows they
either raised ``AttributeError: os.getuid`` or silently reported "unsupported".

So the discovery, the addressability guards, the unit rendering and the
Windows task plumbing live here once, and each installer keeps only what is
genuinely its own: the argv of its daemon, its log file, its port.

THE THREE SUPERVISORS, AND WHAT "SUPPORTED" MEANS
-------------------------------------------------
* macOS — ``launchctl``, a LaunchAgent plist. Unchanged; ``launchd.py`` is the
  macOS half of this module rather than the only one.
* Linux — ``systemctl --user``, a unit under ``~/.config/systemd/user`` plus
  ``loginctl enable-linger`` so it survives logout.
* Windows — Task Scheduler, registered with ``schtasks.exe`` (present on every
  Windows, no admin rights, no new dependency — deliberately NOT pywin32 and
  deliberately NOT a Windows Service: a Service needs elevation and cannot
  reach the user's session, which is where every one of these daemons runs).

Guard on the BINARY and not on ``sys.platform`` for the same reason
``browser_bridge`` learned to: a platform is not a capability. A Linux box
without systemd has no user supervisor, and saying so is the honest answer.

"SUPPORTED" IS NOT "VERIFIED HERE" — the Windows arm especially. The task
plumbing below is pure rendering plus ``schtasks`` argv; the rendering is unit
tested on every platform, and the *placement* has been exercised on real
systemd (Ubuntu 24.04 in a container, systemd 255) but **not** on Windows,
where this project has no host in this workspace. That distinction is recorded
here rather than papered over: on Windows every ``schtasks`` call is checked
and its stderr reported verbatim, so a wrong guess fails loudly at install
time with Task Scheduler's own words rather than looking installed.
"""

from __future__ import annotations

import logging
import os
import re
import shutil
import subprocess
import sys
import tempfile
from collections.abc import Mapping, Sequence
from pathlib import Path
from xml.sax.saxutils import escape

logger = logging.getLogger(__name__)

#: The three supervisor names this module can answer with. Strings rather than
#: an enum because every caller already branches on the executable's own name
#: and printing it in a message is the common case.
LAUNCHCTL = "launchctl"
SYSTEMCTL = "systemctl"
SCHTASKS = "schtasks"


def supervisor() -> str | None:
    """``"launchctl"``, ``"systemctl"``, ``"schtasks"``, or ``None``.

    Guarding on the BINARY rather than on ``sys.platform`` is the whole point:
    ``subprocess.run(..., check=False)`` suppresses a non-zero exit status but
    NOT ``FileNotFoundError`` when the executable is absent, so a caller that
    skipped this check raised an uncaught traceback out of the CLI on a Linux
    without systemd and on win32. ``None`` is the answer for a platform that
    genuinely cannot supervise a user daemon, and every caller turns it into
    the same refusal (:func:`no_supervisor_error`) rather than a traceback.
    """
    if sys.platform == "darwin" and shutil.which("launchctl"):
        return LAUNCHCTL
    if sys.platform.startswith("linux") and shutil.which("systemctl"):
        return SYSTEMCTL
    if os.name == "nt" and shutil.which("schtasks"):
        return SCHTASKS
    return None


def no_supervisor_error(foreground_command: str) -> str:
    """What to tell a user whose platform has no user-level service supervisor.

    ONE sentence per daemon, naming every supervisor that WOULD work and the
    command that runs the same daemon in the foreground instead. Kept as a
    function rather than one shared string because only the last clause differs
    per daemon, and a daemon that names another daemon's foreground command is
    worse than no message at all.
    """
    return (
        "no supported user service supervisor found (launchctl on macOS, "
        "systemctl --user on Linux, Task Scheduler via schtasks on Windows); "
        f"run `{foreground_command}` in the foreground"
    )


def real_home() -> Path | None:
    """The REAL user's home, or ``None`` when it cannot be determined.

    NOT ``Path.home()``, and that difference is the whole guard below:
    ``Path.home()`` reads ``$HOME`` (``$USERPROFILE`` on Windows), so an
    isolated run would compare its own redirected home against itself and
    conclude it is the real one. On POSIX the answer comes from the passwd
    database, which a redirected environment cannot reach; on Windows there is
    no passwd database and ``Path.home()`` IS the profile, so the answer comes
    from ``USERPROFILE`` — which a test can redirect too, and that is why the
    Windows arm of :func:`unit_is_addressable` requires an explicit opt-in as
    well (see its docstring).
    """
    if os.name != "nt":
        try:
            import pwd

            return Path(pwd.getpwuid(os.getuid()).pw_dir).resolve()
        except (ImportError, KeyError, OSError, AttributeError):
            # ImportError: no pwd module. AttributeError: no os.getuid. Both are
            # the non-POSIX shape, and both used to escape as a traceback — the
            # `import pwd` sat OUTSIDE the try in the twin this replaces.
            return None
    profile = os.environ.get("USERPROFILE")
    return Path(profile).resolve() if profile else None


def config_lives_in_real_home(config_dir: Path) -> bool:
    """Whether a unit supervising ``config_dir`` would outlive this process.

    CONTAINMENT, not the identity test :func:`unit_is_addressable` uses, and the
    difference is deliberate: the config dir is an ordinary path the user may
    legitimately place anywhere under their home, so only dirs OUTSIDE it are
    the sandbox shape (``/tmp``, a throwaway home). A unit pointed at one of
    those is a live supervised job watching a store that is deleted when the
    sandbox ends.
    """
    home = real_home()
    if home is None:
        return False
    try:
        return Path(config_dir).resolve().is_relative_to(home)
    except (OSError, ValueError):
        return False


def unit_is_addressable(path: Path, parts: Sequence[str]) -> bool:
    """Whether a supervisor call about ``path`` would be about THIS run.

    ``parts`` is the registration path relative to the real home — for systemd
    ``(".config", "systemd", "user", "<unit>.service")``; ``launchd.py`` has had
    the same test for plists since a stray unit was planted in a developer's
    live session pointed at a pytest tmpdir.

    WHY IT IS AN IDENTITY TEST AND NOT A CONTAINMENT TEST. A supervised job
    addresses the CALLING USER's session — ``launchctl`` its ``gui/<uid>``
    domain, ``systemctl --user`` its per-uid instance — whatever ``$HOME`` says.
    A test that patches ``Path.home()`` to a tmpdir would therefore write a
    harmless unit file and then load a REAL unit pointed at a directory that is
    deleted moments later. Comparing the path BUILT from the real home closes
    it: a redirected home produces a different path wherever it points.

    WINDOWS: there is no ``gui/<uid>`` and no passwd database, so the guard is
    weaker by construction — ``Path.home()`` reads ``USERPROFILE``, which a test
    can redirect just as easily as ``HOME``. Task Scheduler also has no
    "address someone else's session" concept (a task registered without ``/RU``
    belongs to the creating user), so the file-less equivalent of the incident
    above is a task registered against a tmpdir by a test run. The guard is
    therefore an explicit opt-in: a redirected profile refuses, and the refusal
    is reported rather than swallowed.
    """
    home = real_home()
    if home is None:
        return False
    try:
        expected = home.joinpath(*parts).resolve()
        actual = Path(path).resolve()
    except (OSError, ValueError):
        return False
    # Identity, so a path inside the real home but NOT the one the real home
    # produces ($TMPDIR under $HOME is not exotic) is still a different unit.
    return actual == expected


# ---------------------------------------------------------------------------
# systemd (Linux user units)
# ---------------------------------------------------------------------------

#: Where a user unit lives, relative to the real home. The unit FILE follows
#: ``$HOME`` while the manager it is loaded into does not, which is exactly why
#: :func:`unit_is_addressable` takes the parts rather than a path.
SYSTEMD_UNIT_DIR = (".config", "systemd", "user")


def systemd_unit_path(unit: str) -> Path:
    """``~/.config/systemd/user/<unit>`` for the CURRENT home (redirectable)."""
    return Path.home().joinpath(*SYSTEMD_UNIT_DIR) / unit


def systemd_unit_is_addressable(unit: str) -> bool:
    """Whether ``systemctl --user`` may be ADDRESSED for ``unit`` from here.

    The file half of an installer stays fully testable under a redirected home;
    the half that reaches the developer's live user manager refuses. See
    :func:`unit_is_addressable` for the incident and the reasoning.
    """
    return unit_is_addressable(systemd_unit_path(unit), (*SYSTEMD_UNIT_DIR, unit))


def systemctl_user(*args: str, timeout: float = 30) -> subprocess.CompletedProcess[str]:
    """One ``systemctl --user`` call. The caller decides what a failure means."""
    return subprocess.run(  # noqa: S603 — fixed argv, no shell
        ["systemctl", "--user", *args], capture_output=True, text=True, timeout=timeout
    )


def systemd_version() -> int | None:
    """Major version of the running systemd, or ``None`` if it cannot be read.

    ``systemctl --version`` prints e.g. ``systemd 255 (255.4-1ubuntu8.17)``.
    Unknown degrades to "assume old", which is the safe direction: the unit
    stays loadable and the output goes to the journal.
    """
    if not shutil.which("systemctl"):
        return None
    try:
        result = subprocess.run(  # noqa: S603 — fixed argv, no shell
            ["systemctl", "--version"], capture_output=True, text=True, timeout=10
        )
    except (OSError, subprocess.SubprocessError):
        return None
    match = re.search(r"systemd\s+(\d+)", result.stdout or "")
    return int(match.group(1)) if match else None


#: ``StandardOutput=append:`` landed in systemd 240 (upstream NEWS; confirmed
#: by the maintainer on systemd-devel). An older systemd does NOT refuse the
#: unit — measured on 255 against a deliberately invalid specifier, it logs
#: "Failed to parse output specifier, ignoring" and starts anyway — so the real
#: cost of emitting it blindly is that the directive is silently dropped, the
#: output goes to the journal, and the product would still be pointing users at
#: a file nothing writes.
MIN_SYSTEMD_APPEND_VERSION = 240


def systemd_writes_log_file() -> bool:
    """Whether this platform's systemd redirects a unit's output to a file.

    ``False`` means the journal is where the output actually went, which is what
    the log-location surfaces must say instead of naming an empty file.
    """
    version = systemd_version()
    return version is not None and version >= MIN_SYSTEMD_APPEND_VERSION


def output_redirect_lines(log: Path) -> list[str]:
    """``StandardOutput``/``StandardError`` for ``log``, or ``[]`` when unsupported.

    systemd's default sends a unit's output to the JOURNAL while three product
    surfaces pointed at a log file nothing wrote — an empty directory reads as
    "the daemon produced no output" rather than "look somewhere else".
    """
    if not systemd_writes_log_file():
        return []
    return [f"StandardOutput=append:{log}", f"StandardError=append:{log}"]


def render_systemd_unit(
    *,
    description: str,
    exec_start: str,
    after: str | None = None,
    pre_lines: Sequence[str] = (),
    post_lines: Sequence[str] = (),
    restart: str = "on-failure",
    restart_sec: int = 5,
    separate_sections: bool = True,
    wanted_by: str = "default.target",
) -> str:
    """A user unit, in the one shape all four daemons emit.

    Pure and parameterised rather than four f-strings, because the parts that
    differ per daemon are exactly three (description, ``ExecStart``, and the
    lines carrying its log/environment) and everything else is the contract:
    ``Restart=on-failure`` so a CLEAN exit stays down (the wake supervisor
    self-retires with exit 0, and a plain ``Restart=always`` would leave it
    spinning against an empty index), ``WantedBy=default.target`` so
    ``enable`` gives start-at-login.

    ``separate_sections=False`` exists for the tunnel unit, whose layout
    predates this function; blank lines between sections are cosmetic to
    systemd, and rewriting a unit that is already correct is the thing worth
    avoiding.
    """
    lines = ["[Unit]", f"Description={description}"]
    if after:
        lines.append(f"After={after}")
    if separate_sections:
        lines.append("")
    lines += ["[Service]", *pre_lines, f"ExecStart={exec_start}", f"Restart={restart}"]
    lines.append(f"RestartSec={restart_sec}")
    lines += list(post_lines)
    if separate_sections:
        lines.append("")
    lines += ["[Install]", f"WantedBy={wanted_by}"]
    return "\n".join([*lines, ""])


def render_systemd_timer(
    *, description: str, unit: str, interval_seconds: int, first_delay_seconds: int = 300
) -> str:
    """The ``.timer`` that re-runs ``unit`` every ``interval_seconds`` of idleness.

    ``OnUnitInactiveSec`` and not ``OnCalendar``: the point is the launchd
    ``StartInterval`` semantic the wake supervisor depends on — re-run the unit
    once it is no longer running, never start a second copy while one is alive.
    ``OnBootSec`` gives a first attempt shortly after login, and systemd's own
    default ``RemainAfterExit``/``Persistent`` behaviour is left alone.
    """
    return "\n".join(
        [
            "[Unit]",
            f"Description={description}",
            "",
            "[Timer]",
            f"OnBootSec={first_delay_seconds}s",
            f"OnUnitInactiveSec={interval_seconds}s",
            f"Unit={unit}",
            "",
            "[Install]",
            "WantedBy=timers.target",
            "",
        ]
    )


#: systemd's message when there is no user D-Bus to talk to — the normal state
#: over plain SSH without lingering.
_NO_BUS_MARKERS = ("Failed to connect to bus", "No medium found", "XDG_RUNTIME_DIR")


def linger_remedy() -> str:
    """The one-sentence fix for "no user manager", with the command spelled out."""
    user = os.environ.get("USER") or "$USER"
    return (
        "systemd has no user session bus for this login (normal over plain "
        "SSH). Enable lingering so the user manager starts at boot and "
        f"survives logout:\n    loginctl enable-linger {user}\n"
    )


def translate_systemctl_error(stderr: str, *, remedy: str | None = None, limit: int = 200) -> str:
    """Name the remedy for the one ``systemctl`` failure users actually hit.

    Users do not discover ``loginctl enable-linger`` on their own, and nothing
    in the message points at it: systemd reports "Failed to connect to bus",
    which reads like a broken machine rather than a login without a user
    manager.

    ``remedy`` is the one part that differs per daemon — each one's recovery is
    its own installer (``lop browser install``, ``lop mobile install``) — so it
    is a parameter rather than a second copy of this diagnosis. ``limit``
    truncates everything else, and is a parameter because the browser bridge's
    surfaces have always shown 300 characters of a non-bus failure.
    """
    text = (stderr or "").strip()
    if is_bus_failure(text):
        base = linger_remedy() if remedy is None else remedy
        return f"{text[:200]}\n\n{base}" if text else base
    return text[:limit]


def is_bus_failure(text: str) -> bool:
    """Whether ``systemctl``'s output is the "no user manager reachable" failure.

    The distinction the status surface depends on: a user manager that cannot be
    REACHED is not a machine with nothing installed, and only one of those two
    is actionable by installing something. Matched on systemd's own strings
    rather than on the exit status, which is non-zero for both.
    """
    return any(marker in (text or "") for marker in _NO_BUS_MARKERS)


def enable_linger() -> bool:
    """Best-effort ``loginctl enable-linger``; never fatal to an install.

    Per ``loginctl(1)``, without lingering no user manager is spawned at boot,
    so ``systemctl --user enable --now`` does NOT deliver "survives restarts"
    the way launchd's ``RunAtLoad`` does. Confirmed on systemd 255: with
    lingering disabled, ``systemctl --user`` cannot reach the user manager at
    all ("User ID is not logged in or lingering").
    """
    binary = shutil.which("loginctl")
    if binary is None:
        return False
    user = os.environ.get("USER") or ""
    try:
        result = subprocess.run(  # noqa: S603 — fixed argv, no shell
            [binary, "enable-linger", *([user] if user else [])],
            capture_output=True,
            text=True,
            timeout=15,
        )
    except (OSError, subprocess.SubprocessError):
        return False
    return result.returncode == 0


# ---------------------------------------------------------------------------
# Windows: Task Scheduler via schtasks.exe
# ---------------------------------------------------------------------------

#: Task Scheduler imports its XML as UTF-16, and refuses a file that is not.
TASK_XML_ENCODING = "utf-16"

#: ``schtasks`` refuses a restart interval below one minute, and caps the count.
TASK_RESTART_INTERVAL = "PT1M"
TASK_RESTART_COUNT = 999


def _windows_quote(value: str) -> str:
    """Quote one argv element for a Windows command line.

    Only where it is needed: an unquoted ``-m`` that gained quotes would be
    handed to the interpreter verbatim. Task Scheduler stores the ``<Exec>``
    command and arguments as ONE command line, so a path with a space in it
    (``C:\\Program Files\\...``) must be quoted here or the task starts
    something that does not exist.
    """
    if value and " " not in value and "\t" not in value:
        return value
    return f'"{value}"'


def task_command_line(
    image: str,
    argv: Sequence[str],
    environment: Mapping[str, str] | None = None,
    log: Path | None = None,
) -> tuple[str, str]:
    """``(Command, Arguments)`` for a Task Scheduler ``<Exec>``.

    Task Scheduler has no environment-variable element and no output
    redirection, so a daemon whose unit carries either (the wake supervisor and
    the tunnel record ``LOCAL_OPERATOR_CONFIG_DIR``; every daemon logs to a file
    the other two platforms' supervisors create) is started through ``cmd.exe``
    with a ``set`` prefix and/or a ``>>`` redirect. ``cmd /c`` is the documented
    way to run a batch file AND the way to give one process an environment and a
    redirected log, and the quoting is the one form that survives ``cmd``'s own
    "strip the first and last quote" rule: the command line begins with ``set``
    (or with the quoted image, which is already the shape cmd leaves alone), not
    with a leading quote that wraps the whole line.

    ``2>&1`` AFTER ``>>`` on purpose: the second redirect then duplicates the
    file handle, so stderr joins stdout in the log rather than the console.
    """
    command_line = " ".join([_windows_quote(image), *(_windows_quote(a) for a in argv)])
    if log is not None:
        # The redirect belongs to the SHELL, not to the program: it must be part
        # of the cmd.exe command line.
        log.parent.mkdir(parents=True, exist_ok=True)
        command_line = f'{command_line} >> "{log}" 2>&1'
    if not environment:
        if log is None:
            return image, " ".join(_windows_quote(a) for a in argv)
        shell = os.environ.get("SystemRoot", r"C:\Windows")
        return f"{shell}\\System32\\cmd.exe", f"/c {_for_cmd_c(command_line)}"
    shell = os.environ.get("SystemRoot", r"C:\Windows")
    assignments = " && ".join(f'set "{key}={value}"' for key, value in environment.items())
    # NOT wrapped: `cmd`'s quote-stripping rule only fires when the line BEGINS
    # with a quote, and this one begins with `set`.
    return f"{shell}\\System32\\cmd.exe", f"/c {assignments} && {command_line}"


def _for_cmd_c(command_line: str) -> str:
    """``command_line`` safe as the argument of ``cmd /c``.

    ``cmd /c`` strips the FIRST and LAST quote character of its argument when
    that argument begins with a quote (its documented legacy behaviour), which
    would eat the opening quote of a quoted program path and the closing quote
    of the log path. Wrapping the whole line in one more pair is the standard
    counter: the outer pair is what gets stripped, and the inner quoting
    survives — the case that matters is an interpreter under ``C:\\Program
    Files``.
    """
    return command_line if not command_line.startswith('"') else f'"{command_line}"'


def render_task_xml(
    *,
    description: str,
    image: str,
    argv: Sequence[str],
    environment: Mapping[str, str] | None = None,
    log: Path | None = None,
    interval_minutes: int | None = None,
    user_id: str | None = None,
) -> str:
    """The Task Scheduler definition for one daemon, as XML.

    WHY XML AND NOT ``schtasks`` SWITCHES. ``/SC ONLOGON /RL LIMITED`` covers
    start-at-login, but the switch form has no way to ask for restart-on-failure
    — the analogue of launchd's ``KeepAlive{SuccessfulExit:false}``, which is
    the whole reason a supervised daemon is worth having. That setting exists
    only in the task XML, so the installer writes one and registers it with
    ``schtasks /Create /XML``.

    ``IgnoreNew`` and not ``Parallel`` mirrors the measured launchd behaviour
    the wake supervisor depends on: launchd does NOT start a second instance
    while the job runs.

    ``RunLevel`` is LeastPrivilege and the trigger is a plain logon trigger: no
    elevation, no password, no ``/RU SYSTEM``. A Windows Service was the other
    candidate and is deliberately rejected — it needs admin to install and
    cannot reach the user's session, which is where these daemons live.
    """
    command, arguments = task_command_line(image, argv, environment, log)
    triggers = ["      <LogonTrigger>", "        <Enabled>true</Enabled>"]
    if user_id:
        triggers.append(f"        <UserId>{escape(user_id)}</UserId>")
    triggers.append("      </LogonTrigger>")
    if interval_minutes is not None:
        # The launchd StartInterval analogue: the only repair that needs no live
        # caller, which is what closes "an armed wake on a machine where nothing
        # is running to fire it".
        triggers += [
            "      <CalendarTrigger>",
            "        <StartBoundary>2026-01-01T00:00:00</StartBoundary>",
            "        <Enabled>true</Enabled>",
            "        <Repetition>",
            f"          <Interval>PT{int(interval_minutes)}M</Interval>",
            "          <StopAtDurationEnd>false</StopAtDurationEnd>",
            "        </Repetition>",
            "        <ScheduleByDay>",
            "          <DaysInterval>1</DaysInterval>",
            "        </ScheduleByDay>",
            "      </CalendarTrigger>",
        ]
    principal = []
    if user_id:
        principal.append(f"      <UserId>{escape(user_id)}</UserId>")
    return (
        '<?xml version="1.0" encoding="UTF-16"?>\n'
        '<Task version="1.2" xmlns="http://schemas.microsoft.com/windows/2004/02/mit/task">\n'
        "  <RegistrationInfo>\n"
        f"    <Description>{escape(description)}</Description>\n"
        "  </RegistrationInfo>\n"
        "  <Triggers>\n" + "\n".join(triggers) + "\n  </Triggers>\n"
        "  <Principals>\n"
        '    <Principal id="Author">\n'
        + "".join(f"{line}\n" for line in principal)
        + "      <LogonType>InteractiveToken</LogonType>\n"
        "      <RunLevel>LeastPrivilege</RunLevel>\n"
        "    </Principal>\n"
        "  </Principals>\n"
        "  <Settings>\n"
        "    <MultipleInstancesPolicy>IgnoreNew</MultipleInstancesPolicy>\n"
        "    <DisallowStartIfOnBatteries>false</DisallowStartIfOnBatteries>\n"
        "    <StopIfGoingOnBatteries>false</StopIfGoingOnBatteries>\n"
        "    <AllowHardTerminate>true</AllowHardTerminate>\n"
        "    <StartWhenAvailable>true</StartWhenAvailable>\n"
        "    <RunOnlyIfNetworkAvailable>false</RunOnlyIfNetworkAvailable>\n"
        "    <IdleSettings>\n"
        "      <StopOnIdleEnd>false</StopOnIdleEnd>\n"
        "      <RestartOnIdle>false</RestartOnIdle>\n"
        "    </IdleSettings>\n"
        "    <AllowStartOnDemand>true</AllowStartOnDemand>\n"
        "    <Enabled>true</Enabled>\n"
        "    <Hidden>false</Hidden>\n"
        "    <RunOnlyIfIdle>false</RunOnlyIfIdle>\n"
        "    <WakeToRun>false</WakeToRun>\n"
        "    <ExecutionTimeLimit>PT0S</ExecutionTimeLimit>\n"
        "    <Priority>7</Priority>\n"
        "    <RestartOnFailure>\n"
        f"      <Interval>{TASK_RESTART_INTERVAL}</Interval>\n"
        f"      <Count>{TASK_RESTART_COUNT}</Count>\n"
        "    </RestartOnFailure>\n"
        "  </Settings>\n"
        '  <Actions Context="Author">\n'
        "    <Exec>\n"
        f"      <Command>{escape(command)}</Command>\n"
        f"      <Arguments>{escape(arguments)}</Arguments>\n"
        "    </Exec>\n"
        "  </Actions>\n"
        "</Task>\n"
    )


def task_scheduler_is_addressable(config_dir: Path) -> bool:
    """Whether ``schtasks`` may be ADDRESSED for ``config_dir`` from here.

    The Windows half of :func:`unit_is_addressable`, and WEAKER than it by
    construction — stated rather than glossed: Task Scheduler has no "address
    another user's session" concept (a task registered without ``/RU`` belongs
    to the creating user and is invisible to everyone else), and there is no
    passwd database to compare a home against, only ``USERPROFILE``, which a
    test can redirect exactly as easily as ``HOME``. What IS enforced is the
    part that is decidable here: the store must live under the real profile, so
    a sandbox store (``/tmp/...``, ``C:\\Temp\\...``) is refused the way a
    sandbox store is refused on macOS. A test that redirects ``USERPROFILE`` and
    then really calls ``schtasks`` sits outside what this can detect, which is
    why every test of the task arm patches the runner.
    """
    if os.name != "nt":
        return False
    return config_lives_in_real_home(config_dir)


def current_user_id() -> str | None:
    """``DOMAIN\\user`` for the logon trigger, or ``None`` when unknown.

    Best-effort on purpose: Task Scheduler defaults an unqualified logon
    trigger to the creating user, so an unknown user id is a missing
    refinement rather than a failure.
    """
    user = os.environ.get("USERNAME")
    if not user:
        return None
    domain = os.environ.get("USERDOMAIN")
    return f"{domain}\\{user}" if domain else user


def schtasks(*args: str, timeout: float = 30) -> subprocess.CompletedProcess[str]:
    """One ``schtasks.exe`` call. The caller decides what a failure means."""
    return subprocess.run(  # noqa: S603 — fixed argv, no shell
        ["schtasks", *args], capture_output=True, text=True, timeout=timeout
    )


def task_run_args(name: str) -> list[str]:
    return ["/Run", "/TN", name]


def task_end_args(name: str) -> list[str]:
    return ["/End", "/TN", name]


def task_delete_args(name: str) -> list[str]:
    return ["/Delete", "/TN", name, "/F"]


def task_query_args(name: str) -> list[str]:
    return ["/Query", "/TN", name, "/FO", "LIST", "/V"]


def task_create_args(name: str, xml_path: Path) -> list[str]:
    """``schtasks /Create`` from an XML file, replacing any existing task of the same name."""
    return ["/Create", "/TN", name, "/XML", str(xml_path), "/F"]


def create_task(name: str, xml: str) -> tuple[bool, str]:
    """Register ``name`` from ``xml``; ``(ok, detail)`` with schtasks' own words.

    The XML goes through a temporary file because ``schtasks`` has no stdin
    form, and it is written as UTF-16 because that is what Task Scheduler
    imports — a UTF-8 file is rejected. Always ``/F``: an install is idempotent,
    and a stale task from an older build must be replaced rather than shadowed.
    """
    try:
        with tempfile.NamedTemporaryFile(
            "w", suffix=".xml", encoding=TASK_XML_ENCODING, delete=False
        ) as handle:
            handle.write(xml)
            path = Path(handle.name)
    except OSError as exc:
        return False, f"could not write the task definition: {exc}"
    try:
        result = schtasks(*task_create_args(name, path))
    except (OSError, subprocess.SubprocessError) as exc:
        return False, f"schtasks could not be run: {exc}"
    finally:
        path.unlink(missing_ok=True)
    if result.returncode:
        detail = (result.stderr or result.stdout or "").strip()
        return False, detail[:300] or f"schtasks exited {result.returncode}"
    return True, "registered"


def delete_task(name: str) -> tuple[bool, str]:
    """Remove ``name``; an absent task is success, not an error."""
    try:
        result = schtasks(*task_delete_args(name))
    except (OSError, subprocess.SubprocessError) as exc:
        return False, f"schtasks could not be run: {exc}"
    if result.returncode:
        detail = (result.stderr or result.stdout or "").strip()
        if "cannot find" in detail.lower() or "not find" in detail.lower():
            return True, "no such task"
        return False, detail[:300] or f"schtasks exited {result.returncode}"
    return True, "deleted"


def task_state(name: str) -> tuple[bool, bool, str]:
    """``(registered, running, detail)`` for ``name``, from one ``schtasks`` call.

    One call and not a query plus a probe: this runs on the wake persist path
    (through ``ensure_supervisor_installed``), so a second subprocess would be a
    cost paid by every scheduling operation. ``Status`` is the only field that
    distinguishes a live daemon from a registered-but-stopped one, and an
    unparseable status reports ``running=False`` with the raw text as detail
    rather than a confident lie.
    """
    try:
        result = schtasks(*task_query_args(name))
    except (OSError, subprocess.SubprocessError) as exc:
        return False, False, f"schtasks could not be run: {exc}"
    if result.returncode:
        detail = (result.stderr or result.stdout or "").strip()
        return False, False, detail[:200] or "not registered"
    for line in (result.stdout or "").splitlines():
        if line.strip().lower().startswith("status:"):
            status = line.partition(":")[2].strip()
            return True, status.lower() == "running", status
    return True, False, ""
