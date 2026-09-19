"""Install, supervise and introspect the mobile daemon.

One supervised unit re-runs THIS interpreter's package as ``python -m
local_operator.mobile.service`` — re-entering the installed code rather than
a hardcoded binary path means an upgrade (``lop-update``) changes what the
agent runs with no reinstall, and ``restart`` picks it up. That is the omp
mobile lesson applied to a Python entry point.

THREE SUPERVISORS, ONE SHAPE PER PLATFORM (see :mod:`local_operator.supervisors`):
a LaunchAgent plist on macOS, a ``systemd --user`` unit on Linux, and a Task
Scheduler task on Windows. The launchd half is unchanged and is the shape every
other arm is modelled on; the Linux and Windows arms exist because "the daemon
is portable, only the supervisor is macOS-specific" used to mean the relay could
not be installed AT ALL on two of the three platforms, and the CLI crashed
rather than saying so (``FileNotFoundError: launchctl``, ``AttributeError: os.getuid``).

The unit name is fixed (``com.local-operator.mobile`` / ``local-operator-mobile``
/ ``Local Operator Mobile``): it owns the port, so a second daemon cannot
split-brain the control plane — it fails to bind and exits loudly.
"""

from __future__ import annotations

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

from local_operator import launchd, procname, procstate, supervisors
from local_operator.mobile.auth import (
    generate_password,
    load_password,
    store_description,
    store_password,
)
from local_operator.mobile.daemon import DEFAULT_PORT
from local_operator.paths import CONFIG_DIR_ENV, config_dir, log_dir

LABEL = "com.local-operator.mobile"

#: Linux user unit for this daemon. Deliberately NOT suffixed per config root:
#: the label owns the port on every platform, and a per-root unit name would let
#: two units fight over 4098 instead of one failing to bind loudly.
SYSTEMD_UNIT = "local-operator-mobile.service"

#: Task Scheduler task name for this daemon (Windows), same reasoning.
TASK_NAME = "Local Operator Mobile"

#: The refusal every entry point shares when no supervisor exists at all.
NO_SUPERVISOR_ERROR = supervisors.no_supervisor_error("lop mobile serve")

#: The SPA the daemon serves. ``web/dist`` is gitignored, so a source
#: checkout has no bundle until something builds it; a pip/uv wheel ships it
#: via package-data but an in-place source install never does. Install must
#: be able to make it, or every such machine shows "bundle not built".
_WEB_DIR = Path(__file__).parent / "web"
_DIST_INDEX = _WEB_DIR / "dist" / "index.html"


def _bundle_state() -> str:
    """built / buildable / missing-sources — what install can do about dist."""
    if _DIST_INDEX.exists():
        return "built"
    if (_WEB_DIR / "package.json").exists():
        return "buildable"
    return "missing-sources"


def _windows_shim_argv(resolved: str) -> list[str]:
    """The argv prefix that runs an npm shim on Windows, given its PATH.

    Split out from :func:`_shim_argv` so the Windows SPELLING is reachable from
    a test on any host — the alternative is patching ``os.name`` process-wide,
    which ``pathlib`` reads at call time.
    """
    # ``cmd.exe`` from ``COMSPEC`` rather than the literal: that variable is how
    # the console says where its interpreter is, and a host that has moved it
    # means it. The fallback is the one path the platform guarantees.
    return [os.environ.get("COMSPEC") or "cmd.exe", "/c", "call", resolved]


def _shim_argv(name: str) -> list[str] | None:
    """The argv PREFIX that launches the npm shim ``name``, or ``None`` if absent.

    POSIX: the name itself. ``shutil.which`` has already proved it is on PATH,
    and ``execve`` runs the ``#!/bin/sh`` wrapper directly.

    WINDOWS: through the command interpreter, because a bare name CANNOT work
    there and that is invisible from the POSIX path (audit C10). npm installs
    ``pnpm`` as a ``.CMD`` batch file, and

      * ``CreateProcess`` appends only ``.exe`` when a name carries no
        extension, so ``subprocess.run(["pnpm", ...])`` never finds the shim at
        all: it raises ``FileNotFoundError``, which this module turns into a
        bundle that "failed to build" with no stated reason;
      * handing it the resolved ``.CMD`` path is not the supported route either.
        MSDN's ``CreateProcess``: "To run a batch file, you must start the
        command interpreter; set lpApplicationName to cmd.exe and set
        lpCommandLine to the following arguments: /c plus the name of the batch
        file." It does sometimes work anyway — the JDK's ``ProcessImpl`` leans
        on that undocumented behaviour — which is a reason to use the documented
        form rather than to depend on the other.

    Hence ``<comspec> /c call <resolved shim>``: ``call`` rather than a bare
    ``/c``, because cmd strips the outer quotes of a command line it cannot
    disambiguate and the default install location,
    ``C:\\Program Files\\nodejs\\pnpm.CMD``, is exactly such a line.
    """
    found = shutil.which(name)
    if found is None:
        return None
    if os.name == "nt":  # pragma: no cover - exercised on Windows hosts
        return _windows_shim_argv(found)
    return [name]


def _build_bundle() -> str | None:
    """Build the SPA in place. Returns an error string, or None on success.

    pnpm only — the lockfile and packageManager pin are pnpm's, and mixing
    npm here would write a second, unreviewed lockfile. Corepack is tried
    first so a machine with only Node (no global pnpm) still self-heals;
    the packageManager field pins the exact pnpm corepack fetches.

    Both candidates are launched through :func:`_shim_argv`, which is what makes
    this work on Windows at all: there the resolvable ``pnpm`` is a ``.CMD``
    batch file that a bare argv never reaches (audit C10).
    """
    if shutil.which("node") is None:
        # Named with its REMEDY rather than with its mechanism. This is the one
        # refusal an operator meets on a fresh Linux or Windows box, and the
        # previous text ("the bundle needs a one-time `pnpm build`") named a
        # command that cannot be run without the thing that is missing —
        # measured in the Ubuntu and Mint containers, where `mobile.install`
        # failed with exactly that and the container reading could not say what
        # to install. The wheel ships the built bundle, so this is a source
        # checkout (a container, a dev machine), and Node is a one-time cost
        # there rather than a runtime dependency of the daemon.
        # THE REMEDY HAS TO BE ONE THE READER CAN ACTUALLY RUN (design round 2,
        # D7). This sentence used to lead with `apt install nodejs`, which was
        # measured wrong on the very host this branch's own leg runs on: Ubuntu
        # 24.04's archive package is `nodejs 18.19.1`, and Debian freezes it at
        # the distro release, so the operator runs the remedy and gets this
        # identical refusal back. What the reader needs is the version check and
        # the routes that give them a current Node.
        return (
            "node is not installed, and the portal bundle is built once with it "
            "(Node >=22, and the archive package is usually older than that -- "
            "Ubuntu 24.04 ships 18 -- so check `node --version`; "
            "https://nodejs.org, or `nvm install 22`); "
            "re-run `lop mobile install` afterwards"
        )
    try:
        runner = _shim_argv("pnpm")
        if runner is None:
            corepack = _shim_argv("corepack")
            if corepack is None:
                # SAME DEFECT AS THE NODE ARM ABOVE (design round 2, D8): the
                # old sentence's only instruction was `pnpm build`, which is
                # the command that cannot run BECAUSE pnpm is the missing
                # thing. Name how to get pnpm instead.
                return (
                    "neither pnpm nor corepack is on PATH, and the portal bundle "
                    "is built with pnpm: enable Corepack (`corepack enable`; it "
                    "ships with Node) or install pnpm "
                    "(https://pnpm.io/installation), then re-run "
                    "`lop mobile install`"
                )
            subprocess.run(
                [*corepack, "enable"],
                cwd=_WEB_DIR,
                capture_output=True,
                timeout=30,
            )
            runner = [*corepack, "pnpm"]
        for args in (["install", "--frozen-lockfile"], ["build"]):
            result = subprocess.run(
                [*runner, *args], cwd=_WEB_DIR, capture_output=True, text=True, timeout=600
            )
            if result.returncode != 0:
                tail = (result.stderr or result.stdout).strip().splitlines()
                return f"pnpm {' '.join(args)} failed: {tail[-1][:200] if tail else 'unknown'}"
    except (OSError, subprocess.TimeoutExpired) as exc:
        return f"bundle build failed: {exc}"
    return None if _DIST_INDEX.exists() else "build ran but dist/index.html is still missing"


def ensure_bundle(*, build: bool = True) -> tuple[bool, str]:
    """Guarantee the daemon has a UI to serve. (ok, detail-for-status).

    The three states, in the order a fresh machine hits them: a wheel ships
    dist and this is a no-op; a source checkout is buildable and we build
    it; a broken install has neither and we say so rather than serving the
    503 the daemon would show every authed GET.
    """
    state = _bundle_state()
    if state == "built":
        return True, "bundle present"
    if state == "missing-sources":
        return False, "bundle and web sources both missing from the install"
    if not build:
        return False, "bundle missing (web sources present; build skipped)"
    error = _build_bundle()
    if error is not None:
        return False, error
    return True, "built the web bundle"


def plist_path() -> Path:
    return Path.home() / "Library" / "LaunchAgents" / f"{LABEL}.plist"


def systemd_path() -> Path:
    """The Linux user unit this daemon is registered as."""
    return supervisors.systemd_unit_path(SYSTEMD_UNIT)


def task_record_path() -> Path:
    """On Windows, our own copy of the task definition we registered.

    Task Scheduler keeps its registration in its own store (the registry), not
    in a file we own, so this is a RECORD rather than the registration itself:
    it is what ``create_task`` was handed, written where a user can read it, and
    it is what makes "did an install ever run here?" answerable without a
    subprocess. The authoritative question is still asked of ``schtasks``.
    """
    return config_dir() / "supervisor" / "mobile-task.xml"


def log_path() -> Path:
    return log_dir() / "mobile.log"


def render_plist(port: int = DEFAULT_PORT) -> dict[str, object]:
    """The whole supervised-unit plan in one pure function — every consumer
    (install, status, tests) reads the same rendering.

    ``launchd_job`` rather than ``launchd_program``: it sets ``Program`` to the
    branded interpreter image and ``ProgramArguments[0]`` to this daemon's role
    label, which is what stops the four supervised daemons reading as one
    indistinguishable ``Local Operator`` row. macOS names a background item by
    the basename of ``ProgramArguments[0]``, so the pre-branding bare
    ``sys.executable`` is what made ``lop mobile install`` notify that 'python3
    is running in the background'. The trade-off that shape accepts is recorded
    in ``procname.launchd_job``; a machine where no link can be planted gets
    byte-for-byte this plist as it was before.

    ``Program`` is the STABLE shim on a machine with the generation layout, and
    the role label does not move with it — see ``procname.supervised_image`` for
    why a path inside this venv is unsafe for a unit launchd may restart.

    ``EnvironmentVariables`` CARRIES THE STORE, and its absence was a bug on all
    three platforms rather than a Windows one. ``install`` stores the portal
    password under ``config_dir()`` and the daemon reads it back with the same
    function, but a supervised process does NOT inherit the installer's
    environment: with ``LOCAL_OPERATOR_CONFIG_DIR`` set, the installer wrote a
    password into that store and the daemon looked in the default one, then
    refused to start with "no mobile password set. Run `lop mobile install`" —
    naming the store it had just written to. The Windows probe battery found it
    (`mobile.install`, PASS=20/FAIL=1) and it is the same latent bug on macOS and
    Linux. ``wakes`` and ``tunnels`` have always passed the store this way; this
    daemon is the one that did not.
    """
    return {
        "Label": LABEL,
        **procname.launchd_job(
            "local_operator.mobile.service",
            "--port",
            str(port),
            label=procname.branded_argv0(procname.LABEL_MOBILE, port=port),
        ),
        "RunAtLoad": True,
        # Restart on crash, throttled by launchd's own 10s floor; an
        # exit-code-2 (no password) stays down because KeepAlive keys on
        # successful exit only being false — crashes restart, refusals don't
        # flap.
        "KeepAlive": {"SuccessfulExit": False},
        "StandardOutPath": str(log_path()),
        "StandardErrorPath": str(log_path()),
        # SSE-holding daemons must not be App Nap'd into suspending timers.
        "ProcessType": "Interactive",
        "EnvironmentVariables": {CONFIG_DIR_ENV: str(config_dir())},
    }


def refresh_plist_if_stale() -> launchd.PlistRefresh:
    """Bring this daemon's LaunchAgent up to date, and restart it if it changed.

    THE REPAIR ``lop-update`` NEVER HAD FOR THIS DAEMON. ``install`` renders the
    current plist and nothing ever rewrote one written by an older build, so a
    daemon installed before branding shows a bare ``python3.14`` in Activity
    Monitor for the rest of its life — the colleague's symptom, measured on the
    operator's machine as ``com.local-operator.mobile.plist`` (mtime Sep 5)
    running ``python3.14 -m local_operator.mobile.service --port 4098``.

    Deliberately NARROW: it rewrites the plist and restarts the job, and does
    not touch the password, the Keychain or the web bundle, because the wheel
    already ships the bundle and the upgrade path must not pay for a rebuild —
    that is why this is not routed through ``install``.

    Never raises, and never acts outside the real home:
    ``launchd.is_own_plist`` refuses a redirected ``HOME`` because
    ``launchctl`` addresses the REAL user's session whatever the plist path
    says, so a sandboxed run would otherwise restart the operator's daemon.
    """
    name = "mobile"
    try:
        if supervisors.supervisor() != supervisors.LAUNCHCTL:
            # NOT "not-addressable": there is no plist on this platform to
            # repair at all. `is_supported()` was the wrong gate once it began
            # answering True on Linux and Windows, where this function would
            # otherwise have gone looking for a LaunchAgent that cannot exist.
            return launchd.PlistRefresh(name=name, kind="unsupported")
        path = plist_path()
        if not launchd.is_own_plist(path, LABEL):
            return launchd.PlistRefresh(name=name, kind="not-addressable")
        # The port comes off the plist being REPLACED: a repair must not move a
        # daemon someone installed on a non-default port back to the default.
        port = launchd.int_arg(launchd.load(path), "--port", DEFAULT_PORT)
        outcome = launchd.rewrite_if_stale(name=name, path=path, rendered=render_plist(port))
        if outcome.kind != "repaired":
            return outcome
        # bootout + bootstrap through the shared helper, NOT kickstart -k:
        # measured on macOS with a scratch label, a kickstart after a rewrite
        # restarts the job from launchd's in-memory definition and keeps running
        # the OLD argv. See :mod:`local_operator.launchd`.
        reloaded = launchd.reload_job(label=LABEL, path=path, runner=_launchctl)
        if not reloaded.ok:
            # Names the recovery, because the job is DOWN at this point: see
            # `launchd.reload_failure`.
            return reloaded.as_refresh_failure(name=name, path=path, recovery="lop mobile install")
        return outcome
    except Exception as exc:  # noqa: BLE001 — a repair must never fail an upgrade
        return launchd.PlistRefresh(name=name, kind="failed", detail=str(exc))


def _launchctl(*args: str) -> subprocess.CompletedProcess[str]:
    return subprocess.run(["launchctl", *args], capture_output=True, text=True, timeout=15)


def _supervised_pid() -> int | None:
    """The pid this daemon's supervisor currently runs, or ``None``.

    launchd prints ``pid = <n>`` only while a process is alive behind the label;
    systemd answers ``MainPID`` (``0`` when the unit is not running). Task
    Scheduler publishes no pid through ``schtasks`` at all, which is why the
    Windows answer below is "the task reports Running" rather than a pid.
    """
    kind = supervisors.supervisor()
    if kind == supervisors.LAUNCHCTL:
        printed = _launchctl("print", f"{_domain()}/{LABEL}")
        if printed.returncode != 0:
            return None
        match = re.search(r"pid = (\d+)", printed.stdout)
        return int(match.group(1)) if match else None
    if kind == supervisors.SYSTEMCTL:
        shown = supervisors.systemctl_user("show", "-p", "MainPID", "--value", SYSTEMD_UNIT)
        value = shown.stdout.strip() if shown.returncode == 0 else ""
        return int(value) if value.isdigit() and value != "0" else None
    return None


def _listening_pids(port: int) -> set[str] | None:
    """The pids listening on ``port``, or ``None`` when this box cannot be asked.

    ``lsof`` is the mechanism this repo already uses for exactly this question
    (``session/runtime/control.py``); a machine without it — a slim container —
    answers ``None``, and the caller says what it could not check instead of
    pretending it did.
    """
    binary = shutil.which("lsof")
    if binary is None:
        return None
    try:
        listeners = subprocess.run(  # noqa: S603 — fixed argv, no shell
            [binary, "-nP", f"-iTCP:{port}", "-sTCP:LISTEN", "-t"],
            capture_output=True,
            text=True,
            timeout=5,
        )
    except (OSError, subprocess.SubprocessError):
        return None
    return set(listeners.stdout.split())


def _our_daemon_listening(port: int) -> bool:
    """True when the process bound to ``port`` is the one this unit starts.

    Health alone cannot answer this: a stale foreground daemon on the same
    port passes every check while the supervised one fails to bind. Ask the
    supervisor which pid it is running, then ask lsof who owns the port.

    macOS behaviour is unchanged — this is the same ``launchctl print`` + lsof
    pair it always was, and the same False when either says no. The two
    additions are the systemd arm (same question, ``MainPID``) and the
    no-lsof case, which used to escape as an uncaught ``FileNotFoundError`` out
    of ``install()`` and now falls back to the supervisor's own word, with the
    gap named in the install steps.
    """
    kind = supervisors.supervisor()
    if kind == supervisors.SCHTASKS:
        # No pid to cross-check: Task Scheduler's own Running status is the
        # strongest signal schtasks gives, and health + the auth gate below
        # still have to pass before an install reports success.
        #
        # ``running is not False`` rather than a truth test, because the status
        # word is localized: an unrecognised spelling comes back as ``None``
        # rather than as a definite "not running", and treating that as a
        # refusal made this loop unable to ever pass on such a host — a daemon
        # that was up and serving was reported as "daemon did not come up
        # healthy". Where the state cannot be read, the two signals that CAN be
        # — the unauth health endpoint and the auth gate — are what decide.
        registered, running, _detail = supervisors.task_state(TASK_NAME)
        return registered and running is not False
    pid = _supervised_pid()
    if pid is None:
        return False
    owners = _listening_pids(port)
    if owners is None:
        return True
    return str(pid) in owners


def _domain() -> str:
    """``gui/<uid>`` — the launchd domain every caller feeds to ``launchctl``.

    The platform guard is INSIDE this function rather than at its call sites,
    each of which is behind ``if kind == supervisors.LAUNCHCTL``: ``os.getuid``
    does not exist off POSIX, so an unguarded call was an ``AttributeError``
    waiting for any arm that forgot the check, and a guard at the call site is
    invisible to a reader — and to the static scan that grades this branch —
    which cannot see that the arm is unreachable. Guarded here, the function is
    safe to call anywhere on its own merits.
    """
    if procstate.is_windows():
        raise RuntimeError("launchd domains exist only on macOS")
    return f"gui/{os.getuid()}"


def is_supported() -> bool:
    """Whether this machine has a user-level supervisor this installer can drive.

    Binary-guarded rather than platform-guarded (see
    :mod:`local_operator.supervisors`): on a Linux without systemd — Devuan,
    Alpine, most containers, WSL2 without systemd — the answer is False, which
    the entry points turn into a sentence naming the foreground command. Before
    this, that machine got ``FileNotFoundError: 'launchctl'`` out of the CLI and
    a Windows one got ``AttributeError: module 'os' has no attribute 'getuid'``,
    because only ``install()`` was guarded and ``uninstall``/``service_action``
    were not.
    """
    return supervisors.supervisor() is not None


def registration_present() -> bool:
    """Whether this platform's supervisor has a registration for this daemon.

    The unit FILE for launchd/systemd; a ``schtasks /Query`` for Windows, whose
    registration lives in Task Scheduler's own store. The subprocess is why this
    is a status-time call and not something any hot path may use.
    """
    kind = supervisors.supervisor()
    if kind == supervisors.LAUNCHCTL:
        return plist_path().exists()
    if kind == supervisors.SYSTEMCTL:
        return systemd_path().exists()
    if kind == supervisors.SCHTASKS:
        registered, _running, _detail = supervisors.task_state(TASK_NAME)
        return registered
    return False


def render_systemd(port: int = DEFAULT_PORT) -> str:
    """The Linux user unit, in the shape all four daemons' units share.

    ``procname.supervised_image`` when this machine has the generation layout,
    else this interpreter: a unit is re-executed on every restart, and a path
    inside a tree a flip or a prune replaced is a daemon that dies at load
    (see that function for the 2026-09-15 incident).
    """
    image = procname.supervised_image() or Path(sys.executable)
    return supervisors.render_systemd_unit(
        description="Local Operator mobile daemon",
        # The image is quoted for systemd's own grammar, not shell quoting: this
        # is the same rule ``tunnels.install`` uses, hoisted into
        # ``supervisors.quoted``. Measured on systemd 255: an interpreter under
        # a path with a space produced a unit that could never start ("Command
        # /home/a is not executable").
        exec_start=(
            f"{supervisors.quoted(str(image))} -m local_operator.mobile.service --port {port}"
        ),
        # ``Restart=on-failure``: an exit-code-2 (no password) stays down so a
        # refused start does not flap, exactly what the plist's
        # KeepAlive{SuccessfulExit: false} buys on macOS.
        #
        # ``Environment=`` carries the STORE, the ``EnvironmentVariables``
        # analogue in the plist above and the same two reasons its siblings
        # document: the store is part of the contract, and an UNQUOTED
        # assignment truncates it at the first space — silently, so the daemon
        # would read a different store than the installer wrote to and refuse to
        # start.
        post_lines=[
            f"Environment={supervisors.quoted(f'{CONFIG_DIR_ENV}={config_dir()}')}",
            *supervisors.output_redirect_lines(log_path()),
        ],
    )


def render_task_xml(port: int = DEFAULT_PORT) -> str:
    """The Windows Task Scheduler task for this daemon.

    ``environment=`` CARRIES THE STORE. It used to say that this daemon's plist
    recorded no config dir either and the store was the default one — which was
    true of the plist and wrong about the consequence: no supervised process
    inherits the installer's environment, so an install from a store override
    registered a task whose daemon read the DEFAULT store, found no password
    there, and exited 2. The Windows battery reported exactly that
    ("no mobile password set. Run `lop mobile install`" naming the store the
    installer had just written to), and it is the same omission the plist and the
    unit had. Task Scheduler has no native environment element, so
    ``supervisors.render_task_xml`` expresses it through the launcher; see that
    function for the shape.
    """
    image = procname.supervised_image() or Path(sys.executable)
    return supervisors.render_task_xml(
        description="Local Operator mobile daemon (serve the phone portal)",
        image=str(image),
        argv=["-m", "local_operator.mobile.service", "--port", str(port)],
        environment={CONFIG_DIR_ENV: str(config_dir())},
        log=log_path(),
        user_id=supervisors.current_user_id(),
    )


def health(port: int = DEFAULT_PORT, timeout: float = 3.0) -> dict[str, object] | None:
    """Probe the daemon's unauthenticated liveness endpoint."""
    try:
        with urllib.request.urlopen(
            f"http://127.0.0.1:{port}/healthz", timeout=timeout
        ) as response:
            return json.loads(response.read().decode())  # type: ignore[no-any-return]
    except Exception:  # noqa: BLE001 — health is a probe, absence is the answer
        return None


def gate_closed(port: int = DEFAULT_PORT, timeout: float = 3.0) -> bool:
    """Assert the AUTH gate, not mere liveness: a daemon that serves the API
    without a cookie is a boundary failure, however healthy its process."""
    request = urllib.request.Request(f"http://127.0.0.1:{port}/api/sessions")
    try:
        with urllib.request.urlopen(request, timeout=timeout) as response:
            return response.status == 401
    except urllib.error.HTTPError as exc:
        return exc.code == 401
    except Exception:  # noqa: BLE001
        return False


def install(port: int = DEFAULT_PORT, *, dry_run: bool = False) -> dict[str, object]:
    """Idempotent one-shot: bundle, password (kept if present), unit, load, verify.

    One shared prefix — guarantee the UI bundle, generate/keep the password —
    then one arm per supervisor. The arms differ only in HOW the daemon is made
    to run at login; the verification at the end is shared, because "a unit file
    exists" was never the question.

    The BUNDLE comes first, before anything is written. It is the only step that
    can fail for a reason the operator has to go and fix (no Node on a source
    checkout), and running it after the password left a store mutated by an
    install that then reported failure — visible in the Ubuntu/Mint container
    readings, whose FAIL detail was the password progress line rather than the
    reason. A preflight that refuses changes nothing on disk.
    """
    steps: list[str] = []
    kind = supervisors.supervisor()
    if kind is None:
        return {"ok": False, "steps": [], "error": NO_SUPERVISOR_ERROR}

    # The UI is half the product. A missing bundle means every authed GET
    # 503s, so install builds it rather than leaving the phone on a dead
    # page — the wheel normally ships it, a source checkout does not.
    bundle_ok, bundle_detail = ensure_bundle(build=not dry_run)
    steps.append(bundle_detail)
    if not bundle_ok:
        return {"ok": False, "steps": steps, "error": f"web bundle unavailable: {bundle_detail}"}

    password = load_password()
    if password is None:
        password = generate_password()
        if not dry_run:
            store_password(password)
        steps.append(f"generated a new portal password ({store_description()})")
    else:
        steps.append(f"kept the existing portal password ({store_description()})")

    if kind == supervisors.LAUNCHCTL:
        plist_path().parent.mkdir(parents=True, exist_ok=True)
        if not dry_run:
            plist_path().write_bytes(plistlib.dumps(render_plist(port)))
        steps.append(f"wrote {plist_path()}")
        if not dry_run:
            # The reload, not a bare pair: it tolerates an absent job, waits for
            # launchd to release the label, retries the bootstrap past the
            # measured teardown race, and verifies the job is registered
            # afterwards — so the steps below are reporting a daemon that really
            # is loaded. See :mod:`local_operator.launchd`.
            reloaded = launchd.reload_job(label=LABEL, path=plist_path(), runner=_launchctl)
            if not reloaded.ok:
                return {"ok": False, "steps": steps, "error": reloaded.detail[:300]}
            steps.append("loaded the LaunchAgent")
    elif kind == supervisors.SYSTEMCTL:
        loaded, detail = _install_systemd(port, dry_run=dry_run, steps=steps)
        if not loaded:
            return {"ok": False, "steps": steps, "error": detail}
    else:  # schtasks
        registered, detail = _install_task(port, dry_run=dry_run, steps=steps)
        if not registered:
            return {"ok": False, "steps": steps, "error": detail}

    if dry_run:
        steps.append("dry run: skipped load and verification")
        return {"ok": True, "steps": steps}

    # Shared verification: the supervisor must own the port (a stale foreground
    # daemon passes a bare health check while the supervised one fails to bind)
    # AND the auth gate must be closed — never installed-but-unauthenticated.
    deadline = time.time() + 20
    while time.time() < deadline:
        if _our_daemon_listening(port) and health(port) and gate_closed(port):
            steps.append("health check passed and the auth gate is closed")
            return {"ok": True, "steps": steps}
        time.sleep(0.5)
    return {
        "ok": False,
        "steps": steps,
        "error": f"daemon did not come up healthy; see {log_path()}",
    }


def _install_systemd(port: int, *, dry_run: bool, steps: list[str]) -> tuple[bool, str]:
    """Write and enable the Linux user unit. ``(ok, error)``.

    The unit file is written wherever ``systemd_path()`` says — that half is
    fully testable under a redirected home — while the user manager is only
    ADDRESSED when that is the path the real home owns. ``systemctl --user``
    has no sandbox: it reaches the calling user's live instance whatever
    ``$HOME`` says, so without the guard an isolated run would enable a real
    unit pointed at a store that vanishes when the sandbox ends. Same guard, and
    same incident, as the plist half in :mod:`local_operator.launchd`.
    """
    unit = systemd_path()
    unit.parent.mkdir(parents=True, exist_ok=True)
    if not dry_run:
        unit.write_text(render_systemd(port), encoding="utf-8")
    steps.append(f"wrote {unit}")
    if dry_run:
        return True, ""
    if not supervisors.systemd_unit_is_addressable(SYSTEMD_UNIT):
        return False, (
            f"{unit} is not the unit the real home owns; refusing to enable it "
            "from a redirected home"
        )
    # Lingering BEFORE enable --now: without a user manager the enable itself
    # fails with the bus error, and enabling linger is what spawns one.
    # Best-effort, so a refusal does not fail the install.
    if supervisors.enable_linger():
        steps.append("enabled lingering so the daemon survives logout and reboot")
    else:
        steps.append(
            "could not enable lingering; the daemon may not survive logout "
            f"(run: loginctl enable-linger {os.environ.get('USER', '$USER')})"
        )
    supervisors.systemctl_user("daemon-reload")
    loaded = supervisors.systemctl_user("enable", "--now", SYSTEMD_UNIT)
    if loaded.returncode:
        return False, supervisors.translate_systemctl_error(loaded.stderr)
    steps.append(f"enabled the systemd user service ({SYSTEMD_UNIT})")
    return True, ""


def _install_task(port: int, *, dry_run: bool, steps: list[str]) -> tuple[bool, str]:
    """Register and start the Windows scheduled task. ``(ok, error)``.

    The XML is written under the config root first and kept: Task Scheduler
    stores its copy in the registry, so this is the only readable record of what
    was registered, and it is what makes a later diff possible.
    """
    xml = render_task_xml(port)
    record = task_record_path()
    if not dry_run:
        record.parent.mkdir(parents=True, exist_ok=True)
        # Task Scheduler has no stdout redirection, so the log the other two
        # platforms' supervisors create comes from the task's own command line
        # (see supervisors.render_task_xml): this is what keeps log_path() the
        # one place every log surface points at on every platform.
        log_path().parent.mkdir(parents=True, exist_ok=True)
        record.write_text(xml, encoding="utf-8")
    steps.append(f"wrote the task definition to {record}")
    if dry_run:
        return True, ""
    ok, detail = supervisors.create_task(TASK_NAME, xml)
    if not ok:
        return False, f"schtasks could not register the task: {detail}"
    steps.append(f"registered the scheduled task ({TASK_NAME})")
    started = supervisors.schtasks(*supervisors.task_run_args(TASK_NAME))
    if started.returncode:
        return False, (started.stderr or started.stdout or "").strip()[:300] or (
            "schtasks could not start the task"
        )
    steps.append("started the task")
    return True, ""


def uninstall(*, purge: bool = False, dry_run: bool = False) -> dict[str, object]:
    """Stop and deregister this daemon on whichever platform it was installed.

    EVERY verb is guarded here, not only ``install``: ``uninstall`` used to
    call ``launchctl`` (and ``_domain()``, which needs ``os.getuid``) with no
    platform check, so on Linux it raised ``FileNotFoundError: 'launchctl'``
    and on Windows ``AttributeError: module 'os' has no attribute 'getuid'`` —
    a stack trace where the user asked to remove something. An unsupported
    platform now removes what this installer could have written and says so.
    """
    steps: list[str] = []
    kind = supervisors.supervisor()
    if not dry_run:
        if kind == supervisors.LAUNCHCTL:
            _launchctl("bootout", _domain(), str(plist_path()))
            plist_path().unlink(missing_ok=True)
            steps.append("removed the LaunchAgent")
        elif kind == supervisors.SYSTEMCTL:
            if supervisors.systemd_unit_is_addressable(SYSTEMD_UNIT):
                supervisors.systemctl_user("disable", "--now", SYSTEMD_UNIT)
                steps.append("disabled the systemd user service")
            else:
                steps.append("left the unit loaded: not addressable from a redirected home")
            systemd_path().unlink(missing_ok=True)
            steps.append(f"removed {systemd_path()}")
        elif kind == supervisors.SCHTASKS:
            # ``/Delete`` deregisters the task WITHOUT interrupting a running
            # program — Microsoft's own wording for the verb is "This command
            # doesn't delete the program that the task runs or interrupt a
            # running program" — so an uninstall that only deletes leaves the
            # portal daemon serving on its port with the password still loaded.
            # ``/End`` first is the Windows spelling of the ``bootout`` the
            # launchd arm issues and of Linux's ``disable --now``; it is the
            # shape ``service_action`` already uses for ``stop``.
            supervisors.schtasks(*supervisors.task_end_args(TASK_NAME))
            deleted, detail = supervisors.delete_task(TASK_NAME)
            if not deleted:
                # REFUSED rather than quietly successful: the daemon is still
                # registered, so reporting ``ok: True`` told a caller the
                # opposite of the state it left behind.
                steps.append(detail)
                return {"ok": False, "steps": steps, "error": detail}
            steps.append(f"removed the scheduled task ({detail})")
            task_record_path().unlink(missing_ok=True)
        else:
            # REFUSED rather than quietly successful: nothing here could have
            # registered this daemon, so there is nothing to remove, and the
            # step line carries the same sentence `install` gives. (The CLI's
            # uninstall prints `steps` and not `error`, which is why the refusal
            # is in both.)
            steps.append(NO_SUPERVISOR_ERROR)
            return {"ok": False, "steps": steps, "error": NO_SUPERVISOR_ERROR}
    if purge:
        if not dry_run:
            from local_operator.mobile.auth import delete_password

            delete_password()
        steps.append(f"deleted the portal password ({store_description()})")
    return {"ok": True, "steps": steps}


def service_action(action: str) -> dict[str, object]:
    """start|stop|restart, in each platform's own vocabulary.

    The launchd arm's kickstart family fails with "Could not find service" when
    the plist exists but the agent was never bootstrapped (or was booted out and
    not re-loaded). Bootstrap it on demand so the control commands work from
    whatever state launchd is in, not just the state `install` left behind.
    systemd has the same class of gap and closes it differently —
    ``systemctl --user start`` on an unloaded unit fails with "Unit not found",
    which is true and actionable, so no repair is invented here.
    """
    kind = supervisors.supervisor()
    if kind is None:
        return {"ok": False, "error": NO_SUPERVISOR_ERROR}
    if kind == supervisors.SYSTEMCTL:
        if not supervisors.systemd_unit_is_addressable(SYSTEMD_UNIT):
            return {
                "ok": False,
                "error": (
                    f"{systemd_path()} is not the unit the real home owns; not "
                    "addressing the user manager from a redirected home"
                ),
            }
        result = supervisors.systemctl_user(action, SYSTEMD_UNIT)
        ok = result.returncode == 0
        return {
            "ok": ok,
            "error": "" if ok else supervisors.translate_systemctl_error(result.stderr)[:300],
        }
    if kind == supervisors.SCHTASKS:
        if action in ("stop", "restart"):
            # `/End` on a task that is not running exits non-zero, which for a
            # stop is the state the caller asked for; only a task that is not
            # REGISTERED is a real failure, and its stderr says so.
            ended = supervisors.schtasks(*supervisors.task_end_args(TASK_NAME))
            if ended.returncode:
                detail = (ended.stderr or ended.stdout or "").strip()
                if "running" not in detail.lower():
                    return {"ok": False, "error": detail[:300] or "schtasks could not stop it"}
        if action in ("start", "restart"):
            started = supervisors.schtasks(*supervisors.task_run_args(TASK_NAME))
            ok = started.returncode == 0
            return {
                "ok": ok,
                "error": "" if ok else ((started.stderr or started.stdout or "").strip()[:300]),
            }
        return {"ok": True, "error": ""}
    if action in ("start", "restart") and plist_path().exists():
        printed = _launchctl("print", f"{_domain()}/{LABEL}")
        if printed.returncode != 0:
            bootstrap = _launchctl("bootstrap", _domain(), str(plist_path()))
            if bootstrap.returncode != 0:
                return {"ok": False, "error": bootstrap.stderr.strip()[:300]}
    if action == "start":
        result = _launchctl("kickstart", f"{_domain()}/{LABEL}")
    elif action == "stop":
        result = _launchctl("kill", "SIGTERM", f"{_domain()}/{LABEL}")
    else:  # restart
        result = _launchctl("kickstart", "-k", f"{_domain()}/{LABEL}")
    ok = result.returncode == 0
    return {"ok": ok, "error": "" if ok else result.stderr.strip()[:300]}


def status(port: int = DEFAULT_PORT) -> dict[str, object]:
    """What a human (or `lop mobile status`) needs: install state, live
    health, gate state, registered sessions, log path."""
    from local_operator.session.runtime import registry

    probe = health(port)
    records = registry.scan()
    kind = supervisors.supervisor()
    return {
        "installed": registration_present(),
        # Which supervisor that answer came FROM, and where its registration
        # lives on this platform: "installed: yes" with no way to see whether it
        # is launchd, systemd or Task Scheduler is not a diagnosable state.
        "supervisor": kind or "none",
        "registration": str(_registration_path()) if _registration_path() else "",
        "password_set": load_password() is not None,
        "password_store": store_description(),
        "bundle": _bundle_state(),
        "healthy": probe is not None,
        "gate_closed": gate_closed(port),
        "health": probe,
        "port": port,
        "log": str(log_path()),
        "sessions": [
            {
                "pid": record.pid,
                "kind": record.kind,
                "session_id": record.session_id,
                "conversation_name": record.conversation_name,
                "model_label": record.model_label,
                "state": state,
            }
            for record, state in records
        ],
    }


def _registration_path() -> Path | None:
    """Where this platform's registration lives, for status output.

    ``None`` on a platform with no supervisor; on Windows it is the record this
    installer wrote, because Task Scheduler keeps its own copy in the registry.
    """
    kind = supervisors.supervisor()
    if kind == supervisors.LAUNCHCTL:
        return plist_path()
    if kind == supervisors.SYSTEMCTL:
        return systemd_path()
    if kind == supervisors.SCHTASKS:
        return task_record_path()
    return None
