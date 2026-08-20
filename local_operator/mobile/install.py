"""Install, supervise and introspect the mobile daemon (macOS launchd).

One LaunchAgent re-runs THIS interpreter's package as ``python -m
local_operator.mobile.service`` — re-entering the installed code rather than
a hardcoded binary path means an upgrade (``lop-update``) changes what the
agent runs with no reinstall, and ``restart`` picks it up. That is the omp
mobile lesson applied to a Python entry point.

The LaunchAgent label is fixed (``com.local-operator.mobile``): the label owns
the port, so a second daemon cannot split-brain the control plane — it fails
to bind and exits loudly.

Linux/Windows: the daemon itself is portable and foreground-runnable; only
the supervisor is macOS-specific, and the CLI says so rather than writing a
broken unit file.
"""

from __future__ import annotations

import json
import plistlib
import shutil
import subprocess
import sys
import time
import urllib.error
import urllib.request
from pathlib import Path

from local_operator.mobile.auth import generate_password, load_password, store_password
from local_operator.mobile.daemon import DEFAULT_PORT
from local_operator.paths import log_dir

LABEL = "com.local-operator.mobile"

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


def _build_bundle() -> str | None:
    """Build the SPA in place. Returns an error string, or None on success.

    pnpm only — the lockfile and packageManager pin are pnpm's, and mixing
    npm here would write a second, unreviewed lockfile. Corepack is tried
    first so a machine with only Node (no global pnpm) still self-heals;
    the packageManager field pins the exact pnpm corepack fetches.
    """
    if shutil.which("node") is None:
        return "node is not installed; the bundle needs a one-time `pnpm build`"
    try:
        if shutil.which("pnpm") is not None:
            runner = ["pnpm"]
        elif shutil.which("corepack") is not None:
            subprocess.run(
                ["corepack", "enable"],
                cwd=_WEB_DIR,
                capture_output=True,
                timeout=30,
            )
            runner = ["corepack", "pnpm"]
        else:
            return "neither pnpm nor corepack found; run `pnpm build` in local_operator/mobile/web"
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


def log_path() -> Path:
    return log_dir() / "mobile.log"


def render_plist(port: int = DEFAULT_PORT) -> dict[str, object]:
    """The whole supervised-unit plan in one pure function — every consumer
    (install, status, tests) reads the same rendering."""
    return {
        "Label": LABEL,
        "ProgramArguments": [
            sys.executable,
            "-m",
            "local_operator.mobile.service",
            "--port",
            str(port),
        ],
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
    }


def _launchctl(*args: str) -> subprocess.CompletedProcess[str]:
    return subprocess.run(["launchctl", *args], capture_output=True, text=True, timeout=15)


def _our_daemon_listening(port: int) -> bool:
    """True when the process bound to ``port`` is the one this plist starts.

    Health alone cannot answer this: a stale foreground daemon on the same
    port passes every check while the supervised one fails to bind. Ask
    launchd which pid it is running, then ask lsof who owns the port.
    """
    import re

    printed = _launchctl("print", f"{_domain()}/{LABEL}")
    if printed.returncode != 0:
        return False
    match = re.search(r"pid = (\d+)", printed.stdout)
    if not match:
        return False
    pid = match.group(1)
    listeners = subprocess.run(
        ["lsof", "-nP", f"-iTCP:{port}", "-sTCP:LISTEN", "-t"],
        capture_output=True,
        text=True,
        timeout=5,
    )
    return pid in listeners.stdout.split()


def _domain() -> str:
    import os

    return f"gui/{os.getuid()}"


def is_supported() -> bool:
    return sys.platform == "darwin" and shutil.which("launchctl") is not None


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
    """Idempotent one-shot: password (kept if present), plist, load, verify."""
    steps: list[str] = []
    if not is_supported():
        return {
            "ok": False,
            "steps": [],
            "error": (
                "install needs macOS launchd; " "run `lop mobile serve` in the foreground elsewhere"
            ),
        }

    password = load_password()
    if password is None:
        password = generate_password()
        if not dry_run:
            store_password(password)
        steps.append("generated a new portal password (Keychain: lop-mobile)")
    else:
        steps.append("kept the existing portal password")

    # The UI is half the product. A missing bundle means every authed GET
    # 503s, so install builds it rather than leaving the phone on a dead
    # page — the wheel normally ships it, a source checkout does not.
    bundle_ok, bundle_detail = ensure_bundle(build=not dry_run)
    steps.append(bundle_detail)
    if not bundle_ok:
        return {"ok": False, "steps": steps, "error": f"web bundle unavailable: {bundle_detail}"}

    plist_path().parent.mkdir(parents=True, exist_ok=True)
    if not dry_run:
        plist_path().write_bytes(plistlib.dumps(render_plist(port)))
    steps.append(f"wrote {plist_path()}")

    if not dry_run:
        _launchctl("bootout", _domain(), str(plist_path()))  # stale copy: fine if absent
        result = _launchctl("bootstrap", _domain(), str(plist_path()))
        if result.returncode != 0:
            return {"ok": False, "steps": steps, "error": result.stderr.strip()[:300]}
        steps.append("loaded the LaunchAgent")

        # The daemon we just bootstrapped owns the port now; a health check
        # that passed on a leftover foreground process would lie about the
        # supervised one. Wait for OUR pid to be the listener before probing.
        deadline = time.time() + 20
        while time.time() < deadline:
            if _our_daemon_listening(port) and health(port) and gate_closed(port):
                steps.append("health check passed and the auth gate is closed")
                # Never return the password: a `--json` dump or an agent
                # capturing stdout would put it in the transcript.
                return {"ok": True, "steps": steps}
            time.sleep(0.5)
        return {
            "ok": False,
            "steps": steps,
            "error": f"daemon did not come up healthy; see {log_path()}",
        }
    steps.append("dry run: skipped load and verification")
    return {"ok": True, "steps": steps}


def uninstall(*, purge: bool = False, dry_run: bool = False) -> dict[str, object]:
    steps: list[str] = []
    if not dry_run:
        _launchctl("bootout", _domain(), str(plist_path()))
        plist_path().unlink(missing_ok=True)
    steps.append("removed the LaunchAgent")
    if purge:
        if not dry_run:
            from local_operator.mobile.auth import delete_password

            delete_password()
        steps.append("deleted the Keychain password")
    return {"ok": True, "steps": steps}


def service_action(action: str) -> dict[str, object]:
    """start|stop|restart via launchctl.

    The kickstart family fails with "Could not find service" when the plist
    exists but the agent was never bootstrapped (or was booted out and not
    re-loaded). Bootstrap it on demand so the control commands work from
    whatever state launchd is in, not just the state `install` left behind.
    """
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
    from local_operator.mobile import registry

    probe = health(port)
    records = registry.scan()
    return {
        "installed": plist_path().exists(),
        "password_set": load_password() is not None,
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
