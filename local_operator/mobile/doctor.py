"""`lop mobile doctor`: one read-only, redacted diagnosis of phone access.

WHY THIS EXISTS. Diagnosing a teammate's "Asleep / Not answering" phone took
shell commands (`curl` the health route, `lsof` the daemon, `grep` its logs)
that a non-technical user cannot run, and that an agent cannot even SEND over
Team Bridge: Radient's WAF refuses message bodies carrying `curl`/`lsof`/`grep`.
This module answers the same questions in one step, and the same JSON is served
to the desktop's "Copy diagnostics" button (`GET /v1/desktop/mobile/doctor`).

CONSTRAINTS, each load-bearing:

* READ-ONLY. Nothing here starts, stops, repairs or writes anything — not the
  daemon, not the tunnel, not the credential store (opened ``read_only``). The
  one network read is the account's tunnel LIST (a GET), for the duplicate check.
* SHAREABLE. The output is meant to be pasted into a support thread, so it
  carries no token, cookie, password or email, and the home directory is
  written ``~``. Values are chosen from fixed fields, never copied from logs or
  provider bodies, and `_scrub` is a final pass over the whole result; a test
  seeds secrets into config, logs, the store and the environment and asserts
  none of them comes out.
* CALM ABOUT ABSENCE. A daemon that was never installed, or is not running, is
  ``not running`` — never the "OPEN (boundary failure)" an unreachable health
  probe used to be printed as (probe 6d779936753a, fix #2) — and the command's
  exit status is 0 whenever it could produce a report: the doctor reports, it
  does not alarm.

Each check is ``{"id", "status": pass|warn|fail, "summary", "data"}``: the
summary is the plain-language sentence (keyed copy, `mobile.doctor`), the data
the machine-readable facts behind it.
"""

from __future__ import annotations

import asyncio
import datetime
import importlib.metadata
import os
import platform
import re
import shutil
import subprocess
import sys
import time
from pathlib import Path
from typing import Any

from local_operator.i18n.keys import mobile_doctor as _md
from local_operator.i18n.text import tr

_t = tr("mobile.doctor")

SCHEMA = 1
PASS, WARN, FAIL = "pass", "warn", "fail"

#: A healthy daemon answers `/healthz` in single-digit milliseconds on loopback;
#: above this the phone's own request budget is at risk, so it is a warning, not
#: a failure. The probe itself keeps `install.health`'s existing 3 s timeout.
SLOW_HEALTH_MS = 1000
#: Warn at this share of the soft limit: the remaining headroom is what a burst
#: of phone reconnects (one SSE stream each) plus sqlite opens draws on.
FILES_WARN_RATIO = 0.8
#: The log window: "the last day", bounded in BYTES too so a huge un-rotated
#: launchd log stays a cheap read (the tail is what holds the last day anyway).
LOG_WINDOW_S = 24 * 3600
LOG_TAIL_BYTES = 8 * 1024 * 1024
_EMFILE = re.compile(r"Too many open files|EMFILE|\[Errno 24\]")
_SQLITE_OPEN = re.compile(r"unable to open database file")
_STAMP = re.compile(r"^(\d{4}-\d{2}-\d{2})[ T](\d{2}:\d{2}:\d{2})")
_EMAIL = re.compile(r"[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}")
#: The optional desktop-app version a caller may pass; anything else is dropped.
_VERSION = re.compile(r"^[0-9A-Za-z.+-]{1,40}$")


def _check(check_id: str, status: str, summary: str, **data: Any) -> dict[str, Any]:
    return {"id": check_id, "status": status, "summary": summary, "data": data}


# --- versions ------------------------------------------------------------------


def _cloudflared() -> str:
    """cloudflared's version, or why it does not apply.

    With the built-in relay (`tunnels.config.relay_offered`) cloudflared is not
    needed at all, which is worth saying rather than reporting it "missing".
    """
    try:
        from local_operator.tunnels import config as tunnel_config

        if tunnel_config.relay_offered():
            return "n/a (built-in relay)"
    except (OSError, ValueError, ImportError):
        pass
    binary = shutil.which("cloudflared")
    if binary is None:
        return "not installed"
    try:
        out = subprocess.run(  # noqa: S603 — fixed argv, no shell
            [binary, "--version"], capture_output=True, text=True, timeout=5
        ).stdout
    except (OSError, subprocess.SubprocessError):
        return "unknown"
    match = re.search(r"version (\S+)", out)
    return match.group(1) if match else "unknown"


def check_versions(desktop_app: str | None = None) -> dict[str, Any]:
    try:
        lop = importlib.metadata.version("local-operator")
    except importlib.metadata.PackageNotFoundError:
        lop = "unknown"
    python = platform.python_version()
    if sys.platform == "darwin":
        system = f"macOS {platform.mac_ver()[0]} {platform.machine()}"
    else:
        system = f"{platform.system()} {platform.release()} {platform.machine()}"
    desktop = desktop_app if desktop_app and _VERSION.match(desktop_app) else None
    return _check(
        "versions",
        PASS,
        _t(_md.versions_ok(lop=lop, python=python, platform=system)),
        lop=lop,
        python=python,
        platform=system,
        desktop_app=desktop or "unknown",
        cloudflared=_cloudflared(),
    )


# --- the daemon ------------------------------------------------------------------


def daemon_pid(port: int) -> int | None:
    """The phone service's pid: the supervisor's answer, else the port's listener.

    The supervisor is asked first (`install._supervised_pid`, which already
    refuses to inspect a job a sandboxed run does not own); a foreground `lop
    mobile serve` has no supervisor, so the single process listening on the
    port answers instead. Two listeners, or no way to ask, is ``None``.
    """
    from local_operator.mobile import install

    try:
        pid = install._supervised_pid()
    except (OSError, subprocess.SubprocessError, ValueError):
        pid = None
    if pid is not None:
        return pid
    listeners = install._listening_pids(port)
    if listeners and len(listeners) == 1:
        only = next(iter(listeners))
        return int(only) if only.isdigit() else None
    return None


def open_files(pid: int | None) -> dict[str, Any]:
    """``{"open", "soft", "hard", "source"}`` for ``pid``; unknowns are ``None``.

    Linux reads `/proc/<pid>/fd` and `/proc/<pid>/limits` directly. macOS has no
    procfs: the count comes from `lsof -p` (bounded by a timeout) and the limit
    from the record the daemon writes at start (`fdlimit.recorded`). psutil is
    deliberately not used — it is not a dependency of this project.
    """
    from local_operator.mobile import fdlimit

    result: dict[str, Any] = {"open": None, "soft": None, "hard": None, "source": "unknown"}
    if pid is None:
        return result
    proc = Path(f"/proc/{pid}")
    if proc.is_dir():
        try:
            result["open"] = len(os.listdir(proc / "fd"))
            result["source"] = "proc"
        except OSError:
            pass
        try:
            for line in (proc / "limits").read_text().splitlines():
                if line.startswith("Max open files"):
                    parts = line.split()
                    soft, hard = parts[3], parts[4]
                    result["soft"] = int(soft) if soft.isdigit() else None
                    result["hard"] = int(hard) if hard.isdigit() else None
        except OSError:
            pass
    elif shutil.which("lsof"):
        try:
            listed = subprocess.run(  # noqa: S603 — fixed argv, no shell
                ["lsof", "-nP", "-p", str(pid), "-Ff"],
                capture_output=True,
                text=True,
                timeout=5,
            ).stdout
            # One `f<descriptor>` line per open file; cwd/txt/mem rows carry a
            # non-numeric descriptor and are not counted against the limit.
            result["open"] = sum(
                1 for line in listed.splitlines() if line[:1] == "f" and line[1:].isdigit()
            )
            result["source"] = "lsof"
        except (OSError, subprocess.SubprocessError):
            pass
    if result["soft"] is None:
        record = fdlimit.recorded(pid)
        if record is not None:
            result["soft"], result["hard"] = record.get("soft"), record.get("hard")
    return result


def check_daemon(port: int) -> tuple[dict[str, Any], int | None]:
    from local_operator.mobile import install

    installed = install.registration_present()
    started = time.monotonic()
    probe = install.health(port)
    latency_ms = round((time.monotonic() - started) * 1000)
    if probe is None:
        pid = daemon_pid(port)
        if pid is not None:
            # A process HOLDS the port but did not answer: the "Asleep / Not
            # answering" report. Measured on a rig at its descriptor limit: the
            # kernel still completes the TCP handshake, the event loop's accept()
            # fails with EMFILE, and the request hangs — so this is a failure,
            # and the pid is handed on so the files check can show the ceiling.
            return (
                _check(
                    "daemon",
                    FAIL,
                    _t(_md.daemon_not_answering()),
                    installed=installed,
                    state="not answering",
                    pid=pid,
                    port=port,
                    healthz_ms=None,
                    gate="unknown",
                ),
                pid,
            )
        summary = _md.daemon_not_running() if installed else _md.daemon_not_installed()
        # Never installed is an ordinary state (warn, not fail): the doctor is
        # also run on machines that never set phone access up.
        return (
            _check(
                "daemon",
                FAIL if installed else WARN,
                _t(summary),
                installed=installed,
                state="not running",
                pid=None,
                port=port,
                healthz_ms=None,
                gate="not running",
            ),
            None,
        )
    pid = daemon_pid(port)
    closed = install.gate_closed(port)
    if not closed:
        status, summary = FAIL, _t(_md.daemon_gate_open())
    elif latency_ms > SLOW_HEALTH_MS:
        status, summary = WARN, _t(_md.daemon_slow(ms=str(latency_ms)))
    else:
        status, summary = PASS, _t(_md.daemon_ok(ms=str(latency_ms)))
    return (
        _check(
            "daemon",
            status,
            summary,
            installed=installed,
            state="running",
            pid=pid,
            port=port,
            healthz_ms=latency_ms,
            gate="closed" if closed else "open",
            sessions=probe.get("sessions") if isinstance(probe, dict) else None,
        ),
        pid,
    )


def check_files(pid: int | None, running: bool) -> dict[str, Any]:
    if not running:
        return _check("files", PASS, _t(_md.files_not_running()), **open_files(None))
    files = open_files(pid)
    used, soft = files["open"], files["soft"]
    if used is None or soft is None:
        return _check("files", WARN, _t(_md.files_unknown()), **files)
    ratio = used / soft if soft else 1.0
    if ratio >= FILES_WARN_RATIO:
        # AT the limit is a failure (accept() is already returning EMFILE and
        # the phone hangs); near it is a warning.
        summary = _md.files_near(open=str(used), limit=str(soft))
        status = FAIL if used >= soft else WARN
        return _check("files", status, _t(summary), ratio=round(ratio, 3), **files)
    summary = _md.files_ok(open=str(used), limit=str(soft))
    return _check("files", PASS, _t(summary), ratio=round(ratio, 3), **files)


# --- logs ------------------------------------------------------------------------


def _scan_log(path: Path, *, now: float) -> dict[str, int] | None:
    """Count EMFILE and sqlite-open lines from the last day of ``path``'s tail.

    COUNTS ONLY — no log text leaves this function, because a log line can
    carry anything (a path, a prompt, a provider sentence). A line with no
    timestamp of its own (a traceback continuation) inherits the last stamp
    seen; lines before the first stamp count only when the file was written
    within the window.
    """
    try:
        with path.open("rb") as handle:
            handle.seek(0, 2)
            size = handle.tell()
            handle.seek(max(0, size - LOG_TAIL_BYTES))
            tail = handle.read().decode("utf-8", "replace")
        modified = path.stat().st_mtime
    except OSError:
        return None
    recent = now - modified <= LOG_WINDOW_S
    counts = {"emfile": 0, "sqlite_unable_to_open": 0}
    for line in tail.splitlines():
        stamp = _STAMP.match(line)
        if stamp:
            try:
                at = datetime.datetime.fromisoformat(f"{stamp.group(1)} {stamp.group(2)}")
                recent = now - at.timestamp() <= LOG_WINDOW_S
            except ValueError:
                pass
        if not recent:
            continue
        if _EMFILE.search(line):
            counts["emfile"] += 1
        if _SQLITE_OPEN.search(line):
            counts["sqlite_unable_to_open"] += 1
    return counts


def check_logs(now: float | None = None) -> dict[str, Any]:
    from local_operator.paths import RUNTIME_LOG_FILENAME, log_dir

    moment = time.time() if now is None else now
    files: dict[str, Any] = {}
    totals = {"emfile": 0, "sqlite_unable_to_open": 0}
    for name in ("mobile.log", "mobile.log.1", RUNTIME_LOG_FILENAME):
        counts = _scan_log(log_dir() / name, now=moment)
        if counts is None:
            continue
        files[name] = counts
        for key in totals:
            totals[key] += counts[key]
    # The counts cover each file's last `LOG_TAIL_BYTES`, so under a flood they
    # are a lower bound — said in the data rather than implied.
    data = {
        "window_hours": LOG_WINDOW_S // 3600,
        "tail_mib_per_file": LOG_TAIL_BYTES // (1024 * 1024),
        "files": files,
        **totals,
    }
    if totals["emfile"] or totals["sqlite_unable_to_open"]:
        summary = _md.logs_errors(
            emfile=str(totals["emfile"]), sqlite=str(totals["sqlite_unable_to_open"])
        )
        return _check("logs", WARN, _t(summary), **data)
    return _check("logs", PASS, _t(_md.logs_clean()), **data)


# --- login and tunnel --------------------------------------------------------------


def _radient_rows() -> list[Any]:
    from contextlib import closing

    from local_operator.providers.auth_store import AuthStore, AuthStoreError

    try:
        with closing(AuthStore(read_only=True)) as store:
            return [r for r in store.list_credentials("radient") if r.credential_type == "oauth"]
    except (AuthStoreError, OSError, ValueError):
        return []


def check_login() -> dict[str, Any]:
    signed_in = bool(_radient_rows())
    summary = _md.login_ok() if signed_in else _md.login_missing()
    # A yes/no only: no account label, email or token-derived field.
    return _check("login", PASS if signed_in else WARN, _t(summary), signed_in=signed_in)


def _harness_hostname(record: dict[str, Any]) -> str | None:
    for harness in record.get("harnesses") or []:
        if isinstance(harness, dict) and harness.get("id") == "local-operator":
            host = harness.get("hostname")
            return host if isinstance(host, str) else None
    return None


async def _tunnels_for_this_device(value: dict[str, Any]) -> int | None:
    """How many of the account's tunnels belong to THIS device, or ``None``.

    The duplicate a 503-then-409 setup can leave behind: two cloud records for
    one computer, the phone paired to the stale one. Matched on the stored
    record's ``device_id`` (falling back to its name), through the tunnel's own
    pinned login, with one read-only GET. Any failure is ``None`` ("not
    checked"), never a guess.
    """
    import httpx

    from local_operator.tunnels.api import RadientTunnels

    stored = value.get("record")
    record: dict[str, Any] = stored if isinstance(stored, dict) else {}
    device, name = record.get("device_id"), record.get("name")
    credential = value.get("credential_id")
    if not isinstance(credential, int) or not (device or name):
        return None
    try:
        async with httpx.AsyncClient(trust_env=False, timeout=10) as client:
            listed = await RadientTunnels(credential, client).request("GET")
    except (ValueError, OSError, httpx.HTTPError):
        return None
    if isinstance(listed, dict):
        listed = listed.get("tunnels")
    if not isinstance(listed, list):
        return None
    key, wanted = ("device_id", device) if device else ("name", name)
    return sum(1 for item in listed if isinstance(item, dict) and item.get(key) == wanted)


async def check_tunnel(*, remote: bool = True) -> list[dict[str, Any]]:
    from local_operator.tunnels import config as tunnel_config
    from local_operator.tunnels import report

    try:
        value = tunnel_config.load()
    except (OSError, ValueError):
        value = None
    if not value:
        return [_check("tunnel", WARN, _t(_md.tunnel_not_configured()), configured=False)]
    stored = value.get("record")
    record: dict[str, Any] = stored if isinstance(stored, dict) else {}
    try:
        payload = await report.local_payload()
        connector = payload.get("connector") or {}
    except (OSError, ValueError):
        connector = {}
    state = str(connector.get("state") or "unknown")
    reason = str(connector.get("reason") or "")
    data = {
        "configured": True,
        "tunnel_id": value.get("tunnel_id"),
        "hostname": _harness_hostname(record),
        "connector_state": state,
        "connector_reason": reason,
        "cloud_status": record.get("status"),
    }
    if state == "connected":
        tunnel = _check("tunnel", PASS, _t(_md.tunnel_ok(state=state)), **data)
    else:
        problem = _md.tunnel_problem(state=state, reason=reason or "-")
        tunnel = _check("tunnel", FAIL if state == "parked" else WARN, _t(problem), **data)
    count = await _tunnels_for_this_device(value) if remote else None
    if count is None:
        duplicates = _check(
            "tunnel_duplicates", WARN, _t(_md.tunnel_list_unchecked()), checked=False
        )
    elif count > 1:
        duplicates = _check(
            "tunnel_duplicates",
            FAIL,
            _t(_md.tunnel_duplicates(count=str(count))),
            checked=True,
            count=count,
        )
    else:
        duplicates = _check(
            "tunnel_duplicates", PASS, _t(_md.tunnel_single()), checked=True, count=count
        )
    return [tunnel, duplicates]


# --- assembly ----------------------------------------------------------------------


def _homes() -> list[str]:
    from local_operator import supervisors

    found = {str(Path.home())}
    real = supervisors.real_home()
    if real is not None:
        found.add(str(real))
    return sorted((h for h in found if h and h != "/"), key=len, reverse=True)


def _scrub(value: Any, homes: list[str]) -> Any:
    """Final redaction pass: home → ``~``, emails removed, recursively.

    A second line of defence, not the first: every field is chosen from fixed
    facts above. It is here so a future field that carries a path or an account
    label cannot leak the operator's home or address into a pasted report.
    """
    if isinstance(value, dict):
        return {k: _scrub(v, homes) for k, v in value.items()}
    if isinstance(value, list):
        return [_scrub(v, homes) for v in value]
    if isinstance(value, str):
        for home in homes:
            value = value.replace(home, "~")
        return _EMAIL.sub("<email>", value)
    return value


async def collect(
    *, port: int | None = None, desktop_app: str | None = None, remote: bool = True
) -> dict[str, Any]:
    """Every check, redacted. Blocking probes run off the event loop."""
    from local_operator.mobile.daemon import DEFAULT_PORT

    chosen = DEFAULT_PORT if port is None else port
    versions = await asyncio.to_thread(check_versions, desktop_app)
    daemon, pid = await asyncio.to_thread(check_daemon, chosen)
    alive = daemon["data"]["state"] in ("running", "not answering")
    files = await asyncio.to_thread(check_files, pid, alive)
    logs = await asyncio.to_thread(check_logs)
    login = await asyncio.to_thread(check_login)
    tunnel = await check_tunnel(remote=remote)
    checks = [versions, daemon, files, logs, *tunnel, login]
    statuses = {c["status"] for c in checks}
    overall = FAIL if FAIL in statuses else WARN if WARN in statuses else PASS
    report = {
        "schema": SCHEMA,
        "generated_at": datetime.datetime.now(datetime.timezone.utc).isoformat(timespec="seconds"),
        "overall": overall,
        "checks": checks,
    }
    return _scrub(report, _homes())


def text(report: dict[str, Any]) -> str:
    """The human rendering: one line per check, status first."""
    lines = [f"overall: {report['overall']}"]
    for check in report["checks"]:
        lines.append(f"[{check['status']}] {check['id']}: {check['summary']}")
    return "\n".join(lines)
