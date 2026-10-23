#!/usr/bin/env python3
"""Prove the daemon's own session-start hub check with ONE request of evidence.

Run it with the repository's interpreter, from anywhere:

    .venv/bin/python scripts/hub_daemon_check_receipt.py

No arguments, no flags, no environment to set: every value this needs is derived
below or built into a fresh temporary HOME. It exits 0 only when the delta it
measures is exactly the one request the trigger is supposed to cause.

WHAT THE Q-1 FINDING WAS, AND WHY THE FIRST ATTEMPT WAS REJECTED. #2822 attached
the session-start trigger to the desktop create route (``server/routes/
desktop_sessions.py::_note_hub_session_start``), and the audit asked for a
receipt for the trigger's PRODUCTION ATTACHMENT: ``server/app.py``'s lifespan
constructing ``HubSyncRunner`` and handing it to ``app.state.hub_sync``, which is
the object the route reaches. ``scripts/hub_session_check_receipt.py`` (PR #2843)
answered with a script that CONSTRUCTED the runner itself, injected it onto a
FastAPI app it built, and ran a tick on it. That reproduces the injection the
finding is about: it proves the runner checks when somebody calls it, not that
the daemon attaches one. It was rejected for exactly that.

WHAT THIS DOES INSTEAD -- AND THE ONE RULE FOR ANY FUTURE EDIT. Nothing here
constructs a runner, a route, an app or a lifespan. The real ``lop serve``
daemon is started as a subprocess from THIS repository's source; its OWN lifespan
attaches the runner and starts ``run_forever``; the session is created over real
HTTP against it. Every fact below is observed from OUTSIDE that process: the hub
stub's request log, the daemon's ``/health``, and the daemon's status store. If a
future edit needs to import ``HubSyncRunner`` here, the edit has re-created the
rejected harness -- stop, and find a way to observe instead.

THE EVIDENCE IS A DELTA OF ONE, AND WHY A COUNT RATHER THAN A TIMESTAMP. The
runner fires a tick on its own at startup (``STARTUP_DELAY_S`` + jitter), so
``last_checked_at`` moving after a create proves nothing by itself: the startup
tick already wrote it, and both writes can land in the same second. The
unambiguous evidence is the stub's request count taken after the startup pass has
gone quiet and again after the create. The delta must be exactly 1, and that one
request must be the skill read the check makes.

WHAT MAKES THE ONE REQUEST REACHABLE (the fixture, and why it is shaped this
way). A check is per LINKED item, and a fresh isolated config has no links -- so
an unseeded run would produce a delta of zero and prove nothing. This script
therefore seeds exactly two things in the isolated config root before the daemon
starts:

* ONE linked skill baseline (``<config>/hub/baselines/skill-<id>.json``) with a
  ``content_state`` of ``absent``. That is a real, supported link shape -- a
  subscriber whose content is not installed -- and the check still FETCHES the
  remote for it (``hub_sync/check.py::check_hub_skills`` says so explicitly:
  "the check still FETCHES the remote (that is the whole point of subscribing)").
  It is chosen over a present skill because a present one drags in managed-skill
  directory parsing, YAML frontmatter and the unpull journal -- three more ways
  for a receipt about ATTACHMENT to fail for reasons that have nothing to do
  with attachment. The check arm exercised is the same either way.
* ONE dummy OAuth bearer in the isolated auth store (``<config>/auth.db``), the
  value generated per run by this script and never a real token. The baseline's
  tenant makes the check take the ORGANIZATION client, so the request carries a
  bearer and the stub can prove the credential came from the isolated store.

The count is one because a skill link's check is exactly one read
(``GET {base}/skills/{id}``). Agent and team links would each add their own
detail read (``check_hub_agents`` / ``check_hub_teams``); none are seeded, so
those arms have nothing to fetch. The endpoint list is enumerated from
``clients/radient.py`` and ``hub_sync/check.py``, not assumed.

ISOLATION. ``scripts.probe_isolation`` is the FIRST import: it re-homes HOME and
``LOCAL_OPERATOR_CONFIG_DIR`` to a fresh ``mkdtemp`` and drops inherited ``CMUX_*``
before any application module can resolve the operator's real ``~/.local-operator``.
Everything this script writes -- the credential, the baseline, the daemon's
config, sessions, logs, status store, the ``auth.db`` -- lands under that
sandbox. ``RADIENT_*`` and ``LOCAL_OPERATOR_*`` are stripped from the parent
before the import and the daemon's environment is built FROM SCRATCH (not
inherited), so the child gets no credential the operator holds.

EXTERNAL EGRESS. The only hub base the daemon is given is
``RADIENT_API_BASE_URL=http://127.0.0.1:8127`` (the stub), and no real
credential exists in its environment or its store. As a belt on top of that,
``REQUESTS_CA_BUNDLE``/``CURL_CA_BUNDLE`` point at a path that does not exist, so
a request to any real HTTPS host fails TLS verification before bytes leave -- the
hub client is ``requests``. A stub that sees an unexpected request logs it and
the assertion below fails on it.

TIMING. The whole run is bounded by ``BUDGET_S`` (240 s) and a watchdog kills and
reaps at that mark rather than hanging. The budget is spent like this:

    daemon boot to /health           ~5-25 s (measured, printed)
    runner's first tick              45-60 s (STARTUP_DELAY_S + STARTUP_JITTER_S)
    quiet confirmation               +10 s
    session-start throttle           120 s (SESSION_START_CHECK_THROTTLE_S) + 3 s
    create + delta + reap            ~10-15 s

The two runner constants are READ FROM THE MODULE, not restated, and preflight
refuses to start if their arithmetic no longer fits the budget -- a code change
that moves either one fails here with numbers rather than as a timeout.

EXIT CODES. 0 = the receipt passed; 2 = a preflight refused (nothing was
started); 3 = the 240 s budget was exhausted; 4 = the receipt did not hold
(delta wrong, create failed, or the wrong request). Every non-zero path prints
the full request log and, for daemon trouble, the tail of the daemon's own log.

WHAT THIS IS NOT. It is not a test (no CI, no pytest); it is a receipt for a
person on a machine whose resource gate allows one daemon boot and ~150 s of
idle waiting. It asserts nothing about whether an available item should be
APPLIED -- ``check_now`` applies nothing by contract, and the startup tick is
inside its first-run grace.
"""

from __future__ import annotations

import hashlib
import json
import os
import secrets
import signal
import socket
import subprocess
import sys
import threading
import time
import uuid
from dataclasses import dataclass
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any

import requests

# The budget is the whole script's wall clock, so its clock starts before any
# work -- including the temp-home creation below.
_T0 = time.monotonic()

# -- isolation, before anything under local_operator can be imported -----------
#
# The parent's environment is scrubbed FIRST, so no inherited credential can be
# read back through a resolver that this script runs in-process (the preflight
# below resolves the seeded bearer on purpose) or leak into a print.
for _key in tuple(os.environ):
    if _key.startswith(("CMUX_", "RADIENT_", "LOCAL_OPERATOR_")):
        os.environ.pop(_key, None)

_REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_REPO_ROOT))

import scripts.probe_isolation  # noqa: E402,F401  -- must precede every app import

# The stub's base and the org-destination opt-in are this process's own values,
# set after the scrub so the in-process preflight answers what the daemon will.
# ``RADIENT_ORG_ALLOW_NONCANONICAL_BASE`` is the documented opt-in for a rig that
# points the CLI at 127.0.0.1 (see ``providers/radient_credentials.py``); the
# bearer it lets travel is this run's dummy, held only in the sandbox store.
_HUB_PORT = 8127
_HUB_ROOT = f"http://127.0.0.1:{_HUB_PORT}"
_HUB_BASE = f"{_HUB_ROOT}/v1"
os.environ["RADIENT_API_BASE_URL"] = _HUB_BASE
os.environ["RADIENT_ORG_ALLOW_NONCANONICAL_BASE"] = "1"

from local_operator.config import ConfigManager  # noqa: E402
from local_operator.hub_sync import provenance as prov  # noqa: E402
from local_operator.hub_sync import store as status_store  # noqa: E402
from local_operator.hub_sync.runner import (  # noqa: E402
    SESSION_START_CHECK_THROTTLE_S,
    STARTUP_DELAY_S,
    STARTUP_JITTER_S,
)
from local_operator.hub_sync.skill_document import SkillDocument, SkillFile  # noqa: E402
from local_operator.providers.auth_store import AuthStore  # noqa: E402
from local_operator.providers.radient_credentials import (  # noqa: E402
    configured_radient_base_url,
    resolve_radient_oauth_access,
)

#: The receipt's own budget. A hard wall, enforced by a watchdog: a hung daemon
#: must be reaped and reported, never waited on.
BUDGET_S = 240.0
#: "The startup pass has settled": no request for this long.
QUIET_S = 10.0
#: The task's cap on waiting for the startup tick before giving up on it.
STARTUP_CAP_S = 120.0
#: How long the trigger's own request may take to appear after the create.
POST_CREATE_WAIT_S = 20.0
#: Extra wait on top of the throttle, so scheduling jitter cannot land us a
#: millisecond on the wrong side of the runner's monotonic comparison.
THROTTLE_MARGIN_S = 3.0
#: Room the rest of the script needs after the throttle has elapsed: the
#: create, the wait for its request, the assertions and the teardown. Sized from
#: the measured shape of those steps, not padded -- every second here is a second
#: the startup window cannot have.
POST_THROTTLE_RESERVE_S = 20.0

_SANDBOX = Path(os.environ["HOME"])
_CONFIG_DIR = Path(os.environ["LOCAL_OPERATOR_CONFIG_DIR"])
_WORK_DIR = _SANDBOX / "work"

#: Identifies the seeded link. ``local_id`` is a safe single path segment (it
#: names the baseline file); ``hub_id`` is what the check asks the hub for.
_LOCAL_ID = "d4e7a1c9b2f0"
_HUB_ID = "daemon-receipt-skill"
_ALIAS = "daemon-receipt-probe"
_TENANT = "receipt-tenant"
_DESCRIPTION = "fixture skill for the daemon hub-check receipt"

#: The dummy bearer, generated per run. It is written to the isolated store and
#: is the only value the stub should ever see; the real store is never opened.
_DUMMY_BEARER = secrets.token_urlsafe(32)
_DUMMY_REFRESH = "dummy-refresh-never-used"

#: The desktop plane opens only for a bearer in the daemon's environment. This
#: one names this run's loopback daemon and is a credential to nothing else.
_DESKTOP_TOKEN = secrets.token_urlsafe(24)

_SKILL_TEXT = (
    "---\n"
    f"name: {_ALIAS}\n"
    f"description: {_DESCRIPTION}\n"
    "---\n\n"
    "# Daemon receipt probe\n\n"
    "Fixture content for scripts/hub_daemon_check_receipt.py. Never installed.\n"
)


def _say(line: str = "") -> None:
    print(line, flush=True)


def _short_digest(value: str) -> str:
    """A stable, non-reversible label for a secret this script owns."""

    return hashlib.sha256(value.encode("utf-8")).hexdigest()[:12]


class _Fail(Exception):
    """A named refusal. ``code`` decides the exit status; the message is printed."""

    def __init__(self, message: str, *, code: int) -> None:
        super().__init__(message)
        self.code = code


@dataclass(frozen=True)
class _Seen:
    """One request as the stub saw it. Never carries a credential value."""

    seq: int
    epoch: float
    method: str
    path: str
    status: int
    auth: str  # "dummy" | "other" | "absent"


class _HubStub:
    """A loopback hub that answers the one read a skill check makes, and logs.

    The log is the receipt: every request is recorded with method, path, wall
    timestamp, the status this stub answered and a CLASSIFICATION of the
    Authorization header (never its value). Anything that is not the seeded
    skill's read is answered 404 and still logged, so an unexpected caller shows
    up in the output instead of silently vanishing.
    """

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._seen: list[_Seen] = []
        self._server = ThreadingHTTPServer(("127.0.0.1", _HUB_PORT), self._handler())
        self._server.daemon_threads = True
        self._thread: threading.Thread | None = None
        self._manifest = {
            "document_version": 1,
            "name": _ALIAS,
            "description": _DESCRIPTION,
            "files": [{"path": "SKILL.md", "content": _SKILL_TEXT}],
            # Envelope metadata a real hub detail carries; the adapter ignores
            # it when it builds the strict manifest.
            "id": _HUB_ID,
            "tenant_id": _TENANT,
            "revision": "receipt-rev-1",
        }

    # -- HTTP ---------------------------------------------------------------

    def _handler(self) -> type[BaseHTTPRequestHandler]:
        stub = self

        class Handler(BaseHTTPRequestHandler):
            # HTTP/1.1 with an explicit Content-Length on every response: the
            # client is ``requests`` and keep-alive is what a real hub gives it.
            protocol_version = "HTTP/1.1"

            def do_GET(self) -> None:  # noqa: N802 - http.server's spelling
                stub._answer(self, "GET")

            def do_POST(self) -> None:  # noqa: N802
                stub._answer(self, "POST")

            def do_PUT(self) -> None:  # noqa: N802
                stub._answer(self, "PUT")

            def do_PATCH(self) -> None:  # noqa: N802
                stub._answer(self, "PATCH")

            def do_DELETE(self) -> None:  # noqa: N802
                stub._answer(self, "DELETE")

            def do_HEAD(self) -> None:  # noqa: N802
                stub._answer(self, "HEAD")

            def log_message(self, format: str, *args: Any) -> None:
                """Silenced: the printed request log is the evidence, not stderr."""

        return Handler

    def _answer(self, handler: BaseHTTPRequestHandler, method: str) -> None:
        path = handler.path
        answered = HTTPStatus.OK
        payload: dict[str, Any] = self._manifest
        if path != f"/v1/skills/{_HUB_ID}":
            answered = HTTPStatus.NOT_FOUND
            payload = {"error": "not in this stub", "path": path}
        body = json.dumps(payload).encode("utf-8")
        self._record(method, path, int(answered), handler.headers.get("authorization", ""))
        handler.send_response(int(answered))
        handler.send_header("Content-Type", "application/json")
        handler.send_header("Content-Length", str(len(body)))
        handler.end_headers()
        if method != "HEAD":
            handler.wfile.write(body)

    # -- the log ------------------------------------------------------------

    def _record(self, method: str, path: str, status: int, authorization: str) -> None:
        if not authorization:
            auth = "absent"
        elif authorization == f"Bearer {_DUMMY_BEARER}":
            auth = "dummy"
        else:
            auth = "other"
        with self._lock:
            self._seen.append(_Seen(len(self._seen) + 1, time.time(), method, path, status, auth))

    def snapshot(self) -> list[_Seen]:
        with self._lock:
            return list(self._seen)

    def count(self) -> int:
        with self._lock:
            return len(self._seen)

    def last_epoch(self) -> float | None:
        with self._lock:
            return self._seen[-1].epoch if self._seen else None

    # -- lifecycle ----------------------------------------------------------

    def start(self) -> None:
        self._thread = threading.Thread(
            target=self._server.serve_forever, name="hub-stub", daemon=True
        )
        self._thread.start()

    def stop(self) -> None:
        self._server.shutdown()
        self._server.server_close()
        if self._thread is not None:
            self._thread.join(timeout=5.0)


def _stamp(epoch: float) -> str:
    millis = int((epoch % 1) * 1000)
    return f"{time.strftime('%H:%M:%S', time.localtime(epoch))}.{millis:03d}"


def _print_log(seen: list[_Seen], title: str) -> None:
    _say(f"{title} ({len(seen)} request(s)):")
    if not seen:
        _say("  <empty: the stub saw no request at all>")
        return
    for item in seen:
        _say(
            f"  #{item.seq} {_stamp(item.epoch)} {item.method} {item.path}"
            f" -> {item.status} auth={item.auth}"
        )


def _tail(path: Path, lines: int = 40) -> str:
    try:
        text = path.read_text(encoding="utf-8", errors="replace")
    except OSError as error:
        return f"<unreadable daemon log: {error}>"
    return "\n".join(text.splitlines()[-lines:])


def _port_free(port: int) -> bool:
    """Whether a listener could bind this loopback port right now.

    ``SO_REUSEADDR`` matches what ``ThreadingHTTPServer`` sets, so this predicts
    the stub's own bind rather than a stricter rule -- and it still refuses a
    port another process is LISTENING on, which is the case it exists for.
    """

    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        try:
            sock.bind(("127.0.0.1", port))
        except OSError:
            return False
    return True


def _free_port() -> int:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        sock.bind(("127.0.0.1", 0))
        return int(sock.getsockname()[1])


def _wait_until(moment: float) -> None:
    while True:
        left = moment - time.monotonic()
        if left <= 0:
            return
        time.sleep(min(left, 0.5))


# -- the daemon ----------------------------------------------------------------


def _child_env() -> dict[str, str]:
    """The daemon's environment, built from nothing.

    NOT inherited-and-filtered: a variable nobody here named cannot reach the
    child, which is the property that makes "the only base it is given" true.
    """

    return {
        "HOME": str(_SANDBOX),
        "PATH": os.environ.get("PATH", "/usr/bin:/bin:/usr/sbin:/sbin"),
        "TMPDIR": str(_SANDBOX / "tmp"),
        "LANG": "en_US.UTF-8",
        "LC_ALL": "en_US.UTF-8",
        "TERM": "xterm-256color",
        # The isolated root the daemon must use for config, sessions, hub state.
        "LOCAL_OPERATOR_CONFIG_DIR": str(_CONFIG_DIR),
        # The desktop plane's capability, for THIS loopback daemon only.
        "LOCAL_OPERATOR_DESKTOP_TOKEN": _DESKTOP_TOKEN,
        # The isolation contract's literals (scripts/probe_isolation.py), so the
        # daemon cannot launch a desktop, raise a notification or retitle a
        # terminal it does not own.
        "LOCAL_OPERATOR_NO_DESKTOP_LAUNCH": "1",
        "LOCAL_OPERATOR_NO_NOTIFICATIONS": "1",
        "LOCAL_OPERATOR_NO_TERMINAL_TITLE": "1",
        "LOCAL_OPERATOR_NO_SHIMMER": "1",
        # THE ONLY HUB BASE THE DAEMON IS GIVEN.
        "RADIENT_API_BASE_URL": _HUB_BASE,
        # The documented opt-in that lets this rig's org bearer travel to
        # 127.0.0.1; the bearer is this run's dummy in the sandbox store.
        "RADIENT_ORG_ALLOW_NONCANONICAL_BASE": "1",
        # Egress belt: a request to a real HTTPS host cannot verify. The hub
        # client is ``requests``, which reads this; the stub is plain HTTP.
        "REQUESTS_CA_BUNDLE": str(_SANDBOX / "no-such-ca-bundle.pem"),
        "CURL_CA_BUNDLE": str(_SANDBOX / "no-such-ca-bundle.pem"),
        "PYTHONUNBUFFERED": "1",
        "PYTHONDONTWRITEBYTECODE": "1",
    }


def _start_daemon(port: int, log_path: Path) -> subprocess.Popen[bytes]:
    log_path.parent.mkdir(parents=True, exist_ok=True)
    log_file = log_path.open("wb")
    try:
        proc = subprocess.Popen(
            [
                sys.executable,
                "-m",
                "local_operator.cli",
                "serve",
                "--host",
                "127.0.0.1",
                "--port",
                str(port),
            ],
            cwd=str(_REPO_ROOT),
            env=_child_env(),
            stdin=subprocess.DEVNULL,
            stdout=log_file,
            stderr=subprocess.STDOUT,
            # Its own process group, so a survivor anywhere in its tree can be
            # signalled without reaching this script.
            start_new_session=True,
        )
    finally:
        log_file.close()
    return proc


def _open_stub() -> _HubStub:
    """Bind and start the stub, or refuse before a daemon exists to blame."""

    try:
        stub = _HubStub()
        stub.start()
    except OSError as error:
        raise _Fail(
            f"the hub stub could not bind 127.0.0.1:{_HUB_PORT}: {error}", code=2
        ) from error
    return stub


def _open_daemon(port: int, log_path: Path) -> subprocess.Popen[bytes]:
    try:
        return _start_daemon(port, log_path)
    except OSError as error:
        raise _Fail(f"the daemon could not be started: {error}", code=2) from error


def _wait_healthy(proc: subprocess.Popen[bytes], port: int, log_path: Path, until: float) -> float:
    started = time.monotonic()
    url = f"http://127.0.0.1:{port}/health"
    while True:
        code = proc.poll()
        if code is not None:
            raise _Fail(
                f"the daemon exited before answering /health (rc={code}).\n"
                f"daemon log tail ({log_path}):\n{_tail(log_path)}",
                code=4,
            )
        try:
            response = requests.get(url, timeout=2.0)
            if response.status_code == 200:
                return time.monotonic() - started
        except requests.RequestException:
            pass
        if time.monotonic() >= until:
            raise _Fail(
                f"the daemon did not answer {url} before its deadline.\n"
                f"daemon log tail ({log_path}):\n{_tail(log_path)}",
                code=4,
            )
        time.sleep(0.25)


def _descendants(root_pid: int) -> list[int]:
    """Every live descendant of ``root_pid``, found BEFORE anything is killed."""

    try:
        listing = subprocess.run(
            ["ps", "-axo", "pid=,ppid="],
            capture_output=True,
            text=True,
            timeout=5.0,
            check=False,
        ).stdout
    except (OSError, subprocess.SubprocessError):
        return []
    children: dict[int, list[int]] = {}
    for line in listing.splitlines():
        parts = line.split()
        if len(parts) != 2:
            continue
        try:
            child, parent = int(parts[0]), int(parts[1])
        except ValueError:
            continue
        children.setdefault(parent, []).append(child)
    found: list[int] = []
    pending = [root_pid]
    while pending:
        current = pending.pop()
        for child in children.get(current, []):
            if child not in found:
                found.append(child)
                pending.append(child)
    return found


class _Reaper:
    """Idempotent, pid-exact teardown. The watchdog and the normal path share it.

    ``find children BEFORE killing the parent`` is the contract: a killed parent
    re-parents its children, and a re-parented pid is not one this run can name.
    """

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._done = False

    def reap(
        self,
        proc: subprocess.Popen[bytes] | None,
        stub: _HubStub | None,
        ports: list[int],
    ) -> bool:
        """Stop everything this run started; True only when every port came free.

        NEVER RAISES. It runs from a ``finally`` and from the watchdog, where an
        exception would either mask the failure that got us here or kill the
        cleanup half-done. The caller turns a ``False`` into its own refusal.
        """

        with self._lock:
            if self._done:
                return True
            self._done = True

        survivors: list[int] = []
        if proc is not None and proc.poll() is None:
            children = _descendants(proc.pid)
            _say(f"reap       daemon pid {proc.pid}; children found first: {children or 'none'}")
            try:
                proc.terminate()
            except ProcessLookupError:
                pass
            try:
                proc.wait(timeout=12.0)
            except subprocess.TimeoutExpired:
                _say(f"reap       pid {proc.pid} ignored SIGTERM; SIGKILL")
                try:
                    proc.kill()
                    proc.wait(timeout=5.0)
                except (ProcessLookupError, subprocess.TimeoutExpired):
                    pass
            for pid in children:
                if _pid_alive(pid):
                    survivors.append(pid)
            _say(f"reap       daemon pid {proc.pid} exited (rc={proc.returncode})")

        for pid in survivors:
            _say(f"reap       child pid {pid} survived the daemon; SIGKILL")
            _kill_quietly(pid, signal.SIGKILL)

        if stub is not None:
            stub.stop()
            _say("reap       hub stub thread stopped")

        # The listener closes asynchronously in the kernel; give the ports a
        # bounded moment rather than reading a transient bind failure as a leak.
        settle = time.monotonic() + 5.0
        while True:
            held = [f"127.0.0.1:{port}" for port in ports if not _port_free(port)]
            if not held or time.monotonic() >= settle:
                break
            time.sleep(0.2)
        if not held:
            _say(f"SLOT FREE  {', '.join(f'127.0.0.1:{port}' for port in ports)}")
            return True
        _say(f"SLOT HELD  {', '.join(held)} (something still listens there)")
        return False


def _pid_alive(pid: int) -> bool:
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return False
    except PermissionError:
        return True
    return True


def _kill_quietly(pid: int, sig: int) -> None:
    try:
        os.kill(pid, sig)
    except ProcessLookupError:
        pass


# -- fixtures ------------------------------------------------------------------


def _seed() -> dict[str, Any]:
    """Write the dummy bearer and ONE linked skill into the isolated root.

    Both writes are the app's own writers (``AuthStore.upsert_credential``,
    ``provenance.write_baseline``), so the files the daemon reads are the ones
    the product would have written -- not a hand-spelled JSON the parser happens
    to tolerate.
    """

    _CONFIG_DIR.mkdir(parents=True, exist_ok=True)
    store = AuthStore(_CONFIG_DIR / "auth.db", config_dir=_CONFIG_DIR)
    try:
        store.upsert_credential(
            "radient",
            {
                "type": "oauth",
                "access": _DUMMY_BEARER,
                "refresh": _DUMMY_REFRESH,
                # Six hours: far past the run, so no refresh is attempted and
                # the dummy refresh token can never be presented to anything.
                "expires": int(time.time() * 1000) + 6 * 60 * 60 * 1000,
            },
        )
    finally:
        store.close()

    document = SkillDocument(
        name=_ALIAS,
        description=_DESCRIPTION,
        files=(SkillFile("SKILL.md", _SKILL_TEXT),),
    )
    fields = document.to_json()
    prov.write_baseline(
        _CONFIG_DIR,
        prov.BaselineRecord(
            kind="skill",
            local_id=_LOCAL_ID,
            hub_id=_HUB_ID,
            tenant_id=_TENANT,
            fingerprint=prov.fingerprint_snapshot("skill", fields),
            fields=fields,
            recorded_by="pull",
            local_alias=_ALIAS,
            content_state="absent",
        ),
    )
    return fields


def _resolved_credential() -> Any:
    """The store's answer, through the same resolver the runner uses."""

    import asyncio

    return asyncio.run(resolve_radient_oauth_access(_CONFIG_DIR, _HUB_BASE))


def _last_checked_at() -> str | None:
    """The status store's ``last_checked_at`` for the seeded link, if written."""

    try:
        document = status_store.StatusStore(_CONFIG_DIR).load()
    except Exception:  # noqa: BLE001 - an absent store is reported as absent
        return None
    items = document.get("items")
    row = items.get(f"skill:{_LOCAL_ID}") if isinstance(items, dict) else None
    value = row.get("last_checked_at") if isinstance(row, dict) else None
    return str(value) if value else None


# -- preflight -----------------------------------------------------------------


@dataclass(frozen=True)
class _Plan:
    daemon_port: int
    module_path: Path
    head: str
    dirty: bool


def _probe_daemon_interpreter() -> tuple[Path, str]:
    """Resolve what ``sys.executable -m local_operator.cli`` actually imports.

    Run through the SAME interpreter and cwd the daemon will use, so the path
    printed in the receipt is the one the daemon loads -- an installed
    generation answering instead of this worktree is the failure this catches.
    """

    probe = (
        "import fastapi, local_operator, requests, uvicorn; "
        "print(local_operator.__file__); "
        "print(uvicorn.__version__)"
    )
    try:
        finished = subprocess.run(
            [sys.executable, "-c", probe],
            cwd=str(_REPO_ROOT),
            env=_child_env(),
            capture_output=True,
            text=True,
            timeout=60.0,
            check=False,
        )
    except (OSError, subprocess.SubprocessError) as error:
        raise _Fail(f"could not probe the daemon interpreter: {error}", code=2) from error
    if finished.returncode != 0:
        raise _Fail(
            "the daemon interpreter cannot import the server stack "
            f"(rc={finished.returncode}):\n{finished.stdout}{finished.stderr}",
            code=2,
        )
    lines = finished.stdout.strip().splitlines()
    if len(lines) < 2:
        raise _Fail(f"unexpected interpreter probe output: {finished.stdout!r}", code=2)
    module = Path(lines[0]).resolve()
    if _REPO_ROOT not in module.parents:
        raise _Fail(
            f"`{sys.executable} -m local_operator.cli` would load {module}, "
            f"which is NOT this repository ({_REPO_ROOT}). Run with .venv/bin/python.",
            code=2,
        )
    return module, lines[1]


def _git(*args: str) -> str:
    try:
        finished = subprocess.run(
            ["git", "-C", str(_REPO_ROOT), *args],
            capture_output=True,
            text=True,
            timeout=15.0,
            check=False,
            env={
                "PATH": os.environ.get("PATH", "/usr/bin:/bin:/usr/sbin:/sbin"),
                "HOME": str(_SANDBOX),
                # No credential lookups under a scratch HOME (see AGENTS.md's
                # rig guidance: a redirected HOME must not reach a keychain).
                "GIT_CONFIG_SYSTEM": "/dev/null",
                "GIT_CONFIG_GLOBAL": "/dev/null",
                "GIT_TERMINAL_PROMPT": "0",
            },
        )
    except (OSError, subprocess.SubprocessError) as error:
        raise _Fail(f"`git {' '.join(args)}` failed: {error}", code=2) from error
    return finished.stdout.strip()


def _preflight() -> _Plan:
    _say("hub daemon session-start receipt")
    _say("================================")

    module_path, uvicorn_version = _probe_daemon_interpreter()
    _say(f"preflight  python                {sys.executable} ({sys.version.split()[0]})")
    _say(f"preflight  daemon module         {module_path} (uvicorn {uvicorn_version})")

    head = _git("rev-parse", "HEAD")
    if not head:
        raise _Fail("could not resolve `git rev-parse HEAD`", code=2)
    dirty = bool(_git("status", "--porcelain", "--untracked-files=no"))
    _say(f"preflight  repo HEAD             {head} ({'dirty' if dirty else 'clean'})")

    if not _port_free(_HUB_PORT):
        raise _Fail(
            f"127.0.0.1:{_HUB_PORT} is already in use; refusing to run the receipt "
            "beside an unknown listener. Nothing was started and nothing was killed.",
            code=2,
        )
    _say(f"preflight  hub stub port         127.0.0.1:{_HUB_PORT} free")

    daemon_port = _free_port()
    if daemon_port == _HUB_PORT or not _port_free(daemon_port):
        raise _Fail(f"could not reserve a free port for the daemon ({daemon_port})", code=2)
    _say(f"preflight  daemon port           127.0.0.1:{daemon_port} free")

    if _CONFIG_DIR.exists() and any(_CONFIG_DIR.iterdir()):
        raise _Fail(f"the isolated config dir is not empty: {_CONFIG_DIR}", code=2)
    _say(f"preflight  config dir            {_CONFIG_DIR} (fresh)")

    # The fixture, then the two facts that make it usable: the base the daemon
    # will resolve, and the credential the runner will read.
    _seed()
    _say(
        f"preflight  dummy bearer          seeded in {_CONFIG_DIR / 'auth.db'} "
        f"(sha256={_short_digest(_DUMMY_BEARER)}; generated this run, not a real token)"
    )
    seeded = [record for record in prov.iter_baselines(_CONFIG_DIR) if record.kind == "skill"]
    if len(seeded) != 1 or seeded[0].local_id != _LOCAL_ID:
        raise _Fail(
            "the seeded baseline is not readable as exactly one linked skill "
            f"({[record.local_id for record in seeded]}); the tick would have nothing "
            "to check and the delta would be zero for the wrong reason.",
            code=2,
        )
    _say(
        f"preflight  linked skill          {_LOCAL_ID} -> {_HUB_ID} "
        f"(tenant {_TENANT}, content_state {seeded[0].content_state})"
    )

    resolved_base = configured_radient_base_url(ConfigManager(_CONFIG_DIR))
    if resolved_base != _HUB_BASE:
        raise _Fail(
            f"config resolved the hub base to {resolved_base!r}, not {_HUB_BASE!r}", code=2
        )
    _say(f"preflight  hub base              {_HUB_BASE} (RADIENT_API_BASE_URL)")
    _say("preflight  oauth destination     allowed (RADIENT_ORG_ALLOW_NONCANONICAL_BASE=1)")

    access = _resolved_credential()
    if access is None:
        raise _Fail(
            "the isolated store did not resolve an OAuth access for "
            f"{_HUB_BASE}; the check would answer 'no-credential' with no request. "
            "Refusing rather than working around it.",
            code=2,
        )
    if access.access_token != _DUMMY_BEARER or access.kind != "oauth":
        raise _Fail(
            "the store resolved a credential that is NOT this run's dummy bearer "
            f"(kind={access.kind!r}); refusing to continue.",
            code=2,
        )
    _say(f"preflight  credential resolves    kind=oauth sha256={_short_digest(_DUMMY_BEARER)}")

    startup_max = STARTUP_DELAY_S + STARTUP_JITTER_S
    floor = startup_max + SESSION_START_CHECK_THROTTLE_S + THROTTLE_MARGIN_S
    floor += POST_THROTTLE_RESERVE_S + 25.0
    _say(f"preflight  startup delay          {STARTUP_DELAY_S} s + 0-{STARTUP_JITTER_S} s (runner)")
    _say(f"preflight  session throttle       {SESSION_START_CHECK_THROTTLE_S} s (runner)")
    _say(f"preflight  budget                {BUDGET_S} s; floor {floor:.0f} s")
    if floor > BUDGET_S:
        raise _Fail(
            f"the runner's own constants no longer fit the {BUDGET_S} s budget "
            f"(they need at least {floor:.0f} s). Nothing was started.",
            code=2,
        )
    return _Plan(daemon_port=daemon_port, module_path=module_path, head=head, dirty=dirty)


# -- the run -------------------------------------------------------------------


def _create_session(port: int) -> None:
    response = requests.post(
        f"http://127.0.0.1:{port}/v1/desktop/sessions",
        headers={
            "Authorization": f"Bearer {_DESKTOP_TOKEN}",
            "Content-Type": "application/json",
        },
        json={
            # A fresh request id: the receipt journal answers a REPLAY, and a
            # replay is not a session start.
            "request_id": str(uuid.uuid4()),
            "cwd": str(_WORK_DIR),
        },
        timeout=30.0,
    )
    body = response.text.strip()
    _say(f"create     HTTP {response.status_code} {body[:400]}")
    if response.status_code != 200:
        raise _Fail(
            f"the desktop create route refused (HTTP {response.status_code}); "
            "no session began, so no trigger was attached.",
            code=4,
        )


@dataclass
class _Handles:
    """The two things a teardown must stop, both assigned late.

    A holder rather than locals because the watchdog thread, the ``finally`` and
    the steps that fill them in are three different frames: a local assigned
    after a failure would never be seen by the cleanup that needs it.
    """

    stub: "_HubStub | None" = None
    proc: "subprocess.Popen[bytes] | None" = None


def _run(plan: _Plan) -> int:
    started = time.monotonic()
    _WORK_DIR.mkdir(parents=True, exist_ok=True)
    (_SANDBOX / "tmp").mkdir(parents=True, exist_ok=True)
    log_path = _SANDBOX / "daemon.log"
    handles = _Handles()
    reaper = _Reaper()

    def watchdog() -> None:
        """The hard stop at the budget: report, KILL the daemon tree, exit.

        It kills the daemon's whole process GROUP rather than going through
        ``reaper``: the case this exists for is the one where the main thread is
        stuck (inside ``proc.wait``, a socket read, a lock), and a cooperative
        teardown that waits for that thread would be the hang this replaces. The
        group is safe to signal because the daemon was started with
        ``start_new_session=True`` -- its pid is its pgid, and nothing else is in
        it. The stub is a thread of THIS process, so ``os._exit`` closes it.
        """

        time.sleep(max(0.0, BUDGET_S - (time.monotonic() - _T0)))
        _say("")
        _say(f"FAIL       {BUDGET_S:.0f} s budget exhausted; killing the daemon tree")
        _print_log(handles.stub.snapshot() if handles.stub is not None else [], "request log")
        _say(f"daemon log tail ({log_path}):")
        _say(_tail(log_path))
        proc = handles.proc
        if proc is not None and proc.poll() is None:
            try:
                os.killpg(proc.pid, signal.SIGKILL)
                _say(f"reap       pid group {proc.pid} SIGKILL")
            except (ProcessLookupError, PermissionError):
                _kill_quietly(proc.pid, signal.SIGKILL)
                _say(f"reap       pid {proc.pid} SIGKILL")
        os._exit(3)

    threading.Thread(target=watchdog, name="budget-watchdog", daemon=True).start()

    outcome = 0
    try:
        _run_steps(plan, log_path, started, handles)
    except _Fail as failure:
        _say("")
        _say(f"FAIL       {failure}")
        outcome = failure.code
    finally:
        clean = reaper.reap(handles.proc, handles.stub, [plan.daemon_port, _HUB_PORT])
    # A port still held after a PASS is not a pass: the next run cannot bind,
    # and something this invocation started is still running.
    return outcome if clean else 4


def _run_steps(plan: _Plan, log_path: Path, started: float, handles: _Handles) -> None:
    """The measured run. Raises ``_Fail`` on the first thing that is not proven."""

    stub = _open_stub()
    handles.stub = stub
    _say(f"stub       listening on 127.0.0.1:{_HUB_PORT}")
    _say(f"stub       answers GET /v1/skills/{_HUB_ID} with the fixture manifest")

    proc = _open_daemon(plan.daemon_port, log_path)
    handles.proc = proc
    _say(f"daemon     pid {proc.pid}, 127.0.0.1:{plan.daemon_port}, log {log_path}")

    health_until = min(time.monotonic() + 45.0, _T0 + 120.0)
    health_s = _wait_healthy(proc, plan.daemon_port, log_path, health_until)
    _say(f"daemon     /health 200 after {health_s:.1f} s")
    _say("daemon     the lifespan has attached the runner; waiting for its own first tick")

    # -- step 4: the startup tick settles -----------------------------
    #
    # The tick is at STARTUP_DELAY_S + jitter AFTER the runner was built, so
    # this window is sized from the module's own constants rather than a guess.
    # The cap is also bounded by what the rest of the run needs: a cap that ate
    # the throttle would only move the failure to the watchdog.
    startup_max = STARTUP_DELAY_S + STARTUP_JITTER_S
    poll_started = time.monotonic()
    startup_cap = min(
        STARTUP_CAP_S,
        (_T0 + BUDGET_S)
        - poll_started
        - (SESSION_START_CHECK_THROTTLE_S + THROTTLE_MARGIN_S + POST_THROTTLE_RESERVE_S),
    )
    if startup_cap < startup_max + 2.0:
        raise _Fail(
            f"only {startup_cap:.0f} s of the {BUDGET_S:.0f} s budget is left to catch the "
            f"startup tick, which the runner fires {startup_max:.0f} s after boot; the "
            "daemon booted too slowly to finish this receipt inside its budget.",
            code=3,
        )
    _say(f"startup    waiting up to {startup_cap:.0f} s for a request, then {QUIET_S:.0f} s quiet")
    quiet_since: float | None = None
    last_count = 0
    while True:
        now = time.monotonic()
        count = stub.count()
        if count != last_count:
            last_count = count
            quiet_since = now
        if count and quiet_since is not None and now - quiet_since >= QUIET_S:
            break
        if now - poll_started >= startup_cap:
            break
        time.sleep(0.2)

    _print_log(stub.snapshot(), "startup    request log so far")
    before = stub.count()
    if before == 0:
        raise _Fail(
            "the daemon's startup tick produced NO hub request, so there is no "
            "baseline to measure a delta against. The log above is the whole "
            "evidence; the daemon log tail says why the check did not reach the "
            f"stub.\ndaemon log tail ({log_path}):\n{_tail(log_path)}",
            code=4,
        )
    startup_calls = [item for item in stub.snapshot() if item.path == f"/v1/skills/{_HUB_ID}"]
    if not startup_calls:
        raise _Fail(
            "the startup window's request(s) did not include the skill read; "
            "the seeded link was not what the runner checked.",
            code=4,
        )
    if not any(item.status == 200 for item in startup_calls):
        raise _Fail(
            "the startup skill read was answered "
            f"{startup_calls[0].status}, not 200: the stub and the client disagree "
            "about the endpoint, so nothing below would be measuring the trigger.",
            code=4,
        )
    checked_before = _last_checked_at()
    _say(
        f"startup    settled with {before} request(s); the skill check is among them "
        f"(last_checked_at={checked_before or '<unset>'})"
    )

    # -- step 5: wait out the throttle, measured from the last request --
    #
    # The runner claims the window at tick START, which is a moment BEFORE the
    # request the stub logged -- so measuring from the stub's timestamp is the
    # conservative side of the comparison.
    last_epoch = stub.last_epoch()
    if last_epoch is None:  # `before >= 1` should make this unreachable
        raise _Fail("the stub logged no request although its count was non-zero.", code=4)
    ready_at = last_epoch + SESSION_START_CHECK_THROTTLE_S + THROTTLE_MARGIN_S
    wait_s = max(0.0, ready_at - time.time())
    ceiling = (_T0 + BUDGET_S) - POST_THROTTLE_RESERVE_S
    if time.time() + wait_s > ceiling:
        raise _Fail(
            f"the {SESSION_START_CHECK_THROTTLE_S:.0f} s throttle cannot be waited out "
            f"inside the {BUDGET_S:.0f} s budget (needs {wait_s:.0f} s more).",
            code=3,
        )
    _say(
        f"throttle   last hub request {_stamp(last_epoch)}; waiting {wait_s:.1f} s "
        f"({SESSION_START_CHECK_THROTTLE_S:.0f} s throttle + {THROTTLE_MARGIN_S:.0f} s margin)"
    )
    _wait_until(ready_at)

    # -- step 6: ONE real session over HTTP ---------------------------
    created_at = time.monotonic()
    _create_session(plan.daemon_port)
    _say(f"create     one session created in {time.monotonic() - created_at:.1f} s")

    after_deadline = min(time.monotonic() + POST_CREATE_WAIT_S, (_T0 + BUDGET_S) - 8.0)
    while stub.count() <= before and time.monotonic() < after_deadline:
        time.sleep(0.2)
    after = stub.count()
    _print_log(stub.snapshot(), "session    request log")

    delta = after - before
    _say(f"delta      before={before} after={after} delta={delta}")
    if delta != 1:
        raise _Fail(
            f"expected exactly 1 request from the session start, measured {delta}. "
            f"before={before} after={after}; the full log is above.",
            code=4,
        )
    trigger = stub.snapshot()[before]
    if (trigger.method, trigger.path) != ("GET", f"/v1/skills/{_HUB_ID}"):
        raise _Fail(
            f"the one new request was {trigger.method} {trigger.path}, not the "
            f"skill check GET /v1/skills/{_HUB_ID}.",
            code=4,
        )
    if trigger.auth != "dummy":
        raise _Fail(
            "the one new request did not carry this run's dummy bearer "
            f"(auth={trigger.auth}); the check did not use the isolated store.",
            code=4,
        )
    _say(
        f"check      #{trigger.seq} {_stamp(trigger.epoch)} {trigger.method} "
        f"{trigger.path} -> {trigger.status} auth=dummy"
    )

    # The store write happens in the check's worker; give it a moment to land
    # before reading, without making the count depend on it.
    settle = time.monotonic() + 5.0
    while time.monotonic() < settle and _last_checked_at() == checked_before:
        time.sleep(0.1)
    checked_after = _last_checked_at()
    _say(f"store      last_checked_at before={checked_before or '<unset>'}")
    _say(f"store      last_checked_at after ={checked_after or '<unset>'}")
    if checked_after is None:
        raise _Fail(
            "the status store has no last_checked_at for the seeded link after the "
            "check; the check did not record.",
            code=4,
        )

    _say("")
    _say(
        "PASS       the daemon's own lifespan attached the runner, and ONE session "
        "start asked the hub exactly once"
    )
    _say(f"PASS       provenance: HEAD {plan.head}, module {plan.module_path}")
    _say(f"PASS       total {time.monotonic() - started:.1f} s of the {BUDGET_S:.0f} s budget")


def main() -> int:
    if sys.argv[1:]:
        _say("usage: hub_daemon_check_receipt.py   (this script takes no arguments)")
        return 2
    try:
        plan = _preflight()
    except _Fail as failure:
        _say(f"FAIL       {failure}")
        return failure.code
    try:
        return _run(plan)
    except _Fail as failure:
        _say("")
        _say(f"FAIL       {failure}")
        return failure.code


if __name__ == "__main__":
    sys.exit(main())
