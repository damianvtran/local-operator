#!/usr/bin/env python3
"""Prove the daemon's own session-start hub check with ONE request of evidence.

Run it with the repository's interpreter, from anywhere:

    .venv/bin/python scripts/hub_daemon_check_receipt.py

No arguments, no flags, no environment to set: every value this needs is derived
below or built into a fresh temporary HOME. It exits 0 only when the delta it
measures is exactly the one request the trigger is supposed to cause AND that
request is ordered strictly before the earliest tick the runner could fire on
its own.

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

CREATE FIRST, THEN PROVE THE ORDERING. The runner's first automatic tick is its
startup pass, ``STARTUP_DELAY_S`` (45 s) + ``random.uniform(0, STARTUP_JITTER_S)``
after ``run_forever`` starts; the next is one jittered interval after that.
Waiting for the startup pass and then creating (the previous attempt) spent the
session-start throttle and still could not finish inside its budget. So this
script creates ONE session as soon as ``/health`` answers -- seconds after the
daemon exists, tens of seconds before the earliest automatic tick -- and
attributes the check BY ORDERING:

* immediately before the create it asserts the stub has seen ZERO requests and
the status store has NO ``last_checked_at``. Either one non-empty means a check
already ran and the create cannot be shown to be the cause: the run is VOID
(exit 5), printed with its numbers, and nothing is claimed;
* the create's own request timestamp and the one hub request it causes must both
be STRICTLY before ``spawn + STARTUP_DELAY_S`` -- the earliest instant any
automatic tick could fire, derived from the runner's constants below. A create
that is not strictly earlier is ``INCONCLUSIVE`` (exit 6), never a pass: the tick
could have raced it, and a near-miss is not a pass.

THE EVIDENCE IS A DELTA OF ONE FROM ZERO. ``last_checked_at`` alone cannot carry
the claim: an automatic tick writes it too. The baseline assertion above proves
nothing had checked before the create, so the stub's count is 0; after the create
the delta must be exactly 1, and that one request must be the skill read,
carrying this run's dummy bearer. A zero or two fails with the whole log printed,
so a miss is diagnosable rather than ambiguous. The ordering check is what turns
"a request appeared" into "the create caused it".

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

TIMING. The whole run is bounded by ``BUDGET_S`` (90 s), and a watchdog SIGKILLs
the daemon's process group and exits 3 at that mark rather than hanging. The
budget is spent like this:

    daemon boot to /health          ~5-25 s (measured, printed)
    baseline assert + create        < 1 s
    the create's one hub request    < 2 s
    status-store settle             <= 5 s
    reap                            ~2-5 s

The binding constraint is the derivation, not the budget: the create must land
before ``STARTUP_DELAY_S`` (45 s) after spawn, so a daemon that took that long to
boot is INCONCLUSIVE rather than passed. The constants are READ FROM THE
MODULES and printed with their arithmetic -- a code change that moves them fails
here with numbers rather than as a timeout.

EXIT CODES. 0 = the receipt passed; 2 = a preflight refused (nothing was
started); 3 = the 90 s budget was exhausted; 4 = the receipt did not hold (delta
wrong, create failed, or the wrong request); 5 = VOID (the baseline was not empty
before the create, so attribution is impossible); 6 = INCONCLUSIVE (the create
was not strictly before the derived earliest automatic tick). Every non-zero
path prints the full request log and, for daemon trouble, the tail of the
daemon's own log.

WHAT THIS IS NOT. It is not a test (no CI, no pytest); it is a receipt for a
person on a machine whose resource gate allows one daemon boot and ~25 s of its
life. It asserts nothing about whether an available item should be APPLIED --
``check_now`` applies nothing by contract, and the startup tick is inside its
first-run grace.
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
    INTERVAL_JITTER,
    STARTUP_DELAY_S,
    STARTUP_JITTER_S,
)
from local_operator.hub_sync.settings import (  # noqa: E402
    DEFAULT_CHECK_INTERVAL_MIN,
    HubSyncSettings,
)
from local_operator.hub_sync.skill_document import SkillDocument, SkillFile  # noqa: E402
from local_operator.providers.auth_store import AuthStore  # noqa: E402
from local_operator.providers.radient_credentials import (  # noqa: E402
    configured_radient_base_url,
    resolve_radient_oauth_access,
)

#: The receipt's own budget: the lease bound is 90 s. A hard wall, enforced by
#: a watchdog that SIGKILLs the daemon's process group and exits 3.
BUDGET_S = 90.0
#: Cap on the daemon boot to /health. The measured shape is 5-25 s; a boot
#: slower than ~45 s cannot pass attribution anyway (the derived deadline).
HEALTH_CAP_S = 40.0
#: Absolute ceiling for /health, measured from ``_T0``, so a slow preflight
#: cannot eat the create's window silently.
HEALTH_LATEST_S = 50.0
#: How long the create's own check may take to reach the stub. Measured <2 s;
#: this is slack for a loaded machine, not a schedule.
POST_CREATE_WAIT_S = 15.0
#: The status store is written by the check's worker after the request, so the
#: supporting ``last_checked_at`` read gets this long -- never a gate on the count.
STORE_SETTLE_S = 5.0
#: Room after the wait for the request: the assertions and the reaper's own
#: worst case (12 s SIGTERM + 5 s SIGKILL + 5 s port settle), with slack.
POST_CREATE_RESERVE_S = 25.0
#: Exit 5: the baseline was not empty, so no attribution is possible.
EXIT_VOID = 5
#: Exit 6: the create was not strictly before the earliest automatic tick.
EXIT_INCONCLUSIVE = 6

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
    #: Read at the same moment as ``epoch`` but on the monotonic clock: the
    #: attribution comparison uses this one, so a wall-clock step cannot turn a
    #: late request into a pass (or an early one into a false INCONCLUSIVE).
    mono: float
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
            self._seen.append(
                _Seen(
                    len(self._seen) + 1,
                    time.time(),
                    time.monotonic(),
                    method,
                    path,
                    status,
                    auth,
                )
            )

    def snapshot(self) -> list[_Seen]:
        with self._lock:
            return list(self._seen)

    def count(self) -> int:
        with self._lock:
            return len(self._seen)

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
    #: The interval the runner will read FRESH on every tick, from the isolated
    #: config (unset there, so ``DEFAULT_CHECK_INTERVAL_MIN``).
    interval_min: int
    #: A printed description of where that number came from.
    interval_source: str
    #: ``STARTUP_DELAY_S + min jitter``: the earliest automatic tick, seconds
    #: after the runner's loop starts (never before the daemon's spawn).
    earliest_tick_s: float
    #: Earliest SECOND automatic tick: the first plus a jittered interval.
    periodic_tick_s: float


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

    config = ConfigManager(_CONFIG_DIR)
    resolved_base = configured_radient_base_url(config)
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

    # -- the deadline, DERIVED from production code, never from a measurement --
    #
    # ``run_forever`` sleeps ``STARTUP_DELAY_S + uniform(0, STARTUP_JITTER_S)``
    # before its first (startup) tick, then one jittered interval before the
    # next. Jitter is ADDITIVE, so the earliest instant either can fire is the
    # floor of each range -- there is nothing to subtract from STARTUP_DELAY_S.
    # The create is raced against the FIRST of those two instants, because it is
    # the earlier one; the periodic tick is printed to show how much room the
    # run leaves after it. Both numbers are computed from the modules imported
    # above, so a code change moves the receipt's deadline with it.
    raw_interval = config.get_nested_value(("hub", "check_interval_min"), None)
    hub_settings = HubSyncSettings.from_config(config)
    interval_min = hub_settings.interval_min
    interval_source = (
        f"hub.check_interval_min={raw_interval!r} (clamped)"
        if raw_interval is not None
        else (
            "hub.check_interval_min unset -> DEFAULT_CHECK_INTERVAL_MIN="
            f"{DEFAULT_CHECK_INTERVAL_MIN}"
        )
    )
    interval_s = float(interval_min * 60)
    startup_floor = STARTUP_DELAY_S
    startup_ceiling = STARTUP_DELAY_S + STARTUP_JITTER_S
    interval_floor = interval_s * (1.0 - INTERVAL_JITTER)
    interval_ceiling = interval_s * (1.0 + INTERVAL_JITTER)
    periodic_floor = startup_floor + interval_floor
    _say("preflight  derivation            earliest automatic tick, from runner/settings:")
    _say(
        f"derive     STARTUP_DELAY_S       {STARTUP_DELAY_S:.1f} s "
        "(local_operator.hub_sync.runner)"
    )
    _say(
        f"derive     STARTUP_JITTER_S      {STARTUP_JITTER_S:.1f} s -> first tick = "
        f"{STARTUP_DELAY_S:.1f} + U(0, {STARTUP_JITTER_S:.1f}) = "
        f"[{startup_floor:.1f}, {startup_ceiling:.1f}] s after run_forever starts"
    )
    _say(
        f"derive     interval              {interval_min} min = {interval_s:.0f} s "
        f"({interval_source}; local_operator.hub_sync.settings)"
    )
    _say(
        f"derive     INTERVAL_JITTER       {INTERVAL_JITTER:.2f} -> periodic delay = "
        f"{interval_s:.0f} * U({1 - INTERVAL_JITTER:.2f}, {1 + INTERVAL_JITTER:.2f}) = "
        f"[{interval_floor:.0f}, {interval_ceiling:.0f}] s"
    )
    _say(
        f"derive     earliest tick        spawn + {startup_floor:.1f} s "
        "(STARTUP_DELAY_S + min jitter 0; jitter only ADDS)"
    )
    _say(
        f"derive     earliest periodic    spawn + {startup_floor:.1f} + {interval_floor:.0f} = "
        f"spawn + {periodic_floor:.0f} s (zero-length first tick, both jitters at their floor)"
    )
    _say(
        f"derive     attribution deadline spawn + {startup_floor:.1f} s -- the create AND "
        "its hub request must both be strictly earlier"
    )
    if startup_floor <= 0.0 or not 0.0 <= INTERVAL_JITTER < 1.0 or interval_s <= 0.0:
        raise _Fail(
            "the derived deadline is not meaningful (STARTUP_DELAY_S="
            f"{STARTUP_DELAY_S!r}, INTERVAL_JITTER={INTERVAL_JITTER!r}, "
            f"interval={interval_min} min). Nothing was started.",
            code=2,
        )

    reserve = HEALTH_CAP_S + POST_CREATE_WAIT_S + STORE_SETTLE_S + POST_CREATE_RESERVE_S
    _say(
        f"preflight  budget                {BUDGET_S:.0f} s; health cap {HEALTH_CAP_S:.0f} s + "
        f"create wait {POST_CREATE_WAIT_S:.0f} s + settle {STORE_SETTLE_S:.0f} s + teardown "
        f"reserve {POST_CREATE_RESERVE_S:.0f} s = {reserve:.0f} s"
    )
    if reserve > BUDGET_S:
        raise _Fail(
            f"the run's own windows no longer fit the {BUDGET_S:.0f} s budget "
            f"(they need {reserve:.0f} s). Nothing was started.",
            code=2,
        )
    return _Plan(
        daemon_port=daemon_port,
        module_path=module_path,
        head=head,
        dirty=dirty,
        interval_min=interval_min,
        interval_source=interval_source,
        earliest_tick_s=startup_floor,
        periodic_tick_s=periodic_floor,
    )


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


def _arm_watchdog(handles: _Handles, log_path: Path) -> None:
    """Arm the whole-run hard stop: at ``BUDGET_S``, SIGKILL the group and exit 3.

    Armed in ``main`` BEFORE the preflight, because the 90 s figure is the run's
    wall clock, not just the measured section: an interpreter probe that hangs
    must not escape it. It kills the daemon's whole process GROUP rather than
    going through ``reaper`` -- the case this exists for is the one where the
    main thread is stuck (inside ``proc.wait``, a socket read, a lock), and a
    cooperative teardown that waits for that thread would be the hang this
    replaces. The group is safe to signal because the daemon is started with
    ``start_new_session=True``: its pid is its pgid, and nothing else is in it.
    The stub is a thread of THIS process, so ``os._exit`` closes it.
    """

    def watchdog() -> None:
        time.sleep(max(0.0, BUDGET_S - (time.monotonic() - _T0)))
        _say("")
        _say(
            f"FAIL       the {BUDGET_S:.0f} s lease bound is exhausted; "
            "SIGKILLing the daemon process group"
        )
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


def _run(plan: _Plan, handles: _Handles, log_path: Path) -> int:
    started = time.monotonic()
    _WORK_DIR.mkdir(parents=True, exist_ok=True)
    (_SANDBOX / "tmp").mkdir(parents=True, exist_ok=True)
    reaper = _Reaper()

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
    # and something this invocation started is still running. A non-zero
    # attribution code (VOID 5 / INCONCLUSIVE 6 / receipt 4) is preserved --
    # the leak is reported, but it must not disguise why the run failed.
    if not clean:
        _say("FAIL       a port this run started is still held after teardown")
    return outcome if outcome or clean else 4


def _run_steps(plan: _Plan, log_path: Path, started: float, handles: _Handles) -> None:
    """The measured run. Raises ``_Fail`` on the first thing that is not proven."""

    stub = _open_stub()
    handles.stub = stub
    _say(f"stub       listening on 127.0.0.1:{_HUB_PORT}")
    _say(f"stub       answers GET /v1/skills/{_HUB_ID} with the fixture manifest")

    # The clocks BEFORE the spawn. The child cannot exist a millisecond earlier,
    # so ``spawn_mono`` is a sound (slightly early) lower bound on the instant
    # ``run_forever`` starts -- and therefore on its earliest automatic tick.
    # Anchoring on the SPAWN rather than on /health is what makes the deadline
    # conservative: the runner starts inside the lifespan, BEFORE uvicorn answers
    # /health, so a health-anchored deadline would be too late by the boot time.
    # The monotonic pair is what the comparisons use; the wall pair is only for
    # the human-readable stamps, so a clock step cannot decide the verdict.
    spawn_epoch = time.time()
    spawn_mono = time.monotonic()
    proc = _open_daemon(plan.daemon_port, log_path)
    handles.proc = proc
    _say(f"daemon     pid {proc.pid}, 127.0.0.1:{plan.daemon_port}, log {log_path}")

    health_until = min(time.monotonic() + HEALTH_CAP_S, _T0 + HEALTH_LATEST_S)
    health_s = _wait_healthy(proc, plan.daemon_port, log_path, health_until)
    _say(
        f"daemon     /health 200 after {health_s:.1f} s "
        f"(spawn + {time.monotonic() - spawn_mono:.2f} s)"
    )
    _say(
        "daemon     the lifespan has attached the runner; creating NOW, before its "
        f"earliest automatic tick at spawn + {plan.earliest_tick_s:.1f} s"
    )

    # -- create-first: the empty baseline is asserted IMMEDIATELY before it -----
    #
    # Anything the stub has already seen, or a status write already on disk, could
    # have come from the runner's own startup tick. Then the create cannot be
    # shown to be its cause, and no later arithmetic repairs that: VOID, with its
    # numbers printed, so the run is diagnosable and never mistaken for a pass.
    baseline_count = stub.count()
    baseline_checked = _last_checked_at()
    _say(
        f"baseline   requests={baseline_count} "
        f"last_checked_at={baseline_checked or '<absent>'} (asserted empty before the create)"
    )
    if baseline_count != 0 or baseline_checked is not None:
        _print_log(stub.snapshot(), "baseline   request log")
        raise _Fail(
            "VOID: the baseline was not empty before the create "
            f"(requests={baseline_count}, last_checked_at={baseline_checked or '<absent>'}). "
            "A check already ran, so the create cannot be shown to be the cause; "
            "the full log is above.",
            code=EXIT_VOID,
        )

    # -- ONE real session over HTTP, as early as the daemon allows --------------
    #
    # ``create_mono`` is taken BEFORE the POST so it is early rather than late:
    # if even that instant is past the deadline the run cannot be attributed.
    # The wire send (and the check it triggers) is covered by the request's own
    # monotonic stamp below, against the same deadline.
    deadline_mono = spawn_mono + plan.earliest_tick_s
    deadline_wall = spawn_epoch + plan.earliest_tick_s
    create_mono = time.monotonic()
    create_epoch = time.time()
    _say(
        f"create     sending at spawn + {create_mono - spawn_mono:.2f} s; "
        f"attribution deadline spawn + {plan.earliest_tick_s:.1f} s"
    )
    created_at = time.monotonic()
    _create_session(plan.daemon_port)
    _say(f"create     one session created in {time.monotonic() - created_at:.1f} s")

    create_margin = deadline_mono - create_mono
    _say(
        f"order      create {_stamp(create_epoch)} = spawn + "
        f"{create_mono - spawn_mono:.2f} s ({create_margin:.2f} s before the deadline)"
    )
    if not create_mono < deadline_mono:
        _say("INCONCLUSIVE — the periodic tick could have raced the create")
        raise _Fail(
            "the create was sent after the derived earliest automatic tick "
            f"(create spawn + {create_mono - spawn_mono:.2f} s, deadline spawn + "
            f"{plan.earliest_tick_s:.1f} s). A near-miss is not a pass.",
            code=EXIT_INCONCLUSIVE,
        )

    after_deadline = min(
        time.monotonic() + POST_CREATE_WAIT_S,
        _T0 + BUDGET_S - POST_CREATE_RESERVE_S,
    )
    if after_deadline <= time.monotonic():
        raise _Fail(
            f"no budget is left to wait for the create's request inside the "
            f"{BUDGET_S:.0f} s wall (health took {health_s:.1f} s).",
            code=3,
        )
    while stub.count() <= baseline_count and time.monotonic() < after_deadline:
        time.sleep(0.2)
    after = stub.count()
    _print_log(stub.snapshot(), "session    request log")

    delta = after - baseline_count
    _say(f"delta      before={baseline_count} after={after} delta={delta}")
    if delta != 1:
        raise _Fail(
            f"expected exactly 1 request from the session start, measured {delta}. "
            f"before={baseline_count} after={after}; the full log is above.",
            code=4,
        )
    trigger = stub.snapshot()[baseline_count]
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

    # -- attribution BY ORDERING, against the same derived deadline ------------
    request_margin = deadline_mono - trigger.mono
    _say(
        f"order      request #{trigger.seq} {_stamp(trigger.epoch)} = spawn + "
        f"{trigger.mono - spawn_mono:.2f} s ({request_margin:.2f} s before the deadline)"
    )
    _say(
        f"order      deadline {_stamp(deadline_wall)} = spawn + {plan.earliest_tick_s:.1f} s "
        "(STARTUP_DELAY_S + 0 jitter); next automatic tick no earlier than spawn + "
        f"{plan.periodic_tick_s:.0f} s"
    )
    if not trigger.mono < deadline_mono:
        _say("INCONCLUSIVE — the periodic tick could have raced the create")
        raise _Fail(
            "the create's hub request was logged after the derived earliest automatic "
            f"tick (request spawn + {trigger.mono - spawn_mono:.2f} s, deadline "
            f"spawn + {plan.earliest_tick_s:.1f} s). A near-miss is not a pass.",
            code=EXIT_INCONCLUSIVE,
        )

    _say(
        f"check      #{trigger.seq} {_stamp(trigger.epoch)} {trigger.method} "
        f"{trigger.path} -> {trigger.status} auth=dummy"
    )

    # The store write happens in the check's worker; give it a moment to land
    # before reading, without making the count depend on it.
    settle = time.monotonic() + STORE_SETTLE_S
    while time.monotonic() < settle and _last_checked_at() == baseline_checked:
        time.sleep(0.1)
    checked_after = _last_checked_at()
    _say(f"store      last_checked_at before={baseline_checked or '<unset>'}")
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
        "start asked the hub exactly once, strictly before any automatic tick"
    )
    _say(
        f"PASS       ordering: create spawn + {create_mono - spawn_mono:.2f} s, request "
        f"spawn + {trigger.mono - spawn_mono:.2f} s, deadline spawn + "
        f"{plan.earliest_tick_s:.1f} s (next automatic tick >= spawn + "
        f"{plan.periodic_tick_s:.0f} s)"
    )
    _say(
        "PASS       derivation: STARTUP_DELAY_S="
        f"{STARTUP_DELAY_S:.1f} + U(0, STARTUP_JITTER_S={STARTUP_JITTER_S:.1f}) s; "
        f"interval {plan.interval_min} min ({plan.interval_source}); "
        f"INTERVAL_JITTER={INTERVAL_JITTER:.2f}"
    )
    _say(f"PASS       provenance: HEAD {plan.head}, module {plan.module_path}")
    _say(
        f"PASS       total {time.monotonic() - _T0:.1f} s of the {BUDGET_S:.0f} s budget "
        f"(measured section {time.monotonic() - started:.1f} s)"
    )


def main() -> int:
    if sys.argv[1:]:
        _say("usage: hub_daemon_check_receipt.py   (this script takes no arguments)")
        return 2
    log_path = _SANDBOX / "daemon.log"
    handles = _Handles()
    # BEFORE the preflight: the 90 s bound is the whole run's wall clock, and a
    # hanging interpreter probe must not escape it. Nothing is spawned yet, so
    # the watchdog's kill branch simply finds no process to signal.
    _arm_watchdog(handles, log_path)
    try:
        plan = _preflight()
    except _Fail as failure:
        _say(f"FAIL       {failure}")
        return failure.code
    try:
        return _run(plan, handles, log_path)
    except _Fail as failure:
        _say("")
        _say(f"FAIL       {failure}")
        return failure.code


if __name__ == "__main__":
    sys.exit(main())
