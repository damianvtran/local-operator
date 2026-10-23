"""Count the hub checks ONE desktop session create causes, on the real path.

Usage: python scripts/hub_session_check_receipt.py

WHAT THIS PROVES. A merged test asserts the session-create route calls
``HubSyncRunner.note_session_start`` exactly once, and other tests assert the
trigger reaches the runner's throttle and check machinery. None of them watch
the thing the trigger exists for: the hub actually being ASKED, over HTTP, for
the item the operator has linked. This script stands the real pieces up -- a
real ``HubSyncRunner``, the real desktop route behind a real HTTP server, and a
local hub stub that counts requests -- creates ONE session over HTTP, and
prints the request counts and the store's ``last_checked_at`` either side of it.

WHY THE REQUEST COUNT AND NOT THE TIMESTAMP. The runner fires an initial check
at daemon start, so ``last_checked_at`` moving after a create proves nothing by
itself: the startup tick already wrote it, and both writes can land in the same
second. The only unambiguous evidence is a request count taken AFTER the
startup check settled and BEFORE the create, then again after the create. The
delta must be exactly one. The periodic timer loop is deliberately NOT started
here (its first interval is 15 minutes), so nothing but the create's own trigger
can add a request inside that window.

THE THROTTLE. A session start is coalesced when any check started in the last
two minutes, and the startup tick counts as one. The runner takes an injectable
clock for exactly this reason (the merged tests use it); this script advances
that clock past the window before the create, which is what a conversation
opened more than two minutes after boot would see.

ISOLATION. ``scripts.probe_isolation`` re-homes HOME and the config directory
and drops inherited multiplexer identifiers before any application import, so
this never touches the operator's real configuration.

NOT RUN IN CI. It is a receipt for a person, on a machine whose resource gate
allows execution; see the PR body.
"""

from __future__ import annotations

import json
import os
import sys
import threading
import time
import uuid
from collections.abc import AsyncIterator
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any

# Clear every multiplexer identifier BEFORE any application import (see
# scripts/probe_isolation.py): a headless app that inherits a live CMUX id has
# renamed the operator's real cmux workspaces before.
for _key in tuple(os.environ):
    if _key.startswith("CMUX_"):
        os.environ.pop(_key)
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import scripts.probe_isolation  # noqa: E402, F401

# The desktop plane opens for a bearer in the environment. This one is for this
# process's own loopback server and is not a credential to anything else.
os.environ["LOCAL_OPERATOR_DESKTOP_TOKEN"] = "local-receipt-harness"

HUB_ID = "receipt-agent"
INSTRUCTIONS = "## Rules\nBe brief and cite the record you read.\n"
DESCRIPTION = "receipt harness fixture"


class _Clock:
    """The runner's injectable monotonic clock, moved by the script."""

    def __init__(self) -> None:
        self.now = 1_000.0

    def __call__(self) -> float:
        return self.now


class _HubStub:
    """A local hub that answers the one GET a check makes, and counts requests.

    ONE REQUEST PER CHECK, on purpose: a public agent link's check is the
    revision-bearing detail read (``check_hub_agents``), and because the reply
    carries ``instructions`` the legacy zip fallback is never reached. The
    content equals the local agent's, so the check is the cheap unchanged round
    trip the default interval was lowered for -- not a merge.
    """

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._requests: list[tuple[str, str]] = []
        self._server = ThreadingHTTPServer(("127.0.0.1", 0), self._handler())
        self._thread: threading.Thread | None = None
        self.base_url = f"http://127.0.0.1:{int(self._server.server_address[1])}/v1"

    def _handler(self) -> type[BaseHTTPRequestHandler]:
        stub = self

        class Handler(BaseHTTPRequestHandler):
            def do_GET(self) -> None:  # noqa: N802 - http.server's spelling
                stub.record("GET", self.path)
                if self.path == f"/v1/agents/{HUB_ID}":
                    body = json.dumps(
                        {
                            "result": {
                                "instructions": INSTRUCTIONS,
                                "description": DESCRIPTION,
                                "revision": '"receipt-rev-1"',
                                "sync_contract": {"revision_cas": 1},
                            }
                        }
                    ).encode("utf-8")
                    self.send_response(200)
                    self.send_header("Content-Type", "application/json")
                    self.send_header("Content-Length", str(len(body)))
                    self.end_headers()
                    self.wfile.write(body)
                    return
                self.send_error(404, "not in this stub")

            def log_message(self, format: str, *args: Any) -> None:
                """Silence the per-request log; the printed counts are the evidence."""

        return Handler

    def record(self, method: str, path: str) -> None:
        with self._lock:
            self._requests.append((method, path))

    def requests(self) -> list[tuple[str, str]]:
        with self._lock:
            return list(self._requests)

    def count(self) -> int:
        with self._lock:
            return len(self._requests)

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


def _last_checked_at(config_dir: Path, agent_id: str) -> str | None:
    """The status store's ``last_checked_at`` for the linked agent, if written."""

    from local_operator.hub_sync import store as st

    try:
        document = st.StatusStore(config_dir).load()
    except Exception:  # noqa: BLE001 - a missing store is reported as absent
        return None
    items = document.get("items") if isinstance(document, dict) else None
    row = items.get(f"agent:{agent_id}") if isinstance(items, dict) else None
    value = row.get("last_checked_at") if isinstance(row, dict) else None
    return str(value) if value else None


def _wait_for_server(server: Any, timeout: float = 30.0) -> None:
    deadline = time.monotonic() + timeout
    while not server.started:
        if time.monotonic() > deadline:
            raise SystemExit("FAIL: the local desktop API did not start within 30 s")
        time.sleep(0.02)


def _run_receipt(
    *,
    hub: _HubStub,
    api_root: str,
    config_dir: Path,
    cwd: Path,
    clock: _Clock,
    agent_id: str,
) -> int:
    import requests

    from local_operator.hub_sync.runner import SESSION_START_CHECK_THROTTLE_S

    token = os.environ["LOCAL_OPERATOR_DESKTOP_TOKEN"]

    startup_requests = hub.count()
    startup_checked = _last_checked_at(config_dir, agent_id)
    print()
    print("1. startup check (the daemon's first tick, run directly at boot)")
    print(f"   hub requests:    {startup_requests}")
    print(f"   last_checked_at: {startup_checked or '(absent)'}")
    if startup_requests < 1:
        print()
        print("FAIL: the startup check made no request to the hub stub.")
        print("      The linked agent never reached the check, so a create delta proves nothing.")
        return 1

    before = hub.count()
    before_checked = _last_checked_at(config_dir, agent_id)
    print()
    print("2. baseline immediately before the create")
    print(f"   hub requests:    {before}")
    print(f"   last_checked_at: {before_checked or '(absent)'}")

    # The startup tick claimed the session-start window. Move the injected
    # clock past it, as a conversation opened later than two minutes after
    # boot would be, so the create's trigger is allowed to run.
    clock.now += SESSION_START_CHECK_THROTTLE_S + 1.0

    response = requests.post(
        f"{api_root}/v1/desktop/sessions",
        json={"request_id": str(uuid.uuid4()), "cwd": str(cwd)},
        headers={"Authorization": f"Bearer {token}"},
        timeout=60.0,
    )
    if response.status_code != 200:
        print()
        print(f"FAIL: POST /v1/desktop/sessions answered {response.status_code}")
        print(f"      {response.text[:500]}")
        return 1
    session_id = str(response.json()["result"]["session_id"])
    print()
    print(f"3. create: POST /v1/desktop/sessions -> 200 (session {session_id})")

    deadline = time.monotonic() + 30.0
    while hub.count() <= before and time.monotonic() < deadline:
        time.sleep(0.02)
    # Settle: a duplicate trigger fires immediately behind the first one, so
    # one further pass after the count moves is enough to catch it.
    time.sleep(1.0)

    after = hub.count()
    after_checked = _last_checked_at(config_dir, agent_id)
    delta = after - before
    print()
    print("4. after the create")
    print(f"   hub requests:    {after}")
    print(f"   delta:           {delta}  (before {before}, after {after})")
    print(f"   last_checked_at: {after_checked or '(absent)'}")
    print(
        f"   throttle clock:  +{SESSION_START_CHECK_THROTTLE_S + 1.0:.0f}s before the create"
    )
    print()
    print("   hub requests seen, in order:")
    for method, path in hub.requests():
        print(f"     {method} {path}")
    print()
    print("ATTRIBUTION: the delta above is the only NEW hub traffic in the window.")
    print("The startup check's requests sit in the baseline (step 2), not in the")
    print("delta. The periodic timer loop is not running in this harness, so nothing")
    print("but the create's session-start trigger could have added a request after")
    print("the baseline was taken. The timestamps are printed as secondary evidence")
    print("only -- both checks can land in the same second, which is exactly why the")
    print("request count, not last_checked_at, is the assertion.")

    if delta != 1:
        print()
        print(f"FAIL: expected exactly one new hub request from the create; got {delta}.")
        return 1
    print()
    print("PASS: one session create produced exactly one hub check request.")
    return 0


def main() -> int:
    import contextlib

    import uvicorn
    from fastapi import FastAPI

    from local_operator.agents import (
        HUB_SHA256_PREFIX,
        AgentEditFields,
        AgentRegistry,
        hub_fingerprint,
    )
    from local_operator.clients.radient import RadientClient
    from local_operator.config import ConfigManager
    from local_operator.hub_sync import service as svc
    from local_operator.hub_sync.runner import HubSyncRunner
    from local_operator.providers.radient_credentials import configured_radient_base_url
    from local_operator.server.routes import desktop_sessions
    from scripts.probe_isolation import SANDBOX

    config_dir = Path(os.environ["LOCAL_OPERATOR_CONFIG_DIR"])
    config_dir.mkdir(parents=True, exist_ok=True)

    hub = _HubStub()
    hub.start()
    try:
        cm = ConfigManager(config_dir)
        cm.set_config_value("radient_base_url", hub.base_url)
        base_url = configured_radient_base_url(cm)
        if base_url != hub.base_url:
            raise SystemExit(
                f"FAIL: config resolved the hub root to {base_url!r}, not {hub.base_url!r}"
            )

        agents = AgentRegistry(config_dir)
        row = agents.create_agent(
            AgentEditFields(
                name="receipt-coder",
                description=DESCRIPTION,
                tags=[
                    "role",
                    f"hub:{HUB_ID}",
                    f"{HUB_SHA256_PREFIX}{hub_fingerprint(INSTRUCTIONS, DESCRIPTION)}",
                ],
            )
        )
        agents.set_agent_system_prompt(row.id, INSTRUCTIONS)

        client = RadientClient(api_key=None, base_url=base_url)

        def build_ctx() -> Any:
            return svc.HubSyncContext(
                config_dir=config_dir,
                config_manager=cm,
                client_for_tenant=lambda _tenant: client,
                credential="ok",
                agent_registry=agents,
                resolver=None,
            )

        clock = _Clock()
        runner = HubSyncRunner(
            config_manager=cm, build_ctx=build_ctx, startup_delay=0, clock=clock
        )

        @contextlib.asynccontextmanager
        async def lifespan(app: FastAPI) -> AsyncIterator[None]:
            # The daemon's first check, run directly instead of after
            # ``run_forever``'s 45 s startup sleep. ITS requests are the baseline
            # the create's delta excludes.
            await runner.tick(reason="startup")
            try:
                yield
            finally:
                pool = getattr(app.state, "desktop_sessions", None)
                if pool is not None:
                    await pool.close()

        app = FastAPI(lifespan=lifespan)
        app.state.config_manager = cm
        app.state.hub_sync = runner
        app.include_router(desktop_sessions.router)

        server = uvicorn.Server(
            uvicorn.Config(app, host="127.0.0.1", port=0, log_level="warning", lifespan="on")
        )
        api_thread = threading.Thread(target=server.run, name="desktop-api", daemon=True)
        api_thread.start()
        try:
            _wait_for_server(server)
            servers = server.servers
            if not servers:
                raise SystemExit("FAIL: uvicorn started without a bound server")
            port = int(servers[0].sockets[0].getsockname()[1])

            print("hub session-create receipt harness")
            print("---------------------------------")
            print(f"config dir:   {config_dir}")
            print(f"hub stub:     {hub.base_url}  (GET /v1/agents/{HUB_ID})")
            print(f"desktop API:  http://127.0.0.1:{port}")
            print(f"linked agent: {row.id} -> hub id {HUB_ID}")
            return _run_receipt(
                hub=hub,
                api_root=f"http://127.0.0.1:{port}",
                config_dir=config_dir,
                cwd=SANDBOX,
                clock=clock,
                agent_id=row.id,
            )
        finally:
            server.should_exit = True
            api_thread.join(timeout=15.0)
    finally:
        hub.stop()


if __name__ == "__main__":
    sys.exit(main())
