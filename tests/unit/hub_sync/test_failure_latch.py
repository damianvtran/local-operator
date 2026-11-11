"""A startup tick refused by the hub must not latch a failure (operator report, 2026-11-11).

After a restart almost every org team showed the red "Retry Incoming update"
mark. Mechanism, reproduced here through the production runner, check, store and
projections with a REAL ``RadientClient`` over loopback HTTP: the startup tick's
bearer was refused (HTTP 401 "invalid API key format"), each org link stored a
COUNTED ``hub-error``, and a both-changed row (radientdev) kept that error -- and,
past ``MAX_ATTEMPTS``, a retired ``auto_retry`` -- through every later 200,
because a row that stays ``available`` never reached the only arm that cleared
it. These tests pin both halves: a 401 is never a counted failure, and a
successful fetch heals what a fetch recorded.
"""

from __future__ import annotations

import json
import threading
from collections.abc import Iterator
from contextlib import contextmanager
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any

import pytest
from pydantic import SecretStr

from local_operator.agents import AgentRegistry
from local_operator.clients.radient import RadientClient
from local_operator.config import ConfigManager
from local_operator.hub_sync import provenance as prov
from local_operator.hub_sync import service as svc
from local_operator.hub_sync import store as st
from local_operator.hub_sync.identity import LinkOrigin
from local_operator.hub_sync.runner import HubSyncRunner
from local_operator.teams import TeamEditFields, TeamRegistry


@contextmanager
def _hub(docs: dict[str, dict[str, Any]], phase: dict[str, bool]) -> Iterator[str]:
    class Handler(BaseHTTPRequestHandler):
        def log_message(self, format: str, *args: object) -> None:
            pass

        def do_GET(self) -> None:
            hub_id = self.path.rstrip("/").split("/")[-1].split("?")[0]
            if not phase["accepts"]:
                status, body = 401, {"code": "unauthorized", "error": "invalid API key format"}
            elif hub_id in docs:
                status, body = 200, {"msg": "ok", "result": docs[hub_id]}
            else:
                status, body = 404, {"code": "team_not_found", "error": "Team not found."}
            raw = json.dumps(body).encode()
            self.send_response(status)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(raw)))
            self.end_headers()
            self.wfile.write(raw)

    server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        yield f"http://127.0.0.1:{server.server_port}/v1"
    finally:
        server.shutdown()
        server.server_close()
        thread.join()


def _doc(hub_id: str, name: str) -> dict[str, Any]:
    return {
        "id": hub_id,
        "tenant_id": "org-1",
        "name": name,
        "description": "d",
        "manager": "lead",
        "members": [{"role": "coder", "kind": "agent", "count": 1}],
        "instructions": "## Brief\nShip it.",
        "project": "",
        "version": "1.0.0",
    }


@pytest.mark.asyncio
async def test_a_refused_startup_tick_never_latches_a_failure(tmp_path: Path) -> None:
    root = tmp_path / ".local-operator"
    docs: dict[str, dict[str, Any]] = {}
    phase = {"accepts": False}
    with _hub(docs, phase) as base:
        cm = ConfigManager(root)
        agents = AgentRegistry(root)
        teams = TeamRegistry(root)
        ids: dict[str, str] = {}
        for hub_id, name in (("ht-sync", "content"), ("ht-both", "radientdev")):
            docs[hub_id] = _doc(hub_id, name)
            outcome = teams.import_hub_team(docs[hub_id])
            prov.record_team_pull(
                root,
                teams.get_team(outcome.team.id),
                docs[hub_id],
                tenant_id="org-1",
                origin=LinkOrigin(hub_source=base, visibility="org", tenant_id="org-1"),
            )
            ids[name] = outcome.team.id
        # Both sides changed DIFFERENT fields, so the existing merge flow can
        # integrate the hub's brief without an LLM and must keep the local edit.
        teams.update_team(ids["radientdev"], TeamEditFields(description="local description"))
        docs["ht-both"] = {**docs["ht-both"], "instructions": "## Brief\nShip it, upstream."}
        client = RadientClient(api_key=SecretStr("sk-test-not-usable-yet"), base_url=base)

        def build() -> svc.HubSyncContext:
            return svc.HubSyncContext(
                config_dir=root,
                config_manager=cm,
                client_for_tenant=lambda _t: client,
                credential="ok",
                agent_registry=agents,
                team_registry=teams,
                resolver=None,
            )

        runner = HubSyncRunner(config_manager=cm, build_ctx=build, startup_delay=0)
        ctx = build()

        # Refused for longer than MAX_ATTEMPTS: the first tick is the startup one.
        for i in range(st.MAX_ATTEMPTS + 1):
            await runner.tick(reason="startup" if i == 0 else "timer")
        items = ctx.store().load()["items"]
        for name in ("content", "radientdev"):
            item = items[st.item_key("team", ids[name])]
            # Before: error_class=hub-error, attempts=7, auto_retry=False.
            assert item["error_class"] == "no-credential", item
            assert item["error_subclass"] == "rejected"
            assert item["attempts"] == 0 and item["auto_retry"] is True
            assert "HTTP 401: invalid API key format" in item["last_error"]
        snapshot = svc.status_snapshot(ctx)
        assert snapshot["counts"]["failed"] == 0, snapshot
        # The links projection carries the refusal's own sentence for the tooltip.
        pulls = {row["local_id"]: row["pull"] for row in svc.links_snapshot(ctx)["links"]}
        assert pulls[ids["radientdev"]]["error_class"] == "no-credential"
        assert pulls[ids["radientdev"]]["error_subclass"] == "rejected"
        assert "invalid API key format" in pulls[ids["radientdev"]]["safe_message"]

        phase["accepts"] = True
        await runner.tick(reason="timer")

    items = ctx.store().load()["items"]
    synced = items[st.item_key("team", ids["content"])]
    both = items[st.item_key("team", ids["radientdev"])]
    assert synced["error_class"] is None and synced["state"] == "up-to-date"
    # Before: state=available, classification=both-changed, error_class=hub-error
    # with auto_retry False, i.e. "failed" in every projection after the 200.
    assert both["error_class"] is None, both
    assert both["state"] == "applied" and both["auto_retry"] is True
    merged = teams.get_team(ids["radientdev"])
    assert merged.description == "local description"  # the local edit is kept
    assert merged.instructions.strip() == "## Brief\nShip it, upstream."
    assert svc.status_snapshot(ctx)["counts"]["failed"] == 0
