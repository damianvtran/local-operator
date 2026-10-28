"""A failed org-team check must store WHY it failed (radientdev, 2026-10-28).

The operator's org team link stored ``hub-error`` with the sentence "could not
reach the hub: Could not pull the team from the Radient Agent Hub" while the
same ``GET /v1/teams/:id`` (bearer, org tenant) answered 200 on every later read.
That sentence is ``RadientClient.get_team``'s fallback for a request that got NO
response; the transport cause lived only in ``__cause__`` and was dropped, so
nothing could tell a blip from a broken link. These tests drive the REAL client
over loopback HTTP against a hub that answers in the real hub's shapes (read
off api.radienthq.com: ``{"msg", "result"}`` on success, ``{"code", "error"}``
on refusal) and assert what the check stores.
"""

from __future__ import annotations

import json
import threading
import time
from collections.abc import Iterator
from contextlib import contextmanager
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from types import SimpleNamespace
from typing import Any

import pytest
from pydantic import SecretStr

from local_operator.clients.radient import RadientClient
from local_operator.hub_sync import check as chk
from local_operator.hub_sync import provenance as prov
from local_operator.hub_sync.identity import link_origin
from tests.unit.hub_sync.test_link_origin import SOURCE, TENANT, _team

HUB_ID = "1f48dc2e-1a0a-452a-af32-629a29b06bcc"


@contextmanager
def _hub(mode: str) -> Iterator[tuple[str, list[tuple[str, bool]]]]:
    seen: list[tuple[str, bool]] = []
    release = threading.Event()

    class Handler(BaseHTTPRequestHandler):
        def log_message(self, format: str, *args: object) -> None:
            pass

        def _send(self, status: int, payload: dict[str, Any]) -> None:
            body = json.dumps(payload).encode()
            self.send_response(status)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        def do_GET(self) -> None:
            seen.append((self.path, self.headers.get("Authorization") is not None))
            if mode == "stall":
                # Hold the response past the client's read timeout.
                release.wait(5)
                return
            if mode == "500":
                self._send(500, {"code": "internal_error", "error": "Team lookup failed."})
                return
            if mode == "404":
                self._send(404, {"code": "team_not_found", "error": "Team not found."})
                return
            self._send(
                200,
                {
                    "msg": "Team retrieved successfully",
                    "result": {
                        "id": HUB_ID,
                        "tenant_id": TENANT,
                        "name": "crew",
                        "description": "d",
                        "manager": "manager",
                        "members": [],
                        "instructions": "i",
                        "project": "",
                    },
                },
            )

    server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        yield f"http://127.0.0.1:{server.server_port}/v1", seen
    finally:
        release.set()
        server.shutdown()
        server.server_close()
        thread.join()


def _check(tmp_path: Any, base_url: str) -> chk.ItemCheck:
    team = _team("local-1")
    prov.record_team_pull(
        tmp_path,
        team,
        {"id": HUB_ID},
        tenant_id=TENANT,
        origin=link_origin(SOURCE, "org", TENANT),
    )
    registry = SimpleNamespace(
        config_dir=tmp_path, list_teams=lambda: [team], get_team=lambda _: team
    )
    client = RadientClient(api_key=SecretStr("sk-test-org-bearer"), base_url=base_url)
    (result,) = chk.check_hub_teams(registry, client_for_tenant=lambda _: client)
    return result


def test_a_timed_out_org_team_fetch_says_it_timed_out(tmp_path, monkeypatch):
    # The production bound is 20 s; a short one keeps the stall test fast.
    monkeypatch.setattr(chk, "REQUEST_TIMEOUT_S", 0.3)
    with _hub("stall") as (base_url, seen):
        started = time.monotonic()
        result = _check(tmp_path, base_url)
        elapsed = time.monotonic() - started

    assert seen == [(f"/v1/teams/{HUB_ID}", True)]
    assert result.verdict == "unavailable" and result.reason == "hub-error"
    # Before: "could not reach the hub: Could not pull the team from the Radient
    # Agent Hub" -- the cause (a read timeout) was nowhere in it.
    assert result.detail.startswith("could not reach the hub: the hub did not answer in time")
    assert "sk-test-org-bearer" not in result.detail
    assert elapsed < 3


def test_a_refused_org_team_fetch_names_the_status_and_hub_message(tmp_path):
    with _hub("500") as (base_url, _seen):
        result = _check(tmp_path, base_url)

    assert result.reason == "hub-error"
    assert result.detail == "the hub refused the request (HTTP 500: Team lookup failed.)"


def test_a_missing_org_team_stays_hub_item_missing(tmp_path):
    with _hub("404") as (base_url, _seen):
        result = _check(tmp_path, base_url)

    assert result.reason == "hub-item-missing"


def test_a_reachable_org_team_is_checked_normally(tmp_path):
    with _hub("200") as (base_url, seen):
        result = _check(tmp_path, base_url)

    assert seen == [(f"/v1/teams/{HUB_ID}", True)]
    assert result.verdict == "up-to-date", result


@pytest.mark.parametrize(
    ("cause", "phrase"),
    [
        ("ConnectTimeout", "the connection timed out"),
        ("ReadTimeout", "the hub did not answer in time"),
        ("SSLError", "the secure connection failed"),
        ("ConnectionError", "the connection failed"),
    ],
)
def test_the_transport_cause_is_read_from_the_chain(cause, phrase):
    import requests

    from local_operator.clients._http import APIError

    error = APIError("Could not pull the team from the Radient Agent Hub", status_code=None)
    error.__cause__ = getattr(requests.exceptions, cause)("pool internals")
    reason, detail = chk._failure_reason(error)

    assert reason == "hub-error"
    assert detail == (
        f"could not reach the hub: {phrase} (Could not pull the team from the Radient Agent Hub)"
    )
