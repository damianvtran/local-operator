"""The desktop approvals routes: pending requests, human batch decide, the ledger.

Real pieces wherever the property lives in them: the REAL team-bridge store
(``commit_enrollment`` + designation — the authority fence reads the document on
disk), the REAL ask queue writing a real ``asks.jsonl`` and index, the REAL
``TeamBridgeClient`` over real loopback HTTP to a stub hub that records every
request it receives, and the real ``require_desktop`` door. Only the
credential store (``FakeStore``) and the hub's domain are fixtures.
"""

from __future__ import annotations

import ast
import json
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from types import SimpleNamespace
from typing import Any
from urllib.parse import parse_qs, urlsplit

import pytest
import yaml
from fastapi import FastAPI
from fastapi.testclient import TestClient

from local_operator.asks import store as asks_store
from local_operator.server.dependencies import get_config_manager
from local_operator.server.routes import desktop_team_bridge as routes
from local_operator.team_bridge import client as bridge
from local_operator.team_bridge.authority import scope_from_owner
from local_operator.team_bridge.store import OwnerReceipt, OwnerSnapshot, Store
from tests.unit.team_bridge.test_client import FakeRow, FakeStore

TOKEN = "fixture-desktop-capability"
AUTH = {"authorization": f"Bearer {TOKEN}"}
COS = "s-cos"
ACC = "acc_1"


def _owner() -> OwnerSnapshot:
    return OwnerSnapshot(
        credential_id=7,
        account_id=ACC,
        tenant_id="tnt_1",
        endpoint_id="ep_1",
        canonical_base_url="http://hub.invalid/v1",
        enrollment_generation=5,
    )


def _item(message_id: str, *, tenant: str = "tnt_1") -> dict[str, Any]:
    return {
        "message": {
            "id": message_id,
            "thread_id": "bth_" + message_id[5:],
            "type": "request",
            "created_at": "2026-11-15T10:00:00Z",
            "body": {"text": "PRIVATE BODY — must never reach the desktop list"},
            "request": {"state": "pending", "expires_at": "2026-11-22T10:00:00Z"},
            "screening": {"rationale": "PRIVATE RATIONALE"},
        },
        "delivery": {"state": "unread"},
        "thread": {"id": "bth_" + message_id[5:], "tenant_id": tenant},
        "sender": {
            "account_id": "acc_2",
            "endpoint_id": "ep_2",
            "name": "Priya\nRaman",
            "email": "priya@acme.example",
            "role": "member",
        },
    }


class Hub:
    """The four hub routes, recording every request (method, path, query, body)."""

    def __init__(self) -> None:
        self.calls: list[dict[str, Any]] = []
        self.pending = ["bmsg_a", "bmsg_b", "bmsg_c"]
        self.batch_answer: dict[str, dict[str, str]] = {}
        self.ledger = [
            {
                "request_id": "bmsg_old",
                "thread_id": "bth_old",
                "state": "approved",
                "requested_by": {"account_id": "acc_2", "name": "Priya", "email": "p@x"},
                "decided_by": {"account_id": ACC, "name": "Damian", "email": "d@x"},
                "decided_at": "2026-11-14T09:00:00Z",
                "decided_via": "bearer-only",
                "ask_ref": {"ask_id": "a-old", "question_id": "bmsg_old"},
                "body_hash": "f" * 64,
                "note": "PRIVATE NOTE",
            }
        ]

    def answer(self, method: str, path: str, query: dict[str, list[str]], body: Any):
        if method == "GET" and path == "/v1/messaging/requests":
            return {"items": [_item(m) for m in self.pending], "next_cursor": None}
        if method == "POST" and path == "/v1/messaging/requests/decide":
            results = []
            for entry in body["decisions"]:
                mid = entry["message_id"]
                outcome = self.batch_answer.get(mid) or {"status": entry["decision"]}
                results.append({"message_id": mid, **outcome})
                if "status" in outcome and mid in self.pending:
                    self.pending.remove(mid)
            return {"results": results}
        if method == "GET" and path == "/v1/messaging/decisions":
            return {"items": self.ledger, "next_cursor": None}
        return None


@pytest.fixture
def world(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    hub_state = Hub()

    class Handler(BaseHTTPRequestHandler):
        def log_message(self, format: str, *args: object) -> None:
            pass

        def dispatch(self) -> None:
            raw = self.rfile.read(int(self.headers.get("content-length", 0)))
            body = json.loads(raw) if raw else None
            parts = urlsplit(self.path)
            query = parse_qs(parts.query)
            hub_state.calls.append(
                {"method": self.command, "path": parts.path, "query": query, "body": body}
            )
            result = hub_state.answer(self.command, parts.path, query, body)
            code, out = (
                (200, {"msg": "ok", "result": result})
                if result is not None
                else (
                    404,
                    {"code": "not_found", "error": "no such fixture route"},
                )
            )
            encoded = json.dumps(out).encode()
            self.send_response(code)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(encoded)))
            self.end_headers()
            self.wfile.write(encoded)

        do_GET = do_POST = dispatch

    server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()

    root = tmp_path
    owner = _owner()
    store = Store(root)
    store.commit_enrollment(owner, OwnerReceipt("bridge_consent_v1", "ref-1"))
    store.update_designated_session(owner, COS)
    (root / "config.yml").write_text(yaml.safe_dump({"values": {"team_bridge": {"enabled": True}}}))

    creds = FakeStore([FakeRow(7, data={"access": "token-1", "account_id": ACC})])
    real_client = bridge.TeamBridgeClient
    monkeypatch.setattr(bridge, "org_oauth_destination_allowed", lambda _base: True)
    monkeypatch.setattr(
        bridge,
        "TeamBridgeClient",
        lambda config_dir: real_client(
            config_dir, base_url=f"http://127.0.0.1:{server.server_port}/v1", store=creds
        ),
    )
    monkeypatch.setenv("LOCAL_OPERATOR_DESKTOP_TOKEN", TOKEN)
    app = FastAPI()
    app.include_router(routes.router)
    app.dependency_overrides[get_config_manager] = lambda: SimpleNamespace(config_dir=root)
    try:
        with TestClient(app) as http:
            yield SimpleNamespace(http=http, hub=hub_state, root=root, owner=owner)
    finally:
        server.shutdown()
        server.server_close()
        thread.join(10)


def _raise_ask(root: Path, message_id: str, *, session_id: str = COS, owner=None) -> str:
    """Queue a bridge ask exactly as the connector does (keyed-idempotent door)."""

    from local_operator.asks.queue import AskQueue
    from local_operator.team_bridge.ask_issuance import bridge_scope_digest
    from local_operator.team_bridge.asks_bridge import enqueue_bridge_request_ask
    from tests.unit.asks.test_queue import FakeSession

    scope = scope_from_owner(owner or _owner(), session_id)
    queue = AskQueue(FakeSession(), config_dir=root, session_id=session_id)
    receipt = enqueue_bridge_request_ask(
        queue,
        message_id=message_id,
        sender="priya@acme.example",
        excerpt="approve the spend?",
        scope=scope,
        issuance_key=f"iss-{message_id}",
        fingerprint=bridge_scope_digest(scope),
    )
    assert receipt["ok"] is True, receipt
    return str(receipt["details"]["ask_id"]) if "details" in receipt else str(receipt["ask_id"])


def _ask_status(root: Path, session_id: str, ask_id: str) -> str:
    events = asks_store.read_events(asks_store.session_dir(root, session_id))
    folded = asks_store.fold(events, asks_store.now_ms())
    return next(str(r["status"]) for r in folded if r["ask_id"] == ask_id)


def _hub_calls(hub: Hub, path: str) -> list[dict[str, Any]]:
    return [c for c in hub.calls if c["path"] == path]


# --- the list ---------------------------------------------------------------


def test_list_counts_pending_and_carries_references_only(world) -> None:
    ask_a = _raise_ask(world.root, "bmsg_a")
    response = world.http.get(
        "/v1/desktop/team-bridge/requests", params={"expected_account_id": ACC}, headers=AUTH
    )
    assert response.status_code == 200, response.text
    result = response.json()["result"]
    assert result["pending_count"] == 3
    assert [row["message_id"] for row in result["items"]] == ["bmsg_a", "bmsg_b", "bmsg_c"]
    first = result["items"][0]
    assert first["ask"] == {"session_id": COS, "ask_id": ask_a, "status": "open"}
    assert first["decidable"] is True
    assert result["items"][1]["ask"] is None and result["items"][1]["decidable"] is False
    assert first["sender"] == {
        "account_id": "acc_2",
        "endpoint_id": "ep_2",
        "name": "Priya Raman",  # one line: the control character is folded
        "email": "priya@acme.example",
    }
    text = response.text
    assert "PRIVATE BODY" not in text and "PRIVATE RATIONALE" not in text
    (call,) = _hub_calls(world.hub, "/v1/messaging/requests")
    assert call["query"]["state"] == ["pending"] and call["query"]["tenant_id"] == ["tnt_1"]


# --- the batch decide -------------------------------------------------------


def test_mixed_batch_gives_per_item_results_and_settles_asks(world) -> None:
    ask_a = _raise_ask(world.root, "bmsg_a")
    ask_b = _raise_ask(world.root, "bmsg_b")
    world.hub.batch_answer = {"bmsg_b": {"error": "request_already_decided"}}
    response = world.http.post(
        "/v1/desktop/team-bridge/requests/decide",
        json={
            "expected_account_id": ACC,
            "items": [
                {"message_id": "bmsg_a", "decision": "approve"},
                {"message_id": "bmsg_b", "decision": "deny"},
                {"message_id": "bmsg_c", "decision": "approve"},
            ],
        },
        headers=AUTH,
    )
    assert response.status_code == 200, response.text
    assert response.json()["result"]["results"] == [
        {
            "message_id": "bmsg_a",
            "decision": "approve",
            "sent": True,
            "ask_settled": True,
            "status": "approved",
        },
        {
            "message_id": "bmsg_b",
            "decision": "deny",
            "sent": True,
            "ask_settled": True,
            "error": "request_already_decided",
        },
        # No local ask for this request: never sent, no receipt minted for it.
        {
            "message_id": "bmsg_c",
            "decision": "approve",
            "sent": False,
            "ask_settled": False,
            "error": "no_open_ask",
        },
    ]
    (call,) = _hub_calls(world.hub, "/v1/messaging/requests/decide")
    assert call["body"] == {
        "decisions": [
            {
                "message_id": "bmsg_a",
                "decision": "approved",
                "ask_ref": {"ask_id": ask_a, "question_id": "bmsg_a"},
                "decided_via": "bearer-only",
            },
            {
                "message_id": "bmsg_b",
                "decision": "denied",
                "ask_ref": {"ask_id": ask_b, "question_id": "bmsg_b"},
                "decided_via": "bearer-only",
            },
        ]
    }
    # Settled in the LOG (the truth), so the card can never decide them again.
    assert _ask_status(world.root, COS, ask_a) == "withdrawn"
    assert _ask_status(world.root, COS, ask_b) == "withdrawn"
    # A second batch for the same request is refused locally: nothing is sent.
    again = world.http.post(
        "/v1/desktop/team-bridge/requests/decide",
        json={"expected_account_id": ACC, "items": [{"message_id": "bmsg_a", "decision": "deny"}]},
        headers=AUTH,
    )
    assert again.json()["result"]["results"][0]["error"] == "no_open_ask"
    assert len(_hub_calls(world.hub, "/v1/messaging/requests/decide")) == 1
    # And the card's own door: a live queue answering the settled ask re-folds
    # the log from disk and refuses, so no second decide-at-consumption runs.
    from local_operator.asks.queue import AskQueue
    from tests.unit.asks.test_queue import FakeSession

    live = AskQueue(FakeSession(), config_dir=world.root, session_id=COS)
    refused = live.respond(ask_a, {"bridge-bmsg_a": ["approve"]}, by="desktop")
    assert refused["ok"] is False
    assert _ask_status(world.root, COS, ask_a) == "withdrawn"
    # The settled requests leave the list once the hub stops serving them.
    world.hub.pending = ["bmsg_c"]
    listed = world.http.get(
        "/v1/desktop/team-bridge/requests", params={"expected_account_id": ACC}, headers=AUTH
    ).json()["result"]
    assert [row["message_id"] for row in listed["items"]] == ["bmsg_c"]


def test_an_ask_answered_on_the_card_leaves_the_list_and_is_not_decided_again(world) -> None:
    ask_a = _raise_ask(world.root, "bmsg_a")
    # The card's answer: an ``answered`` row in the ask log (what the signed
    # bridge_decide door records before its own decide-at-consumption).
    asks_store.append_event(
        asks_store.session_dir(world.root, COS),
        {
            "v": asks_store.EVENT_SCHEMA,
            "kind": asks_store.EVENT_ANSWERED,
            "ask_id": ask_a,
            "at": asks_store.now_ms(),
            "answers": {"bridge-bmsg_a": ["approve"]},
            "by": {"surface": "desktop"},
        },
    )
    entry = asks_store.read_entry(world.root, COS)
    assert entry is not None
    rows = [dict(r, status="answered") for r in entry["asks"]]
    asks_store.write_entry(world.root, COS, cwd="", asks=rows)

    listed = world.http.get(
        "/v1/desktop/team-bridge/requests", params={"expected_account_id": ACC}, headers=AUTH
    ).json()["result"]
    assert "bmsg_a" not in [row["message_id"] for row in listed["items"]]
    assert listed["pending_count"] == 2
    decided = world.http.post(
        "/v1/desktop/team-bridge/requests/decide",
        json={"expected_account_id": ACC, "items": [{"message_id": "bmsg_a", "decision": "deny"}]},
        headers=AUTH,
    ).json()["result"]["results"]
    assert decided[0]["error"] == "answered_on_card" and decided[0]["sent"] is False
    assert _hub_calls(world.hub, "/v1/messaging/requests/decide") == []


def test_an_ask_outside_the_designated_session_is_not_sent(world) -> None:
    _raise_ask(world.root, "bmsg_a", session_id="s-other")
    decided = world.http.post(
        "/v1/desktop/team-bridge/requests/decide",
        json={
            "expected_account_id": ACC,
            "items": [{"message_id": "bmsg_a", "decision": "approve"}],
        },
        headers=AUTH,
    ).json()["result"]["results"]
    assert decided[0]["sent"] is False
    assert decided[0]["error"] == "not_designated"
    assert _hub_calls(world.hub, "/v1/messaging/requests/decide") == []


@pytest.mark.parametrize(
    "body",
    [
        {
            "expected_account_id": ACC,
            "items": [{"message_id": f"bmsg_{i}", "decision": "approve"} for i in range(51)],
        },
        {"expected_account_id": ACC, "items": []},
        {"expected_account_id": ACC, "items": [{"message_id": "bmsg_a", "decision": "decline"}]},
        {
            "expected_account_id": ACC,
            "items": [
                {"message_id": "bmsg_a", "decision": "approve"},
                {"message_id": "bmsg_a", "decision": "deny"},
            ],
        },
        {
            "expected_account_id": ACC,
            "all": True,
            "items": [{"message_id": "bmsg_a", "decision": "approve"}],
        },
        {"expected_account_id": ACC, "items": [{"message_id": "../x", "decision": "approve"}]},
    ],
    ids=["over-50", "empty", "decline-word", "duplicate", "select-all-flag", "bad-id"],
)
def test_shape_failures_are_refused_whole_with_zero_hub_traffic(world, body) -> None:
    _raise_ask(world.root, "bmsg_a")
    response = world.http.post("/v1/desktop/team-bridge/requests/decide", json=body, headers=AUTH)
    assert response.status_code == 422, response.text
    assert response.json()["detail"]["code"] == "invalid_request"
    assert world.hub.calls == []


def test_over_50_names_the_bound(world) -> None:
    items = [{"message_id": f"bmsg_{i}", "decision": "approve"} for i in range(51)]
    response = world.http.post(
        "/v1/desktop/team-bridge/requests/decide",
        json={"expected_account_id": ACC, "items": items},
        headers=AUTH,
    )
    assert response.json()["detail"]["details"] == {
        "field": "items",
        "rule": "max_items",
        "max": 50,
    }


# --- the ledger ---------------------------------------------------------------


def test_ledger_is_references_only(world) -> None:
    response = world.http.get(
        "/v1/desktop/team-bridge/decisions",
        params={"expected_account_id": ACC, "role": "made", "limit": "20"},
        headers=AUTH,
    )
    assert response.status_code == 200, response.text
    result = response.json()["result"]
    assert result["items"] == [
        {
            "request_id": "bmsg_old",
            "thread_id": "bth_old",
            "decision": "approved",
            "decided_at": "2026-11-14T09:00:00Z",
            "requested_by": {"account_id": "acc_2", "name": "Priya", "email": "p@x"},
            "decided_by": {"account_id": ACC, "name": "Damian", "email": "d@x"},
            "decided_via_claimed": "bearer-only",
            "ask_id": "a-old",
        }
    ]
    assert "PRIVATE NOTE" not in response.text and "f" * 64 not in response.text
    (call,) = _hub_calls(world.hub, "/v1/messaging/decisions")
    assert call["query"]["role"] == ["made"] and call["query"]["limit"] == ["20"]


def test_ledger_refuses_an_unknown_role_and_unknown_params(world) -> None:
    for params in (
        {"expected_account_id": ACC, "role": "all"},
        {"expected_account_id": ACC, "x": "1"},
    ):
        response = world.http.get("/v1/desktop/team-bridge/decisions", params=params, headers=AUTH)
        assert response.status_code == 422
    assert world.hub.calls == []


# --- the door ---------------------------------------------------------------


@pytest.mark.parametrize(
    "method,path",
    [
        ("GET", "/v1/desktop/team-bridge/requests"),
        ("POST", "/v1/desktop/team-bridge/requests/decide"),
        ("GET", "/v1/desktop/team-bridge/decisions"),
    ],
)
def test_no_token_is_401_and_a_foreign_origin_is_403(world, method, path) -> None:
    body = {"expected_account_id": ACC, "items": [{"message_id": "bmsg_a", "decision": "approve"}]}
    kwargs: dict[str, Any] = (
        {"json": body} if method == "POST" else {"params": {"expected_account_id": ACC}}
    )
    assert world.http.request(method, path, **kwargs).status_code == 401
    foreign = {**AUTH, "origin": "https://evil.example"}
    assert world.http.request(method, path, headers=foreign, **kwargs).status_code == 403
    assert world.hub.calls == []


# --- decide authority stays off every model/agent path ------------------------


def _imports(path: Path, package: Path) -> set[str]:
    """Every absolute module/name a file imports, relative imports resolved."""

    parts = list(path.relative_to(package.parent).with_suffix("").parts)
    tree = ast.parse(path.read_text(encoding="utf-8"))
    names: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.ImportFrom):
            if node.level:
                base = parts[: len(parts) - node.level]
                module = ".".join(base + ([node.module] if node.module else []))
            else:
                module = node.module or ""
            names.add(module)
            names.update(f"{module}.{alias.name}" for alias in node.names)
        elif isinstance(node, ast.Import):
            names.update(alias.name for alias in node.names)
    return names


def test_only_the_desktop_route_reaches_the_batch_decide() -> None:
    """No tool, agent, session or connector module can reach batch decide authority.

    ``team_bridge.approvals`` and ``decide_batch_from_human_selection`` are
    imported by the desktop route family (behind the bearer + Origin door) and
    nowhere else in the package; ``decide.py`` defines the executor.
    """

    package = Path(routes.__file__).resolve().parents[2]
    allowed = {
        package / "server" / "routes" / "desktop_team_bridge.py",
        package / "team_bridge" / "approvals.py",
        package / "team_bridge" / "decide.py",
    }
    forbidden = (
        "local_operator.team_bridge.approvals",
        "local_operator.team_bridge.decide.decide_batch_from_human_selection",
    )
    offenders = sorted(
        str(path.relative_to(package))
        for path in package.rglob("*.py")
        if path not in allowed
        and any(name.startswith(forbidden) for name in _imports(path, package))
    )
    assert offenders == []
    # The scan resolves the allowed caller's own import (a vacuous scan proves nothing).
    route = package / "server" / "routes" / "desktop_team_bridge.py"
    assert "local_operator.team_bridge.approvals" in _imports(route, package)


def test_the_desktop_bearer_never_reaches_an_agent_shell() -> None:
    from local_operator.agent_shell import DAEMON_BEARER_ENVS

    assert "LOCAL_OPERATOR_DESKTOP_TOKEN" in DAEMON_BEARER_ENVS


@pytest.mark.asyncio
async def test_client_requests_refuses_an_unknown_state_before_any_io(tmp_path) -> None:
    client = bridge.TeamBridgeClient(tmp_path, base_url="http://127.0.0.1:9/v1", store=FakeStore())
    with pytest.raises(ValueError):
        await client.requests(tenant_id="tnt_1", state="open")
