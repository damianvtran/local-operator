"""The desktop approvals routes: pending requests, human batch decide, the ledger.

Real pieces wherever the property lives in them: the REAL team-bridge store
(``commit_enrollment`` + designation — the authority fence reads the document on
disk), the REAL ask queue writing a real ``asks.jsonl`` and index, the REAL
``TeamBridgeClient`` over real loopback HTTP to a stub hub that records every
request it receives, and the real ``require_desktop`` door. Only the
credential store (``FakeStore``) and the hub's domain are fixtures.

THE OPERATOR PRESENCE PROOF is exercised with a REAL operator key: a file-backed
key created by ``operator.sign.create_key`` in the scratch root and signed by the
real ``sign_challenge`` (the call that raises Touch ID on a presence host) and
verified by the real ``signature_verdict``. Only the anchor's ``presence`` flag
is asserted by the fixture — the OS gesture itself cannot run headless, the
same limit the operator-authority suite documents.
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
DECIDE = "/v1/desktop/team-bridge/requests/decide"
AUTHORIZE = "/v1/desktop/team-bridge/requests/decide/authorize"
LIST = "/v1/desktop/team-bridge/requests"


def _owner() -> OwnerSnapshot:
    return OwnerSnapshot(
        credential_id=7,
        account_id=ACC,
        tenant_id="tnt_1",
        endpoint_id="ep_1",
        canonical_base_url="http://hub.invalid/v1",
        enrollment_generation=5,
    )


def _item(message_id: str, *, tenant: str | None = "tnt_1") -> dict[str, Any]:
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
        "thread": {"id": "bth_" + message_id[5:], **({"tenant_id": tenant} if tenant else {})},
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
        #: Commit the batch, then drop the connection before answering (R1-1).
        self.drop_after_commit = False
        #: Answer the batch 503 without committing anything.
        self.fail_batch = False
        #: Items the hub serves with no tenant at all (S-4).
        self.tenantless: set[str] = set()
        #: Two-page list: items beyond this many go on page 2 (R1-4).
        self.page_size: int | None = None
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
            ids = list(self.pending)
            cursor = None
            if self.page_size is not None:
                start = self.page_size if query.get("after") == ["cur_page2"] else 0
                cursor = "cur_page2" if start == 0 and len(ids) > self.page_size else None
                ids = ids[start : start + self.page_size]
            items = [_item(m, tenant=None if m in self.tenantless else "tnt_1") for m in ids]
            return {"items": items, "next_cursor": cursor}
        if method == "POST" and path == "/v1/messaging/requests/decide":
            if self.fail_batch:
                return "fail"
            results = []
            for entry in body["decisions"]:
                mid = entry["message_id"]
                if mid not in self.pending and mid not in self.batch_answer:
                    outcome: dict[str, str] = {"error": "request_already_decided"}
                else:
                    outcome = self.batch_answer.get(mid) or {"status": entry["decision"]}
                if "omit" in outcome:
                    continue
                results.append({"message_id": mid, **outcome})
                if "status" in outcome and mid in self.pending:
                    self.pending.remove(mid)
                    self.ledger.insert(
                        0,
                        {
                            "request_id": mid,
                            "thread_id": "bth_" + mid[5:],
                            "state": outcome["status"],
                            "decided_by": {"account_id": ACC},
                            "decided_at": "2026-11-15T11:00:00Z",
                            "decided_via": "bearer-only",
                            "ask_ref": entry["ask_ref"],
                        },
                    )
            return "drop" if self.drop_after_commit else {"results": results}
        if method == "GET" and path == "/v1/messaging/decisions":
            return {"items": self.ledger, "next_cursor": None}
        return None


def _presence_host(root: Path, monkeypatch: pytest.MonkeyPatch) -> SimpleNamespace:
    """A real file operator key in ``root``, anchored as a PRESENCE host.

    ``create_key`` writes the canonical file key and the staged anchor makes
    ``sign_challenge`` resolve that backend (the operator-authority suite's
    ``keyed`` recipe). The pinned-anchor read is the one seam replaced: it
    reports this key's anchor, root-owned, with ``presence=True`` — the
    level a Secure Enclave key reports, which no headless run can create.
    ``state.presence = False`` turns the host into a file-only one.
    """

    from dataclasses import replace

    from local_operator import operator as operator_authority
    from local_operator.operator import trust
    from local_operator.operator.keychain import FILE_ONLY
    from local_operator.operator.sign import anchor_for_handle, create_key

    handle = create_key(config_root=root, preference=FILE_ONLY)
    anchor = anchor_for_handle(handle, label="test")
    staged = trust.staging_path(root)
    staged.parent.mkdir(parents=True, exist_ok=True)
    staged.write_bytes(trust.anchor_bytes(anchor))
    state = SimpleNamespace(presence=True, anchor=anchor)

    def load(_uid: Any = None) -> Any:
        return trust.AnchorLoad(
            anchor=replace(anchor, presence=state.presence),
            path=root / "anchor.json",
            root_owned=True,
            reason="",
            exists=True,
        )

    monkeypatch.setattr(operator_authority, "load_anchor", load)
    return state


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
            if result == "drop":
                # Committed, then the connection dies before any response.
                self.close_connection = True
                return
            if result == "fail":
                result, failed = None, True
            else:
                failed = False
            if failed:
                encoded = json.dumps({"code": "messaging_unavailable", "error": "down"}).encode()
                self.send_response(503)
                self.send_header("Content-Type", "application/json")
                self.send_header("Content-Length", str(len(encoded)))
                self.end_headers()
                self.wfile.write(encoded)
                return
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
    presence = _presence_host(root, monkeypatch)
    app = FastAPI()
    app.include_router(routes.router)
    app.dependency_overrides[get_config_manager] = lambda: SimpleNamespace(config_dir=root)
    try:
        with TestClient(app) as http:
            yield SimpleNamespace(
                http=http, hub=hub_state, root=root, owner=owner, presence=presence
            )
    finally:
        server.shutdown()
        server.server_close()
        thread.join(10)


@pytest.fixture(autouse=True)
def _fresh_prompt_state():
    """The challenge store and the prompt gate are process-global: reset per test."""

    from local_operator.team_bridge import approvals

    approvals.reset_for_tests()
    yield
    approvals.reset_for_tests()


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


def _items(*pairs: tuple[str, str]) -> list[dict[str, str]]:
    return [{"message_id": mid, "decision": word} for mid, word in pairs]


def _authorize(world, items: list[dict[str, str]], *, account: str = ACC) -> dict[str, Any]:
    """The desktop's confirm step: the daemon mints, prompts and signs."""

    response = world.http.post(
        AUTHORIZE, json={"expected_account_id": account, "items": items}, headers=AUTH
    )
    assert response.status_code == 200, response.text
    return response.json()["result"]["operator_proof"]


def _decide(
    world,
    items: list[dict[str, str]],
    *,
    proof: Any = "authorize",
    account: str = ACC,
    headers: dict[str, str] = AUTH,
):
    """POST the batch; by default with a fresh proof over exactly ``items``."""

    body: dict[str, Any] = {"expected_account_id": account, "items": items}
    if proof == "authorize":
        proof = _authorize(world, items, account=account)
    if proof is not None:
        body["operator_proof"] = proof
    return world.http.post(DECIDE, json=body, headers=headers)


def _intent_states(root: Path) -> dict[str, str]:
    import sqlite3

    from local_operator.team_bridge.decide_intents import decide_intents_db_path

    path = decide_intents_db_path(root)
    if not path.exists():
        return {}
    with sqlite3.connect(path) as conn:
        rows = conn.execute("SELECT question_id, state FROM decide_intents").fetchall()
    return {str(mid): str(state) for mid, state in rows}


# --- the list ---------------------------------------------------------------


def test_list_counts_pending_and_carries_references_only(world) -> None:
    ask_a = _raise_ask(world.root, "bmsg_a")
    response = world.http.get(
        "/v1/desktop/team-bridge/requests", params={"expected_account_id": ACC}, headers=AUTH
    )
    assert response.status_code == 200, response.text
    result = response.json()["result"]
    # Only the ACTIONABLE rows count (R1-4): bmsg_b/c have no local ask.
    assert result["pending_count"] == 1
    assert result["has_more"] is False and result["next_cursor"] is None
    assert [row["message_id"] for row in result["items"]] == ["bmsg_a", "bmsg_b", "bmsg_c"]
    first = result["items"][0]
    assert first["ask"] == {"session_id": COS, "ask_id": ask_a, "status": "open"}
    assert first["decidable"] is True
    # S-2: the ask card's own screened question text, so a batch is not blind.
    assert first["excerpt"] == "priya@acme.example: approve the spend?"
    assert result["items"][1]["ask"] is None and result["items"][1]["decidable"] is False
    assert result["items"][1]["excerpt"] is None
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
    response = _decide(
        world, _items(("bmsg_a", "approve"), ("bmsg_b", "deny"), ("bmsg_c", "approve"))
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
    again = _decide(world, _items(("bmsg_a", "deny")))
    assert again.json()["result"]["results"][0]["error"] == "no_open_ask"
    assert len(_hub_calls(world.hub, "/v1/messaging/requests/decide")) == 1
    # And the card's own door: a live queue answering the settled ask re-folds
    # the log from disk and refuses, so no second decide-at-consumption runs.
    from local_operator.asks.queue import AskQueue
    from tests.unit.asks.test_queue import FakeSession

    live = AskQueue(FakeSession(), config_dir=world.root, session_id=COS)
    refused = live.respond(ask_a, {"bridge-bmsg_a": ["approve"]}, by="desktop")
    assert refused["ok"] is False
    # Q2: the refusal names the desktop decision, not an agent withdrawal.
    from local_operator.asks.render import BATCH_DECIDED_COPY

    assert refused["error"] == BATCH_DECIDED_COPY
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
    assert listed["pending_count"] == 0  # bmsg_b/c are listed but not decidable here
    decided = _decide(world, _items(("bmsg_a", "deny"))).json()["result"]["results"]
    assert decided[0]["error"] == "answered_on_card" and decided[0]["sent"] is False
    assert _hub_calls(world.hub, "/v1/messaging/requests/decide") == []


def test_an_ask_outside_the_designated_session_is_not_sent(world) -> None:
    _raise_ask(world.root, "bmsg_a", session_id="s-other")
    decided = _decide(world, _items(("bmsg_a", "approve"))).json()["result"]["results"]
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
        ("POST", "/v1/desktop/team-bridge/requests/decide/authorize"),
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


# --- S-1: a decision needs the operator's presence, not just the bearer ------


def test_a_bearer_only_post_is_refused_with_zero_hub_calls(world) -> None:
    """What ``curl`` from an agent shell sends: bearer, no Origin, no proof."""

    ask_a = _raise_ask(world.root, "bmsg_a")
    response = _decide(world, _items(("bmsg_a", "approve")), proof=None)
    assert response.status_code == 403, response.text
    assert response.json()["detail"]["code"] == "operator_signature_required"
    assert world.hub.calls == []
    assert _ask_status(world.root, COS, ask_a) == "open"
    assert _intent_states(world.root) == {}


def test_a_signature_over_a_different_selection_is_refused(world) -> None:
    _raise_ask(world.root, "bmsg_a")
    _raise_ask(world.root, "bmsg_b")
    proof = _authorize(world, _items(("bmsg_a", "deny")))
    # Flipped word, then an added item: neither is what was confirmed.
    flipped = _decide(world, _items(("bmsg_a", "approve")), proof=proof)
    assert flipped.status_code == 403
    assert flipped.json()["detail"]["code"] == "operator_signature_invalid"
    proof = _authorize(world, _items(("bmsg_a", "deny")))
    widened = _decide(world, _items(("bmsg_a", "deny"), ("bmsg_b", "approve")), proof=proof)
    assert widened.json()["detail"]["code"] == "operator_signature_invalid"
    assert world.hub.calls == []


def test_a_replayed_signature_is_refused(world) -> None:
    _raise_ask(world.root, "bmsg_a")
    items = _items(("bmsg_a", "approve"))
    proof = _authorize(world, items)
    first = _decide(world, items, proof=proof)
    assert first.status_code == 200, first.text
    assert first.json()["result"]["results"][0]["status"] == "approved"
    replay = _decide(world, items, proof=proof)
    assert replay.status_code == 403
    assert replay.json()["detail"]["code"] == "operator_signature_invalid"
    assert len(_hub_calls(world.hub, "/v1/messaging/requests/decide")) == 1


def test_a_forged_or_expired_proof_is_refused(world, monkeypatch) -> None:
    from local_operator.team_bridge import approvals

    _raise_ask(world.root, "bmsg_a")
    items = _items(("bmsg_a", "approve"))
    proof = _authorize(world, items)
    forged = _decide(world, items, proof={**proof, "sig": "30" + "ab" * 35})
    assert forged.json()["detail"]["code"] == "operator_signature_invalid"
    monkeypatch.setattr(approvals, "CHALLENGE_TTL_S", -1.0)
    expired = _decide(world, items)
    assert expired.status_code == 403
    assert expired.json()["detail"]["code"] == "operator_signature_expired"
    assert world.hub.calls == []


def test_the_proof_names_the_selection_digest_the_daemon_signed(world) -> None:
    """The signed message is the runtime's own ``signed_message``, re-checkable."""

    from local_operator.operator.verify import signed_message, verify_signature
    from local_operator.team_bridge import approvals

    items = _items(("bmsg_b", "deny"), ("bmsg_a", "approve"))
    response = world.http.post(
        AUTHORIZE, json={"expected_account_id": ACC, "items": items}, headers=AUTH
    )
    result = response.json()["result"]
    digest = approvals.selection_digest(
        account_id=ACC,
        tenant_id="tnt_1",
        selection=[approvals.Selection(i["message_id"], i["decision"]) for i in reversed(items)],
    )
    assert result["selection_digest"] == digest  # submitted order does not matter
    assert result["expires_s"] == 30
    message = signed_message(
        action="decide",
        session_id=approvals.SIGNING_SESSION,
        request_id=f"batch:{digest}",
        challenge=result["operator_proof"]["challenge"],
    )
    assert verify_signature(
        spki=world.presence.anchor.spki,
        message=message,
        signature=bytes.fromhex(result["operator_proof"]["sig"]),
    )
    assert world.hub.calls == []  # authorizing calls nobody


def test_a_host_without_a_presence_key_cannot_batch_decide(world) -> None:
    _raise_ask(world.root, "bmsg_a")
    items = _items(("bmsg_a", "approve"))
    proof = _authorize(world, items)
    world.presence.presence = False  # the anchor is now file-only
    authorize = world.http.post(
        AUTHORIZE, json={"expected_account_id": ACC, "items": items}, headers=AUTH
    )
    assert authorize.status_code == 409
    assert authorize.json()["detail"]["code"] == "operator_presence_required"
    # QA round 2, Q3: one sentence for any count (this selection has ONE item).
    assert authorize.json()["detail"]["message"] == (
        "deciding from the approvals list needs operator authority with Touch ID on this "
        "machine; answer the request on its card instead"
    )
    decided = _decide(world, items, proof=proof)
    assert decided.status_code == 409
    assert decided.json()["detail"]["code"] == "operator_presence_required"
    assert world.hub.calls == []


# --- R1-1: a lost response never strands a decided request ------------------


def test_drop_after_commit_is_settled_from_the_ledger_on_the_next_list(world) -> None:
    """The reviewer's repro: the hub commits, then the connection drops."""

    ask_a = _raise_ask(world.root, "bmsg_a")
    world.hub.drop_after_commit = True
    response = _decide(world, _items(("bmsg_a", "approve")))
    assert response.status_code == 502, response.text
    assert world.hub.pending == ["bmsg_b", "bmsg_c"]  # the hub recorded it
    # Recorded BEFORE the POST, so the lost answer left a durable trace.
    assert _intent_states(world.root) == {"bmsg_a": "pending"}
    assert _ask_status(world.root, COS, ask_a) == "open"

    world.hub.drop_after_commit = False
    listed = world.http.get(LIST, params={"expected_account_id": ACC}, headers=AUTH)
    assert listed.status_code == 200, listed.text
    assert [r["message_id"] for r in listed.json()["result"]["items"]] == ["bmsg_b", "bmsg_c"]
    # The ledger read settled it: no POST, ask retired, intent confirmed.
    assert len(_hub_calls(world.hub, "/v1/messaging/decisions")) == 1
    assert len(_hub_calls(world.hub, "/v1/messaging/requests/decide")) == 1
    assert _ask_status(world.root, COS, ask_a) == "withdrawn"
    assert _intent_states(world.root) == {"bmsg_a": "sent"}


def _answer_on_card(root: Path, message_id: str, ask_id: str, word: str) -> None:
    """The card's ask-log write (``AskQueue.respond``), as QA round 2 drove it."""

    from local_operator.asks.queue import AskQueue
    from tests.unit.asks.test_queue import FakeSession

    queue = AskQueue(FakeSession(), config_dir=root, session_id=COS)
    outcome = queue.respond(ask_id, {f"bridge-{message_id}": [word]}, by="operator")
    assert outcome.get("ok") is True, outcome


def _fail_one_batch(world, message_id: str, word: str) -> None:
    """A signed batch the hub answers 503 without committing (S-7/R2-1 setup)."""

    world.hub.fail_batch = True
    failed = _decide(world, _items((message_id, word)))
    assert failed.status_code >= 500, failed.text
    world.hub.fail_batch = False
    assert _intent_states(world.root) == {message_id: "pending"}


def _sent_pairs(world) -> list[tuple[str, str]]:
    return [
        (d["message_id"], d["decision"])
        for call in _hub_calls(world.hub, "/v1/messaging/requests/decide")
        for d in call["body"]["decisions"]
    ]


def test_an_unsent_batch_word_is_never_carried_by_another_batch(world) -> None:
    """R2-1 / S-7 repro (exit 5): no implicit carriage outside the signed digest.

    Round 2 re-sent the failed ``approve`` under a later batch signed over
    ``bmsg_b`` alone, after the card had said deny. Now the card's deny is
    refused (``decided_from_list``: the door consults exactly the row asserted
    below; the door itself is pinned in ``test_tb02_p1_provenance``), the ask
    stays open, and the unrelated batch carries nothing. Fails if the leftover
    loop comes back: the hub would receive ``bmsg_a`` a second time, approved.
    """

    from local_operator.team_bridge import approvals
    from local_operator.team_bridge.decide_intents import DecideIntents

    ask_a = _raise_ask(world.root, "bmsg_a")
    _raise_ask(world.root, "bmsg_b")
    _fail_one_batch(world, "bmsg_a", "approve")
    unconfirmed = DecideIntents(world.root).unconfirmed_batch_intent(
        approvals._owner_key(world.owner), ask_a, "bmsg_a"
    )
    assert unconfirmed is not None and unconfirmed.word == "approve"  # the card refuses
    assert _ask_status(world.root, COS, ask_a) == "open"

    response = _decide(world, _items(("bmsg_b", "deny")))
    assert response.status_code == 200, response.text
    result = response.json()["result"]
    assert set(result) == {"results"}  # no ``recovered`` field any more
    assert [r["message_id"] for r in result["results"]] == ["bmsg_b"]
    last = _hub_calls(world.hub, "/v1/messaging/requests/decide")[-1]
    assert [d["message_id"] for d in last["body"]["decisions"]] == ["bmsg_b"]
    # The only POST that ever named bmsg_a is the one the hub refused with 503.
    assert [mid for mid, _w in _sent_pairs(world)].count("bmsg_a") == 1
    assert "bmsg_a" in world.hub.pending  # the hub never decided it
    assert _ask_status(world.root, COS, ask_a) == "open"


def test_a_conflicting_reselection_is_refused_at_authorize_before_the_prompt(
    world, monkeypatch
) -> None:
    """Exit 6a (P5): the other word is refused before any mint or prompt."""

    from local_operator.team_bridge import approvals

    _raise_ask(world.root, "bmsg_a")
    _fail_one_batch(world, "bmsg_a", "approve")
    signed: list[str] = []
    real = approvals.sign_selection
    monkeypatch.setattr(
        approvals, "sign_selection", lambda *a, **k: signed.append("x") or real(*a, **k)
    )
    refused = world.http.post(
        AUTHORIZE,
        json={"expected_account_id": ACC, "items": _items(("bmsg_a", "deny"))},
        headers=AUTH,
    )
    assert refused.status_code == 409, refused.text
    detail = refused.json()["detail"]
    assert detail["code"] == "earlier_decision_unconfirmed"
    assert detail["details"] == {"items": [{"message_id": "bmsg_a", "unsent_decision": "approve"}]}
    assert signed == []
    assert approvals._challenges == {}


def test_decide_refuses_a_conflicting_selection_whole_with_zero_hub_calls(world) -> None:
    """Exit 6b: a VALID proof over a conflicting selection changes nothing."""

    from local_operator.team_bridge import approvals

    _raise_ask(world.root, "bmsg_a")
    _raise_ask(world.root, "bmsg_b")
    _fail_one_batch(world, "bmsg_a", "approve")
    calls_before = len(_hub_calls(world.hub, "/v1/messaging/requests/decide"))
    selection = [
        approvals.Selection("bmsg_a", "deny"),
        approvals.Selection("bmsg_b", "approve"),
    ]
    # A proof minted outside ``/authorize``'s check (the daemon's own primitives).
    digest = approvals.selection_digest(account_id=ACC, tenant_id="tnt_1", selection=selection)
    challenge = approvals.mint_challenge(account_id=ACC, tenant_id="tnt_1", digest=digest)
    signature = approvals.sign_selection(world.root, challenge=challenge, digest=digest)
    refused = _decide(
        world,
        _items(("bmsg_a", "deny"), ("bmsg_b", "approve")),
        proof={"challenge": challenge, **signature},
    )
    assert refused.status_code == 409, refused.text
    assert refused.json()["detail"]["code"] == "earlier_decision_unconfirmed"
    assert len(_hub_calls(world.hub, "/v1/messaging/requests/decide")) == calls_before
    assert _intent_states(world.root) == {"bmsg_a": "pending"}  # no bmsg_b row written


def test_a_list_row_carries_its_unsent_decision_and_only_that_word_sends(world) -> None:
    """Exit 9: the row shows the frozen word; selecting it sends it, signed."""

    ask_a = _raise_ask(world.root, "bmsg_a")
    _raise_ask(world.root, "bmsg_b")
    _fail_one_batch(world, "bmsg_a", "approve")
    listed = world.http.get(LIST, params={"expected_account_id": ACC}, headers=AUTH)
    rows = {r["message_id"]: r for r in listed.json()["result"]["items"]}
    assert rows["bmsg_a"]["unsent_decision"] == "approve"
    assert rows["bmsg_b"]["unsent_decision"] is None
    assert len(_hub_calls(world.hub, "/v1/messaging/requests/decide")) == 1  # the list never POSTs

    resent = _decide(world, _items(("bmsg_a", "approve")))
    assert resent.status_code == 200, resent.text
    assert resent.json()["result"]["results"][0]["status"] == "approved"
    assert _sent_pairs(world)[-1] == ("bmsg_a", "approved")
    assert _ask_status(world.root, COS, ask_a) == "withdrawn"
    assert _intent_states(world.root) == {"bmsg_a": "sent"}


def test_a_card_answer_between_record_and_post_is_not_sent(world, monkeypatch) -> None:
    """Exit 8: the fold is re-checked after the intent is recorded, before the POST."""

    from local_operator.team_bridge import decide_intents

    ask_a = _raise_ask(world.root, "bmsg_a")
    real_record = decide_intents.DecideIntents.record

    def record_then_card(self, *args, **kwargs):
        recorded = real_record(self, *args, **kwargs)
        _answer_on_card(world.root, "bmsg_a", ask_a, "deny")  # lands after the record
        return recorded

    monkeypatch.setattr(decide_intents.DecideIntents, "record", record_then_card)
    response = _decide(world, _items(("bmsg_a", "approve")))
    assert response.status_code == 200, response.text
    (result,) = response.json()["result"]["results"]
    assert result["sent"] is False and result["error"] == "answered_on_card"
    assert _hub_calls(world.hub, "/v1/messaging/requests/decide") == []
    assert _intent_states(world.root) == {"bmsg_a": "refused"}
    assert _ask_status(world.root, COS, ask_a) == "answered"


# --- S-6 / S-8: the prompt gate and the published prompt state ---------------


def _post_authorize(world, items, **extra):
    return world.http.post(
        AUTHORIZE, json={"expected_account_id": ACC, "items": items, **extra}, headers=AUTH
    )


def _dismissing_signer(monkeypatch) -> list[str]:
    """``sign_challenge`` as a dismissed Touch ID sheet; returns the call log."""

    from local_operator.operator import sign as sign_mod
    from local_operator.operator.keychain import KeyBackendError

    calls: list[str] = []

    def dismissed(**kwargs):
        calls.append(kwargs["challenge"])
        raise KeyBackendError("the person dismissed the sheet")

    monkeypatch.setattr(sign_mod, "sign_challenge", dismissed)
    return calls


def test_a_concurrent_authorize_is_refused_while_a_prompt_is_up(world, monkeypatch) -> None:
    """Exit 1: single-flight — one signer call, one live challenge, a 429 for the rest."""

    from local_operator.operator import sign as sign_mod
    from local_operator.team_bridge import approvals

    _raise_ask(world.root, "bmsg_a")
    entered, release = threading.Event(), threading.Event()
    calls: list[str] = []
    real = sign_mod.sign_challenge

    def blocking(**kwargs):
        calls.append(kwargs["challenge"])
        entered.set()
        assert release.wait(10)
        return real(**kwargs)

    monkeypatch.setattr(sign_mod, "sign_challenge", blocking)
    first: dict[str, Any] = {}
    worker = threading.Thread(
        target=lambda: first.update(r=_post_authorize(world, _items(("bmsg_a", "approve"))))
    )
    worker.start()
    try:
        assert entered.wait(10)
        second = _post_authorize(world, _items(("bmsg_a", "approve")))
        assert second.status_code == 429, second.text
        assert second.json()["detail"]["code"] == "operator_prompt_in_flight"
        assert len(calls) == 1
        assert len(approvals._challenges) == 1
    finally:
        release.set()
        worker.join(10)
    assert first["r"].status_code == 200, first["r"].text


def test_dismissed_prompts_never_exhaust_the_challenge_store(world, monkeypatch) -> None:
    """Exit 2 (S-8): 65 dismissals, then a real prompt still mints and signs."""

    from local_operator.operator import sign as sign_mod
    from local_operator.team_bridge import approvals

    _raise_ask(world.root, "bmsg_a")
    clock = SimpleNamespace(now=1000.0)
    monkeypatch.setattr(approvals, "_gate_clock", lambda: clock.now)
    real = sign_mod.sign_challenge
    calls = _dismissing_signer(monkeypatch)
    for _ in range(65):
        refused = _post_authorize(world, _items(("bmsg_a", "approve")))
        assert refused.json()["detail"]["code"] == "operator_signature_unavailable"
        clock.now += approvals.PROMPT_COOLDOWN_MAX_S + 1  # step past the cool-down
    assert len(calls) == 65
    assert approvals._challenges == {}
    monkeypatch.setattr(sign_mod, "sign_challenge", real)
    assert _post_authorize(world, _items(("bmsg_a", "approve"))).status_code == 200


def test_a_dismissal_starts_a_doubling_cooldown_that_success_resets(world, monkeypatch) -> None:
    """Exit 3: 10 s, doubling to a 60 s cap; refused calls never reach the signer."""

    from local_operator.operator import sign as sign_mod
    from local_operator.team_bridge import approvals

    _raise_ask(world.root, "bmsg_a")
    clock = SimpleNamespace(now=1000.0)
    monkeypatch.setattr(approvals, "_gate_clock", lambda: clock.now)
    real = sign_mod.sign_challenge
    dismissed = _dismissing_signer(monkeypatch)
    item = _items(("bmsg_a", "approve"))

    assert _post_authorize(world, item).status_code == 409  # dismissal 1 at t=1000
    clock.now += 5
    early = _post_authorize(world, item)
    assert early.status_code == 429, early.text
    assert early.json()["detail"]["code"] == "operator_prompt_cooldown"
    assert early.json()["detail"]["details"] == {"retry_after_s": 5}
    assert early.headers["retry-after"] == "5"
    assert len(dismissed) == 1  # the refused call never prompted
    clock.now += 6  # +11 s: prompts again (and is dismissed: window doubles)
    assert _post_authorize(world, item).json()["detail"]["code"] == "operator_signature_unavailable"
    windows = []
    for _ in range(4):
        clock.now += 1
        windows.append(
            _post_authorize(world, item).json()["detail"]["details"]["retry_after_s"] + 1
        )
        clock.now += windows[-1]
        assert _post_authorize(world, item).status_code == 409
    assert windows == [20, 40, 60, 60]

    monkeypatch.setattr(sign_mod, "sign_challenge", real)
    clock.now += 61
    assert _post_authorize(world, item).status_code == 200  # success resets
    _dismissing_signer(monkeypatch)
    assert _post_authorize(world, item).status_code == 409
    clock.now += 11
    assert _post_authorize(world, item).status_code == 409  # 10 s again, not 120


def test_each_prompt_is_published_on_the_desktop_feed(world, monkeypatch) -> None:
    """Exit 4: ``team_bridge_authorize`` frames name what the prompt is for."""

    frames: list[dict[str, Any]] = []
    world.http.app.state.desktop_feed = SimpleNamespace(
        publish_team_bridge_authorize=lambda payload: frames.append(payload)
    )
    _raise_ask(world.root, "bmsg_a")
    ok = _post_authorize(
        world, _items(("bmsg_a", "approve"), ("bmsg_zz", "deny")), client_ref="c0ffee"
    )
    assert ok.status_code == 200, ok.text
    assert [f["state"] for f in frames] == ["prompting", "signed"]
    first = frames[0]
    assert set(first) == {"authorize_id", "client_ref", "state", "items"}
    assert first["client_ref"] == "c0ffee"
    assert first["items"] == [
        {
            "message_id": "bmsg_a",
            "word": "approve",
            "sender": "priya@acme.example",
            "excerpt": "approve the spend?",
        },
        {"message_id": "bmsg_zz", "word": "deny", "sender": None, "excerpt": None},
    ]
    assert frames[1]["authorize_id"] == first["authorize_id"]
    flat = json.dumps(frames)
    for leaked in ("PRIVATE BODY", "PRIVATE RATIONALE", "body_hash", "rationale"):
        assert leaked not in flat

    _dismissing_signer(monkeypatch)
    _post_authorize(world, _items(("bmsg_a", "approve")))
    assert [f["state"] for f in frames[2:]] == ["prompting", "dismissed"]
    assert frames[2]["client_ref"] is None

    for bad in ("C0FFEE", "x" * 4, "a" * 33, "", 7):
        refused = _post_authorize(world, _items(("bmsg_a", "approve")), client_ref=bad)
        assert refused.status_code == 422, (bad, refused.text)
    assert len(frames) == 4  # a malformed body publishes nothing


def test_a_request_decided_elsewhere_is_settled_by_the_list(world) -> None:
    """Decided at the hub, no local intent at all: the ledger still settles it."""

    ask_a = _raise_ask(world.root, "bmsg_a")
    world.hub.pending = ["bmsg_b"]
    world.hub.ledger.insert(
        0,
        {
            "request_id": "bmsg_a",
            "state": "denied",
            "decided_by": {"account_id": ACC},
            "ask_ref": {"ask_id": ask_a, "question_id": "bmsg_a"},
        },
    )
    world.http.get(LIST, params={"expected_account_id": ACC}, headers=AUTH)
    assert _ask_status(world.root, COS, ask_a) == "withdrawn"
    assert _hub_calls(world.hub, "/v1/messaging/requests/decide") == []


# --- R1-2 / R1-3: per-item outcomes ------------------------------------------


def test_an_item_the_hub_omits_is_reported_sent(world) -> None:
    ask_a = _raise_ask(world.root, "bmsg_a")
    world.hub.batch_answer = {"bmsg_a": {"omit": "1"}}
    (result,) = _decide(world, _items(("bmsg_a", "approve"))).json()["result"]["results"]
    assert result["sent"] is True and result["error"] == "messaging_unavailable"
    assert result["ask_settled"] is False
    assert _ask_status(world.root, COS, ask_a) == "open"
    assert _intent_states(world.root) == {"bmsg_a": "pending"}


def test_a_transient_hub_error_leaves_the_ask_open(world) -> None:
    """Fails if every hub error were treated as terminal (R1-3)."""

    ask_a = _raise_ask(world.root, "bmsg_a")
    world.hub.batch_answer = {"bmsg_a": {"error": "bridge_disabled"}}
    (result,) = _decide(world, _items(("bmsg_a", "approve"))).json()["result"]["results"]
    assert result == {
        "message_id": "bmsg_a",
        "decision": "approve",
        "sent": True,
        "ask_settled": False,
        "error": "bridge_disabled",
    }
    assert _ask_status(world.root, COS, ask_a) == "open"
    assert _intent_states(world.root) == {"bmsg_a": "pending"}


def test_a_re_enrolled_owner_refuses_an_ask_raised_under_the_old_generation(world) -> None:
    """Fails if ``scope_matches_runtime`` were bypassed (R1-3)."""

    from dataclasses import replace

    _raise_ask(world.root, "bmsg_a")
    store = Store(world.root)
    regenerated = replace(world.owner, enrollment_generation=6)
    store.commit_enrollment(regenerated, OwnerReceipt("bridge_consent_v1", "ref-2"))
    store.update_designated_session(regenerated, COS)
    (result,) = _decide(world, _items(("bmsg_a", "approve"))).json()["result"]["results"]
    assert result["sent"] is False and result["error"] == "owner_changed"
    assert _hub_calls(world.hub, "/v1/messaging/requests/decide") == []


# --- R1-4 / S-3 / S-4 / Q1 ---------------------------------------------------


def test_the_list_pages_with_after_and_next_cursor(world) -> None:
    _raise_ask(world.root, "bmsg_a")
    _raise_ask(world.root, "bmsg_c")
    world.hub.page_size = 2
    first = world.http.get(
        LIST, params={"expected_account_id": ACC, "limit": "2"}, headers=AUTH
    ).json()["result"]
    assert [r["message_id"] for r in first["items"]] == ["bmsg_a", "bmsg_b"]
    assert first["pending_count"] == 1 and first["has_more"] is True
    assert first["next_cursor"] == "cur_page2"
    second = world.http.get(
        LIST,
        params={"expected_account_id": ACC, "limit": "2", "after": first["next_cursor"]},
        headers=AUTH,
    ).json()["result"]
    assert [r["message_id"] for r in second["items"]] == ["bmsg_c"]
    assert second["pending_count"] == 1 and second["has_more"] is False
    calls = _hub_calls(world.hub, "/v1/messaging/requests")
    assert calls[-1]["query"]["after"] == ["cur_page2"]


def test_the_list_drops_an_item_with_no_tenant(world) -> None:
    world.hub.tenantless = {"bmsg_b"}
    listed = world.http.get(LIST, params={"expected_account_id": ACC}, headers=AUTH)
    assert [r["message_id"] for r in listed.json()["result"]["items"]] == ["bmsg_a", "bmsg_c"]


def test_the_ledger_drops_a_row_for_another_tenant(world) -> None:
    world.hub.ledger.append({**world.hub.ledger[0], "request_id": "bmsg_x", "tenant_id": "tnt_9"})
    response = world.http.get(
        "/v1/desktop/team-bridge/decisions", params={"expected_account_id": ACC}, headers=AUTH
    )
    assert [r["request_id"] for r in response.json()["result"]["items"]] == ["bmsg_old"]


def test_decide_refusals_speak_of_a_decision(world) -> None:
    bad = world.http.post(
        DECIDE,
        json={"expected_account_id": ACC, "items": _items(("../x", "approve"))},
        headers=AUTH,
    )
    assert "decision" in bad.json()["detail"]["message"]
    assert "read" not in bad.json()["detail"]["message"]
    wrong = world.http.post(
        AUTHORIZE,
        json={"expected_account_id": "acc_9", "items": _items(("bmsg_a", "approve"))},
        headers=AUTH,
    )
    assert wrong.status_code == 409
    assert wrong.json()["detail"]["code"] == "account_mismatch"
    assert "decision" in wrong.json()["detail"]["message"]


def test_the_settled_by_stamp_matches_the_card_copy_key() -> None:
    from local_operator.asks.render import BATCH_DECIDE_VIA
    from local_operator.team_bridge.approvals import SETTLED_BY

    assert SETTLED_BY["via"] == BATCH_DECIDE_VIA
