"""The desktop approvals routes: pending requests, human batch decide, the ledger.

Real pieces wherever the property lives in them: the REAL team-bridge store
(``commit_enrollment`` + designation — the authority fence reads the document on
disk), the REAL ask queue writing a real ``asks.jsonl`` and index, the REAL
``TeamBridgeClient`` over real loopback HTTP to a stub hub that records every
request it receives, and the real ``require_desktop`` door. Only the
credential store (``FakeStore``) and the hub's domain are fixtures.

NO OS PROMPT (decision a-856f). Ordinary approvals are authorized by the
desktop bearer and the explicit selection only. The fixture still stages a REAL
operator key and anchor (``presence`` switchable, or no anchor at all) so the
tests prove the decide works on presence, file-only and keyless hosts alike,
and it replaces ``operator.sign.sign_challenge`` (the call that raises Touch ID)
and the authority-level read with recorders: any call is a regression.
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
    state = SimpleNamespace(presence=True, anchor=anchor, absent=False)

    def load(_uid: Any = None) -> Any:
        if state.absent:
            return trust.AnchorLoad(
                anchor=None, path=root / "anchor.json", root_owned=False, reason="", exists=False
            )
        return trust.AnchorLoad(
            anchor=replace(anchor, presence=state.presence),
            path=root / "anchor.json",
            root_owned=True,
            reason="",
            exists=True,
        )

    monkeypatch.setattr(operator_authority, "load_anchor", load)
    return state


def _signer_guard(monkeypatch: pytest.MonkeyPatch) -> list[str]:
    """Record every reach for the OS presence path; the approvals flow makes none.

    ``sign_challenge`` is the call that raises the Touch ID sheet (and on a
    keychain host, the keychain access); ``authority_level_load`` is the
    presence-level read the retired gate started with. Both are replaced with
    recorders that also FAIL the call, so a regression shows up as a non-empty
    log and an error rather than a silently passing prompt.
    """

    from local_operator import operator as operator_authority
    from local_operator.operator import sign as sign_mod

    calls: list[str] = []

    def signer(**_kwargs: Any) -> Any:
        calls.append("sign_challenge")
        raise AssertionError("ordinary approvals must never sign (a-856f)")

    def level(*_args: Any, **_kwargs: Any) -> Any:
        calls.append("authority_level_load")
        raise AssertionError("ordinary approvals must never read the presence level")

    monkeypatch.setattr(sign_mod, "sign_challenge", signer)
    monkeypatch.setattr(operator_authority, "authority_level_load", level)
    return calls


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
    signer_calls = _signer_guard(monkeypatch)
    app = FastAPI()
    app.include_router(routes.router)
    app.dependency_overrides[get_config_manager] = lambda: SimpleNamespace(config_dir=root)
    try:
        with TestClient(app) as http:
            yield SimpleNamespace(
                http=http,
                hub=hub_state,
                root=root,
                owner=owner,
                presence=presence,
                signer_calls=signer_calls,
            )
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


def _items(*pairs: tuple[str, str]) -> list[dict[str, str]]:
    return [{"message_id": mid, "decision": word} for mid, word in pairs]


def _authorize(world, items: list[dict[str, str]], *, account: str = ACC) -> dict[str, Any]:
    """The shipped desktop's confirm step: now a non-prompting shim (a-856f)."""

    response = world.http.post(
        AUTHORIZE, json={"expected_account_id": account, "items": items}, headers=AUTH
    )
    assert response.status_code == 200, response.text
    return response.json()["result"]["operator_proof"]


def _decide(
    world,
    items: list[dict[str, str]],
    *,
    proof: Any = None,
    account: str = ACC,
    headers: dict[str, str] = AUTH,
):
    """POST the batch. No proof by default (a-856f); ``proof="authorize"`` runs
    the shipped desktop's two-step flow through the compatibility shim."""

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
    assert result["pending_count"] == 1
    assert result["has_more"] is False and result["next_cursor"] is None
    # Gap 2: bmsg_b/c are pending at the hub but have no local ask (screening
    # rejected them or has not run yet), so they are not listed at all.
    assert [row["message_id"] for row in result["items"]] == ["bmsg_a"]
    first = result["items"][0]
    assert first["ask"] == {"session_id": COS, "ask_id": ask_a, "status": "open"}
    assert first["decidable"] is True
    # S-2: the ask card's own screened question text, so a batch is not blind.
    assert first["excerpt"] == "priya@acme.example: approve the spend?"
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
    # The settled requests leave the list once the hub stops serving them
    # (bmsg_c is still pending at the hub, but has no local ask: gap 2).
    world.hub.pending = ["bmsg_c"]
    listed = world.http.get(
        "/v1/desktop/team-bridge/requests", params={"expected_account_id": ACC}, headers=AUTH
    ).json()["result"]
    assert listed["items"] == []


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
    assert listed["pending_count"] == 0 and listed["items"] == []  # bmsg_b/c: no ask
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


# --- a-856f: ordinary approvals never raise an OS prompt ---------------------


@pytest.mark.parametrize("host", ["presence", "file-only", "no-anchor"])
def test_a_batch_decide_succeeds_on_every_host_with_no_signer_call(world, host) -> None:
    """Decision a-856f: no Touch ID, keychain or OS call, on any kind of host.

    A presence host used to prompt; a file-only or keyless host used to be
    refused ``operator_presence_required``. Now all three decide the same way
    — bearer + explicit selection — and the hub records the decision.
    """

    world.presence.presence = host == "presence"
    world.presence.absent = host == "no-anchor"
    ask_a = _raise_ask(world.root, "bmsg_a")
    response = _decide(world, _items(("bmsg_a", "approve")))
    assert response.status_code == 200, response.text
    assert response.json()["result"]["results"][0]["status"] == "approved"
    (call,) = _hub_calls(world.hub, "/v1/messaging/requests/decide")
    assert call["body"]["decisions"][0]["ask_ref"] == {"ask_id": ask_a, "question_id": "bmsg_a"}
    assert world.hub.ledger[0]["request_id"] == "bmsg_a"  # the hub's audit row
    assert _intent_states(world.root) == {"bmsg_a": "sent"}  # the local intent ledger
    assert world.signer_calls == []


def test_the_authorize_shim_answers_at_once_without_prompting(world) -> None:
    """The shipped desktop's two-step flow keeps working, with nothing signed."""

    from local_operator.team_bridge import approvals

    _raise_ask(world.root, "bmsg_a")
    _raise_ask(world.root, "bmsg_b")
    items = _items(("bmsg_b", "deny"), ("bmsg_a", "approve"))
    response = world.http.post(
        AUTHORIZE, json={"expected_account_id": ACC, "items": items}, headers=AUTH
    )
    assert response.status_code == 200, response.text
    result = response.json()["result"]
    assert set(result) == {"operator_proof", "selection_digest", "expires_s"}
    assert result["operator_proof"]["key_id"] == "unsigned"  # a placeholder, not a signature
    assert result["selection_digest"] == approvals.selection_digest(
        account_id=ACC,
        tenant_id="tnt_1",
        selection=[approvals.Selection(i["message_id"], i["decision"]) for i in items],
    )
    assert world.hub.calls == []  # authorizing calls nobody
    # Echoed verbatim, as v0.58.x does: accepted, ignored, and the batch goes out.
    decided = _decide(world, items, proof=result["operator_proof"])
    assert decided.status_code == 200, decided.text
    assert [r["status"] for r in decided.json()["result"]["results"]] == ["denied", "approved"]
    assert world.signer_calls == []


def test_a_proof_carries_no_authority_either_way(world) -> None:
    """A stale, foreign or reused proof neither blocks nor grants anything."""

    _raise_ask(world.root, "bmsg_a")
    stale = {"challenge": "ab" * 32, "sig": "30" + "ab" * 35, "key_id": "old-key"}
    response = _decide(world, _items(("bmsg_a", "approve")), proof=stale)
    assert response.status_code == 200, response.text
    # And no proof can make an item without a local ask decidable.
    unknown = _decide(world, _items(("bmsg_c", "approve")), proof=stale)
    assert unknown.json()["result"]["results"][0]["error"] == "no_open_ask"
    assert len(_hub_calls(world.hub, "/v1/messaging/requests/decide")) == 1


def test_an_ordinary_note_dm_produces_no_approvals_row_and_no_ask(world) -> None:
    """A note is a message, not a decision: it never reaches Approvals.

    The session never raises an ask for a stored note (``stored_kind_not_request``,
    pinned in ``tests/unit/team_bridge/test_h5_envelope_negatives.py``). Here the
    stub hub even serves the note on the requests read — the list still shows
    no row (no local ask), and naming it in a batch sends nothing.
    """

    from local_operator.team_bridge import approvals

    world.hub.pending = ["bmsg_note"]
    original = world.hub.answer

    def answer(method, path, query, body):
        out = original(method, path, query, body)
        if method == "GET" and path == "/v1/messaging/requests":
            for item in out["items"]:
                item["message"]["type"] = "note"
                item["message"].pop("request", None)
        return out

    world.hub.answer = answer
    assert "bmsg_note" not in approvals.bridge_asks_by_message(world.root)
    listed = world.http.get(LIST, params={"expected_account_id": ACC}, headers=AUTH)
    assert listed.status_code == 200, listed.text
    assert listed.json()["result"]["items"] == []
    assert listed.json()["result"]["pending_count"] == 0
    decided = _decide(world, _items(("bmsg_note", "approve")))
    assert decided.json()["result"]["results"][0]["error"] == "no_open_ask"
    assert _hub_calls(world.hub, "/v1/messaging/requests/decide") == []
    assert asks_store.index_asks(world.root) == []  # no ask was raised anywhere


def test_a_screening_rejected_request_has_no_decidable_row(world) -> None:
    """Gap 2: no local ask (rejected, or not screened yet) means no row at all."""

    ask_a = _raise_ask(world.root, "bmsg_a")  # the genuine, screened request
    # bmsg_b: the hub still holds it pending, but screening rejected it here,
    # so no ask was ever raised. bmsg_c: not screened yet — likewise no ask.
    listed = world.http.get(LIST, params={"expected_account_id": ACC}, headers=AUTH)
    result = listed.json()["result"]
    assert [r["message_id"] for r in result["items"]] == ["bmsg_a"]
    assert result["items"][0]["decidable"] is True
    assert result["items"][0]["ask"]["ask_id"] == ask_a
    assert result["pending_count"] == 1
    # Once screening admits bmsg_c and raises its ask, it appears, decidable.
    _raise_ask(world.root, "bmsg_c")
    again = world.http.get(LIST, params={"expected_account_id": ACC}, headers=AUTH).json()
    rows = {r["message_id"]: r["decidable"] for r in again["result"]["items"]}
    assert rows == {"bmsg_a": True, "bmsg_c": True}


def test_a_genuine_request_is_still_decidable_end_to_end(world) -> None:
    ask_a = _raise_ask(world.root, "bmsg_a")
    listed = world.http.get(LIST, params={"expected_account_id": ACC}, headers=AUTH).json()
    assert listed["result"]["items"][0]["decidable"] is True
    decided = _decide(world, _items(("bmsg_a", "deny")))
    assert decided.status_code == 200, decided.text
    assert decided.json()["result"]["results"][0]["status"] == "denied"
    assert _ask_status(world.root, COS, ask_a) == "withdrawn"
    assert world.signer_calls == []


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
    assert listed.json()["result"]["items"] == []  # bmsg_b/c have no local ask (gap 2)
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


def test_a_conflicting_reselection_is_refused_by_the_authorize_shim(world) -> None:
    """Exit 6a (P5): the other word is refused at the shim, as the old client expects."""

    _raise_ask(world.root, "bmsg_a")
    _fail_one_batch(world, "bmsg_a", "approve")
    refused = world.http.post(
        AUTHORIZE,
        json={"expected_account_id": ACC, "items": _items(("bmsg_a", "deny"))},
        headers=AUTH,
    )
    assert refused.status_code == 409, refused.text
    detail = refused.json()["detail"]
    assert detail["code"] == "earlier_decision_unconfirmed"
    assert detail["details"] == {"items": [{"message_id": "bmsg_a", "unsent_decision": "approve"}]}
    assert world.signer_calls == []


def test_decide_refuses_a_conflicting_selection_whole_with_zero_hub_calls(world) -> None:
    """Exit 6b: a direct decide (no shim) over a conflicting selection changes nothing."""

    _raise_ask(world.root, "bmsg_a")
    _raise_ask(world.root, "bmsg_b")
    _fail_one_batch(world, "bmsg_a", "approve")
    calls_before = len(_hub_calls(world.hub, "/v1/messaging/requests/decide"))
    refused = _decide(world, _items(("bmsg_a", "deny"), ("bmsg_b", "approve")))
    assert refused.status_code == 409, refused.text
    assert refused.json()["detail"]["code"] == "earlier_decision_unconfirmed"
    assert len(_hub_calls(world.hub, "/v1/messaging/requests/decide")) == calls_before
    assert _intent_states(world.root) == {"bmsg_a": "pending"}  # no bmsg_b row written


def test_a_list_row_carries_its_unsent_decision_and_only_that_word_sends(world) -> None:
    """Exit 9: the row shows the frozen word; selecting it again sends it."""

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


# --- S-2: the shim publishes no feed frame ------------------------------------


def _post_authorize(world, items, **extra):
    return world.http.post(
        AUTHORIZE, json={"expected_account_id": ACC, "items": items, **extra}, headers=AUTH
    )


def test_the_authorize_shim_publishes_no_feed_frame(world) -> None:
    """PR #4208 security round 1, S-2: a lone ``signed`` frame renders nothing in
    the shipped desktop, so the shim no longer claims a detection signal."""

    frames: list[dict[str, Any]] = []
    world.http.app.state.desktop_feed = SimpleNamespace(
        publish_team_bridge_authorize=lambda payload: frames.append(payload)
    )
    _raise_ask(world.root, "bmsg_a")
    ok = _post_authorize(world, _items(("bmsg_a", "approve")), client_ref="c0ffee")
    assert ok.status_code == 200, ok.text
    assert frames == []
    # ``client_ref`` is still grammar-checked, so a shipped client's body keeps
    # its contract and a malformed one is still refused.
    for bad in ("C0FFEE", "x" * 4, "a" * 33, "", 7):
        refused = _post_authorize(world, _items(("bmsg_a", "approve")), client_ref=bad)
        assert refused.status_code == 422, (bad, refused.text)
    assert world.signer_calls == []


# --- S-1: a desktop CARD answer to a Team Bridge request never prompts --------


def _card(world, *, ask_id: str, message_id: str, word: str):
    import asyncio

    return asyncio.run(
        routes.decide_request_from_card(world.root, ask_id=ask_id, message_id=message_id, word=word)
    )


def test_a_card_decision_settles_the_ask_with_no_signer_call(world) -> None:
    """The bearer path the answer route takes for a terminal-started session."""

    ask_a = _raise_ask(world.root, "bmsg_a")
    assert _card(world, ask_id=ask_a, message_id="bmsg_a", word="deny") == "decided"
    (call,) = _hub_calls(world.hub, "/v1/messaging/requests/decide")
    assert call["body"]["decisions"] == [
        {
            "message_id": "bmsg_a",
            "decision": "denied",
            "ask_ref": {"ask_id": ask_a, "question_id": "bmsg_a"},
            "decided_via": "bearer-only",
        }
    ]
    assert world.hub.ledger[0]["request_id"] == "bmsg_a"
    assert _ask_status(world.root, COS, ask_a) == "withdrawn"
    assert _intent_states(world.root) == {"bmsg_a": "sent"}
    assert world.signer_calls == []


def test_a_card_for_another_ask_keeps_the_runtime_door(world) -> None:
    """A card whose ask is not this machine's Team Bridge ask for that request
    (another ask id, or no ask) is not decided here: ``None``, zero hub calls."""

    _raise_ask(world.root, "bmsg_a")
    assert _card(world, ask_id="a-elsewhere", message_id="bmsg_a", word="approve") is None
    assert _card(world, ask_id="a-none", message_id="bmsg_c", word="approve") is None
    assert _hub_calls(world.hub, "/v1/messaging/requests/decide") == []


def test_a_card_refusal_is_a_card_sentence_and_nothing_is_decided_twice(world) -> None:
    ask_a = _raise_ask(world.root, "bmsg_a")
    world.hub.pending = ["bmsg_b"]  # decided elsewhere: the hub answers already-decided
    with pytest.raises(ValueError, match="already decided"):
        _card(world, ask_id=ask_a, message_id="bmsg_a", word="approve")
    assert _ask_status(world.root, COS, ask_a) == "withdrawn"  # terminal: settled
    assert world.signer_calls == []


def test_a_card_in_a_non_designated_session_is_refused(world) -> None:
    ask_o = _raise_ask(world.root, "bmsg_a", session_id="s-other")
    with pytest.raises(ValueError, match="no longer your Team Bridge session"):
        _card(world, ask_id=ask_o, message_id="bmsg_a", word="approve")
    assert _hub_calls(world.hub, "/v1/messaging/requests/decide") == []


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
    # bmsg_b has no local ask, so its page shows one row (gap 2) — the cursor
    # still follows the hub's page, not the visible row count.
    assert [r["message_id"] for r in first["items"]] == ["bmsg_a"]
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
    for message_id in ("bmsg_a", "bmsg_b", "bmsg_c"):
        _raise_ask(world.root, message_id)
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
