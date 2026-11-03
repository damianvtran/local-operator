"""#3199: allocation and filter failures through the real guarded HTTP path.

Only the provider's response server is synthetic. Requests traverse the native
registration/bind/dispatch ledger, model chain, charging transport and a loopback
HTTP socket. Distinct bodies prevent cached receipts from hiding actual sends.
"""

from __future__ import annotations

import json
import sqlite3
import threading
from contextlib import contextmanager
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

import httpx
import pytest

from local_operator.team_bridge.store import Store, StoreRefused, state_db_path
from tests.unit.team_bridge import test_screening_chain as chain
from tests.unit.team_bridge.test_screening import _item, _setup


@contextmanager
def _provider(monkeypatch, root, *, status=200, error="", fallback=False, script=None, on_request=None):
    observed = []

    class Handler(BaseHTTPRequestHandler):
        def do_POST(self):
            payload = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
            observed.append(payload)
            code, message = (script or {}).get(payload["model"], (status, error))
            if on_request is not None:
                on_request(payload)
            self.send_response(code)
            self.send_header("Content-Type", "application/json")
            self.end_headers()
            self.wfile.write(json.dumps({"error": message} if code != 200 else {"text": chain.DELIVER}).encode())

        def log_message(self, *_args):
            pass

    real_transport = httpx.AsyncHTTPTransport
    server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    chain._config(root, chain=["deepseek/deepseek-flash"] if fallback else None)
    chain._install_wire(monkeypatch, chain._Wire({}))

    class LoopbackTransport(real_transport):
        async def handle_async_request(self, request):
            # The outer production guard already checked the real model route;
            # only the physical destination is replaced with our local provider.
            request.url = httpx.URL(f"http://127.0.0.1:{server.server_port}/screen")
            return await super().handle_async_request(request)

    monkeypatch.setattr(httpx, "AsyncHTTPTransport", LoopbackTransport)
    try:
        yield observed
    finally:
        server.shutdown()
        server.server_close()
        thread.join(timeout=5)


def _identity_item(message_id, account="sender-a", kind="human"):
    item = _item(body=f"ordinary unique message {message_id}", message_id=message_id)
    item["sender"]["account_id"] = account
    item["message"]["sender"] = {"account_id": account, "kind": kind}
    return item


def _counts(root):
    with sqlite3.connect(state_db_path(root)) as conn:
        return conn.execute(
            "SELECT kind, sender_account_id, COUNT(*) FROM screening_attempts "
            "GROUP BY kind, sender_account_id ORDER BY kind, sender_account_id"
        ).fetchall()


def test_twenty_first_sender_a_is_held_b_still_screens_real_http(tmp_path, monkeypatch):
    store, owner = _setup(tmp_path)
    sender = ["sender-a"]
    monkeypatch.setattr(chain, "_item", lambda message_id: _identity_item(message_id, sender[0]))
    with _provider(monkeypatch, tmp_path) as requests:
        for index in range(20):
            result = chain._cycle(tmp_path, owner, f"bmsg_a{index}")
            assert result.disposition == "deliver"
        result = chain._cycle(tmp_path, owner, "bmsg_a21")
        assert (result.disposition, result.hold_code) == ("hold", "budget_exhausted")
        assert len(requests) == 20
        sender[0] = "sender-b"
        assert chain._cycle(tmp_path, owner, "bmsg_b1").disposition == "deliver"
        assert len(requests) == 21
    assert _counts(tmp_path) == [
        ("http", "sender-a", 20), ("http", "sender-b", 1),
        ("logical", "sender-a", 20), ("logical", "sender-b", 1),
    ]
    # Restart retains spend; human/agent kind changes do not create a new account.
    with pytest.raises(StoreRefused, match="sender hourly"):
        Store(tmp_path).reserve_screening_attempt(
            owner, input_proof={"sender_account_id": "sender-a", "sender_kind": "agent"}
        )


@pytest.mark.parametrize("status,error,advances", [
    (400, "data_inspection_failed: Input text data may contain inappropriate content", False),
    (401, "invalid credentials", True),
    (429, "rate limit exceeded", True),
    (503, "service unavailable", True),
])
def test_real_provider_input_filter_does_not_advance_outage_breaker(
    tmp_path, monkeypatch, status, error, advances
):
    store, owner = _setup(tmp_path)
    monkeypatch.setattr(chain, "_item", lambda message_id: _identity_item(message_id))
    with _provider(monkeypatch, tmp_path, status=status, error=error) as requests:
        result = chain._cycle(tmp_path, owner)
        assert result.disposition == "hold"
        assert len(requests) == 1
    state = store.screening_breaker_state(owner)
    failures = (state or {}).get("failures", 0)
    assert bool(failures) is advances, state
    with sqlite3.connect(state_db_path(tmp_path)) as conn:
        assert conn.execute("SELECT COUNT(*) FROM screening_answer_holds").fetchone()[0] == 0
    assert _counts(tmp_path) == [("http", "sender-a", 1), ("logical", "sender-a", 1)]


def test_unknown_legacy_spend_reduces_capacity_without_reset(tmp_path):
    store, owner = _setup(tmp_path)
    for _ in range(20):
        store.reserve_screening_attempt(owner)
    with pytest.raises(StoreRefused, match="sender hourly"):
        store.reserve_screening_attempt(owner, input_proof={
            "sender_account_id": "new-sender", "sender_kind": "human",
        })
    with pytest.raises(StoreRefused):
        Store(tmp_path).reserve_screening_attempt(owner)


@pytest.mark.parametrize("status", [401, 429, 503])
def test_fallback_hop_is_charged_to_same_sender_real_http(tmp_path, monkeypatch, status):
    store, owner = _setup(tmp_path)
    monkeypatch.setattr(chain, "_item", lambda message_id: _identity_item(message_id))
    with _provider(monkeypatch, tmp_path, fallback=True,
                   script={"gpt-5.4": (status, "provider unavailable")}) as requests:
        result = chain._cycle(tmp_path, owner)
        assert result.disposition == "deliver"
        assert [r["model"] for r in requests] == ["gpt-5.4", "deepseek-flash"]
    assert _counts(tmp_path) == [("http", "sender-a", 2), ("logical", "sender-a", 1)]


def test_sender_ledger_change_refuses_before_next_hop_http(tmp_path, monkeypatch):
    store, owner = _setup(tmp_path)
    monkeypatch.setattr(chain, "_item", lambda message_id: _identity_item(message_id))

    def corrupt(_payload):
        with sqlite3.connect(state_db_path(tmp_path)) as conn:
            conn.execute("UPDATE screening_attempts SET sender_account_id = 'sender-b' "
                         "WHERE kind = 'logical'")

    with _provider(monkeypatch, tmp_path, fallback=True, on_request=corrupt,
                   script={"gpt-5.4": (503, "service unavailable")}) as requests:
        result = chain._cycle(tmp_path, owner)
        assert result.disposition == "hold"
        assert len(requests) == 1, "altered sender ledger allowed another HTTP dispatch"
    assert _counts(tmp_path) == [("http", "sender-a", 1), ("logical", "sender-b", 1)]


def test_parallel_last_allowance_and_clock_rollback_keep_spend(tmp_path, monkeypatch):
    from concurrent.futures import ThreadPoolExecutor
    from local_operator.team_bridge import store as store_module

    store, owner = _setup(tmp_path)
    proof = {"sender_account_id": "sender-a", "sender_kind": "human"}
    for _ in range(19):
        store.reserve_screening_attempt(owner, input_proof=proof)

    def reserve(_index):
        try:
            return Store(tmp_path).reserve_screening_attempt(owner, input_proof=proof)
        except StoreRefused as error:
            return error.code

    with ThreadPoolExecutor(max_workers=2) as pool:
        results = list(pool.map(reserve, range(2)))
    assert sum(r == "budget_exhausted" for r in results) == 1
    assert _counts(tmp_path) == [("logical", "sender-a", 20)]
    now = store_module.time.time()
    monkeypatch.setattr(store_module.time, "time", lambda: now - 3600)
    with pytest.raises(StoreRefused, match="sender hourly"):
        Store(tmp_path).reserve_screening_attempt(owner, input_proof=proof)


def test_account_ceilings_shared_across_namespaces_and_unknown_types(tmp_path):
    from dataclasses import replace
    from local_operator.team_bridge.store import OwnerReceipt

    store, owner = _setup(tmp_path)
    for sender in ("sender-a", "sender-b"):
        for _ in range(20):
            store.reserve_screening_attempt(owner, input_proof={
                "sender_account_id": sender, "sender_kind": "human",
            })
    other = replace(owner, tenant_id="tnt_other", endpoint_id="bep_other", credential_id=9)
    store.commit_enrollment(other, OwnerReceipt("bridge_consent_v1", "other"))
    with pytest.raises(StoreRefused, match="account hourly"):
        Store(tmp_path).reserve_screening_attempt(other, input_proof={
            "sender_account_id": "sender-c", "sender_kind": "agent",
        })
    with pytest.raises(StoreRefused):
        Store(tmp_path).reserve_screening_attempt(other)
