"""H5c — the durable outbox: write-intent-before-dispatch, same ref on retry."""

from __future__ import annotations

import json

import httpx
import pytest

from local_operator.team_bridge import client as tb_client
from local_operator.team_bridge.client import BridgeRefusal, TeamBridgeClient
from local_operator.team_bridge.outbox import Outbox, OutboxRefused, send_from_outbox
from tests.unit.team_bridge.test_client import FakeStore, _err, _ok

TENANT = "ten_1"


@pytest.fixture(autouse=True)
def _destination_open(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(tb_client, "org_oauth_destination_allowed", lambda base: True)


def _client(handler) -> TeamBridgeClient:
    return TeamBridgeClient(
        None,
        base_url="https://hub.example/v1",
        store=FakeStore(),
        transport=httpx.MockTransport(handler),
    )


@pytest.mark.asyncio
async def test_a_retry_after_restart_replays_the_same_ref(tmp_path) -> None:
    """The intent is recorded BEFORE any I/O; an unreachable first attempt
    leaves it pending; the retry from a FRESH Outbox instance (the restart
    shape) reuses the SAME client_ref — asserted on the wire."""
    requests: list[httpx.Request] = []

    def flaky(request: httpx.Request) -> httpx.Response:
        requests.append(request)
        if len(requests) == 1:
            raise httpx.ConnectError("no route")
        return _ok({"message": {"id": "bmsg_9"}}, status=201)

    outbox = Outbox(tmp_path)
    intent = outbox.record_send(
        "acct:hub",
        to={"account_id": "acc_2"},
        type="note",
        text="hello",
    )
    with pytest.raises(BridgeRefusal):
        await send_from_outbox(_client(flaky), outbox, intent, tenant_id=TENANT)
    assert [row.intent_id for row in Outbox(tmp_path).pending("acct:hub")] == [intent.intent_id]

    reopened = Outbox(tmp_path)
    pending = reopened.pending("acct:hub")[0]
    assert pending.client_ref == intent.client_ref
    await send_from_outbox(_client(flaky), reopened, pending, tenant_id=TENANT)

    refs = [json.loads(request.content)["client_ref"] for request in requests]
    assert refs == [intent.client_ref, intent.client_ref]
    assert Outbox(tmp_path).pending("acct:hub") == []


@pytest.mark.asyncio
async def test_a_server_refusal_settles_the_intent_out_of_the_retry_pool(tmp_path) -> None:
    outbox = Outbox(tmp_path)
    intent = outbox.record_send(
        "acct:hub", to={"account_id": "acc_2"}, type="request", text="approve?"
    )
    client = _client(lambda request: _err(409, "idempotency_key_reused"))
    with pytest.raises(BridgeRefusal):
        await send_from_outbox(client, outbox, intent, tenant_id=TENANT)
    assert outbox.pending("acct:hub") == []


def test_a_settled_intent_is_never_re_settled(tmp_path) -> None:
    outbox = Outbox(tmp_path)
    intent = outbox.record_send("acct:hub", to={"account_id": "acc_2"}, type="note", text="x")
    outbox.confirm(intent.intent_id, "bmsg_1")
    with pytest.raises(OutboxRefused):
        outbox.refuse(intent.intent_id, "whatever")
    with pytest.raises(OutboxRefused):
        outbox.confirm("no-such-intent", "")


def test_a_damaged_outbox_refuses_instead_of_recreating(tmp_path) -> None:
    import sqlite3

    # A garbage file: sqlite itself refuses to read it — never recreated.
    path = tmp_path / "team_bridge" / "outbox.sqlite3"
    path.parent.mkdir(parents=True)
    path.write_bytes(b"not actually a database")
    with pytest.raises(sqlite3.DatabaseError):
        Outbox(tmp_path).pending("acct:hub")

    # A VALID database WITHOUT our schema generation: also refused, by the
    # module's own guard, before any DDL could run (re-running DDL is not
    # repair).
    other = tmp_path / "other"
    db = other / "team_bridge" / "outbox.sqlite3"
    db.parent.mkdir(parents=True)
    conn = sqlite3.connect(db)
    conn.execute("CREATE TABLE unrelated (x TEXT)")
    conn.commit()
    conn.close()
    with pytest.raises(OutboxRefused):
        Outbox(other).pending("acct:hub")
