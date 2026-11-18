"""Mesh pairing cards in the desktop's Mesh approvals: the routes, the attestation gate.

Operator request (Damian): "Make sure pairing does NOT require running terminal
commands, ideally it should be through the UI mesh approvals." These cells pin the
backend half and, above all, the security amendment A1/A4: a POST carrying only the
desktop bearer — what an agent holding that bearer can send — is 403
``pairing_needs_app`` and spends NOTHING (no attempt, no decision, no answer file),
because the app attestation is consumed BEFORE any comparison. An attestation is
single use and binds side, invite, nonce, act, the exact answer and the deadline; a
body that differs in any of them consumes nothing.

Real store on a temporary filesystem, the real router and ``errors()`` ladder. The
relay is stubbed at ``desktop_mesh._call`` (the plane's one relay client) and refuses
any op the route should not call; the attestation listener is the real
``AttestService`` with an injected ``PeerVerifier`` (as ``test_app_attest`` does), and
frames are fed through its own ``_frame`` path rather than written into the table.
"""

from __future__ import annotations

import json
import os
import time
from pathlib import Path
from typing import Any

import pytest
import pytest_asyncio
from fastapi import FastAPI
from httpx import ASGITransport, AsyncClient

from local_operator.asks import app_attest
from local_operator.config import ConfigManager
from local_operator.network import pairing_attest, store, types
from local_operator.server.routes import desktop_mesh_pairings
from local_operator.server.utils import desktop_mesh

TOKEN = "synthetic-desktop-token"
INVITE = "i_card01"
NETWORK = "n_" + "7" * 24
JOINER = "d_" + "b" * 32
CODE = "481926"


class OkVerifier:
    """The signed app, as far as the listener can tell (tests only)."""

    def verify(self, token: bytes) -> app_attest.Verdict:
        del token
        return app_attest.Verdict(True, "ok", 4242)


class RelayStub:
    """``desktop_mesh._call``: answers the two pairing ops, records every call."""

    def __init__(self, root: Path) -> None:
        self.root = root
        self.calls: list[tuple[str, dict[str, Any]]] = []
        self.confirm_reply: dict[str, Any] = {"status": "sent", "attempts_left": 3}

    def __call__(self, root: Any, op: str, *, timeout: float, **fields: Any) -> dict[str, Any]:
        del root, timeout
        self.calls.append((op, dict(fields)))
        if op == "net_pair_pending":
            return {"value": [p.to_json() for p in store.pending_pairings(self.root)]}
        if op == "net_pair_confirm":
            return dict(self.confirm_reply)
        raise AssertionError(f"the route called an unexpected relay op {op!r}")

    def confirms(self) -> list[dict[str, Any]]:
        return [fields for op, fields in self.calls if op == "net_pair_confirm"]


@pytest_asyncio.fixture
async def api(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    for name in list(os.environ):
        if name.startswith(("CMUX_", "LOP_")):
            monkeypatch.delenv(name)
    monkeypatch.setenv("HOME", str(tmp_path))
    monkeypatch.setenv("LOCAL_OPERATOR_CONFIG_DIR", str(tmp_path))
    monkeypatch.setenv("LOCAL_OPERATOR_DESKTOP_TOKEN", TOKEN)
    monkeypatch.setattr(app_attest, "supported", lambda: True)
    relay = RelayStub(tmp_path)
    monkeypatch.setattr(desktop_mesh, "_call", relay)
    service = pairing_attest.make_service(verifier=OkVerifier(), token_of=lambda _s: b"tok")
    app = FastAPI()
    app.state.config_manager = ConfigManager(tmp_path)
    app.state.mesh_pairing_attest = service
    app.include_router(desktop_mesh_pairings.router)
    async with AsyncClient(
        transport=ASGITransport(app=app),
        base_url="http://localhost",
        headers={"Authorization": f"Bearer {TOKEN}"},
    ) as client:
        yield client, tmp_path, relay, service, app


def _network(root: Path) -> None:
    record = types.NetworkRecord(network_id=NETWORK, name="home-net")
    record.invites.append(types.InviteRecord(invite_id=INVITE, role="drive"))
    store.save(record, root)


def _inviter(
    root: Path, *, stage: str = types.PAIR_STAGE_AWAITING_CONFIRMATION
) -> types.PendingPairing:
    _network(root)
    pending = types.PendingPairing(
        invite_id=INVITE,
        network_id=NETWORK,
        network_name="home-net",
        joiner_device_id=JOINER,
        joiner_name="laptop",
        sas=CODE,
        fingerprint="K7QM-3XPD",
        transcribed=CODE if stage == types.PAIR_STAGE_AWAITING_CONFIRMATION else "",
        expires_at=time.time() + 300,
        stage=stage,
        role="drive",
        nonce="a1" * 16,
    )
    store.save_pending_pairing(pending, root)
    return pending


def _joiner(root: Path) -> types.PendingJoin:
    store.ensure_pending_dir(root)
    pending = types.PendingJoin(
        invite_id=INVITE,
        network_id=NETWORK,
        network_name="home-net",
        inviter_device_id="d_" + "c" * 32,
        inviter_name="desktop",
        sas=CODE,
        fingerprint="K7QM-3XPD",
        pid=os.getpid(),  # a live parker
        expires_at=time.time() + 300,
        nonce="b2" * 16,
    )
    store.save_pending_join(pending, root)
    return pending


def _body(
    nonce: str, *, code: str = CODE, decision: str = "admit", deadline: int | None = None
) -> dict[str, Any]:
    return {
        "decision": decision,
        "code": code,
        "nonce": nonce,
        "deadline": deadline if deadline is not None else int(time.time()) + 60,
    }


async def _attest(
    service: Any, side: str, body: dict[str, Any], *, invite: str = INVITE
) -> dict[str, Any]:
    """The app's main process attesting exactly ``body`` over the real frame path."""
    frame = {
        "op": pairing_attest.ATTEST_OP,
        "side": side,
        "invite_id": invite,
        "nonce": body["nonce"],
        "act": pairing_attest.act_for(body["decision"]),
        "code": body["code"] if body["decision"] == "admit" else "",
        "shares": body.get("shares") or [],
        "deadline": body["deadline"],
    }
    return await service._frame(b"tok", json.dumps(frame).encode())  # noqa: SLF001


def _url(side: str, invite: str = INVITE) -> str:
    return f"/v1/desktop/mesh/pairings/{side}/{invite}/answer"


@pytest.mark.asyncio
async def test_rows_carry_both_sides_and_the_answer_availability(api) -> None:
    client, root, _relay, _service, _app = api
    _inviter(root)
    response = await client.get("/v1/desktop/mesh/pairings")
    assert response.status_code == 200, response.text
    rows = response.json()["result"]["pairings"]
    assert [r["id"] for r in rows] == [f"inviter:{INVITE}"]
    row = rows[0]
    assert row["stage"] == "awaiting_confirmation" and row["code"] == CODE
    assert row["attempts_left"] == 3 and row["nonce"] and row["seconds_left"] > 180
    assert row["can_answer_here"] is True and row["answer_unavailable"] == ""


@pytest.mark.asyncio
async def test_a_host_that_cannot_attest_says_so_and_refuses_every_answer(
    api, monkeypatch: pytest.MonkeyPatch
) -> None:
    client, root, relay, service, _app = api
    pending = _inviter(root)
    monkeypatch.setattr(app_attest, "supported", lambda: False)
    rows = (await client.get("/v1/desktop/mesh/pairings")).json()["result"]["pairings"]
    assert rows[0]["can_answer_here"] is False
    assert rows[0]["answer_unavailable"] == "unsupported_host"
    response = await client.post(_url("inviter"), json=_body(pending.nonce))
    assert response.status_code == 403
    assert response.json()["detail"]["code"] == "pairing_needs_app"
    assert relay.confirms() == []


@pytest.mark.asyncio
async def test_a_bearer_only_answer_is_403_and_spends_nothing(api) -> None:
    """A1: what an agent with only the desktop bearer can do. Both sides."""
    client, root, relay, _service, _app = api
    pending = _inviter(root)
    response = await client.post(_url("inviter"), json=_body(pending.nonce))
    assert response.status_code == 403, response.text
    detail = response.json()["detail"]
    assert detail["code"] == "pairing_needs_app" and "signed" in detail["message"]
    assert relay.confirms() == [], "a bearer-only answer reached the relay"
    assert store.pair_decision(INVITE, root) is None
    invite = store.load(NETWORK, root).invite(INVITE)
    assert invite is not None and invite.attempts == 0

    store.clear_pending_pairing(INVITE, root)
    joiner = _joiner(root)
    response = await client.post(_url("joiner"), json=_body(joiner.nonce, code="000000"))
    assert response.status_code == 403
    assert store.join_answer(INVITE, root) is None, "a bearer-only answer was written"


@pytest.mark.asyncio
async def test_a_matching_attestation_answers_once(api) -> None:
    client, root, _relay, service, _app = api
    joiner = _joiner(root)
    body = _body(joiner.nonce)
    assert (await _attest(service, "joiner", body)) == {"ok": True}
    response = await client.post(_url("joiner"), json=body)
    assert response.status_code == 200, response.text
    assert response.json()["result"]["status"] == "waiting_for_other_device"
    answer = store.join_answer(INVITE, root)
    assert answer is not None and answer.code == CODE and answer.answered_by == "desktop"
    rows = (await client.get("/v1/desktop/mesh/pairings")).json()["result"]["pairings"]
    assert rows[0]["stage"] == "waiting_for_other_device"
    # SINGLE USE: the same body again has nothing left to consume.
    again = await client.post(_url("joiner"), json=body)
    assert again.status_code == 403


@pytest.mark.asyncio
@pytest.mark.parametrize("change", ["code", "nonce", "side", "deadline", "decision"])
async def test_a_differing_body_consumes_nothing(api, change: str) -> None:
    client, root, relay, service, _app = api
    pending = _inviter(root)
    body = _body(pending.nonce)
    await _attest(service, "inviter" if change != "side" else "joiner", body)
    sent = dict(body)
    if change == "code":
        sent["code"] = "111111"
    elif change == "nonce":
        sent["nonce"] = "ff" * 16
    elif change == "deadline":
        sent["deadline"] = body["deadline"] + 1
    elif change == "decision":
        sent["decision"] = "decline"
    response = await client.post(_url("inviter"), json=sent)
    assert response.status_code in (403, 409), response.text
    assert relay.confirms() == []
    if change != "side":
        # The honest body still finds its attestation: nothing was consumed above.
        honest = await client.post(_url("inviter"), json=body)
        assert honest.status_code == 200, honest.text


@pytest.mark.asyncio
async def test_a_deadline_past_the_ceremony_is_refused_before_consuming(api) -> None:
    client, root, relay, service, _app = api
    pending = _inviter(root)
    late = _body(pending.nonce, deadline=int(pending.expires_at) + 120)
    await _attest(service, "inviter", late)
    response = await client.post(_url("inviter"), json=late)
    assert response.status_code == 403
    assert relay.confirms() == []


@pytest.mark.asyncio
async def test_the_inviter_answer_goes_to_the_relay_as_a_code_and_a_mismatch_is_409(api) -> None:
    client, root, relay, service, _app = api
    pending = _inviter(root)
    body = _body(pending.nonce, code="000000")
    await _attest(service, "inviter", body)
    relay.confirm_reply = {"status": "sas_mismatch", "attempts_left": 2}
    response = await client.post(_url("inviter"), json=body)
    assert response.status_code == 409, response.text
    detail = response.json()["detail"]
    assert detail["code"] == "sas_mismatch" and detail["attempts_left"] == 2
    assert "2 attempts left" in detail["message"]
    [sent] = relay.confirms()
    assert sent["code"] == "000000" and "matched" not in sent, "the relay must compare"
    assert sent["nonce"] == pending.nonce and sent["answered_by"] == "desktop"


@pytest.mark.asyncio
async def test_a_stage_one_card_is_not_ready_and_spends_nothing(api) -> None:
    client, root, relay, service, _app = api
    pending = _inviter(root, stage=types.PAIR_STAGE_AWAITING_JOINER)
    body = _body(pending.nonce)
    await _attest(service, "inviter", body)
    rows = (await client.get("/v1/desktop/mesh/pairings")).json()["result"]["pairings"]
    assert rows[0]["stage"] == "awaiting_joiner" and rows[0]["can_answer_here"] is False
    response = await client.post(_url("inviter"), json=body)
    assert response.status_code == 409
    assert response.json()["detail"]["code"] == "pairing_not_ready"
    assert relay.confirms() == []


@pytest.mark.asyncio
async def test_an_ended_ceremony_drops_its_attestation(api) -> None:
    client, root, _relay, service, _app = api
    joiner = _joiner(root)
    body = _body(joiner.nonce)
    await _attest(service, "joiner", body)
    store.clear_pending_join(INVITE, root)  # the window closed / the parker ended
    response = await client.post(_url("joiner"), json=body)
    assert response.status_code == 404
    assert response.json()["detail"]["code"] == "no_pending_pairing"
    assert not service.table._rows, "an ended ceremony left an attestation behind"  # noqa: SLF001


@pytest.mark.asyncio
async def test_joiner_card_strikes_decline_at_the_bound(api) -> None:
    client, root, _relay, service, _app = api
    joiner = _joiner(root)
    statuses = []
    for _ in range(3):
        body = _body(joiner.nonce, code="000000")
        await _attest(service, "joiner", body)
        response = await client.post(_url("joiner"), json=body)
        statuses.append((response.status_code, response.json()))
    assert [s for s, _ in statuses] == [409, 409, 200]
    assert statuses[0][1]["detail"]["attempts_left"] == 2
    assert statuses[2][1]["result"]["status"] == "refused"
    answer = store.join_answer(INVITE, root)
    assert answer is not None and answer.decision == "decline"


def test_the_attest_frame_validator_refuses_bad_shapes() -> None:
    good = {
        "op": pairing_attest.ATTEST_OP,
        "side": "joiner",
        "invite_id": INVITE,
        "nonce": "ab" * 8,
        "act": "respond",
        "code": CODE,
        "shares": ["b", "a"],
        "deadline": int(time.time()) + 30,
    }
    row = pairing_attest.validate_frame(dict(good))
    assert isinstance(row, app_attest.AttestedRow)
    assert row.answers["shares"] == ["a", "b"]
    for bad in (
        {"op": "ask_attest"},
        {"side": "both"},
        {"act": "revise"},
        {"code": "12345"},
        {"deadline": int(time.time()) - 1},
        {"deadline": True},
        {"nonce": ""},
    ):
        assert isinstance(pairing_attest.validate_frame({**good, **bad}), str), bad


def test_the_ask_contract_is_unchanged_by_the_generalisation() -> None:
    service = app_attest.AttestService("s1", verifier=OkVerifier())
    assert service._validator({"op": "mesh_pairing_attest"}) == "bad-op"  # noqa: SLF001
    row = service._validator(  # noqa: SLF001
        {"op": "ask_attest", "session_id": "s1", "ask_id": "a1", "act": "respond", "answers": {}}
    )
    assert isinstance(row, app_attest.AttestedRow) and row.scope == "s1" and row.target == "a1"
