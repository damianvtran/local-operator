"""The desktop's mesh surface: the peer catalogue, the networks tab, the five writes.

WHAT IS PINNED HERE, and each item is a failure the desktop round-1 review
REPRODUCED against a first-cut contract (mesh build plan, Addendum 2):

* **One peer row per DEVICE.** The relay's peer table is one entry per network
  MEMBERSHIP, so a device in two networks arrived twice and the sidebar drew two
  sections holding the same chats.
* **Flat locality fields on a row**, filled on EVERY row including local ones: the
  renderer groups from ``owner_device``/``owner_device_name`` and never from the
  nested transport block, and the merge rule ("an absent key is not a claim") is why
  local rows must be able to UNSET a stale remote mark.
* **The transfer body accepts ``request_id``** — the desktop request models are
  ``extra="forbid"``, so an undeclared key refuses every move — and an unconfirmed
  outcome is never reported as "nothing changed".
* **The token is never in an answer**: the invite route publishes a FILE PATH.
* **The typed confirmation is enforced by the route**, not by the dialog that
  rendered it.
* **A machine in no network costs nothing**: every read answers empty having opened
  no socket and created no directory, and ``include_peers`` absent is byte-identical
  to the pre-mesh answer.

Everything runs against a real store on a real temporary filesystem and through the
real ``errors()`` ladder and response models. The relay is stubbed at
``relay.control_request`` — the ONE seam every mesh read on this plane crosses — so
what is exercised is this backend's own mapping of a relay answer, and a fake that
is asked for an op the route should not have called FAILS rather than returning
nothing.
"""

from __future__ import annotations

import asyncio
import json
import os
from pathlib import Path
from types import SimpleNamespace
from typing import Any, cast

import pytest
import pytest_asyncio
from fastapi import FastAPI
from httpx import ASGITransport, AsyncClient

from local_operator.config import ConfigManager
from local_operator.network import store as network_store
from local_operator.network import types as network_types
from local_operator.resume import SessionRow
from local_operator.server.routes import capabilities as capabilities_module
from local_operator.server.routes import desktop_mesh, desktop_sessions
from local_operator.server.routes.capabilities import (
    capabilities as capabilities_endpoint,
)
from local_operator.server.utils.desktop_sessions import DesktopSessions
from local_operator.session.cleanup import mark_store

MINE = "a" * 12
OTHER = "b" * 12
PEER = "d_" + "9" * 32
PEER_TWO = "d_" + "8" * 32
NET_ONE = "n_" + "1" * 24
NET_TWO = "n_" + "2" * 24
REQUEST_ID = "6f1c2d3e-4a5b-4c6d-8e7f-0a1b2c3d4e5f"

# THE GATE'S PLACEHOLDER TOKEN. This backend only checks that the bearer matches
# what the env capability names, so the value is arbitrary; it is spelled as a
# WORD, the way the neighbouring desktop suites spell it
# (``test_desktop_attention.py``), so nothing has to guess whether a test fixture
# is a credential — and this suite never reads a real token from the environment
# or from this machine's store.
DESKTOP_TOKEN = "synthetic-desktop-token"

_MISSING = object()


class FakeRelay:
    """``relay.control_request``, recording every op and refusing the unexpected.

    AN OP THE ROUTE SHOULD NOT HAVE CALLED IS A FAILURE rather than a benign empty
    answer: several cells below assert that a refusal happened BEFORE the relay was
    asked (a wrong typed confirmation, an unknown network), and a fake that answered
    everything would let that regress silently.
    """

    def __init__(self, answers: dict[str, Any] | None = None) -> None:
        self.answers = answers or {}
        self.calls: list[tuple[str, dict[str, Any]]] = []

    def __call__(self, record: Any, op: str, *, timeout: float = 5.0, **fields: Any) -> Any:
        self.calls.append((op, fields))
        answer = self.answers.get(op, _MISSING)
        if answer is _MISSING:
            raise AssertionError(
                f"the route asked the relay for {op!r}, which this cell did not stub"
            )
        if isinstance(answer, Exception):
            raise answer
        if callable(answer):
            answer = answer(**fields)
        return {"op": "ack", "req": 1, "detail": answer}

    def ops(self) -> list[str]:
        return [op for op, _fields in self.calls]


@pytest_asyncio.fixture
async def mesh_api(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    """The three routers, a real store, and NO network until a test joins one."""
    for name in list(os.environ):
        if name.startswith("CMUX_"):
            monkeypatch.delenv(name)
    monkeypatch.setenv("HOME", str(tmp_path))
    monkeypatch.setenv("LOCAL_OPERATOR_CONFIG_DIR", str(tmp_path))
    monkeypatch.setenv("LOCAL_OPERATOR_DESKTOP_TOKEN", DESKTOP_TOKEN)
    mark_store(tmp_path / "sessions")
    app = FastAPI()
    app.state.config_manager = ConfigManager(tmp_path)
    app.state.desktop_sessions = DesktopSessions(tmp_path)
    app.include_router(desktop_sessions.router)
    app.include_router(desktop_mesh.router)
    app.include_router(capabilities_module.router)
    async with AsyncClient(
        transport=ASGITransport(app=app),
        base_url="http://localhost",
        headers={"Authorization": f"Bearer {DESKTOP_TOKEN}"},
    ) as client:
        yield client, tmp_path.resolve()


def _join(monkeypatch: pytest.MonkeyPatch, relay: FakeRelay | None = None) -> None:
    """Put a relay record and a network on the machine, so ``has_any_network`` is true."""
    from local_operator.network import relay as relay_mod

    monkeypatch.setattr(relay_mod, "control_request", relay or FakeRelay())
    monkeypatch.setattr(
        network_store,
        "find_own_relay",
        lambda root=None: SimpleNamespace(control_port=1, control_key="k"),
    )


def _record(root: Path, network_id: str, name: str, *, members: int = 1) -> None:
    """Write a real network record, so the Networks tab has something to read."""
    record = network_types.NetworkRecord(
        network_id=network_id,
        name=name,
        self_device_id=PEER,
        self_role="admin",
    )
    for index in range(members):
        record.members.append(
            network_types.MemberRecord(
                device_id=PEER if index == 0 else PEER_TWO,
                name="mine" if index == 0 else "other",
                role="admin" if index == 0 else "drive",
                capabilities=["list", "drive"] if index else ["admin"],
                endpoints=["10.0.0.1:4200"] if index else ["127.0.0.1:4200"],
                last_seen_at=1789400123.4 if index else None,
            )
        )
    network_store.save(record, root)


def _member(network: dict[str, Any], device_id: str) -> dict[str, Any]:
    for member in network["members"]:
        if member["device_id"] == device_id:
            return member
    raise AssertionError(f"{device_id} is not a member of {network['network_id']}")


def _peer_row(**overrides: Any) -> SessionRow:
    base: dict[str, Any] = {
        "id": OTHER,
        "mtime": 1789400000.0,
        "name": "build box chat",
        "live_state": "idle",
        "locality": "remote",
        "owner_device": PEER,
        "owner_device_name": "build-box",
        "reachable": True,
        "unreachable_reason": "",
    }
    base.update(overrides)
    return SessionRow(**base)


def _seed_session(root: Path, session_id: str, name: str = "local chat") -> None:
    path = root / "sessions" / session_id
    path.mkdir(parents=True, exist_ok=True)
    (path / "created_at.json").write_text("1700000000")
    (path / "desktop.json").write_text(json.dumps({"version": 1, "cwd": str(root)}))
    (path / "conversation.json").write_text(json.dumps({"name": name}))


# ---------------------------------------------------------------------------
# The capability keys, and the machine that has no mesh at all
# ---------------------------------------------------------------------------


def test_the_two_mesh_keys_are_advertised(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("LOCAL_OPERATOR_DESKTOP_TOKEN", DESKTOP_TOKEN)
    # ``cast`` rather than a subscript on the union the endpoint returns: the answer's
    # shape is pinned by the response model, and pyright reads ``result`` as optional.
    answer = cast(dict[str, Any], asyncio.run(capabilities_endpoint()).result)
    features = cast(dict[str, Any], answer["features"])
    assert features["peers"] == 1
    assert features["session_transfer"] == 1


@pytest.mark.asyncio
async def test_no_network_answers_empty_and_creates_nothing(mesh_api) -> None:
    """The zero-peer property, MEASURED: empty answers, and not one file written.

    ``network/networks`` must not exist afterwards — a desktop read that materialised
    a network root on a machine that has never joined one is the regression this
    asserts against, and it is invisible to a suite that only checks the payload.
    """
    client, root = mesh_api
    peers = await client.get("/v1/desktop/peers")
    assert peers.status_code == 200
    assert peers.json()["result"] == {"peers": [], "self_device_id": None, "degraded": []}

    networks = await client.get("/v1/desktop/networks")
    assert networks.status_code == 200
    assert networks.json()["result"] == {"networks": [], "self_device_id": None}

    assert not (root / "network").exists()


@pytest.mark.asyncio
async def test_a_listing_without_include_peers_is_unchanged(mesh_api) -> None:
    """Absent ⇒ today's answer, and the peer projection is never even imported."""
    client, root = mesh_api
    _seed_session(root, MINE)
    default = await client.get("/v1/desktop/sessions")
    assert default.status_code == 200
    body = default.json()["result"]
    assert [row["id"] for row in body["sessions"]] == [MINE]
    assert body["sessions"][0]["locality"] == "local"
    assert not (root / "network").exists()


# ---------------------------------------------------------------------------
# The peer catalogue: one row per device, deduped across networks
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_peers_collapse_two_memberships_into_one_row(
    mesh_api, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A device in two networks is ONE row, and the reason is glossed for a person."""
    client, root = mesh_api
    (root / "network" / "networks").mkdir(parents=True, exist_ok=True)
    relay = FakeRelay(
        {
            "net_peer_ls": [
                {
                    "device_id": PEER,
                    "name": "build-box",
                    "network_id": NET_ONE,
                    "reachable": False,
                    "reason": "connect_failed:ConnectionRefusedError",
                    "last_seen_at": None,
                },
                {
                    "device_id": PEER,
                    "name": "build-box",
                    "network_id": NET_TWO,
                    "reachable": True,
                    "reason": "",
                    "last_seen_at": 1789400123.4,
                },
            ]
        }
    )
    _join(monkeypatch, relay)
    monkeypatch.setattr(
        "local_operator.session.peer_rows.peer_session_rows",
        lambda root=None, **kwargs: (_peer_row(), _peer_row(id="c" * 12)),
    )
    response = await client.get("/v1/desktop/peers")
    assert response.status_code == 200, response.text
    result = response.json()["result"]
    assert len(result["peers"]) == 1, result
    row = result["peers"][0]
    assert row["device_id"] == PEER
    assert row["reachable"] is True, "one network reached it, so the device is reachable"
    assert row["unreachable_reason"] == ""
    assert row["last_seen_at"] == 1789400123.4, "the newest stamp any network recorded"
    assert row["session_count"] == 2, "counted from the sidebar's own row set"
    assert row["rtt_ms"] is None, "the transport publishes no latency, so no number is sent"


@pytest.mark.asyncio
async def test_an_unreachable_peer_carries_glossed_words(
    mesh_api, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The raw token never reaches a renderer; the sentence a person reads does."""
    client, root = mesh_api
    (root / "network" / "networks").mkdir(parents=True, exist_ok=True)
    _join(
        monkeypatch,
        FakeRelay(
            {
                "net_peer_ls": [
                    {
                        "device_id": PEER,
                        "name": "build-box",
                        "network_id": NET_ONE,
                        "reachable": False,
                        "reason": "connect_failed:ConnectionRefusedError",
                        "last_seen_at": None,
                    }
                ]
            }
        ),
    )
    monkeypatch.setattr(
        "local_operator.session.peer_rows.peer_session_rows", lambda root=None, **kwargs: ()
    )
    row = (await client.get("/v1/desktop/peers")).json()["result"]["peers"][0]
    assert row["reachable"] is False
    assert row["unreachable_reason"], "an unreachable peer must say why"
    assert "ConnectionRefusedError" not in row["unreachable_reason"]
    assert row["session_count"] == 0


@pytest.mark.asyncio
async def test_peers_refuses_when_a_network_exists_and_no_relay_answers(
    mesh_api, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A configured machine with no relay gets a NAMED remedy, not an empty list.

    Reporting ``peers: []`` there would say "you have no peers" about a device the
    user paired yesterday, which is the silent-empty answer every mesh surface here
    refuses by name.
    """
    client, root = mesh_api
    (root / "network" / "networks").mkdir(parents=True, exist_ok=True)
    monkeypatch.setattr(network_store, "find_own_relay", lambda root=None: None)
    response = await client.get("/v1/desktop/peers")
    assert response.status_code == 503, response.text
    detail = response.json()["detail"]
    assert detail["code"] == "relay_unavailable"
    assert "lop network start" in detail["message"]


# ---------------------------------------------------------------------------
# The networks tab: per network, and never collapsed
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_networks_keep_a_device_in_two_networks_as_two_memberships(
    mesh_api, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The graph draws one node with two edges, so the route must not collapse."""
    client, root = mesh_api
    (root / "network" / "networks").mkdir(parents=True, exist_ok=True)
    _record(root, NET_ONE, "home", members=2)
    _record(root, NET_TWO, "lab", members=2)
    relay = FakeRelay(
        {
            # ONE ENTRY PER NETWORK MEMBERSHIP, which is what the relay answers and
            # why the peer CATALOGUE has to collapse them. Here they must NOT be
            # collapsed: each network's own edge is the tab's subject.
            "net_peer_ls": [
                {
                    "device_id": PEER_TWO,
                    "name": "other",
                    "network_id": NET_ONE,
                    "reachable": True,
                    "reason": "",
                },
                {
                    "device_id": PEER_TWO,
                    "name": "other",
                    "network_id": NET_TWO,
                    "reachable": False,
                    "reason": "asked, and it did not answer",
                },
            ],
            # Answered per network, so the two rows cannot come back wearing one
            # network's id (the relay's own answer always describes the network it
            # was asked about).
            "net_show": lambda network="", **kwargs: {
                "network_id": network,
                "name": "home" if network == NET_ONE else "lab",
                "epoch": 3,
                "trust": "active",
                "members_detail": [
                    {
                        "device_id": PEER,
                        "name": "mine",
                        "role": "admin",
                        "capabilities": ["admin"],
                        "active": True,
                        "suspect": False,
                        "endpoints": ["127.0.0.1:4200"],
                    },
                    {
                        "device_id": PEER_TWO,
                        "name": "other",
                        "role": "drive",
                        "capabilities": ["list", "drive"],
                        "active": True,
                        "suspect": False,
                        "endpoints": ["10.0.0.1:4200"],
                    },
                ],
            },
        }
    )
    _join(monkeypatch, relay)
    response = await client.get("/v1/desktop/networks")
    assert response.status_code == 200, response.text
    result = response.json()["result"]
    assert result["self_device_id"] == PEER
    assert len(result["networks"]) == 2, "two networks are two rows"
    by_id = {net["network_id"]: net for net in result["networks"]}
    assert set(by_id) == {NET_ONE, NET_TWO}
    # The device in BOTH networks is a member of both, with its own reachability per
    # network rather than one collapsed verdict — the edge the flat catalogue cannot
    # express and the reason this route exists.
    for net in result["networks"]:
        assert any(member["device_id"] == PEER_TWO for member in net["members"]), net
    one = _member(by_id[NET_ONE], PEER_TWO)
    two = _member(by_id[NET_TWO], PEER_TWO)
    assert one["reachable"] is True
    assert one["reason"] == ""
    assert two["reachable"] is False
    assert two["reason"] == "asked, and it did not answer"
    # THIS DEVICE IS NEVER "unreachable": the relay's peer table skips the device that
    # owns it, so a join that read absence as a verdict drew this machine as dead with
    # "it did not answer" beside it (measured on a live single-device network).
    self_member = _member(by_id[NET_ONE], PEER)
    assert self_member["reachable"] is True
    assert self_member["reason"] == ""
    # The one field ``network_detail`` does not publish, read from the record the
    # relay just refreshed — and read PER MEMBER, not per network.
    assert _member(by_id[NET_ONE], PEER)["last_seen_at"] is None
    assert two["last_seen_at"] == 1789400123.4


@pytest.mark.asyncio
async def test_networks_still_answer_with_no_relay(
    mesh_api, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The tab lists what the device knows and says nobody was reachable."""
    client, root = mesh_api
    (root / "network" / "networks").mkdir(parents=True, exist_ok=True)
    _record(root, NET_ONE, "home", members=2)
    monkeypatch.setattr(network_store, "find_own_relay", lambda root=None: None)
    result = (await client.get("/v1/desktop/networks")).json()["result"]
    assert len(result["networks"]) == 1
    members = result["networks"][0]["members"]
    assert len(members) == 2, "the members come from this device's own record"
    # THE OTHER DEVICE is unreachable, and the tab says why in this file's own words: a
    # stopped relay is not a peer that stayed silent, and the remedy differs.
    other = _member(result["networks"][0], PEER_TWO)
    assert other["reachable"] is False
    assert "relay" in other["reason"], other
    # THIS DEVICE is not a probe result at all — the answer carries its own id.
    self_member = _member(result["networks"][0], PEER)
    assert self_member["reachable"] is True
    assert result["self_device_id"] == PEER


# ---------------------------------------------------------------------------
# include_peers on the list and the search
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_include_peers_adds_remote_rows_with_flat_fields(
    mesh_api, monkeypatch: pytest.MonkeyPatch
) -> None:
    client, root = mesh_api
    (root / "network" / "networks").mkdir(parents=True, exist_ok=True)
    _seed_session(root, MINE)
    _join(monkeypatch)
    monkeypatch.setattr(
        "local_operator.session.peer_rows.peer_session_rows",
        lambda root=None, **kwargs: (
            _peer_row(),
            _peer_row(
                id="c" * 12, reachable=False, unreachable_reason="asked, and it did not answer"
            ),
        ),
    )
    body = await client.get("/v1/desktop/sessions?include_peers=true")
    assert body.status_code == 200, body.text
    rows = body.json()["result"]["sessions"]
    assert [row["id"] for row in rows] == [MINE, OTHER, "c" * 12]
    remote = rows[1]
    assert remote["locality"] == "remote"
    assert remote["owner_device"] == PEER
    assert remote["owner_device_name"] == "build-box"
    assert remote["reachable"] is True
    assert remote["unreachable_reason"] == ""
    unreachable = rows[2]
    assert unreachable["reachable"] is False
    assert unreachable["unreachable_reason"] == "asked, and it did not answer"
    local = rows[0]
    # THE LOCAL ROW CARRIES THE SAME KEYS with the local answer: a client's merge is
    # "an absent key is not a claim", so the row a session moved away from must be
    # able to unset the mark rather than keep a stale one.
    for key, value in (
        ("owner_device", ""),
        ("owner_device_name", ""),
        ("unreachable_reason", ""),
        ("reachable", True),
    ):
        assert local[key] == value, key


@pytest.mark.asyncio
async def test_search_include_peers_matches_by_name_and_never_by_body(
    mesh_api, monkeypatch: pytest.MonkeyPatch
) -> None:
    client, root = mesh_api
    (root / "network" / "networks").mkdir(parents=True, exist_ok=True)
    _join(monkeypatch)
    monkeypatch.setattr(
        "local_operator.session.peer_rows.peer_session_rows",
        lambda root=None, **kwargs: (
            _peer_row(),
            _peer_row(id="c" * 12, name="something else"),
        ),
    )
    body = await client.get("/v1/desktop/sessions/search?q=build&include_peers=true")
    assert body.status_code == 200, body.text
    hits = body.json()["result"]["sessions"]
    assert [hit["id"] for hit in hits] == [OTHER]
    hit = hits[0]
    assert hit["rank"] == 0, "a name match is the catalogue's own name tier"
    assert hit["body_match"] is False, "a peer's transcript is not on this disk"
    assert hit["locality"] == "remote"
    assert hit["owner_device"] == PEER


# ---------------------------------------------------------------------------
# Invite and remove: the token is written, the name is typed
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_invite_answers_a_path_and_never_a_token(
    mesh_api, monkeypatch: pytest.MonkeyPatch
) -> None:
    client, root = mesh_api
    (root / "network" / "networks").mkdir(parents=True, exist_ok=True)
    _record(root, NET_ONE, "home")
    relay = FakeRelay(
        {
            "net_invite": {
                "invite_id": "inv_1",
                "path": str(root / "network" / "outbox" / "inv_1.invite"),
                "expires_at": 1789412400.0,
                "token": "SUPER-SECRET-TOKEN",
            }
        }
    )
    _join(monkeypatch, relay)
    response = await client.post(f"/v1/desktop/networks/{NET_ONE}/invite", json={"role": "drive"})
    assert response.status_code == 200, response.text
    result = response.json()["result"]
    assert result == {
        "token_path": str(root / "network" / "outbox" / "inv_1.invite"),
        "expires_at": 1789412400.0,
    }
    assert "SUPER-SECRET-TOKEN" not in response.text, "the token must not cross this API"


@pytest.mark.asyncio
async def test_invite_refuses_an_unknown_network_and_an_unknown_role(
    mesh_api, monkeypatch: pytest.MonkeyPatch
) -> None:
    client, root = mesh_api
    (root / "network" / "networks").mkdir(parents=True, exist_ok=True)
    _record(root, NET_ONE, "home")
    _join(monkeypatch)
    unknown = await client.post("/v1/desktop/networks/nowhere/invite", json={"role": "drive"})
    assert unknown.status_code == 404, unknown.text
    assert unknown.json()["detail"]["code"] == "unknown_network"
    bad_role = await client.post(f"/v1/desktop/networks/{NET_ONE}/invite", json={"role": "root"})
    assert bad_role.status_code == 422, bad_role.text


@pytest.mark.asyncio
async def test_remove_member_requires_the_network_name_typed_exactly(
    mesh_api, monkeypatch: pytest.MonkeyPatch
) -> None:
    client, root = mesh_api
    (root / "network" / "networks").mkdir(parents=True, exist_ok=True)
    _record(root, NET_ONE, "home")
    relay = FakeRelay({"net_member_rm": {"network_id": NET_ONE, "removed": PEER_TWO, "epoch": 4}})
    _join(monkeypatch, relay)

    wrong = await client.request(
        "DELETE",
        f"/v1/desktop/networks/{NET_ONE}/members/{PEER_TWO}",
        json={"confirm": "home-lab"},
    )
    assert wrong.status_code == 409, wrong.text
    assert wrong.json()["detail"]["code"] == "confirmation_mismatch"
    assert relay.ops() == [], "a wrong confirmation must not reach the relay at all"

    right = await client.request(
        "DELETE",
        f"/v1/desktop/networks/{NET_ONE}/members/{PEER_TWO}",
        json={"confirm": "home"},
    )
    assert right.status_code == 200, right.text
    assert right.json()["result"] == {"network_id": NET_ONE, "removed": PEER_TWO, "epoch": 4}
    assert relay.ops() == ["net_member_rm"]


@pytest.mark.asyncio
async def test_remove_member_refusal_keeps_the_relays_sentence(
    mesh_api, monkeypatch: pytest.MonkeyPatch
) -> None:
    client, root = mesh_api
    (root / "network" / "networks").mkdir(parents=True, exist_ok=True)
    _record(root, NET_ONE, "home")
    from local_operator.network.types import MeshRefusal

    _join(
        monkeypatch,
        FakeRelay({"net_member_rm": MeshRefusal("unknown_member", "d_x is not a member of home")}),
    )
    response = await client.request(
        "DELETE",
        f"/v1/desktop/networks/{NET_ONE}/members/d_x",
        json={"confirm": "home"},
    )
    assert response.status_code == 409, response.text
    assert response.json()["detail"] == {
        "code": "unknown_member",
        "message": "d_x is not a member of home",
    }


# ---------------------------------------------------------------------------
# Create on a peer
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_create_on_a_peer_mints_there_and_keeps_the_id(
    mesh_api, monkeypatch: pytest.MonkeyPatch
) -> None:
    client, root = mesh_api
    (root / "network" / "networks").mkdir(parents=True, exist_ok=True)
    relay = FakeRelay({"peer_session_create": {"session_id": OTHER, "admitted": False}})
    _join(monkeypatch, relay)
    response = await client.post(
        "/v1/desktop/sessions",
        json={"request_id": REQUEST_ID, "cwd": str(root), "peer": PEER},
    )
    assert response.status_code == 200, response.text
    created = response.json()["result"]
    assert created["session_id"] == OTHER
    assert created["binding"] == {"agent": None, "team": None}
    assert created["replayed"] is False
    op, fields = relay.calls[0]
    assert op == "peer_session_create"
    assert fields["peer"] == PEER
    assert not fields.get("cwd"), "a path on THIS machine means nothing on the peer"


@pytest.mark.asyncio
async def test_create_on_a_peer_with_no_relay_is_unconfirmed_and_replayable(
    mesh_api, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A create nobody could answer is 503 (unconfirmed), never 409 (nothing changed).

    Both directions matter: the renderer must not tell the user nothing was created
    when the frame may have landed, and a retry must replay this answer rather than
    mint a second conversation on the peer.
    """
    client, root = mesh_api
    (root / "network").mkdir(exist_ok=True)
    monkeypatch.setattr(network_store, "find_own_relay", lambda root=None: None)
    body = {"request_id": REQUEST_ID, "cwd": str(root), "peer": PEER}
    first = await client.post("/v1/desktop/sessions", json=body)
    assert first.status_code == 503, first.text
    assert first.json()["detail"]["code"] == "relay_unavailable"
    second = await client.post("/v1/desktop/sessions", json=body)
    assert second.status_code == 503, second.text
    assert second.json()["detail"] == first.json()["detail"], "the answer is replayed"


@pytest.mark.asyncio
async def test_create_on_a_peer_refuses_a_target_rather_than_dropping_it(
    mesh_api, monkeypatch: pytest.MonkeyPatch
) -> None:
    client, root = mesh_api
    (root / "network" / "networks").mkdir(parents=True, exist_ok=True)
    relay = FakeRelay({})
    _join(monkeypatch, relay)
    response = await client.post(
        "/v1/desktop/sessions",
        json={
            "request_id": REQUEST_ID,
            "cwd": str(root),
            "peer": PEER,
            "target": {"kind": "agent", "name": "nobody"},
        },
    )
    assert response.status_code == 422, response.text
    assert relay.ops() == [], "the peer must not be asked for a session whose agent is local"


@pytest.mark.asyncio
async def test_create_on_a_peer_that_stopped_answering_is_unconfirmed_not_retried(
    mesh_api, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The one refusal that must NOT release its request id: the session may exist."""
    client, root = mesh_api
    (root / "network" / "networks").mkdir(parents=True, exist_ok=True)
    from local_operator.network.types import MeshRefusal

    _join(
        monkeypatch,
        FakeRelay(
            {"peer_session_create": MeshRefusal("peer_unreachable", "d_x stopped answering")}
        ),
    )
    body = {"request_id": REQUEST_ID, "cwd": str(root), "peer": PEER}
    first = await client.post("/v1/desktop/sessions", json=body)
    assert first.status_code == 503, first.text
    assert first.json()["detail"]["code"] == "peer_unreachable"
    second = await client.post("/v1/desktop/sessions", json=body)
    assert second.status_code == 503, second.text


# ---------------------------------------------------------------------------
# Transfer: one answer, request_id accepted, unconfirmed is not "nothing changed"
# ---------------------------------------------------------------------------


def _move_result(**overrides: Any) -> dict[str, Any]:
    base: dict[str, Any] = {
        "ok": True,
        "session_id": MINE,
        "new_session_id": MINE,
        "mode": "move",
        "from_device": {"device_id": PEER, "name": "build-box"},
        "to_device": {"device_id": PEER, "name": "build-box"},
        "phase": "done",
        "phases": [
            {"phase": "prepared", "at": 1.0},
            {"phase": "handing_off", "at": 2.0},
            {"phase": "committed", "at": 3.0},
            {"phase": "done", "at": 4.0},
        ],
    }
    base.update(overrides)
    return base


@pytest.mark.asyncio
async def test_transfer_answers_one_document_and_accepts_request_id(
    mesh_api, monkeypatch: pytest.MonkeyPatch
) -> None:
    """``request_id`` is declared, so the body is not refused (Addendum 2, C)."""
    client, _root = mesh_api
    seen: list[dict[str, Any]] = []

    def fake_move(session_id: str, **kwargs: Any) -> dict[str, Any]:
        seen.append({"session_id": session_id, **kwargs})
        return _move_result()

    monkeypatch.setattr("local_operator.network.mobility.request_move", fake_move)
    response = await client.post(
        f"/v1/desktop/sessions/{MINE}/transfer",
        json={"to": PEER, "keep": False, "wait_s": 5, "request_id": REQUEST_ID},
    )
    assert response.status_code == 200, response.text
    result = response.json()["result"]
    assert [item["phase"] for item in result["phases"]] == [
        "prepared",
        "handing_off",
        "committed",
        "done",
    ]
    assert [item["progress"] for item in result["phases"]] == [0.25, 0.5, 0.75, 1.0]
    assert all(item["peer"] == PEER for item in result["phases"])
    assert result["locality"] == "remote"
    assert result["owner_device"] == PEER
    assert result["source_retired"] is True
    assert result["mode"] == "move"
    assert seen[0]["session_id"] == MINE
    assert seen[0]["to"] == PEER
    assert seen[0]["wait_s"] == 5.0


@pytest.mark.asyncio
async def test_transfer_refusal_is_a_409_with_the_moves_own_sentence(
    mesh_api, monkeypatch: pytest.MonkeyPatch
) -> None:
    client, _root = mesh_api
    sentence = "that session has a turn in flight; let it finish and try again"
    monkeypatch.setattr(
        "local_operator.network.mobility.request_move",
        lambda session_id, **kwargs: {
            "ok": False,
            "code": "busy",
            "message": sentence,
            "session_id": session_id,
            "phase_reached": None,
            "changed": False,
        },
    )
    response = await client.post(f"/v1/desktop/sessions/{MINE}/transfer", json={"to": PEER})
    assert response.status_code == 409, response.text
    assert response.json()["detail"] == {"code": "busy", "message": sentence}


@pytest.mark.asyncio
async def test_an_unconfirmed_move_is_never_reported_as_nothing_changed(
    mesh_api, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A deadline is a 503 saying the outcome is unknown, and a retry replays it."""
    client, _root = mesh_api
    sentence = "the owner did not finish the move in time; ask for its status"
    monkeypatch.setattr(
        "local_operator.network.mobility.request_move",
        lambda session_id, **kwargs: {
            "ok": False,
            "code": "deadline_exceeded",
            "message": sentence,
            "session_id": session_id,
            "phase_reached": "handing_off",
            "changed": True,
        },
    )
    body = {"to": PEER, "request_id": REQUEST_ID}
    first = await client.post(f"/v1/desktop/sessions/{MINE}/transfer", json=body)
    assert first.status_code == 503, first.text
    assert first.json()["detail"] == {"code": "deadline_exceeded", "message": sentence}
    second = await client.post(f"/v1/desktop/sessions/{MINE}/transfer", json=body)
    assert second.status_code == 503, second.text


@pytest.mark.asyncio
async def test_a_refused_move_leaves_its_request_id_usable(
    mesh_api, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A busy refusal moved nothing, so the user's second press must be able to run."""
    client, _root = mesh_api
    calls: list[str] = []

    def fake_move(session_id: str, **kwargs: Any) -> dict[str, Any]:
        calls.append(session_id)
        return {
            "ok": False,
            "code": "busy",
            "message": "busy right now",
            "session_id": session_id,
            "phase_reached": None,
            "changed": False,
        }

    monkeypatch.setattr("local_operator.network.mobility.request_move", fake_move)
    body = {"to": PEER, "request_id": REQUEST_ID}
    assert (
        await client.post(f"/v1/desktop/sessions/{MINE}/transfer", json=body)
    ).status_code == 409
    assert (
        await client.post(f"/v1/desktop/sessions/{MINE}/transfer", json=body)
    ).status_code == 409
    assert len(calls) == 2, "the claim was released, so the retry ran the move again"


@pytest.mark.asyncio
async def test_a_keep_transfer_never_claims_the_source_was_retired(
    mesh_api, monkeypatch: pytest.MonkeyPatch
) -> None:
    client, _root = mesh_api
    monkeypatch.setattr(
        "local_operator.network.mobility.request_move",
        lambda session_id, **kwargs: _move_result(
            mode="keep", new_session_id="d" * 12, to_device={"device_id": "local", "name": ""}
        ),
    )
    response = await client.post(
        f"/v1/desktop/sessions/{MINE}/transfer", json={"to": "local", "keep": True}
    )
    assert response.status_code == 200, response.text
    result = response.json()["result"]
    assert result["locality"] == "local"
    assert result["source_retired"] is False
    assert result["new_session_id"] == "d" * 12
    assert result["mode"] == "keep"


# ---------------------------------------------------------------------------
# Archive and delete on a peer
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_archive_on_a_peer_runs_on_the_owner_and_not_here(
    mesh_api, monkeypatch: pytest.MonkeyPatch
) -> None:
    client, root = mesh_api
    _seed_session(root, OTHER)
    calls: list[dict[str, Any]] = []

    def fake_owner(root_arg: Any, session_id: str) -> tuple[str, str]:
        return PEER, "build-box"

    def fake_lifecycle(root_arg: Any, session_id: str, **kwargs: Any) -> dict[str, Any]:
        calls.append(kwargs)
        return {"ok": True, "action": "archive", "session_id": session_id, "changed": True}

    monkeypatch.setattr("local_operator.server.utils.desktop_mesh.remote_owner", fake_owner)
    monkeypatch.setattr(
        "local_operator.server.utils.desktop_mesh.lifecycle_on_owner", fake_lifecycle
    )
    response = await client.post(f"/v1/desktop/sessions/{OTHER}/archive", json={"archived": True})
    assert response.status_code == 200, response.text
    assert response.json()["result"] == {"session_id": OTHER, "archived": True}
    assert calls == [{"action": "archive", "peer": PEER, "confirmed": False}]
    # NOTHING TOUCHED THIS DEVICE'S OWN ARCHIVE INDEX: the session is not here.
    from local_operator.session.archived import read_archived

    assert OTHER not in read_archived(root)


@pytest.mark.asyncio
async def test_delete_on_a_peer_keeps_the_refusal_shape(
    mesh_api, monkeypatch: pytest.MonkeyPatch
) -> None:
    client, root = mesh_api
    _seed_session(root, OTHER)
    sentence = "that session has a live runtime; stop it and try again"
    monkeypatch.setattr(
        "local_operator.server.utils.desktop_mesh.remote_owner",
        lambda root_arg, sid: (PEER, "build-box"),
    )
    monkeypatch.setattr(
        "local_operator.server.utils.desktop_mesh.lifecycle_on_owner",
        lambda root_arg, sid, **kwargs: {
            "ok": False,
            "code": "session_delete_refused",
            "message": sentence,
        },
    )
    response = await client.request(
        "DELETE", f"/v1/desktop/sessions/{OTHER}", json={"confirmed": True}
    )
    assert response.status_code == 409, response.text
    assert response.json()["detail"] == {"code": "session_delete_refused", "message": sentence}
    assert (root / "sessions" / OTHER).is_dir(), "a refused delete removes nothing"


@pytest.mark.asyncio
async def test_delete_on_a_peer_removes_it_there_and_forgets_it_here(
    mesh_api, monkeypatch: pytest.MonkeyPatch
) -> None:
    client, root = mesh_api
    _seed_session(root, OTHER)
    monkeypatch.setattr(
        "local_operator.server.utils.desktop_mesh.remote_owner",
        lambda root_arg, sid: (PEER, "build-box"),
    )
    monkeypatch.setattr(
        "local_operator.server.utils.desktop_mesh.lifecycle_on_owner",
        lambda root_arg, sid, **kwargs: {"ok": True, "action": "delete", "deleted": True},
    )
    response = await client.request(
        "DELETE", f"/v1/desktop/sessions/{OTHER}", json={"confirmed": True}
    )
    assert response.status_code == 200, response.text
    assert response.json()["result"] == {"session_id": OTHER, "deleted": True}
    assert (root / "sessions" / OTHER).is_dir(), "the OWNER deletes it; this device does not"


@pytest.mark.asyncio
async def test_an_unreachable_peers_conversation_refuses_in_the_tuis_own_words(
    mesh_api, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A visible remote row must not answer "not found" about the user's own chat.

    The row OPENS when the peer can be reached — mesh slice DB2, pinned against
    the pool and the ladder in ``tests/unit/server/test_desktop_remote_open.py``
    and over two real relays in ``tests/unit/network/test_remote_viewer.py``. What
    this cell keeps is the case that STILL refuses, and the point of it is the
    SENTENCE: it is ``remote_open.unreachable_peer_sentence``, the same composer
    the TUI's own pick refuses this exact state with, so one situation is never
    described two ways on two surfaces. The device is looked for only when no
    local directory holds the id, which is why a machine in no network is
    unaffected.
    """
    client, root = mesh_api
    (root / "network" / "networks").mkdir(parents=True, exist_ok=True)
    _join(monkeypatch)
    monkeypatch.setattr(
        # THE SEAM THE LOOKUP ITSELF USES (``remote_open.remote_row_for``), not the
        # rows producer: the decision "is this id a peer's row" is that function's,
        # and a stub one layer down would leave the cache unfilled and the answer
        # come from the ordinary miss path instead.
        "local_operator.session.remote_open.remote_row_for",
        # ONLY the peer's id is a peer's row, and it is the UNREACHABLE shape: the
        # second half of this test asserts that an id nobody holds is still the
        # ordinary 404.
        lambda session_id, root=None: (
            _peer_row(
                id=session_id,
                reachable=False,
                unreachable_reason="connect_failed:ConnectionRefusedError",
            )
            if session_id == OTHER
            else None
        ),
    )
    response = await client.get(f"/v1/desktop/sessions/{OTHER}")
    assert response.status_code == 409, response.text
    detail = response.json()["detail"]
    assert detail["code"] == "session_is_remote"
    assert "build-box" in detail["message"], detail
    assert "/network doctor build-box" in detail["message"], detail
    assert "ConnectionRefusedError" not in detail["message"], "the raw token reached the user"
    # AND A LOCAL ID IS UNTOUCHED: the ordinary miss is still the ordinary 404.
    missing = await client.get(f"/v1/desktop/sessions/{'f' * 12}")
    assert missing.status_code == 404, missing.text


@pytest.mark.asyncio
async def test_a_local_archive_still_runs_here(mesh_api, monkeypatch: pytest.MonkeyPatch) -> None:
    """The branch is additive: no peer owner ⇒ the existing path, unchanged."""
    client, root = mesh_api
    _seed_session(root, MINE)
    monkeypatch.setattr(
        "local_operator.server.utils.desktop_mesh.remote_owner", lambda root_arg, sid: None
    )
    response = await client.post(f"/v1/desktop/sessions/{MINE}/archive", json={"archived": True})
    assert response.status_code == 200, response.text
    from local_operator.session.archived import read_archived

    assert MINE in read_archived(root)
