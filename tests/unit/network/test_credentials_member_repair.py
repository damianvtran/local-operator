"""The member half of convergence: pull one key now, and schedule pulls on the tick.

WHY THESE CELLS. ``pull_now`` is the primitive the member's ``--repair`` calls and
the cadence's ``enqueue_placement_pull`` schedules; between them they are the
answer to "a share landed, why is the session blind to it". The properties pinned:

* a holder's copy is applied through the SAME ``_pull_blocking`` the tick uses —
  no second implementation, and the durable evidence says ``synced``;
* a device that is not a holder is refused WITHOUT a dial (the refusal is a
  placement fact, not something to ask the owner about);
* the cadence schedules ONE pull per owner and returns ``in_sync`` when the
  device holds nothing of that owner's, so a member with no share costs no dial;
* a member never grants anything: the repair surface can pull and validate, and
  every write it makes is local.

The link is a scripted fake: a real two-relay rig would measure the same branches
and this is the cheap half of the design's test matrix.
"""

from __future__ import annotations

import threading
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest

from local_operator.network import store, types, wire
from local_operator.network.credentials import mcp_state
from local_operator.network.credentials import placement as placement_mod
from local_operator.network.credentials import sync
from local_operator.providers.auth_store import AuthStore

SELF = "d_" + "1" * 32
PEER = "d_" + "2" * 32
NETWORK = "n_" + "a" * 24
KEY = "stub-member-copy"
VALUE = "member-copy-value"


def _payload(value: str) -> dict[str, Any]:
    return {"type": "api_key", "source": "login", "key": value}


class FakeLink:
    def __init__(self) -> None:
        self.frames: list[dict[str, Any]] = []
        self.copy_value = _payload(VALUE)
        self.gen = 1

    def request(self, frame: dict[str, Any], timeout: float | None = None) -> dict[str, Any]:
        self.frames.append(dict(frame))
        if frame.get("kind") == "copy":
            return {
                "detail": {
                    "kind": "copy",
                    "key": frame.get("key"),
                    "gen": self.gen,
                    "value": self.copy_value,
                    "digest": sync.fingerprint(self.copy_value),
                    "value_state": sync.VALUE_STATE_PRESENT,
                }
            }
        return {"detail": {"kind": "ack"}}


class FakeServer:
    def __init__(self, link: FakeLink) -> None:
        self.link = link
        self.reqs = 0

    def _ensure_link(self, device: str) -> FakeLink:
        return self.link

    def _ensure_link_with_reason(self, device: str) -> tuple[FakeLink, str]:
        return self.link, ""

    def _next_relay_req(self) -> int:
        self.reqs += 1
        return self.reqs


def _record(*, self_active: bool = True) -> types.NetworkRecord:
    record = types.NetworkRecord(
        network_id=NETWORK,
        name="home-net",
        self_device_id=SELF,
        self_role="drive",
        self_capabilities=sorted(types.capabilities_for_role("drive")),
        epoch=1,
    )
    record.members.append(
        types.MemberRecord(
            device_id=SELF,
            public_key=wire.b64u(b"a" * 32),
            role="drive",
            capabilities=sorted(types.capabilities_for_role("drive")),
            added_via="invite",
            lifecycle="active" if self_active else "expired",
            removed_at=None if self_active else 1.0,
        )
    )
    record.members.append(
        types.MemberRecord(
            device_id=PEER,
            public_key=wire.b64u(b"b" * 32),
            role="admin",
            capabilities=sorted(types.capabilities_for_role("admin")),
            added_via="invite",
        )
    )
    return record


def _place(
    root: Path, key: str = KEY, *, kind: str = "api-key-static", holder: bool = True
) -> None:
    with placement_mod.mutate(NETWORK, root=root, self_device=PEER) as document:
        document.declare(
            key,
            owner_device=PEER,
            owner_device_name="macbook",
            provider="stub-provider",
            kind=kind,
            by=PEER,
        )
        if holder:
            document.grant(key, SELF, by=PEER)


def _engine(root: Path, link: FakeLink) -> sync.SyncEngine:
    return sync.SyncEngine(
        FakeServer(link),
        root=root,
        self_device=SELF,
        self_device_name="laptop",
        auth_store_factory=lambda: AuthStore(db_path=root / "auth.db", config_dir=root),
    )


def test_pull_now_applies_a_holders_copy_and_records_synced(root: Path) -> None:
    store.save(_record(), root)
    _place(root)
    link = FakeLink()
    engine = _engine(root, link)

    outcome = engine.pull_now(KEY)

    assert outcome == {"ok": True, "state": "synced", "code": ""}
    assert any(frame.get("kind") == "copy" for frame in link.frames)
    assert any(frame.get("kind") == "ack" for frame in link.frames)
    rows = AuthStore(db_path=root / "auth.db", config_dir=root).list_credentials("stub-provider")
    assert [row.data["key"] for row in rows] == [VALUE]
    evidence = mcp_state.load(NETWORK, root)[KEY]
    assert evidence["state"] == "synced"
    assert evidence["owner_device"] == PEER


def test_pull_now_without_a_holder_row_refuses_without_dialling(root: Path) -> None:
    store.save(_record(), root)
    _place(root, holder=False)
    link = FakeLink()
    engine = _engine(root, link)

    outcome = engine.pull_now(KEY)

    assert outcome["ok"] is False
    assert (outcome["state"], outcome["code"]) == ("revoked", "not_a_holder")
    assert link.frames == [], "a placement refusal must not become a dial"


def test_pull_now_of_a_key_this_device_owns_is_a_local_no_op(root: Path) -> None:
    store.save(_record(), root)
    with placement_mod.mutate(NETWORK, root=root, self_device=SELF) as document:
        document.declare(
            KEY,
            owner_device=SELF,
            owner_device_name="laptop",
            provider="stub-provider",
            kind="api-key-static",
            by=SELF,
        )
    link = FakeLink()
    engine = _engine(root, link)

    outcome = engine.pull_now(KEY)

    assert (outcome["ok"], outcome["code"]) == (False, "local_owner")
    assert link.frames == []


def test_placement_pull_is_scheduled_once_per_owner_and_is_inert_when_empty(
    root: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    store.save(_record(), root)
    engine = _engine(root, FakeLink())

    # Nothing of the peer's is held here: the cadence step must not dial.
    assert engine.enqueue_placement_pull(PEER) == "in_sync"

    _place(root)
    blocked = threading.Event()
    monkeypatch.setattr(engine, "_pull_placement_blocking", lambda owner, keys: blocked.wait(5))

    assert engine.enqueue_placement_pull(PEER) == "scheduled"
    assert engine.enqueue_placement_pull(PEER) == "in_flight"
    blocked.set()


def test_a_member_repair_grants_nothing(root: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """The CLI's member path may pull and validate; every write stays local."""
    from local_operator.network import cli as net_cli

    calls: list[tuple[str, dict[str, Any]]] = []

    def relay_answer(op: str, *, timeout: float = 5.0, **fields: Any) -> dict[str, Any]:
        calls.append((op, fields))
        return {"ok": True, "state": "synced", "code": ""}

    monkeypatch.setattr(net_cli, "_relay_answer", relay_answer)
    servers = [{"name": "slack", "url": "https://slack.test/mcp", "refs": [{"id": "SLACK_TOKEN"}]}]
    net_cli._member_repair_receipt(root, servers)

    assert [op for op, _ in calls] == ["credential_placement"]
    assert all(op != "credential_grant" for op, _ in calls)


def test_pull_copy_sync_applies_through_the_shared_apply_path(root: Path) -> None:
    """The CLI's direct dialect, end to end: ask the owner, apply, record.

    This is the fallback's real work, so it is driven through the shipped
    ``MeshCredentialClient`` with a scripted link — the same fake the relay-side
    pull uses — and the store/evidence assertions are the ones that prove the
    value landed rather than the sentence being right.
    """
    from local_operator.network.credentials import client as client_mod
    from local_operator.network.credentials.placement import PlacementDocument

    store.save(_record(), root)
    _place(root)
    link = FakeLink()
    client = client_mod.MeshCredentialClient(
        root=root,
        self_device=SELF,
        network_id=NETWORK,
        placement=PlacementDocument.load(NETWORK, root, self_device=SELF),
        relay=FakeServer(link),
    )

    outcome = client.pull_copy_sync(KEY)

    assert outcome == {"ok": True, "state": "synced", "code": ""}
    assert any(frame.get("kind") == "copy" for frame in link.frames)
    rows = AuthStore(db_path=root / "auth.db", config_dir=root).list_credentials("stub-provider")
    assert [row.data["key"] for row in rows] == [VALUE]
    assert mcp_state.load(NETWORK, root)[KEY]["state"] == "synced"


def test_pull_copy_sync_never_dials_for_an_owned_or_unshared_key(root: Path) -> None:
    from local_operator.network.credentials import client as client_mod
    from local_operator.network.credentials.placement import PlacementDocument

    store.save(_record(), root)
    with placement_mod.mutate(NETWORK, root=root, self_device=SELF) as document:
        document.declare(
            KEY,
            owner_device=SELF,
            owner_device_name="laptop",
            provider="stub-provider",
            kind="api-key-static",
            by=SELF,
        )
    link = FakeLink()
    client = client_mod.MeshCredentialClient(
        root=root,
        self_device=SELF,
        network_id=NETWORK,
        placement=PlacementDocument.load(NETWORK, root, self_device=SELF),
        relay=FakeServer(link),
    )

    outcome = client.pull_copy_sync(KEY)

    assert (outcome["ok"], outcome["code"]) == (False, "local_owner")
    assert link.frames == []
    # A cold, never-shared key is not a revocation and writes no evidence.
    other = "stub-never-shared"
    assert (client.pull_copy_sync(other)["ok"], client.pull_copy_sync(other)["code"]) == (
        False,
        "not_a_holder",
    )
    assert mcp_state.load(NETWORK, root) == {}


def test_the_cli_falls_back_when_the_relay_lacks_the_pull_op(
    root: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """An old relay is not a dead end: the direct dialect runs instead."""
    from local_operator.network import cli as net_cli
    from local_operator.network.credentials.types import credential_key_for_secret
    from local_operator.network.types import MeshRefusal

    store.save(_record(), root)
    _place(root, credential_key_for_secret("SLACK_TOKEN"), kind="store-secret")
    from local_operator.network import identity as identity_mod

    monkeypatch.setattr(
        identity_mod,
        "load",
        lambda *args, **kwargs: SimpleNamespace(device_id=SELF, name="laptop"),
    )

    def no_op(op: str, *, timeout: float = 5.0, **fields: Any) -> dict[str, Any]:
        if op == "credential_sync_pull":
            raise MeshRefusal("unknown_op", "this relay does not know that action")
        return {"kind": "ack", "changed": [], "owners": 0}

    called: list[str] = []
    monkeypatch.setattr(net_cli, "_relay_answer", no_op)
    monkeypatch.setattr(
        net_cli,
        "_direct_copy_pull",
        lambda root_arg, key: (called.append(key) or {"ok": True, "state": "synced", "code": ""}),
    )

    receipt = net_cli._member_repair_receipt(
        root, [{"name": "slack", "url": "", "refs": [{"id": "SLACK_TOKEN"}]}]
    )

    assert receipt["slack"]["fallback"] is True
    assert [row["ok"] for row in receipt["slack"]["pulled"]] == [True]
    assert called == [credential_key_for_secret("SLACK_TOKEN")]


def test_the_cli_prefers_the_relay_when_it_answers(
    root: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from local_operator.network import cli as net_cli
    from local_operator.network.credentials.types import credential_key_for_secret

    store.save(_record(), root)
    _place(root, credential_key_for_secret("SLACK_TOKEN"), kind="store-secret")
    from local_operator.network import identity as identity_mod

    monkeypatch.setattr(
        identity_mod,
        "load",
        lambda *args, **kwargs: SimpleNamespace(device_id=SELF, name="laptop"),
    )

    calls: list[str] = []

    def answers(op: str, *, timeout: float = 5.0, **fields: Any) -> dict[str, Any]:
        calls.append(op)
        if op == "credential_sync_pull":
            return {"ok": True, "state": "synced", "code": ""}
        return {"kind": "ack"}

    monkeypatch.setattr(net_cli, "_relay_answer", answers)
    monkeypatch.setattr(
        net_cli,
        "_direct_copy_pull",
        lambda *args, **kwargs: pytest.fail("the fallback dialled over a working relay"),
    )

    receipt = net_cli._member_repair_receipt(
        root, [{"name": "slack", "url": "", "refs": [{"id": "SLACK_TOKEN"}]}]
    )

    assert "fallback" not in receipt["slack"]
    assert calls == ["credential_placement", "credential_sync_pull"]


def test_a_successful_member_validation_records_synced_with_the_owner(
    root: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The positive is recorded, not merely "not refused".

    The status row then says "synced — borrows from macbook" instead of falling
    back to a bare placement answer, and a later owner-side expiry overwrites it.
    """
    from types import SimpleNamespace

    from local_operator.network import cli as net_cli
    from local_operator.network.credentials import mcp_state
    from local_operator.network.credentials.types import credential_key_for_mcp

    url = "https://linear.test/mcp"
    key = credential_key_for_mcp(url)
    store.save(_record(), root)
    monkeypatch.setattr(
        "local_operator.network.identity.load",
        lambda *a, **k: SimpleNamespace(device_id=SELF, name="laptop"),
    )
    _place(root, key, kind="mcp-rotating")

    def relay_answer(op: str, *, timeout: float = 5.0, **fields: Any) -> dict[str, Any]:
        if op == "credential_grant":
            # A REAL grant carries the token it just minted; a shape-only reply
            # is the failed validation the gitlab test pins.
            return {"kind": "grant", "access_token": "granted-bearer"}
        return {"ok": True, "state": "synced", "code": ""}

    monkeypatch.setattr(net_cli, "_relay_answer", relay_answer)
    servers = [{"name": "linear", "url": url, "refs": [], "shape": "oauth"}]
    receipt = net_cli._member_repair_receipt(root, servers)
    grant = receipt["linear"]["grant"]
    assert grant["ok"] is True and grant["code"] == ""
    assert grant["owner_name"] == "macbook"

    entry = mcp_state.load(NETWORK, root)[key]
    assert entry["state"] == "synced"
    assert entry["owner_name"] == "macbook"


def test_a_refused_member_validation_persists_the_verdict_for_every_surface(
    root: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The CLI row is not enough: ``/mcp`` reads ``server_states``.

    A refused borrow must be written to the evidence document, or the very next
    status pass says "synced — borrows from macbook" again even though nothing
    was proved to work (measured on cloud-node-1).
    """
    from types import SimpleNamespace

    from local_operator.mcp.config import MCPAuthConfig, MCPHttpServerConfig
    from local_operator.mcp.credential_status import server_states
    from local_operator.network import cli as net_cli
    from local_operator.network.credentials import mcp_state
    from local_operator.network.credentials.types import credential_key_for_mcp

    url = "https://cloudflare.test/mcp"
    key = credential_key_for_mcp(url)
    store.save(_record(), root)
    _place(root, key, kind="mcp-rotating")
    monkeypatch.setattr(
        "local_operator.network.identity.load",
        lambda *a, **k: SimpleNamespace(device_id=SELF, name="laptop"),
    )
    monkeypatch.setattr(
        net_cli,
        "_relay_answer",
        lambda op, **fields: {"kind": "refusal", "code": "interactive_required"},
    )

    receipt = net_cli._member_repair_receipt(
        root, [{"name": "cloudflare", "url": url, "refs": [], "shape": "oauth"}]
    )
    assert receipt["cloudflare"]["grant"]["ok"] is False
    assert receipt["cloudflare"]["grant"]["owner_name"] == "macbook"

    evidence = mcp_state.load(NETWORK, root)[key]
    assert evidence["state"] == "expired"
    assert evidence["code"] == "interactive_required"

    rows = server_states(
        str(root),
        base=root,
        configs={
            "cloudflare": MCPHttpServerConfig(url=url, auth=MCPAuthConfig(type="oauth")),
        },
    )
    assert rows["cloudflare"]["state"] == "expired", rows
    assert "macbook" in rows["cloudflare"]["reason"]


@pytest.mark.parametrize(
    "reply",
    [
        {"kind": "grant", "access_token": ""},
        {"kind": "grant"},
    ],
)
def test_a_grant_with_no_token_is_a_failed_validation(
    root: Path, monkeypatch: pytest.MonkeyPatch, reply: dict[str, Any]
) -> None:
    """gitlab: ``kind: grant`` with nothing to serve must never be ``synced``."""
    from types import SimpleNamespace

    from local_operator.network import cli as net_cli
    from local_operator.network.credentials import mcp_state
    from local_operator.network.credentials.types import credential_key_for_mcp

    url = "https://gitlab.test/mcp"
    key = credential_key_for_mcp(url)
    store.save(_record(), root)
    _place(root, key, kind="mcp-rotating")
    monkeypatch.setattr(
        "local_operator.network.identity.load",
        lambda *a, **k: SimpleNamespace(device_id=SELF, name="laptop"),
    )
    monkeypatch.setattr(net_cli, "_relay_answer", lambda op, **fields: dict(reply))

    receipt = net_cli._member_repair_receipt(
        root, [{"name": "gitlab", "url": url, "refs": [], "shape": "oauth"}]
    )
    grant = receipt["gitlab"]["grant"]
    assert grant["ok"] is False
    assert grant["code"] == "no_token"
    # No durable verdict is invented: there is nothing a person can act on.
    assert key not in mcp_state.load(NETWORK, root)
