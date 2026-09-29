"""The Radient org login through the mesh: the borrow seam, driven end to end.

WHY THIS FILE EXISTS (Radient org projection, 2026-09-29). ``resolve_radient_oauth_access``
used to build a PLAIN ``AuthStore``, so on a paired device without the signed-in account
``lop teams pull`` returned ``None`` and printed the re-login remedy — the operator's QA
workaround was using his own login on that peer by hand. The fix routes both Radient
resolvers through ``build_auth_store`` (``providers/radient_credentials.py``), so the mesh
rung can serve the owner's bearer and the kind rule still decides.

THE RIG MIRRORS ``test_credentials_real_link.py`` deliberately: two roots, two real
relays, one real TCP link; the share is the product's own parser and the pull is the
borrower's own control socket; a rotating stub IdP sits behind the OWNER's Radient
refresh so "exactly one refresh POST" is a fact rather than an expectation. Everything
the borrower does goes through APIs that take its root explicitly — ``AuthStore``'s
database derives from the AMBIENT config dir, so the ambient root stays the OWNER's for
the whole test, exactly as in the mirrored module.

CELLS (design memo §6):
* R3 — the borrow serves the owner's bearer; ONE refresh POST; owner audit
  ``credential.grant``; the borrower's ``auth.db`` dump unchanged; no refresh material
  in any frame.
* R4 — the QA topology: a local hub with the peer-side opt-in
  (``RADIENT_ORG_ALLOW_NONCANONICAL_BASE``), set on the BORROWER as in the field run
  (the guard runs where the request is sent).
* P1 — the same local hub WITHOUT the opt-in returns ``None`` and sends the owner
  nothing at all.
* P2 — a network that has not shared radient borrows nothing, dials nothing, and
  writes no observation state.
* P3 — a key-only owner is served by the generic broker, and the org resolver still
  refuses it (person-scope preserved over the mesh).
"""

from __future__ import annotations

import argparse
import json
import sqlite3
import time
from contextlib import contextmanager
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Iterator

import pytest

from local_operator.env import DEFAULT_RADIENT_API_BASE_URL
from local_operator.network import relay, store
from local_operator.network.credentials import placement as placement_mod
from local_operator.providers.radient_credentials import (
    ORG_ALLOW_NONCANONICAL_ENV,
    resolve_radient_oauth_access_sync,
)
from tests.unit.network.test_credentials_owner import (  # noqa: F401 — fixtures by import
    RotatingIdP,
    idp,
)
from tests.unit.network.test_relay_e2e import (  # noqa: F401 — fixtures by import
    _pair,
    devices,
)

#: The canonical hub the production guard allows. Taken from the ONE constant the
#: provider definition itself quotes, so this cannot drift from the real guard.
CANONICAL_HUB = DEFAULT_RADIENT_API_BASE_URL

#: A local hub as the QA topology configures one. Never dialled: the resolver only
#: READS it through the destination guard, and the client is built by the caller.
LOCAL_HUB = "http://127.0.0.1:9/v1"


def _parser() -> argparse.ArgumentParser:
    """The shipped ``lop network`` parser, built the way ``lop`` builds it."""
    from local_operator.network import cli as network_cli

    parser = argparse.ArgumentParser(prog="lop")
    network_cli.add_parser(parser.add_subparsers(dest="command"))
    return parser


def _lop_network(*argv: str) -> int:
    """Run one ``lop network …`` command exactly as typed: parse, then dispatch."""
    from local_operator.network import cli as network_cli

    return network_cli.main(_parser().parse_args(["network", *argv]))


def _share(mesh: Any, key: str = "radient") -> None:
    # No ``--network``: this device is in exactly one, which is how an operator types it.
    assert _lop_network("credential", "share", key, "--with", mesh.b.identity.name) == 0


def _pull(mesh: Any) -> dict[str, Any]:
    """``lop network credentials`` on B, as far as its relay: the leg-1 placement op."""
    found = store.find_own_relay(mesh.b.root)
    assert found is not None, "the borrower's relay published no record"
    reply = relay.control_request(found, "credential_placement", timeout=30.0)
    assert isinstance(reply, dict), reply
    detail = reply.get("detail")
    return detail if isinstance(detail, dict) else reply


def _audit(server: relay.RelayServer, event: str) -> list[dict[str, Any]]:
    server.audit.flush()
    return [row for row in server.audit.tail(500) if row.get("event") == event]


def _dump_auth(root: Path) -> list[tuple[Any, ...]]:
    """Every credential row of ``root``'s ``auth.db``, as the FILE holds it.

    Read-only URI on purpose: the dump itself must not be able to write, because its
    whole job is to prove that the borrow did not. All eight columns are compared —
    a rotated stamp or a moved backoff is a write too.
    """
    conn = sqlite3.connect(f"file:{root / 'auth.db'}?mode=ro", uri=True)
    try:
        return conn.execute(
            "SELECT id, provider, credential_type, data, disabled_cause, identity_key,"
            " created_at, updated_at FROM auth_credentials ORDER BY id"
        ).fetchall()
    finally:
        conn.close()


def _seed_radient_login(
    root: Path, idp: RotatingIdP, *, key_only: bool  # noqa: F811 — the imported fixture
) -> str:
    """Seed the OWNER's store with the login under test; returns the seeded refresh token.

    ``key_only`` seeds the pasted-key login (class 4): the store then holds no OAuth
    row, which is the P3 state where the generic broker still serves and the org
    resolver must still refuse.
    """
    from local_operator.providers.auth_store import AuthStore

    seeded = ""
    auth = AuthStore(db_path=root / "auth.db", config_dir=root)
    try:
        if key_only:
            auth.upsert_credential(
                "radient", {"type": "api_key", "source": "login", "key": "pasted-fixture"}
            )
        else:
            seeded = idp.current_refresh
            auth.upsert_credential(
                "radient",
                {
                    "type": "oauth",
                    "access": "stale-access",
                    # EXPIRED on purpose, so the first grant MUST refresh: a test that
                    # served a live token would prove nothing about the one-POST
                    # property or about a refresh spent on a reply nobody receives.
                    "expires": int(time.time() * 1000) - 60_000,
                    "refresh": seeded,
                    "email": "owner@example.test",
                },
            )
    finally:
        auth.close()
    return seeded


def _seed_borrower_rows(root: Path) -> None:
    """A NON-radient row in B's store, so the dump comparison has content to compare.

    Deliberately not a radient row: a local radient key would short-circuit the
    resolver BEFORE the borrow (the kind rule), and a local radient OAuth row would
    win locally — either way the mesh rung would never be reached.
    """
    from local_operator.providers.auth_store import AuthStore

    auth = AuthStore(db_path=root / "auth.db", config_dir=root)
    try:
        auth.upsert_credential(
            "openai", {"type": "api_key", "source": "login", "key": "borrower-fixture"}
        )
    finally:
        auth.close()


def _stub_radient_refresh(
    monkeypatch: pytest.MonkeyPatch, idp: RotatingIdP  # noqa: F811 — the imported fixture
) -> None:
    """Point the REAL radient refresh fn at the loopback IdP.

    ``refresh_token`` is resolved lazily (``registry._lazy_refresh`` imports the module
    attribute at call time), which is exactly what makes this replacement effective for
    the owner's store path. Every step below the replacement is the product's own:
    the store's lease, the rotated-token persistence, the borrow.
    """

    async def refresh(creds: dict[str, Any], **_: Any) -> dict[str, Any]:
        import httpx

        async with httpx.AsyncClient(timeout=5.0) as client:
            response = await client.post(
                idp.url,
                data={
                    "grant_type": "refresh_token",
                    "refresh_token": str(creds.get("refresh") or ""),
                },
            )
        if response.status_code != 200:
            from local_operator.providers.auth_store import AuthStoreError

            raise AuthStoreError(f"stub IdP refused the exchange: {response.status_code}")
        token = response.json()
        merged = dict(creds)
        merged["access"] = token["access_token"]
        merged["refresh"] = token["refresh_token"]
        merged["expires"] = int(time.time() * 1000) + int(token["expires_in"]) * 1000
        return merged

    from local_operator.providers.oauth import radient as radient_oauth

    monkeypatch.setattr(radient_oauth, "refresh_radient_token", refresh)


@contextmanager
def _radient_mesh(
    devices_fixture: tuple[relay.RelayServer, relay.RelayServer, str, int],
    monkeypatch: pytest.MonkeyPatch,
    idp: RotatingIdP,  # noqa: F811 — the imported fixture
    *,
    key_only: bool,
) -> Iterator[Any]:
    """Owner A (holding the login) and borrower B (not), both with relays up."""
    server_a, server_b, host, port = devices_fixture
    _stub_radient_refresh(monkeypatch, idp)
    record, _host, _port = _pair(devices_fixture, monkeypatch, role="drive")
    # THE AMBIENT ROOT IS THE OWNER'S from here on: the broker's store
    # (``MeshCredentialBroker._auth_store_instance``) derives its database from the
    # ambient config dir, and B is driven only through APIs that take its root
    # explicitly. Same constraint, same shape as ``test_credentials_real_link``.
    monkeypatch.setenv("LOCAL_OPERATOR_CONFIG_DIR", str(server_a.root))
    # B must be able to DIAL A: the pairing leaves A's row on B with no endpoint, and a
    # relay started later syncs only its own row.
    with store.mutate(record.network_id, server_b.root) as copy:
        owner_row = copy.member(server_a.identity.device_id)
        assert owner_row is not None
        owner_row.endpoints = [f"{host}:{port}"]
        store.save(copy, server_b.root)
    server_b.start()
    try:
        seed = _seed_radient_login(server_a.root, idp, key_only=key_only)
        _seed_borrower_rows(server_b.root)
        yield SimpleNamespace(
            a=server_a,
            b=server_b,
            network_id=record.network_id,
            owner=server_a.identity.device_id,
            borrower=server_b.identity.device_id,
            idp=idp,
            seed_refresh=seed,
        )
    finally:
        server_b.stop()


@pytest.fixture()
def radient_mesh(
    devices: tuple[relay.RelayServer, relay.RelayServer, str, int],  # noqa: F811 — the fixture
    monkeypatch: pytest.MonkeyPatch,
    idp: RotatingIdP,  # noqa: F811 — the fixture
) -> Iterator[Any]:
    with _radient_mesh(devices, monkeypatch, idp, key_only=False) as mesh:
        yield mesh


@pytest.fixture()
def key_only_radient_mesh(
    devices: tuple[relay.RelayServer, relay.RelayServer, str, int],  # noqa: F811 — the fixture
    monkeypatch: pytest.MonkeyPatch,
    idp: RotatingIdP,  # noqa: F811 — the fixture
) -> Iterator[Any]:
    with _radient_mesh(devices, monkeypatch, idp, key_only=True) as mesh:
        yield mesh


def _resolve_on_b(mesh: Any, base_url: str) -> Any:
    """The CLI's own bridge, on B's root — the exact call ``lop teams pull`` makes."""
    return resolve_radient_oauth_access_sync(mesh.b.root, base_url)


def test_the_borrow_serves_the_owners_bearer_with_one_refresh(
    radient_mesh: Any, monkeypatch: pytest.MonkeyPatch
) -> None:
    """R3: the feature end to end — one POST, one audit, no leak, no local write."""
    mesh = radient_mesh
    _share(mesh)
    _pull(mesh)
    before = _dump_auth(mesh.b.root)

    # Frames are recorded AROUND THE BORROW only (leg 2 — the peer link; the leg-1
    # control socket is same-host and never carries provider material).
    frames: list[dict[str, Any]] = []
    original_send = relay.PeerLink.send

    def _recording_send(self: Any, frame: dict[str, Any], **kwargs: Any) -> bool:
        frames.append(frame)
        return original_send(self, frame, **kwargs)

    monkeypatch.setattr(relay.PeerLink, "send", _recording_send)
    access = _resolve_on_b(mesh, CANONICAL_HUB)

    assert access is not None, "the peer did not borrow the org login"
    assert access.kind == "oauth"
    assert access.access_token == "access-1", access.access_token  # the IdP's first rotation
    assert access.email == "owner@example.test"
    assert len(mesh.idp.posts) == 1, "the owner refreshed more than once"

    records = _audit(mesh.a, "credential.grant")
    assert len(records) == 1, records
    detail = records[-1]["detail"]
    assert detail.get("act") == mesh.owner and detail.get("sub") == mesh.borrower
    assert detail.get("credential_kind") == "oauth"
    assert detail.get("refreshed") is True

    assert _dump_auth(mesh.b.root) == before, "the borrow wrote into the borrower's auth.db"

    # THE INSTRUMENT IS PROVED ALIVE BEFORE ITS VERDICT IS BELIEVED: the lent
    # access token DID cross this wire, so an empty or blind capture cannot pass.
    assert frames, "the frame recorder captured nothing"
    blob = json.dumps(frames, default=str)
    assert "access-1" in blob, "the capture missed the grant reply itself"
    secrets = {mesh.seed_refresh, *mesh.idp.posts}
    secrets.update(f"refresh-{n}" for n in range(6))
    for secret in sorted(secrets):
        assert secret and secret not in blob, f"refresh material on the wire: {secret!r}"
    assert "refresh_token" not in blob
    assert '"refresh"' not in blob


def test_the_qa_topology_borrows_for_a_local_hub_with_the_peer_opt_in(
    radient_mesh: Any, monkeypatch: pytest.MonkeyPatch
) -> None:
    """R4: the topology the operator ran by hand — a local hub, opt-in on the PEER.

    The opt-in is read by the destination guard INSIDE the borrower's resolve (the
    side that sends), which is why it belongs on the peer in the field run; nothing
    on the owner reads it.
    """
    mesh = radient_mesh
    _share(mesh)
    _pull(mesh)
    monkeypatch.setenv(ORG_ALLOW_NONCANONICAL_ENV, "1")
    access = _resolve_on_b(mesh, LOCAL_HUB)
    assert access is not None, "the QA topology did not borrow"
    assert access.access_token == "access-1"
    assert len(mesh.idp.posts) == 1


def test_a_non_canonical_hub_without_the_opt_in_never_asks_the_owner(
    radient_mesh: Any, monkeypatch: pytest.MonkeyPatch
) -> None:
    """P1: the guard runs FIRST — a refused destination costs the owner nothing."""
    mesh = radient_mesh
    _share(mesh)
    _pull(mesh)
    monkeypatch.delenv(ORG_ALLOW_NONCANONICAL_ENV, raising=False)
    assert _resolve_on_b(mesh, LOCAL_HUB) is None
    assert mesh.idp.posts == [], "a refused destination spent the owner's refresh"
    assert _audit(mesh.a, "credential.grant") == []


def test_a_network_that_has_not_shared_radient_borrows_nothing(radient_mesh: Any) -> None:
    """P2: IN a network, holding another share — radient is still not borrowable.

    The wrapper EXISTS here (B holds a document with a remote owner), so the cell
    exercises the "not shared" answer on the new rung rather than the plain-store path:
    no dial, and no observation state written for a key this device never asked about.
    """
    mesh = radient_mesh
    from local_operator.providers.auth_store import AuthStore

    other = AuthStore(db_path=mesh.a.root / "auth.db", config_dir=mesh.a.root)
    try:
        other.upsert_credential(
            "openai", {"type": "api_key", "source": "login", "key": "owner-other-fixture"}
        )
    finally:
        other.close()
    _share(mesh, "openai")
    _pull(mesh)

    state_path = placement_mod.placement_state_path(mesh.network_id, mesh.b.root)
    assert not state_path.exists(), "the pull itself wrote an observation"

    assert _resolve_on_b(mesh, CANONICAL_HUB) is None
    assert mesh.idp.posts == [], "nothing was shared, so nothing may be dialled"
    assert _audit(mesh.a, "credential.grant") == []
    assert not state_path.exists(), "a resolve for an unshared key wrote observation state"


def test_a_key_only_owner_is_served_but_the_org_resolver_refuses_it(
    key_only_radient_mesh: Any,
) -> None:
    """P3: person-scope survives the mesh — the borrow happens, the KIND still refuses.

    A pasted key proves an application tenant, not a person, so the org resolver must
    answer ``None`` even when the key was successfully borrowed. The owner's audit is
    the proof the borrow RAN: without it, "returns None" would be the same vacuous
    answer base gives for a login that is not there.
    """
    mesh = key_only_radient_mesh
    _share(mesh)
    _pull(mesh)
    assert _resolve_on_b(mesh, CANONICAL_HUB) is None
    records = _audit(mesh.a, "credential.grant")
    assert len(records) == 1, "the borrow never reached the owner"
    assert records[-1]["detail"].get("credential_kind") == "api_key"
    assert mesh.idp.posts == [], "a static key must never provoke a refresh"
