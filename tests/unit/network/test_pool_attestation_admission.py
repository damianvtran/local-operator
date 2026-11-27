"""Pool admission (A8.1): a pool grant is admitted on attestation, never on a card.

WHAT THIS FILE PINS, AND WHY IT IS A SECURITY BOUNDARY RATHER THAN A FEATURE. A
pool member has no screen and no person (``mesh-compute-pool.md`` §3.2), so the
human act that admits every ordinary device — comparing six digits on two
screens — cannot happen at its end. A8.1 replaces it with the provisioning
plane's signature over the pod's own device key, verified against a key set the
operator pinned on THIS device. Everything below exists to show that the
replacement is a NARROWER door, never a wider one:

* a statement naming another network, another device key, another window, or a
  key nobody pinned admits nothing at all, and the refusal names its reason in
  the audit;
* an attestation offered against an ORDINARY (``device``) invite is ignored, so
  the human card that invite promises cannot be skipped by sending one;
* an admitted pool member's capabilities are INTERSECTED with the allowlist
  (R-I11: ``list``, ``view``, ``prompt``, ``steer``, ``stop``, ``slash``,
  ``broker_credential`` and nothing else) at mint AND at admission, and a pool
  grant can never be minted as an ``admin`` member at all;
* it borrows session-scoped credentials only — neither spelling of the GitHub
  adapter's credential — and declares none of its own.

THE GOLDEN VECTOR IS PINNED AS BYTES, NOT RE-DERIVED. ``_CANONICAL_TEMPLATE`` and
``_SIGNATURE_B64`` are literals produced by the provisioning plane's contract
(the same canonical-JSON spelling ``remote_compute/attestation.py`` verifies);
the test rebuilds them from the recorded seeds and refuses the vector if a single
byte or the signature differs. A verifier that drifted would fail the pinned
signature before any relay ran.

ISOLATION: every cell runs against the ``root`` config dir (``tests.unit.network
.conftest``), real loopback sockets on ``127.0.0.1``, and the fixed identities and
clock below. Nothing here reads the operator's config, mesh store or clock.
"""

from __future__ import annotations

import base64
import json
import threading
from argparse import Namespace
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Iterator, Mapping, Sequence

import pytest
from cryptography.hazmat.primitives import serialization
from cryptography.hazmat.primitives.asymmetric.ed25519 import Ed25519PrivateKey

from local_operator.network import identity
from local_operator.network import invite as invite_mod
from local_operator.network import onboard, relay, store, types, wire
from local_operator.network.credentials import github as github_mod
from local_operator.network.credentials import offers as offers_mod
from local_operator.network.handshake import (
    Credential,
    Handshake,
    pair_abort_frame,
    pair_timeout_seconds,
    sas_matches,
)
from local_operator.remote_compute import attestation as attestation_mod
from tests.unit.network import conftest as net_fixtures
from tests.unit.network.test_relay_e2e import (  # noqa: F401 — fixtures by import
    _answer_confirmation,
    _type_the_code,
    serve_shaped_relay,
)

# ---------------------------------------------------------------------------
# The pinned vector: seeds, network, and the bytes the contract signs
# ---------------------------------------------------------------------------

#: The pod's Ed25519 seed. Fixed so the device key inside the statement — and
#: therefore the signed bytes — are a literal this file can pin.
_POOL_DEVICE_SEED_HEX = "3a" * 32
#: The provisioning plane's signing seed (a test control plane; its public key is
#: what the network pins).
_PLANE_SEED_HEX = "5c" * 32
_PLANE_KEY_ID = "pool-test-plane-1"

#: The one network these cells admit into. Fixed because the statement binds it.
_NETWORK_ID = "n_0b1e5a9f00d0123456789abc"
_NETWORK_NAME = "pool-net"
_WORKLOAD_ID = "wl_0123456789abcdef01234567"
_ACCOUNT_ID = "acct_0123456789abcdef01234567"
_ISSUED_AT = 1760000000
_EXPIRES_AT = 1760000600

#: The verifier's clock for every cell here. ``relay._attestation_now`` is the one
#: clock seam the admission reads, so the window rules are exercised at their
#: exact boundaries instead of against the wall clock.
_NOW = _ISSUED_AT + 1

_CANONICAL_TEMPLATE = (
    '{"v":1,"workload_id":"wl_0123456789abcdef01234567",'
    '"account_id":"acct_0123456789abcdef01234567",'
    '"network_id":"n_0b1e5a9f00d0123456789abc",'
    '"device_key":"__DEVICE_KEY__","issued_at":1760000000,"expires_at":1760000600}'
)

#: The plane's signature over ``_CANONICAL_TEMPLATE`` with the device key below —
#: produced by an independent signer and pinned here, so a canonical-spelling or
#: signature-verification change fails without a relay running.
_SIGNATURE_B64 = (
    "lgenupq0-kxROLm6HnV0hyyRuuorvypn9yDfWgV2_VZyyBdHrfGbjUdRS9Z107dTHRFIQNxrE5r_yBPsJHGtAA"
)
#: The plane's public key, unpadded URL-safe base64 — the spelling a pinned
#: ``NetworkRecord.attestation_keys`` value has.
_PLANE_PUBLIC_B64 = "7WpHo52oabVEYVXkCy2T8ePwFnviZzK656PvnY46P9M"

#: A mutable clock, so a cell can move ``now`` without another fixture.
_CLOCK: list[int] = [_NOW]


def _b64u(data: bytes) -> str:
    return base64.urlsafe_b64encode(data).decode("ascii").rstrip("=")


def _public_key(seed_hex: str) -> bytes:
    private = Ed25519PrivateKey.from_private_bytes(bytes.fromhex(seed_hex))
    return private.public_key().public_bytes(
        serialization.Encoding.Raw, serialization.PublicFormat.Raw
    )


def _sign(seed_hex: str, payload: bytes) -> str:
    private = Ed25519PrivateKey.from_private_bytes(bytes.fromhex(seed_hex))
    return _b64u(private.sign(payload))


def _canonical(statement: Mapping[str, Any]) -> bytes:
    """The one signed spelling of a statement (the contract's canonical JSON)."""
    return json.dumps(statement, separators=(",", ":")).encode("utf-8")


def _statement(*, device_key: str, network_id: str = _NETWORK_ID, **over: Any) -> dict[str, Any]:
    statement: dict[str, Any] = {
        "v": 1,
        "workload_id": _WORKLOAD_ID,
        "account_id": _ACCOUNT_ID,
        "network_id": network_id,
        "device_key": device_key,
        "issued_at": _ISSUED_AT,
        "expires_at": _EXPIRES_AT,
    }
    statement.update(over)
    return statement


def _attestation_frame(
    *,
    signer_seed: str = _PLANE_SEED_HEX,
    key_id: str = _PLANE_KEY_ID,
    device_key: str = "",
    **over: Any,
) -> dict[str, str]:
    """The join block's ``attestation`` mapping, signed with a chosen seed."""
    if not device_key:
        device_key = _b64u(_public_key(_POOL_DEVICE_SEED_HEX))
    payload = _canonical(_statement(device_key=device_key, **over))
    return {
        "payload_b64": _b64u(payload),
        "signature": _sign(signer_seed, payload),
        "key_id": key_id,
    }


def _pool_identity(root: Path) -> identity.DeviceIdentity:
    """The pod's identity, SAVED from the fixed seed rather than minted."""
    public = _public_key(_POOL_DEVICE_SEED_HEX)
    pod = identity.DeviceIdentity(
        device_id=identity.device_id_for(public),
        public_key=_b64u(public),
        private_key=_b64u(bytes.fromhex(_POOL_DEVICE_SEED_HEX)),
        name="pool-pod",
    )
    identity.save(pod, root)
    return pod


# ---------------------------------------------------------------------------
# The rig: two relays, A owning the pool grant, B the pod
# ---------------------------------------------------------------------------


@pytest.fixture()
def pool_devices(
    root: Path, monkeypatch: pytest.MonkeyPatch
) -> Iterator[tuple[relay.RelayServer, relay.RelayServer, str, int]]:
    """A listening owner and the pod's dialling identity, on loopback."""
    root_a = root / "a"
    root_b = root / "b"
    identity_a = identity.mint(root_a, name="owner-device")
    pod = _pool_identity(root_b)
    server_a = serve_shaped_relay(
        root_a, monkeypatch, identity=identity_a, audit=relay.AuditLog(root_a)
    )
    server_b = serve_shaped_relay(root_b, monkeypatch, identity=pod, audit=relay.AuditLog(root_b))
    # The verifier's clock, pinned: the window rules are about the statement's
    # timestamps, and a cell that moved them needs no second monkeypatch.
    _CLOCK[0] = _NOW
    monkeypatch.setattr(relay, "_attestation_now", lambda: _CLOCK[0])
    host, port = server_a.bind()
    server_a.bind_control()
    server_a.start()
    # The CLI's joining half resolves the config dir from the AMBIENT environment.
    monkeypatch.setenv("LOCAL_OPERATOR_CONFIG_DIR", str(root_b))
    try:
        yield server_a, server_b, host, port
    finally:
        server_a.stop()
        server_b.stop()


def _init_pool_network(
    server: relay.RelayServer, *, attestation_keys: dict[str, str] | None = None
) -> types.NetworkRecord:
    """The owner's network, with the control plane's key PINNED."""
    record = types.NetworkRecord(
        network_id=_NETWORK_ID,
        name=_NETWORK_NAME,
        created_by=server.identity.device_id,
        self_device_id=server.identity.device_id,
        self_role="admin",
        self_capabilities=sorted(types.capabilities_for_role("admin")),
        listen={"address": "127.0.0.1", "port": server.settings.port, "advertised": []},
        attestation_keys=(
            {_PLANE_KEY_ID: _PLANE_PUBLIC_B64} if attestation_keys is None else attestation_keys
        ),
    )
    from secrets import token_bytes

    state = types.SecretState(
        network_id=record.network_id, epoch=1, secret=wire.b64u(token_bytes(32))
    )
    relay.admit(
        record,
        device_id=server.identity.device_id,
        public_key=server.identity.public_key,
        name=server.identity.name,
        role="admin",
        capabilities=sorted(types.capabilities_for_role("admin")),
        added_by=server.identity.device_id,
        added_via="self",
        root=server.root,
        persist=False,
    )
    store.save(record, server.root)
    store.save_secrets(state, server.root)
    return record


def _mint_pool_invite(
    server: relay.RelayServer,
    record: types.NetworkRecord,
    *,
    role: str = "drive",
    member_kind: types.MemberKind = "pool",
) -> tuple[str, Any]:
    """Mint through the product's own ``invite.mint`` — the R-I11 mint-side clamp."""
    state = store.load_secrets(record.network_id, server.root)
    minted = invite_mod.mint(record, state.secret, role=role, ttl_s=600.0, member_kind=member_kind)
    record.invites.append(minted.record)
    store.save(record, server.root)
    path = store.save_invite_token(minted.record.invite_id, minted.token, server.root)
    assert path.exists()
    return minted.token, minted.envelope


def _join_as_pool(
    server_b: relay.RelayServer,
    *,
    host: str,
    port: int,
    token: str,
    envelope: Any,
    attestation_path: Path | None,
) -> Any:
    """Drive the CLI's own join half, with the pod's attestation file (or none).

    The automated seat is the pod's: it sends this device's own derived code, and
    the inviter still compares it — what replaces the human here is the
    attestation, never a skipped compare.
    """
    from local_operator.network import cli as net_cli

    args = Namespace(
        sas_stdin=False,
        verify=False,
        emit_sas=False,
        name=server_b.identity.name,
        json=True,
        automated=True,
        attestation=str(attestation_path) if attestation_path is not None else "",
    )
    return net_cli._join_one(  # noqa: SLF001 — the CLI's own driver, exercised as it runs
        host=f"{host}:{port}",
        token=token,
        envelope=envelope,
        identity=server_b.identity,
        settings=relay.NetworkSettings(port=0, listen_address="127.0.0.1"),
        args=args,
        wire=wire,
        Handshake=Handshake,
        Credential=Credential,
        pair_abort_frame=pair_abort_frame,
        pair_timeout_seconds=pair_timeout_seconds,
        sas_matches=sas_matches,
        invite_mod=invite_mod,
        store=store,
        relay_mod=relay,
    )


def _join_body(result: Any) -> dict[str, Any]:
    """The CLI's own JSON body out of ``_join_one``'s ``(lines, body)`` answer."""
    assert isinstance(result, tuple) and len(result) == 2, result
    body = result[1]
    assert isinstance(body, dict), body
    return body


def _refused_pool_join(
    server_b: relay.RelayServer,
    *,
    host: str,
    port: int,
    token: str,
    envelope: Any,
    attestation_path: Path | None,
) -> str:
    """Join, and return the refusal the joining side was told.

    The inviter closes the pair socket at a refusal rather than explaining itself
    (an oracle would let a stranger probe token validity), so the sentence the
    abort frame carried is the pod's half of the story. The STABLE CODE is read
    from the relay's audit — that is where a caller branches.
    """
    with pytest.raises(types.MeshRefusal) as caught:
        _join_as_pool(
            server_b,
            host=host,
            port=port,
            token=token,
            envelope=envelope,
            attestation_path=attestation_path,
        )
    return str(caught.value)


def _write_attestation(tmp_path: Path, frame: Mapping[str, str] | None) -> Path | None:
    if frame is None:
        return None
    path = tmp_path / "attestation.json"
    path.write_text(json.dumps(dict(frame)), encoding="utf-8")
    return path


def _audit_rows(server: relay.RelayServer, event: str) -> list[dict[str, Any]]:
    return [row for row in server.audit.tail(limit=500) if row.get("event") == event]


def _refusal_cause(server: relay.RelayServer) -> str:
    rows = _audit_rows(server, "pairing_refused")
    assert rows, f"the relay recorded no pairing_refused row: {server.audit.tail(limit=50)}"
    return str(rows[-1]["detail"].get("cause") or "")


# ---------------------------------------------------------------------------
# The vector
# ---------------------------------------------------------------------------


def test_the_pinned_vector_is_the_contract_bytes_and_signature() -> None:
    """The recorded seeds reproduce the pinned canonical bytes AND signature.

    This is the cross-language half: the bytes are the contract, so a statement
    that reassembles differently (a space, another key order, a re-spelled base64)
    fails the signature check below — which is what makes the relay cells that
    follow evidence about the FORMAT, not just about this verifier.
    """
    device_key = _b64u(_public_key(_POOL_DEVICE_SEED_HEX))
    payload = _canonical(_statement(device_key=device_key))
    assert payload.decode("utf-8") == _CANONICAL_TEMPLATE.replace("__DEVICE_KEY__", device_key)
    assert _sign(_PLANE_SEED_HEX, payload) == _SIGNATURE_B64

    verified = attestation_mod.verify_attestation(
        payload_b64=_b64u(payload),
        signature=_SIGNATURE_B64,
        key_id=_PLANE_KEY_ID,
        keys=attestation_mod.parse_pinned_keys({_PLANE_KEY_ID: _PLANE_PUBLIC_B64}),
        # The relay's own call shape: the grant was minted before the workload
        # existed, so this side cannot compare a workload it never had.
        expect_workload_id=None,
        expect_network_id=_NETWORK_ID,
        expect_device_key=_public_key(_POOL_DEVICE_SEED_HEX),
        now=_NOW,
    )
    assert verified.workload_id == _WORKLOAD_ID
    assert verified.key_id == _PLANE_KEY_ID


# ---------------------------------------------------------------------------
# The positive path
# ---------------------------------------------------------------------------


def _seed_model_login(root: Path) -> str:
    """One OAuth provider login on the OWNER's device — a session-scoped share."""
    from local_operator.providers.auth_store import AuthStore

    auth = AuthStore(db_path=root / "auth.db", config_dir=root)
    try:
        auth.upsert_credential(
            "openai", {"refresh": "r", "access": "a", "email": "owner@corp.example"}
        )
    finally:
        auth.close()
    items = offers_mod.build_items(root, network_id=_NETWORK_ID)
    served = offers_mod.served_keys(items)
    assert served, f"the seeded login did not reach the offer: {items}"
    return served[0]


def test_a_pool_grant_is_admitted_on_attestation_without_a_card(
    pool_devices: tuple[relay.RelayServer, relay.RelayServer, str, int],
    tmp_path: Path,
) -> None:
    """A8.1 end to end: no card, allowlisted capabilities, session-scoped shares only.

    The invite is minted as a ``drive`` pool grant, the widest role a pool member
    may hold (minting ``admin`` is refused by name — see
    ``test_a_pool_invite_may_never_be_minted_as_an_admin``). R-I11's guarantee is
    that a pool member's stored set never contains ``admin``/``trust``/``delete``/
    ``move``/``approve``/``unattended``/``update`` however the invite was written,
    so this cell would fail if the clamp were ever dropped from either seat (mint
    or admission).
    """
    server_a, server_b, host, port = pool_devices
    record = _init_pool_network(server_a)
    seeded_key = _seed_model_login(server_a.root)
    token, envelope = _mint_pool_invite(server_a, record, role="drive")

    # THE MINT-SIDE CLAMP: the saved row and the token already agree.
    minted_row = store.load(_NETWORK_ID, server_a.root).invites[0]
    assert minted_row.member_kind == "pool"
    assert set(minted_row.capabilities) <= types.POOL_ALLOWED_CAPABILITIES, minted_row

    path = _write_attestation(tmp_path, _attestation_frame())
    result = _join_as_pool(
        server_b, host=host, port=port, token=token, envelope=envelope, attestation_path=path
    )
    body = _join_body(result)
    assert body.get("ok") is not False, body

    refreshed = store.load(_NETWORK_ID, server_a.root)
    pod = refreshed.member(server_b.identity.device_id)
    assert pod is not None and pod.active
    assert pod.kind == "pool"
    assert pod.added_via == "invite:pool"
    assert set(pod.capabilities) <= types.POOL_ALLOWED_CAPABILITIES, pod.capabilities
    # PINNED, not merely bounded (security review round 1, MINOR 1): the admitted
    # set is EXACTLY the drive role plus the borrow path, so one capability more
    # fails here rather than showing up later as an op this pod was allowed to run.
    assert set(pod.capabilities) == set(types.capabilities_for_role("drive")) | {
        "broker_credential"
    }, pod.capabilities
    # The set the design states (§3.5): a SESSION-running set, not a membership
    # authority.
    assert "list" in pod.capabilities and "prompt" in pod.capabilities
    # The borrow path, granted by the one existing admission grant site.
    assert "broker_credential" in pod.capabilities, pod.capabilities
    # ... AND IT IS THE DOOR, not a stored name: the relay's own authoriser admits
    # this pod to ``net_broker`` with the row admission wrote (ratified Item 3).
    from local_operator.network import authorizer as authorizer_mod

    broker_link = types.LinkContext(
        link_id="l_pool_broker",
        device_id=pod.device_id,
        instance_id="i_pool",
        network_id=refreshed.network_id,
        epoch=refreshed.epoch,
        capabilities=frozenset(pod.capabilities),
        phase="member",
    )
    broker_authorizer = authorizer_mod.Authorizer(
        SimpleNamespace(
            network=lambda network_id: refreshed,
            local_session_ids=lambda: set(),
        ),  # type: ignore[arg-type]
        server_a.audit,
    )
    assert (
        broker_authorizer.check(broker_link, {"op": "net_broker", "req": 1}).action == "net_broker"
    )

    # NO CARD: nothing parked, nothing awaiting a person.
    assert server_a._ctl_pair_pending({}) == []  # noqa: SLF001 — the CLI's own control op
    assert not _audit_rows(server_a, "pairing_awaiting_confirmation")

    confirmed = _audit_rows(server_a, "pairing_confirmed")
    assert len(confirmed) == 1, confirmed
    assert confirmed[0]["detail"]["answered_by"] == f"attestation:{_PLANE_KEY_ID}"
    assert confirmed[0]["detail"]["workload_id"] == _WORKLOAD_ID

    admitted = _audit_rows(server_a, "member_admitted")
    assert admitted and admitted[-1]["detail"]["member_kind"] == "pool"
    # THE RESULTING SET IS RECORDED, not left to be inferred (security review
    # round 1, MINOR 2): the clamp is silent by design, so the audit row is what
    # answers "what did admission narrow it to".
    recorded_caps = set(admitted[-1]["detail"]["capabilities"])
    assert recorded_caps <= types.POOL_ALLOWED_CAPABILITIES, recorded_caps
    assert recorded_caps == set(pod.capabilities)

    # SESSION-SCOPED GRANT, and the owner is the declarer: the pod borrows, it
    # never owns. Read on A's own placement document.
    from local_operator.network.credentials import placement as placement_mod

    with placement_mod.mutate(
        _NETWORK_ID, server_a.root, self_device=server_a.identity.device_id
    ) as document:
        entry = document.entry(seeded_key)
        assert entry is not None, document
        assert entry.owner_device == server_a.identity.device_id, entry
        holder_scopes = {holder.device: holder.scope for holder in entry.holders if holder.device}
        assert holder_scopes.get(server_b.identity.device_id) == "session", holder_scopes


def test_a_device_invite_with_a_valid_attestation_still_waits_for_a_human(
    pool_devices: tuple[relay.RelayServer, relay.RelayServer, str, int],
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """THE INVARIANT: an attestation never stands in for a person on a device invite.

    The statement here is valid in every dimension — right network, right key,
    right window — and it is still ignored, because the SAVED ROW says ``device``.
    A property that admitted here would let anyone holding a pool grant's
    statement bypass the SAS confirmation every ordinary pair promises.
    """
    server_a, server_b, host, port = pool_devices
    record = _init_pool_network(server_a)
    token, envelope = _mint_pool_invite(server_a, record, member_kind="device")

    _type_the_code(monkeypatch)
    parked: list[dict[str, Any] | None] = []
    thread = threading.Thread(
        target=lambda: parked.append(_answer_confirmation(server_a, admit=False)), daemon=True
    )
    thread.start()
    path = _write_attestation(tmp_path, _attestation_frame())
    try:
        with pytest.raises(types.MeshRefusal) as caught:
            _join_as_pool(
                server_b,
                host=host,
                port=port,
                token=token,
                envelope=envelope,
                attestation_path=path,
            )
    finally:
        thread.join(20)

    # A person WAS asked — the card existed — and their decline is the outcome.
    assert parked and parked[0] is not None, "a device invite showed no confirmation card"
    assert "declined" in str(caught.value), str(caught.value)
    assert store.load(_NETWORK_ID, server_a.root).member(server_b.identity.device_id) is None


# ---------------------------------------------------------------------------
# The refusals — every one of them admits nobody
# ---------------------------------------------------------------------------


def _device_mismatch(**_: Any) -> dict[str, str]:
    other = identity.DeviceIdentity(
        device_id=identity.device_id_for(_public_key("7e" * 32)),
        public_key=_b64u(_public_key("7e" * 32)),
        name="someone-else",
    )
    return _attestation_frame(device_key=other.public_key)


def _other_network(**_: Any) -> dict[str, str]:
    return _attestation_frame(network_id="n_ffffffffffffffffffffffff")


def _tampered(**_: Any) -> dict[str, str]:
    frame = _attestation_frame()
    signature = frame["signature"]
    flipped = ("B" if signature[0] != "B" else "C") + signature[1:]
    return {**frame, "signature": flipped}


def _unknown_key(**_: Any) -> dict[str, str]:
    return _attestation_frame(key_id="a-key-nobody-pinned")


def _future(**_: Any) -> dict[str, str]:
    return _attestation_frame(issued_at=_NOW + 120, expires_at=_NOW + 720)


def _missing(**_: Any) -> None:
    return None


@pytest.mark.parametrize(
    ("make_frame", "code"),
    [
        # The statement is about another tenant: the segmentation guarantee.
        (_other_network, attestation_mod.CODE_NETWORK_MISMATCH),
        # The statement is about another pod's key: a stolen grant spends nothing.
        (_device_mismatch, attestation_mod.CODE_DEVICE_MISMATCH),
        # Signed by someone: only the pinned key's signature counts.
        (_tampered, attestation_mod.CODE_BAD_SIGNATURE),
        # A key id nobody pinned is refused, never searched for.
        (_unknown_key, attestation_mod.CODE_UNKNOWN_KEY_ID),
        # A statement from the future is not current.
        (_future, attestation_mod.CODE_ISSUED_IN_FUTURE),
        # No statement at all.
        (_missing, attestation_mod.CODE_MISSING_ATTESTATION),
    ],
)
def test_a_pool_join_is_refused_unless_its_statement_holds(
    pool_devices: tuple[relay.RelayServer, relay.RelayServer, str, int],
    tmp_path: Path,
    make_frame: Any,
    code: str,
) -> None:
    """Each refusal names its reason on the audit trail and admits nobody."""
    server_a, server_b, host, port = pool_devices
    record = _init_pool_network(server_a)
    token, envelope = _mint_pool_invite(server_a, record)

    frame = make_frame()
    path = _write_attestation(tmp_path, frame)  # type: ignore[arg-type]
    sentence = _refused_pool_join(
        server_b, host=host, port=port, token=token, envelope=envelope, attestation_path=path
    )

    assert sentence, f"a join refused {code} told the pod nothing"
    refreshed = store.load(_NETWORK_ID, server_a.root)
    assert refreshed.member(server_b.identity.device_id) is None
    assert len(refreshed.active_members()) == 1, refreshed.members
    assert net_fixtures.wait_for(lambda: bool(_audit_rows(server_a, "pairing_refused"))).ok
    assert _refusal_cause(server_a) == code
    # NO CARD WAS SHOWN FOR A REFUSED STATEMENT, and the refusal consumed the
    # token: an attestation that failed is not a retryable human mistype.
    assert server_a._ctl_pair_pending({}) == []  # noqa: SLF001
    assert refreshed.invites[0].state == "consumed"


def test_an_expired_statement_is_refused(
    pool_devices: tuple[relay.RelayServer, relay.RelayServer, str, int],
    tmp_path: Path,
) -> None:
    """EXPIRY, driven by moving the verifier's clock past the statement's window."""
    server_a, server_b, host, port = pool_devices
    record = _init_pool_network(server_a)
    token, envelope = _mint_pool_invite(server_a, record)
    path = _write_attestation(tmp_path, _attestation_frame())

    _CLOCK[0] = _EXPIRES_AT + 1
    _refused_pool_join(
        server_b, host=host, port=port, token=token, envelope=envelope, attestation_path=path
    )

    assert store.load(_NETWORK_ID, server_a.root).member(server_b.identity.device_id) is None
    assert net_fixtures.wait_for(lambda: bool(_audit_rows(server_a, "pairing_refused"))).ok
    assert _refusal_cause(server_a) == attestation_mod.CODE_EXPIRED


def test_a_network_with_no_pinned_key_admits_no_pool_member(
    pool_devices: tuple[relay.RelayServer, relay.RelayServer, str, int],
    tmp_path: Path,
) -> None:
    """FAIL CLOSED: an unpinned key set is not "nothing to check against"."""
    server_a, server_b, host, port = pool_devices
    record = _init_pool_network(server_a, attestation_keys={})
    token, envelope = _mint_pool_invite(server_a, record)
    path = _write_attestation(tmp_path, _attestation_frame())

    result = _refused_pool_join(
        server_b, host=host, port=port, token=token, envelope=envelope, attestation_path=path
    )

    assert result, "the pod was told nothing where the statement could not be checked"
    assert store.load(_NETWORK_ID, server_a.root).member(server_b.identity.device_id) is None
    assert net_fixtures.wait_for(lambda: bool(_audit_rows(server_a, "pairing_refused"))).ok
    assert _refusal_cause(server_a) == attestation_mod.CODE_EMPTY_KEY_SET


# ---------------------------------------------------------------------------
# R-I11 and the credential half, at the seam that decides them
# ---------------------------------------------------------------------------


def test_the_mint_clamp_keeps_only_the_pool_allowlist() -> None:
    """R-I11 at MINT: a pool invite is INTERSECTED with the allowlist.

    The invariant is the allowlist, not a named four (security review round 1,
    MAJOR 1): ``approve``, ``unattended`` and ``update`` were in the capability
    vocabulary and were never excluded by the denylist this replaces, so a
    ``drive`` pool invite carried them. Asserting the subset is what makes a
    FUTURE privileged capability fail here rather than land on a borrowed machine.
    """
    record = types.NetworkRecord(network_id="n_clamp", name="clamp-net", self_device_id="d_self")
    from secrets import token_bytes

    minted = invite_mod.mint(record, wire.b64u(token_bytes(32)), role="drive", member_kind="pool")
    for capabilities in (minted.record.capabilities, minted.envelope.capabilities):
        assert set(capabilities) <= types.POOL_ALLOWED_CAPABILITIES, capabilities
        # Named one by one as well as as a set: the subset above is the rule, and
        # this is the list the review asked for, so a failure says WHICH name got
        # through rather than just "a name did".
        for privileged in (
            "admin",
            "trust",
            "delete",
            "move",
            "approve",
            "unattended",
            "update",
            "workset_import",
        ):
            assert privileged not in capabilities, (privileged, capabilities)
    # The session-running set a drive grant resolves to, minus nothing: a ``drive``
    # role is already inside the allowlist, which is what makes the clamp a bound
    # rather than a surprise.
    assert sorted(minted.record.capabilities) == [
        "list",
        "prompt",
        "slash",
        "steer",
        "stop",
        "view",
    ], minted.record.capabilities
    # The row and the token agree — the joiner is never shown a set the record
    # will contradict.
    assert sorted(minted.record.capabilities) == sorted(minted.envelope.capabilities)


def test_the_clamp_is_an_allowlist_over_the_whole_vocabulary() -> None:
    """The clamp itself, over the FULL vocabulary rather than one role's caps.

    Driven from the vocabulary rather than a hand-written list for the reason the
    denylist is gone: a capability added later appears here automatically, so the
    day ``approve``/``unattended``/``update`` — or something nobody has invented —
    joins :data:`types.CAPABILITIES`, this cell shows whether a pool member gets
    it. A role-driven cell cannot: ``drive`` never carried the privileged three.
    """
    clamped = types.clamp_pool_capabilities(sorted(types.CAPABILITIES | {"frobnicate"}))
    assert set(clamped) == set(types.POOL_ALLOWED_CAPABILITIES), clamped
    assert set(clamped) == {
        "list",
        "view",
        "prompt",
        "steer",
        "stop",
        "slash",
        "broker_credential",
    }, clamped
    # The two sets are the same policy read from both ends, and must stay disjoint:
    # a name in both would be a rule that says yes and no at once.
    assert not (types.POOL_ALLOWED_CAPABILITIES & types.POOL_FORBIDDEN_CAPABILITIES)
    # Every role a pool grant may be minted as is INSIDE the allowlist, so the clamp
    # is a bound on a hand-written row rather than a silent narrowing of the roles a
    # person is offered.
    for role in types.POOL_MEMBER_ROLES:
        assert types.capabilities_for_role(role) <= types.POOL_ALLOWED_CAPABILITIES, role


def test_a_pool_invite_may_never_be_minted_as_an_admin() -> None:
    """Code review round 1, MAJOR 1: the ONE widening the clamp must not absorb.

    ``admin`` is membership authority, and a pool member has no person at its end
    to hold it. Narrowing the grant would leave a row whose stored ``role`` still
    reads ``admin`` to ``lowest_id_admin`` and ``owner._is_admin``, so the refusal
    comes before the clamp — refused by name, with the roles that ARE allowed.
    """
    record = types.NetworkRecord(network_id="n_admin", name="admin-net", self_device_id="d_self")
    from secrets import token_bytes

    with pytest.raises(types.MeshRefusal) as caught:
        invite_mod.mint(record, wire.b64u(token_bytes(32)), role="admin", member_kind="pool")
    assert caught.value.code == "bad_role", caught.value.code
    assert "drive" in str(caught.value) and "read" in str(caught.value), str(caught.value)
    # An ORDINARY invite keeps ``admin``: the refusal is about the pool's absent
    # person, not about the role.
    plain = invite_mod.mint(record, wire.b64u(token_bytes(32)), role="admin")
    assert "admin" in plain.record.capabilities


def test_a_device_invite_keeps_its_full_capability_set() -> None:
    """The clamp is a POOL rule: an ordinary invite is untouched, byte for byte."""
    record = types.NetworkRecord(network_id="n_device", name="device-net", self_device_id="d_self")
    from secrets import token_bytes

    material = wire.b64u(token_bytes(32))
    plain = invite_mod.mint(record, material, role="admin")
    assert "admin" in plain.record.capabilities
    # And the token carries no member-kind key at all, so an older joiner parses
    # exactly the bytes it always did.
    assert "member_kind" not in invite_mod.decode(plain.token).payload()


def test_the_offer_a_pool_joiner_sees_drops_the_device_scoped_keys(tmp_path: Path) -> None:
    """§3.5 on the OFFER: neither spelling of the device-bound key is shown.

    QA round 1, Q1: the offer filter excluded the GitHub adapter's placement row
    (``github``) and left ``secret:GITHUB_APP`` — the INSTALLATION KEY, enumerated
    like any other store secret — on the list a pool joiner was shown. A key the
    grant will refuse must not be promised on the joiner's own receipt.
    """
    items = [
        {
            "key": github_mod.GITHUB_KEY,
            "kind": github_mod.GITHUB_KIND,
            "label": "GitHub",
            "share": True,
        },
        {
            "key": f"secret:{github_mod.APP_SECRET_NAME}",
            "kind": "store-secret",
            "label": github_mod.APP_SECRET_NAME,
            "share": True,
        },
        {"key": "provider:openai", "kind": "oauth", "label": "OpenAI", "share": True},
    ]
    kept = relay._pool_offer_items(items)  # noqa: SLF001 — the helper under test
    assert [item["key"] for item in kept] == ["provider:openai"]
    # The one rule both write sites share, asserted on its own spelling: the
    # adapter's row, its installation key, and a case variant of the latter.
    assert offers_mod.is_pool_excluded_key(github_mod.GITHUB_KEY)
    assert offers_mod.is_pool_excluded_key(f"secret:{github_mod.APP_SECRET_NAME}")
    assert offers_mod.is_pool_excluded_key(f"secret:{github_mod.APP_SECRET_NAME.lower()}")
    assert not offers_mod.is_pool_excluded_key("provider:openai")
    assert not offers_mod.is_pool_excluded_key("secret:SOMETHING_ELSE")


def test_the_grant_refuses_every_device_scoped_key_even_when_a_decision_names_it(
    pool_devices: tuple[relay.RelayServer, relay.RelayServer, str, int],
) -> None:
    """The BELT behind that filter: a decision naming a device-scoped key places nothing.

    Reached with the real ``_grant_pair_shares`` — the one admission grant site —
    and a decision naming BOTH spellings of the GitHub credential, because QA round
    1's Q1 was exactly that: the offer excluded one spelling and the grant refused
    the other's sibling. Neither may reach a placement entry.
    """
    server_a, _server_b, _host, _port = pool_devices
    record = _init_pool_network(server_a)
    seeded_key = _seed_model_login(server_a.root)
    app_secret_key = f"secret:{github_mod.APP_SECRET_NAME}"
    # The member row admission writes, so the capability step has a row to edit
    # (this cell is about the grants; the admission itself is covered above).
    relay.admit(
        record,
        device_id="d_pool0000000000000000000000000000",
        public_key=_b64u(_public_key(_POOL_DEVICE_SEED_HEX)),
        name="pool-pod",
        role="drive",
        kind="pool",
        added_via="invite:pool",
        added_by=server_a.identity.device_id,
        root=server_a.root,
    )
    items = [
        {
            "key": github_mod.GITHUB_KEY,
            "kind": github_mod.GITHUB_KIND,
            "label": "GitHub",
            "share": True,
        },
        {
            "key": app_secret_key,
            "kind": "store-secret",
            "label": github_mod.APP_SECRET_NAME,
            "share": True,
        },
        {"key": seeded_key, "kind": "oauth", "label": "OpenAI", "share": True},
    ]
    decision = types.PairDecision(
        invite_id="inv_belt",
        decision="admit",
        matched=True,
        answered_by="attestation:test",
        shares=[github_mod.GITHUB_KEY, app_secret_key, seeded_key],
    )
    granted, reduced = relay._grant_pair_shares(  # noqa: SLF001 — the grant site itself
        record,
        joiner_id="d_pool0000000000000000000000000000",
        decision=decision,
        offer_items=items,
        owner_name=server_a.identity.name,
        root=server_a.root,
        audit=server_a.audit,
        pool_member=True,
    )
    assert granted == [seeded_key], granted
    assert github_mod.GITHUB_KEY in reduced, reduced
    assert app_secret_key in reduced, reduced
    for excluded in (github_mod.GITHUB_KEY, app_secret_key):
        skipped = [
            row["detail"].get("skipped")
            for row in _audit_rows(server_a, "credential.placement")
            if row["detail"].get("credential_key") == excluded
        ]
        assert "pool_session_scope_only" in skipped, (excluded, skipped)
    # NEVER PLACED: the document is the write the grant would have made, so an
    # absent entry for each spelling is the fact a pool member's borrow path would
    # otherwise rest on (QA round 1, Q1).
    from local_operator.network.credentials import placement as placement_mod

    with placement_mod.mutate(
        _NETWORK_ID, server_a.root, self_device=server_a.identity.device_id
    ) as document:
        for excluded in (github_mod.GITHUB_KEY, app_secret_key):
            entry = document.entry(excluded)
            assert entry is None or not any(
                holder.device == "d_pool0000000000000000000000000000" for holder in entry.holders
            ), (excluded, entry)
    pod = record.member("d_pool0000000000000000000000000000")
    assert pod is not None and "broker_credential" in pod.capabilities, pod


def test_an_admitted_pool_member_is_refused_the_forbidden_ops(
    pool_devices: tuple[relay.RelayServer, relay.RelayServer, str, int],
    tmp_path: Path,
) -> None:
    """R-I11 through the relay's OWN authoriser: the privileged ops are refused.

    The link context is built from the member row ADMISSION wrote — not from a
    capability list this cell composed — so a clamp that regressed (at mint or at
    admission) shows up here as an authorised administrative op. ``Authorizer`` is
    the single decision site for every inbound frame (relay ``_dispatch`` calls
    it), which is why asserting at this seam is evidence about the wire rather
    than about a copy of the rule.

    ``unattended`` has no op of its own (it is checked inside
    :meth:`relay.RelayServer._op_session_create`), so that one is driven through
    the create itself — the same stub-link shape ``test_unattended_create`` uses.
    """
    from local_operator.network import authorizer as authorizer_mod

    server_a, server_b, host, port = pool_devices
    record = _init_pool_network(server_a)
    token, envelope = _mint_pool_invite(server_a, record, role="drive")
    path = _write_attestation(tmp_path, _attestation_frame())
    _join_body(
        _join_as_pool(
            server_b, host=host, port=port, token=token, envelope=envelope, attestation_path=path
        )
    )
    refreshed = store.load(_NETWORK_ID, server_a.root)
    pod = refreshed.member(server_b.identity.device_id)
    assert pod is not None
    assert set(pod.capabilities) <= types.POOL_ALLOWED_CAPABILITIES, pod.capabilities

    state = SimpleNamespace(
        network=lambda network_id: refreshed,
        local_session_ids=lambda: set(),
    )
    authorizer = authorizer_mod.Authorizer(state, server_a.audit)  # type: ignore[arg-type]
    link = types.LinkContext(
        link_id="l_pool",
        device_id=pod.device_id,
        instance_id="i_pool",
        network_id=refreshed.network_id,
        epoch=refreshed.epoch,
        capabilities=frozenset(pod.capabilities),
        phase="member",
    )
    # Each op's required capability comes from the relay's own table, so this list
    # cannot drift from the vocabulary it is testing — and every capability the
    # pool denylist names must have its ops refused.
    gated = {
        op: cap
        for op, cap in authorizer_mod.OP_CAPABILITY.items()
        if cap in types.POOL_FORBIDDEN_CAPABILITIES
    }
    assert {*gated.values()} <= types.POOL_FORBIDDEN_CAPABILITIES, gated
    assert "admin" in gated.values(), gated
    # NAMED EXPLICITLY, because the brief's list is a promise ABOUT these ops: the
    # table walk above would pass vacuously if a name were dropped from it.
    for op in (
        "net_update",
        "net_restart",
        "net_epoch",
        "net_trust",
        "net_identity_rotate",
        "net_session_move",
        "net_session_lifecycle",
    ):
        assert op in gated, (op, gated)
    for op in gated:
        with pytest.raises(authorizer_mod.Refusal) as caught:
            authorizer.check(link, {"op": op})
        assert caught.value.code == "not_authorised", (op, caught.value)

    # AND THE OTHER DIRECTION (security review round 1, MINOR 1): the ops the
    # allowlist DOES grant pass the CAPABILITY gate. A cell that only asserted
    # refusals would stay green against an authoriser that refused everything —
    # and these are peer-scope ops on purpose, because a session op is refused by
    # the session-scope rule under the SAME ``not_authorised`` code and would
    # prove nothing about the capability row.
    for op, cap in (
        ("net_member_list", "list"),
        ("net_catalog", "list"),
    ):
        assert cap in pod.capabilities and cap in types.POOL_ALLOWED_CAPABILITIES, (op, cap)
        granted = authorizer.check(link, {"op": op, "req": 1})
        assert granted.action == op, (op, granted)
    # ``broker_credential`` is the fourth allowlisted write, and this cell seeds no
    # login (so the grant site has nothing to place). Its positive direction is
    # asserted where the borrow path exists: the admission cell above, whose pod
    # holds the capability and is admitted to ``net_broker``.

    # ``approve``: the one op that capability gates is INNER (``operator_challenge``
    # rides inside ``net_forward``), so it is asserted through the carrier the relay
    # actually dispatches — a signing challenge is what a remote ALLOW is made from.
    with pytest.raises(authorizer_mod.Refusal) as caught:
        authorizer.check(
            link,
            {"op": "net_forward", "req": 1, "frame": {"op": "operator_challenge", "req": 2}},
        )
    assert caught.value.code == "not_authorised", caught.value

    # ``unattended``: the create gate, on the row admission wrote. Refused BEFORE
    # anything is minted — the assertion on disk is what makes that a claim about
    # the real handler rather than about the capability set alone.
    # ``Any`` because this is a stub standing in for a ``PeerLink`` (the same
    # binding the ``_link()`` helpers in the sibling modules use).
    stub: Any = SimpleNamespace(
        device_id=pod.device_id,
        network_id=refreshed.network_id,
        epoch=refreshed.epoch,
        context=SimpleNamespace(
            capabilities=frozenset(pod.capabilities),
            device_id=pod.device_id,
            network_id=refreshed.network_id,
            epoch=refreshed.epoch,
        ),
    )
    with pytest.raises(types.MeshRefusal) as refused:
        server_a._op_session_create(  # noqa: SLF001 — the gate itself
            stub, {"op": "net_session_create", "req": 1, "cwd": "", "yolo": True}
        )
    assert refused.value.code == "not_permitted", refused.value
    assert "unattended" in str(refused.value), str(refused.value)
    assert not (server_a.root / "sessions").exists(), "a refused create left something on disk"


def test_a_tampered_pool_invite_admits_as_drive_with_allowlisted_capabilities(
    pool_devices: tuple[relay.RelayServer, relay.RelayServer, str, int],
    tmp_path: Path,
) -> None:
    """Code review round 1, MAJOR 1, the ADMISSION seat: the row is not trusted.

    ``invite.mint`` refuses ``role="admin"`` for a pool grant, so this cell writes
    the row that refusal cannot reach: a hand-edited store row naming ``admin`` and
    the whole capability vocabulary plus a name this build does not have. The
    member that comes out must be a ``drive`` member with an allowlisted set —
    both defences are exercised through the REAL ceremony, not by calling the
    helpers, because the row (not the token) is what every later decision reads.
    """
    server_a, server_b, host, port = pool_devices
    record = _init_pool_network(server_a)
    token, envelope = _mint_pool_invite(server_a, record, role="drive")

    stored = store.load(_NETWORK_ID, server_a.root)
    row = stored.invites[0]
    row.role = "admin"
    row.capabilities = sorted({*types.CAPABILITIES, "frobnicate"})
    store.save(stored, server_a.root)

    path = _write_attestation(tmp_path, _attestation_frame())
    body = _join_body(
        _join_as_pool(
            server_b, host=host, port=port, token=token, envelope=envelope, attestation_path=path
        )
    )
    assert body.get("ok") is not False, body

    refreshed = store.load(_NETWORK_ID, server_a.root)
    pod = refreshed.member(server_b.identity.device_id)
    assert pod is not None and pod.active and pod.kind == "pool"
    assert pod.role == "drive", pod.role
    assert set(pod.capabilities) <= types.POOL_ALLOWED_CAPABILITIES, pod.capabilities
    assert "frobnicate" not in pod.capabilities, pod.capabilities
    assert "admin" not in pod.capabilities, pod.capabilities
    admitted = _audit_rows(server_a, "member_admitted")
    assert admitted[-1]["detail"]["role"] == "drive", admitted[-1]


def test_a_pool_member_is_never_selected_or_accepted_as_an_admin(root: Path) -> None:
    """Code review round 1, MAJOR 1, the READERS of ``role``.

    Both readers answer questions a pool member must never be able to answer yes
    to, and both read a stored row that could have been written another way:
    ``lowest_id_admin`` names who rotates the network after a departure (the new
    epoch secret goes to whoever it names), and ``MeshCredentialBroker._is_admin``
    authorises a FORCED refresh of the operator's own logins. A row whose ``kind``
    says ``pool`` is refused by both, whatever its ``role`` says.
    """
    from local_operator.network.credentials.owner import MeshCredentialBroker

    record = types.NetworkRecord(network_id="n_select", name="sel-net", self_device_id="d_self")
    pool_admin = types.MemberRecord(
        device_id="d_" + "a" * 32, role="admin", kind="pool", lifecycle="active"
    )
    device_admin = types.MemberRecord(
        device_id="d_" + "b" * 32, role="admin", kind="device", lifecycle="active"
    )
    record.members.extend([pool_admin, device_admin])
    # The pool member sorts FIRST by id, so a filter that forgot ``kind`` picks it.
    assert pool_admin.device_id < device_admin.device_id
    assert relay.lowest_id_admin(record) == device_admin.device_id
    assert relay.lowest_id_admin(record, excluding=device_admin.device_id) == ""

    # The broker's own read, over a SAVED record: a pool admin is not an admin.
    from local_operator.network import store as store_mod

    store_mod.save(record, root)
    broker = MeshCredentialBroker(root=root, self_device="d_" + "0" * 32, network_id="n_select")
    assert broker._is_admin(pool_admin.device_id) is False  # noqa: SLF001
    assert broker._is_admin(device_admin.device_id) is True  # noqa: SLF001


# ---------------------------------------------------------------------------
# Onboarding (ratified Item 3 / B4): a pool member is PROVISIONED like a drive
# member, with session-scoped credentials only
# ---------------------------------------------------------------------------


class _ProvisionTransport:
    """The node end of ``step_provision``: answers every remote read it makes.

    The step's writes to the node are ``lop`` commands over this transport; the
    only answer its assertions depend on is the MCP report, and an empty server
    list is the honest "nothing to declare" for a node this cell did not script.
    """

    def __init__(self) -> None:
        self.commands: list[tuple[str, ...]] = []

    def run(self, argv: Sequence[str], *, timeout: float, stdin: bytes | None = None) -> Any:
        del timeout, stdin
        self.commands.append(tuple(argv))
        joined = " ".join(argv)
        if "mcp state" in joined:
            return onboard.CommandResult(
                tuple(argv), 0, json.dumps({"ok": True, "servers": []}), "", at=0.0
            )
        return onboard.CommandResult(tuple(argv), 0, "", "", at=0.0)

    def probe(self) -> Any:  # pragma: no cover — step_provision never probes
        raise AssertionError("step_provision must not probe the node")

    def connect(self, credential: Any, *, expected_fingerprint: str) -> Any:
        # pragma: no cover — the join step owns the credentialed connect
        raise AssertionError("step_provision must not connect to the node")

    def copy(self, local_path: Path, remote_path: str) -> Any:  # pragma: no cover
        raise AssertionError("step_provision must not copy to the node")

    def close(self) -> None:  # pragma: no cover — the runner closes its own transport
        return None


def _provision_outcome(
    root: Path,
    monkeypatch: pytest.MonkeyPatch,
    *,
    role: str,
    kind: str,
) -> tuple[dict[str, Any], list[list[str]], types.NetworkRecord]:
    """Run the REAL ``step_provision`` for one member row; return its receipt.

    ``build_items`` and ``grant_key_set`` are the two collaborators whose answers
    are *data* (this device's login store, and the placement write): both are
    stubbed so the cell asserts the GATE — which keys the step offers to write —
    rather than re-testing the store. Everything else (the member row, the
    capability write, the receipt) is the product's own.
    """
    from local_operator.network import identity as identity_mod
    from local_operator.network import onboard as onboard_mod
    from local_operator.network.credentials import placement as placement_mod
    from local_operator.paths import CONFIG_DIR_ENV

    monkeypatch.setenv(CONFIG_DIR_ENV, str(root))
    owner = identity_mod.mint(root, name="owner-device")
    record = types.NetworkRecord(
        network_id=_NETWORK_ID,
        name=_NETWORK_NAME,
        created_by=owner.device_id,
        self_device_id=owner.device_id,
        self_role="admin",
        self_capabilities=sorted(types.capabilities_for_role("admin")),
        listen={"address": "127.0.0.1", "port": 0, "advertised": []},
        attestation_keys={_PLANE_KEY_ID: _PLANE_PUBLIC_B64},
    )
    relay.admit(
        record,
        device_id=owner.device_id,
        public_key=owner.public_key,
        name=owner.name,
        role="admin",
        capabilities=sorted(types.capabilities_for_role("admin")),
        added_by=owner.device_id,
        added_via="self",
        root=root,
        persist=False,
    )
    relay.admit(
        record,
        device_id="d_node",
        public_key=_b64u(_public_key("7e" * 32)),
        name="cloud-node-1",
        role=role,
        kind=kind,  # type: ignore[arg-type]
        added_by=owner.device_id,
        added_via="invite:pool" if kind == "pool" else "invite",
        root=root,
    )
    store.save(record, root)

    items = [
        {
            "key": github_mod.GITHUB_KEY,
            "kind": github_mod.GITHUB_KIND,
            "label": "GitHub",
            "share": True,
        },
        # The INSTALLATION KEY, the second spelling of the same credential
        # (QA round 1, Q1): enumerated as an ordinary store secret, so the
        # provisioning write has to withhold it by the same rule the offer does.
        {
            "key": f"secret:{github_mod.APP_SECRET_NAME}",
            "kind": "store-secret",
            "label": github_mod.APP_SECRET_NAME,
            "share": True,
        },
        {"key": "provider:openai", "kind": "oauth", "label": "OpenAI", "share": True},
    ]
    written: list[list[str]] = []

    def fake_grant_key_set(document: Any, *, keys: Sequence[str], **_kw: Any) -> Any:
        written.append(list(keys))
        return list(keys), []

    monkeypatch.setattr(offers_mod, "build_items", lambda config, *, network_id="": list(items))
    monkeypatch.setattr(placement_mod, "grant_key_set", fake_grant_key_set)

    view = SimpleNamespace(
        run_id="run_1",
        kind="device_onboard",
        record={"what": {"network_id": _NETWORK_ID}},
        device={"device_id": "d_node", "name": "cloud-node-1"},
        what={"network_id": _NETWORK_ID, "role": role},
    )
    run = onboard_mod.OnboardRun(
        "ap_pool_1",
        view=view,
        transport=_ProvisionTransport(),
        resolve=lambda ref: onboard_mod.ResolvedCredential(kind="file", label="x"),
        local_cli=["lop"],
        run_local=lambda argv, **_kw: onboard_mod.CommandResult(
            tuple(argv), 0, json.dumps({"ok": True}), "", at=0.0
        ),
    )
    outcome = run.step_provision()
    assert outcome.ok, outcome.detail
    return outcome.data, written, store.load(_NETWORK_ID, root)


def test_a_pool_drive_member_is_provisioned_session_scoped_only(
    root: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """B4: a pool member gets the model logins it needs and nothing device-bound.

    The step used to refuse every pool member its credential rows. Ratified Item 3
    narrows that rather than removing it: the session-scoped keys are written as
    HOLDER rows on this device (so the pod borrows them), the device-bound GitHub
    key is withheld BY NAME in the receipt, and the node's own capability row gets
    ``broker_credential`` — the half of the pair without which it cannot dial the
    broker at all.
    """
    data, written, reloaded = _provision_outcome(root, monkeypatch, role="drive", kind="pool")

    assert written == [["provider:openai"]], written
    assert data["grants"]["granted"] == ["provider:openai"], data["grants"]
    for excluded in (github_mod.GITHUB_KEY, f"secret:{github_mod.APP_SECRET_NAME}"):
        assert {"key": excluded, "reason": "pool_session_scope_only"} in data["grants"][
            "skipped"
        ], (excluded, data["grants"]["skipped"])
    assert data["capability"]["set"] is True, data["capability"]
    pod = reloaded.member("d_node")
    assert pod is not None and "broker_credential" in pod.capabilities, pod
    assert pod.kind == "pool"


def test_a_pool_read_member_still_gets_no_credential_rows(
    root: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The read role's exclusion is UNCHANGED: narrowing a pool's grants is not
    handing every pool member a borrow path (§1.3)."""
    data, written, reloaded = _provision_outcome(root, monkeypatch, role="read", kind="pool")

    assert written == [], written
    assert data["grants"] == {"granted": [], "skipped": []}, data["grants"]
    assert "pool member" in data["grants_note"], data["grants_note"]
    assert data["capability"]["set"] is False, data["capability"]
    pod = reloaded.member("d_node")
    assert pod is not None and "broker_credential" not in pod.capabilities, pod


def test_a_hand_written_pool_admin_row_gets_no_credential_rows_either(
    root: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Code review round 2, MINOR 1: ``admin`` is refused at mint and clamped at
    admission, so a pool row that still says ``admin`` was written another way.
    It must not be provisioned like an admin — no holder rows, no capability."""
    data, written, _reloaded = _provision_outcome(root, monkeypatch, role="admin", kind="pool")

    assert written == [], written
    assert data["grants"] == {"granted": [], "skipped": []}, data["grants"]
    assert "pool member" in data["grants_note"], data["grants_note"]
    assert data["capability"] == {"set": False, "changed": False}, data["capability"]
