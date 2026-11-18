"""``lop network invite --no-credential-offers``: an invite-level policy, enforced
where the share list is BUILT and where grants are WRITTEN, never only on screen.

What these cells pin, and why each matters:

* the policy travels in the token (so the joiner can fail closed) but is ENFORCED
  from the owner's SAVED invite row, because the token's tag is keyed by the
  network secret the token itself carries — any holder can re-tag an edit;
* under the policy the owner never calls ``offers.build_items`` (it enumerates MCP
  and provider credentials), sends an explicit sealed EMPTY list, grants nothing
  and adds no ``broker_credential`` capability — witnessed at relay frame level
  over the real loopback ceremony (the ``devices`` fixture);
* a token whose copy of the policy was added, dropped or flipped is refused by
  name and its invite burned; an UNKNOWN value is refused at decode;
* a default invite is byte-for-byte the pre-feature token and hello, and a legacy
  invite row (no field) keeps the default ceremony;
* ``--role admin`` under the policy is admitted WITHOUT ``broker_credential``
  (the role's own copy, review round 1 S1) and keeps every other admin
  capability, at mint and again at admission;
* the CLI mints under the policy only through the versioned
  ``net_invite_policy`` op, so a relay that predates it refuses with NOTHING
  written (no row, no token file) instead of minting a default invite (S2);
* a joiner that refuses a share list under the policy aborts the owner with a
  named ``protocol_error``, and granted shares in a result are refused before
  anything is persisted.

Only fake credentials are seeded (``_seed`` writes a synthetic OAuth row into a
tmp root); nothing reads a real profile.
"""

from __future__ import annotations

import argparse
import dataclasses
import json
import threading
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest

from local_operator.network import cli as net_cli
from local_operator.network import invite as invite_mod
from local_operator.network import relay, store, types, wire
from local_operator.network.credentials import offers
from local_operator.network.credentials.types import BROKER_CAPABILITY
from local_operator.network.types import PairDecision
from tests.unit.network import conftest as net_fixtures
from tests.unit.network.test_cli import NETWORK, _parser, _save_network
from tests.unit.network.test_pair_offer import (
    _RecordingHandshake,
    _join_with_recorder,
    _seed,
)
from tests.unit.network.test_relay_e2e import (  # noqa: F401 — fixtures by import
    _answer_confirmation,
    _await_event,
    _init_network,
    _type_the_code,
    devices,
)

Devices = tuple[relay.RelayServer, relay.RelayServer, str, int]


def _payload_of(token: str) -> dict[str, Any]:
    return json.loads(wire.unb64u(token.split(".")[1]).decode("utf-8"))


def _mint_saved(server: relay.RelayServer, record: Any, *, policy: str, role: str = "read") -> Any:
    """Mint, save and file an invite carrying ``policy`` — built from primitives
    every build has (``mint`` without the new keyword, the token MAC, the row's
    JSON), so the SAME cell runs against a build that predates the policy and its
    frame witness shows what that build does with it: it ignores the saved field
    and still builds and sends the share list. On this build the result is
    byte-for-byte what ``mint(..., credential_offers=policy)`` returns."""
    state = store.load_secrets(record.network_id, server.root)
    minted = invite_mod.mint(record, state.secret, role=role, ttl_s=600.0)
    token, row = minted.token, minted.record
    if policy:
        payload = dict(_payload_of(token), **{"credential_offers": policy})
        body = json.dumps(payload, sort_keys=True, separators=(",", ":"), ensure_ascii=False)
        key = wire.invite_key(state.secret, record.network_id, row.invite_id)
        raw = body.encode("utf-8")
        token = f"lop1.{wire.b64u(raw)}.{wire.b64u(wire.invite_mac(key, raw))}"
        row = types.InviteRecord.from_json(dict(row.to_json(), credential_offers=policy))
    record.invites.append(row)
    store.save(record, server.root)
    store.save_invite_token(row.invite_id, token, server.root)
    return SimpleNamespace(token=token, envelope=invite_mod.decode(token), record=row)


def test_the_primitive_mint_matches_the_keyword_mint() -> None:
    """Guard for the helper above: it must not drift from the real mint."""
    material = wire.b64u(b"\x04" * 32)
    real = invite_mod.mint(
        _record(), material, role="read", invite_id="FIXED", now=5.0, credential_offers="none"
    )
    plain = invite_mod.mint(_record(), material, role="read", invite_id="FIXED", now=5.0)
    payload = dict(_payload_of(plain.token), credential_offers="none")
    body = json.dumps(payload, sort_keys=True, separators=(",", ":"), ensure_ascii=False)
    raw = body.encode("utf-8")
    tag = wire.invite_mac(wire.invite_key(material, "net-synthetic", "FIXED"), raw)
    assert real.token == f"lop1.{wire.b64u(raw)}.{wire.b64u(tag)}"


@pytest.fixture()
def build_spy(monkeypatch: pytest.MonkeyPatch) -> list[str]:
    """Record every ``build_items`` call (the store enumeration) and still run it."""
    calls: list[str] = []
    real = offers.build_items

    def _spy(config: Path, *, network_id: str = "") -> list[dict[str, Any]]:
        calls.append(network_id)
        return real(config, network_id=network_id)

    monkeypatch.setattr(offers, "build_items", _spy)
    return calls


# ---------------------------------------------------------------------------
# Token and record: the field, its absence, and its validation
# ---------------------------------------------------------------------------


def _record() -> types.NetworkRecord:
    return types.NetworkRecord(network_id="net-synthetic", name="synthetic", self_device_id="dev-a")


def test_default_token_does_not_carry_the_policy_key() -> None:
    """Default = byte-identical to builds that predate the field (old joiners)."""
    minted = invite_mod.mint(_record(), wire.b64u(b"\x01" * 32), role="drive", now=1.0)
    assert invite_mod.OFFER_POLICY_KEY not in _payload_of(minted.token)
    assert minted.record.credential_offers == ""
    assert minted.record.offers_credentials is True
    assert invite_mod.decode(minted.token).credential_offers == ""


def test_none_policy_is_carried_in_the_tagged_token_and_saved_record() -> None:
    material = wire.b64u(b"\x02" * 32)
    minted = invite_mod.mint(
        _record(), material, role="read", credential_offers=types.CREDENTIAL_OFFERS_NONE
    )
    assert _payload_of(minted.token)[invite_mod.OFFER_POLICY_KEY] == "none"
    assert minted.record.credential_offers == "none"
    assert minted.record.offers_credentials is False
    opened = invite_mod.open_token(minted.token, material_for=lambda _e: material)
    assert opened.credential_offers == "none" and opened.offers_credentials is False
    # Survives the on-disk round trip.
    again = types.InviteRecord.from_json(minted.record.to_json())
    assert again.credential_offers == "none"


def test_a_legacy_invite_row_keeps_the_default_ceremony() -> None:
    """A row saved before the field existed loads as the default policy."""
    row = types.InviteRecord.from_json({"invite_id": "LEGACY", "role": "read"})
    assert row.credential_offers == "" and row.offers_credentials is True
    invite_mod.check_offer_policy(row, {"invite_id": "LEGACY"})


def test_mint_and_decode_refuse_an_unknown_policy() -> None:
    material = wire.b64u(b"\x03" * 32)
    with pytest.raises(types.PairingRefusal) as minted:
        invite_mod.mint(_record(), material, role="read", credential_offers="some")
    assert minted.value.code == "bad_offer_policy"
    good = invite_mod.mint(_record(), material, role="read")
    envelope = dataclasses.replace(invite_mod.decode(good.token), credential_offers="bogus")
    with pytest.raises(types.PairingRefusal) as decoded:
        invite_mod.decode(invite_mod.encode(envelope, material))
    assert decoded.value.code == invite_mod.REASON_INVALID


@pytest.mark.parametrize(
    ("saved", "claimed"),
    [
        ("none", None),  # dropped (an edited token, or an old joiner)
        ("", "none"),  # added
        ("none", "bogus"),  # flipped to something else
        ("", "bogus"),
    ],
)
def test_policy_mismatch_is_refused_by_name(saved: str, claimed: str | None) -> None:
    row = types.InviteRecord(invite_id="I", credential_offers=saved)
    block: dict[str, Any] = {"invite_id": "I"}
    if claimed is not None:
        block[invite_mod.OFFER_POLICY_KEY] = claimed
    with pytest.raises(types.PairingRefusal) as refused:
        invite_mod.check_offer_policy(row, block)
    assert refused.value.code == invite_mod.REASON_OFFER_POLICY
    assert "update lop on the joining device" in refused.value.sentence


def test_matching_policies_pass() -> None:
    invite_mod.check_offer_policy(types.InviteRecord(invite_id="I"), {"invite_id": "I"})
    invite_mod.check_offer_policy(
        types.InviteRecord(invite_id="I", credential_offers="none"),
        {"invite_id": "I", invite_mod.OFFER_POLICY_KEY: "none"},
    )


# ---------------------------------------------------------------------------
# Owner side: the build site and the grant site, unit level
# ---------------------------------------------------------------------------


def test_offer_builder_never_enumerates_under_the_policy(
    tmp_path: Path, build_spy: list[str]
) -> None:
    handshake = SimpleNamespace(peer_capabilities=[wire.PAIR_OFFER_V1])
    send, items, detail = relay._pair_offer_for(  # noqa: SLF001 — the build site itself
        handshake, tmp_path, "net-synthetic", offers_allowed=False
    )
    assert (send, items) == (True, [])
    assert detail["offer_digest"] == offers.digest_of([])
    assert detail["offer_policy"] == "none"
    assert build_spy == []
    # The default path still builds (the control for the spy itself).
    relay._pair_offer_for(handshake, tmp_path, "net-synthetic")  # noqa: SLF001
    assert build_spy == ["net-synthetic"]


def test_grants_refuse_named_shares_and_never_add_the_broker_capability(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Independent of the offer: a decision naming shares still grants nothing."""
    touched: list[str] = []
    monkeypatch.setattr(
        relay, "set_member_capabilities", lambda *a, **k: touched.append("capability")
    )
    recorded: list[dict[str, Any]] = []
    audit = SimpleNamespace(record=lambda event: recorded.append(event.detail))
    decision = PairDecision(invite_id="I", decision="admit", matched=True, shares=["openai"])
    granted, reduced = relay._grant_pair_shares(  # noqa: SLF001 — the grant site itself
        _record(),
        joiner_id="dev-b",
        decision=decision,
        offer_items=[{"key": "openai", "kind": "oauth-rotating", "label": "x", "share": True}],
        owner_name="a",
        root=tmp_path,
        audit=audit,
        offers_allowed=False,
    )
    assert (granted, reduced) == ([], [])
    assert touched == []
    assert recorded and recorded[0]["skipped"] == "invite_policy_none"


class _Wire:
    """A sealed-frame recorder standing in for the socket + codec pair (identity seal)."""

    def __init__(self, replies: list[dict[str, Any]] | None = None) -> None:
        self.sent: list[dict[str, Any]] = []
        self._replies = list(replies or [])

    # socket half
    def sendall(self, frame: dict[str, Any]) -> None:
        self.sent.append(frame)

    # codec half (identity: the frames are already dicts)
    def seal(self, frame: dict[str, Any]) -> dict[str, Any]:
        return frame

    def open(self, frame: dict[str, Any]) -> dict[str, Any]:
        return frame

    # reader half
    def read_record_payload(self, _deadline: float) -> dict[str, Any]:
        return self._replies.pop(0)


def _join_helpers() -> dict[str, Any]:
    from local_operator.network.handshake import pair_abort_frame

    return {
        "pair_abort_frame": pair_abort_frame,
        "invite_mod": SimpleNamespace(remaining_seconds=lambda _envelope: 60.0),
        "pair_timeout_seconds": lambda _remaining: 5.0,
    }


def test_joiner_fails_closed_on_a_non_empty_list_under_the_policy() -> None:
    """R1-3: the refusal also ABORTS the owner on the wire (``protocol_error``),
    which the owner's ready read treats as terminal — the invite is burned at once
    instead of waiting out a retryable ``timeout``."""
    none_env = SimpleNamespace(offers_credentials=False)
    row = [{"key": "openai", "kind": "oauth-rotating", "label": "x", "share": True}]
    wire_ = _Wire()
    with pytest.raises(types.MeshRefusal) as refused:
        net_cli._refuse_offer_against_policy(  # noqa: SLF001
            none_env, row, sock=wire_, codec=wire_, helpers=_join_helpers()
        )
    assert refused.value.code == "protocol_error"
    assert [(f["op"], f["reason"]) for f in wire_.sent] == [("net_pair_abort", "protocol_error")]
    assert "no credential offers" in wire_.sent[0]["detail"]
    quiet = _Wire()
    net_cli._refuse_offer_against_policy(  # noqa: SLF001 — empty is fine
        none_env, [], sock=quiet, codec=quiet, helpers=_join_helpers()
    )
    net_cli._refuse_offer_against_policy(  # noqa: SLF001 — default unaffected
        SimpleNamespace(offers_credentials=True),
        row,
        sock=quiet,
        codec=quiet,
        helpers=_join_helpers(),
    )
    assert quiet.sent == []


def test_joiner_refusal_still_raises_when_the_abort_cannot_be_sent() -> None:
    class _Gone(_Wire):
        def sendall(self, frame: dict[str, Any]) -> None:
            raise BrokenPipeError

    gone = _Gone()
    with pytest.raises(types.MeshRefusal):
        net_cli._refuse_offer_against_policy(  # noqa: SLF001
            SimpleNamespace(offers_credentials=False),
            [{"key": "openai"}],
            sock=gone,
            codec=gone,
            helpers=_join_helpers(),
        )


def test_owner_burns_the_invite_when_the_joiner_aborts_for_the_policy(
    devices: Devices,  # noqa: F811
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """R1-3 end to end: a (synthetic) broken owner sends a share list under a
    ``none`` invite; the joiner refuses AND aborts, so the owner records the
    NAMED terminal refusal (``protocol_error``) and consumes the invite.

    Measured on the previous head (a1e560da) with this same cell: the owner also
    consumed the invite there, because the joiner's socket close fails the
    owner's ready read — but as an anonymous ``error`` (audit ``failed``,
    ``stage=ready_read``), which is what the outcome assertion below tells apart.

    The owner is made to break the policy by forcing its offer builder open
    (``offers_allowed`` ignored) — no honest build of this owner does that."""
    server_a, server_b, host, port = devices
    _seed(server_a.root, "openai", {"refresh": "r", "access": "a", "email": "qa@example.com"})
    record = _init_network(server_a)
    minted = _mint_saved(server_a, record, policy="none")
    real_offer_for = relay._pair_offer_for  # noqa: SLF001

    def _broken_owner(handshake: Any, root: Path, network_id: str, **_kw: Any) -> Any:
        return real_offer_for(handshake, root, network_id, offers_allowed=True)

    monkeypatch.setattr(relay, "_pair_offer_for", _broken_owner)
    _type_the_code(monkeypatch)
    with pytest.raises(types.MeshRefusal) as refused:
        _join_with_recorder(server_b, host=host, port=port, minted=minted)
    assert refused.value.code == "protocol_error"
    _await_event(server_a, "pairing_refused")

    def _row() -> Any:
        return store.load(record.network_id, server_a.root).invite(minted.record.invite_id)

    # The invite's outcome is written after the refusal row (see ``_await_event``).
    net_fixtures.wait_for(lambda: getattr(_row(), "state", "") == "consumed")
    row = _row()
    assert row is not None and row.state == "consumed"
    assert row.outcome == "protocol_error"
    member = store.load(record.network_id, server_a.root).member(server_b.identity.device_id)
    assert member is None or not member.active


@pytest.mark.parametrize("offers_credentials", [False, True])
def test_finish_pairing_refuses_granted_shares_under_the_policy(
    monkeypatch: pytest.MonkeyPatch, offers_credentials: bool
) -> None:
    """QA gap: the joiner's SECOND fail-closed branch (granted shares in the
    result) — no honest owner produces it, so a synthetic result frame does.
    Under ``none`` nothing is persisted; the default control reaches persistence."""
    persisted: list[Any] = []

    class _Persisted(Exception):
        pass

    def _persist(*_a: Any, **_k: Any) -> None:
        persisted.append(True)
        raise _Persisted

    monkeypatch.setattr(net_cli, "_persist_join", _persist)
    result = {"op": "net_pair_result", "req": 1, "admit": True, "shares": ["openai"]}
    wire_ = _Wire([result])
    envelope = SimpleNamespace(offers_credentials=offers_credentials)
    call: dict[str, Any] = dict(
        typed="123456",
        sock=wire_,
        codec=wire_,
        reader=wire_,
        envelope=envelope,
        host="127.0.0.1:1",
        identity=SimpleNamespace(),
        store=SimpleNamespace(),
        handshake=SimpleNamespace(peer_endpoints=()),
        advertised=[],
        fingerprint="f",
        offer_view={"state": "empty", "items": []},
        helpers=_join_helpers(),
    )
    if offers_credentials:
        with pytest.raises(_Persisted):
            net_cli._finish_pairing(**call)  # noqa: SLF001
        assert persisted == [True]
    else:
        with pytest.raises(types.MeshRefusal) as refused:
            net_cli._finish_pairing(**call)  # noqa: SLF001
        assert refused.value.code == "protocol_error"
        assert "granted credentials" in refused.value.sentence
        assert persisted == []


def test_the_invite_parser_accepts_the_flag() -> None:
    parser = _parser()
    args = parser.parse_args(["network", "invite", "--role", "read", "--no-credential-offers"])
    assert args.no_credential_offers is True
    assert parser.parse_args(["network", "invite"]).no_credential_offers is False


# ---------------------------------------------------------------------------
# The real ceremony over loopback: frame-level witnesses
# ---------------------------------------------------------------------------


def test_policy_none_sends_zero_items_builds_nothing_and_grants_nothing(
    devices: Devices,  # noqa: F811
    monkeypatch: pytest.MonkeyPatch,
    build_spy: list[str],
) -> None:
    server_a, server_b, host, port = devices
    # A fake login the default ceremony WOULD offer (see the default cell below).
    _seed(server_a.root, "openai", {"refresh": "r", "access": "a", "email": "qa@example.com"})
    record = _init_network(server_a)
    minted = _mint_saved(server_a, record, policy="none")
    _type_the_code(monkeypatch)
    answered: dict[str, Any] = {}
    thread = threading.Thread(
        target=lambda: answered.update(_answer_confirmation(server_a) or {}), daemon=True
    )
    thread.start()
    try:
        lines, payload = _join_with_recorder(server_b, host=host, port=port, minted=minted)
    finally:
        thread.join(20)
    assert answered, "the inviter's person never answered"
    crypto = _RecordingHandshake.last
    assert [frame.get("op") for frame in crypto.opened] == ["net_pair_offer", "net_pair_result"]
    offer_frame, result_frame = crypto.opened
    assert offer_frame["items"] == []
    assert offer_frame["digest"] == offers.digest_of([])
    assert result_frame.get("shares") == []
    assert build_spy == [], "the owner enumerated its store under --no-credential-offers"
    fresh = store.load(record.network_id, server_a.root)
    member = fresh.member(server_b.identity.device_id)
    assert member is not None and member.active
    assert BROKER_CAPABILITY not in member.capabilities
    from local_operator.network.credentials import placement as placement_mod

    document = placement_mod.PlacementDocument.load(
        record.network_id, server_a.root, self_device=server_a.identity.device_id
    )
    assert document.entry("openai") is None
    assert payload["shares"] == []
    assert not any(line.startswith("available here:") for line in lines)


def test_default_invite_still_builds_and_sends_the_offer(
    devices: Devices,  # noqa: F811
    monkeypatch: pytest.MonkeyPatch,
    build_spy: list[str],
) -> None:
    """The control: same seed, no flag — the share list is built and sent."""
    server_a, server_b, host, port = devices
    _seed(server_a.root, "openai", {"refresh": "r", "access": "a", "email": "qa@example.com"})
    record = _init_network(server_a)
    minted = _mint_saved(server_a, record, policy="")
    _type_the_code(monkeypatch)
    thread = threading.Thread(target=lambda: _answer_confirmation(server_a), daemon=True)
    thread.start()
    try:
        _join_with_recorder(server_b, host=host, port=port, minted=minted)
    finally:
        thread.join(20)
    offer_frame = _RecordingHandshake.last.opened[0]
    assert [item["key"] for item in offer_frame["items"]] == ["openai"]
    assert build_spy, "the default ceremony stopped building its offer"


def test_admin_mint_under_the_policy_drops_only_the_broker_capability() -> None:
    """S1/R1-1 at mint: the ``admin`` role carries ``broker_credential``; under
    ``none`` the token, the envelope and the saved row all lack it and keep every
    OTHER admin capability. The default admin mint is the control."""
    material = wire.b64u(b"\x05" * 32)
    admin = set(types.capabilities_for_role("admin"))
    assert BROKER_CAPABILITY in admin  # the premise the finding rests on
    default = invite_mod.mint(_record(), material, role="admin")
    assert set(default.record.capabilities) == admin
    assert set(_payload_of(default.token)["capabilities"]) == admin
    none = invite_mod.mint(_record(), material, role="admin", credential_offers="none")
    for caps in (
        none.record.capabilities,
        none.envelope.capabilities,
        _payload_of(none.token)["capabilities"],
    ):
        assert set(caps) == admin - {BROKER_CAPABILITY}
    # read/drive never carried it: their sets are unchanged by the policy.
    for role in ("read", "drive"):
        plain = invite_mod.mint(_record(), material, role=role)
        policed = invite_mod.mint(_record(), material, role=role, credential_offers="none")
        assert policed.record.capabilities == plain.record.capabilities


def test_admission_filters_a_none_row_whatever_its_capability_list_says() -> None:
    """S1/R1-1 at admission: the relay reads the row's POLICY, not only its list."""
    record = _record()
    caps = sorted(types.capabilities_for_role("admin"))
    record.invites.append(
        types.InviteRecord(invite_id="N", role="admin", capabilities=caps, credential_offers="none")
    )
    record.invites.append(types.InviteRecord(invite_id="D", role="admin", capabilities=caps))
    admitted = relay._invite_capabilities  # noqa: SLF001 — the admission read itself
    assert admitted(record, "N") == set(caps) - {BROKER_CAPABILITY}
    assert admitted(record, "D") == set(caps)
    assert admitted(record, "missing") == set()


@pytest.mark.parametrize("policy", ["none", ""])
def test_admin_ceremony_admits_without_broker_only_under_the_policy(
    devices: Devices,  # noqa: F811
    monkeypatch: pytest.MonkeyPatch,
    policy: str,
) -> None:
    """S1/R1-1 over the real loopback ceremony, ``--role admin``. The row is
    written with the UNFILTERED admin list (the primitive helper), so this cell
    proves the admission-side filter on its own; the sealed result frame's member
    and the persisted member both lack ``broker_credential`` under ``none`` and
    carry the ordinary admin set by default (the control)."""
    server_a, server_b, host, port = devices
    _seed(server_a.root, "openai", {"refresh": "r", "access": "a", "email": "qa@example.com"})
    record = _init_network(server_a)
    minted = _mint_saved(server_a, record, policy=policy, role="admin")
    _type_the_code(monkeypatch)
    thread = threading.Thread(target=lambda: _answer_confirmation(server_a), daemon=True)
    thread.start()
    try:
        _join_with_recorder(server_b, host=host, port=port, minted=minted)
    finally:
        thread.join(20)
    result_frame = _RecordingHandshake.last.opened[-1]
    assert result_frame["op"] == "net_pair_result"
    admin = set(types.capabilities_for_role("admin"))
    expected = admin - {BROKER_CAPABILITY} if policy else admin
    assert set(result_frame["member"]["capabilities"]) == expected
    member = store.load(record.network_id, server_a.root).member(server_b.identity.device_id)
    assert member is not None and member.active and member.role == "admin"
    assert set(member.capabilities) == expected
    if policy:
        assert result_frame.get("shares") == []


@pytest.mark.parametrize(
    ("saved", "presented"),
    [
        # Literals, not the constant, so the module collects on a build without it.
        ("none", ""),  # dropped
        ("", "none"),  # added
        ("none", "bogus"),  # flipped
    ],
)
def test_a_tampered_token_cannot_change_the_policy(
    devices: Devices,  # noqa: F811
    monkeypatch: pytest.MonkeyPatch,
    build_spy: list[str],
    saved: str,
    presented: str,
) -> None:
    """The holder re-tags an edited payload (it has the material): refused, burned,
    nothing enumerated — the saved row decides, the claim is only compared."""
    server_a, server_b, host, port = devices
    _seed(server_a.root, "openai", {"refresh": "r", "access": "a", "email": "qa@example.com"})
    record = _init_network(server_a)
    minted = _mint_saved(server_a, record, policy=saved)
    edited = dataclasses.replace(minted.envelope, credential_offers=presented)
    forged = SimpleNamespace(
        token=invite_mod.encode(edited, minted.envelope.material), envelope=edited
    )
    _type_the_code(monkeypatch)
    with pytest.raises(types.PairingRefusal) as refused:
        _join_with_recorder(server_b, host=host, port=port, minted=forged)
    assert refused.value.code == invite_mod.REASON_OFFER_POLICY
    assert "update lop on the joining device" in refused.value.sentence
    assert build_spy == []
    _await_event(server_a, "pairing_refused")
    row = store.load(record.network_id, server_a.root).invite(minted.record.invite_id)
    assert row is not None and row.state == "consumed"
    member = store.load(record.network_id, server_a.root).member(server_b.identity.device_id)
    assert member is None or not member.active


# ---------------------------------------------------------------------------
# `lop network invite --no-credential-offers`: the mint command itself
# ---------------------------------------------------------------------------


def _invite_args(*, no_offers: bool) -> argparse.Namespace:
    return argparse.Namespace(
        network=NETWORK,
        role="read",
        expires=600.0,
        hosts="",
        device="",
        print_token=False,
        json=True,
        no_credential_offers=no_offers,
    )


@pytest.fixture()
def minting_root(root: Path, monkeypatch: pytest.MonkeyPatch) -> tuple[Path, list[str]]:
    """A local network (no relay running: the local mint path) and an adoption spy."""
    monkeypatch.setenv("LOCAL_OPERATOR_CONFIG_DIR", str(root))
    record = _save_network(root, NETWORK, "home")
    record.self_capabilities = sorted(types.capabilities_for_role("admin"))
    store.save(record, root)
    store.save_secrets(
        types.SecretState(network_id=NETWORK, epoch=1, secret=wire.b64u(b"s" * 32)), root
    )
    adopted: list[str] = []
    from local_operator.mcp import credential_sources

    monkeypatch.setattr(
        credential_sources,
        "adopt_literal_credentials",
        lambda config: adopted.append(str(config)) or {},
    )
    return root, adopted


@pytest.mark.parametrize("no_offers", [True, False])
def test_invite_command_saves_the_policy_and_skips_adoption_only_when_set(
    minting_root: tuple[Path, list[str]],
    capsys: pytest.CaptureFixture[str],
    no_offers: bool,
) -> None:
    root, adopted = minting_root
    assert net_cli._cmd_invite(_invite_args(no_offers=no_offers)) == 0  # noqa: SLF001
    payload = json.loads(capsys.readouterr().out)
    expected = types.CREDENTIAL_OFFERS_NONE if no_offers else ""
    assert payload["credential_offers"] == expected
    saved = store.load(NETWORK, root).invite(payload["invite_id"])
    assert saved is not None and saved.credential_offers == expected
    token = Path(payload["path"]).read_text(encoding="utf-8").strip()
    assert _payload_of(token).get(invite_mod.OFFER_POLICY_KEY, "") == expected
    # The credential walk that only feeds an offer: skipped exactly when no offer
    # will be built, unchanged otherwise.
    assert adopted == ([] if no_offers else [str(root)])


def _route_control_to(server: relay.RelayServer, monkeypatch: pytest.MonkeyPatch) -> list[str]:
    """Point the CLI's ``_relay_call`` at ``server``'s REAL ``control_dispatch``
    (the socket hop is the only thing skipped), so the reply — including an
    unserved-op refusal — and every side effect come from the relay itself."""
    ops: list[str] = []

    def _request(_record: Any, op: str, *, timeout: float = 5.0, **fields: Any) -> Any:
        ops.append(op)
        return server.control_dispatch(op, {"op": op, "req": 1, **fields})

    monkeypatch.setattr(store, "find_own_relay", lambda *a, **k: object())
    monkeypatch.setattr(relay, "control_request", _request)
    return ops


def _as_older_relay(server: relay.RelayServer, monkeypatch: pytest.MonkeyPatch) -> None:
    """``server`` as the PR base serves it: no ``net_invite_policy`` op, and a
    ``net_invite`` that ignores the unknown ``credential_offers`` field (the base
    ``_ctl_invite`` never read it) — so it still mints and files a token."""
    current = server._control_handlers  # noqa: SLF001
    real_invite = server._ctl_invite  # noqa: SLF001

    def _old_table() -> dict[str, Any]:
        table = dict(current())
        table.pop("net_invite_policy", None)
        table["net_invite"] = lambda frame: real_invite(
            {k: v for k, v in frame.items() if k != "credential_offers"}
        )
        return table

    monkeypatch.setattr(server, "_control_handlers", _old_table)


def _mint_args(record: Any, *, no_offers: bool, role: str = "read") -> argparse.Namespace:
    args = _invite_args(no_offers=no_offers)
    args.network = record.network_id
    args.role = role
    return args


def _outbox_tokens(root: Path) -> list[str]:
    outbox = store.outbox_dir(root)
    return sorted(p.name for p in outbox.glob("*.invite")) if outbox.is_dir() else []


def test_an_older_running_relay_is_never_asked_to_mint_under_the_policy(
    devices: Devices,  # noqa: F811
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    """S2/R1-2: against a relay that predates the policy, the flag mints NOTHING:
    no invite row, no token file, no adoption — the refusal precedes any write.
    The default mint against the same older relay is the negative control: it does
    write a row and a redeemable token, which is exactly what the old
    ``net_invite``-with-a-field path left behind."""
    server_a, _server_b, _host, _port = devices
    record = _init_network(server_a)
    _as_older_relay(server_a, monkeypatch)
    ops = _route_control_to(server_a, monkeypatch)
    adopted: list[str] = []
    from local_operator.mcp import credential_sources

    monkeypatch.setattr(
        credential_sources, "adopt_literal_credentials", lambda c: adopted.append(str(c)) or {}
    )
    before_rows = list(store.load(record.network_id, server_a.root).invites)
    before_tokens = _outbox_tokens(server_a.root)

    assert net_cli._cmd_invite(_mint_args(record, no_offers=True)) == 2  # noqa: SLF001
    err = capsys.readouterr().err
    assert "no invite or token was created" in err
    assert ops == ["net_invite_policy"]
    assert store.load(record.network_id, server_a.root).invites == before_rows
    assert _outbox_tokens(server_a.root) == before_tokens
    assert adopted == []

    # Control: the same older relay DOES mutate for a default mint.
    assert net_cli._cmd_invite(_mint_args(record, no_offers=False)) == 0  # noqa: SLF001
    assert ops[-1] == "net_invite"
    rows = store.load(record.network_id, server_a.root).invites
    assert len(rows) == len(before_rows) + 1 and rows[-1].credential_offers == ""
    assert len(_outbox_tokens(server_a.root)) == len(before_tokens) + 1


@pytest.mark.parametrize("role", ["read", "admin"])
def test_the_running_relay_mints_the_policy_through_the_versioned_op(
    devices: Devices,  # noqa: F811
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
    role: str,
) -> None:
    """The current relay: ``net_invite_policy`` saves ``none`` on the row and in
    the token file it writes, and (admin) without the broker capability; a default
    mint still goes through the unchanged ``net_invite``."""
    server_a, _server_b, _host, _port = devices
    record = _init_network(server_a)
    ops = _route_control_to(server_a, monkeypatch)
    assert net_cli._cmd_invite(_mint_args(record, no_offers=True, role=role)) == 0  # noqa: SLF001
    payload = json.loads(capsys.readouterr().out)
    assert ops == ["net_invite_policy"] and payload["credential_offers"] == "none"
    row = store.load(record.network_id, server_a.root).invite(payload["invite_id"])
    assert row is not None and row.credential_offers == "none"
    assert BROKER_CAPABILITY not in row.capabilities
    token = Path(payload["path"]).read_text(encoding="utf-8").strip()
    assert _payload_of(token)[invite_mod.OFFER_POLICY_KEY] == "none"
    assert net_cli._cmd_invite(_mint_args(record, no_offers=False, role=role)) == 0  # noqa: SLF001
    assert ops[-1] == "net_invite"
    default = json.loads(capsys.readouterr().out)
    assert default["credential_offers"] == ""


@pytest.mark.parametrize("policy", [None, "", "bogus"])
def test_the_versioned_op_refuses_a_missing_or_default_policy(
    devices: Devices,  # noqa: F811
    policy: str | None,
) -> None:
    server_a, _server_b, _host, _port = devices
    record = _init_network(server_a)
    frame: dict[str, Any] = {"op": "net_invite_policy", "req": 1, "network": record.network_id}
    if policy is not None:
        frame["credential_offers"] = policy
    reply = server_a.control_dispatch("net_invite_policy", frame)
    assert reply["op"] == "error" and reply["code"] == "bad_offer_policy"
    assert store.load(record.network_id, server_a.root).invites == []


def test_net_invite_policy_is_a_local_op_only() -> None:
    assert "net_invite_policy" in types.LOCAL_OPS
    assert "net_invite_policy" not in types.OP_CAPABILITY
