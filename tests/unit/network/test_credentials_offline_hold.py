"""The offline hold: a node keeps its borrowed login while the owner sleeps, bounded.

THE INCIDENT (2026-11-13): the owner, a laptop, went into clamshell sleep for
24 minutes. Every node's borrowed model login lapsed at the 15-minute grant end
because nobody was awake to renew it. The operator approved (C5, ask a-e0c9,
"Yes, hold up to 1 hour"): the OWNER states ``hold_until_ms = min(token expiry,
now + offline_hold_s)`` on each grant, and the node serves the grant it already
holds until then while the owner is unreachable, capped on its own clocks.

Every bound is pinned here with a real negative: an older owner (no field) gets
no extension; a hostile owner's number is capped by the borrower's receipt; the
sleep-inclusive clock ends the hold even when the wall clock stepped back; a held
login the provider is resting answers ``quota_blocked`` so the failover chain
moves on; a 401 on a held bearer drops it; receipts never carry the token.

Real path where it matters: two relays on loopback (the repo's ``mesh`` fixture)
with the owner relay stopped, which is what a closed lid does. Synthetic ids only.
"""

from __future__ import annotations

import dataclasses
import json
import time
from typing import Any

import pytest
import yaml

from local_operator.network import audit
from local_operator.network.credentials import (
    OFFLINE_HOLD_MAX_S,
    OFFLINE_HOLD_S,
    effective,
    offline_hold_s,
)
from local_operator.network.credentials import owner as owner_mod
from local_operator.network.credentials.client import GrantCache, MeshCredentialClient
from local_operator.network.credentials.types import BrokerError, CredentialRef, Grant
from local_operator.network.store import audit_path
from tests.unit.network import conftest as net_fixtures
from tests.unit.network.test_credentials_held_offline import (
    _client,
    _state,
    _stop_owner,
)
from tests.unit.network.test_credentials_owner import (  # noqa: F401 — fixtures by import
    STUB_PROVIDER,
    idp,
    stub_provider,
)
from tests.unit.network.test_credentials_real_link import (  # noqa: F401 — fixtures by import
    _pull,
    _share,
    mesh,
)
from tests.unit.network.test_relay_e2e import (  # noqa: F401 — fixtures by import
    _pair,
    devices,
    serve_shaped_relay,
)

# -- pure units: config, the owner's stamp, the wire, the hold formula ----------


def _write_config(root: Any, **credentials: float) -> None:
    (root / "config.yml").write_text(
        yaml.safe_dump({"values": {"network": {"credentials": dict(credentials)}}}),
        encoding="utf-8",
    )


def test_offline_hold_defaults_to_the_approved_hour_and_is_clamped(tmp_path: Any) -> None:
    assert OFFLINE_HOLD_S == 3600.0
    assert offline_hold_s(tmp_path) == 3600.0
    # Never shorter than the grant (that would shrink the pre-existing window)...
    _write_config(tmp_path, grant_ttl_s=900.0, offline_hold_s=10.0)
    assert offline_hold_s(tmp_path) == 900.0
    # ...never past the six-hour ceiling, and a garbled value is the default.
    _write_config(tmp_path, offline_hold_s=10**9)
    assert offline_hold_s(tmp_path) == OFFLINE_HOLD_MAX_S
    (tmp_path / "config.yml").write_text(
        yaml.safe_dump({"values": {"network": {"credentials": {"offline_hold_s": "soon"}}}}),
        encoding="utf-8",
    )
    assert offline_hold_s(tmp_path) == OFFLINE_HOLD_S


def test_the_owner_states_min_token_expiry_and_hold_never_below_the_grant(tmp_path: Any) -> None:
    now = 1_000_000_000_000
    hour = 3_600_000
    # Model OAuth with a 3 h token: the hold is the hour.
    assert owner_mod._offline_hold_until_ms(tmp_path, now, now + 3 * hour, now + 900_000) == (
        now + hour
    )
    # A GitHub App token lives an hour on its own: the hold is its expiry.
    assert owner_mod._offline_hold_until_ms(tmp_path, now, now + 50 * 60_000, now + 900_000) == (
        now + 50 * 60_000
    )
    # A PAT / static key (no known expiry): the hold alone bounds it.
    assert owner_mod._offline_hold_until_ms(tmp_path, now, 0, now + 900_000) == now + hour
    # A token about to expire: never below the grant end the owner already promised.
    assert owner_mod._offline_hold_until_ms(tmp_path, now, now + 1000, now + 1000) == now + 1000


def _grant(**fields: Any) -> Grant:
    base: dict[str, Any] = {
        "access_token": "synthetic-bearer-xyz",
        "kind": "bearer",
        "token_expires_at_ms": 0,
        "grant_expires_at_ms": 0,
        "credential_ref": CredentialRef(
            owner_device="d_owner",
            owner_device_name="owner",
            provider="p",
            kind="oauth",
            credential_id=7,
        ),
        "served_by": "d_owner",
        "grant_id": "g_synthetic",
    }
    base.update(fields)
    return Grant(**base)


def test_hold_until_rides_the_wire_and_an_older_peer_omits_it() -> None:
    stated = _grant(hold_until_ms=123_456)
    assert stated.to_detail()["hold_until_ms"] == 123_456
    assert Grant.from_detail(stated.to_detail()).hold_until_ms == 123_456
    old = _grant()
    assert "hold_until_ms" not in old.to_detail(), "frame to an older peer must not change"
    assert Grant.from_detail(old.to_detail()).hold_until_ms == 0
    # Process-local stamps never travel.
    stamped = _grant(received_wall=1.0, received_boot=2.0, held_until_ms=3)
    assert not {"received_wall", "received_boot", "held_until_ms"} & set(stamped.to_detail())


def test_hold_end_formula_bounds() -> None:
    wall = 1_000_000.0  # receipt, seconds
    boot = 500.0
    grant_exp = int(wall * 1000) + 900_000
    common: dict[str, Any] = {
        "grant_expires_at_ms": grant_exp,
        "received_wall": wall,
        "received_boot": boot,
    }
    cap = 3600.0
    # T7 older owner (no field): ends at the grant end exactly.
    assert _grant(**common).hold_end_ms(cap, boot + 1) == grant_exp
    # T1: the owner's hour, inside a 3 h token.
    hour = int(wall * 1000) + 3_600_000
    g = _grant(**common, hold_until_ms=hour, token_expires_at_ms=hour + 7_200_000)
    assert g.hold_end_ms(cap, boot + 1) == hour
    # T8 hostile owner: 10**18 is capped at receipt + the borrower's own cap.
    hostile = _grant(**common, hold_until_ms=10**18)
    assert hostile.hold_end_ms(cap, boot + 1) == int(wall * 1000 + cap * 1000)
    # ...and a SHORTER borrower cap wins over the owner's hour.
    assert g.hold_end_ms(1200.0, boot + 1) == int(wall * 1000 + 1_200_000)
    # T8 sleep across the boundary / wall clock stepped back: the sleep-inclusive
    # age alone ends the hold, whatever the wall clock reads.
    assert hostile.hold_end_ms(cap, boot + cap) == 0
    # A token expiring before the stated hold still bounds it (defence in depth).
    short = _grant(**common, hold_until_ms=hour, token_expires_at_ms=grant_exp + 1000)
    assert short.hold_end_ms(cap, boot + 1) == grant_exp + 1000
    # A grant this process never stamped on receipt gets no extension.
    assert _grant(grant_expires_at_ms=grant_exp, hold_until_ms=hour).hold_end_ms(cap, boot + 1) == (
        grant_exp
    )


def test_grant_cache_stamps_receipt_once_and_still_has_no_serializer() -> None:
    """T11: the stamp is taken on the borrower's clocks; nothing becomes durable."""
    cache = GrantCache()
    grant = _grant(grant_expires_at_ms=int(time.time() * 1000) + 60_000)
    cache.put("p", "s", grant)
    first = (grant.received_wall, grant.received_boot)
    assert first[0] and first[1]
    cache.put("p", "s", grant)
    assert (grant.received_wall, grant.received_boot) == first, "a re-put restarted the hold"
    for name in ("to_json", "to_dict", "save", "dump", "__getstate__", "__reduce__"):
        assert name not in vars(GrantCache), name


# -- the real path: two relays, the owner stopped ------------------------------


def _hold_records(root: Any) -> list[dict[str, Any]]:
    path = audit_path(root)
    if not path.exists():
        return []
    rows = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line]
    return [row for row in rows if str(row.get("event", "")).startswith("credential.held")]


@pytest.fixture
def short_hold(mesh: Any) -> tuple[MeshCredentialClient, Grant]:  # noqa: F811
    """grant_ttl 2 s and an offline hold of 5 s on BOTH devices, then one borrow."""
    for root in (mesh.a.root, mesh.b.root):
        _write_config(root, grant_ttl_s=2.0, offline_hold_s=5.0)
    _share(mesh)
    _pull(mesh)
    client = _client(mesh)
    grant = client.request_grant_sync(STUB_PROVIDER, session_id="sess-1", model_id="stub-model")
    assert isinstance(grant, Grant), grant
    assert grant.hold_until_ms > grant.grant_expires_at_ms, "the owner stated no hold"
    return client, grant


def test_an_outage_past_the_grant_is_served_and_past_the_hold_lapses_honestly(
    mesh: Any, short_hold: Any  # noqa: F811
) -> None:
    """T1 + T10: served past ``grant_ttl``, refused past the hold, receipts written."""
    client, grant = short_hold
    posts = len(mesh.idp.posts)
    _stop_owner(mesh)
    assert net_fixtures.wait_for(lambda: time.time() * 1000 > grant.grant_expires_at_ms + 100)

    held = client.request_grant_sync(STUB_PROVIDER, session_id="sess-1", model_id="stub-model")
    assert isinstance(held, Grant), "lapsed at the grant end: the hold did not engage"
    assert held.access_token == grant.access_token and held.held_offline
    assert held.grant_expires_at_ms == grant.grant_expires_at_ms, "the grant itself was rewritten"
    assert held.held_until_ms > grant.grant_expires_at_ms
    access = _state(mesh, client)
    assert (access.state, access.usable) == ("held_offline", True)
    assert access.last_grant_expires_at == pytest.approx(held.held_until_ms / 1000)
    # A second serve inside the minute is counted, not written.
    assert isinstance(
        client.request_grant_sync(STUB_PROVIDER, session_id="sess-1", model_id="stub-model"),
        Grant,
    )
    assert len(mesh.idp.posts) == posts, "a held grant refreshed somewhere"
    served = [r for r in _hold_records(mesh.b.root) if r["event"] == "credential.held_served"]
    assert len(served) == 1, served
    assert served[0]["detail"]["serves"] == 1
    assert served[0]["detail"]["grant_id"] == grant.grant_id
    assert served[0]["detail"]["act"] == grant.credential_ref.owner_device
    assert served[0]["detail"]["hold_until"] == held.held_until_ms

    assert net_fixtures.wait_for(
        lambda: time.time() * 1000 > held.held_until_ms + 100, timeout_s=15.0
    )
    lapsed = client.request_grant_sync(STUB_PROVIDER, session_id="sess-1", model_id="stub-model")
    assert isinstance(lapsed, BrokerError) and lapsed.code == "owner_offline", lapsed
    # The copy says what is true: the owner is away and when it was last seen;
    # it never claims the login was revoked or that this device is signed out.
    assert "owns it and was last seen" in lapsed.message, lapsed.message
    assert (_state(mesh, client).state, _state(mesh, client).usable) == (
        "owner_unavailable",
        False,
    )
    assert client.grants.peek(STUB_PROVIDER, "sess-1") is None, "a lapsed hold kept its bearer"
    records = _hold_records(mesh.b.root)
    ended = [r for r in records if r["event"] == "credential.held_lapsed"]
    assert len(ended) == 1, records
    assert ended[0]["detail"]["serves"] == 2 and ended[0]["detail"]["pending"] == 1
    # R1-m1 / Q1: the lapse names the limit the serves were held to, never the plain
    # grant end the cached copy (``held_until_ms == 0``) used to report.
    assert ended[0]["detail"]["bound"] == served[0]["detail"]["bound"] != "grant", records
    # No token, anywhere on this device's disk records.
    raw = audit_path(mesh.b.root).read_text(encoding="utf-8")
    assert grant.access_token not in raw
    for state_file in mesh.b.root.rglob("placement.state.json"):
        assert grant.access_token not in state_file.read_text(encoding="utf-8")


def test_an_older_owner_that_states_no_hold_gets_no_extension(
    mesh: Any, monkeypatch: pytest.MonkeyPatch  # noqa: F811
) -> None:
    """T7: the field is absent on the wire, so the hold ends at the grant end."""
    monkeypatch.setattr(owner_mod, "_offline_hold_until_ms", lambda *_a: 0)
    for root in (mesh.a.root, mesh.b.root):
        _write_config(root, grant_ttl_s=2.0, offline_hold_s=5.0)
    _share(mesh)
    _pull(mesh)
    client = _client(mesh)
    grant = client.request_grant_sync(STUB_PROVIDER, session_id="sess-1")
    assert isinstance(grant, Grant), grant
    assert grant.hold_until_ms == 0, grant.hold_until_ms
    _stop_owner(mesh)
    assert net_fixtures.wait_for(lambda: time.time() * 1000 > grant.grant_expires_at_ms + 100)
    refused = client.request_grant_sync(STUB_PROVIDER, session_id="sess-1")
    assert isinstance(refused, BrokerError) and refused.code == "owner_offline", refused


def test_a_hostile_owner_hold_is_capped_by_the_borrower(
    mesh: Any, monkeypatch: pytest.MonkeyPatch  # noqa: F811
) -> None:
    """T8: a far-future ``hold_until`` ends at receipt + the BORROWER's own cap.

    ``2**52`` ms is a legal peer integer (below ``PEER_NUMBER_CEILING``) about
    140,000 years out; ``10**18`` is over the ceiling and reads as no hold at all
    (``test_a_hold_over_the_peer_ceiling_reads_as_none``).
    """
    monkeypatch.setattr(owner_mod, "_offline_hold_until_ms", lambda *_a: 2**52)
    _write_config(mesh.a.root, grant_ttl_s=2.0, offline_hold_s=21600.0)
    _write_config(mesh.b.root, grant_ttl_s=2.0, offline_hold_s=3.0)
    _share(mesh)
    _pull(mesh)
    client = _client(mesh)
    grant = client.request_grant_sync(STUB_PROVIDER, session_id="sess-1")
    assert isinstance(grant, Grant) and grant.hold_until_ms == 2**52
    _stop_owner(mesh)
    held = client.request_grant_sync(STUB_PROVIDER, session_id="sess-1")
    assert isinstance(held, Grant), held
    assert held.held_until_ms == int(grant.received_wall * 1000 + 3000)
    assert net_fixtures.wait_for(lambda: time.time() * 1000 > held.held_until_ms + 100)
    refused = client.request_grant_sync(STUB_PROVIDER, session_id="sess-1")
    assert isinstance(refused, BrokerError) and refused.code == "owner_offline", refused


def test_a_held_login_the_provider_is_resting_answers_quota_so_failover_moves_on(
    mesh: Any, short_hold: Any  # noqa: F811
) -> None:
    """T9 (#3900): a cooling held login refuses ``quota_blocked``; the slot stays."""
    client, grant = short_hold
    _stop_owner(mesh)
    client.state.cool_down(
        STUB_PROVIDER,
        owner_device=grant.credential_ref.owner_device,
        credential_id=grant.credential_ref.credential_id,
        block_scope="",
        until=time.time() + 600,
    )
    refused = client.request_grant_sync(STUB_PROVIDER, session_id="sess-1", model_id="stub-model")
    assert isinstance(refused, BrokerError) and refused.code == "quota_blocked", refused
    # ``quota_blocked`` with a reset is what ``MeshAwareAuthStore.no_credential_limit``
    # hands the failover driver as ``kind="quota"`` (#3900), so the chain advances.
    assert refused.retry_after_ms > 0, refused
    assert client.grants.peek(STUB_PROVIDER, "sess-1") is not None, "the rest dropped the hold"
    assert not _hold_records(mesh.b.root), "a refused request was counted as a serve"


def test_a_bearer_the_provider_rejected_during_the_hold_is_not_served_again(
    mesh: Any, short_hold: Any  # noqa: F811
) -> None:
    """T4: the 401 report path drops the held slot even though the owner is away."""
    client, grant = short_hold
    _stop_owner(mesh)
    held = client.request_grant_sync(STUB_PROVIDER, session_id="sess-1", model_id="stub-model")
    assert isinstance(held, Grant), held
    client.report_sync(STUB_PROVIDER, kind="unauthorized", session_id="sess-1")
    again = client.request_grant_sync(STUB_PROVIDER, session_id="sess-1", model_id="stub-model")
    assert isinstance(again, BrokerError), "a rejected bearer was re-served from the hold"
    assert client.grants.find_by_token(STUB_PROVIDER, grant.access_token) is None
    # Design review m2: the hold is published as over NOW, so a reader keyed on
    # "the lease has not ended" (#3722's owner-away wait) does not see an hour left.
    access = _state(mesh, client)
    assert access.usable is False, access
    assert access.last_grant_expires_at is not None
    assert access.last_grant_expires_at <= time.time(), access


@pytest.mark.parametrize("path", ["unauthorized", "untrusted", "not-a-holder"])
def test_a_held_grant_dropped_without_a_lapse_writes_its_pending_serves(
    mesh: Any, short_hold: Any, path: str  # noqa: F811
) -> None:
    """R1-m2: a 401, a panic or a revoke drops the slot; its serves are still written.

    The first held serve is written at once and the second only counted, so a drop
    that forgot the tally left one offline serve in memory and in no receipt.
    """
    from local_operator.network import store

    client, grant = short_hold
    _stop_owner(mesh)
    for _ in range(2):
        served = client.request_grant_sync(
            STUB_PROVIDER, session_id="sess-1", model_id="stub-model"
        )
        assert isinstance(served, Grant) and served.held_offline, served
    if path == "unauthorized":
        client.report_sync(STUB_PROVIDER, kind="unauthorized", session_id="sess-1")
    else:
        if path == "untrusted":
            with store.mutate(mesh.network_id, mesh.b.root) as copy:
                copy.trust = "untrusted"
                store.save(copy, mesh.b.root)
        else:
            client.should_borrow = lambda _key: False  # type: ignore[method-assign]
        refused = client.request_grant_sync(
            STUB_PROVIDER, session_id="sess-1", model_id="stub-model"
        )
        assert isinstance(refused, BrokerError), refused
    assert client.grants.peek(STUB_PROVIDER, "sess-1") is None
    rows = [r for r in _hold_records(mesh.b.root) if r["event"] == "credential.held_served"]
    assert sum(r["detail"]["serves"] for r in rows) == 2, rows
    assert {r["detail"]["grant_id"] for r in rows} == {grant.grant_id}
    assert not client._held_receipts, "the tally outlived the dropped grant"


def test_the_settings_row_floor_is_the_grant_ttl_the_reader_floors_at() -> None:
    """R1-m3: a value the reader would silently raise to the grant TTL is refused."""
    from local_operator.network.credentials import GRANT_TTL_S
    from local_operator.settings_io import resolve_key

    row = resolve_key("network.credentials.offline_hold_s")
    assert row is not None
    assert row.minimum == GRANT_TTL_S, row.minimum


@pytest.mark.parametrize("inside_hold", [False, True], ids=["inside-grant", "past-grant"])
def test_a_panic_ends_the_hold_and_keeps_nothing(
    mesh: Any, short_hold: Any, inside_hold: bool  # noqa: F811
) -> None:
    """Design review M1: an untrusted network (panic received or raised) serves nothing.

    A panic only flips ``trust`` and closes the links, so the next dial reads
    ``owner_offline`` — exactly the refusal the hold bridges. The fence in
    ``_hold_admission`` refuses it and drops the slot, without a durable deny.
    """
    from local_operator.network import store

    client, grant = short_hold
    _stop_owner(mesh)
    if inside_hold:
        assert net_fixtures.wait_for(lambda: time.time() * 1000 > grant.grant_expires_at_ms + 100)
    with store.mutate(mesh.network_id, mesh.b.root) as copy:
        copy.trust = "untrusted"
        store.save(copy, mesh.b.root)
    refused = client.request_grant_sync(STUB_PROVIDER, session_id="sess-1", model_id="stub-model")
    assert isinstance(refused, BrokerError), "a panicked node served a held bearer"
    assert client.grants.peek(STUB_PROVIDER, "sess-1") is None, "the slot survived the panic"
    assert _state(mesh, client).usable is False
    assert not _hold_records(mesh.b.root), "a refused request was counted as a serve"


def test_a_revoke_heard_mid_hold_ends_serving(mesh: Any, short_hold: Any) -> None:  # noqa: F811
    """T5: placement losing this holder (owner revoke propagated) ends the hold."""
    client, _grant = short_hold
    _stop_owner(mesh)
    assert isinstance(
        client.request_grant_sync(STUB_PROVIDER, session_id="sess-1"), Grant
    ), "precondition: the hold serves"
    client.should_borrow = lambda _key: False  # type: ignore[method-assign]
    refused = client.request_grant_sync(STUB_PROVIDER, session_id="sess-1")
    assert isinstance(refused, BrokerError) and refused.code == "not_a_holder", refused
    assert client.grants.peek(STUB_PROVIDER, "sess-1") is None


def test_a_forced_refresh_is_never_served_from_the_hold(
    mesh: Any, short_hold: Any  # noqa: F811
) -> None:
    """T12: and it does not end the hold for the next plain request."""
    client, grant = short_hold
    _stop_owner(mesh)
    assert net_fixtures.wait_for(lambda: time.time() * 1000 > grant.grant_expires_at_ms + 100)
    forced = client.request_grant_sync(STUB_PROVIDER, session_id="sess-1", force_refresh=True)
    assert isinstance(forced, BrokerError) and forced.code == "owner_offline", forced
    assert isinstance(client.request_grant_sync(STUB_PROVIDER, session_id="sess-1"), Grant)


def test_receipts_drop_unlisted_fields_and_never_the_token(tmp_path: Any) -> None:
    """T10: the allowlist is the bound; ``access_token``/``token`` are refused by name."""
    audit.write_event(
        tmp_path,
        event="credential.held_served",
        detail={
            "credential_key": "p",
            "act": "d_owner",
            "sub": "d_self",
            "grant_id": "g_1",
            "serves": 3,
            "access_token": "synthetic-bearer-xyz",
            "token": "synthetic-bearer-xyz",
            "account": "someone@example.invalid",
        },
    )
    rows = _hold_records(tmp_path)
    assert rows and rows[0]["detail"] == {
        "credential_key": "p",
        "act": "d_owner",
        "sub": "d_self",
        "grant_id": "g_1",
        "serves": 3,
    }
    assert "synthetic-bearer-xyz" not in audit_path(tmp_path).read_text(encoding="utf-8")


def test_the_owner_returning_flushes_the_pending_count(
    mesh: Any, short_hold: Any, monkeypatch: pytest.MonkeyPatch  # noqa: F811
) -> None:
    """Every offline serve is in some receipt, even when the hold ends by the owner's return."""
    from tests.unit.network.test_credentials_held_offline import _restart_owner

    client, grant = short_hold
    _stop_owner(mesh)
    for _ in range(3):
        assert isinstance(client.request_grant_sync(STUB_PROVIDER, session_id="sess-1"), Grant)
    owner = _restart_owner(mesh, monkeypatch)
    try:
        fresh = client.request_grant_sync(STUB_PROVIDER, session_id="sess-1")
        assert isinstance(fresh, Grant) and not fresh.held_offline, fresh
    finally:
        owner.stop()
    served = [r for r in _hold_records(mesh.b.root) if r["event"] == "credential.held_served"]
    assert sum(r["detail"]["serves"] for r in served) == 3, served
    assert {r["detail"]["grant_id"] for r in served} == {grant.grant_id}


def test_a_hold_over_the_peer_ceiling_reads_as_none() -> None:
    detail = _grant(grant_expires_at_ms=5, hold_until_ms=10**18).to_detail()
    assert Grant.from_detail(detail).hold_until_ms == 0
    detail["hold_until_ms"] = "garbled"
    assert Grant.from_detail(detail).hold_until_ms == 0


def test_readiness_says_held_until_when_not_fresh() -> None:
    """Design review m1: a held grant reads as held, with its end, never as fresh."""
    from types import SimpleNamespace

    from local_operator.network import readiness

    end = 1_794_585_600.0  # 16:00:00 UTC
    row = readiness.selected_model_row(
        SimpleNamespace(device_id="synthetic-peer"),
        {
            "provider": "openai",
            "state": "held_offline",
            "usable": True,
            "owner_label": "device-a",
            "last_grant_expires_at": end,
        },
        selector="openai/gpt",
        session_id="sess-1",
        peer_label="synthetic-peer",
    )
    assert row["ok"] is True and row["code"] == "held_offline", row
    assert "lent before it went offline until 16:00 UTC" in row["detail"], row["detail"]
    assert "verifies the owner again" not in row["detail"]


def test_effective_publishes_the_hold_end_not_the_grant_end(tmp_path: Any) -> None:
    effective._receipts.clear()
    now_ms = int(time.time() * 1000)
    held = dataclasses.replace(
        _grant(grant_expires_at_ms=now_ms - 1000, hold_until_ms=now_ms + 60_000),
        held_offline=True,
        held_until_ms=now_ms + 60_000,
    )
    effective.note_resolution(tmp_path, "p", held)
    receipt = effective._receipts[(str(tmp_path.resolve()), "p")]
    assert receipt.state == "held_offline"
    assert receipt.last_grant_expires_at == pytest.approx((now_ms + 60_000) / 1000)
