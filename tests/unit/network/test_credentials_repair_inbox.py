"""The owner is ASKED to sign in again when a borrowed credential dies (owner-relogin §7).

THE OPERATOR'S ASK, verbatim: "if something expires on a remote device is it able to
refresh from local and can we also do a login for MCP, provider oauth, etc locally and
have that sent to remote to update auth status and refresh tokens so that remote
sessions can use those to refresh on their own while valid (if possible)."

Refresh tokens stay on the owner by construction, and nothing here changes that. What
was missing is the OWNER'S human: a dead borrowed login ended in a doctor row nobody
was looking at (MCP), or in nothing at all (a borrowed bearer's 401 whose owner-side
probe found the grant dead; a provider grant the IdP refused). These cells drive the
repo's two-relay rig (``test_credentials_real_link``'s ``mesh``: two real relays, a
real mutually-authenticated TCP link, a real ``AuthStore``) through the product's own
borrow and report paths, then run the owner's credentials cadence step and read
``notifications.db`` on the owner.

One cell per acceptance row of the design's §7. Rows 2-7 and 9 fail on the parent
commit (no item is ever raised there, and rows 6-7 write no owner record at all).
Row 1 and the row-8 controls are guards: they pass vacuously on the parent, where no
item exists to raise, and fail on an implementation that raises one where it must not.

THE BANNER GATES (``delivery.gates_refuse``) are the item's gates too. The suite arms
the notification kill switch and the rig runs under a redirected HOME, so every cell
opens both explicitly (:func:`gates_open`), and row 8(d) closes the switch alone to
show it is the switch — not the rig guard — that holds the item back. Nothing here
calls a notifier: an item is a row, a banner is another process's job.
"""

from __future__ import annotations

import asyncio
import time
from typing import Any

import pytest

from local_operator.network import relay, store
from local_operator.network.credentials import placement as placement_mod
from local_operator.network.credentials import repair_inbox
from local_operator.network.credentials.messages import render_repair_notice
from local_operator.network.credentials.owner import broker_for_relay
from local_operator.network.credentials.repair import RepairNotice
from local_operator.network.credentials.sync import credentials_sync_step
from local_operator.network.credentials.types import BrokerError, Grant
from local_operator.notifications.store import NotificationStore
from tests.unit.network.test_credentials_owner import (  # noqa: F401 — fixtures by import
    STUB_PROVIDER,
    RotatingIdP,
    idp,
    stub_provider,
)
from tests.unit.network.test_credentials_real_link import (  # noqa: F401 — fixtures by import
    _audit,
    _borrower_client,
    _lop_network,
    _pull,
    _pulled_or_held,
    _share,
    mesh,
)
from tests.unit.network.test_relay_e2e import (  # noqa: F401 — fixtures by import
    _pair,
    devices,
)

MCP_URL = "https://mcp.repair.test/mcp"
MCP_KEY = f"mcp:{MCP_URL}"
PEER_C = "d_" + "c" * 32


@pytest.fixture()
def rig(request: pytest.FixtureRequest) -> Any:
    """The two-relay rig, by name: a parameter called ``mesh`` would shadow the import
    that makes the fixture reachable (the same reason ``test_credentials_real_link``
    reaches its own fixtures through ``getfixturevalue``)."""
    return request.getfixturevalue("mesh")


@pytest.fixture()
def gates_open(monkeypatch: pytest.MonkeyPatch) -> None:
    """Open the two banner gates the item shares: the kill switch and the rig guard."""
    monkeypatch.delenv("LOCAL_OPERATOR_NO_NOTIFICATIONS", raising=False)
    monkeypatch.setattr(
        "local_operator.tui.notify.desktop_belongs_to_this_process", lambda **_: True
    )


@pytest.fixture()
def no_mcp_discovery(monkeypatch: pytest.MonkeyPatch) -> None:
    """The owner's MCP refresh cannot reach an authorization server (none is stubbed).

    The state a server with no metadata — or an unreachable one — leaves behind, and
    the reason an expired MCP token stays expired: the test measures the owner's
    decision, never DNS.
    """

    async def no_endpoints(_url: str) -> None:
        return None

    monkeypatch.setattr("local_operator.mcp.auth.discover_oauth_endpoints", no_endpoints)


def _auth(rig: Any) -> Any:
    from local_operator.providers.auth_store import AuthStore

    return AuthStore(config_dir=rig.a.root)


def _seed_mcp(rig: Any, *, expired: bool, dead: bool = False) -> None:
    """The owner's MCP row for ``MCP_URL``: expired and/or tombstoned, metadata only.

    No token is forged: the values are synthetic, and "dead" is the store's own
    ``grant_dead_at`` verdict, the field an IdP ``invalid_grant`` writes.
    """
    obtained = time.time() - (7200 if expired else 0)
    data: dict[str, Any] = {
        "tokens": {
            "access_token": "-".join(("synthetic", "mcp", "access")),
            "refresh_token": "-".join(("synthetic", "mcp", "refresh")),
            "token_type": "Bearer",
            "expires_in": 3600,
        },
        "tokens_obtained_at": obtained,
        "project_id": MCP_URL,
    }
    auth = _auth(rig)
    try:
        auth.upsert_credential("mcp-oauth", data)
    finally:
        auth.close()
    if dead:
        _tombstone_mcp(rig)


def _tombstone_mcp(rig: Any) -> None:
    """The IdP's dead-grant verdict on the owner's MCP row, through the store's writer.

    ``McpTokenStorage.mark_grant_dead`` is what the refresh path calls on an
    ``invalid_grant``; calling it here is the verdict without a live IdP.
    """
    from local_operator.mcp.auth import McpTokenStorage

    auth = _auth(rig)
    try:
        assert McpTokenStorage(MCP_URL, auth).mark_grant_dead()
        assert McpTokenStorage(MCP_URL, auth).grant_is_dead()
    finally:
        auth.close()


def _tombstone_provider(rig: Any) -> None:
    """The IdP's dead-grant verdict on the owner's provider row, through the store's writer.

    ``_mark_grant_dead`` is the one writer of ``grant_dead_at`` (the refresh path calls
    it on an ``invalid_grant``); no token is touched.
    """
    auth = _auth(rig)
    try:
        rows = auth.list_credentials(STUB_PROVIDER)
        assert rows, "no provider row to tombstone"
        assert auth._mark_grant_dead(rows[0].id, rejected_refresh_token=None)  # noqa: SLF001
        assert auth.grant_is_dead(rows[0].id)
    finally:
        auth.close()


def _share_mcp(rig: Any) -> None:
    assert _lop_network("credential", "share", MCP_KEY, "--with", rig.b.identity.name) == 0
    assert _pulled_or_held(rig, MCP_KEY)


def _borrow(rig: Any, key: str = MCP_KEY, provider: str = "mcp-oauth") -> Any:
    """Borrow through the product's own client, once the second relay leg is up.

    RIG SEQUENCING, NOT A PRODUCT CHANGE: the borrower dials the owner on demand, and
    the ``credential_placement`` pull this file issues just before the first borrow
    dials the same owner concurrently. A borrow issued in that window can lose to the
    pull's own handshake under load and answer ``owner_offline`` — measured here at
    roughly one run in six before this wait, with the same cell passing on the
    previous head. The wait is bounded and always satisfied by a healthy rig; it only
    stops two concurrent dials from racing for the same link.
    """
    client = _borrower_client(rig)
    link, _reason = rig.b._ensure_link_with_reason(rig.owner)  # noqa: SLF001 — the dial seam
    if link is None:
        import time as _time

        deadline = _time.monotonic() + 20.0
        while link is None and _time.monotonic() < deadline:
            _time.sleep(0.25)
            link, _reason = rig.b._ensure_link_with_reason(rig.owner)  # noqa: SLF001
    return client.request_grant_sync(key, provider=provider, session_id="sess-1")


def _owner_tick(rig: Any) -> str:
    """The owner's credentials cadence step for its one member: the real hook."""
    rig.a.audit.flush()
    credentials_sync_step(rig.a, rig.borrower)
    return "ran"


def _items(rig: Any, *, state: str = "all") -> list[dict[str, Any]]:
    rows, _cursor = NotificationStore(rig.a.root).list_items(
        state=state, session_id=repair_inbox.SESSION_ID, limit=100
    )
    return rows


def _doctor_repairs(rig: Any) -> list[dict[str, Any]]:
    return [c for c in rig.a.doctor()["checks"] if c.get("check") == "credential_repair"]


def _report(rig: Any, kind: str, key: str = MCP_KEY) -> dict[str, Any]:
    """B reports a failure it observed, over leg 1 and leg 2 — what ``mcp_bearer`` sends."""
    found = store.find_own_relay(rig.b.root)
    assert found is not None
    reply = relay.control_request(
        found,
        "credential_report",
        timeout=30.0,
        credential_key=key,
        kind=kind,
        session_id="sess-1",
    )
    assert isinstance(reply, dict), reply
    detail = reply.get("detail")
    return detail if isinstance(detail, dict) else reply


def _b_name(rig: Any) -> str:
    return str(rig.b.identity.name)


# ---------------------------------------------------------------------------
# rows 1-5: the MCP grant dead at borrow
# ---------------------------------------------------------------------------


def test_row1_the_borrowers_refusal_is_unchanged(
    rig: Any, gates_open: None, no_mcp_discovery: None
) -> None:
    """Row 1 (guard): the peer still reads ``interactive_required`` and the same words."""
    _seed_mcp(rig, expired=True)
    _share_mcp(rig)

    refused = _borrow(rig)

    assert isinstance(refused, BrokerError), refused
    assert refused.code == "interactive_required"
    # PINNED AS A LITERAL, the bytes the borrower's operator reads on the parent commit
    # (the same cell, run there, prints exactly this): the slice must not move a word
    # of the borrower's copy, and re-deriving the expectation through the renderer
    # would only prove the renderer agrees with itself.
    assert refused.message == (
        "'mcp-oauth' needs an interactive sign-in on device-a before it can be lent out; "
        "the operator there can run '/mcp login <name>'."
    )


def test_row2_the_owner_is_asked_once_and_the_doctor_row_is_untouched(
    rig: Any, gates_open: None, no_mcp_discovery: None
) -> None:
    """Row 2: one unread HIGH item naming B and the command; the doctor row as before."""
    _seed_mcp(rig, expired=True)
    _share_mcp(rig)
    assert isinstance(_borrow(rig), BrokerError)
    assert _items(rig) == [], "an item before the cadence ran: it must not be on the borrow path"

    _owner_tick(rig)

    items = _items(rig)
    assert len(items) == 1, items
    item = items[0]
    assert item["key"] == f"mesh-repair:{rig.network_id}:{MCP_KEY}:g0"
    assert item["read"] is False and item["dismissed"] is False
    assert item["priority"] == "high"
    # D2: the body's quoting in the title, not the CLI's backticks.
    # D3/U5 + D7: an MCP key this device cannot NAME leads with its host, so two
    # such rows stay distinguishable after the list truncates them.
    assert item["title"] == "mcp.repair.test: run '/mcp login <name>' on this device"
    # D5/N2: a list, so two askers do not run together in the pane.
    # D6/U7: the URL is its own item, plain (no backticks: the reel is not Markdown).
    assert item["body_md"] == "\n".join(
        [f"- {render_repair_notice(_b_name(rig), MCP_KEY)}", f"- {MCP_KEY[4:]}"]
    )
    assert item["session_title"] == "/network"
    # No token material: the item is a key name, a device name and a command.
    assert "synthetic" not in item["title"] + item["body_md"]
    repairs = _doctor_repairs(rig)
    assert len(repairs) == 1, repairs
    assert repairs[0]["detail"] == render_repair_notice(_b_name(rig), MCP_KEY)
    # The doctor row is a CLI surface: its remedy keeps the CLI's backticks.
    assert repairs[0]["remedies"] == ["run `/mcp login <name>` on this device"]
    assert repairs[0]["remedies"] == ["run `/mcp login <name>` on this device"]


def test_row3_five_retries_write_nothing(
    rig: Any, gates_open: None, no_mcp_discovery: None, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Row 3: five more refused borrows — same item, same revision, no store write."""
    _seed_mcp(rig, expired=True)
    _share_mcp(rig)
    assert isinstance(_borrow(rig), BrokerError)
    _owner_tick(rig)
    (before,) = _items(rig)

    notifies: list[str] = []
    original = NotificationStore.notify

    def counting(self: NotificationStore, *args: Any, **kwargs: Any) -> Any:
        outcome = original(self, *args, **kwargs)
        notifies.append(outcome.action)
        return outcome

    monkeypatch.setattr(NotificationStore, "notify", counting)
    reports_before = len(_audit(rig.a, "credential.report"))
    for _ in range(5):
        assert isinstance(_borrow(rig), BrokerError)
        _owner_tick(rig)

    # The retries were REAL: each one wrote its own report on the owner.
    assert len(_audit(rig.a, "credential.report")) == reports_before + 5
    (after,) = _items(rig)
    assert after["id"] == before["id"]
    assert after["revision"] == before["revision"] == 1
    assert after["seq"] == before["seq"]
    # The memo answered every pass: the store was not even asked.
    assert notifies == [], notifies


def test_row4_a_second_peer_updates_the_same_item(
    rig: Any, gates_open: None, no_mcp_discovery: None
) -> None:
    """Row 4: C asks too — the SAME row, revision + 1, both named, and no second banner.

    The rig has two relays, so C's borrow is the owner's own ``grant`` handler fed a
    link authenticated as C (a member and a holder on A) — the handler a third relay's
    leg-2 frame would reach.
    """
    _seed_mcp(rig, expired=True)
    _share_mcp(rig)
    assert isinstance(_borrow(rig), BrokerError)
    _owner_tick(rig)
    (first,) = _items(rig)

    from local_operator.network import types, wire

    with store.mutate(rig.network_id, rig.a.root) as record:
        record.members.append(
            types.MemberRecord(
                device_id=PEER_C,
                public_key=wire.b64u(b"c" * 32),
                role="drive",
                capabilities=sorted(types.capabilities_for_role("drive")),
                added_via="invite",
                name="device-c",
            )
        )
        store.save(record, rig.a.root)
    document = placement_mod.PlacementDocument.load(
        rig.network_id, rig.a.root, self_device=rig.owner
    )
    document.grant(MCP_KEY, PEER_C, scope="session", by=rig.owner)
    document.save()
    broker = broker_for_relay(rig.a)
    assert broker is not None

    class _LinkC:
        device_id = PEER_C
        network_id = rig.network_id
        epoch = 1

    reply = broker.grant(
        _LinkC(),
        {"key": MCP_KEY, "provider": "mcp-oauth", "for_session": "sess-c", "from_device": PEER_C},
    )
    assert reply.get("code") == "interactive_required", reply
    _owner_tick(rig)

    (second,) = _items(rig)
    assert second["id"] == first["id"]
    assert second["revision"] == first["revision"] + 1
    assert second["body_md"] == "\n".join(
        [
            f"- {render_repair_notice(_b_name(rig), MCP_KEY)}",
            f"- {render_repair_notice('device-c', MCP_KEY)}",
            f"- {MCP_KEY[4:]}",
        ]
    )
    # NOT ``high``: the ladder banners a high revision, and the operator was already
    # told about this dead login once.
    assert second["priority"] == "normal"
    from local_operator.notifications.delivery import RUNG_PRIORITY, route_item

    assert route_item(rig.a.root, action="updated", priority=second["priority"]).rung == (
        RUNG_PRIORITY
    )


def test_row5_the_repair_closes_the_item_and_the_doctor_row(
    rig: Any, gates_open: None, no_mcp_discovery: None
) -> None:
    """Row 5: a fresh sign-in on A, B's next borrow succeeds, the item auto-dismisses."""
    _seed_mcp(rig, expired=True)
    _share_mcp(rig)
    assert isinstance(_borrow(rig), BrokerError)
    _owner_tick(rig)
    (raised,) = _items(rig)

    # THE REPAIR: what ``/mcp login``'s callback leaves behind — a fresh grant written
    # through the token storage the owner's resolve reads.
    from mcp.shared.auth import OAuthToken

    from local_operator.mcp.auth import McpTokenStorage

    auth = _auth(rig)
    try:
        asyncio.run(
            McpTokenStorage(MCP_URL, auth).set_tokens(
                OAuthToken.model_validate(
                    {
                        "access_token": "-".join(("synthetic", "fresh", "access")),
                        "refresh_token": "-".join(("synthetic", "fresh", "refresh")),
                        "expires_in": 3600,
                    }
                )
            )
        )
    finally:
        auth.close()
    granted = _borrow(rig)
    assert isinstance(granted, Grant), granted
    assert _audit(rig.a, "credential.grant"), "the successful borrow wrote no grant"

    _owner_tick(rig)

    assert _items(rig) == [], "the repaired item is still in the reel"
    (closed,) = _items(rig, state="dismissed")
    assert closed["id"] == raised["id"]
    assert closed["dismissed_via"] == "mesh_repair"
    assert _doctor_repairs(rig) == []
    from local_operator.network.credentials.repair import open_reports

    assert open_reports(rig.a.audit.tail(500)) == []


# ---------------------------------------------------------------------------
# rows 6-7: the two classes that wrote no owner record at all
# ---------------------------------------------------------------------------


def test_row6_a_401_whose_owner_probe_finds_the_grant_dead_asks_the_owner(
    rig: Any, gates_open: None, no_mcp_discovery: None
) -> None:
    """Row 6: B's bearer 401s, A's probe finds the grant dead → one item, ``reauth_required``."""
    _seed_mcp(rig, expired=False)
    _share_mcp(rig)
    assert isinstance(_borrow(rig), Grant)
    # The IdP refuses the owner's grant from here on (its verdict, persisted), and the
    # MCP server refuses the bearer B holds: B reports what it saw.
    _tombstone_mcp(rig)

    ack = _report(rig, "unauthorized")

    # The owner's probe found the grant dead, and the borrower is told so with the
    # owner-side login, instead of a ``refreshed`` that nothing backed.
    assert ack.get("action") == "refresh_failed", ack
    assert ack.get("code") == "grant_invalid", ack
    assert ack.get("remedy", "").startswith("/mcp login"), ack
    reports = _audit(rig.a, "credential.report")
    assert [r["detail"]["failure"] for r in reports[-2:]] == ["unauthorized", "reauth_required"]
    assert reports[-1]["detail"]["sub"] == rig.borrower
    _owner_tick(rig)
    (item,) = _items(rig)
    assert item["body_md"].startswith(f"- {render_repair_notice(_b_name(rig), MCP_KEY)}")
    assert item["priority"] == "high"


def test_row7_a_provider_grant_the_idp_refused_asks_the_owner(rig: Any, gates_open: None) -> None:
    """Row 7: a dead provider login → B is refused → one item on A saying ``/login``.

    THE REFUSAL IS ``no_local_credential``, NOT ``grant_invalid`` — measured here, and
    the design's §2(c) was wrong about it: the owner's cascade catches a tombstoned
    row's ``CredentialInvalidError`` per row and answers "nothing usable", so the
    borrower's code (and its sentence) stay what they were. The owner's record is
    what is new; ``_classify_refresh_failure``'s own ``grant_invalid`` arm is pinned
    separately below, since no store path reaches it today.
    """
    _share(rig)
    _pull(rig)
    _tombstone_provider(rig)

    refused = _borrow(rig, STUB_PROVIDER, STUB_PROVIDER)

    assert isinstance(refused, BrokerError), refused
    assert refused.code == "no_local_credential"
    reports = _audit(rig.a, "credential.report")
    assert reports and reports[-1]["detail"]["failure"] == "reauth_required"
    assert reports[-1]["detail"]["sub"] == rig.borrower
    _owner_tick(rig)
    (item,) = _items(rig)
    assert item["title"] == f"Run '/login {STUB_PROVIDER}' on this device"
    assert item["body_md"] == f"- {render_repair_notice(_b_name(rig), STUB_PROVIDER)}"
    assert rig.idp.posts == [], "a dead grant cost a token POST"


# ---------------------------------------------------------------------------
# row 8: negative controls
# ---------------------------------------------------------------------------


def test_row8a_a_transient_refresh_failure_raises_nothing(
    rig: Any, gates_open: None, no_mcp_discovery: None, monkeypatch: pytest.MonkeyPatch
) -> None:
    """8(a): ``refresh_failed`` — a refresh that could not run just now — is not a re-login.

    The owner's MCP refresh raises the one class ``ensure_mcp_oauth_fresh`` documents
    as raisable (lock acquisition, ``OSError``), which ``_classify_refresh_failure``
    answers ``refresh_failed``: the transient arm, beside the ``grant_invalid`` arm
    that row 7 drives.
    """
    _seed_mcp(rig, expired=False)
    _share_mcp(rig)

    async def lock_busy(*_: Any, **__: Any) -> None:
        raise OSError("refresh lock busy")

    monkeypatch.setattr("local_operator.mcp.auth.ensure_mcp_oauth_fresh", lock_busy)

    refused = _borrow(rig)

    assert isinstance(refused, BrokerError) and refused.code == "refresh_failed", refused
    _owner_tick(rig)
    assert _items(rig) == []
    assert not [
        r for r in _audit(rig.a, "credential.report") if r["detail"].get("failure")
    ], "a transient failure wrote a repair report"


def test_row8b_an_offline_owner_raises_nothing_on_the_owner(rig: Any, gates_open: None) -> None:
    """8(b): ``owner_offline`` is the borrower's own verdict; the owner records nothing."""
    _share(rig)
    _pull(rig)
    rig.a.stop()

    refused = _borrow(rig, STUB_PROVIDER, STUB_PROVIDER)

    assert isinstance(refused, BrokerError) and refused.code == "owner_offline", refused
    repair_inbox.tick(rig.a)
    assert _items(rig) == []


def test_row8c_a_key_this_device_does_not_own_raises_nothing(rig: Any, gates_open: None) -> None:
    """8(c): a repair report for a key owned elsewhere: doctor row yes, item no."""
    from local_operator.network.audit import AuditEvent

    foreign = "elsewhere-provider"
    rig.a.audit.record(
        AuditEvent(
            event="credential.report",
            actor=rig.owner,
            network_id=rig.network_id,
            subject=rig.borrower,
            detail={
                "credential_key": foreign,
                "act": rig.owner,
                "sub": rig.borrower,
                "failure": "reauth_required",
            },
        )
    )
    _owner_tick(rig)

    assert _items(rig) == []
    assert [c["credential_name"] for c in _doctor_repairs(rig)] == [foreign]


def test_row8c_a_peer_cannot_claim_the_owners_verdict(rig: Any, gates_open: None) -> None:
    """8(c'): a holder REPORTING ``reauth_required`` itself opens nothing anywhere.

    The opening words are the owner's verdicts; recorded verbatim from a peer frame,
    they would let any holder ask this operator to sign in again at will.
    """
    _share(rig)
    _pull(rig)

    ack = _report(rig, "reauth_required", key=STUB_PROVIDER)

    assert ack.get("action") == "noted", ack
    last = _audit(rig.a, "credential.report")[-1]
    assert last["detail"]["failure"] == "peer_claimed:reauth_required"
    _owner_tick(rig)
    assert _items(rig) == []
    assert _doctor_repairs(rig) == []


def test_row8d_notifications_off_raises_no_item_and_keeps_the_doctor_row(
    rig: Any, gates_open: None, no_mcp_discovery: None, monkeypatch: pytest.MonkeyPatch
) -> None:
    """8(d): the switch off → no item, doctor row intact; switched back on → raised."""
    _seed_mcp(rig, expired=True)
    _share_mcp(rig)
    assert isinstance(_borrow(rig), BrokerError)
    monkeypatch.setenv("LOCAL_OPERATOR_NO_NOTIFICATIONS", "1")

    _owner_tick(rig)

    assert _items(rig) == []
    assert len(_doctor_repairs(rig)) == 1
    # While the switch stays off the pass is OWED but derives nothing (m1/Q1's fix):
    # one stat, a gate check, no derivation, no store.
    before = dict(repair_inbox.COUNTERS)
    assert repair_inbox.tick(rig.a) == "gated"
    assert repair_inbox.COUNTERS["derivations"] == before["derivations"]
    # The positive control: the SAME dead login, the switch back on, and the peer's
    # next attempt (it asks again within its refusal window) raises the item.
    monkeypatch.delenv("LOCAL_OPERATOR_NO_NOTIFICATIONS")
    assert isinstance(_borrow(rig), BrokerError)
    _owner_tick(rig)
    assert len(_items(rig)) == 1


# ---------------------------------------------------------------------------
# row 9: bounded idle work
# ---------------------------------------------------------------------------


def test_row9_idle_ticks_cost_a_stat_each_and_no_sql(
    rig: Any, gates_open: None, no_mcp_discovery: None, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Row 9: untouched audit → one stat per call, no derivation, no store at all.

    Three members' worth of calls per tick (the cadence calls the step once per
    member), and an unrelated audit row in between: it rings the doorbell but its
    bytes name neither folded event, so it costs a scan and no derivation.
    """
    _seed_mcp(rig, expired=True)
    _share_mcp(rig)
    assert isinstance(_borrow(rig), BrokerError)
    _owner_tick(rig)
    assert len(_items(rig)) == 1

    opened: list[str] = []
    original_init = NotificationStore.__init__

    def counting_init(self: NotificationStore, *args: Any, **kwargs: Any) -> None:
        opened.append("store")
        original_init(self, *args, **kwargs)

    monkeypatch.setattr(NotificationStore, "__init__", counting_init)
    rig.a.audit.flush()
    before = dict(repair_inbox.COUNTERS)
    for _tick in range(5):
        for _member in range(3):
            assert repair_inbox.tick(rig.a) == "idle"
    after = dict(repair_inbox.COUNTERS)
    assert after["stats"] - before["stats"] == 15
    assert after["derivations"] == before["derivations"]
    assert opened == []

    from local_operator.network.audit import AuditEvent

    rig.a.audit.record(AuditEvent(event="peer_connected", actor=rig.borrower))
    rig.a.audit.flush()
    assert repair_inbox.tick(rig.a) == "idle"
    final = dict(repair_inbox.COUNTERS)
    assert final["scans"] - after["scans"] == 1
    assert final["derivations"] == before["derivations"]
    assert opened == []


def test_row7_the_grant_invalid_arm_records_the_repair_and_refresh_failed_does_not(
    rig: Any,
) -> None:
    """Row 7, the classifier's own arm: ``grant_invalid`` writes the row, ``refresh_failed`` not.

    Driven on the owner's real broker because no store path raises
    ``CredentialInvalidError`` through ``get_oauth_access`` today (see the cell above);
    the arm is the briefed write site and must hold the day one does.
    """
    from local_operator.providers.auth_store import (
        AuthStoreError,
        CredentialInvalidError,
    )

    _share(rig)  # the relay builds its broker once it has something to lend
    _pull(rig)
    broker = broker_for_relay(rig.a)
    assert broker is not None
    before = len(_audit(rig.a, "credential.report"))

    dead = broker._classify_refresh_failure(  # noqa: SLF001 — the arm itself
        CredentialInvalidError("refused by the IdP"),
        STUB_PROVIDER,
        by=rig.borrower,
        for_session="sess-9",
    )
    transient = broker._classify_refresh_failure(  # noqa: SLF001
        AuthStoreError("token endpoint unreachable"), STUB_PROVIDER, by=rig.borrower
    )

    assert dead.code == "grant_invalid" and transient.code == "refresh_failed"
    rows = _audit(rig.a, "credential.report")[before:]
    assert [r["detail"]["failure"] for r in rows] == ["reauth_required"]
    assert rows[0]["detail"]["sub"] == rig.borrower and rows[0]["session_id"] == "sess-9"


def test_row7_a_provider_never_signed_in_writes_no_repair(rig: Any, gates_open: None) -> None:
    """Control for the new arm: "nothing signed in" is not "signed in, but dead"."""
    _share(rig)
    _pull(rig)
    auth = _auth(rig)
    try:
        for row in auth.list_credentials(STUB_PROVIDER):
            auth.delete_credential(row.id)
    finally:
        auth.close()

    refused = _borrow(rig, STUB_PROVIDER, STUB_PROVIDER)

    assert isinstance(refused, BrokerError) and refused.code == "no_local_credential", refused
    assert not [
        r for r in _audit(rig.a, "credential.report") if r["detail"].get("failure")
    ], "an empty provider wrote a repair report"
    _owner_tick(rig)
    assert _items(rig) == []


def test_row9_five_minutes_of_the_real_cadence_idle(
    rig: Any,
    gates_open: None,
    no_mcp_discovery: None,
    monkeypatch: pytest.MonkeyPatch,
    capsys: Any,
) -> None:
    """Row 9 on the cadence itself: 20 ticks (5 min at 15 s) of a rig with an item raised.

    The definitions syncer is driven with an injected clock, as its own tests drive it,
    so every independent step runs exactly as the thread would run it — including the
    credentials step's placement pull over the real link, which writes audit rows of its
    own. The measurement the design asked for: stats, derivations, audit growth and
    notification-store writes while nothing about any repair changes.
    """
    from local_operator.network import definitions

    _seed_mcp(rig, expired=True)
    _share_mcp(rig)
    assert isinstance(_borrow(rig), BrokerError)
    _owner_tick(rig)
    assert len(_items(rig)) == 1

    writes: list[str] = []
    original_write = NotificationStore._write  # noqa: SLF001 — the store's one write seam

    def counting_write(self: NotificationStore, body: Any, what: str) -> Any:
        writes.append(what)
        return original_write(self, body, what)

    monkeypatch.setattr(NotificationStore, "_write", counting_write)
    # THE RELAY'S OWN SYNCER THREAD IS STOPPED for the measurement: it ticks on its
    # own clock (15 s, or a rig override), so leaving it running would let the test's
    # stat count move for reasons this cell is not measuring.
    definitions.stop_syncer(rig.a)
    syncer = definitions.DefinitionsSyncer(rig.a)
    rig.a.audit.flush()
    audit_path = rig.a.root / "network" / "audit.jsonl"
    size_before = audit_path.stat().st_size
    before = dict(repair_inbox.COUNTERS)
    start = time.time()
    for tick in range(20):
        syncer.tick(now=start + 15.0 * tick)
        rig.a.audit.flush()
    after = dict(repair_inbox.COUNTERS)
    grown = audit_path.stat().st_size - size_before
    delta = {name: after[name] - before[name] for name in after}
    with capsys.disabled():
        print(
            f"\nrow9 idle cadence: ticks=20 counters={delta} audit_bytes_grown={grown} "
            f"notification_writes={len(writes)}"
        )
    assert delta["stats"] == 20  # one member: one stat per tick
    assert delta["derivations"] == 0
    assert writes == []


def test_row9_grant_rows_cost_no_derivation_while_nothing_is_open(
    rig: Any, gates_open: None
) -> None:
    """Row 9 on a BUSY owner: every successful borrow appends ``credential.grant``.

    A grant can only close an item, so while none is open those rows ring the
    doorbell and cost a scan of the appended bytes, never a derivation (a full read
    of the live audit file: 75-110 ms of CPU on an 8.6 MB file, measured).
    """
    _share(rig)
    _pull(rig)
    assert isinstance(_borrow(rig, STUB_PROVIDER, STUB_PROVIDER), Grant)
    _owner_tick(rig)  # the first pass derives once, from scratch
    before = dict(repair_inbox.COUNTERS)

    for _ in range(3):
        assert isinstance(_borrow(rig, STUB_PROVIDER, STUB_PROVIDER), Grant)
    rig.a.audit.flush()
    assert repair_inbox.tick(rig.a) == "idle"

    after = dict(repair_inbox.COUNTERS)
    assert len(_audit(rig.a, "credential.grant")) >= 4
    assert after["scans"] - before["scans"] == 1
    assert after["derivations"] == before["derivations"]
    assert _items(rig) == []


def _planned(generation: str, body: str = "laptop needs it") -> dict[Any, Any]:
    return {
        ("n_x", "openai"): repair_inbox.PlannedItem(
            network_id="n_x",
            key="openai",
            generation=generation,
            title="Run `/login openai` on this device",
            body=body,
        )
    }


def test_an_aged_out_grant_never_reads_as_a_new_episode(tmp_path: Any) -> None:
    """The design's ``g<seq>`` token, corrected: it may only move FORWARD while open.

    The token is the seq of the grant before the episode. When that grant ages out of
    the 500-row window mid-episode the derivation reads ``g0``; taken at face value
    that is a second item (and a second banner) for one dead login, with the first
    stranded. A lower token is aging; a higher one is a real repair-then-break.
    """
    episodes: dict[Any, Any] = {}
    first = repair_inbox.publish_notices(tmp_path, _planned("g7"), [], episodes=episodes)
    assert first.created == 1
    store = NotificationStore(tmp_path)
    (item,) = store.list_items(state="all", session_id=repair_inbox.SESSION_ID)[0]

    # The grant aged out: g0 now. Same item, nothing new — with or without the memo
    # (a restarted relay finds the standing row instead).
    for memo in (episodes, {}):
        aged = repair_inbox.publish_notices(tmp_path, _planned("g0"), [], episodes=memo)
        assert aged.created == 0, aged
        rows = store.list_items(state="all", session_id=repair_inbox.SESSION_ID)[0]
        assert [r["key"] for r in rows] == [item["key"]]

    # Repaired (grant seq 9), then broken again: the old row is closed as repaired
    # and the new episode is a new item.
    again = repair_inbox.publish_notices(tmp_path, _planned("g9"), [], episodes=episodes)
    assert (again.created, again.closed) == (1, 1), again
    (live,) = store.list_items(state="all", session_id=repair_inbox.SESSION_ID)[0]
    assert live["key"].endswith(":g9")
    (closed,) = store.list_items(state="dismissed", session_id=repair_inbox.SESSION_ID)[0]
    assert closed["id"] == item["id"] and closed["dismissed_via"] == "mesh_repair"


def test_an_operator_dismissal_holds_for_the_episode(tmp_path: Any) -> None:
    """Dismissed by the operator: retries are absorbed, and a new asker is silent too."""
    episodes: dict[Any, Any] = {}
    repair_inbox.publish_notices(tmp_path, _planned("g0"), [], episodes=episodes)
    store = NotificationStore(tmp_path)
    (item,) = store.list_items(state="all", session_id=repair_inbox.SESSION_ID)[0]
    store.apply("dismiss", ids=[(item["id"], int(item["revision"]))], via="explicit")

    repair_inbox.publish_notices(tmp_path, _planned("g0", "two asked"), [], episodes={})

    assert store.list_items(state="all", session_id=repair_inbox.SESSION_ID)[0] == []
    (kept,) = store.list_items(state="dismissed", session_id=repair_inbox.SESSION_ID)[0]
    assert kept["dismissed_via"] == "explicit"


# ---------------------------------------------------------------------------
# round 1 remediation: M1, m1/Q1, m2, m3, U1
# ---------------------------------------------------------------------------


def test_M1_a_dismissed_episode_does_not_come_back_after_a_restart(tmp_path: Any) -> None:
    """M1: the generation floor counts DISMISSED rows, so a restart cannot re-banner.

    The reviewer's probe: raise at ``g7``, the operator dismisses it, the relay
    restarts with the episode's grant aged out of the window (the token reads
    ``g0``), and a fresh ``publish`` with an empty episode memo runs.
    """
    episodes: dict[Any, Any] = {}
    repair_inbox.publish_notices(tmp_path, _planned("g7"), [], episodes=episodes)
    store = NotificationStore(tmp_path)
    (item,) = store.list_items(state="all", session_id=repair_inbox.SESSION_ID)[0]
    store.apply("dismiss", ids=[(item["id"], int(item["revision"]))], via="explicit")

    # A RESTART: no in-process memo, the same content, the grant row gone from view.
    after = repair_inbox.publish_notices(tmp_path, _planned("g0"), [], episodes={})

    assert (after.created, after.updated, after.reopened) == (0, 0, 0), after
    assert store.list_items(state="all", session_id=repair_inbox.SESSION_ID)[0] == []
    (kept,) = store.list_items(state="dismissed", session_id=repair_inbox.SESSION_ID)[0]
    assert kept["dismissed_via"] == "explicit", "the operator's dismissal was overwritten"


def test_M1_a_new_asker_on_a_dismissed_episode_is_absorbed_silently(tmp_path: Any) -> None:
    """M1's other half: content changes land on the DISMISSED row, which absorbs them."""
    episodes: dict[Any, Any] = {}
    repair_inbox.publish_notices(tmp_path, _planned("g0"), [], episodes=episodes)
    store = NotificationStore(tmp_path)
    (item,) = store.list_items(state="all", session_id=repair_inbox.SESSION_ID)[0]
    store.apply("dismiss", ids=[(item["id"], int(item["revision"]))], via="explicit")

    result = repair_inbox.publish_notices(tmp_path, _planned("g0", "- two asked"), [], episodes={})

    assert result.created == 0, result
    assert store.list_items(state="all", session_id=repair_inbox.SESSION_ID)[0] == []


def test_m1_q1_a_repair_made_while_notifications_were_off_still_closes_the_item(
    rig: Any, gates_open: None, no_mcp_discovery: None, monkeypatch: pytest.MonkeyPatch
) -> None:
    """m1 = Q1: the gated pass is OWED, so the switch coming back on closes it.

    QA's repro: item open → switch off → the owner signs in and the peer's borrow
    succeeds → gated tick → switch on → the item must close WITHOUT another grant.
    """
    from mcp.shared.auth import OAuthToken

    from local_operator.mcp.auth import McpTokenStorage

    _seed_mcp(rig, expired=True)
    _share_mcp(rig)
    assert isinstance(_borrow(rig), BrokerError)
    _owner_tick(rig)
    assert len(_items(rig)) == 1

    monkeypatch.setenv("LOCAL_OPERATOR_NO_NOTIFICATIONS", "1")
    auth = _auth(rig)
    try:
        asyncio.run(
            McpTokenStorage(MCP_URL, auth).set_tokens(
                OAuthToken.model_validate(
                    {
                        "access_token": "-".join(("synthetic", "fresh", "access")),
                        "refresh_token": "-".join(("synthetic", "fresh", "refresh")),
                        "expires_in": 3600,
                    }
                )
            )
        )
    finally:
        auth.close()
    assert isinstance(_borrow(rig), Grant)
    assert repair_inbox.tick(rig.a) == "gated"

    monkeypatch.delenv("LOCAL_OPERATOR_NO_NOTIFICATIONS")
    _owner_tick(rig)  # NO further borrow: the owed pass alone must close it

    assert _items(rig) == []
    (closed,) = _items(rig, state="dismissed")
    assert closed["dismissed_via"] == "mesh_repair"


def test_U1_the_owners_own_sign_in_closes_the_item_and_the_doctor_row(
    rig: Any, gates_open: None, no_mcp_discovery: None
) -> None:
    """U1: signing in here is the repair the item asks for, so it must be visible.

    No peer borrow after the sign-in at all — the operator's own ``/mcp login``
    writes the auth store and no audit row, so the tick's login doorbell is what
    sees it.
    """
    from mcp.shared.auth import OAuthToken

    from local_operator.mcp.auth import McpTokenStorage

    _seed_mcp(rig, expired=True)
    _share_mcp(rig)
    assert isinstance(_borrow(rig), BrokerError)
    _owner_tick(rig)
    assert len(_items(rig)) == 1 and len(_doctor_repairs(rig)) == 1

    auth = _auth(rig)
    try:
        asyncio.run(
            McpTokenStorage(MCP_URL, auth).set_tokens(
                OAuthToken.model_validate(
                    {
                        "access_token": "-".join(("synthetic", "signin", "access")),
                        "refresh_token": "-".join(("synthetic", "signin", "refresh")),
                        "expires_in": 3600,
                    }
                )
            )
        )
    finally:
        auth.close()

    _owner_tick(rig)

    assert _items(rig) == [], "the sign-in did not clear the item"
    assert _doctor_repairs(rig) == [], "the sign-in did not clear the doctor row"
    (closed,) = _items(rig, state="dismissed")
    assert closed["dismissed_via"] == "mesh_repair"


def test_U1_a_provider_sign_in_closes_it_too(rig: Any, gates_open: None) -> None:
    """U1, provider class: ``/login meshtest`` rewrites the row and clears the tombstone."""
    _share(rig)
    _pull(rig)
    _tombstone_provider(rig)
    assert isinstance(_borrow(rig, STUB_PROVIDER, STUB_PROVIDER), BrokerError)
    _owner_tick(rig)
    assert len(_items(rig)) == 1

    # What /login persists: the same provider row, rewritten, tombstone cleared.
    auth = _auth(rig)
    try:
        auth.upsert_credential(
            STUB_PROVIDER,
            {
                "type": "oauth",
                "access": "-".join(("synthetic", "fresh", "access")),
                "expires": int(time.time() * 1000) + 3_600_000,
                "refresh": "-".join(("synthetic", "fresh", "refresh")),
                "email": "owner@example.test",
            },
        )
        row = auth.list_credentials(STUB_PROVIDER)[0]
        assert not auth.grant_is_dead(row.id)
    finally:
        auth.close()

    _owner_tick(rig)

    assert _items(rig) == []
    assert _doctor_repairs(rig) == []


def test_m3_a_healthy_sibling_grant_does_not_close_a_provider_repair(rig: Any) -> None:
    """m3: on a multi-login provider, the DEAD row keeps the repair open.

    The reviewer's scenario: the dead row sits beside a live one, and a borrow
    served from the live row writes a ``credential.grant`` for the same provider
    key. Per-key closure would read that as repaired; the dead row still needs the
    sign-in. Once that row is signed in again, the repair closes.
    """
    from local_operator.network import store as net_store
    from local_operator.network.audit import AuditEvent
    from local_operator.network.credentials.repair import repair_notices

    _share(rig)
    _pull(rig)
    auth = _auth(rig)
    try:
        auth.upsert_credential(
            STUB_PROVIDER,
            {
                "type": "oauth",
                "access": "-".join(("synthetic", "second", "access")),
                "expires": int(time.time() * 1000) + 3_600_000,
                "refresh": "-".join(("synthetic", "second", "refresh")),
                "email": "second@example.test",
            },
        )
        rows = auth.list_credentials(STUB_PROVIDER)
        assert len(rows) >= 2, [r.id for r in rows]
        assert auth._mark_grant_dead(rows[0].id, rejected_refresh_token=None)  # noqa: SLF001
    finally:
        auth.close()
    rig.a.audit.record(
        AuditEvent(
            event="credential.report",
            actor=rig.owner,
            network_id=rig.network_id,
            subject=rig.borrower,
            detail={
                "credential_key": STUB_PROVIDER,
                "act": rig.owner,
                "sub": rig.borrower,
                "failure": "reauth_required",
            },
        )
    )
    # A grant for the same key, served from the HEALTHY row.
    rig.a.audit.record(
        AuditEvent(
            event="credential.grant",
            actor=rig.owner,
            network_id=rig.network_id,
            subject=rig.borrower,
            detail={"credential_key": STUB_PROVIDER, "act": rig.owner, "sub": rig.borrower},
        )
    )
    rig.a.audit.flush()
    records = rig.a.audit.tail(500)
    record = next(r for r in net_store.list_networks(rig.a.root) if r.network_id == rig.network_id)

    assert [n.key for n in repair_notices(record, records)] == [
        STUB_PROVIDER
    ], "a sibling's grant closed a repair whose own row is still dead"

    # The remedy: that row signed in again (a fresh row, no tombstone anywhere).
    auth = _auth(rig)
    try:
        for row in auth.list_credentials(STUB_PROVIDER):
            data = dict(row.data)
            data.pop("grant_dead_at", None)
            data["access"] = "-".join(("synthetic", "renewed", str(row.id)))
            auth.upsert_credential(STUB_PROVIDER, data)
        assert not any(auth.grant_is_dead(row.id) for row in auth.list_credentials(STUB_PROVIDER))
    finally:
        auth.close()

    assert repair_notices(record, records) == []


def test_m2_a_failing_store_says_so_once_and_stops_the_re_derivation(
    rig: Any,
    gates_open: None,
    no_mcp_discovery: None,
    monkeypatch: pytest.MonkeyPatch,
    caplog: Any,
) -> None:
    """m2: a store that keeps failing is reported ONCE and not re-derived every tick.

    Before the fix the tick logged at debug (so a regression left no trace) while
    ``publish_notices`` logged a traceback per pass and the doorbell re-armed to
    ``full`` — one full derivation and one traceback per member per tick.
    """
    import logging

    _seed_mcp(rig, expired=True)
    _share_mcp(rig)
    assert isinstance(_borrow(rig), BrokerError)

    original_notify = NotificationStore.notify

    def refusing(self: Any, *args: Any, **kwargs: Any) -> Any:
        raise RuntimeError("the store is unreadable")

    monkeypatch.setattr(NotificationStore, "notify", refusing)
    monkeypatch.setattr(repair_inbox, "BACKOFF_BASE_S", 60.0)
    caplog.set_level(logging.DEBUG, logger="local_operator.network.credentials.repair_inbox")

    assert repair_inbox.tick(rig.a) == "failed"
    warnings = [r for r in caplog.records if r.levelno >= logging.WARNING]
    assert len(warnings) == 1, [r.getMessage() for r in warnings]
    assert "the store is unreadable" in warnings[0].getMessage()

    # Backed off: five more member calls derive nothing and say nothing new.
    before = dict(repair_inbox.COUNTERS)
    for _ in range(5):
        assert repair_inbox.tick(rig.a) == "backoff"
    after = dict(repair_inbox.COUNTERS)
    assert after["derivations"] == before["derivations"]
    assert [r for r in caplog.records if r.levelno >= logging.WARNING] == warnings

    # The store comes back: the next pass after the backoff publishes and resets.
    # (Not ``monkeypatch.undo()``: pytest shares ONE instance per test with the
    # fixtures, so undoing here would also close ``gates_open``'s gates.)
    monkeypatch.setattr(NotificationStore, "notify", original_notify)
    bell = repair_inbox._doorbell_for(rig.a.root / "network" / "audit.jsonl")  # noqa: SLF001
    bell.retry_at = 0.0
    assert repair_inbox.tick(rig.a) == "published"
    assert len(_items(rig)) == 1


# ---------------------------------------------------------------------------
# round 2 remediation: review m1, design D6/D7, UX U7
# ---------------------------------------------------------------------------


def test_review_m1_a_break_after_a_repair_raises_even_if_the_operator_dismissed(
    rig: Any, gates_open: None, no_mcp_discovery: None
) -> None:
    """review m1: dismiss → sign in (the repair) → the same login breaks again.

    No peer borrows in between, so nothing advances the generation. The dismissed
    row must still give way to a NEW high item, because the operator's dismissal was
    answered by a real repair, not by the login staying dead.
    """
    _seed_mcp(rig, expired=True)
    _share_mcp(rig)
    assert isinstance(_borrow(rig), BrokerError)
    _owner_tick(rig)
    (first,) = _items(rig)
    NotificationStore(rig.a.root).apply(
        "dismiss", ids=[(first["id"], int(first["revision"]))], via="explicit"
    )

    # The REPAIR: a fresh sign-in through the token storage `/mcp login` uses.
    from mcp.shared.auth import OAuthToken

    from local_operator.mcp.auth import McpTokenStorage

    auth = _auth(rig)
    try:
        asyncio.run(
            McpTokenStorage(MCP_URL, auth).set_tokens(
                OAuthToken.model_validate(
                    {
                        "access_token": "-".join(("synthetic", "repaired", "access")),
                        "refresh_token": "-".join(("synthetic", "repaired", "refresh")),
                        "expires_in": 3600,
                    }
                )
            )
        )
    finally:
        auth.close()
    _owner_tick(rig)
    (repaired,) = _items(rig, state="dismissed")
    assert (
        repaired["dismissed_via"] == "mesh_repair"
    ), "the sign-in did not register as a repair on the dismissed row"

    # THE SAME LOGIN BREAKS AGAIN: the IdP refuses it once more.
    _tombstone_mcp(rig)
    auth = _auth(rig)
    try:
        rows = [r for r in auth.list_credentials("mcp-oauth") if r.identity_key == MCP_URL]
        data = dict(rows[0].data)
        data["tokens_obtained_at"] = time.time() - 7200
        auth.upsert_credential("mcp-oauth", data)
    finally:
        auth.close()
    assert isinstance(_borrow(rig), BrokerError)
    _owner_tick(rig)

    items = _items(rig)
    assert len(items) == 1, items
    assert items[0]["priority"] == "high", "a new break after a repair is new news"
    assert items[0]["read"] is False
    assert _doctor_repairs(rig), "the doctor row should show the new break"


def test_D6_U7_the_body_is_plain_text_with_the_url_on_its_own_line() -> None:
    """D6/U7: no backticks (the reel does not render Markdown), URL as a list item."""
    notice = RepairNotice(
        network_id="n_x",
        key="mcp:https://mcp.example.invalid/mcp",
        borrower="d_" + "2" * 32,
        borrower_name="build-box",
        login_name="",
        askers=(("d_" + "2" * 32, "build-box"),),
        failure="interactive_required",
        generation="g0",
    )

    body = repair_inbox.render_body(notice.askers, notice)

    assert "`" not in body, body
    assert body.splitlines() == [
        f"- {render_repair_notice('build-box', notice.key)}",
        "- https://mcp.example.invalid/mcp",
    ]


def test_D7_an_unnamed_mcp_key_leads_with_its_host() -> None:
    """D7: the host survives the 140- and 80-column list truncation."""
    rows = [
        RepairNotice(
            network_id="n_x",
            key=f"mcp:https://{host}/mcp",
            borrower="d_" + "2" * 32,
            borrower_name="build-box",
            login_name="",
            askers=(("d_" + "2" * 32, "build-box"),),
            failure="interactive_required",
            generation="g0",
        )
        for host in ("alpha.example.invalid", "beta.example.invalid")
    ]

    titles = [repair_inbox.render_title(n) for n in rows]

    assert titles[0] != titles[1]
    for title, host in zip(titles, ("alpha", "beta"), strict=True):
        assert title.startswith(f"{host}.example.invalid: run '/mcp login <name>'")
        # 80 columns is the narrowest list the renderer clips for; the host is
        # inside the first 24 characters, so it is never the part that is cut.
        assert len(title[: title.index(" ")]) <= 24


# ---------------------------------------------------------------------------
# #3522 follow-up: QA Q3 (re-break banner), review m1 = QA Q4 (atomic re-file),
# review m2 (the body's URL is redacted)
# ---------------------------------------------------------------------------


def _row(store: NotificationStore) -> dict[str, Any]:
    rows = [
        *store.list_items(state="all", session_id=repair_inbox.SESSION_ID)[0],
        *store.list_items(state="dismissed", session_id=repair_inbox.SESSION_ID)[0],
    ]
    (row,) = rows
    return row


def test_Q3_a_re_break_after_a_sign_in_is_bannered_again(tmp_path: Any) -> None:
    """QA Q3: sign in → the same login breaks again, no peer borrow in between.

    Nothing advances the generation, so the break reuses the ``g0`` row the sign-in
    closed. Its revision is the one the FIRST banner claimed; reopening it without
    releasing that watermark gave an item and a badge but no banner leg would
    raise it. The re-break is a new episode and must banner — exactly once.
    """
    episodes: dict[Any, Any] = {}
    repair_inbox.publish_notices(tmp_path, _planned("g0"), [], episodes=episodes)
    store = NotificationStore(tmp_path)
    first = _row(store)
    assert store.claim_delivery(first["id"], "feed"), "the first break's banner"

    # The sign-in: the window shows the key repaired, nothing is open.
    closed = repair_inbox.publish_notices(tmp_path, {}, ["openai"], episodes=episodes)
    assert closed.closed == 1, closed
    assert _row(store)["dismissed_via"] == repair_inbox.DISMISSED_VIA

    # The same login breaks again, still ``g0``.
    again = repair_inbox.publish_notices(tmp_path, _planned("g0"), [], episodes=episodes)
    assert again.reopened == 1, again

    pending = store.undelivered()
    print(f"Q3 undelivered after re-break: {[(r['key'], r['revision']) for r in pending]}")
    assert [r["id"] for r in pending] == [first["id"]]
    claims = [store.claim_delivery(first["id"], "feed") for _ in range(3)]
    print(f"Q3 claim_delivery x3: {claims}")
    assert claims == [True, False, False]


def test_Q4_the_re_file_is_never_observable_as_a_live_claimable_row(tmp_path: Any) -> None:
    """review m1 = QA Q4: an operator's dismissal re-filed as ``mesh_repair`` atomically.

    The first raise was never claimed (the operator dismissed it inside the feed's
    grace), so a moment in which the row is live and unread is a moment a banner leg
    could raise it — for a login that has just been repaired. Deterministic: a hook
    after EVERY committed store write holds the publish while a concurrent reader
    thread polls the row 200 times through its own connection.
    """
    import threading

    repair_inbox.publish_notices(tmp_path, _planned("g0"), [], episodes={})
    store = NotificationStore(tmp_path)
    item = _row(store)
    store.apply("dismiss", ids=[(item["id"], int(item["revision"]))], via="explicit")
    reader = NotificationStore(tmp_path)

    def exposed() -> bool:
        row = reader.get(item["id"])
        assert row is not None
        live_unread = not row["dismissed"] and not row["read"]
        return live_unread and int(row["delivered_rev"]) < int(row["revision"])

    # After EVERY committed write the publish pauses until the reader thread has
    # polled the row 200 more times on its own connection: each intermediate
    # state a write can leave is held still and read, so the outcome does not
    # depend on scheduling luck (QA's free-running poller hit the window 1 in 216).
    seen_by_hook: list[tuple[str, bool]] = []
    polls: list[bool] = []
    lock = threading.Lock()
    done = threading.Event()
    original_write = NotificationStore._write  # noqa: SLF001 — the store's one write seam

    def poll() -> None:
        while not done.is_set():
            flag = exposed()
            with lock:
                polls.append(flag)

    def observed_write(self: NotificationStore, body: Any, what: str) -> Any:
        result = original_write(self, body, what)
        with lock:
            target = len(polls) + 200
        while len(polls) < target:
            time.sleep(0.0005)
        seen_by_hook.append((what, exposed()))
        return result

    thread = threading.Thread(target=poll)
    thread.start()
    try:
        NotificationStore._write = observed_write  # type: ignore[method-assign]
        try:
            refiled = repair_inbox.publish_notices(tmp_path, {}, ["openai"], episodes={})
        finally:
            NotificationStore._write = original_write  # type: ignore[method-assign]
    finally:
        done.set()
        thread.join(timeout=30)

    print(f"Q4 writes seen by the hook: {seen_by_hook}")
    print(f"Q4 reader polls: {len(polls)}, exposed: {sum(polls)}")
    assert refiled.closed == 1, refiled
    assert _row(store)["dismissed_via"] == repair_inbox.DISMISSED_VIA
    assert seen_by_hook and not any(flag for _, flag in seen_by_hook), seen_by_hook
    assert len(polls) >= 200 and not any(polls)
    assert store.undelivered() == []


def test_m2_the_body_shows_scheme_host_and_path_only() -> None:
    """review m2: userinfo, query and fragment never reach the body (or a lock screen)."""
    # Joined at runtime so the fixture reads as what it is: synthetic, not a secret.
    userinfo = ":".join(("operator", "synthetic-pass"))
    query = "=".join(("access_token", "synthetic-query-token"))
    secret_bits = ("operator", "synthetic-pass", "access_token", "synthetic-query-token", "frag-x")
    notice = RepairNotice(
        network_id="n_x",
        key=f"mcp:https://{userinfo}@MCP.example.invalid:8443/v1/mcp?{query}&x=1#frag-x",
        borrower="d_" + "2" * 32,
        borrower_name="build-box",
        login_name="",
        askers=(("d_" + "2" * 32, "build-box"),),
        failure="interactive_required",
        generation="g0",
    )

    body = repair_inbox.render_body(notice.askers, notice)

    print(f"m2 body: {body!r}")
    assert body.splitlines() == [
        f"- {render_repair_notice('build-box', notice.key)}",
        "- https://mcp.example.invalid:8443/v1/mcp",
    ]
    tail = body.splitlines()[-1]
    assert not any(bit in tail for bit in secret_bits), tail


# ---------------------------------------------------------------------------
# #3534 round 1: review m1 (``revived`` inside the cooldown), m2 (``;params``),
# n1 (``expect_via`` in the store), QA Q5 (IPv6 title)
# ---------------------------------------------------------------------------


def test_review_m1_a_re_break_through_revived_is_bannered_inside_the_cooldown(
    tmp_path: Any,
) -> None:
    """The reviewer's sequence: a second asker had dropped the row to ``normal``.

    Then repair → re-break inside 30 minutes of the first banner. The re-raise is a
    priority escalation, so the store answers ``revived``, not ``absorbed``; the
    reopen used to run only on ``absorbed``, and the previous banner's cooldown
    refused the new episode's claim (``claim2=False``) until it ran out.
    """
    episodes: dict[Any, Any] = {}
    repair_inbox.publish_notices(tmp_path, _planned("g0", "laptop"), [], episodes=episodes)
    store = NotificationStore(tmp_path)
    item = _row(store)
    assert store.claim_delivery(item["id"], "feed"), "the first break's banner"
    joined = repair_inbox.publish_notices(
        tmp_path, _planned("g0", "laptop, desk"), [], episodes=episodes
    )
    assert joined.updated == 1 and _row(store)["priority"] == "normal", joined

    repair_inbox.publish_notices(tmp_path, {}, ["openai"], episodes=episodes)
    assert _row(store)["dismissed_via"] == repair_inbox.DISMISSED_VIA
    again = repair_inbox.publish_notices(
        tmp_path, _planned("g0", "laptop, desk"), [], episodes=episodes
    )

    row = _row(store)
    claims = [store.claim_delivery(item["id"], "feed") for _ in range(2)]
    print(f"m1 rebreak: {again} priority={row['priority']} claims={claims}")
    assert again.reopened == 1, again
    assert row["priority"] == "high" and not row["dismissed"] and not row["read"]
    assert claims == [True, False]


def test_review_n1_reopen_refuses_a_row_it_did_not_close(tmp_path: Any) -> None:
    """``expect_via`` is checked inside the store's transaction, not by the caller."""
    store = NotificationStore(tmp_path)
    store.notify("s1", title="t", priority="high", key="k")
    row = store.get_by_key("s1", "k")
    assert row is not None
    assert store.claim_delivery(row["id"], "feed")
    store.apply("dismiss", ids=[(row["id"], int(row["revision"]))], via="explicit")

    refused = store.reopen(row["id"], int(row["revision"]), expect_via="mesh_repair")

    after = store.get(row["id"])
    assert after is not None
    print(f"n1 reopen(explicit row) -> {refused}, dismissed_via={after['dismissed_via']}")
    assert refused is False
    assert after["dismissed"] and after["dismissed_via"] == "explicit"
    assert int(after["delivered_rev"]) == int(after["revision"]), "watermark untouched"


@pytest.mark.parametrize(
    ("url", "shown"),
    [
        ("https://h.example.invalid/a;tok=SYNTH", "https://h.example.invalid/a"),
        ("https://h.example.invalid/a;x=1/b;tok=SYNTH/c", "https://h.example.invalid/a/b/c"),
        ("https://h.example.invalid/mcp;tok=SYNTH?q=1#f", "https://h.example.invalid/mcp"),
        ("https://[2001:db8::1]:8443/mcp;tok=SYNTH", "https://[2001:db8::1]:8443/mcp"),
        ("https://h.example.invalid/plain", "https://h.example.invalid/plain"),
    ],
)
def test_review_m2_path_parameters_are_stripped(url: str, shown: str) -> None:
    """``urlsplit`` leaves ``;params`` in ``path``; a token there must not reach the body."""
    got = repair_inbox._display_url(url)  # noqa: SLF001 — the redaction under test
    print(f"m2 {url} -> {got}")
    assert got == shown
    assert "SYNTH" not in got


def test_Q5_an_ipv6_host_is_bracketed_in_the_title() -> None:
    """QA Q5: a bare IPv6 host ran its colons into the title's ``host: remedy``."""
    notice = RepairNotice(
        network_id="n_x",
        key="mcp:https://[2001:db8::1]:8443/mcp",
        borrower="d_" + "2" * 32,
        borrower_name="build-box",
        login_name="",
        askers=(("d_" + "2" * 32, "build-box"),),
        failure="interactive_required",
        generation="g0",
    )

    title = repair_inbox.render_title(notice)
    body = repair_inbox.render_body(notice.askers, notice)

    print(f"Q5 title: {title!r}")
    assert title.startswith("[2001:db8::1]: run '/mcp login <name>'"), title
    assert body.splitlines()[-1] == "- https://[2001:db8::1]:8443/mcp"
