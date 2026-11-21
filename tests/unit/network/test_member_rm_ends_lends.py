"""G9: a ``member rm`` says what it LENT that member, and what of it is still usable there.

THE GAP (measured on main 794d3053 with this file's rig). A member that is
REACHABLE when it is removed already loses every lend: the epoch frame tells it,
its client denies the network and drops each borrowed grant, and the owner DELETEs
App tokens (#3947 M2) and ends renewal lends (#3955 Q1) — pinned below as a guard
that is green on both trees. A member that is OFFLINE at removal (the lost or
stolen laptop) never learns: the owner refuses its later handshakes silently, so it
reads the owner as asleep and serves its MCP/provider bearers from the offline hold
(24 h by default). No issuer can revoke those, and the receipt said nothing — just
``copies 0``. The fix here is the receipt (the real fix — telling a removed member
at reconnect — is a separate security-reviewed change): ids-only lend ledger on the
owner, ``lends_revoked`` / ``lends_told`` / ``lends_open`` on the receipt.

Real path: the repo's ``held`` rig — two relays on loopback, real pairing, real
``credential share``, the owner lends a GitHub App token and an MCP bearer to B;
grant 2 s, hold 6 s. Synthetic ids only.
"""

from __future__ import annotations

import json
import time
from typing import Any

import pytest

from local_operator.network import relay
from local_operator.network.credentials import github as github_mod
from local_operator.network.credentials import lends as lends_mod
from local_operator.network.credentials import messages as messages_mod
from local_operator.network.credentials.types import BrokerError, Grant
from tests.unit.network import conftest as net_fixtures
from tests.unit.network.test_credentials_github import (  # noqa: F401 — fixtures by import
    _no_secret_broker_daemon,
    app_keypair,
    env,
    github_api,
    owner,
)
from tests.unit.network.test_credentials_held_offline import _stop_owner
from tests.unit.network.test_credentials_hold_github_mcp import (  # noqa: F401
    HOLD_S,
    MCP_KEY,
    _past_window,
    held,
)
from tests.unit.network.test_credentials_owner import idp, stub_provider  # noqa: F401
from tests.unit.network.test_credentials_real_link import (  # noqa: F401
    _lop_network,
    mesh,
)
from tests.unit.network.test_relay_e2e import (  # noqa: F401
    _pair,
    devices,
    serve_shaped_relay,
)

OTHER_HOLDER = "d_" + "7" * 32


def _remove_while_offline(rig: Any, monkeypatch: pytest.MonkeyPatch) -> tuple[dict[str, Any], Any]:
    """Stop B's relay, ``member rm`` it on A, bring B back: the lost-laptop shape."""
    net = rig.mesh
    net.b.stop()
    assert net_fixtures.wait_for(lambda: not any(link.alive for link in net.a.links.values()))
    receipt = net.a._ctl_member_rm(  # noqa: SLF001
        {"network": net.network_id, "device_id": net.borrower}
    )
    server = serve_shaped_relay(
        net.b.root,
        monkeypatch,
        settings=relay.NetworkSettings(port=net.b.settings.port, listen_address="127.0.0.1"),
        identity=net.b.identity,
        audit=net.b.audit,
    )
    server.bind()
    server.bind_control()
    server.start()
    return receipt, server


def _secrets(rig: Any) -> list[str]:
    return [rig.gh.access_token, rig.mcp.access_token]


def _refused_because_removed(client: Any, result: Any) -> bool:
    """True only when B's refusal is the REMOVAL, not the hold simply running out.

    Why both halves (review R1-1): the hold here is 6 s, so any "not a Grant"
    poll longer than that passes on ordinary expiry too — B never learning it was
    removed then still reads green, as ``owner_offline``. The removal is proven by
    B's OWN membership check reading inactive (the self row #4394's verified
    notice flips) AND the exact borrowed-authority refusal ``not_a_holder``.
    ``test_hold_expiry_alone_never_reads_as_removal`` is the negative control.
    """
    return (
        client._self_membership_active() is False  # noqa: SLF001
        and isinstance(result, BrokerError)
        and result.code == "not_a_holder"
    )


def test_an_unreachable_member_rm_names_each_lend_left_usable(
    held: Any, monkeypatch: pytest.MonkeyPatch  # noqa: F811
) -> None:
    """The receipt names each lend a member unreachable at removal can still use.

    What the receipt describes is the residual WHILE B stays unreachable: nothing
    the owner does reaches it, so its MCP hold stays servable until the named end.
    Once B dials back in, #4394's verified removal notice ends that early — so the
    reconnected B below is refused, which is the receipt's "until" being an upper
    bound, never a promise that B keeps serving (the two PRs landed separately,
    and this cell first asserted the pre-#4394 "B never learns" behaviour).
    """
    net = held.mesh
    receipt, server = _remove_while_offline(held, monkeypatch)
    try:
        # B came back and dialled A: first its OWN record must say it was removed
        # (the verified notice, applied to B's self row) — waiting on any refusal
        # instead would also pass once the 6 s hold simply ran out (R1-1).
        learned = net_fixtures.wait_for(
            lambda: not held.client._self_membership_active(),  # noqa: SLF001
            timeout_s=20.0,
        )
        assert learned, "B reconnected and still reads itself an active member"
        _past_window(held.mcp)
        refused = held.client.request_grant_sync(MCP_KEY, provider="mcp-oauth")
        assert _refused_because_removed(held.client, refused), refused
    finally:
        server.stop()
    assert "lends_open" in receipt, f"the removal receipt is silent about lends: {receipt}"
    open_keys = {lend["key"]: lend["until_ms"] for lend in receipt["lends_open"]}
    # The MCP bearer is named with the moment it stops being servable on B — the
    # owner-stated hold end, not the (earlier) grant window end.
    assert MCP_KEY in open_keys, receipt
    assert open_keys[MCP_KEY] == max(held.mcp.grant_expires_at_ms, held.mcp.hold_until_ms)
    assert open_keys[MCP_KEY] > held.mcp.grant_expires_at_ms
    # The live broker still tracks the App token, so the removal asks GitHub to
    # DELETE it: counted as requested, not open (a restarted relay is the next test).
    assert github_mod.GITHUB_KEY not in open_keys, receipt
    assert receipt["lends_revoked"] >= 1 and receipt["lends_told"] == 0, receipt
    assert net_fixtures.wait_for(lambda: not held.api.live(held.gh.access_token), timeout_s=10.0)
    # The sentence an operator reads, with its remedy.
    lines = messages_mod.render_removal_lends(
        "qa-b",
        revoked=receipt["lends_revoked"],
        open_lends=receipt["lends_open"],
        told=receipt["lends_told"],
        now_ms=held.mcp.grant_expires_at_ms - 1,
    )
    assert any("qa-b was not reachable" in line and "rotate" in line for line in lines), lines
    assert any(line.strip().startswith("MCP server ") and "until" in line for line in lines)
    assert any("revocation requested at the source" in line for line in lines), lines
    # The removed member's rows leave the ledger: nothing is ever lent to it again.
    assert lends_mod.close_holder(net.a.root, net.network_id, net.borrower) == []


def test_hold_expiry_alone_never_reads_as_removal(held: Any) -> None:  # noqa: F811
    """Negative control for R1-1: a member NOT removed, its hold aged past expiry.

    The owner goes away (so B serves from the hold), nobody runs ``member rm``,
    and the hold runs out. The old "anything but a Grant" poll passes here —
    which is exactly why it could not tell removal from expiry — while
    ``_refused_because_removed`` must stay False: B is still a member in its own
    eyes and the refusal is ``owner_offline``, not ``not_a_holder``.
    """
    client, net = held.client, held.mesh
    _stop_owner(net)
    _past_window(held.mcp)
    weak_passed = net_fixtures.wait_for(
        lambda: not isinstance(client.request_grant_sync(MCP_KEY, provider="mcp-oauth"), Grant),
        timeout_s=HOLD_S + 5,
    )
    assert weak_passed, "the hold never expired; the control proves nothing"
    expired = client.request_grant_sync(MCP_KEY, provider="mcp-oauth")
    assert isinstance(expired, BrokerError) and expired.code == "owner_offline", expired
    assert client._self_membership_active() is True  # noqa: SLF001
    assert not _refused_because_removed(client, expired), expired


def test_a_reachable_member_rm_ends_the_lends_on_the_member(held: Any) -> None:  # noqa: F811
    """Guard, GREEN on main and head: the reachable path already ends every lend on B."""
    client, net = held.client, held.mesh
    net.a._ctl_member_rm({"network": net.network_id, "device_id": net.borrower})  # noqa: SLF001
    for key, provider in ((MCP_KEY, "mcp-oauth"), (github_mod.GITHUB_KEY, github_mod.GITHUB_KEY)):
        refused = client.request_grant_sync(key, provider=provider)
        assert isinstance(refused, BrokerError) and refused.code == "not_a_holder", refused
    _past_window(held.mcp)
    refused = client.request_grant_sync(MCP_KEY, provider="mcp-oauth")
    assert isinstance(refused, BrokerError) and refused.code == "not_a_holder", refused


def test_a_reachable_member_rm_receipt_says_told_and_names_nothing_open(
    held: Any,  # noqa: F811
) -> None:
    """RED on main: no lend fields. A told member is not listed as holding anything."""
    net = held.mesh
    receipt = net.a._ctl_member_rm(  # noqa: SLF001
        {"network": net.network_id, "device_id": net.borrower}
    )
    assert receipt.get("lends_open") == [], receipt
    assert receipt.get("lends_told", 0) >= 1, receipt
    assert receipt.get("lends_revoked", 0) >= 1, receipt


def test_a_still_member_s_lends_are_untouched_and_absent_from_the_receipt(
    held: Any, monkeypatch: pytest.MonkeyPatch  # noqa: F811
) -> None:
    """Negative control: another holder's row survives B's removal and is never named."""
    net = held.mesh
    until = int(time.time() * 1000) + 600_000
    lends_mod.note_lend(
        net.a.root,
        net.network_id,
        holder=OTHER_HOLDER,
        key=MCP_KEY,
        grant_id="g_other",
        until_ms=until,
        revocable=False,
    )
    receipt, server = _remove_while_offline(held, monkeypatch)
    server.stop()
    assert all(lend["key"] for lend in receipt["lends_open"])
    rows = json.loads(lends_mod.lends_path(net.network_id, net.a.root).read_text())["lends"]
    assert [r for r in rows if r["holder"] == OTHER_HOLDER] == [
        {
            "holder": OTHER_HOLDER,
            "key": MCP_KEY,
            "grant_id": "g_other",
            "until_ms": until,
            "revocable": False,
        }
    ]
    assert all(r["holder"] != net.borrower for r in rows), rows
    assert OTHER_HOLDER not in json.dumps(receipt)


def test_no_secret_reaches_the_ledger_the_receipt_or_the_audit(
    held: Any, monkeypatch: pytest.MonkeyPatch  # noqa: F811
) -> None:
    net = held.mesh
    path = lends_mod.lends_path(net.network_id, net.a.root)
    before = path.read_text(encoding="utf-8")
    receipt, server = _remove_while_offline(held, monkeypatch)
    server.stop()
    net.a.audit.flush()
    audit_text = (net.a.root / "network" / "audit.jsonl").read_text(encoding="utf-8")
    for secret in _secrets(held):
        assert secret and secret not in before
        assert secret not in json.dumps(receipt)
        assert secret not in audit_text
    assert set(json.loads(before)["lends"][0]) == {
        "holder",
        "key",
        "grant_id",
        "until_ms",
        "revocable",
    }


# --- the ledger primitive ---------------------------------------------------


def test_one_row_per_holder_and_key_keeps_the_latest_end_and_prunes_the_past(
    tmp_path: Any,
) -> None:
    now = 1_000_000
    lends_mod.note_lend(
        tmp_path,
        "n_x",
        holder="d_b",
        key="k",
        grant_id="g1",
        until_ms=now + 50,
        revocable=True,
        now_ms=now,
    )
    lends_mod.note_lend(
        tmp_path,
        "n_x",
        holder="d_b",
        key="k",
        grant_id="g2",
        until_ms=now + 20,
        revocable=False,
        now_ms=now,
    )
    lends_mod.note_lend(
        tmp_path,
        "n_x",
        holder="d_c",
        key="k",
        grant_id="g3",
        until_ms=now + 5,
        revocable=False,
        now_ms=now,
    )
    rows = json.loads(lends_mod.lends_path("n_x", tmp_path).read_text())["lends"]
    b = [r for r in rows if r["holder"] == "d_b"]
    # The latest lease id, the LATEST end, and revocable only while every lease is.
    assert b == [
        {"holder": "d_b", "key": "k", "grant_id": "g2", "until_ms": now + 50, "revocable": False}
    ]
    # d_c's lease has ended by now+10: pruned, never returned.
    assert lends_mod.close_holder(tmp_path, "n_x", "d_c", now_ms=now + 10) == []
    assert [r["holder"] for r in lends_mod.close_holder(tmp_path, "n_x", "d_b", now_ms=now)] == [
        "d_b"
    ]


def test_a_relay_less_removal_never_counts_an_app_token_as_revoked(tmp_path: Any) -> None:
    """No broker runs there, so nothing DELETEs it: it is reported as left usable."""
    now = int(time.time() * 1000)
    lends_mod.note_lend(
        tmp_path,
        "n_x",
        holder="d_b",
        key="github",
        grant_id="g",
        until_ms=now + 60_000,
        revocable=True,
    )
    out = lends_mod.removal_lends(tmp_path, "n_x", "d_b", told=False, revokes_at_issuer=False)
    assert out["lends_revoked"] == 0
    assert out["lends_open"] == [{"key": "github", "until_ms": now + 60_000}]


def test_a_ledger_that_cannot_be_written_never_breaks_the_grant(tmp_path: Any) -> None:
    blocker = tmp_path / "network"
    blocker.write_text("not a directory", encoding="utf-8")
    lends_mod.note_lend(
        tmp_path, "n_x", holder="d_b", key="k", grant_id="g", until_ms=10**13, revocable=False
    )
    assert lends_mod.close_holder(tmp_path, "n_x", "d_b") == []


# --- round-1 remediation (G9 review R1-1..R1-6, S1, S3, QA-F1) ----------------


def _owner_broker(net: Any) -> Any:
    from local_operator.network.credentials import owner as owner_mod

    return getattr(net.a, owner_mod._BROKER_ATTR)  # noqa: SLF001


def test_an_app_lend_the_restarted_owner_no_longer_tracks_is_listed_open(
    held: Any, monkeypatch: pytest.MonkeyPatch  # noqa: F811
) -> None:
    """R1-1 / S1: RED on 93359481 — a ledger row alone was counted as revoked.

    The lender's registry is in memory by design, so an owner relay restart
    empties it while ``lends.json`` survives. Emptying the registry here is that
    restart's exact effect on the removal path: no DELETE happens, so the App
    token must be named open, with its end, and nothing counted as requested.
    """
    lender = _owner_broker(held.mesh)._github_lender  # noqa: SLF001
    assert lender.outstanding(key=github_mod.GITHUB_KEY, holder=held.mesh.borrower) >= 1
    with lender._lock:  # noqa: SLF001
        lender._tracked.clear()  # noqa: SLF001
    receipt, server = _remove_while_offline(held, monkeypatch)
    server.stop()
    open_keys = {lend["key"]: lend["until_ms"] for lend in receipt["lends_open"]}
    assert receipt["lends_revoked"] == 0, receipt
    assert github_mod.GITHUB_KEY in open_keys, receipt
    assert open_keys[github_mod.GITHUB_KEY] >= held.gh.grant_expires_at_ms
    # Honest: the token really is still live at GitHub, as the receipt now says.
    assert held.api.live(held.gh.access_token)


def test_a_delete_github_keeps_refusing_never_reads_as_revoked(
    held: Any, monkeypatch: pytest.MonkeyPatch  # noqa: F811
) -> None:
    """QA-F1 / S1: RED on 93359481 — it printed "revoked at the source" regardless."""
    held.api.fail_next_revokes = 1000
    receipt, server = _remove_while_offline(held, monkeypatch)
    server.stop()
    assert receipt["lends_revoked"] >= 1, receipt
    lines = messages_mod.render_removal_lends(
        "qa-b",
        revoked=receipt["lends_revoked"],
        open_lends=receipt["lends_open"],
        told=receipt["lends_told"],
        renewal_ended=receipt["lends_renewal_ended"],
    )
    assert any("revocation requested at the source" in line for line in lines), lines
    assert not any(" revoked" in line for line in lines), lines
    assert held.api.live(held.gh.access_token)


def test_renewal_ends_have_their_own_count_and_never_read_as_revoked(
    tmp_path: Any, monkeypatch: pytest.MonkeyPatch
) -> None:
    """R1-2: RED on 93359481 — a renewal end was added to ``lends_revoked``."""
    from local_operator.network.credentials import renewal as renewal_mod

    now = int(time.time() * 1000)
    lends_mod.note_lend(
        tmp_path,
        "n_x",
        holder="d_b",
        key="openai",
        grant_id="g",
        until_ms=now + 60_000,
        revocable=False,
    )
    monkeypatch.setattr(renewal_mod, "end_lends", lambda *a, **k: 1)
    out = lends_mod.removal_lends(tmp_path, "n_x", "d_b", told=False)
    assert out["lends_revoked"] == 0 and out["lends_renewal_ended"] == 1, out
    lines = messages_mod.render_removal_lends(
        "qa-b",
        revoked=out["lends_revoked"],
        open_lends=out["lends_open"],
        told=0,
        renewal_ended=out["lends_renewal_ended"],
        now_ms=now,
    )
    # One credential never reads as both revoked and usable.
    assert not any("revoked" in line for line in lines), lines
    assert any("stopped waiting on qa-b for 1 lent login" in line for line in lines), lines
    assert any(line.strip().startswith("openai until") for line in lines), lines


def test_a_revocable_row_with_no_registry_reader_is_open(tmp_path: Any) -> None:
    """R1-1: without proof a DELETE will be attempted, the honest default is open."""
    now = int(time.time() * 1000)
    for tracked in (None, lambda key, holder: False):
        lends_mod.note_lend(
            tmp_path,
            "n_x",
            holder="d_b",
            key="github",
            grant_id="g",
            until_ms=now + 60_000,
            revocable=True,
        )
        out = lends_mod.removal_lends(tmp_path, "n_x", "d_b", told=False, revoke_tracked=tracked)
        assert out["lends_revoked"] == 0, out
        assert out["lends_open"] == [{"key": "github", "until_ms": now + 60_000}], out
    lends_mod.note_lend(
        tmp_path,
        "n_x",
        holder="d_b",
        key="github",
        grant_id="g",
        until_ms=now + 60_000,
        revocable=True,
    )
    out = lends_mod.removal_lends(
        tmp_path, "n_x", "d_b", told=False, revoke_tracked=lambda key, holder: True
    )
    assert out["lends_revoked"] == 1 and out["lends_open"] == [], out


def test_the_receipt_shows_labels_names_the_peer_and_says_rotate_once() -> None:
    """R1-3 + R1-5: RED on 93359481 — raw ``mcp:<url>?query`` keys, "it", one tail per key."""
    future = int(time.time() * 1000) + 3_600_000
    lines = messages_mod.render_removal_lends(
        "qa-b",
        revoked=0,
        open_lends=[
            {"key": "mcp:https://mcp.example.com:8443/sse?session=q-fake-7", "until_ms": future},
            {"key": "secret:CRM_API_KEY", "until_ms": future},
            {"key": "anthropic", "until_ms": future},
        ],
        told=0,
    )
    text = "\n".join(lines)
    assert "mcp:" not in text and "q-fake-7" not in text and "/sse" not in text, text
    assert "MCP server mcp.example.com until" in text, text
    assert "CRM_API_KEY until" in text and "secret:" not in text, text
    assert "anthropic until" in text, text
    assert "qa-b was not reachable" in text and "it was not reachable" not in text, text
    assert text.count("rotate") == 1, text


def test_a_corrupt_ledger_is_set_aside_not_overwritten(tmp_path: Any) -> None:
    """R1-6: RED on 93359481 — the next write replaced the unreadable bytes."""
    path = lends_mod.lends_path("n_x", tmp_path)
    path.parent.mkdir(parents=True)
    path.write_text("{not json", encoding="utf-8")
    lends_mod.note_lend(
        tmp_path, "n_x", holder="d_b", key="k", grant_id="g", until_ms=10**13, revocable=False
    )
    aside = path.with_name(path.name + ".corrupt")
    assert aside.read_text(encoding="utf-8") == "{not json"
    assert json.loads(path.read_text(encoding="utf-8"))["lends"][0]["holder"] == "d_b"


@pytest.mark.parametrize("bad", ["../escape", "a/b", "..", "x:y"])
def test_an_unsafe_network_id_never_becomes_a_path(tmp_path: Any, bad: str) -> None:
    """S3: RED on 93359481 — ``lends_path`` joined the id unchecked."""
    from local_operator.network.store import MeshRefusal

    with pytest.raises(MeshRefusal):
        lends_mod.lends_path(bad, tmp_path)
    lends_mod.note_lend(
        tmp_path, bad, holder="d_b", key="k", grant_id="g", until_ms=10**13, revocable=False
    )
    assert not list(tmp_path.rglob("lends.json"))
