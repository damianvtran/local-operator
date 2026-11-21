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
from tests.unit.network.test_credentials_hold_github_mcp import (  # noqa: F401
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


def test_an_unreachable_member_rm_names_each_lend_left_usable(
    held: Any, monkeypatch: pytest.MonkeyPatch  # noqa: F811
) -> None:
    """RED on main: the receipt carried only copy counts while B kept serving the MCP hold."""
    net = held.mesh
    receipt, server = _remove_while_offline(held, monkeypatch)
    try:
        # THE RESIDUAL THE RECEIPT MUST NAME (unchanged by this PR, by design): B
        # never learns, so it serves the MCP bearer from its hold past the window.
        _past_window(held.mcp)
        served = held.client.request_grant_sync(MCP_KEY, provider="mcp-oauth")
        assert isinstance(served, Grant) and served.held_offline, served
    finally:
        server.stop()
    assert "lends_open" in receipt, f"the removal receipt is silent about lends: {receipt}"
    open_keys = {lend["key"]: lend["until_ms"] for lend in receipt["lends_open"]}
    # The MCP bearer is named with the moment it stops being servable on B — the
    # owner-stated hold end, not the (earlier) grant window end.
    assert MCP_KEY in open_keys, receipt
    assert open_keys[MCP_KEY] == max(held.mcp.grant_expires_at_ms, held.mcp.hold_until_ms)
    assert open_keys[MCP_KEY] > held.mcp.grant_expires_at_ms
    # The App token is DELETEd at GitHub by the removal, so it is revoked, not open.
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
    assert any(
        f"'{MCP_KEY}' stays usable on qa-b until" in line and "rotate" in line for line in lines
    ), lines
    assert any("revoked at the source" in line for line in lines), lines
    # The removed member's rows leave the ledger: nothing is ever lent to it again.
    assert lends_mod.close_holder(net.a.root, net.network_id, net.borrower) == []


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
