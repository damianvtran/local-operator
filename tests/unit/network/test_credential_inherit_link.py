"""The authored permission reaches a RUNNING recipient session over two real relays.

``test_credentials_reconcile_link`` proves the projector against a permission written by
calling ``consent.provision_from_run`` directly. This file closes the other half: the
permission is AUTHORED by the shipped operator flow -- ``credential inherit
--request-approval`` -> ``approvals approve`` -> ``credential inherit --approval`` through the
real parser -- and then the product's own cadence is the only driver: the recipient's
``MeshAwareAuthStore`` is built before the approval and before the login, the owner
re-logs in as the SAME account against a rotating loopback IdP that refuses a spent refresh
token, and the session resolves a real Grant with no ``credential share``, no pull and no
restart.

The signer is the existing software file-only operator key; a Touch ID gesture is not
exercised and this file does not claim it.
"""

from __future__ import annotations

import json
import time
from typing import Any

import pytest

from local_operator.network import definitions, store
from local_operator.providers.auth_store import AuthStore
from tests.unit.network import conftest as net_fixtures
from tests.unit.network.test_credential_inherit import software_signer
from tests.unit.network.test_credentials_owner import STUB_PROVIDER  # noqa: F401
from tests.unit.network.test_credentials_real_link import _lop_network
from tests.unit.network.test_credentials_reconcile_link import (  # noqa: F401 — fixtures
    RotatingIdP,
    _holder,
    _login,
    idp,
)
from tests.unit.network.test_credentials_reconcile_link import (  # noqa: F401
    rmesh as reconcile_mesh,
)
from tests.unit.network.test_credentials_reconcile_link import (  # noqa: F401 — fixtures
    stub_provider,
)
from tests.unit.network.test_relay_e2e import devices  # noqa: F401 — fixture by import

# The reconcile suite's two-relay rig, re-exported under the name the cells request: importing
# it directly as ``rmesh`` trips F811 on every parameter that uses it.
rmesh = reconcile_mesh


def author(m: Any, capsys: pytest.CaptureFixture[str], monkeypatch: pytest.MonkeyPatch) -> str:
    """The operator's whole flow through the real CLI; returns the approval id."""
    software_signer(m.a.root, monkeypatch)
    base = ["credential", "inherit", STUB_PROVIDER, "--to", m.recipient, "--network", m.network_id]
    capsys.readouterr()
    assert _lop_network(*base, "--request-approval", "--json") == 0
    aid = json.loads(capsys.readouterr().out)["approval_id"]
    assert _lop_network("approvals", "approve", aid, "--json") == 0
    capsys.readouterr()
    assert _lop_network(*base, "--approval", aid, "--json") == 0
    assert json.loads(capsys.readouterr().out)["state"] == "connected"
    return aid


def status(m: Any, capsys: pytest.CaptureFixture[str]) -> dict[str, Any]:
    capsys.readouterr()
    code = _lop_network(
        "credential", "inherit", STUB_PROVIDER, "--to", m.recipient, "--network", m.network_id,
        "--status", "--json",
    )  # fmt: skip
    out = capsys.readouterr().out
    assert code == 0, out
    return json.loads(out)


def test_authored_permission_then_same_account_relogin_heals_the_running_session(
    rmesh: Any, capsys: pytest.CaptureFixture[str], monkeypatch: pytest.MonkeyPatch
) -> None:
    m = rmesh
    m.grant_caps()
    # The owner is signed in to acct-1 BEFORE authoring: the permission binds to the account
    # the inventory reports, never to a guess.
    _login(m.a.root, m.idp, account="acct-1")
    assert status(m, capsys)["state"] == "no-consent"
    assert m.borrow() is None, "no permission yet: the session must be on its plain path"
    members = store.record_path(m.network_id, m.a.root).read_bytes()

    aid = author(m, capsys, monkeypatch)
    # Authoring wrote the permission and nothing else: no capability, no holder.
    assert store.record_path(m.network_id, m.a.root).read_bytes() == members
    assert _holder(m, m.a.root, m.owner) is None
    assert status(m, capsys)["state"] == "ready"

    # A fresh sign-in of the SAME account (new row version, rotating refresh token).
    _login(m.a.root, m.idp, account="acct-1")
    for _ in range(3):
        m.tick()
    holder = _holder(m, m.b.root, m.recipient)
    assert holder is not None and holder.granted_by == "consent:" + aid, holder
    access = m.borrow()
    assert access is not None, "the RUNNING session did not heal from the AUTHORED permission"
    assert len(m.idp.posts) == 1, m.idp.posts
    assert len(set(m.idp.posts)) == len(m.idp.posts), "a spent refresh token was replayed"


def test_a_new_unapproved_account_costs_no_idp_post_and_status_says_permission_not_delivery(
    rmesh: Any, capsys: pytest.CaptureFixture[str], monkeypatch: pytest.MonkeyPatch
) -> None:
    m = rmesh
    m.grant_caps()
    _login(m.a.root, m.idp, account="acct-1")
    author(m, capsys, monkeypatch)
    for _ in range(2):
        m.tick()
    assert m.borrow() is not None
    posts = len(m.idp.posts)
    owner_auth = AuthStore(config_dir=m.a.root)
    owner_auth.delete_credentials_for_provider(STUB_PROVIDER)
    owner_auth.close()
    _login(m.a.root, m.idp, account="acct-2")  # NOT the approved account
    for _ in range(2):
        m.tick()
    assert m.borrow("other") is None
    assert len(m.idp.posts) == posts, "the unapproved account was refreshed (forbidden IdP POST)"
    report = status(m, capsys)
    text = "\n".join(report["lines"])
    assert report["state"] == "notready" and "scope_mismatch" in text
    assert "delivery" in report["scope"] and "not observed" in report["scope"]


def test_the_authored_permission_heals_on_the_real_shipped_cadence_threads(
    rmesh: Any, capsys: pytest.CaptureFixture[str], monkeypatch: pytest.MonkeyPatch
) -> None:
    """The acceptance path: actual 15 s run loops, never a manual tick/share/pull/restart."""
    from local_operator.network.sync import SYNC_TICK_S

    m = rmesh
    facade = m.session
    m.grant_caps()
    _login(m.a.root, m.idp, account="acct-1")
    aid = author(m, capsys, monkeypatch)
    syncers = [definitions.DefinitionsSyncer(server) for server in (m.a, m.b)]
    assert all(syncer._tick_s == SYNC_TICK_S == 15.0 for syncer in syncers)  # noqa: SLF001
    for syncer in syncers:
        syncer.start()
    try:
        started = time.monotonic()
        _login(m.a.root, m.idp, account="acct-1")
        assert net_fixtures.wait_for(
            lambda: _holder(m, m.b.root, m.recipient) is not None, timeout_s=45
        ), "the running caller never received the holder derived from the authored permission"
        elapsed = time.monotonic() - started
        holder = _holder(m, m.b.root, m.recipient)
        assert holder is not None and holder.granted_by == "consent:" + aid
        access = m.borrow("same-running-session")
        assert m.session is facade and access is not None
        assert m.idp.posts == ["refresh-0"]
        print(f"authored_cadence: same_facade=true tick_s=15 holder_after_s={elapsed:.2f}")
    finally:
        for syncer in syncers:
            syncer.stop()
        for syncer in syncers:
            syncer.join(timeout=15)
            assert not syncer.is_alive(), "the acceptance rig leaked a syncer thread"
