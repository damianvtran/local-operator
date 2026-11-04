"""An account-scoped denial hides THAT account's row from the lend, not the provider.

#3427 QA round 2, Q-4 (comment 6639688647). The owner picks the row for each lend
with a session-hashed order (``AuthStore._selection_order``). When the pick landed on
an account the borrower is denied, the mint-time guard (``consent.serving_guard`` ->
``check_borrow``) refused ``not_a_holder`` and nothing tried the PERMITTED row beside
it: about half of a borrower's sessions were refused although the owner had a login
this borrower may use. The guard was right to refuse that row; the selection was
wrong to offer it.

Real path: two relays on loopback, mutually authenticated, the real
``credential share`` CLI and pull, the owner's real ``AuthStore`` with two native
rows, an account deny written with the owner's own ``consent.restrict_and_revoke``,
and the borrower asking through ``MeshCredentialClient.request_grant_sync`` (relay
leg 1 -> owner leg 2). Synthetic accounts and token labels; nothing leaves loopback.
"""

from __future__ import annotations

import time
from typing import Any

import pytest

from local_operator.network.credentials import consent
from local_operator.network.credentials import placement as placement_mod
from local_operator.network.credentials.types import BrokerError, Grant
from local_operator.providers.auth_store import AuthStore
from tests.unit.network.test_credentials_owner import (  # noqa: F401 — fixtures by import
    STUB_PROVIDER,
    idp,
    stub_provider,
)
from tests.unit.network.test_credentials_real_link import (  # noqa: F401 — fixtures by import
    _borrower_client,
    _pull,
    _share,
    mesh,
)
from tests.unit.network.test_relay_e2e import (  # noqa: F401 — fixtures by import
    _pair,
    devices,
)

DENIED = "account-one"
PERMITTED = "account-two"
#: At least 16 borrower sessions: the owner's order is session-hashed, so a sweep is
#: what exposes "half of them land on the denied row" (QA saw 7 of 14).
SESSIONS = [f"sweep-{i:02d}" for i in range(16)] + ["synthetic-session", ""]


def _row(store: AuthStore, account: str) -> Any:
    return store.upsert_credential(
        STUB_PROVIDER,
        {
            "type": "oauth",
            "access": f"synthetic-access-{account}",
            "refresh": f"synthetic-refresh-{account}",
            "expires": int((time.time() + 3600) * 1000),
            "account_id": account,
            "email": f"{account}@example.invalid",
        },
    )


def _deny(mesh: Any, selector: dict[str, Any]) -> None:  # noqa: F811 — the fixture's name
    consent.restrict_and_revoke(
        mesh.network_id, mesh.owner, mesh.borrower, selector, "synthetic-withdrawal", mesh.a.root
    )
    # Keep the holder row: the denial, not a missing share, must be what decides.
    with placement_mod.mutate(mesh.network_id, mesh.a.root, self_device=mesh.owner) as doc:
        doc.grant(STUB_PROVIDER, mesh.borrower, by=mesh.owner)


def _ask(
    mesh: Any, session_id: str, **kw: Any  # noqa: F811 — the fixture's name
) -> Grant | BrokerError:
    # A FRESH owner decision each time: forget the borrower's cached refusal, so the
    # reply is the owner broker's (and is counted in its audit).
    client = _borrower_client(mesh)
    client.state.clear(STUB_PROVIDER)
    client.state.save()
    return _borrower_client(mesh).request_grant_sync(STUB_PROVIDER, session_id=session_id, **kw)


def _owner_events(mesh: Any) -> list[dict[str, Any]]:  # noqa: F811 — the fixture's name
    mesh.a.audit.flush()
    return [
        row
        for row in mesh.a.audit.tail(2000)
        if str(row.get("event")) in {"credential.grant", "credential.grant_refused"}
    ]


@pytest.fixture
def owner_rows(mesh: Any) -> Any:  # noqa: F811 — the imported fixture
    _share(mesh)
    _pull(mesh)
    store = AuthStore(config_dir=mesh.a.root)
    store.delete_credentials_for_provider(STUB_PROVIDER)
    try:
        yield store
    finally:
        store.close()


def test_every_session_is_lent_the_permitted_row_never_the_denied_one(
    mesh: Any, owner_rows: AuthStore  # noqa: F811
) -> None:
    _row(owner_rows, DENIED)
    permitted = _row(owner_rows, PERMITTED)
    _deny(mesh, {"credential_key": STUB_PROVIDER, "account_ref": DENIED})
    before = len(_owner_events(mesh))

    outcomes: dict[str, str] = {}
    denied_tokens_lent = 0
    for sid in SESSIONS:
        reply = _ask(mesh, sid)
        if isinstance(reply, Grant):
            outcomes[sid] = str(reply.identity.get("account_id"))
            assert reply.credential_ref.credential_id == permitted.id
            denied_tokens_lent += reply.access_token == f"synthetic-access-{DENIED}"
        else:
            outcomes[sid] = f"refused:{reply.code}"

    assert denied_tokens_lent == 0
    assert outcomes == {sid: PERMITTED for sid in SESSIONS}, outcomes
    events = _owner_events(mesh)[before:]
    assert [e["event"] for e in events] == ["credential.grant"] * len(SESSIONS), [
        (e["event"], (e.get("detail") or {}).get("code")) for e in events
    ]


def test_a_denied_only_row_and_an_anchor_to_it_are_still_refused(
    mesh: Any, owner_rows: AuthStore  # noqa: F811
) -> None:
    denied = _row(owner_rows, DENIED)
    _deny(mesh, {"credential_key": STUB_PROVIDER, "account_ref": DENIED})

    # Only the denied account is signed in: nothing permitted exists, nothing is lent,
    # and an anchor to that row cannot make one.
    for sid in SESSIONS[:4]:
        reply = _ask(mesh, sid)
        assert isinstance(reply, BrokerError) and reply.code == "not_a_holder", (sid, reply)
    alone = _ask(mesh, SESSIONS[0], affinity=(int(denied.id), int(denied.created_at)))
    assert isinstance(alone, BrokerError) and alone.code == "not_a_holder", alone

    # A borrower ANCHORED to the denied row (its affinity) beside a permitted row: the
    # anchor is a preference among rows it may be lent, never a way to reach one.
    _row(owner_rows, PERMITTED)
    anchored = _ask(mesh, SESSIONS[0], affinity=(int(denied.id), int(denied.created_at)))
    assert isinstance(anchored, Grant), anchored
    assert anchored.identity.get("account_id") == PERMITTED
    assert anchored.access_token != f"synthetic-access-{DENIED}"


def test_a_provider_wide_deny_still_refuses_every_session(
    mesh: Any, owner_rows: AuthStore  # noqa: F811
) -> None:
    _row(owner_rows, DENIED)
    _row(owner_rows, PERMITTED)
    _deny(mesh, {"credential_key": STUB_PROVIDER})
    for sid in SESSIONS[:6]:
        reply = _ask(mesh, sid)
        assert isinstance(reply, BrokerError) and reply.code == "not_a_holder", (sid, reply)


def test_the_mint_guard_still_refuses_a_denied_row_the_exclusion_missed(
    mesh: Any, owner_rows: AuthStore, monkeypatch: pytest.MonkeyPatch  # noqa: F811
) -> None:
    """The exclusion is an optimisation of WHICH row is offered, not the authority.

    With it disabled, the old behaviour returns for the sessions that hash onto the
    denied row: refused ``not_a_holder`` by the unchanged mint-time guard, and the
    denied account's token is lent to nobody.
    """
    from local_operator.network.credentials import owner as owner_mod

    monkeypatch.setattr(
        owner_mod.MeshCredentialBroker, "_account_denied_row_ids", lambda *_a, **_k: set()
    )
    _row(owner_rows, DENIED)
    _row(owner_rows, PERMITTED)
    _deny(mesh, {"credential_key": STUB_PROVIDER, "account_ref": DENIED})
    replies = [_ask(mesh, sid) for sid in SESSIONS]
    assert not any(
        isinstance(r, Grant) and r.access_token == f"synthetic-access-{DENIED}" for r in replies
    )
    refused = [r for r in replies if isinstance(r, BrokerError)]
    assert refused and {r.code for r in refused} == {"not_a_holder"}
