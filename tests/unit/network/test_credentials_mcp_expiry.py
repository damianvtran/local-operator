"""An MCP token the owner KNOWS is dead must be refused, not lent.

AUDIT ROUND 2, F3 (§4.7). The owner serves MCP access tokens itself, refreshing
through ``ensure_mcp_oauth_fresh``. When that refresh cannot happen — no reachable
authorization server, or a grant the server already rejected — the row keeps the
token it had, and the design's §4.7 row says what the peer must be told:

    ``ensure_mcp_oauth_fresh`` cannot refresh, so the mesh rung returns
    ``interactive_required``; nothing opens a browser on either device.

Shipped behaviour before the fix, measured on the two-device rig: the owner lent the
expired bearer anyway. The peer then read a bare 401 from the MCP server — with no
repair notice for the operator, which is the one remedy §4.7 exists to route home.
``McpTokenStorage.stored_token_expiry`` is the owner's own answer to "is this token
alive", so the owner has everything it needs to refuse.
"""

from __future__ import annotations

import asyncio
import json
import time
from pathlib import Path
from typing import Any

import pytest

from local_operator.network import audit as audit_mod
from local_operator.network.credentials import owner as owner_mod
from local_operator.network.credentials import placement as placement_mod
from local_operator.network.credentials.types import (
    BrokerError,
    Grant,
    credential_key_for_mcp,
)
from local_operator.providers.auth_store import AuthStore

OWNER = "d_" + "1" * 32
BORROWER = "d_" + "2" * 32
NETWORK = "n_" + "3" * 24
SERVER_URL = "https://mcp.example.test/mcp"


@pytest.fixture()
def owner(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Any:
    """An owner whose MCP grant for ``SERVER_URL`` is expired.

    ``LOCAL_OPERATOR_CONFIG_DIR`` is pointed at the root because ``AuthStore`` derives
    its database from the config dir rather than from its ``config_dir`` argument —
    the same isolation rule the owner's other tests state.
    """
    root = tmp_path / "owner"
    root.mkdir()
    monkeypatch.setenv("LOCAL_OPERATOR_CONFIG_DIR", str(root))

    # The refresh is what the owner would use if it could; a test that let it reach the
    # network would be measuring DNS. ``None`` endpoints is exactly the state a server
    # with no metadata (or an unreachable one) leaves behind.
    async def no_endpoints(_url: str) -> None:
        return None

    monkeypatch.setattr("local_operator.mcp.auth.discover_oauth_endpoints", no_endpoints)

    auth = AuthStore(config_dir=root)
    key = credential_key_for_mcp(SERVER_URL)
    auth.upsert_credential(
        "mcp-oauth",
        {
            "tokens": {
                "access_token": "dead-access",
                "refresh_token": "refresh-0",
                "token_type": "Bearer",
                "expires_in": 3600,
            },
            # THE ROOT-LEVEL KEY IS THE ONE ``stored_token_expiry`` READS (the nested
            # copy the SDK writes is not consulted), and it is what makes this row
            # genuinely expired rather than merely old.
            "tokens_obtained_at": time.time() - 7200,
            "project_id": SERVER_URL,
        },
    )
    document = placement_mod.PlacementDocument(NETWORK, root=root, written_by=OWNER)
    document.declare(
        key,
        owner_device=OWNER,
        owner_device_name="owner-laptop",
        provider="mcp-oauth",
        kind="mcp-rotating",
        identity_label=SERVER_URL,
        by=OWNER,
    )
    document.grant(key, BORROWER, scope="session", by=OWNER)
    document.save()
    audit = audit_mod.AuditLog(root=root)
    broker = owner_mod.MeshCredentialBroker(
        root=root,
        self_device=OWNER,
        self_device_name="owner-laptop",
        network_id=NETWORK,
        auth_store=auth,
        audit=audit,
    )
    try:
        yield type(
            "Owner",
            (),
            {"root": root, "auth": auth, "broker": broker, "audit": audit, "key": key},
        )
    finally:
        broker.close()
        audit.close()
        auth.close()


def _resolve(owner: Any) -> Any:
    return asyncio.run(
        owner.broker._resolve_mcp(  # noqa: SLF001 — the owner's own resolve, as the broker calls it
            key=owner.key,
            provider="mcp-oauth",
            for_session="sess-1",
            force=False,
            holder_scope="session",
            by=BORROWER,
        )
    )


def test_an_expired_mcp_token_is_refused_with_the_repair_sentence(owner: Any) -> None:
    """§4.7: the peer is told `interactive_required` and the owner gets the notice."""
    outcome = _resolve(owner)

    assert isinstance(outcome, BrokerError), outcome
    assert outcome.code == "interactive_required"
    assert "mcp login" in outcome.message
    assert "owner-laptop" in outcome.message or OWNER in outcome.message


def test_the_requester_s_sentence_does_not_promise_an_unmade_request(owner: Any) -> None:
    """Review round 2, M3 — the sentence a BORROWER reads about this refusal.

    The ``interactive_required`` arm promised "the operator there has been asked to
    run …", while the design's §4.7 repair op that would ask them is not built (the
    amended row records that). The owner-side message asserted above is the WIRE
    diagnostic; the sentence below is the one rendered at the requester from the code.
    """
    from local_operator.network.credentials.messages import render_broker_error

    error = BrokerError(
        code="interactive_required",
        key=owner.key,
        owner_device=OWNER,
        owner_device_name="owner-laptop",
    )

    sentence = render_broker_error(error, key=owner.key, owner_name="owner-laptop")

    assert "can run" in sentence, sentence
    assert "has been asked" not in sentence, sentence
    assert "/mcp login" in sentence, sentence


def test_the_refusal_is_audited_as_the_owner_s_own_report(owner: Any) -> None:
    """The owner's log must show WHY nothing was lent — that is the forensic trail."""
    _resolve(owner)
    owner.audit.flush()
    rows = [row for row in owner.audit.tail(200) if row.get("event") == "credential.report"]
    assert rows, "no credential.report record for the refused MCP grant"
    assert rows[-1].get("detail", {}).get("failure") == "interactive_required"


def test_a_LIVE_mcp_token_is_still_lent(owner: Any) -> None:
    """The other arm — the fix must not turn a working MCP borrow into a refusal."""
    import sqlite3

    connection = sqlite3.connect(owner.root / "auth.db")
    row_id, data = connection.execute(
        "select id, data from auth_credentials where provider = 'mcp-oauth'"
    ).fetchone()
    payload = json.loads(data)
    payload["tokens_obtained_at"] = time.time()  # issued now; 3600 s of life left
    connection.execute(
        "update auth_credentials set data = ? where id = ?", (json.dumps(payload), row_id)
    )
    connection.commit()
    connection.close()

    outcome = _resolve(owner)
    assert isinstance(outcome, Grant), outcome
    assert outcome.access_token == "dead-access"


def test_a_row_with_no_recorded_lifetime_is_still_lent(owner: Any) -> None:
    """``None`` means "no opinion" — a non-expiring token must never be forced through
    a re-login, which is the rule ``McpTokenStorage.stored_token_expiry`` states at its
    own definition. Driven through the owner's resolve rather than the private helper."""
    import sqlite3

    connection = sqlite3.connect(owner.root / "auth.db")
    row_id, data = connection.execute(
        "select id, data from auth_credentials where provider = 'mcp-oauth'"
    ).fetchone()
    payload = json.loads(data)
    payload.pop("tokens_obtained_at", None)
    payload["tokens"].pop("expires_in", None)
    connection.execute(
        "update auth_credentials set data = ? where id = ?", (json.dumps(payload), row_id)
    )
    connection.commit()
    connection.close()

    outcome = _resolve(owner)
    assert isinstance(outcome, Grant), outcome
