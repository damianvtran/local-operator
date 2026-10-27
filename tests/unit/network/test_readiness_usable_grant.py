"""F-READY-1: readiness asks the runtime's question — "is there a grant it would USE?".

The MCP SDK writes a credential row (``client_info`` only) the moment it DISCOVERS
a server. The runtime never counts that row — or a tombstoned one — as a local
grant (``mcp.auth.server_has_stored_grant``), so a session on that node borrows
the owner's share and works. Readiness counted ANY row as a login and told the
operator "sign in again on <node>": for a non-technical user, a trip to a headless
node to repair a login that never existed. The ``lop network credentials`` ledger
and the join-time offer had the same mismatch.

Every store here is a real ``AuthStore`` written through ``McpTokenStorage``'s own
writers, so the row shapes are the ones a node actually holds — not dict fixtures
that agree with the code by construction.
"""

from __future__ import annotations

import asyncio
import json
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest
from mcp.shared.auth import OAuthToken

from local_operator.mcp.auth import McpTokenStorage, server_has_stored_grant
from local_operator.network import readiness
from local_operator.network import store as net_store
from local_operator.network.credentials import mcp_state
from local_operator.network.credentials.types import credential_key_for_mcp
from local_operator.providers.auth_store import AuthStore
from tests.unit.mcp.test_credential_status import NETWORK
from tests.unit.mcp.test_credential_status import PEER as OWNER
from tests.unit.mcp.test_credential_status import SELF, URL, _place, _record
from tests.unit.network.test_readiness import _facts, _FakeViewer

HOLDER = SimpleNamespace(device_id=SELF, name="holder")
OWNER_PLACEMENT = {
    "owner_device": OWNER,
    "owner_device_name": "macbook",
    "holders": [{"device": SELF, "scope": "session"}],
}


def _write_shape(root: Path, shape: str) -> None:
    """Write one real row shape for ``URL`` into ``root/auth.db``, then close it."""
    auth = AuthStore(db_path=root / "auth.db", config_dir=root)
    try:
        storage = McpTokenStorage(URL, auth)

        def put(token: dict[str, Any]) -> None:
            asyncio.run(storage.set_tokens(OAuthToken.model_validate(token)))

        if shape == "registration-only":
            # What the SDK leaves behind on first discovery: a client, no tokens.
            storage.seed_client_info("synthetic-client")
        elif shape == "dead":
            put({"access_token": "synthetic-access", "refresh_token": "synthetic-refresh"})
            assert storage.mark_grant_dead(rejected_refresh_token="synthetic-refresh")
        elif shape == "expired-access-only":
            put({"access_token": "synthetic-access", "expires_in": 0})
        elif shape == "refresh-present":
            put(
                {
                    "access_token": "synthetic-access",
                    "refresh_token": "synthetic-refresh",
                    "expires_in": 0,
                }
            )
        elif shape == "valid":
            put({"access_token": "synthetic-access", "expires_in": 3600})
        else:  # pragma: no cover — a typo in the parametrize list
            raise AssertionError(shape)
    finally:
        auth.close()


def _holder_root(tmp_path: Path, monkeypatch: pytest.MonkeyPatch, *, placed: bool) -> Path:
    root = tmp_path / "cfg"
    root.mkdir()
    monkeypatch.setenv("LOCAL_OPERATOR_CONFIG_DIR", str(root))
    net_store.save(_record(), root)
    if placed:
        _place(root)
    (root / "mcp.json").write_text(
        json.dumps({"mcpServers": {"linear": {"type": "http", "url": URL}}}),
        encoding="utf-8",
    )
    return root


def _holder_row(root: Path, *, viewer: _FakeViewer) -> dict[str, Any]:
    fact = readiness.mcp_servers_fact(root)
    rows = readiness.mcp_credential_rows(
        HOLDER, _facts(mcp=fact), viewer=viewer, peer_label="holder"
    )
    assert len(rows) == 1
    return rows[0]


def test_registration_only_row_with_synced_share_borrows(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The F-READY-1 report: the share works, so the row says it borrows."""
    root = _holder_root(tmp_path, monkeypatch, placed=True)
    _write_shape(root, "registration-only")
    mcp_state.record(
        NETWORK,
        credential_key_for_mcp(URL),
        state="synced",
        owner_device=OWNER,
        owner_name="macbook",
        root=root,
    )

    row = _holder_row(root, viewer=_FakeViewer(device_id=OWNER, placement=OWNER_PLACEMENT))

    assert row["ok"] is True, row
    assert "holder borrows the login for linear" in row["detail"]
    assert "sign in again" not in json.dumps(row)


def test_registration_only_row_with_owner_expiry_names_the_owner(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """With nothing local to repair, the owner's refusal sends the user to the OWNER."""
    root = _holder_root(tmp_path, monkeypatch, placed=True)
    _write_shape(root, "registration-only")
    mcp_state.record(
        NETWORK,
        credential_key_for_mcp(URL),
        state="expired",
        code="interactive_required",
        owner_device=OWNER,
        owner_name="macbook",
        root=root,
    )

    row = _holder_row(root, viewer=_FakeViewer(device_id=OWNER, placement=OWNER_PLACEMENT))

    assert row["ok"] is False
    assert row["code"] == readiness.CODE_OBSERVED_FAILURE
    assert row["remedies"] == ["sign in again on macbook"]
    assert "holder's stored login" not in row["detail"]


@pytest.mark.parametrize(
    "shape", ["registration-only", "dead", "expired-access-only", "refresh-present", "valid"]
)
def test_readiness_local_answer_matches_the_runtime(
    shape: str, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """One rule: readiness attributes a server to the LOCAL login exactly when the
    runtime would use the local login instead of borrowing."""
    root = _holder_root(tmp_path, monkeypatch, placed=False)
    _write_shape(root, shape)
    auth = AuthStore(db_path=root / "auth.db", config_dir=root)
    try:
        runtime_uses_local = server_has_stored_grant(URL, auth)
    finally:
        auth.close()

    row = _holder_row(root, viewer=_FakeViewer(holds_mcp=False))
    readiness_says_local = row.get("code") == "local_grant_unavailable" or (
        row["ok"] is True and "holds its own login" in row["detail"]
    )

    assert readiness_says_local is runtime_uses_local, (shape, row)
    diagnostics = readiness.mcp_servers_fact(root)["servers"][0]["auth_diagnostics"]
    assert diagnostics["usable_grant"] is runtime_uses_local
    # The row itself still exists in every shape: ``has_row`` keeps meaning ANY row.
    assert diagnostics["has_row"] is True


def _old_node_server(**diagnostics: Any) -> dict[str, Any]:
    """A peer fact from a node that predates ``usable_grant`` (v0.90.14 shape)."""
    return {
        "name": "linear",
        "url": URL,
        "transport": "http",
        "has_row": True,
        "has_placement": True,
        "auth_diagnostics": {"has_row": True, "sharing_mode": "unavailable", **diagnostics},
    }


def test_old_node_without_usable_grant_absent_absent_is_no_row() -> None:
    server = _old_node_server(access_state="absent", refresh_presence="absent")
    rows = readiness.mcp_credential_rows(
        HOLDER,
        _facts(mcp={"servers": [server]}),
        viewer=_FakeViewer(device_id=OWNER, placement=OWNER_PLACEMENT),
        peer_label="holder",
    )
    assert rows[0]["ok"] is True and "borrows" in rows[0]["detail"], rows[0]


def test_old_node_real_expired_login_is_still_a_local_repair() -> None:
    """Negative control: the fallback must not swallow a REAL expired local login."""
    server = _old_node_server(access_state="expired", refresh_presence="absent")
    server["has_placement"] = False
    rows = readiness.mcp_credential_rows(
        HOLDER, _facts(mcp={"servers": [server]}), viewer=_FakeViewer(), peer_label="holder"
    )
    assert rows[0]["code"] == "local_grant_unavailable"
    assert rows[0]["remedies"] == ["sign in again on holder"]


@pytest.mark.parametrize(("shape", "login_here"), [("registration-only", False), ("valid", True)])
def test_shareable_ledger_counts_only_a_usable_login(
    shape: str, login_here: bool, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from local_operator.network import cli as net_cli

    root = _holder_root(tmp_path, monkeypatch, placed=False)
    _write_shape(root, shape)

    rows = net_cli._shareable_servers(SELF)  # noqa: SLF001

    assert [row["login_here"] for row in rows] == [login_here]
    if not login_here:
        assert rows[0]["remedy"] == readiness.mcp_login_remedy(URL)


@pytest.mark.parametrize(("shape", "offered"), [("registration-only", False), ("valid", True)])
def test_offer_serves_only_a_usable_login(
    shape: str, offered: bool, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from local_operator.network.credentials import offers

    root = _holder_root(tmp_path, monkeypatch, placed=False)
    _write_shape(root, shape)

    keys = [row["key"] for row in offers.build_items(root) if row["kind"] == "mcp-rotating"]

    assert keys == ([credential_key_for_mcp(URL)] if offered else [])


def test_header_reference_server_takes_the_reference_path() -> None:
    """Slack's shape: an http server whose header names a secret, no auth block and
    no login row. Its question is whether the peer resolves the name — not a read
    of this device's (possibly WAL-locked) auth.db, which answered "unknown"."""
    server = {
        "name": "slack",
        "url": "https://mcp.slack.example.test/mcp",
        "transport": "http",
        "has_row": False,
        "has_placement": False,
        "refs": [{"id": "SLACK_TOKEN", "set": True}],
    }
    rows = readiness.mcp_credential_rows(
        HOLDER,
        _facts(mcp={"servers": [server]}),
        viewer=_FakeViewer(holds_mcp=None),
        peer_label="holder",
    )
    assert rows[0]["ok"] is True, rows[0]
    assert rows[0]["detail"] == "holder resolves every referenced credential for slack"
