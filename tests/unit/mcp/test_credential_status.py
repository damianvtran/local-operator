"""One read-only status answer per MCP server, and the five shapes it must cover.

THE CONTRACT IS A REMEDY, NOT A MOOD (design §5). Each cell drives the real
function against real files under a private root and asserts the state AND the
sentence a person would act on — because the whole point is that ``auth-required``
alone does not tell them whether to sign in here, sign in THERE, repair a share, or
set a key.

Also pinned: the read is PURE. No placement document, no network record and no
grant store is created by the read, and no value ever appears in a reason (the
sentinel below is planted in the store and asserted absent from every string).
"""

from __future__ import annotations

import asyncio
import json
from pathlib import Path
from typing import Any

import pytest

from local_operator.mcp.config import (
    MCPAuthConfig,
    MCPHttpServerConfig,
    MCPStdioServerConfig,
)
from local_operator.mcp.credential_status import server_states
from local_operator.network import store, types, wire
from local_operator.network.credentials import mcp_state, placement as placement_mod
from local_operator.network.credentials.types import (
    credential_key_for_mcp,
    credential_key_for_secret,
)

SELF = "d_" + "1" * 32
PEER = "d_" + "2" * 32
NETWORK = "n_" + "a" * 24
URL = "https://linear.example.test/mcp"
KEY = credential_key_for_mcp(URL)
SECRET = "LINEAR_API_KEY"
PROBE_TEXT = "status-value-never-printed"
SENTINEL = "SENTINEL-STATUS-VALUE-NEVER-PRINTED"


def _record(*, self_active: bool = True) -> types.NetworkRecord:
    record = types.NetworkRecord(
        network_id=NETWORK,
        name="home-net",
        self_device_id=SELF,
        self_role="admin",
        self_capabilities=sorted(types.capabilities_for_role("admin")),
        epoch=1,
    )
    record.members.append(
        types.MemberRecord(
            device_id=SELF,
            public_key=wire.b64u(b"a" * 32),
            role="admin",
            capabilities=sorted(types.capabilities_for_role("admin")),
            added_via="self",
            lifecycle="active" if self_active else "expired",
            removed_at=None if self_active else 1.0,
        )
    )
    record.members.append(
        types.MemberRecord(
            device_id=PEER,
            public_key=wire.b64u(b"b" * 32),
            role="drive",
            capabilities=sorted(types.capabilities_for_role("drive")),
            added_via="invite",
        )
    )
    return record


def _place(
    root: Path,
    key: str = KEY,
    *,
    owner: str = PEER,
    holder: bool = True,
    kind: str = "mcp-rotating",
) -> None:
    with placement_mod.mutate(NETWORK, root=root, self_device=owner) as document:
        document.declare(
            key,
            owner_device=owner,
            owner_device_name="macbook",
            provider="mcp-oauth" if key.startswith("mcp:") else "secret",
            kind=kind,
            by=owner,
        )
        if holder:
            document.grant(key, SELF, by=owner)


def _grant(root: Path, url: str = URL) -> None:
    from mcp.shared.auth import OAuthToken

    from local_operator.mcp.auth import McpTokenStorage
    from local_operator.providers.auth_store import AuthStore

    store_handle = AuthStore(config_dir=root)
    storage = McpTokenStorage(url, store_handle)
    tokens = OAuthToken(
        **{
            "access_token": "access-" + PROBE_TEXT,
            "token_type": "Bearer",
            "expires_in": 3600,
            "refresh_token": "refresh",
        }
    )
    asyncio.run(storage.set_tokens(tokens))


def _oauth_server(url: str = URL) -> dict[str, Any]:
    return {
        "linear": MCPHttpServerConfig(url=url, auth=MCPAuthConfig(type="oauth")),
    }


def _configs(cwd: Path) -> dict[str, Any]:
    return {
        "linear": MCPHttpServerConfig(url=URL, auth=MCPAuthConfig(type="oauth")),
        "notes": MCPStdioServerConfig(command="/bin/echo", env={"NOTES_TOKEN": "${NOTES_TOKEN}"}),
        "slack": MCPHttpServerConfig(
            url="https://slack.example.test/mcp",
            headers={"Authorization": "${SLACK_AUTH}", "X-Support": "literal"},
        ),
    }


def test_each_state_and_reason(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    root = tmp_path / "cfg"
    monkeypatch.setenv("LOCAL_OPERATOR_CONFIG_DIR", str(root))
    cwd = tmp_path / "proj"
    cwd.mkdir()

    # 1. never signed in anywhere.
    rows = server_states(str(cwd), base=root, configs=_oauth_server())
    assert rows["linear"]["state"] == "needs_operator_login"
    assert "/mcp login" in rows["linear"]["reason"] or "sign-in" in rows["linear"]["reason"]

    # 2. a local grant row -> synced here.
    root.mkdir(parents=True, exist_ok=True)
    _grant(root)
    rows = server_states(str(cwd), base=root, configs=_oauth_server())
    assert (rows["linear"]["state"], rows["linear"]["reason"]) == ("synced", "")

    # 3. an owner's share, no local grant -> synced, and the reason names it.
    #    The local row from step 2 is removed first: a local grant WINS over a
    #    borrow (the manager's documented rule), so leaving it would measure the
    #    wrong branch.
    (root / "auth.db").unlink()
    store.save(_record(), root)
    _place(root)
    rows = server_states(str(cwd), base=root, configs=_oauth_server())
    assert rows["linear"]["state"] == "synced"
    assert "macbook" in rows["linear"]["reason"]

    # 4. the owner's refresh failed -> expired, and the reason names the owner.
    mcp_state.record(
        NETWORK,
        KEY,
        state="expired",
        code="refresh_failed",
        owner_device=PEER,
        owner_name="macbook",
        root=root,
    )
    rows = server_states(str(cwd), base=root, configs=_oauth_server())
    assert rows["linear"]["state"] == "expired"
    assert "macbook" in rows["linear"]["reason"]
    assert "refresh failed" in rows["linear"]["reason"]

    # 5. the share was revoked -> revoked (evidence outlives the placement row).
    mcp_state.record(
        NETWORK, KEY, state="revoked", code="not_a_holder", owner_name="macbook", root=root
    )
    rows = server_states(str(cwd), base=root, configs=_oauth_server())
    assert rows["linear"]["state"] == "revoked"


def test_reference_shapes_and_literals(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    root = tmp_path / "cfg"
    root.mkdir(parents=True)
    monkeypatch.setenv("LOCAL_OPERATOR_CONFIG_DIR", str(root))
    cwd = tmp_path / "proj"
    cwd.mkdir()
    # The USER-SCOPE document: the only file adoption reads, and the one the
    # status's literal scan and the config loader both see. Writing it here keeps
    # the two halves describing the same server (a config object passed in memory
    # would have its literals invisible to the disk scan by construction).
    (root / "mcp.json").write_text(
        json.dumps(
            {
                "mcpServers": {
                    "notes": {
                        "type": "stdio",
                        "command": "/bin/echo",
                        "env": {"NOTES_TOKEN": "${" + SECRET + "}"},
                    },
                    "slack": {
                        "type": "http",
                        "url": "https://slack.example.test/mcp",
                        "headers": {"Authorization": "${SLACK_AUTH}", "X-Support": "literal"},
                    },
                }
            }
        ),
        encoding="utf-8",
    )

    rows = server_states(str(root), base=root)

    # No store row and no share: the missing KEY is named, never a value.
    assert rows["notes"]["state"] == "missing_source"
    assert rows["notes"]["shape"] == "env-ref"
    assert SECRET in rows["notes"]["reason"]

    # A literal header is the pre-adoption state: adoption is the remedy, not a share.
    assert rows["slack"]["state"] == "missing_source"
    assert rows["slack"]["shape"] == "header-ref"
    assert "literal not adopted" in rows["slack"]["reason"]
    assert "lop network mcp credentials --adopt" in rows["slack"]["reason"]

    # Make the reference resolvable: the same row turns synced.
    from local_operator.secrets import access

    secret_store = access.open_store(root, create=True)
    secret_store.initialize()
    secret_store.set(SECRET, PROBE_TEXT.encode())
    rows = server_states(str(root), base=root)
    assert rows["notes"]["state"] == "synced"
    assert rows["notes"]["refs"] == [{"id": SECRET, "set": True}]
    rendered = " ".join(str(row.get("reason") or "") for row in rows.values())
    assert PROBE_TEXT not in rendered


def test_a_share_that_has_not_arrived_is_not_yet_synced(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    root = tmp_path / "cfg"
    root.mkdir(parents=True)
    monkeypatch.setenv("LOCAL_OPERATOR_CONFIG_DIR", str(root))
    cwd = tmp_path / "proj"
    cwd.mkdir()
    store.save(_record(), root)
    _place(root, credential_key_for_secret(SECRET), kind="store-secret")

    configs = {
        "notes": MCPStdioServerConfig(
            command="/bin/echo", env={"NOTES_TOKEN": "${" + SECRET + "}"}
        )
    }
    rows = server_states(str(cwd), base=root, configs=configs)
    assert rows["notes"]["state"] == "not_yet_synced"
    assert "--repair" in rows["notes"]["reason"]


def test_a_removed_device_reads_revoked(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    root = tmp_path / "cfg"
    root.mkdir(parents=True)
    monkeypatch.setenv("LOCAL_OPERATOR_CONFIG_DIR", str(root))
    cwd = tmp_path / "proj"
    cwd.mkdir()
    store.save(_record(self_active=False), root)
    _place(root)

    rows = server_states(str(cwd), base=root, configs=_oauth_server())
    assert rows["linear"]["state"] == "revoked"
    assert "no longer a member" in rows["linear"]["reason"]


def test_zero_peer_is_the_plain_local_answer(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """No network record, no placement, no store: a status read creates nothing."""
    root = tmp_path / "cfg"
    monkeypatch.setenv("LOCAL_OPERATOR_CONFIG_DIR", str(root))
    cwd = tmp_path / "proj"
    cwd.mkdir()
    configs = {"notes": MCPStdioServerConfig(command="/bin/echo")}

    rows = server_states(str(cwd), base=root, configs=configs)

    assert rows["notes"]["state"] == "synced"  # nothing to resolve
    assert not (root / "network").exists()
    assert not (root / "auth.db").exists()


@pytest.mark.parametrize(
    ("field", "value", "shape"),
    [
        ("headers", {"Authorization": "${" + SECRET + "}"}, "header-ref"),
        ("env", {"SERVICE_TOKEN": "${" + SECRET + "}"}, "env-ref"),
    ],
)
def test_a_placement_row_does_not_make_a_reference_server_oauth(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, field: str, value: dict[str, str], shape: str
) -> None:
    """``--repair`` mints ``mcp:<url>`` rows for header/env servers too.

    That placement row is how the holder population is addressed, NOT proof of
    OAuth. Treating it as proof classified slack as an OAuth server and ignored
    its ``Authorization`` reference: a set key read ``needs_operator_login`` and
    a shared-but-uncopied one said sign-in instead of ``not_yet_synced``.
    """
    root = tmp_path / "cfg"
    root.mkdir(parents=True)
    monkeypatch.setenv("LOCAL_OPERATOR_CONFIG_DIR", str(root))
    cwd = tmp_path / "proj"
    cwd.mkdir()
    store.save(_record(), root)
    _place(root)  # the mcp:<url> placement row, as --repair writes it
    # ...and the SECRET's own share row: the reference is a key in its own right.
    with placement_mod.mutate(NETWORK, root=root, self_device=PEER) as document:
        document.declare(
            credential_key_for_secret(SECRET),
            owner_device=PEER,
            owner_device_name="macbook",
            provider="secret",
            kind="store-secret",
            by=PEER,
        )
        document.grant(credential_key_for_secret(SECRET), SELF, by=PEER)
    configs = {
        "slack": MCPHttpServerConfig(url=URL, **{field: value}),
    }

    rows = server_states(str(cwd), base=root, configs=configs)
    assert rows["slack"]["shape"] == shape
    assert rows["slack"]["state"] == "not_yet_synced"

    from local_operator.secrets import access

    secret_store = access.open_store(root, create=True)
    secret_store.initialize()
    secret_store.set(SECRET, PROBE_TEXT.encode())
    rows = server_states(str(cwd), base=root, configs=configs)
    assert rows["slack"]["shape"] == shape
    assert rows["slack"]["state"] == "synced"
