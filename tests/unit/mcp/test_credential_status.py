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
from local_operator.network.credentials import mcp_state
from local_operator.network.credentials import placement as placement_mod
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


def test_reference_shapes_and_literals(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
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
        "notes": MCPStdioServerConfig(command="/bin/echo", env={"NOTES_TOKEN": "${" + SECRET + "}"})
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
    # Explicit keyword construction: a ``**{field: value}`` unpack makes the
    # constructor's ``type`` parameter look like it may receive a dict, which the
    # type checker rejects (review round 5, type-check).
    config = (
        MCPHttpServerConfig(url=URL, headers=value)
        if field == "headers"
        # ``env`` is an EXTRA field on an HTTP config (``extra="allow"``): the
        # status pass reads it as a reference source, but the constructor has no
        # such parameter, so the type checker needs the construct seam.
        else MCPHttpServerConfig.model_construct(url=URL, env=value)
    )
    configs = {"slack": config}

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


def test_durable_evidence_beats_a_live_holder_row(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """cloudflare/minerva-qa: the share is live, the OWNER's sign-in is dead.

    A holder row alone would say ``synced``; the last broker refusal is the more
    recent fact and must win, naming the device a person has to sign in on.
    """
    from local_operator.network.credentials import mcp_state
    from local_operator.network.credentials.types import credential_key_for_mcp

    root = tmp_path / "cfg"
    root.mkdir(parents=True)
    monkeypatch.setenv("LOCAL_OPERATOR_CONFIG_DIR", str(root))
    cwd = tmp_path / "proj"
    cwd.mkdir()
    store.save(_record(), root)
    _place(root)  # owner PEER, SELF is a holder
    mcp_state.record(
        NETWORK,
        credential_key_for_mcp(URL),
        state="expired",
        code="interactive_required",
        owner_device=PEER,
        owner_name="macbook",
        root=root,
    )

    configs = {"linear": MCPHttpServerConfig(url=URL, auth=MCPAuthConfig(type="oauth"))}
    rows = server_states(str(cwd), base=root, configs=configs)

    assert rows["linear"]["state"] == "expired", rows
    assert rows["linear"]["reason"] == "macbook: sign in again", rows["linear"]["reason"]
    assert "sign in again — sign in again" not in rows["linear"]["reason"]


def test_a_configured_server_dropped_by_discovery_still_has_a_row(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """gitlab was configured, absent from ``server_states`` and therefore invisible.

    ``load_all_mcp_configs`` drops disabled entries; the status pass must still
    name every server the user-scope document declares, because ``/mcp`` is the
    only place an operator can read why it is not working.
    """
    import json

    from local_operator.mcp.config import load_all_mcp_configs

    root = tmp_path / "cfg"
    root.mkdir(parents=True)
    monkeypatch.setenv("LOCAL_OPERATOR_CONFIG_DIR", str(root))
    (root / "mcp.json").write_text(
        json.dumps(
            {
                "mcpServers": {
                    "gitlab": {
                        "type": "http",
                        "url": URL,
                        "auth": {"type": "oauth"},
                    }
                },
                "disabledServers": ["gitlab"],
            }
        ),
        encoding="utf-8",
    )
    store.save(_record(), root)

    loaded, _sources = load_all_mcp_configs(str(root))
    assert "gitlab" not in loaded, "the fixture must exercise the dropped-server path"

    rows = server_states(str(root), base=root)
    assert rows["gitlab"]["state"] == "needs_operator_login", rows
    assert rows["gitlab"]["reason"]


def test_a_registration_only_row_is_not_a_login(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Review M3: ``has_row`` is ANY auth row; a discovered client is not a grant."""
    import json

    from local_operator.mcp.auth import McpTokenStorage
    from local_operator.providers.auth_store import AuthStore

    root = tmp_path / "cfg"
    root.mkdir(parents=True)
    monkeypatch.setenv("LOCAL_OPERATOR_CONFIG_DIR", str(root))
    (root / "mcp.json").write_text(
        json.dumps(
            {
                "mcpServers": {"gitlab": {"type": "http", "url": URL, "auth": {"type": "oauth"}}},
                "disabledServers": ["gitlab"],
            }
        ),
        encoding="utf-8",
    )
    store.save(_record(), root)
    grants = AuthStore(db_path=root / "auth.db", config_dir=root)
    McpTokenStorage(URL, grants).seed_client_info("client-1")

    rows = server_states(str(root), base=root)

    assert rows["gitlab"]["state"] == "needs_operator_login", rows
    assert rows["gitlab"]["shape"] == "oauth"


def test_a_ref_free_dropped_server_needs_no_credential(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Review m3: nothing to authenticate is a vacuous ``synced``, not a login."""
    import json

    root = tmp_path / "cfg"
    root.mkdir(parents=True)
    monkeypatch.setenv("LOCAL_OPERATOR_CONFIG_DIR", str(root))
    (root / "mcp.json").write_text(
        json.dumps(
            {
                "mcpServers": {"plain": {"type": "http", "url": URL}},
                "disabledServers": ["plain"],
            }
        ),
        encoding="utf-8",
    )
    store.save(_record(), root)

    rows = server_states(str(root), base=root)

    assert rows["plain"]["shape"] == "none", rows
    assert rows["plain"]["state"] == "synced", rows


def test_a_tombstoned_grant_reads_expired_on_both_paths(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Review M4: a dead marker is a GRANT that failed, not "never signed in".

    Covers both row sources: the enabled server (``_oauth_row``) and one the
    discovery loader dropped (``_fact_row``).
    """
    import asyncio
    import json

    from mcp.shared.auth import OAuthToken

    from local_operator.mcp.auth import McpTokenStorage
    from local_operator.providers.auth_store import AuthStore

    other = "https://dropped.test/mcp"
    root = tmp_path / "cfg"
    root.mkdir(parents=True)
    monkeypatch.setenv("LOCAL_OPERATOR_CONFIG_DIR", str(root))
    (root / "mcp.json").write_text(
        json.dumps(
            {
                "mcpServers": {
                    "enabled": {"type": "http", "url": URL, "auth": {"type": "oauth"}},
                    "dropped": {"type": "http", "url": other, "auth": {"type": "oauth"}},
                },
                "disabledServers": ["dropped"],
            }
        ),
        encoding="utf-8",
    )
    store.save(_record(), root)
    grants = AuthStore(db_path=root / "auth.db", config_dir=root)
    for server_url in (URL, other):
        storage = McpTokenStorage(server_url, grants)
        asyncio.run(
            storage.set_tokens(
                OAuthToken(
                    **{
                        "access_token": "tmp_path",
                        "token_type": "Bearer",
                        "expires_in": 3600,
                        "refresh_token": "tmp_path",
                    }
                )
            )
        )
        assert storage.mark_grant_dead() is True

    rows = server_states(str(root), base=root)

    assert rows["enabled"]["state"] == "expired", rows
    assert "sign in again on this device" in rows["enabled"]["reason"]
    assert rows["dropped"]["state"] == "expired", rows
    assert "sign in again on this device" in rows["dropped"]["reason"]


def test_an_unreadable_network_store_keeps_the_artifacts_verdict(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Review m7: an unreadable record store must not downgrade revoked to login.

    The artifacts-only path is what proves a device lost ACCESS; an early return
    on any read failure skipped it and answered ``needs_operator_login`` for a
    server whose placement documents were still there.
    """
    import json

    from local_operator.network.credentials import placement as placement_mod
    from local_operator.network.credentials.types import credential_key_for_mcp

    root = tmp_path / "cfg"
    root.mkdir(parents=True)
    monkeypatch.setenv("LOCAL_OPERATOR_CONFIG_DIR", str(root))
    (root / "mcp.json").write_text(
        json.dumps(
            {"mcpServers": {"linear": {"type": "http", "url": URL, "auth": {"type": "oauth"}}}}
        ),
        encoding="utf-8",
    )
    store.save(_record(), root)
    with placement_mod.mutate(NETWORK, root=root, self_device=PEER) as document:
        document.declare(
            credential_key_for_mcp(URL),
            owner_device=PEER,
            owner_device_name="macbook",
            provider="mcp-oauth",
            kind="mcp-rotating",
            by=PEER,
        )
        document.grant(credential_key_for_mcp(URL), SELF, by=PEER)

    def unreadable(*args: object, **kwargs: object) -> list[object]:
        raise OSError("network store unreadable")

    monkeypatch.setattr("local_operator.network.store.list_networks", unreadable)

    rows = server_states(str(root), base=root)

    assert rows["linear"]["state"] == "revoked", rows
    assert "revoked" in rows["linear"]["reason"]
