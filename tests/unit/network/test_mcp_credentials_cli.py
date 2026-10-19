"""``lop network mcp credentials --adopt|--repair`` — the owner-side repair surface.

The property this file pins: after adoption a referenced secret is a normal
``secret:<NAME>`` placement key, so ``--repair`` grants it through the SAME
document discipline every other share uses, to the population the design names:
HTTP/SSE holders of the server's ``mcp:<url>`` key, and the active drive/admin
``device`` members ``step_provision`` would provision for stdio. A ``local-only``
mark still wins, and one key's refusal never costs the others.

Every cell drives the shipped parser and handler against a real network record,
a real placement document and the real encrypted store under ``root``. Values are
sentinels; every output assertion also checks they never appear.
"""

from __future__ import annotations

import argparse
from argparse import Namespace
import json
from pathlib import Path
from typing import Any

import pytest

from local_operator.network import cli as net_cli
from local_operator.network import store as network_store
from local_operator.network import types, wire
from local_operator.network.credentials import placement as placement_mod
from local_operator.network.credentials import sync as sync_mod
from local_operator.network.credentials.types import (
    credential_key_for_mcp,
    credential_key_for_secret,
)

SELF = "d_" + "1" * 32
DRIVE = "d_" + "2" * 32
READER = "d_" + "3" * 32
POOL = "d_" + "4" * 32
NETWORK = "n_0123456789abcdef01234567"
URL = "https://gl.test/mcp"
TOKEN_SENTINEL = "SENTINEL-GITLAB-TOKEN"
SLACK_SENTINEL = "SENTINEL-SLACK-BEARER"


@pytest.fixture(autouse=True)
def _isolate(monkeypatch: pytest.MonkeyPatch, root: Path) -> None:
    """Point the handler's ``config_dir()`` at the private root, no broker."""
    monkeypatch.setenv("LOCAL_OPERATOR_CONFIG_DIR", str(root))
    monkeypatch.setattr("local_operator.secrets.client.ensure_broker", lambda *a, **k: False)


def _record() -> types.NetworkRecord:
    record = types.NetworkRecord(
        network_id=NETWORK,
        name="home-net",
        created_by=SELF,
        self_device_id=SELF,
        self_role="admin",
        self_capabilities=sorted(types.capabilities_for_role("admin")),
        epoch=1,
    )
    for device, role, kind in (
        (SELF, "admin", "device"),
        (DRIVE, "drive", "device"),
        (READER, "read", "device"),
        (POOL, "drive", "pool"),
    ):
        record.members.append(
            types.MemberRecord(
                device_id=device,
                public_key=wire.b64u(device.encode()[:32].ljust(32, b"x")),
                name=device[-4:],
                role=role,
                kind=kind,  # type: ignore[arg-type]
                added_via="invite",
            )
        )
    return record


def _write_config(root: Path, servers: dict[str, Any]) -> Path:
    path = root / "mcp.json"
    path.write_text(json.dumps({"mcpServers": servers}, indent=2) + "\n", encoding="utf-8")
    return path


def _seed_url_holders(root: Path, devices: list[str]) -> None:
    key = credential_key_for_mcp(URL)
    with placement_mod.mutate(NETWORK, root=root, self_device=SELF) as document:
        document.declare(
            key,
            owner_device=SELF,
            owner_device_name="mac",
            provider="mcp-oauth",
            kind="mcp-rotating",
            by=SELF,
        )
        for device in devices:
            document.grant(key, device, by=SELF)


def _seed_server_holders(root: Path) -> None:
    _seed_url_holders(root, [SELF, DRIVE])


def _run(capsys: pytest.CaptureFixture[str], *argv: str) -> tuple[int, dict[str, Any]]:
    parser = argparse.ArgumentParser(prog="lop")
    net_cli.add_parser(parser.add_subparsers(dest="command"))
    args = parser.parse_args(["network", "mcp", "credentials", *argv, "--json"])
    rc = net_cli.main(args)
    return rc, json.loads(capsys.readouterr().out)


def _run_human(capsys: pytest.CaptureFixture[str], *argv: str) -> tuple[int, str]:
    parser = argparse.ArgumentParser(prog="lop")
    net_cli.add_parser(parser.add_subparsers(dest="command"))
    args = parser.parse_args(["network", "mcp", "credentials", *argv])
    rc = net_cli.main(args)
    return rc, capsys.readouterr().out


def _run_invite(
    root: Path,
    capsys: pytest.CaptureFixture[str],
    monkeypatch: pytest.MonkeyPatch,
    *,
    as_json: bool,
) -> tuple[int, str]:
    """Invoke the SHIPPED invite handler with the relay answer stubbed."""
    token = root / "inv1.invite"
    token.write_text("TOKEN", encoding="utf-8")
    payload = {
        "invite_id": "inv1",
        "path": str(token),
        "role": "drive",
        "expires_in_s": 600.0,
        "hosts": ["127.0.0.1:4197"],
    }
    monkeypatch.setattr(net_cli, "_relay_call", lambda *a, **k: dict(payload))
    args = argparse.Namespace(
        network="",
        role="drive",
        expires=600.0,
        hosts="",
        device="",
        print_token=False,
        json=as_json,
    )
    rc = net_cli._cmd_invite(args)
    return rc, capsys.readouterr().out


def _entry(key: str, root: Path) -> Any:
    document = placement_mod.PlacementDocument.load(NETWORK, root, self_device=SELF)
    return document.entry(key)


def _holders(key: str, root: Path) -> list[str]:
    entry = _entry(key, root)
    return [holder.device for holder in entry.holders] if entry is not None else []


def _store_value(root: Path, name: str) -> str:
    from local_operator.secrets import access

    return access.open_store(root).get(name).decode("utf-8")


def _servers(root: Path) -> dict[str, Any]:
    return json.loads((root / "mcp.json").read_text(encoding="utf-8"))["mcpServers"]


# ---------------------------------------------------------------------------
# --adopt
# ---------------------------------------------------------------------------


def test_adopt_json_reports_names_and_grants_nothing(
    root: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    _write_config(
        root,
        {"gh": {"type": "stdio", "command": "npx", "env": {"GITHUB_TOKEN": TOKEN_SENTINEL}}},
    )

    rc, payload = _run(capsys, "--adopt")

    assert rc == 0
    assert payload["adopted"] == [
        {"server": "gh", "field": "env", "key": "GITHUB_TOKEN", "secret": "GITHUB_TOKEN"}
    ]
    assert _servers(root)["gh"]["env"] == {"GITHUB_TOKEN": "${GITHUB_TOKEN}"}
    assert _store_value(root, "GITHUB_TOKEN") == TOKEN_SENTINEL
    # --adopt writes the SOURCE half only; no placement is granted by it.
    assert _entry(credential_key_for_secret("GITHUB_TOKEN"), root) is None
    assert TOKEN_SENTINEL not in json.dumps(payload)


# ---------------------------------------------------------------------------
# --repair
# ---------------------------------------------------------------------------


def test_repair_grants_http_refs_to_holders_and_stdio_refs_to_eligible_members(
    root: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    network_store.save(_record(), root)
    _seed_server_holders(root)
    _write_config(
        root,
        {
            "gl": {
                "type": "http",
                "url": URL,
                "headers": {"Authorization": f"Bearer {SLACK_SENTINEL}"},
            },
            "gh": {"type": "stdio", "command": "npx", "env": {"GITHUB_TOKEN": TOKEN_SENTINEL}},
        },
    )

    rc, payload = _run(capsys, "--repair")

    assert rc == 0, payload
    rows = {row["server"]: row for row in payload["servers"]}
    assert rows["gl"]["shape"] == "header-ref"
    assert rows["gh"]["shape"] == "env-ref"
    granted = {item["key"]: item for row in payload["servers"] for item in row.get("grants") or []}
    # HTTP: exactly the mcp:<url> holders (SELF and DRIVE), never READER.
    assert set(granted["secret:MCP_GL_AUTHORIZATION"]["devices"]) == {SELF, DRIVE}
    # stdio: the provisioning population — active drive/admin device members.
    assert set(granted["secret:GITHUB_TOKEN"]["devices"]) == {SELF, DRIVE}
    assert set(_holders(credential_key_for_secret("GITHUB_TOKEN"), root)) == {SELF, DRIVE}
    assert set(_holders(credential_key_for_secret("MCP_GL_AUTHORIZATION"), root)) == {
        SELF,
        DRIVE,
    }
    # The document stores exactly the whole header value (scheme included).
    assert _store_value(root, "MCP_GL_AUTHORIZATION") == f"Bearer {SLACK_SENTINEL}"
    assert _servers(root)["gl"]["headers"] == {"Authorization": "${MCP_GL_AUTHORIZATION}"}
    assert SLACK_SENTINEL not in json.dumps(payload) and TOKEN_SENTINEL not in json.dumps(payload)


def test_an_oauth_server_says_brokered_and_grants_no_copy(
    root: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    network_store.save(_record(), root)
    _write_config(
        root,
        {
            "dd": {
                "type": "http",
                "url": "https://dd.test/mcp",
                "auth": {"type": "oauth"},
            }
        },
    )

    rc, payload = _run(capsys, "--repair")

    assert rc == 0, payload
    row = payload["servers"][0]
    assert row["shape"] == "oauth"
    assert row["note"] == "brokered; no copy"
    assert row["grants"] == []


def test_a_local_only_mark_suppresses_the_grant(
    root: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    network_store.save(_record(), root)
    _write_config(
        root,
        {"gh": {"type": "stdio", "command": "npx", "env": {"GITHUB_TOKEN": TOKEN_SENTINEL}}},
    )
    with sync_mod.mutate(NETWORK, root=root) as state:
        state.record_mark(credential_key_for_secret("GITHUB_TOKEN"), sync_mod.MARK_LOCAL_ONLY)

    rc, payload = _run(capsys, "--repair")

    assert rc == 0, payload
    assert _entry(credential_key_for_secret("GITHUB_TOKEN"), root) is None
    row = payload["servers"][0]
    assert row["grants"] == []
    assert [item["reason"] for item in row["skipped"]] == ["local_only"]


def test_one_key_refusal_does_not_stop_the_others(
    root: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    network_store.save(_record(), root)
    _write_config(
        root,
        {
            "gh": {
                "type": "stdio",
                "command": "npx",
                # GHOST is a pre-existing reference with no store row: the grant
                # loop refuses it and must still grant the adopted sibling.
                "env": {"GHOST": "${GHOST}", "GITHUB_TOKEN": TOKEN_SENTINEL},
            }
        },
    )

    rc, payload = _run(capsys, "--repair")

    assert rc == 0, payload
    row = payload["servers"][0]
    assert [item["key"] for item in row["grants"]] == ["secret:GITHUB_TOKEN"]
    reasons = {item["key"]: item["reason"] for item in row["skipped"]}
    assert reasons["secret:GHOST"] == "no_local_credential"
    assert _entry(credential_key_for_secret("GHOST"), root) is None
    assert set(_holders(credential_key_for_secret("GITHUB_TOKEN"), root)) == {SELF, DRIVE}


def test_a_device_with_no_network_record_grants_nothing(
    root: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    """Adoption still runs; the repair half is a no-op without a network."""
    _write_config(
        root,
        {"gh": {"type": "stdio", "command": "npx", "env": {"GITHUB_TOKEN": TOKEN_SENTINEL}}},
    )

    rc, payload = _run(capsys, "--repair")

    assert rc == 0, payload
    assert payload["servers"][0]["grants"] == []
    assert _servers(root)["gh"]["env"] == {"GITHUB_TOKEN": "${GITHUB_TOKEN}"}


def test_invite_prints_a_conflict_only_adoption_line(
    root: Path, capsys: pytest.CaptureFixture[str], monkeypatch: pytest.MonkeyPatch
) -> None:
    """A conflict means the value will NOT travel; a silent mint hides the decision."""
    from local_operator.secrets import access

    store = access.open_store(root, create=True)
    store.set("MCP_SLACK_AUTHORIZATION", b"a different value")
    _write_config(
        root,
        {
            "slack": {
                "type": "http",
                "url": "https://slack.test/mcp",
                "headers": {"Authorization": f"Bearer {SLACK_SENTINEL}"},
            }
        },
    )

    rc, out = _run_invite(root, capsys, monkeypatch, as_json=False)

    assert rc == 0, out
    assert "0 MCP credentials adopted into the encrypted store; 1 conflict" in out, out


def test_invite_json_carries_counts_only_when_there_are_some(
    root: Path, capsys: pytest.CaptureFixture[str], monkeypatch: pytest.MonkeyPatch
) -> None:
    from local_operator.secrets import access

    store = access.open_store(root, create=True)
    store.set("MCP_SLACK_AUTHORIZATION", b"a different value")
    _write_config(
        root,
        {
            "slack": {
                "type": "http",
                "url": "https://slack.test/mcp",
                "headers": {"Authorization": f"Bearer {SLACK_SENTINEL}"},
            }
        },
    )

    rc, out = _run_invite(root, capsys, monkeypatch, as_json=True)

    assert rc == 0, out
    assert json.loads(out)["mcp_credentials"] == {
        "adopted": 0,
        "reused": 0,
        "conflicts": 1,
    }

    _write_config(root, {})
    rc, out = _run_invite(root, capsys, monkeypatch, as_json=True)

    assert rc == 0, out
    assert "mcp_credentials" not in json.loads(out)


def test_an_oauth_shaped_server_with_refs_shows_the_grant_not_brokered(
    root: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    """OAuth labels the mcp:<url> path; it must not hide the copies beside it."""
    network_store.save(_record(), root)
    _seed_server_holders(root)
    _write_config(
        root,
        {
            "dd": {
                "type": "http",
                "url": URL,
                "auth": {"type": "oauth"},
                "headers": {"X-Api-Key": "literal-key-value"},
            }
        },
    )

    rc, payload = _run(capsys, "--repair")

    assert rc == 0, payload
    row = payload["servers"][0]
    assert row["shape"] == "oauth"
    assert row["note"] == ""
    assert [item["key"] for item in row["grants"]] == ["secret:MCP_DD_X_API_KEY"]
    assert set(row["grants"][0]["devices"]) == {SELF, DRIVE}

    rc, out = _run_human(capsys, "--repair")

    assert rc == 0, out
    assert "granted secret:MCP_DD_X_API_KEY" in out, out
    assert "brokered; no copy" not in out, out


def test_http_read_and_pool_holders_never_receive_a_class_two_copy(
    root: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    """The §1.3 gate applies to the HTTP/SSE branch too.

    A ``read`` holder was approved to borrow the brokered access token, not to
    receive a secret value; a pool member declares no credentials at all.
    """
    network_store.save(_record(), root)
    _seed_url_holders(root, [SELF, DRIVE, READER, POOL])
    _write_config(
        root,
        {
            "gl": {
                "type": "http",
                "url": URL,
                "headers": {"Authorization": f"Bearer {SLACK_SENTINEL}"},
            }
        },
    )

    rc, payload = _run(capsys, "--repair")

    assert rc == 0, payload
    item = payload["servers"][0]["grants"][0]
    assert set(item["devices"]) == {SELF, DRIVE}
    assert set(_holders(credential_key_for_secret("MCP_GL_AUTHORIZATION"), root)) == {
        SELF,
        DRIVE,
    }


def test_invite_prints_a_reused_only_adoption_line(
    root: Path, capsys: pytest.CaptureFixture[str], monkeypatch: pytest.MonkeyPatch
) -> None:
    """A reuse still rewrote the config to references; the mint must say so."""
    from local_operator.secrets import access

    store = access.open_store(root, create=True)
    store.set("GITHUB_TOKEN", b"REUSE_ME_42")
    _write_config(
        root,
        {"gh": {"type": "stdio", "command": "npx", "env": {"GITHUB_TOKEN": "REUSE_ME_42"}}},
    )

    rc, out = _run_invite(root, capsys, monkeypatch, as_json=False)

    assert rc == 0, out
    assert "0 MCP credentials adopted into the encrypted store, 1 reused; 0 conflicts" in out, out

    # The human run already rewrote the literal to a reference; restore it so the
    # JSON run exercises the same reused-only state rather than an empty one.
    _write_config(
        root,
        {"gh": {"type": "stdio", "command": "npx", "env": {"GITHUB_TOKEN": "REUSE_ME_42"}}},
    )

    rc, out = _run_invite(root, capsys, monkeypatch, as_json=True)

    assert rc == 0, out
    assert json.loads(out)["mcp_credentials"] == {"adopted": 0, "reused": 1, "conflicts": 0}


def test_an_oauth_server_with_only_refused_refs_shows_the_skip(
    root: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    """The brokered note must not hide a refused reference either."""
    network_store.save(_record(), root)
    _seed_server_holders(root)
    _write_config(
        root,
        {
            "dd": {
                "type": "http",
                "url": URL,
                "auth": {"type": "oauth"},
                "headers": {"X-Api-Key": "${MISSING_KEY}"},
            }
        },
    )

    rc, out = _run_human(capsys, "--repair")

    assert rc == 0, out
    assert "brokered; no copy" not in out, out
    # One row per targeted device (SELF and DRIVE are both mcp:<url> holders).
    assert "2 skipped" in out, out

    rc, payload = _run(capsys, "--repair")

    assert rc == 0, payload
    row = payload["servers"][0]
    assert row["note"] == ""
    assert [item["reason"] for item in row["skipped"]] == [
        "no_local_credential",
        "no_local_credential",
    ]


def test_an_http_server_without_a_brokered_key_grants_to_the_provisioning_population(
    root: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    """The header-only shape (slack): no mcp:<url> entry, yet the copy must land.

    ``_repair_targets`` falls back to the provisioning population (active
    drive/admin device members, pool excluded) instead of returning nobody.
    """
    network_store.save(_record(), root)
    _write_config(
        root,
        {
            "slack": {
                "type": "http",
                "url": "https://slack.test/mcp",
                "headers": {"Authorization": f"Bearer {SLACK_SENTINEL}"},
            }
        },
    )

    rc, payload = _run(capsys, "--repair")

    assert rc == 0, payload
    item = payload["servers"][0]["grants"][0]
    assert item["key"] == "secret:MCP_SLACK_AUTHORIZATION"
    assert set(item["devices"]) == {SELF, DRIVE}
    assert set(_holders(credential_key_for_secret("MCP_SLACK_AUTHORIZATION"), root)) == {
        SELF,
        DRIVE,
    }


def _seed_oauth_grant(root: Path, *, dead: bool = False) -> None:
    """Give THIS device a usable (or tombstoned) grant for ``URL``."""
    import asyncio

    from mcp.shared.auth import OAuthToken

    from local_operator.mcp.auth import McpTokenStorage

    storage = McpTokenStorage(URL)
    tokens = OAuthToken(
        **{
            "access_token": "oauth-access",
            "token_type": "Bearer",
            "expires_in": 3600,
            "refresh_token": "oauth-refresh",
        }
    )
    asyncio.run(storage.set_tokens(tokens))
    if dead:
        storage.mark_grant_dead(rejected_refresh_token="oauth-refresh")


def _mcp_grants(payload: dict[str, Any]) -> dict[str, set[str]]:
    return {
        str(item.get("key")): set(item.get("devices") or [])
        for row in payload["servers"]
        for item in row.get("grants") or []
        if str(item.get("key") or "").startswith("mcp:")
    }


def test_repair_grants_the_brokered_key_to_eligible_members(
    root: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    """A device approved before the login holds no ``mcp:<url>`` row.

    ``--repair`` grants the brokered key beside the references, to the SAME
    population: active device members with drive/admin. A read holder may borrow
    but is not copied to; a pool member declares no credentials.
    """
    from local_operator.network.credentials.types import credential_key_for_mcp

    _write_config(
        root,
        {
            "linear": {
                "type": "http",
                "url": URL,
                "auth": {"type": "oauth"},
            }
        },
    )
    network_store.save(_record(), root)
    _seed_oauth_grant(root)

    assert net_cli._cmd_mcp_credentials(  # noqa: SLF001
        Namespace(json=True, adopt=False, repair=True), root
    ) == 0
    payload = json.loads(capsys.readouterr().out)

    grants = _mcp_grants(payload)
    assert set(grants) == {credential_key_for_mcp(URL)}
    assert grants[credential_key_for_mcp(URL)] == {SELF, DRIVE}
    assert READER not in grants[credential_key_for_mcp(URL)]
    assert POOL not in grants[credential_key_for_mcp(URL)]


def test_repair_grants_a_dead_owner_grant_on_purpose(
    root: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    """cloudflare/minerva-qa: the owner's grant needs a re-login, not silence.

    Withholding the share would leave the holder with no diagnosis at all; with
    it, the borrow answers ``interactive_required`` and the status says "sign in
    again on <owner>" — the one remedy that can work.
    """
    from local_operator.network.credentials.types import credential_key_for_mcp

    _write_config(
        root,
        {"cloudflare": {"type": "http", "url": URL, "auth": {"type": "oauth"}}},
    )
    network_store.save(_record(), root)
    _seed_oauth_grant(root, dead=True)

    assert net_cli._cmd_mcp_credentials(  # noqa: SLF001
        Namespace(json=True, adopt=False, repair=True), root
    ) == 0
    payload = json.loads(capsys.readouterr().out)

    assert _mcp_grants(payload)[credential_key_for_mcp(URL)] == {SELF, DRIVE}


def _seed_peer_shares(root: Path, keys: list[str], *, owner_name: str = "macbook") -> None:
    """The peer OWNS these keys and this device holds a share of each."""
    with placement_mod.mutate(NETWORK, root=root, self_device=DRIVE) as document:
        for key in keys:
            document.declare(
                key,
                owner_device=DRIVE,
                owner_device_name=owner_name,
                provider="mcp-oauth",
                kind="store-secret",
                by=DRIVE,
            )
            document.grant(key, SELF, by=DRIVE)


def test_a_member_row_renders_the_status_refs_not_an_empty_list(
    root: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    """Real-node QA: a member with two refs set showed ``refs=[]``.

    The row must render the STATUS pass's reference list (which reads the same
    user-scope config), carry the per-ref ``set`` flags, and never emit the
    owner-side "no eligible member holds this server yet" note — this device is
    holding a share, not looking for someone to grant one to.
    """
    _write_config(
        root,
        {
            "notes": {
                "type": "stdio",
                "command": "/bin/echo",
                "env": {
                    "NOTES_TOKEN": "${NOTES_TOKEN}",
                    "NOTES_PROJECT": "${NOTES_PROJECT}",
                },
            }
        },
    )
    from local_operator.secrets import access

    store = access.open_store(root, create=True)
    store.set("NOTES_TOKEN", b"token-value")
    store.set("NOTES_PROJECT", b"project-value")
    network_store.save(_record(), root)
    _seed_peer_shares(
        root,
        ["secret:NOTES_TOKEN", "secret:NOTES_PROJECT"],
    )

    rc, payload = _run(capsys, "--repair")

    assert rc == 0
    row = payload["servers"][0]
    assert sorted(row["refs"]) == ["NOTES_PROJECT", "NOTES_TOKEN"]
    assert sorted(row["ref_states"], key=lambda item: item["id"]) == [
        {"id": "NOTES_PROJECT", "set": True},
        {"id": "NOTES_TOKEN", "set": True},
    ]
    assert row["state"] == "synced", row
    assert row["note"] == ""
    assert row.get("member") is not None


def test_a_member_row_names_the_owner_when_the_borrow_is_expired(
    root: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    """cloudflare/minerva-qa: "sign in again on <owner>", not "synced"."""
    from local_operator.network.credentials import mcp_state
    from local_operator.network.credentials.types import credential_key_for_mcp

    _write_config(
        root,
        {"cloudflare": {"type": "http", "url": URL, "auth": {"type": "oauth"}}},
    )
    network_store.save(_record(), root)
    mcp_state.record(
        NETWORK,
        credential_key_for_mcp(URL),
        state="expired",
        code="interactive_required",
        owner_device=DRIVE,
        owner_name="macbook",
        root=root,
    )

    rc, payload = _run(capsys, "--repair")

    assert rc == 0
    row = payload["servers"][0]
    assert row["state"] == "expired", row
    assert "macbook" in row["reason"], row["reason"]
    assert row["note"] == ""


def test_a_failed_member_validation_never_renders_synced(
    root: Path, capsys: pytest.CaptureFixture[str], monkeypatch: pytest.MonkeyPatch
) -> None:
    """cloud-node-1's blocker: a live holder row + a refused borrow read ``synced``.

    The status pass saw the share and said "borrows from damians-MacBook-Pro";
    the validation had just been REFUSED with ``interactive_required``. A failed
    grant is the newer fact and must win — the row says the owner must sign in
    again, never that the server is synced.
    """
    from local_operator.network.credentials.types import credential_key_for_mcp

    _write_config(
        root,
        {"cloudflare": {"type": "http", "url": URL, "auth": {"type": "oauth"}}},
    )
    network_store.save(_record(), root)
    _seed_peer_shares(root, [credential_key_for_mcp(URL)])
    monkeypatch.setattr(
        net_cli,
        "_member_repair_receipt",
        lambda *a, **k: {
            "cloudflare": {
                "pulled": [],
                "error": "",
                "grant": {
                    "ok": False,
                    "code": "interactive_required",
                    "owner": DRIVE,
                    "owner_name": "damians-MacBook-Pro",
                },
            }
        },
    )

    rc, payload = _run(capsys, "--repair")

    assert rc == 0
    row = payload["servers"][0]
    assert row["member"]["grant"]["ok"] is False
    assert row["state"] == "expired", row
    assert "damians-MacBook-Pro" in row["reason"], row["reason"]
    assert "sign in again" in row["reason"], row["reason"]


def test_a_configured_server_with_no_grant_is_never_unknown(
    root: Path, capsys: pytest.CaptureFixture[str], monkeypatch: pytest.MonkeyPatch
) -> None:
    """gitlab: no grant anywhere must read "needs operator login", not "unknown"."""
    _write_config(
        root,
        {"gitlab": {"type": "http", "url": URL, "auth": {"type": "oauth"}}},
    )
    network_store.save(_record(), root)

    rc, payload = _run(capsys, "--repair")
    assert rc == 0
    row = payload["servers"][0]
    assert row["state"] == "needs_operator_login", row
    assert row["reason"], "an empty reason is not an answer"

    # ...and the same when the status pass does not describe the name at all:
    # the row must still not fall back to a bare ``unknown``.
    monkeypatch.setattr(net_cli, "_credential_states", lambda *a, **k: {})
    rc, payload = _run(capsys, "--repair")
    assert rc == 0
    row = payload["servers"][0]
    assert row["state"] == "needs_operator_login", row
    assert row["reason"]


def test_a_tokenless_grant_never_reads_synced(
    root: Path, capsys: pytest.CaptureFixture[str], monkeypatch: pytest.MonkeyPatch
) -> None:
    """gitlab on the node: ``member.grant.ok`` was true with no token present.

    The row may keep a status's ``needs_operator_login`` (the precise answer for
    a server nothing has signed in to), but a live holder row must not survive a
    tokenless reply as ``synced``.
    """
    from local_operator.network.credentials.types import credential_key_for_mcp

    _write_config(
        root,
        {"gitlab": {"type": "http", "url": URL, "auth": {"type": "oauth"}}},
    )
    network_store.save(_record(), root)
    _seed_peer_shares(root, [credential_key_for_mcp(URL)])
    monkeypatch.setattr(
        net_cli,
        "_member_repair_receipt",
        lambda *a, **k: {
            "gitlab": {
                "pulled": [],
                "error": "",
                "grant": {"ok": False, "code": "no_token", "owner": DRIVE},
            }
        },
    )

    rc, payload = _run(capsys, "--repair")
    assert rc == 0
    row = payload["servers"][0]
    assert row["state"] != "synced", row
    assert row["state"] == "not_yet_synced", row
    assert "no token" in row["reason"], row["reason"]

    # A server NOTHING has signed in to keeps the status's precise answer: the
    # failed grant attempt must not downgrade "never signed in" to "not synced".
    monkeypatch.setattr(
        net_cli,
        "_credential_states",
        lambda *a, **k: {
            "gitlab": {
                "state": "needs_operator_login",
                "reason": "no sign-in found here or on a shared device",
                "owner": "",
                "shape": "oauth",
                "refs": [],
            }
        },
    )
    rc, payload = _run(capsys, "--repair")
    assert rc == 0
    row = payload["servers"][0]
    assert row["state"] == "needs_operator_login", row
    assert "no sign-in" in row["reason"]


def test_an_empty_refused_grant_keeps_the_more_precise_status(
    root: Path, capsys: pytest.CaptureFixture[str], monkeypatch: pytest.MonkeyPatch
) -> None:
    """gitlab: an empty grant must not flatten "never signed in" into "not synced"."""
    _write_config(
        root,
        {"gitlab": {"type": "http", "url": URL, "auth": {"type": "oauth"}}},
    )
    network_store.save(_record(), root)
    monkeypatch.setattr(
        net_cli,
        "_member_repair_receipt",
        lambda *a, **k: {
            "gitlab": {
                "pulled": [],
                "error": "",
                "grant": {"ok": False, "code": "refused", "owner": DRIVE},
            }
        },
    )
    monkeypatch.setattr(
        net_cli,
        "_credential_states",
        lambda *a, **k: {
            "gitlab": {
                "state": "needs_operator_login",
                "reason": "no sign-in found here or on a shared device",
                "owner": "",
                "shape": "oauth",
                "refs": [],
            }
        },
    )

    rc, payload = _run(capsys, "--repair")

    assert rc == 0
    row = payload["servers"][0]
    assert row["state"] == "needs_operator_login", row
    assert "no sign-in" in row["reason"]


def test_a_reference_server_with_no_status_row_still_repairs(
    root: Path, capsys: pytest.CaptureFixture[str], monkeypatch: pytest.MonkeyPatch
) -> None:
    """The non-OAuth fallback: it reads ``ref_names`` and must not crash.

    Review B3: the fallback referenced ``ref_names`` before its assignment, so a
    reference-only server with no status row killed ``--repair`` outright.
    """
    _write_config(
        root,
        {
            "notes": {
                "type": "stdio",
                "command": "/bin/echo",
                "env": {"GHOST_TOKEN": "${GHOST_TOKEN}"},
            }
        },
    )
    network_store.save(_record(), root)
    monkeypatch.setattr(net_cli, "_credential_states", lambda *a, **k: {})

    rc, payload = _run(capsys, "--repair")

    assert rc == 0, payload
    row = payload["servers"][0]
    assert row["refs"] == ["GHOST_TOKEN"]
    assert row["state"] == "missing_source", row
    assert "GHOST_TOKEN" in row["reason"]
