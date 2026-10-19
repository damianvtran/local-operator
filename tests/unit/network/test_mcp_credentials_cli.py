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


def _seed_server_holders(root: Path) -> None:
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
        document.grant(key, SELF, by=SELF)
        document.grant(key, DRIVE, by=SELF)


def _run(capsys: pytest.CaptureFixture[str], *argv: str) -> tuple[int, dict[str, Any]]:
    parser = argparse.ArgumentParser(prog="lop")
    net_cli.add_parser(parser.add_subparsers(dest="command"))
    args = parser.parse_args(["network", "mcp", "credentials", *argv, "--json"])
    rc = net_cli.main(args)
    return rc, json.loads(capsys.readouterr().out)


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
