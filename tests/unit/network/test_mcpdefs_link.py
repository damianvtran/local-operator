"""MCP server definitions over two REAL relays on one host.

The unit file (``test_mcpdefs``) pins the content rules and the merge matrix.
THIS file pins the wire and the drive, the way ``test_readiness_link`` pins its
slice's: the shipped ``lop network mcp push|state`` parser against a real
relay-to-relay link, the rows that actually land in the PEER's ``mcp.json``,
and the old-peer path — which must degrade BY CAPABILITY (skip, naming the
remedy) and never by paying a slow-op deadline against a peer that does not
implement the op.

ONE PROCESS, TWO ROOTS, the constraints the other two-relay files record: the
CLI resolves its config dir from the AMBIENT environment, so each command runs
with the ambient root pointed at the device the operator would be on — the
pushing side, A. Every root here is the per-test ``root`` fixture's.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import pytest

from local_operator.network import cli as net_cli
from local_operator.network import definitions, mcpdefs, store, wire
from tests.unit.network.test_relay_e2e import (  # noqa: F401 — fixtures by import
    _pair,
    devices,
)

Devices = tuple[Any, Any, str, int]


def _parser() -> argparse.ArgumentParser:
    """The shipped ``lop network`` parser, built the way ``lop`` builds it."""
    parser = argparse.ArgumentParser(prog="lop")
    net_cli.add_parser(parser.add_subparsers(dest="command"))
    return parser


def _run(*argv: str) -> int:
    return net_cli.main(_parser().parse_args(["network", "mcp", *argv]))


def _run_json(capsys: pytest.CaptureFixture[str], *argv: str) -> tuple[int, dict[str, Any]]:
    rc = _run(*argv, "--json")
    payload = json.loads(capsys.readouterr().out)
    return rc, payload


def _write_servers(root: Path, servers: dict[str, Any]) -> None:
    root.mkdir(parents=True, exist_ok=True)
    (root / "mcp.json").write_text(
        json.dumps({"mcpServers": servers}, indent=2) + "\n", encoding="utf-8"
    )


def _peer_servers(server_b: Any) -> dict[str, Any]:
    path = server_b.root / "mcp.json"
    if not path.exists():
        return {}
    return json.loads(path.read_text(encoding="utf-8")).get("mcpServers", {})


def _bind_and_start(server_b: Any, record: Any) -> str:
    """Bind the peer's listener and point its OWN row at the real address."""
    host, port = server_b.bind()
    server_b.bind_control()
    server_b.start()
    live = f"{host}:{port}"
    with store.mutate(record.network_id, server_b.root) as copy:
        own = copy.member(server_b.identity.device_id)
        if own is not None:
            own.endpoints = [live]
            store.save(copy, server_b.root)
    return live


def _set_peer_endpoints(server_a: Any, record: Any, device_id: str, endpoints: list[str]) -> None:
    with store.mutate(record.network_id, server_a.root) as copy:
        row = copy.member(device_id)
        assert row is not None
        row.endpoints = list(endpoints)
        store.save(copy, server_a.root)


# ---------------------------------------------------------------------------
# Registered and advertised
# ---------------------------------------------------------------------------


def test_both_ends_serve_the_op_and_the_capability_is_advertised(
    request: pytest.FixtureRequest,
) -> None:
    pair_devices: Devices = request.getfixturevalue("devices")
    server_a, server_b, _host, _port = pair_devices
    for server in (server_a, server_b):
        assert "net_mcp_defs" in server._handlers  # noqa: SLF001 — the slice's seat
        assert "mcp_defs_sync" in server._local_slice_handlers  # noqa: SLF001
    assert wire.MCP_DEFS_V1 in wire.LINK_CAPABILITIES
    # The cadence rode the definitions syncer's step seam at construction.
    assert mcpdefs.mesh_tick_step in definitions._tick_steps()


# ---------------------------------------------------------------------------
# Push: rows land on the peer's own mcp.json
# ---------------------------------------------------------------------------


def test_push_lands_rows_on_the_peer_and_is_idempotent(
    request: pytest.FixtureRequest,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    pair_devices: Devices = request.getfixturevalue("devices")
    server_a, server_b, _host, _port = pair_devices
    record, _h, _p = _pair(pair_devices, monkeypatch)
    capsys.readouterr()
    live = _bind_and_start(server_b, record)
    _set_peer_endpoints(server_a, record, server_b.identity.device_id, [live])
    monkeypatch.setenv("LOCAL_OPERATOR_CONFIG_DIR", str(server_a.root))

    _write_servers(
        server_a.root,
        {
            "gl": {
                "type": "stdio",
                "command": "npx",
                "args": ["-y", "gitlab-mcp"],
                "env": {"GITLAB_TOKEN": "${GITLAB_TOKEN}", "PLAIN": "hunter2"},
            },
            "crm": {"type": "http", "url": "https://m.example/mcp"},
            # The shape table keeps this one off the wire entirely; the push
            # must SAY so on the side that can act on it.
            "leaky": {"type": "http", "url": "https://x.example/ghp_" + "a" * 36},
        },
    )

    rc, payload = _run_json(capsys, "push", "--peer", server_b.identity.name)
    assert rc == 0, payload
    assert payload["ok"] is True
    row = payload["peers"][0]
    assert {item["name"] for item in row["installed"]} == {"crm", "gl"}, row
    assert [item["name"] for item in row["withheld"]] == ["leaky"], row

    landed = _peer_servers(server_b)
    assert set(landed) == {"crm", "gl"}
    assert "leaky" not in (server_b.root / "mcp.json").read_text(encoding="utf-8")
    # The literal NEVER arrives; the reference does; the held value is a
    # placeholder keyed by its own name, so `lop secret set PLAIN ...` there
    # is the actionable remedy (state lists it).
    assert landed["gl"]["env"] == {
        "GITLAB_TOKEN": "${GITLAB_TOKEN}",
        "PLAIN": "${PLAIN}",
    }
    assert "hunter2" not in (server_b.root / "mcp.json").read_text(encoding="utf-8")

    # Idempotence over the wire: a second push finds the peer in sync — ONE
    # state round trip, nothing sent (a held-value row included, which the
    # origin-digest manifest is what makes possible), and no rewrite.
    before = (server_b.root / "mcp.json").read_bytes()
    rc2, payload2 = _run_json(capsys, "push", "--peer", server_b.identity.name)
    assert rc2 == 0, payload2
    row2 = payload2["peers"][0]
    assert row2["code"] == "in_sync", row2
    assert row2["pushed"] is False, row2
    assert (server_b.root / "mcp.json").read_bytes() == before

    # The apply is audited on the peer. The event lands with actor/outcome and
    # empty detail until audit.py's tables gain the row — the same convention
    # `definitions_applied` and the mobility slice's events follow.
    server_b.audit.flush()  # noqa: SLF001 — the writer batches; count after its own flush
    rows = [
        json.loads(line)
        for line in store.audit_path(server_b.root).read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    assert any(row.get("event") == "mcp_defs_applied" for row in rows), rows[-5:]
    server_b.stop()


def test_state_lists_the_servers_and_the_keys_a_mirror_needs(
    request: pytest.FixtureRequest,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    pair_devices: Devices = request.getfixturevalue("devices")
    server_a, _server_b, _host, _port = pair_devices
    monkeypatch.setenv("LOCAL_OPERATOR_CONFIG_DIR", str(server_a.root))
    _write_servers(
        server_a.root,
        {"gl": {"type": "stdio", "command": "npx", "env": {"GITLAB_TOKEN": "${GITLAB_TOKEN}"}}},
    )
    rc, payload = _run_json(capsys, "state")
    assert rc == 0, payload
    rows = {row["name"]: row for row in payload["servers"]}
    assert rows["gl"]["origin"] == ""
    assert {ref["id"]: ref["set"] for ref in rows["gl"]["refs"]} == {"GITLAB_TOKEN": False}

    # The human line carries the key to set — the actionable remedy.
    rc = _run("state")
    out = capsys.readouterr().out
    assert rc == 0
    assert "GITLAB_TOKEN" in out and "needs" in out


# ---------------------------------------------------------------------------
# The old peer: skipped BY CAPABILITY, never by timeout
# ---------------------------------------------------------------------------


def test_an_old_peer_is_refused_by_capability_without_ever_being_asked(
    request: pytest.FixtureRequest,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    """The measured trap: a request against a peer that never advertised the op
    pays a slow-op deadline and then reads as UNREACHABLE. The gate is asserted
    by CALL COUNT — the peer's handler was never entered — and by the refusal
    naming the remedy, which is what a person acts on.
    """
    pair_devices: Devices = request.getfixturevalue("devices")
    server_a, server_b, _host, _port = pair_devices
    # THE CAPABILITY IS PATCHED BEFORE THE PAIRING, because the link A checks
    # is the one the PAIRING advertised: a peer is "old" by what its handshake
    # said, and patching after the fact would leave every live link claiming
    # the feature. A itself also loses the advertisement, which this path does
    # not read.
    monkeypatch.setattr(
        wire,
        "LINK_CAPABILITIES",
        tuple(cap for cap in wire.LINK_CAPABILITIES if cap != wire.MCP_DEFS_V1),
    )
    record, _h, _p = _pair(pair_devices, monkeypatch)
    capsys.readouterr()
    monkeypatch.setenv("LOCAL_OPERATOR_CONFIG_DIR", str(server_a.root))
    live = _bind_and_start(server_b, record)
    _set_peer_endpoints(server_a, record, server_b.identity.device_id, [live])
    calls: list[int] = []
    original = server_b._handlers["net_mcp_defs"]  # noqa: SLF001 — the slice's seat
    server_b._handlers["net_mcp_defs"] = lambda link, frame: (  # noqa: SLF001
        calls.append(1),
        original(link, frame),
    )[1]
    _write_servers(server_a.root, {"gl": {"type": "stdio", "command": "npx", "env": {}}})

    rc, payload = _run_json(capsys, "push", "--peer", server_b.identity.name)
    assert rc == 1, payload
    assert calls == [], "the push asked a peer that never advertised the capability"
    row = payload["peers"][0]
    assert row["code"] == "peer_too_old", row
    assert "lop-update" in str(row["message"]), row
    assert not (server_b.root / "mcp.json").exists()
    server_b.stop()
