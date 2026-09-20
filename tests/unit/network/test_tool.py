"""The ``network`` agent tool: argv discipline, tiers, and what may not leave.

The tool's contract is mostly NEGATIVE — it must not complete a pairing, must not
pass a confirmation, and must not let a token or a key into a result — so the
tests here are about what the tool refuses to do as much as about what it
returns. The end-to-end cases run the real CLI in an isolated config directory:
a stubbed subprocess would prove the stub, and the JSON shapes are the contract
this tool parses (``mesh-ui.md`` §3.3).
"""

from __future__ import annotations

import argparse
import asyncio
import json
import stat
import sys
from pathlib import Path
from typing import Any

import pytest

from local_operator.harness.types import ToolContext
from local_operator.network import cli as net_cli
from local_operator.network import tool as net_tool
from local_operator.network.tool import NetworkParams
from local_operator.tools.registry import DEFAULT_TOOL_NAMES, TOOL_BUILDERS

pytestmark = pytest.mark.usefixtures("isolated_network_config")

_ALL_ACTIONS = (
    "status",
    "init",
    "invite",
    "join",
    "ls",
    "show",
    "peers",
    "member_rm",
    "disconnect",
    "panic",
    "log",
    "doctor",
)


@pytest.fixture()
def isolated_network_config(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    """A private config root for the CHILD CLI, and a PATH with no ``launchctl``.

    The tool hands the child its environment, so pointing
    ``LOCAL_OPERATOR_CONFIG_DIR`` here is what keeps a test from writing a device
    identity, a network record or an audit line into the operator's own store.
    ``PATH`` is narrowed to this interpreter's directory for the second reason:
    ``lop network init`` starts the relay under a LaunchAgent when there is
    launchd to do it with, and a test must not install a plist and load a job
    into the operator's real launchd domain. Without ``launchctl`` on PATH the
    child takes its own documented "no launchd here: run `lop network serve` in
    the foreground" branch, which is the same code path with the side effect
    removed.
    """
    root = tmp_path / "config"
    root.mkdir()
    monkeypatch.setenv("LOCAL_OPERATOR_CONFIG_DIR", str(root))
    monkeypatch.setenv("PATH", str(Path(sys.executable).parent))
    return root


def _call(action: str, **fields: Any) -> Any:
    args = {"action": action, **fields}
    return asyncio.run(net_tool.execute_network("call-1", args, context=ToolContext(cwd=".")))


def _text(result: Any) -> str:
    return "\n".join(part.text for part in result.content)


def _payload(result: Any) -> dict[str, Any]:
    return dict(result.details or {}).get("network") or {}


def _verbs() -> set[str]:
    """Every subcommand the real ``lop network`` parser registers."""
    parser = argparse.ArgumentParser(prog="lop")
    subparsers = parser.add_subparsers(dest="subcommand")
    net_cli.add_parser(subparsers)
    groups = [
        action for action in parser._actions if isinstance(getattr(action, "choices", None), dict)
    ]
    group = groups[0].choices["network"]
    leaves = [
        action for action in group._actions if isinstance(getattr(action, "choices", None), dict)
    ]
    return set(leaves[0].choices)


# ---------------------------------------------------------------------------
# Registration and tiers
# ---------------------------------------------------------------------------


def test_the_tool_is_registered_unconditionally_in_every_session() -> None:
    """``init`` is how the first network comes to exist, so the tool cannot be
    gated on one already existing (``mesh-ui.md`` §3.2)."""
    assert "network" in TOOL_BUILDERS
    assert "network" in DEFAULT_TOOL_NAMES
    tool = TOOL_BUILDERS["network"](ToolContext(cwd="."))
    assert tool is not None
    assert tool.name == "network"
    assert tool.parameters["properties"]["action"]["enum"] == list(_ALL_ACTIONS)


def test_reads_never_prompt_and_every_mutating_action_does() -> None:
    tool = TOOL_BUILDERS["network"](ToolContext(cwd="."))
    assert tool is not None
    assert tool.call_approval_tier is not None
    for action in _ALL_ACTIONS:
        expected = "read" if action in net_tool.READ_ACTIONS else "write"
        assert tool.call_approval_tier({"action": action}) == expected, action
    # The design's split, spelled out so a future action cannot land on the
    # wrong side of the gate silently.
    assert net_tool.READ_ACTIONS | net_tool.WRITE_ACTIONS == set(_ALL_ACTIONS)
    # Two racing epoch/trust mutations is a state nobody designed.
    assert tool.concurrency == "exclusive"


# ---------------------------------------------------------------------------
# Argv discipline — the flags this tool cannot pass
# ---------------------------------------------------------------------------


def test_every_action_maps_to_a_real_cli_verb() -> None:
    verbs = _verbs()
    samples = {
        "status": {},
        "ls": {},
        "show": {"network": "devmesh"},
        "peers": {},
        "doctor": {},
        "log": {"since": "15m"},
        "init": {"network": "devmesh"},
        "invite": {"role": "drive"},
        "join": {"token": "@token.invite"},
        "member_rm": {"network": "devmesh", "device": "d_" + "a" * 32},
        "disconnect": {"network": "devmesh"},
        "panic": {"network": "devmesh"},
    }
    assert set(samples) == set(_ALL_ACTIONS)
    for action, fields in samples.items():
        argv, problem = net_tool._argv_for(NetworkParams(action=action, **fields))
        assert problem == "", (action, problem)
        assert argv[0] == "network" and argv[-1] == "--json", argv
        assert argv[1] in verbs, (action, argv, verbs)


def test_the_argv_carries_no_confirmation_and_no_token_printing_flag() -> None:
    """R17's controls and R3's pairing both need a human. There is no flag that
    finishes either, and a future one must not be reachable from here."""
    forbidden = {"--yes", "-y", "--confirm", "--force", "--print", "--sas-stdin", "--purge"}
    samples = [
        NetworkParams(action="invite", role="admin", network="devmesh"),
        NetworkParams(action="join", token="@/tmp/x.invite"),
        NetworkParams(action="member_rm", network="devmesh", device="d_x"),
        NetworkParams(action="disconnect", network="devmesh"),
        NetworkParams(action="panic", network="devmesh"),
        NetworkParams(action="init", network="devmesh"),
    ]
    for params in samples:
        argv, problem = net_tool._argv_for(params)
        assert problem == ""
        assert not (set(argv) & forbidden), argv


def test_a_missing_argument_is_refused_with_a_sentence_not_a_call() -> None:
    result = _call("show")
    assert result.is_error
    assert "needs 'network'" in _text(result)
    result = _call("member_rm", network="devmesh")
    assert result.is_error
    assert "needs 'network' and 'device'" in _text(result)
    result = _call("join")
    assert result.is_error
    assert "needs 'token'" in _text(result)


@pytest.mark.parametrize(
    ("payload", "survivor"),
    [
        ({"control_key": "deadbeef", "ok": True, "healthy": True}, "healthy"),
        ({"secret": "s3cr3t", "nested": {"material": "m", "keep": 1}}, "keep"),
        ({"peers": [{"token": "t", "device_id": "d_1", "public_key": "p"}]}, "d_1"),
    ],
)
def test_secret_shaped_keys_are_dropped_at_every_depth(
    payload: dict[str, Any], survivor: str
) -> None:
    """A tool result is the most-copied text in the system; the scrub is the
    second line of defence behind "the CLI does not print keys" — and it must
    drop the secret-shaped keys WITHOUT flattening the rest of the payload."""
    scrubbed = net_tool._scrub(payload)
    rendered = json.dumps(scrubbed)
    for marker in ("deadbeef", "s3cr3t", "material", "'m'", '"token"', "control_key"):
        assert marker not in rendered
    assert survivor in rendered
    assert scrubbed


def test_a_pasted_token_goes_to_a_private_file_and_does_not_survive_the_call(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """``ps`` reads argv. The CLI accepts ``@path``, so a token handed to the
    tool as TEXT never becomes an argv entry, and the file is gone afterwards."""
    seen: dict[str, Any] = {}

    async def fake_run(argv: list[str], timeout: float) -> tuple[int, str, str]:
        token_arg = argv[2]
        path = Path(token_arg[1:])
        seen["mode"] = stat.S_IMODE(path.stat().st_mode)
        seen["content"] = path.read_text(encoding="utf-8")
        return 0, json.dumps({"ok": True, "name": "devmesh", "network_id": "n_1"}), ""

    monkeypatch.setattr(net_tool, "_run_cli", fake_run)
    result = _call("join", token="T0KEN-material")

    assert not result.is_error
    assert seen["mode"] == 0o600
    assert seen["content"] == "T0KEN-material"
    assert "T0KEN-material" not in _text(result)
    assert "T0KEN-material" not in json.dumps(result.details or {}, default=str)


# ---------------------------------------------------------------------------
# The real CLI, end to end, in an isolated store
# ---------------------------------------------------------------------------


def test_a_read_on_an_empty_store_reports_rather_than_raises() -> None:
    result = _call("ls")
    assert not result.is_error
    assert "no networks on this device" in _text(result)
    # `peers` is the one read that exits non-zero without a relay, and the
    # sentence is the point: an empty list is not "no peers configured".
    peers = _call("peers")
    assert peers.is_error
    assert "relay is not running" in _text(peers)


def test_init_show_status_and_doctor_round_trip_against_the_real_cli() -> None:
    created = _call("init", network="devmesh")
    assert not created.is_error, _text(created)
    payload = _payload(created)
    assert payload["name"] == "devmesh"
    assert payload["network_id"].startswith("n_")
    # The CLI's own autostart decision, not one this tool made: it either reports
    # a running relay or the sentence a machine without launchd gets.
    assert isinstance(payload["relay"], str)

    listed = _call("ls")
    assert "devmesh" in _text(listed)
    assert _payload(listed)["networks"][0]["members"] == 1

    shown = _call("show", network="devmesh")
    text = _text(shown)
    assert "devmesh" in text and "admin" in text
    assert "list" in text and "view" in text  # the capability vocabulary, not prose

    status = _call("status")
    assert _payload(status)["identity_present"] is True
    assert "devmesh" in _text(status)

    doctor = _call("doctor")
    assert not doctor.is_error
    assert "identity" in _text(doctor)

    log = _call("log", since="1h")
    assert "member_admitted" in _text(log)


def test_the_invite_result_carries_the_path_and_never_the_token() -> None:
    _call("init", network="devmesh")
    invited = _call("invite", role="drive")
    assert not invited.is_error, _text(invited)

    payload = _payload(invited)
    token_path = Path(payload["path"])
    assert token_path.is_file()
    token = token_path.read_text(encoding="utf-8").strip()
    assert payload["role"] == "drive"
    assert payload["expires_in_s"] == 600.0

    rendered = _text(invited) + json.dumps(invited.details or {}, default=str)
    assert token not in rendered
    assert "hand that FILE to the other device" in _text(invited)


def test_join_cannot_be_completed_by_the_tool_and_says_why() -> None:
    """The CLI has no ``--confirm``: the code is read at the joining device's own
    prompt. So the tool reports the refusal and hands the step to the user."""
    _call("init", network="devmesh")
    invited = _call("invite", role="drive")
    token_path = Path(_payload(invited)["path"])

    joined = _call("join", token=f"@{token_path}")
    assert joined.is_error
    text = _text(joined)
    assert "no endpoint" in text
    assert "needs a person at a terminal" in text
    assert "No flag completes this for them" in text


def test_a_refusal_from_the_cli_is_a_sentence_not_a_traceback() -> None:
    result = _call("show", network="nosuchnet")
    assert result.is_error
    assert "not in a network called" in _text(result)
    assert "Traceback" not in _text(result)


def test_status_never_surfaces_the_relay_control_record() -> None:
    """``lop network status --json`` includes the local relay record, which
    carries ``control_key``. No field of that record may reach a result."""
    _call("init", network="devmesh")
    result = _call("status")
    rendered = _text(result) + json.dumps(result.details or {}, default=str)
    assert "control_key" not in rendered
    assert "control_port" not in rendered


def test_the_agent_digest_says_what_a_member_count_rests_on() -> None:
    """The agent's own surface must carry the same caveat the CLI does.

    Round 3's blocker was a count presented as authoritative when it had not been
    checked (Q-R2-1). An operator reads `lop network ls`; an agent reads THIS tool's
    digest of the same JSON, so a marker that only the CLI printed would leave the
    agent acting on an unchecked subset. Both call one owner
    (``relay.membership_marker``) for that reason.
    """
    verified = net_tool._render(
        "ls",
        {
            "networks": [
                {
                    "name": "devmesh",
                    "network_id": "n_" + "a" * 22,
                    "epoch": 1,
                    "role": "admin",
                    "members": 3,
                    "trust": "active",
                    "membership": {
                        "table": {"answered": ["d_1"], "not_answered": [], "complete": True}
                    },
                }
            ]
        },
    )
    assert any("[members verified with all 1 peer(s)]" in line for line in verified), verified

    partial = net_tool._render(
        "ls",
        {
            "networks": [
                {
                    "name": "devmesh",
                    "network_id": "n_" + "a" * 22,
                    "epoch": 1,
                    "role": "admin",
                    "members": 3,
                    "trust": "active",
                    "membership": {
                        "table": {
                            "answered": ["d_1"],
                            "not_answered": [{"device_id": "d_2", "reason": "no_live_link"}],
                            "complete": False,
                        }
                    },
                }
            ]
        },
    )
    assert any("verified with 1 of 2 peer(s)" in line for line in partial), partial
    assert any("verified with all" not in line for line in partial), partial

    # A row that reached the tool WITHOUT a refresh (no relay answering) says so too.
    unread = net_tool._render(
        "ls",
        {
            "networks": [
                {
                    "name": "devmesh",
                    "network_id": "n_" + "a" * 22,
                    "epoch": 1,
                    "role": "admin",
                    "members": 4,
                    "trust": "active",
                    "membership": {
                        "table": {"answered": [], "not_answered": [], "complete": False}
                    },
                }
            ]
        },
    )
    assert any("NOT verified" in line for line in unread), unread


def test_the_agent_digest_carries_a_removed_devices_own_standing() -> None:
    """`show` on a removed device: the sentence, not a healthy-looking member list."""
    lines = net_tool._render(
        "show",
        {
            "name": "devmesh",
            "network_id": "n_" + "a" * 22,
            "epoch": 2,
            "role": "admin",
            "trust": "active",
            "members_detail": [{"device_id": "d_1", "name": "laptop", "active": True}],
            "membership": {
                "state": "removed",
                "sentence": "this device is no longer a member of devmesh (removed by d_9)",
                "remedies": ["re-pair as a new device"],
            },
        },
    )
    body = "\n".join(lines)
    assert "no longer a member of devmesh (removed by d_9)" in body
    assert "re-pair as a new device" in body
