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
from tests.unit.network import conftest as net_fixtures

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
    "sessions",
    "trust",
    "credentials",
    "definitions_state",
)

#: One plausible call per action, in the order the enum declares them. Defined once
#: because TWO properties are asserted over every action — that it reaches a real CLI
#: verb, and that it can never spell a confirmation flag — and a second table would be
#: the place one of them quietly stopped covering an action.
_SAMPLES: dict[str, dict[str, Any]] = {
    "status": {},
    "init": {"network": "devmesh"},
    "invite": {"role": "drive"},
    "join": {"token": "@token.invite"},
    "ls": {},
    "show": {"network": "devmesh"},
    "peers": {},
    "member_rm": {"network": "devmesh", "device": "d_" + "a" * 32},
    "disconnect": {"network": "devmesh"},
    "panic": {"network": "devmesh"},
    "log": {"since": "15m"},
    "doctor": {},
    "sessions": {"peer": "device-b"},
    "trust": {"network": "devmesh"},
    "credentials": {},
    "definitions_state": {},
}


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


async def _run_call(args: dict[str, Any]) -> Any:
    """Await the guarded tool from inside a coroutine.

    The guarded tool's published type is ``Callable[..., Awaitable[ToolResult]]``
    rather than a coroutine function, so ``asyncio.run`` cannot take its call
    directly; awaiting it is exactly what the harness does with it.
    """
    return await net_tool.execute_network("call-1", args, None, None, ToolContext(cwd="."))


def _call(action: str, **fields: Any) -> Any:
    args = {"action": action, **fields}
    return asyncio.run(_run_call(args))


def _text(result: Any) -> str:
    return "\n".join(part.text for part in result.content)


def _payload(result: Any) -> dict[str, Any]:
    return dict(result.details or {}).get("network") or {}


def _verbs() -> set[str]:
    """Every subcommand the real ``lop network`` parser registers."""
    parser = argparse.ArgumentParser(prog="lop")
    subparsers = parser.add_subparsers(dest="subcommand")
    net_cli.add_parser(subparsers)
    group = net_fixtures.subcommands_of(parser)["network"]
    return set(net_fixtures.subcommands_of(group))


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


def test_the_sessions_action_is_tiered_by_what_it_does_not_by_its_name() -> None:
    """One action, both sides of the gate: a listing is silent, a stop is not.

    Sabotage check for why ``call_approval_tier`` takes the arguments: a tier decided by
    the action name alone would have to either prompt for every listing or stay silent
    for a stop that ends somebody's session on another device.
    """
    tool = TOOL_BUILDERS["network"](ToolContext(cwd="."))
    assert tool is not None and tool.call_approval_tier is not None
    tier = tool.call_approval_tier
    assert tier({"action": "sessions", "peer": "device-b"}) == "read"
    assert tier({"action": "sessions", "all_peers": True}) == "read"
    for verb in ("create", "engage", "stop", "delete"):
        assert tier({"action": "sessions", "peer": "device-b", verb: "s1"}) == "write", verb
    # A model that sends the boolean as text is not a read either.
    assert tier({"action": "sessions", "peer": "device-b", "create": "true"}) == "write"
    assert tier({"action": "sessions", "peer": "device-b", "create": "false"}) == "read"


# ---------------------------------------------------------------------------
# Argv discipline — the flags this tool cannot pass
# ---------------------------------------------------------------------------


def test_every_action_maps_to_a_real_cli_verb() -> None:
    verbs = _verbs()
    # Keyed by the tool's own action vocabulary (``NetworkParams.action``'s alias)
    # rather than by ``str``: the mapping is asserted against ``_ALL_ACTIONS`` below,
    # and typing it as ``str`` is what let a typo pass the checker and fail the model.
    samples: dict[net_tool.NetworkAction, dict[str, Any]] = dict(_SAMPLES)  # type: ignore[arg-type]
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
        # The four verbs that COULD spell one if the tool were allowed to: a
        # delete is a dry run because ``--yes`` is absent, and a stop never
        # escalates because ``--force`` is.
        NetworkParams(action="sessions", peer="device-b", delete="s1"),
        NetworkParams(action="sessions", peer="device-b", stop="s1"),
        NetworkParams(action="sessions", peer="device-b", create=True, prompt="hi"),
        NetworkParams(action="trust", network="devmesh", trust_state="untrusted"),
    ]
    for params in samples:
        argv, problem = net_tool._argv_for(params)
        assert problem == ""
        assert not (set(argv) & forbidden), argv


def test_the_tool_never_supplies_the_confirmation_code_itself() -> None:
    """R3's pin: the tool may START a pairing and may never finish one by itself.

    The property is not "no ``--confirm`` anywhere" — the flag has to exist, or an
    agent could not hand the user's half back. It is that the flag is reached by
    exactly ONE argv, carrying exactly the caller's own value, and that NO combination
    of the tool's fields reaches it otherwise. A refactor that defaulted ``confirm`` to
    the code the park returned, or derived the flag from the parked record, would pass
    every other test in this file and fail here.
    """
    for action in _ALL_ACTIONS:
        argv, problem = net_tool._argv_for(NetworkParams(action=action, **_SAMPLES[action]))
        assert problem == "", (action, problem)
        assert "--confirm" not in argv, (action, argv)
    # The parked code is not a confirmation, even when a caller echoes it through every
    # other field of the same call: only 'confirm' produces the flag, and only with its
    # own value.
    argv, problem = net_tool._argv_for(NetworkParams(action="join", token="tok", device="481926"))
    assert problem == ""
    assert "--confirm" not in argv and "--park" in argv, argv
    # And the one branch that carries it carries it once, verbatim.
    argv, problem = net_tool._argv_for(NetworkParams(action="join", confirm="481 926"))
    assert problem == ""
    assert argv.count("--confirm") == 1
    assert argv[argv.index("--confirm") + 1] == "481 926"
    assert "--park" not in argv


def test_the_description_teaches_the_two_phase_pair_and_drops_the_old_claim() -> None:
    """The description is the FIRST thing an agent reads, so it is the place a
    capability that does not exist costs the most: it used to say sessions on other
    devices were not reachable, which the ``sessions`` action now contradicts."""
    tool = TOOL_BUILDERS["network"](ToolContext(cwd="."))
    assert tool is not None
    text = tool.description
    assert "Read and drive a lop mesh network from this device" in text
    assert "creating a session on a peer" in text
    assert "not reachable" not in text
    assert "Pairing and incident controls need a human" in text
    assert "reports what the CLI refused and why" in text


def test_the_invite_action_carries_the_options_the_guide_teaches() -> None:
    """Guide and tool must not disagree about what is possible.

    The guide tells an agent to reach for ``--expires`` and ``--device``; before this,
    the tool had no field for either, so an instruction the guide gives resolved to a
    call that could not carry it.
    """
    argv, problem = net_tool._argv_for(
        NetworkParams(action="invite", role="drive", expires="30m", device="d_abc")
    )
    assert problem == ""
    assert "--expires" in argv and argv[argv.index("--expires") + 1] == "30m"
    assert "--device" in argv and argv[argv.index("--device") + 1] == "d_abc"


def test_the_new_actions_render_what_the_cli_actually_emits() -> None:
    """The digests for ``sessions``/``trust``/``credentials``/``definitions_state``.

    These payloads are the CLI's own ``--json`` shapes, so a render that reads a key
    nobody sends (or prints a raw wire token where the family has a gloss) would only
    show up against a live peer — which is exactly what a unit test of the renderer is
    for. The glossing is asserted, not assumed: ``connect_failed`` is the relay's
    word, and a person reads the sentence.
    """
    listing = {
        "ok": True,
        "sessions": [
            {
                "session_id": "s_1",
                "state": "stored",
                "conversation_name": "Field notes",
                "peer": {"device_id": "d_1", "name": "mbp"},
            }
        ],
        "peers": {
            "d_1": {"name": "mbp", "reachable": True},
            "d_2": {"name": "old-box", "reachable": False, "reason": "connect_failed"},
        },
    }
    lines = net_tool._render("sessions", listing)  # noqa: SLF001 — the renderer under test
    assert any("s_1" in line and "Field notes" in line and "on mbp" in line for line in lines)
    assert any(line.startswith("old-box: did not answer") for line in lines), lines
    assert not any("connect_failed" in line for line in lines), lines

    empty = net_tool._render("sessions", {"ok": True, "sessions": [], "peers": {}})  # noqa: SLF001
    assert empty == ["no sessions are held by other devices right now"]

    # A MUTATION's receipt is the owner's own sentence, never a re-rendering of it.
    assert net_tool._render(
        "sessions", {"ok": True, "detail": "runtime joining"}
    ) == [  # noqa: SLF001
        "runtime joining"
    ]

    assert net_tool._render(  # noqa: SLF001
        "trust",
        {"ok": True, "network_id": "n_1", "trust": "active", "applied_locally": True},
    ) == ["n_1 is now active", "the relay is not running on this device: applied locally"]

    credentials = {
        "ok": True,
        "networks": [
            {
                "network": "home-net",
                "credentials": [
                    {
                        "key": "OPENAI_API_KEY",
                        "kind": "api_key",
                        "owned_here": True,
                        "owner_device": "d_1",
                        "owner_device_name": "mbp",
                    }
                ],
            }
        ],
    }
    assert net_tool._render("credentials", credentials) == [  # noqa: SLF001
        "home-net:",
        "  OPENAI_API_KEY  api_key  owner: this device",
    ]
    assert net_tool._render("credentials", {"ok": True, "networks": []}) == [  # noqa: SLF001
        "nothing is shared with or by this device"
    ]

    assert net_tool._render(  # noqa: SLF001
        "definitions_state",
        {
            "ok": True,
            "agents": {"scout": {}},
            "teams": {"pod": {}},
            "mirrored": {"agents": {"scout": "d_1"}, "teams": {}},
        },
    ) == ["agent: scout (mirrored from d_1)", "team: pod (yours)"]


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
    tool as TEXT never becomes an argv entry, and the file is gone afterwards.

    The parked pair is the path a token join now takes, so THIS is the spawn it is
    sent through: stubbing ``_run_cli`` alone left the real CLI dialling with the
    fake token, which is how this test noticed the change (an ``invite_invalid`` from
    the wire, not an assertion about argv).
    """
    seen: dict[str, Any] = {}

    async def fake_park(argv: list[str]) -> tuple[int, str, str]:
        token_arg = argv[2]
        path = Path(token_arg[1:])
        seen["path"] = path
        seen["mode"] = stat.S_IMODE(path.stat().st_mode)
        seen["content"] = path.read_text(encoding="utf-8")
        return (
            0,
            json.dumps(
                {
                    "ok": True,
                    "status": "awaiting_confirmation",
                    "sas": "481926",
                    "fingerprint": "K7QM-3XPD-4B1N-9T2B",
                    "name": "devmesh",
                    "network_id": "n_1",
                    "seconds_left": 180.0,
                    "sentence": "Ask the user to read back the code 481 926.",
                }
            ),
            "",
        )

    monkeypatch.setattr(net_tool, "_start_parked_join", fake_park)
    result = _call("join", token="T0KEN-material")

    assert not result.is_error
    assert seen["mode"] == 0o600
    assert seen["content"] == "T0KEN-material"
    # The token file is the tool's OWN temporary: it must not outlive the call, or the
    # credential sits in a private directory until something else cleans it up.
    assert not Path(str(seen["path"])).exists()
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


def test_a_join_from_the_tool_parks_and_never_supplies_the_code() -> None:
    """The tool STARTS a pairing; it does not finish one.

    Before the two-phase pair this asserted the opposite shape — a refusal that told
    the agent to hand the step to a human at a terminal — and the sentence it checked
    is gone with the behaviour it described. What is left to assert is the property
    R3 asks for: the tool's own call cannot produce the transcription, so the code
    still has to come from a person, and the only thing that resolves a pairing is a
    SECOND call carrying their value.
    """
    _call("init", network="devmesh")
    invited = _call("invite", role="drive")
    token_path = Path(_payload(invited)["path"])

    joined = _call("join", token=f"@{token_path}")
    assert joined.is_error
    text = _text(joined)
    # WHICH FIRST CLAUSE APPEARS IS AN ENVIRONMENT FACT, not this test's subject: a
    # token minted where nothing is advertised is refused locally with "no endpoint",
    # while a token naming a detected address gets the dial's own refusal instead
    # ("nothing was listening at …"). Asserting only the first made this test pass on
    # the author's machine and fail on CI, where `init` advertised the runner's own
    # address — so both the local refusal and the dial refusal are accepted here, and
    # the teeth are the sentences below, which are the same either way.
    assert "no endpoint" in text or "nothing was listening at" in text, text
    assert "Pairing needs a person" in text
    assert "never the one this tool printed back" in text


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


def test_the_agent_peer_digest_reads_a_reason_the_way_a_person_does() -> None:
    """Round 11's MAJOR (R11-0) and QA round 25's Q-R25-1 — the SAME leak, found twice.

    ``_render("peers", …)`` printed the row's raw ``reason`` — a stage word, two
    endpoint addresses and a Python class name — beside the 34-character device id,
    while the ``doctor`` branch of the SAME function had just been routed through its
    own gloss for exactly that reason, and the test above says what the argument is.
    Both rounds filed it independently, which is the point of the guard in
    ``test_reason_surfaces.py``: the surfaces were swept one at a time and this one was
    always the one nobody had looked at yet.

    The gloss is the SHARED member table (the same function `lop network peers` reads,
    so one peer is described in one voice), the 34-character id is replaced by the name
    every other surface addresses a peer by, and the raw reason and id stay in the row
    the tool puts in ``details`` — this renderer does not consume them.
    """
    import local_operator.resume as resume

    payload: dict[str, Any] = {
        "peers": [
            {
                "reachable": False,
                "device_id": "d_" + "1" * 32,
                "name": "device-b",
                "reason": (
                    "unreachable: 127.0.0.1:0 connect_failed:OSError; "
                    "127.0.0.1:39223 connect_failed:ConnectionRefusedError"
                ),
            },
            {
                "reachable": False,
                "device_id": "d_" + "2" * 32,
                "name": "device-c",
                "reason": "handshake_refused:TimeoutError",
            },
            {"reachable": True, "device_id": "d_" + "3" * 32, "name": "device-d", "reason": ""},
            {"reachable": False, "device_id": "d_" + "4" * 32, "name": "", "reason": "no_endpoint"},
        ]
    }
    lines = net_tool._render("peers", payload)
    body = "\n".join(lines)
    for peer in payload["peers"]:
        assert peer["device_id"] not in body, body
    for leaked in (
        "connect_failed",
        "handshake_refused",
        "unreachable:",
        "ConnectionRefusedError",
        "OSError",
        "TimeoutError",
        "127.0.0.1",
        "no_endpoint",
    ):
        assert leaked not in body, (leaked, body)
    assert lines == [
        "unreachable device-b  no address of it answered",
        "unreachable device-c  the link was refused",
        "reachable   device-d",
        f"unreachable {resume.UNNAMED_DEVICE}  no address published for it",
    ], lines
    # The machine register is untouched: the caller keeps the reason and the id it
    # passed, which is what ``details`` carries into the agent's result.
    assert payload["peers"][0]["reason"].startswith("unreachable: 127.0.0.1:0")
    assert payload["peers"][0]["device_id"] not in body


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


def test_the_agent_digest_of_a_doctor_run_reads_in_words() -> None:
    """QA round 24, Q-R24-2, at the agent's own surface.

    This tool's ``doctor`` digest is a human surface too — the model reads it the way
    a person reads ``lop network doctor`` — and it rendered ``checks[].detail``
    verbatim, so the same stage words, Python class names and endpoint address reached
    it. It goes through ``resume.doctor_detail_words`` now; the raw strings are
    unchanged in the payload this digest is rendered FROM, which is what the diff's
    ``--json`` half of the finding is about.
    """
    checks = [
        {
            "check": "reachability",
            "ok": False,
            "device_id": "d_" + "b" * 32,
            "endpoint": "127.0.0.1:64996",
            "detail": "connect_failed:ConnectionRefusedError",
        },
        {
            "check": "handshake",
            "ok": False,
            "device_id": "d_" + "1" * 32,
            "endpoint": "127.0.0.1:64994",
            "detail": "not_attempted: 127.0.0.1:64994 answered and the doctor budget "
            "ran out before the handshake",
        },
    ]
    lines = net_tool._render(  # noqa: SLF001 — the renderer under test
        "doctor", {"ok": False, "identity_present": True, "checks": checks}
    )
    body = "\n".join(lines)
    for token in ("connect_failed", "not_attempted", "ConnectionRefusedError"):
        assert token not in body, (token, body)
    assert "nothing answered at that address" in body
    assert "it answered, and the doctor ran out of time before the handshake" in body
    # The row's own address is kept once, in its own column; the sentence that
    # repeated it does not.
    assert body.count("127.0.0.1:64994") == 1, body


def test_the_agent_digest_carries_the_audit_state_at_its_own_column() -> None:
    """D40 on the surface a "why is my session stuck" question actually arrives on.

    The digest is the model's only view: the audit fields ride in ``details``, which
    never reaches a provider, so a digest that omitted them left an agent unable to
    distinguish a row that is recorded-but-unpublished from one that does not exist —
    the same blindness the CLI and the panel had. Same words as those two, through the
    one renderer (``relay.audit_status_words``), at this register's own column: every
    value here starts at cell 11 (``installed: ``, ``relay:     ``, ``log:       ``).

    The payload is staged because the transport is not the subject — the numbers are
    the relay's own in production, and the CLI cells drive a real writer's.
    """
    payload = {
        "installed": True,
        "supported": True,
        "identity_present": True,
        "relay_running": True,
        "relay_answering": True,
        "relay": {
            "pid": 4711,
            "audit_recorded_through": 13,
            "audit_published_through": 12,
            "audit_degraded": False,
            "audit_degraded_reason": "",
        },
        "log": "/tmp/network.log",
        "networks": [],
    }
    lines = net_tool._render("status", payload)  # noqa: SLF001 — the renderer under test
    audit_line = next(line for line in lines if line.startswith("audit:"))
    assert audit_line == ("audit:     13 recorded, published through 12 (1 not yet written)"), lines

    # A WEDGED RELAY SPEAKS: never an omission, because the omission is the bug. And a
    # stale relay with no counters says nothing at all rather than zero (absent, not 0).
    wedged = dict(payload, relay=None, relay_answering=False, relay_state="wedged")
    spoken = net_tool._render("status", wedged)  # noqa: SLF001
    assert any(
        line.startswith("audit:") and "audit.jsonl holds the last state" in line for line in spoken
    ), spoken
    stale = dict(payload, relay={"pid": 4711})
    assert not [line for line in net_tool._render("status", stale) if line.startswith("audit:")]


def test_the_agent_digest_does_not_call_a_wedged_relay_not_running() -> None:
    """A relay that is up and silent is NOT "not running", on the model's surface too.

    The registry knows three states and this line read only one of them: the pid lives
    in the relay block, that block is ``None`` whenever the control socket does not
    answer, and the line therefore said ``not running`` about a process that is running
    — the Q-R3-4 contradiction the CLI's block was fixed for, left on the one surface a
    model reads. It matters more now than it did: the audit line directly below says the
    relay is not answering, so a digest whose own relay line disagreed with it would
    have told the model two different things about one process in two adjacent rows,
    and the model cannot look at ``details`` to settle it.

    ``record`` is what carries the pid when the live answer is missing: it is the
    on-disk record of a process that IS there.
    """
    wedged = {
        "installed": True,
        "supported": True,
        "identity_present": True,
        "relay_running": True,
        "relay_answering": False,
        "relay_state": "wedged",
        "relay": None,
        "record": {"pid": 4711},
        "log": "/tmp/network.log",
        "networks": [],
    }
    lines = net_tool._render("status", wedged)  # noqa: SLF001 — the renderer under test
    relay_line = next(line for line in lines if line.startswith("relay:"))
    assert relay_line == (
        "relay:     running (pid 4711), NOT answering its control socket (state: wedged)"
    ), lines
    # And the same fact one row below, in the audit's own words: one process, one story.
    assert any("relay not answering" in line for line in lines), lines

    stopped = dict(wedged, relay_running=False, relay_answering=False, relay_state="stopped")
    stopped_lines = net_tool._render("status", stopped)  # noqa: SLF001 — the renderer under test
    assert any(line == "relay:     not running" for line in stopped_lines), stopped_lines
