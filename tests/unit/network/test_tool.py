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
import os
import socket
import stat
import subprocess
import sys
import time
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
    "ready",
    "sessions",
    "trust",
    "credentials",
    "definitions_state",
    # The user-scope MCP server list (mcpdefs.py): what a peer would receive and
    # which reference keys a mirror still needs.
    "mcp_state",
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
    "ready": {"peer": "device-b"},
    "sessions": {"peer": "device-b"},
    "trust": {"network": "devmesh"},
    "credentials": {},
    "definitions_state": {},
    "mcp_state": {},
}


@pytest.fixture()
def isolated_network_config(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    """A private config root for the CHILD CLI, and a PATH with no ``launchctl``.

    The tool hands the child its environment, so pointing
    ``LOCAL_OPERATOR_CONFIG_DIR`` here is what keeps a test from writing a device
    identity, a network record or an audit line into the operator's own store.
    ``PATH`` is narrowed to this interpreter's directory for the second reason:
    ``lop network init`` starts the relay under the platform's user supervisor
    when there is one to do it with, and a test must not install a plist or a
    systemd unit and load it into the operator's real session. Without
    ``launchctl``/``systemctl`` on PATH the child takes its own documented "no
    user service supervisor here: the relay can run in the foreground instead"
    branch, which is the same code path with the side effect removed.
    """
    root = tmp_path / "config"
    root.mkdir()
    monkeypatch.setenv("LOCAL_OPERATOR_CONFIG_DIR", str(root))
    monkeypatch.setenv("PATH", str(Path(sys.executable).parent))
    _pin_loopback_listener(root)
    return root


def _free_port() -> int:
    """A port the OS just handed back, so "nothing is there" is the common case.

    Deliberately not a fixed number: everything below is about a dial this test does
    not want to succeed, and a constant would eventually be somebody's live port.
    The tiny race (a parallel worker's own ``bind(0)`` taking it back) is why the
    joining test still accepts the two other first clauses.
    """
    probe = socket.socket()
    try:
        probe.bind(("127.0.0.1", 0))
        return int(probe.getsockname()[1])
    finally:
        probe.close()


def _pin_loopback_listener(root: Path) -> None:
    """Loopback, on a port nothing else on this machine is listening on.

    THE DEVICE'S OWN ADDRESS IS REAL NOW, so these tests have to say WHERE it is. A
    device on the default ``0.0.0.0`` listener advertises whatever the interface
    table shows it (``network/addresses.py`` — before that fix, macOS advertised
    nothing), and the default mesh port is the one an operator's OWN relay is
    listening on. ``test_join_cannot_be_completed_by_the_tool_and_says_why`` drives a
    real dial, so without this it reaches the operator's live mesh — a unit test's
    side effect on someone's running network, and the reason its refusal clause would
    depend on what is listening there. Pinning the listener to loopback keeps the
    whole exchange inside the test, which is the same rule that strips ``launchctl``
    from this fixture's PATH.

    Written through the settings registry rather than by hand: it is the route
    ``/settings`` writes, so a config the child CLI cannot read would fail here.
    """
    from local_operator import settings_io
    from local_operator.config import ConfigManager

    manager = ConfigManager(root)
    settings_io.write_setting(manager, settings_io.BY_KEY["network.listen_address"], "127.0.0.1")
    settings_io.write_setting(manager, settings_io.BY_KEY["network.port"], _free_port())


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
    for verb in ("create", "engage", "stop", "delete", "send", "steer", "slash"):
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
        # The pilot acts could not spell one either: the text is DATA after
        # ``--``, so even a payload that opens with a dash-shaped word is
        # delivered to the peer rather than read as a flag.
        NetworkParams(action="sessions", peer="device-b", send="s1", text="--force it"),
        NetworkParams(action="sessions", peer="device-b", steer="s1", text="--yes"),
        NetworkParams(action="sessions", peer="device-b", slash="s1", text="/rename x"),
        NetworkParams(action="sessions", peer="device-b", create=True, prompt="hi"),
        NetworkParams(action="trust", network="devmesh", trust_state="untrusted"),
    ]
    for params in samples:
        argv, problem = net_tool._argv_for(params)
        assert problem == ""
        # Only the FLAG REGION is checked for the forbidden spellings: a pilot
        # act's text rides after ``--``, where the CLI reads it as DATA — the
        # sample above delivers a payload that literally spells ``--yes`` and
        # still cannot act as one, which is the property this test pins.
        flag_region = argv[: argv.index("--")] if "--" in argv else argv
        assert not (set(flag_region) & forbidden), argv


def test_the_tool_exposes_no_way_to_answer_a_park() -> None:
    """R3's pin: the tool may START a pairing and has no way to FINISH one.

    The property is not "the code is only accepted from a field the caller filled in".
    That was the earlier revision, and agent review round 1 (semantic finding 1) showed
    what it is worth: a park PRINTS this device's own derivation, both devices derive
    the SAME digits, so a model holding the code it just printed can echo it back and
    satisfy the very comparison that exists to catch a substitution. The operator's
    ruling was to remove the field rather than guard it, so the pin is now structural —
    no field, no alias, no argv.

    SABOTAGE CHECK, and it is the shape that discriminates: adding a ``confirm`` field
    to ``NetworkParams`` plus a branch spelling ``--confirm`` from it FAILS the sweep
    below. A test written against the old sentence ("the tool never invents the value")
    passes such a sabotage, which is why this one is written against the model's own
    field set and the argv table.
    """
    assert "confirm" not in NetworkParams.model_fields
    assert not any("confirm" in (spec.alias or "") for spec in NetworkParams.model_fields.values())
    for action in _ALL_ACTIONS:
        argv, problem = net_tool._argv_for(NetworkParams(action=action, **_SAMPLES[action]))
        assert problem == "", (action, problem)
        assert "--confirm" not in argv, (action, argv)
    # A caller reaching for any of the names a code could arrive under is refused by the
    # MODEL rather than quietly routed: ``extra="forbid"`` is part of the pin.
    for name in ("confirm", "sas", "code", "confirmation", "typed"):
        # ``dict[str, Any]`` because this is a RAW payload — what a caller (or the
        # harness's tier callback) hands over before validation — and not a
        # ``NetworkParams`` the checker can hold to its field types.
        loose: dict[str, Any] = {"action": "join", "token": "tok", name: "481926"}
        with pytest.raises(Exception):
            NetworkParams(**loose)
    # The parked spelling is the ONLY spelling: a code in any other field changes
    # nothing, because no other field reaches the CLI's argv.
    argv, problem = net_tool._argv_for(NetworkParams(action="join", token="tok"))
    assert problem == ""
    assert argv == ["network", "join", "tok", "--park", "--json"], argv


def test_all_peers_beside_a_mutating_verb_is_refused_not_dropped() -> None:
    """MINOR 5: the CLI reads ``--all-peers`` only on its two listing paths, so beside
    a mutating verb it accepted the flag and dropped it — the class the CLI's own
    ``--force`` guard names. Both halves are asserted, because a refusal that also
    refused the honest listing would be a different bug."""
    argv, problem = net_tool._argv_for(NetworkParams(action="sessions", all_peers=True))
    assert problem == "" and "--all-peers" in argv, (argv, problem)
    argv, problem = net_tool._argv_for(
        NetworkParams(action="sessions", peer="d1", stop="s1", all_peers=True)
    )
    assert argv == [] and "all_peers" in problem, (argv, problem)


def test_the_tier_and_the_argv_read_the_same_verb_set() -> None:
    """MINOR 4, as a regression: ``{"stop": "0"}`` was tiered ``read`` while argv still
    spelled ``--stop 0``, so a mutation rode a call that raised no approval. The two
    readers now share one predicate, and every falsy spelling is asserted on BOTH —
    tier and argv — so a divergence fails here rather than in the audit."""
    for value in ("0", "false", "no", "none", "null", "", "  "):
        for verb in ("stop", "delete", "engage", "send", "steer", "slash"):
            raw: dict[str, Any] = {"action": "sessions", "peer": "d1", verb: value}
            assert net_tool._approval_tier(raw) == "read", (verb, value)
            argv, problem = net_tool._argv_for(NetworkParams(**raw))
            assert problem == "", (verb, value, problem)
            assert f"--{verb}" not in argv, (verb, value, argv)
    # A real operand raises the tier AND reaches argv.
    args: dict[str, Any] = {"action": "sessions", "peer": "d1", "stop": "s_1"}
    assert net_tool._approval_tier(args) == "write"
    argv, problem = net_tool._argv_for(NetworkParams(**args))
    assert problem == "" and argv[argv.index("--stop") + 1] == "s_1", argv
    # 'create' answers to the same rule in both spellings a model may send.
    assert (
        net_tool._approval_tier({"action": "sessions", "peer": "d1", "create": "false"}) == "read"
    )
    assert net_tool._approval_tier({"action": "sessions", "peer": "d1", "create": True}) == "write"
    bad: dict[str, Any] = {"action": "sessions", "peer": "d1", "create": "false"}
    argv, problem = net_tool._argv_for(NetworkParams(**bad))
    assert problem == "" and "--create" not in argv, argv


def test_the_description_teaches_the_two_phase_pair_and_drops_the_old_claim() -> None:
    """The description is the FIRST thing an agent reads, so it is the place a
    capability that does not exist costs the most: it used to say sessions on other
    devices were not reachable, which the ``sessions`` action now contradicts.

    It must also not promise a guarantee this tool does not enforce. The second phase is
    not a field here (``test_the_tool_exposes_no_way_to_answer_a_park``), so the
    description has to say whose step it is — an agent that read "pairing needs a human"
    and then found no way to hand the step over would be told nothing useful.
    """
    tool = TOOL_BUILDERS["network"](ToolContext(cwd="."))
    assert tool is not None
    text = tool.description
    assert "Read and drive a lop mesh network from this device" in text
    # PR-B replaced the tail clause with the pilot family's advertisement (§4):
    # the sessions action now carries send/steer/slash too, and a description
    # that still said only "creating a session on a peer" would hide them.
    assert "send/steer/slash a conversation" in text
    assert "not reachable" not in text
    assert "Pairing and incident controls need a human" in text
    # The two clauses that make the guarantee true as written: the tool parks and
    # returns the code, and ANSWERING it is the person's.
    assert "returns the code for the user to read out" in text
    assert "answering it is theirs to run" in text
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

    # THROUGH ``_scrub``, in the order the real path uses it (``execute_network``
    # scrubs the payload, then renders it). Passing the raw dict here is what let this
    # test pin a contract production could not satisfy: the row's name field was spelled
    # ``key``, the scrubber's markers ate it, and the digest rendered ``None`` where the
    # credential's name belongs (agent review round 1, code findings 4 and 5).
    credentials = {
        "ok": True,
        "networks": [
            {
                "network": "home-net",
                "credentials": [
                    {
                        "credential_name": "OPENAI_API_KEY",
                        "kind": "api_key",
                        "owned_here": True,
                        "owner_device": "d_1",
                        "owner_device_name": "mbp",
                    }
                ],
            }
        ],
    }
    scrubbed = net_tool._scrub(credentials)
    assert "OPENAI_API_KEY" in json.dumps(scrubbed), scrubbed
    assert net_tool._render("credentials", scrubbed) == [  # noqa: SLF001
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

    assert net_tool._render(  # noqa: SLF001
        "mcp_state",
        {
            "ok": True,
            "servers": [
                {
                    "name": "gl",
                    "transport": "stdio",
                    "origin": "",
                    "refs": [{"id": "GITLAB_TOKEN", "set": False}],
                },
                {"name": "crm", "transport": "http", "origin": "d_1", "refs": []},
                {
                    "name": "leaky",
                    "transport": "http",
                    "origin": "",
                    "refs": [],
                    "withheld": "github-token",
                },
            ],
        },
    ) == [
        "server: gl  stdio (yours) — needs: GITLAB_TOKEN",
        "server: crm  http (mirrored from d_1)",
        "server: leaky  http (yours) — will not travel: looks like a github-token",
    ]
    assert net_tool._render("mcp_state", {"ok": True, "servers": []}) == [  # noqa: SLF001
        "no user-scope MCP servers on this device"
    ]


def test_the_credentials_digest_carries_the_shareable_block() -> None:
    """The agent surface mirrors the CLI's device-level ledger (design §2): the
    shareability preflight, read through ``_scrub`` in the order the real path uses.

    Provider-login rows ride the block too (Radient org projection): the scrubber must
    pass an ``identity_label`` through, and the renderer must show it."""
    payload = {
        "ok": True,
        "networks": [],
        "shareable": [
            {
                "server": "slack",
                "url": "https://h.example/mcp",
                "transport": "http",
                "login_here": True,
                "shared_with": [{"device": "d_1", "name": "cloud-node-1", "scope": "session"}],
                "remedy": "lop network credential share mcp:https://h.example/mcp --with <device>",
            },
            {
                "server": "notion",
                "url": "https://n.example/mcp",
                "transport": "http",
                "login_here": False,
                "shared_with": [],
                "remedy": "sign in here first",
            },
            {
                "provider": "radient",
                "kind": "oauth-rotating",
                "identity_label": "owner@example.test",
                "shared_with": [],
                "remedy": "lop network credential share radient --with <device>",
            },
        ],
    }
    scrubbed = net_tool._scrub(payload)  # noqa: SLF001 — the scrub boundary under test
    assert net_tool._render("credentials", scrubbed) == [  # noqa: SLF001
        "shareable here:",
        "  slack  http  login held — share: lop network credential share mcp:https://h.example/mcp"
        " --with <device>",
        "      shared with cloud-node-1 (session)",
        "  notion  http  no login here yet — sign in here first",
        "  radient  oauth-rotating  login held — share: lop network credential share radient"
        " --with <device>",
        "      organization account — share only to your own devices",
        "      signed in as owner@example.test",
    ]
    # An absent block leaves the old rendering (and its fallback) alone.
    assert net_tool._render("credentials", {"ok": True, "networks": []}) == [  # noqa: SLF001
        "nothing is shared with or by this device"
    ]


def test_a_detached_ceremony_is_reaped_by_a_waiter_of_its_own() -> None:
    """The zombie agent review round 1 caught, as a regression.

    A parked ceremony outlives the tool call that starts it, and that call's loop dies
    with the call — so asyncio's child watcher is gone long before the ceremony ends and
    nothing waits for it. What that looked like in the field: ``state=Z`` for 300
    consecutive polls, with a liveness probe answering "alive" for a process that had
    finished. ``_reap_when_it_exits`` is the waiter that outlives the call.

    THE COUNTERFACTUAL IS IN THE TEST, because a test that only asserts "the pid
    disappears" would pass with the helper deleted: two children are spawned, ONE is
    handed to the helper and the other is left alone, and the process table is read for
    both. The unreaped one is the control — it must still be a zombie while the reaped
    one is gone, which is what makes this a measurement of the helper rather than of
    ``Popen``'s own cleanup.
    """
    if not hasattr(os, "waitpid"):
        pytest.skip("no zombies off POSIX — and none to reap")
    from local_operator import procstate

    script = "import sys; sys.exit(0)"
    reaped = subprocess.Popen([sys.executable, "-c", script])
    control = subprocess.Popen([sys.executable, "-c", script])
    try:
        net_tool._reap_when_it_exits(reaped.pid)  # noqa: SLF001 — the helper under test
        deadline = time.time() + 15.0
        gone = False
        corpse = False
        while time.time() < deadline:
            # ``pid_liveness`` answers False only when NOTHING holds the pid, and the
            # control's ``is_zombie`` is the counterfactual — a corpse this helper never
            # touched. Both come from the repo's own probes rather than from ``ps``,
            # which this module's PATH-narrowing fixture would not even find.
            gone = procstate.pid_liveness(reaped.pid) is False
            corpse = procstate.is_zombie(control.pid)
            if gone and corpse:
                break
            time.sleep(0.05)
        assert gone, (
            "the detached ceremony is still holding its pid, so the waiter reaped "
            f"nothing: liveness={procstate.pid_liveness(reaped.pid)!r}"
        )
        assert corpse, (
            "the control child is not a zombie, so this test is not measuring a zombie "
            "at all and its other assertion would prove nothing"
        )
    finally:
        try:
            control.wait(timeout=10.0)
        except subprocess.TimeoutExpired:  # pragma: no cover — defensive
            control.kill()
            control.wait(timeout=10.0)


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


#: A ready payload with the shape the live verb produces when rows fail: the
#: reviewer's Q-1 repro (operator_authority + mcp_credential red), used to pin
#: that the agent path keeps the rows and remedies the CLI/--json already had.
_UNHEALTHY_READY: dict[str, Any] = {
    "ok": False,
    "code": "unhealthy",
    "message": (
        "operator_authority: operator authority is not installed on cloud-node-1 "
        "(no anchor is installed, so the runtime trusts no key yet): an approval that "
        "needs the operator — a write or command offloaded there — parks until someone "
        "installs it; mcp_credential: if the `files` server needs a sign-in, this "
        "device has no MCP login for https://mcp.example.test/files"
    ),
    "identity_present": True,
    "checks": [
        {
            "check": "reachability",
            "device_id": "d_1",
            "device_name": "cloud-node-1",
            "endpoint": "127.0.0.1:4097",
            "ok": True,
            "detail": "ok",
            "observed": {"outcome": "connected", "winner_verified": True},
            "remedies": [],
        },
        {
            "check": "readiness",
            "capability": "operator_authority",
            "device_id": "d_1",
            "device_name": "cloud-node-1",
            "ok": False,
            "code": "not_installed",
            "detail": (
                "operator authority is not installed on cloud-node-1 (no anchor is "
                "installed, so the runtime trusts no key yet): an approval that needs "
                "the operator — a write or command offloaded there — parks until someone "
                "installs it"
            ),
            "remedies": [
                "run `lop operator install` on cloud-node-1 (one privileged step), then "
                "approvals for offloaded work can be answered from this device"
            ],
            "source": "peer",
        },
        {
            "check": "readiness",
            "capability": "mcp_credential",
            "device_id": "d_1",
            "device_name": "cloud-node-1",
            "ok": False,
            "code": "no_credential",
            "detail": (
                "if the `files` server needs a sign-in, this device has no MCP login for "
                "https://mcp.example.test/files — sign in here first"
            ),
            "remedies": [
                "sign in here first, then `lop network "
                "credential share mcp:https://mcp.example.test/files --with cloud-node-1`"
            ],
            "source": "local",
        },
    ],
}


def test_a_ready_report_against_an_empty_store_is_an_error_with_its_rows() -> None:
    """The real CLI, through the tool: a fresh store is an unhealthy report.

    The gate must keep the rows on this path too — the report IS the product,
    and ``is_error`` is how the loop knows the verb did not come back green.
    """
    result = _call("ready")
    assert result.is_error
    assert "FAIL" in _text(result)
    assert _payload(result).get("code") == "unhealthy"


def test_the_ready_digest_keeps_fail_rows_and_remedies_on_an_unhealthy_report(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """QA round 1, Q-1: an unhealthy report IS the report.

    The gate sent ``ok:false`` payloads down the generic refusal branch, whose
    text is ``code: message`` — so the FAIL rows and their remedies, the whole
    product of this verb, never reached an agent while the hint promised them.
    The stub returns the live shape (the reviewer's repro); the real-CLI cell
    above covers the same gate end to end.
    """

    async def fake_cli(argv: list[str], timeout: float) -> tuple[int, str, str]:
        assert argv[:2] == ["network", "ready"]
        return 1, json.dumps(_UNHEALTHY_READY), ""

    monkeypatch.setattr(net_tool, "_run_cli", fake_cli)
    result = _call("ready", peer="cloud-node-1")
    assert result.is_error
    text = _text(result)
    assert "FAIL readiness operator_authority cloud-node-1" in text
    assert "lop operator install" in text
    assert "sign in here first, then" in text
    assert "credential share mcp:https://mcp.example.test/files" in text
    assert _payload(result)["code"] == "unhealthy"


def test_the_agent_digest_reads_a_non_gating_row_as_warn_not_fail() -> None:
    """F8 residual: the digest is one of the surfaces the ruling touched.

    A failed non-gating equipment row reads ``warn … — not required for
    onboarding`` here exactly as on the CLI — never ``FAIL`` — while
    still-gating equipment keeps its honest FAIL. The clause is the receipt's
    own literal, so the two surfaces cannot drift.
    """
    checks: list[dict[str, Any]] = [
        {
            "check": "readiness",
            "capability": "mcp_credential",
            "class": "equipment",
            "device_id": "d_" + "b" * 32,
            "device_name": "cloud-node-1",
            "ok": False,
            "code": "no_credential",
            "detail": (
                "this device has no MCP login for https://mcp.slack.com/mcp; " "sign in here first"
            ),
            "remedies": [],
        },
        {
            "check": "readiness",
            "capability": "operator_authority",
            "class": "equipment",
            "device_id": "d_" + "b" * 32,
            "device_name": "cloud-node-1",
            "ok": False,
            "detail": "no operator authority is installed on cloud-node-1",
            "remedies": [],
        },
    ]
    lines = net_tool._render(  # noqa: SLF001 — the renderer under test
        "ready", {"ok": False, "identity_present": True, "checks": checks}
    )
    body = "\n".join(lines)
    assert "warn readiness mcp_credential cloud-node-1" in body
    assert "— not required for onboarding" in body
    assert "FAIL readiness mcp_credential" not in body
    assert "FAIL readiness operator_authority cloud-node-1" in body


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
    """The tool STARTS a pairing; it does not finish one — and cannot.

    Before the two-phase pair this asserted the opposite shape (a refusal telling the
    agent to hand the step to a terminal). The two-phase pair made the tool park, and
    review round 1 then removed the field that answered a park, so the hint now states
    the property R3 asks for in its final form: the second phase is the person's, and
    this tool has no way to answer a pairing at all.
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
    assert "This tool has no way to answer a pairing" in text


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
    # AND IT SAYS THE HONEST THING ABOUT *WHY*: no read has completed, rather than
    # borrowing the failure words ("no peer answered") about an ask nobody made —
    # the "contradiction" class in the other direction.
    assert any("no table read has completed yet" in line for line in unread), unread
    assert not any("no peer answered" in line for line in unread), unread


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


def test_the_peers_digest_carries_the_build_suffix(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Design §4 through the same helper the CLI uses: a stale peer is the one the
    model should send `lop-update` to, and an unknown build stays silent."""
    from local_operator.network import relay

    monkeypatch.setattr(relay, "build_stamp", lambda: {"version": "0.64.1"})
    lines = net_tool._render(  # noqa: SLF001 — the renderer under test
        "peers",
        {
            "peers": [
                {
                    "reachable": True,
                    "device_id": "d_" + "1" * 32,
                    "name": "fresh",
                    "reason": "",
                    "build": {"version": "0.64.1"},
                },
                {
                    "reachable": True,
                    "device_id": "d_" + "2" * 32,
                    "name": "stale",
                    "reason": "",
                    "build": {"version": "0.63.2"},
                },
                {"reachable": True, "device_id": "d_" + "3" * 32, "name": "quiet", "reason": ""},
            ]
        },
    )
    assert "reachable   fresh  build 0.64.1" in lines, lines
    assert (
        "reachable   stale  build 0.63.2 — behind this device (0.64.1); "
        "ask Local Operator to update it there" in lines
    ), lines
    assert "reachable   quiet" in lines, lines


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


def test_the_agent_digest_of_a_scoped_address_reads_out_of_scope_too() -> None:
    """F9 at the agent's surface (design round 1, D3): the same reading as the
    CLI's doctor — excluded from the decision, named out of scope, and the raw
    dial result does not ride along."""
    from local_operator.network import readiness as readiness_mod

    scoped: dict[str, Any] = {
        "check": "reachability",
        "ok": False,
        "device_id": "d_" + "b" * 32,
        "endpoint": "172.20.0.246:4097",
        "detail": "no_answer",
    }
    readiness_mod.mark_out_of_scope([scoped], ["10.9.0.5"])
    lines = net_tool._render(  # noqa: SLF001 — the renderer under test
        "doctor", {"ok": True, "identity_present": True, "checks": [scoped]}
    )
    body = "\n".join(lines)
    assert "n/a  reachability" in body
    assert "172.20.0.246:4097 — out of scope: " in body
    assert "no_answer" not in body and "FAIL" not in body


def test_the_agent_digest_of_a_ready_run_keeps_refused_and_silent_apart() -> None:
    """The readiness digest reads like the CLI's, through the same reading.

    The second half is the one a docstring cannot carry: a REFUSED connection
    must not read as "nothing answered" on the agent's surface either, or the
    model reports a sleeping machine for one whose relay is simply not running.
    Remedies render under their row; the raw vocabulary stays in ``details``.
    """
    checks = [
        {
            "check": "reachability",
            "device_id": "d_" + "c" * 32,
            "device_name": "pi-box",
            "endpoint": "10.0.0.9:7777",
            "ok": False,
            "detail": "connect_failed:ConnectionRefusedError",
            "observed": {
                "outcome": "refused",
                "source_address": "10.0.0.2",
                "interface": "en0",
                "elapsed_ms": 12.0,
                "budget_s": 3.0,
                "attempted": True,
                "last_seen_at": None,
            },
        },
        {
            "check": "reachability",
            "device_id": "d_" + "b" * 32,
            "device_name": "cloud-node-1",
            "endpoint": "54.1.2.3:7777",
            "ok": False,
            "detail": "connect_failed:TimeoutError",
            "observed": {
                "outcome": "no_answer",
                "source_address": "203.0.113.7",
                "interface": "utun4",
                "elapsed_ms": 3000.0,
                "budget_s": 3.0,
                "attempted": True,
                "last_seen_at": None,
            },
        },
        {
            "check": "readiness",
            "capability": "operator_authority",
            "device_id": "d_" + "b" * 32,
            "device_name": "cloud-node-1",
            "ok": False,
            "code": "not_installed",
            "detail": (
                "operator authority is not installed on cloud-node-1: an approval that "
                "needs the operator parks until someone installs it"
            ),
            "remedies": ["run `lop operator install` on cloud-node-1 (one privileged step)"],
        },
    ]
    lines = net_tool._render(  # noqa: SLF001 — the renderer under test
        "ready", {"ok": False, "identity_present": True, "checks": checks}
    )
    body = "\n".join(lines)
    for token in ("ConnectionRefusedError", "TimeoutError", "connect_failed"):
        assert token not in body, (token, body)
    assert "something answered this address and refused the connection" in body
    assert "nothing answered this address before the budget ran out" in body
    assert (
        "FAIL readiness operator_authority cloud-node-1: operator authority is not installed"
        in body
    )
    assert "  - run `lop operator install` on cloud-node-1 (one privileged step)" in body


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


def test_the_agent_digest_carries_the_running_build_at_its_own_column() -> None:
    """D2 (design round 1): the digest renders the build row the payload carries.

    The CLI block gained the row; the digest — the surface "why is my session
    stuck" actually arrives on — rendered every other row of the same payload
    and silently dropped this one, so a stale relay could not be diagnosed from
    the model's own view. Same words as the CLI (``relay.generation_words``), at
    this register's own column (cell 11), and the flag + remedy ride the stale
    form. No row where the CLI has none: not running, or no generation layout.
    """
    payload = {
        "installed": True,
        "supported": True,
        "identity_present": True,
        "relay_running": True,
        "relay_answering": True,
        "relay": {"pid": 4711},
        "relay_generation": "20260921T125352Z-0.61.12",
        "installed_generation": "20260924T103058Z-509c7450dbf6",
        "relay_generation_stale": False,
        "relay_build": "0.61.12",
        "installed_build": "0.67.4",
        "log": "/tmp/network.log",
        "networks": [],
    }
    lines = net_tool._render("status", payload)  # noqa: SLF001 — the renderer under test
    build_line = next(line for line in lines if line.startswith("build:"))
    assert build_line == "build:     0.61.12", lines
    assert build_line.index("0.61.12") == 11, build_line

    stale = dict(payload, relay_generation_stale=True)
    target = "build:     0.61.12 — behind install 0.67.4; restart the relay"
    assert any(line == target for line in net_tool._render("status", stale)), stale

    # And an UNPROVEN comparison NAMES the readable build with its limit (F10
    # slice A; design round 1 D1): the default is not a verdict, and the value
    # the surface holds is not withheld, on the model's surface too.
    unproven = dict(payload, relay_generation_stale=None)
    assert any(
        line == "build:     0.61.12 — cannot confirm it is current"
        for line in net_tool._render("status", unproven)
    ), unproven

    # The two absences the CLI block has: no layout (nothing to say) and not
    # running (nothing whose build it would be).
    plain = {
        key: value
        for key, value in payload.items()
        if key
        not in (
            "relay_generation",
            "installed_generation",
            "relay_generation_stale",
            "relay_build",
            "installed_build",
        )
    }
    assert not [line for line in net_tool._render("status", plain) if line.startswith("build:")]
    stopped = dict(payload, relay_running=False)
    assert not [line for line in net_tool._render("status", stopped) if line.startswith("build:")]


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


# ---------------------------------------------------------------------------
# PR-B: the pilot acts (send/steer/slash) on a peer's session
# ---------------------------------------------------------------------------


def test_a_pilot_act_spells_json_before_the_act_and_text_after_the_separator() -> None:
    """Design §3A, as a pin — the ordering IS the behaviour here.

    ``--json`` must precede the act flag and the text must ALWAYS follow ``--``:
    the CLI takes the payload as a REMAINDER from the first non-option token, so
    a trailing ``--json`` would be delivered to the peer AS TEXT. The proof case
    is a text that itself begins with ``--json`` — the shape that reads wrong if
    either rule is dropped.
    """
    argv, problem = net_tool._argv_for(  # noqa: SLF001 — the builder under test
        NetworkParams(action="sessions", peer="device-b", send="t1", text="--json is the field")
    )
    assert problem == ""
    assert argv == [
        "network",
        "sessions",
        "--json",
        "--peer",
        "device-b",
        "--send",
        "t1",
        "--",
        "--json is the field",
    ]
    assert argv.count("--json") == 1
    assert argv.index("--json") < argv.index("--send")
    for params, verb in (
        (NetworkParams(action="sessions", peer="device-b", steer="t1", text="x"), "steer"),
        (NetworkParams(action="sessions", peer="device-b", slash="t1", text="x"), "slash"),
    ):
        argv, problem = net_tool._argv_for(params)  # noqa: SLF001
        assert problem == ""
        assert argv.index("--json") < argv.index(f"--{verb}")
        assert argv[-2] == "--" and argv[-1] == "x", argv


def test_the_pilot_guards_are_the_clis_one_act_rule_in_the_tools_words() -> None:
    """Exactly one act; an act beside create/engage/stop/delete is refused with
    the CLI's own contrast; an empty text is refused BEFORE the child exists —
    the CLI's stdin fallback is unreachable behind DEVNULL, so its own "needs
    some text" sentence would name a route this tool does not have."""
    argv, problem = net_tool._argv_for(  # noqa: SLF001
        NetworkParams(action="sessions", peer="d1", send="a", steer="b", text="x")
    )
    assert argv == [] and "one act at a time" in problem and "not a pipeline" in problem

    argv, problem = net_tool._argv_for(  # noqa: SLF001
        NetworkParams(action="sessions", peer="d1", send="a", stop="b", text="x")
    )
    assert argv == []
    assert "acts on the session you name" in problem and "would act on another" in problem

    argv, problem = net_tool._argv_for(  # noqa: SLF001
        NetworkParams(action="sessions", peer="d1", send="a", text="   ")
    )
    assert argv == []
    assert problem == (
        "action='sessions' with 'send' needs 'text': the words to deliver "
        "(this tool cannot pipe a body in)."
    )


def test_this_front_end_never_reaps_a_child_the_cli_is_still_working_inside() -> None:
    """The TUI's own derivation, mirrored for the tool (design §3A).

    ``_PILOT_TIMEOUT_S`` must sit ABOVE the CLI's worst-case act — the number
    the CLI reports its own expiry inside — or a tool call would kill a child
    mid-act and report a timeout about a command that was still working. The
    equality half pins the derivation, so a future edit to any of the CLI's
    three budgets cannot leave a stale literal here.
    """
    assert net_tool._PILOT_TIMEOUT_S > net_cli.PILOT_ACT_TIMEOUT_S  # noqa: SLF001
    assert net_tool._PILOT_TIMEOUT_S == net_cli.PILOT_ACT_TIMEOUT_S + 60.0  # noqa: SLF001
    # The CLI relays stop over a 240 s budget and create/engage over 120 s.
    assert net_tool._STOP_TIMEOUT_S > 240.0  # noqa: SLF001
    assert net_tool._CREATE_ENGAGE_TIMEOUT_S > 120.0  # noqa: SLF001

    def bound(**fields: Any) -> float:
        return net_tool._timeout_for(NetworkParams(action="sessions", **fields))  # noqa: SLF001

    assert bound(peer="d1", send="s", text="x") == net_tool._PILOT_TIMEOUT_S
    assert bound(peer="d1", steer="s", text="x") == net_tool._PILOT_TIMEOUT_S
    assert bound(peer="d1", slash="s", text="x") == net_tool._PILOT_TIMEOUT_S
    assert bound(peer="d1", stop="s") == net_tool._STOP_TIMEOUT_S
    assert bound(peer="d1", create=True, prompt="go") == net_tool._CREATE_ENGAGE_TIMEOUT_S
    assert bound(peer="d1", engage="s") == net_tool._CREATE_ENGAGE_TIMEOUT_S
    assert bound(peer="d1") == net_tool._DEFAULT_TIMEOUT_S


def test_a_pilot_receipt_renders_the_owners_outcome_not_a_code_message() -> None:
    """§3A's render branch: finished carries the reply; the non-completions say
    which they are (running/queued/failed/lost); steer and slash render the
    owner's own receipt; a wrong ``--peer`` rides ``peer_named``."""
    finished = net_tool._render(  # noqa: SLF001 — the renderer under test
        "sessions",
        {
            "verb": "send",
            "session_id": "s_1",
            "peer": "mbp",
            "ok": True,
            "outcome": "finished",
            "reply": "all done",
        },
    )
    assert finished == ["s_1 on mbp: the turn finished.", "all done"]

    running = net_tool._render(  # noqa: SLF001
        "sessions",
        {
            "verb": "send",
            "session_id": "s_1",
            "peer": "mbp",
            "ok": False,
            "outcome": "running",
            "code": "turn_running",
        },
    )
    assert running[0] == "s_1 on mbp took the turn and is still running it."
    assert "lop --resume s_1" in running[-1]

    failed = net_tool._render(  # noqa: SLF001
        "sessions",
        {
            "verb": "send",
            "session_id": "s_1",
            "peer": "mbp",
            "ok": False,
            "outcome": "failed",
            "code": "turn_failed",
            "error": "boom",
        },
    )
    assert failed[-1] == "boom"

    steer = net_tool._render(  # noqa: SLF001
        "sessions",
        {
            "verb": "steer",
            "session_id": "s_1",
            "peer": "mbp",
            "ok": True,
            "outcome": "steered",
            "receipt": "queued behind the current step",
        },
    )
    assert steer[0] == "s_1 on mbp: queued behind the current step"

    slash = net_tool._render(  # noqa: SLF001
        "sessions",
        {
            "verb": "slash",
            "session_id": "s_1",
            "peer": "mbp",
            "command": "rename",
            "ok": False,
            "outcome": "refused",
            "text": "no",
            "style": "error",
        },
    )
    assert slash == ["s_1 on mbp: /rename — no"]

    named = net_tool._render(  # noqa: SLF001
        "sessions",
        {
            "verb": "send",
            "session_id": "s_1",
            "peer": "mbp",
            "ok": True,
            "outcome": "finished",
            "reply": "r",
            "peer_named": "other-box",
        },
    )
    assert named[-1] == "you named other-box; s_1 is held by mbp, which is where this ran"


def test_a_non_completing_send_keeps_its_receipt_and_marks_the_call(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A ``running`` send is the owner's honest NON-completion: the CLI exits 1
    for it, and the tool must render the receipt (outcome + resume hint) rather
    than collapse it to ``code: message`` — while still reporting is_error, so a
    delivery is never read as a completion. The finished arm is the contrast:
    success, with the reply visible."""
    receipt = {
        "session_id": "s_1",
        "peer": "cloud-node-1",
        "verb": "send",
        "ok": False,
        "outcome": "running",
        "code": "turn_running",
    }

    async def fake_cli(argv: list[str], timeout: float) -> tuple[int, str, str]:
        assert argv[:3] == ["network", "sessions", "--json"]
        assert argv[argv.index("--send") + 1] == "s_1"
        assert timeout == net_tool._PILOT_TIMEOUT_S  # noqa: SLF001
        return 1, json.dumps(receipt), ""

    monkeypatch.setattr(net_tool, "_run_cli", fake_cli)
    result = _call("sessions", peer="cloud-node-1", send="s_1", text="hello there")
    assert result.is_error
    text = _text(result)
    assert "still running it" in text
    assert "lop --resume s_1" in text
    assert _payload(result)["outcome"] == "running"

    done = dict(receipt, ok=True, outcome="finished", reply="on it")

    async def fake_done(argv: list[str], timeout: float) -> tuple[int, str, str]:
        return 0, json.dumps(done), ""

    monkeypatch.setattr(net_tool, "_run_cli", fake_done)
    result = _call("sessions", peer="cloud-node-1", send="s_1", text="hello there")
    assert not result.is_error
    assert _text(result).endswith("on it")
