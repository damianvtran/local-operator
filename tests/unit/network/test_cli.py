"""The ``lop network`` surface: the parser, and the refusals that are features."""

from __future__ import annotations

import argparse
import json
import re
import time
from argparse import Namespace
from pathlib import Path
from typing import Any

import pytest

from local_operator import resume
from local_operator.network import audit as audit_mod
from local_operator.network import cli as net_cli
from local_operator.network import readiness, relay, store, types, wire
from local_operator.network.credentials import offers
from tests.unit.network import conftest as net_fixtures

NETWORK = "n_0123456789abcdef01234567"

ACTIONS = (
    "init",
    "invite",
    "join",
    "ls",
    "show",
    "rename",
    "rm",
    "member",
    "peers",
    "serve",
    "start",
    "stop",
    "restart",
    "status",
    "disconnect",
    "panic",
    "trust",
    "log",
    "doctor",
    # Peer readiness (readiness.py): the install question `doctor` does not ask.
    "ready",
    "identity",
    "uninstall",
    # The inviter's half of the pairing human step (mesh-transport-identity §5.3).
    "confirm",
    # The session plane's client half (mesh-session-mobility.md §9.3): what the
    # peers hold, and the three acts on a session that lives on one of them.
    "sessions",
    # Agent and team definitions (definitions.py): the deliberate half of the sync a
    # create performs implicitly, so a peer can be brought up to date without one.
    "definitions",
    # User-scope MCP server definitions (mcpdefs.py): the same deliberate half for
    # the servers an offloaded workload assumes — push is the form an operator
    # reaches for after "the pod has no GitLab server", state is the local ledger.
    "mcp",
    # The credential broker's surfaces (mesh-credentials.md §2.2): the READ of what
    # this device owns and borrows, and the ACT of sharing or revoking one. Both are
    # design verbs, which is why they belong in this list rather than beside it.
    "credentials",
    "credential",
    # The onboarding approval record (remote-onboarding §3.5): file it, read it,
    # answer it, run it — six verbs under one group, the shape the design froze.
    "approvals",
)


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="lop")
    subparsers = parser.add_subparsers(dest="subcommand")
    net_cli.add_parser(subparsers)
    return parser


def _group_parser() -> argparse.ArgumentParser:
    return net_fixtures.subcommands_of(_parser())["network"]


def test_every_action_from_the_design_is_registered() -> None:
    group = _group_parser()
    assert set(net_fixtures.subcommands_of(group)) == set(ACTIONS)


def test_the_ready_verb_takes_a_peer_by_name_and_json() -> None:
    """``lop network ready [--peer <name|id>] [--json]`` — the design's surface.

    ``--peer`` is a name a person types (``cloud-node-1``), so the parser
    carries a string and the RELAY resolves it; ``--json`` is the agent path's
    contract on every leaf verb.
    """
    parsed = _parser().parse_args(["network", "ready", "--peer", "cloud-node-1", "--json"])
    assert parsed.network_command == "ready"
    assert parsed.peer == "cloud-node-1"
    assert parsed.json is True
    # And a bare `lop network ready` is the every-member report, not a usage error.
    bare = _parser().parse_args(["network", "ready"])
    assert bare.peer == "" and bare.json is False


def test_the_mcp_verb_takes_push_and_state_each_with_json() -> None:
    """``lop network mcp push [--peer|--all-peers] [--json]`` and ``state [--json]``.

    The definitions pair's shape for the servers: ``push`` names a device a person
    types (the relay resolves it) or every member, and ``state`` is local.
    """
    parsed = _parser().parse_args(["network", "mcp", "push", "--peer", "cloud-node-1", "--json"])
    assert parsed.mcp_command == "push"
    assert parsed.peer == "cloud-node-1"
    assert parsed.json is True
    bare = _parser().parse_args(["network", "mcp", "push", "--all-peers"])
    assert bare.peer == "" and bare.all_peers is True
    state = _parser().parse_args(["network", "mcp", "state", "--json"])
    assert state.mcp_command == "state" and state.json is True


def test_mcp_push_with_no_peer_names_both_ways_it_accepts(
    capsys: pytest.CaptureFixture[str],
) -> None:
    """A push must name a device or the explicit every-member flag; --json callers
    get the machine code, which is what a script branches on."""
    rc = net_cli.main(_parser().parse_args(["network", "mcp", "push", "--json"]))
    assert rc != 0
    body = capsys.readouterr().out
    assert '"code": "peer_required"' in body, body
    assert "--all-peers" in body


def test_the_mcp_push_receipt_names_the_next_step_for_a_withheld_row(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    """D3/D5 (design round 1): a withheld row must say it was NOT sent and what
    to change — the "state the next move" bar the family's refusals meet — and
    the shape sentence must read "an authorization-bearer", never "a
    authorization-bearer"."""
    detail = {
        "ok": True,
        "message": "1 of 1 device holds this device's MCP servers",
        "peers": [
            {
                "device_id": "d_" + "b" * 32,
                "ok": True,
                "code": "applied",
                "message": "sent 1 MCP server definition",
                "installed": [{"kind": "server", "name": "gl"}],
                "updated": [],
                "conflicts": [],
                "refused": [],
                "withheld": [{"kind": "server", "name": "leaky", "shape": "authorization-bearer"}],
            }
        ],
    }
    monkeypatch.setattr(net_cli, "_relay_answer", lambda op, **fields: dict(detail))
    rc = net_cli.main(_parser().parse_args(["network", "mcp", "push", "--peer", "cloud-node-1"]))
    assert rc == 0
    out = capsys.readouterr().out
    assert out.count("1 of 1 device holds this device's MCP servers") == 1, out
    assert out.count("sent 1 MCP server definition") == 1, out
    assert "withheld server 'leaky': looks like an authorization-bearer" in out, out
    assert "— not sent;" in out and "push again" in out, out


def test_mcp_state_render_marks_a_row_that_will_not_travel(
    root: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    """D4 (design round 1): the local inventory must name a row the shape scan
    will never send BEFORE a push discovers it — the same words as the receipt."""
    monkeypatch.setenv("LOCAL_OPERATOR_CONFIG_DIR", str(root))
    (root / "mcp.json").write_text(
        json.dumps(
            {
                "mcpServers": {
                    "fine": {"type": "http", "url": "https://fine.example/mcp"},
                    "leaky": {"type": "http", "url": "https://x.example/ghp_" + "a" * 36},
                }
            },
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )
    rc = net_cli.main(_parser().parse_args(["network", "mcp", "state"]))
    assert rc == 0
    out = capsys.readouterr().out
    assert "server: leaky  http (yours) — will not travel: looks like a github-token" in out
    assert out.count("will not travel") == 1, out


def test_the_stop_verb_accepts_the_force_the_ladder_names() -> None:
    """Q-R5-2: the busy refusal names `--force`, so `--stop` must take it.

    The owner's ladder declines to signal a target whose turn is in flight and
    states the way past it in the same sentence — a sentence the mesh viewer
    paints verbatim. Before this flag existed the verb named an action it could not
    accept (UX round 2's U7, one front end over); the PARSER is asserted here so
    the flag cannot be dropped while the sentence keeps promising it, and the
    semantics (mode ``immediate`` = the owner's own force) are pinned in
    ``test_refusals``/``test_session_plane``.
    """
    parsed = _parser().parse_args(["network", "sessions", "--peer", "b", "--stop", "s", "--force"])
    assert parsed.force is True
    # AND IT DEFAULTS OFF, so every existing caller keeps the plain stop.
    assert _parser().parse_args(["network", "sessions", "--stop", "s"]).force is False


def test_every_leaf_action_accepts_json() -> None:
    """The agent path drives this CLI and parses it, so ``--json`` is a contract on
    every action that produces output."""
    group = _group_parser()
    missing: list[str] = []
    for name, subparser in net_fixtures.subcommands_of(group).items():
        flags = {option for action in subparser._actions for option in action.option_strings}
        # The groups whose verbs are sub-commands, so ``--json`` is asserted on each
        # LEAF rather than on the group (a bare group prints usage): the mesh slice's
        # own ``definitions`` group, plus ``member``, ``identity`` and ``credential``
        # (the last from the credential broker's surfaces), and ``mcp`` (the
        # definitions pair's shape for the user-scope MCP servers).
        if name in ("member", "identity", "definitions", "credential", "mcp", "approvals"):
            for nested_name, nested_parser in net_fixtures.subcommands_of(subparser).items():
                nested_flags = {
                    option for action in nested_parser._actions for option in action.option_strings
                }
                if "--json" not in nested_flags:
                    missing.append(f"{name} {nested_name}")
            continue
        if "--json" not in flags:
            missing.append(name)
    assert missing == []


def test_the_parser_accepts_a_parent_parser() -> None:
    """``parents=[parent_parser]`` is what makes the globals reachable on these
    subcommands, and it is the same argument every other group in ``cli.py`` uses.

    Both placements parse WITHOUT an error. The value that survives is the
    subparser's, because a subparser re-declares the global with a default —
    argparse behaviour that is repo-wide and identical for ``tunnels``, ``secrets``
    and this group, so what this test pins is acceptance, not precedence.
    """
    parent = argparse.ArgumentParser(add_help=False)
    parent.add_argument("--debug", action="store_true")
    parser = argparse.ArgumentParser(prog="lop", parents=[parent])
    subparsers = parser.add_subparsers(dest="subcommand")
    net_cli.add_parser(subparsers, parent)

    before = parser.parse_args(["--debug", "network", "ls", "--json"])
    assert (before.network_command, before.json) == ("ls", True)
    after = parser.parse_args(["network", "--debug", "ls", "--json"])
    assert (after.debug, after.network_command, after.json) == (True, "ls", True)


def test_the_group_works_without_a_parent_parser() -> None:
    """A test, a tool or an embedder may build the group alone; that path must not
    need the main CLI's globals to exist."""
    parser = argparse.ArgumentParser(prog="lop")
    subparsers = parser.add_subparsers(dest="subcommand")
    net_cli.add_parser(subparsers)
    parsed = parser.parse_args(["network", "status", "--json"])
    assert parsed.network_command == "status" and parsed.json is True


def test_an_unknown_action_is_reported_rather_than_guessed(
    capsys: pytest.CaptureFixture[str],
) -> None:
    assert net_cli.main(Namespace(network_command="teleport")) == 2
    assert "unknown network action" in capsys.readouterr().err


# ---------------------------------------------------------------------------
# Durations
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("text", "expected"),
    [("30s", 30.0), ("10m", 600.0), ("2h", 7200.0), ("90", 90.0)],
)
def test_durations_parse(text: str, expected: float) -> None:
    assert net_cli._duration(text) == expected  # noqa: SLF001


@pytest.mark.parametrize("text", ["", "soon", "-5m", "0s"])
def test_a_bad_duration_is_refused_at_parse_time(text: str) -> None:
    """Refused before a token is minted: an invite whose lifetime nobody could parse
    would otherwise default silently, and the one thing an operator must be able to
    trust about a bearer credential is when it dies."""
    with pytest.raises(argparse.ArgumentTypeError):
        net_cli._duration(text)


def test_maybe_duration_returns_none_for_nonsense() -> None:
    assert net_cli._maybe_duration("15m") == 900.0
    assert net_cli._maybe_duration("whenever") is None
    assert net_cli._maybe_duration("") is None


# ---------------------------------------------------------------------------
# The refusals that are features
# ---------------------------------------------------------------------------


def test_invite_refuses_print_together_with_json(capsys: pytest.CaptureFixture[str]) -> None:
    """``--json`` must never carry the token: a token in a machine-readable stream is
    a token in the agent's transcript."""
    args = Namespace(
        network="",
        role="drive",
        expires=600.0,
        hosts="",
        device="",
        print_token=True,
        json=True,
    )
    assert net_cli._cmd_invite(args) == 2  # noqa: SLF001
    assert "--json must never carry the token" in capsys.readouterr().err


def test_invite_refuses_print_on_a_pipe(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    monkeypatch.setattr("sys.stdout.isatty", lambda: False)
    args = Namespace(
        network="", role="drive", expires=600.0, hosts="", device="", print_token=True, json=False
    )
    assert net_cli._cmd_invite(args) == 2  # noqa: SLF001
    assert "not a terminal" in capsys.readouterr().err


def test_sas_stdin_is_a_test_seam_and_says_so(monkeypatch: pytest.MonkeyPatch) -> None:
    """An agent cannot complete a pairing: the code is typed at a prompt, and the
    harness seam is refused unless the environment says it is a test."""
    monkeypatch.delenv(net_cli.TEST_MODE_ENV, raising=False)
    with pytest.raises(ValueError):
        net_cli._read_code(Namespace(sas_stdin=True, verify=False), "481926", "FP")  # noqa: SLF001


def test_join_without_a_person_is_refused(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """No terminal, no prompt — and a CODE, because ``--json`` carries it.

    The refusal is a ``MeshRefusal`` rather than a bare ``ValueError``: the raw
    exception reached an agent as a non-answer with no machine code, and the sentence
    it now carries names the two-phase pair, which is the path a caller with no
    terminal actually has. The refusal itself STAYS: a prompt needs a person, and the
    park is a deliberate, explicit alternative rather than something this call does
    for you.
    """
    monkeypatch.setattr("sys.stdin.isatty", lambda: False)
    with pytest.raises(types.MeshRefusal) as excinfo:
        net_cli._read_code(Namespace(sas_stdin=False, verify=False), "481926", "FP")  # noqa: SLF001
    assert excinfo.value.code == "join_needs_tty"
    assert "person at a keyboard" in excinfo.value.sentence
    # The sentence has to name the way OUT, or the refusal is a dead end.
    assert "--park" in excinfo.value.sentence and "--confirm" in excinfo.value.sentence


def test_who_answered_is_claimed_only_where_a_person_could_have(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """``answered_by`` is a CLAIM about a person, so it is made only where one could be.

    Agent review round 1 (semantic finding 3) found this field writing ``human`` for a
    machine-supplied code: the rule was "not ``--sas-stdin``, therefore human", and the
    tool's phase two passed no seam, so a script's answer was recorded as a person's —
    into the parked record, the relay frame and the audit, where the question it answers
    is exactly "did a person read this back?". The rule is now the MECHANISM rather than
    the absence of a flag: the explicit pipe seam is the harness, a piped invocation
    that did not announce itself is the harness too (no person can be behind a pipe),
    and only a terminal makes the answer a person's.
    """
    args = Namespace(sas_stdin=False)
    monkeypatch.setattr(net_cli, "_has_terminal", lambda: True)
    assert net_cli._answered_by(args) == "human"  # noqa: SLF001 — the predicate under test
    monkeypatch.setattr(net_cli, "_has_terminal", lambda: False)
    assert net_cli._answered_by(args) == "harness"  # noqa: SLF001
    args.sas_stdin = True
    monkeypatch.setattr(net_cli, "_has_terminal", lambda: True)
    assert net_cli._answered_by(args) == "harness"  # noqa: SLF001


def test_verify_makes_the_fingerprint_the_compared_value(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr("sys.stdin.isatty", lambda: True)
    monkeypatch.setattr("builtins.input", lambda prompt="": "K7QM-3XPD-4B1N-9T2B")
    code = net_cli._read_code(  # noqa: SLF001
        Namespace(sas_stdin=False, verify=True), "481926", "K7QM-3XPD-4B1N-9T2B"
    )
    assert code == "481926"
    monkeypatch.setattr("builtins.input", lambda prompt="": "K7QM-3XPD-4B1N-9T2C")
    assert (
        net_cli._read_code(  # noqa: SLF001
            Namespace(sas_stdin=False, verify=True), "481926", "K7QM-3XPD-4B1N-9T2B"
        )
        == ""
    )


# ---------------------------------------------------------------------------
# Reading the token, and resolving a network
# ---------------------------------------------------------------------------


def test_a_token_can_come_from_a_file(root: Path) -> None:
    path = root / "token.invite"
    path.write_text("lop1.payload.tag\n", encoding="utf-8")
    assert net_cli._read_token(f"@{path}") == "lop1.payload.tag"  # noqa: SLF001
    assert net_cli._read_token("lop1.inline.tag") == "lop1.inline.tag"  # noqa: SLF001


def test_no_argument_reads_the_newest_outbox_file(
    root: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    # The env var and the explicit root are the SAME directory here: `_read_token`
    # resolves the outbox through `paths.config_dir()`, and a test that pointed the
    # two at different places would be testing its own mistake.
    monkeypatch.setenv("LOCAL_OPERATOR_CONFIG_DIR", str(root))
    older = store.save_invite_token("older", "lop1.old.tag", root)
    newer = store.save_invite_token("newer", "lop1.new.tag", root)
    assert older.exists() and newer.exists()
    assert net_cli._read_token("") == "lop1.new.tag"  # noqa: SLF001


def test_no_token_anywhere_is_an_error(root: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """A refusal with a sentence and a code, not a bare ``ValueError`` — the whole
    CLI's rule for a state that is legitimate rather than exceptional."""
    monkeypatch.setenv("LOCAL_OPERATOR_CONFIG_DIR", str(root))
    with pytest.raises(types.MeshRefusal) as excinfo:
        net_cli._read_token("")  # noqa: SLF001
    assert excinfo.value.code == "no_invite_token"
    assert "lop network invite" in excinfo.value.sentence


def _save_network(root: Path, network_id: str, name: str) -> types.NetworkRecord:
    record = types.NetworkRecord(
        network_id=network_id, name=name, self_device_id="d_" + "a" * 32, self_role="admin"
    )
    store.save(record, root)
    return record


def test_resolve_refuses_an_ambiguous_name(root: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """Names are not unique by design, so an ambiguous argument must refuse rather
    than pick — a `rm` that hit the wrong network because two were called "home"
    would be a data loss with a friendly message."""
    monkeypatch.setenv("LOCAL_OPERATOR_CONFIG_DIR", str(root))
    _save_network(root, NETWORK, "home")
    _save_network(root, "n_ffffffffffffffffffffffff", "home")
    with pytest.raises(types.MeshRefusal) as excinfo:
        net_cli._resolve("home")  # noqa: SLF001
    assert excinfo.value.code == "ambiguous_network"
    # The id is unambiguous, and a single network needs no argument at all.
    assert net_cli._resolve(NETWORK).network_id == NETWORK  # noqa: SLF001
    with pytest.raises(types.MeshRefusal):
        net_cli._resolve("")  # noqa: SLF001


def test_resolve_names_an_unknown_network(root: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("LOCAL_OPERATOR_CONFIG_DIR", str(root))
    _save_network(root, NETWORK, "home")
    with pytest.raises(types.MeshRefusal) as excinfo:
        net_cli._resolve("lab")  # noqa: SLF001
    assert excinfo.value.code == "unknown_network"
    assert net_cli._resolve("").network_id == NETWORK  # noqa: SLF001


# ---------------------------------------------------------------------------
# Output contracts
# ---------------------------------------------------------------------------


def test_json_output_is_json_and_nothing_else(
    root: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    monkeypatch.setenv("LOCAL_OPERATOR_CONFIG_DIR", str(root))
    _save_network(root, NETWORK, "home")
    args = Namespace(json=True, network="")
    assert net_cli._cmd_ls(args) == 0  # noqa: SLF001
    payload = json.loads(capsys.readouterr().out)
    assert payload["networks"][0]["network_id"] == NETWORK


def test_human_output_never_prints_the_token(
    root: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    """The invariant stated as a test: `invite` prints the PATH, and the token only
    ever appears on stdout through an explicit TTY ``--print``."""
    monkeypatch.setenv("LOCAL_OPERATOR_CONFIG_DIR", str(root))
    record = _save_network(root, NETWORK, "home")
    record.self_capabilities = sorted(types.capabilities_for_role("admin"))
    store.save(record, root)
    store.save_secrets(
        types.SecretState(network_id=NETWORK, epoch=1, secret=wire.b64u(b"s" * 32)), root
    )
    args = Namespace(
        network=NETWORK,
        role="drive",
        expires=600.0,
        hosts="",
        device="",
        print_token=False,
        json=False,
    )
    assert net_cli._cmd_invite(args) == 0  # noqa: SLF001
    printed = capsys.readouterr().out
    assert "lop1." not in printed
    assert "token written to" in printed
    # And the file it names does hold the token, at 0600.
    path = next(iter(store.outbox_dir(root).glob("*.invite")))
    assert path.read_text(encoding="utf-8").startswith("lop1.")


def test_status_reports_the_local_networks_with_no_relay_running(
    root: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    """`lop network status` is most useful when the relay is DOWN, so it must not
    answer "no networks" then: the store is the source of truth, and the relay's view
    only adds links."""
    monkeypatch.setenv("LOCAL_OPERATOR_CONFIG_DIR", str(root))
    _save_network(root, NETWORK, "home")
    args = Namespace(json=True)
    assert net_cli._cmd_status(args) == 0  # noqa: SLF001
    payload = json.loads(capsys.readouterr().out)
    assert payload["relay_running"] is False
    assert [row["network_id"] for row in payload["networks"]] == [NETWORK]
    assert payload["networks"][0]["links"] == 0


def test_the_status_command_asks_the_relay_for_a_fresh_read(
    root: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    """`lop network status` is a READ THAT ASKS (the "contradiction" class).

    The member block this command prints must come from a pass the command asked
    for, not from the cadence's last tick — the shape that let `status` say "no
    peer answered" beside `peers`' fresh probe. The relay side's semantics are
    tested where the relay is (``test_membership_convergence``); this cell pins
    the seam: the verb passes ``refresh=True`` through to ``relay.status``. The
    payload itself is the existing audit-block fixture, so nothing here
    re-derives the relay's answer.
    """
    monkeypatch.setenv("LOCAL_OPERATOR_CONFIG_DIR", str(root))
    seen: list[dict[str, Any]] = []

    def _status(*args: Any, **fields: Any) -> dict[str, Any]:
        seen.append(dict(fields))
        return _audit_status(root, audit_mod.AuditLog(root))

    monkeypatch.setattr(relay, "status", _status)
    assert net_cli._cmd_status(Namespace(json=False)) == 0  # noqa: SLF001
    capsys.readouterr()
    assert seen and seen[0].get("refresh") is True, seen


def test_the_show_member_line_does_not_say_members_twice() -> None:
    """ONE "members" at the seam (design round 1, N4).

    ``_show_lines`` prefixes the table line with "members: ", and the sentence it
    wraps begins with the same subject — the screen read "members: members NOT
    verified: …". The sentence keeps its subject for the readers that render it
    WITHOUT the prefix (`--json`, the agent digest), and this line drops the
    duplicate instead: the one place that would double it is the one place that
    removes it.
    """
    payload = {
        "name": "devmesh",
        "network_id": "n_" + "a" * 22,
        "epoch": 1,
        "trust": "active",
        "members": 2,
        "membership": {
            "sentence": "this device is an active member of devmesh",
            "remedies": [],
            "table": {
                "sentence": "members NOT verified: no table read has completed yet — retrying"
            },
        },
        "members_detail": [],
    }
    lines = net_cli._show_lines(payload)  # noqa: SLF001
    seam = next(line for line in lines if line.startswith("  members: "))
    assert seam == "  members: NOT verified: no table read has completed yet — retrying", seam


def test_doctor_reports_identity_missing_rather_than_claiming_reachability(
    root: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    """A dead instrument returns a reading, not an error — and a reading that has not
    been taken must not be presented as a result.

    ``ok`` IS THE READING: it is false here because the identity check failed, and
    the exit code follows it. When the top-level ``ok`` was hardcoded true, an agent
    that read the summary and not the array read a failing mesh as a passing one
    (QA round 1).
    """
    monkeypatch.setenv("LOCAL_OPERATOR_CONFIG_DIR", str(root))
    args = Namespace(json=True, peer="")
    assert net_cli._cmd_doctor(args) == 1  # noqa: SLF001
    payload = json.loads(capsys.readouterr().out)
    assert payload["ok"] is False
    assert payload["code"] == "unhealthy"
    assert "identity" in payload["message"]
    assert payload["identity_present"] is False
    checks = {check["check"] for check in payload["checks"]}
    assert "identity" in checks


def test_doctor_is_healthy_and_exits_zero_when_no_check_fails(
    root: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    """The other half of the contract: ``ok: true`` must be EARNED.

    Without this the change could be satisfied by refusing everything, which would
    be a worse instrument than the one it replaced.
    """
    monkeypatch.setenv("LOCAL_OPERATOR_CONFIG_DIR", str(root))
    import local_operator.network.identity as identity_mod

    identity_mod.load_or_mint()
    args = Namespace(json=True, peer="")
    assert net_cli._cmd_doctor(args) == 0  # noqa: SLF001
    payload = json.loads(capsys.readouterr().out)
    assert payload["ok"] is True
    assert "code" not in payload


def test_a_peer_row_is_a_name_and_words_never_the_wire_token() -> None:
    """UX round 5, U28: `/network peers` printed the id and the exception class.

    The row the round measured, on the surface a person reads from the composer::

        unreachable  d_1a2b3c…  pixel-8  connect_failed:ConnectionRefusedError

    Three faults in one line: a stage token plus a Python class name where the
    sibling create arm says "cannot be reached from this device right now", and a
    peer addressed by the 34-character id every other surface replaces with its
    name (``mesh-ui.md`` §1.2 gives the id a column of its own, eight characters
    of it; the sidebar's heading is ``⇄ <label>``). The token and the id are both
    still in this verb's ``--json`` payload — that is the machine surface, and the
    row below asserts they are the fields kept there rather than deleted.

    The composer prints whatever this verb's human lines say (``_publish_network_run``
    in ``tui/app.py``, pinned by ``test_the_receipt_is_the_clis_own_line``), so
    this is the line the user sees.
    """
    device_id = "d_" + "1a2b3c4d5e6f7a8b9c0d1e2f3a4b5c6d"
    row = {
        "device_id": device_id,
        "name": "pixel-8",
        "reachable": False,
        "reason": "connect_failed:ConnectionRefusedError",
    }
    line = net_cli._peer_line(row)  # noqa: SLF001
    assert line == "pixel-8 cannot be reached from this device right now (it did not answer)"
    assert "connect_failed" not in line
    assert "ConnectionRefusedError" not in line
    assert device_id not in line
    # The NAME when the peer has one, and the one shared spelling when it does
    # not (design round 1, D8) — never the id.
    assert net_cli._peer_line({**row, "name": ""}) == (  # noqa: SLF001
        "unnamed device cannot be reached from this device right now (it did not answer)"
    )
    # A peer that answered is a state word, and nothing else.
    assert net_cli._peer_line({**row, "reachable": True, "reason": ""}) == (  # noqa: SLF001
        "pixel-8  reachable"
    )


def test_a_reachable_peer_row_carries_its_build_and_the_behind_hint(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Design §4 / S5: the rows already carry ``build``, so the line states parity.

    One exact string per branch: unknown renders NOTHING (the pins above are that
    A/B), equal and ahead carry the bare version, and behind carries the only
    remedy that runs on the other device. No wire change and no probe — the value
    was already in the row, ``{}`` when no link answered.
    """
    monkeypatch.setattr(relay, "build_stamp", lambda: {"version": "0.64.1", "source_ref": "x"})
    row = {
        "device_id": "d_" + "1a2b3c4d5e6f7a8b9c0d1e2f3a4b5c6d",
        "name": "pixel-8",
        "reachable": True,
        "reason": "",
    }
    assert net_cli._peer_line({**row, "build": {"version": "0.64.1"}}) == (  # noqa: SLF001
        "pixel-8  reachable  build 0.64.1"
    )
    assert net_cli._peer_line({**row, "build": {"version": "0.63.2"}}) == (  # noqa: SLF001
        "pixel-8  reachable  build 0.63.2 — behind this device (0.64.1); "
        "ask Local Operator to update it there"
    )
    assert net_cli._peer_line({**row, "build": {"version": "0.66.0"}}) == (  # noqa: SLF001
        "pixel-8  reachable  build 0.66.0"
    )
    # Unknown stays byte-for-byte the old line: `{}`, absent, and unparsable alike.
    assert net_cli._peer_line({**row, "build": {}}) == "pixel-8  reachable"  # noqa: SLF001
    assert net_cli._peer_line({**row, "build": {"version": "0.28.0rc1"}}) == (  # noqa: SLF001
        "pixel-8  reachable"
    )
    # And the suffix never rides an unreachable row.
    assert (
        net_cli._peer_line(  # noqa: SLF001
            {
                **row,
                "reachable": False,
                "reason": "connect_failed:ConnectionRefusedError",
                "build": {"version": "0.63.2"},
            }
        )
        == "pixel-8 cannot be reached from this device right now (it did not answer)"
    )


def test_a_peer_row_with_several_addresses_glosses_the_whole_list(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    """QA round 21, Q-R21-1: the compound reason reached the user verbatim.

    A member row carries EVERY address its declaring device advertises
    (``relay.advertise_endpoints``: the operator's declared hosts, then the live
    ones), so several candidates is a DESIGNED state rather than an accident. When
    they fail DIFFERENTLY the probe reports each one with its own answer —
    ``unreachable: <endpoint> <detail>; …`` — which is the right shape for
    ``--json`` and the wrong one for a person. The line QA measured was::

        device-b cannot be reached from this device right now (127.0.0.1:0
        connect_failed:OSError; 127.0.0.1:39223 connect_failed:ConnectionRefusedError)

    The reason is built here by the REAL producer rather than typed, and rendered
    through the REAL line function, because the defect lived at that seam: the
    gloss decided ``stage: <sentence>`` by "the tail contains a space", which this
    machine list also satisfies (``resume.peer_reason_words``).
    """
    reason = relay.probe_reason(
        [
            relay.CandidateAttempt("127.0.0.1:0", False, "connect_failed:OSError"),
            relay.CandidateAttempt(
                "127.0.0.1:39223", False, "connect_failed:ConnectionRefusedError"
            ),
        ]
    )
    # The wire shape is deliberate, and asserting it here is what keeps the two
    # halves of this test from drifting apart: this is the string --json keeps.
    assert reason == (
        "unreachable: 127.0.0.1:0 connect_failed:OSError; "
        "127.0.0.1:39223 connect_failed:ConnectionRefusedError"
    )
    row = {
        "device_id": "d_" + "1a2b3c4d5e6f7a8b9c0d1e2f3a4b5c6d",
        "name": "device-b",
        "reachable": False,
        "reason": reason,
    }
    line = net_cli._peer_line(row)  # noqa: SLF001
    # The peer and the plain reason, and nothing else.
    assert line == (
        "device-b cannot be reached from this device right now (no address of it answered)"
    )
    # No address, no port, no stage word and no exception class name.
    assert "127.0.0.1" not in line, line
    assert re.search(r":\d+", line) is None, line
    assert "connect_failed" not in line, line
    assert "OSError" not in line, line
    assert "ConnectionRefusedError" not in line, line
    # SHAPE, NOT PREFIX: a compound written with another stage word is the same
    # leak and takes the same gloss, so a future list shape cannot leak either.
    other_shape = "half_broken: 10.0.0.1:7 bad_endpoint; 10.0.0.2:7 no_answer"
    assert net_cli._peer_line({**row, "reason": other_shape}) == (  # noqa: SLF001
        "device-b cannot be reached from this device right now (no address of it answered)"
    )
    # And the token is not lost: it is this verb's --json field, byte for byte.
    monkeypatch.setattr(net_cli, "_relay_call", lambda *a, **k: {"value": [row]})
    assert net_cli._cmd_peers(Namespace(json=True)) == 0  # noqa: SLF001
    payload = json.loads(capsys.readouterr().out)
    assert payload["peers"][0]["reason"] == reason
    assert payload["peers"][0]["device_id"] == row["device_id"]


def test_a_reason_that_names_an_answer_is_not_read_as_a_machine_list() -> None:
    """Round 10, MAJOR-1: the shape test inverted the relay's OWN sentence.

    ``handshake_not_attempted`` is prose that embeds the endpoint that ANSWERED
    (``relay.handshake_not_attempted_reason``; its winner is by construction a bare
    ``host:port``), so a rule that counted a colon-bearing field as a wire token
    read that sentence as a machine list and told the reader "no address of it
    answered" about a device that was talking — the inverse of the truth, in the one
    state whose remedy differs (a peer that never answered versus a listing that gave
    up on our side).

    The reason comes from the REAL producer and the line from the REAL renderer, and
    BOTH directions are asserted here, because the fix that closed this case is the
    one that could un-close QA round 21's: a machine list of any shape still has to
    be glossed whole.
    """
    reason = relay.handshake_not_attempted_reason("127.0.0.1:39223")
    # The address is not lost — it is what --json still carries, byte for byte.
    assert reason == (
        "handshake_not_attempted: 127.0.0.1:39223 answered and the listing budget ran "
        "out before the handshake"
    )
    row = {
        "device_id": "d_" + "1a2b3c4d5e6f7a8b9c0d1e2f3a4b5c6d",
        "name": "device-b",
        "reachable": False,
        "reason": reason,
    }
    line = net_cli._peer_line(row)  # noqa: SLF001
    # The device ANSWERED and OUR budget ran out: that, and not silence.
    assert line == (
        "device-b cannot be reached from this device right now "
        "(it answered, and the listing ran out of time before the handshake)"
    )
    assert "127.0.0.1" not in line, line
    assert "no address of it answered" not in line, line
    assert "handshake_not_attempted" not in line, line

    # The machine list, from the real producer, is still glossed whole — with the
    # SINGLE-endpoint form of it, which is the case where the list and the prose
    # reason look most alike.
    one = relay.probe_reason([relay.CandidateAttempt("10.0.0.1:7", False, "no_answer")])
    assert one == "no_answer"
    assert net_cli._peer_line({**row, "reason": one}) == (  # noqa: SLF001
        "device-b cannot be reached from this device right now (no address of it answered)"
    )
    listed = relay.probe_reason(
        [
            relay.CandidateAttempt("10.0.0.1:7", False, "no_answer"),
            relay.CandidateAttempt("10.0.0.2:7", False, "bad_endpoint"),
        ]
    )
    assert listed == "unreachable: 10.0.0.1:7 no_answer; 10.0.0.2:7 bad_endpoint"
    assert net_cli._peer_line({**row, "reason": listed}) == (  # noqa: SLF001
        "device-b cannot be reached from this device right now (no address of it answered)"
    )


def test_a_peer_that_answered_and_refused_never_reads_as_silence() -> None:
    """QA round 24, Q-R24-1 — round 10's MAJOR-1, one state over.

    ``relay.dial`` writes ``handshake_refused:<ExceptionClass>`` in the ``except`` arm
    AFTER ``probe.sock`` was handed in as ``connected=``, so this reason always means
    the peer's ADDRESS ANSWERED and the link was not made. QA arranged both forms for
    real: a relay in the same network at a later epoch that answers and closes the
    handshake (``ConnectionError``), and a listener that accepts the dial and never
    speaks (``TimeoutError``) — in both, the human line read "it did not answer".

    The bare refusal codes were given "the link was refused" in round 10 for exactly
    this reason; this stage — the ONE arrival in the field whose own NAME says the
    peer answered — was left falling through to the silence default. The reasons are
    built by the REAL producer and the line by the REAL renderer, and the READING is
    asserted: "something other than the code" is what the silence default satisfied.
    """
    row = {
        "device_id": "d_" + "1a2b3c4d5e6f7a8b9c0d1e2f3a4b5c6d",
        "name": "device-b",
        "reachable": False,
    }
    for exc in (ConnectionResetError("the peer closed the link"), TimeoutError("no welcome")):
        reason = relay.handshake_refused_reason(exc)
        # The wire shape, and the ``--json`` string, unchanged: the class name is
        # what was OBSERVED (a close versus a peer that never spoke).
        assert reason == f"handshake_refused:{exc.__class__.__name__}", reason
        words = resume.peer_reason_words(reason)
        assert words == "the link was refused", (reason, words)
        assert words != "it did not answer", reason
        assert exc.__class__.__name__ not in words, words
        assert ":" not in words and "_" not in words, words
        line = net_cli._peer_line({**row, "reason": reason})  # noqa: SLF001
        assert line == (
            "device-b cannot be reached from this device right now (the link was refused)"
        ), line
        assert "handshake_refused" not in line and "Error" not in line, line
    # The bare refusals answer the same way, which is the point of the arm: the
    # families differ in what was observed, not in whether the peer answered.
    for code in ("epoch_stale", "untrusted", "malformed_frame", "protocol_mismatch"):
        assert resume.peer_reason_words(code) == "the link was refused", code
    assert "it did not answer" not in (
        net_cli._peer_line({**row, "reason": "pair_phase_requires_the_ceremony"})  # noqa: SLF001
    )


def test_the_doctor_row_a_person_reads_carries_no_code_and_no_address(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    """QA round 24, Q-R24-2: doctor's human rows printed the raw vocabulary.

    The rows QA captured from the real binary — ``connect_failed:TimeoutError``,
    ``bad_endpoint`` and, on the state the listing's own fix exists for, ``not_attempted:
    127.0.0.1:64994 answered and the doctor budget ran out before the handshake`` —
    carried a stage word, a Python class name AND the endpoint address on a line a
    person reads. The rows are rendered through ``resume.doctor_detail_words`` now
    and the raw strings stay where the machine reads them: ``checks[].detail`` in
    ``--json``, byte for byte.
    """
    checks = [
        {"check": "identity", "ok": True, "detail": "present"},
        {
            "check": "reachability",
            "ok": False,
            "device_id": "d_" + "6" * 32,
            "endpoint": "10.255.255.1:9",
            "latency_ms": 3000.0,
            "detail": f"{relay.CONNECT_FAILED_PREFIX}TimeoutError",
        },
        {
            "check": "reachability",
            "ok": False,
            "device_id": "d_" + "b" * 32,
            "endpoint": "127.0.0.1:64996",
            "latency_ms": 0.2,
            "detail": f"{relay.CONNECT_FAILED_PREFIX}ConnectionRefusedError",
        },
        {
            "check": "reachability",
            "ok": False,
            "device_id": "d_" + "f" * 32,
            "endpoint": "not-a-host-port",
            "detail": "bad_endpoint",
        },
        {
            "check": "handshake",
            "ok": False,
            "device_id": "d_" + "1" * 32,
            "endpoint": "127.0.0.1:64994",
            # The producer's own string: an address ANSWERED while the doctor's budget
            # expired, which is the state this whole line of work is about.
            "detail": relay.handshake_not_attempted_reason("127.0.0.1:64994", budget="doctor"),
        },
    ]
    payload = {"ok": False, "identity_present": True, "checks": checks}
    monkeypatch.setattr(net_cli, "_relay_call", lambda *a, **k: payload)
    assert net_cli._cmd_doctor(Namespace(json=False, peer="")) == 1  # noqa: SLF001
    human = capsys.readouterr().out
    for token in (
        "connect_failed",
        "bad_endpoint",
        "handshake_not_attempted",
        "not_attempted",
        "TimeoutError",
        "ConnectionRefusedError",
        "the doctor budget ran out",
    ):
        assert token not in human, (token, human)
    # What a person is told instead, per row: the stage the dial reached.
    assert "nothing answered at that address" in human
    assert "the address it publishes cannot be dialled" in human
    assert "it answered, and the doctor ran out of time before the handshake" in human
    # The ADDRESS stays in the row's own column once — and only once, which is what
    # the repeated endpoint inside the sentence broke.
    assert human.count("127.0.0.1:64994") == 1, human
    # ``--json`` is the register the raw detail belongs to, and it keeps it whole.
    monkeypatch.setattr(net_cli, "_relay_call", lambda *a, **k: payload)
    assert net_cli._cmd_doctor(Namespace(json=True, peer="")) == 1  # noqa: SLF001
    machine = json.loads(capsys.readouterr().out)
    assert [row["detail"] for row in machine["checks"]] == [row["detail"] for row in checks]
    assert relay.handshake_not_attempted_reason("127.0.0.1:64994", budget="doctor") in (
        machine["checks"][4]["detail"]
    )


def test_the_ready_verb_reads_refused_and_silent_apart(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    """A refused connection is NOT "nothing answered" on this verb.

    The operator's discriminator (the security-group case): a refusal means the
    host is UP and nothing is listening, and doctor's own renderer maps every
    ``connect_failed:`` stage — refusals included — to "nothing answered at
    that address", which sends someone to check a machine that already
    answered. That is why ``ready`` renders reachability rows through
    ``readiness.reachability_reading`` while doctor's tables stay untouched.
    """
    checks: list[dict[str, Any]] = [
        {"check": "identity", "ok": True, "detail": "present"},
        {
            "check": "reachability",
            "device_id": "d_" + "b" * 32,
            "device_name": "cloud-node-1",
            "endpoint": "54.1.2.3:7777",
            "ok": False,
            "detail": f"{relay.CONNECT_FAILED_PREFIX}TimeoutError",
            "observed": {
                "outcome": "no_answer",
                "source_address": "203.0.113.7",
                "interface": "utun4",
                "elapsed_ms": 3000.2,
                "budget_s": 3.0,
                "attempted": True,
                "last_seen_at": 1789.0,
            },
            "remedies": [
                "on cloud-node-1 check it is up (`lop network status --json` there), then check "
                "that any firewall or security group on the path admits this device's address "
                "(203.0.113.7, via utun4)"
            ],
        },
        {
            "check": "reachability",
            "device_id": "d_" + "c" * 32,
            "device_name": "pi-box",
            "endpoint": "10.0.0.9:7777",
            "ok": False,
            "detail": f"{relay.CONNECT_FAILED_PREFIX}ConnectionRefusedError",
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
            "check": "readiness",
            "capability": "operator_authority",
            "device_id": "d_" + "b" * 32,
            "device_name": "cloud-node-1",
            "ok": False,
            "code": "not_installed",
            "detail": (
                "operator authority is not installed on cloud-node-1: an approval that needs "
                "the operator parks until someone installs it"
            ),
            "remedies": ["run `lop operator install` on cloud-node-1 (one privileged step)"],
        },
    ]
    payload = {"ok": False, "identity_present": True, "checks": checks}
    monkeypatch.setattr(net_cli, "_relay_call", lambda *a, **k: payload)
    assert net_cli._cmd_ready(Namespace(json=False, peer="")) == 1  # noqa: SLF001
    human = capsys.readouterr().out
    for token in ("ConnectionRefusedError", "TimeoutError", "connect_failed"):
        assert token not in human, (token, human)
    assert "something answered this address and refused the connection" in human
    assert "the host is up; nothing is listening on that port" in human
    assert "nothing answered this address before the budget ran out" in human
    assert "this device routes to it from 203.0.113.7 via utun4" in human
    assert (
        "FAIL readiness operator_authority cloud-node-1: operator authority is not installed"
        in human
    )
    assert "    - run `lop operator install` on cloud-node-1 (one privileged step)" in human
    # ``--json`` is the register the raw vocabulary belongs to, and it keeps it.
    monkeypatch.setattr(net_cli, "_relay_call", lambda *a, **k: payload)
    assert net_cli._cmd_ready(Namespace(json=True, peer="")) == 1  # noqa: SLF001
    machine = json.loads(capsys.readouterr().out)
    assert [row["detail"] for row in machine["checks"]] == [row["detail"] for row in checks]
    assert machine["ok"] is False and machine["code"] == "unhealthy" and machine["message"]


def test_an_informational_address_does_not_red_the_report_and_a_real_failure_still_does(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    """Drill finding, 2026-10-03: `ready`'s verdict is the row-level ``ok``.

    A remote-unusable advertised address (the node's VPC-private one) flipped
    informational by ``readiness.mark_informational`` must not ride in the
    ``failures`` list — while the row still SHOWS, with the address that is
    usable. And a legitimate uncovered item (an MCP login) keeps the report red
    and is the ONLY row the message names.
    """
    dead: dict[str, Any] = {
        "check": "reachability",
        "device_id": "d_" + "b" * 32,
        "device_name": "cloud-node-1",
        "endpoint": "172.31.22.23:4097",
        "ok": False,
        "detail": "connect_failed:ConnectionRefusedError",
        "observed": {"outcome": "refused", "attempted": True},
        "remedies": [],
    }
    live: dict[str, Any] = {
        "check": "reachability",
        "device_id": "d_" + "b" * 32,
        "device_name": "cloud-node-1",
        "endpoint": "99.79.190.164:4097",
        "ok": True,
        "detail": "ok",
        "observed": {
            "outcome": "connected",
            "winner": "99.79.190.164:4097",
            "winner_verified": True,
        },
        "remedies": [],
    }
    readiness.mark_informational([dead, live])
    assert dead["ok"] is True

    checks: list[dict[str, Any]] = [
        {"check": "identity", "ok": True, "detail": "present"},
        dead,
        live,
    ]
    payload: dict[str, Any] = {"identity_present": True, "checks": checks}
    monkeypatch.setattr(net_cli, "_relay_call", lambda *a, **k: payload)
    assert net_cli._cmd_ready(Namespace(json=True, peer="")) == 0  # noqa: SLF001
    machine = json.loads(capsys.readouterr().out)
    assert machine["ok"] is True
    row = [r for r in machine["checks"] if str(r.get("endpoint", "")).startswith("172.31")][0]
    assert row["ok"] is True and row["observed"]["informational"] is True

    checks.append(
        {
            "check": "readiness",
            "capability": "mcp_login",
            "device_name": "cloud-node-1",
            "ok": False,
            "detail": "no login for https://slack.example",
            "remedies": [],
        }
    )
    payload = {"identity_present": True, "checks": checks}
    monkeypatch.setattr(net_cli, "_relay_call", lambda *a, **k: payload)
    assert net_cli._cmd_ready(Namespace(json=True, peer="")) == 1  # noqa: SLF001
    machine = json.loads(capsys.readouterr().out)
    assert machine["ok"] is False and machine["code"] == "unhealthy"
    assert "slack.example" in machine["message"]
    assert "172.31.22.23" not in machine["message"]


def test_doctor_marks_a_remote_unusable_address_informational_and_keeps_repairs_red(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    """Drill finding, 2026-10-03, doctor's half: the same run's private-address
    reachability row must not red a healthy mesh — while the credential-repair
    rows for dead owner logins stay visible and keep the report red."""
    dead: dict[str, Any] = {
        "check": "reachability",
        "device_id": "d_" + "b" * 32,
        "endpoint": "172.31.22.23:4097",
        "ok": False,
        "detail": "connect_failed:ConnectionRefusedError",
    }
    handshake: dict[str, Any] = {
        "check": "handshake",
        "device_id": "d_" + "b" * 32,
        "endpoint": "99.79.190.164:4097",
        "ok": True,
        "detail": "ok",
    }
    readiness.mark_informational([dead, handshake])
    assert dead["ok"] is True

    checks: list[dict[str, Any]] = [
        {"check": "identity", "ok": True, "detail": "present"},
        dead,
        handshake,
    ]
    payload: dict[str, Any] = {"identity_present": True, "checks": checks}
    monkeypatch.setattr(net_cli, "_relay_call", lambda *a, **k: payload)
    assert net_cli._cmd_doctor(Namespace(json=True, peer="")) == 0  # noqa: SLF001
    machine = json.loads(capsys.readouterr().out)
    assert machine["ok"] is True
    row = [r for r in machine["checks"] if str(r.get("endpoint", "")).startswith("172.31")][0]
    assert row["ok"] is True and row["observed"]["informational"] is True

    # A dead owner login stays visible AND red; the message names it and never
    # the informational address.
    checks.append(
        {
            "check": "credential_repair",
            "device_id": "d_" + "b" * 32,
            "ok": False,
            "detail": "the owner's minerva-qa login died; the owner must sign in again",
        }
    )
    payload = {"identity_present": True, "checks": checks}
    monkeypatch.setattr(net_cli, "_relay_call", lambda *a, **k: payload)
    assert net_cli._cmd_doctor(Namespace(json=True, peer="")) == 1  # noqa: SLF001
    machine = json.loads(capsys.readouterr().out)
    assert machine["ok"] is False and machine["code"] == "unhealthy"
    assert "minerva-qa" in machine["message"]
    assert "172.31.22.23" not in machine["message"]

    # The human line names the address that works, same treatment as ready's.
    assert net_cli._cmd_doctor(Namespace(json=False, peer="")) == 1  # noqa: SLF001
    human = capsys.readouterr().out
    assert "the peer is reachable at 99.79.190.164:4097" in human


def test_ready_without_a_relay_never_passes_a_check_it_could_not_run(
    root: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    """No relay: the member rows say NOT PROBED and the capabilities say not asked.

    A check that was never RUN did not pass. The rc still follows the checks
    (``ok`` is the verdict, not "the command ran"), and ``--peer`` filters by
    the name a person types even on this no-dial path.
    """
    monkeypatch.setenv("LOCAL_OPERATOR_CONFIG_DIR", str(root))
    from local_operator.network import identity as identity_mod

    identity_mod.mint(root, name="here")
    record = types.NetworkRecord(
        network_id=NETWORK,
        name="home-net",
        self_device_id="d_" + "a" * 32,
        self_role="admin",
        self_capabilities=sorted(types.capabilities_for_role("admin")),
    )
    record.members.append(
        types.MemberRecord(
            device_id="d_" + "b" * 32, name="cloud-node-1", endpoints=["127.0.0.1:9"]
        )
    )
    store.save(record, root)

    monkeypatch.setattr(net_cli, "_relay_call", lambda *a, **k: None)
    assert net_cli._cmd_ready(Namespace(json=True, peer="")) == 1  # noqa: SLF001
    payload = json.loads(capsys.readouterr().out)
    reach = [row for row in payload["checks"] if row["check"] == "reachability"]
    assert reach and reach[0]["probed"] is False
    assert "not probed" in reach[0]["detail"]
    caps = [row for row in payload["checks"] if row["check"] == "readiness"]
    assert {row["capability"] for row in caps} == {
        readiness.CAPABILITY_BUILD,
        *readiness.PEER_SIDE_CHECKS,
    }
    assert all(row["ok"] is False and row["code"] == "not_asked" for row in caps)
    assert payload["ok"] is False and payload["code"] == "unhealthy"

    # The name filter reaches this path too: a peer that is not named is absent
    # from the rows rather than reported on under a filter that was ignored.
    monkeypatch.setattr(net_cli, "_relay_call", lambda *a, **k: None)
    net_cli._cmd_ready(Namespace(json=True, peer="nobody"))  # noqa: SLF001
    filtered = json.loads(capsys.readouterr().out)
    assert not [row for row in filtered["checks"] if row["check"] == "reachability"]


def test_the_federated_listing_header_lines_up_with_its_rows() -> None:
    """Review round 9, NIT: the header was a two-space list of labels.

    With unpadded rows, each label sat wherever its own length left it, so nothing
    lined up with the values under it — and at 64 columns the header is what a user
    reads as the columns (the local ``lop sessions`` table has always padded,
    ``cli.STATE_COLUMN_WIDTH``'s line).

    NOTHING IS CUT, which is why the columns are measured from the rows rather than
    fixed: three of these cells are identity this process did not author — a peer's
    session id, a peer's device name, a conversation title — and a cut id makes two
    rows indistinguishable. The long id below is the length the suite's own fixture
    uses (``_UNSEEN_ROW`` is 19 characters, and a fixed twelve-cell column cut it),
    the wide name is a peer's, and both are asserted whole with the columns after
    them asserted not to move.
    """
    from rich.cells import cell_len

    from local_operator import cli as local_cli

    rows = [
        ("qr7-1-stored-unseen", "東京のマシン", "not running", "Auditing the mesh"),
        ("9f3ac1e0b7d2", "pixel-8", "live", ""),
    ]
    header, *laid_out = net_cli._session_plane_lines(rows)  # noqa: SLF001

    def column_at(line: str, needle: str) -> int:
        """The CELL offset ``needle`` starts at, which is what a column is."""
        return cell_len(line[: line.index(needle)])

    for line, row in zip(laid_out, rows):
        for label, cell in zip(net_cli.SESSIONS_COLUMNS, row):
            if not cell:
                # The conversation is the last column and is not padded, so a row
                # without one has no offset to compare against.
                continue
            assert column_at(header, label) == column_at(line, cell), (label, header, line)
        # Identity is printed WHOLE: the id and the name are not cut to a column.
        assert row[0] in line, line
        assert row[1] in line, line
    # And the local table's own widths are the floor for the two columns the two
    # listings share, so a short listing renders in the shape the sibling verb uses.
    device_width = column_at(header, "STATE") - column_at(header, "DEVICE") - 2
    state_width = column_at(header, "CONVERSATION") - column_at(header, "STATE") - 2
    assert device_width >= local_cli.PEER_COLUMN_WIDTH
    assert state_width >= local_cli.STATE_COLUMN_WIDTH


def test_the_session_planes_state_token_is_said_in_words() -> None:
    """U29's other half: ``stored`` is the catalogue's token, not a sentence.

    ``live`` passes through unchanged on purpose: an unrecognised token is not
    evidence of "not running", and inventing a word for it would be the same
    defect one state over.
    """
    from local_operator.resume import session_state_words

    assert session_state_words("stored") == "not running"
    assert session_state_words("live") == "live"
    assert session_state_words("") == ""


# ---------------------------------------------------------------------------
# The incident receipts, as the CLI renders them
# ---------------------------------------------------------------------------


def _panic_args(**overrides: Any) -> Namespace:
    base = {
        "action": "panic",
        "network": "home-net",
        "json": False,
    }
    base.update(overrides)
    return Namespace(**base)


def test_a_refused_panic_receipt_is_not_a_success_and_names_the_peer(
    root: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    """Q-R1-2's half that lives in the CLI: the receipt is the PEER's, and `ok` is theirs.

    The relay now reports per-peer outcomes; this pins what the operator actually sees
    — a refusing peer is named with its own sentence, `ok` is false so a script can
    branch, and a peer that took it contributes no noise. The relay-side behaviour is
    driven on a real pair in ``test_relay_e2e.py``; this is the rendering contract.
    """
    record = _record_in(root)
    monkeypatch.setattr(net_cli, "_resolve", lambda _target: record)
    monkeypatch.setattr(
        net_cli,
        "_relay_call",
        lambda op, **fields: {
            "network_id": NETWORK,
            "epoch": 2,
            "rotated": True,
            "sent": 2,
            "acked": 1,
            "unacked": [],
            "refused": ["d_" + "b" * 32],
            "failed": [],
            "ok": False,
            "peers": [
                {
                    "device_id": "d_" + "a" * 32,
                    "name": "laptop",
                    "outcome": "acked",
                    "reason": "",
                },
                {
                    "device_id": "d_" + "b" * 32,
                    "name": "phone",
                    "outcome": "refused",
                    "reason": "this link authenticated at the previous epoch, so only "
                    "net_reconcile and ping may dispatch",
                },
            ],
        },
    )
    rc = net_cli._cmd_panic(_panic_args())  # noqa: SLF001 — the CLI's own entry point
    out = capsys.readouterr().out
    assert rc == 1, "a panic one peer refused must not exit 0"
    assert "peers told: 1 of 2 acted on it" in out, out
    assert "phone" in out, out
    assert "net_reconcile" in out, "the peer's own sentence is what a person reads"
    assert "laptop" not in out, out

    monkeypatch.setattr(
        net_cli,
        "_relay_call",
        lambda op, **fields: {
            "network_id": NETWORK,
            "epoch": 2,
            "rotated": True,
            "sent": 1,
            "acked": 1,
            "unacked": [],
            "refused": [],
            "failed": [],
            "ok": True,
            "peers": [
                {"device_id": "d_" + "a" * 32, "name": "laptop", "outcome": "acked", "reason": ""}
            ],
        },
    )
    rc = net_cli._cmd_panic(_panic_args(json=True))  # noqa: SLF001
    payload = json.loads(capsys.readouterr().out)
    assert rc == 0, payload
    assert payload["acked"] == 1 and payload["peers"][0]["outcome"] == "acked", payload


def _record_in(root: Path) -> types.NetworkRecord:
    """A minimal local record: these cells exercise the CLI's rendering, not a relay."""
    return types.NetworkRecord(
        network_id=NETWORK,
        name="home-net",
        epoch=2,
        self_device_id="d_" + "c" * 32,
        self_role="admin",
        self_capabilities=sorted(types.capabilities_for_role("admin")),
    )


# ---------------------------------------------------------------------------
# Design round 3 D40 — the audit trail's state, on the block a person reads
# ---------------------------------------------------------------------------


def _audit_status(root: Path, log: Any, **over: Any) -> dict[str, Any]:
    """``relay.status()``'s shape, with the audit block read off a REAL writer.

    The transport is stubbed; nothing else is. ``published_through`` /
    ``recorded_through`` / ``degraded`` are the writer's own properties, so these cells
    cannot keep passing after the writer's meaning of them moves — the thing a
    hand-typed fixture would let happen (the counters are the whole subject here).
    """
    payload: dict[str, Any] = {
        "installed": True,
        "supported": True,
        "identity_present": True,
        "relay_running": True,
        "relay_answering": True,
        "relay_state": "live",
        "relay": {
            "pid": 4711,
            "audit_published_through": log.published_through,
            "audit_recorded_through": log.recorded_through,
            "audit_degraded": log.degraded,
            "audit_degraded_reason": log.degraded_reason,
            "audit_path": str(audit_mod.audit_path(root)),
        },
        "record": {"pid": 4711},
        "port": 4711,
        "log": str(root / "network.log"),
        "networks": [],
    }
    payload.update(over)
    return payload


def test_the_status_block_prints_the_audit_lag_in_the_readers_words(
    root: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    """D40: a reader has to be able to tell "not yet written" from "no such row".

    The distinction existed for two releases and was reachable only through ``--json``,
    so on this block — the one an operator runs first during an incident — a lagging
    writer and a healthy one were the same picture: nothing. The pair is printed here in
    the payload's own vocabulary (D42: ``file``/``buffered``/``unknown`` stay in the
    Python API), and the lag is worded as the batching it is rather than as a loss.

    The two states come from ONE real writer, one record apart, which is what the
    middle state is: a row that is recorded and still buffered behind the tick.
    """
    monkeypatch.setenv("LOCAL_OPERATOR_CONFIG_DIR", str(root))
    log = audit_mod.AuditLog(root)
    log.record(audit_mod.AuditEvent(event="link_opened"))
    log.record(audit_mod.AuditEvent(event="link_closed"))
    log.flush()

    monkeypatch.setattr(relay, "status", lambda *a, **k: _audit_status(root, log))
    assert net_cli._cmd_status(Namespace(json=False)) == 0  # noqa: SLF001
    steady = capsys.readouterr().out
    assert "audit:      2 recorded, published through 2" in steady, steady
    # The same register as the three lines above it: every value starts at cell 12.
    audit_line = next(line for line in steady.splitlines() if line.startswith("audit:"))
    assert audit_line.index("2 recorded") == 12, audit_line

    # One more record, no flush: the writer holds it, and the block says so.
    log.record(audit_mod.AuditEvent(event="link_opened"))
    assert net_cli._cmd_status(Namespace(json=False)) == 0  # noqa: SLF001
    lagging = capsys.readouterr().out
    assert "audit:      3 recorded, published through 2 (1 not yet written)" in lagging, lagging


def test_an_older_relays_answer_prints_no_audit_line_rather_than_zero(
    root: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    """Absence is not zero, on the human path as well as in the payload.

    A relay of an adjacent build answers ``status`` during an update and carries no
    counters. A reader that defaulted them to ``0`` would render every row as "not yet
    written" — the one answer that is wrong in a way the reader cannot detect — so the
    line is omitted instead, which is honest about having nothing to say.
    """
    monkeypatch.setenv("LOCAL_OPERATOR_CONFIG_DIR", str(root))
    log = audit_mod.AuditLog(root)
    payload = _audit_status(root, log)
    payload["relay"] = {"pid": 4711}  # an answering relay with no counters
    monkeypatch.setattr(relay, "status", lambda *a, **k: payload)
    assert net_cli._cmd_status(Namespace(json=False)) == 0  # noqa: SLF001
    out = capsys.readouterr().out
    assert "audit:" not in out, out
    assert "relay:      running, pid 4711" in out, out


def test_a_wedged_relay_gets_a_sentence_where_the_audit_numbers_would_be(
    root: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    """THE CASE THE FINDING WAS MEASURED ON (D40, a ``SIGSTOP``ped relay).

    With the relay running but not answering, the payload carries ``relay: null`` and no
    ``audit*`` key at all — so a reader who has just found a row missing from
    ``audit.jsonl`` is in exactly the state where the numbers vanish, and vanishing is
    indistinguishable from health unless the surface says why. This cell pins the
    sentence, and pins it directly under the relay line that says the same thing about
    the same process.
    """
    monkeypatch.setenv("LOCAL_OPERATOR_CONFIG_DIR", str(root))
    log = audit_mod.AuditLog(root)
    payload = _audit_status(root, log, relay=None, relay_answering=False, relay_state="wedged")
    monkeypatch.setattr(relay, "status", lambda *a, **k: payload)
    assert net_cli._cmd_status(Namespace(json=False)) == 0  # noqa: SLF001
    out = capsys.readouterr().out
    lines = out.splitlines()
    relay_at = next(n for n, line in enumerate(lines) if line.startswith("relay:"))
    audit_at = next(n for n, line in enumerate(lines) if line.startswith("audit:"))
    assert audit_at == relay_at + 1, out
    assert lines[relay_at].endswith("NOT answering its control socket (state: wedged)"), out
    assert "not answering; audit.jsonl holds the last state" in lines[audit_at], out


def test_a_failed_audit_write_reads_degraded_before_anything_else_on_the_block(
    root: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    """The LOSS signal, on the surface an operator opens first.

    ``AuditLog.record``'s contract is that a failed write is "a degraded flag plus a line
    on stderr" — and the stderr of a background relay is not where anyone looks. The
    failure here is real (the writer's own path is occupied by a directory), so both the
    flag and the reason are the writer's, and the reason printed is the writer's own
    string: an operator acts on errno, not on a summary of it.
    """
    monkeypatch.setenv("LOCAL_OPERATOR_CONFIG_DIR", str(root))
    audit_mod.audit_path(root).parent.mkdir(parents=True, exist_ok=True)
    audit_mod.audit_path(root).mkdir()
    log = audit_mod.AuditLog(root)
    log.record(audit_mod.AuditEvent(event="link_opened"))
    log.flush()
    assert log.degraded is True, "the rig did not make the writer fail; the cell proves nothing"

    monkeypatch.setattr(relay, "status", lambda *a, **k: _audit_status(root, log))
    assert net_cli._cmd_status(Namespace(json=False)) == 0  # noqa: SLF001
    out = capsys.readouterr().out
    assert "audit:      DEGRADED" in out, out
    assert log.degraded_reason in out, (log.degraded_reason, out)


def test_join_accepts_the_advertise_host_the_config_route_used_to_own() -> None:
    """``join --advertise-host``: the joiner can name its own address, like ``init`` can.

    The asymmetry was the bug's other face. ``init --advertise-host`` could declare a
    tunnel or public address at creation, and the mesh docs' remedy for anyone else
    was to hand-edit ``network.advertise_hosts`` into ``config.yml`` — a key with no
    registry row that took no effect and was deleted by the next launch. The device
    that most needs this is the JOINER: it is usually joining precisely because it has
    no dialable address of its own, and without declaring one its member row carried
    nothing every peer could dial (``no_endpoint``).

    The parser is asserted here, and ``_join_one``'s use of it is proven end to end by
    the two-device pairing run on this PR: ``declared_hosts`` is threaded into
    ``advertise_endpoints``, whose ordering is pinned in ``test_addresses``.
    """
    parsed = _parser().parse_args(
        ["network", "join", "tok", "--advertise-host", "203.0.113.7:4097"]
    )

    assert parsed.advertise_hosts == ["203.0.113.7:4097"]
    # REPEATABLE, and the default is empty: an operator with two reachable paths (a
    # tunnel and a LAN address) declares both, in the order they should be tried.
    both = _parser().parse_args(
        [
            "network",
            "join",
            "tok",
            "--advertise-host",
            "tunnel.example.com:4100",
            "--advertise-host",
            "203.0.113.7:4097",
        ]
    )
    assert both.advertise_hosts == ["tunnel.example.com:4100", "203.0.113.7:4097"]
    assert _parser().parse_args(["network", "join", "tok"]).advertise_hosts == []


def test_the_printed_join_command_does_not_pin_an_endpoint_the_token_carries(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str], tmp_path: Path
) -> None:
    """The receipt must print a command that WORKS when it is followed verbatim.

    It used to print ``join @token --host {hosts[0]}``, and ``hosts[0]`` is the record's
    own snapshot — the entry kept ahead of the live ones by design, which can name a port
    the listener does not hold. Following the advice therefore dialled the dead entry with
    a pin, while the plain ``join @token`` (which walks the whole list) succeeded: the
    product instructed the operator to run a failing command (QA round 2, Q-3).
    ``--host`` is an override, so the receipt no longer re-types an endpoint the token
    already carries; the placeholder stays for the state where the token names none,
    because there the flag is exactly what the join asks for.

    The command also names the token FILE rather than the path it happens to have on this
    machine: the line is addressed to another device, which does not share this one's
    ``$HOME`` (design review round 1, D3).

    And it says the file has to TRAVEL, because that is the half the reader cannot infer:
    the command alone describes one step where the trip is two, and the file is the only
    channel a TUI client has — ``--print`` is refused on a non-TTY stdout, which is every
    front end of this family. The clause was lost when a fold of main re-wrote this line
    for the path above, so it is restored here by design round 2's D2-1; the placeholder
    stays, and the two answer different halves of the same sentence.
    """
    token = tmp_path / "inv1.invite"
    token.write_text("TOKEN", encoding="utf-8")
    payload = {
        "invite_id": "inv1",
        "path": str(token),
        "role": "read",
        "expires_in_s": 600.0,
        "hosts": ["127.0.0.1:4197", "192.168.0.155:47800"],
    }
    monkeypatch.setattr(net_cli, "_relay_call", lambda *a, **k: dict(payload))

    args = Namespace(
        network="", role="read", expires=600.0, hosts="", device="", print_token=False, json=False
    )
    assert net_cli._cmd_invite(args) == 0  # noqa: SLF001
    printed = capsys.readouterr().out
    # THE JOIN LINE IS A PLACEHOLDER, and the inviter's own `$HOME` must not appear under
    # the words "on the other device": that device does not have this path, and printing it
    # as though the two shared a filesystem hands one device's home directory to another
    # (design review round 1, D3). The concrete path stays exactly ONCE — on the line above,
    # advice for the machine that really holds the file. The clause that says the file has
    # to be carried over is on it too (design round 2, D2-1).
    assert (
        "then, on the other device: carry that file over and run lop network join @<token-file>"
        in printed
    ), printed
    assert printed.count(str(token)) == 1, printed
    assert "--host 127.0.0.1:4197" not in printed, printed
    assert "--host" not in printed, printed

    # NOTHING NAMES AN ENDPOINT: the flag is what the join will ask for, so it is named.
    monkeypatch.setattr(net_cli, "_relay_call", lambda *a, **k: {**payload, "hosts": []})
    assert net_cli._cmd_invite(args) == 0  # noqa: SLF001
    empty = capsys.readouterr().out
    assert "--host <this device's address:port>" in empty, empty


# ---------------------------------------------------------------------------
# `lop network credentials`: the device-level shareable ledger (design §2)
# ---------------------------------------------------------------------------

SLACK_URL = "https://hooks.slack.com/services/T000/B000/XXXX"
NOTION_URL = "https://mcp.notion.com/mcp"


class _FakeCredentialStore:
    """The one read ``has_stored_row`` makes, without an auth.db on disk."""

    def __init__(self, rows: list[Any]) -> None:
        self._rows = rows

    def list_credentials(self, provider: str) -> list[Any]:
        return list(self._rows)

    def close(self) -> None:
        pass


def _mcp_login_row(url: str) -> Any:
    from types import SimpleNamespace

    return SimpleNamespace(id=1, identity_key=url, updated_at=0)


def _radient_login_row(email: str = "owner@example.test") -> Any:
    """A provider-login row as ``list_credentials(None)`` hands it over.

    The shape is the store's ``StoredCredential`` reduced to what the ledger reads
    (``provider``/``credential_type``/``data``). ``identity_key`` is deliberately
    empty: ``McpTokenStorage.has_stored_row`` matches on it, and a provider row must
    never read as an MCP server's own login.
    """
    from types import SimpleNamespace

    return SimpleNamespace(
        id=2,
        provider="radient",
        credential_type="oauth",
        data={"email": email},
        identity_key="",
        updated_at=0,
    )


def _share_fixture(root: Path, monkeypatch: pytest.MonkeyPatch) -> tuple[str, str]:
    """One network, this device holding the slack login and sharing it with a peer.

    Returns ``(self_device, peer_device)``. The mcp.json declares three servers —
    an http one with a login, an http one without, and a stdio one — so the
    matrix (login x shared x transport) is exercised in one pass, and the stdio
    row proves the exclusion.
    """
    monkeypatch.setenv("LOCAL_OPERATOR_CONFIG_DIR", str(root))
    (root / "mcp.json").write_text(
        json.dumps(
            {
                "mcpServers": {
                    "slack": {"type": "http", "url": SLACK_URL},
                    "notion": {"type": "http", "url": NOTION_URL},
                    "fs": {"command": "npx", "args": ["-y", "x"]},
                }
            }
        ),
        encoding="utf-8",
    )
    from local_operator.network import identity as identity_mod
    from local_operator.network.credentials import placement as placement_mod

    self_device = identity_mod.load_or_mint().device_id
    peer_device = "d_" + "b" * 32
    record = types.NetworkRecord(
        network_id=NETWORK, name="home", self_device_id=self_device, self_role="admin"
    )
    record.members.append(types.MemberRecord(device_id=self_device, name="this-device"))
    record.members.append(types.MemberRecord(device_id=peer_device, name="cloud-node-1"))
    store.save(record, root)
    document = placement_mod.PlacementDocument(NETWORK, root=root)
    key = f"mcp:{SLACK_URL}"
    document.declare(key, owner_device=self_device, owner_device_name="this-device", by=self_device)
    document.grant(key, peer_device, scope="session", by=self_device)
    document.save()
    return self_device, peer_device


def test_credentials_lists_the_shareable_ledger_per_server(
    root: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    """The 2026-09-28 offload gap: a login held here but shared nowhere had no row
    on any surface until `credential share` refused it at share time. The ledger
    answers up front, and its ``--json`` is the contract the tool digest reads.

    PROVIDER LOGINS ARE ROWS HERE TOO (Radient org projection): the store holds a
    Radient OAuth login, so the same listing carries a ``provider`` row beside the
    ``server`` rows — the row the operator needed to see before the share existed."""
    self_device, peer_device = _share_fixture(root, monkeypatch)
    monkeypatch.setattr(
        readiness,
        "_open_store",
        lambda _root: _FakeCredentialStore(
            [_mcp_login_row(SLACK_URL), _radient_login_row("owner@example.test")]
        ),
    )
    assert net_cli._cmd_credentials(Namespace(json=True, network="")) == 0  # noqa: SLF001
    payload = json.loads(capsys.readouterr().out)
    shareable = payload["shareable"]
    # Servers sort by name; the provider rows follow, sorted by provider, and the
    # ledger follows that order.
    assert [row.get("server") or row.get("provider") for row in shareable] == [
        "notion",
        "slack",
        "radient",
    ]
    notion, slack, radient = shareable
    assert slack == {
        "server": "slack",
        "url": SLACK_URL,
        "transport": "http",
        "login_here": True,
        "shared_with": [{"device": peer_device, "name": "cloud-node-1", "scope": "session"}],
        "remedy": f"lop network credential share mcp:{SLACK_URL} --with <device>",
    }
    assert notion == {
        "server": "notion",
        "url": NOTION_URL,
        "transport": "http",
        "login_here": False,
        "shared_with": [],
        "remedy": f"run '/mcp login {NOTION_URL}' here first",
    }
    # The provider row: kind and label from the same classifier the share records
    # with, the held sentence's remedy, no share yet (nothing declares it).
    assert radient == {
        "provider": "radient",
        "kind": "oauth-rotating",
        "identity_label": "owner@example.test",
        "shared_with": [],
        "remedy": "lop network credential share radient --with <device>",
    }
    # stdio servers have no login to share and are excluded entirely.
    assert all(row.get("server") != "fs" for row in shareable)
    # And no field name carries a scrubber marker, so the tool digest cannot eat
    # the row's facts on its way to a model.
    for row in shareable:
        for field_name in row:
            lowered = field_name.lower()
            assert not any(
                marker in lowered for marker in ("token", "secret", "password")
            ), field_name


def test_credentials_shareable_block_renders_login_states_and_shares(
    root: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    """The human register: the three row kinds, the one shipped login sentence,
    and the existing nest for a share that already exists."""
    _share_fixture(root, monkeypatch)
    monkeypatch.setattr(
        readiness,
        "_open_store",
        lambda _root: _FakeCredentialStore([_mcp_login_row(SLACK_URL), _radient_login_row()]),
    )
    assert net_cli._cmd_credentials(Namespace(json=False, network="")) == 0  # noqa: SLF001
    out = capsys.readouterr().out
    assert "home:" in out
    assert "shareable here:" in out
    assert (
        f"  slack  http  login held — share: lop network credential share mcp:{SLACK_URL}"
        " --with <device>" in out
    ), out
    assert "      shared with cloud-node-1 (session)" in out, out
    assert (
        "  radient  oauth-rotating  login held — share: lop network credential share radient"
        " --with <device>" in out
    ), out
    assert "      organization account — share only to your own devices" in out, out
    assert "      signed in as owner@example.test" in out, out
    assert (
        f"  notion  http  no login here yet — run '/mcp login {NOTION_URL}' here first" in out
    ), out
    assert "fs" not in out


def test_credentials_shareable_state_is_three_valued(
    root: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    """An unreadable store is NOT "no login" — the same rule ``has_stored_row``
    states for the share verb; the ledger must not send the operator to sign in
    when it simply could not read."""
    monkeypatch.setenv("LOCAL_OPERATOR_CONFIG_DIR", str(root))
    (root / "mcp.json").write_text(
        json.dumps({"mcpServers": {"slack": {"type": "http", "url": SLACK_URL}}}),
        encoding="utf-8",
    )

    class _Unreadable:
        def list_credentials(self, provider: str) -> Any:
            raise OSError("store unreadable")

        def close(self) -> None:
            pass

    monkeypatch.setattr(readiness, "_open_store", lambda _root: _Unreadable())
    assert net_cli._cmd_credentials(Namespace(json=True, network="")) == 0  # noqa: SLF001
    payload = json.loads(capsys.readouterr().out)
    assert payload["shareable"][0]["login_here"] is None
    assert payload["shareable"][0]["remedy"] == ""
    assert net_cli._cmd_credentials(Namespace(json=False, network="")) == 0  # noqa: SLF001
    out = capsys.readouterr().out
    assert "no login here yet" not in out
    assert "login state not known — this device's credential store could not be read" in out, out


def test_credentials_shareable_read_creates_nothing(
    root: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    """The ledger is an observation: no declare, no placement write, no auth.db —
    the same read-only discipline the readiness report's own cell pins."""
    monkeypatch.setenv("LOCAL_OPERATOR_CONFIG_DIR", str(root))
    (root / "mcp.json").write_text(
        json.dumps({"mcpServers": {"slack": {"type": "http", "url": SLACK_URL}}}),
        encoding="utf-8",
    )
    before = sorted(str(path.relative_to(root)) for path in root.rglob("*"))
    assert net_cli._cmd_credentials(Namespace(json=True, network="")) == 0  # noqa: SLF001
    capsys.readouterr()
    after = sorted(str(path.relative_to(root)) for path in root.rglob("*"))
    assert after == before, f"the listing created files: {sorted(set(after) - set(before))}"


def test_the_provider_ledger_read_creates_nothing(
    root: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    """The PROVIDER half of the ledger is an observation too (Radient org projection).

    The cell above pins the no-store case; this pins the store-PRESENT case the new
    provider arm added: with an ``auth.db`` on disk, a listing must not add a file
    (no placement, no observation state, no new sidecar) and must not write a row.
    The first read is a WARM-UP — SQLite may leave sidecars behind on first open —
    and the comparison starts from there, so what is pinned is that reads add
    nothing of their OWN.
    """
    monkeypatch.setenv("LOCAL_OPERATOR_CONFIG_DIR", str(root))
    from local_operator.providers.auth_store import AuthStore

    auth = AuthStore(db_path=root / "auth.db", config_dir=root)
    auth.upsert_credential(
        "radient",
        {
            "type": "oauth",
            "access": "access-fixture",
            "refresh": "refresh-fixture",
            "expires": int(time.time() * 1000) + 3600_000,
            "email": "owner@example.test",
        },
    )
    auth.close()
    assert net_cli._cmd_credentials(Namespace(json=True, network="")) == 0  # noqa: SLF001
    payload = json.loads(capsys.readouterr().out)
    assert [row.get("provider") for row in payload["shareable"]] == ["radient"]
    before = sorted(str(path.relative_to(root)) for path in root.rglob("*"))
    assert net_cli._cmd_credentials(Namespace(json=True, network="")) == 0  # noqa: SLF001
    capsys.readouterr()
    after = sorted(str(path.relative_to(root)) for path in root.rglob("*"))
    assert after == before, f"the listing created files: {sorted(set(after) - set(before))}"


def test_sharing_radient_records_the_oauth_login_over_an_older_key(
    root: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    """R1: the placement a share writes names the OAuth row, not an older pasted key.

    ``radient-key`` aliases into ``"radient"``, so a store can hold BOTH logins under
    one provider, and ``list_credentials`` is ``ORDER BY id``. Reading the oldest row
    recorded ``api-key-static``/``""`` for a store whose org calls are served from the
    OAuth row — a document the broker narrows on, wrong about the login it describes.
    Driven through the REAL parser, as the operator types it.
    """
    monkeypatch.setenv("LOCAL_OPERATOR_CONFIG_DIR", str(root))
    self_device, peer_device = _share_fixture(root, monkeypatch)
    # The fixture's record is hand-built (never admitted), so its self MEMBER row
    # carries no capabilities; the share's own capability write needs the admin one
    # a real admission would have written.
    record = store.load(NETWORK, root)
    member = record.member(self_device)
    assert member is not None
    member.capabilities = sorted(types.capabilities_for_role("admin"))
    store.save(record, root)

    from local_operator.providers.auth_store import AuthStore

    auth = AuthStore(db_path=root / "auth.db", config_dir=root)
    try:
        auth.upsert_credential(
            "radient", {"type": "api_key", "source": "login", "key": "pasted-fixture"}
        )
        auth.upsert_credential(
            "radient",
            {
                "type": "oauth",
                "access": "access-fixture",
                "refresh": "refresh-fixture",
                "expires": int(time.time() * 1000) + 3600_000,
                "email": "owner@example.test",
            },
        )
    finally:
        auth.close()

    rc = net_cli.main(
        _parser().parse_args(
            ["network", "credential", "share", "radient", "--with", "cloud-node-1"]
        )
    )
    assert rc == 0
    capsys.readouterr()

    from local_operator.network.credentials import placement as placement_mod

    document = placement_mod.PlacementDocument.load(NETWORK, root, self_device=self_device)
    entry = document.entry("radient")
    assert entry is not None, "the share wrote no placement entry"
    assert entry.provider == "radient"
    assert entry.kind == "oauth-rotating"
    assert entry.identity_label == "owner@example.test"


def test_a_refused_share_does_not_create_a_store(
    root: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The share path's reads are GATED ON THE STORE'S EXISTENCE (review round 1,
    MINOR; QA round 1, Q1): on a device that never signed in, a refused share
    must not be the reason an ``auth.db`` appears. Red on base — the ambient
    ``AuthStore`` / ``McpTokenStorage`` constructions wrote one before the
    refusal; the same class ``readiness._mcp_row_exists`` closed for the ledger.
    """
    monkeypatch.setenv("LOCAL_OPERATOR_CONFIG_DIR", str(root))
    assert not (root / "auth.db").exists()
    # (1) The shape read the share verb runs first. The classifier moved to
    # ``credentials.offers`` (one implementation for the ledger, the share verb
    # and the join-time offer); its read-only promise is the same one this cell
    # has always pinned.
    shape = offers.shape_for_key(f"mcp:{NOTION_URL}", root)
    assert shape == ("mcp-rotating", "mcp-oauth", "")
    assert not (root / "auth.db").exists(), "the shape read wrote a store"
    # (2) The refusal itself.
    with pytest.raises(types.MeshRefusal) as refusal:
        net_cli._require_local_credential(f"mcp:{NOTION_URL}", "mcp-oauth")  # noqa: SLF001
    assert "no MCP login" in refusal.value.sentence
    assert not (root / "auth.db").exists(), "the refusal wrote a store"
    # (3) The provider read both halves of the verb share.
    assert offers.provider_rows("openai", root) == []
    assert not (root / "auth.db").exists(), "the provider read wrote a store"


def test_a_damaged_store_degrades_instead_of_raising(
    root: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A store that EXISTS but cannot be opened must not raise out of the share
    path (convergence round 2, MAJOR): ``AuthStore`` connects eagerly, so the
    open-first construction has to answer ``None`` for an unopenable database and
    let every caller render its own degrade (default shape / refusal / no rows) —
    ``network.cli.main`` re-raises non-MeshRefusals, so a leaked DatabaseError
    lands ``credential share`` on the stack-trace panel where base degraded.
    """
    import os

    monkeypatch.setenv("LOCAL_OPERATOR_CONFIG_DIR", str(root))
    database = root / "auth.db"
    for kind in ("corrupt", "locked"):
        if database.exists():
            os.chmod(database, 0o600)
        if kind == "corrupt":
            database.write_bytes(b"not a database")
        else:
            database.write_bytes(b"")
            os.chmod(database, 0o000)
        try:
            shape = offers.shape_for_key(f"mcp:{NOTION_URL}", root)
            assert shape == ("mcp-rotating", "mcp-oauth", ""), kind
            with pytest.raises(types.MeshRefusal) as refusal:
                net_cli._require_local_credential(f"mcp:{NOTION_URL}", "mcp-oauth")  # noqa: SLF001
            assert "no MCP login" in refusal.value.sentence, kind
            assert offers.provider_rows("openai", root) == [], kind
        finally:
            if database.exists():
                os.chmod(database, 0o600)
