"""The ``lop network`` surface: the parser, and the refusals that are features."""

from __future__ import annotations

import argparse
import json
import re
from argparse import Namespace
from pathlib import Path

import pytest

from local_operator.network import cli as net_cli
from local_operator.network import relay, store, types, wire
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
    "identity",
    "uninstall",
    # The inviter's half of the pairing human step (mesh-transport-identity §5.3).
    "confirm",
    # The session plane's client half (mesh-session-mobility.md §9.3): what the
    # peers hold, and the three acts on a session that lives on one of them.
    "sessions",
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
        if name in ("member", "identity"):
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
    monkeypatch.setattr("sys.stdin.isatty", lambda: False)
    with pytest.raises(ValueError) as excinfo:
        net_cli._read_code(Namespace(sas_stdin=False, verify=False), "481926", "FP")  # noqa: SLF001
    assert "person at a keyboard" in str(excinfo.value)


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
