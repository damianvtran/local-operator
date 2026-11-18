"""``lop sessions move`` and the working set (docs/design/move-working-set.md §11.2).

Driven through the REAL parser and ``cli.sessions_move_command`` against the in-process
two-relay rig (``tests/unit/network/test_mobility``) with a real git worktree as the
session's cwd — the same hop a person's shell makes, so the flags, the relay frame and
the owner's gate are all on the path:

* ``--dry-run --json`` returns the manifest, valid against the design's §3.4 schema,
  and moves nothing;
* ``--json`` without confirmation exits 1 with the refusal and the manifest;
* ``--confirm-working-set <digest>`` round-trips the digest the dry run printed;
* an interactive terminal is prompted with the warning naming the destination, and
  only an explicit ``y`` moves; ``--yes`` confirms without a prompt.
"""

from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any

import jsonschema
import pytest

from local_operator import cli
from tests.unit.network.test_mobility import (  # noqa: F401 — `devices` is a fixture
    SESSION,
    Devices,
    _owned_session,
    devices,
)
from tests.unit.network.test_relay_e2e import _pair_settled
from tests.unit.network.workset_rig import Rig, make_rig, point_session_at

_DESIGN = Path(__file__).resolve().parents[2] / "docs" / "design" / "move-working-set.md"


def _schema() -> dict[str, Any]:
    """The §3.4 JSON Schema, read from the design note itself so the two cannot drift."""
    text = _DESIGN.read_text(encoding="utf-8")
    match = re.search(
        r"```json\n(\{\n  \"\$id\": \"lop://session_move/working_set/v1\".*?)\n```", text, re.S
    )
    assert match, "the design note's §3.4 schema block moved"
    return json.loads(match.group(1))


@pytest.fixture()
def root(tmp_path: Path) -> Path:
    """The private config root ``devices`` builds under (``tests/unit/network/conftest``
    defines it for that directory; this file lives where the design's test plan put it)."""
    return tmp_path


@pytest.fixture()
def pair(request: pytest.FixtureRequest) -> Devices:
    value: Devices = request.getfixturevalue("devices")
    _server_a, server_b, _host, _port = value
    server_b.bind_control()
    server_b.start()
    return value


@pytest.fixture()
def staged(pair: Devices, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> tuple[Devices, Rig]:
    """A session on A whose cwd is a dirty worktree; the CLI runs on B (a recall)."""
    rig = make_rig(tmp_path / "ws")
    monkeypatch.setenv("HOME", str(rig.home))
    server_a, server_b, _host, _port = pair
    _pair_settled(pair, monkeypatch, role="admin")
    point_session_at(_owned_session(server_a), rig.feat)
    monkeypatch.setenv("LOCAL_OPERATOR_CONFIG_DIR", str(server_b.root))
    return pair, rig


def _run(argv: list[str]) -> int:
    args = cli.build_cli_parser().parse_args(["sessions", "move", SESSION, "--to", "local", *argv])
    return cli.sessions_move_command(args)


def test_dry_run_json_matches_the_design_schema_and_moves_nothing(
    staged: tuple[Devices, Rig], capsys: pytest.CaptureFixture[str]
) -> None:
    (server_a, server_b, _h, _p), _rig = staged

    code = _run(["--dry-run", "--json"])

    document = json.loads(capsys.readouterr().out)
    assert code == 0
    assert document["dry_run"] is True and document["confirmation_required"] is True
    jsonschema.validate(document["working_set"], _schema())
    assert document["working_set"]["worktrees"][0]["path"] == "~/proj/.worktrees/feat"
    assert (server_a.root / "sessions" / SESSION).is_dir()
    assert not (server_b.root / "sessions" / SESSION).exists()


def test_json_without_confirmation_exits_1_with_the_manifest(
    staged: tuple[Devices, Rig], capsys: pytest.CaptureFixture[str]
) -> None:
    (server_a, _server_b, _h, _p), _rig = staged

    code = _run(["--json"])

    document = json.loads(capsys.readouterr().out)
    assert code == 1
    assert document["code"] == "working_set_unconfirmed"
    assert document["changed"] is False
    jsonschema.validate(document["working_set"], _schema())
    assert (server_a.root / "sessions" / SESSION).is_dir()


def test_the_dry_run_digest_round_trips_through_confirm_working_set(
    staged: tuple[Devices, Rig], capsys: pytest.CaptureFixture[str]
) -> None:
    (server_a, server_b, _h, _p), _rig = staged
    assert _run(["--dry-run", "--json"]) == 0
    digest = json.loads(capsys.readouterr().out)["working_set"]["scope_digest"]

    code = _run(["--confirm-working-set", digest, "--json"])

    receipt = json.loads(capsys.readouterr().out)
    assert code == 0, receipt
    assert receipt["working_set"]["scope_digest"] == digest
    assert receipt["working_set"]["transferred"] is False
    assert not (server_a.root / "sessions" / SESSION).exists()
    assert (server_b.root / "sessions" / SESSION).is_dir()


def test_an_interactive_terminal_is_warned_and_only_y_moves(
    staged: tuple[Devices, Rig],
    capsys: pytest.CaptureFixture[str],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    (server_a, server_b, _h, _p), _rig = staged
    monkeypatch.setattr("sys.stdin.isatty", lambda: True)
    monkeypatch.setattr("sys.stdout.isatty", lambda: True)
    prompts: list[str] = []
    answers = iter(["", "y"])

    def answer(prompt: str) -> str:
        prompts.append(prompt)
        return next(answers)

    monkeypatch.setattr("builtins.input", answer)

    declined = _run([])
    first = capsys.readouterr()
    still_here = (server_a.root / "sessions" / SESSION).is_dir()
    accepted = _run([])
    second = capsys.readouterr()

    assert declined == 1
    # The warning names the session and the DESTINATION device, says plainly that the
    # files do not travel yet (PR A), and lists the worktree and the withheld secret.
    assert f"Moving {SESSION} to device-b mid-conversation" in first.out
    assert "does not bring its working files yet" in first.out
    # A RECALL runs this prompt on the DESTINATION: the files stay on the owner, which
    # is named — never "this device", which here would be the wrong machine.
    assert "They stay on device-a:" in first.out
    assert (
        prompts[0]
        == f"Move {SESSION} to device-b and leave these working files on device-a? [y/N] "
    )
    assert "~/proj/.worktrees/feat" in first.out
    assert "withheld (looks like a secret): ~/proj/.worktrees/feat/.env.local" in first.out
    assert prompts[0].endswith("[y/N] ")
    assert "Not moved" in first.err
    assert still_here
    assert accepted == 0, second
    assert "Working files were not transferred: 1 worktree stays on device-a." in second.out
    # QA round 1, Q-2: the receipt LISTS what stayed, not only the count.
    assert "~/proj/.worktrees/feat  stays on device-a:" in second.out
    assert "withheld (looks like a secret): ~/proj/.worktrees/feat/.env.local" in second.out
    assert "not copied: ~/proj/.worktrees/feat/node_modules" in second.out
    assert (server_b.root / "sessions" / SESSION).is_dir()


def test_yes_confirms_without_a_prompt(
    staged: tuple[Devices, Rig],
    capsys: pytest.CaptureFixture[str],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    (_server_a, server_b, _h, _p), _rig = staged

    def no_prompt(_prompt: str) -> str:
        raise AssertionError("--yes must not prompt")

    monkeypatch.setattr("builtins.input", no_prompt)

    code = _run(["--yes", "--json"])

    receipt = json.loads(capsys.readouterr().out)
    assert code == 0, receipt
    assert receipt["working_set"]["consent"] == "confirmed"
    assert (server_b.root / "sessions" / SESSION).is_dir()


# ---------------------------------------------------------------------------
# The destination probe's flags and the active-move resume opt-out
# ---------------------------------------------------------------------------


def test_the_summary_renders_the_destinations_probe_answer(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Each probe flag reaches the confirmation text the CLI prompt and TUI dialog show
    (``workset_text.summary_lines``); English pinned so the assertion reads the copy."""
    from local_operator.network import workset_text

    monkeypatch.setenv("LOCAL_OPERATOR_LANGUAGE", "en")
    monkeypatch.setattr(
        "local_operator.i18n.resolve.resolve_language", lambda *a, **k: "en", raising=False
    )
    block = {
        "carries_files": False,
        "destination": {"device_id": "d_x", "name": "node2"},
        "source": {"device_id": "d_y", "name": "mac"},
        "totals": {"bytes": 10},
        "worktrees": [
            {
                "path": "~/proj/.worktrees/feat",
                "state": "planned",
                "branch": "feat",
                "remote_url": "git@github.com:owner/proj.git",
                "clone_check": "destination_cannot_fetch",
                "https_rewrite": {"url": "https://github.com/owner/proj.git", "fetch": True},
                "remedy": {
                    "run_on": "lending_device",
                    "command": "lop network credential github include owner/proj",
                },
            }
        ],
        "destination_probe": {
            "state": "answered",
            "missing_tools": ["go", "pnpm"],
            "team_bridge": {"needed": True, "enrolled": False},
        },
    }

    text = "\n".join(workset_text.summary_lines(block, session=SESSION))

    assert "node2 cannot fetch git@github.com:owner/proj.git with its own access" in text
    assert "HTTPS does: https://github.com/owner/proj.git" in text
    assert "lop network credential github include owner/proj" in text
    assert "node2 is missing: go, pnpm" in text
    assert "Team Bridge" in text
    # S6 (round 2): the dry-run copy says the destination was ASKED, and what it was
    # sent — the probe runs before the user has decided to trust it.
    assert "node2 was asked whether it can reach these remotes" in text
    assert "no paths or branches" in text
    unknown = dict(block, destination_probe={"state": "unknown_destination"})
    assert "Could not check whether node2" in "\n".join(
        workset_text.summary_lines(unknown, session=SESSION)
    )


def test_no_engage_on_arrival_reaches_the_move_request(monkeypatch: pytest.MonkeyPatch) -> None:
    from local_operator.network import mobility

    seen: list[dict[str, Any]] = []

    def fake(session_id: str, **kwargs: Any) -> dict[str, Any]:
        seen.append(kwargs)
        return {"ok": False, "code": "refused", "message": "stub", "changed": False}

    monkeypatch.setattr(mobility, "request_move", fake)

    assert _run(["--no-engage-on-arrival", "--json"]) == 1
    assert seen[0]["no_engage_on_arrival"] is True
    assert _run(["--json"]) == 1
    assert seen[1]["no_engage_on_arrival"] is False
    # The two flags contradict each other: refused before anything is sent.
    assert _run(["--engage-on-arrival", "--no-engage-on-arrival"]) == 2
    assert len(seen) == 2


# ---------------------------------------------------------------------------
# Review round 1 (PR #4241): R6/U3 probed prompt, U4 one voice, U5 Ctrl-C,
# U8 mistyped digest, R1 a relay that predates the gate
# ---------------------------------------------------------------------------


def _interactive(monkeypatch: pytest.MonkeyPatch, answers: list[str]) -> list[str]:
    monkeypatch.setattr("sys.stdin.isatty", lambda: True)
    monkeypatch.setattr("sys.stdout.isatty", lambda: True)
    prompts: list[str] = []
    replies = iter(answers)

    def answer(prompt: str) -> str:
        prompts.append(prompt)
        reply = next(replies)
        if reply == "^C":
            raise KeyboardInterrupt
        return reply

    monkeypatch.setattr("builtins.input", answer)
    return prompts


def test_the_interactive_prompt_shows_the_probed_block(
    staged: tuple[Devices, Rig],
    capsys: pytest.CaptureFixture[str],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """R6/U3: the y/N summary omitted the destination's answer that --dry-run shows."""
    from local_operator.network import workset_probe

    (_server_a, server_b, _h, _p), _rig = staged
    # The recall probes THIS device: answer "cannot fetch" for the rig's remote.
    monkeypatch.setattr(
        workset_probe,
        "_check_one",
        lambda url, **_kw: {"clone_check": "destination_cannot_fetch", "detail": "unreachable"},
    )
    _interactive(monkeypatch, [""])

    code = _run([])

    out = capsys.readouterr()
    assert code == 1
    assert "cannot fetch" in out.out, out.out
    assert not (server_b.root / "sessions" / SESSION).exists()


def test_ctrl_c_at_the_prompt_moves_nothing_and_exits_130(
    staged: tuple[Devices, Rig],
    capsys: pytest.CaptureFixture[str],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """U5: Ctrl-C printed a KeyboardInterrupt traceback."""
    (server_a, server_b, _h, _p), _rig = staged
    _interactive(monkeypatch, ["^C"])

    code = _run([])

    out = capsys.readouterr()
    assert code == 130
    assert "Not moved" in out.err and "Traceback" not in out.err
    assert (server_a.root / "sessions" / SESSION).is_dir()
    assert not (server_b.root / "sessions" / SESSION).exists()


def test_a_mistyped_digest_says_it_does_not_match(
    staged: tuple[Devices, Rig],
    capsys: pytest.CaptureFixture[str],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """U8: a digest that matches nothing read as "the files changed since confirmed"."""
    (_server_a, server_b, _h, _p), _rig = staged
    _interactive(monkeypatch, [""])

    code = _run(["--confirm-working-set", "sha256:garbage"])

    out = capsys.readouterr()
    assert code == 1
    assert "does not match this conversation's current working files" in out.out
    assert "changed since they were shown" not in out.out
    assert not (server_b.root / "sessions" / SESSION).exists()


def test_a_scripted_refusal_speaks_once_and_names_the_keeper(
    staged: tuple[Devices, Rig], capsys: pytest.CaptureFixture[str]
) -> None:
    """U4: a non-tty refusal printed the owner's English sentence ("on this device",
    the wrong machine on a recall) and then the keyed summary."""
    code = _run([])

    err = capsys.readouterr().err
    assert code == 1
    assert "on this device" not in err
    assert "this conversation worked in" not in err  # the owner's fallback sentence
    assert "They stay on device-a:" in err
    assert "--confirm-working-set sha256:" in err


@pytest.mark.parametrize(
    "kwargs",
    [
        {"dry_run": True},
        {"confirm_working_set": "sha256:x"},
        {"without_working_set": True},
        {},  # R9: the plain move the TUI sent after discarding its preflight refusal
        {"to": "local"},
    ],
)
def test_a_relay_older_than_the_gate_is_never_sent_a_gated_move(
    monkeypatch: pytest.MonkeyPatch, kwargs: dict[str, Any]
) -> None:
    """R1: a pre-gate relay ignored ``dry_run`` and MOVED the conversation. R9: a plain
    move to it was still forwarded, and on an offload that relay IS the owner, with no
    gate, so the conversation left its worktrees without a word."""
    from types import SimpleNamespace

    from local_operator.network import mobility, relay, store

    sent: list[dict[str, Any]] = []
    old = SimpleNamespace(capabilities=["mesh-net-v1", "session-mobility-v1"])
    monkeypatch.setattr(store, "find_own_relay", lambda root=None: old)
    monkeypatch.setattr(relay, "control_request", lambda *a, **k: sent.append(k) or None)

    args = {"to": "peer", **kwargs}
    refused: dict[str, Any] = dict(mobility.request_move(SESSION, **args))

    assert refused["code"] == "relay_restart_required", refused
    assert "restart" in refused["message"] and "update lop" not in refused["message"]
    assert sent == []
    # Controls: the two ungated shapes still reach the same relay — a ``keep`` copy
    # (the source keeps everything) and a recovery from a local replica.
    mobility.request_move(SESSION, to="peer", keep=True)
    mobility.request_move(SESSION, to="", from_replica=True)
    assert len(sent) == 2
    assert mobility.WORKSET_MOVE_CAPABILITY in relay.wire.LINK_CAPABILITIES
