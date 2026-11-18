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
    assert "~/proj/.worktrees/feat" in first.out
    assert "withheld (looks like a secret): ~/proj/.worktrees/feat/.env.local" in first.out
    assert prompts[0].endswith("[y/N] ")
    assert "Not moved" in first.err
    assert still_here
    assert accepted == 0, second
    assert "Working files were not transferred: 1 worktree stays on device-a." in second.out
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
