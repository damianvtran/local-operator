"""An idle CLI-started session's worktree is planned on move (coordinator retest, v0.111.4).

THE DEFECT: a session started by ``lop exec --workstream`` in a git worktree with an
unpushed commit, staged, unstaged and untracked files ran one turn and went idle. Its
runtime exited, it had no ``desktop.json`` marker and no wake entry, so
``mobility._session_cwd_at`` found no cwd; the worktree appeared only under
``mentioned`` and ``lop sessions move <id> --to <peer> --dry-run --json`` answered
``confirmation_required: false``, ``worktrees: []`` — a real move would have left the
work behind without asking.

Each cell drives the real path (the CLI or ``mobility.request_move`` over the
in-process two-relay rig, with real git) and pins one part of the fix:

* (a) the retest's fixture shape, with ``cwd.json`` written at birth, is PLANNED;
* (b) a session older than ``cwd.json`` is planned from its latest checkpoint row;
* (c) a relative ``cwd.json`` or one naming the session directory is ignored (S-2);
* (d) no cwd anywhere plus a mentioned worktree asks for confirmation, plans
  nothing, says why, and ``--yes`` is refused by #4380's existing rules;
* (e) ``cwd.json`` travels on a move and is remapped under the destination's home,
  and a sender path outside that home is dropped.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest

from local_operator import cli
from local_operator.agent_shell import AGENT_SHELL_ENV
from local_operator.network import mobility, sync
from tests.unit.network.test_mobility import SESSION, Devices, _owned_session
from tests.unit.network.test_mobility import devices as _devices_fixture
from tests.unit.network.test_relay_e2e import _pair_settled
from tests.unit.network.test_workset_move_e2e import _grant_import
from tests.unit.network.test_workset_move_e2e import bhome as _bhome_fixture
from tests.unit.network.test_workset_move_e2e import pair as _pair_fixture
from tests.unit.network.test_workset_move_e2e import rig as _rig_fixture
from tests.unit.network.workset_rig import Rig, git

devices = _devices_fixture
pair = _pair_fixture
rig = _rig_fixture
bhome = _bhome_fixture

RETEST = "~/proj/.worktrees/retest"


def _retest_worktree(rig: Rig) -> Path:
    """The retest's working tree: 1 unpushed commit, and exactly three dirty files —
    one staged (``staged.txt``), one unstaged (``b.txt`` modified) and one untracked
    (the decoy ``.env.local``, which must be withheld)."""
    tree = rig.primary / ".worktrees" / "retest"
    git(rig.primary, "worktree", "add", "-q", "-b", "retest", str(tree))
    git(tree, "push", "-q", "-u", "origin", "retest")
    (tree / "a.txt").write_text("a\nunpushed\n", encoding="utf-8")
    git(tree, "commit", "-q", "-am", "unpushed retest work")
    (tree / "staged.txt").write_text("staged\n", encoding="utf-8")
    git(tree, "add", "staged.txt")
    (tree / "b.txt").write_text("b\nunstaged edit\n", encoding="utf-8")
    (tree / ".env.local").write_text("API_TOKEN=decoy\n", encoding="utf-8")
    return tree


def _write_cwd_record(session: Path, cwd: str) -> None:
    """``cwd.json`` exactly as the birth path writes it (``creation.write_session_cwd``)."""
    (session / "cwd.json").write_text(
        json.dumps({"v": 1, "cwd": cwd, "at": 1.0, "source": "birth"}), encoding="utf-8"
    )


def _mention(session: Path, text: str) -> None:
    with (session / "transcript.jsonl").open("a", encoding="utf-8") as handle:
        handle.write(json.dumps({"id": "m1", "type": "user", "content": text}) + "\n")


def _idle_cli_session(
    pair: Devices, monkeypatch: pytest.MonkeyPatch, rig: Rig, bhome: Path
) -> Path:
    """An idle exec/TUI-created session on A: no ``desktop.json``, no wake entry, no
    runtime record — only its transcript (which mentions the worktree, as a bash turn
    run there does)."""
    server_a, server_b, _host, _port = pair
    _pair_settled(pair, monkeypatch, role="admin", settings=server_b.settings)
    server_a.workset_home = rig.home  # type: ignore[attr-defined]
    server_b.workset_home = bhome  # type: ignore[attr-defined]
    monkeypatch.delenv(AGENT_SHELL_ENV, raising=False)
    source = _owned_session(server_a)
    assert not (source / "desktop.json").exists()
    _mention(source, f"git -C {RETEST} status")
    return source


def _cli(
    server: Any, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str], *flags: str
) -> tuple[int, Any]:
    monkeypatch.setenv("LOCAL_OPERATOR_CONFIG_DIR", str(server.root))
    args = cli.build_cli_parser().parse_args(["sessions", "move", SESSION, *flags])
    capsys.readouterr()
    code = cli.sessions_move_command(args)
    return code, capsys.readouterr()


def test_a_the_retest_fixture_is_planned_from_cwd_json(
    pair: Devices,
    monkeypatch: pytest.MonkeyPatch,
    rig: Rig,
    bhome: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    """THE ACCEPTANCE FIXTURE: the exact command the retest ran now plans the work."""
    server_a, server_b, _host, _port = pair
    source = _idle_cli_session(pair, monkeypatch, rig, bhome)
    _write_cwd_record(source, str(_retest_worktree(rig)))
    _grant_import(server_b, server_a)

    code, out = _cli(
        server_a, monkeypatch, capsys, "--to", server_b.identity.device_id, "--dry-run", "--json"
    )

    answer = json.loads(out.out)
    assert code == 0, answer
    assert answer["confirmation_required"] is True, answer
    block = answer["working_set"]
    assert block["cwd_known"] is True
    (row,) = block["worktrees"]
    assert row["path"] == RETEST and row["state"] == "planned", row
    assert "cwd" in row["sources"], row  # planned BY the cwd, not only mentioned
    assert block["totals"]["worktrees"] == 1
    assert block["totals"]["unpushed_commits"] == 1
    assert block["totals"]["dirty_files"] == 3
    assert f"{RETEST}/.env.local" in [item["path"] for item in block["excluded"]], block
    assert "mentioned" not in block  # it is the cwd now, not a mention
    assert (server_a.root / "sessions" / SESSION).is_dir()


def test_b_a_session_without_cwd_json_is_planned_from_its_latest_checkpoint(
    pair: Devices,
    monkeypatch: pytest.MonkeyPatch,
    rig: Rig,
    bhome: Path,
) -> None:
    """Sessions older than ``cwd.json`` carry their cwd only in the frontend
    checkpoint row; the LATEST one wins (an older row names a stale folder)."""
    server_a, server_b, _host, _port = pair
    source = _idle_cli_session(pair, monkeypatch, rig, bhome)
    tree = _retest_worktree(rig)
    with (source / "transcript.jsonl").open("a", encoding="utf-8") as handle:
        for cwd in (str(rig.other), str(tree)):
            handle.write(
                json.dumps(
                    {
                        "id": f"cp-{Path(cwd).name}",
                        "type": "custom",
                        "payload": {
                            "custom_type": "frontend_state_checkpoint_v1",
                            "details": {"checkpoint_id": "c", "state": {"cwd": cwd}},
                        },
                    }
                )
                + "\n"
            )
    assert not (source / "cwd.json").exists()

    assert mobility._session_cwd_at(server_a.root, SESSION) == str(tree)
    monkeypatch.setenv("LOCAL_OPERATOR_CONFIG_DIR", str(server_a.root))
    shown: dict[str, Any] = dict(
        mobility.request_move(
            SESSION, root=server_a.root, to=server_b.identity.device_id, dry_run=True
        )
    )

    block = shown["working_set"]
    assert [(r["path"], r["state"]) for r in block["worktrees"]] == [(RETEST, "planned")], block
    assert shown["confirmation_required"] is True


@pytest.mark.parametrize("recorded", ["relative/dir", "SESSION_DIR"])
def test_c_a_relative_or_session_dir_cwd_json_is_ignored(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, recorded: str
) -> None:
    """S-2: a relative value would resolve against the RELAY's cwd, and the session
    directory is not a working folder; both fall through to "unknown"."""
    root = tmp_path / "cfg"
    session = root / "sessions" / SESSION
    session.mkdir(parents=True)
    (session / "transcript.jsonl").write_text("", encoding="utf-8")
    monkeypatch.setenv("LOCAL_OPERATOR_CONFIG_DIR", str(root))
    _write_cwd_record(session, str(session) if recorded == "SESSION_DIR" else recorded)

    assert mobility._session_cwd_at(root, SESSION) == ""
    from local_operator.network import workset

    manifest = workset.compute_manifest(root, SESSION, to_device="d", home=tmp_path)
    assert manifest.cwd_known is False
    assert manifest.to_dict()["cwd_unknown_reason"] == "not_recorded"


def test_d_unknown_cwd_with_a_mentioned_worktree_asks_and_yes_is_refused(
    pair: Devices,
    monkeypatch: pytest.MonkeyPatch,
    rig: Rig,
    bhome: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    """No source names the cwd: nothing is planned (a mention is never trusted as
    work), but the move asks, the preview says why, and ``--yes`` refuses through the
    existing #4380 rules — the work never strands silently."""
    server_a, server_b, _host, _port = pair
    _idle_cli_session(pair, monkeypatch, rig, bhome)
    _retest_worktree(rig)
    _grant_import(server_b, server_a)  # B COULD take files: the refusal is not a grant gap
    target = server_b.identity.device_id

    code, out = _cli(server_a, monkeypatch, capsys, "--to", target, "--dry-run", "--json")
    answer = json.loads(out.out)
    assert code == 0, answer
    assert answer["confirmation_required"] is True, answer
    block = answer["working_set"]
    assert block["cwd_known"] is False and block["cwd_unknown_reason"] == "not_recorded"
    assert block["worktrees"] == [] and block["totals"]["worktrees"] == 0
    assert block["mentioned"] == [RETEST]
    assert block["carries_files"] is False  # nothing planned, so nothing would travel

    code, out = _cli(server_a, monkeypatch, capsys, "--to", target, "--dry-run")
    assert code == 0
    assert "working folder isn't recorded" in out.out, out.out
    assert RETEST in out.out

    code, out = _cli(server_a, monkeypatch, capsys, "--to", target, "--yes", "--json")
    document = json.loads(out.out)
    assert code == 1 and document["ok"] is False, document
    assert document["code"] == "working_set_unconfirmed", document
    assert document["consent_declined"]["reason"] == "carry_unavailable", document
    assert (server_a.root / "sessions" / SESSION).is_dir()
    assert not (server_b.root / "sessions" / SESSION).exists()


def test_e_cwd_json_travels_and_is_remapped_under_the_destination_home(
    pair: Devices,
    monkeypatch: pytest.MonkeyPatch,
    rig: Rig,
    bhome: Path,
) -> None:
    """A real move: the sender's ``/Users/x/...`` cwd arrives as the same folder under
    the destination's home (the existing home map), never as the sender's path."""
    server_a, server_b, _host, _port = pair
    assert sync.CWD_RECORD_NAME in sync.COPY_SET_NAMES
    source = _idle_cli_session(pair, monkeypatch, rig, bhome)
    _write_cwd_record(source, "/Users/x/proj")  # the Mac's path; ``bhome/proj`` exists on B
    monkeypatch.setenv("LOCAL_OPERATOR_CONFIG_DIR", str(server_b.root))

    moved = mobility.request_move(SESSION, root=server_b.root, to="local")

    assert moved["ok"] is True, moved
    arrived = json.loads((server_b.root / "sessions" / SESSION / "cwd.json").read_text())
    assert arrived["cwd"] == str(bhome / "proj"), arrived
    assert arrived["v"] == 1 and arrived["source"] == "birth"


def test_e_a_sender_cwd_outside_the_destination_home_is_dropped(
    tmp_path: Path, caplog: pytest.LogCaptureFixture
) -> None:
    """Security note 1: a sender path that does not resolve under the destination's
    home after the remap is never adopted; it is blanked and the drop is logged."""
    from local_operator.network import carry

    home = tmp_path / "home" / "y"
    (home / "proj" / ".worktrees" / "feat").mkdir(parents=True)
    outside = tmp_path / "elsewhere"  # EXISTS here, so the home map would keep it
    outside.mkdir()
    target = tmp_path / "arrived"
    target.mkdir()

    _write_cwd_record(target, "/Users/x/proj/.worktrees/feat")
    assert carry.remap_cwd_record(target, SESSION, home=str(home)) == {
        "from": "/Users/x/proj/.worktrees/feat",
        "to": str(home / "proj" / ".worktrees" / "feat"),
    }

    for foreign in (str(outside), "/etc", "relative/dir"):
        _write_cwd_record(target, foreign)
        with caplog.at_level("INFO", logger="local_operator.network.carry"):
            assert carry.remap_cwd_record(target, SESSION, home=str(home))["to"] == ""
        assert json.loads((target / "cwd.json").read_text())["cwd"] == ""
    assert "does not resolve under this device's home" in caplog.text
    assert oct((target / "cwd.json").stat().st_mode & 0o777) == "0o600"
