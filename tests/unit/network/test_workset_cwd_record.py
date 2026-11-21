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
* (b) a checkpoint-only older session stays unknown and asks before leaving work;
* (c) a relative ``cwd.json`` or one naming the session directory is ignored (S-2);
* (d) no cwd anywhere plus a mentioned worktree asks for confirmation, plans
  nothing, says why, and ``--yes`` is refused (reason ``cwd_unknown`` since review round 1);
* (e) ``cwd.json`` travels on a move and is remapped under the destination's home,
  and a sender path outside that home is dropped.
"""

from __future__ import annotations

import json
import os
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

    # The durable-only seed must drive the actual transfer, not just the preview.
    monkeypatch.setenv("LOCAL_OPERATOR_CONFIG_DIR", str(server_b.root))
    destination_preview: dict[str, Any] = dict(
        mobility.request_move(SESSION, root=server_b.root, to="local", dry_run=True)
    )
    assert destination_preview["ok"] is True, destination_preview
    moved: dict[str, Any] = dict(
        mobility.request_move(
            SESSION,
            root=server_b.root,
            to="local",
            confirm_working_set=destination_preview["working_set"]["scope_digest"],
        )
    )
    assert moved["ok"] is True, moved
    assert moved["working_set"]["transferred"] is True, moved
    landed = bhome / "proj" / ".worktrees" / "retest"
    assert landed.is_dir(), json.dumps(moved, sort_keys=True)
    assert git(landed, "log", "-1", "--format=%s") == "unpushed retest work"
    assert (landed / "staged.txt").read_text() == "staged\n"
    assert (landed / "b.txt").read_text() == "b\nunstaged edit\n"
    assert not (landed / ".env.local").exists()
    assert (server_b.root / "sessions" / SESSION / "cwd.json").is_file()


def test_b_a_checkpoint_only_session_is_unknown_and_requires_confirmation(
    pair: Devices,
    monkeypatch: pytest.MonkeyPatch,
    rig: Rig,
    bhome: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    """Transcript rows are not cwd authority, even for a local pre-release session."""
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

    assert mobility._session_cwd_at(server_a.root, SESSION) == ""
    monkeypatch.setenv("LOCAL_OPERATOR_CONFIG_DIR", str(server_a.root))
    shown: dict[str, Any] = dict(
        mobility.request_move(
            SESSION, root=server_a.root, to=server_b.identity.device_id, dry_run=True
        )
    )

    block = shown["working_set"]
    assert block["worktrees"] == [], block
    assert block["cwd_known"] is False
    assert block["cwd_unknown_reason"] == "not_recorded"
    # Both checkpoint paths can remain passive mentions, never discovery seeds.
    assert block["mentioned"] == ["~/proj/.worktrees/other", RETEST]
    assert shown["confirmation_required"] is True
    code, out = _cli(
        server_a, monkeypatch, capsys, "--to", server_b.identity.device_id, "--dry-run"
    )
    assert code == 0
    assert "working folder is not recorded" in out.out, out.out
    _grant_import(server_b, server_a)
    code, out = _cli(
        server_a, monkeypatch, capsys, "--to", server_b.identity.device_id, "--yes", "--json"
    )
    assert code == 1
    assert json.loads(out.out)["consent_declined"]["reason"] == "cwd_unknown"


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
    assert manifest.to_dict()["cwd_unknown_reason"] == (
        "not_recorded" if recorded == "SESSION_DIR" else "rejected"
    )


@pytest.mark.parametrize("rejected_legacy", [False, True])
def test_d_unknown_cwd_with_a_mentioned_worktree_asks_and_yes_is_refused(
    rejected_legacy: bool,
    pair: Devices,
    monkeypatch: pytest.MonkeyPatch,
    rig: Rig,
    bhome: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    """No validated source names the cwd: nothing is planned (a mention is never trusted as
    work), but the move asks, the preview says why, and ``--yes`` refuses through the
    existing #4380 rules — the work never strands silently."""
    server_a, server_b, _host, _port = pair
    source = _idle_cli_session(pair, monkeypatch, rig, bhome)
    _retest_worktree(rig)
    if rejected_legacy:
        from types import SimpleNamespace

        from local_operator.mobile import attach_client
        from local_operator.wakes import store as wake_store

        protected = rig.home / ".ssh"
        protected.mkdir()
        rejected = str(protected)
        (source / "desktop.json").write_text(json.dumps({"cwd": rejected}), "utf-8")
        monkeypatch.setattr(wake_store, "read_entry", lambda *_: {"cwd": rejected})
        monkeypatch.setattr(
            attach_client, "find_runtime_record", lambda *_: (SimpleNamespace(cwd=rejected), None)
        )
        assert mobility._session_cwd_at(server_a.root, SESSION, home=rig.home) == ""
        assert mobility._session_cwd_at(server_a.root, SESSION, durable_hints=False) == rejected
    _grant_import(server_b, server_a)  # B COULD take files: the refusal is not a grant gap
    target = server_b.identity.device_id

    code, out = _cli(server_a, monkeypatch, capsys, "--to", target, "--dry-run", "--json")
    answer = json.loads(out.out)
    assert code == 0, answer
    assert answer["confirmation_required"] is True, answer
    block = answer["working_set"]
    assert block["cwd_known"] is False
    assert block["cwd_unknown_reason"] == ("rejected" if rejected_legacy else "not_recorded")
    assert block["totals"]["worktrees"] == 0
    assert [(row["state"], row["reason"]) for row in block["worktrees"]] == (
        [("left_behind", "unsafe_path")] if rejected_legacy else []
    )
    assert block["mentioned"] == [RETEST]
    assert block["carries_files"] is False  # nothing planned, so nothing would travel

    code, out = _cli(server_a, monkeypatch, capsys, "--to", target, "--dry-run")
    assert code == 0
    if rejected_legacy:
        assert "not recorded" not in out.out
        assert "stay" in out.out
    else:
        assert "working folder is not recorded" in out.out, out.out
    assert RETEST in out.out

    code, out = _cli(server_a, monkeypatch, capsys, "--to", target, "--yes", "--json")
    document = json.loads(out.out)
    assert code == 1 and document["ok"] is False, document
    assert document["code"] == "working_set_unconfirmed", document
    # Its own reason since review round 1 (L6): the destination CAN take files here.
    assert document["consent_declined"]["reason"] == (
        "cwd_rejected" if rejected_legacy else "cwd_unknown"
    ), document
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

    # ``--without-working-set``: on A this Mac path does not exist, so A's resolver
    # rejects it (review round 1, L4) and the transcript's mention makes a cwd gap.
    # What this cell pins is the record's travel and remap, not A's plan.
    moved = mobility.request_move(SESSION, root=server_b.root, to="local", without_working_set=True)

    assert moved["ok"] is True, moved
    arrived = json.loads((server_b.root / "sessions" / SESSION / "cwd.json").read_text())
    # Canonical (``realpath``) since review round 1 (L2): what was checked is stored.
    assert arrived["cwd"] == os.path.realpath(bhome / "proj"), arrived
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
        "to": os.path.realpath(home / "proj" / ".worktrees" / "feat"),
    }

    for foreign in (str(outside), "/etc", "relative/dir"):
        _write_cwd_record(target, foreign)
        with caplog.at_level("INFO", logger="local_operator.network.carry"):
            assert carry.remap_cwd_record(target, SESSION, home=str(home))["to"] == ""
        assert json.loads((target / "cwd.json").read_text())["cwd"] == ""
    assert "does not resolve to a usable directory under this device's home" in caplog.text
    assert oct((target / "cwd.json").stat().st_mode & 0o777) == "0o600"


# ---------------------------------------------------------------------------
# Review round 1 remediation (PR #4403 ledger L1–L12). Each cell names its ID.
# ---------------------------------------------------------------------------


def _checkpoint(session: Path, cwd: str) -> None:
    """One ``frontend_state_checkpoint_v1`` row naming ``cwd`` (the step-5 source)."""
    with (session / "transcript.jsonl").open("a", encoding="utf-8") as handle:
        handle.write(
            json.dumps(
                {
                    "id": "cp-x",
                    "type": "custom",
                    "payload": {
                        "custom_type": "frontend_state_checkpoint_v1",
                        "details": {"checkpoint_id": "c", "state": {"cwd": cwd}},
                    },
                }
            )
            + "\n"
        )


def _bare_root(tmp_path: Path) -> tuple[Path, Path, Path]:
    """A config root with one session, and a home holding a dirty git repository, a
    protected ``~/.ssh`` repository and nothing else. Returns (root, session, home)."""
    home = tmp_path / "home"
    for repo in (home / "proj", home / ".ssh"):
        repo.mkdir(parents=True)
        git(repo, "init", "-q")
        (repo / "dirty.txt").write_text("x\n", encoding="utf-8")
    root = home / ".local-operator"
    session = root / "sessions" / SESSION
    session.mkdir(parents=True)
    (session / "transcript.jsonl").write_text("", encoding="utf-8")
    return root, session, home


@pytest.mark.parametrize("where", ["outside_home", "protected_ssh", "config_root"])
def test_l1_a_hostile_cwd_record_is_never_a_discovery_seed(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, where: str
) -> None:
    """L1/S1: ``cwd.json`` naming a path outside home, a protected dot directory
    (``~/.ssh``) or the config root is not a cwd: nothing is discovered there (no
    ``git status`` in it) and the cwd stays unknown."""
    root, session, home = _bare_root(tmp_path)
    outside = tmp_path / "elsewhere"
    outside.mkdir()
    git(outside, "init", "-q")
    target = {"outside_home": outside, "protected_ssh": home / ".ssh", "config_root": session}
    named = target[where] if where != "config_root" else root / "sessions"
    _write_cwd_record(session, str(named))
    monkeypatch.setenv("HOME", str(home))  # the resolver's default home

    assert mobility._session_cwd_at(root, SESSION) == ""
    from local_operator.network import workset

    manifest = workset.compute_manifest(root, SESSION, to_device="d", home=home)
    assert manifest.planned == () and manifest.cwd_known is False
    assert manifest.cwd_unknown_reason == "rejected"
    assert [(row.state, row.reason) for row in manifest.worktrees] == [
        ("left_behind", "outside_home" if where == "outside_home" else "unsafe_path")
    ]
    assert all(not row.abs_path and not row.abs_repo for row in manifest.worktrees)


@pytest.mark.parametrize("imported", [False, True])
def test_l1_checkpoint_rows_are_passive(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, imported: bool
) -> None:
    """No transcript cwd becomes a discovery seed, regardless of arrival metadata."""
    root, session, home = _bare_root(tmp_path)
    monkeypatch.setenv("HOME", str(home))
    _checkpoint(session, str(home / "proj"))
    if imported:
        _imported_stamp(session)
    assert mobility._session_cwd_at(root, SESSION) == ""


def test_l1_a_markerless_arrival_never_writes_the_cwd_record_into_desktop_json(
    pair: Devices,
    monkeypatch: pytest.MonkeyPatch,
    rig: Rig,
    bhome: Path,
) -> None:
    """L1/S1 (QA O2): the durable hints stay off the wire ``cwd``, so the legacy
    marker path (``rewrite_host_cwd`` → ``desktop.json`` → wake index) never adopts
    them. Here A and B share one host, so A's path EXISTS on B — the case where the
    legacy "keep a path that exists here" rule wrote A's directory, outside B's home,
    as B's marker."""
    server_a, server_b, _host, _port = pair
    source = _idle_cli_session(pair, monkeypatch, rig, bhome)
    _write_cwd_record(source, str(rig.primary))
    monkeypatch.setenv("LOCAL_OPERATOR_CONFIG_DIR", str(server_b.root))

    moved = mobility.request_move(SESSION, root=server_b.root, to="local", without_working_set=True)

    assert moved["ok"] is True, moved
    landed = server_b.root / "sessions" / SESSION
    marker = landed / "desktop.json"
    if marker.exists():
        cwd = json.loads(marker.read_text(encoding="utf-8")).get("cwd") or ""
        assert os.path.realpath(cwd).startswith(os.path.realpath(bhome) + os.sep), cwd
    arrived = json.loads((landed / "cwd.json").read_text(encoding="utf-8"))
    assert arrived["cwd"] == "" or arrived["cwd"].startswith(os.path.realpath(bhome)), arrived


def test_l2_a_cwd_record_that_cannot_be_validated_refuses_the_promote(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """L2/S2: the carried record is validated IN STAGING, before the atomic promote;
    a record that can be neither remapped nor blanked refuses the adoption."""
    from types import SimpleNamespace
    from typing import cast

    from local_operator.network import carry

    root = tmp_path / "config"
    staged = root / "network" / "staging" / "staged"
    staged.mkdir(parents=True)
    (staged / "transcript.jsonl").write_text("", encoding="utf-8")
    _write_cwd_record(staged, "/etc")

    def refuse(*_a: Any, **_k: Any) -> dict[str, str]:
        raise carry.CwdRecordUntrusted("cannot rewrite")

    monkeypatch.setattr(carry, "remap_cwd_record", refuse)
    outcome = mobility._promote(cast(Any, SimpleNamespace(root=root)), staged, "moved0000001")
    assert outcome.ok is False, outcome
    assert not (root / "sessions" / "moved0000001").exists()


def test_l2_the_remap_reports_a_failed_write_and_blanks_garbage(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """L2/S2: a writer failure is raised, never reported as a rewrite; an unreadable
    record is blanked like a foreign path; the persisted path is CANONICAL."""
    from local_operator.network import carry
    from local_operator.session import creation

    home = tmp_path / "home"
    (home / "real" / "proj").mkdir(parents=True)
    (home / "link").symlink_to(home / "real")
    target = tmp_path / "arrived"
    target.mkdir()

    _write_cwd_record(target, str(home / "link" / "proj"))
    report = carry.remap_cwd_record(target, SESSION, home=str(home))
    assert report["to"] == os.path.realpath(home / "real" / "proj"), report
    assert json.loads((target / "cwd.json").read_text())["cwd"] == report["to"]

    (target / "cwd.json").write_text("{not json", encoding="utf-8")
    carry.remap_cwd_record(target, SESSION, home=str(home))
    assert json.loads((target / "cwd.json").read_text())["cwd"] == ""

    _write_cwd_record(target, "/etc")
    monkeypatch.setattr(creation, "write_session_cwd", lambda *a, **k: False)
    with pytest.raises(carry.CwdRecordUntrusted):
        carry.remap_cwd_record(target, SESSION, home=str(home))


def test_l2_replica_recovery_contains_the_copied_cwd_record(tmp_path: Path) -> None:
    """L2/S2: ``sync.promote_replica`` copies ``cwd.json``; the owner's path is
    remapped or blanked before the recovered session can read it as its cwd."""
    from tests.unit.network.test_sync import _ask, seed

    root = tmp_path / "holder"
    root.mkdir()
    seed(root, "9f3ac1e0b7d2")
    _write_cwd_record(root / "sessions" / "9f3ac1e0b7d2", "/etc")
    sync.sync_from(
        root,
        "9f3ac1e0b7d2",
        ask=_ask(root),
        owner_device="d_owner",
        into=sync.replica_dir(root, "9f3ac1e0b7d2"),
    )
    (root / "sessions" / "9f3ac1e0b7d2").rename(tmp_path / "elsewhere")
    promoted = sync.promote_replica(root, "9f3ac1e0b7d2", new_id="recovered0001")
    record = json.loads(
        (root / "sessions" / promoted["session_id"] / "cwd.json").read_text(encoding="utf-8")
    )
    assert record["cwd"] == "", record


def test_l4_a_stale_cwd_record_plus_a_live_mentioned_worktree_still_asks(
    pair: Devices,
    monkeypatch: pytest.MonkeyPatch,
    rig: Rig,
    bhome: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    """L4/S4: a recorded folder that no longer exists is not a known cwd, so the
    live worktree the session mentions keeps the gap and the move asks."""
    server_a, server_b, _host, _port = pair
    source = _idle_cli_session(pair, monkeypatch, rig, bhome)
    _retest_worktree(rig)
    _write_cwd_record(source, str(rig.home / "removed-folder"))

    code, out = _cli(
        server_a, monkeypatch, capsys, "--to", server_b.identity.device_id, "--dry-run", "--json"
    )
    answer = json.loads(out.out)
    assert code == 0, answer
    assert answer["working_set"]["cwd_known"] is False, answer
    assert answer["working_set"]["mentioned"] == [RETEST]
    assert answer["confirmation_required"] is True


def test_l4_a_stale_legacy_marker_cwd_is_not_known_either(tmp_path: Path) -> None:
    """L4/S4 for the legacy sources too: the manifest counts only a seed that
    resolves to an existing directory under home."""
    from local_operator.network import workset

    root, session, home = _bare_root(tmp_path)
    (session / "desktop.json").write_text(json.dumps({"cwd": str(home / "gone")}), "utf-8")
    assert mobility._session_cwd_at(root, SESSION, home=home) == ""
    assert mobility._session_cwd_at(root, SESSION, durable_hints=False) == str(home / "gone")
    _mention(session, f"cd {home / 'proj'}")
    manifest = workset.compute_manifest(root, SESSION, to_device="d", home=home)
    assert manifest.cwd_known is False and manifest.cwd_gap is True


def _gap(
    pair: Devices,
    monkeypatch: pytest.MonkeyPatch,
    rig: Rig,
    bhome: Path,
    capsys: pytest.CaptureFixture[str],
) -> tuple[Any, Any, str, str]:
    """Test (d)'s state with B granted: (server_a, server_b, target id, dry-run digest)."""
    server_a, server_b, _host, _port = pair
    _idle_cli_session(pair, monkeypatch, rig, bhome)
    _retest_worktree(rig)
    _grant_import(server_b, server_a)
    target = server_b.identity.device_id
    code, out = _cli(server_a, monkeypatch, capsys, "--to", target, "--dry-run", "--json")
    assert code == 0, out
    return server_a, server_b, target, json.loads(out.out)["working_set"]["scope_digest"]


def _b_sessions(server_b: Any) -> list[str]:
    sessions = server_b.root / "sessions"
    return sorted(p.name for p in sessions.iterdir() if p.is_dir()) if sessions.is_dir() else []


def test_l5_keep_is_gated_on_the_cwd_gap_and_its_answer_names_the_folders(
    pair: Devices,
    monkeypatch: pytest.MonkeyPatch,
    rig: Rig,
    bhome: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    """L5 (R1-M1/U8/Q-1, QA Q8's shape): ``--keep`` with no consent is refused and
    copies nothing; confirmed, the copy's answer and receipt name the folders."""
    server_a, server_b, target, digest = _gap(pair, monkeypatch, rig, bhome, capsys)
    before = _b_sessions(server_b)

    code, out = _cli(server_a, monkeypatch, capsys, "--to", target, "--keep", "--json")
    refused = json.loads(out.out)
    assert code == 1 and refused["ok"] is False, refused
    assert refused["code"] == "working_set_unconfirmed", refused
    assert _b_sessions(server_b) == before, "an unconfirmed keep copied the conversation"

    code, out = _cli(
        server_a,
        monkeypatch,
        capsys,
        "--to",
        target,
        "--keep",
        "--confirm-working-set",
        digest,
        "--json",
    )
    answer = json.loads(out.out)
    assert code == 0 and answer["ok"] is True, answer
    block = answer["working_set"]
    assert block["cwd_known"] is False and block["mentioned"] == [RETEST], block
    assert block["transferred"] is False

    code, out = _cli(
        server_a, monkeypatch, capsys, "--to", target, "--keep", "--confirm-working-set", digest
    )
    assert code == 0, out
    assert "does not have the folders this conversation mentions" in out.out, out.out
    assert RETEST in out.out


def test_l6_l8_yes_on_the_gap_says_why_and_never_blames_the_destination(
    pair: Devices,
    monkeypatch: pytest.MonkeyPatch,
    rig: Rig,
    bhome: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    """L6 (U1/R1-m1/Q-2) and L8 (U3/Q-3): the destination CAN take files, so the
    refusal must not say it cannot; the owner's message must not count 0 worktrees."""
    server_a, server_b, target, _digest = _gap(pair, monkeypatch, rig, bhome, capsys)

    code, out = _cli(server_a, monkeypatch, capsys, "--to", target, "--yes", "--json")
    document = json.loads(out.out)
    assert code == 1 and document["consent_declined"]["reason"] == "cwd_unknown", document
    assert "does not know which folder" in document["consent_declined"]["message"]
    assert "cannot take" not in json.dumps(document)
    assert "0 git worktrees" not in document["message"], document["message"]
    assert "working folder is not recorded" in document["message"]

    code, out = _cli(server_a, monkeypatch, capsys, "--to", target, "--yes")
    assert code == 1
    assert "does not know which folder" in out.err and "cannot take" not in out.err, out.err
    assert not (server_b.root / "sessions" / SESSION).exists()


def test_l7_l11_the_confirmed_move_receipt_names_the_folders_that_stayed(
    pair: Devices,
    monkeypatch: pytest.MonkeyPatch,
    rig: Rig,
    bhome: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    """L7 (U2/R1-m2): the receipt lists the mentioned folder that stayed instead of
    "0 worktrees stay". L11 (U4/U7): the refusal's closing line says confirming
    LEAVES the folders, and the summary reads "is not recorded"."""
    server_a, server_b, target, digest = _gap(pair, monkeypatch, rig, bhome, capsys)

    code, out = _cli(server_a, monkeypatch, capsys, "--to", target)
    assert code == 1
    assert "working folder is not recorded" in out.err, out.err
    assert f"--confirm-working-set {digest} to leave them on" in out.err, out.err

    code, out = _cli(server_a, monkeypatch, capsys, "--to", target, "--confirm-working-set", digest)
    assert code == 0, out
    assert "Working files were not transferred: this conversation's working folder" in out.out
    assert RETEST in out.out and "0 worktrees" not in out.out, out.out
    assert (server_b.root / "sessions" / SESSION).is_dir()


def test_l9_the_gap_is_part_of_the_scope_digest() -> None:
    """L9 (Q-4): a gap's digest differs from the empty plan's and from another gap's,
    and the block recompute agrees with the owner's; a non-gap digest is unchanged."""
    from local_operator.network import workset

    empty = workset.Manifest(to_device="d", to_name="")
    gap = workset.Manifest(to_device="d", to_name="", mentioned=("~/a",), cwd_known=False)
    other = workset.Manifest(to_device="d", to_name="", mentioned=("~/a", "~/b"), cwd_known=False)
    known = workset.Manifest(to_device="d", to_name="", mentioned=("~/a",), cwd_known=True)

    assert workset.scope_digest(gap) != workset.scope_digest(empty)
    assert workset.scope_digest(gap) != workset.scope_digest(other)
    assert workset.scope_digest(known) == workset.scope_digest(empty)
    assert workset.block_scope_digest(gap.to_dict(), to_device="d") == workset.scope_digest(gap)


def test_l12_an_answered_root_cwd_is_not_resolved_twice(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """L12 (R1-n2): the owner's "" is an answer; only ``None`` resolves the root."""
    from local_operator.network import workset

    root, _session, home = _bare_root(tmp_path)
    calls: list[str] = []
    monkeypatch.setattr(workset, "recorded_cwd", lambda _root, sid, **_k: calls.append(sid) or "")
    workset.candidate_seeds(root, [SESSION], cwd_hint="", home=home)
    assert calls == []
    workset.candidate_seeds(root, [SESSION], home=home)
    assert calls == [SESSION]


def test_l1_the_cwd_containment_rules_match_the_restore_policy() -> None:
    """``carry`` is stdlib-only, so it spells the restore policy's component rules
    locally; they must never drift from ``workset_transfer``'s."""
    from local_operator.network import carry, workset_transfer

    assert carry._CWD_DENY_TOP == workset_transfer._RESTORE_DENY_TOP
    assert carry._CWD_DENY_ANY == workset_transfer._RESTORE_DENY_ANY


# ---------------------------------------------------------------------------
# Review round 2 remediation (PR #4403 ledger M1–M3, m1, m2, n1)
# ---------------------------------------------------------------------------


def _imported_stamp(session: Path) -> None:
    (session / "mesh.json").write_text(
        json.dumps({"session_id": session.name, "origin": {"kind": "moved", "source_device": "x"}}),
        encoding="utf-8",
    )


@pytest.mark.parametrize("imported", [False, True])
def test_m1_a_fork_checkpoint_never_becomes_a_seed(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, imported: bool
) -> None:
    """Copying journal content never grants it cwd authority, even before origin exists."""
    from local_operator.fork import fork_session

    root, session, home = _bare_root(tmp_path)
    monkeypatch.setenv("HOME", str(home))
    _checkpoint(session, str(home / "proj"))
    if imported:
        _imported_stamp(session)
    fork_id = fork_session(root, SESSION)
    fork_dir = root / "sessions" / fork_id
    assert not (fork_dir / "mesh.json").exists()
    assert mobility._session_cwd_at(root, fork_id) == ""
    (fork_dir / "origin.json").unlink()
    assert mobility._session_cwd_at(root, fork_id) == ""
    assert mobility._session_cwd_at(root, fork_session(root, fork_id)) == ""


def test_m2_replica_recovery_publishes_only_a_stamped_contained_session(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """M2/S2 (r2): the recovered copy is built OFF the store and appears under
    ``sessions/<new_id>`` in one rename, already carrying the normalised ``cwd.json``
    and the import stamp. While the copy is being validated nothing exists at the
    final path, so no reader can take the raw owner path or an unstamped checkpoint
    as local; a refused recovery never appears there at all."""
    from local_operator.network import carry, origin
    from tests.unit.network.test_sync import _ask, seed

    root = tmp_path / "holder"
    root.mkdir()
    seed(root, "9f3ac1e0b7d2")
    _write_cwd_record(root / "sessions" / "9f3ac1e0b7d2", "/etc")
    sync.sync_from(
        root,
        "9f3ac1e0b7d2",
        ask=_ask(root),
        owner_device="d_owner",
        into=sync.replica_dir(root, "9f3ac1e0b7d2"),
    )
    (root / "sessions" / "9f3ac1e0b7d2").rename(tmp_path / "elsewhere")
    final = root / "sessions" / "recovered0001"
    seen: list[str] = []
    real_remap, real_downgrade = carry.remap_cwd_record, origin.downgrade_stamps

    def remap(target: Path, *args: Any, **kwargs: Any) -> Any:
        seen.append(f"remap:{os.path.lexists(final)}")
        return real_remap(target, *args, **kwargs)

    def downgrade(target: Path, *args: Any, **kwargs: Any) -> Any:
        seen.append(f"downgrade:{os.path.lexists(final)}")
        return real_downgrade(target, *args, **kwargs)

    monkeypatch.setattr(carry, "remap_cwd_record", remap)
    monkeypatch.setattr(origin, "downgrade_stamps", downgrade)
    promoted = sync.promote_replica(root, "9f3ac1e0b7d2", new_id="recovered0001")
    assert seen == ["downgrade:False", "remap:False"], seen
    assert promoted["session_id"] == "recovered0001"
    assert json.loads((final / "cwd.json").read_text(encoding="utf-8"))["cwd"] == ""
    stamp = json.loads((final / "mesh.json").read_text(encoding="utf-8"))
    assert stamp["origin"]["source_device"] == "d_owner", stamp

    # A refused recovery: nothing ever exists at the final path.
    other = root / "sessions" / "recovered0002"
    at_refusal: list[bool] = []

    def refuse(*_a: Any, **_k: Any) -> Any:
        at_refusal.append(os.path.lexists(other))
        raise carry.CwdRecordUntrusted("injected")

    monkeypatch.setattr(carry, "remap_cwd_record", refuse)
    with pytest.raises(sync.SyncRefused):
        sync.promote_replica(root, "9f3ac1e0b7d2", new_id="recovered0002")
    assert at_refusal == [False]
    assert not os.path.lexists(other)
    assert (sync.network_dir(root) / sync.QUARANTINE_DIRNAME / "recovered0002").is_dir()


def test_m3_a_stale_legacy_cwd_does_not_shadow_a_valid_cwd_record(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """M3/S4 (r2): discovery validates each LEGACY candidate (wake, marker, runtime)
    with the durable hints' containment rule and falls through when it is unusable, so
    a stale marker no longer hides the real worktree ``cwd.json`` names. The WIRE
    resolver (``durable_hints=False``) is unchanged: legacy sources only, as before."""
    from local_operator.network import workset

    root, session, home = _bare_root(tmp_path)
    monkeypatch.setenv("HOME", str(home))
    (session / "desktop.json").write_text(json.dumps({"cwd": str(home / "gone")}), "utf-8")
    _write_cwd_record(session, str(home / "proj"))

    assert os.path.realpath(mobility._session_cwd_at(root, SESSION)) == os.path.realpath(
        home / "proj"
    )
    assert mobility._session_cwd_at(root, SESSION, durable_hints=False) == str(home / "gone")
    manifest = workset.compute_manifest(root, SESSION, to_device="d", home=home)
    assert manifest.cwd_known is True
    assert [w.path for w in manifest.worktrees] == ["~/proj"], manifest.worktrees


def test_m1_minor_the_cwd_writer_never_unlinks_a_staging_name_it_did_not_create(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """m1/S3 (r2): when the exclusive open of the staging name fails because the name
    already exists, that file is someone else's and stays."""
    from local_operator.session import creation

    monkeypatch.setattr(creation.secrets, "token_hex", lambda _n: "feedface")
    planted = tmp_path / f"{creation.CWD_RECORD_NAME}.feedface.tmp"
    planted.write_text("not mine", encoding="utf-8")
    assert creation.write_session_cwd(tmp_path, str(tmp_path), source="runtime") is False
    assert planted.read_text(encoding="utf-8") == "not mine"


def test_m2_minor_no_posix_uid_skips_the_remap_instead_of_refusing(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """m2/R2-m1: where ``os.getuid`` (and the verified directory handle built on it)
    does not exist, an arriving ``cwd.json`` is not a reason to refuse the session:
    the remap is skipped and the record reads blank, because every reader goes
    through the same unavailable handle."""
    from local_operator.network import carry
    from local_operator.session import creation

    target = tmp_path / "arrived"
    target.mkdir()
    _write_cwd_record(target, "/etc")
    monkeypatch.delattr(os, "getuid")
    assert carry.remap_cwd_record(target, SESSION, home=str(tmp_path)) == {}
    assert creation.read_session_cwd(target) == ""
    assert carry.contained_cwd(str(tmp_path), home=str(tmp_path)) == ""


def test_n1_a_carried_restore_records_the_restored_tree_in_cwd_json(tmp_path: Path) -> None:
    """n1/R2-n1: after a restore with files carried, ``cwd.json`` names the restored
    tree (contained), not the blank the promote's remap left before it existed."""
    from types import SimpleNamespace

    from local_operator.network import workset_move
    from local_operator.session.creation import read_session_cwd

    home = tmp_path / "home"
    (home / "proj-moved" / "sub").mkdir(parents=True)
    root = home / ".local-operator"
    session = root / "sessions" / SESSION
    session.mkdir(parents=True)
    _write_cwd_record(session, "")
    rows = [{"state": "restored", "path": "~/proj", "dest_path": "~/proj-moved"}]
    workset_move._retarget_cwd(SimpleNamespace(root=root), SESSION, "~/proj/sub", rows, home)
    assert read_session_cwd(session) == os.path.realpath(home / "proj-moved" / "sub")


@pytest.mark.parametrize("source_kind", ["desktop", "durable"])
@pytest.mark.parametrize(
    "kind", ["outside_home", "relative", "control", "protected", "missing", "foreign_uid"]
)
def test_rejected_cwd_without_mentions_requires_files_stay_consent(
    pair: Devices,
    monkeypatch: pytest.MonkeyPatch,
    rig: Rig,
    bhome: Path,
    capsys: pytest.CaptureFixture[str],
    source_kind: str,
    kind: str,
) -> None:
    """A positive but unsafe cwd is disclosure, not discovery or transfer authority.

    Exercise the real CLI and owner gates with no transcript mentions. The spies
    make a silently reintroduced git/facts/gitdir discovery fail even if the
    consent result happens to remain correct. Recorded bytes never leave owner
    metadata, including protected names and terminal-control/secret-shaped text.
    """
    from local_operator.network import workset

    server_a, server_b, _host, _port = pair
    source = _idle_cli_session(pair, monkeypatch, rig, bhome)
    (source / "transcript.jsonl").write_text("", encoding="utf-8")
    values = {
        "outside_home": str(rig.base / "outside-private"),
        "relative": "private-relative",
        "control": str(rig.home / "private\x1b[31m\x00cwd"),
        "protected": str(rig.home / ".ssh" / "private"),
        "missing": str(rig.home / "missing-private"),
        "foreign_uid": str(rig.home / "foreign-private"),
    }
    recorded = values[kind]
    if kind == "foreign_uid":
        Path(recorded).mkdir()
        real_stat = os.stat

        def foreign_stat(path: Any, *args: Any, **kwargs: Any) -> Any:
            info = real_stat(path, *args, **kwargs)
            if str(path) == recorded:
                fields = list(info)
                fields[4] = os.getuid() + 1
                return os.stat_result(fields)
            return info

        monkeypatch.setattr(os, "stat", foreign_stat)
    if source_kind == "desktop":
        (source / "desktop.json").write_text(json.dumps({"cwd": recorded}), encoding="utf-8")
    else:
        _write_cwd_record(source, recorded)
    _grant_import(server_b, server_a)
    target = server_b.identity.device_id
    calls: list[str] = []

    def forbidden(*_args: Any, **_kwargs: Any) -> Any:
        calls.append("discovery")
        raise AssertionError("a rejected cwd reached git/facts/gitdir discovery")

    for name in ("_git", "worktree_facts", "_walk_up_for_git"):
        monkeypatch.setattr(workset, name, forbidden)
    code, out = _cli(server_a, monkeypatch, capsys, "--to", target, "--dry-run", "--json")
    preview = json.loads(out.out)
    assert code == 0 and preview["confirmation_required"] is True, preview
    block = preview["working_set"]
    reason = "outside_home" if kind == "outside_home" else "unsafe_path"
    assert block["cwd_unknown_reason"] == "rejected"
    assert block.get("mentioned", []) == [] and block["carries_files"] is False
    assert [(r["state"], r["reason"]) for r in block["worktrees"]] == [("left_behind", reason)]
    assert block["totals"]["worktrees"] == 0 and block["totals"]["bytes"] == 0
    assert "private" not in json.dumps(block) and "\\u001b" not in json.dumps(block)
    assert workset.block_scope_digest(block, to_device=target) == block["scope_digest"]
    manifest = mobility._workset_manifest(server_a, SESSION, to_device=target, carries=True)
    assert not manifest.planned and not manifest.carries_files
    assert all(not row.abs_path and not row.abs_repo for row in manifest.worktrees)
    before = _b_sessions(server_b)
    destination_trees = sorted(bhome.rglob(".git"))
    stale = workset.scope_digest(workset.Manifest(to_device=target, to_name=""))
    for mode in ([], ["--keep"]):
        for flags, expected in (
            ([], "working_set_unconfirmed"),
            (["--yes"], "working_set_unconfirmed"),
            (["--confirm-working-set", stale], "working_set_changed"),
        ):
            code, out = _cli(server_a, monkeypatch, capsys, "--to", target, *mode, *flags, "--json")
            refusal = json.loads(out.out)
            assert code == 1 and refusal["code"] == expected, refusal
            if flags == ["--yes"]:
                assert refusal["consent_declined"]["reason"] == "cwd_rejected", refusal
                assert "not recorded" not in refusal["consent_declined"]["message"]
                assert "cannot take" not in refusal["consent_declined"]["message"]
            assert _b_sessions(server_b) == before
            assert source.is_dir()
    code, out = _cli(server_a, monkeypatch, capsys, "--to", target, "--dry-run")
    assert code == 0 and "stay" in out.out and "private" not in out.out, out
    assert "not recorded" not in out.out and "needs no confirmation" not in out.out
    # Confirmed KEEP really copies the session, but never snapshots any worktree.
    code, out = _cli(
        server_a,
        monkeypatch,
        capsys,
        "--to",
        target,
        "--keep",
        "--confirm-working-set",
        block["scope_digest"],
        "--json",
    )
    answer = json.loads(out.out)
    assert code == 0 and answer["ok"] is True, answer
    assert answer["working_set"]["transferred"] is False
    assert len(_b_sessions(server_b)) == len(before) + 1
    assert sorted(bhome.rglob(".git")) == destination_trees
    # Explicit MOVE consent likewise sends only the conversation, never work.
    code, out = _cli(
        server_a,
        monkeypatch,
        capsys,
        "--to",
        target,
        "--confirm-working-set",
        block["scope_digest"],
        "--json",
    )
    moved = json.loads(out.out)
    assert code == 0 and moved["ok"] is True, moved
    assert moved["working_set"]["transferred"] is False
    assert sorted(bhome.rglob(".git")) == destination_trees
    assert rig.feat.is_dir()
    assert calls == []
    print(
        json.dumps(
            {
                "case": [source_kind, kind],
                "confirmation_required": True,
                "reason": reason,
                "move_keep_yes": "refused",
                "stale_digest": "refused",
                "confirmed_keep": answer["ok"],
                "transferred": answer["working_set"]["transferred"],
                "git_facts_gitdir_calls": len(calls),
            }
        )
    )
