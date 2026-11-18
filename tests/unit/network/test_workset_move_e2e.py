"""The session move CARRIES its working set (design ``move-working-set.md`` §4, §10, PR B).

The operator's request (2026-11-18): "we should be able to move the session and its
working data, worktrees, etc. all to a remote device … Otherwise people could use the
move function and then realize all the work is not available."

Each cell drives the move the way the CLI does — ``mobility.request_move`` over the
in-process two-relay rig from ``test_mobility`` — with a REAL git worktree as the
session's cwd and DIFFERENT homes per device (``server.workset_home``), so the
``~/``-relative remap is exercised for real:

* a confirmed move lands the unpushed commit, the staged, unstaged and untracked
  changes as such on the destination, registered in ``git worktree list``; the
  secret-shaped ``.env.local`` and the ignored ``node_modules/`` do not travel; the
  source worktree is byte-identical and marked; the owner audited the serve;
* a worktree whose outgoing history carries a secret-shaped file is left behind;
* the recall (``--to local``) brings the destination's new work back IN PLACE;
* over the cap → ``working_set_over_cap``, ``changed: False``, nothing moved;
* a transfer failure rolls the move back and sweeps both stagings;
* an older destination (protocol 1) keeps PR A's "confirmed, files stay" behaviour.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest

from local_operator.network import mobility, workset, workset_move
from local_operator.session.placement import read_handoff_journal
from tests.unit.network.test_mobility import (  # noqa: F401 — `devices` is a fixture
    SESSION,
    Devices,
    _owned_session,
    devices,
)
from tests.unit.network.test_relay_e2e import _pair_settled
from tests.unit.network.workset_rig import GIT_ENV, Rig, git, make_rig, point_session_at


@pytest.fixture()
def pair(request: pytest.FixtureRequest) -> Devices:
    value: Devices = request.getfixturevalue("devices")
    _server_a, server_b, _host, _port = value
    server_b.bind_control()
    server_b.start()
    return value


@pytest.fixture()
def rig(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Rig:
    for key, value in GIT_ENV.items():
        monkeypatch.setenv(key, value)
    built = make_rig(tmp_path / "ws")
    monkeypatch.setenv("HOME", str(built.home))
    return built


@pytest.fixture()
def bhome(tmp_path: Path, rig: Rig) -> Path:
    """The destination's own home: a clone of the same origin at ``~/proj``."""
    home = tmp_path / "bhome"
    home.mkdir()
    git(home, "clone", "-q", str(rig.origin), str(home / "proj"))
    return home


def _setup(pair: Devices, monkeypatch: pytest.MonkeyPatch, rig: Rig, bhome: Path) -> Path:
    server_a, server_b, _host, _port = pair
    _pair_settled(pair, monkeypatch, role="admin", settings=server_b.settings)
    server_a.workset_home = rig.home  # type: ignore[attr-defined]
    server_b.workset_home = bhome  # type: ignore[attr-defined]
    source = _owned_session(server_a)
    point_session_at(source, rig.feat)
    return source


def _move(server: Any, monkeypatch: pytest.MonkeyPatch, **kwargs: Any) -> dict[str, Any]:
    monkeypatch.setenv("LOCAL_OPERATOR_CONFIG_DIR", str(server.root))
    return dict(mobility.request_move(SESSION, root=server.root, **kwargs))


def _confirmed(server: Any, monkeypatch: pytest.MonkeyPatch) -> dict[str, Any]:
    preflight = _move(server, monkeypatch, to="local", dry_run=True)
    assert preflight["ok"] is True, preflight
    return dict(preflight["working_set"])


def _state(tree: Path) -> dict[str, Any]:
    return {
        "head": git(tree, "rev-parse", "HEAD"),
        "branch": git(tree, "symbolic-ref", "--short", "HEAD"),
        "staged": git(tree, "diff", "--cached", "--name-status"),
        "unstaged": git(tree, "diff", "--name-status"),
        "untracked": sorted(git(tree, "ls-files", "--others", "--exclude-standard").split()),
    }


def _audit_events(root: Path) -> list[dict[str, Any]]:
    path = root / "network" / "audit.jsonl"
    if not path.is_file():
        return []
    return [json.loads(line) for line in path.read_text("utf-8").splitlines() if line.strip()]


def test_a_confirmed_move_carries_the_working_set(
    pair: Devices, monkeypatch: pytest.MonkeyPatch, rig: Rig, bhome: Path
) -> None:
    server_a, server_b, _host, _port = pair
    source = _setup(pair, monkeypatch, rig, bhome)
    block = _confirmed(server_b, monkeypatch)
    assert block["carries_files"] is True
    before = _state(rig.feat)
    feat_bytes = {p.name: p.read_bytes() for p in rig.feat.iterdir() if p.is_file()}

    moved = _move(server_b, monkeypatch, to="local", confirm_working_set=block["scope_digest"])

    assert moved["ok"] is True, moved
    assert not source.exists()
    ws = moved["working_set"]
    assert ws["transferred"] is True and ws["state"] == "restored", ws
    (row,) = [r for r in ws["worktrees"] if r["path"] == "~/proj/.worktrees/feat"]
    assert row["state"] == "restored", row
    assert row["dest_path"] == "~/proj/.worktrees/feat"
    landed = bhome / "proj" / ".worktrees" / "feat"
    after = _state(landed)
    # The untracked secret-shaped file is withheld; everything else arrives as such.
    assert ".env.local" in before["untracked"]
    assert after == {**before, "untracked": [n for n in before["untracked"] if n != ".env.local"]}
    assert not (landed / ".env.local").exists()
    assert not (landed / "node_modules").exists()
    assert "unpushed work" in git(landed, "log", "-1", "--format=%s")
    assert str(landed) in git(bhome / "proj", "worktree", "list")
    # The arrived session opens IN the restored tree, not the other device's path.
    marker = json.loads((server_b.root / "sessions" / SESSION / "desktop.json").read_text("utf-8"))
    assert marker["cwd"] == str(landed)
    # THE SOURCE NEVER DELETES: the worktree is byte-identical and marked departed.
    assert {p.name: p.read_bytes() for p in rig.feat.iterdir() if p.is_file()} == feat_bytes
    assert _state(rig.feat) == before
    git_dir = Path(git(rig.feat, "rev-parse", "--absolute-git-dir"))
    marker = json.loads((git_dir / "lop-moved.json").read_text("utf-8"))
    assert marker["to_device"] == server_b.identity.device_id
    assert (server_a.root / "network" / "workset-departed" / f"{SESSION}.json").is_file()
    # The owner audited the serve; its served artifacts are gone after ``done``.
    serves = [e for e in _audit_events(server_a.root) if e.get("event") == "session.workset.serve"]
    assert serves, "no session.workset.serve audit row"
    assert all(
        e["detail"]["to_device"] == server_b.identity.device_id for e in serves
    ), serves
    assert not workset_move.outgoing_dir(server_a.root, SESSION).exists()
    assert read_handoff_journal(server_a.root) == {}


def test_a_history_secret_worktree_is_left_behind(
    pair: Devices, monkeypatch: pytest.MonkeyPatch, rig: Rig, bhome: Path
) -> None:
    _server_a, server_b, _host, _port = pair
    _setup(pair, monkeypatch, rig, bhome)
    (rig.feat / ".env").write_text("TOKEN=x\n", encoding="utf-8")
    git(rig.feat, "add", "-f", ".env")
    git(rig.feat, "commit", "-q", "-m", "oops", "--", ".env")
    block = _confirmed(server_b, monkeypatch)

    moved = _move(server_b, monkeypatch, to="local", confirm_working_set=block["scope_digest"])

    assert moved["ok"] is True, moved
    (row,) = [r for r in moved["working_set"]["worktrees"] if r["path"].endswith("feat")]
    assert row["state"] == "left_behind" and row["reason"].startswith("history_"), row
    assert not (bhome / "proj" / ".worktrees" / "feat").exists()
    assert (rig.feat / ".env").is_file()


def test_the_recall_brings_new_work_back_in_place(
    pair: Devices, monkeypatch: pytest.MonkeyPatch, rig: Rig, bhome: Path
) -> None:
    server_a, server_b, _host, _port = pair
    _setup(pair, monkeypatch, rig, bhome)
    block = _confirmed(server_b, monkeypatch)
    moved = _move(server_b, monkeypatch, to="local", confirm_working_set=block["scope_digest"])
    assert moved["ok"] is True, moved
    landed = bhome / "proj" / ".worktrees" / "feat"
    # Work on B: a new commit and a new untracked file.
    (landed / "a.txt").write_text("a\nmore\nfrom b\n", encoding="utf-8")
    git(landed, "commit", "-q", "-am", "work on b")
    (landed / "b-notes.txt").write_text("from b\n", encoding="utf-8")
    on_b = _state(landed)
    # A recalls it (A runs the destination path; B is now the owner).
    monkeypatch.setenv("HOME", str(bhome))
    back_block = _confirmed(server_a, monkeypatch)
    assert back_block["carries_files"] is True, back_block

    back = _move(server_a, monkeypatch, to="local", confirm_working_set=back_block["scope_digest"])

    assert back["ok"] is True, back
    (row,) = [r for r in back["working_set"]["worktrees"] if r["path"].endswith("feat")]
    assert row["state"] == "restored", back["working_set"]
    assert row["dest_path"] == "~/proj/.worktrees/feat" and row["mode"] == "in_place", row
    # B's commit, index and files are back; A's own withheld ``.env.local`` (never sent,
    # never deleted) is still there beside them.
    assert _state(rig.feat) == {**on_b, "untracked": sorted([*on_b["untracked"], ".env.local"])}
    assert (rig.feat / ".env.local").is_file()


def test_over_the_cap_is_refused_and_nothing_moves(
    pair: Devices, monkeypatch: pytest.MonkeyPatch, rig: Rig, bhome: Path
) -> None:
    server_a, server_b, _host, _port = pair
    source = _setup(pair, monkeypatch, rig, bhome)
    block = _confirmed(server_b, monkeypatch)
    # The DESTINATION's cap (the owner's manifest-time cap passed): 1 byte.
    monkeypatch.setattr(workset, "cap_bytes", lambda root=None: 1)

    refused = _move(server_b, monkeypatch, to="local", confirm_working_set=block["scope_digest"])

    assert refused["ok"] is False and refused["code"] == "working_set_over_cap", refused
    assert refused["changed"] is False
    assert source.is_dir()
    assert not (server_b.root / "sessions" / SESSION).exists()
    assert not workset_move.incoming_staging(server_b.root, SESSION).exists()
    assert not (bhome / "proj" / ".worktrees" / "feat").exists()
    assert read_handoff_journal(server_a.root) == {}


def test_a_transfer_failure_rolls_back_and_sweeps_staging(
    pair: Devices, monkeypatch: pytest.MonkeyPatch, rig: Rig, bhome: Path
) -> None:
    server_a, server_b, _host, _port = pair
    source = _setup(pair, monkeypatch, rig, bhome)
    block = _confirmed(server_b, monkeypatch)
    real = workset_move._fetch_artifact
    seen: list[Path] = []

    def broken(transport: Any, **kw: Any) -> Path:
        # Leave a partial ``.part`` behind, then fail as a dropped link would.
        part = kw["into"] / (kw["digest"].removeprefix("sha256:") + ".part")
        part.write_bytes(b"partial")
        seen.append(part)
        raise workset_move.WorksetTransferRefused("working_set_transfer_failed", "link dropped")

    monkeypatch.setattr(workset_move, "_fetch_artifact", broken)
    refused = _move(server_b, monkeypatch, to="local", confirm_working_set=block["scope_digest"])
    monkeypatch.setattr(workset_move, "_fetch_artifact", real)

    assert refused["ok"] is False and refused["code"] == "working_set_transfer_failed", refused
    assert refused["changed"] is False
    assert seen and not seen[0].exists()
    assert not workset_move.incoming_staging(server_b.root, SESSION).exists()
    assert not (server_b.root / "network" / "staging" / SESSION).exists()
    assert source.is_dir()
    # The owner heard the rollback: no parked ``prepared``, no served artifacts left.
    assert read_handoff_journal(server_a.root) == {}
    assert not workset_move.outgoing_dir(server_a.root, SESSION).exists()


def test_an_older_destination_keeps_the_files_on_the_source(
    pair: Devices, monkeypatch: pytest.MonkeyPatch, rig: Rig, bhome: Path
) -> None:
    _server_a, server_b, _host, _port = pair
    _setup(pair, monkeypatch, rig, bhome)
    # The destination speaks protocol 1 (PR A): its prepare and preflight say so.
    monkeypatch.setattr(workset, "WORKSET_TRANSFER_VERSION", 1)
    block = _confirmed(server_b, monkeypatch)
    assert block["carries_files"] is False

    moved = _move(server_b, monkeypatch, to="local", confirm_working_set=block["scope_digest"])

    assert moved["ok"] is True, moved
    assert moved["working_set"]["transferred"] is False
    assert [r["reason"] for r in moved["working_set"]["worktrees"]] == ["not_transferred"]
    assert not (bhome / "proj" / ".worktrees" / "feat").exists()

