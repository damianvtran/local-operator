"""Which worktrees a move's working set holds, and their facts (design §3.2, §3.3, §14).

Real git in ``tmp_path`` (``workset_rig``), read through ``workset.compute_manifest`` —
the exact function the owner's gate and preflight call. Each cell pins one discovery
or fact rule:

* the recorded cwd seeds its worktree; a transcript mention, a provenance record and
  a subagent child's cwd each seed one too, and say which (``sources``);
* a sibling worktree of the same repository that the session never touched is NOT in
  the set;
* staged / unstaged / untracked / unpushed counts are exact;
* a worktree another LIVE session is using is ``left_behind: contended``; one outside
  home is ``outside_home``; one with merge conflicts is ``unmerged``;
* a primary checkout whose ``.worktrees/`` is not ignored does not count its nested
  registered worktrees as its own untracked files;
* the scope digest ignores byte-only edits and changes with the destination device.
"""

from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from local_operator.network import workset
from tests.unit.network.workset_rig import Rig, git, make_rig, point_session_at

SID = "a1b2c3d4e5f6"
CHILD = "0f0e0d0c0b0a"


@pytest.fixture()
def rig(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Rig:
    built = make_rig(tmp_path / "ws")
    monkeypatch.setenv("HOME", str(built.home))
    # The secret classifier's config-dir rule must not see the rig as config.
    monkeypatch.setenv("LOCAL_OPERATOR_CONFIG_DIR", str(tmp_path / "cfg"))
    return built


@pytest.fixture()
def root(tmp_path: Path) -> Path:
    sessions = tmp_path / "cfg" / "sessions" / SID
    sessions.mkdir(parents=True)
    (sessions / "transcript.jsonl").write_text("", encoding="utf-8")
    return tmp_path / "cfg"


def _manifest(root: Path, rig: Rig, **kwargs: object) -> workset.Manifest:
    kwargs.setdefault("to_device", "d_dest")
    return workset.compute_manifest(root, SID, home=rig.home, **kwargs)  # type: ignore[arg-type]


def _rows(manifest: workset.Manifest) -> dict[str, workset.WorktreeFacts]:
    return {row.path: row for row in manifest.worktrees}


def test_the_cwd_seeds_its_worktree_with_exact_counts(root: Path, rig: Rig) -> None:
    point_session_at(root / "sessions" / SID, rig.feat)

    rows = _rows(_manifest(root, rig))

    assert list(rows) == ["~/proj/.worktrees/feat"]  # the sibling `other` is NOT included
    feat = rows["~/proj/.worktrees/feat"]
    assert feat.sources == ("cwd",)
    assert feat.repo == "~/proj"
    assert (feat.branch, feat.upstream) == ("feat", "origin/feat")
    assert feat.unpushed_commits == 1
    assert feat.staged == 2  # s1.txt, s2.txt
    assert feat.unstaged == 2  # b.txt modified, c.txt deleted
    assert feat.untracked == 2  # notes.txt, .env.local (node_modules/ is ignored)
    assert feat.state == "planned"
    assert feat.head == git(rig.feat, "rev-parse", "HEAD")


def test_a_transcript_mention_alone_finds_the_worktree(root: Path, rig: Rig) -> None:
    session = root / "sessions" / SID
    point_session_at(session, rig.home)  # cwd is home: not a repository
    (session / "transcript.jsonl").write_text(
        json.dumps({"type": "user", "content": "cd ~/proj/.worktrees/other && make"}) + "\n",
        encoding="utf-8",
    )

    rows = _rows(_manifest(root, rig))

    assert list(rows) == ["~/proj/.worktrees/other"]
    assert rows["~/proj/.worktrees/other"].sources == ("mention",)


def test_a_provenance_record_seeds_its_worktree(root: Path, rig: Rig) -> None:
    session = root / "sessions" / SID
    point_session_at(session, rig.home)
    records = session / ".file-provenance"
    records.mkdir()
    (records / "p1.json").write_text(
        json.dumps({"file": {"path": str(rig.feat / "b.txt"), "locator": "arg:path"}}),
        encoding="utf-8",
    )

    rows = _rows(_manifest(root, rig))

    assert rows["~/proj/.worktrees/feat"].sources == ("provenance",)


def test_a_subagent_childs_worktree_is_attributed_to_the_child(root: Path, rig: Rig) -> None:
    parent = root / "sessions" / SID
    point_session_at(parent, rig.home)
    (parent / "subagent-lane-job1.v1.json").write_text(
        json.dumps({"child_session_id": CHILD}), encoding="utf-8"
    )
    child = root / "sessions" / CHILD
    point_session_at(child, rig.other)

    rows = _rows(_manifest(root, rig))

    assert workset.session_tree(root, SID) == [SID, CHILD]
    assert rows["~/proj/.worktrees/other"].sources == (f"subagent:{CHILD}",)


def test_a_worktree_another_live_session_uses_is_left_behind_contended(
    root: Path, rig: Rig, monkeypatch: pytest.MonkeyPatch
) -> None:
    from local_operator.session.runtime import registry

    point_session_at(root / "sessions" / SID, rig.feat)
    other = SimpleNamespace(session_id="ffffffffffff", cwd=str(rig.feat / "src"))
    monkeypatch.setattr(registry, "scan", lambda *_a, **_k: [(other, "live")])

    feat = _rows(_manifest(root, rig))["~/proj/.worktrees/feat"]

    assert (feat.state, feat.reason, feat.contended_by) == (
        "left_behind",
        "contended",
        ("ffffffffffff",),
    )


def test_a_worktree_outside_home_is_left_behind(
    root: Path, rig: Rig, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    point_session_at(root / "sessions" / SID, rig.feat)
    elsewhere = tmp_path / "elsewhere"
    elsewhere.mkdir()

    manifest = workset.compute_manifest(root, SID, to_device="d", home=elsewhere)

    assert [(row.state, row.reason) for row in manifest.worktrees] == [
        ("left_behind", "outside_home")
    ]


def test_a_worktree_with_merge_conflicts_is_left_behind_unmerged(root: Path, rig: Rig) -> None:
    other = rig.other
    (other / "a.txt").write_text("theirs\n", encoding="utf-8")
    git(other, "commit", "-q", "-am", "theirs")
    (rig.feat / "a.txt").write_text("ours\n", encoding="utf-8")
    git(rig.feat, "stash", "-q", "--include-untracked")
    with pytest.raises(Exception):  # the merge stops on the conflict; that is the state
        git(rig.feat, "merge", "-q", "other")
    point_session_at(root / "sessions" / SID, rig.feat)

    feat = _rows(_manifest(root, rig))["~/proj/.worktrees/feat"]

    assert (feat.state, feat.reason) == ("left_behind", "unmerged")


def test_a_primarys_nested_worktrees_are_not_its_untracked_files(root: Path, rig: Rig) -> None:
    # The rig's primary does NOT ignore `.worktrees/`, which is the shape the transfer
    # half warned about: git reports each nested worktree as an untracked directory.
    raw = git(rig.primary, "status", "--porcelain", "--untracked-files=all")
    assert ".worktrees/" in raw
    point_session_at(root / "sessions" / SID, rig.primary)

    primary = _rows(_manifest(root, rig))["~/proj"]

    assert primary.untracked == 0, primary
    assert primary.is_primary is True


def test_the_scope_digest_ignores_bytes_and_binds_the_device(root: Path, rig: Rig) -> None:
    point_session_at(root / "sessions" / SID, rig.feat)
    before = workset.scope_digest(_manifest(root, rig))
    with (rig.feat / "b.txt").open("a", encoding="utf-8") as handle:
        handle.write("more\n")

    assert workset.scope_digest(_manifest(root, rig)) == before
    assert workset.scope_digest(_manifest(root, rig, to_device="d_other")) != before
    (rig.feat / "new.txt").write_text("x\n", encoding="utf-8")
    # A new untracked (non-secret) file is the same scope: same tree, same withheld set.
    assert workset.scope_digest(_manifest(root, rig)) == before


def test_the_block_carries_no_contents_and_no_owner_local_paths(root: Path, rig: Rig) -> None:
    point_session_at(root / "sessions" / SID, rig.feat)

    text = json.dumps(_manifest(root, rig).to_dict())

    assert str(rig.home) not in text  # home-relative only; abs_* never leave the owner
    assert "API_URL" not in text  # never a file's contents
