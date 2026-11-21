"""The source keeps every byte of a moved worktree (design §8.6, engine half).

Moving is copy + mark, never move + delete: a failed or regretted move must
lose nothing, and cleanup is a separate operator-approved action. After the
engine's whole source path (snapshot, stage, departure mark) the worktree's
files, index, HEAD and branches are byte-identical; only private refs, the
staged artifact, a journal under the config root and a marker in the git dir
were added.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from local_operator.network import workset_transfer as wt
from tests.unit.network.workset_git_fixtures import (  # noqa: F401
    file_digest,
    git,
    git_env,
    make_rig,
    transfer,
)

# Every test here runs real git; none may read system/global config or prompt
# (``git_env`` is imported for this mark, hence the F401 noqa on the import).
pytestmark = pytest.mark.usefixtures("git_env")

SID = "99aa88bb77cc66dd"


def _source_state(rig) -> dict[str, object]:
    index = Path(git(rig.worktree, "rev-parse", "--path-format=absolute", "--git-path", "index"))
    return {
        "files": file_digest(rig.worktree),
        "index": index.read_bytes(),
        "head": git(rig.worktree, "rev-parse", "HEAD"),
        "symbolic": git(rig.worktree, "symbolic-ref", "HEAD"),
        "branches": git(rig.src_primary, "for-each-ref", "refs/heads", "refs/remotes"),
        "stash": git(rig.src_primary, "stash", "list"),
        "primary_files": {
            k: v
            for k, v in file_digest(rig.src_primary).items()
            if not k.startswith((".worktrees", ".git/"))
        },
    }


def test_a_completed_move_leaves_the_source_byte_identical_and_marked(tmp_path: Path) -> None:
    rig = make_rig(tmp_path)
    before = _source_state(rig)
    snap, _, report = transfer(
        src_worktree=rig.worktree,
        src_primary=rig.src_primary,
        src_home=rig.src_home,
        dst_home=rig.dst_home,
        work=tmp_path / "work",
        session_id=SID,
    )
    assert report.state == "restored"
    config = tmp_path / "config"
    marked = wt.mark_departed(config, SID, "dev-b", [snap], name="B", home=rig.src_home, now=1234.5)
    # Run twice: a retried commit step must not change anything either.
    wt.mark_departed(config, SID, "dev-b", [snap], name="B", home=rig.src_home, now=1234.5)

    assert _source_state(rig) == before
    assert marked == [{"path": "~/proj/.worktrees/feat", "at": 1234.5}]
    marker = Path(git(rig.worktree, "rev-parse", "--absolute-git-dir")) / "lop-moved.json"
    assert json.loads(marker.read_text()) == {
        "to_device": "dev-b",
        "name": "B",
        "at": 1234.5,
        "session_id": SID,
    }
    journal = json.loads((config / "network" / "workset-departed" / f"{SID}.json").read_text())
    (entry,) = journal["worktrees"]
    assert (entry["head"], entry["index_tree"], entry["worktree_tree"]) == (
        snap.head,
        snap.index_tree,
        snap.worktree_tree,
    )
    assert (config / "network" / "workset-departed").stat().st_mode & 0o777 == 0o700
    # The restore point stays on the source.
    refs = git(rig.src_primary, "for-each-ref", "--format=%(refname)", f"refs/lop/move/{SID}")
    assert refs.splitlines() == list(snap.refs)


def test_withheld_history_still_leaves_the_source_untouched(tmp_path: Path) -> None:
    rig = make_rig(tmp_path)
    (rig.worktree / ".env").write_text("A=1\n")
    git(rig.worktree, "add", ".env")
    before = _source_state(rig)
    snap = wt.snapshot(rig.worktree, SID, 0)
    try:
        wt.stage_bundle(rig.src_primary, snap.refs, [], tmp_path / "out")
    except wt.HistoryWithheld:
        pass
    else:  # pragma: no cover - the staged .env must be withheld
        raise AssertionError("staged .env was not withheld")
    assert _source_state(rig) == before
