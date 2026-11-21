"""Where a restored worktree lands, and that it never clobbers (design §7, §10, security review).

Rules under test: paths travel home-relative and are re-rooted under the
destination home; every component is checked on the real filesystem (no
symlink, alias or traversal); an occupied target is never overwritten — the
work lands beside it — except a recall onto a tree this device's own trusted
journal recorded as departed AND that re-measures unchanged since.
"""

from __future__ import annotations

import os
import time
from collections import namedtuple
from pathlib import Path

import pytest

from local_operator.network import workset_transfer as wt
from tests.unit.network.workset_git_fixtures import (  # noqa: F401
    facts,
    file_digest,
    git,
    git_env,
    make_rig,
    serve_origin_as_https,
    transfer,
    write,
)

# Every test here runs real git; none may read system/global config or prompt
# (``git_env`` is imported for this mark, hence the F401 noqa on the import).
pytestmark = pytest.mark.usefixtures("git_env")

SID = "0a1b2c3d4e5f6789"
NOW = time.mktime((2026, 11, 18, 9, 30, 0, 0, 0, -1))
STAMP = "202611180930"


def _move(rig, tmp_path, **kw):
    return transfer(
        src_worktree=kw.pop("src_worktree", rig.worktree),
        src_primary=kw.pop("src_primary", rig.src_primary),
        src_home=kw.pop("src_home", rig.src_home),
        dst_home=kw.pop("dst_home", rig.dst_home),
        work=tmp_path / kw.pop("work", "w1"),
        session_id=SID,
        now=NOW,
        **kw,
    )


def test_home_relative_paths_are_rerooted_under_the_destination_home(tmp_path: Path) -> None:
    src_home, dst_home = tmp_path / "srcHome", tmp_path / "destHome"
    (src_home / "x" / ".worktrees" / "w").mkdir(parents=True)
    dst_home.mkdir()
    rel = wt.home_relative(src_home / "x" / ".worktrees" / "w", home=src_home)
    assert rel == "~/x/.worktrees/w"
    assert (
        wt.remap(rel, home=dst_home) == Path(os.path.realpath(dst_home)) / "x" / ".worktrees" / "w"
    )
    assert wt.home_relative(tmp_path / "elsewhere", home=src_home) is None
    assert wt.home_relative(src_home, home=src_home) is None


@pytest.mark.parametrize(
    "bad", ["/abs/path", "~/../escape", "~/a/../b", "~/a/./b", "~//a", "~/a\\b", "relative", "~/"]
)
def test_unsafe_home_relative_forms_are_refused(tmp_path: Path, bad: str) -> None:
    with pytest.raises(wt.ContainmentError):
        wt.contained_path(tmp_path, bad)


def test_symlinked_component_is_refused_and_nothing_is_written_through_it(tmp_path: Path) -> None:
    rig = make_rig(tmp_path)
    elsewhere = tmp_path / "elsewhere"
    elsewhere.mkdir()
    os.symlink(elsewhere, rig.dst_primary / ".worktrees")
    with pytest.raises(wt.ContainmentError):
        wt.contained_path(rig.dst_home, "~/proj/.worktrees/feat")
    _, _, report = _move(rig, tmp_path)
    assert (report.state, report.reason) == ("left_behind", "unsafe_path")
    assert list(elsewhere.iterdir()) == []


def test_aliased_component_is_refused(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    (tmp_path / "proj").mkdir()
    real_listdir = os.listdir
    # A case-insensitive filesystem (APFS) opens "proj" for "Proj"; emulate it.
    monkeypatch.setattr(os.path, "lexists", lambda p: Path(str(p).replace("Proj", "proj")).exists())
    monkeypatch.setattr(os, "listdir", lambda p: real_listdir(p))
    with pytest.raises(wt.ContainmentError, match="aliased"):
        wt.contained_path(tmp_path, "~/Proj/x")


def test_existing_equal_state_is_an_idempotent_retry(tmp_path: Path) -> None:
    rig = make_rig(tmp_path)
    _, _, first = _move(rig, tmp_path, work="w1")
    _, _, second = _move(rig, tmp_path, work="w2")
    assert (first.mode, second.mode) == ("new", "idempotent")
    assert second.dest_path == first.dest_path == "~/proj/.worktrees/feat"
    assert not list((rig.dst_primary / ".worktrees").glob("feat.moved-*"))


def test_existing_different_tree_lands_beside_and_is_untouched(tmp_path: Path) -> None:
    rig = make_rig(tmp_path)
    occupied = rig.dst_primary / ".worktrees" / "feat"
    write(occupied / "mine.txt", "the destination user's own work\n")
    before = file_digest(occupied)
    _, _, report = _move(rig, tmp_path)
    assert report.mode == "beside"
    assert report.dest_path == f"~/proj/.worktrees/feat.moved-laptop-{STAMP}"
    assert report.remapped == ("~/proj/.worktrees/feat", report.dest_path)
    assert file_digest(occupied) == before
    beside = rig.dst_primary / ".worktrees" / f"feat.moved-laptop-{STAMP}"
    assert facts(beside) == facts(rig.worktree)


def test_source_primary_checkout_lands_as_a_moved_worktree(tmp_path: Path) -> None:
    rig = make_rig(tmp_path)
    write(rig.src_primary / "primary_dirty.txt", "dirty in the primary\n")
    dst_before = git(rig.dst_primary, "rev-parse", "HEAD")
    _, _, report = _move(rig, tmp_path, src_worktree=rig.src_primary, is_primary=True)
    assert report.state == "restored"
    assert report.dest_path == f"~/proj/.worktrees/moved-{SID[:8]}"
    moved = rig.dst_primary / ".worktrees" / f"moved-{SID[:8]}"
    assert (moved / "primary_dirty.txt").read_text() == "dirty in the primary\n"
    # The destination's own primary is never rewritten; main is checked out
    # there, so the moved tree gets its own branch.
    assert git(rig.dst_primary, "rev-parse", "HEAD") == dst_before
    assert report.branch == "main-moved-laptop"


def _depart_and_edit_remotely(rig, tmp_path: Path, config: Path):
    snap, _, report = _move(rig, tmp_path, work="out-a")
    assert report.state == "restored"
    wt.mark_departed(config, SID, "dev-b", [snap], name="B", home=rig.src_home)
    remote = rig.dst_primary / ".worktrees" / "feat"
    write(remote / "feat.txt", "edited on B\n")
    git(remote, "commit", "-qam", "made on B")
    write(remote / "from_b_untracked.txt", "b\n")
    write(remote / "mod_unstaged.txt", "three\n")
    return remote


def _recall(rig, tmp_path: Path, remote: Path, config: Path | None):
    return transfer(
        src_worktree=remote,
        src_primary=rig.dst_primary,
        src_home=rig.dst_home,
        dst_home=rig.src_home,
        work=tmp_path / "back",
        session_id=SID,
        device="dev-b",
        config=config,
        now=NOW,
    )[2]


def test_recall_onto_an_unchanged_departed_tree_updates_in_place(tmp_path: Path) -> None:
    rig = make_rig(tmp_path)
    config = tmp_path / "config-a"
    remote = _depart_and_edit_remotely(rig, tmp_path, config)
    report = _recall(rig, tmp_path, remote, config)
    assert (report.state, report.mode, report.dest_path) == (
        "restored",
        "in_place",
        "~/proj/.worktrees/feat",
    )
    assert report.remapped is None
    assert facts(rig.worktree) == facts(remote)
    assert git(rig.worktree, "log", "-1", "--format=%s") == "made on B"
    assert (rig.worktree / "from_b_untracked.txt").read_text() == "b\n"
    assert git(rig.worktree, "symbolic-ref", "--short", "HEAD") == "feat"


def test_recall_lands_beside_when_the_tree_changed_since_departure(tmp_path: Path) -> None:
    rig = make_rig(tmp_path)
    config = tmp_path / "config-a"
    remote = _depart_and_edit_remotely(rig, tmp_path, config)
    write(rig.worktree / "edited_here_meanwhile.txt", "local edit\n")
    before = file_digest(rig.worktree)
    report = _recall(rig, tmp_path, remote, config)
    assert report.mode == "beside"
    assert report.dest_path == f"~/proj/.worktrees/feat.moved-dev-b-{STAMP}"
    assert file_digest(rig.worktree) == before


def test_recall_without_the_trusted_journal_lands_beside(tmp_path: Path) -> None:
    rig = make_rig(tmp_path)
    config = tmp_path / "config-a"
    remote = _depart_and_edit_remotely(rig, tmp_path, config)
    before = file_digest(rig.worktree)
    # The marker in the tree alone must not unlock an in-place update.
    assert (Path(git(rig.worktree, "rev-parse", "--absolute-git-dir")) / "lop-moved.json").is_file()
    report = _recall(rig, tmp_path, remote, tmp_path / "config-without-journal")
    assert report.mode == "beside"
    assert file_digest(rig.worktree) == before


def test_publish_never_clobbers_a_path_that_appears_mid_restore(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    rig = make_rig(tmp_path)
    real = wt._rename_noreplace

    def racing(src: Path, dst: Path) -> None:
        dst.mkdir()  # somebody creates the target between the check and the publish
        real(src, dst)

    monkeypatch.setattr(wt, "_rename_noreplace", racing)
    _, _, report = _move(rig, tmp_path)
    assert (report.state, report.reason) == ("pending_restore", "destination_conflict")
    target = rig.dst_primary / ".worktrees" / "feat"
    assert target.is_dir() and list(target.iterdir()) == []
    listed = git(rig.dst_primary, "worktree", "list", "--porcelain")
    assert ".lop-restore-" not in listed and "feat" not in listed
    assert not list(target.parent.glob(".lop-restore-*"))


def test_rename_noreplace_refuses_an_existing_empty_directory(tmp_path: Path) -> None:
    src, dst = tmp_path / "src", tmp_path / "dst"
    src.mkdir()
    (src / "f").write_text("x")
    dst.mkdir()
    with pytest.raises(FileExistsError):
        wt._rename_noreplace(src, dst)
    assert (src / "f").read_text() == "x" and list(dst.iterdir()) == []


def test_primary_resolution(tmp_path: Path) -> None:
    rig = make_rig(tmp_path)
    url = str(rig.origin)
    assert wt.resolve_primary(url, "~/proj", home=rig.dst_home).state == "present"
    assert wt.resolve_primary(url, "~/absent", home=rig.dst_home).state == "absent"
    other = rig.dst_home / "other"
    git(tmp_path, "init", "-q", str(other))
    assert wt.resolve_primary(url, "~/other", home=rig.dst_home).reason == "destination_conflict"
    (rig.dst_home / "plain").mkdir()
    assert wt.resolve_primary(url, "~/plain", home=rig.dst_home).reason == "destination_conflict"


def test_remote_urls_compare_across_transports() -> None:
    forms = [
        "git@github.com:damianvtran/local-operator.git",
        "ssh://git@github.com/damianvtran/local-operator",
        "https://user:tok@GitHub.com/damianvtran/local-operator.git/",
    ]
    assert {wt.normalize_remote_url(f) for f in forms} == {"github.com/damianvtran/local-operator"}
    assert wt.normalize_remote_url("https://github.com/a/b") != wt.normalize_remote_url(
        "https://github.com/a/c"
    )


def test_absent_primary_is_cloned_then_restored(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    rig = make_rig(tmp_path, clone_dest=False)
    url = serve_origin_as_https(monkeypatch, rig.origin, rig.src_primary)
    res = wt.clone_primary(url, "~/proj", home=rig.dst_home, pack_bytes=0)
    assert (res.state, res.cloned) == ("present", True)
    assert not list(rig.dst_home.glob(".lop-clone-*"))
    _, _, report = _move(rig, tmp_path)
    assert report.state == "restored"
    assert facts(rig.dst_home / "proj" / ".worktrees" / "feat") == facts(rig.worktree)


def test_clone_refused_without_free_space(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    rig = make_rig(tmp_path, clone_dest=False)
    url = serve_origin_as_https(monkeypatch, rig.origin, rig.src_primary)
    usage = namedtuple("usage", "total used free")
    monkeypatch.setattr(wt.shutil, "disk_usage", lambda _p: usage(10, 10, 0))
    with pytest.raises(wt.MoveWorksetError) as info:
        wt.clone_primary(url, "~/proj", home=rig.dst_home, pack_bytes=1)
    assert info.value.reason == "insufficient_space"
    assert not (rig.dst_home / "proj").exists()
