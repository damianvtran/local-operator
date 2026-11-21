"""Bundles carry only what the destination lacks (design §4 W1/W2/W5, §8.4)."""

from __future__ import annotations

from pathlib import Path

import pytest

from local_operator.network import workset_transfer as wt
from tests.unit.network.workset_git_fixtures import git, git_env, make_rig  # noqa: F401

# Every test here runs real git; none may read system/global config or prompt
# (``git_env`` is imported for this mark, hence the F401 noqa on the import).
pytestmark = pytest.mark.usefixtures("git_env")

SID = "fedcba9876543210"


def _rig_with_history(tmp_path: Path):
    return make_rig(tmp_path, bulk=20)


def test_have_containing_upstream_tip_makes_prerequisites_equal_have(tmp_path: Path) -> None:
    rig = _rig_with_history(tmp_path)
    snap = wt.snapshot(rig.worktree, SID, 0)
    upstream = git(rig.worktree, "rev-parse", "@{u}")
    have = wt.compute_have(rig.dst_primary, [snap.head, upstream, "0" * 40])
    # The destination holds the pushed tip, not the unpushed HEAD or a bogus id.
    assert have == [upstream]

    delta = wt.stage_bundle(rig.src_primary, snap.refs, have, tmp_path / "out-delta")
    full = wt.stage_bundle(rig.src_primary, snap.refs, [], tmp_path / "out-full")

    assert delta.prerequisites == (upstream,)
    assert wt.read_bundle_header(delta.path).prerequisites == (upstream,)
    assert full.prerequisites == ()
    assert delta.bytes * 10 < full.bytes, (delta.bytes, full.bytes)
    # The head ref is covered by nothing the destination holds, so all three ride.
    assert {name for name, _ in wt.read_bundle_header(delta.path).refs} == set(snap.refs)
    assert wt.verify_bundle(rig.dst_primary, delta.path)
    # Artifact is immutable and digest-named in a private directory.
    assert delta.path.name == delta.digest.removeprefix("sha256:")
    assert delta.path.stat().st_mode & 0o777 == 0o600
    assert (tmp_path / "out-delta").stat().st_mode & 0o777 == 0o700


def test_empty_have_makes_a_full_bundle_a_fresh_repo_accepts(tmp_path: Path) -> None:
    rig = make_rig(tmp_path)
    snap = wt.snapshot(rig.worktree, SID, 0)
    full = wt.stage_bundle(rig.src_primary, snap.refs, [], tmp_path / "out")
    empty = tmp_path / "empty"
    git(tmp_path, "init", "-q", str(empty))
    assert full.prerequisites == ()
    assert wt.verify_bundle(empty, full.path)


def test_same_inputs_reuse_the_staged_artifact(tmp_path: Path) -> None:
    rig = make_rig(tmp_path)
    snap = wt.snapshot(rig.worktree, SID, 0)
    have = [git(rig.worktree, "rev-parse", "@{u}")]
    first = wt.stage_bundle(rig.src_primary, snap.refs, have, tmp_path / "out")
    second = wt.stage_bundle(rig.src_primary, snap.refs, have, tmp_path / "out")
    assert first.digest == second.digest and first.path == second.path
    assert first.scanned_digest == second.scanned_digest
    assert sorted(p.name for p in (tmp_path / "out").iterdir()) == sorted(
        [first.path.name, f"{first.path.name}.json"]
    )


def test_verify_failure_replans_once_with_empty_have(tmp_path: Path) -> None:
    rig = make_rig(tmp_path)
    snap = wt.snapshot(rig.worktree, SID, 0)
    upstream = git(rig.worktree, "rev-parse", "@{u}")
    # A destination that claimed the upstream tip but does not hold it (a
    # stale have): the delta bundle's prerequisite is missing there.
    lacking = tmp_path / "lacking"
    git(tmp_path, "init", "-q", str(lacking))
    calls: list[list[str]] = []

    def fetch(have):
        calls.append(list(have))
        return wt.stage_bundle(rig.src_primary, snap.refs, have, tmp_path / f"out{len(calls)}").path

    bundle, replanned = wt.verify_or_replan(fetch, lacking, [upstream])
    assert replanned is True
    assert calls == [[upstream], []]
    assert wt.read_bundle_header(bundle).prerequisites == ()


def test_second_verify_failure_raises_for_rollback(tmp_path: Path) -> None:
    rig = make_rig(tmp_path)
    garbage = tmp_path / "garbage.bundle"
    garbage.write_bytes(b"# v2 git bundle\n\nnot a pack")
    calls: list[list[str]] = []

    def fetch(have):
        calls.append(list(have))
        return garbage

    with pytest.raises(wt.MoveWorksetError) as info:
        wt.verify_or_replan(fetch, rig.dst_primary, ["a" * 40])
    assert info.value.reason == "bundle_unverifiable"
    assert calls == [["a" * 40], []]


def test_a_ref_moved_after_the_scan_is_never_bundled_unscanned(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    rig = make_rig(tmp_path)
    snap = wt.snapshot(rig.worktree, SID, 0)
    real_check = wt.history_check

    def check_then_move(repo, tips, have, **kw):
        result = real_check(repo, tips, have, **kw)
        # Between scan and bundle a writer repoints a snapshot ref.
        git(rig.src_primary, "update-ref", snap.refs[2], git(rig.src_primary, "rev-parse", "main"))
        return result

    monkeypatch.setattr(wt, "history_check", check_then_move)
    with pytest.raises(wt.MoveWorksetError) as info:
        wt.stage_bundle(rig.src_primary, snap.refs, [], tmp_path / "out")
    assert info.value.reason == "bundle_unbound"
    assert [p for p in (tmp_path / "out").iterdir()] == []
