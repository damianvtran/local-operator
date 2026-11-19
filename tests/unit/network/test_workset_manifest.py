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


def test_a_transcript_mention_alone_lists_the_worktree_as_mentioned_never_planned(
    root: Path, rig: Rig
) -> None:
    """Review R8 / security S5 (round 1): a repository the session only READ about is
    shown as ``mentioned`` — not planned, not counted, not a digest input — so it can
    never be auto-carried and a chat-only session needs no confirmation."""
    session = root / "sessions" / SID
    point_session_at(session, rig.home)  # cwd is home: not a repository
    (session / "transcript.jsonl").write_text(
        json.dumps({"type": "user", "content": "cat ~/proj/.worktrees/other/README"}) + "\n",
        encoding="utf-8",
    )

    manifest = _manifest(root, rig)

    assert manifest.worktrees == ()
    assert manifest.empty is True  # no confirmation for a session that only read
    assert manifest.mentioned == ("~/proj/.worktrees/other",)
    block = manifest.to_dict()
    assert block["mentioned"] == ["~/proj/.worktrees/other"]
    assert block["totals"]["worktrees"] == 0


@pytest.mark.parametrize("codec", ["gzip", "zstd"])
def test_a_compressed_transcripts_mention_is_still_read(
    root: Path, rig: Rig, codec: str
) -> None:
    """CI guards (round 1): the mention scan reads through the archive-aware
    resolver, so a session the compression pass archived (``.zst`` as well as
    ``.gz``) seeds the same ``mentioned`` row as a plain one — before the fix the
    ``.zst`` form read as "no transcript" and the mention silently vanished."""
    from local_operator.session import compression

    if codec == "zstd" and compression.available_codec() != "zstd":
        pytest.skip("this build has no zstd codec")
    session = root / "sessions" / SID
    point_session_at(session, rig.home)
    (session / "transcript.jsonl").write_text(
        json.dumps({"type": "user", "content": "cat ~/proj/.worktrees/other/README"}) + "\n",
        encoding="utf-8",
    )
    assert compression.compress_session(session, codec=codec).status == "compressed"
    assert not (session / "transcript.jsonl").exists()

    manifest = _manifest(root, rig)

    assert manifest.mentioned == ("~/proj/.worktrees/other",)
    # A read-only probe: the archive is not re-inflated beside the session.
    assert not (session / "transcript.jsonl").exists()


def test_a_mention_beside_a_cwd_does_not_widen_the_planned_set_or_the_digest(
    root: Path, rig: Rig
) -> None:
    session = root / "sessions" / SID
    point_session_at(session, rig.feat)
    before = _manifest(root, rig)
    (session / "transcript.jsonl").write_text(
        json.dumps({"type": "user", "content": "see ~/proj/.worktrees/other/x"}) + "\n",
        encoding="utf-8",
    )

    after = _manifest(root, rig)

    assert [row.path for row in after.worktrees] == ["~/proj/.worktrees/feat"]
    assert after.mentioned == ("~/proj/.worktrees/other",)
    assert workset.scope_digest(after) == workset.scope_digest(before)


def test_a_subagent_childs_mention_alone_is_mentioned_too(root: Path, rig: Rig) -> None:
    parent = root / "sessions" / SID
    point_session_at(parent, rig.home)
    (parent / "subagent-lane-job1.v1.json").write_text(
        json.dumps({"child_session_id": CHILD}), encoding="utf-8"
    )
    child = root / "sessions" / CHILD
    point_session_at(child, rig.home)
    (child / "transcript.jsonl").write_text(
        json.dumps({"content": "read ~/proj/.worktrees/other/notes"}) + "\n", encoding="utf-8"
    )

    manifest = _manifest(root, rig)

    assert manifest.worktrees == ()
    assert manifest.mentioned == ("~/proj/.worktrees/other",)


def test_strictness_is_decided_per_repository_not_by_its_first_top(
    rig: Rig, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Review R10 (round 2): seeds are ordered cwd, provenance, then mentions, so a
    root's mention of worktree A could be read before a subagent child's cwd in
    worktree B of the SAME repository. Listing that repository quietly because A came
    first meant a failing ``git worktree list`` left ``worktrees == [A]`` and B was
    dropped without a word — a fail-open on the one path the gate exists to close."""
    other = rig.primary / ".worktrees" / "other"
    assert other.is_dir(), other
    real_git = workset._git

    def fail_on_other(top: Path, *args: str, **kwargs: object) -> object:
        # The quiet listing asks the top it was handed; make that one fail, so only a
        # strict listing taken from the top the session WORKED IN can list the repo.
        if args[:2] == ("worktree", "list") and Path(top) == other:
            raise workset.WorksetUnknown("worktree_list", str(top), "git failed")
        return real_git(top, *args, **kwargs)  # type: ignore[arg-type]

    monkeypatch.setattr(workset, "_git", fail_on_other)

    candidates = workset.discover_worktrees(
        [
            workset.Seed(str(other / "README.md"), "mention"),  # first: mention-only
            workset.Seed(str(rig.feat), "subagent:child1"),  # then: a real cwd
        ]
    )

    paths = sorted(str(candidate.path) for candidate in candidates)
    assert paths == sorted([str(other), str(rig.feat)]), paths
    assert rig.feat in [candidate.path for candidate in candidates]


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


def test_a_size_crossing_a_power_of_two_keeps_its_digest(root: Path, rig: Rig) -> None:
    """Review R2 (round 1) then R11 (round 2): an active session writing while queued
    crossed 512/1024 bytes and every resume refused ``working_set_changed``.

    R2 took the bucket out while the move carried no files; R11 takes it out when it
    DOES carry them, because the transfer half (#4245) re-checks the cap against the
    current manifest and the gate re-checks ``over_cap`` — so the bucket only brought
    the queued-move refusal back. What binds a sends-consent is ``carries_files``, and
    nothing about the size.
    """
    from dataclasses import replace

    point_session_at(root / "sessions" / SID, rig.feat)
    small = _manifest(root, rig)
    assert small.total_bytes < 1024, small.total_bytes
    with (rig.feat / "notes.txt").open("a", encoding="utf-8") as handle:
        handle.write("x" * 4096)  # crosses at least two power-of-two buckets
    large = _manifest(root, rig)
    assert large.total_bytes > 2 * small.total_bytes

    assert workset.scope_digest(large) == workset.scope_digest(small)
    sends_small = workset.scope_digest(replace(small, carries_files=True))
    sends_large = workset.scope_digest(replace(large, carries_files=True))
    assert sends_small == sends_large  # R11: no size input at all
    # ...while a "they stay" consent still cannot authorise a move that sends them.
    assert sends_small != workset.scope_digest(small)
    assert workset._artifact_binding(replace(small, carries_files=True)) == {"carries_files": True}


def test_a_broken_gitdir_pointer_fails_closed(root: Path, rig: Rig, tmp_path: Path) -> None:
    """Review R4 (round 1): a ``.git`` file pointing at a removed gitdir read as "no
    working files" and the move went ahead without a word."""
    proj2 = rig.home / "proj2"
    proj2.mkdir()
    (proj2 / ".git").write_text(f"gitdir: {tmp_path / 'gone'}\n", encoding="utf-8")
    (proj2 / "work.txt").write_text("uncommitted\n", encoding="utf-8")
    point_session_at(root / "sessions" / SID, proj2)

    with pytest.raises(workset.WorksetUnknown) as failure:
        _manifest(root, rig)

    assert failure.value.step == "discover"
    assert failure.value.cause == "not_a_repository"


def test_dubious_ownership_is_classed_not_echoed() -> None:
    """U6: the cause is a CODE the surfaces translate, never git's text."""
    text = b"fatal: detected dubious ownership in repository at '/x'\n"
    assert workset.git_failure_cause(text) == "dubious_ownership"
    assert workset.git_failure_cause(exc=FileNotFoundError("git")) == "git_missing"
    assert workset.git_failure_cause(b"fatal: index file smaller than expected") == "index_corrupt"
    assert workset.git_failure_cause(b"something else") == "git_error"


def test_a_broken_repository_only_mentioned_does_not_refuse(
    root: Path, rig: Rig, tmp_path: Path
) -> None:
    proj2 = rig.home / "proj2"
    proj2.mkdir()
    (proj2 / ".git").write_text(f"gitdir: {tmp_path / 'gone'}\n", encoding="utf-8")
    session = root / "sessions" / SID
    point_session_at(session, rig.feat)
    (session / "transcript.jsonl").write_text(
        json.dumps({"content": "ls ~/proj2/src"}) + "\n", encoding="utf-8"
    )

    manifest = _manifest(root, rig)

    assert [row.path for row in manifest.worktrees] == ["~/proj/.worktrees/feat"]


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
