"""The outgoing-object scan withholds secret-shaped history (design §8.1, security review).

A bundle carries every object of the commits it contains, so path filtering
cannot help once a secret is committed (or staged: the snapshot's index tree is
outgoing too). ``stage_bundle`` scans EVERY outgoing object and refuses on any
finding; v1 has no opt-in to carry it anyway (manager decision 2026-11-18).
Secret-shaped strings are assembled at runtime so this file itself carries none.
"""

from __future__ import annotations

import hashlib
import inspect
import json
from pathlib import Path

import pytest

from local_operator.network import workset_transfer as wt
from tests.unit.network.workset_git_fixtures import (  # noqa: F401
    git,
    git_env,
    make_rig,
    write,
)

# Every test here runs real git; none may read system/global config or prompt
# (``git_env`` is imported for this mark, hence the F401 noqa on the import).
pytestmark = pytest.mark.usefixtures("git_env")

SID = "0123456789abcdef"
AWS_KEY_ID = "AKIA" + "Q3VZ7XK2M4N8P6RT"


def _commit(worktree: Path, files: dict[str, bytes | str], message: str = "work") -> None:
    for name, body in files.items():
        path = worktree / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(body if isinstance(body, bytes) else body.encode())
    git(worktree, "add", "--", *files)
    git(worktree, "commit", "-qm", message, "--", *files)


def _check(rig, **kw) -> wt.HistoryCheck:
    snap = wt.snapshot(rig.worktree, SID, 0)
    have = [git(rig.worktree, "rev-parse", "@{u}")]
    return wt.history_check(rig.src_primary, list(snap.refs), have, **kw)


def _whys(check: wt.HistoryCheck) -> set[tuple[str, str]]:
    return {(f["path"], f["why"]) for f in check.findings}


def test_unpushed_commit_adding_env_file_is_withheld_and_nothing_staged(tmp_path: Path) -> None:
    rig = make_rig(tmp_path, dirty=False)
    _commit(rig.worktree, {".env": "PLAIN=1\n"})
    snap = wt.snapshot(rig.worktree, SID, 0)
    have = [git(rig.worktree, "rev-parse", "@{u}")]
    out = tmp_path / "out"
    with pytest.raises(wt.HistoryWithheld) as info:
        wt.stage_bundle(rig.src_primary, snap.refs, have, out)
    assert info.value.reason == "history_secret"
    hits = [f for f in info.value.check.findings if f["path"] == ".env"]
    assert hits and hits[0]["why"] == "history_secret_path" and hits[0]["rule"]
    # Withheld means no bytes exist to serve.
    assert list(out.iterdir()) == []


def test_secret_content_under_an_innocent_name_is_withheld(tmp_path: Path) -> None:
    rig = make_rig(tmp_path, dirty=False)
    _commit(rig.worktree, {"src/settings.py": f"KEY_ID = '{AWS_KEY_ID}'\n"})
    check = _check(rig)
    hit = [f for f in check.findings if f["path"] == "src/settings.py"]
    assert (
        hit and hit[0]["why"] == "history_secret_content" and hit[0]["rule"] == "aws-access-key-id"
    )
    # Class name only, never the value.
    assert AWS_KEY_ID not in json.dumps(check.findings)


def test_a_blob_shared_by_a_benign_and_a_secret_name_is_judged_under_both(tmp_path: Path) -> None:
    rig = make_rig(tmp_path, dirty=False)
    _commit(rig.worktree, {"a.txt": "same bytes\n", "sub/.env": "same bytes\n"})
    assert ("sub/.env", "history_secret_path") in _whys(_check(rig))


def test_staged_secret_rides_the_index_tree_so_it_is_scanned(tmp_path: Path) -> None:
    rig = make_rig(tmp_path, dirty=False)
    write(rig.worktree / "notes.txt", f"id {AWS_KEY_ID}\n")
    git(rig.worktree, "add", "notes.txt")  # staged, never committed
    assert ("notes.txt", "history_secret_content") in _whys(_check(rig))


def test_untracked_secret_rides_the_worktree_tree_unless_excluded(tmp_path: Path) -> None:
    rig = make_rig(tmp_path, dirty=False)
    write(rig.worktree / ".env.local", "X=1\n")
    assert (".env.local", "history_secret_path") in _whys(_check(rig))
    snap = wt.snapshot(rig.worktree, SID, 0, excludes=(".env.local",))
    clean = wt.history_check(
        rig.src_primary, list(snap.refs), [git(rig.worktree, "rev-parse", "@{u}")]
    )
    assert clean.findings == ()


def test_secret_in_a_commit_message_is_withheld(tmp_path: Path) -> None:
    rig = make_rig(tmp_path, dirty=False)
    _commit(rig.worktree, {"ok.txt": "fine\n"}, message=f"use {AWS_KEY_ID}")
    assert any(
        f["why"] == "history_secret_content" and f["path"] == "" for f in _check(rig).findings
    )


def test_unscannable_blobs_are_withheld_not_skipped(tmp_path: Path) -> None:
    rig = make_rig(tmp_path, dirty=False)
    _commit(rig.worktree, {"logo.png": b"\x89PNG\r\n\x1a\n\x00\x01", "big.txt": "x" * 5000})
    findings = {(f["path"], f["why"], f["rule"]) for f in _check(rig, max_blob_bytes=4096).findings}
    assert ("logo.png", "history_unscannable", "binary") in findings
    assert ("big.txt", "history_unscannable", "over_scan_limit") in findings


def test_secret_the_destination_already_holds_is_not_outgoing(tmp_path: Path) -> None:
    rig = make_rig(tmp_path, dirty=False)
    _commit(rig.worktree, {".env": "PUSHED=1\n"})
    git(rig.worktree, "push", "-q", "origin", "feat")
    _commit(rig.worktree, {"next.txt": "next\n"})
    check = _check(rig)
    assert check.findings == ()


def test_clean_history_scan_is_bound_to_the_staged_artifact(tmp_path: Path) -> None:
    rig = make_rig(tmp_path)
    snap = wt.snapshot(rig.worktree, SID, 0)
    have = [git(rig.worktree, "rev-parse", "@{u}")]
    artifact = wt.stage_bundle(rig.src_primary, snap.refs, have, tmp_path / "out")
    objects = git(
        rig.src_primary, "rev-list", "--objects", "--no-object-names", *snap.refs, "--not", *have
    )
    expected = "sha256:" + hashlib.sha256("\n".join(sorted(objects.split())).encode()).hexdigest()
    assert artifact.scanned_digest == expected
    sidecar = json.loads((artifact.path.parent / f"{artifact.path.name}.json").read_text())
    assert sidecar["scanned_digest"] == expected
    assert sidecar["digest"] == artifact.digest


def test_v1_has_no_history_secret_override() -> None:
    for fn in (wt.stage_bundle, wt.history_check):
        assert not any(
            "include" in name or "secret" in name for name in inspect.signature(fn).parameters
        )
