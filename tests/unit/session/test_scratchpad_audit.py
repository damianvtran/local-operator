"""The read-only scratchpad audit: shapes, sizes, pointers, bounds, no writes.

Every row here is a fact an operator acts on before choosing a cleanup switch,
so the unit is the walk's own (allocated bytes) and the negative cases matter
as much as the positive: a symlink that is followed reports bytes the pad does
not hold, and a user pad that is missing from the report is the one pad nobody
is told they can never reclaim.
"""

from __future__ import annotations

import json
import os
from pathlib import Path

from local_operator.session import scratchpad_audit as audit_module
from local_operator.session.scratchpad_audit import audit_scratchpads


def _session(root: Path, name: str, origin: str | None = "subagent") -> Path:
    directory = root / "sessions" / name
    directory.mkdir(parents=True)
    if origin is not None:
        (directory / "origin.json").write_text(json.dumps({"origin": origin}))
    return directory


def _pad(session: Path) -> Path:
    pad = session / "scratchpad"
    pad.mkdir()
    return pad


def _bytes(path: Path) -> int:
    return path.lstat().st_blocks * 512


def _subtree(path: Path) -> int:
    """Allocated bytes of every non-directory entry under ``path``."""
    total = 0
    for base, _dirs, files in os.walk(path):
        for name in files:
            total += _bytes(Path(base) / name)
    return total


def test_reports_the_shape_vocabulary_with_their_real_sizes(tmp_path: Path) -> None:
    """node_modules, build and target are the requirement; the sizes are the
    walk's allocated unit, not an estimate, so the operator can rank them."""
    pad = _pad(_session(tmp_path, "shaped"))
    modules = pad / "node_modules"
    modules.mkdir()
    (modules / "left-pad.js").write_bytes(b"x" * 8192)
    build = pad / "build"
    build.mkdir()
    (build / "out.o").write_bytes(b"x" * 4096)
    target = pad / "target"
    target.mkdir()
    (target / "lib.so").write_bytes(b"x" * 2048)

    audit = audit_scratchpads(tmp_path / "sessions", cap=1)
    [row] = audit.pads

    assert row.over_cap is True
    names = {shape.path: shape for shape in row.shapes}
    assert set(names) == {"node_modules", "build", "target"}
    assert names["node_modules"].bytes == _subtree(modules)
    assert names["build"].bytes == _subtree(build)
    assert names["target"].bytes == _subtree(target)
    assert row.bytes == _subtree(pad)


def test_a_symlinked_node_modules_is_a_link_with_zero_bytes(tmp_path: Path) -> None:
    """The measured failure mode: following the link would report a shared
    store's tree as this pad's, which is exactly the number the row exists to
    refute. The link keeps its own allocation and its target."""
    outside = tmp_path / "shared-store"
    outside.mkdir()
    (outside / "huge.bin").write_bytes(b"x" * 65536)
    pad = _pad(_session(tmp_path, "linked"))
    link = pad / "node_modules"
    link.symlink_to(outside, target_is_directory=True)

    audit = audit_scratchpads(tmp_path / "sessions", cap=1 << 30)
    [row] = audit.pads
    [shape] = row.shapes

    assert shape.kind == "link"
    assert shape.target == str(outside)
    assert shape.bytes == _bytes(link)
    assert shape.bytes < 65536
    assert row.bytes == _bytes(link), "the target's tree is not the pad's bytes"


def test_a_git_file_is_a_worktree_pointer_naming_its_target(tmp_path: Path) -> None:
    """A ``.git`` FILE is a linked worktree: its objects live in the shared
    repository, and the target is what lets the operator recognise which one."""
    pad = _pad(_session(tmp_path, "worktree"))
    tree = pad / "checkout"
    tree.mkdir()
    (tree / ".git").write_text("gitdir: /shared/repo/.git/worktrees/checkout\n")

    audit = audit_scratchpads(tmp_path / "sessions", cap=1 << 30)
    [row] = audit.pads
    [marker] = row.git_markers

    assert marker.kind == "worktree"
    assert marker.path == "checkout/.git"
    assert marker.target == "/shared/repo/.git/worktrees/checkout"


def test_a_git_directory_is_an_independent_repository(tmp_path: Path) -> None:
    """The other ``.git`` shape, and the one a rescue bundle can be written
    for. Its internals are object storage, never reported as build shapes."""
    pad = _pad(_session(tmp_path, "clone"))
    repo = pad / "clone"
    (repo / ".git").mkdir(parents=True)
    (repo / ".git" / "HEAD").write_text("ref: refs/heads/main\n")
    (repo / "src").mkdir()
    (repo / "src" / "main.py").write_text("print(1)\n")

    audit = audit_scratchpads(tmp_path / "sessions", cap=1 << 30)
    [row] = audit.pads

    assert [(marker.path, marker.kind) for marker in row.git_markers] == [
        ("clone/.git", "repository")
    ]
    assert row.shapes == ()
    assert row.bytes == _subtree(pad)


def test_a_user_pad_is_reported_and_never_reclaimed(tmp_path: Path) -> None:
    """Parent and delegated pads are both walked — the user's pad has no
    sidebar row of its own here, and the audit is the only surface that can
    tell an operator the policy will never take its bytes."""
    user = _session(tmp_path, "mine", origin=None)
    pad = _pad(user)
    (pad / "build").mkdir()
    (pad / "build" / "bundle.js").write_bytes(b"x" * 4096)
    _pad(_session(tmp_path, "child"))

    audit = audit_scratchpads(tmp_path / "sessions", cap=1 << 30)
    rows = {row.session: row for row in audit.pads}

    assert rows["mine"].user_visible is True
    assert rows["mine"].origin == ""
    assert rows["child"].user_visible is False
    assert (pad / "build" / "bundle.js").exists(), "the audit never deletes"


def test_hardlinks_are_counted_once_per_pad(tmp_path: Path) -> None:
    """The same inode twice is one set of blocks on disk; a total that counted
    it twice would make a pad look over a cap it is not over."""
    pad = _pad(_session(tmp_path, "hardlinks"))
    first = pad / "a.bin"
    first.write_bytes(b"x" * 8192)
    second = pad / "b.bin"
    os.link(first, second)

    audit = audit_scratchpads(tmp_path / "sessions", cap=1 << 30)
    [row] = audit.pads

    assert row.bytes == _bytes(first)


def test_the_walk_is_bounded_and_marks_truncation(tmp_path: Path, monkeypatch) -> None:
    """A pad that is the problem is a tree; the audit reports what it saw and
    says so rather than walking a pathological tree to completion."""
    pad = _pad(_session(tmp_path, "wide"))
    for index in range(5):
        (pad / f"f{index}.dat").write_bytes(b"x" * 4096)
    monkeypatch.setattr(audit_module, "AUDIT_MAX_ENTRIES", 3)

    audit = audit_scratchpads(tmp_path / "sessions", cap=1 << 30)
    [row] = audit.pads

    assert row.truncated is True
    assert audit.truncated == 1
    assert row.bytes <= _subtree(pad)


def test_the_json_shape_carries_every_row_and_the_cap(tmp_path: Path) -> None:
    """``--json`` is the uncapped view; the text section only prints the top
    rows, so the payload must carry the rest with the shapes and pointers."""
    pad = _pad(_session(tmp_path, "json-row"))
    (pad / "build").mkdir()
    (pad / "build" / "x.o").write_bytes(b"x" * 1024)

    payload = audit_scratchpads(tmp_path / "sessions", cap=4096).as_dict()

    assert payload["count"] == 1
    assert payload["over_cap"] == 0
    [row] = payload["pads"]
    assert row["session"] == "json-row"
    assert row["origin"] == "subagent"
    assert row["shapes"][0]["path"] == "build"
    assert row["shapes"][0]["bytes"] > 0
