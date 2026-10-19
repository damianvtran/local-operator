"""Read-only audit of every session scratchpad under a store.

WHY A SEPARATE WALK
===================

The write policy (``local_operator/scratchpad.py``) and the retention content
layer (``session/delegated_retention.py``) both decide about ONE pad at a time
and both are bounded by what they may change. An operator looking at a store
that has grown asks a different question — WHERE the bytes are across every
session, parent and delegated alike, and which of those pads a policy will
never touch. ``lop sessions cleanup --dry-run`` answers it from this module.

Read-only by construction: nothing here opens a path for writing, removes
anything, or follows a symlink. Every number is ALLOCATED bytes
(``st_blocks * 512``), the unit the write cap's walk uses, so the audit and the
refusal cannot disagree about what "the pad holds" means. Hardlinks are counted
once per pad: the same inode twice is one set of blocks on disk, and a total
that counted it twice would make a pad look over a cap it is not over.

WHAT COUNTS AS A SHAPE
======================

The write policy refuses directories by a SEGMENT vocabulary (``node_modules``,
``build``, ``target`` and the wider
:data:`~local_operator.scratchpad.SCRATCHPAD_REFUSED_SEGMENTS`), so those are
the rows an operator is asked to act on. A shape is reported with the bytes it
actually holds, never an estimate. A SYMLINKED shape is reported as a link with
the link's own allocation (normally 0 B) and its target, and is never followed:
a pad that points ``node_modules`` at a shared store holds no copy, and a walk
that followed the link would report the store's bytes as the pad's.

``.git`` is the carve-out the content layer already documents, and the audit
distinguishes its two shapes because they can be reclaimed differently:

* a ``.git`` FILE is a worktree pointer; the report carries the target it
  names, which is how an operator recognises a linked worktree whose objects
  live in the shared repository;
* a ``.git`` DIR is an independent repository, reported as such and not walked
  for further shapes (its internals are object storage, not build output).

BOUNDS
======

A pad that is the problem is by definition a tree, so the walk is bounded by
DEPTH and ENTRY COUNT and reports ``truncated`` when either cap fires. A
truncated pad's ``bytes`` is a LOWER BOUND, exactly like the write path's
over-cap footprint: more precision would cost a full walk of the worst state to
say the same thing. Nothing is guessed after the cap — the entries visited are
the entries counted.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path

from local_operator.scratchpad import SCRATCHPAD_DIRNAME, scratchpad_budget_bytes

#: How deep below the pad a shape may sit. Eight is the content layer's own
#: bound (``CONTENT_MAX_DEPTH``), so the audit reports as shapes what the
#: reclaimer could actually see; a deeper tree is summarised by its shallowest
#: refused segment, which is the entry the reclaimer would take whole.
AUDIT_MAX_DEPTH = 8

#: How many directory entries one pad may cost the audit before it stops and
#: marks the pad truncated. The content pass has its own cap at 20,000 and
#: ABORTS on it (keeping the pad); the audit's job is to report rather than
#: judge, so its cap is higher — a measured 4.94 GB ``node_modules`` is well
#: over 20,000 entries, and stopping at the reclaimer's cap would hide the
#: shape that caused the problem. It stays finite so one pathological pad
#: cannot hold an interactive ``--dry-run`` hostage.
AUDIT_MAX_ENTRIES = 50_000


@dataclass(frozen=True)
class ShapeEntry:
    """One refused directory/link in a pad, with its real allocated size."""

    path: str
    kind: str  # "directory" or "link"
    bytes: int
    target: str = ""


@dataclass(frozen=True)
class GitMarker:
    """A ``.git`` file (worktree pointer) or directory (independent repo)."""

    path: str
    kind: str  # "worktree", "repository" or "link"
    target: str = ""


@dataclass
class PadAudit:
    """What one session's pad holds, and whether a policy may still take it."""

    session: str
    origin: str
    user_visible: bool
    bytes: int
    cap: int
    over_cap: bool
    shapes: tuple[ShapeEntry, ...] = ()
    git_markers: tuple[GitMarker, ...] = ()
    entries: int = 0
    truncated: bool = False

    def as_dict(self) -> dict[str, object]:
        return {
            "session": self.session,
            "origin": self.origin,
            "user_visible": self.user_visible,
            "bytes": self.bytes,
            "cap": self.cap,
            "over_cap": self.over_cap,
            "entries": self.entries,
            "truncated": self.truncated,
            "shapes": [
                {
                    "path": shape.path,
                    "kind": shape.kind,
                    "bytes": shape.bytes,
                    "target": shape.target,
                }
                for shape in self.shapes
            ],
            "git_markers": [
                {"path": marker.path, "kind": marker.kind, "target": marker.target}
                for marker in self.git_markers
            ],
        }


@dataclass
class ScratchpadAudit:
    """Every pad in one store, largest first."""

    cap: int
    pads: list[PadAudit] = field(default_factory=list)

    @property
    def total_bytes(self) -> int:
        return sum(pad.bytes for pad in self.pads)

    @property
    def over_cap(self) -> int:
        return sum(1 for pad in self.pads if pad.over_cap)

    @property
    def truncated(self) -> int:
        return sum(1 for pad in self.pads if pad.truncated)

    def as_dict(self) -> dict[str, object]:
        return {
            "cap": self.cap,
            "count": len(self.pads),
            "bytes": self.total_bytes,
            "over_cap": self.over_cap,
            "truncated": self.truncated,
            "pads": [pad.as_dict() for pad in self.pads],
        }


class _Walk:
    """One pad's accumulator: allocated bytes, shapes, pointers, bounds."""

    def __init__(self) -> None:
        self.total = 0
        self.entries = 0
        self.truncated = False
        self.shapes: list[ShapeEntry] = []
        self.git_markers: list[GitMarker] = []
        #: ``(st_dev, st_ino)`` of files this pad has already counted. Only
        #: consulted for links with ``st_nlink > 1``: hardlinks must count once,
        #: and the set stays empty on the ordinary pad.
        self._seen: set[tuple[int, int]] = set()

    def count(self, info: os.stat_result) -> int:
        """Allocated bytes for one entry, deduping hardlinks within this pad."""
        if info.st_nlink > 1:
            key = (info.st_dev, info.st_ino)
            if key in self._seen:
                return 0
            self._seen.add(key)
        return info.st_blocks * 512

    def charge(self, tree: str, depth: int = 0) -> bool:
        """Add ``tree``'s allocated bytes; False when a bound was reached.

        Used for the subtrees the walk does NOT otherwise descend into (a
        refused shape, a ``.git`` directory). Symlinks are never followed. The
        entry count is shared with the main walk, so a shape's own size
        consumes the same budget as any other entry.
        """
        stack: list[tuple[str, int]] = [(tree, 0)]
        while stack:
            current, depth = stack.pop()
            try:
                with os.scandir(current) as scan:
                    children = sorted(scan, key=lambda entry: entry.name)
            except OSError:
                continue
            for entry in children:
                self.entries += 1
                if self.entries > AUDIT_MAX_ENTRIES:
                    return False
                try:
                    if entry.is_symlink():
                        self.total += self.count(entry.stat(follow_symlinks=False))
                        continue
                    if entry.is_dir(follow_symlinks=False):
                        if depth >= AUDIT_MAX_DEPTH:
                            self.truncated = True
                            continue
                        stack.append((entry.path, depth + 1))
                        continue
                    self.total += self.count(entry.stat(follow_symlinks=False))
                except OSError:
                    continue
        return True


def _read_pointer(path: str) -> str:
    """The target a ``.git`` FILE names, or ``""`` when it is unreadable.

    The gitdir line is ``gitdir: <path>``; the raw text is returned when the
    file does not carry that prefix, because a malformed pointer is still
    evidence and reporting nothing would hide it.
    """
    try:
        with open(path, "r", encoding="utf-8", errors="replace") as handle:
            text = handle.read(4096).strip()
    except OSError:
        return ""
    prefix = "gitdir:"
    return text[len(prefix) :].strip() if text.startswith(prefix) else text


def _link_target(path: str) -> str:
    try:
        return os.readlink(path)
    except OSError:
        return ""


def _audit_pad(pad: Path, session: str, origin: str, user_visible: bool, cap: int) -> PadAudit:
    """Walk one pad (bounded, symlinks never followed) into a :class:`PadAudit`."""
    # LAZY: the write path's vocabulary, one copy — the same import the content
    # layer makes for the same reason.
    from local_operator.scratchpad import _is_refused_segment

    walk = _Walk()
    stack: list[tuple[str, int]] = [(os.fspath(pad), 0)]
    while stack:
        current, depth = stack.pop()
        try:
            with os.scandir(current) as scan:
                children = sorted(scan, key=lambda entry: entry.name)
        except OSError:
            continue
        for entry in children:
            walk.entries += 1
            if walk.entries > AUDIT_MAX_ENTRIES:
                walk.truncated = True
                stack.clear()
                break
            name = entry.name
            relative = os.path.relpath(entry.path, pad)
            try:
                if entry.is_symlink():
                    # Never followed. The link's own allocation is the only
                    # number it may contribute: a pad pointing node_modules at
                    # a shared store holds no copy, and measuring the target
                    # would report the store's bytes as this pad's.
                    size = walk.count(entry.stat(follow_symlinks=False))
                    walk.total += size
                    target = _link_target(entry.path)
                    if name == ".git":
                        walk.git_markers.append(GitMarker(relative, "link", target))
                    elif _is_refused_segment(name):
                        walk.shapes.append(ShapeEntry(relative, "link", size, target))
                    continue
                if entry.is_dir(follow_symlinks=False):
                    if name == ".git":
                        if not walk.charge(entry.path, depth + 1):
                            walk.truncated = True
                            stack.clear()
                            break
                        walk.git_markers.append(GitMarker(relative, "repository"))
                        continue
                    if _is_refused_segment(name):
                        before = walk.total
                        if not walk.charge(entry.path, depth + 1):
                            walk.truncated = True
                        walk.shapes.append(ShapeEntry(relative, "directory", walk.total - before))
                        continue
                    if depth >= AUDIT_MAX_DEPTH:
                        walk.truncated = True
                        continue
                    stack.append((entry.path, depth + 1))
                    continue
                info = entry.stat(follow_symlinks=False)
                walk.total += walk.count(info)
                if name == ".git":
                    walk.git_markers.append(
                        GitMarker(relative, "worktree", _read_pointer(entry.path))
                    )
            except OSError:
                # An entry that vanished or cannot be inspected contributes
                # nothing; the audit reports what it could see, never a guess.
                continue
    shapes = tuple(sorted(walk.shapes, key=lambda shape: (-shape.bytes, shape.path)))
    return PadAudit(
        session=session,
        origin=origin,
        user_visible=user_visible,
        bytes=walk.total,
        cap=cap,
        over_cap=walk.total > cap,
        shapes=shapes,
        git_markers=tuple(walk.git_markers),
        entries=walk.entries,
        truncated=walk.truncated,
    )


def audit_scratchpads(sessions_dir: Path, *, cap: int | None = None) -> ScratchpadAudit:
    """Audit every session pad under ``sessions_dir``; nothing is modified.

    ``cap`` defaults to the configured pad cap (:func:`scratchpad_budget_bytes`)
    and is copied onto every row, so a report stays readable after the
    operator changes the setting. Parent and delegated sessions are both
    walked: the delegated class is the one a policy may reclaim, and the
    parent class is the one it never will — the report is how an operator sees
    the difference.
    """
    from local_operator.resume import is_user_session_origin, session_origin

    limit = scratchpad_budget_bytes() if cap is None else cap
    audit = ScratchpadAudit(cap=limit)
    try:
        with os.scandir(sessions_dir) as scan:
            entries = sorted(scan, key=lambda entry: entry.name)
    except OSError:
        return audit
    for entry in entries:
        if entry.name.startswith("."):
            continue
        try:
            if not entry.is_dir(follow_symlinks=False):
                continue
        except OSError:
            continue
        pad = Path(entry.path) / SCRATCHPAD_DIRNAME
        try:
            if pad.is_symlink() or not pad.is_dir():
                continue
        except OSError:
            continue
        origin = session_origin(Path(entry.path))
        audit.pads.append(
            _audit_pad(pad, entry.name, origin, is_user_session_origin(origin), limit)
        )
    audit.pads.sort(key=lambda item: (-item.bytes, item.session))
    return audit


__all__ = [
    "AUDIT_MAX_DEPTH",
    "AUDIT_MAX_ENTRIES",
    "GitMarker",
    "PadAudit",
    "ScratchpadAudit",
    "ShapeEntry",
    "audit_scratchpads",
]
