"""No code outside ``session/compression.py`` may probe the PLAIN transcript name.

WHY THIS TEST EXISTS. Compression moved a session's one bulk file to
``transcript.jsonl.zst`` / ``.gz`` and left every sidecar where it was. A read
path that still asks ``(directory / TRANSCRIPT_NAME).is_file()`` therefore does
not fail loudly — it answers ``False``, and ``False`` reads as "this session has
no transcript": an empty conversation, a dropped wake, a session reaped as a
fresh failed create, a replica the mesh does not believe this device holds.
Stage C of the compression work closed the ~20 such probes that existed at
3da141a43; without a guard the next one lands silently, because every one of
them looks like ordinary, careful code.

WHAT IS FLAGGED. An expression that builds the plain name — ``<path> / NAME``
where ``NAME`` is a module constant bound to ``"transcript.jsonl"``
(``TRANSCRIPT_NAME``, ``_TRANSCRIPT_NAME``…, resolved per module) or the string
literal itself — used in the same function with ``.is_file()``, ``.exists()``,
``.stat()``, ``.read_bytes()``, ``.read_text()`` or ``.open(``, directly or
through a local ``x = <path> / NAME`` binding. Those are the existence and
byte probes; the repair is ``compression.has_transcript`` for existence and
``compression.transcript_for_read`` / ``read_head`` / ``read_tail`` /
``iter_lines`` for bytes. A literal in a USER-FACING MESSAGE is not a probe and
is not flagged (no method call on it).

WHAT IS NOT. Writes and creates (``open("wb")``, ``mkdir``), a path handed to
another module that resolves it itself, and ``compression.py`` itself, which
DEFINES the name. Those keep the plain name on purpose; every other surviving
site is in :data:`_ALLOWED` with the one line that says why a compressed
session is not what it holds. The list is keyed ``path::function`` and the
reason is checked by a human, which is the point — a silent surface is how the
twenty probes stage C fixed came to exist.
"""

from __future__ import annotations

import ast
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[3]
PACKAGE = ROOT / "local_operator"

#: The module that DEFINES the plain name and implements the resolver.
COMPRESSION_MODULE = "local_operator/session/compression.py"

#: Method calls that turn a path into an answer about the transcript's existence
#: or bytes. ``mkdir``/``unlink``/``write_text`` are deliberately absent: a
#: creation path legitimately holds the plain name, and removal is the subject
#: of ``test_no_session_deletion``.
_PROBE_METHODS = frozenset({"is_file", "exists", "stat", "read_bytes", "read_text", "open"})

#: Names that are the plain transcript filename even when a module spells its
#: own alias instead of importing ``resume.TRANSCRIPT_NAME``.
_FIXED_NAMES = frozenset({"TRANSCRIPT_NAME", "TRANSCRIPT_FILENAME"})

_PLAIN_LITERAL = "transcript.jsonl"

#: ``path::function`` → why this probe is legitimate. One line each, written so a
#: reviewer can open the site and agree without re-deriving the design.
#:
#: Every row below is a site where the PLAIN name is the right answer, not a
#: missed migration. The reasons say which fact makes it right — an archive
#: early-return above the stat, a directory the compression pass never scans, or
#: a creation write. A site that holds the plain name "for now" does not belong
#: here; it belongs in the resolver.
_ALLOWED: dict[str, str] = {
    # CREATION, not a probe: the fork writes its OWN new journal. The source it
    # copies is read through the codec resolver (fork.py's `codec` branches).
    "local_operator/fork.py::fork_session": "creates the fork's new journal with open('wb')",
    # Reads a LIVE peer's journal. Its runtime holds the session's writer lock,
    # so the file is plain; and a miss here answers "unknown", never "failed".
    "local_operator/mobile/peer_send.py::probe_transcript_for": (
        "probes a live peer's plain journal; a miss answers 'unknown'"
    ),
    # Replicas live at <config>/network/replicas/<id> — outside sessions/, which
    # is the only tree the compression pass walks — and sync writes them plain.
    "local_operator/network/sync.py::_verify_replica_against_cursor": (
        "replica dirs are outside sessions/ and are always written plain"
    ),
    # The four below are archive-guarded ABOVE the stat: `_is_archived` skips an
    # archived entry outright, and `build_anchor` answers None for one, so none of
    # these stats can see an archive.
    "local_operator/session/index_prewarm.py::sessions_needing_warm": (
        "_is_archived(entry) continues before this stat"
    ),
    "local_operator/session/index_prewarm.py::sessions_needing_replay": (
        "_is_archived(entry) continues before this stat"
    ),
    "local_operator/session/index_prewarm.py::_anchor_current": (
        "returns before the stat when the session is archived"
    ),
    "local_operator/session/index_prewarm.py::write_tail_anchor": (
        "build_anchor answers None for an archive, so this settle-stat is plain-only"
    ),
    # This session's own live Transcript: its construction restored the plain
    # journal (stage B). The path is handed to a hook, not resolved here.
    "local_operator/session/session.py::_run_post_tool_hooks": (
        "this session's live Transcript restored the plain journal at construction"
    ),
    # Both probes sit BELOW read_indexed_raw_rows' archive early-return: an
    # archive is answered None (unavailable) before the file is opened.
    "local_operator/session/transcript.py::read_indexed_raw_rows": (
        "below the archive early-return; an archive is answered None first"
    ),
    "local_operator/session/transcript.py::project_indexed_source_subset": (
        "unreachable for an archive: its read_indexed_raw_rows already answered None"
    ),
}


class _Flag:
    __slots__ = ("path", "line", "function")

    def __init__(self, path: str, line: int, function: str) -> None:
        self.path = path
        self.line = line
        self.function = function

    @property
    def key(self) -> str:
        return f"{self.path}::{self.function}"

    def __str__(self) -> str:
        return f"{self.path}:{self.line} in {self.function}"


def _literal_names(tree: ast.Module) -> set[str]:
    """Module-level names bound to the plain filename, plus the fixed spellings."""
    names = set(_FIXED_NAMES)
    for node in tree.body:
        targets: list[ast.expr] = []
        value: ast.expr | None = None
        if isinstance(node, ast.Assign):
            targets, value = list(node.targets), node.value
        elif isinstance(node, ast.AnnAssign):
            targets, value = [node.target], node.value
        if value is None:
            continue
        if isinstance(value, ast.Constant) and value.value == _PLAIN_LITERAL:
            for target in targets:
                if isinstance(target, ast.Name):
                    names.add(target.id)
    return names


def _is_plain_path_expr(node: ast.expr, names: set[str]) -> bool:
    """Whether ``node`` spells ``<expr> / <the plain filename>``."""
    if not isinstance(node, ast.BinOp) or not isinstance(node.op, ast.Div):
        return False
    right = node.right
    if isinstance(right, ast.Name) and right.id in names:
        return True
    return isinstance(right, ast.Constant) and right.value == _PLAIN_LITERAL


def _probe_target(node: ast.expr, names: set[str], bindings: set[str]) -> bool:
    """Whether ``node`` is a path expression the probe methods could be called on."""
    return _is_plain_path_expr(node, names) or (isinstance(node, ast.Name) and node.id in bindings)


def _scan_scope(
    body: list[ast.stmt], names: set[str], path: str, function: str, out: list[_Flag]
) -> None:
    """Flag probes in one function (or module) body.

    ``bindings`` is the set of local names assigned a plain path expression in
    THIS scope — ``transcript = directory / TRANSCRIPT_NAME`` followed by
    ``transcript.is_file()`` is one site, not zero. A nested function is walked
    by the caller as its own scope (its own ``function``), which is why the walk
    here does not recurse into it.
    """
    bindings: set[str] = set()
    for node in body:
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
            continue
        for child in ast.walk(node):
            if isinstance(child, ast.Assign) and _is_plain_path_expr(child.value, names):
                for target in child.targets:
                    if isinstance(target, ast.Name):
                        bindings.add(target.id)
            elif isinstance(child, ast.AnnAssign) and child.value is not None:
                if _is_plain_path_expr(child.value, names) and isinstance(child.target, ast.Name):
                    bindings.add(child.target.id)
        # A method call on a bound name or on the path expression itself.
        for child in ast.walk(node):
            if not isinstance(child, ast.Call):
                continue
            func = child.func
            if not isinstance(func, ast.Attribute) or func.attr not in _PROBE_METHODS:
                continue
            if _probe_target(func.value, names, bindings):
                out.append(_Flag(path, child.lineno, function))


def scan_source(source: str, path: str) -> list[_Flag]:
    tree = ast.parse(source)
    names = _literal_names(tree)
    out: list[_Flag] = []
    _scan_scope(tree.body, names, path, "<module>", out)
    for node in ast.walk(tree):
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            _scan_scope(node.body, names, path, node.name, out)
    return out


def scan_tree(root: Path) -> list[_Flag]:
    out: list[_Flag] = []
    for file in sorted(root.rglob("*.py")):
        relative = file.relative_to(ROOT).as_posix()
        if relative == COMPRESSION_MODULE:
            continue
        out.extend(scan_source(file.read_text(encoding="utf-8"), relative))
    return out


def _unexplained(flags: list[_Flag]) -> list[_Flag]:
    return [flag for flag in flags if flag.key not in _ALLOWED]


def test_no_plain_name_probe_outside_the_resolver() -> None:
    """The live tree: every plain-name existence/byte probe is accounted for."""
    unexplained = _unexplained(scan_tree(PACKAGE))
    assert not unexplained, "plain transcript probes outside the resolver:\n" + "\n".join(
        f"  {flag}" for flag in sorted(unexplained, key=lambda item: (item.path, item.line))
    )


def test_the_allow_list_names_only_live_sites() -> None:
    """A stale row is as bad as a missing one: it hides the next probe there."""
    live = {flag.key for flag in scan_tree(PACKAGE)}
    stale = sorted(key for key in _ALLOWED if key not in live)
    assert not stale, "allow-list rows for sites that are no longer probes:\n" + "\n".join(
        f"  {key}: {_ALLOWED[key]}" for key in stale
    )


# ---------------------------------------------------------------------------
# Self-test: the guard must fail the build on a NEW probe, and pass a resolver
# ---------------------------------------------------------------------------

_FLAGGED_FIXTURE = '''
from pathlib import Path

TRANSCRIPT_NAME = "transcript.jsonl"


def wants_to_know(directory: Path) -> bool:
    return (directory / TRANSCRIPT_NAME).is_file()
'''

_BOUND_FIXTURE = '''
from pathlib import Path

TRANSCRIPT_NAME = "transcript.jsonl"


def also_wants_to_know(directory: Path) -> bytes:
    transcript = directory / TRANSCRIPT_NAME
    return transcript.read_bytes()
'''

_PASSING_FIXTURE = '''
from pathlib import Path

from local_operator.session import compression


def asks_the_resolver(directory: Path) -> bool:
    return compression.has_transcript(directory)


def reads_through_the_resolver(directory: Path) -> bytes | None:
    path = compression.transcript_for_read(directory)
    return path.read_bytes() if path is not None else None


def writes_a_plain_file(directory: Path) -> None:
    # A WRITE: ``write_bytes``/``mkdir`` are not probes, so a creation path is
    # not flagged. ``.open("wb")`` IS flagged by design (the shape is
    # unambiguous only by its arguments, and a guard that reads modes is the
    # brittleness this one exists to avoid) — such a site goes on the allow-list
    # with its reason, exactly as fork.py's does.
    (directory / "transcript.jsonl").write_bytes(b"")
'''


def test_the_guard_flags_a_direct_probe() -> None:
    flags = scan_source(_FLAGGED_FIXTURE, "fixture/direct.py")
    assert [flag.function for flag in flags] == ["wants_to_know"], flags


def test_the_guard_flags_a_probe_through_a_local_binding() -> None:
    flags = scan_source(_BOUND_FIXTURE, "fixture/bound.py")
    assert [flag.function for flag in flags] == ["also_wants_to_know"], flags


def test_the_guard_passes_the_resolver_and_a_plain_write() -> None:
    assert scan_source(_PASSING_FIXTURE, "fixture/passing.py") == []


def test_a_creation_open_is_flagged_and_excused_by_the_allow_list() -> None:
    """``open("wb")`` is a shape, not a question: it is flagged, then argued for."""
    flags = scan_source(_CREATION_FIXTURE, "fixture/creation.py")
    assert [flag.function for flag in flags] == ["creates_the_plain_file"]
    saved = dict(_ALLOWED)
    _ALLOWED["fixture/creation.py::creates_the_plain_file"] = "self-test: creation write"
    try:
        assert _unexplained(flags) == []
    finally:
        _ALLOWED.clear()
        _ALLOWED.update(saved)


_CREATION_FIXTURE = '''
from pathlib import Path

TRANSCRIPT_NAME = "transcript.jsonl"


def creates_the_plain_file(directory: Path) -> None:
    (directory / TRANSCRIPT_NAME).open("wb").close()
'''


def test_the_allow_list_excuses_a_named_site_and_nothing_else() -> None:
    """The allow-list is keyed path::function, so an excused function is not a surface."""
    key = "fixture/direct.py::wants_to_know"
    saved = dict(_ALLOWED)
    _ALLOWED[key] = "self-test: this row is the excuse under test"
    try:
        assert _unexplained(scan_source(_FLAGGED_FIXTURE, "fixture/direct.py")) == []
        # The SAME shape in another function is still flagged.
        assert _unexplained(scan_source(_BOUND_FIXTURE, "fixture/bound.py"))
    finally:
        _ALLOWED.clear()
        _ALLOWED.update(saved)
