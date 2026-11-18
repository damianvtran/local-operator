"""The git engine that carries a moved session's worktrees: snapshot, scan, bundle, restore.

WHY THIS EXISTS. ``lop sessions move`` used to copy the session directory and
nothing else, so a session moved mid-task arrived on the other device without
its worktrees, unpushed commits or dirty files (operator report, 2026-11-18;
``docs/design/move-working-set.md``). The manifest half (which worktrees, what
is withheld, the user's confirmation) lives in ``workset.py``; THIS module is
the mesh-independent git half the move protocol calls once the user has
confirmed. Every function here is pure in the sense that it takes paths and
SHAs and touches only git repositories and the directories it is handed: no
relay, no frames, no audit. The wiring (W1-W5 phases, audit-before-bytes,
recall routing) lives in ``mobility.py`` and calls these.

WHY A SEPARATE MODULE (not ``workset.py`` as design §13 lists). Two coders
worked the halves concurrently; this file avoids a merge collision. The local
``Excluded``/``Recipe`` shapes below mirror the §3.4 manifest items and are meant
to be unified with ``workset.py``'s types once both land.

THE INVARIANTS, each with the check that holds it:

* **The source is never modified.** ``snapshot`` builds both trees in a
  temporary COPY of the index (the real index file is never written — even
  ``git write-tree`` on the real index would rewrite its cache-tree extension),
  writes only new objects and private refs ``refs/lop/move/<sid>/<n>/*``, and
  ``mark_departed`` only adds a journal and a marker. Nothing here deletes a
  user's file (design §8.6; ``test_workset_never_deletes.py``).
* **Only scanned objects leave.** ``stage_bundle`` scans EVERY object the bundle
  will carry (``git rev-list --objects <tips> --not <have>``, which includes the
  snapshot index and worktree trees, i.e. staged bytes too), refuses on any
  finding, and binds the scan to the artifact: the bundle's header refs must be
  exactly the scanned tips and the scan's object digest is recorded beside the
  artifact. v1 has NO opt-in to carry flagged history (manager decision,
  2026-11-18): a flagged worktree is left behind with the reason.
  Shape scanning is BEST EFFORT, not a guarantee: it recognises the harness's
  credential shapes and sensitive path names. Anything it cannot read or
  classify (binary, over the scan limit, unreadable) is withheld for that
  reason, so "not flagged" means "scanned and matched no known shape".
* **The destination never overwrites.** ``restore`` materialises a new worktree
  in a fresh staging directory and publishes it with a no-replace rename into a
  path that does not exist; every path it reads or writes is resolved under the
  destination home with no symlinked, aliased or traversing component
  (``contained_path``). The single in-place exception is a recall onto a
  worktree this device's own trusted journal (``mark_departed``) records, and
  whose state is re-measured unchanged since departure.

All git runs with routing variables stripped (an inherited ``GIT_DIR`` or
``GIT_INDEX_FILE`` must never redirect a write), a fixed ``lop-move`` identity,
``GIT_TERMINAL_PROMPT=0`` and — on every path that reads a bundle or writes a
restore — ``-c credential.helper=`` (design §5). Only the destination's own
origin fetch/clone (``compute_have``/``clone_primary``) keeps credential helpers,
because it authenticates as the destination user against their own remote.

Stdlib only; the denylist and secret guard are imported lazily inside the scan,
because this module is imported by the relay (same import-weight reason as
``carry.py``).
"""

from __future__ import annotations

import ctypes
import ctypes.util
import errno
import hashlib
import json
import os
import re
import shutil
import stat
import subprocess
import sys
import tempfile
import threading
import time
from collections.abc import Callable, Iterator, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Final, TypedDict

#: Private ref namespace for move snapshots. ``refs/lop/wip`` (subagent WIP
#: checkpoints) is the precedent; refs here are shared across linked worktrees
#: (only ``refs/worktree``/``refs/bisect``/``refs/rewritten`` are per-worktree),
#: so ``<n>`` distinguishes the worktrees of one repository in one move.
REF_ROOT: Final = "refs/lop/move"
SNAPSHOT_KINDS: Final = ("head", "index", "worktree")

#: Largest blob the content scan reads (design §8.1: "text files <= 1 MiB").
#: Larger outgoing blobs are withheld as unscannable rather than skipped.
SCAN_MAX_BLOB_BYTES: Final = 1 << 20

#: W1's bounded origin fetch (design §4 W1: "bounded, 120 s").
FETCH_TIMEOUT_S: Final = 120.0

#: W1 clone free-space rule (design §4 W1): ``pack_bytes x 2 + 1 GiB``.
CLONE_HEADROOM_BYTES: Final = 1 << 30

_ROUTING_VARS: Final = frozenset(
    {
        "GIT_DIR",
        "GIT_WORK_TREE",
        "GIT_INDEX_FILE",
        "GIT_COMMON_DIR",
        "GIT_OBJECT_DIRECTORY",
        "GIT_ALTERNATE_OBJECT_DIRECTORIES",
        "GIT_NAMESPACE",
        "GIT_PREFIX",
    }
)
_IDENTITY: Final = ("lop-move", "lop-move@localhost")
#: Absolute, never-existing root the history path scan anchors names under.
_HISTORY_ANCHOR: Final = "/__lop_move_history__"
_SESSION_ID_RE: Final = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_-]{0,127}$")
_DEVICE_SAFE_RE: Final = re.compile(r"[^A-Za-z0-9._-]+")
_HEX_RE: Final = re.compile(r"^[0-9a-f]{40}(?:[0-9a-f]{24})?$")


class MoveWorksetError(Exception):
    """A step refused with a machine reason code (rendered by the surface, §12)."""

    def __init__(self, reason: str, detail: str = "") -> None:
        super().__init__(f"{reason}: {detail}" if detail else reason)
        self.reason = reason
        self.detail = detail


class ContainmentError(MoveWorksetError):
    """A destination path escapes the home, aliases, or traverses a symlink."""

    def __init__(self, detail: str) -> None:
        super().__init__("unsafe_path", detail)


class Excluded(TypedDict):
    """One §3.4 ``excluded`` item (to unify with ``workset.py``).

    ``path`` is repository-relative POSIX here; the manifest renders it
    home-relative. ``rule`` is a denylist rule id, credential CLASS or
    unscannable cause — never a value. ``object`` is the blob/tree id the
    finding is about, so a finding is addressable by digest.
    """

    path: str
    why: str
    rule: str
    object: str


class Recipe(TypedDict):
    """One §3.4 ``recipe``: printed, never run by this module (§9)."""

    kind: str
    run: str
    cwd: str


class HistoryWithheld(MoveWorksetError):
    """``stage_bundle`` refused: the outgoing objects carry findings."""

    def __init__(self, check: HistoryCheck) -> None:
        super().__init__("history_secret", f"{len(check.findings)} finding(s)")
        self.check = check


# --------------------------------------------------------------------------- git


def _environment(extra: dict[str, str] | None = None) -> dict[str, str]:
    env = {k: v for k, v in os.environ.items() if k not in _ROUTING_VARS}
    env["GIT_TERMINAL_PROMPT"] = "0"
    for role in ("AUTHOR", "COMMITTER"):
        env[f"GIT_{role}_NAME"], env[f"GIT_{role}_EMAIL"] = _IDENTITY
    if extra:
        env.update(extra)
    return env


def _git_argv(cwd: Path | str, args: Sequence[str], *, credentials: bool) -> list[str]:
    argv = ["git", "-c", "core.quotepath=false"]
    if not credentials:
        # An empty value resets the helper list: no keychain/osxkeychain prompt
        # can appear on the operator's screen from a bundle or restore step.
        argv += ["-c", "credential.helper="]
    return [*argv, "-C", str(cwd), *args]


def _git(
    cwd: Path | str,
    *args: str,
    env: dict[str, str] | None = None,
    check: bool = True,
    timeout: float | None = None,
    credentials: bool = False,
    input_bytes: bytes | None = None,
) -> subprocess.CompletedProcess[str]:
    # Local git steps carry no timeout: prompts are disabled and no network is
    # touched, and the move phase that calls them owns the wall-clock bound.
    # Only the two network steps (origin fetch/clone) pass one.
    proc = subprocess.run(
        _git_argv(cwd, args, credentials=credentials),
        env=env if env is not None else _environment(),
        capture_output=True,
        input=input_bytes.decode() if input_bytes is not None else None,
        text=True,
        timeout=timeout,
        check=False,
    )
    if check and proc.returncode != 0:
        raise MoveWorksetError(
            "git_failed", f"git {' '.join(args[:3])}: {proc.stderr.strip()[:400]}"
        )
    return proc


def _out(cwd: Path | str, *args: str, **kw: Any) -> str:
    return _git(cwd, *args, **kw).stdout.strip()


def _is_commit(repo: Path, sha: str) -> bool:
    return _git(repo, "cat-file", "-e", f"{sha}^{{commit}}", check=False).returncode == 0


def _toplevel(path: Path) -> Path:
    if _out(path, "rev-parse", "--is-bare-repository") == "true":
        raise MoveWorksetError("bare_repository", str(path))
    return Path(_out(path, "rev-parse", "--show-toplevel"))


def _validate_session_id(session_id: str) -> None:
    if not _SESSION_ID_RE.match(session_id):
        raise MoveWorksetError("invalid_session_id", repr(session_id))


def _validate_repo_relative(path: str) -> str:
    if not path or path.startswith("/") or "\x00" in path or "\\" in path:
        raise MoveWorksetError("invalid_exclude", repr(path))
    parts = path.split("/")
    if any(p in ("", ".", "..") for p in parts):
        raise MoveWorksetError("invalid_exclude", repr(path))
    return path


def snapshot_ref(session_id: str, n: int, kind: str) -> str:
    """``refs/lop/move/<sid>/<n>/<kind>`` — the one spelling of a snapshot ref."""
    _validate_session_id(session_id)
    if kind not in SNAPSHOT_KINDS or n < 0:
        raise MoveWorksetError("invalid_snapshot_ref", f"{n}/{kind}")
    return f"{REF_ROOT}/{session_id}/{n}/{kind}"


# ---------------------------------------------------------------------- snapshot


@dataclass(frozen=True)
class SnapshotRefs:
    """The three private refs of one worktree's snapshot, and what they hold.

    ``index`` is a commit whose tree is the staged state (parent ``head``);
    ``worktree`` is a commit whose tree is the files on disk including
    untracked, minus ignored and excluded (parent ``index``). ``excludes`` are
    the repo-relative paths kept out of both trees.
    """

    worktree_path: str
    session_id: str
    n: int
    head: str
    branch: str | None
    index: str
    worktree: str
    index_tree: str
    worktree_tree: str
    excludes: tuple[str, ...] = ()

    @property
    def refs(self) -> tuple[str, ...]:
        return tuple(snapshot_ref(self.session_id, self.n, k) for k in SNAPSHOT_KINDS)


def _trees(root: Path, head: str, excludes: Sequence[str], scratch_dir: Path) -> tuple[str, str]:
    """(index tree, worktree tree) of ``root``, built in a temporary index copy.

    Excluded paths are reverted to their HEAD entry (or dropped when HEAD lacks
    them) in the INDEX tree as well as kept out of the worktree tree. Deviation
    from design §5.1, which keeps a tracked secret's real-index entry: a secret
    the user merely `git add`-ed would otherwise ride in the index commit. The
    receipt therefore says local edits AND staged edits to that path stay on the
    source; the history scan would withhold the whole worktree otherwise.
    """
    real_index = Path(_out(root, "rev-parse", "--path-format=absolute", "--git-path", "index"))
    fd, tmp_name = tempfile.mkstemp(prefix="lop-move-index-", dir=scratch_dir)
    os.close(fd)
    tmp = Path(tmp_name)
    try:
        env = _environment({"GIT_INDEX_FILE": str(tmp)})
        if real_index.is_file():
            # copy2, not copyfile: git's racy-clean check compares each entry's
            # mtime with the INDEX FILE's mtime. A copy stamped "now" makes a
            # file edited in the same second as the last index write (same
            # size) look clean, and its edit would silently miss the snapshot.
            shutil.copy2(real_index, tmp)
        else:
            tmp.unlink()
            _git(root, "read-tree", head, env=env)
        literal = [f":(literal){p}" for p in excludes]
        for start in range(0, len(literal), 500):
            # check=True: a silently failed revert would let a staged secret
            # ride in the index tree. (A pathspec matching nothing exits 0.)
            _git(root, "reset", "-q", head, "--", *literal[start : start + 500], env=env)
        proc = _git(root, "write-tree", env=env, check=False)
        if proc.returncode != 0:
            # The only way write-tree fails on a well-formed index copy is an
            # unmerged entry: the manifest's "unmerged" left_behind reason.
            raise MoveWorksetError("unmerged", proc.stderr.strip()[:200])
        index_tree = proc.stdout.strip()
        spec = ["."] + [f":(exclude,literal){p}" for p in excludes]
        _git(root, "add", "-A", "--", *spec, env=env)
        worktree_tree = _out(root, "write-tree", env=env)
        return index_tree, worktree_tree
    finally:
        tmp.unlink(missing_ok=True)


def snapshot(worktree: Path, session_id: str, n: int, excludes: Sequence[str] = ()) -> SnapshotRefs:
    """Snapshot ``worktree`` into ``refs/lop/move/<sid>/<n>/{head,index,worktree}`` (§5.1).

    Run after the session is retired (quiescent). Never touches HEAD, the real
    index or a working file. Idempotent: commit dates are pinned to HEAD's
    committer date, so the same tree on the same HEAD yields the same commits
    and a retry rewrites the refs to identical values.

    Raises ``MoveWorksetError`` with ``unborn_head``, ``unmerged`` or
    ``bare_repository``; the caller marks the worktree ``left_behind``.
    """
    _validate_session_id(session_id)
    excludes = tuple(sorted({_validate_repo_relative(p) for p in excludes}))
    root = _toplevel(Path(worktree))
    head_proc = _git(root, "rev-parse", "--verify", "-q", "HEAD^{commit}", check=False)
    if head_proc.returncode != 0:
        raise MoveWorksetError("unborn_head", str(root))
    head = head_proc.stdout.strip()
    branch_proc = _git(root, "symbolic-ref", "-q", "--short", "HEAD", check=False)
    branch = branch_proc.stdout.strip() or None
    git_dir = Path(_out(root, "rev-parse", "--absolute-git-dir"))
    index_tree, worktree_tree = _trees(root, head, excludes, git_dir)
    stamp = _out(root, "log", "-1", "--format=%ct", head)
    env = _environment(
        {"GIT_AUTHOR_DATE": f"{stamp} +0000", "GIT_COMMITTER_DATE": f"{stamp} +0000"}
    )
    index_c = _out(
        root,
        "commit-tree",
        index_tree,
        "-p",
        head,
        "-m",
        f"lop move: index of {session_id}",
        env=env,
    )
    worktree_c = _out(
        root,
        "commit-tree",
        worktree_tree,
        "-p",
        index_c,
        "-m",
        f"lop move: worktree of {session_id}",
        env=env,
    )
    for kind, sha in zip(SNAPSHOT_KINDS, (head, index_c, worktree_c), strict=True):
        _git(root, "update-ref", snapshot_ref(session_id, n, kind), sha)
    return SnapshotRefs(
        worktree_path=str(root),
        session_id=session_id,
        n=n,
        head=head,
        branch=branch,
        index=index_c,
        worktree=worktree_c,
        index_tree=index_tree,
        worktree_tree=worktree_tree,
        excludes=excludes,
    )


# ------------------------------------------------------------------ history scan


@dataclass(frozen=True)
class HistoryCheck:
    """What the outgoing-object scan saw, bound to exactly that object set.

    ``scanned_digest`` = ``sha256:`` of the sorted outgoing object ids; the
    artifact sidecar carries it so the served bundle is provably the scanned
    set. ``findings`` empty means every outgoing blob was read and matched no
    known secret shape — best effort, not a guarantee.
    """

    tips: tuple[str, ...]
    have: tuple[str, ...]
    objects: int
    blobs: int
    scanned_digest: str
    findings: tuple[Excluded, ...]


def _batch(
    repo: Path, ids: Sequence[str], *, check_only: bool
) -> Iterator[tuple[str, str, int, bytes | None]]:
    """Stream ``git cat-file --batch[-check]`` for ``ids`` without buffering all of it.

    A writer thread feeds stdin so a large id list cannot deadlock against a
    full stdout pipe.
    """
    mode = "--batch-check" if check_only else "--batch"
    proc = subprocess.Popen(
        _git_argv(repo, ["cat-file", mode], credentials=False),
        stdin=subprocess.PIPE,
        stdout=subprocess.PIPE,
        stderr=subprocess.DEVNULL,
        env=_environment(),
    )
    assert proc.stdin is not None and proc.stdout is not None

    stdin = proc.stdin

    def feed() -> None:
        try:
            for oid in ids:
                stdin.write(f"{oid}\n".encode())
            stdin.close()
        except BrokenPipeError:
            pass

    writer = threading.Thread(target=feed, daemon=True)
    writer.start()
    try:
        for _ in ids:
            header = proc.stdout.readline().decode().split()
            if len(header) < 2 or header[1] == "missing":
                yield (header[0] if header else "", "missing", 0, None)
                continue
            oid, kind, size = header[0], header[1], int(header[2])
            body = None
            if not check_only:
                body = proc.stdout.read(size)
                proc.stdout.read(1)  # trailing LF
            yield oid, kind, size, body
    finally:
        writer.join(timeout=5)
        proc.stdout.close()
        proc.wait()


def _tree_entries(body: bytes, oid_len: int) -> Iterator[tuple[str, str, str]]:
    """(mode, name, oid hex) of a raw tree object."""
    i = 0
    while i < len(body):
        space = body.index(b" ", i)
        nul = body.index(b"\x00", space)
        mode = body[i:space].decode()
        name = body[space + 1 : nul].decode("utf-8", "surrogateescape")
        oid = body[nul + 1 : nul + 1 + oid_len].hex()
        i = nul + 1 + oid_len
        yield mode, name, oid


def history_check(
    repo: Path,
    tips: Sequence[str],
    have: Sequence[str],
    *,
    max_blob_bytes: int = SCAN_MAX_BLOB_BYTES,
) -> HistoryCheck:
    """Scan EVERY object ``git bundle create <tips> --not <have>`` would carry (§8.1).

    Path rule: each outgoing tree's entries are joined to that tree's path and
    passed to ``denylist.is_sensitive`` (the peer-file-read classifier) — all
    names a blob is reachable under in outgoing trees, not the single name
    ``rev-list`` prints. Content rule: every outgoing blob is read and passed to
    ``secret_guard.credential_classes``. A blob over ``max_blob_bytes``, binary
    (NUL byte or not UTF-8) or unreadable is a finding with why
    ``history_unscannable`` — withheld, never silently skipped. Gitlinks
    (submodules) carry no object and are the manifest's ``submodule`` reason.

    Best effort: shape scanning recognises known credential shapes and names
    only. Raises ``MoveWorksetError("history_unverifiable")`` when the object
    set itself cannot be enumerated.
    """
    from local_operator.supplements.denylist import is_sensitive
    from local_operator.team_bridge.secret_guard import credential_classes

    root = Path(repo)
    tip_ids = tuple(sorted({_out(root, "rev-parse", "--verify", f"{t}^{{commit}}") for t in tips}))
    have_ids = tuple(sorted({h for h in have if _HEX_RE.match(h) and _is_commit(root, h)}))
    proc = _git(root, "rev-list", "--objects", *tip_ids, "--not", *have_ids, check=False)
    if proc.returncode != 0:
        raise MoveWorksetError("history_unverifiable", proc.stderr.strip()[:200])
    listed: dict[str, str] = {}
    for line in proc.stdout.splitlines():
        oid, _, path = line.partition(" ")
        listed.setdefault(oid, path)
    oid_len = len(next(iter(listed), "0" * 40)) // 2
    kinds = {oid: kind for oid, kind, _, _ in _batch(root, list(listed), check_only=True)}
    findings: dict[tuple[str, str, str, str], Excluded] = {}

    def add(path: str, why: str, rule: str, oid: str) -> None:
        findings.setdefault(
            (path, why, rule, oid), Excluded(path=path, why=why, rule=rule, object=oid)
        )

    # Every outgoing object is reachable from an outgoing commit's root tree
    # through outgoing trees only (a new object changes every tree above it),
    # so walking from the roots names each outgoing object under ALL the paths
    # it appears at — ``rev-list`` prints one arbitrary path per object, and a
    # blob shared by ``a.txt`` and ``.env`` must be judged as ``.env`` too.
    entries: dict[str, list[tuple[str, str, str]]] = {}
    trees = [oid for oid, kind in kinds.items() if kind == "tree"]
    for oid, kind, _, body in _batch(root, trees, check_only=False):
        if kind != "tree" or body is None:
            add(listed.get(oid, ""), "history_unscannable", "unreadable", oid)
            continue
        entries[oid] = list(_tree_entries(body, oid_len))
    commits = [oid for oid, kind in kinds.items() if kind == "commit"]
    paths: dict[str, list[str]] = {}
    queue: list[tuple[str, str]] = []
    for oid, kind, _, body in _batch(root, commits, check_only=False):
        if kind != "commit" or body is None:
            add("", "history_unscannable", "unreadable", oid)
            continue
        text = body.decode("utf-8", "replace")
        # A commit message is outgoing bytes too (a pasted token in a message).
        for label in credential_classes(text.partition("\n\n")[2]):
            add("", "history_secret_content", label, oid)
        root_tree = text.split("\n", 1)[0].removeprefix("tree ").strip()
        if root_tree in entries:
            queue.append((root_tree, ""))
    seen: set[tuple[str, str]] = set()
    while queue:
        tree, base = queue.pop()
        if (tree, base) in seen:
            continue
        seen.add((tree, base))
        for mode, name, child in entries.get(tree, ()):
            if mode == "160000" or child not in listed:
                # Gitlinks carry no object (the manifest's ``submodule``
                # reason); a child the destination already has is not sent.
                continue
            full = f"{base}/{name}" if base else name
            if child in entries:
                queue.append((child, full))
                continue
            names = paths.setdefault(child, [])
            if full not in names and len(names) < 64:
                names.append(full)
            # Judged under a virtual anchor, not the checkout's real location:
            # history is a property of the repository, and a checkout that
            # happens to sit under a ROOT rule (the config dir, a scratchpad)
            # must not flag every name, nor resolve a historical name through
            # a symlink that exists in today's working tree. Name, suffix and
            # component rules (``.env``, ``.ssh/id_rsa``) still apply.
            rule = is_sensitive(f"{_HISTORY_ANCHOR}/{full}")
            if rule:
                add(full, "history_secret_path", rule, child)
    blobs = [oid for oid, kind in kinds.items() if kind == "blob"]
    for oid, kind, size, body in _batch(root, blobs, check_only=False):
        names = paths.get(oid) or [listed.get(oid, "")]
        if kind != "blob" or body is None:
            for name in names:
                add(name, "history_unscannable", "unreadable", oid)
            continue
        if size > max_blob_bytes:
            for name in names:
                add(name, "history_unscannable", "over_scan_limit", oid)
            continue
        try:
            if b"\x00" in body:
                raise UnicodeDecodeError("utf-8", body, 0, 1, "nul byte")
            text = body.decode("utf-8")
        except UnicodeDecodeError:
            for name in names:
                add(name, "history_unscannable", "binary", oid)
            continue
        for label in credential_classes(text):
            for name in names:
                add(name, "history_secret_content", label, oid)
    for oid, kind in kinds.items():
        if kind == "missing":
            add(listed.get(oid, ""), "history_unscannable", "unreadable", oid)
    digest = "sha256:" + hashlib.sha256("\n".join(sorted(listed)).encode()).hexdigest()
    return HistoryCheck(
        tips=tip_ids,
        have=have_ids,
        objects=len(listed),
        blobs=len(blobs),
        scanned_digest=digest,
        findings=tuple(
            sorted(findings.values(), key=lambda f: (f["path"], f["why"], f["rule"], f["object"]))
        ),
    )


# ------------------------------------------------------------------------ bundle


@dataclass(frozen=True)
class BundleHeader:
    version: int
    prerequisites: tuple[str, ...]
    refs: tuple[tuple[str, str], ...]  # (ref name, sha)


def read_bundle_header(path: Path) -> BundleHeader:
    """Parse a v2/v3 bundle header (everything before the pack) without git."""
    with open(path, "rb") as fh:
        first = fh.readline().decode().strip()
        if first == "# v2 git bundle":
            version = 2
        elif first == "# v3 git bundle":
            version = 3
        else:
            raise MoveWorksetError("bundle_malformed", "signature")
        prereq: list[str] = []
        refs: list[tuple[str, str]] = []
        while True:
            line = fh.readline()
            if not line:
                raise MoveWorksetError("bundle_malformed", "truncated header")
            text = line.decode().rstrip("\n")
            if text == "":
                break
            if text.startswith("@"):
                continue
            if text.startswith("-"):
                prereq.append(text[1:].split(" ", 1)[0])
            else:
                sha, _, name = text.partition(" ")
                refs.append((name, sha))
    return BundleHeader(version, tuple(prereq), tuple(refs))


@dataclass(frozen=True)
class Artifact:
    """An immutable, digest-named staged bundle (§4 W2).

    ``digest`` is ``sha256:<hex>`` of the file bytes, which is also its file name
    (hex part) under the staging directory, the allow-list W4 serves from.
    """

    digest: str
    path: Path
    bytes: int
    refs: tuple[tuple[str, str], ...]
    have: tuple[str, ...]
    prerequisites: tuple[str, ...]
    scanned_digest: str


def _sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def _private_dir(path: Path) -> Path:
    """Create (0700) or accept ``path``; refuse a symlink in its place."""
    path.mkdir(parents=True, exist_ok=True, mode=0o700)
    if path.is_symlink() or not path.is_dir():
        raise ContainmentError(str(path))
    return path


def _write_json_atomic(path: Path, payload: dict[str, Any]) -> None:
    fd, tmp = tempfile.mkstemp(prefix=".tmp-", dir=path.parent)
    try:
        with os.fdopen(fd, "w") as fh:
            json.dump(payload, fh, sort_keys=True, indent=1)
        os.chmod(tmp, 0o600)
        os.replace(tmp, path)
    except BaseException:
        Path(tmp).unlink(missing_ok=True)
        raise


def stage_bundle(repo: Path, refs: Sequence[str], have: Sequence[str], out_dir: Path) -> Artifact:
    """Bundle ``refs --not <have>`` into ``out_dir/<sha256>`` after scanning it (§4 W2, §8.1).

    ``have=[]`` makes a full bundle. ``have`` entries the repo does not hold as
    commits are dropped (``--not <missing>`` would fail the whole bundle).
    Raises ``HistoryWithheld`` on any scan finding — there is no override in
    v1 — and ``MoveWorksetError("empty_bundle")`` when the destination already
    holds everything.

    The staged directory is the cache: a sidecar ``<hex>.json`` records the
    ref tips, ``have`` and scan digest, so a re-plan with the same inputs
    returns the same artifact even if a fresh pack would not be byte-equal.
    Binding: the written bundle's header refs must equal the scanned tips
    (refs whose tip ``have`` already covers are legitimately omitted by git and
    must be reachable from ``have``); a ref that moved between scan and bundle
    fails the stage instead of serving unscanned bytes.
    """
    root = Path(repo)
    out = _private_dir(Path(out_dir))
    resolved = tuple((r, _out(root, "rev-parse", "--verify", f"{r}^{{commit}}")) for r in refs)
    have_ids = tuple(sorted({h for h in have if _HEX_RE.match(h) and _is_commit(root, h)}))
    check = history_check(root, [sha for _, sha in resolved], have_ids)
    if check.findings:
        raise HistoryWithheld(check)
    key = {
        "refs": [list(r) for r in resolved],
        "have": list(have_ids),
        "scanned_digest": check.scanned_digest,
    }
    for sidecar in sorted(out.glob("*.json")):
        try:
            saved = json.loads(sidecar.read_text())
        except (OSError, ValueError):
            continue
        if {k: saved.get(k) for k in key} != key:
            continue
        cached = out / sidecar.stem
        if cached.is_file() and not cached.is_symlink() and _sha256_file(cached) == sidecar.stem:
            return _artifact(cached, sidecar.stem, resolved, have_ids, check)
    fd, tmp_name = tempfile.mkstemp(prefix=".tmp-", suffix=".bundle", dir=out)
    os.close(fd)
    tmp = Path(tmp_name)
    try:
        tmp.unlink()  # git bundle create takes its own lock on the path
        argv = ["-c", "pack.threads=1", "bundle", "create", str(tmp), *[r for r, _ in resolved]]
        if have_ids:
            argv += ["--not", *have_ids]
        proc = _git(root, *argv, check=False)
        if proc.returncode != 0:
            if "empty bundle" in proc.stderr:
                raise MoveWorksetError("empty_bundle", "destination holds every ref")
            raise MoveWorksetError("bundle_failed", proc.stderr.strip()[:300])
        os.chmod(tmp, 0o600)
        if not verify_bundle(root, tmp):
            raise MoveWorksetError("bundle_unverifiable", "source-side verify")
        header = read_bundle_header(tmp)
        wanted = dict(resolved)
        for name, sha in header.refs:
            if wanted.get(name) != sha:
                raise MoveWorksetError("bundle_unbound", f"{name} is not a scanned tip")
        for name, sha in resolved:
            if name not in dict(header.refs) and not any(
                _git(root, "merge-base", "--is-ancestor", sha, h, check=False).returncode == 0
                for h in have_ids
            ):
                raise MoveWorksetError("bundle_unbound", f"{name} missing from bundle")
        hexdigest = _sha256_file(tmp)
        final = out / hexdigest
        os.replace(tmp, final)
        _write_json_atomic(out / f"{hexdigest}.json", {**key, "digest": f"sha256:{hexdigest}"})
        return _artifact(final, hexdigest, resolved, have_ids, check)
    finally:
        tmp.unlink(missing_ok=True)


def _artifact(
    path: Path, hexdigest: str, resolved: Any, have_ids: Any, check: HistoryCheck
) -> Artifact:
    return Artifact(
        digest=f"sha256:{hexdigest}",
        path=path,
        bytes=path.stat().st_size,
        refs=tuple(resolved),
        have=tuple(have_ids),
        prerequisites=read_bundle_header(path).prerequisites,
        scanned_digest=check.scanned_digest,
    )


def verify_bundle(repo: Path, bundle: Path) -> bool:
    """``git bundle verify`` in ``repo``: header names refs and every prerequisite is present.

    ``git bundle verify`` checks the header and prerequisites only, not the
    pack bytes (a header followed by garbage passes it), so a header with no
    refs is refused here; pack integrity is the sha256 digest of the transfer
    plus the ``index-pack`` that ``git fetch`` runs at restore.
    """
    try:
        if not read_bundle_header(bundle).refs:
            return False
    except (OSError, UnicodeDecodeError, MoveWorksetError):
        return False
    return _git(repo, "bundle", "verify", str(bundle), check=False).returncode == 0


def verify_or_replan(
    fetch: Callable[[Sequence[str]], Path], repo: Path, have: Sequence[str]
) -> tuple[Path, bool]:
    """W5: verify the fetched bundle; on failure re-plan ONCE with ``have=[]``.

    ``fetch(have)`` runs W2-W4 and returns the local bundle path. Returns
    ``(bundle, replanned)``; a second failure raises ``bundle_unverifiable`` so
    the caller rolls the move back (design §4 W5).
    """
    first = fetch(list(have))
    if verify_bundle(repo, first):
        return first, False
    if not have:
        raise MoveWorksetError("bundle_unverifiable", "full bundle failed verify")
    second = fetch([])
    if verify_bundle(repo, second):
        return second, True
    raise MoveWorksetError("bundle_unverifiable", "re-plan with have=[] failed verify")


def compute_have(primary: Path, tips: Sequence[str], *, remote: str | None = None) -> list[str]:
    """W1: the subset of ``tips`` the destination primary holds, after a bounded fetch.

    The fetch uses the destination user's own credentials (helpers kept) and is
    best effort: offline just means a smaller ``have`` and a bigger bundle.
    """
    root = Path(primary)
    if remote:
        try:
            _git(
                root,
                "fetch",
                "--no-tags",
                "--quiet",
                remote,
                check=False,
                timeout=FETCH_TIMEOUT_S,
                credentials=True,
            )
        except subprocess.TimeoutExpired:
            pass
    return [t for t in dict.fromkeys(tips) if _HEX_RE.match(t) and _is_commit(root, t)]


# ------------------------------------------------------------- paths and remap


def home_relative(path: Path | str, *, home: Path) -> str | None:
    """Owner side: ``~/…`` form of ``path``, or ``None`` when outside ``home`` (§7.1)."""
    real = os.path.realpath(path)
    base = os.path.realpath(home)
    if real == base or not real.startswith(base.rstrip("/") + "/"):
        return None
    return "~/" + Path(os.path.relpath(real, base)).as_posix()


def contained_path(home: Path, home_rel: str) -> Path:
    """Destination side: resolve ``~/…`` under ``home`` with every component checked.

    Refused (``ContainmentError``): not ``~/``-prefixed, empty/``.``/``..``
    components, NUL or backslash, any EXISTING component that is a symlink, a
    non-directory parent, or an alias — a name that exists but is not listed
    verbatim in its parent (case-insensitive or normalisation-insensitive
    filesystems such as APFS, where ``Proj`` would open ``proj``). Missing
    components are allowed; the caller creates them with ``_make_dirs``.
    Deliberately not ``carry.translate_host_path``, which guesses (§7.1).
    """
    if not isinstance(home_rel, str) or not home_rel.startswith("~/"):
        raise ContainmentError(f"not home-relative: {home_rel!r}")
    parts = home_rel[2:].split("/")
    if not parts or any(p in ("", ".", "..") or "\x00" in p or "\\" in p for p in parts):
        raise ContainmentError(f"unsafe components: {home_rel!r}")
    current = Path(os.path.realpath(home))
    for i, part in enumerate(parts):
        nxt = current / part
        if os.path.lexists(nxt):
            if part not in os.listdir(current):
                raise ContainmentError(f"aliased component: {home_rel!r}")
            st = os.lstat(nxt)
            if stat.S_ISLNK(st.st_mode):
                raise ContainmentError(f"symlinked component: {home_rel!r}")
            if i < len(parts) - 1 and not stat.S_ISDIR(st.st_mode):
                raise ContainmentError(f"non-directory component: {home_rel!r}")
        current = nxt
    return current


def _make_dirs(home: Path, target: Path) -> None:
    """mkdir each missing parent of ``target`` one level at a time, re-checking each."""
    base = Path(os.path.realpath(home))
    rel = target.relative_to(base)
    current = base
    for part in rel.parts[:-1]:
        current = current / part
        try:
            current.mkdir(mode=0o755)
        except FileExistsError:
            pass
        if current.is_symlink() or not current.is_dir():
            raise ContainmentError(str(current))


def remap(home_rel: str, *, home: Path) -> Path:
    """The destination path for a home-relative source path (§7.1), containment-checked."""
    return contained_path(home, home_rel)


def conflict_path(target_rel: str, device: str, now: float | None = None) -> str:
    """``<path>.moved-<device>-<YYYYmmddHHMM>`` (§7.3): where a conflicting restore lands."""
    dev = _DEVICE_SAFE_RE.sub("-", device).strip("-.") or "device"
    ts = time.strftime("%Y%m%d%H%M", time.localtime(now if now is not None else time.time()))
    return f"{target_rel}.moved-{dev}-{ts}"


def normalize_remote_url(url: str) -> str:
    """Comparable form of a remote URL (§7.2).

    Userinfo stripped, ``.git``/trailing slash dropped, host lower-cased, and
    scp-form (``git@host:a/b``), ``ssh://`` and ``https://`` folded to
    ``host/path`` — the same repository is routinely cloned over ssh on one
    device and https on another. Local paths compare by real path.
    """
    u = url.strip()
    m = re.match(r"^[a-zA-Z][a-zA-Z0-9+.-]*://(?:[^@/]*@)?([^/:]+)(?::\d+)?(/.*)?$", u)
    if u.startswith("file://"):
        u = os.path.realpath(u[len("file://") :])
    elif m:
        u = f"{m.group(1).lower()}{m.group(2) or ''}"
    elif re.match(r"^(?:[^@/]+@)?[^/:]+:(?!//)", u) and not os.path.isabs(u):
        user_host, _, path = u.partition(":")
        u = f"{user_host.rpartition('@')[2].lower()}/{path.lstrip('/')}"
    else:
        u = os.path.realpath(u)
    u = u.rstrip("/")
    return u[:-4] if u.endswith(".git") else u


@dataclass(frozen=True)
class PrimaryResolution:
    """§7.2 outcome. ``state``: ``present`` | ``conflict`` | ``absent`` | ``unsafe``."""

    state: str
    path: Path | None
    remote: str | None = None
    reason: str = ""
    cloned: bool = False


def resolve_primary(remote_url: str, home_rel: str, *, home: Path) -> PrimaryResolution:
    """Find the destination checkout for a moving worktree's repository (§7.2).

    ``present`` only when the remapped path is a non-bare git repository with
    a remote whose normalised URL equals ``remote_url``'s; any other existing
    thing there is ``conflict`` (never clone over or beside an unrelated
    checkout); nothing there is ``absent`` (the caller may ``clone_primary``).
    """
    try:
        path = contained_path(home, home_rel)
    except ContainmentError as exc:
        return PrimaryResolution("unsafe", None, reason=exc.reason)
    if not os.path.lexists(path):
        return PrimaryResolution("absent", path, reason="primary_absent")
    if not path.is_dir() or _git(path, "rev-parse", "--git-dir", check=False).returncode != 0:
        return PrimaryResolution("conflict", path, reason="destination_conflict")
    try:
        top = _toplevel(path)
    except MoveWorksetError:
        return PrimaryResolution("conflict", path, reason="destination_conflict")
    if os.path.realpath(top) != os.path.realpath(path):
        return PrimaryResolution("conflict", path, reason="destination_conflict")
    want = normalize_remote_url(remote_url)
    for name in _out(path, "remote").splitlines():
        url = _git(path, "remote", "get-url", name, check=False).stdout.strip()
        if url and normalize_remote_url(url) == want:
            return PrimaryResolution("present", path, remote=name)
    return PrimaryResolution("conflict", path, reason="destination_conflict")


def _rename_noreplace(src: Path, dst: Path) -> None:
    """Atomic rename that fails with ``FileExistsError`` if ``dst`` exists.

    Plain ``os.rename`` replaces an existing EMPTY directory on POSIX; the
    kernel no-replace forms close that window. Filesystems without support
    (EINVAL/ENOSYS) fall back to an lexists check, whose only race is an empty
    directory appearing in between — nothing a user could lose.
    """
    fn = None
    if sys.platform.startswith("linux"):
        libc = ctypes.CDLL(ctypes.util.find_library("c"), use_errno=True)
        renameat2 = getattr(libc, "renameat2", None)
        if renameat2 is not None:
            fn = lambda: renameat2(-100, os.fsencode(src), -100, os.fsencode(dst), 1)  # noqa: E731
    elif sys.platform == "darwin":
        libc = ctypes.CDLL(ctypes.util.find_library("c"), use_errno=True)
        renamex = getattr(libc, "renamex_np", None)
        if renamex is not None:
            fn = lambda: renamex(os.fsencode(src), os.fsencode(dst), 0x4)  # noqa: E731  RENAME_EXCL
    if fn is not None:
        if fn() == 0:
            return
        err = ctypes.get_errno()
        if err == errno.EEXIST:
            raise FileExistsError(errno.EEXIST, "destination exists", str(dst))
        if err not in (errno.EINVAL, errno.ENOSYS, errno.ENOTSUP):
            raise OSError(err, os.strerror(err), str(dst))
    if os.path.lexists(dst):
        raise FileExistsError(errno.EEXIST, "destination exists", str(dst))
    os.rename(src, dst)


def clone_primary(
    remote_url: str, home_rel: str, *, home: Path, pack_bytes: int, timeout: float | None = None
) -> PrimaryResolution:
    """W1 / manager Q1: clone the absent primary from its origin, no-clobber.

    Only for a primary the user confirmed in the manifest. Free space must hold
    ``pack_bytes x 2 + 1 GiB``; the clone lands in a sibling staging directory
    and is renamed into a path that must still not exist. The destination's
    own credentials authenticate it.
    """
    target = contained_path(home, home_rel)
    if os.path.lexists(target):
        return resolve_primary(remote_url, home_rel, home=home)
    _make_dirs(home, target)
    need = max(pack_bytes, 0) * 2 + CLONE_HEADROOM_BYTES
    if shutil.disk_usage(target.parent).free < need:
        raise MoveWorksetError("insufficient_space", f"need {need} bytes")
    staging = Path(tempfile.mkdtemp(prefix=".lop-clone-", dir=target.parent))
    try:
        dest = staging / target.name
        try:
            _git(
                staging,
                "clone",
                "--quiet",
                "--no-tags",
                remote_url,
                str(dest),
                timeout=timeout,
                credentials=True,
            )
        except subprocess.TimeoutExpired as exc:
            raise MoveWorksetError("clone_failed", "timeout") from exc
        except MoveWorksetError as exc:
            raise MoveWorksetError("clone_failed", exc.detail) from exc
        _rename_noreplace(dest, target)
    finally:
        # Only the staging directory this call created; after a successful
        # publish it is empty, after a failure it holds the partial clone.
        shutil.rmtree(staging, ignore_errors=True)
    res = resolve_primary(remote_url, home_rel, home=home)
    return PrimaryResolution(
        res.state, res.path, res.remote, res.reason, cloned=res.state == "present"
    )


# ----------------------------------------------------------------------- restore


@dataclass(frozen=True)
class RestorePlan:
    """One worktree to restore on the destination (built from the manifest + reply).

    ``path``/``primary`` are the SOURCE's home-relative paths (§7.1).
    ``head``/``index``/``worktree`` are the expected snapshot SHAs — the bundle
    is accepted only if its refs match them. ``upstream_merge`` is the upstream
    branch's ref on the remote (``refs/heads/feat``). ``src_device`` names the
    source in conflict suffixes. ``is_primary`` marks the source's primary
    checkout itself (§7.2 case 4); ``fresh_clone`` says W1 cloned the primary
    for this move, so it may become that checkout.
    """

    session_id: str
    n: int
    bundle_digest: str
    path: str
    primary: str
    remote_url: str
    head: str
    index: str
    worktree: str
    branch: str | None = None
    upstream_merge: str | None = None
    src_device: str = "remote"
    is_primary: bool = False
    fresh_clone: bool = False


@dataclass(frozen=True)
class RestoreReport:
    """``state``: ``restored`` | ``left_behind`` | ``pending_restore``.

    ``mode``: ``new`` | ``idempotent`` | ``in_place`` | ``beside`` (conflict
    suffix). ``remapped`` is ``(from, to)`` home-relative when ``dest_path``
    differs from the source path.
    """

    state: str
    path: str
    dest_path: str | None = None
    branch: str | None = None
    mode: str = ""
    reason: str = ""
    remapped: tuple[str, str] | None = None
    notes: tuple[str, ...] = field(default_factory=tuple)


def _worktrees_of(primary: Path) -> dict[str, str | None]:
    """realpath -> checked-out branch ref (None detached) for every worktree of ``primary``."""
    out: dict[str, str | None] = {}
    path = None
    for line in _out(primary, "worktree", "list", "--porcelain").splitlines() + [""]:
        if line.startswith("worktree "):
            path = os.path.realpath(line[len("worktree ") :])
            out[path] = None
        elif line.startswith("branch ") and path:
            out[path] = line[len("branch ") :]
    return out


def _state(root: Path, excludes: Sequence[str]) -> tuple[str, str, str] | None:
    """(HEAD, index tree, worktree tree) measured now, the way ``snapshot`` measures."""
    proc = _git(root, "rev-parse", "--verify", "-q", "HEAD^{commit}", check=False)
    if proc.returncode != 0:
        return None
    head = proc.stdout.strip()
    git_dir = Path(_out(root, "rev-parse", "--absolute-git-dir"))
    try:
        idx, wt = _trees(root, head, excludes, git_dir)
    except MoveWorksetError:
        return None
    return head, idx, wt


def _choose_branch(
    primary: Path, branch: str, head: str, device: str, *, own: Path | None = None
) -> str:
    """§5.2 branch rule: create; reuse if equal; fast-forward only; else ``-moved-<device>``.

    Never moves a branch checked out in another worktree (``own`` — the tree
    being updated in place — does not count as "another").
    """
    checked_out = {
        ref
        for wt, ref in _worktrees_of(primary).items()
        if ref and (own is None or wt != os.path.realpath(own))
    }
    dev = _DEVICE_SAFE_RE.sub("-", device).strip("-.") or "device"
    candidates = [branch] + [f"{branch}-moved-{dev}" + (f"-{i}" if i else "") for i in range(0, 50)]
    for name in candidates:
        ref = f"refs/heads/{name}"
        if _git(primary, "check-ref-format", ref, check=False).returncode != 0:
            raise MoveWorksetError("invalid_branch", name)
        cur = _git(primary, "rev-parse", "--verify", "-q", ref, check=False).stdout.strip()
        if not cur:
            _git(primary, "branch", "--no-track", name, head)
            return name
        if ref in checked_out:
            continue
        if cur == head:
            return name
        if (
            name == branch
            and _git(primary, "merge-base", "--is-ancestor", cur, head, check=False).returncode == 0
        ):
            _git(primary, "update-ref", ref, head, cur)
            return name
    raise MoveWorksetError("branch_unavailable", branch)


def _set_upstream(primary: Path, branch: str | None, remote: str | None, merge: str | None) -> None:
    # Set even before the remote-tracking ref exists locally (§5.2); the next
    # fetch fills it in and `git status` then reports ahead/behind.
    if branch and remote and merge and merge.startswith("refs/heads/"):
        _git(primary, "config", f"branch.{branch}.remote", remote)
        _git(primary, "config", f"branch.{branch}.merge", merge)


def _journal_entry(config: Path | None, session_id: str, target_rel: str) -> dict[str, Any] | None:
    if config is None:
        return None
    path = Path(config) / "network" / "workset-departed" / f"{session_id}.json"
    try:
        if path.is_symlink():
            return None
        record = json.loads(path.read_text())
    except (OSError, ValueError):
        return None
    if not isinstance(record, dict) or record.get("session_id") != session_id:
        return None
    for entry in record.get("worktrees") or []:
        if isinstance(entry, dict) and entry.get("path") == target_rel:
            return entry
    return None


def _marker_ok(target: Path, session_id: str) -> bool:
    proc = _git(target, "rev-parse", "--absolute-git-dir", check=False)
    if proc.returncode != 0:
        return False
    marker = Path(proc.stdout.strip()) / "lop-moved.json"
    try:
        data = json.loads(marker.read_text())
    except (OSError, ValueError):
        return False
    return isinstance(data, dict) and data.get("session_id") == session_id


def _update_in_place(
    target: Path, primary: Path, from_tree: str, plan: RestorePlan, restore_index: str
) -> str:
    """Switch an unchanged tree from ``from_tree`` to the snapshot, keeping the index split.

    index := from_tree (matches the files) → two-tree ``read-tree -m -u`` to the
    new worktree tree (git refuses to clobber an untracked file in the way) →
    index := new index tree. Any failure restores the original index and the
    caller lands the snapshot beside instead.
    """
    try:
        _git(target, "read-tree", from_tree)
        # read-tree drops stat data; refresh re-stats the (verified equal)
        # files so the two-tree switch sees them up to date. Exit status is
        # ignored: an entry still stale (an excluded, locally edited file) makes
        # the switch below refuse rather than overwrite it.
        _git(target, "update-index", "-q", "--refresh", check=False)
        _git(target, "read-tree", "-m", "-u", from_tree, plan.worktree)
    except MoveWorksetError:
        _git(target, "read-tree", restore_index, check=False)
        raise
    branch = None
    if plan.branch:
        branch = _choose_branch(primary, plan.branch, plan.head, plan.src_device, own=target)
        _git(target, "symbolic-ref", "HEAD", f"refs/heads/{branch}")
    else:
        _git(target, "update-ref", "--no-deref", "HEAD", plan.head)
    _git(target, "read-tree", plan.index)
    _git(target, "update-index", "-q", "--refresh", check=False)
    return branch or ""


def restore(
    artifact_dir: Path,
    plan: RestorePlan,
    *,
    home: Path,
    config: Path | None = None,
    now: float | None = None,
) -> RestoreReport:
    """Restore one worktree from a verified bundle on the destination (§5.2, §7).

    Target choice (§7.3): the remapped source path; the source's primary
    checkout lands as ``<primary>/.worktrees/moved-<sid8>`` unless W1 cloned
    the primary for it. If the target exists: a registered worktree already in
    the snapshot state → ``idempotent``; a tree this device's TRUSTED JOURNAL
    (``config``) recorded as departed, carrying the marker, and re-measured
    unchanged since → updated ``in_place`` (recall, §10); anything else →
    ``beside`` at ``conflict_path``. Never overwrites, never refuses.

    A new tree is built at ``<parent>/.lop-restore-*/<name>`` and published by
    a no-replace rename, then ``git worktree repair`` points the admin entry at
    its final path, so ``git worktree list`` on the primary shows it.

    Returns a report rather than raising for per-worktree outcomes; unexpected
    git failures raise ``MoveWorksetError`` and leave the verified artifact for
    a ``pending_restore`` retry.
    """
    _validate_session_id(plan.session_id)
    hexdigest = plan.bundle_digest.removeprefix("sha256:")
    if not re.fullmatch(r"[0-9a-f]{64}", hexdigest):
        raise MoveWorksetError("invalid_digest", plan.bundle_digest)
    bundle = Path(artifact_dir) / hexdigest
    if Path(artifact_dir).is_symlink() or bundle.is_symlink() or not bundle.is_file():
        return RestoreReport("pending_restore", plan.path, reason="artifact_missing")
    if _sha256_file(bundle) != hexdigest:
        return RestoreReport("pending_restore", plan.path, reason="artifact_corrupt")

    res = resolve_primary(plan.remote_url, plan.primary, home=home)
    if res.state != "present" or res.path is None:
        state = "pending_restore" if res.state == "absent" else "left_behind"
        return RestoreReport(state, plan.path, reason=res.reason)
    primary = res.path
    prefix = f"{REF_ROOT}/{plan.session_id}"
    # Forced: on a recall the same session id's departure refs already exist
    # here and are superseded; the journal, not these refs, holds the
    # departure state the in-place check trusts.
    _git(primary, "fetch", "--no-tags", "--quiet", str(bundle), f"+{prefix}/*:{prefix}/*")
    got_index = _out(
        primary, "rev-parse", "--verify", snapshot_ref(plan.session_id, plan.n, "index")
    )
    got_wt = _out(
        primary, "rev-parse", "--verify", snapshot_ref(plan.session_id, plan.n, "worktree")
    )
    got_head = _out(primary, "rev-parse", "--verify", f"{got_index}^1")
    if (got_head, got_index, got_wt) != (plan.head, plan.index, plan.worktree):
        return RestoreReport("pending_restore", plan.path, reason="snapshot_mismatch")
    want_idx_tree = _out(primary, "rev-parse", f"{plan.index}^{{tree}}")
    want_wt_tree = _out(primary, "rev-parse", f"{plan.worktree}^{{tree}}")

    if plan.is_primary and plan.fresh_clone:
        if _out(primary, "status", "--porcelain", "--untracked-files=all"):
            return RestoreReport("left_behind", plan.path, reason="destination_conflict")
        clean = _out(primary, "rev-parse", "HEAD^{tree}")
        branch = _update_in_place(primary, primary, clean, plan, clean)
        _set_upstream(primary, branch, res.remote, plan.upstream_merge)
        return RestoreReport(
            "restored", plan.path, dest_path=plan.primary, branch=branch or None, mode="in_place"
        )

    target_rel = plan.path
    if plan.is_primary:
        target_rel = f"{plan.primary}/.worktrees/moved-{plan.session_id[:8]}"
    try:
        target = contained_path(home, target_rel)
    except ContainmentError as exc:
        return RestoreReport("left_behind", plan.path, reason=exc.reason)

    mode = "new"
    if os.path.lexists(target):
        registered = _worktrees_of(primary)
        real = os.path.realpath(target)
        if real in registered and _state(target, ()) == (plan.head, want_idx_tree, want_wt_tree):
            return _report(plan, target_rel, "idempotent", registered[real])
        entry = _journal_entry(config, plan.session_id, target_rel)
        if (
            entry is not None
            and real in registered
            and _marker_ok(target, plan.session_id)
            and _state(target, tuple(entry.get("excludes") or ()))
            == (entry.get("head"), entry.get("index_tree"), entry.get("worktree_tree"))
        ):
            try:
                branch = _update_in_place(
                    target, primary, str(entry["worktree_tree"]), plan, str(entry["index_tree"])
                )
            except MoveWorksetError:
                pass  # e.g. an untracked file in the way: land beside instead
            else:
                _set_upstream(primary, branch, res.remote, plan.upstream_merge)
                return _report(
                    plan, target_rel, "in_place", f"refs/heads/{branch}" if branch else None
                )
        mode = "beside"
        base = conflict_path(target_rel, plan.src_device, now)
        for i in range(100):
            candidate = base if i == 0 else f"{base}-{i}"
            target = contained_path(home, candidate)
            if not os.path.lexists(target):
                target_rel = candidate
                break
        else:
            return RestoreReport("left_behind", plan.path, reason="destination_conflict")

    branch = (
        _choose_branch(primary, plan.branch, plan.head, plan.src_device) if plan.branch else None
    )
    _make_dirs(home, target)
    staging = Path(tempfile.mkdtemp(prefix=".lop-restore-", dir=target.parent))
    staged = staging / target.name
    published = False
    try:
        if branch:
            _git(primary, "worktree", "add", "--no-checkout", "--quiet", str(staged), branch)
        else:
            _git(
                primary,
                "worktree",
                "add",
                "--no-checkout",
                "--quiet",
                "--detach",
                str(staged),
                plan.head,
            )
        _git(staged, "read-tree", plan.worktree)
        # checkout-index writes the worktree snapshot's files — untracked ones
        # included, exec bits and symlinks as recorded — into the empty tree;
        # git refuses to write through a symlinked leading path.
        _git(staged, "checkout-index", "-a", "-f")
        _git(staged, "read-tree", plan.index)
        _git(staged, "update-index", "-q", "--refresh", check=False)
        _rename_noreplace(staged, target)
        published = True
        _git(primary, "worktree", "repair", str(target))
    except FileExistsError:
        return RestoreReport("pending_restore", plan.path, reason="destination_conflict")
    finally:
        if not published:
            # Only what this call created: its own staging worktree entry.
            _git(primary, "worktree", "remove", "--force", str(staged), check=False)
            _git(primary, "worktree", "prune", check=False)
        shutil.rmtree(staging, ignore_errors=True) if not published else _rmdir_quiet(staging)
    _set_upstream(primary, branch, res.remote, plan.upstream_merge)
    return _report(plan, target_rel, mode, f"refs/heads/{branch}" if branch else None)


def _rmdir_quiet(path: Path) -> None:
    try:
        path.rmdir()  # empty after the publish; never recursive here
    except OSError:
        pass


def _report(plan: RestorePlan, dest_rel: str, mode: str, branch_ref: str | None) -> RestoreReport:
    branch = branch_ref.removeprefix("refs/heads/") if branch_ref else None
    remapped = (plan.path, dest_rel) if dest_rel != plan.path else None
    return RestoreReport(
        "restored", plan.path, dest_path=dest_rel, branch=branch, mode=mode, remapped=remapped
    )


# ----------------------------------------------------------------------- recipes


def recipes_for(worktree: Path, primary: Path, *, home: Path | None = None) -> list[Recipe]:
    """§9 dependency recipes for a restored worktree — commands to PRINT, never run.

    Dependency trees are left behind (they are platform-specific and GB-scale);
    these recreate them under the dependency-commons rule: a pnpm worktree at
    ``<primary>/.worktrees/<name>`` whose lockfile matches the primary's shares
    the primary's ``node_modules`` by relative symlink; venvs are recreated,
    never symlinked. The AGENTS.md-named setup command row is not derived
    (free text; left to the surface).
    """
    wt = Path(worktree)
    pr = Path(primary)
    cwd = (home_relative(wt, home=home) or str(wt)) if home else str(wt)
    out: list[Recipe] = []

    def add(kind: str, run: str) -> None:
        out.append(Recipe(kind=kind, run=run, cwd=cwd))

    pnpm = wt / "pnpm-lock.yaml"
    if pnpm.is_file():
        shares = (
            wt.parent.name == ".worktrees"
            and os.path.realpath(wt.parent.parent) == os.path.realpath(pr)
            and (pr / "pnpm-lock.yaml").is_file()
            and (pr / "pnpm-lock.yaml").read_bytes() == pnpm.read_bytes()
            and (pr / "node_modules").is_dir()
        )
        if shares:
            add(
                "pnpm_share_primary", "rm -rf node_modules && ln -s ../../node_modules node_modules"
            )
        else:
            add("pnpm", "pnpm install --frozen-lockfile")
    elif (wt / "package-lock.json").is_file():
        add("npm", "npm ci")
    if (wt / "yarn.lock").is_file() and not pnpm.is_file():
        add(
            "yarn",
            (
                "yarn install --immutable"
                if (wt / ".yarnrc.yml").is_file()
                else "yarn install --frozen-lockfile"
            ),
        )
    if (wt / "uv.lock").is_file():
        add("uv", "uv venv && uv sync --frozen")
    else:
        reqs = sorted(p.name for p in wt.glob("requirements*.txt") if p.is_file())
        if reqs:
            pick = "requirements.txt" if "requirements.txt" in reqs else reqs[0]
            add("uv_pip", f"uv venv && uv pip install -r {pick}")
    if not out and ((wt / "Cargo.lock").is_file() or (wt / "go.sum").is_file()):
        add("none", "")
    return out


# --------------------------------------------------------------------- departure


def mark_departed(
    config: Path,
    session_id: str,
    to_device: str,
    worktrees: Sequence[SnapshotRefs],
    *,
    name: str = "",
    home: Path,
    now: float | None = None,
) -> list[dict[str, Any]]:
    """§8.6 at the source's commit: journal + marker, NEVER a delete.

    The journal ``<config>/network/workset-departed/<sid>.json`` is the TRUSTED
    record (it lives in this device's config root, not in the worktree a peer
    or tool could edit) of each tree's departure state: HEAD, index/worktree
    tree ids and excludes. A later recall updates a tree in place only when
    the journal says so AND the tree re-measures to that state; the marker
    ``<git-dir>/lop-moved.json`` alone proves nothing (security review). Returns
    the §3.4 ``source_marked`` rows.
    """
    _validate_session_id(session_id)
    at = now if now is not None else time.time()
    out_dir = _private_dir(Path(config) / "network" / "workset-departed")
    entries: list[dict[str, Any]] = []
    marked: list[dict[str, Any]] = []
    for snap in worktrees:
        rel = home_relative(snap.worktree_path, home=home) or snap.worktree_path
        git_dir = Path(_out(snap.worktree_path, "rev-parse", "--absolute-git-dir"))
        entries.append(
            {
                "path": rel,
                "n": snap.n,
                "head": snap.head,
                "branch": snap.branch,
                "index": snap.index,
                "worktree": snap.worktree,
                "index_tree": snap.index_tree,
                "worktree_tree": snap.worktree_tree,
                "excludes": list(snap.excludes),
                "refs": list(snap.refs),
            }
        )
        _write_json_atomic(
            git_dir / "lop-moved.json",
            {"to_device": to_device, "name": name, "at": at, "session_id": session_id},
        )
        marked.append({"path": rel, "at": at})
    _write_json_atomic(
        out_dir / f"{session_id}.json",
        {
            "version": 1,
            "session_id": session_id,
            "to_device": to_device,
            "name": name,
            "at": at,
            "worktrees": entries,
        },
    )
    return marked


__all__ = [
    "Artifact",
    "BundleHeader",
    "ContainmentError",
    "Excluded",
    "HistoryCheck",
    "HistoryWithheld",
    "MoveWorksetError",
    "PrimaryResolution",
    "Recipe",
    "RestorePlan",
    "RestoreReport",
    "SnapshotRefs",
    "clone_primary",
    "compute_have",
    "conflict_path",
    "contained_path",
    "history_check",
    "home_relative",
    "mark_departed",
    "normalize_remote_url",
    "read_bundle_header",
    "recipes_for",
    "remap",
    "restore",
    "snapshot",
    "snapshot_ref",
    "stage_bundle",
    "verify_bundle",
    "verify_or_replan",
]
