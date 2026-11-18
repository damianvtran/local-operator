"""The session's WORKING SET for a move: which git worktrees it worked in (manifest half).

WHY THIS EXISTS (operator, 2026-11-18): ``lop sessions move`` carries the session
directory — transcript, scratchpad, session memory, todos, wakes, linked projects
(``network/sync.py`` ``COPY_SET_NAMES``/``COPY_SET_TREES``, ``network/carry.py``) —
but NOT the work the session did outside it: git worktrees, their unpushed commits
and their staged/unstaged/untracked changes. A user who moved a coding session
mid-task arrived on a device where none of it existed, and nothing had told them
before the source let go. This module computes, READ-ONLY, what that work is, so
every move surface can show it and the owner can refuse a move whose working set
the user never confirmed (design note ``docs/design/move-working-set.md`` §3, §11).

SCOPE OF THIS MODULE: the manifest only — discovery, per-worktree facts, secret
classification, the scope digest. The transfer and restore halves (snapshot,
bundle, restore, mark-departed) live in ``network/workset_transfer.py`` and import
the frozen dataclasses below; keep their field names stable.

CONSTRAINTS:

- **Runs in the relay**, so stdlib + ``subprocess`` only at import time (the same
  import-weight reason as ``carry.py``); the denylist, secret guard and runtime
  registry are imported lazily inside the functions that need them.
- **Never writes.** Every git call runs with ``GIT_OPTIONAL_LOCKS=0`` so
  ``git status`` cannot refresh the index of a worktree another writer is using,
  and ``GIT_TERMINAL_PROMPT=0`` so a relay thread can never block on a credential
  prompt. Nothing here touches HEAD, the index or a working file.
- **Metadata only.** The manifest carries paths, branch names, counts and sizes —
  never file contents, never a secret value (only the denylist rule id or the
  credential CLASS), and never a full dirty-file list: only the withheld names are
  listed (security lead + manager, 2026-11-18: "minimum needed for consent").
- **Bounded.** Every input has a cap below (sessions in the tree, seeds, mentions,
  status entries, content-scan bytes, history-diff bytes); hitting a cap is
  recorded rather than silently truncating where it would change what is safe.
"""

from __future__ import annotations

import gzip
import hashlib
import json
import logging
import math
import os
import re
import subprocess
from collections.abc import Iterable, Mapping
from dataclasses import asdict, dataclass, field, replace
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)

#: The ``working_set`` block's schema version (design §3.4, ``lop://session_move/working_set/v1``).
WORKSET_VERSION = 1

#: Per-move payload cap (design §8.4): ~6x the measured max delta (43.1 MB, n=60
#: worktrees on the Linux node, 2026-11-18) and equal to the scratchpad cap already in
#: force. Over it the move is refused ``working_set_over_cap`` naming the sizes.
DEFAULT_CAP_BYTES = 256 * 1024 * 1024

#: The states a worktree row may carry (§3.4). PR A computes only the first and last;
#: the transfer half adds the middle three.
WORKTREE_STATES = ("planned", "carried", "restored", "pending_restore", "left_behind")

#: Why a worktree is ``left_behind`` (§3.4 plus the 2026-11-18 rulings). Machine codes
#: only; every surface renders the sentence from its own catalogue (design §12).
#: ``not_transferred`` is PR A's honest arrival state: the move was confirmed, but this
#: build does not carry worktrees yet, so each planned row ends up here on the receipt.
#: ``history_unscannable``: the outgoing history was too large (or unreadable) to scan, which
#: the security ruling treats exactly like a detected secret (no opt-in in v1).
LEFT_BEHIND_REASONS = (
    "contended",
    "outside_home",
    "unmerged",
    "in_progress_operation",
    "submodule",
    "lfs",
    "history_secret",
    "history_unscannable",
    "over_cap",
    "destination_conflict",
    "destination_cannot_clone",
    "unsupported_by_peer",
    "not_transferred",
    # Shared with the transfer half (workset_transfer.py), one spelling each: a path
    # that cannot be recreated safely on the destination, and a destination that
    # lacks the free space the restore needs.
    "unsafe_path",
    "insufficient_space",
)

#: Bounds. Each is a ceiling far above anything observed, so it only stops a
#: pathological input from stalling a relay thread; none is a tuning knob.
_MAX_TREE_SESSIONS = 256  # sessions in one subagent tree
_MAX_MENTIONS = 4000  # distinct path mentions read from transcripts
_MAX_TRANSCRIPT_BYTES = 64 * 1024 * 1024  # per transcript read for mentions
_MAX_STATUS_ENTRIES = 50_000  # porcelain entries per worktree
_CONTENT_SCAN_MAX_BYTES = 1024 * 1024  # §8.1: text files <= 1 MiB are content-scanned
_HISTORY_DIFF_MAX_BYTES = 16 * 1024 * 1024  # outgoing-history patch text scanned for secrets
_GIT_TIMEOUT_S = 30.0

#: An absolute or ``~/`` path inside text. Deliberately loose: a mention only becomes a
#: seed if it resolves to an existing directory inside a git work tree, so a false
#: match costs a few ``stat`` calls and nothing else.
_PATH_MENTION = re.compile(r"(?:(?<=[\s\"'`(=:])|^)((?:~|/)[A-Za-z0-9._@+\-/]{2,512})")


# ---------------------------------------------------------------------------
# Data shapes (frozen; the transfer half imports these)
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class Seed:
    """A path that says the session tree worked somewhere, and why we think so."""

    path: str
    source: str  # "cwd" | "provenance" | "mention", or "subagent:<child_id>" prefix form


@dataclass(frozen=True)
class Excluded:
    """A path withheld from the transfer. ``rule`` is a rule id or credential class,
    never the value."""

    path: str  # home-relative (~/...) when under home
    why: str  # secret_path | secret_content | history_secret_path | history_secret_content
    rule: str = ""

    def to_dict(self) -> dict[str, Any]:
        out: dict[str, Any] = {"path": self.path, "why": self.why}
        if self.rule:
            out["rule"] = self.rule
        return out


@dataclass(frozen=True)
class Recipe:
    """How to recreate a dependency tree on the destination (§9). Listed, never run."""

    kind: str
    run: str
    cwd: str

    def to_dict(self) -> dict[str, Any]:
        return {"kind": self.kind, "run": self.run, "cwd": self.cwd}


@dataclass(frozen=True)
class LeftBehind:
    """A non-git item that stays on the source (dependency tree, build output)."""

    path: str
    kind: str  # dependency_tree | build_output | ignored | loose_file
    recipe: Recipe | None = None

    def to_dict(self) -> dict[str, Any]:
        out: dict[str, Any] = {"path": self.path, "kind": self.kind}
        if self.recipe is not None:
            out["recipe"] = self.recipe.to_dict()
        return out


@dataclass(frozen=True)
class WorktreeFacts:
    """Everything the manifest says about one worktree. All read-only git facts.

    ``path``/``repo`` are home-relative (``~/...``) when under home — the form both
    devices understand (§7) — and ``abs_path``/``abs_repo`` keep the source's own
    absolute form for the transfer half, which runs on this device. The ``abs_*``
    fields never leave the owner (``to_dict`` omits them).
    """

    path: str
    repo: str
    abs_path: str
    abs_repo: str
    head: str = ""
    branch: str = ""
    upstream: str = ""
    remote_url: str = ""
    base: str = ""
    unpushed_commits: int = 0
    staged: int = 0
    unstaged: int = 0
    untracked: int = 0
    dirty_bytes: int = 0
    bundle_bytes: int = 0
    is_primary: bool = False
    sources: tuple[str, ...] = ()
    state: str = "planned"
    reason: str = ""
    contended_by: tuple[str, ...] = ()
    excluded: tuple[Excluded, ...] = ()
    left_behind: tuple[LeftBehind, ...] = ()

    @property
    def dirty_files(self) -> int:
        return self.staged + self.unstaged + self.untracked

    @property
    def bytes(self) -> int:
        return self.dirty_bytes + self.bundle_bytes

    def to_dict(self) -> dict[str, Any]:
        out: dict[str, Any] = {
            "path": self.path,
            "repo": self.repo,
            "head": self.head,
            "state": self.state,
            "unpushed_commits": self.unpushed_commits,
            "staged": self.staged,
            "unstaged": self.unstaged,
            "untracked": self.untracked,
            "bundle_bytes": self.bundle_bytes,
            "snapshot_bytes": self.dirty_bytes,
            "sources": list(self.sources),
            # Coordinator ruling (2026-11-18): whether the DESTINATION can clone the
            # primary with its own credential is the destination's answer, which the
            # owner cannot know at preflight. PR A states that honestly; the transfer
            # half's ``workset_plan`` replaces it with "ok" or turns the row into
            # ``left_behind``/``destination_cannot_clone`` before the confirmation.
            "clone_check": "pending_destination",
        }
        for key in ("branch", "upstream", "base", "remote_url"):
            value = getattr(self, key)
            if value:
                out[key] = value
        if self.is_primary:
            out["primary"] = True
        if self.reason:
            out["reason"] = self.reason
        if self.contended_by:
            out["by"] = list(self.contended_by)
        return out


@dataclass(frozen=True)
class Manifest:
    """The ``working_set`` block (§3.4) and the inputs of its scope digest (§3.5)."""

    to_device: str
    to_name: str
    worktrees: tuple[WorktreeFacts, ...] = ()
    cap_bytes: int = DEFAULT_CAP_BYTES
    state: str = "preflight"
    #: Whether the move that follows this manifest TRANSFERS the planned worktrees.
    #: ``False`` in PR A (the manifest is shown and confirmed, the work stays on the
    #: source and the receipt says so); the transfer half flips it. Surfaces branch on
    #: it for their sentence: "sends its working files there" vs "they stay here".
    carries_files: bool = False
    #: The OWNER (the device that keeps the files when they do not travel). Named on
    #: the block because a recall's surfaces run on the DESTINATION, where "this
    #: device" would name the wrong machine. Not a digest input.
    from_device: str = ""
    from_name: str = ""

    @property
    def empty(self) -> bool:
        """No worktree at all — the move needs no confirmation and the receipt carries
        no ``working_set`` key (frozen-contract parity for chat-only sessions)."""
        return not self.worktrees

    @property
    def planned(self) -> tuple[WorktreeFacts, ...]:
        return tuple(w for w in self.worktrees if w.state == "planned")

    @property
    def excluded(self) -> tuple[Excluded, ...]:
        return tuple(item for w in self.worktrees for item in w.excluded)

    @property
    def left_behind(self) -> tuple[LeftBehind, ...]:
        return tuple(item for w in self.worktrees for item in w.left_behind)

    @property
    def total_bytes(self) -> int:
        return sum(w.bytes for w in self.planned)

    @property
    def over_cap(self) -> bool:
        return self.total_bytes > self.cap_bytes

    def to_dict(self) -> dict[str, Any]:
        planned = self.planned
        return {
            "version": WORKSET_VERSION,
            "scope_digest": scope_digest(self),
            "state": self.state,
            "carries_files": self.carries_files,
            "destination": {"device_id": self.to_device, "name": self.to_name},
            "source": {"device_id": self.from_device, "name": self.from_name},
            "totals": {
                "worktrees": len(planned),
                "unpushed_commits": sum(w.unpushed_commits for w in planned),
                "dirty_files": sum(w.dirty_files for w in planned),
                "bytes": self.total_bytes,
                "cap_bytes": self.cap_bytes,
            },
            "worktrees": [w.to_dict() for w in self.worktrees],
            "excluded": [item.to_dict() for item in self.excluded],
            "left_behind": [item.to_dict() for item in self.left_behind],
        }


# ---------------------------------------------------------------------------
# The scope digest (§3.5) — what "confirmed" means
# ---------------------------------------------------------------------------


def scope_digest(manifest: Manifest) -> str:
    """``sha256:<hex>`` over WHICH trees go to WHICH device and what is withheld.

    Not a digest of the full manifest, deliberately (§3.5): a ``--queue``d session keeps
    editing until its turn ends, so byte-exact consent would refuse every queued move.
    The user confirms the SCOPE — a new worktree, a new withheld path, a different state
    or reason, a different destination device id, or the payload crossing a power-of-two
    bucket changes it, and the owner refuses ``working_set_changed`` rather than widening
    the move silently.

    BOUND TO THE AUTHENTICATED DESTINATION (security lead, 2026-11-18): ``to_device`` is
    the mesh device id the owner resolved/authenticated, never a display name, so a
    digest confirmed for one device cannot authorise a move to another. The lease
    epoch is NOT an input: it is minted by the prepare that the consent precedes, so
    it cannot exist when the user confirms. SEAM FOR THE TRANSFER HALF: it binds the
    staged, immutable artifact digests to the journal's epoch at prepare (P4) — extend
    ``_artifact_binding`` below rather than changing the inputs above, so a PR A
    consent stays comparable.
    """
    total = manifest.total_bytes
    payload = {
        "version": WORKSET_VERSION,
        "to_device": manifest.to_device,
        "worktrees": sorted(
            [w.path, w.repo, w.branch, w.state, w.reason] for w in manifest.worktrees
        ),
        "excluded": sorted([item.path, item.why] for item in manifest.excluded),
        "bytes_bucket": math.ceil(math.log2(max(total, 1))),
        **_artifact_binding(manifest),
    }
    canonical = json.dumps(payload, sort_keys=True, separators=(",", ":"), ensure_ascii=True)
    return "sha256:" + hashlib.sha256(canonical.encode("utf-8")).hexdigest()


def _artifact_binding(manifest: Manifest) -> dict[str, Any]:
    """Extra digest inputs the transfer half contributes. Empty in PR A by design."""
    del manifest
    return {}


# ---------------------------------------------------------------------------
# Discovery (§3.2)
# ---------------------------------------------------------------------------


def _read_json(path: Path) -> Any:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return None


def session_tree(root: Path, session_id: str) -> list[str]:
    """The session plus every subagent child it launched, recursively, in BFS order.

    Children are separate session directories; the parent's lane/stop receipts
    (``subagent-lane-<job>.v1.json`` / ``subagent-stop-<job>.v1.json``,
    ``session/subagent_ledger.py``) name each ``child_session_id``. A child whose
    directory is absent on this device (already moved, deleted) is skipped — it has
    no working set to report here.
    """
    sessions = Path(root) / "sessions"
    seen: list[str] = [session_id]
    queue = [session_id]
    while queue and len(seen) < _MAX_TREE_SESSIONS:
        current = queue.pop(0)
        directory = sessions / current
        if not directory.is_dir():
            continue
        try:
            names = sorted(os.listdir(directory))
        except OSError:
            continue
        for name in names:
            if not (name.startswith("subagent-lane-") or name.startswith("subagent-stop-")):
                continue
            receipt = _read_json(directory / name)
            child = str(receipt.get("child_session_id") or "") if isinstance(receipt, dict) else ""
            if child and child not in seen and re.fullmatch(r"[A-Za-z0-9_-]{1,128}", child):
                if (sessions / child).is_dir():
                    seen.append(child)
                    queue.append(child)
    return seen


def _transcript_texts(session_dir: Path) -> Iterable[str]:
    """The transcript forms readable without the session runtime: plain and ``.gz``.

    Other archive forms are skipped rather than decoded here — a mention is a fallback
    seed (cwd and provenance are the primary ones), and the restore path that reads
    every form belongs to ``sync``/the runtime, not to a read-only relay probe.
    """
    plain = session_dir / "transcript.jsonl"
    try:
        with plain.open("rb") as handle:
            yield handle.read(_MAX_TRANSCRIPT_BYTES).decode("utf-8", "replace")
    except OSError:
        pass
    archived = session_dir / "transcript.jsonl.gz"
    try:
        with gzip.open(archived, "rb") as handle:
            yield handle.read(_MAX_TRANSCRIPT_BYTES).decode("utf-8", "replace")
    except (OSError, EOFError):
        pass


def candidate_seeds(
    root: Path,
    session_ids: list[str],
    *,
    cwd_hint: str = "",
    home: Path | None = None,
) -> list[Seed]:
    """Path seeds for every session in the tree (§3.2 step 2).

    Three sources, in authority order: the recorded cwd (``cwd_hint`` for the root
    session — the owner's ``_source_session_cwd``, which reads the wake index and the
    runtime record before the marker — else ``carry.resolve_cwd``); file-provenance
    records (``.file-provenance/*.json`` ``file.path``, PR #4108; metadata only); and
    literal absolute / ``~/`` path mentions in the transcript, which covers sessions
    older than provenance and ``cd``/``git -C`` usage it cannot see. The session
    directory itself is never a seed (it is under the config root, not a worktree).
    """
    from local_operator.network import carry

    home_path = Path(home) if home is not None else Path.home()
    sessions = Path(root) / "sessions"
    seeds: list[Seed] = []
    mentions_seen: set[str] = set()
    for index, sid in enumerate(session_ids):
        tag = "" if index == 0 else f"subagent:{sid}"
        directory = sessions / sid
        cwd = cwd_hint if (index == 0 and cwd_hint) else carry.resolve_cwd(directory)
        if cwd and os.path.realpath(cwd) != os.path.realpath(directory):
            seeds.append(Seed(cwd, tag or "cwd"))
        provenance = directory / ".file-provenance"
        try:
            records = sorted(provenance.glob("*.json"))
        except OSError:
            records = []
        for record in records[:_MAX_MENTIONS]:
            proof = _read_json(record)
            frozen = proof.get("file") if isinstance(proof, dict) else None
            path = frozen.get("path") if isinstance(frozen, dict) else None
            if isinstance(path, str) and path:
                seeds.append(Seed(path, tag or "provenance"))
        for text in _transcript_texts(directory):
            for match in _PATH_MENTION.finditer(text):
                raw = match.group(1).rstrip(".,;:)\"'`/\\")
                if raw.startswith("~"):
                    raw = str(home_path) + raw[1:]
                if raw in mentions_seen or not raw.startswith(str(home_path) + os.sep):
                    continue
                mentions_seen.add(raw)
                seeds.append(Seed(raw, tag or "mention"))
                if len(mentions_seen) >= _MAX_MENTIONS:
                    break
    return seeds


def _git_env() -> dict[str, str]:
    env = dict(os.environ)
    # Read-only probes: never take the index lock (``status`` would otherwise refresh
    # the index of a tree another writer is using) and never prompt.
    env["GIT_OPTIONAL_LOCKS"] = "0"
    env["GIT_TERMINAL_PROMPT"] = "0"
    env.pop("GIT_DIR", None)
    env.pop("GIT_WORK_TREE", None)
    env.pop("GIT_INDEX_FILE", None)
    return env


class WorksetUnknown(Exception):
    """A fact the consent depends on could not be read (manager ruling, 2026-11-18).

    FAIL CLOSED: an unreadable ``git status`` or ``worktree list`` is exactly the
    silent-loss case this module exists to stop, so it is never read as "clean". The
    owner refuses the move ``working_set_unknown`` naming ``step`` and ``repo``; the
    explicit escape is ``--without-working-set``.
    """

    def __init__(self, step: str, repo: str = "", detail: str = "") -> None:
        super().__init__(f"{step} failed in {repo or '?'}: {detail}")
        self.step = step
        self.repo = repo
        self.detail = detail


def _git(
    cwd: Path | str, *args: str, binary: bool = False, limit: int = 0, required: str = ""
) -> Any:
    """Run one read-only git command; ``None`` on failure (absent ref, not a repo).

    ``required`` names the step when the answer is load-bearing for consent: a failure
    then raises :class:`WorksetUnknown` instead of reading as "nothing there".

    ``limit`` STREAMS: at most ``limit`` bytes are read and the process is killed past
    it, so an enormous ``git log -p`` cannot be buffered whole in the relay (the
    caller sees ``len(out) == limit`` and treats the range as unscannable).
    """
    command = ["git", "-C", str(cwd), *args]
    try:
        if limit:
            with subprocess.Popen(
                command, stdout=subprocess.PIPE, stderr=subprocess.DEVNULL, env=_git_env()
            ) as process:
                assert process.stdout is not None
                out = process.stdout.read(limit)
                truncated = len(out) >= limit
                if truncated:
                    process.kill()
                code = process.wait(timeout=_GIT_TIMEOUT_S)
            if truncated:
                return out if binary else out.decode("utf-8", "replace")
        else:
            completed = subprocess.run(
                command,
                capture_output=True,
                env=_git_env(),
                timeout=_GIT_TIMEOUT_S,
                check=False,
            )
            out, code = completed.stdout, completed.returncode
    except (OSError, subprocess.SubprocessError) as exc:
        if required:
            raise WorksetUnknown(required, str(cwd), type(exc).__name__) from exc
        return None
    if code != 0:
        if required:
            raise WorksetUnknown(required, str(cwd), f"git exited {code}")
        return None
    return out if binary else out.decode("utf-8", "replace").strip()


def _nearest_existing_dir(path: str) -> Path | None:
    current = Path(os.path.normpath(path))
    for _ in range(64):
        try:
            if current.is_dir():
                return current
        except OSError:
            return None
        if current.parent == current:
            return None
        current = current.parent
    return None


def _work_tree_top(directory: Path) -> Path | None:
    """The work-tree root containing ``directory`` by walking up for ``.git`` (a stat
    walk, no subprocess — mentions can be thousands of paths)."""
    current = directory
    for _ in range(64):
        if (current / ".git").exists():
            return current
        if current.parent == current:
            return None
        current = current.parent
    return None


@dataclass(frozen=True)
class Candidate:
    """A worktree this session tree touched, before facts are read."""

    path: Path
    common_dir: Path
    primary: Path
    is_primary: bool
    sources: tuple[str, ...] = field(default_factory=tuple)


def _worktree_list(top: Path) -> list[Path]:
    # REQUIRED (fail closed): a directory with a ``.git`` that git cannot list is a
    # repository whose worktrees this session may have used; "none" would be a guess.
    listing = _git(top, "worktree", "list", "--porcelain", required="worktree_list")
    if not isinstance(listing, str):
        return [top]
    paths = []
    for line in listing.splitlines():
        if line.startswith("worktree "):
            paths.append(Path(line[len("worktree ") :]))
    return paths or [top]


def _worktree_list_quiet(top: Path) -> list[Path]:
    """``_worktree_list`` for a DESCRIPTIVE use (nested-worktree filtering): a failure
    here only means nothing is filtered, so it must not refuse the move."""
    try:
        return _worktree_list(top)
    except WorksetUnknown:
        return []


def discover_worktrees(seeds: Iterable[Seed]) -> list[Candidate]:
    """Resolve seeds to the worktrees that contain them (§3.2 steps 3–5).

    A seed is attributed to the DEEPEST registered worktree containing it, so a path
    inside ``<repo>/.worktrees/x`` attributes to ``x`` and never to the primary checkout
    that physically contains it; a sibling worktree of the same repository nobody
    touched is never included. Every kept worktree records the sources that kept it.
    """
    tops: dict[Path, list[str]] = {}
    for seed in seeds:
        directory = _nearest_existing_dir(seed.path)
        if directory is None:
            continue
        top = _work_tree_top(directory)
        if top is None:
            continue
        tops.setdefault(top, [])
        tops[top].append(f"{seed.source}\0{directory}")
    # Group by repository, read each repository's worktree list once.
    by_repo: dict[Path, tuple[Path, list[Path]]] = {}
    for top in tops:
        common = _git(top, "rev-parse", "--path-format=absolute", "--git-common-dir")
        if not isinstance(common, str) or not common:
            continue
        common_dir = Path(common)
        if common_dir not in by_repo:
            worktrees = _worktree_list(top)
            primary = worktrees[0] if worktrees else top
            by_repo[common_dir] = (primary, worktrees)
    found: dict[Path, Candidate] = {}
    for top, tagged in tops.items():
        common = _git(top, "rev-parse", "--path-format=absolute", "--git-common-dir")
        if not isinstance(common, str) or Path(common) not in by_repo:
            continue
        primary, worktrees = by_repo[Path(common)]
        for item in tagged:
            source, directory = item.split("\0", 1)
            target = _deepest_containing(Path(directory), worktrees)
            if target is None:
                continue
            existing = found.get(target)
            sources = tuple(dict.fromkeys((*(existing.sources if existing else ()), source)))
            found[target] = Candidate(
                path=target,
                common_dir=Path(common),
                primary=primary,
                is_primary=(target == primary),
                sources=sources,
            )
    return sorted(found.values(), key=lambda c: str(c.path))


def _deepest_containing(directory: Path, worktrees: list[Path]) -> Path | None:
    best: Path | None = None
    resolved = os.path.realpath(directory)
    for worktree in worktrees:
        real = os.path.realpath(worktree)
        if resolved == real or resolved.startswith(real.rstrip(os.sep) + os.sep):
            if best is None or len(real) > len(os.path.realpath(best)):
                best = worktree
    return best


# ---------------------------------------------------------------------------
# Per-worktree facts (§3.3)
# ---------------------------------------------------------------------------


def home_relative(path: Path | str, home: Path) -> str:
    """``~/<tail>`` for a path under ``home``, else the absolute path unchanged."""
    text = os.path.normpath(str(path))
    base = os.path.normpath(str(home))
    if text == base:
        return "~"
    if text.startswith(base + os.sep):
        return "~/" + text[len(base) + 1 :]
    return text


def _strip_userinfo(url: str) -> str:
    """Drop ``user:token@`` from an https remote: a URL that embeds a credential must
    not reach a manifest the destination and the user's screen both see."""
    return re.sub(r"^([a-z][a-z0-9+.-]*://)[^/@]*@", r"\1", url, flags=re.IGNORECASE)


def _parse_status(raw: bytes) -> tuple[list[str], list[str], list[str], bool]:
    """``git status --porcelain=v2 -z`` → (staged, unstaged, untracked, unmerged)."""
    staged: list[str] = []
    unstaged: list[str] = []
    untracked: list[str] = []
    unmerged = False
    parts = raw.split(b"\0")
    index = 0
    while index < len(parts) and index < _MAX_STATUS_ENTRIES:
        entry = parts[index].decode("utf-8", "surrogateescape")
        index += 1
        if not entry:
            continue
        kind = entry[0]
        if kind == "1":
            fields = entry.split(" ", 8)
            xy, path = fields[1], fields[8]
        elif kind == "2":
            fields = entry.split(" ", 9)
            xy, path = fields[1], fields[9]
            index += 1  # the original path follows as its own NUL-separated field
        elif kind == "u":
            unmerged = True
            continue
        elif kind == "?":
            untracked.append(entry[2:])
            continue
        else:
            continue
        if xy[0] != ".":
            staged.append(path)
        if xy[1] != ".":
            unstaged.append(path)
    return staged, unstaged, untracked, unmerged


def _in_progress_operation(git_dir: Path) -> bool:
    return any(
        (git_dir / name).exists()
        for name in (
            "MERGE_HEAD",
            "CHERRY_PICK_HEAD",
            "REVERT_HEAD",
            "rebase-merge",
            "rebase-apply",
            "BISECT_LOG",
        )
    )


def classify_exclusions(worktree: Path, entries: Iterable[str], *, home: Path) -> list[Excluded]:
    """Secret-shaped dirty/untracked paths, withheld from the transfer (§8.1).

    Path classifier first (``supplements/denylist.is_sensitive`` — the one the peer
    file-read and upload gates use), then content (``team_bridge/secret_guard
    .credential_classes``) for text files up to 1 MiB. A hit names the rule id or the
    credential class, never the value; the file's content never leaves this function.
    """
    from local_operator.supplements.denylist import is_sensitive
    from local_operator.team_bridge.secret_guard import credential_classes

    out: list[Excluded] = []
    for relative in dict.fromkeys(entries):
        absolute = worktree / relative
        shown = home_relative(absolute, home)
        rule = is_sensitive(str(absolute), cwd=str(worktree))
        if rule:
            out.append(Excluded(shown, "secret_path", rule))
            continue
        try:
            if not absolute.is_file() or absolute.is_symlink():
                continue
            if absolute.stat().st_size > _CONTENT_SCAN_MAX_BYTES:
                continue
            data = absolute.read_bytes()
        except OSError:
            continue
        if b"\0" in data[:8192]:
            continue  # binary: the credential table is text-shaped
        classes = credential_classes(data.decode("utf-8", "replace"))
        if classes:
            out.append(Excluded(shown, "secret_content", classes[0]))
    return out


def _history_exclusions(worktree: Path, *, home: Path) -> tuple[list[Excluded], bool]:
    """Secret-shaped paths or added lines in the OUTGOING history (§8.1, ruled
    2026-11-18: no opt-in in v1). Returns (findings, scanned).

    A bundle carries every blob of the commits it contains, so history cannot be
    path-filtered: any hit leaves the whole worktree behind. The range is ``HEAD --not
    --remotes`` — what no remote-tracking ref has — which is the owner-side stand-in for
    the transfer half's ``<tips> --not <have>``. ``scanned`` is False when the patch
    text exceeded the bound: unscannable is treated exactly like a detected secret.
    """
    from local_operator.supplements.denylist import is_sensitive
    from local_operator.team_bridge.secret_guard import credential_classes

    names = _git(worktree, "log", "--format=", "--name-only", "HEAD", "--not", "--remotes")
    if names is None:
        return [], False  # unverifiable is unscannable (security ruling): never "clean"
    found: list[Excluded] = []
    for name in dict.fromkeys((names or "").splitlines() if isinstance(names, str) else []):
        if not name:
            continue
        rule = is_sensitive(str(worktree / name), cwd=str(worktree))
        if rule:
            found.append(
                Excluded(home_relative(worktree / name, home), "history_secret_path", rule)
            )
    patch = _git(
        worktree,
        "log",
        "-p",
        "--format=",
        "--no-color",
        "--no-ext-diff",
        "HEAD",
        "--not",
        "--remotes",
        binary=True,
        limit=_HISTORY_DIFF_MAX_BYTES + 1,
    )
    if not isinstance(patch, bytes) or len(patch) > _HISTORY_DIFF_MAX_BYTES:
        return found, False
    added = "\n".join(
        line[1:]
        for line in (patch or b"").decode("utf-8", "replace").splitlines()
        if line.startswith("+") and not line.startswith("+++")
    )
    classes = credential_classes(added) if added else []
    if classes:
        found.append(Excluded(home_relative(worktree, home), "history_secret_content", classes[0]))
    return found, True


def dependency_left_behind(worktree: Path, primary: Path, *, home: Path) -> list[LeftBehind]:
    """Dependency trees present in ``worktree`` that never travel, with the §9 recipe."""
    shown_cwd = home_relative(worktree, home)
    out: list[LeftBehind] = []
    node_modules = worktree / "node_modules"
    if node_modules.exists() or node_modules.is_symlink():
        recipe: Recipe | None = None
        if (worktree / "pnpm-lock.yaml").is_file():
            shares = (
                worktree != primary
                and worktree.parent == primary / ".worktrees"
                and (primary / "node_modules").is_dir()
                and _same_bytes(worktree / "pnpm-lock.yaml", primary / "pnpm-lock.yaml")
            )
            recipe = (
                Recipe(
                    "pnpm_share_primary",
                    "rm -rf node_modules && ln -s ../../node_modules node_modules",
                    shown_cwd,
                )
                if shares
                else Recipe("pnpm", "pnpm install --frozen-lockfile", shown_cwd)
            )
        elif (worktree / "package-lock.json").is_file():
            recipe = Recipe("npm", "npm ci", shown_cwd)
        elif (worktree / "yarn.lock").is_file():
            run = (
                "yarn install --immutable"
                if (worktree / ".yarnrc.yml").is_file()
                else "yarn install --frozen-lockfile"
            )
            recipe = Recipe("yarn", run, shown_cwd)
        out.append(LeftBehind(home_relative(node_modules, home), "dependency_tree", recipe))
    venv = worktree / ".venv"
    if venv.exists():
        recipe = None
        if (worktree / "uv.lock").is_file():
            recipe = Recipe("uv", "uv venv && uv sync --frozen", shown_cwd)
        else:
            requirements = sorted(worktree.glob("requirements*.txt"))
            if requirements:
                recipe = Recipe(
                    "uv_pip", f"uv venv && uv pip install -r {requirements[0].name}", shown_cwd
                )
        out.append(LeftBehind(home_relative(venv, home), "dependency_tree", recipe))
    return out


def _same_bytes(a: Path, b: Path) -> bool:
    try:
        return a.read_bytes() == b.read_bytes()
    except OSError:
        return False


def worktree_facts(
    candidate: Candidate | Path,
    *,
    home: Path,
    primary: Path | None = None,
    sources: tuple[str, ...] = (),
) -> WorktreeFacts:
    """Read one worktree's git facts (§3.3), its exclusions and what stays behind.

    Sets ``state: left_behind`` with a machine reason for the shapes the transfer
    cannot carry faithfully (outside home, unmerged/in-progress, submodules, LFS,
    secret-shaped or unscannable outgoing history). Descriptive facts (upstream, base,
    remote) degrade to empty; the load-bearing ones (``status``, the unpushed count)
    raise :class:`WorksetUnknown` rather than read as clean.
    """
    if isinstance(candidate, Candidate):
        path, primary, sources, is_primary = (
            candidate.path,
            candidate.primary,
            candidate.sources,
            candidate.is_primary,
        )
    else:
        path = Path(candidate)
        primary = Path(primary) if primary is not None else path
        is_primary = path == primary
    head = _git(path, "rev-parse", "HEAD") or ""
    branch = _git(path, "symbolic-ref", "-q", "--short", "HEAD") or ""
    upstream = _git(path, "rev-parse", "--abbrev-ref", "--symbolic-full-name", "@{upstream}") or ""
    remote = upstream.split("/", 1)[0] if upstream else "origin"
    remote_url = _strip_userinfo(_git(path, "remote", "get-url", remote) or "")
    base_ref = upstream or "origin/HEAD"
    base = _git(path, "merge-base", "HEAD", base_ref) or ""
    # An unborn branch has no HEAD and nothing unpushed; otherwise the count is
    # load-bearing (it decides whether history is scanned) and must not read as 0.
    unpushed_raw = (
        _git(path, "rev-list", "--count", "HEAD", "--not", "--remotes", required="unpushed")
        if head
        else "0"
    )
    unpushed = int(unpushed_raw) if isinstance(unpushed_raw, str) and unpushed_raw.isdigit() else 0
    bundle_raw = (
        _git(path, "rev-list", "--objects", "--disk-usage", "HEAD", "--not", "--remotes")
        if unpushed
        else "0"
    )
    bundle_bytes = int(bundle_raw) if isinstance(bundle_raw, str) and bundle_raw.isdigit() else 0
    status = _git(
        path,
        "status",
        "--porcelain=v2",
        "-z",
        "--untracked-files=all",
        binary=True,
        required="status",
    )
    staged, unstaged, untracked, unmerged = _parse_status(
        status if isinstance(status, bytes) else b""
    )
    # NESTED REGISTERED WORKTREES ARE NOT THIS TREE'S DIRTY STATE. A primary whose
    # ``.worktrees/`` is not gitignored reports each nested worktree as an untracked
    # directory (an embedded repository); snapshotting that would carry it as a gitlink
    # and double-count it. Each nested worktree is its own manifest row (or not part
    # of this session at all), so it is dropped from this tree's set.
    nested = [
        os.path.relpath(other, path)
        for other in _worktree_list_quiet(path)
        if other != path and str(other).startswith(str(path).rstrip(os.sep) + os.sep)
    ]
    if nested:
        untracked = [
            entry
            for entry in untracked
            if not any(
                entry.rstrip("/") == prefix or entry.startswith(prefix + "/") for prefix in nested
            )
        ]
    dirty_paths = list(dict.fromkeys([*staged, *unstaged, *untracked]))
    dirty_bytes = 0
    for relative in dirty_paths:
        try:
            dirty_bytes += (path / relative).lstat().st_size
        except OSError:
            pass  # a deletion: nothing to carry but the fact
    excluded = classify_exclusions(path, dirty_paths, home=home)
    git_dir_raw = _git(path, "rev-parse", "--path-format=absolute", "--git-dir")
    git_dir = Path(git_dir_raw) if isinstance(git_dir_raw, str) and git_dir_raw else path / ".git"
    reason = ""
    home_text = os.path.normpath(str(home))
    if not os.path.normpath(str(path)).startswith(home_text + os.sep):
        reason = "outside_home"  # §7: home-relative is the only path both devices share
    elif isinstance(status, bytes) and status.count(b"\0") > _MAX_STATUS_ENTRIES:
        # The parse stopped at the bound, so the counts and the secret scan cover only
        # part of the tree. Never present a partial set as the whole: no move can carry
        # tens of thousands of dirty paths under the cap anyway.
        reason = "over_cap"
    elif unmerged:
        reason = "unmerged"
    elif _in_progress_operation(git_dir):
        reason = "in_progress_operation"
    elif (path / ".gitmodules").is_file():
        reason = "submodule"
    elif _uses_lfs(path):
        reason = "lfs"
    if not reason and unpushed:
        history, scanned = _history_exclusions(path, home=home)
        excluded = [*excluded, *history]
        if not scanned:
            reason = "history_unscannable"
        elif history:
            reason = "history_secret"
    return WorktreeFacts(
        path=home_relative(path, home),
        repo=home_relative(primary, home),
        abs_path=str(path),
        abs_repo=str(primary),
        head=head,
        branch=branch,
        upstream=upstream,
        remote_url=remote_url,
        base=base,
        unpushed_commits=unpushed,
        staged=len(staged),
        unstaged=len(unstaged),
        untracked=len(untracked),
        dirty_bytes=dirty_bytes,
        bundle_bytes=bundle_bytes,
        is_primary=is_primary,
        sources=tuple(sources),
        state="left_behind" if reason else "planned",
        reason=reason,
        excluded=tuple(excluded),
        left_behind=tuple(dependency_left_behind(path, primary, home=home)),
    )


def _uses_lfs(path: Path) -> bool:
    try:
        return "filter=lfs" in (path / ".gitattributes").read_text(
            encoding="utf-8", errors="replace"
        )
    except OSError:
        return False


# ---------------------------------------------------------------------------
# Contention and the manifest
# ---------------------------------------------------------------------------


def _live_cwds(root: Path, exclude: set[str]) -> list[tuple[str, str]]:
    """(session_id, cwd) of every LIVE runtime outside the moving tree (§3.2)."""
    try:
        from local_operator.session.runtime import registry

        scanned = registry.scan(Path(root), reap=False)
    except Exception:  # noqa: BLE001 — no registry reads as "no other writer known"
        logger.debug("workset: runtime registry unreadable", exc_info=True)
        return []
    out = []
    for record, state in scanned:
        sid = str(getattr(record, "session_id", "") or "")
        cwd = str(getattr(record, "cwd", "") or "")
        if state == "live" and sid and cwd and sid not in exclude:
            out.append((sid, cwd))
    return out


def compute_manifest(
    root: Path,
    session_id: str,
    *,
    to_device: str,
    to_name: str = "",
    home: Path | None = None,
    cap: int = DEFAULT_CAP_BYTES,
    cwd_hint: str = "",
    from_device: str = "",
    from_name: str = "",
) -> Manifest:
    """The owner's read-only manifest for moving ``session_id`` to ``to_device``.

    ``to_device`` must be the AUTHENTICATED mesh device id of the destination (the
    prepare's link peer, or the id the owner resolved for an offload) — it is a digest
    input. Called at preflight (runtime still serving) and again at prepare after the
    retire (P3), so it must be cheap and deterministic for an unchanged tree.
    """
    home_path = Path(home) if home is not None else Path.home()
    tree = session_tree(Path(root), session_id)
    seeds = candidate_seeds(Path(root), tree, cwd_hint=cwd_hint, home=home_path)
    candidates = discover_worktrees(seeds)
    live = _live_cwds(Path(root), set(tree))
    facts: list[WorktreeFacts] = []
    for candidate in candidates:
        fact = worktree_facts(candidate, home=home_path)
        real = os.path.realpath(candidate.path)
        contenders = tuple(
            sorted(
                sid
                for sid, cwd in live
                if os.path.realpath(cwd) == real
                or os.path.realpath(cwd).startswith(real.rstrip(os.sep) + os.sep)
            )
        )
        if contenders and fact.state == "planned":
            # Another live session's cwd is in this tree: handing it to a second device
            # would give a tree another writer is using to two machines.
            fact = replace(fact, state="left_behind", reason="contended", contended_by=contenders)
        facts.append(fact)
    return Manifest(
        to_device=to_device,
        to_name=to_name,
        worktrees=tuple(facts),
        cap_bytes=int(cap),
        from_device=from_device,
        from_name=from_name,
    )


def arrival_block(manifest: Manifest) -> dict[str, Any]:
    """The receipt's ``working_set`` block for a move that did NOT transfer the files.

    PR A's honest arrival (brief, 2026-11-18): the user confirmed the manifest, the
    session moved, and every planned worktree stayed on the source — so each planned
    row is reported ``left_behind``/``not_transferred`` and the block says
    ``transferred: false``. ``scope_digest`` is the CONFIRMED scope (computed before
    the rewrite), so a surface can match the receipt to the dialog it showed.
    """
    block = manifest.to_dict()
    block["state"] = "left_behind"
    block["transferred"] = False
    for row in block["worktrees"]:
        if row.get("state") == "planned":
            row["state"] = "left_behind"
            row["reason"] = "not_transferred"
    return block


def facts_to_json(facts: WorktreeFacts) -> dict[str, Any]:
    """Every field including the owner-local ``abs_*`` ones (for the transfer half's
    journal, which stays on this device). Never put this on the wire."""
    return asdict(facts)


def manifest_summary(block: Mapping[str, Any]) -> dict[str, int]:
    """Counts a log line or audit row may carry (no paths)."""
    raw = block.get("totals")
    totals: dict[str, Any] = raw if isinstance(raw, dict) else {}
    return {
        "worktrees": len(block.get("worktrees") or []),
        "planned": int(totals.get("worktrees") or 0),
        "bytes": int(totals.get("bytes") or 0),
        "excluded": len(block.get("excluded") or []),
    }
