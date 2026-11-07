"""Rank what a shell command would destroy: LOW (run), HIGH (judge), CRITICAL (refuse).

WHY. The 2026-11-04 incident was an empty-substitution delete of the state root:
a recursive delete whose path collapsed onto ``~/.local-operator`` because a
command substitution printed nothing. The operator's direction was explicit —
NOT a deterministic rule that blocks every delete, but a check that recognises
*dangerous* deletions and sends them for review. So the ranking is narrow:

* CRITICAL — the target is, or is an ancestor of, something whose loss is
  unrecoverable for this harness: the state root (``paths.config_dir()``),
  ``config.yml``, anything in ``secrets/``, the install layout
  (``update.stable_root()`` / generations), ``$HOME``, ``/``. Refused with no
  model call and no auto/yolo bypass; the incident shape lands here.
* HIGH — plausibly intended but expensive to get wrong: another session's
  directory or anything in the state root outside this session's own
  directory, a direct child of ``$HOME``, a git repository root, ``git clean``
  of untracked/ignored work, and every target the resolver could not pin down
  (opaque substitution, empty expansion, unknown cwd). Sent to the forked judge.
* LOW — everything else, including the session's own scratchpad, deep paths,
  ``/tmp`` and build directories. Returns with ZERO model calls: ordinary
  ``rm`` must keep working at no cost.

Pure function of (command, cwd, env, scratchpad); filesystem probes are
best-effort ``stat`` calls whose failure never lowers a rank.
"""

from __future__ import annotations

import enum
import fnmatch
import functools
import os
import re
import sys
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field

from local_operator.guardrails.resolve import Creation, Target, resolve_command

_STATE_DIRNAME = ".local-operator"
_CONFIG_FILE = "config.yml"
_SECRETS_DIRNAME = "secrets"
_SESSIONS_DIRNAME = "sessions"
_DESTROYING_OPS = frozenset({"delete", "move", "clean"})
# Spelled here rather than imported: ``provenance`` imports this module.
_LEDGER_DIRNAME = "guardrails"
_LEDGER_FILENAME = "created.sqlite"


class Risk(enum.IntEnum):
    LOW = 0
    HIGH = 1
    CRITICAL = 2


@dataclass
class RankedTarget:
    target: Target
    risk: Risk
    reason: str
    #: What a refusal or preview names instead of ``path``: for a glob whose
    #: matches were vouched for, the matches — not the directory the glob
    #: lives in, which read as though ``$HOME`` itself were owned (QA Q5).
    shown: str = ""

    @property
    def path(self) -> str:
        return self.target.path or "<unresolved>"


@dataclass
class Verdict:
    risk: Risk
    targets: list[RankedTarget] = field(default_factory=list)
    reasons: list[str] = field(default_factory=list)

    @property
    def critical_targets(self) -> list[RankedTarget]:
        return [t for t in self.targets if t.risk is Risk.CRITICAL]

    @property
    def flagged_targets(self) -> list[RankedTarget]:
        return [t for t in self.targets if t.risk is not Risk.LOW]


def _norm(path: str) -> str:
    return os.path.normpath(os.path.abspath(os.path.expanduser(path)))


def _spellings(path: str) -> set[str]:
    """The normalised path and, when it exists, its resolved form (symlinks).

    R9: on a case-insensitive volume (the macOS default) ``~/.LOCAL-OPERATOR``
    IS the state root. ``realpath`` keeps the caller's case for components it
    did not have to resolve, so each existing component is re-cased from its
    directory listing — the on-disk spelling is what the protected roots use.
    """
    out = {_norm(path)}
    try:
        out.add(os.path.realpath(path))
    except OSError:
        pass
    for spelled in list(out):
        cased = _on_disk_case(spelled)
        if cased is not None:
            out.add(cased)
    return out


try:
    import fcntl as _fcntl
except ImportError:  # pragma: no cover - Windows has no fcntl
    _fcntl = None  # type: ignore[assignment]


def _on_disk_case(path: str) -> str | None:
    """``path`` with its EXISTING prefix in on-disk case; ``None`` if unchanged.

    R2-4: walking the components with ``os.listdir`` cost 10-300 ms per call on
    a real state root (``sessions/`` holds thousands of entries). macOS can
    report a directory's canonical path from an open descriptor
    (``F_GETPATH``) in one syscall; other platforms are case-sensitive in
    practice and keep the listing walk, which runs only where a cased name
    exists but is not spelled that way.
    """
    prefix = path
    rest: list[str] = []
    while prefix not in ("", "/") and not os.path.lexists(prefix):
        prefix, tail = os.path.split(prefix)
        rest.append(tail)
    if prefix in ("", "/"):
        return None
    canonical = _canonical_dir(prefix if os.path.isdir(prefix) else os.path.dirname(prefix))
    if canonical is None:
        return None
    if not os.path.isdir(prefix):
        leaf = os.path.basename(prefix)
        try:
            names = os.listdir(canonical) if leaf not in os.listdir(canonical) else [leaf]
        except OSError:
            return None
        leaf = next((n for n in names if n.casefold() == leaf.casefold()), leaf)
        canonical = os.path.join(canonical, leaf)
    out = os.path.join(canonical, *reversed(rest)) if rest else canonical
    return out if out != path else None


def _canonical_dir(directory: str) -> str | None:
    flag = getattr(_fcntl, "F_GETPATH", None) if _fcntl is not None else None
    if _fcntl is None or flag is None:
        return _canonical_dir_by_listing(directory)
    try:
        fd = os.open(directory, os.O_RDONLY)
    except OSError:
        return None
    try:
        raw = _fcntl.fcntl(fd, flag, bytes(1024))
    except OSError:
        return None
    finally:
        os.close(fd)
    return raw.split(b"\0", 1)[0].decode("utf-8", "surrogateescape")


def _canonical_dir_by_listing(directory: str) -> str | None:
    current = "/"
    for part in [p for p in directory.split("/") if p]:
        candidate = os.path.join(current, part)
        if not os.path.exists(candidate):
            return None
        try:
            names = os.listdir(current)
        except OSError:
            return None
        if part not in names:
            part = next((n for n in names if n.casefold() == part.casefold()), part)
        current = os.path.join(current, part)
    return current


def _under(path: str, root: str) -> bool:
    return path == root or path.startswith(root.rstrip("/") + "/")


@functools.lru_cache(maxsize=8)
def _case_insensitive(probe: str) -> bool:
    """Is the volume holding ``probe`` case-insensitive? Probed once per path
    per process: does the case-swapped spelling of the nearest existing
    ancestor resolve? Unknown → the platform default (APFS/HFS+ on darwin
    are case-insensitive by default)."""
    path = probe
    while path not in ("", "/"):
        flipped = path.swapcase()
        if flipped != path and os.path.lexists(path):
            return os.path.lexists(flipped)
        path = os.path.dirname(path)
    return sys.platform == "darwin"


def _under_evidence(path: str, root: str) -> bool:
    """``_under`` for the evidence check: case-folded on a case-insensitive
    volume, where ``~/.LOCAL-OPERATOR/sessions`` IS the state root
    (review round 9, MINOR-1)."""
    if _under(path, root):
        return True
    return _case_insensitive(root) and _under(path.casefold(), root.casefold())


def _is_ancestor_or_self(path: str, of: str) -> bool:
    return _under(of, path)


@dataclass(frozen=True)
class _Protected:
    state_roots: set[str]
    config_files: set[str]
    secrets_dirs: set[str]
    install_roots: set[str]
    homes: set[str]
    own_session: set[str]
    own_scratchpad: set[str]
    #: The provenance ledger(s) (``provenance.ledger_path``): CRITICAL to
    #: overwrite or delete, because a session that could rewrite it could
    #: claim any directory as its own creation.
    ledger_files: set[str] = field(default_factory=set)


def _protected(env: Mapping[str, str], scratchpad: str | None, session_id: str) -> _Protected:
    from local_operator.paths import config_dir

    key = (
        env.get("HOME"),
        os.path.expanduser("~"),
        env.get("LOCAL_OPERATOR_CONFIG_DIR"),
        str(config_dir()),
        scratchpad,
        session_id,
    )
    cached = _PROTECTED_CACHE.get(key)
    if cached is None:
        if len(_PROTECTED_CACHE) > 64:
            _PROTECTED_CACHE.clear()
        cached = _PROTECTED_CACHE[key] = _build_protected(env, scratchpad, session_id)
    return cached


#: R2-4: the protected roots' spellings depend only on the inputs above, so
#: they are computed once per process per distinct key, not per bash call.
_PROTECTED_CACHE: dict[tuple[str | None, ...], _Protected] = {}


def _build_protected(env: Mapping[str, str], scratchpad: str | None, session_id: str) -> _Protected:
    # Imported lazily: ``update`` is a large module and this runs per bash call
    # only when a command actually destroys something.
    from local_operator.paths import config_dir
    from local_operator.update import generations_dir, stable_root

    homes: set[str] = set()
    for home in (env.get("HOME"), os.path.expanduser("~")):
        if home:
            homes |= _spellings(home)
    states: set[str] = set(_spellings(str(config_dir())))
    override = env.get("LOCAL_OPERATOR_CONFIG_DIR")
    if override:
        states |= _spellings(override)
    for home in list(homes):
        states |= _spellings(os.path.join(home, _STATE_DIRNAME))
    installs = _spellings(str(stable_root())) | _spellings(str(generations_dir()))
    own_session: set[str] = set()
    own_pad: set[str] = set()
    if scratchpad:
        own_pad = _spellings(scratchpad)
        own_session = {os.path.dirname(p) for p in own_pad}
    elif session_id:
        for state in states:
            own_session.add(os.path.join(state, _SESSIONS_DIRNAME, session_id))
    return _Protected(
        state_roots=states,
        config_files={os.path.join(s, _CONFIG_FILE) for s in states},
        secrets_dirs={os.path.join(s, _SECRETS_DIRNAME) for s in states},
        install_roots=installs,
        homes=homes,
        own_session=own_session,
        own_scratchpad=own_pad,
        ledger_files={os.path.join(s, _LEDGER_DIRNAME, _LEDGER_FILENAME) for s in states},
    )


def _pattern_reaches(pattern: str, protected: str) -> bool:
    """Could a glob pattern match ``protected`` or one of its ancestors?

    Component-wise so ``*`` never crosses ``/`` (``fnmatch`` alone would let it).
    """
    pat = [c for c in pattern.split("/") if c]
    prot = [c for c in protected.split("/") if c]
    if len(pat) > len(prot):
        return False
    return all(fnmatch.fnmatchcase(p, q) for p, q in zip(prot, pat, strict=False))


#: The files the provenance checks READ as evidence (``guardrails/evidence.py``,
#: ``provenance.py``): a session's transcript, its lineage and its ask store,
#: matched per ``<state>/sessions/<id>/``. Any write to them — including an
#: append — could forge a creating call, a child, or an operator's answer
#: (review R1-MAJOR-1), so every op on them is CRITICAL.
_EVIDENCE_FILES = ("transcript.jsonl", "origin.json", "asks.jsonl", "subagent-roster*.json")
#: Matched case-insensitively (APFS/HFS+ are case-insensitive, so
#: ``ORIGIN.JSON`` is the same file) and with sqlite's side files (review
#: round 7, MINOR-2).
_EVIDENCE_SUFFIXES = ("", "-wal", "-shm", "-journal")


def _is_evidence_name(name: str) -> bool:
    folded = name.casefold()
    return any(
        fnmatch.fnmatchcase(folded, f + suffix)
        for f in _EVIDENCE_FILES
        for suffix in _EVIDENCE_SUFFIXES
    )


_EVIDENCE_SAMPLES = {"subagent-roster*.json": "subagent-roster.v1.json"}


def _evidence_reason(path: str, target: Target, prot: _Protected) -> str | None:
    for state in prot.state_roots:
        if _under(path, os.path.join(state, _LEDGER_DIRNAME)):
            return "the guardrail's own records"
        sessions = os.path.join(state, _SESSIONS_DIRNAME)
        if not _under(path, sessions) or path == sessions:
            continue
        rel = path[len(sessions) + 1 :].split("/")
        if len(rel) == 2 and _is_evidence_name(rel[1]):
            return "a session's transcript, lineage or ask record"
        if len(rel) == 1 and target.op in ("overwrite", "append"):
            # Writing a directory INTO ``sessions/`` (``mv /tmp/x sessions/``)
            # plants a session the lineage walk would read.
            return "a session directory written in place"
    return None


def _touch_reason(target: Target, prot: _Protected, *, globstar: bool = False) -> str | None:
    """Does ``target`` land in the guard's evidence?

    Evidence = anything at or under ``<state>/sessions/`` outside this
    session's own scratchpad, or under ``<state>/guardrails/`` (review
    R2-MAJOR-1, QA Q6). The CALLER decides the rank by the touch's shape:
    write-shaped → CRITICAL, an unknown program handed the path → HIGH
    (R3-MAJOR-2; see ``resolve.Resolution.touched``). An
    unresolvable part keeps the literal directory above it, so
    ``sessions/$ID`` lands on ``sessions/`` and still counts.
    """
    path = target.path
    if path is None:
        return None
    lexical = {path, target.pattern or path}
    # Cheap prefilter: realpath/F_GETPATH only for words that could be there.
    try:
        lexical.add(os.path.realpath(path))
    except OSError:
        pass
    if target.pattern is None and not any(
        _SESSIONS_DIRNAME in p.casefold() or _LEDGER_DIRNAME in p.casefold() for p in lexical
    ):
        return None
    for spelled in {s for p in lexical for s in _spellings(p)}:
        for state in prot.state_roots:
            if _under_evidence(spelled, os.path.join(state, _LEDGER_DIRNAME)):
                return "the guardrail's own records"
            if _under_evidence(spelled, os.path.join(state, _SESSIONS_DIRNAME)) and not any(
                _under_evidence(spelled, pad) for pad in prot.own_scratchpad
            ):
                return "session state (transcripts, lineage, ask stores)"
        if target.pattern is not None:
            prefix = target.path or ""
            # A glob cannot climb out of its literal directory (``*`` never
            # crosses ``/``), so one rooted in this session's scratchpad stays
            # there.
            if prefix and any(_under_evidence(prefix, pad) for pad in prot.own_scratchpad):
                return None
            for state in prot.state_roots:
                if _pattern_hits(
                    target.pattern,
                    os.path.join(state, _LEDGER_DIRNAME),
                    fold=_case_insensitive(state),
                ):
                    return "the guardrail's own records"
                if _pattern_hits(
                    target.pattern,
                    os.path.join(state, _SESSIONS_DIRNAME),
                    fold=_case_insensitive(state),
                ):
                    return "session state (transcripts, lineage, ask stores)"
            # Lexically elsewhere — but a match may be a SYMLINK into session
            # state (``ln -s ../../<child> $P/m`` then ``$P/m*/origin.json``),
            # so expand it now and judge each match by its realpath (review
            # round 7). Over the cap there is no way to know: protected.
            return _glob_matches_evidence(target.pattern, prot, globstar=globstar)
    return None


#: Bounds on the check-time glob expansion: matches, and directory entries
#: visited (a ``**`` over ``$HOME`` must not walk the disk inside a tool
#: call). Past either bound there is no way to know: protected.
_GLOB_CAP = 2000
_GLOB_WALK_CAP = 20_000


def _expand_glob(pattern: str, *, globstar: bool = False) -> list[tuple[str, bool]] | None:
    """Matches of ``pattern`` as BASH would expand it, or ``None`` past a bound.

    Bash semantics, because that is what the command will do (review round
    8, MAJOR-1: an any-depth ``**`` walked whole repositories, cost up to
    1.7 s per call and ranked ordinary ``**/*.py`` commands HIGH/CRITICAL):

    * ``**`` is ``*`` — one level — unless the command may make it
      recursive (``resolve._RECURSIVE_GLOB_RE``); then it is any depth, following directory links
      once per real directory.
    * Dot entries ALWAYS match (review round 9): bash hides them unless
      ``dotglob``/``GLOBIGNORE``/``nocaseglob``-style options say otherwise,
      and tracking every such option is a losing game. One level of dot
      entries is cheap; the safety walk is a superset of what bash expands.

    Hand-rolled rather than ``glob.glob`` because that bounds only results,
    not the walk. Each match carries whether ANY component on its way was a
    symlink: only those need a ``realpath`` (which costs ~8 lstats each and
    dominated the check on an 800-match ``**/*.py``).
    """
    parts = [c for c in pattern.split("/") if c]
    found: list[tuple[str, bool]] = []
    visited = 0
    walked: set[str] = set()
    stack: list[tuple[str, int, bool]] = [("/", 0, False)]
    while stack:
        base, k, linked = stack.pop()
        if k == len(parts):
            found.append((base, linked))
            if len(found) > _GLOB_CAP:
                return None
            continue
        part = parts[k]
        if not any(ch in part for ch in "*?["):
            nxt = os.path.join(base, part)
            if os.path.lexists(nxt):
                stack.append((nxt, k + 1, linked or os.path.islink(nxt)))
            continue
        deep = part == "**" and globstar
        if deep:
            real = os.path.realpath(base)
            if real in walked:
                continue
            walked.add(real)
        try:
            entries = list(os.scandir(base))
        except OSError:
            continue
        visited += len(entries)
        if visited > _GLOB_WALK_CAP:
            return None
        if deep:
            # ``**`` = zero or more directories: try the NEXT part against
            # these same entries (no second scan), and descend into each
            # directory still inside ``**``.
            nxt_part = parts[k + 1] if k + 1 < len(parts) else None
            if nxt_part is None:
                found.append((base, linked))
            for e in entries:
                via = linked or e.is_symlink()
                if nxt_part is not None and _component_matches(e.name, nxt_part):
                    stack.append((e.path, k + 2, via))
                try:
                    if e.is_dir():
                        stack.append((e.path, k, via))
                except OSError:
                    continue
            continue
        match_part = "*" if part == "**" else part
        for e in entries:
            if fnmatch.fnmatchcase(e.name, match_part):
                stack.append((e.path, k + 1, linked or e.is_symlink()))
    return found


def _component_matches(name: str, part: str) -> bool:
    if not any(ch in part for ch in "*?["):
        return name == part
    return fnmatch.fnmatchcase(name, part)


#: The reason for a glob past the expansion bounds. Unknowable, not a proven
#: write into evidence: ranked HIGH even where the target is write-shaped
#: (review round 8, MAJOR-1).
GLOB_TOO_WIDE = "a glob too wide to check against session state"


def _glob_matches_evidence(pattern: str, prot: _Protected, *, globstar: bool = False) -> str | None:
    matches = _expand_glob(pattern, globstar=globstar)
    if matches is None:
        return GLOB_TOO_WIDE
    for match, linked in matches:
        # No link on the way: the lexical path IS the real one, and the
        # caller already checked it lexically.
        if not linked:
            continue
        real = os.path.realpath(match)
        for state in prot.state_roots:
            if _under_evidence(real, os.path.join(state, _LEDGER_DIRNAME)):
                return "the guardrail's own records (through a link)"
            if _under_evidence(real, os.path.join(state, _SESSIONS_DIRNAME)) and not any(
                _under_evidence(real, pad) for pad in prot.own_scratchpad
            ):
                return "session state (through a link)"
    return None


def _pattern_hits(pattern: str, root: str, *, fold: bool = False) -> bool:
    """Could ``pattern`` match ``root``, an ancestor of it, or anything under it?

    Component-wise over the shared length, so ``~/.local-operator/sess*/x/f``
    reaches ``sessions`` (``_pattern_reaches`` alone stops at patterns no
    longer than ``root``). Over-matches dotfiles on purpose: fail closed.
    """
    if fold:
        pattern, root = pattern.casefold(), root.casefold()
    pat = [c for c in pattern.split("/") if c]
    prot = [c for c in root.split("/") if c]
    return all(fnmatch.fnmatchcase(p, q) for p, q in zip(prot, pat, strict=False))


def _evidence_by_pattern(target: Target, prot: _Protected) -> str | None:
    if target.pattern is None:
        return None
    for state in prot.state_roots:
        guard_dir = os.path.join(state, _LEDGER_DIRNAME, "x")
        if _pattern_reaches(target.pattern, guard_dir):
            return f"glob {target.pattern} reaches the guardrail's own records"
        for name in _EVIDENCE_FILES:
            sample = _EVIDENCE_SAMPLES.get(name, name)
            pat = [c for c in target.pattern.split("/") if c]
            tmpl = [c for c in os.path.join(state, _SESSIONS_DIRNAME, "*", sample).split("/") if c]
            if len(pat) == len(tmpl) and all(
                t == "*" or fnmatch.fnmatchcase(t, p) or p == "*"
                for p, t in zip(pat, tmpl, strict=True)
            ):
                return f"glob {target.pattern} reaches session transcripts or lineage"
    return None


def _critical_reason(path: str, target: Target, prot: _Protected) -> str | None:
    destroying = target.op in _DESTROYING_OPS
    evidence = _evidence_reason(path, target, prot)
    if evidence is not None:
        return evidence
    for cfg in prot.config_files:
        if path == cfg:
            return "the harness config file"
    for ledger in prot.ledger_files:
        if path == ledger:
            return "the guardrail creation ledger"
    for sec in prot.secrets_dirs:
        if _under(path, sec):
            return "the encrypted secret store"
    if not destroying:
        return None
    if path == "/":
        return "the filesystem root"
    for state in prot.state_roots:
        if _is_ancestor_or_self(path, state):
            return (
                "the local-operator state root"
                if path == state
                else "an ancestor of the state root"
            )
    for install in prot.install_roots:
        if _is_ancestor_or_self(path, install):
            return "the installed lop generations"
    for home in prot.homes:
        if _is_ancestor_or_self(path, home):
            return "the home directory" if path == home else "an ancestor of the home directory"
    for ledger in prot.ledger_files:
        # After the state root: deleting ``<state>/guardrails`` is named for
        # what it destroys, and the state root itself keeps its own reason.
        if _is_ancestor_or_self(path, ledger):
            return "the guardrail creation ledger"
    return None


def _critical_by_pattern(target: Target, prot: _Protected) -> str | None:
    if target.pattern is None:
        return None
    evidence = _evidence_by_pattern(target, prot)
    if evidence is not None:
        return evidence
    candidates = [
        *prot.config_files,
        *prot.secrets_dirs,
        *prot.ledger_files,
        *(prot.state_roots if target.op in _DESTROYING_OPS else ()),
        *(prot.install_roots if target.op in _DESTROYING_OPS else ()),
        *(prot.homes if target.op in _DESTROYING_OPS else ()),
    ]
    for protected in candidates:
        if _pattern_reaches(target.pattern, protected):
            return f"glob {target.pattern} matches {protected}"
    return None


def _high_reason(path: str, target: Target, prot: _Protected) -> str | None:
    if target.op == "append":
        # An append destroys nothing; it is ranked only as forged evidence,
        # which ``_critical_reason`` already decided.
        return None
    if target.op == "clean" and target.program == "git clean":
        return "git clean removes untracked work"
    if any(_under(path, own) for own in prot.own_session):
        return None
    if target.op not in _DESTROYING_OPS:
        # S10: overwriting a file inside the state root (outside this session's
        # own directory) rewrites the harness's data — review it.
        for state in prot.state_roots:
            if _under(path, state) and path != state:
                return "overwrites a file inside the local-operator state root"
        return None
    if target.glob_children and not target.glob_all:
        # R3-2: a children-glob directly in a home or in the state root
        # (``~/D*``, ``~/[A-Z]*``, ``Doc*`` at cwd $HOME) can match the
        # user's top-level data. R2-5 only lowered it from CRITICAL; it is
        # judged, unless the pattern is file-shaped (``*.log``) at a home.
        if path in prot.state_roots:
            return "a glob over the state root's top-level entries"
        if path in prot.homes and not _file_shaped(target.pattern):
            return "a glob over the home directory's top-level entries"
        # Q2: ``rm *.log`` removes matching CHILDREN; the directory survives.
        for state in prot.state_roots:
            if _under(path, state):
                break
        else:
            return None
    for state in prot.state_roots:
        if _under(path, state):
            sessions = os.path.join(state, _SESSIONS_DIRNAME)
            if _under(path, sessions) and path != sessions:
                return "another session's files"
            return "inside the local-operator state root"
    for home in prot.homes:
        if os.path.dirname(path) == home and not os.path.isfile(path):
            return "a top-level directory of the home directory"
    try:
        if os.path.isdir(path) and os.path.exists(os.path.join(path, ".git")):
            return "a git repository root"
    except OSError:
        return "an unreadable path"
    return None


def _rank(target: Target, prot: _Protected, session_id: str = "") -> RankedTarget:
    path = target.path
    notes: list[str] = list(target.empty)
    if target.climbs:
        # S2-5b: say where the ``..`` starts from, not that it is inside it.
        notes.append(f"climbs out of {target.path} by an unknown amount (unresolved)")
    elif target.opaque:
        notes.append("contains an unresolvable substitution or variable (opaque)")
    if target.unknown_cwd:
        notes.append("relative to a working directory that cannot be determined")
    if path is None:
        return RankedTarget(target, Risk.HIGH, "; ".join(notes) or "unresolved target")
    for spelling in _spellings(path):
        if target.glob_children and not target.glob_all and not target.opaque:
            # R2-5: ``rm *.log`` at $HOME removes matching CHILDREN; it is
            # CRITICAL only when the pattern itself can reach a protected name.
            reason = _critical_by_pattern(target, prot)
        else:
            reason = _critical_reason(spelling, target, prot) or _critical_by_pattern(target, prot)
        if reason is not None and not target.opaque:
            return RankedTarget(target, Risk.CRITICAL, "; ".join([reason, *notes]))
        if reason is not None and target.opaque and path in prot.state_roots | prot.homes:
            # An opaque component directly under a protected root still names
            # something inside it — and may be empty or ``..``: no benefit of doubt.
            return RankedTarget(target, Risk.CRITICAL, "; ".join([reason, *notes]))
    # S1: fresh-temp/contained-path exemptions cannot prove a privacy-hidden
    # suffix safe. Keep protected-root CRITICAL precedence above, but never let
    # an uninspectable credential-derived target become LOW.
    if target.credential_derived:
        return RankedTarget(target, Risk.HIGH, "; ".join(notes) or "unresolved target")
    # The session's own scratchpad is where the agent is TOLD to work, so a
    # timestamped component there (``$(date +%s)``) stays LOW — but only when the
    # substitution's program cannot print a ``..`` that climbs out of the pad.
    in_own_pad = any(_under(path, pad) for pad in prot.own_scratchpad) and (
        not target.opaque or target.opaque_contained
    )
    # R6: ``$(mktemp …)`` names a fresh temp entry — not "unknown" in the sense
    # that matters once CRITICAL ancestry was ruled out above.
    if target.temp:
        return RankedTarget(target, Risk.LOW, "fresh temporary path ($(mktemp))")
    if notes and not in_own_pad:
        return RankedTarget(target, Risk.HIGH, "; ".join(notes))
    if target.worktree_remove and not target.forced and target.pattern is None:
        # ``git worktree remove`` without --force: git refuses tracked or
        # untracked changes itself, but deletes IGNORED files and a detached
        # HEAD's unreferenced commits. ``worktree_state`` proves those absent
        # (only regenerable ignored output, HEAD reachable from a ref). Anything
        # else (dirty, unknown, not linked) takes the ordinary ranking below.
        from local_operator.guardrails.provenance import worktree_state

        if worktree_state(path) == "clean":
            return RankedTarget(target, Risk.LOW, "a clean linked git worktree")
    for spelling in _spellings(path):
        reason = _high_reason(spelling, target, prot)
        if reason is not None:
            in_state = any(_under(spelling, st) for st in prot.state_roots)
            if target.selected and not in_state and _is_work_tree(spelling):
                # S2-3: a find with a selecting predicate (``-name '*.pyc'``)
                # removes a SUBSET of a working tree — the R6 cleanup shape. It
                # overrides ONLY the working-tree reasons: a repo root, even one
                # at $HOME's top level (``~/<project>``). State-root children,
                # other sessions and non-repo home dirs (``~/Documents``) stay
                # HIGH, because their contents are the data a confused
                # predicate would destroy.
                return RankedTarget(target, Risk.LOW, "find with a selecting predicate")
            owned = _owned_by_session(target, path, prot, session_id)
            if owned is not None:
                reason, matches = owned
                shown = ""
                if target.pattern is not None:
                    shown = ", ".join(matches[:4]) + (
                        f" … and {len(matches) - 4} more" if len(matches) > 4 else ""
                    )
                return RankedTarget(target, Risk.LOW, reason, shown)
            return RankedTarget(target, Risk.HIGH, "; ".join([reason, *notes]))
    if target.selected:
        return RankedTarget(target, Risk.LOW, "find with a selecting predicate")
    return RankedTarget(target, Risk.LOW, "ordinary path")


def _owned_by_session(
    target: Target, path: str, prot: _Protected, session_id: str
) -> tuple[str, list[str]] | None:
    """A LOW reason when this session created what ``target`` destroys, else ``None``.

    Only reached for a target already ranked HIGH by ``_high_reason`` and
    already cleared of CRITICAL ancestry, credential derivation and opaque or
    empty parts (those returned earlier in ``_rank``). The ledger
    (``provenance.owned_record``) must cover the target by realpath AND inode.
    Still HIGH: anything in the state root (another session's files), a
    primary checkout this session did not clone, and a glob with any on-disk
    match the session does not own (or no match at all — then it destroys
    nothing the ledger can vouch for).
    """
    if not session_id or target.credential_derived or target.opaque or target.unknown_cwd:
        return None
    from local_operator.guardrails import provenance

    if target.pattern is not None:
        import glob as _glob

        try:
            matches = _glob.glob(target.pattern)
        except (OSError, ValueError):
            return None
        if not matches:
            return None
        paths = matches
    else:
        paths = [path]
    first: str | None = None
    for candidate in paths:
        real = os.path.realpath(candidate)
        if any(_under(real, st) for st in prot.state_roots):
            return None
        record = provenance.owned_record(real, session_id)
        if record is None:
            return None
        if provenance.is_primary_checkout(real) and not (
            record.kind == "clone" and record.path == real
        ):
            return None
        first = first or record.describe()
    if first is None:
        return None
    reason = first if len(paths) == 1 else f"{len(paths)} matches, all {first}"
    return reason, sorted(paths)


#: ``*.log``, ``build-*.tmp``: a last component that ends in a literal
#: extension. Such a glob matches files by type, not the user's directories.
_FILE_SHAPED_RE = re.compile(r"\.[A-Za-z0-9]{1,8}$")


def _file_shaped(pattern: str | None) -> bool:
    if not pattern:
        return False
    last = pattern.rstrip("/").rsplit("/", 1)[-1]
    # The extension itself must be literal (``*.log``); ``*.[ch]``/``*.*`` are
    # not file-shaped. ``_FILE_SHAPED_RE`` already admits only alphanumerics.
    return bool(_FILE_SHAPED_RE.search(last))


def _is_work_tree(path: str) -> bool:
    try:
        return os.path.isdir(path) and os.path.exists(os.path.join(path, ".git"))
    except OSError:
        return False


def classify_targets(
    targets: Sequence[Target],
    opaque_reasons: Sequence[str],
    *,
    env: Mapping[str, str],
    scratchpad: str | None,
    session_id: str = "",
    touched: Sequence[Target] = (),
    created: Sequence[Creation] = (),
    blind: Sequence[str] = (),
    globstar: bool = False,
) -> Verdict:
    prot = _protected(env, scratchpad, session_id) if (targets or touched or created) else None
    evidence_hits: list[RankedTarget] = []
    if prot is not None:
        # Write-shaped: redirections and known writers' destinations
        # (``touched`` op ``write``), plus everything the resolver itself
        # destroys or creates. Those plant or forge evidence → CRITICAL. A
        # path merely handed to an unknown program (op ``touch``) is HIGH so
        # the judge or the operator can lift it (review R3-MAJOR-2).
        writes = (
            [t for t in touched if t.op != "touch"]
            + list(targets)
            + [Target(path=c.path, op="write", program=c.kind) for c in created]
        )
        for t in writes:
            reason = _touch_reason(t, prot, globstar=globstar)
            if reason is not None:
                risk = Risk.HIGH if reason == GLOB_TOO_WIDE else Risk.CRITICAL
                evidence_hits.append(RankedTarget(t, risk, reason))
        critical = {h.target.path for h in evidence_hits}
        for t in touched:
            # A path already refused as a write needs no second, HIGH row.
            if t.op != "touch" or t.path in critical:
                continue
            reason = _touch_reason(t, prot, globstar=globstar)
            if reason is not None:
                evidence_hits.append(
                    RankedTarget(t, Risk.HIGH, f"{reason}, handed to {t.program or 'a program'}")
                )
    for name in dict.fromkeys(blind):
        evidence_hits.append(
            RankedTarget(
                Target(path=None, op="touch", program=name, unknown_cwd=True),
                Risk.HIGH,
                f"names {name} while the working directory cannot be determined",
                shown=name,
            )
        )
    if not targets and not opaque_reasons and not evidence_hits:
        return Verdict(Risk.LOW)
    prot = prot or _protected(env, scratchpad, session_id)
    ranked = [_rank(t, prot, session_id) for t in targets]
    # An append that is not forging evidence is not a destructive target at all:
    # drop it so it adds no "destructive" annotation and no target text.
    ranked = [r for r in ranked if not (r.target.op == "append" and r.risk is Risk.LOW)]
    # One row per path: a redirection is both a destroyed target and a
    # touched path, and the refusal should not name it twice.
    seen = {(r.target.path, r.risk) for r in ranked}
    ranked.extend(h for h in evidence_hits if (h.target.path, h.risk) not in seen)
    risk = max((r.risk for r in ranked), default=Risk.LOW)
    if opaque_reasons:
        risk = max(risk, Risk.HIGH)
    reasons = [r.reason for r in ranked if r.risk is not Risk.LOW] + list(opaque_reasons)
    return Verdict(risk=risk, targets=ranked, reasons=reasons)


def classify_bash(
    command: str,
    *,
    cwd: str | None,
    env: Mapping[str, str],
    scratchpad: str | None,
    session_id: str = "",
) -> Verdict:
    """Rank a bash command. Pure parse: the command is never executed."""
    home = env.get("HOME") or os.path.expanduser("~")
    resolution = resolve_command(command, cwd=_norm(cwd) if cwd else None, env=env, home=home)
    return classify_targets(
        resolution.targets,
        resolution.opaque_reasons,
        env=env,
        scratchpad=scratchpad,
        session_id=session_id,
        touched=resolution.touched,
        created=resolution.created,
        blind=resolution.blind,
        globstar=resolution.globstar,
    )
