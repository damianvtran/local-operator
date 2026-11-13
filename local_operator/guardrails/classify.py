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

import dataclasses
import enum
import fnmatch
import functools
import os
import re
import stat
import sys
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field

from local_operator.guardrails.resolve import (
    Creation,
    Target,
    kernel_path,
    physical_path,
    resolve_command,
)

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
#: (review R1-MAJOR-1), so every op on them is CRITICAL. ``temporary.json`` is
#: the temporary-chat mark the cleanup pass DELETES on: a shell that could plant
#: one in another session could get that conversation deleted.
_EVIDENCE_FILES = (
    "transcript.jsonl",
    "origin.json",
    "asks.jsonl",
    "subagent-roster*.json",
    "temporary.json",
)
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


#: A session's working directory (``paths.session_scratchpad``). Its files are
#: the agent's own scratch — keys, logs, rigs — never anything the guard reads
#: as provenance, so ANY session's scratchpad is outside the evidence class.
#: Another session's scratchpad still ranks HIGH ("another session's files",
#: ``_high_reason``), which the judge or a verified ask can lift; ranking it
#: CRITICAL made an operator-approved cleanup of a leftover key file
#: impossible (v0.101.22 false refusal).
_SCRATCHPAD_DIRNAME = "scratchpad"


def _in_session_scratchpad(path: str, sessions: str, *, pattern: bool = False) -> bool:
    """Is ``path`` (or every match of glob ``path``) inside SOME session's
    scratchpad, ``<sessions>/<id>/scratchpad[/...]``, judged lexically?

    Callers pass every spelling, the realpath included, so a link in a
    scratchpad resolving into evidence is still caught by its resolved
    spelling. Fail closed on a ``**`` session component (globstar spans
    levels) or a glob in place of the literal ``scratchpad`` component.

    ``..`` never reaches here: callers pass normalised spellings, which is
    exactly why a lexical check cannot see a ``..`` that the kernel resolves
    physically through a link. It is decided on the word as written instead
    (``Target.dotdot``, refused by ``_pad_exemptible``; review R1-MAJOR-1).
    """
    fold = _case_insensitive(sessions)
    p, root = (path.casefold(), sessions.casefold()) if fold else (path, sessions)
    root = root.rstrip("/")
    if not p.startswith(root + "/"):
        return False
    rel = [c for c in p[len(root) + 1 :].split("/") if c]
    if len(rel) < 2 or rel[1] != _SCRATCHPAD_DIRNAME:
        return False
    return not (pattern and "**" in rel[0])


#: ``_touch_reason``'s answer for a path in ANOTHER session's scratchpad that
#: reaches no evidence: not a forgery (so a write there is not CRITICAL), but
#: still someone else's files, so a mention handed to an unknown program
#: keeps the backstop's HIGH and a destroying op keeps ``_high_reason``'s.
OTHER_SCRATCHPAD = "another session's files"


def _tree_read(target: Target, prot: _Protected, session_id: str) -> str | None:
    """Case 7 (2026-11-09): a read of a TASK-TREE session's scratch file.

    The settings lane ran ``node scripts/i18n/ledger-assemble.mjs --receipt
    <auditor pad>/….json``; the auditor was its sibling under one manager,
    and the backstop refused it as another session's files. ``None`` = LOW:
    the file exists, its session is in the caller's subagent tree
    (``provenance.same_task_tree``), the interpreter runs a git-tracked script
    (``Target.script``) and no output flag names the file. Otherwise the
    suffix shown with the HIGH: the lineage fact when there is one, so the
    judge sees it; ``""`` for a stranger's pad (unchanged). Evidence files
    never reach here (``_touch_reason`` returns another reason for them).
    """
    from local_operator.guardrails import provenance

    path = target.path
    if not path or not session_id or target.pattern is not None or target.opaque:
        return ""
    # Review R2-M2: the owner comes from EVERY spelling, the resolved one
    # included. ``<sibling pad>/lnk -> <stranger pad>`` names the sibling
    # lexically but the kernel reads the stranger's file, so a lexical owner
    # alone would carry the tree's LOW to a session outside the tree.
    spellings = {path, os.path.realpath(path)}
    if target.raw_path:
        spellings.add(os.path.realpath(target.raw_path))
    roots = set(prot.state_roots) | {os.path.realpath(s) for s in prot.state_roots}
    relations: dict[str, str] = {}
    for spelling in spellings:
        owner = _pad_owner(spelling, roots)
        if owner is None:
            return ""
        if owner not in relations:
            relation = provenance.same_task_tree(
                session_id, owner, state_root=_owner_root(owner, roots)
            )
            if not relation:
                return ""
            relations[owner] = relation
    if (
        os.path.isfile(path)
        and not os.path.islink(path)
        and not target.as_output
        and provenance.tracked_script(target.script, state_roots=roots)
    ):
        return None
    # Incident 2026-11-13 (0.102.2): ``python3 -c`` printing lines of four logs
    # in the caller's child's pad. A tracked script must be handed a FILE (it
    # could write into a directory it is given); an inline program proven
    # unable to write anything (``Target.read_only_code``) may be handed the
    # pad directory too — it joins file names onto it and only reads.
    #
    # Deliberately READS only. A parent DELETING its descendant's scratchpad
    # keeps ``_high_reason``'s HIGH (destroying ops never come through here):
    # the harness, not the child, creates that directory, so no ledger row
    # vouches for it; the child may still be running and writing there; and
    # its pad holds the child's working evidence the parent's report cites.
    # What a subagent itself creates (``mkdir``, worktrees) is already LOW for
    # the parent through the creation ledger written under every ancestor.
    if (
        target.read_only_code
        and not target.as_output
        and os.path.lexists(path)
        and not os.path.islink(path)
    ):
        return None
    owner = _pad_owner(os.path.realpath(path), roots) or ""
    return f" (session {owner} is this session's {relations.get(owner, '')} in one task tree)"


def _pad_owner(path: str, roots: set[str]) -> str | None:
    """The session id whose directory ``path`` lies under, in any state root."""
    for state in roots:
        sessions = os.path.join(state, _SESSIONS_DIRNAME) + os.sep
        if path.startswith(sessions):
            owner = path[len(sessions) :].split(os.sep, 1)[0]
            return owner or None
    return None


def _owner_root(owner: str, roots: set[str]) -> str:
    """The state root holding ``owner``'s session dir (lineage lives beside it)."""
    for state in sorted(roots):
        if os.path.isdir(os.path.join(state, _SESSIONS_DIRNAME, owner)):
            return state
    return sorted(roots)[0]


def _hardlinked_file(path: str | None) -> bool:
    """Is ``path`` a regular file with more than one name? ``realpath`` cannot
    see a hardlink, so ``<pad>/h`` linked to a ``transcript.jsonl`` looks like
    scratch; appending to it forges the transcript (review R1-MINOR-1)."""
    if not path:
        return False
    try:
        st = os.lstat(path)
    except OSError:
        return False
    return stat.S_ISREG(st.st_mode) and st.st_nlink > 1


def _kernel_reason(target: Target, prot: _Protected) -> str | None:
    """``_touch_reason`` for a word written with ``..`` (``Target.dotdot``).

    The lexical normal form lies about such a word (``<pad>/sub/../<id>/
    asks.jsonl`` with ``sub`` -> a session dir reads as inside the pad; review
    R1-MAJOR-1), so it is judged ONLY where the kernel lands it: inside this
    session's own pad stays exempt (agents tidying their scratch with
    ``rig/../x`` must not hit an unliftable CRITICAL), inside another pad is
    ``OTHER_SCRATCHPAD`` (HIGH), and anything else under ``sessions/`` or
    ``guardrails/`` — or a landing that cannot be known — is CRITICAL.
    """
    kernel = kernel_path(target.raw_path)
    if kernel is None:
        # Opaque, globbed or unresolvable: no exemption, so the literal path
        # decides; under session state that is CRITICAL.
        lexical = target.path or ""
        # An unpinned ``..`` (a missing or re-linked component before it) gets
        # no own-pad exemption either: any part of the same command may make
        # that component a link to anywhere, and every attempt to carve out
        # the safe cases was bypassed by one more move/copy shape (reviews
        # R2-1, R3-1, R4-1 on PR #3811). Name the physical path instead.
        for state in prot.state_roots:
            if _under_evidence(lexical, os.path.join(state, _LEDGER_DIRNAME)):
                return "the guardrail's own records"
            if _under_evidence(lexical, os.path.join(state, _SESSIONS_DIRNAME)):
                return "session state (transcripts, lineage, ask stores)"
        return None
    linked = _hardlinked_file(kernel)
    other_pad = False
    for spelled in _spellings(kernel):
        for state in prot.state_roots:
            sessions = os.path.join(state, _SESSIONS_DIRNAME)
            if _under_evidence(spelled, os.path.join(state, _LEDGER_DIRNAME)):
                return "the guardrail's own records"
            if not _under_evidence(spelled, sessions):
                continue
            if not linked and _in_own_pad(spelled, prot):
                continue
            if not linked and _in_session_scratchpad(spelled, sessions):
                other_pad = True
                continue
            return "session state (transcripts, lineage, ask stores)"
    return OTHER_SCRATCHPAD if other_pad else None


def _own_pad_exemptible(target: Target) -> bool:
    """May this session's OWN scratchpad exempt ``target``'s LEXICAL path?
    Not when the word as written has ``..`` (``Target.dotdot``): its normal
    form may not be where it lands, so a plain literal word of that shape is
    judged by ``_kernel_reason`` instead, and an opaque/globbed one gets no
    exemption (review R1-MAJOR-1). Nor for a hardlinked regular file, whose
    other name may be evidence (R1-MINOR-1)."""
    if target.dotdot:
        return False
    return not _hardlinked_file(target.raw_path or target.path)


def _pad_exemptible(target: Target) -> bool:
    """Can ``target``'s literal path stand for where it lands, in ANY
    session's scratchpad? Not when an unknowable part may print ``..`` or a
    ``..`` follows one (``climbs``), when the word as written has a ``..`` at
    all, or for a hardlinked file (``_own_pad_exemptible``): then a path
    inside a scratchpad may really be the transcript beside it."""
    return (
        _own_pad_exemptible(target)
        and not target.climbs
        and (not target.opaque or target.opaque_contained)
    )


def _in_own_pad(path: str, prot: _Protected) -> bool:
    return any(_under_evidence(path, pad) for pad in prot.own_scratchpad)


def _touch_reason(target: Target, prot: _Protected, *, globstar: bool = False) -> str | None:
    """Does ``target`` land in the guard's evidence?

    Evidence = anything at or under ``<state>/sessions/`` outside every
    session's scratchpad (``_in_session_scratchpad``, decided on each
    spelling including the realpath), or under ``<state>/guardrails/``
    (review R2-MAJOR-1, QA Q6). The CALLER decides the rank by the touch's shape:
    write-shaped → CRITICAL, an unknown program handed the path → HIGH
    (R3-MAJOR-2; see ``resolve.Resolution.touched``). An
    unresolvable part keeps the literal directory above it, so
    ``sessions/$ID`` lands on ``sessions/`` and still counts.

    ``OTHER_SCRATCHPAD`` means "another session's scratchpad, no evidence
    reached": the caller ranks a write there by ``_rank`` (HIGH), not as a
    forgery, and a mention handed to an unknown program stays HIGH.
    """
    path = target.path
    if path is None:
        return None
    lexical = {path, target.pattern or path}
    # Cheap prefilter: realpath/F_GETPATH only for words that could be there.
    # The raw (un-normalised) word's realpath resolves ``..`` physically, the
    # way the kernel will, so a ``sub/../`` through a link is judged where it
    # really lands (review R1-MAJOR-1).
    for spelled in (path, target.raw_path):
        if not spelled:
            continue
        try:
            lexical.add(os.path.realpath(spelled))
        except OSError:
            pass
    if target.pattern is None and not any(
        _SESSIONS_DIRNAME in p.casefold() or _LEDGER_DIRNAME in p.casefold() for p in lexical
    ):
        return None
    if target.dotdot and not target.climbs and not target.opaque and target.pattern is None:
        return _kernel_reason(target, prot)
    exemptible = _pad_exemptible(target)
    own_ok = _own_pad_exemptible(target)
    own_prefix = bool(path) and any(_under_evidence(path, pad) for pad in prot.own_scratchpad)
    if (
        target.pattern is not None
        and exemptible
        and not own_prefix
        and any(
            _in_session_scratchpad(
                _norm(target.pattern), os.path.join(state, _SESSIONS_DIRNAME), pattern=True
            )
            for state in prot.state_roots
        )
    ):
        # ``sessions/*/scratchpad/*.env``: every lexical match is in some
        # pad, so only a match that resolves through a link can reach
        # evidence. Its literal prefix (``sessions/``) is not the target.
        linked = _glob_matches_evidence(target.pattern, prot, globstar=globstar)
        return linked if linked is not None else OTHER_SCRATCHPAD
    other_pad = False
    for spelled in {s for p in lexical for s in _spellings(p)}:
        for state in prot.state_roots:
            sessions = os.path.join(state, _SESSIONS_DIRNAME)
            if (
                exemptible
                and _in_session_scratchpad(spelled, sessions)
                and not (own_ok and _in_own_pad(spelled, prot))
            ):
                other_pad = True
            if _under_evidence(spelled, os.path.join(state, _LEDGER_DIRNAME)):
                return "the guardrail's own records"
            if (
                _under_evidence(spelled, sessions)
                and not (own_ok and _in_own_pad(spelled, prot))
                and not (exemptible and _in_session_scratchpad(spelled, sessions))
            ):
                return "session state (transcripts, lineage, ask stores)"
        if target.pattern is not None:
            prefix = target.path or ""
            # A glob cannot climb out of its literal directory (``*`` never
            # crosses ``/``), so one rooted in this session's scratchpad stays
            # there.
            if (
                prefix
                and own_ok
                and any(_under_evidence(prefix, pad) for pad in prot.own_scratchpad)
            ):
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
    return OTHER_SCRATCHPAD if other_pad else None


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
      and tracking every such option is a losing game. The safety walk is
      a superset of what bash expands. That is cheap for a one-level ``*``;
      under a recursive ``**`` every level includes ``.git``/``.venv``/
      ``node_modules/.pnpm``, so a deep walk over a large tree can hit the
      cap and rank HIGH (round 10, NIT-1) — the fail-closed side.

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
            sessions = os.path.join(state, _SESSIONS_DIRNAME)
            if (
                _under_evidence(real, sessions)
                and not any(_under_evidence(real, pad) for pad in prot.own_scratchpad)
                and not _in_session_scratchpad(real, sessions)
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


def _high_reason(path: str, target: Target, prot: _Protected, *, link: bool = False) -> str | None:
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
    if target.pattern is not None and not target.opaque:
        git_dir = _glob_reaches_git_dir(target.pattern)
        if git_dir is not None:
            return git_dir
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
        if link and os.path.dirname(path) == home:
            # ``link_entry`` proved only the link goes; say so, so a refusal
            # does not read as the pointee directory being deleted.
            return "a symlink at the top level of the home directory; only the link is removed"
        if os.path.dirname(path) == home and not os.path.isfile(path):
            return "a top-level directory of the home directory"
    try:
        if os.path.isdir(path) and os.path.exists(os.path.join(path, ".git")):
            return "a git repository root"
        # The repository's history itself: deleting ``<repo>/.git`` loses every
        # unpushed commit, stash and linked-worktree record just as deleting
        # the root does, so it ranks with the root. Only a git DIRECTORY (a
        # linked worktree's ``.git`` is a one-line pointer file).
        if _is_git_dir(path):
            return _GIT_DIR_REASON
    except OSError:
        return "an unreadable path"
    return None


#: Programs that remove a symlink operand's ENTRY without following it. Not
#: ``rmdir`` (fails on a link), ``shred`` (writes through it) or ``find``.
_LINK_REMOVERS = frozenset({"rm", "grm", "unlink", "gunlink"})


def link_entry(target: Target) -> str | None:
    """The canonical link path when ``target`` removes a symlink ENTRY only, else ``None``.

    WHY LINK AND POINTEE DIFFER (q1029 incident, remote node 0.101.44). ``rm
    ~/q1029`` where ``q1029 -> <own scratchpad>`` unlinks the alias and
    leaves the scratchpad alone, yet the guard ranked and verified the
    POINTEE's realpath: the judge's citation of the ``ln -s`` call could
    never cover a scratchpad born long before it, so the session's own link
    was refused. Such a target is ranked and verified on the link entry.

    Narrow on purpose — anything that may traverse into the pointee keeps
    the pointee ranking: a trailing slash or ``.``/``..`` last component
    (``rm -r link/`` deletes the pointee's contents), a ``..`` anywhere
    (resolved physically), globs, opaque or empty expansions, an unknown cwd
    and programs other than ``rm``/``unlink``. POSIX ``rm -r`` does not
    follow a symlink operand, so ``-r``/``-f`` on the bare link are fine.
    """
    if target.op != "delete" or target.program not in _LINK_REMOVERS:
        return None
    if (
        target.pattern is not None
        or target.opaque
        or target.unknown_cwd
        or target.climbs
        or target.dotdot
        or target.empty
        or target.credential_derived
    ):
        return None
    raw = target.raw_path
    if not raw or raw.endswith("/") or os.path.basename(raw) in ("", ".", ".."):
        return None
    if not os.path.islink(raw):
        return None
    from local_operator.guardrails.provenance import link_path

    return link_path(raw)


def _link_spellings(path: str, link: str) -> set[str]:
    """``_spellings`` without following the link itself: the lexical path and
    the canonical link path (parent resolved), each re-cased from disk.

    Only the PARENT goes through ``_on_disk_case``: handed the link entry
    itself, it opens the path (``F_GETPATH``) or tests ``isdir``, and both
    follow the link, so the pointee's path leaked in as a "spelling" of the
    link. ``rm LINK`` never touches the pointee, and a pointee in another
    session's scratchpad made the refusal say "another session's files" on
    some hash seeds and the link reason on others (#3793's flake). The leaf
    is re-cased from the parent's listing, which never dereferences it.
    """
    out = {_norm(path), link}
    for spelled in list(out):
        parent, name = os.path.split(spelled)
        cased_parent = _on_disk_case(parent) or parent
        try:
            names = os.listdir(cased_parent)
        except OSError:
            names = []
        if name not in names:
            name = next((n for n in names if n.casefold() == name.casefold()), name)
        out.add(os.path.join(cased_parent, name))
    return out


def _spelling_order(path: str, spellings: set[str]) -> list[str]:
    """``spellings`` in a stable order: the caller's own (normalised) spelling
    first, the resolved and re-cased forms after it, sorted.

    ``_rank`` returns the first spelling's reason, so iterating the bare set
    made the refusal text depend on ``PYTHONHASHSEED``; an agent reading a
    refusal must get the same answer for the same command. The lexical
    spelling leads because it names what the command actually addresses.
    """
    lexical = _norm(path)
    return sorted(spellings, key=lambda s: (s != lexical, s))


_GIT_DIR_REASON = "a git repository's .git directory (its history)"


def _is_git_dir(path: str) -> bool:
    """A repository's ``.git`` DIRECTORY (a linked worktree's ``.git`` is a
    one-line pointer file; a link is removed, not followed)."""
    return (
        os.path.basename(path).casefold() == ".git"
        and not os.path.islink(path)
        and os.path.isdir(path)
        and os.path.exists(os.path.join(path, "HEAD"))
    )


def _glob_reaches_git_dir(pattern: str) -> str | None:
    """``_GIT_DIR_REASON`` when the glob ``pattern`` matches a ``.git``
    directory, else ``None``.

    WHY. The ``.git`` rule in ``_high_reason`` judges one path, so
    ``rm -rf ~/repo/.gi*`` (or ``.g?t``, ``.[g]it``, ``cd ~/repo && rm -rf
    .g*``) named only the literal prefix ``~/repo`` and ranked LOW while the
    shell deletes the history (review R1-2, PR #3811). Each match of the
    existing bounded expansion is checked instead. Only a final component
    that can match a LEADING dot counts: bash's ``*``/``?`` never do without
    ``dotglob``, and treating ``rm -rf ~/repo/*`` as a ``.git`` delete would
    over-block ordinary tree cleanups. A leading ``[`` counts conservatively
    (``[.]git`` does not match in bash, but ``fnmatch`` says it does).
    Past the expansion bound this rule simply does not apply; the evidence
    checks own that fail-closed case.
    """
    last = pattern.rstrip("/").rsplit("/", 1)[-1]
    # Casefolded: ``.GI*`` names ``.git`` on a case-insensitive volume; the
    # on-disk matches below decide whether it really exists.
    if not last.startswith((".", "[")) or not fnmatch.fnmatchcase(".git", last.casefold()):
        return None
    matches = _expand_glob(pattern)
    for match, _linked in matches or ():
        try:
            if _is_git_dir(match):
                return _GIT_DIR_REASON
        except OSError:
            return "an unreadable path"
    return None


def _rank(target: Target, prot: _Protected, session_id: str = "") -> RankedTarget:
    if target.landings:
        # An unpinned ``..``: rank the landing its missing and re-linked parts
        # produce (``Target.landings``) as well as ``path``, and keep the worse.
        # On a tie the landing wins, so the judge and the refusal name the file
        # the kernel acts on rather than the lexical ``<pad>/config.yml``
        # (review R2-1, QA Q2-1 on PR #3811). ``raw_path`` is the landing too,
        # so ``_kernel_reason`` judges it, not the check-time lookup.
        readings = [
            _rank(dataclasses.replace(target, path=p, raw_path=p, landings=()), prot, session_id)
            for p in target.landings
        ]
        readings.append(_rank(dataclasses.replace(target, landings=()), prot, session_id))
        return max(readings, key=lambda r: r.risk)
    path = target.path
    notes: list[str] = list(target.empty)
    link = link_entry(target) if path is not None else None
    spellings = _link_spellings(path, link) if path is not None and link else None
    worktree_note = ""
    if target.climbs:
        # S2-5b: say where the ``..`` starts from, not that it is inside it.
        notes.append(f"climbs out of {target.path} by an unknown amount (unresolved)")
    elif target.opaque:
        notes.append("contains an unresolvable substitution or variable (opaque)")
    if target.unknown_cwd:
        notes.append("relative to a working directory that cannot be determined")
    if target.unpinned:
        notes.append(
            "a '..' follows a component that is missing or that this command re-links, "
            "so where it lands cannot be known (unresolved)"
        )
    if path is None:
        return RankedTarget(target, Risk.HIGH, "; ".join(notes) or "unresolved target")
    ordered = _spelling_order(path, spellings if spellings is not None else _spellings(path))
    # A link's OWN path that is protected (``~/.local-operator``, config.yml,
    # the install layout) is still CRITICAL here: those spellings are lexical.
    for spelling in ordered:
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
    in_own_pad = (
        any(_under(path, pad) for pad in prot.own_scratchpad)
        and (not target.opaque or target.opaque_contained)
        # ``path`` is only the lexical reading of an unpinned ``..``: the
        # missing component may become a link out of the pad (resolve.Target).
        and not target.unpinned
    )
    # R6: ``$(mktemp …)`` names a fresh temp entry — not "unknown" in the sense
    # that matters once CRITICAL ancestry was ruled out above.
    if target.temp:
        return RankedTarget(target, Risk.LOW, "fresh temporary path ($(mktemp))")
    if notes and not in_own_pad:
        return RankedTarget(target, Risk.HIGH, "; ".join(notes))
    if target.worktree_remove and target.pattern is None:
        # ``git worktree remove`` without --force: git refuses tracked or
        # untracked changes itself, but deletes IGNORED files and a detached
        # HEAD's unreferenced commits. ``worktree_state`` proves those absent
        # (only regenerable ignored output, HEAD on a branch or reachable from a
        # ref; a branch HEAD needs no pushed/merged test because the removal
        # keeps the branch ref — see ``worktree_state``). Dirty, locked, busy or
        # unknown ranks HIGH with that state named (the one fact the agent can
        # act on); not-linked takes the ordinary ranking below.
        from local_operator.guardrails.provenance import (
            worktree_activity,
            worktree_state,
        )

        # OWN FIRST (operator escalation 2026-11-11, session 316765a40496): a
        # worktree the ledger proves this session or its subagent created is
        # judged on what its removal can LOSE. Before, a ``dirty`` state —
        # here only ignored ``.expo/``/``.pnpm-store/`` git deletes anyway —
        # returned HIGH below before the ledger check was ever reached, and
        # the judge saw no ledger fact, so finished own trees were refused.
        own = _owned_by_session(target, path, prot, session_id)
        state = worktree_state(path, own=own is not None)
        if state != "not-linked":
            # Clean is not idle (operator incident 2026-11-09: a clean worktree
            # whose branch had an OPEN PR under QA was removed, deleting 34
            # tracked files from under it). An open PR is an unliftable refusal
            # at ANY state: an earlier answered ask naming the path was given
            # before that fact was on the table, so it cannot cover it. A
            # running owner or an unknown PR state only withholds the LOW, so
            # the judge sees those facts. ``worktree_activity`` explains both.
            activity = worktree_activity(path, session_id)
            facts = activity.facts()
            if activity.pr_state == "open":
                return RankedTarget(
                    target,
                    Risk.CRITICAL,
                    f"branch {activity.branch} has open {activity.pr_ref}; "
                    f"finish or close it first ({facts})",
                )
            idle = activity.pr_state == "none" and not activity.owners_running
            # OWN + CLEAN does not need the forge (operator escalation
            # 2026-11-11, session 99c51054bdbd): ``git worktree remove --force``
            # of its own clean tree for MERGED #1858 was refused twice, once
            # because "PR state unknown (gh lookup timed out)" sent it to the
            # judge and the judge then timed out (20 s). What the PR lookup
            # protects is OTHER people's work in the tree (the 2026-11-09
            # incident: a tree under QA); for a tree the ledger proves this
            # session created, with nothing uncommitted and no other running
            # owner, removal loses nothing (the branch ref survives) whatever
            # gh says. A KNOWN open PR still refuses above; only an unknown
            # state is waived, and only here. A dirty own tree, a foreign tree
            # and an unproven path keep the judge, and a judge timeout there
            # stays a refusal.
            own_idle = not activity.owners_running and activity.pr_state in ("none", "unknown")
            if own is not None and own_idle and state == "clean":
                # Nothing uncommitted, nobody else in it, no open PR: the
                # creator removing it loses nothing, ``--force`` or not.
                return RankedTarget(
                    target,
                    Risk.LOW,
                    f"{own[0]}; linked worktree with no uncommitted change ({facts})",
                )
            if own is not None and state == "dirty":
                # Real tracked/untracked work in an OWN tree: the judge decides
                # with the ledger row and the exact entries (``build_delete_case``
                # lists both); the reason names them for the agent too.
                return RankedTarget(
                    target,
                    Risk.HIGH,
                    f"{own[0]}; linked worktree state: dirty; {_changes_phrase(path)}; {facts}",
                )
            if idle and state == "clean" and not target.forced:
                return RankedTarget(target, Risk.LOW, f"a clean linked git worktree ({facts})")
            if idle and state == "stale-initializing" and target.force_count == 2:
                return RankedTarget(
                    target,
                    Risk.LOW,
                    f"a clean linked git worktree with a stale initializing lock ({facts})",
                )
            if not idle:
                # Straight to the judge, past the ledger-ownership LOW below:
                # someone else is in the tree, or nobody could say whether a
                # PR/MR is open (no GitHub/GitLab remote, gh/glab failed or timed out).
                return RankedTarget(
                    target,
                    Risk.HIGH,
                    f"a git repository root; linked worktree state: {state}; {facts}",
                )
            if state not in ("clean", "stale-initializing"):
                # QA round 3 (Q3-OBS): dirty, locked, busy, unknown or timeout
                # with no open PR and no owner used to append a note here and
                # fall through to the ordinary-path LOW below, which drops the
                # notes, so a sibling worktree outside any protected root
                # holding uncommitted work ranked LOW even with ``--force``,
                # which deletes that work. Only a clean tree (or a verified
                # stale initializing lock) can lose nothing; every other state
                # goes to the judge with the state named.
                return RankedTarget(
                    target,
                    Risk.HIGH,
                    "; ".join(
                        [f"a git repository root; linked worktree state: {state}; {facts}", *notes]
                    ),
                )
            # Clean but forced, or stale-initializing without ``-f -f`` (git
            # refuses that itself): nothing uncommitted to lose, so the ordinary
            # ranking applies, but the state stays in whatever reason it gives.
            worktree_note = f"linked worktree state: {state}; {facts}"
            notes.append(worktree_note)
    # Every spelling is ranked, not just the first with a reason: the verdict
    # must not hinge on which spelling is looked at first. The most severe
    # outcome wins (the selecting-find exemption applies only when EVERY
    # spelling that has a reason qualifies for it); the reason text is the
    # first in ``ordered``, i.e. the lexical spelling's when it has one.
    reasons = [
        (spelling, reason)
        for spelling in ordered
        if (reason := _high_reason(spelling, target, prot, link=link is not None)) is not None
    ]
    if reasons:
        reason = reasons[0][1]
        if target.selected and all(
            not any(_under(spelling, st) for st in prot.state_roots) and _is_work_tree(spelling)
            for spelling, _ in reasons
        ):
            # S2-3: a find with a selecting predicate (``-name '*.pyc'``)
            # removes a SUBSET of a working tree — the R6 cleanup shape. It
            # overrides ONLY the working-tree reasons: a repo root, even one
            # at $HOME's top level (``~/<project>``). State-root children,
            # other sessions and non-repo home dirs (``~/Documents``) stay
            # HIGH, because their contents are the data a confused
            # predicate would destroy.
            return RankedTarget(target, Risk.LOW, "find with a selecting predicate")
        owned = _owned_by_session(target, path, prot, session_id)
        # Not for a link entry (#3793): ``rm <own link>`` removes only the
        # link, so the pointee's uncommitted work is never at stake.
        if owned is not None and target.pattern is None and link_entry(target) is None:
            # ``rm -rf <own linked worktree>`` is held to the same bar as
            # ``git worktree remove --force`` of it (2026-11-11): ignored
            # output never counts, real uncommitted work goes to the judge.
            from local_operator.guardrails.provenance import worktree_state

            spelling = reasons[0][0]
            if worktree_state(spelling, own=True) == "dirty":
                return RankedTarget(
                    target,
                    Risk.HIGH,
                    f"{owned[0]}; linked worktree state: dirty; {_changes_phrase(spelling)}",
                )
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
    if worktree_note:
        # Never discard the worktree state the branch above recorded.
        return RankedTarget(target, Risk.LOW, f"ordinary path; {worktree_note}")
    return RankedTarget(target, Risk.LOW, "ordinary path")


def _changes_phrase(path: str) -> str:
    from local_operator.guardrails.provenance import describe_changes

    return describe_changes(path)


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

    link = link_entry(target)
    if link is not None:
        if any(_under(link, st) for st in prot.state_roots):
            return None
        record = provenance.owned_link(link, session_id)
        if record is None:
            return None
        return f"a symlink {record.describe()}; only the link is removed", [path]

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
        pad_hits: set[str | None] = set()
        for t in writes:
            reason = _touch_reason(t, prot, globstar=globstar)
            if reason == OTHER_SCRATCHPAD:
                # Not evidence, but another session's files: every write there
                # — an append included, which ``_rank`` leaves LOW because it
                # destroys nothing — is HIGH, liftable by the judge or a
                # verified ask (review R1-MINOR-2). A delete/overwrite also
                # gets ``_rank``'s HIGH row; the (path, risk) dedup keeps one.
                # A redirection is both a target and a touched write: one row.
                if t.path not in pad_hits:
                    pad_hits.add(t.path)
                    evidence_hits.append(RankedTarget(t, Risk.HIGH, OTHER_SCRATCHPAD))
                continue
            if reason is not None:
                risk = Risk.HIGH if reason == GLOB_TOO_WIDE else Risk.CRITICAL
                evidence_hits.append(RankedTarget(t, risk, reason))
        critical = {h.target.path for h in evidence_hits}
        for t in touched:
            # A path already refused as a write needs no second, HIGH row.
            if t.op != "touch" or t.path in critical:
                continue
            reason = _touch_reason(t, prot, globstar=globstar)
            if reason is None:
                continue
            shown = f"{reason}, handed to {t.program or 'a program'}"
            if reason == OTHER_SCRATCHPAD:
                tree = _tree_read(t, prot, session_id)
                if tree is None:
                    continue  # LOW: a tracked script reading a teammate's file
                shown += tree
            evidence_hits.append(RankedTarget(t, Risk.HIGH, shown))
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
    # An alternate parse's word (GNU sed's reading of ``-i ''``) counts only
    # as CRITICAL: under the other reading it is the script, so a HIGH or LOW
    # row for it would invent a target (review M1 on PR #3816).
    ranked = [r for r in ranked if not (r.target.critical_only and r.risk is not Risk.CRITICAL)]
    evidence_hits = [
        h for h in evidence_hits if not (h.target.critical_only and h.risk is not Risk.CRITICAL)
    ]
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
    # The process chdir()s to ``cwd`` as given, so a ``..`` in it lands
    # physically (``abspath`` would collapse it lexically, hence the plain
    # join); unpinnable means the cwd is unknown (relative words HIGH).
    resolution = resolve_command(
        command,
        cwd=physical_path(os.path.join(os.getcwd(), os.path.expanduser(cwd))) if cwd else None,
        env=env,
        home=home,
    )
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
