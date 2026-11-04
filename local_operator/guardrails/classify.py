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
import os
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field

from local_operator.guardrails.resolve import Target, resolve_command

_STATE_DIRNAME = ".local-operator"
_CONFIG_FILE = "config.yml"
_SECRETS_DIRNAME = "secrets"
_SESSIONS_DIRNAME = "sessions"
_DESTROYING_OPS = frozenset({"delete", "move", "clean"})


class Risk(enum.IntEnum):
    LOW = 0
    HIGH = 1
    CRITICAL = 2


@dataclass
class RankedTarget:
    target: Target
    risk: Risk
    reason: str

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


def _critical_reason(path: str, target: Target, prot: _Protected) -> str | None:
    destroying = target.op in _DESTROYING_OPS
    for cfg in prot.config_files:
        if path == cfg:
            return "the harness config file"
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
    return None


def _critical_by_pattern(target: Target, prot: _Protected) -> str | None:
    if target.pattern is None:
        return None
    candidates = [
        *prot.config_files,
        *prot.secrets_dirs,
        *(prot.state_roots if target.op in _DESTROYING_OPS else ()),
        *(prot.install_roots if target.op in _DESTROYING_OPS else ()),
        *(prot.homes if target.op in _DESTROYING_OPS else ()),
    ]
    for protected in candidates:
        if _pattern_reaches(target.pattern, protected):
            return f"glob {target.pattern} matches {protected}"
    return None


def _high_reason(path: str, target: Target, prot: _Protected) -> str | None:
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


def _rank(target: Target, prot: _Protected) -> RankedTarget:
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
            return RankedTarget(target, Risk.HIGH, "; ".join([reason, *notes]))
    if target.selected:
        return RankedTarget(target, Risk.LOW, "find with a selecting predicate")
    return RankedTarget(target, Risk.LOW, "ordinary path")


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
) -> Verdict:
    if not targets and not opaque_reasons:
        return Verdict(Risk.LOW)
    prot = _protected(env, scratchpad, session_id)
    ranked = [_rank(t, prot) for t in targets]
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
    )
