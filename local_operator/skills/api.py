"""Public surface of the skills subsystem (stream C).

Session and tools import ONLY from here — the lazy-import contract the
session relies on (a missing/broken skills module must degrade to
no-skills, not crash startup). Re-exports the discovery model, the semantic
index, the embedding backends, the ``skill://`` resolver, the
``SkillResolver`` adapter factory, and the default root computation.
Root precedence: project-local beats global, native beats ecosystem.
``default_skill_roots`` walks up from cwd to the filesystem root collecting
``.local-operator/skills`` directories (regardless of ``$HOME``), appends
``~/.local-operator/skills``, then the wider ecosystem roots
(``~/.omp/agent/skills``, ``~/.claude/skills``, ``~/.codex/skills``,
``~/.agents/skills``) last — only the ones that exist, so a clean machine
scans nothing extra. ``LOCAL_OPERATOR_SKILL_EXTRA_ROOTS`` replaces that
ecosystem set (colon-separated absolute paths; an empty value disables it).
Earlier roots win name collisions in :func:`discover_skills`, which is what
makes native roots authoritative over imported ones.
"""

from __future__ import annotations

import os
import threading
import time
from collections.abc import Callable, Mapping, MutableMapping, Sequence
from pathlib import Path

from local_operator.skills.discovery import (
    Skill,
    diagnose_missing_skill,
    discover_skills,
    roots_fingerprint,
)
from local_operator.skills.embeddings import (
    ApiEmbedder,
    EmbeddingBackend,
    EmbeddingError,
    LocalEmbedder,
    default_backend_from_env,
)
from local_operator.skills.index import SkillIndex, render_block
from local_operator.skills.protocol import resolve_skill_url

__all__ = [
    "ApiEmbedder",
    "EmbeddingBackend",
    "EmbeddingError",
    "LocalEmbedder",
    "Skill",
    "SkillIndex",
    "default_backend_from_env",
    "default_skill_roots",
    "diagnose_missing_skill",
    "discover_skills",
    "make_skill_resolver",
    "roots_fingerprint",
    "render_block",
    "resolve_skill_url",
]

_SKILLS_SUBDIR = Path(".local-operator") / "skills"

#: Ecosystem skill directories scanned AFTER the native roots, widest first
#: is irrelevant — only existence and order matter, and order is fixed here
#: so two machines with the same installs compute the same root list.
_ECOSYSTEM_SKILL_SUBDIRS: tuple[Path, ...] = (
    Path(".omp") / "agent" / "skills",
    Path(".claude") / "skills",
    Path(".codex") / "skills",
    Path(".agents") / "skills",
)

#: Replaces the ecosystem set when set. Colon-separated absolute paths;
#: an empty value disables ecosystem scanning entirely (a user who wants
#: ONLY native roots gets exactly that).
_EXTRA_ROOTS_ENV = "LOCAL_OPERATOR_SKILL_EXTRA_ROOTS"


def _ecosystem_roots(home: Path) -> list[Path]:
    """Existing ecosystem roots, or the env override, native-root-free.

    Missing default roots are filtered here (not left for
    :func:`discover_skills` to skip) because this list is also the user-facing
    answer to "what am I scanning"; env-provided paths are filtered the same
    way so the override behaves like the set it replaces.
    """
    raw = os.environ.get(_EXTRA_ROOTS_ENV)
    if raw is None:
        candidates = [home / subdir for subdir in _ECOSYSTEM_SKILL_SUBDIRS]
    else:
        candidates = [Path(part).expanduser() for part in raw.split(":") if part.strip()]
    return [candidate for candidate in candidates if candidate.is_dir()]


def default_skill_roots(cwd: Path | None = None) -> list[Path]:
    """Compute the default discovery roots for a working directory.

    Walks up from ``cwd`` to the filesystem root collecting
    ``<dir>/.local-operator/skills`` — regardless of ``$HOME``, so a repo at
    ``/opt``, ``/srv`` or ``/Volumes`` still picks up its project-local
    roots — then appends ``~/.local-operator/skills``, then the ecosystem
    roots (see :data:`_ECOSYSTEM_SKILL_SUBDIRS` / :data:`_EXTRA_ROOTS_ENV`).
    Roots are deduped by realpath, first occurrence wins — the walk-up order
    gives project-local roots priority over global ones, and native roots
    priority over ecosystem ones, matching the collision rule in
    :func:`discover_skills`.
    """
    start = (cwd or Path.cwd()).resolve()
    home = Path.home().resolve()

    candidates: list[Path] = []
    # Walk cwd → filesystem root; every ancestor gets a candidate, so
    # project-local roots work even outside $HOME.
    current = start
    while True:
        candidates.append(current / _SKILLS_SUBDIR)
        if current.parent == current:  # reached "/"
            break
        current = current.parent
    # Global native root, then ecosystem roots: the deepest project root
    # wins collisions, and native beats imported.
    candidates.append(home / _SKILLS_SUBDIR)
    candidates.extend(_ecosystem_roots(home))

    seen: set[str] = set()
    roots: list[Path] = []
    for candidate in candidates:
        try:
            key = str(candidate) if not candidate.exists() else str(candidate.resolve())
        except OSError:
            key = str(candidate)
        if key in seen:
            continue
        seen.add(key)
        roots.append(candidate)
    return roots


#: Minimum seconds between two filesystem probes on the miss path. Bounds a
#: looping or hostile agent to one probe per second per session: a typo'd
#: skill name in a retry loop costs one 0.29 ms fingerprint, not one per read.
_REFRESH_COOLDOWN = 1.0

#: The prefix ``resolve_skill_url`` uses for an unresolvable NAME. It also
#: raises for traversal, dotfiles and unsafe child paths, and those must NOT
#: trigger a rescan -- otherwise a malformed URL drives filesystem work.
_UNKNOWN_PREFIX = "Unknown skill: "


class _RefreshState:
    """Mutable bookkeeping for the miss-path refresh, one instance per resolver.

    Guarded by ``lock``. Subagents run as asyncio tasks on a single loop and
    the resolver is synchronous, so the refresh is already atomic with respect
    to that loop -- but sessions are also built on other execution paths
    (``exec_worker.py``, ``exec_mode.py``), and this read-modify-write is not
    inherently thread-safe. The lock is uncontended in practice and removes the
    question permanently rather than leaving a reader to re-derive the GIL
    argument.
    """

    __slots__ = ("last_check", "fingerprint", "lock")

    def __init__(self) -> None:
        self.last_check: float = 0.0
        # None means "never probed": distinct from an empty tuple, which is a
        # real fingerprint for roots that exist and hold no skills.
        self.fingerprint: tuple[object, ...] | None = None
        self.lock = threading.Lock()


#: Outcome of a miss-path refresh attempt. ``COOLDOWN`` is distinct from
#: ``UNCHANGED`` because it governs whether the DIAGNOSTIC may run: inside the
#: cooldown the resolver does no filesystem work at all, so a looping agent is
#: bounded to one probe per second including diagnosis.
_COOLDOWN = "cooldown"
_UNCHANGED = "unchanged"
_REFRESHED = "refreshed"


def _refresh_if_changed(
    skills: MutableMapping[str, Skill],
    roots: Sequence[Path],
    state: _RefreshState,
) -> str:
    """Rescan the roots into ``skills`` when the tree changed.

    Returns which of :data:`_COOLDOWN`, :data:`_UNCHANGED` or
    :data:`_REFRESHED` happened, because the caller treats them differently.

    Three properties here are load-bearing:

    1. **``skills.update(...)``, never ``skills = {...}``.** The whole
       propagation story is that ``session_factory``, this closure and every
       subagent hold the SAME dict object (``harness/subagent.py`` passes the
       parent's resolver to the child verbatim). Rebinding the local name
       leaves every one of those holders looking at the old mapping, and it
       breaks children SILENTLY -- no exception, no test failure unless one is
       written for it. ``tests/unit/skills/test_api.py`` asserts liveness
       through a CHILD resolver for exactly this reason.
    2. **The refresh only ADDS; it never removes.** A skill deleted from disk
       stays in the mapping. Dropping it would let one agent's cleanup break a
       sibling child mid-read, and the stale entry already degrades gracefully:
       the read returns a clean ``[Errno 2] No such file or directory`` through
       the adapter's ``OSError`` catch. Growth-only is the safe direction.
    3. **The cooldown is checked before the fingerprint**, so the cheap probe
       is itself rate-limited rather than merely the expensive scan.
    """
    now = time.monotonic()
    with state.lock:
        if state.fingerprint is not None and now - state.last_check < _REFRESH_COOLDOWN:
            return _COOLDOWN
        state.last_check = now
        try:
            fingerprint = roots_fingerprint(roots)
        except OSError:
            # The fingerprint swallows OSError per-entry already; this is the
            # belt-and-braces case, and a resolver may never raise.
            return _UNCHANGED
        if fingerprint == state.fingerprint:
            return _UNCHANGED
        state.fingerprint = fingerprint
        try:
            discovered, _warnings = discover_skills(roots)
        except OSError:
            return _UNCHANGED
        # In place. See point 1 above before changing this line.
        skills.update({skill.name: skill for skill in discovered})
        return _REFRESHED


def make_skill_resolver(
    skills: Mapping[str, Skill],
    roots: Sequence[Path] | None = None,
) -> Callable[[str], str | None]:
    """Build the ``SkillResolver`` adapter for the ``read`` tool.

    Contract (docs/REWRITE.md §C, ``harness/types.py``
    ``resolve_internal_url``): a resolver returns content for skill URLs,
    ``None`` for non-skill URLs (caller chains other resolvers), and never
    raises. :func:`resolve_skill_url` itself raises ``ValueError`` for
    unknown names and unsafe paths; this adapter catches it and returns the
    message AS CONTENT, so the available-names list reaches the model as a
    clean tool result and it can self-correct with a retry.

    When ``roots`` is given, an unknown NAME additionally triggers a
    fingerprint-gated rescan, so a skill authored mid-session becomes readable
    without a restart -- in this session and in subagents already running,
    which inherit this exact closure. That was the defect: an agent wrote a
    valid skill and a subagent whose standing instructions named it got
    ``Unknown skill`` for the rest of the session.

    **Cost on the frequent path is one dict lookup.** Nothing above the
    ``resolve_skill_url`` call runs on a hit; the fingerprint (~0.29 ms) and
    the full scan (17-25 ms) are miss-path only, and the scan runs only when
    the fingerprint says the tree actually changed.

    ``roots=None`` is the default so every existing caller -- and the parallel
    guide resolver, whose resources are packaged and cannot change at runtime
    -- keeps today's behaviour exactly.
    """
    state = _RefreshState()
    # Narrowed ONCE here rather than re-tested inside the closure, so the hot
    # path carries no type checks. Only a MutableMapping can be refreshed in
    # place: callers pass a plain dict, the read-only ``Mapping`` annotation
    # stays for the resolver's contract, and a caller that genuinely passes an
    # immutable mapping degrades to today's behaviour instead of raising a
    # TypeError on an error path.
    refresh_roots: Sequence[Path] | None = None
    refresh_target: MutableMapping[str, Skill] | None = None
    if roots is not None and isinstance(skills, MutableMapping):
        refresh_roots = roots
        refresh_target = skills

    def resolver(url: str) -> str | None:
        if not url.startswith("skill://"):
            return None
        try:
            return resolve_skill_url(url, skills)
        except ValueError as exc:
            message = str(exc)
            if (
                refresh_roots is None
                or refresh_target is None
                or not message.startswith(_UNKNOWN_PREFIX)
            ):
                # Traversal, dotfile and unsafe-child-path errors land here and
                # must not cause a rescan.
                return message
            outcome = _refresh_if_changed(refresh_target, refresh_roots, state)
            if outcome == _COOLDOWN:
                # Bounded: no scan AND no diagnostic, so a retry loop cannot
                # turn a typo into repeated filesystem work.
                return message
            if outcome == _REFRESHED:
                try:
                    return resolve_skill_url(url, skills)
                except ValueError as retry_exc:
                    message = str(retry_exc)
                except OSError as retry_exc:
                    return str(retry_exc)
            # Still missing: say WHY, because the bare message covers four
            # distinct causes an agent cannot otherwise tell apart.
            reason = _diagnose(url, refresh_roots)
            return f"{message}\n{reason}" if reason else message
        except OSError as exc:
            # OSError: a SKILL.md deleted or chmod-000'd between discovery and
            # the read. The adapter's contract is "never raises".
            return str(exc)

    return resolver


def _diagnose(url: str, roots: Sequence[Path]) -> str | None:
    """Best-effort explanation for a surviving miss; never raises.

    Diagnosis is a convenience on an error path, so any failure inside it must
    degrade to the plain ``Unknown skill`` message rather than replacing one
    unhelpful result with a traceback.
    """
    from urllib.parse import unquote, urlsplit

    try:
        name = unquote(urlsplit(url).netloc)
        return diagnose_missing_skill(name, roots)
    except Exception:  # noqa: BLE001 -- a diagnostic may never break the read
        return None
