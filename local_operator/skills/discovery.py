"""Skill discovery: scan skill directories under each root into ``Skill`` records.

On-disk format is identical to the wider Agent Skills ecosystem:
a skill is a directory containing a ``SKILL.md`` whose YAML frontmatter
carries ``name``, ``description``, ``enabled``, ``hide``, the Agent
Skills-standard ``disable-model-invocation``, and the optional
gitignore-style ``globs`` (path-based force-include, see
:meth:`local_operator.skills.index.SkillIndex.select`). The body is never
read here — it is fetched on demand through the ``skill://`` protocol, so
discovery stays cheap regardless of skill size.

Deliberate divergences from that ecosystem (see docs/REWRITE.md §C):

- Selection is semantic, done by :mod:`local_operator.skills.index`, so the
  description is the *routing signal* rather than guaranteed context.
- Roots are just the walk-up ``.local-operator/skills`` dirs plus the home
  root (see :func:`local_operator.skills.api.default_skill_roots`).
- Each root is walked to a BOUNDED depth (default 3, see
  :func:`skill_max_depth`), so a grouped library
  (``<root>/<group>/<skill>/SKILL.md``) is discovered. The walk treats a
  directory that contains a ``SKILL.md`` as a BOUNDARY: it is registered and
  never descended into, so a skill's own ``references/``, ``scripts/``,
  ``assets/`` or a vendored source tree can never leak further ``SKILL.md``
  files in as independent skills (the failure mode reported against Codex in
  openai/codex#22275). :func:`scan_skills_dir` and :func:`roots_fingerprint`
  share one walk primitive (:func:`_walk_skill_dirs`), so what the scanner
  sees and what the cache-invalidation fingerprint watches cannot drift.

Invariants maintained for prompt-cache stability:

- Output order is deterministic: ``(name.lower(), name, file_path)``.
- Name collisions resolve to the EARLIEST root; losers are dropped with a
  warning rather than silently shadowed. Within ONE root the shallower skill
  directory wins, and equal depths fall back to walk order (the walk is
  breadth-first with each directory's children sorted by name).
- The same physical ``SKILL.md`` is never loaded twice, even when two roots
  or symlinks point at it (realpath dedupe).
"""

from __future__ import annotations

import os
import stat as stat_module
from collections.abc import Iterator, Sequence
from pathlib import Path, PureWindowsPath
from typing import Literal

import yaml
from pydantic import BaseModel, ConfigDict


class Skill(BaseModel):
    """A semantically routable, progressively disclosed knowledge resource.

    User-authored records are ``resource_type="skill"``. Packaged harness
    guides use ``resource_type="guide"`` so both kinds share one vector index
    without conflating their prompt tags or URL schemes.
    ``base_dir`` contains all paths reachable beneath the resource's protocol
    URL; ``file_path`` is the body returned for a bare URL.
    """

    model_config = ConfigDict(frozen=True)

    name: str
    description: str
    file_path: Path
    base_dir: Path
    source: str
    hide: bool = False
    #: Gitignore-style path patterns from the ``globs`` frontmatter key
    #: (CSV string or YAML list). A pattern matching the session cwd or a
    #: file-path-like token of the selection query force-includes the skill
    #: regardless of the cosine threshold — the skill author saying "this
    #: one is about these paths" outranks an embedder's guess.
    globs: tuple[str, ...] = ()
    resource_type: Literal["skill", "guide", "agent_hint"] = "skill"


def parse_frontmatter(text: str) -> dict[str, object]:
    """Parse the leading ``---`` YAML block of a SKILL.md.

    Returns ``{}`` when there is no frontmatter, the block is unterminated,
    or the YAML is malformed — a broken frontmatter must degrade to
    "missing description" (skill dropped) rather than crash startup.
    """
    if not text.startswith("---"):
        return {}
    lines = text.split("\n")
    end: int | None = None
    for i in range(1, len(lines)):
        if lines[i].strip() == "---":
            end = i
            break
    if end is None:
        return {}
    try:
        data = yaml.safe_load("\n".join(lines[1:end]))
    except yaml.YAMLError:
        return {}
    return data if isinstance(data, dict) else {}


def _skill_from_file(skill_md: Path, base_dir: Path, source: str) -> Skill | None:
    """Build one Skill from a SKILL.md, or None when it must be skipped.

    Drop rules: ``enabled: false`` skips entirely; a missing or
    blank description is dropped silently (the description is the whole
    routing signal, an unnamed one cannot be selected). ``hide`` is the OR of
    ``hide`` and ``disable-model-invocation``.
    """
    try:
        text = skill_md.read_text(encoding="utf-8", errors="replace")
    except OSError:
        return None
    meta = parse_frontmatter(text)

    # Authors write enabled: 0 or enabled: "false" expecting a disabled
    # skill; identity-comparing against the False singleton kept those
    # enabled. Any explicitly-present falsy value (and the string spellings)
    # disables.
    enabled = meta.get("enabled")
    if enabled is not None and (
        not enabled or str(enabled).strip().lower() in ("false", "no", "off")
    ):
        return None

    description = meta.get("description")
    if not isinstance(description, str) or not description.strip():
        return None

    name = meta.get("name")
    if not isinstance(name, str) or not name.strip():
        name = base_dir.name
    else:
        name = name.strip()

    hide = bool(meta.get("hide")) or bool(meta.get("disable-model-invocation"))
    return Skill(
        name=name,
        description=description.strip(),
        file_path=skill_md,
        base_dir=base_dir,
        source=source,
        hide=hide,
        globs=_parse_globs(meta.get("globs")),
    )


def _parse_globs(raw: object) -> tuple[str, ...]:
    """Normalize the ``globs`` frontmatter key to a tuple of patterns.

    The ecosystem writes either a CSV string (``"*.py,src/**/*.ts"``) or a
    YAML list; both are accepted. Non-string list entries and empty items
    are dropped rather than rejected — a stray ``globs: [true]`` must not
    cost the user an otherwise-good skill, and an empty pattern can never
    match anything usefully. Any other shape (a bare int, a mapping) is
    ignored entirely.
    """
    if isinstance(raw, str):
        items: Sequence[object] = raw.split(",")
    elif isinstance(raw, (list, tuple)):
        items = raw
    else:
        return ()
    return tuple(item.strip() for item in items if isinstance(item, str) and item.strip())


#: Environment override for how many path segments below a root a skill
#: directory may sit (``1`` = today's flat ``<root>/<skill>/SKILL.md`` only).
#: Read at CALL time rather than import time so a running session and tests
#: see edits, and so a bad value can never break an import. This is an env var
#: and not a ``/settings`` key on purpose: it follows the precedent of
#: ``LOCAL_OPERATOR_SKILL_EXTRA_ROOTS`` (the subsystem's established knob), and
#: a walk-shape switch that almost nobody needs is not worth a config schema.
SKILL_MAX_DEPTH_ENV = "LOCAL_OPERATOR_SKILL_MAX_DEPTH"

#: ``root/group/skill`` plus one further grouping level: generous enough for
#: real libraries (vendor/product/skill) while keeping the walk -- which the
#: fingerprint repeats on EVERY resolver miss -- cheap and the blast radius of
#: a stray vendored tree small.
_DEFAULT_MAX_DEPTH = 3

#: Clamp range. The floor of 1 is today's flat behaviour. The ceiling exists
#: because each extra level multiplies the directories a miss must stat, and
#: because the deeper a ``SKILL.md`` sits below a root the likelier it is
#: vendored noise rather than an authored skill; 5 is already far past any
#: layout observed in the wider ecosystem.
_MIN_MAX_DEPTH = 1
_MAX_MAX_DEPTH = 5

_SKILL_FILE = "SKILL.md"


def skill_max_depth() -> int:
    """The active depth cap: env override clamped to [1, 5], else the default.

    Never raises: a blank, non-numeric or otherwise invalid value degrades to
    the default rather than costing the operator skill discovery over a typo.
    """
    raw = os.environ.get(SKILL_MAX_DEPTH_ENV)
    if raw is None or not raw.strip():
        return _DEFAULT_MAX_DEPTH
    try:
        value = int(raw.strip())
    except ValueError:
        return _DEFAULT_MAX_DEPTH
    return max(_MIN_MAX_DEPTH, min(_MAX_MAX_DEPTH, value))


def _is_ancestor_or_self(candidate: str, path: str) -> bool:
    """Whether the realpath ``candidate`` is ``path`` itself or an ancestor of it.

    Pure string work on already-resolved paths (``os.path.realpath`` upstream,
    never ``Path.resolve`` -- that raises ``RuntimeError`` on ELOOP on 3.12, the
    reason this module has always used the ``os.path`` spelling). ``relative_to``
    is what makes the ``/`` root case fall out for free, where a
    ``startswith(candidate + os.sep)`` test would not.

    A path that cannot be made relative to the candidate is not under it, so the
    ``ValueError`` is the ordinary "not an ancestor" answer rather than an error.
    """
    try:
        Path(path).relative_to(candidate)
    except ValueError:
        return False
    return True


def _walk_skill_dirs(
    root: str | os.PathLike[str], max_depth: int
) -> Iterator[tuple[str, os.stat_result]]:
    """Yield ``(skill_dir, SKILL.md stat)`` for every skill directory under ``root``.

    THE one walk both :func:`scan_skills_dir` and :func:`roots_fingerprint`
    consume, so the scanner and the cache-invalidation gate cannot disagree
    about which directories are skills; the fingerprint reuses the stat the
    walk already paid for.

    Rules, in the order they matter:

    * **A directory containing ``SKILL.md`` is a boundary.** It is yielded (when
      the file is a regular file) and NEVER descended into, whether or not the
      skill later loads -- a skill dropped for a blank description is still a
      boundary, otherwise its ``references/`` would leak skills the moment its
      frontmatter broke. A ``SKILL.md`` that exists but cannot be stat'd
      (dangling symlink, permissions) is likewise a boundary that is not
      yielded, for the same reason.
    * A directory WITHOUT one is a grouping folder and is descended into while
      a skill inside it could still fall within ``max_depth``.
    * A skill's depth is its path segments below ``root`` (direct child = 1).
    * Dot-prefixed entries are skipped at every level; symlinked directories
      are followed, so a linked-in library still registers. A symlinked child
      that resolves to the current directory or to one of its ancestors is NOT
      descended into (see below).
    * Breadth-first, each directory's children sorted by name. That order is
      the within-root collision precedence: shallower first, then walk order.
    * Never raises. A directory that vanishes or cannot be listed simply
      contributes nothing, matching what the scanner always tolerated.

    WHY the ancestor cut, on top of the depth cap: ``max_depth`` bounds how DEEP
    the walk goes, never how MUCH work it does, and a symlink back up to an
    ancestor makes every level below it re-walk the whole subtree. Measured here
    on 20 sibling groups each holding a ``loop -> <root>`` symlink, cap 5: 22
    fingerprint entries in 8.5 s (scan 4.4 s) before the cut, against 2 entries
    in 4 ms (scan 11 ms) after -- for ONE real skill, and on the per-message
    path (agent review round 1, MINOR-3, measured 421 entries / 198 ms on its
    own tree shape). The
    cut is applied to DESCENT only: such a child is still stat'd and yielded
    when it holds a ``SKILL.md``, so a linked-in skill stays visible; what stops
    is the re-walk. A symlink that is not an ancestor (a linked-in library) is
    followed exactly as before.
    """
    # ``(path, realpath)`` pairs. The realpath is carried rather than resolved
    # per directory: a child of a walked directory is at ``<parent-real>/<name>``
    # by construction, so descending costs no extra syscall -- only a SYMLINKED
    # child is resolved, and only to decide whether it loops back up.
    level: list[tuple[str, str]] = [(os.fspath(root), os.path.realpath(root))]
    for depth in range(1, max_depth + 1):
        next_level: list[tuple[str, str]] = []
        for parent, parent_real in level:
            try:
                with os.scandir(parent) as it:
                    children = sorted(it, key=lambda entry: entry.name)
            except OSError:
                continue
            for child in children:
                if child.name.startswith("."):
                    continue
                try:
                    if not child.is_dir():
                        continue
                except OSError:
                    continue
                child_real = os.path.join(parent_real, child.name)
                revisit = False
                try:
                    # ``DirEntry.is_symlink``, never ``os.path.islink(child.path)``:
                    # the link bit is already in the directory entry, so this asks
                    # the kernel NOTHING, and this walk runs on the per-message
                    # fingerprint path. Only a child that IS a link is resolved.
                    if child.is_symlink():
                        child_real = os.path.realpath(child.path)
                        # Ancestor-or-self: descending would re-walk the very
                        # tree this walk is already inside.
                        revisit = _is_ancestor_or_self(child_real, parent_real)
                except OSError:
                    revisit = True  # unresolvable: treat as a loop, do not descend
                skill_md = os.path.join(child.path, _SKILL_FILE)
                try:
                    st = os.stat(skill_md)
                except FileNotFoundError:
                    # Absent -> grouping folder. A DANGLING symlink named
                    # SKILL.md also lands here; it is a claim to be a skill dir,
                    # so it stays a boundary.
                    if os.path.lexists(skill_md):
                        continue
                    if depth < max_depth and not revisit:
                        next_level.append((child.path, child_real))
                    continue
                except OSError:
                    continue
                if stat_module.S_ISREG(st.st_mode):
                    yield child.path, st
                elif depth < max_depth and not revisit:
                    # A directory that merely happens to be named SKILL.md is
                    # not a skill file, so this folder is still just a group.
                    next_level.append((child.path, child_real))
        level = next_level
        if not level:
            return


def scan_skills_dir(
    dir: Path,
    source: str,
    include_self: bool = False,
    seen: set[str] | None = None,
) -> list[Skill]:
    """Scan one root to a bounded depth: ``<dir>[/<group>...]/<skill>/SKILL.md``.

    The depth cap is :func:`skill_max_depth` (default 3; ``1`` restores the
    flat ``<dir>/<child>/SKILL.md`` layout). A directory holding a ``SKILL.md``
    is registered and never descended into -- see :func:`_walk_skill_dirs` --
    so a skill's ``references/`` or vendored trees cannot register as skills.

    Result order is the walk order: shallower skills first, equal depths by
    breadth-first name order. That is also the within-root collision
    precedence, since :func:`discover_skills` lets the first of two same-named
    skills win. A flat root therefore comes back exactly as the one-level scan
    always produced it.

    Dot-prefixed entries are skipped; symlinked dirs are followed but deduped
    through ``seen`` (a set of realpaths shared across roots by
    :func:`discover_skills`) so one physical file is never loaded twice.
    ``include_self`` additionally accepts ``<dir>/SKILL.md`` (the wider
    ecosystem only uses this for Claude-plugin manifests; kept for parity).
    """
    skills: list[Skill] = []
    if seen is None:
        seen = set()

    def add(skill_md: Path, base_dir: Path) -> None:
        try:
            real = os.path.realpath(skill_md)
        except OSError:
            return
        if real in seen:
            return
        skill = _skill_from_file(skill_md, base_dir, source)
        if skill is None:
            return
        seen.add(real)
        skills.append(skill)

    if include_self:
        self_md = dir / _SKILL_FILE
        if self_md.is_file():
            add(self_md, dir)

    for skill_dir, _ in _walk_skill_dirs(dir, skill_max_depth()):
        base_dir = Path(skill_dir)
        add(base_dir / _SKILL_FILE, base_dir)

    return skills


def _sort_key(skill: Skill) -> tuple[str, str, str]:
    """Deterministic prompt order: case-insensitive name, exact name, path.

    Byte-stable ordering across turns keeps the volatile skills block from
    churning provider prompt caches (matching the established behavior).
    """
    return (skill.name.lower(), skill.name, str(skill.file_path))


def discover_skills(roots: Sequence[Path]) -> tuple[list[Skill], list[str]]:
    """Discover skills across roots; earlier roots win name collisions.

    Returns ``(skills, warnings)``. Warnings name every shadowed skill so a
    user who wonders why an edit to a home-root skill "did nothing" can see
    that a project-root skill of the same name took precedence. Missing
    roots are silently skipped (the home root may not exist yet).
    """
    seen: set[str] = set()
    warnings: list[str] = []
    ordered: list[Skill] = []
    for root in roots:
        root = Path(root).expanduser()
        if not root.is_dir():
            continue
        ordered.extend(scan_skills_dir(root, source=str(root), seen=seen))

    by_name: dict[str, Skill] = {}
    final: list[Skill] = []
    for skill in ordered:
        existing = by_name.get(skill.name)
        if existing is not None:
            # The loser's rule is named accurately, because the two are not
            # the same any more: across roots the earliest root wins, while
            # WITHIN one root the walk order decides and that order is
            # shallower-first (see :func:`scan_skills_dir`). Saying "earlier
            # root wins" for a same-root pair, which is what the bounded-depth
            # walk newly makes possible, would point the reader at a root list
            # that had nothing to do with it.
            same_root = skill.source == existing.source
            rule = "same root: the shallower skill wins" if same_root else "earlier root wins"
            warnings.append(
                f"Skill name conflict: '{skill.name}' from '{skill.file_path}' "
                f"shadowed by '{existing.name}' from '{existing.file_path}' "
                f"({rule})"
            )
            continue
        by_name[skill.name] = skill
        final.append(skill)

    final.sort(key=_sort_key)
    return final, warnings


def roots_fingerprint(roots: Sequence[Path]) -> tuple[object, ...]:
    """Cheap change-detector for the skill tree, used to gate a full rescan.

    Walks together with :func:`scan_skills_dir` BY CONSTRUCTION: both consume
    :func:`_walk_skill_dirs` at :func:`skill_max_depth`, so nested skills the
    scanner registers (and only those -- trees beyond the depth cap and
    anything under a skill boundary are outside both) are exactly the files
    whose ``(mtime_ns, size)`` this watches. A fingerprint that walked
    differently from the scanner would miss changes the scanner would have
    seen, and the entire value of this function is that an unchanged
    fingerprint is a trustworthy "do not bother scanning".

    Root-directory mtime alone is NOT sufficient, and that was measured rather
    than assumed: creating a skill directory bumps the root's mtime, creating a
    ``SKILL.md`` inside an existing directory bumps only the child's, and
    editing a ``SKILL.md`` in place bumps NEITHER. That third row is a real
    case, not a hypothetical -- a skill dropped at discovery for a blank
    ``description`` is absent from the mapping, so repairing its frontmatter in
    place is precisely a miss that root and child mtimes both fail to notice.
    Hence per-file ``(mtime_ns, size)``. Under a grouped library the same holds
    one level down: a grouping folder's own mtime is deliberately NOT recorded
    (a flat root's tuples must stay byte-identical), and does not need to be --
    every event that changes what the scanner registers (a skill added,
    removed, or a ``SKILL.md`` edited or created inside a group) changes some
    watched skill file's presence, mtime or size.

    Never raises, and its ``OSError`` tolerance deliberately matches the
    scanner's: a directory vanishing mid-walk or a symlink loop yields a
    shorter tuple rather than an exception. Making the gate stricter than the
    scanner it gates would turn a tree the scanner survives into a hard failure
    on the read path.

    Costs ~0.29 ms across 8 roots / 57 skills, against 17-25 ms for the full
    scan it decides whether to run.
    """
    entries: list[object] = []
    max_depth = skill_max_depth()
    for raw_root in roots:
        root = Path(raw_root).expanduser()
        try:
            root_stat = root.stat()
        except OSError:
            # Absent roots are still part of the fingerprint: a root that comes
            # into existence has to read as a change, not as "same as before".
            entries.append((str(root), None))
            continue
        entries.append((str(root), root_stat.st_mtime_ns))
        for skill_dir, skill_stat in _walk_skill_dirs(root, max_depth):
            entries.append((skill_dir, skill_stat.st_mtime_ns, skill_stat.st_size))
    return tuple(entries)


def _has_frontmatter_block(text: str) -> bool:
    """Whether ``text`` opens AND closes a ``---`` block.

    :func:`parse_frontmatter` collapses "no frontmatter", "unterminated",
    "malformed YAML" and "well-formed but empty" into the same ``{}``, which is
    the right degradation for discovery but the wrong answer for a diagnostic:
    reporting "malformed frontmatter" to an author whose block is perfectly
    well-formed and merely missing a ``description`` sends them to fix the one
    thing that is not broken. This distinguishes the structural failure from
    the content failure so each gets its own message.
    """
    if not text.startswith("---"):
        return False
    lines = text.split("\n")
    return any(lines[i].strip() == "---" for i in range(1, len(lines)))


def _frontmatter_yaml_error(text: str) -> str | None:
    """The YAML parser's own complaint about a delimited block, or ``None``.

    WHY this exists rather than reusing ``_has_frontmatter_block`` as the
    malformed-YAML discriminator: a block whose ``---`` delimiters are BOTH
    present but whose YAML is invalid is the overwhelmingly common authoring
    error -- an unquoted colon, ``description: Lean 4: formalize proofs`` --
    and it yields ``{}`` with the delimiters intact. Keying "malformed" off
    the delimiters alone therefore sent that author down the *description*
    branch, which told them to add a description their file visibly already
    had, and following that remedy provably does not fix the file. The only
    thing that can tell "invalid YAML" apart from "valid YAML that happens to
    be empty" is the parser, so ask it.

    :func:`parse_frontmatter` deliberately discards this exception -- discovery
    must degrade silently and never pay for error text on the scan path. Here,
    on a miss that has already failed, the re-parse costs nothing that matters
    and it is the difference between a next move and a wrong next move.
    """
    if not text.startswith("---"):
        return None
    lines = text.split("\n")
    end: int | None = None
    for i in range(1, len(lines)):
        if lines[i].strip() == "---":
            end = i
            break
    if end is None:
        # Unterminated: a structural failure, reported by the delimiter branch.
        return None
    try:
        yaml.safe_load("\n".join(lines[1:end]))
    except yaml.YAMLError as exc:
        # Collapse to one line: the parser appends a multi-line context/snippet
        # block that is noise inside a one-line diagnostic.
        return " ".join(str(exc).split())
    return None


# WHY a module constant instead of an inline ``os.name`` test: this is the seam
# both directions of the platform-divergent drive rule are tested through, so
# the POSIX-admits case and the Windows-denies case each run on every CI leg
# rather than only on the leg whose native semantics happen to match.
_DRIVE_RESETS_JOIN = os.name == "nt"


def is_plain_skill_name(name: str) -> bool:
    """Whether ``name`` is a single plain segment safe to join onto a root.

    The shape this admits is a skill NAME as :func:`scan_skills_dir` produces
    it: the frontmatter ``name:`` or the skill's own directory name, always a
    single segment. The bounded-depth walk finds skills in grouping folders
    (``<root>/<group>/<skill>``) but does not widen this grammar -- identity
    is the frontmatter name or the leaf directory name, never the group path
    -- so ``skill://<name>`` stays depth-independent and anything carrying a
    separator, a traversal token or a path anchor can never name a skill
    directory and must not be joined onto a root at all.

    NOT every registered skill name has this shape. :func:`_skill_from_file`
    prefers the frontmatter ``name:`` over the directory name, so a skill
    declaring ``name: group/sub`` registers under a separator-bearing name and
    is rejected here. That is deliberate and its cost is bounded: such a skill
    still resolves at startup and after any later rescan, and loses only the
    mid-session authoring refresh on the read that created it. Widening the
    predicate to admit it would hand every hostile netloc (``skill://..``,
    ``skill://%2fetc``) the filesystem work and the join this guard exists to
    deny, which is not a trade worth making for a name shape the harness
    itself never generates.

    WHY this is a name check and not a resolved-path containment check
    (``protocol._contained``): ``scan_skills_dir`` deliberately FOLLOWS a
    symlinked child directory, so a resolved-target containment test would
    reject skills the scanner accepts and make the diagnostic disagree with
    discovery. Once the name is a single plain segment, ``root / name`` is a
    direct child of ``root`` by construction and there is nothing left to
    escape with -- the check is complete without resolving anything, which is
    also what keeps an unsafe URL from spending filesystem work. Nested
    candidates are never built by joining the name anywhere: the diagnostic
    finds them by WALKING the roots and comparing each skill directory's leaf
    name to ``name``, so the name never becomes part of a path below the root.

    The concrete leaks this closes: ``skill://..`` parses the traversal token
    as the URL's netloc, sailing past the guards that only inspect the PATH
    portion, and ``skill://%2fetc`` decodes to ``/etc`` where ``root / "/etc"``
    is ``/etc`` outright -- an existence oracle for any absolute path, and a
    reader of out-of-root frontmatter ``name`` values.

    THE DRIVE RULE IS DELIBERATELY PLATFORM-DIVERGENT -- do not "simplify" it
    back to one unconditional test. What is rejected where, and why:

    * Separators, NUL, ``.`` and ``..``, and a ROOT anchor are rejected
      EVERYWHERE. They reset or escape the join on every platform.
    * A DRIVE anchor (``D:x``, ``a:b``, ``C:``, and -- since pathlib accepts
      any single printable character as a drive letter -- ``1:x`` and ``#:x``)
      is rejected ONLY when running on Windows.

    The asymmetry is correct because the join that actually executes is the
    RUNNING platform's, not a portable abstraction over both. On Windows
    ``PureWindowsPath(root) / "D:x"`` is ``"D:x"`` outright -- drive-relative
    names are not absolute by ``is_absolute()`` yet still reset the join, so
    the drive is a genuine existence oracle and that door must stay shut. On
    POSIX ``:`` is an ordinary, legal directory character and
    ``PurePosixPath(root) / "a:b"`` is ``<root>/a:b`` -- a contained direct
    child with nothing to escape through. Rejecting it there buys no safety
    and costs a real feature: ``scan_skills_dir`` genuinely registers a
    POSIX skill directory named ``a:b``, so an unconditional drive test makes
    this predicate disagree with the scanner, silently denying that skill the
    mid-session authoring refresh AND its miss-path diagnostic.
    """
    if not name or name in (".", ".."):
        return False
    if "/" in name or "\\" in name or "\x00" in name:
        return False
    # PureWindowsPath parses BOTH separators, so its ``root`` is the superset
    # of the POSIX one and covers either platform in a single test. Checked
    # unconditionally, and kept even though the separator test above already
    # implies it: a root anchor needs a separator, so this is the defence that
    # does not depend on that check having run first.
    shape = PureWindowsPath(name)
    if shape.root:
        return False
    # DRIVE, not anchor: the one term gated on the running platform, per the
    # docstring. ``drive`` rather than ``is_absolute()`` because a
    # drive-RELATIVE name resets the join without being absolute.
    return not (_DRIVE_RESETS_JOIN and shape.drive)


def _diagnose_one(name: str, root: Path, skill_md: Path) -> str | None:
    """Why this one SKILL.md would not load under ``name``, or None if it would.

    Applies exactly the drop rules :func:`_skill_from_file` applies silently,
    in the same order, so the message an author gets describes the rule that
    actually rejected their file.
    """
    try:
        text = skill_md.read_text(encoding="utf-8", errors="replace")
    except OSError as exc:
        return f"{skill_md} could not be read: {exc}"

    meta = parse_frontmatter(text)
    if not meta:
        if not _has_frontmatter_block(text):
            return f"{skill_md} has malformed YAML frontmatter (it must open and close with ---)."
        yaml_error = _frontmatter_yaml_error(text)
        if yaml_error is not None:
            # A well-formed pair of delimiters wrapping YAML that does not
            # parse. Quote the parser rather than guessing: "mapping values are
            # not allowed here" names the unquoted colon, which no rule of ours
            # can locate. A block that parses to nothing (an EMPTY one) falls
            # through instead -- "has no 'description'" is the true answer there.
            return (
                f"{skill_md} has invalid YAML in its frontmatter: {yaml_error}. "
                "Quote any value containing a colon, then read the URL again."
            )

    enabled = meta.get("enabled")
    if enabled is not None and (
        not enabled or str(enabled).strip().lower() in ("false", "no", "off")
    ):
        return f"'{name}' is disabled by 'enabled: false' in {skill_md}."

    description = meta.get("description")
    if not isinstance(description, str) or not description.strip():
        return (
            f"{skill_md} has no 'description' in its frontmatter; skills without one "
            "are not loaded. Add one and read the URL again."
        )

    declared = meta.get("name")
    if isinstance(declared, str) and declared.strip() and declared.strip() != name:
        return (
            f"{skill_md} declares name '{declared.strip()}'; read "
            f"skill://{declared.strip()} (the frontmatter name wins over the "
            "directory name)."
        )
    return None


def diagnose_missing_skill(name: str, roots: Sequence[Path]) -> str | None:
    """Explain why ``name`` did not load, or ``None`` when there is nothing to say.

    ``Unknown skill: <name>`` is emitted identically for four distinct causes --
    not yet scanned, a blank ``description`` silently dropping the skill, a
    frontmatter ``name`` disagreeing with the directory, and collision
    shadowing -- and an agent that hits it has no next move except to give up or
    grep the filesystem, which the system prompt forbids. Every message
    produced here names its own remedy instead.

    Called ONLY on the miss path, after a refresh has already failed to find
    the name, so the filesystem work below is never paid on a hit. It reports
    on the ONE name being read rather than on the tree as a whole: this machine
    emits 37 shadow warnings at startup, and a diagnostic that dumped them all
    would be noise where an answer is needed.
    """
    if not is_plain_skill_name(name):
        # BEFORE ANY FILESYSTEM WORK. ``root / name`` is unguarded join: a
        # traversal token or an absolute path in the NAME position escapes the
        # roots entirely, and the resolver's own path-portion guards never see
        # it because a name is the URL's netloc, not its path. Returning None
        # keeps the bare "Unknown skill" -- which is the honest answer for a
        # name no scan could ever have produced -- and, equally deliberately,
        # spends no probe on a malformed URL (design §3).
        return None
    max_depth = skill_max_depth()
    found: list[tuple[Path, Path]] = []
    for raw_root in roots:
        root = Path(raw_root).expanduser()
        candidate = root / name
        no_skill_md: str | None = None
        try:
            if candidate.is_dir():
                skill_md = candidate / _SKILL_FILE
                if skill_md.is_file():
                    found.append((root, skill_md))
                else:
                    no_skill_md = f"A directory '{name}' exists at {root} but has no SKILL.md."
        except OSError:
            pass
        # Nested candidates: a grouped library keeps the skill below a grouping
        # folder, where the join above cannot reach. Find them by walking the
        # same bounded tree discovery scans (shallow before deep, so the one an
        # author most likely means comes first) and matching the LEAF directory
        # name. Only directories that hold a SKILL.md are candidates: a grouping
        # folder that happens to share the name is not a broken skill, so it
        # earns no "has no SKILL.md" complaint -- that message stays reserved
        # for a direct child of a root, exactly as before, and is only given
        # when no nested skill of that name turned up in the same root.
        nested = [
            (root, Path(skill_dir) / _SKILL_FILE)
            for skill_dir, _ in _walk_skill_dirs(root, max_depth)
            if os.path.basename(skill_dir) == name and Path(skill_dir) != candidate
        ]
        if no_skill_md is not None and not nested:
            return no_skill_md
        found.extend(nested)

    if not found:
        return None

    # Walk in root precedence order and report the first candidate that has a
    # problem. Reaching here at all means the name genuinely missed, so no
    # candidate registered under it -- each one is either unloadable or
    # declares a different frontmatter name, and the earliest such file is the
    # one an author following root precedence will look at first.
    for root, skill_md in found:
        problem = _diagnose_one(name, root, skill_md)
        if problem is not None:
            return problem

    if len(found) > 1:
        # Every candidate loads cleanly on its own terms. On the resolver's
        # miss path this is defensive rather than routine -- if the earliest
        # copy loads, it also claims the name, so the read would have hit. It
        # stays because this helper is callable outside that path (against an
        # arbitrary root list) and because "shadowed" is the one collision
        # outcome a user can otherwise never see: the warning discover_skills
        # raises for it goes to a list printed only at startup.
        return (
            f"'{name}' at {found[1][1]} is shadowed by the one at {found[0][1]} "
            "(earlier roots win). Rename it or edit the winner."
        )
    return None
