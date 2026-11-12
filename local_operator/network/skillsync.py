"""USER SKILLS over the mesh: what makes a peer's ``skill://`` surface the same one.

WHY THIS EXISTS
===============

The operator's goal for a mesh node is that "a session on a mesh node takes on
work exactly like a local one: same team and agent briefs, skills,
credentials". Agent/team briefs (``definitions.py``), MCP servers
(``mcpdefs.py``) and memory (``memory.py``) already travel; user skills did not.
Discovery reads ``<config>/skills`` plus walk-up roots
(``skills/discovery.py``), and nothing in ``network/`` carried that directory, so
a brief that said "read ``skill://minerva-support-workspace``" resolved on the
head and failed on every peer. This module is the missing channel.

WHY NOT A SECTION INSIDE ``net_definitions``
============================================

The definitions bundle is text-only rows capped at ``MAX_BUNDLE_BYTES`` (2 MiB)
for the WHOLE roster; skills are multi-file trees with binaries. Measured on the
head (2026-11-11, names and sizes only): n=30 skills, deterministic-archive
median 11,718 B, largest ordinary skill 82,955 B, and one outlier
(``minerva-design``: 24,773,425 B, 141 files). One bundle cannot hold that, and
half-applying a mixed bundle is the failure ``mcpdefs`` already rejected. So:
its own op (``net_skills``), its own capability string (``wire.SKILLS_SYNC_V1``),
its own audit event, ONE SKILL PER FRAME, and the same cadence seam
(``definitions.add_tick_step``) MCP definitions ride — no new thread.

THE PACKAGING IS THE HUB'S UNTRUSTED-SAFE BUNDLE
=================================================

``hub_sync.skill_bundle`` already builds a deterministic, content-addressed zip
and validates/extracts one with its own extractor (no ``extractall``, no mode
from the archive, path grammar, collision and size checks, tree hash re-verified
after extraction). This module reuses it by import: the SENDER builds with
``executable_as_data=True`` (a ``+x`` script ships as data — the archive always
records 0644 — instead of the skill being refused whole; two of the head's 30
skills carry one), and the RECEIVER runs the unmodified validator, so a bundle
with a traversal path, a symlink or an exec bit in its entry modes is refused by
name and nothing is written. Received files land 0600, exactly like a hub pull:
remote content is never executable merely because it arrived.

THE SYNC FINGERPRINT IS ``tree_hash``
=====================================

The peer's ``state`` answers ``{name: tree_hash, origin, edited}`` for every
skill directory it holds; the sender sends only skills whose hash differs and
that the peer would accept. ``tree_hash`` (not the archive hash) because it is
recomputable from a directory and independent of deflate output.

THE PER-SKILL CAP
=================

:data:`MAX_SKILL_ARCHIVE_BYTES` is 4 MiB: ~50x the largest ordinary head skill
above, and bounded by the transport — a skill travels base64 in ONE JSON record,
and ``wire.MAX_RECORD_BYTES`` is 8 MiB (4 MiB x 4/3 = 5.6 MB plus envelope).
``minerva-design`` is the outlier the cap leaves out; it is NEVER dropped
silently: the push result, ``lop network skills state`` and ``lop network ready``
all name it with its size. Carrying it would need a chunked transfer, which is
follow-on scope.

THE CONFLICT POLICY (definitions', adopted whole)
=================================================

The provenance index gains a ``skills`` section in the same
``<config>/network/definitions.json`` (one index, one lock, disjoint sections).
*Authored wins*: a skill directory with no index entry is the node's own and is
never overwritten. *A mirror follows its own origin only*: a mirror from another
device, or one whose tree no longer hashes to what this device last landed
(``landed``), is kept and reported by name. *Idempotent*: a repeat is
``unchanged`` and writes nothing. *A deleted mirror is re-installed.* *Deletions
are not propagated* (a delete is indistinguishable from a partial view).

CREATE-TIME PUSH
================

A fresh spawn must see the skills on turn 1, so ``relay._ctl_peer_create``
calls :func:`push_for_create` before sending the create. It is BOUNDED by
:data:`CREATE_PUSH_BUDGET_S`: a large first sync must not stall a create, so a
push still running at the budget continues in the background and the receipt
says so.
"""

from __future__ import annotations

import base64
import binascii
import json
import logging
import os
import shutil
import threading
import time
import uuid
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Any, Callable, Mapping, Sequence

from local_operator.network import definitions, wire
from local_operator.network.types import MeshRefusal

if TYPE_CHECKING:  # pragma: no cover - typing only
    from local_operator.network.relay import RelayServer

logger = logging.getLogger(__name__)

#: Checked on arrival so a payload from a future or foreign producer is refused
#: BY NAME rather than partially applied (the definitions/mcpdefs rule).
BUNDLE_KIND = "lop.mesh.skill.v1"
BUNDLE_VERSION = 1

#: The provenance section inside ``<config>/network/definitions.json``.
INDEX_SECTION = "skills"

#: See "THE PER-SKILL CAP" in the module docstring for the measurement.
MAX_SKILL_ARCHIVE_BYTES = 4 * 1024 * 1024

#: Owner-side deadline for ``net_skills``: one apply validates and extracts up
#: to a 4 MiB archive and fsyncs every file, so it runs OFF the link's reader
#: (the definitions/mcpdefs reasoning).
SKILLS_OP_DEADLINE_S = 60.0

#: One request's budget from the requester's side: the peer's own deadline is
#: the outer bound, and a 5.6 MB record over a slow link needs more than the
#: 30 s the row-sized siblings use.
PUSH_TIMEOUT_S = 60.0

#: How long a create waits for the skills push before sending the create anyway.
#: Measured on the two-relay loopback rig (2026-11-11, PR evidence): a FIRST sync
#: of the head's 29 sendable skills (604,494 archive bytes, one frame each, every
#: file fsynced on arrival) took 2.9 s; a one-skill edit took 0.65 s. 10 s is
#: ~3.4x the first-sync measurement, room for a slower link, and still never lets
#: a large first sync hold a create for the push's full per-request budget — past
#: it the push continues in the background and the receipt says so.
CREATE_PUSH_BUDGET_S = 10.0

#: Last push outcome per peer, written by the sender for ``state``/``ready``.
RECEIPTS_NAME = "skills-sync.json"


def skills_root(root: Path) -> Path:
    """The user skill root discovery reads for a config dir (``discovery.py``)."""
    return root / "skills"


def _stage_root(root: Path) -> Path:
    # OUTSIDE ``skills/`` on purpose: a half-extracted tree must never be a
    # discoverable skill, and ``network/`` is on the same filesystem as
    # ``skills/`` (both under the config root), so the final rename is atomic.
    return root / "network" / "skill-stage"


def _assert_inside_root(path: Path, root: Path) -> None:
    """Refuse any write that resolves outside the root being served (mcpdefs' rule)."""
    resolved_root = Path(root).resolve()
    resolved = Path(path).resolve()
    if resolved != resolved_root and resolved_root not in resolved.parents:
        raise MeshRefusal(
            "bad_request", "a skill write resolved outside this device's config directory"
        )


# ---------------------------------------------------------------------------
# Measuring this device's skills
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class LocalSkill:
    """One directory under ``<config>/skills`` as the sync sees it."""

    name: str
    tree_hash: str
    #: ``ready`` (will travel), ``too_large`` or ``refused`` (will not, ``reason`` says why).
    status: str
    reason: str = ""
    size: int = 0
    data: bytes | None = None
    ref: Mapping[str, Any] | None = None
    description: str = ""


#: (root, name) -> (stat fingerprint, LocalSkill). A tick asks once per member
#: per minute, and rebuilding a skill's archive each time would re-read and
#: re-deflate every byte for an unchanged tree; the stat fingerprint (path, size,
#: mtime_ns, mode of every entry) changes whenever the content can have.
_CACHE: dict[tuple[str, str], tuple[tuple[Any, ...], LocalSkill]] = {}
_HASH_CACHE: dict[tuple[str, str], tuple[tuple[Any, ...], str]] = {}
_CACHE_GUARD = threading.Lock()


def _fingerprint(path: Path) -> tuple[Any, ...]:
    rows: list[tuple[Any, ...]] = []
    for current, dirs, files in os.walk(path, followlinks=False):
        dirs.sort()
        for name in sorted(files + [d for d in dirs]):
            full = os.path.join(current, name)
            try:
                info = os.lstat(full)
            except OSError:
                rows.append((full, "gone"))
                continue
            rows.append((full, info.st_size, info.st_mtime_ns, info.st_mode, info.st_ino))
    return tuple(rows)


def _skill_dirs(root: Path, *, with_links: bool = False) -> list[tuple[str, Path, bool]]:
    """``(name, path, is_symlink)`` per top-level entry discovery could read.

    A SYMLINKED skill directory is never followed by the sync (shipping a link's
    target would let a skill reach any file the link points at), but discovery
    DOES follow it (``skills/discovery.py``), so ``skill://x`` resolves here and
    not on a peer. ``with_links=True`` returns those entries flagged, so the
    SENDER can name them as "not synced (symlink)" instead of dropping them
    silently (QA round 1, Q-1); the receiver's ``state`` still lists real
    directories only.
    """
    base = skills_root(root)
    try:
        entries = sorted(os.scandir(base), key=lambda entry: entry.name)
    except OSError:
        return []
    found: list[tuple[str, Path, bool]] = []
    for entry in entries:
        if entry.name.startswith("."):
            continue  # discovery skips dot-entries; so does the sync
        try:
            if entry.is_symlink():
                if with_links and entry.is_dir(follow_symlinks=True):
                    found.append((entry.name, Path(entry.path), True))
                continue
            if not entry.is_dir(follow_symlinks=False):
                continue
        except OSError:
            continue
        found.append((entry.name, Path(entry.path), False))
    return found


def _frontmatter(skill_md: Path) -> Mapping[str, object]:
    """A SKILL.md's frontmatter, or ``{}`` when it cannot be read (bounded read)."""
    from local_operator.hub_sync.skill_bundle import MAX_SKILL_MD_BYTES
    from local_operator.skills.discovery import parse_frontmatter

    try:
        with open(skill_md, "rb") as stream:
            text = stream.read(MAX_SKILL_MD_BYTES).decode("utf-8", errors="replace")
        return parse_frontmatter(text)
    except Exception:  # noqa: BLE001 — a label lookup; the folder name stands in
        return {}


def _registered_name(skill_dir: Path) -> str:
    """The name discovery registers ``skill_dir`` under (frontmatter, else folder).

    Delegates the rule to discovery's own helper so the sync and the resolver
    cannot drift (review round 3, M1: a ``name``-less own skill registers under
    its folder name, and a frontmatter-only check never saw it).
    """
    from local_operator.skills.discovery import skill_name_from_meta

    return skill_name_from_meta(_frontmatter(skill_dir / "SKILL.md"), skill_dir)


# ---------------------------------------------------------------------------
# Discovery as the oracle: which of the receiver's skills a name resolves to
# ---------------------------------------------------------------------------
#
# THE INVARIANT: a peer copy must never shadow, or be shadowed by, a skill the
# receiver owns. Rounds 2 and 3 found the class one layout at a time (grouped,
# folder != name, ``name``-less, ecosystem roots, a stale mirror beside a later
# own skill), each because the sync asked its own question instead of the one
# discovery answers. So the sync now replays discovery's walk over the roots the
# receiver's sessions read, in the same order, with the same name rule:
#
# * ``<config>/skills`` (where mirrors land), then ``ecosystem_skill_roots()``
#   (``~/.omp/agent``, ``~/.claude``, ``~/.codex``, ``~/.agents`` or the
#   ``LOCAL_OPERATOR_SKILL_EXTRA_ROOTS`` override), deduped by realpath exactly
#   as ``default_skill_roots`` does.
# * NOT the cwd walk-up project roots: they outrank ``~/.local-operator/skills``
#   so the receiver's project skill wins anyway, and the relay cannot know a
#   session's cwd (review round 3, out of scope).
# * NOT the packaged builtins: release content is meant to be shadowed by user
#   content, on the sender as on the receiver.
#
# A declaration counts whether or not discovery would LOAD it (``enabled:
# false``, blank description): the conservative direction, deferred as round-3
# m1. Which one RESOLVES is decided by the first loadable one, as discovery does.


@dataclass(frozen=True)
class _Holder:
    """One skill directory that registers a given name on this device."""

    path: Path
    #: Discovery would load it (``_skill_from_file`` accepts it).
    live: bool


def _oracle_roots(root: Path) -> list[Path]:
    from local_operator.skills.api import ecosystem_skill_roots

    roots: list[Path] = []
    seen: set[str] = set()
    for candidate in [skills_root(root), *ecosystem_skill_roots()]:
        key = os.path.realpath(candidate)
        if key in seen:
            continue  # ``default_skill_roots``: first occurrence wins
        seen.add(key)
        roots.append(candidate)
    return roots


def _holders(root: Path) -> dict[str, list[_Holder]]:
    """Every name the receiver's discovery roots register, holders in walk order."""
    from local_operator.skills.discovery import (
        _skill_from_file,
        _walk_skill_dirs,
        skill_max_depth,
    )

    found: dict[str, list[_Holder]] = {}
    seen: set[str] = set()
    for base in _oracle_roots(root):
        try:
            walked = [Path(entry[0]) for entry in _walk_skill_dirs(base, skill_max_depth())]
        except Exception:  # noqa: BLE001 — the walk never raises; belt and braces
            continue
        for skill_dir in walked:
            skill_md = skill_dir / "SKILL.md"
            real = os.path.realpath(skill_md)
            if real in seen:
                continue  # one physical skill linked into two roots is one skill
            seen.add(real)
            live = _skill_from_file(skill_md, skill_dir, str(base)) is not None
            found.setdefault(_registered_name(skill_dir), []).append(_Holder(skill_dir, live))
    return found


@dataclass(frozen=True)
class _Ownership:
    """What the receiver's discovery says about ``name`` beyond ``skills/<name>``."""

    #: The receiver's own skill registering ``name`` outside the mirror slot.
    own: Path
    #: The mirror slot ``skills/<name>`` holds an untouched peer copy (Q-8).
    mirror: bool
    #: ...and that copy is what discovery resolves, shadowing ``own``.
    mirror_wins: bool


def _ownership(
    root: Path,
    name: str,
    holders: Mapping[str, list[_Holder]] | None = None,
    *,
    index: Mapping[str, Any] | None = None,
) -> _Ownership | None:
    """The receiver's own ``name`` outside ``skills/<name>``, or ``None``.

    Anything that registers ``name`` and is not the mirror slot itself is the
    node's own: a mirror is installed only at ``skills/<its name>`` with a
    matching frontmatter (``expected_name``), and an edit that renamed one is
    the node's work.
    """
    rows = (holders if holders is not None else _holders(root)).get(name) or []
    slot = skills_root(root) / name
    slot_real = os.path.realpath(slot / "SKILL.md")
    own = [holder for holder in rows if os.path.realpath(holder.path / "SKILL.md") != slot_real]
    if not own:
        return None
    # The own copy that matters is the one discovery would load, else the first.
    chosen = next((holder for holder in own if holder.live), own[0]).path
    recorded = _recorded(index if index is not None else definitions.read_index(root), name)
    current = _current_tree(slot)
    mirror = (
        recorded is not None
        and current is not None
        and current == str(recorded.get("landed") or "")
    )
    winner = next((holder for holder in rows if holder.live), None)
    mirror_wins = bool(
        mirror and winner is not None and os.path.realpath(winner.path / "SKILL.md") == slot_real
    )
    return _Ownership(own=chosen, mirror=mirror, mirror_wins=mirror_wins)


def _where(root: Path, path: Path) -> str:
    """``skills/<rel>`` inside the config root, else the path with ``~`` for home."""
    try:
        return "skills/" + path.relative_to(skills_root(root)).as_posix()
    except ValueError:
        pass
    home = str(Path.home())
    text = str(path)
    if text == home or text.startswith(home + os.sep):
        return "~" + text[len(home) :]
    return text


def _kept_reason(
    name: str, where: str, *, mirror: bool, mirror_wins: bool, label: str = "this device"
) -> str:
    """One sentence for a skill kept as the node's own, honest about a stale mirror.

    Shared by the receiver's ``apply`` answer and the sender's push text (built
    from ``skill_state``), so both say the same thing. Never "kept as its own"
    while the peer's earlier copy is what resolves (QA round 3, Q-8): the sync
    stops refreshing that copy, since the node's own skill now owns the name,
    but does not delete it. Deleting a file a running session may be reading is
    not clearly reversible, so the operator is told which file to remove.
    """
    if mirror_wins:
        return (
            f"{label} has its own '{name}' (in {where}), but an earlier synced copy at "
            f"skills/{name} still wins there; delete that copy to use its own"
        )
    if mirror:
        return (
            f"{label} has its own skill by that name (in {where}), so it was kept; an "
            f"earlier synced copy at skills/{name} is unused there and can be deleted"
        )
    return f"{label} has its own skill by that name (in {where}), so it was kept"


def _ownership_reason(root: Path, name: str, owned: _Ownership) -> str:
    return _kept_reason(
        name, _where(root, owned.own), mirror=owned.mirror, mirror_wins=owned.mirror_wins
    )


def _grouped_rows(group: str, path: Path, taken: frozenset[str] = frozenset()) -> list[LocalSkill]:
    """Name the skills inside a grouping folder, by the names discovery gives them.

    Discovery walks ``<root>/<group>/<skill>/SKILL.md`` (bounded depth), so a
    grouped skill resolves here as ``skill://<skill>``. The sync ships one
    top-level directory per skill and does not carry groups yet; before this the
    only signal was the FOLDER with the bundle builder's "exactly one root
    SKILL.md" error, which names nothing the operator asked for (QA round 1,
    Q-2). Uses discovery's own walk so the set named is the set that resolves.
    """
    from local_operator.skills.discovery import _walk_skill_dirs, skill_max_depth

    rows: list[LocalSkill] = []
    try:
        nested = [Path(found[0]) for found in _walk_skill_dirs(path, skill_max_depth() - 1)]
    except Exception:  # noqa: BLE001 — unreadable group: fall back to naming the folder
        nested = []
    for skill_dir in sorted(nested):
        name = _registered_name(skill_dir)
        if name in taken:
            # "Move it to skills/<name>" would point at a folder that is already
            # taken by the top-level skill that wins here (review round 2, R2-3).
            reason = (
                f"it sits in the grouping folder '{group}' and is shadowed here by "
                f"~/.local-operator/skills/{name}, which syncs under that name"
            )
        else:
            reason = (
                f"it sits in the grouping folder '{group}', and nested skill folders "
                f"don't sync yet; move it to ~/.local-operator/skills/{name} to sync it"
            )
        rows.append(LocalSkill(name=name, tree_hash="", status="refused", reason=reason))
    return rows


def _measure(name: str, path: Path) -> LocalSkill:
    from local_operator.hub_sync import skill_bundle
    from local_operator.hub_sync.skill_document import (
        SkillValidationError,
        validate_name,
    )

    try:
        validate_name(name)
    except SkillValidationError:
        return LocalSkill(
            name=name,
            tree_hash="",
            status="refused",
            reason="its directory name is not a lower-kebab-case skill name",
        )
    try:
        built = skill_bundle.build_bundle(path, executable_as_data=True)
    except skill_bundle.SkillBundleError as exc:
        status = "too_large" if exc.code == "bundle-too-large" else "refused"
        return LocalSkill(name=name, tree_hash="", status=status, reason=str(exc))
    if built.name != name:
        # DISCOVERY KEYS A SKILL BY ITS FRONTMATTER ``name``, not its folder
        # (``skills/discovery.py``), and the receiver installs by folder. A
        # mismatched skill would land under one name and register under another,
        # shadowing a skill the peer authored by that name without the conflict
        # policy ever being asked (review round 1, M1). The receiver refuses it
        # too (``expected_name``); naming it here says why before a frame is spent.
        return LocalSkill(
            name=name,
            tree_hash=built.tree_hash,
            status="refused",
            reason=(
                f"its SKILL.md names it '{built.name}', not its directory name; "
                "rename one so they match"
            ),
        )
    if built.size > MAX_SKILL_ARCHIVE_BYTES:
        return LocalSkill(
            name=name,
            tree_hash=built.tree_hash,
            status="too_large",
            reason=(
                f"its bundle is {built.size:,} bytes, over the "
                f"{MAX_SKILL_ARCHIVE_BYTES:,}-byte per-skill mesh cap"
            ),
            size=built.size,
        )
    return LocalSkill(
        name=name,
        tree_hash=built.tree_hash,
        status="ready",
        size=built.size,
        data=built.data,
        ref=built.bundle.to_json(),
        description=built.description,
    )


def local_skills(root: Path) -> list[LocalSkill]:
    """Every skill directory under ``<config>/skills``, measured (cached by stat)."""
    out: list[LocalSkill] = []
    entries = _skill_dirs(root, with_links=True)
    top_level = frozenset(
        name
        for name, path, is_link in entries
        if not is_link and os.path.lexists(path / "SKILL.md")
    )
    for name, path, is_link in entries:
        if is_link:
            out.append(
                LocalSkill(
                    name=name,
                    tree_hash="",
                    status="refused",
                    reason=(
                        "not synced (symlink): the sync never follows a link; copy the "
                        "skill into ~/.local-operator/skills to sync it"
                    ),
                )
            )
            continue
        if not os.path.lexists(path / "SKILL.md"):
            grouped = _grouped_rows(name, path, top_level)
            if grouped:
                out.extend(grouped)
                continue
        key = (str(root), name)
        fingerprint = _fingerprint(path)
        with _CACHE_GUARD:
            cached = _CACHE.get(key)
        if cached is not None and cached[0] == fingerprint:
            out.append(cached[1])
            continue
        measured = _measure(name, path)
        with _CACHE_GUARD:
            _CACHE[key] = (fingerprint, measured)
        out.append(measured)
    return out


def _tree_hash(path: Path) -> str:
    """The on-disk tree hash, or ``""`` when the tree cannot be hashed (cached by stat)."""
    from local_operator.hub_sync import skill_bundle

    key = (str(path.parent), path.name)
    fingerprint = _fingerprint(path)
    with _CACHE_GUARD:
        cached = _HASH_CACHE.get(key)
    if cached is not None and cached[0] == fingerprint:
        return cached[1]
    try:
        digest = skill_bundle.tree_hash_of(path, executable_as_data=True)
    except skill_bundle.SkillBundleError:
        digest = ""
    with _CACHE_GUARD:
        _HASH_CACHE[key] = (fingerprint, digest)
    return digest


def _recorded(index: Mapping[str, Any], name: str) -> Mapping[str, Any] | None:
    rows = index.get(INDEX_SECTION)
    entry = rows.get(name) if isinstance(rows, Mapping) else None
    return entry if isinstance(entry, Mapping) else None


def authored_skills(root: Path) -> list[LocalSkill]:
    """This device's OWN skills: everything but untouched mirrors.

    Mirrors are excluded for the reason ``mcpdefs._authored_locally`` gives: a
    device that re-sent what it mirrored would hand a third device a skill as if
    it had authored it, and two peers could echo one skill forever. An EDITED
    mirror is the node's work now, so it counts as authored (and its origin will
    keep its own copy, reported as kept).
    """
    index = definitions.read_index(root)
    out = []
    for skill in local_skills(root):
        recorded = _recorded(index, skill.name)
        if (
            recorded is not None
            and skill.tree_hash
            and (skill.tree_hash == str(recorded.get("landed") or ""))
        ):
            continue
        out.append(skill)
    return out


def skill_state(root: Path) -> dict[str, Any]:
    """What this device holds, per skill: tree hash, mirror origin, edited.

    The receiver's half of the exchange and the readiness fact. Names and hashes
    only; never content.
    """
    index = definitions.read_index(root)
    rows: dict[str, Any] = {}
    for name, path, _is_link in _skill_dirs(root):
        current = _tree_hash(path)
        recorded = _recorded(index, name)
        origin = str(recorded.get("origin") or "") if recorded is not None else ""
        rows[name] = {
            "tree_hash": current,
            "origin": origin,
            "edited": bool(recorded is not None and current != str(recorded.get("landed") or "")),
        }
    # DISCOVERY IS THE ORACLE (see "Discovery as the oracle" above): a name the
    # node registers anywhere outside ``skills/<name>`` (grouped, folder !=
    # name, ``name``-less, an ecosystem root) is reported as the node's own, so
    # the sender says "kept" exactly when ``apply`` would (review rounds 2-3,
    # R2-1 / M1 / M2; QA Q-6). It overrides a ``skills/<name>`` row, because the
    # apply refuses that name for the same reason; ``mirror``/``mirror_wins``
    # carry a stale peer copy still in the slot, so the text stays honest (Q-8).
    holders = _holders(root)
    for name in holders:
        owned = _ownership(root, name, holders, index=index)
        if owned is None:
            continue
        rows[name] = {
            "tree_hash": _tree_hash(owned.own),
            "origin": "",
            "edited": False,
            "where": _where(root, owned.own),
            "mirror": owned.mirror,
            "mirror_wins": owned.mirror_wins,
        }
    return {"kind": BUNDLE_KIND, "version": BUNDLE_VERSION, "skills": rows}


# ---------------------------------------------------------------------------
# Applying one skill (the receiver)
# ---------------------------------------------------------------------------


def _refused(name: str, reason: str) -> dict[str, Any]:
    return {"outcome": "refused", "name": name, "reason": reason}


def apply_skill(root: Path, payload: Mapping[str, Any], *, origin_device: str) -> dict[str, Any]:
    """Install one skill when the policy allows it; report it by name otherwise.

    Returns ``{"outcome": installed|updated|unchanged|conflicts|refused, "name",
    "reason"?}``. Never raises for a bad bundle: a refusal is an answer.
    """
    from local_operator.hub_sync import skill_bundle
    from local_operator.hub_sync.skill_document import (
        SkillValidationError,
        validate_name,
    )

    name = str(payload.get("name") or "")
    if payload.get("kind") != BUNDLE_KIND or payload.get("version") != BUNDLE_VERSION:
        return _refused(name, "it was sent in a skill format this device does not read")
    try:
        validate_name(name)
    except SkillValidationError:
        return _refused(name, "its name is not a lower-kebab-case skill name")
    try:
        ref = skill_bundle.SkillBundleRef.from_json(payload.get("bundle"))
    except skill_bundle.SkillBundleError as exc:
        return _refused(name, str(exc))
    if ref.size > MAX_SKILL_ARCHIVE_BYTES:
        return _refused(name, "its bundle is over the per-skill mesh cap")
    try:
        data = base64.b64decode(str(payload.get("data") or ""), validate=True)
    except (binascii.Error, ValueError):
        return _refused(name, "its archive is not valid base64")
    if len(data) != ref.size:
        return _refused(name, "its archive does not match its declared size")
    origin = str(origin_device or "")
    with definitions.definition_lock(root):
        return _apply_locked(root, name, ref, data, origin)


def _decide(
    recorded: Mapping[str, Any] | None, current: str | None, incoming: str, origin: str
) -> tuple[str, str]:
    """The conflict policy for one name: ``(bucket, reason)`` or ``("", "")`` to install."""
    if (
        recorded is not None
        and current is not None
        and str(recorded.get("digest") or "") == incoming
        and current == str(recorded.get("landed") or "")
    ):
        return "unchanged", ""
    if current is None:
        return "", ""  # absent, or a deleted mirror: (re)install
    if recorded is None:
        return "conflicts", "this device has its own skill by that name, so it was kept"
    if str(recorded.get("origin") or "") != origin:
        return "conflicts", (
            f"that skill here is a copy of {recorded.get('origin') or 'another device'}'s, "
            "so it was kept"
        )
    if current != str(recorded.get("landed") or ""):
        return "conflicts", "the copy of that skill here has local edits, so it was kept"
    return "", ""


def _current_tree(dest: Path) -> str | None:
    """``None`` when absent; ``"unreadable"`` for anything that is not a plain tree."""
    if not os.path.lexists(dest):
        return None
    if dest.is_symlink() or not dest.is_dir():
        return "unreadable"
    return _tree_hash(dest) or "unreadable"


def _apply_locked(root: Path, name: str, ref: Any, data: bytes, origin: str) -> dict[str, Any]:
    from local_operator.config_writer import config_write_lock
    from local_operator.hub_sync import skill_bundle

    dest = skills_root(root) / name
    _assert_inside_root(dest, root)
    index = definitions.read_index(root)
    recorded = _recorded(index, name)
    bucket, reason = _decide(recorded, _current_tree(dest), ref.tree_hash, origin)
    # "unchanged" is re-checked too: an identical stale mirror beside a later own
    # skill is still a clash the operator must hear about (Q-8).
    owned = _ownership(root, name, index=index) if bucket in ("", "unchanged") else None
    if owned is not None:
        bucket, reason = "conflicts", _ownership_reason(root, name, owned)
    if bucket:
        return {"outcome": bucket, "name": name, **({"reason": reason} if reason else {})}
    stage = _stage_root(root) / uuid.uuid4().hex
    _assert_inside_root(stage, root)
    try:
        stage.mkdir(parents=True, mode=0o700)
        archive = stage / "archive.zip"
        fd = os.open(
            archive, os.O_WRONLY | os.O_CREAT | os.O_EXCL | getattr(os, "O_NOFOLLOW", 0), 0o600
        )
        with os.fdopen(fd, "wb") as stream:
            stream.write(data)
        try:
            # The hub's own validator + extractor, unmodified: traversal paths,
            # symlinks, exec/setuid bits in entry modes, collisions and size lies
            # are refused here by name, before anything reaches ``skills/``.
            # ``expected_name``: the archive's SKILL.md must declare the name it
            # installs under, or discovery would register it as another skill
            # and shadow this device's own one (review round 1, M1).
            landed = skill_bundle.extract_archive(
                archive, stage / "tree", expected=ref, expected_name=name
            )
        except skill_bundle.SkillBundleError as exc:
            return _refused(name, str(exc))
        # THE SWAP HOLDS THE CONFIG-ROOT LOCK that ``managed_skill_read`` takes
        # for discovery, so a reader never sees the gap between the two renames.
        # The policy is re-decided inside it: an editor may have touched the
        # directory while the archive was being extracted.
        with config_write_lock(root):
            current = _current_tree(dest)
            bucket, reason = _decide(recorded, current, ref.tree_hash, origin)
            # Re-checked under the lock for the same reason as ``_decide``: an
            # editor may have created the other declaration meanwhile (R2-1).
            owned = _ownership(root, name, index=index) if bucket in ("", "unchanged") else None
            if owned is not None:
                bucket, reason = "conflicts", _ownership_reason(root, name, owned)
            if bucket:
                return {"outcome": bucket, "name": name, **({"reason": reason} if reason else {})}
            skills_root(root).mkdir(parents=True, exist_ok=True)
            if current is not None:
                os.replace(dest, stage / "previous")
            os.replace(stage / "tree", dest)
        index.setdefault(INDEX_SECTION, {})[name] = {
            "origin": origin,
            "digest": ref.tree_hash,
            "landed": landed,
            "name": name,
            "at": time.time(),
        }
        definitions.write_index(root, index)
        return {"outcome": "updated" if current is not None else "installed", "name": name}
    finally:
        shutil.rmtree(stage, ignore_errors=True)


# ---------------------------------------------------------------------------
# Pushing (the sender)
# ---------------------------------------------------------------------------

_PUSH_LOCKS: dict[tuple[str, str], threading.Lock] = {}
_PUSH_LOCKS_GUARD = threading.Lock()


def _push_lock(root: Path, device_id: str) -> threading.Lock:
    # ONE PUSH PER PEER AT A TIME: the create path's background push and the
    # cadence can target the same member, and two concurrent pushes would send
    # the same multi-megabyte bundle twice for one "unchanged" on the far side.
    with _PUSH_LOCKS_GUARD:
        return _PUSH_LOCKS.setdefault((str(root), device_id), threading.Lock())


def _receipts_path(root: Path) -> Path:
    return root / "network" / RECEIPTS_NAME


_RECEIPTS_GUARD = threading.Lock()


def read_receipts(root: Path) -> dict[str, Any]:
    try:
        payload = json.loads(_receipts_path(root).read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return {}
    return payload if isinstance(payload, dict) else {}


def _record_receipt(root: Path, device_id: str, result: Mapping[str, Any]) -> None:
    """Best-effort: a record is not a gate (the mcpdefs rule). Names only."""
    try:
        with _RECEIPTS_GUARD:
            receipts = read_receipts(root)
            receipts[device_id] = {
                "at": time.time(),
                **{
                    key: result.get(key)
                    for key in (
                        "name",
                        "ok",
                        "code",
                        "message",
                        "installed",
                        "updated",
                        "kept",
                        "skipped",
                        "refused",
                    )
                    if key in result
                },
            }
            path = _receipts_path(root)
            path.parent.mkdir(parents=True, exist_ok=True)
            temporary = path.with_name(f".{path.name}.{os.getpid()}.{threading.get_ident()}.tmp")
            temporary.write_text(json.dumps(receipts, sort_keys=True), encoding="utf-8")
            os.chmod(temporary, 0o600)
            os.replace(temporary, path)
    except Exception:  # noqa: BLE001
        logger.debug("skillsync: could not record the push", exc_info=True)


def _skipped_rows(skills: list[LocalSkill]) -> list[dict[str, Any]]:
    return [
        {"name": skill.name, "status": skill.status, "reason": skill.reason}
        for skill in skills
        if skill.status != "ready"
    ]


def push_to_peer(server: "RelayServer", device_id: str) -> dict[str, Any]:
    """Reconcile this device's own skills onto ``device_id``, one frame per skill.

    ONE FRAME PER SKILL so each record stays under ``wire.MAX_RECORD_BYTES`` and a
    refusal names exactly one skill; the peer's ``state`` decides what is sent, so
    a peer already in sync costs one round trip and writes nothing.
    """
    with _push_lock(server.root, device_id):
        result = _push_locked(server, device_id)
    _record_receipt(server.root, device_id, result)
    return result


def _push_locked(server: "RelayServer", device_id: str) -> dict[str, Any]:
    name = server._member_name(device_id)  # noqa: SLF001 — the mesh's own name
    label = name or device_id
    skills = authored_skills(server.root)
    skipped = _skipped_rows(skills)
    # ``name`` rides the result and the receipt so the CLI can say "device-b",
    # not the raw id, wherever the mesh knows a name (QA round 1, Q-5).
    base: dict[str, Any] = {"device_id": device_id, "name": name, "skipped": skipped}
    blocked = definitions.unholdable_capability(server, device_id, "net_skills")
    if blocked:
        # ASKED LOCALLY FIRST, the tick step's rule: the peer would refuse and
        # relay ITS sentence ("… may not do that on this device"), which reads
        # as a statement about THIS device when printed here (QA round 1, Q-3).
        return {
            **base,
            "ok": False,
            "code": f"no_{blocked}",
            "message": (
                f"this device does not hold the '{blocked}' capability on {label}, so its "
                f"skills were not sent; on {label}, run `lop network member grant "
                f"<network> {server.identity.device_id} {blocked}`"
            ),
        }
    link = server._ensure_link(device_id)  # noqa: SLF001 — the one dial seam
    if link is None:
        return {
            **base,
            "ok": False,
            "code": "unreachable",
            "message": f"{label} is not answering right now, so its skills were not synced",
        }
    if wire.SKILLS_SYNC_V1 not in link.capabilities:
        # An old peer never advertised the op; asking anyway would spend a
        # refusal it had to compose. It is skipped with its remedy instead.
        return {
            **base,
            "ok": False,
            "code": "peer_too_old",
            "message": f"{label} predates skill sync; run `lop-update` there, then re-check",
        }
    try:
        state = link.request(
            {
                "op": "net_skills",
                "req": server._next_relay_req(),  # noqa: SLF001
                "locality": "remote",
                "phase": "state",
            },
            timeout=PUSH_TIMEOUT_S,
        )
    except Exception as exc:  # noqa: BLE001 — a probe failure is reported, not raised
        return {**base, "ok": False, "code": "state_failed", "message": str(exc)}
    if state is None:
        # A no-answer is not a refusal (definitions' Q-1): retried next tick.
        return {
            **base,
            "ok": False,
            "code": "no_answer",
            "message": "that device did not answer; its skills will be retried",
        }
    if not isinstance(state, dict) or state.get("op") == "error":
        detail = state if isinstance(state, dict) else {}
        return {
            **base,
            "ok": False,
            "code": str(detail.get("code") or "refused"),
            "message": str(detail.get("message") or "that device did not answer with its skills"),
        }
    detail_in = state.get("detail")
    remote_in = detail_in.get("skills") if isinstance(detail_in, dict) else None
    remote: dict[str, Any] = remote_in if isinstance(remote_in, dict) else {}
    own_id = str(server.identity.device_id)
    outcome: dict[str, list[dict[str, Any]]] = {
        "installed": [],
        "updated": [],
        "unchanged": [],
        "kept": [],
        "refused": [],
    }
    in_sync = 0
    for skill in skills:
        if skill.status != "ready":
            continue
        held = remote.get(skill.name)
        held = held if isinstance(held, dict) else None
        if held is not None and str(held.get("tree_hash") or "") == skill.tree_hash:
            in_sync += 1
            continue
        if held is not None:
            # DECIDED FROM THE PEER'S OWN ANSWER, so a skill the peer will keep
            # costs no multi-megabyte frame every minute. The peer re-decides
            # under its lock anyway; this is economy, not authority.
            held_origin = str(held.get("origin") or "")
            if not held_origin:
                where = str(held.get("where") or "")
                reason = (
                    _kept_reason(
                        skill.name,
                        where,
                        mirror=bool(held.get("mirror")),
                        mirror_wins=bool(held.get("mirror_wins")),
                        label=label,
                    )
                    if where
                    else f"{label} has its own skill by that name, so it was kept"
                )
            elif held_origin != own_id:
                reason = f"that skill on {label} is a copy of another device's, so it was kept"
            elif held.get("edited"):
                reason = f"the copy on {label} has local edits, so it was kept"
            else:
                reason = ""
            if reason:
                outcome["kept"].append({"name": skill.name, "reason": reason})
                continue
        payload = {
            "kind": BUNDLE_KIND,
            "version": BUNDLE_VERSION,
            "name": skill.name,
            "bundle": dict(skill.ref or {}),
            "data": base64.b64encode(skill.data or b"").decode("ascii"),
        }
        try:
            reply = link.request(
                {
                    "op": "net_skills",
                    "req": server._next_relay_req(),  # noqa: SLF001
                    "locality": "remote",
                    "phase": "apply",
                    "skill": payload,
                },
                timeout=PUSH_TIMEOUT_S,
            )
        except Exception as exc:  # noqa: BLE001
            outcome["refused"].append({"name": skill.name, "reason": str(exc)})
            continue
        if not isinstance(reply, dict) or reply.get("op") == "error":
            detail = reply if isinstance(reply, dict) else {}
            outcome["refused"].append(
                {
                    "name": skill.name,
                    "reason": str(detail.get("message") or "that device did not answer"),
                }
            )
            continue
        answer_in = reply.get("detail")
        answer: dict[str, Any] = answer_in if isinstance(answer_in, dict) else {}
        bucket = str(answer.get("outcome") or "refused")
        row = {"name": skill.name}
        if answer.get("reason"):
            row["reason"] = str(answer.get("reason"))
        outcome[
            "kept" if bucket == "conflicts" else bucket if bucket in outcome else "refused"
        ].append(row)
    sent = len(outcome["installed"]) + len(outcome["updated"])
    problems = outcome["refused"]
    parts = []
    sendable = sum(1 for skill in skills if skill.status == "ready")
    if sent:
        parts.append(f"sent {sent} {'skill' if sent == 1 else 'skills'} to {label}")
    elif not problems:
        # THE READY ROW'S SENTENCE, so push and ``lop network ready`` agree: a
        # skill kept as the peer's own is not "held" (QA round 1, Q-5).
        parts.append(
            f"{label} holds {in_sync} of this device's {sendable} sendable skills"
            if sendable
            else f"none of this device's skills can travel to {label}"
        )
    if outcome["kept"]:
        parts.append(f"{len(outcome['kept'])} kept as {label}'s own: " + _names(outcome["kept"]))
    if skipped:
        parts.append(f"{len(skipped)} not sent: " + _describe(skipped))
    if problems:
        parts.append(f"{len(problems)} refused: " + _describe(problems))
    code = "refused_skill" if problems else ("applied" if sent else "in_sync")
    return {
        **base,
        "ok": not problems,
        "code": code,
        "message": "; ".join(parts),
        "in_sync": in_sync,
        "sendable": sendable,
        # HOLDS = every sendable skill is now on the peer as this device's copy;
        # the push summary counts these so it cannot say "1 of 1 device holds"
        # beside a per-peer "holds 0 of 1; 1 kept" (QA round 2, Q-7).
        "holds_all": bool(
            not problems and sendable and in_sync + sent == sendable and not outcome["kept"]
        ),
        **outcome,
    }


def _names(rows: Sequence[Mapping[str, Any]]) -> str:
    return ", ".join(str(row.get("name") or "?") for row in rows)


def _describe(rows: Sequence[Mapping[str, Any]]) -> str:
    return "; ".join(f"{row.get('name') or '?'} ({row.get('reason') or 'refused'})" for row in rows)


def push_for_create(server: "RelayServer", device_id: str) -> dict[str, Any]:
    """The create path's bounded push: wait up to :data:`CREATE_PUSH_BUDGET_S`.

    A push still running at the budget keeps going on a daemon thread (it holds
    the per-peer lock, so the cadence cannot double-send) and the receipt says the
    skills are arriving in the background — never a silent "done".
    """
    box: dict[str, Any] = {}

    def _run() -> None:
        try:
            # BOTH PRE-CHECKS RUN HERE, INSIDE THE BUDGET (review round 1, m1):
            # ``authored_skills`` re-deflates every skill on a cold cache
            # (measured 2.72 s cold vs 0.023 s warm on the head's 30 skills), so
            # outside the ``join`` it could push a create past the 10 s bound.
            if definitions.unholdable_capability(server, device_id, "net_skills"):
                # The tick step's capability rule (m2 / Q-3): the peer would
                # refuse, so a create spends no dial and prints no line.
                box["result"] = {}
                return
            if not any(skill.status == "ready" for skill in authored_skills(server.root)):
                # Nothing of ours can travel, so asking would only add a dial
                # (the tick step's rule). Names nothing: no receipt line.
                box["result"] = {}
                return
            box["result"] = push_to_peer(server, device_id)
        except Exception as exc:  # noqa: BLE001 — a create must not fail on this
            box["result"] = {"ok": False, "code": "failed", "message": str(exc)}

    worker = threading.Thread(target=_run, name="mesh-skills-create-push", daemon=True)
    worker.start()
    worker.join(CREATE_PUSH_BUDGET_S)
    if worker.is_alive():
        label = server._member_name(device_id) or device_id  # noqa: SLF001
        return {
            "ok": True,
            "code": "background",
            "background": True,
            "message": (
                f"skills are still syncing to {label} in the background; a skill may "
                "appear there after the session's first turn"
            ),
        }
    result = box.get("result") or {}
    return {
        key: result.get(key)
        for key in ("ok", "code", "message", "installed", "updated", "kept", "skipped", "refused")
        if key in result
    }


# ---------------------------------------------------------------------------
# The ops
# ---------------------------------------------------------------------------


def make_handler(server: "RelayServer") -> Callable[[Any, dict[str, Any]], dict[str, Any]]:
    """The ``net_skills`` handler: ``state`` (hashes) or ``apply`` (one skill)."""

    def _handle(link: Any, frame: dict[str, Any]) -> dict[str, Any]:
        phase = str(frame.get("phase") or "")
        if phase == "state":
            return {"phase": "state", **skill_state(server.root)}
        if phase == "apply":
            payload = frame.get("skill")
            if not isinstance(payload, Mapping):
                raise MeshRefusal("bad_request", "an apply needs the skill it applies")
            from local_operator.network.audit import AuditEvent

            answer = apply_skill(server.root, payload, origin_device=link.device_id)
            # AUDITED BY NAME, never by content: a skill steers every future
            # session on this device, which is what an incident review asks about.
            try:
                server.audit.record(
                    AuditEvent(
                        event="skills_applied",
                        actor=link.device_id,
                        subject=str(link.device_id),
                        network_id=link.network_id,
                        epoch=link.epoch,
                        detail={
                            bucket: [str(answer.get("name") or "")]
                            for bucket in ("installed", "updated", "conflicts", "refused")
                            if answer.get("outcome") == bucket
                        },
                    )
                )
            except Exception:  # noqa: BLE001 — a record is not a gate
                logger.debug("skillsync: could not record the apply", exc_info=True)
            return {"phase": "apply", **answer}
        raise MeshRefusal("bad_request", f"unknown skills phase {phase!r}")

    return _handle


def local_sync_handler(server: "RelayServer") -> Callable[[dict[str, Any]], dict[str, Any]]:
    """The ``skills_sync`` local verb behind ``lop network skills push``."""

    def _handle(frame: dict[str, Any]) -> dict[str, Any]:
        from local_operator.network import store

        peer = str(frame.get("peer") or "")
        targets: list[str] = []
        if peer:
            targets = [server._resolve_peer(peer)]  # noqa: SLF001 — the one name resolver
        else:
            for record in store.list_networks(server.root):
                for member in record.active_members():
                    if member.device_id != server.identity.device_id and (
                        member.device_id not in targets
                    ):
                        targets.append(member.device_id)
        results = [push_to_peer(server, device_id) for device_id in targets]
        ok = all(bool(item.get("ok")) for item in results) if results else True
        return {
            "ok": ok,
            "peers": results,
            "message": (
                "nothing to push to: this device is in no network with another member"
                if not results
                else (
                    "none of this device's skills can travel"
                    if all(item.get("ok") and not item.get("sendable") for item in results)
                    else (
                        f"{sum(1 for item in results if item.get('holds_all'))} of "
                        f"{len(results)} "
                        f"{'device holds' if len(results) == 1 else 'devices hold'} "
                        "all of this device's sendable skills"
                    )
                )
            ),
        }

    return _handle


def mesh_tick_step(step_server: "RelayServer", device_id: str) -> str:
    """One member's skills push, run INSIDE the mesh-definitions thread's tick.

    Same seam and floors as ``mcpdefs.mesh_tick_step``. Its codes are never in
    ``definitions.POLICY_REFUSAL_CODES`` except a genuine peer policy answer, so a
    kept or oversize skill cannot park the member's other syncs.
    """
    blocked = definitions.unholdable_capability(step_server, device_id, "net_skills")
    if blocked:
        return f"skipped:no_{blocked}"
    if not any(skill.status == "ready" for skill in authored_skills(step_server.root)):
        return "in_sync"  # push-only: nothing of ours can change what the peer holds
    return str(push_to_peer(step_server, device_id).get("code") or "")


#: A stage entry older than this is a crash leftover, never a live apply: one
#: apply is bounded by :data:`SKILLS_OP_DEADLINE_S` (60 s), so an hour is far
#: past any apply that could still be writing into it.
STAGE_STALE_S = 3600.0


def _sweep_stage(root: Path, *, now: float | None = None) -> int:
    """Remove crash-leftover extracted trees from ``network/skill-stage``.

    WHY: an apply removes its stage in ``finally``, but a relay killed between
    extract and that ``finally`` leaves a full 0600 tree behind that nothing else
    would ever delete (review round 1, m4). Run once when the slice installs.
    AGE-GATED rather than \"everything\": a relay object can be constructed on a
    root whose serving relay is mid-apply, and a fresh stage is that apply's.
    Best effort: a sweep failure must never stop the relay installing the op.
    """
    base = _stage_root(root)
    # NEVER THROUGH A LINK: ``scandir`` would resolve a symlinked stage (or a
    # symlinked ``network/``) and the sweep would delete aged entries in its
    # target (review round 2, R2-2). The apply's own stage never is one.
    try:
        if base.is_symlink() or base.parent.is_symlink():
            return 0
        _assert_inside_root(base, root)
    except (OSError, MeshRefusal):
        return 0
    cutoff = (time.time() if now is None else now) - STAGE_STALE_S
    removed = 0
    try:
        entries = list(os.scandir(base))
    except OSError:
        return 0
    for entry in entries:
        try:
            if entry.stat(follow_symlinks=False).st_mtime > cutoff:
                continue
            if entry.is_dir(follow_symlinks=False):
                shutil.rmtree(entry.path)  # never follows a link inside the tree
            else:
                os.unlink(entry.path)  # a file or a link itself, never its target
            removed += 1
        except OSError:
            logger.debug("skillsync: could not sweep %s", entry.path, exc_info=True)
    return removed


def install(server: "RelayServer") -> None:
    """Register ``net_skills`` / ``skills_sync`` and the cadence step."""
    _sweep_stage(server.root)
    definitions.add_tick_step(mesh_tick_step)
    server.register_ops(
        {"net_skills": make_handler(server)},
        local_handlers={"skills_sync": local_sync_handler(server)},
        slow={"net_skills": SKILLS_OP_DEADLINE_S},
    )


def state_rows(root: Path) -> list[dict[str, Any]]:
    """``lop network skills state``: every local skill with its provenance and fate."""
    index = definitions.read_index(root)
    rows = []
    for skill in local_skills(root):
        recorded = _recorded(index, skill.name)
        origin = str(recorded.get("origin") or "") if recorded is not None else ""
        edited = bool(recorded is not None and skill.tree_hash != str(recorded.get("landed") or ""))
        rows.append(
            {
                "name": skill.name,
                "origin": origin,
                "edited": edited,
                "status": skill.status,
                "reason": skill.reason,
                "size": skill.size,
                "tree_hash": skill.tree_hash,
            }
        )
    return rows
