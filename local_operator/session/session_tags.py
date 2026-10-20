"""Durable entity tags shared by desktop and agent callers.

The registry owns identity; assignment order owns primary fallback, so renaming
or reordering categories cannot regroup conversations. Writes use same-directory
replace and are last-writer-wins across processes, like the pin store; there is
no cross-process lock or registry/assignment transaction. Orphans are therefore
filtered on every read rather than resurrected when a name is reused.
Existing assignments retain their original order on a PUT; newly added tags
append in caller order. Changing primary or reordering a request cannot rewrite
assignment history. Usage counts include hidden/archived sessions, but not
assignment files whose session directory has been deleted.
Name-only entries reuse live category IDs for retry safety, except IDs explicitly
claimed by another entry in the same draft (rename plus old-name creation).
Consequently one PUT cannot express delete-and-recreate of the same live name;
the UI must delete and create in separate writes to obtain a new identity.
"""

from __future__ import annotations

import json
import os
import re
import tempfile
from pathlib import Path
from threading import RLock
from typing import Any, TypedDict
from uuid import uuid4

#: The per-entity assignment cap the route, the agent tool and the naming call
#: all quote. One definition so a model-facing cap can never drift from the
#: store's refusal; ``naming.MAX_ASSIGNED_TAGS`` mirrors it because the naming
#: module deliberately imports no store.
MAX_TAGS = 8

#: Registry caps and the reserved display name, exported for the SAME reason:
#: ``tools/tags_tool.py`` pre-checks so a refusal can name the colliding row and
#: the live vocabulary, and a second set of literals there would let the
#: message drift from what this store actually accepts.
MAX_CATEGORIES = 32
MAX_NAME_CHARS = 24
RESERVED_NAME = "Untagged"


class EntityTags(TypedDict):
    tags: list[str]
    primary: str | None
    revision: int


# Serialize conditional read/compare/write within this process only. Other
# processes retain the store's documented last-writer-wins replace semantics;
# this is deliberately not a cross-process CAS or registry transaction.
_assignment_lock = RLock()


class TagRevisionConflict(Exception):
    """A stale assignment token, with the readable snapshot that rejected it."""

    def __init__(self, current: dict[str, Any]) -> None:
        super().__init__("Tag assignment changed; refresh before retrying")
        self.current = current


class TagCreateApplyError(Exception):
    """Creation succeeded but the assignment write did not.

    Registry creation and assignment are two files, so the store cannot offer a
    transaction across them (see the module docstring). A caller must still be
    able to say WHICH half landed rather than report a flat failure that invites
    a blind retry: retrying the create would duplicate a category name, while
    retrying the apply is exactly the right next step. ``created`` is that fact.
    """

    def __init__(self, message: str, *, created: bool) -> None:
        super().__init__(message)
        self.created = created


class TagToggle(TypedDict):
    """The assignment after a toggle, plus what the toggle did.

    ``created`` and ``applied`` are separate facts on purpose: creating a tag
    and assigning it are distinct durable writes, and a caller that collapses
    them into one "changed" flag cannot describe the partial-create failure
    above or tell a create-and-apply from an apply of an existing name.
    """

    tags: list[str]
    primary: str | None
    revision: int
    created: bool
    applied: bool
    #: The registry's canonical spelling, which may differ in case from the
    #: name the caller typed. Receipts quote THIS, so the sentence names the
    #: row the list paints rather than the user's casing.
    name: str


def _read(path: Path) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text())
        return value if isinstance(value, dict) and value.get("version") == 1 else {}
    except (OSError, ValueError):
        return {}


def _write(path: Path, value: dict[str, Any]) -> bool:
    """Replace ``path`` atomically; True iff the bytes on disk changed.

    The return value is what keeps the change marker honest: an identical
    re-write returns False, so a caller that only wants to notice real
    mutations does not have to re-read the file to find out. Nothing about the
    on-disk layout changed -- the identical-content early return was already
    here.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    data = json.dumps(value, ensure_ascii=False) + "\n"
    try:
        if path.read_text() == data:
            return False
    except OSError:
        pass
    fd, name = tempfile.mkstemp(dir=path.parent, prefix=".tags-", suffix=".tmp")
    try:
        with os.fdopen(fd, "w") as stream:
            stream.write(data)
        os.replace(name, path)
    finally:
        Path(name).unlink(missing_ok=True)
    return True


def _bump_revision(root: Path) -> None:
    """Publish a new equality token after a REAL store mutation.

    Mobile's 2 s scan tick cannot re-walk every assignment file to find out
    whether the desktop, the TUI, an agent or auto-tagging changed something;
    it reads this one small file instead and rescans only when the token moves.
    The token is an EQUALITY MARKER, never an order or a timestamp: readers may
    only ask "is it different from the one I saw", so a clock that steps or two
    writes inside one filesystem tick cannot make a change invisible.

    THE ASSIGNMENT MAY HAVE ALREADY LANDED when this runs, so a marker
    failure cannot simply be raised past: the old token is still valid and
    EQUAL, and a client retry writes identical bytes (``_write`` returns
    ``False``) so it never reaches this function to bump it. Readers would
    then serve the pre-write projection until some unrelated real mutation.

    Recovery, in order: retry the write once (the realistic failure --
    ENOSPC/EIO/EACCES on ``tags/`` -- is transient); if the retry also fails,
    REMOVE the stale marker, because a marker that cannot be bumped is worse
    than none: readers then fall back to the stats fingerprint, which the
    landed assignment has already moved. Only if that removal fails too does
    the residual case remain -- a landed mutation no reader can observe until
    the next real write -- and then the marker write's error is re-raised,
    since there is nothing left to report.
    """
    marker = Path(root) / "tags" / "revision"
    # ``except ... as name`` unbinds the name at the end of the clause, so the
    # first error is copied out to survive to the residual raise below.
    first_error: OSError | None = None
    try:
        _write(marker, {"version": 1, "token": uuid4().hex})
        return
    except OSError as exc:
        first_error = exc
    try:
        _write(marker, {"version": 1, "token": uuid4().hex})
    except OSError as second_error:
        try:
            marker.unlink(missing_ok=True)
        except OSError:
            raise second_error from first_error
        return


def read_tags_revision(root: Path) -> str | None:
    """The store's current change token, or None for a store without one.

    This is the STORE-WIDE projection marker: it moves when any category or
    assignment changes. It is NOT the per-session assignment revision, and it
    is never valid as ``expected_revision`` -- the two are independent
    counters, so comparing them fails closed with a guaranteed conflict.

    None covers both cases readers must treat alike -- a store written by a
    build older than this marker, and an unreadable/malformed one -- and the
    daemon's fallback fingerprint is what keeps the first case observable.
    """
    token = _read(Path(root) / "tags" / "revision").get("token")
    return token if isinstance(token, str) and token else None


def read_categories(root: Path) -> list[dict[str, str]]:
    """Unreadable registries are empty; never expose malformed entries."""
    raw = _read(Path(root) / "tags" / "categories.json").get("categories", [])
    if not isinstance(raw, list):
        return []
    result: list[dict[str, str]] = []
    ids: set[str] = set()
    names: set[str] = set()
    for item in raw:
        if not isinstance(item, dict):
            continue
        key, name = item.get("id"), item.get("name")
        if (
            isinstance(key, str)
            and re.fullmatch(r"[a-f0-9]{10}", key)
            and isinstance(name, str)
            and 1 <= len(name.strip()) <= MAX_NAME_CHARS
            and name.strip().casefold() != RESERVED_NAME.casefold()
            and key not in ids
            and name.strip().casefold() not in names
        ):
            result.append({"id": key, "name": name.strip()})
            ids.add(key)
            names.add(name.strip().casefold())
    return result[:MAX_CATEGORIES]


def write_categories(root: Path, desired: list[tuple[str | None, str]]) -> list[dict[str, str]]:
    if len(desired) > MAX_CATEGORIES:
        raise ValueError(f"At most {MAX_CATEGORIES} categories are allowed")
    existing = read_categories(root)
    known = {item["id"] for item in existing}
    by_name = {item["name"].casefold(): item["id"] for item in existing}
    explicit_ids = {key for key, _ in desired if key is not None}
    ids: set[str] = set()
    names: set[str] = set()
    categories = []
    for key, raw_name in desired:
        name = raw_name.strip()
        folded = name.casefold()
        if (
            not 1 <= len(name) <= MAX_NAME_CHARS
            or folded == RESERVED_NAME.casefold()
            or folded in names
        ):
            raise ValueError(
                f"Category names must be unique, 1..{MAX_NAME_CHARS} characters; "
                f"{RESERVED_NAME} is reserved"
            )
        if key is not None and (key not in known or key in ids):
            raise ValueError("Unknown or duplicate category id")
        if key is None:
            # A dropped PUT response may be retried without server-generated
            # IDs. Match the existing vocabulary rather than orphan assignments.
            candidate = by_name.get(folded)
            # Reserve explicit identities across the whole draft, regardless of
            # row order: renaming X must permit a fresh tag with X's old name.
            key = candidate if candidate not in explicit_ids else None
        if key is None:
            key = uuid4().hex[:10]
            while key in known or key in ids:
                key = uuid4().hex[:10]
        if key in ids:
            raise ValueError("Duplicate category id")
        categories.append({"id": key, "name": name})
        ids.add(key)
        names.add(folded)
    if _write(
        Path(root) / "tags" / "categories.json",
        {"version": 1, "categories": categories},
    ):
        _bump_revision(root)
    return categories


def resolve_category_names(
    root: Path, names: list[str], *, categories: list[dict[str, str]] | None = None
) -> list[str]:
    categories = read_categories(root) if categories is None else categories
    lookup = {item["name"].casefold(): item["id"] for item in categories}
    if not 1 <= len(names) <= 8 or any(not name.strip() for name in names):
        raise ValueError("Supply 1..8 non-empty category names")
    unknown = [name for name in names if name.strip().casefold() not in lookup]
    if unknown:
        known = ", ".join(item["name"] for item in categories) or "(none)"
        raise ValueError(f"Unknown tags: {', '.join(unknown)}. Known categories: {known}")
    result = [lookup[name.strip().casefold()] for name in names]
    if len(set(result)) != len(result):
        raise ValueError("Duplicate tags are not allowed")
    return result


def _entity_path(root: Path, entity_type: str, entity_id: str) -> Path:
    if entity_type not in {"session", "project"}:
        raise ValueError("Entity type must be session or project")
    # Match the desktop session-id contract without importing the catalogue
    # (and its runtime dependencies) into this filesystem-only store.
    if entity_type == "session":
        valid = bool(re.fullmatch(r"[a-f0-9]{12}", entity_id))
    else:
        valid = bool(re.fullmatch(r"[A-Za-z0-9_-]+", entity_id))
    if not valid or entity_id in {".", ".."}:
        raise ValueError("Invalid entity id")
    return Path(root) / "tags" / entity_type / f"{entity_id}.json"


def read_entity_tags(
    root: Path,
    entity_type: str,
    entity_id: str,
    *,
    categories: list[dict[str, str]] | None = None,
) -> EntityTags:
    """Prune orphans in the projection, without a read/modify/write race."""
    raw = _read(_entity_path(root, entity_type, entity_id))
    known = {item["id"] for item in (read_categories(root) if categories is None else categories)}
    tags = raw.get("tags", [])
    surviving: list[str] = []
    if isinstance(tags, list):
        for key in tags:
            if isinstance(key, str) and key in known and key not in surviving:
                surviving.append(key)
    surviving = surviving[:MAX_TAGS]
    primary = raw.get("primary")
    if primary not in surviving:
        primary = surviving[0] if surviving else None
    revision = raw.get("revision", 0)
    if type(revision) is not int or revision < 0:
        revision = 0
    return {"tags": surviving, "primary": primary, "revision": revision}


def set_entity_tags(
    root: Path,
    entity_type: str,
    entity_id: str,
    tag_ids: list[str],
    primary: str | None,
    expected_revision: str | int | None = None,
) -> EntityTags:
    # A revision catches add-then-clear (ABA) edits even when tags look unchanged.
    with _assignment_lock:
        return _set_entity_tags(root, entity_type, entity_id, tag_ids, primary, expected_revision)


def _set_entity_tags(
    root: Path,
    entity_type: str,
    entity_id: str,
    tag_ids: list[str],
    primary: str | None,
    expected_revision: str | int | None,
) -> EntityTags:
    path = _entity_path(root, entity_type, entity_id)
    categories = read_categories(root)
    current = read_entity_tags(root, entity_type, entity_id, categories=categories)
    if expected_revision is not None and str(expected_revision) != str(current["revision"]):
        lookup = {item["id"]: item for item in categories}
        raise TagRevisionConflict(
            {
                "tags": [lookup[key] for key in current["tags"]],
                "primary": current["primary"],
                "tags_revision": str(current["revision"]),
            }
        )
    known = {item["id"] for item in categories}
    if len(tag_ids) > MAX_TAGS or len(set(tag_ids)) != len(tag_ids):
        raise ValueError(f"At most {MAX_TAGS} distinct tags are allowed")
    if any(key not in known for key in tag_ids):
        raise ValueError("Unknown category id")
    if primary is not None and primary not in tag_ids:
        raise ValueError("Primary must be one of the assigned tags or null")
    previous = current["tags"]
    ordered = [key for key in previous if key in tag_ids]
    ordered.extend(key for key in tag_ids if key not in ordered)
    resolved_primary = primary if primary is not None else (ordered[0] if ordered else None)
    # Compare the readable state, not request bytes: reordered existing IDs and
    # primary=None fallback are retries, not edits. No-op writes must not make
    # another viewer's otherwise-current token stale (or touch the file).
    if ordered == current["tags"] and resolved_primary == current["primary"]:
        return current
    state: EntityTags = {
        "tags": ordered,
        "primary": resolved_primary,
        "revision": current["revision"] + 1,
    }
    # The relay-wide equality marker is independent of this session's ordered
    # revision; publish it only after the assignment bytes actually changed.
    if _write(path, {"version": 1, **state}):
        _bump_revision(root)
    return state


def effective_primary(
    tags: list[str],
    primary: str | None,
    categories: list[dict[str, str]],
) -> str | None:
    """The primary tag that may group this entity, or None for Untagged.

    THE GROUPING PREDICATE, in the store rather than in a renderer: a bucket is
    a primary only when it is non-null, still in the registry, AND still in this
    entity's assignment. A dangling id or a registry edit that removed a
    category must fall to Untagged rather than create an invisible group or
    resurrect a stale heading. Grouping by "has any tags" is a different
    question and deliberately not answered here; secondary tags never choose a
    bucket.
    """
    if primary is None:
        return None
    known = {item["id"] for item in categories}
    return primary if primary in known and primary in tags else None


def toggle_entity_tag(
    root: Path,
    entity_type: str,
    entity_id: str,
    name: str,
    *,
    expected_revision: str | int | None = None,
) -> TagToggle:
    """Toggle a tag by NAME, creating and applying it when the name is new.

    ONE semantic for every client (TUI now, desktop next): a user activating a
    tag row means "this conversation is in this category, flip that", not "edit
    a registry and then, separately, an assignment". Existing names always win
    over creation (case-insensitive, matching the registry's own uniqueness
    rule), so a retry after a dropped response cannot mint a duplicate.

    PRIMARY RULES, stated here because they are the grouping contract: the first
    assignment becomes primary; adding another tag never moves it; removing a
    non-primary leaves it; removing the primary promotes the oldest surviving
    assignment. Empty assignments have no primary. ``set_entity_tags`` owns the
    append-order mechanics; this function only chooses the primary it wants.

    CAPS ARE CHECKED BEFORE CREATION, so a full assignment list cannot leave a
    newly created category behind with nothing using it.

    Raises :class:`TagCreateApplyError` when the name was created but the
    assignment write failed. Any other write failure propagates unchanged.
    """
    cleaned = name.strip()
    if not 1 <= len(cleaned) <= 24:
        raise ValueError("Tag names must be 1..24 characters")
    if cleaned.casefold() == "untagged":
        raise ValueError("Untagged is reserved")
    with _assignment_lock:
        categories = read_categories(root)
        current = read_entity_tags(root, entity_type, entity_id, categories=categories)
        match = next(
            (item for item in categories if item["name"].casefold() == cleaned.casefold()),
            None,
        )
        created = False
        if match is None:
            if len(current["tags"]) >= 8:
                raise ValueError("At most 8 distinct tags are allowed")
            try:
                desired: list[tuple[str | None, str]] = [
                    (item["id"], item["name"]) for item in categories
                ]
                desired.append((None, cleaned))
                categories = write_categories(root, desired)
            except ValueError as exc:
                raise ValueError(str(exc)) from exc
            match = next(
                (item for item in categories if item["name"].casefold() == cleaned.casefold()),
                None,
            )
            if match is None:  # pragma: no cover - the writer just stored it
                raise ValueError("Tag creation did not publish the category")
            created = True
        tag_id = match["id"]
        if tag_id in current["tags"]:
            tags = [key for key in current["tags"] if key != tag_id]
            primary = current["primary"] if current["primary"] in tags else None
            applied = False
        else:
            if len(current["tags"]) >= 8:
                raise ValueError("At most 8 distinct tags are allowed")
            tags = [*current["tags"], tag_id]
            primary = (
                current["primary"]
                if current["primary"] in current["tags"]
                else current["tags"][0] if current["tags"] else tag_id
            )
            applied = True
        try:
            state = _set_entity_tags(root, entity_type, entity_id, tags, primary, expected_revision)
        except Exception as exc:
            if created:
                raise TagCreateApplyError(str(exc), created=True) from exc
            raise
        return {**state, "created": created, "applied": applied, "name": match["name"]}


def usage_counts(root: Path, entity_type: str = "session") -> dict[str, int]:
    _entity_path(root, entity_type, "aaaaaaaaaaaa")
    categories = read_categories(root)
    counts = {item["id"]: 0 for item in categories}
    for path in (Path(root) / "tags" / entity_type).glob("*.json"):
        try:
            assignment = read_entity_tags(root, entity_type, path.stem, categories=categories)
        except ValueError:
            continue
        if entity_type == "session" and not (Path(root) / "sessions" / path.stem).is_dir():
            continue
        for key in assignment["tags"]:
            counts[key] += 1
    return counts
