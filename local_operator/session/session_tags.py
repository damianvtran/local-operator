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
from typing import Any, TypedDict
from uuid import uuid4


class EntityTags(TypedDict):
    tags: list[str]
    primary: str | None


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

    A failed marker write is allowed to raise, like every other store write.
    The assignment itself has already landed at that point; the request fails
    and its retry is idempotent, which is the same last-writer-wins contract
    the rest of this module states. Swallowing the error would instead leave
    readers serving their cached projection until some unrelated token move.
    """
    _write(Path(root) / "tags" / "revision", {"version": 1, "token": uuid4().hex})


def read_tags_revision(root: Path) -> str | None:
    """The store's current change token, or None for a store without one.

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
            and 1 <= len(name.strip()) <= 24
            and name.strip().casefold() != "untagged"
            and key not in ids
            and name.strip().casefold() not in names
        ):
            result.append({"id": key, "name": name.strip()})
            ids.add(key)
            names.add(name.strip().casefold())
    return result[:32]


def write_categories(root: Path, desired: list[tuple[str | None, str]]) -> list[dict[str, str]]:
    if len(desired) > 32:
        raise ValueError("At most 32 categories are allowed")
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
        if not 1 <= len(name) <= 24 or folded == "untagged" or folded in names:
            raise ValueError(
                "Category names must be unique, 1..24 characters; Untagged is reserved"
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
    surviving = surviving[:8]
    primary = raw.get("primary")
    if primary not in surviving:
        primary = surviving[0] if surviving else None
    return {"tags": surviving, "primary": primary}


def set_entity_tags(
    root: Path, entity_type: str, entity_id: str, tag_ids: list[str], primary: str | None
) -> EntityTags:
    path = _entity_path(root, entity_type, entity_id)
    categories = read_categories(root)
    known = {item["id"] for item in categories}
    if len(tag_ids) > 8 or len(set(tag_ids)) != len(tag_ids):
        raise ValueError("At most 8 distinct tags are allowed")
    if any(key not in known for key in tag_ids):
        raise ValueError("Unknown category id")
    if primary is not None and primary not in tag_ids:
        raise ValueError("Primary must be one of the assigned tags or null")
    previous = read_entity_tags(root, entity_type, entity_id, categories=categories)["tags"]
    ordered = [key for key in previous if key in tag_ids]
    ordered.extend(key for key in tag_ids if key not in ordered)
    if _write(path, {"version": 1, "primary": primary, "tags": ordered}):
        _bump_revision(root)
    return read_entity_tags(root, entity_type, entity_id)


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
