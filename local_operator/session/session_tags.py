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


def _write(path: Path, value: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    data = json.dumps(value, ensure_ascii=False) + "\n"
    try:
        if path.read_text() == data:
            return
    except OSError:
        pass
    fd, name = tempfile.mkstemp(dir=path.parent, prefix=".tags-", suffix=".tmp")
    try:
        with os.fdopen(fd, "w") as stream:
            stream.write(data)
        os.replace(name, path)
    finally:
        Path(name).unlink(missing_ok=True)


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
            key = by_name.get(folded)
        if key is None:
            key = uuid4().hex[:10]
            while key in known or key in ids:
                key = uuid4().hex[:10]
        if key in ids:
            raise ValueError("Duplicate category id")
        categories.append({"id": key, "name": name})
        ids.add(key)
        names.add(folded)
    _write(Path(root) / "tags" / "categories.json", {"version": 1, "categories": categories})
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
    _write(path, {"version": 1, "primary": primary, "tags": ordered})
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
