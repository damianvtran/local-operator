"""The ``library`` tool: read and organize the global asset catalog.

Shape mirrors ``project``: ONE op-based tool (``stats | list | get | update |
tags | conventions``) rather than six tools, because the provider tools array
rides in the prompt-cache prefix and a Library operation is rare per turn.

APPROVAL. Read ops are open. Writes here are LOCAL, REVERSIBLE METADATA — a
name, a description, a tag assignment, a convention row — and never touch a
user file, so the tool as a whole is read-tier. That is the design's §5.1
disposition, not an accident: the Librarian runs unattended and a nightly run
must not block on an approval dialog to rename an asset.

PROVENANCE. A write through this tool is recorded as ``librarian`` with the
supplied confidence; the explicit ``actor`` parameter exists for a future
user-initiated surface, and defaults to the Librarian because that is the
caller the design expects. Effective-name precedence lives in the SQLite store:
a user/librarian value is never overwritten by a later derived refresh.
"""

from __future__ import annotations

from typing import Any, Callable, Literal

from pydantic import BaseModel, ConfigDict, Field, ValidationError

from local_operator.harness.types import (
    AbortSignal,
    AgentTool,
    AgentToolUpdate,
    ToolContext,
    ToolResult,
)
from local_operator.library import service
from local_operator.library.metadata import (
    TAG_CATEGORIES,
    MetadataStore,
    MetadataUnavailable,
    TagRefused,
)
from local_operator.paths import config_dir
from local_operator.tools.builtin import _error, _guard, _text, _validation_error


class LibraryParams(BaseModel):
    model_config = ConfigDict(extra="forbid")

    op: Literal["stats", "list", "get", "update", "tags", "conventions"] = Field(
        description="The verb to run."
    )
    asset_id: str | None = Field(default=None, description="get/update/tags: an `la_` id.")
    q: str = Field(default="", description="list: case-insensitive catalog search.")
    category: str | None = Field(default=None, description="list: comma list of categories.")
    kind: str | None = Field(default=None, description="list: comma list of kinds.")
    origin_group: str | None = Field(
        default=None, description="list: comma list of generated|created|mentioned."
    )
    tag: str | None = Field(default=None, description="list: a tag id.")
    session_id: str | None = Field(
        default=None, description="list: comma list of source conversation ids."
    )
    organized: bool | None = Field(
        default=None, description="list/update: whether the Librarian has organized it."
    )
    highlighted: bool | None = Field(default=None, description="list: highlighted only/never.")
    favorite: bool | None = Field(default=None, description="list: favorites only/never.")
    tag_mode: Literal["all", "any"] = Field(default="all", description="list: tag filter mode.")
    sort: Literal["recent", "name", "kind", "size", "created", "modified", "organized"] = Field(
        default="recent", description="list: sort key."
    )
    order: Literal["asc", "desc"] | None = Field(default=None, description="list: sort direction.")
    limit: int = Field(default=25, ge=1, le=200, description="list: page size.")
    cursor: str | None = Field(default=None, description="list: the previous next_cursor.")
    name: str | None = Field(default=None, description="update: effective name.")
    description: str | None = Field(default=None, description="update: description.")
    locked: bool | None = Field(default=None, description="update: protect from the Librarian.")
    archived: bool | None = Field(default=None, description="update: hide from the default view.")
    confidence: float | None = Field(
        default=None, ge=0.0, le=1.0, description="update/tags: the Librarian's confidence."
    )
    # Default `agent`, not `librarian`: an ordinary agent naming its own
    # deliverable must not be recorded as the Librarian. The Librarian seed is
    # the one caller that passes `actor="librarian"` explicitly.
    actor: Literal["agent", "librarian", "user"] = Field(
        default="agent", description="The provenance recorded on a write."
    )
    # -- tags ------------------------------------------------------------------
    action: Literal["list", "create", "rename", "delete", "merge", "set", "clear"] | None = Field(
        default=None, description="tags: the verb; omit for list."
    )
    tags: list[str] | None = Field(
        default=None, description="tags set: tag ids, at most 12; replaces the set."
    )
    primary: str | None = Field(default=None, description="tags set: one tag id from `tags`.")
    tag_id: str | None = Field(default=None, description="tags rename/delete/merge: the tag id.")
    tag_name: str | None = Field(default=None, description="tags create/rename: 1-24 chars.")
    tag_category: str = Field(default="custom", description="tags create: the tag's category.")
    into: str | None = Field(default=None, description="tags merge: the surviving tag id.")
    # -- conventions -----------------------------------------------------------
    kind_name: str | None = Field(
        default=None, description="conventions: the convention's kind (naming, tag, theme)."
    )
    body: str | None = Field(default=None, description="conventions: the rule text.")
    convention_id: str | None = Field(default=None, description="conventions: a stable id.")


def _store() -> MetadataStore:
    return MetadataStore(config_dir())


def _compact(row: dict[str, Any]) -> dict[str, Any]:
    """A list row small enough for a model to read without spilling by default."""
    return {
        "id": row["id"],
        "name": row["name"],
        "kind": row["kind"],
        "category": row["category"],
        "origin_group": row["origin_group"],
        "size_bytes": row["size_bytes"],
        "modified_at": row["modified_at"],
        "tags": [tag["name"] for tag in row["tags"]],
        "primary_tag": row["primary_tag"],
        "sessions": [
            {"id": item["session_id"], "title": item["session_title"]}
            for item in row["occurrences"][:5]
        ],
        "exists": row["exists"],
        "highlighted": row["highlighted"],
    }


def _stats() -> dict[str, Any]:
    catalog = service.list_assets(config_dir(), service.Query(limit=1), budget_s=None)
    metadata = _store().stats()
    counts = catalog["counts"]
    return {
        "assets": catalog["total"],
        "by_category": counts["category"],
        "by_origin_group": counts["origin_group"],
        "by_kind": counts["kind"],
        "metadata": metadata,
        "state": catalog["state"],
    }


@_guard("library")
async def execute_library(
    tool_call_id: str,
    args: dict[str, Any],
    signal: AbortSignal | None = None,
    on_update: Callable[[AgentToolUpdate], None] | None = None,
    context: ToolContext | None = None,
) -> ToolResult:
    try:
        params = LibraryParams(**args)
    except ValidationError as exc:
        return _validation_error(tool_call_id, "library", exc)
    try:
        return await _dispatch(tool_call_id, params)
    except MetadataUnavailable:
        return _error(
            tool_call_id,
            "library",
            "The Library's metadata store is unavailable; nothing was changed.",
        )
    except TagRefused as exc:
        return _error(tool_call_id, "library", f"tag refused: {exc.code}")
    except service.AssetNotFound:
        return _error(tool_call_id, "library", "no asset with that id is in the Library.")
    except service.InvalidCursor as exc:
        return _error(tool_call_id, "library", f"that page cursor is not valid: {exc}")


async def _dispatch(tool_call_id: str, params: LibraryParams) -> ToolResult:
    op = params.op
    if op == "stats":
        answer = _stats()
        return _text(tool_call_id, "library", _render(answer))

    if op == "list":
        try:
            query = service.Query(
                q=params.q,
                categories=service.parse_csv(params.category, vocabulary=service.CATEGORIES),
                kinds=service.parse_csv(params.kind, vocabulary=service.KINDS),
                origin_groups=service.parse_csv(
                    params.origin_group, vocabulary=("generated", "created", "mentioned")
                ),
                tag=params.tag or None,
                tag_mode=params.tag_mode,
                session_ids=service.parse_csv(params.session_id),
                organized=params.organized,
                highlighted=params.highlighted,
                favorite=params.favorite,
                sort=params.sort,
                order=params.order,
                limit=params.limit,
                cursor=params.cursor or None,
            )
        except service.InvalidFilter as exc:
            return _error(tool_call_id, "library", f"bad filter: {exc}")
        answer = service.list_assets(config_dir(), query, budget_s=None)
        payload = {
            "state": answer["state"],
            "total": answer["total"],
            "rows": [_compact(row) for row in answer["rows"]],
            "next_cursor": answer["next_cursor"],
        }
        return _text(tool_call_id, "library", _render(payload))

    if op == "get":
        if not params.asset_id:
            return _error(tool_call_id, "library", "get needs asset_id.")
        row = service.get_asset(config_dir(), params.asset_id)
        return _text(tool_call_id, "library", _render(row))

    if op == "update":
        if not params.asset_id:
            return _error(tool_call_id, "library", "update needs asset_id.")
        fields = {
            name: value
            for name, value in (
                ("name", params.name),
                ("description", params.description),
                ("favorite", params.favorite),
                ("organized", params.organized),
                ("locked", params.locked),
                ("archived", params.archived),
            )
            if value is not None
        }
        if not fields:
            return _error(tool_call_id, "library", "update needs at least one field.")
        row = service.update_asset(
            config_dir(),
            params.asset_id,
            fields,
            actor=params.actor,
            # An ordinary agent's write is stored under its own provenance,
            # which joins user/librarian as a value derived refreshes never
            # overwrite. Only the Librarian's edits mark a row organized.
            source=params.actor,
            confidence=params.confidence,
        )
        return _text(tool_call_id, "library", _render(_compact(row)))

    if op == "tags":
        return _tags(tool_call_id, params)

    # conventions
    store = _store()
    if params.body is not None:
        if not params.kind_name:
            return _error(tool_call_id, "library", "setting a convention needs kind_name.")
        row = store.set_convention(
            params.convention_id or params.kind_name,
            params.kind_name,
            params.body,
            actor=params.actor,
        )
        return _text(tool_call_id, "library", _render(row))
    rows = store.conventions()
    return _text(tool_call_id, "library", _render({"conventions": rows}))


def _tags(tool_call_id: str, params: LibraryParams) -> ToolResult:
    store = _store()
    action = params.action or "list"
    if action == "list":
        return _text(tool_call_id, "library", _render({"tags": store.list_tags()}))
    if action == "create":
        if not params.tag_name:
            return _error(tool_call_id, "library", "tags create needs tag_name.")
        if params.tag_category not in TAG_CATEGORIES:
            return _error(tool_call_id, "library", "unknown tag category.")
        tag = store.create_tag(params.tag_name, params.tag_category, actor=params.actor)
        return _text(tool_call_id, "library", _render(tag))
    if action in ("rename", "delete", "merge"):
        if not params.tag_id:
            return _error(tool_call_id, "library", f"tags {action} needs tag_id.")
        if action == "rename":
            if not params.tag_name:
                return _error(tool_call_id, "library", "tags rename needs tag_name.")
            return _text(
                tool_call_id, "library", _render(store.rename_tag(params.tag_id, params.tag_name))
            )
        if action == "delete":
            deleted = store.delete_tag(params.tag_id)
            return _text(tool_call_id, "library", _render({"deleted": deleted}))
        if not params.into:
            return _error(tool_call_id, "library", "tags merge needs into.")
        return _text(tool_call_id, "library", _render(store.merge_tags(params.tag_id, params.into)))
    if not params.asset_id:
        return _error(tool_call_id, "library", f"tags {action} needs asset_id.")
    if action == "clear":
        row = service.set_tags(
            config_dir(), params.asset_id, [], None, actor=params.actor
        )
        return _text(tool_call_id, "library", _render(_compact(row)))
    row = service.set_tags(
        config_dir(),
        params.asset_id,
        params.tags or [],
        params.primary,
        actor=params.actor,
        confidence=params.confidence,
    )
    return _text(tool_call_id, "library", _render(_compact(row)))


def _render(payload: Any) -> str:
    import json

    return json.dumps(payload, indent=2, sort_keys=True)


#: The WIRE schema, kept deliberately compact: property names, types and the
#: enums a caller must get right, and nothing else. Pydantic's generated schema
#: carries a ``title``, ``default`` and ``anyOf`` wrapper per optional field;
#: that structural prose measured 5,060 characters and pushed the start context
#: over its CI ceiling. The per-op semantics live in ``tool://library`` and in
#: ``LibraryParams`` (which still validates every call), so this dict is the
#: discovery surface only. ``test_the_wire_schema_matches_the_validator`` pins
#: the property set and types against ``LibraryParams`` so the two cannot drift.
LIBRARY_PARAMETERS: dict[str, Any] = {
    "type": "object",
    "required": ["op"],
    "properties": {
        "op": {
            "type": "string",
            "enum": ["stats", "list", "get", "update", "tags", "conventions"],
        },
        "asset_id": {"type": "string"},
        "q": {"type": "string"},
        "category": {"type": "string"},
        "kind": {"type": "string"},
        "origin_group": {"type": "string"},
        "tag": {"type": "string"},
        "session_id": {"type": "string"},
        "organized": {"type": "boolean"},
        "highlighted": {"type": "boolean"},
        "favorite": {"type": "boolean"},
        "tag_mode": {"type": "string"},
        "sort": {"type": "string"},
        "order": {"type": "string"},
        "limit": {"type": "integer"},
        "cursor": {"type": "string"},
        "name": {"type": "string"},
        "description": {"type": "string"},
        "locked": {"type": "boolean"},
        "archived": {"type": "boolean"},
        "confidence": {"type": "number"},
        "actor": {"type": "string"},
        "action": {"type": "string"},
        "tags": {"type": "array", "items": {"type": "string"}},
        "primary": {"type": "string"},
        "tag_id": {"type": "string"},
        "tag_name": {"type": "string"},
        "tag_category": {"type": "string"},
        "into": {"type": "string"},
        "kind_name": {"type": "string"},
        "body": {"type": "string"},
        "convention_id": {"type": "string"},
    },
}


def build_library_tool(context: ToolContext | None = None) -> AgentTool:
    return AgentTool(
        name="library",
        label="Library",
        description=(
            "Long-term asset catalog across conversations: search, read, "
            "name/describe, tag, conventions. Local metadata writes only; "
            "see tool://library."
        ),
        parameters=LIBRARY_PARAMETERS,
        # Local metadata writes are reversible and the Librarian runs
        # unattended; the design's §5.1 disposition makes this a read-tier
        # tool. A future destructive Library op would be its own write tool.
        approval_tier="read",
        concurrency="exclusive",
        interruptible=False,
        execute=execute_library,
    )


__all__ = ["LIBRARY_PARAMETERS", "LibraryParams", "build_library_tool", "execute_library"]
