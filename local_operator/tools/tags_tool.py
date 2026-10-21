"""The ``tags`` tool: session tag categories for the agent caller.

This is the V2-4 agent surface from the sidebar-tags design (§7.1), and the
only AGENT write path for tag vocabulary. It deliberately does not add a fourth
spelling to ``sessions``: category lifecycle (``create``/``rename``) is not a
session act, and a model that cannot create a category must not be able to
invent one during an assignment.

The store and its validation are the shipped ones (:mod:`local_operator.session
.session_tags`, the same module the desktop routes and ``sessions spawn tags``
use), so a refusal here and a refusal over HTTP are the same rule rather than
two implementations that can drift:

* identity lives in the registry; assignment order owns the primary fallback;
* names are exact, case-insensitive, and unique;
* ``Untagged`` is reserved, 1-24 characters, at most 32 categories;
* at most 8 assignments per session, no duplicates.

``assign`` takes category NAMES (matching the shipped ``sessions spawn tags``)
and never IDs: ids appear in ``list`` for display and for ``rename`` only. A
misspelled name is refused with the known vocabulary — it is never created, and
``create`` is the only creation path. An empty ``tags`` list clears the
session's assignments, because the store's own semantics are a full desired
state.

Every successful ``assign`` receipt names the categories and the session;
``create``/``list``/``rename`` receipts name categories only. That split is what
makes the tool ledger the audit trail for session assignments; nothing here
relies on a separate log line (a refusal names whichever half is available and
carries no audit claim).
"""

from __future__ import annotations

import asyncio
import re
from pathlib import Path
from typing import Any, Callable, Literal

from pydantic import BaseModel, ConfigDict, Field, ValidationError

from local_operator.harness.types import (
    AbortSignal,
    AgentTool,
    AgentToolUpdate,
    ToolContext,
    ToolResult,
)
from local_operator.info.render import plural
from local_operator.paths import config_dir
from local_operator.session.session_tags import (
    MAX_CATEGORIES,
    MAX_NAME_CHARS,
    RESERVED_NAME,
    read_categories,
    read_entity_tags,
    resolve_category_names,
    set_entity_tags,
    usage_counts,
    write_categories,
)
from local_operator.tools.builtin import (
    _error,
    _guard,
    _text,
    _validation_error,
    identifier_text,
)

#: The desktop session-id contract, matching ``session_tags._entity_path`` and
#: the route's own admission check. Defined here rather than imported from the
#: server layer: the tool must not depend on ``local_operator.server``, and the
#: store deliberately does not import the catalogue.
_SESSION_ID_RE = re.compile(r"[a-f0-9]{12}")


class TagsParams(BaseModel):
    """Parameters for the ``tags`` tool (see ``_TAGS_TOOL_DESCRIPTION``)."""

    model_config = ConfigDict(extra="forbid")

    op: Literal["list", "create", "rename", "assign"] = Field(
        description="list; create; rename; assign."
    )
    name: str | None = Field(
        default=None,
        description="create/rename: category name (1-24 chars; 'Untagged' reserved).",
    )
    id: str | None = Field(default=None, description="rename: category id from `list`.")
    session_id: str | None = Field(default=None, description="assign: exact local session id.")
    tags: list[str] | None = Field(
        default=None,
        description=(
            "assign: existing category NAMES, exact case-insensitive, max 8; "
            "an empty list clears. Never ids."
        ),
    )
    primary: str | None = Field(
        default=None,
        description=(
            "assign: one of the assigned names; omit to keep the current primary "
            "(or the first when none survives)."
        ),
    )


_TAGS_TOOL_DESCRIPTION = (
    "List, create, rename and assign the operator's session tag categories. "
    "`assign` takes category NAMES only (ids are for display and `rename`); a "
    "misspelled name is refused, never created — `create` first, then assign."
)


def _sentence(text: str) -> str:
    """Upper-case only the first character.

    ``str.capitalize()`` is NOT this: it lower-cases the rest of the sentence,
    which turned a stored ``Review`` into ``review`` in a refusal. Names are
    operator data; a message may capitalize its own leading word and nothing
    else.
    """
    return text[:1].upper() + text[1:]


def _known_categories(categories: list[dict[str, str]]) -> str:
    """The vocabulary a refusal quotes, so one round can correct the call."""
    if not categories:
        return "no categories exist yet — create one with op='create'"
    return "known categories: " + ", ".join(
        f"{item['name']} (id {item['id']})" for item in categories
    )


def _name_refusal(
    name: str, categories: list[dict[str, str]], *, ignoring_id: str | None = None
) -> str | None:
    """Validate a category name against the store's rules, in readable words.

    ``write_categories`` raises one combined sentence for all of these; the
    tool repeats the checks so a refusal can name the offending value and the
    existing category that collides. The store's write remains the enforcement
    point (a race can still raise), so this is a better message, not a second
    authority — callers keep the ``ValueError`` branch below.
    """
    stripped = name.strip()
    problem = ""
    if not stripped:
        problem = "A category name is required."
    elif len(stripped) > MAX_NAME_CHARS:
        problem = (
            f"Category names are 1-{MAX_NAME_CHARS} characters; "
            f"{stripped!r} is {len(stripped)}."
        )
    elif stripped.casefold() == RESERVED_NAME.casefold():
        problem = f"'{RESERVED_NAME}' is reserved for the untagged view; choose another name."
    else:
        for item in categories:
            if item["id"] == ignoring_id:
                continue
            if item["name"].casefold() == stripped.casefold():
                problem = (
                    f"A category named '{item['name']}' already exists "
                    f"(id {item['id']}). Use op='rename' to change it or "
                    "op='assign' to apply it."
                )
                break
    if not problem:
        return None
    # Every refusal names the live vocabulary, so one round can correct the
    # call without a second `list` — and a misspelling never looks creatable.
    return f"{problem} {_sentence(_known_categories(categories))}."


def _category_by_id(categories: list[dict[str, str]], key: str) -> dict[str, str] | None:
    for item in categories:
        if item["id"] == key:
            return item
    return None


@_guard("tags")
async def execute_tags(
    tool_call_id: str,
    args: dict[str, Any],
    signal: AbortSignal | None = None,
    on_update: Callable[[AgentToolUpdate], None] | None = None,
    context: ToolContext | None = None,
) -> ToolResult:
    try:
        params = TagsParams(**args)
    except ValidationError as exc:
        return _validation_error(tool_call_id, "tags", exc)

    root = config_dir()
    categories = await asyncio.to_thread(read_categories, root)

    if params.op == "list":
        counts = await asyncio.to_thread(usage_counts, root)
        rows = [
            {
                **item,
                "order": order,
                "conversations": counts.get(item["id"], 0),
            }
            for order, item in enumerate(categories)
        ]
        if not rows:
            return _text(
                tool_call_id,
                "tags",
                "No tag categories yet. Create one with tags(op='create', name='...').",
                details={"tags": [], "count": 0},
            )
        lines = [
            f"{row['order']}. {identifier_text(context, row['name'])} — id {row['id']}, "
            f"{plural(row['conversations'], 'conversation')}"
            for row in rows
        ]
        heading = "1 tag category:" if len(rows) == 1 else f"{len(rows)} tag categories:"
        return _text(
            tool_call_id,
            "tags",
            heading + "\n" + "\n".join(lines),
            details={"tags": rows, "count": len(rows)},
        )

    if params.op == "create":
        requested = params.name or ""
        refusal = _name_refusal(requested, categories)
        if refusal is not None:
            return _error(tool_call_id, "tags", refusal)
        if len(categories) >= MAX_CATEGORIES:
            return _error(
                tool_call_id,
                "tags",
                f"The store already holds the maximum {MAX_CATEGORIES} categories; "
                f"rename one instead. {_sentence(_known_categories(categories))}.",
            )
        # Preserve the existing rows by id (a create must not reorder or
        # reassign them); the name-only entry is the documented append path.
        desired: list[tuple[str | None, str]] = [(item["id"], item["name"]) for item in categories]
        desired.append((None, requested.strip()))
        try:
            written = await asyncio.to_thread(write_categories, root, desired)
        except ValueError as exc:
            return _error(tool_call_id, "tags", f"{exc}. {_known_categories(categories)}.")
        created = written[-1]
        return _text(
            tool_call_id,
            "tags",
            f"Created category '{identifier_text(context, created['name'])}' "
            f"(id {created['id']}); {len(written)} categor"
            f"{'y' if len(written) == 1 else 'ies'} in the store.",
            details={"op": "create", "id": created["id"], "name": created["name"]},
        )

    if params.op == "rename":
        key = (params.id or "").strip()
        if not key:
            return _error(
                tool_call_id,
                "tags",
                f"rename needs the category id. {_sentence(_known_categories(categories))}.",
            )
        existing = _category_by_id(categories, key)
        if existing is None:
            return _error(
                tool_call_id,
                "tags",
                f"Unknown category id '{key}'. {_sentence(_known_categories(categories))}.",
            )
        requested = params.name or ""
        refusal = _name_refusal(requested, categories, ignoring_id=key)
        if refusal is not None:
            return _error(tool_call_id, "tags", refusal)
        new_name = requested.strip()
        if existing["name"] == new_name:
            return _text(
                tool_call_id,
                "tags",
                f"Category '{identifier_text(context, existing['name'])}' already has "
                f"that name (id {key}); nothing changed.",
                details={"op": "rename", "id": key, "name": new_name, "changed": False},
            )
        desired: list[tuple[str | None, str]] = [
            (item["id"], new_name if item["id"] == key else item["name"]) for item in categories
        ]
        try:
            await asyncio.to_thread(write_categories, root, desired)
        except ValueError as exc:
            return _error(tool_call_id, "tags", f"{exc}. {_known_categories(categories)}.")
        return _text(
            tool_call_id,
            "tags",
            f"Renamed category '{identifier_text(context, existing['name'])}' to "
            f"'{identifier_text(context, new_name)}' (id {key}); assignments keep the id.",
            details={
                "op": "rename",
                "id": key,
                "name": new_name,
                "previous": existing["name"],
                "changed": True,
            },
        )

    # params.op == "assign" — the only remaining Literal value.
    session_id = (params.session_id or "").strip()
    if not _SESSION_ID_RE.fullmatch(session_id):
        return _error(
            tool_call_id,
            "tags",
            "assign needs an exact session id (12 lowercase hex characters).",
        )
    if params.tags is None:
        return _error(
            tool_call_id,
            "tags",
            "assign needs tags: existing category names, or [] to clear.",
        )
    names = [str(name) for name in params.tags]
    primary_name = (params.primary or "").strip() or None
    if not names:
        if primary_name is not None:
            return _error(
                tool_call_id,
                "tags",
                "primary must be omitted when tags is empty (an empty list clears).",
            )
        tag_ids: list[str] = []
        primary_id: str | None = None
    else:
        try:
            tag_ids = resolve_category_names(root, names, categories=categories)
        except ValueError as exc:
            return _error(tool_call_id, "tags", str(exc))
        # An EXPLICIT primary is resolved here; an omitted one is resolved
        # below, after the session's current assignment is read, because the
        # store's own fallback is assignment order (``ordered[0]``) and would
        # move a second-position primary. Only an explicit name is chosen here.
        primary_id = None
        if primary_name is not None:
            folded = primary_name.casefold()
            chosen = [
                key
                for key, name in zip(tag_ids, names, strict=True)
                if name.strip().casefold() == folded
            ]
            if not chosen:
                return _error(
                    tool_call_id,
                    "tags",
                    f"primary '{primary_name}' must be one of the assigned names "
                    f"({', '.join(name.strip() for name in names)}).",
                )
            primary_id = chosen[0]

    session_dir = Path(root) / "sessions" / session_id
    if not session_dir.is_dir():
        # Remote/mesh parity with the route: a peer's row is refused in words
        # and with the route's code, never silently written here. The lookup is
        # cache-first and pays no relay read when this device holds the id (it
        # does not reach this branch then) or when no network is configured.
        from local_operator.session.remote_open import remote_row_for

        remote = await asyncio.to_thread(remote_row_for, session_id, root)
        if remote is not None:
            holder = remote.owner_device_name or remote.owner_device
            return _error(
                tool_call_id,
                "tags",
                f"Session {session_id} lives on {holder}; tag assignments are local "
                "only (409 session_is_remote).",
                details={"op": "assign", "session_id": session_id, "code": "session_is_remote"},
            )
        return _error(
            tool_call_id,
            "tags",
            f"Unknown session {session_id}: no such directory on this machine and "
            "no mesh device holds it.",
            details={"op": "assign", "session_id": session_id},
        )

    # OMITTED PRIMARY MUST RETAIN, not fall back to assignment order. The
    # store resolves ``primary=None`` as ``ordered[0]``, which is the OLDEST
    # surviving assignment — the desktop UI may have chosen any member, so a
    # later agent call that omits ``primary`` would silently move the session
    # out of the operator's group (``effective_primary`` reads this field).
    # Passing the current primary through when it survives keeps "adding
    # another tag never moves it" true across surfaces. Test the NORMALIZED
    # name, not the raw parameter: the published schema has no ``minLength``,
    # so a model encoding "no primary" as ``""`` (or whitespace) is a routine
    # call shape, and keying retention on ``params.primary is None`` sent that
    # shape down the store's ``ordered[0]`` fallback instead (R-5).
    if primary_name is None:
        current = await asyncio.to_thread(
            read_entity_tags, root, "session", session_id, categories=categories
        )
        retained = current["primary"]
        primary_id = retained if retained in tag_ids else None

    # DIRECTORY ADMISSION, deliberately: delegated and hidden runs are taggable
    # (the route's own note says this is not the user-conversation visibility
    # predicate). The store's single write is the atomic step: an assignment
    # either lands whole or not at all.
    try:
        state = await asyncio.to_thread(
            set_entity_tags, root, "session", session_id, tag_ids, primary_id
        )
    except ValueError as exc:
        return _error(
            tool_call_id,
            "tags",
            f"{exc}. {_sentence(_known_categories(categories))}.",
        )

    # Re-read the vocabulary for the receipt: a concurrent rename may have
    # changed a display name, and the result must name what is now in the
    # store rather than what the call happened to validate against.
    current = await asyncio.to_thread(read_categories, root)
    lookup = {item["id"]: item["name"] for item in current}
    assigned = [lookup[key] for key in state["tags"] if key in lookup]
    primary = lookup.get(state["primary"]) if state["primary"] else None
    details = {
        "op": "assign",
        "session_id": session_id,
        "tags": assigned,
        "primary": primary,
        "tags_revision": str(state["revision"]),
    }
    if not assigned:
        return _text(
            tool_call_id,
            "tags",
            f"Cleared tags on session {session_id}.",
            details=details,
        )
    rendered = ", ".join(identifier_text(context, name) for name in assigned)
    primary_text = f" (primary {identifier_text(context, primary)})" if primary is not None else ""
    return _text(
        tool_call_id,
        "tags",
        f"Tagged session {session_id} with {plural(len(assigned), 'category')}: "
        f"{rendered}{primary_text}.",
        details=details,
    )


def build_tags_tool() -> AgentTool:
    """Unconditional: the tag store sits beside every session's config root.

    Mirrors the desktop route's availability rather than inventing a gate — a
    session that can see its own config directory can use the store. The static
    tier is ``read`` for the same reason ``project``/``library`` are: these are
    local metadata writes, and the destructive act (deleting a category) is not
    in this tool at all.
    """
    return AgentTool(
        name="tags",
        label="Tags",
        description=_TAGS_TOOL_DESCRIPTION,
        parameters=TagsParams.model_json_schema(),
        approval_tier="read",
        # Category writes are read-modify-write over one registry file; the
        # store's in-process lock cannot serialize two sibling calls, so the
        # tool takes the slot rather than racing itself.
        concurrency="exclusive",
        interruptible=False,
        execute=execute_tags,
    )
