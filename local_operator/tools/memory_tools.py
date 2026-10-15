"""The structured memory:// branches of read/write/edit; no new tool schema.

File edit hunks are not a safe optimistic memory protocol. For this scheme only,
old_text is the revision token and new_text is a JSON replacement/forget command.
Routing precedes the filesystem resolver, so a typo cannot create memory:/ files.
"""

from __future__ import annotations

import asyncio
import json
import sqlite3
from pathlib import Path
from typing import Any

from local_operator.harness.types import TextContent, ToolContext, ToolResult
from local_operator.memory import (
    MemoryError,
    MemoryIdentity,
    MemoryStore,
    guard,
    parse_url,
    safe_segment,
)
from local_operator.paths import config_dir


def store_for(context: ToolContext | None) -> MemoryStore:
    if context is None:
        raise MemoryError("Memory needs a host identity. Use lop memory for human inspection.")
    return MemoryStore(
        Path(context.memory_config_dir) if context.memory_config_dir else config_dir(),
        MemoryIdentity(context.session_id, context.team_id, context.agent_id),
        Path(context.session_dir) if context.session_dir else None,
    )


def _run(name: str, args: dict[str, Any], context: ToolContext | None) -> str:
    from local_operator.memory_privacy import guard_encoded

    guard(args)
    guard_encoded(args)
    allowed = {"path", "i", "raw"} if name == "read" else {"path", "i", "content"}
    if name == "edit":
        allowed = {"path", "i", "old_text", "new_text", "edits", "replace_all", "anchor_line"}
    if set(args) - allowed:
        raise MemoryError(
            "Unsupported memory argument; file ranges and filesystem controls do not apply."
        )
    scope, entity, query = parse_url(args["path"].strip())
    guard_encoded(query)
    for key in ("version", "cursor"):
        if query.get(key):
            safe_segment(query[key])
    budget = int(query.get("tokens", 1200))
    if not 1 <= budget <= 2000:
        raise MemoryError("Explicit recall accepts 1–2000 tokens; default is 1200.")
    store = store_for(context)
    from local_operator.memory_capture import policy

    if not policy(store.root.parent)["enabled"]:
        raise MemoryError(
            "Scoped memory is disabled. Enable memory.enabled; "
            "human lop memory inspection remains available."
        )
    if scope:
        store.identity.owner(scope)
    if name == "read":
        if entity:
            if not scope or set(query) - {"related", "version", "tokens"}:
                raise MemoryError(
                    "Read an ID with optional related=true, version=REV or tokens only."
                )
            if "related" in query and query["related"] != "true":
                raise MemoryError("Related inspection accepts related=true only.")
            store.import_session()
            value = store.show(scope, entity, version=query.get("version"))
            if query.get("related") == "true":
                value["related"] = store.related(scope, entity)
            elif "related" in query:
                raise MemoryError("Related inspection accepts related=true only.")
        elif "q" in query:
            if (
                set(query) - {"q", "tokens", "limit"}
                or len(query["q"]) > 2048
                or not 1 <= int(query.get("limit", 10)) <= 20
            ):
                raise MemoryError("Search accepts q, 1–20 results and at most 2048 characters.")
            store.import_session()
            value = store.search(
                query["q"],
                scope,
                tokens=budget,
                limit=int(query.get("limit", 10)),
            )
        else:
            if (
                set(query) - {"tokens", "limit", "cursor"}
                or not 1 <= int(query.get("limit", 20)) <= 50
            ):
                raise MemoryError("Inventory accepts limit=1–50, cursor and tokens only.")
            store.import_session()
            value = store.inventory(
                scope, limit=int(query.get("limit", 20)), cursor=query.get("cursor")
            )
    else:
        if scope is None or query:
            raise MemoryError("Writes need an explicit scope and no query parameters.")
        if name == "write":
            if entity:
                raise MemoryError(
                    "Creation uses memory://SCOPE. Correct with edit and a revision, "
                    "never overwrite blindly."
                )
            payload = json.loads(args["content"])
            store.prepare_fact(payload)
            store.import_session()
            value = store.put(scope, payload)
        else:
            if (
                not entity
                or not args.get("old_text")
                or args.get("edits")
                or args.get("replace_all")
                or args.get("anchor_line")
            ):
                raise MemoryError(
                    "Memory edit needs an ID, old_text=revision, new_text=JSON; "
                    "file hunks are not supported."
                )
            replacement = json.loads(args.get("new_text") or "null")
            forget = replacement == {"forget": True, "confirm": True}
            if isinstance(replacement, dict) and "forget" in replacement and not forget:
                raise MemoryError(
                    'Forget requires {"forget":true,"confirm":true} and the current revision.'
                )
            if not forget:
                store.prepare_fact(replacement)
            store.import_session()
            value = store.put(
                scope,
                replacement,
                entity=entity,
                expected_version=args.get("old_text"),
                forget=forget,
            )

    def encoded() -> str:
        return json.dumps(value, ensure_ascii=False, separators=(",", ":"))

    if name == "read":
        from local_operator.compaction.tokens import count_text_tokens

        # Budget the ACTUAL tool text, not a compact estimate followed by a
        # larger pretty-printed result. Inventory/conflict/association pages
        # shrink explicitly; a fact body is never silently cut or changed.
        if isinstance(value, dict):
            for key in ("related", "conflicts", "items"):
                while value.get(key) and count_text_tokens(encoded()) > budget:
                    value[key].pop()
                    value["truncated"] = True
                    if key == "items":
                        value["next_cursor"] = value[key][-1]["id"] if value[key] else None
        elif isinstance(value, list):
            while value and count_text_tokens(encoded()) > budget:
                value.pop()
        if count_text_tokens(encoded()) > budget:
            raise MemoryError(
                "Result exceeds recall budget. Read one revision or increase tokens up to 2000."
            )
    return encoded()


async def execute_memory(
    call_id: str, name: str, args: dict[str, Any], context: ToolContext | None
) -> ToolResult:
    try:
        content = await asyncio.to_thread(_run, name, args, context)
        return ToolResult(
            tool_call_id=call_id,
            tool_name=name,
            content=[TextContent(text=content)],
            details={"memory": True},
        )
    except (MemoryError, ValueError, TypeError, OSError, sqlite3.Error) as exc:
        # JSON parser messages include source fragments in some implementations;
        # only our deliberately value-free refusals are safe to return verbatim.
        content = (
            str(exc)
            if isinstance(exc, MemoryError)
            else (
                "Invalid memory request or unavailable store. "
                "Read guide://memory; no filesystem fallback."
            )
        )
        return ToolResult(
            tool_call_id=call_id, tool_name=name, content=[TextContent(text=content)], is_error=True
        )
