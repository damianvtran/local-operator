"""The structured memory:// branches of read/write/edit; no new tool schema.

File edit hunks are not a safe optimistic memory protocol. For this scheme only,
old_text is the revision token and new_text is a JSON replacement/forget command.
Routing precedes the filesystem resolver, so a typo cannot create memory:/ files.
"""

from __future__ import annotations

import asyncio
import json
from pathlib import Path
from typing import Any

from local_operator.harness.types import TextContent, ToolContext, ToolResult
from local_operator.memory import (
    MemoryError,
    MemoryIdentity,
    MemoryStore,
    guard,
    parse_url,
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
    guard(args)
    scope, entity, query = parse_url(args["path"].strip())
    store = store_for(context)
    store.import_session()
    if name == "read":
        if entity:
            if not scope or query:
                raise MemoryError("Read an ID without search parameters.")
            value = store.show(scope, entity)
        elif "q" in query:
            value = store.search(
                query["q"],
                scope,
                tokens=int(query.get("tokens", 1200)),
                limit=int(query.get("limit", 10)),
            )
        else:
            value = store.inventory(scope, limit=int(query.get("limit", 20)))
    else:
        if scope is None or query:
            raise MemoryError("Writes need an explicit scope and no query parameters.")
        if name == "write":
            if entity:
                raise MemoryError(
                    "Creation uses memory://SCOPE. Correct with edit and a revision, "
                    "never overwrite blindly."
                )
            value = store.put(scope, json.loads(args["content"]))
        else:
            if (
                not entity
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
            value = store.put(
                scope,
                replacement,
                entity=entity,
                expected_version=args.get("old_text"),
                forget=forget,
            )
    return json.dumps(value, ensure_ascii=False, indent=2)


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
    except (MemoryError, ValueError, TypeError, OSError) as exc:
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
