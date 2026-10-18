"""One scoped engine behind the tiny memory tool and memory:// read/write/edit.

File edit hunks are not a safe optimistic memory protocol. For this scheme only,
old_text is the revision token and new_text is a JSON replacement/forget command.
Routing precedes the filesystem resolver, so a typo cannot create memory:/ files.
"""

from __future__ import annotations

import asyncio
import json
import re
import sqlite3
from pathlib import Path
from typing import Any
from urllib.parse import quote

from local_operator.harness.types import TextContent, ToolContext, ToolResult
from local_operator.memory import (
    MEMORY_TOOL_CONTRACT_ERROR,
    MEMORY_TOOL_OPS,
    MEMORY_TOOL_REQUIRED,
    RECALL_TOKENS,
    MemoryError,
    MemoryIdentity,
    MemoryStore,
    guard,
    parse_url,
    safe_segment,
    structural_reference,
    valid_memory_tool_arguments,
)
from local_operator.paths import config_dir

#: The one ``q`` parameter of a read URL, matched at a parameter boundary so that only
#: its VALUE is ever left out of the text the free-text guards inspect — never the
#: scheme, the scope or another parameter.
_QUERY_VALUE = re.compile(r"(?<=[?&])q=[^&]*")


def store_for(context: ToolContext | None) -> MemoryStore:
    if context is None:
        raise MemoryError("Memory needs a host identity. Use lop memory for human inspection.")
    return MemoryStore(
        Path(context.memory_config_dir) if context.memory_config_dir else config_dir(),
        MemoryIdentity(context.session_id, context.team_id, context.agent_id),
        Path(context.session_dir) if context.session_dir else None,
    )


def _recall_reference(name: str, path: Any) -> str | None:
    """The read call's ``q`` VALUE when it IS a structural reference, else ``None``.

    The ONE exemption the agent read path has, and only ever a read: ``_run`` guards
    its whole argument dict, so the q value was refused here before the store saw it,
    and a benign identifier lookup failed on both layers. Every other argument of the
    same call — and the same value on a write or an edit, where a value is being
    STORED rather than looked up — keeps the refusal it has today.

    A path that does not parse is left to the caller's own ``parse_url``, which refuses
    it in the same order as before this exemption existed: ``ValueError`` here means
    only "not classifiable as a reference", so the caller's guard still speaks first.
    """
    if name != "read" or not isinstance(path, str):
        return None
    try:
        _, _, query = parse_url(path.strip())
    except (MemoryError, ValueError):
        # ValueError: ``parse_url`` parses its query with strict_parsing=True, which
        # rejects a field with no "=". Both are that same refusal by the caller's own
        # parser — catching them here keeps the guard ahead of it, as it was before.
        return None
    value = query.get("q")
    return value if structural_reference(value) else None


def _guarded_path(path: str) -> str:
    """``path`` with its exempt recall-query value left out of the guard's text.

    Only the exempt VALUE is removed — the parameter itself stays, empty, so every
    other character of the call (scheme, scope, entity, any other parameter) still
    reaches ``guard``/``guard_encoded`` unchanged. The value needs no placeholder: it
    was classified as a minted reference, so there is nothing in it to inspect.
    """
    return _QUERY_VALUE.sub("q=", path, count=1)


def _guarded_fact_args(args: Any) -> Any:
    """``args`` with a recall's ``text`` VALUE left out of the free-text guard.

    ``execute_fact_memory`` guards its WHOLE argument dict before it validates or
    delegates, so ``recall(<minted reference>)`` was refused HERE even after the read
    path below admitted the same value: its own store-oriented refusal spoke first.
    The exemption is that one VALUE of that one op — a ``store`` WRITES the same string
    and keeps the guard, because storing a value is not the same act as looking one up,
    and every sibling argument of the same call is inspected unchanged. The parameter
    itself stays, emptied, exactly as the read path keeps its ``q`` and drops only the
    value; there is nothing in an admitted reference to inspect.

    Only the guard is handed this copy. Validation, the store delegation and the
    retained-copy path all still see the caller's original arguments, and
    ``memory_privacy.retained_arguments`` guards those independently and fails closed.

    A non-dict, a call that is not a recall and a text that is not a reference all
    return ``args`` untouched, so their refusals — the contract error included — are
    unchanged.
    """
    if not isinstance(args, dict) or args.get("op") != "recall":
        return args
    return {**args, "text": ""} if structural_reference(args.get("text")) else args


def _run(name: str, args: dict[str, Any], context: ToolContext | None) -> str:
    from local_operator.memory_privacy import guard_encoded

    # Both guards below inspect the whole argument dict, so the single exemption this
    # path has is applied to the dict they receive rather than to the call. See
    # _recall_reference for why it exists and what it covers.
    reference = _recall_reference(name, args.get("path"))
    guarded = args if reference is None else {**args, "path": _guarded_path(args["path"])}
    guard(guarded)
    guard_encoded(guarded)
    allowed = {"path", "i", "raw"} if name == "read" else {"path", "i", "content"}
    if name == "edit":
        allowed = {"path", "i", "old_text", "new_text", "edits", "replace_all", "anchor_line"}
    if set(args) - allowed:
        raise MemoryError(
            "Unsupported memory argument; file ranges and filesystem controls do not apply."
        )
    scope, entity, query = parse_url(args["path"].strip())
    # The parsed query carries the same single exemption, on its value: the q value is
    # emptied and no other key of this call is affected.
    guard_encoded(query if reference is None else {**query, "q": ""})
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
            # A move is an EDIT command rather than a new verb: the fact is being
            # mutated, so it keeps the same optimistic revision guard as correction/
            # forget, and no new tool surface has to be discovered. Its one target is
            # the path's ID; {"move":"scope"} is the whole payload.
            move = replacement.get("move") if isinstance(replacement, dict) else None
            if move is not None:
                if set(replacement) != {"move"} or not isinstance(move, str):
                    raise MemoryError(
                        'Move requires exactly {"move":"session|team|agent|global"} and a revision.'
                    )
                if not scope or not entity:
                    raise MemoryError("Move needs memory://SCOPE/ID, not a bare scope.")
                store.import_session()
                value = store.move(
                    f"{scope}/{entity}",
                    move,
                    expected_version=args.get("old_text"),
                )
                # A write is not budgeted like a read: return the move receipt
                # directly rather than falling through to the recall-budget loop.
                return json.dumps(value, ensure_ascii=False, separators=(",", ":"))
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


async def execute_fact_memory(call_id, args, signal=None, on_update=None, context=None):
    """Plain-text affordance over the SAME structured engine; never a parallel store.

    Validate the entire fact before deriving a title. Clipping a long body into a
    permitted prefix would silently turn rejected input into a different stored fact.
    """
    from local_operator.memory_privacy import guard_encoded

    try:
        # The one exemption this tool has is applied to the dict the guard receives
        # rather than to the call; see _guarded_fact_args for why and what it covers.
        guard_encoded(_guarded_fact_args(args))
        if not valid_memory_tool_arguments(args):
            raise MemoryError(MEMORY_TOOL_CONTRACT_ERROR)
        text = args["text"]
        # The tool's declared domain is durable personal knowledge, bound to
        # the global bank. Scope-bearing calls are INVALID above, never silently
        # stripped/promoted. Work/checkpoints keep explicit four-scope URLs.
        path = "memory://global"
        if args["op"] == "store":
            guard(text, opaque=True)
            payload = {
                "title": text.strip().splitlines()[0][:100] if text.strip() else "",
                "body": text,
            }
            store_for(context).prepare_fact(payload)
            result = await execute_memory(
                call_id, "write", {"path": path, "content": json.dumps(payload)}, context
            )
        else:
            result = await execute_memory(
                call_id, "read", {"path": path + "?q=" + quote(text)}, context
            )
        if not result.is_error:
            value = json.loads(str(getattr(result.content[0], "text", "")))
            if args["op"] == "store":
                value.update(uri=path + "/" + value["id"], lifetime="global/across sessions")
            else:
                value = {"bank": "global/across sessions", "items": value}
                # Count the actual lifetime-labelled receipt inside the existing
                # recall ceiling. Drop whole return candidates only, never bodies
                # or stored facts; report paging rather than hiding that reduction.
                from local_operator.compaction.tokens import count_text_tokens

                while value["items"] and count_text_tokens(json.dumps(value)) > RECALL_TOKENS:
                    value["items"].pop()
                    value["truncated"] = True
            result = result.model_copy(update={"content": [TextContent(text=json.dumps(value))]})
        return result.model_copy(update={"tool_name": "memory"})
    except (MemoryError, ValueError, TypeError, OSError, sqlite3.Error) as exc:
        return ToolResult(
            tool_call_id=call_id,
            tool_name="memory",
            is_error=True,
            content=[
                TextContent(
                    text=(
                        str(exc)
                        if isinstance(exc, MemoryError)
                        else "Memory request refused; no value retained."
                    )
                )
            ],
        )


def build_memory_tool():
    """Always available by default, but never override an explicit role tool allowlist."""
    from local_operator.harness.types import AgentTool

    return AgentTool(
        name="memory",
        label="Memory",
        description=(
            "Store/recall durable personal facts across sessions: user/colleague names, "
            "locations, contacts. Temporary work: read/write/edit memory://session "
            "(or team/agent). "
            "Secrets/IDs: use secret; NAMEs only here."
        ),
        parameters={
            "type": "object",
            "additionalProperties": False,
            "properties": {
                "op": {"type": "string", "enum": list(MEMORY_TOOL_OPS)},
                "text": {"type": "string", "description": "Durable personal fact or query"},
            },
            "required": list(MEMORY_TOOL_REQUIRED),
        },
        approval_tier="write",
        call_approval_tier=lambda args: "read" if args.get("op") == "recall" else "write",
        describe_approval=lambda args, _context: (
            "Store durable personal fact in global memory (across sessions)"
            if args.get("op") == "store"
            else "Recall global personal memory"
        ),
        concurrency="exclusive",
        interruptible=False,
        execute=execute_fact_memory,
    )
