"""Tool registry — the createIf factory table for builtin tools.

The registration model is a
``name -> factory`` table where each factory returns an :class:`AgentTool` or
``None`` when the tool cannot exist in this session (the *createIf*
convention — no separate capability table). ``create_tools`` walks the table
in a stable order so the provider-visible tool list is deterministic, which
matters for prompt-cache stability (the tools array rides in the same prefix
as the system prompt).
"""

from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
from typing import Any, NamedTuple

from local_operator.code_requests.tool import build_code_requests_tool
from local_operator.harness.intent import apply_intent_schema
from local_operator.harness.types import AgentTool, ToolContext
from local_operator.network.tool import build_network_tool
from local_operator.tools import builtin
from local_operator.tools.agent_tool import build_agent_tool
from local_operator.tools.eval import build_eval_tool
from local_operator.tools.image_tool import build_generate_image_tool
from local_operator.tools.lsp import build_lsp_tool
from local_operator.tools.project_tool import (
    build_project_delete_tool,
    build_project_tool,
)
from local_operator.tools.secret_tool import build_secret_tool
from local_operator.tools.team_tool import build_team_delete_tool, build_team_tool
from local_operator.web_fetch.tool import build_web_fetch_tool
from local_operator.web_search.read_tool import build_web_read_tool
from local_operator.web_search.tool import build_web_search_tool

#: Factory table: tool name -> builder (createIf convention). ``wake`` takes
#: the context and returns ``None`` when no wake scheduler is attached, so a
#: session without wakes never advertises a tool that can only error; the
#: table order below is also the provider-visible tool order.
TOOL_BUILDERS: dict[str, Callable[[ToolContext], AgentTool | None]] = {
    "bash": lambda _context: builtin.build_bash_tool(),
    "read": lambda _context: builtin.build_read_tool(),
    "write": lambda _context: builtin.build_write_tool(),
    "edit": lambda _context: builtin.build_edit_tool(),
    "glob": lambda _context: builtin.build_glob_tool(),
    "grep": lambda _context: builtin.build_grep_tool(),
    "eval": lambda _context: build_eval_tool(),
    "lsp": lambda _context: build_lsp_tool(),
    "todo": lambda _context: builtin.build_todo_tool(),
    "web_search": lambda context: build_web_search_tool(context),
    "web_read": lambda context: build_web_read_tool(context),
    "web_fetch": lambda context: build_web_fetch_tool(context),
    "wake": lambda context: builtin.build_wake_tool(context),
    # createIf: proactive-class sessions with a scheduler only (rung 3 — a
    # reactive session pays no schema for a capability it cannot use).
    "patience": lambda context: builtin.build_patience_tool(context),
    "task": lambda context: builtin.build_task_tool(context),
    "wait": lambda context: builtin.build_wait_tool(context),
    "jobs": lambda context: builtin.build_jobs_tool(context),
    "hub": lambda context: builtin.build_hub_tool(context),
    # Unconditional createIf entry: peer messaging rides the registry + loopback
    # substrate every session sits on, so the tool exists in every session even
    # when no peer happens to be running right now (see build_send_tool).
    "send": lambda context: builtin.build_send_tool(context),
    "ask": lambda context: builtin.build_ask_tool(context),
    # createIf: returns None where the encrypted secret store is unreachable,
    # so a session that cannot use it pays no schema for it (design §5.2).
    "secret": lambda context: build_secret_tool(context),
    "list_variables": lambda _context: builtin.build_list_variables_tool(),
    "read_variable": lambda _context: builtin.build_read_variable_tool(),
    "browser": lambda _context: builtin.build_browser_tool(_context),
    # createIf: returns None unless the desktop app publishes a console-capable
    # host. One row, one predicate, and NO settings row for availability — the
    # tool is absent, not hidden, where the app cannot serve it (design
    # ui-console-tab §14.1/§14.2).
    "console": lambda _context: builtin.build_console_tool(_context),
    "agent": lambda context: build_agent_tool(context),
    "team": lambda context: build_team_tool(context),
    "team_delete": lambda context: build_team_delete_tool(context),
    # Unconditional entry, appended rather than inserted so the array's prefix —
    # which the prompt cache keys on — is unchanged for every existing session:
    # `lop network init` is how a first network comes into existence, so a gate
    # on "a relay is configured" would strip the tool from exactly the session
    # that has to create one (see build_network_tool).
    "network": lambda context: build_network_tool(context),
    # createIf: projects need the store beside them; both tools are absent — not
    # hidden — in a session without a registry (mirroring team/team_delete).
    # Appended at the END of both tables on purpose: appending never shifts a
    # provider-visible array prefix, which is what the prompt cache keys on.
    "project": lambda context: build_project_tool(context),
    "project_delete": lambda context: build_project_delete_tool(context),
    # createIf: returns None without a monitor scheduler, so a session that
    # cannot arm a monitor pays no schema for it (footprint rung 3, like
    # `wake`). Appended at the end of both tables for the same cache-prefix
    # reason the two rows above are (design monitor-tool.md §19.1).
    "monitor": lambda context: builtin.build_monitor_tool(context),
    # createIf: rung 3 — only a session that can hold the delegation surface
    # builds it (`context.subagent_launcher is not None`), so a context that
    # cannot delegate pays no schema for it. A built top-level session that
    # cannot delegate still gets it and its `spawn` is refused per call by the
    # CLI guard — the accepted cost of not inventing a second gating
    # convention (design sessions-tool.md §3.3). Appended at the END of both
    # tables on purpose: appending never shifts a provider-visible array
    # prefix, which is what the prompt cache keys on.
    "sessions": lambda context: builtin.build_sessions_tool(context),
    # createIf: rung 3 — the agent-side settle exists only where the queued
    # engine does (`context.withdraw_ask`), so the blocking arm, headless hosts
    # and subagents pay zero schema for it (design ask-nonblocking.md §12).
    # Appended at the END of both tables for the same cache-prefix reason.
    "ask_withdraw": lambda context: builtin.build_ask_withdraw_tool(context),
    # createIf: rung 3 — present only where an image provider is reachable
    # (`imagegen.availability`: persisted rows / store rows / env; sync and
    # socket-free), so a session that cannot generate pays no schema for it
    # (design image-gen §2.4/D9). Appended at the END of both tables on
    # purpose: appending never shifts a provider-visible array prefix, which
    # is what the prompt cache keys on.
    "generate_image": lambda context: build_generate_image_tool(context),
    # createIf: rung 3 — present only where the session has a store root
    # (its own directory plus a resolvable config root), so a reduced host
    # pays no schema for a tool whose every call could only say "no store".
    # Read-only (`list`/`show`) and deferred by default (its schema is the
    # biggest part of its cost; the recommendation hook and the tracked note
    # are what keep it discoverable). Appended at the END of both tables on
    # purpose: appending never shifts a provider-visible array prefix, which
    # is what the prompt cache keys on.
    "code_requests": lambda context: build_code_requests_tool(context),
    # createIf: rung 3 — present only where the session may END A TURN QUIETLY
    # (`context.quiet_end`; absent for subagent children, one-shot hosts,
    # output-contract sessions and under LOP_NO_REPLY=0 — see
    # `Session._quiet_end_callable` and docs/design/quiet-turns.md §4).
    # Deferred (its schema is never needed to form its argumentless call, and
    # the rule is named in the system prompt). Appended at the END of both
    # tables on purpose: appending never shifts a provider-visible array
    # prefix, which is what the prompt cache keys on.
    "no_reply": lambda context: builtin.build_no_reply_tool(context),
}

#: Tool set used when the session does not restrict the names. Kept explicit
#: (not ``list(TOOL_BUILDERS)``) so the default surface is a deliberate
#: decision, and hidden/discoverable tools can join the table later without
#: silently entering every session.
DEFAULT_TOOL_NAMES: list[str] = [
    "bash",
    "read",
    "write",
    "edit",
    "glob",
    "grep",
    "eval",
    "lsp",
    "todo",
    "web_search",
    "web_read",
    "web_fetch",
    "wake",
    "patience",
    "task",
    "wait",
    "jobs",
    "hub",
    "send",
    "ask",
    "secret",
    "list_variables",
    "read_variable",
    "browser",
    "console",
    "agent",
    "team",
    "team_delete",
    "network",
    "project",
    "project_delete",
    "monitor",
    "sessions",
    "ask_withdraw",
    "generate_image",
    "code_requests",
    "no_reply",
]


_NULL_BRANCH: dict[str, Any] = {"type": "null"}


class CollapsedSchema(NamedTuple):
    """A rewritten schema plus the TOP-LEVEL property names it rewrote.

    The two travel together because they must: the loop's validator has to skip
    exactly the properties this rewrite put a top-level ``type`` on, and two
    functions computing that independently would drift. ``unchecked`` is a flat
    name set because the validator checks top-level arguments only — a property
    inside a nested ``$defs`` entry still reports its own name, which is the
    key the loop would look it up by.
    """

    parameters: dict[str, Any]
    unchecked: frozenset[str]


def collapse_optional_nulls(schema: Any) -> CollapsedSchema:
    """Rewrite pydantic's optional-field shape to the plain type it wraps.

    Every ``x: T | None = None`` field renders as
    ``{"anyOf": [<T>, {"type": "null"}], "default": null, "description": …}``.
    The null branch and the ``default: null`` restate what NOT listing the
    property in ``required`` already says — the model may leave it out — and
    the default surface carried 121 of them (123 declarations of that shape, of
    which two are REQUIRED and therefore kept). This rewrites each optional one
    to ``{<T>, "description": …}``.

    ONLY the optional, single-branch case: a REQUIRED nullable property (where
    ``null`` is a meaningful value the model must be able to send) and a
    multi-branch union are left exactly as generated.

    EVERY REWRITE IS REPORTED in :attr:`CollapsedSchema.unchecked` and the
    loop's validator SKIPS that property. ``unchecked`` holds the ROOT-level
    names only, because ``validate_tool_arguments`` reads the root
    ``properties`` and nothing else — a rewrite inside a nested ``$defs`` entry
    is never type-checked by the loop in the first place, so reporting it would
    put a name in the set that no lookup can match. That is what keeps the rewrite honest:
    an optional ``T | None`` had no top-level ``type``, so the loop checked
    nothing and the tool's own pydantic model decided — including coercers tools
    ship on purpose (``hub``'s ``to`` takes a bare id or its JSON; ``jobs``'
    ``job_id`` takes a number). Without it the collapsed ``type`` would make the
    loop enforce a type the tool would have coerced, refusing the call before
    the tool ran (review round 1, MAJOR-1). The names stay HOST-SIDE on
    ``AgentTool.optional_null_unions``: an in-schema marker rode every published
    request for a third of this feature's saving (review round 2, MAJOR-2).

    Builtins only: called from :func:`create_tools`, never on an MCP server's
    schema, which is the server's contract to state.
    """
    unchecked: set[str] = set()

    def walk(node: Any, *, root: bool = False) -> Any:
        if isinstance(node, list):
            return [walk(item) for item in node]
        if not isinstance(node, dict):
            return node
        out: dict[str, Any] = {}
        required = set(node.get("required") or ())
        for key, value in node.items():
            if key == "properties" and isinstance(value, Mapping):
                properties: dict[str, Any] = {}
                for name, prop in value.items():
                    if name in required:
                        properties[name] = walk(prop)
                        continue
                    collapsed, changed = _collapse_one(prop)
                    if changed and root:
                        unchecked.add(name)
                    properties[name] = walk(collapsed)
                out[key] = properties
            else:
                out[key] = walk(value)
        return out

    return CollapsedSchema(walk(schema, root=True), frozenset(unchecked))


def _collapse_one(prop: Any) -> tuple[Any, bool]:
    """``(property, collapsed)`` for one optional-null union; unchanged otherwise."""
    if not isinstance(prop, dict):
        return prop, False
    branches = prop.get("anyOf")
    if not isinstance(branches, list) or len(branches) != 2 or _NULL_BRANCH not in branches:
        return prop, False
    if "default" in prop and prop["default"] is not None:
        return prop, False
    kept = [branch for branch in branches if branch != _NULL_BRANCH]
    if len(kept) != 1 or not isinstance(kept[0], dict):
        return prop, False
    if set(kept[0]) & set(prop) - {"anyOf"}:
        # A key on both levels (a description inside the branch AND beside it)
        # would have to be merged by preference; not a shape pydantic emits, so
        # leave it rather than guess.
        return prop, False
    collapsed = {key: value for key, value in prop.items() if key not in ("anyOf", "default")}
    return {**kept[0], **collapsed}, True


def create_tools(context: ToolContext, enabled: Sequence[str] | None = None) -> list[AgentTool]:
    """Build the tool list for one session.

    ``enabled=None`` builds the default set; an explicit sequence selects from
    the table in the given order, first occurrence winning — duplicate names
    in host config must not produce duplicate provider tools. Names absent
    from the table are skipped — unknown tool names in host config must not
    crash session startup (availability resolves at creation time, never
    at dispatch time).
    """
    if enabled is None:
        names: list[str] = list(DEFAULT_TOOL_NAMES)
    else:
        names = list(dict.fromkeys(enabled))
    tools: list[AgentTool] = []
    for name in names:
        builder = TOOL_BUILDERS.get(name)
        if builder is None:
            continue
        tool = builder(context)
        if tool is not None:
            # The `i` intent property is added HERE, not in each params model:
            # one choke point cannot grow holes as tools are added, and a
            # working line that narrates intent for some calls and mechanics
            # for the rest is worse than one that never tries. The transform
            # only prepends a property inside `parameters`; the tool list this
            # function returns keeps its order, which the prompt cache depends
            # on (see the module docstring).
            collapsed = collapse_optional_nulls(tool.parameters)
            # The names stay on the tool and OFF the wire: the validator reads
            # them here, and ``exclude=True`` keeps them out of every dump.
            tool.optional_null_unions = collapsed.unchecked
            tool.parameters = apply_intent_schema(collapsed.parameters)
            tools.append(tool)
    return tools
