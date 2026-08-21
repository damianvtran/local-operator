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

from collections.abc import Callable, Sequence

from local_operator.harness.intent import apply_intent_schema
from local_operator.harness.types import AgentTool, ToolContext
from local_operator.tools import builtin
from local_operator.tools.agent_tool import build_agent_tool
from local_operator.tools.eval import build_eval_tool
from local_operator.tools.lsp import build_lsp_tool
from local_operator.tools.team_tool import build_team_delete_tool, build_team_tool
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
    "wake": lambda context: builtin.build_wake_tool(context),
    "task": lambda context: builtin.build_task_tool(context),
    "wait": lambda context: builtin.build_wait_tool(context),
    "jobs": lambda context: builtin.build_jobs_tool(context),
    "hub": lambda context: builtin.build_hub_tool(context),
    "ask": lambda context: builtin.build_ask_tool(context),
    "list_variables": lambda _context: builtin.build_list_variables_tool(),
    "read_variable": lambda _context: builtin.build_read_variable_tool(),
    "browser": lambda _context: builtin.build_browser_tool(_context),
    "agent": lambda context: build_agent_tool(context),
    "team": lambda context: build_team_tool(context),
    "team_delete": lambda context: build_team_delete_tool(context),
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
    "wake",
    "task",
    "wait",
    "jobs",
    "hub",
    "ask",
    "list_variables",
    "read_variable",
    "browser",
    "agent",
    "team",
    "team_delete",
]


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
            tool.parameters = apply_intent_schema(tool.parameters)
            tools.append(tool)
    return tools
