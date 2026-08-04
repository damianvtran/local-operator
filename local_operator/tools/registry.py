"""Tool registry — the createIf factory table for builtin tools.

Ports omp's registration model (``coding-agent/src/tools/index.ts``): a
``name -> factory`` table where each factory returns an :class:`AgentTool` or
``None`` when the tool cannot exist in this session (the *createIf*
convention — no separate capability table). ``create_tools`` walks the table
in a stable order so the provider-visible tool list is deterministic, which
matters for prompt-cache stability (the tools array rides in the same prefix
as the system prompt).

Legacy modules in this package (``general.py``, ``google.py``,
``screen_recorder.py``) belong to the old executed-Python tool system and are
NOT registered here.
"""

from __future__ import annotations

from collections.abc import Callable, Sequence

from local_operator.harness.types import AgentTool, ToolContext
from local_operator.tools import builtin

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
    "todo": lambda _context: builtin.build_todo_tool(),
    "wake": lambda context: builtin.build_wake_tool(context),
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
    "todo",
    "wake",
]


def create_tools(
    context: ToolContext, enabled: Sequence[str] | None = None
) -> list[AgentTool]:
    """Build the tool list for one session.

    ``enabled=None`` builds the default set; an explicit sequence selects from
    the table in the given order, first occurrence winning — duplicate names
    in host config must not produce duplicate provider tools. Names absent
    from the table are skipped — unknown tool names in host config must not
    crash session startup (omp resolves availability at creation time, never
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
            tools.append(tool)
    return tools
