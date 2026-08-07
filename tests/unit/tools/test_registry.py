"""Registry tests: factory table semantics and selection."""

from typing import Any, Coroutine, cast

from local_operator.harness.types import AgentTool, ToolContext, ToolResult
from local_operator.tools.registry import (
    DEFAULT_TOOL_NAMES,
    TOOL_BUILDERS,
    create_tools,
)


class _FakeScheduler:
    """Just enough surface for build_wake_tool's capability check."""

    @property
    def schedules(self) -> list[Any]:
        return []

    async def update(self, schedules) -> None:
        pass


def test_default_set_builds_all_builtin_tools() -> None:
    # With a wake scheduler attached, the default surface is the whole table.
    context = ToolContext(cwd=".", wake_scheduler=_FakeScheduler())
    tools = create_tools(context)
    names = [tool.name for tool in tools]
    assert names == DEFAULT_TOOL_NAMES
    assert all(isinstance(tool, AgentTool) for tool in tools)


def test_default_set_drops_wake_without_scheduler() -> None:
    # createIf: no scheduler -> no wake tool (never a tool that can only error).
    tools = create_tools(ToolContext(cwd="."))
    names = [tool.name for tool in tools]
    assert names == [name for name in DEFAULT_TOOL_NAMES if name != "wake"]


def test_every_tool_has_schema_and_metadata() -> None:
    for tool in create_tools(ToolContext(cwd=".", wake_scheduler=_FakeScheduler())):
        # JSON Schema derived from the pydantic params model
        assert tool.parameters.get("type") == "object"
        assert "properties" in tool.parameters
        # A zero-arg tool (e.g. list_variables) legitimately has no params.
        if tool.parameters["properties"] == {}:
            assert tool.name == "list_variables"
        else:
            assert tool.parameters.get("properties")
        # presentation + scheduling metadata are populated
        assert tool.label
        assert tool.description
        assert tool.approval_tier in ("read", "write", "exec")
        assert tool.concurrency in ("shared", "exclusive")


def test_concurrency_tiers_match_scheduling_model() -> None:
    # RT-02/RT-03/RT-04: scheduling classes. write/edit/todo/wake rewrite
    # shared state (a file, the todo list, the schedule list) so they run
    # exclusive; bash/read/glob/grep are independent batch work -> shared.
    tools = {t.name: t for t in create_tools(ToolContext(cwd="."))}
    assert tools["bash"].concurrency == "shared"
    assert tools["read"].concurrency == "shared"
    assert tools["glob"].concurrency == "shared"
    assert tools["grep"].concurrency == "shared"
    assert tools["write"].concurrency == "exclusive"
    assert tools["edit"].concurrency == "exclusive"


def test_bash_is_exec_shared_and_interruptible() -> None:
    tools = {t.name: t for t in create_tools(ToolContext(cwd="."))}
    bash = tools["bash"]
    assert bash.approval_tier == "exec"
    assert bash.concurrency == "shared"
    assert bash.interruptible is True


def test_tiers_match_contract() -> None:
    tools = {t.name: t for t in create_tools(ToolContext(cwd="."))}
    assert tools["read"].approval_tier == "read"
    assert tools["grep"].approval_tier == "read"
    assert tools["write"].approval_tier == "write"
    assert tools["edit"].approval_tier == "write"


def test_enabled_selects_and_orders() -> None:
    context = ToolContext(cwd=".")
    assert [t.name for t in create_tools(context, ["read", "bash"])] == ["read", "bash"]
    assert [t.name for t in create_tools(context, ["bash"])] == ["bash"]


def test_enabled_dedupes_names() -> None:
    # RT-25: duplicate names in host config must not duplicate provider tools.
    tools = create_tools(ToolContext(cwd="."), ["read", "read", "bash", "read"])
    assert [t.name for t in tools] == ["read", "bash"]


def test_unknown_names_are_skipped_not_raised() -> None:
    """Host config may name tools that do not exist; startup must not crash."""
    tools = create_tools(ToolContext(cwd="."), ["read", "does_not_exist", "write"])
    assert [t.name for t in tools] == ["read", "write"]


def test_enabled_empty_list_gives_no_tools() -> None:
    assert create_tools(ToolContext(cwd="."), []) == []


def test_default_names_cover_builder_table() -> None:
    """The default surface is the whole table today; drift is deliberate."""
    assert set(DEFAULT_TOOL_NAMES) == set(TOOL_BUILDERS)


# --- write/edit diff counters (the TUI's +N/-N indicators) -------------------


def test_line_delta_distinguishes_insert_delete_and_rewrite() -> None:
    """A length diff would call a same-size rewrite "no change"; a real match
    must report its churn, and a pure insert must not invent removals."""
    from local_operator.tools.builtin import _line_delta

    assert _line_delta("a\nb\n", "a\nx\nb\n") == (1, 0)
    assert _line_delta("a\nb\nc\n", "a\nc\n") == (0, 1)
    assert _line_delta("a\nb\n", "x\ny\n") == (2, 2)
    assert _line_delta("a\n", "a\n") == (0, 0)
    assert _line_delta("", "a\nb\n") == (2, 0)
    assert _line_delta("a\nb\n", "") == (0, 2)


def test_write_reports_diff_counts_for_new_and_overwritten_files(tmp_path) -> None:
    """The tool card renders +N/-N from these keys, so they must always be
    present on the success path (new file = all additions)."""
    import asyncio

    from local_operator.harness.types import ToolContext
    from local_operator.tools.builtin import execute_write

    ctx = ToolContext(cwd=str(tmp_path))
    created = asyncio.run(
        cast(
            Coroutine[Any, Any, ToolResult],
            execute_write("t1", {"path": "f.txt", "content": "a\nb\nc\n"}, None, None, ctx),
        )
    )
    assert created.details is not None
    assert created.details["added"] == 3
    assert created.details["removed"] == 0
    changed = asyncio.run(
        cast(
            Coroutine[Any, Any, ToolResult],
            execute_write("t2", {"path": "f.txt", "content": "a\nZ\nc\nd\n"}, None, None, ctx),
        )
    )
    assert changed.details is not None
    assert changed.details["added"] == 2
    assert changed.details["removed"] == 1


def test_edit_reports_diff_counts(tmp_path) -> None:
    import asyncio

    from local_operator.harness.types import ToolContext
    from local_operator.tools.builtin import execute_edit, execute_write

    ctx = ToolContext(cwd=str(tmp_path))
    asyncio.run(
        cast(
            Coroutine[Any, Any, ToolResult],
            execute_write("t0", {"path": "f.txt", "content": "a\nb\nc\n"}, None, None, ctx),
        )
    )
    result = asyncio.run(
        cast(
            Coroutine[Any, Any, ToolResult],
            execute_edit(
                "t1", {"path": "f.txt", "old_text": "b", "new_text": "B"}, None, None, ctx
            ),
        )
    )
    assert result.details is not None
    assert (result.details["added"], result.details["removed"]) == (1, 1)
