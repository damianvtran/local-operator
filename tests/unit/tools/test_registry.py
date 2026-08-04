"""Registry tests: factory table semantics and selection."""

from local_operator.harness.types import AgentTool, ToolContext
from local_operator.tools.registry import (
    DEFAULT_TOOL_NAMES,
    TOOL_BUILDERS,
    create_tools,
)


class _FakeScheduler:
    """Just enough surface for build_wake_tool's capability check."""

    @property
    def schedules(self) -> list:
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
        assert tool.parameters.get("properties")
        # presentation + scheduling metadata are populated
        assert tool.label
        assert tool.description
        assert tool.approval_tier in ("read", "write", "exec")
        assert tool.concurrency in ("shared", "exclusive")


def test_concurrency_tiers_match_omp_model() -> None:
    # RT-02/RT-03/RT-04: omp-derived scheduling classes. write/edit/todo/wake
    # rewrite shared state (a file, the todo list, the schedule list) so they
    # run exclusive; bash/read/glob/grep are independent batch work -> shared.
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
