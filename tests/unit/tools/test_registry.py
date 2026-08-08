"""Registry tests: factory table semantics and selection."""

from typing import Any, Coroutine, cast

import pytest

from local_operator.harness.types import AgentTool, ToolContext, ToolResult
from local_operator.tools import builtin
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


class _FakeJobs:
    """Just enough surface for the job-tracking tools' capability check."""

    def get(self, job_id: str, *, owner_id: str | None = None) -> Any:
        return None

    def list(self, *, owner_id: str | None = None) -> list[Any]:
        return []

    async def cancel(self, job_id: str, *, owner_id: str | None = None) -> bool:
        return False


def _launcher(label: str, prompt: str) -> str:
    return "job-fake"


def _engine_context(**kwargs) -> ToolContext:
    """A context carrying every capability the default surface reads, so the
    whole table can build: the wake scheduler, the subagent launcher and the
    job manager."""
    base: dict[str, Any] = dict(
        wake_scheduler=_FakeScheduler(),
        subagent_launcher=_launcher,
        jobs=_FakeJobs(),
    )
    base.update(kwargs)
    return ToolContext(cwd=".", **base)


@pytest.fixture(autouse=True)
def _force_browser_available(monkeypatch):
    """The default-surface assertions include ``browser``, whose builder is
    environment-gated on a reachable CMUX browser. CI has none, so force the
    capability predicate to keep these tests deterministic everywhere."""
    monkeypatch.setattr(builtin, "cmux_browser_available", lambda: True)


def test_default_set_builds_all_builtin_tools() -> None:
    # With every capability attached, the default surface is the whole table.
    context = _engine_context()
    tools = create_tools(context)
    names = [tool.name for tool in tools]
    assert names == DEFAULT_TOOL_NAMES
    assert all(isinstance(tool, AgentTool) for tool in tools)


def test_default_set_drops_wake_without_scheduler() -> None:
    # createIf: no scheduler -> no wake tool (never a tool that can only error).
    tools = create_tools(_engine_context(wake_scheduler=None))
    names = [tool.name for tool in tools]
    assert names == [name for name in DEFAULT_TOOL_NAMES if name != "wake"]


def test_every_tool_has_schema_and_metadata() -> None:
    for tool in create_tools(_engine_context()):
        # JSON Schema derived from the pydantic params model
        assert tool.parameters.get("type") == "object"
        assert "properties" in tool.parameters
        # A zero-arg tool (e.g. list_variables, jobs) legitimately has no params.
        if tool.parameters["properties"] == {}:
            assert tool.name in ("list_variables", "jobs")
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


def test_diff_details_carries_a_rendered_diff_and_honours_the_cap() -> None:
    """write/edit report a rendered unified diff under ``details["diff"]``.

    The TUI's expanded card is powered by this payload (it has no file access
    of its own at render time), so the details must carry the actual safe diff,
    and the list must be bounded on the STORED payload (the transcript
    persists these details, so an unbounded diff would grow the ledger).
    """
    from local_operator.tools import builtin

    details = builtin._diff_details("f.txt", "a\nb\nc\n", "a\nX\nc\nd\n")
    assert details["added"] == 2
    assert details["removed"] == 1
    diff = details["diff"]
    assert isinstance(diff, list) and diff
    assert any(line.startswith("+") for line in diff)
    assert any(line.startswith("-") for line in diff)
    # An unchanged file reports no diff at all (nothing to render).
    unchanged = builtin._diff_details("f.txt", "a\nb\n", "a\nb\n")
    assert "diff" not in unchanged
    # The cap bounds the stored list even for a pathological write.
    big = builtin._diff_details("f.txt", "\n".join(f"line{i}" for i in range(500)), "")
    assert len(big["diff"]) <= builtin._DIFF_DETAILS_CAP_LINES + 1


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
