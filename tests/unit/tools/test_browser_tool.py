"""Tests for the CMUX browser tool (open / goto / screenshot).

The tool shells out to the ``cmux`` CLI; tests monkeypatch ``_run_cmux`` and
``cmux_browser_available`` on the builtin module so no real subprocess or
browser is touched. The detection and command-construction contracts are
what is pinned here.
"""

from __future__ import annotations

import asyncio

import local_operator.tools.builtin as builtin
from local_operator.harness.types import ToolContext


def _ctx() -> ToolContext:
    return ToolContext(cwd="/tmp")


def _run(tool, tool_call_id: str, args: dict, ctx: ToolContext):
    return asyncio.run(tool.execute(tool_call_id, args, context=ctx))


def test_detection_true_with_cmux_env(monkeypatch) -> None:
    monkeypatch.setenv("CMUX_SOCKET", "/tmp/cmux.sock")
    monkeypatch.delenv("CMUX_SURFACE_ID", raising=False)
    assert builtin.cmux_browser_available() is True


def test_detection_true_with_surface_id(monkeypatch) -> None:
    monkeypatch.setenv("CMUX_SURFACE_ID", "s123")
    monkeypatch.delenv("CMUX_SOCKET", raising=False)
    assert builtin.cmux_browser_available() is True


def test_detection_false_without_cmux(monkeypatch) -> None:
    monkeypatch.delenv("CMUX_SOCKET", raising=False)
    monkeypatch.delenv("CMUX_SURFACE_ID", raising=False)
    monkeypatch.setattr("shutil.which", lambda _n: None)
    assert builtin.cmux_browser_available() is False


def test_builder_hides_tool_when_unavailable(monkeypatch) -> None:
    monkeypatch.setattr(builtin, "cmux_browser_available", lambda: False)
    assert builtin.build_browser_tool(_ctx()) is None


def test_builder_offers_tool_when_available(monkeypatch) -> None:
    monkeypatch.setattr(builtin, "cmux_browser_available", lambda: True)
    tool = builtin.build_browser_tool(_ctx())
    assert tool is not None
    assert tool.name == "browser"


def test_open_without_url_errors(monkeypatch) -> None:
    monkeypatch.setattr(builtin, "cmux_browser_available", lambda: True)
    tool = builtin.build_browser_tool(_ctx())
    ctx = _ctx()
    result = _run(tool, "t1", {"action": "open"}, ctx)
    assert result.is_error
    assert "requires a URL" in result.text


def test_open_runs_new_surface_and_records_id(monkeypatch) -> None:
    captured: dict = {}

    async def fake_run(argv, timeout=30.0):
        captured["argv"] = list(argv)
        return 0, '{"surface": "b-42"}'

    monkeypatch.setattr(builtin, "cmux_browser_available", lambda: True)
    monkeypatch.setattr(builtin, "_run_cmux", fake_run)
    tool = builtin.build_browser_tool(_ctx())
    ctx = _ctx()
    result = _run(tool, "t1", {"action": "open", "url": "https://example.com"}, ctx)
    assert not result.is_error
    # Focus must stay with the agent's own pane (--focus false, no focus).
    assert captured["argv"] == [
        "--json",
        "new-surface",
        "--type",
        "browser",
        "--url",
        "https://example.com",
        "--focus",
        "false",
    ]
    assert "b-42" in result.text
    assert ctx.browser.surface_id == "b-42"


def test_goto_reuses_recorded_surface(monkeypatch) -> None:
    captured: dict = {}

    async def fake_run(argv, timeout=30.0):
        captured["surface"] = argv[argv.index("--surface") + 1]
        return 0, "ok"

    monkeypatch.setattr(builtin, "cmux_browser_available", lambda: True)
    monkeypatch.setattr(builtin, "_run_cmux", fake_run)
    tool = builtin.build_browser_tool(_ctx())
    ctx = _ctx()
    ctx.browser = builtin._BrowserSession()
    ctx.browser.surface_id = "b-9"
    result = _run(tool, "t1", {"action": "goto", "url": "https://foo.bar"}, ctx)
    assert not result.is_error
    assert captured["surface"] == "b-9"
    assert "foo.bar" in result.text


def test_goto_without_open_errors(monkeypatch) -> None:
    monkeypatch.setattr(builtin, "cmux_browser_available", lambda: True)
    tool = builtin.build_browser_tool(_ctx())
    ctx = _ctx()  # no browser state recorded
    result = _run(tool, "t1", {"action": "goto", "url": "https://x.y"}, ctx)
    assert result.is_error
    assert "no browser surface open" in result.text


def test_screenshot_writes_default_path(monkeypatch) -> None:
    captured: dict = {}

    async def fake_run(argv, timeout=30.0):
        captured["path"] = argv[-1]
        return 0, "ok"

    monkeypatch.setattr(builtin, "cmux_browser_available", lambda: True)
    monkeypatch.setattr(builtin, "_run_cmux", fake_run)
    tool = builtin.build_browser_tool(_ctx())
    ctx = _ctx()
    ctx.browser = builtin._BrowserSession()
    ctx.browser.surface_id = "b-3"
    result = _run(tool, "t1", {"action": "screenshot"}, ctx)
    assert not result.is_error
    assert captured["path"].endswith(".png")
    assert "Screenshot saved" in result.text


def test_unavailable_reports_clear_error(monkeypatch) -> None:
    monkeypatch.setattr(builtin, "cmux_browser_available", lambda: False)
    tool = builtin.build_browser_tool(_ctx())
    # Force a tool instance so we can exercise the error path directly.
    tool = builtin.AgentTool(
        name="browser",
        label="Browser",
        description="d",
        parameters={},
        approval_tier="read",
        concurrency="shared",
        execute=builtin.execute_browser,
    )
    result = _run(tool, "t1", {"action": "open", "url": "https://x.y"}, _ctx())
    assert result.is_error
    assert "not available" in result.text


def test_unknown_action_errors(monkeypatch) -> None:
    monkeypatch.setattr(builtin, "cmux_browser_available", lambda: True)
    tool = builtin.build_browser_tool(_ctx())
    result = _run(tool, "t1", {"action": "explode"}, _ctx())
    assert result.is_error
