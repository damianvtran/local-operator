"""Tests for the CMUX browser tool (open / goto / screenshot).

The tool shells out to the ``cmux`` CLI; tests monkeypatch ``_run_cmux`` and
``cmux_browser_available`` on the builtin module so no real subprocess or
browser is touched. The detection and command-construction contracts are
what is pinned here.
"""

from __future__ import annotations

import asyncio

import pytest
from pathlib import Path

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
        # The shape real cmux emits — surface_ref, not surface/surface_id/id.
        return 0, '{"pane_ref":"pane:2","surface_ref":"surface:73","type":"browser"}'

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
    assert "surface:73" in result.text
    assert ctx.browser.surface_id == "surface:73"


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
    ctx.browser.surface_id = "surface:9"
    result = _run(tool, "t1", {"action": "goto", "url": "https://foo.bar"}, ctx)
    assert not result.is_error
    assert captured["surface"] == "surface:9"
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
        captured["argv"] = list(argv)
        # cmux writes the file; emulate that so the existence guard passes.
        out_path = argv[argv.index("--out") + 1]
        captured["path"] = out_path
        Path(out_path).write_bytes(b"PNG")
        return 0, "ok"

    monkeypatch.setattr(builtin, "cmux_browser_available", lambda: True)
    monkeypatch.setattr(builtin, "_run_cmux", fake_run)
    tool = builtin.build_browser_tool(_ctx())
    ctx = _ctx()
    ctx.browser = builtin._BrowserSession()
    ctx.browser.surface_id = "surface:3"
    result = _run(tool, "t1", {"action": "screenshot"}, ctx)
    assert not result.is_error
    # The destination MUST be passed as --out; positionally cmux ignores it.
    assert "--out" in captured["argv"]
    assert captured["path"].endswith(".png")
    assert "Screenshot saved" in result.text
    Path(captured["path"]).unlink(missing_ok=True)


def test_screenshot_exit_zero_without_file_is_an_error(monkeypatch, tmp_path) -> None:
    """cmux can exit 0 while writing nothing (e.g. an ignored destination).
    Reporting success then would hand the model a path to a missing file."""

    async def fake_run(argv, timeout=30.0):
        return 0, "OK file:///somewhere/else.png"

    monkeypatch.setattr(builtin, "cmux_browser_available", lambda: True)
    monkeypatch.setattr(builtin, "_run_cmux", fake_run)
    tool = builtin.build_browser_tool(_ctx())
    ctx = _ctx()
    ctx.browser = builtin._BrowserSession()
    ctx.browser.surface_id = "surface:3"
    target = tmp_path / "never-written.png"
    result = _run(tool, "t1", {"action": "screenshot", "path": str(target)}, ctx)
    assert result.is_error
    assert "no file at" in result.text


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


def test_parse_surface_id_reads_real_cmux_payload() -> None:
    """Real cmux emits surface_ref; the other spellings are fallbacks only."""
    real = '{"pane_ref":"pane:2","surface_ref":"surface:73","type":"browser"}'
    assert builtin._parse_surface_id(real) == "surface:73"
    assert builtin._parse_surface_id('{"surface":"surface:1"}') == "surface:1"
    assert builtin._parse_surface_id('{"nothing":"useful"}') == ""


def test_parse_surface_id_rejects_banners_and_errors() -> None:
    """A status banner or error message must never become a --surface value."""
    assert builtin._parse_surface_id("done") == ""
    assert builtin._parse_surface_id("error: could not start") == ""
    assert builtin._parse_surface_id("Created new browser surface.") == ""
    # Ref-shaped trailing token is still accepted from plain text.
    assert builtin._parse_surface_id("Opened surface:42") == "surface:42"


# --- surface-handle parsing: the JSON branch is the PRIMARY path -------------
#
# The previous suite only fed plain text to the guard, so every one of these
# JSON payloads was adopted as a real handle and injected into the next
# `cmux browser --surface <x>` argv. They are the regression tests for that.


@pytest.mark.parametrize(
    "payload",
    [
        # An error payload whose `id` is a request-dedupe token, not a surface.
        '{"ok":false,"error":"browser disabled","id":"req-8f21"}',
        '{"ok":false,"id":"pending"}',
        '{"surface":null,"id":"internal_error"}',
        # Prose smuggled through a surface-shaped key.
        '{"surface_ref":"error: could not start"}',
        # Option-looking value: would become `--surface --help` in the argv.
        '{"id":"--help"}',
        '{"surface_ref":"--focus"}',
        # Sibling refs of the wrong KIND — real cmux output, wrong object.
        '{"pane_ref":"pane:2","window_ref":"window:1"}',
        # Nothing useful at all.
        '{"nothing":"useful"}',
        "",
        "   ",
    ],
)
def test_parse_surface_id_rejects_non_handles(payload: str) -> None:
    assert builtin._parse_surface_id(payload) == ""


@pytest.mark.parametrize(
    "payload",
    [
        # The real shape.
        '{"pane_ref":"pane:2","surface_ref":"surface:73","type":"browser"}',
        # Tolerated spellings.
        '{"surface":"surface:73"}',
        '{"surface_id":"surface:73"}',
        # NDJSON: cmux may send an ack line first.
        '{"type":"ack"}\n{"surface_ref":"surface:73"}',
        # Human preamble before the payload.
        'cmux: connected\n{"surface_ref":"surface:73"}',
        # Trailing line after the payload.
        '{"surface_ref":"surface:73"}\nDone.',
        # Nested under a result envelope.
        '{"ok":true,"result":{"surface_ref":"surface:73"}}',
        # Plain-text fallback.
        "Opened surface:73",
        # The shape REAL cmux emits: pretty-printed, multi-line. A per-line
        # NDJSON scan silently fails on this, which is why it is pinned.
        '{\n  "pane_ref" : "pane:2",\n  "surface_ref" : "surface:73",\n'
        '  "type" : "browser",\n  "window_ref" : "window:1"\n}',
        # Pretty-printed WITH a preamble and a trailing line around it.
        'cmux: connected\n{\n  "surface_ref" : "surface:73"\n}\nDone.',
        # A stray brace in the preamble must not hide the real payload.
        'note: use {braces} carefully\n{"surface_ref":"surface:73"}',
    ],
)
def test_parse_surface_id_accepts_every_real_shape(payload: str) -> None:
    assert builtin._parse_surface_id(payload) == "surface:73"


def test_plain_text_rejects_wrong_kind_refs() -> None:
    """`pane:2` and `window:1` are ref-SHAPED but not surfaces. The old
    shape-only regex adopted both."""
    assert builtin._parse_surface_id("Created pane:2") == ""
    assert builtin._parse_surface_id("window:1") == ""
    assert builtin._parse_surface_id("error:timeout") == ""


def test_open_without_a_handle_is_an_error_not_a_silent_success(monkeypatch) -> None:
    """Reporting success with no handle strands the session: every later call
    can only say "no browser surface open", and the model was told it worked."""

    async def fake_run(argv, timeout=30.0):
        return 0, '{"ok":true,"note":"no ref for you"}'

    monkeypatch.setattr(builtin, "cmux_browser_available", lambda: True)
    monkeypatch.setattr(builtin, "_run_cmux", fake_run)
    tool = builtin.build_browser_tool(_ctx())
    ctx = _ctx()
    result = _run(tool, "t1", {"action": "open", "url": "https://example.com"}, ctx)
    assert result.is_error
    assert "no surface handle" in result.text
    assert ctx.browser is None or not ctx.browser.surface_id


def test_run_cmux_reports_stderr_on_failure() -> None:
    """cmux writes diagnostics to stderr and leaves stdout EMPTY on failure, so
    returning stdout alone produced the blank message "cmux open failed: "."""
    import asyncio

    code, out = asyncio.run(
        builtin._run_cmux(["--json", "definitely-not-a-real-subcommand-xyz"], timeout=20.0)
    )
    assert code != 0
    assert out.strip(), "a failure must never produce an empty diagnostic"
