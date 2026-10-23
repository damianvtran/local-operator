"""Tests for the CMUX browser tool.

The tool shells out to the ``cmux`` CLI; tests monkeypatch ``_run_cmux`` and
``cmux_browser_available`` on the builtin module so no real subprocess and no
real browser is touched. What is pinned here is detection, the command
construction that keeps the user's cmux layout intact, and the guards against
cmux's two silent-success behaviours (a navigation that never lands, and a
``goto`` that turns a non-URL into a Google search).
"""

from __future__ import annotations

import asyncio
import json
import logging
import os
from pathlib import Path
from typing import Any

import pytest

import local_operator.tools.builtin as builtin
from local_operator.harness.types import BrowserSurface, ToolContext

#: Enough of a PNG to satisfy the magic-byte guard without shipping a fixture.
FAKE_PNG = builtin.PNG_MAGIC + b"\x00" * 64


def _ctx() -> ToolContext:
    return ToolContext(cwd="/tmp")


def _run(tool, tool_call_id: str, args: dict[str, Any], ctx: ToolContext):
    return asyncio.run(tool.execute(tool_call_id, args, context=ctx))


def _with_surface(surface_id: str = "surface:3") -> ToolContext:
    ctx = _ctx()
    # Injected the way Session._build_tool_context does it: the holder is owned
    # by the host, not created by the tool, because this context is rebuilt
    # every turn.
    ctx.browser = BrowserSurface()
    ctx.browser.surface_id = surface_id
    return ctx


async def _async(value):
    """A coroutine that resolves to ``value`` — for stubbing an async call."""
    return value


class FakeCmux:
    """Stand-in for the ``cmux`` CLI, routed by argv.

    Modelled on the real command surface as verified against cmux on macOS:
    ``--json new-surface`` answers with ``surface_ref``, ``browser --surface
    <id> eval`` answers with the live document, ``get url`` answers with the
    URL cmux is POINTING AT, and ``screenshot --out`` writes the file.

    ``href`` (live document) and ``pointing`` (what cmux was asked for) are
    separately settable on purpose: their disagreement IS the stale-navigation
    failure the tool exists to catch, and a fake that collapsed them into one
    field could not express the bug.
    """

    def __init__(
        self,
        *,
        href: str = "https://example.com/",
        pointing: str | None = None,
        title: str = "Example Domain",
        ready: str = "complete",
        text: str = "Example Domain\n\nThis domain is for use in examples.",
        dom_text: str = "",
        follow_goto: bool = True,
    ) -> None:
        self.href = href
        self.pointing = href if pointing is None else pointing
        self.title = title
        self.ready = ready
        self.text = text
        # What the layout-independent DOM walk returns when innerText is empty.
        self.dom_text = dom_text
        # Cleared whenever the document is replaced, mirroring a real page
        # losing a window property across a navigation.
        self.marked = False
        # False emulates a navigation cmux accepts and never completes.
        self.follow_goto = follow_goto
        self.value = ""
        self.calls: list[list[str]] = []

    def verbs(self) -> list[str]:
        """The cmux subcommand of each call, for asserting on command choice."""
        return [call[3] if call[:1] == ["browser"] else call[0] for call in self.calls]

    async def __call__(self, argv, timeout: float = 30.0):
        argv = list(argv)
        self.calls.append(argv)
        if argv[:2] == ["--json", "new-surface"]:
            url = argv[argv.index("--url") + 1]
            self.pointing = url
            if self.follow_goto:
                self.href = url
            return 0, '{"pane_ref":"pane:2","surface_ref":"surface:73","type":"browser"}'
        if argv[0] == "close-surface":
            return 0, "OK"
        rest = argv[3:]  # strip ["browser", "--surface", "<id>"]
        verb = rest[0]
        if verb == "eval":
            script = rest[rest.index("--script") + 1]
            if script == builtin._NAV_TOKEN_SET_JS:
                self.marked = True
                return 0, "ok"
            if script == builtin._NAV_TOKEN_GET_JS:
                return 0, "1" if self.marked else "0"
            if script.startswith("(function(sel)"):
                return 0, self.dom_text
            return 0, json.dumps([self.ready, self.href, self.title])
        if verb == "goto":
            self.pointing = rest[1]
            if self.follow_goto:
                self.href = rest[1]
                self.marked = False  # a new document does not carry the marker
            return 0, "OK"
        if verb == "get":
            what = rest[1]
            if what == "url":
                return 0, self.pointing
            if what == "text":
                return 0, self.text
            if what == "value":
                return 0, self.value
            return 1, f"unsupported get: {what}"
        if verb == "click":
            return 0, "OK"
        if verb == "fill":
            self.value = rest[rest.index("--text") + 1]
            return 0, "OK"
        if verb == "snapshot":
            return 0, '- document "Example Domain"\n  - link "Learn more" [ref=e4]'
        if verb == "screenshot":
            Path(rest[rest.index("--out") + 1]).write_bytes(FAKE_PNG)
            return 0, "OK"
        return 1, f"unexpected argv: {argv}"


def _install(monkeypatch, fake: FakeCmux) -> FakeCmux:
    monkeypatch.setattr(builtin, "cmux_browser_available", lambda: True)
    monkeypatch.setattr(builtin, "_run_cmux", fake)
    # Polling delay is pure latency in a test; the settle loop is exercised by
    # its exit conditions, not by wall-clock time.
    monkeypatch.setattr(builtin, "BROWSER_NAV_POLL_S", 0.0)
    monkeypatch.setattr(builtin, "BROWSER_NAV_TIMEOUT_S", 0.05)
    monkeypatch.setattr(builtin, "BROWSER_CLICK_GRACE_S", 0.0)
    return fake


def test_detection_requires_the_binary_not_just_the_environment(monkeypatch) -> None:
    """CMUX_* is inherited by every descendant of a cmux session, including
    ones that crossed into a container or an ssh host with no cmux CLI.
    Detecting on the marker alone advertised a tool whose every action could
    only answer "cmux is not on PATH"."""
    monkeypatch.setenv("CMUX_SURFACE_ID", "s123")
    monkeypatch.setenv("CMUX_SOCKET_PATH", "/tmp/cmux.sock")
    monkeypatch.delenv("CMUX_BUNDLED_CLI_PATH", raising=False)
    monkeypatch.setattr("shutil.which", lambda _n: None)
    assert builtin.cmux_browser_available() is False


def test_detection_true_with_binary_on_path(monkeypatch) -> None:
    monkeypatch.setattr("shutil.which", lambda _n: "/opt/homebrew/bin/cmux")
    assert builtin.cmux_browser_available() is True
    assert builtin._cmux_binary() == "/opt/homebrew/bin/cmux"


def test_detection_falls_back_to_the_bundled_cli(monkeypatch, tmp_path) -> None:
    """cmux's shell integration prepends the app bundle's bin to PATH; a venv
    activation or a login shell that rebuilds PATH drops it while every CMUX_*
    marker survives. CMUX_BUNDLED_CLI_PATH is what recovers that session."""
    bundled = tmp_path / "cmux"
    bundled.write_text("#!/bin/sh\n")
    bundled.chmod(0o755)
    monkeypatch.setattr("shutil.which", lambda _n: None)
    monkeypatch.setenv("CMUX_BUNDLED_CLI_PATH", str(bundled))
    assert builtin._cmux_binary() == str(bundled)


def test_detection_ignores_a_bundled_path_that_is_not_executable(monkeypatch, tmp_path) -> None:
    stale = tmp_path / "cmux"
    stale.write_text("")
    stale.chmod(0o644)
    monkeypatch.setattr("shutil.which", lambda _n: None)
    monkeypatch.setenv("CMUX_BUNDLED_CLI_PATH", str(stale))
    assert builtin.cmux_browser_available() is False


def test_detection_false_without_cmux(monkeypatch) -> None:
    monkeypatch.delenv("CMUX_BUNDLED_CLI_PATH", raising=False)
    monkeypatch.setattr("shutil.which", lambda _n: None)
    assert builtin.cmux_browser_available() is False


def test_detection_never_raises(monkeypatch) -> None:
    """Detection runs while the tool inventory is being built. An exception
    here would take down session start, so it degrades to "no browser"."""

    def boom(_name):
        raise OSError("PATH is unreadable")

    monkeypatch.setattr("shutil.which", boom)
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
    fake = _install(monkeypatch, FakeCmux(href="https://example.com"))
    tool = builtin.build_browser_tool(_ctx())
    ctx = _ctx()
    result = _run(tool, "t1", {"action": "open", "url": "https://example.com"}, ctx)
    assert not result.is_error
    # Focus must stay with the agent's own pane, and the surface must be added
    # as a TAB: `browser open`/`open-split`/`new` split the user's pane.
    assert fake.calls[0] == [
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
    # The host that served the open is named (§16.1): a cmux open used to read
    # identically to one served by the app or the extension.
    assert "on a cmux browser panel" in result.text
    assert builtin._BROWSER_OPEN_CLEANUP_REMINDER in result.text
    assert ctx.browser is not None
    assert ctx.browser.surface_id == "surface:73"


def test_open_reuses_the_surface_it_already_has(monkeypatch) -> None:
    """A fresh surface per navigation leaves a drift of dead browser tabs the
    user closes one at a time, so 'open' becomes 'goto' once one exists."""
    fake = _install(monkeypatch, FakeCmux())
    tool = builtin.build_browser_tool(_ctx())
    ctx = _with_surface("surface:9")
    result = _run(tool, "t1", {"action": "open", "url": "https://foo.bar"}, ctx)
    assert not result.is_error
    assert "new-surface" not in fake.verbs()
    assert "goto" in fake.verbs()
    assert builtin._BROWSER_OPEN_CLEANUP_REMINDER not in result.text
    assert ctx.browser is not None
    assert ctx.browser.surface_id == "surface:9"


def test_goto_reuses_recorded_surface(monkeypatch) -> None:
    fake = _install(monkeypatch, FakeCmux())
    tool = builtin.build_browser_tool(_ctx())
    ctx = _with_surface("surface:9")
    result = _run(tool, "t1", {"action": "goto", "url": "https://foo.bar"}, ctx)
    assert not result.is_error
    assert all(call[2] == "surface:9" for call in fake.calls if call[0] == "browser")
    assert "foo.bar" in result.text


def test_goto_without_open_errors(monkeypatch) -> None:
    monkeypatch.setattr(builtin, "cmux_browser_available", lambda: True)
    tool = builtin.build_browser_tool(_ctx())
    ctx = _ctx()  # no browser state recorded
    result = _run(tool, "t1", {"action": "goto", "url": "https://x.y"}, ctx)
    assert result.is_error
    assert "no browser surface open" in result.text


def test_screenshot_writes_default_path(monkeypatch) -> None:
    fake = _install(monkeypatch, FakeCmux())
    tool = builtin.build_browser_tool(_ctx())
    result = _run(tool, "t1", {"action": "screenshot"}, _with_surface())
    assert not result.is_error
    shot = next(call for call in fake.calls if call[3] == "screenshot")
    # The destination MUST be passed as --out; positionally cmux ignores it,
    # writes into its own temp dir and still exits 0.
    assert "--out" in shot
    written = Path(shot[shot.index("--out") + 1])
    assert written.suffix == ".png"
    assert "Screenshot" in result.text
    assert str(len(FAKE_PNG)) in result.text, "the byte count is the model's only size signal"
    written.unlink(missing_ok=True)


def test_screenshot_exit_zero_without_file_is_an_error(monkeypatch, tmp_path) -> None:
    """cmux can exit 0 while writing nothing (e.g. an ignored destination).
    Reporting success then would hand the model a path to a missing file."""

    async def fake_run(argv, timeout=30.0):
        return 0, "OK file:///somewhere/else.png"

    monkeypatch.setattr(builtin, "cmux_browser_available", lambda: True)
    monkeypatch.setattr(builtin, "_run_cmux", fake_run)
    tool = builtin.build_browser_tool(_ctx())
    ctx = _ctx()
    ctx.browser = BrowserSurface()
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


# --- parser bounds on untrusted subprocess output ---------------------------
#
# cmux output is untrusted input. A pathological payload must degrade to "no
# handle" — an outcome the caller already handles with an honest error — never
# to seconds of CPU or an unexpected-exception tool failure.


def test_deeply_nested_payload_degrades_instead_of_raising() -> None:
    """A recursive walk raised RecursionError at ~2000 levels, and the C JSON
    decoder raises it too (and RecursionError is NOT a ValueError)."""
    for depth in (500, 2000, 20_000):
        blob = '{"a":' * depth + '{"surface_ref":"surface:1"}' + "}" * depth
        result = builtin._parse_surface_id(blob)  # must not raise
        assert result in ("surface:1", "")


def test_pathological_input_is_bounded() -> None:
    """A run of bare "{" made the decode-attempt loop quadratic: 60k of them
    cost over four seconds before the attempt bound."""
    import time

    for blob in ("{" * 60_000, '{"x":1}' * 20_000, "x" * 1_000_000):
        started = time.monotonic()
        builtin._parse_surface_id(blob)
        assert time.monotonic() - started < 1.0


def test_shallowest_handle_wins() -> None:
    """Breadth-first: the top-level handle is what a sane payload means, and a
    nested one should not shadow it."""
    blob = '{"surface_ref":"surface:1","a":{"surface_ref":"surface:99"}}'
    assert builtin._parse_surface_id(blob) == "surface:1"


@pytest.mark.parametrize(
    "payload",
    [
        '{"surface_ref":["surface:73"]}',  # list, not a string
        '{"surface_ref":73}',  # number
        '{"surface_ref":null,"id":"surface:9"}',  # null, and `id` is not trusted
        '{"note":"surface:73"}',  # right shape, wrong key
        '{"surface_ref":"surface:73 extra"}',  # trailing junk in the value
        '{"surface_ref":"surface:"}',  # empty ref
    ],
)
def test_wrong_shaped_values_are_rejected(payload: str) -> None:
    assert builtin._parse_surface_id(payload) == ""


def test_first_success_across_multiple_objects_is_used() -> None:
    """cmux may emit an ack or an error object before the payload."""
    assert (
        builtin._parse_surface_id('{"ok":false,"error":"x"}\n{"surface_ref":"surface:73"}')
        == "surface:73"
    )
    assert builtin._parse_surface_id('{"surface_ref":"surface:73"}\n{"ok":false}') == "surface:73"


def test_payload_larger_than_the_scan_window_still_yields_its_handle() -> None:
    """The 64 KB scan window is a cost bound, not a correctness bound.

    Truncating the head mid-document makes raw_decode fail and the bare-token
    fallback cannot match (the token still carries its JSON quotes), so without
    a quoted-key fallback a legitimate oversized response lost EVERY handle at a
    sharp cliff — total failure rather than degradation. Real cmux sends ~118
    bytes, so this guards a future response that embeds a snapshot or data URI.
    """
    import json

    for filler in (70_000, 5_000_000):
        payload = json.dumps(
            {"ok": True, "surface_ref": "surface:73", "result": {"log": "x" * filler}}
        )
        assert builtin._parse_surface_id(payload) == "surface:73"


def test_quoted_fallback_does_not_invent_a_handle_from_prose() -> None:
    """The fallback is anchored on a ref-SHAPED value under a known key, so
    prose that merely mentions the key name must not produce a handle."""
    assert builtin._parse_surface_id('note: the key is "surface_ref": "not a ref"') == ""
    assert builtin._parse_surface_id('{"ok":false,"error":"denied","id":"req-1"}') == ""
    assert builtin._parse_surface_id('{"pane_ref":"pane:2"}') == ""


def test_goto_refuses_a_flag_shaped_url(monkeypatch) -> None:
    """The URL lands in a POSITIONAL argv slot, so cmux parses a flag-shaped
    value as an option: `goto --help` exits 0 and prints help, which we would
    otherwise report as a successful navigation."""
    calls: list[list[str]] = []

    async def fake_run(argv, timeout=30.0):
        calls.append(list(argv))
        return 0, "ok"

    monkeypatch.setattr(builtin, "cmux_browser_available", lambda: True)
    monkeypatch.setattr(builtin, "_run_cmux", fake_run)
    tool = builtin.build_browser_tool(_ctx())
    ctx = _ctx()
    ctx.browser = BrowserSurface()
    ctx.browser.surface_id = "surface:3"
    for bad in ("--help", "-x", "  --focus"):
        result = _run(tool, "t1", {"action": "goto", "url": bad}, ctx)
        assert result.is_error, f"{bad!r} should be refused"
        assert "flag-shaped" in result.text
    # The refused value must never reach the subprocess. (The stale-handle
    # liveness probe does run first — a `get url` against our OWN surface — so
    # this asserts on the argv content rather than on there being no calls.)
    assert not any("goto" in call for call in calls), calls
    assert not any(bad in call for call in calls for bad in ("--help", "-x", "--focus")), calls


# --- navigation must be PROVEN, not assumed ---------------------------------
#
# `cmux browser get url` answers with the URL cmux was last ASKED for, not the
# URL of the live document, and `goto` exits 0 the instant the request is
# accepted. Measured against real cmux: a 301 the WKWebView never completed
# left `get url` reporting the requested URL for 20+ seconds while the page,
# its title and its screenshot were all still the PREVIOUS document.


def test_goto_that_never_lands_is_an_error(monkeypatch) -> None:
    fake = _install(
        monkeypatch,
        FakeCmux(href="https://rust-lang.org/learn/", title="Learn Rust", follow_goto=False),
    )
    tool = builtin.build_browser_tool(_ctx())
    result = _run(
        tool, "t1", {"action": "goto", "url": "https://iana.org/domains/example"}, _with_surface()
    )
    assert result.is_error, "a navigation that never landed must not report success"
    # Both readings are in the message: the model cannot debug "did not
    # complete" without knowing what it is actually looking at.
    assert "iana.org/domains/example" in result.text
    assert "rust-lang.org/learn" in result.text
    assert fake.href == "https://rust-lang.org/learn/"


def test_goto_settles_through_a_redirect(monkeypatch) -> None:
    """Both readings report POST-redirect state, so www -> apex must settle
    rather than look like a stalled navigation."""
    fake = FakeCmux(follow_goto=False)

    async def redirecting(argv, timeout: float = 30.0):
        code, out = await fake(argv, timeout)
        if list(argv)[3:4] == ["goto"]:
            fake.href = fake.pointing = "https://rust-lang.org/learn/"
            fake.title = "Learn Rust"
        return code, out

    _install(monkeypatch, fake)
    monkeypatch.setattr(builtin, "_run_cmux", redirecting)
    tool = builtin.build_browser_tool(_ctx())
    result = _run(
        tool, "t1", {"action": "goto", "url": "https://www.rust-lang.org/learn"}, _with_surface()
    )
    assert not result.is_error
    # The LANDED url is reported, never the requested one.
    assert "https://rust-lang.org/learn/" in result.text
    assert "Learn Rust" in result.text


def test_open_that_never_loads_keeps_the_handle(monkeypatch) -> None:
    """The surface exists whatever the page did. Dropping the handle would
    leak a tab that nothing — not even 'close' — could reach."""
    _install(
        monkeypatch,
        FakeCmux(href="about:blank", pointing="https://slow.example", follow_goto=False),
    )
    tool = builtin.build_browser_tool(_ctx())
    ctx = _ctx()
    result = _run(tool, "t1", {"action": "open", "url": "https://slow.example"}, ctx)
    assert result.is_error
    assert ctx.browser is not None and ctx.browser.surface_id == "surface:73"


def test_navigation_gives_up_early_when_the_document_is_unreadable(monkeypatch) -> None:
    """A disabled browser panel or a closed surface fails every eval. Waiting
    out the full timeout only delays an error we can already give."""

    calls: list[list[str]] = []

    async def eval_always_fails(argv, timeout: float = 30.0):
        argv = list(argv)
        calls.append(argv)
        if argv[3:4] == ["eval"]:
            return 1, "Error: browser disabled"
        return 0, "https://example.com/"

    monkeypatch.setattr(builtin, "cmux_browser_available", lambda: True)
    monkeypatch.setattr(builtin, "_run_cmux", eval_always_fails)
    monkeypatch.setattr(builtin, "BROWSER_NAV_POLL_S", 0.0)
    monkeypatch.setattr(builtin, "BROWSER_NAV_TIMEOUT_S", 30.0)
    tool = builtin.build_browser_tool(_ctx())
    result = _run(tool, "t1", {"action": "goto", "url": "https://x.y"}, _with_surface())
    assert result.is_error
    assert "browser disabled" in result.text
    assert len([c for c in calls if c[3:4] == ["eval"]]) == 3, "bails after 3 failed probes"


# --- reading ----------------------------------------------------------------


def test_read_returns_page_text_with_its_real_url(monkeypatch) -> None:
    fake = _install(monkeypatch, FakeCmux(text="Hello from the page"))
    tool = builtin.build_browser_tool(_ctx())
    result = _run(tool, "t1", {"action": "read"}, _with_surface())
    assert not result.is_error
    assert "Hello from the page" in result.text
    # Title and URL ride WITH the text so a redirect cannot be filed under the
    # URL the model asked for.
    assert "Example Domain" in result.text
    assert "https://example.com/" in result.text
    read = next(call for call in fake.calls if call[3:5] == ["get", "text"])
    # cmux refuses `get text` with no selector; "body" is what "read the page"
    # means.
    assert read[read.index("--selector") + 1] == "body"


def test_read_falls_back_to_the_dom_when_innertext_is_empty(monkeypatch) -> None:
    """`get text` is innerText, which needs LAYOUT, and a browser surface in a
    background tab may never lay out. Measured on a real results page: both
    `get text --selector body` and `document.body.innerText` returned "" while
    textContent held 15 247 characters. Reporting "(no text)" for a page the
    model can see in its own screenshot is the worst possible answer."""
    _install(monkeypatch, FakeCmux(text="", dom_text="Results for local-operator"))
    tool = builtin.build_browser_tool(_ctx())
    result = _run(tool, "t1", {"action": "read"}, _with_surface())
    assert not result.is_error
    assert "Results for local-operator" in result.text


def test_read_reports_no_text_only_when_the_dom_is_empty_too(monkeypatch) -> None:
    _install(monkeypatch, FakeCmux(text="", dom_text=""))
    tool = builtin.build_browser_tool(_ctx())
    result = _run(tool, "t1", {"action": "read"}, _with_surface())
    assert not result.is_error
    assert "(no text)" in result.text


def test_read_scopes_to_a_selector(monkeypatch) -> None:
    fake = _install(monkeypatch, FakeCmux())
    tool = builtin.build_browser_tool(_ctx())
    result = _run(tool, "t1", {"action": "read", "selector": "main#content"}, _with_surface())
    assert not result.is_error
    read = next(call for call in fake.calls if call[3:5] == ["get", "text"])
    assert read[read.index("--selector") + 1] == "main#content"


def test_read_truncates_a_huge_page(monkeypatch) -> None:
    """Page text is model input: an unbounded body would spend the whole
    context window in one tool call."""
    _install(monkeypatch, FakeCmux(text="x" * (builtin.BROWSER_TEXT_LIMIT_CHARS + 5000)))
    tool = builtin.build_browser_tool(_ctx())
    result = _run(tool, "t1", {"action": "read"}, _with_surface())
    assert not result.is_error
    assert len(result.text) < builtin.BROWSER_TEXT_LIMIT_CHARS + 500
    assert builtin.BASH_TRUNCATION_MARKER.strip() in result.text


def test_snapshot_asks_for_the_compact_tree(monkeypatch) -> None:
    fake = _install(monkeypatch, FakeCmux())
    tool = builtin.build_browser_tool(_ctx())
    result = _run(tool, "t1", {"action": "snapshot"}, _with_surface())
    assert not result.is_error
    assert "[ref=e4]" in result.text, "refs are what make click/type usable"
    snapshot = next(call for call in fake.calls if call[3] == "snapshot")
    assert "--compact" in snapshot


# --- interaction ------------------------------------------------------------


def test_type_uses_fill_so_a_retry_cannot_double_the_input(monkeypatch) -> None:
    """cmux `type` APPENDS keystrokes (verified: typing "XY" into a box holding
    "abc" left "abcXY"); `fill` replaces. A model retrying after a timeout
    would otherwise submit doubled input with no cheap way to notice."""
    fake = _install(monkeypatch, FakeCmux())
    tool = builtin.build_browser_tool(_ctx())
    ctx = _with_surface()
    args = {"action": "type", "selector": "input[name=q]", "text": "hello"}
    assert not _run(tool, "t1", args, ctx).is_error
    result = _run(tool, "t2", args, ctx)
    assert not result.is_error
    assert "fill" in fake.verbs()
    assert "type" not in fake.verbs()
    assert fake.value == "hello", "a second call must not append"
    # The read-back is the only confirmation the model gets that it landed.
    assert "'hello'" in result.text


def test_type_without_a_selector_errors(monkeypatch) -> None:
    _install(monkeypatch, FakeCmux())
    tool = builtin.build_browser_tool(_ctx())
    result = _run(tool, "t1", {"action": "type", "text": "hello"}, _with_surface())
    assert result.is_error
    assert "requires a selector" in result.text


def test_click_that_does_not_navigate_says_so(monkeypatch) -> None:
    """Most clicks open a menu or toggle something. Waiting out a load that
    was never going to happen is pure latency."""
    fake = _install(monkeypatch, FakeCmux())
    tool = builtin.build_browser_tool(_ctx())
    result = _run(tool, "t1", {"action": "click", "selector": "e4"}, _with_surface())
    assert not result.is_error
    assert "no navigation" in result.text
    assert "Example Domain" in result.text
    click = next(call for call in fake.calls if call[3] == "click")
    assert click[click.index("--selector") + 1] == "e4"


def test_click_that_navigates_reports_the_page_it_landed_on(monkeypatch) -> None:
    """The click race, from the other side: cmux is still pointing at the old
    URL for a moment after the click, so a settle sampled immediately agrees
    with itself and reports success on the page we navigated AWAY from."""
    fake = FakeCmux()

    async def click_then_navigate(argv, timeout: float = 30.0):
        code, out = await fake(argv, timeout)
        if list(argv)[3:4] == ["click"]:
            fake.pointing = fake.href = "https://www.iana.org/help/example-domains"
            fake.title = "Example Domains"
        return code, out

    _install(monkeypatch, fake)
    monkeypatch.setattr(builtin, "_run_cmux", click_then_navigate)
    tool = builtin.build_browser_tool(_ctx())
    result = _run(tool, "t1", {"action": "click", "selector": "a"}, _with_surface())
    assert not result.is_error
    assert "no navigation" not in result.text
    assert "Example Domains" in result.text
    assert "iana.org/help/example-domains" in result.text


def test_click_detects_a_post_that_replaces_the_document(monkeypatch) -> None:
    """A form POST to the SAME url changes no URL at all. Measured against
    DuckDuckGo's no-JS search form: the document marker cleared ~0.6 s after
    submit while the URL never moved, and without that signal the result was
    labelled "no navigation" though the whole document had been replaced."""
    fake = FakeCmux(href="https://html.duckduckgo.com/html/", title="DuckDuckGo HTML")

    async def submit_posts(argv, timeout: float = 30.0):
        code, out = await fake(argv, timeout)
        if list(argv)[3:4] == ["click"]:
            # Same URL, new document — exactly what a POST looks like.
            fake.marked = False
            fake.title = "local-operator harness at DuckDuckGo"
        return code, out

    _install(monkeypatch, fake)
    monkeypatch.setattr(builtin, "_run_cmux", submit_posts)
    tool = builtin.build_browser_tool(_ctx())
    result = _run(
        tool, "t1", {"action": "click", "selector": "input[type=submit]"}, _with_surface()
    )
    assert not result.is_error
    assert "no navigation" not in result.text
    assert "local-operator harness at DuckDuckGo" in result.text


def test_click_that_starts_a_navigation_which_stalls_is_an_error(monkeypatch) -> None:
    """Measured: clicking a link left `get url` on the target while the
    document, title and screenshot were all still the previous page."""
    fake = FakeCmux()

    async def click_then_stall(argv, timeout: float = 30.0):
        code, out = await fake(argv, timeout)
        if list(argv)[3:4] == ["click"]:
            fake.pointing = "https://iana.org/domains/example"
        return code, out

    _install(monkeypatch, fake)
    monkeypatch.setattr(builtin, "_run_cmux", click_then_stall)
    tool = builtin.build_browser_tool(_ctx())
    result = _run(tool, "t1", {"action": "click", "selector": "a"}, _with_surface())
    assert result.is_error
    assert "describes the old page" in result.text


@pytest.mark.parametrize("action", ["click", "type"])
def test_flag_shaped_selectors_are_refused(monkeypatch, action: str) -> None:
    """cmux accepts the selector positionally as well as via --selector, and no
    CSS selector or snapshot ref begins with a dash."""
    fake = _install(monkeypatch, FakeCmux())
    tool = builtin.build_browser_tool(_ctx())
    result = _run(
        tool, "t1", {"action": action, "selector": "--help", "text": "x"}, _with_surface()
    )
    assert result.is_error
    assert "flag-shaped" in result.text
    assert not fake.calls, "a refused selector must never reach the subprocess"


# --- goto is an omnibox -----------------------------------------------------


@pytest.mark.parametrize(
    "url",
    [
        "not a url at all",
        "data:text/html,<p>hi</p>",
        "example.com",
        "file:///etc/passwd",
        "javascript:alert(1)",
    ],
)
def test_non_http_urls_are_refused_before_cmux_sees_them(monkeypatch, url: str) -> None:
    """Verified against real cmux: `goto 'not a url at all'` landed on
    https://www.google.com/search?q=not%20a%20url%20at%20all and exited 0 "OK".
    A search-results page would then be read and screenshotted as if it were
    the requested site."""
    fake = _install(monkeypatch, FakeCmux())
    tool = builtin.build_browser_tool(_ctx())
    result = _run(tool, "t1", {"action": "goto", "url": url}, _with_surface())
    assert result.is_error
    assert not fake.calls, "a refused URL must never reach the subprocess"


# --- screenshots ------------------------------------------------------------


def test_screenshot_rejects_a_file_that_is_not_a_png(monkeypatch, tmp_path) -> None:
    """A capture of a surface that never painted lands empty or truncated and
    cmux still exits 0. Catching it here beats failing in an image reader
    several turns away from the cause."""

    target = tmp_path / "shot.png"

    async def writes_garbage(argv, timeout: float = 30.0):
        if list(argv)[3:4] == ["screenshot"]:
            target.write_bytes(b"")
            return 0, "OK"
        return 0, "https://example.com/"

    monkeypatch.setattr(builtin, "cmux_browser_available", lambda: True)
    monkeypatch.setattr(builtin, "_run_cmux", writes_garbage)
    tool = builtin.build_browser_tool(_ctx())
    result = _run(tool, "t1", {"action": "screenshot", "path": str(target)}, _with_surface())
    assert result.is_error
    assert "not a PNG" in result.text


def test_screenshot_resolves_a_relative_path_against_the_session_cwd(monkeypatch, tmp_path) -> None:
    """Relative paths used to resolve against the operator process CWD, and
    `~` was never expanded — it created a literal "~" directory."""
    fake = _install(monkeypatch, FakeCmux())
    tool = builtin.build_browser_tool(_ctx())
    ctx = _with_surface()
    ctx.cwd = str(tmp_path)
    result = _run(tool, "t1", {"action": "screenshot", "path": "shots/page.png"}, ctx)
    shot = next(call for call in fake.calls if call[3] == "screenshot")
    written = Path(shot[shot.index("--out") + 1])
    # cmux itself creates the parent, so the fake failing to write is the
    # expected outcome here; the resolved path is what this pins.
    assert written == tmp_path / "shots" / "page.png"
    assert result.is_error or written.exists()


# --- closing ----------------------------------------------------------------


def test_close_closes_the_surface_and_clears_the_handle(monkeypatch) -> None:
    fake = _install(monkeypatch, FakeCmux())
    tool = builtin.build_browser_tool(_ctx())
    ctx = _with_surface("surface:73")
    result = _run(tool, "t1", {"action": "close"}, ctx)
    assert not result.is_error
    assert fake.calls == [["close-surface", "--surface", "surface:73"]]
    assert ctx.browser is not None
    assert ctx.browser.surface_id == ""


def test_close_with_nothing_open_is_a_no_op(monkeypatch) -> None:
    fake = _install(monkeypatch, FakeCmux())
    tool = builtin.build_browser_tool(_ctx())
    result = _run(tool, "t1", {"action": "close"}, _ctx())
    assert not result.is_error
    assert not fake.calls


def test_close_drops_the_handle_even_when_cmux_fails(monkeypatch) -> None:
    """The tab may already be gone (the user closed it, or cmux restarted).
    Keeping a dead handle strands the session: 'open' reuses it too."""

    async def always_fails(argv, timeout: float = 30.0):
        return 1, "Error: invalid_params: Missing or invalid surface_id"

    monkeypatch.setattr(builtin, "cmux_browser_available", lambda: True)
    monkeypatch.setattr(builtin, "_run_cmux", always_fails)
    tool = builtin.build_browser_tool(_ctx())
    ctx = _with_surface("surface:73")
    result = _run(tool, "t1", {"action": "close"}, ctx)
    assert ctx.browser is not None
    assert ctx.browser.surface_id == ""
    assert "dropped the handle" in result.text


# --- degrading when there is no cmux ----------------------------------------


def test_every_action_degrades_without_cmux(monkeypatch) -> None:
    """Host actions refuse absence; inspection reports it without creating a lane.

    The forced tool must preserve the diagnostic's effect-free missing-record
    result even when no host can run ordinary browser commands.
    """
    monkeypatch.setattr(builtin, "cmux_browser_available", lambda: False)
    # Hermetic: the demotion guard classifies the bridge from the DISCOVERY FILE,
    # so without this the developer's own daemon decides which copy this test
    # sees — and whether it sees the guard at all (a fresh bridge would hand the
    # tool a real backend and there would be no "not available" text at all).
    # Pin "no daemon" explicitly.
    monkeypatch.setattr(builtin, "_bridge_liveness", lambda: (None, None))
    assert builtin.build_browser_tool(_ctx()) is None
    forced = builtin.AgentTool(
        name="browser",
        label="Browser",
        description="d",
        parameters={},
        approval_tier="write",
        concurrency="shared",
        execute=builtin.execute_browser,
    )
    for action in builtin.BROWSER_ACTIONS:
        context = _ctx()
        result = _run(forced, "t1", {"action": action, "url": "https://x.y"}, context)
        if action == "inspect_ownership":
            assert not result.is_error
            assert result.details == {"version": 1, "local_scope_missing": True}
            assert json.loads(result.text) == result.details
            assert context.browser is None, "inspection must not create a browser holder"
        else:
            assert result.is_error, action
            assert "not available" in result.text


def test_unknown_action_lists_the_real_ones(monkeypatch) -> None:
    _install(monkeypatch, FakeCmux())
    tool = builtin.build_browser_tool(_ctx())
    result = _run(tool, "t1", {"action": "teleport"}, _ctx())
    assert result.is_error
    for action in builtin.BROWSER_ACTIONS:
        assert action in result.text


# --- what the model is told the tool IS --------------------------------------
#
# The failure these pin is not mechanical. In a real session the tool was
# advertised and reachable, and the agent still answered a request for
# before/after screenshots by writing a playwright script and running
# `playwright install chromium` (23 s of download), then — told outright to use
# the cmux browser instead — shelled the cmux CLI through `bash` rather than
# calling this tool. A description that lists verbs gives the model nothing to
# choose ON. The deciding property is that this is the user's own browser, so
# its logins persist and they can sign in by hand, which no freshly downloaded
# headless Chromium can offer.


def test_description_states_the_persistence_that_makes_it_worth_choosing(monkeypatch) -> None:
    monkeypatch.setattr(builtin, "cmux_browser_available", lambda: True)
    tool = builtin.build_browser_tool(_ctx())
    assert tool is not None
    text = tool.description.lower()
    # It is the user's real browser, not a headless throwaway...
    assert "real" in text
    # ...which is worth choosing because the session survives.
    assert "persist" in text
    assert "cookies" in text and "logins" in text
    # ...and the user can be asked to authenticate by hand.
    assert "sign in" in text
    # ...so there is never a reason to build a second browser stack.
    assert "never install" in text
    # Lifecycle is near open/close semantics in the high-salience schema, not
    # left solely to a guide the model may never need to read.
    assert "fresh 'open' creates one" in text
    assert "before your final response" in text
    assert "ends only your own tab" in text


def test_unavailable_error_refuses_the_substitution_too(monkeypatch) -> None:
    """A host that forces the tool on reads this message instead of the
    description, so the same rule has to be in both places."""
    monkeypatch.setattr(builtin, "cmux_browser_available", lambda: False)
    forced = builtin.AgentTool(
        name="browser",
        label="Browser",
        description="d",
        parameters={},
        approval_tier="write",
        concurrency="shared",
        execute=builtin.execute_browser,
    )
    result = _run(forced, "t1", {"action": "screenshot", "path": "/tmp/x.png"}, _ctx())
    assert result.is_error
    assert "Do not install or script one instead" in result.text


def _clear_cmux_env(monkeypatch) -> None:
    """Drop every CMUX_* marker: this suite itself may run inside cmux."""
    for name in [key for key in os.environ if key.startswith("CMUX_")]:
        monkeypatch.delenv(name, raising=False)


def test_detection_logs_when_a_cmux_looking_host_resolves_no_cli(monkeypatch, caplog) -> None:
    """Absence is silent to the MODEL on purpose, which also made it silent to
    whoever has to explain it. A session carrying cmux's markers yet resolving
    no CLI is the one anomalous shape, and it should be diagnosable from a log
    line rather than from the agent's behaviour afterwards."""
    _clear_cmux_env(monkeypatch)
    monkeypatch.setenv("CMUX_SURFACE_ID", "s123")
    monkeypatch.setenv("CMUX_SOCKET_PATH", "/tmp/cmux.sock")
    monkeypatch.setattr("shutil.which", lambda _n: None)
    with caplog.at_level(logging.WARNING, logger="local_operator.tools.builtin"):
        assert builtin.cmux_browser_available() is False
    messages = [record.getMessage() for record in caplog.records]
    assert any("CMUX_SURFACE_ID" in message for message in messages), messages
    assert any("CMUX_BUNDLED_CLI_PATH" in message for message in messages), messages


def test_detection_stays_quiet_on_a_host_that_is_not_cmux(monkeypatch, caplog) -> None:
    """Detection runs on every session start. A warning on every non-cmux host
    would be noise, and noise is how a real warning gets ignored."""
    _clear_cmux_env(monkeypatch)
    monkeypatch.setattr("shutil.which", lambda _n: None)
    with caplog.at_level(logging.WARNING, logger="local_operator.tools.builtin"):
        assert builtin.cmux_browser_available() is False
    assert caplog.records == []


# --- the surface must outlive the per-turn ToolContext ------------------------
#
# Session._run_turn calls `self._context.tool_context = self._build_tool_context()`
# at the START OF EVERY TURN. A handle the tool stored on the context it was
# handed therefore lived exactly one turn: the next turn saw browser=None, so
# "open X" then "click Y" in the following message answered "no browser surface
# open", and every turn that opened a browser stranded a cmux tab nothing could
# close. The whole suite passed over it because no test built a SECOND context.


def _session(tmp_path):
    """A Session, for the sake of its real ``_build_tool_context``."""
    from local_operator.harness.types import ModelSpec
    from local_operator.session.session import Session
    from local_operator.session.transcript import Transcript

    def stream(_request, _signal):
        async def gen():
            if False:  # pragma: no cover - never streamed; only the context is used
                yield None

        return gen()

    return Session(
        model=ModelSpec(provider="test", model_id="m", context_window=1000),
        stream_fn=stream,
        tools=[],
        transcript=Transcript(tmp_path / "sess"),
        system_blocks_provider=lambda: [],
    )


def test_surface_survives_a_rebuilt_tool_context(monkeypatch, tmp_path) -> None:
    """Turn 1 opens, turn 2 reads and closes THE SAME surface.

    Each turn gets a context built exactly the way ``_run_turn`` builds it.
    Before the surface was owned by the session, turn 2's context carried
    ``browser=None``: 'read' answered "no browser surface open", 'close' answered
    "No browser surface open.", and 'open' created a SECOND surface — two
    ``new-surface`` calls and zero ``close-surface`` calls for one session.
    """
    fake = _install(monkeypatch, FakeCmux())
    session = _session(tmp_path)
    tool = builtin.build_browser_tool(session._build_tool_context())

    turn_one = session._build_tool_context()
    opened = _run(tool, "t1", {"action": "open", "url": "https://example.com"}, turn_one)
    assert not opened.is_error
    assert turn_one.browser is not None and turn_one.browser.surface_id == "surface:73"

    turn_two = session._build_tool_context()
    assert turn_two is not turn_one, "the fixture must model a REBUILT context"
    assert turn_two.browser is not None, "the handle did not survive the turn boundary"
    assert turn_two.browser.surface_id == "surface:73"

    assert not _run(tool, "t2", {"action": "read"}, turn_two).is_error
    closed = _run(tool, "t3", {"action": "close"}, turn_two)
    assert "Closed browser surface surface:73" in closed.text

    opens = [call for call in fake.calls if call[:2] == ["--json", "new-surface"]]
    assert len(opens) == 1, f"a second surface would be a leaked tab: {opens}"
    assert fake.calls.count(["close-surface", "--surface", "surface:73"]) == 1


def test_dispose_closes_a_surface_the_model_left_open(monkeypatch, tmp_path) -> None:
    """Nothing else can close it: the handle dies with the process, and the tab
    stays in the user's pane for them to close by hand."""
    fake = _install(monkeypatch, FakeCmux())
    session = _session(tmp_path)
    tool = builtin.build_browser_tool(session._build_tool_context())
    args = {"action": "open", "url": "https://example.com"}
    assert not _run(tool, "t1", args, session._build_tool_context()).is_error

    asyncio.run(session.dispose())
    assert fake.calls[-1] == ["close-surface", "--surface", "surface:73"]
    assert session._browser.surface_id == ""


def test_dispose_closes_a_bridge_surface_the_model_left_open(monkeypatch, tmp_path) -> None:
    """The fallback covers extension tabs too, not only cmux surfaces."""
    calls: list[tuple[str, dict[str, Any], str]] = []

    async def fake_call(tool_call_id, action, params, *, surface="", client=None):
        calls.append((action, params, surface))
        return {}, None

    monkeypatch.setattr(builtin, "_bridge_call", fake_call)
    session = _session(tmp_path)
    session._browser.surface_id = "bridge:73:ownednonce"

    asyncio.run(session.dispose())

    assert calls == [("close", {"tab": "bridge:73:ownednonce"}, "bridge:73:ownednonce")]
    assert session._browser.surface_id == ""


def test_dispose_without_a_surface_runs_no_cmux(monkeypatch, tmp_path) -> None:
    fake = _install(monkeypatch, FakeCmux())
    session = _session(tmp_path)
    asyncio.run(session.dispose())
    assert not fake.calls


# --- a stale handle must never drive the user's own tab -----------------------


class StaleSurfaceCmux(FakeCmux):
    """cmux resolving a dead ``--surface`` handle the way it really does.

    Measured on this host against ``--surface surface:999999`` (a handle that
    never existed): ``get url`` returned rc=1 ``Error: invalid_params: Missing
    or invalid surface_id``, while ``get title``, ``get text --selector body``,
    ``eval`` and ``snapshot --compact`` ALL returned rc=0 carrying an unrelated
    tab's content. That fallback is what let a stale handle silently drive and
    report on whatever tab the USER was looking at.
    """

    def __init__(self, live: str = "surface:73") -> None:
        super().__init__(href="https://someone-elses-tab.example/", title="Not your page")
        self.live = live

    async def __call__(self, argv, timeout: float = 30.0):
        argv = list(argv)
        if argv[:2] == ["browser", "--surface"] and argv[2] != self.live:
            if argv[3:5] == ["get", "url"]:
                self.calls.append(argv)
                return 1, "Error: invalid_params: Missing or invalid surface_id"
            # Everything else falls back to the active surface and exits 0.
        return await super().__call__(argv, timeout)


@pytest.mark.parametrize(
    "args",
    [
        {"action": "read"},
        {"action": "snapshot"},
        {"action": "screenshot"},
        {"action": "click", "selector": "a"},
        {"action": "type", "selector": "input", "text": "hi"},
        {"action": "goto", "url": "https://example.com"},
    ],
    ids=lambda a: a["action"],
)
def test_a_dead_handle_is_refused_instead_of_driving_the_active_tab(monkeypatch, args) -> None:
    fake = _install(monkeypatch, StaleSurfaceCmux())
    tool = builtin.build_browser_tool(_ctx())
    ctx = _with_surface("surface:99999")
    result = _run(tool, "t1", dict(args), ctx)
    assert result.is_error, result.text
    assert "surface:99999 is gone" in result.text
    assert "'open'" in result.text, "the model needs to be told how to recover"
    # Cleared, or 'open' would reuse the dead handle and there would be no way back.
    assert ctx.browser is not None
    assert ctx.browser.surface_id == ""
    # Nothing but the liveness probe itself may reach cmux: the action's own
    # verb would have been answered by the user's tab with exit 0.
    assert fake.calls == [["browser", "--surface", "surface:99999", "get", "url"]]


def test_open_recovers_from_a_dead_handle_instead_of_erroring(monkeypatch) -> None:
    """'open' is the recovery verb, so it drops the dead handle and makes a real
    surface rather than telling the model to call the verb it just called."""
    fake = _install(monkeypatch, StaleSurfaceCmux())
    tool = builtin.build_browser_tool(_ctx())
    ctx = _with_surface("surface:99999")
    result = _run(tool, "t1", {"action": "open", "url": "https://example.com"}, ctx)
    assert not result.is_error, result.text
    assert ctx.browser is not None
    assert ctx.browser.surface_id == "surface:73"
    assert any(call[:2] == ["--json", "new-surface"] for call in fake.calls), fake.calls
    assert "goto" not in fake.verbs(), "a goto here would have driven the user's tab"


# --- 'type' must verify the fill, not echo the read-back ----------------------


def test_type_reports_a_fill_that_did_not_take_as_an_error(monkeypatch) -> None:
    """The read-back used to be interpolated into "Value is now 'X'." without
    ever being compared to what was asked for, so a fill that did nothing was
    reported as a success quoting the field's OLD contents as the new ones."""
    fake = _install(monkeypatch, FakeCmux())
    fake.value = "stale contents"

    async def fill_does_nothing(argv, timeout: float = 30.0):
        if argv[3:4] == ["fill"]:
            fake.calls.append(list(argv))
            return 0, "OK"  # exit 0, field untouched — cmux's actual shape here
        return await FakeCmux.__call__(fake, argv, timeout)

    monkeypatch.setattr(builtin, "_run_cmux", fill_does_nothing)
    tool = builtin.build_browser_tool(_ctx())
    args = {"action": "type", "selector": "input[name=q]", "text": "hello"}
    result = _run(tool, "t1", args, _with_surface())
    assert result.is_error
    assert "did not take" in result.text
    assert "'stale contents'" in result.text and "'hello'" in result.text


def test_type_says_so_when_the_value_cannot_be_read_back(monkeypatch) -> None:
    """A contenteditable has no `value` property, so an unreadable read-back is
    not evidence either way — but it must not read as a verified fill."""
    fake = _install(monkeypatch, FakeCmux())

    async def no_value_property(argv, timeout: float = 30.0):
        if argv[3:5] == ["get", "value"]:
            return 1, "Error: not_supported"
        return await FakeCmux.__call__(fake, argv, timeout)

    monkeypatch.setattr(builtin, "_run_cmux", no_value_property)
    tool = builtin.build_browser_tool(_ctx())
    args = {"action": "type", "selector": "div[contenteditable]", "text": "hello"}
    result = _run(tool, "t1", args, _with_surface())
    assert not result.is_error
    assert "unverified" in result.text


def test_type_refuses_flag_shaped_text(monkeypatch) -> None:
    """cmux's parser is flag-greedy in the --text slot: measured, `fill
    --selector a --text --help` exits 0 and prints the browser help, and the
    tool then reported "Typed into a." having typed nothing."""
    fake = _install(monkeypatch, FakeCmux())
    tool = builtin.build_browser_tool(_ctx())
    for bad in ("--help", "--focus true", "-x"):
        result = _run(
            tool, "t1", {"action": "type", "selector": "input", "text": bad}, _with_surface()
        )
        assert result.is_error, f"{bad!r} should be refused"
        assert "flag-shaped text" in result.text
    assert not fake.calls, "a refused value must never reach the subprocess"


# --- typing a referenced secret: the value must not reach the model ----------
#
# The incident shape this lane exists for (KOHO analytics QA, 2026-10-16): a
# `type` call carried a literal password, so the plaintext landed in a
# subagent's transcript, where no shape rule can catch it. The cells below are
# the console tool's secret suite applied to this lane (see
# tests/unit/tools/test_console_tool.py) — same ref convention, same
# session-first precedence, same fail-closed rule — plus the two carriers this
# lane adds: the cmux argv and the bridge wire.

#: Synthetic, and deliberately not credential-SHAPED: a plain value is exactly
#: what the shape pass cannot see, which is why the registration is the defense.
SECRET = "synthetic-test-password"
FRESH = "synthetic-fresh-handover"
STALE = "synthetic-stale-store-value"


def _store() -> Any:
    from local_operator.variables import VariableStore

    return VariableStore(cwd=".")


def _stored_secret(monkeypatch) -> None:
    """The real resolution path, minus the encrypted store on disk."""
    from local_operator.secrets import access

    monkeypatch.setattr(access, "retrieve_secret", lambda name, base=None: SECRET.encode())


def _secret_ctx(store: Any = None, surface_id: str = "surface:3") -> ToolContext:
    ctx = ToolContext(cwd="/tmp", variables=store)
    ctx.browser = BrowserSurface()
    ctx.browser.surface_id = surface_id
    return ctx


def _bridge_only(monkeypatch) -> None:
    """Pin the host probes so a `bridge:` handle takes the bridge lane.

    Liveness readings are pinned to "no record", like test_tool_selection.py's
    `_availability`: an answer that changes with the developer's own daemon is
    not a test.
    """

    async def bridge_reachable(classified=None):
        return True

    async def ui_reachable(classified=None):
        return False

    monkeypatch.setattr(builtin, "cmux_browser_available", lambda: False)
    monkeypatch.setattr(builtin, "bridge_browser_reachable", bridge_reachable)
    monkeypatch.setattr(builtin, "ui_browser_reachable", ui_reachable)
    monkeypatch.setattr(builtin, "_bridge_liveness", lambda: (None, None))
    monkeypatch.setattr(builtin, "_ui_liveness", lambda: (None, None))


def test_a_secret_ref_reaches_the_cmux_field_without_reaching_the_model(monkeypatch) -> None:
    """The negative test, asserted rather than argued.

    Three surfaces of the model's view must be clean — the tool RESULT, the
    arguments it emitted, and the details the call returns — while the fill
    command itself must carry the value (that is how it gets typed), and the
    session's redaction seam must contain it afterwards.
    """
    store = _store()
    store.store_credential("SUDO_PASSWORD", SECRET, "ask")
    from local_operator.secrets import access

    consulted: list[str] = []

    def boom(name: str, base: Path | None = None) -> bytes:
        consulted.append(name)
        raise KeyError(name)

    monkeypatch.setattr(access, "retrieve_secret", boom)
    fake = _install(monkeypatch, FakeCmux())
    tool = builtin.build_browser_tool(_ctx())
    args = {"action": "type", "selector": "input[name=q]", "secret_ref": "SUDO_PASSWORD"}
    result = _run(tool, "t1", args, _secret_ctx(store))

    assert not result.is_error, result.text
    # The fill (never a keystroke `type`, which APPENDS on a retry) carries the
    # value: the one-call disclosure, asserted in BOTH directions so the claim
    # cannot be quietly upgraded to "the value never leaves the runtime".
    fills = [call for call in fake.calls if call[3:4] == ["fill"]]
    assert fills and fills[-1][fills[-1].index("--text") + 1] == SECRET
    assert "type" not in fake.verbs(), "a retry would double the input"
    assert fake.value == SECRET
    # The model's views name the ref and never the value.
    assert SECRET not in result.text
    assert "SUDO_PASSWORD" in result.text
    assert SECRET not in repr(args)
    assert SECRET not in repr(result.details)
    # The session's own redaction seam contains it afterwards.
    assert SECRET in store.redaction_values()
    assert store.redact(f"env SUDO_PASSWORD={SECRET}") == "env SUDO_PASSWORD=[redacted]"
    # The session store answered, so the encrypted store was never consulted.
    assert consulted == []


def test_a_ref_fill_that_did_not_take_names_the_ref_not_the_value(monkeypatch) -> None:
    """The failure sentence keeps the field-contents diagnostic (`got`) and the
    EXPECTED side names the ref: an error text is a model-visible surface too."""
    store = _store()
    store.store_credential("SUDO_PASSWORD", SECRET, "ask")
    from local_operator.secrets import access

    monkeypatch.setattr(access, "retrieve_secret", lambda name, base=None: b"nope")
    fake = _install(monkeypatch, FakeCmux())
    fake.value = "stale contents"

    async def fill_does_nothing(argv, timeout: float = 30.0):
        if argv[3:4] == ["fill"]:
            fake.calls.append(list(argv))
            return 0, "OK"  # exit 0, field untouched — cmux's actual shape here
        return await FakeCmux.__call__(fake, argv, timeout)

    monkeypatch.setattr(builtin, "_run_cmux", fill_does_nothing)
    tool = builtin.build_browser_tool(_ctx())
    result = _run(
        tool,
        "t1",
        {"action": "type", "selector": "input[name=q]", "secret_ref": "SUDO_PASSWORD"},
        _secret_ctx(store),
    )
    assert result.is_error
    assert "did not take" in result.text
    assert "'stale contents'" in result.text
    assert "value stored as 'SUDO_PASSWORD'" in result.text
    assert SECRET not in result.text


def test_a_secret_ref_reaches_the_bridge_field_without_reaching_the_model(monkeypatch) -> None:
    """The same contract on the other lane: the wire carries the value (the
    host has to type it), the result does not."""
    store = _store()
    store.store_credential("SUDO_PASSWORD", SECRET, "ask")
    captured: dict[str, Any] = {}

    async def fake_call(tool_call_id, action, params, *, surface="", client=None):
        captured["action"] = action
        captured["params"] = params
        return {"value": SECRET}, None

    _bridge_only(monkeypatch)
    monkeypatch.setattr(builtin, "bridge_browser_advertisable", lambda: True)
    monkeypatch.setattr(builtin, "_bridge_call", fake_call)
    tool = builtin.build_browser_tool(_ctx())
    assert tool is not None
    args = {"action": "type", "selector": "input[name=pw]", "secret_ref": "SUDO_PASSWORD"}
    result = _run(tool, "t1", args, _secret_ctx(store, surface_id="bridge:9:nonce"))

    assert not result.is_error, result.text
    assert captured["action"] == "type"
    assert captured["params"]["text"] == SECRET
    assert SECRET not in result.text
    assert "SUDO_PASSWORD" in result.text
    assert SECRET not in repr(result.details)
    assert SECRET in store.redaction_values()


def test_a_bridge_ref_fill_that_did_not_take_names_the_ref_not_the_value(monkeypatch) -> None:
    """The wire lane's failure copy, pinned for the same reason as the cmux one:
    the read-back mismatch must name the ref, never the expected value."""
    store = _store()
    store.store_credential("SUDO_PASSWORD", SECRET, "ask")
    from local_operator.secrets import access

    monkeypatch.setattr(access, "retrieve_secret", lambda name, base=None: b"nope")

    async def fake_call(tool_call_id, action, params, *, surface="", client=None):
        return {"value": "stale contents"}, None

    _bridge_only(monkeypatch)
    monkeypatch.setattr(builtin, "bridge_browser_advertisable", lambda: True)
    monkeypatch.setattr(builtin, "_bridge_call", fake_call)
    tool = builtin.build_browser_tool(_ctx())
    result = _run(
        tool,
        "t1",
        {"action": "type", "selector": "input[name=pw]", "secret_ref": "SUDO_PASSWORD"},
        _secret_ctx(store, surface_id="bridge:9:nonce"),
    )
    assert result.is_error
    assert "did not take" in result.text
    assert "'stale contents'" in result.text
    assert "value stored as 'SUDO_PASSWORD'" in result.text
    assert SECRET not in result.text


def test_no_redaction_sink_fails_closed(monkeypatch) -> None:
    """A store that cannot contain the value must not type it.

    Without the registration, a page that echoes the field back would put the
    plaintext into a tool result with nothing to mask it.
    """
    _stored_secret(monkeypatch)
    fake = _install(monkeypatch, FakeCmux())
    tool = builtin.build_browser_tool(_ctx())
    args = {"action": "type", "selector": "input[name=q]", "secret_ref": "SUDO_PASSWORD"}
    result = _run(tool, "t1", args, _secret_ctx(None))
    assert result.is_error
    assert "no redaction sink" in result.text
    assert "`secret` tool" in result.text
    assert "`text`" in result.text
    assert fake.calls == [], "a refused value must never reach the subprocess"


def test_an_unresolvable_secret_ref_is_refused_without_dialling(monkeypatch) -> None:
    from local_operator.secrets import access

    def boom(name: str, base: Path | None = None) -> bytes:
        raise KeyError(name)

    monkeypatch.setattr(access, "retrieve_secret", boom)
    fake = _install(monkeypatch, FakeCmux())
    tool = builtin.build_browser_tool(_ctx())
    result = _run(
        tool,
        "t1",
        {"action": "type", "selector": "input[name=q]", "secret_ref": "NOPE"},
        _secret_ctx(_store()),
    )
    assert result.is_error
    assert "could not resolve secret_ref 'NOPE'" in result.text
    # The remedy names the routes that DO carry the value, and none of them is
    # "paste the value".
    assert "`ask` secret question" in result.text
    assert "`secret` tool" in result.text
    assert "sign in" in result.text
    assert "do not put the value itself in `text`" in result.text
    assert fake.calls == []


def test_the_refusal_closes_the_stores_own_sentence_with_one_period(monkeypatch) -> None:
    """`SecretNotFound`'s text already ends in a period and the refusal template
    appends its own, which printed `...in this store..` (console QA round 1, Q2;
    the same fix applies here)."""
    from local_operator.secrets import access
    from local_operator.secrets.errors import SecretNotFound

    def boom(name: str, base: Path | None = None) -> bytes:
        raise SecretNotFound(f"No secret named {name!r} in this store.")

    monkeypatch.setattr(access, "retrieve_secret", boom)
    fake = _install(monkeypatch, FakeCmux())
    tool = builtin.build_browser_tool(_ctx())
    result = _run(
        tool,
        "t1",
        {"action": "type", "selector": "input[name=q]", "secret_ref": "NOPE"},
        _secret_ctx(_store()),
    )
    assert result.is_error
    assert "in this store. Get the value" in result.text
    assert ".." not in result.text
    assert fake.calls == []


def test_a_session_credential_beats_a_same_named_stored_secret(monkeypatch) -> None:
    """Precedence, pinned: the fresh handover is what the user acted on."""
    store = _store()
    store.store_credential("SUDO_PASSWORD", FRESH, "ask")
    from local_operator.secrets import access

    monkeypatch.setattr(access, "retrieve_secret", lambda name, base=None: STALE.encode())
    fake = _install(monkeypatch, FakeCmux())
    tool = builtin.build_browser_tool(_ctx())
    result = _run(
        tool,
        "t1",
        {"action": "type", "selector": "input[name=q]", "secret_ref": "SUDO_PASSWORD"},
        _secret_ctx(store),
    )
    assert not result.is_error, result.text
    assert fake.value == FRESH
    # The fresh value is what got registered; the stored one was never used.
    assert FRESH in store.redaction_values()
    assert STALE not in store.redaction_values()
    assert FRESH not in result.text and STALE not in result.text


def test_a_loosely_spelled_session_key_still_resolves(monkeypatch) -> None:
    """`secret_ref` is model-authored text, so the read applies the same
    normalize-on-write the store applied: `sudo-password` finds `SUDO_PASSWORD`."""
    store = _store()
    store.store_credential("sudo password", SECRET, "ask")
    from local_operator.secrets import access

    monkeypatch.setattr(access, "retrieve_secret", lambda name, base=None: b"nope")
    fake = _install(monkeypatch, FakeCmux())
    tool = builtin.build_browser_tool(_ctx())
    result = _run(
        tool,
        "t1",
        {"action": "type", "selector": "input[name=q]", "secret_ref": "sudo-password"},
        _secret_ctx(store),
    )
    assert not result.is_error, result.text
    assert fake.value == SECRET


def test_a_raising_session_store_falls_through_to_the_encrypted_store(monkeypatch) -> None:
    """A duck-typed store's accessor can raise; the resolver then degrades to
    "no session value" and lets the encrypted store answer — never an unhandled
    exception out of `_execute_browser`."""
    from local_operator.variables import VariableStore

    class Hostile(VariableStore):
        def credential_env(self) -> dict[str, str]:
            raise RuntimeError("store exploded")

    _stored_secret(monkeypatch)
    fake = _install(monkeypatch, FakeCmux())
    tool = builtin.build_browser_tool(_ctx())
    result = _run(
        tool,
        "t1",
        {"action": "type", "selector": "input[name=q]", "secret_ref": "SUDO_PASSWORD"},
        _secret_ctx(Hostile(cwd=".")),
    )
    assert not result.is_error, result.text
    assert fake.value == SECRET


def test_a_non_dict_session_store_read_falls_through_too(monkeypatch) -> None:
    """The shape guard, separately: an accessor that returns something that is
    not a mapping is treated as "no session value" rather than subscripted."""
    from local_operator.variables import VariableStore

    class Shapeless(VariableStore):
        def credential_env(self) -> Any:
            return ["not", "a", "mapping"]

    _stored_secret(monkeypatch)
    fake = _install(monkeypatch, FakeCmux())
    tool = builtin.build_browser_tool(_ctx())
    result = _run(
        tool,
        "t1",
        {"action": "type", "selector": "input[name=q]", "secret_ref": "SUDO_PASSWORD"},
        _secret_ctx(Shapeless(cwd=".")),
    )
    assert not result.is_error, result.text
    assert fake.value == SECRET


def test_type_refuses_text_and_secret_ref_together(monkeypatch) -> None:
    fake = _install(monkeypatch, FakeCmux())
    tool = builtin.build_browser_tool(_ctx())
    result = _run(
        tool,
        "t1",
        {"action": "type", "selector": "input", "text": "x", "secret_ref": "SUDO_PASSWORD"},
        _with_surface(),
    )
    assert result.is_error
    assert "'text' or 'secret_ref', not both" in result.text
    assert not fake.calls


def test_secret_ref_is_refused_on_actions_that_never_enter_a_field(monkeypatch) -> None:
    """Refused rather than silently ignored: a caller whose ref was dropped
    would believe the secret had been used."""
    fake = _install(monkeypatch, FakeCmux())
    tool = builtin.build_browser_tool(_ctx())
    for action, extra in (("read", {"selector": "body"}), ("click", {"selector": "a"})):
        result = _run(
            tool,
            "t1",
            {"action": action, "secret_ref": "SUDO_PASSWORD", **extra},
            _with_surface(),
        )
        assert result.is_error, action
        assert "only valid for 'type'" in result.text
    assert not fake.calls


def test_a_dash_shaped_secret_value_is_not_refused(monkeypatch) -> None:
    """The `_validate_typed_text` dash guard is skipped for a resolved secret,
    deliberately: the value is the session's, not model-authored text, and a
    password may legitimately start with a dash. The discriminating control is
    the same string as literal `text`, which IS refused — so this cell fails if
    the skip is ever removed."""
    store = _store()
    store.store_credential("SUDO_PASSWORD", "--dash-shaped-secret", "ask")
    from local_operator.secrets import access

    monkeypatch.setattr(access, "retrieve_secret", lambda name, base=None: b"nope")
    fake = _install(monkeypatch, FakeCmux())
    tool = builtin.build_browser_tool(_ctx())
    by_ref = _run(
        tool,
        "t1",
        {"action": "type", "selector": "input[name=q]", "secret_ref": "SUDO_PASSWORD"},
        _secret_ctx(store),
    )
    assert not by_ref.is_error, by_ref.text
    assert fake.value == "--dash-shaped-secret"
    literal = _run(
        tool,
        "t2",
        {"action": "type", "selector": "input[name=q]", "text": "--dash-shaped-secret"},
        _with_surface(),
    )
    assert literal.is_error and "flag-shaped text" in literal.text


def test_a_literal_type_registers_nothing(monkeypatch) -> None:
    """Only a RESOLVED value is registered: `text` is the model's own string,
    already in its context by definition, and registering it would scrub an
    ordinary word out of every later result."""
    store = _store()
    _install(monkeypatch, FakeCmux())
    tool = builtin.build_browser_tool(_ctx())
    result = _run(
        tool, "t1", {"action": "type", "selector": "input", "text": "hello"}, _secret_ctx(store)
    )
    assert not result.is_error, result.text
    assert store.redaction_values() == []


def test_no_log_surface_carries_the_value(monkeypatch, caplog) -> None:
    """A DEBUG level over the whole ref call changes nothing: no logging site on
    the type path touches `params.text` after the resolver fills it."""
    store = _store()
    store.store_credential("SUDO_PASSWORD", SECRET, "ask")
    from local_operator.secrets import access

    monkeypatch.setattr(access, "retrieve_secret", lambda name, base=None: b"nope")
    _install(monkeypatch, FakeCmux())
    tool = builtin.build_browser_tool(_ctx())
    with caplog.at_level(logging.DEBUG):
        result = _run(
            tool,
            "t1",
            {"action": "type", "selector": "input[name=q]", "secret_ref": "SUDO_PASSWORD"},
            _secret_ctx(store),
        )
    assert not result.is_error, result.text
    assert SECRET not in caplog.text
    assert SECRET not in result.text


def test_the_value_is_scrubbed_from_journaled_arguments_too(monkeypatch) -> None:
    """The second line of defense: history REPLAYS the assistant's tool-call
    arguments, so with the value registered the copied turn that is stored and
    replayed must carry the mask, not the plaintext."""
    store = _store()
    store.store_credential("SUDO_PASSWORD", SECRET, "ask")
    from local_operator.secrets import access

    monkeypatch.setattr(access, "retrieve_secret", lambda name, base=None: b"nope")
    _install(monkeypatch, FakeCmux())
    tool = builtin.build_browser_tool(_ctx())
    result = _run(
        tool,
        "t1",
        {"action": "type", "selector": "input[name=q]", "secret_ref": "SUDO_PASSWORD"},
        _secret_ctx(store),
    )
    assert not result.is_error, result.text

    from local_operator.harness.loop import _scrub_history_arguments
    from local_operator.harness.types import Message, TextContent, ToolCall

    message = Message(
        role="assistant",
        content=[TextContent(text="")],
        tool_calls=[
            ToolCall(
                id="c1",
                name="browser",
                arguments={"action": "type", "selector": "input[name=q]", "text": SECRET},
            )
        ],
    )
    scrubbed = _scrub_history_arguments(message, store.redact)
    assert isinstance(scrubbed, Message)
    retained = str(scrubbed.tool_calls[0].arguments)
    assert SECRET not in retained
    assert "[redacted]" in retained


# --- cancellation must not orphan the child ----------------------------------


class HangingProc:
    """A cmux child that never answers, so the only way out is cancellation."""

    def __init__(self) -> None:
        self.killed = False
        self.waited = False
        self.returncode: int | None = None

    async def communicate(self):
        await asyncio.sleep(3600)
        raise AssertionError("unreachable")

    def kill(self) -> None:
        self.killed = True
        self.returncode = -9

    async def wait(self) -> int | None:
        self.waited = True
        return self.returncode


def test_run_cmux_kills_the_child_when_the_turn_is_cancelled(monkeypatch) -> None:
    """CancelledError derives from BaseException, so it passes straight through
    both `except asyncio.TimeoutError` and `except Exception` and used to
    propagate with the child still running and unreaped. The calls most likely
    to be cancelled are the long ones (navigation polls for up to 20 s, 'open'
    allows 30 s), which are also the ones most likely to be wedged, so session
    teardown or an aborted turn orphaned a cmux process each time."""
    proc = HangingProc()
    monkeypatch.setattr(builtin, "_cmux_binary", lambda: "/opt/homebrew/bin/cmux")

    async def fake_exec(*_args, **_kwargs):
        return proc

    monkeypatch.setattr(asyncio, "create_subprocess_exec", fake_exec)

    async def scenario() -> None:
        task = asyncio.ensure_future(builtin._run_cmux(["browser", "--surface", "s", "get", "url"]))
        # Let it get past the spawn and into communicate() before cancelling.
        await asyncio.sleep(0.05)
        task.cancel()
        with pytest.raises(asyncio.CancelledError):
            await task

    asyncio.run(scenario())
    assert proc.killed, "the cmux child was left running"
    assert proc.waited, "an unreaped child is a zombie until the operator exits"


def test_a_running_bridge_with_no_browser_is_not_reported_as_unconfigured(monkeypatch) -> None:
    """Design D3-2: a drop is not an unconfigured host.

    After the daemon severs a silent link, `extension_connected` is false, so
    `state_store.liveness()` answers ABSENT and the demotion guard fires — and it
    used to tell the reader to run `lop browser install` to "set up the bridge",
    on a machine where the bridge is installed, paired and running. The same
    session's later `recover` got the good copy, so the incident's own window was
    the one place the answer was wrong (design D1/D2-1's class, on a third site).

    Two shapes, both keyed on facts the file already carries: the daemon
    latched a silent link (the wedge), or the browser simply is not attached
    right now. Neither names an install, and both carry a typed `error_code` so
    the agent can branch to the remedy instead of substring-matching prose.
    """
    from local_operator.browser_bridge import state as state_store

    monkeypatch.setattr(builtin, "cmux_browser_available", lambda: False)

    def classify(*, extension_unresponsive: bool):
        # `pid` is THIS process, so the classification is about a live daemon;
        # the guard reads the state object directly, so no socket is opened.
        current = state_store.BridgeState(
            pid=os.getpid(),
            port=4099,
            session_key="s" * 32,
            proto=1,
            extension_connected=False,
            paired=False,
            extension_id="a" * 32,
            browser_name="Chrome/153",
            extension_unresponsive=extension_unresponsive,
        )
        return lambda: (state_store.Liveness.ABSENT, current)

    forced = builtin.AgentTool(
        name="browser",
        label="Browser",
        description="d",
        parameters={},
        approval_tier="write",
        concurrency="shared",
        execute=builtin.execute_browser,
    )

    # 1. The wedge: the daemon dropped a link for silence and says so.
    monkeypatch.setattr(builtin, "_bridge_liveness", classify(extension_unresponsive=True))
    wedged = _run(forced, "t1", {"action": "tabs"}, _ctx())
    assert wedged.is_error
    assert "lop browser install" not in wedged.text
    assert "stopped answering" in wedged.text, wedged.text
    assert (wedged.details or {}).get("error_code") == "extension_unresponsive"

    # 2. No browser attached, no latch: still not an unconfigured host.
    monkeypatch.setattr(builtin, "_bridge_liveness", classify(extension_unresponsive=False))
    idle = _run(forced, "t2", {"action": "tabs"}, _ctx())
    assert idle.is_error
    assert "lop browser install" not in idle.text
    assert "remembers this browser" in idle.text, idle.text
    assert (idle.details or {}).get("error_code") == "extension_disconnected"


def test_screenshot_into_an_over_cap_pad_is_refused_before_cmux(monkeypatch, tmp_path) -> None:
    """The REAL tool path (review n4b): a screenshot destination is an ordinary
    path, so it must reach the same pad policy ``write`` does, and the refusal
    must land before cmux is invoked — not after a PNG exists."""
    from local_operator import scratchpad as scratchpad_module

    fake = _install(monkeypatch, FakeCmux())
    pad = tmp_path / "sessions" / "s" / "scratchpad"
    pad.mkdir(parents=True)
    (pad / "bulk.dat").write_bytes(b"x" * 4096)
    config = tmp_path / "cfg"
    config.mkdir()
    (config / "config.yml").write_text(
        "values:\n  session:\n    scratchpad:\n      max_bytes: 4095\n", encoding="utf-8"
    )
    monkeypatch.setenv("LOCAL_OPERATOR_CONFIG_DIR", str(config))
    monkeypatch.setattr(scratchpad_module, "SCRATCHPAD_MAX_BYTES_FLOOR", 1024)

    tool = builtin.build_browser_tool(_ctx())
    ctx = _with_surface()
    ctx.scratchpad_dir = str(pad)
    result = _run(tool, "t1", {"action": "screenshot", "path": str(pad / "shot.png")}, ctx)

    assert result.is_error is True, result.text
    assert "4,095-byte ceiling" in result.text
    assert not (pad / "shot.png").exists()
    assert all(call[3] != "screenshot" for call in fake.calls if call[:1] == ["browser"])


# --- snapshot discovery: scope, window, search -------------------------------
#
# The measured incident (2026-10-23, GitHub's "Create GitHub App" form): a
# 3,000-line accessibility tree was cut mid-tree by the model-facing character
# ceiling, every ref below the cut was unobtainable, and `snapshot` had no way
# to ask for one section. These tests pin the three answers — a subtree, a page,
# and a search term — and the footer that makes a narrowed tree recoverable.

SNAPSHOT_TREE = "\n".join(
    ['- RootWebArea "Create App" [e1]', "  - form", '    - textbox "GitHub App name" [e2]']
    + [f'    - textbox "Setting {index}" [e{index + 3}]' for index in range(200)]
)


def _snapshot_bridge(monkeypatch, result: dict[str, Any]) -> list[dict[str, Any]]:
    """Pin host probes and capture the wire params a snapshot sends."""
    sent: list[dict[str, Any]] = []

    async def fake_call(tool_call_id, action, params, *, surface="", client=None):
        sent.append({"action": action, "params": dict(params)})
        return dict(result), None

    _bridge_only(monkeypatch)
    monkeypatch.setattr(builtin, "bridge_browser_advertisable", lambda: True)
    monkeypatch.setattr(builtin, "_bridge_call", fake_call)
    return sent


def test_snapshot_forwards_the_scope_and_window_to_the_host(monkeypatch) -> None:
    sent = _snapshot_bridge(monkeypatch, {"snapshot": "- form", "total_lines": 1})
    tool = builtin.build_browser_tool(_ctx())
    assert tool is not None
    result = _run(
        tool,
        "t1",
        {
            "action": "snapshot",
            "selector": "form#new_app",
            "start": 41,
            "limit": 20,
            "search": "Active",
        },
        _secret_ctx(surface_id="bridge:9:nonce"),
    )

    assert not result.is_error, result.text
    assert sent and sent[-1]["action"] == "snapshot"
    assert sent[-1]["params"]["selector"] == "form#new_app"
    assert sent[-1]["params"]["start"] == 41
    assert sent[-1]["params"]["limit"] == 20
    assert sent[-1]["params"]["search"] == "Active"
    assert sent[-1]["params"]["tab"] == "bridge:9:nonce"


def test_snapshot_sends_no_window_when_none_was_asked_for(monkeypatch) -> None:
    """The pre-existing whole-tree call must stay byte-identical on the wire."""
    sent = _snapshot_bridge(monkeypatch, {"snapshot": "- form", "total_lines": 1})
    tool = builtin.build_browser_tool(_ctx())
    result = _run(tool, "t1", {"action": "snapshot"}, _secret_ctx(surface_id="bridge:9:nonce"))

    assert not result.is_error, result.text
    assert "selector" not in sent[-1]["params"]
    assert "start" not in sent[-1]["params"]
    assert "limit" not in sent[-1]["params"]
    assert "search" not in sent[-1]["params"]


def test_snapshot_footer_names_the_next_page_a_host_windowed(monkeypatch) -> None:
    """The host reports the arithmetic; the tool must render it as a next call.

    This is the exact failure the incident repeated: the cut was visible, the
    recovery was not. `start=51` in the footer is the whole deliverable.
    """
    _snapshot_bridge(
        monkeypatch,
        {
            "snapshot": "\n".join(f'- textbox "Field {i}" [e{i}]' for i in range(50)),
            "total_lines": 431,
            "matched_lines": 431,
            "returned_lines": 50,
            "start": 1,
            "limit": 50,
            "scoped": True,
        },
    )
    tool = builtin.build_browser_tool(_ctx())
    result = _run(
        tool,
        "t1",
        {"action": "snapshot", "selector": "form", "limit": 50},
        _secret_ctx(surface_id="bridge:9:nonce"),
    )

    assert not result.is_error, result.text
    assert "lines 1-50 of 431" in result.text
    assert "start=51" in result.text
    assert "scoped to 'form'" in result.text
    assert result.details["total_lines"] == 431
    assert result.details["scoped"] is True


def test_snapshot_footer_is_silent_for_a_complete_unscoped_tree(monkeypatch) -> None:
    _snapshot_bridge(
        monkeypatch,
        {
            "snapshot": "- RootWebArea",
            "total_lines": 1,
            "matched_lines": 1,
            "returned_lines": 1,
            "start": 1,
        },
    )
    tool = builtin.build_browser_tool(_ctx())
    result = _run(tool, "t1", {"action": "snapshot"}, _secret_ctx(surface_id="bridge:9:nonce"))

    assert not result.is_error, result.text
    assert "snapshot:" not in result.text


def test_snapshot_windows_locally_when_the_host_reports_no_numbers(monkeypatch) -> None:
    """cmux and an older desktop app ignore the params; paging still works.

    The refs are the point: dropping whole text LINES never renumbers a ref, so
    `e57` seen on page one is still `e57` on page three.
    """
    _snapshot_bridge(monkeypatch, {"snapshot": SNAPSHOT_TREE})
    tool = builtin.build_browser_tool(_ctx())
    result = _run(
        tool,
        "t1",
        {"action": "snapshot", "start": 3, "limit": 2},
        _secret_ctx(surface_id="bridge:9:nonce"),
    )

    assert not result.is_error, result.text
    body, _, footer = result.text.partition("\n\n")
    assert body.splitlines() == [
        '    - textbox "GitHub App name" [e2]',
        '    - textbox "Setting 0" [e3]',
    ]
    assert "lines 3-4 of 203" in footer
    assert "start=5" in footer


def test_snapshot_search_filters_the_local_window(monkeypatch) -> None:
    _snapshot_bridge(monkeypatch, {"snapshot": SNAPSHOT_TREE})
    tool = builtin.build_browser_tool(_ctx())
    result = _run(
        tool,
        "t1",
        {"action": "snapshot", "search": "Active"},
        _secret_ctx(surface_id="bridge:9:nonce"),
    )

    # The synthetic tree has no "Active": the honest answer is the count, not a
    # silent empty page that reads like an empty section.
    assert not result.is_error, result.text
    assert "(empty snapshot)" in result.text
    assert "0 of 203 lines matched 'Active'" in result.text


def test_snapshot_says_when_the_host_ignored_the_selector(monkeypatch) -> None:
    """An old host accepts the param and ignores it.

    Without this sentence the agent reads a full page as "the form's subtree" —
    the most dangerous failure mode of an additive wire field, because the
    result looks successful.
    """
    _snapshot_bridge(monkeypatch, {"snapshot": SNAPSHOT_TREE})
    tool = builtin.build_browser_tool(_ctx())
    result = _run(
        tool,
        "t1",
        {"action": "snapshot", "selector": "#webhook-section"},
        _secret_ctx(surface_id="bridge:9:nonce"),
    )

    assert not result.is_error, result.text
    assert "ignored selector" in result.text
    assert "update the desktop app" in result.text


def test_snapshot_window_is_refused_on_another_action(monkeypatch) -> None:
    # Pin the host: `build_browser_tool` returns None without one, and the test
    # then fails with an AttributeError that says nothing about windows.
    monkeypatch.setattr(builtin, "cmux_browser_available", lambda: True)
    tool = builtin.build_browser_tool(_ctx())
    result = _run(
        tool,
        "t1",
        {"action": "click", "selector": "button", "search": "Active"},
        _secret_ctx(surface_id="bridge:9:nonce"),
    )

    assert result.is_error is True
    assert "'search' only valid for 'snapshot'" in result.text


def test_snapshot_window_values_are_refused_when_unusable(monkeypatch) -> None:
    monkeypatch.setattr(builtin, "cmux_browser_available", lambda: True)
    tool = builtin.build_browser_tool(_ctx())
    for args, expected in (
        ({"action": "snapshot", "start": 0}, "'start' is 1-based"),
        ({"action": "snapshot", "limit": 0}, "'limit' must be at least 1 line"),
        (
            {"action": "snapshot", "limit": builtin.SNAPSHOT_MAX_LINES + 1},
            f"at most {builtin.SNAPSHOT_MAX_LINES} lines",
        ),
        (
            {"action": "snapshot", "search": "x" * (builtin.SNAPSHOT_MAX_SEARCH_CHARS + 1)},
            "name the control, not the page",
        ),
    ):
        result = _run(tool, "t1", args, _secret_ctx(surface_id="bridge:9:nonce"))
        assert result.is_error is True, (args, result.text)
        assert expected in result.text, (args, result.text)


def test_a_host_reporting_zero_matches_is_not_read_as_every_line_matching(monkeypatch) -> None:
    """`matched_lines: 0` is a result, not a falsy value to default away.

    The bug this pins: `result.get("matched_lines") or total` turns "the term is
    not in the tree" into "461 of 461 lines matched", which then renders as a
    paging footer for a page that does not exist.
    """
    _snapshot_bridge(
        monkeypatch,
        {
            "snapshot": "",
            "total_lines": 461,
            "matched_lines": 0,
            "returned_lines": 0,
            "start": 1,
            "limit": 0,
            "search": "Active",
            "scoped": True,
        },
    )
    tool = builtin.build_browser_tool(_ctx())
    result = _run(
        tool,
        "t1",
        {"action": "snapshot", "search": "Active"},
        _secret_ctx(surface_id="bridge:9:nonce"),
    )

    assert not result.is_error, result.text
    assert "0 of 461 lines matched 'Active'" in result.text
    assert "start is past the end" not in result.text


# --- remediation round 1 -----------------------------------------------------
#
# Each of these pins a review finding from round 1, and each fails on the head
# that was reviewed.


def test_cmux_snapshot_windows_the_tree_it_was_handed(monkeypatch) -> None:
    """R1-1: the degrade-to-local-windowing claim must be true for cmux too.

    cmux has no window flags, so `start`/`limit`/`search` have to be applied to
    the text the CLI returned. Before this, `start=2` re-returned page one
    forever and `search` filtered nothing — a claim in the PR that the code did
    not implement.
    """
    fake = FakeCmux()
    monkeypatch.setattr(builtin, "cmux_browser_available", lambda: True)
    monkeypatch.setattr(
        builtin,
        "_run_cmux",
        lambda argv, timeout=30.0: _async((0, "\n".join(f"- row {n}" for n in range(1, 6)))),
    )
    tool = builtin.build_browser_tool(_ctx())
    result = _run(
        tool,
        "t1",
        {"action": "snapshot", "start": 2, "limit": 2},
        _with_surface("surface:3"),
    )

    assert not result.is_error, result.text
    assert "- row 2" in result.text and "- row 3" in result.text
    assert "- row 1" not in result.text and "- row 4" not in result.text
    assert "lines 2-3 of 5" in result.text
    assert "pass start=4" in result.text

    filtered = _run(
        tool,
        "t2",
        {"action": "snapshot", "search": "row 5"},
        _with_surface("surface:3"),
    )
    assert not filtered.is_error, filtered.text
    assert "- row 5" in filtered.text
    assert "- row 1" not in filtered.text
    assert "matching 'row 5'" in filtered.text
    del fake


def test_a_start_past_the_end_never_prints_an_inverted_range(monkeypatch) -> None:
    """R1-5: `lines 500-499 of 203` was honest in its clause and wrong in its numbers."""
    _snapshot_bridge(
        monkeypatch,
        {
            "snapshot": "- row",
            "total_lines": 203,
            "matched_lines": 203,
            "returned_lines": 0,
            "start": 500,
        },
    )
    tool = builtin.build_browser_tool(_ctx())
    result = _run(
        tool,
        "t1",
        {"action": "snapshot", "start": 500},
        _secret_ctx(surface_id="bridge:9:nonce"),
    )

    assert not result.is_error, result.text
    assert "lines 500-499" not in result.text
    assert "start=500 is past the end of 203 lines" in result.text


def test_a_context_cut_page_does_not_name_a_next_start(monkeypatch) -> None:
    """R1-6: following `start + returned` after a character cut skips whole lines.

    The page has to be BOTH windowed (returned < matched) and over the character
    ceiling, or the branch this pins is unreachable and the test passes on the
    defective head (agent review round 2, R2-1).
    """
    lines = [f"- textbox \"Field {n} {'.' * 180}\"" for n in range(1, 901)]
    _snapshot_bridge(
        monkeypatch,
        {
            "snapshot": "\n".join(lines[:50]),
            "total_lines": len(lines),
            "matched_lines": len(lines),
            "returned_lines": 50,
            "start": 1,
            "limit": 50,
        },
    )
    tool = builtin.build_browser_tool(_ctx())
    result = _run(
        tool,
        "t1",
        {"action": "snapshot", "limit": 50},
        _secret_ctx(surface_id="bridge:9:nonce"),
    )

    assert not result.is_error, result.text
    assert "cut to fit the context" in result.text
    assert "pass start=" not in result.text
    assert "lines 1-50 of 900" in result.text


# --- accessibility locators --------------------------------------------------
#
# The measured failure: every id on the target form was framework-generated and
# nine CSS guesses missed. The element's accessible identity — "a checkbox named
# Active" — was never in doubt. These tests pin the tool's half: the locator
# reaches the host unchanged, the answer names what was actually clicked, and a
# host that predates locators is not left to fail as a CSS syntax error.


def _bridge_result(monkeypatch, result: dict[str, Any]) -> list[dict[str, Any]]:
    sent: list[dict[str, Any]] = []

    async def fake_call(tool_call_id, action, params, *, surface="", client=None):
        sent.append({"action": action, "params": dict(params)})
        return dict(result), None

    _bridge_only(monkeypatch)
    monkeypatch.setattr(builtin, "bridge_browser_advertisable", lambda: True)
    monkeypatch.setattr(builtin, "_bridge_call", fake_call)
    return sent


def test_a_locator_reaches_the_host_unchanged(monkeypatch) -> None:
    sent = _bridge_result(
        monkeypatch,
        {
            "navigated": False,
            "url": "https://example.test/",
            "title": "Create App",
            "resolved": {"by": "locator", "role": "checkbox", "name": "Active", "matched": 1},
        },
    )
    tool = builtin.build_browser_tool(_ctx())
    result = _run(
        tool,
        "t1",
        {"action": "click", "selector": 'role=checkbox[name="Active"]'},
        _secret_ctx(surface_id="bridge:9:nonce"),
    )

    assert not result.is_error, result.text
    assert sent[-1]["params"]["selector"] == 'role=checkbox[name="Active"]'
    # The answer names the element that was clicked, not just the locator.
    assert 'role=checkbox[name="Active"] → checkbox "Active"' in result.text


def test_an_ambiguous_locator_says_the_first_match_was_used(monkeypatch) -> None:
    _bridge_result(
        monkeypatch,
        {
            "navigated": False,
            "url": "https://example.test/",
            "title": "Create App",
            "resolved": {
                "by": "locator",
                "role": "radio",
                "name": "Only on this account",
                "matched": 3,
            },
        },
    )
    tool = builtin.build_browser_tool(_ctx())
    result = _run(
        tool,
        "t1",
        {"action": "click", "selector": 'label="Only on this account"'},
        _secret_ctx(surface_id="bridge:9:nonce"),
    )

    assert not result.is_error, result.text
    assert "3 matched, first used" in result.text


def test_a_ref_click_keeps_its_existing_copy(monkeypatch) -> None:
    """The new `resolved` field must not churn the message agents already read."""
    _bridge_result(monkeypatch, {"navigated": False, "url": "https://example.test/", "title": "T"})
    tool = builtin.build_browser_tool(_ctx())
    result = _run(
        tool,
        "t1",
        {"action": "click", "selector": "e57"},
        _secret_ctx(surface_id="bridge:9:nonce"),
    )

    assert result.text.startswith("Clicked e57 (no navigation).")


def test_a_locator_is_accepted_for_upload_too(monkeypatch) -> None:
    """R1-5: `upload` resolves through the same target path, so a locator works."""
    monkeypatch.setattr(builtin, "cmux_browser_available", lambda: True)
    tool = builtin.build_browser_tool(_ctx())
    result = _run(
        tool,
        "t1",
        {"action": "upload", "selector": 'label="Choose file"', "paths": []},
        _secret_ctx(surface_id="bridge:9:nonce"),
    )
    # The paths list is empty, so this refuses on THAT — never on the locator.
    assert "locator" not in result.text.lower()


def test_a_locator_is_refused_where_only_css_makes_sense(monkeypatch) -> None:
    monkeypatch.setattr(builtin, "cmux_browser_available", lambda: True)
    tool = builtin.build_browser_tool(_ctx())
    for action in ("snapshot", "read", "scroll"):
        result = _run(
            tool,
            f"t-{action}",
            {"action": action, "selector": 'role=checkbox[name="Active"]'},
            _secret_ctx(surface_id="bridge:9:nonce"),
        )
        assert result.is_error is True, (action, result.text)
        assert "role= locators resolve a click/type TARGET" in result.text or (
            "snapshot takes a CSS selector" in result.text
        ), (action, result.text)


def test_a_malformed_locator_is_not_blamed_on_the_host() -> None:
    """R1-3: an unterminated quote is the CALLER's typo, not an old host.

    Before this, any `internal` error on a locator-shaped selector said "this
    browser host predates accessibility locators. Update the desktop app" — the
    wrong remedy, and the same misdiagnosis class `_capability_message` exists
    to avoid.
    """
    from local_operator.browser_bridge.backend import BridgeError
    from local_operator.browser_bridge.protocol import ErrorCode

    class HostThatSaysNo:
        host = "extension"

        async def call(self, action, params):
            raise BridgeError(ErrorCode.INTERNAL, "DOM Error while querying")

    _result, problem = asyncio.run(
        builtin._bridge_call(
            "t1",
            "click",
            {"selector": 'role=checkbox[name="Active'},
            surface="bridge:9:nonce",
            client=HostThatSaysNo(),
        )
    )

    assert problem is not None and problem.is_error
    assert "unterminated" in problem.text
    assert "predates accessibility locators" not in problem.text


def test_an_old_host_treating_a_locator_as_css_is_explained(monkeypatch) -> None:
    """The additive-param trap, on the failure path.

    A host that predates locators parses `role=…` as CSS and answers a bare DOM
    error. Without this sentence the agent sees a syntax failure and starts
    guessing — the loop the locator exists to end.
    """
    from local_operator.browser_bridge.backend import BridgeError
    from local_operator.browser_bridge.protocol import ErrorCode

    class HostThatPredatesLocators:
        host = "extension"

        async def call(self, action, params):
            raise BridgeError(ErrorCode.INTERNAL, "DOM Error while querying")

    _result, problem = asyncio.run(
        builtin._bridge_call(
            "t1",
            "click",
            {"selector": 'role=checkbox[name="Active"]'},
            surface="bridge:9:nonce",
            client=HostThatPredatesLocators(),
        )
    )

    assert problem is not None and problem.is_error
    assert "predates accessibility locators" in problem.text
    assert "snapshot ref" in problem.text


def test_brackets_inside_a_quoted_locator_value_are_content() -> None:
    """Round-2 residual: `text="Array [1]"` is well-formed, not trailing content."""
    assert builtin._locator_shape_problem('text="Array [1]"') == ""
    assert builtin._locator_shape_problem('role=checkbox[name="Active [x]"]') == ""
    # Balanced quotes, unbalanced bracket: the bracket is what is wrong.
    assert "unbalanced" in builtin._locator_shape_problem('role=checkbox[name="Active"')
    assert "unterminated" in builtin._locator_shape_problem('role=checkbox[name="Active]')
    assert "unterminated" in builtin._locator_shape_problem('role=checkbox[name="Active')
