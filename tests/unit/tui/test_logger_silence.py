"""While the TUI owns the terminal, logging writes to a file and nowhere else.

The reported defect: raw log lines and a traceback cut across the running
interface. The cause was a ``StreamHandler`` built at import over the real
``sys.stderr`` — Textual's own ``redirect_stderr`` cannot help, because the
handler holds the pre-redirect file object and writes straight past it.

The same guarantee has a second half, added after an MCP stdio server tore the
boot splash in two: a CHILD PROCESS we spawn inherits file DESCRIPTORS, not
Python objects, so no amount of handler surgery inside this process reaches it.
Those cases capture fd 2 itself (:func:`_terminal_fd`) rather than swapping
``sys.stderr``, because a test built on the swap passes with the defect fully
present.

These drive the real ``OperatorApp`` because the interesting question is not
"does the context manager remove handlers" (``tests/unit/test_logger.py`` asks
that) but "with the app actually painting, does a log record change what the
user sees".
"""

from __future__ import annotations

import asyncio
import io
import logging
import logging.handlers
import sys
from pathlib import Path
from typing import Any

import pytest

from local_operator.harness.loop import AgentLoop, LoopContext
from local_operator.harness.types import AgentEndEvent, Message
from local_operator.logger import LOG_FILE_NAME, _is_console_handler, file_logging
from local_operator.mcp.config import MCPStdioServerConfig
from local_operator.mcp.manager import McpManager, McpServerStderr, _stdio_transport
from local_operator.paths import CONFIG_DIR_ENV
from local_operator.providers.failover import ProviderError
from local_operator.tui import run_tui
from local_operator.tui.app import OperatorApp
from local_operator.tui.widgets.welcome import WelcomeView

from ..harness.test_loop import make_config
from .test_app_pilot import FakeSession, _factory


@pytest.fixture(autouse=True)
def restore_root_logger():
    """Undo the process-global logging state these tests deliberately mutate."""
    root = logging.getLogger()
    saved_handlers = list(root.handlers)
    saved_level = root.level
    saved_last_resort = logging.lastResort
    saved_add_handler = logging.Logger.addHandler
    try:
        yield
    finally:
        logging.Logger.addHandler = saved_add_handler  # type: ignore
        logging.lastResort = saved_last_resort
        root.handlers[:] = saved_handlers
        root.setLevel(saved_level)


def _frame(app: OperatorApp) -> list[str]:
    """The painted frame as plain text, one entry per row."""
    return [strip.text for strip in app.screen._compositor.render_strips()]


async def _settled_frame(pilot, app: OperatorApp) -> list[str]:
    """Wait until the splash has stopped changing for reasons of its OWN, then
    return the frame.

    Frame equality alone is not the settled edge and never was: the model label
    lands on the splash's 0.25 s poll, and two consecutive 0.1 s pauses can both
    fall inside that window and agree on a frame that still says ``connecting…``.
    The baseline would then diff against the settled frame with logging fully
    silent. The poll timer RETIRES when the label arrives, so waiting on it is
    waiting on the actual event; the frame comparison stays as the backstop for
    anything else still moving.
    """
    welcome = app.query_one(WelcomeView)
    previous = None
    for _ in range(100):
        await pilot.pause(0.1)
        current = _frame(app)
        if welcome._timer is None and current == previous:
            return current
        previous = current
    return previous or _frame(app)


#: A stdio MCP server that decorates, in miniature. Writes a coloured banner to
#: stderr the way ``workspace-mcp`` writes its logo and panels, then stays up so
#: the transport is live while the frame is compared.
NOISY_SERVER = (
    "import sys, time\n"
    "sys.stderr.write('\\x1b[1;35m3 of 12 services loaded\\x1b[0m\\n')\n"
    "sys.stderr.flush()\n"
    "time.sleep(60)\n"
)

#: A stdio MCP server that fails the way a misconfigured one really does: one
#: line saying what is missing, then a non-zero exit before the handshake.
FAILING_SERVER = (
    "import sys\n"
    "sys.stderr.write('fatal: USER_GOOGLE_EMAIL is not set\\n')\n"
    "sys.exit(3)\n"
)


async def _await_log(path: Path, needle: str, timeout: float = 10.0) -> str:
    """Poll the log file until ``needle`` lands, or fail with what it holds."""
    deadline = asyncio.get_running_loop().time() + timeout
    contents = ""
    while asyncio.get_running_loop().time() < deadline:
        if path.exists():
            contents = path.read_text(encoding="utf-8")
            if needle in contents:
                return contents
        await asyncio.sleep(0.05)
    raise AssertionError(f"{needle!r} never reached the log; it holds:\n{contents}")


@pytest.mark.asyncio
async def test_an_mcp_server_cannot_write_on_the_frame_or_the_terminal(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, terminal_output: Path
) -> None:
    """The reported defect: ``workspace-mcp``'s banner tore the boot splash.

    Measured on the real server before the fix — spawned with its stderr
    inherited (``stderr=None``) it put 2508 bytes of ASCII logo and Rich panels
    onto the PTY the app was painting; the same run with stderr on a pipe put
    zero. This is that in miniature, with the app actually running so the frame
    is a real comparison and not a claim.

    All three halves are asserted together on purpose. Silencing the child by
    discarding its stderr would satisfy the first two and destroy the only
    answer a user has to "why did my server not start" — the trade the rest of
    this module exists to refuse.
    """
    monkeypatch.setenv(CONFIG_DIR_ENV, str(tmp_path))
    log_path = tmp_path / "logs" / LOG_FILE_NAME
    cfg = MCPStdioServerConfig(command=sys.executable, args=["-c", NOISY_SERVER])

    app = OperatorApp(lambda: _factory(FakeSession()))

    with file_logging():
        logging.getLogger().setLevel(logging.INFO)
        async with app.run_test(size=(80, 24)) as pilot:
            before = await _settled_frame(pilot, app)

            stderr_log = McpServerStderr("google_workspace")
            async with _stdio_transport(cfg, lambda: None, stderr_log):
                contents = await _await_log(log_path, "3 of 12 services loaded")
                await pilot.pause()
                after = _frame(app)

    assert after == before
    assert terminal_output.read_bytes() == b""
    # Under the child's own logger name, so one server's output is greppable,
    # and stripped of the escapes that corrupt `less` the way they corrupted
    # the frame.
    assert "local_operator.mcp.server.google_workspace: 3 of 12 services loaded" in contents
    assert "\x1b" not in contents


@pytest.mark.asyncio
async def test_a_failing_mcp_server_still_says_why_in_the_log(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, terminal_output: Path
) -> None:
    """Containment must not cost diagnosability.

    A server that dies during the handshake used to surface as a bare
    ``Connection closed`` — the reason it died went to the terminal, where the
    TUI hid it. Now the reason is in the log at ERROR (found even with the root
    logger at WARNING, which is where a ``LOG_LEVEL`` user leaves it) and is
    quoted into the error text that ``McpStartupOutcome.failures`` carries to
    the TUI's notice, which is the surface a user acts on.
    """
    monkeypatch.setenv(CONFIG_DIR_ENV, str(tmp_path))
    log_path = tmp_path / "logs" / LOG_FILE_NAME
    cfg = MCPStdioServerConfig(command=sys.executable, args=["-c", FAILING_SERVER])

    with file_logging():
        logging.getLogger().setLevel(logging.WARNING)
        with pytest.raises(Exception) as caught:
            await McpManager(cwd=str(tmp_path))._connect_server("workspace", cfg)

    assert terminal_output.read_bytes() == b""
    assert "USER_GOOGLE_EMAIL is not set" in str(caught.value)

    contents = log_path.read_text(encoding="utf-8")
    assert "[ERROR] local_operator.mcp.server.workspace" in contents
    assert "USER_GOOGLE_EMAIL is not set" in contents


@pytest.mark.asyncio
async def test_logging_does_not_touch_the_frame_or_the_terminal(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """A warning and an exception mid-session: frame identical, terminal silent.

    ``sys.stderr`` is a buffer and the console handler is built from it BEFORE
    the app starts, reproducing the import-time handler exactly. Asserting on
    the frame alone would be too weak — Textual renders into its own buffer, so
    a stray stderr write corrupts the user's terminal without ever appearing in
    ``render_strips``. Both are checked.
    """
    monkeypatch.setenv(CONFIG_DIR_ENV, str(tmp_path))
    terminal = io.StringIO()
    monkeypatch.setattr(sys, "stderr", terminal)
    console_handler = logging.StreamHandler(sys.stderr)
    logging.getLogger().addHandler(console_handler)
    logging.getLogger().setLevel(logging.INFO)

    session = FakeSession()
    app = OperatorApp(lambda: _factory(session))
    logger = logging.getLogger("local_operator.providers.failover")

    with file_logging():
        async with app.run_test(size=(80, 24)) as pilot:
            before = await _settled_frame(pilot, app)

            logger.warning("HTTP 400: `temperature` is deprecated for this model")
            try:
                raise RuntimeError("ProviderError: every turn fails")
            except RuntimeError:
                logger.exception("provider call failed")
            await pilot.pause()

            after = _frame(app)

    assert after == before
    assert terminal.getvalue() == ""

    contents = (tmp_path / "logs" / LOG_FILE_NAME).read_text(encoding="utf-8")
    assert "temperature` is deprecated" in contents
    assert "ProviderError: every turn fails" in contents


@pytest.mark.asyncio
async def test_a_failing_provider_turn_does_not_paint_its_warning(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """The reported defect, end to end: a real turn failing with a ``ProviderError``.

    The user switched to ``anthropic/claude-opus-5`` and every turn produced
    ``× invalid request (HTTP 400): `temperature` is deprecated for this
    model.`` in the transcript AND a duplicate log line outside the frame. The
    kind leads the line because a status alone never told anyone what to do; see
    ``ProviderError.__str__``. The stack is gone
    (``AgentLoop`` drops ``exc_info`` for a ``RenderedStreamError``); this
    covers the surviving one-liner, which must go to the file and NOT to the
    screen.

    Both halves are asserted on purpose. Raising the logger's level or dropping
    the record would clear the frame and destroy the only durable evidence that
    a turn failed — a visual bug traded for a diagnosability one.

    The real ``AgentLoop`` runs here rather than a hand-rolled
    ``logger.warning``: the message text, the logger name and the decision not
    to attach a stack all live in ``harness/loop.py``, and a synthetic record
    would keep passing after any of them changed.
    """
    monkeypatch.setenv(CONFIG_DIR_ENV, str(tmp_path))
    terminal = io.StringIO()
    monkeypatch.setattr(sys, "stderr", terminal)
    logging.getLogger().addHandler(logging.StreamHandler(sys.stderr))
    logging.getLogger().setLevel(logging.INFO)

    error = ProviderError(400, "`temperature` is deprecated for this model.")

    def boom(request, signal):
        async def gen():
            raise error
            yield  # pragma: no cover - generator marker

        return gen()

    app = OperatorApp(lambda: _factory(FakeSession()))

    with file_logging():
        async with app.run_test(size=(80, 24)) as pilot:
            await pilot.pause()
            events = [
                event
                async for event in AgentLoop().run(
                    [Message.user("go")], LoopContext(), make_config(boom), None
                )
            ]
            await pilot.pause()
            frame = "\n".join(_frame(app))

    end = next(event for event in events if isinstance(event, AgentEndEvent))
    assert str(error) in (end.error or ""), "the turn must actually have failed"

    assert "WARNING" not in frame
    assert "local_operator.harness.loop" not in frame
    assert "model stream failed" not in frame
    assert terminal.getvalue() == ""

    contents = (tmp_path / "logs" / LOG_FILE_NAME).read_text(encoding="utf-8")
    assert "local_operator.harness.loop" in contents
    assert (
        "model stream failed: invalid request (HTTP 400): "
        "`temperature` is deprecated for this model." in contents
    )


@pytest.mark.asyncio
async def test_run_tui_installs_file_logging_around_the_app(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """The wiring itself: the production entry point, not just the helper.

    ``run_tui`` is the TUI's only production entry point, so the context
    manager being wrapped around ``run_async`` — and not merely available — is
    the thing that keeps the screen clean. A stub app records what the root
    logger looked like from inside the run.
    """
    monkeypatch.setenv(CONFIG_DIR_ENV, str(tmp_path))
    terminal = io.StringIO()
    monkeypatch.setattr(sys, "stderr", terminal)
    console_handler = logging.StreamHandler(sys.stderr)
    logging.getLogger().addHandler(console_handler)

    observed: dict[str, Any] = {}

    class StubApp:
        return_code = 0

        def __init__(self, *args: Any, **kwargs: Any) -> None:
            pass

        async def run_async(self) -> None:
            logging.getLogger("local_operator.stub").warning("inside the run")
            observed["handlers"] = list(logging.getLogger().handlers)

        def resume_hint(self) -> str:
            # run_tui prints this AFTER the app releases the terminal. Empty
            # here: the stub has no session, and a stub that invented an id
            # would put a fake resume command in this test's output.
            return ""

    monkeypatch.setattr("local_operator.tui.app.OperatorApp", StubApp)

    code = await run_tui(lambda: _factory(FakeSession()))

    assert code == 0
    # No terminal-bound handler survived into the run, and the rotating file
    # took its place. (pytest's own capture handlers are neither, and are
    # correctly left alone — silencing the terminal is not silencing logging.)
    assert [h for h in observed["handlers"] if _is_console_handler(h)] == []
    assert any(isinstance(h, logging.handlers.RotatingFileHandler) for h in observed["handlers"])
    assert terminal.getvalue() == ""
    assert "inside the run" in (tmp_path / "logs" / LOG_FILE_NAME).read_text(encoding="utf-8")
    # Restored: the plain REPL and `exec` share this process when the CLI
    # falls back to them.
    assert console_handler in logging.getLogger().handlers
