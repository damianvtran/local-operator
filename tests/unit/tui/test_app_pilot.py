"""OperatorApp Pilot tests — boot, prompt dispatch, slash commands, quit.

Uses a ``FakeSession`` implementing ``SessionProtocol`` so the TUI runs
without providers/network. The factory shape mirrors production: the app
paints first, then awaits the session in a worker.
"""

from __future__ import annotations

from typing import Any

import pytest

from local_operator.tui.app import OperatorApp, SLASH_COMMANDS
from local_operator.tui.widgets.editor import Editor
from local_operator.tui.widgets.tool_card import ToolCard
from local_operator.tui.widgets.transcript import NoticeBlock, TranscriptView


class FakeSession:
    """Records prompts/aborts; satisfies SessionProtocol."""

    def __init__(self) -> None:
        self.prompts: list[str] = []
        self.aborts: list[str] = []
        self.disposed = False
        self._handlers: list[Any] = []

    @property
    def session_id(self) -> str:
        return "sess"

    @property
    def agent_id(self) -> str:
        return "agent"

    @property
    def is_streaming(self) -> bool:
        return False

    @property
    def model_label(self) -> str:
        return "test/model"

    async def prompt(self, text: str, attachments: list[Any] | None = None) -> None:
        self.prompts.append(text)

    def steer(self, text: str) -> None:
        pass

    def abort(self, reason: str = "interrupted") -> None:
        self.aborts.append(reason)

    def subscribe(self, handler: Any) -> Any:
        self._handlers.append(handler)

        def unsubscribe() -> None:
            if handler in self._handlers:
                self._handlers.remove(handler)

        return unsubscribe

    async def dispose(self) -> None:
        self.disposed = True

    def emit(self, event: Any) -> None:
        for handler in list(self._handlers):
            handler(event)


async def _factory(session: FakeSession) -> FakeSession:
    return session


@pytest.mark.asyncio
async def test_boot_typing_sends_prompt() -> None:
    """Boot the app, type text, press Enter: the session records the prompt."""
    session = FakeSession()
    app = OperatorApp(lambda: _factory(session))
    async with app.run_test(size=(80, 24)) as pilot:
        await pilot.pause()
        editor = app.query_one(Editor)
        editor.focus()
        await pilot.pause()
        await pilot.press("h", "i")
        await pilot.press("enter")
        await pilot.pause()
        await pilot.pause()
        assert session.prompts == ["hi"]
        # A user block was appended for the submitted prompt (the boot hint
        # is lifted by the first real block, D9).
        transcript = app.query_one(TranscriptView)
        assert len(transcript.blocks()) == 1


@pytest.mark.asyncio
async def test_exit_command_quits() -> None:
    """``/exit`` handled synchronously quits the app without prompting."""
    session = FakeSession()
    app = OperatorApp(lambda: _factory(session))
    async with app.run_test(size=(80, 24)) as pilot:
        await pilot.pause()
        app.query_one(Editor).focus()
        await pilot.pause()
        await pilot.press("slash", "e", "x", "i", "t", "enter")
        await pilot.pause()
        assert not app.is_running
    assert session.prompts == []


@pytest.mark.asyncio
async def test_quit_alias_quits() -> None:
    session = FakeSession()
    app = OperatorApp(lambda: _factory(session))
    async with app.run_test(size=(80, 24)) as pilot:
        await pilot.pause()
        app.query_one(Editor).focus()
        await pilot.pause()
        await pilot.press("slash", "q", "u", "i", "t", "enter")
        await pilot.pause()
        assert not app.is_running


def test_exit_quit_collapsed_to_one_command() -> None:
    """TUI-014: ONE registry entry; ``quit`` rides as an alias of ``exit``."""
    names = [c.name for c in SLASH_COMMANDS]
    assert "exit" in names
    assert "quit" not in names  # not a separate command
    exit_command = next(c for c in SLASH_COMMANDS if c.name == "exit")
    assert exit_command.aliases == ("quit",)


@pytest.mark.asyncio
async def test_help_and_model_notices() -> None:
    """``/model`` posts a notice; it never prompts (D15 echoes the slash)."""
    session = FakeSession()
    app = OperatorApp(lambda: _factory(session))
    async with app.run_test(size=(80, 24)) as pilot:
        await pilot.pause()
        app.query_one(Editor).focus()
        await pilot.pause()
        await pilot.press("slash", "m", "o", "d", "e", "l", "enter")
        await pilot.pause()
        transcript = app.query_one(TranscriptView)
        # D15: the slash command echoes as a UserBlock before the notice.
        assert len(transcript.blocks()) == 2
    assert session.prompts == []


@pytest.mark.asyncio
async def test_clear_resets_transcript_and_bookkeeping() -> None:
    """TUI-009: /clear resets _streaming_block/_tool_cards AND posts a
    notice that history is untouched (cosmetic clear)."""
    session = FakeSession()
    app = OperatorApp(lambda: _factory(session))
    async with app.run_test(size=(80, 24)) as pilot:
        await pilot.pause()
        app.query_one(Editor).focus()
        await pilot.pause()
        await pilot.press("h", "i", "enter")
        await pilot.pause()
        transcript = app.query_one(TranscriptView)
        assert len(transcript.blocks()) == 1
        # Simulate live bookkeeping so we can prove the reset.
        card = ToolCard("t9", "bash", {"command": "ls"})
        app._tool_cards["t9"] = card
        await pilot.press("slash", "c", "l", "e", "a", "r", "enter")
        await pilot.pause()
        blocks = transcript.blocks()
        assert len(blocks) == 1  # only the "history untouched" notice
        assert isinstance(blocks[0], NoticeBlock)
        assert "untouched" in blocks[0].renderable.plain  # type: ignore[attr-defined]
        assert app._streaming_block is None
        assert app._tool_cards == {}


@pytest.mark.asyncio
async def test_ctrl_l_clears_and_resets() -> None:
    """Ctrl+L runs the same clear path as /clear (TUI-009)."""
    session = FakeSession()
    app = OperatorApp(lambda: _factory(session))
    async with app.run_test(size=(80, 24)) as pilot:
        await pilot.pause()
        app.query_one(Editor).focus()
        await pilot.pause()
        await pilot.press("h", "i", "enter")
        await pilot.pause()
        app._streaming_block = object()  # type: ignore[assignment]
        await pilot.press("ctrl+l")
        await pilot.pause()
        assert app._streaming_block is None
        assert app._tool_cards == {}


@pytest.mark.asyncio
async def test_session_disposed_on_exit() -> None:
    session = FakeSession()
    app = OperatorApp(lambda: _factory(session))
    async with app.run_test(size=(80, 24)) as pilot:
        await pilot.pause()
        await pilot.press("slash", "e", "x", "i", "t", "enter")
        await pilot.pause()
    assert session.disposed


# --- keybinding pilot tests (TUI-026) -------------------------------------


@pytest.mark.asyncio
async def test_ctrl_c_interrupts_and_app_stays_alive() -> None:
    """Ctrl+C posts InterruptRequested (abort the turn) and never exits."""
    session = FakeSession()
    app = OperatorApp(lambda: _factory(session))
    async with app.run_test(size=(80, 24)) as pilot:
        await pilot.pause()
        app.query_one(Editor).focus()
        await pilot.pause()
        await pilot.press("ctrl+c")
        await pilot.pause()
        assert app.is_running  # the app stays alive
        assert session.aborts == ["interrupted"]


@pytest.mark.asyncio
async def test_shift_enter_inserts_newline_without_submit() -> None:
    session = FakeSession()
    app = OperatorApp(lambda: _factory(session))
    async with app.run_test(size=(80, 24)) as pilot:
        await pilot.pause()
        editor = app.query_one(Editor)
        editor.focus()
        await pilot.pause()
        await pilot.press("a")
        await pilot.press("shift+enter")
        await pilot.press("b")
        await pilot.pause()
        # No submit happened; the buffer carries a newline.
        assert session.prompts == []
        assert editor.text == "a\nb"


@pytest.mark.asyncio
async def test_tab_completes_slash_without_losing_focus() -> None:
    """Tab completes /he -> /help and focus stays on the editor (TUI-013)."""
    session = FakeSession()
    app = OperatorApp(lambda: _factory(session))
    async with app.run_test(size=(80, 24)) as pilot:
        await pilot.pause()
        editor = app.query_one(Editor)
        editor.focus()
        await pilot.pause()
        await pilot.press("slash", "h", "e")
        await pilot.press("tab")
        await pilot.pause()
        assert editor.text == "/help"
        assert editor.has_focus  # completion never moves focus


# --- boot failure + reload (TUI-012) --------------------------------------


@pytest.mark.asyncio
async def test_boot_failure_posts_error_and_reload_retries() -> None:
    attempts = {"n": 0}
    session = FakeSession()

    async def flaky_factory() -> FakeSession:
        attempts["n"] += 1
        if attempts["n"] == 1:
            raise RuntimeError("provider is down")
        return session

    app = OperatorApp(flaky_factory)
    async with app.run_test(size=(80, 24)) as pilot:
        await pilot.pause()
        await pilot.pause()
        # Boot failure surfaces as an error notice + 'session error' status.
        transcript = app.query_one(TranscriptView)
        kinds = [type(b).__name__ for b in transcript.blocks()]
        texts = "\n".join(
            getattr(getattr(b, "renderable", None), "plain", "") for b in transcript.blocks()
        )
        assert "NoticeBlock" in kinds
        assert "provider is down" in texts
        assert app._session is None
        # /reload re-runs boot and succeeds this time.
        app.query_one(Editor).focus()
        await pilot.pause()
        await pilot.press("slash", "r", "e", "l", "o", "a", "d", "enter")
        await pilot.pause()
        await pilot.pause()
        assert attempts["n"] == 2
        assert app._session is session


# --- login/logout handlers (TUI-015) ---------------------------------------


@pytest.mark.asyncio
async def test_login_logout_dispatch_to_handler() -> None:
    calls: list[str] = []
    session = FakeSession()
    app = OperatorApp(lambda: _factory(session), login_handler=calls.append)
    async with app.run_test(size=(80, 24)) as pilot:
        await pilot.pause()
        app.query_one(Editor).focus()
        await pilot.pause()
        await pilot.press("slash", "l", "o", "g", "i", "n", "enter")
        await pilot.pause()
        await pilot.press("slash", "l", "o", "g", "o", "u", "t", "enter")
        await pilot.pause()
    assert calls == ["login", "logout"]


@pytest.mark.asyncio
async def test_login_without_handler_notices_cli_command() -> None:
    session = FakeSession()
    app = OperatorApp(lambda: _factory(session))  # login_handler None
    async with app.run_test(size=(80, 24)) as pilot:
        await pilot.pause()
        app.query_one(Editor).focus()
        await pilot.pause()
        await pilot.press("slash", "l", "o", "g", "i", "n", "enter")
        await pilot.pause()
        transcript = app.query_one(TranscriptView)
        texts = "\n".join(
            getattr(getattr(b, "renderable", None), "plain", "") for b in transcript.blocks()
        )
        assert "local-operator login" in texts


# --- skills/mcp introspection (exception-safe) -----------------------------


@pytest.mark.asyncio
async def test_skills_and_mcp_commands_never_crash() -> None:
    """Whatever the discovery layer returns, the handlers stay graceful."""
    session = FakeSession()
    app = OperatorApp(lambda: _factory(session))
    async with app.run_test(size=(80, 24)) as pilot:
        await pilot.pause()
        app.query_one(Editor).focus()
        await pilot.pause()
        await pilot.press("slash", "s", "k", "i", "l", "l", "s", "enter")
        await pilot.pause()
        await pilot.press("slash", "m", "c", "p", "enter")
        await pilot.pause()
        transcript = app.query_one(TranscriptView)
        assert len(transcript.blocks()) >= 2  # echoes + notices/listings
    assert session.prompts == []


# --- autocomplete scoring (sync, I/O-free) --------------------------------
from local_operator.tui.autocomplete import (  # noqa: E402
    SlashCommand,
    complete_command,
    match_commands,
    score_command_text_match,
)


def test_scoring_exact_beats_prefix_beats_fuzzy() -> None:
    assert score_command_text_match("help", "help") == 1000
    assert score_command_text_match("he", "help") == 900
    fuzzy = score_command_text_match("hp", "help")
    assert 1 <= fuzzy <= 40
    assert 1000 > 900 > fuzzy


def test_scoring_case_insensitive_and_no_match() -> None:
    assert score_command_text_match("HELP", "help") == 1000
    assert score_command_text_match("zzz", "help") == 0
    assert score_command_text_match("", "help") == 0


def test_match_orders_by_score_then_registry() -> None:
    commands = [SlashCommand("help"), SlashCommand("history")]
    # "h" is a prefix of both -> flat 900; registry order breaks the tie.
    matches = match_commands("/h", commands)
    assert [name for name, _ in matches] == ["help", "history"]
    # "hi" prefixes only history.
    assert [name for name, _ in match_commands("/hi", commands)] == ["history"]


def test_complete_only_when_unambiguous() -> None:
    commands = [SlashCommand("help"), SlashCommand("history"), SlashCommand("exit")]
    assert complete_command("/hel", commands) == "/help"
    assert complete_command("/ex", commands) == "/exit"
    assert complete_command("/h", commands) is None  # help vs history ambiguous
    assert complete_command("hello", commands) is None  # not a slash token


def test_complete_matches_alias() -> None:
    """TUI-014: the collapsed exit/quit command still completes via alias."""
    commands = [SlashCommand("exit", "Quit", aliases=("quit",))]
    assert complete_command("/q", commands) == "/quit"
    assert complete_command("/ex", commands) == "/exit"
