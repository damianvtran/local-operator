"""OperatorApp Pilot tests — boot, prompt dispatch, slash commands, quit.

Uses a ``FakeSession`` implementing ``SessionProtocol`` so the TUI runs
without providers/network. The factory shape mirrors production: the app
paints first, then awaits the session in a worker.
"""

from __future__ import annotations

import asyncio
import json
from collections.abc import Callable, Sequence
from pathlib import Path
from types import SimpleNamespace
from typing import Any, cast
from unittest.mock import patch

import pytest

from local_operator.harness.types import ImageContent, NoticeEvent, TextContent
from local_operator.session.mcp_status import McpStartupOutcome
from local_operator.session.protocol import CompactionOutcome
from local_operator.tui import theme as theme_mod
from local_operator.tui.app import (
    BOOT_LAYOUT_CLASS,
    COMPOSER_FOCUSED_CLASS,
    PERSIST_HINT,
    SLASH_COMMANDS,
    OperatorApp,
)
from local_operator.tui.autocomplete import ArgumentChoice
from local_operator.tui.events import TurnEnded, TurnStarted
from local_operator.tui.widgets.approval import ApprovalBlock
from local_operator.tui.widgets.assistant import AssistantBlock
from local_operator.tui.widgets.editor import ASIDE_PLACEHOLDER, Editor
from local_operator.tui.widgets.session_picker import SessionPickerScreen
from local_operator.tui.widgets.toast import Toast
from local_operator.tui.widgets.tool_card import ToolCard
from local_operator.tui.widgets.transcript import NoticeBlock, TranscriptView
from local_operator.tui.widgets.welcome import WelcomeView
from tests.unit.tui.conftest import caret_cells, chevron_colour, composer_cells


class FakeSession:
    """Records prompts/aborts; satisfies SessionProtocol."""

    def __init__(self) -> None:
        self.prompts: list[str] = []
        self.aborts: list[str] = []
        self.completions: list[tuple[str, str]] = []
        self.asides: list[list[Any]] = []
        self.adopted: list[list[Any]] = []
        self.disposed = False
        self._handlers: list[Any] = []
        self._history: list[Any] = []
        #: `/compact` requests this fake was asked for, and the answer it gives.
        #: A fake carries no history, so the honest default is the refusal a
        #: real session returns for an empty conversation; tests that want a
        #: successful pass stage their own outcome.
        self.compactions = 0
        self.compact_outcome = CompactionOutcome(
            ran=False,
            reason="nothing_to_compact",
            detail="nothing to compact: the whole conversation is ~0 tokens",
        )
        self.preflight_calls = 0
        self.preflight_notice: str | None = None

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

    @property
    def model(self) -> Any:
        return None

    def set_model(self, model: Any) -> None:
        pass

    @property
    def goal(self) -> str:
        return getattr(self, "_goal", "")

    def set_goal(self, text: str) -> str:
        self._goal = (text or "").strip()
        return self._goal

    async def seed_history(self, messages: list[Any]) -> None:
        pass

    async def preflight_usage(self) -> None:
        self.preflight_calls += 1
        if self.preflight_notice is not None:
            self.emit(NoticeEvent(text=self.preflight_notice, kind="warning"))

    async def prompt(self, text: str, images: Sequence[ImageContent] | None = None) -> None:
        self.prompts.append(text)

    def steer(self, text: str, images: Sequence[ImageContent] | None = None) -> None:
        pass

    def set_approval_handler(self, handler: object | None) -> None:
        # The TUI installs its own approval gate on boot (the stdin gate
        # deadlocks under a full-screen app); fakes only need to accept it.
        self.approval_handler = handler

    def abort(self, reason: str = "interrupted") -> None:
        self.aborts.append(reason)

    def subscribe(self, handler: Any) -> Any:
        self._handlers.append(handler)

        def unsubscribe() -> None:
            if handler in self._handlers:
                self._handlers.remove(handler)

        return unsubscribe

    @property
    def conversation_name(self) -> str:
        return getattr(self, "_name", "")

    def set_conversation_name(self, text: str, *, user_set: bool = True) -> str:
        self._name = (text or "").strip()
        return self._name

    async def complete_once(self, system: str, prompt: str) -> str:
        # No title: the naming worker must be inert in the pilot tests, and
        # an empty completion is exactly the "model said nothing usable"
        # path that generate_title resolves to None.
        self.completions.append((system, prompt))
        return ""

    async def dispose(self) -> None:
        self.disposed = True

    async def complete_aside(
        self,
        turns: list[Any],
        *,
        on_delta: Callable[[str], None] | None = None,
        on_usage: Callable[[Any], None] | None = None,
    ) -> str:
        # Recorded, not answered. The aside's own suites drive the real
        # Session; here the only contract that matters is that the app can
        # call it, so a fake that returned prose would invite pilot tests to
        # assert on words no model produced.
        self.asides.append(list(turns))
        return ""

    async def adopt_aside(self, messages: list[Any]) -> None:
        self.adopted.append(list(messages))

    async def compact_now(self) -> CompactionOutcome:
        return self._answer_compaction()

    def _answer_compaction(self) -> CompactionOutcome:
        """Count the request and answer with the staged outcome.

        Split out so subclasses can override :meth:`compact_now` (to raise, or
        to block) without losing the count every test reads.
        """
        self.compactions += 1
        return self.compact_outcome

    def history(self) -> list[Any]:
        return getattr(self, "_history", [])

    def emit(self, event: Any) -> None:
        for handler in list(self._handlers):
            handler(event)


async def _factory(session: FakeSession) -> FakeSession:
    return session


def _renderable_plain(renderable) -> str:
    """Recursively flatten a Rich renderable (incl. Group/Padding) to text."""
    from rich.console import Group
    from rich.padding import Padding
    from rich.text import Text

    if isinstance(renderable, Text):
        return renderable.plain
    if isinstance(renderable, Group):
        return "\n".join(_renderable_plain(child) for child in renderable.renderables)
    if isinstance(renderable, Padding):
        return _renderable_plain(renderable.renderable)
    if isinstance(renderable, str):
        return renderable
    return ""


def _transcript_text(app) -> str:
    transcript = app.query_one(TranscriptView)
    parts = []
    for b in transcript.blocks():
        parts.append(_renderable_plain(getattr(b, "renderable", "")))
    return "\n".join(parts)


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
        assert session.preflight_calls == 1
        # A user block was appended for the submitted prompt (the boot hint
        # is lifted by the first real block, D9).
        transcript = app.query_one(TranscriptView)
        assert len(transcript.blocks()) == 1


@pytest.mark.asyncio
async def test_boot_renders_non_blocking_quota_warning() -> None:
    session = FakeSession()
    session.preflight_notice = (
        "anthropic quota exhausted — falling back to openai/gpt-5.3-codex (high effort)"
    )
    app = OperatorApp(lambda: _factory(session))

    async with app.run_test(size=(100, 30)) as pilot:
        await pilot.pause()
        await pilot.pause()

        text = _transcript_text(app)
        assert "anthropic quota exhausted" in text
        assert "falling back to openai/gpt-5.3-codex (high effort)" in text
        assert session.prompts == []


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
async def test_model_opens_the_picker_instead_of_reporting_a_label() -> None:
    """``/model`` opens the model list, and still never prompts.

    Typing ``/model`` and pressing Enter goes through the command picker, whose
    completion adds the terminating space — and that space is exactly the handover
    that opens the model list. So the keystrokes a user actually presses reach the
    catalogue without ``/model`` ever being submitted as a command.

    Reporting the current label here was a dead end: the status band already shows
    it, while "which models could I switch to" had no way to be asked at all.
    """
    session = FakeSession()
    app = OperatorApp(lambda: _factory(session))
    async with app.run_test(size=(80, 24)) as pilot:
        await pilot.pause()
        editor = app.query_one(Editor)
        editor.focus()
        await pilot.pause()
        await pilot.press("slash", "m", "o", "d", "e", "l", "enter")
        await pilot.pause()
        assert editor.text == "/model ", editor.text
        assert editor.model_picker.is_open()
        # NOTHING was submitted: completing a command whose argument drives its
        # own list is not running it, so there is no echoed UserBlock and no
        # notice — just the list.
        assert app.query_one(TranscriptView).blocks() == []
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
    """Tab completes /he -> /help (trailing space = the argument slot) and
    focus stays on the editor (TUI-013)."""
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
        assert editor.text == "/help "
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
        # And the splash SURVIVES it. A session that never constructed is the most
        # infrastructure-y report in the app, and the worst moment to lose the one
        # block that says what to do next: the credential warning and the boot
        # hints both live there. Retiring the empty state here left a single red
        # line over an empty screen.
        assert app.query_one(WelcomeView).display is True
        assert app.screen.has_class(BOOT_LAYOUT_CLASS)
        # /reload re-runs boot and succeeds this time.
        app.query_one(Editor).focus()
        await pilot.pause()
        await pilot.press("slash", "r", "e", "l", "o", "a", "d", "enter")
        await pilot.pause()
        await pilot.pause()
        assert attempts["n"] == 2
        assert app._session is session


# --- login/logout through the provider controller --------------------------


@pytest.mark.asyncio
async def test_login_lists_providers_from_the_controller() -> None:
    """Bare /login lists loginable providers — the controller is the only
    path now that the CLI login_handler seam is gone.

    Esc before Enter is what makes it BARE: completing `/login` now opens the
    provider list instead, so dismissing that list is the only way left to run
    the command with no argument.
    """
    app = OperatorApp(lambda: _factory(FakeSession()), provider_controller=FakeProviderController())
    async with app.run_test(size=(80, 24)) as pilot:
        await pilot.pause()
        app.query_one(Editor).focus()
        await pilot.pause()
        await pilot.press("slash", "l", "o", "g", "i", "n", "escape", "enter")
        await pilot.pause()
        text = _transcript_text(app)
    assert "openrouter" in text and "deepseek" in text


@pytest.mark.asyncio
async def test_logout_routes_to_the_controller() -> None:
    controller = FakeProviderController()
    app = OperatorApp(lambda: _factory(FakeSession()), provider_controller=controller)
    async with app.run_test(size=(80, 24)) as pilot:
        await pilot.pause()
        app.query_one(Editor).focus()
        await pilot.pause()
        for ch in "/logout openrouter":
            await pilot.press("slash" if ch == "/" else ("space" if ch == " " else ch))
        await pilot.press("enter")
        await pilot.pause()
        await pilot.pause()
    assert controller.logouts == ["openrouter"]


def _provider_rows(app) -> list[tuple[str, str]]:
    """``(id, detail)`` for every row the provider list is offering."""
    rows: list[tuple[str, str]] = []
    for name, choice in app.query_one(Editor).picker.suggestions():
        assert isinstance(choice, ArgumentChoice), "the picker is not in argument mode"
        rows.append((name, choice.detail))
    return rows


@pytest.mark.asyncio
async def test_login_list_reports_all_three_credential_states() -> None:
    """`/login ` offers every loginable provider with where the user stands.

    Three states, not two: an env key runs a turn but is not a login, so calling
    it one would promise a stored account `/logout` could remove.
    """

    class EnvKeyController(FakeProviderController):
        def is_usable(self, provider):
            # deepseek has DEEPSEEK_API_KEY in the environment but no stored login.
            return self.has_any_credential(provider) or provider == "deepseek"

    app = OperatorApp(lambda: _factory(FakeSession()), provider_controller=EnvKeyController())
    async with app.run_test(size=(100, 30)) as pilot:
        await pilot.pause()
        app.query_one(Editor).focus()
        await pilot.pause()
        app.query_one(Editor).text = "/login "
        await pilot.pause()
        assert _provider_rows(app) == [
            ("openrouter", "logged in"),
            ("deepseek", "env key"),
            ("xai-oauth", "needs login"),
        ]


@pytest.mark.asyncio
async def test_a_search_alias_reaches_the_provider_it_names() -> None:
    """`grok` is how users refer to xAI; the row still completes to the id."""
    app = OperatorApp(lambda: _factory(FakeSession()), provider_controller=FakeProviderController())
    async with app.run_test(size=(100, 30)) as pilot:
        await pilot.pause()
        app.query_one(Editor).focus()
        await pilot.pause()
        app.query_one(Editor).text = "/login grok"
        await pilot.pause()
        assert [name for name, _ in _provider_rows(app)] == ["xai-oauth"]


@pytest.mark.asyncio
async def test_logout_rows_name_the_credential_they_will_remove() -> None:
    """`/logout` offers only what can be removed, and each row says what goes.

    Not "logged in": this list is FILTERED to providers holding a credential, so
    that state is true of every row by construction — a column with no bits in
    it, holding cells the description needs at narrow widths. The kind is what
    differs between rows and what the keystroke destroys.
    """
    app = OperatorApp(lambda: _factory(FakeSession()), provider_controller=FakeProviderController())
    async with app.run_test(size=(100, 30)) as pilot:
        await pilot.pause()
        app.query_one(Editor).focus()
        await pilot.pause()
        app.query_one(Editor).text = "/logout "
        await pilot.pause()
        assert _provider_rows(app) == [("openrouter", "remove api key")]


@pytest.mark.asyncio
async def test_logout_with_nothing_stored_says_so_where_the_list_would_have_been() -> None:
    """An empty set and "nothing matched your query" render identically — as
    nothing at all. Only the first is worth a sentence, because no amount of
    retyping would have produced a row.

    It is said in the PICKER, not the transcript. The sentence answers a UI event,
    so a transcript line repeats on every re-entry into the argument state (see the
    test below) — and a transcript is a record, while an empty list is a transient
    state of the input. The row is dim and unselectable: `is_open()` stays False, so
    Enter still submits the buffer and no click can action a sentence.
    """

    class NoCredentials(FakeProviderController):
        def credentials(self):
            return []

    app = OperatorApp(lambda: _factory(FakeSession()), provider_controller=NoCredentials())
    async with app.run_test(size=(100, 30)) as pilot:
        await pilot.pause()
        app.query_one(Editor).focus()
        await pilot.pause()
        app.query_one(Editor).text = "/logout "
        await pilot.pause()
        picker = app.query_one(Editor).picker
        assert not picker.is_open(), "a sentence is not a suggestion"
        assert picker.display is True, "and it is on screen anyway"
        assert "log out of" in picker.render_text(60).plain
        assert picker._index_at(0) is None, "the row cannot be clicked into a command"
        assert "log out of" not in _transcript_text(app), "the transcript is a record"
        # And it did NOT end the empty state. Opening a list is not the
        # conversation starting: a fresh session that collapsed its boot
        # composition to report that a command the user has not run yet has
        # nothing to offer would spend the whole empty state on that sentence.
        assert app.query_one(WelcomeView).display is True
        assert app.screen.has_class(BOOT_LAYOUT_CLASS)


@pytest.mark.asyncio
async def test_re_entering_an_empty_argument_state_leaves_the_transcript_alone() -> None:
    """Ten re-entries, zero transcript blocks.

    The sentence is raised from a UI event, so every route back into the argument
    state raises it again: typing `/logout `, backspacing, typing the space. As a
    transcript notice that stacked four identical rows in as many keystrokes, each
    one also taking a row off the splash that shares the region. In the picker the
    tenth re-entry looks exactly like the first.
    """

    class NoCredentials(FakeProviderController):
        def credentials(self):
            return []

    app = OperatorApp(lambda: _factory(FakeSession()), provider_controller=NoCredentials())
    async with app.run_test(size=(96, 28)) as pilot:
        await pilot.pause()
        editor = app.query_one(Editor)
        editor.focus()
        await pilot.pause()
        transcript = app.query_one(TranscriptView)
        for _ in range(10):
            editor.text = "/logout"
            await pilot.pause()
            editor.text = "/logout "
            await pilot.pause()
            await pilot.pause()
            assert transcript.blocks() == []
        assert "log out of" in editor.picker.render_text(60).plain
        # One row, however many times it was set: the picker holds a string, not a
        # list it appends to.
        assert editor.picker.styles.height is not None
        assert editor.picker.region.height == 1


@pytest.mark.asyncio
async def test_an_unreadable_credential_store_says_that_instead() -> None:
    """ "You have no credentials" and "I cannot tell" are different answers, and the
    informational row carries whichever one is true."""

    class RaisingStore(FakeProviderController):
        def credentials(self):
            raise RuntimeError("database is locked")

    app = OperatorApp(lambda: _factory(FakeSession()), provider_controller=RaisingStore())
    async with app.run_test(size=(100, 30)) as pilot:
        await pilot.pause()
        app.query_one(Editor).focus()
        await pilot.pause()
        app.query_one(Editor).text = "/logout "
        await pilot.pause()
        rendered = app.query_one(Editor).picker.render_text(80).plain
        assert "unreadable" in rendered
        assert "no stored credentials" not in rendered


@pytest.mark.asyncio
async def test_choosing_a_row_runs_the_existing_logout_path() -> None:
    """The list is a way to reach `_cmd_logout`, not a second implementation.

    Two Enters, even with a single row: on a destructive list "there is only one
    match" is not evidence that the user meant it — an empty query matches
    everything, and here everything happens to be one credential. The first
    Enter names it in the buffer, the second removes it.
    """
    controller = FakeProviderController()
    app = OperatorApp(lambda: _factory(FakeSession()), provider_controller=controller)
    async with app.run_test(size=(100, 30)) as pilot:
        await pilot.pause()
        app.query_one(Editor).focus()
        await pilot.pause()
        app.query_one(Editor).text = "/logout "
        await pilot.pause()
        assert len(app.query_one(Editor).picker.suggestions()) == 1, "premise: one match"
        await pilot.press("enter")
        await pilot.pause()
        assert controller.logouts == [], "an unnamed row is completed, not removed"
        assert app.query_one(Editor).text == "/logout openrouter"
        await pilot.press("enter")
        await pilot.pause()
        await pilot.pause()
    assert controller.logouts == ["openrouter"]


@pytest.mark.asyncio
async def test_choosing_a_row_runs_the_existing_login_path() -> None:
    controller = FakeProviderController()
    app = OperatorApp(lambda: _factory(FakeSession()), provider_controller=controller)
    async with app.run_test(size=(100, 30)) as pilot:
        await pilot.pause()
        app.query_one(Editor).focus()
        await pilot.pause()
        app.query_one(Editor).text = "/login deepseek"
        await pilot.pause()
        await pilot.press("enter")
        await pilot.pause()
        await pilot.pause()
    assert controller.logins == ["deepseek"]


@pytest.mark.asyncio
async def test_login_still_lists_every_provider_when_the_store_cannot_be_read() -> None:
    """An unreadable credential store costs the STATE column, not the app.

    The handler runs on a keystroke, so an exception out of it takes the whole
    TUI down — and the moment the store is unreadable is exactly the moment a
    user reaches for `/login`. The catalogue comes from the in-memory registry
    and is still entirely answerable, so every provider is still offered; only
    the state is blank, because a blank claims nothing and any of the three
    states would claim something the app cannot know.
    """
    app = OperatorApp(lambda: _factory(FakeSession()), provider_controller=RaisingStoreController())
    async with app.run_test(size=(100, 30)) as pilot:
        await pilot.pause()
        app.query_one(Editor).focus()
        await pilot.pause()
        app.query_one(Editor).text = "/login "
        await pilot.pause()
        assert app.is_running, "a locked credential store must not take the app down"
        assert _provider_rows(app) == [
            ("openrouter", ""),
            ("deepseek", ""),
            ("xai-oauth", ""),
        ]


@pytest.mark.asyncio
async def test_logout_says_the_store_is_unreadable_rather_than_claiming_it_is_empty() -> None:
    """`/logout` asks a question only the store can answer, so there is no
    degraded list — but "you have no credentials" is a different answer from "I
    cannot tell", and only one of them is true when the file is locked."""
    app = OperatorApp(lambda: _factory(FakeSession()), provider_controller=RaisingStoreController())
    async with app.run_test(size=(100, 30)) as pilot:
        await pilot.pause()
        app.query_one(Editor).focus()
        await pilot.pause()
        app.query_one(Editor).text = "/logout "
        await pilot.pause()
        assert app.is_running
        assert _provider_rows(app) == []
        picker = app.query_one(Editor).picker
        rendered = picker.render_text(60).plain
        assert "store unreadable" in rendered, rendered
        assert "no stored credentials" not in rendered


@pytest.mark.asyncio
async def test_logout_offers_one_row_per_credential_not_per_provider() -> None:
    """`openai` and `openai-device` share one stored credential, so logging out
    of either removes the same account. Two rows for one outcome is a choice the
    user cannot make correctly — while `/login` must still offer both, because
    they are two different ways to sign in."""
    app = OperatorApp(
        lambda: _factory(FakeSession()), provider_controller=CollidingStorageController()
    )
    async with app.run_test(size=(100, 30)) as pilot:
        await pilot.pause()
        app.query_one(Editor).focus()
        await pilot.pause()
        app.query_one(Editor).text = "/logout "
        await pilot.pause()
        assert _provider_rows(app) == [("openai", "remove oauth")]

        app.query_one(Editor).text = "/login "
        await pilot.pause()
        assert [name for name, _ in _provider_rows(app)] == ["openai", "openai-device"]


@pytest.mark.asyncio
async def test_a_provider_row_describes_what_its_id_does_not_already_say() -> None:
    """The registry name restated the id on twelve rows out of twelve.

    `openai / OpenAI (ChatGPT Plus/Pro)` spent half the description column
    re-spelling the id in title case and parenthesised the only part that
    distinguishes the row. That parenthetical is also the ONLY thing telling
    `openai` from `openai-device` and `xai` from `xai-oauth` apart, so it is
    what makes those four near-duplicates choosable. Where the name says nothing
    the id does not, the cell is blank — that is the honest answer.
    """
    app = OperatorApp(lambda: _factory(FakeSession()), provider_controller=RealRegistryController())
    async with app.run_test(size=(100, 30)) as pilot:
        await pilot.pause()
        app.query_one(Editor).focus()
        await pilot.pause()
        app.query_one(Editor).text = "/login "
        await pilot.pause()
        described = {
            name: choice.description for name, choice in app.query_one(Editor).picker.suggestions()
        }
    assert described["openai"] == "ChatGPT Plus/Pro"
    assert described["openai-device"] == "ChatGPT device code"
    assert described["xai"] == "Grok API key"
    assert described["xai-oauth"] == "Grok OAuth"
    assert described["anthropic"] == "Claude Pro/Max"
    assert described["deepseek"] == ""
    assert described["openrouter"] == ""
    assert described["radient"] == ""


@pytest.mark.asyncio
async def test_a_provider_query_the_user_typed_over_is_not_answered() -> None:
    """The opening message is one message-loop tick old, and a tick is enough to
    abandon the command. Answering it anyway attached a notice — and a list — to
    a command that no longer exists in the buffer."""

    class NoCredentials(FakeProviderController):
        def has_any_credential(self, provider):
            return False

        def credentials(self):
            return []

    app = OperatorApp(lambda: _factory(FakeSession()), provider_controller=NoCredentials())
    async with app.run_test(size=(100, 30)) as pilot:
        await pilot.pause()
        app.query_one(Editor).focus()
        await pilot.pause()
        editor = app.query_one(Editor)
        editor.text = "/logout "
        editor.text = "how do I write a parser?"
        await pilot.pause()
        await pilot.pause()
        picker = editor.picker
        assert not picker.is_open()
        assert picker.display is False, "nothing is said about an abandoned command"
        assert "log out" not in _transcript_text(app)


@pytest.mark.asyncio
async def test_login_without_controller_points_at_the_cli() -> None:
    """Degrading to a pointer notice is the contract when the TUI is embedded
    without a controller — it must never crash or silently do nothing.

    The two routes degrade to different PLACES, because they answer different
    things. Opening the list is a UI event that repeats on every re-entry into the
    argument state, so it says why the list is empty inside the list. Running the
    bare command is something the user did once, so it lands in the transcript,
    which is the record of what they did.
    """
    app = OperatorApp(lambda: _factory(FakeSession()))  # no controller
    async with app.run_test(size=(80, 24)) as pilot:
        await pilot.pause()
        app.query_one(Editor).focus()
        await pilot.pause()
        await pilot.press("slash", "l", "o", "g", "i", "n", "enter")  # opens the list
        await pilot.pause()
        picker = app.query_one(Editor).picker
        assert not picker.is_open(), "no controller, no rows"
        assert "local-operator login" in picker.render_text(70).plain
        assert "local-operator login" not in _transcript_text(app)

        # No Esc needed: with no rows the list never opened, so Enter on the
        # completed `/login ` goes straight to the command dispatch — and THAT is
        # a command the user ran, so it is recorded.
        await pilot.press("enter")
        await pilot.pause()
        assert _transcript_text(app).count("local-operator login") == 1


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


def test_completion_takes_the_top_match_when_ambiguous() -> None:
    """With several matches the picker highlights the top-ranked one and Tab
    applies it — registration order breaking ties, same ranking the scoring
    produces (there is no more "refuse when ambiguous" completion)."""
    editor = Editor(commands=[SlashCommand("help"), SlashCommand("history"), SlashCommand("exit")])
    editor.text = "/h"  # help and history tie at 900; registry order wins
    assert editor.picker.highlighted_name() == "help"
    editor.text = "/hello"  # nothing matches: the picker closes
    assert not editor.picker.is_open()


def test_completion_matches_alias() -> None:
    """TUI-014: the collapsed exit/quit command still completes via alias —
    the alias wins the ranking slot, so the inserted word is the alias."""
    editor = Editor(commands=[SlashCommand("exit", "Quit", aliases=("quit",))])
    editor.text = "/q"
    assert editor.picker.highlighted_name() == "quit"
    editor.picker.choose(0)  # the mouse path: select row 0 and complete
    assert editor.text == "/quit "


# --- provider-controller slash commands -----------------------------------


class FakeModel:
    def __init__(self, provider: str, model_id: str) -> None:
        self.provider = provider
        self.model_id = model_id


class FakeProviderController:
    """Minimal stand-in for ProviderController (sync + immediate fetches)."""

    def __init__(self) -> None:
        self.set_model_calls: list[Any] = []
        self.usage_reports: list[Any] = []
        # Every `fetch_usage` argument list, so a test can prove the panel's
        # refresh key actually re-fetches rather than repainting stale numbers.
        self.usage_calls: list[Any] = []
        self.usage_error: Exception | None = None
        self.logins: list[str] = []
        self.logouts: list[str] = []

    def login_providers(self) -> list[Any]:
        return [
            _FakeDef("openrouter", "OpenRouter", None, ("router",)),
            _FakeDef("deepseek", "DeepSeek", None, ("ds",)),
            _FakeDef("xai-oauth", "xAI OAuth", "xai", ("grok",)),
        ]

    def provider(self, pid):
        for d in self.login_providers():
            if d.id == pid:
                return d
        return None

    def is_usable(self, provider):
        # An env key counts, so this is wider than has_any_credential. The fake
        # answers both because the app asks the narrow one for "logged in" and the
        # wide one for "would a turn work".
        return self.has_any_credential(provider)

    def usable_providers(self) -> set[str] | None:
        # The set shape the picker's filter asks for: one answer for the whole
        # registry instead of one probe per provider. `None` would mean the store
        # could not be read at all, which this fake never simulates.
        return {d.id for d in self.login_providers() if self.is_usable(d.id)}

    def static_catalogue(self):
        from local_operator.providers.controller import CatalogueEntry

        return [
            CatalogueEntry(
                provider="openrouter",
                model_id="deepseek/deepseek-chat",
                label="DeepSeek Chat",
                context_window=64_000,
                input_price=0.14,
                output_price=0.28,
                connected=True,
                aggregated=True,
            )
        ]

    async def live_catalogue(self, *, ttl_s=None):
        return self.static_catalogue(), {"openrouter": "ok"}

    def has_any_credential(self, provider):
        return provider in ("openrouter",)

    def credentials(self):
        return [
            _FakeCred(1, "openrouter", "api_key", {"source": "login"}),
            _FakeCred(2, "deepseek", "oauth", {"expires": 9999999999999, "email": "a@b.c"}),
        ]

    def usage_enabled_providers(self):
        return ["openrouter", "zai"]

    def usage_reportable_providers(self):
        # The real controller narrows "has an endpoint" by "and a credential that
        # can reach it"; `/provider` renders this one, not the wider list.
        return [p for p in self.usage_enabled_providers() if self.has_any_credential(p)]

    def resolve_model(self, provider, model_id):
        return FakeModel(provider, model_id)

    def set_login_callbacks(self, factory):
        # The TUI installs its own transcript-rendering callbacks before every
        # flow; without this the flow dies on an AttributeError the app catches
        # and reports as "login failed", which looks exactly like a real failure.
        self.login_callbacks = factory

    async def login(self, provider):
        self.logins.append(provider)
        return f"logged in {provider}"

    async def logout(self, provider):
        self.logouts.append(provider)
        return f"removed {provider}"

    async def fetch_usage(self, provider_ids=None):
        self.usage_calls.append(provider_ids)
        if self.usage_error is not None:
            raise self.usage_error
        return self.usage_reports


class _FakeDef:
    def __init__(self, pid, name, store_as, aliases=()):
        self.id = pid
        self.name = name
        self.store_credentials_as = store_as
        self.login = object()  # truthy -> has interactive login
        # Mirrors ProviderDefinition.search_aliases: the other names a user would
        # type for this provider, which is what makes `grok` reach `xai-oauth`.
        self.search_aliases = aliases


class _FakeCred:
    def __init__(self, ident, provider, ctype, data):
        self.id = ident
        self.provider = provider
        self.credential_type = ctype
        self.data = data
        self.identity_key = None


class RaisingStoreController(FakeProviderController):
    """A controller whose credential store cannot be read.

    `database is locked` is one other local-operator process away, and every
    read below goes to the same SQLite file.
    """

    def has_any_credential(self, provider):
        raise RuntimeError("database is locked")

    def is_usable(self, provider):
        raise RuntimeError("database is locked")

    def credentials(self):
        raise RuntimeError("database is locked")


class CollidingStorageController(FakeProviderController):
    """Two providers that file their credential under ONE storage id.

    The real registry has two such pairs (openai/openai-device and
    xai/xai-oauth); the default fake had a `store_credentials_as` provider with
    nothing to collide with, so the dedupe branch never ran.
    """

    def login_providers(self):
        return [
            _FakeDef("openai", "OpenAI (ChatGPT Plus/Pro)", None, ("gpt",)),
            _FakeDef("openai-device", "OpenAI (ChatGPT device code)", "openai", ("gpt",)),
        ]

    def has_any_credential(self, provider):
        return provider in ("openai", "openai-device")

    def credentials(self):
        return [_FakeCred(1, "openai", "oauth", {})]


class RealRegistryController(FakeProviderController):
    """The fake controller over the REAL provider registry.

    The descriptions are derived from registry names, so the only test that can
    fail when a name changes is one that reads the actual registry.
    """

    def login_providers(self):
        from local_operator.providers.registry import list_login_providers

        return list_login_providers()

    def has_any_credential(self, provider):
        return False

    def is_usable(self, provider):
        return False

    def credentials(self):
        return []


@pytest.mark.asyncio
async def test_model_switch_calls_session_set_model() -> None:
    session = FakeSession()
    set_models: list[Any] = []

    def set_model(spec):
        set_models.append(spec)

    session.set_model = set_model  # type: ignore[attr-defined]
    ctrl = FakeProviderController()
    app = OperatorApp(lambda: _factory(session), provider_controller=ctrl)
    async with app.run_test(size=(80, 24)) as pilot:
        await pilot.pause()
        app.query_one(Editor).focus()
        # /model openrouter/deepseek/deepseek-chat
        for key in "s", "p", "a", "c", "e":
            pass
        await pilot.press(
            "slash",
            "m",
            "o",
            "d",
            "e",
            "l",
            "space",
            "o",
            "p",
            "e",
            "n",
            "r",
            "o",
            "u",
            "t",
            "e",
            "r",
            "slash",
            "d",
            "e",
            "e",
            "p",
            "s",
            "e",
            "e",
            "k",
            "slash",
            "d",
            "e",
            "e",
            "p",
            "s",
            "e",
            "e",
            "k",
            "-",
            "c",
            "h",
            "a",
            "t",
            "enter",
        )
        await pilot.pause()
    assert len(set_models) == 1
    assert set_models[0].provider == "openrouter"


@pytest.mark.asyncio
async def test_provider_command_renders_listing() -> None:
    session = FakeSession()
    ctrl = FakeProviderController()
    app = OperatorApp(lambda: _factory(session), provider_controller=ctrl)
    async with app.run_test(size=(80, 24)) as pilot:
        await pilot.pause()
        app.query_one(Editor).focus()
        await pilot.press("slash")
        for key in "p", "r", "o", "v", "i", "d", "e", "r":
            await pilot.press(key)
        await pilot.press("enter")
        await pilot.pause()
        texts = _transcript_text(app)
    assert "openrouter" in texts
    assert "OpenRouter" in texts
    assert session.prompts == []


@pytest.mark.asyncio
async def test_accounts_command_renders_credentials() -> None:
    session = FakeSession()
    ctrl = FakeProviderController()
    app = OperatorApp(lambda: _factory(session), provider_controller=ctrl)
    async with app.run_test(size=(80, 24)) as pilot:
        await pilot.pause()
        app.query_one(Editor).focus()
        await pilot.press("slash")
        for key in "a", "c", "c", "o", "u", "n", "t", "s":
            await pilot.press(key)
        await pilot.press("enter")
        await pilot.pause()
        texts = _transcript_text(app)
    assert "openrouter" in texts
    assert "api_key (login)" in texts


def _usage_reports(*, used: float = 5.0):
    from local_operator.providers.usage import UsageAmount, UsageLimit, UsageReport

    return [
        UsageReport(
            provider="openrouter",
            limits=[
                UsageLimit(
                    id="openrouter:credits",
                    label="Credits",
                    amount=UsageAmount(used=used, limit=50.0, unit="usd"),
                )
            ],
        )
    ]


class _ControlledUsageController(FakeProviderController):
    """Keeps the first network request pending so ordering is observable."""

    def __init__(self) -> None:
        super().__init__()
        self.first_release = asyncio.Event()
        self.first_started = asyncio.Event()
        self.first_cancelled = False

    async def fetch_usage(self, provider_ids=None):
        self.usage_calls.append(provider_ids)
        if len(self.usage_calls) == 1:
            self.first_started.set()
            try:
                await self.first_release.wait()
            except asyncio.CancelledError:
                self.first_cancelled = True
                raise
            return _usage_reports(used=5.0)
        return _usage_reports(used=42.0)


async def _run_usage_command(pilot, app) -> None:
    """Type `/usage` and submit it, the way a user reaches the panel."""
    app.query_one(Editor).focus()
    await pilot.press("slash")
    for key in "u", "s", "a", "g", "e":
        await pilot.press(key)
    await pilot.press("enter")
    for _ in range(4):
        await pilot.pause()


@pytest.mark.asyncio
async def test_usage_command_opens_the_panel_with_the_report() -> None:
    """`/usage` opens the popup rather than appending to the transcript.

    A quota report is reference material: appended as a block it was pushed off
    screen by the next turn and could not be re-read without re-fetching.
    """
    from local_operator.tui.widgets.usage_panel import UsagePanel

    session = FakeSession()
    ctrl = FakeProviderController()
    ctrl.usage_reports = _usage_reports()
    app = OperatorApp(lambda: _factory(session), provider_controller=ctrl)
    async with app.run_test(size=(80, 24)) as pilot:
        await pilot.pause()
        await _run_usage_command(pilot, app)
        panel = app.query_one(UsagePanel)
        assert panel.is_open
        text = "\n".join(panel.render_lines_for_test())
        assert app.focused is panel
    assert "openrouter" in text
    assert "Credits" in text


@pytest.mark.asyncio
async def test_escape_closes_the_usage_panel_and_returns_focus() -> None:
    """The panel takes focus to receive its keys, so it must hand focus back —
    otherwise dismissing it leaves the user typing into nothing."""
    from local_operator.tui.widgets.usage_panel import UsagePanel

    ctrl = FakeProviderController()
    ctrl.usage_reports = _usage_reports()
    app = OperatorApp(lambda: _factory(FakeSession()), provider_controller=ctrl)
    async with app.run_test(size=(80, 24)) as pilot:
        await pilot.pause()
        await _run_usage_command(pilot, app)
        panel = app.query_one(UsagePanel)
        assert panel.is_open
        await pilot.press("escape")
        for _ in range(3):
            await pilot.pause()
        assert not panel.is_open
        assert isinstance(app.focused, Editor)


@pytest.mark.asyncio
async def test_dismissed_usage_request_cannot_reopen_the_panel() -> None:
    """Esc closes the request as well as its card; a late network response must
    not reverse an explicit user action."""
    from local_operator.tui.widgets.usage_panel import UsagePanel

    ctrl = _ControlledUsageController()
    app = OperatorApp(lambda: _factory(FakeSession()), provider_controller=ctrl)
    async with app.run_test(size=(80, 24)) as pilot:
        await pilot.pause()
        await _run_usage_command(pilot, app)
        panel = app.query_one(UsagePanel)
        await ctrl.first_started.wait()
        await pilot.press("escape")
        for _ in range(4):
            await pilot.pause()
        ctrl.first_release.set()
        for _ in range(4):
            await pilot.pause()

        assert ctrl.first_cancelled
        assert not panel.is_open


@pytest.mark.asyncio
async def test_usage_result_ready_between_close_and_dismiss_handler_is_ignored() -> None:
    """Closing invalidates the request synchronously, before the app can receive
    the dismissal message and cancel its worker."""
    from local_operator.tui.widgets.usage_panel import UsagePanel

    ctrl = _ControlledUsageController()
    app = OperatorApp(lambda: _factory(FakeSession()), provider_controller=ctrl)
    async with app.run_test(size=(80, 24)) as pilot:
        await pilot.pause()
        await _run_usage_command(pilot, app)
        panel = app.query_one(UsagePanel)
        await ctrl.first_started.wait()

        # This is the exact cross-queue gap inside `action_dismiss`: the panel is
        # already closed, while the app has not yet handled `UsageDismissed`.
        panel.close()
        ctrl.first_release.set()
        for _ in range(4):
            await pilot.pause()

        assert not ctrl.first_cancelled
        assert not panel.is_open


@pytest.mark.asyncio
async def test_usage_refresh_supersedes_a_slower_request() -> None:
    """A stale first response must not overwrite the report returned by the
    refresh that replaced it."""
    from local_operator.tui.widgets.usage_panel import UsagePanel

    ctrl = _ControlledUsageController()
    app = OperatorApp(lambda: _factory(FakeSession()), provider_controller=ctrl)
    async with app.run_test(size=(80, 24)) as pilot:
        await pilot.pause()
        await _run_usage_command(pilot, app)
        panel = app.query_one(UsagePanel)
        await ctrl.first_started.wait()
        await pilot.press("r")
        for _ in range(6):
            await pilot.pause()
        ctrl.first_release.set()
        for _ in range(4):
            await pilot.pause()
        text = "\n".join(panel.render_lines_for_test())

        assert ctrl.first_cancelled
        assert len(ctrl.usage_calls) == 2
        assert "84%" in text
        assert "10%" not in text


@pytest.mark.asyncio
async def test_r_refetches_without_closing_the_panel() -> None:
    """Refresh is the whole reason the panel holds focus: the numbers go stale
    while they are being read, and re-typing the command to see new ones would
    make the panel worse than the transcript block it replaced."""
    from local_operator.tui.widgets.usage_panel import UsagePanel

    ctrl = FakeProviderController()
    ctrl.usage_reports = _usage_reports()
    app = OperatorApp(lambda: _factory(FakeSession()), provider_controller=ctrl)
    async with app.run_test(size=(80, 24)) as pilot:
        await pilot.pause()
        await _run_usage_command(pilot, app)
        panel = app.query_one(UsagePanel)
        before = len(ctrl.usage_calls)
        await pilot.press("r")
        for _ in range(4):
            await pilot.pause()
        assert len(ctrl.usage_calls) == before + 1
        assert panel.is_open


@pytest.mark.asyncio
async def test_a_failed_fetch_is_reported_inside_the_panel() -> None:
    """The panel is what has focus and what carries the key that retries, so an
    error anywhere else asks the user to look away from the fix."""
    from local_operator.tui.widgets.usage_panel import UsagePanel

    ctrl = FakeProviderController()
    ctrl.usage_error = RuntimeError("network is down")
    app = OperatorApp(lambda: _factory(FakeSession()), provider_controller=ctrl)
    async with app.run_test(size=(80, 24)) as pilot:
        await pilot.pause()
        await _run_usage_command(pilot, app)
        panel = app.query_one(UsagePanel)
        text = "\n".join(panel.render_lines_for_test())
        assert panel.is_open
    assert "network is down" in text
    assert "r refresh" in text


@pytest.mark.asyncio
async def test_all_three_usage_surfaces_agree_for_an_api_key_only_install(monkeypatch) -> None:
    """`/provider`'s "report quota" list, bare `/usage`'s targets and
    `/usage <provider>`'s up-front warning are three surfaces answering one
    question, and they used to give three answers: with only `ANTHROPIC_API_KEY`
    set, `/provider` advertised anthropic, bare `/usage` rendered "no usage data",
    and `/usage anthropic` correctly said it needs a login."""
    from local_operator.providers.controller import (
        ControllerAuthStore,
        ProviderController,
    )
    from tests.unit.providers.test_controller import _USAGE_ENV_VARS, FakeAuthStore

    for name in _USAGE_ENV_VARS:
        monkeypatch.delenv(name, raising=False)
    monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-ant-test")
    controller = ProviderController(
        cast(ControllerAuthStore, FakeAuthStore()), login_callbacks=None
    )
    app = OperatorApp(lambda: _factory(FakeSession()), provider_controller=controller)

    # Surface 1: `/provider` must not advertise what `/usage` cannot deliver.
    assert app._provider_usage_state() == []
    # Surface 2: the bare `/usage` target list is the same list.
    assert controller.usage_reportable_providers() == []
    # Surface 3: `/usage anthropic` refuses up front, with the actionable reason.
    #
    # Run under `run_test` rather than with an injected notice callback: the
    # REFUSING branches go through `_system_notice`, so they keep the boot
    # composition a rejected command never earned the right to collapse (see
    # `_cmd_usage`), and that method writes to the transcript rather than to the
    # `notice` parameter.
    async with app.run_test(size=(100, 30)) as pilot:
        await pilot.pause()
        app._cmd_usage("anthropic", app._notice)
        await pilot.pause()
        notices = [
            (block._text, block._token)
            for block in app.query_one(TranscriptView).blocks()
            if isinstance(block, NoticeBlock)
        ]
        welcome_kept = app.query_one(WelcomeView).display
    assert notices == [("anthropic reports usage only after /login anthropic", "warning")]
    assert welcome_kept is True


@pytest.mark.asyncio
async def test_provider_and_the_login_list_report_credentials_in_the_same_words() -> None:
    """Two surfaces, one question, one vocabulary.

    `/provider` rendered a provider with no credential as `—` while the `/login`
    picker called the same provider `needs login`. A dash is not an answer: a
    user with no credential reads a dash and cannot tell "none", "unknown" and
    "not supported" apart.
    """
    app = OperatorApp(lambda: _factory(FakeSession()), provider_controller=FakeProviderController())
    async with app.run_test(size=(100, 30)) as pilot:
        await pilot.pause()
        app.query_one(Editor).focus()
        await pilot.pause()
        app.query_one(Editor).text = "/provider"
        await pilot.press("enter")
        await pilot.pause()
        await pilot.pause()
        listed = [
            line
            for line in _transcript_text(app).split("\n")
            if "deepseek" in line or "xai-oauth" in line
        ]
        app.query_one(Editor).text = "/login "
        await pilot.pause()
        states = dict(_provider_rows(app))

    assert listed, "premise: /provider rendered its listing"
    assert all("needs login" in line for line in listed), listed
    assert all("—" not in line for line in listed), listed
    assert states["deepseek"] == "needs login", "and the picker says the same thing"


@pytest.mark.asyncio
async def test_run_tui_forwards_provider_controller(monkeypatch) -> None:
    """F3 regression: run_tui must pass provider_controller to OperatorApp so
    the slash-command surface is live (not a pointer, not a crash)."""
    import local_operator.tui.app as app_mod
    from local_operator.tui.app import OperatorApp

    seen: dict[str, Any] = {}
    fake_controller = object()

    class _SpyApp(OperatorApp):
        def __init__(self, *a, **kw):
            seen["controller"] = kw.get("provider_controller")
            super().__init__(*a, **kw)

        async def run_async(self, **kwargs: Any) -> None:
            return None

    # run_tui lazy-imports OperatorApp from local_operator.tui.app at call
    # time, so patching that module attribute is what routes the spy in.
    monkeypatch.setattr(app_mod, "OperatorApp", _SpyApp)
    called = []

    async def factory():
        called.append(1)
        return _SpyApp()  # type: ignore[return-value]

    # Await run_tui with a fake session factory that must not await forever.
    async def factory2() -> Any:
        return object()

    from local_operator.tui import run_tui

    await run_tui(factory2, theme_name="dark", provider_controller=fake_controller)
    assert seen["controller"] is fake_controller


# --- /goal and /loop -------------------------------------------------------


class GoalSession(FakeSession):
    """FakeSession with the goal surface and a recording prompt()."""

    def __init__(self) -> None:
        super().__init__()
        self._goal = ""
        self.fail_on_prompt = False

    @property
    def goal(self) -> str:
        return self._goal

    def set_goal(self, text: str) -> str:
        self._goal = (text or "").strip()
        return self._goal

    async def prompt(self, text: str, images: Sequence[ImageContent] | None = None) -> None:
        if self.fail_on_prompt:
            raise RuntimeError("boom")
        self.prompts.append(text)


async def _type_command(pilot, app, command: str) -> None:
    app.query_one(Editor).focus()
    await pilot.press("slash")
    for ch in command:
        await pilot.press("space" if ch == " " else ch)
    await pilot.press("enter")
    await pilot.pause()


@pytest.mark.asyncio
async def test_goal_set_show_and_clear() -> None:
    session = GoalSession()
    app = OperatorApp(lambda: _factory(session))
    async with app.run_test(size=(80, 24)) as pilot:
        await pilot.pause()
        await _type_command(pilot, app, "goal ship it")
        assert session.goal == "ship it"
        await _type_command(pilot, app, "goal")
        assert "ship it" in _transcript_text(app)
        await _type_command(pilot, app, "goal clear")
        assert session.goal == ""


@pytest.mark.asyncio
async def test_loop_requires_a_goal() -> None:
    session = GoalSession()
    app = OperatorApp(lambda: _factory(session))
    async with app.run_test(size=(80, 24)) as pilot:
        await pilot.pause()
        await _type_command(pilot, app, "loop")
        assert "set a goal first" in _transcript_text(app)
    assert session.prompts == []


@pytest.mark.asyncio
async def test_loop_runs_bounded_iterations() -> None:
    session = GoalSession()
    session.set_goal("finish the parser")
    app = OperatorApp(lambda: _factory(session))
    async with app.run_test(size=(80, 24)) as pilot:
        await pilot.pause()
        await _type_command(pilot, app, "loop 2")
        for _ in range(12):
            await pilot.pause()
            if not app._loop_running:
                break
        text = _transcript_text(app)
    assert len(session.prompts) == 2
    assert "loop finished after 2" in text


@pytest.mark.asyncio
async def test_loop_rejects_out_of_range_and_garbage() -> None:
    session = GoalSession()
    session.set_goal("g")
    app = OperatorApp(lambda: _factory(session))
    async with app.run_test(size=(80, 24)) as pilot:
        await pilot.pause()
        await _type_command(pilot, app, "loop 99")
        await _type_command(pilot, app, "loop abc")
        text = _transcript_text(app)
    assert "between 1 and" in text
    assert "usage: /loop" in text
    assert session.prompts == []


@pytest.mark.asyncio
async def test_loop_stops_on_turn_error() -> None:
    session = GoalSession()
    session.set_goal("g")
    session.fail_on_prompt = True
    app = OperatorApp(lambda: _factory(session))
    async with app.run_test(size=(80, 24)) as pilot:
        await pilot.pause()
        await _type_command(pilot, app, "loop 5")
        for _ in range(12):
            await pilot.pause()
            if not app._loop_running:
                break
        text = _transcript_text(app)
    assert "loop stopped" in text  # did not spin through all 5


@pytest.mark.asyncio
async def test_interrupt_cancels_running_loop() -> None:
    session = GoalSession()
    session.set_goal("g")
    app = OperatorApp(lambda: _factory(session))
    async with app.run_test(size=(80, 24)) as pilot:
        await pilot.pause()
        app._loop_running = True
        app.action_interrupt()
        assert app._loop_cancelled is True


# -- MCP status band + startup toast -----------------------------------------


class FakeMcpManager:
    """The five methods the app asks of a manager, plus a way to drive a drop."""

    def __init__(self, configured: list[str], connected: list[str]) -> None:
        self._configured = list(configured)
        self._connected = list(connected)
        self._callback: Any = None
        self.inner_calls: list[list[Any]] = []

    def get_all_server_names(self) -> list[str]:
        return sorted(self._configured)

    def get_connected_servers(self) -> list[str]:
        return sorted(self._connected)

    def get_connection_status(self, name: str) -> str:
        return "connected" if name in self._connected else "disconnected"

    def set_on_tools_changed(self, callback: Any) -> None:
        self._callback = callback

    @property
    def on_tools_changed(self) -> Any:
        return self._callback

    def install_incumbent(self) -> None:
        """Stand in for the composition root's own subscriber, which the app
        must chain rather than clobber."""
        self._callback = self.inner_calls.append

    def fire(self) -> None:
        """What ``set_on_tools_changed`` does on connect/disconnect."""
        assert self._callback is not None
        self._callback([])

    def drop(self, name: str) -> None:
        self._connected.remove(name)
        self.fire()


class McpSession(FakeSession):
    """A session carrying the two attributes the composition root records."""

    def __init__(self, manager: Any = None, startup: Any = None) -> None:
        super().__init__()
        self.mcp_manager = manager
        self.mcp_startup = startup


def _band(app) -> str:  # type: ignore[no-untyped-def]
    from textual.widgets import Static

    return app.query_one("#status-band", Static).render().plain


@pytest.mark.asyncio
async def test_no_mcp_means_no_segment_and_no_toast() -> None:
    """The whole feature is invisible on a machine that does not use MCP. A
    ``⊙ 0 MCP`` and a "0 servers" toast on every launch would be pure noise."""
    session = McpSession(manager=None, startup=McpStartupOutcome())
    app = OperatorApp(lambda: _factory(session))
    async with app.run_test(size=(100, 24)) as pilot:
        for _ in range(6):
            await pilot.pause()
        assert "MCP" not in _band(app)
        assert app.query_one(Toast).display is False


@pytest.mark.asyncio
async def test_the_band_counts_connected_servers_and_the_toast_reports_startup() -> None:
    manager = FakeMcpManager(["github", "linear", "slack"], ["github", "linear"])
    startup = McpStartupOutcome(
        configured=("github", "linear", "slack"),
        connected=("github", "linear"),
        failures={"slack": "command not found: slack-mcp"},
        tool_count=31,
    )
    session = McpSession(manager=manager, startup=startup)
    app = OperatorApp(lambda: _factory(session))
    async with app.run_test(size=(100, 24)) as pilot:
        for _ in range(6):
            await pilot.pause()
        assert "⊙ 2 MCP" in _band(app)
        toast = app.query_one(Toast)
        assert toast.display is True
        assert "2 of 3 servers up, 31 tools" in toast.message
        assert "slack" in toast.message


@pytest.mark.asyncio
async def test_a_server_dropping_updates_the_count_live() -> None:
    """The reference snapshots this count at boot and lets it go stale. Here
    ``set_on_tools_changed`` drives a repaint, so a server dying is visible."""
    manager = FakeMcpManager(["github", "linear"], ["github", "linear"])
    session = McpSession(manager=manager, startup=McpStartupOutcome())
    app = OperatorApp(lambda: _factory(session))
    async with app.run_test(size=(100, 24)) as pilot:
        for _ in range(6):
            await pilot.pause()
        assert "⊙ 2 MCP" in _band(app)
        manager.drop("linear")
        for _ in range(4):
            await pilot.pause()
        band = _band(app)
        assert "⊙ 1 MCP" in band
        # …and the surviving server's neighbour being down turns the lamp: the
        # count alone cannot say whether 1 of 2 is a failure or a config change.
        assert app._mcp_status().failed is True


@pytest.mark.asyncio
async def test_the_app_chains_the_composition_roots_subscriber() -> None:
    """Clobbering the incumbent callback would freeze the agent's TOOL LIST at
    boot — a far worse bug than a stale counter. The app reads the incumbent and
    calls it from its own wrapper."""
    manager = FakeMcpManager(["github"], ["github"])
    manager.install_incumbent()
    session = McpSession(manager=manager, startup=McpStartupOutcome())
    app = OperatorApp(lambda: _factory(session))
    async with app.run_test(size=(100, 24)) as pilot:
        for _ in range(6):
            await pilot.pause()
        manager.fire()
        for _ in range(4):
            await pilot.pause()
        assert manager.inner_calls, "the incumbent tool-merge callback was dropped"


@pytest.mark.asyncio
async def test_a_failure_survives_the_toast_dismissing() -> None:
    """A toast that dismisses is not a record. The failure lands in the
    transcript as a notice AND is reachable through ``/mcp``, so the information
    outlives the five or ten seconds the overlay is up."""
    manager = FakeMcpManager(["slack"], [])
    startup = McpStartupOutcome(
        configured=("slack",),
        failures={"slack": "command not found: slack-mcp"},
    )
    session = McpSession(manager=manager, startup=startup)
    app = OperatorApp(lambda: _factory(session))
    async with app.run_test(size=(100, 24)) as pilot:
        for _ in range(6):
            await pilot.pause()
        toast = app.query_one(Toast)
        assert toast.display is True
        toast.dismiss_toast()
        await pilot.pause()
        assert toast.display is False
        assert toast.message == ""
        # The durable half — appended, but WITHOUT ending the empty state: the
        # conversation has not started just because a server failed to start, and
        # collapsing the boot composition on launch would mean a user with one
        # broken server never saw the centred prompt the toast interrupted.
        text = _transcript_text(app)
        assert "MCP slack failed: command not found: slack-mcp" in text
        welcome = app.query_one(WelcomeView)
        assert welcome.display is True, "an infrastructure notice must not retire the splash"


@pytest.mark.asyncio
async def test_mcp_command_reports_per_server_state_not_just_the_config() -> None:
    """``/mcp`` used to dump the configured command and never say whether it
    worked, which is the only question it gets run to answer."""
    from local_operator.mcp.config import MCPStdioServerConfig

    manager = FakeMcpManager(["slack"], [])
    startup = McpStartupOutcome(
        configured=("slack",),
        failures={"slack": "command not found: slack-mcp"},
    )
    session = McpSession(manager=manager, startup=startup)
    app = OperatorApp(lambda: _factory(session))
    async with app.run_test(size=(100, 24)) as pilot:
        for _ in range(6):
            await pilot.pause()
        app.query_one(Toast).dismiss_toast()
        block = None
        with patch(
            "local_operator.mcp.config.load_all_mcp_configs",
            return_value=({"slack": MCPStdioServerConfig(command="slack-mcp")}, {}),
        ):
            block = app._mcp_block()
        assert block is not None
        listing = _renderable_plain(block.renderable)
        assert "slack" in listing
        assert "disconnected" in listing
        assert "command not found: slack-mcp" in listing


@pytest.mark.asyncio
async def test_mcp_command_puts_the_status_in_a_column() -> None:
    """Crammed into the detail string the status landed wherever the name ended,
    so the shorter name pushed the longer status LEFT and the two facts a reader
    scans for formed no column. Both fields are padded to their widest."""
    from local_operator.mcp.config import MCPStdioServerConfig

    manager = FakeMcpManager(["github", "gh"], ["github"])
    startup = McpStartupOutcome(
        configured=("github", "gh"),
        connected=("github",),
        failures={"gh": "command not found: gh"},
    )
    session = McpSession(manager=manager, startup=startup)
    app = OperatorApp(lambda: _factory(session))
    async with app.run_test(size=(100, 24)) as pilot:
        for _ in range(6):
            await pilot.pause()
        app.query_one(Toast).dismiss_toast()
        configs = {
            "github": MCPStdioServerConfig(command="npx -y server-github"),
            "gh": MCPStdioServerConfig(command="gh mcp serve"),
        }
        with patch("local_operator.mcp.config.load_all_mcp_configs", return_value=(configs, {})):
            block = app._mcp_block()
        assert block is not None
        # Branch rows only: the listing now leads with a dim caption naming what
        # it lists (the block is its own receipt, so it has to say what it is),
        # and this test is about the SERVER rows' column alignment.
        rows = [
            row
            for row in _renderable_plain(block.renderable).split("\n")
            if row.lstrip().startswith(("├─", "└─"))
        ]
        assert len(rows) == 2
        # The caption is what makes the block its own receipt now that `/mcp`
        # no longer echoes; the filter above discards it, so it is pinned here.
        assert _renderable_plain(block.renderable).splitlines()[0] == "MCP servers"
        # `connected` is a substring of `disconnected`, so each row is located by
        # its own status word and the two start columns compared directly. Before
        # the fix the SHORTER name pushed the LONGER status four cells left.
        connected_at = next(row.index("connected") for row in rows if "disconnected" not in row)
        disconnected_at = next(row.index("disconnected") for row in rows if "disconnected" in row)
        assert connected_at == disconnected_at, rows
        # The detail after the status column lines up too, or the padding only
        # moved the ragged edge one field to the right.
        assert rows[0].index("npx") == rows[1].index("command not found"), rows


@pytest.mark.asyncio
async def test_a_discovery_failure_keeps_an_alarm_in_the_band() -> None:
    """Discovery raising leaves no manager and no server list, so the band used
    to render exactly like a machine that never configured MCP — while the toast
    saying otherwise dismissed itself ten seconds later."""
    startup = McpStartupOutcome(failures={"discovery": "config unreadable"})
    session = McpSession(manager=None, startup=startup)
    app = OperatorApp(lambda: _factory(session))
    async with app.run_test(size=(100, 24)) as pilot:
        for _ in range(6):
            await pilot.pause()
        band = _band(app)
        assert "⊙ MCP" in band, band
        # No count: the config layer never produced one, so any number is a lie.
        assert "0 MCP" not in band
        app.query_one(Toast).dismiss_toast()
        await pilot.pause()
        assert "⊙ MCP" in _band(app)


@pytest.mark.asyncio
async def test_the_band_refreshes_even_when_the_incumbent_callback_raises() -> None:
    """The chained wrapper calls the composition root's subscriber first, and
    ``McpManager._fire_tools_changed`` swallows and logs whatever comes out of
    it — so a raising incumbent used to leave the band asserting a count that was
    no longer true, which is the exact staleness the live segment exists to
    remove. The repaint is scheduled in a ``finally``."""
    manager = FakeMcpManager(["github", "linear"], ["github", "linear"])

    def exploding_incumbent(tools: list[Any]) -> None:
        raise RuntimeError("refresh_tools blew up")

    manager.set_on_tools_changed(exploding_incumbent)
    session = McpSession(manager=manager, startup=McpStartupOutcome())
    app = OperatorApp(lambda: _factory(session))
    async with app.run_test(size=(100, 24)) as pilot:
        for _ in range(6):
            await pilot.pause()
        assert "⊙ 2 MCP" in _band(app)
        manager._connected.remove("linear")
        # The manager swallows the incumbent's exception; the app must not lose
        # the repaint with it.
        with pytest.raises(RuntimeError):
            manager.fire()
        for _ in range(4):
            await pilot.pause()
        assert "⊙ 1 MCP" in _band(app)


@pytest.mark.asyncio
async def test_a_toast_erases_no_transcript_row_outside_its_own_columns() -> None:
    """A/B on the COMPOSITED frame, which is the only place this was visible.

    The toast host was ``width: 1fr``: a widget owns its whole region and Textual
    blanks all of it, so a 35-cell card on a 96-cell screen erased the other 59
    cells of every row it covered — the transcript row read
    `· line 0 ABCDEFGHIJ…` out to column 90 before the toast and nothing at all
    after it. The layer keeps the toast out of the LAYOUT; it does not keep it
    off the screen.
    """
    session = McpSession(manager=None, startup=McpStartupOutcome())
    app = OperatorApp(lambda: _factory(session))
    async with app.run_test(size=(96, 28)) as pilot:
        for _ in range(4):
            await pilot.pause()
        transcript = app.query_one(TranscriptView)
        # ONE tall block rather than many: scrolled to the bottom, a block taller
        # than the viewport fills every row of it with text, including the row the
        # card lands on. Separate blocks put their adaptive gap row there instead,
        # which would make this A/B pass by having nothing to lose.
        transcript.append_block(
            NoticeBlock(
                "\n".join(f"line {index} " + "ABCDEFGHIJ" * 8 for index in range(40)), "info"
            )
        )
        app._set_welcome_visible(False)
        for _ in range(4):
            await pilot.pause()

        def rows() -> list[str]:
            return [
                "".join(segment.text for segment in strip)
                for strip in app.screen._compositor.render_strips()
            ]

        before = rows()
        toast = app.query_one(Toast)
        toast.show(
            "⊙ MCP: 1 of 2 servers up, 9 tools\nfailed: b — spawn ENOENT", duration_ms=60_000
        )
        for _ in range(4):
            await pilot.pause()
        after = rows()

        card = toast.region
        assert card.height == 2, card
        # NOT vacuous: at least one row the card covers carries transcript text to
        # the left of it, which is exactly what the full-width host used to wipe.
        # Only SOME of them — the transcript's own top padding row is blank by
        # design and the card starts on it.
        assert any(
            before[y][: card.x].strip() for y in range(card.y, card.bottom)
        ), f"nothing to lose on rows {card.y}..{card.bottom - 1}"
        # The DAMAGE: every column outside the card is byte-identical.
        for y in range(card.y, card.bottom):
            for column in range(len(before[y])):
                if card.x <= column < card.right:
                    continue
                assert before[y][column] == after[y][column], (
                    f"row {y} column {column} was erased outside the card "
                    f"{card}: {before[y]!r} -> {after[y]!r}"
                )
        # …then the mechanism that guarantees it, and proof the card really is
        # showing there, so this is not a vacuous pass.
        assert app.query_one("#toast-host").region == card
        assert "MCP: 1 of 2 servers up" in after[card.y]


# --- model access: what the picker offers, what a switch confirms ------------


class _AccessController(FakeProviderController):
    """A catalogue spanning the three credential situations the picker must tell
    apart: a stored credential, none at all, and a local server that needs none.

    ``store_error`` is the store failing to answer — SQLite locked, file gone —
    which the real controller reports as ``None`` from ``usable_providers`` rather
    than as an empty set, because "I cannot tell" and "you have nothing" are
    different answers.
    """

    def __init__(
        self, stored: tuple[str, ...] = ("openrouter",), store_error: bool = False
    ) -> None:
        super().__init__()
        self.stored = set(stored)
        self.store_error = store_error

    def login_providers(self):
        return [
            _FakeDef("openrouter", "OpenRouter", None, ("router",)),
            _FakeDef("anthropic", "Anthropic", None, ("claude",)),
            _FakeDef("ollama", "Ollama", None, ()),
        ]

    def has_any_credential(self, provider):
        if self.store_error:
            raise RuntimeError("credential store locked")
        return provider in self.stored

    def is_usable(self, provider):
        # Ollama stands in for `allows_missing_api_key`: a local server runs with
        # no credential at all, so a filter keyed on credentials alone would hide
        # the one provider that always works.
        return provider == "ollama" or self.has_any_credential(provider)

    def usable_providers(self):
        if self.store_error:
            return None
        return {d.id for d in self.login_providers() if self.is_usable(d.id)}

    def static_catalogue(self):
        from local_operator.providers.controller import CatalogueEntry

        usable = self.usable_providers()
        return [
            CatalogueEntry(
                provider=provider,
                model_id=model_id,
                label=model_id,
                context_window=200_000,
                input_price=3.0,
                output_price=15.0,
                connected=usable is None or provider in usable,
                aggregated=provider == "openrouter",
            )
            for provider, model_id in (
                ("openrouter", "deepseek/deepseek-chat"),
                ("anthropic", "claude-opus-5"),
                ("ollama", "qwen3:8b"),
            )
        ]

    async def live_catalogue(self, *, ttl_s=None):
        return self.static_catalogue(), {}


async def _open_model_picker(app, pilot):
    """Type ``/model `` the way a user does and settle the live refresh."""
    editor = app.query_one(Editor)
    editor.focus()
    await pilot.pause()
    await pilot.press("slash", "m", "o", "d", "e", "l", "space")
    # Twice: the first frame is the static catalogue, the second is the worker's
    # live one. Asserting on the first would test the wrong list.
    await pilot.pause()
    await pilot.pause()
    return editor.model_picker


@pytest.mark.asyncio
async def test_the_model_list_offers_what_the_user_can_actually_run() -> None:
    """The list is a set of choices. A row whose only outcome is a login prompt is
    not one, and it costs a line of a fourteen-row window."""
    ctrl = _AccessController()
    app = OperatorApp(lambda: _factory(FakeSession()), provider_controller=ctrl)
    async with app.run_test(size=(90, 24)) as pilot:
        await pilot.pause()
        picker = await _open_model_picker(app, pilot)
        offered = {row.selector for row in picker.rows()}
    # openrouter has a credential; ollama needs none by definition; anthropic has
    # neither and is the one the user cannot act on.
    assert offered == {"openrouter/deepseek/deepseek-chat", "ollama/qwen3:8b"}, offered


@pytest.mark.asyncio
async def test_the_hidden_models_are_counted_with_the_command_that_reveals_them() -> None:
    """Discoverability was the whole argument for the old show-everything list.
    The footer chrome keeps it without crowding the persistence instruction."""
    ctrl = _AccessController()
    app = OperatorApp(lambda: _factory(FakeSession()), provider_controller=ctrl)
    async with app.run_test(size=(90, 24)) as pilot:
        await pilot.pause()
        picker = await _open_model_picker(app, pilot)
        chrome = picker.render_text(90).plain
    assert "1 hidden" in chrome, chrome
    assert "/login <provider>" in chrome, chrome


@pytest.mark.asyncio
async def test_an_unreadable_credential_store_shows_every_model_not_none() -> None:
    """An empty picker claims the user owns no models, which is exactly what the
    app failed to find out. Showing an unfiltered list is the recoverable error."""
    ctrl = _AccessController(store_error=True)
    app = OperatorApp(lambda: _factory(FakeSession()), provider_controller=ctrl)
    async with app.run_test(size=(90, 24)) as pilot:
        await pilot.pause()
        picker = await _open_model_picker(app, pilot)
        offered = {row.selector for row in picker.rows()}
        chrome = picker.render_text(90).plain
    assert "anthropic/claude-opus-5" in offered, offered
    assert len(offered) == 3, offered
    # …and it says so, rather than quietly presenting a list it could not filter.
    assert "credential check unavailable" in chrome, chrome


@pytest.mark.asyncio
async def test_a_new_credential_reaches_the_list_without_a_restart() -> None:
    """`/login anthropic` then `/model` is one continuous action. Rows built once
    at boot would make the user restart the app to see what they just unlocked."""
    ctrl = _AccessController()
    app = OperatorApp(lambda: _factory(FakeSession()), provider_controller=ctrl)
    async with app.run_test(size=(90, 24)) as pilot:
        await pilot.pause()
        picker = await _open_model_picker(app, pilot)
        assert not any(row.provider == "anthropic" for row in picker.rows())

        ctrl.stored.add("anthropic")  # what a completed /login stores
        editor = app.query_one(Editor)
        editor.text = ""  # close the list; the next `/model` rebuilds it
        await pilot.pause()
        picker = await _open_model_picker(app, pilot)
        offered = {row.selector for row in picker.rows()}
    assert "anthropic/claude-opus-5" in offered, offered


class _SwitchableSession(FakeSession):
    """A session whose label follows ``set_model``, as the real one's does — the
    confirmation names the model, so a frozen label would not test it."""

    def __init__(self) -> None:
        super().__init__()
        self._label = "openrouter/deepseek/deepseek-chat"

    @property
    def model_label(self) -> str:
        return self._label

    def set_model(self, model) -> None:
        self._label = f"{model.provider}/{model.model_id}"


@pytest.mark.asyncio
async def test_switching_confirms_access_instead_of_warning_about_it() -> None:
    """The old line told the user to go and check something the app knew, on every
    provider change including the ones that were fine."""
    session = _SwitchableSession()
    ctrl = _AccessController(stored=("openrouter", "anthropic"))
    app = OperatorApp(lambda: _factory(session), provider_controller=ctrl)
    async with app.run_test(size=(90, 24)) as pilot:
        await pilot.pause()
        app._run_slash_command("/model anthropic/claude-opus-5")
        await pilot.pause()
        text = _transcript_text(app)
    assert "anthropic/claude-opus-5" in text, text
    assert "anthropic logged in" in text, text
    assert "make sure you are logged in" not in text, text


@pytest.mark.asyncio
async def test_switching_without_a_credential_names_the_one_fix() -> None:
    """ "needs login" is the same word `/provider` and the `/login` picker use, and
    the command is the entire remedy — no second surface to go and consult."""
    session = _SwitchableSession()
    ctrl = _AccessController()
    app = OperatorApp(lambda: _factory(session), provider_controller=ctrl)
    async with app.run_test(size=(90, 24)) as pilot:
        await pilot.pause()
        app._run_slash_command("/model anthropic/claude-opus-5")
        await pilot.pause()
        text = _transcript_text(app)
    assert "anthropic needs login — /login anthropic" in text, text
    assert "make sure you are logged in" not in text, text


@pytest.mark.asyncio
async def test_a_hidden_model_is_still_reachable_by_typing_its_selector() -> None:
    """Filtering the LIST is not a lock on the command. A user who knows the id —
    from `/provider`, from docs, from the model they used yesterday — types it and
    gets the switch plus the one thing they are missing, not a refusal."""
    session = _SwitchableSession()
    ctrl = _AccessController()
    app = OperatorApp(lambda: _factory(session), provider_controller=ctrl)
    async with app.run_test(size=(90, 24)) as pilot:
        await pilot.pause()
        editor = app.query_one(Editor)
        editor.focus()
        editor.text = "/model anthropic/claude-opus-5"
        await pilot.pause()
        assert editor.model_picker.suggestions() == [], "premise: the row is hidden"
        await pilot.press("enter")
        await pilot.pause()
        text = _transcript_text(app)
    assert session.model_label == "anthropic/claude-opus-5"
    assert "anthropic needs login — /login anthropic" in text, text


@pytest.mark.asyncio
async def test_a_failed_credential_check_is_reported_as_itself() -> None:
    """Neither a confirmation the app cannot make nor the old blanket warning: the
    store is what broke, and naming it is what makes it fixable."""
    session = _SwitchableSession()
    ctrl = _AccessController(store_error=True)
    app = OperatorApp(lambda: _factory(session), provider_controller=ctrl)
    async with app.run_test(size=(90, 24)) as pilot:
        await pilot.pause()
        app._run_slash_command("/model anthropic/claude-opus-5")
        await pilot.pause()
        text = _transcript_text(app)
    assert "cannot check anthropic credentials: credential store locked" in text, text
    assert "logged in" not in text, text


@pytest.mark.asyncio
async def test_a_failing_turn_shows_the_providers_own_error() -> None:
    """The 400 that started this: a switch-time guess about logins was on screen
    while the real reason — a rejected parameter — came from the provider. Only
    the provider's own words tell the user what to change."""
    session = FakeSession()

    async def prompt(text, images=None):
        raise RuntimeError("HTTP 400: `temperature` is deprecated for this model.")

    session.prompt = prompt  # type: ignore[assignment]
    app = OperatorApp(lambda: _factory(session), provider_controller=_AccessController())
    async with app.run_test(size=(90, 24)) as pilot:
        await pilot.pause()
        app.query_one(Editor).focus()
        await pilot.press("h", "i", "enter")
        await pilot.pause()
        text = _transcript_text(app)
    assert "HTTP 400: `temperature` is deprecated for this model." in text, text


# --- /resume ---------------------------------------------------------------


def _resume_factory(
    boots: list[str | None],
    history_text: str = "resumed history",
    assistant_text: str = "resumed answer",
):
    """A resume factory that records the id it was asked to boot.

    ``None`` is a real value here, not a missing one: it is what ``/new`` asks
    for, and ``create_session`` reads it as "start a fresh conversation".
    """

    async def resume_factory(resume_id: str | None):
        boots.append(resume_id)
        session = FakeSession()
        session._history = [
            SimpleNamespace(role="user", text=history_text, tool_calls=None),
            # Production Message objects use an empty list, not None, for a
            # prose-only assistant reply.
            SimpleNamespace(role="assistant", text=assistant_text, tool_calls=[]),
        ]
        return session

    return resume_factory


def _seed_session(tmp_path: Path, session_id: str, prompt: str = "") -> None:
    """Lay down one resumable session transcript under a temp config dir.

    ``recent_sessions`` globs ``<config_dir>/sessions/*`` for transcripts, so
    a real file is the only honest way to make the listing and the resume
    both resolve — same convention the ``--resume`` CLI path trusts.

    ``prompt`` writes a real opening user message, which is what the picker
    names the row by; without one the session is legitimately nameless.
    """
    sess_dir = tmp_path / "sessions" / session_id
    sess_dir.mkdir(parents=True, exist_ok=True)
    body = ""
    if prompt:
        body = (
            json.dumps(
                {
                    "id": "e1",
                    "ts": 0,
                    "type": "message",
                    "payload": {
                        "kind": "message",
                        "role": "user",
                        "content": [{"text": prompt}],
                    },
                }
            )
            + "\n"
        )
    (sess_dir / "transcript.jsonl").write_text(body)


@pytest.mark.asyncio
async def test_a_bare_resume_opens_the_picker_naming_each_session(tmp_path, monkeypatch) -> None:
    """A bare ``/resume`` opens the picker instead of printing ids.

    The old behaviour dumped ``<hex id>  3h ago`` rows into the transcript,
    which pushed the conversation up, could not be navigated, and left the
    user to retype an id read off the scrollback. The picker is a two-way
    surface and names each row by its opening message.
    """
    monkeypatch.setenv("LOCAL_OPERATOR_CONFIG_DIR", str(tmp_path))
    _seed_session(tmp_path, "aabbcc", prompt="Make an asteroids game")

    session = FakeSession()
    boots: list[str | None] = []
    app = OperatorApp(
        lambda: _factory(session),
        resume_factory=_resume_factory(boots),
    )
    async with app.run_test(size=(90, 24)) as pilot:
        await pilot.pause()
        editor = app.query_one(Editor)
        editor.focus()
        await pilot.press("/", "r", "e", "s", "u", "m", "e", "enter")
        await pilot.pause()
        await pilot.pause()
        assert boots == [], "a bare /resume must not boot a session"
        picker = app.screen
        assert isinstance(picker, SessionPickerScreen)
        card = "\n".join(picker.render_lines_for_test())
        # Named by the opening message, not just listed by id.
        assert "Make an asteroids game" in card, card
        assert "aabbcc" in card, card


@pytest.mark.asyncio
async def test_choosing_in_the_picker_resumes_that_session(tmp_path, monkeypatch) -> None:
    """Enter on a row is what actually resumes it — the picker's whole job."""
    monkeypatch.setenv("LOCAL_OPERATOR_CONFIG_DIR", str(tmp_path))
    _seed_session(tmp_path, "aabbcc", prompt="the only session")

    session = FakeSession()
    boots: list[str | None] = []
    app = OperatorApp(
        lambda: _factory(session),
        resume_factory=_resume_factory(boots),
    )
    async with app.run_test(size=(90, 24)) as pilot:
        await pilot.pause()
        app.query_one(Editor).focus()
        await pilot.press("/", "r", "e", "s", "u", "m", "e", "enter")
        await pilot.pause()
        await pilot.press("enter")
        await pilot.pause()
        await pilot.pause()
    assert boots == ["aabbcc"], boots


@pytest.mark.asyncio
async def test_escaping_the_picker_resumes_nothing(tmp_path, monkeypatch) -> None:
    """A cancelled picker leaves the session on screen exactly as it was."""
    monkeypatch.setenv("LOCAL_OPERATOR_CONFIG_DIR", str(tmp_path))
    _seed_session(tmp_path, "aabbcc", prompt="the only session")

    session = FakeSession()
    boots: list[str | None] = []
    app = OperatorApp(
        lambda: _factory(session),
        resume_factory=_resume_factory(boots),
    )
    async with app.run_test(size=(90, 24)) as pilot:
        await pilot.pause()
        app.query_one(Editor).focus()
        await pilot.press("/", "r", "e", "s", "u", "m", "e", "enter")
        await pilot.pause()
        await pilot.press("escape")
        await pilot.pause()
        await pilot.pause()
    assert boots == [], boots


@pytest.mark.asyncio
async def test_resume_id_rebinds_and_reloads(tmp_path, monkeypatch) -> None:
    """``/resume <id>`` swaps the factory to that id and reboots the app."""
    monkeypatch.setenv("LOCAL_OPERATOR_CONFIG_DIR", str(tmp_path))
    _seed_session(tmp_path, "cafe01")

    session = FakeSession()
    boots: list[str | None] = []
    app = OperatorApp(
        lambda: _factory(session),
        resume_factory=_resume_factory(boots),
    )
    async with app.run_test(size=(80, 24)) as pilot:
        await pilot.pause()
        editor = app.query_one(Editor)
        editor.focus()
        await pilot.press(
            "/", "r", "e", "s", "u", "m", "e", " ", "c", "a", "f", "e", "0", "1", "enter"
        )
        await pilot.pause()
        await pilot.pause()
        assert boots == ["cafe01"], boots


@pytest.mark.asyncio
async def test_resume_replaces_the_visible_transcript(tmp_path, monkeypatch) -> None:
    """Switching sessions must show only the resumed conversation's history."""
    monkeypatch.setenv("LOCAL_OPERATOR_CONFIG_DIR", str(tmp_path))
    _seed_session(tmp_path, "cafe01")

    session = FakeSession()
    session._history = [
        SimpleNamespace(role="user", text="history from the current session", tool_calls=None)
    ]

    async def dispose_with_terminal_event() -> None:
        # A real session emits terminal events while dispose is awaited. The
        # old controller must already be detached or this notice is queued
        # behind the transcript replacement and lands in the resumed session.
        session.emit(NoticeEvent(text="stale event from disposed session", kind="warning"))
        session.disposed = True

    session.dispose = dispose_with_terminal_event  # type: ignore[method-assign]
    boots: list[str | None] = []
    app = OperatorApp(
        lambda: _factory(session),
        resume_factory=_resume_factory(boots, "history from the resumed session"),
    )
    async with app.run_test(size=(80, 24)) as pilot:
        await pilot.pause()
        editor = app.query_one(Editor)
        editor.focus()
        await pilot.press(
            "/", "r", "e", "s", "u", "m", "e", " ", "c", "a", "f", "e", "0", "1", "enter"
        )
        await pilot.pause()
        await pilot.pause()

        text = _transcript_text(app)
        assert boots == ["cafe01"], boots
        assert "history from the resumed session" in text, text
        assistant_texts = [
            block.text()
            for block in app.query_one(TranscriptView).blocks()
            if isinstance(block, AssistantBlock)
        ]
        assert assistant_texts == ["resumed answer"], assistant_texts
        assert "history from the current session" not in text, text
        assert "/resume cafe01" not in text, text
        assert "resuming session cafe01" not in text, text
        assert "stale event from disposed session" not in text, text


@pytest.mark.asyncio
async def test_a_session_swap_never_shows_the_splash(tmp_path, monkeypatch) -> None:
    """Conversation A goes straight to conversation B — no logo in between.

    The splash is the transcript's EMPTY STATE, and a swap clears the ledger
    before repopulating it, so for a while "the transcript has no content" was
    true of a screen that was only mid-substitution. It was true for the whole
    of the session factory too, because the clear used to happen BEFORE the
    await: the user watched conversation A, then the centred logo, then
    conversation B.

    Asserted at every frame Textual paints rather than at the end, which is the
    only way to catch a state that exists only in the middle: ``post_display_hook``
    is Textual's own after-a-frame callback. The factory is made slow on purpose,
    because a free one hides the very window this is about.
    """
    monkeypatch.setenv("LOCAL_OPERATOR_CONFIG_DIR", str(tmp_path))
    _seed_session(tmp_path, "swap01")

    current = FakeSession()
    current._history = [
        SimpleNamespace(role="user", text="the conversation being left", tool_calls=None)
    ]

    async def slow_resume_factory(_resume_id: str | None):
        await asyncio.sleep(0.05)  # a real factory is not free
        resumed = FakeSession()
        resumed._history = [
            SimpleNamespace(role="user", text="the conversation arrived at", tool_calls=None)
        ]
        return resumed

    app = OperatorApp(lambda: _factory(current), resume_factory=slow_resume_factory)
    splash_frames: list[bool] = []
    watching = False
    original_hook = type(app).post_display_hook

    def hook(self) -> None:
        original_hook(self)
        if watching:
            splash_frames.append(self.query_one(WelcomeView).display)

    monkeypatch.setattr(type(app), "post_display_hook", hook)

    async with app.run_test(size=(80, 24)) as pilot:
        await pilot.pause()
        assert app.query_one(TranscriptView).blocks(), "the session being left never rendered"
        assert app.query_one(WelcomeView).display is False

        watching = True
        # Exactly what `_resume_session` does before running the worker.
        app._session_factory = lambda: slow_resume_factory("swap01")  # type: ignore[assignment]
        await app._reload_session()
        for _ in range(6):
            await pilot.pause()
        watching = False

        assert splash_frames, "no frame was painted across the swap"
        assert not any(splash_frames), (
            f"the splash was painted on {sum(splash_frames)} of "
            f"{len(splash_frames)} frames across the swap"
        )
        assert "the conversation arrived at" in _transcript_text(app)


@pytest.mark.asyncio
async def test_a_swap_onto_an_empty_session_still_lands_on_the_splash(tmp_path) -> None:
    """The suppression is about the TRANSITION, not about the destination.

    ``/new`` swaps onto a conversation with no history, and that is a genuinely
    empty app: the splash is right for it. A fix that simply stopped the swap
    from ever showing the splash would leave the fresh session on a blank
    screen with no boot card and no hints.
    """
    current = FakeSession()
    current._history = [SimpleNamespace(role="user", text="something", tool_calls=None)]

    async def new_factory(_resume_id: str | None):
        return FakeSession()  # no history: this is what /new asks for

    app = OperatorApp(lambda: _factory(current), resume_factory=new_factory)
    async with app.run_test(size=(80, 24)) as pilot:
        await pilot.pause()
        assert app.query_one(WelcomeView).display is False

        app._session_factory = lambda: new_factory(None)  # type: ignore[assignment]
        await app._reload_session()
        await pilot.pause()

        assert app.query_one(WelcomeView).display is True
        assert app.screen.has_class(BOOT_LAYOUT_CLASS)
        assert app.query_one(TranscriptView).blocks() == []


@pytest.mark.asyncio
async def test_a_swap_leaves_the_ledger_matching_the_new_sessions_history() -> None:
    """The screen is a PROJECTION of the model's context, after the reorder too.

    The swap now builds the replacement session BEFORE clearing the ledger, so
    that the substitution paints in one frame. That reordering must not weaken
    the rule it moved around: what is on screen at the end is the new session's
    ``history()``, all of it and nothing else — including on the path where the
    replacement fails to construct, which leaves no conversation at all rather
    than the previous one under an error saying there is no session.
    """
    first = FakeSession()
    first._history = [
        SimpleNamespace(role="user", text="first-session prompt", tool_calls=None),
        SimpleNamespace(role="assistant", text="first-session answer", tool_calls=[]),
    ]
    second = FakeSession()
    second._history = [
        SimpleNamespace(role="user", text="second-session prompt", tool_calls=None),
        SimpleNamespace(role="assistant", text="second-session answer", tool_calls=[]),
        SimpleNamespace(role="user", text="second-session follow-up", tool_calls=None),
    ]

    app = OperatorApp(lambda: _factory(first))
    async with app.run_test(size=(80, 24)) as pilot:
        await pilot.pause()
        assert len(app.query_one(TranscriptView).blocks()) == 2

        app._session_factory = lambda: _factory(second)  # type: ignore[assignment]
        await app._reload_session()
        await pilot.pause()

        text = _transcript_text(app)
        assert len(app.query_one(TranscriptView).blocks()) == 3
        assert "first-session prompt" not in text, text
        assert "first-session answer" not in text, text
        for line in ("second-session prompt", "second-session answer", "second-session follow-up"):
            assert line in text, text

        # And the failure path clears too: a swap that lands on nothing must
        # not leave the previous conversation standing under the error.
        async def broken_factory():
            raise RuntimeError("factory exploded")

        app._session_factory = broken_factory  # type: ignore[assignment]
        await app._reload_session()
        await pilot.pause()

        after = _transcript_text(app)
        assert "second-session prompt" not in after, after
        assert "factory exploded" in after, after
        assert app.query_one(WelcomeView).display is True


@pytest.mark.asyncio
async def test_resume_long_history_opens_at_the_latest_turn(tmp_path, monkeypatch) -> None:
    """A replay batch settles with the resumed conversation's tail in view."""
    monkeypatch.setenv("LOCAL_OPERATOR_CONFIG_DIR", str(tmp_path))
    _seed_session(tmp_path, "long01")

    async def resume_factory(_resume_id: str | None):
        resumed = FakeSession()
        resumed._history = [
            SimpleNamespace(
                role="user",
                text=f"resumed history line {index}",
                tool_calls=None,
            )
            for index in range(30)
        ]
        return resumed

    app = OperatorApp(
        lambda: _factory(FakeSession()),
        resume_factory=resume_factory,
    )
    async with app.run_test(size=(60, 12)) as pilot:
        await pilot.pause()
        editor = app.query_one(Editor)
        editor.focus()
        for key in "/resume long01":
            await pilot.press(key if key != " " else "space")
        await pilot.press("enter")
        for _ in range(3):
            await pilot.pause()

        transcript = app.query_one(TranscriptView)
        assert len(transcript.blocks()) == 30
        max_scroll_y = max(0, transcript.virtual_size.height - transcript.size.height)
        assert transcript.scroll_offset.y == max_scroll_y


@pytest.mark.asyncio
async def test_resume_at_latest_passes_the_sentinel_verbatim(tmp_path, monkeypatch) -> None:
    """``/resume @latest`` must hand the factory the ``@latest`` symbol, not
    strip the ``@`` (C14-02). resume.py only resolves the newest session on an
    EXACT ``@latest`` match; a stripped ``latest`` falls through to a literal
    ``sessions/latest`` path and fails to boot."""
    monkeypatch.setenv("LOCAL_OPERATOR_CONFIG_DIR", str(tmp_path))
    _seed_session(tmp_path, "aabbcc")

    session = FakeSession()
    boots: list[str | None] = []
    app = OperatorApp(
        lambda: _factory(session),
        resume_factory=_resume_factory(boots),
    )
    async with app.run_test(size=(80, 24)) as pilot:
        await pilot.pause()
        editor = app.query_one(Editor)
        editor.focus()
        # Type ``/resume @latest``
        for key in "/resume @latest":
            await pilot.press(key if key != " " else "space")
        await pilot.press("enter")
        await pilot.pause()
        await pilot.pause()
        assert boots == ["@latest"], boots


@pytest.mark.asyncio
async def test_a_tall_usage_overlay_never_scrolls_the_screen_or_steals_width() -> None:
    """A floating overlay must leave the layout beneath it alone.

    It did not. The `/usage` card is sized by the widget and positioned by an
    offset on the `toast` layer; layers keep it out of the FLOW, but nothing
    kept it out of the SCROLLABLE REGION. A card taller than its resting
    offset allowed pushed the screen's virtual height past its size, so
    Textual drew a screen scrollbar — which the app has no use for (the
    transcript is the scrolling surface and the input is docked) and which
    cost two cells of width, reflowing the transcript behind a popup that is
    supposed to change nothing. `overflow: hidden` on `Screen` is the guard.
    """
    from local_operator.tui.widgets.usage_panel import UsagePanel
    from tests.unit.tui.test_usage_panel import _many_reports

    session = FakeSession()
    app = OperatorApp(lambda: _factory(session))
    async with app.run_test(size=(100, 24)) as pilot:
        await pilot.pause()
        width_before = app.screen.size.width
        panel = app.query_one(UsagePanel)
        panel.display = True
        # Deliberately taller than the terminal so the overlay MUST overflow.
        panel.show_reports(_many_reports())
        await pilot.pause()
        await pilot.pause()
        assert app.screen.show_vertical_scrollbar is False
        assert app.screen.size.width == width_before
        # And the card's own content box still gets every row it composed:
        # the pinned height carries the padding, it does not eat the footer.
        assert panel.size.height == len(panel.render_lines_for_test())


# --- the boot default: is it settable, and did it land -----------------------


def _unwrapped(text: str) -> str:
    """Text with all whitespace removed.

    The transcript wraps a notice to the widget's width, breaking both prose and
    a tmp-dir path across lines. Comparing what was WRITTEN with what was
    RENDERED has to ignore the wrap, or the assertion is really about whichever
    terminal size the test happened to pick.
    """
    return "".join(text.split())


@pytest.mark.asyncio
async def test_a_bare_model_names_the_current_pair_and_the_command_that_keeps_it() -> None:
    """The complaint this closes: nothing on the model surfaces said a default
    existed, so `/model default` was reachable only by already knowing the word.
    Both the notice above the list and the list's own footer now say it."""
    session = _SwitchableSession()
    ctrl = _AccessController()
    app = OperatorApp(lambda: _factory(session), provider_controller=ctrl)
    async with app.run_test(size=(90, 24)) as pilot:
        await pilot.pause()
        app._run_slash_command("/model")
        await pilot.pause()
        await pilot.pause()
        text = _transcript_text(app)
        picker = app.query_one(Editor).model_picker
        footer = picker.render_text(90).plain.split("\n")[-1]
    # The subject of the sentence: "make THIS the default" needs a this.
    assert "openrouter/deepseek/deepseek-chat" in _unwrapped(text), text
    assert _unwrapped(PERSIST_HINT) in _unwrapped(text), text
    # …and the list itself, which is the surface the user is actually reading.
    assert PERSIST_HINT in footer, footer


@pytest.mark.asyncio
async def test_rejected_model_commands_keep_the_boot_composition() -> None:
    """A typo changed no state and must not permanently collapse the splash."""
    session = _SwitchableSession()
    ctrl = _AccessController()
    app = OperatorApp(lambda: _factory(session), provider_controller=ctrl)
    async with app.run_test(size=(100, 30)) as pilot:
        await pilot.pause()
        welcome = app.query_one(WelcomeView)
        for command in ("/model missing-slash", "/model bogus-provider/model"):
            app._run_slash_command(command)
            await pilot.pause()
            assert welcome.display is True, command
            assert app.screen.has_class("boot"), command


@pytest.mark.asyncio
async def test_an_entered_rejected_model_command_keeps_the_boot_composition() -> None:
    """The SAME rule as the direct-call test above, now that the submit path
    agrees with it.

    This test used to assert the opposite — that entering the command dismissed
    the splash — because the submit handler echoed every slash command and
    retired the splash on its own. Two paths into one command answering "has the
    session started?" differently was the bug underneath: the echo, not the
    command, was deciding. With the echo on the registry (`SlashCommand.echo`)
    and the empty-state edge back on `_append_block`, a rejected selector is a
    rejected selector however it was typed.
    """
    session = _SwitchableSession()
    ctrl = _AccessController()
    app = OperatorApp(lambda: _factory(session), provider_controller=ctrl)
    async with app.run_test(size=(100, 30)) as pilot:
        await pilot.pause()
        editor = app.query_one(Editor)
        editor.text = "/model missing-slash"
        await pilot.press("enter")
        await pilot.pause()
        await pilot.pause()
        painted = "\n".join(strip.text for strip in app.screen._compositor.render_strips())
        welcome_display = app.query_one(WelcomeView).display
        boot = app.screen.has_class("boot")

    # The rejection is still ON SCREEN, under the surviving splash: keeping the
    # boot composition must not cost the user the answer to what went wrong.
    assert "usage: /model <provider>/<model-id>" in painted, painted
    assert welcome_display is True
    assert boot


def test_help_uses_one_column_wider_than_every_command_name() -> None:
    """The two longest aliases used to consume the literal 14-cell column and
    glue directly onto descriptions: `/model, /modelsSwitch model…`."""
    app = OperatorApp(lambda: _factory(FakeSession()))
    rows = _renderable_plain(app._help_block().renderable).splitlines()
    for command in SLASH_COMMANDS:
        names = ", ".join(f"/{name}" for name in command.names)
        row = next(row for row in rows if row.startswith(names))
        assert row[len(names) :].startswith("  "), row


def test_help_mentions_the_window_title_toggle() -> None:
    """The title lives outside the app's frame, so the help is where a user who
    notices it and wants it off can discover the toggle without source-diving."""
    app = OperatorApp(lambda: _factory(FakeSession()))
    text = _renderable_plain(app._help_block().renderable)
    assert "window title" in text
    assert "display.terminal_title" in text
    assert "LOCAL_OPERATOR_NO_TERMINAL_TITLE" in text


@pytest.mark.asyncio
async def test_a_switch_admits_it_is_session_only_and_names_the_persist_command() -> None:
    """A switch that looks permanent and is not is the actual bug: the old
    "(next turn)" said WHEN it applied and never said for how long, so the next
    launch coming back on the old model read as the switch having been lost."""
    session = _SwitchableSession()
    ctrl = _AccessController(stored=("openrouter", "anthropic"))
    app = OperatorApp(lambda: _factory(session), provider_controller=ctrl)
    async with app.run_test(size=(90, 24)) as pilot:
        await pilot.pause()
        app._run_slash_command("/model anthropic/claude-opus-5")
        await pilot.pause()
        text = _transcript_text(app)
    assert _unwrapped("this session") in _unwrapped(text), text
    assert _unwrapped("/model default") in _unwrapped(text), text
    # The access clause is unchanged by the new one sharing the line.
    assert _unwrapped("anthropic logged in") in _unwrapped(text), text


@pytest.mark.asyncio
async def test_model_default_confirms_both_keys_and_the_file_it_wrote(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A bare "saved" is a claim the user cannot check without relaunching.
    The provider is the half that rides along silently — it is written from the
    selector's left side and never typed as a setting of its own."""
    import yaml

    monkeypatch.setenv("LOCAL_OPERATOR_CONFIG_DIR", str(tmp_path))
    # A file on disk, so ConfigManager builds its own Config rather than handing
    # back (and mutating) the module-level DEFAULT_CONFIG singleton.
    (tmp_path / "config.yml").write_text("version: 0.0.0\nvalues:\n  hosting: openrouter\n")
    session = _SwitchableSession()
    ctrl = _AccessController(stored=("openrouter", "anthropic"))
    app = OperatorApp(lambda: _factory(session), provider_controller=ctrl)
    async with app.run_test(size=(90, 24)) as pilot:
        await pilot.pause()
        app._run_slash_command("/model default anthropic/claude-opus-5")
        await pilot.pause()
        text = _transcript_text(app)
    written = yaml.safe_load((tmp_path / "config.yml").read_text())["values"]
    assert written["hosting"] == "anthropic", written
    assert written["model_name"] == "claude-opus-5", written
    # What it wrote, under the names the config file uses…
    assert _unwrapped("hosting anthropic, model_name claude-opus-5") in _unwrapped(text), text
    # …and where, so the user can go and read or undo it.
    assert str(tmp_path / "config.yml") in _unwrapped(text), text


@pytest.mark.asyncio
async def test_model_default_alone_persists_the_model_already_in_use(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The sentence a user has right after switching is "make THIS the default".
    Answering it used to mean retyping the selector back at the app, which is the
    transcription step that made the default feel like a separate, hidden system."""
    import yaml

    monkeypatch.setenv("LOCAL_OPERATOR_CONFIG_DIR", str(tmp_path))
    (tmp_path / "config.yml").write_text("version: 0.0.0\nvalues:\n  hosting: openrouter\n")
    session = _SwitchableSession()
    ctrl = _AccessController(stored=("openrouter", "anthropic"))
    app = OperatorApp(lambda: _factory(session), provider_controller=ctrl)
    async with app.run_test(size=(90, 24)) as pilot:
        await pilot.pause()
        app._run_slash_command("/model anthropic/claude-opus-5")
        await pilot.pause()
        app._run_slash_command("/model default")
        await pilot.pause()
        text = _transcript_text(app)
    written = yaml.safe_load((tmp_path / "config.yml").read_text())["values"]
    assert (written["hosting"], written["model_name"]) == ("anthropic", "claude-opus-5"), written
    assert _unwrapped("hosting anthropic, model_name claude-opus-5") in _unwrapped(text), text


@pytest.mark.asyncio
async def test_every_model_default_surface_says_it_the_same_way() -> None:
    """D14. One instruction had four wordings on four surfaces a user meets
    within two keystrokes. The canonical sentence now names the consequence —
    future sessions — rather than merely saying an unspecified pair is saved.

    The defect is the divergence, so all four surfaces are checked together.
    The footer is checked unwrapped and whole because it is the tightest site:
    an instruction that is consistent only after truncation is not consistent.
    """
    session = _SwitchableSession()
    ctrl = _AccessController(stored=("openrouter", "anthropic"))
    app = OperatorApp(lambda: _factory(session), provider_controller=ctrl)
    async with app.run_test(size=(90, 30)) as pilot:
        await pilot.pause()
        app._run_slash_command("/model")
        await pilot.pause()
        await pilot.pause()
        picker = app.query_one(Editor).model_picker
        footer = picker.render_text(picker.size.width or 90).plain.split("\n")[-1]
        bare_notice = _transcript_text(app)
        app._run_slash_command("/model anthropic/claude-opus-5")
        await pilot.pause()
        receipt = _transcript_text(app)
        app._run_slash_command("/help")
        await pilot.pause()
        help_text = _transcript_text(app)

    # 1. the notice a bare `/model` prints above the list, 2. the switch receipt,
    # 3. the picker's own footer, 4. the `/help` row.
    assert _unwrapped(PERSIST_HINT) in _unwrapped(bare_notice), bare_notice
    assert _unwrapped(PERSIST_HINT) in _unwrapped(receipt), receipt
    assert PERSIST_HINT in footer, footer
    assert _unwrapped(PERSIST_HINT) in _unwrapped(help_text), help_text
    model_row = next(c for c in SLASH_COMMANDS if c.name == "model")
    assert PERSIST_HINT in model_row.description, model_row.description

    # And the four it replaced are gone from all of them, so there is no second
    # phrasing left for the user to meet.
    everything = _unwrapped(bare_notice + receipt + footer + help_text)
    for stale in (
        "as the boot default",
        "to make it the boot default",
        "saves the boot default",
        "/model default persists it",
    ):
        assert _unwrapped(stale) not in everything, stale

    # The receipt is two clauses now, not a run-on of four separators — and it
    # still says the two things that made it necessary: the scope, and the access
    # state of the provider it just switched to.
    switch_line = next(line for line in _unwrapped(receipt).split("·") if _unwrapped("→") in line)
    assert _unwrapped("(this session)") in switch_line, switch_line
    assert _unwrapped("from the next turn") not in _unwrapped(receipt), receipt
    assert _unwrapped("anthropic logged in") in _unwrapped(receipt), receipt


@pytest.mark.asyncio
async def test_model_default_hint_survives_every_supported_narrow_footer() -> None:
    """More terminal width must never hide words from the approved persistence
    instruction; the 50-column footer must keep the command whole."""
    for size in ((50, 20), (60, 22), (80, 24), (100, 30)):
        ctrl = _AccessController()
        app = OperatorApp(lambda: _factory(FakeSession()), provider_controller=ctrl)
        async with app.run_test(size=size) as pilot:
            await pilot.pause()
            await _open_model_picker(app, pilot)
            await pilot.pause()
            painted = "\n".join(strip.text for strip in app.screen._compositor.render_strips())
        assert PERSIST_HINT in painted, (size, painted)


@pytest.mark.asyncio
async def test_boot_warms_session_imports_before_awaiting_the_factory() -> None:
    """The import cost must be paid in a thread, ahead of the factory await.

    ``create_session`` does not yield until it has imported the engine, the
    provider stack and the MCP SDK — measured at ~700 ms — so awaiting it
    directly from the boot worker freezes the compositor and the key handler
    for that whole window: the user's first keystrokes land in a dead screen
    and appear in a burst afterwards. Ordering is the contract (warm, THEN
    build) and the thread hop is what makes the warm-up non-blocking, so both
    are asserted here.
    """
    import threading

    calls: list[str] = []
    warm_thread: list[int] = []

    def fake_warm() -> None:
        warm_thread.append(threading.get_ident())
        calls.append("warm")

    async def factory() -> FakeSession:
        calls.append("factory")
        return FakeSession()

    monkeypatch = pytest.MonkeyPatch()
    monkeypatch.setattr("local_operator.session_factory.warm_session_imports", fake_warm)
    try:
        app = OperatorApp(factory)
        async with app.run_test(size=(100, 30)) as pilot:
            await pilot.pause()
            for _ in range(50):
                if calls.count("factory"):
                    break
                await pilot.pause()
            loop_thread = threading.get_ident()
    finally:
        monkeypatch.undo()

    assert calls[:2] == ["warm", "factory"], calls
    assert warm_thread and warm_thread[0] != loop_thread


class _MeasuredSession(FakeSession):
    """A session that can report what it is already carrying."""

    def __init__(self, tokens: int = 42_318) -> None:
        super().__init__()
        self.tokens = tokens
        self.measure_calls = 0

    @property
    def model(self) -> Any:
        return SimpleNamespace(context_window=1_000_000, reasoning_effort="", reasoning=False)

    async def measure_preloaded_context(self) -> int:
        self.measure_calls += 1
        return self.tokens


def _ctx_estimate(app: OperatorApp) -> bool:
    """Whether the band's reading is a local estimate rather than the wire."""
    status = app._status
    assert status is not None
    return status.context_is_estimate


def _ctx_tokens(app: OperatorApp) -> int:
    """The band's context reading, with the ``_status is None`` case ruled out.

    The attribute is Optional until compose runs; every caller here is inside
    ``run_test``, where it never is.
    """
    status = app._status
    assert status is not None
    return status.context_tokens


async def _settle(pilot, predicate, tries: int = 60) -> bool:
    for _ in range(tries):
        if predicate():
            return True
        await pilot.pause()
    return predicate()


@pytest.mark.asyncio
async def test_context_reads_before_the_first_message() -> None:
    """The band must not claim an empty context while the prompt is loaded.

    The provider's exact ``prompt_tokens`` only exists after a turn, so without
    a boot measurement the one number a user checks before deciding what to ask
    is blank precisely when the system prompt, skills index and every tool
    schema are already spent.
    """
    session = _MeasuredSession()
    app = OperatorApp(lambda: _factory(session))
    async with app.run_test(size=(100, 30)) as pilot:
        assert await _settle(pilot, lambda: _ctx_tokens(app) == 42_318)
        assert _ctx_estimate(app) is True
        assert session.prompts == [], "measuring must not send anything"


@pytest.mark.asyncio
async def test_a_real_turn_supersedes_the_estimate() -> None:
    """An estimate is a stand-in, not a competitor: the wire number wins."""
    session = _MeasuredSession()
    app = OperatorApp(lambda: _factory(session))
    async with app.run_test(size=(100, 30)) as pilot:
        assert await _settle(pilot, lambda: _ctx_tokens(app) == 42_318)

        app.post_message(TurnEnded(aborted=False, error=None, context_tokens=51_007))
        assert await _settle(pilot, lambda: _ctx_tokens(app) == 51_007)
        assert _ctx_estimate(app) is False

        # And a later re-measure (an MCP server connecting) must not walk it back.
        session.tokens = 9_999
        app._measure_preloaded_context(session)
        for _ in range(10):
            await pilot.pause()
        assert _ctx_tokens(app) == 51_007


@pytest.mark.asyncio
async def test_a_turn_landing_mid_measurement_still_wins() -> None:
    """The race the cheap outer check cannot cover.

    A measurement that started while the context was unknown can finish AFTER
    a turn has reported the provider's exact count. Deciding only at dispatch
    would let the stale estimate overwrite the better number, so the result is
    re-checked against the state at the moment it lands.
    """
    session = _MeasuredSession()
    app = OperatorApp(lambda: _factory(session))
    async with app.run_test(size=(100, 30)) as pilot:
        assert await _settle(pilot, lambda: _ctx_tokens(app) == 42_318)

        # Park a second measurement mid-flight, with the reading still an
        # estimate so the dispatch-time check lets it through.
        release = asyncio.Event()

        async def slow_measure() -> int:
            await release.wait()
            return 9_999

        session.measure_preloaded_context = slow_measure  # type: ignore[method-assign]
        app._measure_preloaded_context(session)
        await pilot.pause()

        # The turn lands while that is parked...
        app.post_message(TurnEnded(aborted=False, error=None, context_tokens=51_007))
        assert await _settle(pilot, lambda: _ctx_tokens(app) == 51_007)

        # ...and the late estimate must not undo it.
        release.set()
        for _ in range(15):
            await pilot.pause()
        assert _ctx_tokens(app) == 51_007
        assert _ctx_estimate(app) is False


@pytest.mark.asyncio
async def test_a_growing_tool_inventory_re_measures() -> None:
    """MCP schemas are the biggest term, and they land after boot."""
    session = _MeasuredSession()
    app = OperatorApp(lambda: _factory(session))
    async with app.run_test(size=(100, 30)) as pilot:
        assert await _settle(pilot, lambda: _ctx_tokens(app) == 42_318)

        session.tokens = 88_000
        app._measure_preloaded_context(session)
        assert await _settle(pilot, lambda: _ctx_tokens(app) == 88_000)
        assert _ctx_estimate(app) is True


@pytest.mark.asyncio
async def test_a_session_without_the_capability_is_not_a_crash() -> None:
    """Reduced hosts (embedders, these fakes) have no measurement to offer."""
    session = FakeSession()
    app = OperatorApp(lambda: _factory(session))
    async with app.run_test(size=(100, 30)) as pilot:
        for _ in range(10):
            await pilot.pause()
        assert _ctx_tokens(app) == 0
        assert _ctx_estimate(app) is False


@pytest.mark.asyncio
async def test_reload_drops_the_dead_sessions_context() -> None:
    """An exact count belongs to one conversation, and dies with it.

    The dispatch guard reads "a non-zero reading that is not an estimate" as
    "the band already knows better than you". Left standing across a reload
    that is a lie about a session that no longer exists, and it also suppresses
    the replacement session's own measurement.
    """
    first = _MeasuredSession()
    app = OperatorApp(lambda: _factory(first))
    async with app.run_test(size=(100, 30)) as pilot:
        assert await _settle(pilot, lambda: _ctx_tokens(app) == 42_318)

        app.post_message(TurnEnded(aborted=False, error=None, context_tokens=51_007))
        assert await _settle(pilot, lambda: _ctx_tokens(app) == 51_007)

        second = _MeasuredSession(tokens=20_000)
        app._session_factory = lambda: _factory(second)  # type: ignore[assignment]
        await app._reload_session()

        assert await _settle(pilot, lambda: _ctx_tokens(app) == 20_000)
        assert second.measure_calls >= 1, "the new session must be measured"


@pytest.mark.asyncio
async def test_the_newest_measurement_wins_not_the_slowest() -> None:
    """Several measurements can be in flight; they finish out of order.

    Neither precedence guard can break that tie — both only ask whether the
    reading is exact, which is false for every estimate — so without
    cancellation the LAST to land wins rather than the newest, and the band
    settles on a smaller inventory than the session actually has.
    """
    session = _MeasuredSession()
    app = OperatorApp(lambda: _factory(session))
    async with app.run_test(size=(100, 30)) as pilot:
        assert await _settle(pilot, lambda: _ctx_tokens(app) == 42_318)

        slow_release = asyncio.Event()

        async def slow() -> int:
            await slow_release.wait()
            return 30_000

        async def fast() -> int:
            return 90_000

        session.measure_preloaded_context = slow  # type: ignore[method-assign]
        app._measure_preloaded_context(session)
        await pilot.pause()

        session.measure_preloaded_context = fast  # type: ignore[method-assign]
        app._measure_preloaded_context(session)
        assert await _settle(pilot, lambda: _ctx_tokens(app) == 90_000)

        # Releasing the superseded measurement must not walk the band back.
        slow_release.set()
        for _ in range(15):
            await pilot.pause()
        assert _ctx_tokens(app) == 90_000


@pytest.mark.asyncio
async def test_a_dead_sessions_turn_cannot_restore_its_context() -> None:
    """The race the reload reset alone does not close.

    A plain ``/reload`` deliberately keeps the controller SUBSCRIBED across
    ``dispose()`` so the dying session's ``agent_end`` can settle its live tool
    cards. That same event carries a ``context_tokens``, and it is posted to
    the message pump — so whether it arrives before or after the reload's reset
    is scheduling. Arriving after, it reinstates an exact reading for a
    conversation that no longer exists, and the exact-count guard then
    suppresses the replacement session's own measurement: the reset undone.
    """
    first = _MeasuredSession()
    app = OperatorApp(lambda: _factory(first))
    async with app.run_test(size=(100, 30)) as pilot:
        assert await _settle(pilot, lambda: _ctx_tokens(app) == 42_318)

        second = _MeasuredSession(tokens=20_000)
        app._session_factory = lambda: _factory(second)  # type: ignore[assignment]
        await app._reload_session()
        assert await _settle(pilot, lambda: _ctx_tokens(app) == 20_000)

        # The dying session's turn lands late, carrying its own exact count.
        app._session = None
        app.post_message(TurnEnded(aborted=True, error=None, context_tokens=51_007))
        for _ in range(15):
            await pilot.pause()

        assert _ctx_tokens(app) == 20_000, "a dead session's count was adopted"
        assert _ctx_estimate(app) is True


def _resumed(*messages: Any) -> Any:
    """A session whose ``history()`` is the given messages."""

    class _Resumed(FakeSession):
        def history(self) -> list[Any]:
            return list(messages)

    return _Resumed()


def _blocks_by_type(app: OperatorApp) -> dict[str, int]:
    from local_operator.tui.widgets.assistant import AssistantBlock as _AB
    from local_operator.tui.widgets.tool_card import ToolCard as _TC
    from local_operator.tui.widgets.transcript import NoticeBlock as _NB
    from local_operator.tui.widgets.transcript import UserBlock as _UB

    counts = {"user": 0, "assistant": 0, "tool": 0, "notice": 0}
    for block in app.query_one(TranscriptView).blocks():
        if isinstance(block, _UB):
            counts["user"] += 1
        elif isinstance(block, _AB):
            counts["assistant"] += 1
        elif isinstance(block, _TC):
            counts["tool"] += 1
        elif isinstance(block, _NB):
            counts["notice"] += 1
    return counts


class TestResumeReplaysTheWholeConversation:
    """What ``--resume`` puts back on screen.

    The old rule was "prompts, plus assistant messages carrying no tool calls".
    An agent turn is text AND tool_calls in ONE message, so that excluded the
    prose too: on a real 396-message session it mounted 6 blocks and dropped 74
    assistant messages, 215 calls and 215 results. The screen read as a list of
    questions nobody had answered.
    """

    @pytest.mark.asyncio
    async def test_prose_that_accompanies_a_tool_call_is_kept(self) -> None:
        from local_operator.harness.types import Message, ToolCall

        session = _resumed(
            Message.user("find the bug"),
            Message(
                role="assistant",
                content=[TextContent(text="Reading the file first.")],
                tool_calls=[ToolCall(id="c1", name="read", arguments={"path": "a.py"})],
            ),
            Message(
                role="tool",
                content=[TextContent(text="file contents")],
                tool_call_id="c1",
                tool_name="read",
            ),
            Message.assistant("Found it."),
        )
        app = OperatorApp(lambda: _factory(session))
        async with app.run_test(size=(100, 30)) as pilot:
            assert await _settle(pilot, lambda: _blocks_by_type(app)["tool"] == 1)
            counts = _blocks_by_type(app)
        # The old rule scored assistant=1 here; the turn's own sentence was lost.
        assert counts == {"user": 1, "assistant": 2, "tool": 1, "notice": 0}

    @pytest.mark.asyncio
    async def test_a_tool_call_is_paired_with_its_result(self) -> None:
        """Results can arrive several messages after the call that asked for
        them, so they are indexed rather than matched positionally."""
        from local_operator.harness.types import Message, ToolCall
        from local_operator.tui.widgets.tool_card import ToolCard

        session = _resumed(
            Message(
                role="assistant",
                tool_calls=[
                    ToolCall(id="a", name="read", arguments={"path": "x"}),
                    ToolCall(id="b", name="bash", arguments={"command": "ls"}),
                ],
            ),
            Message(
                role="tool",
                content=[TextContent(text="boom")],
                tool_call_id="b",
                tool_name="bash",
                is_error=True,
            ),
            Message(
                role="tool", content=[TextContent(text="ok")], tool_call_id="a", tool_name="read"
            ),
        )
        app = OperatorApp(lambda: _factory(session))
        async with app.run_test(size=(100, 30)) as pilot:
            assert await _settle(pilot, lambda: _blocks_by_type(app)["tool"] == 2)
            cards = [b for b in app.query_one(TranscriptView).blocks() if isinstance(b, ToolCard)]
        assert [c.tool_name for c in cards] == ["read", "bash"], "call order, not result order"
        assert [c._state for c in cards] == ["success", "error"]

    @pytest.mark.asyncio
    async def test_a_call_with_no_result_is_interrupted_not_complete(self) -> None:
        """A session killed mid-turn leaves a call with no answer. Rendering it
        as done would invent an outcome that never happened."""
        from local_operator.harness.types import Message, ToolCall
        from local_operator.tui.widgets.tool_card import ToolCard

        session = _resumed(
            Message(
                role="assistant",
                tool_calls=[ToolCall(id="orphan", name="bash", arguments={"command": "sleep 99"})],
            ),
        )
        app = OperatorApp(lambda: _factory(session))
        async with app.run_test(size=(100, 30)) as pilot:
            assert await _settle(pilot, lambda: _blocks_by_type(app)["tool"] == 1)
            card = next(
                b for b in app.query_one(TranscriptView).blocks() if isinstance(b, ToolCard)
            )
        assert card._state == "interrupted"

    @pytest.mark.asyncio
    async def test_a_replayed_card_reports_no_duration(self) -> None:
        """The transcript records what a tool did, never how long it took.
        ``0.0s`` on every row is a wrong number, not a missing one."""
        from local_operator.harness.types import Message, ToolCall
        from local_operator.tui.widgets.tool_card import ToolCard

        session = _resumed(
            Message(role="assistant", tool_calls=[ToolCall(id="c", name="read", arguments={})]),
            Message(role="tool", content=[TextContent(text="ok")], tool_call_id="c"),
        )
        app = OperatorApp(lambda: _factory(session))
        async with app.run_test(size=(100, 30)) as pilot:
            assert await _settle(pilot, lambda: _blocks_by_type(app)["tool"] == 1)
            card = next(
                b for b in app.query_one(TranscriptView).blocks() if isinstance(b, ToolCard)
            )
            rendered = card._build_row(100).plain
        assert card._duration is None
        assert "0.0s" not in rendered, rendered

    @pytest.mark.asyncio
    async def test_a_failed_turn_says_so_instead_of_vanishing(self) -> None:
        """The reported symptom: two identical prompts and nothing between them.

        The turn had errored, and an assistant message with neither prose nor a
        call was skipped — so the screen implied the second prompt was a
        duplicate rather than a retry.
        """
        from local_operator.harness.types import Message

        session = _resumed(
            Message.user("do the thing"),
            Message(role="assistant", stop_reason="error"),
            Message.user("do the thing"),
        )
        app = OperatorApp(lambda: _factory(session))
        async with app.run_test(size=(100, 30)) as pilot:
            assert await _settle(pilot, lambda: _blocks_by_type(app)["notice"] == 1)
            counts = _blocks_by_type(app)
        assert counts["user"] == 2 and counts["notice"] == 1

    @pytest.mark.asyncio
    async def test_a_fresh_session_still_replays_nothing(self) -> None:
        """The splash must not be retired by an empty history."""
        app = OperatorApp(lambda: _factory(_resumed()))
        async with app.run_test(size=(100, 30)) as pilot:
            for _ in range(10):
                await pilot.pause()
            assert _blocks_by_type(app) == {"user": 0, "assistant": 0, "tool": 0, "notice": 0}


@pytest.mark.asyncio
async def test_new_starts_a_fresh_conversation() -> None:
    """``/new`` had no equivalent: ``/clear`` keeps the conversation the model
    sees, ``/reload`` reboots the same one, ``/resume`` moves to another
    existing one. Starting fresh meant quitting the app."""
    boots: list[str | None] = []
    app = OperatorApp(lambda: _factory(FakeSession()), resume_factory=_resume_factory(boots))
    async with app.run_test(size=(100, 30)) as pilot:
        await pilot.pause()
        app._run_slash_command("/new")
        assert await _settle(pilot, lambda: boots == [None])
    # None, not "" and not a sentinel id: create_session branches on
    # `resume is not None`, so this is the cold-launch path.
    assert boots == [None]


@pytest.mark.asyncio
async def test_new_replaces_the_visible_ledger() -> None:
    """The transcript must not outlive the conversation it describes."""
    boots: list[str | None] = []
    app = OperatorApp(lambda: _factory(FakeSession()), resume_factory=_resume_factory(boots))
    async with app.run_test(size=(100, 30)) as pilot:
        await pilot.pause()
        app._append_block(NoticeBlock("from the old conversation", "info"))
        assert any(
            "old conversation" in _renderable_plain(getattr(b, "renderable", ""))
            for b in app.query_one(TranscriptView).blocks()
        )
        app._run_slash_command("/new")
        assert await _settle(pilot, lambda: boots == [None])
        for _ in range(10):
            await pilot.pause()
        assert not any(
            "old conversation" in _renderable_plain(getattr(b, "renderable", ""))
            for b in app.query_one(TranscriptView).blocks()
        )


@pytest.mark.asyncio
async def test_new_without_a_capable_launcher_says_so() -> None:
    """Embedders may supply no session factory beyond the first."""
    app = OperatorApp(lambda: _factory(FakeSession()))
    async with app.run_test(size=(100, 30)) as pilot:
        await pilot.pause()
        app._run_slash_command("/new")
        for _ in range(6):
            await pilot.pause()
        text = "\n".join(
            _renderable_plain(getattr(b, "renderable", ""))
            for b in app.query_one(TranscriptView).blocks()
        )
    assert "unavailable" in text


# --- "am I focused?", both sides of it ----------------------------------------
#
# The composer answered neither side. It drew no caret at all while the buffer
# was empty, so clicking into it changed nothing on the frame; and its bright
# chevron — the affordance meant to carry focus (D23) — was driven by
# `#input-dock:focus-within`, which Textual never re-applied on blur, so once
# lit it stayed lit for the life of the process. Both directions are pinned
# here, off a real composed frame rather than off widget state: what the user
# was missing is what the terminal was SENT.


@pytest.mark.asyncio
async def test_the_focused_composer_shows_a_caret_and_the_blurred_one_shows_none() -> None:
    """The reported case exactly: an EMPTY composer, clicked into.

    A caret on an empty field has nowhere to go but the placeholder's first
    cell, so the placeholder starts one column later while the caret is drawn
    and both survive — the block is on a blank cell, and the copy is unbroken.
    Blurring takes the caret away again, which is the half that makes the
    presence of one mean anything.
    """
    app = OperatorApp(lambda: _factory(FakeSession()))
    async with app.run_test(size=(120, 40)) as pilot:
        await pilot.pause()
        editor = app.query_one(Editor)
        editor.focus()
        await pilot.pause()

        assert editor.text == "", "premise: the reported state is an EMPTY composer"
        focused = composer_cells(app)
        assert caret_cells(focused) == [" "], "the focused composer drew no caret"
        placeholder = [text for text, _, _ in focused if "Message Local Operator" in text]
        assert placeholder, f"the caret broke the placeholder up: {focused}"

        app.set_focus(None)
        await pilot.pause()
        assert not caret_cells(composer_cells(app)), "the caret survived the blur"

        editor.focus()
        await pilot.pause()
        assert caret_cells(composer_cells(app)) == [" "], "the caret did not come back"


@pytest.mark.asyncio
async def test_the_caret_holds_still_across_consecutive_frames() -> None:
    """A caret that is absent from half the frames is not a focus signal.

    Sampled across four stock blink periods (500 ms each), empty and with text:
    the empty state is new, and it is the one where a reintroduced blink would
    strobe against a static splash.
    """
    app = OperatorApp(lambda: _factory(FakeSession()))
    async with app.run_test(size=(120, 40)) as pilot:
        await pilot.pause()
        editor = app.query_one(Editor)
        editor.focus()
        await pilot.pause()
        assert editor.cursor_blink is False

        for text in ("", "hello"):
            editor.load_text(text)
            await pilot.pause()
            samples = set()
            for _ in range(8):
                await asyncio.sleep(0.25)
                await pilot.pause()
                samples.add(tuple(composer_cells(app)))
            assert len(samples) == 1, f"the composer row changed between frames: {text!r}"
            assert caret_cells(next(iter(samples))), f"no caret at all with text={text!r}"


@pytest.mark.asyncio
async def test_the_chevron_brightens_on_focus_and_never_spends_the_accent() -> None:
    """The second affordance, in both directions — it used to only go on.

    And it used to go on GREEN. The accent means "a turn is live" and the
    composer is focused in nearly every frame, so the chevron sat permanently
    accent two rows above the band's streaming spinner, which is accent for a
    different reason (D5). Focus is a brightness step in the same neutral ramp
    now: `fg` focused, `dim` blurred, a 3.86x luminance move where the accent
    was 2.15x. The accent assertion is the load-bearing one — a future pass
    reaching for green here reintroduces the collision, and the two-meanings
    bug is invisible in any frame where no turn is running.

    The class is asserted alongside the painted colour because the class is the
    mechanism the stylesheet reads: a bright chevron with the class off would be
    a stale frame, and the class on with a dim chevron would be a broken rule.
    The blur is a tool card taking focus — a real product surface, the one a
    user reaches by clicking or tabbing to a tool row, and it leaves the
    composer on screen, which is exactly when "are my keys still going to the
    composer?" is a live question.
    """
    app = OperatorApp(lambda: _factory(FakeSession()))
    async with app.run_test(size=(120, 40)) as pilot:
        await pilot.pause()
        dock = app.query_one("#input-dock")
        focused_ink = theme_mod.semantic_color("fg").lower()
        accent = theme_mod.semantic_color("accent").lower()
        dim = theme_mod.semantic_color("dim").lower()

        app.query_one(Editor).focus()
        await pilot.pause()
        assert dock.has_class(COMPOSER_FOCUSED_CLASS)
        assert chevron_colour(composer_cells(app)) == focused_ink
        assert chevron_colour(composer_cells(app)) != accent, "focus spent the accent"

        card = ToolCard("t1", "bash", {"command": "ls"})
        app._append_block(card)
        await pilot.pause()
        card.focus()
        await pilot.pause()
        assert not dock.has_class(COMPOSER_FOCUSED_CLASS)
        assert chevron_colour(composer_cells(app)) == dim, "the chevron stayed lit"

        app.query_one(Editor).focus()
        await pilot.pause()
        assert dock.has_class(COMPOSER_FOCUSED_CLASS)
        assert chevron_colour(composer_cells(app)) == focused_ink


@pytest.mark.asyncio
async def test_the_read_only_composer_shows_neither_caret_nor_bright_chevron() -> None:
    """The subagent page's composer refuses every key, and looks it.

    Today this holds because `_set_composer_read_only` drops `can_focus` and
    pushes focus off the widget, so `TextArea._draw_cursor`'s read-only branch
    (`show_cursor and has_focus`, a DIFFERENT expression from the one every
    other state takes) resolves False. That is a chain of three decisions in
    two files, and restoring focusability at either end would put a caret in a
    field that ignores it — the most misleading thing this mode could paint.
    Pinned on the frame so it goes red there rather than in a bug report.
    """
    app = OperatorApp(lambda: _factory(FakeSession()))
    async with app.run_test(size=(120, 40)) as pilot:
        await pilot.pause()
        editor = app.query_one(Editor)
        editor.focus()
        await pilot.pause()
        assert caret_cells(composer_cells(app)) == [" "], "premise: it starts focused"

        app._set_composer_read_only(True)
        await pilot.pause()
        assert editor.read_only is True
        cells = composer_cells(app)
        assert not caret_cells(cells), "a caret is pointing into a field that refuses keys"
        assert chevron_colour(cells) == theme_mod.semantic_color("dim").lower()

        app._set_composer_read_only(False)
        app.query_one(Editor).focus()
        await pilot.pause()
        assert caret_cells(composer_cells(app)) == [" "], "the composer never came back"


@pytest.mark.asyncio
async def test_the_aside_keeps_the_caret_because_it_keeps_the_composer() -> None:
    """`/btw` opens a card the composer types INTO, so focus never moves.

    It is the one mode where the two affordances have to stay ON while the
    conversation's own surface has visibly stepped aside, and the placeholder
    is a sentence about the mode rather than an invitation — so this is also
    the proof that the caret's cell is taken from the field, not from the copy,
    whatever the copy happens to say.
    """
    app = OperatorApp(lambda: _factory(FakeSession()))
    async with app.run_test(size=(120, 40)) as pilot:
        await pilot.pause()
        app.query_one(Editor).focus()
        await pilot.pause()

        assert app._open_aside() is not None
        await pilot.pause()
        assert app.query_one(Editor).placeholder == ASIDE_PLACEHOLDER
        cells = composer_cells(app)
        assert caret_cells(cells) == [" "], "the aside took the caret with it"
        assert chevron_colour(cells) == theme_mod.semantic_color("fg").lower()
        # The CONSTANT, not a copy literal: the claim is that the caret takes
        # its cell from the field rather than out of the placeholder, which is
        # true whatever the placeholder says. Spelling the sentence out here
        # made this test fail on a wording change that did not touch the caret.
        assert [
            text for text, _, _ in cells if ASIDE_PLACEHOLDER in text
        ], f"the caret broke the aside's own placeholder up: {cells}"


def _osc_titles(writes: Sequence[str]) -> list[str]:
    """The OSC 0 title payloads, in order, deduped only by the caller.

    Saved titles (`\x1b[22;2t`) and restores (`\x1b[23;2t`) are different
    evidence and are asserted separately, so this helper extracts only the
    "what would the tab title read" payloads.
    """
    out: list[str] = []
    for data in writes:
        if data.startswith("\x1b]0;") and data.endswith("\x07"):
            out.append(data.removeprefix("\x1b]0;").removesuffix("\x07"))
    return out


def _spy_driver_writes(app: OperatorApp, writes: list[str]) -> None:
    """Record the real driver's writes while still forwarding them.

    Tests want the app-level contract (`_start_terminal_title` writes through
    the driver and `on_unmount` restores through it), not a fake driver with a
    growing list of Textual methods to impersonate.
    """
    assert app._driver is not None
    original = app._driver.write

    def spy(data: str) -> None:
        writes.append(data)
        original(data)

    app._driver.write = spy  # type: ignore[method-assign]


@pytest.mark.asyncio
async def test_app_wires_the_terminal_title_to_boot_and_turn_state(
    monkeypatch,
) -> None:  # noqa: ANN001
    """The app, not just the band, must prove the wiring works.

    ``run_test`` uses Textual's headless driver, so `_start_terminal_title`
    normally exits early and the app-level contract would otherwise go
    unexercised: boot attaches a title writer, the writer sees the cwd fallback
    on first stable paint, and a started turn flips it to the working state.
    """
    writes: list[str] = []
    monkeypatch.setattr(OperatorApp, "is_headless", property(lambda self: False))
    app = OperatorApp(lambda: _factory(FakeSession()))
    async with app.run_test(size=(100, 30)) as pilot:
        _spy_driver_writes(app, writes)
        app._stop_terminal_title()
        app._start_terminal_title()
        await pilot.pause()
        titles = _osc_titles(writes)
        assert titles[-1] == "lo › lo-terminal-title"
        assert "lo ›" not in titles

        app.on_turn_started(TurnStarted())
        await pilot.pause()
        latest = _osc_titles(writes)[-1]
        assert latest.endswith(" lo-terminal-title")
        assert latest.split(" ")[1] in {"⣾", "⣽", "⣻", "⢿", "⡿", "⣟", "⣯", "⣷"}


@pytest.mark.asyncio
async def test_session_swap_clears_a_parked_approval_title(monkeypatch) -> None:  # noqa: ANN001
    """A dead session must not leave `lo ! …` standing on the replacement.

    This is the bug the review found: `_approval` was cleared during a session
    swap, but the title's derived `attention` state was not, so `/reload` or
    `/new` from a parked approval left the next idle session claiming it still
    owed one.
    """
    writes: list[str] = []
    monkeypatch.setattr(OperatorApp, "is_headless", property(lambda self: False))
    app = OperatorApp(lambda: _factory(FakeSession()))
    async with app.run_test(size=(100, 30)) as pilot:
        _spy_driver_writes(app, writes)
        app._stop_terminal_title()
        app._start_terminal_title()
        await pilot.pause()
        approval = ApprovalBlock("bash", "run a command")
        app._approval = approval
        app._refresh_working_activity()
        await pilot.pause()
        assert _osc_titles(writes)[-1] == "lo ! lo-terminal-title"

        await app._reload_session()
        await pilot.pause()
        assert _osc_titles(writes)[-1] == "lo › lo-terminal-title"


@pytest.mark.asyncio
async def test_unmount_restores_the_terminals_own_title(monkeypatch) -> None:  # noqa: ANN001
    """The app owns the terminal and therefore the restore.

    Pinned here rather than in the pure title unit tests because the ordering is
    app-level: `on_unmount` must restore before any awaited teardown can fail
    and strand the shell under this session's title.
    """
    writes: list[str] = []
    monkeypatch.setattr(OperatorApp, "is_headless", property(lambda self: False))
    app = OperatorApp(lambda: _factory(FakeSession()))
    async with app.run_test(size=(100, 30)) as pilot:
        _spy_driver_writes(app, writes)
        app._stop_terminal_title()
        app._start_terminal_title()
        await pilot.pause()
    assert "\x1b[22;2t" in writes
    assert writes[-1] == "\x1b[23;2t"
