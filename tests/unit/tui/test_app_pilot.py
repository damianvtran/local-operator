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


def _renderable_plain(renderable) -> str:
    """Recursively flatten a Rich renderable (incl. Group/Padding) to text."""
    from rich.text import Text
    from rich.console import Group
    from rich.padding import Padding

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


# --- login/logout through the provider controller --------------------------


@pytest.mark.asyncio
async def test_login_lists_providers_from_the_controller() -> None:
    """Bare /login lists loginable providers — the controller is the only
    path now that the CLI login_handler seam is gone."""
    app = OperatorApp(lambda: _factory(FakeSession()), provider_controller=FakeProviderController())
    async with app.run_test(size=(80, 24)) as pilot:
        await pilot.pause()
        app.query_one(Editor).focus()
        await pilot.pause()
        await pilot.press("slash", "l", "o", "g", "i", "n", "enter")
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


@pytest.mark.asyncio
async def test_login_without_controller_points_at_the_cli() -> None:
    """Degrading to a pointer notice is the contract when the TUI is embedded
    without a controller — it must never crash or silently do nothing."""
    app = OperatorApp(lambda: _factory(FakeSession()))  # no controller
    async with app.run_test(size=(80, 24)) as pilot:
        await pilot.pause()
        app.query_one(Editor).focus()
        await pilot.pause()
        await pilot.press("slash", "l", "o", "g", "i", "n", "enter")
        await pilot.pause()
        assert "local-operator login" in _transcript_text(app)


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
        self.logins: list[str] = []
        self.logouts: list[str] = []

    def login_providers(self):
        return [
            _FakeDef("openrouter", "OpenRouter", None),
            _FakeDef("deepseek", "DeepSeek", None),
            _FakeDef("xai-oauth", "xAI OAuth", "xai"),
        ]

    def provider(self, pid):
        for d in self.login_providers():
            if d.id == pid:
                return d
        return None

    def has_any_credential(self, provider):
        return provider in ("openrouter",)

    def credentials(self):
        return [
            _FakeCred(1, "openrouter", "api_key", {"source": "login"}),
            _FakeCred(2, "deepseek", "oauth", {"expires": 9999999999999, "email": "a@b.c"}),
        ]

    def usage_enabled_providers(self):
        return ["openrouter", "zai"]

    def resolve_model(self, provider, model_id):
        return FakeModel(provider, model_id)

    async def login(self, provider):
        self.logins.append(provider)
        return f"logged in {provider}"

    async def logout(self, provider):
        self.logouts.append(provider)
        return f"removed {provider}"

    async def fetch_usage(self, provider_ids=None):
        return self.usage_reports


class _FakeDef:
    def __init__(self, pid, name, store_as):
        self.id = pid
        self.name = name
        self.store_credentials_as = store_as
        self.login = object()  # truthy -> has interactive login


class _FakeCred:
    def __init__(self, ident, provider, ctype, data):
        self.id = ident
        self.provider = provider
        self.credential_type = ctype
        self.data = data
        self.identity_key = None


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


@pytest.mark.asyncio
async def test_usage_command_renders_report() -> None:
    from local_operator.providers.usage import UsageAmount, UsageLimit, UsageReport

    session = FakeSession()
    ctrl = FakeProviderController()
    ctrl.usage_reports = [
        UsageReport(
            provider="openrouter",
            limits=[
                UsageLimit(
                    id="openrouter:credits",
                    label="Credits",
                    amount=UsageAmount(used=5.0, limit=50.0, unit="usd"),
                )
            ],
        )
    ]
    app = OperatorApp(lambda: _factory(session), provider_controller=ctrl)
    async with app.run_test(size=(80, 24)) as pilot:
        await pilot.pause()
        app.query_one(Editor).focus()
        await pilot.press("slash")
        for key in "u", "s", "a", "g", "e":
            await pilot.press(key)
        await pilot.press("enter")
        await pilot.pause()
        await pilot.pause()
        texts = _transcript_text(app)
    assert "openrouter" in texts
    assert "Credits" in texts


@pytest.mark.asyncio
async def test_run_tui_forwards_provider_controller(monkeypatch) -> None:
    """F3 regression: run_tui must pass provider_controller to OperatorApp so
    the slash-command surface is live (not a pointer, not a crash)."""
    import local_operator.tui.app as app_mod
    from local_operator.tui.app import OperatorApp

    seen: dict = {}
    fake_controller = object()

    class _SpyApp(OperatorApp):
        def __init__(self, *a, **kw):
            seen["controller"] = kw.get("provider_controller")
            super().__init__(*a, **kw)

        async def run_async(self):
            return None

    # run_tui lazy-imports OperatorApp from local_operator.tui.app at call
    # time, so patching that module attribute is what routes the spy in.
    monkeypatch.setattr(app_mod, "OperatorApp", _SpyApp)
    called = []

    async def factory():
        called.append(1)
        return _SpyApp()  # type: ignore[return-value]

    # Await run_tui with a fake session factory that must not await forever.
    async def factory2():
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

    async def prompt(self, text: str, attachments: list[Any] | None = None) -> None:
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
