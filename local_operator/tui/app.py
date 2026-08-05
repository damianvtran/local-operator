"""The full-screen Local Operator TUI (Textual).

Layout (top→bottom): the scrolling transcript, the input panel on the
``surface`` elevation step with the ``❯`` chevron, and the full-width status
BAND on the ``sunken`` ground. No bordered boxes anywhere (the only line in
the app is the input's thin top rule, ``$lo-dim``, with the focus accent
moving to the chevron — D23/D24). Structure comes from symbols, tint, and
spacing; the one space of edge padding sits left/right/bottom, never along
the top while scrolling.

The agent never imports this module; the app subscribes to
``SessionProtocol`` events through
:class:`~local_operator.tui.events.EventController`, which posts Textual
messages so all widget mutation happens on the Textual thread.

The session is injected as a FACTORY and awaited lazily in a worker so the
app paints first (session construction can take a moment: providers, skills,
MCP discovery). A boot failure surfaces as an error notice + ``session
error`` status and can be retried with ``/reload`` (TUI-012).
"""

from __future__ import annotations

import os
from typing import Any, Awaitable, Callable

from local_operator.tui.widgets.transcript import (
    NoticeBlock,
    RichBlock,
    TranscriptView,
    UserBlock,
    WorkingBlock,
)
from rich.console import Group
from rich.style import Style
from rich.terminal_theme import TerminalTheme
from rich.text import Text
from textual.app import App, ComposeResult
from textual.binding import Binding
from textual.containers import Container, Horizontal
from textual.widgets import Static

from local_operator.session.protocol import SessionProtocol
from local_operator.tui import theme as theme_mod
from local_operator.tui.autocomplete import SlashCommand
from local_operator.tui.events import (
    AssistantDelta,
    AssistantMessageEnd,
    AssistantMessageStart,
    CompactionEnded,
    CompactionStarted,
    EventController,
    NoticePosted,
    RetryEnded,
    RetryStarted,
    StartFlushTimer,
    ToolEnded,
    ToolStarted,
    ToolUpdated,
    TurnBoundaryEnd,
    TurnBoundaryStart,
    TurnEnded,
    TurnStarted,
)
from local_operator.tui.markdown_theme import brand_markdown_theme, install_markdown_theme
from local_operator.tui.widgets.assistant import AssistantBlock
from local_operator.tui.widgets.editor import (
    Editor,
    EditorQuit,
    EditorSubmitted,
    InterruptRequested,
)
from local_operator.tui.widgets.status_line import StatusLine, format_cost
from local_operator.tui.widgets.tool_card import ToolCard

#: Slash commands handled synchronously before any prompt is sent. One
#: registry entry per command; aliases live on the entry (TUI-014).
SLASH_COMMANDS: list[SlashCommand] = [
    SlashCommand("help", "Show available commands"),
    SlashCommand("exit", "Quit the app", aliases=("quit",)),
    SlashCommand("clear", "Clear the transcript (history is untouched)"),
    SlashCommand("reload", "Retry starting the session"),
    SlashCommand("model", "Show or switch model (provider/id)", aliases=("models",)),
    SlashCommand("provider", "List providers and their login/usage state"),
    SlashCommand("accounts", "List stored credentials"),
    SlashCommand("usage", "Show provider usage quota"),
    SlashCommand("compact", "About context compaction"),
    SlashCommand("skills", "List loaded skills"),
    SlashCommand("mcp", "List MCP servers"),
    SlashCommand("login", "Authenticate a provider"),
    SlashCommand("logout", "Remove stored provider credentials"),
]

#: One dim line shown until the first transcript block lands (D9).
BOOT_HINT = "type a message, or /help for commands"


class OperatorApp(App[None]):
    """Full-screen TUI over one ``SessionProtocol``."""

    TITLE = "Local Operator"
    CSS_PATH = "local_operator.tcss"

    BINDINGS = [
        Binding("ctrl+c", "interrupt", "Interrupt", show=False),
        Binding("ctrl+l", "clear_transcript", "Clear transcript", show=False),
    ]

    def __init__(
        self,
        session_factory: Callable[[], Awaitable[SessionProtocol]],
        theme_name: str = "dark",
        login_handler: Callable[[str], None] | None = None,
        provider_controller: Any | None = None,
    ) -> None:
        super().__init__()
        theme_mod.set_theme(theme_name)  # dark is the product's island night
        self._session_factory = session_factory
        self._login_handler = login_handler  # TUI-015: injected by the CLI
        # Full provider/model/credential/usage facade behind the slash
        # commands; ``None`` degrades /provider /usage /model-switch to
        # pointer notices (same contract as login_handler).
        self._providers = provider_controller
        self._session: SessionProtocol | None = None
        self._controller: EventController | None = None
        self._status: StatusLine | None = None
        self._streaming_block: AssistantBlock | None = None
        self._tool_cards: dict[str, ToolCard] = {}
        self._boot_hint: NoticeBlock | None = None
        self._working_block: WorkingBlock | None = None
        self._total_cost: float = 0.0
        # Serializes interactive login flows so two /login commands can never
        # race the one suspended terminal.
        self._login_lock: Any | None = None

    # -- composition --------------------------------------------------------
    def compose(self) -> ComposeResult:
        yield TranscriptView()
        # The status line IS the input box's top row: the band
        # docks at the top of the input panel and carries the structural rule
        # styling, so it can never be overdrawn or pushed off-screen by the
        # editor. One row does double duty — zero extra height (D3/D17).
        with Container(id="input-dock"):
            yield Static(id="status-band")
            with Horizontal(id="input-row"):
                yield Static("❯", id="prompt-chevron")
                yield Editor(commands=SLASH_COMMANDS)

    def get_css_variables(self) -> dict[str, str]:
        """Brand tokens as the stylesheet's single source of truth."""
        return {
            **theme_mod.tcss_variable_map(theme_mod.current_theme()),
            **super().get_css_variables(),
        }

    # -- lifecycle ----------------------------------------------------------
    async def on_mount(self) -> None:
        install_markdown_theme()
        try:
            self.console.push_theme(brand_markdown_theme())  # D1 markdown ramp
        except Exception:
            pass  # headless consoles without a pushable theme keep defaults
        self.ansi_theme_dark = _brand_terminal_theme()

        transcript = self.query_one(TranscriptView)
        transcript.set_on_clear(self._on_transcript_cleared)  # TUI-009 hook

        self._status = StatusLine(self.query_one("#status-band"))
        self._status.update(model_label="connecting…", cwd=os.getcwd())
        self.query_one(Editor).focus()

        self._boot_hint = NoticeBlock(BOOT_HINT, "info")
        transcript.append_block(self._boot_hint)

        # Await the session in a worker so the app paints first.
        self.run_worker(self._boot_session(), thread=False, group="session")

    async def _boot_session(self) -> None:
        """Await the session factory; on failure surface + offer /reload."""
        try:
            session = await self._session_factory()
        except Exception as error:  # TUI-012: construction error path
            self._on_boot_failed(error)
            return
        self._session = session
        self._controller = EventController(session, self)
        self._controller.subscribe()
        assert self._status is not None
        self._status.update(model_label=session.model_label)

    def _on_boot_failed(self, error: Exception) -> None:
        self._append_block(NoticeBlock(f"session failed to start: {error}", "error"))
        assert self._status is not None
        self._status.update(model_label="session error", streaming=False)

    async def _reload_session(self) -> None:
        """Dispose the current session (if any) and re-run boot."""
        if self._controller is not None:
            self._controller.dispose()
            self._controller = None
        if self._session is not None:
            try:
                await self._session.dispose()
            except Exception:
                pass
            self._session = None
        assert self._status is not None
        self._status.update(model_label="connecting…", streaming=False)
        await self._boot_session()

    # -- resize (TUI-017 / D5) ----------------------------------------------
    def on_resize(self, event) -> None:  # type: ignore[no-untyped-def]
        """Re-fit width-sensitive chrome after a terminal resize."""
        if self._status is not None:
            self.call_after_refresh(self._status.refresh)

    # -- input --------------------------------------------------------------
    def on_editor_submitted(self, message: EditorSubmitted) -> None:
        """Slash commands run synchronously BEFORE any prompt is sent."""
        text = message.text.strip()
        if not text:
            return
        if text.startswith("/"):
            self._append_block(UserBlock(text))  # D15: echo the command
            self._run_slash_command(text)
            return
        self._submit_prompt(text)

    def on_editor_quit(self, message: EditorQuit) -> None:
        self.exit()

    def on_interrupt_requested(self, message: InterruptRequested) -> None:
        if self._session is not None:
            self._session.abort("interrupted")

    def action_interrupt(self) -> None:
        """App-level Ctrl+C: interrupt the turn, never exit."""
        if self._session is not None:
            self._session.abort("interrupted")

    def action_clear_transcript(self) -> None:
        self._clear_transcript()

    def _clear_transcript(self) -> None:
        transcript = self.query_one(TranscriptView)
        transcript.clear_blocks()  # fires the on_clear hook
        transcript.append_block(NoticeBlock("transcript cleared — history is untouched", "info"))

    def _on_transcript_cleared(self) -> None:
        """TUI-009: /clear and Ctrl+L reset the app's block bookkeeping."""
        if self._working_block is not None:
            self._working_block.stop()
            self._working_block = None
        self._streaming_block = None
        self._tool_cards = {}
        self._boot_hint = None

    async def on_unmount(self) -> None:
        if self._status is not None:
            self._status.dispose()
        if self._controller is not None:
            self._controller.dispose()
        if self._session is not None:
            await self._session.dispose()

    def _submit_prompt(self, text: str) -> None:
        self._append_block(UserBlock(text))
        if self._session is None:
            self._append_block(NoticeBlock("session is still starting…", "warning"))
            return
        session = self._session
        assert self._status is not None
        self._status.update(streaming=True)

        async def run_prompt() -> None:
            try:
                await session.prompt(text)
            except Exception as error:  # surface, never crash the app
                self._append_block(NoticeBlock(str(error), "error"))
            finally:
                # agent_end usually flips this first; a redundant update is a
                # no-op, and this covers sessions that end without agent_end.
                assert self._status is not None
                self._status.update(streaming=False)

        self.run_worker(run_prompt(), thread=False, group="turns")

    # -- transcript helpers ---------------------------------------------------
    def _append_block(self, block) -> None:
        """Append a block, lifting the boot hint on the first real block."""
        if self._boot_hint is not None:
            self.query_one(TranscriptView).remove_block(self._boot_hint)
            self._boot_hint = None
        self.query_one(TranscriptView).append_block(block)

    # -- slash commands -----------------------------------------------------
    def _run_slash_command(self, text: str) -> None:
        """Dispatch a typed slash command (with arguments) to its handler."""
        parts = text.split(maxsplit=1)
        command = parts[0].lower()
        arg = parts[1].strip() if len(parts) > 1 else ""

        def notice(body: str, kind: str = "info") -> None:
            self._append_block(NoticeBlock(body, kind))

        if command in ("/exit", "/quit"):
            self.exit()
        elif command == "/help":
            self._append_block(self._help_block())
        elif command == "/clear":
            self._clear_transcript()
        elif command == "/reload":
            notice("reloading session…")
            self.run_worker(self._reload_session(), thread=False, group="session")
        elif command == "/model":
            self._cmd_model(arg, notice)
        elif command == "/provider":
            self._cmd_providers(notice)
        elif command == "/accounts":
            self._cmd_accounts(notice)
        elif command == "/usage":
            self._cmd_usage(arg, notice)
        elif command == "/compact":
            notice("compaction runs automatically when the context fills up.")
        elif command == "/skills":
            block = self._skills_block()
            if block is not None:
                self._append_block(block)
            else:
                notice("no skills configured.")
        elif command == "/mcp":
            block = self._mcp_block()
            if block is not None:
                self._append_block(block)
            else:
                notice("no MCP servers configured.")
        elif command == "/login":
            self._cmd_login(arg, notice)
        elif command == "/logout":
            self._cmd_logout(arg, notice)
        else:
            notice(f"unknown command: {command} — try /help", "warning")

    # -- model --------------------------------------------------------------
    def _cmd_model(self, arg: str, notice: Callable[[str, str], None]) -> None:
        """``/model`` — show the current spec, or switch with ``provider/id``."""
        session = self._session
        if not arg:
            label = session.model_label if session else "no session"
            notice(f"model: {label}")
            return
        if session is None or not hasattr(session, "set_model"):
            notice("session is still starting…", "warning")
            return
        provider, sep, model_id = arg.partition("/")
        if not sep or not model_id:
            notice(
                "usage: /model <provider>/<model-id> (e.g. openrouter/deepseek/deepseek-chat)",
                "warning",
            )
            return
        provider = provider.lower()  # build_model_spec is case-insensitive
        try:
            spec = self._providers.resolve_model(provider, model_id) if self._providers else None
        except Exception as error:  # unknown provider/hosting
            notice(f"cannot resolve {provider}: {error}", "error")
            return
        if spec is None:
            notice("provider controller unavailable — cannot infer model spec", "warning")
            return
        old_label = session.model_label
        session.set_model(spec)
        if self._status is not None:
            self._status.update(model_label=session.model_label)
        notice(f"model: {old_label} → {session.model_label} (next turn)")
        if old_label.partition("/")[0] != provider:
            notice("switched provider — make sure you are logged in", "warning")

    # -- providers / accounts / usage --------------------------------------
    def _cmd_providers(self, notice: Callable[[str, str], None]) -> None:
        """``/provider`` — list loginable providers and their state."""
        if self._providers is None:
            notice("run: local-operator provider (TUI lacks the provider facade)", "warning")
            return
        try:
            items: list[tuple[str, str]] = []
            for definition in self._providers.login_providers():
                state = "logged in" if self._providers.has_any_credential(definition.id) else "—"
                marker = "*" if definition.store_credentials_as else " "
                items.append((f"{marker}{definition.id}", f"{definition.name} · {state}"))
            use = self._provider_usage_state()
            if use:
                items.append(("usage", ", ".join(use) + " report quota"))
            block = RichBlock(_tree_listing(items)) if items else None
            if block is not None:
                self._append_block(block)
            else:
                notice("no login providers.")
        except Exception as error:  # never crash the app on a provider read
            notice(f"provider list failed: {error}", "error")

    def _provider_usage_state(self) -> list[str]:
        """Provider ids that can report usage, readable by credential state."""
        if self._providers is None:
            return []
        return [
            p
            for p in self._providers.usage_enabled_providers()
            if self._providers.has_any_credential(p)
        ]

    def _cmd_accounts(self, notice: Callable[[str, str], None]) -> None:
        """``/accounts`` — list stored credentials (OAuth + pasted keys)."""
        if self._providers is None:
            notice("run: local-operator login status (TUI lacks the provider facade)", "warning")
            return
        try:
            rows = self._providers.credentials()
            if not rows:
                notice("no stored credentials.")
                return
            now_ms = int(self._clock_ms())
            items: list[tuple[str, str]] = []
            for row in rows:
                identity = (
                    row.identity_key or row.data.get("email") or row.data.get("account_id") or "-"
                )
                if row.credential_type == "oauth":
                    expires = row.data.get("expires")
                    state = "expired" if expires is not None and int(expires) < now_ms else "active"
                    detail = f"oauth · {state} · {identity}"
                else:
                    source = row.data.get("source") or "stored"
                    detail = f"api_key ({source}) · {identity}"
                items.append((f"[{row.id}] {row.provider}", detail))
            block = RichBlock(_tree_listing(items)) if items else None
            if block is not None:
                self._append_block(block)
        except Exception as error:
            notice(f"accounts failed: {error}", "error")

    def _clock_ms(self) -> float:
        import time

        return time.time() * 1000

    def _cmd_usage(self, arg: str, notice: Callable[[str, str], None]) -> None:
        """``/usage [provider]`` — fetch live quota for a provider (or all)."""
        if self._providers is None:
            notice("run: local-operator usage (TUI lacks the provider facade)", "warning")
            return
        target = arg.lower() if arg else ""
        if target and not self._providers.provider(target):
            # A usage-only provider (e.g. zai) has a quota fetcher but no
            # registry login entry — accept it when usage_supported says so.
            from local_operator.providers.usage import usage_supported

            if not usage_supported(target):
                notice(f"unknown provider: {target}", "warning")
                return
        notice("fetching usage…")
        self.run_worker(self._fetch_usage_worker(target or None), thread=False, group="usage")

    async def _fetch_usage_worker(self, provider: str | None) -> None:
        """Worker that fetches usage and posts the result as a block."""

        def notice(body: str, kind: str = "info") -> None:
            self._append_block(NoticeBlock(body, kind))

        try:
            assert self._providers is not None
            reports = await self._providers.fetch_usage([provider] if provider else None)
        except Exception as error:
            notice(f"usage fetch failed: {error}", "error")
            return
        self._append_block(self._usage_block(reports, provider))

    def _usage_block(self, reports, requested: str | None) -> RichBlock:
        """Render one or more usage reports as a compact table."""
        dim = Style(color=theme_mod.semantic_color("dim"))
        muted = Style(color=theme_mod.semantic_color("muted"))
        accent = Style(color=theme_mod.semantic_color("accent"))
        lines: list[Text] = []
        if not reports:
            lines.append(
                Text(
                    "no usage data — this provider has no quota endpoint or no credential",
                    style=dim,
                )
            )
        for report in reports:
            head = Text()
            head.append(report.provider, style=accent)
            if report.identity:
                head.append(f"  ({report.identity})", style=muted)
            head.append(" —", style=dim)
            lines.append(head)
            if report.notes:
                lines.append(Text(f"  {report.notes}", style=dim))
            for limit in report.limits:
                a = limit.amount
                bar = self._usage_bar(a.fraction())
                status_tint = {
                    "ok": theme_mod.semantic_color("success"),
                    "warning": theme_mod.semantic_color("warning"),
                    "exhausted": theme_mod.semantic_color("danger"),
                    "unknown": dim,
                }.get(limit.effective_status(), dim)
                left = Text()
                left.append(bar, style=status_tint)
                label = f" {limit.label}"
                if limit.window:
                    label += f" ({limit.window})"
                if a.unit != "unknown" and a.used is not None:
                    unit = a.unit
                    used = f"{a.used:.2f}" if unit == "usd" else f"{a.used:g}"
                    label += f" — {used} {unit}"
                left.append(label, style=dim)
                lines.append(left)
        return RichBlock(Group(*lines))

    def _usage_bar(self, fraction: float | None, width: int = 12) -> str:
        """A minimal filled/empty bar; unknown fraction renders all-dots."""
        if fraction is None:
            return "·" * width
        fraction = max(0.0, min(1.0, fraction))
        filled = round(fraction * width)
        return "█" * filled + "░" * (width - filled)

    # -- login / logout -----------------------------------------------------
    def _cmd_login(self, arg: str, notice: Callable[[str, str], None]) -> None:
        """``/login [provider]`` — list loginable providers, or run a flow."""
        if self._providers is None:
            if self._login_handler is not None:
                try:
                    self._login_handler("login")
                except Exception as error:  # never let a handler crash the app
                    notice(f"login failed: {error}", "error")
                return
            notice("run: local-operator login", "warning")
            return
        if not arg:
            items = [(p.id, p.name) for p in self._providers.login_providers()]
            self._append_block(RichBlock(_tree_listing(items)))
            return
        provider = arg.lower()
        if self._providers.provider(provider) is None:
            notice(f"unknown provider: {provider}", "warning")
            return
        definition = self._providers.provider(provider)
        if getattr(definition, "login", None) is None:
            notice(f"provider '{provider}' has no interactive login.", "warning")
            return
        notice(f"logging in to {provider}…")
        self.run_worker(self._login_flow(provider), thread=False, group="login")

    async def _login_flow(self, provider: str) -> None:
        """Yield the terminal to the interactive login flow, then report back.

        The flow prints the authorization URL and reads the pasted code/code
        against the real terminal; ``App.suspend()`` hands control back for
        the duration. A lock serializes concurrent /login commands.
        """

        async def notice(body: str, kind: str = "info") -> None:
            self._append_block(NoticeBlock(body, kind))

        assert self._providers is not None
        if self._login_lock is None:
            self._login_lock = _LoginLock()
        if self._login_lock.locked():
            await notice("a login is already in progress.", "warning")
            return
        self._login_lock.acquire()
        try:
            # ``App.suspend`` is a synchronous context manager (Textual 8.x):
            # it yields the terminal synchronously, so it must be entered with
            # a plain ``with`` even though the body awaits (we are already on
            # the event loop inside run_worker). ``async with`` here is a
            # TypeError caught and swallowed by the except below — keep them
            # matched.
            with self.suspend():
                # Inside the suspended terminal, plain print()/input() on the
                # CLI callbacks are safe; the flow drives the loopback server
                # and/or paste prompt entirely on the event loop.
                message = await self._providers.login(provider)
            await notice(message)
        except Exception as error:
            await notice(f"login failed: {error}", "error")
        finally:
            self._login_lock.release()

    def _cmd_logout(self, arg: str, notice: Callable[[str, str], None]) -> None:
        """``/logout [provider]`` — remove stored credentials for a provider."""
        if self._providers is None:
            if self._login_handler is not None:
                try:
                    self._login_handler("logout")
                except Exception as error:  # never let a handler crash the app
                    notice(f"logout failed: {error}", "error")
            else:
                notice("run: local-operator logout <provider>", "warning")
            return
        if not arg:
            notice("usage: /logout <provider>", "warning")
            return
        provider = arg.lower()
        self.run_worker(self._logout_worker(provider), thread=False, group="login")

    async def _logout_worker(self, provider: str) -> None:
        def notice(body: str, kind: str = "info") -> None:
            self._append_block(NoticeBlock(body, kind))

        try:
            assert self._providers is not None
            message = await self._providers.logout(provider)
            notice(message)
        except Exception as error:
            notice(f"logout failed: {error}", "error")

    def _help_block(self) -> RichBlock:
        """Two-column help: command muted padded 10, description dim (D16)."""
        muted = Style(color=theme_mod.semantic_color("muted"))
        dim = Style(color=theme_mod.semantic_color("dim"))
        lines = []
        for command in SLASH_COMMANDS:

            names = ", ".join(f"/{name}" for name in command.names)
            line = Text()
            line.append(names.ljust(14), style=muted)
            line.append(command.description, style=dim)
            lines.append(line)
        return RichBlock(Group(*lines))

    def _skills_block(self) -> RichBlock | None:
        """Graceful introspection of the skills stream (exception-safe)."""
        try:
            from pathlib import Path

            from local_operator.skills.api import default_skill_roots
            from local_operator.skills.discovery import discover_skills

            skills, _warnings = discover_skills(default_skill_roots(Path(os.getcwd())))
            visible = [skill for skill in skills if not skill.hide]
            if not visible:
                return None
            return RichBlock(_tree_listing([(skill.name, skill.description) for skill in visible]))
        except Exception:
            return None

    def _mcp_block(self) -> RichBlock | None:
        """Graceful introspection of the MCP configs (exception-safe)."""
        try:
            from local_operator.mcp.config import load_all_mcp_configs

            configs, _sources = load_all_mcp_configs(os.getcwd())
            if not configs:
                return None
            items: list[tuple[str, str]] = []
            for name, cfg in configs.items():
                detail = getattr(cfg, "command", None) or getattr(cfg, "url", None) or ""
                items.append((name, str(detail)))
            return RichBlock(_tree_listing(items))
        except Exception:
            return None

    # -- event messages (posted by EventController) -------------------------
    def on_start_flush_timer(self, message: StartFlushTimer) -> None:
        """TUI-024: the flush timer starts HERE, on the app thread."""
        if self._controller is not None:
            self._controller.start_flush_timer()

    def on_turn_started(self, message: TurnStarted) -> None:
        assert self._status is not None
        self._status.update(streaming=True)
        # D25: the ONE aggregate working line appears for the turn.
        if self._working_block is None:
            self._working_block = WorkingBlock()
            self._append_block(self._working_block)

    def on_turn_ended(self, message: TurnEnded) -> None:
        assert self._status is not None
        self._dismiss_working_block()
        updates: dict[str, object] = {"streaming": False}
        if message.context_tokens:
            updates["context_tokens"] = message.context_tokens
        cost = self._cost_for(message.usage)
        if cost is not None:
            self._total_cost += cost
            updates["cost"] = format_cost(self._total_cost)
        elif message.usage is not None and getattr(message.usage, "input_tokens", 0):
            # D20: the turn billed tokens but pricing is unknown — render an
            # explicit "unavailable" so the segment's absence reads as that,
            # not as "free".
            updates["cost"] = "$—"
        self._status.update(**updates)
        if message.error:
            self._append_block(NoticeBlock(message.error, "error"))
        elif message.aborted:
            self._append_block(NoticeBlock("interrupted", "warning"))

    def _dismiss_working_block(self) -> None:
        """Stop and remove the aggregate working line at turn end (D25)."""
        if self._working_block is not None:
            self._working_block.stop()
            self.query_one(TranscriptView).remove_block(self._working_block)
            self._working_block = None

    def _cost_for(self, usage) -> float | None:
        """Best-effort turn cost from the registry pricing (never raises)."""
        if usage is None or self._session is None:
            return None
        try:
            from local_operator.model.configure import calculate_cost
            from local_operator.model.registry import get_model_info

            provider, _, model_id = self._session.model_label.partition("/")
            info = get_model_info(provider, model_id)
            if not (info.input_price or info.output_price):
                # No pricing in the registry: a confident $0.0000 would read
                # as "this turn was free" — treat as unknown instead.
                return None
            return calculate_cost(
                info, getattr(usage, "input_tokens", 0), getattr(usage, "output_tokens", 0)
            )
        except Exception:
            return None

    def on_turn_boundary_start(self, message: TurnBoundaryStart) -> None:
        """turn_start: the spinner is already carried by the status band."""

    def on_turn_boundary_end(self, message: TurnBoundaryEnd) -> None:
        """turn_end: reconcile orphaned RUNNING tool cards (TUI-008/019)."""
        for card in list(self._tool_cards.values()):
            card.mark_interrupted()
        self._tool_cards.clear()

    def on_assistant_message_start(self, message: AssistantMessageStart) -> None:
        block = AssistantBlock()
        self._streaming_block = block
        self._append_block(block)

    def on_assistant_delta(self, message: AssistantDelta) -> None:
        if self._streaming_block is not None:
            self._streaming_block.update_text(message.text)

    def on_assistant_message_end(self, message: AssistantMessageEnd) -> None:
        if self._streaming_block is not None:
            # TUI-020: adopt the authoritative text carried by the event.
            self._streaming_block.update_text(message.text)
            self._streaming_block.finalize_text()
            self._streaming_block = None

    def on_tool_started(self, message: ToolStarted) -> None:
        event = message.event
        card = ToolCard(event.tool_call_id, event.tool_name, event.args, event.intent)
        self._tool_cards[event.tool_call_id] = card
        self._append_block(card)

    def on_tool_updated(self, message: ToolUpdated) -> None:
        """TUI-007: stream the partial result into the card's summary."""
        card = self._tool_cards.get(message.event.tool_call_id)
        if card is None:
            return
        detail = _partial_text(message.event.partial_result)
        if detail:
            card.set_partial_detail(detail)

    def on_tool_ended(self, message: ToolEnded) -> None:
        event = message.event
        card = self._tool_cards.pop(event.tool_call_id, None)
        if card is None:
            return
        if event.is_error:
            card.mark_failed(_first_line(event.result.text))
        else:
            card.mark_done()

    def on_notice_posted(self, message: NoticePosted) -> None:
        self._append_block(NoticeBlock(message.text, message.kind))

    def on_compaction_started(self, message: CompactionStarted) -> None:
        self._append_block(NoticeBlock("compacting context…", "info"))

    def on_compaction_ended(self, message: CompactionEnded) -> None:
        if message.success:
            self._append_block(NoticeBlock("context compacted", "info"))
        else:
            self._append_block(NoticeBlock("compaction failed", "error"))

    def on_retry_started(self, message: RetryStarted) -> None:
        body = f"retry {message.attempt}: {message.error}"
        if message.fallback_model:
            body += f" → falling back to {message.fallback_model}"
        self._append_block(NoticeBlock(body, "warning"))

    def on_retry_ended(self, message: RetryEnded) -> None:
        if message.success:
            self._append_block(NoticeBlock("retry succeeded", "info"))
        else:
            self._append_block(NoticeBlock("retry failed", "error"))


def _first_line(text: str) -> str:
    """First non-empty line of a tool result (error summary stays one line)."""
    for line in text.splitlines():
        stripped = line.strip()
        if stripped:
            return stripped
    return ""


def _partial_text(partial_result) -> str:
    """First text run of a streaming tool update."""
    for content in getattr(partial_result, "content", None) or []:
        text = getattr(content, "text", None)
        if text:
            return text
    return ""


def _tree_listing(items: list[tuple[str, str]]) -> Group:
    """Tree-glyph section: ├─ / └─, name in the string tint, detail dim (D4)."""
    if not items:
        return Group()
    name_style = Style(color=theme_mod.semantic_color("string"))
    dim = Style(color=theme_mod.semantic_color("dim"))
    lines = []
    last_index = len(items) - 1
    for index, (name, detail) in enumerate(items):
        branch = "└─ " if index == last_index else "├─ "
        line = Text()
        line.append(branch, style=dim)
        line.append(name, style=name_style)
        if detail:
            line.append("  " + detail, style=dim)
        lines.append(line)
    return Group(*lines)


class _LoginLock:
    """A tiny async-free reentrancy guard for interactive login flows.

    ``App.suspend()`` plus the login callbacks must not be entered twice; a
    boolean plus an acquire/release pair is all the serialization needed
    (the lock is only ever touched from the app's event loop).
    """

    __slots__ = ("_held",)

    def __init__(self) -> None:
        self._held = False

    def acquire(self) -> None:
        self._held = True

    def release(self) -> None:
        self._held = False

    def locked(self) -> bool:
        return self._held


def _hex_to_rgb(value: str) -> tuple[int, int, int]:
    value = value.lstrip("#")
    return (int(value[0:2], 16), int(value[2:4], 16), int(value[4:6], 16))


def _brand_terminal_theme() -> TerminalTheme:
    """Map the ANSI palette onto the brand ramp (no Monokai in this house)."""
    tokens = theme_mod.BRAND_TOKENS["dark"]
    ansi = [
        tokens["bg"],
        tokens["danger"],
        tokens["accent"],
        tokens["amber"],
        tokens["fg"],
        tokens["muted"],
        tokens["dim"],
        tokens["edge"],
    ]
    bright = [
        tokens["string"],
        tokens["danger"],
        tokens["accent"],
        tokens["amber"],
        tokens["fg"],
        tokens["muted"],
        tokens["dim"],
        tokens["surface"],
    ]
    return TerminalTheme(
        _hex_to_rgb(tokens["bg"]),
        _hex_to_rgb(tokens["fg"]),
        [_hex_to_rgb(c) for c in ansi],
        [_hex_to_rgb(c) for c in bright],
    )


async def create_app(
    session_factory: Callable[[], Awaitable[SessionProtocol]],
    theme_name: str = "dark",
    login_handler: Callable[[str], None] | None = None,
    provider_controller: Any | None = None,
) -> OperatorApp:
    """Construct an :class:`OperatorApp` (test/embedding helper)."""
    return OperatorApp(
        session_factory,
        theme_name,
        login_handler=login_handler,
        provider_controller=provider_controller,
    )
