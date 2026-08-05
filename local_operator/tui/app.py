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
from typing import Awaitable, Callable

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

#: Slash commands handled synchronously before any prompt is sent (TUI-014:
#: ``/quit`` is an alias of ``/exit`` — one command, one registry entry).
SLASH_COMMANDS: list[SlashCommand] = [
    SlashCommand("help", "Show available commands"),
    SlashCommand("exit", "Quit the app", aliases=("quit",)),
    SlashCommand("clear", "Clear the transcript (history is untouched)"),
    SlashCommand("reload", "Retry starting the session"),
    SlashCommand("model", "Show the current model"),
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
    ) -> None:
        super().__init__()
        theme_mod.set_theme(theme_name)  # dark is the product's island night
        self._session_factory = session_factory
        self._login_handler = login_handler  # TUI-015: injected by the CLI
        self._session: SessionProtocol | None = None
        self._controller: EventController | None = None
        self._status: StatusLine | None = None
        self._streaming_block: AssistantBlock | None = None
        self._tool_cards: dict[str, ToolCard] = {}
        self._boot_hint: NoticeBlock | None = None
        self._working_block: WorkingBlock | None = None
        self._total_cost: float = 0.0

    # -- composition --------------------------------------------------------
    def compose(self) -> ComposeResult:
        yield TranscriptView()
        # The status line IS the input box's top row (omp trick): the band
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
        parts = text.split(maxsplit=1)
        command = parts[0].lower()

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
            label = self._session.model_label if self._session else "no session"
            notice(f"model: {label}")
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
            if self._login_handler is not None:
                try:
                    self._login_handler("login")
                except Exception as error:
                    notice(f"login failed: {error}", "error")
            else:
                notice("run: local-operator login")
        elif command == "/logout":
            if self._login_handler is not None:
                try:
                    self._login_handler("logout")
                except Exception as error:
                    notice(f"logout failed: {error}", "error")
            else:
                notice("run: local-operator logout")
        else:
            notice(f"unknown command: {command} — try /help", "warning")

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
) -> OperatorApp:
    """Construct an :class:`OperatorApp` (test/embedding helper)."""
    return OperatorApp(session_factory, theme_name, login_handler=login_handler)
