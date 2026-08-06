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
from typing import TYPE_CHECKING, Any, Awaitable, Callable, NamedTuple, Protocol

from local_operator.tui.widgets.welcome import MODEL_PENDING
from local_operator.tui.widgets.transcript import (
    GAP_CLASS,
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

from local_operator.logger import current_log_file
from local_operator.session import naming
from local_operator.session.protocol import SessionProtocol
from local_operator.tui import theme as theme_mod
from local_operator.tui.autocomplete import ArgumentChoice, SlashCommand
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
    ModelQueryOpened,
    Editor,
    EditorQuit,
    EditorSubmitted,
    InterruptRequested,
    ProviderQueryOpened,
)
from local_operator.tui.widgets.model_picker import ModelRow
from local_operator.tui.widgets.status_line import McpStatus, StatusLine, format_cost
from local_operator.tui.widgets.toast import Toast, format_mcp_startup
from local_operator.tui.widgets.tool_card import ToolCard
from local_operator.tui.widgets.welcome import WelcomeView, session_welcome_info

if TYPE_CHECKING:  # keeps the provider graph off the TUI's runtime import path
    from local_operator.providers.controller import CatalogueEntry
    from local_operator.providers.oauth.callback_server import LoginCallbacks

#: Slash commands handled synchronously before any prompt is sent. One
#: registry entry per command; aliases live on the entry (TUI-014).
SLASH_COMMANDS: list[SlashCommand] = [
    SlashCommand("help", "List all commands"),
    SlashCommand("exit", "Quit the app", aliases=("quit",)),
    SlashCommand("clear", "Clear the transcript (history is untouched)"),
    SlashCommand("reload", "Retry starting the session"),
    SlashCommand("model", "Show or switch model (provider/id)", aliases=("models",)),
    SlashCommand("provider", "List providers and their login/usage state"),
    SlashCommand("accounts", "List stored credentials"),
    SlashCommand("usage", "Show provider usage quota"),
    SlashCommand("goal", "Show, set, or clear the session goal"),
    SlashCommand("loop", "Iterate autonomously toward the goal"),
    SlashCommand("compact", "Explain context compaction"),
    SlashCommand("skills", "List loaded skills"),
    SlashCommand("mcp", "List MCP servers"),
    SlashCommand("login", "Authenticate a provider"),
    SlashCommand("logout", "Remove stored provider credentials"),
]


#: ``/loop`` defaults and hard ceiling. A loop spends real tokens per
#: iteration, so it is bounded by construction — an unbounded "keep going"
#: is how an agent burns a budget unattended.
DEFAULT_LOOP_ITERATIONS = 3
MAX_LOOP_ITERATIONS = 25

#: The prompt each loop iteration submits. Deliberately references the
#: standing goal (carried in the system prompt) rather than restating it, so
#: the goal text is never duplicated into the conversation history.
LOOP_PROMPT = (
    "Continue working toward the standing goal. Make concrete progress with "
    "the tools available, then briefly state what advanced and what remains. "
    "If the goal is already fully met, say so plainly and stop."
)

#: How often the band re-counts running background jobs. Nothing emits an
#: event when a job settles, so the subagent segment either polls or goes
#: stale while the user watches it; a 1 Hz pass over a dict of at most a few
#: dozen rows is the cheaper of those two costs. The band is only repainted
#: when the count actually CHANGES.
JOB_POLL_INTERVAL_S = 1.0

#: Class the Screen carries while the session has no content. It selects the
#: boot layout in the stylesheet (centred, clamped input card) and is flipped in
#: exactly one place — see ``OperatorApp._set_welcome_visible``.
BOOT_LAYOUT_CLASS = "boot"

#: Class the Screen carries, on top of ``BOOT_LAYOUT_CLASS``, while the terminal
#: is wide enough for the boot input to read as a CARD rather than as a bar. It
#: is a measurement, not a mode — see ``OperatorApp._sync_boot_card``.
BOOT_CARD_CLASS = "boot-card"

#: The boot card's clamp, duplicated from the stylesheet's
#: ``Screen.boot.boot-card #input-shell`` rule because the app has to know the
#: width the sheet WOULD resolve in order to decide whether to apply it at all.
BOOT_CARD_FRACTION = (7, 10)
BOOT_CARD_MIN_WIDTH = 75

#: Widest the card ever gets. The proportion alone has no upper bound: on a
#: 190-column terminal `70%` resolves to 131 cells, which is a bar again — a
#: borderless surface reads as a card only while the ground beside it is legible
#: as margin, and past a hundred cells the fill is the frame. Still comfortably
#: wider than the widest thing above it (the logo lockup, 59 cells).
BOOT_CARD_MAX_WIDTH = 100

#: Smallest total inset (both sides together) that reads as a card. Below it the
#: panel keeps the full width: 1 to 3 cells of ground beside a borderless fill
#: reads as a misaligned full-width bar, because there is no edge to attribute
#: the offset to. 8 cells is 4 a side — four times the app's own gutter, which is
#: the smallest offset that is unambiguously a margin.
BOOT_CARD_MIN_INSET = 8

#: Spare rows the boot composition needs before it spends any on chrome: one for
#: the ground row above the card, and at least one still unspent — otherwise the
#: separator comes straight out of the splash, whose degradation ladder pays in
#: whole sections (the mark, the hints), never in single rows.
BOOT_COMPOSITION_MIN_SPARE = 2


def boot_card_width(box: int) -> int:
    """Width the card's clamp resolves to inside a content box of ``box`` cells.

    The stylesheet's three declarations, in one expression, so the app can ask what
    the sheet would do before deciding whether to let it: ``min(box, cap,
    max(floor, proportion))``. The box wins over the floor, which is what keeps a
    20-cell terminal from being handed a 75-cell panel.
    """
    numerator, denominator = BOOT_CARD_FRACTION
    proportion = box * numerator // denominator
    return min(box, BOOT_CARD_MAX_WIDTH, max(BOOT_CARD_MIN_WIDTH, proportion))


class NoticeFn(Protocol):
    """The `notice` callback every slash-command handler is handed.

    Declared as a Protocol rather than ``Callable[[str, str], None]`` because
    the real closures default ``kind`` — a plain two-positional Callable makes
    every ``notice("...")`` call site a type error while the code is correct.
    """

    def __call__(self, body: str, kind: str = "info") -> None: ...


class _ProviderRows(NamedTuple):
    """The provider list's rows, plus what to say when it could not be built.

    Two fields rather than an exception or a bare empty list, because "there is
    nothing to log out of" and "the credential store cannot be read" produce the
    same empty list and are not the same news. The caller says exactly one of
    them, so the user is never told they have no credentials by a store that
    never answered.
    """

    choices: list[ArgumentChoice]
    problem: str


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
        provider_controller: Any | None = None,
    ) -> None:
        super().__init__()
        theme_mod.set_theme(theme_name)  # dark is the product's island night
        self._session_factory = session_factory
        # Full provider/model/credential/usage facade behind the slash
        # commands; ``None`` degrades /provider /usage /model-switch to
        # pointer notices when it is absent.
        self._providers = provider_controller
        self._session: SessionProtocol | None = None
        self._controller: EventController | None = None
        self._status: StatusLine | None = None
        self._streaming_block: AssistantBlock | None = None
        self._tool_cards: dict[str, ToolCard] = {}
        self._welcome: WelcomeView | None = None
        self._working_block: WorkingBlock | None = None
        self._total_cost: float = 0.0
        # Auto-naming fires ONCE per session. Latched here rather than on the
        # session holder because the app is what schedules the call, and a
        # second submit arriving while the first title is still in flight
        # must not queue a second provider request.
        self._name_requested: bool = False
        # Last subagent count painted, so the 1 Hz poll repaints only on a
        # real change instead of every tick.
        # (task jobs, bash jobs) last painted, so a repaint only happens on change.
        self._subagents_shown: tuple[int, int] = (0, 0)
        # Serializes interactive login flows so two /login commands can never
        # race the one suspended terminal.
        self._login_lock: Any | None = None
        # ``/loop`` state: one loop at a time, cooperatively cancellable at
        # the turn boundary (never mid-turn, so a turn is never half-applied).
        self._loop_running: bool = False
        self._loop_cancelled: bool = False

    # -- composition --------------------------------------------------------
    def compose(self) -> ComposeResult:
        # The welcome splash is the transcript's EMPTY STATE, so it is mounted
        # INSIDE the transcript rather than beside it: that hands it exactly the
        # region above the input panel with no arithmetic here. It supersedes the
        # old D9 boot-hint line, which was a real transcript block and would have
        # hidden the splash on mount.
        with TranscriptView():
            yield WelcomeView(lambda: session_welcome_info(self._session, self._providers))
        # Two containers for one panel: the dock is the docked POSITIONER, and
        # the shell is the panel the user sees — the fill, the padding, and the
        # boot layout's clamp. A docked widget cannot be centred by its parent,
        # so the clamp has to sit on a child of the dock rather than on the dock
        # itself; see the tcss.
        #
        # The status line IS the input box's last row: the band docks at the
        # bottom of the shell and carries the structural rule styling, so it can
        # never be overdrawn or pushed off-screen by the editor, and it travels
        # with the input when the panel becomes a card. One row does double duty
        # — zero extra height (D3/D17).
        with Container(id="input-dock"):
            with Container(id="input-shell"):
                yield Static(id="status-band")
                editor = Editor(commands=SLASH_COMMANDS)
                with Horizontal(id="input-row"):
                    yield Static("❯", id="prompt-chevron")
                    yield editor
                # The picker is the editor's, but it cannot be the editor's
                # CHILD: it has to draw across the full panel width, outside the
                # chevron row. Mounted here it lands between the input row and
                # the bottom-docked status band — under the text it completes,
                # above the footer — and it claims zero rows while closed.
                yield editor.picker
                # Same placement rule, same reason. The two are mutually
                # exclusive — the buffer parse that opens one closes the other —
                # so they can share the row band without ever competing for it.
                yield editor.model_picker
        # The toast slot lives on its own CSS layer (see the tcss), so it
        # overlays the transcript's top-right corner without taking a row from
        # it. Mounted once and kept: showing a message never has to await a
        # mount, and there is no window where two cards exist at once.
        with Container(id="toast-host"):
            yield Toast()

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
        # Cached: every appended block asks the splash to hide, and that path
        # should not pay for a DOM query per block.
        self._welcome = self.query_one(WelcomeView)
        # An empty transcript is the boot layout's whole precondition, and the
        # session starts empty — so the app opens in it. Set here rather than in
        # the stylesheet's base rules because this is state, not style: it is the
        # same flag `_set_welcome_visible` flips for every later transition.
        self._set_welcome_visible(True)

        self._status = StatusLine(self.query_one("#status-band", Static))
        self._status.update(model_label=MODEL_PENDING, cwd=os.getcwd())
        editor = self.query_one(Editor)
        # Installed here rather than in the Editor's constructor: the editor is
        # built inside `compose`, before the app has anything for it to call, and a
        # widget that reached back into its host would invert the dependency this
        # whole module is arranged around.
        editor.set_model_handler(self._on_model_row_chosen)
        editor.focus()
        # The count has no event to hang off (see JOB_POLL_INTERVAL_S).
        self.set_interval(JOB_POLL_INTERVAL_S, self._poll_subagents)

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
        self._status.update(
            model_label=session.model_label,
            effort=_effort_label(session),
            context_window=_context_window(session),
            conversation_name=session.conversation_name,
        )
        self._wire_mcp_status(session)
        self._report_mcp_startup(session)

    def _on_boot_failed(self, error: Exception) -> None:
        """Report a session that never constructed, WITHOUT retiring the splash.

        A :meth:`_system_notice`, by the same predicate every other
        infrastructure report uses: the conversation has not started just because
        the harness has something to say. Here that matters more than anywhere
        else — the splash is where the credential warning and the boot hints
        live, so a plain ``_append_block`` retired the one block that tells the
        user what to do next at exactly the moment they need it, leaving a single
        red line over an empty screen.
        """
        self._system_notice(f"session failed to start: {error}", "error")
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
        # A reload is a new conversation: its title and its one naming
        # attempt both reset, or the old name would outlive its session.
        self._name_requested = False
        # The MCP segment is cleared too: the old session's manager is gone, so
        # a lingering count would describe servers nothing is connected to any
        # more. _boot_session repaints it from the new session's manager.
        self._status.update(
            model_label=MODEL_PENDING,
            streaming=False,
            effort="",
            conversation_name="",
            mcp=McpStatus(),
        )
        await self._boot_session()

    # -- MCP status ---------------------------------------------------------
    def _wire_mcp_status(self, session: SessionProtocol) -> None:
        """Paint the band's MCP segment and keep it LIVE for the session.

        Deliberately better than the reference here: OpenCode snapshots its MCP
        count at boot and never revisits it, so a server that dies leaves the
        indicator claiming it is still there. ``set_on_tools_changed`` fires on
        connect, disconnect and list-changed, which is exactly the set of events
        that can move the count.

        With NO manager there is nothing to subscribe to, but the segment is still
        painted once: discovery may have failed, and that state comes from the boot
        record rather than from the manager that does not exist. Returning without
        painting left the band identical to a machine that never configured MCP.
        """
        manager = getattr(session, "mcp_manager", None)
        if manager is None:
            self._refresh_mcp_status()
            return
        # CHAIN, never replace: the incumbent callback is the composition root's,
        # and it is what keeps the agent's tool inventory in step with MCP state.
        # Clobbering it would freeze the tool list at whatever booted, which is a
        # far worse bug than a stale counter. See McpManager.on_tools_changed.
        incumbent = manager.on_tools_changed

        def on_tools_changed(tools: list[Any]) -> Any:
            # The refresh is scheduled in a `finally`, so the band is repainted
            # even when the incumbent raises. `refresh_tools` is not documented
            # infallible and `McpManager._fire_tools_changed` swallows and logs
            # whatever comes out of here — so without this the one event that
            # moves the count would leave the band asserting a count that is no
            # longer true, which is the exact staleness the live segment exists
            # to remove. Ordering costs nothing: `call_later` only queues.
            # The incumbent's return value is handed straight back so an async
            # incumbent is still awaited by the manager (it task-ifies a returned
            # coroutine).
            try:
                return incumbent(tools) if incumbent is not None else None
            finally:
                # Hop onto the message pump rather than mutating widgets from the
                # manager's task: this module's whole arrangement is that widget
                # mutation happens on the Textual thread.
                self.call_later(self._refresh_mcp_status)

        manager.set_on_tools_changed(on_tools_changed)
        self._refresh_mcp_status()

    def _refresh_mcp_status(self) -> None:
        """Re-read the manager and repaint the band's MCP segment."""
        if self._status is not None:
            self._status.update(mcp=self._mcp_status())

    def _mcp_status(self) -> McpStatus:
        """The band's MCP segment state, read LIVE from the manager.

        Live per-server state is taken from the manager and nothing else. A boot
        failure that later reconnects must clear the danger tint, and a record of
        what happened at startup can never say that — the manager's per-server
        status is the only thing that knows the current truth. ``connecting`` is
        not a failure: the startup gate leaves slow servers in that state on every
        launch, and tinting it danger would make a red lamp the normal boot.

        The ONE thing the manager cannot report is its own absence. When discovery
        raised, there is no manager and no server list, so a bare ``McpStatus()``
        would render exactly like a machine that never configured MCP — and the
        toast saying so dismisses itself after ten seconds. That single fact comes
        from the boot record, which is the only thing that knows it.

        Exception-safe like its sibling ``_mcp_block``: three manager methods are
        called here, none of them ours, and this runs from a manager callback on
        every tools-changed event. A raise would take out the repaint of a band
        that reports nine other segments — an empty MCP segment is a far cheaper
        failure than a frozen status line.
        """
        manager = getattr(self._session, "mcp_manager", None)
        if manager is None:
            startup = getattr(self._session, "mcp_startup", None)
            return McpStatus(discovery_failed=bool(getattr(startup, "failed", False)))
        try:
            configured = manager.get_all_server_names()
            return McpStatus(
                configured=len(configured),
                connected=len(manager.get_connected_servers()),
                failed=any(
                    manager.get_connection_status(name) == "disconnected" for name in configured
                ),
            )
        except Exception:
            return McpStatus()

    def _report_mcp_startup(self, session: SessionProtocol) -> None:
        """Raise the startup toast, and leave a DURABLE record of any failure.

        The toast dismisses itself, which makes it a notification and not a
        record. So a failure also lands in the transcript as a notice the user
        can scroll back to, and ``/mcp`` reports per-server state on demand. The
        toast is the interruption; those two are the evidence.

        Successes get no notice — a transcript line per launch saying everything
        worked is exactly the log-spam the borderless/quiet mandate exists to
        prevent, and the band's count already says it.

        None of the three ends the empty state. The session has not started
        talking just because a server failed to start, and collapsing the boot
        composition on launch would mean a user with one broken server never saw
        the centred prompt — while the toast has already interrupted them with
        the same failure. So the record goes through :meth:`_system_notice`: it
        lands under the splash, survives the toast, and is there when the
        conversation does begin.
        """
        outcome = getattr(session, "mcp_startup", None)
        if outcome is None:
            return
        toast = self.query_one(Toast)
        payload = format_mcp_startup(outcome, max_cells=toast.content_cells)
        if payload is None:
            return
        text, duration_ms = payload
        toast.show(text, duration_ms=duration_ms)
        # No "server" in the wording: one failure key is ``discovery`` (the
        # config layer itself), and "MCP server discovery failed" would name a
        # server that does not exist.
        for name, error in sorted(outcome.failures.items()):
            self._system_notice(f"MCP {name} failed: {error}", "error")

    # -- resize (TUI-017 / D5) ----------------------------------------------
    def on_resize(self, event) -> None:  # type: ignore[no-untyped-def]
        """Re-fit width-sensitive chrome after a terminal resize."""
        if self._status is not None:
            self.call_after_refresh(self._status.refresh)
        # The EVENT's width, not the app's: during a resize `self.size` is still the
        # previous frame's, and one stale cell is enough to put the card threshold on
        # the wrong side of itself — at 85 columns it decided "bar" for a box that was
        # about to be exactly wide enough for the card.
        self._sync_boot_layout(width=event.size.width)

    def _sync_boot_layout(self, *, width: int | None = None) -> None:
        """Re-measure everything about the boot layout that the sheet cannot.

        Called from ``on_resize`` and from ``_set_welcome_visible``: a resize is not
        the only way the answers change — the layout itself comes and goes with the
        splash, and on the first frame no resize event has happened yet. ``width``
        is the resize event's, when there is one.
        """
        self._sync_boot_card(self.size.width if width is None else width)
        # The card's width is a function of the terminal, so it resolves from the
        # event itself. The composition is a function of how tall the SPLASH turned
        # out, which is only knowable from a laid-out frame — and neither caller has
        # one yet (a mount has not arranged, a resize is arranging). So it measures
        # one refresh later, and re-measures itself until it stops changing.
        self.call_after_refresh(self._sync_boot_composition)

    def _sync_boot_card(self, terminal_width: int) -> None:
        """Decide whether this terminal is wide enough for the boot CARD.

        The stylesheet cannot ask how wide the terminal is, so the width the card
        WOULD take is resolved here and the class only goes on when the ground left
        beside it is wide enough to read as a margin. Computing a width in Python
        for want of a media query is the same move ``Toast._refit`` already makes.
        """
        # The content box, not the terminal: `Screen`'s one-cell inset is outside
        # the percentage the sheet resolves.
        box = max(0, terminal_width - 2)
        card = boot_card_width(box)
        self.screen.set_class(box - card >= BOOT_CARD_MIN_INSET, BOOT_CARD_CLASS)

    def _sync_boot_composition(self) -> None:
        """Centre the boot composition — splash, separator, card — in the screen.

        The splash and the card are ONE block to the eye, and on a tall terminal
        resting that block on the bottom of the screen left the upper two thirds
        empty. The stylesheet cannot fix it: the card is DOCKED, so no alignment
        reaches it, and how many rows to reserve below it depends on how tall the
        splash turned out at this width. So the composition's two chrome
        quantities — the ground row above the card and the slack below it — are
        measured here, the same way ``Toast._refit`` measures a width the sheet
        cannot ask for.

        Both are CONDITIONAL, and on the same measurement. Every row this reserves
        comes out of the splash's budget (WelcomeView.get_content_height), and the
        splash pays in whole sections — a 28-row terminal that reserved rows to
        centre a block that already fills the region would trade the mark for air.
        So they are only spent out of rows that are empty anyway, which is why a
        96x28 frame is untouched and a 190x48 one is centred.

        The spare count adds back what this method has already spent, which makes
        it invariant under its own change: re-running converges instead of halving
        the reserve every pass.
        """
        dock = self.query_one("#input-dock", Container)
        welcome = self._welcome
        if not self.screen.has_class(BOOT_LAYOUT_CLASS) or welcome is None or not welcome.display:
            # The conversation layout is a full-width bar with nothing under it.
            # Rows left reserved here would be a hole below a populated transcript.
            self._reserve_boot_rows(dock, gap=False, lift=0)
            return
        region = self.query_one(TranscriptView).content_region
        # The empty rows ABOVE the block, read off the frame rather than derived:
        # the region's content is bottom-aligned, so the block's own offset inside
        # it IS the slack — and unlike `region.height - welcome.height` it does not
        # count a notice mounted under the splash as spare.
        spare = welcome.region.y - region.y + dock.styles.padding.bottom
        if dock.has_class(GAP_CLASS):
            spare += 1
        if spare < BOOT_COMPOSITION_MIN_SPARE:
            self._reserve_boot_rows(dock, gap=False, lift=0)
            return
        self._reserve_boot_rows(dock, gap=True, lift=(spare - 1) // 2)

    def _reserve_boot_rows(self, dock: Container, *, gap: bool, lift: int) -> None:
        """Apply the composition's separator row and its lift, once.

        The separator is the app's ONE vertical separator class rather than a second
        spacing rule: a receipt under the splash and the card under the hints are
        the same rhythm case, one row of ground where a block change needs reading
        as a block change. The lift is padding INSIDE the dock — the dock is the
        positioner, so reserving rows in it moves the panel it holds without
        anything else in the layout knowing.
        """
        if dock.has_class(GAP_CLASS) == gap and dock.styles.padding.bottom == lift:
            return
        dock.set_class(gap, GAP_CLASS)
        dock.styles.padding = (0, 0, lift, 0)
        # The measurement above read a laid-out height that this change invalidates,
        # so it runs again once the new frame exists. The equality gate above is
        # what ends the chain — the total it measures does not move, so the second
        # pass agrees with the first.
        self.call_after_refresh(self._sync_boot_composition)

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
        self._interrupt()

    def action_interrupt(self) -> None:
        """App-level Ctrl+C: interrupt the turn, never exit."""
        self._interrupt()

    def _interrupt(self) -> None:
        """Abort the running turn AND stop any ``/loop`` in flight.

        Without cancelling the loop, an interrupt would abort one turn and the
        loop would immediately submit the next — the user would have to press
        Ctrl+C once per remaining iteration to actually stop.
        """
        if self._loop_running:
            self._loop_cancelled = True
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
        # An empty transcript is the welcome view's whole precondition, so the
        # clear hook is also what brings it back — and with it the boot layout,
        # since `_set_welcome_visible` drives both from this one condition. One
        # mechanism for both directions rather than a second "should the splash
        # show" rule that could disagree with this one. The "transcript cleared"
        # notice appended right after this lands UNDER the splash (it goes
        # through TranscriptView.append_block, not _append_block), which is why
        # the receipt for the action survives alongside the restored splash.
        self._set_welcome_visible(True)

    def _set_welcome_visible(self, visible: bool) -> None:
        """Show or hide the splash AND swap the input layout with it.

        Both ride ONE condition — "the transcript has no content" — because two
        would eventually disagree, and the way that failure looks is a centred
        boot card sitting under a populated transcript. The layouts themselves
        live in the stylesheet (`Screen.boot`); this only flips the flag that
        selects between them, so there is no second layout written in Python to
        keep in step with the first.

        The width class and the vertical reserve are re-measured here rather than
        being second conditions: they answer "how wide is this terminal" and "how
        many rows are going spare", which are facts about the frame, not about the
        session — and they have to be right on the FIRST boot frame, before any
        resize event has happened.
        """
        if self._welcome is not None:
            self._welcome.set_visible(visible)
        self.screen.set_class(visible, BOOT_LAYOUT_CLASS)
        self._sync_boot_layout()

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
        # Detached, and deliberately AFTER the turn is dispatched: the title
        # is decoration, and decoration must never sit in front of the user's
        # first reply.
        self._maybe_name_conversation(text)

    # -- conversation naming --------------------------------------------------
    def _maybe_name_conversation(self, text: str) -> None:
        """Schedule the one auto-naming call for this conversation.

        Skipping a low-signal opener does NOT spend the attempt: "hi" is
        usually followed by the actual request, and latching on the greeting
        would leave the conversation permanently unnamed.
        """
        session = self._session
        if session is None or self._name_requested:
            return
        if session.conversation_name:
            return  # already named: a restored session, or an explicit rename
        if naming.is_low_signal(text):
            return
        self._name_requested = True
        self.run_worker(self._name_conversation_worker(session, text), thread=False, group="naming")

    async def _name_conversation_worker(self, session: SessionProtocol, text: str) -> None:
        """Await the title off the turn's path and paint it when it lands.

        ``generate_title`` absorbs every failure and bounds its own wait, so
        there is nothing to catch here — a provider that raises or hangs just
        leaves the band nameless, which is the intended degradation.
        """
        title = await naming.generate_title(text, session.complete_once)
        if not title or session is not self._session:
            return  # no title, or the session was reloaded out from under it
        stored = session.set_conversation_name(title, user_set=False)
        if self._status is not None:
            self._status.update(conversation_name=stored)

    # -- background jobs ------------------------------------------------------
    def _poll_subagents(self) -> None:
        """Repaint the counter segments only when a running count changes."""
        agents = self._job_count("task")
        jobs = self._job_count("bash")
        if (agents, jobs) == self._subagents_shown:
            return
        self._subagents_shown = (agents, jobs)
        if self._status is not None:
            self._status.update(subagents=agents, jobs=jobs)

    def _job_count(self, kind: str) -> int:
        """Running jobs of one ``kind`` — ``task`` (subagents) or ``bash``.

        The two are counted separately because they are different things an
        operator tracks: a subagent is delegated reasoning with no other
        representation on screen, while a backgrounded shell command already has
        a tool card. Summing them would hide which kind is running.

        ``queued`` is excluded to match ``AsyncJobManager``'s own running count
        (``harness/jobs.py``): a job admitted to the ledger but held behind the
        capacity gate carries ``status == "running"`` and has not started, so
        counting it would report work that is not yet happening — and disagree
        with the number the harness itself reports.

        Never raises: a status segment must not be able to take the app down.
        """
        manager = getattr(self._session, "jobs", None)
        if manager is None:
            return 0
        try:
            return sum(
                1
                for job in manager.list()
                if job.status == "running"
                and job.type == kind
                and not getattr(job, "queued", False)
            )
        except Exception:
            return 0

    # -- transcript helpers ---------------------------------------------------
    def _append_block(self, block, *, ends_empty_state: bool = True) -> None:
        """Append a block, retiring the welcome view — and the boot layout — on
        the first one. That is the authoritative "the session has content" edge;
        both layouts hang off it (see `_set_welcome_visible`).

        ``ends_empty_state=False`` appends WITHOUT ending it, because the
        predicate is "the CONVERSATION has started", not "something got drawn". A
        system notice about infrastructure — an MCP server that failed to
        connect — is not conversation content, and letting one collapse the boot
        composition would mean anyone with a single broken server never saw the
        centred prompt at all. The notice is still appended and still scrolls
        back; it simply lands under the splash, exactly as the ``/clear`` receipt
        does.
        """
        if ends_empty_state:
            self._set_welcome_visible(False)
        self.query_one(TranscriptView).append_block(block)

    # -- slash commands -----------------------------------------------------
    def _notice(self, body: str, kind: str = "info") -> None:
        """Append a notice block.

        A METHOD rather than the local closure it used to be, because the picker's
        worker and its choose-callback both need to report and neither runs inside
        a dispatch. One implementation means every path renders notices the same.
        """
        self._append_block(NoticeBlock(body, kind))

    def _system_notice(self, body: str, kind: str = "info") -> None:
        """A notice about the HARNESS that leaves the empty state intact.

        Separate from :meth:`_notice` because the two answer different questions.
        ``_notice`` reports on something the user just did, so the conversation
        has started by definition. This one reports on infrastructure the user
        did not ask about — an MCP server that failed to connect, a provider
        controller that is missing — and the session has not started talking just
        because a subsystem announced itself. Routing those through ``_notice``
        collapsed the boot composition on launch for anyone with one broken
        server, which is how the centred prompt became unreachable.
        """
        self._append_block(NoticeBlock(body, kind), ends_empty_state=False)

    def _editor(self) -> Editor:
        """The input editor. Queried rather than held: Textual owns the widget."""
        return self.query_one(Editor)

    def _run_slash_command(self, text: str) -> None:
        """Dispatch a typed slash command (with arguments) to its handler."""
        parts = text.split(maxsplit=1)
        command = parts[0].lower()
        arg = parts[1].strip() if len(parts) > 1 else ""
        notice = self._notice

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
        elif command == "/goal":
            self._cmd_goal(arg, notice)
        elif command == "/loop":
            self._cmd_loop(arg, notice)
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
    def _cmd_model(self, arg: str, notice: NoticeFn) -> None:
        """``/model`` — open the picker; ``/model provider/id`` — switch directly.

        A bare ``/model`` OPENS THE LIST rather than printing the current label.
        Printing it was a dead end: the answer to "which model am I on" is already
        on the status band, and the question a user actually has at that moment is
        "which models can I switch to", which they had no way to ask. The label is
        still reported, as the picker's current-row marker.
        """
        session = self._session
        if not arg:
            self._open_model_picker()
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
        if self._providers is None:
            notice("provider controller unavailable — cannot infer model spec", "warning")
            return
        # Validate the provider BEFORE switching. resolve_model does not raise
        # on an unknown provider — it returns a spec with base_url=None — so a
        # typo would silently reconfigure the session and only fail on the next
        # turn, reading as a network/auth error instead of a typo.
        if self._providers.provider(provider) is None:
            notice(f"unknown provider: {provider} — see /provider", "warning")
            return
        try:
            spec = self._providers.resolve_model(provider, model_id)
        except Exception as error:  # unresolvable hosting/model pair
            notice(f"cannot resolve {provider}: {error}", "error")
            return
        old_label = session.model_label
        session.set_model(spec)
        if self._status is not None:
            # The window and the effort belong to the SPEC, not the session:
            # a switch that repainted only the label would leave the context
            # percentage measured against the previous model's window.
            self._status.update(
                model_label=session.model_label,
                effort=_effort_label(session),
                context_window=_context_window(session),
            )
        suffix, warning = self._model_access_note(provider)
        notice(f"model: {old_label} → {session.model_label} (next turn){suffix}")
        if warning:
            notice(warning, "warning")

    def _model_access_note(self, provider: str) -> tuple[str, str | None]:
        """``(suffix, warning)`` answering "can I actually run this model now".

        Replaces "switched provider — make sure you are logged in", which asked
        the user to go and check something the app already knew, fired on every
        provider change including the ones that were fine, and — because it was
        the last thing on screen when the next turn died on an unrelated HTTP 400
        — read as the diagnosis for a failure that had nothing to do with auth.

        Credential state ONLY. Proving access with a live completion would spend
        money and a second of latency on a keystroke, and the turn the user is
        about to send is the real proof anyway: when that fails, the transcript
        prints the provider's own error (:meth:`_submit_prompt`), which is the
        message worth reading.

        The three states are :meth:`_credential_state`'s, not a fourth vocabulary
        invented here, so `/model`, `/provider` and the `/login` picker keep
        describing one situation with one set of words.
        """
        providers = self._providers
        assert providers is not None
        try:
            state = self._credential_state(provider, providers.has_any_credential(provider))
        except Exception as error:  # the STORE failed, not the provider
            # Named rather than swallowed or guessed at: "I could not check" is a
            # real answer, where a confirmation would claim access nobody verified.
            return "", f"cannot check {provider} credentials: {error}"
        if state == "needs login":
            return "", f"{provider} needs login — /login {provider}"
        return f" · {provider} {state}", None

    # -- model picker -------------------------------------------------------
    def on_model_query_opened(self, message: ModelQueryOpened) -> None:
        """The buffer just entered ``/model …`` — fill the list.

        Populating on the MESSAGE rather than only from the command handler is what
        makes every route into the list identical: typing `/model ` by hand, being
        completed into it by the command picker, or dispatching `/model` all end up
        here. Before this, only the dispatched route had rows.
        """
        message.stop()
        self._populate_model_picker()

    def _open_model_picker(self) -> None:
        """Reopen the model list after ``/model`` was dispatched as a command.

        Writes the BUFFER rather than opening the widget, because the buffer is the
        single authority on which picker is showing — that is what keeps the two
        lists mutually exclusive without either knowing about the other, so a picker
        opened behind its back would be closed again by the next keystroke's resync.
        It also leaves the query editable, which is the point: the user keeps typing
        to filter.

        Writing it posts ``ModelQueryOpened``, which is what fills the rows, so this
        method deliberately does nothing else. It exists because the keystroke route
        (the command picker completes `/model ` and stops) never clears the buffer,
        while a dispatched `/model` does — and the list has to come back.
        """
        self._editor().begin_model_query()

    def _populate_model_picker(self) -> None:
        """Paint the catalogue immediately, then refresh it from the providers.

        Stale-then-update, not load-then-show. The shipped registry is already in
        memory, so the list appears on the keystroke that asked for it; a spinner
        over an empty list would be slower AND less useful, because the model the
        user wants is usually one we already know about. The live fetch then adds
        what shipped too late to be in the registry — which is the whole reason this
        exists: after logging in to Anthropic, `claude-opus-5` has to be findable
        without the user already knowing its id.
        """
        rows, note = self._catalogue_rows(
            self._providers.static_catalogue() if self._providers else []
        )
        self._editor().model_picker.set_rows(
            rows,
            current=self._current_selector(),
            # The access note leads. The footer truncates at the picker's width and
            # every other clause is background — "cached: anthropic, openrouter,
            # radient" pushed `/login <provider>` off the end at 100 columns, which
            # cost the one clause the user can act on.
            status=_status_line(note, "checking providers…" if self._providers else ""),
        )
        if self._providers is not None:
            self.run_worker(self._refresh_catalogue(), thread=False, group="catalogue")

    async def _refresh_catalogue(self) -> None:
        """Worker: replace the picker's rows with the live catalogue."""
        if self._providers is None:
            return
        try:
            entries, statuses = await self._providers.live_catalogue()
        except Exception as error:  # noqa: BLE001 — a picker must not take the app down
            rows, note = self._catalogue_rows(self._providers.static_catalogue())
            self._editor().model_picker.set_rows(
                rows,
                current=self._current_selector(),
                status=_status_line(note, f"live model list unavailable: {error}"),
            )
            return
        rows, note = self._catalogue_rows(entries)
        self._editor().model_picker.set_rows(
            rows,
            current=self._current_selector(),
            status=_status_line(note, _catalogue_status(statuses)),
        )

    def _catalogue_rows(self, entries: list["CatalogueEntry"]) -> tuple[list[ModelRow], str]:
        """``(rows, note)`` — the models this user can actually run, and what was cut.

        HIDDEN, not demoted. The list used to be the whole registry with the
        unreachable models tinted and sorted last, and the demotion still lost:
        the window shows fourteen rows, so `/model opus` filled the screen with
        four providers' opus rows of which exactly one could run, and every miss
        cost a keystroke that turned into a login prompt instead of a switch. A
        picker is a list of choices; a row that cannot be chosen is not one.

        Discoverability is what the old behaviour was protecting, so the count
        goes to the footer instead of into the rows: "42 hidden — /login
        <provider>" answers "is there more" without spending the visible list on
        models the user cannot use.

        Two rows survive the filter regardless. The session's CURRENT model stays,
        because its `●` is what answers "what am I on" and dropping it would make a
        broken configuration invisible rather than obvious. And when the credential
        store cannot be read at all, EVERY row stays: an empty picker claims the
        user owns no models, which is precisely what the app failed to find out.
        """
        usable = self._usable_providers()
        current = self._current_selector()
        rows = [
            ModelRow(
                provider=entry.provider,
                model_id=entry.model_id,
                label=entry.label,
                context_window=entry.context_window,
                input_price=entry.input_price,
                output_price=entry.output_price,
                connected=entry.connected,
                aggregated=entry.aggregated,
            )
            for entry in entries
            if usable is None or entry.provider in usable or entry.selector == current
        ]
        if usable is None:
            return rows, "credential check unavailable — showing every model"
        hidden = len(entries) - len(rows)
        return rows, (f"{hidden} hidden — /login <provider>" if hidden else "")

    def _usable_providers(self) -> set[str] | None:
        """Providers a turn could run on, or ``None`` when that cannot be read.

        Re-read on every populate rather than cached, which is what makes a login
        take effect without a restart: `/login anthropic` stores a credential and
        the next `/model` rebuilds its rows from this call.

        Guarded even though the controller already swallows a store failure — the
        facade is duck-typed and an embedding host supplies its own — because the
        one thing this must never do is take the picker down with it.
        """
        providers = self._providers
        if providers is None:
            return None
        try:
            return providers.usable_providers()
        except Exception:  # noqa: BLE001 — a credential read never costs the list
            return None

    def _current_selector(self) -> str | None:
        """The session's model as ``provider/id``, or None before it starts."""
        session = self._session
        return session.model_label if session is not None else None

    def _on_model_row_chosen(self, row: ModelRow) -> None:
        """Switch to a chosen model, or start a login when it needs one.

        The two outcomes share one keystroke on purpose. A user who opens the list
        to find a model they cannot yet run has already told us what they want; the
        useful response is to begin the login, not to refuse and make them retype
        the provider name into a different command.
        """
        notice = self._notice
        if not row.connected:
            notice(f"{row.provider} needs a login first — starting it now", "warning")
            self._cmd_login(row.provider, notice)
            return
        self._cmd_model(row.selector, notice)

    # -- goal / loop --------------------------------------------------------
    def _cmd_goal(self, arg: str, notice: NoticeFn) -> None:
        """``/goal`` — show; ``/goal <text>`` — set; ``/goal clear`` — unset.

        The goal is a standing objective carried in the prompt's volatile
        tail, so it survives every turn (and compaction) without being
        re-typed, and ``/loop`` uses it as the thing to iterate toward.
        """
        session = self._session
        if session is None or not hasattr(session, "set_goal"):
            notice("session is still starting…", "warning")
            return
        if not arg:
            current = session.goal
            notice(f"goal: {current}" if current else "no goal set — /goal <text> to set one")
            return
        if arg.lower() in ("clear", "none", "reset"):
            session.set_goal("")
            notice("goal cleared")
            return
        stored = session.set_goal(arg)
        notice(f"goal set (applies from the next turn): {stored}")

    def _cmd_loop(self, arg: str, notice: NoticeFn) -> None:
        """``/loop [n]`` — iterate toward the goal; ``/loop stop`` cancels.

        Each iteration is a real turn that asks the agent to advance the
        standing goal, so the loop is bounded, interruptible, and visible in
        the transcript rather than a hidden background process.
        """
        session = self._session
        if arg.lower() in ("stop", "cancel", "abort"):
            if self._loop_running:
                self._loop_cancelled = True
                notice("loop will stop after the current turn")
            else:
                notice("no loop is running")
            return
        if session is None:
            notice("session is still starting…", "warning")
            return
        if self._loop_running:
            notice("a loop is already running — /loop stop to cancel", "warning")
            return
        if not getattr(session, "goal", ""):
            notice("set a goal first: /goal <text>", "warning")
            return
        iterations = DEFAULT_LOOP_ITERATIONS
        if arg:
            try:
                iterations = int(arg)
            except ValueError:
                notice(f"usage: /loop [n] (1..{MAX_LOOP_ITERATIONS})", "warning")
                return
        if iterations < 1 or iterations > MAX_LOOP_ITERATIONS:
            notice(f"iterations must be between 1 and {MAX_LOOP_ITERATIONS}", "warning")
            return
        self._loop_cancelled = False
        notice(f"looping toward the goal ({iterations} iteration(s)) — /loop stop to cancel")
        self.run_worker(self._loop_worker(iterations), thread=False, group="loop")

    async def _loop_worker(self, iterations: int) -> None:
        """Run up to ``iterations`` goal-advancing turns, sequentially."""

        def notice(body: str, kind: str = "info") -> None:
            self._append_block(NoticeBlock(body, kind))

        session = self._session
        if session is None:
            return
        self._loop_running = True
        completed = 0
        try:
            for index in range(iterations):
                if self._loop_cancelled:
                    break
                self._append_block(UserBlock(f"[loop {index + 1}/{iterations}]"))
                if self._status is not None:
                    self._status.update(streaming=True)
                try:
                    await session.prompt(LOOP_PROMPT)
                except Exception as error:  # surface and stop; never spin
                    notice(f"loop stopped: {error}", "error")
                    break
                finally:
                    if self._status is not None:
                        self._status.update(streaming=False)
                completed = index + 1
        finally:
            self._loop_running = False
        if self._loop_cancelled:
            notice(f"loop cancelled after {completed} iteration(s)")
        else:
            notice(f"loop finished after {completed} iteration(s)")

    # -- providers / accounts / usage --------------------------------------
    def _cmd_providers(self, notice: NoticeFn) -> None:
        """``/provider`` — list loginable providers and their state."""
        if self._providers is None:
            notice("run: local-operator provider (TUI lacks the provider facade)", "warning")
            return
        try:
            items: list[tuple[str, str]] = []
            for definition in self._providers.login_providers():
                # ONE state vocabulary, resolved in ONE place. This line and the
                # `/login` picker answer the same question on two surfaces, and
                # they had already drifted: the picker said "needs login" where
                # this said "—", so a user with no credential read a dash and had
                # to guess whether it meant unknown, unsupported or absent.
                state = self._credential_state(
                    definition.id, self._providers.has_any_credential(definition.id)
                )
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
        """Provider ids `/provider` may advertise as reporting quota.

        Delegated rather than recomputed. This line and bare `/usage`'s target
        list have to be the same set or `/provider` promises a table that `/usage`
        then renders empty — which is what a locally-written `is_usable` filter
        did for the five OAuth-only providers when the user held only an API key.
        """
        if self._providers is None:
            return []
        return self._providers.usage_reportable_providers()

    def _cmd_accounts(self, notice: NoticeFn) -> None:
        """``/accounts`` — list stored credentials (OAuth + pasted keys)."""
        if self._providers is None:
            notice("run: local-operator login status (TUI lacks the provider facade)", "warning")
            return
        try:
            rows = self._providers.credentials()
            if not rows:
                # No terminal period: the app's short notices do not carry one
                # (`/provider`'s warning does not), and two spellings of the same
                # register read as two different kinds of message.
                notice("no stored credentials")
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

    def _cmd_usage(self, arg: str, notice: NoticeFn) -> None:
        """``/usage [provider]`` — fetch live quota for a provider (or all)."""
        if self._providers is None:
            notice("provider controller unavailable — usage cannot be fetched", "warning")
            return
        target = arg.lower() if arg else ""
        if target:
            from local_operator.providers.usage import usage_kinds, usage_supported

            if self._providers.provider(target) is None and not usage_supported(target):
                notice(f"unknown provider: {target}", "warning")
                return
            wants_oauth, wants_key = usage_kinds(target)
            # "No endpoint" and "an endpoint you cannot reach" look identical in an
            # empty table, and only the second is something a user can act on. Said
            # up front rather than after a request that was always going to answer
            # nothing.
            if not wants_oauth and not wants_key:
                notice(f"{target} publishes no usage or quota endpoint", "warning")
                return
            if not self._providers.is_usable(target):
                need = "an API key" if wants_key else "an OAuth login"
                notice(f"{target} needs {need} before it can report usage", "warning")
                return
            if wants_oauth and not wants_key and not self._providers.has_any_credential(target):
                notice(f"{target} reports usage only after /login {target}", "warning")
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
        # `label`, NOT accent. Green in this app means "a turn is live" — the
        # band's always-on brand glyph was moved off it for exactly that reason,
        # and a static section heading painted the same green teaches the user
        # that green also means "heading", which weakens the liveness signal
        # everywhere else.
        heading = Style(color=theme_mod.semantic_color("label"))
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
            head.append(report.provider, style=heading)
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
                amount_text = _format_usage_amount(a)
                if amount_text:
                    label += f" — {amount_text}"
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

    # -- provider argument list ---------------------------------------------
    def on_provider_query_opened(self, message: ProviderQueryOpened) -> None:
        """The buffer just entered ``/login …`` or ``/logout …`` — fill the list.

        Answered on the MESSAGE for the same reason the model list is: every route
        into the list (typing the space, being completed into it by the command
        picker) then arrives at one place with one set of rows.
        """
        message.stop()
        editor = self._editor()
        picker = editor.picker
        if editor.provider_command != message.command:
            # The message is one message-loop tick old, and a tick is enough for
            # the user to have deleted the command or typed over it. Verified:
            # setting the buffer to `/logout ` and then to a sentence still
            # appended the notice below, attaching it to a command that no longer
            # exists. The buffer is the authority on which list is open, here as
            # much as in the editor's own resync.
            return
        if self._providers is None:
            # Same degradation as the handlers themselves: no controller means no
            # credential store to read, so the list is empty and the user is
            # pointed at the CLI rather than left watching nothing happen.
            picker.set_choices([])
            picker.set_notice(
                f"provider controller unavailable — run: local-operator {message.command}"
            )
            return
        choices, problem = self._provider_choices(message.command)
        picker.set_choices(choices)
        # An empty list and "no match for what you typed" render identically — as
        # nothing at all. Only one of them is worth saying, and this is it: there is
        # no credential to remove, so no query would have helped. `/login` always
        # has rows, so an empty one there IS the query, which the user can read
        # back off their own buffer.
        reason = ""
        if not choices and message.command == "logout":
            reason = problem or "no stored credentials — nothing to log out of."
        # Said WHERE THE LIST WOULD HAVE BEEN, not in the transcript. This answers a
        # UI event, so it fires again every time the buffer re-enters the argument
        # state — `/logout `, backspace, space, and four identical rows have stacked
        # up in what is supposed to be a record of the conversation, each one also
        # taking a row off the splash that shares that region (see D-01). In the
        # picker it is in the user's eye-line, self-clearing, unrepeatable, and it
        # costs the transcript nothing.
        picker.set_notice(reason)

    def _provider_choices(self, command: str) -> _ProviderRows:
        """Provider rows for ``/login`` or ``/logout``, in registry order.

        The registry is in memory and cannot fail; the credential store is SQLite
        and can — one other local-operator process holding a write lock is enough
        for `database is locked`, and this method runs on a KEYSTROKE, so an
        exception out of it takes the whole TUI down. The moment the store is
        unreadable is exactly the moment a user reaches for `/login`, so the state
        column degrades and the catalogue survives.
        """
        providers = self._providers
        assert providers is not None
        logout = command == "logout"
        if not logout:
            return _ProviderRows(self._login_choices(), "")
        # ONE store read for the KINDS, up front: `/logout` needs the kind of every
        # credential it offers to remove, and reading them per row would re-scan
        # the store once per provider.
        stored_kinds = self._stored_credential_kinds()
        if stored_kinds is None:
            # `/logout` asks a question only the store can answer — which
            # credentials exist. There is no degraded list to offer, so say what
            # is wrong instead of rendering an empty one that reads as "you have
            # no credentials".
            return _ProviderRows([], "credential store unreadable — /logout cannot list anything")
        choices: list[ArgumentChoice] = []
        seen_storage: set[str] = set()
        for definition in providers.login_providers():
            storage = definition.store_credentials_as or definition.id
            # BOTH reads have to agree before a destructive row is offered: the
            # facade's predicate (whose rule about storage aliasing and disabled
            # rows is not the UI's to re-derive) and the credential map, which is
            # the record of what would actually be deleted. Either one alone can
            # produce a row that promises to remove something that is not there.
            #
            # Guarded per row: one provider whose read blows up must not delete
            # the rest of a list that is otherwise answerable.
            kinds = stored_kinds.get(storage)
            if kinds is None or not self._has_credential(definition.id):
                # Only what can actually be removed. Offering a provider the user
                # never logged into is a row whose only outcome is a no-op notice.
                continue
            # `xai` and `xai-oauth` (and openai/openai-device) share one credential
            # row, so both would log the same account out. Two rows for one outcome
            # is a choice the user cannot make correctly.
            if storage in seen_storage:
                continue
            seen_storage.add(storage)
            choices.append(
                ArgumentChoice(
                    name=definition.id,
                    description=_provider_summary(definition.id, definition.name),
                    # `claude` finds anthropic, `qwen` finds alibaba. The alias only
                    # makes the row FINDABLE — the completion is still the id.
                    aliases=tuple(definition.search_aliases),
                    detail=_removal_detail(kinds),
                    # Every row on THIS list destroys a credential, so the danger
                    # tint is a property of the command rather than of a row that
                    # went wrong — the same red the tool card spends on a failed
                    # outcome, saying the same thing. Never set for `/login`, where
                    # the identical treatment would paint an ordinary catalogue as
                    # a wall of failures.
                    alert=True,
                )
            )
        return _ProviderRows(choices, "")

    def _login_choices(self) -> list[ArgumentChoice]:
        """Every loginable provider, with where the user stands on each."""
        providers = self._providers
        assert providers is not None
        return [
            ArgumentChoice(
                name=definition.id,
                description=_provider_summary(definition.id, definition.name),
                aliases=tuple(definition.search_aliases),
                # Blank when the store could not be read: the catalogue is still
                # entirely answerable from the registry, and a row with no state
                # claims nothing, where any of the three states would claim
                # something the app does not know. With every detail blank the
                # column collapses to nothing and the descriptions take the cells.
                detail=self._stored_state(definition.id) or "",
            )
            for definition in providers.login_providers()
        ]

    def _stored_state(self, provider_id: str) -> str | None:
        """:meth:`_credential_state` for one provider, or ``None`` when it failed.

        Guarded per ROW, not per list: one provider whose read blows up must not
        delete the other eleven from a catalogue that is otherwise answerable.
        """
        providers = self._providers
        assert providers is not None
        try:
            return self._credential_state(provider_id, providers.has_any_credential(provider_id))
        except Exception:  # a credential read never costs the user the list
            return None

    def _has_credential(self, provider_id: str) -> bool:
        """``has_any_credential``, guarded — a failed read offers nothing.

        False rather than True on failure: `/logout` acts on what this returns,
        and offering a row the store could not confirm invites a keystroke whose
        only outcome is an error.
        """
        providers = self._providers
        assert providers is not None
        try:
            return providers.has_any_credential(provider_id)
        except Exception:  # a credential read never costs the user the list
            return False

    def _stored_credential_kinds(self) -> dict[str, tuple[str, ...]] | None:
        """Storage id -> the ``credential_type`` of each credential filed under it.

        ``None`` — distinct from an empty map — when the store could not be read
        at all, because "you have no credentials" and "I cannot tell" are
        different answers and only one of them is true when SQLite is locked.

        A tuple per id, not one value: nothing stops a provider holding both a
        pasted key and an OAuth login, and `/logout` removes the lot — a row that
        named only the first would understate what the keystroke does.
        """
        providers = self._providers
        assert providers is not None
        try:
            rows = providers.credentials()
        except Exception:  # never crash the app on a provider read
            return None
        kinds: dict[str, tuple[str, ...]] = {}
        for row in rows:
            kinds[row.provider] = (*kinds.get(row.provider, ()), row.credential_type)
        return kinds

    def _credential_state(self, provider_id: str, stored: bool) -> str:
        """Where the user stands with ``provider_id``, in three states not two.

        An environment key is a WORKING credential — it is the tier the stream
        cascade resolves — but it is not a login, so reporting it as one would
        suggest a stored account that `/logout` could remove. `/provider` renders
        the same three strings from this same method, so the two surfaces cannot
        drift into answering one question two ways.

        `/logout` does NOT use this: see :func:`_removal_detail`.
        """
        providers = self._providers
        assert providers is not None
        if stored:
            return "logged in"
        return "env key" if providers.is_usable(provider_id) else "needs login"

    # -- login / logout -----------------------------------------------------
    def _cmd_login(self, arg: str, notice: NoticeFn) -> None:
        """``/login [provider]`` — list loginable providers, or run a flow."""
        if self._providers is None:
            notice("provider controller unavailable — run: local-operator login", "warning")
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

    def _login_callbacks(self, definition: object) -> LoginCallbacks:
        """Login hooks that render into the transcript instead of the terminal.

        The CLI's hooks print with ``print()`` and read with ``input()``, which
        a Textual app cannot host: the previous implementation wrapped the whole
        flow in ``App.suspend()`` and so tore the UI down mid-login, then blocked
        on a paste prompt the user had no reason to expect.

        ``on_manual_code_input`` is deliberately ABSENT. The loopback callback
        server is the real path — it is already listening before the URL is
        shown, and the browser redirect completes the flow with no typing. A
        paste prompt is a fallback for a browser on a different machine, which
        is a CLI situation; offering it here would mean reading stdin while the
        app owns it.
        """
        # ``callback_server`` is imported HERE: it drags in http.server, ssl and
        # email (~138 ms, 150-odd modules) for a loopback listener that only a
        # login needs, and this module is what every interactive session imports.
        from local_operator.providers.oauth.callback_server import LoginCallbacks

        def on_auth_url(url: str, instructions: str | None = None) -> None:
            lines = [
                Text(
                    "opening your browser to authorize…",
                    style=Style(color=theme_mod.semantic_color("muted")),
                ),
                Text(url, style=Style(color=theme_mod.semantic_color("signal"))),
            ]
            if instructions:
                lines.append(Text(instructions, style=Style(color=theme_mod.semantic_color("dim"))))
            self._append_block(RichBlock(Group(*lines)))

        def on_progress(message: str) -> None:
            self._append_block(NoticeBlock(message, "info"))

        return LoginCallbacks(on_auth_url=on_auth_url, on_progress=on_progress)

    async def _login_flow(self, provider: str) -> None:
        """Run the login on the event loop, reporting into the transcript.

        No ``App.suspend()``: the flow needs the terminal only if it reads from
        it, and with the loopback server doing the capture it does not. A lock
        serializes concurrent /login commands.
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
            self._providers.set_login_callbacks(self._login_callbacks)
            message = await self._providers.login(provider)
            await notice(message, "success")
            # Nothing else needs poking: the splash re-polls its credential
            # warning whenever it becomes visible again (`set_visible(True)`
            # calls `_poll`), and it is hidden right now because the notice
            # above is a transcript block.
        except Exception as error:
            await notice(f"login failed: {error}", "error")
        finally:
            self._login_lock.release()

    def _cmd_logout(self, arg: str, notice: NoticeFn) -> None:
        """``/logout [provider]`` — remove stored credentials for a provider."""
        if self._providers is None:
            notice(
                "provider controller unavailable — run: local-operator logout <provider>", "warning"
            )
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
        # Where the logs went. Console logging is off while the TUI owns the
        # terminal (see `local_operator.logger.file_logging`), so without this
        # line the file is unfindable without reading the source. `/help` and
        # not the transcript: a path printed on every launch is noise, a path
        # in the help the user already opens when lost is not.
        log_file = current_log_file()
        if log_file is not None:
            footer = Text()
            footer.append("logs".ljust(14), style=muted)
            footer.append(str(log_file), style=dim)
            lines.append(Text())
            lines.append(footer)
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
        """Per-server MCP state: connection status plus the error when it failed.

        Extended beyond a config dump deliberately. The startup toast dismisses
        itself, so this is one of the two places a failure has to remain
        readable afterwards (the other is the transcript notice). Listing the
        configured command alone answered "what did I ask for" and never "did it
        work", which is the only question a user runs ``/mcp`` to settle.

        Both fields are padded into COLUMNS. Crammed straight into the detail
        string, the status landed wherever the server name happened to end —
        `connected` at column 15 under a long name, `disconnected` at column 11
        under a short one — so the shorter name pushed the longer status left and
        the two facts a reader scans for (which server, is it up) formed no
        column at all. Padding is the whole fix: no glyph and no colour added.

        Still exception-safe end to end: introspection that crashes the app is
        worse than introspection that declines to answer.
        """
        try:
            from local_operator.mcp.config import load_all_mcp_configs

            configs, _sources = load_all_mcp_configs(os.getcwd())
            if not configs:
                return None
            manager = getattr(self._session, "mcp_manager", None)
            startup = getattr(self._session, "mcp_startup", None)
            failures = dict(startup.failures) if startup is not None else {}
            rows: list[tuple[str, str, str]] = []
            for name, cfg in configs.items():
                target = getattr(cfg, "command", None) or getattr(cfg, "url", None) or ""
                status = manager.get_connection_status(name) if manager is not None else "unknown"
                # The boot error is quoted only while the server is STILL down: a
                # server that has since reconnected must not keep reporting the
                # failure it recovered from, or /mcp becomes a permanent accusation.
                error = failures.get(name, "") if status != "connected" else ""
                rows.append((name, status, error or target))
            name_column = max(len(name) for name, _, _ in rows)
            status_column = max(len(status) for _, status, _ in rows)
            items = [
                (name.ljust(name_column), f"{status.ljust(status_column)}  {detail}".rstrip())
                for name, status, detail in rows
            ]
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
        # Build the segments as typed locals and make ONE call: a
        # dict[str, object] splatted into update() erases every parameter type,
        # so a wrong-typed segment would only surface as a render glitch.
        # `None` means "leave this segment alone" in update()'s contract.
        context_tokens: int | None = message.context_tokens or None
        cost_text: str | None = None
        cost = self._cost_for(message.usage)
        if cost is not None:
            self._total_cost += cost
            cost_text = format_cost(self._total_cost)
        elif message.usage is not None and getattr(message.usage, "input_tokens", 0):
            # D20: the turn billed tokens but pricing is unknown — render an
            # explicit "unavailable" so the segment's absence reads as that,
            # not as "free".
            cost_text = "$—"
        self._status.update(
            streaming=False,
            context_tokens=context_tokens,
            context_window=_context_window(self._session),
            cost=cost_text,
        )
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
        """Best-effort turn cost from the resolved pricing (never raises)."""
        if usage is None or self._session is None:
            return None
        try:
            from local_operator.model.configure import calculate_cost, resolve_model_info

            provider, _, model_id = self._session.model_label.partition("/")
            # resolve_model_info, NOT get_model_info: aggregators carry a
            # placeholder registry entry with zero prices, so the static lookup
            # reported "pricing unknown" for every OpenRouter model even though
            # the session had already resolved the real numbers. Memoized, so
            # this is a dict hit after the first turn.
            info = resolve_model_info(provider, model_id)
            if not (info.input_price or info.output_price):
                # Genuinely no pricing: a confident $0.0000 would read as
                # "this turn was free" — treat as unknown instead.
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
        # Hand the card the FULL result text and details, not just a summary
        # line: the text backs click-to-expand and the details carry the
        # write/edit +N/-N counters. Without this the card can only ever show
        # the headline, so both features stay dark.
        result_text = event.result.text
        details = event.result.details
        if event.is_error:
            card.mark_failed(_first_line(result_text), result_text, details)
        else:
            card.mark_done(result_text, details)

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


def _model_spec(session) -> Any | None:
    """The session's active ``ModelSpec``, or ``None`` when it has none.

    Defensive because the TUI also runs against reduced hosts — embedders
    and the pilot fakes supply a session without the spec accessor — and a
    status segment must degrade to "unknown" rather than take the app down.
    """
    return getattr(session, "model", None)


def _context_window(session) -> int:
    """The active model's context window, or 0 when it is unknown.

    Zero is meaningful downstream: the usage segment renders ``12.4k/—``
    rather than inventing a denominator to divide by.
    """
    window = getattr(_model_spec(session), "context_window", 0) or 0
    return int(window) if window > 0 else 0


def _effort_label(session) -> str:
    """The model's reasoning-effort label, or "" when it has none.

    ``ModelSpec.reasoning_effort`` is read first so a provider-level effort
    knob shows its actual level ("high") the moment the spec grows one.
    Until then the only reasoning signal on the spec is the boolean, and a
    model that reasons at the provider's default effort is reported as
    ``reasoning`` — non-reasoning models render nothing, which is what makes
    the segment's presence informative.
    """
    spec = _model_spec(session)
    if spec is None:
        return ""
    explicit = getattr(spec, "reasoning_effort", None)
    if explicit:
        return str(explicit).strip().lower()
    return "reasoning" if getattr(spec, "reasoning", False) else ""


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


def _format_usage_amount(amount) -> str:
    """The number for one usage row, or ``""`` when the amount carries none.

    ``used`` first, then ``remaining``, then ``limit``, then an explicit fraction.
    A remaining-only balance — exactly what both account-balance fetchers report,
    since neither vendor gives a limit to derive spend from — used to print no
    number at all, leaving a row labelled "Balance" that never said how much and,
    for DeepSeek, no digit anywhere on screen.

    Units come from ``UNIT_LABELS`` rather than the raw key, so a row reads
    ``519.86 USD`` and ``30%`` instead of ``519.86 usd`` and ``30 percent``. A unit
    with no label (``unknown`` — a CNY balance already naming its currency in the
    label) prints the bare number rather than dropping it.
    """
    # Imported here, not at module scope: this file deliberately keeps the
    # provider layer off the TUI's import graph, and by the time a report exists
    # the module is already loaded.
    from local_operator.providers.usage import UNIT_LABELS

    unit = getattr(amount, "unit", "unknown")
    label = UNIT_LABELS.get(unit, unit)

    def with_unit(value: float) -> str:
        text = f"{value:.2f}" if unit == "usd" else f"{value:g}"
        if not label:
            return text
        # "%" is a suffix, every other label is a separate word.
        return f"{text}{label}" if label == "%" else f"{text} {label}"

    if amount.used is not None:
        return with_unit(amount.used)
    if amount.remaining is not None:
        return f"{with_unit(amount.remaining)} left"
    if amount.limit is not None:
        return f"{with_unit(amount.limit)} limit"
    if amount.used_fraction is not None:
        # The bar already shows this, but a bar cannot be read off precisely.
        return f"{amount.used_fraction * 100:g}% used"
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


#: How a stored credential's ``credential_type`` reads to a user about to lose
#: it. The same two words `/accounts` prints, so one credential does not have
#: two names across two surfaces.
#:
#: Short on purpose. The whole string is `remove <kind>`, and the detail column
#: is dropped whole once reserving it would squeeze the provider id — measured
#: on the 40-cell boot card, `remove oauth login` (18 cells) took the column
#: away from every row while `remove oauth` (12) keeps it. Losing the column at
#: the width where the description has ALREADY collapsed would leave the row
#: saying nothing but its id, which is the state D-06 exists to fix.
_CREDENTIAL_KINDS = {"oauth": "oauth", "api_key": "api key"}


def _removal_detail(kinds: tuple[str, ...]) -> str:
    """What `/logout <id>` will REMOVE, for the picker's detail column.

    Not "logged in". `/logout` offers only providers that HAVE a credential, so
    that state is true of every row by construction — a column carrying no bits
    at all, holding cells the description needs at narrow widths. The KIND is
    what differs between rows and what the user is about to lose, so the row
    states its own consequence instead of restating its own precondition.

    ``kinds`` is non-empty by construction: the caller only builds a row once the
    credential map has an entry for its storage id, so there is no such thing
    here as a removal with nothing to remove.
    """
    labels = {_CREDENTIAL_KINDS.get(kind, kind) for kind in kinds}
    if len(labels) == 1:
        return f"remove {labels.pop()}"
    # Both a pasted key and an OAuth login under one id: `/logout` takes the lot,
    # and a row naming only the first would understate the keystroke.
    return f"remove {len(kinds)} credentials"


def _provider_summary(provider_id: str, name: str) -> str:
    """The part of a provider's registry name the id does not already say.

    The registry name is written for a CLI listing where the id is not adjacent:
    `OpenAI (ChatGPT Plus/Pro)`, `DeepSeek`, `OpenRouter`. Printed next to the id
    in a picker it restated it on twelve rows out of twelve, with the only
    disambiguating part parenthesised at the end of each — while `openai` vs
    `openai-device` and `xai` vs `xai-oauth` are told apart by NOTHING ELSE.

    So: the parenthetical when there is one, empty when the name is just the id
    in title case, the name itself otherwise. An empty cell is the correct answer
    for `deepseek` — there is nothing more to say about it than its id already
    says, and saying it twice is what cost the four near-duplicate rows their
    distinguishing text below 41 cells.
    """
    name = name.strip()
    if name.endswith(")") and "(" in name:
        return name[name.index("(") + 1 : -1].strip()
    if _squashed(name) == _squashed(provider_id):
        return ""
    return name


def _squashed(value: str) -> str:
    """``value`` reduced to its letters and digits, lowercased.

    So `OpenRouter` and `openrouter` compare equal, and so do `xAI OAuth` and
    `xai-oauth` — punctuation and case are exactly the difference between a name
    and the id it restates.
    """
    return "".join(char for char in value.lower() if char.isalnum())


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
    provider_controller: Any | None = None,
) -> OperatorApp:
    """Construct an :class:`OperatorApp` (test/embedding helper)."""
    return OperatorApp(
        session_factory,
        theme_name,
        provider_controller=provider_controller,
    )


def _catalogue_status(statuses: dict[str, str]) -> str:
    """One line summarising what the catalogue does NOT know.

    Silence when every provider answered, because a footer that always says
    something is a footer nobody reads. The interesting cases are the ones where a
    user hunting for a model released last week would otherwise conclude it does
    not exist: a cached list, or a provider whose live fetch failed.

    An ``unauthenticated`` provider is deliberately NOT reported here. Its models
    are the ones the picker now hides, and `_catalogue_rows` already counts them
    — two footers counting one fact ("3 provider(s) need a login · 42 hidden")
    reads as two separate problems.
    """
    cached = sorted(p for p, s in statuses.items() if s == "cached")
    stale = sorted(p for p, s in statuses.items() if s in ("unavailable", "empty"))
    bits: list[str] = []
    if cached:
        bits.append(f"cached: {', '.join(cached)}")
    if stale:
        bits.append(f"no live list: {', '.join(stale)}")
    return " · ".join(bits)


def _status_line(*bits: str) -> str:
    """Join the picker's footer clauses, dropping the empty ones.

    The separator matches the one the footer itself uses between the row count
    and the status, so a two-clause status does not read as a different kind of
    seam from the one beside it.
    """
    return " · ".join(bit for bit in bits if bit)
