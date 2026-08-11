"""The full-screen Local Operator TUI (Textual).

Layout (top→bottom): the scrolling transcript, the input panel on the
``surface`` elevation step with the ``❯`` chevron, and the full-width status
BAND on the ``sunken`` ground. No bordered boxes anywhere; the app draws two
lines, both ``$lo-dim`` and neither an edge of anything — the input's thin top
rule (with focus brightening the chevron, D23/D24) and the user
prompt's gutter bar in the transcript. Structure comes from symbols, tint, and
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

import asyncio
import logging
import os
import time
from typing import TYPE_CHECKING, Any, Awaitable, Callable, NamedTuple, Protocol

from rich.console import Group
from rich.style import Style
from rich.terminal_theme import TerminalTheme
from rich.text import Text
from textual.app import App, ComposeResult
from textual.binding import Binding
from textual.containers import Container, Horizontal
from textual.css.query import NoMatches
from textual.events import DescendantBlur, DescendantFocus
from textual.geometry import Size
from textual.widgets import Static

from local_operator.ansi import strip_control_sequences
from local_operator.harness.intent import (
    ACTIVITY_RESPONDING,
    batch_activity,
    tool_activity,
)

# Free at runtime: `session.protocol` below already imports `harness.types` at
# module level, so this adds no work to the boot path the lazy-import
# discipline protects. The aside builds the request-scoped turns it hands to
# `complete_aside` out of these two.
from local_operator.harness.types import AgentMessage, Message
from local_operator.logger import current_log_file

# A leaf table (`re` and `dataclasses` only), so importing it here costs the
# boot path nothing the lazy-import discipline above is protecting.
from local_operator.model.effort import default_effort, next_effort
from local_operator.session import naming
from local_operator.session.protocol import SessionProtocol
from local_operator.tui import theme as theme_mod
from local_operator.tui.autocomplete import ArgumentChoice, ArgumentMode, SlashCommand
from local_operator.tui.costs import job_cost, turn_cost
from local_operator.tui.events import (
    AssistantDelta,
    AssistantMessageEnd,
    AssistantMessageStart,
    CompactionEnded,
    CompactionStarted,
    ContextUsageReported,
    EventController,
    NoticePosted,
    RetryEnded,
    RetryStarted,
    StartFlushTimer,
    SubagentEnded,
    SubagentProgress,
    SubagentStarted,
    ToolComposing,
    ToolEnded,
    ToolStarted,
    ToolUpdated,
    TurnBoundaryEnd,
    TurnBoundaryStart,
    TurnEnded,
    TurnStarted,
)
from local_operator.tui.glyphs import display_name
from local_operator.tui.markdown_theme import (
    brand_markdown_theme,
    install_markdown_theme,
)
from local_operator.tui.widgets.approval import ApprovalBlock
from local_operator.tui.widgets.aside_panel import ASIDE_PROMPT, AsidePanel
from local_operator.tui.widgets.assistant import AssistantBlock
from local_operator.tui.widgets.editor import (
    ASIDE_PLACEHOLDER,
    DEFAULT_PLACEHOLDER,
    READ_ONLY_PLACEHOLDER,
    ArgumentQueryOpened,
    Editor,
    EditorQuit,
    EditorSubmitted,
    InterruptRequested,
    ModelQueryOpened,
    StopRequested,
)
from local_operator.tui.widgets.model_picker import ModelRow
from local_operator.tui.widgets.session_picker import SessionPickerScreen
from local_operator.tui.widgets.status_line import (
    McpStatus,
    StatusLine,
    SubagentBand,
    format_context_tokens,
    format_cost,
)
from local_operator.tui.widgets.subagent_panel import (
    JobStats,
    SubagentPanel,
    job_elapsed,
)
from local_operator.tui.widgets.subagent_view import SubagentView, SubagentViewDismissed
from local_operator.tui.widgets.toast import Toast, format_mcp_startup
from local_operator.tui.widgets.todo_panel import TodoPanel
from local_operator.tui.widgets.tool_card import ToolCard, clean_intent
from local_operator.tui.widgets.transcript import (
    DEFAULT_ACTIVITY,
    GAP_CLASS,
    NoticeBlock,
    NoticeKind,
    RichBlock,
    TranscriptView,
    UserBlock,
    WorkingBlock,
)
from local_operator.tui.widgets.usage_panel import (
    UsageDismissed,
    UsagePanel,
    UsageRefreshRequested,
)
from local_operator.tui.widgets.welcome import (
    MODEL_PENDING,
    WelcomeView,
    session_welcome_info,
)

if TYPE_CHECKING:  # keeps the provider graph off the TUI's runtime import path
    from local_operator.providers.controller import CatalogueEntry
    from local_operator.providers.oauth.callback_server import LoginCallbacks


#: ONE sentence for ONE instruction, carried verbatim by every surface that
#: mentions ``/model default``: the bare-``/model`` notice, the switch receipt,
#: the model picker's footer and the ``/help`` row. They used to say it four
#: different ways within two keystrokes of each other — "saves this provider and
#: model as the boot default", "to make it the boot default", "saves the boot
#: default", "persists it" — so a user met a new phrasing on every surface
#: instead of learning one string.
#:
#: Sized by its TIGHTEST site. The picker footer truncates at the card's width
#: and this clause sits after the access note, so it has to be complete and short.
#: "Saves provider and model" named the payload but not why it mattered; "saves
#: this for new sessions" names the consequence, fits the same slot, and includes
#: the article the clipped phrase lacked.
PERSIST_HINT = "/model default saves this for new sessions"

#: Slash commands handled synchronously before any prompt is sent. One
#: registry entry per command; aliases live on the entry (TUI-014).
#:
#: ``echo`` says whether running the command leaves a user row in the visible
#: ledger. It USED to be unconditional, on the reasoning that typing a command
#: is the same visible commitment as sending a prompt. That reasoning had the
#: wrong subject: a prompt is echoed because the transcript is the only record
#: of what the user said, whereas every handler below already reports what it
#: did — ``/usage`` opens the panel that IS the answer, ``/provider`` prints the
#: list, ``/model p/id`` names both labels — so the echo was a row restating a
#: row underneath it. The reading record kept the keystrokes and gained nothing.
#:
#: So the test is not "did the user commit to something" but "would the receipt
#: be missing something without it", and exactly one thing qualifies: an
#: argument that becomes part of what the MODEL is told. Comment per entry
#: below; the table is pinned in ``tests/unit/tui/test_slash_echo.py``.
SLASH_COMMANDS: list[SlashCommand] = [
    # The help table is the receipt.
    SlashCommand("help", "List all commands"),
    # The app is gone; there is no ledger left to read.
    SlashCommand("exit", "Quit the app", aliases=("quit",)),
    # Empties the surface the echo would land on — it was wiped a line later.
    SlashCommand("clear", "Clear the transcript (history is untouched)"),
    # Replaces the transcript; a row describing the old one would not survive.
    SlashCommand("new", "Start a new conversation"),
    # "reloading session…" while it runs; `_reconcile_reload` posts the receipt
    # that survives the ledger being rebuilt, and it names what came back.
    SlashCommand("reload", "Reboot the session, keeping this conversation"),
    # The picker (or "resuming session <id>…") is the receipt, and a resume
    # replaces the transcript anyway.
    SlashCommand(
        "resume",
        "Pick a past conversation to resume, or resume one (id)",
        aliases=("recall",),
    ),
    # The switch receipt names the old AND new label — strictly more than the
    # typed selector, which may have been elided to `default`.
    SlashCommand(
        "model",
        # Terse by necessity — the description column wraps past ~55 cells — but
        # it still carries PERSIST_HINT verbatim rather than a fifth paraphrase.
        # The `<provider>/<id>` shape it used to show moved to the tip pool, which
        # has the room for it (`welcome.TIPS`).
        f"Switch model; {PERSIST_HINT}",
        aliases=("models",),
    ),
    # Next to `/model` because it is the same question one level down: which
    # model, and then how hard it thinks.
    #
    # NOT an echo. The argument is a setting, not words the model is given, and
    # the receipt names the resulting level — the durable fact — where the typed
    # word is only how it was reached. Exactly `/approvals`' rule.
    SlashCommand(
        "effort",
        "Show or set reasoning effort (shift+tab cycles)",
        # OPTIONAL: the space offers this model's rungs, and a bare `/effort`
        # still prints the ladder with the current one marked. The list is what
        # the printed ladder could never be — the rungs are OFFERED rather than
        # transcribed by hand from a line of prose.
        arguments=ArgumentMode.OPTIONAL,
    ),
    # The listing is the receipt.
    SlashCommand("provider", "List providers and their login/usage state"),
    # The listing is the receipt.
    SlashCommand("accounts", "List stored credentials"),
    # The panel is the receipt — the row the owner reported as noise.
    SlashCommand("usage", "Show provider usage quota"),
    # THE exception. `/goal <text>` is the one command whose argument reaches
    # the model: the goal rides the system prompt's volatile tail on every later
    # turn (`Session.set_goal`). Words the model is given are the transcript's
    # subject matter, and they belong to the user, so they get a user row rather
    # than being paraphrased into a system notice. `_cmd_goal` writes that row
    # itself, only on the branch that actually stored something — the flag is
    # the permission, not the trigger.
    SlashCommand("goal", "Show, set, or clear the session goal", echo=True),
    # Not an exception: LOOP_PROMPT is app-authored, not the user's words, and
    # `_loop_worker` already labels every iteration it starts (`· loop 1/3`), so
    # no agent output here is left unattributed.
    SlashCommand("loop", "Iterate autonomously toward the goal"),
    # NOT an exception, and the reason IS the feature. The question does reach
    # the model, but only for one off-the-record request that never joins the
    # conversation (`SessionProtocol.complete_aside`) — so a user row in the
    # ledger would be the one trace the aside promises not to leave, and would
    # still be sitting there after Esc claimed to have thrown the exchange
    # away. The card is the receipt; `^f` inside it is how an exchange gets a
    # row, as a real turn rather than an echo.
    SlashCommand("btw", "Ask a side question off the record (esc closes it)"),
    # NOT an echo, and the receipt is the reason. The pass narrates itself
    # through the same `compacting context…` / `context compacted · 128.4k →
    # 21.9k tokens` notices the automatic one emits, and a refusal says why it
    # did not run — nothing typed here reaches the model, so a user row above
    # that would only restate the word.
    SlashCommand("compact", "Compact the context now"),
    # The receipt states the resulting mode, which is the durable fact; the
    # typed argument is only how it was reached.
    SlashCommand(
        "approvals",
        # Names the SCOPE word, not the modes: the modes are rows in the list a
        # space opens, where they can carry which one is live and which one the
        # next launch will use. `default` is the half a list cannot teach on its
        # own, because a user has to suspect it exists to go looking for it —
        # the same job `PERSIST_HINT` does on `/model`.
        "Show or set tool approval mode; add default to keep it",
        arguments=ArgumentMode.OPTIONAL,
    ),
    # The listing is the receipt.
    SlashCommand("skills", "List loaded skills"),
    # The listing is the receipt.
    SlashCommand("mcp", "List MCP servers"),
    # The flow narrates itself: URL block, progress notices, then success.
    # REQUIRED for both: bare, neither has anything to run — the provider list
    # IS the command, which is why completing the word opens it instead of
    # submitting a no-op over the list it just drew.
    SlashCommand("login", "Authenticate a provider", arguments=ArgumentMode.REQUIRED),
    # The worker reports the removal, naming the provider.
    SlashCommand("logout", "Remove stored provider credentials", arguments=ArgumentMode.REQUIRED),
]


def slash_command_for(text: str) -> SlashCommand | None:
    """The registry entry a typed line invokes, or ``None`` if nothing matches.

    Resolves through :attr:`SlashCommand.names`, so an alias answers with the
    same entry as its primary name — ``/quit`` must not get a different echo
    policy from ``/exit`` just because it was spelled the other way.

    Matching is case-insensitive because registry names are lowercase and this
    is the ONE resolver both the echo permission and
    :meth:`OperatorApp._run_slash_command`'s dispatch read. Only one function
    ever decides what a typed word means, so ``/Usage`` cannot echo as one
    command and run as another.
    """
    token = text.split(maxsplit=1)[0].lower() if text.strip() else ""
    if not token.startswith("/"):
        return None
    name = token[1:]
    return next((entry for entry in SLASH_COMMANDS if name in entry.names), None)


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

#: TUI diagnostics go to the log FILE, never the terminal — stderr belongs to
#: the rendered app (see ``local_operator.logger.file_logging``).
logger = logging.getLogger("local_operator.tui.app")

#: How long the second Ctrl+C counts as a DOUBLE press. Short on purpose: this
#: is a deliberate double-tap, not a mode. Long enough and a user interrupting
#: two turns in a row would quit the app by accident, which is the one outcome a
#: stop key must never produce. (omp uses 500 ms for the same gesture; this is a
#: little longer because our first press also has to be READ — it prints the
#: resume command — where omp's only clears the editor.)
DOUBLE_INTERRUPT_WINDOW_S = 1.5

#: Sessions the ``/resume`` picker offers. Higher than the recovery listing's
#: ten because the picker can scroll and filter, and a conversation from a
#: fortnight ago is exactly the one worth searching for; low enough that
#: opening it never costs more than a few dozen short transcript reads.
RESUME_PICKER_LIMIT = 50

#: Class the Screen carries while the session has no content. It selects the
#: boot layout in the stylesheet (centred, clamped input card) and is flipped in
#: exactly one place — see ``OperatorApp._set_welcome_visible``.
BOOT_LAYOUT_CLASS = "boot"

#: Class the Screen carries while the full-page subagent view is open. Same
#: device as the boot class — a mode is one class on the Screen and the rest is
#: the stylesheet's problem — and it is what greys the dock: the composer is
#: read-only in that mode, and chrome belonging to the PARENT session has to
#: recede rather than disappear, or the page stops reading as the same app.
#: Flipped in exactly two places, ``_open_subagent_view``/``_close_subagent_view``.
SUBAGENT_LAYOUT_CLASS = "subagent"

#: Class the SCREEN carries while the `/btw` aside card owns the composer. The
#: transcript is inert then — Enter goes to the card — so it recedes behind it,
#: which is the same rule the subagent page applies to the chrome it leaves
#: behind. Flipped in exactly two places, ``_open_aside``/``_close_aside``.
ASIDE_LAYOUT_CLASS = "aside"

#: Class the input dock carries while the COMPOSER holds focus, which is what
#: brightens the chevron from `dim` to `fg` (D23). NOT the accent: that means
#: "a turn is live", and a composer is focused in nearly every frame, so an
#: accent chevron sat permanently two rows above the band's streaming spinner
#: meaning something else (D5, round 11).
#:
#: A class rather than the `:focus-within` pseudo-class the sheet used to ask
#: for: Textual only re-applies focus styles on nodes reachable by walking UP
#: from the widget that gained or lost focus, and the rule's subject is the
#: chevron — the editor's sibling — so the treatment went on at boot and never
#: went off again. Flipped in exactly one place, see
#: ``OperatorApp._sync_composer_focus``.
COMPOSER_FOCUSED_CLASS = "-composer-focused"

#: How long after a terminal resize the floating overlay cards re-measure
#: themselves. They are hosted in `width: auto` containers, so Textual sends
#: them no resize event, and the dock's re-arrange lands AFTER the refresh
#: callbacks — measured, a card syncing one or two refreshes deep still read
#: the pre-resize composer width. 50 ms is past the arrange on every size the
#: tests sweep and is also the debounce a drag-resize wants: one re-measure per
#: settled size rather than one per intermediate column.
RESIZE_REFIT_DELAY_S = 0.05

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

#: The ``Screen``'s own gutter, both edges together. Every boot measurement is
#: taken inside it — the card's width and the composition's row budget alike —
#: because the percentage the sheet resolves and the rows the layout has to
#: spend are both properties of the CONTENT box, not of the terminal.
SCREEN_INSET = 2


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

    def __call__(self, body: str, kind: NoticeKind = "info") -> None: ...


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
        # Open the aside WITHOUT spending the composer's contents, which is the
        # gesture `/btw <question>` cannot offer: submitting a slash command
        # consumes the whole line, so a user who is half way through a prompt
        # when the side question occurs to them would have to delete their
        # draft to ask it. The draft is stashed and Esc puts it back.
        #
        # `ctrl+b` because TextArea binds neither it nor anything near it, so
        # the composer keeps every editing key it had.
        Binding("ctrl+b", "aside", "Aside", show=False),
        # Promote the aside exchange into the conversation. Only meaningful
        # while the card is open, and the action says so rather than the
        # binding: a key that silently does nothing elsewhere is worse than one
        # that explains itself.
        Binding("ctrl+f", "fork_aside", "Fork aside into the chat", show=False),
        # Esc is the key a user reaches for to make the agent stop, so it is the
        # same stop as Ctrl+C rather than a second, weaker notion of "pause" the
        # engine has no concept of.
        #
        # NOT `priority=True`: a priority binding is matched before the key is
        # dispatched to the focused widget, which stole Esc from the editor's
        # picker-close path — the model/provider/command lists could no longer be
        # dismissed. Bubbling is exactly the precedence wanted: the editor
        # consumes Esc (and stops the event) only while it has a list open, and
        # every other time the key arrives here.
        Binding("escape", "stop", "Stop", show=False),
        # Cycle reasoning effort, the gesture omp uses for the same setting.
        #
        # `priority=True` is REQUIRED here, and for the opposite reason Esc must
        # not have it: Textual's own `Screen` binds `shift+tab` to
        # `focus_previous` (verified against textual 8.2.8's
        # `Screen.BINDINGS`), and the Screen is an ancestor of every widget, so
        # a bubbling binding on the App is reached only after the Screen has
        # already consumed the key. A priority binding is matched before the
        # focused widget sees it, which is also what keeps the composer's own
        # key interception (`Editor._on_key`) out of this — one path for the
        # key, whatever holds focus.
        #
        # It costs reverse focus cycling, which nothing in this app needs: focus
        # moves by Tab, by click, and by the `ToolCard` actions that call
        # `screen.focus_next`/`focus_previous` directly rather than through the
        # key.
        Binding("shift+tab", "cycle_effort", "Cycle reasoning effort", show=False, priority=True),
    ]

    def __init__(
        self,
        session_factory: Callable[[], Awaitable[SessionProtocol]],
        theme_name: str = "dark",
        provider_controller: Any | None = None,
        resume_factory: Callable[[str | None], Awaitable[SessionProtocol]] | None = None,
    ) -> None:
        super().__init__()
        theme_mod.set_theme(theme_name)  # dark is the product's island night
        self._session_factory = session_factory
        # ``/resume <id>`` rebinds the session factory to a resume-specific one
        # (the CLI wires it to ``create_session`` with ``args.resume`` mutated)
        # and reloads — the "proper /resume command" the app is asked for. A
        # bare ``/resume`` lists recent sessions instead.
        #
        # ``None`` is a first-class argument, not a missing one: ``create_session``
        # branches on ``args.resume is not None``, so handing it None asks for a
        # BRAND NEW conversation through the identical path. That is what backs
        # ``/new``, and it is why this is one factory rather than two.
        self._resume_factory = resume_factory
        # Full provider/model/credential/usage facade behind the slash
        # commands; ``None`` degrades /provider /usage /model-switch to
        # pointer notices when it is absent.
        self._providers = provider_controller
        self._session: SessionProtocol | None = None
        self._controller: EventController | None = None
        self._status: StatusLine | None = None
        self._streaming_block: AssistantBlock | None = None
        self._tool_cards: dict[str, ToolCard] = {}
        # Rows for calls the model is still dictating. Separate from
        # `_tool_cards`, which holds calls that are RUNNING: a composing row has
        # no execution behind it yet, and treating the two as one dictionary let
        # an update for a running tool land on a row that had not started.
        self._composing_cards: dict[str, ToolCard] = {}
        self._welcome: WelcomeView | None = None
        #: The empty state as it is CURRENTLY APPLIED to the screen, or `None`
        #: before the first resolution. Held so `_set_welcome_visible` can drop
        #: the calls that ask for the state already on screen — a 396-message
        #: replay asks for "hidden" once per block, and each one re-measured the
        #: whole boot composition for a splash that went away at block one.
        self._welcome_visible: bool | None = None
        #: Set while a session SWAP is replacing the ledger, and read by
        #: `_set_welcome_visible`. The splash is the transcript's empty state,
        #: and mid-swap the transcript is empty only because the replacement's
        #: blocks have not been mounted yet — which is not the same fact. Left
        #: unguarded the user watched conversation A give way to the centred
        #: logo and then to conversation B. Only the SHOW is deferred: hiding
        #: the splash is never wrong, and a swap out of an empty session has to
        #: hide it before the replacement's first block mounts under it.
        self._swapping_session = False
        #: A show `_set_welcome_visible` withheld during a swap. Applied by
        #: `_reload_session` once the answer is settled, which is what still
        #: puts the splash up for a swap onto a genuinely empty session.
        self._welcome_pending = False
        # Whatever held focus when the usage panel opened, so closing it returns
        # the user to the composer they were typing in rather than to nothing.
        # The approval card follows the same discipline for the same reason.
        self._usage_focus_restore: Any | None = None
        self._working_block: WorkingBlock | None = None
        #: What the working line says when nothing narrower is running. Set by
        #: the events that describe the whole turn rather than one row of it
        #: (compaction, a provider retry); everything else is DERIVED from the
        #: live cards, so it cannot drift out of step with the ledger.
        self._working_fallback: str = DEFAULT_ACTIVITY
        #: Whether a compaction pass is in flight, from `compaction_start` to
        #: `compaction_end`. Tracked HERE rather than asked of the session
        #: because both triggers announce themselves through those events, so one
        #: flag covers the automatic pass and `/compact` alike. It is what stops
        #: a prompt being sent into a history that is being rewritten.
        self._compacting: bool = False
        #: Whether the working line on screen was mounted by a compaction (a
        #: manual pass has no turn to mount one), so only the handler that
        #: started it takes it away.
        self._compaction_owns_working_block: bool = False
        #: A prompt typed DURING a pass, sent when the pass ends.
        self._prompt_held_for_compaction: str = ""
        #: This session's OWN spend, accumulated per turn. The number the band
        #: shows is this plus every child's — see :meth:`_spend_total`.
        self._total_cost: float = 0.0
        #: Delegated spend, keyed by job id and holding the LAST cost observed
        #: for that child. A dict rather than a running sum because a child's
        #: figure grows while it works, so each tick has to replace its entry
        #: rather than add to a total; and entries are never removed, because
        #: ``AsyncJobManager`` sweeps settled jobs out of the ledger after a
        #: retention window and a spend counter that goes DOWN when a finished
        #: child is evicted is worse than no counter at all.
        self._subagent_costs: dict[str, float] = {}
        # Auto-naming fires ONCE per session. Latched here rather than on the
        # session holder because the app is what schedules the call, and a
        # second submit arriving while the first title is still in flight
        # must not queue a second provider request.
        self._name_requested: bool = False
        # Last subagent count painted, so the 1 Hz poll repaints only on a
        # real change instead of every tick.
        # (task jobs, bash jobs) last painted, so a repaint only happens on change.
        self._subagents_shown: tuple[int, int] = (0, 0)
        # The dock band (subagent task list + todo list) above the input. Both
        # are refs rather than query_one lookups because they are mounted once
        # and repainted on a scheduler tick — a live handle avoids a relookup
        # per poll and makes the handlers below read as plain calls. Trajectory
        # opens through the same handle's on_open callback, which pushes the
        # child's retained event list as a modal.
        self._subagent_panel: SubagentPanel | None = None
        self._todo_panel: TodoPanel | None = None
        # Serializes interactive login flows so two /login commands can never
        # race the one suspended terminal.
        self._login_lock: Any | None = None
        # ``/loop`` state: one loop at a time, cooperatively cancellable at
        # the turn boundary (never mid-turn, so a turn is never half-applied).
        self._loop_running: bool = False
        self._loop_cancelled: bool = False
        # The one pending tool-approval prompt, if any. The TUI owns approvals
        # (see widgets/approval.py) because the default stdin gate deadlocks
        # under a full-screen app; `_approve_all` is the session-scoped "allow
        # all" the prompt's `a` answer latches, and `_approvals_denied` is the
        # TURN-scoped latch that drains the asks belonging to a turn the user
        # stopped (a queued asker wakes when the front prompt settles, and
        # without the latch it would mount a fresh question for a dead turn).
        self._approval: ApprovalBlock | None = None
        self._approve_all: bool = False
        # What a NEW session opens in, read from config at mount and rewritten
        # by `/approvals default <mode>`. Held beside `_approve_all` rather than
        # re-read per use because the two are ONE state to a reader — "is the
        # gate armed, and will it still be tomorrow" — and the list, the band
        # and the bare report all have to answer both halves from the same pair.
        self._approvals_default_auto: bool = False
        # Which TURN a stop belongs to, rather than a flag someone has to clear.
        # `_turn_epoch` counts turn boundaries; `_approvals_denied_epoch` records
        # the epoch a stop/teardown armed the deny latch in. An asker captures the
        # epoch it entered in and denies if the latch covers that epoch, so a
        # `TurnStarted` racing the wake cannot un-deny a stopped turn's tools.
        self._turn_epoch: int = 0
        self._approvals_denied_epoch: int | None = None
        # Tool cards the last turn boundary marked interrupted. Read once, by
        # `on_turn_ended`, to decide whether an abort still owes the user a
        # standalone notice or has already said it on the rows themselves.
        self._interrupted_cards: int = 0
        # Ctrl+C ladder: one press interrupts, a second within
        # DOUBLE_INTERRUPT_WINDOW_S exits, and a third while the exit is under
        # way ends the process outright.
        self._last_interrupt_at: float = 0.0
        # The one live "ctrl+c again to exit" hint, replaced rather than
        # repeated so a run of interrupts leaves one row instead of N.
        self._exit_hint: NoticeBlock | None = None
        # The MAIN transcript, held rather than looked up. Once the full-page
        # subagent view is open there are two `TranscriptView`s in the screen,
        # so `query_one(TranscriptView)` is ambiguous exactly while a turn may
        # still be streaming into this one — every internal append goes through
        # `_transcript_view()` instead.
        self._transcript: TranscriptView | None = None
        # The full-page subagent view while the app is in that mode, and what
        # the mode has to put back on the way out: the composer's read-only
        # state and whatever held focus when it opened.
        self._subagent_view: SubagentView | None = None
        self._subagent_focus_restore: Any | None = None
        # What the aside borrowed and owes back. The card has no input of its
        # own — the ONE composer is pointed at it — so opening the aside has to
        # stash whatever the user had half typed for the main chat, and Esc has
        # to hand it back. `None` means the aside is closed; `""` is a real
        # value (the draft was empty) and must not collapse into it.
        self._aside_draft: str | None = None
        # The reasoning-effort level the USER picked, or None while the model's
        # own default stands. Held on the app rather than only on the session
        # spec because the session is replaceable — `/new`, `/reload` and
        # `/resume` build a new one — and the choice belongs to the person, not
        # to the conversation they made it in. Re-applied in `_boot_session`,
        # and dropped by `_spec_with_chosen_effort` when a model arrives that
        # cannot take it.
        self._effort_choice: str | None = None
        # The model label the "not adjustable" answer was last given for, so
        # `shift+tab` says it ONCE per model instead of once per press. A user
        # probing an unfamiliar key four times got four warning rows, which is
        # the transcript noise the key's silence exists to avoid.
        self._effort_refusal_shown: str | None = None

    # -- composition --------------------------------------------------------
    def compose(self) -> ComposeResult:
        # The welcome splash is the transcript's EMPTY STATE, so it is mounted
        # INSIDE the transcript rather than beside it: that hands it exactly the
        # region above the input panel with no arithmetic here. It supersedes the
        # old D9 boot-hint line, which was a real transcript block and would have
        # hidden the splash on mount.
        self._transcript = TranscriptView(id="transcript")
        with self._transcript:
            yield WelcomeView(lambda: session_welcome_info(self._session, self._providers))
        # The dock band: subagent task list + todo list, sitting between the
        # transcript and the composer. It is a transparent POSITIONER (zero own
        # height when empty) holding one filled body per panel; the two panels
        # each manage their own `display` so the band collapses to nothing when
        # neither has content. Holding a ref lets the 1 Hz poll and the
        # Subagent*/tool-end handlers repaint it without a relookup per tick.
        self._subagent_panel = SubagentPanel(on_open=self._open_subagent_view)
        self._todo_panel = TodoPanel()
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
            # The dock band (subagent + todo) lives INSIDE the same bottom-docked
            # container as the input shell, ABOVE it (D-15-01). A sibling
            # `dock: bottom` overlapped the input (Textual anchors same-edge
            # docks to the bottom edge independently), and a margin to fix that
            # violates the sheet's one-margin rule. As a child here, the band is
            # a normal-flow row the dock's vertical layout reserves before the
            # shell; it collapses to zero when both panels are hidden.
            with Container(id="band"):
                yield self._subagent_panel
                yield self._todo_panel
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
        # The usage panel shares the toast's layer for the same reason: it is an
        # overlay the user opens to READ, and taking rows from the transcript to
        # show it would scroll away the work they opened it to reason about.
        # Mounted once and kept hidden, so `/usage` never awaits a mount before
        # it can show its loading state.
        with Container(id="usage-host"):
            yield UsagePanel()
        # The aside card, on the same layer and mounted once for the same
        # reasons — with one of its own. `/btw` can be typed while the agent is
        # mid-turn, which is when the question is most likely ("what are you
        # doing?"), so the card must appear on the keystroke rather than after
        # a mount; and it must not take a row from the transcript, because the
        # conversation it is a question ABOUT has to stay legible behind it.
        with Container(id="aside-host"):
            yield AsidePanel()

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

        transcript = self._transcript_view()
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
        # Straight after the band exists and before the session is asked for:
        # the saved mode has to be in force by the time the first tool can ask,
        # and the band has to say so on the boot frame rather than on whichever
        # later event happens to repaint it.
        self._load_approvals_default()
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

    @staticmethod
    async def _warm_session_imports() -> None:
        """Import the factory's heavy dependencies in a thread, before awaiting it.

        ``create_session`` is a coroutine whose body does not yield until it has
        imported the engine, the provider stack and the MCP SDK — roughly 700 ms
        of pure import on a warm disk. Awaited directly from this worker that is
        700 ms in which the app paints nothing and services no key event, so the
        first thing a user types on a fresh launch lands in a frozen screen and
        appears in a burst afterwards. Import releases the GIL for its file I/O
        and between bytecode switches, so doing it in a thread first spreads the
        same work into stalls the compositor can absorb (measured: 699 ms worst
        case before, 16 ms after).

        No error handling here on purpose: ``warm_session_imports`` owns the
        "never raises" contract, and a module that genuinely cannot import is
        the factory's to report in its own words on the next line.
        """
        from local_operator.session_factory import warm_session_imports

        await asyncio.to_thread(warm_session_imports)

    async def _construct_session(self) -> Any:
        """Warm the factory's imports and await it. RAISES; adopts nothing.

        Split from :meth:`_adopt_session` because the two halves want to happen
        at different moments on the two paths through here. On a cold boot they
        are one step. On a session SWAP the construction is the slow half —
        hundreds of milliseconds of factory — and the screen must go on showing
        the conversation being replaced for the whole of it, so the ledger is
        not torn down until this has returned and the replacement can be
        painted in the same breath.
        """
        # Inside the caller's guard, not ahead of it. `warm_session_imports`
        # itself never raises, but the import that reaches it and the thread hop
        # around it can, and anything that escapes the boot worker leaves the
        # user with a splash and no explanation.
        await self._warm_session_imports()
        return await self._session_factory()

    def _adopt_session(self, session: Any) -> None:
        """Take ``session`` as the app's, and put its history on screen.

        SYNCHRONOUS end to end, and that is load-bearing rather than incidental:
        on the swap path the caller clears the old ledger immediately before
        calling this, and the two are one frame only for as long as nothing here
        yields to the event loop. An ``await`` added between the clear and
        :meth:`_render_resumed_history` is an empty transcript the compositor
        gets to paint, and an empty transcript is the boot splash's whole
        precondition. Anything genuinely slow belongs in a worker, the way
        :meth:`_measure_preloaded_context` already puts its measurement in one.
        """
        self._session = session
        # Before the band is painted below: the freshly built spec carries the
        # MODEL's default effort, and a `/reload` or `/new` that dropped the
        # user's chosen level would repaint the band with a level they did not
        # choose and are no longer running.
        spec = _model_spec(session)
        if spec is not None and self._effort_choice is not None and hasattr(session, "set_model"):
            session.set_model(self._spec_with_chosen_effort(spec))
        # The refusal is latched per model, and this session may be on another
        # one; a stale latch would swallow the answer on the model that needs it.
        self._effort_refusal_shown = None
        # Approvals must be answered ON SCREEN from here on: the factory's
        # default gate reads stdin, which this app has taken over, so leaving it
        # installed hangs the first write/exec tool call forever.
        session.set_approval_handler(self.request_tool_approval)
        self._controller = EventController(session, self)
        self._controller.subscribe()
        assert self._status is not None
        self._status.update(
            model_label=session.model_label,
            model_name=_model_name(session),
            effort=_effort_label(session),
            context_window=_context_window(session),
            conversation_name=session.conversation_name,
        )
        self._wire_mcp_status(session)
        self._report_mcp_startup(session)
        self._render_resumed_history(session)
        self._measure_preloaded_context(session)

    async def _boot_session(self) -> None:
        """Await the session factory; on failure surface + offer /reload."""
        try:
            session = await self._construct_session()
        except Exception as error:  # TUI-012: construction error path
            self._on_boot_failed(error)
            return
        self._adopt_session(session)

    def _measure_preloaded_context(self, session: Any) -> None:
        """Fill the context segment before the first turn, off the boot path.

        Without this the reading stays blank until a provider's first usage
        response arrives, which tells the user their context is empty at the
        exact moment it is most loaded: system prompt, environment block,
        skills index and every tool schema are already spent. See
        ``Session.measure_preloaded_context`` for what is counted.

        Called again whenever the tool inventory moves, because MCP servers
        connect AFTER boot and their schemas are the largest single term in
        that sum — a figure measured before they land understates the context
        by more than it reports.

        Run as a worker rather than awaited, for the same reason the rest of
        boot is: resolving system blocks touches the skills index, and the app
        must paint before that finishes.
        """
        measure = getattr(session, "measure_preloaded_context", None)
        if measure is None:  # reduced hosts (embedders, pilot fakes)
            return
        if (
            self._status is not None
            and self._status.context_tokens
            and not self._status.context_is_estimate
        ):
            # A turn already reported the provider's exact count. Nothing local
            # improves on that, so do not even spend the measurement.
            return

        async def run() -> None:
            try:
                tokens = await measure()
            except Exception:  # a status estimate must never take the app down
                logger.debug("preloaded context measurement failed", exc_info=True)
                return
            if not tokens or self._status is None:
                return
            if self._status.context_tokens and not self._status.context_is_estimate:
                # A turn finished while this was in flight; the exact count wins.
                return
            self._status.update(
                context_tokens=tokens,
                context_is_estimate=True,
                context_window=_context_window(session),
            )

        # EXCLUSIVE on its own group. `on_tools_changed` fires per server
        # connect, reconnect and list-changed, so several of these can be in
        # flight, and each re-resolves the system blocks — so they finish out of
        # order. Neither guard above can break the tie: both only ask whether
        # the reading is exact, which is false for every one of them, so the
        # LAST to land wins rather than the newest, leaving the band reporting a
        # smaller inventory than the session now has. The same hazard is already
        # guarded this way for the usage panel. A superseded measurement has
        # nothing worth finishing, so cancelling it costs nothing.
        self.run_worker(run(), name="context-preload", group="context-preload", exclusive=True)

    def _render_resumed_history(self, session: Any) -> None:
        """Replay a resumed session's prior messages onto the transcript.

        ``--resume`` restores the conversation into LLM context but the TUI
        transcript is a separate surface, so without this the app opens on a
        blank screen that reads as a failed resume even though the model sees
        everything.

        This replays what the conversation WAS: prompts, assistant prose, and
        every tool call with the result it got. The previous version rendered
        prompts plus assistant messages that carried no tool calls, on the
        theory that tool rows were too noisy to replay — which sounded
        reasonable and was measurably wrong. An agent turn is
        ``text + tool_calls`` in ONE message, so ``not tool_calls`` excluded the
        prose too: on a real 396-message conversation the old rule mounted
        **6 blocks** — 5 prompts and 1 reply — dropping 74 assistant messages
        that had text, all 215 tool calls and all 215 results. What resumed on
        screen was a list of questions with no answers, which reads as a
        session that never ran rather than one being continued.

        Tool rows are replayed settled, never running, and with no duration:
        see :meth:`ToolCard.restore`. A call whose result is missing from the
        transcript is shown ``interrupted`` rather than complete — that is what
        a session killed mid-turn actually left behind.

        Guarded: a fresh session has an empty history, and a /clear already
        retired the splash — this must not fight either.
        """
        try:
            history = list(session.history())
        except Exception:
            return  # defensive: reduced hosts may lack the accessor

        # Results are keyed by the call they answer, and a tool message can sit
        # several messages after its call (one assistant turn issues a batch).
        # Indexing first is what lets each call render WITH its outcome instead
        # of as a second, orphaned row.
        results: dict[str, Any] = {}
        for message in history:
            if getattr(message, "role", None) == "tool":
                call_id = getattr(message, "tool_call_id", None)
                if call_id:
                    results[call_id] = message

        appended = False
        transcript = self._transcript_view()
        # ONE mount for the whole conversation. Per-block mounting made Textual
        # re-walk its stylesheet, invalidate the container and schedule a settle
        # callback 297 times over on a 396-message session, for a layout that is
        # only looked at once — see `TranscriptView.batch_append`.
        with transcript.batch_append():
            for message in history:
                role = getattr(message, "role", None)
                if role == "tool":
                    continue  # already rendered beside the call that asked for it
                text = getattr(message, "text", "") or ""
                text = text.strip() if isinstance(text, str) else ""
                if role == "user":
                    if text:
                        self._append_block(UserBlock(text))
                        appended = True
                    continue
                if role != "assistant":
                    continue
                if text:
                    block = AssistantBlock()
                    block.update_text(text)
                    block.finalize_text()
                    self._append_block(block)
                    appended = True
                tool_calls = getattr(message, "tool_calls", None) or []
                for call in tool_calls:
                    self._replay_tool_call(call, results)
                    appended = True
                if not text and not tool_calls:
                    # An assistant message with neither prose nor a call is a
                    # turn that FAILED. Skipping it is what left a resumed
                    # session showing a prompt and nothing after it, with no
                    # hint that the answer had errored rather than never been
                    # asked for.
                    if getattr(message, "stop_reason", None) in ("error", "aborted"):
                        reason = "turn failed" if message.stop_reason == "error" else "interrupted"
                        self._append_block(NoticeBlock(reason, "error"))
                        appended = True
        if appended:
            # Replay is mounted as one synchronous batch, before Textual can
            # remeasure the growing container between blocks. Land the reader on
            # the latest turn and ARM the anchor there, so the first thing the
            # resumed session streams carries them with it rather than growing
            # off the bottom of a viewport pinned to the replay's last frame.
            transcript.follow_tail()

    def _replay_tool_call(self, call: Any, results: dict[str, Any]) -> None:
        """Mount one settled tool row for a call from a previous session.

        The card is built exactly as a live one is — same constructor, same
        summary derivation from the arguments — so a resumed row is
        indistinguishable from the row the user watched run, apart from the
        duration the transcript never recorded.
        """
        card = ToolCard(
            getattr(call, "id", "") or "",
            getattr(call, "name", "") or "",
            getattr(call, "arguments", None) or {},
        )
        self._append_block(card)
        result = results.get(getattr(call, "id", "") or "")
        if result is None:
            # No result recorded: the session ended between the call and its
            # answer. Showing it as complete would invent an outcome.
            card.restore(state="interrupted")
            return
        result_text = getattr(result, "text", "") or ""
        payload = getattr(result, "provider_payload", None) or {}
        details = payload.get("details") if isinstance(payload, dict) else None
        if getattr(result, "is_error", False):
            card.restore(
                state="error",
                result_text=result_text,
                details=details,
                error=_first_line(result_text),
            )
        else:
            card.restore(state="success", result_text=result_text, details=details)

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
        # `model_name` goes with the label it belongs to. Leaving it set is not
        # cosmetic: the name is resolved against the label, and a name the
        # registry does not own — an aggregator's listing string such as
        # `MoonshotAI: Kimi K2` — has nothing to collide with, so it would be
        # accepted and painted where the app meant to say "session error".
        self._status.update(model_label="session error", model_name="", streaming=False)

    async def _reload_session(self, *, keep_context: bool = False) -> None:
        """Dispose the current session, boot its replacement, and rebuild the
        ledger from what that replacement ACTUALLY holds.

        The transcript is always cleared and always replayed from the new
        session's own ``history()`` (see :meth:`_render_resumed_history`), so
        the screen is a PROJECTION of the model's context rather than a second
        record of it maintained beside it. The bug that forced that rule: a
        plain ``/reload`` kept the visible ledger while booting a session whose
        history was empty, so the user read a whole conversation off the screen
        while the agent answered "I have no prior turn in this session". The
        missing history was the smaller half of it — nothing detected or
        reported the divergence, because nothing compared the two surfaces.

        ``keep_context`` states the CALLER'S INTENT — ``/reload`` continues this
        conversation, ``/new`` and ``/resume`` deliberately do not — and it is
        verified rather than trusted: a reload that meant to keep the
        conversation and came back with less of it says so, instead of leaving
        the user to discover it from an answer that has forgotten everything.
        """
        previous_id = self._conversation_id()
        previous_len = self._history_length()
        # A subagent page is a window onto THIS session's job ledger. The
        # session is about to be disposed and the ledger with it, so the page
        # would go on standing over a conversation it no longer describes
        # (reporting `gone` for every job) with the composer still read-only.
        # Leaving the mode first is the only outcome that is not a lie.
        self._close_subagent_view()
        # The aside goes too, and this is the general form of the interrupt
        # case. It is a question ABOUT a conversation that is being replaced,
        # answered from a context that is about to be torn down, and it is
        # holding the composer's placeholder, its command list, its prompt
        # history and the user's stashed draft. A card left standing across a
        # session swap keeps every one of those pointed at a session that no
        # longer exists.
        self._close_aside()
        # Deny first: `dispose` AWAITS teardown, and a turn parked on an
        # unanswered on-screen approval never reaches it. Measured: `/reload`
        # with a parked question stalled for the whole 5s dispose budget while
        # unmount with the identical turn returned immediately.
        self._deny_queued_approvals()
        # The working line belongs to the turn being thrown away. Left standing
        # it does two kinds of damage: the widget goes on animating a turn that
        # no longer exists, and the stale `_working_block` reference makes
        # `_start_working_block` return early — so the replacement session's
        # turns get no working line at all, for the rest of the app's life.
        # `clear_blocks` unmounts the widget below but cannot clear a reference
        # it does not own, which is exactly why this is here and not there.
        self._dismiss_working_block()
        # And settle the cards that turn was running. A session torn down
        # mid-turn never delivers the `turn_end` that normally reconciles them,
        # and `_current_activity` derives the working line's label from these
        # two dictionaries — so without this the NEXT session's first turn
        # opened by announcing `running bash` for a bash call that died with
        # the previous session.
        self._retire_live_tool_cards()
        # And the COUNT is per-turn too. It exists so an aborted turn does not
        # print a standalone "interrupted" notice on top of cards that each
        # already say so, and it is cleared only by the turn-end handler, which
        # the dying session never reaches. Left set, the replacement session's
        # first aborted turn would suppress its notice on the strength of cards
        # that belonged to a conversation the user cannot see any more.
        self._interrupted_cards = 0
        if self._controller is not None:
            # BEFORE disposal, always: the ledger is being rebuilt from the
            # replacement session, so the dying session's terminal events have
            # nothing left to land on. They are Textual messages, queued while
            # disposal is awaited and handled after `clear_blocks`, so leaving
            # the controller subscribed contaminates the new conversation with
            # the old one's `agent_end` — including the `context_tokens` and
            # `cost` it carries.
            self._controller.dispose()
            self._controller = None
        if self._session is not None:
            try:
                await self._session.dispose()
            except Exception:
                pass
            self._session = None
        # The pending approval belonged to the session that just died. Left
        # set, the NEW session's first write/exec approval queued behind a
        # question that is no longer on screen and nothing could answer it.
        self._approval = None
        # Same argument, and a sharper failure. `_compacting` and the prompt it
        # holds belong to the session that just died, and their ONLY other
        # writer is `on_compaction_ended` — which can never run after this
        # point, because the controller is disposed above precisely so the
        # dying session's terminal events are dropped. Left set, `_compacting`
        # stays True for the life of the app: every later prompt falls into the
        # hold, is answered "queued — sends when compaction finishes", and is
        # never sent, so a fully booted session can no longer reach the model
        # at all. A manual pass is documented as minutes on a long
        # conversation and slash commands are not gated on it, so `/reload`
        # during one is an ordinary thing for a user to do.
        #
        # The held prompt is not SENT — it was typed into the conversation
        # being thrown away, and replaying it would hand the replacement text
        # its user never addressed to it. But it is not destroyed either: the
        # app promised "queued — sends when compaction finishes" before taking
        # it, and `clear_blocks()` below removes both that promise and the
        # user's own echo of what they typed, so a silent discard loses words
        # the user watched being accepted. It goes back to the composer, which
        # is the repo's existing answer for the same problem — `_close_aside`
        # hands the stashed main-chat draft back rather than dropping it.
        # Sending stays the user's decision, which is what it was before the
        # hold took it.
        self._compacting = False
        held, self._prompt_held_for_compaction = self._prompt_held_for_compaction, ""
        if held:
            editor = self._editor()
            # Never clobber something typed since: the draft in the box is
            # newer than the prompt we are handing back.
            if not editor.text.strip():
                editor.load_text(held)
        # Build the replacement BEFORE tearing the ledger down, and paint the
        # substitution in ONE frame. The clear used to happen here, ahead of
        # `await self._boot_session()`, and the factory is the slow half of a
        # swap — so the compositor spent the whole of it painting an empty
        # transcript, which is the boot splash's entire precondition. What the
        # user reported is exactly that: conversation A, the centred logo, then
        # conversation B. `_construct_session` adopts nothing and `_adopt_session`
        # never yields, so from here the old blocks stay up until the new ones
        # can go up with them in the same refresh.
        #
        # `_swapping_session` closes the same hazard from the other side.
        # `clear_blocks` still fires the empty-state hook a line below, and the
        # flag is what stops that hook resolving a splash for a transcript that
        # is empty only between two statements. The ordering removes the window;
        # the flag removes the transition through the state. Both, because
        # either alone leaves the bug reachable — an `await` added inside the
        # adopt would bring the window back, and a reader who removes the flag
        # gets a splash on any frame that manages to land in the gap.
        self._swapping_session = True
        try:
            try:
                session = await self._construct_session()
            except Exception as error:  # TUI-012: construction error path
                # The clear happens on the failure path too, and this is the
                # discipline it protects: the ledger is only ever as old as the
                # session on screen, so a swap that lands on nothing must not
                # leave the previous conversation standing under an error that
                # says there is no session at all.
                carried_spend = self._reset_ledger_for_swap()
                self._on_boot_failed(error)
            else:
                carried_spend = self._reset_ledger_for_swap()
                self._adopt_session(session)
        finally:
            self._swapping_session = False
            if self._welcome_pending:
                # Nothing the replacement had to show retired it, so the swap
                # really did land on an empty app — `/new`, or a boot that
                # failed. One transition, at the end, instead of one through the
                # middle.
                self._set_welcome_visible(True)
        self._reconcile_reload(previous_id, previous_len, carried_spend, keep_context=keep_context)

    def _reset_ledger_for_swap(self) -> tuple[float, dict[str, float]]:
        """Clear the transcript and the session-scoped band, for a swap.

        Returns the spend totals it just zeroed, for `_reconcile_reload` to put
        back if the reload turns out to have landed on the conversation that was
        already open.

        Split out of :meth:`_reload_session` so the success and failure paths
        can share it and so the whole of it sits between `_construct_session`
        and `_adopt_session` with no await in it — see the ordering argument at
        the call site.
        """
        # Unconditional, and this is the whole discipline: the ledger is only
        # ever as old as the session on screen. Clearing after controller
        # detachment keeps old-session events out of the replacement
        # conversation. The view's public reset is used so streaming,
        # tool-card, approval and welcome-state bookkeeping are reset by the
        # same hook as `/clear`, without `/clear`'s "history is untouched"
        # receipt — which would be a lie on every path through here.
        self._transcript_view().clear_blocks()
        assert self._status is not None
        # A reload may land on a different conversation, so its title and its
        # one naming attempt both reset, or the old name would outlive its
        # session. A `/reload` that continues this conversation re-derives the
        # name from the session it boots, exactly as a `--resume` launch does.
        self._name_requested = False
        # The spend ledger is the dead conversation's — unless the reload lands
        # back on the SAME conversation, which `/reload` now does. Reset it here
        # so `/new` and `/resume` cannot inherit a figure they did not spend,
        # and hand the totals to `_reconcile_reload`, which puts them back when
        # the conversation turns out to be the one that was already open.
        # Money is the segment where a user is least able to tell a
        # carried-over number from a real one, in EITHER direction.
        carried_spend = (self._total_cost, dict(self._subagent_costs))
        self._total_cost = 0.0
        self._subagent_costs.clear()
        # The MCP segment is cleared too: the old session's manager is gone, so
        # a lingering count would describe servers nothing is connected to any
        # more. `_adopt_session` repaints it from the new session's manager.
        #
        # The context reading goes with them, and for a sharper reason than
        # tidiness: an exact ``prompt_tokens`` left standing is what the
        # measurement guard reads to decide the band already knows better than
        # any estimate. It belonged to the conversation that just died, so the
        # replacement session would be denied its own measurement and the band
        # would report a number for history no longer on screen — until the new
        # session's first turn happened to end.
        self._status.update(
            model_label=MODEL_PENDING,
            # Cleared with the label it describes: a name left standing beside
            # `MODEL_PENDING` would be the dead session's model.
            model_name="",
            streaming=False,
            effort="",
            conversation_name="",
            mcp=McpStatus(),
            context_tokens=0,
            context_is_estimate=False,
            # "" empties the segment; `None` would mean "leave it alone" and the
            # dead conversation's figure would stay on screen until the new one's
            # first turn ended.
            cost="",
        )
        return carried_spend

    def _conversation_id(self) -> str:
        """Which conversation is open, or "" when none is."""
        session = self._session
        return getattr(session, "session_id", "") if session is not None else ""

    def _history_length(self) -> int:
        """How many messages the model currently has, as the session reports it."""
        session = self._session
        if session is None:
            return 0
        try:
            return len(session.history())
        except Exception:  # reduced hosts (embedders, pilot fakes) may lack it
            return 0

    def _reconcile_reload(
        self,
        previous_id: str,
        previous_len: int,
        carried_spend: tuple[float, dict[str, float]],
        *,
        keep_context: bool,
    ) -> None:
        """Check the rebooted session against the one it replaced, and SAY so.

        The ledger has already been rebuilt from the new session's own history,
        so the screen cannot disagree with the model by the time this runs. What
        it can be is emptier than the user asked for, and that is what this
        catches: ``/reload`` is a promise to continue this conversation, and
        when the replacement comes back with less of it — no resume-capable
        launcher, a transcript that never reached disk, a boot that failed
        outright — the promise was not kept. Reporting it is the difference
        between a user who knows to re-establish context and the one who found
        out from an answer that had forgotten the question.

        Spend travels the other way. It was charged to a conversation that is
        still open, so without this the band walks its own total back to zero
        every time a reload refreshes the MCP servers.
        """
        if previous_id and self._conversation_id() == previous_id:
            total, children = carried_spend
            self._total_cost = total
            self._subagent_costs.update(children)
            if self._status is not None:
                spend = self._spend_total()
                self._status.update(cost=format_cost(spend) if spend else "")
        if not keep_context:
            return
        remaining = self._history_length()
        if previous_len and remaining < previous_len:
            # `_on_boot_failed` has already said WHY when the boot is what
            # failed; this says what it cost, which is the part the user is
            # about to act on.
            self._system_notice(
                f"reload could not restore this conversation — "
                f"{_messages(previous_len)} are no longer in the model's context",
                "warning",
            )
            return
        if self._session is None:
            return  # nothing was there to lose, and the error notice stands
        self._system_notice(
            f"session reloaded — {_messages(remaining)} still in context"
            if remaining
            else "session reloaded"
        )

    def _cmd_reload(self, notice: NoticeFn) -> None:
        """``/reload`` — re-run boot against the SAME conversation.

        What a reload is FOR is the boot: MCP servers reconnected, credentials
        re-read, the model re-resolved. The conversation is not what went
        wrong, and with ``/new`` standing as the explicit fresh start there is
        no reason for this to discard it. It discarded it anyway: the launch
        factory is ``create_session`` bound to the args the app started with,
        which name no session to resume, so every ``/reload`` minted a NEW
        transcript directory. The model lost the conversation while the screen
        went on showing it.

        So rebind to this conversation's id first, exactly as
        :meth:`_resume_session` does — a reload IS a resume of the session
        already open, and ``Session`` seeds its context from the transcript it
        is pointed at. Gated on that transcript being on disk, because
        ``resume_dir`` refuses an id it cannot find: a reload before the first
        turn persisted would otherwise fail to boot at all. The same gate keeps
        the command's original purpose intact — a session that never
        constructed has no id to rebind to, so it retries the launch factory
        exactly as it always did.
        """
        resume_id = self._resumable_session_id()
        if resume_id and self._resume_factory is not None:
            self._session_factory = lambda: self._resume_factory(resume_id)  # type: ignore[misc]
        # Transient by design: the reload rebuilds the ledger from the session
        # it boots, so this line is the acknowledgement WHILE that happens and
        # `_reconcile_reload`'s receipt is what the user is left holding.
        notice("reloading session…")
        self.run_worker(self._reload_session(keep_context=True), thread=False, group="session")

    def _cmd_resume(self, arg: str, notice: NoticeFn) -> None:
        """``/resume`` — pick a conversation; ``/resume <id>`` — resume one.

        A bare ``/resume`` opens :class:`SessionPickerScreen`: choosing a past
        conversation is a two-way question, so it gets a surface that can hold
        a cursor and hand an answer back, rather than a block of ids printed
        into the transcript that the user then has to read and retype. Rows
        are named by their opening message, because a column of hex ids is not
        something anyone recognises their own work in.

        An explicit id (or the ``@latest`` sentinel) skips the picker
        entirely — a user who already knows which session they want, or who is
        replaying a command from their shell history, should not be made to
        answer a prompt. Without a CLI-provided resume factory the resume path
        is unavailable and the command says so instead of opening a picker
        whose every choice would fail.
        """
        from local_operator.paths import config_dir
        from local_operator.resume import RESUME_LATEST, recent_session_rows

        # Rejections go through ``_system_notice`` (see `_cmd_usage`): nothing
        # ran, so the boot composition must survive them.
        if self._resume_factory is None:
            self._system_notice("resume requires a resume-capable launcher — see CLI", "warning")
            return

        if not arg:
            rows = recent_session_rows(config_dir(), limit=RESUME_PICKER_LIMIT)
            if not rows:
                self._system_notice("no previous sessions to resume", "warning")
                return

            def _resume_choice(session_id: str | None) -> None:
                # Dismissed with Esc (or on an empty filter) — the session on
                # screen is left exactly as it was, with nothing said: a
                # cancelled picker is not an event worth a transcript line.
                if session_id:
                    self._resume_session(session_id, notice)

            self.push_screen(SessionPickerScreen(rows, time.time()), _resume_choice)
            return

        # ``@latest`` is the oldest part of the CLI vocabulary (--resume
        # accepts it), so the sentinel must survive verbatim: resume.py only
        # resolves the newest session on an EXACT ``RESUME_LATEST`` match.
        # A bare arg is the same request, spelled the way a user would type it
        # without remembering the symbol.
        self._resume_session(arg.strip() or RESUME_LATEST, notice)

    def _resume_session(self, resume_id: str, notice: NoticeFn) -> None:
        """Rebind the factory to ``resume_id`` and reboot onto that session.

        Shared by the picker and by ``/resume <id>`` so both paths reload
        identically. Failures inside the new factory surface through
        ``_on_boot_failed`` exactly as a bad ``--resume`` does.
        """
        if self._resume_factory is None:
            self._system_notice("resume unavailable: no resume-capable launcher", "warning")
            return
        self._session_factory = lambda: self._resume_factory(resume_id)  # type: ignore[misc]
        notice(f"resuming session {resume_id}…")
        self.run_worker(self._reload_session(), thread=False, group="session")

    def _cmd_new(self, notice: NoticeFn) -> None:
        """``/new`` — start a fresh conversation without leaving the app.

        There was no way to do this: ``/clear`` wipes the SCREEN and keeps the
        conversation the model sees, ``/reload`` reboots the SAME conversation,
        and ``/resume`` moves to a different existing one. Starting genuinely
        fresh meant quitting and relaunching, which also throws away the
        terminal state, the MCP connections and the warm imports.

        Implemented through the resume factory with ``None`` rather than a
        second factory: ``create_session`` already branches on
        ``args.resume is not None``, so this is the same code path a cold
        launch takes, which is exactly what "new session" should mean. The
        ledger is rebuilt from the session that boots, as it is on every
        reload, which for a conversation with no history is an empty screen.
        """
        if self._resume_factory is None:
            self._system_notice("new session unavailable: no session-capable launcher", "warning")
            return
        self._session_factory = lambda: self._resume_factory(None)  # type: ignore[misc]
        notice("starting a new session…")
        self.run_worker(self._reload_session(), thread=False, group="session")

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
                # The incumbent above merged the new tools into the session, so
                # the context estimate this repaints is now measurably wrong —
                # MCP schemas are the biggest term in it.
                self.call_later(self._measure_preloaded_context, session)

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
        """Re-fit size-sensitive chrome after a terminal resize."""
        if self._status is not None:
            self.call_after_refresh(self._status.refresh)
        # The EVENT's size, not the app's: during a resize `self.size` is still the
        # previous frame's, and one stale cell is enough to put the card threshold on
        # the wrong side of itself — at 85 columns it decided "bar" for a box that was
        # about to be exactly wide enough for the card. It is also the last handler
        # to run BEFORE the screen arranges, which is what lets the composition below
        # land in the first frame the terminal ever sees.
        self._sync_boot_layout(size=event.size)
        # The overlay cards are hosted in `width: auto` containers, so a
        # terminal resize does not resize THEM and Textual delivers them no
        # event — the aside in particular is sized AND PLACED from the
        # composer's column, so without this it keeps the old width and the
        # shared edges that are its whole composition come apart.
        #
        # A TIMER, not `call_after_refresh`, and that is measured rather than
        # cautious: the dock's re-arrange lands after the refresh callbacks, so
        # both one and two refreshes deep the card still read the pre-resize
        # shell width (118 cells at a terminal that had just become 60) and
        # painted itself to it. `force` then stops the card's own no-op guard
        # from comparing two stale numbers and agreeing with itself.
        #
        # ``RESIZE_REFIT_DELAY_S`` is also a debounce, which a drag-resize
        # needs anyway: one re-measure per settled size instead of one per
        # intermediate column.
        self.set_timer(RESIZE_REFIT_DELAY_S, lambda: self._sync_overlay_layout(force=True))
        # And one BEST-EFFORT pass right away. It is often wrong about the
        # width (the dock has not re-arranged) but it is never wrong about the
        # screen having SHRUNK, and repainting the card at whatever the screen
        # can currently hold keeps the ~50 ms before the timer from showing a
        # card overhanging the frame with its prose clipped mid-word.
        self.call_after_refresh(self._sync_overlay_layout, force=True)

    def on_welcome_view_block_resized(self, message: WelcomeView.BlockResized) -> None:
        """The splash changed height, so the composition around it has moved.

        The block grows and shrinks after the first frame — the model label lands
        when the session factory resolves, and a credential warning appears with
        it. Without this the boot composition would be centred for the block that
        was measured at startup and left there.
        """
        self._sync_boot_layout()

    def _sync_boot_layout(self, *, size: Size | None = None) -> None:
        """Re-measure everything about the boot layout that the sheet cannot.

        Called from ``on_resize``, from ``_set_welcome_visible`` and from the splash
        itself when its block changes height: a resize is not the only way the
        answers change — the layout comes and goes with the splash, and on the first
        frame no resize event has happened yet. ``size`` is the resize event's, when
        there is one.

        Everything below is ARITHMETIC on quantities this method cannot invalidate,
        and it is applied synchronously. That is the whole fix for TUI boot: the
        composition used to measure a laid-out frame and re-measure itself one
        refresh later, but ``call_after_refresh`` resumes before the compositor has
        re-arranged, so every pass read last frame's splash offset against the
        padding it had just written and double-counted its own reserve. The lift
        overshot, undershot, and converged over two dozen PAINTED frames — the
        splash falling from the top of the screen while the card rose from the
        bottom until the two met. That is the "logo and screen coming together"
        effect; it was never an animation anyone declared.
        """
        size = self.size if size is None else size
        # Card first: the shell's width decides how tall the shell measures, which
        # is a term in the composition below.
        self._sync_boot_card(size.width)
        self._sync_boot_composition(size)

    def _sync_boot_card(self, terminal_width: int) -> None:
        """Decide whether this terminal is wide enough for the boot CARD.

        The stylesheet cannot ask how wide the terminal is, so the width the card
        WOULD take is resolved here and the class only goes on when the ground left
        beside it is wide enough to read as a margin. Computing a width in Python
        for want of a media query is the same move ``Toast._refit`` already makes.
        """
        # The content box, not the terminal: `Screen`'s inset is outside the
        # percentage the sheet resolves.
        box = max(0, terminal_width - SCREEN_INSET)
        card = boot_card_width(box)
        self.screen.set_class(box - card >= BOOT_CARD_MIN_INSET, BOOT_CARD_CLASS)

    def _sync_boot_composition(self, size: Size) -> None:
        """Centre the boot composition — splash, separator, card — in the screen.

        The splash and the card are ONE block to the eye, and on a tall terminal
        resting that block on the bottom of the screen left the upper two thirds
        empty. The stylesheet cannot fix it: the card is DOCKED, so no alignment
        reaches it, and how many rows to reserve below it depends on how tall the
        splash turned out at this width. So the composition's two chrome
        quantities — the ground row above the card and the slack below it — are
        resolved here, the same way ``Toast._refit`` resolves a width the sheet
        cannot ask for.

        Both are CONDITIONAL, and on the same measurement. Every row this reserves
        comes out of the splash's budget (``WelcomeView.spare_rows``), and the
        splash pays in whole sections — a 28-row terminal that reserved rows to
        centre a block that already fills the region would trade the mark for air.
        So they are only spent out of rows that are empty anyway, which is why a
        96x28 frame is untouched and a 190x48 one is centred.

        NOTHING here is read back off a laid-out frame. Both terms are asked of the
        widgets as functions of a size — ``get_content_height`` for the dock's
        children, ``spare_rows`` for the splash — so the answer does not depend on
        the reserve that is about to be written, and one pass is final. Reading the
        frame is what made this measurement circular, and a circular measurement
        applied once per paint is an animation.
        """
        dock = self.query_one("#input-dock", Container)
        welcome = self._welcome
        if not self.screen.has_class(BOOT_LAYOUT_CLASS) or welcome is None or not welcome.display:
            # The conversation layout is a full-width bar with nothing under it.
            # Rows left reserved here would be a hole below a populated transcript.
            self._reserve_boot_rows(dock, gap=False, lift=0)
            return
        box = Size(max(0, size.width - SCREEN_INSET), max(0, size.height - SCREEN_INSET))
        transcript = self._transcript_view()
        # Rows the region above the card would have with NO reserve in it, and the
        # width the splash is drawn at. The transcript's own gutter is part of the
        # sheet's boot layout (it drops its top row there) and part of neither
        # widget's content height, so it comes out of the budget here. So does the
        # scrollbar column: `scrollbar-gutter: stable` reserves it permanently
        # (D27) and `styles.gutter` does not count it, so a width measured without
        # it is one cell wider than the splash can ever be drawn. One cell is
        # nothing in the middle of a degradation tier and everything at its edge —
        # at 25 columns it measured the 22-cell block (19 rows) for a splash that
        # renders at 21 (9 rows), and centred the frame around ten rows that are
        # not there. The app and the layout engine can only agree here if this
        # width is the one the engine will hand the widget.
        gutter = transcript.styles.gutter
        region = max(0, box.height - self._boot_dock_height(dock, box, size) - gutter.height)
        width = max(0, box.width - gutter.width - transcript.scrollbar_size_vertical)
        spare = welcome.spare_rows(region, width)
        if spare < BOOT_COMPOSITION_MIN_SPARE:
            self._reserve_boot_rows(dock, gap=False, lift=0)
            return
        # One row buys the separator; the rest is split between the ground above
        # the splash and the ground below the card, the card taking the smaller
        # half so an odd row lands overhead where the eye does not read it as a
        # gap under the panel.
        self._reserve_boot_rows(dock, gap=True, lift=(spare - 1) // 2)

    def _boot_dock_height(self, dock: Container, box: Size, viewport: Size) -> int:
        """Rows the input dock occupies with no composition reserve in it.

        Asked of the dock's CHILDREN rather than read off ``dock.outer_size``,
        because the dock's own height already contains the lift this measurement
        exists to compute. The children — the subagent/todo band and the input
        shell — are content-sized and answer from their own state, so the total is
        available before the first arrange and cannot move under the reserve.
        """
        width = boot_card_width(box.width) if self.screen.has_class(BOOT_CARD_CLASS) else box.width
        total = 0
        for child in dock.children:
            if not child.display:
                continue
            gutter, margin = child.styles.gutter, child.styles.margin
            inner = max(0, width - gutter.width)
            total += (
                child.get_content_height(Size(inner, box.height), viewport, inner)
                + gutter.height
                + margin.height
            )
        return total

    def _reserve_boot_rows(self, dock: Container, *, gap: bool, lift: int) -> None:
        """Apply the composition's separator row and its lift, once.

        The separator is the app's ONE vertical separator class rather than a second
        spacing rule: a receipt under the splash and the card under the hints are
        the same rhythm case, one row of ground where a block change needs reading
        as a block change. The lift is padding INSIDE the dock — the dock is the
        positioner, so reserving rows in it moves the panel it holds without
        anything else in the layout knowing.

        It does NOT schedule a re-measure. The caller's arithmetic is already the
        fixed point, and a self-rescheduling measurement is what animated the boot
        screen; the equality gate is now only here to keep an unchanged sync from
        dirtying the layout.
        """
        if dock.has_class(GAP_CLASS) == gap and dock.styles.padding.bottom == lift:
            return
        dock.set_class(gap, GAP_CLASS)
        dock.styles.padding = (0, 0, lift, 0)

    # -- input --------------------------------------------------------------
    def on_editor_submitted(self, message: EditorSubmitted) -> None:
        """Slash commands run synchronously BEFORE any prompt is sent."""
        text = message.text.strip()
        if not text:
            return
        # The aside owns the composer while it is up. EVERYTHING goes to it,
        # slash-shaped lines included: the card is a MODE, its footer says so
        # (`esc close · enter ask again`) and its placeholder says so, and a
        # composer that ran `/model` from inside a popup about the last turn
        # would be answering a question the user cannot see they asked. Esc is
        # one keystroke away and it is the only way out, which is the whole
        # bargain a modal surface makes.
        if self._aside_is_open():
            self._ask_aside(text)
            return
        if text.startswith("/"):
            # Nothing but dispatch. A slash command writes its own rows: the
            # ledger row for the one command permitted one (``SLASH_COMMANDS``,
            # ``echo``) is written by its HANDLER, at the point its effect
            # landed, because only the handler knows whether it did.
            #
            # No ``_set_welcome_visible(False)`` here any more, for the same
            # reason. It was a second authority over the boot composition,
            # contradicting the one ``_append_block`` documents itself as
            # owning, and it only ever agreed with anything because the echo
            # below it always appended. Without that, it would retire the splash
            # for a command that draws nothing into the transcript — a
            # ``/usage`` panel over a screen with no splash and no ledger.
            self._run_slash_command(text)
            return
        self._submit_prompt(text)

    def on_editor_quit(self, message: EditorQuit) -> None:
        self.exit()

    def on_descendant_focus(self, event: DescendantFocus) -> None:
        """Repaint the composer's focus affordance whenever focus lands.

        The stylesheet used to ask for this with `#input-dock:focus-within`, and
        it never turned back off: Textual re-applies styles after a focus change
        by walking up from the widget that gained or lost focus looking for an
        ancestor whose rules mention `focus-within`, and it records that flag on
        the node the rule SELECTS — the chevron, a SIBLING of the editor and so
        never on that walk. Measured on the shipped build, blurring the composer
        left the chevron painting its focused ink on every later frame while
        `:focus-within` itself correctly reported False.

        Both events, because a focus change is two facts: something took focus
        and something lost it, and only the pair covers "the composer went dark
        because the usage panel lit up".
        """
        self._sync_composer_focus()

    def on_descendant_blur(self, event: DescendantBlur) -> None:
        self._sync_composer_focus()

    def _sync_composer_focus(self) -> None:
        """Mark the input dock while the composer — and only it — has focus.

        On the dock rather than the editor because the affordance it drives is
        the chevron, which is the editor's sibling and unreachable from it in a
        stylesheet with no sibling combinator. Class changes go through
        `App.update_styles`, which re-applies the node AND its descendants, so
        the chevron follows in both directions.

        Reading `has_focus` rather than trusting the event makes this
        idempotent: two events describe one transition and either order has to
        land on the same class, and a path that moves focus without an event we
        handle still resolves the next time one fires.

        `NoMatches` and nothing wider: the only expected miss is a stripped
        harness with no composer mounted. Swallowing every exception here would
        hide a rename behind a silently dead affordance, which is precisely the
        failure this method exists to fix.
        """
        try:
            dock = self.query_one("#input-dock")
            focused = self._editor().has_focus
        except NoMatches:
            return
        dock.set_class(focused, COMPOSER_FOCUSED_CLASS)

    def on_interrupt_requested(self, message: InterruptRequested) -> None:
        """Ctrl+C from the composer — the NORMAL path, since it holds focus.

        Routed through the action rather than straight to ``_interrupt`` so the
        double-press ladder applies here too; going direct meant the app-level
        binding had the ladder and the key people actually press did not.
        """
        self.action_interrupt()

    def on_stop_requested(self, message: StopRequested) -> None:
        """Esc from the composer (the editor consumes the key so it never blurs)."""
        self.action_stop()

    def action_interrupt(self) -> None:
        """Ctrl+C: interrupt the turn. Twice in quick succession: leave.

        Leaves the full-page subagent view first, for the same reason
        ``action_clear_transcript`` does: this key acts on the PARENT's turn
        and its "ctrl+c again to exit" warning is a transcript block, so from
        inside the mode the interrupt landed on a surface the user could not
        see. The second press then quit the app with the warning never
        rendered — the one outcome that ladder exists to prevent.

        A single Ctrl+C must never exit — the whole point of the first press is
        to stop the agent and keep the session. But a user who wants OUT should
        not have to find ``/exit``, so the second press within
        :data:`DOUBLE_INTERRUPT_WINDOW_S` quits and prints the command that
        brings this exact session back (see :meth:`resume_hint`).

        There is deliberately NO third "hard exit" rung. omp has one because its
        teardown can hang on an extension's IPC; ours cannot outlive
        ``Session.dispose``, which bounds its own wait at 5 s. And it could not
        work anyway: Textual stops dispatching input once ``exit()`` is called,
        so a further Ctrl+C never reaches this method — a rung that cannot fire
        is worse than no rung, because the docstring promises an escape the user
        does not have.
        """
        now = time.monotonic()
        if now - self._last_interrupt_at < DOUBLE_INTERRUPT_WINDOW_S:
            self._interrupt()  # stop the work before dropping the terminal
            self.exit()
            return
        self._last_interrupt_at = now
        self._interrupt()
        # Short, and WITHOUT the command. The full `local-operator --resume <id>`
        # belongs in the exit block printed to the terminal, where it can be
        # copied; in a transcript line it is the unreachable copy (the alt screen
        # is discarded on exit), it wrapped at 60 columns and split across rows at
        # 30, and every interrupt added another identical row. What this line owes
        # the user is that a second press exits and that doing so is recoverable.
        #
        # `warning`, not `info`: it says the app is one keystroke from closing,
        # which is the loudest thing on the frame when it appears.
        #
        # Replaced rather than repeated, so three interrupts leave one hint.
        self._close_subagent_view()
        # The aside goes with it, for the reason the hint below exists at all.
        # The card floats over the transcript at one elevation step, so a
        # notice appended behind it is drawn where it cannot be read — and this
        # particular notice is the ONLY warning before a second press quits the
        # app. It is also the rule `action_stop` already states and honours:
        # a key that acts on the conversation returns to the conversation
        # first. Ctrl+C keeps its meaning; it just stops doing it invisibly.
        self._close_aside()
        resumable = bool(self.resume_hint())
        text = "ctrl+c again to exit" + (" — the session can be resumed" if resumable else "")
        if self._exit_hint is not None:
            self._transcript_view().remove_block(self._exit_hint)
        self._exit_hint = NoticeBlock(text, "note")
        self._append_block(self._exit_hint)

    def _resumable_session_id(self) -> str:
        """This conversation's id IF ``--resume`` could reopen it, else "".

        Gated on the transcript EXISTING, not merely on having a session id:
        ``resume_dir`` refuses an id whose transcript is not on disk (a typo
        must fail rather than open an empty session that looks resumed), so an
        id offered before the first turn persisted is one every resume path is
        guaranteed to reject.

        Read by :meth:`resume_hint`, which must not advertise a command that
        cannot work, and by :meth:`_cmd_reload`, which must not rebind the
        session factory to an id that would fail to boot.
        """
        session = self._session
        if session is None:
            return ""
        session_id = getattr(session, "session_id", "")
        if not session_id:
            return ""
        from local_operator.paths import config_dir
        from local_operator.resume import TRANSCRIPT_NAME

        transcript = config_dir() / "sessions" / session_id / TRANSCRIPT_NAME
        return session_id if transcript.is_file() else ""

    def resume_hint(self) -> str:
        """``local-operator --resume <id>`` for this session, or "" when there is none.

        Read by :func:`run_tui` after the app has released the terminal, so the
        command lands in the user's scrollback where they can copy it, rather
        than in a frame that is about to be torn down.
        """
        session_id = self._resumable_session_id()
        return f"local-operator --resume {session_id}" if session_id else ""

    def _interrupt(self) -> None:
        """Abort the running turn AND stop any ``/loop`` in flight.

        Without cancelling the loop, an interrupt would abort one turn and the
        loop would immediately submit the next — the user would have to press
        Ctrl+C once per remaining iteration to actually stop.
        """
        if self._loop_running:
            self._loop_cancelled = True
        # A turn parked on an approval cannot see the abort signal until the
        # callback returns, so the prompt is denied first: aborting alone would
        # leave the engine waiting on a future nobody is going to answer.
        self._deny_queued_approvals()
        if self._session is not None:
            self._session.abort("interrupted")

    def action_stop(self) -> None:
        """Esc: stop. One press, one meaning, wherever focus happens to be.

        A pending approval is NOT answered here and left at that. Esc used to
        deny just the front prompt and return, which meant that with two
        approval-gated tools in one batch the user had to press it once per
        prompt before it finally meant "stop" — and each press looked like it
        had done nothing to the run. Stopping denies every queued prompt (via
        the latch in :meth:`_deny_queued_approvals`) and aborts the turn.

        ``n`` remains the answer that refuses ONE tool and lets the turn carry
        on; that is the distinction Esc should not have been carrying.

        Dismissing the aside comes FIRST, ahead of even the subagent page: the
        card floats over everything, its own footer says ``esc close``, and it
        is holding the user's main draft hostage until the key is honoured.
        Esc meaning "abort the agent's turn" while a popup is up promising to
        close would abort work from a surface the user believes is a side
        conversation — and the turn is exactly what they were asking about.

        Leaving the full-page subagent view is next, and consumes the key for
        the same reason: it is the one thing the mode's own hint promises.

        With nothing running Esc does nothing — in particular it must not clear
        the composer, which would throw away typed text on the key people press
        to cancel.
        """
        if self._close_aside():
            return
        if self._close_subagent_view():
            return
        pending = self._approval is not None and not self._approval.answered
        if pending or (self._session is not None and self._session.is_streaming):
            self._interrupt()

    # -- tool approvals -------------------------------------------------------
    async def request_tool_approval(
        self, tool_name: str, description: str, *, job_id: str | None = None
    ) -> bool:
        """The session's approval gate while this app owns the terminal.

        Awaited by the engine on its own loop (the harness awaits the callback
        inline before executing a write/exec tier tool), which is the same loop
        the app runs on — so the prompt is mounted directly here rather than
        marshalled across threads.

        Serialization: only ONE prompt is live at a time. A tool batch can ask
        twice concurrently, and two cards competing for focus would leave the
        second unanswerable; the later ask waits for the earlier card to settle
        and is then asked in turn.

        The deny latch is re-read after every await, and that is what makes the
        stop paths correct. Settling only the FRONT prompt was not enough: the
        queued asker woke, saw no live prompt, and mounted a BRAND NEW question
        — after Ctrl+C had already aborted the turn. Worse, write/exec tier
        tools are not interruptible, so the runner parked on this callback is
        settled by nothing but this future; the post-abort question was genuinely
        live, and on teardown it mounted into a screen that was going away.

        ``job_id`` names the background job this call belongs to, and it is what
        keeps the latch honest. The latch is TURN-scoped — ``_turn_epoch``
        advances on a parent ``TurnStarted`` and on nothing else — so a subagent
        still running after its parent's turn ended carried that turn's dead
        epoch, and any stop latched in it denied the child's write/exec tools
        with no card mounted and no way for the user to see it had happened. A
        denial nobody is shown reads as the tool simply not working.

        Refusing to apply the latch to a job costs no stop that exists:
        ``Session.abort`` only aborts the parent's turn signal and never touches
        ``self.jobs``, and ``action_stop`` is gated on the parent streaming, so
        Esc could not stop a background child before this change either. Making
        it able to is a real feature — an explicit ``jobs.cancel``, not a
        silent denial standing in for one.
        """
        # Captured BEFORE the first await: this asker belongs to the turn that was
        # running when the engine called it, and no later turn's start can move it.
        epoch = self._turn_epoch
        if job_id is None and self._approvals_are_denied(epoch):
            return False
        if self._approve_all:
            return True
        while self._approval is not None and not self._approval.answered:
            await self._approval.wait()
            # Same exemption as the check above, and it has to be: a job's ask
            # that merely QUEUED behind a foreground prompt would otherwise be
            # denied on waking by a latch that has nothing to do with it.
            if job_id is None and self._approvals_are_denied(epoch):
                return False
            if self._approve_all:
                return True
        # The prompt is a transcript block, and the subagent page hides the
        # transcript — mounted behind it the question was invisible while
        # still taking focus, so the page's own ↑↓ stopped working and the
        # turn parked on an answer the user could not see they owed. The app
        # needs an answer, so the mode yields.
        self._close_subagent_view()
        # The aside yields for the identical reason, one layer up: the card
        # floats OVER the transcript, so the question would be drawn behind it
        # while still taking focus off the composer the card is pointed at —
        # a turn parked on an answer the user can neither see nor reach.
        self._close_aside()
        block = ApprovalBlock(tool_name, description, on_answer=self._latch_approval_answer)
        self._approval = block
        self._append_block(block)
        # The turn is now parked on the user, and the working line says so —
        # this is the one wait in a turn that the agent is not responsible for.
        self._refresh_working_activity()
        try:
            return await block.wait()
        finally:
            block.restore_focus()
            if self._approval is block:
                self._approval = None
            self._refresh_working_activity()

    def _latch_approval_answer(self, answer: str) -> None:
        """What an answer means beyond the one call. Runs BEFORE the future.

        Called synchronously from :meth:`ApprovalBlock.resolve` rather than off a
        posted message: a queued asker wakes the instant the future resolves, and
        a flag latched a few pump hops later was read stale — so the user pressed
        "allow all" and was immediately asked again for the next tool of the same
        batch.
        """
        if answer == "a":
            self._set_approve_all(True)
            # States the CHANGE, not its duration: "for the rest of this
            # session" stays in the transcript at full warning after
            # `/approvals ask` has re-armed the gate, leaving the loudest ink on
            # screen asserting something no longer true.
            self._append_block(
                NoticeBlock(
                    "tool approvals: auto — /approvals ask restores prompting",
                    "warning",
                )
            )

    def _set_approve_all(self, approve_all: bool) -> None:
        """THE writer of the auto-approve mode, band included.

        One method because the band is an ASSERTION about the gate, and the two
        were set from four places apiece with nothing tying them together. That
        is not hypothetical tidiness: an owner's frame showed `! auto-approve`
        beside two tool calls reporting `User denied approval`, and the first
        hour of diagnosing it went on establishing whether the band was lying —
        which the code could not answer, because nothing made it impossible.
        Now the band cannot say anything the gate does not do, since the only
        way to change either is to change both.

        The persisted default rides along for the same reason: `! auto-approve`
        and `! auto-approve always` are two different promises, and which one is
        true is decided by the pair of flags this method owns.
        """
        self._approve_all = approve_all
        if self._status is not None:
            self._status.update(
                approvals_auto=approve_all, approvals_always=self._approvals_default_auto
            )

    def _deny_queued_approvals(self) -> None:
        """Refuse the live prompt AND every ask still queued behind it.

        The latch is the load-bearing part: the queued asker is parked on the
        front prompt's future, so settling that future WAKES it, and without a
        latch it would go on to mount a fresh question for a turn that is being
        stopped or an app that is being torn down.

        Only for paths that END the turn (stop, teardown). A path that merely
        clears the screen must use :meth:`_settle_live_approval`, because the
        turn is still running and its remaining tools still deserve to ask.

        Recorded as the EPOCH it was armed in rather than as a bare flag. A bare
        flag has to be cleared by someone, the only available someone was the
        next ``TurnStarted``, and that put a security gate's correctness on the
        order of an asyncio future callback against the Textual message pump: a
        ``TurnStarted`` already in the queue when the stop lands cleared the latch
        before the parked asker re-read it, and the stopped turn's tool got a
        fresh question. An epoch cannot be cleared early because it is not
        cleared at all — a later turn simply carries a higher number.
        """
        self._approvals_denied_epoch = self._turn_epoch
        self._settle_live_approval()

    def _settle_live_approval(self) -> None:
        """Refuse the VISIBLE prompt only, leaving later asks free to ask.

        Used by ``/clear``: the widget holding the future is about to be removed,
        so the question cannot be left pending — but the turn was not stopped, and
        latching the deny flag here denied every later write/exec tool of the run
        with no prompt at all, while ``/approvals`` reported the opposite.
        """
        approval = self._approval
        if approval is not None and not approval.answered:
            approval.resolve(False)
            approval.restore_focus()
        self._approval = None

    def _approvals_are_denied(self, epoch: int) -> bool:
        """Is an ask from ``epoch`` refused by a stop that has already happened?

        ``<=`` and not ``==``: a stop in epoch 4 refuses the asks of epoch 4 and,
        on teardown, anything still parked from earlier. A strict equality check
        would let an asker that entered before the stop mount its question after.
        """
        armed = self._approvals_denied_epoch
        return armed is not None and epoch <= armed

    def _answer_live_approval_as_allowed(self) -> None:
        """Answer the visible prompt with YES, the way the ``A`` key does.

        Through the widget rather than around it, so the row repaints as a
        receipt (`✓ allowed ...`) instead of vanishing with no record of what the
        user just authorised.
        """
        approval = self._approval
        if approval is not None and not approval.answered:
            # `y`, not `a`: the COMMAND is what changed the mode, and it prints
            # its own notice. Answering as the `A` key would run the keystroke's
            # latch hook too, so the frame carried the same statement twice in the
            # loudest ink it has.
            approval.resolve(True, answer="y")
            approval.restore_focus()
        self._approval = None

    def _allow_approvals_again(self) -> None:
        """Retire the deny latch so tools can ask questions again.

        Only for ``/approvals ask``, whose whole promise is "tools will prompt
        again"; a latch left armed would make that statement false for the rest
        of the run. The turn boundary deliberately does NOT call this — a turn
        that starts simply carries a higher epoch than the latch, so the drain
        cannot be cut short by message ordering.
        """
        self._approvals_denied_epoch = None

    def action_clear_transcript(self) -> None:
        # Ctrl+L is global, so it can land while the subagent page is open. It
        # acts on the CONVERSATION, so it returns to the conversation first —
        # wiping a transcript the user cannot currently see is the one outcome
        # that would read as the key having done nothing.
        self._close_subagent_view()
        # Same rule for the aside card, and the same sentence: Ctrl+L acts on
        # the CONVERSATION, and the aside is a floating question about it that
        # is also holding the composer. Wiping the ledger under an open card
        # leaves the user looking at a question whose subject is gone.
        self._close_aside()
        self._clear_transcript()

    def _cmd_approvals(self, arg: str, notice: NoticeFn) -> None:
        """``/approvals [default] [ask|auto]`` — the mode, and how long it lasts.

        "Allow all" disarms a safety gate, so it needs a stated mode and a route
        back; a one-way switch answered by a single keystroke is the part that
        made it a footgun rather than a shortcut. Bare ``/approvals`` reports,
        which is also how a user who cannot remember what they pressed finds out.

        ``default`` is the SAME word ``/model`` uses for the same promotion, in
        the same position, and that is the whole argument for spelling it this
        way rather than as ``/approvals-default`` or a ``--save`` flag: a user
        learns "put ``default`` in front of the value to keep it" once and it
        holds on the second command they meet it on. It also composes the same
        way — ``/model default p/id`` switches AND saves, and so does this, so
        nobody has to run two commands to end up in the state they asked for.

        The two scopes are genuinely different commitments — "auto-approve for
        the next hour" versus "auto-approve every session from now on" — so the
        receipts do not share a sentence: a session change says ``(this
        session)`` and names the command that would make it stick, and a saved
        one names the FILE and the KEY, which is the only claim a user can check
        without quitting.
        """
        argument = arg.strip().lower()
        persist = argument == "default" or argument.startswith("default ")
        mode = argument[len("default ") :].strip() if persist else argument
        if persist and not mode:
            # Bare `/approvals default` means "keep the mode I am in", exactly as
            # bare `/model default` means "keep the model I am on". The sentence
            # a user has after switching is "make THIS the default", and making
            # them retype the word they just typed is a transcription exercise.
            mode = "auto" if self._approve_all else "ask"
        if mode in ("ask", "on", "prompt"):
            wanted_auto = False
        elif mode in ("auto", "off", "yolo"):
            wanted_auto = True
        elif mode:
            # Rejected: nothing changed, so the boot composition survives it
            # (see `_cmd_usage`). The receipts that DID run stay on `notice`.
            self._system_notice(
                f"unknown approval mode {mode!r} — use ask or auto, "
                "or default ask / default auto to keep it",
                "warning",
            )
            return
        else:
            self._report_approvals(notice)
            return
        if wanted_auto:
            # The prompt ON SCREEN is part of "every tool runs without asking".
            # Setting the mode and leaving it pending printed the loudest notice
            # in the app next to a question still waiting for an answer, with the
            # tool behind it parked on a future nothing was going to settle — the
            # `A` key never had this gap because it answers through the widget.
            self._answer_live_approval_as_allowed()
        else:
            # Also clears the turn-scoped deny latch: this command's whole
            # promise is "tools will prompt again", and a latch left armed
            # would make that statement false for the rest of the run.
            self._allow_approvals_again()
        saved_to = ""
        problem = ""
        if persist:
            # Written BEFORE the flags move so the band cannot advertise `always`
            # on the strength of a write that failed. The live switch happens
            # either way: a read-only config dir is a reason not to promise the
            # next launch anything, not a reason to refuse the session the mode
            # it asked for.
            saved_to, problem = self._save_approvals_default(wanted_auto)
            if not problem:
                self._approvals_default_auto = wanted_auto
        self._set_approve_all(wanted_auto)
        if problem:
            notice(problem, "warning")
        elif persist:
            # Names the file and the key, like `/model default`. "Saved" alone is
            # a claim the user cannot check without relaunching.
            kind = "warning" if wanted_auto else "info"
            notice(
                f"tool approvals: {mode_word(wanted_auto)} — saved to {saved_to}: "
                f"tool_approval_mode {mode_word(wanted_auto)} "
                "(this session and every new one)",
                kind,
            )
        elif wanted_auto:
            notice(
                "tool approvals: auto — every tool runs without asking (this session) — "
                "/approvals default auto saves it for new sessions",
                "warning",
            )
        else:
            notice(
                "tool approvals: ask — write and command tools will prompt again " "(this session)"
            )

    def _report_approvals(self, notice: NoticeFn) -> None:
        """What a bare ``/approvals`` prints: the live mode AND the saved one.

        Both halves, always, because the state is genuinely two-valued and the
        band can only carry one of them. A session switched away from its saved
        default is the case that needs saying out loud — the user is one relaunch
        from a mode they last chose days ago, and nothing else on the screen
        would ever mention it.

        The `warning` tint follows the LIVE mode, not the saved one: the tint
        answers "is the gate disarmed right now".
        """
        live = mode_word(self._approve_all)
        saved = mode_word(self._approvals_default_auto)
        effect = (
            "every tool runs without asking"
            if self._approve_all
            else "write and command tools prompt before running"
        )
        if live == saved:
            notice(
                f"tool approvals: {live} — {effect}; new sessions open the same way",
                "warning" if self._approve_all else "info",
            )
            return
        notice(
            f"tool approvals: {live} (this session) — {effect}; "
            f"new sessions open in {saved} — /approvals default {live} changes that",
            "warning" if self._approve_all else "info",
        )

    def _save_approvals_default(self, auto: bool) -> tuple[str, str]:
        """Write the boot default to config: ``(saved_to, problem)``.

        Same shape and the same failure policy as ``/model default``'s write: a
        config dir that cannot be written is reported in the receipt and does not
        take down a working prompt. Imported at the call site, like that one, so
        the TUI's import cost does not carry the config stack for a command most
        sessions never run.
        """
        try:
            from local_operator.config import ConfigManager
            from local_operator.paths import config_dir

            manager = ConfigManager(config_dir())
            manager.set_config_value("tool_approval_mode", mode_word(auto))
            return _home_relative(str(manager.config_file)), ""
        except Exception as error:  # config write failure
            return "", (
                f"approvals mode changed for this session, but could not save default: {error}"
            )

    def _load_approvals_default(self) -> None:
        """Adopt the saved boot default, at mount, before the session exists.

        The read half of ``/approvals default``. A config value nothing reads
        back is not persistence, it is a file the app writes to itself — so this
        runs before the first turn can ask for approval, and it goes through
        :meth:`_set_approve_all` like every other route, which is what makes the
        band right on the boot frame rather than on the first command.

        Silent on failure. An unreadable config means the safe mode, and a
        session that opened prompting is a session that asks a question the user
        can answer — the failure mode in the other direction is a gate the user
        believes is armed.
        """
        try:
            from local_operator.config import ConfigManager
            from local_operator.paths import config_dir

            saved = ConfigManager(config_dir()).get_config_value("tool_approval_mode", "ask")
        except Exception:
            logger.debug("could not read the saved tool approval mode", exc_info=True)
            return
        self._approvals_default_auto = str(saved).strip().lower() == "auto"
        self._set_approve_all(self._approvals_default_auto)

    def _clear_transcript(self) -> None:
        self._transcript_view().clear_blocks()  # fires the on_clear hook
        # ``ends_empty_state=False``: the receipt reports on the CLEAR, so the
        # session has not started talking and the splash the clear just restored
        # must survive it. Going through ``_append_block`` rather than straight to
        # the transcript is what re-centres the composition around the receipt's
        # own row — the clear hook above resolved it for a region that did not yet
        # contain one, and the difference is the splash settling a row after the
        # frame is up.
        self._append_block(
            NoticeBlock("transcript cleared — history is untouched", "info"),
            ends_empty_state=False,
        )

    def _on_transcript_cleared(self) -> None:
        """TUI-009: /clear and Ctrl+L reset the app's block bookkeeping."""
        # The working line went with the transcript. Whether the TURN did is a
        # different question — /clear does not stop it — so this is also the
        # test for whether one has to be mounted again below.
        was_working = self._working_block is not None
        if self._working_block is not None:
            self._working_block.stop()
            self._working_block = None
        # The prompt's widget is about to be removed with the rest of the
        # transcript, so the turn awaiting it is denied rather than orphaned —
        # but NOT latched: /clear does not stop the turn, and latching here
        # denied every later write/exec tool of the run with no prompt.
        self._settle_live_approval()
        self._exit_hint = None  # its widget went with the transcript
        self._streaming_block = None
        self._tool_cards = {}
        self._composing_cards = {}
        # An empty transcript is the welcome view's whole precondition, so the
        # clear hook is also what brings it back — and with it the boot layout,
        # since `_set_welcome_visible` drives both from this one condition. One
        # mechanism for both directions rather than a second "should the splash
        # show" rule that could disagree with this one. The "transcript cleared"
        # notice appended right after this lands UNDER the splash (see
        # ``_clear_transcript``), which is why the receipt for the action
        # survives alongside the restored splash.
        self._set_welcome_visible(True)
        if was_working:
            # The agent is still working, so something has to still say so —
            # clearing the screen mid-turn otherwise left a live turn looking
            # finished, with the next tool row arriving out of an empty session.
            # ``ends_empty_state=False`` for the same reason the receipt uses it:
            # the splash was just restored and the turn's output has not started
            # arriving again yet.
            self._start_working_block(ends_empty_state=False)

    def _set_welcome_visible(self, visible: bool) -> None:
        """Show or hide the splash AND swap the input layout with it.

        Both ride ONE condition — "the transcript has no content" — because two
        would eventually disagree, and the way that failure looks is a centred
        boot card sitting under a populated transcript. The layouts themselves
        live in the stylesheet (`Screen.boot`); this only flips the flag that
        selects between them, so there is no second layout written in Python to
        keep in step with the first.

        The width class and the vertical reserve are re-resolved here rather than
        being second conditions: they answer "how wide is this terminal" and "how
        many rows are going spare", which are facts about the frame, not about the
        session. On the boot path this call runs at mount, before the terminal's
        size is known, and resolves nothing; the resize that follows it does the
        work, and still does it before the first arrange. On ``/clear`` the size
        is known and this call is the one that re-centres.

        Two things this refuses to do, both measured on a real 396-message
        session swap. It does not re-apply the state already on screen: the
        replay asks for "hidden" once per mounted block, and 296 of those 297
        calls re-resolved the whole boot composition — the card width, the dock
        height, the splash's spare rows — for a splash that had been gone since
        the first one. And it does not SHOW the splash while a swap is in flight
        (`_swapping_session`), because between the clear and the replay "the
        transcript has no content" is true of a screen that is mid-substitution,
        not of an app with nothing to show — that transition is the centred logo
        the user saw between two conversations. The withheld show is remembered
        and `_reload_session` applies it once the answer is settled, so a swap
        onto an EMPTY session — what ``/new`` is — still lands on the splash.
        The HIDE is never deferred: it is never wrong, and a swap OUT of an
        empty session has to retire the splash before the replacement's first
        block mounts underneath it.
        """
        if visible and self._swapping_session:
            self._welcome_pending = True
            return
        self._welcome_pending = False
        if visible == self._welcome_visible:
            return
        self._welcome_visible = visible
        if self._welcome is not None:
            self._welcome.set_visible(visible)
        self.screen.set_class(visible, BOOT_LAYOUT_CLASS)
        self._sync_boot_layout()

    async def on_unmount(self) -> None:
        # Before disposing the session: dispose awaits teardown, and a turn
        # parked on an unanswered approval would never reach it.
        self._deny_queued_approvals()
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
        # A turn already running is STEERED, never re-prompted: `prompt()`
        # rejects a concurrent call outright (the session serializes turns on a
        # lock), so sending one here surfaced "session is already streaming" as
        # an error and threw the user's text away. Steering is the supported
        # mid-turn channel — the engine drains the queue at its next tool/message
        # boundary, which is exactly "send it after the current step finishes".
        if session.is_streaming:
            session.steer(text)
            # "boundary" is engine vocabulary the UI never defines, and this is
            # the line answering "did my text just get thrown away?" — so it says
            # when it will be sent, in the tense it will be sent in, at the
            # `note` weight: above `info`/dim, which is too quiet for an answer
            # the user is waiting for, and below `warning`, which is an alarm this
            # is not. Three warning-tinted rows on one frame for routine receipts
            # is how the loudest ink in the palette stops meaning anything.
            self._append_block(NoticeBlock("queued — sends when this step finishes", "note"))
            # Still worth a title: the steering message can be the first thing
            # in the conversation that actually says what the task is.
            self._maybe_name_conversation(text)
            return
        # A COMPACTION holds the same lock a turn does, and `is_streaming` is
        # False for it (no turn is running), so without this the fall-through
        # called `prompt()`, which raises while the history is being rewritten —
        # the user's row would sit in the ledger as if sent, the agent would
        # never see it, and the text would be gone. That is the identical failure
        # the steer branch above exists to prevent, one lock holder over. Held
        # and sent by `on_compaction_ended`, in the same "it will be sent"
        # language, because a pass is minutes on a long conversation.
        if self._compacting:
            self._prompt_held_for_compaction = text
            self._append_block(NoticeBlock("queued — sends when compaction finishes", "note"))
            self._maybe_name_conversation(text)
            return
        self._start_turn(text)
        # Detached, and deliberately AFTER the turn is dispatched: the title
        # is decoration, and decoration must never sit in front of the user's
        # first reply.
        self._maybe_name_conversation(text)

    def _start_turn(self, text: str) -> None:
        """Run one user prompt as a turn, reporting a failure as a notice.

        Extracted from :meth:`_submit_prompt` so a prompt HELD through a
        compaction is dispatched by exactly the same path when the pass ends —
        the alternative was re-entering `_submit_prompt`, which would write the
        user's row into the ledger a second time.
        """
        session = self._session
        if session is None or self._status is None:
            return
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
        # Delegated spend BEFORE the equality gate, because it moves on ticks
        # where the counts do not: a single child working for two minutes holds
        # `agents` at 1 the whole time while its bill climbs, so gating the
        # harvest on a count change would freeze the parent's total for exactly
        # as long as the child was busy.
        before = self._spend_total()
        self._harvest_subagent_costs()
        total = self._spend_total()
        if (agents, jobs) != self._subagents_shown or total != before:
            self._subagents_shown = (agents, jobs)
            if self._status is not None:
                self._status.update(
                    subagents=agents,
                    jobs=jobs,
                    cost=format_cost(total) if total else None,
                )
        # The band's belt to the event stream's suspenders: elapsed time and
        # job status move with no Subagent*/tool-end event at all, so the 1 Hz
        # poll is what keeps them live. Both panels gate their own repaint on
        # an equality/membership fingerprint, so a no-change tick is nearly
        # free — and never raises, since a status surface may not take the app
        # down (see the panels' own docstrings).
        self._refresh_band()

    def _refresh_band(self) -> None:
        """Repaint the dock band (subagent + todo) from live session state.

        Called on the 1 Hz poll and from the Subagent* event handlers. Safe to
        call before the session is booted: both panels tolerate a ``None``
        session (they read ``getattr``/``getattr(session, 'jobs', None)`` and
        treat a missing manager as empty).
        """
        session = self._session
        if self._subagent_panel is not None:
            self._subagent_panel.sync(session)
        if self._todo_panel is not None:
            self._todo_panel.sync(session)
        # The open subagent page rides the SAME tick, for the same reason: a
        # child's elapsed time and its last tool both move with no event, and
        # a page that only advanced on relayed events sat frozen through every
        # long tool call the child made. `show()` is a no-op when nothing moved.
        self._refresh_subagent_view()
        # BOTH overlay cards ride the same tick. Neither is in the layout, so a
        # dock-band height change does not resize them and Textual emits no
        # resize event for either. Re-measure after the band has repainted;
        # otherwise a todo/subagent appearing under an open tall card lifts the
        # input into it. `sync_layout` is a no-op when the measurement has not
        # moved.
        self.call_after_refresh(self._sync_overlay_layout)

    def _sync_overlay_layout(self, *, force: bool = False) -> None:
        """Re-measure the floating cards against the live screen and dock.

        ``force`` skips each card's no-op fast path. A resize needs it: the
        card's own fingerprint is read one refresh after the event, when the
        dock has been told to re-arrange but has not finished, so the guard
        compares two stale numbers, agrees with itself, and returns.
        """
        for panel in (self._usage_panel(), self._aside_panel()):
            if panel is not None:
                panel.sync_layout(force=force)

    def on_text_area_changed(self, message) -> None:  # noqa: ANN001 - Textual event type
        """The composer's line count moved — re-anchor the aside above it.

        The band tick is not enough for this card and is for the usage card.
        The aside is bottom-anchored one row above the dock AND is the surface
        the user is actively typing into, so the thing that moves the ceiling
        is the composer wrapping — which `_refresh_band` never hears about.
        Measured at 120x40: a four-line question grows the dock from row 35 to
        row 32 while the card's host offset stays put, so the footer carrying
        ``esc close`` paints straight over the first line of the prompt.

        ``call_after_refresh``: the dock has not re-laid out yet at Changed
        time, so measuring now reads the ceiling the card already has.
        """
        if self._aside_is_open():
            self.call_after_refresh(self._sync_overlay_layout)

    def _open_subagent_view(self, job_id: str) -> None:
        """Enter the full-page subagent view for one task job.

        A MODE of this screen rather than a modal over it: the transcript
        region is replaced by the child's transcript while the dock — band,
        status line, composer — stays exactly where it was and is greyed
        (``Screen.subagent`` in the stylesheet). The user is meant to read
        this as the same app looking somewhere else, which a blacked-out
        overlay could not say, and the parent's turn keeps painting into the
        main transcript underneath rather than being suspended behind a screen.

        Opening it a second time RETARGETS the page. The band stays live and
        un-dimmed under the view precisely so a reader can hop between
        subagents without going back up a level to do it.
        """
        if self._subagent_view is not None:
            self._refresh_subagent_view(job_id)
            return
        # Captured before anything is blurred: this is where Esc puts the
        # user back, and it is almost always the composer.
        self._subagent_focus_restore = self.focused
        view = SubagentView(job_id)
        self._subagent_view = view
        self._transcript_view().display = False
        self.screen.mount(view, before=self.query_one("#input-dock"))
        self.screen.add_class(SUBAGENT_LAYOUT_CLASS)
        self._set_composer_read_only(True)
        self._refresh_subagent_view(job_id)

    def _close_subagent_view(self) -> bool:
        """Leave the view and put the conversation back. True if it was open.

        Everything the mode changed is restored here and nowhere else: the
        main transcript was only hidden (never cleared, never re-rendered), so
        the conversation comes back with its blocks, its scroll position and
        any half-typed prompt exactly as they were left.
        """
        view = self._subagent_view
        if view is None:
            return False
        self._subagent_view = None
        view.remove()
        if self._subagent_panel is not None:
            self._subagent_panel.mark_current(None)
        self.screen.remove_class(SUBAGENT_LAYOUT_CLASS)
        self._transcript_view().display = True
        self._set_composer_read_only(False)
        # The band goes back to describing THIS session. Dropping the overlay
        # rather than writing the parent's numbers back is what makes the
        # restoration exact: they never left, they were only shadowed, and
        # every turn that ended while the page was open has been updating
        # them underneath the whole time.
        if self._status is not None:
            self._status.set_subagent(None)
        restore = self._subagent_focus_restore
        self._subagent_focus_restore = None
        try:
            (restore or self._editor()).focus()
        except Exception:
            pass  # the widget that had focus is gone; the mode still closed
        return True

    def on_subagent_view_dismissed(self, message: SubagentViewDismissed) -> None:
        """The page's ``esc`` hint was clicked — same exit as the key itself."""
        message.stop()
        self._close_subagent_view()

    def _refresh_subagent_view(self, job_id: str | None = None) -> None:
        """Repaint the open view from the ledger's CURRENT state.

        Called from the Subagent* handlers and the 1 Hz poll, which is what
        makes a running child's page live: the old modal snapshotted the
        trajectory at open and then sat still while the subagent kept working.
        Reading the manager here rather than holding the job keeps the page
        honest about a job that settled — or was swept — while it was open.
        """
        view = self._subagent_view
        if view is None:
            return
        job_id = job_id or view.job_id
        session = self._session
        manager = getattr(session, "jobs", None) if session is not None else None
        try:
            job = manager.get(job_id) if manager is not None else None
        except Exception:
            job = None
        view.show(
            job_id=job_id,
            label=str(getattr(job, "label", "") or job_id),
            # A swept job reads as `gone`, not as `running`: the row it was
            # opened from has been evicted from the ledger, and claiming the
            # child is still working would be the one wrong answer.
            status=str(getattr(job, "status", "") or "gone") if job is not None else "gone",
            queued=bool(getattr(job, "queued", False)),
            elapsed=job_elapsed(job) if job is not None else "0s",
            # The instruction the parent delegated. Recorded on the job at
            # REGISTRATION (`AsyncJob.prompt`), which is the only place it
            # survives: `Session.prompt` feeds the user message straight into
            # the turn pipeline without emitting an event, so no amount of
            # trajectory carries it. `None` — a job type that records none —
            # is distinct from `""`, and neither prints a row.
            prompt=str(getattr(job, "prompt", None) or ""),
            events=getattr(job, "trajectory", None) or [],
            progress=str((getattr(job, "latest_details", None) or {}).get("progress") or ""),
        )
        if self._subagent_panel is not None:
            self._subagent_panel.mark_current(job_id)
        self._point_band_at(job)

    def _point_band_at(self, job: Any) -> None:
        """Make the status band describe the CHILD while its page is open.

        The band sits four rows under the page and was still reporting the
        parent's model, context and spend over a frame whose whole subject is
        another session — actively misleading, because a child launched with a
        ``model_spec`` override is on a DIFFERENT model than the band claimed.

        Facts the child has not reported are OMITTED rather than filled in
        from the parent: an empty overlay field drops its segment, and a
        missing number is a smaller lie than somebody else's number. The model
        is the exception and cannot be omitted — a child that recorded none of
        its own inherited the parent's, so that IS the child's model.
        """
        if self._status is None:
            return
        # Read from the PANEL rather than derived here. This runs on the 1 Hz
        # poll, and deriving would price the child on the event loop — the
        # resolve that costs 13 s for a model the shipped registry does not
        # describe, which is exactly the child a `model_spec` override
        # produces. The panel already takes that reading off-thread; two
        # conventions for one call is one convention too many.
        panel = self._subagent_panel
        job_id = str(getattr(job, "id", "") or "")
        stats = panel.stats_for(job_id) if panel is not None and job_id else JobStats()
        if not stats.model_label:
            # No reading has landed yet. The model is the one segment the band
            # never drops, and naming it costs nothing: both halves are plain
            # attribute reads — the child's own label if the engine recorded
            # one, else the parent's, which is what an unrecorded child is
            # running. The NUMBERS stay absent until the real reading arrives,
            # because those are what cost 13 s to derive.
            stats = JobStats(
                model_label=str(
                    getattr(job, "model_label", "")
                    or getattr(self._session, "model_label", "")
                    or ""
                )
            )
        # `job_cost` answers None both for "no price for this model" and for
        # "no usage recorded yet". The band distinguishes them: a child that
        # has billed tokens gets the `$—` this app already uses for
        # unpriceable spend (see `on_turn_ended`), and one that has reported
        # nothing at all gets no segment, because it has not spent anything we
        # know of. `billed` carries that distinction off the same reading, so
        # the job is not probed a second time on a repeating timer.
        if stats.cost is not None:
            cost = format_cost(stats.cost)
        elif stats.billed:
            cost = "$—"
        else:
            cost = ""
        self._status.set_subagent(
            SubagentBand(
                model_label=stats.model_label,
                label=strip_control_sequences(str(getattr(job, "label", "") or "")),
                context_tokens=stats.context_tokens,
                context_window=stats.context_window,
                cost=cost,
                # No duration. The page's title carries the child's age three
                # rows above this, and the band's right group was otherwise
                # reproducing the row beneath it cell for cell. The band keeps
                # the two NUMBERS instead, which is what the dock cannot
                # always spare the cells for.
                duration=None,
            )
        )

    def _set_composer_read_only(self, read_only: bool) -> None:
        """Make the composer visibly inert, or give it back.

        Read-only rather than hidden or disabled. Hidden would break the
        promise that this is the same app in another mode; Textual's
        ``disabled`` also removes it from the tab order and paints it with a
        style this app does not otherwise use. ``read_only`` keeps the widget,
        keeps any half-typed prompt, and refuses every edit — and the
        stylesheet's ``Screen.subagent`` rules are what make that refusal
        visible before the user tries it.
        """
        try:
            editor = self._editor()
        except Exception:
            return  # a stripped harness with no composer
        editor.read_only = read_only
        # The placeholder is the composer's own voice, and while the mode is
        # on it was still printing `Message Local Operator…` — an imperative
        # invitation from a field that refuses every key, greyed to the point
        # of being hard to read but present enough to invite. State the
        # constraint where the hands are.
        editor.placeholder = READ_ONLY_PLACEHOLDER if read_only else DEFAULT_PLACEHOLDER
        # Not focusable while inert: a caret in a field that refuses every key
        # is the most misleading thing this mode could paint, and ↑↓ have to
        # reach the transcript that the hint says they scroll. The caret is
        # what this line actually removes — it follows `has_focus`, so the
        # blur below is what takes it off the frame.
        editor.can_focus = not read_only
        if read_only and editor.has_focus:
            # Dropping `can_focus` does not move focus that is already there,
            # and a still-focused composer keeps BOTH affordances lit — the
            # caret and, through `_sync_composer_focus`, the bright chevron —
            # on a surface whose entire argument is that the dock is not where
            # you are. The blur is what turns them off; the stylesheet's
            # `Screen.subagent` chevron rule then states the same thing a
            # second time, because it is a property of the MODE.
            self.screen.set_focus(None)

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
    def _transcript_view(self) -> TranscriptView:
        """The MAIN conversation's transcript.

        Held from :meth:`compose` rather than queried, because the full-page
        subagent view mounts a SECOND ``TranscriptView`` — the child's — into
        the same screen, and it does so while the parent's turn may still be
        streaming. A `query_one` by type would start raising ``TooManyMatches``
        at exactly that moment, i.e. it would turn an observability surface
        into a way of crashing the conversation it is observing. The `#id`
        re-lookup is the recovery path for a handle that no longer belongs to
        the tree — it is not reachable today (the main transcript is never
        removed or reparented) and is kept only so that a future screen
        rebuild degrades to a query instead of appending into a detached
        widget.
        """
        view = self._transcript
        if view is None or view.parent is None:
            view = self._transcript = self.query_one("#transcript", TranscriptView)
        return view

    def _append_block(
        self, block, *, ends_empty_state: bool = True, pin_tail: bool = False
    ) -> None:
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

        ``pin_tail=True`` also holds the block at the BOTTOM as later blocks
        arrive — see :meth:`TranscriptView.pin_tail`. Only the working line uses
        it, and only one block can be pinned at a time.

        A block that lands UNDER the splash takes rows out of the same region the
        composition is measured against, so the reserve is recomputed here rather
        than left centred for a region that no longer exists.
        """
        if ends_empty_state:
            self._set_welcome_visible(False)
        transcript = self._transcript_view()
        if pin_tail:
            transcript.pin_tail(block)
        else:
            transcript.append_block(block)
        if not ends_empty_state:
            self._sync_boot_layout()

    # -- slash commands -----------------------------------------------------
    def _notice(self, body: str, kind: NoticeKind = "info") -> None:
        """Append a notice block.

        A METHOD rather than the local closure it used to be, because the picker's
        worker and its choose-callback both need to report and neither runs inside
        a dispatch. One implementation means every path renders notices the same.
        """
        self._append_block(NoticeBlock(body, kind))

    def _system_notice(self, body: str, kind: NoticeKind = "info") -> None:
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

    def _echo_user_command(self, text: str) -> None:
        """Write a slash command into the ledger as the user's own row, IF its
        registry entry permits one.

        Called BY THE HANDLER, once its effect has landed — not by
        :meth:`on_editor_submitted`, which cannot know whether it did. Written
        before dispatch, the row claimed the model had been given words for
        ``/goal``'s read-only form, for ``/goal clear``, and for a set rejected
        because the session had not started yet; that last one also retired the
        boot splash for a command that changed nothing, defeating the
        ``_system_notice`` its own handler uses to prevent exactly that.

        The registry still decides WHETHER (``SlashCommand.echo``), which is
        what this reads: flipping the flag turns the row off wherever a handler
        asks for one, so the policy stays next to the command rather than
        scattered across the handlers that implement it.
        """
        entry = slash_command_for(text)
        if entry is None or not entry.echo:
            return
        self._append_block(UserBlock(text))

    def _editor(self) -> Editor:
        """The input editor. Queried rather than held: Textual owns the widget."""
        return self.query_one(Editor)

    def _run_slash_command(self, text: str) -> None:
        """Dispatch a typed slash command (with arguments) to its handler.

        The typed word is resolved to its registry PRIMARY name first, so the
        branches below only ever have to know one spelling. They used to match
        raw literals, and the two advertised aliases the registry carries that
        no branch repeated — ``/models`` and ``/recall`` — reached the ``else``:
        ``/help`` printed them, the picker completed them, and running one
        answered "unknown command: /recall". ``/quit`` only worked because its
        branch happened to list it by hand, which is the same bug with a patch
        on it.

        Resolving through :func:`slash_command_for` also means the echo lookup
        in :meth:`on_editor_submitted` and this dispatch cannot disagree about
        what a word means — one resolver, so an alias cannot echo as one command
        and run as another.
        """
        parts = text.split(maxsplit=1)
        entry = slash_command_for(text)
        command = f"/{entry.name}" if entry is not None else parts[0].lower()
        arg = parts[1].strip() if len(parts) > 1 else ""
        notice = self._notice

        if command == "/exit":
            self.exit()
        elif command == "/help":
            self._append_block(self._help_block())
        elif command == "/clear":
            self._clear_transcript()
        elif command == "/reload":
            self._cmd_reload(notice)
        elif command == "/new":
            self._cmd_new(notice)
        elif command == "/resume":
            self._cmd_resume(arg, notice)
        elif command == "/model":
            self._cmd_model(arg, notice)
        elif command == "/effort":
            self._cmd_effort(arg, notice)
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
        elif command == "/btw":
            # The RAW line, not just the argument: `/btw` is the one command
            # that has to retract its own entry from the prompt history, and
            # what was recorded is what the user typed.
            self._cmd_btw(arg, text)
        elif command == "/compact":
            self._cmd_compact()
        elif command == "/approvals":
            self._cmd_approvals(arg, notice)
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
            # ``parts[0]``, not the lowered ``command``: with the echo gone this
            # line is the ONLY place the mistyped word appears, so it has to
            # show what was actually typed — telling someone ``/USGE`` is
            # unknown by printing ``/usge`` invites them to hunt for a second
            # typo they did not make.
            #
            # ``_system_notice``: a command that did not run changed nothing, so
            # the conversation has not started and the boot composition must
            # survive a typo — the same rule ``_cmd_model`` applies to a
            # rejected selector. It also keeps the splash on screen behind the
            # warning, so a first-action typo never leaves a blank frame.
            self._system_notice(f"unknown command: {parts[0]} — try /help", "warning")

    # -- context ------------------------------------------------------------
    def _cmd_compact(self) -> None:
        """``/compact`` — run a compaction pass now.

        It used to print "compaction runs automatically when the context fills
        up" and change nothing, which is a help topic, not a command: the owner
        reported it as a command that "just echoes a description". Compaction on
        demand is a real need — the moment before handing the agent a big task
        is exactly when a user wants the context cleared, and waiting for the
        automatic threshold means paying full price for every turn until then.

        The pass itself is ``SessionProtocol.compact_now``, which is the SAME
        pass the automatic gate runs (same strategy resolution, same events), so
        the two cannot drift. This method only decides whether there is a
        session to ask, and hands the awaiting to a worker: a compaction is one
        summarization request over the whole conversation, minutes on a long
        one, and blocking the dispatch would freeze the UI for all of it.
        """
        session = self._session
        if session is None:
            # ``_system_notice``: nothing ran, so the conversation has not
            # started and the boot composition must survive it — the same rule
            # every rejecting branch in `_run_slash_command` follows.
            self._system_notice("no session yet — there is no context to compact", "warning")
            return
        # NOT exclusive. Textual's exclusivity CANCELS the running worker, and
        # cancelling a pass mid-summarization would leave the session having
        # announced `compacting context…` with no end event to retire it. The
        # session refuses a second concurrent pass itself (``already_running``),
        # which is one authority for the whole question instead of two.
        self.run_worker(self._compact_worker(session), thread=False, group="compact")

    async def _compact_worker(self, session: SessionProtocol) -> None:
        """Await one manual compaction and report what it did.

        A pass that RUNS narrates itself through the ``compaction_start`` /
        ``compaction_end`` events the automatic path already emits — one
        vocabulary for both triggers, and the receipt with the token pair is
        rendered from the end event in :meth:`on_compaction_ended`. So this
        worker only reports what those events never cover: a refusal, and a
        crash. Both go through ``_system_notice`` because nothing changed.

        ``failed`` is the one ``ran=False`` that is not a refusal — the pass
        started and blew up — so it is styled ``error`` like every other
        failure, not ``warning`` like the states a user can act their way out
        of. It used to land as a ``warning`` UNDER the end event's bare
        ``compaction failed`` error: two notices for one event, the empty one
        louder than the one carrying the reason. :meth:`on_compaction_ended`
        now leaves the manual case to this line.
        """
        try:
            outcome = await session.compact_now()
        except Exception as exc:  # noqa: BLE001 — a failed pass must not kill the app
            logger.debug("manual compaction failed", exc_info=True)
            self._system_notice(f"compaction failed: {exc}", "error")
            return
        if not outcome.ran:
            kind: NoticeKind = "error" if outcome.reason == "failed" else "warning"
            self._system_notice(outcome.detail or "compaction did not run", kind)
        # A pass that RAN needs nothing further here: the band is settled from
        # the END EVENT, in `on_compaction_ended`, so the automatic trigger gets
        # the same treatment instead of leaving its own reading stale.

    def _settle_context_reading(self, tokens_before: int, tokens_after: int) -> None:
        """Move the band's context reading to what a compaction left behind.

        The band reports the size of the next REQUEST — system blocks, tool
        schemas and history — while a compaction's two figures measure the
        HISTORY alone. The pass changes only the history, so the saving
        transfers exactly, and taking it off is the only way the band agrees
        with the receipt that just said the context shrank. Left alone, the
        reading stays at the pre-compaction size until the next turn's usage
        arrives, which is the "it did nothing" frame ``/compact`` exists to stop
        showing — and an automatic pass shows it for the same reason.

        Floored at the post-pass history, which a live run made necessary: on a
        200k-window vision session the band read 12.3k after the pass and on the
        text model it read ZERO, because compaction's local ruler runs 7-17%
        above a provider's own count (see ``Session.measure_preloaded_context``)
        and the estimated saving can exceed the provider figure it is subtracted
        from. A band claiming an empty context right after a compaction is a
        worse lie than a stale one, and the kept history is a floor the next
        request cannot go under.

        Demoted to an ESTIMATE even when the figure it was subtracted from was
        the provider's own: the arithmetic came from compaction's tokenizer, so
        the result is no longer a number the wire reported.
        """
        saved = tokens_before - tokens_after
        if self._status is None or saved <= 0:
            return
        self._status.update(
            context_tokens=max(tokens_after, self._status.context_tokens - saved),
            context_is_estimate=True,
            context_window=_context_window(self._session),
        )

    # -- model --------------------------------------------------------------
    def _cmd_model(self, arg: str, notice: NoticeFn) -> None:
        """``/model`` — open the picker; ``/model provider/id`` — switch directly.

        A bare ``/model`` OPENS THE LIST rather than printing the current label.
        Printing it was a dead end: the answer to "which model am I on" is already
        on the status band, and the question a user actually has at that moment is
        "which models can I switch to", which they had no way to ask. The label is
        still reported, as the picker's current-row marker.

        Every route out of here names ``/model default``, because a switch is
        SESSION-scoped and does not look it. The user who picks a model has just
        answered "which model do I want", the app confirms the switch, and nothing
        on screen says the next launch comes back on the old one — so the command
        that fixes that was reachable only by already knowing it existed.
        """
        session = self._session
        if not arg:
            # ``_system_notice``, NOT ``notice``: opening the picker is the user
            # configuring the app, not starting a conversation, and a plain
            # notice ends the empty state — which collapses the boot
            # composition and makes the centred prompt unreachable. That is the
            # same failure a broken MCP server caused before ``_system_notice``
            # existed; routing a hint about the app's own settings through the
            # conversation path reintroduced it.
            self._system_notice(self._persist_hint_notice())
            self._open_model_picker()
            return
        # ``/model default [<provider>/<id>]`` PERSISTS the choice as the boot
        # default (config ``hosting`` + ``model_name``), so later launches
        # open on it. Distinct from ``/model <p>/<id>``, which only switches
        # the running session: the user phrase "make this the default" has no
        # other home, and the picker's current-row marker already covers "which
        # am I on". Writing config here keeps the one-word habit — ``/model
        # default openrouter/deepseek/deepseek-chat`` — and the live switch
        # stays available on the same command.
        lowered = arg.lower()
        persist_default = lowered == "default" or lowered.startswith("default ")
        target = arg[len("default ") :].strip() if persist_default else arg
        if session is None or not hasattr(session, "set_model"):
            # A rejected command changed nothing, so the conversation has not
            # started: `_system_notice` keeps the boot composition intact where
            # `notice` would collapse it for a typo.
            self._system_notice("session is still starting…", "warning")
            return
        if persist_default and not target:
            # Bare ``/model default`` means "keep the one I am on". Demanding the
            # selector back was the gap behind "how do I even set a default": the
            # sentence a user has right after switching is "make THIS the
            # default", and answering it by retyping
            # `openrouter/deepseek/deepseek-chat` is a transcription exercise the
            # app can do for itself off the session's own label.
            target = session.model_label
            if not target:
                self._system_notice("usage: /model default <provider>/<model-id>", "warning")
                return
        provider, sep, model_id = target.partition("/")
        if not sep or not model_id:
            self._system_notice(
                "usage: /model <provider>/<model-id> (e.g. openrouter/deepseek/deepseek-chat)",
                "warning",
            )
            return
        provider = provider.lower()  # build_model_spec is case-insensitive
        if self._providers is None:
            self._system_notice(
                "provider controller unavailable — cannot infer model spec", "warning"
            )
            return
        # Validate the provider BEFORE switching. resolve_model does not raise
        # on an unknown provider — it returns a spec with base_url=None — so a
        # typo would silently reconfigure the session and only fail on the next
        # turn, reading as a network/auth error instead of a typo.
        if self._providers.provider(provider) is None:
            self._system_notice(f"unknown provider: {provider} — see /provider", "warning")
            return
        try:
            spec = self._providers.resolve_model(provider, model_id)
        except Exception as error:  # unresolvable hosting/model pair
            self._system_notice(f"cannot resolve {provider}: {error}", "error")
            return
        old_label = session.model_label
        # The chosen effort rides along when the new model accepts it: a user
        # who dropped to `low` for cost did not mean "until I switch models".
        session.set_model(self._spec_with_chosen_effort(spec))
        # A different model may well have a dial, so the per-model refusal latch
        # goes with the old one.
        self._effort_refusal_shown = None
        persist_result: str | None = None
        saved_to = ""
        if persist_default:
            # Persist as the boot default. Written independently of the live
            # switch above so the two stay composable (``/model default p/m``
            # both switches AND persists; a future flag could persist-only).
            # Failure to write is reported but not fatal — the session already
            # switched, and a read-only config dir should not take down a
            # working prompt. The status line is repainted regardless (the
            # ``return`` below used to skip it, leaving the band's model label
            # stale on the very config the user just asked to make permanent).
            try:
                from local_operator.config import ConfigManager
                from local_operator.paths import config_dir

                manager = ConfigManager(config_dir())
                manager.set_config_value("hosting", provider)
                manager.set_config_value("model_name", model_id)
                saved_to = _home_relative(str(manager.config_file))
            except Exception as error:  # config write failure
                persist_result = f"model switched, but could not save default: {error}"
        if self._status is not None:
            # The window and the effort belong to the SPEC, not the session:
            # a switch that repainted only the label would leave the context
            # percentage measured against the previous model's window.
            self._status.update(
                model_label=session.model_label,
                model_name=_model_name(session),
                effort=_effort_label(session),
                context_window=_context_window(session),
            )
        suffix, warning = self._model_access_note(provider)
        if persist_result is not None:
            notice(persist_result, "warning")
        elif persist_default:
            # Names both halves, the file and the keys. "saved" alone is a claim
            # the user cannot check without quitting and relaunching, and the
            # PROVIDER is the half that rides along silently — it is written from
            # the selector's left side, never typed as its own setting.
            notice(
                f"boot default saved to {saved_to}: hosting {provider}, "
                f"model_name {model_id} (used from the next launch){suffix}"
            )
        else:
            # "(next turn)" alone read as permanent — the complaint behind this
            # wording. The scope and the one command that widens it belong on the
            # line that announces the switch, not in documentation the user would
            # have to already suspect exists.
            #
            # TWO clauses and no more. This carried four separators — a
            # parenthetical with a comma in it, the access note's ` · `, then a
            # ` — ` onto a sentence of its own — and wrapped at 80 columns into a
            # run-on. "(this session)" is the half that answers "for how long";
            # "from the next turn" answered "starting when", which nothing had
            # asked and which the very next receipt demonstrates anyway.
            notice(
                f"model: {old_label} → {session.model_label} "
                f"(this session){suffix} — {PERSIST_HINT}"
            )
        if warning:
            notice(warning, "warning")

    def _persist_hint_notice(self) -> str:
        """The line a bare ``/model`` prints above the list.

        It says the one thing the picker cannot: that a pick is session-scoped
        until ``/model default`` writes it, and that "the default" is a provider
        AND a model, not two separate settings to hunt for. The current label is
        repeated here even though the status band carries it, because it is the
        subject of the sentence — "make THIS the default" needs a this, and with
        no session yet there is no this, so the label is all that varies.
        """
        session = self._session
        label = session.model_label if session is not None else ""
        if not label:
            return PERSIST_HINT
        return f"model: {label} — {PERSIST_HINT}"

    # -- reasoning effort ---------------------------------------------------
    def _effort_levels(self) -> tuple[str, ...]:
        """The ladder the ACTIVE model accepts, ascending; empty when it has none.

        Read off the spec rather than recomputed from the model id: the spec is
        where ``build_model_spec`` already resolved it, and a second derivation
        here is how the band and the wire end up disagreeing about which levels
        exist.
        """
        spec = _model_spec(self._session)
        return tuple(getattr(spec, "reasoning_efforts", ()) or ())

    def _apply_effort(self, level: str | None, *, remember: bool = True) -> bool:
        """Put ``level`` on the session's spec and repaint the band.

        Through ``set_model`` because the spec IS the request: the loop rereads
        ``config.model`` every turn, so the level takes effect on the next one
        and every wire client reads it from the same field. Remembered on the
        app as well, because a session can be REPLACED under a running app —
        ``/new``, ``/reload`` and ``/resume`` all rebuild one — and a setting
        that silently reverted on a reload would be a band asserting a level
        that is not in force, which is the one thing this segment must never do.

        ``remember=False`` puts a level in force WITHOUT recording it as a
        choice, which is what ``/effort auto`` needs: restoring the model's own
        default is the user withdrawing a preference, and a withdrawal that got
        filed as a preference would then ride onto the next model they switched
        to.

        Returns whether the level actually LANDED. Callers print the receipt off
        that rather than off having asked: a host whose session cannot take a
        spec (the embedders and pilot fakes this module degrades for) would
        otherwise get a line announcing a level nothing is running.
        """
        session = self._session
        spec = _model_spec(session)
        if session is None or spec is None or not hasattr(session, "set_model"):
            return False
        self._effort_choice = level if remember else None
        session.set_model(spec.model_copy(update={"reasoning_effort": level}))
        # READ BACK rather than trusting the write. A host is free to accept a
        # spec and keep its own, and the whole value of this return is that a
        # receipt is never printed for a level the session is not carrying —
        # which `hasattr(set_model)` alone does not establish.
        landed = getattr(_model_spec(session), "reasoning_effort", None) == level
        if self._status is not None:
            self._status.update(effort=_effort_label(session))
        return landed

    def _spec_with_chosen_effort(self, spec: Any) -> Any:
        """``spec`` carrying the level the user picked, when the model takes it.

        A chosen effort belongs to the USER, not to the model they happened to
        be on when they chose it, so it rides across a ``/model`` switch and a
        session rebuild. It cannot ride onto a model with a different ladder,
        though: ``xhigh`` on a model that stops at ``high`` is a 400 on the next
        turn. When the new model cannot take it the choice is FORGOTTEN rather
        than kept in the wings — a preference that vanishes and then reappears
        two switches later is spookier than one the user re-picks.
        """
        choice = self._effort_choice
        if choice is None:
            return spec
        if choice not in tuple(getattr(spec, "reasoning_efforts", ()) or ()):
            self._effort_choice = None
            return spec
        return spec.model_copy(update={"reasoning_effort": choice})

    #: How the listing marks the level in force. The app already owns a
    #: current-row glyph — the model picker's ``_CURRENT_MARK`` — and brackets
    #: were three readings at once here: "current" (intended), "optional" (every
    #: CLI convention), and "the token to type" (``/accounts`` wraps ids in them).
    #: Bound to the level with no space so the mark cannot read as its own word.
    EFFORT_MARK = "●"

    def _effort_listing(self, levels: tuple[str, ...], current: str | None) -> str:
        """The one-line answer a bare ``/effort`` prints: the ladder, current marked.

        ``auto`` leads the ladder as a rung of its own, because it IS one: it is
        the state a model with no level set is in, it is the argument that
        returns to that state, and rendering it here is what lets the marker
        answer "where am I" for a model that boots unset — every OpenAI
        reasoning model, by design. Without it, the one family that starts
        unset got a listing with nothing marked at all.

        The scope clause is the counterpart of ``PERSIST_HINT`` on a bare
        ``/model``: that command's listing volunteers that a pick persists, so
        this one has to volunteer that a level does not.
        """
        rungs = ("auto", *levels)
        marked = current or "auto"
        rendered = " ".join(
            f"{self.EFFORT_MARK}{name}" if name == marked else name for name in rungs
        )
        return f"reasoning effort: {rendered} — shift+tab cycles, this session only"

    def _cmd_effort(self, arg: str, notice: NoticeFn) -> None:
        """``/effort`` — list this model's levels; ``/effort <level>`` — set one.

        Bare ``/effort`` LISTS rather than opening a picker, which is the
        opposite of ``/model``'s choice and for the reason that made ``/model``
        open one: the question behind a bare command is "what can I pick", and
        for the model that is a searchable catalogue of hundreds while here it
        is at most five words that fit on the row the answer is printed on. A
        widget would also be a second way to do what ``shift+tab`` already does
        in one keystroke, so the listing names that key instead of competing
        with it.

        The level is SESSION-scoped and deliberately not persistable, which is
        where this parts company with ``PERSIST_HINT``. The model is a standing
        preference — you want the same one next launch — while effort is a
        per-task dial: raised for a hard refactor, dropped for chat. Freezing
        one task's dial into every future session is the failure mode, so there
        is no ``/effort default`` to write one, and the receipt says how long
        the choice lasts instead of pointing at a command that would extend it.
        """
        session = self._session
        spec = _model_spec(session)
        if session is None or spec is None:
            self._system_notice("session is still starting…", "warning")
            return
        levels = self._effort_levels()
        label = getattr(session, "model_label", "") or "this model"
        if not levels:
            # Says so rather than accepting a level it would silently drop: the
            # request carries no effort key for this model, so any level the app
            # took here would be a claim the wire does not back.
            self._system_notice(_effort_unavailable(label))
            return
        current = getattr(spec, "reasoning_effort", None)
        wanted = arg.strip().lower()
        if not wanted:
            self._system_notice(self._effort_listing(levels, current))
            return
        if wanted == "auto":
            # The way BACK, the same role `/approvals ask` plays: cycling can
            # never reach "unset", so without this an explicit level would be
            # permanent for the session once any level had been chosen.
            #
            # Restores the MODEL's documented default rather than blanking the
            # field, so the band goes on naming the level actually in force —
            # `high` on Anthropic — rather than only the fact that it reasons.
            restored = default_effort(getattr(spec, "model_id", "") or "")
            if not self._apply_effort(restored, remember=False):
                self._system_notice("session cannot change model settings", "warning")
                return
            # `notice` and the arrow shape, like every other branch that CHANGED
            # something. It read as a rejection before — a `_system_notice` with
            # no old value — which left the one command that withdraws a setting
            # as the only one whose receipt did not show the withdrawal.
            destination = restored or "the provider's default"
            scope = "(the model's own default)" if restored else "(nothing sent)"
            notice(f"reasoning effort: {current or 'auto'} → {destination} {scope}")
            return
        if wanted not in levels:
            # The model is not named: the band names it one row below, and the
            # levels listed are its by construction. Same space-separated ladder
            # as the listing, so the two can be pattern-matched, and `auto` is
            # advertised here because a rejection is where a user is looking for
            # what they may type.
            self._system_notice(
                f"unknown effort level {wanted!r} — this model takes "
                f"{' '.join(levels)}, or auto",
                "warning",
            )
            return
        if wanted == current:
            self._system_notice(
                f"reasoning effort: already {wanted} — /effort auto restores the model's default"
            )
            return
        if not self._apply_effort(wanted):
            # The spec never moved, so no receipt: a line claiming a level the
            # request will not carry is the failure this whole segment exists
            # to prevent.
            self._system_notice("session cannot change model settings", "warning")
            return
        # `notice`, not `_system_notice`: this one CHANGED something, so it is a
        # receipt for an action rather than an answer about the app's settings.
        notice(f"reasoning effort: {current or 'provider default'} → {wanted} (this session)")

    def action_cycle_effort(self) -> None:
        """``shift+tab`` — step one level up this model's ladder, wrapping.

        Prefers the BAND as the receipt and stays out of the transcript: the
        level sitting next to the model label is the whole feature, and a key a
        user may press four times to get round a five-rung ladder would
        otherwise leave four rows in a reading record meant to hold the
        conversation. Same test the echo policy applies to slash commands.

        But "the band is the receipt" is only true when the band still HAS the
        segment. Measured on the shipped drop ladder with a cost figure on the
        row, the effort rung is shed below 118 columns — so at 80 and 100, the
        two commonest terminal widths, pressing the key changed a billable
        setting and moved nothing on screen at all. When the band could not
        speak, the transcript does, in the same words ``/effort`` uses.

        Two states refuse rather than cycle, and both say why, because a key
        that silently does nothing reads as broken: no session yet, and a model
        with no ladder. The second is latched per model — the unbounded version
        answered four presses with four warning rows, which is the noise this
        docstring's first paragraph exists to avoid.
        """
        if self._session is None:
            # Not "this model has no levels": there is no model yet, and naming
            # the wrong reason is worse than naming none.
            self._system_notice("session is still starting…", "warning")
            return
        if self.screen is not self.screen_stack[0] or self._overlay_list_is_open():
            # A modal (`/resume`'s picker) covers the band, and an open
            # completion list is a surface where shift+tab means "previous
            # suggestion" everywhere else. Cycling under either changes a
            # billable setting with the receipt hidden, and pins a choice onto a
            # model the user is in the middle of choosing.
            return
        levels = self._effort_levels()
        label = getattr(self._session, "model_label", "") or "this model"
        if not levels:
            if self._effort_refusal_shown != label:
                self._effort_refusal_shown = label
                self._system_notice(_effort_unavailable(label), "warning")
            return
        current = getattr(_model_spec(self._session), "reasoning_effort", None)
        chosen = next_effort(levels, current)
        if not self._apply_effort(chosen):
            return
        if self._status is not None and not self._status.is_showing("effort"):
            self._notice(f"reasoning effort: {current or 'auto'} → {chosen} (this session)")

    def _overlay_list_is_open(self) -> bool:
        """Whether the composer is showing a completion or model list."""
        try:
            editor = self.query_one(Editor)
        except Exception:  # no composer on reduced hosts
            return False
        return editor._picker.is_open() or editor._model_picker.is_open()

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
            status=_status_line(
                note, "checking providers…" if self._providers else "", PERSIST_HINT
            ),
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
                status=_status_line(note, f"live model list unavailable: {error}", PERSIST_HINT),
            )
            return
        rows, note = self._catalogue_rows(entries)
        self._editor().model_picker.set_rows(
            rows,
            current=self._current_selector(),
            status=_status_line(note, _catalogue_status(statuses), PERSIST_HINT),
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
            # A rejected command changed nothing, so the conversation has not
            # started: `_system_notice` keeps the boot composition intact where
            # `notice` would collapse it for a typo.
            self._system_notice("session is still starting…", "warning")
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
        # The ONE user row a slash command writes, and it is written here
        # because this is the line that knows the words were taken: from the
        # next turn on they ride the system prompt's volatile tail, so they are
        # transcript subject matter and they belong to the user who typed them.
        #
        # ``stored``, not ``arg``: ``set_goal`` trims and length-caps, and the
        # row's whole claim is "this is what the model is being told".
        #
        # The notice below therefore reports STATUS only. It used to repeat the
        # goal text, which with the row above it would be the same duplication
        # this change removed everywhere else.
        self._echo_user_command(f"/goal {stored}")
        from local_operator.session.goal import MAX_GOAL_CHARS

        if len(stored) == MAX_GOAL_CHARS and len(arg.strip()) > MAX_GOAL_CHARS:
            # Gated on the CAP, not on "the result came back shorter". `set_goal`
            # is reached through a duck-typed `hasattr`, so an implementation
            # that normalised further — collapsing newlines in a pasted goal —
            # would make a length comparison announce a cap that never applied.
            #
            # Said out loud at all because the row above now carries the text and
            # is attributed to the USER: a silent cut leaves the ledger claiming
            # they typed something ending mid-word. The receipt this replaced
            # printed the stored text and was equally silent about the cut, but
            # it was at least system-attributed while doing it.
            notice(
                f"goal set — shortened to the {MAX_GOAL_CHARS}-character cap, "
                "applies from the next turn",
                "warning",
            )
            return
        notice("goal set — applies from the next turn")

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
            # A rejected command changed nothing, so the conversation has not
            # started: `_system_notice` keeps the boot composition intact where
            # `notice` would collapse it for a typo.
            self._system_notice("session is still starting…", "warning")
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
                self._system_notice(f"usage: /loop [n] (1..{MAX_LOOP_ITERATIONS})", "warning")
                return
        if iterations < 1 or iterations > MAX_LOOP_ITERATIONS:
            self._system_notice(
                f"iterations must be between 1 and {MAX_LOOP_ITERATIONS}", "warning"
            )
            return
        self._loop_cancelled = False
        notice(f"looping toward the goal ({iterations} iteration(s)) — /loop stop to cancel")
        self.run_worker(self._loop_worker(iterations), thread=False, group="loop")

    async def _loop_worker(self, iterations: int) -> None:
        """Run up to ``iterations`` goal-advancing turns, sequentially."""

        def notice(body: str, kind: NoticeKind = "info") -> None:
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
                # Re-checked every iteration, not captured once: `/reload`
                # replaces `self._session` underneath a running loop, and the
                # captured reference went on driving the DISPOSED session — the
                # old one took the remaining prompts, the new one took none, and
                # `_loop_running` stayed True so `/loop` answered "already
                # running". Same guard `_name_conversation_worker` already makes.
                if session is not self._session:
                    break
                # A NOTICE, not a `UserBlock`: this line is app-authored chrome
                # (the comment on `SLASH_COMMANDS["loop"]` says as much of the
                # prompt it announces), and a user block now paints the gutter
                # rule that means "the human wrote this". `/loop 10` would stack
                # ten of them, so the loudest new mark in the transcript would
                # belong to the one turn the user did not type.
                notice(f"loop {index + 1}/{iterations}", "note")
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
        elif session is not self._session:
            # Reported as stopped, not finished: the remaining iterations never
            # ran, and "finished" on a reload reads as the loop having completed.
            notice(f"loop stopped by reload after {completed} iteration(s)", "warning")
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
            block = RichBlock(_tree_listing(items, "providers")) if items else None
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
            block = RichBlock(_tree_listing(items, "stored credentials")) if items else None
            if block is not None:
                self._append_block(block)
        except Exception as error:
            notice(f"accounts failed: {error}", "error")

    def _clock_ms(self) -> float:
        import time

        return time.time() * 1000

    def _cmd_usage(self, arg: str, notice: NoticeFn) -> None:
        """``/usage [provider]`` — fetch live quota for a provider (or all)."""
        # ``_system_notice`` for every branch that REJECTS: a command that did
        # not run changed nothing, so the conversation has not started and the
        # boot composition must survive it — the rule `_cmd_model` already
        # applied to a bad selector, and the one the unknown-command branch of
        # `_run_slash_command` applies. It became load-bearing when the submit
        # handler stopped retiring the splash for every slash command: before
        # that the echo hid the difference, and `/logout` typed as a first
        # action collapsed the splash where `/USGE` did not.
        if self._providers is None:
            self._system_notice(
                "provider controller unavailable — usage cannot be fetched", "warning"
            )
            return
        target = arg.lower() if arg else ""
        if target:
            from local_operator.providers.usage import usage_kinds, usage_supported

            if self._providers.provider(target) is None and not usage_supported(target):
                self._system_notice(f"unknown provider: {target}", "warning")
                return
            wants_oauth, wants_key = usage_kinds(target)
            # "No endpoint" and "an endpoint you cannot reach" look identical in an
            # empty table, and only the second is something a user can act on. Said
            # up front rather than after a request that was always going to answer
            # nothing.
            if not wants_oauth and not wants_key:
                self._system_notice(f"{target} publishes no usage or quota endpoint", "warning")
                return
            if not self._providers.is_usable(target):
                need = "an API key" if wants_key else "an OAuth login"
                self._system_notice(f"{target} needs {need} before it can report usage", "warning")
                return
            if wants_oauth and not wants_key and not self._providers.has_any_credential(target):
                self._system_notice(f"{target} reports usage only after /login {target}", "warning")
                return
        self._open_usage_panel(target)

    def _open_usage_panel(self, target: str) -> None:
        """Show the panel in its loading state and start the fetch.

        The panel opens BEFORE the request. The fetch crosses the network once
        per logged-in provider, and a command whose only immediate effect was a
        transcript notice read as a command that had not run.
        """
        panel = self._usage_panel()
        if panel is None:
            self._append_block(NoticeBlock("usage panel unavailable", "warning"))
            return
        self._usage_focus_restore = self.focused
        generation = panel.start_fetch(target)
        panel.focus()
        # A second command replaces the first request. Without exclusivity a slow
        # response can overwrite the newer provider's report.
        self.run_worker(
            self._fetch_usage_worker(target or None, generation),
            thread=False,
            group="usage",
            exclusive=True,
        )

    def _usage_panel(self) -> UsagePanel | None:
        """The mounted panel, or None before compose (or in a stripped harness)."""
        try:
            return self.query_one(UsagePanel)
        except Exception:  # noqa: BLE001 — the panel is optional chrome
            return None

    async def _fetch_usage_worker(self, provider: str | None, generation: int) -> None:
        """Fetch usage and paint only if this request still owns the panel.

        A failure is reported INSIDE the panel rather than as a transcript
        notice: the panel is what has focus and what carries the key that
        retries, so sending the error anywhere else asks the user to look away
        from the surface holding the fix.
        """
        panel = self._usage_panel()
        try:
            assert self._providers is not None
            reports = await self._providers.fetch_usage([provider] if provider else None)
        except Exception as error:
            if panel is None:
                self._append_block(NoticeBlock(f"usage fetch failed: {error}", "error"))
            elif panel.accepts_request(generation):
                panel.show_error(f"usage fetch failed: {error}")
            return
        if panel is None:
            self._append_block(NoticeBlock("usage panel unavailable", "warning"))
            return
        if panel.accepts_request(generation):
            panel.show_reports(reports)

    def on_usage_refresh_requested(self, message: UsageRefreshRequested) -> None:
        """``r`` in the panel — re-run the same fetch behind the same view."""
        message.stop()
        if self._providers is None:
            return
        panel = self._usage_panel()
        if panel is None:
            return
        target = panel.target
        generation = panel.start_fetch(target)
        self.run_worker(
            self._fetch_usage_worker(target or None, generation),
            thread=False,
            group="usage",
            exclusive=True,
        )

    def on_usage_dismissed(self, message: UsageDismissed) -> None:
        """Esc/q in the panel — give focus back to whatever had it."""
        message.stop()
        # Dismissal retires the request as well as its surface. `show_reports`
        # opens the panel, so allowing a late response through would resurrect a
        # card the user explicitly closed.
        self.workers.cancel_group(self, "usage")
        restore = self._usage_focus_restore
        self._usage_focus_restore = None
        if restore is not None and getattr(restore, "is_mounted", False):
            restore.focus()  # type: ignore[union-attr]
        else:
            self._editor().focus()

    # -- the /btw aside ------------------------------------------------------
    #
    # The contract, because everything below is downstream of it: an aside
    # READS the conversation and never writes to it. The question and the
    # answer never enter the transcript and never enter the model's context for
    # the main conversation; Esc throws the exchange away and puts the user's
    # draft back. See ``widgets/aside_panel`` for why, and
    # ``SessionProtocol.complete_aside`` for the half that enforces it.
    #
    # ``^f`` is the user's opt-out, and the only one: it appends the exchange
    # as ordinary turns through ``adopt_aside``.
    def _aside_panel(self) -> AsidePanel | None:
        """The mounted card, or None before compose (or in a stripped harness)."""
        try:
            return self.query_one(AsidePanel)
        except Exception:  # noqa: BLE001 — the card is optional chrome
            return None

    def _aside_is_open(self) -> bool:
        panel = self._aside_panel()
        return panel is not None and panel.is_open

    def _cmd_btw(self, arg: str, typed: str = "") -> None:
        """``/btw [question]`` — open the aside, and ask if a question came with it.

        Bare ``/btw`` opens an empty card rather than printing a usage line.
        The command's whole job is to put the user in the aside, and they are
        already at a composer that now points at it; refusing to open until the
        question is retyped on one line would make the two forms behave like
        different commands.

        ``typed`` is the raw line, and retracting it is part of the contract.
        Every other route into the aside is covered by the composer's recording
        being off while the card is open, but THIS line is submitted while the
        card is still closed — so ``Editor._submit`` has already filed
        ``/btw are the subagents stuck?`` in the prompt history by the time
        this runs, question and all, on a surface that then promises Esc
        discards it. It is also the commonest way in.
        """
        if typed:
            self._editor().forget_last_prompt(typed)
        panel = self._open_aside()
        if panel is None:
            return
        if arg:
            self._ask_aside(arg)

    def action_aside(self) -> None:
        """Ctrl+B — open the aside WITHOUT spending the composer's draft."""
        self._open_aside()

    def _open_aside(self) -> AsidePanel | None:
        """Show an empty card and point the composer at it, stashing the draft.

        Idempotent: a second ``/btw`` while the card is up must not wipe the
        exchange in progress, and must not overwrite the stashed draft with the
        aside text the user has half typed.
        """
        panel = self._aside_panel()
        if panel is None:
            self._system_notice("aside unavailable", "warning")
            return None
        if panel.is_open:
            return panel
        # The aside needs the composer, and the full-page subagent view holds
        # it READ-ONLY — an aside opened over that page would have a card
        # inviting questions above a field that refuses every key, and its own
        # Esc would then hand back the wrong placeholder. So the page yields,
        # the same way it yields to a tool approval that needs an answer. Esc
        # out of the aside lands on the conversation, which is what Esc
        # promises everywhere else.
        self._close_subagent_view()
        editor = self._editor()
        self._aside_draft = editor.text
        editor.clear_content()
        editor.placeholder = ASIDE_PLACEHOLDER
        # The command list is borrowed too, and returned in `_close_aside`.
        # `on_editor_submitted` routes a slash-shaped line to the aside as a
        # question, so every row the picker offered here was dead: `/model`
        # became a question ABOUT `/model`, and with a row highlighted Enter
        # completed instead of asking. The picker also grows the dock, which
        # pulls the card's ceiling down and re-anchors it INTO the picker's
        # rows — the two surfaces drew through each other.
        editor.set_commands([])
        # And the prompt HISTORY, which is the one borrowed affordance that is
        # part of the contract rather than a convenience. `Editor._submit`
        # records before it posts, so without this an aside question survived
        # Esc: UP recalled it and the next Enter sent it to the agent as a real
        # turn — on a card that prints "esc discards it". It also stops the
        # aside burying the last thing the user actually said.
        editor.set_records_history(False)
        # Focus stays where it is, which is the composer: the aside is a place
        # to keep typing, so taking focus to the card would be taking it away
        # from the only input either surface has.
        editor.focus()
        # The conversation is INERT while the aside owns Enter, and the
        # stylesheet's rule is that what recedes is what is inert. `opacity`,
        # not `text-opacity` — measured, the latter does not reach the mounted
        # blocks. 60%, not the subagent page's 45%: the aside is a question
        # ABOUT the conversation, so it has to stay readable behind the card.
        self.screen.add_class(ASIDE_LAYOUT_CLASS)
        panel._on_height = self._reserve_ground_for_aside
        panel.open()
        self._sync_aside_fork_hint()
        return panel

    def _reserve_ground_for_aside(self, height: int) -> None:
        """Keep the transcript's own rows out from under the card.

        Receding the conversation says nothing about the part the card COVERS,
        and what it covers is the tail — measured at 120x24 the last thing the
        user asked was behind it, which is exactly the context they opened the
        aside to ask about. Padding the transcript by the card's height pushes
        the conversation up so the card rests on the END of it.

        The reserved rows change what "the end" means in pixels, so the anchor
        is consulted rather than the raw offset: it already knows whether the
        reader is following, and it re-measures the extent AFTER the padding
        lands. Yanking someone who had scrolled back to re-read something is
        what this must not do, and re-reading is a thing people do WHILE asking
        an aside about it.
        """
        transcript = self._transcript_view()
        if transcript is None:
            return
        padding = transcript.styles.padding
        if padding.bottom == height:
            return
        at_end = transcript.is_near_bottom()
        transcript.styles.padding = (padding.top, padding.right, height, padding.left)
        if at_end:
            transcript.follow_tail()

    def _close_aside(self) -> bool:
        """Dismiss the aside and give the main chat back. True if it was open.

        Everything the aside borrowed is returned in one place: the composer's
        placeholder, the draft it stashed, and the in-flight request. Whatever
        the user had half typed INTO the aside is dropped — it was a question
        they decided not to ask, and restoring it into the main composer would
        arm Enter with an aside question aimed at the conversation.
        """
        panel = self._aside_panel()
        if panel is None or not panel.is_open:
            return False
        # Retire the request as well as the surface: workers and messages are
        # separate queues, so cancelling alone cannot stop a delta already
        # posted. ``close`` bumps the card's generation, which drops it.
        self.workers.cancel_group(self, "aside")
        panel.close()
        panel._on_height = None
        # Give the transcript its own last rows back, and land the reader at
        # the end of the conversation they came back to.
        #
        # DROP the inline rule rather than write a number back. The sheet's own
        # bottom padding is the conversation's trailing row of ground (1 in the
        # conversation layout, 0 in the boot layout, `TranscriptView` and
        # `Screen.boot TranscriptView`), so any constant here is wrong in one of
        # the two: 0 ate the row this seam exists for, and 1 would leave a row
        # of slack under the splash after a `/btw` opened and closed on the boot
        # screen. Clearing the rule hands the question back to the cascade,
        # which is the only place that knows which layout is up.
        transcript = self._transcript_view()
        if transcript is not None:
            transcript.styles.clear_rule("padding")
            transcript.follow_tail()
        draft = self._aside_draft
        self._aside_draft = None
        editor = self._editor()
        editor.placeholder = DEFAULT_PLACEHOLDER
        editor.load_text(draft or "")
        editor.set_commands(SLASH_COMMANDS)
        editor.set_records_history(True)
        self.screen.remove_class(ASIDE_LAYOUT_CLASS)
        editor.focus()
        return True

    def _ask_aside(self, question: str) -> None:
        """Put a question to the aside and stream the answer into the card."""
        panel = self._aside_panel()
        if panel is None:
            return
        session = self._session
        if session is None or not hasattr(session, "complete_aside"):
            # Reported IN the card, not as a transcript notice: the card is
            # what the user is looking at, and a notice behind it would be the
            # one row the aside promised not to write.
            generation = panel.ask(question)
            panel.fail_answer(generation, "the session is still starting…")
            return
        generation = panel.ask(question)
        self._sync_aside_fork_hint()
        self.run_worker(
            self._aside_worker(session, question, generation),
            thread=False,
            group="aside",
            exclusive=True,
        )

    async def _aside_worker(self, session: SessionProtocol, question: str, generation: int) -> None:
        """One off-the-record request, streamed into the card.

        The turns handed to ``complete_aside`` are this aside's whole history
        plus the new question, so a follow-up ("and why is that slower?")
        resolves against what was already said HERE as well as the
        conversation. They are request-scoped: the session appends them to its
        live context for this call and keeps none of it.

        The in-flight assistant text is prepended when the session does not
        already carry it. The loop appends an assistant message to the context
        only once it settles, so mid-turn the sentence the user can SEE on
        screen is the one thing missing from what the model would be shown —
        and "what are you doing right now?" is the question this surface exists
        for. The tail check closes the one-hop window after ``message_end``,
        where the context has the settled message and the TUI has not yet
        cleared the block it streamed into.

        Spend IS counted. An aside carries the whole conversation, so it is not
        free, and a status line that a ctrl+b popup could move without moving
        would be a number the user cannot act on.
        """
        panel = self._aside_panel()
        if panel is None:
            return
        turns: list[AgentMessage] = []
        streaming = self._streaming_block
        # ``AssistantBlock.text`` is a METHOD, not a property — reading it
        # without the call handed the model a bound function and blew up the
        # one path this branch exists for.
        in_flight = streaming.text() if streaming is not None else ""
        if in_flight.strip():
            history = session.history()
            tail = history[-1] if history else None
            if getattr(tail, "text", None) != in_flight:
                turns.append(Message.assistant(in_flight))
        for turn in panel.turns[:-1]:
            if not turn.forkable:
                continue
            turns.append(Message.user(turn.question))
            turns.append(Message.assistant(turn.answer))
        turns.append(Message.user(ASIDE_PROMPT.format(question=question)))
        try:
            answer = await session.complete_aside(
                turns,
                on_delta=lambda delta: panel.append_answer(generation, delta),
                on_usage=self._charge_aside,
            )
        except Exception as error:  # noqa: BLE001 — any provider failure is the card's news
            # ``CancelledError`` is a ``BaseException``, so a worker cancelled
            # by `_close_aside` never lands here — no re-raise clause needed.
            panel.fail_answer(generation, str(error))
            # The question goes back in the composer so the footer's `enter ask
            # again` is a real retry, editable first. It is the only gesture
            # anyone wants under a failed answer.
            if panel.accepts(generation):
                self._editor().load_text(question)
            self._sync_aside_fork_hint()
            return
        panel.settle_answer(generation, answer)
        self._sync_aside_fork_hint()

    def _charge_aside(self, usage) -> None:  # noqa: ANN001 - harness Usage
        """Fold an aside's provider usage into the session's running cost.

        Through the SAME ``_cost_for`` the turn path uses, so the status line
        cannot price a turn one way and an aside another. The band repaints on
        its own tick; nothing is forced here, because a delta-rate repaint of
        the whole band is exactly the input lag the working line was fixed for.
        """
        cost = self._cost_for(usage)
        if cost is not None:
            self._total_cost += cost

    def _aside_can_fork(self) -> bool:
        """Whether ``^f`` would work right now — the card cannot know alone.

        Two inputs, one on each side: the card must hold a settled exchange,
        and the session must not be mid-turn. The second is not caution — the
        loop owns the message list while it runs and pairs each tool call with
        its result, so splicing a user message into a live batch produces a
        request no provider will accept (``Session.adopt_aside`` refuses it).
        """
        panel = self._aside_panel()
        if panel is None or not panel.is_open or not panel.fork_messages():
            return False
        session = self._session
        return session is not None and not session.is_streaming

    def _sync_aside_fork_hint(self) -> None:
        """Keep the card's footer honest about whether ``^f`` is live."""
        panel = self._aside_panel()
        if panel is not None:
            panel.set_fork_available(self._aside_can_fork())

    def action_fork_aside(self) -> None:
        """Ctrl+F — promote the aside exchange into the conversation.

        The one way an aside leaves a trace, and it is deliberate, explicit and
        the user's.

        Every refusal is STATED, and stated ON THE CARD. Silent would leave the
        user concluding the key is broken; a ``_system_notice`` would append a
        warning row to the conversation — from a surface whose title says
        nothing here joins it — drawn behind the card, so they would find it
        only after dismissing. The card refused, so the card says so.
        """
        panel = self._aside_panel()
        if panel is None or not panel.is_open:
            # ^f is unbound everywhere else, so pressed outside the aside it
            # would otherwise be a key that does nothing with no explanation.
            self._system_notice("ctrl+f forks an open aside — ctrl+b opens one", "warning")
            return
        pairs = panel.fork_messages()
        if not pairs:
            panel.set_notice("ask something first — there is nothing to fork")
            return
        session = self._session
        if session is None or not hasattr(session, "adopt_aside"):
            panel.set_notice("the session is still starting…")
            return
        if session.is_streaming:
            panel.set_notice("the agent is mid-turn — stop it first, then ^f")
            return
        panel.set_notice("")
        self.run_worker(self._fork_aside_worker(session, pairs), thread=False, group="aside-fork")

    async def _fork_aside_worker(
        self, session: SessionProtocol, pairs: list[tuple[str, str]]
    ) -> None:
        """Append the exchange to history and the transcript, then to the screen.

        The QUESTION is appended verbatim, not wrapped in ``ASIDE_PROMPT``.
        That wrapper is scaffolding for one request — it tells the model the
        turn is off the record, which is the opposite of what a forked turn is
        — and a transcript carrying it would replay the instruction forever.

        The card is closed BEFORE the rows are mounted so the conversation the
        user asked to keep is what they are left looking at, rather than a
        popup over it.
        """
        messages: list[Message] = []
        for question, answer in pairs:
            messages.append(Message.user(question))
            messages.append(Message.assistant(answer))
        try:
            await session.adopt_aside(messages)
        except Exception as error:  # noqa: BLE001 — surfaced, never swallowed
            self._system_notice(f"could not fork the aside: {error}", "warning")
            return
        self._close_aside()
        for question, answer in pairs:
            self._append_block(UserBlock(question))
            block = AssistantBlock()
            block.update_text(answer)
            block.finalize_text()
            self._append_block(block)
        plural = "exchange" if len(pairs) == 1 else "exchanges"
        self._notice(f"forked {len(pairs)} aside {plural} into the chat")

    # -- argument lists -------------------------------------------------------
    def on_argument_query_opened(self, message: ArgumentQueryOpened) -> None:
        """The buffer just entered ``/<command> …`` — fill that command's list.

        Answered on the MESSAGE for the same reason the model list is: every route
        into the list (typing the space, being completed into it by the command
        picker) then arrives at one place with one set of rows.

        ONE handler for every list-taking command, dispatching on the word. The
        registry states THAT a command offers values
        (:attr:`SlashCommand.arguments`) and this states WHICH, at the moment the
        list opens — which is what lets every row carry live state: the credential
        a ``/logout`` row would remove, the mode ``/approvals`` is in and the one
        it will boot in, the rungs THIS model accepts.
        """
        message.stop()
        editor = self._editor()
        picker = editor.picker
        if editor.argument_command != message.command:
            # The message is one message-loop tick old, and a tick is enough for
            # the user to have deleted the command or typed over it. Verified:
            # setting the buffer to `/logout ` and then to a sentence still
            # appended the notice below, attaching it to a command that no longer
            # exists. The buffer is the authority on which list is open, here as
            # much as in the editor's own resync.
            return
        if message.command == "approvals":
            picker.set_choices(self._approval_choices())
            picker.set_notice("")
            return
        if message.command == "effort":
            levels = self._effort_levels()
            picker.set_choices(self._effort_choices(levels))
            # The one list that is legitimately empty for a reason no query would
            # fix: this model takes no effort key at all, which is what a bare
            # `/effort` says in the transcript. Said here too, because a user who
            # opened the list is looking at the list, not at the ledger.
            label = getattr(self._session, "model_label", "") or "this model"
            picker.set_notice("" if levels else _effort_unavailable(label))
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

    def _approval_choices(self) -> list[ArgumentChoice]:
        """The four rows ``/approvals`` offers: two modes × two scopes.

        FOUR rows rather than two, and rather than a ``default`` row that then
        asks for a second word. The scope is the part of this decision that was
        invisible — "auto-approve for the next hour" and "auto-approve forever"
        were the same keystroke — so it is a COLUMN the eye can compare down,
        not a modifier the user has to remember to prepend. A ``default`` row
        leading to a second list would put the mode back behind a keystroke the
        list cannot show, which is the complaint this whole change answers.

        Every row is a complete, runnable argument (``default auto`` is what the
        buffer receives and what the handler parses), so choosing from the list
        and typing it out are the same command reaching the same code.

        ``detail`` carries the scope AND where the user already stands, because
        those are one question at the moment of choosing: `this session ·
        current` is the live mode, `every session · saved` is what the next
        launch will open in, and a user who set one but not the other can see
        both marks at once. ``alert`` on ``default auto`` only: it is the one
        row whose effect outlives the window it is typed in.
        """
        live_auto = self._approve_all
        saved_auto = self._approvals_default_auto
        session_mark = " · current"
        saved_mark = " · saved"
        return [
            ArgumentChoice(
                "ask",
                "Prompt before write and command tools",
                aliases=("on", "prompt"),
                detail="this session" + ("" if live_auto else session_mark),
            ),
            ArgumentChoice(
                "auto",
                "Run every tool without asking",
                aliases=("off", "yolo"),
                detail="this session" + (session_mark if live_auto else ""),
            ),
            ArgumentChoice(
                "default ask",
                "Prompt, saved as the default",
                detail="every session" + ("" if saved_auto else saved_mark),
            ),
            ArgumentChoice(
                "default auto",
                "Never prompt, saved as the default",
                detail="every session" + (saved_mark if saved_auto else ""),
                alert=True,
            ),
        ]

    def _effort_choices(self, levels: tuple[str, ...]) -> list[ArgumentChoice]:
        """``auto`` plus this model's ladder, the rung in force marked.

        ``auto`` leads for the reason it leads the printed listing: it is a rung
        of its own — the state a model with no level set is in, and the argument
        that returns to it — and without it every OpenAI reasoning model, which
        boots unset, would open a list with nothing marked.

        No per-level description. ``low``/``medium``/``high`` are the words
        themselves; prose beside each would be invented distinctions in a column
        the user is scanning for one they already understand.
        """
        current = getattr(_model_spec(self._session), "reasoning_effort", None) or "auto"
        rungs = ("auto", *levels)
        return [
            ArgumentChoice(
                name,
                "The model's own default" if name == "auto" else "",
                detail="current" if name == current else "",
            )
            for name in rungs
        ]

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
        # Rejections go through ``_system_notice`` (see `_cmd_usage`): nothing
        # ran, so the boot composition must survive them.
        if self._providers is None:
            self._system_notice(
                "provider controller unavailable — run: local-operator login", "warning"
            )
            return
        if not arg:
            items = [(p.id, p.name) for p in self._providers.login_providers()]
            # The only one of the five listings with no empty guard. With the
            # echo gone the listing IS the receipt, and `_tree_listing` drops
            # the caption with the rows on an empty list — so an empty registry
            # appended a blank block that retired the splash and rendered
            # nothing at all.
            if not items:
                self._system_notice("no providers support interactive login", "warning")
                return
            self._append_block(RichBlock(_tree_listing(items, "providers with interactive login")))
            return
        provider = arg.lower()
        if self._providers.provider(provider) is None:
            self._system_notice(f"unknown provider: {provider}", "warning")
            return
        definition = self._providers.provider(provider)
        if getattr(definition, "login", None) is None:
            self._system_notice(f"provider '{provider}' has no interactive login.", "warning")
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

        async def notice(body: str, kind: NoticeKind = "info") -> None:
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
        # Rejections go through ``_system_notice`` (see `_cmd_usage`): nothing
        # ran, so the boot composition must survive them.
        if self._providers is None:
            self._system_notice(
                "provider controller unavailable — run: local-operator logout <provider>",
                "warning",
            )
            return
        if not arg:
            self._system_notice("usage: /logout <provider>", "warning")
            return
        provider = arg.lower()
        self.run_worker(self._logout_worker(provider), thread=False, group="login")

    async def _logout_worker(self, provider: str) -> None:
        def notice(body: str, kind: NoticeKind = "info") -> None:
            self._append_block(NoticeBlock(body, kind))

        try:
            assert self._providers is not None
            message = await self._providers.logout(provider)
            notice(message)
        except Exception as error:
            notice(f"logout failed: {error}", "error")

    def _help_block(self) -> RichBlock:
        """Two-column help with a gutter wider than every command name."""
        muted = Style(color=theme_mod.semantic_color("muted"))
        dim = Style(color=theme_mod.semantic_color("dim"))
        named_commands = [
            (", ".join(f"/{name}" for name in command.names), command) for command in SLASH_COMMANDS
        ]
        # A literal 14 worked until `/model, /models` and `/resume, /recall`
        # became the two rows this feature rewrote. Both exceeded the floor and
        # glued straight onto their descriptions. Derive one shared column and
        # preserve the two-cell gutter for every row.
        name_width = max((len(names) for names, _ in named_commands), default=0) + 2
        lines = []
        for names, command in named_commands:
            line = Text()
            line.append(names.ljust(name_width), style=muted)
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
            footer.append("logs".ljust(name_width), style=muted)
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
            return RichBlock(
                _tree_listing(
                    [(skill.name, skill.description) for skill in visible], "loaded skills"
                )
            )
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
            return RichBlock(_tree_listing(items, "MCP servers"))
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
        # A new turn may ask questions again — expressed by MOVING PAST the latch
        # rather than by clearing it. Clearing raced the drain: a `TurnStarted`
        # already in the message queue when the stop landed ran before the parked
        # asker woke, so the stopped turn's write/exec tool got a fresh question.
        # An epoch bump cannot arrive early for an asker that captured the old one.
        self._turn_epoch += 1
        # D25: the ONE aggregate working line appears for the turn.
        self._working_fallback = DEFAULT_ACTIVITY
        self._start_working_block()

    def on_turn_ended(self, message: TurnEnded) -> None:
        assert self._status is not None
        self._dismiss_working_block()
        # Build the segments as typed locals and make ONE call: a
        # dict[str, object] splatted into update() erases every parameter type,
        # so a wrong-typed segment would only surface as a render glitch.
        # `None` means "leave this segment alone" in update()'s contract.
        #
        # Gated on the session still being here. A plain `/reload` deliberately
        # keeps the controller SUBSCRIBED across `dispose()` so the dying
        # session's `agent_end` can settle its live tool cards — and that same
        # event carries a `context_tokens` for the conversation being thrown
        # away. `_post` queues it, so whether it arrives before or after the
        # reload's reset is a scheduling race; arriving after, it reinstates an
        # exact reading for a dead session AND the exact-count guard then
        # suppresses the replacement session's own measurement, which is
        # precisely the staleness the reset exists to prevent. The cards still
        # settle either way; only the number is refused.
        context_tokens: int | None = None
        if self._session is not None:
            context_tokens = message.context_tokens or None
        cost_text: str | None = None
        cost = self._cost_for(message.usage)
        if cost is not None:
            self._total_cost += cost
        # Fold in whatever the children have spent BEFORE reading the total: a
        # turn that delegated has almost certainly moved their figures, and the
        # 1 Hz poll would otherwise be what first showed it.
        self._harvest_subagent_costs()
        total = self._spend_total()
        if cost is not None or (self._session is not None and self._subagent_costs):
            # A turn that priced nothing itself still has a total worth showing
            # once a child has spent — a parent whose entire turn was one `task`
            # call reports no usage of its own, and reading "$—" beside a working
            # subagent says the session is free when it is not.
            #
            # Gated on a LIVE session for the same reason `context_tokens` is
            # above. A plain `/reload` keeps the controller subscribed so the
            # dying session's `agent_end` can settle its cards, and that event
            # arrives after the reset has cleared the band; `_cost_for` already
            # returns None for a dead session, so without this clause the leftover
            # dict alone would repaint the dead conversation's total into a band
            # that was just emptied on purpose.
            cost_text = format_cost(total)
        elif message.usage is not None and getattr(message.usage, "input_tokens", 0):
            # D20: the turn billed tokens but pricing is unknown — render an
            # explicit "unavailable" so the segment's absence reads as that,
            # not as "free".
            cost_text = "$—"
        self._status.update(
            streaming=False,
            context_tokens=context_tokens,
            # The provider's own prompt_tokens: exact, and it supersedes the
            # boot estimate permanently. `None` leaves the flag alone, so a turn
            # that reported no usage does not demote a standing estimate.
            context_is_estimate=None if context_tokens is None else False,
            context_window=_context_window(self._session),
            cost=cost_text,
        )
        if message.error:
            self._append_block(NoticeBlock(message.error, "error"))
        elif message.aborted and not self._interrupted_cards:
            # Only when NOTHING was in flight. A stopped turn already says so on
            # each card it stopped (`⊘ interrupted`), and that per-card mark is
            # the more useful of the two because it names WHICH tool stopped;
            # adding a standalone notice spent a row and a gap restating it, and
            # N+1 rows when several tools were running.
            self._append_block(NoticeBlock("interrupted", "warning"))
        self._interrupted_cards = 0

    def _start_working_block(self, *, ends_empty_state: bool = True) -> None:
        """Mount the turn's working line, pinned to the foot of the transcript.

        Idempotent: ``agent_start`` is the only event that opens a turn, but the
        line is also re-mounted after a ``/clear`` that landed mid-turn, and the
        two must not stack.
        """
        if self._working_block is not None:
            return
        self._working_block = WorkingBlock(*self._current_activity())
        self._append_block(self._working_block, ends_empty_state=ends_empty_state, pin_tail=True)

    def _dismiss_working_block(self) -> None:
        """Stop and remove the aggregate working line at turn end (D25).

        The one exit for every way a turn can finish — completed, failed, or
        aborted — because they all arrive as the same ``agent_end``.
        """
        if self._working_block is not None:
            self._working_block.stop()
            self._transcript_view().remove_block(self._working_block)
            self._working_block = None
        self._working_fallback = DEFAULT_ACTIVITY

    def _refresh_working_activity(self) -> None:
        """Re-derive what the working line says after the turn's state moved."""
        if self._working_block is not None:
            self._working_block.set_activity(*self._current_activity())

    def _current_activity(self) -> tuple[str, str]:
        """What the agent is doing right now: ``(label, phase)``.

        DERIVED from the same state the transcript is drawn from rather than
        latched by each handler, so the line cannot disagree with the ledger
        under it — the failure mode of a latch is a working line still naming a
        tool whose row settled two events ago, which is exactly the lie this
        line was reported for telling.

        The PHASE is the coarse state and is what the elapsed clock is keyed to;
        the label is what the row says. They are separate because a batch losing
        one of three calls, a tool name arriving in fragments and the model
        revising its intent all change the label without the agent having
        changed what it is doing, and restarting the clock for those made the
        row contradict the receipt two lines above it.

        Read in priority order, most specific first. Running work outranks a
        call still being dictated, which outranks prose, which outranks the
        whole-turn fallback; a turn with no tools at all therefore never leaves
        the last two, and a turn between two tool batches falls back to
        "thinking", which is the honest description of a model call in flight.
        """
        if self._approval is not None and not self._approval.answered:
            # Nothing is running: the turn is parked on the question on screen,
            # and "thinking" under an unanswered prompt blames the model for a
            # wait that belongs to the user.
            return ("waiting for approval", "approval")
        if self._tool_cards:
            return (self._batch_phrase(list(self._tool_cards.values())), "running")
        if self._composing_cards:
            # The tool's NAME is deliberately absent. It arrives in fragments —
            # `wr` then `write` — and the ledger row above follows those because
            # its name column is an identifier field; a status sentence is not,
            # and `composing wr` reads as a typo rather than as a state.
            count = len(self._composing_cards)
            noun = "a call" if count == 1 else f"{count} calls"
            return (f"composing {noun}", "composing")
        if self._streaming_block is not None:
            return (ACTIVITY_RESPONDING, ACTIVITY_RESPONDING)
        return (self._working_fallback, self._working_fallback)

    @staticmethod
    def _batch_phrase(cards: list[ToolCard]) -> str:
        """The model's intent for a single call; a plain count for a batch.

        A batch drops the intent rather than suffixing it. `+2 more` buried the
        count behind arithmetic, put a `+` sign in prose, and — with a real
        intent — concatenated a sentence with a number: `Auditing merged MRs +2
        more` presents one call's stated purpose as the whole turn's activity,
        which the three rows above it immediately contradict. The count is the
        one fact this row has that appears nowhere else in the frame, so it is
        what a batch says.
        """
        return batch_activity(
            [tool_activity(display_name(card.tool_name), card.intent) for card in cards]
        )

    def _cost_for(self, usage) -> float | None:
        """Best-effort cost of ONE parent turn, or ``None`` when unpriceable.

        Delegates the arithmetic to :mod:`local_operator.tui.costs` so this and
        the subagent surfaces cannot disagree about what a turn cost. Never
        raises: a price is not worth a broken frame.
        """
        if self._session is None:
            return None
        return turn_cost(self._session.model_label, usage)

    def _harvest_subagent_costs(self) -> None:
        """Record what each child has spent, keyed by job id.

        REPLACES each entry rather than adding to a running total, because a
        child's figure grows while it works: the same job is observed many times
        and only its latest value is its spend.

        Called from the 1 Hz poll rather than only at turn end. A delegated child
        outlives the turn that launched it — the parent finishes, the band goes
        idle, and the child keeps spending for minutes — so a turn-end-only
        harvest would leave the total frozen through exactly the period when it
        is moving. This is also why the entries are never dropped: settled jobs
        leave the ledger after ``AsyncJobManager``'s retention window, and a
        total that falls when a finished child is evicted is worse than none.
        """
        session = self._session
        manager = getattr(session, "jobs", None)
        if manager is None:
            return
        try:
            jobs = manager.list()
        except Exception:  # noqa: BLE001 — a status number never takes the app down
            return
        label = getattr(session, "model_label", "")
        for job in jobs:
            cost = job_cost(job, default_model_label=label)
            if cost is not None:
                self._subagent_costs[job.id] = cost

    def _spend_total(self) -> float:
        """Everything this session has spent: its own turns plus its children's.

        ONE blended number, which is what the band renders. A split
        (``$0.42 +$0.19``) was the alternative and is the wrong trade here: the
        cost segment sheds at rung 8 of a 12-rung drop ladder
        (``status_line._DROP_LADDER``), so widening it buys a breakdown at the
        price of the whole segment disappearing sooner on a narrow terminal. The
        band answers "what has this session cost me", which is one number; the
        per-child breakdown already has a home with more room in the subagent
        panel and the full-page view, where each row carries its own figure.
        """
        return self._total_cost + sum(self._subagent_costs.values())

    def on_turn_boundary_start(self, message: TurnBoundaryStart) -> None:
        """turn_start: one model call is beginning.

        The gap this opens is the one the working line exists for. After a tool
        batch settles the agent goes quiet for as long as the next model call
        takes, with no card to show for it; re-deriving here is what puts
        "thinking…" back on the line instead of leaving the last tool's name up
        until the next event happens to arrive.
        """
        self._refresh_working_activity()

    def on_turn_boundary_end(self, message: TurnBoundaryEnd) -> None:
        """turn_end: reconcile orphaned RUNNING tool cards (TUI-008/019).

        The count is kept because it decides whether an aborted turn ALSO needs a
        standalone "interrupted" notice: each card it marks already says so, and
        naming the tool that stopped is the more useful of the two statements.
        """
        self._interrupted_cards = self._retire_live_tool_cards()
        self._refresh_working_activity()

    def _retire_live_tool_cards(self) -> int:
        """Settle every card still claiming to be live, and say how many.

        Composing rows count too: a turn that ends while the model is still
        dictating a call leaves a row that will never start, and leaving it
        "live" strands a spinner on a finished turn.

        Shared with :meth:`_reload_session` rather than living only on the
        ``turn_end`` path, because a session torn down MID-TURN never delivers a
        ``turn_end`` at all. On a plain ``/reload`` the ledger is preserved, so
        without this the dead turn's cards animate forever — and, less
        obviously, ``_current_activity`` derives the working line's label from
        those cards, so the NEXT turn opened by announcing ``running bash`` for
        a bash call that had died with the previous session.
        """
        cards = list(self._tool_cards.values()) + list(self._composing_cards.values())
        for card in cards:
            card.mark_interrupted()
        self._tool_cards.clear()
        self._composing_cards.clear()
        return len(cards)

    def on_assistant_message_start(self, message: AssistantMessageStart) -> None:
        """A message opened — but nothing is MOUNTED until text actually arrives.

        A tool-use turn opens a message and goes straight to the tool calls with
        no prose at all (every Anthropic tool turn looks like this, so it is the
        common shape, not an edge case). Mounting the block here spent two rows
        on it regardless: the empty block's own row, plus the blank row the
        spacing rule opens above a block of a different kind. That read as a
        hole between the working line and the first tool row — the excess
        spacing reported against the tool ledger was this, not the ledger.

        Deferring the mount to the first delta costs nothing: the block carries
        no state of its own before it has text, so there is nothing to hold.
        """
        self._streaming_block = None

    def _ensure_streaming_block(self) -> AssistantBlock:
        """The block for the message being streamed, mounted on first use."""
        block = self._streaming_block
        if block is None:
            block = AssistantBlock()
            self._streaming_block = block
            self._append_block(block)
            # Only on the MOUNT, not on every delta: the deltas arrive at 30 Hz
            # and would all re-derive the same phrase.
            self._refresh_working_activity()
        return block

    def on_assistant_delta(self, message: AssistantDelta) -> None:
        # Empty deltas are not text: flushing one would mount a block that then
        # sits blank until real content lands, which is the hole this avoids.
        if not message.text:
            return
        self._ensure_streaming_block().update_text(message.text)

    def on_assistant_message_end(self, message: AssistantMessageEnd) -> None:
        # An empty authoritative text is not an instruction to erase what
        # streamed. Adopting it unconditionally destroyed the prose the deltas
        # had already painted and left an empty block mounted in its place; the
        # controller only falls back to its own buffer when the text is None, so
        # a provider or abort path reporting "" reaches here. Keep whatever the
        # block has, and mount nothing when nothing streamed.
        if not message.text:
            block = self._streaming_block
            self._streaming_block = None
            if block is not None:
                block.finalize_text()
            self._refresh_working_activity()
            return
        block = self._ensure_streaming_block()
        # TUI-020: adopt the authoritative text carried by the event.
        block.update_text(message.text)
        block.finalize_text()
        self._streaming_block = None
        # The prose is settled, so "responding…" is over: whatever the turn does
        # next — another model call, a tool batch — the line must stop claiming
        # text is still arriving.
        self._refresh_working_activity()

    def on_context_usage_reported(self, message: ContextUsageReported) -> None:
        """Move the context reading DURING a turn, not only when it ends.

        An agentic turn is many model calls over many minutes, and each one
        reports the context it ran against. Waiting for ``agent_end`` meant the
        band showed the pre-turn size for the whole time the agent was working
        — the exact stretch a user watches it for. Reported as exact, because
        it is the provider's own number.
        """
        assert self._status is not None
        self._status.update(
            context_tokens=message.context_tokens,
            context_is_estimate=False,
            context_window=_context_window(self._session),
        )

    def on_tool_composing(self, message: ToolComposing) -> None:
        """Show the call the model is still dictating (TUI-026).

        Mounted as soon as the tool's NAME is known, which is many seconds — for
        a file, minutes — before the call itself exists. Until this landed the
        screen held completely still while a large `write` streamed, and the only
        reasonable reading of that frame was that the agent had hung.
        """
        event = message.event
        card = self._composing_cards.get(event.tool_call_id)
        if card is None:
            card = ToolCard(event.tool_call_id, event.tool_name)
            self._composing_cards[event.tool_call_id] = card
            self._append_block(card)
        card.set_composing(event.argument_bytes, event.tool_name)
        # The intent arrives from the STREAM, as soon as the model has closed
        # its `i` string — many seconds, for a large `write` minutes, before the
        # call exists. That silence is the longest in a turn and the one the
        # working line was reported for saying nothing through. `or` rather than
        # assignment: a later frame reporting none must not erase one already
        # shown, which would blank the line mid-dictation.
        card.intent = clean_intent(getattr(event, "intent", None)) or card.intent
        self._refresh_working_activity()

    def on_tool_started(self, message: ToolStarted) -> None:
        event = message.event
        # Adopt the row that announced this call rather than mounting a second
        # one: the composing card already sits in the right place in the ledger,
        # and swapping it out would flicker a row away and an identical row back
        # at the exact moment the call starts running.
        card = self._adopt_composing_card(event.tool_call_id, event.tool_name)
        if card is not None:
            card.begin_running(event.tool_name, event.args, event.intent)
        else:
            card = ToolCard(event.tool_call_id, event.tool_name, event.args, event.intent)
            self._append_block(card)
        self._tool_cards[event.tool_call_id] = card
        self._refresh_working_activity()

    def _adopt_composing_card(self, tool_call_id: str, tool_name: str) -> ToolCard | None:
        """The composing row for this call, if one is on screen.

        Matched by id first. A provider that does not send a call id until the
        end of its stream leaves the row keyed by a placeholder, so the fallback
        takes the oldest composing row with the same tool name — the calls of one
        batch start in the order they were composed.
        """
        card = self._composing_cards.pop(tool_call_id, None)
        if card is not None:
            return card
        for key, candidate in self._composing_cards.items():
            if candidate.tool_name == tool_name:
                del self._composing_cards[key]
                return candidate
        return None

    def on_tool_updated(self, message: ToolUpdated) -> None:
        """TUI-007: feed the partial result into the card's LIVE view.

        Into the expansion, not the summary: the summary is the command, which
        a running row must keep (the card is the fact, the working line is the
        intent). The card neither repaints nor bounds anything here — it
        coalesces the update onto its own 1 Hz clock, so a command printing a
        line a millisecond costs one repaint a second.
        """
        card = self._tool_cards.get(message.event.tool_call_id)
        if card is None:
            return
        detail = _partial_text(message.event.partial_result)
        if detail:
            card.set_partial_detail(detail)

    def on_tool_ended(self, message: ToolEnded) -> None:
        event = message.event
        card = self._tool_cards.pop(event.tool_call_id, None)
        # Before the early return below: a batch that just lost one of three
        # calls still has to drop its count, and a call that ended with no card
        # on screen still ended.
        self._refresh_working_activity()
        # A todo change is visible the moment its tool settles, rather than on
        # the next 1 Hz poll: the receipt and the band update together. The
        # subagent panel needs no such hook — its data changes only through
        # Subagent* events, which already repaint.
        if str(getattr(event, "tool_name", "") or "") == "todo":
            self._refresh_band()
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
        # Compaction is a whole-turn state with no card of its own, and it is
        # slow: without this the line says "thinking" through a minute of
        # summarisation the notice above it announced and then never revisits.
        self._working_fallback = "compacting context"
        # A MANUAL pass has no turn around it, so no `agent_start` mounted a
        # working line for the fallback to fill and no `_status.update` marked
        # the session busy — the reason above applied only to the automatic
        # trigger, and a multi-minute pass under an idle band is also what makes
        # a user reasonably start typing (see `_submit_prompt`). Mounted here
        # instead, and remembered so only the line this handler started is the
        # line it takes away.
        self._compacting = True
        self._compaction_owns_working_block = self._working_block is None
        if self._compaction_owns_working_block:
            self._start_working_block()
            if self._status is not None:
                self._status.update(streaming=True)
        self._refresh_working_activity()

    def on_compaction_ended(self, message: CompactionEnded) -> None:
        if message.success:
            self._append_block(NoticeBlock(compaction_receipt(message), "info"))
            # BOTH triggers, not just `/compact`: the band reports the size of
            # the next request, and an automatic pass that left it at the
            # pre-compaction figure is the same stale frame for the same reason.
            # The event carries the figures, so this is the one place that can
            # serve both.
            self._settle_context_reading(message.tokens_before, message.tokens_after)
        elif message.reason != "manual":
            # AUTOMATIC only. A manual pass is narrated by `_compact_worker`,
            # which holds the outcome and therefore the reason it failed;
            # printing this bare line in front of that one gave the user two
            # notices for one event, with the empty one the louder of the two.
            # The automatic trigger has no other narrator, so it keeps it.
            self._append_block(NoticeBlock("compaction failed", "error"))
        self._compacting = False
        if self._compaction_owns_working_block:
            self._compaction_owns_working_block = False
            self._dismiss_working_block()
            if self._status is not None:
                self._status.update(streaming=False)
        self._working_fallback = DEFAULT_ACTIVITY
        self._refresh_working_activity()
        # A prompt typed DURING the pass was held rather than sent into a
        # history being rewritten; now the history is settled, it goes.
        held, self._prompt_held_for_compaction = self._prompt_held_for_compaction, ""
        if held:
            self._start_turn(held)

    def on_retry_started(self, message: RetryStarted) -> None:
        body = f"retry {message.attempt}: {message.error}"
        if message.fallback_model:
            body += f" → falling back to {message.fallback_model}"
        self._append_block(NoticeBlock(body, "warning"))
        # A retry is backoff, not thinking, and the wait is the whole point of
        # saying so: the attempt number is what tells a watcher whether the
        # provider is recovering or the turn is stuck in a loop.
        self._working_fallback = f"retrying (attempt {message.attempt})"
        self._refresh_working_activity()

    def on_retry_ended(self, message: RetryEnded) -> None:
        if message.success:
            self._append_block(NoticeBlock("retry succeeded", "info"))
        else:
            self._append_block(NoticeBlock("retry failed", "error"))
        self._working_fallback = DEFAULT_ACTIVITY
        self._refresh_working_activity()

    # Subagent events arrive through the Controller's `_post` on the shared
    # stream; each is an immediate repaint trigger for the band, paired with
    # the 1 Hz poll as the belt (see `_refresh_band`). The panel re-reads the
    # manager itself, so the handlers only need to fire the refresh, never to
    # carry job data.
    def on_subagent_started(self, message: SubagentStarted) -> None:
        self._refresh_band()

    def on_subagent_progress(self, message: SubagentProgress) -> None:
        self._refresh_band()

    def on_subagent_ended(self, message: SubagentEnded) -> None:
        self._refresh_band()


def _model_spec(session) -> Any | None:
    """The session's active ``ModelSpec``, or ``None`` when it has none.

    Defensive because the TUI also runs against reduced hosts — embedders
    and the pilot fakes supply a session without the spec accessor — and a
    status segment must degrade to "unknown" rather than take the app down.
    """
    return getattr(session, "model", None)


def _model_name(session) -> str:
    """The model's resolved human name, or "" when the host cannot say.

    Read off the spec rather than looked up here because ``build_model_spec``
    already resolved the metadata through the catalogue, so this is the ONLY
    place the band can learn the name of a model no registry row covers — an
    OpenRouter id, an Ollama tag — without a disk read inside a repaint.

    ``getattr`` for the same reason :func:`_model_spec` uses it: the pilot fakes
    and embedding hosts supply specs of their own shape, and a missing name is a
    segment that falls back to the selector, not a crash.
    """
    return str(getattr(_model_spec(session), "display_name", "") or "")


def _context_window(session) -> int:
    """The active model's context window, or 0 when it is unknown.

    Zero is meaningful downstream: the usage segment renders ``12.4k/—``
    rather than inventing a denominator to divide by.
    """
    window = getattr(_model_spec(session), "context_window", 0) or 0
    return int(window) if window > 0 else 0


#: How the two compaction mechanisms are named in the receipt. ``context-full``
#: is the config spelling of "ask a model to summarize the history in words";
#: printing it verbatim asks the reader to know the setting, and the word that
#: distinguishes it from the other mechanism is the mechanism itself.
_STRATEGY_LABELS = {"snapcompact": "snapcompact", "context-full": "summary"}


def compaction_receipt(message: CompactionEnded) -> str:
    """``context compacted`` plus what the pass achieved.

    Compaction is the slowest thing a session does with nothing to show: the
    transcript on screen is untouched, only the model's view of it shrank. The
    bare notice therefore asked the user to take a minute of waiting on faith,
    and the owner's report of a ``/compact`` that "just echoes a description"
    is the same complaint one step earlier — a compaction with no visible
    result cannot be told from one that never ran.

    The token pair is that result, and the strategy names WHICH pass produced
    it (a vision model's image frames or a written summary) — the only
    externally visible difference between the two mechanisms. Both are
    dropped when the pass reported no figures (a host on an older event) or
    when they fail to show a reduction, because a receipt claiming a saving
    that did not happen is worse than the bare line.
    """
    before, after = message.tokens_before, message.tokens_after
    if before <= 0 or after <= 0 or after >= before:
        return "context compacted"
    saved = f"{(1 - after / before) * 100:.0f}% smaller"
    detail = f"{format_context_tokens(before)} → {format_context_tokens(after)} tokens ({saved})"
    label = _STRATEGY_LABELS.get(message.strategy, message.strategy)
    if label:
        detail += f", via {label}"
    return f"context compacted · {detail}"


def mode_word(auto: bool) -> str:
    """The approval mode as the ONE word every surface spells it with.

    The receipts, the bare report, the picker rows, the config file and the
    command's own argument all say ``ask`` or ``auto``, and they say it through
    here. The alternative is what the feature had — ``auto`` in the receipt,
    ``True`` in the band's parameter, ``off``/``yolo`` accepted but never
    printed — which is how a user ends up unsure whether two lines are
    describing the same setting.
    """
    return "auto" if auto else "ask"


def _effort_unavailable(label: str) -> str:
    """The ONE sentence for "this model has no dial", used by key and command.

    One string because it is one fact, and two phrasings of one fact read as two
    authors. "Not adjustable" rather than "has no reasoning-effort levels"
    because the latter is false of a model like ``deepseek-reasoner``, which
    reasons at a depth the API exposes no name for: the band says ``reasoning``
    for it, and the old wording had the transcript contradicting the band one
    row above it. Opens with the same ``reasoning effort:`` subject as every
    other line in the feature.
    """
    return f"reasoning effort: not adjustable on {label}"


def _effort_label(session) -> str:
    """The model's reasoning-effort label, or "" when it has none.

    Three states, three words, because the segment's job after the effort
    control landed is to name the LEVEL in force:

    - a level is set → that level (``high``), the ordinary case;
    - the model has a ladder and no level is set → ``auto``, the same word
      ``/effort auto`` uses for it, so the band and the command share one
      vocabulary. It used to read ``reasoning``, which is not a rung — putting
      a category noun in a value slot made it look like a sixth level, and on
      OpenAI (seeded with nothing by design) that was the state most users met
      first;
    - the model reasons with no ladder at all (``deepseek-reasoner``) →
      ``reasoning``, which is all that can honestly be said about it.

    Non-reasoning models render nothing, which is what makes the segment's
    presence informative.
    """
    spec = _model_spec(session)
    if spec is None:
        return ""
    explicit = getattr(spec, "reasoning_effort", None)
    if explicit:
        return str(explicit).strip().lower()
    if getattr(spec, "reasoning_efforts", ()):
        return "auto"
    return "reasoning" if getattr(spec, "reasoning", False) else ""


def _messages(count: int) -> str:
    """``1 message`` / ``12 messages`` — the reload receipts count out loud."""
    return f"{count} message" if count == 1 else f"{count} messages"


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


def _tree_listing(items: list[tuple[str, str]], caption: str) -> Group:
    """Tree-glyph section: ├─ / └─, name in the string tint, detail dim (D4).

    ``caption`` names WHAT the tree lists, on a dim row above it. It exists
    because these listings are the receipt for the command that printed them
    (see ``SLASH_COMMANDS``), and a receipt that does not say what it is only
    reads while the keystroke is still fresh: rendered at 120x40, three
    consecutive listings stacked into one anonymous run of tree glyphs, and
    the provider list and the credential list are the pair a reader is most
    likely to confuse — both are one row per provider id.

    A dim TEXT row, not a rule or a boxed header: the transcript's chrome is
    borderless by mandate (``docs/REWRITE.md``), and the tree glyphs already
    carry the grouping. This only has to answer "of what".

    REQUIRED, with no empty default: the whole reason the slash commands that
    print these no longer echo what was typed is that the listing is the
    receipt, and a listing that can silently omit its caption is not one. The
    same reasoning as ``SlashCommand.echo``'s pinned policy table — a new call
    site has to state its answer.
    """
    if not items:
        return Group()
    name_style = Style(color=theme_mod.semantic_color("string"))
    dim = Style(color=theme_mod.semantic_color("dim"))
    lines: list[Text] = [Text(caption, style=dim)]
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


def _home_relative(path: str) -> str:
    """``~/.local-operator/config.yml`` rather than the full ``/Users/…`` form.

    The prefix is the same on every machine and costs a third of the line the
    confirmation has to spend saying WHERE it wrote. A path outside the home
    tree — the ``LOCAL_OPERATOR_CONFIG_DIR`` override, a test's tmp dir — is
    left absolute, because there is no shorter honest rendering of it.
    """
    home = os.path.expanduser("~")
    if home in ("", "/") or not path.startswith(home + os.sep):
        return path
    return "~" + path[len(home) :]
