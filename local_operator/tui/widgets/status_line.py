"""Status line — the full-width BAND on the kit's ``sunken`` ground (D3/D17).

The character refinement supersedes the thin border-row trick: the status
line is a full-width band at the bottom of the input dock, painted on the
``sunken`` ground with segments separated by ``·``:

    π openrouter/kimi-k2 · high · ~/local-operator
                    2 agents · 49.6%/1M · $0.12 · 41m1s · Parser rewrite

(one row on a wide terminal; wrapped here only to fit this docstring)

Left group: brand glyph · provider/model · effort · working dir (+ the
shimmering working indicator while a turn streams — faithful: the working
text rides the shimmer sweep; when shimmer is off a static dim spinner keeps
it legible, D26). Right group, right-aligned: subagents · context usage ·
cost · active duration · conversation name.

**Nine segments do not fit an 80-column terminal, and the band is exactly one
row at every width.** Both facts are load-bearing, so overflow is not left to
truncation — truncation would clip whichever segment happened to be last and
leave a half-written number on screen. Instead :data:`_DROP_LADDER` names an
explicit reduction order, applied step by step until the row fits. The
ordering protects what the operator steers by (provider/model, working dir,
context usage) and spends the transient counters (background jobs, elapsed
time) first. All widths measured with ``rich.cells.cell_len`` (one width
model).
"""

from __future__ import annotations

import time
from dataclasses import dataclass
from pathlib import Path
from typing import Callable

from rich.cells import cell_len
from rich.style import Style
from rich.text import Text
from textual.widgets import Static

from local_operator.tui import theme as theme_mod
from local_operator.tui.widgets.tool_card import truncate_cells

#: Spinner frames shown while the session is streaming (~12.5 fps glyph
#: cadence when shimmer is disabled).
_SPINNER_FRAMES = ("⣾", "⣽", "⣻", "⢿", "⡿", "⣟", "⣯", "⣷")
_SPINNER_INTERVAL_S = 0.08

#: Segment icons. Every one is a SINGLE cell in ``rich.cells.cell_len`` — this
#: is a hard requirement, not a preference: the band's whole layout is measured
#: arithmetic, and a glyph that renders two cells wide on one terminal and one
#: on another makes the right group's edge alignment drift. Emoji are excluded
#: for exactly that reason (the omp reference uses them; it can afford to).
#: Geometric and technical symbols only, so no Nerd font is required.
ICON_MODEL = "◆"
ICON_EFFORT = "◐"
ICON_CWD = "⌂"
ICON_AGENTS = "◍"
ICON_JOBS = "⊞"
ICON_CONTEXT = "▦"
ICON_COST = "◈"
ICON_DURATION = "◷"
#: MCP servers. ``⊙`` (U+2299) is the reference's glyph and measures ONE cell
#: in ``rich.cells.cell_len``, so it satisfies the single-cell rule above; it
#: was checked rather than assumed, because the band's arithmetic is exact and
#: a two-cell glyph would drift the right group's edge by one column.
ICON_MCP = "⊙"

#: Separators point INWARD: the left group's chevrons aim right and the right
#: group's aim left, so both runs converge on the centre gap and frame it. A
#: symmetric separator (the middot this replaced) left the two groups reading as
#: one long run once the gap narrowed.
_SEP_LEFT = "›"
_SEP_RIGHT = "‹"

#: Legacy separator, still used by nothing in the band. Kept only because the
#: intra-group width arithmetic in :data:`_MIN_GROUP_GAP` is stated relative to
#: it in the comments below.
_SEPARATOR = " · "

#: Minimum cells between the left and right groups. Deliberately WIDER than the
#: 3-cell intra-group separator so the seam between the groups always dominates
#: the seams inside them — otherwise the band's identity-left/numbers-right
#: architecture dissolves into one run at ordinary widths.
_MIN_GROUP_GAP = 4

#: Reduction order, cheapest loss first. Each entry is applied in turn until
#: the row fits the available width.
#:
#: The ordering principle: shed what the user already knows or can re-derive,
#: and protect what predicts their next decision. In order —
#:
#: * ``name`` is a label the user typed; they know it.
#: * ``duration`` is re-derivable from the transcript.
#: * ``subagents`` is a counter, but NOT re-derivable without scrolling, which
#:   is why it outlasts the two above rather than going first.
#: * the two SHORTEN steps outrank every remaining drop, because they recover
#:   width while keeping the segment: a basename still answers "where am I" and
#:   a bare model id still answers "who is replying". Together they free ~35
#:   cells on a realistic label, more than cost and context cost combined, so
#:   dropping numbers to preserve a fully-qualified path would be a bad trade.
#: * ``effort`` is a static setting the user chose. It does not change while
#:   they watch, so it goes before either live number — an earlier version had
#:   it OUTLIVING context usage, which meant a band could show `high` but not
#:   `49.6%/1M`: it kept the field nobody re-reads and dropped the one that
#:   says compaction is coming.
#: * ``cost``, then ``cwd``, then ``context``, then ``mcp``: context usage is the
#:   one an operator acts on, and the MCP indicator outlives even that WHEN IT IS
#:   AN ALARM — it is the cheapest segment to keep and the only one that can turn
#:   into one. A healthy count is only a courtesy, so it sheds like one; see
#:   :func:`drop_ladder`.
#:
#: The brand glyph, the streaming spinner and the model label are NEVER dropped.
#: When even the irreducible row overflows, ``_render`` emits spinner, glyph and a
#: TRUNCATED label rather than shedding any of them: `deepse…` still answers which
#: model is replying, a band reduced to a bare glyph on an empty tinted strip
#: reads as broken rather than as compressed, and a streaming band that renders
#: identically to an idle one is the one thing the colour budget must never allow.
_DROP_LADDER: tuple[str, ...] = (
    # A label the user typed and already knows.
    "name",
    # Re-derivable from the transcript.
    "duration",
    # Counters. Not re-derivable without scrolling, so they outlast the two
    # above; jobs go before agents because a backgrounded shell command is
    # visible in the transcript as a tool card, while a subagent is not.
    "jobs",
    "subagents",
    # Shorten before dropping: a basename still answers "where am I" and a bare
    # model id still answers "who is replying".
    "shorten-cwd",
    "shorten-model",
    # A static session setting the user chose — it does not change while they
    # watch, so it goes before either live number.
    "effort",
    "cost",
    # The working directory goes before the context number. Its shorten step has
    # already reduced it to a basename by now, so what remains is ~7 cells of
    # "where am I" against ~9 cells of "how close is compaction" — and the second
    # is the one that predicts the operator's next action. An earlier version had
    # these the other way round, which contradicted this very ladder's rationale.
    "cwd",
    "context",
    # DEAD LAST *when it is an alarm*, mirroring the reference's
    # `flexShrink={0}` on this indicator. Two reasons it then outlives even the
    # context number. It is the narrowest segment in the band — `⊙ 3 MCP` is 7
    # cells against context's ~9 and a path's 7-plus — so dropping it buys the
    # least width of anything here. And its failure branch is an ALARM: the
    # danger-tinted glyph is the only place the band admits the agent is missing
    # tools it was configured to have, and a cramped terminal is exactly where a
    # user would otherwise conclude the tools were never configured. Kept in the
    # ladder rather than omitted from it so the very narrowest widths still get
    # one graceful aligned step before ``_render`` falls back to the truncated
    # tail.
    "mcp",
)


def _mcp_before_cwd(ladder: tuple[str, ...]) -> tuple[str, ...]:
    """``ladder`` with the mcp rung moved to just ahead of ``cwd``."""
    rungs = [step for step in ladder if step != "mcp"]
    rungs.insert(rungs.index("cwd"), "mcp")
    return tuple(rungs)


#: The ladder for a band whose MCP segment is NOT an alarm. Precomputed rather
#: than rebuilt per render: ``_render`` walks a ladder on every repaint, and the
#: spinner repaints it eight times a second.
_DROP_LADDER_QUIET: tuple[str, ...] = _mcp_before_cwd(_DROP_LADDER)


def drop_ladder(status: McpStatus) -> tuple[str, ...]:
    """Which reduction order this band uses, given its MCP state.

    The mcp rung's place is earned by the ALARM, not by the segment. Unconditional
    last place meant a healthy `⊙ 2 MCP` outranked both the working directory and
    the full model label: at 40 cells the band read `◆ model › ⊙ 2 MCP` where the
    same terminal with no MCP configured showed `◆ test/model › ⌂ local-operator`.
    A count nobody has to act on was costing the user "where am I" AND "which
    provider". So a neutral count sheds early, just ahead of the working
    directory, and a danger one still outlives everything.
    """
    return _DROP_LADDER if mcp_semantic(status) == "danger" else _DROP_LADDER_QUIET


def format_context_tokens(tokens: int) -> str:
    """Compact context estimate: ``12.4k`` / ``1.2m`` style, plain under 1k."""
    if tokens >= 1_000_000:
        return f"{tokens / 1_000_000:.1f}m"
    if tokens >= 1_000:
        return f"{tokens / 1_000:.1f}k"
    return str(tokens)


def format_window(window: int) -> str:
    """Abbreviate a context window for the denominator: ``1M``, ``200k``.

    Capital ``M`` and lower-case ``k`` are the conventional units for model
    windows, and a whole window renders without a decimal (``1M``, not
    ``1.0M``) — the denominator is a label, not a measurement.
    """
    if window >= 1_000_000:
        scaled = window / 1_000_000
        return f"{scaled:.0f}M" if scaled == int(scaled) else f"{scaled:.1f}M"
    if window >= 1_000:
        scaled = window / 1_000
        return f"{scaled:.0f}k" if scaled == int(scaled) else f"{scaled:.1f}k"
    return str(window)


def format_context_usage(tokens: int, window: int) -> str:
    """Context usage as ``49.6%/1M``; ``12.4k/—`` when the window is unknown.

    A percentage needs a denominator. When the registry has no window for the
    model, showing a bare percentage would invent one, so the spend is reported
    against an explicit unknown instead.

    The em dash rather than ``?`` because it is already this UI's glyph for an
    absent value — unknown cost renders ``$—`` and a provider with no stored
    credential renders ``—``. Two spellings of "unknown" in one row would read
    as two different states.
    """
    if tokens <= 0:
        return ""
    if window <= 0:
        return f"{format_context_tokens(tokens)}/—"
    return f"{tokens / window * 100:.1f}%/{format_window(window)}"


def format_cost(cost: float) -> str:
    """Compact dollar cost: ``$0.0021`` under a cent, ``$0.12`` above."""
    if cost < 0.01:
        return f"${cost:.4f}"
    if cost < 1.0:
        return f"${cost:.3f}"
    return f"${cost:.2f}"


def format_duration(seconds: float) -> str:
    """Active processing time: ``9s``, ``41m1s``, ``1h2m``.

    Units are dropped once they stop carrying information: past an hour the
    seconds are noise, and a whole minute renders as ``5m`` rather than
    ``5m0s``. Sub-second work renders as ``0s`` rather than vanishing, so a
    finished turn always leaves a mark.
    """
    total = int(seconds)
    if total < 60:
        return f"{total}s"
    if total < 3600:
        minutes, secs = divmod(total, 60)
        return f"{minutes}m{secs}s" if secs else f"{minutes}m"
    hours, remainder = divmod(total, 3600)
    minutes = remainder // 60
    return f"{hours}h{minutes}m" if minutes else f"{hours}h"


def format_agents(count: int) -> str:
    """``1 agent`` / ``3 agents``; empty at zero (the segment disappears).

    Zero renders as nothing rather than "0 agents": a band that permanently
    reports the absence of background work is nine characters of noise, and
    the segment appearing is itself the signal.
    """
    if count <= 0:
        return ""
    return f"{count} agent" if count == 1 else f"{count} agents"


def format_jobs(count: int) -> str:
    """``2 jobs`` — backgrounded shell work, empty at zero.

    Separate from the agent counter because they are different things an
    operator tracks: a subagent is delegated reasoning, a job is a process this
    session started and can outlive the turn. omp shows both, and collapsing
    them into one number would hide which kind is running.
    """
    if count <= 0:
        return ""
    return f"{count} job" if count == 1 else f"{count} jobs"


@dataclass(frozen=True)
class McpStatus:
    """The facts the MCP segment renders from.

    Passed as ONE value rather than four ``update`` keywords: the band's
    update signature is already at ten segments, and the count is meaningless
    without knowing whether anything was configured — splitting them invites a
    caller to set one and leave the others stale.

    ``configured`` is how many servers the config asked for, ``connected`` how
    many are actually up right now, and ``failed`` whether any of them did not
    come up. All three are read LIVE from the manager; ``configured`` alone
    decides whether the segment carries a COUNT, and ``discovery_failed``
    (below) is the one state that renders without one.
    """

    configured: int = 0
    connected: int = 0
    failed: bool = False
    #: Discovery itself failed, so there is no server list to count. Distinct
    #: from ``configured == 0``, which means "this machine does not use MCP":
    #: here the machine HAS a config and it could not be read, and a segment
    #: hidden for the same reason as an unconfigured one would report a broken
    #: setup as an absent feature.
    discovery_failed: bool = False


def format_mcp(status: McpStatus) -> str:
    """``3 MCP`` — connected SERVERS, never pluralised; empty when unconfigured.

    The count is servers rather than tools because a server is the thing that
    can drop: "31 MCP" would tell an operator nothing about the github server
    having died, which is the event this segment exists to show. The tool total
    belongs in the startup toast, where there is room to say both.

    ``MCP`` stays singular the way ``2 jobs`` does not, because it is an
    initialism naming the protocol, not a count of a noun — the reference
    renders it that way and "3 MCPs" reads as three protocols.

    Zero CONFIGURED servers renders nothing: a permanent ``⊙ 0 MCP`` on a
    machine with no ``.mcp.json`` is seven cells asserting the absence of a
    feature the user never asked for. Zero CONNECTED with servers configured
    does render, because that is a real and interesting state.

    A DISCOVERY failure renders the bare initialism with no number. The config
    layer never produced a server list on that path, so every count would be a
    fiction — ``0 MCP`` in particular would claim the machine asked for nothing,
    which is the opposite of what happened. The toast that reports the failure
    dismisses itself after ten seconds; without this the band it leaves behind
    is indistinguishable from a machine with no MCP at all.
    """
    if status.discovery_failed:
        return "MCP"
    if status.configured <= 0:
        return ""
    return f"{status.connected} MCP"


def mcp_semantic(status: McpStatus) -> str:
    """Which semantic tint the ``⊙`` glyph carries.

    An ALARM OR NOTHING. Only the failure branch gets a colour of its own:
    `danger` when a configured server did not come up, and the band's neutral
    ramp otherwise. FAILURE WINS even when other servers connected — a partial
    outcome is the dangerous one, because a `⊙ 2 MCP` that looks fine on a
    machine where a third server died costs the user a turn wondering why the
    agent cannot reach that server's tools.

    The healthy rung used to be `success` #57c785. That is 5.08 dE2000 from the
    accent #38c96a — for scale, this file's own comments reject 3.06 as
    imperceptible, and this glyph is ONE cell. The band's single accent site is
    the running indicator, so a healthy count put a second, indistinguishable
    green ten cells from the spinner and "green means a turn is live" stopped
    being true. The neutral ramp says the same three things without spending the
    accent: `muted` once something is up, `dim` while nothing is (16.66 dE2000
    apart, and both are already band vocabulary).

    Public because the startup toast paints the same lamp: two surfaces deriving
    the same state independently is how they end up disagreeing, and a toast
    saying green while the band says red is worse than either alone.
    """
    if status.failed or status.discovery_failed:
        return "danger"
    if status.connected > 0:
        return "muted"
    return "dim"


def format_cwd(cwd: str, *, short: bool) -> str:
    """The working dir as ``~/rel/path``, or just its basename when ``short``.

    Home-relative beats absolute (the ``/Users/<name>`` prefix is the same on
    every row of every session) and beats the bare basename, which is
    ambiguous the moment two checkouts share a directory name — the basename
    is what the overflow ladder falls back to, not what it starts from.
    """
    if not cwd:
        return ""
    path = Path(cwd)
    if short:
        return path.name or cwd
    try:
        return "~/" + str(path.relative_to(Path.home()))
    except ValueError:
        # Outside the home tree (/tmp, /opt, a mounted volume): the absolute
        # path is already the shortest honest rendering.
        return str(path)


def format_model_label(label: str, *, short: bool) -> str:
    """``provider/model``, or just the model id when ``short``.

    The provider is the droppable half: an operator who has switched
    providers knows which one they are on, but two providers' model ids are
    rarely confusable, so ``openrouter/moonshotai/kimi-k2`` degrades to
    ``kimi-k2`` rather than disappearing. Only the LAST path segment is kept
    — vendor-scoped ids carry two prefixes, and dropping one of them would
    leave a label that is still too long to help.
    """
    if not label:
        return ""
    if not short:
        return label
    return label.rpartition("/")[2] or label


class StatusLine:
    """Draws status segments into the bottom status band widget.

    Owns no layout row: the band widget (a ``Static`` docked at the bottom of
    the input dock) is repainted through :meth:`refresh`. Call :meth:`update`
    whenever a segment changes; call :meth:`refresh` after resizes so the
    overflow ladder follows the new width (one width model:
    ``rich.cells.cell_len``).

    Every :meth:`update` parameter is optional and ``None`` means "leave this
    segment unchanged" — callers assemble partial updates from whatever the
    event they are handling actually knows.
    """

    def __init__(self, dock: Static, *, clock: Callable[[], float] = time.monotonic) -> None:
        # A `Static`, not a bare `Widget`: the band is repainted by handing
        # it a rich renderable, which only content widgets accept.
        self._dock = dock
        # Injected so duration is testable without sleeping. Monotonic, not
        # wall clock: a duration that jumps when the system clock is adjusted
        # is worse than no duration.
        self._clock = clock
        self._model_label: str = ""
        self._effort: str = ""
        self._cwd: str = ""
        self._context_tokens: int = 0
        self._context_window: int = 0
        self._subagents: int = 0
        self._jobs: int = 0
        self._streaming: bool = False
        self._cost: str = ""
        self._conversation_name: str = ""
        self._mcp: McpStatus = McpStatus()
        # Cumulative ACTIVE processing time: the sum of turn durations, not
        # wall clock since launch. A session left open over lunch has not
        # been working for two hours, and reporting that it has makes the
        # number useless for judging what a task actually cost.
        self._active_seconds: float = 0.0
        self._turn_started_at: float | None = None
        self._spinner_index: int = 0
        self._spinner_timer = None

    # -- segment setters ----------------------------------------------------
    def update(
        self,
        *,
        model_label: str | None = None,
        effort: str | None = None,
        cwd: str | None = None,
        context_tokens: int | None = None,
        context_window: int | None = None,
        subagents: int | None = None,
        jobs: int | None = None,
        streaming: bool | None = None,
        cost: str | None = None,
        conversation_name: str | None = None,
        mcp: McpStatus | None = None,
    ) -> None:
        """Update any subset of segments and repaint the band."""
        if model_label is not None:
            self._model_label = model_label
        if effort is not None:
            self._effort = effort
        if cwd is not None:
            self._cwd = cwd
        if context_tokens is not None:
            self._context_tokens = context_tokens
        if context_window is not None:
            self._context_window = context_window
        if subagents is not None:
            self._subagents = subagents
        if jobs is not None:
            self._jobs = jobs
        if mcp is not None:
            self._mcp = mcp
        if cost is not None:
            self._cost = cost
        if conversation_name is not None:
            self._conversation_name = conversation_name
        if streaming is not None and streaming != self._streaming:
            self._streaming = streaming
            self._mark_turn_boundary(streaming)
            self._sync_spinner_timer()
        self.refresh()

    def refresh(self) -> None:
        """Rebuild the band content, truncated to the dock's inner width."""
        width = max(self._dock.size.width, 10)
        self._dock.update(self._render(width))

    def dispose(self) -> None:
        """Stop the spinner timer (idempotent)."""
        self._stop_spinner()

    # -- duration ------------------------------------------------------------
    def _mark_turn_boundary(self, streaming: bool) -> None:
        """Start or bank the active-time clock on a streaming transition.

        Only called on an actual change (``update`` guards it), so a
        redundant ``streaming=False`` from the prompt worker's ``finally``
        cannot bank the same interval twice.
        """
        if streaming:
            self._turn_started_at = self._clock()
        elif self._turn_started_at is not None:
            self._active_seconds += self._clock() - self._turn_started_at
            self._turn_started_at = None

    def _elapsed(self) -> float:
        """Banked active time plus the turn in flight, so the segment ticks
        live off the spinner's repaint rather than needing its own timer."""
        if self._turn_started_at is None:
            return self._active_seconds
        return self._active_seconds + (self._clock() - self._turn_started_at)

    # -- rendering ----------------------------------------------------------
    def _render(self, width: int) -> Text:
        dim = Style(color=theme_mod.semantic_color("dim"))
        muted = Style(color=theme_mod.semantic_color("muted"))
        # Separators must sit BELOW the things they separate or they read as
        # content: `faint` is the ramp step under `dim` and exists for
        # exactly this — the dots recede and the segments group themselves.
        seam = Style(color=theme_mod.semantic_color("faint"))
        accent = Style(color=theme_mod.semantic_color("accent"))

        dropped: set[str] = set()
        short: set[str] = set()
        # Walk the ladder until the row fits. Building the row is a handful of
        # string joins and the ladder is nine steps, so the worst case is ten
        # cheap rebuilds on a resize — far cheaper than measuring segments
        # independently and getting the separator arithmetic subtly wrong.
        for step in (None, *drop_ladder(self._mcp)):
            if step is not None:
                target = step.partition("shorten-")[2]
                (short if target else dropped).add(target or step)
            left = self._left_text(dropped, short, dim, muted, seam, accent)
            right = self._right_text(dropped, dim, seam)
            # The fit test reserves the group gap rather than asking whether the
            # composed row happens to fit. `_compose` pads with `max(1, …)`, so a
            # row could "fit" with the two groups ONE cell apart — tighter than
            # the 3-cell ` · ` separator used inside each group, which makes the
            # whole left/right architecture read as one undifferentiated run and
            # abuts a filesystem path against a percentage. Reachable by dragging
            # a window one cell at ordinary widths like 98 or 116 (D3).
            if cell_len(left.plain) + cell_len(right.plain) + _MIN_GROUP_GAP <= width:
                return self._compose(left, right, width, dim)

        # Even the irreducible row overflows. Truncate the model label rather
        # than shedding it (D7): `deepse…` still answers which model is
        # replying, whereas the ladder's old final rung dropped the label and
        # left a bare glyph on an empty tinted strip, which reads as broken
        # rather than as compressed.
        #
        # The SPINNER comes first and is never dropped, so "green means a turn is
        # live" holds at every width. Without it this path emitted glyph + label
        # for both states, making the streaming band byte-identical to the idle
        # one — the exact confusion the band's colour budget exists to prevent,
        # reintroduced at the narrow end. One cell of the label is a cheaper loss
        # than the liveness signal.
        tail = Text()
        if self._streaming:
            from local_operator.tui.shimmer import shimmer_enabled

            # Same gate the wide path uses: with animation off the band states
            # itself in words instead of leaning on a moving glyph.
            if shimmer_enabled():
                tail.append(_SPINNER_FRAMES[self._spinner_index], style=accent)
                tail.append(" ", style=dim)
        label = format_model_label(self._model_label, short=True) if self._model_label else ""
        if label:
            tail.append(f"{ICON_MODEL} ", style=dim)
            tail.append(
                truncate_cells(label, max(1, width - cell_len(tail.plain))),
                style=Style(color=theme_mod.semantic_color("fg")),
            )
        tail.truncate(width, overflow="ellipsis")
        return tail

    def _left_text(
        self,
        dropped: set[str],
        short: set[str],
        dim: Style,
        muted: Style,
        seam: Style,
        accent: Style,
    ) -> Text:
        """model › effort › cwd › mcp (+ the working indicator).

        Each segment is ``icon value``, the icon a step dimmer than its value so
        it frames the number rather than competing with it. Separators point
        RIGHT here and LEFT in the other group, so the two chevron runs aim at
        the centre gap and frame it — which is what makes a borderless band read
        as two groups instead of one long run.

        The MCP segment is the one exception to the ramp: there the GLYPH carries
        the state colour and the text stays neutral foreground. Tinting `3 MCP`
        danger would read as "the number 3 is wrong"; tinting the glyph reads as
        a status lamp beside a plain count, which is what it is.
        """
        # (icon, value, value style, icon style) — the icon style is None for
        # every segment whose glyph is pure framing, and set only where the
        # glyph itself is the signal.
        parts: list[tuple[str, str, Style, Style | None]] = []
        if self._model_label and "model" not in dropped:
            parts.append(
                (
                    ICON_MODEL,
                    format_model_label(self._model_label, short="model" in short),
                    Style(color=theme_mod.semantic_color("fg")),
                    None,
                )
            )
        if self._effort and "effort" not in dropped:
            parts.append(
                (ICON_EFFORT, self._effort, Style(color=theme_mod.semantic_color("label")), None)
            )
        if self._cwd and "cwd" not in dropped:
            rendered = format_cwd(self._cwd, short="cwd" in short)
            if rendered:
                parts.append(
                    (ICON_CWD, rendered, Style(color=theme_mod.semantic_color("signal")), None)
                )
        if "mcp" not in dropped:
            mcp = format_mcp(self._mcp)
            if mcp:
                parts.append(
                    (
                        ICON_MCP,
                        mcp,
                        Style(color=theme_mod.semantic_color("fg")),
                        Style(color=theme_mod.semantic_color(mcp_semantic(self._mcp))),
                    )
                )

        left = Text()
        for index, (icon, text, style, icon_style) in enumerate(parts):
            if index:
                left.append(f" {_SEP_LEFT} ", style=seam)
            left.append(f"{icon} ", style=icon_style or dim)
            left.append(text, style=style)
        if self._streaming:
            # The aggregate working LINE (WorkingBlock) carries the shimmer; the
            # band keeps a quiet activity glyph so a still frame still reads
            # "live". With shimmer off that line is static too, so the band
            # spells the state out rather than relying on a glyph the eye may
            # read as decoration.
            from local_operator.tui.shimmer import shimmer_enabled

            if parts:
                left.append(f" {_SEP_LEFT} ", style=seam)
            # ACCENT: the band's running indicator, and the accent budget's whole
            # point is that seeing green means a turn is live. The trailing
            # " working" word stays dim so the colour is the signal and the word
            # is only the caption.
            left.append(_SPINNER_FRAMES[self._spinner_index], style=accent)
            if not shimmer_enabled():
                left.append(" working", style=dim)
        return left

    def _right_text(self, dropped: set[str], dim: Style, seam: Style) -> Text:
        """agents ‹ jobs ‹ context ‹ cost ‹ duration ‹ name.

        Colour groups by KIND rather than giving every field its own hue, which
        would be a rainbow: counters share `label`, the two numbers an operator
        acts on take `signal` (context) and `warning` (cost, because spend is a
        caution), and the least volatile fields stay neutral. Green is not used
        at all — it belongs to the running indicator.
        """
        parts: list[tuple[str, str, Style]] = []
        if "subagents" not in dropped:
            agents = format_agents(self._subagents)
            if agents:
                parts.append((ICON_AGENTS, agents, Style(color=theme_mod.semantic_color("label"))))
        if "jobs" not in dropped:
            jobs = format_jobs(self._jobs)
            if jobs:
                parts.append((ICON_JOBS, jobs, Style(color=theme_mod.semantic_color("label"))))
        if "context" not in dropped:
            usage = format_context_usage(self._context_tokens, self._context_window)
            if usage:
                parts.append((ICON_CONTEXT, usage, Style(color=theme_mod.semantic_color("signal"))))
        if self._cost and "cost" not in dropped:
            parts.append((ICON_COST, self._cost, Style(color=theme_mod.semantic_color("warning"))))
        if "duration" not in dropped:
            elapsed = self._elapsed()
            # Zero means "nothing has run yet" — an idle band should not claim a
            # 0s task. Any real turn banks at least a few ms.
            if elapsed > 0:
                parts.append((ICON_DURATION, format_duration(elapsed), dim))
        if self._conversation_name and "name" not in dropped:
            parts.append(
                ("", self._conversation_name, Style(color=theme_mod.semantic_color("muted")))
            )

        right = Text()
        for index, (icon, text, style) in enumerate(parts):
            if index:
                right.append(f" {_SEP_RIGHT} ", style=seam)
            if icon:
                right.append(f"{icon} ", style=dim)
            right.append(text, style=style)
        return right

    def _compose(self, left: Text, right: Text, width: int, dim: Style) -> Text:
        """Left group, filler, right group — right-aligned to the band edge."""
        gap = max(_MIN_GROUP_GAP, width - cell_len(left.plain) - cell_len(right.plain))
        row = Text()
        row.append_text(left)
        row.append(" " * gap, style=dim)
        row.append_text(right)
        return row

    def render_text(self, width: int) -> Text:
        """Public render entry (tests): segments joined, fitted to width."""
        return self._render(width)

    # -- spinner ------------------------------------------------------------
    def _sync_spinner_timer(self) -> None:
        if self._streaming and self._spinner_timer is None:
            self._spinner_timer = self._dock.set_interval(
                _SPINNER_INTERVAL_S, self._advance_spinner
            )
        elif not self._streaming and self._spinner_timer is not None:
            self._stop_spinner()

    def _advance_spinner(self) -> None:
        self._spinner_index = (self._spinner_index + 1) % len(_SPINNER_FRAMES)
        self.refresh()

    def _stop_spinner(self) -> None:
        if self._spinner_timer is not None:
            self._spinner_timer.stop()
            self._spinner_timer = None
