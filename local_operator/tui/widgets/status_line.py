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

from local_operator.model.naming import model_label as model_label_forms
from local_operator.tui import theme as theme_mod
from local_operator.tui.widgets.tool_card import format_duration, truncate_cells

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
#: Auto-approve indicator. Deliberately NOT one of the geometric segment icons
#: above: this one is an ALARM, not a reading, and it reuses the app-wide
#: warning glyph so it reads the same as a warning notice in the transcript.
ICON_APPROVALS = "!"

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
    # Second-to-last, just ahead of mcp. This rung only EXISTS while the
    # tool-approval gate has been disarmed for the session, so it is an alarm by
    # the same argument mcp's failure branch is one — but it is the WIDEST
    # segment in the band (`! auto-approve` is 14 cells against `⊙ 3 MCP`'s 7),
    # and this ladder sheds by what a drop BUYS. Dropping the widest alarm first
    # is what lets the narrow one survive to the very last step, so a cramped
    # terminal keeps saying something rather than falling straight to the
    # truncated tail. The mode is still not silent when it goes: `/approvals`
    # reports it, and the notice that latched it is in the transcript.
    "approvals",
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


#: Rungs whose painted width is not knowable from the ladder. ``cwd`` is as wide
#: as the user's path — a basename alone runs from 3 cells to 30 — so it can
#: never be relied on as the thing that still fits at the narrowest width.
_UNBOUNDED_RUNGS = frozenset({"cwd"})


def _narrowest_survivor_last(rungs: list[str]) -> list[str]:
    """Re-seat ``approvals`` so it is not the last rung standing — usually.

    Every ``_x_before_cwd`` helper below promotes one rung, and promoting a rung
    leaves whatever FOLLOWED it at the end of the ladder. In this ladder that is
    reliably ``approvals`` — the 14-cell segment the authored order sheds FIRST,
    precisely because dropping it buys the most width. Left last it outlives the
    narrow segments it was supposed to make room for, inverting the argument.

    Factored out rather than repeated: it was written inline for the mcp move,
    and the next promotion (context) silently did not re-apply it.

    **The exception is load-bearing, not a let-out.** The repair is refused when
    it would strand an UNBOUNDED rung last, because the render walk is monotone:
    it sheds down the ladder until the row fits and can never add a segment
    back. Ending on ``cwd`` therefore means the band sheds the armed
    ``! auto-approve`` alarm to make room for a path that then does not fit
    either, and paints neither.

    The decisive argument is PARITY WITH MAIN, not blank cells. Main's boot band
    carries no estimate, so it uses the quiet ladder and keeps the alarm; a band
    that ends on ``cwd`` disagrees with main about which of {path, alarm}
    survives at 172 of the widths swept (20-95 cells x 7 basenames), and this
    rule brings that to zero. Blank cells are the symptom and they do not argue
    it cleanly either way — from 47 cells up the swap trades an ink-heavy path
    for a shorter alarm, which the blank-cell count scores as worse and main
    scores as correct.

    Concretely, with a 24-character basename and the gate disarmed: the alarm
    first fits at 36 cells, and from 36 to 46 the previous ordering painted
    ``◆ kimi-k2-thinking`` alone. A 14-cell alarm that always fits beats an
    unbounded path that may not, and the alarm is the only place the band admits
    the approval gate is disarmed.

    So the rule is really "the last survivor must be BOUNDED, and should be the
    narrowest such rung". Those agree everywhere except the quiet estimate
    ladder, where the only other candidate is the path.
    """
    if rungs[-1] != "approvals":
        return rungs
    repaired = [step for step in rungs if step != "approvals"]
    repaired.insert(len(repaired) - 1, "approvals")
    if repaired[-1] in _UNBOUNDED_RUNGS:
        return rungs
    return repaired


def _mcp_before_cwd(ladder: tuple[str, ...]) -> tuple[str, ...]:
    """``ladder`` with the mcp rung moved to just ahead of ``cwd``.

    A healthy `⊙ 2 MCP` is a nicety, and last place had it outranking both the
    working directory and the full model label. The rung moves rather than the
    render site special-casing the segment.
    """
    rungs = [step for step in ladder if step != "mcp"]
    rungs.insert(rungs.index("cwd"), "mcp")
    return tuple(_narrowest_survivor_last(rungs))


#: The ladder for a band whose MCP segment is NOT an alarm. Precomputed rather
#: than rebuilt per render: ``_render`` walks a ladder on every repaint, and the
#: spinner repaints it eight times a second.
_DROP_LADDER_QUIET: tuple[str, ...] = _mcp_before_cwd(_DROP_LADDER)


def _context_before_cwd(ladder: tuple[str, ...]) -> tuple[str, ...]:
    """``ladder`` with the context rung moved to just ahead of ``cwd``.

    The full order puts cwd first because the context number "predicts the
    operator's next action". That is true of a number the model reported, and
    false of the boot ESTIMATE: before a single turn, nothing has been spent,
    nothing is approaching compaction, and the figure is a static property of a
    session that has not done anything yet. Meanwhile "where am I" is exactly
    as load-bearing as it always was — more so, since a fresh session is when a
    user is most likely to be checking they opened the right directory.

    Shedding the estimate first is therefore the same trade ``_mcp_before_cwd``
    exists to make: rank by what the segment is worth RIGHT NOW, not by what
    its slot is usually worth. It inherits the same tail rule for the same
    reason — promoting context out of last place in the quiet ladder is exactly
    what left ``approvals`` stranded there.
    """
    rungs = [step for step in ladder if step != "context"]
    rungs.insert(rungs.index("cwd"), "context")
    return tuple(_narrowest_survivor_last(rungs))


#: Estimate variants of both ladders, precomputed for the same reason.
_DROP_LADDER_ESTIMATE: tuple[str, ...] = _context_before_cwd(_DROP_LADDER)
_DROP_LADDER_QUIET_ESTIMATE: tuple[str, ...] = _context_before_cwd(_DROP_LADDER_QUIET)


def drop_ladder(status: McpStatus, *, context_estimated: bool = False) -> tuple[str, ...]:
    """Which reduction order this band uses, given its MCP state.

    The mcp rung's place is earned by the ALARM, not by the segment. Unconditional
    last place meant a healthy `⊙ 2 MCP` outranked both the working directory and
    the full model label: at 40 cells the band read `◆ model › ⊙ 2 MCP` where the
    same terminal with no MCP configured showed `◆ test/model › ⌂ local-operator`.
    A healthy count is a nicety and a failed one is the only warning the band
    carries, so the rung moves rather than the segment being special-cased at the
    render site.

    ``context_estimated`` demotes the context rung for the same kind of reason —
    see :func:`_context_before_cwd`.
    """
    if mcp_semantic(status) == "danger":
        return _DROP_LADDER_ESTIMATE if context_estimated else _DROP_LADDER
    return _DROP_LADDER_QUIET_ESTIMATE if context_estimated else _DROP_LADDER_QUIET


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


def format_model_label(label: str, *, short: bool, name: str = "") -> str:
    """A model's human name, or its selector when no name can be trusted.

    ``label`` is the selector the rest of the app identifies a model by
    (``anthropic/claude-opus-5``); ``name`` is the resolved ``ModelInfo.name``
    when the caller has one, and is what lets an aggregator id no registry row
    covers still read as ``MoonshotAI: Kimi K2``. Callers without it still get
    the curated name, which is every direct provider.

    Both the choice of name and the ``short`` narrowing are
    ``model/naming.py``'s: it refuses a name that two models answer to, so the
    23 cells of ``anthropic/claude-opus-5`` become the 13 of ``Claude Opus 5``
    without the band ever printing a string that fails to say which model is
    replying. Where it refuses, the selector comes back and ``short`` keeps only
    the last path segment — the behaviour this segment has always had, now
    reached only when a name would have been a guess.
    """
    if not label:
        return ""
    provider, _, model_id = label.partition("/")
    forms = model_label_forms(provider, model_id, name)
    return forms.compact if short else forms.full


@dataclass(frozen=True)
class SubagentBand:
    """The CHILD's readings, shown in place of the session's own.

    While the full-page subagent view is up, the band four rows under it was
    still describing the PARENT — its model, its context, its spend — over a
    frame whose entire subject is a different session. A child frequently runs
    a different model than its parent (``run_subagent`` takes a ``model_spec``
    override), so that was not merely redundant, it was wrong.

    An OVERLAY rather than a save-and-restore of the band's own fields,
    because the parent does not stop while the page is open: its turn keeps
    ending, and every ``on_turn_ended`` writes a fresh cost and context into
    those fields. Saving them on the way in would hand back a snapshot that
    went stale in the reader's hand. Shadowed, the parent's numbers stay live
    underneath and are revealed intact the moment the overlay is dropped.

    Every field is OMITTABLE and empty means omitted, never zero: a child that
    has reported no usage has no context reading and no spend, and the band's
    own segments already disappear on exactly those emptiness tests. Printing
    the parent's figure in a child's frame is the one thing this class exists
    to prevent, so there is no field here that can fall back to it.
    """

    #: Never empty in practice: a child with no recorded model of its own is
    #: running the parent's, so the caller resolves that before constructing
    #: this. The band never drops the model segment, and "which model is
    #: replying" has an answer here even when nothing else does.
    model_label: str = ""
    #: The child's own name, which REPLACES the parent's running-agent counter
    #: while the overlay is up. Without it the band interleaves three owners
    #: with no mark — the model, context, cost and duration are the child's,
    #: the cwd and MCP are shared, and the parent's `◍ 2 agents` sat between
    #: them. Naming the child at the head of the right-hand group makes
    #: everything after it belong to one session, and the count it displaces
    #: is redundant to a reader who is already inside one of those agents and
    #: has the whole list four rows below.
    label: str = ""
    context_tokens: int = 0
    context_window: int = 0
    #: Already formatted, because the vocabulary for "billed but unpriceable"
    #: (``$—``) belongs to the caller that knows the difference between a
    #: child that spent nothing and one nobody can price.
    cost: str = ""
    #: The child's own age. ``None`` leaves the segment off rather than
    #: reporting the parent's cumulative active time under a child's title.
    duration: float | None = None


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
        # The resolved ``ModelInfo.name`` for the selector above, "" when the
        # host could not supply one. Kept beside the selector rather than
        # replacing it because the selector is still what the segment falls back
        # to, and because a caller that knows neither (the boot frame, before a
        # session exists) must not have to invent one.
        self._model_name: str = ""
        self._effort: str = ""
        self._cwd: str = ""
        self._context_tokens: int = 0
        # Provenance of the reading above: True while it is the host's LOCAL
        # estimate of what is already loaded, False once a provider's exact
        # prompt_tokens has replaced it. Lives here rather than on the app
        # because both readers need it — the app for precedence, the band for
        # how hard to fight to keep the segment on screen.
        self._context_is_estimate: bool = False
        self._context_window: int = 0
        self._subagents: int = 0
        self._jobs: int = 0
        self._streaming: bool = False
        self._cost: str = ""
        self._conversation_name: str = ""
        self._mcp: McpStatus = McpStatus()
        # Which segments the drop ladder shed on the LAST render. Every segment
        # is dropped until something has been rendered, which is the honest
        # starting state: nothing has been shown yet.
        self._dropped: frozenset[str] = frozenset(_DROP_LADDER)
        # True once the user has answered a tool-approval prompt with "allow
        # all": a session-wide mode with no persistent indicator is how a
        # disarmed gate gets forgotten about.
        self._approvals_auto: bool = False
        # Cumulative ACTIVE processing time: the sum of turn durations, not
        # wall clock since launch. A session left open over lunch has not
        # been working for two hours, and reporting that it has makes the
        # number useless for judging what a task actually cost.
        self._active_seconds: float = 0.0
        self._turn_started_at: float | None = None
        self._spinner_index: int = 0
        self._spinner_timer = None
        #: The child's readings, shadowing the session's own while the
        #: full-page subagent view is up. ``None`` is the ordinary band.
        self._subagent: SubagentBand | None = None

    @property
    def context_tokens(self) -> int:
        """Tokens currently attributed to the context segment.

        Read by the boot-time estimator, which must not overwrite a provider's
        exact ``prompt_tokens`` if a turn beat it to the finish.
        """
        return self._context_tokens

    @property
    def context_is_estimate(self) -> bool:
        """Whether :attr:`context_tokens` is a local estimate, not the wire."""
        return self._context_is_estimate

    def set_subagent(self, band: SubagentBand | None) -> None:
        """Describe a CHILD session instead of this one, or stop (``None``).

        Idempotent and cheap enough to call from the 1 Hz refresh that keeps
        the open page live: an unchanged overlay repaints nothing.
        """
        if band == self._subagent:
            return
        self._subagent = band
        self.refresh()

    # -- segment setters ----------------------------------------------------
    def update(
        self,
        *,
        model_label: str | None = None,
        model_name: str | None = None,
        effort: str | None = None,
        cwd: str | None = None,
        context_tokens: int | None = None,
        context_is_estimate: bool | None = None,
        context_window: int | None = None,
        subagents: int | None = None,
        jobs: int | None = None,
        streaming: bool | None = None,
        cost: str | None = None,
        conversation_name: str | None = None,
        mcp: McpStatus | None = None,
        approvals_auto: bool | None = None,
    ) -> None:
        """Update any subset of segments and repaint the band."""
        if model_label is not None:
            self._model_label = model_label
        if model_name is not None:
            self._model_name = model_name
        if effort is not None:
            self._effort = effort
        if cwd is not None:
            self._cwd = cwd
        if context_tokens is not None:
            self._context_tokens = context_tokens
        if context_is_estimate is not None:
            self._context_is_estimate = context_is_estimate
        if context_window is not None:
            self._context_window = context_window
        if subagents is not None:
            self._subagents = subagents
        if jobs is not None:
            self._jobs = jobs
        if mcp is not None:
            self._mcp = mcp
        if approvals_auto is not None:
            self._approvals_auto = approvals_auto
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

    def is_showing(self, segment: str) -> bool:
        """Whether ``segment`` survived the drop ladder on the last render.

        Exists because "the band said it" is not the same claim as "the band was
        told it": at ordinary widths the ladder sheds segments, and a caller that
        treats the band as its only receipt has to know when the band could not
        deliver one. Reads the last render rather than re-measuring, so the
        answer is about the row the user is looking at.
        """
        return segment not in self._dropped

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
        live off the spinner's repaint rather than needing its own timer.

        Under a subagent overlay this is the CHILD's age instead — wall clock
        since it was launched, which is the only duration a child has. The
        parent's banked active time is not a fact about the child and would be
        a bigger number than its whole life on a frame titled with its name.
        """
        if self._subagent is not None:
            return self._subagent.duration or 0.0
        if self._turn_started_at is None:
            return self._active_seconds
        return self._active_seconds + (self._clock() - self._turn_started_at)

    # -- overlay ---------------------------------------------------------------
    # Four segments answer from the overlay when one is set, and the rest are
    # left alone deliberately: cwd, MCP, effort and the approval alarm describe
    # the HOST, which the child shares — it runs in the same directory, on the
    # parent's live MCP surface, under the same approval policy. Re-pointing
    # those would say something changed when nothing did. The counters (agents,
    # jobs) are the parent's ledger, and that ledger is what the page is a
    # window onto.
    def _shown_model_label(self) -> str:
        if self._subagent is not None:
            return self._subagent.model_label
        return self._model_label

    def _shown_model_name(self) -> str:
        """The resolved name for whichever model the segment is describing.

        Inherited under an overlay ONLY when the overlay names the same model.
        Returning nothing unconditionally looked like the safe reading of the
        overlay's rule and was not: ``job_stats`` documents an inherited spec as
        the NORMAL path and ``harness/subagent.py`` sets the child's label from a
        child built on the parent's spec, so the common case is a child on the
        parent's own model — and then the same model was spelled two ways seconds
        apart, ``MoonshotAI: Kimi K2`` with the page closed and
        ``openrouter/moonshotai/kimi-k2`` with it open, because the fallback
        lookup only reaches curated rows. A user reads that as a model switch.
        That is D10, the defect this app has already shipped once between its
        splash and its band.

        A DIFFERENT selector still gets nothing, which is the rule that actually
        matters: the parent's name must never be attributed to another model. The
        child then falls back to its curated name, exactly as any caller with no
        resolved metadata does.
        """
        if self._subagent is not None:
            if self._subagent.model_label == self._model_label:
                return self._model_name
            return ""
        return self._model_name

    def _shown_context(self) -> tuple[int, int]:
        """``(tokens, window)`` for the context segment."""
        if self._subagent is not None:
            return self._subagent.context_tokens, self._subagent.context_window
        return self._context_tokens, self._context_window

    def _shown_cost(self) -> str:
        if self._subagent is not None:
            return self._subagent.cost
        return self._cost

    def _shown_context_is_estimate(self) -> bool:
        """A child's reading is always the provider's own, never an estimate.

        Which matters beyond a tone: the drop ladder fights harder to keep an
        ESTIMATED reading on screen, and inheriting the parent's estimate flag
        would have the band making that trade on behalf of a number that has
        nothing to do with it.
        """
        return self._subagent is None and self._context_is_estimate

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
        for step in (
            None,
            *drop_ladder(self._mcp, context_estimated=self._shown_context_is_estimate()),
        ):
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
            #
            # Reserved only when there IS a right group. Charged unconditionally,
            # the gap bought separation from nothing: the two-group layout was
            # abandoned four columns early and the spinner hopped across the model
            # label at a width where the row it reflowed to was no narrower.
            gap = _MIN_GROUP_GAP if right.plain else 0
            if cell_len(left.plain) + cell_len(right.plain) + gap <= width:
                # Recorded so a caller can ask whether the band actually SAID
                # what it was asked to show. The effort segment needs it: it is
                # the one segment a keystroke changes, so when the ladder has
                # shed it the app owes the user a receipt somewhere else (see
                # ``OperatorApp.action_cycle_effort``). Set on the fitting pass
                # only — the intermediate rungs are attempts, not what shipped.
                self._dropped = frozenset(dropped)
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
        model_label = self._shown_model_label()
        label = (
            format_model_label(model_label, short=True, name=self._shown_model_name())
            if model_label
            else ""
        )
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
        model_label = self._shown_model_label()
        if model_label and "model" not in dropped:
            parts.append(
                (
                    ICON_MODEL,
                    format_model_label(
                        model_label,
                        short="model" in short,
                        name=self._shown_model_name(),
                    ),
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
        if self._subagent is not None and self._subagent.label:
            # Never dropped: it is what says whose numbers follow it, and a
            # band that sheds the owner while keeping the figures is worse
            # than one that sheds a figure.
            parts.append(
                (
                    ICON_AGENTS,
                    truncate_cells(self._subagent.label, 24),
                    Style(color=theme_mod.semantic_color("label")),
                )
            )
        elif "subagents" not in dropped:
            agents = format_agents(self._subagents)
            if agents:
                parts.append((ICON_AGENTS, agents, Style(color=theme_mod.semantic_color("label"))))
        if "jobs" not in dropped:
            jobs = format_jobs(self._jobs)
            if jobs:
                parts.append((ICON_JOBS, jobs, Style(color=theme_mod.semantic_color("label"))))
        if "context" not in dropped:
            usage = format_context_usage(*self._shown_context())
            if usage:
                parts.append((ICON_CONTEXT, usage, Style(color=theme_mod.semantic_color("signal"))))
        cost = self._shown_cost()
        if cost and "cost" not in dropped:
            parts.append((ICON_COST, cost, Style(color=theme_mod.semantic_color("warning"))))
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
        if self._approvals_auto and "approvals" not in dropped:
            # LAST, so its right edge is the band's right edge: everything else
            # here is right-ALIGNED as a group, which means a segment placed first
            # slides left every time a sibling appears (measured: column 86 -> 74
            # -> 64 -> 51 at a fixed 100 cells). An alarm that moves is an alarm
            # the eye has to find.
            #
            # The glyph rides INSIDE the styled text rather than in the icon slot,
            # because the loop below paints icons `dim` — which made the one alarm
            # in the band its quietest mark (4.18:1, against the same `!` at
            # 9.4:1 in the transcript).
            parts.append(
                (
                    "",
                    f"{ICON_APPROVALS} auto-approve",
                    Style(color=theme_mod.semantic_color("warning"), bold=True),
                )
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
        """Left group, filler, right group — right-aligned to the band edge.

        With no right group there is nothing to align and nothing to separate, so
        the filler is not emitted at all. Padding unconditionally pushed a row
        that exactly fitted its frame `_MIN_GROUP_GAP` cells past it, on trailing
        blanks that aligned nothing.
        """
        if not right.plain:
            return left
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
