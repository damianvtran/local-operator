"""Tool execution card — ONE LINE PER ACTION, rendered as a FILLED card.

The character refinement (docs/REWRITE.md §D): tool calls are subtle
background-filled cards — one elevation step brighter than the ground
(kit ``surface`` on ``bg``), full-width single rows with 1-cell inner
padding, NO border. Elevation is a background step, never a shadow or line.

Row anatomy — one COLUMN per field, so a ledger is scanned down, not read
across::

     bash     pytest -q                                     ✓  0.4s
     edit     tui/theme.py                           +12 -3 ✓  0.1s
     bash     false                          exit status 1 ✗  0.2s
     grep     needle                           interrupted ⊘  5.0s

- a per-TOOL icon (``local_operator.tui.glyphs``) leads the row: shape
  before name, so the one ``edit`` in a run of ``read``s is found without
  parsing ten words. It doubles as the QUIET STATIC running marker (D25 —
  the shimmer rides the aggregate working line, not individual rows),
  painted in the accent while running so a still frame reads "live" (D26
  fallback). Exactly one cell wide, enforced in ``glyphs`` against the same
  ``cell_len`` this row's arithmetic uses.
- tool NAME ljust'd into an 8-cell column spine (D7). The column paid for
  the icon: the glyph now carries identity at a glance, so the name needs
  less room than it did when it was the only thing telling one row from the
  next. The name and the summary each carry a two-step tint ramp: while the
  tool is LIVE the name is the string green and the summary is ``muted``;
  once it settles both drop one step (``muted``/``dim``) so the running row
  is always the brightest thing in the transcript.
- command/summary ellipsized to the remaining budget, with absolute paths
  compacted against the cwd/home so a deep path does not eat the row
- diff counters for file-mutating tools: ``+N`` in the success tint and
  ``-N`` in the danger tint, rendered ONLY when the tool result actually
  reported them (an unknown count renders nothing — never ``+0 -0``)
- status right-aligned (D6): EMPTY while running — no trailing glyph, the
  column stays clear until the duration lands (D28); the glyph carries the
  outcome ink and the duration rides beside it as a second run — ``✓``
  colored ``success`` with a dim duration (narrowing D12: see the rationale
  at ``_outcome_runs`` and
  ``bindings.BY_ELEMENT["tool.status.success_glyph"].note``); ``✗ error``
  danger with the duration dim as a second run (D13); ``⊘ interrupted`` dim
  when the turn ended before completion (TUI-019). The duration is right-justified
  into a fixed :data:`DURATION_COL` so the outcome glyph sits in a STABLE
  column — otherwise ``✓ 0.4s`` and ``✓ 12.3s`` put the glyph one cell
  apart, and the pass/fail scan that right-alignment exists to serve has to
  hunt for the answer.
- the reason in front of the glyph shortens differently per state. An error
  message is CONTENT and truncates — ``internal error: worker di…`` still
  names the failure. ``interrupted`` is a constant restating ``⊘``, so it is
  all-or-nothing: truncated to ``inte…`` it was byte-identical to a real
  failure truncated to ``inte…``, in the leftmost and longest column, which
  is the one the eye reaches first. Below the width that holds it whole the
  word goes and the glyph carries the state alone.
- the outcome glyph is what a still, COLOURLESS frame reads: ``✓``/``✗``/
  ``⊘`` separate success, failure and interruption without a single colour,
  and their absence is what says "still running". Tint is a second channel
  on top of the glyph, never the only one.
- an ``⟨expand⟩`` hint trailing the summary — the whole ROW is the click
  target AND a focus stop, and activating it (mouse click, or Enter/Space
  with the row focused) reveals the tool's full output indented beneath the
  summary; activating again collapses back to exactly one row. The hint
  appears while the pointer is on the row or the row holds focus, and flips
  to ``⟨collapse⟩`` when open, so a settled transcript reads as content
  rather than as a wall of controls.
- a row with NOTHING to reveal answers anyway: activating it flashes
  :data:`NO_OUTPUT_NOTICE` (or :data:`RUNNING_NOTICE`) in the same slot for
  :data:`NOTICE_SECONDS`. Silence was read in the field as the app being
  broken — "when I click to expand these lines, nothing happens" — and an
  affordance that sometimes does nothing has to say which time this is.
- the two share a slot but NOT a priority. The hint is an offer: chrome,
  dropped at the D8 summary floor. The notice is the answer to a keystroke
  the user just pressed, so it outranks the summary and shortens along
  :data:`NOTICE_LADDER` — phrase, then ``⟨∅⟩``/``⟨⋯⟩`` — rather than
  vanishing. Shedding them together is how the reported bug came back at 46
  columns, where activating an inert row repainted identical bytes.

State also reaches the ground: the card's background is ``raised`` while
running, ``surface`` once it settles, and the warm ``tint-danger`` ground
when it failed. Outcome is legible from the row's fill alone, at a glance,
without reading a single glyph.

Widths measured through ``rich.cells.cell_len`` only (one width model).
"""

from __future__ import annotations

import json
import os
import re
import time
from collections.abc import Mapping, Sequence
from typing import Any

from rich.cells import cell_len
from rich.style import Style
from rich.text import Text
from textual.timer import Timer

from local_operator.ansi import strip_control_sequences
from local_operator.tui import bindings
from local_operator.tui.glyphs import display_name, tool_icon
from local_operator.tui.widgets.transcript import (
    TOOL_NAME_COL,
    TOOL_NAME_COL_MAX,
    ExpandableActionBlock,
    TranscriptView,
    wrap_cells,
)

#: Control-sequence stripping lives in `local_operator.ansi` because the
#: headless renderer needs the identical behaviour and must not import a
#: Textual widget module to get it. Aliased here so the call sites read local.
_strip_control_sequences = strip_control_sequences


#: Outcome glyphs. Plain unicode, NOT Nerd Font: these three are the only
#: thing distinguishing success from failure from interruption in a still,
#: colourless frame, so they may not depend on the user having a patched
#: font. The per-tool ICON at the head of the row may (see `tui.glyphs`) —
#: a wrench where a magnifier belongs loses a scanning aid; a missing ✗
#: loses the answer.
ICON_SUCCESS = "✓"
ICON_ERROR = "✗"
ICON_INTERRUPTED = "⊘"
#: Expansion affordance trailing the summary. Both spellings are the same
#: click target; only the label flips so the row always says what a click does.
EXPAND_HINT = "⟨expand⟩"
COLLAPSE_HINT = "⟨collapse⟩"

#: Answers given in the hint slot when the affordance has nothing to open.
#: An expander that silently ignores half its activations is indistinguishable
#: from a frozen app, which is precisely how it was reported.
#:
#: Bracketed in the SAME idiom as the expand affordance they stand in for.
#: Bare, they read as summary text — `todo     a no output` is one space and
#: one colour step from the argument beside it, and with no colour at all it
#: is just `a no output`. This slot is the direct remedy for "nothing
#: happens when I click", so it has to be the least ambiguous thing on the
#: row, and the app already owns a bracket for "this is chrome, not content".
NO_OUTPUT_NOTICE = "⟨no output⟩"
RUNNING_NOTICE = "⟨still running⟩"

#: The same two answers at three cells, for a row too narrow to spell them.
#: The feedback has to survive FURTHER DOWN the width ladder than the expand
#: affordance does, and the reason is the whole point of the slot: the offer
#: is decoration a tight row can drop, but an activation that leaves a
#: byte-identical frame IS the reported bug ("when I click to expand these
#: lines, nothing happens") reappearing at 46 columns. So the phrase shortens
#: to its glyph rather than vanishing: ∅ for "there is nothing here", ⋯ for
#: "not yet". Brackets are kept at every rung so the slot never reads as
#: content, and both inner glyphs are measured single-width below.
TERSE_NO_OUTPUT_NOTICE = "⟨∅⟩"
TERSE_RUNNING_NOTICE = "⟨⋯⟩"
#: Full phrase -> glyph -> nothing, per notice. The row walks this in order
#: and takes the first rung that fits.
NOTICE_LADDER: dict[str, tuple[str, ...]] = {
    NO_OUTPUT_NOTICE: (NO_OUTPUT_NOTICE, TERSE_NO_OUTPUT_NOTICE),
    RUNNING_NOTICE: (RUNNING_NOTICE, TERSE_RUNNING_NOTICE),
}
#: How long that answer stays on the row. Long enough to read at a glance,
#: short enough that it is gone before the eye returns — it is feedback for
#: a keystroke, not a state the row is in.
NOTICE_SECONDS = 2.0

#: Tool-name column FLOOR (D7: a spine for the eye to scan names). Eight, not
#: ten: the icon column now carries identity, and the two cells it costs come
#: out of the name rather than out of the summary. The transcript owns the
#: shared value and may widen it — see :attr:`TranscriptView.tool_name_col`.
#: Tool -> ledger category, mirroring `glyphs.PLAIN_TOOL_ICONS`' own
#: shell/read/mutate/search/meta grouping rather than inventing a second
#: taxonomy. The axis is what the call DID to the machine, which is the
#: question a ledger is scanned for: reading is safe, mutating is not, and
#: executing is the one you re-read before trusting.
#:
#: Anything absent — an `mcp__*` call, a new builtin — falls through to
#: `tool.row.name_settled`, the neutral this column has always used. An
#: unclassified tool is quiet, never mis-filed.
_TOOL_CATEGORY: dict[str, str] = {
    "read": "tool.row.name_read",
    "glob": "tool.row.name_read",
    "grep": "tool.row.name_read",
    "web_fetch": "tool.row.name_read",
    "web_search": "tool.row.name_read",
    "browser": "tool.row.name_read",
    "list_variables": "tool.row.name_read",
    "read_variable": "tool.row.name_read",
    "write": "tool.row.name_mutate",
    "edit": "tool.row.name_mutate",
    "bash": "tool.row.name_exec",
    "eval": "tool.row.name_exec",
    "task": "tool.row.name_meta",
    "agent": "tool.row.name_meta",
    "hub": "tool.row.name_meta",
    "todo": "tool.row.name_meta",
    "send": "tool.row.name_meta",
    "wake": "tool.row.name_meta",
    "ask": "tool.row.name_meta",
}


def _category_element(tool_name: str) -> str:
    """The binding id for ``tool_name``'s settled name span.

    Case-insensitive because ``tool_name`` is MODEL-controlled — a provider
    that echoes ``Bash`` back must land in the same category as ``bash``,
    exactly as :func:`glyphs.tool_icon` already reasons about its own lookup.

    The map stores WHOLE element ids rather than a category name spliced into
    an f-string. The f-string version left `tool.row.name_` as the only
    literal in the source, so the binding-table coverage gate — which reads
    this module's AST to prove every element it paints is declared — saw a
    prefix that matches no binding and could no longer verify the five that
    are. A gate that cannot see the call sites is not a gate.
    """
    return _TOOL_CATEGORY.get(tool_name.strip().lower(), "tool.row.name_settled")


NAME_COL = TOOL_NAME_COL
#: The ceiling that widening respects. Past roughly this width the eye stops
#: scanning a column of names and starts reading a list of them.
NAME_COL_MAX = TOOL_NAME_COL_MAX
#: Below this ROW width the column never grows: at 60 columns the summary needs
#: those cells more than the name does. Measured against the card's own inner
#: width — the value `_build_row` works in, already reduced by the transcript's
#: padding — not the terminal's, so the growth engages at a frame around six
#: cells wider. Named for what it is compared against, because a threshold that
#: does not mean what it says is what makes the next measurement disagree.
NAME_GROWTH_MIN_ROW = 70
#: Right-justification width for the duration, so the outcome glyph in front
#: of it lands in the same column whether the tool took 0.4s or 12.3s. Five
#: cells covers every duration the format produces up to ``9999s``.
DURATION_COL = 5
#: Minimum summary budget before we drop the expand hint (D8 floor).
_SUMMARY_FLOOR = 16
#: Indent of the expanded output block, aligned under the tool name column.
OUTPUT_INDENT = 2
#: Expanded output is capped: a 20k-line bash dump would otherwise turn the
#: transcript into a scroll trap. The head is kept (it carries the command's
#: framing) and the remainder is announced on a dim marker row.
EXPAND_MAX_LINES = 40
#: Per-ARGUMENT cap in the expansion. Much tighter than the output cap because
#: a payload argument is unbounded by design — `write` carries a whole file in
#: `content` — and the block exists to answer "what was this call", which a
#: dozen rows settles. A truncated command is the one thing it must never be,
#: and no realistic command reaches this.
INPUT_MAX_LINES = 12
#: Rows the LIVE view spends on a still-running tool's output. Tighter than
#: :data:`EXPAND_MAX_LINES` on purpose: this block is on screen while the
#: transcript underneath it is being scrolled and re-laid-out, so its height
#: is a reflow budget rather than a reading budget, and twenty rows is already
#: a quarter of a standard frame.
#:
#: The TAIL is kept, where the settled block keeps the head — the two answer
#: different questions. A receipt is read from the top ("what did this do");
#: a live view is watched at the bottom ("what is it doing NOW"), and a
#: `seq 10000` whose live view froze on lines 1-40 would be indistinguishable
#: from a hung command, which is the exact anxiety this feature exists to
#: relieve.
LIVE_MAX_LINES = 20
#: Characters of a partial payload the card will look at, counted from the
#: END. bash re-sends its whole accumulated output every 500 ms (a SNAPSHOT,
#: not a delta — see `execute_bash._emit_update`), so a command that has
#: printed 40 MB delivers 40 MB per update and a naive `splitlines()` of it is
#: O(total output) twice a second, forever. Only the tail can survive the line
#: cap anyway, so everything before it is dropped without being parsed and
#: per-update work becomes a constant. Sized well above ``LIVE_MAX_LINES`` ×
#: a wide terminal so the cap is reached by the line count, not by the slice.
LIVE_INGEST_CHARS = 64 * 1024
#: The live block's heading, stating the one thing a half-drawn card must not
#: leave ambiguous. An expanded running card that showed a command and an
#: empty output area is byte-identical to a FINISHED call that printed
#: nothing, and those two frames mean opposite things.
LIVE_HEADER_RUNNING = "⋯ running"
#: Appended when nothing has STREAMED to this card yet. "no output" alone is
#: the SETTLED card's answer and means "never will"; this one means "not yet",
#: and the difference is whether the user should keep waiting.
#:
#: It is a statement about the CARD's buffer, not about the tool, and against
#: the real producer that is a narrow window: bash's first 500 ms update is
#: `--- stdout ---\n(empty)\n--- stderr ---\n(empty)`, four non-blank lines,
#: which fills the buffer and lifts this caveat on a command that has printed
#: nothing. Deliberately not special-cased — the card would have to parse one
#: tool's private banner format to tell an empty payload from a full one, and
#: it refuses that everywhere else (see :meth:`set_partial_detail`). The
#: banners are themselves legible as "nothing yet", so the frame stays honest;
#: it is this clause that stops being load-bearing, not the card.
LIVE_HEADER_PENDING = "no output yet"
#: How often a live card repaints, in seconds. ONE timer drives both the
#: duration on the collapsed row and the streaming body, for the reason the
#: subagent panel gives for its own single tick: a repaint scheduled per
#: arriving event turns a chatty producer into a repaint storm, and a card
#: that is dirty is repainted whole on the next tick instead. 1 Hz is the
#: app's existing cadence for "this is still going" (the composing clock ran
#: at exactly this rate before it was folded into this one) and it is a
#: CEILING: bash's own 500 ms emit floor cannot push the card past it.
CLOCK_INTERVAL_S = 1.0
#: Search details are persisted separately from the token-bounded model text.
#: The expansion gets enough of each provider snippet to identify the page,
#: while the source URL remains the path to the complete document.
SEARCH_EXPANDED_SNIPPET_CHARS = 360
#: Last-resort width for a card built before it has been laid out. Content
#: rendered at this width is corrected by the first resize.
FALLBACK_WIDTH = 80

#: Argument names that identify WHAT a tool is acting on, as opposed to the
#: payload it is acting WITH. A summary built from these stays about the
#: subject; one built from ``content`` or ``new_text`` is a preview of a
#: file body squeezed into forty cells, which tells the reader nothing.
IDENTITY_ARGS = frozenset(
    {
        "command",
        "path",
        "file_path",
        "url",
        "pattern",
        "query",
        "name",
        "target",
        "message",
    }
)


def _format_bytes(count: int) -> str:
    """A byte count at a glance: `812 B`, `12.4 KB`, `1.2 MB`.

    One decimal above a kilobyte, because the point of the number is that it
    MOVES — a counter that ticks tells the user the model is still dictating,
    which is the whole reason the composing row exists.
    """
    if count < 1024:
        return f"{count} B"
    if count < 1024 * 1024:
        return f"{count / 1024:.1f} KB"
    return f"{count / (1024 * 1024):.1f} MB"


def format_duration(seconds: float) -> str:
    """Active processing time: ``9s``, ``41m1s``, ``1h2m``, ``4d5h``.

    Units are dropped once they stop carrying information: past an hour the
    seconds are noise, and a whole minute renders as ``5m`` rather than
    ``5m0s``. Sub-second work renders as ``0s`` rather than vanishing, so a
    finished turn always leaves a mark.

    BOUNDED AT SIX CELLS over the whole domain, which callers rely on. The
    widest strings are ``59m59s``, ``23h59m`` and ``99d23h``; from 100 days it
    is ``100d+``. It used to end at ``{h}h{m}m`` with an unbounded hours field,
    so ``100h30m`` was 7 cells and ``1000h30m`` was 8 — and
    :data:`WorkingBlock._CLOCK_COL` RESERVES cells for this rather than
    measuring them, so a wider string pushed its row over the terminal (review
    round 15).

    The days branch is also the more readable answer at that magnitude, which
    is the reason it is a branch and not a clamp: clipping the number to fit
    would render ``100h40m`` as ``100h4…``, and ``100h4m``, ``100h40m`` and
    ``100h45m`` all collapse to that same string. Prose survives truncation
    because the reader reconstructs it; a duration does not — and this number
    is load-bearing exactly when it is largest, since a phase that has been
    running for days is the "is this stuck" question the clock exists to
    answer (design round 14).
    """
    total = int(seconds)
    if total < 60:
        return f"{total}s"
    if total < 3600:
        minutes, secs = divmod(total, 60)
        return f"{minutes}m{secs}s" if secs else f"{minutes}m"
    if total < 86_400:
        hours, remainder = divmod(total, 3600)
        minutes = remainder // 60
        return f"{hours}h{minutes}m" if minutes else f"{hours}h"
    days, remainder = divmod(total, 86_400)
    if days > 99:
        # The cap, so the width is bounded by CONSTRUCTION and not by how large
        # anyone expected the input to get.
        #
        # It says `100d+` and not `99d+` because the threshold IS 100 days, and
        # a cap has to name the bound it actually fired at. `99d+` was the first
        # spelling and design round 15 caught it: the row shows `99d23h`, then a
        # minute later shows `99d+`, which reads as the duration having got
        # SMALLER. That is the same "the number is now lying to you" failure as
        # the clipped `100h4…` this branch exists to avoid, arriving at exactly
        # the magnitude the clock matters most.
        return "100d+"
    hours = remainder // 3600
    return f"{days}d{hours}h" if hours else f"{days}d"


def truncate_cells(text: str, width: int, ellipsis: str = "…") -> str:
    """Truncate ``text`` to at most ``width`` cells (one width model).

    Measurement and slicing both go through ``rich.cells.cell_len``; mixing
    ``len()`` with cell width is a crash class the history documents.
    """
    if width <= 0:
        return ""
    if cell_len(text) <= width:
        return text
    if width <= cell_len(ellipsis):
        return ellipsis[:width]
    out: list[str] = []
    used = 0
    target = width - cell_len(ellipsis)
    for char in text:
        w = cell_len(char)
        if used + w > target:
            break
        out.append(char)
        used += w
    # `rstrip()` so the cut never lands as "word …". Without it the same message
    # truncates in two different typographic styles depending on the terminal
    # width — tight at one width, with a stray space at the next — which is most
    # visible on the error row, whose whole job is to read cleanly beside the
    # outcome glyph. Every caller benefits (band model label, picker
    # descriptions, tool arguments), and the result is never WIDER than before.
    return "".join(out).rstrip() + ellipsis


def compact_path(text: str) -> str:
    """Shrink an absolute path against the cwd, then the home directory.

    Only whole-token absolute paths are rewritten: a sentence that merely
    mentions a slash keeps its exact wording. Deep absolute paths otherwise
    consume the entire summary budget and the informative tail — the file
    actually being touched — is the part that gets ellipsized away.
    """
    if not text.startswith("/") or " " in text:
        return text
    try:
        cwd = os.getcwd()
    except OSError:  # cwd deleted underneath us: fall back to the raw path
        cwd = ""
    if cwd and cwd != "/" and text.startswith(cwd + "/"):
        return text[len(cwd) + 1 :]
    home = os.path.expanduser("~")
    if home and home != "/" and text.startswith(home + "/"):
        return "~/" + text[len(home) + 1 :]
    return text


def _scalar_text(value: object) -> str:
    """One argument value flattened to a single compact line ("" if unusable)."""
    if isinstance(value, str):
        return compact_path(value.strip()).replace("\n", " ").strip()
    if isinstance(value, (int, float, bool)):
        return str(value)
    return ""


def _send_summary(args: dict[str, object]) -> str:
    """The send row: HOW it lands, WHO gets it, then a preview of the body.

    The delivery-mode marker LEADS, and that ordering is the whole point. The
    row builder truncates the composed summary from the right as one string, so
    whatever sits rightmost dies first — with the marker after the target, three
    cards with the same peer and three different delivery promises painted
    byte-identical rows at ordinary widths (measured: a 37-cell conversation
    name collides at 62 columns, a 26-cell ULID session id at 52). One of those
    rows woke a peer and one did not, and the reader could not tell.

    This is the same fix ``_describe_wake_approval`` already landed for the same
    class of defect ("the BOUND leads the interval… a wake firing eight times
    and one that never stops painted the same text at three widths"): put the
    DISCRIMINATOR ahead of the free-text identity, so the field that separates
    two rows is the one that survives.

    The message preview carries no cap of its own. The row's own
    ``truncate_cells`` already sheds it to the available budget; an extra
    40-cell bound only left a third of the line empty at 100+ columns while
    protecting nothing — the marker survives either way because it now leads.
    """
    # Both labels come from the tool's own helpers so the card and the approval
    # prompt can never disagree about WHICH peer or WHICH delivery mode a call
    # names — two copies of that precedence would drift the first time a mode is
    # added. Imported in-function: this widget module must not pull the tool
    # module in at import time.
    from local_operator.tools.builtin import (
        peer_send_mode_label,
        peer_send_target_label,
    )

    mode = peer_send_mode_label(args)
    # `who` may be a raw conversation substring, i.e. model-controlled text; it
    # goes through the same scalar flattening every other summary field does.
    who = _scalar_text(peer_send_target_label(args)) or "?"
    message = _scalar_text(args.get("message"))
    parts = [part for part in (mode, who, message) if part]
    return " · ".join(parts)


def _summary_from_args(tool_name: str, args: dict[str, object]) -> str:
    """One-line summary of WHAT the tool is acting on.

    Identity arguments win over payload arguments. A ``write`` call carries
    both ``path`` and ``content``; joining the first two scalars in argument
    order buries the filename under the first sixty characters of the file
    being written, which is the one thing the row is for. When no argument
    is recognisably an identity — an unknown or MCP-provided tool — the
    generic first-two-scalars scan still applies, in argument order.
    """
    if tool_name == "send":
        # A dedicated branch rather than the identity scan: the generic path
        # would join target + message and lose the delivery-mode marker, the
        # one word that says whether the peer was woken, steered, or quietly
        # mailboxed.
        return _send_summary(args) or tool_name
    parts = [
        text
        for key, value in args.items()
        if key in IDENTITY_ARGS and (text := _scalar_text(value))
    ]
    if not parts:
        parts = [text for value in args.values() if (text := _scalar_text(value))]
    return " ".join(parts[:2]) or tool_name


def _argument_text(value: object) -> str:
    """One argument value as the expansion should print it.

    Unlike :func:`_scalar_text` this keeps NEWLINES — a heredoc or a multi-line
    patch is the shape of the thing being reported, and flattening it to one
    line is what made the collapsed row unable to carry it in the first place.
    Structured values are JSON so they are at least readable and unambiguous;
    ``default=str`` because an argument dict reaching here has already been
    through the provider and may hold anything.
    """
    if isinstance(value, str):
        return value.strip()
    if isinstance(value, (int, float, bool)):
        return str(value)
    try:
        return json.dumps(value, default=str)
    except (TypeError, ValueError):
        return str(value)


def clean_intent(intent: str | None) -> str | None:
    """A model-supplied intent, safe to paint, or ``None``.

    The harness sanitises and bounds this before it reaches the event, so this
    is a boundary re-check rather than the primary defence: the TUI also runs
    against embedders and replayed transcripts that construct cards directly,
    and an escape sequence in a string painted on every frame clears the
    terminal (the same hazard ``tool_name`` is guarded against below).

    An empty or whitespace-only intent collapses to ``None`` so callers can
    write ``intent or <fallback>`` and get the fallback rather than a blank row.

    CASE and a trailing period are normalised because this string shares a row
    with app-authored labels. The prompt asks the model for sentence case, and
    every other micro-label in this app is lowercase (`interrupted`,
    `context compacted`, `never sent`, `thinking`), so an unnormalised intent
    made the one row that alternates between the two look like two different
    components from one frame to the next. Only an ordinary capitalised word is
    lowered — `MCP`, `SQL` and `README.md` keep their shape.
    """
    if not intent:
        return None
    cleaned = _strip_control_sequences(" ".join(intent.split())).rstrip(".")
    head, _, rest = cleaned.partition(" ")
    if head[:1].isupper() and head[1:].islower():
        cleaned = head.lower() + (" " + rest if rest else "")
    return cleaned or None


def _diff_counts(details: dict[str, Any] | None) -> tuple[int, int]:
    """``(added, removed)`` line counts from a tool result's ``details``.

    Unknown, malformed, or negative counts collapse to zero so the renderer
    can stay honest: a card only ever shows a counter it was actually told.
    ``bool`` is excluded explicitly — it is an ``int`` subclass in Python and
    ``details={"added": True}`` must not print ``+1``.
    """
    if not isinstance(details, Mapping):
        return (0, 0)

    def _count(value: object) -> int:
        if isinstance(value, bool) or not isinstance(value, int):
            return 0
        return value if value > 0 else 0

    return (_count(details.get("added")), _count(details.get("removed")))


def _search_result_output(details: dict[str, Any] | None) -> list[str]:
    """Structured web-search rows: provider, page name, URL, and short snippet."""
    if not isinstance(details, Mapping):
        return []
    sources = details.get("sources")
    if not isinstance(sources, Sequence) or isinstance(sources, (str, bytes)) or not sources:
        return []

    provider = _strip_control_sequences(str(details.get("provider") or "search"))
    auth_mode = _strip_control_sequences(str(details.get("auth_mode") or ""))
    lines = [f"Provider: {provider}" + (f" ({auth_mode})" if auth_mode else ""), "Sources:"]
    for index, source in enumerate(sources, start=1):
        if not isinstance(source, Mapping):
            continue
        title = _strip_control_sequences(
            " ".join(str(source.get("title") or "Untitled result").split())
        )
        url = _strip_control_sequences(" ".join(str(source.get("url") or "").split()))
        snippet = _strip_control_sequences(" ".join(str(source.get("snippet") or "").split()))
        lines.append(f"{index}. {title}")
        if url:
            lines.append(f"   {url}")
        if snippet:
            if len(snippet) > SEARCH_EXPANDED_SNIPPET_CHARS:
                snippet = snippet[: SEARCH_EXPANDED_SNIPPET_CHARS - 1].rstrip() + "…"
            lines.append(f"   {snippet}")
    lines.append("Ask Operator to web_fetch result N (or `read <url>`) for the full page.")
    return lines


def _is_fetch_details(tool_name: str, details: dict[str, Any] | None) -> bool:
    """Whether this result is a web fetch (the tool, or a ``read <url>`` sugar).

    ``read`` serves both files and URLs, so the tool name alone cannot tell them
    apart. A fetch's details carry ``render_method`` and ``final_url`` (a file
    read carries neither), so the presence of those keys is what selects the
    fetch card without a fragile name check on every ``read``.
    """
    if tool_name == "web_fetch":
        return True
    if tool_name != "read" or not isinstance(details, Mapping):
        return False
    return "render_method" in details and "final_url" in details


def _fetch_result_output(details: dict[str, Any] | None) -> list[str]:
    """Structured web-fetch rows: the request/final URL, status, and render meta.

    Rendered from ``details`` (which never reaches the provider) rather than the
    model-facing preview, so the card can always show the final URL, status,
    content-type, render method, byte/line counts, cache state, and \u2014 when the
    render looked sparse \u2014 a one-line nudge toward ``browser``. The preview body
    itself is appended by the body painter from the result text; this helper owns
    only the header rows, mirroring ``_search_result_output``'s split of duties.
    """
    if not isinstance(details, Mapping):
        return []
    url = _strip_control_sequences(str(details.get("url") or ""))
    final = _strip_control_sequences(str(details.get("final_url") or url))
    if not url and not final:
        return []
    status = details.get("status")
    ctype = _strip_control_sequences(str(details.get("content_type") or ""))
    method = _strip_control_sequences(str(details.get("render_method") or ""))
    cache = _strip_control_sequences(str(details.get("cache") or "miss"))
    lines_n = details.get("lines")
    byte_n = details.get("bytes")

    meta = " · ".join(
        part
        for part in (
            f"{status}" if status is not None else "",
            ctype,
            f"cache {cache}",
        )
        if part
    )
    fetched = f"Fetched: {url}"
    if final and final != url:
        fetched += f"  (final: {final})"
    if meta:
        fetched += f"  ·  {meta}"
    rows: list[str] = []
    # F1: a non-2xx leads with a prominent error row (painted in the danger
    # treatment by the body painter) so a block/error page is visually distinct
    # from successful content — the card equivalent of the tool result's
    # is_error flag. The ``http_error`` boolean is authoritative; fall back to
    # the status int for older cached shapes without the flag.
    http_error = details.get("http_error")
    if http_error is None and isinstance(status, int):
        http_error = not (200 <= status < 300)
    if http_error and isinstance(status, int):
        reason = _HTTP_REASONS.get(status, "Error")
        rows.append(f"⚠ HTTP {status} {reason} — error/block page, not page content.")
    rows.append(fetched)

    render_bits = " · ".join(
        part
        for part in (
            method,
            f"{lines_n} lines" if isinstance(lines_n, int) else "",
            # D2: humanise (KB/MB) so the structured row agrees with the binary
            # notice body two lines below, which already prints e.g. "2.4 MB".
            _humanize_bytes(byte_n) if isinstance(byte_n, int) else "",
        )
        if part
    )
    if render_bits:
        rows.append(f"Rendered: {render_bits}")
    if details.get("low_quality"):
        rows.append("sparse/JS-gated — try `browser` for the full page.")
    return rows


#: Short reason phrases for the statuses a fetch card most often shows, kept in
#: sync with ``web_fetch/tool.py::_STATUS_REASONS`` so the card's error row reads
#: the same as the model-facing lead. A code not listed falls back to "Error".
_HTTP_REASONS: dict[int, str] = {
    400: "Bad Request",
    401: "Unauthorized",
    403: "Forbidden",
    404: "Not Found",
    405: "Method Not Allowed",
    408: "Request Timeout",
    410: "Gone",
    429: "Too Many Requests",
    451: "Unavailable For Legal Reasons",
    500: "Internal Server Error",
    502: "Bad Gateway",
    503: "Service Unavailable",
    504: "Gateway Timeout",
}

#: Matches the model-facing preview header's lead line for BOTH the 2xx compact
#: form (``[200] https://…``) and the non-2xx warning form
#: (``⚠ HTTP 403 Forbidden — … https://…``), built by
#: ``web_fetch/tool.py::_header_line``. Used to strip that block from the CARD
#: body since the structured rows carry it (D1).
_FETCH_HEADER_LEAD_RE = re.compile(r"^(?:\[(?:HTTP )?\d{3}\] \S|⚠ HTTP \d{3}\b)")
#: The header's metadata line: ``method · ctype · cache …`` (a ``·``-joined meta
#: line). Recognised structurally so a change to the field set does not silently
#: reintroduce the duplication.
_FETCH_HEADER_META_RE = re.compile(r"^\S.* · .*cache ")
#: The non-2xx header's third line — the parenthetical that labels the body as
#: the error response. Stripped from the card body alongside the lead/meta lines
#: (the structured error row already carries the "not page content" message).
_FETCH_HEADER_NOTE_RE = re.compile(r"^\(The body below is the error response")


def _humanize_bytes(count: int) -> str:
    """``2517000`` → ``2.4 MB``. Matches ``web_fetch/render.py::_human_bytes`` so
    the structured card row and the binary-notice body agree (D2). Kept local
    rather than imported to avoid a UI→service-package dependency for one format.
    """
    if count < 0:
        return f"{count} B"
    value = float(count)
    for unit in ("B", "KB", "MB", "GB"):
        if value < 1024 or unit == "GB":
            return f"{value:.0f} {unit}" if unit == "B" else f"{value:.1f} {unit}"
        value /= 1024
    return f"{count} B"


def _strip_fetch_header(lines: list[str]) -> list[str]:
    """Drop the model-facing header block from a fetch card body (D1).

    The preview text the model sees leads with ``[status] url`` (or, for a
    non-2xx, ``⚠ HTTP … — … url``) then a ``method · ctype · cache`` line, and —
    for a non-2xx — a third ``(The body below is the error response…)`` note (see
    ``web_fetch/tool.py::_header_line``). The card's structured
    ``Fetched:/Rendered:`` rows and its own error row already carry those fields,
    so painting the preview verbatim doubled them. This removes that leading
    block — and only that block — leaving the real content. Defensive: if the body
    does not start with the recognised header (a future shape change, a binary
    notice), nothing is stripped and the body is returned unchanged.
    """
    if not lines or not _FETCH_HEADER_LEAD_RE.match(lines[0]):
        return lines
    start = 1
    # The meta line immediately follows the lead line; strip it only when present.
    if start < len(lines) and _FETCH_HEADER_META_RE.match(lines[start]):
        start += 1
    # The non-2xx header carries a third note line; strip it too when present.
    if start < len(lines) and _FETCH_HEADER_NOTE_RE.match(lines[start]):
        start += 1
    # Collapse a single blank separator the header left behind so the body does
    # not open with an empty row under the structured rows.
    while start < len(lines) and not lines[start].strip():
        start += 1
    return lines[start:]


def _clamp_runs(runs: list[tuple[str, Style]], limit: int) -> list[tuple[str, Style]]:
    """Trim styled runs from the tail so their total is at most ``limit``.

    The last line of defence for the ONE-LINE guarantee: whatever a status
    segment wants to say, the row it lives on is finite.
    """
    if limit <= 0:
        return []
    out: list[tuple[str, Style]] = []
    used = 0
    for text, style in runs:
        room = limit - used
        if room <= 0:
            break
        if cell_len(text) > room:
            text = truncate_cells(text, room, ellipsis="")
            if not text:
                break
        out.append((text, style))
        used += cell_len(text)
    return out


def _row_text() -> Text:
    """A ``Text`` carrying wrap-suppression flags for the Rich-only paths.

    IMPORTANT, because the previous docstring claimed a guarantee this does
    not provide: Textual's ``Content.from_rich_text`` DISCARDS ``no_wrap`` and
    ``overflow`` when a Rich ``Text`` crosses into a widget, so these flags do
    nothing for the composed card. They are kept because they DO apply when
    the same row is measured or exported through Rich directly (tests, SVG
    export), and dropping them would change those paths.

    What actually holds the one-row rule, in order:

    1. ``ToolCard { height: 1 }`` in the stylesheet — the real enforcement. A
       collapsed card is one cell tall, so wrapped content is clipped, never
       reflowed. ``ToolCard.tool-expanded`` relaxes it to ``height: auto``
       precisely because expansion is the one case that may be taller.
    2. The status segment is hard-trimmed to ``width - 3`` in ``_build_row``,
       so no state's label can outgrow its column.
    3. Diff counters are dropped first when the cap bites, so the outcome
       glyph and duration always survive.

    Every segment is measured with ``rich.cells.cell_len``, never ``len``, so
    CJK, emoji and ZWJ sequences account correctly.
    """
    return Text(no_wrap=True, overflow="ellipsis")


class ToolCard(ExpandableActionBlock):
    """A tool execution: ONE row, on a state-tinted elevation step.

    Lifecycle: construct with ``tool_call_id``/``tool_name`` (running),
    :meth:`mark_done` on success, :meth:`mark_failed` on error,
    :meth:`mark_interrupted` when the turn ends first. Both terminal calls
    accept the tool's result text and ``details`` payload; that is what
    powers the diff counters and the click-to-expand output. Passing them is
    optional so a host that has not wired the result through still gets a
    correct — merely quieter — card.

    The card is a FOCUS STOP. Expansion used to be mouse-only, which made it
    invisible to anyone driving the app from the keyboard and unreachable in
    a terminal without mouse reporting; ``can_focus`` puts every row in the
    screen's tab order, so Shift+Tab out of the composer lands on the last
    action and Up/Down walk the ledger from there.
    """

    #: Shared by every expandable action row; naming it here lets the base
    #: toggle expansion without knowing this subclass's CSS selector.
    EXPANDED_CLASS = "tool-expanded"

    def __init__(
        self,
        tool_call_id: str,
        tool_name: str,
        args: dict[str, object] | None = None,
        intent: str | None = None,
        user_run: bool = False,
    ) -> None:
        super().__init__()
        self.tool_call_id = tool_call_id
        # tool_name is MODEL-CONTROLLED: the loop takes it from the tool call
        # itself, so a hallucinated or injected name reaches the frame. It was
        # the one raw-text entry point the sanitisation pass missed, and it is
        # the worst one to miss — the name is rendered on every row, so an
        # erase-display in it clears the terminal without the tool even running.
        self.tool_name = _strip_control_sequences(tool_name)
        self.add_class("tool-card", "tool-running")
        # The row's IDENTITY: what the call was made against. Sanitised at the
        # boundary because argument values are model-controlled too (a bash
        # command can carry a colour code).
        #
        # Deliberately NOT the ``intent``, which is available here and used by
        # the working line instead. The two say different kinds of thing: an
        # intent is the model's CLAIM about what it is doing, the arguments are
        # what it actually ran. A user scrolls the ledger back precisely when
        # those two might disagree, and a card captioned with the claim would
        # hide the disagreement — so the receipt keeps the fact and the
        # transient working line carries the claim. (Reported from the field as
        # bash rows that could not say which command they had run.)
        self._summary = _strip_control_sequences(_summary_from_args(tool_name, args or {}))
        #: The call's arguments, kept WHOLE for the expansion. The collapsed
        #: row can only ever show a truncated identity, so without this there
        #: was no state of the card — collapsed or expanded — from which a user
        #: could learn what actually ran: the expansion showed exit code, stdout
        #: and stderr and never the command. A receipt that cannot name what it
        #: is a receipt for is not auditable.
        self._args: dict[str, object] = dict(args or {})
        #: The tail of the tool's output SO FAR, one entry per line, shown in
        #: the EXPANSION of a running card and dropped when the real result
        #: lands. It used to be a single collapsed-to-one-line string painted
        #: over the summary, and nothing restored the summary afterwards: a
        #: bash row whose last fragment was `--- stderr --- (empty)` carried
        #: that as its settled receipt forever, so four finished rows read the
        #: same and none could say which command it had run.
        #:
        #: Moving it into the expansion settles that class of defect rather
        #: than patching it. The collapsed row is the IDENTITY and now never
        #: yields it to progress; progress is what the user opens the card for
        #: ("if we wish to"), so it costs no rows until then.
        self._live: list[str] = []
        #: Lines the bound has discarded off the FRONT of ``_live``, announced
        #: on a marker row so a truncated live view never reads as complete.
        self._live_dropped = 0
        #: True when the payload itself was sliced before parsing, so the drop
        #: count above is a floor rather than a total (see
        #: :meth:`set_partial_detail`) and the marker must not quote a number.
        self._live_elided = False
        #: Set by an arriving update, consumed by the clock tick. The
        #: coalescing itself comes from `set_partial_detail` never repainting
        #: and the tick repainting on its own schedule; what this bit adds is
        #: the answer to "can the card have changed HEIGHT" — only a tick that
        #: took new output may have, and only that tick needs the container's
        #: gap and scroll work (see :meth:`_tick_clock`).
        self._live_dirty = False
        #: The model's stated reason for this call, sanitised, or None. Carried
        #: on the card but never RENDERED by it (see ``_summary`` above): the
        #: working line reads it from here so that "which calls are live and
        #: what did the model say it was doing" stays one dictionary rather than
        #: two that can disagree.
        self.intent = clean_intent(intent)
        self._state: str = "running"
        # Live-row bookkeeping: bytes dictated so far, when dictation started,
        # and the ONE clock that keeps a live row visibly moving — through a
        # provider's silence while composing, and through a slow command's
        # while running. Both states need exactly the same 1 Hz repaint, and
        # two timers for it would be two things to leak.
        self._compose_bytes: int = 0
        self._compose_facts: str = ""
        self._compose_started: float | None = None
        self._clock_timer: Timer | None = None
        self._duration: float | None = None
        self._error: str = ""
        #: When THIS card's execution began, or ``None`` when the card cannot
        #: know. ``None`` is the replay case and it is not a missing value to
        #: be defaulted: a card rebuilt by a surface that is re-painting a
        #: recorded trajectory (`subagent_view.entry_block`) was constructed
        #: when the page painted, not when the tool started, so a duration
        #: measured from here would be how long ago the panel drew the row.
        #: The settled states already refuse to invent that number — see
        #: :meth:`restore` — and the running state has to refuse it too.
        self._started: float | None = time.monotonic()
        self._expanded = False
        #: A host that knows the user ran this call TO SEE its output (the
        #: composer's bang-mode) asks the card to open the moment it settles,
        #: because a collapsed receipt hides exactly the bytes the command was
        #: run for behind a click nobody asked for. Consumed once, at settle.
        self._open_on_settle = False
        #: The call was typed by the USER (bang-mode), not issued by the model.
        #: Without a visible say-so the receipt is byte-identical to an
        #: agent-run bash row, so a reader scrolling back cannot tell their
        #: own commands from the agent's, and the model reading the context
        #: cannot tell a command it ran from one that "just appeared". The
        #: row answers with a `you:` chip ahead of the summary; the system
        #: prompt answers for the model. A user-run command is also one the
        #: user ran to SEE, so the flag implies open-on-settle.
        self.user_run = user_run
        if user_run:
            self._open_on_settle = True
        self._hovered = False
        #: True while the row holds keyboard focus. Tracked separately from
        #: hover: both light the hint, but a pointer leaving the row must not
        #: put out a hint the keyboard is still standing on.
        self._focused = False
        #: One-shot answer occupying the hint slot ("" = nothing to say).
        self._notice = ""
        self._notice_timer: Any = None
        self._added = 0
        self._removed = 0
        #: Cleaned result lines, populated once the tool finishes.
        self._output: list[str] = []
        #: The write/edit tool's rendered unified diff (``details["diff"]``),
        #: colourised at render time. ``None`` (not []) when the tool reported
        #: no diff — a read or bash card expands to its raw output instead.
        self._diff: list[str] | None = None
        #: True when this card holds a web_fetch (or a ``read <url>`` sugar)
        #: result, so the body painter and the rest-visibility rule can select
        #: the fetch presentation. Set in :meth:`_absorb_result`.
        self._is_fetch_card = False
        #: Rows the card currently occupies (1 collapsed, N expanded).
        self._row_count = 1
        #: ``_row_count`` as of the last content APPLIED to the widget, or -1
        #: when nothing has been applied yet. ``_refresh_row`` compares the two
        #: to decide whether the update needs a layout pass; the sentinel makes
        #: the first application always lay out.
        self._applied_rows = -1
        #: The width the applied content was folded to, or -1 when nothing has
        #: been applied. ``on_resize`` compares against it so a height-only
        #: resize does not rebuild a row that would come out identical.
        self._built_width = -1
        self._refresh_row()

    # -- lifecycle ----------------------------------------------------------
    def mark_done(self, result_text: str = "", details: dict[str, Any] | None = None) -> None:
        """Record success with elapsed duration; the row goes quiet."""
        self._settle_live()
        self._duration = self._elapsed()
        self._state = "success"
        self._absorb_result(result_text, details)
        self.remove_class("tool-running")
        self.add_class("tool-success")
        self._refresh_row()
        self._apply_open_on_settle()
        self.finalize()

    def mark_failed(
        self, error: str, result_text: str = "", details: dict[str, Any] | None = None
    ) -> None:
        """Record failure with a ONE-line error message.

        ``result_text`` defaults to the error itself: a failed tool's full
        message is frequently a stack trace or a multi-line diagnostic, and
        that is exactly what the expansion exists to show.
        """
        # `_settle_live`, not just a duration: `mark_failed` was the ONE settle
        # path that never stopped the clock. Harmless while only a composing
        # row had one — a failed call has almost always started running — but
        # the clock now runs for the whole of `running` too, so every failing
        # tool would have left a 1 Hz timer repainting a finalized card.
        self._settle_live()
        self._duration = self._elapsed()
        self._state = "error"
        self._error = _strip_control_sequences(" ".join(error.split())) or "error"
        self._absorb_result(result_text or error, details)
        self.remove_class("tool-running")
        self.add_class("tool-error")
        self._refresh_row()
        self._apply_open_on_settle()
        self.finalize()

    def mark_interrupted(self) -> None:
        """Turn ended before this tool completed: dim 'interrupted' state."""
        was_composing = self._state == "composing"
        self._settle_live()
        if was_composing:
            # The call was never sent, so the row must stop saying it is being
            # written. It keeps the size as a record of how far the model got.
            size = _format_bytes(self._compose_bytes) if self._compose_bytes else "nothing"
            # Facts first, and `_compose_facts` set so the label-shed ladder can
            # reach this row too: gated on the composing STATE, the ladder built
            # for the live row could not touch the record it turns into, and
            # three materially different interrupted rows painted identically
            # from 40 columns down.
            self._compose_facts = f"{size} composed"
            self._summary = f"never sent · {self._compose_facts}"
        self._duration = self._elapsed()
        self._state = "interrupted"
        self.remove_class("tool-running")
        self.add_class("tool-interrupted")
        self._refresh_row()
        self._apply_open_on_settle()
        self.finalize()

    def open_on_settle(self) -> None:
        """Ask the card to open the moment it settles, once.

        Bang-mode's contract: the user typed the command to read its output,
        so the settled card starts expanded instead of collapsing the result
        behind `⟨expand⟩`. A settle with nothing to reveal (empty output) is
        a no-op — :meth:`can_expand` decides — and a user collapse after the
        open is final, because the flag is consumed at the first settle.
        """
        self._open_on_settle = True

    def _apply_open_on_settle(self) -> None:
        """Honour :meth:`open_on_settle` now that a result is absorbed.

        Runs between the settle's own `_refresh_row` and `finalize` so the
        block is measured at its expanded height from its first frame. The
        gap refresh is `refresh_gap_around` — the same call the live-growth
        tick makes — because the height changes the spacing above as well as
        below; the transcript's sticky-bottom anchor follows the growth on
        its own.
        """
        if not self._open_on_settle:
            return
        self._open_on_settle = False
        if self._expanded or not self.can_expand():
            return
        self._expanded = True
        self.set_class(True, self.EXPANDED_CLASS)
        self._refresh_row()
        parent = self.parent if isinstance(self.parent, TranscriptView) else None
        if parent is not None:
            parent.refresh_gap_around(self)

    def _elapsed(self) -> float | None:
        """Seconds this call has been running, or ``None`` when unknowable.

        ``None`` propagates rather than collapsing to ``0.0``: a card that
        never learned when its call started must paint a blank column, not a
        number that says the tool returned instantly.
        """
        return None if self._started is None else time.monotonic() - self._started

    def restore(
        self,
        *,
        state: str,
        result_text: str = "",
        details: dict[str, Any] | None = None,
        error: str = "",
        duration_s: float | None = None,
    ) -> None:
        """Adopt a card for a call from a PREVIOUS session, or another agent's.

        Separate from :meth:`mark_done` / :meth:`mark_failed` for one reason:
        those three compute ``_duration`` from ``self._started``, which for a
        row mounted during replay is the moment the row was mounted. That is
        not how long the tool took, it is how long ago the app painted the row,
        and it renders as ``0.0s`` on every card in a resumed conversation.
        The transcript does not record durations, so replay leaves
        ``_duration`` at ``None`` and the column paints blank.

        ``state`` is ``"success"``, ``"error"``, ``"interrupted"`` — the third
        for a call whose result is not in the transcript, which is what a
        session killed mid-turn leaves behind — or ``"running"``.

        ``"running"`` exists for the same reason the other three do, one step
        further on. `subagent_view.entry_block` rebuilds a child's whole
        trajectory as blocks, and an entry with no outcome yet is a call that
        is STILL GOING: the card has to stay live. It just must not time
        itself, because its ``_started`` is when the page painted the row —
        so ``_started`` is cleared here and the running row blanks its
        duration exactly as its settled siblings blank theirs. Without this
        the replayed live row counted up from zero (and reset to zero every
        time an earlier entry changed and the page rebuilt it), inventing
        precisely the number this method exists to refuse.
        """
        self._settle_live()
        self._state = state
        self._started = None
        # Replay stays blank for legacy rows, while newly persisted executions
        # show the exact same receipt as the live card did.
        self._duration = max(0.0, duration_s) if duration_s is not None else None
        if state == "running":
            # Still live: it keeps `tool-running`, it is not finalized, and the
            # expansion still offers the command. Only the clock is withheld.
            self._refresh_row()
            return
        self.remove_class("tool-running")
        if state == "error":
            self._error = _strip_control_sequences(" ".join(error.split())) or "error"
            self._absorb_result(result_text or error, details)
            self.add_class("tool-error")
        elif state == "interrupted":
            self.add_class("tool-interrupted")
        else:
            self._absorb_result(result_text, details)
            self.add_class("tool-success")
        self._refresh_row()
        self._apply_open_on_settle()
        self.finalize()

    def set_composing(self, argument_bytes: int, tool_name: str = "") -> None:
        """Show that the model is still WRITING this call's arguments.

        A separate state from `running`, because nothing has run: the row must
        not claim a tool is executing while the request is still being dictated.

        It shows BOTH what has arrived and how long it has been going, and the
        second half is not decoration. Providers pause mid-call — measured on a
        real Anthropic stream, a `write` block opened and then sent nothing for
        eighty seconds before delivering fourteen kilobytes in under a second.
        A byte count alone is static text through the whole of that pause, which
        is precisely the frame a user reads as a hung agent. A clock that ticks
        says "still going" during exactly the silence that needs it said.
        """
        if self._state not in ("composing", "running"):
            return
        entering = self._state != "composing"
        self._state = "composing"
        # The name can arrive in FRAGMENTS: the first announcement fires on the
        # first piece so the row appears at once, so a provider that splits
        # `write` sends `wr` and then `write`. Following it keeps the visible row
        # and its icon honest for the whole dictation, not just once the call
        # starts running.
        renamed = bool(tool_name and tool_name != self.tool_name)
        if renamed:
            self.tool_name = _strip_control_sequences(tool_name)
        # Two reasons the shared column may have moved: this row's NAME changed,
        # or the row just stopped contributing a name at all by entering
        # `composing`. The second matters because the card is mounted as
        # `running` and flipped here — with the same name it was constructed
        # with, so a rename-only check never fired and a dictated name kept the
        # column it had already widened.
        if renamed or entering:
            parent = self.parent
            if isinstance(parent, TranscriptView):
                parent.invalidate_name_col()
        self._compose_bytes = argument_bytes
        if self._compose_started is None:
            self._compose_started = time.monotonic()
        self._start_clock()
        self._render_composing()

    def _render_composing(self) -> None:
        started = self._compose_started or time.monotonic()
        elapsed = max(0, int(time.monotonic() - started))
        clock = format_duration(elapsed)
        # The FACTS lead and the label sheds whole, the same shape this file
        # already uses for a tool summary. Boilerplate-first, `composing…` was
        # protected by the truncation while the two things that move were cut:
        # below 39 columns the row stopped changing at all, and two calls
        # dictating 12.4 KB and 61 B painted identically.
        #
        # The size joins only once there IS one. Providers commonly open the
        # call and deliver its arguments in one late burst, so a leading `0 B`
        # sat on screen for two minutes on a measured run — a number that never
        # moves reads as a stuck counter, which is what this row exists to fix.
        #
        # `composing`, not `writing`: `write` is a TOOL in this app, so
        # `write  writing the call…` read as a stutter and `read  writing the
        # call…` read as wrong.
        facts = f"{_format_bytes(self._compose_bytes)} · {clock}" if self._compose_bytes else clock
        self._compose_facts = facts
        self._summary = f"composing… {facts}"
        self._refresh_row()

    def _start_clock(self) -> None:
        """Run the 1 Hz repaint for as long as this row is live (idempotent).

        Guarded on the message pump, the way :meth:`_flash_notice` is: a card
        built but not yet mounted — which every unit test holding one directly
        is — has no loop to schedule against, and ``set_interval`` would raise
        out of a lifecycle method and leave an unawaited coroutine behind. With
        no clock the row simply stops animating between events, which is the
        right degradation for a timer whose entire job is cosmetic.

        A REPLAYED live card (``restore(state="running")``) gets none: it has
        no start time to count from and nothing streams into it, so every
        tick would repaint an unchanged row. See :attr:`_started`.
        """
        if self._started is None:
            return
        if self._clock_timer is None and self.is_running:
            self._clock_timer = self.set_interval(CLOCK_INTERVAL_S, self._tick_clock)

    def _tick_clock(self) -> None:
        """The card's ONE repaint point while it is live.

        Three things ride this tick, and none of them may schedule a repaint of
        its own: the composing row's byte clock, the running row's duration,
        and the coalesced live-output buffer. The last is why the bit exists —
        bash re-sends its whole accumulated output every 500 ms and a `yes`
        loop would otherwise repaint a growing widget twice a second forever.
        """
        if self._state == "composing":
            self._render_composing()
            return
        if self._state != "running":
            # Belt for a clock that outlived its row: every settle path calls
            # `_settle_live`, so this should be unreachable.
            self._stop_clock()
            return
        # The duration is on the row whether or not output moved, so the tick
        # always REPAINTS a running card — the clock IS the "not hung" signal
        # and skipping the paint would freeze it.
        grew = self._live_dirty
        self._live_dirty = False
        # …but only a tick that took new output can have changed the card's
        # HEIGHT, and only a height change concerns the container. Collapsed,
        # the card is one row and cannot reflow anything at all; expanded with
        # nothing new, the gap work below would run once a second per open card
        # — a linear `_blocks.index` scan — to re-settle a layout that did not
        # move.
        if not self._expanded or not grew:
            self._refresh_row()
            return
        parent = self.parent if isinstance(self.parent, TranscriptView) else None
        self._refresh_row()
        if parent is None:
            return
        # A card that changed height changes the gap its neighbours need, above
        # as well as below (see `refresh_gap_around`).
        #
        # It does NOT scroll. Keeping the tail in view as a live card grew was
        # sampled and re-pinned here, which made this a third private copy of
        # the sticky-bottom rule. `TranscriptView` now follows its own extent
        # (`_size_updated`), so a growing card is anchored by the same state
        # machine as a streaming message — and, unlike the copy, honours the
        # release when the reader has scrolled up to read.
        parent.refresh_gap_around(self)

    def on_mount(self) -> None:
        """Start the clock for a card that was mounted already RUNNING.

        The common path — a call the model dictated — starts it in
        :meth:`begin_running`, but a card built straight from
        ``tool_execution_start`` (a provider that never streams a compose
        event, or a replayed batch) is constructed running and mounted after,
        and ``__init__`` has no message pump to schedule against.
        """
        if self._state in ("running", "composing"):
            self._start_clock()

    def _stop_clock(self) -> None:
        """Retire the clock; the row is about to become something else."""
        if self._clock_timer is not None:
            self._clock_timer.stop()
            self._clock_timer = None

    def _settle_live(self) -> None:
        """Stop the clock and drop the streamed tail: the real result is here.

        The live buffer is a stand-in for a result that had not arrived. Once
        it has, keeping the tail would leave the card holding two accounts of
        the same output — and the streamed one is the truncated, out-of-date
        one.
        """
        self._stop_clock()
        self._live = []
        self._live_dropped = 0
        self._live_elided = False
        self._live_dirty = False

    def begin_running(
        self, tool_name: str, args: dict[str, object] | None, intent: str | None
    ) -> None:
        """Adopt a composing row as the real execution of the call it announced.

        The same widget rather than a fresh one: the composing row already sits
        in the transcript in the right place, and replacing it would make the
        ledger flicker a row out and an identical row back in at the moment the
        call finally starts.
        """
        self._stop_clock()
        # The name comes from the EXECUTION, not from the announcement. The first
        # compose event fires on the first name fragment — deliberately, so the
        # row appears immediately — and a provider that splits `write` into `wr`
        # and `ite` would otherwise leave `wr` on the settled row forever, in the
        # ledger, on the icon, and in the summary built from it.
        self.tool_name = _strip_control_sequences(tool_name)
        # The duration clock RESTARTS here. `_started` was set when the row was
        # mounted, which for an adopted row is when the model began dictating —
        # so a `write` that executed in 0.1s settled as `✓ 2.4s`, and on the
        # reported 1m41s case would have read `✓ 101s`. Two receipts on one
        # ledger would then be measuring different things with no way to tell
        # which from the row.
        self._started = time.monotonic()
        self._state = "running"
        # The same construction the constructor uses, so an adopted row is
        # byte-identical to one that had never been a composing row — including
        # taking the ARGUMENTS rather than the intent (see the constructor for
        # why the receipt keeps the fact).
        self._summary = _strip_control_sequences(_summary_from_args(self.tool_name, args or {}))
        self._args = dict(args or {})
        self._live = []
        self._live_dropped = 0
        self._live_elided = False
        self._live_dirty = False
        # THE INVARIANT: `_compose_facts` is non-empty exactly while `_summary`
        # is the one built FROM it. `_build_row` reads it as "this row's summary
        # is boilerplate plus two moving numbers, so shed the boilerplate before
        # you truncate the numbers" — a ladder that is right for a composing row
        # and catastrophic for any other, because on any other row it sheds the
        # SUMMARY and keeps the numbers.
        #
        # Nothing cleared it here, so it stayed set for the whole life of the
        # card, and the ladder fires whenever `width < _label_min_width()` — a
        # threshold computed FROM the summary's own length. Measured at 80
        # columns: a 75-character command needs 97, so every bash row in the
        # ledger, running AND settled, painted `199 B · 1s` where its command
        # belonged. `read` and `grep` escaped only because a path summary is
        # short enough to clear the threshold; the defect was never about bash,
        # it was about any summary longer than the frame.
        #
        # The `· 1s` was the same bug's other half: the frozen compose clock,
        # last ticked before dictation ended, sitting beside the real duration
        # in the status column and disagreeing with it. One row, two clocks,
        # neither labelled — and the one the eye reached first was the wrong one.
        self._compose_facts = ""
        # The EXECUTION's intent supersedes whatever the announcement carried,
        # for the same reason the name does: a partial intent scraped from a
        # still-streaming call is a draft, and this event carries the finished
        # one. Kept when the execution reports none, so a compose-time intent is
        # not thrown away by a provider that omits it on start.
        self.intent = clean_intent(intent) or self.intent
        # The same 1 Hz clock the composing row used, now counting EXECUTION.
        # Without it the duration on a running row is painted once, at zero,
        # and then holds still for the whole call — which is the frame this
        # change exists to stop producing, since a timer that never moves is
        # indistinguishable from a hung command.
        self._start_clock()
        self._refresh_row()

    def set_partial_detail(self, detail: str) -> None:
        """Take the tool's output SO FAR into the card's live view.

        The payload is a SNAPSHOT, not a delta: bash — the only producer of
        ``on_update`` in the codebase — re-sends its whole accumulated stdout
        and stderr every 500 ms, so each call REPLACES the buffer. Appending
        would duplicate the entire output on every update.

        Two bounds, both load-bearing under a chatty command:

        * only the last :data:`LIVE_INGEST_CHARS` of the payload are parsed, so
          the per-update cost stops growing with the total output. Without it a
          command that has printed 40 MB is 40 MB of ``splitlines()`` twice a
          second, and only the tail could ever have been displayed anyway.
        * only the last :data:`LIVE_MAX_LINES` lines are kept, so the widget's
          height is bounded no matter what the command does.

        The card does NOT re-parse bash's ``--- stdout ---`` / ``--- stderr ---``
        banners into separate streams. They already distinguish the two, the
        settled expansion shows the identical framing so the card does not
        change vocabulary when the call ends, and a card that understood one
        tool's private format would silently mangle the next producer's.

        Nothing is repainted here. The tick owns the repaint (see
        :meth:`_tick_clock`); this only marks the buffer dirty.
        """
        if self._state != "running":
            return
        # An exact drop count is only knowable when the WHOLE payload was
        # parsed. Past the slice the card has not seen the earlier output and
        # must not put a number on it: unfixed, a `seq 100000` snapshot showed
        # `… 10903 earlier lines` — the lines dropped from the 64 KB tail, not
        # the 99981 actually missing. A marker that quantifies what it cannot
        # count is worse than one that admits the gap, because it reads as a
        # measurement.
        elided = len(detail) > LIVE_INGEST_CHARS
        if elided:
            detail = detail[-LIVE_INGEST_CHARS:]
        lines = [
            _strip_control_sequences(line.rstrip()) for line in detail.expandtabs(4).splitlines()
        ]
        while lines and not lines[-1]:
            lines.pop()
        if not lines:
            return
        # Counted from the SNAPSHOT's own line count, not accumulated across
        # updates: each payload is the whole output, so the number of lines
        # scrolled off is a property of this payload alone. Accumulating would
        # multiply the count by the number of updates.
        self._live_elided = elided
        self._live_dropped = max(0, len(lines) - LIVE_MAX_LINES)
        self._live = lines[-LIVE_MAX_LINES:]
        self._live_dirty = True

    def _absorb_result(self, result_text: str, details: dict[str, Any] | None) -> None:
        """Capture the expansion payload and diff counters.

        Web search is rendered from structured details, not the model-facing
        text: the latter is deliberately token-capped, while the expansion
        must reliably retain every candidate's page name, URL, and snippet.
        Write/edit tools prefer their rendered diff; all other tools expand to
        cleaned result text.
        """
        self._added, self._removed = _diff_counts(details)
        name = self.tool_name.lower()
        search_output = _search_result_output(details) if name == "web_search" else []
        # A web_fetch (and a `read <url>`, which records tool_name "read" but
        # carries a fetch's details shape) leads with structured header rows and
        # then the rendered preview body, so the card shows what was fetched and
        # a slice of the content, with the spill footer already inside the text.
        fetch_output: list[str] = []
        if not search_output and _is_fetch_details(name, details):
            fetch_header = _fetch_result_output(details)
            if fetch_header:
                # D1: the structured rows above OWN the metadata (status, final
                # URL, method, ctype, cache), so strip the model-facing preview's
                # own leading header block before showing the body — otherwise the
                # card prints the same four fields twice. The model-facing text
                # keeps its header (the model benefits from the status line); this
                # is purely the CARD's presentation, mirroring how web_search lets
                # its structured rows replace, not duplicate, the model text.
                body = _strip_fetch_header(self._clean_output(result_text))
                fetch_output = fetch_header + [""] + body
        # Remembered so the body painter and the rest-visibility rule can select
        # the fetch presentation without re-inspecting details every repaint.
        self._is_fetch_card = bool(fetch_output)
        self._output = search_output or fetch_output or self._clean_output(result_text)
        diff = details.get("diff") if isinstance(details, Mapping) else None
        # Canonical follower details are immutable sequences, not concrete
        # lists. They still carry the same unified diff as the live executor.
        if isinstance(diff, Sequence) and not isinstance(diff, (str, bytes)) and diff:
            self._diff = [str(line) for line in diff]
        else:
            self._diff = None

    def _clean_output(self, result_text: str) -> list[str]:
        """Normalise the result into displayable rows (empty = nothing to show).

        A single-line result that merely repeats the summary carries no new
        information, so the card stays inert rather than advertising an
        expansion that reveals what is already on screen.
        """
        if not result_text:
            return []
        lines = [
            _strip_control_sequences(line.rstrip())
            for line in result_text.expandtabs(4).splitlines()
        ]
        while lines and not lines[0]:
            lines.pop(0)
        while lines and not lines[-1]:
            lines.pop()
        if not lines:
            return []
        if len(lines) == 1 and lines[0].strip() == self._summary.strip():
            return []
        return lines

    # -- expansion ----------------------------------------------------------
    def can_expand(self) -> bool:
        """True when the card holds more than the one-line summary can show.

        Three sources: the plain result output, a write/edit diff (which can be
        present on its own — a new-file write's summary line is one sentence
        while its diff is the whole file), and a call that is STILL RUNNING.

        The third is what makes "show me what it is doing" reachable. A live
        card has neither output nor diff yet, so it used to answer an
        activation with ``⟨still running⟩`` and stay shut — which told the user
        the call was alive and then refused to say anything more about it,
        during precisely the stretch they wanted to look. It opens on its
        arguments (the command) plus whatever has streamed so far, and both are
        strictly more than the row can hold.

        Still deliberately NOT "or it has arguments" in the SETTLED states:
        that would make every row expandable and retire the inert-row answer
        (``⟨no output⟩``), which is a separate affordance decision. A finished
        call that printed nothing still cannot be opened to see what it ran;
        that gap is known.
        """
        return bool(self._output) or bool(self._diff) or self._state == "running"

    def _after_toggle(self) -> None:
        """An expand/collapse supersedes any earlier inert-row answer."""
        self._clear_notice(repaint=False)

    def _on_inert_activation(self) -> None:
        self._flash_notice()

    def _has_activation_feedback(self) -> bool:
        return bool(self._notice)

    def _clear_activation_feedback(self) -> None:
        # A notice answers an activation on THIS row. Carrying it past blur
        # would leave "no output" looking like a permanent state.
        self._clear_notice(repaint=False)

    def _flash_notice(self) -> None:
        """Put the inert-row answer in the hint slot for a couple of seconds.

        A call still being DICTATED has no output yet but will; a settled one
        never will. Saying which is the difference between "wait" and "there is
        nothing here", and the row is the only place the user is looking.

        ``composing`` is now the ONLY live state that reaches here. A running
        card expands — onto its command and its streamed output — so
        ``activate`` toggles it rather than falling through to this, and the
        `⟨still running⟩` answer belongs to the row that genuinely has nothing
        behind the affordance yet.
        """
        self._notice = RUNNING_NOTICE if self._state == "composing" else NO_OUTPUT_NOTICE
        self._refresh_row()
        if self._notice_timer is not None:
            self._notice_timer.stop()
            self._notice_timer = None
        if self.is_running:
            self._notice_timer = self.set_timer(NOTICE_SECONDS, self._clear_notice)
        # A card with no running message pump (built but not yet mounted, or a
        # unit test holding one directly) has no clock to schedule against, and
        # `set_timer` would leave an unawaited coroutine behind. The notice then
        # simply persists until the next repaint — the right degradation for a
        # purely cosmetic timer, and never a crash.

    def _clear_notice(self, repaint: bool = True) -> None:
        """Retire the one-shot answer (no-op when there is none)."""
        if self._notice_timer is not None:
            self._notice_timer.stop()
            self._notice_timer = None
        if not self._notice:
            return
        self._notice = ""
        if repaint:
            self._refresh_row()

    # -- resize (TUI-017: rebuild the row when the width changes) -----------
    def on_resize(self, event) -> None:  # type: ignore[no-untyped-def]
        """Re-fit the row at the new width.

        Guarded on the WIDTH, because the card's content is a pure function of
        its state and the width it is folded to — a resize that only changed
        the HEIGHT reproduces the rows byte for byte. That is not a rare case:
        an expanded card's own content sets its height, so every expansion
        raises a Resize that landed straight back here. Measured on a session
        replay, this handler was a third of the ``_refresh_row`` calls: 645
        builds for 215 cards, ~366 ms.
        """
        size = getattr(event, "size", None)
        if size is not None and size.width == self._built_width:
            return
        self._refresh_row()

    # -- text selection (TUI-021) -------------------------------------------
    #: The icon field on the summary row: the per-tool glyph and its separator,
    #: which every rung of :meth:`_build_row` writes as ``icon + " "`` and every
    #: width budget in this file counts as ``+ 2``.
    ICON_COLS = 2

    def copy_gutter(self, index: int) -> int:
        """The icon field on the summary row; the expansion's indent below it.

        Both are gutter for the same reason, and the summary row's case is the
        one worth arguing. ``tool_icon`` returns a NERD FONT private-use
        codepoint (``\\uf120`` for bash), which is the strongest possible form
        of the failure a copy gutter exists to prevent: pasted into a bug report
        or an issue it is a replacement box, and it carries nothing the row does
        not immediately restate — the tool NAME is the next field along, in
        text. It is a fixed leading field on a single row, which is a gutter by
        the same test that makes ``NoticeBlock``'s kind glyph one, and treating
        the two differently would have been the inconsistency.

        The rest of the row stays: name column, what ran, outcome glyph and
        duration are the receipt, and a user who selects a settled row wants it.

        Below the summary, every row is written by
        ``_append_input_body``/``_append_output_body``/``_append_diff_body``/
        ``_append_live_body``, each of which opens with
        ``"\\n" + " " * OUTPUT_INDENT``. That indent is the card's own layout,
        and it is the thing standing between a copied stderr and a paste that
        goes straight into a bug report — or, for a diff, between ``+ added``
        and something ``git apply`` will not read.
        """
        return self.ICON_COLS if index == 0 else OUTPUT_INDENT

    # -- rendering ----------------------------------------------------------
    def refresh_row(self) -> None:
        """Repaint at the current width — the ledger's shared column moved.

        Public because the transcript owns the name column and has to be able to
        say "re-render, the spine changed"; everything else about the row is the
        card's own business.
        """
        self._refresh_row()

    def _refresh_row(self) -> None:
        """Rebuild the card at its OWN width (D3).

        Width resolution walks from the most authoritative source down: the
        widget's laid-out size, its container, the app console, and only
        then :data:`FALLBACK_WIDTH`. Reaching the last step means there is
        no app to paint into yet, so the content is measured but not
        applied: ``_row_count`` — which the spacing and scroll accounting
        both read — stays truthful, and ``on_resize`` paints the real thing
        the moment there is a real width.

        Finalization is bypassed deliberately: a resize or an expand must be
        able to re-fit a settled card, and the content it produces is a pure
        function of the card's state, never new history.

        The LAYOUT pass is asked for only when the row count moved. Textual's
        ``Static.update`` reflows by default and a reflow re-arranges the whole
        transcript — 7.8 ms across 173 widgets on a 161-block screen — but a
        card's footprint is its row count and nothing else: collapsed, the
        sheet pins it to ``height: 1`` and the row cannot reflow anything at
        all. Without this guard the 1 Hz clock on every running card, and every
        pointer crossing a card's edge, each reflowed the entire screen to
        repaint one row.
        """
        # `fold_width(0)` walks size → container → the parent transcript's
        # scrollable content region, and returns 0 only when none of the three
        # can answer. That third rung is new and is what stops a card built
        # before its first layout pass from fitting itself to the whole
        # terminal (the console rung below) or to 80 columns: both are the
        # wrong column for a block inside the transcript, and the row visibly
        # re-fitted a frame later. The console rung is kept as the last resort
        # BEFORE the fallback because reaching it is also how this method
        # detects that there is no app to paint into at all.
        width = self.fold_width(0)
        detached = False
        if width <= 0:
            try:
                width = self.app.console.width
            except Exception:
                width = FALLBACK_WIDTH
                detached = True
        content = self._build_content(width)
        self._row_count = max(1, len(content.plain.splitlines()))
        if detached:
            return
        self._built_width = width
        moved = self._row_count != self._applied_rows
        self._applied_rows = self._row_count
        was_finalized = self._finalized
        self._finalized = False
        try:
            self.set_content(content, layout=moved)
        finally:
            self._finalized = was_finalized

    def _build_content(self, width: int) -> Text:
        """The card: the one-row summary, plus the CALL and its result expanded.

        The call comes FIRST, before the output, because it is the question a
        reader opens the card to answer. Reported from the field on an expanded
        bash card that listed exit code, stdout and stderr and never the
        command — leaving no state of the card, collapsed or expanded, from
        which a user could learn what had run.
        """
        row = self._build_row(width)
        if not self._expanded:
            return row
        # A RUNNING card has no result to show and must not look like one that
        # came back empty. Its own block states the state and carries whatever
        # has streamed, and it takes precedence over the settled blocks, which
        # are empty at this point anyway.
        if self._state == "running":
            self._append_input_body(row, width)
            self._append_live_body(row, width)
        elif self._state == "success" and self._diff:
            # A settled write/edit expands to its DIFF ALONE. The arguments are
            # the same change stated twice — `old_text`/`new_text` (or the
            # whole `content`) escaped into flat `\n`-ridden lines above a
            # unified diff that already shows every one of those bytes with
            # markers and colour — and the path is on the summary row. Painting
            # both buried the readable form under the unreadable one; the diff
            # is the record, so it is the whole expansion. A FAILED edit takes
            # the plain-output branch below (no diff was produced) and keeps
            # its argument block: there the args are the only account of what
            # was attempted, and the error only makes sense next to them.
            self._append_diff_body(row, width)
        elif self.tool_name.lower() == "web_search" and self._output:
            self._append_input_body(row, width)
            self._append_search_body(row, width)
        elif self._is_fetch_card and self._output:
            self._append_input_body(row, width)
            self._append_fetch_body(row, width)
        elif self._output:
            self._append_input_body(row, width)
            self._append_output_body(row, width)
        else:
            self._append_input_body(row, width)
        return row

    def _append_live_body(self, row: Text, width: int) -> None:
        """The in-progress block: what state this call is in, then its output.

        The HEADER is not decoration. Expanded, a running call and a finished
        call that printed nothing produce the same frame — a command and then
        nothing — and they mean opposite things ("wait" against "that is all
        there is"). The header is the only thing on the card that separates
        them, so it is stated in words rather than left to the absence of a
        ✓, and it carries the elapsed time for the same reason the row does.

        The TAIL is shown, oldest-dropped, with the drop announced. See
        :data:`LIVE_MAX_LINES` for why this block keeps the end where the
        settled block keeps the beginning.
        """
        dim = bindings.style("tool.live.dim")
        accent = bindings.style("tool.live.header")
        line_width = max(1, width - 2 - OUTPUT_INDENT)
        indent = " " * OUTPUT_INDENT
        # State, then time, then the caveat — read left to right that is "it is
        # running / for this long / and has printed nothing", which is the
        # order the questions are asked in. The caveat last also means it
        # simply disappears when the first line arrives, instead of the row
        # rewriting itself around a moving middle.
        #
        # Both trailing clauses are dropped for a REPLAYED live card, which
        # knows neither. It has no start time (see :attr:`_started`), and
        # nothing streams into it — the surface rebuilding it never calls
        # `set_partial_detail` — so `no output yet` there is not a caveat that
        # will lift, it is a permanent claim about a child's tool that this
        # card has no way to make. `⋯ running` alone is the whole of what it
        # honestly knows.
        header = LIVE_HEADER_RUNNING
        elapsed = self._elapsed()
        if elapsed is not None:
            header = f"{header} · {format_duration(max(0, int(elapsed)))}"
            if not self._live:
                header = f"{header} · {LIVE_HEADER_PENDING}"
        row.append("\n" + indent, style=dim)
        # See `bindings.BY_ELEMENT["tool.live.header"].note` for why this rides
        # `accent`.
        row.append(truncate_cells(header, line_width), style=accent)
        if self._live_elided or self._live_dropped > 0:
            plural = "s" if self._live_dropped != 1 else ""
            marker = (
                "… earlier output not shown"
                if self._live_elided
                else f"… {self._live_dropped} earlier line{plural}"
            )
            row.append("\n" + indent, style=dim)
            row.append(truncate_cells(marker, line_width), style=dim)
        for line in self._live:
            row.append("\n" + indent, style=dim)
            row.append(truncate_cells(line, line_width), style=dim)

    def _append_input_body(self, row: Text, width: int) -> None:
        """The arguments the call was made with, one labelled block per key.

        WRAPPED, not truncated: the collapsed row is where a command is cut to
        fit, and the expansion exists precisely to show what the cut hid. Long
        payloads are still bounded per key — a ``write`` carries the whole file
        in ``content``, and a card that painted it would be a file viewer with
        a tool row on top — so each value keeps its head and reports the rest as
        a count, which is the same bargain the output block strikes.
        """
        if not self._args:
            return
        dim = bindings.style("tool.args.dim")
        label = bindings.style("tool.args.label")
        body = bindings.style("tool.args.value")
        line_width = max(1, width - 2 - OUTPUT_INDENT)
        indent = " " * OUTPUT_INDENT
        for key, value in self._args.items():
            text = _strip_control_sequences(_argument_text(value))
            if not text:
                continue
            # The KEY is model-controlled too — an MCP server or a hallucinated
            # call names it — so it gets the same stripping the value and
            # `tool_name` get, and a bound: an unbounded key drove the value's
            # width budget to a single cell while wrapping over several rows.
            key_text = truncate_cells(_strip_control_sequences(str(key)), TOOL_NAME_COL_MAX)
            head = f"{key_text}: "
            # The label is wrapped WITH the value rather than printed and then
            # the value clipped to fit beside it. Clipping silently deleted the
            # middle of the first row — a 100-column `pytest … -k expansion
            # --maxfail=1` rendered as `-k expa… --maxfail=1`, which is not a
            # shortened command but a different one, and this block exists
            # precisely so that a command the collapsed row had to cut can be
            # read whole.
            rows: list[str] = []
            for index, source in enumerate(text.splitlines() or [""]):
                rows.extend(wrap_cells(f"{head}{source}" if not index else source, line_width))
            shown = rows[:INPUT_MAX_LINES]
            for index, line in enumerate(shown):
                row.append("\n" + indent, style=dim)
                if index == 0 and line.startswith(head):
                    row.append(head, style=label)
                    row.append(line[len(head) :], style=body)
                else:
                    row.append(line, style=body)
            hidden = len(rows) - len(shown)
            if hidden > 0:
                row.append("\n" + indent, style=dim)
                marker = f"… {hidden} more line{'s' if hidden != 1 else ''}"
                row.append(truncate_cells(marker, line_width), style=dim)

    def _append_output_body(self, row: Text, width: int) -> None:
        """The plain-result expansion (bash/read/etc.): one line per row.

        The output block reuses the card's own inner padding budget and
        truncates per line: one output line is one row, so the expanded
        height is exactly what the marker promises and never reflows.
        """
        dim = bindings.style("tool.output.dim")
        body = bindings.style("tool.output.error") if self._state == "error" else dim
        line_width = max(1, width - 2 - OUTPUT_INDENT)
        indent = " " * OUTPUT_INDENT
        shown = self._output[:EXPAND_MAX_LINES]
        for line in shown:
            row.append("\n" + indent, style=dim)
            row.append(truncate_cells(line, line_width), style=body)
        hidden = len(self._output) - len(shown)
        if hidden > 0:
            marker = f"… {hidden} more line{'s' if hidden != 1 else ''}"
            row.append("\n" + indent, style=dim)
            row.append(truncate_cells(marker, line_width), style=dim)

    def _append_search_body(self, row: Text, width: int) -> None:
        """Search expansion hierarchy: titles lead, URLs signal, snippets recede."""
        fg = bindings.style("tool.search.title")
        signal = bindings.style("tool.search.url")
        muted = bindings.style("tool.search.snippet")
        dim = bindings.style("tool.search.dim")
        line_width = max(1, width - 2 - OUTPUT_INDENT)
        indent = " " * OUTPUT_INDENT
        shown = self._output[:EXPAND_MAX_LINES]
        for line in shown:
            stripped = line.strip()
            if stripped.startswith(("http://", "https://")):
                ink = signal
            elif stripped[:1].isdigit() and ". " in stripped:
                ink = fg
            elif stripped.startswith(("Provider:", "Sources:")):
                ink = dim
            else:
                # See `bindings.BY_ELEMENT["tool.search.snippet"].note`.
                ink = muted
            row.append("\n" + indent, style=dim)
            row.append(truncate_cells(line, line_width), style=ink)
        hidden = len(self._output) - len(shown)
        if hidden > 0:
            marker = f"… {hidden} more search line{'s' if hidden != 1 else ''}"
            row.append("\n" + indent, style=dim)
            row.append(truncate_cells(marker, line_width), style=dim)

    def _append_fetch_body(self, row: Text, width: int) -> None:
        """Fetch expansion: the header rows read as metadata, the body as content.

        The header rows (Fetched:/Rendered:/the low-quality nudge) recede to dim
        so the rendered preview beneath them \u2014 the actual page content \u2014 reads at
        normal contrast, the same hierarchy the search card uses to keep its
        structural rows from competing with the result.
        """
        signal = bindings.style("tool.fetch.signal")
        muted = bindings.style("tool.fetch.snippet")
        dim = bindings.style("tool.fetch.dim")
        danger = bindings.style("tool.fetch.error")
        line_width = max(1, width - 2 - OUTPUT_INDENT)
        indent = " " * OUTPUT_INDENT
        shown = self._output[:EXPAND_MAX_LINES]
        for line in shown:
            stripped = line.strip()
            if stripped.startswith("⚠ HTTP"):
                # See `bindings.BY_ELEMENT["tool.fetch.error"].note` (F1).
                ink = danger
            elif stripped.startswith("Fetched:"):
                # See `bindings.BY_ELEMENT["tool.fetch.signal"].note` (D3).
                self._append_fetched_row(row, line, line_width, indent, dim, signal)
                continue
            elif stripped.startswith("Rendered:"):
                ink = dim
            elif stripped.startswith("sparse/JS-gated"):
                # See `bindings.BY_ELEMENT["tool.fetch.signal"].note`.
                ink = signal
            else:
                ink = muted
            row.append("\n" + indent, style=dim)
            row.append(truncate_cells(line, line_width), style=ink)
        hidden = len(self._output) - len(shown)
        if hidden > 0:
            marker = f"… {hidden} more line{'s' if hidden != 1 else ''}"
            row.append("\n" + indent, style=dim)
            row.append(truncate_cells(marker, line_width), style=dim)

    def _append_fetched_row(
        self,
        row: Text,
        line: str,
        line_width: int,
        indent: str,
        dim: Style,
        signal: Style,
    ) -> None:
        """Paint the ``Fetched:`` row with its URL(s) in ``signal``, rest dim (D3).

        The row is ``Fetched: <url>[  (final: <url>)][  ·  <meta>]``. URLs are
        lifted to the anchor colour; the ``Fetched:``/``(final: )`` labels and the
        ``·``-joined status/ctype/cache metadata stay dim. Tokens are measured
        against the shared ``line_width`` budget so this row obeys the same width
        law as every other painted line — once the budget is spent the rest of the
        row is dropped rather than wrapped, matching the card's one-pass model.
        """
        row.append("\n" + indent, style=dim)
        remaining = line_width
        # Split on spaces but keep them, so reconstructed spacing matches the
        # source line exactly; a URL token is any run starting with a scheme.
        for token in re.split(r"(\s+)", line):
            if remaining <= 0 or not token:
                break
            painted = truncate_cells(token, remaining)
            if not painted:
                break
            ink = signal if token.startswith(("http://", "https://")) else dim
            row.append(painted, style=ink)
            remaining -= cell_len(painted)

    def _append_diff_body(self, row: Text, width: int) -> None:
        """The write/edit expansion: the unified diff, coloured by hunk line.

        ``+`` added in the success green, ``-`` removed in danger, ``@@``
        hunk markers muted, context lines dim — the same ink law as the
        counters in the summary row, so the pill on the one-line summary and
        the expanded body tell the same story. Only the leading marker
        character is coloured here; the text rides the card's default so a
        coloured line never reads as a wall of tint. The nameless ``---/+++``
        file headers are filtered out below rather than tinted.
        """
        success = bindings.style("tool.diff.added")
        danger = bindings.style("tool.diff.removed")
        muted = bindings.style("tool.diff.hunk")
        dim = bindings.style("tool.diff.context")
        line_width = max(1, width - 2 - OUTPUT_INDENT)
        indent = " " * OUTPUT_INDENT
        # The `---`/`+++` file headers are dropped: the tool diffs one file's
        # before/after in memory, so difflib emits them NAMELESS (`--- ` /
        # `+++ `) and the path already heads the summary row. Two blank-label
        # rows above every diff were pure chrome. Stripped POSITIONALLY — the
        # first two lines, and only when they are exactly the nameless header
        # pair — never by pattern over the body: a removed content line that
        # itself begins `--` (a SQL/Lua comment, say) renders as `--- …` inside
        # the body, and a pattern filter would silently delete the very record
        # this expansion now solely carries (review round 1, F1/D1).
        diff = list(self._diff or [])
        if len(diff) >= 2 and diff[0].rstrip() == "---" and diff[1].rstrip() == "+++":
            diff = diff[2:]
        shown = diff[:EXPAND_MAX_LINES]
        for raw in shown:
            line = raw.rstrip()
            prefix = line[:1] if line else ""
            if prefix == "@":
                ink = muted
            elif prefix == "+":
                ink = success
            elif prefix == "-":
                ink = danger
            else:
                ink = dim
            row.append("\n" + indent, style=dim)
            row.append(truncate_cells(line, line_width), style=ink)
        hidden = len(diff) - len(shown)
        if hidden > 0:
            marker = f"… {hidden} more diff line{'s' if hidden != 1 else ''}"
            row.append("\n" + indent, style=dim)
            row.append(truncate_cells(marker, line_width), style=dim)

    def _name_col(self, width: int) -> int:
        """The ledger's shared name column, in cells.

        Read from the transcript rather than fixed here, because the column is a
        SPINE: every card has to agree on it or the ledger stops being a column.
        The transcript widens it to fit the longest name on screen when the frame
        can afford it, so two MCP tools sharing a seven-character prefix stay
        distinguishable at the widths where there is obviously room.
        """
        parent = self.parent
        if isinstance(parent, TranscriptView) and width >= NAME_GROWTH_MIN_ROW:
            return parent.tool_name_col
        return NAME_COL

    def _build_row(self, width: int) -> Text:
        """The single summary row — the ONE-LINE guarantee lives here."""
        dim = bindings.style("tool.row.dim")
        # See `bindings.BY_ELEMENT["tool.row.name_running"].note` for the
        # two-step fade this implements.
        running = self._state in ("running", "composing")
        # Liveness outranks identity: a live row keeps the fade's green, and
        # the category hue applies only once the row has settled. Two signals
        # on one span would mean the ledger said "what kind" and "is it live"
        # in the same place, which is how the outcome column lost its own
        # meaning before D12 was narrowed.
        name_style = (
            bindings.style("tool.row.name_running")
            if running
            else bindings.style(_category_element(self.tool_name))
        )
        summary_element = "tool.row.summary_running" if running else "tool.row.summary_settled"
        summary_style = bindings.style(summary_element)
        width = max(width - 2, 10)  # 1-cell inner padding each side (kit rule)

        # Status segment (right-aligned), capped at width // 3 (D8) and then
        # hard-clamped so no state can ever push the row past its card.
        status_cap = max(8, width // 3)
        status_runs = _clamp_runs(self._status_runs(status_cap), max(0, width - 3))
        status_cells = sum(cell_len(text) for text, _style in status_runs)

        # Prefix: icon + space + name column + space. The status segment and
        # both separator cells are part of the summary budget (TUI-018). The
        # name column is cell-width bound and ADAPTIVE: below the full column
        # width the name shrinks (truncated by CELL width — len() on a wide
        # CJK/emoji name would break the spine) before the row overflows its
        # card and clips the status off-screen.
        icon = tool_icon(self.tool_name)
        label = display_name(self.tool_name)
        name_budget = width - (2 + status_cells + 2)
        if name_budget < 2 and self._state in ("error", "interrupted"):
            # The status is the only segment carrying free text, so it is the
            # one that gives way first. A failing row used to be the ONLY row
            # in a narrow ledger to lose its name — its neighbours kept three
            # cells of identity while it rendered `<icon>  … ✗ 0.0s`, spending
            # a cell on an ellipsis to say nothing and dropping the one fact
            # that makes the row worth reading: WHICH tool failed.
            status_runs = _clamp_runs(self._status_runs(status_cap, terse=True), max(0, width - 3))
            status_cells = sum(cell_len(text) for text, _style in status_runs)
            name_budget = width - (2 + status_cells + 2)
        if name_budget < 2:
            # Too narrow for even a shrunken name: degrade to icon + status
            # so the outcome column survives. This is the last rung.
            row = _row_text()
            row.append(icon + " ", style=dim)
            if status_runs:
                used = cell_len(row.plain)
                pad = max(1, width - used - status_cells)
                row.append(" " * pad, style=dim)
                for text, style in status_runs:
                    row.append(text, style=style)
            return row

        name_col = min(self._name_col(width), name_budget)
        name = truncate_cells(label, name_col)
        name = name + " " * max(0, name_col - cell_len(name))
        prefix_cells = 2 + name_col + 1

        # The trailing slot holds ONE of two things, never both: the expand
        # affordance when there is output to reveal, or the one-shot answer
        # when the user activated a row that has none. They are mutually
        # exclusive by construction — a notice is only ever set on a row that
        # cannot expand — so they share a budget and a column.
        #
        # They do NOT share a priority. The affordance is an OFFER: chrome,
        # and the D8 floor drops it before it may eat into the summary. The
        # notice is an ANSWER to a keystroke the user just pressed, and it
        # outranks the summary — which on a no-output row is the least
        # interesting thing present, has not changed, and is still readable
        # underneath. Treating the two the same is how the ORIGINAL bug came
        # back: at 46 columns the floor dropped the notice too, so activating
        # an inert row left a byte-identical frame.
        slot = ""
        slot_element = "tool.row.slot_offer"
        remaining = max(0, width - prefix_cells - status_cells - 2)
        if self.can_expand():
            # See `bindings.BY_ELEMENT["tool.row.slot_offer"].note`.
            if (
                self.tool_name.lower() == "web_search"
                or self._is_fetch_card
                or self._hovered
                or self._focused
            ):
                offer = COLLAPSE_HINT if self._expanded else EXPAND_HINT
                if remaining - (cell_len(offer) + 1) >= _SUMMARY_FLOOR:
                    slot = offer
        elif self._notice:
            # See `bindings.BY_ELEMENT["tool.row.slot_notice"].note`.
            slot_element = "tool.row.slot_notice"
            for rung in NOTICE_LADDER.get(self._notice, (self._notice,)):
                if cell_len(rung) + 1 <= remaining:
                    slot = rung
                    break
        slot_cells = cell_len(slot) + 1 if slot else 0
        budget = max(0, remaining - slot_cells)
        # The ladder below belongs to the compose-shaped summary ALONE, and the
        # state test is what says so. `_compose_facts` was the only gate, and it
        # outlived the state that set it: every row that had ever composed took
        # this branch for the rest of its life, where "shed the label, keep the
        # facts" means "shed the COMMAND, keep a byte count from a minute ago".
        # `begin_running` now clears the field, and this makes the invariant
        # checkable from the render side too, so the next way to leave it set
        # cannot reopen the same hole.
        # A user-run receipt leads with a five-cell attribution chip. It sits
        # INSIDE the summary budget so the truncation ladder keeps it ahead of
        # the command — scrolling back to find your own `! git status` works
        # exactly when the row is narrow enough to hide everything else.
        if self.user_run:
            chip = "you: "
            chip_cells = cell_len(chip)
            if chip_cells < budget:
                row_chip = chip
                budget -= chip_cells
            else:
                row_chip = ""
        else:
            row_chip = ""
        composed = self._compose_facts and self._state in ("composing", "interrupted")
        if composed:
            # The label is shed WHOLE before the facts are touched, the same
            # ladder this file uses for the key hints and the approval clause.
            # Truncating instead protected `composing…` — seventeen cells that
            # say nothing new next to a row that is already visibly live — while
            # cutting the byte count and the clock, which are the only two
            # things on the row that move.
            #
            # The trigger is the CARD's width, not the residual budget, because
            # the budget is not monotone in the width: the status segment's cap
            # (`max(8, width // 3)`) crosses the reason's length on the way down
            # and hands cells BACK, so a budget test shed the label at 66,
            # restored it at 62 and shed it again at 54. A width the full form
            # needs is a constant, so comparing against it can only flip once —
            # the same argument `_verbose_min_width` makes for the approval row.
            summary = self._summary
            if width < self._label_min_width() or cell_len(summary) > budget:
                summary = truncate_cells(self._compose_facts, budget)
        else:
            # The IDENTITY, in every other state, truncated by the one rule the
            # settled row uses — which is the whole of the owner's ask: "the
            # command abbreviated within the available horizontal line space for
            # all bash/command line calls". One rule, so a call cannot read as
            # one thing running and another thing done.
            #
            # Streaming output no longer displaces it. It used to, and there was
            # no state of the card from which the command could then be read;
            # the stream now goes to the EXPANSION, where the user asks for it.
            summary = truncate_cells(self._summary, budget)

        row = _row_text()
        # See `bindings.BY_ELEMENT["tool.row.icon_running"].note`.
        #
        # A SETTLED icon takes the name's own category ink rather than one
        # flat grey. The name is already category-coded
        # (`tool.row.name_read`/`_mutate`/`_exec`/`_meta`), but the glyph
        # beside it was `dim` for every tool — so the one mark that carries
        # identity by SHAPE was also the one mark with no colour, and a
        # settled ledger read as a wall of grey. Reusing `name_style` keeps
        # icon and name in one ink per category and mints no new token.
        # A running or failed row is unaffected: those branches own the icon.
        icon_style = bindings.style("tool.row.icon_running") if running else name_style
        row.append(icon + " ", style=icon_style)
        row.append(name, style=name_style)
        row.append(" ", style=dim)
        if row_chip:
            # See `bindings.BY_ELEMENT["tool.row.chip_running"].note`.
            chip_element = "tool.row.chip_settled" if not running else "tool.row.chip_running"
            row.append(row_chip, style=bindings.style(chip_element))
        row.append(summary, style=summary_style)

        # ONE right-aligned tail: the slot and the status share a single pad,
        # so the slot is a COLUMN instead of trailing the summary wherever
        # that happens to end. Appended after the summary it landed at a
        # different cell on every row and slid as the summary truncated —
        # jogging left and right under the eye while the outcome beside it
        # was pinned precisely so that would not happen (D27).
        tail_cells = slot_cells + status_cells
        if tail_cells:
            used = cell_len(row.plain)
            row.append(" " * max(1, width - used - tail_cells), style=dim)
            if slot:
                row.append(slot, style=bindings.style(slot_element))
                row.append(" ", style=dim)
            for text, style in status_runs:
                row.append(text, style=style)
        return row

    @property
    def contributes_name(self) -> bool:
        """Whether this row's name may widen the ledger's shared column.

        A call still being DICTATED must not: the name is model-controlled and
        arrives in fragments, so one announced 200-character name took the column
        to its cap, shifted every settled receipt beside it, and kept the width
        after the row settled as `never sent`. The same argument the column already
        makes for a pending approval — a name earns the column when the call it
        names has actually started.
        """
        return self._state != "composing" and self._summary[:10] != "never sent"

    def _ledger_name_col(self) -> int:
        """The shared column's current width, or the fixed floor off-ledger."""
        parent = self.parent
        return parent.tool_name_col if isinstance(parent, TranscriptView) else NAME_COL

    def _label_min_width(self) -> int:
        """Card width at or above which the summary keeps its whole label.

        Counted, not discovered by comparing against a live budget: the budget
        moves non-monotonically as the frame narrows, because the status segment's
        cap sheds its reason on the way down and returns those cells to the
        summary. A width the full row needs is a constant, so the label can only
        flip once — shed as the terminal narrows, never restored.

        Everything the row spends before the summary at its WIDEST: the inner
        padding, icon, name column and their separators, the widest status the
        card can show, and the label itself.
        """
        widest_status = sum(cell_len(text) for text, _style in self._status_runs())
        return (
            2  # inner padding, one cell each side (kit rule)
            + 2  # icon and its separator
            # The WIDEST column the ledger can hand this row: the threshold has
            # to be a constant, so it cannot depend on the frame that is being
            # tested against it.
            + max(NAME_COL, self._ledger_name_col())
            + 1  # separator after the name column
            + cell_len(self._summary)
            + (widest_status + 2 if widest_status else 0)
        )

    def _status_runs(self, cap: int = 0, *, terse: bool = False) -> list[tuple[str, Style]]:
        """The right-aligned status as (text, style) runs (D12/D13/D28).

        Diff counters ride in FRONT of the outcome glyph and are the first
        thing dropped when the cap bites: how a write went is core, how much
        it wrote is meta. ``terse`` goes one step further and drops the
        failure reason too; see :meth:`_outcome_runs`.
        """
        if self._state == "composing":
            # Nothing has RUN, so there is no execution time to report. The
            # dictation clock rides in the summary instead (`_render_composing`)
            # where it is next to the byte count it belongs with.
            return []
        if self._state == "running":
            # D28 said "no trailing glyph until the duration lands", and the
            # glyph still waits — an outcome column may only ever show an
            # outcome. But the DURATION lands immediately, and withholding it
            # left the one row on screen that is actually consuming time as the
            # only row with no time on it. Reported against a call at 34s whose
            # card said nothing while the working line said 34s: the card was
            # not stuck, it was silent, and a silent row and a stuck row are the
            # same frame.
            #
            # Two blanks stand in for `<glyph> `, so the number lands in exactly
            # the column the ✓ beside it will use when it settles. The spine
            # holds and the row does not jump on settling.
            #
            # A REPLAYED live row reports nothing at all. `subagent_view`
            # rebuilds a child's trajectory into cards and leaves the
            # outcome-less ones running, where `_started` is when the PAGE
            # painted the row — so this counted up from zero and reset to zero
            # every time an earlier entry changed. A clock started from the
            # wrong zero is worse than no clock, and the settled rows on that
            # same page already blank the column for exactly this reason.
            elapsed = self._elapsed()
            if elapsed is None:
                return []
            dim = bindings.style("tool.status.running_duration")
            text = format_duration(max(0, int(elapsed)))
            return [("  ", dim), (text.rjust(DURATION_COL), dim)]
        core = self._outcome_runs(cap, terse=terse)
        diff = self._diff_runs()
        if not diff:
            return core
        core_cells = sum(cell_len(text) for text, _style in core)
        diff_cells = sum(cell_len(text) for text, _style in diff)
        if cap and core_cells + diff_cells > cap:
            return core
        return diff + core

    def _diff_runs(self) -> list[tuple[str, Style]]:
        """``+N`` / ``-N`` counters, tinted success/danger. Empty when unknown."""
        runs: list[tuple[str, Style]] = []
        if self._added > 0:
            runs.append((f"+{self._added} ", bindings.style("tool.status.diff_added")))
        if self._removed > 0:
            runs.append((f"-{self._removed} ", bindings.style("tool.status.diff_removed")))
        return runs

    def _outcome_runs(self, cap: int = 0, *, terse: bool = False) -> list[tuple[str, Style]]:
        """The settled outcome as runs: ``[reason] <glyph> <duration>``.

        ONE shape for all three settled states, which is what makes the
        column a column. The duration is right-justified into
        :data:`DURATION_COL` and the glyph sits immediately in front of it,
        so ``✓``, ``✗`` and ``⊘`` land on the same cell whatever preceded
        them and whether the tool took 0.4s or 12.3s. A column of pass/fail
        marks that wobbles by a cell per row is a column the eye has to read
        instead of scan — which defeats the reason the status segment is
        right-aligned in the first place.

        Interrupted used to be the exception: it returned ``⊘ interrupted``
        with no duration at all and sat six cells left of its neighbours.
        That is the worst row to leave out of the column, because one Esc
        marks EVERY tool still in flight, so the hole opened across a whole
        run of rows exactly where an operator scans to find where work
        stopped.

        ``terse`` drops the reason text. The caller reaches for it when the
        row is too narrow to keep both a tool NAME and a message: identity
        outranks explanation, because a row that cannot say which tool failed
        has stopped being a ledger entry.
        """
        dim = bindings.style("tool.status.duration")
        if self._duration is None:
            # A REPLAYED row: the transcript records what a tool did, never how
            # long it took. `self._duration or 0.0` rendered that as `0.0s`,
            # which is not a missing number, it is a wrong one — it says every
            # tool in a resumed conversation returned instantly. Blank keeps
            # the column aligned and says nothing, which is the truth.
            duration = " " * DURATION_COL
        else:
            elapsed = self._duration
            # Sub-second precision where it distinguishes tools, and the SAME
            # grammar as everything else past a minute: the composing row above
            # this one says `1m57s`, and `117s` two seconds later on the same
            # row is the app disagreeing with itself about how it writes a
            # duration.
            if elapsed < 10:
                duration = f"{elapsed:.1f}s"
            elif elapsed < 60:
                duration = f"{elapsed:.0f}s"
            else:
                duration = format_duration(elapsed)
            duration = duration.rjust(DURATION_COL)
        if self._state == "success":
            # See `bindings.BY_ELEMENT["tool.status.success_glyph"].note`.
            success_ink = bindings.style("tool.status.success_glyph")
            return [(f"{ICON_SUCCESS} ", success_ink), (duration, dim)]

        # The GLYPH sits in the status column with the duration, not at the head
        # of the reason (D20). The right edge is where an operator scans a run
        # of tool rows for pass/fail, and putting `✗` before a right-aligned
        # message moved the failed row's glyph ~25 cells left of the `✓` on every
        # neighbouring row — so the scan found a hole exactly where the answer
        # should be. The reason keeps the space to the glyph's left.
        #
        # `abbreviates` is what separates the two failing states. An error
        # message is CONTENT: `internal error: worker di…` still carries the
        # failure even cut short, so it truncates. "interrupted" is not — it
        # is a constant restating what ⊘ already means, so truncating it buys
        # nothing and costs discrimination: at 46 columns a real failure and
        # a user stop both painted `inte…`, identical in the leftmost and
        # longest column, which is the one the eye lands on first. Below the
        # width that holds the word whole it is dropped, and the glyph column
        # carries the state alone — which it can, being distinguishable from
        # ✓ and ✗ with no colour at all.
        if self._state == "interrupted":
            # `bindings.BY_ELEMENT["tool.status.interrupted"]` deliberately
            # keeps `dim`, not a hue: see its note.
            glyph, reason = ICON_INTERRUPTED, "interrupted"
            tint = bindings.style("tool.status.interrupted")
            abbreviates = False
        else:
            danger = bindings.style("tool.status.error_glyph")
            glyph, reason, tint = ICON_ERROR, self._error, danger
            abbreviates = True

        runs: list[tuple[str, Style]] = []
        if not terse and reason:
            if cap:
                # Uncapped, the caller clamps downstream and the reason rides
                # whole; capped, it has to fit in front of the glyph.
                room = max(0, cap - cell_len(f"{glyph}  ") - cell_len(duration))
                if abbreviates:
                    reason = truncate_cells(reason, room)
                elif cell_len(reason) > room:
                    reason = ""
            # A reason cut down to a bare ellipsis is a cell spent saying
            # "there were words here". Drop it and give the cell back to the
            # columns that still mean something.
            if reason and reason != "…":
                runs.append((f"{reason} ", tint))
        runs.append((f"{glyph} ", tint))
        runs.append((duration, dim))
        return runs

    # -- FINALIZED-BLOCK protocol -------------------------------------------
    def settled_rows(self) -> int:
        """Rows settled now: one collapsed, the whole card when expanded."""
        return self._row_count if self._finalized else 0

    def spans_multiple_rows(self) -> bool:
        """Exact: the card already tracks its own height, collapsed or not."""
        return self._row_count > 1
