"""Transient overlay notice — the app's one toast slot.

A toast reports something that happened without the user asking (MCP servers
came up, one of them did not) and then gets out of the way. Two properties are
load-bearing:

**SINGLE SLOT.** There is exactly one toast, and showing a new one replaces
the current one and cancels its timer. Stacking is the failure mode this
avoids: three servers failing must produce one summary line, not a column of
three cards marching down over the transcript. The reference does the same —
its ``currentToast`` is a single nullable value, not a queue.

**BORDERLESS FILLED CARD.** Per the minimalism mandate, elevation is a
background step, never a border or a shadow. The toast takes the ramp's top
step (``$lo-overlay``) because it floats above everything, which is exactly
the treatment ``ToolCard`` already uses for a filled card one step above its
ground.

Placement is top-right, following the reference. Bottom-right — Textual's own
default for notifications — is unavailable here: the input dock and the status
band are docked to the bottom, so a bottom-right toast would land on top of
the editor the user is typing into.

Timers: the auto-dismiss is a real ``set_timer``, and it is stopped on
dismissal AND on unmount. A Textual timer that outlives its widget is both a
shutdown warning and a test flake, and this suite has already been debugged
for exactly that once.
"""

from __future__ import annotations

from rich.cells import cell_len
from rich.style import Style
from rich.text import Text
from textual.screen import Screen
from textual.widgets import Static

from local_operator.session.mcp_status import McpStartupOutcome
from local_operator.tui import theme as theme_mod
from local_operator.tui.widgets.status_line import ICON_MCP, McpStatus, mcp_semantic
from local_operator.tui.widgets.tool_card import truncate_cells

#: How long a toast stays up. Both values come from the reference's observed
#: variants (3 s warnings / 5 s errors / 10 s update-failed); the split here is
#: by ACTIONABILITY rather than by severity name. A success summary is a
#: courtesy the user can ignore, so it takes the 5 s default. A failure names a
#: server and an error the user has to read, probably re-read, and then act on,
#: so it holds for 10 s — the longest variant the reference uses.
TOAST_DEFAULT_MS = 5000
TOAST_FAILURE_MS = 10000

#: Width cap, from the reference's ``maxWidth = min(60, width - 6)``. 60 cells
#: is a comfortable measure for two lines of prose; the reserve keeps the card
#: from growing to the full terminal on a wide screen, where a 200-cell toast
#: would read as a banner rather than as a note. The floor exists so a 24-cell
#: terminal still gets a card instead of a negative clamp.
TOAST_MAX_WIDTH = 60
TOAST_WIDTH_RESERVE = 6
TOAST_MIN_WIDTH = 20

#: The card's own left+right padding (see the tcss rule), subtracted to get the
#: cells actually available to text.
TOAST_PADDING_CELLS = 2

#: The screen's own edge padding (one cell each side, see the `Screen` rule), so
#: the card's outer width can be clamped to the box it is actually painted in.
TOAST_SCREEN_PADDING_CELLS = 2


def toast_max_width(terminal_width: int) -> int:
    """The card's outer width for a terminal of ``terminal_width`` cells.

    The FLOOR is clamped by the box, not just by the reserve. A bare
    ``max(MIN, …)`` made the card 20 cells wide on a 16-cell terminal, and a
    card wider than the screen is hard-clipped by the compositor — which eats
    the ellipsis ``truncate_cells`` put there, so the one cue that text was cut
    is the first thing lost. Nothing may paint outside the screen's own edge
    padding, so that is the last word on width.
    """
    return max(
        1,
        min(
            TOAST_MAX_WIDTH,
            max(TOAST_MIN_WIDTH, terminal_width - TOAST_WIDTH_RESERVE),
            terminal_width - TOAST_SCREEN_PADDING_CELLS,
        ),
    )


def format_mcp_startup(
    outcome: McpStartupOutcome,
    max_cells: int = TOAST_MAX_WIDTH - TOAST_PADDING_CELLS,
) -> tuple[Text, int] | None:
    """The startup summary as ``(text, duration_ms)``, or ``None`` to stay quiet.

    ONE coalesced message, never one per server: the toast is an overlay over
    the user's work, so it says how much came up and how many tools that bought
    on the first line, and names what failed on the second. At most two lines.

    Multiple failures are NAMED but not explained here — the full error text per
    server would run to a paragraph, and it is already in the transcript notice
    and in ``/mcp``, both of which survive the dismissal. A single failure does
    carry its error, because that is the common case and the one where the fix
    is usually visible in the message ("command not found: gh").

    Returns ``None`` when there is nothing worth interrupting for; see
    :attr:`McpStartupOutcome.reportable`.
    """
    if not outcome.reportable:
        return None

    connected = len(outcome.connected)
    total = len(outcome.configured)
    tools = outcome.tool_count

    text = Text()
    text.append(f"{ICON_MCP} ", style=Style(color=theme_mod.semantic_color(_semantic(outcome))))
    if not total:
        # Discovery itself failed, so there is no server tally to report — the
        # config layer never produced one. "0 of 0 servers up" would be both
        # meaningless and quietly wrong about what broke.
        head = "MCP discovery failed"
    elif outcome.failed:
        head = f"MCP: {connected} of {total} {_plural(total, 'server')} up"
    else:
        head = f"MCP ready: {connected} {_plural(connected, 'server')}"
    if tools:
        head += f", {tools} {_plural(tools, 'tool')}"
    # ``- 2`` for the glyph and its trailing space, which the head shares a row
    # with: budgeting the full width here would let the first line wrap and turn
    # a two-line note into three.
    text.append(
        truncate_cells(head, max(1, max_cells - 2)),
        style=Style(color=theme_mod.semantic_color("fg")),
    )

    if outcome.failures:
        names = sorted(outcome.failures)
        if len(names) == 1:
            # Says FAILED, like the multi-server variant: `gh — command not
            # found: gh` never named the state, so the reader had to infer which
            # server the head line meant and read the name twice to do it. Both
            # variants now open on the same word, which is what makes the second
            # line scannable as a failure list rather than as prose.
            detail = f"failed: {names[0]} — {outcome.failures[names[0]]}"
        else:
            detail = "failed: " + ", ".join(names)
        text.append("\n")
        text.append(
            truncate_cells(detail, max(1, max_cells)),
            style=Style(color=theme_mod.semantic_color("danger")),
        )

    return text, TOAST_FAILURE_MS if outcome.failed else TOAST_DEFAULT_MS


def format_notice_toast(
    message: str,
    kind: str,
    max_cells: int = TOAST_MAX_WIDTH - TOAST_PADDING_CELLS,
) -> Text:
    """A warning/error notice bounded to two single-cell-clamped lines.

    Provider payloads are deliberately retained in full for the later
    transcript, but an overlay cannot inherit their unbounded length: one verbose
    HTTP error would otherwise grow a screen-tall card over the splash. Route
    notices put ``Fallback: …`` on their last line; preserve that decision while
    truncating the diagnosis above it.
    """
    lines = [" ".join(line.split()) for line in message.splitlines() if line.strip()]
    diagnosis = lines[0] if lines else "Provider warning"
    fallback = next(
        (line for line in reversed(lines[1:]) if line.lower().startswith("fallback:")),
        lines[1] if len(lines) > 1 else "",
    )
    semantic = "danger" if kind == "error" else "warning"
    glyph = "✗" if kind == "error" else "!"

    text = Text()
    text.append(
        f"{glyph} ",
        style=Style(color=theme_mod.semantic_color(semantic), bold=True),
    )
    text.append(
        truncate_cells(diagnosis, max(1, max_cells - 2)),
        style=Style(color=theme_mod.semantic_color("fg")),
    )
    if fallback:
        text.append("\n")
        text.append(
            truncate_cells(fallback, max(1, max_cells)),
            style=Style(color=theme_mod.semantic_color("fg")),
        )
    return text


def _semantic(outcome: McpStartupOutcome) -> str:
    """The toast's lamp colour, derived through the band's own rule so the two
    surfaces can never disagree about what state they are reporting."""
    return mcp_semantic(
        McpStatus(
            configured=len(outcome.configured),
            connected=len(outcome.connected),
            failed=outcome.failed,
            # No configured servers but a recorded failure IS the discovery
            # failure: the config layer never produced a list. Passing it keeps
            # the derivation total — the band reaches this state through the same
            # flag, so neither surface can tint it differently.
            discovery_failed=not outcome.configured and outcome.failed,
        )
    )


def _plural(count: int, noun: str) -> str:
    """``1 server`` / ``2 servers``. Only for real nouns — the band's ``MCP`` is
    an initialism and stays singular there."""
    return noun if count == 1 else f"{noun}s"


class Toast(Static):
    """The single toast slot: a filled card that hides itself on a timer.

    Mounted once and kept for the app's lifetime rather than mounted per
    message, so showing a toast never has to await a mount and there is no
    window in which two cards exist at once.
    """

    def __init__(self) -> None:
        super().__init__("")
        # Hidden means ``display: none`` — zero rows, so an empty slot cannot
        # reserve a row of the transcript it overlays.
        self.display = False
        self._timer = None
        # The plain text currently showing. Kept alongside the renderable
        # because Textual's content accessor is a version-specific internal, and
        # "what is this toast saying" is a question both the tests and a future
        # caller are entitled to ask without reaching into the framework.
        self._message: str = ""

    def on_mount(self) -> None:
        self._refit()

    def on_resize(self, event) -> None:  # type: ignore[no-untyped-def]
        """Re-clamp after a terminal resize, so the cap tracks the width."""
        self._refit()

    @property
    def message(self) -> str:
        """What the card is saying right now; empty once dismissed."""
        return self._message

    @property
    def content_cells(self) -> int:
        """Cells available to TEXT inside the card at the current width."""
        return max(1, toast_max_width(self.app.size.width) - TOAST_PADDING_CELLS)

    def show(self, text: str | Text, *, duration_ms: int = TOAST_DEFAULT_MS) -> None:
        """Replace whatever is showing, and re-arm the dismissal timer.

        Replacement is the point: the previous timer is stopped before the new
        one is set, so a second toast can never dismiss the first one's
        successor early.
        """
        self._stop_timer()
        self._message = text.plain if isinstance(text, Text) else text
        self.update(text)
        self.display = True
        self._refit()
        self._timer = self.set_timer(duration_ms / 1000, self.dismiss_toast)

    def dismiss_toast(self) -> None:
        """Hide the card and drop its timer (idempotent).

        Named ``dismiss_toast`` rather than ``dismiss`` because ``Widget`` and
        ``Screen`` already use ``dismiss`` for modal results, and shadowing it
        on a plain widget is how a future modal refactor breaks quietly.
        """
        self._stop_timer()
        self.display = False
        self._message = ""
        self.update("")

    def on_unmount(self) -> None:
        """Teardown must not leave a live timer behind (see the module note)."""
        self._stop_timer()

    def on_click(self, event) -> None:  # type: ignore[no-untyped-def]
        """A click dismisses the card early.

        The timer is a floor on how long the notice is READABLE, not a sentence
        the user has to serve: the failure variant holds ten seconds over the
        transcript, and a reader who is done with it should get their screen
        back. The mouse rather than a key: ``CommandPicker.on_click`` already
        establishes click-to-act as this app's affordance, and Esc is already
        spoken for as the picker's "not now".

        ``event.stop()`` because the toast floats on its own layer over the
        transcript — letting the click through would scroll or focus whatever
        happens to sit beneath it, which is not what the user aimed at.
        """
        event.stop()
        self.dismiss_toast()

    def _card_width(self) -> int:
        """The card's outer width for the message it is currently showing.

        Measured from the TEXT rather than read back from ``outer_size``: the
        offset below is applied in the same pass that shows the card, and at
        that point the layout has not run, so ``outer_size`` still describes the
        previous message. The card is ``width: auto``, so this is the same
        arithmetic Textual will do — longest line plus the padding, capped.
        """
        cap = toast_max_width(self.app.size.width)
        if not self._message:
            return cap
        longest = max(cell_len(line) for line in self._message.split("\n"))
        return max(1, min(cap, longest + TOAST_PADDING_CELLS))

    def _refit(self) -> None:
        """Clamp the card, then pull its HOST over to hug it.

        The host used to be ``width: 1fr``. A widget owns its whole region, so a
        full-width host on the toast layer blanked every column of every row it
        covered — the card painted 35 cells of notice and the host erased the
        other 59 cells of transcript with it. Sizing the host to the card and
        offsetting it to the right edge is what confines the damage to the cells
        the notice actually occupies.
        """
        width = self._card_width()
        self.styles.max_width = toast_max_width(self.app.size.width)
        parent = self.parent
        # The HOST is offset, never the screen: a Toast mounted straight onto the
        # screen would otherwise shift the whole app sideways, which is a far
        # louder bug than a left-aligned toast.
        if parent is not None and not isinstance(parent, Screen):
            # Right-aligned by offset rather than by `align-horizontal`, which
            # cannot right-align a host that is only as wide as its child.
            # ``- TOAST_SCREEN_PADDING_CELLS`` lands the card's right edge on the
            # screen's own inset instead of on the terminal's last column.
            parent.styles.offset = (
                max(0, self.app.size.width - TOAST_SCREEN_PADDING_CELLS - width),
                0,
            )

    def _stop_timer(self) -> None:
        if self._timer is not None:
            self._timer.stop()
            self._timer = None
