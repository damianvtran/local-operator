"""Input editor — TextArea inverted to chat semantics.

Textual's TextArea defaults to newline-on-Enter; this product wants
submit-on-Enter. The subclass inverts that and takes the terminal key idioms:

- ``Enter`` submits (posts :class:`EditorSubmitted`); with the command picker
  open it first completes the highlighted command, THEN submits
- ``Shift+Enter`` inserts a newline
- ``Ctrl+C`` copies a live range (posts :class:`EditorCopied`); with no range
  it posts :class:`InterruptRequested` (abort the turn) — never exits
- ``Ctrl+D`` on an EMPTY buffer quits; otherwise it falls through to delete
- ``Up``/``Down`` move the picker's highlight while it is open; otherwise they
  cycle prompt history when the caret sits at the top/bottom edge of the
  buffer, and inside the text they keep their cursor-move meaning
- ``Tab`` completes the highlighted command WITHOUT submitting
- ``Esc`` dismisses the picker, leaving the typed text alone
- ``Option+←``/``Option+→`` (``Ctrl+←``/``Ctrl+→`` off macOS) move the caret by
  WORD, and the ``Shift`` variants select by word. Every encoding a terminal
  can emit for that chord works, including the ``Esc``-prefixed one that used
  to abort the turn; see :meth:`_on_key`'s escape-coalescing block
- ``!`` on an EMPTY composer enters shell mode (the bang is consumed, not
  typed). Enter then runs the buffer as a local shell command instead of
  sending a prompt; Esc or backspace on an empty shell-mode buffer leaves
  the mode. omp and opencode both ship this gesture; the dedicated-mode
  half is opencode's (caret-at-start ``!`` flips a mode, Esc/backspace
  empty leaves it) and the submit-of-a-leading-bang half is omp's, so a
  history recall of ``! ls`` still runs as a command.

Key interception happens in :meth:`_on_key`, which runs BEFORE TextArea's
document-insert path, so a handled key never reaches the buffer. Unhandled
keys fall through to the stock editor behavior.

A drag over the composer SELECTS and does not copy. The transcript copies on
release because a highlight there is read-only text being taken; in the
composer a highlight is usually the first half of a retype or delete, so
copying on release clobbered the clipboard with text the user was discarding.
The copy gesture is explicit: highlight, then Ctrl+C — :meth:`_on_key` routes
the press to :meth:`action_copy` while a real range is live. See
:meth:`Editor._copy_drag` for why no other key can carry it.

Every way of MAKING that highlight has to produce a document selection, since
that is the only state :meth:`action_copy` can read. Three do: a drag, the
shift+arrow/shift+end family ``TextArea`` already binds, and — since
:meth:`_on_click` — double-click for the word and triple-click for the line.
The click chain was the gap the field report found: Textual answers it with a
SCREEN selection, which a ``TextArea`` can neither paint nor copy, so the
gesture users reach for first in an input box highlighted nothing and the
Ctrl+C that followed cleared the draft instead of copying.

``cmd+C`` reaches this widget on SOME terminals, and whether it does is a
property of the terminal rather than of this code. An earlier revision of this
docstring asserted it "cannot be made to" work; that was measured only against
Terminal.app and is FALSE for the terminal the defect was reported from.

The mechanism is the kitty keyboard protocol, which encodes every key as
``CSI <codepoint> ; <modifiers> u`` with modifiers ``1 + bitmask`` and
**Super=8** — so Cmd+c arrives as ``ESC[99;9u`` and Cmd+v as ``ESC[118;9u``.
Textual's ``linux_driver`` already pushes the flags that ask for this
(``ESC[>25u``), so the keys arrive with no driver change here. **Ghostty and
Terminal.app are the two this project measured**, one on each side of the
line; the rest are reported by their own projects and are orientation, not
tested coverage. Reported to implement it: kitty, Ghostty 1.0+, WezTerm
(opt-in), foot, Alacritty 0.13+, contour, xterm.js (opt-in), and more recently
iTerm2 and Windows Terminal. Reported NOT to: **Terminal.app**, xterm, urxvt,
st, PuTTY, Konsole, VTE/GNOME Terminal.

In Terminal.app the two chords deliver ZERO bytes and are unreachable by any
application, so ``Ctrl+C``/``Ctrl+V`` remain the portable baseline and the only
keys this widget can promise everywhere. Both chords are therefore bound: the
Ctrl pair works on every terminal, and the Cmd pair is bound IN ADDITION where
the terminal forwards it. Neither replaces the other. See
``docs/evidence/cmd-chords/MEASURED.md`` for the byte captures behind each
claim.

Because the Terminal.app failure is SILENT — the terminal eats the key, so a
Mac user reaching for cmd+C there sees no toast, no flicker, nothing — the
portable key is advertised where a user will actually meet it: one entry in
``TIPS`` (``widgets/welcome.py``), the splash's rotating hint row. This
paragraph is documentation for us and reaches no user; the tip is the surface
(design round 1, D4, and agent review round 1, R1-6, which caught the docstring
being described as though it were discoverability).

The caret is SOLID and never blinks, and on an EMPTY composer it gets a cell of
its own to the left of the placeholder rather than inverting the placeholder's
first letter; see ``cursor_blink`` in :meth:`__init__` and :meth:`render_line`.

The editor OWNS its :class:`CommandPicker` (built in ``__init__``, mounted by
the app as a sibling below the input row, since the picker must draw outside
the chevron+editor row). One picker always exists, so every completion path —
Tab, Enter, mouse click — runs through the same code with no "no picker
attached" variant to keep in step.

The picker under the field serves two lists. While the command WORD is open it
offers commands; once the word is terminated by a space, ``/model`` hands the
model picker its query and every command the registry marks as taking a value
(``SlashCommand.arguments``) puts the SAME command picker into argument mode
over that command's values — providers for ``/login``, modes for
``/approvals``, this model's rungs for ``/effort``. Which one is live is
decided by parsing the buffer (``slash_context`` versus ``slash_argument``), so
two lists can never be open at once.

Nothing here filters what may be SUBMITTED. The list ranks what is typed and
offers a completion; a user who knows the value types it and presses Enter, and
never sees the difference.
"""

from __future__ import annotations

import asyncio
import base64
import re
import shlex
import time
from bisect import bisect_right
from collections.abc import Iterable, Mapping
from dataclasses import dataclass
from functools import partial
from pathlib import Path
from stat import S_ISREG
from typing import Callable

from rich.cells import cell_len
from rich.segment import Segment
from rich.style import Style as RichStyle
from textual import events
from textual.actions import SkipAction
from textual.binding import Binding
from textual.content import Content
from textual.expand_tabs import expand_tabs_inline
from textual.geometry import Offset
from textual.message import Message
from textual.strip import Strip
from textual.style import Style as ContentStyle
from textual.widgets import TextArea
from textual.widgets.text_area import Edit, EditResult, Selection

from local_operator.clipboard import MAX_CLIPBOARD_READ_BYTES, read_clipboard
from local_operator.harness.types import ImageContent
from local_operator.imaging import bound_image_for_model
from local_operator.media import ImageInfo, sniff_image, sniff_image_file
from local_operator.tui.autocomplete import ArgumentMode, SlashCommand
from local_operator.tui.widgets.command_picker import (
    CommandPicker,
    CompletionMode,
    PickerMode,
    completion_for,
    ghost_for,
    slash_argument,
    slash_argument_context,
    slash_context,
    slash_token_span,
    slash_word,
)
from local_operator.tui.widgets.model_picker import ModelPicker, ModelRow

#: Refused above this, as a plain text paste. Providers cap an image at 5 MB of
#: base64, which is 3.75 MB of bytes; 4 MB of source is comfortably inside that
#: after the 4/3 inflation and still holds any screenshot. The point is that
#: the refusal happens HERE, where it is one visible path in the composer the
#: user can act on, rather than as a provider 400 mid-turn.
MAX_ATTACHMENT_BYTES = 4 * 1024 * 1024

#: How long a clipboard read may run before the composer says it is working.
#:
#: Under this, the pause is short enough to read as key latency and a card
#: would flicker through the shared notice slot on every paste. Over it, the
#: composer is visibly unresponsive and owes the user an explanation: measured
#: reads on a loaded machine are 1.3-2.0 s for an ordinary screenshot, against
#: a 2.0 s hard ceiling (ux round 1, U3).
#:
#: 350 ms is above the ~100 ms threshold at which an interface stops feeling
#: instant and below the fastest real read observed, so a healthy idle-machine
#: paste (0.26 s) still says nothing at all.
PASTE_READING_NOTICE_DELAY_S = 0.35

#: How long the in-flight card stays up once raised, however fast the read then
#: finishes.
#:
#: Without it a read landing just past the delay above shows the card for the
#: few milliseconds between the timer firing and the read returning \u2014 measured
#: at ~42 ms for reads near 400 ms (design round 2, D10). A card that appears
#: and vanishes inside one frame reads as a glitch rather than as an
#: explanation, which is worse than never showing it: the user cannot read it,
#: but they do see something flicker.
#:
#: 400 ms is long enough for the card to be legible at a glance and short
#: enough that it is gone before a user who pasted successfully looks away from
#: their own text. It only ever DELAYS the retirement, never the paste itself:
#: the marker or text lands the instant the read returns, and only the card
#: lingers.
#:
#: The floor binds on the attach / retire-in-finally routes. Failure
#: notices that ``Toast.show`` a replacement card early are not held to
#: it — the slot stays occupied, so there is no flash to prevent. "The
#: floor guarantees 400 ms" is therefore a statement about the progress
#: card's own lifetime, not about every notice that can occupy the slot
#: (issue #422).
PASTE_READING_NOTICE_MIN_S = 0.4

#: A paste is treated as paths only if EVERY segment looks like one. Requiring
#: a separator is what keeps prose out: "see screenshot.png" splits into two
#: segments, one of which has no `/`, so the whole paste stays text.
_PATH_SEGMENT = re.compile(r"^(?:~|\.{0,2}/|/)")


#: One attachment marker. The number is the key into ``Editor._attachments``;
#: the tail (``, 1568x200``) is a label for the user and is matched loosely so
#: changing it later cannot orphan every marker already in a draft.
#:
#: Also the ATOMIC unit for editing: backspace and delete take the whole marker
#: rather than a bracket, because a half-eaten ``[Image #2, 1568x20`` is neither
#: text the user meant nor a reference anything can resolve.
#:
#: The tail excludes ``[`` as well as ``]``: the app only ever writes ``WxH``
#: there, so a bracket inside one can only mean the tail ran past its own
#: marker into the next. Without that, deleting the closing bracket of a stale
#: marker sitting in front of a live one merged the pair into a single match
#: whose start is not where the live marker begins - the chip vanished for ten
#: keystrokes of an ordinary cleanup while the image stayed attached and sent,
#: and the live marker dropped out of the atomic set (design round 20, D12).
#: It also states the assumption `cite`'s ``str.find`` already relies on: a
#: marker cannot contain another marker.
IMAGE_MARKER = re.compile(r"\[Image #([1-9]\d*)(?:,[^\]\n\[]*)?\]")


#: Appended to a marker whose image was downscaled on the way in. One glyph,
#: two cells, and it buys back most of what the honest label costs: every 16:9
#: screenshot now bounds to the same 1568x882, so three different captures
#: pasted together would otherwise read as three identical markers where the
#: source dimensions had told them apart (design round 1, D1). It also answers
#: the question the number itself provokes — "why is this not the size I
#: pasted?" — which no bare figure can.
#:
#: Chosen over spending width on both sizes: the marker sits inline in the
#: user's prompt text and is an atomic editing unit, so it has to stay short.
#: ``IMAGE_MARKER`` already matches it, since the tail is matched loosely for
#: exactly this kind of later change.
RESIZED_MARK = " ↓"


#: How long a BARREN multi-click keeps the next Ctrl+C off the interrupt rung.
#:
#: The claim has to outlive the gesture by long enough for a hand to move from
#: mouse to keyboard and no longer. Measured against the app's own reflex
#: budget: `CLICK_CHAIN_TIME_THRESHOLD` is 0.9 s for a pair of CLICKS, and this
#: is the same order of reaction for the press that follows them.
#:
#: The bound is the whole safety property. A permanent flag would be D17 again —
#: the composer holding a stale claim, the interrupt silently diverted, and the
#: exit ladder unreachable for a user who has forgotten they ever clicked. When
#: it expires, Ctrl+C means exactly what it means on an untouched composer.
BARREN_CLICK_WINDOW_S = 1.5


#: How far a second click may drift and still read as ONE gesture.
#:
#: Textual's chain needs an EXACT cell match, so a fast pair one column apart is
#: two singles (design round 2, D2-1). Two cells is the jitter of a hand on a
#: trackpad, not a deliberate move to another word: at three or more the user is
#: plausibly placing a second caret somewhere else, and this must not swallow
#: that. It only suppresses one interrupt press; it never makes a selection.
#:
#: A RADIUS, applied as `max(abs(dx), abs(dy))`. Read against `Offset.clamped`
#: it was one-sided on both axes — negatives became zero, so any leftward or
#: upward drift measured as no drift at all and armed the claim from arbitrarily
#: far away (agent review round 3, R3-2).
_NEAR_MISS_CELLS = 2


#: How long after a first click a second may land and still read as ONE gesture.
#:
#: Deliberately WIDER than `CLICK_CHAIN_TIME_THRESHOLD` (0.9 s), which is the
#: window Textual's own chain uses. This branch is gated on its own constant
#: because it measures elapsed time from a LATER instant than the chain does:
#: `_on_click` runs after mouse-up and dispatch, a consistent +25 to +28 ms
#: later. Sharing the chain's number therefore closed the protective window ~26
#: ms BEFORE the window it protects, leaving a ~20 ms band where a pair got
#: neither a selection nor protection (design review round 3, D3-1).
#:
#: The margin makes protection OUTLIVE selection by design rather than racing
#: it. It costs nothing when the pair did chain — a chained click never reaches
#: this branch — so the only presses it can absorb are ones that already
#: produced no range.
_NEAR_MISS_WINDOW_S = 1.0


def _was_downscaled(bounded: ImageInfo, source: ImageInfo) -> bool:
    """Did the bound make the image SMALLER, as opposed to merely different?

    Compares pixel counts rather than the ``WxH`` strings. A portrait phone
    photo inside the bounds is EXIF-rotated on the way in, so its dimensions
    change (3024x4032 from 4032x3024) while not one pixel is lost — and a naive
    string comparison marked it ``↓``, asserting a shrink that never happened
    (review round 2, F8). The mark is a claim about fidelity, so it has to be
    tested as one.
    """
    if not source.width or not source.height or not bounded.width or not bounded.height:
        return False
    return bounded.width * bounded.height < source.width * source.height


def _bounded_dimensions(payload: bytes, info: ImageInfo) -> str:
    """The marker's label: ``WxH`` of the bytes actually attached.

    Read back from the BOUNDED payload rather than carried over from ``info``,
    which describes the file on disk. Once the paste path started resizing,
    reusing the source dimensions would print a marker claiming 2560x1440 next
    to a 1568x882 attachment — a receipt for something that was never sent.

    Carries :data:`RESIZED_MARK` when the two differ, so the label still
    distinguishes one paste from another and says why the number moved.

    A header sniff and not a decode: :func:`sniff_image` reads a fixed prefix,
    so this costs microseconds and cannot be made expensive by the payload. It
    is also applied to bytes this process just produced, so failure is not
    expected — but it stays best-effort anyway, degrading to the source label
    and then to no label at all, because a marker is a convenience and must
    never be the reason an attachment is lost.
    """
    bounded = sniff_image(payload)
    if bounded is None or not bounded.dimensions:
        return info.dimensions
    if _was_downscaled(bounded, info):
        return f"{bounded.dimensions}{RESIZED_MARK}"
    return bounded.dimensions


def _marker_indices(text: str) -> list[int]:
    """Every marker number in ``text``, in order, without duplicates.

    One walk of the buffer, shared by everything that has to agree about which
    markers exist. Two implementations of "the markers in this text" is exactly
    what let a restore rebind images one marker left (review round 18).
    """
    indices: list[int] = []
    for match in IMAGE_MARKER.finditer(text):
        index = int(match.group(1))
        if index not in indices:
            indices.append(index)
    return indices


@dataclass(frozen=True)
class Attachment:
    """An image, and the marker text the app wrote to cite it.

    The marker is kept because the NUMBER alone cannot say which citation is
    the app's. A prompt drag-copied out of the transcript and pasted back
    carries `[Image #1, 1568x200]` verbatim, so a draft can hold two citations
    of one number where only one of them is the app's own claim; picking by
    document order gave the chip - and, through the atomic-token gate, the
    editing behaviour - to whichever landed first, which is the impostor as
    soon as the user pastes at the top of the draft (design round 19, D4).

    Kept ON the attachment rather than in a parallel dict so it rides every
    round trip the images already make: the aside stash, the compaction hold,
    `EditorSubmitted`, and the `/reload` hand-back. A parallel dict is the
    thing this feature has repeatedly got wrong.
    """

    image: ImageContent
    #: Exactly the text issued at :meth:`_attach_pasted_images`, e.g.
    #: ``[Image #1, 1568x200]``.
    marker: str


def cite(text: str, index: int, attachment: Attachment) -> tuple[int, int] | None:
    """The span of the APP's citation of ``index`` in ``text``, or ``None``.

    A SPAN, not an offset, because the citation in the buffer is not always the
    recorded marker: the fallback below resolves a differently-tailed one, and
    a caller that assumed `len(attachment.marker)` measured a marker the user
    had lengthened as if it were still short (review round 26).

    Prefers the citation whose text matches the marker byte for byte, and only
    falls back to the first citation of the number when no exact match
    survives. That fallback is deliberate: the marker's tail is matched loosely
    on purpose so that editing `1568x200` cannot orphan a draft, and keying
    only on an exact match would reverse that decision. Degrading to document
    order needs BOTH an impostor and an edited tail, and lands on the previous
    behaviour rather than on something worse.
    """
    exact = text.find(attachment.marker)
    if exact != -1:
        return exact, exact + len(attachment.marker)
    for match in IMAGE_MARKER.finditer(text):
        if int(match.group(1)) == index:
            return match.span()
    return None


def resolve_markers(text: str, attachments: Mapping[int, Attachment]) -> list[ImageContent]:
    """The attachments ``text`` cites, in the order it cites them.

    THE one place a marker becomes an image, shared by the composer and by any
    caller holding a stashed prompt. A second implementation of this walk is
    what let a restore bind images to the wrong markers (review round 18), so
    there is deliberately only one.

    Ordered by where the APP's citation sits, which is the same order and the
    same set the chip paints - "what is chipped is what is sent" is one
    predicate, not two that happen to agree.
    """
    cited = [
        (span[0], index)
        for index, attachment in attachments.items()
        if (span := cite(text, index, attachment)) is not None
    ]
    return [attachments[index].image for _, index in sorted(cited)]


def _marker_runs(
    start: int,
    end: int,
    selected: tuple[int, int],
    caret: int | None,
) -> list[tuple[int, int, bool]]:
    """``[start, end)`` split into painted runs of one style each.

    Three things want a say over the same cells — the chip, the selection over
    part of it, and the caret standing in it — and they interleave in any
    order, so this walks the columns rather than trying to do interval algebra
    on three overlapping ranges. A marker is at most a few dozen characters and
    a composer at most eight rows, so the walk costs nothing worth optimising
    and is checkable by reading it.

    ``caret`` is DROPPED rather than given a style: ``TextArea`` has already
    painted that one cell inverted, and the chip must not paint over it.
    """
    runs: list[tuple[int, int, bool]] = []
    for column in range(start, end):
        if column == caret:
            continue
        is_selected = selected[0] <= column < selected[1]
        if runs and runs[-1][1] == column and runs[-1][2] is is_selected:
            runs[-1] = (runs[-1][0], column + 1, is_selected)
        else:
            runs.append((column, column + 1, is_selected))
    return runs


def _pasted_paths(pasted: str) -> list[str]:
    """The paste read as a list of filesystem paths, or ``[]``.

    Terminals deliver a DROPPED file as its path, shell-quoted, with spaces
    backslash-escaped. ``shlex`` is exactly the grammar they are quoting for,
    so it is what unpicks it — hand-rolled unescaping is how a path with a
    space becomes two paths that do not exist.

    This branch handles drag-and-drop, and it also catches the one terminal
    that synthesises a path for a clipboard IMAGE: **cmux** watches the
    pasteboard, writes ``$TMPDIR/clipboard-<stamp>-<hash>.png``, and
    bracket-pastes that filename. No other emulator does this — Ghostty,
    Terminal.app and iTerm2 all paste text only — so a path is not how a
    clipboard image usually arrives, and :meth:`Editor._attach_clipboard_image`
    is the terminal-independent route (issue #372).

    Newlines separate multi-file drops on some terminals and are inside a
    filename on none of them, so they split first.
    """
    text = pasted.strip()
    if not text or len(text) > 4096:
        # A real path list is short. This bound is what stops a pasted essay
        # being shlex-parsed on the keystroke that pasted it.
        return []
    segments: list[str] = []
    for line in text.splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            parsed = shlex.split(line)
        except ValueError:
            # Unbalanced quotes: prose, not a path list.
            return []
        if not parsed:
            return []
        segments.extend(parsed)
    if not segments:
        return []
    if not all(_PATH_SEGMENT.match(segment) for segment in segments):
        return []
    return [str(Path(segment).expanduser()) for segment in segments]


class EditorSubmitted(Message):
    """Posted when the user submits the editor (Enter without Shift)."""

    def __init__(
        self,
        text: str,
        images: list[ImageContent] | None = None,
        attachments: Mapping[int, Attachment] | None = None,
        *,
        shell: bool = False,
    ) -> None:
        super().__init__()
        self.text = text
        #: Attachments pasted into this prompt, in marker order. Defaulted so
        #: every existing construction site (and every test) keeps working
        #: unchanged — a submit with nothing attached is still the common case.
        self.images = images or []
        #: The same attachments as an index→image MAP, for a handler that may
        #: have to hand this prompt BACK to the composer — held through a
        #: compaction, or restored by `/reload`. It has to ride the message:
        #: `Editor._submit` clears the buffer synchronously right after posting,
        #: and Textual delivers on a later tick, so a handler that re-reads the
        #: widget sees an empty map and silently drops the images (review round
        #: 19). Restores need the numbers, which the ordered list has lost.
        self.attachments = dict(attachments or {})
        #: True when this submit is a SHELL command, not a prompt. The editor
        #: is the authority: it alone knows whether the bang-mode flag was on
        #: at Enter, and the buffer is cleared before the app sees the
        #: message, so the flag has to ride along the same way attachments do.
        #: Defaulted so every existing construction site keeps working.
        self.shell = shell


class EditorCopyStale(Message):
    """Posted when the buffer is edited after a copy receipt was raised.

    The receipt is a claim about text the user can see. Editing that text makes
    the claim false while it is still on screen, so the app drops the card —
    the clipboard is untouched, only the assertion about it is withdrawn.

    Carries nothing: "what I said a moment ago no longer holds" needs no
    payload, and the app decides whether a card of its own is still showing.
    """

    def __init__(self) -> None:
        super().__init__()


#: Name of the app hook the composer calls to raise and retire the in-flight
#: read card. Looked up with ``getattr`` so a host without one (the widget
#: tests, any embedder) simply has no progress card rather than an error.
#:
#: **THE TIMER MUST BE SET ON THIS WIDGET, AND THAT IS THE WHOLE POINT.**
#: ``self.set_timer(...)`` here is load-bearing in a way the call site cannot
#: show, so the mechanism is recorded here rather than left to be rediscovered.
#:
#: The first revision posted an ``EditorPasteReading`` message from the timer \u2014
#: the idiom every other notice in this widget uses \u2014 and the card reached the
#: screen 2-3 ms before the paste it was meant to narrate, at every read
#: duration (2.018 s for a 2.0 s read, 5.020 s for a 5.0 s read; ux round 2,
#: U3). Measured on this exact shape, with the read on a worker thread and the
#: action awaiting it:
#:
#: =============================  ==========  ================================
#: how the callback is scheduled  when it ran  against a 1.5 s read
#: =============================  ==========  ================================
#: ``self.set_timer``             0.153 s      DURING the read  <- what we use
#: ``self.post_message``          1.510 s      after the read
#: ``self.app.call_next``         1.510 s      after the read
#: ``self.app.set_timer``         1.510 s      after the read
#: =============================  ==========  ================================
#:
#: The discriminator is WHICH PUMP the awaited action occupies, not "the
#: Editor's pump is blocked and the App's is free". Bindings dispatch via
#: ``App._check_bindings``, so the awaited ``action_system_paste`` runs on
#: the App's pump; ``MessagePump.set_timer`` wraps every callback in
#: ``partial(self.call_next, callback)``, so ``self.set_timer`` is a
#: ``call_next`` on the EDITOR's pump. ``call_next`` queues onto
#: ``_next_callbacks``, which is flushed by ``_flush_next_callbacks`` after
#: each dispatched message rather than through the message queue — and the
#: Editor is the pump whose flush still runs while this action is
#: suspended on the App. The App's flush does not, which is why
#: ``self.app.set_timer`` and ``self.app.call_next`` both land late. A
#: message posted to THIS widget and handled by THIS widget arrives on
#: time at 0.000 s rather than late: messages take the queue, and the
#: queue that is held is the App's, not the Editor's. The late cases in
#: the table are the ones that occupy the App's pump.
#:
#: The practical consequence, stated plainly because it is the thing a future
#: reader will get wrong: **"simplify" this to ``self.app.set_timer`` and U3
#: returns exactly**, with every test still passing except
#: ``test_the_reading_card_is_on_screen_while_the_read_is_still_running``. The
#: event loop is fine throughout \u2014 timers keep firing \u2014 so the card is
#: verifiably on screen at 0.4 s, 0.7 s and 1.0 s of a 1.3 s read.
#:
#: This rests on Textual internals (``set_timer`` \u2192 ``call_next``, and where
#: ``_flush_next_callbacks`` is driven from) that are not part of its public
#: contract, so it is pinned by a test that asserts VISIBILITY DURING the
#: stall rather than delivery, and would fail if a future Textual changed the
#: scheduling (code round 3, F8).
PASTE_READING_HOOK = "show_clipboard_reading_notice"


class EditorPasteAttached(Message):
    """Posted when a clipboard paste DID attach an image.

    Exists so the app can retire a paste notice that is still held behind an
    actionable card. Without it that notice surfaces when the slot frees and
    contradicts a composer the user can see holding the image (design round 1,
    D3) — the same staleness :class:`EditorCopyStale` answers for the copy
    receipt, and it uses the same machinery.

    Posted by BOTH ingestion routes. An earlier version reasoned that the path
    route "cannot leave a stale notice behind because it never raises one",
    which conflates RAISING a notice with FALSIFYING one: the path route never
    raises the card, but it does make a held one false, and the held card
    belongs to the composer's paste slot regardless of which route filled the
    buffer. That gap left the original stale-toast frame reachable through the
    route cmux users actually hit — and through the very gesture the notice
    recommends (round 2, D8/D3).
    """

    def __init__(self) -> None:
        super().__init__()


class EditorPasteEmpty(Message):
    """Posted when a paste that could only have been an image attached nothing.

    Raised ONLY from the empty-paste branch — the payload the terminal sends
    when ``Cmd+V`` had no text to give, which on macOS means the pasteboard
    held image bytes, a file URL, or nothing at all. It is deliberately NOT
    raised for an ordinary text paste that happens not to be a path: that paste
    inserts its text, so the user can already see what happened, and a notice
    there would fire on every quote pasted into a prompt.

    That narrowness is the whole design of this notice. The reported bug (issue
    #372) is that a failed image paste was indistinguishable from a dead
    keystroke: nothing inserted, nothing said. The empty-paste branch is the
    only place where the user performed a gesture and the composer can end up
    with literally no visible response, so it is the only place that owes them
    one. It fires at most once per keypress, and only for a keypress that
    otherwise produces nothing at all.

    ``reason`` names the outcome, and only where the code genuinely knows it
    (review round 1, D2/U2). The first version said "no image on the clipboard"
    for every case, which is false in two reachable ones: over SSH the
    clipboard was never read, and an oversized screenshot IS on the clipboard.
    Both mislead a user into the one move that cannot help — re-copying.

    Three values, no more, because three is what the code can establish:

    * ``"nothing"`` — the clipboard was read and had nothing attachable on it.
      This is the deliberately vague one: an empty clipboard, a text-only one,
      a missing ``xclip`` and a wedged daemon are indistinguishable by design
      (see :mod:`local_operator.clipboard`) and a message naming a cause here
      would be guessing.
    * ``"unattachable"`` — image BYTES were found and refused: too large even
      after bounding, undecodable, or a decompression bomb.
    * ``"unreadable"`` — a copied FILE was found and could not be attached.
      Distinct from ``"unattachable"`` because the causes differ (a non-image
      file, an unreadable path, a mixed selection hitting the all-or-nothing
      rule) and so does the advice that would help (round 2, D10).
    * ``"remote"`` — the read was refused because the session is remote. Not a
      statement about the clipboard at all.
    * ``"too_large"`` — the clipboard held TEXT past the paste budget. Declined
      rather than truncated, because a silently shortened paste is damage the
      user cannot see until they read back what they sent; named rather than
      collapsed into ``"nothing"``, because the clipboard was not empty (code
      round 2, F7).
    * ``"timeout"`` — the clipboard did not answer inside
      :data:`~local_operator.clipboard.CLIPBOARD_TIMEOUT_S`. Also not a
      statement about what was on it. Split out of ``"nothing"`` because the
      collapse was actively misleading on the shape users hit most: under CPU
      load 8 of 10 reads timed out, and every one told a user holding a valid
      screenshot that their clipboard was empty (ux round 1, U3). A retry is
      the move that helps here and the move that cannot help there, so one
      sentence could not serve both.

    The app owns the wording, the same way :class:`EditorCopyStale` leaves the
    card to the app; this only says which of the three happened.

    THIS NOTICE IS NOT A DISCOVERY SURFACE, and an earlier revision's attempt
    to make it one is recorded here because the reasoning looks right and is
    not. It carried a ``suggest_system_paste`` flag that appended "Try
    ctrl+v." on the empty-paste route only, on the argument that telling a user
    who just pressed ``Ctrl+V`` to press ``Ctrl+V`` reads as the app missing
    the key. The flag was unsalvageable in both directions (design/ux round 1,
    D1/U1):

    * The route it fired on requires a bare ``ESC[200~ ESC[201~``, which only
      cmux and multiplexers send \u2014 and there ``Cmd+V`` already works, so the
      hint reached exactly the users who did not need it.
    * The route where a user IS stranded delivers zero bytes, so no ``Paste``
      event exists and no notice of any kind can be raised.
    * Even on the route it did fire on, ``reason="nothing"`` means the
      clipboard held nothing attachable \u2014 so ``Ctrl+V`` would return the same
      empty answer. The advice could not have helped even where it appeared.

    ``ctrl+v`` is taught ambiently instead, by
    :data:`~local_operator.tui.widgets.welcome.TIPS` and ``/help``, which do
    not depend on an event the terminal never sends.
    """

    def __init__(self, reason: str = "nothing") -> None:
        super().__init__()
        self.reason = reason


class EditorCopied(Message):
    """Posted when a composer drag finishes on a real selection AND copy is armed.

    Posted only by :meth:`Editor.action_copy` (the highlight-then-Ctrl+C
    press). A drag never reaches this: drag-copy is the transcript's
    gesture, not the composer's.

    Carries the text rather than leaving the app to re-read the widget: the
    selection is live state, and by the time a message is delivered a later
    keystroke may already have moved the caret — which, through
    ``TextArea._watch_selection``, is exactly the thing that collapses it. The
    same "ride the message" rule as :class:`EditorSubmitted`'s attachments.

    Separate from ``TextSelected`` (the transcript's copy trigger) because the
    composer never reaches ``Screen.selections`` at all; see
    :meth:`Editor._copy_drag` for why. The app answers both with one
    clipboard write and one toast, so the two gestures stay one behaviour.
    """

    def __init__(self, text: str) -> None:
        super().__init__()
        #: The document text the drag covered, as ``TextArea.selected_text``
        #: reports it at release.
        self.text = text


class InterruptRequested(Message):
    """Posted on Ctrl+C: abort the running turn, never exit the app."""

    def __init__(self) -> None:
        super().__init__()


class StopRequested(Message):
    """Posted on Esc with no picker open: stop whatever the agent is doing.

    Separate from :class:`InterruptRequested` (Ctrl+C) because the app answers
    them differently: Esc first refuses a pending tool-approval prompt, and only
    aborts when there is a turn to abort, while Ctrl+C always interrupts.
    """

    def __init__(self) -> None:
        super().__init__()


class ShellModeChanged(Message):
    """Posted when bang-mode is entered or left.

    The editor owns the mode (it is a property of the composer, like the
    slash picker) and the app owns the chrome that has to follow it — the
    dock class that recolors the chevron. The message is the seam: a
    widget that reached into ``#input-dock`` would invert the dependency
    this module is arranged around.
    """

    def __init__(self, active: bool) -> None:
        super().__init__()
        self.active = active


class EditorQuit(Message):
    """Posted on Ctrl+D with an empty buffer."""

    def __init__(self) -> None:
        super().__init__()


class InlineCommandRequested(Message):
    """Posted when an INLINE slash command is run out of the middle of a draft.

    A command typed at the start of the buffer runs through the ordinary submit
    path (``EditorSubmitted`` with a slash-shaped ``text``): the whole buffer IS
    the command, so submitting and clearing is exactly right. An INLINE command
    is different — the user typed a message and then remembered to route it, so
    the ``/command`` token is spliced OUT of the draft and the rest of the
    message is left in the composer untouched. This message carries just the
    command text (``/team ops``) for the app to dispatch, while the editor keeps
    ownership of the surviving draft.

    Split from ``EditorSubmitted`` deliberately: a submit clears the buffer and
    records history, neither of which an inline run wants — the draft is still
    unsent, and the command is not a prompt the user would page back to.
    """

    def __init__(self, command_text: str) -> None:
        super().__init__()
        self.command_text = command_text


class ModelQueryOpened(Message):
    """Posted when the buffer enters ``/model …`` and the model list appears.

    The editor knows WHEN the list opens (it parses the buffer) but nothing about
    what belongs in it, and the app knows the catalogue but not the keystrokes.
    This message is the seam. It fires on the closed→open TRANSITION only, so
    typing a query does not re-trigger a provider fetch per character.
    """

    def __init__(self) -> None:
        super().__init__()


class ArgumentQueryOpened(Message):
    """Posted when the buffer enters ``/<command> …`` for a list-taking command.

    Carries the command WORD because every list is a different set — every
    loginable provider, only the ones holding a credential, the two approval
    modes, the rungs THIS model accepts — and the app cannot recover which one
    from the buffer once the message is queued.

    Like :class:`ModelQueryOpened` it fires on the TRANSITION only, so the app
    builds the rows once per opening rather than once per keystroke. That is
    what makes a list affordable when filling it costs a credential-store read.
    """

    def __init__(self, command: str) -> None:
        super().__init__()
        self.command = command


class RefreshArgumentChoices(Message):
    """Posted when the rows an open argument list should offer have changed
    UNDER the same command word — which :class:`ArgumentQueryOpened` can never
    see, because it fires on the command-word transition only.

    Exists for two-level arguments: ``/mcp`` offers its subcommands in the
    first argument slot and the servers that subcommand can act on in the
    second, so the choice set changes when the subcommand is completed while
    the command word stands still. The editor tracks the SUBCOMMAND slot and
    posts this when it changes; the app refills the picker exactly like an
    opening. Keeping this per-command and explicit (rather than refilling
    every list per keystroke) preserves the cost argument
    :class:`ArgumentQueryOpened` documents — a credential-store read per
    character typed.
    """

    def __init__(self, command: str) -> None:
        super().__init__()
        self.command = command


class ArgumentHighlightChanged(Message):
    """Posted when the row the user's eye is on in an ARGUMENT list changes.

    Carries the command word and the highlighted value name (``None`` when the
    list closed or emptied). The one consumer is live preview: ``/theme``
    applies the highlighted theme as the user arrows or hovers through the
    rows, and restores the real one when the list goes away. Every other
    command ignores it — previewing is an explicit opt-in on the app side,
    because most argument rows (a credential to remove, a provider to log
    into) have no meaningful "try it on" semantics.
    """

    def __init__(self, command: str, name: str | None) -> None:
        super().__init__()
        self.command = command
        self.name = name


#: The composer's resting prose. Named rather than inlined because the app
#: swaps it for :data:`READ_ONLY_PLACEHOLDER` while the full-page subagent
#: view is open and has to be able to put this one back.
#:
#: It carried a ``ctrl+v pastes an image`` suffix for one release, added as the
#: discoverability answer for a user whose `Cmd+V` was swallowed by the
#: terminal with no way to learn the key that worked. That argument no longer
#: holds: `Cmd+V` is now bound and arrives on every terminal implementing the
#: kitty keyboard protocol, so the Mac gesture a user already knows simply
#: works and needs no teaching. Terminal.app still cannot deliver the chord,
#: and ``/help`` names ``ctrl+v`` as the fallback there — a conditional
#: statement needs room to qualify itself, which a one-line placeholder does
#: not have and a help table does. Removed rather than reworded because
#: permanent composer real estate is the most expensive surface in the app.
DEFAULT_PLACEHOLDER = "Message Local Operator…"

#: What the composer says while it refuses input. It names the state AND the
#: consequence, because the only useful thing to tell someone whose keys are
#: being ignored is how to get a composer that accepts them.
READ_ONLY_PLACEHOLDER = "Read-only — press esc to reply"

#: What the composer says while the `/btw` aside card owns it — what the FIELD
#: does, in ``DEFAULT_PLACEHOLDER``'s shape, and nothing about how to leave.
#:
#: It used to read "Aside — esc returns to the chat". That was right while the
#: card floated 88 cells wide four rows away; once the two share a column and
#: sit one row apart, the card's own pinned footer already says `esc back to
#: the chat` two rows above, and two nearly consecutive rows both opening with
#: `esc`, in two different verbs, read as repetition in a unit whose whole
#: argument is that it is one thing. The card keeps the exit — its footer is
#: chrome and `esc` is never shed from it — and this says what typing here
#: will do. Half the cells, so it also survives a narrow terminal.
ASIDE_PLACEHOLDER = "Ask the aside…"

#: What the composer says in bang-mode. The first clause is opencode's
#: sentence — what typing here WILL DO — and the second names the way out,
#: because entry is taught three ways (tip, placeholder, and the ``$``
#: marker — a shape change, not a hue, so it survives ``NO_COLOR``; #385)
#: while exit was taught by nothing on screen (design round 1, D2): the
#: placeholder is the one surface guaranteed visible the moment the mode
#: opens on an empty buffer, which is exactly when a first-timer looks
#: for the door.
SHELL_PLACEHOLDER = "Run a command… — esc to leave"


class Editor(TextArea):
    """Multiline prompt editor with submit-on-Enter, history, slash-completion."""

    #: Word-wise caret movement on the macOS chord, ADDED to the ``ctrl+arrow``
    #: bindings ``TextArea`` already ships rather than replacing them: the ctrl
    #: chord is the Linux/Windows convention and costs nothing to keep, so both
    #: platforms' muscle memory works in the same build.
    #:
    #: This half only covers terminals that encode option+arrow as a CSI
    #: sequence with modifier 3 (``\x1b[1;3D``), which Textual's parser resolves
    #: to the ``alt+left`` key name. The other two encodings are handled
    #: elsewhere: ``\x1bb``/``\x1bf`` (readline meta) already parse to
    #: ``ctrl+left``/``ctrl+right`` and hit the inherited bindings, and the
    #: ``Esc``-prefixed form arrives as two separate events and is coalesced in
    #: :meth:`_on_key`. All three are pinned in
    #: ``tests/unit/tui/test_word_caret.py``.
    #:
    #: ``show=False`` to match every other binding in this app — the footer is
    #: not a key reference here (see ``OperatorApp.BINDINGS``).
    #:
    #: Textual MERGES a subclass's ``BINDINGS`` with its bases' rather than
    #: overriding them, so declaring this attribute does not cost the editor any
    #: inherited key (verified against textual 8.2.8: the instance's resolved
    #: binding map holds ``alt+left`` AND ``ctrl+left``).
    #: NOTE the vertical chords are deliberately ABSENT from this table. They
    #: are handled in :meth:`_on_key` by :attr:`VERTICAL_CHORD_KEYS` instead,
    #: because a key with a ``Binding`` and NO ``_on_key`` branch reaches the
    #: action and nothing else — which is exactly how a previous revision
    #: destroyed a typed slash command (code round 2 F5, ux round 2 U6). See
    #: that table.
    #: ``ctrl+v`` is THE clipboard paste in this composer, and it OVERRIDES
    #: ``TextArea``'s own ``ctrl+v`` → ``action_paste`` rather than sitting
    #: beside it. Two separate facts make this binding the whole fix for
    #: issue #372 outside cmux, and both were measured rather than assumed:
    #:
    #: 1. **``Cmd+V`` is only reachable on some terminals.** With an image-only
    #:    pasteboard, Terminal.app delivers ZERO bytes on ``Cmd+V`` and beeps —
    #:    not an empty bracketed paste, nothing at all, captured with a
    #:    raw-mode PTY probe — so there it is unreachable by any application.
    #:    Ghostty, which speaks the kitty keyboard protocol, instead FORWARDS
    #:    it as ``ESC[118;9u`` (Super+v), which Textual decodes as ``super+v``.
    #:    ``Ctrl+V`` delivers on both, so it is the portable key; ``super+v``
    #:    is bound beside it below rather than instead of it. An earlier
    #:    revision of this note claimed both terminals sent nothing, which was
    #:    true only of Terminal.app. See
    #:    ``docs/evidence/cmd-chords/MEASURED.md``.
    #: 2. **Textual's ``ctrl+v`` pastes the WRONG buffer.**
    #:    ``TextArea.action_paste`` inserts ``App.clipboard``, Textual's
    #:    INTERNAL buffer, which a copy made in any other application never
    #:    fills. Left in place it makes ``Ctrl+V`` a key that silently does
    #:    nothing on exactly the gesture a user means by it.
    #:
    #: Overriding works because Textual RESOLVES A KEY TO THE MOST DERIVED
    #: BINDING even though it merges the tables: verified against textual
    #: 8.2.8 by inspecting the resolved binding map (``ctrl+v`` maps to this
    #: action alone) and by pressing the key in a pilot (``action_paste`` does
    #: not run). That is the opposite of the ``alt+left`` note above, where
    #: merging was the point — same mechanism, different consequence, because
    #: here the keys collide.
    #:
    #: ``super+v`` rides the SAME binding, and that placement is deliberate.
    #:
    #: THE DISPATCH RULE, measured against textual 8.2.8 rather than assumed,
    #: because an earlier revision of this file stated it backwards and the
    #: wrong version is load-bearing enough to mislead: ``_on_key`` runs
    #: FIRST, and a binding fires only if ``_on_key`` did not consume the key
    #: with ``event.stop()``. A normal ``Binding`` does not bypass ``_on_key``
    #: at all — ``super+v`` enters it exactly like ``ctrl+v``, finds no branch
    #: claiming the key, and falls through to this action. (The one shape that
    #: genuinely skips ``_on_key`` is a PRIORITY binding, which is dispatched
    #: ahead of the focused widget entirely. None here are priority.)
    #:
    #: So the choice is not "binding or ``_on_key``" but whether the press has
    #: to be read against state ``_on_key`` owns. Paste does not: NOTHING in
    #: ``_on_key`` inspects ``ctrl+v``, no other handler claims the key, and no
    #: picker or mode changes what it means. The binding is therefore the whole
    #: behaviour. ``super+c`` is the case that goes the other way — see
    #: :meth:`_on_key`, where it joins ``ctrl+c``.
    #:
    #: Binding it CANNOT double-paste, which is the load-bearing measurement.
    #: Ghostty forwards ``super+v`` only when it has nothing to paste itself:
    #: with TEXT on the clipboard it consumes the key and bracket-pastes (21
    #: bytes, ``ESC[200~…ESC[201~``, no key event), and only with an
    #: image-only pasteboard does the app see ``ESC[118;9u``. The two cases are
    #: disjoint, so this action and the terminal's own paste can never both
    #: run for one press.
    #:
    #: Do NOT recommend ``performable:super+v=paste_from_clipboard`` to make
    #: this more reliable: measured, that prefix makes the image-only case
    #: deliver ZERO bytes, i.e. strictly worse. The default binding already
    #: forwards.
    #:
    #: ``show=False`` to match every other binding in this app; the key is
    #: taught in ``/help`` and in the paste notice instead.
    BINDINGS = [
        Binding("alt+left", "cursor_word_left", "Cursor word left", show=False),
        Binding("alt+right", "cursor_word_right", "Cursor word right", show=False),
        Binding("alt+shift+left", "cursor_word_left(True)", "Select word left", show=False),
        Binding("alt+shift+right", "cursor_word_right(True)", "Select word right", show=False),
        Binding("ctrl+v,super+v", "system_paste", "Paste from the system clipboard", show=False),
    ]

    #: The CSI-modifier spelling of every vertical option chord, mapped to the
    #: plain key it must be INDISTINGUISHABLE from.
    #:
    #: Vertical chords are normalised to their plain arrow at the top of
    #: :meth:`_on_key` rather than bound as actions, and the distinction is
    #: load-bearing. ``up``/``down`` are claimed by FOUR handlers in this widget
    #: — the model picker, the command picker, history navigation, and finally
    #: TextArea's caret move — and every one of them lives inside ``_on_key``
    #: and gates on the literal key name. A bound chord with no ``_on_key``
    #: branch of its own passes straight through those four rungs to its
    #: action, so ``alt+up`` reached only a reimplementation of the last two
    #: and silently closed an open picker, overwriting a half-typed slash
    #: command with a history entry on exactly the terminals where the key had
    #: previously been a harmless no-op.
    #:
    #: Rewriting the key is therefore not a shortcut but the correctness
    #: argument: there is ONE implementation of what ``up`` means, and the chord
    #: cannot drift from it because it does not have its own. Anything added to
    #: the plain-arrow path in future is inherited for free.
    VERTICAL_CHORD_KEYS = {
        "alt+up": "up",
        "alt+down": "down",
        "alt+shift+up": "shift+up",
        "alt+shift+down": "shift+down",
    }

    #: Every key that, arriving on the message pump turn after an ``escape``,
    #: proves that escape was the first half of an option+arrow chord rather
    #: than a press of its own. Maps the key to what the chord should DO.
    #:
    #: ``None`` means "cancel the escape and let the key through to its ordinary
    #: handler". The vertical arrows use it: ``⌥↑``/``⌥↓`` is a real macOS chord
    #: (move by paragraph), so a user who has just learned ``⌥←`` works will try
    #: it, and on an Esc-prefixed terminal that used to stop the turn AND
    #: overwrite the draft from history — strictly worse than the bug this class
    #: set out to fix (ux round 1, U2). They map to ``None`` rather than to a
    #: paragraph action deliberately: this composer's ``up``/``down`` already
    #: carry history navigation and caret movement, and inventing a paragraph
    #: motion here would be a second, competing meaning for the same physical
    #: key. Cancelling the stop and passing the key through gives ``⌥↑`` exactly
    #: the behaviour of ``↑``, which is the conservative reading and never
    #: destroys a turn or a draft.
    #:
    #: An earlier revision excluded the vertical arrows entirely, reasoning that
    #: swallowing an Esc for them would lose a stop. That was written when the
    #: deferral was a 100 ms wall-clock window; at one pump turn the race is
    #: microseconds wide and the same argument that admits ``left``/``right``
    #: admits these.
    #:
    #: Values are UNBOUND METHODS, not action-name strings. A string went
    #: through ``getattr(self, f"action_{name}")`` and could only fail at
    #: keypress time if an action were ever renamed (code round 1, F4); a
    #: direct reference fails at import instead.
    ESCAPE_CHORD_KEYS: dict[str, tuple[Callable[..., None], bool] | None] = {
        "left": (TextArea.action_cursor_word_left, False),
        "right": (TextArea.action_cursor_word_right, False),
        "shift+left": (TextArea.action_cursor_word_left, True),
        "shift+right": (TextArea.action_cursor_word_right, True),
        "up": None,
        "down": None,
        "shift+up": None,
        "shift+down": None,
    }

    #: Maximum remembered prompts.
    HISTORY_LIMIT = 200

    #: Command words whose ARGUMENT is a model selector. Both spellings, because
    #: `/models` is a registered alias and a user who typed the alias should get
    #: the same list.
    MODEL_COMMANDS = ("model", "models")

    #: Commands that DESTROY what a chosen row names. `/logout` removes a
    #: credential and an OAuth one costs another browser round trip to get back,
    #: so its rows are gated harder than the shared ambiguity rule gates the
    #: rest: see :meth:`_picker_choice_is_unambiguous` and :meth:`_apply_command`.
    DESTRUCTIVE_COMMANDS = ("logout",)

    #: Commands whose ARGUMENT is a NAME optionally followed by a free-text
    #: message (`/team <name> <request>`, `/agent <name> <message>`). Completing
    #: the name adds a trailing SPACE and parks the caret after it so the user
    #: can keep typing the message — it does NOT submit, unlike the enum-tail
    #: argument commands (`/login`, `/effort`, the model list) where the
    #: argument IS the whole command and Enter runs it. A blank Enter after that
    #: space is the attach-only switch, and a typed message then Enter sends it:
    #: both already emerge from the tokenizer closing the argument list on the
    #: space and the app's dispatch collapsing a bare name to attach-only, so no
    #: dispatch/registry change is needed — only completion inserts the space.
    #:
    #: A local class tuple mirroring ``MODEL_COMMANDS`` rather than a new
    #: ``ArgumentMode`` member: the mode enum is a shared surface (the mobile
    #: daemon projects it, ``set_commands``/``opens_a_list`` branch on it), and a
    #: new member would force a re-audit of every ``arguments is/==`` site for a
    #: two-command editor-local exception. Both spellings, because the alias is
    #: itself a runnable command — same reason ``MODEL_COMMANDS`` lists
    #: ``models``.
    NAME_ARGUMENT_COMMANDS = ("team", "teams", "agent", "agents")

    #: The discoverability hint shown the moment a NAME+message name is completed
    #: with an empty tail (ux round 1, U1/U2). Names both outcomes of the parked
    #: caret — a blank Enter switches, a typed message sends — so the divergence
    #: from the enum-tail commands (Enter-on-a-row runs immediately) is visible
    #: BEFORE the user commits, not inferred from a transcript notice afterwards.
    #: Shown in the picker's own notice row and withdrawn as soon as a message
    #: character is typed. See :meth:`_name_switch_hint`.
    NAME_SWITCH_HINT = "Enter to switch · type a message to send"

    #: The attachment chip's two grounds, on top of everything ``TextArea``
    #: already declares. Component classes rather than hexes in Python so the
    #: colours sit in the stylesheet beside every other composer colour and
    #: follow the theme's ``$lo-*`` variables through a theme switch.
    COMPONENT_CLASSES = TextArea.COMPONENT_CLASSES | {
        "text-area--image-marker",
        "text-area--image-marker-selected",
        # Slash-command syntax highlighting (see :meth:`_paint_slash`). Component
        # classes rather than Python hexes for the same reason the chip uses
        # them: the colours live in the stylesheet beside every other composer
        # colour and follow the theme's ``$lo-*`` variables through a theme
        # switch for free.
        "text-area--slash-command",  # recognized /command word
        "text-area--slash-argument",  # recognized team/agent NAME
        "text-area--slash-unknown",  # a leading /word that is NOT a command
    }

    def __init__(
        self,
        placeholder: str = DEFAULT_PLACEHOLDER,
        commands: list[SlashCommand] | None = None,
    ) -> None:
        # Built BEFORE super().__init__: TextArea's constructor loads its
        # initial document, which funnels through load_text() and therefore
        # through _sync_picker().
        self._picker = CommandPicker(
            self._apply_command, self._on_picker_highlight, self._on_picker_preview
        )
        self._model_picker = ModelPicker(self._apply_model)
        # Which list-taking command the argument list is currently open for, or
        # None when the buffer is not in one. This is the transition edge the
        # ArgumentQueryOpened message rides: without it the app would rebuild
        # the rows on every character typed into the query. Assigned here for
        # the same reason as the pickers — _sync_picker() reads it during
        # super().__init__().
        self._argument_command: str | None = None
        #: The first argument SLOT of a two-level command (``/mcp login``)
        #: whose choice set depends on it; ``None`` for one-level commands.
        self._argument_subcommand: str | None = None
        # Command words (primaries AND aliases) whose argument opens the value
        # list, and the subset of those the bare command cannot stand without.
        # DERIVED from the registry in :meth:`set_commands` rather than listed
        # here: a hand-kept tuple beside a registry that already states the fact
        # is a second source of truth, and the way it fails is a command whose
        # description advertises options the editor never offers.
        self._argument_commands: tuple[str, ...] = ()
        self._required_argument_commands: tuple[str, ...] = ()
        # Every registered command word (primaries AND aliases), lower-cased,
        # for the highlighter's "is this leading /word a real command?" oracle,
        # AND for the inline tokenizer's nested-slash rule (a slash inside an
        # engaged command's argument is plain text). Derived from the registry in
        # :meth:`set_commands` — the render pass must not import the app (which
        # owns ``slash_command_for``): editor.py is imported BY app.py, so the
        # dependency only goes one way. The same name-in-``names`` membership
        # ``slash_command_for`` uses is reproduced here so the highlight cannot
        # claim a command the dispatch would reject.
        self._command_names: frozenset[str] = frozenset()
        # Commands whose argument is a prompt (goal/loop/team/agent/btw). Same
        # derive-from-registry reason as the tuples above; read during the
        # inline run to decide reassemble-to-front vs splice-and-run.
        self._prompt_commands: tuple[str, ...] = ()
        # The team/agent NAMES the open argument list is offering, pushed by the
        # app in ``on_argument_query_opened`` (see :meth:`set_name_choices`).
        # A frozenset so the render pass tests membership in O(1): the render
        # path must never walk the team/agent registries itself — that is I/O,
        # and ``render_line`` runs on every keystroke-frame. Empty until the
        # list has opened at least once; a name hand-typed in full before the
        # list ever filled goes un-highlighted, an accepted affordance gap.
        self._name_choices: frozenset[str] = frozenset()
        # Which command FAMILY (`team` vs `agent`) the snapshot above was filled
        # for. The two families offer disjoint rosters, so a snapshot is only
        # valid to paint against the family it came from. Tracked because the
        # multiline branch of ``_sync_picker`` preserves the snapshot across a
        # newline, and an atomic word-swap `/team <name>\n…` → `/agent <name>\n…`
        # while already multiline would otherwise keep the team roster in place
        # and paint a team name green under `/agent` (a name that is not a valid
        # agent). ``None`` until a list has filled the snapshot.
        self._name_choices_family: str | None = None
        # Per-render-pass memo for :meth:`_slash_runs` (CR1). ``render_line`` is
        # called once per visible screen row and the runs are identical for every
        # row of a frame; this caches the parse against a key of every input it
        # reads, so the compute runs once and the other rows reuse it. ``None``
        # until the first row of the first frame that asks.
        # The key is a heterogeneous snapshot of every input _slash_runs reads
        # (text, two frozensets, the argument-commands tuple, picker mode/flags),
        # so it is typed ``tuple[object, ...]`` — it is only ever compared for
        # equality, never indexed.
        self._slash_runs_cache: (
            tuple[tuple[object, ...], tuple[int, list[tuple[int, int, str]]] | None] | None
        ) = None
        # Guards the picker resync inside ``load_text`` so ``_set_text_and_caret``
        # can move the caret first and sync ONCE at the final position (D5). Set
        # BEFORE ``super().__init__`` because TextArea's constructor loads the
        # initial document through ``load_text`` → ``_sync_picker``.
        self._suspend_picker_sync = False
        # The escape action held for one pump turn, or ``None`` when no escape
        # is in flight (the resting state). See the escape-coalescing block
        # below :meth:`_on_key` for why an escape is ever held at all.
        # Typed as returning ``object`` because the actions held here are
        # usually ``post_message`` calls, which return ``bool``; the return is
        # discarded either way.
        self._pending_escape: Callable[[], object] | None = None
        # tab_behavior="indent": Tab NEVER moves focus (TUI-013). Command
        # completion consumes the key first; otherwise it indents.
        super().__init__(placeholder=placeholder, soft_wrap=True, tab_behavior="indent")
        # The caret is SOLID and never blinks. Blinking made the caret cell flip
        # twice a second, and on the boot splash — where nothing else repaints —
        # that 2 Hz invert beside a static logo WAS the startup animation users
        # called obnoxious. Blink-off pays a second time: stock blink runs
        # `refresh_lines` every 500 ms for the life of the app, so a completely
        # idle session was writing a row down the ssh pipe twice a second for a
        # caret that had not moved.
        #
        # Blink-off is NOT the same lever as whether a caret is drawn at all.
        # An earlier pass suppressed the caret on an empty buffer, because
        # Textual has nowhere to put one in a cell grid except ON a character:
        # the placeholder branch of `TextArea.render_line` inverts cell 0 of
        # `Message Local Operator…`, so the copy read `▉essage Local Operator…`
        # with the block measuring 13.76:1 against the panel — the loudest thing
        # on the identity screen, sitting on a word (D-05).
        #
        # Suppressing it was the wrong half to give up. It left the composer
        # with NO focus affordance in the state a first-time user meets it in:
        # clicking the empty field changed nothing on the frame, so there was no
        # way to tell whether the next keystroke would land in it. The caret is
        # back, and the collision is resolved where it actually is — the cell
        # grid — by giving the caret its OWN cell and starting the placeholder
        # at column 1 while it is drawn. See :meth:`render_line`.
        self.cursor_blink = False
        self._history: list[str] = []
        self._history_index: int | None = None  # None = not navigating
        # See :meth:`set_records_history`. On by default; the aside turns it
        # off for as long as it owns the composer.
        self._records_history = True
        self._draft: str = ""  # buffer text saved when history nav starts
        #: and its attachments, saved with it. Kept together because they are
        #: one unsent message: restoring the text without these resolves its
        #: markers to nothing, and restoring neither loses the user's work.
        self._draft_attachments: dict[int, Attachment] = {}
        self._on_model_chosen: Callable[[ModelRow], None] | None = None
        #: Attachments pasted into the current draft, keyed by the number in
        #: their ``[Image #N, WxH]`` marker. A DICT rather than a list because
        #: the marker in the text is the authority on what gets sent: deleting
        #: it deletes the attachment, so the two cannot be kept in step by
        #: position. A number in use is never handed out twice, so nothing
        #: renumbers under the cursor; a number the draft has stopped citing
        #: IS reused, which the next comment explains.
        self._attachments: dict[int, Attachment] = {}
        #: Next marker number to hand out. Fully DERIVED — `_sync_next_marker`
        #: recomputes it from the buffer immediately before every issuance, so
        #: the field only carries the value between those two lines and the
        #: other writes to it cannot affect any number actually issued.
        #:
        #: A consequence worth stating because it reverses what this comment
        #: used to claim: numbers are no longer monotonic. Paste, paste, delete
        #: #2, paste now yields #1 and #2 rather than #1 and #3. Nothing
        #: renumbers — the surviving markers do not move, which is the property
        #: the old monotonic rule existed to protect — the freed label is
        #: simply available again. Safe by construction: the sync returns
        #: `max(cited) + 1`, which cannot collide with a number the text still
        #: cites (review round 20).
        self._next_marker = 1
        #: The marker a mouse press landed inside, as ``(row, start, end)``,
        #: held until the button comes back up. The press alone cannot decide:
        #: the same press also arms ``TextArea``'s drag-selection, and a drag
        #: that begins inside a marker has to stay a drag. See
        #: :meth:`_on_mouse_up`.
        self._pressed_marker: tuple[int, int, int] | None = None
        #: Whether this widget has announced a copy that the buffer has not yet
        #: been edited past. The receipt on screen is a claim about text the
        #: user can see, so the first edit after a copy withdraws it. A flag and
        #: not the text itself: nothing reads the content, and holding the
        #: user's copied draft on the widget buys nothing (review round 2, F9).
        #: See :meth:`edit` and :meth:`load_text`.
        self._copied = False
        #: The range the CLICK CHAIN made, while it is still the live one.
        #:
        #: A double-click is reflexive where a drag or shift+arrow is
        #: deliberate, so the two get different answers on the second Ctrl+C:
        #: an accidental range gives the key back to the exit ladder once it
        #: has been copied, a deliberate one keeps copying (R1-2). Holding the
        #: SELECTION rather than a bare flag is what makes "still the same
        #: highlight" checkable, after three flag lifetimes in this widget each
        #: cost a lost draft.
        self._click_selection: Selection | None = None
        #: When a multi-click last produced NO RANGE, as a monotonic timestamp.
        #:
        #: The narrowest fact that covers both remaining draft-loss paths: the
        #: blank LAST row `_line_break_span` cannot give a range to without
        #: breaking D7, and the near-miss pair Textual scores as two singles
        #: (design round 2, D2 and D2-1). Both end the same way — a deliberate
        #: gesture, a frame identical to the one before it, then Ctrl+C clears
        #: the draft.
        #:
        #: A TIMESTAMP and not a flag, because the question the Ctrl+C rung asks
        #: is "did a multi-click JUST happen and produce nothing?" — bounded,
        #: recent and gesture-scoped. Gating on live selection STATE instead is
        #: D17's lost-draft bug, where a highlight made minutes ago diverted the
        #: key and the second reflexive tap exited the app. This claim expires
        #: on its own (:data:`BARREN_CLICK_WINDOW_S`), so it cannot outlive the
        #: gesture the way three earlier flag lifetimes each did.
        self._barren_click_at: float | None = None
        #: The last SINGLE click as `(monotonic, screen offset)`, for detecting
        #: the near-miss pair described in `_on_click`. Only ever read to decide
        #: whether two singles were one gesture; it never makes a selection.
        self._single_click_at: tuple[float, Offset] | None = None
        #: Bang-mode: Enter runs a local shell command instead of sending a
        #: prompt. Owned here because it is a property of the COMPOSER (the
        #: same way the slash picker is), not of the app: the key that enters
        #: it is intercepted before TextArea inserts, and the key that leaves
        #: it has to win over Esc-means-stop. The app follows via
        #: :class:`ShellModeChanged`.
        self._shell_mode = False
        #: The aside turns this off for as long as it owns the composer: a
        #: bang-mode command run from a card that promised "off the record"
        #: would both break the promise and eat the ``!`` the user typed as
        #: the start of a question. Default on; the main chat is the only
        #: surface that runs commands.
        self._allows_shell = True
        self.set_commands(commands or [])

    def render_line(self, y: int) -> Strip:
        """Draw the empty composer's caret in a cell of its OWN.

        Textual's placeholder branch inverts cell 0 of the placeholder when the
        caret is drawn, which paints the block on top of the ``M`` of
        ``Message Local Operator…`` — the caret and the copy competing for one
        cell. Nothing in a cell grid can hold both, so the placeholder moves one
        cell right while the caret is on screen and the caret takes the column
        the first typed character will occupy. Both survive: a solid block at
        the head of the field, and the invitation still readable as words.

        The shift is conditional on the caret being drawn rather than permanent
        so the resting (blurred) composer keeps its copy aligned with the column
        typed text starts in; the one-cell move IS the focus transition, and it
        happens on the same frame as the chevron going accent.

        This mirrors ``TextArea.render_line``'s own placeholder branch — padding
        is the only difference — because that branch runs BEFORE any hook a
        subclass could reach and owns the wrap. Every other row is the base
        class's, then post-processed by :meth:`_paint_markers` so an attachment
        marker reads as an object rather than as text the user typed.
        """
        if not self.text and self.placeholder:
            theme = self._theme
            cursor_style = theme.cursor_style if theme else None
            # ONE condition for both the reserved cell and the block painted in
            # it. Computing them separately would indent the copy by a cell
            # with nothing in it on any frame where the caret cannot be drawn.
            caret = bool(self._draw_cursor) and cursor_style is not None
            placeholder = Content.from_text(self.placeholder)
            if caret:
                placeholder = placeholder.pad_left(1)
            lines = placeholder.wrap(self.content_size.width)
            if y < len(lines):
                content = lines[y].stylize(self.get_visual_style("text-area--placeholder"))
                if caret and y == 0:
                    assert cursor_style is not None  # narrowed by `caret`
                    content = content.stylize(ContentStyle.from_rich_style(cursor_style), 0, 1)
                return Strip(content.render_segments(self.visual_style), content.cell_length)
        # Slash highlighting LAST so its own line-0/leading-`/` bail is the cheap
        # rejection on the common prose path. The two passes never contend for
        # the same cells: a marker opens with `[` and lives in the message tail,
        # the command word and name open with `/` on line 0 — so order is
        # immaterial for correctness (see :meth:`_paint_slash`).
        #
        # NO ghost-specific caret pass runs here, and adding one back is a
        # regression, not a fix. #378 shipped `_paint_ghost_caret`, which moved
        # the block one cell LEFT so it sat on the last typed character instead
        # of on the ghost's first cell. Users read a block ON a committed
        # character as the vi-normal-mode / overwrite idiom — "the next
        # keystroke replaces this `e`" — even though the text was fully
        # committed and the caret genuinely belonged after it. The composer
        # contradicted itself between its GHOSTED and UN-GHOSTED states: a
        # buffer with no completion on offer put the caret after the text,
        # while every buffer carrying a ghost pulled it one cell back.
        #
        # Every autosuggestion implementation surveyed keeps the caret at the
        # TRUE insertion point, which is exactly where the ghost starts: fish
        # renders the suggestion after the cursor; zsh-autosuggestions holds
        # `CURSOR == $#BUFFER` with the ghost in `POSTDISPLAY`;
        # prompt_toolkit's `AppendAutoSuggestion` appends fragments and leaves
        # `cursor_position` at the end of the text; Textual's own
        # `Input.render_line` builds `value + suggestion[len(value):]` and then
        # applies the cursor style ON TOP of the first ghost cell. There is no
        # precedent, terminal or GUI, for retreating the caret a cell. (Browser
        # URL bars use the other coherent convention — insert the completion and
        # SELECT it — which is a selection, not a displaced caret.)
        # So the base class's paint stands: block at the insertion point, on the
        # ghost's first cell. :meth:`_paint_ghost_ink` only re-inks that one
        # cell; it does not move the caret.
        return self._paint_ghost_ink(
            self._paint_slash(self._paint_markers(super().render_line(y), y), y), y
        )

    def _paint_ghost_ink(self, strip: Strip, y: int) -> Strip:
        """Dim the ONE ghost cell the caret block covers. The caret does not move.

        The caret belongs at the insertion point (see :meth:`render_line`), and
        with a block caret that cell is the ghost's first character. Textual
        applies ``text-area--cursor`` on top of it wholesale, so the ghost's
        head is painted in the caret's full ink — measured ``#1e1a14`` on
        ``#e9e5db``, a 13.76:1 contrast, byte-identical to a caret sitting on a
        character the user actually typed. Rendered frames of ``/resum`` show
        the previewed ``e`` as solid black-on-cream while its own trailing
        ``' '`` stays dim: the boundary between committed and previewed text
        lands INSIDE the ghost, so the first suggested character reads as
        already-typed.

        The fix is ink, not position: keep the caret's background (the block is
        load-bearing — the boot composer, the read-only composer and the
        attachment chip all assert an inverted caret cell) and restore the
        suggestion's own foreground over it. That is 3.29:1 against the cursor
        ground — comfortably legible, and visibly not the 13.76:1 of committed
        text — so the cell reads as "caret, on previewed text", which is
        exactly what it is.

        A post-pass over the finished ``Strip``, matching the idiom
        :meth:`_paint_markers` and :meth:`_paint_slash` already establish, and
        gated on a ghost being shown so ordinary editing pays one boolean.
        """
        if not self.suggestion or not self._draw_cursor:
            return strip
        row, column = self.selection.end
        # The ghost's gates guarantee a single line with the caret at its end,
        # but the wrap still decides WHICH screen row carries that column, and
        # only that row may be repainted.
        wrapped = self.wrapped_document
        absolute_y = self.scroll_offset.y + y
        if absolute_y >= wrapped.height:
            return strip
        row_line, _section_start = wrapped.offset_to_location(Offset(0, absolute_y))
        if row_line != row:
            return strip
        caret_x = wrapped.location_to_offset((row, column)).x + self.gutter_width
        if caret_x >= strip.cell_length:
            return strip
        # Foreground ONLY. ``post_style`` merges, so giving it just the colour
        # leaves the cursor background the base class painted intact — the
        # block keeps its shape and only the glyph inside it changes ink.
        ghost_ink = RichStyle(color=self.get_component_rich_style("text-area--suggestion").color)
        left, rest = strip.divide([caret_x, strip.cell_length])
        caret_cell, right = rest.divide([1, rest.cell_length])
        return Strip.join(
            [
                left,
                Strip(Segment.apply_style(caret_cell, post_style=ghost_ink)),
                right,
            ]
        )

    # -- public API ---------------------------------------------------------
    @property
    def picker(self) -> CommandPicker:
        """The slash-command picker. The app mounts it below the input row."""
        return self._picker

    @property
    def model_picker(self) -> ModelPicker:
        """The model list shown while the buffer holds ``/model <query>``."""
        return self._model_picker

    @property
    def argument_command(self) -> str | None:
        """Which command the buffer's argument list is open for, if any.

        Exposed so the app can check that an ``ArgumentQueryOpened`` it is
        handling still describes the buffer. The message is one message-loop tick
        old, and a tick is enough for the user to have typed over the command —
        answering a stale one attaches rows to a command that no longer exists.
        The buffer stays the single authority; this is how a reader outside the
        widget asks it, rather than parsing the text a second time.
        """
        return self._argument_command

    def set_model_handler(self, handler: Callable[[ModelRow], None] | None) -> None:
        """Install what a chosen model DOES.

        A callback rather than a message, because the two outcomes are different
        kinds of thing — switching the session model, or starting a login for a
        provider that has no credential — and only the app knows how to do either.
        The editor's job ends at "the user chose this row".
        """
        self._on_model_chosen = handler

    def prompt_history(self) -> list[str]:
        """Recorded prompts, oldest first.

        Named ``prompt_history`` rather than ``history``: ``TextArea`` already
        owns a ``history`` attribute for its undo stack, and shadowing it with
        a method of an unrelated type is a live footgun for anything that
        reaches for the base class's own edit history.
        """
        return list(self._history)

    def set_records_history(self, records: bool) -> None:
        """Whether a submitted line joins the recallable prompt history.

        Off while the `/btw` aside owns the composer, and that is a CONTRACT
        rather than a preference: the card prints "off the record — nothing
        here joins the chat, esc discards it", and a question still sitting in
        the UP history after Esc falsifies it — one press recalls it and the
        next Enter sends it to the agent as a real turn. It also stops the
        aside burying the last thing the user actually said to the agent, which
        is what UP is for.

        Borrowed and returned the same way the placeholder, the draft and the
        command list are (``OperatorApp._open_aside`` / ``_close_aside``).
        Suppressed at the RECORD, not unwound afterwards: ``_submit`` records
        before it posts, so anything that unwinds has to guess how many entries
        to remove and gets it wrong the moment two asides repeat a question.
        """
        self._records_history = records

    def forget_last_prompt(self, text: str) -> None:
        """Retract the entry ``text`` just recorded, if it is still the newest.

        The one case :meth:`set_records_history` cannot cover: the line that
        OPENS the aside (``/btw <question>``) is submitted while the aside is
        still closed, so ``_submit`` has already recorded it by the time the
        handler runs — and it carries the question verbatim, which the card
        then promises to discard.

        Exact rather than a blind pop: the caller passes the line it is
        retracting and nothing happens unless that is still the newest entry,
        so a race that recorded something else in between cannot silently eat
        a real prompt. Note ``_record_history`` drops a consecutive duplicate,
        so re-asking the same question twice leaves nothing to retract the
        second time and the guard is what makes that a no-op instead of a bug.
        """
        stripped = text.strip()
        if stripped and self._history and self._history[-1] == stripped:
            self._history.pop()
        self._history_index = None

    def forget_prompt(self, text: str) -> None:
        """Remove the newest history entry equal to ``text``, wherever it sits.

        The recall seam: an Esc-recalled steer is UNSENT — it goes back to the
        composer as a draft — so Up-arrow must not offer it as a past prompt
        while the composer already holds it as the present one. Unlike
        :meth:`forget_last_prompt` the entry need not be the newest: the user
        can submit something else between the steer and the recall, and the
        recalled line then sits mid-history. One entry per call: the same text
        sent twice is two sends, and recalling one of them retracts one.

        Navigation is reset rather than re-aimed: a parked index would shift
        under the removal, and a recall replaces the buffer anyway, so the
        draft navigation was stashing is already gone.
        """
        stripped = text.strip()
        if not stripped:
            return
        for index in reversed(range(len(self._history))):
            if self._history[index] == stripped:
                del self._history[index]
                break
        self._history_index = None

    def remember_draft(self) -> bool:
        """Push the CURRENT buffer into prompt history without submitting it.

        For the paths that throw a draft away on the user's behalf — Ctrl+C on
        a half-typed prompt. Discarding text the user typed should never be
        final when making it recoverable costs one entry: after this, ``up``
        brings it straight back.

        History is the right home rather than a bespoke stash because it is
        already the place a user looks for "what I typed a moment ago", and it
        already de-duplicates, caps itself at ``HISTORY_LIMIT`` and drops
        blanks. ``_record_history`` also resets ``_history_index``, so the next
        ``up`` starts from this entry rather than from wherever an interrupted
        history walk had got to.

        HONOURS ``_records_history``, and returns whether it recorded. While
        the aside owns the composer that flag is off, and it is a contract the
        card states out loud — "off the record — nothing here joins the chat".
        Recording there would put a question the user explicitly kept out of
        the conversation one ``up`` and one Enter away from being sent to the
        agent as a real turn, which is the exact failure
        :meth:`set_records_history` exists to prevent. The caller uses the
        return value to decide whether the draft is safe to discard.
        """
        if not self._records_history:
            return False
        self._record_history(self.text)
        return True

    def set_commands(self, commands: list[SlashCommand]) -> None:
        """Slash commands offered to the picker (sync, no I/O).

        Also re-derives which words open a VALUE list after their space, and
        which of those the bare command cannot stand without. Both sets include
        aliases, because a user who typed the alias must get the same list —
        that is the bug ``MODEL_COMMANDS`` spells ``models`` out for.
        """
        self._picker.set_commands(commands)
        self._argument_commands = tuple(
            name
            for command in commands
            if command.arguments is not ArgumentMode.NONE
            for name in command.names
        )
        self._required_argument_commands = tuple(
            name
            for command in commands
            if command.arguments is ArgumentMode.REQUIRED
            for name in command.names
        )
        # Commands whose argument is a PROMPT (goal/loop/team/agent/btw). Engaging
        # one inline reassembles it to the FRONT of the composer with the draft as
        # its argument, rather than splicing-and-running — see
        # :meth:`_reassemble_prompt_command`. Derived from the registry (aliases
        # included) so the set cannot drift from the flag it reads.
        self._prompt_commands = tuple(
            name for command in commands if command.consumes_prompt for name in command.names
        )
        # Lower-cased vocabulary (primaries AND aliases), shared by the
        # highlighter's "is this a real command?" oracle and the inline
        # tokenizer's nested-slash rule (a slash inside an engaged command's
        # argument is plain text). Matches the case-insensitive way
        # ``slash_command_for`` resolves a typed word, so the highlight cannot
        # claim a command the dispatch would reject.
        self._command_names = frozenset(
            name.lower() for command in commands for name in command.names
        )

    def opens_a_list(self, name: str) -> bool:
        """Whether completing the command word ``name`` opens a list INSTEAD of
        running it.

        True for the model picker's words and for every ``REQUIRED`` argument
        command: there, opening the list IS the outcome of the keystroke, and
        submitting as well would run a no-op and clear the buffer the list was
        just drawn over. An ``OPTIONAL`` one answers something when bare
        (``/approvals`` reports the mode, ``/effort`` the ladder), so Enter
        still sends it and the list is left as an offer for the next keystroke.
        """
        lowered = name.lower()
        return lowered in self.MODEL_COMMANDS or lowered in self._required_argument_commands

    def _is_name_argument_command(self, name: str | None) -> bool:
        """Whether ``name``'s argument is a NAME followed by a free-text message.

        The one predicate behind both asks: completion routes team/agent rows to
        :meth:`_complete_name_argument` (space, no submit), and the highlighter
        only paints an argument NAME for these commands. ``name`` may be ``None``
        because ``_apply_command`` reads ``_argument_command`` while a list is
        open — defensive, mirroring :meth:`opens_a_list`'s lowercase test.
        """
        return (name or "").lower() in self.NAME_ARGUMENT_COMMANDS

    @staticmethod
    def _name_command_family(name: str | None) -> str | None:
        """The roster FAMILY a NAME+message command word belongs to.

        ``team``/``teams`` share the team roster and ``agent``/``agents`` share
        the agent roster, but the two rosters are disjoint — a team name is not a
        valid agent and vice versa. The highlighter's name snapshot is therefore
        only valid to paint against the family it was filled for, so both the
        snapshot and the buffer's LEADING command word are reduced to a family
        key and compared. ``None`` for anything that is not a NAME+message
        command (its argument is not a roster name at all).
        """
        lowered = (name or "").lower()
        if lowered in ("team", "teams"):
            return "team"
        if lowered in ("agent", "agents"):
            return "agent"
        return None

    def _leading_command_word(self) -> str | None:
        """The lower-cased command word on the buffer's FIRST non-blank line.

        Distinct from :meth:`_command_word`, which #250 made caret-anchored (it
        answers "which slash token is the caret on", for the inline mid-draft
        picker). The highlighter and its snapshot preservation are about the
        LEADING command — the one on ``first`` that owns a multi-line body — so
        they must read that line regardless of where the caret sits (on the
        message body the caret-anchored word would be ``None`` and the name would
        wrongly go dark). Mirrors the parse in :meth:`_compute_slash_runs`.
        """
        line = next((line for line in self.text.split("\n") if line.strip()), "")
        stripped = line.lstrip()
        if not stripped.startswith("/"):
            return None
        return stripped[1:].partition(" ")[0].lower()

    def _name_switch_hint(self, list_argument: str) -> str | None:
        """The U1/U2 hint, or ``None`` when it must not show.

        Shows ONLY in the one state it answers: a NAME+message command whose
        argument is a completed name followed by a single terminating space and
        nothing else — i.e. exactly the parked-caret resting state
        ``_complete_name_argument`` produces. ``list_argument`` is the picker's
        current argument query (everything after ``/<cmd> ``), so:

        * a bare ``/team `` (empty argument) is still choosing a NAME — the row
          list is up and answers the question, no hint;
        * ``/team frontend-guild `` (name + one trailing space) is the moment the
          two outcomes appear — HINT;
        * ``/team frontend-guild fix`` (a message is being typed) has a non-empty
          tail past the space — the user has chosen "send", withdraw the hint.

        Gated on the command being a NAME+message one so it never intrudes on the
        enum-tail argument lists, and returned as text the caller feeds to the
        picker notice channel (empty tail only, so it cannot cover a real row).
        """
        if not self._is_name_argument_command(self._argument_command):
            return None
        # A completed name reads as `<name> ` — one internal space that
        # terminates the name and an empty tail. `rstrip(" ")` then a re-add of a
        # single space would be fragile; instead require exactly: a non-empty
        # first token, and everything after the FIRST space is blank.
        name, sep, tail = list_argument.partition(" ")
        if not name or not sep or tail.strip():
            return None
        # `/team chart ` is NOT a completed name — `chart` is the reserved
        # subcommand, and the space leads into the second-slot team list that
        # feeds the chart, not into a switch/send choice. The attach-or-send
        # semantics this hint names do not apply, so it must not show. (A team
        # literally named `chart` is talked to with `/team =chart …`, whose
        # leading `=` makes the first token not equal `chart`.)
        if self._argument_command in ("team", "teams") and name.lower() == "chart":
            return None
        return self.NAME_SWITCH_HINT

    def set_name_choices(self, names: frozenset[str]) -> None:
        """The team/agent names the OPEN argument list is offering, for the
        highlighter.

        Pushed by the app from ``on_argument_query_opened`` beside the
        ``picker.set_choices`` call — the same "app fills the rows when the list
        opens" contract, extended to hand the widget a cheap immutable snapshot
        it can test membership against on the render path. The render pass must
        never walk the registries itself (that is app-side I/O on every frame),
        so name recognition rides this snapshot; it is cleared when the argument
        command changes to a non-name one (see :meth:`_sync_picker`).

        Refreshes the composer when the snapshot actually changes: the names
        arrive one message-loop tick AFTER the keystroke that opened the list
        (the app answers ``ArgumentQueryOpened``), by which point ``render_line``
        has already painted — and cached — the line without them. Without the
        refresh the name would only light up on the NEXT keystroke, a visible
        lag. Gated on a real change so an unchanged re-push is not a repaint.
        """
        # The family the snapshot is valid for: the command whose argument list
        # is open right now. Recorded alongside the names so the multiline
        # preservation in ``_sync_picker`` can reject a snapshot inherited across
        # a family switch (see :attr:`_name_choices_family`).
        family = self._name_command_family(self._argument_command)
        if names == self._name_choices and family == self._name_choices_family:
            return
        self._name_choices = names
        self._name_choices_family = family
        self.refresh()

    @property
    def shell_mode(self) -> bool:
        """Whether Enter will run a local shell command instead of a prompt."""
        return self._shell_mode

    def set_shell_mode(self, active: bool) -> None:
        """Enter or leave bang-mode. No-op when already in that state.

        The placeholder swap is the mode's own voice, the same way the aside
        and the subagent page each have one. Restoring goes through
        :data:`DEFAULT_PLACEHOLDER` rather than whatever was showing, because
        the only other placeholders are modes that refuse this one (the aside
        owns the composer; the subagent page is read-only) and a user leaving
        bang-mode is returning to the resting composer.
        """
        if self._shell_mode == active:
            return
        self._shell_mode = active
        self.placeholder = SHELL_PLACEHOLDER if active else self.resting_placeholder
        self.post_message(ShellModeChanged(active))

    def set_allows_shell(self, allowed: bool) -> None:
        """Whether bang-mode may be entered from this composer.

        The aside turns this off; leaving bang-mode first so a mode that
        started in the main chat cannot survive the card taking the field.
        """
        if not allowed:
            self.set_shell_mode(False)
        self._allows_shell = allowed

    def clear_content(self) -> None:
        """Empty the buffer and leave history navigation."""
        self.text = ""
        self._history_index = None
        # Attachments belong to the text that referenced them. A draft cleared
        # without dropping them would send the previous prompt's screenshots
        # along with the next unrelated question. The counter resets too, so
        # each prompt numbers from #1 rather than carrying a running total the
        # user never sees the start of.
        self._attachments.clear()
        self._draft_attachments.clear()
        self._next_marker = 1

    def begin_model_query(self) -> None:
        """Put the buffer into the state that shows the model list.

        Exists so the app can reopen the list after ``/model`` has been submitted
        (the command picker completes the word, which terminates it, which submits
        it and clears the buffer) without reaching into the editor's private cursor
        helpers. Writing the text rather than poking the widget is deliberate: the
        buffer is what decides which picker is open, so anything that opens one has
        to go through the buffer or be undone by the next resync.
        """
        # Through the shared helper so the picker is re-derived with the caret at
        # the end (in the argument slot), not at the origin the ``text`` setter
        # leaves it at. Caret-anchored detection means a sync at the origin would
        # read ``/model `` as the command WORD and never post ``ModelQueryOpened``,
        # so the list would stay empty until the next keystroke.
        self._set_text_and_caret("/model ", len("/model "))
        self._history_index = None

    # -- key interception ---------------------------------------------------
    async def _on_key(self, event: events.Key) -> None:
        """Handle chat keys before TextArea's insert path sees them."""
        key = event.key
        # A CSI-modifier vertical chord IS its plain arrow, and is rewritten to
        # one here so that every handler below — both pickers, history, the
        # caret — sees the key it already gates on. This is the whole fix for
        # F5/U6: the alternative (binding `alt+up` to an action) skips `_on_key`
        # and therefore skips the pickers, which is how `⌥↑` came to close an
        # open list and overwrite a half-typed slash command from history.
        #
        # The Esc-prefixed encoding needs no rewrite: `ESCAPE_CHORD_KEYS` maps
        # its arrows to a pass-through, so it arrives here already spelled
        # `up`/`down`. Both encodings therefore converge on ONE implementation
        # of what the arrow means, which is the property that stops them
        # drifting apart again.
        #
        # THE INVARIANT, stated because getting it wrong has now caused a defect
        # in three consecutive rounds, each time on the encoding the previous
        # fix did not touch:
        #
        #   A key that arrived as its own SELF-CONTAINED chord must never be
        #   treated as the tail of an escape chord.
        #
        # The two encodings carry the same intent in structurally different
        # shapes. `\x1b[1;3A` is ONE event that already means "option+up";
        # `\x1b\x1b[A` is TWO events whose first is indistinguishable from a
        # real Escape until the second arrives. Only the second kind is
        # evidence about a pending escape. `self_contained` records which kind
        # this was BEFORE the rewrite erases the distinction, because after the
        # rewrite both spellings read as `up` and the difference is
        # unrecoverable (code round 3, F8).
        self_contained = key in self.VERTICAL_CHORD_KEYS
        if self_contained:
            key = self.VERTICAL_CHORD_KEYS[key]
        # FIRST, ahead of every other branch: an arrow completing a held escape
        # is the second half of an option+arrow chord, and it must be read as
        # that before any picker or history handler claims it. See the
        # escape-coalescing block below `_on_key` for the whole rationale.
        #
        # `not self_contained` is the invariant above doing its work. Without
        # it a rewritten `alt+up` landing while an escape was held was read as
        # that escape's second half and CANCELLED it, silently dropping every
        # meaning Esc has — the turn kept running, bang-mode stayed on, an open
        # list refused to dismiss. The horizontal chords never had the bug only
        # because `alt+left` is not rewritten and so was never in this table.
        if (
            self._pending_escape is not None
            and not self_contained
            and key in self.ESCAPE_CHORD_KEYS
        ):
            chord = self.ESCAPE_CHORD_KEYS[key]
            # The escape is DROPPED, not run: the user pressed one chord, so
            # whatever that escape would have done — stop the turn, leave shell
            # mode, dismiss a list — must not also happen.
            self._cancel_escape()
            if chord is None:
                # A chord with no motion of its own (the vertical arrows): the
                # key carries on to its ordinary handler below, so `⌥↑` behaves
                # exactly like `↑`. Not consumed, deliberately.
                pass
            else:
                action, select = chord
                action(self, select)
                event.stop()
                event.prevent_default()
                return
        # Any other key ends the chord window: the escape stood alone, so its
        # action is owed now and must land BEFORE this key is handled (an Esc
        # then a typed character must stop the turn, then type the character).
        if self._pending_escape is not None:
            self._flush_escape()
        # An advertised answer key, while a question is up and the buffer is
        # empty, answers the question instead of being typed.
        #
        # Routed rather than implemented here: the app owns which prompt is
        # live and which keys it advertises. This is the FIRST thing checked so
        # the key never reaches TextArea's insert path, and it is scoped to an
        # empty buffer, so the moment there is a draft the composer takes
        # everything again.
        #
        # The alternative — moving FOCUS to the prompt whenever the composer was
        # empty — is what this replaces, and it was worse than the defect it
        # fixed: the composer is empty exactly when a user is about to type, so
        # the first character of an intended steer landed on the card and `y`
        # AUTHORISED a pending `rm -rf` (F3, agent review round 2).
        # ...but the pickers own their keys first. `_sync_picker` has not run
        # yet at this point, so an OPEN picker is asked directly: while a slash
        # or model list is up, Tab COMPLETES the row and belongs to it. Routed
        # ahead of that, the prompt's Tab swallow silently broke completion for
        # as long as any question was live — `/mod` + Tab left `/mod` (found by
        # driving the combination rather than reasoning about it).
        picker_open = self._model_picker.is_open() or self._picker.is_open()
        router = None if picker_open else getattr(self.app, "route_key_to_live_prompt", None)
        if callable(router):
            try:
                if router(event):
                    event.stop()
                    event.prevent_default()
                    return
            except Exception:  # pragma: no cover - hosts with no prompt surface
                pass
        # Re-derive the picker state from the buffer BEFORE routing. The
        # buffer is the only authority on whether a command word is open, and
        # syncing here means routing never depends on a queued Changed
        # message having been drained first.
        self._sync_picker()
        if self._model_picker.is_open():
            # Checked BEFORE the command picker, though the two are mutually
            # exclusive by construction (`slash_context` closes on the space that
            # `slash_argument` opens on). Ordering it first makes that invariant
            # cheap to keep: if the two ever did overlap, the picker the user is
            # looking at is the one holding a query they just typed into.
            if key == "escape":
                # Esc closes the LIST, not the command: the text survives so a
                # user who wanted to type the id by hand can carry on.
                #
                # Deferred like every other escape meaning, so `⌥←` with the
                # model list open moves by word instead of closing the list and
                # nudging one character (ux round 1, U3).
                self._defer_escape(self._model_picker.close)
                event.stop()
                event.prevent_default()
                return
            if key in ("up", "down"):
                self._model_picker.move(-1 if key == "up" else +1)
                event.stop()
                event.prevent_default()
                return
            if key in ("pageup", "pagedown"):
                self._model_picker.page(-1 if key == "pageup" else +1)
                event.stop()
                event.prevent_default()
                return
            if key in ("home", "end"):
                self._model_picker.jump(to_end=key == "end")
                event.stop()
                event.prevent_default()
                return
            if key == "d" and not self._model_picker.query_text():
                # `d` SAVES the highlighted row as the boot default (#369).
                #
                # Gated on an EMPTY query, which is the whole reason this is
                # safe here: every printable key in this picker belongs to the
                # filter, so `d` typed while narrowing (`/model d` → deepseek)
                # must keep filtering. Only once the query is empty is `d` free
                # to be an action, and that is exactly the state a user is in
                # after arrowing to a row they already found.
                row = self._model_picker.highlighted()
                if row is not None:
                    handler = getattr(self.app, "_persist_default_from_picker", None)
                    if callable(handler):
                        handler()
                        event.stop()
                        event.prevent_default()
                        return
            if key in ("tab", "enter"):
                row = self._model_picker.highlighted()
                if row is not None:
                    # Tab COMPLETES the selector into the buffer, Enter ACTS on
                    # it. Unlike the command picker there is no ambiguity gate:
                    # every row here names one concrete model, and the worst case
                    # of a wrong pick is a switch the user reverses with another
                    # `/model` — not autonomous work or a deleted credential.
                    if key == "tab":
                        self._complete_model(row)
                    else:
                        self._model_picker.choose(self._model_picker.selected_index)
                    event.stop()
                    event.prevent_default()
                    return
        if key == "escape" and self._picker.is_pending():
            # The rows are one message-loop tick behind the keystroke that opened
            # the list — the app answers ArgumentQueryOpened — and for that tick
            # the picker is in argument mode holding nothing, so an `is_open()`
            # gate DROPPED the Esc: the user dismissed the list and then watched
            # it appear anyway. Dismissing an empty argument list records the
            # query, which is what the arriving rows are checked against.
            self._defer_escape(self._picker.dismiss)
            event.stop()
            event.prevent_default()
            return
        if self._picker.is_open():
            if key == "escape":
                self._defer_escape(self._picker.dismiss)
                event.stop()
                event.prevent_default()
                return
            if key in ("up", "down"):
                # The picker owns the arrows while it is open: history nav and
                # caret movement both stay reachable one Esc away, and an open
                # list that ignored Up/Down would look broken.
                self._picker.move(-1 if key == "up" else +1)
                event.stop()
                event.prevent_default()
                return
            if key in ("tab", "enter"):
                # Both keys insert the SAME completion, so the highlighted row
                # can never mean two different commands depending on the key.
                #
                # Enter only SENDS when the choice is unambiguous: one match, or
                # a name the user typed in full. The registry's blast radius is
                # not uniform — `/loop` starts autonomous work and `/logout`
                # removes credentials — and with a fuzzy matcher `/lo` highlights
                # `loop` while `login` and `logout` also match, so "Enter runs
                # whatever the matcher picked" could start a loop for a user
                # reaching for login, rewriting their text and running it in one
                # keystroke. When it is ambiguous, Enter completes (Tab's
                # behaviour) and a second Enter sends — the extra keystroke only
                # appears where the intent genuinely is not clear (D16).
                name = self._picker.highlighted_name()
                if name is not None:
                    # Decided BEFORE the completion is applied: completing
                    # rewrites the text and re-syncs the picker, so asking
                    # afterwards would measure the completed word (always one
                    # exact match) and submit unconditionally.
                    unambiguous = self._picker_choice_is_unambiguous(name)
                    if self._picker.mode is PickerMode.ARGUMENT:
                        self._resolve_argument(name, key, unambiguous)
                    elif unambiguous:
                        self._apply_command(name)
                        # A command whose ARGUMENT drives its own list is not run
                        # by completing it — completing it IS opening the list. The
                        # trailing space `_apply_command` adds is what opens the
                        # model picker, so submitting as well echoed `/model` into
                        # the transcript, cleared the buffer, and made the app put
                        # the query back to reopen a list the keystroke had already
                        # opened. One keystroke, one outcome.
                        #
                        # Runs through the shared run path so an inline command is
                        # spliced out and its draft kept, exactly like the
                        # argument phase — `_apply_command` moved the caret to the
                        # end of the completed word, so the run path re-parses at
                        # that caret and finds this token.
                        if key == "enter" and not self.opens_a_list(name):
                            self._run_command_from_buffer()
                    elif key == "tab":
                        # Tab never sends, so it can safely take the highlighted
                        # row: that is the whole point of a completion key.
                        self._apply_command(name)
                    else:
                        # Ambiguous Enter. Grow the word to the matches' longest
                        # COMMON prefix and leave the list open, rather than
                        # completing to the highlighted row.
                        #
                        # Completing to the row put the highest-blast-radius
                        # candidate into the buffer ready to run — `/lo`
                        # highlights `loop`, so a reflex double-Enter started
                        # autonomous work for a user reaching for `/login`. The
                        # common prefix cannot be the wrong command by
                        # construction: it is the part every candidate agrees on.
                        # This is also the shell idiom, which means the two-Enter
                        # rule stops being a special case users have to learn.
                        self._extend_to_common_prefix()
                    event.stop()
                    event.prevent_default()
                    return
        if key == "escape":
            # LAST escape branch on purpose: every picker case above has already
            # returned, so this is Esc with nothing on screen to dismiss.
            #
            # Bang-mode takes it first, and keeps the draft. opencode's Esc in
            # shell mode is "leave the mode", not "scrap the command": a user
            # who typed `ls` and changed their mind still has `ls` to send as
            # a prompt, and a user who entered the mode by accident is one
            # keystroke from the resting composer. Stop is one Esc away after.
            if self._shell_mode:
                # Deferred, and this is the branch where it matters most.
                # Bang-mode is a full-width editable buffer with a visible
                # caret, so fixing a typo in a half-typed command is exactly why
                # someone reaches for `⌥←`. Running this synchronously ejected
                # the user from shell mode on an Esc-prefixed terminal, and the
                # ejection is INVISIBLE — the mode's only indicator is the
                # placeholder, which is hidden the moment the buffer has text.
                # The user saw an identical frame and their next Enter sent the
                # command to the model as a prompt instead of running it
                # (code round 1 F1, ux round 1 U1).
                self._defer_escape(lambda: self.set_shell_mode(False))
                event.stop()
                event.prevent_default()
                return
            # It has to be handled rather than left to bubble, because
            # ``TextArea`` binds Escape to ``blur``. That made Esc a silent trap:
            # the first press moved focus out of the composer (so the user's next
            # keystrokes went nowhere) and only a LATER press, once focus had
            # already left, reached the app's stop. Consuming the key here keeps
            # focus put and gives Esc one meaning — stop what the agent is doing.
            #
            # DEFERRED, not posted: on an Esc-prefixed terminal this same
            # `escape` may be the first half of an option+arrow chord, and
            # stopping the turn because the user moved the caret by a word is
            # issue #370. The key is still consumed immediately (focus stays
            # put); only the message is held. See the escape-coalescing block
            # below `_on_key` for the window, the terminals, and the limits.
            self._defer_escape(lambda: self.post_message(StopRequested()))
            event.stop()
            event.prevent_default()
            return
        if key == "enter":
            self._submit()
            event.stop()
            event.prevent_default()
            return
        if key == "shift+enter":
            # Explicit newline; TextArea's stock path would also submit here,
            # so insert the newline ourselves and consume the key.
            self.insert("\n")
            event.stop()
            event.prevent_default()
            return
        if key in ("ctrl+c", "super+c"):
            # BOTH chords copy, and `super+c` is routed HERE rather than left
            # to the binding it already had. `TextArea.BINDINGS` carries
            # `ctrl+c,super+c` -> `copy`, so Cmd+C already reached
            # `action_copy` on a kitty-protocol terminal and already produced
            # the receipt — but with no `super+c` branch here to consume it,
            # the key fell straight through to that binding, so it reached ONLY
            # `action_copy` and skipped the click-chain collapse below. (The
            # rule is that `_on_key` runs FIRST and a binding fires only on the
            # keys it does not consume; adding this branch is what claims the
            # chord.) Measured on the tree
            # before this change, with a double-click selection live: `ctrl+c`
            # left the selection collapsed at `((0,11),(0,11))` while
            # `super+c` left it at `((0,6),(0,11))`. A range that never
            # collapses is a Cmd+C that can never hand the key back to the
            # draft and interrupt rungs — R1-2's bug, reintroduced for the
            # other chord. One route for both keys is what stops the two
            # drifting apart again.
            #
            # `super+c` deliberately carries NO interrupt meaning: the `else`
            # branch below is gated on the key. Cmd+C is not an interrupt
            # gesture on macOS, nothing binds `super+c` at app level, and the
            # exit ladder stays Ctrl+C's alone — a Cmd+C that cleared the draft
            # would be a new way to lose work, not a convenience.
            #
            # A REAL RANGE makes this press a copy, ahead of its interrupt
            # meaning. ``TextArea``'s ``ctrl+c`` binding never runs — the key
            # is consumed here first — so the highlight-then-Ctrl+C sequence
            # the field report names would otherwise be the one copy gesture
            # with no effect at all. Only a real range qualifies: a selection
            # is STATE that persists until the caret moves, but a collapsed
            # caret is the resting state of the composer, and gating the
            # interrupt on "some range is live" is D17's lost-draft bug this
            # ordering exists to avoid. A stale LIVE range is safe to copy
            # through: it has start != end, so it survives the checks in the
            # app's interrupt rung that guard the exit ladder, and the press
            # means "take this", not "scrap my draft".
            if self.selected_text:
                clicked = self._click_selection is not None
                self.action_copy()
                if clicked:
                    # A range made by the CLICK CHAIN collapses once it has been
                    # copied, which hands the key back to the draft and
                    # interrupt rungs on the very next press.
                    #
                    # Scoped to the multi-click gesture on purpose, because the
                    # two kinds of range differ in what the user meant. A drag
                    # or shift+arrow range is DELIBERATE: `action_copy`'s rule
                    # that a live range keeps copying, and that the highlight
                    # stays up as the receipt's visible subject, is written for
                    # those and is load-bearing (D25 pins that a blurred
                    # composer still paints the copy it defers to).
                    #
                    # A double-click is REFLEXIVE, and this change made it the
                    # gesture users reach for first. The range it leaves
                    # persists until the caret moves, so without this collapse
                    # Ctrl+C could never interrupt a running turn at any number
                    # of presses (agent review round 1, R1-2). Esc still stopped
                    # the agent, but Ctrl+C is the first rung of the exit ladder
                    # and a user reaching for it repeatedly got silence.
                    #
                    # So the accidental range gives the key back after one copy
                    # while the deliberate one keeps the documented behaviour.
                    # One press still copies exactly once either way.
                    #
                    # THE MEASURED LADDER, re-measured because round 1's own
                    # figures did not reproduce and the corrected numbers are
                    # what a future reader should trust (agent review round 2,
                    # R2-2). Same harness, `session.running = True`, counting
                    # interrupts after a double-click on a word:
                    #
                    #   press 1  copies, draft intact       interrupts 0
                    #   press 2  clears the draft           interrupts 0
                    #   press 3  interrupts                 interrupts 1
                    #   press 4  interrupts and EXITS       interrupts 2
                    #
                    # An UNTOUCHED composer reaches its first interrupt on press
                    # 2 and exits on press 3. The residual cost is therefore
                    # ONE EXTRA TAP: a user mid-turn who reflexively
                    # double-clicks needs three presses to interrupt where an
                    # untouched composer needs two.
                    #
                    # That cost is ACCEPTED deliberately. The press it adds is
                    # the copy the user actually asked for by double-clicking,
                    # and the alternative — a copy that also interrupts — makes
                    # one keystroke mean two things and is worse. The ladder
                    # stays reachable and monotonic in every case, which is the
                    # property that matters; it is one rung longer, not broken.
                    #
                    # To the END, not the start: that is where a caret lands
                    # after any other "act on this selection" edit, and it
                    # leaves the user positioned to type on after the text they
                    # just took.
                    self.selection = Selection(self.selection.end, self.selection.end)
            elif key == "ctrl+c":
                self.post_message(InterruptRequested())
            else:
                # `super+c` with nothing selected: do nothing, and consume the
                # key so it cannot fall through to `TextArea`'s own
                # `ctrl+c,super+c` -> `copy` binding, which would raise
                # `SkipAction` and let the press keep walking the chain. There
                # is no app-level `super+c` to reach, so the visible result is
                # the same either way — but consuming it here keeps this method
                # the single authority on what the chord means rather than
                # leaving half the answer in an inherited binding.
                pass
            event.stop()
            event.prevent_default()
            return
        if key == "ctrl+d" and not self.text:
            self.post_message(EditorQuit())
            event.stop()
            event.prevent_default()
            return
        # Bang-mode entry. Consumed rather than inserted, matching opencode:
        # the bang is the MODE SWITCH, not the first character of the command,
        # so the buffer the user then types is the command itself. Gated on an
        # EMPTY buffer so a `!` in `echo hi!` or a mid-sentence `wow!` is just
        # a character, and on `_allows_shell` so the aside cannot eat one.
        if (
            not self._shell_mode
            and self._allows_shell
            and not self.read_only
            and not self.text
            and event.character == "!"
        ):
            self.set_shell_mode(True)
            event.stop()
            event.prevent_default()
            return
        if key == "up" and self._caret_at_top_edge() and self._history:
            self._navigate_history(-1)
            event.stop()
            event.prevent_default()
            return
        if (
            key == "down"
            and self._caret_at_bottom_edge()
            and (self._history_index is not None or self._history)
        ):
            self._navigate_history(+1)
            event.stop()
            event.prevent_default()
            return
        if key != event.key:
            # A rewritten vertical chord that no branch above claimed. It cannot
            # be handed to ``super()``, which would resolve the ORIGINAL
            # `alt+up` against TextArea's bindings and find nothing — the silent
            # no-op this rewrite exists to remove. Run the plain arrow's own
            # action instead, which is where TextArea's `up`/`down` bindings
            # would have landed anyway.
            moves = {
                "up": self.action_cursor_up,
                "down": self.action_cursor_down,
                "shift+up": lambda: self.action_cursor_up(True),
                "shift+down": lambda: self.action_cursor_down(True),
            }
            move = moves.get(key)
            if move is not None:
                # `_restart_blink` for the same reason the rest of this method
                # exists: `TextArea._on_key` calls it on every cursor key, and
                # the whole correctness argument for the rewrite is that a
                # chord is INDISTINGUISHABLE from its plain arrow. Inert today
                # — this composer ships `cursor_blink = False`, so there is no
                # timer to reset — but leaving it out would make that claim
                # true only by accident of a setting somewhere else, and the
                # divergence would surface as `⌥↑` parking the caret mid-blink
                # where `↑` leaves it solid (ux round 3, U10).
                self._restart_blink()
                move()
                event.stop()
                event.prevent_default()
                return
        await super()._on_key(event)

    # -- escape/arrow chord coalescing ---------------------------------------
    #
    # WHY THIS EXISTS. There is no single byte sequence for option+arrow. Three
    # encodings are in the field and which one a user gets is a terminal
    # preference, not a platform fact:
    #
    #   \x1b[1;3D   CSI with modifier 3  -> parses to `alt+left`
    #               (Ghostty, kitty, WezTerm, iTerm2 in CSI mode)
    #   \x1bb       readline meta-b      -> parses to `ctrl+left`
    #               (iTerm2's default ⌥← preset, Terminal.app "Option as Meta")
    #   \x1b\x1b[D  Esc-prefixed         -> parses to TWO events: `escape`,
    #               then `left` (iTerm2 and Terminal.app "Esc+", a very common
    #               setting; also what the CSI form degrades to whenever
    #               TEXTUAL_DISABLE_KITTY_KEY is set)
    #
    # The first two reach a binding on their own. The third does not, and before
    # this code it was actively DESTRUCTIVE: the composer saw a bare `escape`,
    # ran its escape action — which at the bottom of `_on_key` posts
    # `StopRequested` — and then moved the caret one character. On that terminal
    # config, pressing ⌥← to fix a typo ABORTED THE AGENT'S TURN (issue #370).
    #
    # THE KEY INSIGHT: THE PARSER HAS ALREADY RESOLVED THE TIMING AMBIGUITY.
    # This widget does not need a wall-clock window at all, because by the time
    # a key reaches `_on_key` the question "was anything typed after that Esc?"
    # is settled. Measured against textual 8.2.8's `XTermParser.feed`:
    #
    #   feed("\x1b")       -> []                      nothing emitted at all
    #   feed("\x1b\x1b[D") -> escape @+0.030ms,       both from ONE parse pass,
    #                         left   @+0.060ms        ~30 MICROseconds apart
    #
    # A lone `\x1b` emits NOTHING until the parser has waited out its own
    # `ESCAPE_DELAY` and concluded that no sequence followed. So a bare `escape`
    # arriving here is already proof that nothing came after it. Conversely the
    # chord's `escape` and `left` are emitted together and land on the message
    # queue back to back, in order — when `_on_key` handles that `escape`, the
    # `left` is ALREADY QUEUED behind it.
    #
    # Therefore the escape action is deferred by exactly ONE message-pump turn
    # (`call_later`), not by a duration. That is sufficient by construction: a
    # queued arrow overtakes the deferred callback and cancels it, and if no
    # arrow was queued the callback runs on the very next turn. Esc-to-stop
    # costs one loop turn instead of 100 ms of wall clock — measured at 9 ms
    # mean / 14 ms worst case even with the event loop under sustained load,
    # against 37 ms / 197 ms for `call_after_refresh`, which waits for a repaint
    # and was rejected for exactly that reason. `set_timer(0, ...)` is not an
    # option: it raises ZeroDivisionError in this version of Textual.
    #
    # WHAT BREAKS IF THE DEFERRAL IS REMOVED: ⌥←/⌥→ goes back to stopping the
    # agent on every Esc-prefixed terminal. Replacing it with a wall-clock delay
    # is a regression in the opposite direction — it makes the app's stop key
    # perceptibly late to buy certainty the parser already provides for free.
    #
    # THE INHERENT LIMIT, stated plainly: a user who presses Esc and then an
    # arrow so quickly that both land in one pump turn is byte-indistinguishable
    # from the chord, and is read as the chord. That is a far tighter race than
    # the old window (one event-loop turn, not a tenth of a second) and cannot
    # realistically be hit by hand.
    #
    # EVERY ESCAPE MEANING IS DEFERRED, not just the bottom one. An earlier
    # revision deferred only the stop-the-turn branch and let the picker-dismiss,
    # model-picker-close and shell-mode-exit branches run synchronously, arguing
    # that a visible list closing late would read as lag and that those presses
    # were unambiguous anyway. Both halves of that argument were wrong:
    #
    #   - The lag argument was sized against a 100 ms wall-clock window. At one
    #     pump turn the delay is ~4 ms and measurably inside the noise of the
    #     synchronous path, so there is no perceptible cost to paying it
    #     everywhere.
    #   - The unambiguity argument does not survive contact with shell mode,
    #     which is a full-width editable buffer with a visible caret — the state
    #     where word movement is MOST useful, not least. It left `⌥←` ejecting
    #     the user from bang-mode with an identical frame, flipping what Enter
    #     does (F1/U1). The picker case was the same defect with visible
    #     feedback (U3).
    #
    # Deferring uniformly also removes a whole class of future bug: any escape
    # meaning added later is covered by default instead of silently becoming the
    # next uncovered branch. The cost is that an escape's effect lands one pump
    # turn later than the keypress, which is not observable to a user.

    def _defer_escape(self, action: Callable[[], object]) -> None:
        """Hold ``action`` for one pump turn, in case a queued arrow follows.

        A second escape arriving while one is pending FLUSHES the first
        immediately rather than replacing it: two presses must produce two
        actions (Esc-Esc is the subagent-cancel escalation ladder, where
        collapsing the pair would silently drop a rung).
        """
        self._flush_escape()
        self._pending_escape = action
        # `call_later`, not `call_after_refresh`: both order correctly behind a
        # queued arrow, but `call_after_refresh` waits for a repaint, which under
        # a busy loop pushed the stop out to ~197 ms — reintroducing the very
        # latency this mechanism exists to avoid.
        self.call_later(self._flush_escape)

    def _flush_escape(self) -> None:
        """Run a held escape action now, if one is still pending.

        Safe to call more than once, and safe to call after
        :meth:`_cancel_escape` has already dropped the action: the queued
        ``call_later`` callback cannot be unscheduled, so it always arrives and
        finds the pending slot empty when the chord claimed it.
        """
        action = self._pending_escape
        # Cleared BEFORE the action runs: the action can post messages that
        # re-enter the composer, and a half-cleared pending state would let the
        # same escape fire twice.
        self._pending_escape = None
        if action is not None:
            action()

    def _cancel_escape(self) -> None:
        """Drop a held escape action without running it — the chord case."""
        self._pending_escape = None

    # NO ``super()`` CALL IN EITHER HOOK BELOW, deliberately. Textual dispatches
    # EVERY matching handler it finds walking the MRO (see
    # ``MessagePump._get_dispatch_methods``), so ``Widget._on_blur`` already runs
    # on its own — calling it here as well runs the base handler TWICE, which
    # posts a second ``DescendantBlur`` and left the app mis-tracking focus (it
    # showed "session is still starting" for a booted session). These override
    # points ADD behaviour; they do not chain.

    def _on_blur(self, event: events.Blur) -> None:
        # A held escape must not outlive the composer's focus. Losing focus
        # means the arrow that would have completed the chord is never coming,
        # so the press was a real Esc and its action is owed now — a deferral
        # resolving later would stop a turn the user started after looking away.
        self._flush_escape()
        # Same argument for the click gesture: focus moving elsewhere ends it,
        # and a claim carried across the boundary would absorb a press aimed at
        # whatever the user turned to (R3-3).
        self.retire_barren_click()

    def _on_unmount(self) -> None:
        # Teardown drops the action rather than flushing it: there is no surface
        # left for a stop, a steer-recall or a mode exit to act on.
        #
        # In practice this is a BACKSTOP, not the usual path. A real
        # ``remove()`` blurs the widget first, so ``_on_blur`` has already
        # flushed by the time this runs; only an unfocused teardown reaches here
        # with something still pending. Kept because the invariant it guarantees
        # — the slot is always settled by teardown, so no queued ``call_later``
        # can fire into a widget that is gone — must not depend on focus state.
        self._cancel_escape()

    # -- submit -------------------------------------------------------------
    def _submit(self) -> None:
        # A session transition leaves the ordinary composer visible and editable,
        # but Enter must not post an event that can still reach the old session.
        # Keeping the draft in place also makes both success and failure paths
        # explicit user decisions without introducing a visible transition mode.
        blocked = getattr(self.app, "composer_submission_blocked", None)
        if callable(blocked) and blocked():
            return
        text = self.text
        # Bang-mode is a MODE, but a recalled `! ls` is TEXT that still means
        # the same thing — omp's submit path, which treats a leading bang as
        # the command even when the dedicated mode is off. Detected HERE so
        # the history entry and the message agree about what was sent.
        shell = self._shell_mode or (
            self._allows_shell
            and text.lstrip().startswith("!")
            and not text.lstrip().startswith("/")
        )
        recorded = text
        if shell and self._shell_mode and text.strip() and not text.lstrip().startswith("!"):
            # History of a dedicated-mode submit is stored WITH the bang so
            # Up-arrow recall re-runs as a command rather than as a prompt
            # that happens to look like one. The buffer itself never held
            # the bang (it was consumed on entry), so it is prefixed here.
            recorded = f"! {text}"
        # Checked HERE, before the post, because that is the only place the
        # entry can be prevented rather than removed afterwards — see
        # :meth:`set_records_history`.
        if recorded.strip() and self._records_history:
            self._record_history(recorded)
        self._picker.close()
        # Only the attachments the text STILL REFERS TO. The marker is the
        # authority: pasting three screenshots and deleting two must send one,
        # because the deleted markers are the user saying they changed their
        # mind, and silently sending all three is both surprising and expensive.
        self.post_message(
            EditorSubmitted(text, self.referenced_images(), self._attachments, shell=shell)
        )
        # Leave the mode WITH the buffer: a submit that stayed in bang-mode
        # would leave the placeholder saying "Run a command…" over an empty
        # field the next Enter would treat as a no-op, which reads as a
        # stuck mode. opencode resets to normal on submit for the same reason.
        self.set_shell_mode(False)
        self.clear_content()
        # The submit seam. The composer the claim described is now empty and a
        # turn is starting, so the next Ctrl+C is aimed at that turn — absorbing
        # it would eat a genuine interrupt on a prompt the user just decided was
        # wrong, which is well inside a reaction-time budget (R3-3).
        self.retire_barren_click()

    def referenced_images(self) -> list[ImageContent]:
        """The attachments the buffer still cites, in the order it cites them.

        Reading the TEXT rather than a parallel list is what makes deleting a
        marker delete the attachment, with no bookkeeping on every keystroke to
        keep the two in step. Order comes from the text too, so moving a marker
        moves its image with it.

        A marker the user retyped or duplicated by hand resolves to the same
        image; one whose number was never issued resolves to nothing. Both are
        the same rule — the number names an attachment, and text that names
        nothing sends nothing.
        """
        return resolve_markers(self.text, self._attachments)

    def attachments(self) -> dict[int, Attachment]:
        """The index→image map, for a caller that will hand the draft back.

        A COPY of the mapping, not a list, because identity is the thing that
        has to survive the round trip. Handing back a list and re-keying it by
        position is how the aside came to rebind every image one marker left
        when a single marker did not resolve (review round 18): two walks of
        the buffer under different rules cannot be relied on to agree.
        """
        return dict(self._attachments)

    def adopt_attachments(self, attachments: Mapping[int, Attachment]) -> None:
        """Restore an index→image map onto the buffer's markers.

        For handing a prompt BACK — a draft the aside borrowed, or one restored
        after a reload. Only the numbers the text actually cites are kept, so a
        stash that outlived an edit cannot resurrect an attachment the text no
        longer mentions.
        """
        cited = _marker_indices(self.text)
        self._attachments = {index: attachments[index] for index in cited if index in attachments}
        self._sync_next_marker()

    def _sync_next_marker(self) -> None:
        """Never hand out a number already standing in the text.

        Derived from the BUFFER, never from ``_attachments``: a marker whose
        attachment was dropped but whose text survives — delete then undo — is
        invisible to the map, and numbering over it revived the dead marker
        onto the next pasted image (review round 18). Called at every seam that
        replaces the text wholesale AND immediately before issuing a number:
        markers can arrive as ordinary text — a prompt copied out of the
        transcript and pasted back — which no replacement seam ever sees.
        """
        self._next_marker = max(_marker_indices(self.text), default=0) + 1

    # -- attachment markers as atomic tokens ----------------------------------
    def _marker_span(self, row: int, column: int, *, before: bool) -> tuple[int, int] | None:
        """The marker the caret is about to eat, as ``(start, end)`` columns.

        ``before`` asks about backspace (the marker ending at the caret, or the
        one the caret is standing inside); otherwise about delete (the marker
        starting at the caret, or the one it is inside).

        Standing INSIDE counts for both. A caret in the middle of
        ``[Image #2, 15|68x200]`` is not editing text the user typed — there is
        nothing meaningful to change one character of — so either key takes the
        whole token.

        Which markers get that treatment is the whole subtlety, and the two
        rules below read like a contradiction until you see what separates
        them: **the fragment is forbidden for text the app wrote, and permitted
        for text that merely looks like it.**

        - The app's own marker is ATOMIC. Backspace inside it used to remove
          the closing bracket and leave ``[Image #2, 1568x20`` hanging, which
          is neither prose nor a resolvable reference — the originally reported
          bug, and still the reason this method exists.
        - Anything else is PROSE, and editable one character at a time, even
          though it matches the same grammar. A number that resolves to
          nothing, or a second citation of one that does, is text as far as the
          frame is concerned; a click that selected all nineteen characters of
          it made the gesture disagree with the paint (design round 18, D7).
          Leaving a fragment there is not a defect: the user is editing their
          own text and we have no claim on it.

        So the chip, this gate and :func:`resolve_markers` are one predicate —
        what is chipped is what is atomic is what is sent — and ``cite()``
        decides it in one place.

        The honest caveat, because a reader will otherwise find it from a
        frame: the app's marker leaves the protected set whenever ``cite()``
        falls back. Edit the tail of the app's own marker while a copy of that
        number is also in the draft and the fallback hands the chip, and this
        gate, to the copy — so the app's marker becomes prose and CAN be
        fragmented (design round 20, D4 residual, ``d4w``). That is the known
        cost of keeping the tail loosely matched; see :func:`cite`.
        """
        line = self.document.get_line(row)
        if "[" not in line:
            # Every keystroke lands here, and a marker always opens with a
            # bracket. `_marker_cells` keeps the same guard for the same reason.
            return None
        for match in IMAGE_MARKER.finditer(line):
            start, end = match.span()
            if not (
                start < column < end
                or (before and column == end)
                or (not before and column == start)
            ):
                continue
            # Resolved only once a candidate EXISTS. `_first_citation_columns`
            # walks the document prefix, and this runs on backspace, delete,
            # ctrl+w and every mouse-down: asking it unconditionally took the
            # common keystroke from O(line) to O(document), measured at 0.34 ms
            # on a 2000-line paste, which is exactly the draft a user then
            # holds backspace through (review round 20).
            #
            # Markers cannot overlap, so this candidate is the only one the
            # caret can be touching - not chipped means not atomic, full stop.
            if start not in self._first_citation_columns(row):
                return None
            return start, end
        return None

    def _delete_marker(self, *, before: bool) -> bool:
        """Delete the marker at the caret, if there is one. Reports whether."""
        if self.selection.start != self.selection.end:
            # A real selection is the user's own range; never widen it.
            return False
        row, column = self.selection.end
        span = self._marker_span(row, column, before=before)
        if span is None:
            return False
        start, end = span
        match = IMAGE_MARKER.match(self.document.get_line(row)[start:end])
        self.delete((row, start), (row, end), maintain_selection_offset=False)
        # The attachment goes with its marker. This is the whole contract:
        # what the text no longer cites is no longer sent, so removing the
        # reference has to remove the payload rather than orphan it.
        #
        # Measured AFTER the delete, and the guard is a UNION because the two
        # halves answer different questions. Four narrower rules were tried and
        # each was wrong in a way the next round found:
        #
        # - On the NUMBER: a foreign citation - a copy of a different prompt's
        #   marker - inherited an attachment it never named (design round 19).
        # - On the marker TEXT: deleting the stale copy that `cite`'s fallback
        #   had chipped took the live image with it, while the marker naming it
        #   was still in the buffer (round 20, D11).
        # - On the deleted TOKEN matching the marker: editing the tail first
        #   made the token differ, so the pop was skipped and the attachment
        #   leaked - then hand-typing any `[Image #1...]` resurrected it, which
        #   design round 16's D1 forbids (round 22).
        # - On `cite` ALONE: its fallback resolves any citation of the number,
        #   so deleting the app's own marker handed the image, the chip and the
        #   send to whatever else mentioned that number - reachable by typing a
        #   bare `[Image #1]` and then deleting the real marker, which is the
        #   previous bullet's forbidden state two keystrokes apart (round 23).
        #
        # So: release when the buffer can no longer cite this attachment AT ALL,
        # OR when the token just deleted was the app's own marker and no copy of
        # it survives. D6's exact duplicate fails both and keeps the image.
        if match is not None:
            self._release_deleted(int(match.group(1)), match.group(0))
        return True

    def _release_deleted(self, index: int, token: str) -> None:
        """Apply the release rule to a marker that was just removed as a token."""
        attachment = self._attachments.get(index)
        if attachment is not None and (
            cite(self.text, index, attachment) is None
            or (token == attachment.marker and attachment.marker not in self.text)
        ):
            del self._attachments[index]

    def _delete_marker_past_spaces(self, *, before: bool) -> bool:
        """Delete the marker separated from the caret only by spaces.

        For ctrl+w, and only for ctrl+w. A paste inserts ``[Image #1, 10x20] ``
        WITH a trailing space, so the caret it leaves is one column past the
        marker's end and :meth:`_marker_span` - which asks about the character
        the caret is touching - correctly finds nothing. Textual's word-delete
        then ate ``] `` and stopped, rebuilding the hanging fragment this whole
        mechanism exists to prevent AND orphaning the attachment, which any
        later ``[Image #1]`` revived (design round 22, D13).

        Backspace and delete deliberately do NOT come through here: at that
        caret the character before really is a space, and eating a whole marker
        instead of it would be a surprise. A word-delete has already said it
        will cross whitespace to find the thing it removes.
        """
        if self.selection.start != self.selection.end:
            return False
        row, column = self.selection.end
        line = self.document.get_line(row)
        edge = column
        while edge > 0 and line[edge - 1] == " ":
            edge -= 1
        if edge == column:
            return False  # no whitespace to cross; the plain check already ran
        span = self._marker_span(row, edge, before=before)
        if span is None:
            return False
        start, end = span
        match = IMAGE_MARKER.match(line[start:end])
        # The spaces go with it: ctrl+w removes the word AND the run it crossed.
        self.delete((row, start), (row, column), maintain_selection_offset=False)
        if match is not None:
            self._release_deleted(int(match.group(1)), match.group(0))
        return True

    def _offset_at(self, row: int, column: int) -> int:
        """``(row, column)`` as an offset into :attr:`text`.

        `+ len(newline)`, not `+ 1`: `self.text` joins with the document's OWN
        separator, and a CRLF buffer - which a paste can carry in - shifted
        every offset by one per preceding line, putting the chip two cells off
        the marker it belongs to (review round 22).
        """
        separator = len(self.document.newline)
        return sum(len(self.document.get_line(r)) + separator for r in range(row)) + column

    def _release_uncited(self, indices: Iterable[int]) -> None:
        """Drop the named attachments if the buffer no longer cites them.

        ``indices`` is the whole guard. :meth:`edit` passes only the
        attachments the removal actually TOUCHED, so an edit elsewhere in the
        draft cannot adjudicate a marker it never went near. Two earlier rules
        both failed by asking a question about the buffer as a whole:

        - "does it still parse" released a marker the user was mid-way through
          repairing, so a backspace thirty columns away destroyed an image
          whose next keystroke restored the text perfectly (round 24);
        - "is the number mentioned anywhere" fixed that only for damage AFTER
          `#N` - damage inside the `[Image #` prefix breaks the mention too -
          while letting an unrelated bare `[Image #1` fragment elsewhere in the
          draft keep a properly deleted image alive for a typed marker to
          revive (round 25).

        Scoping to the touched range answers both, and needs no threshold for
        how damaged a marker may be before it stops counting.
        """
        text = self.text
        for index in indices:
            attachment = self._attachments.get(index)
            if attachment is not None and cite(text, index, attachment) is None:
                del self._attachments[index]

    def action_delete_left(self) -> None:
        # Empty-buffer backspace leaves bang-mode, matching opencode. The
        # bang was consumed on entry, so there is no character to delete:
        # the mode IS the character, and backspace on nothing is the
        # honest inverse of the key that entered.
        if self._shell_mode and not self.text:
            self.set_shell_mode(False)
            return
        if not self._delete_marker(before=True):
            super().action_delete_left()

    def action_delete_right(self) -> None:
        if not self._delete_marker(before=False):
            super().action_delete_right()

    def action_delete_word_left(self) -> None:
        # A marker is ONE word for this purpose too: ctrl+w stopping inside it
        # leaves the same broken fragment backspace used to.
        if self._delete_marker(before=True) or self._delete_marker_past_spaces(before=True):
            return
        super().action_delete_word_left()

    # -- attachment markers as painted objects --------------------------------
    def _first_citation_columns(self, line_index: int) -> set[int]:
        """Columns on ``line_index`` that open the APP's own citation of a marker.

        The chip, the atomic-token gate and the send all have to name the same
        citation, so all three go through :func:`cite`. Keying on "first in
        document order" instead handed the chip to an impostor pasted above the
        real one (design round 19, D4); keying on the number alone chipped a
        hand-typed duplicate (round 18, D4).

        Only called for rows that contain a bracket, so ordinary prose never
        pays for it, and a composer draft is a handful of lines.
        """
        line = self.document.get_line(line_index)
        # Document offset of this line's first column, so a citation found in
        # whole-buffer coordinates can be tested against this row.
        base = self._offset_at(line_index, 0)
        text = self.text
        columns: set[int] = set()
        for index, attachment in self._attachments.items():
            span = cite(text, index, attachment)
            if span is not None and base <= span[0] < base + len(line):
                columns.add(span[0] - base)
        return columns

    def _marker_edges(self, row: int) -> set[int]:
        """Both edge columns of every ATOMIC marker on ``row``.

        The same predicate the chip and the atomic-token gate use — only a
        citation the app itself owns is protected, so a hand-typed lookalike
        stays ordinary prose that a word span may cross (design round 18, D7).

        Used to stop a word span straddling a marker; see :meth:`_word_span`.
        """
        line = self.document.get_line(row)
        if "[" not in line:
            # Same cheap guard as `_marker_span`: no bracket, no marker, and
            # this runs on every multi-click.
            return set()
        chipped = self._first_citation_columns(row)
        edges: set[int] = set()
        for match in IMAGE_MARKER.finditer(line):
            if match.start() in chipped:
                edges.update(match.span())
        return edges

    def _marker_cells(self, y: int) -> list[tuple[int, int, bool]]:
        """``(x_start, x_end, selected)`` for marker cells drawn on screen row ``y``.

        SCREEN row, not document line. The composer soft-wraps, so one document
        line is one-or-more rows and a marker can straddle a break — the wrap
        lands on the space inside ``[Image #1, 1568x200]`` often enough that
        assuming row == line would leave half a chip unpainted at 60 columns.
        ``wrapped_document`` is the only thing that knows where a document
        column ended up once wrapping moved it.

        Cells are reported per SELECTION state rather than per marker, so a
        drag that covers half a marker paints half of it. That is deliberate:
        :meth:`_delete_marker` refuses to widen a real selection, so a
        half-covered marker really does delete by halves, and painting it whole
        would promise an atomicity the next keystroke does not honour.

        The caret's own cell is left out entirely. The chip is opaque, so the
        first version swallowed the caret whole: parking it inside
        ``[Image #1, 1568x200]`` painted all twenty cells one flat navy and the
        composer stopped saying where the next keystroke would land.
        """
        wrapped = self.wrapped_document
        absolute_y = self.scroll_offset.y + y
        if absolute_y >= wrapped.height:
            return []
        # x=0 resolves to the first document column drawn on this row, which is
        # also the wrap offset that opened its section.
        line_index, section_start = wrapped.offset_to_location(Offset(0, absolute_y))
        line = self.document.get_line(line_index)
        if "[" not in line:
            # The hot path — render_line runs on every frame of every keystroke
            # and most composer lines are prose. A marker always opens with a
            # bracket, so this rejects without re-deriving the grammar.
            return []
        offsets = wrapped.get_offsets(line_index)
        section_index = bisect_right(offsets, section_start)
        wraps_on = section_index < len(offsets)
        section_end = offsets[section_index] if wraps_on else len(line)

        selection_start, selection_end = self._selected_columns(line_index, len(line))
        theme = self._theme
        caret_row, caret_column = self.selection.end
        caret = (
            caret_column
            if self._draw_cursor
            and caret_row == line_index
            and theme is not None
            and theme.cursor_style is not None
            else None
        )
        gutter = self.gutter_width
        cells: list[tuple[int, int, bool]] = []
        chipped = self._first_citation_columns(line_index)
        for match in IMAGE_MARKER.finditer(line):
            # A chip is a claim that an image is attached here, so it is painted
            # only when the number RESOLVES, and only at the citation that
            # actually carries it. Painting from the text pattern alone drew a
            # full chip for a marker typed by hand, and for one brought back by
            # undo after its attachment had been dropped (design round 16, D1);
            # painting every resolving citation then drew a second chip for a
            # hand-typed duplicate of a live number, advertising dimensions for
            # an image that `referenced_images` sends once (design round 17,
            # D4). Same rule both times: what is chipped is what is sent.
            if match.start() not in chipped:
                continue
            start = max(match.start(), section_start)
            end = min(match.end(), section_end)
            if start >= end:
                continue  # this marker lives entirely on another row
            runs = _marker_runs(start, end, (selection_start, selection_end), caret)
            for run_start, run_end, selected in runs:
                x_start = wrapped.location_to_offset((line_index, run_start)).x
                if wraps_on and run_end >= section_end:
                    # `run_end` IS the wrap offset, and location_to_offset reads
                    # that as column 0 of the NEXT row. The chip runs to the end
                    # of this row's text instead.
                    x_end = cell_len(
                        expand_tabs_inline(line[section_start:section_end], self.indent_width)
                    )
                else:
                    x_end = wrapped.location_to_offset((line_index, run_end)).x
                cells.append((x_start + gutter, x_end + gutter, selected))
        return cells

    def _selected_columns(self, row: int, line_length: int) -> tuple[int, int]:
        """The columns of document line ``row`` the selection covers.

        An empty range (``(0, 0)`` when the caret is a point, or when the
        selection is on other lines entirely) means "nothing on this line".
        """
        top, bottom = sorted(self.selection)
        if top == bottom or not top[0] <= row <= bottom[0]:
            return 0, 0
        return (
            top[1] if row == top[0] else 0,
            bottom[1] if row == bottom[0] else line_length,
        )

    def _paint_markers(self, strip: Strip, y: int) -> Strip:
        """Repaint the marker cells of an already-rendered row.

        Post-processing rather than a hook into ``TextArea._render_line``: that
        method owns wrapping, tab expansion, the caret and the selection, and
        the only seam a subclass gets is the finished :class:`Strip`.

        The styles go on as ``post_style``. ``Strip.apply_style`` applies its
        argument as the BASE, and every segment ``TextArea`` hands back already
        carries an explicit fg and bg from the theme, so a base style is
        discarded on arrival and the chip would never appear.
        """
        cells = self._marker_cells(y)
        if not cells:
            return strip
        width = strip.cell_length
        edges = sorted({0, width} | {x for start, end, _ in cells for x in (start, end)})
        styles = {
            False: self.get_component_rich_style("text-area--image-marker"),
            True: self.get_component_rich_style("text-area--image-marker-selected"),
        }
        pieces: list[Strip] = []
        for left, piece in zip(edges, strip.divide(edges[1:])):
            selected = next((state for start, end, state in cells if start <= left < end), None)
            if selected is None:
                pieces.append(piece)
                continue
            pieces.append(self._overlay(piece, styles[selected]))
        return Strip.join(pieces)

    @staticmethod
    def _overlay(strip: Strip, style: RichStyle) -> Strip:
        """``strip`` with ``style`` laid ON TOP of each segment's own style."""
        return Strip(Segment.apply_style(strip, post_style=style), strip.cell_length)

    def _slash_runs(self) -> tuple[int, list[tuple[int, int, str]]] | None:
        """The document-column spans to highlight, and the line they sit on.

        Returns ``(line_index, [(col_start, col_end, component_class), …])`` or
        ``None`` when nothing on the buffer is a slash surface. The runs are the
        leading ``/command`` token (always, when the line starts with ``/``) and,
        for a NAME+message command whose typed name is a known team/agent, the
        NAME token — never the free-text message tail, which is the whole point:
        the user sees what is command versus what will be sent.

        Memoized per render pass (CR1). ``render_line(y)`` calls this once per
        VISIBLE screen row, and the runs are identical for every row of a frame —
        they are a property of the buffer and picker state, not of ``y``. The
        cache key captures every input the computation reads, so a keystroke that
        changes any of them (buffer text, the name/command snapshots, the picker
        state that gates the unknown treatment) recomputes on the next row that
        asks, while the other rows of the same frame reuse the result. Composer
        buffers are small, so this is a micro-optimisation, not a hot-path fix —
        but it makes the "cheap on the hot path" claim exact and stops the
        per-row line-list allocation on prose frames.
        """
        # Every value the compute reads, so a stale run can never survive an
        # input change. ``_command_names``/``_name_choices`` are frozensets and
        # ``_argument_commands`` a tuple — all hashable and cheap to compare.
        key = (
            self.text,
            self._command_names,
            self._name_choices,
            self._argument_commands,
            self._picker.mode,
            self._picker.is_open(),
            self._picker.is_pending(),
        )
        cached = self._slash_runs_cache
        if cached is not None and cached[0] == key:
            return cached[1]
        runs = self._compute_slash_runs()
        self._slash_runs_cache = (key, runs)
        return runs

    def _compute_slash_runs(self) -> tuple[int, list[tuple[int, int, str]]] | None:
        """The uncached body of :meth:`_slash_runs` — see its docstring.

        "Recognized" for the command word is membership in ``_command_names`` —
        the same case-insensitive ``name in entry.names`` test the app's
        ``slash_command_for`` uses, reproduced here because editor.py cannot
        import the app it is imported by. An unrecognized word gets the muted
        ``slash-unknown`` treatment so an inert ``/teem`` reads as text that WILL
        be sent — EXCEPT while the command picker is still choosing (``/te``),
        where the word is a prefix in progress, not yet a typo.
        """
        lines = self.text.split("\n")
        first = next((i for i, line in enumerate(lines) if line.strip()), None)
        if first is None:
            return None
        line = lines[first]
        indent = len(line) - len(line.lstrip())
        rest = line[indent:]
        if not rest.startswith("/"):
            return None
        # The command token runs from the slash through the first whitespace.
        word_end = next((i for i, ch in enumerate(rest) if i > 0 and ch.isspace()), len(rest))
        cmd_start, cmd_end = indent, indent + word_end
        word = rest[1:word_end].lower()
        # Single content-line discipline, identical to ``slash_context`` — once a
        # newline follows the command line the buffer is a message body, and a
        # stray command highlight there would contradict "this is prose".
        #
        # EXCEPT for the NAME+message commands. ``/team``·``/agent`` are DEFINED as
        # ``/<cmd> <name> <free-text message>`` where the message is expected to
        # span lines, and the command on the FIRST content line still dispatches
        # as that command across the newline (``slash_command_for`` splits on
        # ``maxsplit=1``, so ``/team lopdev\n…`` is still ``/team``). The command
        # is therefore still live, so its command and name tokens must keep their
        # highlight over a multi-line body — otherwise every token goes dark the
        # instant the user adds a newline, which is exactly the reported bug.
        # Only these commands are exempt; every ordinary command still goes dark
        # on the newline that turns it into abandoned-command prose. This is a
        # LEADING-command rule (the command on ``first``): #250's inline
        # mid-draft commands are a separate, caret-anchored concern that the
        # highlighter has never painted and this change does not touch.
        multiline = len(lines) > first + 1
        if multiline and not self._is_name_argument_command(word):
            return None
        runs: list[tuple[int, int, str]] = []
        if word in self._command_names:
            runs.append((cmd_start, cmd_end, "text-area--slash-command"))
        else:
            # Suppress the "unknown" flash while the word is still being picked:
            # a prefix under an open command list is in progress, not wrong. The
            # recognized and name highlights are unaffected by this gate. On a
            # multi-line buffer the picker is already closed (``slash_context``
            # returns None once a newline follows the leading word), so
            # ``picking`` is naturally False — and an unknown word never reaches
            # here on multiline anyway, since only recognized NAME+message words
            # clear the guard above.
            picking = self._picker.mode is PickerMode.COMMAND and (
                self._picker.is_open() or self._picker.is_pending()
            )
            if not picking:
                runs.append((cmd_start, cmd_end, "text-area--slash-unknown"))
        # The NAME token, only for /team·/agent and only when the typed name is a
        # known team/agent (an exact snapshot hit — a half-typed name stays prose
        # rather than flickering). The name is the first whitespace-delimited
        # token of the command line's argument, read straight from the command
        # line (``rest``) rather than through ``slash_argument``: that helper is
        # caret-anchored and single-line (#250) and returns None both on the
        # multi-line body these commands carry AND whenever the caret sits on the
        # message rather than the name. Deriving from the first content line is
        # byte-for-byte identical to the old single-line ``slash_argument`` path
        # on a single-line buffer (that line IS the whole command there) and
        # keeps painting the name once the message wraps or moves the caret away.
        if self._is_name_argument_command(word):
            _, sep, argument = rest[1:].partition(" ")
            if sep:
                lead = len(argument) - len(argument.lstrip())
                name = argument[lead:].split(" ", 1)[0]
                # ``/team chart`` is #258's two-level form: ``chart`` is a
                # RESERVED subcommand in the first argument slot (the team name
                # the chart wants lives in the SECOND slot, after ``chart ``), so
                # the first token is never a roster name to paint. Guarding it
                # here — mirroring the same reserved-word exclusion in
                # :meth:`_name_switch_hint` — keeps the highlight correct even if
                # a team were literally registered as ``chart`` (talked to as
                # ``/team =chart``, whose first token is ``=chart`` not ``chart``)
                # and, more importantly, means a future change that widened the
                # snapshot or read the second-slot token could not resurrect a
                # ``chart`` mispaint. The first-token rule already makes the
                # realistic case correct (``chart`` is not in the roster); this
                # makes it correct by construction.
                reserved = word in ("team", "teams") and name.lower() == "chart"
                if name and not reserved and name.lower() in self._name_choices:
                    # The argument tail begins one cell past the command token
                    # (its terminating space), then any extra spaces the user
                    # typed before the name.
                    name_start = cmd_end + 1 + lead
                    runs.append((name_start, name_start + len(name), "text-area--slash-argument"))
        return (first, runs) if runs else None

    def _slash_cells(self, y: int) -> list[tuple[int, int, str]]:
        """``(x_start, x_end, component_class)`` for slash cells on screen row ``y``.

        Mirrors :meth:`_marker_cells`: SCREEN row, not document line, so a long
        ``/team <name> <message>`` that soft-wraps maps its document columns to
        the right screen x on whichever wrapped row carries them. The command
        token is short and rarely wraps; the name token can, so the same
        wrap-boundary math the marker pass uses is reused here.
        """
        runs = self._slash_runs()
        if runs is None:
            return []
        line_index, spans = runs
        wrapped = self.wrapped_document
        absolute_y = self.scroll_offset.y + y
        if absolute_y >= wrapped.height:
            return []
        row_line, section_start = wrapped.offset_to_location(Offset(0, absolute_y))
        if row_line != line_index:
            # Only the command line's screen rows can carry the tokens.
            return []
        line = self.document.get_line(line_index)
        offsets = wrapped.get_offsets(line_index)
        section_index = bisect_right(offsets, section_start)
        wraps_on = section_index < len(offsets)
        section_end = offsets[section_index] if wraps_on else len(line)
        gutter = self.gutter_width
        cells: list[tuple[int, int, str]] = []
        for col_start, col_end, component in spans:
            start = max(col_start, section_start)
            end = min(col_end, section_end)
            if start >= end:
                continue  # this token lives entirely on another wrapped row
            x_start = wrapped.location_to_offset((line_index, start)).x
            if wraps_on and end >= section_end:
                # ``end`` IS the wrap offset, which location_to_offset reads as
                # column 0 of the NEXT row; the token runs to this row's text end.
                x_end = cell_len(
                    expand_tabs_inline(line[section_start:section_end], self.indent_width)
                )
            else:
                x_end = wrapped.location_to_offset((line_index, end)).x
            cells.append((x_start + gutter, x_end + gutter, component))
        return cells

    def _paint_slash(self, strip: Strip, y: int) -> Strip:
        """Overlay the slash-command / name highlight on an already-rendered row.

        Foreground-only component styles (see the tcss) laid on as ``post_style``
        for the same reason as the chip: every segment ``TextArea`` returns
        carries an explicit fg/bg, so a base style is discarded on arrival.
        Foreground-only is deliberate — it composes with the cursor's inverse and
        the selection ground without fighting them, so the pass need not exclude
        the caret cell the way the opaque chip does.
        """
        cells = self._slash_cells(y)
        if not cells:
            return strip
        width = strip.cell_length
        edges = sorted({0, width} | {x for start, end, _ in cells for x in (start, end)})
        styles = {component: self.get_component_rich_style(component) for _, _, component in cells}
        pieces: list[Strip] = []
        for left, piece in zip(edges, strip.divide(edges[1:])):
            component = next((comp for start, end, comp in cells if start <= left < end), None)
            if component is None:
                pieces.append(piece)
                continue
            pieces.append(self._overlay(piece, styles[component]))
        return Strip.join(pieces)

    async def _on_mouse_down(self, event: events.MouseDown) -> None:
        """Note a press that landed inside a marker; the release decides.

        Neither ``super()`` nor ``prevent_default``: Textual invokes EVERY
        ``_on_mouse_down`` up the MRO, so ``TextArea``'s still places the caret
        and arms its drag-selection. This only records.
        """
        row, column = self.get_target_document_location(event)
        # `before=False` reads as "at the start of, or inside", which is exactly
        # the marker's own cells — clicking the cell one past the closing
        # bracket is a click on the next character, not on the marker.
        span = self._marker_span(row, column, before=False)
        self._pressed_marker = None if span is None else (row, *span)

    async def _on_mouse_up(self, event: events.MouseUp) -> None:
        """A press and release inside one marker selects the whole marker.

        Selecting it is what makes it removable by the obvious gesture: the
        selection is a real one, so :meth:`_delete_marker` stands aside and
        backspace deletes exactly the span — click, backspace, gone.

        Runs BEFORE ``TextArea._on_mouse_up`` (Editor is first on the MRO), so
        the base class still finalises ``_selecting`` and releases the mouse
        afterwards; it never touches the selection, so nothing overwrites this.

        A drag over the composer is NEVER a copy. Drag-copy is the
        transcript's gesture, where a highlight is read-only text being taken;
        in the composer a highlight is usually the first half of a replace or
        delete, so copying on release clobbered the clipboard with text the
        user was about to throw away. The copy gesture here is explicit:
        highlight, then Ctrl+C — see :meth:`action_copy`.
        """
        pressed = self._pressed_marker
        self._pressed_marker = None
        # A press that became a drag keeps the range ``_on_mouse_move`` built:
        # that range is the user's, not ours.
        if pressed is not None and self.selection.start == self.selection.end:
            row, start, end = pressed
            self.selection = Selection((row, start), (row, end))
            # Not a copy. This selection is the app's own doing — the user
            # clicked one marker, which means "act on this chip" (the gesture
            # exists so backspace can delete it whole), not "take a copy of
            # it". Copying here would put text on the clipboard and raise a
            # receipt for it on a CLICK, which is the same noise a drag-copy
            # would be.
            return

    async def _on_click(self, event: events.Click) -> None:
        """Double-click selects the WORD, triple-click selects the LINE.

        The reported defect: "I can't properly copy via ctrl/cmd+C on
        highlighted text in the composer." Measured under a real pty, the
        highlight the user made by double-clicking was never a composer
        selection at all, so the Ctrl+C that followed found nothing to copy and
        fell through to the interrupt rung — which CLEARS THE DRAFT. The user
        was not failing to copy a selection; they were destroying their prompt
        by asking for one.

        The mechanism is entirely in Textual 8.2.8 and is worth stating,
        because reasoning about it from the widget's own code gets it wrong.
        ``Widget._on_click`` (``textual/widget.py``) reacts to a click chain by
        calling ``text_select_all()`` on chain 2 and on the container at chain
        3. Those are SCREEN selections — ``Screen.selections``, the read-only
        band the transcript uses — and for a ``TextArea`` they are inert twice
        over:

        * the entry it writes is ``Selection(start=None, end=None)``, and
          ``Widget.get_selection`` returns text only for a ``Text``/``Content``
          visual, so a ``TextArea`` contributes nothing to
          ``Screen.get_selected_text()``;
        * nothing paints. The band is applied by ``Content.render_strips``,
          and this widget renders its own lines, so the frame after a
          double-click is IDENTICAL to the frame before it (captured: the
          caret moved, no highlight appeared).

        Meanwhile ``TextArea``'s own ``selection`` — the reactive that both
        paints the composer's highlight and backs ``selected_text`` — is left
        collapsed by the mouse-down, and no base-class handler ever widens it:
        ``TextArea`` binds word/line selection to ``f5``/``f6``/``f7`` only and
        has no click-chain handling whatsoever. So the gesture every user
        reaches for first in an input box selected nothing, showed nothing, and
        then ate the draft.

        Fixed at the source rather than by teaching Ctrl+C to look somewhere
        else: the press is right to ask for ``selected_text``, and the bug is
        that the double-click never produced one. This handler gives the chain
        the document meaning it has in every other editor, using
        ``TextArea``'s own primitives — ``select_line`` for the line, and the
        base class's ``_word_pattern`` boundaries for the word, so what counts
        as a word here is what counts as one for ctrl+arrow navigation rather
        than a second definition drifting beside it.

        ``super()`` is deliberately NOT called. Its only behaviour for this
        widget is the inert screen selection above, and leaving it in place
        costs a real defect: the entry puts this widget in the map the app's
        transcript-copy path walks, so a copy would join text from a widget
        that painted no band. A selection that cannot render and cannot be
        copied is not worth keeping for symmetry.

        Verified empirically rather than reasoned about: with ``super()`` the
        screen map holds ``{Editor(): Selection(start=None, end=None)}``;
        without it the map stays empty while ``selected_text`` is the word
        either way. Note that ``Screen.clear_selection()`` is NOT part of this
        argument — it fires from ``Screen``'s MouseUp handler whenever the
        down and up offsets match, unconditionally, whether or not
        ``selections`` holds anything (``screen.py``), so the entry does not
        cause it. An earlier version of this note claimed it did (agent review
        round 1, R1-5).

        Chains beyond 3 are folded onto the line: ``CLICK_CHAIN_TIME_THRESHOLD``
        keeps counting for as long as the user keeps clicking in place, and a
        fourth click that silently collapsed the selection would read as the
        highlight flickering off.

        Two shapes of row are NOT prose and are answered before the word span,
        because running a word span over either is what turns this gesture back
        into the defect it fixes:

        * **An attachment marker is ATOMIC.** ``[Image #1, 1568x200]`` is a
          token the app put in the buffer, not text the user typed, and
          :meth:`_marker_span` is the one predicate that decides which markers
          get that treatment. A word span over it selected ``Image`` alone, so
          the commonest edit after a double-click — typing over the selection —
          left ``[x #1, 1568x200]`` in the draft: that no longer matches
          ``IMAGE_MARKER``, so :func:`resolve_markers` drops it and THE IMAGE
          IS SILENTLY NOT SENT, with a corrupted token left behind that is
          neither chip nor prose (agent review round 1, R1-1). Chain 1 already
          selected the whole marker through :meth:`_on_mouse_up`; this asks the
          same question rather than inventing a second answer. Only chain 2
          needs to ask — chain 3 takes the whole line, which contains the
          marker entire.
        * **An EMPTY row has no word and no line to take**, so both chains fell
          through to a collapsed range: the frame after the gesture was
          byte-identical to the frame before it, and the Ctrl+C that followed
          found no range and CLEARED THE DRAFT — the reported defect, reached
          through the new gesture on the commonest shape of a real prompt,
          paragraphs split by a blank line (design round 1, D2). A deliberate
          selection gesture answers with a live range; see
          :meth:`_line_break_span`.

        The selection this makes is an ORDINARY one — the same state a drag
        leaves — so every rule that already governs a composer highlight applies
        unchanged: backspace deletes the span, typing replaces it, and Ctrl+C
        copies it through :meth:`action_copy`. It is explicitly NOT a copy on
        its own: like the marker click in :meth:`_on_mouse_up`, selecting is the
        app's response to the gesture, and putting text on the clipboard from a
        CLICK is the same unasked-for write that :meth:`_copy_drag` exists to
        refuse.
        """
        if event.chain < 2:
            # A single click is a caret move, which the base class's mouse-down
            # has already done. Nothing to widen.
            #
            # But a NEAR-MISS PAIR arrives here as two chain-1 clicks, and it is
            # the same user making the same gesture. Textual's chain needs an
            # EXACT cell match as well as its time window (`App._on_event`
            # compares `_click_chain_last_offset`), so a fast double-click that
            # drifts ONE COLUMN is two singles: no range, then Ctrl+C, then the
            # draft is gone (design round 2, D2-1). The jittery clicker and the
            # slow clicker are the same person, and widening only the time
            # window bought them half the protection.
            #
            # So a second click that lands within the chain WINDOW but off by a
            # few cells still counts as a barren multi-click for the purpose of
            # the next Ctrl+C. It deliberately does NOT select — guessing a
            # range from a gesture Textual scored as two singles would paint a
            # highlight the user did not ask for. It only declines to let the
            # next press clear the draft.
            now = time.monotonic()
            previous = self._single_click_at
            self._single_click_at = (now, event.screen_offset)
            if previous is not None:
                elapsed = now - previous[0]
                # ABSOLUTE distance on BOTH axes. `Offset.clamped` restricts x
                # and y to values above zero rather than taking a magnitude, so
                # every negative component read as ZERO drift and a second click
                # ANY distance to the left — or on an earlier row — landed
                # inside the band and armed the claim (agent review round 3,
                # R3-2). The stated bound is a radius around the first click,
                # and a radius has no sign; measured end to end, `dx=-6` and
                # `dy=-3` both armed while their positive twins correctly did
                # not.
                offset = event.screen_offset - previous[1]
                drift = max(abs(offset.x), abs(offset.y))
                if elapsed <= _NEAR_MISS_WINDOW_S and drift <= _NEAR_MISS_CELLS:
                    self._barren_click_at = now
            return
        # A real chain supersedes the near-miss bookkeeping.
        self._single_click_at = None
        row, column = self.get_target_document_location(event)
        try:
            line = self.document[row]
        except IndexError:  # pragma: no cover - the resolver clamps to a real row
            # `get_target_document_location` clamps BOTH coordinates through
            # `wrapped_document.offset_to_location`, so a click below the last
            # line resolves to the last line rather than raising (measured: a
            # click 40 rows below a one-line document answers `(0, 3)`). Kept
            # as a guard because that clamp is Textual's promise and not this
            # widget's — the earlier comment here claimed instead that a click
            # always lands on a real row, which is not what makes this safe
            # (agent review round 1, R1-4).
            return
        # CLAMPED ONCE, BEFORE THE GATE, so the marker predicate and the word
        # span ask about the SAME column. The composer is wider than its text,
        # so a click to the right of a line resolves to a column PAST its last
        # character. `_word_span` clamps internally and `_marker_span` does not,
        # and that disagreement reintroduced R1-1's exact data loss: on
        # `look at [Image #1, 1568x200]` the gate answered `None` for raw column
        # 28/33/68 while the word path clamped straight back INTO the marker and
        # took the closing `]` as its own punctuation span. Typing over that
        # bracket corrupts the token, `resolve_markers` drops it, and the image
        # is silently not sent — reachable from a bare paste with no editing at
        # all (agent review round 2, R2-1).
        #
        # The clamp is the word path's own rule (`min(column, len(line) - 1)`),
        # lifted here so one column feeds both branches rather than each
        # deciding separately where the line ends. `max(..., 0)` keeps an empty
        # line at column 0, which is the row `_line_break_span` answers.
        column = min(column, max(len(line) - 1, 0))
        if not line:
            self.selection = self._line_break_span(row)
        elif event.chain == 2:
            # Computed INSIDE the chain-2 branch: chain 3 takes the whole line
            # and the blank row takes its break, so neither can use a marker
            # span, and evaluating one for them read as though they consulted it
            # (agent review round 2, R2-6).
            marker = self._marker_span(row, column, before=False)
            if marker is not None:
                start, end = marker
            else:
                # Not ON a marker, but a word span may still RUN INTO one: the
                # bracket and the space beside it are both non-word characters,
                # so `_word_pattern` merges them into a span that eats a
                # marker's edge. The edges go in as barriers so the span stops
                # at the token instead of straddling it (R2-1).
                start, end = self._word_span(line, column, self._marker_edges(row))
            self.selection = Selection((row, start), (row, end))
        else:
            self.select_line(row)
        # Record that THIS range came from the click chain, so the Ctrl+C rung
        # can tell a reflexive selection from a deliberate one (R1-2). Written
        # after the assignment because `watch_selection` clears it on every
        # change — including the one just made.
        self._click_selection = self.selection
        # A multi-click that produced NO RANGE is a BARREN gesture, and the next
        # Ctrl+C must not take the interrupt rung. See `barren_multi_click`.
        self._barren_click_at = (
            None if self.selection.start != self.selection.end else time.monotonic()
        )
        # Consumed so the base class's screen-selection path never runs; see the
        # docstring on why its result is inert for a TextArea and actively
        # harmful to keep.
        event.stop()
        event.prevent_default()

    def _line_break_span(self, row: int) -> Selection:
        """A LIVE range for a multi-click on an empty row.

        An empty row has no word and no line content, so both chains previously
        left the selection collapsed. That is the reported defect in miniature:
        a deliberate selection gesture answered with a frame byte-identical to
        the one before it, after which the user presses Ctrl+C, finds no range,
        and the press falls through to the rung that CLEARS THE DRAFT (design
        round 1, D2). The blank line between two paragraphs is the commonest
        shape of a real prompt, so this is reachable by ordinary use.

        The range taken is the row's own LINE BREAK — ``(row, 0)`` to
        ``(row + 1, 0)`` — which is the honest answer to "select this line":
        the only character the line has is the newline that ends it, so a copy
        yields ``\\n`` and a backspace deletes the blank line. That keeps the
        gesture's meaning (a range over what was pointed at) and keeps D17's
        ladder intact for the case D17 is about — a SINGLE click still leaves
        no range, so Ctrl+C still interrupts and still clears the draft.

        The LAST row is the exception and must stay collapsed: there is no
        following row to reach, so a range there would have to run backwards
        into the previous line and take a character the user did not point at.
        That also keeps the empty composer correct — one empty row, no break to
        take, nothing selected, which design round 1 (D7) verified as good and
        which this must not regress.
        """
        if row + 1 >= self.document.line_count:
            return Selection((row, 0), (row, 0))
        return Selection((row, 0), (row + 1, 0))

    def _word_span(self, line: str, column: int, barriers: Iterable[int] = ()) -> tuple[int, int]:
        """The word around ``column``, as ``(start, end)`` columns of ``line``.

        Boundaries come from ``TextArea``'s own ``_word_pattern`` so a
        double-click selects exactly what ctrl+left/ctrl+right treat as a word;
        defining "word" a second time here is how the two would drift.

        ``barriers`` are extra columns a span may not CROSS — the edges of the
        app's own attachment markers. **They are expected to be TOKEN
        BOUNDARIES: a barrier landing inside a word splits it**, so
        ``barriers=[12]`` on ``look at [Image #1, …]`` yields ``'Ima'`` rather
        than ``'Image'``. That is correct for marker edges, which never fall
        mid-word, and it is the precondition a future caller has to meet —
        passing, say, syntax-highlight offsets would return half words rather
        than misbehave visibly (agent review round 3, R3-5). Out-of-range,
        negative, unsorted and duplicated values are filtered and degrade to the
        ordinary word span.

        ``_word_pattern`` sees a line as
        characters, so it happily runs one span across a marker edge: ``]`` and
        the space after it are both non-word characters, so a bare paste's
        ``[Image #1, 1568x200] `` gave the trailing space the span ``'] '``,
        taking the closing bracket of an ATOMIC token with it. Typing over that
        corrupts the marker, :func:`resolve_markers` drops it, and the image is
        silently not sent — R1-1's data loss reached from the other side of the
        token (agent review round 2, R2-1). The same run at the opening edge
        gave ``' ['``.

        Passing the edges as barriers keeps ONE definition of "word" rather than
        adding a second: the boundaries are still ``_word_pattern``'s, with the
        marker's own edges added to them. That is what atomicity already means
        everywhere else in this widget — a marker is a single object, so a span
        either takes it whole (the gate above does that) or does not touch it.

        A click on whitespace selects the whitespace run rather than jumping to
        a neighbouring word: the user pointed at a gap, and silently selecting
        the word to its left would take text they did not point at. Clamped to
        the line so a click past the last character (the composer is wider than
        its text) selects the final word rather than an empty range at the end.

        Two consequences of deferring to ``_word_pattern`` are known and
        ACCEPTED, both recorded in design round 1 rather than demanded:

        * a punctuation run is its own span, so ``run the thing. then`` takes
          ``". "`` including the trailing space, and ``don't``/``well-known``
          split at the apostrophe and the hyphen where macOS and browsers would
          take the whole word (D5). Fixing that means a SECOND definition of
          "word" living beside ctrl+arrow's, and the two drifting apart is a
          worse defect than the one it cures: the boundary a user learns from
          navigating is the boundary they then expect from a double-click.
        * a click in a gap yields one space and a truthful ``copied 1
          character`` receipt (D6). It follows from the whitespace rule above
          and is the honest answer to where the pointer was.

        The vertical analogue of the clamp is NOT a defect here: the composer
        auto-sizes to its content (`height: auto`, capped at 8 rows), so it
        never owns a row below its last line and a click cannot land in empty
        space under a short draft. Measured on one-, two- and thirty-line
        drafts: spare rows are always zero (agent review round 1, R1-3).
        """
        if not line:
            return (0, 0)
        column = min(column, len(line) - 1)
        # Boundaries are the ZERO-WIDTH positions between a word and a non-word
        # character, which is what `_word_pattern` matches; bracketing them with
        # the line's own ends turns them into spans. Marker edges join them as
        # boundaries of the same kind, deduplicated and ordered so the pairwise
        # walk below still sees ascending, non-overlapping spans.
        bounds = sorted(
            {
                0,
                len(line),
                *(m.start() for m in self._word_pattern.finditer(line)),
                *(edge for edge in barriers if 0 < edge < len(line)),
            }
        )
        for start, end in zip(bounds, bounds[1:]):
            if start <= column < end:
                return (start, end)
        return (bounds[-2], bounds[-1])  # pragma: no cover - column is clamped above

    def _copy_drag(self) -> None:
        """A composer drag never copies. Kept as a named no-op so the tests
        that used to drive this path still have a method to name, and so a
        future reader looking for the old release-copies rule finds the
        reason it is gone rather than a silent absence.

        The release-copies rule was imported from the transcript, where it is
        right: a highlight over an answer is read-only text being taken, and no
        key can carry it (below). In the composer the same gesture is usually
        the first half of an EDIT — drag a phrase to retype it, drag a word to
        delete it — so a copy on release clobbered the user's clipboard with
        text they were about to throw away. Reported: "when you highlight in
        the composer it automatically copies … often you're just highlighting
        to select and potentially cut/delete a part, the copy can end up
        clearing something you have in the clipboard." So the release copies
        NOTHING: a composer highlight is selection, not copy. The copy gesture
        is explicit — highlight, then Ctrl+C — see :meth:`action_copy`.

        Why no other key can substitute, and therefore why the transcript's
        release-copies rule exists there at all:

        * **The release does not reach the app.** ``OperatorApp.on_text_selected``
          copies ``Screen.get_selected_text()``, but a ``TextArea`` never
          contributes to a screen selection. ``TextArea._watch_selection`` calls
          ``app.clear_selection()`` on every caret move, and the mouse-down that
          begins a composer drag moves the caret — so ``Screen.selections`` is
          wiped on the first event of the gesture and stays empty for the rest
          of it. The base class also captures the mouse on press, so ``Screen``'s
          own select machinery is bypassed and ``_select_state`` never leaves
          ``None``.

        * **No key can rescue it on its own.** ``TextArea`` binds
          ``ctrl+c,super+c`` to ``action_copy``, and neither binding is what
          runs here: :meth:`_on_key` consumes BOTH chords first, Ctrl+C
          because it is also this app's interrupt and Cmd+C so the two keys
          share one route. That is deliberate — the interrupt cannot become
          conditional on a live highlight, and a copy chord that skipped
          ``_on_key`` skipped the click-chain collapse with it.
          :meth:`_on_key` therefore routes the press to :meth:`action_copy`
          itself when a real range is live, which is the one copy sequence the
          composer has.

          Whether cmd+C ARRIVES at all is the terminal's decision, not this
          widget's: a kitty-protocol terminal forwards it as ``super+c``
          (Ghostty measured; kitty, WezTerm, foot, Alacritty and contour
          reported), and Terminal.app consumes it and delivers zero bytes
          (measured). Ctrl+C is the chord that works on both, which is why it is
          the one the UI advertises. An earlier version of this note said cmd+C
          never arrives anywhere; that was measured only against Terminal.app.
          See ``docs/evidence/cmd-chords/MEASURED.md``.

        An earlier version of this method copied on an ARMED release
        (``copy_on_release``, set by an explicit copy so the next drag
        matched the transcript's). That was a hidden mode whose lifetime
        could not be stated correctly: the arm outlived the highlight that
        authorised it (review round 1, F1) and nothing on screen showed it
        was on (design round 1, D2). Dropped rather than patched — a
        composer copy is always the explicit press, never a drag.
        """
        return

    @property
    def copy_in_flight(self) -> bool:
        """Is a copy gesture still in progress?

        THE predicate the app's Ctrl+C rung asks, named once here rather than
        duck-typed from outside. It was previously two `getattr` probes into
        this class's privates, which a rename would have degraded silently to
        "no copy in flight" — i.e. to always clearing the draft — with no test
        failing (R17 MINOR-2). A property breaks loudly instead, and gives the
        question a name that says what it means.

        ONE moment, not two. This used to also test a `_copy_gesture` flag
        covering the window after a release in which the copied highlight was
        still on screen — the release-copies rule the composer no longer has.
        Dropping that rule (see :meth:`_copy_drag`) removed the only place the
        flag was ever set, so it sat permanently False and this predicate had
        quietly reduced to `_selecting` alone. It is spelled that way now: a
        term that can never be true is not protection, it is a reader being
        told the key is deferred through a window that does not exist.

        The explicit press needs no window of its own. `action_copy` documents
        why the press deliberately claims no gesture flag: a highlight outlives
        the press that copied it, so deferring the NEXT Ctrl+C for as long as
        the highlight happens to remain is D17/D20's lost draft. A live range
        makes the next press a copy on its own merits, and collapsing the caret
        hands the key back to the draft and interrupt rungs.
        """
        return self._selecting

    @property
    def barren_multi_click(self) -> bool:
        """Did a multi-click JUST happen and produce no range?

        The predicate the app's Ctrl+C rung asks before clearing the draft,
        named here for the same reason :attr:`copy_in_flight` is: the app must
        not reach into this widget's privates and degrade silently to "no" — the
        answer that costs a draft — if a field is ever renamed.

        It covers the two remaining paths to the sentence this whole change
        exists to eliminate, "my draft disappeared" (design round 2):

        * **D2** — the blank LAST row. :meth:`_line_break_span` must stay
          collapsed there (there is no following row to reach, and widening
          backwards would take a character the user did not point at, which is
          also what keeps the empty composer correct — D7). But shift+enter is
          this composer's newline, so a user starting a second paragraph SITS on
          a blank last row while they think. Double-clicking there painted a
          byte-identical frame and the Ctrl+C that followed cleared the draft.
        * **D2-1** — the near-miss pair. See :meth:`_on_click`.

        Deliberately NOT "is some selection live". That is D17's lost-draft bug:
        a selection is STATE that persists until the caret moves, so a highlight
        made minutes ago diverted the interrupt, armed nothing, and the second
        reflexive tap exited the app with the draft never filed. This is a
        bounded, recent, GESTURE-scoped fact instead — it expires on its own
        after :data:`BARREN_CLICK_WINDOW_S`, so the exit ladder is always at
        most one stale window away and never permanently unreachable.

        It suppresses exactly ONE press. The claim is retired by the rung that
        reads it, so a user who genuinely wants to interrupt presses again and
        gets the ordinary ladder.

        THREE BOUNDS, and the property only holds with all three (agent review
        round 3, which found one defect per missing bound):

        * **Armed narrowly.** The near-miss branch measures an absolute radius
          on both axes; `Offset.clamped` made it one-sided, so any leftward or
          upward drift armed it from any distance (R3-2).
        * **Read narrowly.** The app's rung takes it only when there is a draft
          to protect. On an empty composer it protected nothing and swallowed a
          genuine interrupt, chainably (R3-1).
        * **Retired eagerly.** :meth:`retire_barren_click` drops it at every
          seam that ends the gesture — an edit, a whole-buffer replacement, a
          submit, a blur, a turn starting — rather than waiting for the window.
          The window bounds how long the claim can be wrong; it does not stop it
          being wrong (R3-3, and R4-1 for the replacement seam).
        """
        at = self._barren_click_at
        return at is not None and (time.monotonic() - at) < BARREN_CLICK_WINDOW_S

    def retire_barren_click(self) -> None:
        """Spend the barren claim, so it suppresses one press and no more.

        Separate from reading it because the rung that declines to clear the
        draft is also the moment the gesture is over: leaving the claim up would
        make a SECOND press a no-op too, and two silent presses is the shape of
        an unreachable exit ladder.

        ALSO CALLED FROM EVERY SEAM THAT ENDS THE GESTURE, not only from the
        rung that spends it. The claim answers "did a multi-click JUST happen
        and produce nothing?", and the window alone does not make that true — it
        only bounds how long it can be wrong for. A user who clicked, TYPED,
        submitted and then reached for Ctrl+C to stop the turn they had just
        started was still inside the window and lost the interrupt (agent review
        round 3, R3-3).

        So it retires wherever the gesture provably ended: a document edit (the
        hand is on the keyboard, the click is over), a whole-buffer replacement
        (:meth:`load_text` — the funnel ``edit()`` does NOT run for, so history
        recall and a compaction hand-back reach it and nothing else, R4-1), a
        submit (the composer it described has been emptied and a turn has
        started), a blur, and a turn starting under it. Each of those makes the
        next Ctrl+C mean something the click cannot speak for.
        """
        self._barren_click_at = None
        # The near-miss bookkeeping is half of the same claim: leaving the first
        # click's timestamp behind lets a later, unrelated click pair with it and
        # re-arm from a gesture the user has already moved on from.
        self._single_click_at = None

    def action_copy(self) -> None:
        """Copy the live range. THE composer's copy gesture.

        The composer cannot rely on Textual's binding reaching this method on
        its own: Ctrl+C is the app's interrupt (see :meth:`_copy_drag`), and
        the inherited ``ctrl+c,super+c`` binding reaches this action WITHOUT
        passing any :meth:`_on_key` branch, so the click-chain collapse is
        skipped. (`_on_key` runs first and a binding fires only on the keys it
        does not consume — see the BINDINGS table for the measurement. The
        earlier wording here said the binding "bypasses" `_on_key`, which
        inverts the rule; code round 2, F6.) So :meth:`_on_key` routes both
        copy chords here directly. This override exists so
        that path — and any other caller of the ``copy`` action — writes the
        clipboard through the SAME message a transcript copy uses, rather than
        ``super().action_copy()``: Textual's base writes silently, and this
        app's rule is that a copy says so. One clipboard write and one receipt
        for every gesture, or the toast becomes evidence about which key
        carried it.

        No gesture claim is raised here, and that is deliberate rather than an
        omission. A press is over the instant it lands, but the highlight it
        copied outlives it — that range sits there until the caret moves — so a
        claim armed by the press would defer the NEXT Ctrl+C for as long as the
        stale highlight happened to remain: the user presses on a minutes-old
        range expecting the draft rung and gets a re-copy instead, with the exit
        ladder's second tap a lost draft away (D17/D20). A flag for that window
        did once exist (``_copy_gesture``) and was deleted once the composer
        stopped copying on release, because nothing set it any more. The receipt
        flag IS set — the toast it drives is a claim about the clipboard, and
        editing the copied text falsifies it (D3).

        A live range on a subsequent press copies AGAIN rather than
        interrupting: that is the explicit-copy rule itself, not a leftover
        deferral. Collapsing the caret (an arrow, a click) is what hands the
        key back to the draft/interrupt rungs.
        """
        text = self.selected_text
        if not text:
            raise SkipAction()
        self._copied = True
        self.post_message(EditorCopied(text))

    @property
    def resting_placeholder(self) -> str:
        """The placeholder for a composer in no special mode.

        ONE answer now: :data:`DEFAULT_PLACEHOLDER`. This used to choose
        between that and a variant naming the paste key, retiring the hint once
        the user had pasted. The hint is gone — the native ``Cmd+V`` works on
        every terminal that forwards it, so a Mac user reaches for the gesture
        they already know and it simply works, and permanent composer real
        estate is too expensive for a key that no longer needs teaching.
        Terminal.app still cannot deliver the chord and ``ctrl+v`` remains the
        fallback there; ``/help`` documents both, which is the right home for a
        conditional statement.

        Kept as a PROPERTY rather than inlined at its four call sites: the
        modes that own the composer's voice (bang-mode, the aside, read-only,
        the subagent page hand-back) each need to ask "what does a resting
        composer say?" on the way out, and one authority is what stops them
        drifting apart if that answer ever gains a second case again.
        """
        return DEFAULT_PLACEHOLDER

    def watch_selection(self, selection: Selection) -> None:
        """Retire the CLICK-CHAIN claim when the highlight it describes changes.

        ``_copied`` is edit-scoped and deliberately survives this seam: `edit`
        and `load_text` clear it, because the receipt on screen is a claim about
        text the user can still see, and moving the caret is not the user
        editing that text. Posting ``EditorCopyStale`` here would take down a
        toast that is still perfectly true (design round 1, D3; R18-1).

        This seam used to also retire a ``_copy_gesture`` flag standing for "a
        hand is still completing a copy", which covered the window between a
        release-copy and its highlight leaving the screen. The composer has no
        release-copy any more (see :meth:`_copy_drag`), so nothing set that flag
        and nothing here could retire it; it and its ``_copied_selection``
        companion are gone rather than left as a lifetime rule with no subject.
        The explicit press needs no such window — see :attr:`copy_in_flight`.
        """
        super_watch = getattr(super(), "watch_selection", None)
        if super_watch is not None:
            super_watch(selection)
        # The click-chain claim has the same lifetime for the same reason: it
        # describes ONE highlight, so it ends the moment that stops being the
        # highlight on screen. A bare flag would outlive its subject and hand
        # the exit ladder a range the user made some other way (R1-2).
        # Spelled as an explicit `!=`: `not in (None, selection)` reads as a
        # set-membership test on a field whose whole point is the identity of
        # ONE specific range (agent review round 2, R2-4).
        claim = getattr(self, "_click_selection", None)
        if claim is not None and claim != selection:
            self._click_selection = None
        # A caret move changes the ghost's answer: it renders AT the caret, so
        # one set while the caret sat at the end of `/mcp login` renders
        # mid-word once the caret moves back into it (`/mcp loGHOSTgin notion`,
        # reproduced) — gate 1 is what refuses that, and this is where the
        # question gets re-asked.
        #
        # RE-DERIVED, not merely cleared. Clearing was cheaper but made two
        # routes to the same caret position disagree: `left` then `right`
        # restored the preview (``action_cursor_right`` re-derives in its
        # `finally`) while `left` then `end` left it blank, with the same caret
        # and the same open list (review round 1, U5). Asking the gates is the
        # only answer that cannot depend on which key got you here.
        #
        # `getattr` for the same reason as above — a reactive watcher can fire
        # during base-class construction, before this subclass's attributes
        # exist, and `_ghost_completion` reads several of them.
        if hasattr(self, "_picker"):
            self._sync_ghost()

    # -- paste ----------------------------------------------------------------
    async def _on_paste(self, event: events.Paste) -> None:
        """Attach a pasted image, from a path OR from the system clipboard.

        Textual's ``Paste`` carries TEXT only — there is no binary channel at
        the terminal, so an image never arrives as bytes here. There are two
        ways it can still reach the composer, and both are handled:

        **A path in the text.** A drag-and-drop lands this way on every
        terminal, and a clipboard image lands this way in **cmux**, which
        watches the pasteboard, writes
        ``$TMPDIR/clipboard-<stamp>-<hash>.png`` and bracket-pastes that
        filename. That is the branch this widget shipped with, and it is why
        the gap below stayed invisible: cmux is where this code was developed,
        so ``Cmd+V`` on a screenshot appeared to work everywhere.

        **An EMPTY paste.** Kept as belt and braces, NOT as the fix it was
        once claimed to be. The original reasoning here — that a textless
        pasteboard makes the terminal send a bare ``ESC[200~ ESC[201~``, which
        Textual's ``XTermParser`` turns into ``Paste(text='')`` — is false
        about the terminal, and the parser half of it is true but irrelevant.
        MEASURED with a raw-mode PTY probe: with a PNG on the pasteboard,
        Terminal.app delivers **zero bytes** on ``Cmd+V`` and beeps, and a
        kitty-protocol terminal forwards the chord as ``super+v`` rather than
        as a paste. Either way there is no ``Paste`` event, so this branch
        never ran on the emulators it was written for and issue #372 stayed
        open. The reachable routes are ``Ctrl+V`` and ``Cmd+V`` ->
        :meth:`action_system_paste`; see this class's ``BINDINGS`` and
        :mod:`local_operator.clipboard` for the byte table.

        This branch survives because a host that DOES synthesise an empty paste
        (a multiplexer, a helper bracketing an empty selection) still reaches
        the clipboard through it, it costs nothing when nothing sends one, and
        deleting it would drop a working route on the strength of two
        emulators' behaviour — the exact single-environment reasoning that
        caused the original bug. Finder's ``Cmd+C`` is handled on both routes
        alike: it puts a ``public.file-url`` flavor on the pasteboard and is
        routed back into the path branch.

        Anything that is not a readable image path is left alone: this returns
        without touching the event, and ``TextArea._on_paste`` inserts it as
        usual.

        NOTE the dispatch rule, which is not obvious and cost a real bug here.
        Textual calls EVERY ``_on_paste`` up the MRO, so the base handler runs
        on its own — ``await super()._on_paste(event)`` does not delegate to
        it, it runs it a second time, and an ordinary text paste came out
        duplicated. Suppressing the base is ``prevent_default``; letting it run
        is doing nothing.

        The ``await`` below is safe against that rule even though it lands
        BEFORE ``prevent_default``. ``MessagePump._get_dispatch_methods`` is a
        generator that re-checks ``_no_default_action`` at the top of each MRO
        step, and the pump fully awaits one handler before resuming it, so the
        base handler cannot start while this one is suspended. The same
        sequencing is why two fast pastes cannot interleave: the pump dispatches
        one message at a time, so marker issuance stays in paste order.
        """
        if not event.text.strip():
            # A payload with no text in it is the clipboard-image signal. The
            # clipboard is consulted, but the event is consumed ONLY if that
            # produced an attachment — the same shape the path branch below
            # uses, and the correction from review round 1 (F1/D1).
            #
            # Consuming unconditionally was a real regression on a gesture that
            # has nothing to do with images: an indent, a tab or a blank line
            # copied and pasted into the composer is ordinary in a multi-line
            # prompt, it worked before this feature existed, and it silently
            # vanished with a toast about images the user was not pasting. The
            # code cannot tell a terminal-synthesised `""` from user-authored
            # whitespace by inspecting the payload, so it stops trying: falling
            # through inserts whitespace the user did have on their clipboard,
            # and for the genuinely empty payload inserting `""` is a no-op
            # nobody can see.
            if event.text:
                # WHITESPACE THE USER COPIED. It reaches this branch because
                # the payload has no text in it, but it is not the terminal's
                # empty-paste signal, and the clipboard is not consulted at
                # all: whatever an image read found, the user asked for these
                # characters on this keypress and trading them for a marker
                # they did not request is the round-1 D1 defect at one tenth
                # the reach (round 2, D9 — reproduced here before fixing:
                # pasting an indent with a PNG on the pasteboard replaced the
                # indent with `[Image #1, 1568x200]`).
                #
                # Skipping the read also makes an ordinary indent paste cost
                # nothing again, rather than paying a clipboard round trip an
                # editing gesture never needed (round 2, U7).
                return
            attached = await self._attach_clipboard_image()
            if attached is None:
                return
            event.prevent_default()
            event.stop()
            self.insert(attached)
            return
        attached = await self._attach_pasted_images(event.text)
        if attached is None:
            return
        event.prevent_default()
        event.stop()
        # Posted from the PATH branch too, not just the clipboard one. A held
        # notice is falsified by an image arriving, whichever route delivered
        # it — and this is the route the notice's own advice sends the user
        # down ("paste a file path instead"), so without this the card that
        # gave the instruction reappears to deny it worked (round 2, D8/D3).
        self.post_message(EditorPasteAttached())
        self.insert(attached)

    async def _attach_clipboard_image(self, *, allow_text: bool = False) -> str | None:
        """Read the clipboard and attach (or insert) whatever is on it.

        Shared by both callers: the empty-paste branch above, and
        :meth:`action_system_paste`, which is the one that actually fires
        outside cmux. ONE implementation deliberately — the two routes must
        produce byte for byte the same marker, the same bounds and the same
        notices, and the original bug was precisely a second route behaving
        differently from the one that got tested.

        Shapes, tried in the order they are likely. IMAGE BYTES first, the
        reported case — a screenshot on the pasteboard. Then FILE URLS, which
        is Finder's ``Cmd+C``, routed into :meth:`_attach_pasted_images` so a
        copied file behaves exactly like the same file dragged in, down to the
        all-or-nothing rule for a multi-file selection.

        ``allow_text`` is what separates the two callers, and the difference is
        not cosmetic. The empty-paste branch is reached only for a payload the
        TERMINAL sent, so inserting clipboard text there would type characters
        the terminal never delivered. A ``Ctrl+V`` is the user asking for the
        system clipboard by name, and text is the ordinary thing on it —
        declining to insert it would leave the key silently useless in the
        common case, which is the defect Textual's own ``ctrl+v`` already has
        here (it pastes ``App.clipboard``, a buffer nothing outside the app
        writes). Off by default so the narrower caller cannot acquire the
        broader behaviour by accident.

        The read runs in a thread for the same reason the decode does: it
        shells out to ``osascript``/``wl-paste``/``xclip``/PowerShell, and this
        is the keystroke handler. The measured macOS read is ~0.6 s for an 8 MB
        Retina screenshot, which is a visible freeze inline and nothing at all
        off the loop. ``local_operator.clipboard`` bounds the WHOLE read \u2014 both
        shapes, every subprocess \u2014 with one 2 s deadline and never raises, so a
        wedged clipboard daemon costs one pause.

        The read is bounded by :data:`~local_operator.clipboard.
        MAX_CLIPBOARD_READ_BYTES` and NOT by :data:`MAX_ATTACHMENT_BYTES`.
        Those are different budgets and conflating them broke the exact gesture
        this branch exists for (review round 1, U1): a native screenshot on a
        Retina display is 8.4-8.5 MB on the pasteboard and bounds down to
        0.28 MB, so capping the READ at the 4 MB attachment budget threw the
        screenshot away before :func:`~local_operator.imaging.
        bound_image_for_model` \u2014 whose entire job is to make it attachable \u2014
        could run. ``_attach_image_bytes`` remains the authority on what may be
        attached, and applies the attachment cap after the resize.

        Returns ``None`` when nothing was attached, and posts
        :class:`EditorPasteEmpty` on the way out so the app can say so. That
        notice is the other half of the reported bug: before it, a ``Cmd+V``
        that attached nothing was indistinguishable from a broken keyboard. It
        carries the outcomes the app can honestly name \u2014 see
        :class:`EditorPasteEmpty`.

        Every outcome below belongs to a keypress that would otherwise produce
        no visible response at all \u2014 an empty payload the terminal sent, or a
        ``Ctrl+V`` that found nothing to insert \u2014 which is what makes a notice
        right here and noise anywhere else. Whitespace the user copied is
        handled by :meth:`_on_paste` and never reaches this method.
        """
        # ACKNOWLEDGE A SLOW READ. The await below holds this widget's input
        # until the clipboard answers, so a read that outlives
        # `PASTE_READING_NOTICE_DELAY_S` gets a card explaining the pause
        # rather than leaving the composer looking hung (ux round 1, U3).
        #
        # `self.set_timer` and NOT `self.app.set_timer`: the timer has to be
        # set on THIS widget, whose `call_next` queue is still flushed while
        # this action is suspended. The App's is not, and neither is a message
        # posted to this widget. See `PASTE_READING_HOOK` for the measured
        # table and why the obvious simplification reintroduces U3.
        #
        # A timer rather than an unconditional call, because the fast read is
        # the common one and a card that appears and vanishes on every paste is
        # worse than the pause it explains. `set_timer` is stopped in the
        # `finally`, so a read that beats the delay never raises anything.
        notice: Callable[..., object | None] | None = getattr(self.app, PASTE_READING_HOOK, None)
        raised = False
        raised_at = 0.0
        reading_notice = None
        # The owner of the card THIS paste raised, returned by the hook.
        # Carried onto the deferred retirement so a later paste's card
        # cannot be dismissed by this one's timer (D15 / issue #422). Same
        # class as the steer-receipt guard (PR #452): the owning context
        # rides the event, and a mismatch drops the delivery.
        paste_owner: object | None = None

        def _raise_reading_notice(show: Callable[..., object | None]) -> None:
            # Nonlocal rather than a return value: a timer callback's result
            # goes nowhere, and the `finally` below has to know whether there
            # is a card to retire. Retiring one that was never raised would
            # withdraw whatever else happens to hold the shared slot.
            nonlocal raised, raised_at, paste_owner
            raised = True
            raised_at = time.monotonic()
            paste_owner = show(True)

        if notice is not None:
            # `notice` is bound as an argument rather than closed over, so the
            # narrowing this branch establishes survives into the callback.
            reading_notice = self.set_timer(
                PASTE_READING_NOTICE_DELAY_S, partial(_raise_reading_notice, notice)
            )
        try:
            contents = await asyncio.to_thread(read_clipboard)
        finally:
            if reading_notice is not None:
                reading_notice.stop()
            if raised and notice is not None:
                # RETIRED HERE, on every outcome, rather than by each branch
                # below. The text branch posts no `EditorPasteAttached` (it
                # attaches nothing to falsify a failure claim) and so left the
                # progress card sitting over a completed paste for its full 5 s
                # on the most common ctrl+v shape there is (design round 2,
                # D7). A progress card is retired by ANY outcome, not only by
                # outcomes that contradict it, so the retirement belongs on the
                # one path every outcome passes through.
                #
                # Held for `PASTE_READING_NOTICE_MIN_S` from when it was
                # RAISED, so a read that finishes just past the delay does not
                # flash the card for a few milliseconds (D10). Scheduled rather
                # than slept: the paste must land now, and only the card waits.
                #
                # SAFE ONLY BECAUSE THE TWO CARDS HAVE DIFFERENT OWNERS. This
                # retirement is a `Toast.withdraw`, which matches on owner and
                # fires late by construction — after an outcome has already
                # replaced the progress card with its own notice. While they
                # shared an owner it destroyed that notice for reads landing in
                # `[DELAY, DELAY + MIN)`, showing the failure card for 0 ms at a
                # 0.74 s read: a keypress with a visible pause and no response,
                # which is issue #372's own symptom (design round 3, D14).
                #
                # Owner matching is not enough against a LATER card of the
                # SAME owner (D15): a second ctrl+v raises another progress
                # card, and this timer would retire it. The per-paste token
                # the hook returned at raise is what names THIS paste's card;
                # a mismatch is a no-op. `self.set_timer` for the same pump
                # reason as above.
                shown_for = time.monotonic() - raised_at
                owned = paste_owner

                def _retire_this_card(
                    show: Callable[..., object | None] = notice,
                    token: object | None = owned,
                ) -> None:
                    show(False, owner=token)

                if shown_for >= PASTE_READING_NOTICE_MIN_S:
                    _retire_this_card()
                else:
                    self.set_timer(PASTE_READING_NOTICE_MIN_S - shown_for, _retire_this_card)
        if contents.image is not None:
            markers = await self._attach_image_bytes([contents.image.data])
            if markers is not None:
                self.post_message(EditorPasteAttached())
                return markers
            # An image WAS on the clipboard and could not be attached: too
            # large even after bounding, an undecodable payload, a
            # decompression bomb. Reported as its own outcome rather than as
            # "no image", because the two lead to different moves — cropping
            # versus copying something (review round 1, D2/U2).
            self.post_message(EditorPasteEmpty(reason="unattachable"))
            return None
        if contents.paths:
            # Rejoins the path branch rather than duplicating it: shell quoting
            # is what `_pasted_paths` exists to undo, and these paths come from
            # an API rather than a terminal, so they are quoted here to keep one
            # parser rather than two.
            markers = await self._attach_pasted_images(
                " ".join(shlex.quote(path) for path in contents.paths)
            )
            if markers is not None:
                self.post_message(EditorPasteAttached())
                return markers
            # A COPIED FILE that would not attach, which is a different failure
            # from an oversized image: the usual causes are a non-image file,
            # an unreadable path, or a mixed multi-file selection hitting the
            # all-or-nothing rule. "Try a smaller one" answers none of them,
            # and neither does "paste its file path" \u2014 the path is exactly what
            # this branch just tried (round 2, D10).
            self.post_message(EditorPasteEmpty(reason="unreadable"))
            return None
        if allow_text and contents.text:
            # Returned as the "attached" string so the caller inserts it the
            # same way it inserts markers: one insertion point for every
            # clipboard shape, rather than a second `insert` call that could
            # drift on selection-replacement semantics.
            #
            # No `EditorPasteAttached` here. That message exists to retire a
            # notice claiming an IMAGE could not be attached, and pasting text
            # does not falsify such a claim \u2014 the image the user was told about
            # is still unattached.
            return contents.text
        if contents.refused_remote:
            reason = "remote"
        elif contents.text_too_large:
            # Checked before "nothing" for the same reason as the timeout: the
            # clipboard was NOT empty, it held text the composer declined, and
            # reporting an empty clipboard sent a user who had copied several
            # megabytes looking for a clipboard problem they did not have
            # (code round 2, F7).
            reason = "too_large"
        elif contents.timed_out:
            # Checked BEFORE "nothing": a read that never finished cannot
            # report what was on the clipboard, and saying it was empty is a
            # claim about a question that was never answered (U3).
            reason = "timeout"
        else:
            reason = "nothing"
        self.post_message(EditorPasteEmpty(reason=reason))
        return None

    async def action_system_paste(self) -> None:
        """``Ctrl+V``: paste what is on the SYSTEM clipboard.

        The reachable half of issue #372, and the reason this action exists
        rather than the composer relying on the terminal's own paste: with an
        image on the pasteboard Terminal.app swallows ``Cmd+V`` and sends the
        app NOTHING at all (measured \u2014 see this class's ``BINDINGS`` for the
        byte table). ``Ctrl+V`` is the paste keystroke every terminal delivers,
        so it is the portable one; ``Cmd+V`` is bound to this same action and
        reaches it on terminals that speak the kitty keyboard protocol, where
        it arrives as ``super+v``. **Do not "simplify" this back onto the paste
        event**; that is the shape that shipped broken.

        It REPLACES ``TextArea.action_paste``, which inserts ``App.clipboard``
        \u2014 Textual's internal buffer, written by this app's own copy gestures
        and by nothing else. On the gesture a user means by ``Ctrl+V`` (copy in
        a browser, paste here) the inherited action inserts nothing whatsoever,
        so reading the system clipboard is strictly more correct for every
        shape, text included.

        The read is threaded and bounded by :meth:`_attach_clipboard_image`,
        and so are the notices, so ``Ctrl+V`` and an empty bracketed paste
        report an unusable clipboard identically.

        ``read_only`` is honoured because the base action honours it: this is a
        keyboard edit, and a read-only composer must not become writable just
        because the key was rebound.

        **A live selection is REPLACED, not inserted into.** "Select the thing
        you want to swap out, then paste over it" is universal text-editing
        behaviour; it is what `Cmd+V` does through ``TextArea._on_paste`` and
        what stock ``TextArea.action_paste`` does. An earlier revision called
        ``insert`` here on the argument that one insertion point for every
        clipboard shape avoids drift on selection-replacement semantics — but
        ``insert`` is the branch that does NOT replace, so that reasoning
        picked the drift instead of avoiding it and shipped a paste that
        corrupts the buffer: selecting `WORD` and pasting `NEW` gave
        `keep WORDNEW keep` against `keep NEW keep` on every other route
        (ux round 1, U2 — reproduced before fixing, both the text and the
        image shape).

        ``_replace_via_keyboard`` is the exact call the base class's own paste
        makes, so this route now shares its semantics rather than approximating
        them: an empty selection degenerates to an insert at the caret, which
        is why one call serves both cases.
        """
        if self.read_only:
            return
        pasted = await self._attach_clipboard_image(allow_text=True)
        if pasted is None:
            # Nothing usable was found. `_attach_clipboard_image` has already
            # posted the notice that says so, and inserting nothing is the
            # honest outcome.
            return
        # `move_cursor` to the edit's end mirrors `TextArea._on_paste`: without
        # it `maintain_selection_offset=False` leaves the caret where the
        # replaced range began, so the next keystroke types BEFORE the text
        # just pasted.
        result = self._replace_via_keyboard(pasted, *self.selection)
        if result is not None:
            self.move_cursor(result.end_location)

    async def _attach_pasted_images(self, pasted: str) -> str | None:
        """Load every path in ``pasted`` as an attachment; return the markers.

        ``None`` means "this was not an image paste" — the caller then lets
        Textual insert the text verbatim. That is the common case and it must
        stay cheap and lossless.

        ALL-or-nothing across the paste. A multi-file drag where one file is a
        PDF becomes a plain text paste of every path, rather than silently
        attaching two of three and leaving the user to notice which. Mixed
        results are the shape a user cannot see and cannot correct.

        Async because the bytes are BOUNDED before they are attached, and
        bounding means decoding — see :func:`~local_operator.imaging.
        bound_image_for_model`. This is a keystroke handler, so a
        multi-hundred-millisecond decode inline would freeze the whole app: a
        20 MP screenshot measures ~315 ms.

        Be precise about what the thread buys, because it is not everything
        (review round 1, F3). The LOOP keeps running — the transcript still
        paints, other widgets still respond, a running turn still streams. This
        WIDGET's own input does not: the pump awaits one handler before
        dispatching the next message to the same widget, so keystrokes typed
        during the decode are queued and flush when it ends. Nothing is lost and
        the order is preserved, but the composer does go quiet for the duration,
        which it did not before this bound existed. Accepted rather than hidden
        behind a worker + late marker insertion, because a marker that appears
        after the user has typed past its position is a worse bug than a pause.
        """
        candidates = _pasted_paths(pasted)
        if not candidates:
            return None

        payloads: list[bytes] = []
        for path in candidates:
            # STAT FIRST. Two things this closes, both measured in review round
            # 17 against the previous read-then-check order:
            #
            # - The 4 MB cap ran on `len(data)`, i.e. AFTER the cost it exists
            #   to prevent. A 601 MB file behind a valid 100x100 PNG header
            #   took peak RSS to 618 MB before the cap fired, allocated
            #   synchronously on the keystroke that pasted it.
            # - `open()` on a FIFO blocks forever, and this runs inline on the
            #   event loop, so a named pipe (or a stalled network mount) is a
            #   HUNG UI rather than a failed paste.
            #
            # `sniff_image_file` cannot help with either: it reads 64 KB, and a
            # header is not a size bound.
            try:
                stat = Path(path).stat()
            except (OSError, ValueError):
                # ValueError for a NUL byte in the name, which is not an
                # OSError and would otherwise escape onto the keystroke.
                return None
            # Bounded by the INGEST ceiling, not the attachment budget. Those
            # are different bounds (see `MAX_CLIPBOARD_READ_BYTES`) and using
            # the smaller one here refused files the resize was about to make
            # attachable — the same mistake U1 found on the clipboard route,
            # left behind on this one. It produced a contradiction between two
            # of this feature's own paths: one 8.6 MB screenshot attached as
            # `[Image #1, 1568x523 ↓]` via Cmd+V and was refused via Finder
            # Cmd+C, which lands here (round 3, D12).
            #
            # The gate itself stays, and stays BEFORE the read, because that is
            # what it was for: a 601 MB file behind a valid PNG header took
            # peak RSS to 618 MB before the cap fired, and `open()` on a FIFO
            # blocks forever on the event loop. Both are still closed; only the
            # threshold moves to the bound that is actually about ingest.
            if not S_ISREG(stat.st_mode) or stat.st_size > MAX_CLIPBOARD_READ_BYTES:
                return None
            # An EARLY gate, not the authoritative one: `_attach_image_bytes`
            # sniffs again below, and that second sniff is what actually decides
            # sendability for both routes. This one exists to avoid reading a
            # 4 MB file that was never going to be attachable — a header read of
            # 64 KB is much cheaper than the whole file.
            #
            # `sendable` and not merely "recognised": HEIC sniffs fine and no
            # provider will take it, so attaching it would trade a readable
            # path in the prompt for a 400 later in the turn.
            info = sniff_image_file(path)
            if info is None or not info.sendable:
                return None
            try:
                data = Path(path).read_bytes()
            except (OSError, ValueError):
                return None
            if len(data) > MAX_CLIPBOARD_READ_BYTES:
                # The stat above is the real gate; this catches a file that grew
                # between the two calls. Same ceiling as the stat, for the same
                # reason: `_attach_image_bytes` applies the ATTACHMENT cap after
                # bounding, which is the only place it belongs.
                return None
            payloads.append(data)

        return await self._attach_image_bytes(payloads)

    async def _attach_image_bytes(self, payloads: list[bytes]) -> str | None:
        """Bound each payload, attach it, and return the markers it earned.

        The shared tail of BOTH ingestion routes — a path in the paste text and
        the system clipboard — so the two cannot drift. They must produce byte
        for byte the same marker, apply the same bound, and honour the same
        all-or-nothing rule, because from the user's side they are one gesture
        (``Cmd+V`` on a screenshot) that merely takes different roads depending
        on which terminal is running. Two copies of this tail is exactly how
        one route would quietly start attaching unbounded bytes.

        ``None`` means nothing was attached, and NOTHING has been mutated: the
        loop below completes every bound before a single marker is issued, so a
        refusal in the third image cannot leave the first two attached with
        markers the caller then discards.

        Sniffs the BYTES rather than trusting the caller. The path branch has
        already sniffed the file, and the clipboard backend already knows what
        it asked for, but this is the last gate before an ``ImageContent``
        reaches the history, and an unsendable block there is not a failed
        paste — it is a session that answers every later prompt with the same
        provider 400 (see :mod:`local_operator.imaging`).
        """
        loaded: list[tuple[ImageContent, str]] = []
        for data in payloads:
            info = sniff_image(data)
            # `sendable` and not merely "recognised": HEIC sniffs fine and no
            # provider will take it, so attaching it would trade a readable
            # path in the prompt for a 400 later in the turn.
            if info is None or not info.sendable:
                return None
            # BOUND before attaching, in a thread. The bytes are whatever the
            # screen produced, and a provider refuses an image over 2000 pixels
            # on its long edge as soon as the request carries more than twenty
            # of them (see local_operator.imaging). Forwarding verbatim was
            # therefore not "lossless", it was a delayed fault: a 2206x266
            # paste sat harmlessly in the history for a hundred turns and then
            # wedged the session permanently the moment the twenty first
            # screenshot arrived, because the block is in the HISTORY and every
            # later request — including the compaction that is supposed to be
            # the escape hatch — re-sends it and earns the same 400.
            #
            # `to_thread` and not an inline call: this runs on the keystroke
            # that pasted, and a 20 MP screenshot decodes in ~315 ms.
            try:
                payload, wire_mime, _summary = await asyncio.to_thread(
                    bound_image_for_model, data, info
                )
            except ValueError:
                # Undecodable, a decompression bomb, or too large to send with
                # no decoder available. A text paste of the path is the honest
                # outcome: the user keeps what they pasted and can see it was
                # not attached, where a silently dropped attachment is the shape
                # nobody notices until the model answers about nothing.
                return None
            if len(payload) > MAX_ATTACHMENT_BYTES:
                # AFTER the bound, which is the only place this cap belongs.
                # Applied to the SOURCE bytes it refuses images the resize was
                # about to make attachable: a Retina screenshot is 8.4-8.5 MB
                # on the pasteboard and 0.28 MB once bounded, so a pre-bound
                # gate discarded the exact gesture this feature exists for
                # (review round 1, U1). What must not reach a provider is the
                # payload actually sent, and this is it.
                return None
            loaded.append(
                (
                    ImageContent(
                        data=base64.b64encode(payload).decode("ascii"),
                        mime_type=wire_mime,
                    ),
                    # The marker reports what was ATTACHED, not what was on the
                    # clipboard or on disk. A marker reading 2560x1440 beside a
                    # 1568x882 attachment is a receipt for something that was
                    # never sent, and the whole point of the dimensions is that
                    # the user can check them at a glance.
                    _bounded_dimensions(payload, info),
                )
            )
        if not loaded:
            return None

        # From the BUFFER, immediately before issuing. Every OTHER seam derives
        # the counter, but issuance read it blind, so text carrying a marker
        # that arrived as TEXT — a prompt drag-copied out of the transcript and
        # pasted back to re-run it, which is a gesture this branch built — was
        # invisible to the counter, and the next paste re-issued a number
        # already on screen. The chip then landed on last turn's marker while
        # the real attachment rendered as prose (design round 18, D4). Deriving
        # here as well means a draft can never hold the same number twice.
        self._sync_next_marker()
        markers = []
        for image, dimensions in loaded:
            index = self._next_marker
            self._next_marker += 1
            # The dimensions are for the USER, not the model — the model gets
            # the pixels. They are what makes the marker checkable at a glance:
            # "1568x200" is recognisably the screenshot just taken, where a
            # bare "[Image #3]" could be anything. Omitted rather than faked
            # when the header did not carry them.
            marker = f"[Image #{index}, {dimensions}]" if dimensions else f"[Image #{index}]"
            # Recorded with the image, not derived later: this exact string is
            # what tells the app's own citation apart from a copy of it.
            self._attachments[index] = Attachment(image, marker)
            markers.append(marker)
        return " ".join(markers) + " "

    # -- command completion -------------------------------------------------
    def edit(self, edit: Edit) -> EditResult:
        """Every buffer mutation funnels through here — resync the picker.

        Hooking the two mutation funnels (:meth:`edit` for inserts/deletes/
        undo, :meth:`load_text` for whole-buffer replacement) keeps the picker
        exact for keystrokes, pastes, and history navigation alike, and does
        it synchronously. Listening for ``TextArea.Changed`` instead would put
        the picker one message-loop tick behind the buffer, which is the tick
        in which Enter decides whether to complete.
        """
        # Only a REMOVAL can uncite an attachment, and only the attachments it
        # touched. Both halves are measured BEFORE the edit, because afterwards
        # the range is gone and the citation positions have moved.
        touched = self._attachments_touched_by(edit)
        # Select-to-overwrite is the commonest edit in any input, and when it
        # follows a copy it falsifies the copy's receipt: the user drags a word
        # to replace it, types, and a receipt asserting a copy of characters
        # that no longer exist sits on screen for another five seconds (design
        # round 1, D3). The clipboard keeps what it took — that is what a copy
        # IS, and silently un-copying would be worse — but the CLAIM is retired
        # the moment its subject is edited away.
        stale_receipt = self._copied
        # A buffer mutation means the hand has left the mouse, so the barren
        # click is over whatever the window still says (R3-3). This is the
        # keystroke seam, and it covers paste and history recall too, because
        # they funnel through here as well.
        self.retire_barren_click()
        result = super().edit(edit)
        if stale_receipt:
            self._copied = False
            self.post_message(EditorCopyStale())
        self._sync_picker()
        if touched:
            self._release_uncited(touched)
        return result

    def _attachments_touched_by(self, edit: Edit) -> list[int]:
        """Which attachments a pending edit could plausibly uncite.

        ONE rule: the removal's range overlaps the citation's span. Empty for a
        pure insertion, because adding text cannot remove a citation and
        sweeping there would adjudicate a marker the user is typing through -
        every printable character breaks the grammar while it is being typed
        into.

        There was a second clause - release an ALREADY uncitable attachment
        when the cut text names its number, so the map could not keep an image
        no text can reach. It is gone, because a text test cannot tell the
        damaged marker the user is clearing away from ordinary prose that
        happens to say `#1`, and in a draft about images that is ordinary prose.
        Deleting `screenshot #1 from yesterday` destroyed a repairable
        attachment, irreversibly, since a retyped marker may not revive it
        (review round 26).

        The cost of dropping it is that an attachment whose marker was damaged
        and then cleared away stays in the map, uncited and unsent, until the
        draft is submitted or cleared. It is invisible and it is never sent;
        the only way to reach it is to type its number, which is `cite`'s
        documented fallback and the residual already recorded there. Holding an
        image nobody can see is a smaller wrong than destroying one the user
        is halfway through repairing.
        """
        if edit.from_location == edit.to_location or not self._attachments:
            return []
        top, bottom = sorted((edit.from_location, edit.to_location))
        cut = self.get_text_range(top, bottom)
        if not cut:
            return []
        start = self._offset_at(*top)
        end = start + len(cut)
        text = self.text
        touched: list[int] = []
        for index, attachment in self._attachments.items():
            span = cite(text, index, attachment)
            # SPAN overlap, and the span comes from `cite` rather than from
            # `len(attachment.marker)`. Two separate lessons: a containment test
            # missed a tail delete, because the head stays outside the cut
            # (design round 24, D16); and measuring with the RECORDED marker's
            # length misjudged a citation the user had lengthened, so a cut past
            # the old end escaped the test and reopened D16 (round 26).
            if span is not None and start < span[1] and span[0] < end:
                touched.append(index)
        return touched

    def load_text(self, text: str) -> None:
        """Whole-buffer replacement — the OTHER mutation funnel.

        ``edit()`` does not run for this path (Textual's ``text`` setter calls
        ``load_text`` directly), so the copy receipt has to be stood down here
        too. Without it the flag survived every submit, ``/clear``, history
        step and ``begin_model_query``, and the next keystroke — whenever it
        came — withdrew whatever receipt happened to be on screen by then
        (review round 2, F5).

        No ``EditorCopyStale`` is posted: replacing the buffer is not the user
        editing the text their receipt describes, and the card, if it is still
        up, is about a copy that remains perfectly true.
        """
        self._copied = False
        # The FIFTH retirement seam, for the same reason the receipt is stood
        # down here: this funnel bypasses ``edit()`` entirely, so the keystroke
        # seam never runs for it. A barren claim raised just before an Up-arrow
        # history recall, a ``/clear``, or a compaction/aside/steer hand-back
        # therefore survived a whole-buffer swap and absorbed the next Ctrl+C —
        # aimed at text the click had never seen (agent review round 4, R4-1).
        # One no-op press rather than a lost draft, but the claim is wrong the
        # instant the buffer it described is gone, so it ends here too.
        self.retire_barren_click()
        super().load_text(text)
        # Suppressed by :meth:`_set_text_and_caret`, which moves the caret AFTER
        # this returns and then syncs ONCE at the final position. Without the
        # guard this sync runs with the caret still at the origin: for a
        # completion that leaves an argument (``/team security ``) it sees no
        # active command there, NULLS ``_argument_command``, and the later
        # caret-anchored sync then re-posts ``ArgumentQueryOpened`` — whose app
        # handler calls ``set_notice("")`` and erases the U1/U2 parked-caret hint
        # the editor set for /team·/agent (design review round 3, D5). One sync at
        # the right place is both correct and cheaper.
        if not self._suspend_picker_sync:
            self._sync_picker()

    def _caret_offset(self) -> int:
        """The caret as a whole-buffer offset, for the slash parsers.

        Inline detection keys on WHERE the caret is: which slash token the user
        is editing depends on it, so every parse the picker runs is anchored
        here rather than at the end of the buffer. Reuses the same
        ``_offset_at`` the attachment chips measure with, so a CRLF buffer a
        paste carried in is counted the same way in both places.
        """
        row, column = self.selection.end
        return self._offset_at(row, column)

    def _set_text_and_caret(self, text: str, caret_offset: int) -> None:
        """Replace the buffer and park the caret, then re-derive the pickers.

        The ``text`` setter re-syncs the picker BEFORE the caret has moved (it
        runs inside ``load_text``, and ``move_cursor`` comes after), so with
        caret-anchored detection that sync reads the OLD caret position — for a
        completion, the origin, which names the wrong token or none. Every
        completion therefore has to re-sync once the caret is where the
        completion put it; funnelling that through one helper is what keeps the
        rule from being forgotten at one of the several completion sites.

        The setter's own sync is SUPPRESSED (not merely repeated): a stray
        origin-anchored sync does not just waste work, it nulls
        ``_argument_command`` and makes the final sync look like a fresh argument
        opening, which re-fires ``ArgumentQueryOpened`` and wipes an editor-set
        notice (D5). So exactly one sync runs, at the final caret.
        """
        self._suspend_picker_sync = True
        try:
            self.text = text
            self.move_cursor(self._location_at_offset(caret_offset))
        finally:
            self._suspend_picker_sync = False
        self._sync_picker()

    def _location_at_offset(self, offset: int) -> tuple[int, int]:
        """A whole-buffer ``offset`` as a ``(row, column)`` document location.

        The inverse of :meth:`_offset_at`, so a completion that computed WHERE to
        put the caret as an offset (the splice point of an inline command) can
        hand it back to ``move_cursor``. Walks lines counting the document's own
        separator, matching ``_offset_at`` exactly, so the two agree on a CRLF
        buffer a paste carried in. Clamped so a stale offset lands at the buffer
        end rather than raising.
        """
        separator = len(self.document.newline)
        remaining = max(0, offset)
        for row in range(self.document.line_count):
            length = len(self.document.get_line(row))
            if remaining <= length:
                return row, remaining
            remaining -= length + separator
        return self._end_of_buffer()

    def _splice_command(self, token_start: int, token_end: int) -> None:
        """Remove the ``[token_start, token_end)`` command token from the draft.

        The heart of "a command run mid-text is removed from the text where it
        was entered": the user typed a message and then a ``/command`` to route
        it, so once the command runs the token has done its job and must not be
        left sitting in the prose that is about to be sent. One adjoining
        whitespace character is removed with it — the space or newline the user
        typed to set the command apart from the message — so ``fix this /team``
        becomes ``fix this`` rather than ``fix this ``, ``/team fix this``
        becomes ``fix this`` rather than `` fix this``, and a command on its OWN
        line above a draft collapses that line away instead of leaving a blank
        one. The PRECEDING separator is preferred (that is the one the inline
        gesture adds — a space before an appended command, a newline above a
        message); the following one is taken only when the token opened the
        buffer.
        """
        text = self.text
        start, end = token_start, token_end
        if start > 0 and text[start - 1] in " \t\n":
            start -= 1
        elif end < len(text) and text[end] in " \t\n":
            end += 1
        self.text = f"{text[:start]}{text[end:]}"
        self.move_cursor(self._location_at_offset(start))

    def _sync_picker(self) -> None:
        """Re-derive EVERY list from the buffer.

        The buffer is the single authority, so no picker holds state another could
        contradict: `slash_context` is live while the command word is open and
        `slash_argument` takes over on the terminating space, which makes "exactly
        one list is showing" a property of the parse rather than a rule the widgets
        have to cooperate on. The command picker serves both the word and a
        provider argument, so the branch below is which LIST it derives, not which
        widget is visible.

        Every parse is anchored at the CARET (:meth:`_caret_offset`): inline
        detection means "which slash token is active" is a question about where
        the caret sits, not just what the buffer contains.
        """
        cursor = self._caret_offset()
        list_argument = slash_argument(
            self.text, self._argument_commands, cursor, self._command_names
        )
        if list_argument is None:
            self._argument_command = None
            self._argument_subcommand = None
            # No argument list is open, so the name snapshot is ordinarily stale:
            # drop it so a highlight can never outlive the list that filled it
            # (e.g. the command word was deleted back to `/tea`).
            #
            # EXCEPT for a multi-line LEADING NAME+message command WHOSE FAMILY
            # MATCHES the snapshot. `slash_argument` is single-line and (post
            # #250) caret-anchored, so it returns None the instant the user adds
            # a newline to `/team <name> <message>` — but that command's body is
            # DEFINED to span lines and the leading command still dispatches, so
            # its command and name tokens must stay highlighted. Clearing the
            # snapshot here would blank the name token on the first newline (the
            # reported bug), so keep it while the buffer is a live multi-line name
            # command of the SAME family the snapshot was filled for.
            #
            # The family gate is what makes the within-family "cannot mispaint"
            # guarantee hold across a family switch: an atomic word-swap
            # `/team <team-name>\n…` → `/agent <team-name>\n…` while already
            # multiline never re-opens a list (multiline suppresses it), so
            # without this the team roster would survive and paint a team name
            # green under `/agent`. Dropping the snapshot on a family mismatch
            # falls the name token back to prose, which is correct: the name is
            # not a valid member of the new family, and no list is open to
            # re-derive the right roster until the buffer returns to a single
            # line. Uses the LEADING word, not the caret-anchored one, because the
            # caret is normally down in the message body. The leading family is
            # named out so the guard below reads as its two conditions: "still a
            # multi-line name command" AND "same roster family".
            leading_family = self._name_command_family(self._leading_command_word())
            keep = self._is_multiline_name_command() and (
                self._name_choices_family == leading_family
            )
            if not keep:
                self._name_choices = frozenset()
                self._name_choices_family = None
            self._picker.sync(self.text, cursor)
        else:
            command = self._command_word()
            if command != self._argument_command:
                self._argument_command = command
                self._argument_subcommand = None
                # Drop the previous command's rows before asking for this one's.
                # `/login` offers every provider and `/logout` only the ones with a
                # credential, so carrying them across would briefly offer a logout
                # from an account the user never had.
                self._picker.set_choices([])
                # Clear the name snapshot too: the app re-pushes it for a
                # NAME+message command in ``on_argument_query_opened``; leaving
                # the previous command's names would highlight `/agent frontend`
                # against team names.
                self._name_choices = frozenset()
                self._name_choices_family = None
                self.post_message(ArgumentQueryOpened(command or ""))
            elif command in ("mcp", "team", "teams"):
                # `/mcp` and `/team` are two-level: `/mcp` reserves verbs in the
                # first argument slot and offers servers in the second; `/team`
                # reserves the `chart` subcommand in the first slot and, once
                # `chart ` is present, re-offers TEAM NAMES in the second (the
                # [name] the chart wants). ArgumentQueryOpened fires on the
                # command WORD, so without this the first-slot rows would stay
                # up after the subcommand was completed and the second slot
                # would have nothing to offer. Tracked rather than refreshed per
                # keystroke: the row set only changes when the sub-slot does.
                first_tok, sep, _tail = (list_argument or "").partition(" ")
                if command == "mcp":
                    # `/mcp` is per-verb: every verb has a distinct server list,
                    # so the tracking key is the verb token and any change posts
                    # a refresh (verbs while it is bare, servers once a verb is
                    # chosen — the builder reads the space itself).
                    #
                    # The SEPARATOR is part of the key, not just the token. The
                    # builder flips from the verb rows to that verb's server
                    # rows precisely on the terminating space, so a token-only
                    # key made `login` and `login ` the same state and posted no
                    # refresh across the one transition that changes the answer:
                    # the server rows were unreachable by typing, and the picker
                    # sat empty and closed. This is the identical trap the
                    # `/team` branch below documents for `chart` → `chart `,
                    # which is why that branch tracks a boundary rather than a
                    # token (#377 review round 2). Keyed on the whole
                    # `verb + sep` so both edges cross: entering the server slot
                    # and backspacing out of it.
                    subcommand = f"{first_tok.lower()}{sep}" if list_argument else ""
                    if subcommand != self._argument_subcommand:
                        self._argument_subcommand = subcommand
                        self.post_message(RefreshArgumentChoices(command))
                else:
                    # `/team` has exactly ONE thing that changes its choice set:
                    # crossing into or out of the `chart ` SECOND slot (first
                    # token is the reserved `chart` word AND a space follows —
                    # first-slot team names ⇄ second-slot chart targets). Track
                    # that boolean, not the raw first token: a team-NAME first
                    # token (`/team security `) is the talk path whose choice set
                    # never changes, so it must NOT post a refresh. The refresh
                    # handler clears the notice, which would wipe the switch/send
                    # parked-caret hint the block below sets — the exact D5
                    # regression #250 guards
                    # (`test_completing_a_team_name_keeps_the_parked_hint`). And
                    # tracking the boolean (not the token) is what makes the
                    # second slot refill at all: `chart`→`chart ` leaves the
                    # token unchanged but flips the boundary, so a token-keyed
                    # tracker would never fire on the space that opens slot two.
                    chart_second_slot = "chart" if (first_tok.lower() == "chart" and sep) else ""
                    # Normalize the prior state: both ``None`` (never tracked)
                    # and ``""`` (first slot) mean "not in the chart second
                    # slot", so entering ``/team `` for the first time is NOT a
                    # boundary crossing and must not post a refresh — otherwise
                    # the very first sync would clear the switch/send notice
                    # before it is set (D5). Only a real cross into or out of
                    # ``chart `` fires.
                    was_chart_slot = self._argument_subcommand == "chart"
                    is_chart_slot = chart_second_slot == "chart"
                    self._argument_subcommand = chart_second_slot
                    if was_chart_slot != is_chart_slot:
                        self.post_message(RefreshArgumentChoices(command))
            self._picker.sync_argument(list_argument)
            # U1/U2 discoverability hint. The moment a NAME+message name is
            # autofilled (or hand-typed) to `/<cmd> <name> ` with an empty tail,
            # the picker closes and NOTHING on screen says the two outcomes now
            # available: a blank Enter SWITCHES (attach-only), a typed message
            # SENDS. That divergence from the enum-tail commands (where Enter on a
            # row runs immediately) is invisible, so a first-timer reads the park
            # as a dropped keystroke. We reuse the picker's own notice row — the
            # codebase's bounded, self-clearing "say it where the list was" idiom
            # (see CommandPicker.set_notice) — rather than inventing a widget. It
            # shows only in the name-complete-empty-tail state and is withdrawn
            # the instant a message character is typed (the tail stops being
            # empty) or the name is edited (the space is gone), so it never sits
            # over a live message. Managed here because for these commands the
            # notice channel is otherwise unused (the app sets it to "").
            #
            # GATED to NAME+message commands only (CR4, round 2). `set_notice`
            # runs on EVERY keystroke of the argument, but the app owns the
            # notice channel for the OTHER argument commands — `/logout` ("no
            # stored credentials…"), `/effort` ("this model takes no effort
            # setting"), `/mcp`'s builder notices — and sets them ONCE on the
            # command-word change (`on_argument_query_opened`), never per
            # keystroke. An unconditional write here called `set_notice("")` on
            # each resync of those commands and erased a notice that is, for an
            # empty-by-construction list, the entire content the user is reading.
            # Only `/team`·`/agent` reach this write, and for those the app
            # always sets the notice to "", so the hint owns the channel cleanly.
            if self._is_name_argument_command(self._argument_command):
                self._picker.set_notice(self._name_switch_hint(list_argument) or "")
        # The ghost is re-derived from the freshly synced picker, on the one
        # path every keystroke already takes. Placed after both list branches so
        # it reads the picker state this sync just settled, never the previous
        # keystroke's.
        self._sync_ghost()
        argument = slash_argument(self.text, self.MODEL_COMMANDS, cursor, self._command_names)
        if argument is None:
            if self._model_picker.is_open():
                self._model_picker.close()
            return
        if self._model_picker.is_open():
            self._model_picker.set_query(argument)
        else:
            self._model_picker.open(argument)
            # Transition only. Posting per keystroke would re-fetch every provider
            # for each character the user types into the query.
            self.post_message(ModelQueryOpened())

    def _on_picker_highlight(self, name: str | None) -> None:
        """Relay the picker's highlight to the app (see ArgumentHighlightChanged).

        The picker reports a display NAME with no idea which command's list it
        is; the buffer knows, and this widget owns the buffer, so the pairing
        happens here. A ``None`` command (the word was deleted under the open
        list) is reported as a close — the preview must not outlive its list.
        """
        command = self._argument_command
        self.post_message(ArgumentHighlightChanged(command or "", name if command else None))

    def _on_picker_preview(self, name: str | None) -> None:
        """Re-derive the ghost whenever the ACCEPT TARGET moves.

        THE one place the ghost answers to the picker, and the reason it is a
        separate channel from :meth:`_on_picker_highlight`: the ghost is a
        prediction about Tab, not a report of what the eye is on. The picker
        fires this from every site that can move the selection or take the rows
        away — arrows, wheel, page/home/end, a refilled list, dismissal, close
        — so "the ghost is re-derived only on keystrokes" stops being true.
        Four separate defects (a stale row after an arrow, a hover-driven
        preview, a ghost outliving Esc, and a preview that never came back)
        were all that one gap (review round 1, U1-U3).

        ``name`` is passed through because the picker reports it before
        ``highlighted_name()`` would return it, and ``None`` means the list is
        no longer offering anything — which clears the ghost.
        """
        if name is None:
            self.suggestion = ""
            return
        self._sync_ghost(name)

    def _command_word(self) -> str | None:
        """The lower-cased command word of the slash token AT THE CARET.

        Caret-anchored to match every other parse: with inline detection the
        active command is wherever the caret is, so reading the first non-blank
        line would name the wrong token for ``fix this /model`` or a command
        dropped on a later line. Both the word phase and the argument phase
        resolve through :func:`slash_argument`'s companion by sharing the same
        line/slash split.
        """
        return slash_word(self.text, self._caret_offset(), self._command_names)

    def _is_multiline_name_command(self) -> bool:
        """Whether the buffer is a LEADING NAME+message command with a body.

        The one state where the name snapshot must survive `_sync_picker`'s
        "no argument list is open" branch: `/team <name>` (or `/agent …`) on the
        first content line, followed by a newline and a message. `slash_argument`
        reports no list on multiline (and #250 also anchors it at the caret), but
        the leading command is still live and its name token must stay painted,
        so recognition of the typed name has to keep the snapshot the list left
        behind. Reads the LEADING word (:meth:`_leading_command_word`), not the
        caret-anchored one, because the caret is usually down in the message
        body. Ordinary commands never reach this — only the NAME+message ones
        carry a multi-line body by design.
        """
        lines = self.text.split("\n")
        first = next((i for i, line in enumerate(lines) if line.strip()), None)
        if first is None or len(lines) <= first + 1:
            return False
        return self._is_name_argument_command(self._leading_command_word())

    def _complete_model(self, row: ModelRow) -> None:
        """Put ``row``'s selector in the ``/model`` argument without acting on it.

        No trailing space, unlike a command completion: the selector IS the whole
        argument, and a trailing space would terminate it and close the list — so
        Tab would appear to both fill the field and give up on it.

        Splices only the argument span so an inline ``/model`` keeps any trailing
        message; falls back to rewriting the whole buffer when the parse cannot
        locate the argument (the caret raced a delete), which is the pre-inline
        behaviour and still correct for the common start-of-buffer case.
        """
        context = slash_argument_context(
            self.text, self.MODEL_COMMANDS, self._caret_offset(), self._command_names
        )
        if context is None:
            self.text = f"/model {row.selector}"
            self.move_cursor(self._end_of_buffer())
            return
        text = self.text
        self._set_text_and_caret(
            f"{text[: context.start]}{row.selector}{text[context.end :]}",
            context.start + len(row.selector),
        )

    def _apply_model(self, row: ModelRow) -> None:
        """Hand a chosen row to the app and clear (or splice) the buffer.

        The buffer is emptied HERE rather than by the handler because the command
        never reaches the submit path: choosing from the list is the whole
        interaction, so leaving `/model anthropic/claude-opus-5` behind would
        invite a second Enter that ran the switch again.

        Inline, only the ``/model <selector>`` token is spliced out and the rest
        of the draft is kept — the same rule every other inline command follows,
        so choosing a model to switch to mid-draft does not throw the message
        away. Whole-buffer (the ordinary case) still clears completely.
        """
        self._model_picker.close()
        handler = self._on_model_chosen
        context = slash_argument_context(
            self.text, self.MODEL_COMMANDS, self._caret_offset(), self._command_names
        )
        surviving = (
            (self.text[: context.token_start] + self.text[context.end :]).strip()
            if context is not None
            else ""
        )
        if context is not None and surviving:
            self._splice_command(context.token_start, context.end)
        else:
            self.clear_content()
        if handler is not None:
            handler(row)

    def _picker_choice_is_unambiguous(self, name: str) -> bool:
        """Whether Enter may SEND the highlighted command, not just insert it.

        Unambiguous means one of three things:

        * the user ARROWED onto this row. The gate exists because the matcher may
          have picked the row on the user's behalf, and an explicit move is the
          direct answer to that — the user read the list and chose. It is also the
          muscle memory every comparable picker (fzf, an editor command palette)
          has already taught: move, Enter, done;
        * the user typed the name in full, so they named it rather than letting
          the matcher choose; or
        * there is exactly one match, so the highlighted row is the only command
          the query could mean — EXCEPT on a destructive list, below.

        Everything else completes and waits for a second Enter. The point is the
        registry's uneven blast radius: `/usage` is harmless and `/loop` and
        `/logout` are not, so a fuzzy pick must not be *run* on one keystroke.

        "Exactly one match" is not evidence on a DESTRUCTIVE list, and that is the
        one place the distinction pays. The matcher is a subsequence matcher, so a
        query that spells nothing can still leave a single survivor: `/logout oer`
        reached openrouter, `/logout dpsk` deepseek and `/logout xoh` xai-oauth —
        each one Enter away from deleting a credential the user never named. A
        typo one letter off a real id lands in exactly this shape. So `/logout`
        asks for the id in full or an explicit move, and nothing else counts.
        """
        query = self._picker_query()
        if query is None:
            return False
        if self._picker.chosen_by_hand:
            return True
        if query.strip().lower() == name.strip().lower():
            return True
        return not self._argument_is_destructive() and len(self._picker.suggestions()) <= 1

    def _argument_is_destructive(self) -> bool:
        """Whether the OPEN argument list removes something when a row is chosen.

        TWO conditions, OR-ed — the command WORD, or the highlighted ROW's own
        ``alert`` flag. Both, because they protect different things and neither
        subsumes the other:

        * ``DESTRUCTIVE_COMMANDS`` is the FLOOR. It is a property of the command
          and holds however the app filled the rows in, so a list that arrives
          empty, late, or without flags is still gated. It must not be replaced
          by the row check: that would make credential safety depend on data the
          app happened to set, and any `/logout` row that ever shipped without
          ``alert=True`` would silently lose the protection it has today.

        * The ROW flag is the PRECISION. A command word is the wrong
          granularity for a two-level command: under `/mcp`, `remove` and
          `logout` destroy outright and `reauth` forgets the stored credential
          before re-authorizing (`_cmd_mcp` runs `_mcp_logout` first), while
          `login` only ever adds a grant. Adding `mcp` to the tuple would tax
          that one harmless sibling — `/mcp login lin` + Enter would fill
          instead of running. ``ArgumentChoice.alert`` is set per row, so the
          gate follows what a row actually does rather than what its command
          word is called, using machinery that already exists rather than a
          second confirmation mechanism.

        Without the row condition, `/mcp remove fsy` — three characters that
        spell nothing, narrowed by the fuzzy matcher to one survivor — deleted a
        server config from disk on a single Enter, because the command word is
        `mcp` and not `logout`. That is the `/logout oer` → openrouter hazard
        this gate was written for, reached through a word the tuple could not
        see. The same hole existed for `/mcp logout`.

        Note ``alert`` is also set on rows that are merely CONSEQUENTIAL rather
        than deleting (`/approvals default auto`, whose effect outlives the
        window). Gating those the same way is the safe direction: the only cost
        is that Enter fills and a second Enter runs.
        """
        if self._picker.mode is not PickerMode.ARGUMENT:
            return False
        if self._argument_command in self.DESTRUCTIVE_COMMANDS:
            return True
        choice = self._picker.highlighted_choice()
        return choice is not None and choice.alert

    def _picker_query(self) -> str | None:
        """The text the open list is matching against, or ``None`` when closed.

        One gate, two lists: the command word and a provider argument are parsed
        by different functions but judged by the same rule, so the destructive
        case each protects (`/lo` reaching `loop`, `/logout an` reaching a stored
        credential) cannot drift apart.
        """
        cursor = self._caret_offset()
        if self._picker.mode is PickerMode.ARGUMENT:
            return slash_argument(self.text, self._argument_commands, cursor, self._command_names)
        context = slash_context(self.text, cursor, self._command_names)
        return None if context is None else context.query

    def _completion_for(self, mode: CompletionMode, name: str) -> tuple[str, int] | None:
        """``(new_text, new_caret)`` for accepting ``name``, or ``None``.

        The widget-side adapter over the pure :func:`completion_for`: it supplies
        this editor's vocabulary and live caret so both the completion sites and
        the ghost ask the question the same way. Everything else about a
        completion (running the command, the inline reassembly, the
        trailing-space policy) stays where it was.
        """
        return completion_for(
            self.text,
            self._caret_offset(),
            mode,
            name,
            self._argument_commands,
            self._command_names,
        )

    def _completion_mode(self) -> CompletionMode | None:
        """Which slot the OPEN picker would complete into, or ``None``.

        Reads the picker's own mode rather than re-parsing, so the ghost can
        never describe a different slot from the one Tab would act on.
        """
        if not self._picker.is_open():
            return None
        if self._picker.mode is not PickerMode.ARGUMENT:
            return CompletionMode.COMMAND
        if self._is_name_argument_command(self._argument_command):
            return CompletionMode.NAME_ARGUMENT
        return CompletionMode.ARGUMENT

    def _ghost_completion(self, name: str | None = None) -> str:
        """The dimmed inline preview of what Tab would insert, or ``""``.

        Ghost text is rendered by Textual's own ``suggestion`` reactive (the
        ``text-area--suggestion`` component, inserted at the caret in
        ``TextArea._render_line``). Nothing here paints: a bespoke render pass
        would duplicate the framework AND could not be ordered correctly against
        :meth:`_paint_markers`/:meth:`_paint_slash`, which post-process a
        finished ``Strip``.

        Using the native path means living with two of its properties, and
        gates 1-3 below are what put both out of reach. They are not caution;
        removing any one of them reintroduces a specific, reproduced defect.
        Gate 4 is a later addition of a different KIND — not a wrap hazard that
        would mispaint, but an honest preview that renders to nothing — so its
        full reasoning lives at the code site rather than here, where it would
        outweigh the three gates that prevent actual corruption.

        1. **Caret at the END of the command's line, with no selection.** The
           ghost adds cells the DOCUMENT does not have, while
           :meth:`_slash_cells` and :meth:`_marker_cells` compute their x-ranges
           from document columns. Any highlighted run at or after the caret
           therefore slides against the rendered strip — observed as the ghost
           painted in the command colour with the real text's highlight stripped
           off. With the caret at the line end there is no such run left to
           misalign. **Relaxing this gate silently reintroduces that mispaint**:
           driving the widget with the gate bypassed renders ``/mc`` + ghost
           ``p `` as ``/p mc``, and no test of the ghost's TEXT would catch it.
        2. **The ghost fits the remaining width.** Textual injects the
           suggestion AFTER the wrap sections are divided, so a long ghost
           neither wraps nor crops and simply overruns the composer.
        3. **Single-line buffer.** Same wrap machinery, and a multi-line draft is
           not a state the command lists are live in anyway.
        4. **The ghost has a visible character.** A completion that appends only
           the command's trailing space paints nothing at all, so it is withheld
           rather than rendered as an empty preview. See the comment at the gate
           itself for why this is a rendering judgement and not a change to what
           Tab commits.

        Beyond the gates, the ghost is shown only when accepting the row is a
        pure APPEND to the buffer (see :func:`ghost_for`) — a fuzzy match
        rewrites typed characters, so no ghost can honestly describe it.
        """
        mode = self._completion_mode()
        if mode is None:
            return ""
        row = name if name is not None else self._picker.highlighted_name()
        if row is None:
            return ""
        # Nothing typed to complete FROM, and no row deliberately chosen. A
        # bare `/` offers the whole registry in registration order, so ghosting
        # its first row is not extending a query, it is naming the top of an
        # unfiltered list — the feature's first frame spent on a coin flip
        # (review round 1, U7). The picker already models "too little typed to
        # guess" for its fuzzy tail (`FUZZY_MIN_QUERY_CHARS`); this is the same
        # judgement one keystroke earlier.
        #
        # `chosen_by_hand` is the exemption, and it is the same signal the
        # ambiguity gate trusts to let Enter send: once the user has ARROWED to
        # a row they have named it themselves, so previewing it is reporting
        # their choice rather than guessing at one. An ARGUMENT slot is exempt
        # outright — its empty-query list enumerates that command's own values
        # (`/mcp ` → the verbs), which IS the answer to what was typed.
        if (
            mode is CompletionMode.COMMAND
            and not self._picker_query()
            and not self._picker.chosen_by_hand
        ):
            return ""
        text = self.text
        # Gate 3, and gate 1's selection half. A selection means Tab's edit is
        # not an insertion at a caret at all.
        if "\n" in text or self.selection.start != self.selection.end:
            return ""
        caret = self._caret_offset()
        if caret != len(text):
            return ""
        ghost = ghost_for(self._completion_for(mode, row), text)
        if not ghost:
            return ""
        # Gate 4: a ghost carrying no VISIBLE information is not shown.
        #
        # Every command's completion ends in a trailing space, so a fully typed
        # command (`/resume`, `/mcp`) has the single-character ghost `' '` —
        # the state every user passes through on the way to Enter. It draws
        # zero dim pixels by construction, so with the caret back at the true
        # insertion point (see :meth:`render_line`) the frame is a block on a
        # blank cell one past the text: pixel-identical to the composer with no
        # ghost at all. Painting it therefore communicates nothing, while the
        # attempt to make it visible is what motivated #378's caret
        # displacement and the overwrite misreading that followed.
        #
        # THIS DOES NOT CHANGE WHAT TAB DOES. The invariant is
        # ``buffer + ghost == buffer after Tab``, stated over a ghost that IS
        # shown; withholding a preview is not an edit to the completion, and
        # Tab still inserts the trailing space here exactly as before. The
        # honesty rule (a ghost must be a pure append — :func:`ghost_for`)
        # stays where it is, in the completion module, which answers "can this
        # completion be described as an append?" from the completion alone.
        # Whether an honest append is worth PAINTING depends on the rendered
        # frame, so it belongs here with gates 1-3 (caret position, width,
        # single line), all of which are likewise about what the composer can
        # legibly show rather than about what Tab commits.
        if not ghost.strip():
            return ""
        # Gate 2. The row's text cells are ``content_size.width`` less the
        # gutter, and the caret sits at ``column`` within them (single-line by
        # gate 3, so the document column IS the screen column). A ghost that
        # would not fit is dropped ENTIRELY rather than rendered overrunning:
        # cropping it would show fewer characters than Tab inserts, which breaks
        # the same invariant from the other direction.
        width = self.content_size.width - self.gutter_width
        column = self.selection.end[1]
        # ``>=``, not ``>``. Textual reserves the cell AT the caret for the
        # caret itself, so a ghost ending exactly at the content edge still
        # pushes the rendered strip one cell past the box — measured at w=19
        # (`/analytic` + `s `) and w=13 (`/mc` + `p `), where the strip came
        # back one wider than the same row with no ghost. The boundary case is
        # the only one that matters here, and it was the one the original
        # comparison admitted (review round 1, B2).
        if width <= 0 or column + len(ghost) >= width:
            return ""
        return ghost

    def _sync_ghost(self, name: str | None = None) -> None:
        """Push the current ghost into Textual's ``suggestion`` reactive.

        The whole write, and the ONE place it happens. Its inputs are the
        buffer, the picker's accept target, and the available width, so it is
        driven from exactly the three things that can change those:
        :meth:`_sync_picker` (a keystroke), :meth:`_on_picker_preview` (the
        selection moved or the rows went away), and :meth:`_on_resize` (the
        width gate's answer changed). Anything re-deriving the ghost from a
        fourth place is a path waiting to go stale.
        """
        self.suggestion = self._ghost_completion(name)

    def _on_resize(self) -> None:
        """Re-check the width gate when the composer's width changes.

        Gate 2's answer is a function of the terminal width, and nothing else
        re-asked it: a ghost admitted at 100 columns stayed painted when the
        terminal was narrowed to 13, where it overran and cropped the user's
        own text (``/usage`` rendering as ``/usag``) — the exact failure the
        gate exists to prevent. The inverse was equally wrong: a ghost
        correctly withheld at 18 columns did not come back on widening until
        the user typed another character (review round 1, U4).

        Cheap and idempotent, so it needs no guard. ``super()`` first because
        the base class re-wraps here, and the gate measures against the wrapped
        geometry.
        """
        super()._on_resize()
        self._sync_ghost()

    def action_cursor_right(self, select: bool = False) -> None:
        """Move the caret one cell right. NEVER accept the ghost.

        ``TextArea.action_cursor_right`` inserts ``suggestion`` when one is set,
        which is the fish/zsh-autosuggest convention. This widget deliberately
        does not take it: Tab is the single accept key the ghost's invariant
        (``buffer + ghost == buffer after Tab``) is stated over, and issue #370
        wants ``alt+←/→`` and ``cmd+←/→`` as caret motion in this same composer.
        A ``→`` that sometimes types five characters and sometimes moves one cell
        would make that family of chords mean two different things depending on
        whether a list happens to be open.

        Clearing the suggestion around the super() call is what skips the insert
        without reimplementing the caret arithmetic, which is non-trivial
        (selection collapse, soft-wrap, end-of-document). The ghost is then
        re-derived rather than restored: the caret has moved, so whether it is
        still honest is exactly the question :meth:`_ghost_completion` answers.
        """
        self.suggestion = ""
        try:
            super().action_cursor_right(select)
        finally:
            self._sync_ghost()

    def _apply_command(self, name: str) -> None:
        """What CHOOSING a row does — the picker's ``on_choose`` callback.

        A command word is completed into the buffer and left there. A NON-
        destructive argument row is RUN, matching the model picker: a click names
        one exact row with a pointer, which is not the guess the keyboard
        ambiguity gate protects against, and a click that only filled the field
        would leave the user reaching for Enter to finish a choice they already
        made.

        A `/logout` row is filled in instead. The picker sits directly above the
        input row — the row a user clicks to place the caret — so "one click, one
        credential gone" is a misclick away, and an OAuth credential is not
        recoverable without another browser login. Confirming with Enter is the
        same two-step the keyboard gate already requires of the same command, so
        the mouse is not learning a separate rule.

        For a command the trailing space is load-bearing, not cosmetic: most
        commands take an argument (``/model provider/id``), and it is also what
        closes the picker — the word is now whitespace-terminated, so the list
        drops away on the same keystroke that chose from it.
        """
        if self._picker.mode is PickerMode.ARGUMENT:
            # A clicked team/agent row fills the name and a space and waits for
            # the message, exactly like Tab/Enter on the same row — a click on a
            # NAME+message row is "chosen, now type the message", never "switch
            # and discard whatever you were about to say".
            if self._is_name_argument_command(self._argument_command):
                self._complete_name_argument(name)
                return
            if self._argument_is_destructive():
                self._complete_argument(name)
                return
            self._run_argument(name)
            return
        # The string arithmetic lives in ``completion_for`` so the inline ghost
        # is derived from the SAME function this commits: the dimmed cells the
        # user sees are exactly the characters Tab writes, by construction
        # rather than by two builders agreeing (see :meth:`_ghost_completion`).
        # It replaces ONLY the command token ``[start, end)``, because inline
        # detection means a message may follow it and that prose must survive.
        completed = self._completion_for(CompletionMode.COMMAND, name)
        if completed is None:
            return
        self._set_text_and_caret(*completed)

    def _resolve_argument(self, name: str, key: str, unambiguous: bool) -> None:
        """Tab/Enter on an argument row: complete it, or run it.

        Enter runs only what the same gate calls unambiguous, for the reason the
        gate exists at all: `/logout` DELETES a credential, so acting on a row the
        fuzzy matcher guessed would make one mis-keystroke destructive. An
        ambiguous Enter completes instead — the buffer then holds the exact id,
        which is one match, so the second Enter runs it.

        A NAME+message command (`/team`, `/agent`) short-circuits ALL of that:
        Tab and Enter, ambiguous or not, fill the name and a space and never
        submit — "one match" is not "run" for these, it is "ready for the
        message". See :meth:`_complete_name_argument`.
        """
        if self._is_name_argument_command(self._argument_command):
            self._complete_name_argument(name)
            return
        if key == "tab" or not unambiguous:
            self._complete_argument(name)
            return
        self._run_argument(name)

    def _complete_argument(self, name: str) -> None:
        """Put ``name`` in the argument slot without acting on it.

        No trailing space, for the same reason as :meth:`_complete_model`: the
        space terminates the argument, so the matcher would stop matching and Tab
        would appear to fill the field and abandon it in one keystroke.
        """
        # Shared with the ghost through ``completion_for`` (see
        # :meth:`_apply_command`). It replaces the argument SPAN ``[start,
        # end)`` rather than the buffer tail: inline detection means the command
        # may not be the last thing in the draft, so trimming the argument's
        # length off the end would corrupt a trailing message.
        completed = self._completion_for(CompletionMode.ARGUMENT, name)
        if completed is None:
            return
        self._set_text_and_caret(*completed)

    def _run_argument(self, name: str) -> None:
        """Complete ``name`` and run, so the command's own handler runs it.

        Dispatching the finished text rather than calling a callback keeps ONE
        implementation of what `/login anthropic` means: the app's slash dispatch.
        A second path would be a second place for the login flow to drift.
        """
        self._complete_argument(name)
        self._run_command_from_buffer()

    def _run_command_from_buffer(self) -> None:
        """Run the slash command the caret is on — inline-splice or whole-buffer.

        The ONE run path both the word phase and the argument phase funnel into,
        so "how a command runs" is decided in one place regardless of which list
        chose it. Two shapes, told apart by whether ANY of the draft survives
        once the command token is removed:

        * whole-buffer — the command is all the user typed (``/usage``,
          ``/logout anthropic``). Goes through :meth:`_submit`, which clears the
          buffer and records history, exactly as a start-of-line command always
          has. The app dispatches the slash-shaped ``EditorSubmitted.text``.

        * inline — the user typed a message and then a command to route it
          (``fix this /team ops``). The token is spliced OUT (see
          :meth:`_splice_command`), the surviving message is LEFT in the
          composer, and the command is dispatched through
          :class:`InlineCommandRequested`. No clear, no history entry: the draft
          is still unsent, and the command was not a prompt to page back to.

        The command's argument, by the inline contract, is everything from the
        word to the END OF ITS LINE (see :func:`slash_argument_context`), so a
        message kept apart from the command sits before the slash or on another
        line.
        """
        span = slash_token_span(self.text, self._caret_offset(), self._command_names)
        if span is None:
            # No command token at the caret — nothing to extract. Fall back to a
            # plain submit so a stray call still does the least surprising thing.
            self._submit()
            return
        token_start, token_end = span
        command_text = self.text[token_start:token_end].strip()
        remainder = self.text[:token_start] + self.text[token_end:]
        if not remainder.strip():
            # Whole-buffer command: the command IS the draft. Ordinary submit.
            self._submit()
            return
        # Inline. A PROMPT command (goal/loop/team/agent/btw) reassembles to the
        # front with the draft as its argument and STAGES — never auto-runs —
        # because its argument is free text the user is still writing, and
        # treating the trailing draft as a name/request would silently consume it
        # (the D1 data-loss). A non-prompt command (a selector or a listing, like
        # `/usage`) splices out and runs, keeping the surrounding draft.
        word, _, typed_argument = command_text[1:].partition(" ")
        word = word.lower()
        if word in self._prompt_commands:
            # A prompt command with an ARGUMENT LIST (``/team``/``/agent``) and no
            # name chosen yet does not reassemble on the word alone — the name is
            # picked from the autofill first. `_apply_command` already completed
            # the word to ``/team `` and opened that list; leaving it open is the
            # whole interaction. Reassembly happens when the NAME row is chosen
            # (see :meth:`_resolve_argument`). A prompt command with no list
            # (``/goal``/``/loop``/``/btw``) reassembles now: the draft is its
            # argument directly.
            if word in self._argument_commands and not typed_argument.strip():
                return
            self._reassemble_prompt_command(token_start, token_end)
            return
        self._picker.close()
        self._splice_command(token_start, token_end)
        self.post_message(InlineCommandRequested(command_text))

    def _reassemble_prompt_command(self, token_start: int, token_end: int) -> None:
        """Move an inline PROMPT command to the front, draft as its argument.

        The safe resolution of "a message typed, then a command to route it" for
        commands whose argument is FREE TEXT (``/goal``, ``/loop``, ``/team``,
        ``/agent``, ``/btw``). Rather than guess which trailing words are a name
        and which are the message — the guess that silently ate a user's request
        when ``/team`` treated ``and then ship it`` as a team name (D1) — the
        composer is rewritten to ``/<command> <the rest of the draft>`` and left
        STAGED: nothing is submitted, the caret lands at the end, and the command
        word paints as recognised (the resync below). The user reads the
        assembled line and presses Enter when it is right.

        These commands are "start of the composer" commands by nature; this is
        the affordance that lets them be reached by typing anywhere and then
        assembled, without losing a keystroke of the draft.
        """
        text = self.text
        # ``command`` is the whole token — ``/goal`` when no argument was typed,
        # or ``/team ops`` when the user hand-typed one before engaging. In the
        # intended flow (word engaged, name picked from the autofill) there is no
        # typed argument, so ``command`` is just ``/<word>`` and the draft becomes
        # its argument cleanly. If a user DID hand-type an argument and then
        # engage inline, the draft is appended AFTER it (``/team ops`` + ``review
        # this`` -> ``/team ops review this``): the two read in the opposite order
        # to how they were typed, but the result still parses (name ``ops``,
        # request ``review this``) and is STAGED not run, so the user sees and can
        # fix it before sending — deliberately preferred over guessing which
        # typed words to reorder (review round 2, minor-1).
        command = text[token_start:token_end].strip()
        # The rest of the draft, with the command token and one adjoining
        # separator removed, is what becomes the command's argument. Reuse the
        # same separator rule the splice uses so ``msg /goal`` and ``/goal\nmsg``
        # both collapse to just ``msg``.
        start, end = token_start, token_end
        if start > 0 and text[start - 1] in " \t\n":
            start -= 1
        elif end < len(text) and text[end] in " \t\n":
            end += 1
        rest = (text[:start] + text[end:]).strip()
        assembled = f"{command} {rest}" if rest else f"{command} "
        # Staged, not submitted: the user reviews the assembled prompt and sends
        # it themselves. ``_set_text_and_caret`` re-derives the pickers so the
        # command word at the front is recognised and, for a list-taking command
        # with no argument yet, its argument list opens.
        self._set_text_and_caret(assembled, len(assembled))

    def _complete_name_argument(self, name: str) -> None:
        """Fill a team/agent NAME and a trailing space, WITHOUT submitting.

        The inverse of :meth:`_complete_argument`'s "no space so the matcher
        keeps matching". Here the space is load-bearing in the other direction:
        it terminates the name so the argument list closes (``slash_argument``
        goes to an empty match set, verified), and it opens the free-text
        message tail the user now types. Parking the caret after it is what lets
        `/team frontend-guild ` become either an attach-only switch (a blank
        Enter — the app's dispatch strips the bare name) or `/team frontend-guild
        fix the bug` (a typed message then Enter) without either being a picker
        gesture. Neither Tab nor Enter ever submits for these commands, because
        for a NAME+message command "the name is chosen" is not "run it", it is
        "ready for the message" — see :attr:`NAME_ARGUMENT_COMMANDS`.

        Setting the buffer funnels through ``_set_text_and_caret`` →
        ``_sync_picker``, which re-derives the picker as an argument list with no
        matches, so the list closes on the same keystroke that completed the name.

        INLINE (a draft typed before the command) reassembles instead: the whole
        ``/<cmd> <name>`` construct moves to the FRONT with the surviving draft as
        its message, so ``review this /team`` + picking ``frontend-guild`` becomes
        ``/team frontend-guild review this`` staged — the same safe reassembly
        every prompt command uses, never consuming the draft as a name.
        """
        context = slash_argument_context(
            self.text, self._argument_commands, self._caret_offset(), self._command_names
        )
        if context is None:
            return
        # Fill the name+space through the shared helper (see
        # :meth:`_apply_command`). ``completion_for`` models BOTH edits this
        # completion can make — the span replacement, and the inline
        # reassembly that moves the whole construct to the front when a draft
        # survives outside the token — so the buffer this writes is the exact
        # buffer the ghost predicted (review round 1, B1). Reading the answer
        # from there rather than recomputing the reassembly here is what makes
        # that guarantee structural instead of a coincidence of two edits
        # happening to agree.
        # An INLINE engage (a draft outside the token) therefore arrives here
        # already reassembled, and is STAGED rather than submitted — the user
        # reads the assembled line and presses Enter themselves.
        # ``_set_text_and_caret`` re-derives the pickers either way, so the
        # command word paints as recognised and a list-taking command opens its
        # argument list.
        completed = self._completion_for(CompletionMode.NAME_ARGUMENT, name)
        if completed is None:
            return
        self._set_text_and_caret(*completed)

    def _extend_to_common_prefix(self) -> None:
        """Grow the typed word to the matches' longest common prefix, no further.

        No trailing space and NO close, unlike :meth:`_apply_command`: the word is
        still a prefix of several commands, so the list has to stay up for the
        user to keep narrowing. Returns having changed nothing when the query is
        already the common prefix — the honest outcome, since there is no
        keystroke-free way to tell the candidates apart at that point.
        """
        context = slash_context(self.text, self._caret_offset(), self._command_names)
        if context is None:
            return
        names = [name for name, _ in self._picker.suggestions()]
        if not names:
            return
        # Case-insensitively, since the query is matched that way. The surviving
        # prefix is taken from the FIRST name so the inserted text keeps the
        # registry's own casing rather than the user's.
        shared = names[0]
        for name in names[1:]:
            while shared and not name.lower().startswith(shared.lower()):
                shared = shared[:-1]
            if not shared:
                return
        if len(shared) <= len(context.query):
            return
        # Grow only the word SPAN ``[start, end)``; a trailing inline message
        # survives untouched, and the caret lands at the new end of the word so
        # the user keeps narrowing where they were typing.
        text = self.text
        self._set_text_and_caret(
            f"{text[: context.start]}/{shared}{text[context.end :]}",
            context.start + len(shared) + 1,
        )

    # -- history ------------------------------------------------------------
    def _caret_row(self) -> int:
        return self.selection.end[0]

    def _caret_at_top_edge(self) -> bool:
        return self._caret_row() == 0

    def _caret_at_bottom_edge(self) -> bool:
        return self._caret_row() == self.document.line_count - 1

    def _record_history(self, text: str) -> None:
        stripped = text.strip()
        if stripped and (not self._history or self._history[-1] != stripped):
            self._history.append(stripped)
            if len(self._history) > self.HISTORY_LIMIT:
                self._history.pop(0)
        self._history_index = None

    def _navigate_history(self, direction: int) -> None:
        """Recall a previous prompt, WITHOUT carrying this draft's attachments.

        Marker numbers restart at #1 on every submit, so a recalled prompt's
        ``[Image #1]`` and the current draft's ``[Image #1]`` are different
        images with the same name. Leaving ``_attachments`` alone while
        replacing the text therefore resolved the recalled marker against
        whatever the live draft happened to hold: review round 17 reproduced
        paste-submit-paste-Up-Enter sending the SECOND screenshot under the
        first one's label, which is the worst possible failure for this feature
        — silent, and the model answers about a picture the user did not send.

        Recalled prompts have no attachments at all. The images went with the
        message when it was submitted, and the transcript owns them now; the
        text comes back as a starting point, and its markers resolve to nothing
        until the user pastes again. The DRAFT's attachments are stashed and
        restored with its text, because that is still the same unsent message.
        """
        if not self._history:
            return
        if self._history_index is None:
            if direction < 0:
                self._draft = self.text
                self._draft_attachments = dict(self._attachments)
                self._history_index = len(self._history) - 1
            else:
                return  # Down with no navigation active: nothing to restore
        else:
            self._history_index += direction
        if self._history_index >= len(self._history):
            # Past the newest entry: restore the draft and exit navigation.
            self._history_index = None
            self.text = self._draft
            self._attachments = dict(self._draft_attachments)
            self._sync_next_marker()
            self.move_cursor(self._end_of_buffer())
            return
        self._history_index = max(0, self._history_index)
        self.text = self._history[self._history_index]
        self._attachments.clear()
        # From the BUFFER, not from the now-empty map: history entries number
        # from #1 too, so leaving the counter alone issued a number already
        # standing in the recalled text and the next paste stole its marker
        # (review round 18).
        self._sync_next_marker()
        self.move_cursor(self._end_of_buffer())

    def _end_of_buffer(self) -> tuple[int, int]:
        last_row = self.document.line_count - 1
        return last_row, len(self.document.get_line(last_row))
