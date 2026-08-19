"""Input editor — TextArea inverted to chat semantics.

Textual's TextArea defaults to newline-on-Enter; this product wants
submit-on-Enter. The subclass inverts that and takes the terminal key idioms:

- ``Enter`` submits (posts :class:`EditorSubmitted`); with the command picker
  open it first completes the highlighted command, THEN submits
- ``Shift+Enter`` inserts a newline
- ``Ctrl+C`` posts :class:`InterruptRequested` (abort the turn) — never exits
- ``Ctrl+D`` on an EMPTY buffer quits; otherwise it falls through to delete
- ``Up``/``Down`` move the picker's highlight while it is open; otherwise they
  cycle prompt history when the caret sits at the top/bottom edge of the
  buffer, and inside the text they keep their cursor-move meaning
- ``Tab`` completes the highlighted command WITHOUT submitting
- ``Esc`` dismisses the picker, leaving the typed text alone

Key interception happens in :meth:`_on_key`, which runs BEFORE TextArea's
document-insert path, so a handled key never reaches the buffer. Unhandled
keys fall through to the stock editor behavior.

Releasing a drag over the composer COPIES the highlighted text, the same rule
the transcript follows and for the same reason — neither Ctrl+C nor cmd+C is
available to bind here. See :meth:`Editor._copy_drag`.

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
from bisect import bisect_right
from collections.abc import Iterable, Mapping
from dataclasses import dataclass
from pathlib import Path
from stat import S_ISREG
from typing import Callable

from rich.cells import cell_len
from rich.segment import Segment
from rich.style import Style as RichStyle
from textual import events
from textual.content import Content
from textual.expand_tabs import expand_tabs_inline
from textual.geometry import Offset
from textual.message import Message
from textual.strip import Strip
from textual.style import Style as ContentStyle
from textual.widgets import TextArea
from textual.widgets.text_area import Edit, EditResult, Selection

from local_operator.harness.types import ImageContent
from local_operator.imaging import bound_image_for_model
from local_operator.media import ImageInfo, sniff_image, sniff_image_file
from local_operator.tui.autocomplete import ArgumentMode, SlashCommand
from local_operator.tui.widgets.command_picker import (
    CommandPicker,
    PickerMode,
    slash_argument,
    slash_context,
)
from local_operator.tui.widgets.model_picker import ModelPicker, ModelRow

#: Refused above this, as a plain text paste. Providers cap an image at 5 MB of
#: base64, which is 3.75 MB of bytes; 4 MB of source is comfortably inside that
#: after the 4/3 inflation and still holds any screenshot. The point is that
#: the refusal happens HERE, where it is one visible path in the composer the
#: user can act on, rather than as a provider 400 mid-turn.
MAX_ATTACHMENT_BYTES = 4 * 1024 * 1024

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

    Terminals deliver a dropped or copied file as its path, shell-quoted:
    Ghostty writes a clipboard image to a temp file and pastes that name, and a
    Finder drag arrives with spaces backslash-escaped. ``shlex`` is exactly the
    grammar they are quoting for, so it is what unpicks it — hand-rolled
    unescaping is how a path with a space becomes two paths that do not exist.

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


class EditorCopied(Message):
    """Posted when a drag over the composer finishes on a real selection.

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


class EditorQuit(Message):
    """Posted on Ctrl+D with an empty buffer."""

    def __init__(self) -> None:
        super().__init__()


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


#: The composer's resting prose. Named rather than inlined because the app
#: swaps it for :data:`READ_ONLY_PLACEHOLDER` while the full-page subagent
#: view is open and has to be able to put this one back.
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


class Editor(TextArea):
    """Multiline prompt editor with submit-on-Enter, history, slash-completion."""

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

    #: The attachment chip's two grounds, on top of everything ``TextArea``
    #: already declares. Component classes rather than hexes in Python so the
    #: colours sit in the stylesheet beside every other composer colour and
    #: follow the theme's ``$lo-*`` variables through a theme switch.
    COMPONENT_CLASSES = TextArea.COMPONENT_CLASSES | {
        "text-area--image-marker",
        "text-area--image-marker-selected",
    }

    def __init__(
        self,
        placeholder: str = DEFAULT_PLACEHOLDER,
        commands: list[SlashCommand] | None = None,
    ) -> None:
        # Built BEFORE super().__init__: TextArea's constructor loads its
        # initial document, which funnels through load_text() and therefore
        # through _sync_picker().
        self._picker = CommandPicker(self._apply_command)
        self._model_picker = ModelPicker(self._apply_model)
        # Which list-taking command the argument list is currently open for, or
        # None when the buffer is not in one. This is the transition edge the
        # ArgumentQueryOpened message rides: without it the app would rebuild
        # the rows on every character typed into the query. Assigned here for
        # the same reason as the pickers — _sync_picker() reads it during
        # super().__init__().
        self._argument_command: str | None = None
        # Command words (primaries AND aliases) whose argument opens the value
        # list, and the subset of those the bare command cannot stand without.
        # DERIVED from the registry in :meth:`set_commands` rather than listed
        # here: a hand-kept tuple beside a registry that already states the fact
        # is a second source of truth, and the way it fails is a command whose
        # description advertises options the editor never offers.
        self._argument_commands: tuple[str, ...] = ()
        self._required_argument_commands: tuple[str, ...] = ()
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
        #: The text of the copy this widget most recently announced, held only
        #: until the buffer is edited. It is the SUBJECT of the receipt on
        #: screen: once the user types over what they highlighted, the card is
        #: making a claim about characters that are gone. See :meth:`edit`.
        self._copied_text: str | None = None
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
        return self._paint_markers(super().render_line(y), y)

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
        self.text = "/model "
        self.move_cursor(self._end_of_buffer())
        self._history_index = None

    # -- key interception ---------------------------------------------------
    async def _on_key(self, event: events.Key) -> None:
        """Handle chat keys before TextArea's insert path sees them."""
        key = event.key
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
                self._model_picker.close()
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
            self._picker.dismiss()
            event.stop()
            event.prevent_default()
            return
        if self._picker.is_open():
            if key == "escape":
                self._picker.dismiss()
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
                        if key == "enter" and not self.opens_a_list(name):
                            self._submit()
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
            # It has to be handled rather than left to bubble, because
            # ``TextArea`` binds Escape to ``blur``. That made Esc a silent trap:
            # the first press moved focus out of the composer (so the user's next
            # keystrokes went nowhere) and only a LATER press, once focus had
            # already left, reached the app's stop. Consuming the key here keeps
            # focus put and gives Esc one meaning — stop what the agent is doing.
            self.post_message(StopRequested())
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
        if key == "ctrl+c":
            self.post_message(InterruptRequested())
            event.stop()
            event.prevent_default()
            return
        if key == "ctrl+d" and not self.text:
            self.post_message(EditorQuit())
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
        await super()._on_key(event)

    # -- submit -------------------------------------------------------------
    def _submit(self) -> None:
        text = self.text
        # Checked HERE, before the post, because that is the only place the
        # entry can be prevented rather than removed afterwards — see
        # :meth:`set_records_history`.
        if text.strip() and self._records_history:
            self._record_history(text)
        self._picker.close()
        # Only the attachments the text STILL REFERS TO. The marker is the
        # authority: pasting three screenshots and deleting two must send one,
        # because the deleted markers are the user saying they changed their
        # mind, and silently sending all three is both surprising and expensive.
        self.post_message(EditorSubmitted(text, self.referenced_images(), self._attachments))
        self.clear_content()

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

        The release is also where a drag over the composer becomes a COPY, for
        the same reason it is in the transcript — see :meth:`_copy_drag` and
        ``OperatorApp.on_text_selected``.
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
            # receipt for it on a CLICK, which is precisely the noise
            # :meth:`_copy_drag` avoids by ignoring collapsed selections.
            return
        self._copy_drag()

    def _copy_drag(self) -> None:
        """Put a composer drag on the clipboard, exactly as the transcript does.

        Reported from the field: text highlighted in the composer "doesn't copy
        properly" — the drag paints a highlight and the clipboard keeps whatever
        it held before. Both halves of the app's copy story miss this widget:

        * **The release does not reach the app.** ``OperatorApp.on_text_selected``
          copies ``Screen.get_selected_text()``, but a ``TextArea`` never
          contributes to a screen selection. ``TextArea._watch_selection`` calls
          ``app.clear_selection()`` on every caret move, and the mouse-down that
          begins a composer drag moves the caret — so ``Screen.selections`` is
          wiped on the first event of the gesture and stays empty for the rest
          of it. Measured before this fix: after a drag over the composer,
          ``editor.selected_text`` was ``'summarise the inges'`` while
          ``screen.get_selected_text()`` was ``None`` and the clipboard was ``''``.
          The base class also captures the mouse on press, so ``Screen``'s own
          select machinery is bypassed and ``_select_state`` never leaves ``None``.

        * **No key can rescue it.** ``TextArea`` binds ``ctrl+c,super+c`` to
          ``action_copy``, and neither key arrives: cmd+C is eaten by the
          terminal (Ghostty binds ``super+c=copy_to_clipboard:mixed`` without
          ``performable:``), and Ctrl+C is consumed by :meth:`_on_key` as this
          app's interrupt before any binding runs. That is deliberate — the
          interrupt cannot become conditional on a live highlight — so the
          composer is left with a highlight and no way to spend it.

        The fix is the rule the transcript already states: **the release IS the
        copy**. The gesture carries itself, the user gets the same toast, and no
        key changes meaning.

        Only a real range copies. A plain click leaves ``start == end`` and is
        not a copy — nor is the marker click above, which the caller returns on
        before reaching here because that selection is the app's, made so
        backspace can delete the chip whole.

        ``selected_text`` is the DOCUMENT's text, which is the right claim for
        an input: what a user copies out of a field is what they typed and can
        paste back, so an attachment marker copies as the ``[Image #1, …]`` text
        that cites it — the same characters :meth:`_submit` would send, and the
        ones that re-cite the image if pasted into another draft. In a read-only
        composer (subagent view) that text is the app's rather than the user's,
        and it copies too: what the field shows is what a drag over it takes.

        Only THIS widget's own drag copies, which ``_selecting`` is the record
        of — ``TextArea._on_mouse_down`` sets it and ``TextArea._on_mouse_up``
        clears it, and this runs in between (review round 1, F1/F2). Without
        that gate every mouse-up delivered here copies, and two of them are not
        copy gestures at all:

        * A drag that STARTS in the transcript and is released over the composer
          — the ordinary way to select to the end of the answer, since the
          composer is docked below it. ``TextArea`` leaves a selection live
          after its own drag, so the composer still holds the last range the
          user highlighted in it, and re-copying that range overwrites the
          transcript copy the same release just made. Measured: the user dragged
          the agent's answer and got their own draft, with a toast confirming
          it.
        * A bare mouse-up over a selection made with shift+arrows, which no
          mouse gesture asked to copy.

        A guard phrased as "did the PRESS land in this widget" would fix the
        first and miss the second; ``_selecting`` answers the question that
        actually matters, which is whether this widget is mid-drag.
        """
        if not self._selecting:
            return
        text = self.selected_text
        if not text:
            return
        self._copied_text = text
        self.post_message(EditorCopied(text))

    # -- paste ----------------------------------------------------------------
    async def _on_paste(self, event: events.Paste) -> None:
        """Attach pasted images instead of pasting the path to them.

        Textual's ``Paste`` carries TEXT only — there is no binary channel at
        the terminal, so an image never arrives as bytes here. What arrives on
        the owner's setup is a PATH: Ghostty writes a clipboard image to
        ``$TMPDIR/clipboard-<stamp>-<hash>.png`` and bracketed-pastes the
        filename. Finder's Cmd+C and a drag-and-drop both land the same way.
        That is the hot path, and it is why this hooks paste rather than
        binding a key to read the system clipboard.

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
        attached = await self._attach_pasted_images(event.text)
        if attached is None:
            return
        event.prevent_default()
        event.stop()
        self.insert(attached)

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

        loaded: list[tuple[ImageContent, str]] = []
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
            if not S_ISREG(stat.st_mode) or stat.st_size > MAX_ATTACHMENT_BYTES:
                return None
            info = sniff_image_file(path)
            # `sendable` and not merely "recognised": HEIC sniffs fine and no
            # provider will take it, so attaching it would trade a readable
            # path in the prompt for a 400 later in the turn.
            if info is None or not info.sendable:
                return None
            try:
                data = Path(path).read_bytes()
            except (OSError, ValueError):
                return None
            if len(data) > MAX_ATTACHMENT_BYTES:
                # The stat above is the real gate; this catches a file that grew
                # between the two calls.
                return None
            # BOUND before attaching, in a thread. The bytes on disk are
            # whatever the screen produced, and a provider refuses an image over
            # 2000 pixels on its long edge as soon as the request carries more
            # than twenty of them (see local_operator.imaging). Forwarding
            # verbatim was therefore not "lossless", it was a delayed fault: a
            # 2206x266 paste sat harmlessly in the history for a hundred turns
            # and then wedged the session permanently the moment the twenty
            # first screenshot arrived, because the block is in the HISTORY and
            # every later request — including the compaction that is supposed to
            # be the escape hatch — re-sends it and earns the same 400.
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
            loaded.append(
                (
                    ImageContent(
                        data=base64.b64encode(payload).decode("ascii"),
                        mime_type=wire_mime,
                    ),
                    # The marker reports what was ATTACHED, not what is on disk.
                    # A marker reading 2560x1440 beside a 1568x882 attachment is
                    # a receipt for something that was never sent, and the whole
                    # point of the dimensions is that the user can check them at
                    # a glance.
                    _bounded_dimensions(payload, info),
                )
            )

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
        # Select-to-overwrite is the commonest edit in any input, and after this
        # widget started copying on release it also became a clipboard write:
        # the user drags a word to replace it, types, and a receipt asserting a
        # copy of characters that no longer exist sits on screen for another
        # five seconds (design round 1, D3). The clipboard keeps what it took
        # — that is what a copy IS, and silently un-copying would be worse —
        # but the CLAIM is retired the moment its subject is edited away.
        stale_receipt = self._copied_text is not None
        result = super().edit(edit)
        if stale_receipt:
            self._copied_text = None
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
        super().load_text(text)
        self._sync_picker()

    def _sync_picker(self) -> None:
        """Re-derive EVERY list from the buffer.

        The buffer is the single authority, so no picker holds state another could
        contradict: `slash_context` is live while the command word is open and
        `slash_argument` takes over on the terminating space, which makes "exactly
        one list is showing" a property of the parse rather than a rule the widgets
        have to cooperate on. The command picker serves both the word and a
        provider argument, so the branch below is which LIST it derives, not which
        widget is visible.
        """
        list_argument = slash_argument(self.text, self._argument_commands)
        if list_argument is None:
            self._argument_command = None
            self._picker.sync(self.text)
        else:
            command = self._command_word()
            if command != self._argument_command:
                self._argument_command = command
                # Drop the previous command's rows before asking for this one's.
                # `/login` offers every provider and `/logout` only the ones with a
                # credential, so carrying them across would briefly offer a logout
                # from an account the user never had.
                self._picker.set_choices([])
                self.post_message(ArgumentQueryOpened(command or ""))
            self._picker.sync_argument(list_argument)
        argument = slash_argument(self.text, self.MODEL_COMMANDS)
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

    def _command_word(self) -> str | None:
        """The lower-cased command word on the buffer's first non-blank line."""
        line = next((line for line in self.text.split("\n") if line.strip()), "")
        stripped = line.lstrip()
        if not stripped.startswith("/"):
            return None
        return stripped[1:].partition(" ")[0].lower()

    def _complete_model(self, row: ModelRow) -> None:
        """Put ``row``'s selector in the buffer without acting on it.

        No trailing space, unlike a command completion: the selector IS the whole
        argument, and a trailing space would terminate it and close the list — so
        Tab would appear to both fill the field and give up on it.
        """
        self.text = f"/model {row.selector}"
        self.move_cursor(self._end_of_buffer())

    def _apply_model(self, row: ModelRow) -> None:
        """Hand a chosen row to the app and clear the buffer.

        The buffer is cleared HERE rather than by the handler because the command
        never reaches the submit path: choosing from the list is the whole
        interaction, so leaving `/model anthropic/claude-opus-5` behind would
        invite a second Enter that ran the switch again.
        """
        self._model_picker.close()
        handler = self._on_model_chosen
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

        Read off the buffer's command word rather than the rows: what a keystroke
        destroys is a property of the command, and a per-row flag would make the
        gate depend on data the app happened to fill in.
        """
        return (
            self._picker.mode is PickerMode.ARGUMENT
            and self._argument_command in self.DESTRUCTIVE_COMMANDS
        )

    def _picker_query(self) -> str | None:
        """The text the open list is matching against, or ``None`` when closed.

        One gate, two lists: the command word and a provider argument are parsed
        by different functions but judged by the same rule, so the destructive
        case each protects (`/lo` reaching `loop`, `/logout an` reaching a stored
        credential) cannot drift apart.
        """
        if self._picker.mode is PickerMode.ARGUMENT:
            return slash_argument(self.text, self._argument_commands)
        context = slash_context(self.text)
        return None if context is None else context.query

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
            if self._argument_is_destructive():
                self._complete_argument(name)
                return
            self._run_argument(name)
            return
        context = slash_context(self.text)
        if context is None:
            return
        # Everything from the slash onward IS the command word (that is what
        # makes it a command word), so the prefix is all that must survive.
        self.text = f"{self.text[: context.start]}/{name} "
        self.move_cursor(self._end_of_buffer())

    def _resolve_argument(self, name: str, key: str, unambiguous: bool) -> None:
        """Tab/Enter on an argument row: complete it, or run it.

        Enter runs only what the same gate calls unambiguous, for the reason the
        gate exists at all: `/logout` DELETES a credential, so acting on a row the
        fuzzy matcher guessed would make one mis-keystroke destructive. An
        ambiguous Enter completes instead — the buffer then holds the exact id,
        which is one match, so the second Enter runs it.
        """
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
        argument = slash_argument(self.text, self._argument_commands)
        if argument is None:
            return
        # The argument is by construction the TAIL of the buffer (everything after
        # the command word's space, on the only non-blank line), so trimming its
        # length off the end leaves exactly `…/logout ` to append onto.
        self.text = f"{self.text[: len(self.text) - len(argument)]}{name}"
        self.move_cursor(self._end_of_buffer())

    def _run_argument(self, name: str) -> None:
        """Complete ``name`` and submit, so the command's own handler runs it.

        Submitting the finished text rather than calling a callback keeps ONE
        implementation of what `/login anthropic` means: the app's slash dispatch.
        A second path would be a second place for the login flow to drift.
        """
        self._complete_argument(name)
        self._submit()

    def _extend_to_common_prefix(self) -> None:
        """Grow the typed word to the matches' longest common prefix, no further.

        No trailing space and NO close, unlike :meth:`_apply_command`: the word is
        still a prefix of several commands, so the list has to stay up for the
        user to keep narrowing. Returns having changed nothing when the query is
        already the common prefix — the honest outcome, since there is no
        keystroke-free way to tell the candidates apart at that point.
        """
        context = slash_context(self.text)
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
        self.text = f"{self.text[: context.start]}/{shared}"
        self.move_cursor(self._end_of_buffer())
        self._sync_picker()

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
