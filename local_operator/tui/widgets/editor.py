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
from textual.actions import SkipAction
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
#: because entry is taught three ways (tip, placeholder, green chevron) while
#: exit was taught by nothing on screen (design round 1, D2): the placeholder
#: is the one surface guaranteed visible the moment the mode opens on an
#: empty buffer, which is exactly when a first-timer looks for the door.
SHELL_PLACEHOLDER = "Run a command… — esc to leave"


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
        self._picker = CommandPicker(self._apply_command, self._on_picker_highlight)
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
        #: The GESTURE claim, distinct from the receipt flag above and with a
        #: different lifetime. `_copied` answers "is the receipt on screen still
        #: true?", which ends at the first EDIT to the copied text. This answers
        #: "is a hand still completing a copy?", which ends as soon as the
        #: highlight the copy took stops being the highlight on screen. Fusing
        #: them gave one of the two the wrong lifetime in whichever direction
        #: the fusion leaned (R18-1, agent review round 18).
        self._copy_gesture = False
        #: The selection `_copy_drag` took, while `_copy_gesture` holds.
        self._copied_selection: Selection | None = None
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
        return self._paint_slash(self._paint_markers(super().render_line(y), y), y)

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
        self.placeholder = SHELL_PLACEHOLDER if active else DEFAULT_PLACEHOLDER
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
                self.set_shell_mode(False)
                event.stop()
                event.prevent_default()
                return
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
                self.action_copy()
            else:
                self.post_message(InterruptRequested())
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
        await super()._on_key(event)

    # -- submit -------------------------------------------------------------
    def _submit(self) -> None:
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
                if name and name.lower() in self._name_choices:
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
          ``ctrl+c,super+c`` to ``action_copy``, and neither key arrives:
          cmd+C is eaten by the terminal (Ghostty binds
          ``super+c=copy_to_clipboard:mixed`` without ``performable:``), and
          Ctrl+C is consumed by :meth:`_on_key` as this app's interrupt before
          any binding runs. That is deliberate — the interrupt cannot become
          conditional on a live highlight. :meth:`_on_key` therefore routes
          the press to :meth:`action_copy` itself when a real range is live,
          which is the one copy sequence the composer has.

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
        """Is a copy gesture still in progress or still visible on screen?

        THE predicate the app's Ctrl+C rung asks, named once here rather than
        duck-typed from outside. It was previously two `getattr` probes into
        this class's privates, which a rename would have degraded silently to
        "no copy in flight" — i.e. to always clearing the draft — with no test
        failing (R17 MINOR-2). A property breaks loudly instead, and gives the
        question a name that says what it means.

        Two moments, because a copy spans both: the drag itself, and the window
        after release in which the highlight it took is still the highlight on
        screen. `watch_selection` ends the second the moment that stops being
        true, so neither flag can outlive the gesture the way three earlier
        predicates did.
        """
        return self._selecting or self._copy_gesture

    def action_copy(self) -> None:
        """Copy the live range. THE composer's copy gesture.

        The composer cannot rely on Textual's binding reaching this method on
        its own: cmd+C is eaten by the terminal and Ctrl+C is the app's
        interrupt (see :meth:`_copy_drag`), so :meth:`_on_key` routes the
        highlight-then-Ctrl+C sequence here directly. This override exists so
        that path — and any other caller of the ``copy`` action — writes the
        clipboard through the SAME message a transcript copy uses, rather than
        ``super().action_copy()``: Textual's base writes silently, and this
        app's rule is that a copy says so. One clipboard write and one receipt
        for every gesture, or the toast becomes evidence about which key
        carried it.

        The gesture flags are deliberately NOT set here. ``_copy_gesture`` is
        what defers Ctrl+C's interrupt meaning while a copy's highlight is on
        screen, and a highlight outlives every gesture — it sits there until
        the caret moves. A press is over the instant it lands, so leaving the
        gesture armed after one would defer the NEXT press for as long as the
        stale highlight happens to remain: the user presses Ctrl+C on a
        minutes-old range expecting the draft rung and gets a re-copy instead,
        with the exit ladder's second tap a lost draft away (D17/D20). The
        receipt flag IS set — the toast it drives is a claim about the
        clipboard, and editing the copied text falsifies it (D3).

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

    def watch_selection(self, selection: Selection) -> None:
        """Retire the copy receipt's GESTURE claim when the highlight changes.

        ``_copied`` is edit-scoped: `edit` and `load_text` clear it, because the
        receipt on screen is a claim about text the user can still see. But the
        app's Ctrl+C rung asks a GESTURE-scoped question — "is a hand still
        completing a copy?" — and an edit-scoped flag answers it wrongly in two
        directions that each cost a user their draft (D20, then D22).

        Retiring here makes the flag answer both questions with one lifetime:
        the copy's claim ends when the highlight it took stops being the
        highlight on screen, whether that is a caret move collapsing it or a new
        selection replacing it. `_copied_selection` is what makes "the same
        highlight" checkable rather than merely "a highlight".

        Deliberately NOT posting ``EditorCopyStale``: moving the caret is not
        the user editing the text their receipt describes, so the toast remains
        true and stays up. Only ``_copy_gesture`` ends here, and it is a
        SEPARATE field from ``_copied`` for exactly that reason — pointing the
        receipt flag at this lifetime left a receipt on screen asserting a copy
        of characters the user had since deleted, which is design round 1's D3
        verbatim (R18-1, agent review round 18).
        """
        super_watch = getattr(super(), "watch_selection", None)
        if super_watch is not None:
            super_watch(selection)
        # `getattr` with a default because a reactive watcher can fire during
        # base-class construction, before this subclass has set its own
        # attributes — an AttributeError there takes the whole widget down.
        if getattr(self, "_copy_gesture", False) and selection != getattr(
            self, "_copied_selection", None
        ):
            self._copy_gesture = False
            self._copied_selection = None

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
        # Select-to-overwrite is the commonest edit in any input, and when it
        # follows a copy it falsifies the copy's receipt: the user drags a word
        # to replace it, types, and a receipt asserting a copy of characters
        # that no longer exist sits on screen for another five seconds (design
        # round 1, D3). The clipboard keeps what it took — that is what a copy
        # IS, and silently un-copying would be worse — but the CLAIM is retired
        # the moment its subject is edited away.
        stale_receipt = self._copied
        result = super().edit(edit)
        if stale_receipt:
            self._copied = False
            self._copy_gesture = False
            self._copied_selection = None
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
        self._copy_gesture = False
        self._copied_selection = None
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
            # caret is normally down in the message body.
            keep = self._is_multiline_name_command() and (
                self._name_choices_family == self._name_command_family(self._leading_command_word())
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
                    subcommand = first_tok.lower() if list_argument else ""
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
        cursor = self._caret_offset()
        if self._picker.mode is PickerMode.ARGUMENT:
            return slash_argument(self.text, self._argument_commands, cursor, self._command_names)
        context = slash_context(self.text, cursor, self._command_names)
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
        context = slash_context(self.text, self._caret_offset(), self._command_names)
        if context is None:
            return
        # Replace ONLY the command token ``[start, end)`` with ``/name ``. The
        # word used to run to the end of the buffer, so a completion could splice
        # to the end; inline detection means a message may follow the token
        # (``fix this /te|`` or ``/te| ship it``), and that suffix is the user's
        # prose — it must survive verbatim. The trailing space is load-bearing:
        # it terminates the word (closing the command list) and, for a
        # list-taking command, opens the argument list.
        text = self.text
        completed = f"{text[: context.start]}/{name} {text[context.end :]}"
        self._set_text_and_caret(completed, context.start + len(name) + 2)

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
        context = slash_argument_context(
            self.text, self._argument_commands, self._caret_offset(), self._command_names
        )
        if context is None:
            return
        # Replace the argument SPAN ``[start, end)`` rather than the buffer tail:
        # inline detection means the command may not be the last thing in the
        # draft, so trimming the argument's length off the end would corrupt a
        # trailing message. ``end`` is the end of the command's line, which is
        # the whole argument by construction.
        text = self.text
        self._set_text_and_caret(
            f"{text[: context.start]}{name}{text[context.end :]}", context.start + len(name)
        )

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
        # Fill the name+space into the argument SPAN (not the buffer tail), so a
        # trailing inline message would survive. ``context.end`` is the end of the
        # command's line, the whole argument by construction.
        text = self.text
        filled = f"{text[: context.start]}{name} {text[context.end :]}"
        caret = context.start + len(name) + 1
        # If a draft survives outside the command token, this is an INLINE engage:
        # reassemble to the front rather than leaving the command mid-draft. The
        # name is already filled, so the token now reads ``/team <name>``; the
        # reassembly moves it to the front with the rest of the draft appended.
        outside = (text[: context.token_start] + text[context.end :]).strip()
        if outside:
            # Recompute the token span on the FILLED text (the name changed its
            # length) and reassemble from there.
            self.text = filled
            self.move_cursor(self._location_at_offset(caret))
            span = slash_token_span(self.text, caret, self._command_names)
            if span is not None:
                self._reassemble_prompt_command(*span)
            return
        self._set_text_and_caret(filled, caret)

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
