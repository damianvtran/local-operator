"""Terminal window title — the session's name and run state, outside the app.

A terminal UI owns its own frame and nothing else. The moment a user runs more
than one session — cmux workspaces down a sidebar, tmux windows, a row of
terminal tabs — the only thing that identifies a session from OUTSIDE is the
title its process sets, and a title that says ``local-operator`` five times
identifies nothing. So the app writes the SAME two facts the status band
already carries, into the one surface a window switcher can read:

    lo › Reduce agent RAM usage      idle — the user's turn
    lo ⣻ Reduce agent RAM usage      a turn is running (the separator animates)
    lo ! Reduce agent RAM usage      parked on an approval the user owes

Three things are deliberate.

- **The separator carries the state, not a word.** The title is clipped to
  whatever width the switcher gives it (cmux's sidebar shows ~24 cells), so
  state has to live in the leftmost cells or it is the first thing lost. A
  word like ``working`` would be spent before the session name is reached.
- **The glyph vocabulary is the band's, not a second one.** The working
  separator is :data:`SPINNER_FRAMES`, the exact sequence the status band
  spins (the band imports it from here, so there is one definition), and the
  attention mark is ``!``, the same alarm the band and the transcript's
  warning notices use. In cmux the title and the band are both on screen at
  once; two different vocabularies for one state would read as two states.
- **Idle is ``›``, the band's own inward chevron** (``_SEP_LEFT``) rather than
  ASCII ``>``. It is the mark this UI already uses to point at what follows,
  and "the user's turn" is exactly a prompt pointing at the user.

Everything here is either a pure function or a small state holder with an
injected sink, because the alternative — writing to ``sys.stdout`` — is a
correctness bug and not a style choice: Textual serialises terminal output
through a writer thread, and a second writer interleaves escape bytes into the
middle of a frame. The app passes a sink that funnels into ``driver.write``,
which is the same path Textual itself uses to emit OSC 52 for clipboard copy.
"""

from __future__ import annotations

import os
import re
from pathlib import Path
from typing import Callable, Literal

from local_operator.session.naming import cut_on_a_word
from local_operator.tui.settings import settings_get

#: Environment kill switch, mirroring ``shimmer``/``nerd_icons``. Wanted by
#: anything that captures raw terminal output — a CI job, a recording, a
#: `script(1)` transcript — where an OSC string every 80 ms is noise in the
#: capture rather than a title anyone sees.
_ENV_DISABLE = "LOCAL_OPERATOR_NO_TERMINAL_TITLE"

#: The brand, in the product's own lowercase voice (the welcome wordmark is
#: ``local operator``, and the launcher is ``lop``/``lo``). Two cells: the
#: title's whole budget is what a sidebar row shows before it ellipsises, so
#: the brand is charged the minimum that still identifies the agent.
BRAND = "lo"

#: Idle separator — the band's inward chevron, i.e. "your turn".
SEP_IDLE = "›"

#: Attention separator. The app-wide alarm mark: the status band's
#: auto-approve indicator and the transcript's ``warning`` notices both use
#: ``!``, so a user who has learned it in one place has learned it here.
SEP_ATTENTION = "!"

#: Failed separator, for a session whose last turn ended in an ERROR.
#:
#: Without it a died-with-an-error session was indistinguishable from one that
#: finished cleanly — both rendered ``lo › name`` — in the exact surface a user
#: scans to find what needs them. That is not hypothetical: it is how the
#: incident behind this change was found, by opening sessions one at a time to
#: see which had failed.
#:
#: ``✗`` rather than another ``!`` because the two states differ in what they
#: ask of the user: ``!`` means "come and answer something" (a turn is alive and
#: parked on you), while this means "this one is over and it did not work".
#: Sharing a glyph would spend the attention mark on a session that no longer
#: needs any. It matches the transcript's own ``error`` notice mark, so the
#: vocabulary is one the user has already learned one surface in.
SEP_FAILED = "✗"

#: The working animation, and the SINGLE definition of it in the TUI:
#: ``status_line`` imports this rather than declaring its own copy. Both
#: indicators describe one fact (a turn is running) and in a tiled terminal
#: they are visible simultaneously, so they must be the same sequence — two
#: braille spinners drawn from different tuples read as two separate things
#: happening. Braille dots animate in one cell at every font size, which is
#: what a title bar can afford.
SPINNER_FRAMES: tuple[str, ...] = ("⣾", "⣽", "⣻", "⢿", "⡿", "⣟", "⣯", "⣷")

#: Seconds between spinner frames (~12.5 fps), the band's cadence.
SPINNER_INTERVAL_S = 0.08

#: Control characters, stripped from every label before it reaches the wire.
#: This is a security boundary and not tidiness: conversation names are
#: MODEL-GENERATED, and an OSC string is terminated by BEL or ST — a name
#: containing either would close the title sequence early and leave the rest
#: of the model's text being interpreted by the terminal as commands.
#:
#: The whitespace controls (``\t``-``\r``, U+0009-U+000D) are deliberately NOT
#: in this class. They are removed too, but by ``str.split`` a line later, so
#: they collapse to a SPACE instead of vanishing: deleting the newline in
#: ``"two\nlines"`` yields ``twolines``, which is a different word.
_CONTROL_CHARS = re.compile(r"[\x00-\x08\x0e-\x1f\x7f-\x9f]")

#: Hard cap on the label. Conversation names are already capped at 80 by
#: ``session.naming``, but the cwd fallback and any future caller are not, and
#: an unbounded escape sequence written 12.5 times a second is worth refusing
#: on its own terms. No switcher shows this many cells anyway.
MAX_LABEL_CHARS = 80

#: Save/restore BOTH the icon name and the window title using xterm window
#: operations, matching :func:`osc_title`, which deliberately sets both via
#: OSC 0 so tab bars / multiplexers that read the icon channel see the same
#: session label. ``22;2t``/``23;2t`` would cover only the window title and
#: could therefore leave a pane/tab label stranded as `lo …` after exit even
#: though the restore ran. ``0`` is the "both channels" subcode.
#:
#: Terminals that do not implement these window ops ignore them, which is still
#: preferable to trying to read the current title back and replay it — there is
#: no portable readback path, and an attempted restore guessed from nowhere is
#: worse than an ignored save/restore pair.
PUSH_TITLE = "\x1b[22;0t"
POP_TITLE = "\x1b[23;0t"

#: ``failed`` is TERMINAL and turn-scoped, exactly like the other three: it is
#: entered when a turn retires with an error and left the moment the next turn
#: starts (which moves the band to ``working``). It is deliberately NOT sticky
#: beyond that — a title is a statement about the session's CURRENT state, and a
#: cross that outlived the failure it described would be worse than no mark.
TitleState = Literal["idle", "working", "attention", "failed"]


def terminal_title_enabled() -> bool:
    """Whether the app may set the terminal title (env gate + config flag).

    Same two-tier shape as ``shimmer_enabled`` and ``nerd_icons_enabled``: an
    environment kill switch for a capture or a CI run, and a
    ``display.terminal_title`` config flag for a persistent preference. On by
    default — a terminal that does not support OSC 0 ignores the sequence,
    and the save/restore pair means even a terminal that does support it is
    left exactly as it was found.
    """
    if os.environ.get(_ENV_DISABLE):
        return False
    return bool(settings_get("display.terminal_title", True))


def sanitize_label(value: str | None) -> str:
    """``value`` with control characters removed and whitespace collapsed.

    Returns ``""`` for anything that sanitises away to nothing, which callers
    read as "no label" rather than as an empty title.

    Over-long labels are cut on a WORD boundary with an ellipsis, through the
    same helper the stored title uses (``naming.cut_on_a_word``), so the tab and
    the band shorten a name identically. A bare slice ended the tab mid-word —
    `…and reconcile the ledge` — which reads as a string that ran out of buffer
    rather than as a name that was shortened, and the tab is where truncation
    bites hardest because most terminals shorten the label AGAIN to fit the tab
    strip (design review D3).

    For a CONVERSATION NAME this is now a backstop rather than the working cut:
    ``MAX_TITLE_CHARS`` and ``MAX_LABEL_CHARS`` are both 80, so a title arrives
    already shortened by the store (D6). It still does real work for the callers
    with no cap of their own — ``cwd_label`` on a deep path, and a subagent
    label — which is why the cut lives on both sides rather than being deleted
    here.
    """
    if not value:
        return ""
    cleaned = " ".join(_CONTROL_CHARS.sub(" ", value).split())
    return cut_on_a_word(cleaned, MAX_LABEL_CHARS)


def cwd_label(cwd: str | None) -> str:
    """The working directory's basename, the label used before a name exists.

    A conversation is named after its first substantive prompt, so every
    session spends its opening minutes unnamed — and an unnamed row in a
    sidebar of five sessions is the one case this feature exists to fix. The
    directory is what the user picked the session for, so it stands in until
    the real name lands. Filesystem roots yield ``""`` (a title reading
    ``lo › /`` says less than ``lo ›``).
    """
    if not cwd:
        return ""
    path = Path(cwd)
    if path.name in ("", path.anchor):
        return ""
    return sanitize_label(path.name)


def build_title(label: str, state: TitleState, frame: int = 0) -> str:
    """The title string for ``label`` in ``state`` — pure, hence testable.

    The state→separator contract is the whole feature, so it is a function of
    its arguments and nothing else:

    * ``idle`` → ``lo › label``
    * ``working`` → ``lo ⣻ label`` (the frame steps through
      :data:`SPINNER_FRAMES`)
    * ``attention`` → ``lo ! label``
    * ``failed`` → ``lo ✗ label``

    Without a label the separator still trails the brand (``lo ›``): the state
    is the half of this that a switcher can always show, and dropping it would
    make a running session indistinguishable from a finished one.
    """
    if state == "working":
        separator = SPINNER_FRAMES[frame % len(SPINNER_FRAMES)]
    elif state == "attention":
        separator = SEP_ATTENTION
    elif state == "failed":
        separator = SEP_FAILED
    else:
        separator = SEP_IDLE
    clean = sanitize_label(label)
    return f"{BRAND} {separator} {clean}" if clean else f"{BRAND} {separator}"


def osc_title(title: str) -> str:
    """``title`` wrapped in OSC 0 (icon name AND window title) with a BEL.

    OSC 0 rather than OSC 2 because tabbed terminals and multiplexers read the
    ICON name for a tab/pane label — which is the surface this feature is
    aimed at — and OSC 2 sets only the window title. BEL rather than ST
    terminates it: every terminal accepts BEL, while ST is still refused by
    some older emulators.
    """
    return f"\x1b]0;{title}\x07"


class TerminalTitle:
    """Owns the title's state and writes it, coalescing repeats.

    Constructed by the app with a ``write`` sink (see the module docstring for
    why it is injected) and driven by the status band, which is already the
    one place that knows the session name, whether a turn is running, and
    which spinner frame is current. This class therefore holds no policy: it
    stores what it is told, renders it, and refuses to write a byte when the
    rendered title has not changed.

    That refusal is load-bearing. The spinner ticks 12.5 times a second for
    the whole of every turn; while idle, the same state is re-asserted by
    every band repaint. Deduplicating here means an idle session writes to the
    terminal exactly zero times per second, and no caller has to remember to
    check first.
    """

    def __init__(self, write: Callable[[str], None], *, enabled: bool = True) -> None:
        self._write = write
        #: When disabled the object still exists and still accepts state — it
        #: simply never writes. A null object rather than an ``Optional``
        #: everywhere is what keeps the call sites free of feature checks.
        self._enabled = enabled
        self._label = ""
        self._state: TitleState = "idle"
        self._frame = 0
        #: The last string actually written; ``None`` means nothing has been.
        self._written: str | None = None
        #: Whether the terminal's own title was saved and is owed a restore.
        self._pushed = False

    @property
    def enabled(self) -> bool:
        """Whether this instance writes anything at all."""
        return self._enabled

    @property
    def current(self) -> str:
        """The title as it would be rendered right now (tests, diagnostics)."""
        return build_title(self._label, self._state, self._frame)

    def start(self) -> None:
        """Save the terminal's existing title, then paint ours.

        Idempotent, so a second call (a resumed app, a test) cannot stack two
        saves onto the terminal's title stack and leave one unbalanced.
        """
        if not self._enabled or self._pushed:
            return
        self._write(PUSH_TITLE)
        self._pushed = True
        self.emit()

    def set_label(self, label: str) -> None:
        """Name the session (``""`` falls back to the bare branded state)."""
        cleaned = sanitize_label(label)
        if cleaned == self._label:
            return
        self._label = cleaned
        self.emit()

    def set_state(self, state: TitleState) -> None:
        """Move to ``idle``/``working``/``attention``.

        Leaving ``working`` resets the local frame index, but the next title
        paint is still brought back into phase with the status band's current
        spinner frame by :meth:`set_frame`. That shared phase is the real
        invariant: in a tiled terminal the band and the tab title can be on
        screen together, and two working indicators stepping differently read
        as two different jobs.
        """
        if state == self._state:
            return
        self._state = state
        if state != "working":
            self._frame = 0
        self.emit()

    def set_frame(self, frame: int) -> None:
        """Advance the working animation (ignored outside ``working``).

        Takes the band's own frame index, so the two spinners step together
        instead of drifting apart on two independent clocks.

        The "ignored outside working" part is load-bearing. The status band
        keeps syncing its spinner index after a turn ends, and storing those
        idle indices here let the NEXT turn's first `set_state("working")`
        emit a stale frame before the band had a chance to push frame 0. That
        produced a visible one-tick flash in the title (`⣻` then `⣾`) on a
        fresh turn — exactly the broken restart the band reset exists to avoid.
        """
        if self._state != "working":
            return
        frame %= len(SPINNER_FRAMES)
        if frame == self._frame:
            return
        self._frame = frame
        self.emit()

    def emit(self) -> None:
        """Write the current title unless it is already on screen.

        Silent until :meth:`start` has saved the terminal's own title. That
        ordering is the whole value of the save: a setter called before
        ``start`` — the band is populated with the cwd before the app attaches
        a writer — would paint OUR title first, and the "original" the
        terminal then saved would be it. Exit would restore this session's
        name into the user's shell, which is the failure ``stop`` exists to
        prevent.
        """
        if not self._enabled or not self._pushed:
            return
        title = self.current
        if title == self._written:
            return
        self._written = title
        self._write(osc_title(title))

    def stop(self) -> None:
        """Give the terminal its title back (idempotent).

        Called on unmount. A session that exits leaving ``lo › …`` in the tab
        of a shell the user keeps typing in is a worse outcome than never
        having set it, so the restore is paired with :meth:`start` by the
        ``_pushed`` flag rather than by trusting the caller's symmetry.
        """
        if not self._pushed:
            return
        self._pushed = False
        self._written = None
        self._write(POP_TITLE)
