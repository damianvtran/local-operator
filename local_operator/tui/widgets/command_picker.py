"""Slash-command picker — the suggestion list under the input editor.

Typing ``/`` opens a list of commands; the soft (fuzzy) matching and its
score tiers are NOT reimplemented here. They live in
:mod:`local_operator.tui.autocomplete` and this widget calls
:func:`~local_operator.tui.autocomplete.match_commands` verbatim, because the
ranking is the part users build muscle memory on: a second scoring function
would drift from the one Tab/Enter already apply, and the picker would then
highlight a different command than the one that gets run.

Layout — a borderless two-column list, one row per suggestion (D4):

    ❯ /model  /models    Show or switch model (provider/id)
      /mcp               List MCP servers (login/logout/reauth <name> to manage OAuth)

* The 2-cell selection gutter lines up with ``#prompt-chevron``, so the
  highlighted ``❯`` sits directly under the prompt's own ``❯`` and every
  command name starts in the same column as the editor's text.
* The primary column fits its content, clamped to 12..32 cells, then two
  cells of gap, then the description fills what is left.
* Under 41 cells the description is dropped entirely — a description squeezed
  into a handful of cells is noise, and the command name is the part the user
  is actually choosing between.

The SAME widget also presents a command's ARGUMENT (``/login <provider>``) in
:attr:`PickerMode.ARGUMENT`: bare names instead of ``/name``, and a
right-aligned ``detail`` column carrying the state the user is choosing by. A
second widget was the alternative, which is how a codebase ends up with two
lists that look almost the same and behave almost the same.

ONE ROW PER SUGGESTION is enforced structurally, not hopefully: every row is
padded/truncated to EXACTLY the render width (so Textual has nothing to wrap)
and the widget's height is pinned to the row count on every repaint. Textual's
``Content.from_rich_text`` discards Rich's ``no_wrap``/``overflow`` flags when
a ``Text`` crosses into a widget — see ``tool_card._row_text`` — so those flags
cannot be relied on and the pinned height is what actually holds the contract.
Widths are measured with ``rich.cells.cell_len`` only, so CJK and emoji
descriptions account for their real cell cost instead of their code-point
count.
"""

from __future__ import annotations

from enum import Enum
from typing import Callable, NamedTuple, Sequence

from rich.cells import cell_len
from rich.style import Style
from rich.text import Text
from textual import events
from textual.dom import NoScreen
from textual.message import Message
from textual.widgets import Static

from local_operator.tui import theme as theme_mod
from local_operator.tui.autocomplete import (
    ArgumentChoice,
    SlashCommand,
    match_choices,
    match_commands,
)

# The project has exactly ONE cell-accurate truncator and it lives in
# tool_card. Importing it across widget modules is deliberate: a local copy is
# how the len()/cell_len split that this helper exists to prevent gets
# reintroduced, one module at a time.
from local_operator.tui.widgets.tool_card import truncate_cells

#: Row budget. Eight rows matches the editor's own ``max-height: 8``, so a full
#: picker never towers over the field that opened it, and it still shows more
#: than half of the command registry at once.
MAX_VISIBLE_ROWS = 8

#: On a short terminal the budget shrinks to a third of the screen, so the
#: picker can never squeeze the transcript to nothing. At the standard 24 rows
#: this resolves to exactly ``MAX_VISIBLE_ROWS``; at 10 rows it resolves to 3,
#: which is the floor omp clamps its own picker to.
_SCREEN_HEIGHT_DIVISOR = 3

#: An ARGUMENT list gets its own, larger budget: half the screen, less the rows
#: that are never the list's to take.
#:
#: ``MAX_VISIBLE_ROWS`` is reasoned from the COMMAND list, where every row is a
#: described one-liner the user READS. An argument list is a set they SCAN, and
#: `/login` is the one surface whose entire job is answering "what is
#: supported" — capping the twelve providers at eight hid a third of the answer
#: (openrouter, which this app's catalogue is built around, among it) while
#: seven rows of the region above the list sat empty. The splash degrades to
#: make the room, which is the right trade for as long as the list is open.
#:
#: The three subtracted rows are the ones BELOW the list inside the screen box
#: (``Screen.size`` already excludes the app's one-cell edge padding): the
#: prompt row the picker hangs off, the status band, and the blank line between
#: them. At the 28-row default this resolves to 10 of the 12 providers, and at
#: 20 rows to 6.
_ARGUMENT_HEIGHT_DIVISOR = 2
_ARGUMENT_CHROME_ROWS = 3

#: Floor for the argument budget on a short terminal, matching the floor the
#: command list clamps to.
_ARGUMENT_ROWS_MIN = 3

#: The selection mark. The app's prompt and user blocks already speak ``❯``
#: (SPINE_INDENT is 2 cells for exactly this reason); the picker reuses that
#: vocabulary rather than introducing a second cursor glyph.
_CURSOR = "❯"

#: Gutter width. THREE, not two: the prompt occupies ``❯`` plus a space and the
#: editor's own text starts in the third cell, so a two-cell gutter left every
#: suggestion one cell to the LEFT of the text it completes into — while the
#: tcss beside it claimed the two columns agreed. On the boot card, where the
#: prompt rail is the only structure on screen, that one cell is the whole
#: composition. The cursor still lands in the gutter's first cell, directly
#: under the prompt chevron.
_GUTTER_CELLS = 3

#: Primary column: fit-to-content, clamped. Below 12 the names of short
#: commands stop forming a column at all; above 32 a single long name pushes
#: every description off the row.
_PRIMARY_COLUMN_MIN = 12
_PRIMARY_COLUMN_MAX = 32
_PRIMARY_COLUMN_GAP = 2

#: A description narrower than this is dropped rather than shown as three
#: characters and an ellipsis.
_MIN_DESCRIPTION_CELLS = 10

#: At or below this width the row collapses to the command name only.
DESCRIPTION_COLLAPSE_WIDTH = 40

#: Right-edge breathing room, so no row ever paints into the last cell.
_EDGE_MARGIN = 2

#: Width assumed for the one repaint that can happen before layout has
#: measured the widget. Height is pinned to the ROW COUNT, which is
#: width-independent, so the worst case is a single frame of narrow rows that
#: the following Resize corrects — never a list that silently doubles height.
_MIN_RENDER_WIDTH = 20

#: FLOOR for the NAME column of an argument row before its ``detail`` is
#: dropped — the minimum, not the answer: see :meth:`CommandPicker._name_floor`,
#: which raises it to the widest id actually offered. The name is the text Tab
#: types into the buffer, so a truncated one is unusable — the user cannot read
#: what to complete to. ``detail`` is worth a lot (at `/logout` it names the
#: credential being removed) but never worth that.
_MIN_NAME_CELLS = _PRIMARY_COLUMN_MIN


class PickerMode(Enum):
    """Which kind of list the picker is currently showing.

    Read by the editor, which has to know whether Tab is completing a command
    WORD (rewrite everything from the slash, add a trailing space) or a command's
    ARGUMENT (replace the tail, no trailing space — the space would terminate the
    argument and close the very list Tab just used).
    """

    COMMAND = "command"
    ARGUMENT = "argument"
    #: The ``$skill`` manual-invocation list. A third mode rather than a reuse
    #: of ``COMMAND`` because the two complete differently: a command word is
    #: inline and caret-anchored, while a ``$`` token is only ever the FIRST
    #: token of the buffer, and Enter on a skill row must not run anything —
    #: an invocation produces a PROMPT the user still has to write.
    SKILL = "skill"


#: One rendered row: its display name and the thing it stands for. A UNION
#: rather than the :class:`~local_operator.tui.autocomplete.Completable`
#: protocol the matcher is generic over, because rendering needs the concrete
#: fields (a command's aliases, a choice's ``detail``) and dispatching on the
#: item's type makes "argument rows carry a detail column" impossible to get
#: wrong — there is no mode flag to fall out of step with the payload.
_Suggestion = tuple[str, "SlashCommand | ArgumentChoice"]


class _RowStyles(NamedTuple):
    """The styles one row paints with, resolved once from its selection state."""

    selected: bool
    ground: Style
    name: Style
    alias: Style
    description: Style
    cursor: Style


class SlashContext(NamedTuple):
    """Where the active command word sits, and the word typed so far.

    ``start`` indexes the ``/`` itself and ``end`` the first cell past the word,
    so a completion can rebuild JUST that span and leave the rest of the draft
    untouched. Before inline detection the word always ran to the end of the
    buffer, so a completion could splice from ``start`` to the end; now the word
    can have a message typed after it (``fix this /team``, or ``/team\\nfix
    this``), and only ``[start, end)`` is the command — everything outside it is
    the user's prose and must survive the completion verbatim.
    """

    start: int
    query: str
    end: int


#: A slash opens a command only at a WORD BOUNDARY: the buffer start, or right
#: after whitespace. This is what keeps ``src/foo`` and ``and/or`` from opening
#: the picker — the ``/`` there is glued to a preceding non-space character, so
#: it is punctuation inside a word, not the start of a command. The rule is the
#: same one a shell or an editor command palette uses to tell a path apart from
#: a command, and it is the ONE thing that makes inline detection safe to run on
#: every keystroke of ordinary prose.
def _is_boundary(line: str, index: int) -> bool:
    """Whether ``line[index]`` (a ``/``) begins a fresh token."""
    return index == 0 or line[index - 1].isspace()


def _line_of_cursor(text: str, cursor: int | None) -> tuple[str, int, int]:
    """The line the cursor sits on, as ``(line, line_start_offset, column)``.

    ``cursor`` is a whole-buffer offset; ``None`` means "the end of the buffer",
    which is where a user typing at the end of their draft is. Clamped into
    range so a stale caret (a resync racing a delete) cannot index out of the
    text. Offsets are measured the same way ``Editor._offset_at`` does, so a
    caret computed there indexes correctly here. Lines are split on ``\\n`` and
    a trailing ``\\r`` from a CRLF buffer is stripped below, so every consumer of
    the returned line agrees on the word regardless of the paste's line endings.
    """
    if cursor is None or cursor > len(text):
        cursor = len(text)
    if cursor < 0:
        cursor = 0
    line_start = text.rfind("\n", 0, cursor) + 1  # 0 when no newline precedes
    line_end = text.find("\n", cursor)
    if line_end == -1:
        line_end = len(text)
    line = text[line_start:line_end]
    # A CRLF buffer — which a draft/held-prompt restore can carry in, since
    # ``load_text`` does not normalise — leaves a trailing ``\r`` on every line
    # split on ``\n``. ``slash_context`` tolerated it (``\r`` is whitespace, so
    # it terminated the word), but ``slash_word``/``slash_argument_context`` use
    # ``.partition(" ")``, which does NOT treat ``\r`` as a separator: the word
    # came back as ``"team\r"`` (matching no command) and argument values
    # carried the ``\r`` into dispatch. Stripping it here fixes every consumer at
    # once, because they all read the line through this one helper. The column is
    # unaffected: the ``\r`` sits at the END of the line, past any caret the
    # parsers care about (review round 1, minor-1).
    if line.endswith("\r"):
        line = line[:-1]
    return line, line_start, cursor - line_start


def _boundary_slashes(line: str) -> list[int]:
    """Indices of every boundary ``/`` on ``line`` (buffer start / after space)."""
    return [i for i, ch in enumerate(line) if ch == "/" and _is_boundary(line, i)]


def _active_slash(line: str, column: int, commands: frozenset[str] = frozenset()) -> int | None:
    """Index within ``line`` of the boundary ``/`` the cursor is editing.

    Normally the active token is the LAST boundary ``/`` at or before the cursor:
    a user typing ``a /foo /ba|`` (caret at ``|``) is editing ``/ba``, not
    ``/foo``. Returns ``None`` when the cursor is not inside any boundary slash
    token — before every slash on the line, or on a ``/`` glued to a word.

    ``commands`` closes the "already inside a command" case: once the line holds
    a RECOGNISED command that has been TERMINATED by a space (``/team security
    …``), that command owns the rest of its line as its argument, so a second
    ``/team`` typed INSIDE the request ("improve the /team command") is plain
    argument text — not a new command to highlight or re-open the picker for.
    The earliest such command on the line wins and claims everything after it;
    only when no earlier recognised command has claimed the caret's position does
    the last-slash-before-caret rule apply. Empty ``commands`` (the pure-parser
    default) disables claiming and keeps the simple behaviour.
    """
    slashes = _boundary_slashes(line)
    candidate: int | None = None
    for index in slashes:
        word, sep, _ = line[index + 1 :].partition(" ")
        if sep and commands and word.lower() in commands:
            # A recognised, terminated command. Its argument runs to the end of
            # the line, so if the caret is anywhere past this slash it is inside
            # THIS command — return it and stop, ignoring every later slash.
            if column > index:
                return index
            # Caret is before this command entirely; nothing earlier can claim.
            return candidate
        if index <= column:
            candidate = index
    return candidate


def slash_context(
    text: str, cursor: int | None = None, commands: frozenset[str] = frozenset()
) -> SlashContext | None:
    """The command word being typed at the cursor, or ``None`` to hide the list.

    Inline and caret-aware, unlike the original first-line-only rule: the picker
    shows whenever the caret is inside a boundary-slash token whose word is not
    yet terminated by whitespace, WHEREVER that token is in the draft. This is
    what lets a user who has typed a message remember to route it — appending
    ``/team`` at the caret, or dropping it on its own line — and still get the
    menu. ``src/foo`` never triggers because its ``/`` is not at a boundary.

    Command-WORD phase only: the instant a space terminates the word the user is
    typing an ARGUMENT (``/model gpt``), which :func:`slash_argument` owns — a
    command list there is stale advice. ``cursor`` defaults to the end of the
    buffer, the common "typing at the end" case.

    ``commands`` is the recognised command vocabulary; passing it makes a slash
    typed INSIDE an already-engaged command's argument (``/team a /team b``) read
    as plain text rather than a nested command (see :func:`_active_slash`).
    """
    line, line_start, column = _line_of_cursor(text, cursor)
    slash = _active_slash(line, column, commands)
    if slash is None:
        return None
    # The word runs from the slash to the first whitespace after it. A space
    # BETWEEN the slash and the caret means the word is already terminated and
    # the caret is out in argument (or message) territory — not the command.
    word_end = slash + 1
    while word_end < len(line) and not line[word_end].isspace():
        word_end += 1
    if column > word_end:
        return None
    return SlashContext(
        line_start + slash,
        line[slash + 1 : word_end],
        line_start + word_end,
    )


def slash_token_span(
    text: str, cursor: int | None = None, commands: frozenset[str] = frozenset()
) -> tuple[int, int] | None:
    """The ``[start, end)`` offsets of the whole slash TOKEN at the caret.

    ``start`` is the ``/`` and ``end`` is the end of its line — the command word
    plus its inline argument, which by the inline contract runs to the line end.
    Returns ``None`` when the caret is not on a boundary-slash token. This is the
    span a RUN splices out of an inline draft, exposed so the editor never has to
    reach for the module-private tokenizer. ``commands`` is the recognised
    vocabulary, so a nested slash inside an engaged command's argument is ignored.
    """
    line, line_start, column = _line_of_cursor(text, cursor)
    slash = _active_slash(line, column, commands)
    if slash is None:
        return None
    return line_start + slash, line_start + len(line)


def slash_word(
    text: str, cursor: int | None = None, commands: frozenset[str] = frozenset()
) -> str | None:
    """The lower-cased command word of the slash token AT THE CARET, or ``None``.

    The word regardless of PHASE — whether it is still being typed or already
    terminated by a space — so a caller that needs to know "which command is the
    caret on" (the editor deciding which argument list to fill) gets one answer
    whether the buffer reads ``/team`` or ``/team ops``. :func:`slash_context`
    and :func:`slash_argument` answer the narrower phase-specific questions.
    ``commands`` is the recognised vocabulary for the nested-slash rule.
    """
    line, _, column = _line_of_cursor(text, cursor)
    slash = _active_slash(line, column, commands)
    if slash is None:
        return None
    return line[slash + 1 :].partition(" ")[0].lower()


class SlashArgument(NamedTuple):
    """The argument being typed, and the spans it occupies in the buffer.

    ``start`` and ``end`` are whole-buffer offsets bracketing the argument text,
    so a completion can replace JUST the argument and leave a trailing inline
    message intact — the argument-phase twin of :class:`SlashContext`.
    ``token_start`` indexes the ``/`` that opens the whole construct, so a RUN
    can splice the entire ``/cmd arg`` out of an inline draft (not just its
    argument). ``value`` is the argument text (possibly ``""``).
    """

    value: str
    start: int
    end: int
    token_start: int


def slash_argument_context(
    text: str,
    commands: tuple[str, ...],
    cursor: int | None = None,
    known: frozenset[str] = frozenset(),
) -> SlashArgument | None:
    """The ARGUMENT being typed after one of ``commands``, with its span.

    The mirror image of :func:`slash_context`: that one is live while the command
    word is still open, this one takes over the instant the word is terminated by
    a space. Together they mean a single buffer drives two different lists without
    either having to know about the other — ``/mo`` offers commands, ``/model ``
    offers models, and the handover happens on the space the user was going to
    type anyway.

    Caret-aware and inline like :func:`slash_context`: the argument is the text
    from the word-terminating space to the END OF THE LINE the command is on.
    End-of-line, not end-of-buffer, is what makes a command droppable on its own
    line above a multi-line draft — the line below is the message, not the
    argument. On the command's own line the argument runs to the line end, so a
    trailing message on the SAME line (``/team ops ship it``) is read as part of
    the argument; putting the command last, or on its own line, is how a user
    keeps the two apart (documented as the inline contract).

    ``None`` when the caret is not in the argument phase of one of ``commands``.
    ``known`` is the FULL recognised vocabulary (a superset of ``commands``) used
    only for the nested-slash rule: a slash inside an engaged command's argument
    is not a new token. It defaults to ``commands`` when omitted.
    """
    line, line_start, column = _line_of_cursor(text, cursor)
    slash = _active_slash(line, column, known or frozenset(commands))
    if slash is None:
        return None
    word, sep, argument = line[slash + 1 :].partition(" ")
    if not sep or word.lower() not in commands:
        return None
    # The caret must be in the ARGUMENT, i.e. past the terminating space; while
    # it is still on the word itself that is command-word phase, which
    # ``slash_context`` owns. ``slash + 1 + len(word)`` is the space's column.
    space_column = slash + 1 + len(word)
    if column <= space_column:
        return None
    arg_start = line_start + space_column + 1
    return SlashArgument(argument, arg_start, line_start + len(line), line_start + slash)


def slash_argument(
    text: str,
    commands: tuple[str, ...],
    cursor: int | None = None,
    known: frozenset[str] = frozenset(),
) -> str | None:
    """The argument text being typed after one of ``commands``, else ``None``.

    A thin projection of :func:`slash_argument_context` onto just the text, for
    the many callers that only rank against the argument and never need to
    rewrite the span. Returns ``""`` when the command word is complete but
    nothing has been typed after it (the whole-catalogue state). ``known`` is the
    full vocabulary for the nested-slash rule.
    """
    context = slash_argument_context(text, commands, cursor, known)
    return None if context is None else context.value


def skill_token(text: str, cursor: int | None = None) -> SlashContext | None:
    """The ``$skill`` token being typed at the start of the buffer, or ``None``.

    The ``$`` counterpart to :func:`slash_context`, and deliberately STRICTER
    than it in two ways.

    It is **not inline**. A ``/`` opens a command anywhere a boundary allows,
    because a user who has typed a message may still want to route it. A ``$``
    is only an invocation as the buffer's FIRST token: mid-draft, ``$`` is
    overwhelmingly money or a shell variable, and there is no second syntax to
    disambiguate them. Restricting the position is what removes the need for an
    escape rule — ``a $5 coffee`` cannot be a token by construction.

    It is **word-phase only**, like :func:`slash_context`: the terminating
    space means the user has moved on to the request, where a skill list is
    stale advice.

    ``query`` is ``""`` for a bare ``$``, which opens the list on the full set.
    The caret must be INSIDE the token; moving it out into the request closes
    the list, which is what makes the picker phase a property of the parse.
    """
    if not text.startswith("$"):
        return None
    cursor = len(text) if cursor is None else cursor
    end = 1
    while end < len(text) and not text[end].isspace():
        end += 1
    if cursor > end:
        return None
    return SlashContext(0, text[1:end], end)


class CompletionMode(Enum):
    """Which slot :func:`completion_for` is completing.

    The four completion sites in ``Editor`` differ only in which span they
    rewrite and whether a trailing space follows the inserted name, and that
    difference is exactly what this enum names.
    """

    #: The command WORD — ``/mc`` → ``/mcp `` (``Editor._apply_command``).
    COMMAND = "command"
    #: An enum-tail ARGUMENT — ``/mcp lo`` → ``/mcp login``, no trailing space
    #: so the matcher keeps matching (``Editor._complete_argument``).
    ARGUMENT = "argument"
    #: A NAME+message argument — ``/team fro`` → ``/team frontend-guild ``, the
    #: space opening the message tail (``Editor._complete_name_argument``).
    NAME_ARGUMENT = "name_argument"
    #: A ``$skill`` invocation — ``$res`` → ``$research ``. Takes the trailing
    #: space for the same reason ``NAME_ARGUMENT`` does: the space closes the
    #: list and opens the request tail the user is about to type.
    SKILL = "skill"


def completion_for(
    text: str,
    caret: int,
    mode: CompletionMode,
    row_name: str,
    commands: tuple[str, ...],
    known: frozenset[str] = frozenset(),
) -> tuple[str, int] | None:
    """The buffer and caret that accepting ``row_name`` produces — pure.

    THE single source of truth for "what does choosing this row put in the
    buffer". The three ``Editor`` completion methods delegate their string
    arithmetic here and keep only their side effects (running the command, the
    inline reassembly), and the composer's inline GHOST TEXT is derived from
    the very same call. That shared derivation is what makes the ghost's
    invariant true by construction rather than by two implementations agreeing:
    the dimmed cells the user sees are computed from the same ``new_text`` that
    Tab commits, so ``buffer + ghost == buffer_after_tab`` cannot drift the way
    two parallel string builders would.

    Every mode replaces a SPAN — ``[start, end)`` — rather than splicing to the
    end of the buffer, because inline detection means a message may follow the
    command token (``fix this /te|``) and that suffix is the user's prose. Note
    the consequence for the ghost: a span replacement is only an APPEND when
    the row extends what was typed, which is why the ghost rule tests
    ``new_text.startswith(text)`` instead of assuming it.

    Returns ``None`` when the caret is not in the slot ``mode`` names — the
    parse the caller would otherwise have had to repeat.
    """
    if mode is CompletionMode.SKILL:
        token = skill_token(text, caret)
        if token is None:
            return None
        # Same trailing-space contract as the command word: it terminates the
        # token, closes the list, and opens the request. The suffix beyond the
        # token is preserved because a user can complete a `$skill` typed in
        # front of a request they already wrote.
        completed = f"${row_name} {text[token.end :].lstrip()}"
        return completed, len(row_name) + 2
    if mode is CompletionMode.COMMAND:
        context = slash_context(text, caret, known)
        if context is None:
            return None
        # The trailing space is load-bearing, not cosmetic: it terminates the
        # word (closing the command list) and, for a list-taking command, opens
        # the argument list. It is part of what Tab commits, so it is part of
        # the ghost too.
        completed = f"{text[: context.start]}/{row_name} {text[context.end :]}"
        return completed, context.start + len(row_name) + 2
    argument = slash_argument_context(text, commands, caret, known)
    if argument is None:
        return None
    # NAME+message commands take the terminating space (it opens the message
    # tail); enum-tail arguments must NOT, or the matcher would stop matching
    # and Tab would appear to fill the field and abandon it in one keystroke.
    suffix = " " if mode is CompletionMode.NAME_ARGUMENT else ""
    filled = f"{text[: argument.start]}{row_name}{suffix}{text[argument.end :]}"
    caret_after = argument.start + len(row_name) + len(suffix)
    if mode is CompletionMode.NAME_ARGUMENT:
        # INLINE ENGAGE. When a draft survives outside the command token,
        # ``Editor._complete_name_argument`` does NOT stop at the span
        # replacement above: it hands off to ``_reassemble_prompt_command``,
        # which moves the whole ``/<cmd> <name>`` construct to the FRONT of the
        # buffer with the surviving draft as its message. That second edit has
        # to be modelled HERE, in the one function both Tab and the ghost read,
        # or the ghost describes an edit that never happens — it previewed a
        # dimmed append while Tab reordered the entire buffer (review round 1,
        # B1). Modelling it rather than special-casing the renderer is what
        # keeps "one function, one answer" true; a renderer-side exception
        # would reintroduce exactly the two-implementations drift this shared
        # helper exists to prevent.
        #
        # The result is deliberately NOT an append, so ``ghost_for``'s
        # ``startswith`` rule withholds the ghost of its own accord. That is
        # the honest outcome: no string appended at the caret can describe a
        # whole-buffer reordering, the same reason a fuzzy match shows nothing.
        outside = (text[: argument.token_start] + text[argument.end :]).strip()
        if outside:
            return _reassembled_completion(filled, caret_after, known)
    return filled, caret_after


def _reassembled_completion(
    filled: str, caret: int, known: frozenset[str]
) -> tuple[str, int] | None:
    """Apply the inline reassembly to an already-filled NAME_ARGUMENT buffer.

    Mirrors ``Editor._reassemble_prompt_command`` on the FILLED text, which is
    the state that method actually runs against (the name is in place before
    the token span is recomputed). Kept beside :func:`completion_for` so the
    prediction and the edit are read together and cannot drift apart.
    """
    span = slash_token_span(filled, caret, known)
    if span is None:
        return None
    token_start, token_end = span
    command = filled[token_start:token_end].strip()
    # One adjoining separator goes with the token, matching the splice rule so
    # ``msg /goal`` and ``/goal\nmsg`` both collapse to just ``msg``. The
    # PRECEDING separator is preferred; the following one is taken only when
    # the token opened the buffer.
    start, end = token_start, token_end
    if start > 0 and filled[start - 1] in " \t\n":
        start -= 1
    elif end < len(filled) and filled[end] in " \t\n":
        end += 1
    rest = (filled[:start] + filled[end:]).strip()
    assembled = f"{command} {rest}" if rest else f"{command} "
    return assembled, len(assembled)


def ghost_for(completion: tuple[str, int] | None, text: str) -> str:
    """The dimmed remainder to preview, or ``""`` when none can be honest.

    The ghost is shown IF AND ONLY IF the completion is a pure APPEND to what
    the user typed. Completion rewrites a SPAN (see :func:`completion_for`), so
    for a prefix match that happens to look like an append but for a FUZZY one
    it rewrites characters already on screen: ``/lg`` + Tab yields ``/login ``,
    which is not ``/lg`` plus anything. No string appended at the caret can
    describe that edit, so any ghost there would display characters Tab does
    not produce — a lie about the next keystroke. The picker row already
    carries the meaning in that case, so showing nothing costs nothing.

    ``startswith`` is deliberately CASE-SENSITIVE. ``/MCP lo`` + Tab inserts
    the registry's own casing (``login``), so a case-insensitive check would
    pass and then paint a ghost whose visible characters differ from the ones
    Tab commits. Same rule, same reason: only an exact append is honest.
    """
    if completion is None:
        return ""
    new_text, _caret = completion
    return new_text[len(text) :] if new_text.startswith(text) else ""


#: Below this many typed characters the fuzzy tail is suppressed. A one- or
#: two-letter query matches an arbitrary-looking set by subsequence — `/u`
#: offered `usage, quit, accounts, logout` and `/g` offered
#: `goal, usage, login, logout`. The correct command ranked first every time,
#: but rows 2+ taught the user that the list is unreliable, which is the
#: fastest way to make them stop reading it. Typo tolerance lives at three
#: characters and up (`/cmpct`, `/lgout`), so nothing the feature exists for
#: is affected.
FUZZY_MIN_QUERY_CHARS = 3


def command_suggestions(query: str, commands: list[SlashCommand]) -> list[tuple[str, SlashCommand]]:
    """``(display_name, command)`` suggestions for a typed command word.

    A bare ``/`` cannot go through :func:`match_commands`:
    ``score_command_text_match`` scores an empty prefix at 0 by contract (it is
    what makes "no match" and "nothing typed" distinguishable for the completion
    path, and it is pinned by test), so asking it would answer "no commands" for
    the keystroke whose entire purpose is "show me the commands". The full
    registry, in registration order, IS the answer to ``/``.

    Short queries PREFER prefix matches — see :data:`FUZZY_MIN_QUERY_CHARS` — but
    only when there are some. An empty return closes the picker, and a closed
    picker takes the Tab and Enter guards down with it: Tab falls through to
    stock TextArea behaviour and indents the user's message, Enter submits the
    raw text to the agent. So the gate is a preference, not a filter. It has to
    be, because the queries with no prefix match at all are exactly the natural
    abbreviations the fuzzy matcher exists for — `/lg` for login and logout,
    `/ls` for models and skills, `/qt` for quit, `/md` for model.
    """
    if not query:
        return [(command.name, command) for command in commands]
    matches = match_commands(f"/{query}", commands)
    if len(query) >= FUZZY_MIN_QUERY_CHARS:
        return matches
    lowered = query.lower()
    prefixed = [pair for pair in matches if pair[0].lower().startswith(lowered)]
    return prefixed or matches


def argument_suggestions(
    query: str, choices: list[ArgumentChoice]
) -> list[tuple[str, ArgumentChoice]]:
    """``(display_name, choice)`` suggestions for a command's typed ARGUMENT.

    The same shape and the same short-query preference as
    :func:`command_suggestions`, deliberately: the two lists appear in the same
    place, are driven by the same keys and are ranked by the same scorer, so a
    user cannot tell which one they are in — and does not need to.

    The one behavioural difference is that an argument list may legitimately be
    EMPTY when the set itself is empty (``/logout`` with nothing stored), which
    is a real answer rather than a failed match. The caller distinguishes those
    two cases, because only one of them is worth saying out loud.
    """
    if not query:
        return [(choice.name, choice) for choice in choices]
    matches = match_choices(query, choices)
    if len(query) >= FUZZY_MIN_QUERY_CHARS:
        return matches
    lowered = query.lower()
    prefixed = [pair for pair in matches if pair[0].lower().startswith(lowered)]
    return prefixed or matches


def _pad_to(row: Text, width: int, style: Style) -> Text:
    """Pad ``row`` out to exactly ``width`` cells under ``style``.

    Exact width is what makes the one-row rule structural: a row that already
    fills the render width leaves Textual nothing to wrap, and a row-wide
    background tint needs the full span or the highlight reads as a ragged
    smear instead of a selected row.
    """
    missing = width - cell_len(row.plain)
    if missing > 0:
        row.append(" " * missing, style=style)
    return row


class CommandPicker(Static):
    """The suggestion list for the slash command being typed.

    Driven entirely from outside for the keyboard (the editor keeps focus and
    routes Up/Down/Tab/Enter/Esc in, so the caret never leaves the text) and
    from its own mouse handlers for click/hover. Choosing a row calls the
    ``on_choose`` callback with the display name to insert; the picker itself
    never touches the buffer.
    """

    class RowsResized(Message):
        """The picker's pinned height changed, so the dock around it has moved.

        The picker is a child of ``#input-dock``, and the boot composition
        centres the splash against the dock's measured height
        (``_sync_boot_composition``). That measurement is deliberately NOT a
        self-rescheduling read of a laid-out frame — it is arithmetic run at
        known moments — so when this widget's row count changes the app has no
        other way to learn the dock got taller or shorter.

        It went wrong exactly where the two disagreed (R7-3): a delayed session
        adoption composed the dock while the picker still held its ONE-row
        loading reserve, then replaced it with a two-row roster that nobody
        re-measured. The dock kept the lift computed for the shorter list and
        floated four rows above the bottom, with a visible gap under the status
        band, while the identical non-delayed screen sat flush. Same class of
        problem, and same remedy, as ``WelcomeView.BlockResized``.

        Posted only when the height actually CHANGES, so an unchanged repaint
        (the common case — every keystroke repaints) costs nothing.
        """

    def __init__(
        self,
        on_choose: Callable[[str], None],
        on_highlight: Callable[[str | None], None] | None = None,
        on_preview: Callable[[str | None], None] | None = None,
    ) -> None:
        super().__init__()
        self._on_choose = on_choose
        #: Observer for the row an ACCEPT KEY would take — a SEPARATE question
        #: from ``on_highlight``, and the reason this is its own channel rather
        #: than another caller of that one. ``on_highlight`` answers "what is
        #: the eye on", so it prefers the HOVER and only reports ARGUMENT rows;
        #: both are right for a row preview and wrong for the composer's inline
        #: ghost, which is a prediction about what Tab will insert. Reusing the
        #: highlight channel left the ghost a row behind on every arrow press in
        #: COMMAND mode (nothing reported there at all) and repainted it to the
        #: hovered row while Tab still acted on the keyboard selection (review
        #: round 1, U1/U2). Fired from the same state-change sites, so any way
        #: the accept target moves reaches it.
        self._on_preview = on_preview
        #: Last name reported to ``_on_preview``, de-duplicated for the same
        #: reason as ``_reported_highlight``.
        self._reported_preview: str | None = None
        #: Rows currently pinned by ``_pin_height``. Tracked here rather than
        #: read back off ``styles.height`` so the change test is a plain int
        #: comparison and cannot be confused by a Scalar's units (R7-3).
        self._pinned_rows: int | None = None
        #: Observer for the row the user is CONSIDERING — the hover target when
        #: the mouse is over a row, else the keyboard highlight — called with
        #: ``None`` when an argument list stops showing rows. It exists for
        #: live preview (``/theme``): the preview has to track what the eye is
        #: on, which is not always what Enter would choose. Only ARGUMENT rows
        #: report; a command-word list has nothing to preview.
        self._on_highlight = on_highlight
        #: Last name reported to ``_on_highlight``, so the observer hears each
        #: change once — mouse-move events arrive per cell, not per row.
        self._reported_highlight: str | None = None
        #: True only while ``set_choices`` seeds a highlight: the interim
        #: row-0 state must not reach the observer (see ``set_choices``).
        self._suppress_report = False
        self._commands: list[SlashCommand] = []
        self._command_names: frozenset[str] = frozenset()
        self._choices: list[ArgumentChoice] = []
        self._mode = PickerMode.COMMAND
        self._matches: list[_Suggestion] = []
        self._selected = 0
        self._window_start = 0
        self._hovered: int | None = None
        self._query = ""
        self._dismissed_query: str | None = None
        # Set when an ARGUMENT list has nothing to offer AND that is worth saying.
        # Not a match: it is never selectable, so it lives beside the rows rather
        # than among them (see set_notice).
        self._notice = ""
        # True only while the one notice row is a TRANSIENT loading reserve —
        # rows are expected to replace it (see set_loading_reserve). Kept as explicit
        # state rather than inferred from the notice text, because the editor
        # must gate Tab/Enter on “this row will be replaced” — a notice like
        # `/logout`'s “no stored credentials” is a real answer that must keep
        # normal keys (U2-2), and a text match would couple key routing to
        # copy (U2-1's fix originally tried exactly that and drifted).
        self._loading = False
        # Set by an arrow press, cleared whenever the candidate set changes: the
        # difference between "the matcher put the highlight here" and "I moved it
        # here", which is what the editor's Enter gate needs to know.
        self._chosen_by_hand = False
        # Closed picker takes no layout space at all — `visible: hidden` would
        # still reserve the rows and leave a hole above the status band.
        self.display = False

    # -- public API ---------------------------------------------------------
    def set_commands(self, commands: list[SlashCommand]) -> None:
        """Replace the offered command registry."""
        self._commands = list(commands)
        # Full recognised vocabulary (primaries AND aliases), lower-cased, for the
        # nested-slash rule: a slash typed inside an engaged command's argument is
        # plain text, and the tokenizer needs the vocabulary to know a command has
        # claimed the rest of the line. Cached so it is not rebuilt per keystroke.
        self._command_names = frozenset(
            name.lower() for command in commands for name in command.names
        )

    def set_choices(self, choices: list[ArgumentChoice], highlight: str | None = None) -> None:
        """Replace the values offered for the current command's ARGUMENT.

        Re-derives the visible rows immediately, because the app fills these in
        answer to a posted message — one message-loop tick after the keystroke
        that opened the list. Without the resync the picker would sit closed on
        the empty set it was opened with until the user typed another character.

        ``highlight`` seeds the selection onto the named row when the list
        opens bare (empty query, nothing chosen by hand). It exists for lists
        where the highlight has a SIDE EFFECT: ``/theme`` previews the
        highlighted row live, so a list that opened on row 0 flashed every
        non-default user to the default theme before they touched a key
        (review round 1, F2). Seeding the row where the user already IS makes
        the first report a no-op — and is where a browse should start anyway.
        """
        self._choices = list(choices)
        # SKILL rides the same fill path as ARGUMENT: both are app-pushed
        # ``ArgumentChoice`` sets that land one message-loop tick after the
        # keystroke that opened the list, so both need the immediate re-derive
        # below or the picker sits closed on the empty set it opened with until
        # the user types another character. Gating this on ARGUMENT alone is
        # exactly why a bare ``$`` painted nothing.
        if self._mode in (PickerMode.ARGUMENT, PickerMode.SKILL):
            matches = argument_suggestions(self._query, self._choices)
            seeding = highlight is not None and not self._query and not self._chosen_by_hand
            if seeding:
                # Silence `_apply`'s own report: it fires for row 0 before the
                # seed lands, and for a previewing list that one report IS the
                # flash — the observer would try row 0 on and take it off
                # again one message later.
                self._suppress_report = True
            try:
                # The CURRENT mode, not a hard-coded ARGUMENT: re-applying the
                # wrong one here would be read as a mode CHANGE by `_apply` and
                # would reset the highlight, the window and the Esc latch on
                # every fill.
                self._apply(self._mode, self._query, matches)
            finally:
                self._suppress_report = False
            if seeding:
                names = [name for name, _ in self._matches]
                if highlight in names and names.index(highlight) != self._selected:
                    self._selected = names.index(highlight)
                    self._scroll_to_selection()
                    self._repaint()
                self._report_highlight()

    def set_loading_reserve(self, text: str) -> None:
        """Reserve the notice row for rows that are still ARRIVING (U2-1).

        The team/agent name lists open before the session has adopted a
        registry: without a reserved row the picker is hidden, and the first
        real row then PUSHES the dock up by one line exactly while the user
        is mid-type. This shows the same one dim row ``set_notice`` paints —
        non-selectable, never a match — but marks it TRANSIENT so the editor
        can consume Tab/Enter until real rows land (U2-2): both keys act on
        a highlighted row, and with no rows yet they would append spaces to
        the buffer or submit and discard the query the catch-up exists to
        protect.

        Passing "" clears the reserve (adoption landed, or the roster is
        authoritatively empty). The flag is dropped by ``_apply`` on a mode
        change and by ``_close`` for the same reason the notice is: it
        described a list that is no longer showing.
        """
        text = text.strip()
        if text and self._query == self._dismissed_query:
            # A queued refresh can land after Esc. The dismissal belongs to
            # this exact query, so a late loading reserve must not resurrect
            # the row the user just collapsed; adoption's real rows are
            # checked against the same latch by ``_apply`` (U2-2).
            self._loading = False
            self.set_notice("")
            return
        self._loading = bool(text)
        self.set_notice(text)

    def is_loading(self) -> bool:
        """True while the notice row is a transient loading reserve.

        Deliberately NARROWER than :meth:`is_pending`: ``is_pending`` covers
        the one message-loop tick before any fill lands and must stay
        permissive (other argument lists rely on keys passing through),
        while this marks only the app-authored “rows are coming” window
        where Tab/Enter are known to have nothing to act on.
        """
        return self._loading

    def set_notice(self, text: str) -> None:
        """Say why an ARGUMENT list is empty, IN THE LIST'S OWN PLACE.

        One dim row where the rows would have been, in the overflow marker's
        vocabulary. The alternative — reporting it into the transcript — repeats
        without bound: the message answers a UI event, so every re-entry into the
        argument state (type `/logout `, backspace, space again) appended another
        identical line to what is supposed to be a record of the conversation.
        Said here it is in the user's eye-line, self-clearing, unrepeatable, and it
        costs the transcript nothing.

        NOT a match. ``_matches`` stays empty, so ``is_open()`` is False,
        ``_index_at`` returns None for the row — a click or a hover cannot action
        it, exactly as for the overflow count — and every key the editor routes at
        an open picker still goes to the buffer. Passing ``""`` withdraws it.
        """
        text = text.strip()
        if text == self._notice:
            return
        self._notice = text
        if self._matches:
            # Rows are showing: they answer the question the notice would.
            return
        if text:
            self.display = True
            self._repaint()
        else:
            self._close()

    @property
    def mode(self) -> PickerMode:
        """Whether the rows are commands or one command's argument values."""
        return self._mode

    def is_open(self) -> bool:
        """True when suggestions are showing."""
        return bool(self._matches)

    def is_pending(self) -> bool:
        """True for an ARGUMENT list that is open in principle but has no rows yet.

        The app fills an argument list in answer to a posted message, so for one
        message-loop tick the picker is in argument mode holding nothing —
        showing as closed while being, from the user's point of view, a list they
        just opened. A key that only reaches an ``is_open()`` picker is silently
        dropped in that window.

        False once :meth:`dismiss` has recorded the query, so a dismissed list
        stops swallowing the key that dismissed it.
        """
        return (
            self._mode is PickerMode.ARGUMENT
            and not self._matches
            and self._dismissed_query is None
        )

    def suggestions(self) -> list[_Suggestion]:
        """All current matches, best first (not just the visible window)."""
        return list(self._matches)

    def highlighted_name(self) -> str | None:
        """Display name of the highlighted row, or ``None`` when closed."""
        if not self._matches:
            return None
        return self._matches[self._selected][0]

    def highlighted_choice(self) -> ArgumentChoice | None:
        """The highlighted ARGUMENT row itself, or ``None``.

        The row OBJECT, not just its name, so a caller can read the flags the
        app set on it. The safety gate needs ``alert`` — whether choosing this
        row destroys something — and that is a per-ROW fact the command word
        cannot answer: ``/mcp remove <server>`` deletes a config while
        ``/mcp login <server>`` does not, and both live under the same word
        (see ``Editor._argument_is_destructive``).

        Returns ``None`` in COMMAND mode, where the match is a
        :class:`SlashCommand` and the question does not apply.
        """
        if not self._matches or self._mode is not PickerMode.ARGUMENT:
            return None
        choice = self._matches[self._selected][1]
        return choice if isinstance(choice, ArgumentChoice) else None

    @property
    def selected_index(self) -> int:
        """Index of the highlighted row within :meth:`suggestions`."""
        return self._selected

    @property
    def chosen_by_hand(self) -> bool:
        """True when the user arrowed onto the current row themselves."""
        return self._chosen_by_hand

    @property
    def hovered_index(self) -> int | None:
        """Index under the mouse, or ``None``."""
        return self._hovered

    def visible_window(self) -> tuple[int, int, int]:
        """``(start, end, total)`` — which suggestions the rows are showing.

        Exposed because the rendered rows alone cannot tell a caller whether
        anything is hidden, and "the list is longer than it looks" is exactly
        what a capped picker must be able to answer.
        """
        total = len(self._matches)
        end = min(total, self._window_start + self._row_budget())
        return self._window_start, end, total

    def sync(self, text: str, cursor: int | None = None) -> None:
        """Re-derive the COMMAND suggestions from the editor's current ``text``.

        ``cursor`` is a whole-buffer offset locating the active slash token, so
        an inline ``/team`` typed in the middle of a draft opens the list; the
        editor passes its caret offset. ``None`` means the end of the buffer.
        """
        context = slash_context(text, cursor, self._command_names)
        if context is None:
            # Left slash context entirely: forget the dismissal, so the next
            # `/` opens a fresh picker.
            self._dismissed_query = None
            self._mode = PickerMode.COMMAND
            self._close()
            return
        matches = command_suggestions(context.query, self._commands)
        self._apply(PickerMode.COMMAND, context.query, matches)

    def sync_skills(self, text: str, cursor: int | None = None) -> None:
        """Re-derive the ``$skill`` suggestions from the editor's ``text``.

        The third sibling of :meth:`sync` and :meth:`sync_argument`, and it
        closes the same way they do: leaving the token forgets the dismissal,
        so the next ``$`` opens a fresh list rather than inheriting an Esc.

        Rows are :class:`ArgumentChoice` and go through
        :func:`argument_suggestions`, so a skill name is ranked by the very
        scorer that ranks commands and providers — ``$cr`` finds
        ``code-review`` by the rule the user already learned from ``/lgt``
        finding ``logout``.
        """
        token = skill_token(text, cursor)
        if token is None:
            self._dismissed_query = None
            self._mode = PickerMode.SKILL
            self._close()
            return
        self._apply(PickerMode.SKILL, token.query, argument_suggestions(token.query, self._choices))

    def sync_argument(self, query: str) -> None:
        """Re-derive the ARGUMENT suggestions for the current command.

        The editor calls this INSTEAD of :meth:`sync` while the buffer holds a
        command whose argument drives a list, so the two can never both be
        showing: which list is up is a property of the buffer parse, not of two
        widgets agreeing to take turns.

        A transient loading reserve is different from an ordinary empty list:
        its rows are intentionally withheld until adoption. Re-applying an
        empty match set must keep that reserve (and its Tab/Enter gate) alive;
        otherwise the key-routing pre-sync clears ``_loading`` immediately
        before asking whether it is loading — exactly the U2-2 race.
        """
        if self._loading:
            self._query = query
            if query == self._dismissed_query:
                self._close()
                return
            self._dismissed_query = None
            self._reset_rows()
            self.display = True
            self._repaint()
            self._report_highlight()
            return
        self._apply(PickerMode.ARGUMENT, query, argument_suggestions(query, self._choices))

    def _apply(self, mode: PickerMode, query: str, matches: Sequence[_Suggestion]) -> None:
        """Adopt a freshly derived candidate set, whichever list produced it.

        ``Sequence``, not ``list``: a list is invariant, so the concrete
        ``list[tuple[str, SlashCommand]]`` the command matcher returns is not a
        ``list[_Suggestion]`` — only a read-only view of one.
        """
        if mode is not self._mode:
            # A mode change is a different list of different things. Carrying the
            # highlight, the window or Esc's "not now" across it would point them
            # at rows that no longer exist. The notice goes with them: it was about
            # THAT list.
            self._mode = mode
            self._dismissed_query = None
            self._selected = 0
            self._window_start = 0
            self._chosen_by_hand = False
            self._notice = ""
            # The loading reserve goes with the notice it was riding: it
            # described THAT list's transient window (see set_loading_reserve).
            self._loading = False
        self._query = query
        if query == self._dismissed_query:
            self._close()
            return
        # The token changed, so Esc's "not now" has expired. Latching the
        # dismissal until the slash is deleted would leave a user who pressed
        # Esc once with no way to get the list back while still typing.
        self._dismissed_query = None
        if not matches:
            if mode is PickerMode.ARGUMENT and self._notice:
                # No rows, but something to say in their place. The list stays up
                # holding the one informational row, and holds it across every
                # re-derivation — the user editing the argument of a command with
                # nothing to offer does not make the answer any less true.
                self._reset_rows()
                self.display = True
                self._repaint()
                # The rows are gone even though the surface stays up, so the
                # observers have to hear it: this is the ONE exit from `_apply`
                # that used to return without reporting, and the composer's
                # inline ghost outlived the row it described because of it. The
                # list would sit showing "credential store unreadable" while the
                # composer still promised `/mcp login`, and Tab — the key the
                # dim text exists to describe — inserted a literal tab (UX
                # review round 2, U9). `_reset_rows` has already emptied
                # `_matches`, so the report is what turns that into a cleared
                # preview. Every other path out of `_apply` reports; this one
                # must too, or "a notice replaced the rows" becomes invisible to
                # anything watching the accept target.
                self._report_highlight()
                return
            self._close()
            return
        if [name for name, _ in matches] != [name for name, _ in self._matches]:
            # A different candidate set means the old highlight pointed at a
            # different command; keeping the index would silently move the
            # selection under the user's fingers. It also retires an explicit
            # choice — the row the user arrowed onto is gone.
            self._selected = 0
            self._window_start = 0
            self._chosen_by_hand = False
        self._matches = list(matches)
        self.display = True
        # Real rows landed, so the transient loading reserve is over wherever
        # it was set from: Tab/Enter have a highlighted row to act on again
        # (U2-2). The notice itself is already superseded — rows answer the
        # question it was asking — but `_notice` is only withdrawn by an
        # explicit ``set_notice("")``/close, so clear the FLAG here rather
        # than the text: `_repaint` never paints the notice once matches
        # exist, and the app's next fill resets both together.
        self._loading = False
        self._scroll_to_selection()
        self._repaint()
        self._report_highlight()

    def move(self, delta: int) -> None:
        """Move the highlight by ``delta`` rows, wrapping at both ends."""
        if not self._matches:
            return
        self._selected = (self._selected + delta) % len(self._matches)
        # An arrow press is the user reading the list and picking a row, which
        # is the whole of what the ambiguity check is worried about. Recording it
        # is what lets Enter send on the first press after a deliberate move
        # while still requiring two on a word the matcher chose alone.
        self._chosen_by_hand = True
        self._scroll_to_selection()
        self._repaint()
        self._report_highlight()

    def scroll_rows(self, delta: int) -> None:
        """Move the highlight by ``delta`` rows for a WHEEL notch, clamped.

        Deliberately not :meth:`move`: that wraps, which suits a discrete
        arrow press and not a scroll gesture — a wheel that teleports from the
        last row back to the first reads as the menu having reset itself.

        ``_chosen_by_hand`` is set for the same reason :meth:`move` sets it: a
        wheel notch is the user reading the list and landing on a row, which
        is exactly the deliberate choice the ambiguity check looks for.
        """
        if not self._matches:
            return
        target = max(0, min(len(self._matches) - 1, self._selected + delta))
        if target == self._selected:
            return
        self._selected = target
        self._chosen_by_hand = True
        self._scroll_to_selection()
        self._repaint()
        self._report_highlight()

    def dismiss(self) -> None:
        """Hide the picker for the CURRENT word without touching the text."""
        self._dismissed_query = self._query
        self._close()

    def close(self) -> None:
        """Hide the picker (submission, completion — not a dismissal)."""
        self._close()

    def choose(self, index: int) -> None:
        """Highlight ``index`` and hand its command to the editor."""
        if not 0 <= index < len(self._matches):
            return
        self._selected = index
        self._on_choose(self._matches[index][0])

    # -- mouse --------------------------------------------------------------
    # Public handler names on purpose: Textual dispatches `_on_<event>` and
    # then `on_<event>`, so the base Widget keeps its own click/leave
    # bookkeeping (`mouse_hover`, which drives every `:hover` rule) instead of
    # being shadowed by an override that would silently latch it on.
    def on_click(self, event: events.Click) -> None:
        # Stop the click here: the input dock below is not a click target, and
        # letting it bubble hands the event to a parent mid-completion.
        event.stop()
        index = self._index_at(event.y)
        if index is not None:
            self.choose(index)

    # Stopped for the same reason the click is: the menu floats over the
    # transcript, so a wheel left to bubble scrolls the conversation behind
    # it as well — two surfaces moving for one gesture.
    def on_mouse_scroll_down(self, event: events.MouseScrollDown) -> None:
        event.stop()
        self.scroll_rows(1)

    def on_mouse_scroll_up(self, event: events.MouseScrollUp) -> None:
        event.stop()
        self.scroll_rows(-1)

    def on_mouse_move(self, event: events.MouseMove) -> None:
        index = self._index_at(event.y)
        if index != self._hovered:
            self._hovered = index
            self._repaint()
            self._report_highlight()
        # The hand pointer only over a ROW: the picker's padding and notice
        # rows are not click targets, and a static `pointer` rule on the
        # widget would promise the click the empty rows cannot keep. Setting
        # the inline rule is what makes the shape follow the hover — the
        # property's own observer re-runs `Screen.update_pointer_shape()`,
        # and no-ops when the value did not change.
        self.styles.pointer = "pointer" if index is not None else "default"

    def on_leave(self, event: events.Leave) -> None:
        if self._hovered is not None:
            self._hovered = None
            self._repaint()
            self._report_highlight()
        self.styles.pointer = "default"

    def on_resize(self, event: events.Resize) -> None:
        """Re-truncate every row against the new width."""
        if self._matches or self._notice:
            self._repaint()

    # -- rendering ----------------------------------------------------------
    def render_rows(self, width: int) -> list[Text]:
        """One row per VISIBLE suggestion, each exactly ``width`` cells."""
        start, end, _total = self.visible_window()
        return [self._row(index, width) for index in range(start, end)]

    def render_text(self, width: int) -> Text:
        """The full renderable: the visible rows plus the overflow marker.

        With no rows at all it is the informational row, or nothing — the two
        states the picker can be VISIBLE in without a single suggestion.
        """
        if not self._matches:
            return self._notice_row(width) if self._notice else Text()
        rows = self.render_rows(width)
        overflow = self._overflow_row(width)
        if overflow is not None:
            rows.append(overflow)
        out = Text()
        for index, row in enumerate(rows):
            if index:
                out.append("\n")
            out.append_text(row)
        return out

    def _repaint(self) -> None:
        # Matching, selection and the visible window are all resolved without
        # a screen, so the state machine is fully exercisable (and testable)
        # off-app; only PAINTING needs one, because Static.update has to reach
        # the app console to build its visual.
        if not self.is_mounted or not (self._matches or self._notice):
            return
        width = max(self.size.width, _MIN_RENDER_WIDTH)
        if not self._matches:
            # The informational row stands alone: one row, no window and no
            # overflow count, because there is nothing to count.
            self._pin_height(1)
            self.update(self._notice_row(width))
            return
        rows = self.render_rows(width)
        overflow = self._overflow_row(width)
        row_count = len(rows) + (0 if overflow is None else 1)
        # Pin the height: `auto` would measure content before layout knows the
        # real width and settle one row too tall per suggestion, exactly the
        # trap ToolCard documents.
        self._pin_height(row_count)
        self.update(self.render_text(width))

    def _pin_height(self, rows: int) -> None:
        """Pin the picker to ``rows`` and announce a CHANGE to the dock (R7-3).

        The single place the height is written, so no path can resize the
        picker without the boot composition hearing about it. The comparison is
        against the last value THIS method wrote rather than ``styles.height``,
        which is a Scalar and compares awkwardly, and the message is posted only
        on a real change so the per-keystroke repaint stays free.
        """
        # The height is written UNCONDITIONALLY. Only the notification is
        # gated: `styles.height` is also cleared/re-derived from outside this
        # widget (a close hides it, a re-open re-applies the sheet), so an
        # early return here would let a stale pin survive a reopen.
        self.styles.height = rows
        if rows == self._pinned_rows:
            return
        self._pinned_rows = rows
        # Guarded: the widget repaints before it is on a screen during compose,
        # and posting from there raises rather than reaching the app.
        try:
            self.post_message(self.RowsResized())
        except NoScreen:  # pragma: no cover - pre-mount repaint
            pass

    def _row(self, index: int, width: int) -> Text:
        """The row for suggestion ``index``, dispatched on what it stands for."""
        name, item = self._matches[index]
        styles = self._row_styles(index)
        if isinstance(item, ArgumentChoice):
            return self._argument_row(name, item, width, styles)
        return self._command_row(name, item, width, styles)

    def _row_styles(self, index: int) -> _RowStyles:
        """Ground and text styles for row ``index`` — shared by both kinds."""
        selected = index == self._selected
        hovered = index == self._hovered

        # ONE green: the accent is spent on the highlighted command NAME.
        #
        # Selection is carried by HUE, not elevation. Pure luminance steps could
        # not do it — surface->raised measures 1.096:1 and surface->overlay
        # 1.218:1, both imperceptible — so the highlight rested entirely on the
        # accent and hover (which has no accent) gave a mouse user almost no
        # feedback about which row a click would run. `tint-select` is the same
        # move `tint-danger` already makes on a failed tool row: elevation says
        # "this is a row", hue says "this is its state" (D8).
        # Hover is ADDITIVE and selection stays dominant. Written the other way
        # round — hover overwriting the ground — pointing at the selected row
        # swapped its clearly-tinted ground for the faintest step in the ramp, so
        # the highlight vanished under the pointer and the row read as LESS
        # selected than its neighbours. A mouse user arrowing to a row and then
        # reaching for the mouse watched the picker appear to lose its place.
        ground = theme_mod.semantic_color("surface")
        if hovered:
            # `overlay`, not `raised`: raised measures dE2000 3.06 against
            # surface, which is the very step the comment above rejects as
            # imperceptible. Every row here is a click target that RUNS a command
            # and some of them are destructive, so "which row will this click
            # hit" has to be answerable.
            ground = theme_mod.semantic_color("overlay")
        if selected:
            ground = theme_mod.semantic_color("tint-select-hi" if hovered else "tint-select")
        row_bg = Style(bgcolor=ground)
        name_style = row_bg + Style(color=theme_mod.semantic_color("accent" if selected else "fg"))
        # `dim`, not `faint`. An alias is a typeable command, and the picker is
        # where a user DISCOVERS that `/quit` and `/models` exist — at `faint`
        # that discovery rendered at 1.7:1 against its own ground, so the row
        # promised two names and hid one. `faint` stays what it is: chrome, for
        # the band's separators (D2).
        #
        # One step up on the selected row: `dim` over the green-tinted ground
        # falls to 3.97:1, just under AA, and the selected row is the one the user
        # is actually reading. The three-tier hierarchy holds everywhere else.
        alias_style = row_bg + Style(color=theme_mod.semantic_color("muted" if selected else "dim"))
        description_style = row_bg + Style(color=theme_mod.semantic_color("muted"))
        # The cursor is MUTED, not the accent name style: the input's focused
        # chevron is already accent at the same column on the adjacent row, so
        # two identical green chevrons a row apart read as a duplicated caret
        # exactly when the user is mid-keystroke (D17).
        cursor_style = row_bg + Style(color=theme_mod.semantic_color("muted"))
        return _RowStyles(
            selected=selected,
            ground=row_bg,
            name=name_style,
            alias=alias_style,
            description=description_style,
            cursor=cursor_style,
        )

    def _gutter(self, styles: _RowStyles) -> Text:
        # Padded from the constant rather than written out: a hard-coded two-cell
        # mark under a three-cell gutter would shift only the SELECTED row, which
        # is the one row a misalignment is guaranteed to be noticed on.
        row = Text()
        mark = (_CURSOR if styles.selected else "").ljust(_GUTTER_CELLS)
        row.append(mark, style=styles.cursor)
        return row

    def _command_row(self, name: str, command: SlashCommand, width: int, s: _RowStyles) -> Text:
        row = self._gutter(s)
        row_bg = s.ground

        primary = f"/{name}"
        aliases = tuple(other for other in command.names if other != name)
        alias_run = "  " + " ".join(f"/{alias}" for alias in aliases) if aliases else ""

        description = command.description.strip()
        if description and width > DESCRIPTION_COLLAPSE_WIDTH:
            column = max(1, min(self._primary_column(), width - _GUTTER_CELLS - _EDGE_MARGIN * 2))
            budget = max(1, column - _PRIMARY_COLUMN_GAP)
            used = self._append_primary(row, primary, alias_run, budget, s.name, s.alias)
            gap = max(_PRIMARY_COLUMN_GAP, column - used)
            row.append(" " * gap, style=row_bg)
            remaining = width - _GUTTER_CELLS - used - gap - _EDGE_MARGIN
            if remaining > _MIN_DESCRIPTION_CELLS:
                row.append(truncate_cells(description, remaining), style=s.description)
                return _pad_to(row, width, row_bg)
            # Not enough room after the name column to say anything useful:
            # rebuild as a name-only row rather than ship a stub description.
            row = self._gutter(s)

        budget = max(1, width - _GUTTER_CELLS - _EDGE_MARGIN)
        self._append_primary(row, primary, alias_run, budget, s.name, s.alias)
        return _pad_to(row, width, row_bg)

    def _argument_row(self, name: str, choice: ArgumentChoice, width: int, s: _RowStyles) -> Text:
        """``name  description                     detail`` — no leading slash.

        The slash is COMMAND vocabulary. Prefixing an argument with it would read
        as `/login /anthropic`, which is not something the user can type. Aliases
        are absent for the same reason: `claude` makes anthropic FINDABLE, but the
        only text that completes into the buffer is the provider id, so listing
        the alias would advertise input the command does not accept.
        """
        row = self._gutter(s)
        row_bg = s.ground
        # `danger` only when the state is a problem; an unfinished login is not
        # one. Tinting every un-logged-in provider red would make the ordinary
        # `/login` list read as a wall of failures.
        detail_style = row_bg + Style(
            color=theme_mod.semantic_color("danger" if choice.alert else "muted")
        )
        # The NAME carries the danger too: on `/logout`-style lists the detail
        # column may hold an innocuous state word ("connected") or nothing at
        # all, so tinting only the detail could leave a destructive row
        # visually identical to a benign one — which is what happened on the
        # `/mcp logout` list. The name is the cell every destructive row has.
        name_style = s.name
        if choice.alert and not s.selected:
            name_style = row_bg + Style(color=theme_mod.semantic_color("danger"))

        span = max(1, width - _GUTTER_CELLS - _EDGE_MARGIN)
        detail = choice.detail.strip()
        # Reserve the widest detail in the MATCH SET, not this row's own. The
        # state is a column, and a column the user can scan has one left edge;
        # right-aligning each string to its own row's trailing edge started the
        # three credential states at three different x, so "which of these am I
        # logged into" meant reading eight strings instead of scanning an edge.
        #
        # Reserved BEFORE the description, and the description dropped first when
        # only one of the two fits: at `/logout` the detail names the credential
        # about to be REMOVED, which no other column on the row says.
        column_cells = self._detail_column()
        reserved = column_cells + _PRIMARY_COLUMN_GAP if column_cells else 0
        if reserved and span - reserved < self._name_floor():
            # Uniform by construction: every row reserves the same width against
            # the same floor, so the column is dropped for the whole list at once
            # and can never leave a ragged half of one behind.
            column_cells = 0
            reserved = 0
        # D2: a row whose OWN detail is empty places NOTHING at the shared column
        # edge, so it need not pay for the column — its description reclaims the
        # full body. This is the safe half of D2: detail-BEARING rows still all
        # reserve the same ``column_cells`` and so keep the one scannable left
        # edge (the /login/`/theme`/`/mcp logout` state scan the column exists
        # for), while the many empty-detail rows a mixed list carries (34 of 35
        # `/theme` rows, most `/mcp` and `/login` rows, `/agent`'s specialists
        # vs. packaged roles) stop being truncated to make room for a column
        # they contribute nothing to. Per-row reclaim for a SHORT-but-nonempty
        # detail is deliberately NOT done: shrinking one row's reserve below the
        # set width would start its detail at a different x than its neighbours',
        # which is exactly the ragged-edge regression the shared column forbids
        # (see the right-align note above). ``_append_detail`` already no-ops on
        # an empty detail, so the only change here is not charging its body.
        row_reserved = reserved if detail else 0
        body = span - row_reserved

        description = choice.description.strip()
        if description and width > DESCRIPTION_COLLAPSE_WIDTH:
            column = max(1, min(self._primary_column(), body))
            clipped = truncate_cells(name, max(1, column - _PRIMARY_COLUMN_GAP))
            row.append(clipped, style=name_style)
            used = cell_len(clipped)
            gap = max(_PRIMARY_COLUMN_GAP, column - used)
            remaining = body - used - gap
            if remaining > _MIN_DESCRIPTION_CELLS:
                row.append(" " * gap, style=row_bg)
                row.append(truncate_cells(description, remaining), style=s.description)
                return self._append_detail(row, width, detail, column_cells, detail_style, row_bg)
            # Not enough room after the name column to say anything useful:
            # rebuild as a name-only row rather than ship a stub description.
            row = self._gutter(s)

        row.append(truncate_cells(name, max(1, body)), style=name_style)
        return self._append_detail(row, width, detail, column_cells, detail_style, row_bg)

    def _append_detail(
        self,
        row: Text,
        width: int,
        detail: str,
        column_cells: int,
        detail_style: Style,
        row_bg: Style,
    ) -> Text:
        """Left-align ``detail`` inside the reserved column and pad to ``width``.

        The COLUMN is right-aligned to the row's trailing edge; the string inside
        it is not, so every row's detail begins at the same x. Its cells were
        reserved out of the body budget above, so padding to the column's start
        can only ever ADD space — never truncate content already appended.
        """
        if column_cells and detail:
            _pad_to(row, width - _EDGE_MARGIN - column_cells, row_bg)
            row.append(detail, style=detail_style)
        return _pad_to(row, width, row_bg)

    def _append_primary(
        self,
        row: Text,
        primary: str,
        alias_run: str,
        budget: int,
        name_style: Style,
        alias_style: Style,
    ) -> int:
        """Append the name (plus aliases when they fit); return cells used.

        Aliases are all-or-nothing: half an alias list is worse than none,
        and the name is the part being chosen, so it gets the whole budget
        when the two cannot both fit.
        """
        if alias_run and cell_len(primary) + cell_len(alias_run) <= budget:
            row.append(primary, style=name_style)
            row.append(alias_run, style=alias_style)
            return cell_len(primary) + cell_len(alias_run)
        clipped = truncate_cells(primary, budget)
        row.append(clipped, style=name_style)
        return cell_len(clipped)

    def _overflow_row(self, width: int) -> Text | None:
        start, end, total = self.visible_window()
        hidden = total - (end - start)
        if hidden <= 0:
            return None
        dim = Style(
            color=theme_mod.semantic_color("dim"),
            bgcolor=theme_mod.semantic_color("surface"),
        )
        row = Text()
        row.append(" " * _GUTTER_CELLS, style=dim)
        row.append(
            truncate_cells(f"… {hidden} more", max(1, width - _GUTTER_CELLS - _EDGE_MARGIN)),
            style=dim,
        )
        return _pad_to(row, width, dim)

    def _notice_row(self, width: int) -> Text:
        """The informational row: why this list is empty, in the marker's voice.

        Deliberately the overflow marker's exact treatment — dim on the dock's own
        surface, text starting at the name column — because the two say the same
        KIND of thing: a fact about the list rather than a row in it. A second
        style here would advertise it as something the user can act on.
        """
        dim = Style(
            color=theme_mod.semantic_color("dim"),
            bgcolor=theme_mod.semantic_color("surface"),
        )
        row = Text()
        row.append(" " * _GUTTER_CELLS, style=dim)
        row.append(
            truncate_cells(self._notice, max(1, width - _GUTTER_CELLS - _EDGE_MARGIN)),
            style=dim,
        )
        return _pad_to(row, width, dim)

    def _primary_column(self) -> int:
        """Fit-to-content name column, clamped to the 12..32 cell band."""
        widest = 0
        for name, item in self._matches:
            if isinstance(item, ArgumentChoice):
                # No slash and no alias run: an argument row's primary column is
                # the bare value, which is all that is typeable.
                cells = cell_len(name)
            else:
                aliases = tuple(other for other in item.names if other != name)
                cells = cell_len(f"/{name}")
                if aliases:
                    cells += cell_len("  " + " ".join(f"/{alias}" for alias in aliases))
            widest = max(widest, cells + _PRIMARY_COLUMN_GAP)
        return max(_PRIMARY_COLUMN_MIN, min(_PRIMARY_COLUMN_MAX, widest))

    def _detail_column(self) -> int:
        """Fit-to-content DETAIL column: the widest detail in the match set.

        Unclamped, unlike :meth:`_primary_column`. The detail is generated by the
        app from a closed vocabulary (three credential states, two credential
        kinds), not typed by a user, and the row already drops the column whole
        when reserving it would squeeze the name past :meth:`_name_floor`.
        """
        widest = 0
        for _name, item in self._matches:
            if isinstance(item, ArgumentChoice):
                widest = max(widest, cell_len(item.detail.strip()))
        return widest

    def _name_floor(self) -> int:
        """Cells the NAME column keeps before ``detail`` is dropped.

        The widest id ACTUALLY offered, not a constant. A fixed floor of twelve
        answers "would a twelve-cell name fit" for a list whose longest name is
        thirteen, which is how `openai-device` rendered as `openai-devi…` beside
        an intact `needs login` at one exact render width — the detail column
        keeping cells from the only text on the row the user can type.

        Clamped at ``_PRIMARY_COLUMN_MAX``: past that the name column truncates
        anyway, so letting the floor run away would drop the detail at every
        width and buy nothing.
        """
        widest = max((cell_len(name) for name, _ in self._matches), default=0)
        return max(_MIN_NAME_CELLS, min(_PRIMARY_COLUMN_MAX, widest))

    # -- window -------------------------------------------------------------
    def _row_budget(self) -> int:
        try:
            screen_height = self.screen.size.height
        except NoScreen:
            screen_height = 0
        if screen_height <= 0:
            return MAX_VISIBLE_ROWS
        if self._mode is PickerMode.ARGUMENT:
            # The screen-height guard stays — a picker that squeezed the
            # transcript to nothing is the failure this whole method exists for —
            # but the ceiling does not: see ``_ARGUMENT_HEIGHT_DIVISOR``.
            return max(
                _ARGUMENT_ROWS_MIN,
                screen_height // _ARGUMENT_HEIGHT_DIVISOR - _ARGUMENT_CHROME_ROWS,
            )
        return max(1, min(MAX_VISIBLE_ROWS, screen_height // _SCREEN_HEIGHT_DIVISOR))

    def _scroll_to_selection(self) -> None:
        budget = self._row_budget()
        if self._selected < self._window_start:
            self._window_start = self._selected
        elif self._selected >= self._window_start + budget:
            self._window_start = self._selected - budget + 1
        self._window_start = max(0, min(self._window_start, max(0, len(self._matches) - budget)))

    def _index_at(self, y: int) -> int | None:
        """Suggestion index at content row ``y``, or ``None``.

        Returns ``None`` for the overflow marker row and for the informational row
        of an empty list: both are facts about the list, not choices in it, and
        clicking a fact must not run a command. The informational case falls out of
        the window being empty — there is no index for any ``y``.
        """
        start, end, _total = self.visible_window()
        index = self._window_start + y
        if not start <= index < end:
            return None
        return index

    def _report_highlight(self) -> None:
        """Tell the observer which ARGUMENT row the user's eye is on now.

        The reported row is the HOVER target when the pointer is over one,
        else the keyboard highlight — the same precedence the row grounds
        paint, so what previews is always the row that reads as active.
        De-duplicated on the name: a mouse crossing five cells of one row and
        a repaint that reproduced the same set both say nothing new.
        """
        self._report_preview()
        if self._on_highlight is None or self._suppress_report:
            return
        name: str | None = None
        if self._mode is PickerMode.ARGUMENT and self._matches:
            index = self._hovered if self._hovered is not None else self._selected
            if 0 <= index < len(self._matches):
                name = self._matches[index][0]
        if name != self._reported_highlight:
            self._reported_highlight = name
            self._on_highlight(name)

    def _report_preview(self) -> None:
        """Tell the observer which row an ACCEPT KEY would take right now.

        Deliberately different from :meth:`_report_highlight` on both axes:

        * **Both modes.** A command-word list has nothing to *preview* in the
          ``/theme`` sense, but it certainly has an accept target — the ghost
          in COMMAND mode is the feature's most common state.
        * **Keyboard selection only, never the hover.** Tab acts on
          ``_selected``, so a ghost following the pointer would promise a row
          the key will not insert. Resting the pointer over the list while
          reaching for Tab was enough to trigger it (U2).

        Reports ``None`` when the list is not showing rows, which is what
        retires the ghost on dismissal and on close (U3).
        """
        if self._on_preview is None or self._suppress_report:
            return
        name: str | None = None
        if self._matches and self.display and 0 <= self._selected < len(self._matches):
            name = self._matches[self._selected][0]
        if name != self._reported_preview:
            self._reported_preview = name
            self._on_preview(name)

    def _reset_rows(self) -> None:
        """Drop every row and the state that pointed into them, but not the
        notice: a list can be showing its informational row with no rows at all."""
        self._matches = []
        self._selected = 0
        self._window_start = 0
        self._hovered = None

    def _close(self) -> None:
        self._reset_rows()
        # Release the hand BEFORE the surface disappears. Textual only
        # re-evaluates the pointer on a mouse/style event; removing a picker
        # under a stationary pointer otherwise leaves OSC 22 at `pointer`
        # until the person moves again.
        self.styles.pointer = "default"
        # The notice belonged to the list that is now gone. Esc, a completion and a
        # submission all arrive here, and each one is the user done with it.
        self._notice = ""
        self._loading = False
        self.display = False
        self._report_highlight()
