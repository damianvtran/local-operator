"""The sigil grammar — where a ``/``, ``$`` or ``@`` token starts, and what it spans.

Two surfaces read the same grammar. The composer
(``tui/widgets/command_picker.py``) reads it on every keystroke to decide
whether the character just typed opens a picker, and the submit-side resolver
(``local_operator/references.py``) reads it again on the finished draft to
decide which spans to expand. They must agree exactly: a boundary rule that
holds in the composer but not at submit time is a token the user saw
highlighted and the agent never received.

That agreement is why the whole ``@`` parser lives here and not only the
boundary rule. :func:`at_token` and :func:`split_token` are what both surfaces
call, so the span the picker highlights and the span the resolver expands come
from ONE function rather than two that agree today.

WHY THIS IS NOT IN ``command_picker.py``
----------------------------------------
``references.py`` is a SESSION-layer module and ``command_picker.py`` is a
Textual widget. A session-layer import of a widget module is the wrong
direction, and it is not merely untidy: it would put ``textual`` on the
``session_factory`` import path, which ``tests/unit/test_import_graph.py:163``
fails on outright (``_assert_absent(session_factory_modules, "textual", "TUI
front end; the server has no terminal")``). The server, the scheduler and the
exec worker all funnel through that composition root and none of them has a
terminal. So the grammar lives here, below both callers, and neither owns it.

CONSTRAINT — this module must stay host-free: no Textual, no Rich, no
``local_operator`` import at module scope. It sits below both readers the same
way ``harness/rows.py`` sits below the renderers that must not import each
other. Its entire dependency set is ``typing``, and that is the property the
import-graph guard in ``tests/unit/test_sigils.py`` pins — adding a host import
here is exactly the regression that guard exists to catch.

Adding a new sigil rule belongs HERE, not in a host. A rule made in one reader
is a rule the other will not make.
"""

from __future__ import annotations

from typing import NamedTuple


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


#: A sigil opens a token only at a WORD BOUNDARY: the line start, or right
#: after whitespace. This is what keeps ``src/foo`` and ``and/or`` from opening
#: the picker — the ``/`` there is glued to a preceding non-space character, so
#: it is punctuation inside a word, not the start of a command. The rule is the
#: same one a shell or an editor command palette uses to tell a path apart from
#: a command, and it is the ONE thing that makes inline detection safe to run on
#: every keystroke of ordinary prose. ``$`` leans on it harder still: it is the
#: whole reason ``costs$5`` and ``a$b`` cannot open a skill list.
def is_boundary(line: str, index: int) -> bool:
    """Whether ``line[index]`` (a sigil) begins a fresh token."""
    return index == 0 or line[index - 1].isspace()


def _line_of_cursor(text: str, cursor: int | None) -> tuple[str, int, int]:
    """``(line, line_start, column)`` for ``cursor`` within ``text``.

    Pure: stdlib string operations over its two arguments, no module state.

    Deliberately NOT shared with ``command_picker._line_of_cursor``. That one is
    a widget-module function (importing it would put Textual on the
    ``session_factory`` path, which ``test_import_graph.py:163`` fails on), and
    it carries concerns this caller does not have: CRLF stripping and caret
    clamping for a LIVE caret in a draft buffer. Duplicating those concerns
    into a module that must stay host-free is worse than four lines that do
    only what a submitted, finished string needs.
    """
    if cursor is None or cursor > len(text):
        cursor = len(text)
    if cursor < 0:
        cursor = 0
    line_start = text.rfind("\n", 0, cursor) + 1  # 0 when no newline precedes
    line_end = text.find("\n", cursor)
    if line_end == -1:
        line_end = len(text)
    return text[line_start:line_end], line_start, cursor - line_start


def _token_end(line: str, at: int) -> tuple[int, str]:
    """``(end, query)`` for the ``@`` token opening at ``line[at]``.

    ``end`` indexes the first cell past the token, so a completion can rebuild
    just that span.

    Grammar, not resolver policy, which is why it lives here rather than beside
    its second caller: it answers only "where does this token stop", the same
    question :func:`is_boundary` answers for the other end, and it consults
    nothing about the filesystem, the deny-list or the approval gate. The
    submit-side resolver (``references._reference_tokens``) imports it back
    from here so the span it expands and the span the composer highlighted are
    produced by ONE function — two implementations of "where the token ends" is
    exactly the drift this module exists to prevent.

    Two forms. UNQUOTED terminates on whitespace only — never on ``/``, which
    is the single most important difference from ``skill_token``: a path IS
    slashes, and a token that ended at the first one could never name
    ``src/app.py``. QUOTED (``@"my file.txt"``) runs to the closing quote and
    yields the content between the quotes, which is the only way to reference a
    name containing a space; an unterminated quote falls back to the whitespace
    rule so a half-typed ``@"`` still parses instead of swallowing the line.
    """
    if at + 1 < len(line) and line[at + 1] == '"':
        close = line.find('"', at + 2)
        if close != -1:
            return close + 1, line[at + 2 : close]
    end = at + 1
    while end < len(line) and not line[end].isspace():
        end += 1
    return end, line[at + 1 : end]


def _active_at(line: str, column: int) -> int | None:
    """Index of the boundary ``@`` the caret is editing, or ``None``.

    The last boundary sigil at or before the caret is the one being edited
    (``@a @sr|`` is ``@sr``), which is ``_active_sigil``'s rule in
    ``command_picker.py:655-667`` — expressed here through :func:`is_boundary`
    so the composer and the submit-side resolver cannot disagree about where a
    token starts.
    """
    candidate: int | None = None
    for index, char in enumerate(line):
        if char == "@" and is_boundary(line, index) and index <= column:
            candidate = index
    return candidate


def at_token(text: str, cursor: int | None = None) -> SlashContext | None:
    """The ``@`` token the caret sits in, or ``None``.

    The ``@`` counterpart to ``skill_token``, with three deliberate
    differences and no others:

    - a BARE ``@`` returns ``query=""`` and opens the list on the cwd, rather
      than being special-cased closed — ``@`` is how you ask "what is here";
    - the token MAY contain ``/`` and terminates on whitespace only (see
      :func:`_token_end`);
    - there are no ``commands``/``prompt_commands``/``name_commands``
      parameters and no claiming. ``$`` needs them because a terminated command
      owns the rest of its line as an argument and the two sigils would fight
      over one caret; ``@`` has no command-argument arbitration to do.

    An email address is not a token and never was: ``user@host.com`` has a
    non-space character before the ``@``, so :func:`is_boundary` is False and
    the parse stops before any filesystem work.

    Word-phase only, like every other parser here: the caret must be INSIDE the
    token, so moving out into the request closes the list.

    Hand-typed UNQUOTED spaces are unsupported, exactly as in a shell —
    ``@my file.txt`` references ``my``. Use ``@"my file.txt"``.
    """
    line, line_start, column = _line_of_cursor(text, cursor)
    at = _active_at(line, column)
    if at is None:
        return None
    end, query = _token_end(line, at)
    if column > end:
        return None
    return SlashContext(line_start + at, query, line_start + end)


def split_token(query: str) -> tuple[str, str]:
    """Split a token's ``query`` into ``(dir_part, name_query)`` at the LAST ``/``.

    This is how a shell completes a path, which is the behaviour a terminal
    user already has in their fingers, and it makes deepening lazy by
    construction: one directory is scanned per segment typed, never the tree.

    ::

        ""             -> ("",             "")
        "sr"           -> ("",             "sr")
        "src/"         -> ("src/",         "")
        "src/ap"       -> ("src/",         "ap")
        "../sibling/x" -> ("../sibling/",  "x")
    """
    cut = query.rfind("/")
    if cut == -1:
        return "", query
    return query[: cut + 1], query[cut + 1 :]
