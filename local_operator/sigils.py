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
other. Its entire dependency set is the standard library, and that is the
property the import-graph guard in ``tests/unit/test_sigils.py`` pins — adding
a host import here is exactly the regression that guard exists to catch.

Adding a new sigil rule belongs HERE, not in a host. A rule made in one reader
is a rule the other will not make.
"""

from __future__ import annotations

import re
from collections.abc import Collection, Sequence
from string import ascii_letters, digits
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


def at_token_spans(line: str) -> list[tuple[int, int, str]]:
    """EVERY ``@`` token on ``line``, as ``(start, end, query)``.

    The whole-line counterpart of :func:`at_token`, and it exists because the
    two questions are genuinely different and both are asked: the picker needs
    "the token the CARET is in" (one answer), while the composer's reference
    ink needs "every token this line names" (all of them), so a draft holding
    three references can be painted without the caret deciding which two go
    dark. Both read the same boundary rule and the same :func:`_token_end`, so
    the spans cannot disagree with each other or with the resolver.

    A bare ``@`` is not returned. An empty query is not a reference —
    ``references._reference_tokens`` keeps a token only when it has one, and
    ``@`` alone is how the operator asks "what is here", not a path.

    Pure and host-free, like everything in this module: reads its argument and
    the two rules above, consults no filesystem and no widget. Whether a span
    is worth painting is the caller's question, not the grammar's.
    """
    spans: list[tuple[int, int, str]] = []
    index = 0
    while index < len(line):
        if line[index] != "@" or not is_boundary(line, index):
            index += 1
            continue
        end, query = _token_end(line, index)
        if query:
            spans.append((index, end, query))
        # ``max`` so a token that ends where it starts (a lone `@`) still
        # advances; ``end`` is always >= index + 1 in practice.
        index += max(end - index, 1)
    return spans


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


# --- The `$skill` trigger and its block grammar -------------------------------
#
# BOTH live here, and the reason is the import graph rather than tidiness. Two
# readers need them: ``skills/invoke.py`` writes the block at submit, and
# ``references.py`` / ``harness/rows.py`` must RECOGNISE it — the first to keep
# an ``@path`` inside a skill body from being read as a reference, the second to
# strip it from a transcript row. ``skills/invoke.py`` is not importable from
# either: ``local_operator/skills/__init__.py`` pulls ``discovery`` in, which is
# the whole tree the picker exists to keep off its boot path. So the grammar
# sits below all three, beside the boundary rule it is built on, and
# ``skills.invoke`` re-exports these names as its public surface.

#: A block only counts when the marker, the preamble line and the closer are all
#: present. The preamble is what tells an APPENDED block from a marker the
#: operator quoted: a pasted or forwarded block is the same bytes (accepted
#: cost, recorded at ``harness/rows.py``), but a sentence that merely names the
#: tag has no preamble and is never hidden.
SKILL_BLOCK_OPEN = "<operator-skills>"
SKILL_BLOCK_CLOSE = "</operator-skills>"
SKILL_BLOCK_JOIN = "\n\n"
SKILL_BLOCK_PREAMBLE = (
    "The user invoked the following skills directly. Follow them for this "
    "request. Their reference files, if any, are listed at the end of each body "
    "and are read with `skill://<name>/<path>`."
)

_ZERO_WIDTH_SPACE = "\u200b"
_DEFUSED_SKILL_OPEN = SKILL_BLOCK_OPEN.replace("<", "<" + _ZERO_WIDTH_SPACE, 1)
_DEFUSED_SKILL_CLOSE = SKILL_BLOCK_CLOSE.replace("<", "<" + _ZERO_WIDTH_SPACE, 1)
#: A body line that spells an element head would otherwise read as an
#: attachment to :func:`skill_block_names` (and, one day, to the receipt). The
#: same one-character neutralisation as the block markers, and for the same
#: reason: the sequence must stop matching without changing what the model reads.
_DEFUSED_ELEMENT_HEAD = "<" + _ZERO_WIDTH_SPACE + "skill name="

#: The resolver's name class, in one place: an alphanumeric head, then
#: alphanumerics, dots, underscores and hyphens. Explicitly ASCII, because every
#: caller's token grammar is ASCII-classed and a Unicode letter here would span
#: a token the resolver would never read.
_SKILL_NAME_HEAD = frozenset(ascii_letters + digits)
_SKILL_NAME_TAIL = _SKILL_NAME_HEAD | frozenset("._-")

_SKILL_ELEMENT_HEAD = '<skill name="'
_SKILL_ELEMENT_HEAD_RE = re.compile(r'<skill name="([^"]*)">')


class SkillMention(NamedTuple):
    """One resolved ``$name`` in a message.

    ``start``/``end`` are the span of the token that FIRES — the trailing
    punctuation that made a longer name unresolvable is deliberately outside it,
    because it stays the user's prose. ``name`` is the discovered name, which is
    what dedupes ``$Research`` against ``$research``; ``text`` is the token as
    typed, so a surface can paint exactly those characters.
    """

    start: int
    end: int
    text: str
    name: str
    leading: bool


def _skill_name_end(text: str, start: int) -> int:
    """First index past the name-class run starting at ``start``.

    ``start`` when the character there cannot open a name. Shared by the raw
    span scanner and :func:`skill_mentions` so the extent cannot drift between
    the ink and the resolver.
    """
    if start >= len(text) or text[start] not in _SKILL_NAME_HEAD:
        return start
    end = start + 1
    while end < len(text) and text[end] in _SKILL_NAME_TAIL:
        end += 1
    return end


def skill_name_spans(text: str) -> list[tuple[int, int, str]]:
    """EVERY boundary ``$name`` run in ``text``, as ``(start, end, name)``.

    The unresolved counterpart of :func:`skill_mentions`, and the extent the
    composer's ink is painted from: ``$research,`` spans ``research`` (the comma
    is outside the class) while ``$research.`` spans ``research.`` — the dot is
    INSIDE the class, so it joins the run and the resolver then has to trim it
    (:func:`skill_mentions`). Reporting the untrimmed run is what lets the ink
    and the resolver agree on where the characters end even when they disagree
    about whether the token fires.
    """
    spans: list[tuple[int, int, str]] = []
    index = 0
    while index < len(text):
        if text[index] != "$" or not is_boundary(text, index):
            index += 1
            continue
        end = _skill_name_end(text, index + 1)
        if end > index + 1:
            spans.append((index, end, text[index + 1 : end]))
        # ``max`` so a bare ``$`` (no usable name) still advances.
        index += max(end - index, 1)
    return spans


def _resolve_mention(
    text: str,
    start: int,
    end: int,
    exact: frozenset[str],
    by_lower: dict[str, str],
    *,
    leading: bool,
) -> tuple[int, str] | None:
    """``(end, discovered name)`` for the run, trimming trailing punctuation.

    The full run is tried first, then the name with trailing ``.``/``-``/``_``
    removed one character at a time. That is what makes ``$research.`` fire on
    ``research`` while a skill genuinely named with a dot still wins on the full
    run.

    CASE is the injection guard. An exact match always fires. A
    case-insensitive match fires only where there is positive evidence the user
    is naming a skill rather than an environment variable: at the LEADING
    position that is the sigil itself (legacy behaviour, preserved), and inline
    it is a lowercase letter in the token — so ``$PATH``/``$DEBUG`` cannot fire
    a ``path``/``debug`` skill mid-sentence.
    """
    cursor = end
    while True:
        candidate = text[start + 1 : cursor]
        if candidate in exact:
            return cursor, candidate
        if leading or any(char.islower() for char in candidate):
            lowered = by_lower.get(candidate.lower())
            if lowered is not None:
                return cursor, lowered
        if cursor > start + 2 and text[cursor - 1] in "._-":
            cursor -= 1
            continue
        return None


def skill_mentions(
    text: str,
    names: Collection[str],
    *,
    skip_spans: Sequence[tuple[int, int]] = (),
) -> list[SkillMention]:
    """Every ``$name`` in ``text`` that resolves to a discovered skill.

    THE trigger rule. The submit path, the composer's ink and the picker's
    evidence gate all answer "does this token fire" from here, so they cannot
    disagree about a token the user saw highlighted.

    - **Boundary** (:func:`is_boundary`): ``a$b``, ``costs$5`` and ``user@host``
      are not tokens; neither is a name in backticks, quotes or parentheses,
      because all of those put a non-space character before the ``$``.
    - **Whole text, not line by line**: a token after a newline is a token like
      any other, and only the buffer's first token is *leading*
      (``text[:start].strip()`` is empty).
    - **Dedupe by resolved name**, first-seen order: ``$research … $Research``
      attaches one body, not two.
    - The vocabulary decides: ``$100``, ``$5/unit`` and an unknown name are
      prose, exactly as before.

    ``skip_spans`` are regions that are already spoken for — a skill body in a
    block that arrived with the message. A token inside one is ignored BEFORE
    the dedupe, not after: a body quoting ``$research`` must not suppress a real
    `$research` later in the same message, and deduping first would let it.

    The leading rule is fixed: a leading token is resolved case-insensitively
    (the behaviour every pre-inline `$skill` line had), an inline one needs a
    lowercase letter. There is deliberately no switch — the ink and the submit
    path must answer with one rule, and a caller that scoped it would be the
    second rule this module exists to prevent.

    ``names`` is a collection of discovered names. Pure and host-free; the
    caller owns what a mention means.
    """
    exact = frozenset(names)
    by_lower: dict[str, str] = {}
    for name in names:
        by_lower.setdefault(name.lower(), name)
    mentions: list[SkillMention] = []
    seen: set[str] = set()
    for start, end, _raw in skill_name_spans(text):
        if any(begin <= start < finish for begin, finish in skip_spans):
            continue
        leading = not text[:start].strip()
        resolved = _resolve_mention(text, start, end, exact, by_lower, leading=leading)
        if resolved is None:
            continue
        token_end, name = resolved
        if name in seen:
            continue
        seen.add(name)
        mentions.append(
            SkillMention(
                start=start,
                end=token_end,
                text=text[start:token_end],
                name=name,
                leading=leading,
            )
        )
    return mentions


def defuse_skill_block(body: str) -> str:
    """Neutralize block markers appearing in a skill BODY.

    Same vector and same remedy as ``references._defuse``: a SKILL.md whose text
    quotes a marker would close the block early for :func:`skill_block_spans`,
    and the operator's prose after it would be stripped from the transcript row
    as if it were payload. A zero-width space inside each marker is invisible to
    a reader and stops the literal sequence the scanner matches. Nothing else
    about the body is escaped — the model must read it verbatim.
    """
    return (
        body.replace(SKILL_BLOCK_CLOSE, _DEFUSED_SKILL_CLOSE)
        .replace(SKILL_BLOCK_OPEN, _DEFUSED_SKILL_OPEN)
        .replace(_SKILL_ELEMENT_HEAD, _DEFUSED_ELEMENT_HEAD)
    )


def skill_block_spans(text: str) -> list[tuple[int, int]]:
    """Half-open ``[start, end)`` spans of every COMPLETE skill block.

    The exact counterpart of ``references.reference_block_spans``, and the same
    three parts are load-bearing: the open marker, the block's own preamble line
    (which separates an appended block from a quoted marker) and the closer. An
    unclosed opener reports nothing — but does not stop the scan, or it would
    pair with the NEXT block's closer and swallow the operator's prose between
    them. Rescanning from the next opener keeps that later block's own span.
    """
    spans: list[tuple[int, int]] = []
    opener = SKILL_BLOCK_OPEN + SKILL_BLOCK_JOIN + SKILL_BLOCK_PREAMBLE
    cursor = 0
    while True:
        start = text.find(opener, cursor)
        if start == -1:
            return spans
        close = text.find(SKILL_BLOCK_CLOSE, start + len(opener))
        nested = text.find(SKILL_BLOCK_OPEN, start + len(SKILL_BLOCK_OPEN))
        if close == -1 or (nested != -1 and nested < close):
            cursor = start + len(SKILL_BLOCK_OPEN)
            continue
        end = close + len(SKILL_BLOCK_CLOSE)
        spans.append((start, end))
        cursor = end


def skill_block_names(text: str) -> list[str]:
    """The ``<skill name=…>`` heads inside every complete block, in order.

    Read from the TOP-LEVEL element heads only: once an element is open, its
    body is skipped until the matching ``</skill>``. A body quoting a head is
    the skill documenting this feature, not a second attachment, and the test
    for that is structural rather than a docstring promise.

    Bodies are also defused on the way in (:func:`defuse_skill_block`), so a
    real payload never carries a spellable head inside one; this scan is the
    reader-side half of the same guarantee, for payloads written before that
    neutralisation existed.
    """
    names: list[str] = []
    for start, end in skill_block_spans(text):
        depth = 0
        for line in text[start:end].splitlines():
            if depth:
                if line.startswith("</skill>"):
                    depth -= 1
                continue
            head = _SKILL_ELEMENT_HEAD_RE.match(line)
            if head is not None:
                names.append(head.group(1))
                depth = 1
    return names
