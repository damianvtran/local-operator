"""The sigil grammar — where a ``/``, ``$`` or ``@`` token starts, and what it spans.

Two surfaces read the same grammar. The composer
(``tui/widgets/command_picker.py``) reads it on every keystroke to decide
whether the character just typed opens a picker, and the submit-side resolver
(``local_operator/references.py``) reads it again on the finished draft to
decide which spans to expand. They must agree exactly: a boundary rule that
holds in the composer but not at submit time is a token the user saw
highlighted and the agent never received.

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
