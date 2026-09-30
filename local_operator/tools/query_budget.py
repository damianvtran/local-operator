"""A soft wall-clock budget for shell QUERIES: advisory at 10 s, stop at 60 s.

Why this exists
---------------
:mod:`local_operator.tools.search_guard` refuses a search whose *root* is
unbounded. It cannot see the shape that cost a live session seven minutes::

    for f in $(find ~/.local-operator/sessions -maxdepth 2 -name transcript.jsonl \\
              -mmin -720 | head -80); do grep -ql PATTERN "$f"; done

Every piece of that is locally reasonable — the ``find`` is depth- AND
time-bounded, so the static guard correctly lets it through — and the aggregate
is a walk of the whole store with a content search per hit, run to answer a
question the store has an API for (the ``sessions`` tool, and the digest index
behind ``/resume``). Measured here: the same loop over 80 transcripts took 37.8 s
and was still running when ``timeout(1)`` fired at 75 s in an earlier fleet run
of the wider shape; nothing in the harness noticed either one.

So this module is the second half of the operator's norm — **query shallow, then
deepen on signals**. Shallow at plan time is ``search_guard``'s job (refuse the
unbounded root; make the shallow shape the easy one to write). Noticing that a
query has stopped being shallow is this one's.

What it does
------------
For a command classified as QUERY-SHAPED (:func:`query_class`), the bash tool
ticks alongside its existing 250 ms poll:

* at :data:`ADVISORY_AFTER_S` (10 s) it emits ONE advisory line, on the live
  update channel and carried into the result;
* at the configured budget (:data:`QUERY_BUDGET_SECONDS_DEFAULT`, 60 s) it KILLS
  the command's process group and says so in the result — unless the command
  carries the inline grant ``LOCAL_OPERATOR_ALLOW_SLOW_QUERY=1``, which skips the
  stop but NOT the advisory.

Both paths are VISIBLE. An advisory that existed only in a log would be exactly
the silent multi-minute query this exists to end, so the line rides the same
channels the result and the live card already carry.

What it deliberately does NOT do
--------------------------------
* **Not a rewrite.** The command is stopped and re-explained, never edited into
  something the agent did not ask for — the posture #1416 established for the
  static guard, and for the same reason: substituting a path back into arbitrary
  shell silently changes what was asked for.
* **Not a timeout.** A build, an install or a test suite is untouched; the
  tool's own ``timeout`` parameter and its default are unchanged, and a
  non-query command's semantics are identical to before this module existed.
* **Not a cost model.** It bounds one command group's WALL time, and it inherits
  the poll loop the tool already runs rather than adding one. A query that is
  cheap and long (a ``find`` over a slow network mount) is stopped the same as a
  runaway one; that is the intended trade, and the inline grant is the escape.
* **Not a clock.** Nothing here measures speed for its own sake: the thresholds
  are decisions about WHEN TO ASK THE MODEL TO NARROW, and they are seconds for
  catastrophe rather than windows for precision (see AGENTS.md, "Timing,
  flakes").

Classification errs toward INCLUDING a command, and that is safe by
construction: a trivial command never reaches 10 s, so a false positive costs
nothing, while a false negative costs the seven minutes above.
"""

from __future__ import annotations

import os
import re
from dataclasses import dataclass

from local_operator.tools import search_guard

# ---------------------------------------------------------------------------
# Settings, next to the code that reads them. ``path`` mirrors the registry rows
# in ``settings_io.py`` (pinned together by
# ``test_query_budget_rows_share_the_consumer_paths``), and the reader in
# ``tools/builtin.py`` resolves an absent key against exactly these constants.
# ---------------------------------------------------------------------------

QUERY_BUDGET_ENABLED_PATH: tuple[str, ...] = ("bash", "query_budget", "enabled")
QUERY_BUDGET_STOP_PATH: tuple[str, ...] = ("bash", "query_budget", "stop")
QUERY_BUDGET_SECONDS_PATH: tuple[str, ...] = ("bash", "query_budget", "seconds")

#: Master switch. Off means no advisory and no stop, for every command.
QUERY_BUDGET_ENABLED_DEFAULT = True
#: Off turns the stop into a warn-only arm: the advisory still fires, nothing is
#: killed. Separate from ``enabled`` so an operator can watch before enforcing.
QUERY_BUDGET_STOP_DEFAULT = True
#: The stop threshold. Deliberately well below ``bash.timeout``'s 120 s default:
#: the budget is about the COMMAND SHAPE, and a query still running at a minute
#: is not going to get narrow on its own.
QUERY_BUDGET_SECONDS_DEFAULT = 60

#: When the advisory fires. Ten seconds is "this is no longer a lookup": the
#: scoped shapes the static guard steers to measure in milliseconds, and the
#: whole-store loops this exists for measure in minutes.
ADVISORY_AFTER_S = 10.0

#: The inline grant, read per SEGMENT off the command's own assignments — the
#: same convention as ``search_guard.ALLOW_ENV`` and ``sleep_guard.ALLOW_ENV``.
#: An inherited process-environment value is deliberately NOT consulted: it would
#: make the grant global and invisible in the transcript.
ALLOW_ENV = "LOCAL_OPERATOR_ALLOW_SLOW_QUERY"

#: The programs whose presence makes a command a query. These are the walkers
#: ``search_guard`` reasons about, plus ``du`` — the set is shared with the
#: static guard's root rules on purpose, so the two arms of the norm agree about
#: what a "search/walk" is.
WALK_PROGRAMS = frozenset(
    {"grep", "egrep", "fgrep", "rg", "ripgrep", "ag", "ack", "find", "fd", "locate", "du"}
)

#: Shell keywords that introduce a COMMAND rather than being one, so the word
#: after them is the real command word: `do grep …`, `then grep …`, and the same
#: for a loop or branch header (`while grep -q …`). Without these, the measured
#: loop body — `do grep -ql PATTERN "$f"` — classifies as the command `do`.
_SHELL_PREFIXES = frozenset({"do", "then", "else", "if", "elif", "while", "until"})

#: Tokens that separate words inside a substitution body. A plain ``str.split``
#: is not enough because the body is arbitrary shell: `$(find . | head)` has a
#: pipeline in it, and the walk word is what matters, not its position.
_BODY_SPLIT_RE = re.compile(r"[\s;|&()<>]+")


@dataclass(frozen=True)
class Budget:
    """The resolved ``bash.query_budget`` decision for one command.

    Pure and frozen, so the two thresholds are unit targets with no process,
    no clock and no config in sight — the same shape ``memory_guard.Budget``
    uses for its ceiling.
    """

    enabled: bool = QUERY_BUDGET_ENABLED_DEFAULT
    stop: bool = QUERY_BUDGET_STOP_DEFAULT
    seconds: int = QUERY_BUDGET_SECONDS_DEFAULT

    def advisory_due(self, elapsed_s: float) -> bool:
        """Has this command run long enough to be told it is a long query?"""
        return self.enabled and elapsed_s >= ADVISORY_AFTER_S

    def stop_due(self, elapsed_s: float, *, allowed: bool = False) -> bool:
        """Should this command's group be killed now?

        ``allowed`` is the inline grant: it suppresses the STOP only. The
        advisory is a statement of fact (the command has run this long) and is
        still true under a grant, which is why the two are separate predicates
        rather than one verdict enum.
        """
        return self.enabled and self.stop and not allowed and elapsed_s >= self.seconds


def _first_walk_word(body: str) -> str | None:
    """The first walk program named in ``body``, or None.

    Best-effort and quote-blind inside the body: `$(echo "find")` reads as a
    walk. That direction is the safe one — see the module docstring on why a
    false positive costs nothing and a false negative costs minutes.
    """
    for token in _BODY_SPLIT_RE.split(body):
        name = os.path.basename(token.strip("'\""))
        if name in WALK_PROGRAMS:
            return name
    return None


def _read_balanced(text: str, open_paren_index: int) -> tuple[str, int]:
    """``(body, end_index)`` for a ``$(`` at ``open_paren_index``.

    Unbalanced (the segment was cut by a top-level pipe or ``;`` the shared
    splitter recognised first — which is exactly what happens to
    ``$(find … | head)``) yields the rest of the text, because a truncated
    substitution still contains the walk word that matters.
    """
    depth = 0
    i = open_paren_index
    n = len(text)
    while i < n:
        ch = text[i]
        if ch == "(":
            depth += 1
        elif ch == ")":
            depth -= 1
            if depth == 0:
                return text[open_paren_index + 1 : i], i + 1
        elif ch == "\\":
            i += 1
        i += 1
    return text[open_paren_index + 1 :], n


def _substitution_bodies(text: str) -> list[str]:
    """The bodies of ``$(…)`` and backtick substitutions, quote-aware.

    Single quotes suppress substitution (``echo '$(find .)'`` is data), double
    quotes do not (``echo "$(find .)"`` runs it), and a backslash escapes the
    next character — the same reading ``search_guard``'s splitter applies to
    separators, applied here to substitution openers.
    """
    out: list[str] = []
    i = 0
    n = len(text)
    quote: str | None = None
    while i < n:
        ch = text[i]
        if quote == "'":
            if ch == "'":
                quote = None
            i += 1
            continue
        if ch == "\\":
            i += 2
            continue
        if ch == '"':
            # Toggling, not opening: inside a double quote a `"` CLOSES it,
            # and the substitution scan below stays live in both states —
            # `"$(find .)"` really does run the find.
            quote = None if quote == '"' else '"'
            i += 1
            continue
        if quote is None and ch == "'":
            quote = "'"
            i += 1
            continue
        if ch == "$" and text[i + 1 : i + 2] == "(":
            body, i = _read_balanced(text, i + 1)
            out.append(body)
            continue
        if ch == "`":
            end = text.find("`", i + 1)
            body = text[i + 1 : end if end != -1 else n]
            out.append(body)
            i = end + 1 if end != -1 else n
            continue
        i += 1
    return out


def _command_word(segment: str) -> str | None:
    """The command word of a segment, past leading assignments and keywords."""
    text, _assigns = search_guard._strip_env_assignments(segment)
    if not text:
        return None
    word, rest = search_guard._read_word(text)
    hops = 0
    while word in _SHELL_PREFIXES and hops < 4:
        # ``_read_word`` leaves the remainder's leading whitespace in place, so
        # each hop is taken from a stripped remainder — otherwise the second hop
        # reads the space and returns an empty word.
        word, rest = search_guard._read_word(rest.lstrip())
        hops += 1
    if not word:
        return None
    return os.path.basename(search_guard._unquote(word))


def _segment_query_class(segment: str, *, piped: bool) -> str | None:
    """What makes THIS segment a query, or None.

    ``piped`` is load-bearing: a stage downstream of a pipe reads the previous
    stage's stdout, so `pytest | grep fail` is an ordinary filter and not a
    query. Its command word is checked only for a genuine walk, and the
    substitution arm still applies — `… | $(find …)` would walk.
    """
    if not segment.strip():
        return None
    body = _substitution_bodies(segment)
    for candidate in body:
        walk = _first_walk_word(candidate)
        if walk is not None:
            return f"a $() substitution running {walk}"
    if piped:
        return None
    word = _command_word(segment)
    if word is not None and word in WALK_PROGRAMS:
        return word
    return None


def query_class(command: str) -> str | None:
    """A short phrase naming what makes ``command`` a query, or None.

    The phrase goes into the advisory and the stop, so it is written to be read
    as the reason: "`grep`", "a $() substitution running `find`".
    """
    for segment, piped, _terminator in search_guard._split(command):
        cls = _segment_query_class(segment, piped=piped)
        if cls is not None:
            return cls
    return None


def is_query_shaped(command: str) -> bool:
    """Is this command a filesystem walk/search rather than an ordinary command?

    Kept as its own name because it is the question the bash tool asks; the
    phrase :func:`query_class` returns is only for the message.
    """
    return query_class(command) is not None


def allow_slow_query(command: str) -> bool:
    """Does the command grant itself the slow-query escape, per SEGMENT?

    Read off the command's own leading assignments, never the process
    environment — the same rule ``search_guard.ALLOW_ENV`` follows, so the grant
    is visible in the transcript and cannot leak from an inherited value.
    """
    for segment, _piped, _terminator in search_guard._split(command):
        _stripped, assigns = search_guard._strip_env_assignments(segment)
        if search_guard._truthy(assigns.get(ALLOW_ENV)):
            return True
    return False


def advisory_message(elapsed_s: float, cls: str, seconds: int) -> str:
    """The one live line for a query that has crossed :data:`ADVISORY_AFTER_S`.

    Plain text — the tool card paints Text, so markdown lands literally (the
    ``MEMORY_EXCEEDED_FALLBACK`` note in ``builtin`` says the same).
    """
    return (
        f"QUERY BUDGET: this shell query ({cls}) has run {elapsed_s:.0f}s. A scoped "
        f"query answers in milliseconds; this one is stopped at {seconds}s unless it "
        f"is narrowed or prefixed with {ALLOW_ENV}=1."
    )


def stop_message(elapsed_s: float, cls: str) -> str:
    """The stop receipt: what happened, why, and every way out of it."""
    return (
        f"STOPPED AT SOFT QUERY BUDGET ({elapsed_s:.0f}s): this command runs a shell "
        f"query ({cls}) — a filesystem walk rather than a scoped read — and was "
        "killed at its budget so one search cannot spend minutes of a turn.\n"
        "Narrow it and re-run:\n"
        "  - scope the root: a directory, a depth bound (find -maxdepth N), or a time "
        "filter (-mmin -60) instead of a whole tree;\n"
        "  - prefer the grep/glob tools — recursive, .gitignore-aware and pruned — "
        "over a shell walk;\n"
        "  - bound the output with --max-count N or a pipe to head;\n"
        "  - for a conversation lookup use the sessions tool or the /resume search: "
        "both read a digest index, not every transcript;\n"
        f"  - a justified slow query: prefix it with {ALLOW_ENV}=1 (the advisory still "
        "shows, the stop is skipped);\n"
        "  - or change the budget in /settings (bash.query_budget.enabled/stop/seconds)."
    )
