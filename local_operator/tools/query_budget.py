"""A soft wall-clock budget for shell QUERIES: advisory at 10 s, stop at 60 s.

Why this exists
---------------
:mod:`local_operator.tools.search_guard` refuses a search whose *root* is
unbounded. It cannot see the shape that cost a live session seven minutes::

    for f in $(find ~/.local-operator/sessions -maxdepth 2 -name transcript.jsonl \\
              -mmin -720 | head -80); do grep -ql PATTERN "$f"; done

Every piece of that is locally reasonable — the ``find`` is depth-bounded, so the
static guard correctly lets it through — and the aggregate is a walk of the whole
store with a content search per hit, run to answer a question the store has an API
for (the ``sessions`` tool, and the digest index behind ``/resume``). Measured
here: 37.8 s over 80 transcripts, and still running when ``timeout(1)`` fired at
75 s on the fleet run of the wider shape.

So this module is the second half of the operator's norm — **query shallow, then
deepen on signals**. Shallow at plan time is ``search_guard``'s job. Noticing
that a query has stopped being shallow is this one's.

What it does
------------
A command is QUERY-SHAPED when it has at least one WALK-ISH occurrence and NO
LONG-RUNNER segment anywhere (see :func:`query_class` for both halves, and for
why the veto is load-bearing rather than a refinement). For such a command the
bash tool ticks alongside its existing 250 ms poll:

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
  something the agent did not ask for — the posture #1416 established.
* **Not a timeout.** A build, an install or a test suite is untouched (that is
  what the long-runner veto enforces); the tool's own ``timeout`` parameter and
  its default are unchanged, and a non-query command's semantics are identical to
  before this module existed. When BOTH apply the SMALLER bound wins: a budget
  above ``timeout`` never fires, because the tool's deadline kills the command
  first and reports ``TIMEOUT``.
* **Not a cost model.** It bounds one command group's WALL time, and it inherits
  the poll loop the tool already runs rather than adding one.
* **Not a clock.** The thresholds decide WHEN TO ASK THE MODEL TO NARROW; they
  are seconds for catastrophe rather than windows for precision (see AGENTS.md,
  "Timing, flakes").

Limits, stated rather than implied
----------------------------------
* **The grant is per CALL, not per segment** — once a segment carries it, the
  budget is off for the whole command, because the budget's tick is a property of
  the process group and there is no honest way to attribute an elapsed second to
  one stage of it. What IS enforced is WHERE it may be written: on a walk-ish
  segment, as a standalone statement, or as an ``env`` prefix of a walk
  (:func:`allow_slow_query`) — so a grant on an unrelated neighbour does not
  silently disarm the guard (review Q2).
* **A query hidden behind an opaque shell** — ``eval "$cmd"``, a script file, a
  string built at runtime — is not classified. ``bash -c '…'``/``sh -c '…'`` IS
  (they are parsed recursively, review Q3), which is the wrapper that matters in
  practice because it is also how a loop carries the grant.
"""

from __future__ import annotations

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

#: The inline grant. Read off the COMMAND's own assignments, never the process
#: environment — an inherited value would make the grant global and invisible in
#: the transcript, the rule ``search_guard.ALLOW_ENV`` set.
ALLOW_ENV = "LOCAL_OPERATOR_ALLOW_SLOW_QUERY"


@dataclass(frozen=True)
class Budget:
    """The resolved policy for one command: what the tool got from the three keys.

    Pure by construction, and that is the point: the THRESHOLDS are decisions a
    reviewer can read and a unit test can drive with a float, with no process and
    no clock anywhere near them. The bash tool owns only the wiring — an elapsed
    time in, a bool out.
    """

    enabled: bool = QUERY_BUDGET_ENABLED_DEFAULT
    stop: bool = QUERY_BUDGET_STOP_DEFAULT
    seconds: int = QUERY_BUDGET_SECONDS_DEFAULT

    def advisory_due(self, elapsed_s: float) -> bool:
        """Has the command crossed the soft mark? (Enabled, and 10 s of wall.)"""
        return self.enabled and elapsed_s >= ADVISORY_AFTER_S

    def stop_due(self, elapsed_s: float, *, allowed: bool = False) -> bool:
        """Should the command be killed now?

        ``allowed`` is the inline grant, resolved by :func:`allow_slow_query`. It
        skips the STOP and nothing else: the advisory is the signal the operator
        asked for, and a grant is permission to be slow, not permission to be
        invisible. A budget that is not a stop (``stop=False``) is a warn-only
        arm, which is what an operator watching before enforcing wants.
        """
        return self.enabled and self.stop and not allowed and elapsed_s >= self.seconds


#: How deep a ``bash -c '…'`` chain and a substitution are followed before the
#: classifier gives up. Bounded because the nesting is the model's, not ours.
_MAX_DEPTH = 3

#: A walker with no recursion flag to detect: it descends by construction, and
#: with no operand it starts at ``.``. Same set the static guard's ROOT rules
#: apply to, so the two arms of the norm agree about what a "walk" is.
_ALWAYS_WALKERS = frozenset({"find", "fd", "locate", "du"})

#: grep-family programs that walk only WHEN given a recursion flag. A bare
#: ``grep -c x f.txt`` reads one file and is not a query — which is what keeps
#: ``pytest; grep -c ok log`` out of the budget (review M2).
_RECURSION_FLAGGED = frozenset({"grep", "egrep", "fgrep"})

#: Programs that walk the tree by default but degrade to a stdin FILTER when they
#: are downstream of a pipe with no path operand (verified for ``rg``). ALIASED to
#: the static guard's set rather than restated: two copies of "what counts as a
#: stream filter" is how the pipe starts meaning two different things to one
#: command.
_STDIN_FILTER_PROGRAMS = search_guard._STDIN_FILTER_PROGRAMS

#: Shell keywords that introduce a COMMAND rather than being one, so the word
#: after them is the real command word: `do find …`, `then grep …`, and a
#: loop/branch header (`while find …`). Without the hop the measured loop body
#: classifies as the command `do`. `for` is deliberately absent: the word after it
#: is a VARIABLE, and the loop it introduces is covered by the substitution and
#: body arms.
_SHELL_PREFIXES = frozenset({"do", "then", "else", "if", "elif", "while", "until"})

#: Programs whose own runtime is the work — a build, a test run, an install, a
#: package manager, a container or a compiler. A command containing one is not a
#: query, however many walks sit beside it.
#:
#: WHY THIS VETO EXISTS, measured (review M2): `pytest -q tests/unit; grep -c
#: FAILED out.log` is a normal agent command. Classifying it "query-shaped" put
#: the whole command on a 60 s budget and killed a test suite with a message
#: blaming a filesystem walk. The classifier's own docstring used to claim a
#: false positive costs nothing "because a trivial command never reaches 10 s" —
#: true for `grep x f` alone, false the moment a 40-minute suite is the same
#: command. The veto is the fix, and it is deliberately generous: missing a
#: genuine query costs one slow search, while killing a build costs the turn.
LONG_RUNNERS = frozenset(
    {
        # build systems and task runners
        "make",
        "gmake",
        "ninja",
        "cmake",
        "bazel",
        "gradle",
        "mvn",
        # test runners
        "pytest",
        "tox",
        "nox",
        # language toolchains and package managers
        "npm",
        "pnpm",
        "yarn",
        "bun",
        "cargo",
        "rustc",
        "go",
        "dotnet",
        "swift",
        "xcodebuild",
        "pip",
        "pip3",
        "uv",
        "poetry",
        "conda",
        "rake",
        "bundle",
        "composer",
        # interpreters: their own run is a program, not a lookup
        "python",
        "python3",
        "node",
        "deno",
        "ruby",
        "perl",
        "php",
        "java",
        "javac",
        # infra and containers
        "docker",
        "podman",
        "kubectl",
        "helm",
        "terraform",
        "pulumi",
        "ansible",
    }
)

#: Deliberately NOT in the set, and the boundary is the interesting part: `git`,
#: `gh`, `curl`, `wget`, `rsync`, `ssh`, `aws`, `az`, `gcloud`, `cc`, `ld`, `ls`.
#: Each is normally SUB-SECOND, and a veto is not neutral — it suppresses the
#: budget for a genuine walk sitting beside it (`git commit -m x && grep -rn p
#: src/` is a real query, and so is `curl -s url | grep -rn p .`). The set carries
#: only programs whose ordinary invocation is minutes, because a false negative
#: costs a runaway search and a false positive costs a build.

#: Tokens that separate words inside a substitution body, for the walk-word scan.
_WORD_SPLIT_RE = re.compile(r"[\s;|&()<>]+")


def _command_word(text: str) -> tuple[str | None, str]:
    """``(effective command word, text after it)`` for one segment.

    Consumes the env assignments, a WRAPPER chain (``timeout 300``, ``sudo``,
    ``env X=1``) and a leading shell keyword (``do``), so the word returned is the
    one that would actually run. Sharing the wrapper peel with ``search_guard``
    keeps both guards reading the same word — the blind spot review m2 found is
    the same one in both.
    """
    word, rest = search_guard._peel(text)
    hops = 0
    while word in _SHELL_PREFIXES and hops < 4:
        word, rest = search_guard._peel(rest.lstrip())
        hops += 1
    return word, rest


def _is_walk_ish(segment: str, *, piped: bool) -> str | None:
    """The walk program this segment runs, or ``None``.

    The four cases, in the order the design states them: an always-walker; a
    grep-family word WITH a recursion flag (a pipe does not exempt these — a piped
    ``grep -r`` still walks its own path); a ripgrep-family word that is not
    serving as a stdin filter; and nothing else.
    """
    program, rest = _command_word(segment)
    if program is None:
        return None
    if program in _ALWAYS_WALKERS:
        return program
    if program in _RECURSION_FLAGGED:
        if search_guard._RECURSIVE_SHORT_RE.search(rest) or search_guard._RECURSIVE_LONG_RE.search(
            rest
        ):
            return program
        return None
    if program in _STDIN_FILTER_PROGRAMS:
        if piped and not search_guard._has_path_operand(segment):
            return None
        return program
    return None


def _has_long_runner(text: str, depth: int = 0) -> bool:
    """Does the work in ``text`` include a long-running program anywhere?"""
    if depth > _MAX_DEPTH:
        return False
    for segment, _piped, _op in search_guard._split(text):
        if not segment.strip():
            continue
        inner = search_guard._shell_c_inner(segment)
        if inner is not None:
            if _has_long_runner(inner, depth + 1):
                return True
            continue
        program, _rest = _command_word(segment)
        if program in LONG_RUNNERS:
            return True
    return False


def query_class(command: str, *, _depth: int = 0) -> str | None:
    """A short phrase naming what makes ``command`` a query, or ``None``.

    QUERY-SHAPED means: at least one WALK-ISH occurrence, and NO long-runner
    segment anywhere.

    A walk-ish occurrence is (a) a segment whose effective command word is an
    always-walker, a recursion-flagged grep, or a ripgrep-family program that is
    not filtering stdin; (b) the same inside a ``$(...)``/backtick substitution;
    or (c) the same as a loop body's command. ``bash -c '…'``/``sh -c '…'`` are
    parsed with these same rules rather than treated as opaque (review Q3).

    The trailing phrase is what the advisory and the stop embed, so it is written
    to read as the reason: "`grep`", "a substitution running `find`".
    """
    if _depth > _MAX_DEPTH or not command.strip():
        return None
    walk: str | None = None
    for segment, piped, _op in search_guard._split(command):
        if not segment.strip():
            continue
        inner = search_guard._shell_c_inner(segment)
        if inner is not None:
            if _has_long_runner(inner, _depth + 1):
                return None
            nested = query_class(inner, _depth=_depth + 1)
            if nested is not None:
                walk = walk or nested
            continue
        program, _rest = _command_word(segment)
        if program in LONG_RUNNERS:
            # THE VETO, and it is command-wide: a build or a test run anywhere in
            # the command means the elapsed seconds are not a query's, so no
            # segment of this command may put it on the query budget.
            return None
        found = _is_walk_ish(segment, piped=piped)
        if found is not None:
            walk = walk or f"`{found}`"
        for body in _substitution_bodies(segment):
            if _has_long_runner(body, _depth + 1):
                return None
            nested = query_class(body, _depth=_depth + 1)
            if nested is not None:
                walk = walk or f"a substitution running {nested}"
    return walk


def is_query_shaped(command: str) -> bool:
    """Is this command a filesystem walk rather than an ordinary command?

    Kept as its own name because it is the question the bash tool asks; the
    phrase :func:`query_class` returns is only for the message.
    """
    return query_class(command) is not None


def _assignment_prefix(segment: str) -> tuple[dict[str, str], str]:
    """Leading ``NAME=value`` assignments and the text after them.

    The order is swapped from ``search_guard._strip_env_assignments`` (which
    returns ``(text, assigns)``) because every caller here wants the MAPPING
    first; keeping one spelling of the unpack is what stops the two from being
    confused for each other.
    """
    text, assigns = search_guard._strip_env_assignments(segment)
    return assigns, text


def _export_grant(segment: str) -> bool:
    """Is this a standalone ``export LOCAL_OPERATOR_ALLOW_SLOW_QUERY=1``?"""
    words = search_guard._words(segment)
    if not words or search_guard._unquote(words[0]).strip() != "export":
        return False
    for word in words[1:]:
        token = search_guard._unquote(word)
        name, sep, value = token.partition("=")
        if sep and name == ALLOW_ENV and search_guard._truthy(value):
            return True
    return False


def _env_prefix(segment: str) -> tuple[dict[str, str], str]:
    """Assignments carried by a leading ``env`` and the command text after them."""
    _assigns, text = _assignment_prefix(segment)
    words = search_guard._words(text)
    if not words or search_guard._unquote(words[0]).strip() != "env":
        return {}, ""
    assigns: dict[str, str] = {}
    index = 1
    while index < len(words):
        token = search_guard._unquote(words[index])
        if token.startswith("-"):
            index += 1
            continue
        name, sep, value = token.partition("=")
        if not sep or not _ASSIGNMENT_NAME_RE.match(name):
            break
        assigns[name] = value
        index += 1
    return assigns, " ".join(words[index:])


_ASSIGNMENT_NAME_RE = re.compile(r"^[A-Za-z_][A-Za-z0-9_]*$")


def allow_slow_query(command: str, *, _depth: int = 0) -> bool:
    """Does the command grant itself the slow-query escape?

    Read off the command's own text and never the process environment — the rule
    ``search_guard.ALLOW_ENV`` follows, so the grant is visible in the transcript
    and cannot leak from an inherited value. Accepted in exactly three positions
    (review Q1/Q2):

    1. a leading assignment on a segment that WALKS — ``ALLOW=1 grep -rn p .``;
    2. a STANDALONE ``ALLOW=1`` / ``export ALLOW=1`` statement — the only form a
       compound command can carry, since ``ALLOW=1 for …`` is a bash syntax error
       and a loop header cannot take a prefix;
    3. an ``env ALLOW=1 …`` prefix of a walk.

    A grant that sits anywhere else — ``ALLOW=1 echo hi; <the loop>`` — is NOT
    honoured. It has to be written deliberately, and "deliberately" means next to
    the thing it excuses.
    """
    if _depth > _MAX_DEPTH:
        return False
    for segment, _piped, _op in search_guard._split(command):
        if not segment.strip():
            continue
        assigns, rest = _assignment_prefix(segment)
        if search_guard._truthy(assigns.get(ALLOW_ENV)):
            if not rest.strip():
                return True
            # The assignment must prefix the walk itself, not a neighbour.
            if query_class(segment, _depth=_depth + 1) is not None:
                return True
        if _export_grant(segment):
            return True
        env_assigns, env_rest = _env_prefix(segment)
        if search_guard._truthy(env_assigns.get(ALLOW_ENV)) and (
            not env_rest.strip() or query_class(segment, _depth=_depth + 1) is not None
        ):
            return True
    return False


def _read_balanced(text: str, open_index: int) -> tuple[str, int]:
    """Body of the ``(...)`` opened at ``open_index``, and the index after it.

    Quote-aware. An UNBALANCED body returns the rest of the text rather than
    nothing: the segment splitter cuts a command at a ``|`` even inside ``$( )``,
    so the half-substitution that reaches here is the common case, and dropping it
    would lose the walk it contains (the measured loop is exactly that shape).
    """
    depth = 0
    i = open_index
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
            quote = None if quote == '"' else '"'
            i += 1
            continue
        if quote is None and ch == "'":
            quote = "'"
            i += 1
            continue
        if ch == "(":
            depth += 1
        elif ch == ")":
            depth -= 1
            if depth == 0:
                return text[open_index + 1 : i], i + 1
        i += 1
    return text[open_index + 1 :], n


def _substitution_bodies(text: str) -> list[str]:
    """Bodies of the ``$(...)`` and backtick substitutions in ``text``.

    Quote-aware in both directions: a substitution inside SINGLE quotes is data
    and is skipped, while one inside DOUBLE quotes really runs and is collected.
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
            # Toggling, not opening: inside a double quote a `"` CLOSES it, and
            # the scan stays live in both states — `"$(find .)"` really runs.
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


def advisory_message(elapsed_s: float, cls: str, seconds: int) -> str:
    """The one live line for a query that has crossed :data:`ADVISORY_AFTER_S`.

    Plain text — the tool card paints Text, so markdown lands literally (the
    ``MEMORY_EXCEEDED_FALLBACK`` note in ``builtin`` says the same).
    """
    return (
        f"QUERY BUDGET: this shell query ({cls}) has run {elapsed_s:.0f}s. A scoped "
        f"query answers in milliseconds; this one is stopped at {seconds}s unless it "
        f"is narrowed or granted with {ALLOW_ENV}=1."
    )


def stop_message(elapsed_s: float, cls: str) -> str:
    """The stop receipt: what happened, why, and every way out of it.

    It names the WORKING grant spellings rather than a generic "prefix": the
    prefix form does not parse in front of a ``for`` loop (review Q1/m1), and a
    message that offers a command bash rejects is advice that costs a turn.
    """
    return (
        f"STOPPED AT SOFT QUERY BUDGET ({elapsed_s:.0f}s): this command runs a shell "
        f"query ({cls}) — a filesystem walk rather than a scoped read — and was "
        "killed at its budget so one search cannot spend minutes of a turn. Builds, "
        "installs and test runs are never budgeted; this one carried no such step.\n"
        "Narrow it and re-run:\n"
        "  - scope the root: a directory, a depth bound (find -maxdepth N), or one "
        "file, instead of a whole tree;\n"
        "  - prefer the grep/glob tools — recursive, .gitignore-aware and pruned — "
        "over a shell walk;\n"
        "  - bound the output with --max-count N or a pipe to head;\n"
        "  - for a conversation lookup use the sessions tool or the /resume search: "
        "both read a digest index, not every transcript;\n"
        f"  - a justified slow query: put {ALLOW_ENV}=1 on the search itself, or as "
        f"its own statement before a loop ({ALLOW_ENV}=1; for f in ...); the "
        "advisory still shows, the stop is skipped;\n"
        "  - or change the budget in /settings (bash.query_budget.enabled/stop/seconds)."
    )
