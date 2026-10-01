"""Block unbounded shell searches and steer them to the search tools.

Why this exists
---------------
Agents routinely bypass the ``grep``/``glob`` tools and shell out to a raw
recursive search from the top of a repository, where it walks vendored trees
that no one wants searched. Measured on the operator's own live session
``1418a2d96931``::

    grep -rn "was replaced while this app was running" --include=*.ts .   # 41 s

run in a repo whose root holds ``node_modules`` (5.0 GB / 110 602 files),
``.git`` (2.8 GB) and ``.worktrees`` (5.6 GB) against a 662-file source tree.
The same search scoped to ``src/`` is 0.06 s, and the ``grep`` TOOL — which
already carries ripgrep, .gitignore semantics and a fixed prune set — is
faster still. Nothing routed the model to it.

What it does, and what it deliberately refuses to do
----------------------------------------------------
It *blocks with a suggestion*, and never rewrites the command. A rewrite has to
substitute a path back into arbitrary shell (arrays, ``<(...)``, heredocs,
``$'...'``, backticks); a rewrite that gets that wrong silently changes what the
agent asked for, which is worse than a refusal. A refusal is deterministic, says
why, and leaves full expression intact through an explicit escape hatch.

The predicate is narrow on purpose. A segment is blocked only when ALL of these
hold, so the legitimate shapes pass untouched:

1. its first word is a recursive-capable search binary
   (``grep egrep fgrep rg ripgrep ag ack find fd locate``) — ``git grep`` is
   exempt because it is structured and honours .gitignore — or a walker that
   needs no recursive flag at all (``du``, which descends by construction and
   with no operand starts at ``.``);
2. a recursive flag is present (``-r``/``-R``/bundled ``-rn``), or it is
   ``find``/``fd`` with a name/type/path predicate and no ``-maxdepth``;
3. the search *root* is unbounded: no path operand at all, or a path that is
   ``.``/``./``/``..``/``/``/``~``, a bare glob, an unresolved ``$VAR``/``$(...)``,
   a known-heavy directory (``node_modules``, ``.git``, ``out``, ...), the
   SESSION STORE (``~/.local-operator`` and its ``sessions`` subtree), or a
   REPOSITORY ROOT (a resolved directory holding ``.git`` or ``node_modules``
   as a direct child).

A single named file, a scoped directory (``grep -rn PATTERN src/``), a piped
stage that reads stdin (``... | grep -v node_modules``, ``... | rg PATTERN``) and
a quoted mention (``echo "grep -rn x ."``) all pass. So does ``find``/``fd``
against a NAMED directory (``find ~/Downloads -name '*.png'``) — the author chose
that scope — and, at an unbounded/store/repo root, any ``find``/``fd`` that is
SHALLOW or TIME-BOUNDED (``-maxdepth``, or ``-mmin``/``-mtime``/``-newer``):
``find ~/.local-operator/sessions -maxdepth 2 -name transcript.jsonl -mmin -720``
is the shape agents should write, so it passes. Those are exactly the
false-positive classes the tests pin, and the segment splitter is quote- and
escape-aware so shell syntax the model writes never reads as a second command.

The two new root classes exist because "a named directory is the author's own
scope" is true for ``~/Downloads`` and false for a checkout or the store: a walk
of either is the multi-minute query this guard family exists to stop, and the
store one has an API (the ``sessions`` tool, and the digest index behind
``/resume``) that answers the same question in milliseconds.

Escape hatch: prepend ``LOCAL_OPERATOR_ALLOW_UNBOUNDED_SEARCH=1`` to the command
to run it as written. That is a per-call grant read off the command itself, so
the agent keeps full expression without a config edit.
"""

from __future__ import annotations

import os
import re
from dataclasses import dataclass

#: The env var an agent sets inline to run an unbounded search as written.
ALLOW_ENV = "LOCAL_OPERATOR_ALLOW_UNBOUNDED_SEARCH"

#: Programs that search file contents or the filesystem recursively. ``git`` is
#: NOT here: ``git grep`` is matched separately and exempted, because it is
#: structured, revspec-bounded and honours .gitignore.
SEARCH_PROGRAMS = frozenset(
    {"grep", "egrep", "fgrep", "rg", "ripgrep", "ag", "ack", "find", "fd", "locate"}
)

#: Directory basenames that make a recursion expensive rather than useful. This
#: is the shell-side mirror of ``builtin._GREP_PRUNE_DIRS``, widened by the
#: build/vendor trees the greps in the wild actually hit.
HEAVY_DIRS = frozenset(
    {
        "node_modules",
        ".git",
        "out",
        "dist",
        "build",
        ".venv",
        "venv",
        "vendor",
        "target",
        "coverage",
        ".next",
        ".worktrees",
        ".tox",
        "__pycache__",
    }
)

#: Bundled short flags containing a recursive ``r``/``R`` (``-rn``, ``-rni``,
#: ``-R``, ``-ir``). ``--recursive`` is matched separately.
_RECURSIVE_SHORT_RE = re.compile(r"(?<![\w-])-[A-Za-z]*[rR][A-Za-z]*(?![\w-])")
_RECURSIVE_LONG_RE = re.compile(r"(?<![\w-])--recursive(?![\w-])")
#: A find/fd predicate that means "walk for these files" (as opposed to a bare
#: `find .`), paired with the absence of ``-maxdepth`` below.
_FIND_PREDICATE_RE = re.compile(r"(?<![\w-])-(?:name|iname|path|ipath|type|regex)(?![\w-])")
_MAXDEPTH_RE = re.compile(r"(?<![\w-])-maxdepth(?![\w-])")
#: A token that is only glob metacharacters (``*``, ``*.ts`` is NOT bare — it
#: has a stem — but ``*`` and ``**`` are).
_BARE_GLOB_RE = re.compile(r"^[*?]+$")
#: Leading ``NAME=value`` assignments on a segment.
_ENV_ASSIGN_RE = re.compile(r"^[A-Za-z_][A-Za-z0-9_]*=")
#: An operand that never resolves to a concrete, bounded location: ``$VAR``,
#: ``$(...)``, a backtick substitution, or a brace expansion.
_UNRESOLVED_RE = re.compile(r"[`$]|\$\(")

#: Search programs that recurse BY DEFAULT, so the absence of a `-r` flag says
#: nothing about the scope. `rg`/`ag`/`ack` walk the cwd unless given a path;
#: `fd`/`locate` walk the cwd (fd) or the whole filesystem index (locate) with
#: no path at all. `find` is deliberately NOT here — `find` needs a path or its
#: own predicate to do anything, which the predicate branch below handles.
_IMPLICITLY_RECURSIVE = frozenset({"rg", "ripgrep", "ag", "ack"})

#: Walkers that are recursive with no path operand and must not be exempted as a
#: pipe filter: `fd`/`locate` always walk. `find` is added to this reasoning by
#: its own predicate branch rather than the flag scan.
_ALWAYS_RECURSIVE = frozenset({"fd", "locate"})

#: Programs that walk the tree BY CONSTRUCTION and have no recursive flag to
#: scan for. `du` sums disk usage as it descends and, given no operand at all,
#: starts at `.` exactly as `find` does — which is why it is a walker here and
#: not a plain command. Kept OUT of ``SEARCH_PROGRAMS`` (whose members all have a
#: recursion flag the classifier reads) and folded into :data:`ALL_PROGRAMS`
#: instead, so the shared predicates that *are* flag-driven never see it.
WALK_BY_CONSTRUCTION = frozenset({"du"})

#: Every program this guard's ROOT rules apply to. A root class (store, repo,
#: unbounded) is a property of the PATH, not of the program, so it is decided
#: once and shared by the whole family.
ALL_PROGRAMS = SEARCH_PROGRAMS | WALK_BY_CONSTRUCTION

#: ``fd``'s own spelling of a depth bound (``-d N``, ``--max-depth N``). Accepted
#: for ``fd`` ALONE: in GNU ``find``, ``-depth`` is post-order traversal — it
#: changes the ORDER of the walk, not its size, so accepting ``-d`` for `find`
#: would trade a refusal for a slower walk.
_FD_DEPTH_RE = re.compile(r"(?<![\w-])(?:-d|--max-depth)(?![\w-])")

#: A LITERAL duration, for the wrapper peel below (``timeout 300 find …``).
_DURATION_RE = re.compile(r"^\d+(?:\.\d+)?[smhd]?$")

#: Programs that run ANOTHER command, so the command word is theirs and not the
#: first word's. Without this the guards have a blind spot precisely where an
#: agent already knows it is about to be slow: `timeout 300 find / …`,
#: `sudo find / …`, `env X=1 grep -rn …` all read as their wrapper (review m2).
#: `bash -c`/`sh -c` are handled separately, because their argument is a COMMAND
#: STRING that has to be parsed again rather than skipped.
WRAPPER_WORDS = frozenset({"timeout", "sudo", "env", "time", "nice", "command", "nohup", "xargs"})

#: Short flags of a wrapper that take their value as the NEXT token, so the value
#: is not read as the command word (``timeout -s KILL 300 find …``).
_WRAPPER_VALUE_FLAGS: dict[str, frozenset[str]] = {
    "timeout": frozenset({"s", "k"}),
    "sudo": frozenset({"u", "g", "p", "C", "h", "r", "t", "U", "D"}),
    "nice": frozenset({"n"}),
    "xargs": frozenset({"I", "n", "P", "s", "L", "a", "d", "E"}),
}

#: The same, spelled long and with a space rather than an ``=``.
_WRAPPER_LONG_VALUE_FLAGS = frozenset(
    {
        "--signal",
        "--kill-after",
        "--user",
        "--group",
        "--prompt",
        "--chdir",
        "--adjustment",
        "--max-args",
        "--max-lines",
        "--max-chars",
        "--delimiter",
        "--arg-file",
    }
)

#: Shells whose ``-c`` argument is a command string. Their inner text is parsed
#: with the SAME rules, recursively — the wrapper an agent reaches for to make a
#: loop carry an inline grant (review Q3/m2).
_SHELL_C_WORDS = frozenset({"bash", "sh", "zsh", "dash", "ksh", "ash"})

#: The deepest a ``bash -c '…'`` chain is followed. Bounded because the nesting is
#: written by the model, not by us: a guard that recurses without a floor can be
#: made to recurse for as long as the command is long.
_MAX_PEEL_DEPTH = 3

#: The store's directory name and the env var that can relocate it. Deliberately
#: a literal rather than an import of ``local_operator.paths``: this module is a
#: leaf (stdlib only) so the guard can be reasoned about — and tested — without
#: the config layer, and ``paths.config_dir()`` resolves exactly these two terms.
STORE_DIRNAME = ".local-operator"
#: The subtree that actually holds conversations. The store class is scoped to it
#: (plus the root itself) rather than to the whole config root — see
#: :func:`_is_store_root`.
_SESSIONS_DIRNAME = "sessions"
CONFIG_DIR_ENV = "LOCAL_OPERATOR_CONFIG_DIR"
#: The agent's scratch tree. It lives UNDER the store (``sessions/<id>/scratchpad``
#: on this install, or wherever ``LOCAL_OPERATOR_SCRATCHPAD`` points), and it is
#: the one part of the store an agent is TOLD to work in — so it is exempt from
#: the store class rather than lumped in with the transcripts (review M4).
SCRATCHPAD_DIRNAME = "scratchpad"
SCRATCHPAD_ENV = "LOCAL_OPERATOR_SCRATCHPAD"
#: A store root written as a shell would spell it: ``~/.local-operator``,
#: ``$HOME/.local-operator`` or ``${HOME}/.local-operator``. Checked before the
#: unresolved rule so the STORE nudge (which names the `sessions` tool) is what
#: the model reads, instead of the generic unresolved-root one. The TAIL decides
#: whether it is the store: the root itself and ``sessions/`` are, and a smaller
#: child (``skills/``, ``attachments/``, the scratchpad) is not.
_STORE_STR_RE = re.compile(r"^(?:~|\$HOME|\$\{HOME\})/\.local-operator")

#: A heredoc opener: `<<EOF`, `<<-EOF`, `<<'EOF'`, `<<"EOF"`.
_HEREDOC_RE = re.compile(r"<<-?\s*(['\"]?)([A-Za-z_][A-Za-z0-9_]*)\1")


def _strip_heredocs(command: str) -> str:
    """Blank out heredoc BODIES so their text is never read as commands.

    `cat <<EOF … grep -rn x . … EOF` is one command whose body is data; a
    splitter that reads the body sees a second segment and blocks a shell that
    would never run it. Each body is replaced by a single space.

    Quote-aware, and FAIL-CLOSED. Quote tracking matters because `<<` inside a
    quoted string is not an operator: `git commit -m 'fix << a' && grep -rn p .`
    is a normal command, and a scanner that treated the quoted `<<` as an opener
    would swallow everything after it — silently dropping the unbounded grep
    (review M3). An opener with no terminating delimiter line is left IN PLACE
    rather than truncated, so the worst case is a command the guard still sees.
    """
    out: list[str] = []
    i = 0
    n = len(command)
    quote: str | None = None
    while i < n:
        ch = command[i]
        if quote is not None:
            out.append(ch)
            if ch == "\\" and i + 1 < n:
                out.append(command[i + 1])
                i += 2
                continue
            if ch == quote:
                quote = None
            i += 1
            continue
        if ch in ("'", '"'):
            quote = ch
            out.append(ch)
            i += 1
            continue
        if ch == "\\" and i + 1 < n:
            out.append(ch)
            out.append(command[i + 1])
            i += 2
            continue
        if ch == "<" and command[i + 1 : i + 2] == "<":
            m = _HEREDOC_RE.match(command, i)
            if m is not None:
                delim = m.group(2)
                nl = command.find("\n", m.end())
                if nl != -1:
                    pos = nl + 1
                    delim_at = -1
                    while pos <= n:
                        line_end = command.find("\n", pos)
                        line = command[pos : line_end if line_end != -1 else n]
                        if line.strip() == delim:
                            delim_at = line_end if line_end != -1 else n
                            break
                        if line_end == -1:
                            break
                        pos = line_end + 1
                    if delim_at != -1:
                        # Keep the opener, drop the body, resume at the delimiter.
                        out.append(command[i : nl + 1])
                        out.append(" ")
                        i = delim_at
                        continue
                # No terminator: leave the opener verbatim (fail closed).
            out.append(ch)
            i += 1
            continue
        out.append(ch)
        i += 1
    return "".join(out)


def _segments(command: str) -> list[tuple[str, bool]]:
    """Split ``command`` into flat command segments.

    Returns ``(text, consumes_stdin)`` per segment. ``consumes_stdin`` is True
    when the segment follows a single ``|`` (a pipe), which means it reads the
    previous stage's stdout — no path-based dedicated tool can replace a stream
    filter, so such a segment is never a block candidate.

    Splitting is quote- and escape-aware: a ``;``/``|``/``&`` inside quotes or
    behind a backslash is content, not an operator. The boundary is registered
    on the SEGMENTS so a piped downstream stage is distinguishable from a fresh
    command after ``&&``/``;``/``||``/``&``/newline.
    """
    return [(text, piped) for text, piped, _ in _split(command)]


def _split(command: str) -> list[tuple[str, bool, str]]:
    """:func:`_segments` plus the operator run that ENDED each segment.

    The third field is ``"&"``, ``"&&"``, ``";"``, ``"|"``, ``"||"``, ``"\\n"``
    (a run of newlines collapses to one) or ``""`` for the last segment. The
    search guard never needs it; ``sleep_guard`` does, because a segment ended
    by a single ``&`` runs in the background of the shell and blocks nothing —
    which is the one fact about a ``sleep`` that decides whether it holds the
    tool call. One splitter serves both guards so they cannot disagree about
    where a command ends.
    """
    command = _strip_heredocs(command)
    out: list[tuple[str, bool, str]] = []
    buf: list[str] = []
    quote: str | None = None
    piped = False
    i = 0
    n = len(command)
    while i < n:
        ch = command[i]
        if quote is not None:
            buf.append(ch)
            if ch == quote:
                quote = None
            elif ch == "\\" and quote == '"' and i + 1 < n:
                buf.append(command[i + 1])
                i += 2
                continue
            i += 1
            continue
        if ch in ("'", '"'):
            quote = ch
            buf.append(ch)
            i += 1
            continue
        if ch == "\\":
            buf.append(ch)
            if i + 1 < n:
                buf.append(command[i + 1])
            i += 2
            continue
        if ch in ";|&\n":
            # Collapse runs (``&&``, ``||``) and note a single ``|`` pipe.
            j = i
            while j < n and command[j] == ch:
                j += 1
            text = "".join(buf).strip()
            if text:
                out.append((text, piped, command[i:j] if ch != "\n" else "\n"))
            buf = []
            piped = ch == "|" and (j - i) == 1
            i = j
            continue
        buf.append(ch)
        i += 1
    text = "".join(buf).strip()
    if text:
        out.append((text, piped, ""))
    return out


def _strip_env_assignments(segment: str) -> tuple[str, dict[str, str]]:
    """Drop leading ``NAME=value`` assignments; return them and the remainder.

    Only leading assignments are environment: once the first non-assignment
    word appears, later ``a=b`` tokens are operands. Values are unquoted so the
    caller can read the escape hatch off the command itself.
    """
    assigns: dict[str, str] = {}
    rest = segment.lstrip()
    while True:
        m = _ENV_ASSIGN_RE.match(rest)
        if not m:
            break
        name = rest[: m.end() - 1]
        value, after = _read_word(rest[m.end() :])
        assigns[name] = value
        rest = after.lstrip()
        if not rest:
            break
    return rest, assigns


def _read_word(text: str) -> tuple[str, str]:
    """Read one shell word, returning ``(unquoted_value, remainder_after_word)``."""
    out: list[str] = []
    quote: str | None = None
    i = 0
    n = len(text)
    while i < n:
        ch = text[i]
        if quote is not None:
            if ch == quote:
                quote = None
            elif ch == "\\" and quote == '"' and i + 1 < n:
                out.append(text[i + 1])
                i += 1
            else:
                out.append(ch)
            i += 1
            continue
        if ch in ("'", '"'):
            quote = ch
            i += 1
            continue
        if ch == "\\" and i + 1 < n:
            out.append(text[i + 1])
            i += 2
            continue
        if ch.isspace():
            break
        out.append(ch)
        i += 1
    return "".join(out), text[i:]


def _unquote(token: str) -> str:
    """Strip surrounding quotes from an already-whole token."""
    if len(token) >= 2 and token[0] == token[-1] and token[0] in ("'", '"'):
        return token[1:-1]
    return token


def _program(segment: str) -> str | None:
    """The command word of a segment, unquoted, path-stripped and wrapper-peeled.

    ``/usr/bin/rg`` -> ``rg``; ``timeout 300 find`` -> ``find``. The peel is what
    closes the wrapper blind spot (review m2) — see :func:`_peel`.
    """
    program, _rest = _peel(segment)
    return program


def _peel(segment: str) -> tuple[str | None, str]:
    """``(command word, text after it)`` with leading wrappers dropped.

    Both guards read the command word, and both were reading the WRAPPER's: an
    agent that writes ``timeout 300 find / …`` — which is what it does when it
    suspects the search is slow — got a clean bill of health. The peel skips a
    wrapper and its own flags, values and assignments, so the word that is
    classified is the one that walks.

    Best-effort by construction, and the failure mode is the OLD behaviour
    (returning the wrapper word), never a block: an unparseable wrapper chain
    returns what it has.
    """
    remainder, _assigns = _strip_env_assignments(segment)
    text = remainder or segment
    word, rest = _read_word(text)
    for _ in range(_MAX_PEEL_DEPTH + 1):
        literal = _unquote(word).strip()
        if literal not in WRAPPER_WORDS:
            return (os.path.basename(literal) or None), rest
        rest = rest.lstrip()
        value_flags = _WRAPPER_VALUE_FLAGS.get(literal, frozenset())
        while True:
            token, after = _read_word(rest)
            if not token:
                return None, ""
            spelling = _unquote(token)
            if spelling.startswith("-") and spelling != "-":
                takes_value = (len(spelling) == 2 and spelling[1] in value_flags) or (
                    spelling in _WRAPPER_LONG_VALUE_FLAGS
                )
                rest = after.lstrip()
                if takes_value:
                    _value, rest = _read_word(rest)
                    rest = rest.lstrip()
                continue
            if _DURATION_RE.match(spelling) or ("=" in spelling and not spelling.startswith("=")):
                # A bare duration is `timeout`'s; an assignment is `env`'s. Both
                # are the wrapper's own arguments, never the command word.
                rest = after.lstrip()
                continue
            word, rest = token, after
            break
    return None, ""


def _shell_c_inner(segment: str, *, depth: int = 0) -> str | None:
    """The command string inside ``bash -c '…'`` / ``sh -c '…'``, or ``None``.

    The wrapper whose argument is a COMMAND rather than a target: skipping it
    would hide the loop behind the shell's own name, which is both the natural
    way to write a compound command and the only way to hang an inline grant on
    one (review Q3). Bounded by :data:`_MAX_PEEL_DEPTH`.
    """
    if depth > _MAX_PEEL_DEPTH:
        return None
    remainder, _assigns = _strip_env_assignments(segment)
    text = remainder or segment
    word, rest = _read_word(text)
    literal = _unquote(word).strip()
    if os.path.basename(literal) not in _SHELL_C_WORDS:
        return None
    rest = rest.lstrip()
    while True:
        token, after = _read_word(rest)
        spelling = _unquote(token)
        if not token:
            return None
        if spelling in ("-c", "--command"):
            inner, _tail = _read_word(after.lstrip())
            inner = inner.strip()
            return inner or None
        if not spelling.startswith("-"):
            # `bash script.sh` — a SCRIPT, not a command string. Nothing to parse.
            return None
        rest = after.lstrip()


def _is_file_like(path: str) -> bool:
    """Does the final segment read as a FILENAME rather than a directory?

    A single named file is never a tree walk, even under a heavy directory:
    ``grep -rn foo build/notes.txt`` and ``grep -rn foo .worktrees/wt/src/x.py``
    read one file, and the fleet reads worktrees by path every day (review M1).
    Recognised by a filename suffix on the last segment, so a bare directory name
    (``node_modules``) and a trailing slash (``node_modules/``) still read as
    directories, and a dotfile (``.gitignore``) does too.
    """
    p = _unquote(path).strip()
    parts = [seg for seg in re.split(r"[/\\]", p) if seg and seg != "."]
    if not parts:
        return False
    last = parts[-1]
    return bool(re.search(r"\.[A-Za-z0-9]{1,6}$", last)) and not last.startswith(".")


def _is_unbounded_root(path: str) -> bool:
    """Is ``path`` a root that reaches the whole tree rather than a part of it?

    This is the STRING rule only — the store and repository classes are decided
    by :func:`_classify_root`, which calls this last so an existing verdict is
    never lost to a probe.
    """
    p = _unquote(path).strip()
    if not p or _UNRESOLVED_RE.search(p):
        # No path, or one that does not resolve here ($VAR, $(...), backticks):
        # what it points at cannot be vouched for, so it is treated as unbounded.
        return True
    if p in (".", "./", "..", "../", "/", "~", "~/"):
        return True
    if _BARE_GLOB_RE.match(p):
        return True
    parts = [seg for seg in re.split(r"[/\\]", p) if seg and seg != "."]
    if not parts:
        return True
    # A filename is read, not walked — never a tree walk whatever it sits in.
    if _is_file_like(p):
        return False
    return any(part in HEAVY_DIRS for part in parts)


def _resolve(path: str) -> str | None:
    """Best-effort absolute path for a CONCRETE operand, or ``None``.

    ``$VAR``/``$(...)``/backtick roots return ``None`` so the unresolved rule
    keeps them: expanding them here would guess at the shell's environment and
    could classify a path the command will not actually walk. Nothing here
    raises — a path this layer cannot even make absolute is simply not a
    candidate for the filesystem probes below.
    """
    p = _unquote(path).strip()
    if not p or _UNRESOLVED_RE.search(p):
        return None
    try:
        return os.path.abspath(os.path.expanduser(p))
    except Exception:  # noqa: BLE001 — a path this layer cannot read is not a block
        return None


def _store_roots() -> tuple[str, ...]:
    """This machine's session-store roots, most specific first.

    Resolved per call rather than cached: the env var is read from the
    environment on every call by ``paths.config_dir()`` too, and a session can
    set it inline — a cached root would then classify the store as bounded for
    the rest of the process. ``STORE_DIRNAME`` mirrors ``paths.DEFAULT_CONFIG_DIRNAME``;
    the two-term resolution (override, else ``$HOME``) mirrors ``paths.config_dir()``.
    """
    roots: list[str] = []
    override = os.environ.get(CONFIG_DIR_ENV)
    if override:
        roots.append(os.path.abspath(os.path.expanduser(override)))
    roots.append(os.path.join(os.path.expanduser("~"), STORE_DIRNAME))
    return tuple(roots)


def _is_store_root(path: str) -> bool:
    """Does ``path`` name the part of the store that holds conversations?

    NARROW ON PURPOSE (review M4). What this class exists for is the TRANSCRIPT
    store: a walk of it is the multi-minute query, and the `sessions` tool answers
    the same question in milliseconds. Everything else under the store root is an
    ordinary directory an agent may reasonably search — ``skills/``,
    ``attachments/``, ``agents/``, ``logs/``, the config file itself — and every
    one of those PASSED before this class existed, so classing the whole root
    would be a regression dressed as a guard.

    So: the store root itself, and anything under ``<root>/sessions/``, with the
    one exemption the agent is TOLD to work in — ``sessions/<id>/scratchpad/**``,
    including the directory itself, and wherever ``LOCAL_OPERATOR_SCRATCHPAD``
    points. A single named FILE is never a tree walk and passes, as it does under
    a heavy directory.
    """
    p = _unquote(path).strip()
    if not p or _is_file_like(p):
        return False
    if _is_scratchpad_path(p):
        return False
    literal = _STORE_STR_RE.match(p)
    if literal is not None:
        # A spelling we cannot resolve ($HOME): the TAIL decides, exactly as the
        # resolved arm does — the root itself and `sessions/` are the store, and
        # a smaller child is not.
        tail = [seg for seg in p[literal.end() :].split("/") if seg]
        return not tail or tail[0] == _SESSIONS_DIRNAME
    resolved = _resolve(p)
    if resolved is None:
        return False
    for root in _store_roots():
        if resolved == root:
            return True
        if _is_under(resolved, os.path.join(root, _SESSIONS_DIRNAME)):
            return True
    return False


def _is_under(path: str, root: str) -> bool:
    """Is ``path`` the root itself, or inside it? Path-component exact."""
    return path == root or path.startswith(root + os.sep)


def _scratchpad_roots() -> tuple[str, ...]:
    """Where ``LOCAL_OPERATOR_SCRATCHPAD`` points, when the variable is set.

    Read per call rather than cached, like the store roots: a session sets it, and
    a cached value would keep exempting a tree the current command is not in.
    """
    root = os.environ.get(SCRATCHPAD_ENV)
    if not root:
        return ()
    try:
        return (os.path.abspath(os.path.expanduser(root)),)
    except Exception:  # noqa: BLE001 — an unreadable var is simply not a root
        return ()


def _is_scratchpad_path(path: str) -> bool:
    """Is this the agent's own scratch tree rather than the transcript store?

    Three spellings, because all three reach a real scratch read: the literal
    ``$LOCAL_OPERATOR_SCRATCHPAD`` the harness prints at an agent, its resolved
    path, and the ``sessions/<id>/scratchpad`` shape the store uses by default.
    """
    if path.startswith(f"${SCRATCHPAD_ENV}") or path.startswith(f"${{{SCRATCHPAD_ENV}}}"):
        # The env spelling is the scratch tree exactly when the variable IS set;
        # with it unset the unresolved rule still decides, which is the honest
        # answer for a path this process cannot see.
        return bool(os.environ.get(SCRATCHPAD_ENV))
    resolved = _resolve(path)
    if resolved is None:
        return False
    for root in _scratchpad_roots():
        if _is_under(resolved, root):
            return True
    for root in _store_roots():
        sessions = os.path.join(root, _SESSIONS_DIRNAME)
        if not _is_under(resolved, sessions) or resolved == sessions:
            continue
        parts = resolved[len(sessions) + 1 :].split(os.sep)
        if len(parts) >= 2 and parts[1] == SCRATCHPAD_DIRNAME:
            return True
    return False


def _is_repo_root(path: str) -> bool:
    """Does ``path`` resolve to a checkout whose walk hits vendored trees?

    The probe is BEST-EFFORT and its failure mode is "not a repo", which is the
    whole safety property: a path that is absent, is a file, or cannot be
    stat'd falls back to the string rules and is never itself a reason to block.
    A guard that refused a command because ``os.path.isdir`` raised would start
    blocking legitimate searches on a host with an unreadable mount or a
    dangling symlink, which is a worse failure than missing one repo root.
    """
    resolved = _resolve(path)
    if resolved is None:
        return False
    try:
        if not os.path.isdir(resolved):
            return False
        # `.git` is a DIRECTORY in a normal checkout and a FILE in a linked
        # worktree, so it is tested with exists(); node_modules is a directory
        # or it is not the tree the exclusion list is about.
        return os.path.exists(os.path.join(resolved, ".git")) or os.path.isdir(
            os.path.join(resolved, "node_modules")
        )
    except OSError:
        return False


def _classify_root(path: str) -> str | None:
    """The class of an unbounded root (``store``/``repo``/``unbounded``), else None.

    STRING rules first, then the filesystem probe. The order is what keeps the
    MESSAGE honest: ``du -sh ~`` is an unbounded root because the author wrote
    ``~``, not because this host happens to have a checkout under it, and a
    message that called it "a repository root" was simply wrong (review m3).
    The probe may only ever ADD a class, never relabel one the string rules
    already reached.
    """
    if _is_scratchpad_path(path):
        # BOUNDED, not merely unclassed: the scratch spelling is often a `$VAR`,
        # which the unresolved rule below would otherwise call unbounded — and the
        # whole point of the exemption is that the tree an agent is TOLD to work
        # in is searchable (review M4).
        return None
    if _is_store_root(path):
        return "store"
    if _is_unbounded_root(path):
        return "unbounded"
    if _is_repo_root(path):
        return "repo"
    return None


def _is_depth_bounded(rest: str, program: str) -> bool:
    """Is this find-family walk bounded by DEPTH?

    Depth is the only bound that makes a walk smaller. A time filter does not:
    ``find`` still visits and stats every entry and filters afterwards, so
    ``-mmin``/``-mtime``/``-newer`` prune the OUTPUT, not the walk — measured on
    the real store, same predicate, ``find ~/.local-operator/sessions -name
    transcript.jsonl -mmin -720`` took 41.7 s against 2.3 s with ``-maxdepth 2``
    (review M3, and ``-atime``/``-ctime``/``-anewer`` are "older than" filters
    that select MORE of the tree, not less). So the relaxation is depth-only, and
    the measured shape still passes because it carries ``-maxdepth`` too.
    """
    if _MAXDEPTH_RE.search(rest):
        return True
    return program == "fd" and bool(_FD_DEPTH_RE.search(rest))


@dataclass(frozen=True)
class _Reason:
    """Why a segment is blocked, plus the advice that fits THAT class.

    A dataclass rather than a bare string because the class decides the fix: a
    repo-root grep wants the `grep` tool, a store-root walk wants the `sessions`
    tool, and `du` wants `df`. One flat message would either say nothing useful
    or name a tool that does not do the job (`grep` cannot replace `du`).
    """

    text: str
    advice: tuple[str, ...]
    allow_env: str = ALLOW_ENV


def _search_reason(segment: str) -> _Reason | None:
    """The reason this segment is an unbounded search, or None if it is fine."""
    # `_peel` consumes the env assignments, the command word and any wrapper
    # chain, so `timeout 300 find /` is judged as `find` and its operands are read
    # from the text AFTER the word that walks (review m2).
    program, rest = _peel(segment)
    if program is None:
        return None
    if program in WALK_BY_CONSTRUCTION:
        return _walker_reason(program, segment)
    if program not in SEARCH_PROGRAMS:
        return None
    if not rest.strip():
        # Bare `grep` with no arguments reads stdin / errors — not a tree walk.
        return None

    if program in _IMPLICITLY_RECURSIVE or program in _ALWAYS_RECURSIVE:
        # Enumeration flags (`--files`, `--type-list`) list what WOULD be
        # searched — a cheap listing, never a walk. `--files-with-matches`/`-l`
        # is NOT one of them: it searches content and prints filenames, so it
        # still walks (round 2, M-b). (`fd`/`locate` have no such flag; the scan
        # is harmless.)
        for word in _words(rest):
            if _RG_ENUMERATION_RE.match(word):
                return None
        recursive = True
    else:
        recursive = bool(_RECURSIVE_SHORT_RE.search(rest) or _RECURSIVE_LONG_RE.search(rest))
    is_find = program in ("find", "fd", "locate")
    if not recursive and not is_find:
        return None

    # Operand tokens after the pattern: flags and their arguments are skipped.
    paths = _path_operands(rest, program)

    if is_find:
        # The FIND FAMILY is judged on its roots first, because a DEPTH bound makes
        # an otherwise unbounded root acceptable — and that decision has to happen
        # BEFORE the predicate rule below, or a bounded walk with no `-maxdepth`
        # spelled the predicate's way would still be refused.
        roots = paths or ["."]
        for root in roots:
            cls = _classify_root(root)
            if cls is None:
                continue
            if _is_depth_bounded(rest, program):
                return None
            return _Reason(
                text=_root_text(program, root, cls, walking=True), advice=_advice("find", cls)
            )
        # Every root is BOUNDED — the author named a scope, and at a named scope
        # the existing predicate rule already said this is fine (`find ~/Downloads
        # -name '*.png'` is a legitimate bounded search). No new block is added
        # here: the depth rule above only ever RELAXES what the root classes would
        # otherwise refuse, it never refuses a named root.
        return None

    if not recursive:
        return None
    # Every candidate root is classified, not just string-matched: the store and
    # repository classes are decided by a probe, and a grep rooted at either is
    # the walk this guard exists to stop.
    for root in paths or ["."]:
        cls = _classify_root(root)
        if cls is None:
            continue
        return _Reason(
            text=_root_text(program, root, cls, walking=False),
            advice=_advice("grep", cls),
        )
    return None


def _walker_reason(program: str, segment: str) -> _Reason | None:
    """The du-shaped arm: a walker with no recursive flag and no depth escape.

    ``du`` descends by construction, and given no operand at all it starts at
    ``.`` exactly as ``find`` does — so the ROOT is the whole decision, and there
    is no ``-maxdepth`` rule to apply because the tool's own ``-d``/``--max-depth``
    is a reporting bound rather than a walk bound the string layer can vouch for.
    """
    _, rest = _peel(segment)
    if not rest.strip():
        # `du` with no operand: walks the cwd.
        return _Reason(
            text=_root_text(program, ".", "unbounded", walking=True), advice=_advice("du", None)
        )
    paths = _path_operands(rest, program) or ["."]
    for root in paths:
        cls = _classify_root(root)
        if cls is None:
            continue
        return _Reason(text=_root_text(program, root, cls, walking=True), advice=_advice("du", cls))
    return None


#: How each root class reads in a refusal. `the session store` is spelled out
#: rather than left as the raw path because the useful information is WHICH
#: store the walk would read, not the string the model already typed.
_ROOT_WORDS = {
    "store": "the session store",
    "repo": "a repository root",
    "unbounded": "an unbounded root",
}


def _root_text(program: str, root: str, cls: str | None, *, walking: bool) -> str:
    kind = _ROOT_WORDS.get(cls or "unbounded", "an unbounded root")
    verb = "walks" if walking else "recurses from"
    return f"`{program}` {verb} {kind} ({root})"


#: The advice bullets, kept as named constants because the same sentence is
#: quoted in tests and in the AGENTS.md inventory — a copy that drifted would be
#: a second definition of what the escape hatch is called.
_GREP_TOOL_BULLET = (
    "use the `grep` tool — it is recursive, honours .gitignore, prunes those "
    "trees, and is far faster (glob for filename patterns);"
)
_NARROW_BULLET = "narrow the path, e.g. `grep -rn PATTERN src/` instead of `.`;"
_SESSIONS_BULLET = (
    "search the stored conversations with the `sessions` tool (its stored-session "
    "search and `peek`) or the `/resume` picker — both read a digest index, not "
    "every transcript, so the answer arrives in milliseconds;"
)
_BOUND_BULLET = (
    "bound the walk: `find ... -maxdepth N` or a time filter (`-mmin -60`), or use "
    "the `glob` tool for a filename pattern;"
)
_DU_BULLET = (
    "scope it, e.g. `du -sh <dir>` (add `-d 1` to keep the reporting shallow) "
    "rather than a whole tree;"
)
_DF_BULLET = "use `df -h` for filesystem headroom — it answers that without walking anything."


def _advice(family: str, cls: str | None) -> tuple[str, ...]:
    """The actionable bullets for a refusal, by tool family and root class.

    The STORE class leads with the `sessions` tool in every family: it is the
    only advice that answers the question the walk was asked, and for `du` it is
    the only one that is about conversations at all. The family then decides
    what else is true — `grep`/`glob` for content search, `df` for sizes, and
    never `grep` for `du`, which no search tool can replace.
    """
    store = cls == "store"
    if family == "du":
        return (_SESSIONS_BULLET, _DU_BULLET, _DF_BULLET) if store else (_DU_BULLET, _DF_BULLET)
    if family == "find":
        return (_SESSIONS_BULLET, _BOUND_BULLET) if store else (_BOUND_BULLET,)
    return (
        (_SESSIONS_BULLET, _GREP_TOOL_BULLET, _NARROW_BULLET)
        if store
        else (
            _GREP_TOOL_BULLET,
            _NARROW_BULLET,
        )
    )


#: Short flags that take a value as the NEXT token; that token is not a path.
#: NOT `g P E w l L` — for grep those select the engine or flip a boolean and
#: consume nothing, so listing them made `grep -rn -E 'a|b' src/` swallow the
#: pattern and lose `src/` (review B1). The set is deliberately small and every
#: member is a flag whose value is genuinely a separate token.
_FLAGS_WITH_VALUE = frozenset({"e", "f", "m", "A", "B", "C", "d", "D", "t"})
_LONG_FLAGS_WITH_VALUE_RE = re.compile(
    r"^--(?:include|exclude|exclude-dir|include-dir|glob|iglob|type|max-depth|maxdepth|depth|"
    r"max-count|context|after-context|before-context|regexp|file|label|encoding|color|colour)="
)

#: Long ENUMERATION flags: they LIST rather than search content, so they are
#: never a content walk. `--files` lists project files (our `glob` replaces it);
#: `--type-list` prints the regex-type table. `--files-with-matches` (`-l`) is
#: deliberately NOT here — it still SEARCHES content and prints filenames, so
#: `rg -l NEEDLE .` walks the tree (review round 2, M-b).
_RG_ENUMERATION_RE = re.compile(r"^--(?:files|type-list)$")


def _short_flag_bundle(bundle: str) -> tuple[bool, bool]:
    """Parse a short-flag bundle (the text after ``-``) left to right.

    Returns ``(consumes_next_token, supplies_pattern)``. A short flag's value is
    the REST OF THE TOKEN when anything follows it (``-efoo`` -> value ``foo``,
    ``-m5`` -> value ``5``), and the NEXT token only when the flag is the last
    character (``-e`` / ``-m``). Getting this wrong is what left `grep -rn -efoo
    src/` refused (review round 2, M-a): the inline value was ignored, the
    pattern was lost, and `src/` was read as the pattern.
    """
    for idx, ch in enumerate(bundle):
        if ch in _FLAGS_WITH_VALUE:
            rest = bundle[idx + 1 :]
            return (rest == "", ch in ("e", "f"))
    return (False, False)


def _path_operands(rest: str, program: str) -> list[str]:
    """Operand tokens that look like search roots (not flags, not the pattern).

    Best-effort by design and biased toward *not* finding a path: a token that
    could be a flag argument is skipped, and a flag that takes a value consumes
    the next token. ``include``/``type``/``glob`` filter arguments are never
    roots.

    The pattern is the first bare word UNLESS it arrived via ``-e``/``-f``/
    ``--regexp``/``--file``, in which case the first bare word is already a
    path — the distinction that made `grep -rn -e P src/` look like an unbounded
    search (review B1, QA Q1).
    """
    words = _words(rest)
    out: list[str] = []
    if not words:
        return out
    # The pattern is the first non-flag word for grep-family and for `fd`/`locate`
    # (both take `PATTERN [PATH...]`), unless a pattern flag supplied it. `find`
    # takes no pattern, so its first bare word is already a path — and `du` is
    # the same shape as `find`: every bare word it takes is a PATH, so treating
    # its first word as a pattern would swallow `du -sh src` and read it as
    # "no operand, walks the cwd".
    skip_next = False
    pattern_consumed = program in ("find", "du")
    for raw in words:
        if skip_next:
            skip_next = False
            continue
        if raw == "--":
            continue
        if raw.startswith("--"):
            # `--include=...` carries its value inline; a bare `--type` consumes
            # the next token. `--regexp`/`--file` supply the PATTERN, so they
            # also mark the pattern consumed.
            if "=" in raw:
                name, _, _val = raw.partition("=")
                if name in ("--regexp", "--file"):
                    pattern_consumed = True
                continue
            name = raw[2:]
            if name in ("regexp", "file"):
                pattern_consumed = True
                skip_next = True
                continue
            if _LONG_FLAGS_WITH_VALUE_RE.match(raw + "=") or name in (
                "include",
                "exclude",
                "exclude-dir",
                "include-dir",
                "glob",
                "iglob",
                "type",
                "max-count",
                "context",
                "after-context",
                "before-context",
                "label",
                "encoding",
                "color",
                "colour",
                "max-depth",
                "maxdepth",
                "depth",
            ):
                skip_next = True
            continue
        if raw.startswith("-") and len(raw) > 1:
            # A short-flag bundle. `-e`/`-f` supply the pattern; any other
            # value-taking flag's INLINE tail (>1 char follows) is its value.
            consumes_next, supplies_pattern = _short_flag_bundle(raw[1:])
            if supplies_pattern:
                pattern_consumed = True
            if consumes_next:
                skip_next = True
            continue
        # A bare word: the pattern unless it was supplied by a flag.
        if not pattern_consumed:
            pattern_consumed = True
            continue
        out.append(_unquote(raw))
    return out


def _words(segment: str) -> list[str]:
    """Split a segment into shell words, keeping quotes so a quoted path stays one word."""
    out: list[str] = []
    buf: list[str] = []
    quote: str | None = None
    i = 0
    n = len(segment)
    while i < n:
        ch = segment[i]
        if quote is not None:
            buf.append(ch)
            if ch == quote:
                quote = None
            elif ch == "\\" and quote == '"' and i + 1 < n:
                buf.append(segment[i + 1])
                i += 2
                continue
            i += 1
            continue
        if ch in ("'", '"'):
            quote = ch
            buf.append(ch)
            i += 1
            continue
        if ch.isspace():
            if buf:
                out.append("".join(buf))
                buf = []
            i += 1
            continue
        buf.append(ch)
        i += 1
    if buf:
        out.append("".join(buf))
    return out


def _is_git_grep(segment: str) -> bool:
    """True for ``git grep …`` / ``git -c … grep …`` — structured and exempt."""
    words = _words(segment)
    for idx, w in enumerate(words):
        if w == "git":
            return "grep" in words[idx + 1 : idx + 4]
    return False


#: Programs that read STDIN instead of walking when they are the downstream of a
#: pipe and no path operand is given. `rg` filters stdin (verified: `printf x |
#: rg needle` prints the stdin line); `grep -r` does NOT — it still recurses the
#: cwd — which is why this is a per-program set and not a blanket pipe exemption
#: (review M2).
_STDIN_FILTER_PROGRAMS = frozenset({"rg", "ripgrep", "ag", "ack"})

#: Programs that walk the filesystem for FILES rather than searching content
#: stdin. A piped `find`/`fd`/`locate` still walks, so it is never exempted as a
#: stream filter.
_WALK_PROGRAMS = frozenset({"find", "fd", "locate"})


def check_search_interception(
    command: str,
    *,
    enabled: bool = True,
    block_unbounded: bool = True,
    _depth: int = 0,
) -> str | None:
    """Return a block message when ``command`` is an unbounded search, else None.

    ``enabled=False`` disables the check entirely; ``block_unbounded=False``
    leaves the caller to warn-and-run rather than refuse. The inline
    ``LOCAL_OPERATOR_ALLOW_UNBOUNDED_SEARCH`` grant is read per SEGMENT, so an
    agent can grant it to one command without a config change.

    ``bash -c '…'``/``sh -c '…'`` are PARSED, not skipped (review m2/Q3): the
    argument is a command string, so it is checked with these same rules — the
    wrapper is how a compound command is written, and it must not be a way to
    hide one. The grant on the wrapping segment carries into the inner command,
    which is what makes ``ALLOW=1 bash -c '<search>'`` the working form for it.
    """
    if not enabled or _depth > _MAX_PEEL_DEPTH:
        return None
    for segment, piped in _segments(command):
        if _is_git_grep(segment):
            continue
        stripped, assigns = _strip_env_assignments(segment)
        if not stripped:
            continue
        inner = _shell_c_inner(stripped)
        if inner is not None:
            if _truthy(assigns.get(ALLOW_ENV)):
                continue
            nested = check_search_interception(
                inner,
                enabled=enabled,
                block_unbounded=block_unbounded,
                _depth=_depth + 1,
            )
            if nested is not None:
                return nested
            continue
        program = _program(stripped)
        if piped and program in _STDIN_FILTER_PROGRAMS and not _has_path_operand(stripped):
            # A ripgrep stage with no path operand reads the previous stage's
            # stdout — a stream filter no file-search tool can replace, so it is
            # exempt. This is deliberately NOT a blanket pipe exemption: a piped
            # `grep -r` and a piped `find` still walk the cwd, and a piped stage
            # that names a path walks that path (review M2).
            continue
        # The inline grant is read PER SEGMENT, so it applies to the command the
        # agent wrote and never leaks to another. The process-environment arm is
        # deliberately NOT consulted: an inherited value would make the grant
        # silently global and invisible in the transcript (review m1).
        if _truthy(assigns.get(ALLOW_ENV)):
            continue
        reason = _search_reason(stripped)
        if reason is None:
            continue
        return _block_message(reason, command, blocked=block_unbounded)
    return None


def _has_path_operand(segment: str) -> bool:
    """Does this segment name a path for the search tool to walk?

    Used only to decide whether a PIPED stage is a stream filter or a real walk:
    `... | grep -v node_modules` has no operand, `cat f | grep -rn p .` does.
    """
    program, rest = _peel(segment)
    if program is None or program not in SEARCH_PROGRAMS:
        return False
    return bool(_path_operands(rest, program))


def _truthy(value: str | None) -> bool:
    return value is not None and value.strip().lower() not in ("", "0", "false", "no", "off")


def _block_message(reason: _Reason, command: str, *, blocked: bool) -> str:
    """The refusal, stated so the model can act on it rather than guess.

    ``command`` is accepted for the caller's symmetry and deliberately unused:
    echoing the command back would put a credential the model typed into a second
    place for no gain, and the model already knows what it asked for.
    """
    del command
    verb = "blocked" if blocked else "warning"
    lines = [
        f"{verb}: {reason.text} — a recursive search from the repository root "
        "walks vendored and generated trees (node_modules, .git, out) that will "
        "not contain the answer, and has taken tens of seconds where a scoped "
        "search takes milliseconds.",
        "Do one of:",
    ]
    lines.extend(f"  - {item}" for item in reason.advice)
    lines.append(f"  - to run it exactly as written, prefix it with `{reason.allow_env}=1`.")
    return "\n".join(lines)
