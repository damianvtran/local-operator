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
   exempt because it is structured and honours .gitignore;
2. a recursive flag is present (``-r``/``-R``/bundled ``-rn``), or it is
   ``find``/``fd`` with a name/type/path predicate and no ``-maxdepth``;
3. the search *root* is unbounded: no path operand at all, or a path that is
   ``.``/``./``/``..``/``/``/``~``, a bare glob, an unresolved ``$VAR``/``$(...)``,
   or a known-heavy directory (``node_modules``, ``.git``, ``out``, ...).

A single named file, a scoped directory (``grep -rn PATTERN src/``), a piped
stage (``... | grep -v node_modules``) and a quoted mention (``echo "grep -rn x ."``)
all pass. Those are exactly the false-positive classes the tests pin, and the
segment splitter is quote- and escape-aware so shell syntax the model writes
never reads as a second command.

Escape hatch: prepend ``LOCAL_OPERATOR_ALLOW_UNBOUNDED_SEARCH=1`` to the command
to run it as written. That is a per-call grant read off the command itself, so
the agent keeps full expression without a config edit.
"""

from __future__ import annotations

import os
import re

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
_MAXDEPTH_RE = re.compile(r"(?<![\w-])-(?:maxdepth|mindepth)(?![\w-])")
#: A token that is only glob metacharacters (``*``, ``*.ts`` is NOT bare — it
#: has a stem — but ``*`` and ``**`` are).
_BARE_GLOB_RE = re.compile(r"^[*?]+$")
#: Leading ``NAME=value`` assignments on a segment.
_ENV_ASSIGN_RE = re.compile(r"^[A-Za-z_][A-Za-z0-9_]*=")
#: An operand that never resolves to a concrete, bounded location: ``$VAR``,
#: ``$(...)``, a backtick substitution, or a brace expansion.
_UNRESOLVED_RE = re.compile(r"[`$]|\$\(")

#: Search programs that recurse BY DEFAULT, so the absence of a `-r` flag says
#: nothing about the scope. `rg`/`ag`/`ack` walk the cwd unless given a path, so
#: they are always treated as recursive.
_IMPLICITLY_RECURSIVE = frozenset({"rg", "ripgrep", "ag", "ack"})

#: A heredoc opener: `<<EOF`, `<<-EOF`, `<<'EOF'`, `<<"EOF"`.
_HEREDOC_RE = re.compile(r"<<-?\s*(['\"]?)([A-Za-z_][A-Za-z0-9_]*)\1")


def _strip_heredocs(command: str) -> str:
    """Blank out heredoc BODIES so their text is never read as commands.

    `cat <<EOF … grep -rn x . … EOF` is one command whose body is data; a
    splitter that reads the body sees a second segment and blocks a shell that
    would never run it. Each body is replaced by a single space, and the
    delimiter line is kept as a harmless word.
    """
    out: list[str] = []
    i = 0
    n = len(command)
    while i < n:
        m = _HEREDOC_RE.search(command, i)
        if m is None:
            out.append(command[i:])
            break
        out.append(command[i : m.end()])
        delim = m.group(2)
        # Skip to the start of the line after the opener, then to the
        # delimiter line (a line whose stripped content is the delimiter).
        nl = command.find("\n", m.end())
        if nl == -1:
            break
        pos = nl + 1
        while pos <= n:
            line_end = command.find("\n", pos)
            line = command[pos : line_end if line_end != -1 else n]
            if line.strip() == delim:
                out.append(" ")
                pos = line_end if line_end != -1 else n
                break
            if line_end == -1:
                pos = n
                break
            pos = line_end + 1
        i = pos
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
    command = _strip_heredocs(command)
    out: list[tuple[str, bool]] = []
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
                out.append((text, piped))
            buf = []
            piped = ch == "|" and (j - i) == 1
            i = j
            continue
        buf.append(ch)
        i += 1
    text = "".join(buf).strip()
    if text:
        out.append((text, piped))
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
    """The first word of a segment, unquoted and path-stripped (``/usr/bin/rg`` -> ``rg``)."""
    word, _ = _read_word(segment)
    if not word:
        return None
    return os.path.basename(_unquote(word))


def _is_unbounded_root(path: str) -> bool:
    """Is ``path`` a search root that reaches the whole tree rather than a part of it?"""
    p = _unquote(path).strip()
    if not p or _UNRESOLVED_RE.search(p):
        # No path, or one that does not resolve here ($VAR, $(...), backticks):
        # what it points at cannot be vouched for, so it is treated as unbounded.
        return True
    if p in (".", "./", "..", "../", "/", "~", "~/", "."):
        return True
    if _BARE_GLOB_RE.match(p):
        return True
    parts = [seg for seg in re.split(r"[/\\]", p) if seg and seg != "."]
    return any(part in HEAVY_DIRS for part in parts)


def _search_reason(segment: str) -> str | None:
    """The reason this segment is an unbounded search, or None if it is fine."""
    program = _program(segment)
    if program is None or program not in SEARCH_PROGRAMS:
        return None

    rest = segment
    # Drop the program word itself, then any leading env assignments before it.
    remainder, _assigns = _strip_env_assignments(segment)
    if remainder:
        rest = remainder
    _, rest = _read_word(rest)  # consumes the program word
    if not rest.strip():
        # Bare `grep` with no arguments reads stdin / errors — not a tree walk.
        return None

    if program in _IMPLICITLY_RECURSIVE:
        # `--files` lists what WOULD be searched rather than searching it; it is
        # a cheap enumeration, not a content walk, so it is never blocked.
        if "--files" in _words(rest):
            return None
        recursive = True
    else:
        recursive = bool(_RECURSIVE_SHORT_RE.search(rest) or _RECURSIVE_LONG_RE.search(rest))
    is_find = program in ("find", "fd", "locate")
    if not recursive:
        if not (is_find and _FIND_PREDICATE_RE.search(rest) and not _MAXDEPTH_RE.search(rest)):
            return None

    # Operand tokens after the pattern: flags and their arguments are skipped.
    paths = _path_operands(rest, program)
    unbounded = [p for p in paths if _is_unbounded_root(p)]
    # A recursion with NO explicit path operand searches the cwd (`.`).
    if not paths:
        unbounded = ["."]
    if not unbounded:
        return None
    return f"`{program}` recurses from an unbounded root ({', '.join(unbounded[:2])})"


#: Short flags that take a value as the NEXT token; that token is not a path.
_FLAGS_WITH_VALUE = frozenset(
    {"e", "f", "m", "A", "B", "C", "d", "D", "t", "T", "g", "P", "E", "w", "l", "L"}
)
_LONG_FLAGS_WITH_VALUE_RE = re.compile(
    r"^--(?:include|exclude|exclude-dir|include-dir|glob|iglob|type|max-depth|maxdepth|depth|"
    r"max-count|context|after-context|before-context|regexp|file|label|encoding|color|colour)="
)


def _path_operands(rest: str, program: str) -> list[str]:
    """Operand tokens that look like search roots (not flags, not the pattern).

    Best-effort by design and biased toward *not* finding a path: a token that
    could be a flag argument is skipped, and a flag that takes a value consumes
    the next token. ``include``/``type``/``glob`` filter arguments are never
    roots.
    """
    words = _words(rest)
    out: list[str] = []
    if not words:
        return out
    # The pattern is the first non-flag word for grep-family; for find it is the
    # path list itself, so there is no leading pattern to skip.
    skip_next = False
    pattern_consumed = program in ("find", "fd", "locate")
    for idx, w in enumerate(words):
        raw = w
        if skip_next:
            skip_next = False
            continue
        if raw == "--":
            continue
        if raw.startswith("--"):
            # `--include=...` carries its value inline; a bare `--type` consumes
            # the next token.
            if "=" in raw or _LONG_FLAGS_WITH_VALUE_RE.match(raw + "="):
                continue
            name = raw[2:]
            if len(name) == 1 or name in (
                "include",
                "exclude",
                "exclude-dir",
                "include-dir",
                "glob",
                "type",
                "max-count",
                "context",
                "e",
            ):
                skip_next = True
            continue
        if raw.startswith("-") and len(raw) > 1:
            # A short-flag bundle: if any flag needs a value and none is inline
            # (`-e PATTERN`), the next token is that value.
            letters = [c for c in raw[1:] if c.isalpha()]
            if letters and letters[-1] in _FLAGS_WITH_VALUE:
                skip_next = True
            continue
        # A bare word: for grep-family the first one is the pattern.
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


def check_search_interception(
    command: str,
    *,
    enabled: bool = True,
    block_unbounded: bool = True,
) -> str | None:
    """Return a block message when ``command`` is an unbounded search, else None.

    ``enabled=False`` disables the check entirely; ``block_unbounded=False``
    leaves the caller to warn-and-run rather than refuse. The inline
    ``LOCAL_OPERATOR_ALLOW_UNBOUNDED_SEARCH`` grant is read per SEGMENT, so an
    agent can grant it to one command without a config change.
    """
    if not enabled:
        return None
    for segment, piped in _segments(command):
        if piped:
            # Reads the previous stage's stdout — a dedicated file-search tool
            # cannot replace a stream filter.
            continue
        if _is_git_grep(segment):
            continue
        stripped, assigns = _strip_env_assignments(segment)
        if _truthy(assigns.get(ALLOW_ENV)) or _truthy(os.environ.get(ALLOW_ENV)):
            continue
        if not stripped:
            continue
        reason = _search_reason(stripped)
        if reason is None:
            continue
        return _block_message(reason, command, blocked=block_unbounded)
    return None


def _truthy(value: str | None) -> bool:
    return value is not None and value.strip().lower() not in ("", "0", "false", "no", "off")


def _block_message(reason: str, command: str, *, blocked: bool) -> str:
    """The refusal, stated so the model can act on it rather than guess."""
    verb = "blocked" if blocked else "warning"
    return (
        f"{verb}: {reason} — a recursive search from the repository root walks "
        "vendored and generated trees (node_modules, .git, out) that will not contain "
        "the answer, and has taken tens of seconds where a scoped search takes "
        "milliseconds.\n"
        "Do one of:\n"
        "  - use the `grep` tool — it is recursive, honours .gitignore, prunes those "
        "trees, and is far faster (glob for filename patterns);\n"
        "  - narrow the path, e.g. `grep -rn PATTERN src/` instead of `.`;\n"
        f"  - to run it exactly as written, prefix it with `{ALLOW_ENV}=1`."
    )
