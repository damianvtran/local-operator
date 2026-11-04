"""Which paths a shell command would remove, move away or overwrite — read, never run.

WHY THIS EXISTS. On 2026-11-04 an unattended, auto-approved session issued one
compound shell call whose first segment was a recursive delete of a path built
from the state root plus an empty command substitution. The substitution
produced nothing, the path collapsed onto the state root itself, and the whole
of it (config, secrets, agents, projects, memory) was gone. The step the model
meant was a scratchpad clear later in the same line. Nothing between the model
and the shell understood WHAT a command deletes: approval tiers are moot under
auto approval, and ``tools.search_guard`` only looks at searches.

This module answers one question for :mod:`local_operator.guardrails.classify`:
*which concrete paths does this command destroy?* It is a static parser:

* Nothing is executed. A substitution body is inspected as text: a body that is
  known to print nothing becomes ``""`` and the word is tagged
  ``empty_expansion`` (the incident shape); any other body makes the word
  ``opaque``. Opaque is never treated as safe.
* Variables expand from the environment the bash child would actually receive,
  overlaid by assignments EARLIER in the same command, so a path built in a
  variable resolves the same way the shell would. Unset expands to ``""`` and
  tags ``empty_expansion``. Loop/read variables are opaque (runtime values).
* ``cd``/``pushd`` move the working directory; one this parser cannot resolve
  makes it unknown, and a relative target under an unknown cwd is flagged.
* Globs are never expanded against the disk; the caller matches the pattern
  against the paths it protects.

The segment splitter and the wrapper vocabulary are IMPORTED from
``tools.search_guard`` so the two guards cannot disagree about where a command
ends or which word is the program. Every uncertainty resolves toward MORE
scrutiny (opaque, empty, unknown cwd), never less.
"""

from __future__ import annotations

import os
import re
from collections.abc import Mapping
from dataclasses import dataclass, field

from local_operator.tools.search_guard import (
    _DURATION_RE,
    _MAX_PEEL_DEPTH,
    _SHELL_C_WORDS,
    _WRAPPER_LONG_VALUE_FLAGS,
    _WRAPPER_VALUE_FLAGS,
    WRAPPER_WORDS,
    _split,
)

#: Programs whose output never contains ``/`` or ``..`` in normal use, so a
#: substitution of one cannot move a path out of the directory it sits in.
_CONTAINED_PROGRAMS = frozenset({"date", "gdate", "uuidgen", "hostname", "whoami", "id"})
#: Substitution bodies known to print nothing. Matching is on the stripped
#: body text; anything not in this set is opaque.
_EMPTY_BODIES = frozenset(
    {"", "echo", "true", ":", "printf ''", 'printf ""', "echo ''", 'echo ""', "echo -n"}
)
_NAME_RE = re.compile(r"[A-Za-z_][A-Za-z0-9_]*")
_ASSIGN_RE = re.compile(r"^([A-Za-z_][A-Za-z0-9_]*)=")
#: Words that may open a segment without being its program.
_PREFIX_WORDS = frozenset({"then", "do", "else", "elif", "if", "while", "until", "!", "{", "("})
_DECLARE_WORDS = frozenset({"export", "declare", "typeset", "local", "readonly"})
#: Marker for an opaque stretch inside an expanded word; NUL never occurs in argv.
OPAQUE_MARK = "\x00"


# ---------------------------------------------------------------------------
# Scanning: a segment into raw words (parts) and redirections
# ---------------------------------------------------------------------------


@dataclass
class _Part:
    #: "bare" (unquoted: tilde, $, glob apply), "dq" (double-quoted: $ only),
    #: "sq" (single-quoted: literal), "sub" (a command-substitution body).
    kind: str
    text: str


@dataclass
class _RawWord:
    parts: list[_Part] = field(default_factory=list)
    #: The scanner could not balance a quote or substitution: never trust it.
    broken: bool = False

    def add(self, kind: str, text: str) -> None:
        if self.parts and self.parts[-1].kind == kind and kind in ("bare", "dq", "sq"):
            self.parts[-1].text += text
        else:
            self.parts.append(_Part(kind, text))


def _read_sub(text: str, i: int) -> tuple[str, int, bool]:
    """Body of a ``$(`` substitution starting at ``text[i]`` (just past ``$(``).

    Depth-counted and quote-aware so a nested substitution stays one word.
    Returns ``(body, index_after, balanced)``.
    """
    depth = 1
    quote: str | None = None
    j = i
    n = len(text)
    while j < n:
        ch = text[j]
        if quote:
            if ch == "\\" and quote == '"':
                j += 2
                continue
            if ch == quote:
                quote = None
        elif ch in ("'", '"'):
            quote = ch
        elif ch == "\\":
            j += 2
            continue
        elif ch == "(":
            depth += 1
        elif ch == ")":
            depth -= 1
            if depth == 0:
                return text[i:j], j + 1, True
        j += 1
    return text[i:], n, False


def _read_dollar(text: str, i: int, word: _RawWord, kind: str) -> int:
    """Consume a ``$…`` form at ``text[i] == '$'`` into ``word``; return next index."""
    n = len(text)
    nxt = text[i + 1 : i + 2]
    if nxt == "(" and text[i + 2 : i + 3] != "(":
        body, j, ok = _read_sub(text, i + 2)
        word.add("sub", body)
        word.broken |= not ok
        return j
    if nxt == "(":
        # $(( arithmetic )) — evaluates to a number, never a path we can know.
        body, j, ok = _read_sub(text, i + 2)
        word.add("sub", "\x01arith")
        word.broken |= not ok
        return j
    if nxt == "{":
        end = text.find("}", i + 2)
        if end == -1:
            word.broken = True
            return n
        word.add(kind, text[i : end + 1])
        return end + 1
    m = _NAME_RE.match(text, i + 1)
    if m is not None:
        word.add(kind, text[i : m.end()])
        return m.end()
    if nxt and nxt in "@*#?$!-0123456789":
        word.add(kind, text[i : i + 2])
        return i + 2
    word.add(kind, "$")
    return i + 1


def scan(segment: str) -> tuple[list[_RawWord], list[tuple[str, _RawWord]]]:
    """Split one segment into raw words and ``(operator, target)`` redirections."""
    words: list[_RawWord] = []
    redirects: list[tuple[str, _RawWord]] = []
    cur: _RawWord | None = None
    pending_redirect: str | None = None
    i = 0
    n = len(segment)

    def finish() -> None:
        nonlocal cur, pending_redirect
        if cur is None:
            return
        if pending_redirect is not None:
            redirects.append((pending_redirect, cur))
            pending_redirect = None
        else:
            words.append(cur)
        cur = None

    while i < n:
        ch = segment[i]
        if ch.isspace():
            finish()
            i += 1
            continue
        if ch in "<>" or (ch.isdigit() and cur is None and re.match(r"\d+[<>]", segment[i:])):
            # Redirection operator: [n]> [n]>> >| &> <  (and <<, already body-stripped).
            finish()
            m = re.match(r"\d*(>>|>\||>&|>|<<<|<<-?|<&|<>|<)", segment[i:])
            op = m.group(1) if m else ch
            i += m.end() if m else 1
            if op in (">&", "<&"):
                # fd duplication (2>&1) — the target is a descriptor, not a file.
                fd = re.match(r"\s*[\d-]+", segment[i:])
                if fd:
                    i += fd.end()
                    continue
            pending_redirect = op
            continue
        if ch == "&" and segment[i + 1 : i + 2] == ">":
            finish()
            i += 2
            if segment[i : i + 1] == ">":
                i += 1
                pending_redirect = ">>"
            else:
                pending_redirect = ">"
            continue
        if cur is None:
            cur = _RawWord()
        if ch == "'":
            end = segment.find("'", i + 1)
            if end == -1:
                cur.broken = True
                cur.add("sq", segment[i + 1 :])
                i = n
            else:
                cur.add("sq", segment[i + 1 : end])
                i = end + 1
            continue
        if ch == '"':
            j = i + 1
            while j < n and segment[j] != '"':
                c = segment[j]
                if c == "\\" and j + 1 < n:
                    nxt = segment[j + 1]
                    # An escaped char is literal: record it as quoted text so a
                    # later expansion pass never re-reads ``\$`` as a variable.
                    if nxt in '"\\$`':
                        cur.add("sq", nxt)
                    else:
                        cur.add("dq", "\\" + nxt)
                    j += 2
                elif c == "$":
                    j = _read_dollar(segment, j, cur, "dq")
                elif c == "`":
                    end = segment.find("`", j + 1)
                    if end == -1:
                        cur.broken = True
                        j = n
                    else:
                        cur.add("sub", segment[j + 1 : end])
                        j = end + 1
                else:
                    cur.add("dq", c)
                    j += 1
            if j >= n:
                cur.broken = True
            i = j + 1
            continue
        if ch == "\\":
            if i + 1 < n:
                cur.add("sq", segment[i + 1])
            i += 2
            continue
        if ch == "$":
            i = _read_dollar(segment, i, cur, "bare")
            continue
        if ch == "`":
            end = segment.find("`", i + 1)
            if end == -1:
                cur.broken = True
                i = n
            else:
                cur.add("sub", segment[i + 1 : end])
                i = end + 1
            continue
        cur.add("bare", ch)
        i += 1
    finish()
    if pending_redirect is not None:
        redirects.append((pending_redirect, _RawWord(broken=True)))
    return words, redirects


# ---------------------------------------------------------------------------
# Expansion: a raw word into its text, against the env the child would get
# ---------------------------------------------------------------------------

_DOLLAR_RE = re.compile(r"\$\{([^}]*)\}|\$([A-Za-z_][A-Za-z0-9_]*)|\$([@*#?$!\-0-9])")
_BRACE_DEFAULT_RE = re.compile(r"^([A-Za-z_][A-Za-z0-9_]*)(:?[-=])(.*)$")


@dataclass
class Expanded:
    """One word after expansion, plus what the expansion could not know."""

    text: str
    #: A glob pattern (``fnmatch`` syntax, literals escaped) when the word globs.
    pattern: str | None = None
    #: Human-readable notes for every expansion that produced nothing.
    empty: list[str] = field(default_factory=list)
    #: True when any stretch is unknowable statically (``OPAQUE_MARK`` in text).
    opaque: bool = False
    #: Bodies of command substitutions in the word: they RUN, so they are
    #: resolved as commands too.
    substitutions: list[str] = field(default_factory=list)
    #: Every opaque stretch came from a substitution whose program cannot print
    #: a path climb (``date``, ``uuidgen``…). Lets a timestamped name inside the
    #: session's own scratchpad stay LOW; an arbitrary body could print ``..``.
    opaque_contained: bool = True
    #: Every character came from an unquoted expansion that produced nothing,
    #: so bash drops the word entirely.
    vanished: bool = False


def _glob_escape(text: str) -> str:
    return re.sub(r"([*?\[])", r"[\1]", text)


def _expand_dollars(text: str, env: Mapping[str, str], out: Expanded) -> tuple[str, bool]:
    """Expand ``$`` forms in quoted/unquoted text; returns (text, produced_any)."""
    produced = False
    pieces: list[str] = []
    pos = 0
    for m in _DOLLAR_RE.finditer(text):
        pieces.append(text[pos : m.start()])
        pos = m.end()
        if text[m.start() : m.start() + 1] != "$":
            continue
        name = m.group(2)
        value: str | None
        if m.group(1) is not None:
            inner = m.group(1)
            if _NAME_RE.fullmatch(inner):
                name = inner
            else:
                dm = _BRACE_DEFAULT_RE.match(inner)
                if dm is None:
                    out.opaque = True
                    out.opaque_contained = False
                    pieces.append(OPAQUE_MARK)
                    continue
                name = dm.group(1)
                current = env.get(name)
                use_default = current is None or (dm.group(2).startswith(":") and current == "")
                if use_default:
                    pieces.append(dm.group(3))
                    produced = produced or bool(dm.group(3))
                    continue
        if name is None:
            # Positional and special parameters ($1, $@, $$): runtime values.
            out.opaque = True
            out.opaque_contained = False
            pieces.append(OPAQUE_MARK)
            continue
        value = env.get(name)
        if value is None or value == "":
            out.empty.append(f"${name} expanded empty" + ("" if value == "" else " (unset)"))
            continue
        if OPAQUE_MARK in value:
            out.opaque = True
            out.opaque_contained = False
        pieces.append(value)
        produced = True
    pieces.append(text[pos:])
    return "".join(pieces), produced


def expand(word: _RawWord, env: Mapping[str, str], home: str) -> Expanded:
    """Expand a scanned word the way bash would, without running anything."""
    out = Expanded(text="")
    text_parts: list[str] = []
    pattern_parts: list[str] = []
    globbed = False
    literal_seen = False
    for index, part in enumerate(word.parts):
        if part.kind == "sub":
            body = part.text.strip()
            out.substitutions.append(part.text)
            if body in _EMPTY_BODIES:
                out.empty.append(f"$({body}) expanded empty")
            else:
                out.opaque = True
                first = body.split(None, 1)[0] if body else ""
                if os.path.basename(first) not in _CONTAINED_PROGRAMS:
                    out.opaque_contained = False
                text_parts.append(OPAQUE_MARK)
                pattern_parts.append(OPAQUE_MARK)
            continue
        if part.kind == "sq":
            text_parts.append(part.text)
            pattern_parts.append(_glob_escape(part.text))
            literal_seen = True
            continue
        raw = part.text
        if part.kind == "bare" and index == 0 and raw.startswith("~"):
            head, sep, tail = raw.partition("/")
            if head == "~":
                raw = home + sep + tail
            else:
                # ``~user`` names another account's home: unknowable here.
                out.opaque = True
                out.opaque_contained = False
                raw = OPAQUE_MARK + sep + tail
        expanded, produced = _expand_dollars(raw, env, out)
        if part.kind == "dq":
            text_parts.append(expanded)
            pattern_parts.append(_glob_escape(expanded))
            literal_seen = True
            continue
        if expanded:
            literal_seen = literal_seen or produced or bool(_DOLLAR_RE.sub("", part.text))
        if any(c in expanded for c in "*?["):
            globbed = True
        text_parts.append(expanded)
        pattern_parts.append(expanded)
    out.text = "".join(text_parts)
    out.pattern = "".join(pattern_parts) if globbed else None
    out.opaque = out.opaque or OPAQUE_MARK in out.text or word.broken
    if word.broken:
        out.opaque_contained = False
    out.vanished = not out.text and not literal_seen and not word.broken
    return out


# ---------------------------------------------------------------------------
# Walking a command: segments, state, and the destructive programs
# ---------------------------------------------------------------------------


@dataclass
class Target:
    """A path a command would remove, move away, or overwrite."""

    #: Absolute, normalised path; for a glob/opaque word, the deepest literal
    #: directory above the unknowable part. ``None`` when nothing literal remains.
    path: str | None
    op: str  # "delete" | "move" | "overwrite" | "clean"
    program: str
    pattern: str | None = None
    empty: list[str] = field(default_factory=list)
    opaque: bool = False
    opaque_contained: bool = False
    unknown_cwd: bool = False


@dataclass
class _State:
    env: dict[str, str]
    cwd: str | None
    home: str


def _absolute(text: str, cwd: str | None) -> tuple[str | None, bool]:
    """``(normalised absolute path, cwd_was_unknown)`` for a word's text."""
    if text.startswith("/"):
        return os.path.normpath("/" + text.lstrip("/")), False
    if cwd is None:
        return None, True
    return os.path.normpath(os.path.join(cwd, text)), False


def _target(exp: Expanded, op: str, program: str, state: _State) -> Target:
    text = exp.text
    pattern = exp.pattern
    opaque = exp.opaque
    if opaque or pattern is not None:
        # The literal directory above the first unknowable/glob character.
        cut = len(text)
        for marker in (OPAQUE_MARK, "*", "?", "["):
            at = text.find(marker)
            if at != -1:
                cut = min(cut, at)
        literal = text[:cut]
        literal = literal[: literal.rfind("/") + 1] if "/" in literal else ""
        if opaque:
            pattern = None
        path, unknown = _absolute(literal or ".", state.cwd)
        if pattern is not None:
            base = state.cwd or ""
            pattern = pattern if pattern.startswith("/") else os.path.join(base, pattern)
            pattern = os.path.normpath(pattern) if not unknown else None
    else:
        path, unknown = _absolute(text or ".", state.cwd)
    return Target(
        path=path,
        op=op,
        program=program,
        pattern=pattern,
        empty=list(exp.empty),
        opaque=opaque,
        opaque_contained=opaque and exp.opaque_contained,
        unknown_cwd=unknown,
    )


@dataclass
class Resolution:
    targets: list[Target] = field(default_factory=list)
    #: The parser met a construct it cannot follow that MAY delete (``eval`` of a
    #: deleting word, an unbalanced quote around one): reasons, for the judge.
    opaque_reasons: list[str] = field(default_factory=list)


def _operands(words: list[Expanded], value_flags: frozenset[str] = frozenset()) -> tuple[list[Expanded], list[str]]:
    """Split argv tail into (operands, flags); ``--`` ends the flags."""
    operands: list[Expanded] = []
    flags: list[str] = []
    it = iter(range(len(words)))
    ended = False
    skip = False
    for i in it:
        if skip:
            skip = False
            continue
        w = words[i]
        t = w.text
        if not ended and t == "--":
            ended = True
            continue
        if not ended and t.startswith("-") and len(t) > 1 and not w.opaque:
            flags.append(t)
            if t in value_flags:
                skip = True
            continue
        if w.vanished:
            continue
        operands.append(w)
    return operands, flags


def _has_short(flags: list[str], letters: str) -> bool:
    return any(
        not f.startswith("--") and any(c in f[1:] for c in letters) for f in flags
    )


#: Programs whose operands are removed. ``grm``/``gunlink`` are the GNU names on
#: a Homebrew coreutils install.
DELETE_PROGRAMS = frozenset({"rm", "grm", "rmdir", "unlink", "gunlink", "shred"})
#: Words that run the REST of the segment as a command (beyond search_guard's list).
_EXTRA_WRAPPERS = frozenset({"builtin", "exec", "doas", "caffeinate", "stdbuf", "ionice"})
_RM_VALUE_FLAGS: frozenset[str] = frozenset()
_SHRED_VALUE_FLAGS = frozenset({"-n", "-s", "--iterations", "--size"})
_TRUNCATE_VALUE_FLAGS = frozenset({"-s", "-r", "--size", "--reference"})
_MV_VALUE_FLAGS = frozenset({"-t", "-S", "--suffix", "--target-directory"})
_RSYNC_VALUE_FLAGS = frozenset({"-e", "--rsh", "--exclude", "--include", "--filter", "-f"})
#: find: the predicates that make a match destructive.
_FIND_EXEC = frozenset({"-exec", "-execdir", "-ok", "-okdir"})


def _peel_wrappers(words: list[Expanded]) -> tuple[list[Expanded], bool]:
    """Drop leading wrappers; ``(remaining words, fed_from_stdin)``.

    Mirrors ``search_guard._peel``'s vocabulary so both guards agree on which
    word is the program. ``xargs`` is recorded: its operands arrive on stdin,
    so whatever it deletes is unknowable here.
    """
    stdin_fed = False
    for _ in range(_MAX_PEEL_DEPTH + 2):
        if not words:
            return words, stdin_fed
        name = os.path.basename(words[0].text)
        if name not in WRAPPER_WORDS and name not in _EXTRA_WRAPPERS:
            return words, stdin_fed
        if name == "xargs":
            stdin_fed = True
        value_flags = _WRAPPER_VALUE_FLAGS.get(name, frozenset())
        rest = words[1:]
        while rest:
            t = rest[0].text
            if t.startswith("-") and t != "-":
                takes = (len(t) == 2 and t[1] in value_flags) or t in _WRAPPER_LONG_VALUE_FLAGS
                rest = rest[2:] if takes else rest[1:]
                continue
            if _DURATION_RE.match(t) or (_ASSIGN_RE.match(t) and name == "env"):
                rest = rest[1:]
                continue
            break
        words = rest
    return words, stdin_fed


def _strip_grouping(text: str) -> str:
    """Drop subshell/group delimiters the splitter leaves on a segment's edges.

    Only UNBALANCED ones: ``( cd x && rm y )`` splits into ``( cd x`` and
    ``rm y )``, while the ``)`` closing a ``$(…)`` inside the segment is
    balanced and must stay or the substitution reads as broken.
    """
    text = text.strip()
    while text[:1] in ("(", "{") and text[:2] != "((":
        text = text[1:].lstrip()
    while text[-1:] in (")", "}") and (
        text.count(text[-1]) > text.count("(" if text[-1] == ")" else "{")
    ):
        text = text[:-1].rstrip()
    return text


def _tilde_value(value: str, home: str) -> str:
    """Bash expands a leading ``~`` in an assignment value (``W=~/x``) too."""
    if value == "~" or value.startswith("~/"):
        return home + value[1:]
    return value


class _Walker:
    def __init__(self, env: Mapping[str, str], cwd: str | None, home: str) -> None:
        self.state = _State(env=dict(env), cwd=cwd, home=home)
        self.result = Resolution()

    # -- entry -------------------------------------------------------------
    def walk(self, command: str, depth: int = 0) -> None:
        if depth > _MAX_PEEL_DEPTH:
            if re.search(r"\b(rm|rmdir|unlink|shred|find|mv|rsync|truncate)\b", command):
                self.result.opaque_reasons.append("nested shell too deep to resolve")
            return
        for text, _piped, _op in _split(command):
            self._segment(text, depth)

    # -- one segment ---------------------------------------------------------
    def _segment(self, text: str, depth: int) -> None:
        text = _strip_grouping(text)
        if not text:
            return
        raw_words, redirects = scan(text)
        st = self.state
        words = [expand(w, st.env, st.home) for w in raw_words]
        for w in words:
            for body in w.substitutions:
                if body.strip() and body.strip() not in _EMPTY_BODIES:
                    self.walk(body, depth + 1)
        for op, raw in redirects:
            if op in (">", ">|"):
                exp = expand(raw, st.env, st.home)
                if exp.text.startswith("/dev/"):
                    continue
                self._add(exp, "overwrite", ">")
        # Leading prefix words and assignments.
        while words and words[0].text in _PREFIX_WORDS and not words[0].opaque:
            words = words[1:]
        raw_assigns: list[str] = []
        while words and _ASSIGN_RE.match(words[0].text):
            raw_assigns.append(words[0].text)
            words = words[1:]
        if not words:
            # A bare assignment segment sets a shell variable for what follows.
            for a in raw_assigns:
                name, _, value = a.partition("=")
                st.env[name] = _tilde_value(value, st.home)
            return
        name = os.path.basename(words[0].text)
        if name in _DECLARE_WORDS:
            for w in words[1:]:
                m = _ASSIGN_RE.match(w.text)
                if m:
                    st.env[m.group(1)] = _tilde_value(w.text[m.end() :], st.home)
            return
        if name in ("cd", "pushd"):
            self._cd(words[1:])
            return
        if name == "popd":
            st.cwd = None
            return
        if name == "for" and len(words) >= 2:
            # A loop variable is a runtime value.
            st.env[words[1].text] = OPAQUE_MARK
            return
        if name == "read":
            for w in words[1:]:
                if not w.text.startswith("-"):
                    st.env[w.text] = OPAQUE_MARK
            return
        words, stdin_fed = _peel_wrappers(words)
        if not words:
            return
        name = os.path.basename(words[0].text)
        args = words[1:]
        if name in _SHELL_C_WORDS:
            self._shell_c(args, depth)
            return
        if name == "eval":
            inner = " ".join(w.text for w in args)
            if any(w.opaque for w in args):
                if re.search(r"\b(rm|rmdir|unlink|shred|find|mv|rsync)\b", inner):
                    self.result.opaque_reasons.append("eval of an unresolvable command")
                return
            self.walk(inner, depth + 1)
            return
        if name in DELETE_PROGRAMS:
            flags_with_value = _SHRED_VALUE_FLAGS if name == "shred" else _RM_VALUE_FLAGS
            operands, _flags = _operands(args, flags_with_value)
            if stdin_fed:
                self._add_stdin(name)
            for w in operands:
                self._add(w, "delete", name)
            return
        if name == "find":
            self._find(args)
            return
        if name in ("mv", "gmv"):
            self._mv(args, stdin_fed)
            return
        if name in ("cp", "gcp", "install"):
            operands, _ = _operands(args, _MV_VALUE_FLAGS)
            if len(operands) >= 2:
                self._add_overwrite_dest(operands[-1], name)
            return
        if name == "rsync":
            self._rsync(args)
            return
        if name == "git":
            self._git(args)
            return
        if name in ("truncate", "gtruncate"):
            operands, _ = _operands(args, _TRUNCATE_VALUE_FLAGS)
            for w in operands:
                self._add(w, "overwrite", name)
            return
        if name == "tee":
            operands, flags = _operands(args)
            if not _has_short(flags, "a") and "--append" not in flags:
                for w in operands:
                    self._add(w, "overwrite", name)
            return
        if name == "dd":
            for w in args:
                if w.text.startswith("of="):
                    w2 = Expanded(text=w.text[3:], empty=w.empty, opaque=w.opaque)
                    self._add(w2, "overwrite", name)
            return

    # -- program handlers ------------------------------------------------------
    def _cd(self, args: list[Expanded]) -> None:
        st = self.state
        operands = [w for w in args if not (w.text.startswith("-") and len(w.text) > 1)]
        if not operands or (operands[0].vanished and not operands[0].opaque):
            st.cwd = st.home
            return
        w = operands[0]
        if w.opaque or w.pattern is not None or w.text == "-":
            st.cwd = None
            return
        path, _unknown = _absolute(w.text, st.cwd)
        st.cwd = path

    def _shell_c(self, args: list[Expanded], depth: int) -> None:
        for i, w in enumerate(args):
            if w.text in ("-c", "--command") and i + 1 < len(args):
                inner = args[i + 1]
                if inner.opaque:
                    self.result.opaque_reasons.append("shell -c with an unresolvable script")
                    return
                self.walk(inner.text, depth + 1)
                return
            if not w.text.startswith("-"):
                return

    def _find(self, args: list[Expanded]) -> None:
        roots: list[Expanded] = []
        i = 0
        while i < len(args) and not (
            args[i].text.startswith("-") or args[i].text in ("(", "!", "\\(")
        ):
            roots.append(args[i])
            i += 1
        rest = args[i:]
        texts = [w.text for w in rest]
        destructive = "-delete" in texts
        for j, t in enumerate(texts):
            if t in _FIND_EXEC and j + 1 < len(texts):
                prog = os.path.basename(texts[j + 1])
                if prog in DELETE_PROGRAMS or prog in _SHELL_C_WORDS:
                    destructive = True
        if not destructive:
            return
        if not roots:
            roots = [Expanded(text=".")]
        for w in roots:
            self._add(w, "delete", "find")

    def _mv(self, args: list[Expanded], stdin_fed: bool) -> None:
        operands, flags = _operands(args, _MV_VALUE_FLAGS)
        target_dir = None
        for k, w in enumerate(args):
            if w.text in ("-t", "--target-directory") and k + 1 < len(args):
                target_dir = args[k + 1]
            elif w.text.startswith("--target-directory="):
                target_dir = w
        if stdin_fed:
            self._add_stdin("mv")
        if target_dir is not None:
            sources = operands
        else:
            sources = operands[:-1]
            if len(operands) >= 2:
                self._add_overwrite_dest(operands[-1], "mv")
        for w in sources:
            self._add(w, "move", "mv")

    def _rsync(self, args: list[Expanded]) -> None:
        operands, flags = _operands(args, _RSYNC_VALUE_FLAGS)
        if not any(f.startswith("--delete") or f == "--remove-source-files" for f in flags):
            return
        if "--remove-source-files" in flags:
            for w in operands[:-1]:
                self._add(w, "move", "rsync")
        if any(f.startswith("--delete") for f in flags) and operands:
            dest = operands[-1]
            if ":" in dest.text.split("/", 1)[0]:
                return  # remote destination: not this machine's paths
            self._add(dest, "clean", "rsync")

    def _git(self, args: list[Expanded]) -> None:
        cwd_override: Expanded | None = None
        while args and args[0].text.startswith("-"):
            if args[0].text == "-C" and len(args) > 1:
                cwd_override = args[1]
                args = args[2:]
                continue
            args = args[1:]
        if not args or args[0].text != "clean":
            return
        _operands_, flags = _operands(args[1:])
        if _has_short(flags, "n") or "--dry-run" in flags:
            return
        if not (_has_short(flags, "xXd")):
            return
        root = cwd_override or Expanded(text=".")
        self._add(root, "clean", "git clean")

    # -- recording ---------------------------------------------------------------
    def _add(self, exp: Expanded, op: str, program: str) -> None:
        if exp.vanished and not exp.empty:
            return
        self.result.targets.append(_target(exp, op, program, self.state))

    def _add_overwrite_dest(self, exp: Expanded, program: str) -> None:
        if exp.opaque or exp.pattern is not None:
            return
        path, _ = _absolute(exp.text, self.state.cwd)
        if path is not None and os.path.isdir(path):
            return  # moving/copying INTO a directory replaces nothing named here
        self._add(exp, "overwrite", program)

    def _add_stdin(self, program: str) -> None:
        self.result.targets.append(
            Target(
                path=self.state.cwd,
                op="delete",
                program=f"xargs {program}",
                opaque=True,
                unknown_cwd=self.state.cwd is None,
            )
        )


def resolve_command(
    command: str, *, cwd: str | None, env: Mapping[str, str], home: str
) -> Resolution:
    """Every path ``command`` would destroy, resolved statically. Never runs it."""
    walker = _Walker(env, cwd, home)
    try:
        walker.walk(command)
    except Exception as exc:  # noqa: BLE001 — a parser fault must not wave a call through
        walker.result.opaque_reasons.append(f"resolver error: {type(exc).__name__}")
    return walker.result
