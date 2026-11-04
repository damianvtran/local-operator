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
    _unquote,
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
#: Same, for a ``$(mktemp …)`` value carried through a variable (R6): still
#: unknowable, but known to be a fresh entry under the temp directory.
TEMP_MARK = "\x02"


# ---------------------------------------------------------------------------
# Scanning: a segment into raw words (parts) and redirections
# ---------------------------------------------------------------------------


@dataclass
class _Part:
    #: "bare" (unquoted: tilde, brace, glob apply), "dq" (double-quoted text),
    #: "sq" (single-quoted or escaped: literal), "sub" (a command-substitution
    #: body), "proc" (a process-substitution body: runs, expands to a /dev/fd
    #: path), "pbare"/"pdq" (one ``$NAME``/``${…}`` form, unquoted/quoted —
    #: atomic so brace expansion never splits a ``${…}``).
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
    pkind = "pbare" if kind == "bare" else "pdq"
    if nxt == "{":
        end = _match_brace(text, i + 2)
        if end == -1:
            word.broken = True
            return n
        word.parts.append(_Part(pkind, text[i : end + 1]))
        return end + 1
    m = _NAME_RE.match(text, i + 1)
    if m is not None:
        word.parts.append(_Part(pkind, text[i : m.end()]))
        return m.end()
    if nxt and nxt in "@*#?$!-0123456789":
        word.parts.append(_Part(pkind, text[i : i + 2]))
        return i + 2
    word.add(kind, "$")
    return i + 1


def _match_brace(text: str, i: int) -> int:
    """Index of the ``}`` closing a ``${`` whose body starts at ``i``; -1 if none.

    Depth- and quote-aware so a nested default (``${A:-${B}}``) stays one form.
    """
    depth = 1
    quote: str | None = None
    j = i
    while j < len(text):
        ch = text[j]
        if quote:
            if ch == quote:
                quote = None
            elif ch == "\\" and quote == '"':
                j += 1
        elif ch in ("'", '"'):
            quote = ch
        elif ch == "\\":
            j += 1
        elif ch == "{":
            depth += 1
        elif ch == "}":
            depth -= 1
            if depth == 0:
                return j
        j += 1
    return -1


_ANSI_ESCAPES = {
    "a": "\a",
    "b": "\b",
    "e": "\x1b",
    "E": "\x1b",
    "f": "\f",
    "n": "\n",
    "r": "\r",
    "t": "\t",
    "v": "\v",
    "\\": "\\",
    "'": "'",
    '"': '"',
    "?": "?",
}
_ANSI_RE = re.compile(
    r"\\(x[0-9a-fA-F]{1,2}|u[0-9a-fA-F]{1,4}|U[0-9a-fA-F]{1,8}|[0-7]{1,3}|.)", re.S
)


def _decode_ansi_c(body: str) -> str:
    """Decode a ``$'…'`` body the way bash does (the escapes a model emits)."""

    def one(m: re.Match[str]) -> str:
        tok = m.group(1)
        if tok[0] in "xuU" and len(tok) > 1:
            return chr(int(tok[1:], 16))
        if tok[0] in "01234567":
            return chr(int(tok, 8) & 0xFF)
        return _ANSI_ESCAPES.get(tok, "\\" + tok)

    return _ANSI_RE.sub(one, body)


def _ansi_c_end(text: str, i: int) -> int:
    """Index of the closing ``'`` of a ``$'`` body starting at ``i``; -1 if none."""
    j = i
    while j < len(text):
        if text[j] == "\\":
            j += 2
            continue
        if text[j] == "'":
            return j
        j += 1
    return -1


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
        if ch in "<>" and segment[i + 1 : i + 2] == "(":
            # Process substitution ``<(…)``/``>(…)``: the body RUNS as a command
            # and the word becomes a /dev/fd path, never a file we write.
            body, j, ok = _read_sub(segment, i + 2)
            if cur is None:
                cur = _RawWord()
            cur.parts.append(_Part("proc", body))
            cur.broken |= not ok
            i = j
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
        if ch == "\\" and segment[i + 1 : i + 2] == "\n":
            i += 2
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
            # Backslash-newline is a line continuation: it vanishes (R7).
            if i + 1 < n and segment[i + 1] != "\n":
                cur.add("sq", segment[i + 1])
            i += 2
            continue
        if ch == "$" and segment[i + 1 : i + 2] == "'":
            # ANSI-C quoting: decoded, then literal (R5).
            end = _ansi_c_end(segment, i + 2)
            if end == -1:
                cur.broken = True
                i = n
            else:
                cur.add("sq", _decode_ansi_c(segment[i + 2 : end]))
                i = end + 1
            continue
        if ch == "$" and segment[i + 1 : i + 2] == '"':
            # Locale translation ``$"…"`` behaves as a double-quoted string.
            i += 1
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
    #: The opaque part is a ``$(mktemp …)`` path: a fresh temp entry (R6).
    temp: bool = False


def _glob_escape(text: str) -> str:
    return re.sub(r"([*?\[])", r"[\1]", text)


#: Upper bound on the words one brace expression may produce. Past it the word
#: is opaque: ``{1..100000}`` is not something to enumerate per bash call.
MAX_BRACE_RESULTS = 1024
_BRACE_RANGE_RE = re.compile(r"^(-?\d+|[A-Za-z])\.\.(-?\d+|[A-Za-z])(?:\.\.(-?\d+))?$")


def _brace_alternatives(body: str) -> list[str] | None:
    """The alternatives of one ``{…}`` body, or ``None`` when it is not a brace
    expression (bash then leaves it literal: ``{a}``, ``{}``)."""
    m = _BRACE_RANGE_RE.match(body)
    if m:
        a, b, step = m.group(1), m.group(2), m.group(3)
        if a.lstrip("-").isdigit() and b.lstrip("-").isdigit():
            lo, hi = int(a), int(b)
            inc = abs(int(step)) if step else 1
            inc = inc or 1
            count = abs(hi - lo) // inc + 1
            if count > MAX_BRACE_RESULTS:
                raise _TooMany
            rng = range(lo, hi + 1, inc) if lo <= hi else range(lo, hi - 1, -inc)
            return [str(x) for x in rng]
        if a.isalpha() and b.isalpha():
            lo, hi = ord(a), ord(b)
            rng = range(lo, hi + 1) if lo <= hi else range(lo, hi - 1, -1)
            return [chr(x) for x in rng]
        return None
    alts: list[str] = []
    depth = 0
    start = 0
    for k, ch in enumerate(body):
        if ch == "{":
            depth += 1
        elif ch == "}":
            depth -= 1
        elif ch == "," and depth == 0:
            alts.append(body[start:k])
            start = k + 1
    if not alts:
        return None
    alts.append(body[start:])
    return alts


class _TooMany(Exception):
    pass


def _brace_expand_text(text: str) -> list[str]:
    """Bash brace expansion over one unquoted text run (``\\x01`` marks atoms)."""
    depth = 0
    open_at = -1
    for k, ch in enumerate(text):
        if ch == "{":
            if depth == 0:
                open_at = k
            depth += 1
        elif ch == "}" and depth:
            depth -= 1
            if depth == 0:
                alts = _brace_alternatives(text[open_at + 1 : k])
                if alts is None:
                    continue
                head, tail = text[:open_at], text[k + 1 :]
                out: list[str] = []
                for alt in alts:
                    for expanded_alt in _brace_expand_text(alt):
                        for rest in _brace_expand_text(tail):
                            out.append(head + expanded_alt + rest)
                            if len(out) > MAX_BRACE_RESULTS:
                                raise _TooMany
                return out
    return [text]


def _brace_variants(parts: list[_Part]) -> list[list[_Part]] | None:
    """Every part list a word brace-expands to; ``None`` when past the bound.

    Only ``bare`` text takes part in braces. Other parts (quoted text, ``$…``,
    substitutions) are atoms: they are swapped for placeholder characters while
    the brace structure is found, then restored in each variant.
    """
    if not any(p.kind == "bare" and "{" in p.text for p in parts):
        return [parts]
    atoms: list[_Part] = []
    flat: list[str] = []
    for p in parts:
        if p.kind == "bare":
            flat.append(p.text)
        else:
            flat.append(chr(0xE000 + len(atoms)))
            atoms.append(p)
    try:
        texts = _brace_expand_text("".join(flat))
    except (_TooMany, RecursionError):
        return None
    variants: list[list[_Part]] = []
    for t in texts:
        out: list[_Part] = []
        buf = ""
        for ch in t:
            code = ord(ch) - 0xE000
            if 0 <= code < len(atoms):
                if buf:
                    out.append(_Part("bare", buf))
                    buf = ""
                out.append(atoms[code])
            else:
                buf += ch
        if buf:
            out.append(_Part("bare", buf))
        variants.append(out)
    return variants


def _tilde(text: str, home: str) -> tuple[str, bool]:
    """Leading tilde expansion; ``(text, opaque)`` — ``~user`` is unknowable."""
    if not text.startswith("~"):
        return text, False
    head, sep, tail = text.partition("/")
    if head == "~":
        return home + sep + tail, False
    if head in ("~+", "~-"):
        return OPAQUE_MARK + sep + tail, True
    return OPAQUE_MARK + sep + tail, True


def _param(form: str, env: Mapping[str, str], home: str, out: Expanded, depth: int = 0) -> str:
    """Expand one ``$NAME`` / ``${…}`` form; unknowable forms become opaque."""
    if form.startswith("${"):
        inner = form[2:-1]
    else:
        inner = form[1:]
    if _NAME_RE.fullmatch(inner):
        value = env.get(inner)
        if value is None or value == "":
            out.empty.append(f"${inner} expanded empty" + ("" if value == "" else " (unset)"))
            return ""
        if TEMP_MARK in value:
            out.temp = True
            value = value.replace(TEMP_MARK, OPAQUE_MARK)
        if OPAQUE_MARK in value:
            out.opaque = True
            out.opaque_contained = False
        return value
    dm = _BRACE_DEFAULT_RE.match(inner)
    if dm is not None and depth < 4:
        name, op, default = dm.group(1), dm.group(2), dm.group(3)
        current = env.get(name)
        use_default = current is None or (op.startswith(":") and current == "")
        if not use_default and current is not None:
            if OPAQUE_MARK in current:
                out.opaque = True
                out.opaque_contained = False
            return current
        # Bash expands the default word itself: ``$``, a leading ``~`` and
        # nested ``${}`` (S1 — the incident shape was an unset variable plus a
        # fallback). Its quotes are stripped as part of that word.
        raw_words, _ = scan(default.strip() or "''")
        pieces: list[str] = []
        for raw in raw_words or [_RawWord()]:
            sub = expand_one(raw.parts, env, home, depth + 1)
            out.empty.extend(sub.empty)
            out.substitutions.extend(sub.substitutions)
            if sub.opaque:
                out.opaque = True
                out.opaque_contained = out.opaque_contained and sub.opaque_contained
            pieces.append(sub.text)
        value = " ".join(pieces)
        if not value:
            out.empty.append(f"${{{name}{op}…}} expanded empty")
        return value
    # ``${#X}``, ``${X%…}``, ``${X/…}``, ``${X:+…}``, ``$1``, ``$@``…: runtime values.
    out.opaque = True
    out.opaque_contained = False
    return OPAQUE_MARK


def expand_one(
    parts: list[_Part], env: Mapping[str, str], home: str, depth: int = 0, broken: bool = False
) -> Expanded:
    """Expand ONE brace variant of a word (tilde, parameters, substitutions)."""
    out = Expanded(text="")
    text_parts: list[str] = []
    pattern_parts: list[str] = []
    globbed = False
    literal_seen = False
    for index, part in enumerate(parts):
        if part.kind in ("sub", "proc"):
            body = part.text.strip()
            out.substitutions.append(part.text)
            if part.kind == "proc":
                # Expands to a /dev/fd path: never a target the command names.
                text_parts.append("/dev/fd/63")
                pattern_parts.append("/dev/fd/63")
                literal_seen = True
                continue
            if body in _EMPTY_BODIES:
                out.empty.append(f"$({body}) expanded empty")
                continue
            out.opaque = True
            out.opaque_contained = out.opaque_contained and _contained_body(body)
            if _is_mktemp(body):
                out.temp = True
            text_parts.append(OPAQUE_MARK)
            pattern_parts.append(OPAQUE_MARK)
            continue
        if part.kind == "sq":
            text_parts.append(part.text)
            pattern_parts.append(_glob_escape(part.text))
            literal_seen = True
            continue
        if part.kind in ("pbare", "pdq"):
            value = _param(part.text, env, home, out, depth)
            if part.kind == "pdq":
                literal_seen = True
                text_parts.append(value)
                pattern_parts.append(_glob_escape(value))
            else:
                literal_seen = literal_seen or bool(value)
                if any(c in value for c in "*?["):
                    globbed = True
                text_parts.append(value)
                pattern_parts.append(value)
            continue
        raw = part.text
        if part.kind == "bare" and index == 0:
            raw, opaque = _tilde(raw, home)
            if opaque:
                out.opaque = True
                out.opaque_contained = False
        if part.kind == "dq":
            text_parts.append(raw)
            pattern_parts.append(_glob_escape(raw))
            literal_seen = True
            continue
        literal_seen = literal_seen or bool(raw)
        if any(c in raw for c in "*?["):
            globbed = True
        text_parts.append(raw)
        pattern_parts.append(raw)
    out.text = "".join(text_parts)
    out.pattern = "".join(pattern_parts) if globbed else None
    out.opaque = out.opaque or OPAQUE_MARK in out.text or broken
    if broken:
        out.opaque_contained = False
    out.vanished = not out.text and not literal_seen and not broken
    return out


def expand_all(word: _RawWord, env: Mapping[str, str], home: str) -> list[Expanded]:
    """Every word ``word`` becomes after brace expansion (R3), each expanded.

    Past :data:`MAX_BRACE_RESULTS` the word is ONE opaque result: unknowable,
    never assumed safe.
    """
    variants = _brace_variants(word.parts)
    if variants is None:
        out = expand_one(word.parts, env, home, broken=True)
        out.opaque = True
        out.opaque_contained = False
        out.empty.append("brace expansion too large to enumerate")
        return [out]
    return [expand_one(v, env, home, broken=word.broken) for v in variants]


def expand(word: _RawWord, env: Mapping[str, str], home: str) -> Expanded:
    """The first brace variant (callers that need one word: program names, cd)."""
    return expand_all(word, env, home)[0]


#: Programs whose output is a single path component in normal use, so a
#: substitution of one cannot climb out of the directory it sits in — but only
#: when called with no ``+FORMAT``/path-like argument (S4: ``date +../..``).
def _contained_body(body: str) -> bool:
    words = body.split()
    if not words or os.path.basename(words[0]) not in _CONTAINED_PROGRAMS:
        return False
    # S4: an argument that could print a path (``date +../..``, ``id /x``) voids
    # containment; a plain ``+%s``-style format cannot.
    return not any("/" in w or "." in w for w in words[1:])


def _is_mktemp(body: str) -> bool:
    words = body.split()
    return (
        bool(words)
        and os.path.basename(words[0]) == "mktemp"
        and not any("/" in w or ".." in w for w in words[1:] if not w.startswith("-"))
    )


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
    #: The word globs its CHILDREN (``dir/*.log``): ``path`` is the directory
    #: the glob lives in, and the directory itself is not removed (Q2).
    glob_children: bool = False
    #: The glob matches every entry of ``path`` (``*``, ``.*``): as good as
    #: removing the directory's contents wholesale.
    glob_all: bool = False
    #: ``find ROOT <selecting predicate> -delete``: removes a SUBSET of ROOT's
    #: descendants, so ROOT is ranked only against CRITICAL ancestry (R6).
    selected: bool = False
    #: The opaque part is a fresh ``$(mktemp …)`` entry (R6).
    temp: bool = False
    #: A ``..`` follows an unknowable component: the real target is somewhere
    #: ABOVE ``path`` by an unknown amount (S4/S2-5b), so ``path`` must not be
    #: presented to the judge as where the deletion lands.
    climbs: bool = False


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
    contained = opaque and exp.opaque_contained
    climbs = False
    if opaque and OPAQUE_MARK in text:
        # S4: a ``..`` AFTER an unknowable component climbs by an unknown amount
        # from an unknown place; containment is void.
        tail = text[text.index(OPAQUE_MARK) :]
        if ".." in tail.split("/"):
            contained = False
            climbs = True
    glob_children = False
    glob_all = False
    if pattern is not None and not opaque:
        last = text.rstrip("/").rsplit("/", 1)[-1]
        head = text.rstrip("/")[: -len(last)] if last else text
        if not any(c in head for c in "*?["):
            glob_children = True
            glob_all = last in ("*", ".*", "*.*") or set(last) <= {"*", "?"}
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
    is_temp = (
        exp.temp
        and opaque
        and text.startswith(OPAQUE_MARK)
        and ".." not in text.split("/")
        and not exp.empty
    )
    if is_temp:
        # ``mktemp`` prints an ABSOLUTE path under the temp directory, so the
        # word is anchored there — not at the cwd the empty literal prefix
        # would otherwise resolve to.
        import tempfile

        path, unknown = os.path.realpath(tempfile.gettempdir()), False
    return Target(
        path=path,
        op=op,
        program=program,
        pattern=pattern,
        empty=list(exp.empty),
        opaque=opaque,
        opaque_contained=contained,
        unknown_cwd=unknown,
        glob_children=glob_children,
        glob_all=glob_all,
        temp=is_temp,
        climbs=climbs,
    )


@dataclass
class Resolution:
    targets: list[Target] = field(default_factory=list)
    #: The parser met a construct it cannot follow that MAY delete (``eval`` of a
    #: deleting word, an unbalanced quote around one): reasons, for the judge.
    opaque_reasons: list[str] = field(default_factory=list)


def _operands(
    words: list[Expanded], value_flags: frozenset[str] = frozenset()
) -> tuple[list[Expanded], list[str]]:
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
    return any(not f.startswith("--") and any(c in f[1:] for c in letters) for f in flags)


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
#: find predicates that select a SUBSET of the walk (R6).
_FIND_SELECTING = frozenset(
    {
        "-name",
        "-iname",
        "-path",
        "-ipath",
        "-regex",
        "-iregex",
        "-newer",
        "-mtime",
        "-mmin",
        "-empty",
        "-size",
        "-user",
        "-perm",
        "-wholename",
        "-iwholename",
    }
)
_HEREDOC_OPEN_RE = re.compile(r"<<-?\s*(['\"]?)([A-Za-z_][A-Za-z0-9_]*)\1")


def _heredoc_bodies(command: str) -> list[str]:
    """The bodies of every heredoc in ``command`` (R2: walked when a shell reads one).

    ``search_guard._split`` strips bodies so data is never read as commands;
    this recovers them for the one case where the body IS the command.
    """
    bodies: list[str] = []
    for m in _HEREDOC_OPEN_RE.finditer(command):
        delim = m.group(2)
        nl = command.find("\n", m.end())
        if nl == -1:
            continue
        lines = command[nl + 1 :].split("\n")
        body: list[str] = []
        for line in lines:
            if line.strip() == delim:
                break
            body.append(line)
        bodies.append("\n".join(body))
    return bodies


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


#: The delete family, as words, for the "a shell runs text we cannot see" checks.
_DELETE_WORD_RE = re.compile(
    r"(?<![\w.-])(rm|grm|rmdir|unlink|shred|truncate|rsync|mv|find|rmtree|unlinkSync|"
    r"remove|removedirs|rimraf)(?![\w-])"
)
#: Interpreter one-liner flags whose argument is code (Q3 — v1 only scans the
#: text for a delete call and ranks it HIGH; v2 may parse the language).
_INTERPRETER_CODE_FLAGS = {
    "python": ("-c",),
    "python3": ("-c",),
    "python2": ("-c",),
    "perl": ("-e", "-E"),
    "ruby": ("-e",),
    "node": ("-e", "--eval"),
    "osascript": ("-e",),
}
#: A pipe from one of these into a shell runs text fetched at run time.
_NETWORK_FETCH_RE = re.compile(r"curl|wget|fetch|http|https|nc|ncat|aria2c|xh")
_INTERPRETER_DELETE_RE = re.compile(
    r"\b(rmtree|os\.remove|os\.unlink|os\.rmdir|removedirs|\.unlink\(|\.rmdir\(|"
    r"unlinkSync|rmSync|rmdirSync|File\.delete|FileUtils\.rm|unlink\s*\(|"
    r"rm\s+-|do shell script)"
)


def _in_place_edit(name: str, args: list[Expanded]) -> bool:
    """``sed -i``/``--in-place``, or perl with ``i`` in a short cluster (``-pi``)."""
    for w in args:
        t = w.text
        if t.startswith("--in-place"):
            return True
        if t.startswith("-") and not t.startswith("--"):
            # R3-3: ``i`` anywhere in a short cluster (``-Ei``, ``-ni``, ``-pi``)
            # — but only BEFORE the first script letter (R4-2): in
            # ``-e's/in/out/'`` everything after ``e`` is the script text.
            flags, _script = _split_cluster(name, t[1:])
            if "i" in flags:
                return True
    return False


def _split_cluster(name: str, cluster: str) -> tuple[str, str | None]:
    """A short-option cluster's flag letters, and its attached script if any.

    sed's script letters are ``e``/``f``; perl's are ``e``/``E``. The first one
    ends the flags: the rest of the word, if any, is its value (``-es/x/y/``).
    ``None`` means no script letter; ``""`` means the value is the NEXT word.
    """
    letters = ("e", "E") if name == "perl" else ("e", "f")
    for k, ch in enumerate(cluster):
        if ch in letters:
            return cluster[:k], cluster[k + 1 :]
    return cluster, None


@dataclass
class _Feed:
    """What reaches a pipeline stage's stdin from the stages before it."""

    #: Heredoc/herestring bodies bound to upstream stages (R3-1).
    bodies: list[str] = field(default_factory=list)
    #: The upstream stages' raw text, nearest last (R3-4).
    stages: list[str] = field(default_factory=list)


def _herestring_texts(segment: str) -> list[str]:
    """The (expanded-free) text of every ``<<<`` word in ``segment``."""
    _words, redirects = scan(segment)
    out: list[str] = []
    for op, raw in redirects:
        if op == "<<<":
            out.append("".join(p.text for p in raw.parts))
    return out


def _stage_program(stage: str) -> str | None:
    """A stage's program name after wrapper peeling (``sudo``, ``timeout 30``…)."""
    words, _ = scan(_strip_grouping(stage))
    texts = ["".join(p.text for p in w.parts) for w in words]
    while texts and _ASSIGN_RE.match(texts[0]):
        texts = texts[1:]
    peeled = [Expanded(text=t) for t in texts]
    peeled, _fed = _peel_wrappers(peeled)
    return os.path.basename(peeled[0].text) if peeled else None


def _fd_redirect_merge(pieces: list[tuple[str, bool, str]]) -> list[tuple[str, bool, str]]:
    """Re-join segments ``search_guard._split`` cut at a redirect's ``&`` (Q1).

    The shared splitter reads the ``&`` in ``2>&1``/``>&2``/``2>&-``/``&>file``
    as a background operator. For this resolver that turned every ``2>&1`` into
    an empty, opaque overwrite target. A segment ending in ``>``/``<`` cut by a
    single ``&`` is glued back to what follows, operator included.
    """
    out: list[tuple[str, bool, str]] = []
    for text, piped, op in pieces:
        if out:
            prev_text, prev_piped, prev_op = out[-1]
            if prev_op == "&" and re.search(r"[<>]$", prev_text):
                out[-1] = (prev_text + "&" + text, prev_piped, op)
                continue
        out.append((text, piped, op))
    return out


@dataclass
class _Snapshot:
    env: dict[str, str]
    cwd: str | None
    home: str


class _Walker:
    def __init__(self, env: Mapping[str, str], cwd: str | None, home: str) -> None:
        self.state = _State(env=dict(env), cwd=cwd, home=home)
        self.result = Resolution()
        #: Paths an earlier segment of THIS command made into a symlink (S7):
        #: what they point at is resolved only at run time.
        self.linked: set[str] = set()
        #: ``for VAR in a b c`` literal lists: a target using $VAR fans out.
        self.for_lists: dict[str, list[str]] = {}

    # -- state isolation (R4) ----------------------------------------------------
    def _save(self) -> _Snapshot:
        st = self.state
        return _Snapshot(dict(st.env), st.cwd, st.home)

    def _restore(self, snap: _Snapshot) -> None:
        self.state.env, self.state.cwd, self.state.home = snap.env, snap.cwd, snap.home

    def _isolated_walk(self, command: str, depth: int) -> None:
        """Walk text that runs in a CHILD shell: its cd/assignments never leak."""
        snap = self._save()
        try:
            self.walk(command, depth)
        finally:
            self._restore(snap)

    # -- entry -------------------------------------------------------------
    def walk(self, command: str, depth: int = 0) -> None:
        if depth > _MAX_PEEL_DEPTH:
            if _DELETE_WORD_RE.search(command):
                self.result.opaque_reasons.append("nested shell too deep to resolve")
            return
        # S2-4: each heredoc body belongs to the segment whose ``<<`` opened it.
        # ``_split`` blanks the bodies but keeps the openers, in order, so the
        # bodies are dealt out to segments by counting the openers in each.
        pending_bodies = list(_heredoc_bodies(command))
        pieces = _fd_redirect_merge(_split(command))
        # R3-1: what each pipeline stage writes to its stdout is the NEXT
        # stage's stdin. A heredoc/herestring bound to an upstream stage
        # (``cat <<EOF | bash``) is therefore the script a pipe-fed shell runs.
        upstream_bodies: list[str] = []
        groups: list[_Snapshot] = []
        # R4-1: whether a stage reads the previous one's stdout comes from the
        # PREVIOUS piece's operator. ``_split`` reports ``|&`` as a ``|`` run
        # followed by an ``&`` that clears its own ``piped`` flag, so the flag
        # cannot be trusted for ``|&``; the operator can.
        piped_in = [i > 0 and pieces[i - 1][2] == "|" for i in range(len(pieces))]
        # R4-1: bodies produced inside a ``{ …; }``/``( … )`` group are the
        # group's output, which a ``} |``/``) |`` closer pipes onward.
        group_bodies: list[list[str]] = []
        for index, (text, _flag, op) in enumerate(pieces):
            piped = piped_in[index]
            # R4: a subshell ``( … )``, any pipeline stage and a ``&`` job run in
            # a child process, so their cd/assignments do not reach later
            # segments. A subshell spans several pieces (``( cd x && rm y )``),
            # so it is tracked as a stack: state is saved at its ``(`` and
            # restored after the piece holding its unmatched ``)``.
            stripped = text.strip()
            while stripped.startswith("(") and not stripped.startswith("(("):
                groups.append(self._save())
                stripped = stripped[1:].lstrip()
            closes = 0
            tail = stripped
            while tail.endswith(")") and tail.count(")") > tail.count("("):
                closes += 1
                tail = tail[:-1].rstrip()
            brace_opens = 0
            head = text.strip()
            while head.startswith(("{ ", "{\n", "{\t", "(")) and not head.startswith("(("):
                brace_opens += 1
                head = head[1:].lstrip()
            for _ in range(brace_opens):
                group_bodies.append([])
            nxt_piped = index + 1 < len(pieces) and piped_in[index + 1]
            isolated = piped or nxt_piped or op == "&"
            openers = len(_HEREDOC_OPEN_RE.findall(text))
            heredocs, pending_bodies = pending_bodies[:openers], pending_bodies[openers:]
            # The whole pipeline's text, so a pipe-fed shell can ask whether
            # anything upstream names a delete (S2-2).
            start = index
            while start > 0 and piped_in[start]:
                start -= 1
            end = index
            while end + 1 < len(pieces) and piped_in[end + 1]:
                end += 1
            pipeline = " | ".join(p[0] for p in pieces[start : end + 1])
            if not piped:
                upstream_bodies = []
            feed = _Feed(
                bodies=list(upstream_bodies),
                stages=[p[0] for p in pieces[start:index]],
            )
            if isolated:
                snap = self._save()
                self._segment(
                    text, depth, heredocs, stdin_piped=piped, pipeline=pipeline, feed=feed
                )
                self._restore(snap)
            else:
                self._segment(
                    text, depth, heredocs, stdin_piped=piped, pipeline=pipeline, feed=feed
                )
            produced = heredocs + _herestring_texts(text)
            upstream_bodies = upstream_bodies + produced
            for bodies in group_bodies:
                bodies.extend(produced)
            tail_text = text.strip()
            group_closes = closes + (
                1 if tail_text == "}" or tail_text.endswith((" }", ";}")) else 0
            )
            for _ in range(min(group_closes, len(group_bodies))):
                # The closer's stage pipes the WHOLE group's output onward.
                upstream_bodies = upstream_bodies + group_bodies.pop()
            for _ in range(closes):
                if groups:
                    self._restore(groups.pop())

    # -- one segment ---------------------------------------------------------
    def _segment(
        self,
        text: str,
        depth: int,
        heredocs: list[str] | None = None,
        *,
        stdin_piped: bool = False,
        pipeline: str = "",
        feed: _Feed | None = None,
    ) -> None:
        text = _strip_grouping(text)
        if not text:
            return
        raw_words, redirects = scan(text)
        st = self.state
        words: list[Expanded] = []
        for w in raw_words:
            words.extend(expand_all(w, st.env, st.home))
        for w in words:
            for body in w.substitutions:
                if body.strip() and body.strip() not in _EMPTY_BODIES:
                    self._isolated_walk(body, depth + 1)
        herestrings: list[str] = []
        for op, raw in redirects:
            if op in (">", ">|"):
                for exp in expand_all(raw, st.env, st.home):
                    if exp.text.startswith("/dev/"):
                        continue
                    self._add(exp, "overwrite", ">")
            elif op == "<<<":
                herestrings.append(expand(raw, st.env, st.home).text)
        # Leading prefix words and assignments.
        while words and words[0].text in _PREFIX_WORDS and not words[0].opaque:
            words = words[1:]
        raw_assigns: list[str] = []
        while words and _ASSIGN_RE.match(words[0].text):
            text_value = words[0].text
            if words[0].temp and text_value.count(OPAQUE_MARK) == 1:
                text_value = text_value.replace(OPAQUE_MARK, TEMP_MARK)
            raw_assigns.append(text_value)
            words = words[1:]
        if not words:
            # A bare assignment segment sets a shell variable for what follows.
            for a in raw_assigns:
                self._assign(a)
            return
        name = os.path.basename(words[0].text)
        if name in _DECLARE_WORDS:
            for w in words[1:]:
                if _ASSIGN_RE.match(w.text):
                    self._assign(w.text)
            return
        if name in ("cd", "pushd"):
            self._cd(words[1:])
            return
        if name == "popd":
            st.cwd = None
            return
        if name == "for" and len(words) >= 2:
            self._for(words[1:])
            return
        if name == "read":
            for w in words[1:]:
                if not w.text.startswith("-"):
                    st.env[w.text] = OPAQUE_MARK
            return
        if name == "env" and any(
            w.text in ("-S", "--split-string") or w.text.startswith("-S") for w in words[1:3]
        ):
            # S6: ``env -S 'cmd …'`` runs its argument as a command line.
            joined = " ".join(w.text for w in words[1:])
            if _DELETE_WORD_RE.search(joined):
                self.result.opaque_reasons.append("env -S runs a command string")
            return
        words, stdin_fed = _peel_wrappers(words)
        if not words:
            return
        name = os.path.basename(words[0].text)
        args = words[1:]
        if name in _SHELL_C_WORDS:
            self._shell(
                args,
                depth,
                text,
                heredocs or [],
                herestrings,
                stdin_piped,
                pipeline or text,
                feed,
            )
            return
        if name in ("source", "."):
            self._source(
                args, depth, text, heredocs or [], herestrings, stdin_piped, pipeline, feed
            )
            return
        if name == "perl" and _in_place_edit(name, args):
            self._in_place(name, args)
            return
        if name in _INTERPRETER_CODE_FLAGS:
            self._interpreter(name, args)
            return
        if name in ("ssh",):
            if _DELETE_WORD_RE.search(" ".join(w.text for w in args)):
                self.result.opaque_reasons.append("remote command over ssh (not resolved)")
            return
        if name == "eval":
            inner = " ".join(w.text for w in args)
            if any(w.opaque for w in args):
                if _DELETE_WORD_RE.search(inner):
                    self.result.opaque_reasons.append("eval of an unresolvable command")
                return
            self.walk(inner, depth + 1)
            return
        if name in ("ln", "gln"):
            self._ln(args)
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
        if name in ("sed", "gsed") and _in_place_edit(name, args):
            self._in_place(name, args)
            return
        if name == "dd":
            for w in args:
                if w.text.startswith("of="):
                    # S8: bash expands ``~`` after ``=`` in an argument.
                    value = _tilde_value(w.text[3:], st.home)
                    w2 = Expanded(
                        text=value,
                        empty=w.empty,
                        opaque=w.opaque,
                        opaque_contained=w.opaque_contained,
                    )
                    self._add(w2, "overwrite", name)
            return

    def _in_place(self, name: str, args: list[Expanded]) -> None:
        """In-place edit (``sed -i``, ``perl -pi``): its files are rewritten."""
        # S5: an in-place edit rewrites its files. R2-3/S2-5a: when the
        # script came through a flag (``-e``/``-f``, or a cluster ending in
        # ``e`` such as ``-pe``), EVERY operand is a file; otherwise the
        # first operand is the script.
        files: list[Expanded] = []
        script_by_flag = False
        k = 0
        while k < len(args):
            t = args[k].text
            if t == "--":
                files.extend(args[k + 1 :])
                break
            if t.startswith("--expression=") or t.startswith("--file="):
                # R3-3: the joined long form carries its script in the same word.
                script_by_flag = True
                k += 1
                continue
            if t in ("--expression", "--file"):
                script_by_flag = True
                k += 2
                continue
            if t.startswith("-") and not t.startswith("--") and len(t) > 1:
                # R4-2/R4-3: scan the cluster letter by letter. The first script
                # letter (sed ``e``/``f``, perl ``e``/``E``; sed's ``-E`` is
                # extended regex) takes the rest of the word as its script, or
                # the next word when nothing follows it.
                _flags, attached = _split_cluster(name, t[1:])
                if attached is not None:
                    script_by_flag = True
                    k += 1 if attached else 2
                    continue
                k += 1
                continue
            if t.startswith("--"):
                k += 1
                continue
            files.append(args[k])
            k += 1
        for w in files if script_by_flag else files[1:]:
            self._add(w, "overwrite", name)
        return

    # -- program handlers ------------------------------------------------------
    def _assign(self, text: str) -> None:
        name, _, value = text.partition("=")
        value = _tilde_value(value, self.state.home)
        self.state.env[name] = value
        if name == "HOME":
            # S7: a later ``~`` resolves against the NEW home, as bash does.
            self.state.home = value or self.state.home

    def _shell(
        self,
        args: list[Expanded],
        depth: int,
        segment: str,
        heredocs: list[str],
        herestrings: list[str],
        stdin_piped: bool,
        pipeline: str = "",
        feed: _Feed | None = None,
    ) -> None:
        """A shell word: walk its ``-c`` script, or whatever feeds its stdin.

        R1/S2: ``-c`` may sit inside a short-flag cluster (``-lc``, ``-euc``).
        Option words may come before AND after it (S2-1: ``bash -c -x 'script'``
        runs ``script``), so options are skipped until the first non-option
        word. ``-o``/``+o``/``-O``/``+O`` and any cluster containing ``o``/``O``
        (``-euo pipefail``) consume the next word as their value (S2-2).

        Without ``-c`` the shell runs a script file or its stdin. A heredoc or
        herestring body is walked (R2). A pipe or a file is invisible here, and
        running a repo script through ``bash`` is everyday work, so it is opaque
        ONLY when the segment (or, for a pipe, the whole pipeline) names a
        delete, or when the pipe comes from a network fetch (S2-2/R2-1).
        """
        i = 0
        script: Expanded | None = None
        wants_c = False
        while i < len(args):
            t = args[i].text
            if t == "--":
                i += 1
                if wants_c and i < len(args):
                    script = args[i]
                break
            if t in ("--rcfile", "--init-file"):
                i += 2
                continue
            if t in ("-c", "--command"):
                wants_c = True
                i += 1
                continue
            if (t.startswith("-") or t.startswith("+")) and len(t) > 1 and not t.startswith("--"):
                cluster = t[1:]
                if "c" in cluster:
                    wants_c = True
                # ``-o pipefail``/``-euo pipefail``: the option name is a value.
                i += 2 if ("o" in cluster or "O" in cluster) else 1
                continue
            if t.startswith("--"):
                i += 1
                continue
            if wants_c:
                script = args[i]
            break  # the script (with -c) or a script file operand
        if script is not None:
            if script.opaque or not script.text.strip():
                if _DELETE_WORD_RE.search(segment):
                    self.result.opaque_reasons.append("shell -c with an unresolvable script")
                return
            self._isolated_walk(script.text, depth + 1)
            return
        if wants_c:
            if _DELETE_WORD_RE.search(segment):
                self.result.opaque_reasons.append("shell -c script could not be located")
            return
        self._walk_stdin_bodies(heredocs, herestrings, depth)
        if heredocs or herestrings:
            return
        self._piped_script(segment, stdin_piped, pipeline, feed, depth)

    def _piped_script(
        self, segment: str, stdin_piped: bool, pipeline: str, feed: _Feed | None, depth: int
    ) -> None:
        """A shell whose script is a pipe or a file (R3-1, R3-4)."""
        if stdin_piped and feed is not None and feed.bodies:
            # Upstream heredoc/herestring text IS the script: walk it.
            self._walk_stdin_bodies(feed.bodies, [], depth)
            return
        self._unseen_script(segment, stdin_piped, pipeline, feed)

    def _walk_stdin_bodies(self, heredocs: list[str], herestrings: list[str], depth: int) -> None:
        for body in list(herestrings) + list(heredocs):
            self._isolated_walk(body, depth + 1)

    def _unseen_script(
        self, segment: str, stdin_piped: bool, pipeline: str, feed: _Feed | None = None
    ) -> None:
        """Record a shell running text this command does not contain — if it matters."""
        haystack = pipeline if stdin_piped else segment
        # R3-4: a fetch is a PROGRAM upstream (``curl …``, ``timeout 30 wget``),
        # not a word in an argument (``cat ncat.log``, ``echo http://x``).
        fetched = (
            stdin_piped
            and feed is not None
            and any(
                _NETWORK_FETCH_RE.fullmatch(_stage_program(stage) or "") for stage in feed.stages
            )
        )
        if _DELETE_WORD_RE.search(haystack) or fetched:
            self.result.opaque_reasons.append(
                "shell reads a script from stdin or a file (not resolved)"
            )

    def _source(
        self,
        args: list[Expanded],
        depth: int,
        segment: str,
        heredocs: list[str],
        herestrings: list[str],
        stdin_piped: bool,
        pipeline: str,
        feed: _Feed | None = None,
    ) -> None:
        """``source``/``.``: process substitutions were already walked as
        commands; a ``/dev/stdin`` source reads its heredoc/herestring or pipe."""
        if not args:
            return
        target = args[0].text
        for body in args[0].substitutions:
            # ``source <(producer)`` runs the producer's OUTPUT. An echo/printf
            # producer's payload is visible text — walk it; any other producer
            # is opaque when it names a delete.
            words = body.split(None, 1)
            if words and words[0] in ("echo", "printf") and len(words) > 1:
                self._isolated_walk(_unquote(words[1].strip()), depth + 1)
            elif _DELETE_WORD_RE.search(body):
                self.result.opaque_reasons.append("source of generated text (not resolved)")
        if target in ("/dev/stdin", "-", "/dev/fd/0"):
            self._walk_stdin_bodies(heredocs, herestrings, depth)
            if not (heredocs or herestrings):
                self._piped_script(segment, stdin_piped, pipeline or segment, feed, depth)

    def _interpreter(self, name: str, args: list[Expanded]) -> None:
        """Q3 (v1 hardening): an interpreter one-liner whose code calls a delete."""
        flags = _INTERPRETER_CODE_FLAGS.get(name, ())
        for k, w in enumerate(args):
            if w.text in flags and k + 1 < len(args):
                if _INTERPRETER_DELETE_RE.search(args[k + 1].text):
                    self.result.opaque_reasons.append(
                        f"{name} one-liner calls a delete (not resolved)"
                    )
                return

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

    def _for(self, words: list[Expanded]) -> None:
        """``for VAR in WORDS``: R6 — literal lists bind VAR to their values."""
        st = self.state
        var = words[0].text
        values: list[Expanded] = []
        seen_in = False
        for w in words[1:]:
            if not seen_in:
                seen_in = w.text == "in"
                continue
            if w.text.startswith("for") or w.text == "in":
                continue
            values.append(w)
        if not values:
            st.env[var] = OPAQUE_MARK
            return
        if any(v.opaque or v.pattern is not None for v in values):
            st.env[var] = OPAQUE_MARK
            return
        # The body runs once per value; the walker sees it once, so the variable
        # carries EVERY value and ``_add`` fans a target out per value.
        st.env[var] = values[0].text
        self.for_lists[var] = [v.text for v in values]

    def _ln(self, args: list[Expanded]) -> None:
        operands, flags = _operands(args, frozenset({"-t", "--target-directory"}))
        if not (_has_short(flags, "s") or "--symbolic" in flags) or len(operands) < 2:
            return
        link = operands[-1]
        path, _ = _absolute(link.text, self.state.cwd)
        if path is not None:
            self.linked.add(path.rstrip("/"))

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
        # R6: a selecting predicate before the action narrows what is removed to
        # a subset of descendants; the root is then ranked only for CRITICAL
        # ancestry (a confused predicate under the state root is still refused).
        selected = False
        for j, t in enumerate(texts):
            if t in ("-delete", *_FIND_EXEC):
                break
            if t in _FIND_SELECTING:
                selected = True
            # ``-type f``/``-mindepth`` are NOT selecting (S2-3): they still
            # empty the directory, just as ``rm -rf X/*`` would.
        if not roots:
            roots = [Expanded(text=".")]
        for w in roots:
            self._add(w, "delete", "find", selected=selected)

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
    def _add(self, exp: Expanded, op: str, program: str, *, selected: bool = False) -> None:
        if exp.vanished and not exp.empty:
            return
        target = _target(exp, op, program, self.state)
        target.selected = selected
        if target.path is not None:
            stripped = target.path.rstrip("/") or "/"
            if any(stripped == ln or stripped.startswith(ln + "/") for ln in self.linked):
                # S7: a symlink made earlier in this same command — its pointee
                # is decided at run time.
                target.opaque = True
                target.opaque_contained = False
        self.result.targets.append(target)
        # R6: a literal ``for`` list binds its variable to each value in turn.
        for var, values in self.for_lists.items():
            first = values[0]
            if not first or first not in exp.text or exp.text.count(first) != 1:
                continue
            for other in values[1:]:
                alt = Expanded(
                    text=exp.text.replace(first, other),
                    pattern=exp.pattern.replace(first, other) if exp.pattern else None,
                    empty=list(exp.empty),
                    opaque=exp.opaque,
                    opaque_contained=exp.opaque_contained,
                    temp=exp.temp,
                )
                extra = _target(alt, op, program, self.state)
                extra.selected = selected
                self.result.targets.append(extra)
            del var

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
