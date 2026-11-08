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

import fnmatch
import os
import re
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field

from local_operator.tools.search_guard import (
    _DURATION_RE,
    _MAX_PEEL_DEPTH,
    _SHELL_C_WORDS,
    _WRAPPER_LONG_VALUE_FLAGS,
    _WRAPPER_VALUE_FLAGS,
    WRAPPER_WORDS,
    _unquote,
    lex,
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
# Analysis-only provenance; contains the ordinary opaque marker so existing
# expansion/climb rules still apply without ever retaining a credential value.
CREDENTIAL_MARK = OPAQUE_MARK + "credential" + OPAQUE_MARK
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


def scan(
    segment: str, *, heredoc: bool = False
) -> tuple[list[_RawWord], list[tuple[str, _RawWord]]]:
    """Split a segment, or tokenize expansion-active heredoc data.

    Heredoc data has literal quotes/operators/comments but the SAME dollar,
    backtick and escape rules below. Reuse those readers, not another shell
    scanner that could disagree about private-variable references (R19-1).
    """
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
        if heredoc and ch not in "$\\`":
            if cur is None:
                cur = _RawWord()
            cur.add("sq", ch)
            i += 1
            continue
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
        if not heredoc and ch == "$" and segment[i + 1 : i + 2] == "'":
            # ANSI-C quoting: decoded, then literal (R5).
            end = _ansi_c_end(segment, i + 2)
            if end == -1:
                cur.broken = True
                i = n
            else:
                cur.add("sq", _decode_ansi_c(segment[i + 2 : end]))
                i = end + 1
            continue
        if not heredoc and ch == "$" and segment[i + 1 : i + 2] == '"':
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
    # Unsupported operations remain unknown, but they still read their base
    # parameter. Never evaluate a private value to implement the operation.
    # A known indirect NAME can be followed without inventing Bash evaluation.
    base = inner.lstrip("#!")
    name = _NAME_RE.match(base)
    private = False
    if name is not None:
        value = env.get(name.group(), "")
        if inner.startswith("!") and _NAME_RE.fullmatch(value):
            value = env.get(value, "")
        private = CREDENTIAL_MARK in value or _private_reads(
            base[name.end() :], env, home, depth + 1
        )
    out.opaque = True
    out.opaque_contained = False
    return CREDENTIAL_MARK if private else OPAQUE_MARK


def _private_reads(body: str, env: Mapping[str, str], home: str, depth: int) -> bool:
    """Explicit expansion dependencies, not arbitrary program-output dataflow.

    R19-1: a substitution's output stays opaque, but a parsed private read must
    survive on that VALUE. The existing lexer owns comments/body attribution;
    the existing tokenizer owns quotes/escapes. Literal heredoc data and quoted
    symbols are not reads. Nothing here walks or taints an unrelated target.
    """
    if "$" not in body and "`" not in body:
        return False
    parsed = lex(body)
    for (segment, _piped, _op), bodies in zip(parsed.segments, parsed.bodies):
        words, redirects = scan(segment)
        active = words + [raw for op, raw in redirects if op not in ("<<", "<<-")]
        if any(CREDENTIAL_MARK in expand_one(w.parts, env, home, depth).text for w in active):
            return True
        delimiters = [raw for op, raw in redirects if op in ("<<", "<<-")]
        for delimiter, data in zip(delimiters, bodies):
            if any(part.kind != "bare" for part in delimiter.parts):
                continue
            data_words, _ = scan(data, heredoc=True)
            if any(
                CREDENTIAL_MARK in expand_one(w.parts, env, home, depth).text for w in data_words
            ):
                return True
    return False


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
            marker = CREDENTIAL_MARK if _private_reads(body, env, home, depth + 1) else OPAQUE_MARK
            text_parts.append(marker)
            pattern_parts.append(marker)
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
    #: ``append`` is a write that destroys nothing (``>>``, ``tee -a``,
    #: ``sqlite3 FILE``, a copy INTO a directory). It is ranked only where the
    #: write itself is the danger: the guard's own evidence (transcripts,
    #: lineage, ask store, ledger) — see ``classify._evidence_reason``.
    op: str  # "delete" | "move" | "overwrite" | "clean" | "append"
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
    #: The word as WRITTEN has a ``..`` component. ``path`` is normalised
    #: lexically, but the kernel resolves ``..`` physically AFTER following a
    #: link, so ``<pad>/sub/../<id>/asks.jsonl`` with ``sub`` -> a session dir
    #: lands in that session's ask store while ``path`` reads as inside the
    #: pad (review R1-MAJOR-1, PR #3602). ``classify`` therefore never grants
    #: a scratchpad exemption — any pad, this session's own included — to a
    #: target carrying this flag; see ``classify._pad_exemptible``.
    dotdot: bool = False
    #: The absolute but UN-normalised join of the word (``None`` for a glob or
    #: opaque word). ``os.path.realpath`` of it resolves ``..`` physically, so
    #: ``classify`` adds that spelling to the ones it judges.
    raw_path: str | None = None
    # A judge cannot approve a root deliberately hidden for privacy. Keep this
    # distinct from ordinary opaque substitutions that can receive review.
    credential_derived: bool = False
    #: ``git worktree remove <path>``: git itself refuses a dirty or primary
    #: checkout unless forced, so the classifier may rank a CLEAN linked
    #: worktree LOW (see ``provenance.clean_linked_worktree``).
    worktree_remove: bool = False
    #: ``--force``/``-f`` on that removal: git deletes uncommitted work, so the
    #: clean-worktree exemption no longer holds and the ordinary ranking applies.
    forced: bool = False


@dataclass(frozen=True)
class Creation:
    """A directory a command may CREATE, for the provenance ledger.

    WHY. The dangerous-delete guard ranks ``~/<name>`` and repository roots
    HIGH, which used to refuse an agent cleaning up the rig directories and
    worktrees it had made itself. The ledger (``guardrails/provenance.py``)
    remembers what a session created so a later delete of exactly that can rank
    LOW. Only literal, statically known paths qualify: a glob or substitution
    could name anything, and a creation record must never be a guess.

    ``path`` is an exact directory (``mkdir``, ``git worktree add``,
    ``git clone``). For ``mktemp -d TEMPLATE`` the name is random, so ``path``
    is ``None`` and ``parent``/``pattern`` describe where to look: the ledger
    snapshots the matches before the call and records only new ones after it.
    """

    kind: str  # "mkdir" | "mktemp" | "worktree" | "clone"
    path: str | None = None
    parent: str | None = None
    pattern: str | None = None


@dataclass
class _State:
    env: dict[str, str]
    cwd: str | None
    home: str
    private_cwd: bool = False


def _absolute(text: str, cwd: str | None) -> tuple[str | None, bool]:
    """``(normalised absolute path, cwd_was_unknown)`` for a word's text."""
    if text.startswith("/"):
        return os.path.normpath("/" + text.lstrip("/")), False
    if cwd is None:
        return None, True
    return os.path.normpath(os.path.join(cwd, text)), False


def _joined(text: str, cwd: str | None) -> str | None:
    """The absolute join of ``text`` WITHOUT normalising: ``..`` survives so
    ``realpath`` can resolve it physically (see ``Target.raw_path``)."""
    if text.startswith("/"):
        return text
    return None if cwd is None else os.path.join(cwd, text)


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
        raw_path = None
    else:
        path, unknown = _absolute(text or ".", state.cwd)
        raw_path = _joined(text or ".", state.cwd)
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
        credential_derived=(
            CREDENTIAL_MARK in exp.text or (state.private_cwd and not exp.text.startswith("/"))
        ),
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
        dotdot=".." in exp.text.split("/"),
        raw_path=None if is_temp else raw_path,
    )


@dataclass
class Resolution:
    targets: list[Target] = field(default_factory=list)
    #: The parser met a construct it cannot follow that MAY delete (``eval`` of a
    #: deleting word, an unbalanced quote around one): reasons, for the judge.
    opaque_reasons: list[str] = field(default_factory=list)
    #: Directories the command may create (see :class:`Creation`). Never used
    #: to rank THIS command; the loop records them after it runs.
    created: list[Creation] = field(default_factory=list)
    #: Paths checked against the guard's own evidence (``<state>/sessions/``,
    #: ``<state>/guardrails/``) by ``classify._touch_reason``. Two shapes, by
    #: ``Target.op``:
    #:
    #: * ``"write"`` — WRITE-SHAPED: every output redirection and the
    #:   destination of a known writer (``ln``, ``rsync``, ``ditto``, ``cp``,
    #:   ``install``, ``tar -x -C``, ``unzip -d``, ``touch``, ``truncate``,
    #:   ``dd of=``, ``perl -i``, ``sqlite3`` without ``-readonly``). With the
    #:   resolver's own destroyed/created targets these rank CRITICAL: they
    #:   are how a forged child session or transcript gets planted (review
    #:   R2-MAJOR-1, QA Q6).
    #: * ``"touch"`` — a path-shaped word handed to a program this resolver
    #:   neither models nor knows to be read-only (``python3 ana.py <path>``).
    #:   What it does with the path is unknowable, so it ranks HIGH and the
    #:   context-free judge (or the operator) can lift it — CRITICAL there
    #:   refused routine transcript analysis with no way through (R3-MAJOR-2).
    #:
    #: Sources (``cp``/``rsync``/``zip``/``tar -c`` inputs) are not touches.
    touched: list[Target] = field(default_factory=list)
    #: Words naming a guard evidence file (``origin.json``, ``transcript.jsonl``
    #: …) handed to a non-read-only program while the working directory is
    #: unresolvable (``cd $(…)``, a glob or CDPATH ``cd``): it may be the one
    #: under ``sessions/``. ``classify`` ranks each HIGH (review R6-MINOR-1).
    blind: list[str] = field(default_factory=list)
    #: The command may make ``**`` recursive (set by ``resolve_command`` from
    #: the raw text; see ``_RECURSIVE_GLOB_RE``). Only then does ``**`` match
    #: any depth (bash's default treats it as ``*``; review round 8, MAJOR-1).
    globstar: bool = False


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
_SQLITE_VALUE_FLAGS = frozenset({"-cmd", "-init", "-separator", "-newline", "-nullvalue"})
_MKDIR_VALUE_FLAGS = frozenset({"-m", "--mode", "--context"})
_MKTEMP_VALUE_FLAGS = frozenset({"-p", "--suffix"})
_GIT_CLONE_VALUE_FLAGS = frozenset(
    {
        "-b",
        "--branch",
        "-o",
        "--origin",
        "--depth",
        "--reference",
        "-c",
        "--config",
        "--template",
        "--separate-git-dir",
        "-u",
        "--upload-pack",
        "--shallow-since",
        "--shallow-exclude",
        "--filter",
        "-j",
        "--jobs",
        "--reference-if-able",
    }
)
_GIT_WORKTREE_ADD_VALUE_FLAGS = frozenset({"-b", "-B", "--reason", "--orphan"})
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
    #: R5 fallback: earlier bodies in the command line not provably sunk.
    earlier: list[_Produced] = field(default_factory=list)


@dataclass
class _Produced:
    """One heredoc/herestring body and what is known about where it went."""

    body: str
    #: Its segment redirects stdout to a file (``cat > notes.md <<EOF``).
    to_file: bool = False
    #: The next pipeline stage's text when the segment pipes onward.
    consumer: str | None = None
    #: The pipeline holding the segment ended before the current one.
    pipeline_ended: bool = False

    def provably_sunk(self) -> bool:
        if self.to_file:
            return True
        if self.consumer is not None and self.pipeline_ended:
            program = _stage_program(self.consumer)
            return program is not None and program not in _SHELL_C_WORDS | {"source", "."}
        return False


#: Words that open/close a compound command whose output a ``|`` after the
#: closer pipes onward, like a ``{ …; }`` group (R5-2).
_COMPOUND_OPENERS = ("if", "for", "while", "until", "case", "select")
_COMPOUND_CLOSERS = ("fi", "done", "esac")
_TRAILING_REDIRECT_RE = re.compile(r"(?:\s+\d*(?:>>|>&|<&|>|<)\s*\S*|\s+&>>?\s*\S*)+\s*$")


_UNSEEN_SCRIPT_REASON = "shell reads a script from stdin or a file (not resolved)"


def _strip_trailing_redirects(text: str) -> str:
    """``} 2>&1`` → ``}`` (R5-1): a redirect after a closer is still a closer."""
    return _TRAILING_REDIRECT_RE.sub("", text.strip())


def _group_opens(text: str) -> int:
    """How many groups/compound commands open at the head of a piece."""
    count = 0
    head = text.strip()
    while True:
        if head.startswith(("{ ", "{\n", "{\t", "(")) and not head.startswith("(("):
            count += 1
            head = head[1:].lstrip()
            continue
        word = head.split(None, 1)[0] if head else ""
        if word in _COMPOUND_OPENERS:
            count += 1
            head = head[len(word) :].lstrip()
            # Only one compound keyword per piece; what follows is its test.
            break
        break
    return count


def _group_closes(text: str) -> int:
    """How many ``}``/``fi``/``done``/``esac`` closers end a piece (R5-1/R5-2)."""
    tail = _strip_trailing_redirects(text)
    count = 0
    while tail:
        if tail == "}" or tail.endswith((" }", ";}")):
            tail = tail[:-1].rstrip().rstrip(";").rstrip()
            count += 1
            continue
        last = tail.rsplit(None, 1)[-1]
        if last in _COMPOUND_CLOSERS:
            tail = tail[: -len(last)].rstrip().rstrip(";").rstrip()
            count += 1
            continue
        break
    return count


#: A STDOUT redirect: fd-less or ``1>``, or ``&>`` (R6-2). ``2>err.log`` only
#: moves stderr, so the body still reaches the next reader. ``scan`` drops fd
#: numbers, hence a dedicated, quote-blind pattern: it can only fail toward
#: "not sunk" (walked) because a match inside quotes merely misses a sink.
_STDOUT_REDIRECT_RE = re.compile(r"(?:(?<![\w&<>])1?|&)(?:>>|>\||>)(?!&)\s*([^\s;|&<>]+)")


def _redirects_stdout_to_file(segment: str) -> bool:
    """The segment sends stdout to a named file or /dev/null (R6-2/R6-3).

    ``/dev/null`` is a sink; other ``/dev/*`` targets (``/dev/stdout``,
    ``/dev/fd/N``) are not, since they hand the body on.
    """
    for target in _STDOUT_REDIRECT_RE.findall(_strip_quoted(segment)):
        if target == "/dev/null" or not target.startswith("/dev/"):
            return True
    return False


def _strip_quoted(text: str) -> str:
    """Blank quoted spans so a ``>`` inside a string is not read as a redirect."""
    return re.sub(r"'[^']*'|\"(?:\\.|[^\"\\])*\"", " ", text)


#: R7-2: an interpreter body that itself shells out (``os.system`` in a
#: ``python3 -`` heredoc) is sunk by design — interpreter-executed shell is out
#: of scope for the heredoc fallback (see the PR's "Not addressed").
#: R6-3: programs that CONSUME a heredoc as data (commit/PR text, a config,
#: an interpreter script that is not a shell). A body fed to one of these and
#: not piped onward is sunk. Deliberately excluded: shells, ``exec``,
#: ``coproc``, ``eval``, ``source``/``.`` and anything unknown such as a shell
#: function — those can hand the body to a shell. Filters that ECHO their
#: input (cat, tee, grep, sed, jq, …) are excluded too: inside a group or
#: compound command whose output is piped on, the echoed body is still a
#: script, so they count as sunk only via a stdout redirect to a file.
_NON_SHELL_CONSUMERS = frozenset(
    {"git", "gh", "glab", "python", "python3", "node", "ruby", "kubectl", "psql", "sqlite3"}
)


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


def _fd_redirect_merge(
    pieces: list[tuple[str, bool, str]], bodies: list[list[str]]
) -> tuple[list[tuple[str, bool, str]], list[list[str]]]:
    """Re-join segments ``search_guard._split`` cut at a redirect's ``&`` (Q1).

    The shared splitter reads the ``&`` in ``2>&1``/``>&2``/``2>&-``/``&>file``
    as a background operator. For this resolver that turned every ``2>&1`` into
    an empty, opaque overwrite target. A segment ending in ``>``/``<`` cut by a
    single ``&`` is glued back to what follows, operator included.
    """
    out: list[tuple[str, bool, str]] = []
    out_bodies: list[list[str]] = []
    for (text, piped, op), owned in zip(pieces, bodies):
        if out:
            prev_text, prev_piped, prev_op = out[-1]
            if prev_op == "&" and re.search(r"[<>]$", prev_text):
                out[-1] = (prev_text + "&" + text, prev_piped, op)
                out_bodies[-1] = out_bodies[-1] + owned
                continue
        out.append((text, piped, op))
        out_bodies.append(list(owned))
    return out, out_bodies


@dataclass
class _Snapshot:
    env: dict[str, str]
    cwd: str | None
    home: str
    private_cwd: bool = False


class _Walker:
    def __init__(self, env: Mapping[str, str], cwd: str | None, home: str) -> None:
        from local_operator.tools.shell_env import is_credential_shaped

        # S1: inherited credentials are unknown paths, never empty expansions.
        # This analysis-only copy must not alter the actual child's grants. An
        # ordinary literal assignment later in the script can still replace it.
        private_env = {
            name: CREDENTIAL_MARK if is_credential_shaped(name) else value
            for name, value in env.items()
        }
        self.state = _State(env=private_env, cwd=cwd, home=home)
        self.result = Resolution()
        #: Paths an earlier segment of THIS command made into a symlink (S7):
        #: what they point at is resolved only at run time.
        self.linked: set[str] = set()
        #: ``for VAR in a b c`` literal lists: a target using $VAR fans out.
        self.for_lists: dict[str, list[str]] = {}

    # -- state isolation (R4) ----------------------------------------------------
    def _save(self) -> _Snapshot:
        st = self.state
        return _Snapshot(dict(st.env), st.cwd, st.home, st.private_cwd)

    def _restore(self, snap: _Snapshot) -> None:
        self.state.env, self.state.cwd, self.state.home = snap.env, snap.cwd, snap.home
        self.state.private_cwd = snap.private_cwd

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
        lexed = lex(command)
        pieces, owned_bodies = _fd_redirect_merge(lexed.segments, lexed.bodies)
        # R7-1/R8, structural: if the text ends inside a construct the ONE
        # shared scanner tracks (quote, ``$(``, ``${``, backtick, heredoc), its
        # segments are a guess. A delete word anywhere then fails toward
        # scrutiny — the next tokenizer gap of this kind, not one spelling.
        if _DELETE_WORD_RE.search(command) and lexed.open:
            self.result.opaque_reasons.append(
                "command ends inside an unterminated quote, substitution or heredoc"
            )
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
        # R5 structural fallback: every body produced so far, with whether it is
        # PROVABLY not a shell's script — redirected into a file by its own
        # segment, or consumed by a non-shell in a pipeline that has ended.
        # Precise attribution above handles the common shapes; this list
        # exists because every review round found a new syntax shape that
        # defeated it, so a pipe-fed shell also walks every earlier body not
        # provably sunk elsewhere (fails toward scrutiny).
        produced_so_far: list[_Produced] = []
        for index, (text, _flag, op) in enumerate(pieces):
            piped = piped_in[index]
            if not piped:
                # A new pipeline starts: earlier pipelines have ENDED, so their
                # bodies' fates are now known.
                for item in produced_so_far:
                    item.pipeline_ended = True
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
            brace_opens = _group_opens(text)
            for _ in range(brace_opens):
                group_bodies.append([])
            nxt_piped = index + 1 < len(pieces) and piped_in[index + 1]
            isolated = piped or nxt_piped or op == "&"
            heredocs = owned_bodies[index]
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
                earlier=[item for item in produced_so_far if not item.provably_sunk()],
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
            to_file = _redirects_stdout_to_file(text) or (
                not nxt_piped and _stage_program(text) in _NON_SHELL_CONSUMERS
            )
            consumer = pieces[index + 1][0] if nxt_piped else None
            for body in produced:
                produced_so_far.append(_Produced(body=body, to_file=to_file, consumer=consumer))
            group_closes = closes + _group_closes(text)
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
            if op in (">", ">|", ">>", ">&", "<>"):
                for exp in expand_all(raw, st.env, st.home):
                    if exp.text.startswith("/dev/"):
                        continue
                    self.result.touched.append(_target(exp, "write", op, st))
                    if op in (">", ">|", ">>"):
                        self._add(exp, "append" if op == ">>" else "overwrite", op)
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
            if any(a.startswith("CDPATH=") for a in raw_assigns):
                # ``CDPATH=… cd rel`` resolves ``rel`` against CDPATH, which
                # this resolver does not model (review R6-MINOR-2).
                st.env["CDPATH"] = OPAQUE_MARK
            self._cd(words[1:])
            return
        if name == "popd":
            st.cwd = None
            st.env.pop("OLDPWD", None)
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
        # Kept for the backstop: a wrapper's own flags can carry a path
        # (``env -Q<path>``, ``xargs -a <path>``) that peeling would hide.
        unpeeled = words
        words, stdin_fed = _peel_wrappers(words)
        if not words:
            self._backstop(os.path.basename(unpeeled[0].text), unpeeled, raw_assigns, set())
            return
        name = os.path.basename(words[0].text)
        args = words[1:]
        # An inline assignment of a pager/git variable can run a program even
        # under a read-only subcommand (``GIT_PAGER=sh git log``).
        assigned = [a.split("=", 1)[0] for a in raw_assigns]
        if not read_only_argv(name, [w.text for w in args], assigned=assigned):
            exempt = self._touch(name, args)
            # The program word too: after ``env``/``xargs`` peeling a path can
            # BE the program (``xargs -Q <path>``).
            self._backstop(name, unpeeled, raw_assigns, exempt)
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
                # R9-1: an opaque word's text omits its substitution; the
                # substitution's own text (a heredoc'd script) is what eval runs.
                haystack = " ".join([inner, *(b for w in args for b in w.substitutions)])
                if _DELETE_WORD_RE.search(haystack):
                    self.result.opaque_reasons.append("eval of an unresolvable command")
                    self._private_script_target(inner)
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
                self._add_overwrite_dest(operands[-1], name, operands[:-1])
            return
        if name == "rsync":
            self._rsync(args)
            return
        if name in ("mkdir", "gmkdir"):
            self._mkdir(args)
            return
        if name in ("mktemp", "gmktemp"):
            self._mktemp(args)
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
            appending = _has_short(flags, "a") or "--append" in flags
            for w in operands:
                self._add(w, "append" if appending else "overwrite", name)
            return
        if name == "sqlite3":
            # ``sqlite3 FILE [SQL]`` writes FILE whatever the SQL says; the
            # guard's ledger is a sqlite file (R1-MAJOR-1).
            operands, _ = _operands(args, _SQLITE_VALUE_FLAGS)
            # ``-readonly`` opens the file read-only (R3-MAJOR-2 false positive).
            if operands and not any(w.text == "-readonly" for w in args):
                self._add(operands[0], "append", name)
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
                    self._private_script_target(script.text)
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
        """A shell whose script is a pipe or a file (R3-1, R3-4, R5)."""
        if stdin_piped and feed is not None:
            walked = list(feed.bodies)
            # R5: plus every earlier delete-bearing body not provably sunk
            # elsewhere — the class of shapes precise attribution misses.
            walked += [
                item.body
                for item in feed.earlier
                if item.body not in walked and _DELETE_WORD_RE.search(item.body)
            ]
            if walked:
                self._walk_stdin_bodies(walked, [], depth)
                # R6-1: walking an earlier body says nothing about the stream
                # this shell actually reads, so a fetch upstream still counts.
                self._fetched_script(stdin_piped, feed)
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
        if _DELETE_WORD_RE.search(haystack):
            self.result.opaque_reasons.append(_UNSEEN_SCRIPT_REASON)
        else:
            self._fetched_script(stdin_piped, feed)

    def _fetched_script(self, stdin_piped: bool, feed: _Feed | None) -> None:
        """R3-4: a fetch is a PROGRAM upstream (``curl …``, ``timeout 30 wget``),
        not a word in an argument (``cat ncat.log``, ``echo http://x``)."""
        if (
            stdin_piped
            and feed is not None
            and any(
                _NETWORK_FETCH_RE.fullmatch(_stage_program(stage) or "") for stage in feed.stages
            )
            and _UNSEEN_SCRIPT_REASON not in self.result.opaque_reasons
        ):
            self.result.opaque_reasons.append(_UNSEEN_SCRIPT_REASON)

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
        previous = st.cwd
        # A privacy-hidden cd leaves relative deletion roots uninspectable too.
        # This is cwd provenance, not a whole-command taint: a later successful
        # literal cd clears it, and the existing shell snapshots isolate it.
        private_target = st.private_cwd or CREDENTIAL_MARK in (
            operands[0].text if operands else st.home
        )
        if not operands or (operands[0].vanished and not operands[0].opaque):
            path = st.home
        else:
            w = operands[0]
            if w.pattern is not None:
                # A glob cd lands somewhere the resolver cannot name; if the
                # pattern can reach session state, that is a mention (R6-MAJOR-2).
                self.result.touched.append(
                    Target(path=w.text, op="touch", program="cd", pattern=w.pattern)
                )
                path = None
            elif w.opaque:
                path = None
            elif st.env.get("CDPATH") and not w.text.startswith(("/", "~", "./", "../")):
                path = None  # CDPATH search: unresolvable (R6-MINOR-2)
            elif w.text == "-":
                # Bash updates OLDPWD after each cd, including cd -. Track it
                # in the existing env snapshot so subshells/pipelines cannot
                # leak it. An untracked/relative value still fails closed.
                oldpwd = st.env.get("OLDPWD", "")
                path = os.path.normpath(oldpwd) if oldpwd.startswith("/") else None
            else:
                path, _unknown = _absolute(w.text, st.cwd)
        try:
            searchable = (
                path is not None
                and os.path.isdir(path)
                and os.access(path, os.X_OK, effective_ids=os.access in os.supports_effective_ids)
            )
        except OSError:
            searchable = False
        # Existence alone is not success: a mode-000 directory may pass isdir
        # while the shell cannot search it. Prefer effective IDs where supported.
        if not searchable:
            # A failed cd preserves the shell's old cwd/OLDPWD. We cannot
            # assume success (or recover it after an unmodelled popd), so
            # invalidate BOTH instead of letting cd - revive a stale path.
            st.cwd = None
            st.private_cwd = private_target
            st.env.pop("OLDPWD", None)
            return
        st.private_cwd = False
        if previous is None:
            st.env.pop("OLDPWD", None)
        else:
            st.env["OLDPWD"] = previous
        st.cwd = path

    def _private_script_target(self, text: str) -> None:
        # A consuming shell/eval has an unresolved deleting script, not ordinary
        # argument data. Retain private-root provenance across that uncertainty
        # path so a judge cannot consent to roots it was deliberately not shown.
        if CREDENTIAL_MARK in text:
            self.result.targets.append(
                Target(None, "delete", "shell", opaque=True, credential_derived=True)
            )

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
            st.env[var] = (
                CREDENTIAL_MARK if any(CREDENTIAL_MARK in v.text for v in values) else OPAQUE_MARK
            )
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
                self._add_overwrite_dest(operands[-1], "mv", sources)
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
        if args and args[0].text in ("worktree", "clone"):
            self._git_tree(args, cwd_override)
            return
        if not args or args[0].text != "clean":
            return
        _operands_, flags = _operands(args[1:])
        if _has_short(flags, "n") or "--dry-run" in flags:
            return
        if not (_has_short(flags, "xXd")):
            return
        root = cwd_override or Expanded(text=".")
        self._add(root, "clean", "git clean")

    def _touch(self, name: str, args: list[Expanded]) -> set[int]:
        """Record a known writer's destinations as ``write`` touches.

        Returns the ids of words that are POSITIVELY identified sources of a
        fully modelled copy (``cp``/``rsync``/``zip``/``tar -c`` with its
        destination also identified): the only non-read-only words the
        backstop below may skip.
        """
        st = self.state
        surface = _write_surface(name, args, st.home)
        if name in ("ln", "gln"):
            self._link_targets(args)
        if surface is None:
            return set()
        dests, fully = surface
        for w in dests:
            if w.text and not w.vanished:
                self.result.touched.append(_target(w, "write", name, st))
        if (
            fully
            and dests
            and name in _COPY_FAMILY
            and (name not in ("tar", "gtar", "bsdtar") or _tar_mode(args) == "create")
        ):
            dest_ids = {id(w) for w in dests}
            return {id(w) for w in args if id(w) not in dest_ids}
        return set()

    def _link_targets(self, args: list[Expanded]) -> None:
        """A link's TARGET, resolved where the link will live, is a touch.

        WHY (review round 7). A glob is expanded at check time to catch a
        symlink into session state — but a link made in the SAME command
        does not exist yet. ``ln -s ../../<child> $P/m && sed -i … $P/m*/…``
        would then pass, so a link pointing into ``sessions/``/``guardrails/``
        ranks HIGH on its own.
        """
        st = self.state
        operands = _plain_operands(args, _LN_FLAGS)
        dest = _cluster_value(args, "t", "--target-directory")
        if dest:
            sources, link = operands, dest[0].text
        elif len(operands) >= 2:
            sources, link = operands[:-1], operands[-1].text
        else:
            sources, link = operands, "."
        link_abs, _ = _absolute(link, st.cwd)
        if link_abs is None:
            return
        base = link_abs if (dest or os.path.isdir(link_abs)) else os.path.dirname(link_abs)
        for w in sources:
            if not w.text or w.opaque:
                continue
            raw = w.text if w.text.startswith("/") else os.path.join(base, w.text)
            self.result.touched.append(Target(path=os.path.normpath(raw), op="touch", program="ln"))

    def _backstop(
        self, name: str, args: list[Expanded], assigns: list[str], exempt: set[int]
    ) -> None:
        """THE INVARIANT (review R5): a non-read-only call that MENTIONS
        protected state ranks at least HIGH.

        WHY A BACKSTOP. The writer table above is best-effort: five review
        waves each found a parser gap (a cluster, a dashless key, a glued
        value, a script-embedded ``w``) that turned a write into LOW. This
        pass does not parse programs at all. It marks, as a ``touch`` (HIGH
        in ``classify``), every path-looking SUBSTRING of every argument and
        inline assignment — so a path inside a quoted script (``1w <p>``,
        ``--pager=tee <p>``) or a glued value still counts — plus the
        effective cwd. ``classify._touch_reason`` keeps only those under
        ``<state>/sessions/`` (outside this session's scratchpad) or
        ``<state>/guardrails/``. Exempt: a strictly read-only program (never
        reaches here) and the identified sources of a fully modelled copy.
        A misparse can therefore cost a judge call, never a LOW.
        """
        st = self.state
        words = [w for w in args if id(w) not in exempt]
        texts = [w.text for w in words] + [a.split("=", 1)[1] for a in assigns if "=" in a]
        for text in texts:
            for raw in _MENTION_RE.findall(text):
                if OPAQUE_MARK in raw or TEMP_MARK in raw:
                    continue  # an opaque part: the word itself is added below
                path = _mention_path(raw, st.home, st.cwd)
                if path is None:
                    continue
                if _GLOB_CHARS.search(path):
                    # ``sess*``, ``s[e]ssions``, ``se?sions``: compared as a
                    # PATTERN in classify, never as a literal prefix (R6-MAJOR-2).
                    self.result.touched.append(
                        Target(
                            path=_literal_prefix(path),
                            op="touch",
                            program=name or "a program",
                            pattern=path,
                        )
                    )
                else:
                    self.result.touched.append(
                        Target(path=path, op="touch", program=name or "a program")
                    )
            if st.cwd is None:
                for part in re.split(r"[\s/'\"=]+", text):
                    if part and _names_evidence(part):
                        self.result.blind.append(part)
        for w in words:
            if w.opaque:
                self.result.touched.append(_target(w, "touch", name, st))
        if st.cwd and not st.private_cwd:
            self.result.touched.append(Target(path=st.cwd, op="touch", program=name))

    # -- creation (provenance ledger) ------------------------------------------
    def _literal_dir(self, exp: Expanded, base: str | None = None) -> str | None:
        """The absolute path a LITERAL word names, or ``None`` if not literal."""
        if exp.opaque or exp.pattern is not None or exp.empty or not exp.text:
            return None
        if self.state.private_cwd and not exp.text.startswith("/"):
            return None
        path, _unknown = _absolute(exp.text, base if base is not None else self.state.cwd)
        return path

    def _create(self, kind: str, path: str | None) -> None:
        if path is not None:
            self.result.created.append(Creation(kind=kind, path=path))

    def _mkdir(self, args: list[Expanded]) -> None:
        operands, _flags = _operands(args, _MKDIR_VALUE_FLAGS)
        for w in operands:
            self._create("mkdir", self._literal_dir(w))

    def _mktemp(self, args: list[Expanded]) -> None:
        operands, flags = _operands(args, _MKTEMP_VALUE_FLAGS)
        if not (_has_short(flags, "d") or "--directory" in flags):
            return
        if _has_short(flags, "tu") or "--dry-run" in flags or len(operands) != 1:
            # No template (a random name under $TMPDIR, already LOW) or a form
            # whose placement this parser does not model: record nothing.
            return
        template = operands[0]
        base = self.state.cwd
        for i, w in enumerate(args):
            t = w.text
            if t == "-p" and i + 1 < len(args):
                base = self._literal_dir(args[i + 1]) or None
            elif t.startswith("--tmpdir="):
                base = self._literal_dir(Expanded(text=t.split("=", 1)[1])) or None
        path = self._literal_dir(template, base)
        if path is None or "XXX" not in os.path.basename(path):
            return
        name = os.path.basename(path)
        stem = name.rstrip("X")
        pattern = _glob_escape(stem) + "?" * (len(name) - len(stem))
        self.result.created.append(
            Creation(kind="mktemp", parent=os.path.dirname(path), pattern=pattern)
        )

    def _git_tree(self, args: list[Expanded], cwd_override: Expanded | None) -> None:
        base = self.state.cwd
        if cwd_override is not None:
            base = self._literal_dir(cwd_override)
            if base is None:
                return
        if args[0].text == "clone":
            operands, _ = _operands(args[1:], _GIT_CLONE_VALUE_FLAGS)
            if not operands:
                return
            if len(operands) >= 2:
                self._create("clone", self._literal_dir(operands[1], base))
                return
            url = operands[0]
            if url.opaque or url.pattern is not None:
                return
            # git's own default: the URL's last component minus ``.git``.
            name = re.split(r"[/:]", url.text.rstrip("/"))[-1]
            name = name[:-4] if name.endswith(".git") else name
            if name and name not in (".", ".."):
                self._create("clone", self._literal_dir(Expanded(text=name), base))
            return
        if len(args) < 2:
            return
        sub = args[1].text
        if sub == "add":
            operands, _ = _operands(args[2:], _GIT_WORKTREE_ADD_VALUE_FLAGS)
            if operands:
                self._create("worktree", self._literal_dir(operands[0], base))
        elif sub == "remove":
            operands, flags = _operands(args[2:])
            forced = _has_short(flags, "f") or "--force" in flags
            for w in operands:
                saved = self.state.cwd
                self.state.cwd = base
                try:
                    before = len(self.result.targets)
                    self._add(w, "delete", "git worktree remove")
                    for target in self.result.targets[before:]:
                        target.worktree_remove = True
                        target.forced = forced
                finally:
                    self.state.cwd = saved

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

    def _add_overwrite_dest(
        self, exp: Expanded, program: str, sources: list[Expanded] | None = None
    ) -> None:
        if exp.opaque or exp.pattern is not None:
            return
        path, _ = _absolute(exp.text, self.state.cwd)
        if path is not None and os.path.isdir(path):
            # Moving/copying INTO a directory replaces nothing named here, but
            # it CREATES ``dir/<source name>``: recorded as an append so a
            # fake session directory moved into ``<state>/sessions/`` is ranked
            # (``classify._evidence_reason``) and nothing else changes.
            for src in sources or ():
                if src.opaque or src.pattern is not None or not src.text:
                    continue
                name = os.path.basename(src.text.rstrip("/"))
                if name:
                    self._add(Expanded(text=os.path.join(path, name)), "append", program)
            return
        self._add(exp, "overwrite", program)

    def _add_stdin(self, program: str) -> None:
        self.result.targets.append(
            Target(
                path=self.state.cwd,
                op="delete",
                program=f"xargs {program}",
                credential_derived=self.state.private_cwd,
                opaque=True,
                unknown_cwd=self.state.cwd is None,
            )
        )


#: Programs that only READ the paths they are given (review R2-MAJOR-1): the
#: one place the evidence check trusts a path operand under the state root.
#: Interpreters are never here — what ``python x.py <path>`` does with a path
#: is unknowable. ``echo``/``printf`` write only through a redirection, which
#: is checked separately for every program.
#:
#: STRICT (review R6): only programs that cannot write a file or run another
#: program by ANY flag. ``rg --pre``, ``less -o``, ``file -C``, ``sqlite3
#: .once/.shell`` and ``sed``'s own ``w`` each did, so they are not here: a
#: call to them that mentions session state goes through the backstop (HIGH,
#: liftable by the judge) instead of being trusted.
READ_ONLY_PROGRAMS = frozenset(
    {
        "ls",
        "cat",
        "head",
        "tail",
        "grep",
        "egrep",
        "fgrep",
        "stat",
        "du",
        "df",
        "wc",
        "jq",
        "echo",
        "printf",
        "true",
        "test",
        "[",
        "pwd",
        "which",
        "readlink",
        "realpath",
        "basename",
        "dirname",
        "sleep",
        # Shell words that move, bind or branch without writing a file; their
        # bodies and later segments are walked on their own.
        "cd",
        "pushd",
        "popd",
        "export",
        "declare",
        "local",
        "readonly",
        "typeset",
        "unset",
        "set",
        "for",
        "if",
        "while",
        "until",
        "case",
        "do",
        "then",
        "else",
        "elif",
        "fi",
        "done",
        "esac",
        "!",
    }
)
_READ_ONLY_GIT = frozenset({"log", "status", "show", "diff", "rev-parse", "ls-files"})
#: Git words that make even a read-only subcommand write a file or run a
#: program (review R3-MAJOR-1: ``git log --output=<file>`` wrote the lineage
#: registry at LOW). ``-c``/``--config-env`` can set ``core.pager`` and
#: friends; ``-p``/``--paginate``/``--pager`` start the pager.
_GIT_UNSAFE = (
    "--output",
    "-o",
    "-c",
    "--config-env",
    "--ext-diff",
    "--textconv",
    "--exec-path",
    "--git-dir",
    "--work-tree",
    "--pager",
    "--paginate",
    "-p",
    "--no-index",
)
#: Variables that make git (or its pager) run something else.
_GIT_UNSAFE_ENV_RE = re.compile(r"^(GIT_|PAGER$|LESS|EDITOR$|VISUAL$)")
_FIND_WRITES = frozenset(
    {"-delete", "-exec", "-execdir", "-ok", "-okdir", "-fprint", "-fprint0", "-fprintf", "-fls"}
)


def read_only_argv(name: str, argv: list[str], *, assigned: Sequence[str] = ()) -> bool:
    """Is ``name argv…`` a provably read-only invocation? Conservative: unknown → no."""
    if name in READ_ONLY_PROGRAMS:
        return True
    if name == "find":
        return not any(a in _FIND_WRITES for a in argv)
    if name == "git":
        if any(_GIT_UNSAFE_ENV_RE.match(a) for a in assigned):
            return False
        rest = list(argv)
        while rest and rest[0].startswith("-"):
            if rest[0] != "-C" and rest[0].split("=", 1)[0] in _GIT_UNSAFE:
                return False
            rest = rest[2:] if rest[0] == "-C" else rest[1:]
        if not rest or rest[0] not in _READ_ONLY_GIT:
            return False
        sub, tail = rest[0], rest[1:]
        for a in tail:
            if a == "--":
                break
            key = a.split("=", 1)[0]
            if key in _GIT_UNSAFE and not (key == "-p" and sub in ("log", "show", "diff")):
                return False
            # ``-o<file>`` / ``--output<…>`` glued forms.
            if a.startswith("--output") or (a.startswith("-o") and sub in ("log", "show", "diff")):
                return False
        return True
    return False


# -- the evidence writer table (review R4-MAJOR-1) ---------------------------
#
# WHY A TABLE THAT FAILS CLOSED. ``_touch`` used to stop at any program it had
# heard of, so an UNMODELLED write by a known program (``git -C <s> checkout``,
# ``tar -cf <s>/origin.json``, ``perl -e 'open…' <s>``) ranked LOW with no
# judge call. Now a program skips the HIGH fallback only when its WHOLE write
# surface for the given mode is enumerated below AND every flag it was given
# is one this table knows. Anything else — an unknown flag, an unlisted git
# subcommand, a perl without ``-i`` — keeps its known destinations as CRITICAL
# writes and ALSO sends every path-shaped word to the HIGH fallback.


_LN_VALUE_FLAGS = frozenset({"-t", "--target-directory", "-S", "--suffix"})
_TAR_VALUE_FLAGS = frozenset({"-f", "--file", "-C", "--directory", "-T", "--files-from", "-X"})


@dataclass(frozen=True)
class _Flags:
    """The flags a fully-modelled program may carry: short letters (clusters
    allowed), long names, and the flags whose value is the NEXT word."""

    letters: str = ""
    longs: frozenset[str] = frozenset()
    values: frozenset[str] = frozenset()


def _flags_known(args: list[Expanded], spec: _Flags) -> bool:
    """Is every flag in ``args`` one ``spec`` lists? ``--`` ends the flags."""
    skip = False
    for w in args:
        t = w.text
        if skip:
            skip = False
            continue
        if t == "--":
            return True
        if not t.startswith("-") or t == "-":
            continue
        if t.startswith("--"):
            key = t.split("=", 1)[0]
            if key not in spec.longs and key not in spec.values:
                return False
            skip = key in spec.values and "=" not in t
            continue
        if t in spec.values:
            skip = True
            continue
        cluster = t[1:]
        for k, c in enumerate(cluster):
            if f"-{c}" in spec.values:
                skip = k == len(cluster) - 1  # ``-f FILE`` vs glued ``-fFILE``
                break
            if c not in spec.letters:
                return False
    return True


_CP_FLAGS = _Flags(
    "aRrfinpvLPHlsxTXcEN",
    frozenset(
        {
            "--recursive",
            "--archive",
            "--force",
            "--interactive",
            "--no-clobber",
            "--verbose",
            "--preserve",
            "--no-preserve",
            "--parents",
            "--dereference",
            "--no-dereference",
            "--link",
            "--symbolic-link",
            "--update",
            "--reflink",
            "--sparse",
            "--one-file-system",
            "--no-target-directory",
            "--remove-destination",
            "--attributes-only",
            "--strip-trailing-slashes",
            "--backup",
        }
    ),
    frozenset({"-t", "--target-directory", "-S", "--suffix"}),
)
_LN_FLAGS = _Flags(
    "sfnvhiFTLPrb",
    frozenset(
        {
            "--symbolic",
            "--force",
            "--no-dereference",
            "--verbose",
            "--interactive",
            "--relative",
            "--logical",
            "--physical",
            "--no-target-directory",
            "--backup",
        }
    ),
    _LN_VALUE_FLAGS,
)
_RSYNC_FLAGS = _Flags(
    "avzrlptgoDhPnucqHAXWSxRbmiOJ8",
    frozenset(
        {
            "--archive",
            "--verbose",
            "--compress",
            "--recursive",
            "--links",
            "--perms",
            "--times",
            "--group",
            "--owner",
            "--devices",
            "--specials",
            "--human-readable",
            "--progress",
            "--partial",
            "--dry-run",
            "--update",
            "--checksum",
            "--quiet",
            "--hard-links",
            "--acls",
            "--xattrs",
            "--whole-file",
            "--sparse",
            "--one-file-system",
            "--relative",
            "--stats",
            "--info",
            "--itemize-changes",
            "--delete",
            "--delete-before",
            "--delete-during",
            "--delete-after",
            "--delete-delay",
            "--delete-excluded",
            "--exclude",
            "--include",
            "--filter",
            "--ignore-existing",
            "--size-only",
            "--no-perms",
            "--no-owner",
            "--no-group",
            "--omit-dir-times",
            "--prune-empty-dirs",
            "--mkpath",
            "--times",
            "--safe-links",
            "--copy-links",
            "--copy-unsafe-links",
        }
    ),
    _RSYNC_VALUE_FLAGS,
)
_DITTO_FLAGS = _Flags(
    "VXckx",
    frozenset(
        {
            "--keepParent",
            "--rsrc",
            "--norsrc",
            "--extattr",
            "--noextattr",
            "--acl",
            "--noacl",
            "--qtn",
            "--noqtn",
            "--sequesterRsrc",
            "--nocache",
            "--hfsCompression",
            "--nohfsCompression",
            "--zlibCompressionLevel",
        }
    ),
)
_INSTALL_FLAGS = _Flags(
    "dcpsvDCbS",
    frozenset({"--directory", "--verbose", "--compare", "--preserve-timestamps"}),
    frozenset({"-m", "--mode", "-o", "--owner", "-g", "--group", "-t", "--target-directory"}),
)
_TOUCH_FLAGS = _Flags(
    "amch",
    frozenset({"--no-create", "--no-dereference"}),
    frozenset({"-r", "--reference", "-d", "--date", "-t", "-A"}),
)
_TRUNCATE_FLAGS = _Flags(
    "co",
    frozenset({"--no-create", "--io-blocks"}),
    frozenset({"-s", "--size", "-r", "--reference"}),
)
_UNZIP_FLAGS = _Flags(
    "qonjluvfaLCDKXV",
    frozenset(),
    frozenset({"-d", "-x", "-P"}),
)
_ZIP_FLAGS = _Flags(
    "rqyj0123456789vumDTXgFS@",
    frozenset({"--recurse-paths", "--quiet", "--symlinks", "--junk-paths", "--move"}),
    frozenset({"-x", "-i", "-n", "-P", "-O", "--out"}),
)
_TAR_FLAGS = _Flags(
    "xctruvfzjJZapPkmhOSlw",
    frozenset(
        {
            "--extract",
            "--get",
            "--create",
            "--list",
            "--append",
            "--update",
            "--verbose",
            "--gzip",
            "--bzip2",
            "--xz",
            "--zstd",
            "--preserve-permissions",
            "--keep-old-files",
            "--strip-components",
            "--exclude",
            "--no-same-owner",
            "--same-owner",
            "--overwrite",
            "--to-stdout",
            "--numeric-owner",
            "--absolute-names",
        }
    ),
    _TAR_VALUE_FLAGS,
)
_MKDIR_FLAGS = _Flags("pv", frozenset({"--parents", "--verbose"}), _MKDIR_VALUE_FLAGS)
_META_FLAGS = _Flags(
    "RfvhHLPxsdcrlpw",
    frozenset(
        {
            "--recursive",
            "--verbose",
            "--silent",
            "--quiet",
            "--no-dereference",
            "--dereference",
            "--changes",
            "--reference",
        }
    ),
)
_MV_FLAGS = _Flags(
    "finvuhFTbc",
    frozenset(
        {
            "--force",
            "--interactive",
            "--no-clobber",
            "--verbose",
            "--update",
            "--no-target-directory",
            "--strip-trailing-slashes",
            "--backup",
        }
    ),
    _MV_VALUE_FLAGS,
)
_TEE_FLAGS = _Flags("aip", frozenset({"--append", "--ignore-interrupts"}))
_SED_FLAGS = _Flags(
    "nEreisuz",
    frozenset(
        {
            "--quiet",
            "--silent",
            "--regexp-extended",
            "--in-place",
            "--separate",
            "--unbuffered",
            "--null-data",
            "--posix",
        }
    ),
    frozenset({"-e", "--expression", "-f", "--file", "-l"}),
)
#: A sed script that writes a file of its own (``w FILE``, ``s/a/b/w FILE``).
_SED_WRITES_RE = re.compile(r"[wWe]")
#: A path-looking filename after a ``w``/``W`` (``1w /x``, ``s/a/b/gw ~/x``).
_SED_W_PATH_RE = re.compile(r"[wW]\s*((?:/|~|\$\{?HOME\}?)[^\s;}'\"]*)")
_DD_KEYS = frozenset(
    {
        "if",
        "of",
        "bs",
        "ibs",
        "obs",
        "count",
        "skip",
        "seek",
        "conv",
        "status",
        "iflag",
        "oflag",
        "cbs",
        "iseek",
        "oseek",
    }
)


def _flag_value(args: list[Expanded], short: str, long: str) -> list[Expanded]:
    """Values of ``-X v``, ``-Xv``, ``--long v`` and ``--long=v`` in ``args``."""
    out: list[Expanded] = []
    for k, w in enumerate(args):
        t = w.text
        if (t == long or (short and t == short)) and k + 1 < len(args):
            out.append(args[k + 1])
        elif t.startswith(long + "="):
            out.append(Expanded(text=t.split("=", 1)[1], empty=w.empty))
        elif short and t.startswith(short) and len(t) > len(short) and not t.startswith("--"):
            out.append(Expanded(text=t[len(short) :], empty=w.empty))
    return out


def _plain_operands(args: list[Expanded], spec: _Flags) -> list[Expanded]:
    operands, _ = _operands(args, spec.values)
    return operands


def _copy_dest(args: list[Expanded], spec: _Flags) -> list[Expanded]:
    # Cluster-aware: GNU ``-at<dir>`` / ``-st<dir>`` glue the value after
    # other letters (review R5-MINOR-3). ``-t`` only means a target directory
    # where the program's own table says so.
    target_dir = _cluster_value(args, "t", "--target-directory") if "-t" in spec.values else []
    if target_dir:
        return target_dir
    operands = _plain_operands(args, spec)
    return operands[-1:] if len(operands) >= 2 else []


def _write_surface(
    name: str, args: list[Expanded], home: str
) -> tuple[list[Expanded], bool] | None:
    """``(destinations, fully_modelled)`` for a known writer, else ``None``.

    ``destinations`` are the paths the call WRITES (sources and inputs are
    reads). ``fully_modelled`` is True only when that list is the program's
    whole write surface for this mode and every flag given is known; a False
    sends the call's path-shaped words to the HIGH fallback as well.
    """
    if name in DELETE_PROGRAMS:
        return [], True  # every operand is the resolver's own delete target
    if name in ("mv", "gmv"):
        # Sources and a literal destination are resolver targets; a GLOB
        # destination is not (``_add_overwrite_dest`` skips patterns), so name
        # it here too (review R6-MAJOR-2: ``mv f <state>/s[e]ssions/…``).
        return _copy_dest(args, _MV_FLAGS), _flags_known(args, _MV_FLAGS)
    if name in ("tee",):
        return [], _flags_known(args, _TEE_FLAGS)  # resolver targets
    if name in ("mkdir", "gmkdir"):
        return [], _flags_known(args, _MKDIR_FLAGS)  # resolver creations
    if name in ("chmod", "chown", "chgrp", "xattr"):
        return [], _flags_known(args, _META_FLAGS)  # metadata only, no content
    if name in ("sed", "gsed"):
        # ``-i`` operands are resolver targets; a script's own ``w`` is not.
        scripts = " ".join(w.text for w in args if not w.text.startswith("-"))
        # Its own ``w``/``W`` commands write: a parseable filename there is a
        # destination (review R5-MINOR-1); a script that is not a provably
        # read-only one is never fully modelled, so the backstop sees it.
        dests = [
            Expanded(text=_tilde_value(m, home))
            for m in _SED_W_PATH_RE.findall(" ".join(w.text for w in args))
        ]
        return dests, _flags_known(args, _SED_FLAGS) and not _SED_WRITES_RE.search(scripts)
    if name in ("cp", "gcp"):
        return _copy_dest(args, _CP_FLAGS), _flags_known(args, _CP_FLAGS)
    if name in ("ln", "gln"):
        operands = _plain_operands(args, _LN_FLAGS)
        dest = _cluster_value(args, "t", "--target-directory")
        if not dest and len(operands) == 1:
            # ``ln -s TARGET`` links into the cwd under the target's basename.
            dest = [Expanded(text=os.path.basename(operands[0].text.rstrip("/")) or ".")]
        elif not dest:
            dest = operands[-1:] if len(operands) >= 2 else []
        return dest, _flags_known(args, _LN_FLAGS)
    if name == "rsync":
        return _copy_dest(args, _RSYNC_FLAGS), _flags_known(args, _RSYNC_FLAGS)
    if name == "ditto":
        return _copy_dest(args, _DITTO_FLAGS), _flags_known(args, _DITTO_FLAGS)
    if name in ("install", "ginstall"):
        # ``-d`` may sit inside a cluster (``-dv``, ``-dm755``): a regression
        # in round 4 read only the bare spelling (review R5-MAJOR-1).
        if any(w.text == "--directory" or _short_has(w.text, "d", _INSTALL_FLAGS) for w in args):
            return _plain_operands(args, _INSTALL_FLAGS), _flags_known(args, _INSTALL_FLAGS)
        return _copy_dest(args, _INSTALL_FLAGS), _flags_known(args, _INSTALL_FLAGS)
    if name in ("touch", "gtouch"):
        return _plain_operands(args, _TOUCH_FLAGS), _flags_known(args, _TOUCH_FLAGS)
    if name in ("truncate", "gtruncate"):
        return _plain_operands(args, _TRUNCATE_FLAGS), _flags_known(args, _TRUNCATE_FLAGS)
    if name == "dd":
        dests = [
            Expanded(text=_tilde_value(w.text[3:], home), empty=w.empty)
            for w in args
            if w.text.startswith("of=")
        ]
        known = all("=" in w.text and w.text.split("=", 1)[0] in _DD_KEYS for w in args)
        return dests, known
    if name == "unzip":
        dest = _flag_value(args, "-d", "-d") or [Expanded(text=".")]
        return dest, _flags_known(args, _UNZIP_FLAGS)
    if name == "zip":
        out = _flag_value(args, "-O", "--out")
        operands = _plain_operands(args, _ZIP_FLAGS)
        return out + operands[:1], _flags_known(args, _ZIP_FLAGS)
    if name in ("tar", "gtar", "bsdtar"):
        if args and args[0].text.isalpha():
            # Old-style key (``tar cf OUT src``): the first word IS a flag
            # cluster, so read it as one (review R5-MAJOR-2).
            args = [Expanded(text="-" + args[0].text, empty=args[0].empty), *args[1:]]
        mode = _tar_mode(args)
        known = _flags_known(args, _TAR_FLAGS)
        if mode == "extract":
            return _cluster_value(args, "C", "--directory") or [Expanded(text=".")], known
        if mode == "create":
            # The ARCHIVE is written; ``-C`` names where inputs are read.
            return _cluster_value(args, "f", "--file"), known
        return [], known
    if name == "perl":
        # Never fully modelled: like every interpreter, what the script does
        # with a path is unknowable, so its paths also go to the fallback.
        return _perl_inplace(args), False
    if name == "sqlite3":
        # ``.output FILE``, ``.once``, ``.shell`` write or run elsewhere even
        # under ``-readonly``: never fully modelled, so a mention of session
        # state is HIGH through the backstop (review R6, pre-existing escapes).
        operands, _ = _operands(args, _SQLITE_VALUE_FLAGS)
        readonly = any(w.text == "-readonly" for w in args)
        return ([] if readonly else operands[:1]), False
    if name == "git":
        return _git_writes(args), False
    return None


def _cluster_value(args: list[Expanded], letter: str, long: str) -> list[Expanded]:
    """Like ``_flag_value``, but finds ``letter`` inside a short cluster too:
    ``tar -czf OUT`` and ``tar -czfOUT`` both name ``OUT`` (review R4-MAJOR-1:
    ``tar -cf <s>/origin.json`` read the cluster as a mode and missed it)."""
    out = _flag_value(args, "", long)
    for k, w in enumerate(args):
        t = w.text
        if not t.startswith("-") or t.startswith("--") or letter not in t[1:]:
            continue
        rest = t[1:].split(letter, 1)[1]
        if rest:
            out.append(Expanded(text=rest, empty=w.empty))
        elif k + 1 < len(args):
            out.append(args[k + 1])
    return out


def _short_has(text: str, letter: str, spec: _Flags) -> bool:
    """Is ``-<letter>`` inside the short cluster ``text`` (before any value)?"""
    if not text.startswith("-") or text.startswith("--"):
        return False
    for c in text[1:]:
        if c == letter:
            return True
        if f"-{c}" in spec.values:
            return False  # the rest of the cluster is that flag's value
    return False


def _perl_inplace(args: list[Expanded]) -> list[Expanded]:
    if not any(_PERL_INPLACE_RE.match(w.text) for w in args):
        return []
    out = []
    script_seen = False
    skip = False
    for w in args:
        if skip:
            skip = False
            continue
        if w.text.startswith("-"):
            if re.match(r"^-[a-zA-Z]*[eE]$", w.text):
                skip = script_seen = True  # the next word is the program text
            continue
        if not script_seen:
            script_seen = True  # the first operand is the script file
            continue
        out.append(w)
    return out


def _git_writes(args: list[Expanded]) -> list[Expanded]:
    """The paths a (non-read-only) git call is known to WRITE.

    ``-C``/``--git-dir``/``--work-tree`` count as writes: a non-read-only
    subcommand there can rewrite any file of that tree (``checkout``,
    ``reset --hard``, ``apply``, ``stash pop``). Clone and worktree-add
    destinations are the resolver's creations, checked as writes in classify.
    """
    out: list[Expanded] = []
    rest = list(args)
    while rest and rest[0].text.startswith("-"):
        t = rest[0].text
        if t in ("-C", "--git-dir", "--work-tree", "--namespace") and len(rest) > 1:
            if t != "--namespace":
                out.append(rest[1])
            rest = rest[2:]
            continue
        if t.startswith(("--git-dir=", "--work-tree=")):
            out.append(Expanded(text=t.split("=", 1)[1], empty=rest[0].empty))
        rest = rest[2:] if t == "-c" else rest[1:]
    sub = rest[0].text if rest else ""
    tail = rest[1:]
    out += _flag_value(tail, "", "--output") + _flag_value(tail, "", "--output-directory")
    out += _flag_value(tail, "", "--prefix")  # checkout-index --prefix=<dir>/
    for k, w in enumerate(tail):
        t = w.text
        if t == "-o" and k + 1 < len(tail):
            out.append(tail[k + 1])
        elif t.startswith("-o") and len(t) > 2 and not t.startswith("--"):
            out.append(Expanded(text=t[2:], empty=w.empty))
    operands = [w for w in tail if not w.text.startswith("-")]
    if sub == "init":
        out += operands[-1:] or [Expanded(text=".")]
    elif sub == "bundle" and len(operands) >= 2 and operands[0].text == "create":
        out.append(operands[1])
    return out


_PERL_INPLACE_RE = re.compile(r"^-[lnpaesw0]*i")


def _tar_mode(args: list[Expanded]) -> str:
    """``extract`` / ``create`` / ``list`` for a tar argv (unknown → ``extract``,
    the writing reading, so a planter is never missed)."""
    skip = False
    for k, w in enumerate(args):
        t = w.text
        if skip:
            skip = False
            continue
        if t in ("--create", "--append", "--update"):
            return "create"
        if t == "--list":
            return "list"
        if t in ("--extract", "--get"):
            return "extract"
        if t in _TAR_VALUE_FLAGS:
            skip = True  # ``-C DIR`` / ``-f FILE``: the value is not a mode
            continue
        # A short cluster (``-czf``), or the dashless first word (``tar xf``).
        if (t.startswith("-") and not t.startswith("--")) or (k == 0 and t.isalpha()):
            letters = t.lstrip("-")
            # Letters after ``f``/``C``/``T``/``X`` in a cluster are its value.
            letters = re.split(r"[fCTX]", letters, maxsplit=1)[0]
            if "x" in letters:
                return "extract"
            if any(c in letters for c in "cru"):
                return "create"
            if "t" in letters:
                return "list"
    return "extract"


#: Programs whose sources the backstop may skip once ``_write_surface`` has
#: identified their destination (review R5 exemption (b)).
_COPY_FAMILY = frozenset({"cp", "gcp", "rsync", "zip", "tar", "gtar", "bsdtar"})
#: A path-looking run inside any word: absolute, ``~/``, ``$HOME``/``${HOME}``,
#: or a relative run containing ``/``. Deliberately over-matching.
_MENTION_RE = re.compile(
    r"(?:\$\{?HOME\}?|~|\.{1,2})?/[^\s'\"`;|&<>(){}]*"
    # A relative run may not start mid-word (``-st/abs`` is the absolute
    # ``/abs`` glued to a flag, not a relative ``-st/…``).
    # Its first component may itself be a glob (``lnk*/origin.json``,
    # ``*/x``): that spelling skipped the backstop (review round 8, MAJOR-2).
    r"|(?<![\w.~$*?\[\]-])[\w.*?\[][\w.*?\[\]-]*/[^\s'\"`;|&<>(){}]*"
)


_GLOB_CHARS = re.compile(r"[*?\[{]")
#: Basenames of the files the guard reads as evidence (``classify``'s
#: ``_EVIDENCE_FILES`` plus the ``guardrails/`` records).
_EVIDENCE_NAMES = (
    "origin.json",
    "transcript.jsonl",
    "asks.jsonl",
    "subagent-roster*",
    "created.sqlite",
    "lineage.jsonl",
)


def _names_evidence(word: str) -> bool:
    """Could ``word`` be an evidence file's name? Case-insensitive, with
    sqlite's ``-wal``/``-shm``/``-journal`` side files, and in BOTH
    directions so a glob word (``ORIG*.json``) counts (review round 7)."""
    folded = word.casefold()
    for name in _EVIDENCE_NAMES:
        for candidate in (name, name + "-wal", name + "-shm", name + "-journal"):
            if fnmatch.fnmatchcase(folded, candidate) or fnmatch.fnmatchcase(candidate, folded):
                return True
    return False


def _literal_prefix(pattern: str) -> str:
    """The deepest directory of ``pattern`` above its first glob component."""
    parts = pattern.split("/")
    for k, part in enumerate(parts):
        if _GLOB_CHARS.search(part):
            return "/".join(parts[:k]) or "/"
    return os.path.dirname(pattern) or "/"


def _mention_path(raw: str, home: str, cwd: str | None) -> str | None:
    """An absolute, normalised spelling of a ``_MENTION_RE`` match, or None."""
    text = re.sub(r"^\$\{?HOME\}?", home, raw)
    if text.startswith("~/"):
        text = home + text[1:]
    if not text.startswith("/"):
        if cwd is None:
            return None
        text = os.path.join(cwd, text)
    return os.path.normpath(text)


#: Anything that can make ``**`` recursive for this command (review round 9,
#: MAJOR-1): the word ``globstar`` (``shopt -s globstar``, ``bash -O
#: globstar``) and shells where ``**/`` recurses by default or by ``setopt``.
#: A bare ``-O`` is NOT here (round 10, MINOR-1): ``curl -O``/``gcc -O2``
#: would turn on the deep walk for ordinary cleanups and hit the cap.
#: Over-matching only costs a wider, still bounded, walk; a cap overflow
#: ranks HIGH, never CRITICAL.
_RECURSIVE_GLOB_RE = re.compile(r"globstar|\b(?:zsh|ksh|fish|setopt)\b")


def resolve_command(
    command: str, *, cwd: str | None, env: Mapping[str, str], home: str
) -> Resolution:
    """Every path ``command`` would destroy, resolved statically. Never runs it."""
    walker = _Walker(env, cwd, home)
    # Raw text, not parsed words, so a nested ``bash -c``/``zsh -c`` string
    # counts too. Deliberately one-way: ``shopt -u globstar`` is ignored
    # because a wider walk only errs safe (review round 9, NIT-1).
    walker.result.globstar = bool(_RECURSIVE_GLOB_RE.search(command))
    try:
        walker.walk(command)
    except Exception as exc:  # noqa: BLE001 — a parser fault must not wave a call through
        walker.result.opaque_reasons.append(f"resolver error: {type(exc).__name__}")
    return walker.result
