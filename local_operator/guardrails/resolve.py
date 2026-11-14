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
import itertools
import os
import re
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field

from local_operator.tools.search_guard import (
    _DURATION_RE,
    _MAX_PEEL_DEPTH,
    _SHELL_C_WORDS,
    _WRAPPER_LONG_VALUE_FLAGS,
    _WRAPPER_OWN_LONG_VALUE_FLAGS,
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
#: Prefix of a ``for VAR in <glob>`` binding: ``FOR_GLOB_MARK + <key> +
#: OPAQUE_MARK``, where ``<key>`` indexes ``_Walker.for_globs``. It contains
#: OPAQUE_MARK, so every use of ``$VAR`` that is not a whole recorded word
#: (``cd $f``, ``x-$f``, ``$f/..``) stays opaque exactly as before; only a
#: word that IS the variable is pinned to the glob (see ``_Walker._add``).
#: Carried in the VALUE, not a side table keyed by name, so a reassignment
#: (``f=$(cmd)``) can never inherit a stale glob, and a copy (``g=$f``)
#: carries it correctly.
FOR_GLOB_MARK = OPAQUE_MARK + "for-glob:"


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
    #: Where each ``$NAME``/``${NAME}`` value landed in ``text`` (and in
    #: ``pattern``): ``(name, text_start, text_end, pattern_start, pattern_end,
    #: quoted)``. A ``for`` list fans a target out by rewriting exactly these
    #: spans — never by searching the text for the variable's VALUE, which
    #: once turned ``$l`` = ``es`` into ``s<fr>sions`` (incident 2026-11-09).
    refs: list[tuple[str, int, int, int, int, bool]] = field(default_factory=list)


def _glob_escape(text: str) -> str:
    return re.sub(r"([*?\[])", r"[\1]", text)


#: Upper bound on the words one brace expression may produce. Past it the word
#: is opaque: ``{1..100000}`` is not something to enumerate per bash call.
MAX_BRACE_RESULTS = 1024
#: Same bound for the combinations nested literal ``for`` lists fan one word into.
_MAX_FOR_COMBINATIONS = MAX_BRACE_RESULTS
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
        ref = _param_ref(part.text, env) if part.kind in ("pbare", "pdq") else None
        text_at = sum(map(len, text_parts)) if ref is not None else 0
        pattern_at = sum(map(len, pattern_parts)) if ref is not None else 0
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
            if ref is not None and value:
                out.refs.append(
                    (
                        ref,
                        text_at,
                        sum(map(len, text_parts)),
                        pattern_at,
                        sum(map(len, pattern_parts)),
                        part.kind == "pdq",
                    )
                )
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


def _param_ref(form: str, env: Mapping[str, str]) -> str | None:
    """The variable whose value ``form`` expands to verbatim, else ``None``.

    ``$NAME``/``${NAME}``, and ``${NAME:-word}`` while NAME is set (its value
    is used, not the default). Anything else (a default word, ``${#x}``) is
    not a plain reference, so a ``for`` list never rewrites it.
    """
    inner = form[2:-1] if form.startswith("${") else form[1:]
    if _NAME_RE.fullmatch(inner):
        return inner
    dm = _BRACE_DEFAULT_RE.match(inner)
    if dm is not None:
        current = env.get(dm.group(1))
        if current is not None and not (dm.group(2).startswith(":") and current == ""):
            return dm.group(1)
    return None


def _fan_out(exp: Expanded, values: Mapping[str, str]) -> Expanded:
    """``exp`` with every recorded ref to a name in ``values`` rewritten."""
    text, pattern = exp.text, exp.pattern
    # Right to left, so earlier offsets stay valid.
    for name, ts, te, ps, pe, quoted in sorted(exp.refs, key=lambda r: r[1], reverse=True):
        if name not in values:
            continue
        text = text[:ts] + values[name] + text[te:]
        if pattern is not None:
            pattern = (
                pattern[:ps]
                + (_glob_escape(values[name]) if quoted else values[name])
                + pattern[pe:]
            )
    return Expanded(
        text=text,
        pattern=pattern,
        empty=list(exp.empty),
        opaque=exp.opaque,
        opaque_contained=exp.opaque_contained,
        temp=exp.temp,
    )


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
    #: pad (review R1-MAJOR-1, PR #3602). ``classify`` therefore decides a
    #: scratchpad exemption for such a word ONLY from where the kernel lands
    #: it (``classify._kernel_reason``), never from ``path``.
    dotdot: bool = False
    #: The absolute but UN-normalised join of the word (``None`` for a glob or
    #: opaque word). ``os.path.realpath`` of it resolves ``..`` physically, so
    #: ``classify`` adds that spelling to the ones it judges.
    raw_path: str | None = None
    #: A ``..`` in the word whose physical landing cannot be pinned now (a
    #: component before it is missing or unsearchable, or the ``..`` follows a
    #: glob component): ``path`` is only the lexical reading. ``classify``
    #: keeps any CRITICAL that reading earns but never ranks the target LOW,
    #: not even inside this session's scratchpad, because the missing component
    #: may be created as a link by an earlier part of the same command
    #: (``ln -s <repo>/sub <pad>/n && rm -rf <pad>/n/../.git``). Also set when
    #: the component EXISTS but an earlier part of the command named it — as
    #: an ``ln`` link, a ``mv``/``cp``/``rsync`` destination, or a delete or
    #: creation target — since it may be a different link by then
    #: (``_Walker._unpin_relinked``, review R1-1 on PR #3811).
    unpinned: bool = False
    #: For an ``unpinned`` word: where it lands if every missing component
    #: becomes a directory and every component an earlier ``ln -s``/``cp -s``
    #: of THIS command re-linked to a literal target points there
    #: (:func:`_relinked_landing`). ``classify`` ranks this reading alongside
    #: ``path`` and keeps the worse, so a ``..`` that crosses a link after the
    #: missing part, or follows a re-link into the state root or another
    #: session, stays CRITICAL and the judge is shown that file (review R2-1,
    #: QA Q2-1 on PR #3811). Empty when it equals ``path`` or cannot be built.
    landings: tuple[str, ...] = ()
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
    #: Double force is narrowly allowed for a verified stale git initialization lock.
    force_count: int = 0
    #: A target only an ALTERNATE parse produces (GNU sed's reading of BSD
    #: ``-i ''``, review M1 on PR #3816): ``classify`` keeps it only when it
    #: ranks CRITICAL, so it can raise a verdict to CRITICAL and never add a
    #: HIGH for what is, under the other reading, the script.
    critical_only: bool = False
    #: A ``touch`` handed to an interpreter (``node``/``python``) running a
    #: script FILE: that script's absolute path. ``classify`` lets a read of a
    #: same-task-tree session's scratch file rank LOW only when the script is
    #: git-tracked (case 7, 2026-11-09: ``node scripts/i18n/ledger-assemble.mjs
    #: --receipt <auditor pad>/…json`` from a sibling lane).
    script: str | None = None
    #: A ``touch`` handed to ``python -c CODE`` whose CODE is proven unable to
    #: write, delete, rename or run anything (:func:`python_code_read_only`,
    #: :meth:`_Walker._inline_read_only`). Like ``script``, it lets ``classify``
    #: rank a read of a same-task-tree session's scratch LOW (incident
    #: 2026-11-13); it never lowers a delete, a write or a stranger's pad.
    read_only_code: bool = False
    #: A ``touch`` handed to ``python -c CODE`` that ``read_only_code`` did
    #: NOT prove (``_python_inline_code``): #3918's proof refused the code or
    #: could not pin the interpreter, so ``classify._tree_read`` gives it no
    #: lineage credit either (convergence of #3906 with #3918).
    unproven_code: bool = False
    #: The mention sits in a word the command names as an OUTPUT (``-o``,
    #: ``--out*``, ``--write*``, glued or as the next word): never a read.
    as_output: bool = False
    #: The program may run code this check cannot see on the mention (a
    #: ``sed`` with ``e``/``s///e``/``-f``, ``_sed_executes``): never a read,
    #: so ``classify._tree_read`` gives it no lineage credit (QA Q2, #3906).
    executes: bool = False
    #: The mention is the value of an environment assignment, inline
    #: (``A=<p> cmd``) or an ``env`` operand (``env A=<p> cmd``): that NAME.
    #: Told to the judge, never a ranking input (PR #3906 round 1, case C).
    assigned: str | None = None


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

    #: ``symlink`` is the link ENTRY ``ln -s[f] SRC LINK`` makes, never the
    #: directory it points at (see ``provenance.owned_link``).
    kind: str  # "mkdir" | "mktemp" | "worktree" | "clone" | "symlink"
    path: str | None = None
    parent: str | None = None
    pattern: str | None = None


@dataclass
class _State:
    env: dict[str, str]
    cwd: str | None
    home: str
    private_cwd: bool = False
    #: ``set -P`` / ``set -o physical``: every later ``cd`` resolves ``..``
    #: physically, as ``cd -P`` does (review R2-MINOR-2).
    physical: bool = False


def kernel_path(raw: str | None) -> str | None:
    """Where the kernel lands ``raw`` (absolute, un-normalised), resolving
    ``..`` AFTER links as path lookup does; ``None`` if that is unknowable.

    ``os.path.realpath`` of the nearest EXISTING ancestor, plus the remaining
    components. Those must hold no ``..``: below a missing directory the
    lookup itself fails, and a guess there must not grant an exemption
    (so ``mkdir -p <pad>/n && rm -f <pad>/n/../x`` is judged unknowable at
    check time, before ``n`` exists — deliberately).
    """
    if not raw or not raw.startswith("/"):
        return None
    parts = [c for c in raw.split("/") if c and c != "."]
    for cut in range(len(parts), -1, -1):
        head = "/" + "/".join(parts[:cut])
        try:
            exists = os.path.lexists(head) if cut == len(parts) else os.path.isdir(head)
        except OSError:
            exists = False
        if not exists:
            continue
        rest = parts[cut:]
        if ".." in rest:
            return None
        try:
            return os.path.join(os.path.realpath(head), *rest)
        except OSError:
            return None
    return None


def physical_path(raw: str) -> str | None:
    """Where ``rm``/``>``/``mv`` act on the absolute word ``raw``: every ``..``
    applied AFTER the links before it are followed, the FINAL component kept
    as written (``rm`` removes a trailing link itself, not what it points at;
    the trailing-slash form is covered by ``classify._spellings`` adding the
    ``realpath``). ``None`` when a component before a ``..`` cannot be
    resolved (missing, unsearchable).

    WHY. ``os.path.normpath`` collapses ``..`` lexically, but path lookup does
    not: with ``~/other/sl -> <repo>/subdir``, ``~/other/sl/../.git`` IS
    ``<repo>/.git``, while normpath reads ``~/other/.git``. Ranking the lexical
    form let any CRITICAL or HIGH target be reached at LOW through a link plus
    ``..`` (security review on PR #3802). A word with no ``..`` cannot differ,
    so it keeps the lexical form; one whose physical landing is the same place
    keeps the spelling it was written with, so reasons and provenance read as
    the agent wrote them.
    """
    lexical = os.path.normpath("/" + raw.lstrip("/"))
    if ".." not in raw.split("/"):
        return lexical
    head, _sep, last = raw.rstrip("/").rpartition("/")
    if last == "..":
        # ``sl/..`` names the parent of what ``sl`` resolves to.
        physical = kernel_path(raw)
    else:
        parent = kernel_path(head or "/")
        physical = None if parent is None else os.path.join(parent, last)
    if physical is None:
        return None
    try:
        same = os.path.realpath(os.path.dirname(lexical)) == os.path.realpath(
            os.path.dirname(physical)
        ) and os.path.basename(lexical) == os.path.basename(physical)
    except OSError:
        same = False
    return lexical if same else os.path.normpath(physical)


def _physical(text: str, cwd: str | None) -> tuple[str | None, bool, bool]:
    """``(absolute path, cwd_was_unknown, unpinned)`` for a word's text, ``..``
    resolved the way the kernel resolves it (:func:`physical_path`).

    ``unpinned``: the word has a ``..`` whose physical landing cannot be known
    now (a component before it is missing or unsearchable — it may be created
    as a link by an earlier part of this very command). ``path`` is then the
    lexical reading, so a protected name it spells is still caught, and the
    caller must not let that reading earn a LOW (``Target.unpinned``).
    """
    if text.startswith("/"):
        raw = text
    elif cwd is None:
        return None, True, False
    else:
        raw = os.path.join(cwd, text)
    path = physical_path(raw)
    if path is None:
        return os.path.normpath("/" + raw.lstrip("/")), False, True
    return path, False, False


#: Links :func:`_relinked_landing` follows in ONE walk, in total, before it
#: gives up — Linux's MAXSYMLINKS (macOS stops at 32), after which the kernel
#: itself fails the lookup with ELOOP. A bound on nesting alone was not enough:
#: ``l_k -> l_{k+1}/../l_{k+1}`` resolved each sibling again at every level, so
#: 16 planted links took ~20 s and 24 would take hours, before and outside the
#: judge timeout (review R3-2, PR #3811).
_MAX_RELINK_HOPS = 40


def _relinked_landing(
    raw: str,
    links: Mapping[str, str],
    *,
    follow_last: bool = False,
    memo: dict[str, str] | None = None,
) -> str | None:
    """Where the kernel lands the absolute, un-normalised ``raw`` if every
    MISSING component becomes a directory and each component in ``links``
    (physical link path -> literal target, ``_Walker.link_targets``) is the
    symlink an earlier part of this command made. ``None`` past a link loop.

    WHY. ``physical_path`` cannot pin a ``..`` after a component that is
    missing or re-linked, and its lexical fallback collapses EVERY ``..``,
    including one that comes after an existing link further on:
    ``rm -f <pad>/nope/../slk/../config.yml`` with ``slk -> <state>/secrets``
    reads as ``<pad>/config.yml`` while the kernel, once ``nope`` exists,
    removes ``<state>/config.yml`` (review R2-1, PR #3811). Walking the word
    one component at a time — dropping only each ``<missing>/..`` pair,
    following existing links and literal re-links as lookup does — gives the
    landing the command produces in its ordinary reading. A re-link whose
    target cannot be known (``mv``, ``cp -a``, a variable) is not in
    ``links``, so it reads as what is on disk now; the target stays unpinned
    and is never ranked below HIGH, which is the conservative answer there.
    ``follow_last``: resolve the final component too (a link's own target).
    ``memo`` (link path -> where it resolves) is valid only while ``links``
    and the filesystem are unchanged; the walker clears its copy whenever
    ``link_targets`` changes. ``None`` too once :data:`_MAX_RELINK_HOPS`
    links were followed: unknown, which ranks HIGH (CRITICAL if the lexical
    reading is), the fail-closed answer for a lookup the kernel refuses too.
    """
    return _relinked_walk(raw, links, follow_last, [_MAX_RELINK_HOPS], {} if memo is None else memo)


def _relinked_walk(
    raw: str, links: Mapping[str, str], follow_last: bool, budget: list[int], memo: dict[str, str]
) -> str | None:
    """One :func:`_relinked_landing` walk; ``budget`` is shared by every
    recursive call of it, so the total number of links followed is bounded."""
    comps = [c for c in raw.split("/") if c not in ("", ".")]
    cur = "/"
    for k, comp in enumerate(comps):
        if comp == "..":
            cur = os.path.dirname(cur)
            continue
        nxt = os.path.join(cur, comp)
        if k == len(comps) - 1 and not follow_last:
            return nxt
        pointee = links.get(nxt)
        if pointee is None:
            try:
                pointee = os.readlink(nxt) if os.path.islink(nxt) else None
            except OSError:
                pointee = None
        if pointee is None:
            # A directory, a file, or missing: missing reads as a directory.
            cur = nxt
            continue
        resolved = memo.get(nxt)
        if resolved is None:
            budget[0] -= 1
            if budget[0] < 0:
                return None
            resolved = _relinked_walk(
                pointee if pointee.startswith("/") else os.path.join(cur, pointee),
                links,
                True,
                budget,
                memo,
            )
            if resolved is None:
                return None
            memo[nxt] = resolved
        cur = resolved
    return cur


def _physical_pattern(pattern: str) -> tuple[str, bool]:
    """``(pattern, unpinned)``: an absolute glob with its LITERAL directory
    prefix resolved by :func:`physical_path` (bash's expansion opens
    ``sl/../`` through the link, so ``~/other/sl/../.wor*`` matches
    ``<repo>/.worktrees``). A ``..`` AFTER a glob component (``*/../x``) cannot
    be pinned before expansion: the lexical form is kept for the CRITICAL
    name checks and the result is ``unpinned``."""
    parts = pattern.split("/")
    cut = next((k for k, part in enumerate(parts) if _GLOB_CHARS.search(part)), len(parts))
    prefix, rest = "/".join(parts[:cut]) or "/", parts[cut:]
    if ".." in rest:
        return os.path.normpath(pattern), True
    resolved = physical_path(prefix)
    if resolved is None:
        return os.path.normpath(pattern), True
    return os.path.normpath(os.path.join(resolved, *rest)), False


def _absolute(text: str, cwd: str | None) -> tuple[str | None, bool]:
    """``(absolute path, cwd_was_unknown)`` for a word's text: ``..`` resolved
    physically where it can be (:func:`_physical`), lexically otherwise."""
    path, unknown, _unpinned = _physical(text, cwd)
    return path, unknown


def _logical(text: str, cwd: str | None) -> str | None:
    """A LOGICAL ``cd``'s destination: bash (and zsh, ``sh``) canonicalise
    ``..`` lexically BEFORE ``chdir``, so ``cd ~/other/sl/..`` really lands in
    ``~/other`` even when ``sl`` is a link (measured 2026-11-11, ``pwd -P``).
    This is the one place a lexical collapse matches the kernel."""
    if text.startswith("/"):
        return os.path.normpath("/" + text.lstrip("/"))
    return None if cwd is None else os.path.normpath(os.path.join(cwd, text))


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
    unpinned = False
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
        path, unknown, unpinned = _physical(literal or ".", state.cwd)
        if pattern is not None:
            base = state.cwd or ""
            pattern = pattern if pattern.startswith("/") else os.path.join(base, pattern)
            pattern, pattern_unpinned = (None, False) if unknown else _physical_pattern(pattern)
            unpinned = unpinned or pattern_unpinned
        raw_path = None
    else:
        path, unknown, unpinned = _physical(text or ".", state.cwd)
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
        unpinned=unpinned and not is_temp,
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


#: Interpreters whose first positional word is a script FILE they run.
_INTERPRETER_RE = re.compile(r"(?:node|python(?:3(?:\.\d+)?)?)")
#: An option naming an output: ``-o``, ``--out``, ``--output``, ``--outfile``,
#: ``--write``, ``--write-to``… alone (value in the next word) or glued, and
#: the in-place forms ``-i``/``--in-place``/``--inplace`` (review R2-m2): a
#: script told to rewrite its input writes the file it was handed.
_OUTPUT_FLAG_RE = re.compile(r"(?:-o|-i|--out[\w-]*|--write[\w-]*|--in-?place)")
_OUTPUT_GLUED_RE = re.compile(r"(?:-o.|-i.|--out[\w-]*=|--write[\w-]*=|--in-?place=)")


#: CPython's long options (``Python/getopt.c``, 3.12–3.14). The valued one
#: takes its value from the NEXT word only (getopt matches the whole word, so
#: ``--opt=value`` is not this option and fails closed as unknown).
_PY_VALUED_LONG_OPTS = frozenset({"--check-hash-based-pycs"})
_PY_FLAG_LONG_OPTS = frozenset(
    {"--help", "--help-env", "--help-xoptions", "--help-all", "--version"}
)


def _python_inline_code(name: str, words: list[Expanded]) -> bool:
    """Does this ``python`` run inline CODE (``-c``, glued or clustered)?

    WHY (convergence of PR #3906 with #3918, 2026-11-13): #3918 made
    ``python -c`` a PROOF-carrying lane — its CODE is visible, so a read of a
    task-tree pad earns LOW only when :func:`python_code_read_only` proves it
    and the interpreter is pinned (``Target.read_only_code``). #3906's lineage
    credit (``classify._tree_read``) grants LOW to code it cannot see at all
    (``node -e``, a script file); without this flag it also credited an
    UNPROVEN ``python -c`` — a ``write_text``/``unlink`` the proof had just
    refused, or proven code under a repathed interpreter — so #3918's HIGH
    rows ranked LOW on a child's pad. Such a target carries
    ``Target.unproven_code`` and gets no lineage credit.

    Parses the option prefix the way CPython does: ``-c`` (alone, glued
    ``-cCODE`` or clustered ``-Ic``) means inline code; ``-m`` or the first
    non-option word (a script, ``-`` for stdin) ends the scan; ``-W``/``-X``
    take a value. An opaque word in the prefix fails closed (``True``).

    ``words[0]`` must be the INTERPRETER word: callers holding a wrapped argv
    (``env``/``command``/``nice``/``xargs python3 -c …``) slice the wrapper
    prefix off first, or ``python3`` itself reads as a script and ends the
    scan (convergence review m1). Long options (m2): CPython's valued one
    (``--check-hash-based-pycs <mode>``) skips its value, so the mode is not
    taken for a script; the flag-only ones (``--help``, ``--version`` …) are
    stepped over; any OTHER ``--`` word fails closed (``True``) — a spelling
    this table does not model (a ``=`` glued value, a future option) must not
    turn ``-c`` code into a credited script file.
    """
    if not _INTERPRETER_RE.fullmatch(name) or name.startswith("node"):
        return False
    i = 1
    while i < len(words):
        if words[i].opaque:
            return True
        text = words[i].text
        if text == "--" or text == "-" or not text.startswith("-"):
            return False
        if text.startswith("--"):
            if text in _PY_VALUED_LONG_OPTS:
                i += 1  # the value is the next word, never the script
            elif text not in _PY_FLAG_LONG_OPTS:
                return True
        else:
            flags = text[1:]
            for j, ch in enumerate(flags):
                if ch == "c":
                    return True
                if ch == "m":
                    return False
                if ch in "WX":
                    i += j == len(flags) - 1  # the value is the next word
                    break
        i += 1
    return False


def _interpreter_script(name: str, words: list[Expanded], cwd: str | None) -> str | None:
    """The script ``node``/``python`` runs, as an absolute path, else ``None``.

    Only the plain shape ``<interpreter> <script> …``: any option before the
    script (``-m``, ``-c``, ``-e``, ``--require x``) gives ``None``, so the
    caller keeps the conservative ranking rather than guess which word runs.
    """
    if not _INTERPRETER_RE.fullmatch(name) or cwd is None:
        return None
    idx = next((i for i, w in enumerate(words) if os.path.basename(w.text) == name), None)
    if idx is None or idx + 1 >= len(words):
        return None
    word = words[idx + 1]
    if word.opaque or word.pattern is not None or not word.text or word.text.startswith("-"):
        return None
    return os.path.normpath(os.path.join(cwd, os.path.expanduser(word.text)))


def _output_word_ids(words: list[Expanded]) -> set[int]:
    """``id()`` of each word the command names as an output (see ``Target.as_output``)."""
    out: set[int] = set()
    for i, w in enumerate(words):
        if _OUTPUT_GLUED_RE.match(w.text):
            out.add(id(w))
        if _OUTPUT_FLAG_RE.fullmatch(w.text) and i + 1 < len(words):
            out.add(id(words[i + 1]))
    return out


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
                takes = (
                    (len(t) == 2 and t[1] in value_flags)
                    or t in _WRAPPER_LONG_VALUE_FLAGS
                    or t in _WRAPPER_OWN_LONG_VALUE_FLAGS.get(name, ())
                    # env clusters too (``-iC /tmp``), with the rule
                    # ``_env_split`` uses, so both agree on the program word.
                    or (name == "env" and _env_takes_next_word(t))
                )
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


#: WHY AN ALLOW-LIST OVER THE WHOLE AST (review r1 M1 + r2 M3, PR #3918):
#: two rounds of a DENY-list over Python's attribute graph each found a new
#: route to ``getattr`` by a string — ``string.Formatter().get_field("0.unlink"
#: …)``, ``pathlib.operator.methodcaller("unlink")``, then a generator's
#: ``gi_frame.f_builtins["getattr"](p, "un"+"link")()`` — and every one deleted
#: real files at LOW. The attribute graph is open-ended, so a deny-list can
#: never be shown complete. :func:`python_code_read_only` therefore admits a
#: program only when EVERY node type, module, name, attribute, keyword and
#: subscript is on one of the lists below; anything not listed fails closed to
#: the backstop's HIGH (a judge call, never a silent LOW). A read the lists do
#: not cover costs a judge call — widen a list only with a reason here.
#:
#: Statement and expression node types, by ``type(node).__name__`` (so
#: ``ast`` stays a lazy import). Deliberately absent: ``Lambda``,
#: ``FunctionDef``/``ClassDef`` (a body can run on attribute access or
#: decorate), ``Match`` (class patterns are ``getattr`` by string),
#: ``NamedExpr``, ``With``, ``Try``, ``While``, ``Await``/``Yield``,
#: ``Global``/``Nonlocal``, ``Starred``, ``Delete``, ``Pow`` and the
#: bit operators (nothing a read needs). ``GeneratorExp`` stays: the incident
#: itself joins one, and the M3 route was the generator's ``gi_frame``
#: ATTRIBUTE, which the attribute allow-list never admits — without that a
#: generator is just a lazy list comprehension.
_PY_ALLOWED_NODES = frozenset(
    {
        # statements
        "Module",
        "Expr",
        "Assign",
        "AugAssign",
        "For",
        "If",
        "Pass",
        "Break",
        "Continue",
        "Import",
        "ImportFrom",
        "alias",
        # expressions
        "BoolOp",
        "BinOp",
        "UnaryOp",
        "Compare",
        "IfExp",
        "Call",
        "keyword",
        "Attribute",
        "Subscript",
        "Slice",
        "Name",
        "Constant",
        "List",
        "Tuple",
        "Dict",
        "Set",
        "JoinedStr",
        "FormattedValue",
        "ListComp",
        "SetComp",
        "DictComp",
        "GeneratorExp",
        "comprehension",
        # contexts and operators (``Del`` and ``Pow``/bit ops are absent)
        "Load",
        "Store",
        "And",
        "Or",
        "Add",
        "Sub",
        "Mult",
        "Div",
        "FloorDiv",
        "Mod",
        "USub",
        "UAdd",
        "Not",
        "Eq",
        "NotEq",
        "Lt",
        "LtE",
        "Gt",
        "GtE",
        "In",
        "NotIn",
        "Is",
        "IsNot",
    }
)
#: Modules a read may import, by exact name (a dotted ``import json.tool``
#: loads a CLI whose ``main()`` writes ``argv[2]``, so dotted names fail).
#: The module object is harmless only because the attribute allow-list below
#: limits what can be taken off it: no module-valued attribute is listed, so
#: re-exports (``pathlib.os``, ``glob.operator``) are unreachable. ``string``,
#: ``functools``, ``operator``, ``itertools`` (callable-taking helpers) and
#: ``pprint`` (``pprint(obj, stream)`` writes to a positional stream) are out.
_PY_READ_MODULES = frozenset(
    {"pathlib", "json", "re", "glob", "fnmatch", "collections", "datetime"}
)
#: Builtins a read may name. None takes a callable it later calls through a
#: positional argument, so a held callable can only run by a direct call the
#: walk sees. ``open`` is admitted only as a callee with a literal read mode
#: (:func:`_py_open_read_only`). Absent on purpose: ``getattr``/``exec``/
#: ``eval``/``compile``/``__import__``/``vars``/``globals``/``type``,
#: ``map``/``filter``/``iter``/``next`` (callables or protocol reach),
#: ``isinstance`` (no read needs it), ``input``/``breakpoint``/``help``.
_PY_ALLOWED_BUILTINS = frozenset(
    {
        "print",
        "open",
        "len",
        "sorted",
        "reversed",
        "min",
        "max",
        "sum",
        "any",
        "all",
        "enumerate",
        "zip",
        "range",
        "str",
        "int",
        "float",
        "bool",
        "bytes",
        "list",
        "dict",
        "set",
        "tuple",
        "repr",
        "abs",
        "round",
    }
)
#: Every attribute name a read may use, on ANY object: the check is untyped,
#: so each name must be harmless on every object reachable from the modules
#: and builtins above. Notably absent: ``replace`` (``Path.replace`` RENAMES;
#: untyped, so ``str.replace`` goes too), ``get`` (``b.get("getattr")`` once a
#: dict of builtins is in hand), ``format``/``format_map`` (attribute walks by
#: string), ``open``/``write*``/``unlink``/``rename``/``touch``/``mkdir``/
#: ``chmod``/``copy``/``move``, every dunder, ``gi_frame``/``f_builtins``-
#: style introspection, and any name that hands back a module able to act.
#: Checked transitively on CPython 3.12 and 3.14: the only modules reachable
#: through these names are the allowed ones plus ``glob.stat`` (the ``stat``
#: module: constants and pure mode predicates). ``stat`` itself is there for
#: ``Path.stat()`` (``os.stat``: metadata only). ``sub``/``subn`` are
#: further limited in :func:`python_code_read_only` to a literal replacement.
_PY_ALLOWED_ATTRS = frozenset(
    {
        # pathlib
        "Path",
        "PurePath",
        "read_text",
        "read_bytes",
        "iterdir",
        "glob",
        "rglob",
        "exists",
        "is_file",
        "is_dir",
        "is_symlink",
        "stat",
        "st_size",
        "st_mtime",
        "name",
        "suffix",
        "suffixes",
        "stem",
        "parent",
        "parents",
        "parts",
        "as_posix",
        "joinpath",
        "relative_to",
        "with_suffix",
        "with_name",
        "resolve",
        # file object from a read-mode ``open``
        "read",
        "readline",
        "readlines",
        # str / bytes
        "splitlines",
        "split",
        "rsplit",
        "strip",
        "lstrip",
        "rstrip",
        "join",
        "startswith",
        "endswith",
        "lower",
        "upper",
        "count",
        "find",
        "rfind",
        "index",
        "partition",
        "rpartition",
        "isdigit",
        "isspace",
        "decode",
        "encode",
        # containers (in-memory only)
        "keys",
        "values",
        "items",
        "append",
        "extend",
        "add",
        # json (keyword hooks are refused by the keyword allow-list)
        "loads",
        "load",
        "dumps",
        # re
        "compile",
        "search",
        "match",
        "fullmatch",
        "findall",
        "finditer",
        "sub",
        "subn",
        "escape",
        "group",
        "groups",
        "start",
        "end",
        "span",
        "IGNORECASE",
        "MULTILINE",
        "DOTALL",
        "I",
        "M",
        "S",
        # glob / fnmatch
        "iglob",
        "fnmatch",
        "filter",
        # collections
        "Counter",
        "most_common",
        # datetime
        "datetime",
        "date",
        "time",
        "now",
        "fromtimestamp",
        "isoformat",
        "strftime",
    }
)
#: Keyword arguments a call may pass. ``key=``, ``default=``, ``object_hook=``,
#: ``cls=``, ``opener=`` take callables; ``file=``/``stream=`` take a writer.
#: ``mode=`` is admitted only for builtin ``open`` with a literal read mode.
_PY_ALLOWED_KEYWORDS = frozenset(
    {
        "encoding",
        "errors",
        "reverse",
        "sep",
        "end",
        "flush",
        "maxsplit",
        "flags",
        "count",
        "indent",
        "sort_keys",
        "ensure_ascii",
        "recursive",
        "start",
        "mode",
    }
)
_PY_READ_MODES = frozenset({"r", "rb", "rt", "br", "tr"})


def _py_open_read_only(call: object) -> bool:
    """Is a builtin ``open(...)`` call literally a read?

    At most ``file, mode`` positionally: ``opener`` is ``open``'s eighth
    positional parameter and can do anything with the path.
    """
    import ast

    assert isinstance(call, ast.Call)
    if len(call.args) > 2:
        return False
    mode: ast.expr | None = call.args[1] if len(call.args) > 1 else None
    for kw in call.keywords:
        if kw.arg == "mode":
            mode = kw.value
    if mode is None:
        return True
    return (
        isinstance(mode, ast.Constant)
        and isinstance(mode.value, str)
        and mode.value in _PY_READ_MODES
    )


def _py_small_int(node: object) -> bool:
    """A literal int, optionally negated (``x[0]``, ``x[-2500:]``)."""
    import ast

    if isinstance(node, ast.UnaryOp) and isinstance(node.op, (ast.USub, ast.UAdd)):
        node = node.operand
    return (
        isinstance(node, ast.Constant)
        and isinstance(node.value, int)
        and not isinstance(node.value, bool)
    )


def python_code_read_only(code: str) -> bool:
    """Can ``python -c CODE`` be PROVEN to write, delete and run nothing?

    WHY (incident 2026-11-13, lop-remote-node-2 0.102.2): a parent printing
    FAILED/passed lines from its child coder's logs with a ``pathlib``
    one-liner was refused as dangerous-delete — the backstop treats every
    path a non-read-only program mentions as a possible write, and nothing
    told a print loop from ``shutil.rmtree``.

    CONSTRAINT: guardrails only get more precise, never looser for a real
    delete, so this proves, it does not guess — by CONSTRUCTION (see
    ``_PY_ALLOWED_NODES``): every node type, import, name, attribute, keyword
    and subscript must be on an allow-list. Consequences the lists alone do
    not express, each enforced below:

    - imports only as unconditional top-level statements (direct children
      of the module): static analysis cannot tell which branch runs, so an
      ``import re as exec`` under ``if 0:`` (or in a loop that never
      iterates, a function, a comprehension) would be counted as imported
      while at run time the name is still the real builtin (review r3 M4 —
      real files deleted);
    - no binding of ANY name in ``dir(builtins)`` — assignment, loop or
      comprehension target, import alias — not only the allowed ones, and
      no binding of an imported name; so a callee name is either an allowed
      builtin that is never rebound or a top-level import, whatever branch
      runs;
    - a ``Name`` is loaded only if it is an allowed builtin, a top-level
      import, or bound by this program;
    - a call's callee is an allowed builtin, an imported name, or an allowed
      attribute — never a local variable or a subscript, so a callable taken
      as a value (``s = re.sub``) cannot dodge the per-call checks;
    - no ``*``/``**`` anywhere; keywords only from ``_PY_ALLOWED_KEYWORDS``;
    - subscripts only by a literal int or int slice (``b["getattr"]`` is out);
    - builtin ``open`` only as a callee with a literal read mode;
    - ``sub``/``subn`` only with literal leading arguments, so the
      replacement is a string, never a callable;
    - no dunder or ``_``-prefixed name or alias.

    Anything else — a parse error, an unlisted node or name — is ``False``
    and keeps the conservative ranking (a judge call, never a LOW).
    """
    import ast
    import builtins

    try:
        tree = ast.parse(code)
    except (SyntaxError, ValueError):
        return False
    nodes = list(ast.walk(tree))
    if any(type(n).__name__ not in _PY_ALLOWED_NODES for n in nodes):
        return False
    # Only an unconditional top-level import is known to have run before a
    # later call by its name; anything nested may never execute.
    top_level = {id(n) for n in tree.body}
    imported: set[str] = set()
    bound: set[str] = set()
    for node in nodes:
        if isinstance(node, (ast.Import, ast.ImportFrom)) and id(node) not in top_level:
            return False
        if isinstance(node, ast.Import):
            for a in node.names:
                if a.name not in _PY_READ_MODULES:
                    return False
                imported.add(a.asname or a.name)
        elif isinstance(node, ast.ImportFrom):
            if node.level or node.module not in _PY_READ_MODULES:
                return False
            for a in node.names:
                if a.name not in _PY_ALLOWED_ATTRS:
                    return False  # also ``*``
                imported.add(a.asname or a.name)
        elif isinstance(node, ast.Name) and isinstance(node.ctx, ast.Store):
            bound.add(node.id)
    if any(n.startswith("_") for n in imported | bound):
        return False
    # Rebinding a builtin or an import would let a later call by that name
    # run whatever was bound; an import shadowing a builtin likewise. Every
    # builtin name, not just the allowed ones: an alias that shadows ``exec``
    # is exactly how a denied builtin would sneak back in as a callee.
    builtin_names = frozenset(dir(builtins)) | _PY_ALLOWED_BUILTINS
    if bound & (imported | builtin_names) or imported & builtin_names:
        return False
    callees = {id(n.func) for n in nodes if isinstance(n, ast.Call)}
    known = _PY_ALLOWED_BUILTINS | imported | bound
    for node in nodes:
        if isinstance(node, ast.Name):
            if node.id not in known:
                return False
            if node.id == "open" and id(node) not in callees:
                return False  # ``o = open`` can be called later with any mode
        elif isinstance(node, ast.Attribute):
            if node.attr not in _PY_ALLOWED_ATTRS or not isinstance(node.ctx, ast.Load):
                return False
        elif isinstance(node, ast.Subscript):
            if not isinstance(node.ctx, ast.Load):
                return False
            index = node.slice
            if isinstance(index, ast.Slice):
                parts = (index.lower, index.upper, index.step)
                if any(p is not None and not _py_small_int(p) for p in parts):
                    return False
            elif not _py_small_int(index):
                return False
        elif isinstance(node, ast.Dict):
            if any(k is None for k in node.keys):
                return False  # ``{**d}``
        elif isinstance(node, ast.comprehension):
            if node.is_async:
                return False
        elif isinstance(node, ast.FormattedValue):
            pass  # its spec is a JoinedStr whose parts this walk checks too
        elif isinstance(node, ast.Call):
            func = node.func
            if isinstance(func, ast.Name):
                if func.id not in _PY_ALLOWED_BUILTINS and func.id not in imported:
                    return False
                callee = func.id
            elif isinstance(func, ast.Attribute):
                callee = func.attr
            else:
                return False
            if any(kw.arg not in _PY_ALLOWED_KEYWORDS for kw in node.keywords):
                return False  # also ``**kw`` (``arg is None``)
            if callee == "open" and isinstance(func, ast.Name):
                if not _py_open_read_only(node):
                    return False
            elif any(kw.arg == "mode" for kw in node.keywords):
                return False
            if callee in ("sub", "subn"):
                # ``re.sub(p, repl, s)`` / ``rx.sub(repl, s)``: untyped, so
                # every argument before the subject must be a literal string.
                if not node.args or any(
                    not (isinstance(a, ast.Constant) and isinstance(a.value, str))
                    for a in node.args[:-1]
                ):
                    return False
    return True


#: Interpreter flags that change nothing the read-only proof relies on. Any
#: other flag before ``-c`` (``-X``, ``-W``, ``-m``…) makes it unprovable.
_PY_SAFE_FLAGS = frozenset({"-I", "-E", "-s", "-S", "-B", "-u", "-q", "-P", "-b", "-bb"})
#: Environment that makes ``python`` import or run code from somewhere else
#: at start-up, write a file, or change which interpreter/stdlib runs:
#: ``PYTHONPATH``/``PYTHONHOME``/``PYTHONPLATLIBDIR``/``PYTHONUSERBASE``
#: (where ``site``/stdlib/``sitecustomize`` load from), ``PYTHONSTARTUP``/
#: ``PYTHONINSPECT`` (run a file / go interactive), ``PYTHONWARNINGS``
#: (``error::mod.Cls`` imports ``mod`` at start-up; review r1 m1),
#: ``PYTHONBREAKPOINT`` (imports a callable by name), ``PYTHON_PRESITE``
#: (imports a module before ``site``), ``PYTHONEXECUTABLE`` (macOS
#: ``sys.executable``), ``PYTHONCASEOK`` (case-insensitive import: shadowing),
#: ``PYTHONPYCACHEPREFIX`` (writes bytecode there), ``PYTHONIOENCODING``
#: (codec looked up by name), ``PYTHONSAFEPATH`` (alters ``sys.path``; harmless
#: when set, refused to keep the proof's assumptions fixed). Harmless knobs an
#: operator commonly exports (``PYTHONUNBUFFERED``, ``PYTHONDONTWRITEBYTECODE``,
#: ``PYTHONUTF8``) are deliberately not here: refusing them would re-break the
#: incident read for no safety gain.
_PY_STARTUP_ENV = (
    "PYTHONPATH",
    "PYTHONHOME",
    "PYTHONPLATLIBDIR",
    "PYTHONUSERBASE",
    "PYTHONSTARTUP",
    "PYTHONINSPECT",
    "PYTHONWARNINGS",
    "PYTHONBREAKPOINT",
    "PYTHON_PRESITE",
    "PYTHONEXECUTABLE",
    "PYTHONCASEOK",
    "PYTHONPYCACHEPREFIX",
    "PYTHONIOENCODING",
    "PYTHONSAFEPATH",
)
#: A cwd entry ``python -c`` would import before the stdlib (``sys.path[0]``
#: is the cwd): ``pathlib`` importing a planted ``fnmatch.py`` runs it.
_PY_SHADOW_SUFFIXES = (".py", ".pyc", ".so", ".pyd", ".pth")


def _cwd_shadows_imports(cwd: str) -> bool:
    try:
        entries = list(os.scandir(cwd))
    except OSError:
        return True
    for e in entries:
        if e.name.endswith(_PY_SHADOW_SUFFIXES):
            return True
        try:
            if e.is_dir() and os.path.exists(os.path.join(e.path, "__init__.py")):
                return True
        except OSError:
            return True
    return False


def _in_place_edit(name: str, args: list[Expanded]) -> bool:
    """``sed -i``/``-I``/``--in-place``, or perl with ``i`` in a short cluster (``-pi``).

    BSD sed's ``-I`` (review F1 on PR #3906) edits in place exactly like
    ``-i`` (suffix glued or as the next word), so it is a write too; without
    it ``sed -I '' 's/a/b/' <file>`` fell through to an unmodelled mention.
    """
    in_place = "iI" if name in ("sed", "gsed") else "i"
    for w in args:
        t = w.text
        if t.startswith("--in-place"):
            return True
        if t.startswith("-") and not t.startswith("--"):
            # R3-3: ``i`` anywhere in a short cluster (``-Ei``, ``-ni``, ``-pi``)
            # — but only BEFORE the first script letter (R4-2): in
            # ``-e's/in/out/'`` everything after ``e`` is the script text.
            flags, _script = _split_cluster(name, t[1:])
            if any(ch in flags for ch in in_place):
                return True
    return False


#: ``env``'s short options that take a value (GNU ``-u``/``-C``/``-a``, BSD
#: ``-u``/``-P``/``-L``/``-U``), and the long ones. ONE table, owned by
#: ``search_guard`` and read by both ``_peel_wrappers`` and ``_env_split``: a
#: private copy here once let ``-C <dir>`` skip its value in the ``-S`` finder
#: while the peel took ``<dir>`` for the program (QA Q-3-1 on PR #3906).
_ENV_VALUE_LETTERS = _WRAPPER_VALUE_FLAGS["env"]
_ENV_LONG_VALUE_FLAGS = _WRAPPER_OWN_LONG_VALUE_FLAGS["env"]
#: How many ``env -S "env -S '…'"`` layers are spliced before failing closed.
_MAX_ENV_SPLIT_DEPTH = 4


def _env_split_words(text: str) -> str:
    """Rewrite env's ``-S`` word separator ``\\_`` as a space, outside single quotes.

    env splits its ``-S`` value with its own grammar, not the shell's:
    ``env -S 'rm\\_-rf\\_x'`` runs ``rm -rf x``, while ``scan()`` read the
    whole thing as one word named ``rm\\_-rf\\_x`` (review R3-1 on PR #3906).
    Inside double quotes env makes ``\\_`` a literal space, which is the same
    thing once ``scan()`` honours the quotes; inside single quotes it stays
    literal.

    This, plus ``scan()``, only APPROXIMATES env's ``-S`` rules (review R3-2):
    ``\\c`` truncation is not modelled and ``;``/``&&``/``|`` are split as
    shell operators where env passes them on as literal words. Each of those
    over-ranks (more programs and targets than env would run), never under.
    """
    out: list[str] = []
    quote = ""
    i = 0
    while i < len(text):
        ch = text[i]
        if quote == "'":
            if ch == "'":
                quote = ""
            out.append(ch)
            i += 1
            continue
        if ch == "\\" and i + 1 < len(text):
            if text[i + 1] == "_":
                out.append(" ")
            else:
                out.append(text[i : i + 2])
            i += 2
            continue
        if ch in "'\"":
            quote = "" if quote == ch else (quote or ch)
        out.append(ch)
        i += 1
    return "".join(out)


def _env_takes_next_word(t: str) -> bool:
    """Does env option word ``t`` take the NEXT word as its value?

    A short cluster takes it when its first valued letter ends the word
    (``-C``, ``-iC``); anything after that letter is a glued value
    (``-C/tmp``). Long options take it only without ``=``.
    """
    if t.startswith("--"):
        return "=" not in t and t in _ENV_LONG_VALUE_FLAGS
    for k, ch in enumerate(t[1:], start=1):
        if ch in _ENV_VALUE_LETTERS:
            return k == len(t) - 1
    return False


def _env_split(
    args: list[Expanded],
) -> tuple[list[Expanded], Expanded | None, list[Expanded]] | None:
    """Find ``-S``/``--split-string`` in ``env args…``'s option prefix.

    Returns ``(options before it, its value word, words after the value)``, or
    ``None`` when env carries no ``-S``. A missing value is ``None`` in the
    middle slot. Flags clustered before ``S`` (``-iS``) come back as their own
    option word so the caller keeps them.

    WHY (review on PR #3906): the check used to look at the first two words
    only, so ``env -u FOO -S 'rm -rf …'`` hid its command string behind a
    valued flag. Every option word up to the first operand or ``--`` is
    scanned, valued flags skip their value, and ``S`` anywhere in a short
    cluster counts; the rest of the cluster (``-S'rm -rf ~'``) is the value.
    """
    i = 0
    while i < len(args):
        t = args[i].text
        start = i
        i += 1
        if t == "--" or t == "-" or not t.startswith("-"):
            return None
        if t.startswith("--"):
            key, eq, value = t.partition("=")
            if key == "--split-string":
                if eq:
                    return args[:start], Expanded(text=value, opaque=args[start].opaque), args[i:]
                value_word = args[i] if i < len(args) else None
                return args[:start], value_word, args[i + 1 :]
            if key in _ENV_LONG_VALUE_FLAGS and not eq:
                i += 1
            continue
        for k, ch in enumerate(t[1:], start=1):
            if ch == "S":
                before = list(args[:start])
                if k > 1:
                    before.append(Expanded(text=t[:k]))
                if k < len(t) - 1:
                    glued = Expanded(text=t[k + 1 :], opaque=args[start].opaque)
                    return before, glued, args[i:]
                value_word = args[i] if i < len(args) else None
                return before, value_word, args[i + 1 :]
            if ch in _ENV_VALUE_LETTERS:
                if _env_takes_next_word(t):
                    i += 1  # the value is the next word
                break  # otherwise the rest of the word is the value
    return None


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


def _sed_quiet(args: list[Expanded]) -> bool:
    """Whether a sed command line asks for ``-n``/``--quiet``/``--silent``.

    Review M1 (PR #3816): under GNU sed, ``-n -i ''`` runs the empty script
    ``''`` with output suppressed and so truncates the next word's file in
    place; the BSD ``-i ''`` reading must not apply then. Scans every option
    word, a cluster up to its first script letter (``-ni``, ``-in``), and
    stops at ``--``. A script word that merely looks like ``-n`` only makes
    the reading stricter.
    """
    for w in args:
        t = w.text
        if t == "--":
            return False
        if t in ("--quiet", "--silent"):
            return True
        if t.startswith("-") and not t.startswith("--") and len(t) > 1:
            if "n" in _split_cluster("sed", t[1:])[0]:
                return True
    return False


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


#: A word that changes the shell's directory, or how a later ``cd`` resolves,
#: anywhere in a ``for`` body (review round 2 m1 on PR #3816). Quote- and
#: context-blind on purpose: a match only makes a relative glob pin opaque,
#: so an ``echo cd`` over-blocking is the safe direction.
_BODY_MOVES_RE = re.compile(
    r"(?<![\w./~$-])(?:cd|pushd|popd)(?![\w./-])"
    r"|(?<![\w./~$-])set(?:\s+\S+)*?\s+(?:[-+][A-Za-z]*P|[-+]o\b|\$)"
)
_LOOP_OPENERS = ("for", "while", "until", "select")
#: Words that may sit in front of a loop keyword or ``done`` in command
#: position (review round 3 m1). ``time`` takes an optional ``-p``.
_LOOP_LEAD_WORDS = ("time", "!", "if", "then", "else", "elif", "do")


def _command_word(text: str) -> str:
    """The word in command position of a piece, past reserved lead words.

    Strips leading ``time [-p]``, ``!``, ``if``, ``then``, ``else``, ``elif``,
    ``do`` and group openers ``{``/``(`` (not ``((``, arithmetic), and trailing
    ``)``/``}`` glued to the word (``done)`` closes a loop in a subshell).
    """
    head = text.strip()
    while head:
        word = head.split(None, 1)[0]
        if word in _LOOP_LEAD_WORDS:
            head = head[len(word) :].lstrip()
            if word == "time" and head.split(None, 1)[:1] == ["-p"]:
                head = head[2:].lstrip()
        elif head[:1] in ("(", "{") and head[:2] != "((":
            head = head[1:].lstrip()
        else:
            return word.rstrip(")}")
    return ""


def _for_body_moves(pieces: Sequence[tuple[str, bool, str]], index: int) -> bool:
    """Whether the loop opened at ``pieces[index]`` changes directory in its body.

    Review round 2 m1 (PR #3816): the use-site cwd check in
    ``_Walker._for_glob_words`` sees the body ONCE, in order, with subshell
    state restored, so two shapes real bash runs against the state root slip
    past it: a ``cd`` AFTER the use (``do : > $f; cd ~/.local-operator;
    done`` re-anchors ``$f`` from the second iteration on) and a ``cd`` in a
    subshell (``do (cd ~/.local-operator && sed -i … $f)``). Rather than
    model iterations and subshells, a relative glob is pinned only when the
    body, nested loops and subshells included, holds no ``cd``/``pushd``/
    ``popd``/``set -P`` at all.

    The extent runs to the matching ``done`` (nested loops counted). Review
    round 3 m1: ``done`` and the loop openers count only in COMMAND position
    (:func:`_command_word`) — ``echo done`` is an argument, and a nested
    ``time while …`` / ``then while …`` is a loop whose ``done`` must not
    close this one. Either mistake ended the scan before a later ``cd``.
    When no matching ``done`` is found the extent is unknown, so this fails
    closed (``True``: do not pin).
    """
    depth = 0
    body: list[str] = []
    for text, _flag, _op in pieces[index:]:
        word = _command_word(text)
        if word in _LOOP_OPENERS:
            depth += 1
        elif word == "done":
            depth -= 1
        if not body and depth != 1:
            return True  # this piece does not open a loop as parsed: fail closed
        # The header (``for f in <words>``) is excluded: a list word is data.
        body.append(text if body else "")
        if depth <= 0:
            return bool(_BODY_MOVES_RE.search("\n".join(body[1:])))
    return True


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
    physical: bool = False


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
        #: ``PATH`` as the command inherits it: ``_inline_read_only`` trusts
        #: a bare ``python3`` only while this command has not changed it.
        self.inherited_path = env.get("PATH")
        self.result = Resolution()
        #: Non-read-only programs an earlier part of THIS command runs (each
        #: reaches ``_backstop``). Any of them could plant a module in the cwd
        #: before a later ``python -c``, so ``_inline_read_only`` holds only
        #: for the first one (see there).
        self.programs_run = 0
        #: Paths an earlier segment of THIS command made into a symlink (S7):
        #: what they point at is resolved only at run time.
        self.linked: set[str] = set()
        #: Every path an earlier part of THIS command deleted, moved, copied or
        #: wrote to, or created (lexical and resolved spellings, no trailing
        #: ``/``). Any of them may be a symlink by the time a later word's
        #: ``..`` is looked up (``rm -rf <pad>/d && mv <link> <pad>/d``), so a
        #: ``..`` after one is unpinned (review R1-1, PR #3811); see
        #: :meth:`_unpin_relinked`. ``linked`` is the ``ln -s`` half of the same
        #: fact and stays separate because S7 also uses it on its own.
        self.named: set[str] = set()
        #: The ``ln -s``/``cp -s`` links an earlier part of THIS command made
        #: with a LITERAL target: link path (as :func:`_relinked_landing`
        #: spells it) -> target as written. Any later naming of the path drops
        #: its entry (:meth:`_name`), so only the latest known link is used.
        self.link_targets: dict[str, str] = {}
        #: :func:`_relinked_landing` memo for the current ``link_targets``;
        #: cleared on every change to it (:meth:`_set_link`, :meth:`_name`).
        self._landing_memo: dict[str, str] = {}
        #: ``for VAR in a b c`` literal lists: a target using $VAR fans out.
        self.for_lists: dict[str, list[str]] = {}
        #: The loop variable each ``for_lists`` entry derives from (itself for
        #: the loop variable; ``VAR`` for ``D=…/$VAR`` inside the body), so one
        #: iteration rewrites every related reference together.
        self.for_roots: dict[str, str] = {}
        #: ``for VAR in <glob> …`` bindings, by the key inside the variable's
        #: ``FOR_GLOB_MARK`` value: one ``(absolute text, absolute pattern)``
        #: per list word (``pattern`` is ``None`` for a literal word), plus the
        #: cwd of the ``for`` itself and whether every word was absolute. Bash
        #: expands a relative glob to RELATIVE words, so the pin holds only
        #: where the cwd still equals the loop's (review B1, PR #3816).
        self.for_globs: dict[str, tuple[list[tuple[str, str | None]], str | None, bool]] = {}
        #: Whether the ``for`` being walked changes directory anywhere in its
        #: body; set per piece by ``walk`` from :func:`_for_body_moves`.
        self.for_body_moves = False

    # -- state isolation (R4) ----------------------------------------------------
    def _save(self) -> _Snapshot:
        st = self.state
        return _Snapshot(dict(st.env), st.cwd, st.home, st.private_cwd, st.physical)

    def _restore(self, snap: _Snapshot) -> None:
        self.state.env, self.state.cwd, self.state.home = snap.env, snap.cwd, snap.home
        self.state.private_cwd = snap.private_cwd
        self.state.physical = snap.physical

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
            # Read by ``_for`` when this piece opens a ``for`` (review round 2 m1).
            self.for_body_moves = _for_body_moves(pieces, index)
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
    def _splice_env_split(
        self,
        words: list[Expanded],
        split: tuple[list[Expanded], Expanded | None, list[Expanded]],
    ) -> list[Expanded]:
        """Replace ``env``'s ``-S`` value with the words env will run.

        Three rules, all from review R3-1 on PR #3906:

        * The raw ``-S`` text is scanned for a delete word FIRST, every layer,
          whatever the splice makes of it. That is round 2's floor: a splice
          that mis-splits (env's grammar is not the shell's) can lower a
          mention, never this reason.
        * env's own separator ``\\_`` is honoured (``_env_split_words``).
        * A spliced ``env … -S …`` is spliced again, up to
          ``_MAX_ENV_SPLIT_DEPTH`` layers; deeper nesting fails closed as an
          opaque command string, since a bounded walk is the only kind the
          model cannot stretch.

        ``$VAR``/``${VAR}`` and ``~`` are expanded shell-style: env expands
        ``${VAR}`` itself and leaves ``~`` alone, so this pins nothing it should
        not and can only over-rank.
        """
        head = [words[0]]
        inner = words[1:]
        for _depth in range(_MAX_ENV_SPLIT_DEPTH):
            before, value, after = split
            raw = value.text if value is not None else " ".join(w.text for w in inner)
            if _DELETE_WORD_RE.search(raw):
                self.result.opaque_reasons.append("env -S runs a command string")
            if value is None or value.opaque:
                # Unknowable value: fall through to the peel and the backstop.
                return [*head, *inner]
            split_words, split_redirects = scan(_env_split_words(value.text))
            spliced: list[Expanded] = []
            for raw_word in split_words:
                spliced.extend(expand_all(raw_word, self.state.env, self.state.home))
            # env passes ``>``/``|`` through as literal words; keep the
            # operands visible rather than drop them.
            for _op, raw_word in split_redirects:
                spliced.extend(expand_all(raw_word, self.state.env, self.state.home))
            head = [*head, *before]
            inner = [*spliced, *after]
            nested = (
                _env_split(inner[1:])
                if inner and os.path.basename(inner[0].text) == "env"
                else None
            )
            if nested is None:
                return [*head, *inner]
            head = [*head, inner[0]]
            inner = inner[1:]
            split = nested
        self.result.opaque_reasons.append("env -S nested too deep to resolve")
        return [*head, *inner]

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
        assign_words: list[Expanded] = []
        while words and _ASSIGN_RE.match(words[0].text):
            text_value = words[0].text
            if words[0].temp and text_value.count(OPAQUE_MARK) == 1:
                text_value = text_value.replace(OPAQUE_MARK, TEMP_MARK)
            raw_assigns.append(text_value)
            # Same length (one marker char for another): the refs still line up.
            assign_words.append(
                Expanded(text=text_value, refs=list(words[0].refs), pattern=words[0].pattern)
            )
            words = words[1:]
        if not words:
            # A bare assignment segment sets a shell variable for what follows.
            for w in assign_words:
                self._assign_word(w)
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
        if name == "set":
            self._set_options(words[1:])
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
        split = _env_split(words[1:]) if name == "env" else None
        if split is not None:
            # S6 / review R2-1 on PR #3906: ``env -S 'cmd …'`` splits its value
            # into words and runs them as if they had been env's own argv, so
            # the split words are spliced in place of ``-S`` and the segment
            # carries on as a plain ``env``: options and assignments around it
            # still peel, and the real program's targets and mentions are
            # ranked. Returning early here once dropped non-delete commands
            # (``env -i -u A -S 'node x <stranger pad>'``) to LOW. Tilde and
            # ``${VAR}`` are expanded shell-style, which can only over-rank.
            words = self._splice_env_split(words, split)
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
            exempt = self._touch(name, args) | _read_inputs(name, args)
            # The program word too: after ``env``/``xargs`` peeling a path can
            # BE the program (``xargs -Q <path>``).
            executes = name in ("sed", "gsed") and _sed_executes(args)
            self._backstop(
                name, unpeeled, raw_assigns, exempt, executes, len(unpeeled) - len(words)
            )
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
            operands, flags = _operands(args, _MV_VALUE_FLAGS)
            if len(operands) >= 2:
                dest = operands[-1]
                if name != "install" and _no_clobber_new(dest, flags, self.state.cwd):
                    # ``cp -n src <new path>`` replaces nothing: it CREATES the
                    # destination (an append, ranked only where creation itself
                    # matters — evidence, another session's pad). The sources
                    # are reads either way.
                    self._add(dest, "append", name)
                else:
                    self._add_overwrite_dest(dest, name, operands[:-1], flags)
                if (
                    name != "install"
                    and len(operands) == 2
                    and (_has_short(flags, "s") or "--symbolic-link" in flags)
                ):
                    # ``cp -s SRC DEST`` makes DEST a symlink to SRC as written.
                    self._record_link(dest, operands[0], flags)
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
        #: Words only GNU sed's reading of ``-i ''`` makes files (review M1).
        gnu_files: list[Expanded] = []
        script_by_flag = False
        quiet = name == "sed" and _sed_quiet(args)
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
                if (
                    # Review M1 (PR #3816): never ``gsed``, which is always GNU,
                    # and never with ``-n``/``--quiet``/``--silent``: GNU's
                    # ``-i`` takes only an ATTACHED suffix, so there ``''`` is
                    # the script, the next word a file, and ``-n`` with an empty
                    # script truncates it (``gsed -n -i '' config.yml``).
                    name == "sed"
                    and not quiet
                    and _flags.endswith("i")
                    and k + 1 < len(args)
                    and args[k + 1].text == ""
                    and not args[k + 1].opaque
                    and not args[k + 1].empty
                ):
                    # BSD sed (macOS ``sed``): ``-i ''`` is ``-i`` with an EMPTY
                    # backup suffix, so the '' is not the script and the word
                    # after it is not a file. Read as GNU instead, '' would be
                    # the script, which edits nothing: either way the next word
                    # is never rewritten, so it is never a target. Before,
                    # ``sed -i '' '/"$k"/d' f`` ranked the SCRIPT as a path (an
                    # unset ``$k`` made it an "expanded empty" HIGH), and with
                    # ``-e`` the '' itself resolved to the cwd as an overwrite.
                    # Only a literal '' qualifies: ``-i "$SUF"`` may hold a name.
                    # ``sed`` is GNU on Linux, though, where the next word IS a
                    # file: rank it under that reading too, max of both. It is
                    # ``critical_only`` because without ``-n`` GNU rewrites it
                    # unchanged, so only a protected path (where even that is
                    # forbidden) should move the verdict; a script word read
                    # as a path must not re-create the HIGH this skip removed.
                    if k + 2 < len(args) and not args[k + 2].text.startswith("-"):
                        gnu_files.append(args[k + 2])
                    k += 2
                    continue
                # R2-2: BSD ``-I`` ending a cluster takes the next word as its
                # suffix (``-I .bak``, ``-EI ''``); it is not the script.
                k += 2 if name in ("sed", "gsed") and t.endswith("I") else 1
                continue
            if t.startswith("--"):
                k += 1
                continue
            files.append(args[k])
            k += 1
        chosen = files if script_by_flag else files[1:]
        for w in chosen:
            self._add(w, "overwrite", name)
        chosen_ids = {id(w) for w in chosen}
        for w in gnu_files:
            if id(w) in chosen_ids:
                continue  # both readings already agree it is a file
            before = len(self.result.targets)
            self._add(w, "overwrite", name)
            for target in self.result.targets[before:]:
                target.critical_only = True
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

    def _set_options(self, args: list[Expanded]) -> None:
        """Track ``set -P``/``set +P``/``set -o physical``: physical mode
        changes where a later ``cd a/..`` lands (review R2-MINOR-2)."""
        st = self.state
        i = 0
        while i < len(args):
            text = args[i].text
            if args[i].opaque:
                # ``set $X``: may switch physical mode on; assume the worse.
                st.physical = True
            elif text in ("-o", "+o") and i + 1 < len(args):
                if args[i + 1].opaque or args[i + 1].text == "physical":
                    st.physical = text == "-o"
                i += 1
            elif text == "--" or not text.startswith(("-", "+")):
                break
            elif "P" in text[1:]:
                st.physical = text[0] == "-"
            i += 1

    def _cd(self, args: list[Expanded]) -> None:
        st = self.state
        # ``cd -P`` (or ``set -P``) resolves ``..`` AFTER following links, so
        # ``cd -P <pad>/sub/..`` with ``sub`` -> a session dir lands in
        # ``sessions/``, not the pad (review R2-MINOR-2). ``-L``/``-P``: the
        # last one wins, as in bash.
        physical = st.physical
        for w in args:
            if w.text == "--" or not w.text.startswith("-") or len(w.text) == 1:
                break
            for flag in w.text[1:]:
                if flag in "LP":
                    physical = flag == "P"
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
            elif physical:
                # Where the kernel lands it, links resolved: physical mode also
                # sets ``$PWD`` to that, so a later logical ``cd ..`` climbs
                # from the real directory. ``None`` (unknown cwd, which ranks
                # later relative words HIGH) when it cannot be pinned.
                raw = _joined(w.text, st.cwd)
                path = kernel_path(raw) if raw is not None else None
                # A component an earlier part of this command re-linked is
                # followed at RUN time, not as it is now (review round 2,
                # PR #3811): unknown cwd, so later relative words rank HIGH.
                if raw is not None and self._names_any(
                    {os.path.normpath(raw[: m.start()] or "/") for m in re.finditer(r"/|$", raw)}
                ):
                    path = None
            else:
                # Logical: lexical, as the shell itself canonicalises it.
                path = _logical(w.text, st.cwd)
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
        if (
            st.cwd is not None
            and any(v.pattern is not None for v in values)
            and len(values) <= _MAX_FOR_COMBINATIONS
            and all(not v.opaque and not v.empty and CREDENTIAL_MARK not in v.text for v in values)
        ):
            # FIELD CASE (2026-11-11, session a85e507ffdfa): ``for f in
            # src/i18n/catalogues/*/ui.teamBridge.json; do sed -i '' '/…/d' $f;
            # done`` in the session's own worktree bound ``$f`` to OPAQUE_MARK,
            # so the edit's target collapsed onto the worktree ROOT and went
            # to the judge as an opaque delete there. Bash iterates the glob's
            # matches (or the pattern itself when nothing matches), so
            # ``CMD $f`` inside the body touches exactly what ``CMD <glob>``
            # would: the target IS that glob, ranked by the existing glob rules
            # (CRITICAL when it reaches a protected name, LOW for an ordinary
            # match set). Resolved against the cwd of the ``for`` itself, as
            # bash expands it there. But bash binds ``$f`` to the RELATIVE match
            # (``config.yml``), so a ``cd``/``pushd`` in the body re-anchors it:
            # ``for f in *.yml; do cd ~/.local-operator && sed -i … $f`` edits
            # the state root's config.yml (review B1, reproduced in real bash).
            # The loop's cwd is recorded so ``_for_glob_words`` pins only where
            # the use site's cwd still equals it, or every word was absolute.
            key = str(len(self.for_globs))
            entries: list[tuple[str, str | None]] = []
            for v in values:
                text = v.text if v.text.startswith("/") else os.path.join(st.cwd, v.text)
                pattern = None
                if v.pattern is not None:
                    pattern = (
                        v.pattern
                        if v.pattern.startswith("/")
                        else os.path.join(_glob_escape(st.cwd), v.pattern)
                    )
                entries.append((text, pattern))
            # A relative word is never pinned when the body moves directory
            # ANYWHERE (review round 2 m1): ``absolute`` is then the only way
            # to pin, and a relative glob stays opaque at every use site.
            absolute = all(v.text.startswith("/") for v in values)
            if not absolute and self.for_body_moves:
                entries_cwd: str | None = None
            else:
                entries_cwd = st.cwd
            self.for_globs[key] = (entries, entries_cwd, absolute)
            st.env[var] = FOR_GLOB_MARK + key + OPAQUE_MARK
            return
        if any(v.opaque or v.pattern is not None for v in values):
            st.env[var] = (
                CREDENTIAL_MARK if any(CREDENTIAL_MARK in v.text for v in values) else OPAQUE_MARK
            )
            return
        if len(values) > _MAX_FOR_COMBINATIONS:
            # Review R1-B1 (PR #3710): a list too long to enumerate is
            # unknowable, exactly as ``MAX_BRACE_RESULTS`` treats a brace
            # list. Truncating it instead hid ``~`` at position 1101 and
            # ranked ``rm -rf $d`` LOW.
            st.env[var] = (
                CREDENTIAL_MARK if any(CREDENTIAL_MARK in v.text for v in values) else OPAQUE_MARK
            )
            return
        # The body runs once per value; the walker sees it once, so the variable
        # carries EVERY value and ``_add`` fans a target out per value.
        st.env[var] = values[0].text
        self.for_lists[var] = [v.text for v in values]
        self.for_roots[var] = var

    def _ln(self, args: list[Expanded]) -> None:
        operands, flags = _operands(args, frozenset({"-t", "--target-directory"}))
        if not (_has_short(flags, "s") or "--symbolic" in flags) or len(operands) < 2:
            return
        link = operands[-1]
        path, _ = _absolute(link.text, self.state.cwd)
        if path is not None:
            self.linked.add(path.rstrip("/"))
        # Provenance (q1029 incident, remote node 0.101.44): a session that
        # made ``ln -s <pad> ~/q1029`` could not remove that link later. Only
        # the plain two-operand form names the link itself; ``-t DIR`` or
        # several sources put links INSIDE a directory, and a trailing slash
        # asks ln to place the link inside an existing directory. The ledger
        # re-checks after the call that the link really appeared there
        # (``provenance.commit_creations``), so a dereferenced directory
        # destination records nothing.
        if (
            len(operands) == 2
            and not _cluster_value(args, "t", "--target-directory")
            and not link.text.endswith("/")
        ):
            self._create("symlink", self._literal_dir(link))
        # AFTER the provenance record, never before: ``_create`` calls
        # ``_name``, which drops any ``link_targets`` entry for the path it
        # names. Run second, it erased the target ``_record_link`` had just
        # stored, so ``ln -sfn <secrets> PAD/l && rm -f PAD/l/../config.yml``
        # lost its re-linked landing and fell from CRITICAL to HIGH
        # (convergence of PR #3811 onto #3793). ``_record_link`` names the
        # path itself before storing, so this order invalidates the stale
        # target and keeps the new one.
        if len(operands) == 2:
            self._record_link(link, operands[0], flags)

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
            if target_dir.text.startswith("--target-directory="):
                value = target_dir.text.split("=", 1)[1]
                target_dir = None if target_dir.opaque else Expanded(text=value)
            if target_dir is not None:
                self._add_into_dir(target_dir, "mv", sources)
        else:
            sources = operands[:-1]
            if len(operands) >= 2:
                self._add_overwrite_dest(operands[-1], "mv", sources, flags)
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

    def _inline_read_only(self, name: str, words: list[Expanded], assigns: list[str]) -> bool:
        """Is this exactly ``python[3] [safe flags] -c CODE [argv…]`` with CODE
        proven read-only (:func:`python_code_read_only`)?

        Fail closed on everything the proof cannot see: a wrapper or inline
        assignment in front (``env``, ``PYTHONPATH=…``), start-up environment
        that imports other code, an opaque or substituted CODE word, an
        unknown cwd or one holding importable modules that would shadow the
        stdlib, and any earlier non-read-only program or write in the same
        command (it may plant such a module before python starts). Trailing
        argv words are inert: CODE cannot reach ``sys.argv`` without ``sys``.
        """
        st = self.state
        if not words or not _INTERPRETER_RE.fullmatch(name) or name.startswith("node"):
            return False
        # Pin the interpreter (review r1 M2): ``./python3``, ``/abs/python3``
        # or a ``PATH`` this command rewrote can run a planted executable
        # named ``python3`` that ignores CODE. Only a bare word looked up on
        # the INHERITED ``PATH`` is the interpreter the proof is about.
        if words[0].text != name or "/" in words[0].text or words[0].opaque or assigns:
            return False
        if st.env.get("PATH") != self.inherited_path:
            return False
        if self.programs_run or self.named or self.linked or self.result.targets:
            return False
        if any(t.op != "touch" for t in self.result.touched) or self.result.created:
            return False
        if any(st.env.get(k) for k in _PY_STARTUP_ENV):
            return False
        if st.cwd is None or _cwd_shadows_imports(st.cwd):
            return False
        rest = words[1:]
        while rest and rest[0].text in _PY_SAFE_FLAGS:
            rest = rest[1:]
        if len(rest) < 2 or rest[0].text != "-c":
            return False
        code = rest[1]
        if code.opaque or code.pattern is not None or code.substitutions:
            return False
        if OPAQUE_MARK in code.text or TEMP_MARK in code.text:
            return False
        return python_code_read_only(code.text)

    def _backstop(
        self,
        name: str,
        args: list[Expanded],
        assigns: list[str],
        exempt: set[int],
        executes: bool = False,
        prefix: int = 0,
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
        script = _interpreter_script(name, args, st.cwd)
        read_only_code = self._inline_read_only(name, args, assigns)
        # ``args[prefix]`` is the interpreter word: ``args`` is the UNPEELED
        # argv, so without the slice a wrapper (``env python3 -c …``) hides
        # the ``-c`` and the write earns lineage credit (review m1).
        unproven_code = not read_only_code and _python_inline_code(name, args[prefix:])
        self.programs_run += 1
        outputs = _output_word_ids(args)
        words = [w for w in args if id(w) not in exempt]
        # ``prefix``: how many leading words are wrappers (``env -u X A=…``)
        # before the program, so an ``A=…`` among them is an assignment, as
        # is a shell inline one. Recorded on the Target (``assigned``) so the
        # judge is told the path only arrives as an environment value.
        lead = {id(w) for w in args[:prefix]}
        texts = [
            (
                w.text,
                id(w) in outputs,
                (m.group(1) if id(w) in lead and (m := _ASSIGN_RE.match(w.text)) else None),
            )
            for w in words
        ] + [(a.split("=", 1)[1], False, a.split("=", 1)[0]) for a in assigns if "=" in a]
        for text, as_output, assigned in texts:
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
                        Target(
                            path=path,
                            op="touch",
                            program=name or "a program",
                            script=script,
                            read_only_code=read_only_code and not as_output,
                            unproven_code=unproven_code,
                            as_output=as_output,
                            executes=executes,
                            assigned=assigned,
                        )
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
            self._name(path)

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
                        target.force_count = sum(
                            1 if f == "--force" else f[1:].count("f")
                            for f in flags
                            if f.startswith("-")
                        )
                finally:
                    self.state.cwd = saved

    # -- recording ---------------------------------------------------------------
    def _add(self, exp: Expanded, op: str, program: str, *, selected: bool = False) -> None:
        if exp.vanished and not exp.empty:
            return
        pinned = self._for_glob_words(exp) if op in ("overwrite", "append") else None
        if pinned is not None:
            # ``$f`` from ``for f in <glob>``: one target per list word, each
            # the glob (or literal) bash iterates — never the opaque cwd.
            # WRITES ONLY (in-place edits and redirections here; a copy's
            # overwritten destination is folded the same way in
            # ``_add_overwrite_dest``):
            # a delete/move keeps the opaque reading, because a glob DELETE
            # ranks by rules that are looser than the opaque judge for some
            # shapes (``rm -rf .worktrees/*`` ranks LOW today, foreign trees
            # included), and this fix must not widen what a loop may remove.
            for word in pinned:
                self._add(word, op, program, selected=selected)
            return
        target = _target(exp, op, program, self.state)
        target.selected = selected
        self._unpin_relinked(target, exp.text)
        if target.path is not None:
            stripped = target.path.rstrip("/") or "/"
            if any(stripped == ln or stripped.startswith(ln + "/") for ln in self.linked):
                # S7: a symlink made earlier in this same command — its pointee
                # is decided at run time.
                target.opaque = True
                target.opaque_contained = False
        self.result.targets.append(target)
        # R6: a literal ``for`` list binds its variable to each value in turn.
        # Only the spans the word's own ``$VAR`` references filled are
        # rewritten (``Expanded.refs``); a target that never names the loop
        # variable is recorded once, as written.
        iterations = self._iterations(exp)
        if iterations is None:
            # Too many combinations to enumerate: the word is unknowable,
            # never truncated (review R1-B1).
            target.opaque = True
            target.opaque_contained = False
            return
        for _roots, values in iterations:
            fanned = _fan_out(exp, values)
            extra = _target(fanned, op, program, self.state)
            extra.selected = selected
            self._unpin_relinked(extra, fanned.text)
            self.result.targets.append(extra)

    def _unpin_relinked(self, target: Target, text: str) -> None:
        """Mark ``target`` unpinned when a component before one of its ``..``
        was named earlier in this command, then remember ``target`` itself.

        WHY. ``physical_path`` resolves ``..`` against the filesystem as it is
        at CHECK time, but the command runs its parts in order: in
        ``ln -sfn ~/.local-operator/secrets <pad>/lnk && rm -f <pad>/lnk/../config.yml``
        the ``rm`` follows the NEW link, so the kernel deletes the state root's
        ``config.yml`` while the check-time lookup (``lnk`` -> ``<pad>/build``)
        reads ``<pad>/config.yml`` — LOW (review R1-1, PR #3811). The same goes
        for a path an earlier part deleted (and so may recreate), moved or
        copied onto, or created. Where the new link will point is often
        knowable, but not in general (``mv``, ``cp -a`` of a link, variables),
        so the target is ``unpinned``: ``classify`` keeps any CRITICAL its
        lexical or check-time reading earns and otherwise ranks it HIGH, never
        LOW. Only a component BEFORE a ``..`` matters — without a ``..`` the
        path is named through the link itself and S7 (``_add``) covers it.
        """
        raw = _joined(text, self.state.cwd)
        if raw and (self.linked or self.named) and ".." in raw.split("/"):
            parts = raw.split("/")
            last = max(k for k, part in enumerate(parts) if part == "..")
            prefixes = {
                os.path.normpath("/" + "/".join(parts[:k]).lstrip("/"))
                for k in range(1, last + 1)
                if parts[k - 1] not in ("", ".", "..")
            }
            if self._names_any(prefixes):
                target.unpinned = True
        if target.unpinned and target.raw_path is not None:
            landing = self._landing(target.raw_path)
            if landing is not None and landing != target.path:
                target.landings = (landing,)
        for spelled in (target.path, os.path.normpath(raw) if raw else None):
            if spelled:
                self._name(spelled)

    def _name(self, path: str) -> None:
        """Record that an earlier part of this command named ``path`` (see
        ``named``); it may no longer be the link ``link_targets`` recorded."""
        path = path.rstrip("/") or "/"
        self.named.add(path)
        if not self.link_targets:
            # Nothing to invalidate: skip the walk, which only maps ``path``
            # to its ``link_targets`` key (review R3-2, PR #3811).
            return
        key = self._landing(path)
        for stale in (path, key):
            if stale is not None and self.link_targets.pop(stale, None) is not None:
                self._landing_memo.clear()

    def _landing(self, path: str, *, follow_last: bool = False) -> str | None:
        """:func:`_relinked_landing` against this command's ``link_targets``,
        memoised until they change."""
        return _relinked_landing(
            path, self.link_targets, follow_last=follow_last, memo=self._landing_memo
        )

    def _set_link(self, key: str, pointee: str) -> None:
        self.link_targets[key] = pointee
        self._landing_memo.clear()

    def _names_any(self, prefixes: set[str]) -> bool:
        """Whether an earlier part of this command named (or linked) any of
        ``prefixes``, compared by spelling AND by physical location.

        WHY. After ``cd -P`` the cwd is the realpath (``/private/var/…`` on
        macOS), while a path named before it kept the spelling it was written
        with (``/var/…``): ``ln -sfn <state> <pad>/lnk && cd -P <pad> &&
        cd -P lnk`` missed the re-link and ranked a later delete LOW (review
        R3-3, PR #3811). The final component is kept as spelled — resolving it
        would follow the very link in question.
        """
        named = self.linked | self.named
        if not named or not prefixes:
            return False
        if prefixes & named:
            return True

        def physical(p: str) -> str:
            head, tail = os.path.split(p)
            try:
                return os.path.join(os.path.realpath(head), tail)
            except OSError:
                return p

        return bool({physical(p) for p in prefixes} & {physical(n) for n in named})

    def _record_link(self, link: Expanded, pointee: Expanded, flags: list[str]) -> None:
        """Remember ``link -> pointee`` for :func:`_relinked_landing` when both
        words are literal and ``link`` itself becomes the link: not ``-t``, and
        not an existing directory unless ``-n``/``-T`` (otherwise the link is
        made INSIDE it). Anything else stays unknown, which ranks HIGH."""
        path, _ = _absolute(link.text, self.state.cwd)
        if path is None:
            return
        self._name(path)
        literal = all(
            not w.opaque and w.pattern is None and not w.empty and w.text for w in (link, pointee)
        )
        no_deref = _has_short(flags, "nT") or bool(
            {"--no-dereference", "--no-target-directory"} & set(flags)
        )
        if not literal or any(f.startswith(("-t", "--target-directory")) for f in flags):
            return
        if os.path.isdir(path) and not no_deref:
            return
        key = self._landing(path)
        if key is not None:
            self._set_link(key, pointee.text)

    def _link_state(self, src: Expanded) -> tuple[bool, str | None]:
        """``(known, link target)`` of what ``mv``/``cp`` of ``src`` puts at
        its destination: a link to ``target``, or (``target`` ``None``) not a
        link. Unknown for a non-literal source, one missing now (something
        unmodelled may create it), or one this command already changed in a
        way ``link_targets`` does not record.

        ``cp`` without ``-P``/``-a``/``-R`` copies what a link points at, not
        the link; treating it as a link anyway only ever reads the riskier
        landing (a copied link-to-directory without ``-R`` fails anyway).
        """
        if src.opaque or src.pattern is not None or src.empty or not src.text:
            return False, None
        path, _ = _absolute(src.text, self.state.cwd)
        if path is None:
            return False, None
        path = path.rstrip("/") or "/"
        key = self._landing(path)
        if key is None:
            return False, None
        if key in self.link_targets:
            return True, self.link_targets[key]
        if self._names_any({path, key}):
            return False, None
        try:
            if os.path.islink(key):
                return True, os.readlink(key)
            return os.path.lexists(key), None
        except OSError:
            return False, None

    def _carry_link(self, dest: str, state: tuple[bool, str | None]) -> None:
        """Record at ``dest`` the link ``state`` a move/copy put there.

        WHY (review R3-1, PR #3811). Naming the destination dropped any link
        it held, so ``ln -sfn <state>/secrets <pad>/a && mv <pad>/a <pad>/b &&
        rm -f <pad>/b/../config.yml`` read ``b`` as a fresh directory, and the
        refusal named ``<pad>/config.yml`` while rename(2) keeps the link and
        the kernel removes the state root's ``config.yml``. The target string
        moves as written, so a relative one resolves from the new place, as
        the kernel does. An unknown link state records nothing: the ``..``
        after ``dest`` is already unpinned (``dest`` is named), and an unpinned
        ``..`` landing lexically in session state is CRITICAL regardless.
        """
        known, pointee = state
        key = self._landing(dest)
        if known and pointee is not None and key is not None:
            self._set_link(key, pointee)

    def _for_glob_words(self, exp: Expanded) -> list[Expanded] | None:
        """The list words a whole-word ``$VAR`` bound by a glob ``for`` stands for.

        Only when the word is EXACTLY that one value (``$f``, ``"$f"``,
        ``${f}``): a prefix, suffix or second expansion leaves it opaque, the
        fail-closed reading it had before.
        """
        text = exp.text
        if not text.startswith(FOR_GLOB_MARK) or not text.endswith(OPAQUE_MARK):
            return None
        key = text[len(FOR_GLOB_MARK) : -len(OPAQUE_MARK)]
        binding = self.for_globs.get(key)
        if binding is None or exp.substitutions or exp.empty or exp.temp:
            return None
        entries, loop_cwd, absolute = binding
        # ``loop_cwd`` is ``None`` when the body moves directory anywhere
        # (review round 2 m1): a relative glob is then never pinned.
        if not absolute and (loop_cwd is None or self.state.cwd != loop_cwd):
            # Review B1: a relative match moved with a ``cd``/``pushd`` (or an
            # unknowable one, cwd ``None``) in the body; the pin would name
            # the wrong file. The opaque reading is the fail-closed one.
            return None
        return [Expanded(text=t, pattern=p, opaque_contained=False) for t, p in entries]

    def _iterations(self, exp: Expanded) -> list[tuple[tuple[str, ...], dict[str, str]]] | None:
        """``(loop vars, {ref name: value})`` for every iteration after the first.

        ``None`` when the combinations exceed :data:`_MAX_FOR_COMBINATIONS`:
        the caller marks the word opaque. Never a truncated list (review
        R1-B1: values past the cap were silently dropped, hiding ``~``).
        """
        names = {r[0] for r in exp.refs if r[0] in self.for_lists}
        # Variables move in step only when they share a loop AND a list length.
        # A stale binding from an earlier loop of a different length (review
        # R1-M3: ``D=$y`` from a 3-value loop, ``$y`` rebound by a 2-value
        # one) is its own family, so the word takes the cartesian product —
        # a superset of what bash runs, never a dropped value.
        families: dict[tuple[str, int], list[str]] = {}
        for n in sorted(names):
            families.setdefault((self.for_roots[n], len(self.for_lists[n])), []).append(n)
        if not families:
            return []
        total = 1
        for _, length in families:
            total *= length
        if total > _MAX_FOR_COMBINATIONS:
            return None
        # Nested loops run every combination, so the word does too.
        ranges = [range(length) for _, length in families]
        out: list[tuple[tuple[str, ...], dict[str, str]]] = []
        for combo in itertools.islice(itertools.product(*ranges), 1, None):
            values: dict[str, str] = {}
            for family, i in zip(families.values(), combo, strict=True):
                values.update({n: self.for_lists[n][i] for n in family})
            out.append((tuple(root for root, _ in families), values))
        return out

    def _assign_word(self, word: Expanded) -> None:
        """Bind an assignment word, carrying any ``for`` fan-out it inherits.

        ``D=$S/x-$l`` inside a loop body binds D once per iteration, so a
        later ``$D`` target fans out with the loop like a direct ``$l`` one.
        """
        self._assign(word.text)
        name = word.text.partition("=")[0]
        self.for_lists.pop(name, None)
        self.for_roots.pop(name, None)
        if "=" not in word.text or not word.refs:
            return
        iterations = self._iterations(word)
        if iterations is None:
            # Unenumerable: the variable is unknowable (review R1-B1).
            self.state.env[name] = OPAQUE_MARK
            return
        if not iterations:
            return
        # One loop: the variable moves in step with it. Several (nested): it
        # carries every combination as its own list, so a later ``$NAME``
        # still fans out over all of them.
        roots = iterations[0][0]
        self.for_roots[name] = roots[0] if len(roots) == 1 else name
        self.for_lists[name] = [self.state.env.get(name, "")] + [
            _tilde_value(_fan_out(word, values).text.partition("=")[2], self.state.home)
            for _, values in iterations
        ]

    def _add_overwrite_dest(
        self,
        exp: Expanded,
        program: str,
        sources: list[Expanded] | None = None,
        flags: list[str] | None = None,
    ) -> None:
        pinned = self._for_glob_words(exp)
        if pinned is not None:
            # Review m2 (PR #3816): ``for f in ~/.local-operator/*.yml; do cp x
            # $f`` hit the opaque early return below and named NOTHING (LOW),
            # while the direct ``cp x ~/.local-operator/config.yml`` is
            # CRITICAL. The destination is overwritten, so it takes the same
            # pin as a redirection; never looser, since before it named nothing.
            for word in pinned:
                self._add(word, "overwrite", program)
            return
        if exp.opaque or exp.pattern is not None:
            return
        path, _ = _absolute(exp.text, self.state.cwd)
        no_target_dir = _has_short(flags or [], "T") or "--no-target-directory" in (flags or [])
        # A destination this command itself made a link to an existing
        # directory stays that link: the move lands INSIDE it (review R3-1).
        resolved = None if path is None else self._landing(path, follow_last=True)
        into = not no_target_dir and resolved is not None and os.path.isdir(resolved)
        kept_link = (
            into
            and path is not None
            and self._landing(path.rstrip("/") or "/") in self.link_targets
        )
        if path is not None and not kept_link:
            # The destination may BECOME a link (``mv <link> <dest>`` after an
            # earlier ``rm -rf <dest>``): see ``_unpin_relinked``.
            self._name(path)
        if path is not None and into:
            self._add_into_dir(exp, program, sources or [])
            return
        states = [self._link_state(s) for s in sources or ()]
        self._add(exp, "overwrite", program)
        if path is not None and len(states) == 1:
            self._carry_link(path.rstrip("/") or "/", states[0])

    def _add_into_dir(self, exp: Expanded, program: str, sources: list[Expanded]) -> None:
        """``mv``/``cp`` of ``sources`` INTO the directory ``exp``."""
        if exp.opaque or exp.pattern is not None:
            return
        path, _ = _absolute(exp.text, self.state.cwd)
        if path is None:
            return
        # Moving/copying INTO a directory replaces nothing named here, but
        # it CREATES ``dir/<source name>``: recorded as an append so a
        # fake session directory moved into ``<state>/sessions/`` is ranked
        # (``classify._evidence_reason``) and nothing else changes.
        for src in sources:
            if src.opaque or src.pattern is not None or not src.text:
                continue
            name = os.path.basename(src.text.rstrip("/"))
            if name:
                state = self._link_state(src)
                landed = os.path.join(path, name)
                self._add(Expanded(text=landed), "append", program)
                self._carry_link(landed, state)

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


#: Readers that are read-only EXCEPT for one known writing switch or shape,
#: so they cannot sit in :data:`READ_ONLY_PROGRAMS` (which admits no flag
#: inspection). WHY (operator case, 2026-11-09): ``for s in …; done | sort |
#: uniq`` with the cwd under ``<state>/sessions`` was refused as "session
#: state handed to sort/uniq" — reading stdin or a file is not a delete.
#: Writing shapes stay on the backstop:
#:
#: * ``sort -o FILE`` / ``--output`` writes (also a modelled write target in
#:   ``_write_surface``, so a write into evidence is CRITICAL);
#:   ``--compress-program`` runs a program; ``-T``/``--temporary-directory``
#:   writes temp files there.
#: * ``uniq IN OUT`` writes its second operand.
#: * ``file -C`` / ``--compile`` writes a ``.mgc`` (review R6).
#: * ``shasum``/``md5``/``*sum`` have no writing switch (``-c`` reads).
_GUARDED_READERS = frozenset(
    {"sort", "uniq", "file", "shasum", "md5", "md5sum", "sha1sum", "sha256sum", "sha512sum"}
)


#: Long options per guarded reader, split by whether they can write.
#: WHY a closed allowlist (review round 2, M2): getopt_long accepts any
#: unambiguous PREFIX, so ``sort --out=F``, ``--outp F``, ``--compress-prog=sh``,
#: ``--temp=D`` and ``file --comp`` are the writing switches under other
#: spellings — measured on this host (``/usr/bin/sort`` 2.3-Apple wrote ``o.txt``
#: from ``--out=``; ``file --comp`` wrote ``magic.mgc``). A long option counts as
#: read-only only when it is a prefix of a known read-only option AND of no
#: writing one; anything unknown (another platform's spelling, a typo) fails
#: closed to the backstop — a misread costs a judge call, never a LOW write.
_SORT_WRITING_LONGS = ("--output", "--compress-program", "--temporary-directory")
_SORT_READ_LONGS = (
    "--ignore-leading-blanks", "--dictionary-order", "--ignore-case",
    "--general-numeric-sort", "--ignore-nonprinting", "--month-sort",
    "--human-numeric-sort", "--numeric-sort", "--random-sort", "--random-source",
    "--reverse", "--sort", "--version-sort", "--batch-size", "--check", "--debug",
    "--files0-from", "--key", "--merge", "--buffer-size", "--field-separator",
    "--stable", "--unique", "--zero-terminated", "--parallel", "--help", "--version",
    "--mmap", "--radixsort", "--mergesort", "--qsort", "--heapsort",
)  # fmt: skip
_FILE_WRITING_LONGS = ("--compile",)
_FILE_READ_LONGS = (
    "--apple", "--brief", "--checking-printout", "--dereference", "--exclude",
    "--exclude-quiet", "--extension", "--files-from", "--keep-going", "--list",
    "--magic-file", "--mime", "--mime-type", "--mime-encoding", "--no-buffer",
    "--no-dereference", "--no-pad", "--print0", "--preserve-date", "--raw",
    "--separator", "--special-files", "--uncompress", "--uncompress-noreport",
    "--parameter", "--help", "--version",
)  # fmt: skip


def _long_prefixes(key: str, options: Sequence[str]) -> bool:
    """Could ``key`` (``--x``, value stripped) be getopt_long's abbreviation of
    any of ``options``? ``--`` alone ends options and abbreviates nothing."""
    return len(key) > 2 and any(o.startswith(key) for o in options)


def _longs_read_only(longs: list[str], reads: Sequence[str], writes: Sequence[str]) -> bool:
    return all(
        k == "--" or (_long_prefixes(k, reads) and not _long_prefixes(k, writes)) for k in longs
    )


def _guarded_reader_read_only(name: str, argv: list[str]) -> bool:
    shorts = [a[1:] for a in argv if a.startswith("-") and not a.startswith("--") and a != "-"]
    longs = [a.split("=", 1)[0] for a in argv if a.startswith("--")]
    if name == "sort":
        # Fail closed on any cluster holding o/T, even as another flag's value
        # (``-to``): a misread costs a judge call, never a LOW write.
        if any("o" in c or "T" in c for c in shorts):
            return False
        return _longs_read_only(longs, _SORT_READ_LONGS, _SORT_WRITING_LONGS)
    if name == "uniq":
        # A second operand is the output file. ``-f N``/``-s N`` values count
        # as operands here — fail closed, never an unseen write.
        operands = [a for a in argv if not a.startswith("-") or a == "-"]
        return len(operands) <= 1
    if name == "file":
        if any("C" in c for c in shorts):
            return False
        return _longs_read_only(longs, _FILE_READ_LONGS, _FILE_WRITING_LONGS)
    return True  # shasum/md5/*sum


#: Programs that only READ their path operands and print a digest or a
#: description (incident 2026-11-09: ``shasum`` of a subagent's receipts, kept
#: apart from ``READ_ONLY_PROGRAMS`` so a sibling edit there never conflicts).
#: None of them has a flag that writes a file. ``shasum``/``md5``/``*sum``
#: also sit in :data:`_GUARDED_READERS` (PR #3700); this set is checked
#: first in :func:`read_only_argv`, so the overlap is harmless.
_DIGEST_READERS = frozenset(
    {"shasum", "sha1sum", "sha224sum", "sha256sum", "sha384sum", "sha512sum", "b2sum"}
    | {"md5", "md5sum", "cksum", "sum", "cmp", "diff"}
)
#: ImageMagick ``-write`` and coder-prefixed inputs (``msl:script``, a script
#: format that can write files) make ``identify`` more than a read.
_MAGICK_CODER_RE = re.compile(r"^[A-Za-z0-9]{2,}:")


#: ``gh`` subcommands whose ``-F``/``--body-file``/``--notes-file`` value is a
#: file gh READS and posts (incident 2026-11-09: ``gh pr comment -F <child's
#: scratch file>`` was ranked as a write to another session's files).
_GH_BODY_FILE_COMMANDS = frozenset(
    {
        ("pr", "comment"),
        ("pr", "create"),
        ("pr", "edit"),
        ("pr", "review"),
        ("pr", "merge"),
        ("issue", "comment"),
        ("issue", "create"),
        ("issue", "edit"),
        ("release", "create"),
        ("release", "edit"),
    }
)
_GH_BODY_FILE_FLAGS = ("-F", "--body-file", "--notes-file")


def _read_inputs(name: str, args: list[Expanded]) -> set[int]:
    """Ids of words a non-read-only program provably only READS.

    The backstop skips them, exactly like the identified sources of a fully
    modelled copy. Deliberately narrow: today only ``gh``'s body/notes file;
    every other word of the call (``--repo``, a positional path) still meets
    the backstop, and a glued ``--body-file=<p>`` is a read too.
    """
    if name != "gh":
        return set()
    positional = [w.text for w in args if not w.text.startswith("-")][:2]
    if tuple(positional) not in _GH_BODY_FILE_COMMANDS:
        return set()
    out: set[int] = set()
    for i, w in enumerate(args):
        if w.text in _GH_BODY_FILE_FLAGS and i + 1 < len(args):
            out.add(id(args[i + 1]))
        elif w.text.startswith(("--body-file=", "--notes-file=")):
            out.add(id(w))
        elif w.text.startswith("-F") and len(w.text) > 2 and not w.text.startswith("--"):
            out.add(id(w))
    return out


def _sed_read_only(argv: list[str]) -> bool:
    """Is this ``sed`` call provably a pure read (prints, never writes)?

    WHY (field report 2026-11-12, case B): ``sed -n '1,135p' <file>`` is how
    agents page a file, yet sed was never trusted, so every path it named hit
    the backstop as "handed to sed" and a reviewer could not read the frozen
    files it was reviewing. Trusted only when every flag is a known non-writing
    one (no ``-i``, no ``-f``, no unknown or abbreviated long option) and the
    script — the ``-e`` values or the first operand, never the input files
    (``_sed_script_text``) — holds no ``w``/``W``/``e`` at all. That letter
    test is deliberately crude: it also rejects ``s/we/x/p``, which costs a
    judge call, never a missed ``w FILE`` or ``s///e``.
    """
    words = [Expanded(text=a) for a in argv]
    script = _sed_script_text(words)
    if script is None or _SED_WRITES_RE.search(script):
        return False
    return _flags_known(words, _SED_READ_FLAGS)


def _sed_executes(words: list[Expanded]) -> bool:
    """May this ``sed`` run code this check cannot see (``e``, ``s///e``, ``-f``)?

    WHY (QA Q2 on PR #3906): a sed that is not provably read-only reaches the
    backstop, and its FILE operands then looked like plain reads, so from a
    lineage caller ``sed 's/x/rm -rf ~/e' <parent file>`` or ``sed -f ~/x
    <parent file>`` earned ``classify._tree_read``'s read credit. Such a sed is
    an executor, not a reader: the caller marks its mentions ``executes`` and
    they get no lineage credit. Crude like ``_sed_read_only``: ANY ``e`` in the
    script counts (``/needle/p`` included), which costs a judge call, never a
    missed ``e`` command; a script in a file (``-f``/``--file``) is unknowable.
    """
    script = _sed_script_text(words)
    return script is None or "e" in script


def _identify_read_only(argv: list[str]) -> bool:
    return not any(a.lstrip("+-").startswith("write") or _MAGICK_CODER_RE.match(a) for a in argv)


def read_only_argv(name: str, argv: list[str], *, assigned: Sequence[str] = ()) -> bool:
    """Is ``name argv…`` a provably read-only invocation? Conservative: unknown → no."""
    if name in READ_ONLY_PROGRAMS or name in _DIGEST_READERS:
        return True
    if name in _GUARDED_READERS:
        return _guarded_reader_read_only(name, argv)
    if name == "identify":
        return _identify_read_only(argv)
    if name == "magick":
        return bool(argv) and argv[0] == "identify" and _identify_read_only(argv[1:])
    if name in ("sed", "gsed"):
        return _sed_read_only(argv)
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
    "nEreiIsuz",
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
#: ``sed`` flags that never write: ``_SED_FLAGS`` without ``-i``/``--in-place``
#: and without ``-f`` (a script file cannot be inspected here).
_SED_READ_FLAGS = _Flags(
    "nErsuz",
    frozenset(
        {
            "--quiet",
            "--silent",
            "--regexp-extended",
            "--separate",
            "--unbuffered",
            "--null-data",
            "--posix",
        }
    ),
    frozenset({"-e", "--expression", "-l", "--line-length"}),
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


def _sed_script_text(args: list[Expanded]) -> str | None:
    """The text of sed's SCRIPT only, or ``None`` when it comes from a file.

    WHY (field report 2026-11-12): the write check used to scan every
    non-flag word, FILE operands included, so the ``e`` in ``…/sessions/…``
    matched ``_SED_WRITES_RE`` and a pure ``sed -n '1,135p' <pad file>`` was
    never fully modelled — the path went to the backstop as "handed to sed".
    The script is each ``-e``/``--expression`` value, else the first operand;
    every later operand is an input file. ``-f`` (script in a file) is
    unknowable, so the caller keeps the conservative ranking.
    """
    scripts: list[str] = []
    operands: list[str] = []
    i = 0
    while i < len(args):
        t = args[i].text
        i += 1
        if t == "--":
            operands.extend(w.text for w in args[i:])
            break
        if t.startswith("--"):
            key, eq, value = t.partition("=")
            if key in ("--file",):
                return None
            if key == "--expression":
                if eq:
                    scripts.append(value)
                elif i < len(args):
                    scripts.append(args[i].text)
                    i += 1
            elif key == "--line-length" and not eq:
                i += 1
            continue
        if t.startswith("-") and t != "-":
            for k, ch in enumerate(t[1:], start=1):
                if ch in "efl":
                    if ch == "f":
                        return None
                    glued = t[k + 1 :]
                    if glued:
                        value = glued
                    elif i < len(args):
                        value = args[i].text
                        i += 1
                    else:
                        value = ""
                    if ch == "e":
                        scripts.append(value)
                    break
            else:
                if t.endswith("I"):
                    i += 1  # R2-2: BSD ``-I <suffix>``; the suffix is no script
            continue
        operands.append(t)
    if not scripts and operands:
        scripts.append(operands[0])
    return " ".join(scripts)


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
    if name == "sort":
        # ``sort -o FILE`` writes FILE (it may also be an input: ``sort -o x x``).
        # Never fully modelled, so its other path words reach the backstop too.
        # ``--out=F``/``--outp F`` are getopt_long abbreviations of ``--output``
        # (review round 2, M2): name them too, so a write into evidence stays
        # CRITICAL rather than falling to the backstop's HIGH.
        return _cluster_value(args, "o", "--output") + _sort_output_abbrev(args), False
    if name in ("mkdir", "gmkdir"):
        return [], _flags_known(args, _MKDIR_FLAGS)  # resolver creations
    if name in ("chmod", "chown", "chgrp", "xattr"):
        return [], _flags_known(args, _META_FLAGS)  # metadata only, no content
    if name in ("sed", "gsed"):
        # ``-i`` operands are resolver targets; a script's own ``w`` is not.
        scripts = _sed_script_text(args)
        # Its own ``w``/``W`` commands write: a parseable filename there is a
        # destination (review R5-MINOR-1); a script that is not a provably
        # read-only one is never fully modelled, so the backstop sees it.
        dests = [
            Expanded(text=_tilde_value(m, home))
            for m in _SED_W_PATH_RE.findall(" ".join(w.text for w in args))
        ]
        if scripts is None:
            # ``-f FILE``: the script is a file this check cannot read.
            return dests, False
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


def _sort_output_abbrev(args: list[Expanded]) -> list[Expanded]:
    """Values of ``--output`` abbreviations (``--o``…``--outpu``). No other sort
    long option starts with ``o``, so even ``--o`` means ``--output``."""
    out: list[Expanded] = []
    for k, w in enumerate(args):
        key, eq, value = w.text.partition("=")
        if key == "--output" or not _long_prefixes(key, ("--output",)):
            continue  # the full spelling is ``_cluster_value``'s
        if eq:
            out.append(Expanded(text=value, empty=w.empty))
        elif k + 1 < len(args):
            out.append(args[k + 1])
    return out


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


def _no_clobber_new(dest: Expanded, flags: list[str], cwd: str | None) -> bool:
    """``cp -n``/``--no-clobber``/``--update=none`` onto a path that does not exist.

    Only then is the destination a pure creation. Any flag that can force a
    replacement (``-f``, ``-i`` — which may be answered yes —, ``--remove-
    destination``, another ``--update``) keeps it an overwrite, and so does an
    existing entry (a dangling symlink included), a directory (``cp`` writes
    INTO it), an opaque or globbed word, or an unknown cwd.
    """
    no_clobber = False
    for flag in flags:
        if flag in ("--no-clobber", "--update=none"):
            no_clobber = True
        elif flag.startswith("--"):
            if flag.startswith(("--force", "--remove-destination", "--update", "--interactive")):
                return False
        elif any(c in flag[1:] for c in "fi"):
            return False
        elif _short_has(flag, "n", _CP_FLAGS):
            no_clobber = True
    if not no_clobber or dest.opaque or dest.pattern is not None or not dest.text:
        return False
    path, _ = _absolute(dest.text, cwd)
    return path is not None and not os.path.lexists(path) and not dest.text.endswith("/")


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
    # Physically where it can be pinned (a ``..`` after a link lands where the
    # kernel puts it); the lexical spelling otherwise, as before.
    return (None if _GLOB_CHARS.search(text) else physical_path(text)) or os.path.normpath(text)


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
