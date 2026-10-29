"""Identifier metadata verifier for reduce-v2, not a general word allowlist.

These source extractors mirror reference recall.py (rev7 contract repository).
They recognise the documented identifier-pool transformation only. Ordinary
words are NOT identifiers: allowing arbitrary source words after an I1 marker
would reopen synthetic instruction assembly after record provenance was fixed.
No reduction choice or model logic runs here; all returned values are derived
from the client's source, and every emitted identifier may be used once.
"""

from __future__ import annotations

import re

_PATH_RE = re.compile(
    r"(?<![\w:/.])(?:~|\.{1,2})?/(?:[\w.@+-]+/)*[\w.@+-]+"
    r"|(?<![\w/.:-])(?:[\w.-]+/)+[\w-]+\.[A-Za-z0-9]{1,8}\b"
)
_URL_RE = re.compile(r"https?://[^\s\"'<>()\[\]{}`]+")
_SHA_RE = re.compile(r"\b(?=[0-9a-f]*[a-f])(?=[0-9a-f]*[0-9])[0-9a-f]{7,40}\b")
# [ \t]*, not \s*: with \s* an issue id could hold a line break ("PR\n#42"),
# and reduce-v2's identifier pool then wrote "#42" at column 0, which parses as
# no record (a P-frame violation, reproduced on a 3-message span). No other
# extractor can cross a "\n".
_ISSUE_RE = re.compile(r"(?:\b(?:PR|MR|issue|pull)[ \t]*)?[#!]\d{1,6}\b", re.I)
# Anchored at a line start (after indentation and common modifiers): prose like
# "the type for" or "a function expected" otherwise yields "for"/"expected".
_DEF_RE = re.compile(
    r"(?m)^[ \t]*(?:export\s+|pub(?:\(\w+\))?\s+|async\s+|public\s+"
    r"|private\s+|static\s+|default\s+)*"
    r"(?:def|class|func|function|interface|struct|type|fn|const|let|var)\s+"
    r"(?:\([^)]*\)\s*)?([A-Za-z_]\w{2,})"
)
_CALL_RE = re.compile(
    r"\b([a-z_][a-z0-9]*_[a-z0-9_]+|[a-z]+[A-Z]\w+|[A-Z][a-z]+(?:[A-Z][a-z0-9]+)+)\s*\("
)
_ERROR_CODE_RE = re.compile(
    r"\b(?:E[A-Z]{3,}|[A-Z]{1,4}\d{3,5}|[A-Z][A-Za-z]+(?:Error|Exception|Panic))\b"
)
_ERROR_LINE_RE = re.compile(r"(?im)^.*\b(?:error|failed|failure|panic|traceback|fatal)\b.*$")
_ENV_RE = re.compile(r"\b[A-Z][A-Z0-9]*_[A-Z0-9_]{2,}\b")
_PORT_RE = re.compile(r"(?:localhost|127\.0\.0\.1|0\.0\.0\.0|\bport)[:= ]\s*(\d{2,5})\b", re.I)


def identifiers(text: str) -> set[str]:
    out = {m[0].rstrip(".,;:") for m in _URL_RE.finditer(text)}
    no_urls = _URL_RE.sub(" ", text)
    for match in _PATH_RE.finditer(no_urls):
        value = match[0].rstrip(".,;:")
        if (
            len(value) >= 4
            and not value.startswith("//")
            and (
                value.count("/") >= 2
                or value.startswith(("~/", "./", "../"))
                or re.search(r"\.[A-Za-z0-9]{1,8}$", value)
            )
        ):
            out.add(value)
    out.update(_SHA_RE.findall(text))
    out.update(re.sub(r"[ \t]+", " ", m[0]) for m in _ISSUE_RE.finditer(text))
    out.update(
        s for s in _DEF_RE.findall(text) if len(s) >= 6 or "_" in s or any(c.isupper() for c in s)
    )
    out.update(_CALL_RE.findall(text))
    out.update(_ERROR_CODE_RE.findall(text))
    out.update(_ENV_RE.findall(text))
    out.update(_PORT_RE.findall(text))
    for match in _ERROR_LINE_RE.finditer(text):
        value = match[0].strip()[:80]
        if len(value) >= 12 and " " not in value:
            out.add(value)
    # The identifier pool uses _short(v, 120), not arbitrary prefixes.
    return {value if len(value) <= 120 else value[:120].rstrip() + "…" for value in out}
