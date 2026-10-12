"""Promote exact successful current-turn HTML, never arbitrary files or a web bundler.

The adapter accepts a deliberately small standalone document profile. Unsupported
head resources stay file callouts rather than becoming blank CSP-blocked previews.
The existing validator, attachment store and opaque sandbox remain authoritative.
"""

import re
from html import escape
from html.parser import HTMLParser
from typing import Sequence

from local_operator.supplements.evidence import Dataset
from local_operator.supplements.validate import ValidationResult, validate_output

_DOCUMENT = re.compile(
    r"\A\s*<!doctype\s+html\s*>\s*<html(?:\s+lang=['\"][\w-]+['\"])?\s*>\s*"
    r"<head\s*>(?P<head>.*?)</head\s*>\s*<body\s*>(?P<body>.*?)</body\s*>\s*</html\s*>\s*\Z",
    re.I | re.S,
)


class _HeadProfile(HTMLParser):
    def __init__(self, raw: str) -> None:
        super().__init__()
        self.raw = raw
        self.removed: list[tuple[int, int]] = []
        self.title = ""
        self.titles = 0
        self.title_start = 0
        self.problem = ""
        self.current = ""

    def node_offset(self) -> int:
        line, column = self.getpos()
        return sum(len(item) + 1 for item in self.raw.split("\n")[: line - 1]) + column

    def handle_starttag(self, tag: str, attrs: list[tuple[str, str | None]]) -> None:
        if self.current:
            self.problem = "nested head elements are unsupported"
        if tag not in {"title", "meta", "style", "script"}:
            self.problem = "unsupported document head; preview requires inline resources"
        if tag == "meta" and [(key, (value or "").casefold()) for key, value in attrs] != [
            ("charset", "utf-8")
        ]:
            self.problem = "only UTF-8 charset metadata is supported"
        if tag in {"title", "style"} and attrs:
            self.problem = "unsupported head attributes"
        if tag == "script" and any(key not in {"type"} for key, _value in attrs):
            self.problem = "script dependencies are not supported"
        if tag == "meta":
            begin = self.node_offset()
            self.removed.append((begin, begin + len(self.get_starttag_text() or "")))
        if tag == "title":
            self.titles += 1
            self.title_start = self.node_offset()
        if tag != "meta":
            self.current = tag

    def handle_endtag(self, tag: str) -> None:
        if tag != self.current:
            self.problem = "unbalanced document head"
        if tag == "title":
            begin = self.node_offset()
            end = self.raw.find(">", begin) + 1
            self.removed.append((self.title_start, end))
        self.current = ""

    def handle_startendtag(self, tag: str, attrs: list[tuple[str, str | None]]) -> None:
        self.handle_starttag(tag, attrs)
        if tag != "meta":
            self.problem = "self-closing document head elements are unsupported"

    def handle_data(self, data: str) -> None:
        if self.current == "title":
            self.title += data
        if not self.current and data.strip():
            self.problem = "unwrapped document head text"


def promote(
    content: str, source: str, evidence: Sequence[Dataset]
) -> tuple[ValidationResult | None, str]:
    """Return a validated component or fixed refusal copy; never echo document bytes."""
    match = _DOCUMENT.fullmatch(content)
    if match is None:
        return None, "prepared:unsupported"
    head, body = match.group("head", "body")
    checker = _HeadProfile(head)
    checker.feed(head)
    checker.close()
    if checker.problem or checker.current:
        return None, "prepared:unsupported"
    if checker.titles != 1 or not checker.title.strip():
        return None, "prepared:unsupported"
    # Only document scaffolding is adapted; inline style/script order stays exactly
    # as authored. Reject unsupported resources rather than stripping them silently.
    fragment_head = head
    for start, end in sorted(checker.removed, reverse=True):
        fragment_head = fragment_head[:start] + fragment_head[end:]
    fragment = fragment_head + body
    title = checker.title.strip()
    result = validate_output(
        f'<component title="{escape(title, quote=True)}" '
        f'source="{escape(source, quote=True)}"><html>{fragment}</html></component>',
        evidence,
        authored=True,
    )
    if not result.components:
        return (
            result,
            "prepared:unsupported",
        )
    return result, ""
