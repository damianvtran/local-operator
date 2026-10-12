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
_TITLE = re.compile(r"<title\s*>(.*?)</title\s*>", re.I | re.S)
_CHARSET = re.compile(r"<meta\s+charset\s*=\s*['\"]?utf-8['\"]?\s*/?>", re.I)


class _HeadProfile(HTMLParser):
    def __init__(self) -> None:
        super().__init__()
        self.problem = ""
        self.current = ""

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
        if tag != "meta":
            self.current = tag

    def handle_endtag(self, tag: str) -> None:
        if tag != self.current:
            self.problem = "unbalanced document head"
        self.current = ""

    def handle_startendtag(self, tag: str, attrs: list[tuple[str, str | None]]) -> None:
        self.handle_starttag(tag, attrs)
        if tag != "meta":
            self.problem = "self-closing document head elements are unsupported"

    def handle_data(self, data: str) -> None:
        if not self.current and data.strip():
            self.problem = "unwrapped document head text"


def promote(
    content: str, source: str, evidence: Sequence[Dataset]
) -> tuple[ValidationResult | None, str]:
    """Return a validated component or fixed refusal copy; never echo document bytes."""
    match = _DOCUMENT.fullmatch(content)
    if match is None:
        return None, "Preview requires a standalone HTML document with head and body."
    head, body = match.group("head", "body")
    titles = list(_TITLE.finditer(head))
    if len(titles) != 1 or not titles[0].group(1).strip() or "<" in titles[0].group(1):
        return None, "Preview requires one plain-text document title."
    checker = _HeadProfile()
    checker.feed(head)
    checker.close()
    if checker.problem or checker.current:
        return None, "Preview requires inline resources and supported UTF-8 document metadata."
    # Only document scaffolding is adapted; inline style/script order stays exactly
    # as authored. Reject unsupported resources rather than stripping them silently.
    fragment = _CHARSET.sub("", _TITLE.sub("", head)) + body
    title = titles[0].group(1).strip()
    result = validate_output(
        f'<component title="{escape(title, quote=True)}" '
        f'source="{escape(source, quote=True)}"><html>{fragment}</html></component>',
        evidence,
    )
    if not result.components:
        return (
            result,
            "Preview unavailable: inline dependencies or data provenance could not be validated.",
        )
    return result, ""
