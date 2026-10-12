"""Fail-closed copy scan for the SSR login gate, not the unextracted relay SPA.

The general Python sink ratchet cannot see text inside a multiline HTML
constant. This HTML-aware gate is unconditional in check.py: a new literal
must be extracted rather than hidden by lowering or resetting a ceiling.
Nontranslatable exceptions are exact and documented in docs/design/i18n.md.
"""

from __future__ import annotations

import ast
import re
from html.parser import HTMLParser

_TOKEN = re.compile(r"__[A-Z_]+__")
_COPY_ATTRIBUTES = {
    "alt",
    "title",
    "placeholder",
    "aria-label",
    "aria-description",
    "aria-valuetext",
}
# The identity is a fixed wordmark, not an instruction. Scope the exception to
# its actual element so writing the same words on a button still fails.
_BRAND = ("h1", "local operator")
# The only script on this gate clears private storage; these are wire keys,
# not visible copy. New JS literals fail even inside this otherwise opaque sink.
_SCRIPT_KEYS = {"lo-mobile-command:", "lo-mobile-draft:"}
# Structural renderer tokens are not copy. Keep this closed list rather than
# exempting the renderer: a future string replacing a generated Msg must fail.
_RENDERER_TOKENS = {
    "LANG",
    "DIR",
    "MARK_DATA_URI",
    "TITLE",
    "PASSWORD",
    "SUBMIT",
    "ERROR",
    "rtl",
    "ur",
    "ltr",
    "en",
    "relay.login",
    "",
    '<p class="error" role="alert">',
    "</p>",
    r"__([A-Z_]+)__",
}


def _script_literals(code: str) -> list[str]:
    """Recognize comments only outside quoted literals, preserving escapes.

    This is a conservative copy lexer, not a JavaScript evaluator. Backtick
    content (including interpolations) is checked as a whole; new dynamic
    script copy is not an exemption. Escapes remain raw so only the two exact
    storage keys are allowlisted. Consuming an escaped quote as one unit is
    essential: otherwise a URL/comment-looking suffix can hide following copy.
    """
    literals: list[str] = []
    index = 0
    while index < len(code):
        if code.startswith("//", index):
            index += 2
            while index < len(code) and code[index] not in "\r\n\u2028\u2029":
                index += 1
        elif code.startswith("/*", index):
            end = code.find("*/", index + 2)
            index = len(code) if end < 0 else end + 2
        elif code[index] in "\"'`":
            quote = code[index]
            index += 1
            start = index
            while index < len(code):
                if code[index] == "\\":
                    index += 2
                elif code[index] == quote:
                    break
                else:
                    index += 1
            literals.append(code[start:index])
            index += 1
        else:
            index += 1
    return literals


class _LoginCopy(HTMLParser):
    def __init__(self) -> None:
        super().__init__(convert_charrefs=True)
        self.tags: list[str] = []
        self.problems: list[str] = []

    def check_copy(self, value: str, where: str) -> None:
        if _TOKEN.sub("", value).strip():
            self.problems.append(
                f"relay.login: hardcoded {where}: {value!r}"
            )  # i18n: ignore developer diagnostic

    def handle_starttag(self, tag: str, attrs: list[tuple[str, str | None]]) -> None:
        for name, value in attrs:
            if name in _COPY_ATTRIBUTES or (tag == "input" and name == "value"):
                self.check_copy(value or "", f"{tag}[{name}]")
        self.tags.append(tag)

    def handle_endtag(self, tag: str) -> None:
        if tag in self.tags:
            self.tags = self.tags[: len(self.tags) - 1 - self.tags[::-1].index(tag)]

    def handle_data(self, data: str) -> None:
        tag = self.tags[-1] if self.tags else ""
        if tag == "style" or not data.strip():
            return
        if tag == "script":
            # Comments describe cleanup for developers. The executable string
            # literals are checked separately so new JS-visible copy cannot hide.
            for literal in _script_literals(data):
                if literal not in _SCRIPT_KEYS:
                    self.check_copy(literal, "script literal")
        elif (tag, data.strip()) != _BRAND:
            self.check_copy(data, f"{tag or 'document'} text")


def check_login_source(source: str) -> list[str]:
    """Scan the exact template without importing/booting the daemon."""
    tree = ast.parse(source)
    renderer_problems: list[str] = []
    for function in tree.body:
        if isinstance(function, ast.FunctionDef) and function.name == "_render_login_html":
            for statement in function.body:
                # The first statement is its developer-only docstring.
                if isinstance(statement, ast.Expr) and isinstance(statement.value, ast.Constant):
                    continue
                for literal in ast.walk(statement):
                    if (
                        isinstance(literal, ast.Constant)
                        and isinstance(literal.value, str)
                        and literal.value not in _RENDERER_TOKENS
                    ):
                        renderer_problems.append(
                            f"relay.login: hardcoded renderer literal: {literal.value!r}"
                        )  # i18n: ignore developer diagnostic
    for node in ast.walk(tree):
        if isinstance(node, ast.Assign) and any(
            isinstance(target, ast.Name) and target.id == "_LOGIN_HTML" for target in node.targets
        ):
            if not isinstance(node.value, ast.Constant) or not isinstance(node.value.value, str):
                return [
                    "relay.login: template must remain a literal for copy enforcement"
                ]  # i18n: ignore developer diagnostic
            parser = _LoginCopy()
            parser.feed(node.value.value)
            return renderer_problems + parser.problems
    return ["relay.login: missing SSR template"]  # i18n: ignore developer diagnostic


def check_login(root) -> list[str]:
    path = root / "local_operator/mobile/daemon.py"
    # Mini-tree ratchet tests deliberately omit this unrelated production file.
    return check_login_source(path.read_text(encoding="utf-8")) if path.is_file() else []
