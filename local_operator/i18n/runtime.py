"""The ICU MessageFormat SUBSET this project ships, parsed and rendered.

This is the in-house runtime the RFC commits to (§2.3, operator decision §11.5:
no new Python dependency). It exists so catalogues can hold one message string
per key and every surface formats it identically without pulling in Babel.

THE SUBSET, exactly — anything outside it is a loud ``MessageSyntaxError``, not
a best-effort parse:

* ``{name}`` interpolation — numbers formatted per locale, dates as
  ``medium`` date + ``short`` time, everything else ``str()``;
* ``{name, number}`` and ``{name, number, integer|percent|decimal}``;
* ``{name, date}`` / ``{name, time}`` with ``short|medium|long`` (date) and
  ``short|medium`` (time) skeletons;
* ``{name, plural, ...}`` with category selectors (``one``/``few``/``many``/
  ``other`` per the generated tables) plus exact ``=N`` selectors, ``#`` as the
  argument's number inside the selected branch, and nesting;
* ``{name, select, ...}`` with string selectors (``other`` mandatory);
* escaping: ``''`` -> ``'``, and ``'{'``/``'}'``/``'#'`` for literal braces and
  hash; nested arguments inside any branch.

Deliberately absent (each currently errors instead of mis-rendering): plural
``offset:``, ``{... `` argument formats outside the list above, number
skeletons like ``::currency``, and skeleton-pattern syntax (``::``). The RFC's
parity story is "render every message with both runtimes" — a growing subset is
fine, an unstated semantic difference is not.

PARSING IS CACHED and pure: :func:`parse_message` returns an immutable AST that
:func:`render_message` walks with a params mapping. Errors carry the source
offset so a red parity test names the position without a debugger.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import date, datetime
from decimal import Decimal
from functools import lru_cache
from typing import Any, Mapping, Sequence, Union

from . import format as fmt


class MessageError(ValueError):
    """Base for both parse-time and render-time message errors."""


class MessageSyntaxError(MessageError):
    """The message source is outside the supported ICU subset."""


class MessageFormatError(MessageError):
    """The message parsed, but cannot render with the given params/locale."""


@dataclass(frozen=True)
class _Text:
    value: str


@dataclass(frozen=True)
class _Hash:
    """A ``#`` inside a plural branch — renders the plural's number."""


@dataclass(frozen=True)
class _Arg:
    name: str
    kind: str  # simple | number | date | time | plural | select
    skeleton: str | None
    parts: tuple[tuple[str, tuple["_Node", ...]], ...] = ()


_Node = Union[_Text, _Hash, _Arg]

_NUMBER_SKELETONS = ("", "integer", "percent", "decimal")
_DATE_SKELETONS = ("short", "medium", "long")
_TIME_SKELETONS = ("short", "medium")
_CATEGORY_RE = str.isalpha


class _Parser:
    def __init__(self, source: str) -> None:
        self.source = source
        self.index = 0
        # How many plural branches enclose the current position. `#` is the
        # current plural's number at ANY depth inside a plural — including a
        # `select` nested in one — and the RENDERER binds it to the nearest
        # enclosing plural (ICU semantics); the parser only needs to know
        # whether some plural encloses it at all.
        self.plural_depth = 0

    # -- primitives ---------------------------------------------------------

    def error(self, message: str, *, at: int | None = None) -> "MessageSyntaxError":
        where = self.index if at is None else at
        return MessageSyntaxError(f"{message} (at offset {where} of {self.source!r})")

    def peek(self) -> str | None:
        return self.source[self.index] if self.index < len(self.source) else None

    def expect(self, char: str) -> None:
        if self.peek() != char:
            raise self.error(f"expected {char!r}")
        self.index += 1

    def skip_ws(self) -> None:
        while (c := self.peek()) is not None and c in " \t\r\n":
            self.index += 1

    def read_word(self) -> str:
        start = self.index
        while (c := self.peek()) is not None and (c.isalnum() or c in "_-."):
            self.index += 1
        if self.index == start:
            raise self.error("expected a word")
        return self.source[start : self.index]

    def read_name(self) -> str:
        start = self.index
        c = self.peek()
        if c is None or not (c.isalpha() or c == "_"):
            raise self.error("expected an argument name")
        self.index += 1
        while (c := self.peek()) is not None and (c.isalnum() or c == "_"):
            self.index += 1
        return self.source[start : self.index]

    # -- message body -------------------------------------------------------

    def parse(self) -> tuple[_Node, ...]:
        """Parse nodes until end-of-input or the closing ``}`` of a branch."""
        nodes: list[_Node] = []
        text: list[str] = []

        def flush() -> None:
            if text:
                nodes.append(_Text("".join(text)))
                text.clear()

        while True:
            c = self.peek()
            if c is None or c == "}":
                break
            if c == "'":
                after = self.source[self.index + 1] if self.index + 1 < len(self.source) else None
                if after == "'":
                    text.append("'")
                    self.index += 2
                    continue
                if after is not None and after in "{#}":
                    # ICU QUOTED RUN, not a single escaped character: literal
                    # text from here to the next single quote (or the end of
                    # input), with the closing quote consumed. This is what
                    # lets a message carry its own braces — '{n}' is a literal
                    # brace pair around an n, where {n} would be interpolation.
                    self.index += 1
                    end = self.source.find("'", self.index)
                    if end == -1:
                        text.append(self.source[self.index :])
                        self.index = len(self.source)
                    else:
                        text.append(self.source[self.index : end])
                        self.index = end + 1
                    continue
                text.append("'")
                self.index += 1
                continue
            if c == "#" and self.plural_depth > 0:
                flush()
                nodes.append(_Hash())
                self.index += 1
                continue
            if c == "{":
                flush()
                nodes.append(self.parse_argument())
                continue
            text.append(c)
            self.index += 1
        flush()
        return tuple(nodes)

    # -- arguments ----------------------------------------------------------

    def parse_argument(self) -> _Arg:
        open_at = self.index
        self.expect("{")
        self.skip_ws()
        name = self.read_name()
        self.skip_ws()
        c = self.peek()
        if c == "}":
            self.index += 1
            return _Arg(name=name, kind="simple", skeleton=None)
        if c != ",":
            raise self.error(f"expected ',' or '}}' after argument name {name!r}")
        self.index += 1
        self.skip_ws()
        if self.source.startswith("offset:", self.index):
            raise self.error("plural `offset:` is not in the supported subset", at=open_at)
        kind_word = self.read_word()
        if kind_word not in ("number", "date", "time", "plural", "select"):
            raise self.error(f"unsupported argument type {kind_word!r}", at=open_at)
        self.skip_ws()
        if kind_word in ("plural", "select"):
            return self.parse_branching(name, kind_word, open_at)
        return self.parse_formatted(name, kind_word, open_at)

    def parse_formatted(self, name: str, kind: str, open_at: int) -> _Arg:
        skeleton: str | None = None
        if self.peek() == ",":
            self.index += 1
            self.skip_ws()
            skeleton = self.read_word()
            self.skip_ws()
        self.expect("}")
        if kind == "number" and (skeleton or "") not in _NUMBER_SKELETONS:
            raise self.error(f"unsupported number skeleton {skeleton!r}", at=open_at)
        if kind == "date" and (skeleton or "medium") not in _DATE_SKELETONS:
            raise self.error(f"unsupported date skeleton {skeleton!r}", at=open_at)
        if kind == "time" and (skeleton or "short") not in _TIME_SKELETONS:
            raise self.error(f"unsupported time skeleton {skeleton!r}", at=open_at)
        return _Arg(name=name, kind=kind, skeleton=skeleton)

    def parse_branching(self, name: str, kind: str, open_at: int) -> _Arg:
        self.expect(",")
        parts: list[tuple[str, tuple[_Node, ...]]] = []
        seen: set[str] = set()
        if kind == "plural":
            self.plural_depth += 1
        try:
            while True:
                self.skip_ws()
                if self.peek() == "}":
                    self.index += 1
                    break
                selector = self.read_selector(kind, open_at)
                if selector in seen:
                    raise self.error(f"duplicate selector {selector!r}", at=open_at)
                seen.add(selector)
                self.skip_ws()
                self.expect("{")
                body = self.parse()
                self.expect("}")
                parts.append((selector, body))
        finally:
            if kind == "plural":
                self.plural_depth -= 1
        if not parts:
            raise self.error(f"{kind} argument {name!r} has no branches", at=open_at)
        return _Arg(name=name, kind=kind, skeleton=None, parts=tuple(parts))

    def read_selector(self, kind: str, open_at: int) -> str:
        if kind == "plural" and self.peek() == "=":
            self.index += 1
            start = self.index
            if (c := self.peek()) is not None and c in "+-":
                self.index += 1
            while (c := self.peek()) is not None and c.isdigit():
                self.index += 1
            if self.index == start or self.source[start : self.index].lstrip("+-") == "":
                raise self.error("empty plural '=' selector", at=open_at)
            return "=" + self.source[start : self.index]
        word = self.read_word()
        if kind == "plural" and not word.isalpha():
            raise self.error(f"invalid plural selector {word!r}", at=open_at)
        return word


@lru_cache(maxsize=4096)
def parse_message(source: str) -> tuple[_Node, ...]:
    """Parse ``source`` into an immutable AST (cached — messages are static).

    The returned tuple is shared, so treat it as read-only. Raises
    :class:`MessageSyntaxError` with the source offset for anything outside the
    subset.
    """
    parser = _Parser(source)
    nodes = parser.parse()
    if parser.index != len(source):
        # `parse` stops at a '}'; a top-level stray brace lands here.
        raise parser.error("unmatched '}'")
    return nodes


def message_arguments(source: str) -> tuple[tuple[str, str], ...]:
    """``(name, kind)`` for every argument in ``source``, first-appearance order.

    Kinds: ``simple``, ``number``, ``date``, ``time``, ``plural``, ``select``.
    Nested branches are walked. The key generator reads this to type generated
    functions (RFC §2.4), and the catalogue parity check compares the tuples
    across locales — a translated message that drops or renames an argument
    must fail, not silently render without its value.
    """
    seen: dict[str, str] = {}

    def walk(nodes: Sequence[_Node]) -> None:
        for node in nodes:
            if isinstance(node, _Arg):
                seen.setdefault(node.name, node.kind)
                for _, body in node.parts:
                    walk(body)

    walk(parse_message(source))
    return tuple(seen.items())


def plural_selectors(source: str) -> tuple[tuple[str, ...], ...]:
    """Selector tuples of every plural argument in ``source`` (walk order).

    The catalogue parity check reads these: every literal selector must be a
    plural category of the target locale, and a plural must define ``other``.
    ``=N`` exact selectors are locale-independent and stay in the tuple for
    the caller to recognise (they are valid anywhere).
    """
    found: list[tuple[str, ...]] = []

    def walk(nodes: Sequence[_Node]) -> None:
        for node in nodes:
            if isinstance(node, _Arg):
                if node.kind == "plural":
                    found.append(tuple(selector for selector, _ in node.parts))
                for _, body in node.parts:
                    walk(body)

    walk(parse_message(source))
    return tuple(found)


# ---------------------------------------------------------------------------
# Rendering
# ---------------------------------------------------------------------------


def _interpolate(value: Any, locale: str) -> str:
    """``{name}``'s default formatting: numbers, dates, else ``str()``."""
    if isinstance(value, bool):
        return str(value)
    if isinstance(value, (int, float, Decimal)):
        return fmt.format_number(value, locale)
    if isinstance(value, datetime):
        return fmt.format_datetime(value, locale)
    if isinstance(value, date):
        return fmt.format_date(value, locale)
    return str(value)


def _select_plural(
    locale: str, value: Any, parts: Sequence[tuple[str, tuple[_Node, ...]]]
) -> tuple[_Node, ...]:
    selectors = [selector for selector, _ in parts]
    for selector in selectors:
        if selector.startswith("="):
            try:
                if Decimal(str(value)) == Decimal(selector[1:]):
                    return dict(parts)[selector]
            except (ValueError, ArithmeticError):
                continue
    category = fmt.plural_category(locale, value)
    for selector, body in parts:
        if selector == category:
            return body
    for selector, body in parts:
        if selector == "other":
            return body
    raise MessageFormatError(
        f"plural value {value!r} selected category {category!r}, which the message "
        f"does not define (have: {', '.join(selectors)})"
    )


def _select_string(value: Any, parts: Sequence[tuple[str, tuple[_Node, ...]]]) -> tuple[_Node, ...]:
    key = str(value)
    for selector, body in parts:
        if selector == key:
            return body
    for selector, body in parts:
        if selector == "other":
            return body
    raise MessageFormatError(f"select value {key!r} matched nothing (no 'other' branch)")


def _render(
    nodes: Sequence[_Node],
    params: Mapping[str, Any],
    locale: str,
    plural_value: Any,
) -> str:
    out: list[str] = []
    for node in nodes:
        if isinstance(node, _Text):
            out.append(node.value)
            continue
        if isinstance(node, _Hash):
            out.append(fmt.format_number(plural_value, locale))
            continue
        assert isinstance(node, _Arg)
        if node.name not in params:
            have = ", ".join(sorted(params)) or "none"
            raise MessageFormatError(
                f"message argument {node.name!r} was not supplied (have: {have})"
            )
        value = params[node.name]
        if node.kind == "simple":
            out.append(_interpolate(value, locale))
        elif node.kind == "number":
            out.append(fmt.format_number(value, locale, skeleton=node.skeleton or None))
        elif node.kind == "date":
            out.append(fmt.format_date(value, locale, style=node.skeleton or "medium"))
        elif node.kind == "time":
            out.append(fmt.format_time(value, locale, style=node.skeleton or "short"))
        elif node.kind == "plural":
            body = _select_plural(locale, value, node.parts)
            out.append(_render(body, params, locale, plural_value=value))
        elif node.kind == "select":
            body = _select_string(value, node.parts)
            out.append(_render(body, params, locale, plural_value=plural_value))
        else:  # pragma: no cover - the parser admits no other kinds
            raise MessageFormatError(f"unknown argument kind {node.kind!r}")
    return "".join(out)


def render_message(source: str, params: Mapping[str, Any], locale: str) -> str:
    """Render ``source`` with ``params`` for ``locale``.

    Raises :class:`MessageSyntaxError` (outside the subset) or
    :class:`MessageFormatError` (missing param, no matching branch). Callers
    that must degrade — a user-facing paint path — catch :class:`MessageError`;
    the checker and parity tests let it fly, because a red locus beats a
    silent literal.
    """
    return _render(parse_message(source), params, locale, plural_value=None)
