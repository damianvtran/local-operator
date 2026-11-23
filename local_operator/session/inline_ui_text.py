"""Lower data-only inline UI into the terminal's existing Markdown vocabulary."""

from __future__ import annotations

import json
import math
import re
import unicodedata
from collections.abc import Iterator, Sequence
from typing import Any, Literal, NotRequired, TypedDict, TypeGuard

from markdown_it import MarkdownIt

# Match the paired renderer's display budget; never silently slice a dataset.
_MAX_SOURCE = 16 * 1024
_MAX_ROWS = 32
_MAX_COLUMNS = 8
_MAX_TEXT = 256
_MAX_SERIES = 4
_TAG = "lop-ui"
_MARKDOWN = MarkdownIt().enable("table")
_LINE_END = re.compile(r"\r\n|\r|\n")
_MAYBE_OPENER = re.compile(r" {0,3}(?:`{3,}|~{3,})[ \t]*([a-z-]*)[ \t]*")
_SHORT_OPENER = re.compile(r"( {0,3})(`{1,2}|~{1,2})")
_CATEGORY_TYPES = ("bar", "line", "pie", "histogram")
_GRID_TYPES = ("table", "spreadsheet", "heatmap")


def _utf16_length(text: str) -> int:
    return len(text.encode("utf-16-le", "surrogatepass")) // 2


def _text(value: object) -> TypeGuard[str]:
    return isinstance(value, str) and _utf16_length(value) <= _MAX_TEXT


def _label(value: object) -> TypeGuard[str]:
    return _text(value) and bool(value.strip())


def _number(value: object) -> TypeGuard[int | float]:
    # JSON booleans are not numbers, nor are integers outside the renderer's
    # finite floating-point range.
    if not isinstance(value, (int, float)) or isinstance(value, bool):
        return False
    try:
        return math.isfinite(value)
    except OverflowError:
        return False


def _safe(value: object) -> str:
    if value is None:
        return ""
    # Encode Markdown punctuation (including URL colons and table pipes) so
    # the Markdown parser emits literal text, even after TUI bare-URL linking.
    # Drop terminal/format controls and flatten line breaks before encoding.
    pieces = []
    for char in str(value):
        if char.isspace():
            pieces.append(" ")
        elif not unicodedata.category(char).startswith("C"):
            pieces.append(char if char.isalnum() else f"&#{ord(char)};")
    return "".join(pieces).strip()


def _table(title: str, columns: Sequence[object], rows: Sequence[Sequence[object]]) -> str:
    def row(values: Sequence[object]) -> str:
        return "| " + " | ".join(_safe(cell) for cell in values) + " |"

    rule = "| " + " | ".join(["---"] * len(columns)) + " |"
    return "\n".join(
        [f"**{_safe(title)}**", "", row(columns), rule, *(row(values) for values in rows)]
    )


def _series(value: dict[str, Any]) -> tuple[list[str], list[list[object]]] | None:
    mode, labels, series = (value.get(key) for key in ("mode", "labels", "series"))
    if (
        mode not in ("grouped", "stacked", "percent", "dual")
        or not isinstance(labels, list)
        or not 0 < len(labels) <= _MAX_ROWS
        or not all(map(_label, labels))
        or not isinstance(series, list)
        or not 0 < len(series) <= _MAX_SERIES
        or ("rightUnit" in value and not _text(value["rightUnit"]))
    ):
        return None
    for item in series:
        if (
            not isinstance(item, dict)
            or not _label(item.get("label"))
            or not isinstance(item.get("values"), list)
            or len(item["values"]) != len(labels)
            or not all(map(_number, item["values"]))
            or ("axis" in item and item["axis"] not in ("left", "right"))
        ):
            return None
    headers = [item["label"] for item in series]
    if mode == "dual":
        if (
            not _label(value.get("unit"))
            or not _label(value.get("rightUnit"))
            or {item.get("axis") for item in series} != {"left", "right"}
        ):
            return None
        headers = [
            f"{item['label']} ({value['rightUnit' if item['axis'] == 'right' else 'unit']})"
            for item in series
        ]
    numeric_rows = list(zip(*(item["values"] for item in series), strict=True))
    rows: Sequence[Sequence[object]] = numeric_rows
    if mode == "percent":
        if any(min(row) < 0 or max(row) <= 0 for row in numeric_rows):
            return None
        # Scale first so finite inputs cannot overflow while being summed.
        scaled = [[cell / max(row) for cell in row] for row in numeric_rows]
        rows = [[f"{cell / sum(row):.1%}" for cell in row] for row in scaled]
    elif mode != "dual" and value.get("unit"):
        headers = [f"{header} ({value['unit']})" for header in headers]
    return ["x", *headers], [[label, *row] for label, row in zip(labels, rows, strict=True)]


def _invalid_constant(token: str) -> None:
    raise ValueError(f"Non-JSON constant: {token}")


class _Widget(TypedDict):
    payload: dict[str, Any]
    terminal: str


def _decode(source: str) -> _Widget | None:
    if len(source) > _MAX_SOURCE or _utf16_length(source) > _MAX_SOURCE:
        return None
    try:
        value = json.loads(source, parse_constant=_invalid_constant)
    except (ValueError, RecursionError):
        return None
    return _validate_widget(value)


def _validate_widget(value: object) -> _Widget | None:
    if not isinstance(value, dict) or not _label(value.get("title")):
        return None
    kind = value.get("type")
    if kind in (*_CATEGORY_TYPES, "heatmap", "series") and "unit" in value:
        if not _text(value["unit"]):
            return None
    unit = value.get("unit")
    if kind in (*_CATEGORY_TYPES, "scatter"):
        data = value.get("data")
        if not isinstance(data, list) or not 0 < len(data) <= _MAX_ROWS:
            return None
        for point in data:
            if not isinstance(point, list) or len(point) != 2 or not _number(point[1]):
                return None
            if not (_number(point[0]) if kind == "scatter" else _label(point[0])):
                return None
            if (kind == "pie" and point[1] <= 0) or (kind == "histogram" and point[1] < 0):
                return None
        if kind == "scatter":
            columns = [value.get("xLabel"), value.get("yLabel")]
            if not all(map(_label, columns)):
                return None
        else:
            columns = ["x", unit or "y"]
        fields = ("xLabel", "yLabel") if kind == "scatter" else ("unit",)
        return _widget(value, columns, data, "data", *fields)
    if kind == "series":
        grid = _series(value)
        if grid is None:
            return None
        series = [
            {key: item[key] for key in ("label", "values", "axis") if key in item}
            for item in value["series"]
        ]
        widget = _widget(value, *grid, "mode", "labels", "unit", "rightUnit")
        widget["payload"]["series"] = series
        return widget
    if kind not in _GRID_TYPES:
        return None
    columns, rows = value.get("columns"), value.get("rows")
    if (
        not isinstance(columns, list)
        or not 0 < len(columns) <= _MAX_COLUMNS
        or not all(map(_label, columns))
        or not isinstance(rows, list)
        or len(rows) > _MAX_ROWS
    ):
        return None
    for row in rows:
        if not isinstance(row, list) or len(row) != len(columns):
            return None
        for cell in row:
            if _number(cell) or (kind != "heatmap" and _text(cell)):
                continue
            if cell is not None or kind == "table":
                return None
    if kind == "heatmap":
        labels = value.get("rowLabels")
        if not isinstance(labels, list) or len(labels) != len(rows) or not all(map(_label, labels)):
            return None
        columns = [unit or "#", *columns]
        rows = [[label, *row] for label, row in zip(labels, rows, strict=True)]
        return _widget(value, columns, rows, "columns", "rows", "rowLabels", "unit")
    return _widget(value, columns, rows, "columns", "rows")


def _widget(
    value: dict[str, Any], columns: Sequence[object], rows: Sequence[Sequence[object]], *fields: str
) -> _Widget:
    # Only schema fields cross the wire. In particular, unknown plugin URLs,
    # options and nested styling data never accompany a validated widget.
    payload = {key: value[key] for key in ("type", "title", *fields) if key in value}
    return {"payload": payload, "terminal": _table(value["title"], columns, rows)}


def _parts(source: str, *, streaming: bool) -> Iterator[str | _Widget]:
    """Yield prose and complete validated widgets in source order."""
    if "`" not in source and "~" not in source:
        yield source
        return
    starts = [0, *(match.end() for match in _LINE_END.finditer(source))]
    tail_start = starts[-1]
    tail = source[tail_start:]
    partial = _MAYBE_OPENER.fullmatch(tail)
    if "lop-ui" not in source and not (partial and _TAG.startswith(partial[1])):
        if not (streaming and _SHORT_OPENER.fullmatch(tail)):
            yield source
            return
    parse_source = source
    if streaming and (short := _SHORT_OPENER.fullmatch(tail)):
        # Complete the marker for the block parser only. Its nesting context
        # determines whether this could become a widget in the next delta.
        parse_source = source[:tail_start] + short[1] + short[2][0] * 3
        partial = _MAYBE_OPENER.fullmatch(parse_source[tail_start:])
    cursor = 0
    # Use the same block grammar as Rich rather than mistaking samples in
    # quotes, lists, longer code fences, or HTML blocks for top-level widgets.
    for token in _MARKDOWN.parse(parse_source):
        if token.type != "fence" or token.level or token.map is None:
            continue
        start, end = token.map
        info = token.info.strip()
        if info.split(maxsplit=1)[:1] != [_TAG]:
            if (
                start == len(starts) - 1
                and partial
                and (streaming or partial[1])
                and _TAG.startswith(partial[1])
            ):
                yield source[cursor:tail_start]
                cursor = len(source)
            continue
        low = starts[start]
        high = starts[end] if end < len(starts) else len(source)
        if streaming and start == len(starts) - 1:
            # The info string can still become a different language. Do not
            # commit a separator until the opener itself is confirmed.
            yield source[cursor:low]
            cursor = high
            continue
        last_line = source[starts[end - 1] : high].strip(" \t\r\n")
        closed = (
            end > start + 1
            and last_line.startswith(token.markup)
            and not last_line.strip(token.markup[0])
        )
        # At EOF a live closing marker is still provisional: the next delta
        # can append a non-whitespace character and make it ordinary content.
        confirmed = not streaming or end < len(starts)
        widget = _decode(token.content) if closed and confirmed and info == _TAG else None
        yield source[cursor:low]
        # Separate retained prose so removing a block cannot join Markdown
        # constructs across the gap.
        if widget is not None:
            yield widget
        yield "\n"
        cursor = high
    yield source[cursor:]


INLINE_UI_V1 = "inline-ui-v1"


class InlineWidgetPlacement(TypedDict):
    # UTF-16 offset into the prose, for direct use by JavaScript clients.
    offset: int
    payload: dict[str, Any]


class InlineUiEnvelope(TypedDict):
    version: Literal[1]
    widgets: list[InlineWidgetPlacement]


class InlineUiProjection(TypedDict):
    text: str
    inline_ui: NotRequired[InlineUiEnvelope]


def project_inline_ui(
    source: str, *, streaming: bool, source_version: int | None, capabilities: frozenset[str]
) -> InlineUiProjection:
    """Project one assistant snapshot for streaming, history, or an older viewer.

    Source provenance is explicit: an unmarked historical Markdown fence is
    a code sample, not a component. Only version-1 readers get widget data;
    the plain text field is identical for old and new readers. Offsets count
    UTF-16 code units, not Python code points or transport bytes.
    """
    if source_version != 1:
        return {"text": source}
    rich = INLINE_UI_V1 in capabilities
    text: list[str] = []
    widgets: list[InlineWidgetPlacement] = []
    offset = 0
    for part in _parts(source, streaming=streaming):
        if isinstance(part, str):
            text.append(part)
            if rich:
                offset += _utf16_length(part)
        elif rich:
            widgets.append({"offset": offset, "payload": part["payload"]})
    result: InlineUiProjection = {"text": "".join(text)}
    if widgets:
        result["inline_ui"] = {"version": 1, "widgets": widgets}
    return result


def terminal_inline_ui(source: str, *, streaming: bool) -> str:
    """Replace top-level ``lop-ui`` fences without altering other Markdown."""
    return "".join(
        part if isinstance(part, str) else f"\n{part['terminal']}\n"
        for part in _parts(source, streaming=streaming)
    )


def terminal_inline_ui_overlay(text: str, message_id: str, envelope: object) -> str:
    """Render an attached runtime's typed widget placements, not its source fences."""
    if not isinstance(envelope, dict) or envelope.get("version") != 1:
        return text
    messages = envelope.get("messages")
    row = messages.get(message_id) if isinstance(messages, dict) else None
    if (
        not isinstance(row, dict)
        or row.get("version") != 1
        or not isinstance(basis := row.get("text"), str)
        or not isinstance(widgets := row.get("widgets"), list)
    ):
        return text
    try:
        # Server placement offsets use UTF-16 (JavaScript's string indexing).
        # Split only at a valid boundary; malformed optional data leaves prose.
        wire = basis.encode("utf-16-le")
        parts = []
        lower = 0
        for item in widgets:
            if not isinstance(item, dict) or type(offset := item.get("offset")) is not int:
                return text
            upper = offset * 2
            if not lower <= upper <= len(wire) or not (
                widget := _validate_widget(item.get("payload"))
            ):
                return text
            parts.extend(
                (wire[lower:upper].decode("utf-16-le"), "\n\n", widget["terminal"], "\n\n")
            )
            lower = upper
        parts.append(wire[lower:].decode("utf-16-le"))
    except UnicodeError:
        return text
    return "".join(parts)
