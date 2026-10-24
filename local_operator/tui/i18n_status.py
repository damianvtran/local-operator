"""Render-only bidi projection for localized TUI copy.

``Msg``/raw rendering remains the semantic representation used by logs, wire
messages and clipboard flows. These helpers add isolates only when a caller is
constructing terminal paint; they never reorder or shape text themselves.
"""

from __future__ import annotations

import unicodedata
from typing import Any, Callable, Mapping, Sequence

from rich.text import Text

from local_operator.i18n.display import TechnicalSpan, isolated_status_text
from local_operator.i18n.messages import Msg, RenderedMessage, render_record
from local_operator.i18n.text import active_locale, tr_record

_RLI, _LRI, _PDI = "\u2067", "\u2066", "\u2069"
_PROJECTION_CONTROLS = "\u2066\u2067\u2069"


def strip_display_controls(text: str) -> str:
    """Remove only this renderer's isolates at a semantic/copy boundary."""
    return text.translate({ord(control): None for control in _PROJECTION_CONTROLS})


def _occurrence_spans(text: str, value: str) -> list[TechnicalSpan]:
    spans: list[TechnicalSpan] = []
    start = 0
    while (offset := text.find(value, start)) >= 0:
        spans.append(TechnicalSpan(offset, offset + len(value), value))
        start = offset + len(value)
    return spans


def technical_literal_spans(
    record: RenderedMessage, literals: Sequence[str]
) -> tuple[TechnicalSpan, ...]:
    """Locate owner-declared technical tokens authored inside catalogue prose."""
    if any(not literal for literal in literals):
        raise ValueError("technical display literals must be non-empty")
    return tuple(
        sorted(
            (span for literal in literals for span in _occurrence_spans(record.text, literal)),
            key=lambda span: span.start,
        )
    )


def balance_display_line(line: Text) -> None:
    """Close isolates left open by a head-preserving Rich single-line fit.

    Rich truncation counts these controls as zero cells but may place the
    ellipsis before their closing PDI. Styles and raw storage stay untouched;
    only the final painted line receives the missing delimiters.
    """
    depth = 0
    for char in line.plain:
        if char in (_RLI, _LRI):
            depth += 1
        elif char == _PDI:
            depth -= 1
    if depth < 0:
        raise ValueError("display line has an unmatched closing isolate")
    if depth:
        line.append(_PDI * depth)


def technical_parameter_spans(
    record: RenderedMessage,
    params: Mapping[str, Any],
    names: Sequence[str],
) -> tuple[TechnicalSpan, ...]:
    """Locate complete, caller-classified string parameters after ICU formatting.

    Parameter names are explicit because ASCII alone does not make a human name
    or free prose technical. Every occurrence is isolated, which covers a value
    reused by nested/select placeholders without maintaining hand-built offsets.
    """
    spans: list[TechnicalSpan] = []
    for name in names:
        if name not in params:
            continue
        value = params[name]
        if not isinstance(value, str) or not value:
            raise ValueError(f"technical display parameter {name!r} must be a non-empty string")
        occurrences = _occurrence_spans(record.text, value)
        found = bool(occurrences)
        spans.extend(occurrences)
        if not found:
            raise ValueError(f"technical display parameter {name!r} is absent from rendered copy")
    return tuple(sorted(spans, key=lambda span: span.start))


def _record_display(
    record: RenderedMessage,
    message: Msg,
    technical: Sequence[TechnicalSpan],
    technical_params: Sequence[str],
    technical_literals: Sequence[str],
) -> Text:
    if record.source_locale != "ur":
        return Text(record.text)
    spans = (
        tuple(technical)
        + technical_parameter_spans(record, message.params, technical_params)
        + technical_literal_spans(record, technical_literals)
    )
    return Text(isolated_status_text(record, spans))


def display_tr(
    namespace: str,
    *,
    technical_params: Sequence[str] = (),
    technical_literals: Sequence[str] = (),
) -> Callable[[Msg], str]:
    """Build the conventional per-namespace resolver for terminal paint."""
    resolve = tr_record(namespace)
    classified = tuple(technical_params)
    literals = tuple(technical_literals)

    def display(message: Msg) -> str:
        return _record_display(resolve(message), message, (), classified, literals).plain

    return display


def message_display(
    message: Msg,
    *,
    locale: str | None = None,
    technical: Sequence[TechnicalSpan] = (),
    technical_params: Sequence[str] = (),
    technical_literals: Sequence[str] = (),
) -> Text:
    """Paint one catalogued message with source-aware Urdu and LTR islands."""
    record = render_record(message.code, message.params, locale=locale or active_locale())
    return _record_display(record, message, technical, technical_params, technical_literals)


def technical_display(text: str, *, locale: str | None = None) -> Text:
    """Paint complete technical lines as LTR, including code/diff whitespace.

    Inline catalogue parameters keep the stricter complete-token validator;
    standalone terminal content may contain spaces and unbalanced diff markers.
    Existing controls are rejected so nested paint cannot double-isolate them.
    """
    source = locale or active_locale()
    if source != "ur" or not text:
        return Text(text)
    if any(
        char
        in "\u061c\u200e\u200f\u202a\u202b\u202c\u202d\u202e\u2066\u2067\u2068\u2069\u2028\u2029"
        or (unicodedata.category(char) == "Cc" and char != "\n")
        for char in text
    ):
        raise ValueError("technical display requires plain text without control sequences")
    return Text("\n".join("\u2066" + line + "\u2069" for line in text.split("\n")))


def localized_display(
    text: str,
    *,
    locale: str | None = None,
    source_locale: str | None = None,
    technical: Sequence[TechnicalSpan] = (),
    technical_literals: Sequence[str] = (),
) -> Text:
    """Paint an already-rendered TUI string without changing its stored value.

    Registry and palette objects intentionally remain plain strings for their
    non-TUI consumers. Their final TUI paint uses this projection; callers that
    know a fallback's actual source pass it, otherwise the active locale is the
    source of the already-resolved registry copy.
    """
    source = source_locale if source_locale is not None else (locale or active_locale())
    if source != "ur":
        return Text(text)
    record = RenderedMessage(text, source)
    spans = tuple(technical) + technical_literal_spans(record, technical_literals)
    return Text(isolated_status_text(record, spans))


def status_display(
    message: Msg,
    *,
    locale: str | None = None,
    technical: Sequence[TechnicalSpan] = (),
) -> Text:
    """Compatibility entry point for the original status-only prototype."""
    if not message.code.startswith("tui.status."):
        raise ValueError("status display only accepts the owned tui.status namespace")
    return message_display(message, locale=locale, technical=technical)
