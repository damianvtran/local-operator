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
        start = 0
        found = False
        while (offset := record.text.find(value, start)) >= 0:
            found = True
            spans.append(TechnicalSpan(offset, offset + len(value), value))
            start = offset + len(value)
        if not found:
            raise ValueError(f"technical display parameter {name!r} is absent from rendered copy")
    return tuple(sorted(spans, key=lambda span: span.start))


def _record_display(
    record: RenderedMessage,
    message: Msg,
    technical: Sequence[TechnicalSpan],
    technical_params: Sequence[str],
) -> Text:
    if record.source_locale != "ur":
        return Text(record.text)
    spans = tuple(technical) + technical_parameter_spans(record, message.params, technical_params)
    return Text(isolated_status_text(record, spans))


def display_tr(namespace: str, *, technical_params: Sequence[str] = ()) -> Callable[[Msg], str]:
    """Build the conventional per-namespace resolver for terminal paint."""
    resolve = tr_record(namespace)
    classified = tuple(technical_params)

    def display(message: Msg) -> str:
        return _record_display(resolve(message), message, (), classified).plain

    return display


def message_display(
    message: Msg,
    *,
    locale: str | None = None,
    technical: Sequence[TechnicalSpan] = (),
    technical_params: Sequence[str] = (),
) -> Text:
    """Paint one catalogued message with source-aware Urdu and LTR islands."""
    record = render_record(message.code, message.params, locale=locale or active_locale())
    return _record_display(record, message, technical, technical_params)


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
    return Text(isolated_status_text(RenderedMessage(text, source), technical))


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
