"""Semantic Info copy and its *last*, cell-fitted terminal projection.

The clipboard renderer shares messages, never painted text. Keeping provenance
on a str-compatible value lets the existing row fitters measure ordinary cells,
then reuse the shared bidi adapter without attributing English fallback to Urdu.
No snapshot fields, protocol tokens or stored/exported bytes gain bidi controls.
"""
from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass
from typing import Any

from rich.text import Text

from local_operator.i18n.keys import tui_info as keys, wire_session_status as duration_keys
from local_operator.i18n.messages import Msg
from local_operator.i18n.text import tr_record
from local_operator.tui.i18n_status import localized_display


@dataclass(frozen=True)
class _Part:
    text: str
    source: str | None
    literals: tuple[str, ...] = ()


class InfoText(str):
    """Plain semantic text with paint-only source provenance, not an envelope."""

    parts: tuple[_Part, ...]

    def __new__(cls, parts: tuple[_Part, ...]) -> InfoText:
        value = super().__new__(cls, "".join(part.text for part in parts))
        value.parts = parts
        return value

    def __add__(self, other: str) -> InfoText:
        return InfoText(self.parts + _parts(other))

    def __radd__(self, other: str) -> InfoText:
        return InfoText(_parts(other) + self.parts)


def _parts(value: str) -> tuple[_Part, ...]:
    return value.parts if isinstance(value, InfoText) else (_Part(value, None),)


def join_info(separator: str, values: Iterable[str]) -> InfoText:
    parts: list[_Part] = []
    for value in values:
        if parts:
            parts.extend(_parts(separator))
        parts.extend(_parts(value))
    return InfoText(tuple(parts))


_resolve = {name: tr_record(name) for name in (
    "tui.info", "tui.small", "tui.welcome", "wire.session_status",
)}
# Explicit ownership, not ASCII detection: these are IDs and versions,
# key chords and paths. Human labels, free error prose and conversation names
# are deliberately absent. Commands below are protected as whole instructions.
_TECHNICAL_ARGS = frozenset({
    "latest", "latest_known", "version",
    "protocol", "variable", "kind", "stamp", "source_ref",
    "info_copy_key", "copy_key", "INFO_COPY_KEY", "target", "time_strftime",
})
_LITERAL_TOKENS = (
                      '/info',
                      '/reload',
                      '/update',
                      '`lop-update`',
                      'lop-update',
                      'sys.path',
                      'PyPI',
                      'MCP',
                  )


def info_text(message: Msg) -> InfoText:
    namespace = next(name for name in _resolve if message.code.startswith(name + "."))
    record = _resolve[namespace](message)
    literals = [str(value) for name, value in message.params.items()
                if name in _TECHNICAL_ARGS and str(value)]
    literals.extend(token for token in _LITERAL_TOKENS if token in record.text)
    if "pid" in message.params:
        pid = message.params["pid"]
        literals.extend((f"'lop stop --pid {pid}'", f"'/stop {pid}'",
                         f"'lop stop --pid {pid} --force'"))
    literals.extend(("'lop stop --pid N'", "'/stop N'", "'lop stop --pid N --force'"))
    parts: list[_Part] = []
    pending: list[str] = []
    for text, owner in record.parts or ((record.text, None),):
        nested = message.params.get(owner) if owner is not None else None
        if isinstance(nested, InfoText):
            # ICU already selected/reordered this interpolation. Preserve its
            # actual source (including fallback) and whole command ownership.
            if pending:
                parts.append(_Part("".join(pending), record.source_locale, tuple(literals)))
                pending.clear()
            parts.extend(nested.parts)
        else:
            pending.append(text)
    if pending:
        parts.append(_Part("".join(pending), record.source_locale, tuple(literals)))
    return InfoText(tuple(parts))


def info_duration(seconds: float) -> InfoText:
    """Localized human units with Info's unchanged ASCII numeric thresholds.

    Reuse the audited session-status keys, but resolve in the enclosing Info
    locale, not the heavyweight catalogue's independently configured locale.
    None is a missing field, handled by the whole field message, never zero.
    """
    total = int(max(0.0, seconds))
    if total < 60:
        message = duration_keys.duration_seconds(count=total)
    elif total < 3600:
        message = duration_keys.duration_minutes(count=total // 60)
    elif total < 86400:
        message = duration_keys.duration_hours_minutes(
            hours=total // 3600, minutes=(total % 3600) // 60,
        )
    else:
        # Keep ungrouped 1000-day output, matching the legacy formatter.
        message = duration_keys.duration_days_hours(
            days=str(total // 86400), hours=(total % 86400) // 3600,
        )
    return info_text(message)


class InfoRow(Text):
    """A Rich row records semantic spans; projection happens after truncation."""

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        self.copy_parts: list[tuple[int, _Part]] = []
        initial = args[0] if args and isinstance(args[0], InfoText) else None
        super().__init__(*args, **kwargs)
        if initial is not None:
            offset = 0
            for part in initial.parts:
                self.copy_parts.append((offset, part))
                offset += len(part.text)

    def append(self, text: str | Text, style: Any = None) -> InfoRow:
        if isinstance(text, InfoText):
            offset = len(self.plain)
            for part in text.parts:
                self.copy_parts.append((offset, part))
                offset += len(part.text)
        super().append(text, style)
        return self


def copy_slice(value: str, start: int, end: int) -> InfoText:
    """Preserve source when wrapping a note; offsets refer to plain characters."""
    offset = 0
    parts = []
    for part in _parts(value):
        left, right = max(start - offset, 0), min(end - offset, len(part.text))
        if left < right:
            # A command wider than the viewport must fold, but each surviving
            # fragment still belongs to the same technical island.
            literals = tuple(
                part.text[max(left, start):min(right, end)]
                for start, end in _literal_ranges(part)
                if max(left, start) < min(right, end)
            )
            parts.append(_Part(part.text[left:right], part.source, literals))
        offset += len(part.text)
    return InfoText(tuple(parts))


def _literal_ranges(part: _Part) -> Iterable[tuple[int, int]]:
    for literal in part.literals:
        start = 0
        while literal and (start := part.text.find(literal, start)) >= 0:
            yield start, start + len(literal)
            start += len(literal)


def info_wrap_end(value: str, start: int, end: int, width: int) -> int:
    """Keep a fitting RTL technical instruction whole across a cell wrap.

    English's established wrapping is unchanged. Oversized instructions retain
    fragment ownership via copy_slice rather than overflowing the viewport.
    """
    from rich.cells import cell_len

    offset = 0
    for part in _parts(value):
        if part.source == "ur":
            for left, right in _literal_ranges(part):
                left, right = left + offset, right + offset
                if start < left < end < right and cell_len(value[left:right]) <= width:
                    return left
        offset += len(part.text)
    return end


def paint_info_row(row: Text) -> Text:
    """Project only surviving semantic spans, retaining Rich styles and glyphs.

    Rich has already fitted/truncated this row. Insert isolates afterward, so
    neither their zero width nor codepoint padding can affect rung selection.
    English fallback explicitly stays LTR even in an Urdu session.
    """
    if not isinstance(row, InfoRow):
        return row
    output = Text()
    cursor = 0
    for offset, part in row.copy_parts:
        end = min(offset + len(part.text), len(row.plain))
        if offset >= end:
            continue
        output.append_text(row[cursor:offset])
        segment = row[offset:end]
        if part.source == "ur":
            # The adapter validates actual whole tokens; a token shed/cropped by
            # fit is no longer a declared full literal on this painted segment.
            projected = localized_display(segment.plain, source_locale=part.source,
                                          technical_literals=part.literals)
            if projected.plain != segment.plain:
                # All semantic segment ink is homogeneous at Info's row boundary.
                projected.stylize(segment.style)
                if segment.spans:
                    projected.stylize(segment.spans[0].style)
                segment = projected
        output.append_text(segment)
        cursor = end
    output.append_text(row[cursor:])
    return output


def info_plural(count: int, noun: str) -> InfoText:
    """Info's closed noun vocabulary; the legacy CLI plural API stays stable."""
    message = {
        "session": keys.count_session,
        "runtime": keys.count_runtime,
        "trajectory": keys.count_trajectory,
        "subagent": keys.count_subagent,
    }[noun](count=count)
    return info_text(message)


def info_state(code: str) -> str:
    """Translate known human states without rewriting machine snapshot codes.

    Unknown future codes remain diagnostic identifiers, not invented English.
    This fallback makes an older client honest about a newer registry record.
    """
    states = {
        "running": keys.state_running, "starting": keys.state_starting,
        "queued": keys.state_queued, "pausing": keys.state_pausing,
        "paused": keys.state_paused, "completed": keys.state_completed,
        "failed": keys.state_failed, "cancelled": keys.state_cancelled,
        "gone": keys.state_gone, "live": keys.state_live,
        "stale": keys.state_stale, "unknown": keys.state_unknown,
    }
    render = states.get(code or "unknown")
    return info_text(render()) if render else code


def info_leaving(phrase: str) -> str:
    """Known record phrases are protocol discriminants, never stored translations."""
    from local_operator.session.runtime.types import (
        BUILD_DRAIN_PROGRESS_S, LEAVING_FOR_BUILD, LEAVING_FOR_BUILD_OVERDUE,
        LEAVING_ON_SIGNAL, SIGNAL_DRAIN_S, bound_text,
    )
    if phrase == LEAVING_ON_SIGNAL:
        return info_text(keys.leaving_signal(bound=bound_text(SIGNAL_DRAIN_S)))
    if phrase == LEAVING_FOR_BUILD:
        return info_text(keys.leaving_build())
    if phrase == LEAVING_FOR_BUILD_OVERDUE:
        return info_text(keys.leaving_overdue(bound=bound_text(BUILD_DRAIN_PROGRESS_S)))
    return phrase  # External/future record payload, not locally authored prose.


def info_update(phase: str, pair: str) -> str:
    """Mirror update_short's facts without changing its machine/CLI callers."""
    from local_operator.session.runtime.types import UPDATE_FAILED, UPDATING, UPDATING_DONE

    target = pair.split("→", 1)[1].strip() if "→" in pair else ""
    if phase == UPDATING:
        return info_text(keys.update_target(target=target) if target else keys.status_updating())
    if phase == UPDATING_DONE:
        return info_text(
                   keys.update_done_target(target=target) if target else keys.status_updated(),
               )
    if phase == UPDATE_FAILED:
        return info_text(keys.status_update_failed())
    return ""
