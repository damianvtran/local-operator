"""Opt-in plain status paint prototype, deliberately not wired into global tr.

Only the owned tui.status namespace is admitted. Existing TUI callers continue
using their raw translation strings until renderer evidence and audited locale
eligibility permit adoption. The Rich Text here is paint-only; callers must keep
Msg/raw text separately for semantics, logs and clipboard/receipt paths.
"""

from __future__ import annotations

from typing import Sequence

from rich.text import Text

from local_operator.i18n.display import TechnicalSpan, isolated_status_text
from local_operator.i18n.messages import Msg, render_record


def status_display(
    message: Msg,
    *,
    locale: str | None = None,
    technical: Sequence[TechnicalSpan] = (),
) -> Text:
    """ONE plain status boundary: no markup parsing, ANSI decoding or reversal.

    Hints must be explicit owner-declared offsets of complete technical tokens
    in the post-format raw record. This is not automatic coverage of placeholders
    or wire.error/SDK consumers; span-aware ICU adoption is intentionally split.
    """
    if not message.code.startswith("tui.status."):
        raise ValueError("status display only accepts the owned tui.status namespace")
    record = render_record(message.code, message.params, locale=locale)
    return Text(isolated_status_text(record, technical))
