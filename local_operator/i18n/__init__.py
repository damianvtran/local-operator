"""Local Operator i18n: runtime, catalogues, resolution and formatting (M0).

The public surface, grouped by job:

* :mod:`~local_operator.i18n.runtime` — parse/render the ICU MessageFormat
  subset the catalogues use;
* :mod:`~local_operator.i18n.format` — locale formatting (numbers, dates,
  times, relative time, durations, byte sizes) over generated `Intl` tables;
* :mod:`~local_operator.i18n.catalogues` — the flat JSON message maps and
  their content hashes;
* :mod:`~local_operator.i18n.messages` — the ``{code, params, text}`` envelope;
* :mod:`~local_operator.i18n.resolve` — what language this process speaks.

Everything here is stdlib-only (operator decision §11.5): the locale data is
generated at build time by ``scripts/i18n/emit.mjs`` and committed, so the
wheel carries data rather than a dependency.
"""

from __future__ import annotations

from . import catalogues, format, messages, resolve, runtime
from .messages import Msg, envelope, msg
from .resolve import (
    DEFAULT_LANGUAGE,
    SUPPORTED_LOCALES,
    resolve_language,
    shipped_locales,
)
from .runtime import (
    MessageError,
    MessageFormatError,
    MessageSyntaxError,
    parse_message,
    render_message,
)

__all__ = [
    "DEFAULT_LANGUAGE",
    "Msg",
    "MessageError",
    "MessageFormatError",
    "MessageSyntaxError",
    "SUPPORTED_LOCALES",
    "catalogues",
    "envelope",
    "format",
    "messages",
    "msg",
    "parse_message",
    "render_message",
    "resolve",
    "resolve_language",
    "runtime",
    "shipped_locales",
]
