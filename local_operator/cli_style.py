"""ANSI styling and encoding fallbacks for the plain-print CLI paths.

Why its own module, stdlib-only: the credential prompt, the ``config`` and
``credential`` subcommands, the startup preflight and the ``login`` handlers all
print coloured status *directly* — not through Textual — and every one of them
sits on the CLI startup path that ``test_import_graph`` forbids from importing
the provider graph, Textual or asyncio. A raw ``\\033[1;31m`` literal, which is
what every one of these call sites used before, ignores three environments the
tool is routinely run in:

- ``NO_COLOR`` set (the community convention for "emit no colour anywhere"),
- a non-tty stdout (a pipe or a captured CI log), where the escape bytes land
  in the middle of text a scraper is trying to parse, and
- ``TERM=dumb`` or a legacy Windows console that never learned to interpret
  SGR, which renders the literal escapes as visible ``[1;31m`` garbage.

The encoding half exists for the same class of caller: the credential prompt
draws a box-drawing banner, and a stdout whose encoding cannot represent
``─``/``╭`` (``PYTHONIOENCODING=ascii``, a legacy Windows code page) crashed the
prompt with ``UnicodeEncodeError`` before it could ask for anything.
"""

from __future__ import annotations

import os
import sys
from typing import IO

#: SGR parameter strings for the four semantic roles the CLI paints. Kept as
#: bare parameters (not full escapes) so :func:`paint` owns the ``ESC[…m`` /
#: reset framing in exactly one place — a literal reset forgotten at a call
#: site is how colour used to bleed into the line below it.
ERROR = "1;31"
WARNING = "1;33"
SUCCESS = "1;32"
INFO = "1;34"
CYAN = "1;36"


def colour_enabled(stream: IO[str] | None = None) -> bool:
    """Whether ANSI colour should be emitted to ``stream`` (default stdout).

    The three gates, in the order a surprised user would check them: an
    explicit ``NO_COLOR`` opt-out wins over everything (its mere presence, per
    the convention, regardless of value); then the stream must be a real
    terminal, so a redirected pipe or captured log stays plain; then ``TERM``
    must not advertise a terminal that cannot interpret SGR.
    """
    if stream is None:
        stream = sys.stdout
    if os.environ.get("NO_COLOR") is not None:
        return False
    isatty = getattr(stream, "isatty", None)
    if not callable(isatty) or not isatty():
        return False
    if os.environ.get("TERM") == "dumb":
        return False
    return True


def paint(text: str, code: str, *, stream: IO[str] | None = None) -> str:
    """Wrap ``text`` in the SGR ``code`` when colour is enabled for ``stream``.

    Returns the text untouched otherwise, so a call site reads the same whether
    or not the terminal takes colour — the gate lives here, never at the call
    site, which is what keeps a new print path from reintroducing a raw escape.
    """
    if not colour_enabled(stream):
        return text
    return f"\033[{code}m{text}\033[0m"


def can_encode(text: str, stream: IO[str] | None = None) -> bool:
    """Whether ``stream`` (default stdout) can encode ``text`` without error.

    Used by the credential banner to decide between its box-drawing and ASCII
    forms. A ``None``/unknown encoding is treated as capable: the default
    interpreter encoding is UTF-8, and refusing to draw the nice banner on a
    stream that simply did not report its encoding would punish the common case
    for the rare one.
    """
    if stream is None:
        stream = sys.stdout
    encoding = getattr(stream, "encoding", None)
    if not encoding:
        return True
    try:
        text.encode(encoding)
    except (UnicodeEncodeError, LookupError):
        return False
    return True
