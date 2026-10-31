"""Read-only locale bootstrap for argparse's pre-startup help.

Help/version exit before normal startup may repair config, load credentials or
activate a process-wide locale. Keep that boundary independent of the manager:
a malformed editor save must never be renamed merely to display argument help.

The reader itself lives in :func:`local_operator.i18n.boot.configured_language`
— the boot activation needs the same read for the same reason (a refused or
help-only run must create nothing), and two copies of "read ``language`` without
constructing a manager" is how one of them grows a side effect later.
"""

from __future__ import annotations

from typing import Any

from .resolve import resolve_language


def help_locale() -> str:
    """Resolve eligible help copy without construction or config side effects.

    The shared reader already absorbs YAML/I/O failures, but UTF-8 decoding is
    outside that contract; ``configured_language`` absorbs that too. It always
    returns a STRING for the resolver: ``None`` would invoke the implicit
    ConfigManager fallback and could rename the very file help must leave alone.
    """
    from .boot import configured_language

    return resolve_language(configured_language())


def help_text(message: Any, locale: str | None = None) -> str:
    """Render one key for an argparse surface in the help locale.

    ``locale`` is the snapshot ``build_cli_parser`` already resolved; a standalone
    caller (a test, or a failure raised after the build) passes ``None`` and
    resolves fresh. This is the explicit-locale render the module docstring
    requires: ``tr()`` would read the PROCESS locale, which a ``--help`` or usage
    failure has not activated yet, so a localised exec/skills line would still
    paint English there.
    """
    return str(message.envelope(locale=locale or help_locale())["text"])
