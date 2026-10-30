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
