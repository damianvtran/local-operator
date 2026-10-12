"""Read-only locale bootstrap for argparse's pre-startup help.

Help/version exit before normal startup may repair config, load credentials or
activate a process-wide locale. Keep that boundary independent of the manager:
a malformed editor save must never be renamed merely to display argument help.
"""

from __future__ import annotations

from local_operator.config import read_config_values
from local_operator.paths import config_dir

from .resolve import DEFAULT_LANGUAGE, resolve_language


def help_locale() -> str:
    """Resolve eligible help copy without construction or config side effects.

    The shared reader already absorbs YAML/I/O failures, but UTF-8 decoding is
    outside that contract. An undecodable file is equally unavailable here.
    Always pass a STRING to the resolver: ``None`` would invoke its implicit
    ConfigManager fallback and could rename the very file help must leave alone.
    """
    try:
        values = read_config_values(config_dir())
    except UnicodeDecodeError:
        values = None
    language = values.get("language") if values is not None else None
    return resolve_language(language if isinstance(language, str) else DEFAULT_LANGUAGE)
