"""Advisory routing: a runtime-log record always, console text only for a person.

THE FAILURE THIS EXISTS FOR (2026-10-24). Two true-but-unrelated startup
advisories — the config-drift notice and the duplicate-project-row notice —
were printed to stderr at CLI startup. The desktop app captures a child
``local-operator update``'s streams and builds the modal's diagnosis from the
HEAD of that stream (``installDiagnosisLines``), so those two lines filled the
modal's six-line budget and the updater's own cause — line ~7, *"fresh PyPI
latest is 0.88.4, not 0.89.0; retry lop update"* — was never shown. The
advisory was true; it just answered a question nobody had asked at that
moment.

THE RULE, ONE RULE. An advisory is always recorded on the module logger at
WARNING, so every runtime log that carries warnings carries it. It is rendered
to a console only when that console is a terminal — a person watching a run.
A machine-captured stream (a pipe, a file, a test harness) gets the ordinary
sentences and the machine-readable report, and nothing else.

HOW THE CONSOLE HALF IS ENFORCED. The CLI installs its stderr handler through
:func:`local_operator.logger.configure_console_logging`; that call carries
:func:`console_filter` when the surface is the CLI (see
``configure_cli_logging``). The filter drops records marked with
:data:`ADVISORY_ATTRIBUTE` when the handler's own stream is not a TTY, and
leaves every other record — errors, failures, warnings that are not
advisories — untouched. Filtering at the handler rather than choosing a log
level is deliberate: the record still exists at WARNING for any other handler
(a file the caller attached, ``caplog``, the TUI's log), which is what "always
recorded" has to mean.

DO NOT route a failure through this module. An update that cannot proceed is a
failure and must reach stderr on every surface; only notices that a run can
finish without belong here.
"""

from __future__ import annotations

import logging
from typing import Callable, TextIO

#: The ``LogRecord`` attribute that marks a record as an advisory.
#:
#: Spelled as an attribute rather than a name prefix because the filter must
#: never guess from message text: a genuine error that happens to quote an
#: advisory sentence is still an error.
ADVISORY_ATTRIBUTE = "lop_advisory"


def _is_tty(stream: TextIO) -> bool:
    """Whether ``stream`` is a terminal, treating any unreadable answer as no."""
    try:
        return bool(stream.isatty())
    except (AttributeError, OSError, ValueError):
        return False


def console_filter(stream: TextIO) -> Callable[[logging.LogRecord], bool]:
    """A handler filter: advisories render only when ``stream`` is a terminal.

    Bound to the handler's own stream (the object the handler writes to), not
    to ``sys.stderr`` at filter time: the CLI swaps ``sys.stderr`` in tests and
    the TUI redirects fd 2 under ``file_logging``, and the handler keeps the
    object it was constructed with. Asking that object is asking the actual
    destination.
    """

    def keep(record: logging.LogRecord) -> bool:
        if not getattr(record, ADVISORY_ATTRIBUTE, False):
            return True
        return _is_tty(stream)

    return keep


def emit(message: str, *, logger: logging.Logger, surface: str = "cli") -> None:
    """Record one advisory, and let the console handler decide about the screen.

    ``surface`` keeps the one distinction the pre-existing call sites already
    made: a ``daemon`` has no person watching its stderr (it is a launchd log
    file), so it records at DEBUG exactly as before and never claims a
    console. Every other surface records a WARNING carrying
    :data:`ADVISORY_ATTRIBUTE`, which the CLI's handler renders only on a TTY.
    """
    if surface == "daemon":
        logger.debug("%s", message)
        return
    logger.warning("%s", message, extra={ADVISORY_ATTRIBUTE: True})


__all__ = ["ADVISORY_ATTRIBUTE", "console_filter", "emit"]
