"""Localised argparse FAILURE messages: the parser's ``error()`` path.

WHY A SUBCLASS. argparse composes its failure sentences from English templates
inside the standard library (``the following arguments are required: …``,
``invalid choice: … (choose from …)``) and prints them as
``<prog>: error: <message>``. The harness's CLI help is keyed and rendered in the
user's language, but those failure lines were not, so a Vietnamese or Hindi user
who mistyped a subcommand got an English error under a localised ``--help``.
Patching ``argparse`` internals is not an option; the one public seam is
``ArgumentParser.error(message)``, which every failure funnels through (argparse
catches its own ``ArgumentError`` and calls it with ``str(err)``). This subclass
overrides only that method.

WHAT IT RECOGNISES. Exactly the shapes the user can cause by typing and that carry
no other copy of their own:

* an unknown subcommand or a bad flag value (``invalid choice``),
* a missing required argument (``the following arguments are required``),
* the final ``<prog>: error: …`` line itself.

Anything else — ``unrecognized arguments``, ``expected one argument``,
``ambiguous option``, a ``type=`` validator's own message — passes through as
argparse (or the validator's author) wrote it, unchanged. Recognition is a match on
argparse's template, and a message that does not match is never rewritten, so a
future CPython wording change degrades to English, not to a wrong sentence. The
templates are stable across the supported 3.12+ range; 3.14 changed how the
CHOICES are rendered (``'a', 'b'`` instead of ``a, b``), which is why the choices
tail is captured verbatim rather than rebuilt: the technical tokens stay as argparse
printed them. ``tests/unit/i18n/test_cli_failures_slice.py`` pins the English bytes
and fails if a template stops matching.

The locale is resolved when the failure happens (a config read with no side
effects, ``cli_help.help_locale``), because subparsers are built by argparse itself
with only its own keyword arguments and cannot be handed the build's snapshot.
"""

from __future__ import annotations

import argparse
import re
import sys
from typing import Any

from local_operator.i18n.cli_help import help_text
from local_operator.i18n.keys import cli_failures as _f

# ``argument --hosting: invalid choice: 'zzz' (choose from a, b)``. The argument
# name is always present for this shape (argparse raises it from an action).
_INVALID_CHOICE = re.compile(
    r"argument (?P<argument>.+?): invalid choice: (?P<value>.+?) \(choose from (?P<choices>.*)\)\Z",
    re.DOTALL,
)
_REQUIRED_PREFIX = "the following arguments are required: "


def localise_failure(message: str, locale: str | None = None) -> str:
    """Return ``message`` rendered in the help locale when it is a known shape."""
    match = _INVALID_CHOICE.match(message)
    if match:
        return help_text(
            _f.invalid_choice(
                argument=match["argument"], value=match["value"], choices=match["choices"]
            ),
            locale,
        )
    if message.startswith(_REQUIRED_PREFIX):
        return help_text(_f.required(names=message[len(_REQUIRED_PREFIX) :]), locale)
    return message


class LocalizedArgumentParser(argparse.ArgumentParser):
    """``ArgumentParser`` whose failure line follows the user's language.

    ``add_subparsers`` builds children with ``type(parser)``, so every subcommand
    parser inherits this behaviour without being touched.
    """

    def error(self, message: str) -> Any:  # argparse types this NoReturn
        self.print_usage(sys.stderr)
        locale_message = localise_failure(message)
        self.exit(
            2,
            help_text(_f.error_line(prog=self.prog, message=locale_message)) + "\n",
        )

    def fail_invalid_command(self, group: str, command: object) -> Any:
        """Refuse a command group invoked without a valid verb (localised)."""
        self.error(help_text(_f.invalid_command(group=group, command=str(command))))
