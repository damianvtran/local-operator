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
overrides only that method; parser control flow and exit statuses stay argparse's.

WHAT IT RECOGNISES. The stdlib's invalid choices, required arguments/groups,
unrecognized/ambiguous options, nargs failures, int/float conversion failures,
mutually exclusive conflicts, explicit values on zero-argument options, and
argument/error wrappers. Unknown third-party validator bodies pass through
unchanged: guessing at their prose could rewrite user data. Our own seconds
validator and exec positional refusal use keys at their source.

Recognition matches argparse's templates, leaving unknown bodies untouched, so a
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
from local_operator.i18n.keys import cli_failure_details as _d
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
    # Match whole stdlib templates, not substrings: quoted user values may
    # themselves contain error-like text. Never rebuild option/choice tokens.
    if message.startswith("unrecognized arguments: "):
        return help_text(
            _d.unrecognized(arguments=message[len("unrecognized arguments: ") :]), locale
        )
    match = re.fullmatch(r"ambiguous option: (.+?) could match (.+)", message, re.DOTALL)
    if match:
        return help_text(_d.ambiguous(option=match[1], matches=match[2]), locale)
    match = re.fullmatch(r"one of the arguments (.+) is required", message, re.DOTALL)
    if match:
        return help_text(_d.required_group(names=match[1]), locale)
    # The wrapper belongs to argparse even when the body belongs to a plugin
    # or was already localized by our validator. Unknown bodies stay verbatim.
    match = re.fullmatch(r"argument (.+?): (.+)", message, re.DOTALL)
    if match:
        body = _localise_argument_body(match[2], locale)
        return help_text(
            _d.argument(argument=match[1], message=body if body is not None else match[2]), locale
        )
    return message


def _localise_argument_body(message: str, locale: str | None) -> str | None:
    """Recognize only argparse-owned bodies; arbitrary validators are opaque."""
    exact = {
        "expected one argument": _d.expected_one,
        "expected at most one argument": _d.expected_optional,
        "expected at least one argument": _d.expected_some,
    }
    if message in exact:
        return help_text(exact[message](), locale)
    match = re.fullmatch(r"expected ([0-9]+) arguments?", message)
    if match:
        return help_text(_d.expected_count(count=int(match[1])), locale)
    match = re.fullmatch(r"invalid (int|float) value: (.+)", message, re.DOTALL)
    if match:
        return help_text(_d.invalid_number(type=match[1], value=match[2]), locale)
    prefix = "not allowed with argument "
    if message.startswith(prefix):
        return help_text(_d.conflict(argument=message[len(prefix) :]), locale)
    prefix = "ignored explicit argument "
    if message.startswith(prefix):
        return help_text(_d.ignored_value(value=message[len(prefix) :]), locale)
    return None


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
