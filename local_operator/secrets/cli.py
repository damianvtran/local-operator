"""``lop secret`` — argument registration for the encrypted store (design §5.1).

`AGENTS.md`'s tool-surface footprint ladder makes a skill plus ``bash`` rung 2
and a new core tool rung 5, so the CLI carries the full verb set and the agent
tool (PR 3) stays deliberately thin.

This module is the STDLIB-ONLY half: ``cli.py`` imports it on every ``lop``
invocation to build the parser, so nothing here may reach for the crypto stack
or sqlite3. The verbs live in :mod:`local_operator.secrets.handlers`, which
:func:`main` imports once a ``secret`` verb has actually been dispatched.

**The hard requirement of that half is stdout purity on ``get``.** The whole
point of the store is that an agent writes

.. code-block:: bash

    curl -H "Authorization: Bearer $(lop secret get GITHUB_TOKEN)" ...

and the value crosses a pipe into the child's argv without ever entering the
transcript. ``$( )`` strips trailing newlines but nothing else: one banner
line, one ANSI colour code, one progress note on stdout and every consumer
silently receives a corrupted credential — which fails as a confusing 401 from
a remote service, not as an error anyone traces back to here. So ``get``
writes the exact stored bytes to the stdout BUFFER and nothing else, every
other verb writes its human output to stdout only in non-value form, and every
diagnostic goes to stderr. ``tests/unit/secrets/test_cli.py`` asserts the
exact bytes.

``set`` reads the value from STDIN and there is deliberately no ``--value``
flag: argv is readable by any same-uid process through ``ps`` and
``KERN_PROCARGS2`` (design §2.2, spike 2), so a value on the command line is a
value already leaked.
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

from local_operator.secrets.errors import KINDS

#: Default variable name exported by ``lop secret file``. Named for the
#: overwhelmingly common case — a Google service-account JSON, of which the
#: operator has nine — while ``--env-var`` covers everything else.
DEFAULT_FILE_ENV_VAR = "GOOGLE_APPLICATION_CREDENTIALS"


def add_parser(subparsers: Any) -> None:
    """Register ``lop secret`` and its verbs.

    Stdlib-only and importable without touching the crypto stack: this runs on
    every ``lop`` invocation including ``--version``, and
    ``tests/unit/test_import_graph.py`` pins that the CLI's import graph stays
    free of heavy dependencies. That is why ``KINDS`` comes from ``errors``
    rather than from ``store``, and why the verbs themselves live in
    :mod:`local_operator.secrets.handlers`, which :func:`main` imports only
    once a verb has actually been dispatched.
    """
    parser = subparsers.add_parser(
        "secret",
        help="Encrypted long-term secrets an agent can retrieve without a prompt",
    )
    actions = parser.add_subparsers(dest="secret_command")

    get_parser = actions.add_parser(
        "get", help="Write a secret's value to stdout with no trailing newline"
    )
    get_parser.add_argument("name")

    set_parser = actions.add_parser("set", help="Store a NEW secret; the value is read from stdin")
    set_parser.add_argument("name")
    set_parser.add_argument("--description", default="", help="Human note shown by list")
    set_parser.add_argument("--kind", choices=KINDS, default="string")
    set_parser.add_argument(
        "--from-file",
        type=Path,
        help="Read the value from this file instead of stdin (the file is not removed)",
    )

    update_parser = actions.add_parser(
        "update", help="Replace an existing secret's value; read from stdin"
    )
    update_parser.add_argument("name")
    update_parser.add_argument("--description", default=None)
    update_parser.add_argument("--from-file", type=Path)

    list_parser = actions.add_parser("list", help="Names and descriptions; never values")
    list_parser.add_argument("--json", action="store_true")

    describe_parser = actions.add_parser("describe", help="Metadata for one secret; never a value")
    describe_parser.add_argument("name")
    describe_parser.add_argument("--json", action="store_true")

    for name in ("rm", "delete"):
        remove_parser = actions.add_parser(name, help="Remove a secret permanently")
        remove_parser.add_argument("name")
        remove_parser.add_argument(
            "--yes", action="store_true", help="Do not prompt for confirmation"
        )

    actions.add_parser("rotate", help="Re-seal every record under a new master key")

    status_parser = actions.add_parser("status", help="Where the store is and what mode it is in")
    status_parser.add_argument("--json", action="store_true")

    audit_parser = actions.add_parser("audit", help="Show or verify the audit chain")
    audit_parser.add_argument(
        "--verify", action="store_true", help="Check the hash chain and report the first break"
    )
    audit_parser.add_argument("--limit", type=int, default=20)
    audit_parser.add_argument("--json", action="store_true")

    file_parser = actions.add_parser(
        "file", help="Materialise a file secret at a private path for one command"
    )
    file_parser.add_argument("name")
    file_parser.add_argument("--env-var", default=DEFAULT_FILE_ENV_VAR)
    # nargs="*" rather than argparse.REMAINDER: REMAINDER starts consuming at
    # the first token after the positional, so `file NAME --env-var VAR -- cmd`
    # swallowed `--env-var` into the command list and silently used the default
    # variable name. With "*", argparse still stops option parsing at the `--`
    # separator, so flags belonging to the CHILD command (`-- curl -H ...`)
    # arrive intact while our own options are parsed before it.
    file_parser.add_argument("command", nargs="*")

    run_parser = actions.add_parser("run", help="Run a command with named secrets in its env")
    run_parser.add_argument(
        "--secret",
        action="append",
        default=[],
        metavar="NAME[=VAR]",
        help="Secret to export; repeatable. NAME=VAR exports it under a different variable",
    )
    # Same reason as `file` above: REMAINDER would swallow `--secret`.
    run_parser.add_argument("command", nargs="*")

    # The two passphrase-tier verbs. Opt-in by design: the operator rejected
    # admin-gated and per-access-prompting stores, so the no-prompt keyfile
    # default ships as the default and `harden` is an upgrade the operator
    # chooses. Neither reads its passphrase from argv, for the same reason
    # `set` does not read a value there.
    actions.add_parser(
        "harden",
        help="Wrap the master key with a passphrase; unlock once per boot afterwards",
    )
    actions.add_parser(
        "unlock",
        help=(
            "Unlock a hardened store for this boot; authorizes THIS terminal to read "
            "secrets until the shell exits"
        ),
    )

    broker_parser = actions.add_parser("broker", help="Inspect or control the secret broker")
    broker_actions = broker_parser.add_subparsers(dest="broker_command")
    broker_actions.add_parser("status", help="Is a broker running, and what does it hold?")
    broker_actions.add_parser("start", help="Start a broker if one is not already running")
    broker_actions.add_parser("stop", help="Stop the running broker")
    broker_actions.add_parser(
        "restart",
        help="Stop the running broker and start a fresh one (use after `lop-update`)",
    )
    broker_actions.add_parser(
        "run", help="Run a broker in the foreground (for debugging; normally started on demand)"
    )


def main(args: argparse.Namespace) -> int:
    """Entry point for ``lop secret``; see :mod:`local_operator.secrets.handlers`.

    A deliberately thin shim. The import is function-local because this module
    is imported during argument registration on every ``lop`` invocation, and
    the handlers pull in the crypto stack and sqlite3 — a cost that belongs to
    the run that actually uses a secret, not to ``lop --version``.
    """
    from local_operator.secrets.handlers import dispatch

    return dispatch(args)
