"""``lop team-bridge`` argument registration — STDLIB-ONLY, on every ``lop`` start.

Kept OUT of ``local_operator.team_bridge.connector`` on purpose: that package's
``__init__`` imports the connector lock (and with it ``wakes.lock``), and
``cli.build_parser`` runs this on every invocation, ``--version`` and
``--help`` included. The verbs live in
:mod:`local_operator.team_bridge.connector.cli` (``receiver``) and
:mod:`local_operator.team_bridge.setup_cli` (``setup``), imported only once a
``team-bridge`` verb is dispatched (the ``tunnels/arguments.py`` split).
"""

from __future__ import annotations

import argparse
from typing import Any

from local_operator.i18n.keys import cli_team_bridge as _k

#: Lines ``logs`` shows by default: a screenful, enough for the last few loop
#: iterations' content-free ``stage=… code=…`` records.
DEFAULT_LOG_LINES = 50


def add_parser(subparsers: Any, parent_parser: argparse.ArgumentParser | None = None) -> None:
    """Register ``lop team-bridge setup`` and ``lop team-bridge receiver <verb>``.

    Consent is deliberately NOT part of this group: it is granted in the app
    (#3022's bearer boundary). See :mod:`local_operator.team_bridge.setup_cli`.

    Two children: ``setup`` (a read-only readiness check plus receiver placement,
    :mod:`local_operator.team_bridge.setup_cli`) and ``receiver`` (the service
    lifecycle, :mod:`local_operator.team_bridge.connector.cli`). The help text
    uses the product name ("agent messages") the rest of the product shows.
    """

    from local_operator.i18n.cli_help import help_locale as resolve_help_locale

    locale = resolve_help_locale()

    def text(message: Any) -> str:
        return str(message.envelope(locale=locale)["text"])

    parents = [parent_parser] if parent_parser is not None else []
    group = subparsers.add_parser("team-bridge", help=text(_k.help_group()), parents=parents)
    group_actions = group.add_subparsers(dest="team_bridge_command")
    # NO FLAGS BUT --json, and that is the point: this verb cannot accept
    # consent, name an account, pick a tenant or designate a session (see
    # setup_cli's module docstring for why a terminal path to consent was
    # removed rather than hardened). It reads what the app committed and places
    # the receiver when that decision is already present.
    setup = group_actions.add_parser("setup", help=text(_k.help_setup()))
    setup.add_argument("--json", action="store_true", help=text(_k.help_json()))
    receiver = group_actions.add_parser("receiver", help=text(_k.help_receiver()))
    verbs = receiver.add_subparsers(dest="receiver_command")
    for name, help_message in (
        ("status", _k.help_status()),
        ("install", _k.help_install()),
        ("remove", _k.help_remove()),
    ):
        child = verbs.add_parser(name, help=text(help_message))
        child.add_argument("--json", action="store_true", help=text(_k.help_json()))
    logs = verbs.add_parser("logs", help=text(_k.help_logs()))
    logs.add_argument("--lines", type=int, default=DEFAULT_LOG_LINES, help=text(_k.help_lines()))
