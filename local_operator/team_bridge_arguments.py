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


def add_parser(
    subparsers: Any,
    parent_parser: argparse.ArgumentParser | None = None,
    *,
    help_locale: str | None = None,
) -> None:
    """Register ``lop team-bridge setup`` and ``lop team-bridge receiver <verb>``.

    Consent is deliberately NOT part of this group: it is granted in the app
    (#3022's bearer boundary). See :mod:`local_operator.team_bridge.setup_cli`.

    Two children: ``setup`` (a read-only readiness check plus receiver placement,
    :mod:`local_operator.team_bridge.setup_cli`) and ``receiver`` (the service
    lifecycle, :mod:`local_operator.team_bridge.connector.cli`). The help text
    uses the product name ("agent messages") the rest of the product shows.
    """

    # One build owns ONE locale snapshot, and ``build_cli_parser`` hands its own
    # to every group it registers. Resolving again here would read the config a
    # second time per start (and break the one-build-one-snapshot contract the
    # i18n help slices pin), so the parameter is the whole point. Standalone
    # callers, which have no build to inherit from, still resolve freshly.
    if help_locale is None:
        from local_operator.i18n.cli_help import help_locale as resolve_help_locale

        help_locale = resolve_help_locale()

    def text(message: Any) -> str:
        return str(message.envelope(locale=help_locale)["text"])

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
    # The agent-picture surface, registered on THIS group rather than as a second
    # top-level parser: argparse refuses a duplicate subparser name, and a
    # duplicate here took the whole `lop` binary down (review R1-1). The verbs
    # live in `team_bridge/cli.py` and are imported only on dispatch.
    #
    # Help strings are the one exception to the keyed-copy rule in this file,
    # stated plainly with the pragma the i18n ratchet expects: the extraction
    # tranche for these strings is not opened here, and inventing eight
    # translations without the independent audit the contract requires would be
    # worse than shipping English. `scripts/i18n/check.py` reads the pragma.
    avatar = group_actions.add_parser(
        "avatar",  # i18n: ignore CLI command name
        # i18n: ignore developer CLI help pending the CLI copy-extraction tranche
        help="The chief of staff's picture as teammates see it on the bridge",
    )
    avatar_actions = avatar.add_subparsers(dest="avatar_command", required=True)
    for name, blurb in (
        ("status", "What the hub holds and what this machine would publish"),
        ("publish", "Publish this machine's picture now, whatever the hub holds"),
        ("clear", "Remove the hub's copy; local agent files are untouched"),
    ):
        child = avatar_actions.add_parser(
            name,  # i18n: ignore CLI command name
            help=blurb,  # i18n: ignore developer CLI help (see the pragma above)
        )
        child.add_argument(
            "--json",
            action="store_true",
            # i18n: ignore developer CLI flag help (see the pragma above)
            help="Print the operation's answer as JSON",
        )
        child.add_argument(
            "--tenant",
            default=None,
            # i18n: ignore developer CLI flag help (see the pragma above)
            help="Organization id; defaults to the one this machine selected",
        )
