"""``lop agents picture set|remove|show`` -- an agent's profile picture from the shell.

WHY A CLI AT ALL: the desktop editor is the operator's path, but an AGENT (Aida
included) sets pictures by running a command through ``bash``, and a scripted
setup does the same. It calls the same registry write the HTTP route does
(``AgentRegistry.set_agent_avatar``), so the gate and re-encode are identical.

A separate module, not more of ``cli.py``: that file carries a hard-coded-copy
ceiling (``i18n/baseline.json``), and this command's copy is keyed from day one
(``cli.agents_picture``). It writes from ANOTHER process than a running server,
so the in-process ``agent_avatar.subscribe`` hook does not fire there; the
desktop feed's avatar probe reconciles on ``agent.yml`` instead.

Exit codes: 0 done (including "nothing to remove"), 1 refused or not found.
Refusals print the catalogued sentence to stderr and, under ``show``/``--json``,
the machine reason word, so an agent can branch on it.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

from local_operator.i18n.keys import cli_agents_picture as _p
from local_operator.i18n.text import tr

_t = tr("cli.agents_picture")
#: Refusal sentences are the wire catalogue's, shared with the HTTP route.
_tw = tr("wire.errors.agent_avatar")


def add_parser(agents_subparsers: Any, *, parent_parser: Any, help_locale: str | None) -> None:
    """Register ``picture`` under ``lop agents`` (called from ``cli.build_parser``)."""

    def _h(message: Any) -> str:
        return message.envelope(locale=help_locale)["text"]

    picture = agents_subparsers.add_parser(
        "picture", help=_h(_p.help_group()), parents=[parent_parser]
    )
    verbs = picture.add_subparsers(dest="picture_command", required=True)
    set_parser = verbs.add_parser("set", help=_h(_p.help_set()), parents=[parent_parser])
    set_parser.add_argument("agent", help=_h(_p.help_agent()))
    set_parser.add_argument("file", help=_h(_p.help_file()))
    remove_parser = verbs.add_parser("remove", help=_h(_p.help_remove()), parents=[parent_parser])
    remove_parser.add_argument("agent", help=_h(_p.help_agent()))
    show_parser = verbs.add_parser("show", help=_h(_p.help_show()), parents=[parent_parser])
    show_parser.add_argument("agent", help=_h(_p.help_agent()))


def _resolve(agent_registry: Any, ref: str) -> Any:
    """Name first, then id: names are what people and agents type."""
    row = agent_registry.get_agent_by_name(ref)
    if row is not None:
        return row
    try:
        return agent_registry.get_agent(ref)
    except KeyError:
        return None


def _refuse(reason: str) -> int:
    from local_operator import agent_avatar

    sys.stderr.write(
        json.dumps({"error": reason, "message": _tw(agent_avatar.refusal_message(reason))}) + "\n"
    )
    return 1


def run(args: argparse.Namespace, agent_registry: Any) -> int:
    from local_operator import agent_avatar

    row = _resolve(agent_registry, str(args.agent))
    if row is None:
        sys.stderr.write(_t(_p.error_no_agent(agent=str(args.agent))) + "\n")
        return 1
    command = args.picture_command
    if command == "show":
        # The meta JSON exactly as the API carries it (``null`` when unset), so an
        # agent can compare the sha against a listing row.
        payload = {
            "agent_id": row.id,
            "name": row.name,
            "avatar": agent_avatar.meta_json(row.avatar),
        }
        print(json.dumps(payload, indent=2))
        return 0
    if command == "remove":
        removed = agent_registry.clear_agent_avatar(row.id)
        message = _p.remove_done(agent=row.name) if removed else _p.remove_none(agent=row.name)
        print(_t(message))
        return 0
    path = Path(str(args.file)).expanduser()
    try:
        with path.open("rb") as handle:
            # One byte past the cap is enough to refuse as too_large without
            # reading an arbitrarily large file into memory.
            data = handle.read(agent_avatar.INPUT_MAX_BYTES + 1)
    except OSError as exc:
        sys.stderr.write(_t(_p.error_read_file(path=str(path), error=str(exc))) + "\n")
        return 1
    try:
        meta = agent_registry.set_agent_avatar(row.id, data)
    except agent_avatar.AvatarError as exc:
        return _refuse(exc.reason)
    print(_t(_p.set_done(agent=row.name, mime=meta.mime, bytes=str(meta.bytes))))
    return 0
