"""``lop team-bridge avatar publish|clear|status`` — the agent-callable path.

WHY A CLI AT ALL. The picture is published by the owner's own machine, at the
status boundary, with no dialog: the local definition is the source of truth.
That covers the ordinary case, and it covers nothing else — an agent working in
a shell has no dialog to click, and a person whose boundary has not run yet
needs to say "now" (an explicit save) or "remove it" (a clear). Both are one
thin wrapper over :mod:`local_operator.team_bridge.avatar_publish`, so this
module holds no rules of its own: every guard, comparison and refusal lives in
the module the status boundary uses, and the CLI cannot drift from it.

    lop team-bridge avatar status    what the hub holds vs what this machine has
    lop team-bridge avatar publish   publish this machine's picture now
    lop team-bridge avatar clear     remove the HUB copy (local files untouched)

Exit code 0 means the requested state holds; 1 means it does not, and the
reason is printed. ``status`` is read-only and contacts the hub with exactly the
status call every other surface makes.

Terminal copy here is deliberately plain English: it is developer-facing CLI
output naming endpoints, digests and states — technical identifiers by the
i18n contract, not customer copy. The extraction tranche for the ``lop`` CLI's
help strings is tracked with the ``skills`` block's, not per command.
"""

from __future__ import annotations

import argparse
import asyncio
import json
from typing import Any

from local_operator.team_bridge import avatar_publish

#: What the CLI can be asked to do, in the order a person meets them.
ACTIONS = ("status", "publish", "clear")


def configure_parser(parser: argparse.ArgumentParser) -> None:
    """Attach ``avatar`` and its three actions to the ``team-bridge`` parser."""

    actions = parser.add_subparsers(dest="team_bridge_command", required=True)
    avatar = actions.add_parser(
        "avatar",  # i18n: ignore CLI command name
        # i18n: ignore developer CLI help (extraction tranche tracked)
        help="The chief of staff's picture as teammates see it on the bridge",
    )
    avatar_actions = avatar.add_subparsers(dest="avatar_command", required=True)
    avatar_actions.add_parser(
        "status",
        # i18n: ignore developer CLI help (extraction tranche tracked)
        help="What the hub holds and what this machine would publish",
    )
    avatar_actions.add_parser(
        "publish",
        # i18n: ignore developer CLI help (extraction tranche tracked)
        help="Publish this machine's picture now, whatever the hub holds",
    )
    avatar_actions.add_parser(
        "clear",
        # i18n: ignore developer CLI help (extraction tranche tracked)
        help="Remove the hub's copy; local agent files are untouched",
    )
    for action in avatar_actions.choices.values():
        action.add_argument(
            "--json",
            action="store_true",
            # i18n: ignore developer CLI help (extraction tranche tracked)
            help="Print the operation's answer as JSON",
        )
        action.add_argument(
            "--tenant",
            default=None,
            # i18n: ignore developer CLI help (extraction tranche tracked)
            help="Organization id; defaults to the one this machine selected",
        )


def main(args: argparse.Namespace) -> int:
    """Run one avatar command and print its answer. Returns the exit code."""

    try:
        answer = asyncio.run(_run(args))
    except Exception as exc:  # noqa: BLE001 - a CLI reports the refusal it was given
        code = getattr(exc, "code", None)
        print(f"team-bridge avatar: {type(exc).__name__}{f' ({code})' if code else ''}")
        return 1
    if getattr(args, "json", False):
        print(json.dumps(answer, ensure_ascii=False, indent=2))
    else:
        print(_sentence(answer))
    return 0 if answer.get("ok", True) else 1


def _client_for(root: Any) -> Any:
    """The bridge client this command uses. A named seam: tests inject a fake."""

    from local_operator.team_bridge.client import TeamBridgeClient

    return TeamBridgeClient(root)


async def _run(args: argparse.Namespace) -> dict[str, Any]:
    from local_operator.paths import config_dir as resolve_config_dir

    root = resolve_config_dir()
    client = _client_for(root)
    tenant_id = args.tenant
    if tenant_id is None:
        # The row every surface would select for this machine: the same rule the
        # status discipline uses, so the CLI cannot act on a different
        # organization than the one the app is showing.
        from local_operator.team_bridge.client import selected_tenant_id

        tenant_id = selected_tenant_id(await client.status())
        if tenant_id is None:
            return {"ok": False, "action": "none", "detail": "no organization is enrolled"}
    action = args.avatar_command
    if action == "status":
        return _report("status", avatar_publish.plan_avatar(await client.status(), tenant_id, root))
    if action == "publish":
        return _report(
            "publish", await avatar_publish.publish_agent_avatar(client, tenant_id, root)
        )
    if action == "clear":
        return _report("clear", await avatar_publish.clear_agent_avatar(client, tenant_id))
    raise ValueError(f"unknown avatar action: {action}")


def _report(action: str, plan: avatar_publish.AvatarPlan) -> dict[str, Any]:
    """The CLI's answer for one plan. Digests and ids only — never a local path."""

    if action == "publish":
        # "Already the same picture" is the hub's 204 arm, and it is a success:
        # the requested state (hub holds this file) holds either way.
        ok = plan.endpoint_id is not None and plan.digest is not None
    elif action == "clear":
        ok = plan.endpoint_id is not None
    else:
        ok = True
    return {
        "ok": bool(ok),
        "action": action,
        "planned": plan.action,
        "endpoint_id": plan.endpoint_id,
        "agent_id": plan.agent_id,
        "name": plan.name,
        "local_sha256": plan.digest,
        "local_bytes": len(plan.data) if plan.data is not None else None,
        "hub_sha256": plan.hub_sha256,
        "hub_hash": plan.hub_hash,
    }


def _sentence(answer: dict[str, Any]) -> str:
    """One line a person can read, plus the facts that make it checkable."""

    if not answer["ok"]:
        return f"team-bridge avatar {answer['action']}: {answer.get('detail') or 'not done'}"
    hub = answer["hub_sha256"] or "nothing"
    local = answer["local_sha256"] or "nothing"
    return (
        f"team-bridge avatar {answer['action']}: {answer['planned']} "
        f"(endpoint {answer['endpoint_id'] or '-'}, agent {answer['agent_id'] or '-'}, "
        f"local {local}, hub {hub})"
    )


__all__ = ["ACTIONS", "configure_parser", "main"]
