"""``lop community …`` — the script surface of the community page (TUI doc, "Commands for scripts").

WHY. A person in a shell needs the same member operations the ``/community``
page offers, with answers a script can branch on: ``--json`` everywhere and the
documented exit codes (``service.EXIT_CODES``). Every rule lives in
:mod:`local_operator.community.service` and the client; this module parses
arguments and prints.

    lop community status                          joined? which guidelines?
    lop community guidelines [--version]          the current guidelines text
    lop community join --accept-guidelines 2026-11
    lop community channels [--filter joined|not-joined|private]
    lop community read '#help' [--limit 50]
    lop community post '#help' --text -           (exit 0/3/4/5/6/7/8)
    lop community cancel <submission-id>          (exit 0 cancelled, 9 already posted)
    lop community leave --yes
                                                  (post/join/leave/cancel/read: exit 10
                                                   inside an agent's shell)

NOT FROM AN AGENT'S SHELL (review R1-1). This command posts as the owner and
reads strangers' text, and it has no approval card: the person typing it IS
the approval. An agent's ``bash`` call carries ``LOCAL_OPERATOR_AGENT_SHELL``
(``agent_shell.in_agent_shell``), and there the verbs in
:data:`AGENT_SHELL_VERBS` exit 10 before any request, naming the bridge ops
that keep the RFC v2.4 §5.9 controls: ``community_post`` asks the owner per
post, ``community_read`` marks the text untrusted and taints the session. A
community post read by an agent must never be able to say "run ``lop community
post …``" and have it publish. The ``team-bridge setup`` precedent: remove the
terminal consent path from the agent rather than harden it. The marker is not
a security boundary (``agent_shell``'s module doc says why); it stops the naive
route, which is the one an injected instruction takes. ``status``,
``guidelines`` and ``channels`` stay available: they show the operator's own
membership, the operator-published guidelines and the directory, not members'
posts, and they change nothing.

TERMINAL SAFETY (review R1-3). Names, purposes, guidelines and post text are
server-sourced and partly stranger-authored, so every text line is printed
through ``strip_control_sequences`` (no OSC 52 clipboard writes, screen
clears or title changes), and ``--json`` escapes the C1 range that
``json.dumps(ensure_ascii=False)`` would otherwise emit raw.

DARK. Hidden from ``lop --help`` (the subparser takes no ``help=``, the
``resume-click`` idiom) and refused unless ``LOCAL_OPERATOR_COMMUNITY=1``
(``community_enabled``), so nothing advertises the feature before the pilot.

EXPLICIT ACCEPTANCE. ``join`` needs ``--accept-guidelines VERSION`` naming the
CURRENT version (decision 7 in the TUI doc: that flag is the explicit act); a
stale version exits 8 with the current one, and nothing is sent.

COPY. Terminal lines here are plain English, the ``team_bridge/cli.py``
precedent for a developer-facing, ``--json``-first CLI; the extraction tranche
for CLI copy is tracked with the other ``lop`` help strings. The interactive
page (``tui/widgets/community_view.py``) is fully keyed.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import sys
from typing import Any

from local_operator.community import service

#: The directory filters the CLI offers, mapped to the contract's
#: ``DirFilter`` (``not-joined`` is the TUI's word for ``public`` + not a
#: member, which the directory answers as ``public`` and we narrow locally).
_FILTERS = {"joined": "joined", "not-joined": "public", "private": "private"}

#: The verbs an agent's shell may not run: everything that acts as the owner
#: or pulls members' posts into the agent's context (see "NOT FROM AN AGENT'S
#: SHELL" above). Each maps to the route the refusal names instead.
AGENT_SHELL_VERBS: dict[str, str] = {
    "post": "use the bridge tool, op=community_post: the owner approves each post",
    "cancel": "posts made through bridge op=community_post are the owner's to cancel",
    "read": (
        "use the bridge tool, op=community_read: it marks community text as " "untrusted for you"
    ),
    "join": "joining accepts the guidelines, which is the owner's act (/community in the app)",
    "leave": "leaving is the owner's act (/community in the app)",
}


def add_parser(subparsers: Any, parent_parser: argparse.ArgumentParser | None = None) -> None:
    """Register ``lop community`` (hidden: no ``help=``). Stdlib-only."""

    parents = [parent_parser] if parent_parser is not None else []
    parser = subparsers.add_parser("community", parents=parents)
    verbs = parser.add_subparsers(dest="community_command")

    def verb(name: str) -> argparse.ArgumentParser:
        child = verbs.add_parser(name, parents=parents)
        child.add_argument("--json", action="store_true")
        child.add_argument("--community", default=None)
        return child

    verb("status")
    guidelines = verb("guidelines")
    guidelines.add_argument("--version", action="store_true", dest="version_only")
    join = verb("join")
    join.add_argument("--accept-guidelines", required=True, metavar="VERSION")
    channels = verb("channels")
    channels.add_argument("--filter", choices=sorted(_FILTERS), default=None)
    read = verb("read")
    read.add_argument("channel")
    read.add_argument("--limit", type=int, default=50)
    post = verb("post")
    post.add_argument("channel")
    post.add_argument("--text", required=True, help="the post, or - to read stdin")
    # The hub answers a submission synchronously (200 published / 422
    # refused), so there is nothing to stop waiting for; accepted so scripts
    # written against the documented surface keep working.
    post.add_argument("--no-wait", action="store_true")
    cancel = verb("cancel")
    cancel.add_argument("submission_id")
    leave = verb("leave")
    leave.add_argument("--yes", action="store_true")


def main(args: argparse.Namespace, *, client_factory: Any = None) -> int:
    """Run one ``lop community`` verb. Returns the exit code."""

    from local_operator.community import community_enabled

    if not community_enabled():
        _say(
            args,
            {"ok": False, "code": "community_disabled"},
            "community: not enabled on this machine (LOCAL_OPERATOR_COMMUNITY=1)",
        )
        return 1
    command = getattr(args, "community_command", None)
    if not command:
        print("usage: lop community {status,guidelines,join,channels,read,post,cancel,leave}")
        return 2
    refusal = _agent_shell_refusal(command)
    if refusal is not None:
        _say(args, {"ok": False, "code": "agent_shell", "use": refusal}, refusal)
        return service.EXIT_CODES["agent_shell"]
    try:
        return asyncio.run(_run(command, args, client_factory))
    except Exception as exc:  # noqa: BLE001 - a CLI reports the refusal it was given
        outcome = service.outcome_from_error(exc)
        _say(
            args,
            {"ok": False, **outcome.as_json()},
            f"community {command}: {outcome.kind} ({outcome.code or type(exc).__name__})",
        )
        return outcome.exit_code


def _agent_shell_refusal(command: str) -> str | None:
    """The model-facing refusal for ``command`` in an agent's shell, else ``None``."""

    from local_operator.agent_shell import in_agent_shell

    route = AGENT_SHELL_VERBS.get(command)
    if route is None or not in_agent_shell():
        return None
    return (
        f"community {command}: not run from an agent's shell (exit 10) — this command "
        f"acts or reads as the owner with no approval step. Instead, {route}."
    )


#: DEL and C1 controls survive ``json.dumps(ensure_ascii=False)`` as raw code
#: points (C0 is escaped already); a terminal may act on them, so ``--json``
#: escapes them losslessly. A script still gets the exact server string back.
_C1 = {code: f"\\u{code:04x}" for code in range(0x7F, 0xA0)}


def _say(args: Any, payload: dict[str, Any], line: str) -> None:
    from local_operator.ansi import strip_control_sequences

    if getattr(args, "json", False):
        print(json.dumps(payload, ensure_ascii=False, indent=2).translate(_C1))
    else:
        # The one print path for text mode, so every server-sourced field in a
        # line (names, purposes, guidelines, posts, refusal references) is
        # stripped here rather than at each call site. Newlines and tabs stay.
        print(strip_control_sequences(line))


def _config_dir() -> Any:
    from local_operator.paths import config_dir

    return config_dir()


async def _run(command: str, args: Any, client_factory: Any) -> int:
    client = service.make_client(_config_dir(), client_factory)
    wanted = getattr(args, "community", None)
    if command in ("status", "guidelines", "join"):
        community, joined = await service.any_community(client, wanted)
        if community is None:
            _say(args, {"ok": False, "code": "no_community"}, "community: none available")
            return 1
        boot = await client.bootstrap(community.id)
        if command == "status":
            payload = {
                "ok": True,
                "community": community.id,
                "name": community.name,
                "joined": boot.membership is not None,
                "guidelines_version": boot.guidelines.version,
                "accepted_version": boot.membership.guidelines_version if boot.membership else None,
            }
            state = "joined" if boot.membership else "not joined"
            _say(args, payload, f"{community.name}: {state} · guidelines {boot.guidelines.version}")
            return 0
        if command == "guidelines":
            payload = {"ok": True, "version": boot.guidelines.version}
            if not args.version_only:
                payload["text"] = boot.guidelines.text
            _say(
                args,
                payload,
                (
                    boot.guidelines.version
                    if args.version_only
                    else (
                        f"Community Guidelines ({boot.guidelines.version})\n\n"
                        f"{boot.guidelines.text}"
                    )
                ),
            )
            return 0
        # join
        current = boot.guidelines.version
        if args.accept_guidelines != current:
            _say(
                args,
                {"ok": False, "code": "guidelines", "version": current},
                f"community join: the current guidelines are {current}; read them "
                f"(lop community guidelines) and pass --accept-guidelines {current}",
            )
            return service.EXIT_CODES["guidelines"]
        if boot.membership is not None:
            _say(args, {"ok": True, "state": "already_joined"}, f"{community.name}: already joined")
            return 0
        joined_boot = await client.join(
            community.id, guidelines_version=current, accept_guidelines=True
        )
        _say(
            args,
            {"ok": True, "state": "joined", "guidelines_version": current},
            f"Joined {joined_boot.community.name} · guidelines {current} accepted",
        )
        return 0

    community = await service.joined_community(client, wanted)
    if community is None:
        _say(
            args,
            {"ok": False, "code": "not_joined"},
            "community: not joined (lop community join --accept-guidelines VERSION)",
        )
        return 1
    if command == "channels":
        wire = _FILTERS.get(args.filter) if args.filter else None
        rows = await service.list_channels(client, community.id, filter=wire)
        if args.filter == "not-joined":
            rows = [row for row in rows if not row.member]
        payload = {"ok": True, "channels": [row.model_dump(exclude_none=True) for row in rows]}
        lines = [
            f"#{row.name}"
            + (f"  {row.purpose}" if row.purpose else "")
            + ("  [private]" if row.visibility == "private" else "")
            + ("  joined" if row.member else "")
            for row in rows
        ]
        _say(args, payload, "\n".join(lines) or "no channels")
        return 0
    if command == "read":
        if not 1 <= args.limit <= 100:
            _say(args, {"ok": False, "code": "invalid"}, "community read: --limit must be 1..100")
            return 2
        channel = await service.find_channel(client, community.id, args.channel)
        if channel is None:
            _say(
                args,
                {"ok": False, "code": "channel_not_found"},
                f"community read: no joined channel #{service.channel_word(args.channel)}",
            )
            return 1
        page = await client.history(community.id, channel.id)
        items = list(page.items)[-args.limit :]
        payload = {
            "ok": True,
            "channel": f"#{channel.name}",
            "messages": [item.model_dump(exclude_none=True) for item in items],
        }
        lines = [
            f"{m.created_at}  {m.author.display_name}"
            + ("  [Agent]" if m.author.kind == "agent" else "")
            + f"\n  {m.text or ''}"
            for m in items
        ]
        _say(args, payload, "\n".join(lines) or f"No messages in #{channel.name} yet.")
        return 0
    if command == "post":
        text = sys.stdin.read() if args.text == "-" else args.text
        if not text.strip():
            _say(args, {"ok": False, "code": "invalid"}, "community post: empty text")
            return 2
        channel = await service.find_channel(client, community.id, args.channel)
        if channel is None:
            _say(
                args,
                {"ok": False, "code": "channel_not_found"},
                f"community post: no joined channel #{service.channel_word(args.channel)}",
            )
            return 1
        boot = await client.bootstrap(community.id)
        key = None
        try:
            from local_operator.community.client import new_idempotency_key

            key = new_idempotency_key()
            result = await client.submit(
                community.id,
                channel.id,
                text=text,
                idempotency_key=key,
                guidelines_version=boot.membership.guidelines_version if boot.membership else None,
            )
            outcome = service.outcome_from_submission(result)
        except Exception as exc:  # noqa: BLE001
            outcome = service.outcome_from_error(exc, submission_id=key)
        _say(
            args,
            {"ok": outcome.kind == "posted", **outcome.as_json()},
            _post_line(outcome, channel.name),
        )
        return outcome.exit_code
    if command == "cancel":
        try:
            result = await client.cancel_submission(community.id, args.submission_id)
            outcome = service.outcome_from_submission(result)
        except Exception as exc:  # noqa: BLE001
            outcome = service.outcome_from_error(exc, submission_id=args.submission_id)
        if outcome.kind == "posted":
            outcome = service.PostOutcome(
                "too_late", submission_id=args.submission_id, message_id=outcome.message_id
            )
        line = {
            "cancelled": "Cancelled · not posted",
            "too_late": "Already posted · cancel was too late",
        }.get(outcome.kind, f"community cancel: {outcome.kind}")
        _say(args, {"ok": outcome.kind == "cancelled", **outcome.as_json()}, line)
        return outcome.exit_code
    if command == "leave":
        if not args.yes:
            _say(
                args,
                {"ok": False, "code": "confirm"},
                "community leave: leaving ends your channels and DMs; pass --yes to leave. "
                "To switch orgs instead, use /orgs in the app.",
            )
            return 1
        already = await client.leave(community.id)
        _say(
            args,
            {"ok": True, "state": "already_left" if already else "left"},
            f"Left {community.name}" if not already else "Already left",
        )
        return 0
    return 2


def _post_line(outcome: service.PostOutcome, channel: str) -> str:
    kind = outcome.kind
    if kind == "posted":
        return f"Posted to #{channel} · {outcome.message_id}"
    if kind == "checking":
        return f"Checking · lop community cancel {outcome.submission_id}"
    if kind == "refused":
        return f"Not posted: {outcome.category} · {outcome.reference}"
    if kind == "rate_limited":
        return f"Slow down ({outcome.scope}) · retry in {outcome.retry_after_s} s"
    if kind == "moderation_unavailable":
        return "Moderation unavailable · not posted · retry later"
    if kind == "offline":
        return "Offline · not posted"
    if kind == "restricted":
        return f"You can't post in #{channel}"
    if kind == "guidelines":
        return f"Guidelines updated to {outcome.version} · accept them first"
    return f"Not posted ({outcome.code})"


__all__ = ["add_parser", "main"]
