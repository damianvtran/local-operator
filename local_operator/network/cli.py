"""``lop network``: argument registration (stdlib-only) and the verbs.

THE STDLIB-ONLY HALF is :func:`add_parser`. ``local_operator/cli.py`` imports it on
every ``lop`` invocation to build the parser, so nothing in this module's import
graph may reach ``cryptography``, ``socket``-heavy machinery or the config store.
Every verb's real work is imported inside the handler that needs it.

EVERY ACTION SUPPORTS ``--json``, following the mobile group's contract: the
agent-facing tool drives this CLI and parses it, so a human-readable sentence and a
machine-readable payload are both first-class and neither is a rendering of the
other.

WHERE A MUTATION GOES. The relay is the single writer of the network records while
it is running, so every mutation tries the relay's loopback control socket FIRST
and only falls back to writing the store in-process when no relay is running —
reported as such, because two writers on one file is exactly the thing the design's
"one writer" rule exists to avoid. Reading (``ls``, ``show``, ``log``, ``doctor``)
needs no relay at all: the store is the source of truth, and a status command that
refused to work when the relay was down would be useless precisely when it is
needed.

THE TWO REFUSALS THAT ARE FEATURES, NOT FRUSTRATIONS:

* ``invite`` writes the token to a 0600 FILE and prints the path. A token on stdout
  is a token in the agent's transcript, and the transcript is replayed to the
  provider on every later turn. ``--print`` exists for a TTY and is refused with
  ``--json``.
* ``join`` requires a HUMAN to transcribe a code shown on the other device. There
  is no flag anywhere that accepts a SAS on the inviter's side, and ``--sas-stdin``
  (the test seam) is refused unless ``LOP_NETWORK_TEST_MODE=1``. An agent therefore
  cannot complete a pairing on its own, which is the property R3 asks for.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path
from typing import Any, Callable

#: The default invite lifetime. A string flag parsed by :func:`_duration`, because
#: ``--expires 10m`` is what a person types and ``600`` is what they would have to
#: compute.
DEFAULT_INVITE_TTL_S = 600.0

#: The environment variable that unlocks ``--sas-stdin``. Named and checked in ONE
#: place: a test seam that any environment can open is not a test seam.
TEST_MODE_ENV = "LOP_NETWORK_TEST_MODE"

_ACTIONS = (
    "init",
    "invite",
    "join",
    "ls",
    "show",
    "rename",
    "rm",
    "member",
    "peers",
    "serve",
    "start",
    "stop",
    "restart",
    "status",
    "disconnect",
    "panic",
    "trust",
    "log",
    "doctor",
    "identity",
    "uninstall",
    # The session plane's client half (mesh-session-mobility.md §9.3): listing what
    # the peers hold, and driving a session that lives on one of them.
    "sessions",
)


def _duration(text: str) -> float:
    """``30s`` / ``10m`` / ``2h`` / ``90`` (seconds) → seconds.

    Refused at PARSE time rather than after a token was minted: an invite whose
    lifetime nobody could parse would otherwise default silently, and the one thing
    an operator must be able to trust about a bearer credential is when it dies.
    """
    raw = text.strip().lower()
    if not raw:
        raise argparse.ArgumentTypeError("give a duration like 30s, 10m, 2h or 600")
    unit = raw[-1]
    number = raw[:-1] if unit in ("s", "m", "h") else raw
    multiplier = {"s": 1.0, "m": 60.0, "h": 3600.0}.get(unit, 1.0)
    try:
        value = float(number)
    except ValueError:
        raise argparse.ArgumentTypeError(
            f"{text!r} is not a duration like 30s, 10m or 2h"
        ) from None
    if value <= 0:
        raise argparse.ArgumentTypeError("a duration must be greater than zero")
    return value * multiplier


def add_parser(subparsers: Any, parent_parser: Any = None) -> None:
    """Register ``lop network`` and its actions.

    ``parent_parser`` is threaded through so the position-independent globals
    (``--debug``, ``--agent``) keep working on these subcommands exactly as they do
    on ``tunnels``/``secrets`` — the parser is built with ``parents=[…]`` when the
    caller supplies it, and the group still works standalone for a test that has no
    parent.
    """
    parents = [parent_parser] if parent_parser is not None else []
    parser = subparsers.add_parser(
        "network",
        help="Mesh networks: pair devices, drive sessions across them",
        parents=parents,
    )
    actions = parser.add_subparsers(dest="network_command")

    init = actions.add_parser("init", help="Create a network on this device")
    init.add_argument("name")
    init.add_argument("--listen-address", default="", help="Bind address (127.0.0.1 = dial-only)")
    init.add_argument("--port", type=int, default=0)
    init.add_argument("--advertise-host", action="append", default=[], dest="advertise_hosts")
    init.add_argument("--no-start", action="store_true", help="Do not start the relay")
    init.add_argument("--json", action="store_true")

    invite = actions.add_parser("invite", help="Mint a single-use invite token")
    invite.add_argument("--network", default="")
    invite.add_argument("--role", choices=("read", "drive", "admin"), default="drive")
    invite.add_argument("--expires", type=_duration, default=DEFAULT_INVITE_TTL_S)
    invite.add_argument("--hosts", default="", help="Comma-separated host:port to advertise")
    invite.add_argument(
        "--device",
        default="",
        help="Bind the token to one device id (only that device may redeem it)",
    )
    invite.add_argument(
        "--print",
        action="store_true",
        dest="print_token",
        help="Print the token (TTY only; refused with --json)",
    )
    invite.add_argument("--json", action="store_true")

    join = actions.add_parser("join", help="Join a network from an invite")
    join.add_argument(
        "token", nargs="?", default="", help="A token, @path, or nothing for the newest"
    )
    join.add_argument("--host", default="", help="Override the endpoint to dial")
    join.add_argument("--verify", action="store_true", help="Compare the 160-bit fingerprint")
    join.add_argument("--emit-sas", action="store_true", help="Print this device's code, then wait")
    join.add_argument("--name", default="", help="The name this device will be known by")
    join.add_argument(
        "--sas-stdin",
        action="store_true",
        help=f"Read the code from stdin ({TEST_MODE_ENV}=1 only)",
    )
    join.add_argument("--json", action="store_true")

    ls = actions.add_parser("ls", help="Networks this device is in")
    ls.add_argument("--json", action="store_true")

    show = actions.add_parser("show", help="Members, roles, endpoints and audit tail")
    show.add_argument("network")
    show.add_argument("--json", action="store_true")

    rename = actions.add_parser("rename", help="Rename a network locally")
    rename.add_argument("network")
    rename.add_argument("name")
    rename.add_argument("--json", action="store_true")

    rm = actions.add_parser("rm", help="Forget a network locally")
    rm.add_argument("network")
    rm.add_argument("--json", action="store_true")

    member = actions.add_parser("member", help="Membership administration")
    member_actions = member.add_subparsers(dest="member_command")
    member_rm = member_actions.add_parser("rm", help="Revoke a member (rotates the secret)")
    member_rm.add_argument("network")
    member_rm.add_argument("device")
    member_rm.add_argument("--json", action="store_true")

    peers = actions.add_parser("peers", help="Reachable peers right now")
    peers.add_argument("--json", action="store_true")

    # `lop network sessions`: the session plane, from a shell. Read-only by
    # default (list what the peers hold); `--create` / `--engage` / `--stop` are
    # the three acts, and each names ONE peer because a session lives on exactly
    # one device. No flag here does anything to a session on THIS device: the
    # local surface for that is `lop sessions` / `lop stop` (design: one
    # implementation, four front ends).
    net_sessions = actions.add_parser(
        "sessions",
        help="List, create, warm or stop sessions on other devices",
        description=(
            "Talk to the SESSION PLANE across the mesh: the sessions another "
            "device holds. Local sessions are `lop sessions` and `lop stop`."
        ),
    )
    net_sessions.add_argument("--peer", default="", help="the device to ask (id or name)")
    net_sessions.add_argument(
        "--all-peers", action="store_true", help="list every peer's sessions, merged"
    )
    net_sessions.add_argument("--create", action="store_true", help="create a session on --peer")
    net_sessions.add_argument("--engage", metavar="SESSION", default="", help="warm that session")
    net_sessions.add_argument("--stop", metavar="SESSION", default="", help="stop that session")
    net_sessions.add_argument("--cwd", default="", help="with --create: where it should run")
    net_sessions.add_argument("--name", default="", help="with --create: its title")
    net_sessions.add_argument("--prompt", default="", help="with --create: its first turn")
    net_sessions.add_argument("--json", action="store_true")

    serve = actions.add_parser("serve", help="Run the relay in the foreground")
    serve.add_argument("--port", type=int, default=0)
    serve.add_argument("--address", default="")
    serve.add_argument("--no-launchd", action="store_true", help="Never touch a plist")
    serve.add_argument("--json", action="store_true")

    for action in ("start", "stop", "restart"):
        child = actions.add_parser(action, help=f"{action.capitalize()} the relay")
        child.add_argument("--json", action="store_true")

    status = actions.add_parser("status", help="Install state, links, log paths")
    status.add_argument("--json", action="store_true")

    disconnect = actions.add_parser("disconnect", help="Leave a network; stop trusting it")
    disconnect.add_argument("network", nargs="?", default="")
    disconnect.add_argument("--json", action="store_true")

    panic = actions.add_parser("panic", help="Incident: broadcast a revoke and rotate")
    panic.add_argument("network", nargs="?", default="")
    panic.add_argument("--json", action="store_true")

    trust = actions.add_parser("trust", help="Re-admit a network after a panic")
    trust.add_argument("network")
    trust.add_argument("--active", action="store_true", help="Mark it active again")
    trust.add_argument("--untrusted", action="store_true", help="Mark it untrusted")
    trust.add_argument("--json", action="store_true")

    log = actions.add_parser("log", help="The audit log")
    log.add_argument("--network", default="")
    log.add_argument("--follow", "-f", action="store_true")
    log.add_argument("--since", default="", help="A duration back from now, e.g. 15m")
    log.add_argument("--limit", type=int, default=50)
    log.add_argument("--export", type=Path, default=None, help="Copy the log out; never pruned")
    log.add_argument("--json", action="store_true")

    confirm = actions.add_parser(
        "confirm", help="Answer a parked pairing: compare the code, then admit or refuse"
    )
    confirm.add_argument("invite_id", nargs="?", default="", help="Which parked pairing")
    confirm.add_argument(
        "--list", action="store_true", dest="list_pending", help="Show the queue, answer nothing"
    )
    confirm.add_argument("--decline", action="store_true", help="Refuse it instead of admitting")
    confirm.add_argument(
        "--sas-stdin",
        action="store_true",
        help=f"Read the answer from stdin ({TEST_MODE_ENV}=1 only; the harness's seam)",
    )
    confirm.add_argument("--json", action="store_true")

    doctor = actions.add_parser("doctor", help="Diagnose the mesh: reachability, epochs, identity")
    doctor.add_argument("--peer", default="")
    doctor.add_argument("--json", action="store_true")

    identity = actions.add_parser("identity", help="This device's key")
    identity_actions = identity.add_subparsers(dest="identity_command")
    identity_rotate = identity_actions.add_parser(
        "rotate", help="Replace the device key and announce the change"
    )
    identity_rotate.add_argument("--json", action="store_true")
    identity_show = identity_actions.add_parser("show", help="This device's id and fingerprint")
    identity_show.add_argument("--json", action="store_true")

    uninstall = actions.add_parser(
        "uninstall", help="Remove the LaunchAgent (and --purge the store)"
    )
    uninstall.add_argument(
        "--purge",
        action="store_true",
        help="Delete the network records, invites, queues and audit log (NOT the identity)",
    )
    uninstall.add_argument(
        "--purge-identity",
        action="store_true",
        dest="purge_identity",
        help="Also destroy the device keypair; needs a terminal and a typed confirmation",
    )
    uninstall.add_argument(
        "--network",
        action="append",
        default=[],
        dest="purge_networks",
        help="Scope --purge to one network (repeatable; default: every network here)",
    )
    uninstall.add_argument("--json", action="store_true")


# ---------------------------------------------------------------------------
# Dispatch
# ---------------------------------------------------------------------------


def main(args: argparse.Namespace) -> int:
    """Dispatch ``lop network <action>``; imports stay function-local."""
    action = getattr(args, "network_command", None) or "ls"
    handler = _HANDLERS.get(action)
    if handler is None:
        print(f"unknown network action {action!r}", file=sys.stderr)
        return 2
    try:
        return int(handler(args))
    except Exception as exc:  # noqa: BLE001 — a CLI reports, it does not traceback
        from local_operator.network.types import MeshRefusal

        if isinstance(exc, MeshRefusal):
            # A --json CALLER GETS A BODY, not just a non-zero exit. The refusal
            # already carries both halves (a machine code and a sentence for the
            # person); printing only the sentence left the structured half
            # unreachable on the one surface a script drives, so a consumer would
            # have to match on English to tell "that device is unreachable" from
            # "it refused the op" — which is the branch this slice's refusals
            # exist to make possible. Stdout carries the payload, stderr keeps the
            # coloured sentence for a human watching the terminal.
            if _json_mode(args):
                print(json.dumps({"ok": False, "code": exc.code, "message": exc.sentence}))
            print(f"\033[1;31m{exc.sentence}\033[0m", file=sys.stderr)
            return 1
        raise


# ---------------------------------------------------------------------------
# Shared helpers
# ---------------------------------------------------------------------------


def _json_mode(args: argparse.Namespace) -> bool:
    return bool(getattr(args, "json", False))


def _emit(args: argparse.Namespace, payload: dict[str, Any], lines: list[str]) -> int:
    """Print a payload as JSON or as lines — one or the other, never both.

    Printing both would put machine output in a pipe that asked for lines and
    human prose in a pipe that parses JSON.
    """
    if _json_mode(args):
        print(json.dumps(payload, indent=2, sort_keys=True, default=str))
    else:
        for line in lines:
            print(line)
    return 0 if payload.get("ok", True) else 1


def _relay_call(op: str, *, timeout: float = 5.0, **fields: Any) -> dict[str, Any] | None:
    """Run one op on the running relay, or ``None`` when there is no relay.

    ``timeout`` is a parameter because the session-plane ops are not all fast:
    a spawn takes seconds and the kill-switch ladder is allowed MINUTES (its
    SIGTERM rung waits out a drain the receiver owns). The default stays the
    original 5 s so every fast status op behaves exactly as it did; the slow
    ones pass their own budget and say why at the call site.
    """
    from local_operator.network import relay, store

    record = store.find_own_relay()
    if record is None:
        return None
    reply = relay.control_request(record, op, timeout=timeout, **fields)
    if reply is None:
        return None
    if reply.get("op") == "error":
        from local_operator.network.types import MeshRefusal

        raise MeshRefusal("relay_refused", str(reply.get("message") or "the relay refused"))
    detail = reply.get("detail")
    return detail if isinstance(detail, dict) else {"value": detail}


def _resolve(target: str, root: Path | None = None) -> Any:
    """One network by id or name, refusing an ambiguous name rather than guessing."""
    from local_operator.network import store, types

    records = store.list_networks(root)
    if not target:
        if len(records) == 1:
            return records[0]
        raise types.MeshRefusal(
            "ambiguous_network",
            "name a network: this device is in "
            + (", ".join(f"{row.name} ({row.network_id})" for row in records) or "none"),
        )
    matches = store.match_networks(records, target)
    if not matches:
        raise types.MeshRefusal(
            "unknown_network", f"this device is not in a network called {target!r}"
        )
    if len(matches) > 1:
        raise types.MeshRefusal(
            "ambiguous_network",
            f"{target!r} matches {len(matches)} networks; use the network id",
        )
    return matches[0]


def _summarise(record: Any, links: int = 0) -> dict[str, Any]:
    return {
        "network_id": record.network_id,
        "name": record.name,
        "epoch": record.epoch,
        "role": record.self_role,
        "trust": record.trust,
        "members": len(record.active_members()),
        "links": links,
        "stale": record.stale,
    }


# ---------------------------------------------------------------------------
# Lifecycle of a network
# ---------------------------------------------------------------------------


def _cmd_init(args: argparse.Namespace) -> int:
    """Create a network on this device: a record, a fresh secret, and our own row.

    The first member row is OUR OWN, added with ``added_via: "self"`` and the role
    an admin invite would grant. A network with no members would be one nobody can
    be authorised in, and writing our row at creation is what makes the very next
    handshake verifiable.
    """
    from local_operator.network import store, types
    from local_operator.network.identity import load_or_mint, network_root

    imported = _import_relay()
    settings = imported.NetworkSettings.from_config()
    listen_address = args.listen_address or settings.listen_address
    port = args.port or settings.port
    identity = load_or_mint()

    record = types.NetworkRecord(
        network_id=store.new_network_id(),
        name=args.name,
        created_by=identity.device_id,
        self_device_id=identity.device_id,
        self_role="admin",
        self_capabilities=sorted(types.capabilities_for_role("admin")),
        listen={
            "address": listen_address,
            "port": port,
            "advertised": list(args.advertise_hosts or settings.advertise_hosts),
        },
    )
    from secrets import token_bytes

    from local_operator.network import wire

    state = types.SecretState(
        network_id=record.network_id,
        epoch=1,
        secret=wire.b64u(token_bytes(32)),
    )
    imported.admit(
        record,
        device_id=identity.device_id,
        public_key=identity.public_key,
        name=identity.name,
        role="admin",
        capabilities=sorted(types.capabilities_for_role("admin")),
        added_by=identity.device_id,
        added_via="self",
        endpoints=[],
        root=None,
        persist=False,
    )
    store.save(record, None)
    store.save_secrets(state, None)
    _audit(
        "member_admitted",
        network_id=record.network_id,
        epoch=record.epoch,
        actor=identity.device_id,
        detail={"role": "admin", "member_kind": "device", "epoch": record.epoch},
    )

    started = ""
    if not args.no_start:
        started = _autostart()
    return _emit(
        args,
        {
            "ok": True,
            "network_id": record.network_id,
            "name": record.name,
            "epoch": record.epoch,
            "device_id": identity.device_id,
            "listen": record.listen,
            "identity_dir": str(network_root()),
            "relay": started,
        },
        [
            f"created network {record.name} ({record.network_id}), epoch {record.epoch}",
            f"this device is {identity.device_id}",
            "next: lop network invite --role drive   (the token is written to a file, not printed)",
            started,
        ],
    )


def _unwrap_list_answer(live: Any) -> Any:
    """A list-shaped relay answer, unwrapped from the control client's envelope.

    ``_relay_call`` wraps a non-dict detail as ``{"value": ...}`` so a caller
    always has a mapping to read; the verbs whose answer is a LIST
    (``net_ls``, ``net_peer_ls``) have to unwrap it. ``_cmd_peers`` does that
    inline; ``_cmd_ls`` did not, so `lop network ls` answered
    ``TypeError: string indices must be integers`` on every device whose relay
    was RUNNING — and worked on every device whose relay was not, which is why
    it survived: the fallback path below reads the store directly. Found while
    driving the session plane end to end.
    """
    if isinstance(live, dict):
        return live.get("value")
    return live


def _cmd_ls(args: argparse.Namespace) -> int:
    from local_operator.network import store

    live = _unwrap_list_answer(_relay_call("net_ls"))
    records = store.list_networks()
    rows = live if live else [_summarise(record) for record in records]
    if not rows:
        return _emit(args, {"ok": True, "networks": []}, ["no networks on this device"])
    return _emit(
        args,
        {"ok": True, "networks": rows},
        [
            f"{row['name']}  {row['network_id']}  epoch {row['epoch']}  {row['role']}  "
            f"{row['members']} member(s)  {row['trust']}"
            + (f"  [{row['stale']}]" if row.get("stale") else "")
            for row in rows
        ],
    )


def _cmd_show(args: argparse.Namespace) -> int:
    live = _relay_call("net_show", network=args.network)
    if live is not None:
        return _emit(args, {"ok": True, **live}, _show_lines(live))
    from local_operator.network.relay import members_digest_of

    record = _resolve(args.network)
    payload = _summarise(record)
    payload["members_detail"] = [
        {
            "device_id": member.device_id,
            "name": member.name,
            "role": member.role,
            "capabilities": list(member.capabilities),
            "active": member.active,
            "endpoints": list(member.endpoints),
            "suspect": member.suspect,
        }
        for member in record.members
    ]
    payload["rotations"] = dict(record.rotations)
    payload["members_digest"] = members_digest_of(record)
    payload["invites"] = [
        {"invite_id": invite.invite_id, "state": invite.state, "role": invite.role}
        for invite in record.invites
    ]
    return _emit(args, {"ok": True, **payload}, _show_lines(payload))


def _show_lines(payload: dict[str, Any]) -> list[str]:
    lines = [
        f"{payload.get('name')}  {payload.get('network_id')}  epoch {payload.get('epoch')}  "
        f"trust {payload.get('trust')}",
    ]
    for member in payload.get("members_detail") or []:
        mark = "active" if member["active"] else "REMOVED"
        suspect = "  [suspect: key may be copied]" if member.get("suspect") else ""
        lines.append(
            f"  {mark:7} {member['device_id']}  {member['role']:5}  "
            f"{', '.join(member['capabilities'])}{suspect}"
        )
    for invite in payload.get("invites") or []:
        lines.append(f"  invite {invite['invite_id']}  {invite['role']}  {invite['state']}")
    return lines


def _cmd_rename(args: argparse.Namespace) -> int:
    from local_operator.network import store

    record = _resolve(args.network)
    previous = record.name
    record.name = args.name
    store.save(record, None)
    return _emit(
        args,
        {"ok": True, "network_id": record.network_id, "name": record.name, "previous": previous},
        [f"renamed {previous} to {record.name} on this device"],
    )


def _cmd_rm(args: argparse.Namespace) -> int:
    """Forget a network locally. The audit trail is NOT touched: what happened
    outlives the membership, which is the whole point of a separate log."""
    from local_operator.network import store

    record = _resolve(args.network)
    removed = store.forget(record.network_id)
    return _emit(
        args,
        {"ok": True, "network_id": record.network_id, "removed": removed},
        [f"forgot {record.name} on this device ({len(removed)} file(s) removed)"],
    )


def _cmd_member_rm(args: argparse.Namespace) -> int:
    """Revoke a member: tombstone, rotate, bump the epoch, fan out (R5)."""
    live = _relay_call("net_member_rm", network=args.network, device_id=args.device)
    if live is not None:
        return _emit(
            args,
            {"ok": True, **live},
            [
                f"removed {live.get('removed')} from {args.network}; "
                f"epoch is now {live.get('epoch')}",
                f"queued for {live.get('queued', 0)} offline peer(s)",
            ],
        )
    imported = _import_relay()
    from local_operator.network import store

    record = _resolve(args.network)
    state = store.require_secrets(record.network_id)
    outcome = imported.remove_member(record, state, device_id=args.device, by=record.self_device_id)
    for member in record.active_members():
        if member.device_id == record.self_device_id:
            continue
        # The "no secret for a removed peer" rule lives in the WRITER, which refuses
        # to persist one — so this loop does not have to remember it.
        store.enqueue_frame(
            member.device_id,
            imported.epoch_frame(
                record, state, reason="member_removed", target_device_id=member.device_id
            ),
            removed=False,
        )
    _audit(
        "member_removed",
        network_id=record.network_id,
        epoch=outcome.epoch,
        actor=record.self_device_id,
        subject=args.device,
        detail={
            "initiated_by": record.self_device_id,
            "rekeyed": True,
            "epoch_after": outcome.epoch,
        },
    )
    return _emit(
        args,
        {
            "ok": True,
            "network_id": record.network_id,
            "removed": args.device,
            "epoch": outcome.epoch,
            "relay": "not running — applied locally and queued",
        },
        [
            f"removed {args.device} from {record.name}; epoch is now {outcome.epoch}",
            "the relay is not running, so the rotation is queued for delivery",
        ],
    )


# ---------------------------------------------------------------------------
# Pairing
# ---------------------------------------------------------------------------


def _cmd_invite(args: argparse.Namespace) -> int:
    """Mint an invite, write the token to a FILE, and print the path.

    Refuses ``--print`` with ``--json`` and on a non-TTY stdout: the token is a
    bearer credential and every path that puts it in a machine-readable stream is a
    path to a transcript.
    """
    if args.print_token and _json_mode(args):
        print(
            "--print and --json are mutually exclusive: --json must never carry the token",
            file=sys.stderr,
        )
        return 2
    if args.print_token and not sys.stdout.isatty():
        print(
            "refusing --print: stdout is not a terminal, so the token would land in a pipe, "
            "a log or an agent transcript. Read the file instead.",
            file=sys.stderr,
        )
        return 2
    hosts = [host for host in (args.hosts or "").split(",") if host]
    live = _relay_call(
        "net_invite",
        network=args.network,
        role=args.role,
        ttl_s=float(args.expires),
        hosts=hosts,
        device_id=args.device,
    )
    if live is not None:
        payload = {"ok": True, **live}
        path = str(live.get("path") or "")
    else:
        payload, path = _invite_locally(args, hosts)
    if args.print_token:
        print(Path(path).read_text(encoding="utf-8").strip())
        return 0
    return _emit(
        args,
        payload,
        [
            f"invite {payload['invite_id']} for role {payload['role']}, "
            f"expires in {int(float(payload.get('expires_in_s') or 0))}s",
            f"token written to {path}",
            "it is single use and is not printed: read that file, or run this with a TTY "
            "and --print",
            f"then, on the other device: lop network join @{path}"
            + (f" --host {payload['hosts'][0]}" if payload.get("hosts") else ""),
        ],
    )


def _invite_locally(args: argparse.Namespace, hosts: list[str]) -> tuple[dict[str, Any], str]:
    from local_operator.network import store
    from local_operator.network.invite import mint as mint_invite

    record = _resolve(args.network)
    state = store.require_secrets(record.network_id)
    minted = mint_invite(
        record,
        state.secret,
        role=args.role,
        ttl_s=float(args.expires),
        hosts=hosts or None,
        device_id=args.device,
    )
    record.invites.append(minted.record)
    store.save(record)
    path = store.save_invite_token(minted.record.invite_id, minted.token)
    _audit(
        "invite_minted",
        network_id=record.network_id,
        epoch=record.epoch,
        actor=record.self_device_id,
        detail={
            "role": minted.record.role,
            "expires_at": minted.record.expires_at,
            "bound_device": minted.record.device_id,
        },
    )
    return (
        {
            "ok": True,
            "invite_id": minted.record.invite_id,
            "path": str(path),
            "expires_at": minted.record.expires_at,
            "expires_in_s": minted.record.ttl_s,
            "role": minted.record.role,
            "hosts": list(minted.record.hosts),
            "network_id": record.network_id,
            "network_name": record.name,
            "relay": "not running — minted locally",
        },
        str(path),
    )


def _cmd_join(args: argparse.Namespace) -> int:
    """Join a network from an invite, with a human at this keyboard.

    Five steps, and the last two are the ones a machine may not skip: this device
    displays ITS OWN derived code, the human transcribes the code the other device
    shows, and the inviter's human confirms the other way. A mismatch burns the
    invite — refusing rather than warning is what makes pairing resistant to a relay
    in the middle.
    """
    from local_operator.network import invite as invite_mod
    from local_operator.network import relay as relay_mod
    from local_operator.network import store, wire
    from local_operator.network.handshake import (
        Credential,
        Handshake,
        pair_abort_frame,
        pair_timeout_seconds,
        sas_matches,
    )
    from local_operator.network.identity import load_or_mint
    from local_operator.network.types import MeshRefusal

    token = _read_token(args.token)
    envelope = invite_mod.decode(token)
    hosts = invite_mod.host_candidates(envelope, args.host or None)
    if not hosts:
        raise MeshRefusal(
            "no_host",
            "that invite names no endpoint; pass --host host:port",
        )
    identity = load_or_mint(name=args.name)
    settings = relay_mod.NetworkSettings.from_config()
    last_reason = ""
    for host in hosts:
        link_result = _join_one(
            host=host,
            token=token,
            envelope=envelope,
            identity=identity,
            settings=settings,
            args=args,
            wire=wire,
            Handshake=Handshake,
            Credential=Credential,
            pair_abort_frame=pair_abort_frame,
            pair_timeout_seconds=pair_timeout_seconds,
            sas_matches=sas_matches,
            invite_mod=invite_mod,
            store=store,
            relay_mod=relay_mod,
        )
        if link_result is None:
            continue
        return _emit(args, {"ok": True, **link_result[1]}, link_result[0])
    raise MeshRefusal(
        "join_failed",
        "could not join: "
        f"{last_reason or 'every endpoint in the invite refused or was unreachable'}",
    )


def _read_token(argument: str) -> str:
    """A token inline, ``@path``, or nothing at all (the newest outbox file).

    EVERY failure is a NAMED refusal carrying the path. `join @missing` used to
    raise `FileNotFoundError` out of `Path.read_text` and print a traceback, which
    tells the operator neither what was wrong nor what to do — the rule the whole
    CLI now follows: a file whose absence is a legitimate state is a refusal with a
    sentence, never a stack.
    """
    from local_operator.network import types

    if argument.startswith("@"):
        path = Path(argument[1:])
        try:
            return path.read_text(encoding="utf-8").strip()
        except FileNotFoundError:
            raise types.MeshRefusal(
                "token_unreadable",
                f"there is no invite token file at {path}. Check the path (it is the one "
                "`lop network invite` printed), or pass the token itself as the argument "
                "instead of @path.",
            ) from None
        except (OSError, UnicodeDecodeError) as exc:
            raise types.MeshRefusal(
                "token_unreadable",
                f"could not read the invite token at {path}: {exc.__class__.__name__}. "
                "Check the path and its permissions, or pass the token itself as the "
                "argument instead of @path.",
            ) from None
    if argument:
        return argument
    from local_operator.network import store

    files = sorted(store.outbox_dir().glob("*.invite"), key=lambda path: path.stat().st_mtime)
    if not files:
        raise types.MeshRefusal(
            "no_invite_token",
            "no token was given and this device has no invite file in its outbox. Mint "
            "one on the other device with `lop network invite`, bring the file across, "
            "then run `lop network join @<path>`.",
        )
    return files[-1].read_text(encoding="utf-8").strip()


def _join_one(
    *,
    host: str,
    token: str,
    envelope: Any,
    identity: Any,
    settings: Any,
    args: argparse.Namespace,
    **helpers: Any,
) -> tuple[list[str], dict[str, Any]] | None:
    """One dial attempt: handshake, the human step, then admission."""
    import socket

    from local_operator.network import wire
    from local_operator.network.identity import mint_instance_id
    from local_operator.network.types import HandshakeRefusal, MeshRefusal

    Handshake = helpers["Handshake"]
    Credential = helpers["Credential"]
    store = helpers["store"]
    invite_mod = helpers["invite_mod"]

    address, _, port_text = host.rpartition(":")
    try:
        port = int(port_text)
    except ValueError:
        return None
    try:
        sock = socket.create_connection(
            (address or host, port), timeout=settings.handshake_timeout_s
        )
    except OSError:
        return None
    deadline = wire.deadline_in(settings.handshake_timeout_s)
    try:
        sock.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)
        handshake = Handshake.new(
            role="dialer",
            identity=identity,
            network_id=envelope.network_id,
            epoch=envelope.epoch,
            instance_id=mint_instance_id(),
            session_protocol=_session_protocol(),
            mode="join",
            capabilities=list(wire.LINK_CAPABILITIES),
            build={},
        )
        handshake.join_block = {
            "invite_id": envelope.invite_id,
            "joiner_public_key": identity.public_key,
            "joiner_name": args.name or identity.name,
        }
        handshake.send_hello(sock)
        reader = wire.FrameReader(sock)
        handshake.read_challenge(reader, deadline)
        credential = Credential(
            "invite",
            envelope.epoch,
            wire.invite_key(envelope.material, envelope.network_id, envelope.invite_id),
        )
        handshake.send_auth(sock, credential)
        handshake.read_welcome(reader, deadline)
        result = handshake.establish()
        codec = handshake.codec()
        fingerprint = wire.transcript_fingerprint(bytes.fromhex(result.transcript_hash))
        if args.emit_sas:
            print(json.dumps({"sas": result.sas, "fingerprint": fingerprint}))
            sys.stdout.flush()
        else:
            print(invite_mod.joiner_prompt(envelope, result.sas, fingerprint))
            sys.stdout.flush()
        typed = _read_code(args, result.sas, fingerprint)
        if typed is None:
            sock.sendall(codec.seal(helpers["pair_abort_frame"](req=1, reason="declined_local")))
            raise MeshRefusal("declined", "this device declined the pairing")
        sock.sendall(codec.seal({"op": "net_pair_ready", "req": 1, "sas": typed}))
        # The wait is the CONFIRM budget, not the handshake timeout: the other side
        # now has to reach a human, and timing out at 10 s would fail every honest
        # pairing on a device whose relay is a daemon.
        remaining = max(0.0, envelope.issued_at + envelope.ttl_s - time.time())
        answer = codec.open(
            reader.read_record_payload(
                wire.deadline_in(helpers["pair_timeout_seconds"](remaining or envelope.ttl_s))
            )
        )
        if answer.get("op") == "net_pair_abort":
            raise MeshRefusal(
                str(answer.get("reason") or "aborted"),
                (
                    invite_mod.sas_mismatch_sentence()
                    if answer.get("reason") == "sas_mismatch"
                    else f"the pairing was refused ({answer.get('reason')})"
                ),
            )
        if not answer.get("admit"):
            raise MeshRefusal("not_admitted", "the other device did not admit this machine")
        record = _persist_join(answer, envelope, identity, host, store)
        return (
            [
                f"joined {record.name} ({record.network_id}) at epoch {record.epoch}",
                f"members: {len(record.active_members())}",
                "next: lop network peers   ·   lop sessions --all-peers",
            ],
            {
                "network_id": record.network_id,
                "name": record.name,
                "epoch": record.epoch,
                "members": len(record.active_members()),
                "device_id": identity.device_id,
                "inviter": envelope.inviter_device_id,
                "role": record.self_role,
                "fingerprint": fingerprint,
            },
        )
    except MeshRefusal:
        raise
    except HandshakeRefusal:
        # The reason is deliberately dropped here: it is already in the local audit
        # record, and this device's human gets the one sentence that matters — that
        # the other side refused — rather than our own refusal vocabulary.
        return None
    except (wire.LinkCryptoError, OSError, TimeoutError):
        return None
    finally:
        try:
            sock.close()
        except OSError:
            pass


def _read_code(args: argparse.Namespace, derived: str, fingerprint: str) -> str | None:
    """Get the code from the HUMAN, or from the test seam. Never from the peer.

    ``--sas-stdin`` exists for the e2e harness and is refused outside
    ``LOP_NETWORK_TEST_MODE=1``: it is a seam for a script feeding a prompt a human
    would type, not a way for an agent to complete a pairing.
    """
    import os

    if args.sas_stdin:
        if os.environ.get(TEST_MODE_ENV) != "1":
            raise ValueError(f"--sas-stdin is a test seam and requires {TEST_MODE_ENV}=1")
        typed = sys.stdin.readline().strip()
    elif sys.stdin.isatty():
        prompt = (
            "type the fingerprint shown on the other device (or its 6-digit code): "
            if args.verify
            else "type the code shown there: "
        )
        typed = input(prompt).strip()
    else:
        raise ValueError(
            "joining needs a person at a keyboard: run this in a terminal (or use "
            f"{TEST_MODE_ENV}=1 with --sas-stdin from a harness)"
        )
    if not typed:
        return None
    # ``--verify`` makes the FINGERPRINT the thing compared, which is worth 160
    # bits against the digits' ~20; the digits are still accepted because a private
    # LAN pairing is the common case and the code is what people read off a screen.
    if args.verify and not args.sas_stdin:
        if typed.replace("-", "").replace(" ", "").upper() != fingerprint.replace("-", ""):
            return ""
        return derived
    return typed


def _session_protocol() -> int:
    """The session protocol this build speaks, passed through the join hello.

    Function-local because this module is imported while the CLI builds its parser.
    """
    from local_operator.session.runtime.types import PROTOCOL_VERSION

    return int(PROTOCOL_VERSION)


def _persist_join(
    answer: dict[str, Any], envelope: Any, identity: Any, host: str, store: Any
) -> Any:
    """Write the network record and its secret from the admission frame.

    The member list comes from the frame because the joiner must hold the
    INVITER'S public key: without it, every later handshake from that device would
    be unverifiable, and a member list the joiner invented from the welcome frame
    would name a device it cannot check. ``material`` is written to the SEPARATE
    secrets file, never into the record.
    """
    from local_operator.network.types import (
        MemberRecord,
        MeshRefusal,
        NetworkRecord,
        SecretState,
    )

    network = answer.get("network") or {}
    member = answer.get("member") or {}
    material = str(answer.get("secret") or "")
    rows = [
        MemberRecord.from_json(row)
        for row in (answer.get("members") or [])
        if isinstance(row, dict)
    ]
    if not rows:
        rows = [
            MemberRecord.from_json(member),
            MemberRecord(
                device_id=envelope.inviter_device_id,
                name=envelope.inviter_name,
                role="admin",
                added_via="invite",
            ),
        ]
    self_row = next((row for row in rows if row.device_id == identity.device_id), None)
    record = NetworkRecord(
        network_id=str(network.get("network_id") or envelope.network_id),
        name=str(network.get("name") or envelope.network_name),
        epoch=int(network.get("epoch") or envelope.epoch),
        sequence=int(network.get("sequence") or 0),
        trust=str(network.get("trust") or "active"),
        self_device_id=identity.device_id,
        self_role=self_row.role if self_row else "read",
        self_capabilities=list(self_row.capabilities) if self_row else [],
        created_at=time.time(),
        created_by=envelope.inviter_device_id,
        listen={"address": "", "port": 0, "advertised": [host]},
        rotations={str(key): str(value) for key, value in (answer.get("rotations") or {}).items()},
        members=rows,
    )
    if not material:
        raise MeshRefusal(
            "no_secret", "the other device admitted this one without sending the network secret"
        )
    state = SecretState(network_id=record.network_id, epoch=record.epoch, secret=material)
    store.save(record, None)
    store.save_secrets(state, None)
    return record


# ---------------------------------------------------------------------------
# The relay's own lifecycle
# ---------------------------------------------------------------------------


def _cmd_serve(args: argparse.Namespace) -> int:
    """Run the relay in the foreground until signalled (``--no-launchd`` implied).

    Foreground by design: a self-daemonizing service is one nobody can stop, and
    supervision belongs to launchd or to the terminal that started it.
    """
    relay_mod = _import_relay()
    settings = relay_mod.NetworkSettings.from_config()
    if args.port:
        settings = _replace(settings, port=int(args.port))
    if args.address:
        settings = _replace(settings, listen_address=args.address)
    if _json_mode(args):
        print(
            json.dumps(
                {
                    "ok": True,
                    "serving": True,
                    "listen": settings.listen_address,
                    "port": settings.port,
                }
            )
        )
        sys.stdout.flush()
    server = relay_mod.RelayServer(settings=settings)
    server.serve_forever()
    return 0


def _autostart() -> str:
    """Start the relay after ``init``/``join`` unless it is already up.

    Pairing must not require knowing that a daemon exists, so the streamlined
    install/deploy/pair path starts it; a machine without launchd degrades to a
    sentence naming the command to run in the foreground.
    """
    relay_mod = _import_relay()
    if relay_mod.health() is not None:
        return "the relay is already running"
    if not relay_mod.is_supported():
        return (
            "no launchd here: run `lop network serve` in the foreground "
            f"(log {relay_mod.log_path()})"
        )
    result = relay_mod.install()
    if result.get("ok"):
        return "the relay is installed and running"
    if result.get("reason") == "isolated_home":
        # Not a failure, and it must not read as one: an isolated HOME is how every
        # test harness runs, and the operator (or the harness author) needs to know
        # that the missing LaunchAgent is expected and what to do instead.
        return str(result.get("error"))
    return f"the relay did not start: {result.get('error')}"


def _cmd_service(action: str) -> Callable[[argparse.Namespace], int]:
    def _run(args: argparse.Namespace) -> int:
        relay_mod = _import_relay()
        result = relay_mod.service_action(action)
        ok = bool(result.get("ok"))
        return _emit(
            args,
            {"ok": ok, "action": action, "error": result.get("error", "")},
            [f"relay {action} ok" if ok else f"relay {action} failed: {result.get('error')}"],
        )

    return _run


def _cmd_sessions(args: argparse.Namespace) -> int:
    """``lop network sessions`` — the session plane, across the mesh.

    Every act here is a LOCAL op on this device's relay, which owns the links and
    is the only process that speaks the mesh: the CLI asks its own relay, and the
    relay asks the peer. That is the same shape ``stream_open`` uses, and it is
    why no verb in this file ever opens a peer link itself.

    ``--stop`` is the interesting one, and it is deliberately a THIN call: the
    peer runs its own kill-switch ladder (its own pid proofs, its own rungs, its
    own sentences) and this side renders the peer's ``StopOutcome`` vocabulary
    verbatim rather than inventing a second set of words for "did it stop".
    """
    from local_operator.network.types import MeshRefusal

    peer = str(getattr(args, "peer", "") or "")
    session_id = str(getattr(args, "stop", "") or "")
    engage = str(getattr(args, "engage", "") or "")

    if session_id:
        if not peer:
            raise MeshRefusal("peer_required", "--stop needs --peer: a session lives on one device")
        # A LONG budget on purpose: the peer runs its OWN ladder, whose SIGTERM
        # rung waits out a drain only the owning machine can bound, and a CLI
        # that gave up at 5 s would report "nothing happened" about a stop that
        # was working.
        detail = (
            _relay_call(
                "peer_session_stop",
                peer=peer,
                session_id=session_id,
                mode="graceful",
                timeout=240.0,
            )
            or {}
        )
        return _emit(
            args,
            {"ok": detail.get("outcome") not in ("", "refused"), **detail},
            [
                str(detail.get("detail") or ""),
                f"rung: {detail.get('rung')}  outcome: {detail.get('outcome')}",
            ],
        )

    if engage:
        if not peer:
            raise MeshRefusal(
                "peer_required", "--engage needs --peer: a session lives on one device"
            )
        detail = (
            _relay_call(
                "peer_session_engage",
                peer=peer,
                session_id=engage,
                cwd=str(getattr(args, "cwd", "") or ""),
                timeout=120.0,  # a spawn, bounded by the relay's own engage deadline
            )
            or {}
        )
        return _emit(
            args,
            {"ok": bool(detail.get("engaged")), **detail},
            [str(detail.get("detail") or ""), f"engaged: {bool(detail.get('engaged'))}"],
        )

    if getattr(args, "create", False):
        if not peer:
            raise MeshRefusal(
                "peer_required", "--create needs --peer: the peer mints the session id"
            )
        detail = (
            _relay_call(
                "peer_session_create",
                peer=peer,
                cwd=str(getattr(args, "cwd", "") or ""),
                name=str(getattr(args, "name", "") or ""),
                prompt=str(getattr(args, "prompt", "") or ""),
                timeout=120.0,  # a spawn plus its first turn's admission
            )
            or {}
        )
        return _emit(
            args,
            {"ok": bool(detail.get("session_id")), **detail},
            [
                f"session: {detail.get('session_id') or '-'}",
                f"admitted: {bool(detail.get('admitted'))}",
                str(detail.get("detail") or ""),
            ],
        )

    if not peer and not getattr(args, "all_peers", False):
        raise MeshRefusal(
            "peer_required",
            "name a device with --peer, or ask every device with --all-peers",
        )
    payload = _relay_call("peer_session_rows") or {}
    remote = [row for row in (payload.get("sessions") or []) if isinstance(row, dict)]
    if peer:
        wanted = peer.lower()
        remote = [
            row
            for row in remote
            if str((row.get("peer") or {}).get("device_id") or "").lower() == wanted
            or str((row.get("peer") or {}).get("name") or "").lower() == wanted
        ]
    lines = []
    for row in remote:
        block = row.get("peer") or {}
        lines.append(
            f"{row.get('session_id')}  {block.get('name') or block.get('device_id') or '?'}"
            f"  {row.get('state') or '?'}  {row.get('conversation_name') or ''}".rstrip()
        )
    return _emit(
        args,
        {"ok": True, "sessions": remote, "peers": payload.get("peers") or {}},
        lines or ["no sessions are held by other devices right now"],
    )


def _cmd_status(args: argparse.Namespace) -> int:
    relay_mod = _import_relay()
    payload = relay_mod.status()
    relay_line = "not running"
    if payload["relay"]:
        relay_line = f"running, pid {payload['relay']['pid']}"
    lines = [
        f"installed:  {'yes' if payload['installed'] else 'no'}"
        + ("" if payload["supported"] else "  (no launchd on this platform)"),
        f"identity:   {'present' if payload['identity_present'] else 'missing'}",
        f"relay:      {relay_line}",
        f"log:        {payload['log']}",
    ]
    for network in payload.get("networks") or []:
        links = network.get("links")
        count = len(links) if isinstance(links, list) else int(links or 0)
        lines.append(
            f"  {network['name']}  {network['network_id']}  epoch {network['epoch']}  "
            f"{count} link(s)"
        )
    return _emit(args, {"ok": True, **payload}, lines)


def _cmd_peers(args: argparse.Namespace) -> int:
    live = _relay_call("net_peer_ls")
    if live is None:
        return _emit(
            args,
            {"ok": False, "peers": [], "error": "the relay is not running"},
            ["the relay is not running; start it with `lop network start`"],
        )
    # ``net_peer_ls`` answers a LIST (a peer table), so the control client wraps it
    # as ``{"value": [...]}`` rather than pretending it is a mapping.
    rows = live.get("value")
    if not isinstance(rows, list):
        rows = live.get("peers") if isinstance(live.get("peers"), list) else []
    return _emit(
        args,
        {"ok": True, "peers": rows},
        [
            f"{'reachable' if row['reachable'] else 'unreachable':11} {row['device_id']}  "
            f"{row['name']}  {row.get('reason', '')}"
            for row in rows
        ]
        or ["no peers: this device is the only member of its networks"],
    )


def _cmd_disconnect(args: argparse.Namespace) -> int:
    record = _resolve(args.network)
    live = _relay_call("net_disconnect", network=record.network_id)
    if live is None:
        from local_operator.network import store
        from local_operator.network.relay import set_trust

        set_trust(record, trust="disconnected", reason="this device disconnected")
        secrets_file = store.secrets_path(record.network_id)
        if secrets_file.exists():
            secrets_file.unlink()
        live = {"network_id": record.network_id, "reachable_peers": 0, "secret_deleted": True}
    return _emit(
        args,
        {"ok": True, **live},
        [
            f"left {record.name}: peers notified where reachable "
            f"({live.get('reachable_peers', 0)}), links closed, local secret deleted, "
            "audit trail kept",
        ],
    )


def _cmd_panic(args: argparse.Namespace) -> int:
    record = _resolve(args.network)
    live = _relay_call("net_panic_local", network=record.network_id)
    if live is None:
        from local_operator.network import store
        from local_operator.network.relay import panic, set_trust

        state = store.require_secrets(record.network_id)
        member = record.self_member()
        is_admin = bool(member and "admin" in member.capabilities)
        panic(record, state, by=record.self_device_id, is_admin=is_admin)
        set_trust(record, trust="untrusted", reason="this device raised a panic")
        _audit(
            "panic_raised",
            network_id=record.network_id,
            epoch=record.epoch,
            actor=record.self_device_id,
            detail={
                "epoch_before": record.epoch - (1 if is_admin else 0),
                "epoch_after": record.epoch,
                "reachable_peers": 0,
            },
        )
        live = {
            "network_id": record.network_id,
            "epoch": record.epoch,
            "rotated": is_admin,
            "broadcast_to": 0,
        }
    return _emit(
        args,
        {"ok": True, **live},
        [
            f"panic raised on {record.name}: epoch {live.get('epoch')}"
            + (
                ", every other device is told to stop trusting the network"
                if live.get("rotated")
                else ""
            ),
            "each device must be re-admitted with `lop network trust <net> --active`",
        ],
    )


def _cmd_trust(args: argparse.Namespace) -> int:
    target = "active" if args.active else "untrusted" if args.untrusted else "active"
    record = _resolve(args.network)
    live = _relay_call("net_trust_local", network=record.network_id, trust=target)
    applied_locally = live is None
    if live is None:
        from local_operator.network.relay import set_trust

        set_trust(record, trust=target, reason="operator")
        _audit(
            "trust_changed",
            network_id=record.network_id,
            epoch=record.epoch,
            actor=record.self_device_id,
            detail={"from": record.trust, "to": target, "reason": "operator"},
        )
        live = {"network_id": record.network_id, "trust": target}
    return _emit(
        args,
        {"ok": True, **live, "applied_locally": applied_locally},
        [
            f"{record.name} is now {target}"
            + (" (the relay is not running: applied locally)" if applied_locally else "")
        ],
    )


def _cmd_uninstall(args: argparse.Namespace) -> int:
    """Remove the LaunchAgent; ``--purge`` forgets networks; ``--purge-identity``
    destroys the device keypair behind a typed confirmation.

    The receipt describes EXACTLY what was deleted, because the previous version's
    summary ("deleted this device's mesh identity and network records") was true of
    neither half: it removed top-level files only, and it claimed the keypair.
    """
    relay_mod = _import_relay()
    result = relay_mod.uninstall(
        purge=bool(args.purge),
        purge_identity=bool(args.purge_identity),
        networks=list(args.purge_networks or []) or None,
    )
    deleted = result.get("deleted") or {}
    lines = [*[f"  {step}" for step in result.get("steps", [])]]
    if deleted:
        lines.append(f"  deleted: {json.dumps(deleted, sort_keys=True)}")
    lines.append(f"  device identity: {result.get('identity', 'kept')}")
    return _emit(args, {"ok": bool(result.get("ok")), **result}, lines)


# ---------------------------------------------------------------------------
# Reading: log, doctor, identity
# ---------------------------------------------------------------------------


def _cmd_log(args: argparse.Namespace) -> int:
    """The audit log: read, follow, or export.

    Export exists because retention is a DEFAULT, not a guarantee: a copy taken
    before the caps prune it is never touched by rotation, which is what an incident
    review needs.
    """
    from local_operator.network import store
    from local_operator.network.audit import AuditLog

    if args.export is not None:
        path = store.audit_path()
        if not path.exists():
            return _emit(args, {"ok": False, "error": "no audit log yet"}, ["no audit log yet"])
        args.export.write_text(path.read_text(encoding="utf-8"), encoding="utf-8")
        return _emit(
            args,
            {"ok": True, "exported": str(args.export), "bytes": args.export.stat().st_size},
            [f"exported the audit log to {args.export}"],
        )
    since: float | None = None
    if args.since:
        seconds = _maybe_duration(args.since)
        if seconds is None:
            print(f"--since {args.since!r} is not a duration like 15m or 2h", file=sys.stderr)
            return 2
        since = time.time() - seconds
    log = AuditLog()
    if args.follow:
        import subprocess

        path = str(store.audit_path())
        print(f"following {path} (ctrl-c to stop)")
        return subprocess.call(["tail", "-F", "-n", str(args.limit), path])
    records = log.tail(int(args.limit), network_id=args.network or None, since=since)
    if _json_mode(args):
        print(json.dumps({"ok": True, "records": records}, indent=2, sort_keys=True, default=str))
        return 0
    for record in records:
        detail = record.get("detail") or {}
        print(
            f"{record.get('ts_iso')}  {record.get('event'):22} {record.get('outcome'):8} "
            f"{record.get('actor', ''):12} {json.dumps(detail, sort_keys=True) if detail else ''}"
        )
    return 0


def _maybe_duration(text: str) -> float | None:
    if not text:
        return None
    try:
        return _duration(text)
    except argparse.ArgumentTypeError:
        return None


def _cmd_confirm(args: argparse.Namespace) -> int:
    """Answer a parked pairing: show both codes, then admit or refuse.

    THE INVITER'S HALF of §5.3's human step, on the device whose relay is a
    launchd daemon and therefore has no terminal to prompt at. The relay parks the
    pairing in a 0600 record carrying the code THIS device derived and the code the
    joiner transcribed; this command shows both to a person and records their
    answer where the waiting pairing loop will pick it up.

    NO TERMINAL, NO ANSWER. Without a TTY this refuses (naming the foreground
    `serve` alternative), so an agent cannot complete a pairing by hand — the same
    property the joiner's prompt enforces from the other side. ``--sas-stdin`` is
    the harness's seam and is refused outside ``LOP_NETWORK_TEST_MODE=1``.
    """
    from local_operator.network import store, types, wire

    rows = _pending_pairings()
    if args.list_pending:
        return _emit(
            args,
            {"ok": True, "pending": rows},
            [f"{row['invite_id']}  {row['joiner_device_id']}\n{row['prompt']}" for row in rows]
            or ["no device is waiting to pair with this one"],
        )
    wanted = args.invite_id
    chosen = next((row for row in rows if row.get("invite_id") == wanted), None) if wanted else None
    if chosen is None and rows and not wanted:
        chosen = rows[0]
    if chosen is None:
        raise types.MeshRefusal(
            "no_pending_pairing",
            "no device is waiting to pair with this one"
            + (f" under invite {wanted}" if wanted else "")
            + ". Run `lop network confirm --list` to see the queue, or mint an invite "
            "with `lop network invite`.",
        )

    invite_id = str(chosen.get("invite_id") or "")
    admit = False
    if not args.decline:
        print(str(chosen.get("prompt") or ""))
        print(
            f"YOUR screen shows {wire.sas_display(str(chosen.get('sas') or ''))}; the other "
            f"device should show the same six digits."
        )
        answer = _read_confirmation(args)
        admit = answer == "yes"
    live = _relay_call(
        "net_pair_confirm",
        invite_id=invite_id,
        decision="admit" if admit else "decline",
        matched=admit,
        reason="" if admit else "declined",
        answered_by="harness" if args.sas_stdin else "human",
    )
    if live is None:
        # No relay running: the pairing loop cannot be waiting either, so this
        # records the answer for a relay that starts later — and says so, rather
        # than reporting a success that nobody is watching for.
        decision = types.PairDecision(
            invite_id=invite_id,
            decision="admit" if admit else "decline",
            matched=admit,
            reason="" if admit else "declined",
        )
        store.save_pair_decision(decision, None)
        _audit(
            "pairing_confirmed" if admit else "pairing_refused",
            actor=str(chosen.get("joiner_device_id") or ""),
            subject=str(chosen.get("network_id") or ""),
            cause="" if admit else "declined",
        )
        live = {
            "invite_id": invite_id,
            "decision": decision.decision,
            "matched": admit,
            "joiner_device_id": chosen.get("joiner_device_id"),
            "relay": "not running: the answer is recorded for when it is",
        }
    return _emit(
        args,
        {"ok": True, **live},
        [
            (
                f"admitted {chosen.get('joiner_device_id')} to {chosen.get('network_name')}"
                if admit
                else f"refused the pairing with {chosen.get('joiner_device_id')}; the "
                "invite is burned"
            )
        ],
    )


def _pending_pairings() -> list[dict[str, Any]]:
    """The parked pairings: the relay's view when it runs, the files otherwise."""
    from local_operator.network import store

    live = _relay_call("net_pair_pending")
    rows = live.get("value") if isinstance(live, dict) else None
    if isinstance(rows, list):
        return [row for row in rows if isinstance(row, dict)]
    return [pending.to_json() for pending in store.pending_pairings()]


def _read_confirmation(args: argparse.Namespace) -> str:
    """The human's answer, or a refusal explaining how to give one."""
    import os

    from local_operator.network import types

    if args.sas_stdin:
        if os.environ.get(TEST_MODE_ENV) != "1":
            raise types.MeshRefusal(
                "test_seam_closed",
                f"--sas-stdin is the test harness's seam and requires {TEST_MODE_ENV}=1. "
                "Run `lop network confirm` at a terminal: the comparison is the human "
                "check, and answering it from a script makes it a formality.",
            )
        typed = sys.stdin.readline().strip().lower()
        return "yes" if typed in ("y", "yes") else "no"
    if not _has_terminal():
        raise types.MeshRefusal(
            "confirm_needs_tty",
            "confirming a pairing needs a terminal you can answer at, because the code "
            "must be read off the OTHER device's screen by a person. Run this from a "
            "terminal, or run the relay in the foreground with `lop network serve` so "
            "its own prompt appears there.",
        )
    try:
        typed = input("do the two devices show the same code? [y/N] ").strip().lower()
    except (EOFError, KeyboardInterrupt):
        typed = ""
    return "yes" if typed in ("y", "yes") else "no"


def _has_terminal() -> bool:
    """Whether this process can ask a human directly (the relay's rule, reused)."""
    try:
        return bool(sys.stdin.isatty() and sys.stdout.isatty())
    except (ValueError, OSError):
        return False


def _cmd_doctor(args: argparse.Namespace) -> int:
    """Diagnose the mesh. Reports; never asserts reachability it has not proven."""
    live = _relay_call("net_doctor", peer=args.peer)
    payload = live if live is not None else _doctor_locally(args)
    lines = []
    for check in payload.get("checks", []):
        state = "ok " if check.get("ok") else "FAIL"
        lines.append(
            f"{state} {check.get('check', '')} {check.get('device_id', '')} "
            f"{check.get('endpoint', '')} {check.get('detail', '')}"
            + (f" {check['latency_ms']}ms" if check.get("latency_ms") is not None else "")
        )
    if not lines:
        lines.append("nothing to check: no networks, or no other members yet")
    if not payload.get("identity_present", True):
        lines.append(
            "this device has no device identity (identity_missing): run `lop network init`, or "
            "re-pair with a new invite"
        )
    return _emit(args, {"ok": True, **payload}, lines)


def _doctor_locally(args: argparse.Namespace) -> dict[str, Any]:
    """The checks that need no relay: identity, records, epochs, endpoints.

    Deliberately NOT a claim about reachability: with no relay running there is
    nothing here that can dial, and reporting "unreachable" from a process that
    never tried would be the dead-instrument failure the repo warns about.
    """
    from local_operator.network import store
    from local_operator.network.identity import identity_path

    checks: list[dict[str, Any]] = []
    identity_file = identity_path()
    checks.append(
        {
            "check": "identity",
            "ok": identity_file.exists(),
            "detail": "present" if identity_file.exists() else "identity_missing",
        }
    )
    for record in store.list_networks():
        checks.append(
            {
                "check": "network",
                "ok": not record.stale,
                "detail": record.stale or "ok",
                "network_id": record.network_id,
            }
        )
        for member in record.active_members():
            if member.device_id == record.self_device_id:
                continue
            checks.append(
                {
                    "check": "endpoint",
                    "device_id": member.device_id,
                    "endpoint": (member.endpoints or [""])[0],
                    "ok": bool(member.endpoints),
                    "detail": (
                        "not probed (the relay is not running)"
                        if member.endpoints
                        else "no_endpoint"
                    ),
                }
            )
    return {
        "checks": checks,
        "identity_present": identity_file.exists(),
        "identity_dir": str(identity_file.parent),
        "relay": "not running",
    }


def _cmd_identity_show(args: argparse.Namespace) -> int:
    from local_operator.network.identity import load

    identity = load()
    if identity is None:
        return _emit(
            args,
            {"ok": False, "error": "identity_missing"},
            ["this device has no mesh identity yet; run `lop network init`"],
        )
    return _emit(
        args,
        {
            "ok": True,
            "device_id": identity.device_id,
            "name": identity.name,
            "generation": identity.generation,
            "fingerprint": identity.fingerprint(),
            "rotated_from": identity.rotated_from,
        },
        [
            f"device id   {identity.device_id}",
            f"fingerprint {identity.fingerprint()}",
            f"name        {identity.name}   generation {identity.generation}",
            "the private key stays in a 0600 file and is never printed, logged or sent",
        ],
    )


def _cmd_identity_rotate(args: argparse.Namespace) -> int:
    """Replace the device key and announce the change to every network.

    A rotation mints a NEW device id (an id is a key fingerprint, so it cannot
    survive a key change) and produces a statement signed by the OLD key — which is
    what proves continuity, and which is why the old identity object is returned by
    ``rotate`` rather than discarded.
    """
    from local_operator.network import store
    from local_operator.network.identity import load, rotate

    previous = load()
    if previous is None:
        return _emit(args, {"ok": False, "error": "identity_missing"}, ["no identity to rotate"])
    new, old = rotate()
    relay_mod = _import_relay()
    announced = {"sent": 0, "queued": 0}
    if relay_mod.health() is not None:
        server = relay_mod.RelayServer(identity=new)
        announced = relay_mod.announce_identity_rotation(server, old, new)
        server.stop()
    else:
        for record in store.list_networks():
            if record.self_device_id == old.device_id:
                record.self_device_id = new.device_id
                member = record.member(old.device_id)
                if member is not None:
                    member.previous_ids = [*member.previous_ids, member.device_id]
                    member.device_id = new.device_id
                    member.public_key = new.public_key
                    member.rotated_at = time.time()
                store.save(record)
        announced = {"sent": 0, "queued": 0}
    _audit(
        "device_rotated",
        actor=new.device_id,
        detail={"old_device": old.device_id, "new_device": new.device_id},
    )
    return _emit(
        args,
        {
            "ok": True,
            "device_id": new.device_id,
            "previous_device_id": old.device_id,
            "generation": new.generation,
            "announced": announced,
        },
        [
            f"rotated this device from {old.device_id} to {new.device_id}",
            f"announced to {announced['sent']} peer(s); queued for {announced['queued']}",
            "a peer that never receives the statement will see this device as unknown and must "
            "re-pair — nothing can prove continuity without the old key",
        ],
    )


# ---------------------------------------------------------------------------
# Small shared imports kept in one place
# ---------------------------------------------------------------------------


def _import_relay() -> Any:
    from local_operator.network import relay

    return relay


def _replace(settings: Any, **changes: Any) -> Any:
    from dataclasses import replace

    return replace(settings, **changes)


def _audit(event: str, **fields: Any) -> None:
    """Record one semantic event from a CLI mutation.

    The relay owns the log while it runs; a CLI mutation with the relay down is
    still a semantic event, and an unrecorded `member rm` is exactly the gap an
    incident review cannot afford. Best effort by construction (``AuditLog.record``
    never raises).
    """
    from local_operator.network.audit import AuditEvent, AuditLog

    log = AuditLog()
    log.record(AuditEvent(event=event, **fields))
    log.close()


def _guard_member_subcommand(args: argparse.Namespace) -> int:
    """``lop network member`` with no verb is a usage error, not the destructive one.

    A group whose default action is ``rm`` would make a typo revoke a member, so the
    verb is required and the message says which one.
    """
    if getattr(args, "member_command", None) != "rm":
        print("usage: lop network member rm <network> <device>", file=sys.stderr)
        return 2
    return _cmd_member_rm(args)


def _guard_identity_subcommand(args: argparse.Namespace) -> int:
    """``lop network identity`` defaults to showing the device, never to rotating it.

    Rotating a device key is unrecoverable for a peer that misses the statement, so
    it can only be reached by naming it.
    """
    if getattr(args, "identity_command", None) == "show":
        return _cmd_identity_show(args)
    if getattr(args, "identity_command", None) == "rotate":
        return _cmd_identity_rotate(args)
    return _cmd_identity_show(args)


#: The dispatch table. Defined after the guards because it NAMES them, and a table
#: that referred to a function defined below it would be a NameError at import — on
#: the CLI's startup path, which is the one place a typo here would be paid by every
#: ``lop`` invocation.
_HANDLERS: dict[str, Callable[[argparse.Namespace], int]] = {
    "init": _cmd_init,
    "invite": _cmd_invite,
    "join": _cmd_join,
    "ls": _cmd_ls,
    "show": _cmd_show,
    "rename": _cmd_rename,
    "rm": _cmd_rm,
    "member": _guard_member_subcommand,
    "peers": _cmd_peers,
    "sessions": _cmd_sessions,
    "serve": _cmd_serve,
    "start": _cmd_service("start"),
    "stop": _cmd_service("stop"),
    "restart": _cmd_service("restart"),
    "status": _cmd_status,
    "disconnect": _cmd_disconnect,
    "panic": _cmd_panic,
    "trust": _cmd_trust,
    "log": _cmd_log,
    "confirm": _cmd_confirm,
    "doctor": _cmd_doctor,
    "identity": _guard_identity_subcommand,
    "uninstall": _cmd_uninstall,
}
