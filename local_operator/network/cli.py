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
from typing import Any, Callable, Sequence

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
    # The credential broker's surfaces (mesh-credentials.md; build plan §2.2).
    # ``credentials`` (plural) is the READ: what this device owns and what it
    # borrows. ``credential`` (singular) is the ACT: share or revoke one, on the
    # device that owns it. The two spellings are the family's own convention
    # (``peers`` lists, ``member`` acts) and the split is deliberate: a listing
    # that could also mutate is a listing nobody can run to find out what is true.
    "credentials",
    "credential",
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
    # GRANT/REVOKE edit what a peer may do ON THIS DEVICE (build plan §0 finding 3):
    # the default `drive` role cannot move, delete or borrow a login, and the only
    # other way to widen it is a re-pair that burns the device id.
    for verb, words in (
        ("grant", "Allow a peer more on this device (move, delete, broker_credential, ...)"),
        ("revoke", "Take capabilities back from a peer on this device"),
    ):
        caps = member_actions.add_parser(verb, help=words)
        caps.add_argument("network")
        caps.add_argument("device")
        caps.add_argument("capabilities", nargs="+", metavar="capability")
        caps.add_argument("--json", action="store_true")

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
    # ``--yolo`` IS ACCEPTED HERE AND THEN DECLINED by this verb's ``--create`` (see
    # ``_cmd_sessions`` and both relay guards), so the global help sentence — which
    # ``cli._propagate_global_flags`` adds to every subcommand — advertised something
    # this verb refuses (QA round 1, Q3). The flag is still DECLARED THERE, once, so
    # that ``--create --yolo`` gets the refusal sentence instead of "unrecognized
    # arguments"; only the sentence about it changes. ``setattr`` rather than the bare
    # attribute form because ``ArgumentParser`` is a typed object and a direct
    # assignment is a pyright error.
    setattr(net_sessions, "yolo_is_refused", True)
    net_sessions.add_argument("--peer", default="", help="the device to ask (id or name)")
    net_sessions.add_argument(
        "--all-peers", action="store_true", help="list every peer's sessions, merged"
    )
    net_sessions.add_argument("--create", action="store_true", help="create a session on --peer")
    net_sessions.add_argument("--engage", metavar="SESSION", default="", help="warm that session")
    net_sessions.add_argument("--stop", metavar="SESSION", default="", help="stop that session")
    # THE THREE LIFECYCLE ACTS RUN ON THE OWNER (design §8). They are flags on this
    # verb rather than a new subcommand because they are the same question as
    # `--stop` ("do a thing to a conversation that lives elsewhere") and they take
    # the same `--peer`.
    net_sessions.add_argument(
        "--archive", metavar="SESSION", default="", help="hide that session on its device"
    )
    net_sessions.add_argument(
        "--unarchive", metavar="SESSION", default="", help="restore that session on its device"
    )
    net_sessions.add_argument(
        "--delete",
        metavar="SESSION",
        default="",
        help="delete that session on its device (a dry run until --yes)",
    )
    net_sessions.add_argument(
        "--yes",
        action="store_true",
        help="with --delete: actually delete it (without this the owner only rehearses)",
    )
    net_sessions.add_argument(
        "--force",
        action="store_true",
        help="with --stop: signal a target whose turn is in flight (as `lop stop --force`)",
    )
    net_sessions.add_argument("--cwd", default="", help="with --create: where it should run")
    net_sessions.add_argument("--name", default="", help="with --create: its title")
    net_sessions.add_argument("--prompt", default="", help="with --create: its first turn")
    # WHO THE SESSION IS (definitions.py). Three flags, each naming a DISTINCT
    # thing the local product can put on a session, because collapsing them would
    # make one word mean two and the frame already has to say which it got:
    #
    # * ``--profile`` — an attachable persona (a role, a specialist or a packaged
    #   seed). This is ``exec --profile``'s word and ``/agent``'s behaviour.
    # * ``--agent-id`` / the global ``--agent NAME`` — a LEGACY named agent row,
    #   which selects its own hosting/model/prompt (``--agent``'s word, and the
    #   flag is the global one rather than a second spelling declared here:
    #   ``parent_parser`` already defines it and a duplicate option string on a
    #   child parser is an argparse conflict, so the name is reused rather than
    #   shadowed). Documented in the help text below.
    # * ``--team`` — a team whose roster and briefs this session manages.
    net_sessions.add_argument(
        "--profile",
        default="",
        help="with --create: the role/specialist/seed the session runs as (see --agent for "
        "a legacy named agent)",
    )
    # ``--agent`` IS DECLARED HERE rather than inherited. The ``network``
    # subcommands are built with ``parents=[parent_parser]`` only when a caller
    # supplies one, and ``sessions`` is created without it — so the product's
    # global ``--agent NAME`` was not a flag this verb accepted at all, and a user
    # typing the spelling they use everywhere else got "unrecognized arguments".
    # The name and the dest match the global flag exactly (``agent_name``), so the
    # two spellings cannot come to mean different things when both are accepted.
    net_sessions.add_argument(
        "--agent",
        "--agent-name",
        dest="agent_name",
        default="",
        help="with --create: a legacy named agent to run the session as",
    )
    net_sessions.add_argument(
        "--agent-id",
        default="",
        dest="create_agent_id",
        help="with --create: a legacy agent by ID (the global --agent NAME selects one by name)",
    )
    net_sessions.add_argument(
        "--team", default="", help="with --create: the team whose roster the session manages"
    )
    net_sessions.add_argument(
        "--effort",
        default="",
        help="with --create: the reasoning level the session is born on",
    )
    net_sessions.add_argument("--json", action="store_true")

    # `lop network definitions`: what makes an agent or a team resolvable on the
    # OTHER device. Two verbs — `push` (reconcile this device's definitions onto a
    # peer or every peer) and `state` (what this device holds and what it last
    # received). Read-only by default, like `sessions`.
    definitions = actions.add_parser(
        "definitions",
        help="Sync agent and team definitions so a peer can resolve a name",
        description=(
            "A session created on another device can name an agent profile or a team, "
            "and that device needs the definition to resolve the name. `push` sends "
            "this device's definitions; a create that names one pushes it first, so "
            "this verb is for doing it deliberately — a cleaned-up peer, or a new "
            "machine you have just paired."
        ),
    )
    definitions_actions = definitions.add_subparsers(dest="definitions_command")
    definitions_push = definitions_actions.add_parser(
        "push", help="Send this device's agent and team definitions to a peer"
    )
    definitions_push.add_argument("--peer", default="", help="the device to send them to")
    definitions_push.add_argument("--all-peers", action="store_true", help="every linked peer")
    definitions_push.add_argument("--json", action="store_true")
    definitions_state = definitions_actions.add_parser(
        "state", help="What this device holds, and what it mirrored from a peer"
    )
    definitions_state.add_argument("--json", action="store_true")

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

    actions.add_parser(
        "credentials",
        help="What this device owns, and what it borrows from whom",
    ).add_argument("--json", action="store_true")

    credential = actions.add_parser("credential", help="Share or revoke one credential")
    credential_actions = credential.add_subparsers(dest="credential_command")
    cred_share = credential_actions.add_parser(
        "share", help="Let another device borrow a credential this one holds"
    )
    cred_share.add_argument("key", help="A provider name, or mcp:<server-url>")
    cred_share.add_argument(
        "--with",
        required=True,
        dest="device",
        help="The device that may borrow it (name or device id)",
    )
    cred_share.add_argument(
        "--scope",
        choices=("session", "device"),
        default="session",
        help="'session' bounds the grant to the session that asks (the default)",
    )
    # ``--network`` ON BOTH VERBS, because the handler resolves one (QA round 1, Q1):
    # it read ``args.network`` and the parser never defined it, so every share and
    # revoke typed at a shell died with ``AttributeError`` — the only test set the
    # attribute by hand. Empty resolves the one network this device is in, exactly
    # like ``invite``/``log``; with several, ``_resolve`` asks for the name.
    cred_share.add_argument(
        "--network", default="", help="Network name or id (default: the only one)"
    )
    cred_share.add_argument("--json", action="store_true")
    cred_revoke = credential_actions.add_parser(
        "revoke", help="Stop letting a device borrow a credential"
    )
    cred_revoke.add_argument("key", help="A provider name, or mcp:<server-url>")
    cred_revoke.add_argument(
        "--from",
        required=True,
        dest="device",
        help="The device that may no longer borrow it (name or device id)",
    )
    cred_revoke.add_argument(
        "--network", default="", help="Network name or id (default: the only one)"
    )
    cred_revoke.add_argument("--json", action="store_true")

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


#: The listing verbs' client-side deadline is `relay.LISTING_CLIENT_TIMEOUT_S`:
#: the relay's own probe budget plus slack, so a client outwaits the server it
#: asked instead of inventing its own, shorter deadline. It is NOT spelled here
#: — three inline ``+ 8.0`` copies of one deadline is how the sidebar's peer
#: catalogue came to use the socket default and silently list nothing (QA round
#: 10, Q-R10-1).


def _listing_timeout() -> float:
    """How long a listing verb waits for THIS device's relay.

    The number lives beside the relay's own probe budget
    (``relay.LISTING_CLIENT_TIMEOUT_S``) rather than here, because it is the
    same wait the sidebar's peer catalogue makes with no command line in front
    of it: one fan-out, one deadline.
    """
    return float(_import_relay().LISTING_CLIENT_TIMEOUT_S)


#: The refusal code the whole `lop network` family carries when this device's own
#: relay could not be asked. ONE SPELLING, because it is the string a `--json`
#: consumer branches on (`peers` is the verb the guide documents it on), and a
#: second spelling of one code is how a family comes to read as two voices.
CODE_RELAY_UNAVAILABLE = "relay_unavailable"


def _relay_call(
    op: str, *, timeout: float = 5.0, allow_no_answer: bool = False, **fields: Any
) -> dict[str, Any] | None:
    """Run one op on the running relay, or REFUSE when no relay answers it.

    ``timeout`` is a parameter because the session-plane ops are not all fast:
    a spawn takes seconds and the kill-switch ladder is allowed MINUTES (its
    SIGTERM rung waits out a drain the receiver owns). The default stays the
    original 5 s so every fast status op behaves exactly as it did; the slow
    ones pass their own budget and say why at the call site.

    A MISSING ANSWER IS NOT AN OUTCOME (QA round 4, Q-R4-1). This used to return
    ``None`` for "the relay did not answer", and every caller read that ``None``
    for itself: the session family collapsed it to ``{}`` and then DERIVED ``ok``
    from the absence, so ``sessions --stop`` answered ``{"ok": true}`` with rc 0
    for a stop that never left this device, ``--engage`` answered a bare
    ``{"ok": false}`` with no code and no sentence, and ``--all-peers`` presented
    an empty peer set as a result. So the unavailable condition is a refusal now,
    raised where the answer is lost rather than re-interpreted at each call site,
    carrying the family's ``code`` + ``message`` (the same shape ``peers`` ships)
    with a sentence that names the remedy.

    ``allow_no_answer`` is the deliberate opt-out for a caller that handles a
    missing answer ITSELF: either because the verb has a LOCAL spelling of the same
    op (``ls``/``show``/``invite``/``member_rm``/``disconnect``/``panic``/``trust``/
    ``confirm``/``doctor``/``pending``, each of which does that work here and says so
    in its own payload), or because it emits the family's refusal document itself
    (``peers``). Taking it means accepting that no peer was asked. A verb whose whole
    contract is "did the relay/peer do it" must not take it — inventing an outcome
    for a thing that never happened is the defect this parameter exists to make
    visible.
    """
    from local_operator.network import relay, store

    record = store.find_own_relay()
    reply = relay.control_request(record, op, timeout=timeout, **fields) if record else None
    if reply is None:
        if allow_no_answer:
            return None
        from local_operator.network.types import MeshRefusal

        raise MeshRefusal(CODE_RELAY_UNAVAILABLE, _relay_unavailable_message())
    if reply.get("op") == "error":
        from local_operator.network.types import MeshRefusal

        # THE INNER CODE CROSSES THIS HOP (review round 2). The local control path
        # composes ``code``/``message`` for exactly this reason, and overwriting the
        # code with a flat ``relay_refused`` threw the reason away: a surface could see
        # that the relay refused but never which refusal it was (`definition_conflict`
        # and `session_not_found` read identically), and the sentence was the only
        # thing left to branch on. The fallback stays for an older relay that sends a
        # sentence only.
        raise MeshRefusal(
            str(reply.get("code") or "relay_refused"),
            str(reply.get("message") or "the relay refused"),
        )
    detail = reply.get("detail")
    return detail if isinstance(detail, dict) else {"value": detail}


def _relay_answer(op: str, *, timeout: float = 5.0, **fields: Any) -> dict[str, Any]:
    """``_relay_call`` for a verb with NO local spelling: the answer, or a refusal.

    The session-piloting verbs (``--stop``/``--engage``/``--create`` and the
    merged listing) all ask this device's relay to ask a peer, so a missing answer
    is a refusal and never a result. ``_relay_call`` already raises for it; the
    ``None`` branch survives so a future caller that flips ``allow_no_answer`` on
    gets the same refusal instead of a ``TypeError`` on its way to an invented
    outcome.
    """
    detail = _relay_call(op, timeout=timeout, **fields)
    if detail is None:  # pragma: no cover — ``_relay_call`` refuses before this
        from local_operator.network.types import MeshRefusal

        raise MeshRefusal(CODE_RELAY_UNAVAILABLE, _relay_unavailable_message())
    return detail


def _reported(detail: dict[str, Any], field: str, *, verb: str) -> Any:
    """The one field a verb's receipt rests on, or a refusal naming the gap.

    A RECEIPT THAT DOES NOT CARRY THE FACT IT IS A RECEIPT FOR IS NOT A RECEIPT
    (QA round 4, Q-R4-1). ``--stop`` derived ``ok`` from
    ``outcome not in ("", "refused")``, which is TRUE for a missing ``outcome``,
    so any answer that lost the field read as a success; ``--engage`` and
    ``--create`` read a missing ``engaged``/``session_id`` as a bare
    ``{"ok": false}`` with nothing an operator could act on. The relay's own ops
    always send these fields (``relay._op_session_stop`` and its neighbours), so an
    answer without one is a relay that is not this build's — named, rather than
    guessed at.
    """
    from local_operator.network.types import MeshRefusal

    if not isinstance(detail, dict) or field not in detail:
        raise MeshRefusal(
            f"{verb}_unreported",
            f"this device's relay answered `{verb}` without reporting `{field}`, so whether "
            "the peer acted is unknown; restart the relay with `lop network restart` and "
            "ask again.",
        )
    return detail[field]


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
    """One network's row on the NO-RELAY path (`ls`/`show` when nothing answers).

    The membership block is present and says it is UNVERIFIED, because this row is
    built from the local record alone: no peer was asked, so the member count is
    this device's last known table rather than a checked one, and a count that does
    not say which it is is the failure that made an incomplete list act like an
    authoritative one (Q-R2-1). The state sentence is the same one the relay's own
    row carries (:func:`membership_state`), so a removed device reads the truth
    whether or not its relay is up — which is precisely when it has nothing else.
    """
    from local_operator.network.relay import membership_state

    state = membership_state(record)
    return {
        "network_id": record.network_id,
        "name": record.name,
        "epoch": record.epoch,
        "role": record.self_role,
        "trust": record.trust,
        "members": len(record.active_members()),
        "links": links,
        "stale": record.stale,
        "self_device_id": record.self_device_id,
        "membership_state": state["state"],
        "membership": {
            **state,
            "table": {
                "complete": False,
                "answered": [],
                "not_answered": [],
                "oldest_answer_age_s": None,
                "learned": [],
                "sentence": (
                    "members NOT verified: this device's relay did not answer, so no "
                    "peer was asked for its table"
                ),
            },
        },
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

    # ONE ANSWER TO "WHERE CAN PEERS REACH US", used for BOTH the record's listen
    # block and our own member row. They are the two things a joiner receives, so
    # deriving them separately is how the row said `endpoints: []` while the
    # record said `advertised: ["52.27.70.210:4200"]` (QA round 1, F-2).
    settings = _replace(settings, listen_address=listen_address, port=port)
    advertised = imported.advertise_endpoints(settings, declared=tuple(args.advertise_hosts or []))

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
            "advertised": advertised,
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
        # OUR OWN ROW CARRIES OUR OWN ENDPOINTS. It is the row every joiner
        # receives in the admission frame and the row every member receives in a
        # member list, so leaving it empty is what made a freshly paired device
        # permanently undialable (QA round 1, F-2).
        endpoints=list(advertised),
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
    from local_operator.network.relay import membership_lines, membership_marker

    # A LISTING THAT CONTACTS ITS PEERS, so it needs the relay's own budget plus
    # slack rather than the 5 s default: the table it reports is refreshed from the
    # members (Q-R2-1), and a short client timeout would fall back to the local
    # snapshot the refresh is there to replace. ``allow_no_answer`` because
    # ``ls``/``show`` DO have a local spelling: this device's own records, listed
    # with the table marked unverified (``_summarise``).
    live = _unwrap_list_answer(
        _relay_call("net_ls", timeout=_listing_timeout(), allow_no_answer=True)
    )
    records = store.list_networks()
    rows = live if live else [_summarise(record) for record in records]
    if not rows:
        return _emit(args, {"ok": True, "networks": []}, ["no networks on this device"])
    # THE LOCAL HALF OF THE STATE, STATED ON EVERY ROW (QA round 10, Q-R10-5).
    # ``trust`` answers "does this device still trust the network", which is not
    # the question an operator asks of the row: a network whose secret was
    # deleted by ``/network disconnect`` is trusted and unusable, and this line
    # used to say only ``active``. The secret file is a fact of THIS device, so
    # it is read once here and stamped on the rows whichever path built them —
    # the relay's checked table and the local ``_summarise`` fallback.
    missing: set[str] = set()
    for record in records:
        if not store.secrets_path(record.network_id).exists():
            missing.add(record.network_id)
    for row in rows:
        if isinstance(row, dict):
            row["secret_missing"] = row.get("network_id") in missing
    lines: list[str] = []
    for row in rows:
        lines.append(
            f"{row['name']}  {row['network_id']}  epoch {row['epoch']}  {row['role']}  "
            f"{row['members']} member(s)  {row['trust']}"
            + ("  [no secret — rejoin]" if row.get("secret_missing") else "")
            + (f"  [{row['stale']}]" if row.get("stale") else "")
            + membership_marker(row)
        )
        lines.extend(membership_lines(row))
    return _emit(args, {"ok": True, "networks": rows}, lines)


def _cmd_show(args: argparse.Namespace) -> int:
    # Same budget as `peers`/`ls`: `net_show` contacts every member before it
    # reports the table, so a 5 s client timeout would time out on the relay's own
    # work and answer from the stale local record.
    live = _relay_call(
        "net_show", network=args.network, timeout=_listing_timeout(), allow_no_answer=True
    )
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
    from local_operator.network.relay import membership_lines

    lines = [
        f"{payload.get('name')}  {payload.get('network_id')}  epoch {payload.get('epoch')}  "
        f"trust {payload.get('trust')}",
    ]
    # THE TABLE'S PROVENANCE, printed before the table itself: a member list is the
    # thing an operator reads and acts on, so what it rests on comes first.
    table = (payload.get("membership") or {}).get("table") or {}
    if int(payload.get("members") or 0) > 1 and table.get("sentence"):
        lines.append(f"  members: {table['sentence']}")
    lines.extend(membership_lines(payload))
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


def _cmd_member_caps(args: argparse.Namespace) -> int:
    """``member grant|revoke <net> <dev> <cap...>``: edit a peer's LOCAL row.

    The relay does it when running (it owns the record's writers); otherwise the
    same primitive runs here under the same lock. Either way the change is local
    to this device, needs no rotation and no broadcast, and is audited.
    """
    verb = args.member_command
    caps = [str(item) for item in args.capabilities]
    record = _resolve(args.network)
    from local_operator.network.relay import capability_change_lines

    if verb == "grant":
        change, applied = _apply_capability_change(record, args.device, grant=caps)
    else:
        change, applied = _apply_capability_change(record, args.device, revoke=caps)
    payload = {
        "ok": True,
        "network_id": record.network_id,
        "network": record.name,
        "device_id": change.device_id,
        "added": list(change.added),
        "removed": list(change.removed),
        "capabilities": list(change.capabilities),
        "changed": change.changed,
        "applied": applied,
    }
    return _emit(args, payload, capability_change_lines(change, network_name=record.name))


def _apply_capability_change(
    record: Any,
    device_id: str,
    *,
    grant: list[str] | tuple[str, ...] = (),
    revoke: list[str] | tuple[str, ...] = (),
) -> tuple[Any, str]:
    """Edit ``device_id``'s LOCAL member row, through the relay or in-process.

    ONE IMPLEMENTATION, extracted from ``_cmd_member_caps`` and shared with
    ``credential share`` — which must grant ``broker_credential`` on the owner for the
    device it just shared with, and must do it the same way ``member grant`` does.
    Two copies of this two-path write (relay when running, the same primitive under the
    same lock when not) would be two places for the lock and the audit record to be
    forgotten in one of them.

    ``Any`` for the change, not ``CapabilityChange``: it is defined in ``relay.py``,
    which this module imports lazily so ``lop network --help`` stays stdlib-fast.
    """
    from local_operator.network import store
    from local_operator.network.relay import (
        CapabilityChange,
        capability_change_event,
        set_member_capabilities,
    )

    fields: dict[str, Any] = {"grant": list(grant)} if grant else {"revoke": list(revoke)}
    live = _relay_call(
        "net_member_caps",
        network=record.network_id,
        device_id=device_id,
        allow_no_answer=True,
        **fields,
    )
    if live is not None:
        return (
            CapabilityChange(
                device_id=str(live.get("device_id") or device_id),
                name=_member_name(record, device_id),
                added=tuple(live.get("added") or ()),
                removed=tuple(live.get("removed") or ()),
                capabilities=tuple(live.get("capabilities") or ()),
            ),
            "relay",
        )
    from local_operator.network.audit import AuditLog

    with store.mutate(record.network_id) as fresh:
        change = set_member_capabilities(fresh, device_id=device_id, **fields)
        if change.changed:
            store.save(fresh)
            log = AuditLog()
            log.record(capability_change_event(fresh, change))
            log.close()
    return change, "locally (relay not running)"


def _cmd_credentials(args: argparse.Namespace) -> int:
    """``lop network credentials [--json]``: who owns what, and what this device borrows.

    THE PULL COMES FIRST. A share happens on the OWNER, and until this device has
    heard about it the local document does not know the key exists — so a listing that
    only read the file would report "nothing is shared with you" immediately after
    the operator shared something. The relay asks every other active member for its
    document (one bounded ``net_broker`` frame each), and the listing renders what it
    learned. ``allow_no_answer`` because a listing must still work with the relay
    down: it then shows what this device already knew, which is honest.
    """
    from local_operator.network.credentials.state import PlacementState
    from local_operator.network.identity import load as load_identity

    pulled = _relay_call("credential_placement", timeout=_listing_timeout(), allow_no_answer=True)
    identity = load_identity()
    self_device = identity.device_id if identity is not None else ""
    self_name = identity.name if identity is not None else ""

    # ENUMERATED FROM THE DOCUMENTS, NOT THE NETWORK RECORDS, and the difference is
    # not stylistic: the placement files are the authority for "what is shared", and a
    # device that has been shared a credential but whose record has not been re-read
    # would otherwise print "nothing is shared" — the exact state this listing exists
    # to make visible. The record is joined only for the NAME.
    networks: list[dict[str, Any]] = []
    lines: list[str] = []
    for document in _placement_documents():
        record = _record_for(document.network_id)
        network_name = record.name if record is not None else document.network_id
        state = PlacementState.load(document.network_id)
        keys: list[dict[str, Any]] = []
        if not document.entries:
            continue
        for key in sorted(document.entries):
            entry = document.entries[key]
            row: dict[str, Any] = {
                "key": key,
                "kind": entry.kind,
                "owner_device": entry.owner_device,
                "owner_device_name": entry.owner_device_name,
                "identity_label": entry.identity_label,
                "owned_here": entry.owner_device == self_device,
                "holders": [
                    {
                        "device": h.device,
                        "name": _member_name(record, h.device) if record is not None else "",
                        "scope": h.scope,
                    }
                    for h in entry.holders
                ],
            }
            if not row["owned_here"] and entry.is_holder(self_device):
                row["observation"] = state.status(key) or "not_asked"
            keys.append(row)
        if not keys:
            lines.append(f"{network_name}: nothing is shared in this network yet")
        else:
            lines.append(f"{network_name}:")
            for row in keys:
                owner = (
                    "this device"
                    if row["owned_here"]
                    else (row["owner_device_name"] or row["owner_device"])
                )
                who = f" ({row['identity_label']})" if row["identity_label"] else ""
                lines.append(f"  {row['key']:<14} {row['kind']:<15} owner: {owner}{who}")
                for holder in row["holders"]:
                    # The owner's own row and this device's are not "shares": the first
                    # is what makes the entry coherent and the second is the reader.
                    if holder["device"] in (self_device, row["owner_device"]):
                        continue
                    lines.append(
                        f"      shared with {holder['name'] or holder['device']} "
                        f"({holder['scope']})"
                    )
                if row.get("observation") and row["observation"] not in ("active", "not_asked"):
                    lines.append(f"      borrowed: {row['observation']}")
        networks.append(
            {
                "network_id": document.network_id,
                "network": network_name,
                "credentials": keys,
            }
        )
    payload = {
        "ok": True,
        "self_device": self_device,
        "self_device_name": self_name,
        "networks": networks,
        "refreshed": bool(pulled),
    }
    # A MEMBER WHOSE DOCUMENT COULD NOT BE MERGED IS SAID OUT LOUD (review round 5,
    # NIT 2). ``pull_placement`` refuses one unreadable document BY NAME and merges the
    # rest, which is the resilience we want — but that reply reached no surface: this
    # listing reduced the whole pull to ``bool(pulled)``, so a merge that used to raise
    # became INVISIBLE. Trading a loud failure for a silent one is not resilience. The
    # names go to STDERR (the listing's own output is the payload, and a machine
    # reading ``--json`` parses one document, not two), and the same list rides the
    # payload so a script can branch on it.
    skipped = [row for row in (pulled or {}).get("skipped") or [] if isinstance(row, dict)]
    if skipped:
        payload["skipped"] = skipped
        for row in skipped:
            device = str(row.get("device") or "")
            name = _device_name_anywhere(device)
            reason = str(row.get("reason") or "unreadable_document")
            print(
                f"\033[1;33m{name or device}: its sharing list could not be read "
                f"({reason}); the rest were merged. Nothing was changed for it — it is "
                f"merged on the next listing, or ask that device to re-share.\033[0m",
                file=sys.stderr,
            )
    return _emit(args, payload, lines)


def _cmd_credential(args: argparse.Namespace) -> int:
    """``credential share|revoke <key> --with/--from <dev>``: the owner's own verb.

    BOTH HALVES ARE REQUIRED AND NEITHER IS SUFFICIENT. The placement document is
    what the broker reads (``holders`` is the authorisation, and absence is a
    refusal), and ``broker_credential`` on the borrower's LOCAL member row is what the
    transport's chokepoint checks before the frame is even dispatched. A share that
    wrote only one of them would look granted and refuse at runtime, naming the
    capability rather than the share — so the two are written in one command and the
    payload reports both.
    """
    from local_operator.network.credentials import placement as placement_mod
    from local_operator.network.identity import load as load_identity
    from local_operator.network.types import MeshRefusal

    verb = str(getattr(args, "credential_command", "") or "")
    key = str(args.key)
    record = _resolve(args.network)
    identity = load_identity()
    if identity is None:
        raise MeshRefusal("no_identity", "this device has no key yet; join a network first")
    device = _resolve_device(record, args.device)
    if device is None:
        raise MeshRefusal(
            "not_a_member",
            f"{args.device!r} is not a device in {record.name}; run 'lop network show' for "
            "the member list",
        )
    if device == identity.device_id:
        raise MeshRefusal(
            "not_owner",
            "this device already uses its own login; a share is for ANOTHER device",
        )

    if verb == "share":
        kind, provider, label = _credential_shape(key)
        _require_local_credential(key, provider)
    else:
        kind, provider, label = "", "", ""

    with placement_mod.mutate(record.network_id, self_device=identity.device_id) as document:
        entry = document.entry(key)
        if verb == "share":
            if entry is None:
                # DECLARED BY THE DEVICE THAT HOLDS IT, which is why this verb only
                # runs where the credential is. ``declare`` refuses on behalf of another
                # device, so a share can never move ownership by accident.
                entry = document.declare(
                    key,
                    owner_device=identity.device_id,
                    owner_device_name=identity.name,
                    provider=provider,
                    kind=kind,
                    identity_label=label,
                    by=identity.device_id,
                )
            entry = document.grant(key, device, scope=str(args.scope), by=identity.device_id)
            action = "sharing"
        else:
            entry = document.revoke(key, device, by=identity.device_id)
            action = "revoked"
        entry_json = entry.to_json()

    capability = ""
    if verb == "share":
        _change, capability = _apply_capability_change(record, device, grant=["broker_credential"])
    else:
        # The capability goes with the share. Leaving ``broker_credential`` on a device
        # that no longer holds anything is a grant that can only ever be refused, and
        # the next reader of the member table would have to guess why.
        held = _held_keys(record, device)
        if not held:
            _change, capability = _apply_capability_change(
                record, device, revoke=["broker_credential"]
            )
    _audit_placement(record, identity.device_id, device, key, entry=entry_json, action=action)
    holders = [row["device"] for row in entry_json.get("holders") or []]
    payload = {
        "ok": True,
        "action": action,
        "key": key,
        "network_id": record.network_id,
        "network": record.name,
        "device": device,
        "device_name": _member_name(record, device),
        "scope": str(getattr(args, "scope", "") or ""),
        "holders": holders,
        "capability_applied": capability,
        "owner_device": entry_json.get("owner_device", ""),
    }
    name = _member_name(record, device) or device
    lines = [
        f"{action} {key!r} with {name}" if verb == "share" else f"{action} {key!r} from {name}",
        f"borrowers now: {', '.join(holders) or 'none'}",
        f"broker_credential on {name}: {capability or 'unchanged'}",
    ]
    if verb == "revoke":
        # THE TRUE REVOCATION LATENCY, said where the operator acts (QA round 1, Q5;
        # design §3.7). A revoke stops NEW grants at once, but no provider offers a
        # per-bearer revoke: an access token already lent keeps working AT THE
        # PROVIDER until it expires. This build's borrower drops it at the grant's
        # expiry (``grant_ttl_s``); a copy taken out of that process lives until the
        # token's own expiry. An incident response that believed "revoked" meant
        # "dead" would stop looking too early, so the payload and the lines say both.
        from local_operator.network.credentials import grant_ttl_s

        ttl_s = int(grant_ttl_s())
        # WHAT A COPY OUTLIVES DEPENDS ON THE CREDENTIAL IN HAND (review round 3, F3).
        # An OAuth access token dies at its own expiry; a STATIC API KEY never expires,
        # so "until the token expires" was a bound that does not exist — false in
        # exactly the case where the remedy matters most. Read from the entry the
        # revoke just wrote, so the receipt describes what was actually lent.
        static = str(entry_json.get("kind") or "") == "api-key-static"
        copied = (
            "valid at the provider until the key is rotated there (a static key never expires)"
            if static
            else "valid at the provider until the token expires"
        )
        payload["revocation"] = {
            "new_grants": "refused now",
            "lent_grant_max_s": ttl_s,
            "copied_bearer": copied,
        }
        lines.append(
            f"new borrows by {name}: refused now; a grant already lent is dropped by "
            f"{name} within {ttl_s} s"
        )
        if static:
            lines.append(
                f"a copy of the key taken out of that device never expires: to end it, "
                f"rotate the {key!r} key at the provider"
            )
        else:
            lines.append(
                "a bearer copied out of that device stays valid at the provider until the "
                f"token expires: to end it now, sign out of {key!r} at the provider"
            )
    return _emit(args, payload, lines)


def _credential_shape(key: str) -> tuple[str, str, str]:
    """``(placement kind, provider, identity label)`` for a key.

    Read from the OWNER's own store, so the document records what is actually signed
    in rather than what the operator typed: a ``kind`` that disagreed with the row
    would make the broker's narrowing rule and the operator's expectation diverge.
    """
    from local_operator.network.credentials.types import is_mcp_key, mcp_url_from_key

    config = _config_dir()
    if is_mcp_key(key):
        url = mcp_url_from_key(key)
        try:
            from local_operator.mcp.auth import McpTokenStorage

            if McpTokenStorage(url).has_stored_row():
                return "mcp-rotating", "mcp-oauth", ""
        except Exception:  # noqa: BLE001 — an unreadable store is "not signed in"
            pass
        return "mcp-rotating", "mcp-oauth", ""
    rows = _provider_rows(key, config)
    if not rows:
        return "oauth-rotating", key, ""
    row = rows[0]
    label = str(row.data.get("email") or row.data.get("account_id") or "")
    kind = "oauth-rotating" if row.credential_type == "oauth" else "api-key-static"
    return kind, key, label


def _require_local_credential(key: str, provider: str) -> None:
    """Refuse to share something this device does not hold.

    A declared entry for a credential that is not here would put a row in every
    device's document whose owner cannot serve it — the borrow would fail at
    ``no_local_credential`` with a sentence telling the operator to sign in on the
    device they just shared FROM, which is the confusing half of a lazy check.
    """
    from local_operator.network.credentials.types import is_mcp_key, mcp_url_from_key
    from local_operator.network.types import MeshRefusal

    if is_mcp_key(key):
        try:
            from local_operator.mcp.auth import McpTokenStorage

            if McpTokenStorage(mcp_url_from_key(key)).has_stored_row():
                return
        except Exception:  # noqa: BLE001
            pass
        raise MeshRefusal(
            "no_local_credential",
            f"this device has no MCP login for {mcp_url_from_key(key)}; run '/mcp login "
            f"{mcp_url_from_key(key)}' here first",
        )
    rows = _provider_rows(provider, _config_dir())
    if not rows:
        raise MeshRefusal(
            "no_local_credential",
            f"this device has no credential for {provider!r}; run 'lop login {provider}' "
            "here first — a share is only meaningful on the device that holds the login",
        )


def _provider_rows(provider: str, config: Any) -> list[Any]:
    """This device's rows for a provider, or ``[]``. Never raises."""
    try:
        from local_operator.providers.auth_store import AuthStore

        return list(AuthStore(config_dir=config).list_credentials(provider))
    except Exception:  # noqa: BLE001 — an unreadable store is "no credential here"
        return []


def _config_dir() -> Any:
    from local_operator.paths import config_dir

    return config_dir()


def _networks() -> list[Any]:
    from local_operator.network import store

    return list(store.list_networks())


def _placement_documents() -> list[Any]:
    """Every placement document on this device, loaded. The listing's authority."""
    from local_operator.network.credentials import placement as placement_mod

    return [
        placement_mod.PlacementDocument.load(path.parent.name)
        for path in sorted(
            placement_mod.credentials_root().glob(f"*/{placement_mod.PLACEMENT_FILENAME}")
        )
    ]


def _record_for(network_id: str) -> Any:
    """This device's record for ``network_id``, or ``None``. For the NAME only."""
    for record in _networks():
        if record.network_id == network_id:
            return record
    return None


def _held_keys(record: Any, device: str) -> list[str]:
    """Keys ``device`` still holds in ``record``'s network, so a revoke can decide
    whether the peer's ``broker_credential`` capability is still earning its place."""
    from local_operator.network.credentials import placement as placement_mod

    document = placement_mod.PlacementDocument.load(record.network_id)
    return [key for key in sorted(document.entries) if document.is_holder(key, device)]


def _resolve_device(record: Any, name: str) -> str | None:
    """A device id from a name, an id, or an unambiguous tail of one.

    Members are addressed by NAME in every human-facing surface and by ID on the
    wire, and an operator reading `lop network show` has both. Matching either here
    (and refusing an ambiguous tail) is what stops a share from silently landing on
    the wrong device — a mistargeted share spends the wrong machine's quota.
    """
    wanted = str(name or "").strip()
    if not wanted:
        return None
    for member in record.active_members():
        if member.device_id == wanted or member.name == wanted:
            return str(member.device_id)
    matches = [m for m in record.active_members() if m.device_id.endswith(wanted)]
    return str(matches[0].device_id) if len(matches) == 1 else None


def _audit_placement(
    record: Any, self_device: str, device: str, key: str, *, entry: dict[str, Any], action: str
) -> None:
    """One ``credential.placement`` record for a share or a revoke.

    Best effort: a log that cannot be written must not make a share the operator just
    made appear to have failed — the document is the record of truth, and this is the
    audit trail beside it.
    """
    try:
        from local_operator.network.audit import AuditEvent, AuditLog

        log = AuditLog()
        log.record(
            AuditEvent(
                event="credential.placement",
                actor=self_device,
                subject=device,
                network_id=record.network_id,
                epoch=record.epoch,
                actor_kind="device",
                detail={
                    "credential_key": key,
                    "act": self_device,
                    "sub": device,
                    "owner_device": str(entry.get("owner_device") or ""),
                    "holders": len(entry.get("holders") or []),
                },
            )
        )
        log.close()
    except Exception:  # noqa: BLE001 — see the docstring
        pass


def _device_name_anywhere(device_id: str) -> str:
    """A device's display name from ANY of this device's records, or ``""``.

    Searched across records rather than one network's: the pull reports the members it
    dialled, and this CLI renders a NAME beside an id wherever it has one (``_member_name``
    for a known network). A name is never load-bearing — an unresolved device prints its
    id — so a lookup that finds nothing is not a failure.
    """
    for record in _networks():
        name = _member_name(record, device_id)
        if name:
            return name
    return ""


def _member_name(record: Any, device_id: str) -> str:
    member = record.member(device_id)
    return str(member.name) if member is not None else ""


def _cmd_member_rm(args: argparse.Namespace) -> int:
    """Revoke a member: tombstone, rotate, bump the epoch, fan out (R5)."""
    # ``allow_no_answer``: a revocation HAS a local spelling — the tombstone, the
    # epoch rotation and the queue write all happen here (below), and the payload
    # says the rotation is queued rather than fanning out.
    live = _relay_call(
        "net_member_rm", network=args.network, device_id=args.device, allow_no_answer=True
    )
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
        allow_no_answer=True,
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

    # AN EXPIRED TOKEN IS A LOCAL FACT, so it is refused locally with its own code.
    # The listener deliberately answers a refused invite by CLOSING rather than by
    # explaining (an open port that explains is an oracle), so without this check
    # the most common pairing failure — an invite left in a chat log overnight —
    # reached the user as "could not join", i.e. as a network problem, and they
    # retried the network instead of minting a new token (QA round 1, F-4).
    now = time.time()
    expires_at = envelope.issued_at + envelope.ttl_s
    if expires_at <= now:
        raise MeshRefusal(
            "invite_expired",
            f"that invite expired {int(now - expires_at)}s ago. Invites are short-lived on "
            "purpose, so a token left where someone else could read it stops working; "
            "mint a fresh one on the other device with `lop network invite` and bring the "
            "new file across.",
        )

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
        if isinstance(link_result, str):
            # A per-host sentence, kept so the NEXT host is still tried and so the
            # final refusal can say what actually happened at each one.
            last_reason = link_result
            continue
        if link_result is None:
            continue
        return _emit(args, {"ok": True, **link_result[1]}, link_result[0])
    raise MeshRefusal(
        "join_failed",
        "could not join: "
        f"{last_reason or 'every endpoint in the invite refused or was unreachable'}. "
        "A token that was already redeemed, one the other device no longer accepts, and "
        "a wrong address all end the same way on the wire, because a peer that explains "
        "every refusal tells an attacker which tokens are real. If you are not certain "
        "the token is still open, mint a fresh one with `lop network invite`.",
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
) -> tuple[list[str], dict[str, Any]] | str | None:
    """One dial attempt: handshake, the human step, then admission.

    Returns ``(lines, payload)`` on success, a SENTENCE describing what happened at
    this host when it failed, or ``None`` when the host argument itself was
    unusable. The sentence is why the caller can say more than "could not join":
    "nothing was listening at 52.27.70.210:4200" and "the handshake stopped with
    ConnectionResetError" are different problems, and the second is what a refused
    or already-redeemed token looks like from here (QA round 1, F-4).
    """
    import socket

    from local_operator.network import wire
    from local_operator.network.handshake import refusal_from_pairing
    from local_operator.network.identity import mint_instance_id
    from local_operator.network.types import HandshakeRefusal, MeshRefusal

    Handshake = helpers["Handshake"]
    Credential = helpers["Credential"]
    store = helpers["store"]
    invite_mod = helpers["invite_mod"]
    relay_mod = helpers["relay_mod"]

    address, _, port_text = host.rpartition(":")
    try:
        port = int(port_text)
    except ValueError:
        return None
    try:
        sock = socket.create_connection(
            (address or host, port), timeout=settings.handshake_timeout_s
        )
    except OSError as exc:
        return f"nothing was listening at {host} ({exc.__class__.__name__})"
    deadline = wire.deadline_in(settings.handshake_timeout_s)
    # ONE ANSWER, used for the hello we send AND for our own durable record: the
    # inviter copies the hello's list onto our member row, and `listen` is what
    # every later reader of this record sees. Deriving them separately is how the
    # record came to claim the INVITER's address (QA round 1, F-7).
    advertised = relay_mod.advertise_endpoints(settings)
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
            # The inviter records THESE on our member row. Without them it kept the
            # observed source address, an ephemeral NAT port, so a paired device
            # could never be dialled back (QA round 1, F-2).
            endpoints=advertised,
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
        # pairing on a device whose relay is a daemon. It is also the SAME number the
        # prompt printed a moment ago — ``invite_mod.remaining_seconds`` is the one owner
        # of "what is left", so the promise and this wait cannot drift with the token's
        # age (agent review round 1, MAJOR 2).
        remaining = invite_mod.remaining_seconds(envelope)
        answer = codec.open(
            reader.read_record_payload(wire.deadline_in(helpers["pair_timeout_seconds"](remaining)))
        )
        if answer.get("op") == "net_pair_abort":
            # ``refusal_from_pairing`` OWNS the reason -> sentence map, and it lives
            # in ``handshake`` beside the frames it describes. This call used to go
            # through ``invite_mod``, which never had the helper: every wrong-SAS
            # pairing — a typo in six digits, the normal user path — died with
            # ``AttributeError: module 'local_operator.network.invite' has no
            # attribute 'sas_mismatch_sentence'`` instead of refusing (QA round 1,
            # F-1). Going through the one function also gets the invite-shaped
            # reasons (``invite_already_used``, ``invite_in_use``) their sentences.
            # The refusing device's own sentence comes too: it is the only place the
            # joiner can learn which id was refused and what to do about it (Q-R3-3).
            raise refusal_from_pairing(
                str(answer.get("reason") or "aborted"), detail=str(answer.get("detail") or "")
            )
        if not answer.get("admit"):
            raise MeshRefusal("not_admitted", "the other device did not admit this machine")
        record = _persist_join(
            answer,
            envelope,
            identity,
            host,
            store,
            peer_endpoints=handshake.peer_endpoints,
            advertised=advertised,
        )
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
    except HandshakeRefusal as refusal:
        # NOT named as a peer refusal: the listener closes the socket on a refusal
        # rather than explaining (an oracle would let a stranger probe token
        # validity), so what arrives here is a frame that never came. The sentence
        # is still the honest one, and it reaches the user instead of being
        # dropped on the floor.
        #
        # BEFORE ``MeshRefusal``, and it has to be: ``HandshakeRefusal`` is a
        # SUBCLASS of it, so the broader clause would swallow this one and the
        # sentence below would be unreachable.
        return f"the handshake at {host} stopped: {refusal.sentence}"
    except MeshRefusal:
        raise
    except (wire.LinkCryptoError, OSError, TimeoutError) as exc:
        return f"the handshake at {host} stopped ({exc.__class__.__name__})"
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
    answer: dict[str, Any],
    envelope: Any,
    identity: Any,
    host: str,
    store: Any,
    *,
    peer_endpoints: list[str] | None = None,
    advertised: list[str] | None = None,
) -> Any:
    """Write the network record and its secret from the admission frame.

    The member list comes from the frame because the joiner must hold the
    INVITER'S public key: without it, every later handshake from that device would
    be unverifiable, and a member list the joiner invented from the welcome frame
    would name a device it cannot check. ``material`` is written to the SEPARATE
    secrets file, never into the record.

    ``advertised`` is what THIS device says about itself, and it is the honest
    value: the record used to be written with ``advertised: [host]`` where ``host``
    is the INVITER's dial address, so a joiner's durable record claimed to be
    reachable at the other machine's address — a mis-dial waiting for the first
    consumer of ``listen.advertised`` (QA round 1, F-7). ``peer_endpoints`` is
    the inviter's own answer, from its ``welcome``; it fills the inviter's row
    when the admission frame's copy carries none (an older inviter, or one
    admitted before this build).
    """
    from local_operator.network.types import (
        MemberRecord,
        MeshRefusal,
        NetworkRecord,
        SecretState,
        trust_state,
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
    # WHERE WE CAN BE REACHED, from our own settings rather than from the address
    # we happened to dial.
    own_endpoints = [str(item) for item in (advertised or []) if str(item)]
    for row in rows:
        if row.device_id == identity.device_id:
            # OUR OWN ROW IS OUR DECLARATION, never the address the OTHER device
            # observed. The frame's copy of this row carries the inviter's fallback
            # when we declared nothing, which on a NAT'd joiner is an ephemeral port
            # — a dead address written into our own durable record. An empty list is
            # the honest answer here; this device's relay fills it from its live
            # listening port on its next publish.
            row.endpoints = list(own_endpoints)
        elif not row.endpoints:
            # The inviter's row. Prefer its own declared endpoints (from the
            # ``welcome``), and fall back to the address that actually worked —
            # the invite named it, so it is evidence, not a guess.
            row.endpoints = list(peer_endpoints or []) or [host]
    record = NetworkRecord(
        network_id=str(network.get("network_id") or envelope.network_id),
        name=str(network.get("name") or envelope.network_name),
        epoch=int(network.get("epoch") or envelope.epoch),
        sequence=int(network.get("sequence") or 0),
        trust=trust_state(network.get("trust") or "active"),
        self_device_id=identity.device_id,
        self_role=self_row.role if self_row else "read",
        self_capabilities=list(self_row.capabilities) if self_row else [],
        created_at=time.time(),
        created_by=envelope.inviter_device_id,
        # ``address``/``port`` are ours; ``advertised`` is ours. Nothing here names
        # the inviter any more.
        listen={"address": "", "port": 0, "advertised": list(own_endpoints)},
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
    # THE JOINER AUDITS ITS OWN ADMISSION. Every other mutating verb records one
    # and, being admitted is the mutation a joiner makes. Without this the joining
    # device's log stayed empty — no `audit.jsonl` was even created — while the
    # inviter had the full pairing history, so an incident review on the joiner
    # had nothing to read (QA round 1, F-3).
    _audit(
        "member_admitted",
        network_id=record.network_id,
        network_name=record.name,
        epoch=record.epoch,
        actor="self",
        subject=identity.device_id,
        # THE SAME DETAIL SHAPE THE INVITER WRITES for the same event, and only the
        # keys the audit writer's per-event whitelist keeps — a key the whitelist
        # does not know is dropped by the writer, so passing one here would claim a
        # record the file never gets.
        detail={"role": record.self_role, "member_kind": "device", "epoch": record.epoch},
    )
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
        payload: dict[str, Any] = {"ok": ok, "action": action}
        if not ok:
            # THE FAMILY'S SHAPE, even here: `code` + `message`, because a consumer
            # that reads `code` (which GUIDE.md tells it to) saw nothing at all on
            # this verb. The relay's own `reason` is the NAME when it has one —
            # `no_launchd` and `isolated_home` are diagnoses with their own
            # remedies, not launchctl failures — and `service_failed` covers the
            # rest. `error` is kept beside them: it is what this verb has always
            # carried.
            message = str(result.get("error") or "")
            payload["code"] = str(result.get("reason") or "") or "service_failed"
            payload["message"] = message
            payload["error"] = message
        return _emit(
            args,
            payload,
            [f"relay {action} ok" if ok else f"relay {action} failed: {payload.get('message')}"],
        )

    return _run


#: The peer's stop vocabulary that means "the session is no longer running", so
#: this side's ``ok`` is derived from a NAMED outcome instead of from the absence of
#: one (Q-R4-1). The words are ``relay._STOP_OUTCOME_WORD``'s, i.e. the peer's own
#: ``control.StopOutcome.method`` rendered for a viewer, and the set mirrors
#: `control.ENDED_METHODS` ("the stop did its job"). ``not_running`` belongs here
#: because it IS the answer to "did it stop": the peer holds no runtime for that id
#: and the relay emits it as an outcome rather than an error (`relay._op_session_stop`).
#: ``refused`` (no identity proof) and ``skipped`` (busy, deliberately left alone,
#: `control.LEFT_ALONE_METHODS`) are the two that did NOT act — a command that did
#: not act must never report success, and the old
#: ``outcome not in ("", "refused")`` reported success for both of those AND for a
#: missing outcome.
_STOP_ENDED_OUTCOMES = frozenset({"stopped", "killed", "already-gone", "not_running"})


#: The session plane's listing header (UX round 5, U29), and the grid its rows print
#: into (review round 9, NIT).
#:
#: The rows are four CLI columns that had no header, so three of them were for the
#: reader to infer from shape alone — the same defect the local `lop sessions`
#: table never had (its columns are named, ``cli.STATE_COLUMN_WIDTH``'s line).
#:
#: ALIGNED, AND NOTHING IS CUT. The header added for them was a two-space list of
#: labels beside two-space rows, so each label sat wherever its own length left it:
#: at 64 columns — where the header is what a user reads as the columns — nothing
#: lined up with the values under it. Fixed widths were the other way to fix that,
#: and they are wrong for THREE of these four columns, because the text in them is
#: identity this process did not author: a peer's session id, a peer's device name, a
#: conversation title minted elsewhere. The local table can cut what it prints because
#: its cells are a listing of known shapes; here a cut id makes two rows
#: indistinguishable and a cut name answers the wrong question — the suite's own
#: fixture is a 19-character id, `_UNSEEN_ROW`'s. So each column is as wide as the
#: widest cell it actually prints, and the header is one more row of that grid.
#:
#: Named in the order the rows print them; ``CONVERSATION`` is left unmeasured because
#: it is last and nothing follows it to be displaced.
SESSIONS_COLUMNS = ("SESSION", "DEVICE", "STATE", "CONVERSATION")


def _session_plane_lines(rows: Sequence[tuple[str, str, str, str]]) -> list[str]:
    """The federated listing's header and rows, laid out on one grid.

    CELLS, NOT CHARACTERS, because two of these columns carry text this process did
    not author (a peer's device name can be CJK) and a column is a display width:
    ``cli._pad_cell`` is the local table's own padder and rich's ``cell_len`` its own
    measure, so the two listings cannot disagree about what a cell is (design round 1,
    D2 — the finding that made ``cli._pad_cell`` exist).

    A column is never narrower than the local table's width for the same column
    (``cli.PEER_COLUMN_WIDTH``, ``cli.STATE_COLUMN_WIDTH``), so a short listing renders
    in the shape the sibling verb uses and one session reads the same in both. Those
    two names are imported here rather than at module scope for the reason every other
    import of that module in this file is: its docstring's import-cost contract.
    """
    from rich.cells import cell_len

    from local_operator.cli import PEER_COLUMN_WIDTH, STATE_COLUMN_WIDTH, _pad_cell

    floors = (0, PEER_COLUMN_WIDTH, STATE_COLUMN_WIDTH)
    measured = SESSIONS_COLUMNS[:-1]
    widths = [
        max(len(label), floors[index], *(cell_len(cells[index]) for cells in rows))
        for index, label in enumerate(measured)
    ]

    def laid_out(cells: Sequence[str]) -> str:
        padded = [_pad_cell(text, width) for text, width in zip(cells, widths)]
        # Right-trimmed: a row with no conversation ends in the blanks of the columns
        # before it otherwise, and trailing whitespace is a diff nobody can see.
        return "  ".join([*padded, cells[-1]]).rstrip()

    # The header is the labels through the SAME lay-out: it is one more row of the
    # grid, so a column cannot move in the header and stay in the rows.
    return [laid_out(SESSIONS_COLUMNS), *(laid_out(cells) for cells in rows)]


#: What a stop outcome leaves the user with, when it leaves them with anything
#: (UX round 3, U21).
#:
#: WHY A REMEDY AND NOT THE TOKEN. This receipt printed the relay's own sentence
#: and then ``rung: none  outcome: not_running`` under it — the stop ladder's rung
#: name and the wire word for the outcome, on the surface a user reads after asking
#: for something to stop. For an outcome that ENDED the session the sentence above
#: already says what happened, so the second line was pure noise; for
#: ``not_running`` it was the worse half of a dead end, because the session stays
#: in the listing and nothing on screen said so.
#:
#: The keys are the peer's own vocabulary (`relay._STOP_OUTCOME_WORD`, which
#: renders `control.StopOutcome.method`), and an outcome with no entry prints no
#: second line rather than a guess: the peer's sentence is the answer, and a
#: remedy invented here for a state this side cannot see would be the kind of
#: claim `_STOP_ENDED_OUTCOMES` itself refuses to make about ``ok``.
_STOP_REMEDIES: dict[str, str] = {
    # The session is in the listing and the user has just been told it is not
    # running: say why it is still there, or the frame reads as a stale row.
    "not_running": "It stays in your sessions until it is removed on that device.",
    # ``skipped`` is the ladder declining to signal a target with a turn in
    # flight — a deliberate leave-alone, and the one outcome whose remedy is a
    # flag on this very command.
    "skipped": "A turn is in flight there; --force stops it anyway.",
}


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

    EVERY VERB HERE ASKS AND WAITS: none of them has a local spelling of its own
    op (the session lives on another device and only this device's relay can ask
    it), so none of them may take ``allow_no_answer`` and none of them may read a
    missing answer as a result (Q-R4-1). ``ok`` is a claim about a NAMED field the
    peer reported — never about the absence of one.
    """
    from local_operator.network.types import MeshRefusal

    peer = str(getattr(args, "peer", "") or "")
    session_id = str(getattr(args, "stop", "") or "")
    engage = str(getattr(args, "engage", "") or "")

    if bool(getattr(args, "force", False)) and not session_id:
        # A USAGE error, not a refusal (the guide's rc 2): ``--force`` means one
        # thing only — the owner's ladder's way PAST a turn in flight — and a flag
        # that is accepted and then quietly dropped is the same class of untruth as
        # a sentence naming a flag that does not exist (Q-R5-2).
        print("--force applies to --stop only", file=sys.stderr)
        return 2

    # ARCHIVE, RESTORE AND DELETE: routed to the OWNER, never replicated (§8.1).
    # The guards, the confirmation semantics and the retention interaction stay in
    # one place, on the device that owns the disk — so this side's whole job is to
    # carry the request and render the owner's words verbatim.
    for flag, action in (
        ("archive", "archive"),
        ("unarchive", "unarchive"),
        ("delete", "delete"),
    ):
        target = str(getattr(args, flag, "") or "")
        if not target:
            continue
        if not peer:
            raise MeshRefusal(
                "peer_required",
                f"--{flag} needs --peer: the conversation lives on one device, and only "
                "its owner may change it",
            )
        from local_operator.network import mobility

        detail = mobility.lifecycle(
            target,
            action=action,  # type: ignore[arg-type]
            peer=peer,
            # A DELETE WITHOUT A CONFIRMATION IS THE OWNER'S DRY RUN: it runs every
            # guard and removes nothing, which is what makes the destructive form a
            # deliberate second invocation rather than the default.
            confirmed=bool(getattr(args, "yes", False)),
        )
        lines = [str(detail.get("message") or "")]
        if flag == "delete" and detail.get("ok") and not getattr(args, "yes", False):
            lines.append(f"Nothing was deleted. Run it again with --yes to delete it on {peer}.")
        return _emit(args, {"ok": bool(detail.get("ok")), **detail}, lines)

    if session_id:
        if not peer:
            raise MeshRefusal("peer_required", "--stop needs --peer: a session lives on one device")
        # A LONG budget on purpose: the peer runs its OWN ladder, whose SIGTERM
        # rung waits out a drain only the owning machine can bound, and a CLI
        # that gave up at 5 s would report "nothing happened" about a stop that
        # was working.
        #
        # ``--force`` IS THE PEER'S OWN ``lop stop --force``, BY MODE. The ladder
        # declines to signal a target that reports a turn in flight and names
        # ``--force`` as the way past it; the flag is real on the owner's verb, so
        # the VIEWER must accept it too — a surface that offers an action it cannot
        # accept is the defect UX round 2 called U7, and the owner may be a machine
        # the operator cannot sit down at. ``mode`` is the ladder's own wire
        # spelling (relay._op_session_stop maps ``immediate`` to
        # ``control.stop_session(force=True)``, whose meaning is documented at
        # its busy rung: escalate past it, and admit the record-field identity
        # proof when the socket cannot answer).
        detail = _relay_answer(
            "peer_session_stop",
            peer=peer,
            session_id=session_id,
            mode="immediate" if bool(getattr(args, "force", False)) else "graceful",
            timeout=240.0,
        )
        outcome = str(_reported(detail, "outcome", verb="stop") or "")
        # THE REMEDY, IN WORDS (UX round 3, U21) — and never a rung name. See
        # `_STOP_REMEDIES` for why an outcome with no remedy prints no second line.
        lines = [str(detail.get("detail") or "")]
        remedy = _STOP_REMEDIES.get(outcome)
        if remedy:
            lines.append(remedy)
        return _emit(args, {"ok": outcome in _STOP_ENDED_OUTCOMES, **detail}, lines)

    if engage:
        if not peer:
            raise MeshRefusal(
                "peer_required", "--engage needs --peer: a session lives on one device"
            )
        detail = _relay_answer(
            "peer_session_engage",
            peer=peer,
            session_id=engage,
            cwd=str(getattr(args, "cwd", "") or ""),
            timeout=120.0,  # a spawn, bounded by the relay's own engage deadline
        )
        engaged = bool(_reported(detail, "engaged", verb="engage"))
        # ONE FACT, SAID ONCE (UX round 3, U21), the same rule the create receipt
        # above now follows: the relay's own sentence is the line, and the boolean
        # was a second copy of it — ``engaged: True`` printed beside
        # ``runtime joining``, a flag and its own English translation. The
        # machine-readable ``engaged`` key stays in the ``--json`` payload.
        return _emit(
            args,
            {"ok": engaged, **detail},
            [str(detail.get("detail") or "") or ("engaged" if engaged else "not engaged")],
        )

    if getattr(args, "create", False):
        if not peer:
            raise MeshRefusal(
                "peer_required", "--create needs --peer: the peer mints the session id"
            )
        # ``--yolo`` IS REFUSED HERE, BEFORE THE RELAY IS ASKED, and the sentence is
        # the relay's own (literally — the two guards are spelled once each and kept
        # in step by a test). A flag this verb inherits from the global set must not
        # be silently dropped: ``lop network sessions --create --yolo`` looks
        # exactly like ``lop exec --yolo`` at the call site, and accepting it while
        # the far end ran gated would have been the worse answer of the two.
        if getattr(args, "yolo", False):
            raise MeshRefusal(
                "not_permitted",
                "a session created on another device cannot start unattended (yolo): that "
                "would make that machine run tools with nobody there to see them. Create "
                "it here, or start it on your own device with yolo.",
            )
        profile = str(getattr(args, "profile", "") or "")
        agent_name = str(getattr(args, "agent_name", "") or "")
        agent_id = str(getattr(args, "create_agent_id", "") or "")
        team = str(getattr(args, "team", "") or "")
        effort = str(getattr(args, "effort", "") or "")
        if agent_name and agent_id:
            # ``--agent`` and ``--agent-id`` select the same slot (the exec verb's
            # own rule), so naming both is a usage error rather than a silent
            # preference for one.
            print("--agent/--agent-id select the same agent; name one", file=sys.stderr)
            return 2
        if profile and (agent_name or agent_id):
            # Deliberately NOT a usage error: they are different things (an
            # attachable persona and a legacy agent row) and the frame carries
            # both, so a caller that names both gets both — the profile's
            # instructions and the agent's routing. Nothing here silently drops
            # one of them, which is what the old behaviour did to both.
            pass
        detail = _relay_answer(
            "peer_session_create",
            peer=peer,
            cwd=str(getattr(args, "cwd", "") or ""),
            name=str(getattr(args, "name", "") or ""),
            prompt=str(getattr(args, "prompt", "") or ""),
            model=(
                {
                    "provider": str(getattr(args, "hosting", "") or ""),
                    "model_id": str(getattr(args, "model", "") or ""),
                }
                if (getattr(args, "hosting", None) or getattr(args, "model", None))
                else None
            ),
            profile=profile,
            agent_name=agent_name,
            agent_id=agent_id,
            team=team,
            effort=effort,
            timeout=120.0,  # a spawn plus its first turn's admission
        )
        minted = str(_reported(detail, "session_id", verb="create") or "")
        prompt = str(getattr(args, "prompt", "") or "")
        # THE RECEIPT NAMES THE DEVICE, AND DOES NOT CALL AN ABSENT PROMPT A
        # FAILURE (UX round 1, U2/U3). ``admitted`` is the relay's word for "the
        # first prompt was admitted" — ``_op_session_create`` sets it only
        # inside ``if prompt:`` — so the promptless create, which is the first
        # thing a user types, printed ``admitted: False`` beside ``ok: True``
        # and read as a failure for a session the peer had just minted, spawned
        # and taken a runtime for. The DEVICE is named because the session is
        # not on this machine and is not opened: without it the one fact a user
        # needs next (which peer to ask) is only in the command they typed a
        # moment ago and have already scrolled past. The machine-readable
        # ``admitted`` key is unchanged — a ``--json`` consumer branches on it.
        lines = [f"session: {minted or '-'}", f"created on {peer}"]
        # WHO IT RUNS AS, SAID IN THE RECEIPT. The identity half of this verb is new,
        # and a receipt that named the device but not the agent would leave the one
        # fact the user just chose unreported — they cannot re-read it from the
        # session, because the session is on another machine. Both halves are printed
        # from the PEER's own reply (never from this device's registries, which is a
        # claim about a store that does not hold this session), and the honest half is
        # printed too: an agent whose instructions are not attachable ran its own
        # instructions on that agent's model, and the user is told which of the two
        # happened.
        agent_info = detail.get("agent") if isinstance(detail.get("agent"), dict) else None
        team_info = detail.get("team") if isinstance(detail.get("team"), dict) else None
        if agent_info:
            applied = bool(agent_info.get("instructions_applied"))
            lines.append(
                f"agent: {agent_info.get('name')}"
                + ("" if applied else " (routing only — its instructions are not attachable)")
            )
        if team_info:
            lines.append(f"team: {team_info.get('name')}")
        if profile or agent_name or agent_id:
            model_detail = detail.get("model")
            if isinstance(model_detail, dict) and model_detail.get("detail"):
                lines.append(str(model_detail["detail"]))
        # WHAT THIS CREATE DID ABOUT DEFINITIONS (QA round 1, Q1). The receipt reported a
        # session that ran the PEER's divergent copy as a plain success — no key in
        # ``--json``, no word about a push that had been refused — which is the "it
        # worked" report this slice exists to make impossible. The reconciliation block
        # is the relay's (it is the half that saw the push) and is printed whenever it is
        # not clean; ``--json`` carries the whole block either way (``**detail``).
        notes = detail.get("definitions")
        if isinstance(notes, dict) and not notes.get("ok"):
            lines.append(
                f"definitions: not reconciled onto {peer} — "
                f"{notes.get('message') or notes.get('code') or 'the push did not complete'}"
            )
        if isinstance(notes, dict) and notes.get("unpinned"):
            names = ", ".join(repr(str(name)) for name in notes["unpinned"])
            lines.append(
                f"no copy of {names} on this device: that device's own revision of it is "
                "what runs there"
            )
        if prompt:
            # ONE FACT, SAID ONCE (UX round 3, U21). This printed the boolean
            # ``first prompt admitted: True`` AND the relay's own sentence for the
            # same fact (``prompt admitted``) — a flag and its own English
            # translation, on two lines, about one event. The sentence is the
            # receipt's register and the boolean was the second copy; the
            # machine-readable ``admitted`` key is untouched in the ``--json``
            # payload (`**detail` below), which is where a consumer reads it.
            admitted = bool(detail.get("admitted"))
            lines.append(
                str(detail.get("detail") or "")
                or ("prompt admitted" if admitted else "the prompt was not admitted")
            )
        else:
            lines.append(
                "no prompt sent — the session exists on that device; "
                f"/network sessions --peer {peer} lists it, --engage warms it, "
                "--stop ends it"
            )
        return _emit(args, {"ok": bool(minted), **detail}, lines)

    if not peer and not getattr(args, "all_peers", False):
        raise MeshRefusal(
            "peer_required",
            "name a device with --peer, or ask every device with --all-peers",
        )
    # The merge is a LISTING, so it is the one verb here whose success does not
    # rest on a single reported word: what it must not do is dress an unreachable
    # relay up as "no sessions" (Q-R4-1), which is why this call refuses instead
    # of falling through to an empty set. When the relay IS up, the peer block per
    # row is what labels an empty answer (``reachable`` + ``reason``).
    payload = _relay_answer("peer_session_rows", timeout=_listing_timeout())
    # Bound to a narrow local first: `payload.get` is `Any | None`, and the two
    # comprehensions below iterate it as a mapping (`projection.RelayPeerCatalog`
    # takes the same precaution for the same reason).
    raw_facts = payload.get("peers")
    facts: dict[str, Any] = raw_facts if isinstance(raw_facts, dict) else {}
    remote = [row for row in (payload.get("sessions") or []) if isinstance(row, dict)]
    matched: list[dict[str, Any]] = []
    if peer:
        wanted = peer.lower()
        # THE DEVICES THE LISTING NAMED, FROM THE ANSWER'S OWN PEER BLOCK rather
        # than from a second lookup: the fan-out already resolved every member
        # this device holds, so an empty peer set for this token is the SAME
        # fact the write half refuses on (``--create --peer ghost``).
        matched = [
            block
            for device_id, block in facts.items()
            if isinstance(block, dict)
            and (str(device_id).lower() == wanted or str(block.get("name") or "").lower() == wanted)
        ]
        if not matched:
            # AN UNKNOWN TOKEN IS REFUSED, NOT ANSWERED (QA round 11, Q-R11-3).
            # This used to fall through to the empty-listing sentence — a claim
            # about EVERY device, made false by the peer that was holding
            # sessions at that moment — while the write half of the same family
            # refused the identical token correctly. One vocabulary, so a typo
            # reads the same whichever half of the family it reaches.
            raise MeshRefusal(
                "unknown_peer",
                f"this device is not in a network with anything called {peer!r}",
            )
        remote = [
            row
            for row in remote
            if str((row.get("peer") or {}).get("device_id") or "").lower() == wanted
            or str((row.get("peer") or {}).get("name") or "").lower() == wanted
        ]
    lines: list[str] = []
    cells: list[tuple[str, str, str, str]] = []
    from local_operator.resume import session_state_words

    for row in remote:
        block = row.get("peer") or {}
        # A ROW THIS DEVICE HOLDS IS NAMED AS SUCH, not as ``?`` (UX round 2,
        # U13). The federated listing includes this machine's own rows
        # (``locality``/``peer: None``), and the device column rendered the
        # absent peer block as a question mark — so the one column that answers
        # "which device holds what" answered "unknown" about the device the user
        # is sitting at.
        holder = str(block.get("name") or block.get("device_id") or "").strip()
        if not holder:
            holder = str(payload.get("device_name") or "this device")
        # THE STATE IN WORDS (UX round 5, U29). This printed the catalogue's raw
        # token — ``stored`` — in a headerless row of CLI columns, so a reader had
        # to infer three of the four columns from shape alone and read a token no
        # other surface uses (the sidebar paints that session under ``⇄`` with a
        # row mark, and the words the app has for the condition are in
        # ``resume.session_state_words``).
        cells.append(
            (
                str(row.get("session_id") or ""),
                holder,
                session_state_words(str(row.get("state") or "")) or "?",
                str(row.get("conversation_name") or ""),
            )
        )
    if remote:
        # Only when there are rows: with none, the sentence below IS the listing
        # and a header above it would name columns nothing is printed under. Header
        # and rows are laid out TOGETHER, because a column's width is the width of
        # the cells actually printed in it (`_session_plane_lines`).
        lines.extend(_session_plane_lines(cells))
    # A DEVICE THAT COULD NOT BE ASKED IS NAMED, IN EVERY CASE (Q-R4-1, UX round
    # 2 U14). The sibling family already does this (``lop sessions --all-peers``
    # prints ``<device>: unreachable (<reason>)``), and this one used to answer
    # with a complete-looking listing that silently omitted the device: with a
    # peer's relay down, ``--all-peers`` said "no sessions are held by other
    # devices right now" — a claim about EVERY device, established about none.
    unasked = [
        (device_id, block)
        for device_id, block in sorted(facts.items())
        if isinstance(block, dict) and not block.get("reachable")
    ]
    for device_id, block in unasked:
        # THE WORDS, NOT THE TOKEN (UX round 3, U23). This line printed the
        # relay's raw reason — ``connect_failed:ConnectionRefusedError`` — on a
        # surface a user reads, which is a Python class name in place of a fact.
        # The gloss is the shared one (`resume.peer_reason_words`, design round 1
        # D3) so this listing, `/sessions`'s own `--all-peers` line and the
        # sidebar tooltip say the same thing about the same state. The raw reason
        # is not lost: it is this verb's ``--json`` field, which is the machine
        # surface. `lop network doctor` renders its own rows through
        # `resume.doctor_detail_words` for the same reason and in the same two
        # registers (round 24, Q-R24-2): that table is a SECOND one because a
        # doctor row is about one ADDRESS and these sentences are about a MEMBER,
        # which is argued in that function's docstring.
        from local_operator.resume import peer_reason_words

        lines.append(
            f"{str(block.get('name') or device_id)}: unreachable "
            f"({peer_reason_words(str(block.get('reason') or ''))})"
        )
    if not remote:
        # The sentence is narrowed to what was actually established. Reaching
        # here with every peer asked means the global one is true; with a peer
        # unasked it is not, and the failure line above already says which
        # device is missing from the answer.
        if peer:
            block = matched[0]
            if block.get("reachable"):
                lines.insert(0, f"no sessions are held by {block.get('name') or peer} right now")
        elif unasked:
            lines.insert(0, "no sessions are held by the devices that answered")
        else:
            lines.insert(0, "no sessions are held by other devices right now")
    return _emit(
        args,
        {"ok": True, "sessions": remote, "peers": payload.get("peers") or {}},
        lines,
    )


def _status_membership_lines(row: dict[str, Any]) -> list[str]:
    """`status` prints this device's own standing; the body is ``relay``'s."""
    from local_operator.network.relay import membership_lines

    return membership_lines(row)


def _cmd_status(args: argparse.Namespace) -> int:
    relay_mod = _import_relay()
    payload = relay_mod.status()
    # THE HUMAN LINE IS DERIVED FROM THE SAME FIELDS AS THE PAYLOAD, so the two
    # cannot disagree: `relay_running` and `relay_answering` are the facts, and a
    # relay that is up but silent says so here instead of reading as "not running"
    # beside its own pid (Q-R3-4).
    relay_line = "not running"
    if payload.get("relay_running"):
        pid = (payload.get("relay") or {}).get("pid") or (payload.get("record") or {}).get("pid")
        if payload.get("relay_answering"):
            relay_line = f"running, pid {pid}"
        else:
            relay_line = (
                f"running (pid {pid}), NOT answering its control socket "
                f"(state: {payload.get('relay_state')})"
            )
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
        # A network whose own standing is not `active` says so here too: `status` is
        # the first command an agent runs (the guide's step 1), so it is where a
        # removed device must find the sentence rather than the member list it
        # holds (Q-R3-2). The member-count marker is NOT printed here — this command
        # does not refresh the table, and the row's own `membership.table` says so.
        lines.extend(_status_membership_lines(network))
    return _emit(args, {"ok": True, **payload}, lines)


def _peer_line(row: Any) -> str:
    """One ``/network peers`` row, in the words a person reads.

    A NAME, A STATE WORD AND A REASON IN WORDS — never the wire token, and never
    the 34-character device id (UX round 5, U28). The row printed
    ``unreachable d_1a2b3c…  pixel-8  connect_failed:ConnectionRefusedError``: a
    stage token plus a Python class name where the sibling create arm says
    "cannot be reached from this device right now", and a peer addressed by the
    id every other surface replaces with its NAME (``mesh-ui.md`` §1.2 gives the
    id a column of its own, eight characters of it; the sidebar's heading is
    ``⇄ <label>``; a peer with no name is ``resume.UNNAMED_DEVICE`` on all of
    them, design round 1 D8). The gloss is the SHARED one
    (``resume.peer_reason_words``, design round 1 D3) — the same function the
    ``--all-peers`` listing and the sidebar's tooltip read, so this line cannot
    drift into a second vocabulary for ``connect_failed:ConnectionRefusedError``.

    The token is not lost: it is the ``reason`` field of this verb's ``--json``
    payload, which is the machine surface and the detail view these lines
    summarise (the row also keeps ``device_id`` there, for the same reason).
    """
    from local_operator.resume import UNNAMED_DEVICE, peer_reason_words

    name = str(row.get("name") or "").strip() or UNNAMED_DEVICE
    if row.get("reachable"):
        return f"{name}  reachable"
    return (
        f"{name} cannot be reached from this device right now "
        f"({peer_reason_words(str(row.get('reason') or ''))})"
    )


def _cmd_peers(args: argparse.Namespace) -> int:
    # A LISTING PROBES ITS PEERS, so it needs the relay's own listing budget plus
    # slack rather than the 5s default: with the default, one unreachable member
    # would make the control call time out and this verb would report "the relay is
    # not running" about a relay that is running perfectly well.
    # ``allow_no_answer`` because THIS verb is the family's own emitter: it ships the
    # refusal itself, below, and keeping that branch here (rather than letting
    # ``_relay_call`` raise) is what keeps its documented payload — ``code`` +
    # ``message`` + an empty ``peers`` — stable for the direct-call tests.
    live = _relay_call("net_peer_ls", timeout=_listing_timeout(), allow_no_answer=True)
    if live is None:
        # ONE REFUSAL SHAPE FOR THE WHOLE FAMILY: ``code`` + ``message``, the same
        # keys every other `lop network` refusal uses and the only shape an agent
        # path has to parse. This one used to answer with a bare ``error`` key, so
        # a consumer that read ``code`` saw nothing at all (QA round 1, F-6). The
        # sentence distinguishes a stopped relay from a wedged one (Q-R3-4).
        # The CODE is the shared constant, so the whole family — this branch and
        # every ``_relay_call`` refusal — branches on one string.
        message = _relay_unavailable_message()
        return _emit(
            args,
            {
                "ok": False,
                "code": CODE_RELAY_UNAVAILABLE,
                "message": message,
                "peers": [],
            },
            [message],
        )
    # ``net_peer_ls`` answers a LIST (a peer table), so the control client wraps it
    # as ``{"value": [...]}`` rather than pretending it is a mapping.
    raw_rows = live.get("value")
    if not isinstance(raw_rows, list):
        raw_rows = live.get("peers")
    rows: list[Any] = raw_rows if isinstance(raw_rows, list) else []
    return _emit(
        args,
        {"ok": True, "peers": rows},
        [_peer_line(row) for row in rows]
        or ["no peers: this device is the only member of its networks"],
    )


def _cmd_disconnect(args: argparse.Namespace) -> int:
    record = _resolve(args.network)
    live = _relay_call("net_disconnect", network=record.network_id, allow_no_answer=True)
    if live is None:
        from local_operator.network import store
        from local_operator.network.relay import set_trust

        set_trust(record, trust="disconnected", reason="this device disconnected")
        secrets_file = store.secrets_path(record.network_id)
        if secrets_file.exists():
            secrets_file.unlink()
        # NO RELAY ANSWERED, so nobody was told anything — and the receipt says so
        # rather than "peers notified (0)", which reads as "there were none"
        # (QA round 1, Q-R1-2: the receipt must describe what the peers did).
        live = {
            "network_id": record.network_id,
            "reachable_peers": 0,
            "secret_deleted": True,
            "sent": 0,
            "acked": 0,
            "unacked": [],
            "refused": [],
            "failed": [],
            "peers": [],
            "ok": False,
            "relay": "did not answer",
        }
    return _emit(
        args,
        {**live},
        _disconnect_lines(record.name, live),
    )


def _peer_report_lines(live: dict[str, Any], *, verb: str) -> list[str]:
    """One line per peer that did not simply take it, naming the peer and its reason.

    A COUNT IS NOT AN ANSWER to "did the incident land" (QA round 1, Q-R1-2): the
    operator's next move is per device — re-admit this one, chase that one — so the
    device's name and the reason it gave are printed, and the per-peer rows stay in
    the ``--json`` payload for anything that parses.
    """
    rows = live.get("peers")
    if not isinstance(rows, list):
        return []
    lines: list[str] = []
    for row in rows:
        if not isinstance(row, dict) or row.get("outcome") == "acked":
            continue
        outcome = str(row.get("outcome") or "?")
        reason = str(row.get("reason") or "")
        lines.append(
            f"  {row.get('name') or row.get('device_id')}: {verb} it was not taken "
            f"({outcome}{': ' + reason if reason else ''})"
        )
    return lines


def _disconnect_lines(name: str, live: dict[str, Any]) -> list[str]:
    sent = int(live.get("sent") or 0)
    acked = int(live.get("acked") or 0)
    lines = [f"left {name}: local secret deleted, links closed, audit trail kept"]
    if sent:
        lines.append(f"peers told: {acked} of {sent} took the leave")
    else:
        lines.append("no peer was told: this device held no link to anybody")
    lines += _peer_report_lines(live, verb="the leave")
    return lines


def _cmd_panic(args: argparse.Namespace) -> int:
    record = _resolve(args.network)
    live = _relay_call("net_panic_local", network=record.network_id, allow_no_answer=True)
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
        # NO RELAY ANSWERED: the latch is set here, and no peer was asked at all.
        # ``ok: False`` is the honest answer for that (the old receipt said
        # ``ok: true, broadcast_to: 0`` and printed "every other device is told to
        # stop trusting the network" — Q-R1-2's class of lie, in the degraded path).
        live = {
            "network_id": record.network_id,
            "epoch": record.epoch,
            "rotated": is_admin,
            "sent": 0,
            "acked": 0,
            "unacked": [],
            "refused": [],
            "failed": [],
            "peers": [],
            "ok": False,
            "relay": "did not answer",
        }
    lines = [
        f"panic raised on {record.name}: this device is untrusted at epoch {live.get('epoch')} "
        "and refuses peer traffic until it is re-admitted",
    ]
    sent = int(live.get("sent") or 0)
    if sent:
        lines.append(f"peers told: {live.get('acked', 0)} of {sent} acted on it")
    else:
        lines.append("no peer was told: no relay answered, so nothing was broadcast")
    lines += _peer_report_lines(live, verb="the panic")
    # THE PROHIBITION, IN THE COPY AND NOT ONLY IN THE DESIGN (§1.5/§1.6.3): "stop
    # the network" reads as "stop what is running", and nothing here stops anything.
    lines.append(
        "sessions on other devices are NOT stopped — they keep running and are "
        "unreachable from here"
    )
    lines.append("re-admit each device with `lop network trust <network> --active`")
    return _emit(args, {**live}, lines)


def _cmd_trust(args: argparse.Namespace) -> int:
    target = "active" if args.active else "untrusted" if args.untrusted else "active"
    record = _resolve(args.network)
    live = _relay_call(
        "net_trust_local", network=record.network_id, trust=target, allow_no_answer=True
    )
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
            + (" (the relay is not running: applied locally)" if applied_locally else ""),
            *_secret_caveat(record),
        ],
    )


def _secret_caveat(record: Any) -> list[str]:
    """What TRUST does not answer: whether this device can still USE the network.

    A TRUST RECEIPT THAT SAYS ``active`` FOR A NETWORK WITH NO SECRET (QA round
    10, Q-R10-5). ``disconnect`` deletes this device's secret and keeps the
    record, so the very next verb in the family refuses — correctly, in the
    words ``store.require_secrets`` owns — while the receipt for the act the
    user just performed read as ``usable``. The state a person acts on is the
    pair of facts, so both are stated, in the refusal's own vocabulary rather
    than a second one for the same condition.
    """
    from local_operator.network.store import secrets_path

    if secrets_path(record.network_id).exists():
        return []
    return [
        "⚠ this device has no secret for it (deleted by `/network disconnect`), so it "
        "is trusted but not usable: rejoin with `lop network join` and a fresh invite"
    ]


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
            # `code` + `message` like the rest of the family (GUIDE.md §"every
            # refusal names the component"): this one carried a bare `error`, so the
            # documented reader saw no code and no sentence to act on.
            message = "no audit log yet"
            return _emit(
                args,
                {"ok": False, "code": "no_audit_log", "message": message, "error": message},
                [message],
            )
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
        allow_no_answer=True,
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

    live = _relay_call("net_pair_pending", allow_no_answer=True)
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
    """Diagnose the mesh. Reports; never asserts reachability it has not proven.

    ``ok`` ANSWERS "IS THE MESH HEALTHY", derived from the check rows rather than
    asserted. It used to be hardcoded ``true`` while a row in the same payload
    said ``ok: false``, so a consumer that read the summary line and not the array
    read a failing mesh as a passing one — and ``ok`` is exactly the key an agent
    path checks first (QA round 1). The failing rows are also named in ``code`` /
    ``message``, so the refusal family keeps one shape.

    The relay is given the LISTING budget, not the 5 s default, because this op
    DIALS every endpoint it reports on: with the default it timed out on its own
    work, the fallback below answered instead, and the operator read a relay
    state from the one path that could not check it (QA round 2, Q-R2-6).
    """
    live = _relay_call(
        "net_doctor", peer=args.peer, timeout=_listing_timeout(), allow_no_answer=True
    )
    payload = live if live is not None else _doctor_locally(args)
    checks = list(payload.get("checks") or [])
    # THE ROW'S OWN WORDS, NOT ITS FIELD (round 24, Q-R24-2). This used to render
    # ``detail`` verbatim, which put a stage word, a Python class name AND — on the
    # state the listing's own round-10 fix exists for — the endpoint address on a
    # human line. These lines are the human register; the raw strings are unchanged
    # and still ride in ``checks[].detail`` of the payload ``--json`` prints.
    from local_operator.resume import doctor_detail_words

    lines = []
    for check in checks:
        state = "ok " if check.get("ok") else "FAIL"
        lines.append(
            f"{state} {check.get('check', '')} {check.get('device_id', '')} "
            f"{check.get('endpoint', '')} {doctor_detail_words(str(check.get('detail', '')))}"
            + (f" {check['latency_ms']}ms" if check.get("latency_ms") is not None else "")
        )
    if not lines:
        lines.append("nothing to check: no networks, or no other members yet")
    if not payload.get("identity_present", True):
        lines.append(
            "this device has no device identity (identity_missing): run `lop network init`, or "
            "re-pair with a new invite"
        )
    failures = [
        f"{check.get('check', '')}: {doctor_detail_words(str(check.get('detail', ''))) or 'failed'}"
        for check in checks
        if not check.get("ok")
    ]
    healthy = not failures and bool(payload.get("identity_present", True))
    answer: dict[str, Any] = {**payload, "ok": healthy}
    if not healthy:
        answer["code"] = "unhealthy"
        answer["message"] = "; ".join(failures) or "the device has no identity"
    return _emit(args, answer, lines)


def _unprobed_detail(has_endpoint: bool, relay_up: bool) -> str:
    """Why a row in the LOCAL doctor fallback carries no reachability answer.

    Three cases, and the sentence names which one it is: a member that declared
    nothing, a relay that is up but did not answer the dialling op (a wedged relay
    and a stopped one are different incidents), and no relay at all.
    """
    if not has_endpoint:
        return "no_endpoint"
    if relay_up:
        return (
            "not probed: the relay is running but did not answer the doctor op, so no "
            "address was dialled"
        )
    return "not probed: no relay is running on this device, so nothing here can dial"


def _relay_state() -> tuple[str, bool]:
    """What THIS MACHINE says the relay is doing: ``(the line, is it up)``.

    A diagnostic that asserts a state it never checked is the dead-instrument
    failure in its purest form, and it is what shipped: the fallback returned the
    constant ``"not running"``, so `doctor --json` reported an absent relay beside
    a `status --json` in the same capture reporting it running with a pid (QA
    round 2, Q-R2-6).

    Three answers, in the order they can be established: a relay that ANSWERS over
    its own control socket is running; a relay RECORD whose socket did not answer is
    running but not answering — worth telling apart, because "it crashed" and "it is
    wedged" want different remedies — and no record is not running.

    THE RECORD IS READ BY :func:`store.scan_own_relay`, which reports ``wedged`` as
    well as ``live``. That is the difference between this branch being REACHED and
    only being POSSIBLE: a SIGSTOPped relay stops writing heartbeats, so within the
    timeout its record classifies as ``wedged`` and a live-only scan returns nothing
    — the branch added for the wedge sat behind a read that could never see one, and
    `doctor` went on calling a stopped process "not running" (QA round 3, Q-R3-4).
    """
    from local_operator.network import relay as relay_mod
    from local_operator.network import store

    live = relay_mod.health()
    if live is not None:
        return f"running, pid {live.get('pid')}", True
    record, state = store.scan_own_relay()
    if record is not None:
        detail = (
            "its control socket did not answer this probe, and its heartbeat has gone "
            "stale as well, so its owner is not reporting either"
            if state == "wedged"
            else "its control socket did not answer this probe"
        )
        return f"running (pid {record.pid}), and {detail}", True
    return "not running", False


def _relay_unavailable_message() -> str:
    """Why an op that needed the relay did not get an answer — named, not guessed.

    "THE RELAY IS NOT RUNNING" IS A CLAIM ABOUT A PROCESS, and this family of verbs
    printed it whenever the CONTROL SOCKET went unanswered — including on a machine
    whose own status payload carried the relay's live pid. The two incidents want
    different remedies (start it, versus kill -CONT or restart a wedged one), so the
    record is consulted rather than assumed (QA round 3, Q-R3-4).
    """
    from local_operator.network import store

    record, state = store.scan_own_relay()
    if record is None:
        return "the relay is not running; start it with `lop network start`"
    suffix = (
        " and its heartbeat has gone stale, so its owner is not reporting"
        if state == "wedged"
        else ""
    )
    return (
        f"this device's relay (pid {record.pid}) is running but did not answer its "
        f"control socket{suffix}; that is a wedged relay rather than a stopped one — "
        "stop and start it with `lop network restart`"
    )


def _doctor_locally(args: argparse.Namespace) -> dict[str, Any]:
    """The checks that need no relay: identity, records, epochs, endpoints.

    Deliberately NOT a claim about reachability: with no relay answering there is
    nothing here that can dial, and reporting "unreachable" from a process that
    never tried would be the dead-instrument failure the repo warns about. So an
    endpoint row is ``ok: false`` with the reason it was not dialled — a check
    that was never RUN did not PASS, and the old ``ok: true`` beside
    ``not probed (the relay is not running)`` is how this command came to certify
    a mesh it had not looked at (Q-R2-6).
    """
    from local_operator.network import store
    from local_operator.network.identity import identity_path
    from local_operator.network.relay import membership_state

    relay_line, relay_up = _relay_state()
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
        # THE SAME MEMBERSHIP ROW THE RELAY'S OWN DOCTOR EMITS, so a device whose
        # relay is down still learns what its own standing is — which is exactly the
        # device a removed member is, once nothing answers it (Q-R3-2).
        standing = membership_state(record)
        if standing["state"] != "active":
            checks.append(
                {
                    "check": "membership",
                    "network_id": record.network_id,
                    "ok": False,
                    "code": standing["state"],
                    "detail": standing["sentence"],
                    "remedies": standing["remedies"],
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
                    "ok": False,
                    "probed": False,
                    "detail": _unprobed_detail(bool(member.endpoints), relay_up),
                }
            )
    return {
        "checks": checks,
        "identity_present": identity_file.exists(),
        "identity_dir": str(identity_file.parent),
        "relay": relay_line,
    }


def _cmd_identity_show(args: argparse.Namespace) -> int:
    from local_operator.network.identity import load

    identity = load()
    if identity is None:
        return _emit(
            args,
            {
                "ok": False,
                "code": "identity_missing",
                "message": "this device has no mesh identity yet; run `lop network init`",
                "error": "identity_missing",
            },
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
    from local_operator.network.identity import load, rotate, rotation_statement

    previous = load()
    if previous is None:
        message = "no identity to rotate: this device has one only after `lop network init`"
        return _emit(
            args,
            {"ok": False, "code": "identity_missing", "message": message, "error": message},
            [message],
        )
    new, old = rotate()
    relay_mod = _import_relay()
    announced = {"sent": 0, "queued": 0}
    if relay_mod.health() is not None:
        server = relay_mod.RelayServer(identity=new)
        announced = relay_mod.announce_identity_rotation(server, old, new)
        server.stop()
    else:
        # NOTHING IS ANSWERING, so this process owns the record for the moment — and
        # it edits it the way every other writer here does: the statement is built
        # per network (it names the network it rotates within), the record is
        # re-read INSIDE the store's lock, and the row is re-checked against what is
        # on disk, so no save from another thread is reverted from this process's
        # older listing. The statement is KEPT ON THE ROW, because the table is the
        # only route to a peer this verb cannot reach: whoever pulls it later can
        # verify the hop instead of seeing an unfamiliar device.
        for listed in store.list_networks():
            if listed.self_device_id != old.device_id:
                continue
            statement = rotation_statement(old, new, listed.network_id)
            try:
                # A network forgotten between the listing and this rewrite is not a
                # reason to abandon the rest of them (the announce path's rule).
                with store.mutate(listed.network_id) as record:
                    if record.self_device_id != old.device_id:
                        continue
                    record.self_device_id = new.device_id
                    member = record.member(old.device_id)
                    if member is not None:
                        member.previous_ids = [*member.previous_ids, member.device_id]
                        member.device_id = new.device_id
                        member.public_key = new.public_key
                        member.rotation_proof = dict(statement)
                        member.rotated_at = time.time()
                    store.save(record)
            except FileNotFoundError:
                continue
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
            "a peer that never gets the statement — by frame or in this device's member row, "
            "which carries it — sees this device as unknown and must re-pair: nothing can "
            "prove continuity without the old key",
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


def _guard_credential_subcommand(args: argparse.Namespace) -> int:
    """``lop network credential`` with no verb is a usage error, not a default act.

    Neither verb is safe as a default: ``share`` widens who may spend the operator's
    account, and ``revoke`` silently cuts a working device off. So the verb is
    required and the message names both, which is the same rule ``member`` states.
    """
    verb = getattr(args, "credential_command", None)
    if verb in ("share", "revoke"):
        return _cmd_credential(args)
    print(
        "usage: lop network credential share  <provider|mcp:<url>> --with <device> "
        "[--scope session|device]\n"
        "       lop network credential revoke <provider|mcp:<url>> --from <device>\n"
        "       lop network credentials [--json]   # what is shared, and with whom",
        file=sys.stderr,
    )
    return 2


def _guard_member_subcommand(args: argparse.Namespace) -> int:
    """``lop network member`` with no verb is a usage error, not the destructive one.

    A group whose default action is ``rm`` would make a typo revoke a member, so the
    verb is required and the message says which one.
    """
    verb = getattr(args, "member_command", None)
    if verb in ("grant", "revoke"):
        return _cmd_member_caps(args)
    if verb != "rm":
        print(
            "usage: lop network member rm <network> <device>\n"
            "       lop network member grant|revoke <network> <device> <capability>...",
            file=sys.stderr,
        )
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
def _cmd_definitions(args: argparse.Namespace) -> int:
    """``lop network definitions push|state`` — agent and team definitions.

    WHY THIS VERB EXISTS WHEN THE CREATE ALREADY PUSHES. The create path reconciles
    only the names its own frame mentions, on the way to one session — which is the
    right scope for a create and the wrong scope for the two cases a person actually
    has in front of them: a peer that has just been cleaned out and should be brought
    back to a known state, and a freshly paired machine that should be able to resolve
    every name BEFORE the first create is attempted. Without this verb both of those
    are "name it in a create and hope the push succeeds", which is a diagnostic
    surface with no way to see the answer.

    ``state`` is deliberately local and relay-free: it reports what THIS device holds
    and what it has mirrored from elsewhere (the provenance index), so it answers
    "why does the peer resolve this name to the wrong thing" without a peer being up.
    """
    from local_operator.network import definitions
    from local_operator.network.types import MeshRefusal
    from local_operator.paths import config_dir

    verb = str(getattr(args, "definitions_command", None) or "state")
    root = config_dir()
    if verb == "push":
        peer = str(getattr(args, "peer", "") or "")
        # ``--all-peers`` and an empty ``--peer`` are ONE request ("every member"):
        # the relaying handler already treats an absent peer as "every paired
        # member", so passing a flag through would only create a second spelling of
        # the same instruction for the two ends to disagree about.
        if not peer and not bool(getattr(args, "all_peers", False)):
            raise MeshRefusal(
                "peer_required",
                "name a device with --peer, or ask every device with --all-peers",
            )
        detail = _relay_answer("definitions_sync", peer=peer, timeout=90.0)
        rows = [item for item in (detail.get("peers") or []) if isinstance(item, dict)]
        lines = [str(detail.get("message") or "")] if detail.get("message") else []
        for item in rows:
            label = str(item.get("device_id") or "?")
            # The per-peer sentence is the PUSH's own (it is the only party that saw
            # the state round trip), so it is printed rather than re-derived here.
            lines.append(f"{label}: {item.get('message') or item.get('code') or 'no answer'}")
            # THE ROWS ARE NAMED, not counted (QA round 1, Q2). "sent 1 agent
            # definition(s)" told the operator that something happened on a device whose
            # row they could not see; the two facts they act on are WHICH row landed and
            # that a re-install was a re-install.
            for verb, key in (("installed", "installed"), ("updated", "updated")):
                for row in item.get(key) or []:
                    if isinstance(row, dict):
                        lines.append(f"  {verb} {row.get('kind') or 'row'} {row.get('name')!r}")
            for conflict in item.get("conflicts") or []:
                if isinstance(conflict, dict):
                    lines.append(
                        f"  {conflict.get('kind') or 'row'} {conflict.get('name')!r}: "
                        f"{conflict.get('reason') or 'refused'}"
                    )
        return _emit(
            args, {"ok": bool(detail.get("ok")), "peers": rows}, lines or ["nothing to do"]
        )

    # ``state``: this device's own definitions, plus what was mirrored here.
    index = definitions.read_index(root)
    state = definitions.definition_state(root)
    mirrored = {
        kind: {
            str(name): str(row.get("origin") or "")
            for name, row in (index.get(kind) or {}).items()
            if isinstance(row, dict)
        }
        for kind in ("agents", "teams")
    }
    lines = []
    for kind in ("agents", "teams"):
        for name in sorted(state.get(kind) or {}):
            origin = mirrored[kind].get(name) or ""
            lines.append(
                f"{kind[:-1]}: {name}" + (f" (mirrored from {origin})" if origin else " (yours)")
            )
    if not lines:
        lines = ["no agent or team definitions on this device"]
    return _emit(
        args,
        {
            "ok": True,
            "agents": state.get("agents") or {},
            "teams": state.get("teams") or {},
            "mirrored": mirrored,
        },
        lines,
    )


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
    "definitions": _cmd_definitions,
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
    # ``credential`` has a sub-verb, so it needs the same "tell me what you meant"
    # guard ``member`` and ``identity`` have: running bare `lop network credential`
    # must print the two verbs rather than doing nothing.
    "credentials": _cmd_credentials,
    "credential": _guard_credential_subcommand,
}
