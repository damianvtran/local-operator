"""One human inspection/correction runner for CLI and the TUI slash command.

Explicit scope owners are a human control, never accepted by memory:// tools.
A correction carries the revision the human READ (optimistic, race-safe) and a forget
additionally requires --yes, so inspection can never destroy user data. Every row list
and search print is a paste-ready reference (``scope/id``) that show/edit/forget accept
as-is; printed hints are complete commands for the surface that printed them (``lop
memory`` or ``/memory``) so nobody has to reconstruct flags from prose.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import shlex
import sqlite3
import textwrap
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, NoReturn

from local_operator.memory import (
    MAX_BULK_FORGET,
    SCOPES,
    MemoryError,
    MemoryIdentity,
    MemoryStore,
    quota_relief,
)
from local_operator.memory_capture import MEMORY_DEFAULT_ENABLED, policy
from local_operator.memory_duplicates import duplicates
from local_operator.paths import config_dir

OPERATIONS = (
    "list",
    "show",
    "search",
    "move",
    "edit",
    "forget",
    "status",
    "inspect",
    "consolidate",
)
#: Fields a flag-based edit may change; everything else is carried over from the
#: revision being corrected, so a one-word fix never needs the whole record retyped.
_EDITABLE = (
    "title",
    "body",
    "tags",
    "aliases",
    "source",
    "status",
    "verified_at",
    "expires_at",
    "pinned",
    "subject",
    "created_at",
    "source_time",
    "replaces",
)
USAGE = (
    "Commands: list · show ID · search WORDS · move ID [ID…] --to SCOPE · status · "
    "inspect (read-only preview) · consolidate (read-only duplicate preview) · "
    "edit ID --revision REV (--title T | --body B | --tag T | --alias A | --verified) · "
    "forget ID --revision REV --yes · forget --scope S [--older-than D] [--unverified] "
    "[--tag T] [--title-matches RE] [--body-matches RE] (dry run; confirm with "
    "--yes --expect FINGERPRINT). ID is the SCOPE/ID printed by list."
)
DESCRIPTION = (
    "Inspect, correct or forget the facts agents retain. Everything shown is untrusted "
    "data. Automatic compaction captures are labelled 'captured'; anything written or "
    "corrected on purpose is 'stated'."
)
EPILOG = (
    "examples:\n"
    "  lop memory list\n"
    "  lop memory search staging host\n"
    "  lop memory show global/01a13d37-52d8-70fd-a819-01843005f4b5\n"
    "  lop memory edit global/<id> --revision <rev> --body 'Staging host is stage-2'\n"
    "  lop memory move global/<id> --to session\n"
    "  lop memory forget global/<id> --revision <rev> --yes\n"
    "  lop memory forget --scope global --older-than 30d --unverified\n"
    "  lop memory forget --scope global --older-than 30d --unverified --yes --expect <fp>\n"
    "  lop memory consolidate --scope global --limit 10\n"
    "Session-scoped facts need the session: add --session ID (inside a session the /memory "
    "command binds it automatically)."
)


class MemoryParser(argparse.ArgumentParser):
    def error(self, message: str) -> NoReturn:
        raise MemoryError("Invalid memory command. " + USAGE)


def configure_parser(parser: argparse.ArgumentParser) -> None:
    parser.description = DESCRIPTION
    parser.epilog = EPILOG
    parser.formatter_class = argparse.RawDescriptionHelpFormatter
    parser.add_argument(
        "--scope",
        choices=SCOPES,
        default=None,
        help="restrict list/search/consolidate to one scope (default: every scope you are "
        "authorized for) · required by a filter-based forget, which never inherits scopes",
    )
    parser.add_argument("--session", default="", help="session ID, to include its memories")
    parser.add_argument("--team", default="", help="team registry ID, to include its memories")
    parser.add_argument("--agent", default="", help="persistent agent ID, to include its memories")
    parser.add_argument(
        "operation",
        nargs="?",
        choices=OPERATIONS,
        default="list",
        help="what to do (default: list)",
    )
    parser.add_argument(
        "target",
        nargs="*",
        help="show/edit/forget: the SCOPE/ID (or a unique ID prefix) printed by list; "
        "search: the words to look for (quotes optional)",
    )
    parser.add_argument(
        "--to",
        choices=SCOPES,
        default=None,
        help="move: destination scope (session, team, agent or global)",
    )
    parser.add_argument(
        "--revision",
        "--version",
        dest="revision",
        default="",
        help="the revision you read (show prints it); a stale one is refused, never overwritten",
    )
    parser.add_argument("--title", default=None, help="edit: new title")
    parser.add_argument("--body", default=None, help="edit: new body")
    parser.add_argument(
        "--tag",
        action="append",
        default=None,
        help="edit: set the tags (repeat; replaces all) · forget filters: match any of "
        "these tags (repeat)",
    )
    parser.add_argument(
        "--alias", action="append", default=None, help="edit: set the aliases (repeat; replaces)"
    )
    parser.add_argument("--verified", action="store_true", help="edit: mark verified just now")
    parser.add_argument(
        "--replace",
        dest="replacement",
        default="",
        help="edit: a complete replacement fact as QUOTED JSON (field flags are easier)",
    )
    parser.add_argument(
        "--resolve",
        action="store_true",
        help="edit: also supersede every conflicting sibling revision shown by show",
    )
    parser.add_argument(
        "--yes",
        action="store_true",
        help="forget: confirm the removal (with filters: perform the previewed set)",
    )
    parser.add_argument(
        "--older-than",
        dest="older_than",
        default="",
        metavar="DURATION",
        help="forget filters: match facts created at least this long ago (90m, 12h, 7d, 2w)",
    )
    parser.add_argument(
        "--unverified",
        action="store_true",
        help="forget filters: match facts that were never verified",
    )
    parser.add_argument(
        "--title-matches",
        dest="title_matches",
        default="",
        metavar="REGEX",
        help="forget filters: match the title (case-insensitive regular expression)",
    )
    parser.add_argument(
        "--body-matches",
        dest="body_matches",
        default="",
        metavar="REGEX",
        help="forget filters: match the body (case-insensitive regular expression)",
    )
    parser.add_argument(
        "--expect",
        default="",
        metavar="FINGERPRINT",
        help="forget filters: the fingerprint printed by the dry run being confirmed",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="forget filters: allow removing a whole scope or more than half of it",
    )
    parser.add_argument("--related", action="store_true", help="show: also list linked facts")
    parser.add_argument(
        "--cursor", default="", help="list/inspect: continue after this ID (printed as More:)"
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="inspect: entries per page (1–50, default 20) · "
        "consolidate: max groups (1–50, default 10)",
    )
    parser.add_argument(
        "--json", dest="raw", action="store_true", help="print machine-readable JSON"
    )


def _bound(store: MemoryStore, scope: str | None) -> None:
    """Say how to bind a session instead of the registry's generic refusal (UX U1)."""
    if scope == "session" and not store.identity.session_id:
        raise MemoryError(
            "No session is bound here. Pass --session ID to read a session's memories "
            "(inside a session, /memory binds it automatically)."
        )


def _searched(store: MemoryStore, scope: str | None) -> list[str]:
    return [scope] if scope else [name for name, _ in store.identity.scopes()]


def _context(args: argparse.Namespace, store: MemoryStore, prefix: str) -> str:
    """The flags a pasted command needs to reach the same owners this one reached."""
    if not prefix.startswith("lop"):
        return ""  # the slash command is already bound to its session
    parts = []
    for flag in ("session", "team", "agent"):
        value = getattr(args, flag, "")
        if value:
            parts.append(f"--{flag} {value}")
    return (" " + " ".join(parts)) if parts else ""


def _device_name(device: str, root: Path) -> str:
    """The mesh's display name for ``device``, or ``""`` when this device has none.

    Best effort by contract: a name is cosmetic, never authority, and an
    unresolved peer still prints its id. Read-only across the network records so
    a receipt for a peer in any paired network can be labelled, not only the
    network the receipt happened to be written under.
    """
    try:
        from local_operator.network import store as network_store

        for record in network_store.list_networks(root):
            member = record.member(device)
            if member is not None and member.name:
                return str(member.name)
    except Exception:  # noqa: BLE001 — a label must never fail a status read
        return ""
    return ""


def _status(root: Path, store: MemoryStore) -> dict[str, Any]:
    from local_operator.network.memory import peer_status, settings

    try:
        sync_on, sync_scopes, excluded = settings(root)
        sync_error = ""
    except MemoryError as exc:
        sync_on, sync_scopes, excluded, sync_error = False, [], [], str(exc)
    values = policy(root)
    # A receipt stores the stable device id; the name is resolved here, at the one
    # place both the human renderer and `--json` read, so a failing peer is a
    # machine an operator can recognise rather than an opaque 32-hex token.
    peers = peer_status(root)
    for peer in peers:
        peer["name"] = _device_name(str(peer.get("device") or ""), root)
    return {
        "policy": values,
        "sync": {
            # Effective state: network.memory.enabled AND memory.enabled (see settings()).
            "enabled": sync_on,
            "configured": sync_on or not values["enabled"],
            "scopes": sync_scopes,
            "exclude_devices": excluded,
            "error": sync_error,
            "trust": (
                "Existing paired receiver-admin peers with memory-sync-v1 only; no new grants."
            ),
            "peers": peers,
            "retention": (
                "Tombstones/conflicts are retained; bounded saturation refuses new writes, "
                "never silently evicts facts."
            ),
        },
        "usage": store.inventory(),
        "limits": (
            "Shape guards are heuristic: they catch recognized sensitive patterns, not every "
            "secret. Unrecognized secrets can still be stored and journaled — keep values in "
            "lop secret and only the secret NAME in memory."
        ),
        "default_enabled": MEMORY_DEFAULT_ENABLED,
    }


#: Filter-based forget refuses to remove the whole scope, or more than this share of
#: it, without --force. Half is defensible: retiring stale entries is a minority of
#: a bank, while "this whole bank is junk" is a decision that should carry its own
#: explicit confirmation.
BULK_FORGET_MAX_FRACTION = 0.5

#: Preview rows rendered. The JSON payload keeps every row; the human view stays
#: readable even when a --force preview spans hundreds of facts.
BULK_PREVIEW_ROWS = 50

_DURATION = re.compile(r"^(\d+)([mhdw])$")
_DURATION_UNITS = {"m": 60.0, "h": 3600.0, "d": 86400.0, "w": 604800.0}


def _duration(text: str) -> float:
    """Seconds for a compact duration. Suffix only: a bare number is ambiguous."""
    match = _DURATION.match(text.strip().lower())
    if not match or int(match.group(1)) == 0:
        raise MemoryError(
            "--older-than needs a positive duration like 90m, 12h, 7d or 2w "
            "(minutes, hours, days, weeks)."
        )
    return int(match.group(1)) * _DURATION_UNITS[match.group(2)]


def _unique_prefixes(ids: list[str]) -> dict[str, str]:
    """Shortest prefix (from 8 characters) that is unique within ``ids``.

    WHY not a fixed 12: UUIDv7 leads with a millisecond timestamp, so ids minted
    in the same millisecond share it — review round 1 reproduced two different
    facts printed as ``01a1523f-6cc``. A preview must let a human act on a line
    without guessing which fact it names.
    """
    values = sorted(set(ids))
    widths = {value: min(8, len(value)) for value in values}
    changed = True
    while changed:
        changed = False
        buckets: dict[str, list[str]] = {}
        for value in values:
            buckets.setdefault(value[: widths[value]], []).append(value)
        for bucket in buckets.values():
            if len(bucket) > 1:
                for value in bucket:
                    if widths[value] < len(value):
                        widths[value] += 1
                        changed = True
    return {value: value[: widths[value]] for value in values}


def _fingerprint(items: list[dict[str, Any]]) -> str:
    """Short digest of the previewed (id, revision) set.

    The confirmed run must carry it back: a fact edited, forgotten, added or newly
    matched between preview and confirmation changes the digest, and the command is
    refused before anything is removed. This is the existing revision semantics,
    applied to a set instead of one row.
    """
    digest = hashlib.sha256()
    for row in sorted(items, key=lambda item: item["id"]):
        digest.update(f"{row['id']}:{row['version']}\n".encode())
    return digest.hexdigest()[:12]


def _bulk_flags(args: argparse.Namespace) -> list[str]:
    """The filter flags, quoted, so the printed confirmation command is paste-safe."""
    parts = ["--scope", args.scope]
    if args.older_than:
        parts += ["--older-than", args.older_than]
    if args.unverified:
        parts.append("--unverified")
    for tag in args.tag or []:
        parts += ["--tag", shlex.quote(tag)]
    if args.title_matches:
        parts += ["--title-matches", shlex.quote(args.title_matches)]
    if args.body_matches:
        parts += ["--body-matches", shlex.quote(args.body_matches)]
    return parts


def _bulk_filter_text(args: argparse.Namespace) -> str:
    """The filter echoed back in words, for a human who did not just read the flags."""
    parts = [f"scope {args.scope}"]
    if args.older_than:
        parts.append(f"older than {args.older_than.strip().lower()}")
    if args.unverified:
        parts.append("never verified")
    if args.tag:
        parts.append("tag " + " or ".join(repr(tag) for tag in args.tag))
    if args.title_matches:
        parts.append(f"title matches /{args.title_matches}/i")
    if args.body_matches:
        parts.append(f"body matches /{args.body_matches}/i")
    return " · ".join(parts)


def _bulk_forget(
    args: argparse.Namespace,
    store: MemoryStore,
    common: dict[str, Any],
    prefix: str,
) -> dict[str, Any]:
    """Dry-run-first filtered forget; the whole safety shape lives in this function."""
    if args.revision:
        raise MemoryError(
            "A filter-based forget re-checks each fact's own revision; --revision applies to "
            "one explicit ID."
        )
    if args.title is not None or args.body is not None or args.alias or args.verified:
        raise MemoryError("Field-edit flags (--title/--body/--alias/--verified) apply to `edit`.")
    if not (
        args.older_than or args.unverified or args.tag or args.title_matches or args.body_matches
    ):
        raise MemoryError(
            f"Give one memory ID, or filters to preview a bulk forget: {prefix} forget "
            "--scope S [--older-than D] [--unverified] [--tag T] [--title-matches RE] "
            "[--body-matches RE]."
        )
    if not args.scope:
        # Deliberate asymmetry with list/search: those may span every authorised
        # scope, but a bulk REMOVAL must name its bank. Inheriting the scope set
        # would leave the global personal bank one omitted flag away from a
        # cleanup aimed at session scope. Naming `global` is therefore deliberate.
        raise MemoryError(
            "A filter-based forget needs an explicit --scope (global, session, team or "
            "agent); it never inherits every scope. Name `--scope global` deliberately for "
            "the personal bank."
        )
    if args.expect and not args.yes:
        raise MemoryError(
            "--expect confirms a preview; pass it with --yes. Run the preview first "
            f"({prefix} forget {' '.join(_bulk_flags(args))}{common['context']})."
        )
    seconds = _duration(args.older_than) if args.older_than else None
    selection = store.select(
        args.scope,
        older_than=seconds,
        unverified=args.unverified,
        tags=args.tag,
        title_matches=args.title_matches or None,
        body_matches=args.body_matches or None,
    )
    items = selection["items"]
    total = selection["total"]
    if len(items) > MAX_BULK_FORGET:
        # The preview must never print a confirmation the store will reject: a
        # command that cannot run reads as a bug in the store, not a limit.
        raise MemoryError(
            f"This filter matches {len(items)} fact(s), but one bulk forget accepts at "
            f"most {MAX_BULK_FORGET}. Nothing was previewed or forgotten. Narrow the "
            "filter (for example add --older-than, --tag or --title-matches) and run "
            "again; a larger cleanup has to be done in chunks."
        )
    base = {
        **common,
        "scope": args.scope,
        "filter_text": _bulk_filter_text(args),
    }
    if not items:
        # Also the idempotent second run of a confirmed command: report zero
        # matches and succeed rather than failing on an already-clean store.
        return {
            **base,
            "action": "bulk-forget",
            "items": [],
            "count": 0,
            "total": total,
            "tokens": 0,
            "total_tokens": selection["total_tokens"],
            "fingerprint": "",
            "read_only": True,
        }
    fingerprint = _fingerprint(items)
    fraction = len(items) / total if total else 1.0
    guarded = len(items) == total or fraction > BULK_FORGET_MAX_FRACTION
    if not args.yes:
        # Deliberately WITHOUT --force: a guard whose own suggested command already
        # defeats it trains people to paste past the guard. The warning explains
        # the refusal and the two real remedies instead.
        command = (
            f"{prefix} forget {' '.join(_bulk_flags(args))}{common['context']} --yes"
            f" --expect {fingerprint}"
        )
        return {
            **base,
            "action": "bulk-forget-preview",
            "items": items,
            "count": len(items),
            "total": total,
            "fraction": round(fraction, 3),
            "tokens": selection["tokens"],
            "total_tokens": selection["total_tokens"],
            "fingerprint": fingerprint,
            "threshold": BULK_FORGET_MAX_FRACTION,
            "guarded": guarded,
            "confirm": command,
            "read_only": True,
        }
    if not args.expect:
        raise MemoryError(
            "Confirming a filter-based forget needs the fingerprint from its dry run. "
            f"Nothing was forgotten. Run: {prefix} forget {' '.join(_bulk_flags(args))}"
            f"{common['context']}"
        )
    if args.expect != fingerprint:
        raise MemoryError(
            "The selection changed since the preview you confirmed (fingerprint mismatch). "
            "Nothing was forgotten. Run the preview again: "
            f"{prefix} forget {' '.join(_bulk_flags(args))}{common['context']}"
        )
    if guarded and not args.force:
        raise MemoryError(
            f"Refused: this would forget {len(items)} of {total} fact(s) in {args.scope} "
            f"({fraction:.0%}) — the whole scope, or more than "
            f"{BULK_FORGET_MAX_FRACTION:.0%} of it, needs --force. Nothing was forgotten. "
            "If that is really intended, confirm the preview with --force."
        )
    done = store.forget_many(args.scope, {row["id"]: row["version"] for row in items})
    result = {
        **base,
        "action": "bulk-forgotten",
        "count": len(done["forgotten"]),
        "tokens": done["tokens"],
        "quota": done["quota"],
    }
    if done.get("warning"):
        result["warning"] = done["warning"]
    return result


def _consolidate(
    args: argparse.Namespace,
    store: MemoryStore,
    common: dict[str, Any],
) -> dict[str, Any]:
    """Read-only duplicate preview. Never writes: no import_session, no DML."""
    limit = 10 if args.limit is None else args.limit
    if not 1 <= limit <= 50:
        raise MemoryError("Duplicate preview limit must be 1–50.")
    return {
        **duplicates(store.scan(args.scope), limit=limit),
        **common,
        "action": "consolidate",
        "scope": args.scope,
        "limit": limit,
        "read_only": True,
    }


def execute(
    args: argparse.Namespace,
    *,
    config: Path | None = None,
    identity: MemoryIdentity | None = None,
    session_dir: Path | None = None,
    prefix: str = "/memory",
) -> dict[str, Any]:
    root = config or config_dir()
    current = identity or MemoryIdentity(args.session, args.team, args.agent)
    directory = session_dir or (
        root / "sessions" / current.session_id if current.session_id else None
    )
    store = MemoryStore(root, current, directory)
    scope = args.scope
    if getattr(args, "to", None) and args.operation != "move":
        # Same rule as --limit: a silently ignored flag reads as if it applied.
        # Checked BEFORE the status early return, which used to swallow --to.
        raise MemoryError("--to applies to `move` only.")
    if args.operation == "status":
        return _status(root, store)
    if args.limit is not None and args.operation not in ("inspect", "consolidate"):
        # A silently ignored flag reads as a filter that was applied.
        raise MemoryError("--limit applies to `inspect` and `consolidate` only.")
    filter_flags = {
        "--older-than": bool(args.older_than),
        "--unverified": bool(args.unverified),
        "--title-matches": bool(args.title_matches),
        "--body-matches": bool(args.body_matches),
        "--expect": bool(args.expect),
        "--force": bool(args.force),
    }
    if args.operation != "forget":
        stray = [name for name, used in filter_flags.items() if used]
        if args.tag and args.operation != "edit":
            stray.append("--tag")
        if stray:
            raise MemoryError(
                f"{', '.join(stray)} "
                + ("applies" if len(stray) == 1 else "apply")
                + " to a filter-based `forget` only (--tag also sets tags on `edit`)."
            )
    if args.operation == "inspect":
        # Inspect lists; it never targets a row. A silently ignored target reads
        # as a filter that was applied (same argument as the --limit refusal
        # above), so point at the verb that does read one fact.
        if args.target:
            raise MemoryError(
                f"`{prefix} inspect` lists instead of reading one fact; use "
                f"`{prefix} show {args.target[0]}` for that row."
            )
        # Deliberately BEFORE import_session(): that call writes a session's
        # portable log into the store. Inspection must not change a row, so it
        # never reaches any mutable store method.
        from local_operator.memory_inspect import DEFAULT_LIMIT, inspect_entries

        _bound(store, scope)
        return {
            **inspect_entries(
                root,
                current,
                scope=scope,
                limit=DEFAULT_LIMIT if args.limit is None else args.limit,
                cursor=args.cursor or None,
            ),
            "agent_memory": policy(root)["enabled"],
            "searched": _searched(store, scope),
            "bound_session": current.session_id,
            "context": _context(args, store, prefix),
        }
    if args.operation == "consolidate":
        # Also BEFORE import_session(): that call writes a session's portable log
        # into the index, and consolidate promises "read-only, changed nothing".
        if args.target:
            raise MemoryError(
                f"`{prefix} consolidate` reviews the whole scope; it never targets a row. "
                f"Use `{prefix} show {args.target[0]}` for that fact."
            )
        _bound(store, scope)
        return _consolidate(
            args,
            store,
            {
                "agent_memory": policy(root)["enabled"],
                "searched": _searched(store, scope),
                "bound_session": current.session_id,
                "context": _context(args, store, prefix),
            },
        )
    _bound(store, scope)
    store.import_session()
    words = list(args.target)
    agent_memory = policy(root)["enabled"]
    common = {
        "agent_memory": agent_memory,
        "searched": _searched(store, scope),
        "bound_session": current.session_id,
        "context": _context(args, store, prefix),
    }
    if args.operation == "list":
        return {**store.inventory(scope, cursor=args.cursor or None), **common}
    if args.operation == "search":
        if not words:
            raise MemoryError(f"Search needs words, e.g. {prefix} search staging host.")
        return {"results": store.search(" ".join(words), scope), "query": True, **common}
    if args.operation == "move":
        if not args.to:
            raise MemoryError(f"Move needs a destination: {prefix} move SCOPE/ID --to session.")
        if not words:
            raise MemoryError(f"A move needs at least one ID: {prefix} move SCOPE/ID --to session.")
        if len(words) > 64:
            raise MemoryError("A move accepts at most 64 memories at once.")
        try:
            value = store.move(
                list(words),
                args.to,
                expected_version=args.revision or None,
                scope=scope,
            )
        except MemoryError as exc:
            if "not found" in str(exc):
                raise MemoryError(f"{exc} List what you can see with: {prefix} list") from None
            raise
        return {**value, "context": common["context"]}
    if args.operation == "forget":
        if not words:
            return _bulk_forget(args, store, common, prefix)
        if (
            args.older_than
            or args.unverified
            or args.tag
            or args.title_matches
            or args.body_matches
            or args.expect
            or args.force
        ):
            raise MemoryError(
                "Give either one memory ID or filter flags to preview a bulk forget, " "not both."
            )
    if not words:
        raise MemoryError(f"An explicit memory ID is required: {prefix} {args.operation} SCOPE/ID.")
    if len(words) > 1:
        if words[1].lstrip().startswith(("{", "[")):
            raise MemoryError(
                "invalid JSON: pass a complete replacement as QUOTED --replace '{...}', "
                "or change fields with --title/--body/--tag/--alias."
            )
        raise MemoryError(f"One memory ID at a time; got {len(words)} values. {USAGE}")
    try:
        scope, entity = store.resolve(words[0], scope)
    except MemoryError as exc:
        if "not found" in str(exc):
            raise MemoryError(f"{exc} List what you can see with: {prefix} list") from None
        raise
    reference = f"{scope}/{entity}"
    ctx = common["context"]
    if args.operation == "show":
        view = store.show(scope, entity, version=args.revision or None)
        if args.related:
            view["related"] = store.related(scope, entity)
        return {**view, "reference": reference, "context": ctx}
    head = store.show(scope, entity)
    title = (head["fact"] or {}).get("title", "")
    if head["status"] == "forgotten":
        raise MemoryError(f"{reference} was already forgotten; it cannot be changed.")
    if not args.revision:
        raise MemoryError(
            "Pass the revision you read so a concurrent change is never overwritten: "
            f"--revision {head['version']} (current; {prefix} show {reference}{ctx} lists it)."
        )
    if args.operation == "forget":
        if not args.yes:
            raise MemoryError(
                f"Nothing forgotten. This would permanently remove '{title}' ({reference}) "
                "here and on trusted synced peers. To confirm, run: "
                f"{prefix} forget {reference}{ctx} --revision {args.revision} --yes"
            )
        done = store.put(scope, {}, entity=entity, expected_version=args.revision, forget=True)
        return {**done, "action": "forgotten", "reference": reference, "title": title}
    flags = {
        "title": args.title,
        "body": args.body,
        "tags": args.tag,
        "aliases": args.alias,
    }
    changed = {key: value for key, value in flags.items() if value is not None}
    if args.verified:
        changed["verified_at"] = time.time()
    if args.replacement and changed:
        raise MemoryError("Use either --replace JSON or field flags, not both.")
    if args.replacement:
        try:
            replacement = json.loads(args.replacement)
        except json.JSONDecodeError:
            # Never echo the text: a half-typed value may be a secret.
            raise MemoryError(
                'invalid JSON: quote the whole value, e.g. --replace \'{"title":"…",'
                '"body":"…"}\', or change fields with --title/--body instead.'
            ) from None
        if not isinstance(replacement, dict):
            raise MemoryError("invalid JSON: --replace needs one JSON object.")
    elif changed:
        try:
            base = store.show(scope, entity, version=args.revision)["fact"]
        except MemoryError:
            raise MemoryError(
                "Memory changed concurrently or that revision is unknown. "
                f"Read the current one: {prefix} show {reference}{ctx}"
            ) from None
        if base is None:
            raise MemoryError("That revision has no body to correct.")
        replacement = {key: base[key] for key in _EDITABLE if key in base}
        replacement.update(changed)
    else:
        raise MemoryError(
            "Nothing to change. Pass --title/--body/--tag/--alias/--verified, "
            "or a complete --replace '{...}'."
        )
    source = replacement.get("source", {})
    if isinstance(source, dict):
        replacement["source"] = dict(source, author="user", kind=source.get("kind", "reported"))
    if replacement.get("status") == "candidate":
        # A human correction is a deliberate statement: it stops being an unreviewed capture
        # (and capture refreshes, which only touch candidates, can no longer overwrite it).
        replacement["status"] = "active"
    done = store.put(
        scope,
        replacement,
        entity=entity,
        expected_version=args.revision,
        merge=args.resolve,
    )
    return {
        **done,
        "action": "corrected",
        "reference": reference,
        "title": replacement.get("title", title),
    }


def run(
    text: str = "",
    *,
    config: Path | None = None,
    identity: MemoryIdentity | None = None,
    session_dir: Path | None = None,
    prefix: str = "/memory",
) -> tuple[str, bool]:
    parser = MemoryParser(prog="memory", add_help=False)
    configure_parser(parser)
    try:
        try:
            tokens = shlex.split(text)
        except ValueError:
            raise MemoryError("Could not parse the command: unbalanced quote.") from None
        # Intermixed so `search staging --scope session host` reads like the CLI does.
        args = parser.parse_intermixed_args(tokens)
        result = execute(
            args, config=config, identity=identity, session_dir=session_dir, prefix=prefix
        )
        return json.dumps(result, ensure_ascii=False, indent=2), False
    except (MemoryError, ValueError, TypeError, OSError, sqlite3.Error) as exc:
        return (
            str(exc)
            if isinstance(exc, MemoryError)
            else "Memory request refused or store unavailable. No unsafe payload included."
        ), True


def _span(seconds: float) -> str:
    seconds = max(0, int(seconds))
    for unit, size in (("d", 86400), ("h", 3600), ("m", 60)):
        if seconds >= size:
            return f"{seconds // size}{unit}"
    return "<1m"


def _age(seconds: float) -> str:
    span = _span(seconds)
    return "just now" if span == "<1m" else f"{span} ago"


def _date(value: Any, empty: str = "unknown", *, now: float | None = None) -> str:
    """Human UTC stamp plus relative age; raw epochs are an implementation detail."""
    if not isinstance(value, (int, float)):
        return empty
    stamp = datetime.fromtimestamp(value, timezone.utc).strftime("%Y-%m-%d %H:%M UTC")
    current = time.time() if now is None else now
    if value > current:
        return f"{stamp} (in {_span(value - current)})"
    return f"{stamp} ({_age(current - value)})"


def _source(source: dict[str, Any]) -> str:
    """Provenance as a sentence; it is untrusted data, so it is shown, never obeyed."""
    kind = source.get("kind") or "unknown"
    # Inferred/summary/compaction facts are weaker than reported/observed ones (D6).
    weak = (
        " — weaker: not stated by a person" if kind in ("inferred", "summary", "compaction") else ""
    )
    parts = [f"by {source.get('author') or 'unknown'} ({kind}){weak}"]
    if source.get("session"):
        parts.append(f"session {source['session']}")
    return " · ".join(parts)


def _row(row: dict[str, Any]) -> list[str]:
    """Two short lines so an 80-column terminal never clips the ID or splits a status."""
    title = row["title"] if len(row["title"]) <= 64 else row["title"][:63] + "…"
    origin = row.get("origin", "stated")
    state = origin if row["status"] in ("active", "candidate") else f"{origin} · {row['status']}"
    # NoticeBlock is single-colour, so state is carried by a marker, not by hue (D6): a
    # captured/disputed/expired row must not read like a verified one.
    marker = "  " if (origin == "stated" and row["status"] == "active") else "! "
    return [f"{row['scope']}/{row['id']}", f"{marker}{title}  [{state}]"]


def _searched_line(value: dict[str, Any], prefix: str) -> str:
    names = ", ".join(value.get("searched") or ["global"])
    line = f"Searched: {names}."
    if "session" not in (value.get("searched") or []) and not value.get("bound_session"):
        line += " Add --session ID to include a session's memories (captured items live there)."
    return line


#: What the operator can DO about a non-healthy sync receipt. The peer's own
#: refusal sentence wins when it has one (it names the exact conflict); these are
#: the fallbacks so a code whose refusal carried no sentence is still actionable.
_PEER_REMEDY: dict[str, str] = {
    "bad_memory": "Resolve the named memory conflict on the refusing peer; memory sync "
    "retries automatically after about 30 minutes (restarting the retrying device picks it "
    "up sooner).",
    # A receiver-side capability refusal (`_push_to_peer` reports `skipped:no_admin`
    # when the peer does not grant this device the admin the memory stream requires).
    # Without this row it fell through to advice that could not change the answer.
    "skipped:no_admin": "Grant receiver-admin to this device on the peer, or exclude "
    "this peer from memory sync.",
    "peer_too_old": "Run lop-update on both peers.",
    "refused": "Grant this device memory access on the peer.",
    "not_authorised": "Grant this device memory access on the peer.",
    "capability_denied": "Grant memory-sync-v1 on the peer.",
    "not_a_member": "Re-pair this device with the peer's network.",
    "policy": "Review the peer's memory policy.",
    "unreachable": "The peer is offline; sync resumes when it returns.",
    "push_failed": "A transport failure; it retries automatically.",
    "memory_sync_failed": "A local failure while pushing; inspect `lop memory status --json`.",
    "scope_or_version_mismatch": "Run lop-update on both peers; their memory-sync versions differ.",
}


def _status_text(value: dict[str, Any]) -> str:
    policy_ = value["policy"]
    sync = value["sync"]

    def onoff(flag: bool) -> str:
        return "on" if flag else "off"

    capture_locators = policy_["enabled"] and policy_["capture"] and policy_["locators"]
    lines = [
        f"Agent memory: {onoff(policy_['enabled'])} (memory.enabled)"
        + ("" if policy_["enabled"] else " — agents cannot read or write it; facts are kept"),
        f"Compaction capture: {onoff(policy_['enabled'] and policy_['capture'])} (memory.capture)",
        f"Capture locator context: {onoff(capture_locators)} "
        "(memory.capture.locators; default off)",
        f"Recall suggestions: {onoff(policy_['enabled'] and policy_['suggest'])} (memory.suggest)",
    ]
    if sync.get("error"):
        lines.append(f"Trusted-peer sync: off — {sync['error']} (network.memory)")
    else:
        lines.append(
            f"Trusted-peer sync: {onoff(sync['enabled'])} (network.memory.enabled, "
            "and memory.enabled)"
            + (f" · scopes {', '.join(sync['scopes'])}" if sync["enabled"] else "")
            + (
                f" · {len(sync['exclude_devices'])} excluded device(s)"
                if sync["exclude_devices"]
                else ""
            )
        )
    from local_operator.network.memory import HEALTHY_PEER_CODES

    peers = sync.get("peers") or []
    # A receipt that is not a healthy code is a peer whose memory stream is NOT
    # moving. Counting it under "synced" was the round-2 defect: the operator was
    # told everything was fine while every later op queued behind a refusal.
    failing = [p for p in peers if str(p.get("code") or "") not in HEALTHY_PEER_CODES]
    summary = f"Peers synced: {len(peers) - len(failing)}"
    if peers:
        if failing:
            summary += f" · {len(failing)} failing"
    else:
        summary += " (none yet)"
    lines.append(summary)
    for receipt in failing:
        device = str(receipt.get("device") or "unknown device")
        code = str(receipt.get("code") or "unknown")
        # The peer's own sentence names the exact conflict and the way out; the map
        # is the fallback for a code whose refusal carried no sentence. The sentence
        # is UNTRUSTED terminal input from a paired peer: strip control sequences
        # with the repo's own helper BEFORE whitespace collapsing (a newline inside
        # an OSC payload must not survive), because this is the first route by which
        # arbitrary peer text reaches the operator's terminal.
        from local_operator.ansi import strip_control_sequences

        message = " ".join(strip_control_sequences(str(receipt.get("message") or "")).split())
        fallback = _PEER_REMEDY.get(code, "Run `lop memory status --json` for this peer's receipt.")
        # Peer verbosity must not own this page's hierarchy: an unbounded sentence
        # pushed `1 failing` out of a 40-row viewport. The excerpt is bounded, and
        # the code-keyed remedy is printed separately when the sentence was cut
        # before it could carry one; the full text stays in `--json`.
        truncated = len(message) > PEER_EXCERPT_CHARS
        if truncated:
            message = message[:PEER_EXCERPT_CHARS].rsplit(" ", 1)[0].rstrip() + " …"
        # The display name arrives verbatim from a peer's pairing handshake
        # (`MemberRecord.name`), and an older build may already have stored a
        # hostile or oversized one. The render point is therefore the boundary
        # that protects every row, not just names admitted after a fix: strip the
        # same control sequences as the sentence and bound the width, or a name
        # is a second unauthenticated route into the terminal.
        name = " ".join(strip_control_sequences(str(receipt.get("name") or "")).split())
        if len(name) > PEER_NAME_CHARS:
            name = name[: PEER_NAME_CHARS - 1].rstrip() + "…"
        label = name or device
        if label != device:
            label = f"{label} ({device})"
        lines += _shaped(f"Not syncing with {label}: {code}. {message or fallback}")
        if truncated:
            if code in _PEER_REMEDY and _PEER_REMEDY[code] not in message:
                lines += _shaped(f"Remedy: {fallback}")
            lines += _shaped("Full peer text: `lop memory status --json`.")
    usage = value["usage"]
    lines.append(
        f"Stored: {usage['active_tokens']} tokens in use · {usage['ops']} edits recorded · "
        f"{usage.get('storage_bytes', 0)} bytes"
    )
    quotas = usage.get("quotas") or []
    if quotas:
        # Wrapped at the repo's row bar: a NoticeBlock clips a longer line at the
        # pane edge instead of wrapping it, and a clipped percentage/headroom is
        # the one number this line exists to show.
        lines += _shaped(
            "Quota: "
            + " · ".join(
                f"{row['scope']} {row['used_tokens']}/{row['cap_tokens']} "
                f"({row['percent']}%, {row['headroom_tokens']} left)"
                for row in quotas
            )
        )
        near = [row for row in quotas if row["percent"] >= 80]
        # ONE LINE PER SCOPE, each with its own relief. A combined sentence used to
        # tell a session at 96% to move facts "to session scope" — a session->session
        # move the CLI refuses. The advice is `quota_relief`, the same source the
        # per-write warning reads, so the two surfaces cannot contradict each other.
        for row in near:
            advice, hint = quota_relief(str(row["scope"]))
            lines += _shaped(
                f"Near cap: {row['scope']} {row['percent']}% "
                f"({row['used_tokens']}/{row['cap_tokens']}, {row['headroom_tokens']} left). "
                + advice
            )
            if hint:
                # The command sits on its own line so wrapping never splits it.
                lines += _shaped(f"Free space: {hint}.")
    lines.append(value["limits"])
    return "\n".join(lines)


_PROPOSALS = {
    "keep": "keep",
    "session-scoped": "move to a session",
    "team": "move to a team",
    "review-work-scope": "review as work scope",
}

#: How much of a peer's refusal sentence the status page shows. A peer controls
#: its own sentence length; without a bound a 2,448-character diagnostic pushed
#: the `1 failing` header out of a 40-row viewport. The full text stays in
#: `lop memory status --json`, and the code-keyed remedy is printed beside it.
PEER_EXCERPT_CHARS = 240

#: How much of a peer's display name is shown. A name is peer-supplied and was
#: stored verbatim; bounded at 48 so the name plus the prefix stays under the
#: 80-column bar even before the stable id is appended.
PEER_NAME_CHARS = 48

#: The repo's own list-row bar (see ``_row``: "an 80-column terminal never
#: clips the ID or splits a status"). Every line of this page is shaped to it.
SHAPED_WIDTH = 80


def _shaped(line: str, width: int = SHAPED_WIDTH) -> list[str]:
    """Wrap one rendered line to the 80-column bar on word boundaries.

    ``break_long_words=False``/``break_on_hyphens=False``: an identifier, a
    timestamp or a path longer than the bar overflows on one line rather than
    being split into unreadable halves (measured before this: 44% of lines
    exceeded 80 columns and tokens were cut mid-word). Continuation lines hang
    under the line's own indent so a wrapped block still reads as one block.
    """
    if not line.strip():
        return [line]
    indent = line[: len(line) - len(line.lstrip())]
    return textwrap.wrap(
        line.strip(),
        width=width,
        initial_indent=indent,
        subsequent_indent=indent + "  ",
        break_long_words=False,
        break_on_hyphens=False,
    ) or [indent]


def _inspect_text(value: dict[str, Any], prefix: str) -> str:
    """Render the read-only preview.

    The word "proposal" and the trailing "NOT applied" are load-bearing: a person
    must never read this page as a record of a change. Withheld content says WHY
    it is withheld, in the same voice the write guard refuses it.

    Ordering is stated and visible: rows arrive newest-first by the fact's own
    date (see ``memory_inspect.inspect_entries``), and a row whose proposal is
    not "keep" carries a gutter arrow. At 20+ rows the three rows that need a
    decision must be findable without reading every block of prose.
    """
    entries = value["entries"]
    counts = value["counts"]
    now = value.get("generated_at") or time.time()
    lines = ["Read-only preview — nothing below is moved, edited or forgotten."]
    if not value.get("agent_memory", True):
        lines.append("Agent memory is OFF (memory.enabled=false): agents cannot use these.")
    by_scope = counts.get("by_scope") or {}
    summary = " · ".join(f"{name} {count}" for name, count in sorted(by_scope.items()))
    lines += _shaped(
        f"{counts['total']} entr{'y' if counts['total'] == 1 else 'ies'} in the authorized "
        f"scopes ({summary or 'none'}) · {counts['returned']} shown"
    )
    if not entries:
        # A stale cursor spends its emptiness on the CURSOR, not the store: the
        # counts above are real, so "nothing retained" would be a false claim
        # about data this page simply stopped listing at.
        if value.get("cursor_stale"):
            lines += _shaped(
                "Cursor no longer available — start from the top (the entry it named is "
                "gone or no longer in an authorized scope)."
            )
        else:
            lines.append("Nothing retained in these scopes.")
        return "\n".join(lines)
    suggested = counts.get("suggestions")
    if suggested is None:
        suggested = sum(1 for entry in entries if entry["disposition"]["proposal"] != "keep")
    decision = (
        f"{suggested} of {counts['returned']} shown "
        f"carr{'ies' if suggested == 1 else 'y'} a move suggestion (→)."
        if suggested
        else "No row on this page carries a move suggestion."
    )
    lines += _shaped(f"Order: newest first by each fact's own date. {decision}")
    for entry in entries:
        proposed = entry["disposition"]
        state = (
            entry["origin"]
            if entry["status"] in ("active", "candidate")
            else f"{entry['origin']} · {entry['status']}"
        )
        age = _date(entry["created_at"], "age unknown", now=now)
        # The arrow is the scan column D1 asked for: two cells of gutter, so the
        # references still line up while suggestion rows stand out.
        marker = "→ " if proposed["proposal"] != "keep" else "  "
        lines.append("")
        # A gate-refused identifier is shown as withheld, never as the literal
        # "None" (which reads like an id) or the raw column (the leak itself).
        reference = entry["reference"] or "[id withheld — the privacy guard would refuse it]"
        revision = entry["revision"] or "withheld"
        lines += _shaped(f"{marker}{reference}  [{state} · {age}]")
        if entry["title_exported"]:
            lines += _shaped(f"  {entry['title']}")
        else:
            lines.append("  [title withheld — the privacy guard would refuse it]")
        provenance = entry["provenance"]
        author = provenance.get("author") or "unknown"
        kind = provenance.get("kind") or "unknown"
        meta = [f"by {author} ({kind})"]
        if provenance.get("session"):
            meta.append(f"session {provenance['session']}")
        # The flag is emitted by _entry as a SIBLING of the provenance dict, not
        # inside it; reading it from the dict silently dropped the disclosure.
        if entry["provenance_redacted"]:
            meta.append("locator withheld by the privacy guard")
        lines += _shaped(
            "  Source: " + " · ".join(meta) + f" · revision {revision}"
            f" · {entry['revisions']} revision(s)"
        )
        if entry["body_preview"] is not None:
            suffix = "…" if entry["body_truncated"] else ""
            lines += _shaped(f"  Preview: {entry['body_preview']}{suffix}")
        else:
            lines.append("  Preview withheld — the privacy guard would refuse this body.")
        markers = ", ".join(entry["work_markers"]) or "none"
        lines += _shaped(
            f"  Proposal: {_PROPOSALS.get(proposed['proposal'], proposed['proposal'])}"
            f" → {proposed['destination']} — {proposed['reason']} (markers: {markers}; "
            "NOT applied)"
        )
    if value.get("next_cursor"):
        lines.append("")
        # Every flag that shaped this page, so the pasted command shows the same
        # rows at the same size (a hint that dropped --limit changed 5 to 20).
        more = f"{prefix} inspect --cursor {value['next_cursor']}"
        if value.get("limit"):
            more += f" --limit {value['limit']}"
        if value.get("scope"):
            more += f" --scope {value['scope']}"
        more += value.get("context") or ""
        # ONE line, like every other paste-ready hint this runner prints
        # (`More:`/`Next:` in the list view): a command is copied, not read, and
        # an 80-column wrap would split it mid-flag. The 80-column bar applies
        # to the prose and row blocks, not to a command token stream.
        lines.append(f"More: {more}")
    elif value.get("next_cursor_withheld"):
        lines.append("")
        # The listing is longer than what was shown, but the key to the next
        # page is itself a withheld identifier. Saying nothing here would read
        # as "that is all there is" in an audit view that exists to be trusted.
        lines += _shaped(
            "More rows exist, but paging cannot reach them: the next row's "
            "identifier is withheld by the privacy guard, so no cursor can be "
            "printed. Raise --limit to see the following rows on this page."
        )
    # The recipe, not just the implication: a move is a write in the destination
    # scope followed by a forget of the original, and forget needs the revision
    # the read just printed.
    lines.append("")
    lines += _shaped(
        "No change was made. To carry out a proposal, read one fact with "
        f"`{prefix} show SCOPE/ID`, state it again in the destination scope "
        f"(memory:// for an agent), then `{prefix} forget SCOPE/ID --revision REV --yes`."
    )
    return "\n".join(lines)


def _quota_receipt(value: dict[str, Any]) -> list[str]:
    """The headroom/warning tail every WRITE receipt carries.

    Centralized because four branches (moved/corrected/forgotten) must agree: the
    store has always returned ``quota`` and ``warning``, and a receipt that drops
    them makes the 80/90 warning invisible exactly where a write happened.
    """
    lines: list[str] = []
    quota = value.get("quota") or {}
    if quota:
        lines += _shaped(
            f"Quota: {quota['used_tokens']}/{quota['cap_tokens']} tokens "
            f"({quota['percent']}%, {quota['headroom_tokens']} left)"
        )
    if value.get("warning"):
        lines += _shaped(value["warning"])
    return lines


def _bulk_preview_text(value: dict[str, Any]) -> str:
    """Render the dry run. "Nothing was deleted" is load-bearing: a filtered run
    looks like a batch operation, and a preview must never read like a receipt."""
    lines = [f"Filter: {value['filter_text']}"]
    lines += _shaped(
        f"{value['count']} match(es) of {value['total']} fact(s) in {value['scope']} "
        f"({value['fraction']:.0%}) · {value['tokens']:,} token(s) would be freed "
        f"(of {value['total_tokens']:,} in use)."
    )
    if value["guarded"]:
        lines += _shaped(
            f"Warning: that is the whole scope, or more than {value['threshold']:.0%} of it; "
            "confirming without --force is refused. Narrow the filter, or add --force "
            "deliberately if the whole set is really intended."
        )
    prefixes = _unique_prefixes([row["id"] for row in value["items"]])
    for row in value["items"][:BULK_PREVIEW_ROWS]:
        # Collapse whitespace: a title is untrusted data and must stay on one line.
        title = " ".join(row["title"].split())
        if len(title) > 64:
            title = title[:63] + "…"
        lines.append(f"  {row['scope']}/{prefixes[row['id']]}  {title}  [{row['tokens']} tokens]")
    hidden = len(value["items"]) - BULK_PREVIEW_ROWS
    if hidden > 0:
        lines.append(f"  … and {hidden} more (--json lists every row).")
    lines.append("")
    lines.append("Nothing was deleted.")
    lines += _shaped(f"To forget exactly these {value['count']} fact(s), run:")
    lines.append(f"  {value['confirm']}")
    return "\n".join(lines)


def _bulk_empty_text(value: dict[str, Any]) -> str:
    """Zero matches is a success, not an error: the idempotent repeat run lands here."""
    return (
        f"Filter: {value['filter_text']}\n"
        f"No facts in {value['scope']} match. Nothing was forgotten."
    )


def _bulk_receipt_text(value: dict[str, Any]) -> str:
    if value["count"] == 0:
        return _bulk_empty_text(value)
    lines = _shaped(
        f"Forgotten {value['count']} fact(s) in {value['scope']} · {value['tokens']:,} "
        "token(s) freed. The bodies are removed here, and the removal propagates to trusted "
        "synced peers when memory sync is on. It cannot be undone; store new facts instead."
    )
    return "\n".join(lines + _quota_receipt(value))


def _consolidate_text(value: dict[str, Any], prefix: str) -> str:
    """Render the duplicate preview. The word "proposes" and the closing "nothing
    was changed" keep a review aid from reading as an applied merge."""
    lines = ["Read-only duplicate preview — nothing is forgotten, edited or merged."]
    if not value.get("agent_memory", True):
        lines.append("Agent memory is OFF (memory.enabled=false): agents cannot use these.")
    where = f"in {value['scope']}" if value.get("scope") else "across the authorized scopes"
    lines += _shaped(
        f"Checked {value['fact_count']} fact(s) {where} · {value['group_count']} "
        f"duplicate group(s) found · {len(value['groups'])} shown (limit {value['limit']})."
    )
    if not value["groups"]:
        lines.append("No identical or near-duplicate candidates. Nothing to consolidate.")
        return "\n".join(lines)
    ctx = value.get("context", "")
    prefixes = _unique_prefixes(
        [member["id"] for group in value["groups"] for member in group["members"]]
    )
    for number, group in enumerate(value["groups"], 1):
        kind = "identical" if group["exact"] else "near-duplicate"
        lines.append("")
        lines += _shaped(
            f"Group {number} — {kind} ({len(group['members'])} facts in {group['scope']}):"
        )
        for member in group["members"]:
            title = " ".join(member["title"].split())
            if len(title) > 64:
                title = title[:63] + "…"
            lines.append(
                f"  {group['scope']}/{prefixes[member['id']]}  {title}  "
                f"[{member['tokens']} tokens]"
            )
        for pair in group["pairs"]:
            if pair["exact"]:
                # The comparison normalises case/punctuation/whitespace; say so,
                # or a reviewer may expect a byte-for-byte match.
                evidence = "identical title and body (case/punctuation-insensitive)"
            elif pair["title_equal"]:
                evidence = f"identical title · combined token overlap {pair['similarity']:.0%}"
            else:
                evidence = f"{pair['similarity']:.0%} combined title+body token overlap"
            if not pair["exact"] and pair["shared"]:
                # The store computes the shared tokens; the human view must show
                # them too, or "the shared/differing tokens" claim in the guide is
                # false and a reviewer has to open both facts.
                evidence += f" · shared: {', '.join(pair['shared'])}"
            if pair["a_only"] or pair["b_only"]:
                evidence += (
                    f" · only first: {', '.join(pair['a_only']) or '—'}"
                    f" · only second: {', '.join(pair['b_only']) or '—'}"
                )
            lines += _shaped(
                f"  Evidence {prefixes[pair['a']]} vs {prefixes[pair['b']]}: {evidence}"
            )
        first, second = group["members"][0], group["members"][1]
        lines.append(f"  Review: {prefix} show {group['scope']}/{first['id']}{ctx}")
        lines.append(
            f"  If the second is redundant: {prefix} forget {group['scope']}/{second['id']}"
            f"{ctx} --revision {second['version']} --yes"
        )
    lines.append("")
    lines += _shaped(
        "Nothing was changed. Consolidate only proposes: review with show, merge with edit, "
        "and retire the redundant fact with forget."
    )
    if value.get("limited"):
        lines.append(
            f"More groups exist ({value['group_count']} total); raise --limit to view them."
        )
    return "\n".join(lines)


def render_human(text: str, failed: bool = False, prefix: str = "/memory") -> str:
    """Readable receipt for CLI and slash command; ``--json`` keeps the raw payload."""
    if failed:
        return text
    value = json.loads(text)
    # Action dispatch comes FIRST: a bulk preview carries ``items`` and would
    # otherwise render as an inventory page that wrongly claims nothing matched.
    if value.get("action") == "bulk-forget-preview":
        return _bulk_preview_text(value)
    if value.get("action") == "bulk-forget":
        return _bulk_empty_text(value)
    if value.get("action") == "bulk-forgotten":
        return _bulk_receipt_text(value)
    if value.get("action") == "consolidate":
        return _consolidate_text(value, prefix)
    if "entries" in value:
        return _inspect_text(value, prefix)
    if "results" in value:
        rows = value["results"]
        if not rows:
            return "No matching memories. " + _searched_line(value, prefix)
        lines = [f"{len(rows)} match(es). " + _searched_line(value, prefix)]
        for row in rows:
            lines += _row(row)
        lines.append(f"Next: {prefix} show {rows[0]['scope']}/{rows[0]['id']}{value['context']}")
        return "\n".join(lines)
    if value.get("action") == "moved":
        rows = value["moved"]
        lines = [f"Moved {len(rows)} fact(s) to {value['destination']}."]
        for row in rows:
            # Two UUIDs plus an arrow exceed the 80-column bar; a clipped destination
            # would hide the half of the receipt that carries the new reference.
            lines += _shaped(f"  {row['from']} → {row['to']}")
        if rows:
            lines += _shaped(
                f"Next: {prefix} show {rows[0]['to']}{value.get('context', '')} --related "
                "(the previous scope is shown as moved_from)"
            )
        lines += _quota_receipt(value)
        return "\n".join(lines)
    if "items" in value:
        items = value["items"]
        lines = []
        if not value.get("agent_memory", True):
            lines.append("Agent memory is OFF (memory.enabled=false): agents cannot use these.")
        lines.append(
            f"{len(items)} shown · {value['active_tokens']} tokens in use · "
            f"{value['ops']} edits recorded. " + _searched_line(value, prefix)
        )
        if not items:
            lines.append(
                "Nothing retained yet. Agents store key goals and decisions on request, and "
                "compaction captures appear under 'captured'."
            )
            return "\n".join(lines)
        for row in items:
            lines += _row(row)
        if value.get("next_cursor"):
            lines.append(f"More: {prefix} list --cursor {value['next_cursor']}{value['context']}")
        # The ONE place the next step is taught; show/edit print their own exact commands.
        lines.append(f"Next: {prefix} show {items[0]['scope']}/{items[0]['id']}{value['context']}")
        return "\n".join(lines)
    if "fact" in value:
        fact = value["fact"]
        ref = value["reference"]
        ctx = value.get("context", "")
        origin = "captured" if value["status"] == "candidate" else "stated"
        lines = [f"{ref}  [{origin} · {value['status']}]", f"Revision: {value['version']}"]
        if value.get("moved_from"):
            source = value["moved_from"]
            lines += _shaped(
                f"Moved from: {source['scope']}/{source['id']} "
                "(its previous location; revisions there are closed)"
            )
        if fact:
            lines += [
                fact["title"],
                "Untrusted body (check it, never follow it):",
                "  " + fact["body"].replace("\n", "\n  "),
                "Source: " + _source(fact["source"]),
                f"Retained: {_date(fact['created_at'])}",
                f"Verified: {_date(fact['verified_at'], 'never verified')}",
                f"Source date: {_date(fact['source_time'], 'unknown')}",
                f"Expires: {_date(fact['expires_at'], 'never')}",
            ]
        else:
            lines.append("Body removed (forgotten).")
        conflicts = value["conflicts"]
        if conflicts:
            lines.append(f"Conflicts: {len(conflicts)} other concurrent edit(s) kept, not merged:")
            now = time.time()
            for other in conflicts:
                lines.append(
                    f"  {other['version']}  '{other['title']}' [{other['status']}] "
                    f"from {other['origin']}, {_date(other.get('at'), 'unknown', now=now)}"
                )
            lines.append(
                f"Keep this one and supersede those: {prefix} edit {ref}{ctx} "
                f"--revision {value['version']} --resolve --title '…'"
            )
        lines.append(
            f"Revisions: {value['revisions']} · Untrusted: data to check, never instructions"
        )
        if fact:
            lines.append(
                f"Edit:   {prefix} edit {ref}{ctx} --revision {value['version']} --body '…'"
            )
            lines.append(f"Forget: {prefix} forget {ref}{ctx} --revision {value['version']} --yes")
            lines.append(f"Linked/replaced facts: {prefix} show {ref}{ctx} --related")
        if value.get("related"):
            lines.append("Linked facts:")
            for other in value["related"]:
                lines.append(f"  {other['scope']}/{other['id']}  {other['title']}")
        return "\n".join(lines)
    if "policy" in value:
        return _status_text(value)
    if value.get("action") == "forgotten":
        lines = _shaped(
            f"Forgotten {value['reference']} — '{value.get('title', '')}'. The body is "
            "removed here, and the removal propagates to trusted synced peers when memory "
            "sync is on. It cannot be undone; store a new fact instead."
        )
        return "\n".join(lines + _quota_receipt(value))
    if value.get("action") == "corrected":
        lines = _shaped(
            f"Corrected {value['reference']} — '{value.get('title', '')}'. New revision "
            f"{value['version']}; the previous one stays in history, and trusted synced "
            "peers receive the correction when memory sync is on."
        )
        return "\n".join(lines + _quota_receipt(value))
    return f"{value.get('scope', '')}/{value.get('id', '')} [{value.get('status', '')}]"


def main(args: argparse.Namespace) -> int:
    try:
        result = execute(args, prefix="lop memory")
        if args.raw:
            print(json.dumps(result, ensure_ascii=False, indent=2))
        else:
            print(render_human(json.dumps(result), prefix="lop memory"))
        return 0
    except (MemoryError, ValueError, TypeError, OSError, sqlite3.Error) as exc:
        print(
            str(exc)
            if isinstance(exc, MemoryError)
            else "Memory request refused or store unavailable."
        )
        return 1
