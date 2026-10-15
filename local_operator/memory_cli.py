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
import json
import shlex
import sqlite3
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, NoReturn

from local_operator.memory import SCOPES, MemoryError, MemoryIdentity, MemoryStore
from local_operator.memory_capture import MEMORY_DEFAULT_ENABLED, policy
from local_operator.paths import config_dir

OPERATIONS = ("list", "show", "search", "edit", "forget", "status")
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
    "Commands: list · show ID · search WORDS · status · "
    "edit ID --revision REV (--title T | --body B | --tag T | --alias A | --verified) · "
    "forget ID --revision REV --yes. ID is the SCOPE/ID printed by list."
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
    "  lop memory forget global/<id> --revision <rev> --yes\n"
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
        help="restrict list/search to one scope (default: every scope you are authorized for)",
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
        "--revision",
        "--version",
        dest="revision",
        default="",
        help="the revision you read (show prints it); a stale one is refused, never overwritten",
    )
    parser.add_argument("--title", default=None, help="edit: new title")
    parser.add_argument("--body", default=None, help="edit: new body")
    parser.add_argument(
        "--tag", action="append", default=None, help="edit: set the tags (repeat; replaces all)"
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
    parser.add_argument("--yes", action="store_true", help="forget: confirm the removal")
    parser.add_argument("--related", action="store_true", help="show: also list linked facts")
    parser.add_argument("--cursor", default="", help="list: continue after this ID")
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


def _status(root: Path, store: MemoryStore) -> dict[str, Any]:
    from local_operator.network.memory import peer_status, settings

    try:
        sync_on, sync_scopes, excluded = settings(root)
        sync_error = ""
    except MemoryError as exc:
        sync_on, sync_scopes, excluded, sync_error = False, [], [], str(exc)
    values = policy(root)
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
            "peers": peer_status(root),
            "retention": (
                "Tombstones/conflicts are retained; bounded saturation refuses new writes, "
                "never silently evicts facts."
            ),
        },
        "usage": store.inventory(),
        "limits": (
            "Shape guards are heuristic: they catch recognised sensitive patterns, not every "
            "secret. Unrecognised secrets can still be stored and journaled — keep values in "
            "lop secret and only the secret NAME in memory."
        ),
        "default_enabled": MEMORY_DEFAULT_ENABLED,
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
    if args.operation == "status":
        return _status(root, store)
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
    if not words:
        raise MemoryError(f"An explicit memory ID is required: {prefix} {args.operation} SCOPE/ID.")
    if len(words) > 1:
        if words[1].lstrip().startswith(("{", "[")):
            raise MemoryError(
                "invalid JSON: pass a complete replacement as QUOTED --replace '{...}', "
                "or change fields with --title/--body/--tag/--alias."
            )
        raise MemoryError(f"One memory ID at a time; got {len(words)} values. {USAGE}")
    scope, entity = store.resolve(words[0], scope)
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
    parts = [f"by {source.get('author') or 'unknown'} ({source.get('kind') or 'unknown'})"]
    if source.get("session"):
        parts.append(f"session {source['session']}")
    return " · ".join(parts)


def _row(row: dict[str, Any]) -> list[str]:
    """Two short lines so an 80-column terminal never clips the ID or splits a status."""
    title = row["title"] if len(row["title"]) <= 64 else row["title"][:63] + "…"
    origin = row.get("origin", "stated")
    state = origin if row["status"] in ("active", "candidate") else f"{origin} · {row['status']}"
    return [f"{row['scope']}/{row['id']}", f"  {title}  [{state}]"]


def _searched_line(value: dict[str, Any], prefix: str) -> str:
    names = ", ".join(value.get("searched") or ["global"])
    line = f"Searched: {names}."
    if "session" not in (value.get("searched") or []) and not value.get("bound_session"):
        line += " Add --session ID to include a session's memories (captured items live there)."
    return line


def _status_text(value: dict[str, Any]) -> str:
    policy_ = value["policy"]
    sync = value["sync"]

    def onoff(flag: bool) -> str:
        return "on" if flag else "off"

    lines = [
        f"Agent memory: {onoff(policy_['enabled'])} (memory.enabled)"
        + ("" if policy_["enabled"] else " — agents cannot read or write it; facts are kept"),
        f"Compaction capture: {onoff(policy_['enabled'] and policy_['capture'])} (memory.capture)",
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
    peers = sync.get("peers") or []
    lines.append(f"Peers synced: {len(peers)}" + ("" if peers else " (none yet)"))
    usage = value["usage"]
    lines.append(
        f"Stored: {usage['active_tokens']} active tokens · {usage['ops']} history entries · "
        f"{usage.get('storage_bytes', 0)} bytes"
    )
    lines.append(value["limits"])
    return "\n".join(lines)


def render_human(text: str, failed: bool = False, prefix: str = "/memory") -> str:
    """Readable receipt for CLI and slash command; ``--json`` keeps the raw payload."""
    if failed:
        return text
    value = json.loads(text)
    if "results" in value:
        rows = value["results"]
        if not rows:
            return "No matching memories. " + _searched_line(value, prefix)
        lines = [f"{len(rows)} match(es). " + _searched_line(value, prefix)]
        for row in rows:
            lines += _row(row)
        lines.append(f"Next: {prefix} show {rows[0]['scope']}/{rows[0]['id']}{value['context']}")
        return "\n".join(lines)
    if "items" in value:
        items = value["items"]
        lines = []
        if not value.get("agent_memory", True):
            lines.append("Agent memory is OFF (memory.enabled=false): agents cannot use these.")
        lines.append(
            f"{len(items)} shown · {value['active_tokens']} active tokens · "
            f"{value['ops']} history entries. " + _searched_line(value, prefix)
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
        if fact:
            lines += [
                fact["title"],
                fact["body"],
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
        return (
            f"Forgotten {value['reference']} — '{value.get('title', '')}'. The body is removed "
            "here, and the removal propagates to trusted synced peers when memory sync is on. "
            "It cannot be undone; store a new fact instead."
        )
    if value.get("action") == "corrected":
        return (
            f"Corrected {value['reference']} — '{value.get('title', '')}'. New revision "
            f"{value['version']}; the previous one stays in history, and trusted synced "
            "peers receive the correction when memory sync is on."
        )
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
