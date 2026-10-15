"""One human inspection/correction runner for CLI and the TUI slash command.

Explicit scope owners are a human control, never accepted by memory:// tools.
JSON replacement plus a revision keeps correction auditable and race-safe; a
forget additionally requires --yes, so inspection can never destroy user data.
"""

from __future__ import annotations

import argparse
import json
import shlex
import sqlite3
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from local_operator.memory import SCOPES, MemoryError, MemoryIdentity, MemoryStore
from local_operator.memory_capture import MEMORY_DEFAULT_ENABLED, policy
from local_operator.paths import config_dir


class MemoryParser(argparse.ArgumentParser):
    def error(self, message: str) -> None:
        raise MemoryError(
            "Invalid memory command. Use list/show/search/edit/forget/status; "
            "edit needs ID --version REV --json JSON; forget needs ID --version REV --yes."
        )


def configure_parser(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--scope", choices=SCOPES, default="global")
    parser.add_argument("--session", default="")
    parser.add_argument("--team", default="")
    parser.add_argument("--agent", default="")
    parser.add_argument(
        "operation",
        nargs="?",
        choices=("list", "show", "search", "edit", "forget", "status"),
        default="list",
    )
    parser.add_argument("target", nargs="?", default="")
    parser.add_argument("--version", default="")
    parser.add_argument("--json", dest="replacement", default="")
    parser.add_argument("--yes", action="store_true")
    parser.add_argument("--related", action="store_true")


def execute(
    args: argparse.Namespace,
    *,
    config: Path | None = None,
    identity: MemoryIdentity | None = None,
    session_dir: Path | None = None,
) -> dict[str, Any] | list[Any]:
    root = config or config_dir()
    current = identity or MemoryIdentity(args.session, args.team, args.agent)
    directory = session_dir or (
        root / "sessions" / current.session_id if current.session_id else None
    )
    store = MemoryStore(root, current, directory)
    scope = args.scope
    if args.operation == "status":
        from local_operator.config import read_config_values
        from local_operator.network.memory import peer_status

        values = read_config_values(root) or {}
        network = values.get("network", {})
        network = network if isinstance(network, dict) else {}
        mesh = network.get("memory", {})
        mesh = mesh if isinstance(mesh, dict) else {}
        return {
            "policy": policy(root),
            "sync": {
                "enabled": mesh.get("enabled", MEMORY_DEFAULT_ENABLED),
                "scopes": mesh.get("scopes", list(SCOPES)),
                "exclude_devices": mesh.get("exclude_devices", []),
                "trust": ("Existing paired receiver-admin peers with memory-sync-v1 only; "
                          "no new grants."),
                "transport": "incremental all-scope net_memory; no transcript copy required",
                "peers": peer_status(root),
                "retention": "Tombstones/conflicts are retained; bounded saturation refuses new writes, never silently evicts facts.",
            },
            "usage": store.inventory(),
            "limits": ("Shape guards are heuristic; opaque free-text secrets may escape detection. "
                       "Secret NAME references only."),
        }
    store.import_session()
    if args.operation == "list":
        return store.inventory(scope)
    if args.operation == "search":
        return store.search(args.target, scope)
    if not args.target:
        raise MemoryError("An explicit memory ID is required.")
    if args.operation == "show":
        view = store.show(scope, args.target, version=args.version or None)
        if args.related:
            view["related"] = store.related(scope, args.target)
        return view
    if not args.version:
        raise MemoryError("Read the current revision first and pass --version REV.")
    if args.operation == "forget":
        if not args.yes:
            raise MemoryError(
                "Nothing forgotten. Confirm this ID and revision with --yes; "
                "deletion removes its body on trusted synced peers too."
            )
        return store.put(scope, {}, entity=args.target, expected_version=args.version, forget=True)
    if not args.replacement:
        raise MemoryError(
            "Edit needs a complete replacement --json object, never a blind overwrite."
        )
    replacement = json.loads(args.replacement)
    source = replacement.get("source", {}) if isinstance(replacement, dict) else {}
    if isinstance(replacement, dict) and isinstance(source, dict):
        replacement["source"] = dict(source, author="user", kind=source.get("kind", "reported"))
    return store.put(scope, replacement, entity=args.target, expected_version=args.version)


def run(
    text: str = "",
    *,
    config: Path | None = None,
    identity: MemoryIdentity | None = None,
    session_dir: Path | None = None,
) -> tuple[str, bool]:
    parser = MemoryParser(prog="memory", add_help=False)
    configure_parser(parser)
    try:
        args = parser.parse_args(shlex.split(text))
        result = execute(args, config=config, identity=identity, session_dir=session_dir)
        return json.dumps(result, ensure_ascii=False, indent=2), False
    except (MemoryError, ValueError, TypeError, OSError, sqlite3.Error) as exc:
        return (
            str(exc)
            if isinstance(exc, MemoryError)
            else "Memory request refused or store unavailable. No unsafe payload included."
        ), True


def _date(value: Any) -> str:
    return (
        datetime.fromtimestamp(value, timezone.utc).strftime("%Y-%m-%d %H:%M UTC")
        if isinstance(value, (int, float))
        else "not verified/unknown"
    )


def render_human(text: str, failed: bool = False) -> str:
    """Compact slash receipt; the CLI retains its machine-readable JSON output."""
    if failed:
        return text
    value = json.loads(text)
    if isinstance(value, list):
        if not value:
            return "No matching memories. Try specific names, tags or another authorised scope."
        return "\n".join(
            f"{row['scope']}/{row['id']}  {row['title']} [{row['status']}]" for row in value
        )
    if "items" in value:
        items = value["items"]
        lines = [
            f"{len(items)} shown · {value['active_tokens']} active tokens · "
            f"{value['ops']} history ops"
        ]
        if not items:
            lines.append(
                "No retained memories in this scope. Store key goals and decisions explicitly."
            )
        for row in items:
            lines.append(f"{row['scope']}/{row['id']}  {row['title']} [{row['status']}]")
        lines.append("Inspect: /memory --scope SCOPE show ID · Search: /memory search 'words'")
        return "\n".join(lines)
    if "fact" in value:
        fact = value["fact"]
        lines = [
            f"{value['scope']}/{value['id']} [{value['status']}]",
            f"Revision: {value['version']}",
        ]
        if fact:
            lines += [
                fact["title"],
                fact["body"],
                "Source: " + json.dumps(fact["source"], ensure_ascii=False),
                f"Retained: {_date(fact['created_at'])} · Verified: {_date(fact['verified_at'])}",
                f"Source date: {_date(fact['source_time'])} · Expires: {_date(fact['expires_at'])}",
            ]
        lines.append(
            f"Conflicts: {len(value['conflicts'])} shown · "
            f"Revisions: {value['revisions']} · Untrusted data"
        )
        lines.append("Correction needs --version REV --json JSON; forget also needs --yes.")
        return "\n".join(lines)
    if "policy" in value:
        return (
            "Local policy: "
            + json.dumps(value["policy"])
            + "\nTrusted sync: "
            + json.dumps(value["sync"], ensure_ascii=False)
            + "\n"
            + value["limits"]
        )
    return (f"{value.get('scope', '')}/{value.get('id', '')} [{value.get('status', '')}]\n"
            f"Revision: {value.get('version', '')}")


def main(args: argparse.Namespace) -> int:
    try:
        print(json.dumps(execute(args), ensure_ascii=False, indent=2))
        return 0
    except (MemoryError, ValueError, TypeError, OSError, sqlite3.Error) as exc:
        print(
            str(exc)
            if isinstance(exc, MemoryError)
            else "Memory request refused or store unavailable."
        )
        return 1
