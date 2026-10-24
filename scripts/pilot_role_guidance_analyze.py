"""Read-only analyzer for the LOH-9 S5 role-guidance pilot transcripts.

The pilot driver lives in the session scratchpad (the design retains pilot
scripts there); this analyzer is committed so a reviewer can re-derive the
numbers from the retained JSONL transcripts without trusting the driver's
own summary. It reads only the files named on the command line, never the
role/team registries, and prints facts with no pass/fail verdict.

Transcript row kinds the pilot writes (see the driver in the PR):
  ``snapshot``      - every agent/team row and its store, at a named point
  ``tool_call``     - a REAL ``agent``/``team`` tool call, its receipt text and
                      ``details`` (``setup: true`` marks pre-scenario baselines)
  ``task_launch``   - a ``task(agent=...)`` decision, resolution-checked
                      against the session composite (``spawned`` is false: the
                      pilot does not spawn children)
  ``user_confirmation`` - the scripted "yes" for the promote scenario

Limits stated in the PR: no production ``op``/``scope`` telemetry exists;
user-surface promotion (TUI/desktop) bypasses agent-tool receipts entirely;
and scripted decisions are not live-model behaviour.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any


def load(path: Path) -> list[dict[str, Any]]:
    """Parse one JSONL transcript; blank lines are skipped."""
    rows: list[dict[str, Any]] = []
    for line in path.read_text().splitlines():
        line = line.strip()
        if line:
            rows.append(json.loads(line))
    return rows


def rows_in(snapshot: dict[str, Any]) -> set[tuple[str, str]]:
    """Every addressable row in a snapshot as ``(kind:name, scope)``."""
    rows = {(f"agent:{a['name']}", a["scope"]) for a in snapshot.get("agents", [])}
    rows |= {(f"team:{t['name']}", t["scope"]) for t in snapshot.get("teams", [])}
    return rows


def scenario_facts(path: Path) -> dict[str, Any]:
    """Derive one scenario's facts from its transcript alone."""
    rows = load(path)
    snapshots = [row for row in rows if row.get("kind") == "snapshot"]
    before = rows_in(snapshots[0]) if snapshots else set()
    after = rows_in(snapshots[-1]) if snapshots else set()
    added = sorted(key for key, scope in after - before if scope == "global")
    removed = sorted(key for key, scope in before - after)
    existing = {key.split(":", 1)[1] for key, _scope in before}

    creates: dict[str, int] = {}
    durable_attempts: list[dict[str, Any]] = []
    launches: list[dict[str, Any]] = []
    for row in rows:
        if row.get("kind") == "tool_call" and not row.get("setup"):
            args = row.get("args") or {}
            details = row.get("details") or {}
            op = args.get("op")
            if op in {"create", "install"}:
                scope = str(details.get("scope", "<none>"))
                creates[scope] = creates.get(scope, 0) + 1
            attempted_durable = (
                args.get("scope") == "global"
                or op == "promote"
                or details.get("scope") == "global"
                or "needs the user's confirmation" in (row.get("text") or "")
            )
            if attempted_durable:
                durable_attempts.append(
                    {"op": op, "name": args.get("name"), "scope": details.get("scope")}
                )
        elif row.get("kind") == "task_launch":
            agent = str((row.get("args") or {}).get("agent") or "")
            launches.append(
                {
                    "agent": agent,
                    "resolved": row.get("resolved") is not None,
                    "pre_existing": agent.split(":", 1)[-1] in existing,
                    "spawned": row.get("spawned"),
                }
            )
    return {
        "transcript": path.name,
        "account_rows_added": added,
        "account_rows_removed": removed,
        "creates_by_details_scope": dict(sorted(creates.items())),
        "launches": launches,
        "durable_write_attempts": durable_attempts,
        "user_confirmation_rows": sum(1 for row in rows if row.get("kind") == "user_confirmation"),
    }


def main(argv: list[str]) -> int:
    """Print one JSON object per transcript path (sorted by name)."""
    paths = [Path(arg) for arg in argv[1:]]
    if not paths:
        print("usage: pilot_role_guidance_analyze.py <transcript.jsonl> [...]", file=sys.stderr)
        return 2
    for path in sorted(paths):
        print(json.dumps(scenario_facts(path), indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv))
