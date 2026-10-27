"""`lop memory triage --scope global`: return misrouted work state to its session.

WHY this exists: until the plain-text ``memory`` tool defaulted to session scope,
every ``memory(op="store")`` wrote the GLOBAL bank, so the one bank every session
shares filled with one session's goals, lanes and dated work logs (2026-10-27: 78
facts, 15,892/16,000 tokens, 25 from a single session) and then refused writes
everywhere. A fact's ``source.session`` names the session that wrote it, so the
natural home of a misrouted fact is known; what is NOT known automatically is
which facts are misrouted. The marker scan (``memory.fact_markers``) is a shape
rule with false negatives, so this command PREVIEWS a plan and a human confirms
it — the same preview/fingerprint/confirm contract as a filter-based forget and
``compact``.

Constraints, each deliberate:

* **Preview is read-only** — the inspect module's ``mode=ro``/``query_only``
  connection, no import, no reindex, no export.
* **Apply never deletes.** It runs ``MemoryStore.move`` (a put parented to the
  source head plus a source tombstone, the shape old peers already understand)
  with ``destination_owner`` set to the fact's SOURCE session, never the caller's.
* **Human path only.** Apply opens an UNBOUND store: ``move`` refuses
  ``destination_owner`` from any identity bound to a session, which every agent
  store is, so the agent tool routes cannot reach it.
* A fact with no marker stays unless the human names it (``--include``); a fact
  whose source session directory is gone stays (its log has nowhere to go).
* Groups apply independently and each ``move`` is all-or-nothing, so one refused
  group (a conflict, a supersession link, a full bank) never half-moves another.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

from local_operator.memory import MemoryError, MemoryIdentity, MemoryStore, fact_markers
from local_operator.memory_inspect import _exportable, _read_only_connection

#: ``MemoryStore.move`` accepts at most this many targets per call.
_MOVE_BATCH = 64
#: Preview title clip: enough to recognise a row; ``lop memory show`` reads it whole.
_TITLE_CHARS = 72


def _fingerprint(rows: list[tuple[str, str, str]]) -> str:
    """Digest of the planned ``(entity, revision, destination)`` set.

    A fact edited, forgotten or newly matched between preview and confirmation
    changes it, and the confirmed run is refused before anything moves.
    """
    digest = hashlib.sha256()
    for entity, revision, destination in sorted(rows):
        digest.update(f"{entity}:{revision}:{destination}\n".encode())
    return digest.hexdigest()[:12]


def _session_name(directory: Path | None) -> str:
    if directory is None:
        return ""
    from local_operator.resume import read_title_state

    state = read_title_state(directory)
    name = state.text if state else ""
    return name if _exportable(name) else ""


def plan(
    root: Path,
    *,
    only: list[str] | None = None,
    include: list[str] | None = None,
) -> dict[str, Any]:
    """The read-only triage plan for the global bank, grouped by source session."""
    store = MemoryStore(root, MemoryIdentity())
    only_set = set(only or [])
    wanted = list(include or [])
    groups: dict[str, dict[str, Any]] = {}
    kept = {"facts": 0, "tokens": 0}
    stranded = {"facts": 0, "tokens": 0}
    total = {"facts": 0, "tokens": 0}
    planned: list[tuple[str, str, str]] = []
    matched_includes: set[str] = set()
    if store.path.exists():
        with _read_only_connection(store.path) as db:
            rows = db.execute(
                "SELECT entity_id,op_id,fact,tokens FROM current WHERE scope='global' "
                "ORDER BY entity_id"
            ).fetchall()
            for row in rows:
                entity, revision, tokens = row["entity_id"], row["op_id"], row["tokens"]
                fact = json.loads(row["fact"])
                total["facts"] += 1
                total["tokens"] += tokens
                markers = fact_markers(fact)
                named = [ref for ref in wanted if len(ref) >= 8 and entity.startswith(ref)]
                matched_includes.update(named)
                if not markers and not named:
                    kept["facts"] += 1
                    kept["tokens"] += tokens
                    continue
                source = (fact.get("source") or {}).get("session") or ""
                if only_set and source not in only_set:
                    kept["facts"] += 1
                    kept["tokens"] += tokens
                    continue
                directory = store._session_directory(source, db) if source else None
                if directory is None:
                    stranded["facts"] += 1
                    stranded["tokens"] += tokens
                    continue
                group = groups.setdefault(
                    source,
                    {
                        "session": source,
                        "name": _session_name(directory),
                        "destination": f"session:{source}",
                        "facts": [],
                        "tokens": 0,
                        "markers": [],
                    },
                )
                title = fact.get("title")
                group["facts"].append(
                    {
                        "id": entity,
                        "revision": revision,
                        "tokens": tokens,
                        "markers": markers or ["included"],
                        "title": (
                            str(title)[:_TITLE_CHARS] if _exportable(title) else "[withheld]"
                        ),
                    }
                )
                group["tokens"] += tokens
                for name in markers or ["included"]:
                    if name not in group["markers"]:
                        group["markers"].append(name)
                planned.append((entity, revision, source))
    unmatched = [ref for ref in wanted if ref not in matched_includes]
    if unmatched:
        raise MemoryError(
            f"--include names {len(unmatched)} ID(s) that match no global fact (a prefix needs "
            "at least 8 characters). Nothing was moved."
        )
    ordered = sorted(groups.values(), key=lambda item: (-item["tokens"], item["session"]))
    moving = sum(len(group["facts"]) for group in ordered)
    moving_tokens = sum(group["tokens"] for group in ordered)
    return {
        "action": "triage-preview",
        "scope": "global",
        "groups": ordered,
        "moving": {"facts": moving, "tokens": moving_tokens},
        "kept": kept,
        "stranded": stranded,
        "total": total,
        "cap_tokens": MemoryStore.token_cap("global"),
        "fingerprint": _fingerprint(planned),
        # Echoed so the printed apply command reproduces THIS selection exactly.
        "only": sorted(only_set),
        "include": wanted,
        "read_only": True,
    }


def apply(
    root: Path,
    *,
    expect: str,
    only: list[str] | None = None,
    include: list[str] | None = None,
) -> dict[str, Any]:
    """Move the previewed plan, refusing a stale fingerprint before anything moves."""
    current = plan(root, only=only, include=include)
    if expect != current["fingerprint"]:
        raise MemoryError(
            "The global bank changed since the preview you confirmed (fingerprint mismatch). "
            "Nothing was moved. Run the preview again and confirm its new fingerprint."
        )
    # Unbound on purpose: ``move`` honours ``destination_owner`` only from an
    # identity bound to no session (see the module docstring). A fact edited in the
    # instant between this re-plan and its group's move moves at its new revision:
    # ``move`` takes no per-fact revision for a batch, and the fingerprint above is
    # the same set-level guard `compact` gives. It still never deletes a fact.
    store = MemoryStore(root, MemoryIdentity())
    results: list[dict[str, Any]] = []
    for group in current["groups"]:
        ids = [f"global/{fact['id']}" for fact in group["facts"]]
        moved: list[dict[str, Any]] = []
        refused = ""
        for start in range(0, len(ids), _MOVE_BATCH):
            try:
                receipt = store.move(
                    ids[start : start + _MOVE_BATCH],
                    "session",
                    scope="global",
                    destination_owner=group["session"],
                )
            except MemoryError as exc:
                # The store's refusals are value-free by construction; the group
                # stays in place and the next group still runs.
                refused = str(exc)
                break
            moved.extend(receipt["moved"])
        results.append(
            {
                "session": group["session"],
                "name": group["name"],
                "moved": len(moved),
                "tokens": sum(row["tokens"] for row in moved),
                "planned": len(ids),
                "refused": refused,
            }
        )
    with _read_only_connection(store.path) as db:
        used = db.execute(
            "SELECT coalesce(sum(tokens),0) FROM current WHERE scope='global'"
        ).fetchone()[0]
    return {
        "action": "triaged",
        "scope": "global",
        "groups": results,
        "global_tokens": used,
        "cap_tokens": current["cap_tokens"],
    }


def render(value: dict[str, Any], prefix: str) -> list[str]:
    """Human lines for a preview or an applied receipt (``--json`` keeps the payload)."""
    cap = value["cap_tokens"]
    if value["action"] == "triaged":
        lines = []
        for group in value["groups"]:
            label = group["session"] + (f" ({group['name']})" if group["name"] else "")
            if group["refused"]:
                lines.append(
                    f"{label}: moved {group['moved']}/{group['planned']} — {group['refused']}"
                )
            else:
                lines.append(f"{label}: moved {group['moved']} fact(s), {group['tokens']:,} tokens")
        if not lines:
            lines.append("Nothing to move.")
        lines.append(f"Global bank now {value['global_tokens']:,}/{cap:,} tokens.")
        return lines
    total, moving = value["total"], value["moving"]
    lines = [
        f"Global bank: {total['facts']} facts, {total['tokens']:,}/{cap:,} tokens. Read-only "
        "preview — nothing has moved.",
    ]
    for group in value["groups"]:
        label = group["session"] + (f" ({group['name']})" if group["name"] else "")
        lines.append(
            f"→ session:{label}: {len(group['facts'])} fact(s), {group['tokens']:,} tokens "
            f"[{', '.join(group['markers'])}]"
        )
        for fact in group["facts"]:
            lines.append(f"    {fact['id'][:13]}  {fact['tokens']:>4}  {fact['title']}")
    after = total["tokens"] - moving["tokens"]
    lines.append(
        f"Would move {moving['facts']} fact(s), {moving['tokens']:,} tokens; global would hold "
        f"{after:,}/{cap:,} ({100.0 * after / cap:.0f}%)."
    )
    if value["stranded"]["facts"]:
        lines.append(
            f"{value['stranded']['facts']} work-shaped fact(s) stay: their source session "
            "directory is gone (move one by hand: `move global/ID --to session --session LIVE_ID`)."
        )
    lines.append(
        f"{value['kept']['facts']} fact(s) stay in global (no work marker); name one with "
        "--include ID to move it to its source session."
    )
    if moving["facts"]:
        lines.append(
            f"Apply: {prefix} triage --scope global --apply --expect {value['fingerprint']}"
            + "".join(f" --only {group}" for group in value.get("only", []))
            + "".join(f" --include {ref}" for ref in value.get("include", []))
        )
    return lines
