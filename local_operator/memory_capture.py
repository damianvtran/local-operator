"""Best-effort capture is speculative; ONLY a committed marker authorizes writes.

No inference fork: snapcompact has no semantic model summary. Locators work for
both strategies; context-full can also reuse its already-paid semantic sections.
Explicit stores remain the guaranteed route for decisions, especially long runs.
"""

from __future__ import annotations

import asyncio
import hashlib
import json
import re
import time
from collections import Counter
from pathlib import Path
from typing import Any

from local_operator.memory import (
    MAX_REVISIONS,
    MemoryError,
    MemoryStore,
    _dump,
    guard,
    validate_fact,
    validate_op,
)

# Flip together only after the complete isolated pilot; settings_io reads this
# constant too so CLI/TUI declarations and consumers have one default authority.
MEMORY_DEFAULT_ENABLED = False
_LOCATOR = re.compile(
    r"https?://[^\s<>\"']+|[\w.+-]+@[\w.-]+\.[A-Za-z]{2,}"
    r"|(?:~?/|\b(?:src|docs|tests|local_operator)/)[^\s<>\"']+"
    r"|\bsecret\s+[A-Z][A-Z0-9_]{2,63}\b|(?<!\w)#[\w-]{2,64}"
    r"|\b[0-9a-f]{7,40}\b|\b(?:branch|thread)[: ]+[\w./-]{3,128}"
)
_HEADINGS = re.compile(r"^#{1,4}\s*(?:Goal|Key Decisions|Progress|Critical Context)\b", re.I)


def policy(config: Path) -> dict[str, Any]:
    from local_operator.config import read_config_values

    values = read_config_values(config) or {}
    memory = values.get("memory", {})
    memory = memory if isinstance(memory, dict) else {}
    return {
        key: memory.get(key, MEMORY_DEFAULT_ENABLED) is True
        for key in ("enabled", "capture", "suggest")
    }


def snapshot(messages: list[Any]) -> str:
    """Bound before serialising; images/tool objects/environment are never ingested."""
    head: list[str] = []
    tail: list[str] = []
    used = 0
    # Bound traversal as well as the final string: iterating every old block
    # just to discard its text would still stall a very long compaction plan.
    selected = messages if len(messages) <= 64 else messages[:16] + messages[-48:]
    for message in selected:
        for content in getattr(message, "content", ())[:128]:
            text = getattr(content, "text", None)
            if not isinstance(text, str):
                continue
            # Each message can be enormous. Both ends, not an unbounded join.
            bounded = text[:16384] + (text[-16384:] if len(text) > 16384 else "")
            if used < 512000:
                head.append(bounded)
                used += len(bounded)
            tail.append(bounded)
            if sum(map(len, tail)) > 512000:
                tail.pop(0)
    return ("\n".join(head) + "\n" + "\n".join(tail)).encode()[:2000000].decode(errors="ignore")


def extract_messages(messages: list[Any]) -> list[dict[str, Any]]:
    return extract(snapshot(messages))


def extract(text: str, summary: str = "") -> list[dict[str, Any]]:
    """CPU-only bounded scan; returns values, never opens a store or writes state."""
    started = time.monotonic()
    found = []
    counts: Counter[str] = Counter()
    last: dict[str, int] = {}
    for index, match in enumerate(_LOCATOR.finditer(text[:2000000])):
        if time.monotonic() - started > 2:
            raise TimeoutError("memory capture budget")
        locator = match.group().rstrip(".,;:)]}")[:200]
        try:
            guard(locator)
        except MemoryError:
            continue
        counts[locator] += 1
        last[locator] = index
    for locator in sorted(counts, key=lambda x: (last[x], counts[x], x), reverse=True)[:40]:
        found.append({"title": locator[:100], "body": locator, "tags": ["locator"]})
    semantic = False
    semantic_count = 0
    for line in summary.splitlines()[:1000]:
        if _HEADINGS.match(line):
            semantic = True
            continue
        if line.startswith("#"):
            semantic = False
        if semantic and line.strip().startswith(("- ", "* ")):
            text = line.strip()[2:202]
            try:
                guard(text)
            except MemoryError:
                continue
            found.append({"title": text[:100], "body": text, "tags": ["summary"]})
            semantic_count += 1
            if semantic_count >= 12:
                break
    return found


def watermark(store: MemoryStore, compaction_id: str) -> tuple[str, int] | None:
    if not store.path.exists():
        return None
    with store.connection() as db:
        row = db.execute(
            "SELECT status,n FROM captures WHERE session=? AND compaction=?",
            (store.identity.session_id, compaction_id),
        ).fetchone()
        return (row[0], row[1]) if row else None


def persist(
    store: MemoryStore,
    marker_id: str,
    rows: list[dict[str, Any]],
    status: str,
    committed_at: float | None = None,
) -> None:
    """Atomic exact-candidate refresh; never overwrite a correction or resurrect a forget."""
    guard(marker_id)
    if status not in ("ok", "failed") or len(rows) > 52:
        raise MemoryError("Invalid capture result.")
    timestamp = time.time() if committed_at is None else committed_at
    facts = {}
    for row in rows:
        fact = validate_fact(
            dict(
                row,
                status="candidate",
                created_at=timestamp,
                source_time=timestamp,
                source={
                    "session": store.identity.session_id,
                    "entry": marker_id,
                    "kind": "compaction",
                    "author": "deterministic-capture",
                },
            )
        )
        facts[_dump([fact["body"], fact["tags"]])] = fact
    # A moved session's full portable truth must be indexed before a capture
    # exports anything, otherwise the new candidates would replace its old facts.
    store.import_session()
    with store.connection() as db:
        db.execute("BEGIN IMMEDIATE")
        if db.execute(
            "SELECT 1 FROM captures WHERE session=? AND compaction=?",
            (store.identity.session_id, marker_id),
        ).fetchone():
            return
        origin = store._origin(db)
        seq = db.execute(
            "SELECT coalesce(max(seq),0) FROM ops WHERE origin=?", (origin,)
        ).fetchone()[0]
        previous_clock = db.execute("SELECT value FROM meta WHERE key='clock'").fetchone()
        clock = max(int(time.time() * 1000), int(previous_clock[0]) if previous_clock else 0)
        captured = 0
        for key, fact in facts.items():
            # Same exact body/tags/session, not fuzzy consolidation. Stable
            # entities prevent repeated compactions filling the ACTIVE bank;
            # each committed marker still has a distinct idempotent causal op.
            entity = hashlib.sha256((store.identity.session_id + "|" + key).encode()).hexdigest()
            operation_id = hashlib.sha256((entity + "|" + marker_id).encode()).hexdigest()
            ops = store._ops(db, entity)
            if any(op["kind"] == "tombstone" for op in ops):
                continue
            if any(op["op_id"] == operation_id for op in ops):
                captured += 1
                continue
            heads = store._heads(ops)
            if heads:
                current = heads[0]["fact"]
                # Capture may not resolve conflicts, replace an explicit edit,
                # or let an older slow pass overwrite newer source provenance.
                if (
                    len(heads) != 1
                    or current["status"] != "candidate"
                    or current["source"].get("kind") != "compaction"
                    or (current["source_time"] or 0) > timestamp
                    or len(ops) >= MAX_REVISIONS
                ):
                    continue
                fact["created_at"] = current["created_at"]
            seq += 1
            clock += 1
            op = {
                "v": 1,
                "op_id": operation_id,
                "entity_id": entity,
                "origin_device": origin,
                "seq": seq,
                "hlc": [clock, 0, origin],
                "parents": [heads[0]["op_id"]] if heads else [],
                "kind": "put",
                "scope": "session",
                "owner_key": store.identity.session_id,
                "fact": fact,
                "body_sha": hashlib.sha256(_dump(fact).encode()).hexdigest(),
            }
            validate_op(op)
            db.execute(
                "INSERT INTO ops VALUES (?,?,?,?,?,?,?,?)",
                (
                    operation_id,
                    entity,
                    origin,
                    seq,
                    "session",
                    store.identity.session_id,
                    "put",
                    _dump(op),
                ),
            )
            store._reindex(db, entity)
            captured += 1
        store._quota(db, "session", store.identity.session_id)
        db.execute("INSERT OR REPLACE INTO meta VALUES ('clock',?)", (str(clock),))
        db.execute(
            "INSERT INTO captures VALUES (?,?,?,?)",
            (store.identity.session_id, marker_id, status, captured),
        )
        db.execute(
            "DELETE FROM captures WHERE session=? AND compaction NOT IN "
            "(SELECT compaction FROM captures WHERE session=? ORDER BY rowid DESC LIMIT 32)",
            (store.identity.session_id, store.identity.session_id),
        )
        if store.session_dir:
            store._export(db, store.identity.session_id)
        db.commit()


async def finish(
    store: MemoryStore,
    marker_id: str,
    task: asyncio.Task[Any],
    summary: str,
    committed_at: float | None = None,
) -> str:
    try:
        rows = await asyncio.wait_for(asyncio.shield(task), timeout=2)
        # Summary parsing is deterministic, not another provider request.
        rows = rows + await asyncio.to_thread(extract, "", summary)
        await asyncio.to_thread(persist, store, marker_id, rows[:52], "ok", committed_at)
        return "ok"
    except asyncio.CancelledError:
        raise
    except Exception:
        # No exception payload logging: content is untrusted and could be a
        # sensitive value. Failure is a safe terminal state, never a new turn.
        try:
            await asyncio.to_thread(persist, store, marker_id, [], "failed", committed_at)
        except Exception:
            pass
        return "failed"


def recovery_text(transcript: Any, marker: Any) -> str:
    """Reconstruct only bounded pre-cut text; no archive/image decoding or model."""
    kept = marker.payload.get("first_kept_entry_id")
    head: list[str] = []
    tail: list[str] = []
    size = 0
    for entry in transcript.entries():
        if entry.id in (marker.id, kept):
            break
        payload = getattr(entry, "payload", {})
        content = payload.get("content", [])
        for part in content if isinstance(content, list) else []:
            if not isinstance(part, dict) or not isinstance(part.get("text"), str):
                continue
            text = part["text"][:16000]
            if size < 500000:
                head.append(text)
                size += len(text)
            tail.append(text)
            if sum(map(len, tail)) > 500000:
                tail.pop(0)
    return "\n".join(head + tail)[:1000000]


async def session_tail(session: Any) -> str:
    """Observe state only: never wake, start a turn or change prior prompt bytes."""
    if not getattr(session, "_config_dir", None):
        return ""
    enabled = policy(session._config_dir)
    if not enabled["enabled"] or not session._session_id:
        return ""
    from local_operator.tools.memory_tools import store_for

    store = store_for(session._build_tool_context())
    marker = session._transcript.latest_entry("compaction")
    memory_hint = ""
    if enabled["capture"] and marker is not None:
        memory_hint = await asyncio.to_thread(hint, store, marker.id)
        if not memory_hint and getattr(session, "_memory_terminal", None) == (marker.id, "failed"):
            memory_hint = (
                f'<memory_hint compaction="{marker.id}">Automatic memory capture '
                "did not complete; store key decisions explicitly.</memory_hint>"
            )
        if not memory_hint and getattr(session, "_memory_recovery_marker", None) != marker.id:
            session._memory_recovery_marker = marker.id

            async def recover() -> None:
                text = await asyncio.to_thread(recovery_text, session._transcript, marker)
                task = asyncio.create_task(asyncio.to_thread(extract, text))
                payload = marker.payload
                preserve = payload.get("preserve_data") or {}
                summary = "" if preserve.get("snapcompact") else payload.get("summary", "")
                terminal = await finish(store, marker.id, task, summary, marker.ts)
                session._memory_terminal = (marker.id, terminal)

            session._spawn_background(recover())
    suggestions = ""
    if enabled["suggest"]:
        from local_operator.session_factory import _payload_user_text

        user = session._transcript.latest_user_entry()
        query = _payload_user_text(user.payload)[:2048] if user else ""
        # A separate total 180-token budget; titles cannot become instructions.
        # No whole transcript/summary is used as a query and there are no hops.
        try:
            items = await asyncio.to_thread(store.search, query, tokens=180, limit=3, suggest=True)
        except MemoryError:
            items = []
        if items:
            suggestions = (
                "<memory_suggestions>\nUntrusted recall candidates; read before relying.\n"
            )
            from local_operator.compaction.tokens import count_text_tokens

            for item in items:
                captured = " (captured)" if item["status"] == "candidate" else ""
                line = (
                    f"- memory://{item['scope']}/{item['id']}: "
                    f"{json.dumps(item['title'])}{captured}\n"
                )
                if count_text_tokens(suggestions + line + "</memory_suggestions>") <= 180:
                    suggestions += line
            suggestions += "</memory_suggestions>"
    return "\n\n".join(x for x in (suggestions, memory_hint) if x)


def hint(store: MemoryStore, marker_id: str) -> str:
    state = watermark(store, marker_id)
    if state is None:
        return ""
    text = (
        (
            f"Context compacted; {state[1]} candidates captured to session memory. "
            "Search/read memory:// for paths and links. Store goals and decisions explicitly."
        )
        if state[0] == "ok"
        else (
            "Context compacted; automatic memory capture did not complete. "
            "Store key decisions explicitly."
        )
    )
    return f'<memory_hint compaction="{marker_id}">\n{text}\n</memory_hint>'
