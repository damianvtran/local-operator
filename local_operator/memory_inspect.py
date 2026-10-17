"""Read-only inspection and re-scope PREVIEW over the existing memory store.

Why this module exists: agents sometimes store session-scoped work in the
GLOBAL bank. The operator needs to see that mess and decide what to do about
it. Deciding is a human act, so this surface only *reports*: it lists entries
across every authorized scope with provenance, withholds anything the existing
privacy guard would refuse, and shows a proposed destination beside each entry.

It is READ-ONLY by construction, not by convention:

* it never calls ``MemoryStore.put``/``apply_ops``/``import_session``/``resolve``
  — those are the store's only mutating entry points;
* it opens the database with ``mode=ro`` and ``PRAGMA query_only=ON``, so the
  connection itself cannot write, checkpoint or create a store. A missing
  database is reported as empty, never created;
* it does not reindex, migrate, tombstone or export anything.

There is deliberately NO ``--apply`` flag, no hidden write path and no
"migrate" verb. A future reader who is tempted to add one is adding a data
migration to a command whose whole contract is that a human can look without
changing anything; do that as a separate, explicitly reviewed surface.

Bodies and titles are withheld when ``local_operator.memory.guard`` refuses
them, using the SAME rule the write path applies — including the stricter
``opaque=True`` pass ``validate_fact`` runs on title/body. Without that pass
this surface would print a value the write path refused, which is the one
thing a listing must never do. Such a row is already in the store (an
arbitrary import can put it there), but re-exporting it through a listing
would defeat the rule the guard exists to enforce. What remains is the safe
header: ID, scope, status, age, provenance and the preview disposition.
``lop memory show`` is the deliberate per-ID read a human uses after this
preview; it is not a bulk export either.
"""

from __future__ import annotations

import json
import sqlite3
import time
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Iterator

from local_operator.memory import (
    FACT_STATUSES,
    MAX_FACT_TIMESTAMP,
    MemoryError,
    MemoryIdentity,
    MemoryStore,
    guard,
    origin_label,
    safe_segment,
    work_progress_markers,
)

#: Stable contract for `--json` consumers. Bump only on a breaking shape change;
#: additive keys keep this value so an agent's parser does not need a version
#: dance for a new column.
SCHEMA = "memory.inspect.v1"

#: Human-readable body preview length. Full bodies are NOT exported: the preview
#: exists to let a person recognise the row, and `lop memory show <ref>` is the
#: deliberate read of one full fact.
PREVIEW_CHARS = 160

DEFAULT_LIMIT = 20
MAX_LIMIT = 50

#: The page's ordering key: the fact's own timestamp, read from its JSON.
#: ``json_valid`` guards the expression so ONE corrupt imported row sorts last
#: instead of failing the whole listing with a SQLite "malformed JSON" error.
#: It does NOT make the row skippable: ``json.loads`` in :func:`_entry` raises
#: on that row, and ``run``/``main`` catch at PAGE scope, so a corrupt row
#: refuses its whole page (fail-closed, leaks nothing). That partial-page
#: semantics is a known, accepted deferral (PR #2561, QA Q7) — an earlier
#: version of this comment claimed the parser skipped the row, which it never
#: did.
_CREATED_AT = (
    "CASE WHEN json_valid(fact) THEN "
    "COALESCE(CAST(json_extract(fact,'$.created_at') AS REAL),0) ELSE 0 END"
)

HELP = (
    "Read-only preview. Nothing listed here is moved, edited or forgotten. A proposal "
    "is a suggestion for a human to act on with memory:// or `lop memory`; it is never "
    "applied by this command. Bodies the privacy guard refuses are withheld."
)


@contextmanager
def _read_only_connection(path: Path) -> Iterator[sqlite3.Connection]:
    """Open the store without any write capability.

    ``query_only=ON`` is belt-and-braces beside ``mode=ro``: it also refuses a
    write that a future edit might attempt through this connection. Opening a
    missing file is impossible by design — the caller checks ``exists`` first,
    so inspection can never initialise an empty store the way the mutable
    connection does.
    """
    # ``immutable=1`` keeps SQLite from creating even the empty ``-wal``/``-shm``
    # sidecars, but it also skips WAL replay. It is therefore used ONLY when no
    # live writer is attached (a WAL-mode writer holds ``-shm``, and pending
    # committed frames live in a non-empty ``-wal``); otherwise ``mode=ro``
    # reads through the WAL so the preview shows committed state. Both forms
    # refuse writes at the file level.
    sidecars = [Path(str(path) + suffix) for suffix in ("-wal", "-shm")]
    quiet = not any(item.exists() for item in sidecars)
    uri = f"{path.resolve().as_uri()}?mode=ro" + ("&immutable=1" if quiet else "")
    db = sqlite3.connect(uri, uri=True, timeout=5)
    try:
        db.row_factory = sqlite3.Row
        db.execute("PRAGMA query_only=ON")
        yield db
    finally:
        db.close()


def _exportable(text: Any, *, opaque: bool = True) -> bool:
    """Whether the existing privacy guard would let this value out.

    Uses the store's own rule rather than a second opinion: a value the write
    path refuses is exactly a value a listing must not print.

    ``opaque`` MUST match the pass the write path runs on the same field.
    ``validate_fact`` applies ``guard(value, opaque=True)`` to title, body,
    subject, tags and aliases — that pass is what enables the long-hex/base64
    rule, because an unnamed 40-hex string is how a leaked API key would be
    kept. Defaulting to it here is what makes this listing no stronger a
    reader than the write path is a writer. Only source locators are written
    through the plain ``guard(dict)`` pass, so only they pass ``opaque=False``
    (see :func:`_provenance`).
    """
    if not isinstance(text, str):
        return False
    try:
        guard(text, opaque=opaque)
    except MemoryError:
        return False
    return True


def _provenance(fact: dict[str, Any]) -> tuple[dict[str, Any], bool]:
    """Safe provenance fields; refused values are dropped, never echoed.

    Every string here goes through the same plain guard the write path applied
    to the fact. That is one rule for all six fields, because each one reaches
    the human page and ``--json``: ``author`` and ``tool`` are free text,
    ``session`` is validated as a safe segment, and ``uri``/``entry`` can carry
    sensitive paths. A field that happens to be legitimate passes; nothing is
    exempted by name.
    """
    source = fact.get("source")
    if not isinstance(source, dict):
        return {}, False
    kept: dict[str, Any] = {}
    redacted = False
    for key in ("kind", "author", "session", "tool", "uri", "entry"):
        value = source.get(key)
        if not isinstance(value, str) or not value:
            continue
        # ``opaque=False`` matches the write path exactly: ``validate_fact``
        # runs its opaque title/body pass over free-text fields only, and
        # guards the WHOLE fact — ``source`` included — through the plain
        # recursive ``guard(dict)``. Every provenance string therefore gets
        # that same plain pass, with no field exempt: an earlier version
        # trusted ``author`` (free text) and echoed a card number from an
        # imported row. ``kind`` is a closed vocabulary and ``session`` is
        # ``safe_segment``'d + guarded at write, so both pass unchanged — that
        # is verified in the tests, not assumed.
        if _exportable(value, opaque=False):
            kept[key] = value
        else:
            redacted = True
    return kept, redacted


def _disposition(
    scope: str,
    markers: list[str],
    provenance: dict[str, Any],
    team_id: str,
    *,
    withheld: bool = False,
) -> dict[str, Any]:
    """The PREVIEW proposal for one entry. Descriptive only — nothing consumes it.

    The vocabulary is closed on purpose (keep / session-scoped / team /
    review-work-scope) so a pager, a human and an agent all read the same four
    words. ``applied`` is emitted as a literal ``False`` on every row: a caller
    that tries to treat this as a plan receipt has to ignore an explicit field,
    not merely the prose around it.
    """
    if scope != "global":
        return {
            "proposal": "keep",
            "destination": f"{scope} bank",
            "reason": f"Already in the {scope} bank; re-scoping is not proposed.",
            "applied": False,
        }
    if not markers:
        # A withheld row must not have its proposal read as a judgement about
        # content the page refuses to show; say the marker scan is all that was
        # consulted, so the detail lines up with the redaction promise.
        reason = (
            "Content withheld by the privacy guard; no work/lane/progress marker in the "
            "stored text."
            if withheld
            else "No work/lane/progress marker; a durable global fact is plausible."
        )
        return {
            "proposal": "keep",
            "destination": "global",
            "reason": reason,
            "applied": False,
        }
    marker_text = ", ".join(markers)
    session = provenance.get("session")
    if session:
        return {
            "proposal": "session-scoped",
            "destination": f"session/{session}",
            "reason": (
                f"Work-shaped content ({marker_text}) in the global bank, and the recorded "
                "source session is the likely home."
            ),
            "applied": False,
        }
    if team_id:
        return {
            "proposal": "team",
            "destination": f"team/{team_id}",
            "reason": (
                f"Work-shaped content ({marker_text}) in the global bank, no source session "
                "recorded; this inspector is bound to a team, but ownership must be "
                "confirmed before anything is shared with it."
            ),
            "applied": False,
        }
    return {
        "proposal": "review-work-scope",
        "destination": "needs a human decision",
        "reason": (
            f"Work-shaped content ({marker_text}) in the global bank with no recorded "
            "source session; the destination needs a human, not a guess."
        ),
        "applied": False,
    }


@dataclass(frozen=True, slots=True)
class _Row:
    """One row's raw material, BEFORE any field gate has run.

    A gate may read only fields declared here, and :func:`_project` is the ONLY
    builder of an exported row. Together those two facts are the structural
    guarantee: a stored value has no route to the human page or ``--json``
    except through a gate in :data:`ENTRY_GATES`.
    """

    entity_id: Any
    scope: Any
    op_id: Any
    revisions: Any
    fact: dict[str, Any]
    now: float
    provenance: dict[str, Any]
    provenance_redacted: bool
    markers: list[str]
    disposition: dict[str, Any]


def _gate_segment(value: Any) -> str | None:
    """A store identifier or scope: shape-checked, never content-guarded.

    ``structural=True`` is the writer's own exemption for minted identifiers
    (see ``memory.safe_segment``): a legitimate hex id can trip the guard's
    card/IBAN shapes by chance, and guard-refusing it would blank every
    install rather than withhold one suspicious row.
    """
    try:
        return safe_segment(value, structural=True)
    except MemoryError:
        return None


def _gate_timestamp(value: Any) -> float | None:
    """A freshness timestamp, by the write path's own rule.

    ``validate_fact`` accepts ``type(x) in (int, float)`` inside
    ``0..MAX_FACT_TIMESTAMP`` (so a bool is refused) and nothing else; a value
    it refuses is exactly a value this reader must not repeat.
    """
    if type(value) not in (int, float) or not 0 <= value <= MAX_FACT_TIMESTAMP:
        return None
    return value


def _gate_id(raw: _Row) -> str | None:
    return _gate_segment(raw.entity_id)


def _gate_reference(raw: _Row) -> str | None:
    scope = _gate_segment(raw.scope)
    entity = _gate_segment(raw.entity_id)
    return f"{scope}/{entity}" if scope and entity else None


def _gate_scope(raw: _Row) -> str | None:
    return _gate_segment(raw.scope)


def _gate_status(raw: _Row) -> str:
    """The closed vocabulary ``validate_fact`` enforces.

    "unknown" is the existing fallback for a missing status and is the honest
    answer for a value this reader refuses to repeat.
    """
    value = raw.fact.get("status")
    return value if isinstance(value, str) and value in FACT_STATUSES else "unknown"


def _gate_origin(raw: _Row) -> str:
    # The store's own label over the GATED status, so an imported status cannot
    # select a label the write path would never produce.
    return origin_label({"status": _gate_status(raw)})


def _gate_created_at(raw: _Row) -> float | None:
    return _gate_timestamp(raw.fact.get("created_at"))


def _gate_age(raw: _Row) -> float | None:
    created = _gate_timestamp(raw.fact.get("created_at"))
    return None if created is None else max(0.0, raw.now - float(created))


def _gate_verified_at(raw: _Row) -> float | None:
    return _gate_timestamp(raw.fact.get("verified_at"))


def _gate_expires_at(raw: _Row) -> float | None:
    return _gate_timestamp(raw.fact.get("expires_at"))


def _gate_revision(raw: _Row) -> str | None:
    return _gate_segment(raw.op_id)


def _gate_revisions(raw: _Row) -> int:
    try:
        return max(0, int(raw.revisions))
    except (TypeError, ValueError):
        return 0


def _gate_provenance(raw: _Row) -> dict[str, Any]:
    # Already gated field by field in ``_provenance``; copied so no caller can
    # mutate the shared mapping.
    return dict(raw.provenance)


def _gate_provenance_redacted(raw: _Row) -> bool:
    return bool(raw.provenance_redacted)


def _gate_title(raw: _Row) -> Any:
    title = raw.fact.get("title")
    return title if _exportable(title) else None


def _gate_title_exported(raw: _Row) -> bool:
    return _exportable(raw.fact.get("title"))


def _gate_body_preview(raw: _Row) -> str | None:
    body = raw.fact.get("body")
    return body[:PREVIEW_CHARS] if _exportable(body) and isinstance(body, str) else None


def _gate_body_truncated(raw: _Row) -> bool:
    body = raw.fact.get("body")
    return bool(_exportable(body) and isinstance(body, str) and len(body) > PREVIEW_CHARS)


def _gate_body_exported(raw: _Row) -> bool:
    # Constant on purpose: the preview is the only body bytes this surface ever
    # emits, and a caller must not read this field as permission to fetch more.
    return False


def _gate_redacted(raw: _Row) -> bool:
    return not (_exportable(raw.fact.get("title")) and _exportable(raw.fact.get("body")))


def _gate_scope_fit_candidate(raw: _Row) -> bool:
    return _gate_segment(raw.scope) == "global" and bool(raw.markers)


def _gate_work_markers(raw: _Row) -> list[str]:
    # ``work_progress_markers`` returns only marker NAMES from a closed
    # vocabulary, never the matched text; copying keeps the row independent.
    return list(raw.markers)


def _gate_disposition(raw: _Row) -> dict[str, Any]:
    return dict(raw.disposition)


#: **Every** key an exported entry row may carry, paired with the gate that
#: decides what leaves the process. :func:`_project` builds the row by iterating
#: THIS table, so a key cannot be added inline anywhere else.
#:
#: Why a table instead of a dict literal (three consecutive review rounds, each
#: finding one more ungated field — ``author``/``status``, then
#: ``verified_at``/``expires_at``): gating field by field is a fix that only
#: covers the fields someone thought of. The leak returned because the output
#: shape was written and the gates were added afterwards. Registering a field
#: here is now the same act as gating it, and
#: ``test_every_exported_field_is_gated_by_the_table`` fails on a table entry
#: with no gate. Do NOT inline a key into a dict literal in this module: a value
#: with no gate in front of it is exactly the defect this table exists to end.
ENTRY_GATES: dict[str, Callable[[_Row], Any]] = {
    "id": _gate_id,
    "reference": _gate_reference,
    "scope": _gate_scope,
    "status": _gate_status,
    "origin": _gate_origin,
    "created_at": _gate_created_at,
    "age_seconds": _gate_age,
    "verified_at": _gate_verified_at,
    "expires_at": _gate_expires_at,
    "revision": _gate_revision,
    "revisions": _gate_revisions,
    "provenance": _gate_provenance,
    "provenance_redacted": _gate_provenance_redacted,
    "title": _gate_title,
    "title_exported": _gate_title_exported,
    "body_preview": _gate_body_preview,
    "body_truncated": _gate_body_truncated,
    "body_exported": _gate_body_exported,
    "redacted": _gate_redacted,
    "scope_fit_candidate": _gate_scope_fit_candidate,
    "work_markers": _gate_work_markers,
    "disposition": _gate_disposition,
}


def _project(raw: _Row) -> dict[str, Any]:
    """Build the exported row by ITERATING :data:`ENTRY_GATES`.

    Never list keys here. The dict comprehension is the whole point: there is no
    statement in this function that can emit a value the table does not name.
    """
    return {name: gate(raw) for name, gate in ENTRY_GATES.items()}


def _entry(row: sqlite3.Row, now: float, team_id: str) -> dict[str, Any]:
    fact = json.loads(row["fact"])
    scope = str(row["scope"])
    title = fact.get("title")
    body = fact.get("body")
    provenance, provenance_redacted = _provenance(fact)
    # Same field set as ``scope_fit`` on the write path: a row this listing
    # calls a candidate is exactly a row a new global write would refuse.
    haystack = "\n".join(str(fact.get(key) or "") for key in ("title", "body", "subject"))
    for key in ("tags", "aliases"):
        value = fact.get(key) or []
        if isinstance(value, list):
            haystack += "\n" + "\n".join(str(item) for item in value)
    markers = work_progress_markers(haystack)
    withheld = not (_exportable(title) and _exportable(body))
    return _project(
        _Row(
            entity_id=row["entity_id"],
            scope=scope,
            op_id=row["op_id"],
            revisions=row["revisions"],
            fact=fact,
            now=now,
            provenance=provenance,
            provenance_redacted=provenance_redacted,
            markers=markers,
            disposition=_disposition(scope, markers, provenance, team_id, withheld=withheld),
        )
    )


def inspect_entries(
    root: Path,
    identity: MemoryIdentity,
    *,
    scope: str | None = None,
    limit: int = DEFAULT_LIMIT,
    cursor: str | None = None,
    now: float | None = None,
) -> dict[str, Any]:
    """List authorized entries and their preview dispositions; change nothing.

    The page is ordered newest-first by the fact's OWN ``created_at`` (entry ID
    breaks ties), not by entity ID. Entity IDs are time-ordered only for rows
    this build wrote; an imported/legacy row carries a hash-shaped ID and would
    otherwise sort to the top regardless of age — measured on a 25-entry store
    where three 3h/1d/9d-old rows sat above twenty rows dated minutes ago. The
    cursor remains the entry ID a person was given, so the printed ``More:``
    command is unchanged; its ordering key is looked up under the same
    authorization predicate as the page itself.
    """
    if not 1 <= limit <= MAX_LIMIT:
        raise MemoryError(f"Inspection limit must be 1–{MAX_LIMIT}.")
    if cursor:
        # Same structural rule inventory applies; the cursor is spliced into SQL.
        safe_segment(cursor)
    store = MemoryStore(root, identity)
    current = time.time() if now is None else now
    authorized = [name for name, _ in identity.scopes()]
    report: dict[str, Any] = {
        "schema": SCHEMA,
        "read_only": True,
        "store": {"path": str(store.path), "exists": store.path.exists()},
        "generated_at": current,
        "authorized_scopes": authorized,
        "scope": scope,
        "limit": limit,
        "cursor": cursor or None,
        "counts": {"returned": 0, "total": 0, "by_scope": {}},
        "entries": [],
        "has_more": False,
        "next_cursor": None,
        #: True when the supplied cursor no longer names an authorized row (the
        #: entry was forgotten or re-scoped since it was printed). The page is
        #: then empty BY THE CURSOR, not by the store, so counts stay real.
        "cursor_stale": False,
        "help": HELP,
    }
    if not store.path.exists():
        return report
    # The store's single authorization rule (authorized owner pairs per scope).
    # Reused rather than restated: two surfaces answering "who may read this?"
    # differently is the defect this shares the rule to avoid.
    where, values = store._where(scope)
    try:
        with _read_only_connection(store.path) as db:
            has_current = db.execute(
                "SELECT 1 FROM sqlite_master WHERE type='table' AND name='current'"
            ).fetchone()
            if not has_current:
                return report
            page_where = where
            page_values: list[Any] = list(values)
            cursor_stale = False
            if cursor:
                # The cursor's ordering key is read under the same predicate as
                # the page: a cursor naming another owner's row yields an empty
                # page rather than an ordering hint about a scope we cannot read.
                anchor = db.execute(
                    f"SELECT {_CREATED_AT} FROM current WHERE {where} AND entity_id=?",
                    [*values, cursor],
                ).fetchone()
                if anchor is None:
                    # Do NOT return here: the totals query below is
                    # cursor-independent, so returning early left ``counts`` at
                    # zero and the page claimed "Nothing retained in these
                    # scopes" while the store held entries. An auditing tool
                    # must never say that because a cursor aged out.
                    cursor_stale = True
                else:
                    page_where = f"{where} AND ({_CREATED_AT},entity_id)<(?,?)"
                    page_values = [*values, float(anchor[0]), cursor]
            rows = (
                []
                if cursor_stale
                else db.execute(
                    "SELECT entity_id,scope,op_id,fact,"
                    "(SELECT count(*) FROM ops o WHERE o.entity_id=current.entity_id) "
                    f"AS revisions FROM current WHERE {page_where} "
                    f"ORDER BY {_CREATED_AT} DESC,entity_id DESC LIMIT ?",
                    # One extra row answers "is there more?" truthfully instead of
                    # treating a full page as proof of a next one.
                    [*page_values, limit + 1],
                ).fetchall()
            )
            totals = db.execute(
                f"SELECT scope,count(*) FROM current WHERE {where} GROUP BY scope", values
            ).fetchall()
    except sqlite3.Error as exc:
        raise MemoryError(
            "Memory store could not be read read-only (SQLite "
            + type(exc).__name__
            + "); no memory was changed."
        ) from exc
    has_more = len(rows) > limit
    page = rows[:limit]
    report["cursor_stale"] = cursor_stale
    by_scope = {str(row[0]): int(row[1]) for row in totals}
    report["counts"]["by_scope"] = by_scope
    report["counts"]["total"] = sum(by_scope.values())
    report["entries"] = [_entry(row, current, identity.team_id) for row in page]
    report["counts"]["returned"] = len(report["entries"])
    # Counted on THIS page on purpose: the header's job is "which of the rows in
    # front of me need a decision?". A store-wide total would be a number the
    # reader cannot act on without paging to it.
    report["counts"]["suggestions"] = sum(
        1 for entry in report["entries"] if entry["disposition"]["proposal"] != "keep"
    )
    if has_more:
        report["has_more"] = True
        # Display order IS the paging order, so the last row shown is the key
        # the next page continues strictly after.
        report["next_cursor"] = page[-1]["entity_id"]
    return report
