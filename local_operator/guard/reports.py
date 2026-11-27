"""Agent-drafted Guard reports, confirmed only by a person (Guard RFC v0.9 §9.2', slice L2b).

WHAT. An agent that notices a credential exposed in its own work, or a
guardrail that blocked legitimate work, can DRAFT a one-shot report through the
``guard`` tool (``tools/guard_tool.py``). A draft is one row in
``<config_dir>/guard/reports.sqlite3`` holding exactly two facts: the tenant and
a closed-enum ``kind``. Nothing else is accepted, so no secret, path, command or
free text can ride a report (RFC §5.5 "enum-only events"; redaction is not a
filter here, it is the absence of any field that could carry content).

THE MODEL CANNOT SEND ONE (§9.2'). A report leaves this computer only after a
person confirms it at a host surface (``/guard`` in the TUI today; the desktop
approvals panel and the phone in slices U1/M1). :meth:`GuardReports.confirm`
asks :func:`harness.approval.admit_increasing` -- the same policy the consent
accept uses -- and nothing else. The tool can only call :meth:`draft`; it never
holds the capability, so a model, a peer ``send``, ``bridge_decide`` or
``--yolo`` cannot reach ``confirm``.

NOTHING IS SENT IN THIS SLICE. The server half (the pending-action create +
signed atomic consume, agent-server A3) is not built. A confirmed report is
therefore parked locally as ``confirmed`` for the later uploader; this module
imports no network client. That keeps the invariant checkable today: every
path to the wire starts at a ``confirmed`` row, and only ``confirm`` writes one.

ZERO-COST WHERE GUARD IS ABSENT. Construction does no I/O; the file is created
by the first :meth:`draft`, which the tool only offers where an org has turned
Guard on for this member (``guard.surface.guard_available``).
"""

from __future__ import annotations

import os
import sqlite3
import time
import uuid
from dataclasses import dataclass
from pathlib import Path
from typing import Final

GUARD_REPORTS_SCHEMA_VERSION: Final = 1

#: The closed report vocabulary. ``credential_exposure`` and
#: ``excessive_guardrail`` are the two the operator asked agents to self-report;
#: ``other`` exists so a person can still flag something without the agent
#: inventing a free-text kind. Extending this set is a server contract change
#: (§10' releases per-``kind`` counts), never a client-side convenience.
REPORT_KINDS: Final = ("credential_exposure", "excessive_guardrail", "other")

DRAFTED: Final = "drafted"
CONFIRMED: Final = "confirmed"
DISCARDED: Final = "discarded"

#: Drafts the model may hold open per tenant. A model looping on ``report_draft``
#: must not bury the member's ``/guard`` screen; the server's own quota is 3 per
#: member per kind per day (§9.2'), so one open draft per kind is all a person
#: could usefully confirm. A repeat draft of an open kind returns the existing row.
_BUSY_TIMEOUT_S: Final = 5.0

_DDL: Final = (
    "CREATE TABLE drafts (id TEXT PRIMARY KEY, tenant TEXT NOT NULL, kind TEXT NOT NULL,"
    " created_at REAL NOT NULL, state TEXT NOT NULL, decided_at REAL, decided_via TEXT)",
)


class GuardReportRefused(Exception):
    """A report operation was refused; nothing was drafted, confirmed or sent."""

    def __init__(self, code: str, message: str) -> None:
        super().__init__(message)
        self.code = code


@dataclass(frozen=True)
class ReportDraft:
    id: str
    tenant: str
    kind: str
    created_at: float
    state: str
    decided_via: str | None = None


def guard_reports_path(config_dir: Path | str) -> Path:
    return Path(config_dir) / "guard" / "reports.sqlite3"


class GuardReports:
    """Draft (tool), confirm / discard (host only) for one config root."""

    def __init__(self, config_dir: Path | str) -> None:
        self._path = guard_reports_path(config_dir)

    @property
    def path(self) -> Path:
        return self._path

    def exists(self) -> bool:
        return self._path.is_file()

    def _connect(self) -> sqlite3.Connection:
        # Same house rules as guard/consent.py: 0700 dir, 0600 file, an unknown
        # schema generation is refused rather than silently re-created.
        self._path.parent.mkdir(parents=True, exist_ok=True)
        os.chmod(self._path.parent, 0o700)
        existed = self._path.exists() and self._path.stat().st_size > 0
        conn = sqlite3.connect(self._path, timeout=_BUSY_TIMEOUT_S, isolation_level=None)
        conn.row_factory = sqlite3.Row
        try:
            os.chmod(self._path, 0o600)
            version = conn.execute("PRAGMA user_version").fetchone()[0]
            if existed and version != GUARD_REPORTS_SCHEMA_VERSION:
                raise GuardReportRefused(
                    "report_store_damaged", "the Guard report store has an unknown schema"
                )
            if not version:
                conn.execute("BEGIN IMMEDIATE")
                for statement in _DDL:
                    conn.execute(statement)
                conn.execute(f"PRAGMA user_version = {GUARD_REPORTS_SCHEMA_VERSION}")
                conn.execute("COMMIT")
        except BaseException:
            conn.close()
            raise
        return conn

    # -- the tool's one verb ------------------------------------------------

    def draft(self, tenant: str, kind: str) -> tuple[ReportDraft, bool]:
        """``(draft, created)``. An open draft of the same kind is returned, not duplicated."""

        if kind not in REPORT_KINDS:
            raise GuardReportRefused(
                "report_kind_unknown", f"kind must be one of {', '.join(REPORT_KINDS)}"
            )
        if not tenant:
            raise GuardReportRefused("report_tenant_required", "no Guard organization")
        conn = self._connect()
        try:
            conn.execute("BEGIN IMMEDIATE")
            row = conn.execute(
                "SELECT * FROM drafts WHERE tenant = ? AND kind = ? AND state = ?",
                (tenant, kind, DRAFTED),
            ).fetchone()
            if row is not None:
                conn.execute("COMMIT")
                return _draft(row), False
            draft = ReportDraft(
                id=f"grd_{uuid.uuid4().hex[:16]}",
                tenant=tenant,
                kind=kind,
                created_at=time.time(),
                state=DRAFTED,
            )
            conn.execute(
                "INSERT INTO drafts (id, tenant, kind, created_at, state) VALUES (?, ?, ?, ?, ?)",
                (draft.id, draft.tenant, draft.kind, draft.created_at, draft.state),
            )
            conn.execute("COMMIT")
            return draft, True
        finally:
            conn.close()

    # -- reads ----------------------------------------------------------------

    def drafts(self, *, state: str = DRAFTED) -> list[ReportDraft]:
        if not self.exists():
            return []
        conn = self._connect()
        try:
            rows = conn.execute(
                "SELECT * FROM drafts WHERE state = ? ORDER BY created_at", (state,)
            ).fetchall()
        finally:
            conn.close()
        return [_draft(row) for row in rows]

    def get(self, draft_id: str) -> ReportDraft | None:
        if not self.exists():
            return None
        conn = self._connect()
        try:
            row = conn.execute("SELECT * FROM drafts WHERE id = ?", (draft_id,)).fetchone()
        finally:
            conn.close()
        return None if row is None else _draft(row)

    # -- host-only answers ------------------------------------------------------

    def confirm(
        self, draft_id: str, *, capability: bool, signature: bool | None, surface: str
    ) -> ReportDraft:
        """Mark a draft ``confirmed`` iff the answer is ADMITTED (§9.2').

        A refusal changes nothing: the draft stays ``drafted`` for the person.
        """

        from local_operator.harness.approval import admit_increasing

        if not admit_increasing(capability=capability, signature=signature):
            raise GuardReportRefused(
                "report_answer_not_admitted",
                "a Guard report is sent only when you confirm it in /guard",
            )
        return self._settle(draft_id, CONFIRMED, surface)

    def discard(self, draft_id: str, *, surface: str) -> ReportDraft:
        """Discard is the safe direction, so it is ordinary (no admission)."""

        return self._settle(draft_id, DISCARDED, surface)

    def _settle(self, draft_id: str, state: str, surface: str) -> ReportDraft:
        conn = self._connect()
        try:
            conn.execute("BEGIN IMMEDIATE")
            changed = conn.execute(
                "UPDATE drafts SET state = ?, decided_at = ?, decided_via = ?"
                " WHERE id = ? AND state = ?",
                (state, time.time(), surface, draft_id, DRAFTED),
            ).rowcount
            row = conn.execute("SELECT * FROM drafts WHERE id = ?", (draft_id,)).fetchone()
            conn.execute("COMMIT")
        finally:
            conn.close()
        if row is None:
            raise GuardReportRefused("report_not_found", "no such Guard report draft")
        if not changed:
            raise GuardReportRefused("report_already_answered", "this report was already answered")
        return _draft(row)


def _draft(row: sqlite3.Row) -> ReportDraft:
    return ReportDraft(
        id=row["id"],
        tenant=row["tenant"],
        kind=row["kind"],
        created_at=float(row["created_at"]),
        state=row["state"],
        decided_via=row["decided_via"],
    )


__all__ = [
    "CONFIRMED",
    "DISCARDED",
    "DRAFTED",
    "GuardReportRefused",
    "GuardReports",
    "REPORT_KINDS",
    "ReportDraft",
    "guard_reports_path",
]
