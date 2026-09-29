"""The owner-side repair notice for an ``interactive_required`` refusal, DERIVED.

WHY DERIVED, NOT STORED (decision memo, next-wave item D). The design's §4.7 row
originally asked for a ``credential_repair`` op that would raise a durable notice
on the owning device. No such op is built, and this module is why it does not
need to be: when the broker refuses a dead MCP login it already writes a
``credential.report {failure: "interactive_required"}`` audit record
(``owner.py``), and when the operator repairs the login the next successful
borrow writes a ``credential.grant`` — so "is a repair open, and for which key"
is a function of two event types already in the audit log. No new event, no new
op and no wire change; ``lop network doctor`` (both its relay path and its local
fallback) and the ``/network`` panel read this ONE derivation, so the surfaces
cannot disagree about whether a notice is open.

THE PREDICATE, stated once: a report with ``failure=interactive_required`` for
key K is OPEN unless a LATER ``credential.grant`` for K exists in the read
window. The window is the audit tail (the live file's last ``REPAIR_TAIL_ROWS``
records, the same read ``lop network log`` uses). The DIRECTION of that bound is
deliberate: a row is only reported while its own report is inside the window,
and any later grant is inside it too — so the bound can drop a notice (and the
next refused borrow writes a fresh report, so an actively-failing key stays
visible), but it can never show a stale one. Reading the compressed history is
not needed for the same reason: an old grant never clears a report newer than
it, and a report older than the whole window no longer names a live complaint.

WHAT THE ROWS NAME, and the invariant they must not blur. The report records the
BORROWER (``detail.sub`` — the device that asked) and the owner writes it for
itself (``detail.act``); the row therefore names the asker beside the key, and
the remedy runs HERE, because an interactive login is the one repair a borrower
cannot perform for the owner (design §4.7) — nothing in this module lets a
remote trigger a login or a browser, it only reads this device's own audit.
"""

from __future__ import annotations

from typing import Any, Iterable, Mapping

from local_operator.network.audit import AuditLog
from local_operator.network.credentials.messages import (
    render_repair_notice,
    repair_command,
)

#: How many tail records the derivation reads. Bounded WORK by design — the tail
#: is capped, so this reads a slice of the live file, never the gzip history — and
#: the value only trades "a notice can age out of the window" against "read less":
#: at the ~2 rows/min a quiet two-peer link writes, 500 rows is a few hours of
#: history, and a key that keeps failing keeps re-reporting itself into the
#: window.
REPAIR_TAIL_ROWS = 500


def open_reports(records: Iterable[Mapping[str, Any]]) -> list[dict[str, Any]]:
    """The open ``interactive_required`` reports among ``records`` (oldest-first).

    ONE ENTRY PER KEY, carrying the LATEST open report for that key — the newest
    asker is the freshest fact about an unresolved repair. A later grant for the
    same key clears it: the login has been repaired, and the next borrow will
    succeed whether or not the asker ever retried (per-key, not per-borrower, for
    exactly that reason).
    """
    open_for: dict[str, dict[str, Any]] = {}
    for record in records:
        event = str(record.get("event") or "")
        detail = record.get("detail")
        if event not in ("credential.report", "credential.grant"):
            continue
        if not isinstance(detail, Mapping):
            continue
        key = str(detail.get("credential_key") or "")
        if not key:
            continue
        if event == "credential.grant":
            open_for.pop(key, None)
            continue
        if str(detail.get("failure") or "") == "interactive_required":
            open_for[key] = dict(record)
    return list(open_for.values())


def repair_checks(
    record: Any, *, log: AuditLog | None = None, limit: int = REPAIR_TAIL_ROWS
) -> list[dict[str, Any]]:
    """The ``credential_repair`` check rows for ONE network's open reports.

    ``record`` is the ``NetworkRecord`` the caller is already iterating (the relay's
    doctor and the CLI's local fallback both hold it): reports are matched to it by
    ``network_id``, and the borrower's display name is resolved through its member
    table — a name is never load-bearing, an unresolvable device prints its id.

    The row's shape follows the ``membership`` check beside it: ``ok: false`` with a
    human ``detail`` and ``remedies``. The credential's name rides as
    ``credential_name``, NOT ``key``: the agent tool drops any field whose NAME
    carries a secret marker, and "key" is one — the same spelling the credentials
    listing had to adopt (``network/cli.py``), so this row survives the tool's
    scrubber with the one fact it exists to convey.
    """
    notices = [
        report
        for report in open_reports((log or AuditLog()).tail(limit))
        if str(report.get("network_id") or "") == str(record.network_id)
    ]
    rows: list[dict[str, Any]] = []
    for report in notices:
        detail = report.get("detail") or {}
        key = str(detail.get("credential_key") or "")
        borrower = str(detail.get("sub") or "")
        name = _member_name(record, borrower)
        rows.append(
            {
                "check": "credential_repair",
                "network_id": str(record.network_id),
                "ok": False,
                "credential_name": key,
                "device_id": borrower,
                "device_name": name,
                # The wire diagnostic's own wording, recomposed HERE for this
                # device's operator: the sender's name is the borrower's, and
                # "here" is the device the report and the login live on.
                "detail": render_repair_notice(name or borrower, key),
                "remedies": [f"run `{repair_command(key)}` on this device"],
            }
        )
    return rows


def _member_name(record: Any, device_id: str) -> str:
    member = record.member(device_id)
    return str(member.name) if member is not None else ""
