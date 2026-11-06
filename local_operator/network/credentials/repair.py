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

TWO OPENING WORDS, ONE PREDICATE. ``interactive_required`` is the refusal the MCP
borrow path writes when the owner's own row holds no usable token. Two other
terminations meant the same thing — "only a fresh sign-in HERE helps" — and wrote
nothing this predicate could see: the owner's report-driven refresh probe finding a
dead grant (a borrowed bearer's 401, ``owner._one_owner_refresh``) and a provider
refresh the IdP refused (``owner._classify_refresh_failure`` → ``grant_invalid``).
Both now write ``failure: "reauth_required"`` (the owner-relogin design, §3.5), and
:data:`REPAIR_FAILURES` is the whole vocabulary: the doctor row, the ``/network``
panel and the inbox item (``credentials/repair_inbox.py``) all read
:func:`repair_notices`, so a new consumer adds no second source of truth.
"""

from __future__ import annotations

from dataclasses import dataclass, field
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


#: The failure words that OPEN a repair: the ones only this device writes, about
#: its own login, when it has found that nothing but a fresh sign-in here can help.
#: ``interactive_required`` is the MCP borrow refusal's; ``reauth_required`` is the
#: one the owner writes when its refresh probe (a borrower's 401) or a provider
#: refresh (``grant_invalid``) proved the grant dead. A borrower's own attribution
#: (``unauthorized``, ``invalid``, ``quota`` …) is evidence, never a verdict, and
#: opens nothing — and ``owner.report`` refuses to record either word verbatim from a
#: peer, so neither can be put into this log by the device that would be named.
REPAIR_FAILURES: frozenset[str] = frozenset({"interactive_required", "reauth_required"})


@dataclass
class _KeyFold:
    """What the window says about ONE key, beside its latest open report.

    Kept apart from ``open_for`` (the dict :func:`open_reports` has always
    returned) so that dict's insertion order — which the doctor rows inherit —
    is exactly what it was before these facts were folded too.
    """

    #: Every device that asked since the last grant, in first-seen order. The
    #: doctor row names the LATEST asker; the inbox item names all of them.
    askers: list[str] = field(default_factory=list)
    #: ``seq`` of the latest ``credential.grant`` for the key in the window, 0 for
    #: none: the episode token the inbox item's key carries (see ``generation``).
    grant_seq: int = 0
    #: Whether any grant for the key is in the window at all — with no open report,
    #: that is positive evidence the login was repaired (or never broken).
    granted: bool = False


def _fold(
    records: Iterable[Mapping[str, Any]],
) -> tuple[dict[str, dict[str, Any]], dict[str, _KeyFold]]:
    """ONE pass over the window: the open reports, and the per-key facts beside them."""
    open_for: dict[str, dict[str, Any]] = {}
    facts: dict[str, _KeyFold] = {}
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
        fold = facts.setdefault(key, _KeyFold())
        if event == "credential.grant":
            open_for.pop(key, None)
            fold.askers = []
            fold.granted = True
            seq = record.get("seq")
            fold.grant_seq = int(seq) if isinstance(seq, int) else fold.grant_seq
            continue
        if str(detail.get("failure") or "") in REPAIR_FAILURES:
            open_for[key] = dict(record)
            asker = str(detail.get("sub") or "")
            if asker and asker not in fold.askers:
                fold.askers.append(asker)
    return open_for, facts


def open_reports(records: Iterable[Mapping[str, Any]]) -> list[dict[str, Any]]:
    """The open repair reports among ``records`` (oldest-first).

    ONE ENTRY PER KEY, carrying the LATEST open report for that key — the newest
    asker is the freshest fact about an unresolved repair. A later grant for the
    same key clears it: the login has been repaired, and the next borrow will
    succeed whether or not the asker ever retried (per-key, not per-borrower, for
    exactly that reason). A report opens only on a :data:`REPAIR_FAILURES` word.
    """
    return list(_fold(records)[0].values())


@dataclass(frozen=True)
class RepairNotice:
    """One open repair, resolved for THIS device's operator — the one derivation.

    The doctor's ``credential_repair`` row (:func:`repair_checks`) and the inbox
    item (``repair_inbox.publish_notices``) are both renderings of this, so they
    cannot disagree about whether a repair is open, for which key, or what to run.
    """

    network_id: str
    #: The placement key (``mcp:<url>`` or a provider name). Carried here, but no
    #: tool-visible payload may name a field ``key`` — see :func:`repair_checks`.
    key: str
    #: The LATEST asker (device id) and its display name, ``""`` when unresolved.
    borrower: str
    borrower_name: str
    #: The server name THIS device resolves for an MCP key, ``""`` otherwise.
    login_name: str
    #: ``(device id, display name)`` for every asker since the last grant.
    askers: tuple[tuple[str, str], ...]
    #: The opening word of the latest report (one of :data:`REPAIR_FAILURES`).
    failure: str
    #: The episode token: ``g<seq>`` of the grant that closed the PREVIOUS episode,
    #: ``g0`` when the window holds none. "Broken → repaired → broken again" is a new
    #: token (a later grant) and so a new inbox item; "still broken" keeps it.
    generation: str

    @property
    def command(self) -> str:
        """The login that repairs it, spelled once (``messages.repair_command``)."""
        return repair_command(self.key, name=self.login_name)

    @property
    def remedy(self) -> str:
        """The doctor row's remedy line — and the inbox item's title."""
        return f"run `{self.command}` on this device"

    def sentence(self, peer_label: str) -> str:
        """The producer's own sentence (``messages.render_repair_notice``) for one asker."""
        return render_repair_notice(peer_label, self.key, name=self.login_name)

    @property
    def detail(self) -> str:
        """The doctor row's ``detail``: the sentence for the LATEST asker."""
        return self.sentence(self.borrower_name or self.borrower)


def repair_notices(record: Any, records: Iterable[Mapping[str, Any]]) -> list[RepairNotice]:
    """The open repairs among ``records`` that belong to ``record``'s network.

    ``record`` is a ``NetworkRecord``: reports are matched to it by ``network_id``,
    and every asker's display name is resolved through its member table — a name
    is never load-bearing, an unresolvable device prints its id.
    """
    open_for, facts = _fold(records)
    notices: list[RepairNotice] = []
    for report in open_for.values():
        if str(report.get("network_id") or "") != str(record.network_id):
            continue
        detail = report.get("detail") or {}
        key = str(detail.get("credential_key") or "")
        borrower = str(detail.get("sub") or "")
        fold = facts.get(key) or _KeyFold()
        notices.append(
            RepairNotice(
                network_id=str(record.network_id),
                key=key,
                borrower=borrower,
                borrower_name=_member_name(record, borrower),
                login_name=_login_name(key),
                askers=tuple((asker, _member_name(record, asker)) for asker in fold.askers),
                failure=str(detail.get("failure") or ""),
                generation=f"g{fold.grant_seq}",
            )
        )
    return notices


def repaired_keys(records: Iterable[Mapping[str, Any]]) -> set[str]:
    """Keys the window shows REPAIRED: a grant in it, and no open report after that.

    The inbox closes an item only on this positive evidence. A key that merely
    aged out of the window is not here: its item stays (the login is still dead
    as far as anything here knows), where the doctor row — a pull surface that
    must never show a stale complaint — simply stops showing it.
    """
    open_for, facts = _fold(records)
    return {key for key, fold in facts.items() if fold.granted and key not in open_for}


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
    rows: list[dict[str, Any]] = []
    for notice in repair_notices(record, (log or AuditLog()).tail(limit)):
        rows.append(
            {
                "check": "credential_repair",
                "network_id": notice.network_id,
                "ok": False,
                "credential_name": notice.key,
                "device_id": notice.borrower,
                "device_name": notice.borrower_name,
                # The wire diagnostic's own wording, recomposed HERE for this
                # device's operator: the sender's name is the borrower's, and
                # "here" is the device the report and the login live on.
                "detail": notice.detail,
                "remedies": [notice.remedy],
            }
        )
    return rows


def _login_name(key: str) -> str:
    """The server NAME this device lists for an ``mcp:<url>`` key, or ``""``.

    The row's remedy runs HERE, so the name must be the one THIS device resolves
    (review R2/Q2): ``/mcp login`` looks its argument up in the local config, and
    a URL — or another device's name — is refused there. ``""`` keeps the
    command a form instead of an unrunnable paste.
    """
    if not key.startswith("mcp:"):
        return ""
    from local_operator.network.credentials.types import mcp_url_from_key
    from local_operator.network.readiness import mcp_server_name_for_url
    from local_operator.paths import config_dir

    return mcp_server_name_for_url(config_dir(), mcp_url_from_key(key))


def _member_name(record: Any, device_id: str) -> str:
    member = record.member(device_id)
    return str(member.name) if member is not None else ""
