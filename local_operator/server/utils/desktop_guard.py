"""The desktop's Radient Guard surface: the TUI's ``/guard`` engine calls, as data.

WHY THIS MODULE EXISTS AND IS THIN. The TUI's ``/guard`` screen
(``tui/widgets/guard_view.py``) and the desktop/phone clients must agree on what
a person is told and what they may do, and two implementations would drift. So
every decision here is one of the SAME functions the TUI calls --
``surface.org_states``, ``surface.is_sharing``/``is_stopped``,
``surface.badge_kind``, ``GuardConsentHost.revoke``/``share_again``,
``GuardReports.drafts``/``discard``, ``consent.check_point`` -- and this module
only (a) composes them into the wire models of ``server/models/desktop_guard.py``
and (b) maps the engine's refusals onto the plane's ``{code, text}`` envelope.

WHAT IT REFUSES TO DO. There is no grant here. A first grant or a widening is
answered only by the typed ``guard_consent`` card (``guard/consent_op.py``);
this module imports no ``accept``/``accept_signed``, takes no scope, digest or
signature, and cannot send a ``/v1/guard`` grant. Report confirmation is
likewise absent (it is the authority-increasing half of
``guard.reports.GuardReports.confirm`` and has no typed card on this plane yet).
``share_again`` is the one person-only write: it lifts a sticky "no" so the
org's prompt can return, so it is refused wherever the TUI's capability gate
(``surface.host_can_answer``) refuses -- a backend started inside an agent's
shell or a desktop console pane. The gate is a fact about how the daemon was STARTED,
so the routes pass it in from the capture ``lop serve`` takes before it drops its own
agent-shell markers (``surface.capture_host_can_answer``); this module never reads the
environment. It is a per-process gate, not a per-caller one: what keeps a caller in an
agent's shell out of a person-started daemon is the desktop bearer, which agent shells
do not receive.

ZERO COST WHERE GUARD IS ABSENT. A non-Team account reads one cached membership
record and nothing else: no Guard file is opened or created, no transport is
resolved, no ``/v1/guard/*`` request is made (``surface.org_states`` returns
before any file read). Every write below refuses first with
``guard_unavailable`` for such an account, before it can build a store or a
cadence file.

BLOCKING. Everything here is synchronous (SQLite, one best-effort network
read for ``share_again`` and ``revoke``); the routes run it with
``asyncio.to_thread`` so a slow Radient never stalls a streaming turn.
"""

from __future__ import annotations

from collections.abc import Iterable
from pathlib import Path
from typing import Any

from local_operator.guard import surface

# ``guard.consent`` / ``guard.reports`` are imported inside the functions that use them: this
# module is imported by ``server.app`` on every ``lop serve``, and the consent engine (SQLite,
# the install key) is first-use cost for the one account in a hundred that has Guard at all.
# ``guard.surface`` is stdlib-only, so it is safe at module scope.

#: ``decided_via`` for an answer given through these routes; ``surface._SURFACE`` in the TUI
#: renders it as "Answered on the desktop app". The phone relay will pass ``mobile``.
SURFACE = "desktop"

#: Notices a person is shown (§15.4' C3: only grants and widenings carry a receipt; a stop
#: is already said on the screen that made it) -- the same filter ``guard_view.receipt_text``
#: applies.
RECEIPT_KINDS = frozenset({"grant", "widen"})


class GuardRefusal(Exception):
    """A request the Guard surface refuses. ``status`` is the HTTP class, ``code`` the token."""

    def __init__(self, status: int, code: str, message: str) -> None:
        super().__init__(message)
        self.status = status
        self.code = code
        self.message = message


def _unavailable() -> GuardRefusal:
    # 404, not 403: for an account where Guard is not on, the resource does not exist, and
    # a client keys the whole "hide Guard" state on this one code.
    return GuardRefusal(404, "guard_unavailable", "Radient Guard is not available on this account.")


def _require_available(root: Path) -> None:
    """Team membership AND an org-raised prompt on this computer (``guard_available``)."""

    if not surface.guard_available(root):
        raise _unavailable()


def _engine(error: Exception) -> GuardRefusal:
    return GuardRefusal(409, error.code or "guard_refused", str(error))


def _scopes(values: Any) -> list[str]:
    """Known scopes only, in the engine's sorted order; an unknown one is dropped, not shown."""

    from local_operator.guard.consent import SCOPES

    return [scope for scope in sorted(set(values or ())) if scope in SCOPES]


def _prompt(row: Any) -> dict[str, Any]:
    return {
        "approval_id": row.approval_id or None,
        "collection_rev": row.collection_rev,
        "reason": row.reason,
        "widening": row.widening,
        "asked_scopes": _scopes(row.asked_scopes),
        "grant_scopes": _scopes(row.grant_scopes),
    }


def _org(state: surface.OrgState, host: Any, can_answer: bool) -> dict[str, Any]:
    consent = host.consent(state.tenant)
    live = consent is not None and consent.get("state") == "live"
    return {
        "tenant": state.tenant,
        "org": surface.one_line(state.org),
        "role": state.role,
        "state": state.state,
        "reason": state.reason,
        "since": state.since or None,
        "answered_via": state.answered_via,
        "scopes_shared": _scopes(consent.get("scopes")) if live and consent else [],
        "prompt": _prompt(state.prompt) if state.prompt is not None else None,
        "can_revoke": surface.is_sharing(state),
        # Offered AND admitted: a host the TUI would hide the button on must not
        # advertise it here either.
        "can_share_again": surface.is_stopped(state) and can_answer,
    }


def _find(root: Path, tenant: str) -> surface.OrgState:
    from local_operator.guard.consent import GuardConsentRefused
    from local_operator.guard.reports import GuardReportRefused

    try:
        for state in surface.org_states(root):
            if state.tenant == tenant:
                return state
    except (GuardConsentRefused, GuardReportRefused) as error:
        raise _engine(error) from None
    raise GuardRefusal(404, "guard_org_unknown", "This computer has no Guard record for that org.")


def _fresh_org(root: Path, tenant: str, can_answer: bool) -> dict[str, Any] | None:
    """The org's row AFTER a write, or ``None`` when it no longer lists (never raises)."""

    from local_operator.guard.consent import GuardConsentHost

    try:
        host = GuardConsentHost(root, transport=None, mirror=False)
        for state in surface.org_states(root):
            if state.tenant == tenant:
                return _org(state, host, can_answer)
    except Exception:  # noqa: BLE001 - write done; a re-read failure must not undo it
        return None
    return None


# ---------------------------------------------------------------------------
# Reads
# ---------------------------------------------------------------------------


def status(root: Path, *, can_answer: bool, listed: set[int]) -> dict[str, Any]:
    """``GET /v1/desktop/guard``. Local facts only; zero network, zero files for non-Team.

    ``can_answer`` is the daemon's start-time capture (``surface.served_host_can_answer``),
    NOT a read of this process's environment, which ``lop serve`` has already cleaned
    (security review S-1). ``listed`` is the set of receipt ids this daemon has handed out:
    ``notice_shown`` accepts only ids that were first listed here (S-2).
    """

    from local_operator.guard.consent import (
        GuardConsentHost,
        GuardConsentRefused,
        team_tenants,
    )
    from local_operator.guard.reports import GuardReportRefused, GuardReports

    base: dict[str, Any] = {
        "backup_days": surface.BACKUP_DAYS,
        "this_computer": surface.computer_name(),
        "can_answer": can_answer,
        "team_url": surface.team_url(),
    }
    try:
        states = surface.org_states(root)
        drafts = GuardReports(root).drafts() if states else []
        host = GuardConsentHost(root, transport=None, mirror=False)
        # Read inside the guard: a damaged store can fail here as well as in ``org_states``.
        raw_notices = host.notices(unshown_only=True) if states else []
    except (GuardConsentRefused, GuardReportRefused):
        # A damaged or newer-generation store: say so, never an exception (review R1-3).
        return {**base, "mode": "unavailable"}
    if not states:
        # The TUI's ladder: no org listed and neither a Team member nor an insights admin
        # is the §14' card; a Team account with nothing raised is "Guard is off for you".
        if not surface.insights_tenants(root) and not team_tenants(root):
            card = surface.non_team_card(None)
            return {
                **base,
                "mode": "non_team",
                "non_team": {"trial_days": card.trial_days, "org": card.org, "url": card.url},
            }
        return {**base, "mode": "none"}

    known = {state.tenant: state.org for state in states}
    reports = [d for d in drafts if d.tenant in known]
    notices = [
        {
            "id": int(n["id"]),
            "tenant": n["tenant"],
            "kind": n["kind"],
            "org": surface.one_line(str(n.get("org_display_name") or "")),
            "scopes": _scopes(n.get("scopes")),
            "at": surface.to_minute(n.get("at") or 0.0),
        }
        for n in raw_notices
        if n.get("kind") in RECEIPT_KINDS
    ]
    _remember(listed, (n["id"] for n in notices))
    return {
        **base,
        "mode": "orgs",
        "badge": surface.badge_kind(states, reports_waiting=len(reports)),
        "orgs": [_org(state, host, can_answer) for state in states],
        "reports": [
            {
                "id": d.id,
                "tenant": d.tenant,
                "org": surface.one_line(known[d.tenant]),
                "kind": d.kind,
                "created_at": surface.to_minute(d.created_at),
            }
            for d in reports
        ],
        "notices": notices,
        "insights": [
            {
                "tenant": tenant,
                "org": surface.one_line(name),
                "url": surface.insights_url(tenant),
            }
            for tenant, name in surface.guard_insights_tenants(root)
        ],
    }


def non_team_card(root: Path) -> dict[str, Any]:
    """``GET .../non-team-card``: the §14' card with its one eligibility read.

    The read is ``GET /v1/team-trial/eligibility`` (never ``/v1/guard/*``). Any failure -- signed
    out, offline, a malformed body -- degrades to "See Radient Team", exactly as the TUI's
    card does, because a card that errors is worse than one without a trial.
    """

    from local_operator.guard.consent import team_tenants

    if team_tenants(root) or surface.insights_tenants(root):
        raise GuardRefusal(
            404, "guard_not_applicable", "The Radient Team card is only for accounts without Team."
        )
    try:
        body = surface.load_trial_eligibility(root)
    except Exception:  # noqa: BLE001 - the card falls back to "See Radient Team"
        body = None
    card = surface.non_team_card(body)
    return {"trial_days": card.trial_days, "org": card.org, "url": card.url}


# ---------------------------------------------------------------------------
# Writes
# ---------------------------------------------------------------------------


def revoke(root: Path, tenant: str, *, can_answer: bool) -> dict[str, Any]:
    """``POST .../orgs/{tenant}/revoke``: "Stop sharing". The TUI's own ``host.revoke``."""

    from local_operator.guard.consent import (
        GuardConsentHost,
        GuardConsentRefused,
        default_transport,
    )

    _require_available(root)
    state = _find(root, tenant)
    if not surface.is_sharing(state):
        raise GuardRefusal(409, "guard_not_sharing", "This computer is not sharing with that org.")
    host = GuardConsentHost(root, transport=lambda: default_transport(root))
    try:
        outcome = host.revoke(tenant)
    except GuardConsentRefused as error:
        raise _engine(error) from None
    # ``revoke`` commits the local stop BEFORE its request, so a failed request is still
    # "stopped on this computer": a 200 with ``revoke_pending``, retried at the next
    # admission point -- never a 5xx that would read as "still sharing" (§5.4, C5).
    done = outcome.status == "revoked"
    return {
        "tenant": tenant,
        "status": "revoked" if done else "revoke_pending",
        "code": "" if done else outcome.code,
        "org": _fresh_org(root, tenant, can_answer),
    }


def share_again(root: Path, tenant: str, *, can_answer: bool) -> dict[str, Any]:
    """``POST .../orgs/{tenant}/share-again``: lift the sticky answer, then re-read the offer."""

    from local_operator.guard import consent
    from local_operator.guard.consent import GuardConsentHost, GuardConsentRefused

    _require_available(root)
    state = _find(root, tenant)
    # The TUI hides "Share again" for a host that cannot answer (U12); the same gate here,
    # read from the start-time capture because the served daemon's own environment no longer
    # carries the markers (security review S-1).
    if not can_answer:
        raise GuardRefusal(
            409,
            "guard_answer_not_admitted",
            "Share again is answered by a person, not from an agent-driven backend.",
        )
    if not surface.is_stopped(state):
        raise GuardRefusal(409, "guard_not_stopped", "That org is not in a stopped state.")
    host = GuardConsentHost(root, transport=lambda: consent.default_transport(root))
    try:
        host.share_again(tenant)
    except GuardConsentRefused as error:
        raise _engine(error) from None
    # U2/C4: read the org's offer now rather than at the next 6 h tick so its prompt can
    # follow. ``check_point`` never raises; the next admission point retries anyway.
    consent.check_point(root, boundary="share_again")
    return {"tenant": tenant, "org": _fresh_org(root, tenant, can_answer)}


def discard_report(root: Path, draft_id: str) -> dict[str, Any]:
    """``POST .../reports/{id}/discard``: the safe direction; ordinary, no admission."""

    from local_operator.guard.reports import GuardReportRefused, GuardReports

    _require_available(root)
    try:
        GuardReports(root).discard(draft_id, surface=SURFACE)
    except GuardReportRefused as error:
        if error.code in ("report_not_found", "report_already_answered"):
            raise GuardRefusal(
                404, "guard_report_unknown", "There is no open Guard report with that id."
            ) from None
        raise _engine(error) from None
    return {"id": draft_id, "state": "discarded"}


#: Receipt ids remembered per daemon. A receipt table is tens of rows; this only bounds a
#: pathological one so the set cannot grow without limit.
_LISTED_CAP = 1024


def _remember(listed: set[int], ids: Iterable[int]) -> None:
    """Record the receipt ids ``status`` just handed out (``notice_shown``'s allow-list)."""

    fresh = [int(i) for i in ids]
    if len(listed) + len(fresh) > _LISTED_CAP:
        listed.clear()
    listed.update(fresh)


def notice_shown(root: Path, notice_id: int, *, listed: set[int]) -> dict[str, Any]:
    """``POST .../notices/{id}/shown``: mark a receipt shown (shared with the TUI).

    An acknowledgement a client makes AFTER it rendered the receipt, so only an id that
    :func:`status` returned from this daemon is accepted: a bearer holder cannot dismiss a
    receipt no surface here has displayed by guessing small integers (security review S-2).
    Idempotent: a repeat for a receipt already shown is the same 200 with no second write.
    The receipt itself stays in the store; this only flips ``shown``.
    """

    from local_operator.guard.consent import GuardConsentHost

    _require_available(root)
    if notice_id not in listed:
        raise GuardRefusal(
            404, "guard_notice_unknown", "There is no Guard receipt with that id to mark shown."
        )
    host = GuardConsentHost(root, transport=None, mirror=False)
    pending = {
        int(n["id"]) for n in host.notices(unshown_only=True) if n.get("kind") in RECEIPT_KINDS
    }
    if notice_id in pending:
        host.mark_notice_shown(notice_id)
    return {"id": notice_id, "shown": True}


def notice_due(root: Path, surface_name: str) -> dict[str, Any]:
    """``POST .../notice-due``: the §15.4' C6 once-a-day cadence. Asking consumes the day.

    ``surface_name`` is the surface the authenticated caller IS. The plane has one shared
    bearer and no per-caller identity, so the request model binds it per capability: this
    router is desktop-only (``features.guard`` is in ``DESKTOP_ONLY_FEATURES``), hence only
    ``desktop`` is accepted and a caller cannot consume another surface's day (S-3).
    """

    _require_available(root)
    return {"due": surface.should_show_notice(root, surface=surface_name)}
