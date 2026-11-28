"""Wire shapes for the desktop's Radient Guard surface (``features.guard``).

WHAT THIS IS. ``GET /v1/desktop/guard`` and its four small writes expose the
same in-process Guard surface the TUI's ``/guard`` screen renders
(``guard/surface.py``, ``guard/consent.py``, ``guard/reports.py``) to the two
clients that cannot call Python: the desktop app and, through the phone relay,
the phone. The contract page is ``docs/design/guard-desktop-routes.md``; this
module is its machine form, and the two are changed together.

WHAT IS DELIBERATELY ABSENT. There is NO grant model here. A first grant or a
widening is answered only by the typed ``guard_consent`` card in
``GET /v1/desktop/approvals`` (``guard/consent_op.py``): a signed, digest-bound
act the engine verifies. Nothing in this module can carry a scope, a digest or a
signature in a request body, and the request models below forbid extra keys
(``routes/desktop_sessions.Input`` convention) so a client cannot invent one.
Report confirmation is likewise not a route here (see the contract).

STATES ARE ENUMS, TEXT IS THE CLIENT'S. The indicator sentences are copy the
client keys and translates itself (§15.4' C9); the backend ships the closed
enums the TUI also switches on, never a rendered sentence. The only free text
in a response is an org's own display name, which is flattened to one line and
capped before it leaves (the S-3 shape ``surface.non_team_card`` already uses).
"""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, Field

#: ``guard.surface.STATE_*``. ``paused`` carries a ``reason`` (``PAUSE_REASONS``);
#: ``sharing_pending`` is "still sharing, a widening is waiting" and is NEVER
#: shown as paused (§13.3' ``consent_stale``).
OrgStateName = Literal[
    "sharing",
    "sharing_pending",
    "waiting",
    "not_sharing",
    "paused",
    "will_ask",
]

#: ``guard.consent.SCOPES``; a scope this build does not know is dropped before
#: it reaches a client (a client cannot describe it to the person).
ScopeName = Literal[
    "decisions",
    "posture",
    "task_time",
    "review_cycles",
    "reports",
    "trace_detail",
    "admin_share",
]

#: ``none`` = a Team account where no org has turned Guard on for this member
#: (TUI: "Guard is off for you"). ``orgs`` = at least one org is listed.
#: ``non_team`` = the §14' card. ``unavailable`` = a damaged or newer-generation
#: Guard store (the TUI's "unavailable" card): read-only, nothing to act on.
GuardMode = Literal["none", "orgs", "non_team", "unavailable"]

#: The status-bar badge the TUI paints, as an enum (``guard_view.badge_text``).
GuardBadge = Literal["", "waiting", "report_waiting", "sharing_pending", "sharing"]


class GuardPrompt(BaseModel):
    """The queued consent prompt for one org, summarised for display.

    Display only: answering it is the ``guard_consent`` card under
    ``GET /v1/desktop/approvals`` (``approval_id`` names it). ``approval_id`` is
    ``None`` when this computer has no operator key to file the card against, in
    which case the person answers in the terminal ``/guard`` instead.
    """

    approval_id: str | None = None
    collection_rev: int
    reason: str
    #: ``reason == "widened"``: the buttons read "Keep current" / "Share the new items".
    widening: bool
    asked_scopes: list[ScopeName]
    grant_scopes: list[ScopeName]


class GuardOrg(BaseModel):
    """One Team org that has raised Guard for this member (``surface.OrgState``)."""

    tenant: str
    #: Display name, one line, capped.
    org: str
    role: str
    state: OrgStateName
    #: ``PAUSE_REASONS`` for ``paused`` (an unknown server reason is passed
    #: through as given; a client shows the plain "paused" line for it), else "".
    reason: str = ""
    #: Epoch seconds of the cause where the copy names a date (Console stop), else ``None``.
    since: float | None = None
    #: The surface the last answer was given on (``tui`` | ``desktop`` | ``mobile`` | ""),
    #: for "Answered on <surface>".
    answered_via: str = ""
    #: What THIS computer is sharing with the org right now; empty unless a live consent exists.
    scopes_shared: list[ScopeName] = Field(default_factory=list)
    prompt: GuardPrompt | None = None
    #: The ``surface`` predicates the TUI buttons use, so a client does not re-derive them:
    #: ``can_revoke`` = "Stop sharing" is offered; ``can_share_again`` = "Share again" is
    #: offered AND this backend would accept it (a host that cannot answer refuses it).
    can_revoke: bool
    can_share_again: bool


class GuardReportDraft(BaseModel):
    """An agent-drafted report awaiting a person (enum-only; no free text exists)."""

    id: str
    tenant: str
    org: str
    kind: Literal["credential_exposure", "excessive_guardrail", "other"]
    #: Epoch seconds floored to the minute, as the TUI shows it (security review S-4).
    created_at: float


class GuardNotice(BaseModel):
    """An unshown receipt (§15.4' C3): a grant or widening this person may not have made."""

    id: int
    tenant: str
    kind: Literal["grant", "widen"]
    org: str
    scopes: list[ScopeName]
    #: Epoch seconds FLOORED to the minute (the TUI renders ``%H:%M``; security review S-4).
    at: float


class GuardInsightsLink(BaseModel):
    """Console link for an owner/admin. ADVERTISING, never authorization: Console checks."""

    tenant: str
    org: str
    url: str


class GuardNonTeamCard(BaseModel):
    """The §14' non-Team card, with NO network read behind it.

    ``trial_days``/``org`` are ``None`` here by design: they come from
    ``GET /v1/team-trial/eligibility``, which ``GET /v1/desktop/guard/non-team-card``
    performs on demand. ``url`` is first-party and carries no token.
    """

    trial_days: int | None = None
    org: str | None = None
    url: str


class GuardStatus(BaseModel):
    """``GET /v1/desktop/guard``: everything the TUI ``/guard`` overview + badge show."""

    mode: GuardMode
    badge: GuardBadge = ""
    orgs: list[GuardOrg] = Field(default_factory=list)
    reports: list[GuardReportDraft] = Field(default_factory=list)
    notices: list[GuardNotice] = Field(default_factory=list)
    insights: list[GuardInsightsLink] = Field(default_factory=list)
    non_team: GuardNonTeamCard | None = None
    #: Days backup copies of deleted step-by-step timing survive (RFC §5.4 "X");
    #: ``guard.surface.BACKUP_DAYS``. ``None`` = not measured: a client must not
    #: render the trace-detail revoke sentence without it.
    backup_days: int | None = None
    #: This computer's short name, for "this computer" lines and receipts.
    this_computer: str = ""
    #: Whether this backend accepts the person-only writes (``share-again``). It is
    #: ``surface.host_can_answer()`` AS CAPTURED when ``lop serve`` started (the daemon later
    #: drops the markers it is read from): False for a backend started inside an agent's
    #: shell or a desktop console pane.
    can_answer: bool = False
    #: Console pages (first-party, non-bearer).
    team_url: str


class GuardRevokeResult(BaseModel):
    """``POST .../revoke``. The local stop is committed before the request is sent."""

    tenant: str
    #: ``revoked`` = Radient confirmed. ``revoke_pending`` = stopped on this computer,
    #: Radient not reached yet; the next admission point retries (§5.4). Never an error:
    #: this install is no longer sharing in either case.
    status: Literal["revoked", "revoke_pending"]
    #: The engine's code when ``revoke_pending`` (``guard_signed_out``,
    #: ``guard_revoke_failed``, ...).
    code: str = ""
    org: GuardOrg | None = None


class GuardShareAgainResult(BaseModel):
    """``POST .../share-again``: the sticky answer is lifted; the org's prompt returns."""

    tenant: str
    #: The org's state after the lift and one best-effort admission-point read.
    org: GuardOrg | None = None


class GuardReportDiscardResult(BaseModel):
    id: str
    state: Literal["discarded"]


class GuardNoticeShownResult(BaseModel):
    id: int
    shown: Literal[True] = True


class GuardNoticeDueRequest(BaseModel):
    """``POST .../notice-due``: once-a-day cadence for the §15.4' C6 host notice.

    ``desktop`` only, bound per capability (security review S-3): the plane has one shared
    bearer and no per-caller identity, and this router is desktop-only, so the one surface a
    caller here can be is the desktop. A phone-relay twin carries its own authenticated
    surface when it exists.
    """

    model_config = {"extra": "forbid", "strict": True}
    surface: Literal["desktop"]


class GuardNoticeDue(BaseModel):
    #: ``True`` at most once per local day per surface; asking CONSUMES the day.
    due: bool


__all__ = [
    "GuardBadge",
    "GuardInsightsLink",
    "GuardMode",
    "GuardNonTeamCard",
    "GuardNotice",
    "GuardNoticeDue",
    "GuardNoticeDueRequest",
    "GuardNoticeShownResult",
    "GuardOrg",
    "GuardPrompt",
    "GuardReportDiscardResult",
    "GuardReportDraft",
    "GuardRevokeResult",
    "GuardShareAgainResult",
    "GuardStatus",
    "OrgStateName",
    "ScopeName",
]
