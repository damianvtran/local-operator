"""What a host surface shows for Radient Guard (Guard RFC v0.9 §5.2.3, §5.3, §14', slice L2b).

WHY A SEPARATE, PURE MODULE. The TUI (``tui/widgets/guard_view.py``), the
``guard`` tool (``tools/guard_tool.py``) and the guide gate
(``guides/discovery.py``) must agree on three questions, and two copies of each
answer would drift:

* **Is Guard available here?** (:func:`guard_available`) -- the account is on
  Radient Team (the cached, zero-network ``hub_org.eligibility`` record) AND an
  org has turned Guard on for it, which this computer knows only because the
  admission point queued a prompt (L2a): the consent store exists. A non-Team
  account fails the first test before any file is touched, so it builds no
  Guard file, sees no tool and is never told about Guard.
* **May this person open team insights?** (:func:`insights_tenants`) -- only an
  ``owner`` or ``admin`` of a Team org (RFC §5.6). The answer is the Console
  link, never figures: insights are aggregate, k-anonymous releases rendered
  first-party in Console, and nothing here puts a count into model context.
* **What does the passive indicator say?** (:func:`org_states`) -- one
  :class:`OrgState` per Team org, from local facts only: never "paused" for a
  pending widening (§13.3' ``consent_stale``).

EVERYTHING HERE IS LOCAL. No function opens a socket; the only reads are the
cached membership record and the two Guard SQLite files, and those are opened
only when they already exist.
"""

from __future__ import annotations

import datetime as _dt
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Final
from urllib.parse import quote

#: Console pages this client links to. First-party and non-bearer: opening one
#: asks the browser to sign in; no token or figure travels in the URL.
CONSOLE_ORIGIN: Final = "https://console.radienthq.com"
CONSOLE_GUARD_PATH: Final = "/dashboard/guard"
CONSOLE_TEAM_PATH: Final = "/dashboard/team"

#: Roles that may read org-level Guard insights (§5.6 admin reads).
INSIGHTS_ROLES: Final = frozenset({"owner", "admin"})

#: The indicator's states (§5.3, §13.3'). Only the first three are derivable on
#: this computer today; the server-side stopping states arrive with A2's status
#: body and map through the same ``reason_*`` strings.
STATE_SHARING: Final = "sharing"
STATE_SHARING_PENDING: Final = "sharing_pending"  # consent_stale: still sharing
STATE_WAITING: Final = "waiting"  # first grant pending; nothing is shared yet
STATE_NOT_SHARING: Final = "not_sharing"  # declined (sticky)
STATE_PAUSED: Final = "paused"

PAUSE_REASONS: Final = (
    "entitlement_lapsed",
    "org_off",
    "membership_ended",
    "scope_revoked",
    "revoke_pending",
)


@dataclass(frozen=True)
class OrgState:
    tenant: str
    org: str
    role: str
    state: str
    reason: str = ""
    #: The pending prompt row (``guard.consent.PromptRow``) when one is queued.
    prompt: Any = None
    #: When the last answer was given on another surface ("Answered on <surface>").
    answered_via: str = ""


def _team_orgs(config_dir: Path) -> list[Any]:
    from local_operator.hub_org import eligibility

    return eligibility.eligible_orgs(Path(config_dir))


def guard_available(config_dir: Path | str | None) -> bool:
    """Team membership AND an org-raised Guard prompt on this computer. Never raises."""

    if config_dir is None:
        return False
    try:
        if not _team_orgs(Path(config_dir)):
            return False
        from local_operator.guard.consent import guard_consent_path

        return guard_consent_path(config_dir).is_file()
    except Exception:  # noqa: BLE001 - a gate read fails closed
        return False


def insights_tenants(config_dir: Path | str | None) -> list[tuple[str, str]]:
    """``[(tenant_id, name)]`` where this account is owner/admin on Team. Zero network."""

    if config_dir is None:
        return []
    try:
        return [
            (org.tenant_id, org.tenant_name)
            for org in _team_orgs(Path(config_dir))
            if org.role in INSIGHTS_ROLES
        ]
    except Exception:  # noqa: BLE001
        return []


def insights_url(tenant_id: str) -> str:
    """The opaque first-party Console link for one org's Guard insights."""

    return f"{CONSOLE_ORIGIN}{CONSOLE_GUARD_PATH}?tenant={quote(tenant_id, safe='')}"


def team_url() -> str:
    return f"{CONSOLE_ORIGIN}{CONSOLE_TEAM_PATH}"


def org_states(config_dir: Path | str) -> list[OrgState]:
    """The indicator state for every Team org. ``[]`` for a non-Team account (no file read)."""

    orgs = _team_orgs(Path(config_dir))
    if not orgs:
        return []
    from local_operator.guard.consent import GuardConsentHost, guard_consent_path

    if not guard_consent_path(config_dir).is_file():
        return []
    host = GuardConsentHost(config_dir, transport=None)
    pending = {row.tenant: row for row in host.pending()}
    revoke_pending = _revoke_pending_tenants(host)
    states: list[OrgState] = []
    for org in orgs:
        tenant = org.tenant_id
        row = pending.get(tenant)
        name = row.org_display_name if row is not None else org.tenant_name
        consent = host.consent(tenant)
        live = consent is not None and consent.get("state") == "live"
        sticky = host.sticky(tenant)
        if tenant in revoke_pending:
            state = OrgState(tenant, name, org.role, STATE_PAUSED, "revoke_pending")
        elif live and row is not None:
            state = OrgState(tenant, name, org.role, STATE_SHARING_PENDING, prompt=row)
        elif live:
            state = OrgState(tenant, name, org.role, STATE_SHARING)
        elif row is not None:
            state = OrgState(tenant, name, org.role, STATE_WAITING, prompt=row)
        elif sticky == "revoked":
            state = OrgState(tenant, name, org.role, STATE_PAUSED, "scope_revoked")
        elif sticky == "declined":
            state = OrgState(tenant, name, org.role, STATE_NOT_SHARING)
        else:
            continue  # this org never raised Guard for this member
        states.append(
            OrgState(
                state.tenant,
                state.org,
                state.role,
                state.state,
                state.reason,
                state.prompt,
                _last_answer_surface(host, tenant),
            )
        )
    return states


def _revoke_pending_tenants(host: Any) -> set[str]:
    try:
        rows = host._rows("SELECT tenant FROM consents WHERE state = ?", ("revoke_pending",))
    except Exception:  # noqa: BLE001
        return set()
    return {str(row[0]) for row in rows}


def _last_answer_surface(host: Any, tenant: str) -> str:
    """The surface the newest decided prompt for ``tenant`` was answered on."""

    try:
        rows = host._rows(
            "SELECT decided_via FROM prompts WHERE tenant = ? AND decided_via IS NOT NULL"
            " ORDER BY decided_at DESC LIMIT 1",
            (tenant,),
        )
    except Exception:  # noqa: BLE001
        return ""
    return str(rows[0][0] or "") if rows else ""


# ---------------------------------------------------------------------------
# Notice cadence (§5.2.3: the badge shows on every attach; the notice at most
# once per day per surface)
# ---------------------------------------------------------------------------


def _cadence_path(config_dir: Path | str) -> Path:
    return Path(config_dir) / "guard" / "notice_cadence.json"


def should_show_notice(
    config_dir: Path | str, *, surface: str, today: _dt.date | None = None
) -> bool:
    """True at most once per local day per surface. Never raises (False on error)."""

    import json

    day = (today or _dt.date.today()).isoformat()
    path = _cadence_path(config_dir)
    try:
        data = json.loads(path.read_text(encoding="utf-8")) if path.is_file() else {}
        if not isinstance(data, dict):
            data = {}
        if data.get(surface) == day:
            return False
        data[surface] = day
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(data), encoding="utf-8")
        return True
    except (OSError, ValueError):
        return False


# ---------------------------------------------------------------------------
# The non-Team card (§14', §15.4' C1)
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class NonTeamCard:
    #: ``None`` = "See Radient Team"; an int = "Start {trial_days}-day free trial".
    trial_days: int | None
    url: str


def non_team_card(eligibility: Mapping[str, Any] | None) -> NonTeamCard:
    """The C1 card from ``GET /v1/team-trial/eligibility``'s body (or ``None``).

    The trial CTA appears only when the server says ``eligible`` AND supplies a
    positive whole ``trial_days`` -- the length is never a literal here (§14'),
    so a body without it falls back to "See Radient Team" rather than guessing.
    """

    days: int | None = None
    if isinstance(eligibility, Mapping) and eligibility.get("eligible") is True:
        raw = eligibility.get("trial_days")
        if isinstance(raw, int) and not isinstance(raw, bool) and raw > 0:
            days = raw
    return NonTeamCard(trial_days=days, url=team_url())


__all__ = [
    "INSIGHTS_ROLES",
    "NonTeamCard",
    "OrgState",
    "PAUSE_REASONS",
    "STATE_NOT_SHARING",
    "STATE_PAUSED",
    "STATE_SHARING",
    "STATE_SHARING_PENDING",
    "STATE_WAITING",
    "guard_available",
    "insights_tenants",
    "insights_url",
    "non_team_card",
    "org_states",
    "should_show_notice",
    "team_url",
]
