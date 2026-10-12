"""Desktop routes for agent messages (Team Bridge harness lane, H1).

The UI lane's four HTTP surfaces over :mod:`local_operator.team_bridge`:

- ``GET  /v1/desktop/team-bridge/status``    — the eligibility/state read.
- ``GET  /v1/desktop/team-bridge/settings``  — endpoint, consent record, policies.
- ``PUT  /v1/desktop/team-bridge/settings``  — state changes (the consented write).
- ``GET  /v1/desktop/team-bridge/allowlist`` — own policy entries + honest defaults.
- ``PUT  /v1/desktop/team-bridge/allowlist`` — the bounded replace-set.

WHAT THESE ROUTES ARE NOT. They never flip a launch flag, never retry against a
second deployment, and never touch a credential: every call goes through
:class:`~local_operator.team_bridge.client.TeamBridgeClient`, which addresses
exactly the hub base URL the rest of the harness resolves and authenticates with
the signed-in Radient OAuth session — or refuses with the sign-in state. With the
server dark (``BRIDGE_ENABLED`` off, RFC §5.5) the upstream answers 503
``messaging_unavailable``; the STATUS route renders that as a state
(``unavailable`` with the refusal), which is the fail-closed behaviour the
contract asks for, and the other routes carry the §6.5 code through.

NONE of the routes gate on local ``team_bridge.enabled``: the composed surface
gate is capability × eligible × enabled and it is the RENDERER's (RFC §5.4), so a
user can open Settings → Agent messages on a fresh machine and see the on-ramp;
the routes must answer exactly what state the machine is in, not refuse to say.

THE STATUS CACHE IS PROCESS-WIDE (RFC §5.1: coalesce per account across
sessions). The route reads it through
:func:`~local_operator.team_bridge.client.get_status_discipline`, so every UI
surface and every request in this daemon share one cached answer per account and
one in-flight read. A ``409 consent_version_unknown`` PUT invalidates it, because
the refusal means the server serves a consent version this daemon has not read
yet — re-serving the cached one would loop (§5.2's "refresh status and
re-render").

REFUSAL SHAPE. Errors answer the desktop plane's ``{"code", "message"}`` detail
(the shape ``_job_refusal`` in ``desktop_sessions`` states for the wake/monitor
families, so one client reader unwraps them all), with ``details`` and
``retry_after_seconds`` riding along where §6.5 defines them. ``code`` is the
server's §6.5 code verbatim (``bridge_not_available``, ``recipient_unavailable``,
``consent_version_unknown``, ``rate_limited`` …), plus two client-side codes:
``bridge_signin_required`` (the terminal sign-in state, §13.2) and
``bridge_unreachable`` (transport failure; retryable).

IMPORT DISCIPLINE. The client stack pulls ``httpx``, so it is imported INSIDE the
handlers, never at module scope: this module rides ``server.app``'s import path on
every ``lop serve`` boot, and the desktop_radient rule (a module-scope httpx was
~350 ms of every boot) applies here unchanged.
"""

from __future__ import annotations

from datetime import datetime, timezone
from typing import TYPE_CHECKING, Any, Literal, NoReturn

from fastapi import APIRouter, Depends, HTTPException, Query
from pydantic import BaseModel, ConfigDict, Field, model_validator

from local_operator.config import ConfigManager
from local_operator.config_watch import process_watcher
from local_operator.server.dependencies import get_config_manager
from local_operator.server.desktop import require_desktop
from local_operator.server.features import feature_flags
from local_operator.server.models.schemas import CRUDResponse
from local_operator.server.routes.desktop_sessions import reply

if TYPE_CHECKING:  # pragma: no cover - typing only, never imported at runtime
    # The class lives in a module that pulls httpx; importing it here would put
    # that on ``lop serve``'s boot path through this router (see the module
    # docstring), so it is reached for STATICALLY only and the runtime import
    # stays inside the handlers.
    from local_operator.team_bridge.client import BridgeRefusal

router = APIRouter(tags=["Desktop agent messages"], dependencies=[Depends(require_desktop)])


class _Strict(BaseModel):
    """Input bodies are strict: an unknown field is a client bug, not a no-op.

    The upstream ignores unknown fields (§6.6 additive rule), and that is right
    for the WIRE; a desktop body is ours on both ends, so a typo in a field name
    must fail here rather than silently drop the user's edit.
    """

    model_config = ConfigDict(extra="forbid")


class TeamBridgeSettingsPut(_Strict):
    """The settings write (§5.2). ``state`` is REQUIRED — the body is a state
    assertion, and a label-only rename is refused rather than treated as "keep
    the current state". A pause or an opt-out is sent ALONE: bundling a name
    change with ``paused``/``disabled`` is refused by the client before it
    reaches the wire (allowlist edits and label edits are separate requests).
    """

    tenant_id: str | None = Field(default=None, max_length=128)
    state: Literal["active", "paused", "disabled"]
    consent_version: str | None = Field(default=None, max_length=128)
    consented_via: Literal["dialog", "settings"] | None = None
    label: str | None = Field(default=None, max_length=64)
    #: The account this write was CAPTURED for (H1 review round 1, R6/Q1).
    #: REQUIRED on this plane: the body is ours on both ends, and without the
    #: guard a dialog captured for account A could enroll account B after a
    #: login switch — before any renderer could reject a response. The client
    #: compares it to the EXACT pinned dispatch snapshot and refuses with
    #: ``account_mismatch`` before the first byte leaves; it is never sent
    #: upstream.
    expected_account_id: str = Field(min_length=1, max_length=128)


class AllowlistTargetIn(_Strict):
    endpoint_id: str | None = Field(default=None, max_length=128)
    account_id: str | None = Field(default=None, max_length=128)

    @model_validator(mode="after")
    def _exactly_one(self) -> "AllowlistTargetIn":
        named = [value for value in (self.endpoint_id, self.account_id) if value]
        if len(named) != 1:
            raise ValueError("a target names exactly one of endpoint_id or account_id")
        return self


class AllowlistEntryIn(_Strict):
    direction: Literal["send", "receive"]
    effect: Literal["allow", "deny"]
    target: AllowlistTargetIn


class TeamBridgeAllowlistPut(_Strict):
    """The bounded replace-set (§6.1): at most 200 entries, whole-policy."""

    tenant_id: str | None = Field(default=None, max_length=128)
    entries: list[AllowlistEntryIn] = Field(max_length=200)
    #: The same captured-account guard the settings write carries (review
    #: round 1, R6): a policy surface must not be the one mutation that
    #: skipped the receipt.
    expected_account_id: str = Field(min_length=1, max_length=128)


def _iso(seconds: float | None) -> str | None:
    if seconds is None:
        return None
    return datetime.fromtimestamp(seconds, tz=timezone.utc).isoformat()


def _refusal_detail(refusal: "BridgeRefusal") -> dict[str, Any]:
    detail: dict[str, Any] = {"code": refusal.code, "message": refusal.message}
    if refusal.details:
        detail["details"] = refusal.details
    if refusal.retry_after_seconds is not None:
        detail["retry_after_seconds"] = refusal.retry_after_seconds
    return detail


def _raise_bridge(refusal: "BridgeRefusal", *, refresh_status: bool = False) -> NoReturn:
    """Map one :class:`BridgeRefusal` onto the desktop refusal shape.

    The status is the server's own when it sent one (they are the §6.5 classes);
    a client-side refusal (the destination guard, a locally-detected over-bound
    body) has none, and the two client-side codes carry their own: a refused
    destination is a configuration condition (409), an unreachable hub is a
    gateway failure (502) exactly as ``desktop_radient`` reads an unreachable
    upstream.
    """

    status = refusal.http_status
    if status is None:
        status = 409 if refusal.code == "bridge_destination_refused" else 502
    detail = _refusal_detail(refusal)
    if refresh_status:
        detail["refresh_status"] = True
    raise HTTPException(status, detail)


def _raise_sign_in(exc: Exception) -> NoReturn:
    raise HTTPException(
        401,
        {"code": "bridge_signin_required", "message": str(exc)},
    )


@router.get("/v1/desktop/team-bridge/status", response_model=CRUDResponse[Any])
async def status(
    boundary: str = Query(default="surface_open"),
    config_manager: ConfigManager = Depends(get_config_manager),
) -> CRUDResponse[Any]:
    """This machine's agent-messages state, as one composed answer.

    Always 200 when the app can answer at all: the answer is a STATE
    (``signed_out`` / ``unavailable`` / ``update_required`` / ``ready``), and a
    surface that asked "what is my state" must not have to treat its own state
    as an error. ``boundary`` names the natural boundary that triggered the read
    (§5.1) and is validated leniently by the discipline; the cache floor makes
    rapid repeats free.
    """

    from local_operator.team_bridge.client import (
        consent_for_display,
        get_status_discipline,
        selected_tenant_id,
        surface_projection,
    )
    from local_operator.team_bridge.settings import TeamBridgeSettings

    check = await get_status_discipline(config_manager.config_dir).check(boundary)
    # The local keys come from the DIRECTORY-scoped process watcher's last-good
    # snapshot, NOT the injected manager's cache: the server constructs ONE
    # manager (``server/app.py``) and never reloads it, so an external write
    # (the CLI, the settings page, another process) left this block stale
    # forever — measured with a retained manager in review round 3 (Q6): a
    # fresh reader saw the new pair while this route still returned the old
    # one, and an explicit reload of the same manager fixed it without any
    # cache involvement. ``poll_now`` re-stats and re-parses; a malformed or
    # missing document KEEPS the last-good snapshot (no ``.bad`` backup, no
    # defaults gap), and a document whose change moved no REGISTERED key still
    # adopts (``team_bridge.onboarding`` is deliberately unregistered). The
    # watcher is the existing directory-scoped singleton the session consumers
    # already run — read here at the boundary that renders it; no server-wide
    # reload/subscription is introduced, and the manager stays the root/auth
    # source it always was.
    watcher = process_watcher(config_manager.config_dir)
    watcher.poll_now()
    local = TeamBridgeSettings.from_values(watcher.values)
    consent = consent_for_display(check)
    # §5.6's A–G reading, computed ONCE (the same function the availability
    # record uses), so a renderer binds to the backend's policy instead of
    # re-deriving it. The onboarding row also needs the LOCAL dismissal flag:
    # "once" in state B is decided by ``team_bridge.onboarding``.
    projection = surface_projection(check, onboarding_seen=local.onboarding == "seen")
    result = {
        # The capability THIS route family advertises (a renderer reads the
        # same number off ``GET /v1/capabilities``; it rides here so one
        # response can drive the whole surface decision).
        "capability": feature_flags()["team_bridge"],
        "state": check.state,
        "boundary": check.boundary,
        "from_cache": check.from_cache,
        "checked_at": _iso(check.checked_at),
        "fetched_at": _iso(check.fetched_at),
        "update_required": check.update_required,
        "account_id": check.account_id,
        # The §3.2 v1 selection, ONE row for every field below (review round
        # 1, R2): the first eligible tenant in the server's own order, else
        # the retained endpoint's row — never "several rows, so none".
        "tenant_id": selected_tenant_id(check.payload),
        "surface_state": projection["surface_state"],
        #: §5.6 F's sentence when this answer PROVES F — the neutral
        #: "Agent messages are turned off" in v1 (review round 2, R9) — else
        #: None (review round 1, Q3 / design round 1, D3).
        "state_note": check.state_note,
        "visibility": projection["visibility"],
        "read_only": projection["read_only"],
        "can_receive": projection["can_receive"],
        "can_decide": projection["can_decide"],
        "local": {
            "enabled": local.enabled,
            "onboarding": local.onboarding,
            "session": local.session,
        },
        # The admission-gate receipt (§8, §13.9): screening is an INVARIANT of
        # this client, not a setting — no registered key can turn it off (a
        # test pins the registry against it) — and the receipt lets a surface
        # state that plainly instead of implying a toggle exists.
        "screening": {"required": True},
        #: Whether the captured-account guard is ENFORCED on the mutation
        #: routes (review round 1, R6): the UI requires this receipt before an
        #: Enable, and it is only true because the client compares the
        #: captured account to the pinned dispatch snapshot. Capability stays
        #: route presence; this receipt is the enforcement signal.
        "consent_account_guard": True,
        #: Served-text slots refused on this answer — a surface refreshes
        #: rather than rendering refused or stale bytes (review round 1, R3).
        "texts_rejected": sorted(check.rejected),
        "status": check.payload,
        "consent": (
            {"version": consent[0], "text": consent[1], "source": consent[2]}
            if consent is not None
            else None
        ),
        "error": _refusal_detail(check.error) if check.error is not None else None,
    }
    return reply(result)


@router.get("/v1/desktop/team-bridge/settings", response_model=CRUDResponse[Any])
async def get_settings(
    tenant_id: str | None = Query(default=None),
    config_manager: ConfigManager = Depends(get_config_manager),
) -> CRUDResponse[Any]:
    """The caller's endpoint + consent record + policies (§6.1)."""

    from local_operator.team_bridge.client import (
        BridgeRefusal,
        BridgeSignInRequired,
        TeamBridgeClient,
    )

    client = TeamBridgeClient(config_manager.config_dir)
    try:
        payload = await client.settings(tenant_id=tenant_id)
    except BridgeSignInRequired as exc:
        _raise_sign_in(exc)
    except BridgeRefusal as exc:
        _raise_bridge(exc)
    return reply(payload)


@router.put("/v1/desktop/team-bridge/settings", response_model=CRUDResponse[Any])
async def put_settings(
    body: TeamBridgeSettingsPut,
    config_manager: ConfigManager = Depends(get_config_manager),
) -> CRUDResponse[Any]:
    """State changes and the consented enable (§5.2).

    ``409 consent_version_unknown`` additionally invalidates the shared status
    cache before it is surfaced: the client's move is §5.2's "refresh status and
    re-render", and re-serving the cached (stale) consent version would make the
    next PUT fail the same way.
    """

    from local_operator.team_bridge.client import (
        BridgeRefusal,
        BridgeSignInRequired,
        TeamBridgeClient,
        get_status_discipline,
    )

    client = TeamBridgeClient(config_manager.config_dir)
    try:
        payload = await client.put_settings(
            state=body.state,
            tenant_id=body.tenant_id,
            consent_version=body.consent_version,
            consented_via=body.consented_via,
            label=body.label,
            expected_account_id=body.expected_account_id,
        )
    except BridgeSignInRequired as exc:
        # BEFORE the ValueError arm, deliberately: BridgeSignInRequired IS a
        # ValueError (the tunnels' LoginRequired shape), and a subclass caught
        # by the broader arm would surface as a 422 caller bug.
        _raise_sign_in(exc)
    except BridgeRefusal as exc:
        if exc.refreshes_status:
            get_status_discipline(config_manager.config_dir).invalidate()
        _raise_bridge(exc, refresh_status=exc.refreshes_status)
    except ValueError as exc:
        # The client's §5.2 precondition refusals (a bundled pause+rename, a
        # missing consent echo for a known-to-be-required case): caller bugs, in
        # the desktop plane's validation register.
        raise HTTPException(422, {"code": "invalid_request", "message": str(exc)}) from None
    return reply(payload)


@router.get("/v1/desktop/team-bridge/allowlist", response_model=CRUDResponse[Any])
async def get_allowlist(
    tenant_id: str | None = Query(default=None),
    config_manager: ConfigManager = Depends(get_config_manager),
) -> CRUDResponse[Any]:
    """The caller's own policy entries + the honest default-state reading.

    ``view`` is the S-7 apparatus a surface needs and cannot derive from the
    entries alone without re-implementing the RFC's rules: which direction each
    allow entry switches to listed-only, and the two cues — adding an allow
    entry restricts that direction to NEW threads (the class default stops
    applying), and removing an allow entry does NOT stop an open thread (only a
    deny does). The copy is the UI lane's; the reading is data (§13.7).
    """

    from local_operator.team_bridge.client import (
        BridgeRefusal,
        BridgeSignInRequired,
        TeamBridgeClient,
        allowlist_view,
    )

    client = TeamBridgeClient(config_manager.config_dir)
    try:
        payload = await client.allowlist(tenant_id=tenant_id)
    except BridgeSignInRequired as exc:
        _raise_sign_in(exc)
    except BridgeRefusal as exc:
        _raise_bridge(exc)
    return reply({"allowlist": payload.get("allowlist"), "view": allowlist_view(payload)})


@router.put("/v1/desktop/team-bridge/allowlist", response_model=CRUDResponse[Any])
async def put_allowlist(
    body: TeamBridgeAllowlistPut,
    config_manager: ConfigManager = Depends(get_config_manager),
) -> CRUDResponse[Any]:
    """The bounded replace-set write (§6.1/§9). ``entries`` is whole-policy."""

    from local_operator.team_bridge.client import (
        BridgeRefusal,
        BridgeSignInRequired,
        TeamBridgeClient,
        allowlist_view,
    )

    client = TeamBridgeClient(config_manager.config_dir)
    try:
        payload = await client.put_allowlist(
            [entry.model_dump(exclude_none=True) for entry in body.entries],
            tenant_id=body.tenant_id,
            expected_account_id=body.expected_account_id,
        )
    except BridgeSignInRequired as exc:
        # Before the ValueError arm — BridgeSignInRequired IS a ValueError.
        _raise_sign_in(exc)
    except BridgeRefusal as exc:
        _raise_bridge(exc)
    except ValueError as exc:
        raise HTTPException(422, {"code": "invalid_request", "message": str(exc)}) from None
    return reply({"allowlist": payload.get("allowlist"), "view": allowlist_view(payload)})
