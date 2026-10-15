"""Desktop routes for agent messages (Team Bridge harness lane, H1).

The UI lane's HTTP surfaces over :mod:`local_operator.team_bridge`:

- ``GET  /v1/desktop/team-bridge/status``    — the eligibility/state read.
- ``GET  /v1/desktop/team-bridge/settings``  — endpoint, consent record, policies.
- ``PUT  /v1/desktop/team-bridge/settings``  — state changes (the consented write).
- ``GET  /v1/desktop/team-bridge/allowlist`` — own policy entries + honest defaults.
- ``PUT  /v1/desktop/team-bridge/allowlist`` — the bounded replace-set.
- ``GET  /v1/desktop/team-bridge/inbox``     — one page of agent messages, METADATA
  ONLY (the READ slice): ids, delivery/screening state and coarse codes, no body,
  and zero local writes — the connector's own cursors are not this surface's.

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

import re
from datetime import datetime, timezone
from typing import TYPE_CHECKING, Any, Literal, NoReturn

from fastapi import APIRouter, Depends, HTTPException, Query, Request
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


#: What a query that names something this surface does not accept is told. ONE
#: fixed sentence for every grammar/closed-set violation on the read query: the
#: rejected VALUE never rides the response (S-RSA-7 — a validation error's own
#: text can quote the caller's input back), and a caller bug is one condition,
#: not one condition per field.
#:
#: AND IT IS THE ONLY ANSWER (review R-6). Every bound on the read query is
#: enforced in ``TeamBridgeInboxQuery``, never by ``Query(...)``: FastAPI's own
#: validation answers with a body that echoes the offending ``input``, so a bound
#: declared on the parameter puts the caller's value straight back on the wire —
#: the one thing this sentence exists to prevent. The handler therefore declares
#: each query name as a plain optional string and lets the model do the parsing,
#: the bounds and the closed sets.
INVALID_READ_QUERY = "this read named a value the agent-messages read surface does not accept"

#: What a read that cannot open this machine's store is told (review R-7). It
#: says what happened to THIS READ and diagnoses nothing: a perfectly healthy
#: store lands here when the process cannot create its WAL side file (a
#: read-only config directory), so "it needs repair" was wrong for a benign case
#: (review M2-3). Fail-closed is the same either way — the read is refused, never
#: repaired and never worked around.
STATE_UNREADABLE = "this machine's agent-messages store cannot be read right now"


class TeamBridgeInboxQuery(_Strict):
    """The inbox read query (§6.1), STRICT and grammar-checked before any I/O.

    ``expected_account_id`` is the captured-account receipt the sibling writes
    carry, and it is the account this read is FOR: a renderer that captured the
    inbox for A must not be answered with B's page after a sign-in switch. The
    ids and cursor are validated by GRAMMAR here, before any upstream path is
    built: the transport interpolates ids into ``/messaging/messages/{id}`` and
    folds them into query strings, and a ``..``, ``?`` or ``#`` in one rewrites
    the request (review S-RSA-5). Reject, never coerce.

    ``state`` is the upstream DELIVERY state, a CLOSED set (§6.1). The cursor is
    opaque: it carries a position and nothing else (the server rebuilds the
    authorization filter), so this model bounds its shape and never decodes it.
    """

    expected_account_id: str = Field(min_length=1, max_length=128)
    tenant_id: str = Field(min_length=1, max_length=128)
    after: str | None = Field(default=None, max_length=512)
    limit: int = Field(default=50, ge=1, le=100)
    state: str | None = Field(default=None, max_length=16)
    thread_id: str | None = Field(default=None, max_length=80)
    sender_account_id: str | None = Field(default=None, max_length=80)

    @model_validator(mode="after")
    def _grammar(self) -> "TeamBridgeInboxQuery":
        from local_operator.team_bridge.read_context import (
            ACCOUNT_ID_PATTERN,
            CURSOR_PATTERN,
            DELIVERY_STATES,
            TENANT_ID_PATTERN,
            THREAD_ID_PATTERN,
        )

        checks = (
            (self.expected_account_id, ACCOUNT_ID_PATTERN, "expected_account_id"),
            (self.tenant_id, TENANT_ID_PATTERN, "tenant_id"),
            (self.after, CURSOR_PATTERN, "after"),
            (self.thread_id, THREAD_ID_PATTERN, "thread_id"),
            (self.sender_account_id, ACCOUNT_ID_PATTERN, "sender_account_id"),
        )
        for value, pattern, name in checks:
            # ``fullmatch``, not ``match`` (review R-5): these patterns are
            # anchored with ``^…$``, and ``$`` also matches BEFORE a trailing
            # newline, so ``bth_1\n`` passed the grammar and reached the
            # transport. A full match is what "outside the accepted grammar"
            # means for a value interpolated into a path or a query string.
            if value is not None and re.fullmatch(pattern, value) is None:
                raise ValueError(f"{name} is outside the accepted grammar")
        if self.state is not None and self.state not in DELIVERY_STATES:
            raise ValueError("state is not one of the delivery states")
        return self


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


#: The READ slice's closed refusal table (review S-RSA-7): code → (status, FIXED
#: sentence). The read path's sources (``PullRefused``, ``StoreRefused``,
#: ``TeamBridgeTypeError``, the screening pipeline's hold vocabulary) can carry
#: server- or provider-supplied text, so the response never quotes them: the code
#: is the machine-readable half, the sentence here the human one, and neither is
#: ``str(exception)``. §6.5 codes that the UPSTREAM sends keep riding
#: ``_raise_bridge`` unchanged (this table is the LOCAL read path's vocabulary).
_READ_REFUSALS: dict[str, tuple[int, str]] = {
    "bridge_signin_required": (401, "sign in to Radient and try again"),
    "state_unreadable": (409, STATE_UNREADABLE),
    "no_access": (401, "sign in to Radient and try again"),
    "not_signed_in": (401, "sign in to Radient and try again"),
    "account_mismatch": (
        409,
        "this read was captured for a different account; sign in as that account and try again",
    ),
    "invalid_tenant": (
        422,
        "this machine is enrolled for a different organization than the one this read named",
    ),
    "history_context_unavailable": (
        409,
        "this machine has no retained agent messages for the signed-in account",
    ),
    "not_enrolled": (
        409,
        "this machine has no committed agent-messages owner for the signed-in account",
    ),
    "not_oauth": (
        409,
        "agent messages need the Radient OAuth sign-in, not an application key",
    ),
    "owner_mismatch": (
        409,
        "the signed-in account is not the account this machine enrolled for agent messages",
    ),
    "owner_changed": (
        409,
        "the local agent-messages owner changed while this read was in flight; try again",
    ),
    "credential_changed": (
        409,
        "the sign-in behind this read changed while it was in flight; refresh and try again",
    ),
    "consent_revoked": (
        409,
        "this machine no longer has its agent-messages consent record; enable the feature again",
    ),
    "stale_generation": (
        409,
        "the local agent-messages owner was re-enrolled; refresh and try again",
    ),
    "no_destination_evidence": (
        502,
        "the agent-messages hub this machine would read could not be verified",
    ),
    "destination_mismatch": (
        409,
        "the configured agent-messages hub is not the one this machine enrolled with",
    ),
    "bad_limit": (422, "limit must be between 1 and 100"),
    "malformed_page": (
        502,
        "the agent-messages hub answered in a shape this build cannot read",
    ),
    "wrong_tenant": (
        409,
        "the answer carried a message from another organization's namespace",
    ),
    "wrong_recipient": (
        409,
        "the answer carried a message addressed to another endpoint",
    ),
    "invalid_request": (422, INVALID_READ_QUERY),
}


def _raise_read(code: str) -> NoReturn:
    """One read-path refusal, by its CLOSED table entry.

    An unlisted code is a 502 with a generic sentence rather than a leak: the
    vocabulary is closed, so a code that is not in the table is a bug here, and
    guessing a status for it is how the table stops being one.
    """

    status, message = _READ_REFUSALS.get(
        code, (502, "the agent messages read could not be completed")
    )
    raise HTTPException(status, {"code": code, "message": message})


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

    # The read family's availability receipt lives with the read family, and the
    # import stays inside the handler: the read module reaches the client stack,
    # which must never ride ``lop serve``'s boot path (module docstring).
    from local_operator.team_bridge.read_context import READ_AVAILABILITY
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
        # The slice's availability receipt (spec M-11 / §7), NEXT TO the capability
        # so one answer drives the surface decision: the capability says which read
        # shape a renderer is talking to, this says which read-family operations
        # exist on this build. M0 ships the metadata inbox; each other flag turns
        # true with the operation it names, never ahead of it.
        "availability": dict(READ_AVAILABILITY),
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


# -- the READ slice: agent messages, METADATA ONLY (RFC §6.1 reads) -------------


#: Every query name this read accepts. Anything else is REFUSED rather than
#: dropped (review Q5): ``TeamBridgeInboxQuery``'s ``extra=forbid`` is about the
#: model, and FastAPI DROPS unknown query names silently — which is how a caller's
#: typo (``threadId``) quietly becomes a wider read than the one they asked for.
READ_QUERY_PARAMS = frozenset(
    {
        "expected_account_id",
        "tenant_id",
        "after",
        "limit",
        "state",
        "thread_id",
        "sender_account_id",
    }
)


@router.get("/v1/desktop/team-bridge/inbox", response_model=CRUDResponse[Any])
async def get_inbox(
    request: Request,
    # Plain optional STRINGS, deliberately (review R-6): every bound is enforced by
    # ``TeamBridgeInboxQuery`` below, because a ``Query(...)`` bound would let
    # FastAPI answer 422 with the caller's own value echoed back in the body.
    expected_account_id: str | None = Query(default=None),
    tenant_id: str | None = Query(default=None),
    after: str | None = Query(default=None),
    limit: str | None = Query(default=None),
    state: str | None = Query(default=None),
    thread_id: str | None = Query(default=None),
    sender_account_id: str | None = Query(default=None),
    config_manager: ConfigManager = Depends(get_config_manager),
) -> CRUDResponse[Any]:
    """One page of this machine's agent messages: METADATA ONLY, zero writes.

    READ means what the person can see, never what the machine may receive: the
    answer is ids, delivery state, hashes and this machine's coarse screening
    state. A message BODY is not in it — returning one needs the shared content
    predicate (owner standing, a validated envelope, a native receipt bound to
    the fetched bytes, its currency rules and a confirmed read intent), which is
    not implemented, and a route that returned bytes without it would serve
    unreviewed third-party text to a renderer.

    NO BRIDGE-STORE WRITES, deliberately and structurally: the connector owns the
    inbox checkpoint and the walk terminal, and a UI page that moved them would
    make the person's browsing change whether their agent gets woken
    (``pull.inbox``'s ``merge_page`` is exactly that write). Nothing here merges
    refs either, and the store is opened READ-ONLY — this route never creates,
    migrates or touches ``team_bridge/state.sqlite3``, so a machine with nothing
    enrolled answers without a database being made for it (review R-1/M-9).
    (Reading the signed-in credential row can still create the SHARED ``auth.db``
    side files, exactly as ``/status`` does; that store is not this surface's, and
    the invariant here is the bridge store.) Unknown query names are refused, not
    dropped — see :data:`READ_QUERY_PARAMS`.

    Two standing modes answer honestly through ``mode``/``read_only``: ENABLED
    machines read under receive standing, and a machine whose local switch is off
    reads retained history for the same authenticated participant — reads only,
    never a receive, and a machine with no retained namespace for the signed-in
    account is refused rather than guessed at.

    It does not screen, ACK, wake, claim, or decide, and it never advertises a
    decision surface: the read capability key is its own
    (``team_bridge_read``), so a renderer that cannot see it says "update the
    backend" instead of probing operations that do not exist here.
    """

    from pydantic import ValidationError

    from local_operator.team_bridge.client import BridgeRefusal, BridgeSignInRequired
    from local_operator.team_bridge.read_context import (
        ReadRefused,
        inbox_view,
        read_inbox_page,
        resolve_read_context,
        wire_for,
    )

    if set(request.query_params) - READ_QUERY_PARAMS:
        _raise_read("invalid_request")

    raw: dict[str, Any] = {
        "expected_account_id": expected_account_id,
        "tenant_id": tenant_id,
        "after": after,
        "limit": limit,
        "state": state,
        "thread_id": thread_id,
        "sender_account_id": sender_account_id,
    }
    try:
        # Only the names the caller actually SENT: a None must not overwrite a
        # model default (the model is where every bound and closed set lives).
        query = TeamBridgeInboxQuery(
            **{name: value for name, value in raw.items() if value is not None}
        )
    except ValidationError:
        _raise_read("invalid_request")

    # ONE wire per request, handed to BOTH builders (review R-2): the standing
    # resolution and the page must not construct their clients by two rules.
    wire = wire_for(config_manager.config_dir)
    try:
        context = await resolve_read_context(
            config_manager.config_dir,
            expected_account_id=query.expected_account_id,
            wire=wire,
        )
    except BridgeSignInRequired as exc:
        # Before the BridgeRefusal arm where the two overlap, deliberately.
        _raise_sign_in(exc)
    except BridgeRefusal as exc:
        _raise_bridge(exc)
    except ReadRefused as exc:
        _raise_read(exc.code)

    if query.tenant_id != context.tenant_id:
        # The named organization IS the scope: the endpoint a page resolves
        # through is the membership the caller names, so answering for a tenant
        # this machine did not enroll with would be a cross-tenant read dressed
        # as a filter. The upstream refuses the same shape (`invalid_tenant`).
        _raise_read("invalid_tenant")

    try:
        result = await read_inbox_page(
            context,
            config_dir=config_manager.config_dir,
            wire=wire,
            after=query.after,
            limit=query.limit,
            state=query.state,
            thread_id=query.thread_id,
            sender_account_id=query.sender_account_id,
        )
    except BridgeSignInRequired as exc:
        _raise_sign_in(exc)
    except BridgeRefusal as exc:
        _raise_bridge(exc)
    except ReadRefused as exc:
        _raise_read(exc.code)
    return reply(inbox_view(result))
