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

#: The server mints conversation ids as ``bcv_`` + hex; validating the grammar
#: BEFORE interpolating an id into an upstream path is the pair lane's rule
#: (review S-RSA-5) and applies unchanged here.
_CONVERSATION_ID = re.compile(r"^bcv_[0-9a-z]{8,64}$")
#: The conversation domain's other two interpolated path/query values. Both are
#: opaque server-minted strings; this route family only has to prove they
#: cannot change the SHAPE of the upstream URL (`..`, `/`, `?`, `%`), so the
#: rule is a conservative charset rather than a re-implementation of the
#: server's mint.
_MESSAGE_ID = re.compile(r"^bcmsg_[0-9a-z]{8,64}$")
#: Mirrors ``read_context.ACCOUNT_ID_PATTERN``. Spelled here rather than imported
#: because that module rides the httpx boot path this file must not pull at
#: module scope; ``test_account_grammar_matches_the_reader`` pins the two equal.
_ACCOUNT_ID = re.compile(r"^acc_[A-Za-z0-9_-]{1,64}$")


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


class TeamBridgeMessageQuery(_Strict):
    """The single-message read query (M1a): the captured account and the org.

    No id rides the query: the message id is a PATH segment, and it is validated
    by its own grammar before any I/O (a malformed one is the family's 404, not a
    422 — §2). These two names are the same captured-account receipt and org scope
    the page carries, so a detail asked for under another account or another org is
    refused before a request is built.
    """

    expected_account_id: str = Field(min_length=1, max_length=128)
    tenant_id: str = Field(min_length=1, max_length=128)

    @model_validator(mode="after")
    def _grammar(self) -> "TeamBridgeMessageQuery":
        from local_operator.team_bridge.read_context import (
            ACCOUNT_ID_PATTERN,
            TENANT_ID_PATTERN,
        )

        checks = (
            (self.expected_account_id, ACCOUNT_ID_PATTERN, "expected_account_id"),
            (self.tenant_id, TENANT_ID_PATTERN, "tenant_id"),
        )
        for value, pattern, name in checks:
            # ``fullmatch``, not ``match`` (review R-5): these patterns are
            # anchored ``^…$``, and ``$`` also matches BEFORE a trailing newline,
            # so a value carrying one would reach the transport's path or query.
            if re.fullmatch(pattern, value) is None:
                raise ValueError(f"{name} is outside the accepted grammar")
        return self


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
    # The ONE 404 of this family (spec §2): a message id outside the closed
    # grammar, or the upstream's single-coded 404 for a message this account is
    # not a participant of. The sentence is the spec's own fixed text — a
    # malformed id and an absent message are deliberately indistinguishable, so
    # this path cannot be probed for which ids exist.
    "message_not_found": (404, "Message not found."),
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
    request: Request,
    body: TeamBridgeSettingsPut,
    config_manager: ConfigManager = Depends(get_config_manager),
) -> CRUDResponse[Any]:
    """State changes and the consented enable (§5.2).

    ``409 consent_version_unknown`` additionally invalidates the shared status
    cache before it is surfaced: the client's move is §5.2's "refresh status and
    re-render", and re-serving the cached (stale) consent version would make the
    next PUT fail the same way.
    """

    _refuse_write_scope(request)
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
    request: Request,
    body: TeamBridgeAllowlistPut,
    config_manager: ConfigManager = Depends(get_config_manager),
) -> CRUDResponse[Any]:
    """The bounded replace-set write (§6.1/§9). ``entries`` is whole-policy."""

    _refuse_write_scope(request)
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

#: The single-message read accepts these names and nothing else, by the same rule:
#: a filter the upstream would ignore must not be silently dropped, because the
#: caller believes it narrowed the read.
READ_DETAIL_QUERY_PARAMS = frozenset({"expected_account_id", "tenant_id"})


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
    """One page of this machine's agent messages: METADATA ONLY, logically read-only.

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
    the invariant here is the bridge store.) The claim is the LOGICAL one (spec
    M-10) and is stated rather than summed up: no table row and no main-file byte
    of ``state.sqlite3`` changes, while SQLite's own ``-wal``/``-shm`` side files
    DO appear when the store is in WAL mode — a reader needs them, they are not
    this route's write, and a whole-tree hash therefore moves where the table
    digest does not (review Q-3). Unknown query names are refused, not
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


@router.get("/v1/desktop/team-bridge/messages/{message_id}", response_model=CRUDResponse[Any])
async def get_message(
    message_id: str,
    request: Request,
    # Plain optional STRINGS for the same reason as the page (review R-6): a
    # ``Query(...)`` bound answers 422 with the caller's own value echoed back in
    # the body, and this family never quotes caller input.
    expected_account_id: str | None = Query(default=None),
    tenant_id: str | None = Query(default=None),
    config_manager: ConfigManager = Depends(get_config_manager),
) -> CRUDResponse[Any]:
    """ONE agent message: metadata always, the BODY only when screening admits it.

    The body is served through :func:`~local_operator.team_bridge.read_context.
    admit_read_content` and nothing else: the fetched envelope must bind to this
    owner and hash to what the server said it sent, a native receipt must still
    be the policy's and this enrollment's, its recomputed body hash and envelope
    digest must equal the receipt's, and a CONFIRMED read intent must carry that
    exact tuple. A message that fails ANY term — unreviewed, held, blocked by the
    native bounce, not yet read upstream, hash-mismatched — answers its metadata
    with ``content: null``; it is never an error, and the screener's own reason is
    never read, let alone returned.

    The message id is validated by grammar BEFORE any upstream path is built
    (review S-RSA-5) and a value outside it is this family's 404, so a caller
    cannot use a malformed id to make the transport build a path of its choosing.
    The whole read is logically read-only in the sense the store's read-only
    accessors define (M-10, review Q-3): one read-only proof snapshot, no
    ``_init``, no migration, no row and no main-file byte of
    ``team_bridge/state.sqlite3`` changed, no ``merge_refs``, no checkpoint, no
    ACK intent, no screening call, and no database created to answer for a
    machine that has none. SQLite's own ``-wal``/``-shm`` side files are the one
    exception, and they are declared: a reader of a WAL store needs them.
    """

    from pydantic import ValidationError

    from local_operator.team_bridge.client import BridgeRefusal, BridgeSignInRequired
    from local_operator.team_bridge.read_context import (
        MESSAGE_ID_PATTERN,
        ReadRefused,
        detail_view,
        read_message_detail,
        resolve_read_context,
        wire_for,
    )

    if set(request.query_params) - READ_DETAIL_QUERY_PARAMS:
        _raise_read("invalid_request")
    # BEFORE any I/O, and deliberately indistinguishable from an absent message:
    # the id is interpolated into the upstream path, and a replacement for
    # ``validate_envelope``'s own id check it is not — this one runs before the
    # transport is allowed to build anything at all.
    if re.fullmatch(MESSAGE_ID_PATTERN, message_id) is None:
        _raise_read("message_not_found")
    raw: dict[str, Any] = {
        "expected_account_id": expected_account_id,
        "tenant_id": tenant_id,
    }
    try:
        query = TeamBridgeMessageQuery(
            **{name: value for name, value in raw.items() if value is not None}
        )
    except ValidationError:
        _raise_read("invalid_request")

    # ONE wire per request, by the same rule the page uses (review R-2).
    wire = wire_for(config_manager.config_dir)
    try:
        context = await resolve_read_context(
            config_manager.config_dir,
            expected_account_id=query.expected_account_id,
            wire=wire,
        )
    except BridgeSignInRequired as exc:
        _raise_sign_in(exc)
    except BridgeRefusal as exc:
        _raise_bridge(exc)
    except ReadRefused as exc:
        _raise_read(exc.code)

    if query.tenant_id != context.tenant_id:
        # The named organization IS the scope, exactly as on the page: answering
        # for a tenant this machine did not enroll with would be a cross-tenant
        # read dressed as a filter.
        _raise_read("invalid_tenant")

    try:
        detail = await read_message_detail(
            context,
            config_dir=config_manager.config_dir,
            message_id=message_id,
            tenant_id=query.tenant_id,
            wire=wire,
        )
    except BridgeSignInRequired as exc:
        _raise_sign_in(exc)
    except BridgeRefusal as exc:
        _raise_bridge(exc)
    except ReadRefused as exc:
        _raise_read(exc.code)
    return reply(detail_view(detail))


# ---------------------------------------------------------------------------
# Conversations (docs/TEAM-BRIDGE-CONVERSATIONS.md)
# ---------------------------------------------------------------------------
#
# These routes are PASS-THROUGH by design. Membership, the human-only rule,
# agent consent/joined_seq, the screening fetch/admit split, unread counters and
# every authorization decision live on the server; duplicating any of them here
# would create a second, subtly different policy and a second place to get
# tenant scoping wrong. What this plane adds is the desktop contract: strict
# bodies, the shared refusal mapping, and the same no-store/no-body-cache rules
# as the pair lane — an answer is returned to the renderer and never persisted.
#
# A body cached here would break team_bridge/store.py's §13.9 rule, so there is
# deliberately no store call in this section: conversations are read live.


class ConversationMemberIn(_Strict):
    account_id: str = Field(min_length=1, max_length=128)
    kind: Literal["human", "agent"] = "human"
    endpoint_id: str | None = Field(default=None, max_length=128)
    display_label: str = Field(min_length=1, max_length=64)


class ConversationCreateIn(_Strict):
    tenant_id: str | None = Field(default=None, max_length=128)
    kind: Literal["dm", "group", "channel"]
    name: str | None = Field(default=None, max_length=128)
    topic: str | None = Field(default=None, max_length=512)
    visibility: Literal["private", "tenant"] | None = None
    agent_capable: bool = False
    members: list[ConversationMemberIn] = Field(default_factory=list, max_length=500)


class ConversationSendIn(_Strict):
    tenant_id: str | None = Field(default=None, max_length=128)
    client_ref: str = Field(min_length=1, max_length=128)
    body: str = Field(min_length=1)
    agent_label: str | None = Field(default=None, max_length=64)


class ConversationReadIn(_Strict):
    tenant_id: str | None = Field(default=None, max_length=128)
    seq: int | None = Field(default=None, ge=0)
    muted: bool | None = None

    @model_validator(mode="after")
    def _something_to_do(self) -> "ConversationReadIn":
        if self.seq is None and self.muted is None:
            raise ValueError("a read write needs seq, muted, or both")
        return self


class ConversationConsentIn(_Strict):
    tenant_id: str | None = Field(default=None, max_length=128)
    decision: Literal["accept", "decline"]
    consent_version: str = Field(min_length=1, max_length=128)


class ConversationAdmitIn(_Strict):
    tenant_id: str | None = Field(default=None, max_length=128)
    verdict: Literal["allow", "reject"]
    body_hash: str = Field(min_length=1, max_length=128)


def _conversation_client(config_manager: ConfigManager):
    """The transport, imported INSIDE the handler (the httpx boot rule)."""

    from local_operator.team_bridge.client import TeamBridgeClient

    return TeamBridgeClient(config_manager.config_dir)


def _require_conversation_id(conversation_id: str) -> None:
    """Refuse a malformed conversation id BEFORE it is interpolated upstream.

    The id comes from the path and the client builds the upstream path from it,
    so this is the pair lane's review S-RSA-5 rule applied to the new family: a
    value outside the minted grammar is this route family's 404/400, never a
    path of the caller's choosing. One helper, so a new route cannot forget it.
    """

    if _CONVERSATION_ID.fullmatch(conversation_id) is None:
        _raise_read("invalid_request")


def _require_message_id(message_id: str) -> None:
    """Refuse a message id that could reshape the upstream path. See above."""

    if _MESSAGE_ID.fullmatch(message_id) is None:
        _raise_read("invalid_request")


def _require_account_id(account_id: str) -> None:
    """Refuse an account id carrying URL syntax. See above."""

    if _ACCOUNT_ID.fullmatch(account_id) is None:
        _raise_read("invalid_request")


async def _conversation_reply(awaitable: Any) -> Any:
    """Map the transport's two refusal families, exactly like the pair routes."""

    from local_operator.team_bridge.client import BridgeRefusal, BridgeSignInRequired

    try:
        return await awaitable
    except BridgeSignInRequired as exc:
        _raise_sign_in(exc)
    except BridgeRefusal as exc:
        _raise_bridge(exc)


#: The desktop's ONE accepted multi-organization scope for the two reads. The
#: organization set is SERVER-DERIVED from the signed-in account's active
#: memberships, and eligibility is applied per org by the same resolver the pair
#: lane uses; the renderer therefore never supplies a tenant list (the design
#: note's tenant-isolation invariant) and never gets to name a second scope.
ACCOUNT_SCOPE = "account"


def _refuse_write_scope(request: Request) -> None:
    """Writes never take a scope: each names exactly ONE tenant.

    An account-wide read is an aggregate over memberships; a write cannot be
    aggregated, because the server would have to guess which organization the
    send/read/join was meant to land in. That is the "communicate only in the
    expected space" invariant, so ``scope`` is refused here before any I/O. The
    check is the same closed-table pattern the reads use, over an empty accepted
    set: it names ONLY ``scope``, so a write keeps whatever other query names its
    own route already accepts (``endpoint_id`` on admit) without this becoming a
    second table.
    """

    if "scope" in request.query_params:
        _raise_read("invalid_request")


@router.get("/v1/desktop/team-bridge/conversations", response_model=CRUDResponse[Any])
async def list_conversations(
    request: Request,
    tenant_id: str | None = Query(default=None),
    limit: str | None = Query(default=None),
    before: str | None = Query(default=None),
    cursor: str | None = Query(default=None),
    scope: str | None = Query(default=None),
    config_manager: ConfigManager = Depends(get_config_manager),
) -> CRUDResponse[Any]:
    """One page, tenant-scoped or (with ``scope=account``) merged across orgs.

    The account page is ordered by activity across the caller's organizations and
    carries each row's own ``tenant_id``; conversations are never merged into one
    row. Its position is the single opaque ``cursor`` the server mints, so a
    per-tenant ``before`` or a ``tenant_id`` beside it would name a conflicting
    scope and is refused here rather than being resolved by the server.
    """

    if set(request.query_params) - {
        "tenant_id",
        "limit",
        "before",
        "cursor",
        "scope",
    }:
        _raise_read("invalid_request")
    account_wide = scope == ACCOUNT_SCOPE
    if scope is not None and not account_wide:
        _raise_read("invalid_request")
    if account_wide:
        # ``after`` is not a name of this read at all, so the closed table above
        # refuses it on every page; these two are the page-specific conflicts.
        if tenant_id is not None or before is not None:
            _raise_read("invalid_request")
    elif cursor is not None:
        # The merged page pages on ``cursor``; a tenant page pages on ``before``.
        # Accepting both would leave it to the server which one wins.
        _raise_read("invalid_request")
    parsed_limit: int | None = None
    if limit is not None:
        try:
            parsed_limit = int(limit)
        except ValueError:
            _raise_read("invalid_request")
    client = _conversation_client(config_manager)
    payload = await _conversation_reply(
        client.conversations(
            tenant_id=tenant_id,
            limit=parsed_limit,
            before=before,
            cursor=cursor,
            scope=scope,
        )
    )
    return reply(payload)


@router.post("/v1/desktop/team-bridge/conversations", response_model=CRUDResponse[Any])
async def create_conversation(
    request: Request,
    body: ConversationCreateIn,
    config_manager: ConfigManager = Depends(get_config_manager),
) -> CRUDResponse[Any]:
    """Create (or find, for a dm) a conversation. The payload is the wire shape."""

    _refuse_write_scope(request)
    client = _conversation_client(config_manager)
    payload = {
        key: value
        for key, value in body.model_dump().items()
        if value is not None and not (key == "members" and not value)
    }
    payload["members"] = [member.model_dump(exclude_none=True) for member in body.members]
    result = await _conversation_reply(client.create_conversation(payload))
    return reply(result)


@router.get(
    "/v1/desktop/team-bridge/conversations/{conversation_id}",
    response_model=CRUDResponse[Any],
)
async def get_conversation(
    conversation_id: str,
    tenant_id: str | None = Query(default=None),
    config_manager: ConfigManager = Depends(get_config_manager),
) -> CRUDResponse[Any]:
    """One conversation: members, read state and latest seq. No body, ever."""
    _require_conversation_id(conversation_id)
    client = _conversation_client(config_manager)
    return reply(
        await _conversation_reply(client.conversation(conversation_id, tenant_id=tenant_id))
    )


# Archive and unarchive are TWO static paths, deliberately not one
# ``{action}`` route: a dynamic segment there matched ``/messages`` and made
# every send a 422 (found by the route test). A closed path is also the guard
# the pair-lane reader relies on — the id grammar is checked before any I/O.
@router.post(
    "/v1/desktop/team-bridge/conversations/{conversation_id}/archive",
    response_model=CRUDResponse[Any],
)
async def conversation_archive(
    conversation_id: str,
    request: Request,
    config_manager: ConfigManager = Depends(get_config_manager),
) -> CRUDResponse[Any]:
    """Archive (read-only for every member)."""
    _require_conversation_id(conversation_id)
    _refuse_write_scope(request)
    client = _conversation_client(config_manager)
    return reply(
        await _conversation_reply(client.conversation_lifecycle(conversation_id, "archive"))
    )


@router.post(
    "/v1/desktop/team-bridge/conversations/{conversation_id}/unarchive",
    response_model=CRUDResponse[Any],
)
async def conversation_unarchive(
    conversation_id: str,
    request: Request,
    config_manager: ConfigManager = Depends(get_config_manager),
) -> CRUDResponse[Any]:
    """Unarchive (owner only, server-enforced)."""
    _require_conversation_id(conversation_id)
    _refuse_write_scope(request)
    client = _conversation_client(config_manager)
    return reply(
        await _conversation_reply(client.conversation_lifecycle(conversation_id, "unarchive"))
    )


@router.post(
    "/v1/desktop/team-bridge/conversations/{conversation_id}/members",
    response_model=CRUDResponse[Any],
)
async def add_conversation_members(
    conversation_id: str,
    request: Request,
    members: list[ConversationMemberIn],
    tenant_id: str | None = Query(default=None),
    config_manager: ConfigManager = Depends(get_config_manager),
) -> CRUDResponse[Any]:
    """Invite members. Humans are invited; agents stay invited until their owner consents."""
    _require_conversation_id(conversation_id)
    _refuse_write_scope(request)

    client = _conversation_client(config_manager)
    payload = [member.model_dump(exclude_none=True) for member in members]
    return reply(
        await _conversation_reply(
            client.conversation_members(conversation_id, payload, tenant_id=tenant_id)
        )
    )


@router.delete(
    "/v1/desktop/team-bridge/conversations/{conversation_id}/members/{account_id}",
    response_model=CRUDResponse[Any],
)
async def remove_conversation_member(
    conversation_id: str,
    account_id: str,
    request: Request,
    config_manager: ConfigManager = Depends(get_config_manager),
) -> CRUDResponse[Any]:
    """Remove a human member (owner only; the server owns succession)."""
    _require_conversation_id(conversation_id)
    _require_account_id(account_id)
    _refuse_write_scope(request)

    client = _conversation_client(config_manager)
    return reply(
        await _conversation_reply(client.conversation_remove_member(conversation_id, account_id))
    )


@router.post(
    "/v1/desktop/team-bridge/conversations/{conversation_id}/members/{account_id}/consent",
    response_model=CRUDResponse[Any],
)
async def conversation_consent(
    conversation_id: str,
    account_id: str,
    request: Request,
    body: ConversationConsentIn,
    config_manager: ConfigManager = Depends(get_config_manager),
) -> CRUDResponse[Any]:
    """Record an agent owner's participation decision (the agent only then becomes active)."""
    _require_conversation_id(conversation_id)
    _require_account_id(account_id)
    _refuse_write_scope(request)

    client = _conversation_client(config_manager)
    return reply(
        await _conversation_reply(
            client.conversation_consent(
                conversation_id,
                account_id,
                decision=body.decision,
                consent_version=body.consent_version,
                tenant_id=body.tenant_id,
            )
        )
    )


@router.post(
    "/v1/desktop/team-bridge/conversations/{conversation_id}/join",
    response_model=CRUDResponse[Any],
)
async def conversation_join(
    conversation_id: str,
    request: Request,
    config_manager: ConfigManager = Depends(get_config_manager),
) -> CRUDResponse[Any]:
    """Accept an invitation, or self-join a tenant-visible channel."""
    _require_conversation_id(conversation_id)
    _refuse_write_scope(request)

    client = _conversation_client(config_manager)
    return reply(await _conversation_reply(client.conversation_join(conversation_id)))


@router.post(
    "/v1/desktop/team-bridge/conversations/{conversation_id}/leave",
    response_model=CRUDResponse[Any],
)
async def conversation_leave(
    conversation_id: str,
    request: Request,
    config_manager: ConfigManager = Depends(get_config_manager),
) -> CRUDResponse[Any]:
    """Leave. Leaving is the leave/re-add boundary: re-entry needs a fresh invite."""
    _require_conversation_id(conversation_id)
    _refuse_write_scope(request)

    client = _conversation_client(config_manager)
    return reply(await _conversation_reply(client.conversation_leave(conversation_id)))


@router.get(
    "/v1/desktop/team-bridge/conversations/{conversation_id}/messages",
    response_model=CRUDResponse[Any],
)
async def conversation_history(
    conversation_id: str,
    request: Request,
    tenant_id: str | None = Query(default=None),
    limit: str | None = Query(default=None),
    before_seq: str | None = Query(default=None),
    after_seq: str | None = Query(default=None),
    endpoint_id: str | None = Query(default=None),
    config_manager: ConfigManager = Depends(get_config_manager),
) -> CRUDResponse[Any]:
    """One history page. Bodies are present only where the server's rule allows them."""

    _require_conversation_id(conversation_id)
    if set(request.query_params) - {
        "tenant_id",
        "limit",
        "before_seq",
        "after_seq",
        "endpoint_id",
    }:
        _raise_read("invalid_request")
    if before_seq and after_seq:
        _raise_read("invalid_request")
    parsed_limit: int | None = None
    if limit is not None:
        try:
            parsed_limit = int(limit)
        except ValueError:
            _raise_read("invalid_request")
    # The two cursors are parsed, not interpolated: `int("x")` raises and an
    # unwrapped raise is a 500 on a caller-controlled value. The server answers
    # invalid_request for the same shape, so both ends agree on the refusal.
    parsed_before: int | None = None
    parsed_after: int | None = None
    try:
        parsed_before = int(before_seq) if before_seq else None
        parsed_after = int(after_seq) if after_seq else None
    except ValueError:
        _raise_read("invalid_request")
    if (parsed_before is not None and parsed_before < 0) or (
        parsed_after is not None and parsed_after < 0
    ):
        _raise_read("invalid_request")
    client = _conversation_client(config_manager)
    return reply(
        await _conversation_reply(
            client.conversation_messages(
                conversation_id,
                tenant_id=tenant_id,
                limit=parsed_limit,
                before_seq=parsed_before,
                after_seq=parsed_after,
                endpoint_id=endpoint_id,
            )
        )
    )


@router.post(
    "/v1/desktop/team-bridge/conversations/{conversation_id}/messages",
    response_model=CRUDResponse[Any],
)
async def send_conversation_message(
    conversation_id: str,
    request: Request,
    body: ConversationSendIn,
    config_manager: ConfigManager = Depends(get_config_manager),
) -> CRUDResponse[Any]:
    """Append one message. The body is returned to the renderer and never stored here."""
    _require_conversation_id(conversation_id)
    _refuse_write_scope(request)

    client = _conversation_client(config_manager)
    return reply(
        await _conversation_reply(
            client.conversation_send(
                conversation_id,
                client_ref=body.client_ref,
                text=body.body,
                tenant_id=body.tenant_id,
                agent_label=body.agent_label,
            )
        )
    )


@router.get(
    "/v1/desktop/team-bridge/conversations/{conversation_id}/messages/{message_id}/body",
    response_model=CRUDResponse[Any],
)
async def conversation_message_body(
    conversation_id: str,
    message_id: str,
    tenant_id: str | None = Query(default=None),
    endpoint_id: str | None = Query(default=None),
    config_manager: ConfigManager = Depends(get_config_manager),
) -> CRUDResponse[Any]:
    """The agent classification fetch: body + binding hash. Not a context path."""
    _require_conversation_id(conversation_id)
    _require_message_id(message_id)

    client = _conversation_client(config_manager)
    return reply(
        await _conversation_reply(
            client.conversation_body(
                conversation_id, message_id, tenant_id=tenant_id, endpoint_id=endpoint_id
            )
        )
    )


@router.post(
    "/v1/desktop/team-bridge/conversations/{conversation_id}/messages/{message_id}/admit",
    response_model=CRUDResponse[Any],
)
async def conversation_admit(
    conversation_id: str,
    message_id: str,
    request: Request,
    body: ConversationAdmitIn,
    endpoint_id: str | None = Query(default=None),
    config_manager: ConfigManager = Depends(get_config_manager),
) -> CRUDResponse[Any]:
    """Record the harness's screening verdict. A reject is terminal on the server."""
    _require_conversation_id(conversation_id)
    _require_message_id(message_id)
    _refuse_write_scope(request)

    client = _conversation_client(config_manager)
    return reply(
        await _conversation_reply(
            client.conversation_admit(
                conversation_id,
                message_id,
                verdict=body.verdict,
                body_hash=body.body_hash,
                tenant_id=body.tenant_id,
                endpoint_id=endpoint_id,
            )
        )
    )


@router.post(
    "/v1/desktop/team-bridge/conversations/{conversation_id}/read",
    response_model=CRUDResponse[Any],
)
async def conversation_read(
    conversation_id: str,
    request: Request,
    body: ConversationReadIn,
    config_manager: ConfigManager = Depends(get_config_manager),
) -> CRUDResponse[Any]:
    """Advance the read cursor and/or mute; the answer is the post-state."""
    _require_conversation_id(conversation_id)
    _refuse_write_scope(request)

    client = _conversation_client(config_manager)
    return reply(
        await _conversation_reply(
            client.conversation_read(
                conversation_id, seq=body.seq, muted=body.muted, tenant_id=body.tenant_id
            )
        )
    )


@router.get("/v1/desktop/team-bridge/unread", response_model=CRUDResponse[Any])
async def conversation_unread(
    request: Request,
    tenant_id: str | None = Query(default=None),
    scope: str | None = Query(default=None),
    config_manager: ConfigManager = Depends(get_config_manager),
) -> CRUDResponse[Any]:
    """The ONE badge read: per-conversation counts plus the unmuted total.

    With ``scope=account`` the server iterates the caller's active memberships
    and answers ``{total, organizations: [...]}`` in the same one authorized
    read, so the badge never fans out into N per-org calls. A ``tenant_id``
    beside it is refused: the aggregate's organization set is the server's to
    derive, not the renderer's.
    """

    if set(request.query_params) - {"tenant_id", "scope"}:
        _raise_read("invalid_request")
    account_wide = scope == ACCOUNT_SCOPE
    if scope is not None and not account_wide:
        _raise_read("invalid_request")
    if account_wide and tenant_id is not None:
        _raise_read("invalid_request")
    client = _conversation_client(config_manager)
    return reply(await _conversation_reply(client.unread(tenant_id=tenant_id, scope=scope)))
