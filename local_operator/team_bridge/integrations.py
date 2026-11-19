"""Radient Team Integrations — the agent-driven provisioning plane (RFC §8, Phase 3a).

WHAT THIS MODULE IS. The harness half of "set up Alertmanager on #alerts-prod
for me": the HTTP calls to agent-server's organization admin API
(``/v1/tenants/:tenantid/integrations/…``, agent-server PR #401), the mapping of
its flat ``{error, code, details}`` refusals onto the RFC's five user states
(§8.2 U5), the host-built consent ask each widening mutation needs (§8.2 C4),
and the sealing of a show-once credential into the local secret store (§8.2
S-84). The ``integrations`` tool (``tools/integrations_tool.py``) is a thin
dispatcher over this; the consent seam the session implements lives on
``ToolContext.integration_consent``.

WHY EVERY WIDENING MUTATION NEEDS A HUMAN-ANSWERED ASK. A prompt-injected local
session (threat TA6) must not be able to publish an app into ``#alerts-prod`` or
mint a bearer credential on its own. The server enforces half of that: for a
Local Operator caller (JWT ``aud`` in ``BRIDGE_LOP_AUDIENCES``) a publish, mint,
rotate, hook create or install carries a single-use, 15-minute
``confirmation_commitment`` bound to ``(tenant, app, revision, action, target,
payload hash, preparing account, surface)`` plus an ``ask_ref``. The harness
enforces the other half: the commitment is only ever sent after a PERSON
answered the host-filed ask that froze that exact commitment with its
affirmative option, in this run, and the answer has not been spent before.
"Answered in chat", a timeout, a decline, a dismissal, a typed reply and a
model-authored ask are all refused — the model cannot construct consent
(``AskQueue.enqueue(consent=…)`` is host-only and the ``ask`` tool never passes
it).

WHY THE RAW CREDENTIAL NEVER LEAVES THIS MODULE. A tool result is persisted in
the transcript and replayed to the provider on every later turn; a log line
ships to whatever collects logs. The ``rdi_at_…`` / ``rdi_hk_…`` value the
server answers ONCE is therefore taken from the response dict here, written
straight into ``SecretStore`` and registered with the runtime redaction ledger,
and the dict entry is deleted before anything else can read the response. What
travels on is metadata only: secret name, ids, the server's ``prefix_hint``,
length and a short SHA-256 fingerprint (a fingerprint lets a human match two
copies without either being shown).

WHY THE SECRET NAME IS PREFLIGHTED. ``SecretStore.set`` refuses an existing
name (``SecretExists``) — correctly, because the store keeps no previous value.
If that refusal came AFTER the server minted, the show-once value would be lost
with a live credential left behind on the server. So the name is checked before
the draft and again immediately before the minting call; only an explicit
rotate writes with ``update``.

Nothing here is a vault; see ``guide://credentials`` for what the store defends.
"""

from __future__ import annotations

import hashlib
import json
import logging
import re
from dataclasses import dataclass, field
from typing import Any, Mapping, Sequence

from local_operator.team_bridge.client import (
    BridgeRefusal,
    BridgeSignInRequired,
    TeamBridgeClient,
    _envelope_result,
    _refusal,
)

logger = logging.getLogger(__name__)

#: The ask question id every integration consent ask carries. The consent
#: record persists the affirm/decline LABELS it was filed with, so a language
#: change between asking and answering cannot flip an answer (the hub-pull rule).
QUESTION_ID = "team_integration"

#: The consent ``op`` the session's seam matches on. Distinct from
#: ``hub_pull`` so neither grant can ever be spent on the other's action.
CONSENT_OP = "team_integration"

#: The ask's timeout equals the server's commitment TTL
#: (agent-server ``integrationCommitmentTTL = 15 * time.Minute``,
#: internal/services/integration_service.go). A yes that arrives after the
#: commitment died could only be answered with ``409 commitment_already_used``,
#: so the card must not outlive what it authorizes.
ASK_TIMEOUT_S = 15 * 60

#: Where a Personal / lapsed plan is sent to start the Team plan or its free
#: trial (RFC §8.2 U5 state 1, §8.3 item 1).
BILLING_URL = "https://console.radienthq.com/dashboard/billing"

#: The closed v1 scope vocabulary (RFC §4.2, CR-2). Checked locally so an
#: unknown scope is refused before any request, matching the server's
#: ``400 invalid_scope``.
SCOPES: tuple[str, ...] = ("alerts:ingest", "chat:write", "chat:write.thread")

#: Sources (wire field ``vendor``) the server accepts for an automation.
HOOK_VENDORS: tuple[str, ...] = ("datadog", "grafana", "alertmanager", "incidentio", "generic")

#: Show-once credential prefixes (RFC §4.1). Sealing refuses anything else: a
#: value of an unexpected shape is a protocol surprise, not something to store.
SEALABLE_PREFIXES: tuple[str, ...] = ("rdi_at_", "rdi_hk_", "rdi_whs_", "whsec_")

#: Ids are server-generated ``<prefix>_<32 hex>`` (RFC §4.1). Validated before
#: they are spliced into a path so a model-supplied ``../`` never reaches one.
_ID_RE = {
    "app": re.compile(r"^iap_[0-9a-f]{32}$"),
    "token": re.compile(r"^itk_[0-9a-f]{32}$"),
    "hook": re.compile(r"^ihk_[0-9a-f]{32}$"),
    "conversation": re.compile(r"^bcv_[0-9a-f]{32}$"),
    "tenant": re.compile(r"^[A-Za-z0-9_\-]{1,64}$"),
}


class IntegrationRefusal(Exception):
    """A typed, model-facing refusal. ``state`` is one of :data:`STATES`."""

    def __init__(
        self, state: str, message: str, *, code: str = "", http_status: int | None = 0
    ) -> None:
        super().__init__(message)
        self.state = state
        self.code = code
        self.http_status = http_status


#: The RFC §8.2 U5 user states plus the transport states around them. Each
#: maps to a sentence telling the MODEL what to do next (tool results are
#: model-facing and stay English by design; docs/I18N.md "Add a string" 1).
STATES = (
    "team_plan_required",  # U5 (1)
    "insufficient_role",  # U5 (2)
    "queued",  # U5 (3)
    "declined",  # U5 (4)
    "timed_out",  # U5 (5)
    "commitment_already_used",
    "unavailable",  # flag off: the routes are unregistered (404)
    "sign_in_required",
    "invalid",
    "secret_exists",
    "refused",
)


def check_id(kind: str, value: Any) -> str:
    text = str(value or "").strip()
    if not _ID_RE[kind].match(text):
        raise IntegrationRefusal("invalid", f"{kind}_id {text!r} is not a valid {kind} id.")
    return text


def check_scopes(scopes: Sequence[str] | None) -> list[str]:
    out = [str(s).strip() for s in (scopes or ()) if str(s).strip()]
    unknown = [s for s in out if s not in SCOPES]
    if unknown:
        raise IntegrationRefusal(
            "invalid",
            f"unknown scope(s) {unknown}; the closed v1 vocabulary is {list(SCOPES)}.",
        )
    return out


#: The coded 404s the integration service itself answers
#: (agent-server internal/errors/integration.go). Any OTHER 404 under these
#: routes is the router's: the feature is dark.
_RESOURCE_NOT_FOUND_CODES = frozenset(
    {"not_found", "conversation_not_found", "conversation_message_not_found"}
)


def map_refusal(refusal: BridgeRefusal, *, org_name: str = "") -> IntegrationRefusal:
    """One server refusal onto the five-state UX (RFC §8.2 U5, §9.4).

    404 with no integration code is the dark deployment: agent-server leaves the
    routes UNREGISTERED with ``TEAM_INTEGRATIONS_ENABLED`` off (S-70), so a probe
    reads exactly like an unknown path. A coded 404 (``not_found``) is a real
    missing app/token and stays a plain refusal.
    """

    code = refusal.code
    status = refusal.http_status
    org = f"“{org_name}”" if org_name else "this organization"
    if code == "team_plan_required":
        trial = bool((refusal.details or {}).get("trial_available"))
        start = "start the free trial or the Team plan" if trial else "start the Team plan"
        return IntegrationRefusal(
            "team_plan_required",
            f"Team Integrations needs an active Radient Team plan or free trial for {org}. "
            f"Tell the user to {start} at {BILLING_URL}, then try again. Do not retry before that.",
            code=code,
            http_status=status,
        )
    if code == "insufficient_role":
        return IntegrationRefusal(
            "insufficient_role",
            f"Only an owner or admin of {org} can manage integrations. Tell the user an owner "
            "or admin must run this; do not retry.",
            code=code,
            http_status=status,
        )
    if code == "commitment_already_used":
        return IntegrationRefusal(
            "commitment_already_used",
            "The confirmation expired or was already used (it lives 15 minutes and is single "
            "use). Nothing changed. Call the same op again with reask=true to file a fresh "
            "approval for the user.",
            code=code,
            http_status=status,
        )
    if status == 404 and code not in _RESOURCE_NOT_FOUND_CODES:
        return IntegrationRefusal(
            "unavailable",
            "Team Integrations is not available on this Radient deployment yet (the API is "
            "not enabled). Tell the user; nothing was changed.",
            code=code,
            http_status=status,
        )
    return IntegrationRefusal(
        "refused",
        f"Radient refused the request: {refusal.message} (code {code}, HTTP {status}).",
        code=code,
        http_status=status,
    )


class IntegrationsClient:
    """The admin API over the Team Bridge client's pinned Radient OAuth session.

    Riding :class:`TeamBridgeClient` is the point, not a convenience: it is the
    one transport that pins the exact OAuth row, re-checks the account across a
    refresh, refuses ``rad-…`` keys and non-canonical bases, and maps refusals
    from the flat error body without reading anything else from it (RFC §8.1
    "shared Radient trust layer").
    """

    def __init__(self, bridge: TeamBridgeClient) -> None:
        self._bridge = bridge

    async def _call(
        self, method: str, path: str, body: Mapping[str, Any] | None = None
    ) -> dict[str, Any]:
        try:
            response = await self._bridge._authorized(method, path, body=body, params=None)
        except BridgeSignInRequired as exc:
            raise IntegrationRefusal(
                "sign_in_required",
                f"{exc} Tell the user to sign in to Radient (`lop login radient`).",
            ) from None
        except BridgeRefusal as exc:
            raise map_refusal(exc) from None
        if response.status_code >= 400:
            raise map_refusal(_refusal(response)) from None
        if response.status_code == 204:
            return {}
        return _envelope_result(response)

    async def messaging_status(self) -> dict[str, Any]:
        return await self._call("GET", "/messaging/status")

    async def channels(self, tenant_id: str) -> dict[str, Any]:
        return await self._call("GET", f"/tenants/{tenant_id}/integrations/channels")

    async def list_apps(self, tenant_id: str) -> dict[str, Any]:
        return await self._call("GET", f"/tenants/{tenant_id}/integrations/apps")

    async def get_app(self, tenant_id: str, app_id: str) -> dict[str, Any]:
        return await self._call("GET", f"/tenants/{tenant_id}/integrations/apps/{app_id}")

    async def create_draft(self, tenant_id: str, body: Mapping[str, Any]) -> dict[str, Any]:
        return await self._call("POST", f"/tenants/{tenant_id}/integrations/apps", body)

    async def gated(
        self, tenant_id: str, op: str, target: Mapping[str, str], body: Mapping[str, Any]
    ) -> dict[str, Any]:
        """One gated mutation: ``prepare`` or the confirmed call (same path)."""
        return await self._call("POST", gated_path(tenant_id, op, target), body)

    async def revoke(self, tenant_id: str, app_id: str, kind: str, target_id: str) -> None:
        segment = "tokens" if kind == "token" else "hooks"
        await self._call(
            "DELETE", f"/tenants/{tenant_id}/integrations/apps/{app_id}/{segment}/{target_id}"
        )


def gated_path(tenant_id: str, op: str, target: Mapping[str, str]) -> str:
    base = f"/tenants/{tenant_id}/integrations/apps/{target['app_id']}"
    if op == "publish_app":
        return f"{base}/publish"
    if op == "mint_token":
        return f"{base}/tokens"
    if op == "rotate_token":
        return f"{base}/tokens/{target['token_id']}/rotate"
    if op == "create_hook":
        return f"{base}/hooks"
    if op == "add_to_channel":
        return f"{base}/installations"
    raise IntegrationRefusal("invalid", f"op {op!r} is not a gated op.")


def canonical(value: Any) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False)


def action_key(
    op: str, tenant_id: str, target: Mapping[str, str], payload: Mapping[str, Any], secret_name: str
) -> str:
    """The identity a consent grant binds to: what, where, with which payload.

    A yes to minting a ``chat:write`` token into ``A`` is not a yes to an
    ``alerts:ingest`` token into ``B``; every field the server binds the
    commitment to, plus the local secret name, is in the key.
    """

    material = canonical(
        {
            "op": op,
            "tenant": tenant_id,
            "target": dict(target),
            "payload": dict(payload),
            "secret": secret_name,
        }
    )
    return hashlib.sha256(material.encode("utf-8")).hexdigest()


def ask_ref_for(session_id: str, ask_id: str) -> str:
    """The audit reference the server stores with the consumed commitment.

    ``≤128`` bytes, no whitespace or control characters (``validAskRef``).
    """

    sid = re.sub(r"[^A-Za-z0-9_\-]", "", session_id)[:64]
    return f"lop:{sid}:{ask_id}"[:128]


# ---------------------------------------------------------------------------
# Sealing
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class SealedCredential:
    """What a sealed credential may say about itself: metadata, never the value."""

    secret_name: str
    kind: str  # "app_token" | "hook_secret"
    length: int
    fingerprint: str
    prefix: str
    rotated: bool = False

    def as_dict(self) -> dict[str, Any]:
        return {
            "secret_name": self.secret_name,
            "credential": self.kind,
            "prefix": self.prefix,
            "length": self.length,
            "value_fingerprint": self.fingerprint,
            "sealed": True,
            "written_with": "update" if self.rotated else "set",
        }


def preflight_secret_name(name: str, *, rotate: bool = False, base: Any = None) -> str:
    """Refuse a name that cannot receive the credential, BEFORE anything is minted.

    New credentials need an absent name (``set`` would raise ``SecretExists``
    after the show-once value was already minted); a rotation needs a present
    one (``update`` is the only overwrite, and only an explicit rotate uses it).
    """

    from local_operator.secrets.errors import SecretNotFound, SecretStoreError
    from local_operator.secrets.keys import store_path
    from local_operator.secrets.store import is_reserved_secret_name, validate_name

    try:
        canonical_name = validate_name(name)
    except SecretStoreError as exc:
        raise IntegrationRefusal("invalid", f"secret_name: {exc}") from None
    if is_reserved_secret_name(canonical_name):
        raise IntegrationRefusal("invalid", f"secret_name {canonical_name!r} is a reserved name.")
    if not store_path(base).exists():
        if rotate:
            raise IntegrationRefusal(
                "invalid", f"no secret named {canonical_name!r} exists to rotate into."
            )
        return canonical_name
    from local_operator.secrets.access import open_store

    store = open_store(base, create=False)
    try:
        store.describe(canonical_name)
        present = True
    except SecretNotFound:
        present = False
    if present and not rotate:
        raise IntegrationRefusal(
            "secret_exists",
            f"A secret named {canonical_name!r} already exists, so nothing was created. Pick a "
            "new secret_name (check `lop secret list`), or use rotate_token to replace it.",
        )
    if rotate and not present:
        raise IntegrationRefusal(
            "invalid", f"no secret named {canonical_name!r} exists to rotate into."
        )
    return canonical_name


def seal_from_response(
    response: dict[str, Any],
    *,
    secret_name: str,
    kind: str,
    description: str,
    rotate: bool = False,
    session_id: str | None = None,
    base: Any = None,
) -> SealedCredential | None:
    """Pop ``response["secret"]`` and seal it. The dict no longer holds it after.

    Registration with the runtime ledger happens FIRST, before the store write:
    should the write raise, every redaction sink already knows the value, so an
    exception message or a later output carrying it is still masked.
    """

    raw = response.pop("secret", None)
    if not raw:
        return None
    value = str(raw)
    del raw
    from local_operator.secrets import runtime

    runtime._LEDGER.register(value)
    if not value.startswith(SEALABLE_PREFIXES):
        raise IntegrationRefusal(
            "refused",
            "Radient returned a credential of an unexpected shape; it was not stored. "
            "Revoke it from the console and report this.",
        )
    from local_operator.secrets.access import open_store

    store = open_store(base, create=True)
    if rotate:
        store.update(
            secret_name, value.encode("utf-8"), description=description, session_id=session_id
        )
    else:
        store.set(
            secret_name, value.encode("utf-8"), description=description, session_id=session_id
        )
    sealed = SealedCredential(
        secret_name=secret_name,
        kind=kind,
        length=len(value),
        fingerprint=hashlib.sha256(value.encode("utf-8")).hexdigest()[:12],
        prefix=next(p for p in SEALABLE_PREFIXES if value.startswith(p)),
        rotated=rotate,
    )
    del value
    return sealed


# ---------------------------------------------------------------------------
# Consent ask copy (user-facing: keyed, see i18n catalogue wire.team_integrations)
# ---------------------------------------------------------------------------


@dataclass
class AskCopy:
    """Everything the human needs to judge the action, rendered by the host."""

    op: str
    app_name: str
    org_name: str
    channels: list[str] = field(default_factory=list)
    scopes: list[str] = field(default_factory=list)
    credential: str = ""  # "" | "app_token" | "hook_secret"
    secret_name: str = ""
    detail: str = ""  # op-specific subject: channel name, token id, source


def ask_question(copy: AskCopy) -> tuple[dict[str, Any], str, str]:
    """``(question, affirm_label, decline_label)`` for one consent ask."""

    from local_operator.i18n.keys import wire_team_integrations as k
    from local_operator.i18n.text import tr

    _t = tr("wire.team_integrations")
    app, org = copy.app_name, copy.org_name
    if copy.op == "publish_app":
        head = _t(k.ask_publish_question(app_name=app, org_name=org))
    elif copy.op == "mint_token":
        head = _t(k.ask_mint_token_question(app_name=app, org_name=org))
    elif copy.op == "rotate_token":
        head = _t(k.ask_rotate_token_question(app_name=app, org_name=org, token_id=copy.detail))
    elif copy.op == "create_hook":
        head = _t(k.ask_create_hook_question(app_name=app, org_name=org, source=copy.detail))
    else:
        head = _t(k.ask_add_to_channel_question(app_name=app, org_name=org, channel=copy.detail))
    scope_words = {
        "alerts:ingest": _t(k.scope_alerts_ingest()),
        "chat:write": _t(k.scope_chat_write()),
        "chat:write.thread": _t(k.scope_chat_write_thread()),
    }
    lines = [head]
    if copy.op in ("publish_app", "add_to_channel", "create_hook", "mint_token", "rotate_token"):
        channels = ", ".join(f"#{c}" for c in copy.channels) or _t(k.ask_details_none())
        lines.append(_t(k.ask_details_channels(channels=channels)))
    if copy.scopes:
        perms = "; ".join(f"{s} ({scope_words.get(s, s)})" for s in copy.scopes)
        lines.append(_t(k.ask_details_permissions(permissions=perms)))
    if copy.credential == "app_token":
        lines.append(_t(k.ask_details_credential_token(secret_name=copy.secret_name)))
    elif copy.credential == "hook_secret":
        lines.append(_t(k.ask_details_credential_hook(secret_name=copy.secret_name)))
    else:
        lines.append(_t(k.ask_details_credential_none()))
    if copy.op == "publish_app":
        affirm = _t(k.option_approve_publish())
        affirm_desc = (
            _t(k.option_approve_publish_description(secret_name=copy.secret_name))
            if copy.credential
            else _t(k.option_approve_publish_description_no_credential())
        )
        decline_desc = _t(k.option_cancel_draft_description())
    else:
        affirm = _t(k.option_approve())
        affirm_desc = _t(k.option_approve_description())
        decline_desc = _t(k.option_cancel_description())
    decline = _t(k.option_cancel())
    question = {
        "id": QUESTION_ID,
        "question": "\n".join(lines),
        # Opening a consent card is not consent. Keep Enter on the safe action
        # and do not recommend granting access; approval requires navigation.
        "options": [
            {"label": decline, "description": decline_desc},
            {"label": affirm, "description": affirm_desc},
        ],
        "multi": False,
        "secret": False,
    }
    return question, affirm, decline


__all__ = [
    "ASK_TIMEOUT_S",
    "AskCopy",
    "BILLING_URL",
    "CONSENT_OP",
    "HOOK_VENDORS",
    "IntegrationRefusal",
    "IntegrationsClient",
    "QUESTION_ID",
    "SCOPES",
    "SealedCredential",
    "action_key",
    "ask_question",
    "ask_ref_for",
    "check_id",
    "check_scopes",
    "map_refusal",
    "preflight_secret_name",
    "seal_from_response",
]
