"""The ``integrations`` tool: agent-driven Radient Team Integrations provisioning.

RFC "Radient Team Integrations" §8 (C3, approved 2026-11-17). An owner or admin
of a Radient organization can say "set up Alertmanager on #alerts-prod" and the
agent provisions it without the console: draft an integration app, get ONE
human approval, publish, and seal the show-once credential into the local
secret store. The wire and the sealing live in
:mod:`local_operator.team_bridge.integrations`; the consent ladder in
:mod:`local_operator.team_bridge.integration_consent`.

GATE (rung 3, createIf; RFC §8.1 R4/S-83). Built only where all three hold:

* ``context.ask_user`` is bound — a human can answer here. Subagents, exec,
  scheduler runs and headless responders have no hook, so prompt injection
  arriving through a channel post (TA3) can never reach an admin operation;
* ``context.integration_consent`` is bound — the session's seam that alone can
  say a PERSON answered (see that module for why a tool cannot);
* a Radient OAuth row exists — the admin API refuses ``rad-…`` keys, so without
  one every call could only say "sign in".

DEFERRED (``tools/deferral.py``): provisioning is rare, so the schema stays off
the wire until ``read tool://integrations`` or a first call.

THE TWO-STEP SHAPE every widening op follows (publish_app, mint_token,
rotate_token, create_hook, add_to_channel): the first call obtains a server
commitment (the draft's, or ``prepare: true``) and files ONE host-authored ask
that freezes it; the model ends the turn. When the person's answer arrives as
its own turn, the model repeats the call; the seam grants only on that
person's affirmative answer in this run, and the tool sends the FROZEN
commitment and payload — never ones the model re-supplies. ``revoke`` narrows
reach (the server ungates it) and runs at once behind the write-tier approval.

OUTPUT is metadata only. Tool results are persisted and replayed to the
provider; no path here puts a raw ``rdi_at_``/``rdi_hk_`` value in one.
"""

from __future__ import annotations

import json
import logging
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field, ValidationError

from local_operator.harness.types import AgentTool, ToolContext, ToolResult
from local_operator.tools.builtin import _error, _guard, _text, _validation_error

logger = logging.getLogger(__name__)

TOOL_NAME = "integrations"

READ_OPS = frozenset({"status", "list"})
GATED_OPS = frozenset(
    {"publish_app", "mint_token", "rotate_token", "create_hook", "add_to_channel"}
)

#: Server commitment ``action`` per op (agent-server integration_service.go
#: ``integrationAction*``). Checked when a commitment is received: a commitment
#: for a different action than the one the ask describes is never frozen.
SERVER_ACTION = {
    "publish_app": "publish",
    "mint_token": "mint_token",
    "rotate_token": "rotate_token",
    "create_hook": "create_hook",
    "add_to_channel": "install",
}

_DESCRIPTION = (
    "Provision Radient Team Integrations for the user's organization (owner/admin; Team plan or "
    "trial). Read guide://team-bridge-integrations first. Ops: status, list, draft_app, "
    "publish_app, add_to_channel, create_hook, mint_token, rotate_token, revoke. draft_app and "
    "each credential/channel op file ONE approval ask and return state=queued: end the turn; "
    "when the user's answer arrives, repeat the SAME call (or pass ask_id). Credentials are "
    "sealed into the local secret store as secret_name and never shown; results are metadata."
)


class IntegrationsParams(BaseModel):
    """Parameters for the ``integrations`` tool (see ``_DESCRIPTION``)."""

    model_config = ConfigDict(extra="forbid")

    op: Literal[
        "status",
        "list",
        "draft_app",
        "publish_app",
        "add_to_channel",
        "create_hook",
        "mint_token",
        "rotate_token",
        "revoke",
    ]
    tenant_id: str | None = Field(
        default=None, description="Organization id; optional with one org."
    )
    app_id: str | None = Field(default=None, description="iap_… app id.")
    name: str | None = Field(
        default=None, description="draft_app: app name; create_hook: automation name."
    )
    description: str | None = Field(default=None, description="draft_app: short description.")
    scopes: list[str] | None = Field(
        default=None, description="alerts:ingest | chat:write | chat:write.thread."
    )
    channel_ids: list[str] | None = Field(
        default=None,
        description="bcv_… channels: draft_app/create_hook destinations; add_to_channel takes one.",
    )
    credential: Literal["token", "hook", "none"] | None = Field(
        default=None, description="draft_app: initial credential bundled into publish."
    )
    vendor: Literal["datadog", "grafana", "alertmanager", "generic"] | None = Field(
        default=None, description="Automation source (wire field vendor)."
    )
    secret_name: str | None = Field(
        default=None, description="Local secret store name the credential is sealed into."
    )
    label: str | None = Field(default=None, description="Token label.")
    expires_in_days: int | None = Field(
        default=None, description="Token expiry (default 90, max 365)."
    )
    token_id: str | None = Field(default=None, description="itk_… (rotate_token, revoke).")
    hook_id: str | None = Field(default=None, description="ihk_… (revoke).")
    grace_seconds: int | None = Field(
        default=None, description="rotate_token overlap: 0, 3600 or 86400."
    )
    ask_id: str | None = Field(default=None, description="The approval ask to act on.")
    reask: bool = Field(default=False, description="File a fresh approval after a timeout/expiry.")


def _approval_tier(args: dict[str, Any]) -> Literal["read", "write", "exec"]:
    return "read" if args.get("op") in READ_OPS else "write"


def _describe_approval(args: dict[str, Any], cwd: str) -> str:
    del cwd  # the target is remote
    op = args.get("op")
    subject = args.get("app_id") or args.get("name") or ""
    return f"integrations {op} {subject}".strip()


def _result(tool_call_id: str, payload: dict[str, Any]) -> ToolResult:
    """Model-facing JSON (English by design: tool results are model-facing)."""

    return _text(
        tool_call_id, TOOL_NAME, json.dumps(payload, ensure_ascii=False, indent=1), details=payload
    )


def _refused(tool_call_id: str, state: str, message: str, **extra: Any) -> ToolResult:
    payload = {"ok": False, "state": state, "message": message, **extra}
    return _error(tool_call_id, TOOL_NAME, json.dumps(payload, ensure_ascii=False), details=payload)


# What the model is told for each non-granted consent state (RFC U5 3-5).
_CONSENT_SENTENCES = {
    "queued": (
        "An approval ask is queued for the user. A receipt is not consent: end the turn now and "
        "say what is waiting. When the answer arrives, repeat this exact call."
    ),
    "declined": (
        "The user cancelled (or declined to answer). Nothing was created; the draft, if any, stays "
        "inactive. Do not retry or re-ask in this conversation unless the user asks again."
    ),
    "timed_out": (
        "The approval lapsed before a person answered; nothing changed and any draft is kept. "
        "Re-ask (reask=true) only once the user is back and wants to continue."
    ),
    "not_human": (
        "That ask was not answered by a person on the card (withdrawn, answered in chat, "
        "or a typed "
        "reply), so it is not consent. Nothing changed. If the user still wants this, call again "
        "with reask=true and let them answer the card."
    ),
}


def _gate_bridge_login() -> bool:
    """Is there exactly one Radient OAuth row (the bridge client's pin rule)? Sync, no socket."""

    try:
        from local_operator.providers.auth_store import shared_auth_store

        rows = [
            r
            for r in shared_auth_store().list_credentials("radient")
            if r.credential_type == "oauth"
        ]
    except Exception:  # noqa: BLE001 - an unreadable store is "not signed in", fail closed
        return False
    return len(rows) == 1


def _make_client(context: ToolContext) -> Any:
    from local_operator.team_bridge.client import TeamBridgeClient
    from local_operator.team_bridge.integrations import IntegrationsClient

    factory = getattr(context, "integrations_client_factory", None)
    if callable(factory):  # tests inject a faked HTTP layer here
        return factory()
    return IntegrationsClient(TeamBridgeClient())


async def _resolve_tenant(client: Any, tenant_id: str | None) -> tuple[str, str]:
    """``(tenant_id, org_name)``. With no id, exactly one organization must exist."""

    from local_operator.team_bridge.integrations import IntegrationRefusal, check_id

    status = await client.messaging_status()
    tenants = [t for t in status.get("tenants") or [] if isinstance(t, dict)]
    if tenant_id:
        tid = check_id("tenant", tenant_id)
        for t in tenants:
            if t.get("tenant_id") == tid:
                return tid, str(t.get("tenant_name") or tid)
        return tid, tid
    if len(tenants) != 1:
        names = [{"tenant_id": t.get("tenant_id"), "name": t.get("tenant_name")} for t in tenants]
        raise IntegrationRefusal(
            "invalid",
            f"pass tenant_id: this account belongs to {len(tenants)} organizations: {names}",
        )
    t = tenants[0]
    return str(t.get("tenant_id")), str(t.get("tenant_name") or t.get("tenant_id"))


def _app_name(detail: dict[str, Any]) -> str:
    app = detail.get("app") or detail.get("summary") or {}
    return str(app.get("name") or "")


def _strip_secrets(value: Any) -> Any:
    """Belt and braces for read results: drop any ``secret`` key before the model sees it."""

    if isinstance(value, dict):
        return {k: _strip_secrets(v) for k, v in value.items() if k != "secret"}
    if isinstance(value, list):
        return [_strip_secrets(v) for v in value]
    return value


async def _op_status(client: Any, params: IntegrationsParams) -> dict[str, Any]:
    from local_operator.team_bridge.integrations import IntegrationRefusal

    status = await client.messaging_status()
    orgs = []
    for t in status.get("tenants") or []:
        if not isinstance(t, dict):
            continue
        tid = str(t.get("tenant_id") or "")
        if params.tenant_id and tid != params.tenant_id:
            continue
        row: dict[str, Any] = {
            "tenant_id": tid,
            "name": t.get("tenant_name"),
            "eligible": t.get("eligible"),
        }
        try:
            ch = await client.channels(tid)
            row["can_manage_integrations"] = bool(ch.get("can_manage_integrations"))
            row["channels"] = [
                {
                    "id": c.get("id"),
                    "name": c.get("name"),
                    "installed_app_ids": c.get("installed_app_ids"),
                }
                for c in ch.get("channels") or []
            ]
            row["state"] = "ready" if row["can_manage_integrations"] else "insufficient_role"
        except IntegrationRefusal as exc:
            row["state"] = exc.state
            row["message"] = str(exc)
        orgs.append(row)
    return {"ok": True, "op": "status", "organizations": orgs}


class _Run:
    """One call's state, so the gated path reads as the ladder it is."""

    def __init__(
        self, tool_call_id: str, params: IntegrationsParams, context: ToolContext, client: Any
    ) -> None:
        self.tool_call_id = tool_call_id
        self.p = params
        self.context = context
        self.client = client
        seam = context.integration_consent
        if seam is None:
            raise ValueError("integration consent must be bound before constructing a call")
        self.seam = seam
        self.tenant = ""
        self.org = ""

    # -- payload construction (what the server binds the commitment to) --

    def gated_shape(self) -> tuple[dict[str, str], dict[str, Any], str, str, bool]:
        """``(target, payload, secret_name, credential, rotate)`` for a post-publish op."""

        from local_operator.team_bridge.integrations import (
            IntegrationRefusal,
            check_id,
            check_scopes,
        )

        p = self.p
        app_id = check_id("app", p.app_id)
        target = {"app_id": app_id}
        if p.op == "publish_app":
            return target, {}, "", "", False
        if p.op == "mint_token":
            payload: dict[str, Any] = {"scopes": check_scopes(p.scopes or ["chat:write"])}
            if p.label:
                payload["label"] = p.label
            if p.expires_in_days:
                payload["expires_in_days"] = int(p.expires_in_days)
            return target, payload, self.need_secret_name(), "app_token", False
        if p.op == "rotate_token":
            target["token_id"] = check_id("token", p.token_id)
            payload = {}
            if p.grace_seconds is not None:
                payload["grace_seconds"] = int(p.grace_seconds)
            if p.expires_in_days:
                payload["expires_in_days"] = int(p.expires_in_days)
            return target, payload, self.need_secret_name(), "app_token", True
        if p.op == "create_hook":
            if not p.vendor or not (p.name or "").strip():
                raise IntegrationRefusal("invalid", "create_hook needs vendor and name.")
            ids = [check_id("conversation", c) for c in p.channel_ids or []]
            if not ids:
                raise IntegrationRefusal(
                    "invalid", "create_hook needs channel_ids (installed channels)."
                )
            payload = {
                "name": (p.name or "").strip(),
                "vendor": p.vendor,
                "destinations": [{"match_labels": {}, "target_conversation_id": i} for i in ids],
            }
            return target, payload, self.need_secret_name(), "hook_secret", False
        # add_to_channel
        ids = [check_id("conversation", c) for c in p.channel_ids or []]
        if len(ids) != 1:
            raise IntegrationRefusal("invalid", "add_to_channel takes exactly one channel id.")
        return target, {"conversation_id": ids[0]}, "", "", False

    def need_secret_name(self) -> str:
        from local_operator.team_bridge.integrations import IntegrationRefusal

        name = (self.p.secret_name or "").strip()
        if not name:
            raise IntegrationRefusal(
                "invalid", "secret_name is required: the credential is sealed there."
            )
        return name

    # -- the ops --

    async def draft_app(self) -> ToolResult:
        from local_operator.team_bridge.integrations import (
            AskCopy,
            IntegrationRefusal,
            action_key,
            check_id,
            check_scopes,
            preflight_secret_name,
        )

        p = self.p
        name = (p.name or "").strip()
        if not name:
            raise IntegrationRefusal("invalid", "draft_app needs name.")
        scopes = check_scopes(p.scopes or ["chat:write"])
        channels = [check_id("conversation", c) for c in p.channel_ids or []]
        credential = p.credential or "none"
        secret_name = ""
        body: dict[str, Any] = {"name": name, "scopes": scopes, "proposed_channel_ids": channels}
        if p.description:
            body["description"] = p.description
        if credential != "none":
            # Preflight BEFORE the draft (U2): a collision found after publish
            # would strand a show-once credential the user never received.
            secret_name = preflight_secret_name(self.need_secret_name())
            if credential == "token":
                spec: dict[str, Any] = {"kind": "token", "scopes": scopes}
                if p.label:
                    spec["label"] = p.label
                if p.expires_in_days:
                    spec["expires_in_days"] = int(p.expires_in_days)
            else:
                if not p.vendor:
                    raise IntegrationRefusal("invalid", "credential=hook needs vendor.")
                if not channels:
                    raise IntegrationRefusal(
                        "invalid", "credential=hook needs channel_ids to route to."
                    )
                spec = {
                    "kind": "hook",
                    "vendor": p.vendor,
                    "name": name,
                    "destinations": [
                        {"match_labels": {}, "target_conversation_id": c} for c in channels
                    ],
                }
            body["initial_credential"] = spec
        draft = await self.client.create_draft(self.tenant, body)
        app = draft.get("app") or {}
        app_id = check_id("app", app.get("id"))
        commitment = draft.get("commitment") or {}
        names = await self.channel_names(channels)
        copy = AskCopy(
            op="publish_app",
            app_name=name,
            org_name=self.org,
            channels=names,
            scopes=scopes,
            credential={"token": "app_token", "hook": "hook_secret"}.get(credential, ""),
            secret_name=secret_name,
        )
        key = action_key("publish_app", self.tenant, {"app_id": app_id}, {}, "")
        return self.file_ask(
            "publish_app",
            key,
            commitment,
            copy,
            target={"app_id": app_id},
            payload={},
            secret_name=secret_name,
            credential=copy.credential,
            rotate=False,
            snapshot={"revision": app.get("revision"), "channel_ids": channels, "scopes": scopes},
            extra={"app_id": app_id, "draft": True},
        )

    async def channel_names(self, ids: list[str]) -> list[str]:
        if not ids:
            return []
        try:
            ch = await self.client.channels(self.tenant)
        except Exception:  # noqa: BLE001 - names are a reading aid; ids still identify
            return ids
        by_id = {c.get("id"): c.get("name") for c in ch.get("channels") or []}
        return [str(by_id.get(i) or i) for i in ids]

    def file_ask(
        self,
        op: str,
        key: str,
        commitment: dict[str, Any],
        copy: Any,
        *,
        target: dict[str, str],
        payload: dict[str, Any],
        secret_name: str,
        credential: str,
        rotate: bool,
        snapshot: dict[str, Any],
        extra: dict[str, Any] | None = None,
    ) -> ToolResult:
        from local_operator.team_bridge.integrations import (
            IntegrationRefusal,
            ask_question,
        )

        digest = str(commitment.get("confirmation_commitment") or "")
        if not digest or commitment.get("action") != SERVER_ACTION[op]:
            # The ask must describe exactly what the server will bind; a
            # missing or other-action commitment is never frozen into one.
            raise IntegrationRefusal(
                "refused", "Radient did not return a matching confirmation commitment."
            )
        revision = snapshot.get("revision")
        prepared_revision = commitment.get("revision")
        if (
            type(revision) is not int
            or revision < 1
            or type(prepared_revision) is not int
            or prepared_revision != revision
        ):
            # GET and prepare are separate requests. Another administrator may
            # edit between them; that administrator's authority is not consent
            # from this person to the new grant. Never file an old card against
            # a newer commitment (including when revision metadata is missing).
            raise IntegrationRefusal(
                "refused",
                "Radient's prepared revision does not match the reviewed app snapshot; retry.",
            )
        question, affirm, decline = ask_question(copy)
        consent = {
            "action_key": key,
            "integration_op": op,
            "tenant_id": self.tenant,
            "target": target,
            "payload": payload,
            "commitment": digest,
            "commitment_expires_at": str(commitment.get("expires_at") or ""),
            "secret_name": secret_name,
            "credential": credential,
            "rotate": rotate,
            "app_name": copy.app_name,
            "review_version": 1,
            "review_snapshot": snapshot,
            "commitment_revision": prepared_revision,
            "affirm": affirm,
            "decline": decline,
        }
        outcome = self.seam.file(question, consent, tool_call_id=self.tool_call_id)
        if outcome.state != "queued":
            raise IntegrationRefusal(
                "refused", f"the approval ask could not be filed: {outcome.detail}"
            )
        return _result(
            self.tool_call_id,
            {
                "ok": True,
                "op": op,
                "state": "queued",
                "ask_id": outcome.ask_id,
                "expires_at": consent["commitment_expires_at"],
                **(extra or {}),
                "message": _CONSENT_SENTENCES["queued"],
            },
        )

    async def gated(self) -> ToolResult:
        from local_operator.team_bridge.integrations import (
            AskCopy,
            IntegrationRefusal,
            action_key,
            ask_ref_for,
            preflight_secret_name,
            seal_from_response,
        )

        p = self.p
        target, payload, secret_name, credential, rotate = self.gated_shape()
        key = action_key(p.op, self.tenant, target, payload, secret_name)
        outcome = self.seam.lookup(key, ask_id=(p.ask_id or "").strip())
        if outcome.state == "unavailable":
            raise IntegrationRefusal("refused", f"consent is unavailable: {outcome.detail}")
        if outcome.state == "granted":
            frozen = outcome.frozen
            review = frozen.get("review_snapshot") or {}
            revision = review.get("revision") if isinstance(review, dict) else None
            if (
                frozen.get("review_version") != 1
                or type(revision) is not int
                or revision < 1
                or revision != frozen.get("commitment_revision")
            ):
                # An approval queued by an older host may omit destinations,
                # inherited scopes, or bind a different displayed revision.
                # Never grandfather that card into a credential mint. Lookup
                # has consumed its answer; repeating the op obtains a fresh ask.
                raise IntegrationRefusal(
                    "refused",
                    "This approval lacks a complete frozen grant review; "
                    "repeat the operation with reask=true for a fresh approval.",
                )
            # The frozen record IS the authority: its tenant/target/payload and
            # commitment are what the person saw; the key equality above is
            # what proves this call asks for that same action.
            secret_name = str(frozen.get("secret_name") or "")
            credential = str(frozen.get("credential") or "")
            rotate = bool(frozen.get("rotate"))
            if credential:
                preflight_secret_name(secret_name, rotate=rotate)
            body = {
                **dict(frozen.get("payload") or {}),
                "confirmation_commitment": str(frozen["commitment"]),
                "ask_ref": ask_ref_for(self.seam.session_id, outcome.ask_id),
            }
            response = await self.client.gated(
                self.tenant, p.op, dict(frozen.get("target") or {}), body
            )
            sealed = None
            if credential:
                from local_operator.secrets.access import session_id
                from local_operator.secrets.errors import SecretStoreError

                try:
                    sealed = seal_from_response(
                        response,
                        secret_name=secret_name,
                        kind=credential,
                        description=(
                            f"Radient integration {credential} for app "
                            f"{frozen.get('app_name') or target['app_id']}"
                        ),
                        rotate=rotate,
                        session_id=session_id(),
                    )
                except SecretStoreError as exc:
                    # The server minted, the store refused: the show-once value
                    # is gone (and was never shown). Name the ids so the live
                    # credential can be revoked rather than left orphaned.
                    response.pop("secret", None)
                    ids = {
                        "token_id": (response.get("token") or {}).get("id"),
                        "hook_id": (response.get("hook") or {}).get("id"),
                    }
                    raise IntegrationRefusal(
                        "refused",
                        f"Radient created the credential but the local secret store refused it "
                        f"({type(exc).__name__}); it was not shown or kept. "
                        "Revoke it with op=revoke "
                        f"and {ids}, then retry with a usable secret_name.",
                    ) from None
            else:
                response.pop("secret", None)
            return _result(
                self.tool_call_id, self.granted_summary(p.op, response, sealed, outcome.ask_id)
            )
        # ``reask`` reopens a lapsed, non-human or cancelled ask (the user came
        # back and asked again); it never bypasses an OPEN one, which would put
        # two cards for one action in front of them.
        if outcome.state == "queued" or (
            outcome.state in ("timed_out", "not_human", "declined") and not p.reask
        ):
            return _refused(
                self.tool_call_id,
                outcome.state,
                _CONSENT_SENTENCES[outcome.state],
                ask_id=outcome.ask_id,
                op=p.op,
            )
        # NONE, or an explicit re-ask after a lapse: obtain a fresh commitment.
        if credential:
            preflight_secret_name(secret_name, rotate=rotate)
        detail = await self.client.get_app(self.tenant, target["app_id"])
        app_name = _app_name(detail) or target["app_id"]
        prepared = await self.client.gated(self.tenant, p.op, target, {**payload, "prepare": True})
        commitment = prepared.get("commitment") or {}
        subject = {
            "rotate_token": target.get("token_id", ""),
            "create_hook": payload.get("vendor", ""),
        }.get(p.op, "")
        app = detail.get("app") or {}
        channel_ids: list[str] = []
        if p.op in ("mint_token", "rotate_token"):
            # A token can address every installed channel, not a model's
            # proposed subset. Installation changes bump the app revision, so
            # the snapshot/prepare equality below fences concurrent changes.
            channel_ids = [str(row["conversation_id"]) for row in detail.get("installations") or []]
        elif p.op == "create_hook":
            channel_ids = [str(row["target_conversation_id"]) for row in payload["destinations"]]
        elif p.op == "add_to_channel":
            channel_ids = [payload["conversation_id"]]
        channels = await self.channel_names(channel_ids)
        if p.op == "add_to_channel":
            subject = channels[0]
        if p.op == "publish_app":
            app = detail.get("app") or {}
            channel_ids = list(app.get("proposed_channel_ids") or [])
            channels = await self.channel_names(channel_ids)
            spec = app.get("initial_credential") or {}
            credential = {"token": "app_token", "hook": "hook_secret"}.get(
                str(spec.get("kind") or ""), ""
            )
            if credential:
                # A re-asked publish keeps the secret name the first ask froze.
                prior = self.prior_secret_name(key)
                if not prior:
                    raise IntegrationRefusal(
                        "invalid", "the original draft's secret_name is unknown here; draft again."
                    )
                secret_name = preflight_secret_name(prior)
        scopes = list(payload.get("scopes") or [])
        if p.op == "publish_app":
            scopes = list(app.get("scopes") or [])
        elif p.op == "rotate_token":
            # Rotation inherits the existing token's immutable grant; request
            # arguments contain neither that grant nor a trustworthy summary.
            token = next(
                (row for row in detail.get("tokens") or [] if row.get("id") == target["token_id"]),
                None,
            )
            if token is None or not token.get("scopes"):
                raise IntegrationRefusal(
                    "refused", "Radient did not return the token's effective scopes."
                )
            scopes = list(token["scopes"])
        copy = AskCopy(
            op=p.op,
            app_name=app_name,
            org_name=self.org,
            channels=channels,
            scopes=scopes,
            credential=credential,
            secret_name=secret_name,
            detail=subject,
        )
        return self.file_ask(
            p.op,
            key,
            commitment,
            copy,
            target=target,
            payload=payload,
            secret_name=secret_name,
            credential=credential,
            rotate=rotate,
            snapshot={
                "revision": app.get("revision"),
                "channel_ids": channel_ids,
                "scopes": scopes,
            },
            extra={"app_id": target["app_id"]},
        )

    def prior_secret_name(self, key: str) -> str:
        queue = self.seam._queue_getter()
        if queue is None:
            return ""
        for record in queue.records():
            consent = record.get("consent") or {}
            if consent.get("action_key") == key and consent.get("secret_name"):
                return str(consent["secret_name"])
        return ""

    def granted_summary(
        self, op: str, response: dict[str, Any], sealed: Any, ask_id: str
    ) -> dict[str, Any]:
        out: dict[str, Any] = {"ok": True, "op": op, "state": "done", "ask_id": ask_id}
        app = response.get("app") or {}
        if app:
            out["app"] = {"id": app.get("id"), "name": app.get("name"), "status": app.get("status")}
        if response.get("installations") is not None:
            out["installations"] = [
                i.get("conversation_id") for i in response.get("installations") or []
            ]
        if response.get("installation"):
            out["installation"] = response["installation"].get("conversation_id")
        token = response.get("token")
        if token:
            out["token"] = {
                "id": token.get("id"),
                "prefix_hint": token.get("prefix_hint"),
                "scopes": token.get("scopes"),
                "expires_at": token.get("expires_at"),
            }
        if response.get("rotated"):
            r = response["rotated"]
            out["rotated_out"] = {
                "id": r.get("id"),
                "state": r.get("state"),
                "grace_expires_at": r.get("grace_expires_at"),
            }
        hook = response.get("hook")
        if hook:
            out["hook"] = {
                "id": hook.get("id"),
                "vendor": hook.get("vendor"),
                "status": hook.get("status"),
            }
        if response.get("ingest_path"):
            out["ingest_path"] = response["ingest_path"]
        if sealed is not None:
            out["credential"] = sealed.as_dict()
            out["message"] = (
                "The credential is sealed in the local secret store as "
                f"{sealed.secret_name}; it was "
                "never shown here. Verify with `lop secret describe "
                f"{sealed.secret_name}`. Deploy it per guide://team-bridge-integrations."
            )
        return out

    async def revoke(self) -> ToolResult:
        from local_operator.team_bridge.integrations import IntegrationRefusal, check_id

        app_id = check_id("app", self.p.app_id)
        if self.p.token_id:
            kind, target = "token", check_id("token", self.p.token_id)
        elif self.p.hook_id:
            kind, target = "hook", check_id("hook", self.p.hook_id)
        else:
            raise IntegrationRefusal("invalid", "revoke needs token_id or hook_id.")
        await self.client.revoke(self.tenant, app_id, kind, target)
        return _result(
            self.tool_call_id,
            {
                "ok": True,
                "op": "revoke",
                "state": "done",
                "revoked": {"kind": kind, "id": target},
                "message": "Revoked on Radient; the local secret, if any, was left in place.",
            },
        )


@_guard(TOOL_NAME)
async def execute_integrations(
    tool_call_id: str,
    args: dict[str, Any],
    signal: Any = None,
    on_update: Any = None,
    context: ToolContext | None = None,
) -> ToolResult:
    from local_operator.team_bridge.integrations import IntegrationRefusal

    try:
        params = IntegrationsParams(**args)
    except ValidationError as exc:
        return _validation_error(tool_call_id, TOOL_NAME, exc)
    if context is None or context.integration_consent is None or context.ask_user is None:
        return _refused(tool_call_id, "refused", "integrations needs an interactive session.")
    client = _make_client(context)
    run: _Run | None = None
    try:
        if params.op == "status":
            return _result(tool_call_id, await _op_status(client, params))
        run = _Run(tool_call_id, params, context, client)
        run.tenant, run.org = await _resolve_tenant(client, params.tenant_id)
        if params.op == "list":
            apps = await client.list_apps(run.tenant)
            return _result(
                tool_call_id,
                {"ok": True, "op": "list", "tenant_id": run.tenant, **_strip_secrets(apps)},
            )
        if params.op == "draft_app":
            return await run.draft_app()
        if params.op == "revoke":
            return await run.revoke()
        return await run.gated()
    except IntegrationRefusal as exc:
        message = str(exc)
        org = run.org if run is not None else ""
        if exc.state in ("team_plan_required", "insufficient_role") and org:
            # U5 (2): name the organization the refusal is about.
            message = message.replace("this organization", f"“{org}”")
        return _refused(tool_call_id, exc.state, message, op=params.op, code=exc.code or None)


def build_integrations_tool(context: ToolContext) -> AgentTool | None:
    """createIf: a human can answer, the session's consent seam exists, Radient is signed in."""

    if context.ask_user is None or getattr(context, "integration_consent", None) is None:
        return None
    if getattr(context, "integrations_client_factory", None) is None and not _gate_bridge_login():
        return None
    return AgentTool(
        name=TOOL_NAME,
        label="Integrations",
        description=_DESCRIPTION,
        parameters=IntegrationsParams.model_json_schema(),
        approval_tier="write",
        call_approval_tier=_approval_tier,
        # Exclusive: two concurrent gated calls could both read "no ask" and
        # file two cards for one action.
        concurrency="exclusive",
        interruptible=True,
        describe_approval=_describe_approval,
        execute=execute_integrations,
    )


__all__ = ["IntegrationsParams", "build_integrations_tool", "execute_integrations"]
