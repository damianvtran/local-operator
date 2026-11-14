"""Explicit, owner-pinned desktop setup; the existing health GET stays read-only.

The desktop bearer gains only the same bounded authority as tunnel setup apply:
no terms/purchase/package build/grant/bind/repoint operation. Polling is strictly
local metadata; a fresh quote requires an explicit plan POST, and run requires
an explicit owner. Dropped responses must not abandon a blocking installation.
"""

from __future__ import annotations

import asyncio
from decimal import Decimal, InvalidOperation
from typing import Any, Literal

from fastapi import APIRouter, Depends, HTTPException, Query, Request
from pydantic import (
    BaseModel,
    ConfigDict,
    Field,
    StrictInt,
    StrictStr,
    field_validator,
    model_validator,
)

from local_operator.i18n.keys import wire_tunnel_setup as _wd
from local_operator.owned_tasks import join_owned
from local_operator.server.desktop import require_desktop
from local_operator.server.models.schemas import CRUDResponse
from local_operator.server.routes.desktop_sessions import reply

router = APIRouter(tags=["Desktop tunnel setup"], dependencies=[Depends(require_desktop)])
# Client budgets include existing bounded upstream/probe/bundle/supervisor work,
# not just measured fast fake-upstream latency. Expiry cancels only the waiter;
# safe drain can exceed this budget, rather than orphaning a to_thread install.
PLAN_DEADLINE_S = 120
RUN_DEADLINE_S = 900
# Mirrors `tunnels.gateway.CONSOLE_URL`; a literal so this route's import stays
# off the tunnel gateway (the boot-time import graph is pinned in
# `tests/unit/test_import_graph.py`). `test_desktop_tunnel_setup` holds them equal.
_CONSOLE_URL = "https://console.radienthq.com/dashboard/tunnels"


class SetupRequest(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True)
    action: Literal["plan", "run"]
    credential_id: StrictInt | None = Field(default=None, ge=1)
    accept_monthly_price: StrictStr | None = Field(default=None, max_length=64)

    @field_validator("accept_monthly_price")
    @classmethod
    def price_is_money(cls, value: str | None) -> str | None:
        if value is not None:
            try:
                amount = Decimal(value)
            except InvalidOperation as exc:
                raise ValueError("invalid_monthly_price") from exc
            if not amount.is_finite() or amount < 0:
                raise ValueError("invalid_monthly_price")
        return value

    @model_validator(mode="after")
    def explicit_run_owner(self) -> SetupRequest:
        if self.action == "run" and self.credential_id is None:
            raise ValueError("credential_id_required_for_run")
        if self.action == "plan" and self.accept_monthly_price is not None:
            raise ValueError("price_acceptance_is_run_only")
        return self


def _refusal(status: int, code: str) -> HTTPException:
    messages = {
        "setup_in_progress": _wd.in_progress,
        "setup_deadline": _wd.deadline,
        "setup_unavailable": _wd.unavailable,
        "setup_intent_corrupt": _wd.corrupt_intent,
    }
    return HTTPException(status, messages[code]().envelope(code=code))


def _contract(plan: dict[str, Any], *, local: bool) -> dict[str, Any]:
    # Whitelist the transport. Neither connector park detail nor cached/cloud
    # records, token-bearing provider responses or exception text can leak here.
    step = str(plan["step"])
    blockers = {
        "complete": "none",
        "apply": "none",
        "owner_choice": "choose_account",
        "owner_missing": "owner_missing",
        "repoint": "repoint",
        "login": "login",
        "quote": "quote",
        "quote_changed": "quote",
        "quote_unread": "quote_unread",
        "account": "account",
        "cloudflared": "cloudflared",
        "mobile_bundle": "mobile_bundle",
        "supervisor": "supervisor",
        "balance": "balance",
        "stopped": "stopped",
        "verification": "verification",
        "failed": "failed",
        # A suspended tunnel with billing in order: the same primary run as
        # `apply` (an older renderer keeps showing its button), with its own
        # step and copy so the desktop can say "Reconnect".
        "reconnect": "none",
        # Lapsed billing reuses `balance` so a renderer that predates the split
        # still shows its add-credit callout; `step` carries the distinction.
        "billing_lapsed": "balance",
        "disabled": "disabled",
        "network": "network",
        "radient_unavailable": "radient_unavailable",
        "reconnecting": "reconnecting",
    }
    blocker = blockers.get(step, "failed")
    kinds = {
        "login": "sign_in",
        "choose_account": "choose_account",
        "owner_missing": "restore_owner",
        # `run` is the ONE kind that makes the desktop render its primary action
        # button (posting the plan's own credential_id), which is what turns the
        # re-point from a sentence into something the operator can do.
        "repoint": "run",
        "quote": "accept_quote",
        "quote_unread": "check_quote",
        "account": "check_account",
        "cloudflared": "install_cloudflared",
        "mobile_bundle": "prepare_mobile",
        "supervisor": "enable_supervisor",
        "balance": "add_credit",
        "verification": "check_status",
        "failed": "check_status",
        "stopped": "resume_explicitly",
        "none": "run",
        # The kinds `features.tunnel_remedy_kinds` advertises, shared with
        # `gateway.REMEDY_KIND` (the phone's 503 and `GET /v1/desktop/tunnel`):
        # a console toggle is a link, and the two waits render no button because
        # they recover by themselves.
        "disabled": "open_console",
        "network": "wait_network",
        "radient_unavailable": "wait_radient",
        # The post-sign-in no-spend run holds `setup.lock` (U10): no button,
        # the desktop polls until it settles.
        "reconnecting": "wait_reconnect",
    }
    quote = None
    if not local and plan.get("account_valid"):
        quote = {
            "monthly_price_usd": plan.get("monthly_price_usd"),
            "positive_balance": plan.get("positive_balance") is True,
            "eligible": plan.get("activation_required") is False,
            "account_valid": True,
        }
    from local_operator.tunnels import setup

    choice_ids = {c if isinstance(c, int) else c["id"] for c in plan.get("credential_choices", [])}
    choices = [c for c in setup.local_credential_choices() if c["id"] in choice_ids]
    receipt_steps = plan.get("receipt_steps", [])
    receipts = [
        {
            "step": receipt_steps[i] if i < len(receipt_steps) else "completed",
            "ok": True,
            "text": text,
        }
        for i, text in enumerate(plan.get("receipts", []))
    ]
    result = {
        "step": step,
        "blocked_on": blocker,
        "configured": bool(plan.get("configured")),
        "ready": bool(plan.get("ready")),
        "verified": bool(plan.get("verified")) and not local,
        "url": plan.get("url") if plan.get("verified") and not local else None,
        "credential_id": plan.get("credential_id"),
        "identity_label": plan.get("identity_label"),
        "credential_choices": choices,
        "quote": quote,
        "receipts": receipts,
        "done": [r["step"] for r in receipts],
        "remedy": {
            "kind": kinds.get(blocker, "check_status"),
            "text": plan["remedy"],
            "command": plan.get("apply_command"),
            # The console is the whole remedy for these two, so the link
            # travels as data rather than only inside the sentence.
            "url": _CONSOLE_URL if blocker in ("disabled", "balance") else None,
        },
        "source": "local" if local else "fresh",
        "in_progress": False,
        "progress": [],
        "active_credential_id": None,
        "request_timeout_ms": (RUN_DEADLINE_S if not local else PLAN_DEADLINE_S) * 1000,
    }
    for field in ("billing_outcome", "reservation_outcome", "failure_stage", "accepted_price_usd"):
        if field in plan:
            result[field] = plan[field]
    connector = plan.get("connector")
    if isinstance(connector, dict):
        result["connector"] = {"state": connector.get("state")}
    # Which connector carries this device's tunnel, for the desktop's setup
    # copy (local-operator-ui C2). The live gateway answer wins; without one,
    # the marker the service leaves after a /connect that offered the relay.
    from local_operator.tunnels import config as tunnel_config

    live = connector.get("transport") if isinstance(connector, dict) else None
    if live in (tunnel_config.TRANSPORT_RELAY, tunnel_config.TRANSPORT_CLOUDFLARE):
        result["transport"] = live
    else:
        result["transport"] = (
            tunnel_config.TRANSPORT_RELAY
            if tunnel_config.relay_offered()
            else tunnel_config.TRANSPORT_CLOUDFLARE
        )
    if result["verified"]:
        result["remedy"] = None
    return result


def _identity_fields(plan: dict[str, Any]) -> dict[str, Any]:
    """The identity facts a reconnecting plan keeps from the local plan it replaces."""
    return {
        key: plan[key]
        for key in ("credential_id", "identity_label", "configured", "credential_choices")
        if key in plan
    }


async def _reconnecting_reply(credential_id: int | None) -> CRUDResponse[Any]:
    """The POST's answer while the background reconnect holds `setup.lock`.

    The identity comes from this machine's own local plan, never from the
    request body (#3945 round 1, Q4): echoing ``credential_id: 999`` stated a
    client value as fact. An id this device does not hold gets the answer the
    route gives it without the lock — `observe`'s `owner_choice` — with nothing
    echoed back.
    """
    from local_operator.tunnels import setup

    local = await setup.observe_local(credential_id)
    known = {c["id"] for c in local.get("credential_choices", [])}
    if credential_id is not None and credential_id not in known:
        refusal = setup.unknown_credential_plan(local.get("credential_choices", []))
        return reply(_contract(refusal, local=True))
    return reply(_contract(setup.reconnecting_plan(**_identity_fields(local)), local=True))


async def drain(app: Any) -> None:
    """Join setup before shutdown closes its auth/worker prerequisites."""
    operation = getattr(app.state, "tunnel_setup_task", None)
    if operation is not None:
        try:
            await join_owned(operation)
        finally:
            if operation.done() and app.state.tunnel_setup_task is operation:
                app.state.tunnel_setup_task = None


@router.get("/v1/desktop/tunnel/setup", response_model=CRUDResponse[Any])
async def local_plan(
    request: Request, credential_id: int | None = Query(default=None, ge=1)
) -> CRUDResponse[Any]:
    from local_operator.tunnels import setup

    try:
        plan = await setup.observe_local(credential_id)
        if await asyncio.to_thread(setup.auto_reconnecting):
            # The background heal is visible as its own state (U10) rather than
            # as the stale "not active" plan it is busy clearing.
            plan = setup.reconnecting_plan(**_identity_fields(plan))
        result = _contract(plan, local=True)
        active = getattr(request.app.state, "tunnel_setup_task", None)
        result["in_progress"] = active is not None and not active.done()
        result["progress"] = (
            list(getattr(request.app.state, "tunnel_setup_progress", []))
            if result["in_progress"]
            else []
        )
        result["active_credential_id"] = (
            getattr(request.app.state, "tunnel_setup_credential", None)
            if result["in_progress"]
            else None
        )
        if result["in_progress"]:
            result.update(
                blocked_on="in_progress",
                ready=False,
                remedy={
                    "kind": "check_status",
                    "text": _wd.in_progress().envelope()["text"],
                    "command": None,
                    "url": None,
                },
            )
        return reply(result)
    except setup.CorruptSetupIntent:
        raise _refusal(503, "setup_intent_corrupt") from None
    except (ValueError, OSError):
        raise _refusal(503, "setup_unavailable") from None


@router.post("/v1/desktop/tunnel/setup", response_model=CRUDResponse[Any])
async def setup_action(body: SetupRequest, request: Request) -> CRUDResponse[Any]:
    from local_operator.tunnels import setup

    # App-scoped, not per-account: one record/service belongs to this backend.
    # Assignment happens without an await, so admission is atomic on its loop.
    active = getattr(request.app.state, "tunnel_setup_task", None)
    if active is not None and not active.done():
        raise _refusal(409, "setup_in_progress")
    if await asyncio.to_thread(setup.auto_reconnecting):
        # The unattended post-sign-in run owns `setup.lock`. Answer its state,
        # not a 409: the person pressed a button to get phone access back, and
        # that is exactly what is already happening (U10). Nothing is started,
        # so the lock's single-owner guarantee is untouched.
        return await _reconnecting_reply(body.credential_id)

    deadline = PLAN_DEADLINE_S if body.action == "plan" else RUN_DEADLINE_S
    # Ephemeral, fixed-copy progress only—not another persisted setup journal.
    # Source callbacks contain no provider response/error/credential payload.
    request.app.state.tunnel_setup_progress = [_wd.checking().envelope()["text"]]
    request.app.state.tunnel_setup_credential = body.credential_id

    def progress(text: str) -> None:
        messages = request.app.state.tunnel_setup_progress
        if len(messages) < 20:
            messages.append(text)

    async def execute() -> dict[str, Any]:
        if body.action == "plan":
            plan = await setup.observe(body.credential_id)
        else:
            plan = await setup.apply(
                body.credential_id, body.accept_monthly_price, progress=progress
            )
        result = _contract(plan, local=False)
        result["request_timeout_ms"] = deadline * 1000
        return result

    operation = asyncio.create_task(execute(), name="desktop-tunnel-setup")
    request.app.state.tunnel_setup_task = operation
    try:
        # wait_for cancels the JOIN waiter on expiry; join_owned drains the
        # operation first. Never release admission while its worker can write.
        waiter = asyncio.create_task(asyncio.wait_for(join_owned(operation), timeout=deadline))
        # The request itself also owns/drains this waiter. A second cancellation
        # cannot interrupt wait_for's cancellation cleanup and leave it behind.
        return reply(await join_owned(waiter))
    except setup.SetupInProgress:
        if await asyncio.to_thread(setup.auto_reconnecting):
            # Lost the race to the background run between the check above and
            # the claim: same answer as if it had been seen first.
            return await _reconnecting_reply(body.credential_id)
        raise _refusal(409, "setup_in_progress") from None
    except setup.CorruptSetupIntent:
        raise _refusal(503, "setup_intent_corrupt") from None
    except asyncio.TimeoutError:
        raise _refusal(504, "setup_deadline") from None
    except (ValueError, OSError, KeyError, TypeError):
        raise _refusal(503, "setup_unavailable") from None
    finally:
        if operation.done() and request.app.state.tunnel_setup_task is operation:
            request.app.state.tunnel_setup_task = None
