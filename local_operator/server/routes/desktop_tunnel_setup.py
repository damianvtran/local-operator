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
    }
    blocker = blockers.get(step, "failed")
    kinds = {
        "login": "sign_in",
        "choose_account": "choose_account",
        "owner_missing": "restore_owner",
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
            "url": None,
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
    if result["verified"]:
        result["remedy"] = None
    return result


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
        result = _contract(await setup.observe_local(credential_id), local=True)
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
