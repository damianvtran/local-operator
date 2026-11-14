"""Turn on phone approvals from the desktop app — design v2 §6 (C3).

``GET /v1/desktop/operator/setup`` reports the state the Settings → Phone access
block draws (``off`` / ``finish_setup`` / ``on``, whether an OS prompt can appear
here, the level achieved and the last attempt's receipts). ``POST`` runs ONE
gesture-only attempt through :func:`operator.authority_setup.setup_authority`.

WHAT THE BEARER BUYS, AND WHAT IT DOES NOT. ``require_desktop`` is consent UX, not
a presence proof (design v2 fact 12): any same-user process that can read the claim
record can hold the bearer. So this route adds no authority of its own — every
guarantee lives in ``setup_authority`` and is enforced there for every caller:
the OS admin prompt is the only way the anchor lands, a privileged server process is
refused, and on Linux the headless ``sudo`` fallback is refused (``terminal_sudo``
is hard-wired ``False`` here; there is no TTY and passwordless sudo would pass with
no human). The request model forbids extra fields, so there is no way to hand this
route a stored admin password (``sudo_secret``) or anything else.
"""

from __future__ import annotations

import asyncio
from typing import Any

from fastapi import APIRouter, Depends, HTTPException, Request
from pydantic import BaseModel, ConfigDict

from local_operator.owned_tasks import join_owned
from local_operator.server.desktop import require_desktop
from local_operator.server.models.schemas import CRUDResponse
from local_operator.server.routes.desktop_sessions import reply

router = APIRouter(tags=["Desktop operator setup"], dependencies=[Depends(require_desktop)])


class OperatorSetupRequest(BaseModel):
    """An empty, closed body: the click is the whole request.

    ``extra="forbid"`` is the point. A client (or an agent holding the bearer) that
    sends ``sudo_secret``, a password, a route or a purpose gets a 422 — this path
    takes no input that could replace the person answering the OS prompt.
    """

    model_config = ConfigDict(extra="forbid", strict=True)


def _envelope(outcome: str) -> dict[str, Any]:
    from local_operator.operator.authority_setup import outcome_message

    return outcome_message(outcome).envelope(code=outcome)


def _decorate(result: dict[str, Any], outcome: str | None) -> dict[str, Any]:
    """Add the keyed sentences a client renders, beside the machine codes."""
    from local_operator.operator.authority_setup import file_only_message

    if outcome is not None:
        result["message"] = _envelope(outcome)
    # §6 item 5 / U18: a file-only host is told plainly, never left to infer a
    # presence gate it does not have.
    result["file_only_notice"] = (
        file_only_message().envelope(code="file_only") if result.get("file_only") else None
    )
    return result


@router.get("/v1/desktop/operator/setup", response_model=CRUDResponse[Any])
async def operator_setup_state(request: Request) -> CRUDResponse[Any]:
    from local_operator.operator.authority_setup import describe_state
    from local_operator.paths import config_dir

    state = await asyncio.to_thread(describe_state, config_dir(), terminal_sudo=False)
    active = getattr(request.app.state, "operator_setup_task", None)
    if active is not None and not active.done():
        state["in_progress"] = True
    return reply(_decorate(state, state["blocked"]))


@router.post("/v1/desktop/operator/setup", response_model=CRUDResponse[Any])
async def operator_setup_run(body: OperatorSetupRequest, request: Request) -> CRUDResponse[Any]:
    from local_operator.operator.authority_setup import (
        OUTCOME_IN_PROGRESS,
        REFUSALS,
        SetupInProgress,
        setup_authority,
    )
    from local_operator.paths import config_dir

    del body  # Validated (and closed) by the model; it carries nothing.
    # App-scoped admission, assigned without an await so it is atomic on the loop;
    # the module's own lock is the cross-process half (a TUI attempt).
    active = getattr(request.app.state, "operator_setup_task", None)
    if active is not None and not active.done():
        raise HTTPException(409, _envelope(OUTCOME_IN_PROGRESS))

    async def execute() -> dict[str, Any]:
        return await asyncio.to_thread(
            setup_authority, config_dir(), actor="desktop", terminal_sudo=False
        )

    operation = asyncio.create_task(execute(), name="desktop-operator-setup")
    request.app.state.operator_setup_task = operation
    try:
        # NO DEADLINE: the work is a person answering an OS prompt, and the
        # prompt process cannot be cancelled with its waiter. A dropped response
        # leaves the attempt running; `GET` reports `in_progress` meanwhile and
        # its receipts afterwards. join_owned keeps admission held until it ends.
        result = await join_owned(operation)
    except SetupInProgress:
        raise HTTPException(409, _envelope(OUTCOME_IN_PROGRESS)) from None
    finally:
        if operation.done() and request.app.state.operator_setup_task is operation:
            request.app.state.operator_setup_task = None
    outcome = str(result["outcome"])
    if outcome in REFUSALS:
        # Host refusals happen before any key, staging or prompt: a 409 whose
        # envelope code is the outcome, so a client branches on the code.
        raise HTTPException(409, _envelope(outcome))
    return reply(_decorate(result, outcome))
