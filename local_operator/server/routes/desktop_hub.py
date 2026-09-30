"""Desktop Agent Hub update routes (design B5.1).

The sidebar's "update available" indicator polls ``GET /v1/desktop/hub/updates``
constantly, so that route is a STORE READ: no network, O(items), and it works in
manual mode too (the runner keeps checking whichever way ``hub.auto_update.*``
is set). Every mutation runs through ``receipts(request).run(..., retry_safe=True)``
like ``profiles/sync``, so a lost response is replayed rather than re-run.

Check and apply run through ``app.state.hub_sync.run_exclusive`` (the runner)
when the daemon has one, so a button press queues behind a running tick on the
runner's ONE in-process lock; ``service.apply_items`` additionally takes the
cross-process lease (a CLI run or a second daemon), and a lease that stays held
past its wait answers 409. Without a runner (a bare router in a test) the work
runs on a one-shot service context built from the same pieces — still under the
lease. There is no third path: classification, merging and wording all live in
``local_operator.hub_sync.service``.

Imported lazily inside handlers: this module is mounted on every ``lop serve``
boot and the hub stack (registries, ``requests`` clients, the merge core) is dead
weight until someone asks (the server-shape guard in ``tests/unit/test_import_graph.py``).
"""

from __future__ import annotations

import asyncio
from typing import Any, Callable, Literal, TypeVar

from fastapi import APIRouter, Depends, HTTPException, Request
from pydantic import Field

from local_operator.config import ConfigManager
from local_operator.providers.auth_store import AuthStore
from local_operator.server.dependencies import (
    get_config_manager,
    get_provider_auth_store,
)
from local_operator.server.desktop import require_desktop
from local_operator.server.models.schemas import CRUDResponse
from local_operator.server.routes.desktop_sessions import (
    Input,
    RequestID,
    errors,
    receipts,
    reply,
)

router = APIRouter(tags=["Desktop hub"], dependencies=[Depends(require_desktop)])

Kind = Literal["agent", "team"]
T = TypeVar("T")


class HubCheck(Input):
    request_id: RequestID
    kind: Kind | None = None
    name: str | None = Field(default=None, min_length=1, max_length=128)


class HubApply(Input):
    request_id: RequestID
    kind: Kind
    name: str = Field(min_length=1, max_length=128)
    prefer: Literal["local", "remote"] | None = None
    acknowledge_unknown_baseline: bool = False
    #: ``replace`` discards one side wholesale and is the ONLY such operation; it
    #: needs ``confirm_replace`` so a stray field cannot trigger it.
    replace: Literal["remote", "local"] | None = None
    confirm_replace: bool = False
    dry_run: bool = False


class HubApplyAll(Input):
    request_id: RequestID
    kind: Kind | None = None


class HubRetry(Input):
    request_id: RequestID
    kind: Kind
    name: str = Field(min_length=1, max_length=128)


async def _run(
    request: Request,
    config_manager: ConfigManager,
    provider_auth_store: AuthStore,
    work: Callable[[Any], T],
) -> T:
    """Run ``work(ctx)`` (synchronous, blocking) under the runner's lock when there is one.

    ``HubBusy`` (another process holds the update lease) is the caller's 409: the
    request did nothing and is safe to repeat.
    """

    from local_operator.hub_sync import service as svc

    runner = getattr(request.app.state, "hub_sync", None)
    try:
        if runner is not None:
            return await runner.run_exclusive(work)
        for_tenant, credential = await svc.build_clients(config_manager, provider_auth_store)
        ctx = svc.HubSyncContext(
            config_dir=config_manager.config_dir,
            config_manager=config_manager,
            client_for_tenant=for_tenant,
            credential=credential,  # type: ignore[arg-type]
        )
        return await asyncio.to_thread(work, ctx)
    except svc.HubBusy as busy:
        raise _refusal(409, str(busy)) from None


def _refusal(status: int, message: str) -> HTTPException:
    return HTTPException(status, message)


def _raise_for(reports: Any) -> None:
    """Map a SINGLE-item outcome onto the status vocabulary of the team pull route."""

    if len(reports) != 1:
        return
    report = reports[0]
    if report.error_class == "no-credential":
        from local_operator.providers.radient_credentials import ORG_LOGIN_REMEDY

        raise _refusal(401, ORG_LOGIN_REMEDY)
    if report.error_class == "concurrent-edit":
        raise _refusal(409, report.message or "the item changed while the update was computed")
    if report.outcome == "refused":
        raise _refusal(422, report.message or "the merged result was refused")


@router.get("/v1/desktop/hub/updates", response_model=CRUDResponse)
async def updates(
    request: Request,
    config_manager: ConfigManager = Depends(get_config_manager),
):
    """The status store, as the sidebar polls it. No network."""

    from local_operator.hub_sync import service as svc

    def read() -> dict[str, Any]:
        ctx = svc.HubSyncContext(
            config_dir=config_manager.config_dir,
            config_manager=config_manager,
            client_for_tenant=lambda _tenant: None,
        )
        return svc.status_snapshot(ctx)

    async with errors(request):
        try:
            return reply(await asyncio.to_thread(read))
        except OSError as exc:
            raise _refusal(503, f"the hub status store is unreadable: {exc}") from None


@router.post("/v1/desktop/hub/updates/check", response_model=CRUDResponse)
async def check(
    body: HubCheck,
    request: Request,
    config_manager: ConfigManager = Depends(get_config_manager),
    provider_auth_store: AuthStore = Depends(get_provider_auth_store),
):
    """Check now. Applies nothing, even with auto on: the user asked to *check*."""

    from local_operator.hub_sync import service as svc

    def work(ctx: Any) -> dict[str, Any]:
        kinds = (body.kind,) if body.kind else ("agent", "team")
        names = [body.name] if body.name else None
        checks = svc.check_items(ctx, kinds=kinds, names=names)
        if body.name and not checks:
            raise _refusal(404, f"no linked {body.kind or 'item'} named {body.name!r}")
        return {
            "reports": [svc.check_report(c).to_json() for c in checks],
            "status": svc.status_snapshot(ctx),
        }

    async def mutate() -> dict[str, Any]:
        return await _run(request, config_manager, provider_auth_store, work)

    async with errors(request):
        return reply(
            await receipts(request).run(
                "hub-check:" + body.request_id, body.model_dump(), mutate, retry_safe=True
            )
        )


@router.post("/v1/desktop/hub/updates/apply", response_model=CRUDResponse)
async def apply(
    body: HubApply,
    request: Request,
    config_manager: ConfigManager = Depends(get_config_manager),
    provider_auth_store: AuthStore = Depends(get_provider_auth_store),
):
    """Apply ONE item (click-to-update). ``dry_run`` previews without writing."""

    from local_operator.hub_sync import service as svc

    if body.replace and not body.confirm_replace:
        raise _refusal(
            422,
            "replace discards one side of the item; pass confirm_replace: true to confirm",
        )
    if body.replace and body.prefer:
        raise _refusal(422, "pass either prefer or replace, not both")

    def work(ctx: Any) -> dict[str, Any]:
        checks = svc.check_items(ctx, kinds=(body.kind,), names=[body.name])
        if not checks:
            raise _refusal(404, f"no linked {body.kind} named {body.name!r}")
        report = svc.apply_items(
            ctx,
            kind=body.kind,
            checks=checks,
            prefer=body.prefer or "none",
            replace=body.replace,
            acknowledge_unknown_baseline=body.acknowledge_unknown_baseline,
            dry_run=body.dry_run,
        )
        _raise_for(report.reports)
        return {
            "reports": [r.to_json() for r in report.reports],
            "status": svc.status_snapshot(ctx),
        }

    async def mutate() -> dict[str, Any]:
        return await _run(request, config_manager, provider_auth_store, work)

    async with errors(request):
        return reply(
            await receipts(request).run(
                "hub-apply:" + body.request_id, body.model_dump(), mutate, retry_safe=True
            )
        )


@router.post("/v1/desktop/hub/updates/apply-all", response_model=CRUDResponse)
async def apply_all(
    body: HubApplyAll,
    request: Request,
    config_manager: ConfigManager = Depends(get_config_manager),
    provider_auth_store: AuthStore = Depends(get_provider_auth_store),
):
    """Apply every item currently ``available`` (B4.4): agents first, then teams.

    Cannot carry a conflict decision: a ``needs-review`` item is skipped, never
    forced, so a batch can never discard anyone's work.
    """

    from local_operator.hub_sync import service as svc

    def work(ctx: Any) -> dict[str, Any]:
        report = svc.apply_items(ctx, kind=body.kind, all_available=True)
        return {
            "reports": [r.to_json() for r in report.reports],
            "counts": report.counts(),
            "status": svc.status_snapshot(ctx),
        }

    async def mutate() -> dict[str, Any]:
        return await _run(request, config_manager, provider_auth_store, work)

    async with errors(request):
        return reply(
            await receipts(request).run(
                "hub-apply-all:" + body.request_id, body.model_dump(), mutate, retry_safe=True
            )
        )


@router.post("/v1/desktop/hub/updates/retry", response_model=CRUDResponse)
async def retry(
    body: HubRetry,
    request: Request,
    config_manager: ConfigManager = Depends(get_config_manager),
    provider_auth_store: AuthStore = Depends(get_provider_auth_store),
):
    """Clear the backoff for one item, then check and (per the auto setting) apply it."""

    from local_operator.hub_sync import service as svc

    def work(ctx: Any) -> dict[str, Any]:
        svc.clear_retry(ctx, body.kind, body.name)
        checks = svc.check_items(ctx, kinds=(body.kind,), names=[body.name])
        if not checks:
            raise _refusal(404, f"no linked {body.kind} named {body.name!r}")
        auto = ctx.settings().auto_for(body.kind)
        report = svc.apply_items(ctx, kind=body.kind, checks=checks, auto=True, dry_run=not auto)
        _raise_for(report.reports)
        if any(r.outcome == "would-merge" for r in report.reports):
            # Manual mode: the retry only PROVED the update computes. Retire the stale
            # failure so the row offers the update rather than another retry (U10).
            svc.clear_failure(ctx, body.kind, body.name)
        return {
            "reports": [r.to_json() for r in report.reports],
            "status": svc.status_snapshot(ctx),
        }

    async def mutate() -> dict[str, Any]:
        return await _run(request, config_manager, provider_auth_store, work)

    async with errors(request):
        return reply(
            await receipts(request).run(
                "hub-retry:" + body.request_id, body.model_dump(), mutate, retry_safe=True
            )
        )
