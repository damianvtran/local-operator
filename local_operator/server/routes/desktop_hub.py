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
from typing import Any, Callable, Literal, Mapping, NoReturn, TypeVar

from fastapi import APIRouter, Depends, HTTPException, Request
from pydantic import Field, StrictBool, model_validator

from local_operator.config import ConfigManager
from local_operator.hub_sync.identity import Visibility
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

Kind = Literal["agent", "team", "skill"]
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
    #: SOURCE-RESOLUTION ARM. ``resolve_source: true`` is the explicit act that
    #: confirms source+scope for a legacy unknown-origin link and materializes its
    #: immutable identity + guarded baseline. It is mutually exclusive with the
    #: merge arms above and never runs implicitly.
    resolve_source: bool = False
    hub_source: str | None = None
    visibility: Literal["public", "org"] | None = None
    tenant_id: str | None = Field(default=None, min_length=1, max_length=128)


class HubApplyAll(Input):
    request_id: RequestID
    kind: Kind | None = None


class HubRetry(Input):
    request_id: RequestID
    kind: Kind
    name: str = Field(min_length=1, max_length=128)


class SkillPull(Input):
    """``POST /v1/desktop/hub/skills/pull`` -- the UI's ``hub.skillPull`` arm.

    The body is the desktop contract's exact snake_case spelling (composed by
    ``desktop-contract.ts``'s ``hub.skillPull`` case and pinned by its zod
    schema): a request id, the published skill's ``hub_id``, the full
    credential-free API source, the paired scope (public carries no tenant, org
    requires one), the local alias, and the OPTIONAL initial policy the install
    dialog drafts. ``StrictBool`` on those three follows the policies module's
    own rule -- a policy value must refuse a truthy string by name, never
    coerce -- and the pairing guard in the handler is the server-side twin of
    the UI's ``assertPairedPublicationTarget``: the one field where failing
    open would be a disclosure.
    """

    request_id: RequestID
    hub_id: str = Field(min_length=1, max_length=128)
    hub_source: str = Field(min_length=1, max_length=2048)
    visibility: Literal["public", "org"]
    tenant_id: str | None = Field(default=None, min_length=1, max_length=128)
    local_alias: str = Field(min_length=1, max_length=128)
    subscribe: StrictBool | None = None
    auto_push: StrictBool | None = None
    confirm_public_egress: StrictBool | None = None
    #: ``form`` is optional on a pull: the REMOTE row decides the carrier, and a
    #: caller that names one is asserting what it expects to install (a mismatch
    #: refuses rather than silently installing the other form). ``dry_run`` returns
    #: the bundle summary and publisher and writes nothing.
    form: Literal["document", "bundle"] | None = None
    dry_run: StrictBool = False
    #: The bundle confirmation (design §6.2): the publisher id the prompt named
    #: and the exact archive hash, or ``confirm`` after both were shown.
    confirm_publisher_id: str | None = Field(default=None, min_length=1, max_length=256)
    confirm_bundle_hash: str | None = Field(default=None, min_length=64, max_length=64)
    confirm: StrictBool = False


class SkillSubscribe(Input):
    """``POST /v1/desktop/hub/skills/subscribe`` -- link without installing.

    Same remote identity fields as ``SkillPull`` and none of its install policy:
    a subscription is subscribe-on / auto-push-off by definition, so there is no
    policy to draft. The scope pairing is enforced server-side exactly as on the
    pull (public must not name a tenant; org must name one).
    """

    request_id: RequestID
    hub_id: str = Field(min_length=1, max_length=128)
    hub_source: str = Field(min_length=1, max_length=2048)
    visibility: Literal["public", "org"]
    tenant_id: str | None = Field(default=None, min_length=1, max_length=128)
    local_alias: str = Field(min_length=1, max_length=128)


class SkillUnpull(Input):
    """``POST /v1/desktop/hub/skills/unpull`` -- remove the copy, keep the link."""

    request_id: RequestID
    link_id: str = Field(min_length=1, max_length=256)


class SkillPublish(Input):
    """``POST /v1/desktop/hub/skills/publish`` -- the UI's ``hub.skillPublish`` arm.

    The body is the desktop contract's exact spelling (composed by
    ``desktop-contract.ts``'s ``hub.skillPublish`` case and pinned by its zod
    schema). ``source_skill`` names a locally authored skill from the SAME
    sessionless catalogue the picker reads; ``local_alias`` is the managed name
    the publication becomes. The paired scope rule is the pull route's.

    The receipt pairing is the contract's own: a DRY RUN may not carry a
    receipt or a confirmation, and a CONFIRM must carry both. Enforced here, as
    the renderer does, so an unreviewed publication cannot be expressed by
    omitting a field.
    """

    request_id: RequestID
    dry_run: bool
    source_skill: str = Field(min_length=1, max_length=128)
    local_alias: str = Field(min_length=1, max_length=128)
    visibility: Literal["public", "org"]
    tenant_id: str | None = Field(default=None, min_length=1, max_length=128)
    subscribe: StrictBool
    auto_push: StrictBool
    confirm_public_egress: StrictBool
    preview_receipt: str | None = Field(default=None, min_length=1, max_length=4096)
    confirm_publish: Literal[True] | None = None
    #: Explicit carrier selection (§6.3): ``--bundle`` / ``form:"bundle"``. Never
    #: an automatic fallback -- a text skill that outgrows document bounds must
    #: name the bundle form rather than route around the generalizer.
    form: Literal["document", "bundle"] = "document"

    @model_validator(mode="after")
    def _reviewed_draft(self) -> "SkillPublish":
        if self.dry_run:
            if self.preview_receipt is not None or self.confirm_publish is not None:
                raise ValueError("a dry-run publication cannot carry a confirmation")
        elif self.preview_receipt is None or self.confirm_publish is not True:
            raise ValueError("an initial skill publication requires its reviewed draft receipt")
        return self


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


def _refusal(status: int, message: str, *, code: str | None = None) -> HTTPException:
    """A refusal, TYPED when it carries a category code.

    The desktop plane's refusals carry ``{"code", "message"}`` as the detail
    (the desktop_mesh.py precedent): the renderer keys on
    ``DesktopControlError.code``, which desktop-api.ts reads from
    ``detail.code`` when detail is an object, and a declared code always wins
    over a status-derived one. A plain string body drops the category, so
    every refusal whose error has a LinkError-style ``code`` must use this
    form; routes without a code keep the plain string body unchanged.
    """

    if code is None:
        return HTTPException(status, message)
    return HTTPException(status, {"code": code, "message": message})


#: Bundle refusals that are a STATE problem, not a malformed request: the
#: caller's body was well-formed, but the current local/remote state cannot
#: accept it. Everything else named ``bundle-*`` is a 422 (an unusable or
#: unconfirmed input), and every code is still carried on the error body so the
#: renderer can word it by name.
_BUNDLE_STATE_REFUSALS = frozenset(
    {
        "bundle-local-changes",
        "bundle-conflict",
        "bundle-form-changed",
        "bundle-alias-rename-unsupported",
        "bundle-unsupported-server",
    }
)


def _raise_skill_refusal(error: Any) -> NoReturn:
    """The refusal vocabulary every skill route shares, mapped once.

    A missing or foreign remote row is ONE plain 404 (never a distinguishable
    403), a missing credential is the 401 whose remedy is signing in, a policy
    CAS loss is the 409 the link controls reload on, an unreachable hub is the
    502 their retry class covers, and every other named refusal is a 422 that
    carries its code.
    """

    code = getattr(error, "code", "") or ""
    if code == "no-credential":
        from local_operator.providers.radient_credentials import ORG_LOGIN_REMEDY

        raise _refusal(401, ORG_LOGIN_REMEDY) from None
    if code == "skill-not-found":
        raise _refusal(404, str(error), code=code) from None
    if code == "policy-revision-changed":
        raise _refusal(409, str(error), code=code) from None
    if code in _BUNDLE_STATE_REFUSALS:
        # The request is well-formed; the CURRENT STATE refuses it (local edits,
        # both sides moved, a flipped form, a server without the carrier).
        raise _refusal(409, str(error), code=code) from None
    if code == "hub-upstream":
        raise _refusal(502, str(error), code=code) from None
    raise _refusal(422, str(error), code=code or None) from None


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
        kinds = (body.kind,) if body.kind else ("agent", "team", "skill")
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

    if body.resolve_source:
        # Mutually exclusive with the merge arms: this arm writes identity/scope
        # metadata only and never merges content.
        if body.prefer or body.replace or body.dry_run:
            raise _refusal(422, "resolve_source cannot be combined with prefer/replace/dry_run")
        if not body.hub_source or not body.visibility:
            raise _refusal(422, "resolve_source needs hub_source and visibility")
        if body.visibility == "org" and not body.tenant_id:
            raise _refusal(422, "an org source resolution needs tenant_id")
        if body.visibility == "public" and body.tenant_id:
            raise _refusal(422, "a public source resolution must not name an organization")
        # Bind the guarded values ONCE, before the closure: pyright does not
        # carry the guard's narrowing into a nested function body, and these
        # are exactly the values the runtime checks above vouched for.
        requested_source: str = body.hub_source
        requested_visibility: Visibility = body.visibility

        def resolve_work(ctx: Any) -> dict[str, Any]:
            from local_operator.hub_sync.identity import LinkError as _IdentityError
            from local_operator.hub_sync.identity import approve_source
            from local_operator.providers.radient_credentials import (
                configured_radient_base_url,
            )

            # The supplied source must match the ONE configured credential-free
            # base -- never an arbitrary URL gaining credential egress.
            approved = (configured_radient_base_url(ctx.config_manager),)
            try:
                source = approve_source(requested_source, approved)
            except _IdentityError as error:
                raise _refusal(422, str(error), code=getattr(error, "code", "") or None) from None
            if ctx.credential == "none":
                raise _refusal(401, "no hub credential is configured for this destination")
            try:
                record = svc.resolve_source(
                    ctx,
                    kind=body.kind,
                    name=body.name,
                    hub_source=source,
                    visibility=requested_visibility,
                    tenant_id=body.tenant_id,
                )
            except _IdentityError as error:
                raise _refusal(422, str(error), code=getattr(error, "code", "") or None) from None
            except LookupError as error:
                # An unknown/unlinked name is a 404-able "no such thing", never
                # the AttributeError-500 this arm used to produce.
                raise _refusal(404, str(error)) from None
            # resolve_source MATERIALIZES the identity on the record it returns;
            # the Optional on the record type covers adopt/legacy records, not
            # this arm. An EXPLICIT check, not ``assert``: under -O a stripped
            # assert would return a half-built payload with a null link.
            identity = record.identity
            if identity is None:
                raise _refusal(500, "the source resolution did not record an identity")
            return {
                "link": identity.to_json(),
                "resolved": {
                    "source": identity.hub_source,
                    "visibility": identity.visibility,
                    "tenant_id": identity.tenant_id,
                    # The ACTUAL baseline provenance literal this arm writes. The
                    # resolve arm always records "adopt-unknown" (a source/scope
                    #  confirmed but no content common-ancestor vouched for); the
                    # separate equal-text path records "adopt". Exposed under a
                    # stable key so a client keys on the enum, not on a field name.
                    "baseline_state": record.recorded_by,
                    "scope_known": record.scope_known,
                },
            }

        async def resolve_mutate() -> dict[str, Any]:
            return await _run(request, config_manager, provider_auth_store, resolve_work)

        async with errors(request):
            return reply(
                await receipts(request).run(
                    "hub-resolve:" + body.request_id, body.model_dump(), resolve_mutate
                )
            )

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


class LinkPolicyPatch(Input):
    request_id: RequestID
    #: Both independent values are optional: a patch may set one and leave the
    #: other, but at least one must be present (validated in the service).
    subscribe: bool | None = None
    auto_push: bool | None = None
    #: The DELIBERATE public-egress consent: enabling auto-push on a PUBLIC origin
    #: additionally requires this explicit acknowledgement, which is bound into
    #: the persisted grant. Strict bool -- a truthy string cannot stand in for
    #: consent.
    confirm_public_egress: bool = False
    #: The authority for the write: the policy revision the editor last saw. A
    #: mismatch is a named 409 (the config was edited by someone else), never a
    #: last-write-wins clobber. Named ``expected_policy_revision`` so the client
    #: cannot confuse it with any other revision token.
    expected_policy_revision: str = Field(min_length=1, max_length=256)


@router.get("/v1/desktop/hub/links", response_model=CRUDResponse)
async def links(
    request: Request,
    config_manager: ConfigManager = Depends(get_config_manager),
    kind: str | None = None,
):
    """The full local-only links inventory: every link, healthy or not. No network.

    Gated on the ``hub_links:1`` capability so an older renderer never sees rows
    it cannot act on. Skill rows are NEW and capability-gated HERE: an old typed
    agent/team-only consumer asks for its kinds explicitly, so
    ``?kind=agent|team`` never returns a skill row. The ``/hub/updates``
    projection keeps its shape but DOES list skill rows now (that is the surface
    a skill failure is retried from); this route's own kind filter is what keeps
    the typed inventory clean.
    """

    from local_operator.hub_sync import service as svc
    from local_operator.hub_sync.identity import LinkError

    kinds: tuple[str, ...] | None = None
    if kind:
        if kind not in ("agent", "team", "skill"):
            raise _refusal(422, f"unknown link kind {kind!r}")
        kinds = (kind,)

    def read() -> dict[str, Any]:
        ctx = svc.HubSyncContext(
            config_dir=config_manager.config_dir,
            config_manager=config_manager,
            client_for_tenant=lambda _tenant: None,
        )
        return svc.links_snapshot(ctx, kinds=kinds)

    async with errors(request):
        try:
            return reply(await asyncio.to_thread(read))
        except OSError as exc:
            raise _refusal(503, f"the hub link store is unreadable: {exc}") from None
        except LinkError as error:
            # A malformed local policy/grant (read_policy validation) refuses the
            # INVENTORY with its own code instead of surfacing a bare 500; the
            # renderer keys on the code (F3).
            raise _refusal(422, str(error), code=getattr(error, "code", "") or None) from None


@router.patch("/v1/desktop/hub/links/{link_id}", response_model=CRUDResponse)
async def patch_link(
    link_id: str,
    body: LinkPolicyPatch,
    request: Request,
    config_manager: ConfigManager = Depends(get_config_manager),
):
    """Set one link's independent subscribe/auto_push, under a revision CAS.

    A policy edit is LOCAL config work (no network): turning auto_push ON demands
    a service eligibility attestation, but turning a value OFF or changing
    subscribe never requires a working hub connection.
    """

    from local_operator.hub_sync import provenance as prov
    from local_operator.hub_sync import service as svc
    from local_operator.hub_sync.identity import LinkError
    from local_operator.hub_sync.policies import update_policy

    def work() -> dict[str, Any]:
        record = None
        for candidate in prov.iter_baselines(config_manager.config_dir):
            if candidate.identity is not None and candidate.identity.link_id == link_id:
                record = candidate
                break
        if record is None or record.identity is None:
            raise _refusal(404, "no link with that id is known locally")
        # SERVICE-COMPUTED eligibility, not a hardcoded refusal: the same
        # fact the inventory reports, so a client that saw can_auto_push=true
        # cannot be refused here, and one that saw false cannot force it.
        ctx = svc.HubSyncContext(
            config_dir=config_manager.config_dir,
            config_manager=config_manager,
            client_for_tenant=lambda _t: None,
        )
        row = next(
            (link for link in svc.links_snapshot(ctx)["links"] if link["link_id"] == link_id),
            None,
        )
        eligible = bool(row and row["eligibility"]["can_auto_push"])
        reason = str((row or {}).get("eligibility", {}).get("reason") or "")
        try:
            snapshot = update_policy(
                config_manager,
                record.identity,
                expected_revision=body.expected_policy_revision,
                subscribe=body.subscribe,
                auto_push=body.auto_push,
                # The attestation is the SERVICE-computed eligibility; public
                # enable additionally demands the explicit egress consent below.
                # The REASON is passed through so ``update_policy`` applies its
                # ONE public-consent carve-out for the first grant (round-4
                # cross-surface fix); every earlier gate still refuses.
                can_enable_auto_push=eligible,
                confirm_public_egress=body.confirm_public_egress,
                ineligible_reason=reason or "unsupported-precondition",
            )
        except LinkError as error:
            code = getattr(error, "code", "") or ""
            # TYPED refusal: the renderer branches on `write.error.code` (e.g.
            # policy-revision-changed vs unsafe-publication), so the code must
            # ride the detail — a string body would flatten every refusal into
            # one untyped shape (F3).
            if code == "policy-revision-changed":
                raise _refusal(409, str(error), code=code) from None
            raise _refusal(422, str(error), code=code or None) from None
        return {"link_id": link_id, "policy": snapshot.to_json()}

    async def mutate() -> dict[str, Any]:
        return await asyncio.to_thread(work)

    async with errors(request):
        return reply(
            await receipts(request).run(
                "hub-link-policy:" + body.request_id,
                {"link_id": link_id, **body.model_dump()},
                mutate,
                retry_safe=True,
            )
        )


@router.post("/v1/desktop/hub/skills/pull", response_model=CRUDResponse)
async def skill_pull(
    body: SkillPull,
    request: Request,
    config_manager: ConfigManager = Depends(get_config_manager),
    provider_auth_store: AuthStore = Depends(get_provider_auth_store),
):
    """Install one remote skill as a NEW local link (the UI's ``hub.skillPull``).

    The scoped pair is enforced HERE as well as in the renderer: a public pull
    that names a tenant, or an org pull that omits one, is refused before any
    credential could be routed. Everything else is the service's: source
    approval against the ONE configured base (never an arbitrary URL gaining
    credential egress), the fetch through the same seam the linked checks use,
    the deterministic identity mint, the managed skill transaction (journaled,
    alias-ownership checked, fingerprint/policy CAS), and the inventory row the
    response carries. Replays are receipted like every other mutation, so a
    lost response is replayed, not re-installed.
    """

    if body.visibility == "public" and body.tenant_id:
        raise _refusal(422, "a public skill pull must not name an organization")
    if body.visibility == "org" and not body.tenant_id:
        raise _refusal(422, "an org skill pull needs tenant_id")

    from local_operator.hub_sync import service as svc
    from local_operator.hub_sync.identity import LinkError

    def work(ctx: Any) -> dict[str, Any]:
        try:
            return svc.install_skill(
                ctx,
                hub_source=body.hub_source,
                hub_id=body.hub_id,
                visibility=body.visibility,
                tenant_id=body.tenant_id,
                alias=body.local_alias,
                subscribe=body.subscribe,
                auto_push=body.auto_push,
                confirm_public_egress=bool(body.confirm_public_egress),
                confirm_publisher_id=body.confirm_publisher_id,
                confirm_bundle_hash=body.confirm_bundle_hash,
                confirm=bool(body.confirm),
                dry_run=bool(body.dry_run),
                form=body.form,
            )
        except LinkError as error:
            _raise_skill_refusal(error)

    async def mutate() -> dict[str, Any]:
        return await _run(request, config_manager, provider_auth_store, work)

    async with errors(request):
        return reply(
            await receipts(request).run(
                "hub-skill-pull:" + body.request_id, body.model_dump(), mutate, retry_safe=True
            )
        )


@router.post("/v1/desktop/hub/skills/subscribe", response_model=CRUDResponse)
async def skill_subscribe(
    body: SkillSubscribe,
    request: Request,
    config_manager: ConfigManager = Depends(get_config_manager),
    provider_auth_store: AuthStore = Depends(get_provider_auth_store),
):
    """Link a remote skill WITHOUT installing it (the UI's ``hub.skillSubscribe``).

    The paired scope is enforced HERE as well as in the renderer, like the pull:
    a public subscribe that names a tenant, or an org subscribe that omits one,
    is refused before any credential could be routed. The service then runs the
    SAME identity/fetch/envelope chain the pull runs (so a missing or foreign row
    is the same nondisclosing 404) and writes a baseline + subscription policy
    with no content. Replays are receipted, so a lost response returns the same
    link instead of creating a second one.
    """

    if body.visibility == "public" and body.tenant_id:
        raise _refusal(422, "a public skill subscribe must not name an organization")
    if body.visibility == "org" and not body.tenant_id:
        raise _refusal(422, "an org skill subscribe needs tenant_id")

    from local_operator.hub_sync import service as svc
    from local_operator.hub_sync.identity import LinkError

    def work(ctx: Any) -> dict[str, Any]:
        try:
            return svc.subscribe_skill(
                ctx,
                hub_source=body.hub_source,
                hub_id=body.hub_id,
                visibility=body.visibility,
                tenant_id=body.tenant_id,
                alias=body.local_alias,
            )
        except LinkError as error:
            _raise_skill_refusal(error)

    async def mutate() -> dict[str, Any]:
        return await _run(request, config_manager, provider_auth_store, work)

    async with errors(request):
        return reply(
            await receipts(request).run(
                "hub-skill-subscribe:" + body.request_id,
                body.model_dump(),
                mutate,
                retry_safe=True,
            )
        )


@router.post("/v1/desktop/hub/skills/unpull", response_model=CRUDResponse)
async def skill_unpull(
    body: SkillUnpull,
    request: Request,
    config_manager: ConfigManager = Depends(get_config_manager),
    provider_auth_store: AuthStore = Depends(get_provider_auth_store),
):
    """Remove one skill link's managed copy, keeping the link (``hub.skillUnpull``).

    The link row comes back with ``content_state: absent``; baseline, alias
    reservation, policy and subscription are untouched, so nothing is unlinked
    and no other origin can claim the alias. Repeating the request is a success
    (the intent is already true), never a 404.
    """

    from local_operator.hub_sync import service as svc
    from local_operator.hub_sync.identity import LinkError

    def work(ctx: Any) -> dict[str, Any]:
        try:
            return svc.unpull_skill(ctx, link_id=body.link_id)
        except LinkError as error:
            _raise_skill_refusal(error)

    async def mutate() -> dict[str, Any]:
        return await _run(request, config_manager, provider_auth_store, work)

    async with errors(request):
        return reply(
            await receipts(request).run(
                "hub-skill-unpull:" + body.request_id,
                body.model_dump(),
                mutate,
                retry_safe=True,
            )
        )


@router.post("/v1/desktop/hub/skills/publish", response_model=CRUDResponse)
async def skill_publish(
    body: SkillPublish,
    request: Request,
    config_manager: ConfigManager = Depends(get_config_manager),
    provider_auth_store: AuthStore = Depends(get_provider_auth_store),
):
    """Publish one locally authored skill as a NEW hub row (``hub.skillPublish``).

    Dry run projects the exact bytes a confirm would send and returns the
    preview receipt (no network); confirm re-reads the authored files, refuses
    drift, publishes through the credentialed client, then materializes the
    managed baseline/policy and answers the REAL ``links_snapshot`` row. The
    refused-publication path writes nothing local except the review receipt.
    """

    if body.visibility == "public" and body.tenant_id:
        raise _refusal(422, "a public skill publication must not name an organization")
    if body.visibility == "org" and not body.tenant_id:
        raise _refusal(422, "an org skill publication needs tenant_id")

    from local_operator.hub_sync import service as svc
    from local_operator.hub_sync.identity import LinkError

    def work(ctx: Any) -> dict[str, Any]:
        try:
            return svc.publish_skill(
                ctx,
                source_skill=body.source_skill,
                local_alias=body.local_alias,
                visibility=body.visibility,
                tenant_id=body.tenant_id,
                subscribe=body.subscribe,
                auto_push=body.auto_push,
                confirm_public_egress=body.confirm_public_egress,
                dry_run=body.dry_run,
                preview_receipt=body.preview_receipt,
                form=body.form,
            )
        except LinkError as error:
            code = getattr(error, "code", "") or ""
            if code == "no-credential":
                from local_operator.providers.radient_credentials import (
                    ORG_LOGIN_REMEDY,
                )

                raise _refusal(401, ORG_LOGIN_REMEDY) from None
            if code == "unauthorized":
                from local_operator.providers.radient_credentials import (
                    ORG_LOGIN_REMEDY,
                )

                raise _refusal(401, str(error), code=code) from None
            if code == "skill-not-found":
                raise _refusal(404, str(error), code=code) from None
            if code in (
                "preview_mismatch",
                "preview_expired",
                "preview_stale",
                "policy-revision-changed",
            ) or code in _BUNDLE_STATE_REFUSALS:
                raise _refusal(409, str(error), code=code) from None
            if code in ("hub-upstream", "hub-error"):
                raise _refusal(502, str(error), code=code) from None
            if code in ("not_a_member", "insufficient_role", "forbidden"):
                raise _refusal(403, str(error), code=code) from None
            raise _refusal(422, str(error), code=code or None) from None

    async def mutate() -> dict[str, Any]:
        result = await _run(request, config_manager, provider_auth_store, work)
        receipt = result.get("local_link_receipt")
        if isinstance(receipt, Mapping) and receipt.get("state") == "pending-reconciliation":
            # A PENDING PUBLICATION IS NOT AN OUTCOME FOR THIS REQUEST ID. The
            # remote row exists, but local reconciliation has not landed; recording
            # the pending answer would make the UI's same-id retry (its Confirm
            # shape) replay that answer forever instead of re-entering the
            # service, which is the only reconciler. Unclaimed withdraws the
            # claim inside the receipt journal's critical section and still
            # renders this answer, so the next attempt runs for real.
            from local_operator.server.utils.desktop_receipts import Unclaimed

            raise Unclaimed(result)
        return result

    async with errors(request):
        return reply(
            await receipts(request).run(
                "hub-skill-publish:" + body.request_id, body.model_dump(), mutate, retry_safe=True
            )
        )
