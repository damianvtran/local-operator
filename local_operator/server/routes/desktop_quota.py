"""``GET /v1/desktop/quota-notice`` — the pre-emptive "no quota" read.

WHY A ROUTE OF ITS OWN. The desktop renderer wants one question answered on
an empty session: "is there definite, fresh evidence the account behind the
selected model cannot send?" The evidence lives behind the same machinery
``GET /v1/desktop/usage`` exposes, but the answer is a decision (see
``providers/quota_notice.py``) plus the copy for it — not a report dump. Two
reasons this is not folded into ``desktop_catalogues.py``: the model-access
work edits that module, and the usage route returns raw reports INCLUDING
``identity``, which can be an email. This route is deliberately narrower:
it answers with the verdict and never echoes a credential's identity.

CONTRACT (additive; the UI degrades on a 404, so an old backend is a no-op):
``{state, provider, kind, model_free, title, body, actions, resets_at_ms?,
checked_at_ms, age_ms, source}`` — see ``QUOTA_NOTICE_SCHEMA`` in the
renderer's contract module for the UI-side twin. States ``depleted``,
``limit_reached`` and ``unverified`` are the ones that show a notice; ``ok``,
``unknown`` and ``not_applicable`` are all "show nothing", distinguished for
diagnostics. ``unverified`` is reserved for the Radient verification round.

HOW THE EVIDENCE IS GATHERED. Cache-first with one bounded refresh:
- A fresh cache row answers with no network (``source: "cached"``).
- ``refresh=true``, an empty cache, or a stale row triggers one live fetch
  under ``asyncio.wait_for`` bound — the fetchers' own HTTP timeout is 10 s,
  far too slow for a banner on session open, and the cross-process lease
  inside ``fetch_usage`` keeps N sessions from fanning out at once anyway.
- A timed-out or failed refresh falls back to whatever the cache holds; the
  verdict's own freshness rule then decides whether it may say anything
  (usually it may not, and the response says ``unknown``).

The controller comes from ``get_desktop_auth`` and is closed in ``finally``;
every store read stays on the loop thread, for the sqlite thread-affinity
reason ``desktop_catalogues.py`` documents. The Radient sentence is fetched
from the shared ``radient_recovery`` machinery (its /me probe is cached for
180 s), so no Radient string is authored here.
"""

from __future__ import annotations

import asyncio
import time
from collections.abc import Sequence
from typing import Literal

from fastapi import APIRouter, Depends, HTTPException, Query
from pydantic import BaseModel

from local_operator.providers.billing_links import BillingKind
from local_operator.providers.quota_notice import QuotaVerdict, evaluate_quota_notice
from local_operator.providers.radient_recovery import get_recovery_facts, recovery_line
from local_operator.providers.registry import credential_provider_id
from local_operator.providers.usage import UsageReport
from local_operator.providers.usage_cache import USAGE_REPORT_TTL_MS
from local_operator.server.desktop import require_desktop
from local_operator.server.models.schemas import CRUDResponse
from local_operator.server.routes.auth import get_desktop_auth
from local_operator.server.routes.desktop_sessions import reply
from local_operator.server.utils.desktop_auth import DesktopAuth

router = APIRouter(tags=["Desktop quota"], dependencies=[Depends(require_desktop)])

#: Wall bound for the one live refresh this route may trigger. Sized under the
#: sensible interaction budget for a banner that appears while the user is
#: deciding to type: a slow provider must not hold the empty-session paint.
LIVE_REFRESH_BOUND_S = 4.0


class QuotaActionResponse(BaseModel):
    """One call to action. ``resend_verification`` is reserved (Radient round)."""

    id: Literal["open_url", "resend_verification", "refresh"]
    label: str
    url: str | None = None


class QuotaNoticeResult(BaseModel):
    """The whole answer. No identity, no email — see the module docstring."""

    state: Literal["ok", "depleted", "limit_reached", "unverified", "unknown", "not_applicable"]
    provider: str
    kind: BillingKind
    model_free: bool
    title: str
    body: str
    actions: list[QuotaActionResponse]
    resets_at_ms: int | None = None
    checked_at_ms: int
    age_ms: int | None = None
    source: Literal["cached", "live"]


def _now_ms() -> int:
    return int(time.time() * 1000)


def _needs_live(reports: Sequence[UsageReport], now_ms: int) -> bool:
    """Whether the cached set is too old (or absent) to answer from."""
    if not reports:
        return True
    return any(
        report.fetched_at <= 0 or now_ms - report.fetched_at >= USAGE_REPORT_TTL_MS
        for report in reports
    )


def _response(
    verdict: QuotaVerdict,
    *,
    provider: str,
    source: Literal["cached", "live"],
    checked_at_ms: int,
) -> QuotaNoticeResult:
    return QuotaNoticeResult(
        state=verdict.state,
        provider=provider,
        kind=verdict.kind,
        model_free=verdict.model_free,
        title=verdict.title,
        body=verdict.body,
        actions=[QuotaActionResponse(id=a.id, label=a.label, url=a.url) for a in verdict.actions],
        resets_at_ms=verdict.resets_at_ms,
        checked_at_ms=checked_at_ms,
        age_ms=verdict.age_ms,
        source=source,
    )


@router.get("/v1/desktop/quota-notice", response_model=CRUDResponse[QuotaNoticeResult])
async def quota_notice(
    provider: str | None = Query(default=None, max_length=64),
    model: str | None = Query(default=None, max_length=200),
    refresh: bool = False,
    auth: DesktopAuth = Depends(get_desktop_auth),
):
    manager = auth.config_manager
    requested = provider or (str(manager.get_config_value("hosting") or "") if manager else "")
    model_id = model or (str(manager.get_config_value("model_name") or "") if manager else "")
    if not requested:
        # No selection to check — the same "show nothing" the client does for
        # every non-warning state, so a renderer needs no extra branch.
        blank = QuotaVerdict("not_applicable", "none", False)
        return reply(_response(blank, provider="", source="cached", checked_at_ms=_now_ms()))

    controller = auth.controller()
    try:
        definition = controller.provider(requested)
        if definition is None:
            raise HTTPException(422, "Unknown provider")
        resolved = definition.id

        reports: list[UsageReport] = controller.cached_usage_reports(resolved)
        source: Literal["cached", "live"] = "cached"
        now_ms = _now_ms()
        if refresh or _needs_live(reports, now_ms):
            attempted_at = _now_ms()
            try:
                live = await asyncio.wait_for(
                    controller.fetch_usage([resolved], force_refresh=refresh),
                    timeout=LIVE_REFRESH_BOUND_S,
                )
            except Exception:  # noqa: BLE001 — a failed refresh falls back to the cache row
                live = None
            if live:
                reports = live
                # ``source`` says where the ANSWER'S DATA came from, not that a
                # network call happened: a failed probe serves the previous
                # numbers (``_fetch_provider_cached`` keeps last-good), so a
                # payload nothing refreshed is still a cached answer. An empty
                # fetch is likewise not a refresh — it has no evidence to
                # overrule the cache with.
                source = "live" if any(r.fetched_at >= attempted_at for r in live) else "cached"

        now_ms = _now_ms()
        radient_line: str | None = None
        if credential_provider_id(resolved) == "radient":
            # The one provider whose sentence is owned elsewhere. The probe is
            # cache-fronted (180 s) and never raises; a timeout leaves the
            # module's neutral rendering to stand in rather than a blank.
            try:
                facts = await asyncio.wait_for(
                    get_recovery_facts(store=auth.store), timeout=LIVE_REFRESH_BOUND_S
                )
                radient_line = recovery_line(facts)
            except Exception:  # noqa: BLE001 — the module's neutral text stands in
                radient_line = None

        expected = controller.expected_oauth_identities(resolved)
        verdict = evaluate_quota_notice(
            provider=resolved,
            model=model_id,
            reports=reports,
            expected_identities=expected,
            api_key_present=bool(expected) and await _api_key_present(controller, resolved),
            entry=controller.entry_for(resolved, model_id) if model_id else None,
            now_ms=now_ms,
            radient_line=radient_line,
        )
        return reply(_response(verdict, provider=resolved, source=source, checked_at_ms=now_ms))
    finally:
        controller.close()


async def _api_key_present(controller, provider: str) -> bool:
    """Whether a NON-OAuth API-key credential exists for ``provider``.

    The mixed-credentials rule needs "would the skipped API-key route have had
    something to spend": the environment value the registry resolves for the
    provider (``resolve_env_key``, alias-aware), or an ``api_key``-typed row
    in the auth store. ``AuthStore.get_api_key`` is deliberately NOT the
    probe: its cascade returns an OAuth row's own access token for Kimi, which
    is the credential the OAuth route ALREADY fetched — counting it would
    make every OAuth-only login look "mixed" and silence the notice forever.

    An unreadable store answers ``True``: it cannot vouch that the key is
    ABSENT, and absence is the half the notice needs — silence is the honest
    direction every other unreadable input on this route takes too. The
    encrypted secret store's provider rows are not consulted (a read can
    start its broker from a banner path); a credential that lives only there
    keeps today's behaviour on the OAuth route.
    """
    from local_operator.providers.registry import resolve_env_key

    try:
        if resolve_env_key(provider):
            return True
    except Exception:  # noqa: BLE001 — an unreadable env answer cannot prove absence
        return True
    try:
        rows = controller.auth_store.list_credentials(credential_provider_id(provider))
    except Exception:  # noqa: BLE001 — unreadable store cannot prove the absence
        return True
    return any(getattr(row, "credential_type", None) == "api_key" for row in rows)
