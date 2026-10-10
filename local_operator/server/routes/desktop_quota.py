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
  A forced refresh is additionally floored (``REFRESH_FLOOR_MS``) so a
  focus-refetch loop cannot spend a provider's per-IP budget.
- A timed-out or failed refresh falls back to whatever the cache holds; the
  verdict's own freshness rule (``quota_notice.report_is_fresh``, mirrored
  here) then decides whether it may say anything (usually it may not, and
  the response says ``unknown``).

THE FREE-MODEL RULE reads the pair's row from the CACHED LISTING first
(``initial_catalogue``, network-free — the same rows the picker paints), with
``entry_for``'s registry answer as the fallback: an aggregator ships no static
rows, so the listing is the only place a quoted ``0.0/0.0`` can come from.

THE RADIENT SENTENCE is fetched from the shared ``radient_recovery``
machinery ONLY once the verdict says ``depleted`` for Radient (its /me probe
is cached for 180 s, but a healthy user must not pay it), so no Radient
string is authored here.

The controller comes from ``get_desktop_auth`` and is closed in ``finally``;
every store read stays on the loop thread, for the sqlite thread-affinity
reason ``desktop_catalogues.py`` documents.
"""

from __future__ import annotations

import asyncio
import time
from collections.abc import Sequence
from typing import TYPE_CHECKING, Literal

from fastapi import APIRouter, Depends, HTTPException, Query
from pydantic import BaseModel

from local_operator.providers.billing_links import BillingKind
from local_operator.providers.quota_notice import (
    QuotaVerdict,
    evaluate_quota_notice,
    report_is_fresh,
)
from local_operator.providers.radient_recovery import get_recovery_facts, recovery_line
from local_operator.providers.registry import credential_provider_id
from local_operator.providers.usage import UsageReport
from local_operator.server.desktop import require_desktop
from local_operator.server.models.schemas import CRUDResponse
from local_operator.server.routes.auth import get_desktop_auth
from local_operator.server.routes.desktop_sessions import reply
from local_operator.server.utils.desktop_auth import DesktopAuth

if TYPE_CHECKING:  # runtime import would put providers.controller on the app boot path
    from local_operator.providers.controller import CatalogueEntry, ProviderController

router = APIRouter(tags=["Desktop quota"], dependencies=[Depends(require_desktop)])

#: Wall bound for the one live refresh this route may trigger. Sized under the
#: sensible interaction budget for a banner that appears while the user is
#: deciding to type: a slow provider must not hold the empty-session paint.
LIVE_REFRESH_BOUND_S = 4.0

#: Minimum interval between FORCED (``refresh=true``) live probes. The forced
#: path bypasses the TTL and the lease — right for one click, a rate-limit
#: hazard for a loop (the renderer refetches on focus; an empty session is
#: exactly when a user's attention is on the banner). Inside the floor the
#: answer is the cache the caller just asked about: still honest, because
#: ``source`` and ``age_ms`` say how old it is. Deliberately far under the
#: 5-minute TTL — a top-up checkout takes longer than this floor, so the
#: "I topped up" click still re-probes.
REFRESH_FLOOR_MS = 15_000


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
    """Whether the cached set is too old (or absent) to answer from.

    Mirrors ``quota_notice.report_is_fresh`` — the verdict is the authority on
    freshness and this is the route's spend decision, so they share one rule
    rather than two boundary spellings (round-1 m5).
    """
    if not reports:
        return True
    return any(not report_is_fresh(report, now_ms) for report in reports)


def _refresh_floor_blocks(reports: Sequence[UsageReport], now_ms: int) -> bool:
    """Whether a FORCED refresh is too soon after the newest cached report.

    Only the forced path is floored: a TTL-triggered refresh is already
    bounded by the TTL itself and is not what loops. An empty set is never
    blocked — the first open must be able to fetch.
    """
    newest = max((report.fetched_at for report in reports), default=0)
    return newest > 0 and now_ms - newest < REFRESH_FLOOR_MS


def _quota_model_entry(
    controller: "ProviderController", provider: str, model_id: str
) -> "CatalogueEntry | None":
    """The pair's catalogue row for the free-model rule, CACHED LISTING first.

    WHY NOT ``entry_for`` ALONE (round-1 M1). ``entry_for`` resolves
    ``static_models()`` plus the session spec; for an aggregator both are empty
    (no static rows ship for openrouter/radient), so every listing-derived
    ``:free`` route answered ``None`` and a spent account still got the "no
    balance" notice on a model it could send on. The quoted zeroes live in the
    CACHED LISTING's rows — the same ones the picker paints — so the free flag
    is read from the source that computes it, not re-derived here.
    ``initial_catalogue`` is synchronous and network-free (a peek at the
    document: ``cached_available_models``), which is what makes it safe on
    this path.

    The registry fallback keeps registry-described pairs (deepseek and
    friends) working when no listing document exists. NEVER-FABRICATE holds
    in both branches: each prices a stated zero to exactly ``0.0`` and a
    silence to the unknown ``-1``. The read is guarded — a broken frame must
    not take the notice route down with it.
    """
    try:
        rows = controller.initial_catalogue()
    except Exception:  # noqa: BLE001 — a broken frame falls back to the registry row
        rows = []
    for row in rows:
        if row.provider == provider and row.model_id == model_id:
            return row
    return controller.entry_for(provider, model_id)


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
        if _needs_live(reports, now_ms) or (refresh and not _refresh_floor_blocks(reports, now_ms)):
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
        expected = controller.expected_oauth_identities(resolved)
        api_key_present = bool(expected) and await _api_key_present(controller, resolved)
        entry = _quota_model_entry(controller, resolved, model_id) if model_id else None
        verdict = evaluate_quota_notice(
            provider=resolved,
            model=model_id,
            reports=reports,
            expected_identities=expected,
            api_key_present=api_key_present,
            entry=entry,
            now_ms=now_ms,
            radient_line=None,
        )
        if verdict.state == "depleted" and credential_provider_id(resolved) == "radient":
            # The /me probe runs AFTER the verdict (round-1 m3): only a
            # depleted Radient notice ever renders the sentence, so a healthy
            # user must not pay a probe — or wait behind one — for a body they
            # will never see. The probe is cache-fronted (180 s) and a timeout
            # leaves the module's neutral rendering to stand in.
            radient_line: str | None = None
            try:
                facts = await asyncio.wait_for(
                    get_recovery_facts(store=auth.store), timeout=LIVE_REFRESH_BOUND_S
                )
                radient_line = recovery_line(facts)
            except Exception:  # noqa: BLE001 — the module's neutral text stands in
                radient_line = None
            if radient_line is not None:
                verdict = evaluate_quota_notice(
                    provider=resolved,
                    model=model_id,
                    reports=reports,
                    expected_identities=expected,
                    api_key_present=api_key_present,
                    entry=entry,
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
