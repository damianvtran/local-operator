"""The advisory "no quota" notice: verdict + copy, one pure function.

WHY THIS EXISTS. A session's very first send can be refused for a reason the
user could have seen BEFORE typing: the account has no credit left, or its
plan window is spent. Every surface that can answer "why won't this send"
already exists (``usage_health`` reads the reports, the controller holds the
cache, ``entry_for`` prices the running model) — what was missing is the one
decision "does the evidence justify a notice, and what does it say", made
identically wherever it is asked. The desktop route and (later) the TUI
cached read both call :func:`evaluate_quota_notice`; neither authors a
sentence of its own.

THE HONESTY RULES, each one load-bearing:

- Warn only on FRESH, DEFINITE evidence. A report older than the cache TTL, a
  report the fetchers marked ``usage_unavailable`` or ``credential_invalid``,
  a report ``usage_health`` cannot reduce (``unknown``), or a report set that
  does not cover every expected account — each returns ``unknown`` and the
  caller shows nothing. Balances can be topped up out of band, so a stale
  "empty" verdict is a claim we are not entitled to make.
- EVERY account must be depleted. One usable sibling account is a working
  provider, and a notice telling a multi-account user their provider is empty
  when a second login would send is worse than no notice — it is misleading
  advice.
- A window whose ``resets_at_ms`` has already passed cannot support a
  depleted verdict: it may have refilled since the fetch. Same rule the
  account preflight applies (``auth_store``), scoped to the windows that
  would have to be spent for THIS verdict.
- A model the pricing knows to be free (both prices an explicit ``0.0``, and
  not a meta-route) suppresses the notice entirely — "0.0" is a quoted zero,
  "-1" is unknown, and ``routed`` meta-routes are never free. The user can
  still send on a free model, so a warning would be false; the suppression is
  total rather than softened so there is never a half-warning.
- Mixed credentials whose other route was NOT fetched suppress the notice:
  when a provider has an OAuth login AND a live API key, the fetch only runs
  the OAuth route (see ``ProviderController._fetch_provider``), so the
  API-key half is unproven. This is reachable on Kimi (whose two routes are
  genuinely different products) and, conservatively, on Z.AI (whose two
  routes run the same fetcher — a missed notice there is the accepted cost,
  because a false "empty" costs more than a missed one).
- Radient's sentence is NOT authored here. ``radient_recovery.recovery_line``
  owns every Radient string (its remedy depends on account facts this module
  cannot see), and the caller passes the fetched line in. When none is passed
  the module's own neutral rendering is used, never a second string set.

STATE VOCABULARY. ``not_applicable`` (no endpoint, or a free model),
``unknown`` (no usable evidence), ``ok`` (evidence says an account can send),
``depleted`` (every account's balance/probe is empty — balance providers and
Radient), ``limit_reached`` (a plan window is spent — subscription
providers). ``unverified`` is RESERVED for the Radient verification work
(PR2): it will be produced from the shared ``account_state`` classifier once
that exists, and PR1 deliberately does not guess it from ``recovery_line``
text.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Literal, Sequence

from local_operator.providers.billing_links import BillingKind, billing_link_for
from local_operator.providers.radient_recovery import RecoveryFacts, recovery_line
from local_operator.providers.registry import get_provider_definition, provider_brand
from local_operator.providers.usage import (
    UsageReport,
    usage_health,
    usage_kinds,
    usage_supported,
)
from local_operator.providers.usage_cache import USAGE_REPORT_TTL_MS

if TYPE_CHECKING:  # pragma: no cover - import weight guard, not a behaviour
    from local_operator.providers.controller import CatalogueEntry

QuotaState = Literal[
    "ok",
    "depleted",
    "limit_reached",
    "unverified",
    "unknown",
    "not_applicable",
]


@dataclass(frozen=True)
class QuotaAction:
    """One call to action the notice offers. ``url`` is present for open_url."""

    id: Literal["open_url", "resend_verification", "refresh"]
    label: str
    url: str | None = None


@dataclass(frozen=True)
class QuotaVerdict:
    """What the evidence says, and the copy for it. Empty copy = show nothing.

    ``age_ms`` is the age of the NEWEST report behind the verdict — what a
    surface would print as "checked 40s ago". ``resets_at_ms`` is absolute
    epoch-ms of the earliest known window reset (the moment the account is
    usable again), absent when the evidence carries no reset time.
    """

    state: QuotaState
    kind: BillingKind
    model_free: bool
    title: str = ""
    body: str = ""
    actions: tuple[QuotaAction, ...] = ()
    resets_at_ms: int | None = None
    age_ms: int | None = None


def evaluate_quota_notice(
    *,
    provider: str,
    model: str,
    reports: Sequence[UsageReport],
    expected_identities: Sequence[str] = (),
    api_key_present: bool = False,
    entry: CatalogueEntry | None = None,
    now_ms: int,
    radient_line: str | None = None,
) -> QuotaVerdict:
    """The one decision. Pure: no network, no store, no clock — pass ``now_ms``.

    ``reports`` is the provider's report list (cached or live; the caller has
    already tried a bounded refresh when the cache was stale). ``entry`` is
    ``ProviderController.entry_for(provider, model)``'s answer for the model
    the session is running — the free-model suppression's only input.
    ``expected_identities`` and ``api_key_present`` describe the credential
    set the reports must cover; both default empty so a caller checking a
    lone API-key provider passes neither.
    """
    link = billing_link_for(provider)
    default_kind: BillingKind = link.kind if link is not None else "none"
    model_free = (
        entry is not None
        and entry.input_price == 0.0
        and entry.output_price == 0.0
        and not entry.routed
    )

    if not usage_supported(provider):
        # No endpoint, no evidence, ever: API-key openai/anthropic/google/
        # mistral/xai live here. Their remedy is the post-failure link only.
        return QuotaVerdict("not_applicable", default_kind, model_free)
    if model_free:
        return QuotaVerdict("not_applicable", default_kind, True)

    if not reports:
        return QuotaVerdict("unknown", default_kind, model_free)
    if any(
        report.fetched_at <= 0 or now_ms - report.fetched_at > USAGE_REPORT_TTL_MS
        for report in reports
    ):
        # One stale report poisons the set: we can neither trust its numbers
        # nor prove the account behind it is covered.
        return QuotaVerdict("unknown", default_kind, model_free)

    if expected_identities:
        seen = {report.identity for report in reports if report.identity}
        uncovered = [identity for identity in expected_identities if identity not in seen]
        if uncovered:
            # A report whose identity is None cannot be matched by label, but
            # it IS a report for one of this provider's accounts (the fetch
            # enumerates per account): an account row with no email and no
            # account_id lands here, its identity_key being unusable as a
            # label. Count those reports as covering the unmatched accounts —
            # if the numbers do not add up, an account truly has no report
            # and "every account is dead" is unprovable.
            blank = sum(1 for report in reports if not report.identity)
            if blank < len(uncovered):
                # A logged-in account with no report at all: the payload is
                # incomplete, so "every account is dead" is unprovable.
                return QuotaVerdict("unknown", default_kind, model_free)
        oauth_route, api_route = usage_kinds(provider)
        if api_key_present and oauth_route and api_route:
            # An OAuth login AND a live API key, where the API-key route
            # exists but was NOT fetched (the fetcher only runs it when no
            # OAuth identity is stored — Kimi's two products, Z.AI's two
            # sign-ins). The skipped route's numbers are unproven, so the
            # verdict cannot speak for the account. Conservative by design:
            # a same-fetcher pair (Z.AI runs one fetcher for both routes)
            # loses a notice it could perhaps have earned, and a false
            # "empty" costs more than a missed one.
            return QuotaVerdict("unknown", default_kind, model_free)

    depleted_health = []
    for report in reports:
        if report.usage_unavailable or report.credential_invalid:
            return QuotaVerdict("unknown", default_kind, model_free)
        health = usage_health(report, model, now_ms=now_ms)
        if health.state == "unknown":
            return QuotaVerdict("unknown", default_kind, model_free)
        if health.state != "depleted":
            # An account that can still accept a message.
            return QuotaVerdict("ok", default_kind, model_free)
        if _spent_window_rolled_over(report, now_ms):
            return QuotaVerdict("unknown", default_kind, model_free)
        depleted_health.append(health)

    kind: BillingKind = default_kind
    if default_kind != "radient":
        # Report-shaped rather than table-shaped for the one provider whose
        # two credentials report two products: evidence from a ROLLING window
        # (a measurable row carrying a reset instant) is a subscription,
        # anything else is a balance. Kimi resolves through this (its coding
        # plan reports windows, its API key reports a balance); OpenRouter's
        # lifetime spend cap stays a balance here, which is why the
        # discriminator is the reset instant and not "is the fraction
        # measurable" — a key cap has a fraction and never resets.
        windowed = any(_has_rolling_window(report) for report in reports)
        kind = "subscription" if windowed else "balance"

    resets_after = [
        health.reset_after_ms for health in depleted_health if health.reset_after_ms is not None
    ]
    resets_at_ms = now_ms + min(resets_after) if resets_after else None
    age_ms = max(0, now_ms - max(report.fetched_at for report in reports))
    state: QuotaState = "limit_reached" if kind == "subscription" else "depleted"
    title, body, actions = _compose(
        state=state,
        kind=kind,
        provider=provider,
        link_url=link.url if link is not None else None,
        dashboard=link.dashboard if link is not None else "",
        binding_labels=depleted_health[0].binding_labels,
        resets_at_ms=resets_at_ms,
        now_ms=now_ms,
        radient_line=radient_line,
    )
    return QuotaVerdict(
        state=state,
        kind=kind,
        model_free=model_free,
        title=title,
        body=body,
        actions=actions,
        resets_at_ms=resets_at_ms,
        age_ms=age_ms,
    )


def _has_rolling_window(report: UsageReport) -> bool:
    """Whether this report's evidence includes a rolling plan window.

    A rolling window is a measurable row WITH a reset instant: the thing a
    subscription plan refills on its own. A spend cap (OpenRouter's key
    limit) is measurable but never resets — you must raise it or add credit —
    so it stays balance-shaped, and the copy that says "top up" is the true
    one for it.
    """
    return any(
        limit.amount.fraction() is not None and limit.resets_at_ms is not None
        for limit in report.limits
    )


def _spent_window_rolled_over(report: UsageReport, now_ms: int) -> bool:
    """A spent window whose reset time has already passed — no longer evidence.

    Scoped to the windows that would have to stay spent for THIS verdict: a
    measurable row at 100%, or a denominator-less balance at or below zero.
    Their timestamps say "usable again at T"; past T the number may simply be
    old (balances refill, windows roll), so the report stops being able to
    support "depleted".
    """
    for limit in report.limits:
        fraction = limit.amount.fraction()
        spent = (fraction is not None and fraction >= 1.0) or (
            fraction is None and limit.amount.remaining is not None and limit.amount.remaining <= 0
        )
        if spent and limit.resets_at_ms is not None and limit.resets_at_ms <= now_ms:
            return True
    return False


def _compose(
    *,
    state: QuotaState,
    kind: BillingKind,
    provider: str,
    link_url: str | None,
    dashboard: str,
    binding_labels: Sequence[str],
    resets_at_ms: int | None,
    now_ms: int,
    radient_line: str | None,
) -> tuple[str, str, tuple[QuotaAction, ...]]:
    """The copy for a warning state. One home for every sentence."""
    definition = get_provider_definition(provider)
    brand = provider_brand(definition) if definition is not None else provider

    if kind == "radient":
        line = radient_line or recovery_line(RecoveryFacts(signed_in=None))
        actions = [QuotaAction("refresh", "I topped up")]
        if link_url:
            actions.insert(0, QuotaAction("open_url", "Top up at Radient", url=link_url))
        return ("No credit left on Radient", line, tuple(actions))

    if state == "limit_reached":
        when = _reset_phrase(resets_at_ms, now_ms)
        label = binding_labels[0].lower() if binding_labels else ""
        body = f"{brand} limit reached — {label + ' ' if label else ''}resets {when}."
        actions = [QuotaAction("refresh", "Check again")]
        if link_url:
            actions.insert(0, QuotaAction("open_url", f"Open {dashboard}", url=link_url))
        return (f"{brand} limit reached", body, tuple(actions))

    body = (
        f"No balance on {brand} — top up at {dashboard}."
        if link_url
        else f"No balance on {brand} — top up to keep using this provider."
    )
    actions = [QuotaAction("refresh", "I topped up")]
    if link_url:
        actions.insert(0, QuotaAction("open_url", f"Top up at {dashboard}", url=link_url))
    return (f"No balance on {brand}", body, tuple(actions))


def _reset_phrase(resets_at_ms: int | None, now_ms: int) -> str:
    """Coarse human phrase for a reset instant. Coarse on purpose: the exact
    time is in ``resets_at_ms`` for a renderer that wants it, and a body
    string is generated once and must stay true as minutes pass."""
    if resets_at_ms is None:
        return "after the window rolls over"
    remaining_ms = max(0, resets_at_ms - now_ms)
    minutes = remaining_ms // 60_000
    if minutes < 2:
        return "in under a minute"
    if minutes < 90:
        return f"in {minutes} min"
    hours = remaining_ms // 3_600_000
    if hours < 48:
        rest = (remaining_ms % 3_600_000) // 60_000
        return f"in {hours}h {rest}m" if rest else f"in {hours}h"
    return f"in {remaining_ms // 86_400_000} days"
