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
- Radient's sentence is NOT authored here, and neither is its classification.
  ``radient_recovery.recovery_line`` owns every Radient string and
  ``radient_recovery.account_state`` owns the verified / unverified /
  unreadable ladder both it and this module read; the caller passes the
  probed ``RecoveryFacts`` in. When none is passed the facts are "unreadable"
  and the neutral sentence stands, never a second string set.
- The resend button is an offer, not a claim: it appears only for an
  ``unverified`` account whose stored credential can actually call
  ``POST /auth/signup/resend`` (``resend_available``; that route accepts a
  Radient OAuth JWT only — see ``desktop_radient``'s ``signup.resend``), and
  only for a grant that has a ticket to reissue (``pending`` / ``expired``).
  Everything else degrades to the verification-page link.

STATE VOCABULARY. ``not_applicable`` (no endpoint, or a free model),
``unknown`` (no usable evidence), ``ok`` (evidence says an account can send),
``depleted`` (every account's balance/probe is empty — balance providers and
Radient), ``limit_reached`` (a plan window is spent — subscription
providers). ``unverified`` (Radient only) means the account is empty AND free
signup credits are waiting behind email verification, so verifying comes
before any top-up; it is derived from ``radient_recovery.account_state`` and
never guessed from sentence text.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Literal, Sequence

from local_operator.providers.billing_links import BillingKind, billing_link_for
from local_operator.providers.radient_recovery import (
    CLAIM_URL,
    RecoveryFacts,
    account_state,
    recovery_line,
)
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


def report_is_fresh(report: UsageReport, now_ms: int) -> bool:
    """One spelling of "this report is young enough to judge on".

    The VERDICT is the authority on freshness — it is the thing that refuses
    to speak on stale numbers — and the route mirrors this exact rule when it
    decides whether to spend a refresh before asking it, so the two cannot
    disagree at the TTL boundary (round-1 m5: the route spelled the rule
    ``>= TTL``, the verdict ``> TTL`` — one rule, two boundaries). Inclusive
    at the boundary on purpose: a report exactly TTL old still counts as
    fresh, and the route refetches only once the verdict itself would reject
    the row.
    """
    return report.fetched_at > 0 and now_ms - report.fetched_at <= USAGE_REPORT_TTL_MS


def evaluate_quota_notice(
    *,
    provider: str,
    model: str,
    reports: Sequence[UsageReport],
    expected_identities: Sequence[str] = (),
    api_key_present: bool = False,
    entry: CatalogueEntry | None = None,
    now_ms: int,
    radient_facts: RecoveryFacts | None = None,
    resend_available: bool = False,
) -> QuotaVerdict:
    """The one decision. Pure: no network, no store, no clock — pass ``now_ms``.

    ``reports`` is the provider's report list (cached or live; the caller has
    already tried a bounded refresh when the cache was stale). ``entry`` is
    ``ProviderController.entry_for(provider, model)``'s answer for the model
    the session is running — the free-model suppression's only input.
    ``expected_identities`` and ``api_key_present`` describe the credential
    set the reports must cover; both default empty so a caller checking a
    lone API-key provider passes neither. ``radient_facts`` is the probed
    ``/me`` verification state (Radient only; ``None`` reads as unreadable) and
    ``resend_available`` says the stored credential can call the resend route.
    """
    primary_link = billing_link_for(provider)
    default_kind: BillingKind = primary_link.kind if primary_link is not None else "none"
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
    if any(not report_is_fresh(report, now_ms) for report in reports):
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

    # The link is resolved AFTER the kind, and WITH it (round-1 M2): a
    # provider that sells two products has two surfaces, and the variant table
    # exists precisely for the kind the evidence named. Resolved earlier, that
    # table was dead code on this path and a spent Kimi plan linked to the
    # API-key top-up page. ``billing_link_for`` falls back to the primary
    # entry for a kind the provider does not sell.
    link = billing_link_for(provider, kind=kind)

    resets_after = [
        health.reset_after_ms for health in depleted_health if health.reset_after_ms is not None
    ]
    resets_at_ms = now_ms + min(resets_after) if resets_after else None
    age_ms = max(0, now_ms - max(report.fetched_at for report in reports))
    state: QuotaState = "limit_reached" if kind == "subscription" else "depleted"
    facts = radient_facts if radient_facts is not None else RecoveryFacts(signed_in=None)
    if kind == "radient" and account_state(facts) == "unverified":
        state = "unverified"
    title, body, actions = _compose(
        state=state,
        kind=kind,
        provider=provider,
        link_url=link.url if link is not None else None,
        dashboard=link.dashboard if link is not None else "",
        binding_labels=depleted_health[0].binding_labels,
        resets_at_ms=resets_at_ms,
        now_ms=now_ms,
        radient_facts=facts,
        resend_available=resend_available,
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
    radient_facts: RecoveryFacts,
    resend_available: bool,
) -> tuple[str, str, tuple[QuotaAction, ...]]:
    """The copy for a warning state. One home for every sentence."""
    definition = get_provider_definition(provider)
    brand = provider_brand(definition) if definition is not None else provider

    if kind == "radient":
        # The sentence and the unverified/verified split both come from
        # radient_recovery; this branch only turns them into buttons.
        line = recovery_line(radient_facts)
        verification = radient_facts.verification
        if state == "unverified" and verification is not None:
            claim = verification.claim_url or CLAIM_URL
            actions = [QuotaAction("open_url", "Open verification page", url=claim)]
            if resend_available and verification.signup_grant in ("pending", "expired"):
                # ``none`` has no ticket to reissue (the route would answer
                # 409); ``email_verified: false`` alone proves nothing about one.
                actions.append(QuotaAction("resend_verification", "Resend verification email"))
            actions.append(QuotaAction("refresh", "I verified"))
            return ("Verify your email to claim your free credits", line, tuple(actions))
        # Verified or unreadable: the remedy is a top-up. A verified payload's
        # own (https-checked) top-up URL wins over the table's constant.
        topup = (
            verification.first_topup.topup_url
            if verification is not None and verification.first_topup is not None
            else None
        ) or link_url
        actions = [QuotaAction("refresh", "I topped up")]
        if topup:
            actions.insert(0, QuotaAction("open_url", "Top up at Radient", url=topup))
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
