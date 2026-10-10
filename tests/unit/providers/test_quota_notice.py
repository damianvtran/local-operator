"""The pre-emptive notice verdict: the matrix the design's §2 rules describe.

WHY THESE EXIST. ``evaluate_quota_notice`` is the one place that decides
whether the harness may tell a user "no quota" before a send is refused, and
every rule it applies exists because the opposite behaviour is a lie:
- a stale or unavailable report is not evidence (balances change out of band);
- one healthy account means the provider works (multi-account users);
- a spent window that has since reset proves nothing;
- a free model can send even on an empty account, so the notice is suppressed
  entirely rather than softened;
- the shared Radient sentence is the module's, not a second copy.

The fixtures are canned ``UsageReport`` objects — this function is pure (no
clock, no store, no network), which is exactly why the matrix can be this
wide. Route-level behaviour (caching, live refresh, auth) is pinned in
``tests/unit/server/test_desktop_quota_notice.py``.
"""

from __future__ import annotations

import pytest

from local_operator.providers.billing_links import BILLING_LINKS
from local_operator.providers.controller import CatalogueEntry
from local_operator.providers.quota_notice import evaluate_quota_notice
from local_operator.providers.radient_recovery import (
    CLAIM_URL,
    FirstTopupFacts,
    RecoveryFacts,
    VerificationFacts,
    account_state,
    recovery_line,
)
from local_operator.providers.usage import UsageAmount, UsageLimit, UsageReport
from local_operator.providers.usage_cache import USAGE_REPORT_TTL_MS

#: Frozen "now" for the whole matrix; every timestamp derives from it.
NOW_MS = 1_800_000_000_000


def _balance_row(remaining: float, *, resets_at_ms: int | None = None) -> UsageLimit:
    """A denominator-less balance row, the shape DeepSeek/Kimi/Radient emit."""
    return UsageLimit(
        id="probe:balance",
        label="Credit balance (USD)",
        amount=UsageAmount(remaining=remaining),
        window="lifetime",
        resets_at_ms=resets_at_ms,
        shared=True,
    )


def _window_row(
    *,
    used: float,
    total: float,
    resets_at_ms: int | None,
    label: str = "5 hour",
    window: str = "5 hour",
) -> UsageLimit:
    """A measured plan window, the shape Anthropic/OpenAI/Kimi-OAuth emit."""
    return UsageLimit(
        id="probe:window",
        label=label,
        amount=UsageAmount(used=used, limit=total, used_fraction=used / total),
        window=window,
        resets_at_ms=resets_at_ms,
        shared=True,
    )


def _report(
    provider: str,
    *rows: UsageLimit,
    identity: str | None = None,
    fetched_at: int = NOW_MS,
    usage_unavailable: bool = False,
    credential_invalid: bool = False,
) -> UsageReport:
    return UsageReport(
        provider=provider,
        fetched_at=fetched_at,
        limits=list(rows),
        identity=identity,
        usage_unavailable=usage_unavailable,
        credential_invalid=credential_invalid,
    )


def _entry(
    provider: str, model_id: str, *, input_price: float, output_price: float, routed: bool = False
) -> CatalogueEntry:
    return CatalogueEntry(
        provider=provider,
        model_id=model_id,
        label=model_id,
        context_window=64_000,
        input_price=input_price,
        output_price=output_price,
        connected=True,
        routed=routed,
    )


def test_depleted_balance_carries_the_topup_link() -> None:
    verdict = evaluate_quota_notice(
        provider="deepseek",
        model="deepseek-chat",
        reports=[_report("deepseek", _balance_row(0.0))],
        now_ms=NOW_MS,
    )
    assert verdict.state == "depleted"
    assert verdict.kind == "balance"
    assert verdict.title == "No balance on DeepSeek"
    assert verdict.body == "No balance on DeepSeek — top up at the DeepSeek platform."
    assert [a.id for a in verdict.actions] == ["open_url", "refresh"]
    assert verdict.actions[0].url == BILLING_LINKS["deepseek"].url


def test_positive_balance_is_not_depleted_and_shows_nothing() -> None:
    """A funded, fraction-less balance is no notice — and NOT a false "ok".

    ``usage_health`` deliberately cannot convert a bare positive balance into
    a percentage (denominator-less rows are depleted only when every row is
    <= 0), so the report reads ``unknown`` here. That is the honest answer:
    the notice's question is answered (not empty), the health question is not,
    and both non-depleted states show nothing.
    """
    verdict = evaluate_quota_notice(
        provider="deepseek",
        model="deepseek-chat",
        reports=[_report("deepseek", _balance_row(5.0))],
        now_ms=NOW_MS,
    )
    assert verdict.state == "unknown"
    assert verdict.title == "" and verdict.body == ""


def test_measured_headroom_is_a_definite_ok() -> None:
    """A measured window with headroom IS provable: state ok, no copy."""
    verdict = evaluate_quota_notice(
        provider="anthropic",
        model="claude-opus-4",
        reports=[_report("anthropic", _window_row(used=20, total=100, resets_at_ms=None))],
        now_ms=NOW_MS,
    )
    assert verdict.state == "ok"
    assert verdict.title == "" and verdict.body == ""


def test_mixed_currency_balance_is_unknown_not_depleted() -> None:
    """One zeroed wallet beside a funded one is not an empty account.

    ``usage_health`` refuses to guess here (it cannot know the currencies'
    relation), and the notice must not escalate an indeterminable report.
    """
    verdict = evaluate_quota_notice(
        provider="deepseek",
        model="deepseek-chat",
        reports=[_report("deepseek", _balance_row(0.0), _balance_row(20.0))],
        now_ms=NOW_MS,
    )
    assert verdict.state == "unknown"


def test_spent_window_that_already_reset_is_unknown() -> None:
    """A depleted window whose reset instant has passed may have refilled."""
    verdict = evaluate_quota_notice(
        provider="anthropic",
        model="claude-opus-4",
        reports=[
            _report(
                "anthropic",
                _window_row(used=100, total=100, resets_at_ms=NOW_MS - 60_000),
            )
        ],
        now_ms=NOW_MS,
    )
    assert verdict.state == "unknown"


def test_stale_report_is_unknown() -> None:
    verdict = evaluate_quota_notice(
        provider="deepseek",
        model="deepseek-chat",
        reports=[
            _report("deepseek", _balance_row(0.0), fetched_at=NOW_MS - USAGE_REPORT_TTL_MS - 1)
        ],
        now_ms=NOW_MS,
    )
    assert verdict.state == "unknown"


def test_usage_unavailable_is_unknown() -> None:
    verdict = evaluate_quota_notice(
        provider="deepseek",
        model="deepseek-chat",
        reports=[_report("deepseek", usage_unavailable=True)],
        now_ms=NOW_MS,
    )
    assert verdict.state == "unknown"


def test_credential_invalid_is_unknown_not_quota() -> None:
    """A dead grant is a sign-in problem; its notice is the sibling design's."""
    verdict = evaluate_quota_notice(
        provider="anthropic",
        model="claude-opus-4",
        reports=[
            _report(
                "anthropic",
                _window_row(used=100, total=100, resets_at_ms=NOW_MS + 3_600_000),
                credential_invalid=True,
            )
        ],
        now_ms=NOW_MS,
    )
    assert verdict.state == "unknown"


def test_missing_account_is_unknown() -> None:
    """One fresh report per expected account, or say nothing."""
    verdict = evaluate_quota_notice(
        provider="anthropic",
        model="claude-opus-4",
        reports=[
            _report(
                "anthropic",
                _window_row(used=100, total=100, resets_at_ms=NOW_MS + 3_600_000),
                identity="a@example.com",
            )
        ],
        expected_identities=["a@example.com", "b@example.com"],
        now_ms=NOW_MS,
    )
    assert verdict.state == "unknown"


def test_every_account_depleted_is_a_notice() -> None:
    verdict = evaluate_quota_notice(
        provider="anthropic",
        model="claude-opus-4",
        reports=[
            _report(
                "anthropic",
                _window_row(used=100, total=100, resets_at_ms=NOW_MS + 3_600_000),
                identity="a@example.com",
            ),
            _report(
                "anthropic",
                _window_row(used=100, total=100, resets_at_ms=NOW_MS + 7_200_000),
                identity="b@example.com",
            ),
        ],
        expected_identities=["a@example.com", "b@example.com"],
        now_ms=NOW_MS,
    )
    assert verdict.state == "limit_reached"
    assert verdict.kind == "subscription"
    # The EARLIEST reset is when the account can act again.
    assert verdict.resets_at_ms == NOW_MS + 3_600_000


def test_one_healthy_account_suppresses_the_notice() -> None:
    verdict = evaluate_quota_notice(
        provider="anthropic",
        model="claude-opus-4",
        reports=[
            _report(
                "anthropic",
                _window_row(used=100, total=100, resets_at_ms=NOW_MS + 3_600_000),
                identity="a@example.com",
            ),
            _report(
                "anthropic",
                _window_row(used=10, total=100, resets_at_ms=NOW_MS + 3_600_000),
                identity="b@example.com",
            ),
        ],
        expected_identities=["a@example.com", "b@example.com"],
        now_ms=NOW_MS,
    )
    assert verdict.state == "ok"


def test_kimi_mixed_credentials_suppress_the_notice() -> None:
    """OAuth login + API key: the fetch ran one route, the other is unproven.

    Kimi's two routes are genuinely different products (coding-plan windows vs
    Moonshot balance), so "we cannot prove both are dead" applies: silence.
    """
    reports = [
        _report(
            "kimi",
            _window_row(used=100, total=100, resets_at_ms=NOW_MS + 3_600_000, label="Plan"),
            identity="k@example.com",
        )
    ]
    suppressed = evaluate_quota_notice(
        provider="kimi",
        model="k3",
        reports=reports,
        expected_identities=["k@example.com"],
        api_key_present=True,
        now_ms=NOW_MS,
    )
    assert suppressed.state == "unknown"

    proven = evaluate_quota_notice(
        provider="kimi",
        model="k3",
        reports=reports,
        expected_identities=["k@example.com"],
        api_key_present=False,
        now_ms=NOW_MS,
    )
    assert proven.state == "limit_reached"
    assert proven.kind == "subscription"
    # The coding-plan console, not the API-key top-up page: the link is
    # resolved FROM the derived kind (round-1 M2 — before the fix the variant
    # table was dead code on this path and this click went to platform.kimi.com).
    assert proven.actions[0].url == "https://www.kimi.com/code/console"


def test_zai_mixed_credentials_also_suppress_conservatively() -> None:
    """A second credential exists and its route was skipped: stay silent.

    Z.AI runs the SAME fetcher on both routes, so this is the conservative
    end of the mixed-credentials rule rather than a necessity — a false
    "empty" costs more than a missed notice, and the rule's rationale (an
    unfetched credential's numbers are unproven) holds either way.
    """
    verdict = evaluate_quota_notice(
        provider="zai",
        model="glm-4",
        reports=[
            _report(
                "zai",
                _window_row(
                    used=100, total=100, resets_at_ms=NOW_MS + 3_600_000, label="Token quota"
                ),
                identity="z@example.com",
            )
        ],
        expected_identities=["z@example.com"],
        api_key_present=True,
        now_ms=NOW_MS,
    )
    assert verdict.state == "unknown"


def test_zai_oauth_only_still_warns() -> None:
    """Without the second credential, the exhausted plan is a notice."""
    verdict = evaluate_quota_notice(
        provider="zai",
        model="glm-4",
        reports=[
            _report(
                "zai",
                _window_row(
                    used=100, total=100, resets_at_ms=NOW_MS + 3_600_000, label="Token quota"
                ),
                identity="z@example.com",
            )
        ],
        expected_identities=["z@example.com"],
        api_key_present=False,
        now_ms=NOW_MS,
    )
    assert verdict.state == "limit_reached"


def test_kimi_api_key_balance_stays_balance_shaped() -> None:
    verdict = evaluate_quota_notice(
        provider="kimi",
        model="k3",
        reports=[_report("kimi", _balance_row(0.0))],
        now_ms=NOW_MS,
    )
    assert verdict.state == "depleted"
    assert verdict.kind == "balance"
    # The region-resolved top-up page, not the coding-plan console.
    assert verdict.actions[0].url == "https://platform.kimi.com/console/pay"


def test_openrouter_key_cap_is_balance_not_subscription() -> None:
    """A measurable spend cap that never resets is a budget, not a plan.

    Pins the discriminator: a fraction is not enough to call evidence
    "subscription" — the reset instant is. OpenRouter's key limit reads as a
    balance and gets the top-up copy.
    """
    cap = UsageLimit(
        id="openrouter:credits",
        label="Credits",
        amount=UsageAmount(used=10.0, limit=10.0),
        window="lifetime",
    )
    verdict = evaluate_quota_notice(
        provider="openrouter",
        model="some-model",
        reports=[_report("openrouter", cap)],
        now_ms=NOW_MS,
    )
    assert verdict.state == "depleted"
    assert verdict.kind == "balance"


def test_free_model_suppresses_entirely() -> None:
    verdict = evaluate_quota_notice(
        provider="deepseek",
        model="deepseek-free",
        reports=[_report("deepseek", _balance_row(0.0))],
        entry=_entry("deepseek", "deepseek-free", input_price=0.0, output_price=0.0),
        now_ms=NOW_MS,
    )
    assert verdict.state == "not_applicable"
    assert verdict.model_free is True
    assert verdict.title == "" and verdict.body == "" and verdict.actions == ()


def test_unknown_price_is_not_free() -> None:
    """``-1`` is "nobody quoted it", not "it is free" — the notice still stands."""
    verdict = evaluate_quota_notice(
        provider="deepseek",
        model="deepseek-chat",
        reports=[_report("deepseek", _balance_row(0.0))],
        entry=_entry("deepseek", "deepseek-chat", input_price=-1.0, output_price=-1.0),
        now_ms=NOW_MS,
    )
    assert verdict.state == "depleted"
    assert verdict.model_free is False


def test_routed_meta_route_is_never_free() -> None:
    """A zero-priced meta-route (``radient/auto``) dispatches elsewhere: not free."""
    verdict = evaluate_quota_notice(
        provider="radient",
        model="radient/auto",
        reports=[_report("radient", _balance_row(0.0))],
        entry=_entry("radient", "radient/auto", input_price=0.0, output_price=0.0, routed=True),
        now_ms=NOW_MS,
    )
    assert verdict.model_free is False
    assert verdict.state == "depleted"


def test_provider_without_a_fetcher_is_not_applicable() -> None:
    """API-key google/anthropic/openai/mistral/xai can never warn pre-emptively."""
    verdict = evaluate_quota_notice(
        provider="google",
        model="gemini-3-pro",
        reports=[_report("google", _balance_row(0.0))],
        now_ms=NOW_MS,
    )
    assert verdict.state == "not_applicable"


def test_no_reports_is_unknown() -> None:
    verdict = evaluate_quota_notice(
        provider="deepseek", model="deepseek-chat", reports=[], now_ms=NOW_MS
    )
    assert verdict.state == "unknown"


def test_radient_body_is_the_shared_builder_sentence() -> None:
    """The Radient body is ``radient_recovery.recovery_line``'s output verbatim.

    Compared against the builder with the same facts rather than a string
    literal, so a rewrite of the shared module cannot strand a second copy of
    the sentence here.
    """
    facts = RecoveryFacts(
        signed_in=True,
        verification=None,
    )
    line = recovery_line(facts)
    verdict = evaluate_quota_notice(
        provider="radient",
        model="radient/some-model",
        reports=[_report("radient", _balance_row(0.0))],
        radient_facts=facts,
        now_ms=NOW_MS,
    )
    assert verdict.state == "depleted"
    assert verdict.kind == "radient"
    assert verdict.body == line
    assert verdict.actions[0].url == BILLING_LINKS["radient"].url
    assert verdict.actions[-1].id == "refresh"


def test_radient_without_a_fetched_line_uses_the_builders_own_neutral_rendering() -> None:
    verdict = evaluate_quota_notice(
        provider="radient",
        model="radient/some-model",
        reports=[_report("radient", _balance_row(0.0))],
        now_ms=NOW_MS,
    )
    assert verdict.body == recovery_line(RecoveryFacts(signed_in=None))


def test_limit_reached_copy_names_the_window_and_reset() -> None:
    verdict = evaluate_quota_notice(
        provider="anthropic",
        model="claude-opus-4",
        reports=[
            _report("anthropic", _window_row(used=100, total=100, resets_at_ms=NOW_MS + 7_200_000))
        ],
        now_ms=NOW_MS,
    )
    assert verdict.state == "limit_reached"
    assert verdict.title == "Anthropic limit reached"
    assert "5 hour" in verdict.body and "in 2h" in verdict.body
    assert verdict.actions[0].id == "open_url"
    assert verdict.actions[0].url == BILLING_LINKS["anthropic"].url


def test_age_ms_is_the_newest_report() -> None:
    verdict = evaluate_quota_notice(
        provider="anthropic",
        model="claude-opus-4",
        reports=[
            _report(
                "anthropic",
                _window_row(used=100, total=100, resets_at_ms=NOW_MS + 3_600_000),
                identity="a@example.com",
                fetched_at=NOW_MS - 3_000,
            ),
            _report(
                "anthropic",
                _window_row(used=100, total=100, resets_at_ms=NOW_MS + 3_600_000),
                identity="b@example.com",
                fetched_at=NOW_MS - 1_000,
            ),
        ],
        expected_identities=["a@example.com", "b@example.com"],
        now_ms=NOW_MS,
    )
    assert verdict.age_ms == 1_000


# --- Radient account states (PR2) -------------------------------------------


def _radient_verdict(
    facts: RecoveryFacts | None, *, balance: float = 0.0, resend: bool = True, **extra
):
    return evaluate_quota_notice(
        provider="radient",
        model="radient/some-model",
        reports=[_report("radient", _balance_row(balance))],
        radient_facts=facts,
        resend_available=resend,
        now_ms=NOW_MS,
        **extra,
    )


def _facts(**verification) -> RecoveryFacts:
    return RecoveryFacts(signed_in=True, verification=VerificationFacts(**verification))


@pytest.mark.parametrize(
    "verification,expected",
    [
        (None, "unreadable"),
        (VerificationFacts(signup_grant="claimed"), "verified"),
        (VerificationFacts(email_verified=True), "verified"),
        (VerificationFacts(signup_grant="pending"), "unverified"),
        (VerificationFacts(signup_grant="expired"), "unverified"),
        (VerificationFacts(email_verified=False), "unverified"),
        (VerificationFacts(signup_grant="none"), "unreadable"),
        (VerificationFacts(), "unreadable"),
        # A claimed grant outranks a stale `email_verified: false`: the credits
        # were claimed, so the account cannot be waiting on verification.
        (VerificationFacts(signup_grant="claimed", email_verified=False), "verified"),
    ],
)
def test_account_state_ladder(verification, expected) -> None:
    assert account_state(RecoveryFacts(signed_in=True, verification=verification)) == expected


@pytest.mark.parametrize("grant", ["pending", "expired"])
def test_unverified_empty_account_says_verify_first_and_offers_resend(grant: str) -> None:
    facts = _facts(signup_grant=grant, grant_amount=5.0)
    verdict = _radient_verdict(facts)
    assert verdict.state == "unverified"
    assert verdict.kind == "radient"
    assert verdict.body == recovery_line(facts)
    assert [a.id for a in verdict.actions] == ["open_url", "resend_verification", "refresh"]
    assert verdict.actions[0].url == CLAIM_URL


def test_unverified_without_a_resend_capable_credential_degrades_to_the_link() -> None:
    """An API-key-only account: the upstream route is JWT-only, so no button."""
    verdict = _radient_verdict(_facts(signup_grant="pending"), resend=False)
    assert verdict.state == "unverified"
    assert [a.id for a in verdict.actions] == ["open_url", "refresh"]


def test_unverified_with_no_grant_ticket_never_offers_resend() -> None:
    """``email_verified: false`` with no ticket would only 409; open the page instead."""
    verdict = _radient_verdict(_facts(email_verified=False))
    assert verdict.state == "unverified"
    assert "resend_verification" not in [a.id for a in verdict.actions]


def test_unverified_uses_the_payloads_own_claim_url() -> None:
    url = "https://console.radienthq.com/dashboard/verification?x=1"
    verdict = _radient_verdict(_facts(signup_grant="pending", claim_url=url))
    assert verdict.actions[0].url == url


def test_verified_empty_account_is_a_topup_with_the_payloads_topup_url() -> None:
    url = "https://console.radienthq.com/dashboard/billing?first=1"
    facts = _facts(
        signup_grant="claimed",
        first_topup=FirstTopupFacts(
            bonus_amount=5.0, minimum_purchase=5.0, bonus_received=False, topup_url=url
        ),
    )
    verdict = _radient_verdict(facts)
    assert verdict.state == "depleted"
    assert verdict.body == recovery_line(facts)
    assert verdict.actions[0].url == url
    assert [a.id for a in verdict.actions] == ["open_url", "refresh"]


@pytest.mark.parametrize("facts", [None, RecoveryFacts(signed_in=False), _facts()])
def test_unreadable_radient_account_gets_the_neutral_topup_notice(facts) -> None:
    """No claim about verification when the probe proved nothing."""
    verdict = _radient_verdict(facts)
    assert verdict.state == "depleted"
    assert verdict.body == recovery_line(RecoveryFacts(signed_in=None))
    assert verdict.actions[0].url == BILLING_LINKS["radient"].url


def test_positive_radient_balance_is_silent_even_for_an_unverified_account() -> None:
    """Unverified is a refinement of an EMPTY account, never a warning of its own."""
    verdict = _radient_verdict(_facts(signup_grant="pending"), balance=3.0)
    # A bare positive balance reads "unknown" (see the deepseek case above);
    # what matters is that nothing is shown.
    assert verdict.state in {"ok", "unknown"}
    assert verdict.actions == ()
    assert verdict.title == "" and verdict.body == ""


def test_stale_radient_report_is_unknown_even_when_unverified() -> None:
    verdict = evaluate_quota_notice(
        provider="radient",
        model="radient/some-model",
        reports=[
            _report("radient", _balance_row(0.0), fetched_at=NOW_MS - USAGE_REPORT_TTL_MS - 1)
        ],
        radient_facts=_facts(signup_grant="pending"),
        resend_available=True,
        now_ms=NOW_MS,
    )
    assert verdict.state == "unknown"
    assert verdict.actions == ()


def test_free_model_on_unverified_radient_is_not_applicable() -> None:
    verdict = _radient_verdict(
        _facts(signup_grant="pending"),
        entry=_entry("radient", "radient/free", input_price=0.0, output_price=0.0),
    )
    assert verdict.state == "not_applicable"
    assert verdict.model_free is True
