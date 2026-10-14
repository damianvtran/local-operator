"""The Radient usage-limit recovery: branches, cache, gates, and the contract.

WHAT THIS PINS. A Radient quota failure is the moment the user needs the one
fact their account holds and the provider's error does not: whether the free
signup grant is waiting in an email. These tests drive the real sentence
builder over every frozen-contract state (pending / expired / none / claimed),
the tolerated degradations (missing object, unreachable probe, nothing signed
in, a store that will not read), the trigger gates (only quota-labelled text,
only the Radient provider), and the short-TTL cache the display path leans on
(one probe per window, and none at all when there is nothing to probe).

Hermeticity: the autouse fixture pins the secret reader to ``None`` and every
store-backed case injects its own ``AuthStore`` under ``tmp_path``, so no test
may read the operator's real credential store; the probe seam is monkeypatched
in every test that reaches a fetch, so none may touch the network either.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import httpx
import pytest

from local_operator.providers import radient_recovery as rr
from local_operator.providers.auth_store import AuthStore
from local_operator.providers.failover import ProviderError

RENDERED_QUOTA = "out of credits (HTTP 402): insufficient credits"
#: What a runtime older than the 402 label split rendered for the same failure.
LEGACY_RENDERED_402 = "rate limit or quota exceeded (HTTP 402): insufficient credits"


@pytest.fixture(autouse=True)
def _hermetic(monkeypatch: pytest.MonkeyPatch):
    """No test may read the operator's store or environment for a key."""
    import local_operator.providers.registry as registry

    monkeypatch.delenv("RADIENT_API_KEY", raising=False)
    monkeypatch.setattr(registry, "provider_secret_value", lambda *args, **kwargs: None)
    rr.reset_recovery_cache()
    yield
    rr.reset_recovery_cache()


def _oauth_store(tmp_path: Path, token: str = "tok-1") -> AuthStore:
    """A real store holding one Radient OAuth row, like a completed login."""
    store = AuthStore(tmp_path / "auth.db")
    store.upsert_credential("radient", {"type": "oauth", "access": token, "refresh": "r"})
    return store


def _verification(**overrides: Any) -> rr.VerificationFacts | None:
    payload: dict[str, Any] = {"email_verified": False, "signup_grant": "pending"}
    payload.update(overrides)
    return rr.parse_verification(payload)


def _facts(**overrides: Any) -> rr.RecoveryFacts:
    fields: dict[str, Any] = {
        "signed_in": True,
        "verification": _verification(signup_grant="pending", grant_amount=5),
    }
    fields.update(overrides)
    return rr.RecoveryFacts(**fields)


# ---------------------------------------------------------------------------
# The sentence builder — every frozen-contract branch
# ---------------------------------------------------------------------------


def _verified_facts(first_topup: dict[str, Any] | None, **grant: Any) -> rr.RecoveryFacts:
    payload: dict[str, Any] = {"email_verified": True, "signup_grant": "claimed"}
    payload.update(grant)
    if first_topup is not None:
        payload["first_topup"] = first_topup
    return rr.RecoveryFacts(signed_in=True, verification=rr.parse_verification(payload))


#: The frozen contract's first-top-up object for an account whose bonus is still on offer.
_BONUS_ON_OFFER: dict[str, Any] = {
    "bonus_amount": 10,
    "minimum_purchase": 5,
    "bonus_received": False,
    "topup_url": "https://console.radienthq.com/dashboard/billing",
}


def test_pending_grant_names_inbox_amount_and_claim_url() -> None:
    """The core promise: an unclaimed grant points at the inbox, not a top-up."""
    line = rr.recovery_line(_facts())
    assert line.startswith(
        "You haven't verified your email yet. Verify to claim $5 in free credits "
        "and start using Local Operator for free."
    )
    assert "Check your inbox" in line
    assert rr.CLAIM_URL in line
    assert "Top up" not in line and rr.TOPUP_URL not in line


def test_pending_without_amount_omits_the_money() -> None:
    """``grant_amount`` is optional in the contract; its absence is not a zero."""
    line = rr.recovery_line(
        rr.RecoveryFacts(signed_in=True, verification=_verification(signup_grant="pending"))
    )
    assert "Verify to claim your free credits" in line
    assert "$" not in line


@pytest.mark.parametrize(
    ("amount", "expected"),
    [(5, "$5"), (5.0, "$5"), (2.5, "$2.50"), (10, "$10")],
)
def test_the_grant_amount_is_formatted_without_trailing_zero_cents(
    amount: float, expected: str
) -> None:
    """Matches the verification email's subject, so the figure reads the same."""
    line = rr.recovery_line(_facts(verification=_verification(grant_amount=amount)))
    assert f"claim {expected} in free credits" in line


def test_pending_prefers_the_payloads_own_claim_url() -> None:
    """The frozen object carries ``claim_url``; use it when the backend states one."""
    line = rr.recovery_line(
        _facts(verification=_verification(signup_grant="pending", claim_url="https://elsewhere/x"))
    )
    assert "https://elsewhere/x" in line
    assert rr.CLAIM_URL not in line


def test_expired_asks_for_a_new_link_and_never_claims_an_email_is_waiting() -> None:
    """The mail itself is dead, so no instruction to check the inbox."""
    line = rr.recovery_line(_facts(verification=_verification(signup_grant="expired")))
    assert rr.CLAIM_URL in line
    assert "expired" in line and "Request a new one" in line
    assert "inbox" not in line.lower()
    assert line.startswith("You haven't verified your email yet.")


def test_none_with_an_unverified_email_points_at_the_verification_page() -> None:
    line = rr.recovery_line(_facts(verification=_verification(signup_grant="none")))
    assert line.startswith("You haven't verified your email yet.")
    assert f"Open {rr.CLAIM_URL}" in line
    assert "inbox" not in line.lower()


def test_none_without_an_explicit_unverified_flag_is_not_claimed_unverified() -> None:
    """``none`` alone (no ticket) cannot say whether the email is verified: neutral."""
    facts = rr.RecoveryFacts(
        signed_in=True, verification=rr.parse_verification({"signup_grant": "none"})
    )
    assert rr.recovery_line(facts) == rr._neutral_text()


def test_claimed_with_the_bonus_on_offer_gets_the_topup_link_and_the_bonus_line() -> None:
    line = rr.recovery_line(_verified_facts(_BONUS_ON_OFFER))
    assert line.splitlines() == [
        "You're out of credits. Top up in the Radient console: "
        "https://console.radienthq.com/dashboard/billing",
        "Get an extra $10 free on your first top-up of $5 or more.",
    ]


def test_claimed_with_the_bonus_already_received_omits_the_bonus_line() -> None:
    line = rr.recovery_line(_verified_facts({**_BONUS_ON_OFFER, "bonus_received": True}))
    assert line == (
        "You're out of credits. Top up in the Radient console: "
        "https://console.radienthq.com/dashboard/billing"
    )


def test_claimed_without_first_topup_still_shows_the_topup_link() -> None:
    """An OLDER backend: the link stays, the bonus line is never invented."""
    line = rr.recovery_line(_verified_facts(None))
    assert line == f"You're out of credits. Top up in the Radient console: {rr.TOPUP_URL}"


def test_a_verified_email_without_a_claimed_grant_is_still_verified() -> None:
    """``email_verified`` alone settles it: the user has nothing left to verify."""
    line = rr.recovery_line(_verified_facts(None, signup_grant="none"))
    assert line.startswith("You're out of credits. Top up")
    assert "verified your email" not in line


def test_the_topup_url_comes_from_first_topup_when_named() -> None:
    line = rr.recovery_line(
        _verified_facts({**_BONUS_ON_OFFER, "topup_url": "https://console.example/billing"})
    )
    assert "https://console.example/billing" in line
    assert rr.TOPUP_URL not in line


@pytest.mark.parametrize(
    "first_topup",
    [
        {"bonus_received": False},  # no figures: an offer that cannot be quoted
        {"bonus_amount": 10, "bonus_received": False},  # no minimum
        {"minimum_purchase": 5, "bonus_received": False},  # no bonus
        {"bonus_amount": 10, "minimum_purchase": 5},  # bonus_received not stated
        {"bonus_amount": 10, "minimum_purchase": 5, "bonus_received": "no"},  # wrong shape
        {"bonus_amount": True, "minimum_purchase": 5, "bonus_received": False},  # bool figure
        {"bonus_amount": 0, "minimum_purchase": 5, "bonus_received": False},  # no bonus at all
        {"bonus_amount": 10, "minimum_purchase": -1, "bonus_received": False},
        "not-an-object",
    ],
)
def test_the_bonus_line_needs_every_figure_and_a_definite_false(first_topup: Any) -> None:
    """The bonus is an OFFER: unreadable or unstated means no line, never a guess."""
    line = rr.recovery_line(_verified_facts(first_topup))  # type: ignore[arg-type]
    assert "extra" not in line
    assert line.startswith("You're out of credits. Top up")


@pytest.mark.parametrize(
    "unsafe",
    [
        "http://insecure.example/x",
        "https://a b/x",
        "https://x/\x1b[31m",  # C0 ESC
        # Agent review round 1, R1-3: a C1 CSI reaches a terminal as an escape
        # introducer, a Cf bidi override reorders what the reader sees, and a
        # userinfo lookalike reads as the console host while addressing evil.
        "https://x.example/\x9b31m",
        "https://x.example/\u202e",
        "https://console.radienthq.com@evil.example/p",
        "javascript:alert(1)",
        7,
    ],
)
def test_a_non_https_control_or_lookalike_url_is_never_echoed(unsafe: Any) -> None:
    """URLs are printed into a terminal: fall back to the known console page."""
    verified = rr.recovery_line(_verified_facts({**_BONUS_ON_OFFER, "topup_url": unsafe}))
    assert rr.TOPUP_URL in verified and str(unsafe) not in verified
    unverified = rr.recovery_line(_facts(verification=_verification(claim_url=unsafe)))
    assert rr.CLAIM_URL in unverified and str(unsafe) not in unverified


def test_an_unknown_grant_state_is_unreadable_not_guessed() -> None:
    facts = rr.RecoveryFacts(signed_in=True, verification=rr.parse_verification({"x": 1}))
    assert rr.recovery_line(facts) == rr._neutral_text()


def test_missing_verification_object_degrades_to_the_neutral_text() -> None:
    """Tolerating an OLDER BACKEND: the object's absence is a fallback, not an error."""
    assert rr.recovery_line(rr.RecoveryFacts(signed_in=True, verification=None)) == (
        rr._neutral_text()
    )
    assert rr.parse_verification(None) is None
    assert rr.parse_verification("not-an-object") is None


def test_unreadable_state_words_both_links_conditionally() -> None:
    """Offline, signed out or /me failed: claim neither state."""
    for facts in (rr.RecoveryFacts(signed_in=None), rr.RecoveryFacts(signed_in=False)):
        text = rr.recovery_line(facts)
        assert text == rr._neutral_text()
        assert text.startswith("You're out of credits.")
        assert "If you haven't verified your email yet" in text and rr.CLAIM_URL in text
        assert "Otherwise, top up" in text and rr.TOPUP_URL in text
        assert "extra" not in text, "no bonus can be offered for an account never read"


def test_every_branch_text_is_inside_the_dedupe_family() -> None:
    """A branch text MUST carry a family marker or a retried render would stack it."""
    texts = [
        rr.recovery_line(_facts()),
        rr.recovery_line(_facts(verification=_verification(signup_grant="expired"))),
        rr.recovery_line(_facts(verification=_verification(signup_grant="none"))),
        rr.recovery_line(_verified_facts(_BONUS_ON_OFFER)),
        rr.recovery_line(_verified_facts(None)),
        rr._neutral_text(),
    ]
    for text in texts:
        assert rr._carries_family_text(text), text
        assert rr.append_recovery_line_once(f"err\n{text}", texts[0]) == f"err\n{text}"


def test_parse_drops_wrong_shapes_rather_than_rejecting_the_object() -> None:
    facts = rr.parse_verification(
        {"email_verified": "yes", "signup_grant": 7, "grant_amount": True, "claim_url": 3}
    )
    assert facts is not None
    assert facts.email_verified is None
    assert facts.signup_grant is None
    assert facts.grant_amount is None
    assert facts.claim_url is None
    assert facts.first_topup is None


def test_first_topup_is_parsed_from_the_frozen_shape() -> None:
    facts = rr.parse_verification({"first_topup": _BONUS_ON_OFFER})
    assert facts is not None and facts.first_topup == rr.FirstTopupFacts(
        bonus_amount=10.0,
        minimum_purchase=5.0,
        bonus_received=False,
        topup_url="https://console.radienthq.com/dashboard/billing",
    )


# ---------------------------------------------------------------------------
# Trigger gates — only quota-labelled text, only the Radient provider
# ---------------------------------------------------------------------------


def test_applies_only_to_quota_labelled_errors_for_radient() -> None:
    assert rr.usage_limit_recovery_applies(RENDERED_QUOTA, "radient")
    # The radient-key login flavour stores under ``radient`` and spends the
    # same account, so it is covered; the normalizer decides, not a string compare.
    assert rr.usage_limit_recovery_applies(RENDERED_QUOTA, "radient-key")
    assert not rr.usage_limit_recovery_applies(RENDERED_QUOTA, "openai")
    assert not rr.usage_limit_recovery_applies(RENDERED_QUOTA, None)
    assert not rr.usage_limit_recovery_applies(RENDERED_QUOTA, "")


def test_applies_only_to_the_402_not_to_a_429_rate_limit() -> None:
    """A 429's remedy is waiting; advice about balances would be wrong for it."""
    rate_limited = "rate limit or quota exceeded (HTTP 429, retry in 42s): slow down"
    assert not rr.usage_limit_recovery_applies(rate_limited, "radient")
    # ...but a rendering written by a runtime older than the 402 label split
    # (a follower attached to an older owner) is still a 402 and still earns it.
    assert rr.usage_limit_recovery_applies(LEGACY_RENDERED_402, "radient")


def test_applies_to_no_other_kind() -> None:
    """The auth hint's own subject must stay with the auth hint (no double-fire)."""
    assert not rr.usage_limit_recovery_applies(
        "authentication failed (HTTP 401): invalid x-api-key", "radient"
    )
    assert not rr.usage_limit_recovery_applies("transient provider error: boom", "radient")
    assert not rr.usage_limit_recovery_applies("", "radient")


# ---------------------------------------------------------------------------
# The probe seam, the cache, and the no-raise contract
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_probe_runs_once_per_ttl_window(tmp_path, monkeypatch) -> None:
    store = _oauth_store(tmp_path)
    calls: list[str] = []

    async def probe(token: str):
        calls.append(token)
        return _verification(grant_amount=5)

    monkeypatch.setattr(rr, "_probe_verification_async", probe)

    first = await rr.get_recovery_facts(store=store)
    second = await rr.get_recovery_facts(store=store)

    assert calls == ["tok-1"]
    assert first == second
    assert first.verification is not None and first.verification.grant_amount == 5


@pytest.mark.asyncio
async def test_cache_expiry_reprobes(tmp_path, monkeypatch) -> None:
    store = _oauth_store(tmp_path)
    calls: list[str] = []

    async def probe(token: str):
        calls.append(token)
        return None

    monkeypatch.setattr(rr, "_probe_verification_async", probe)
    monkeypatch.setattr(rr, "_TTL_S", 0.0)

    await rr.get_recovery_facts(store=store)
    await rr.get_recovery_facts(store=store)

    assert calls == ["tok-1", "tok-1"]


@pytest.mark.asyncio
async def test_force_refresh_bypasses_a_warm_cache_and_replaces_it(tmp_path, monkeypatch) -> None:
    """A user-driven re-check must not be served the cache it is asking about.

    The cache keeps a hint cheap (one probe per window); it is the WRONG
    answer to the click that just changed the account's state, which is why
    ``desktop_quota``'s forced re-read (``refresh`` past the floor) passes
    ``force_refresh`` — R1-X1, the "I verified" flow. The fresh result must
    also REPLACE the cache, or every later plain read would keep the old
    answer for the rest of the window.
    """
    store = _oauth_store(tmp_path)
    calls: list[str] = []
    grants = ["pending", "claimed"]

    async def probe(token: str):
        calls.append(token)
        return _verification(signup_grant=grants[min(len(calls) - 1, 1)])

    monkeypatch.setattr(rr, "_probe_verification_async", probe)

    warm = await rr.get_recovery_facts(store=store)
    assert warm.verification is not None and warm.verification.signup_grant == "pending"

    # A plain read serves the warm cache — that is the cache's job.
    plain = await rr.get_recovery_facts(store=store)
    assert plain == warm and calls == ["tok-1"]

    # The forced read bypasses it and sees the changed state...
    forced = await rr.get_recovery_facts(store=store, force_refresh=True)
    assert forced.verification is not None and forced.verification.signup_grant == "claimed"
    assert calls == ["tok-1", "tok-1"]
    # ...and refreshes the cache for every reader after it.
    assert await rr.get_recovery_facts(store=store) == forced


def test_the_sync_twin_force_refreshes_too(tmp_path, monkeypatch) -> None:
    """The twins share one cache: a divergence here would be a trap."""
    store = _oauth_store(tmp_path)
    calls: list[str] = []

    def probe(token: str):
        calls.append(token)
        return _verification(signup_grant="pending" if len(calls) == 1 else "claimed")

    monkeypatch.setattr(rr, "_probe_verification_sync", probe)

    warm = rr.get_recovery_facts_sync(store=store)
    assert warm.verification is not None and warm.verification.signup_grant == "pending"
    forced = rr.get_recovery_facts_sync(store=store, force_refresh=True)
    assert forced.verification is not None and forced.verification.signup_grant == "claimed"
    assert calls == ["tok-1", "tok-1"]


@pytest.mark.asyncio
async def test_probe_failure_is_swallowed_and_degrades_to_the_generic_line(
    tmp_path, monkeypatch
) -> None:
    store = _oauth_store(tmp_path)

    async def probe(token: str):
        raise RuntimeError("boom")

    monkeypatch.setattr(rr, "_probe_verification_async", probe)

    facts = await rr.get_recovery_facts(store=store)

    assert facts.signed_in is True and facts.verification is None
    assert rr.recovery_line(facts) == rr._neutral_text()


@pytest.mark.asyncio
async def test_no_probe_without_a_credential(tmp_path, monkeypatch) -> None:
    """No stored credential means no fetch at all, and the sign-in remedy."""
    calls: list[str] = []

    async def probe(token: str):
        calls.append(token)
        return None

    monkeypatch.setattr(rr, "_probe_verification_async", probe)

    facts = await rr.get_recovery_facts(store=AuthStore(tmp_path / "empty.db"))

    assert calls == []
    assert facts.signed_in is False


@pytest.mark.asyncio
async def test_store_read_failure_degrades_to_the_generic_line(monkeypatch) -> None:
    import local_operator.providers.auth_store as auth_store

    def broken():
        raise RuntimeError("store unavailable")

    monkeypatch.setattr(auth_store, "shared_auth_store", broken)

    facts = await rr.get_recovery_facts()

    assert facts.signed_in is None
    assert rr.recovery_line(facts) == rr._neutral_text()


# ---------------------------------------------------------------------------
# Append semantics — additive, idempotent, and never raising
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_append_adds_the_line_only_for_the_trigger(tmp_path, monkeypatch) -> None:
    store = _oauth_store(tmp_path)
    calls: list[str] = []

    async def probe(token: str):
        calls.append(token)
        return _verification(grant_amount=5)

    monkeypatch.setattr(rr, "_probe_verification_async", probe)

    out = await rr.append_usage_limit_recovery_async(RENDERED_QUOTA, "radient", store=store)
    assert out.startswith(RENDERED_QUOTA)
    assert "Check your inbox" in out

    # Non-Radient and non-quota: byte-identical, and no probe is spent either.
    assert (
        await rr.append_usage_limit_recovery_async(RENDERED_QUOTA, "openai", store=store)
        == RENDERED_QUOTA
    )
    assert (
        await rr.append_usage_limit_recovery_async(
            "authentication failed (HTTP 401): invalid x-api-key", "radient", store=store
        )
        == "authentication failed (HTTP 401): invalid x-api-key"
    )
    assert calls == ["tok-1"]


@pytest.mark.asyncio
async def test_append_is_idempotent_across_retries(tmp_path, monkeypatch) -> None:
    """A retried render of the same failure must not stack a second remedy."""
    store = _oauth_store(tmp_path)
    calls: list[str] = []

    async def probe(token: str):
        calls.append(token)
        return _verification(grant_amount=5)

    monkeypatch.setattr(rr, "_probe_verification_async", probe)

    once = await rr.append_usage_limit_recovery_async(RENDERED_QUOTA, "radient", store=store)
    twice = await rr.append_usage_limit_recovery_async(once, "radient", store=store)

    assert twice == once
    # The marker check short-circuits BEFORE the fetch, so no extra probe.
    assert calls == ["tok-1"]


@pytest.mark.asyncio
async def test_a_different_branchs_line_does_not_stack(tmp_path, monkeypatch) -> None:
    """The family guard: expired-after-pending must not add a second sentence."""
    store = _oauth_store(tmp_path)

    async def probe(token: str):
        return _verification(signup_grant="expired")

    monkeypatch.setattr(rr, "_probe_verification_async", probe)

    expired_line = await rr.usage_limit_recovery_line(store=store)
    text = f"{RENDERED_QUOTA}\n{expired_line}"

    out = await rr.append_usage_limit_recovery_async(text, "radient", store=store)
    assert out == text


@pytest.mark.asyncio
async def test_sync_and_async_paths_agree(tmp_path, monkeypatch) -> None:
    store = _oauth_store(tmp_path)
    verification = _verification(grant_amount=5)

    async def probe_async(token: str):
        return verification

    def probe_sync(token: str):
        return verification

    monkeypatch.setattr(rr, "_probe_verification_async", probe_async)
    monkeypatch.setattr(rr, "_probe_verification_sync", probe_sync)

    async_out = await rr.append_usage_limit_recovery_async(RENDERED_QUOTA, "radient", store=store)
    rr.reset_recovery_cache()
    sync_out = rr.append_usage_limit_recovery(RENDERED_QUOTA, "radient", store=store)

    assert async_out == sync_out


def test_append_recovery_line_once_handles_empty_inputs() -> None:
    assert rr.append_recovery_line_once("error", "") == "error"
    assert rr.append_recovery_line_once("", rr._neutral_text()) == rr._neutral_text()


# ---------------------------------------------------------------------------
# The cached arm and the pending predicate — the sync, never-blocking read
# ---------------------------------------------------------------------------


def test_the_cached_line_is_none_when_only_a_probe_could_decide(tmp_path, monkeypatch) -> None:
    """Cold cache plus a stored credential: nothing knowable, nothing probed."""
    store = _oauth_store(tmp_path)
    calls: list[str] = []

    async def probe_async(token: str):
        calls.append(f"async:{token}")
        return _verification(grant_amount=5)

    def probe_sync(token: str):
        calls.append(f"sync:{token}")
        return _verification(grant_amount=5)

    monkeypatch.setattr(rr, "_probe_verification_async", probe_async)
    monkeypatch.setattr(rr, "_probe_verification_sync", probe_sync)

    assert rr.usage_limit_recovery_line_cached(store=store) is None
    assert calls == [], "the cached arm must never probe, either twin"
    assert rr.usage_limit_recovery_pending(RENDERED_QUOTA, "radient", store=store) is True
    assert rr.append_usage_limit_recovery_cached(RENDERED_QUOTA, "radient", store=store) == (
        RENDERED_QUOTA
    )


def test_the_cached_line_answers_locally_without_a_credential(tmp_path, monkeypatch) -> None:
    """The network-free read is still an answer: the neutral text, no probe."""
    calls: list[str] = []

    def probe_sync(token: str):
        calls.append(token)
        return None

    monkeypatch.setattr(rr, "_probe_verification_sync", probe_sync)
    empty = AuthStore(tmp_path / "empty.db")

    line = rr.usage_limit_recovery_line_cached(store=empty)

    assert line == rr._neutral_text()
    assert calls == []
    assert rr.usage_limit_recovery_pending(RENDERED_QUOTA, "radient", store=empty) is False
    out = rr.append_usage_limit_recovery_cached(RENDERED_QUOTA, "radient", store=empty)
    assert out == f"{RENDERED_QUOTA}\n{rr._neutral_text()}"


def test_the_cached_line_degrades_to_the_generic_line_on_a_store_failure(monkeypatch) -> None:
    """An unreadable store is still a definite answer, so pending stays false."""
    import local_operator.providers.auth_store as auth_store

    def broken():
        raise RuntimeError("store unavailable")

    monkeypatch.setattr(auth_store, "shared_auth_store", broken)

    assert rr.usage_limit_recovery_line_cached() == rr._neutral_text()
    assert rr.usage_limit_recovery_pending(RENDERED_QUOTA, "radient") is False


@pytest.mark.asyncio
async def test_the_cached_line_serves_a_warm_cache(tmp_path, monkeypatch) -> None:
    store = _oauth_store(tmp_path)

    async def probe(token: str):
        return _verification(grant_amount=5)

    monkeypatch.setattr(rr, "_probe_verification_async", probe)
    await rr.get_recovery_facts(store=store)

    line = rr.usage_limit_recovery_line_cached(store=store)
    assert line is not None and "Check your inbox" in line
    assert rr.usage_limit_recovery_pending(RENDERED_QUOTA, "radient", store=store) is False


def test_pending_is_false_for_every_off_trigger(tmp_path) -> None:
    store = _oauth_store(tmp_path)
    assert rr.usage_limit_recovery_pending(RENDERED_QUOTA, "openai", store=store) is False
    assert rr.usage_limit_recovery_pending(RENDERED_QUOTA, None, store=store) is False
    assert (
        rr.usage_limit_recovery_pending("authentication failed (HTTP 401)", "radient", store=store)
        is False
    )


# ---------------------------------------------------------------------------
# Family dedupe — keyed on phrases unique to OUR copy, never on provider prose
# or a payload-supplied URL
# ---------------------------------------------------------------------------


def test_the_family_dedupe_is_independent_of_the_claim_url() -> None:
    """Review round 1, R2's repro: a non-console ``claim_url`` must not open a
    hole in the family guard — a branch flip between retries (pending, then
    expired) previously appended a second remedy under the first."""
    custom = "https://elsewhere/x"
    pending_line = rr.recovery_line(
        _facts(verification=_verification(signup_grant="pending", grant_amount=5, claim_url=custom))
    )
    expired_line = rr.recovery_line(
        _facts(verification=_verification(signup_grant="expired", claim_url=custom))
    )
    text = f"{RENDERED_QUOTA}\n{pending_line}"

    assert pending_line.startswith("You haven't verified your email yet") and custom in pending_line
    assert expired_line.startswith("You haven't verified your email yet")
    assert rr.append_recovery_line_once(text, expired_line) == text


def test_a_provider_body_quoting_our_opener_does_not_suppress_the_remedy() -> None:
    """R1-2's repro, pinned: the guard must read OUR copy, not the provider's.

    On the pre-fix head, ``str(ProviderError(402, "You're out of credits. Add
    funds to continue."))`` carried the old "You're out of credits" marker, so
    the substring guard returned the text unchanged even though the trigger
    applies.
    """
    raw = str(ProviderError(402, "You're out of credits. Add funds to continue."))

    assert rr.usage_limit_recovery_applies(raw, "radient")
    assert not rr._carries_family_text(raw)
    line = rr.recovery_line(_facts())
    assert rr.append_recovery_line_once(raw, line) == f"{raw}\n{line}"


@pytest.mark.asyncio
async def test_a_colliding_body_still_gains_the_remedy_end_to_end(tmp_path, monkeypatch) -> None:
    store = _oauth_store(tmp_path)

    async def probe(token: str):
        return _verification(grant_amount=5)

    monkeypatch.setattr(rr, "_probe_verification_async", probe)

    raw = str(ProviderError(402, "You're out of credits. Add funds to continue."))
    out = await rr.append_usage_limit_recovery_async(raw, "radient", store=store)

    assert out.startswith(raw)
    assert "Check your inbox" in out


@pytest.mark.asyncio
async def test_a_custom_claim_url_line_also_skips_the_probe(tmp_path, monkeypatch) -> None:
    """The marker check runs BEFORE the fetch, so a re-render costs no request."""
    store = _oauth_store(tmp_path)
    calls: list[str] = []

    async def probe(token: str):
        calls.append(token)
        return _verification(grant_amount=5)

    monkeypatch.setattr(rr, "_probe_verification_async", probe)
    pending_line = rr.recovery_line(
        _facts(
            verification=_verification(
                signup_grant="pending", grant_amount=5, claim_url="https://elsewhere/x"
            )
        )
    )
    text = f"{RENDERED_QUOTA}\n{pending_line}"

    assert await rr.append_usage_limit_recovery_async(text, "radient", store=store) == text
    assert calls == []


# ---------------------------------------------------------------------------
# The wire shape — read through a mock transport, exactly like usage.py tests
# ---------------------------------------------------------------------------


def _me_handler(verification: Any):
    requests: list[httpx.Request] = []

    def handler(request: httpx.Request) -> httpx.Response:
        requests.append(request)
        body: dict[str, Any] = {"result": {"account": {"tenant_id": "t-1"}}}
        if verification is not _MISSING:
            body["result"]["verification"] = verification
        return httpx.Response(200, json=body)

    return handler, requests


_MISSING = object()


@pytest.mark.asyncio
async def test_fetch_reads_the_frozen_object() -> None:
    handler, requests = _me_handler(
        {
            "email_verified": False,
            "signup_grant": "pending",
            "grant_amount": 5.0,
            "claim_url": rr.CLAIM_URL,
        }
    )
    async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as client:
        facts = await rr.fetch_me_verification(client, "tok-1")

    assert facts is not None
    assert facts.signup_grant == "pending" and facts.grant_amount == 5.0
    assert requests[0].url.host == "api.radienthq.com"
    assert requests[0].url.path == "/v1/me"
    assert requests[0].headers["authorization"] == "Bearer tok-1"


@pytest.mark.asyncio
async def test_fetch_missing_object_is_none() -> None:
    handler, _ = _me_handler(_MISSING)
    async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as client:
        assert await rr.fetch_me_verification(client, "tok-1") is None


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "status,body",
    [
        (401, {"error": "unauthorized"}),
        (500, {"error": "boom"}),
    ],
)
async def test_fetch_non_200_is_none(status: int, body: dict[str, Any]) -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(status, json=body)

    async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as client:
        assert await rr.fetch_me_verification(client, "tok-1") is None


@pytest.mark.asyncio
async def test_fetch_unparseable_body_is_none() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, content=b"<html>not json</html>")

    async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as client:
        assert await rr.fetch_me_verification(client, "tok-1") is None


@pytest.mark.asyncio
async def test_fetch_does_not_follow_redirects() -> None:
    requests: list[httpx.Request] = []

    def handler(request: httpx.Request) -> httpx.Response:
        requests.append(request)
        return httpx.Response(302, headers={"location": "https://elsewhere.invalid"})

    async with httpx.AsyncClient(
        transport=httpx.MockTransport(handler), follow_redirects=True
    ) as client:
        assert await rr.fetch_me_verification(client, "tok-1") is None

    assert len(requests) == 1


def test_fetch_sync_twin_reads_the_same_object() -> None:
    handler, _ = _me_handler({"signup_grant": "expired", "grant_amount": 5})
    with httpx.Client(transport=httpx.MockTransport(handler)) as client:
        facts = rr.fetch_me_verification_sync(client, "tok-1")
    assert facts is not None and facts.signup_grant == "expired"


def test_api_key_row_resolves_as_the_probe_bearer(tmp_path) -> None:
    """An API-key operator is probed with the credential they hold, not told to sign in."""
    store = AuthStore(tmp_path / "auth.db")
    store.upsert_credential("radient", {"type": "api_key", "key": "key-9"})
    assert rr._resolve_token(store) == "key-9"
