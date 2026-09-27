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

RENDERED_QUOTA = "rate limit or quota exceeded (HTTP 402): insufficient credits"


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


def test_pending_grant_names_email_amount_and_claim_url() -> None:
    """The core promise: an unclaimed grant points at the email, not a top-up."""
    line = rr.recovery_line(_facts())
    assert "check your email" in line
    assert "Radient verification link" in line
    assert "$5.00" in line
    assert rr.CLAIM_URL in line


def test_pending_without_amount_omits_the_money() -> None:
    """``grant_amount`` is optional in the contract; its absence is not a zero."""
    line = rr.recovery_line(
        rr.RecoveryFacts(signed_in=True, verification=_verification(signup_grant="pending"))
    )
    assert "check your email" in line
    assert "$" not in line


def test_pending_prefers_the_payloads_own_claim_url() -> None:
    """The frozen object carries ``claim_url``; use it when the backend states one."""
    line = rr.recovery_line(
        _facts(verification=_verification(signup_grant="pending", claim_url="https://elsewhere/x"))
    )
    assert "https://elsewhere/x" in line


def test_expired_points_at_the_console_and_never_claims_an_email_is_waiting() -> None:
    """The instruction this implements verbatim: do NOT claim a link is waiting."""
    line = rr.recovery_line(_facts(verification=_verification(signup_grant="expired")))
    assert rr.CLAIM_URL in line
    assert "expired" in line
    assert "email" not in line.lower()


def test_none_points_at_the_console_claim_page() -> None:
    line = rr.recovery_line(_facts(verification=_verification(signup_grant="none")))
    assert rr.CLAIM_URL in line
    assert "no signup grant" in line


def test_claimed_and_unknown_states_fall_back_to_the_generic_console_line() -> None:
    """A claimed grant with a quota failure is a balance problem, not a claim one."""
    assert rr.recovery_line(_facts(verification=_verification(signup_grant="claimed"))) == (
        rr._GENERIC_LINE
    )
    assert rr.recovery_line(_facts(verification=_verification(signup_grant="weird"))) == (
        rr._GENERIC_LINE
    )


def test_missing_verification_object_degrades_to_the_generic_line() -> None:
    """Tolerating an OLDER BACKEND: the object's absence is a fallback, not an error."""
    assert rr.recovery_line(rr.RecoveryFacts(signed_in=True, verification=None)) == (
        rr._GENERIC_LINE
    )
    assert rr.parse_verification(None) is None
    assert rr.parse_verification("not-an-object") is None


def test_unreadable_store_degrades_to_the_generic_line() -> None:
    """``signed_in=None`` must not claim either sign-in state."""
    assert rr.recovery_line(rr.RecoveryFacts(signed_in=None)) == rr._GENERIC_LINE


def test_no_stored_credential_names_the_sign_in_fix() -> None:
    line = rr.recovery_line(rr.RecoveryFacts(signed_in=False))
    assert "No Radient account is signed in" in line
    assert "/login radient" in line
    assert "Settings" in line


def test_parse_drops_wrong_shapes_rather_than_rejecting_the_object() -> None:
    facts = rr.parse_verification(
        {"email_verified": "yes", "signup_grant": 7, "grant_amount": True, "claim_url": 3}
    )
    assert facts is not None
    assert facts.email_verified is None
    assert facts.signup_grant is None
    assert facts.grant_amount is None
    assert facts.claim_url is None


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
async def test_probe_failure_is_swallowed_and_degrades_to_the_generic_line(
    tmp_path, monkeypatch
) -> None:
    store = _oauth_store(tmp_path)

    async def probe(token: str):
        raise RuntimeError("boom")

    monkeypatch.setattr(rr, "_probe_verification_async", probe)

    facts = await rr.get_recovery_facts(store=store)

    assert facts.signed_in is True and facts.verification is None
    assert rr.recovery_line(facts) == rr._GENERIC_LINE


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
    assert rr.recovery_line(facts) == rr._GENERIC_LINE


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
    assert "check your email" in out

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
    assert rr.append_recovery_line_once("", rr._GENERIC_LINE) == rr._GENERIC_LINE


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
    """The network-free read is still an answer: the sign-in remedy, no probe."""
    calls: list[str] = []

    def probe_sync(token: str):
        calls.append(token)
        return None

    monkeypatch.setattr(rr, "_probe_verification_sync", probe_sync)
    empty = AuthStore(tmp_path / "empty.db")

    line = rr.usage_limit_recovery_line_cached(store=empty)

    assert line is not None and "No Radient account is signed in" in line
    assert calls == []
    assert rr.usage_limit_recovery_pending(RENDERED_QUOTA, "radient", store=empty) is False
    out = rr.append_usage_limit_recovery_cached(RENDERED_QUOTA, "radient", store=empty)
    assert "No Radient account is signed in" in out


def test_the_cached_line_degrades_to_the_generic_line_on_a_store_failure(monkeypatch) -> None:
    """An unreadable store is still a definite answer, so pending stays false."""
    import local_operator.providers.auth_store as auth_store

    def broken():
        raise RuntimeError("store unavailable")

    monkeypatch.setattr(auth_store, "shared_auth_store", broken)

    assert rr.usage_limit_recovery_line_cached() == rr._GENERIC_LINE
    assert rr.usage_limit_recovery_pending(RENDERED_QUOTA, "radient") is False


@pytest.mark.asyncio
async def test_the_cached_line_serves_a_warm_cache(tmp_path, monkeypatch) -> None:
    store = _oauth_store(tmp_path)

    async def probe(token: str):
        return _verification(grant_amount=5)

    monkeypatch.setattr(rr, "_probe_verification_async", probe)
    await rr.get_recovery_facts(store=store)

    line = rr.usage_limit_recovery_line_cached(store=store)
    assert line is not None and "check your email" in line
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
# Family dedupe — keyed on OUR sentence, never on a payload-supplied URL
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

    assert pending_line.startswith("Radient: ") and custom in pending_line
    assert expired_line.startswith("Radient: ")
    assert rr.append_recovery_line_once(text, expired_line) == text


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
