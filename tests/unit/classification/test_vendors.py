"""The wire codec, the error taxonomy, and per-leg credential resolution.

Every test here drives the real leg classes over ``httpx.MockTransport``: the
request body, the status mapping and the response parsing are the parts a live
call would exercise, and doing it through the transport keeps the suite
hermetic. The one live call lives in ``test_live_openrouter.py``.
"""

from __future__ import annotations

import json
from collections.abc import Callable
from typing import Any

import httpx
import pytest
from pydantic import SecretStr

from local_operator.classification.types import (
    DecisionRequest,
    DecisionSchemaError,
    DecisionVendorError,
    Question,
)
from local_operator.classification.vendors import (
    OPENROUTER_MODEL,
    OpenRouterVendor,
    RadientVendor,
    TypeSafeVendor,
    build_vendor,
    questions_payload,
    request_body,
)
from tests.unit.classification.support import TEST_KEY

pytestmark = pytest.mark.asyncio

KEY = TEST_KEY


def request_of(*questions: Question) -> DecisionRequest:
    return DecisionRequest(state={"request": "deploy core to qa"}, questions=questions)


async def credential_of(vendor: Any, manager: Any) -> str | None:
    """The plain string a leg would send, or ``None`` — keeps assertions short."""
    value = await vendor.credential(manager)
    return value.get_secret_value() if value is not None else None


def choice_question(criteria: dict[str, str] | None = None) -> Question:
    return Question(
        id="recommend_skill",
        kind="choice",
        instructions="Which skill, if any, fits?",
        criteria=criteria or {"minerva-deploy": "Deploys a service", "none": "Nothing fits"},
    )


def client_for(handler: Callable[[httpx.Request], httpx.Response]) -> httpx.AsyncClient:
    return httpx.AsyncClient(transport=httpx.MockTransport(handler), timeout=httpx.Timeout(5.0))


def json_response(payload: object, status: int = 200) -> httpx.Response:
    return httpx.Response(status, json=payload)


# ---------------------------------------------------------------------------
# The request shape (§3)
# ---------------------------------------------------------------------------


async def test_choice_criteria_are_strings_and_score_criteria_are_an_array() -> None:
    """The measured shape, asserted as a shape rather than as "it works".

    A ``choice`` criterion value that is an object is what the contract forbids
    on the alpha route; a ``score`` criterion is an array because the vendor
    answers a score question with a float index into it. Both are asserted on
    the serialized payload, which is what actually goes on the wire.
    """
    payload = questions_payload(
        (
            choice_question(),
            Question(id="effort", kind="score", instructions="how hard?", criteria=("low", "high")),
            Question(
                id="needs_skill",
                kind="noul",
                instructions="needed?",
                criteria={"true": "yes", "false": "no"},
            ),
        )
    )
    assert payload["recommend_skill"] == {
        "type": "choice",
        "instructions": "Which skill, if any, fits?",
        "criteria": {"minerva-deploy": "Deploys a service", "none": "Nothing fits"},
    }
    assert all(isinstance(value, str) for value in payload["recommend_skill"]["criteria"].values())
    assert payload["effort"]["criteria"] == ["low", "high"]
    assert payload["needs_skill"]["criteria"] == {"true": "yes", "false": "no"}
    # And the whole body survives JSON round-tripping, i.e. no tuple leaks.
    assert json.loads(json.dumps(request_body(request_of(choice_question()), "m")))["model"] == "m"


async def test_the_leg_sends_the_measured_body_shape(manager) -> None:
    seen: list[dict[str, Any]] = []

    def handler(request: httpx.Request) -> httpx.Response:
        seen.append(json.loads(request.content))
        return json_response(
            {
                "model": "typesafe/jev-1.13-20260917",
                "answers": {
                    "recommend_skill": {
                        "type": "choice",
                        "choice": "minerva-deploy",
                        "probabilities": {"minerva-deploy": 0.91, "none": 0.09},
                        "confidence": 0.88,
                    }
                },
                "usage": {"input_tokens": 519, "output_tokens": 102, "cost": 0.000021798},
            }
        )

    vendor = OpenRouterVendor(manager, client=client_for(handler))
    response = await vendor.decide(request_of(choice_question()), timeout_s=5.0)

    assert seen == [
        {
            "model": OPENROUTER_MODEL,
            "state": {"request": "deploy core to qa"},
            "questions": {
                "recommend_skill": {
                    "type": "choice",
                    "instructions": "Which skill, if any, fits?",
                    "criteria": {"minerva-deploy": "Deploys a service", "none": "Nothing fits"},
                }
            },
        }
    ]
    assert response.vendor == "openrouter"
    assert response.model == "typesafe/jev-1.13-20260917"
    assert response.input_tokens == 519
    assert response.output_tokens == 102
    assert response.cost_usd == pytest.approx(0.000021798)
    answer = response.answers["recommend_skill"]
    assert (answer.kind, answer.value, answer.confidence) == ("choice", "minerva-deploy", 0.88)
    assert answer.probabilities == {"minerva-deploy": 0.91, "none": 0.09}
    assert response.latency_s >= 0.0


async def test_the_model_override_replaces_the_vendors_own_id(manager) -> None:
    seen: list[dict[str, Any]] = []

    def handler(request: httpx.Request) -> httpx.Response:
        seen.append(json.loads(request.content))
        return json_response({"model": "x", "answers": {}})

    vendor = OpenRouterVendor(manager, model="typesafe/jev-next", client=client_for(handler))
    await vendor.decide(request_of(choice_question()), timeout_s=5.0)
    assert seen[0]["model"] == "typesafe/jev-next"


# ---------------------------------------------------------------------------
# The answer shapes (§3)
# ---------------------------------------------------------------------------


async def test_a_noul_answer_is_a_probability_with_no_distribution(manager) -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        return json_response(
            {
                "model": "jev-1.13-20260917",
                "answers": {"needs_skill": {"type": "noul", "noul": 0.59}},
                "usage": {"input_tokens": 380, "output_tokens": 60, "cost": 0.00001596},
            }
        )

    question = Question(
        id="needs_skill", kind="noul", instructions="needed?", criteria={"true": "y", "false": "n"}
    )
    response = await OpenRouterVendor(manager, client=client_for(handler)).decide(
        request_of(question), timeout_s=5.0
    )
    answer = response.answers["needs_skill"]
    assert (answer.kind, answer.value) == ("noul", 0.59)
    assert answer.probabilities == {}
    assert answer.confidence is None


async def test_a_score_answer_is_the_float_the_vendor_reported(manager) -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        return json_response(
            {
                "answers": {
                    "effort": {
                        "type": "score",
                        "score": 1.11,
                        "legend": {"0": "trivial", "1": "routine", "2": "complex"},
                        "probabilities": {"0": 0.01, "1": 0.87, "2": 0.12},
                        "confidence": 0.81,
                    }
                }
            }
        )

    question = Question(id="effort", kind="score", instructions="how hard?", criteria=("a", "b"))
    response = await TypeSafeVendor(manager, client=client_for(handler)).decide(
        request_of(question), timeout_s=5.0
    )
    answer = response.answers["effort"]
    assert (answer.kind, answer.value, answer.confidence) == ("score", 1.11, 0.81)
    assert answer.probabilities == {"0": 0.01, "1": 0.87, "2": 0.12}


async def test_a_missing_cost_stays_missing(manager) -> None:
    """No price row exists for a decision model, so the contract's fallback is moot."""

    def handler(request: httpx.Request) -> httpx.Response:
        return json_response({"answers": {}, "usage": {"input_tokens": 12}})

    response = await TypeSafeVendor(manager, client=client_for(handler)).decide(
        request_of(choice_question()), timeout_s=5.0
    )
    assert response.cost_usd is None
    assert response.input_tokens == 12
    # The vendor sent a count for input and nothing for output, so output is
    # "not reported" rather than zero — a distinction the cost line depends on.
    assert response.output_tokens is None


async def test_absent_counts_are_none_while_a_reported_zero_is_zero(manager) -> None:
    """``usage`` is optional on this wire, and ``0`` is a legal figure in it.

    Both directions matter. A body with no ``usage`` block at all (the shape the
    live route is free to send, and the one that produced a fabricated
    ``tokens=0/0`` beside a real cost) must come back as ``None``; and a leg that
    really did report zero output tokens — Radient's documented billing shape —
    must come back as ``0``, because collapsing the two would make the operator's
    cost line unable to say which happened.
    """

    def no_usage(request: httpx.Request) -> httpx.Response:
        return json_response({"answers": {}})

    def explicit_zero(request: httpx.Request) -> httpx.Response:
        return json_response(
            {
                "answers": {},
                "usage": {"input_tokens": 0, "output_tokens": 0, "cost": 0.000021},
            }
        )

    absent = await TypeSafeVendor(manager, client=client_for(no_usage)).decide(
        request_of(choice_question()), timeout_s=5.0
    )
    assert absent.input_tokens is None
    assert absent.output_tokens is None
    assert absent.cost_usd is None

    reported = await TypeSafeVendor(manager, client=client_for(explicit_zero)).decide(
        request_of(choice_question()), timeout_s=5.0
    )
    assert reported.input_tokens == 0
    assert reported.output_tokens == 0
    assert reported.cost_usd == pytest.approx(0.000021)


async def test_an_unanswered_question_is_not_a_failure(manager) -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        return json_response({"answers": {}})

    response = await TypeSafeVendor(manager, client=client_for(handler)).decide(
        request_of(choice_question()), timeout_s=5.0
    )
    assert response.answers == {}


async def test_an_invented_choice_id_is_refused(manager) -> None:
    """The model cannot invent an option; if it names one, the leg is unusable."""

    def handler(request: httpx.Request) -> httpx.Response:
        return json_response(
            {"answers": {"recommend_skill": {"type": "choice", "choice": "made-up"}}}
        )

    with pytest.raises(DecisionVendorError, match="never offered") as caught:
        await TypeSafeVendor(manager, client=client_for(handler)).decide(
            request_of(choice_question()), timeout_s=5.0
        )
    assert caught.value.kind == "response"


async def test_a_200_without_an_answers_block_is_a_leg_failure(manager) -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        return json_response({"model": "x"})

    with pytest.raises(DecisionVendorError, match="no answers block"):
        await TypeSafeVendor(manager, client=client_for(handler)).decide(
            request_of(choice_question()), timeout_s=5.0
        )


# ---------------------------------------------------------------------------
# The error taxonomy (§3, §4)
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("status", "kind"),
    [(401, "auth"), (403, "auth"), (429, "rate-limit"), (529, "overloaded"), (503, "server")],
)
async def test_a_refusal_is_a_fall_through_error(manager, status: int, kind: str) -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(status, json={"error": {"message": "nope"}})

    with pytest.raises(DecisionVendorError) as caught:
        await TypeSafeVendor(manager, client=client_for(handler)).decide(
            request_of(choice_question()), timeout_s=5.0
        )
    assert (caught.value.kind, caught.value.status) == (kind, status)


async def test_a_plain_400_is_weather_not_our_bug(manager) -> None:
    """An unknown model id answers 400 with no field path — that is the next leg's turn."""

    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(400, json={"error": {"message": "no such model", "code": 400}})

    with pytest.raises(DecisionVendorError) as caught:
        await OpenRouterVendor(manager, client=client_for(handler)).decide(
            request_of(choice_question()), timeout_s=5.0
        )
    assert caught.value.kind == "http"
    assert not isinstance(caught.value, DecisionSchemaError)


async def test_a_field_level_400_is_a_schema_error_naming_the_question(manager) -> None:
    """Measured 2026-09-18: the alpha route reports a malformed question as a 400.

    The body is the real one, Zod ``path`` and all, because the discriminator in
    ``_schema_error`` keys off that path: without it a 400 and a 422 are
    indistinguishable from a bad model id.
    """
    body = {
        "error": {
            "message": json.dumps(
                [
                    {
                        "expected": "record",
                        "code": "invalid_type",
                        "path": ["questions", "recommend_skill", "criteria"],
                        "message": "Invalid input: expected record, received array",
                    }
                ]
            ),
            "code": 400,
        }
    }

    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(400, json=body)

    with pytest.raises(DecisionSchemaError) as caught:
        await OpenRouterVendor(manager, client=client_for(handler)).decide(
            request_of(choice_question()), timeout_s=5.0
        )
    assert caught.value.shape == "recommend_skill"
    assert caught.value.status == 400
    assert "criteria" in str(caught.value)


async def test_a_422_zod_list_is_a_schema_error_too(manager) -> None:
    body = {"error": {"message": [{"path": ["questions", "recommend_guide"], "code": "invalid"}]}}

    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(422, json=body)

    with pytest.raises(DecisionSchemaError) as caught:
        await TypeSafeVendor(manager, client=client_for(handler)).decide(
            request_of(choice_question()), timeout_s=5.0
        )
    assert caught.value.shape == "recommend_guide"


async def test_a_schema_complaint_that_names_no_question_disables_nothing(manager) -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(422, json={"error": {"message": "invalid body"}})

    with pytest.raises(DecisionSchemaError) as caught:
        await TypeSafeVendor(manager, client=client_for(handler)).decide(
            request_of(choice_question()), timeout_s=5.0
        )
    assert caught.value.shape is None


async def test_a_transport_failure_is_the_fall_through_class(manager) -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        raise httpx.ConnectTimeout("timed out")

    with pytest.raises(DecisionVendorError) as caught:
        await TypeSafeVendor(manager, client=client_for(handler)).decide(
            request_of(choice_question()), timeout_s=5.0
        )
    assert caught.value.kind == "transport"
    assert not isinstance(caught.value, DecisionSchemaError)


async def test_an_error_body_echoing_our_key_is_scrubbed(manager) -> None:
    """A hostile/misconfigured upstream that echoes the bearer must not reach a log."""

    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(
            401, json={"error": {"message": f"bad token {KEY} sk-abcdefghij1234"}}
        )

    vendor = TypeSafeVendor(manager, client=client_for(handler))
    vendor._key = SecretStr(KEY)  # the credential this leg would send
    with pytest.raises(DecisionVendorError) as caught:
        await vendor.decide(request_of(choice_question()), timeout_s=5.0)
    assert KEY not in str(caught.value)
    assert "sk-abcdefghij1234" not in str(caught.value)


# ---------------------------------------------------------------------------
# Credential resolution, per leg (§3's table)
# ---------------------------------------------------------------------------


async def test_typesafe_prefers_its_own_key_then_the_jev_alias(bare_manager) -> None:
    assert await credential_of(TypeSafeVendor(bare_manager), bare_manager) is None
    bare_manager.set_credential("JEV_API_KEY", "jev-key", write=False)
    assert await credential_of(TypeSafeVendor(bare_manager), bare_manager) == "jev-key"
    bare_manager.set_credential("TYPESAFE_API_KEY", "typesafe-key", write=False)
    assert await credential_of(TypeSafeVendor(bare_manager), bare_manager) == "typesafe-key"


async def test_openrouter_falls_back_to_the_dev_key(bare_manager) -> None:
    assert await credential_of(OpenRouterVendor(bare_manager), bare_manager) is None
    bare_manager.set_credential("OPENROUTER_API_KEY_DEV", "dev-key", write=False)
    assert await credential_of(OpenRouterVendor(bare_manager), bare_manager) == "dev-key"
    bare_manager.set_credential("OPENROUTER_API_KEY", "prod-key", write=False)
    assert await credential_of(OpenRouterVendor(bare_manager), bare_manager) == "prod-key"


async def test_radient_falls_back_to_the_static_key_when_the_session_has_none(bare_manager) -> None:
    """A store with no Radient row resolves through the static tier, not to None."""
    assert await credential_of(RadientVendor(bare_manager), bare_manager) is None
    bare_manager.set_credential("RADIENT_API_KEY", "radient-key", write=False)
    assert await credential_of(RadientVendor(bare_manager), bare_manager) == "radient-key"


async def test_a_leg_with_no_credential_refuses_before_any_http(bare_manager) -> None:
    calls: list[httpx.Request] = []

    def handler(request: httpx.Request) -> httpx.Response:  # pragma: no cover - must not run
        calls.append(request)
        return json_response({"answers": {}})

    with pytest.raises(DecisionVendorError) as caught:
        await TypeSafeVendor(bare_manager, client=client_for(handler)).decide(
            request_of(choice_question()), timeout_s=5.0
        )
    assert caught.value.kind == "auth"
    assert calls == []


async def test_the_resolved_credential_is_memoized_inside_its_ttl(manager) -> None:
    manager.set_credential("TYPESAFE_API_KEY", "first", write=False)
    vendor = TypeSafeVendor(manager)
    assert await credential_of(vendor, manager) == "first"
    manager.set_credential("TYPESAFE_API_KEY", "second", write=False)
    # Inside the TTL the memo answers, so the store is not read per message; the
    # TTL is the bound on how stale that can be (see the expiry test).
    assert await credential_of(vendor, manager) == "first"


async def test_build_vendor_refuses_an_unknown_leg(manager) -> None:
    with pytest.raises(KeyError):
        build_vendor("not-a-leg", manager)


# ---------------------------------------------------------------------------
# The credential memo: TTL, and the once-only re-resolve on a 401
# ---------------------------------------------------------------------------


async def test_the_credential_memo_ttl_matches_the_servers_auth_cache(bare_manager) -> None:
    """``internal/cache/auth_cache.go``: ``defaultAuthCacheTTL = 5 * time.Minute``."""
    from local_operator.classification.vendors import CREDENTIAL_TTL_S

    assert CREDENTIAL_TTL_S == 300.0


async def test_a_rotated_credential_is_picked_up_after_the_ttl(bare_manager) -> None:
    """The TTL is what stops a memo outliving a rotated or logged-out credential.

    Driven by an injected clock rather than by sleeping: a five-minute TTL that
    only a five-minute test can exercise is a TTL nobody tests.
    """
    now = [0.0]
    vendor = TypeSafeVendor(bare_manager, credential_ttl_s=300.0, clock=lambda: now[0])
    bare_manager.set_credential("TYPESAFE_API_KEY", "original", write=False)
    assert await credential_of(vendor, bare_manager) == "original"

    bare_manager.set_credential("TYPESAFE_API_KEY", "rotated", write=False)
    now[0] = 299.0
    # Still inside the TTL: the memo answers, and no store read happens.
    assert await credential_of(vendor, bare_manager) == "original"

    now[0] = 300.5
    assert await credential_of(vendor, bare_manager) == "rotated"


async def test_the_memo_is_per_instance_not_a_module_global(bare_manager, tmp_path) -> None:
    """Two services in one process may be signed into different accounts."""
    from local_operator.credentials import CredentialManager

    other_dir = tmp_path / "other"
    other_dir.mkdir()
    other = CredentialManager(other_dir)
    other.set_credential("TYPESAFE_API_KEY", "other-account-key", write=False)
    bare_manager.set_credential("TYPESAFE_API_KEY", "this-account-key", write=False)

    first = TypeSafeVendor(bare_manager)
    second = TypeSafeVendor(other)
    assert await credential_of(first, bare_manager) == "this-account-key"
    assert await credential_of(second, other) == "other-account-key"


async def test_a_401_invalidates_the_memo_once_and_re_resolves_exactly_once(manager) -> None:
    """A revoked key must not become a per-message retry loop.

    The REQUEST count IS the assertion: one initial attempt plus one retry, not a
    loop. There are three resolves and the extra one is the *fallback tier* lookup
    (this manager has no ``JEV_API_KEY``, so the walk past the refused tier finds
    nothing and the leg fails) — still one pass down the tier list, which is what
    "exactly once" means here: the tier index only advances, so the next message
    starts below the tier that refused.
    """
    resolves: list[int] = []
    requests: list[httpx.Request] = []

    class _Counting(TypeSafeVendor):
        async def _resolve_key(self, manager):  # type: ignore[no-untyped-def]
            resolves.append(1)
            return await super()._resolve_key(manager)

    def handler(request: httpx.Request) -> httpx.Response:
        requests.append(request)
        return httpx.Response(401, json={"error": {"message": "revoked"}})

    vendor = _Counting(manager, client=client_for(handler))
    with pytest.raises(DecisionVendorError) as caught:
        await vendor.decide(request_of(choice_question()), timeout_s=5.0)
    assert caught.value.kind == "auth"
    assert len(requests) == 2
    assert len(resolves) == 3
    # The tier that refused is behind us for the rest of the session: tiers are
    # (authstore, TYPESAFE_API_KEY, JEV_API_KEY) and the env key was index 1.
    assert vendor._tier == 2


async def test_a_401_that_heals_on_the_retry_answers_the_call(manager) -> None:
    resolves: list[int] = []
    requests: list[httpx.Request] = []

    class _Counting(TypeSafeVendor):
        async def _resolve_key(self, manager):  # type: ignore[no-untyped-def]
            resolves.append(1)
            return await super()._resolve_key(manager)

    def handler(request: httpx.Request) -> httpx.Response:
        requests.append(request)
        if len(requests) == 1:
            return httpx.Response(401, json={"error": {"message": "expired"}})
        return json_response(
            {
                "answers": {
                    "recommend_skill": {"type": "choice", "choice": "minerva-deploy"},
                }
            }
        )

    vendor = _Counting(manager, client=client_for(handler))
    response = await vendor.decide(request_of(choice_question()), timeout_s=5.0)
    assert response.answers["recommend_skill"].value == "minerva-deploy"
    assert len(requests) == 2
    assert len(resolves) == 2

    # The refreshed credential is memoized: a second call resolves nothing new.
    await vendor.decide(request_of(choice_question()), timeout_s=5.0)
    assert len(resolves) == 2
    assert len(requests) == 3


async def test_a_403_is_handled_the_same_way(manager) -> None:
    requests: list[httpx.Request] = []

    def handler(request: httpx.Request) -> httpx.Response:
        requests.append(request)
        return httpx.Response(403, json={"error": {"message": "forbidden"}})

    with pytest.raises(DecisionVendorError) as caught:
        await TypeSafeVendor(manager, client=client_for(handler)).decide(
            request_of(choice_question()), timeout_s=5.0
        )
    assert caught.value.kind == "auth"
    assert len(requests) == 2


async def test_a_429_is_not_retried(manager) -> None:
    """Only an auth refusal invalidates a credential; a rate limit is the cascade's."""
    requests: list[httpx.Request] = []

    def handler(request: httpx.Request) -> httpx.Response:
        requests.append(request)
        return httpx.Response(429, json={"error": {"message": "slow down"}})

    with pytest.raises(DecisionVendorError) as caught:
        await TypeSafeVendor(manager, client=client_for(handler)).decide(
            request_of(choice_question()), timeout_s=5.0
        )
    assert caught.value.kind == "rate-limit"
    assert len(requests) == 1


# ---------------------------------------------------------------------------
# Where the credential actually lives: the login row in auth.db comes FIRST
#
# ``ProviderController.login`` writes a pasted key into the AuthStore under the
# provider id (``upsert_credential(store_credentials_as or provider_id, {"key":
# <pasted>, "source": "login", "type": "api_key"})``). A leg that read only
# ``CredentialManager`` — env and legacy ``credentials.env`` — ignored
# ``lop login typesafe`` entirely, silently: the user saw "Stored API key" and
# the classifier fell through to the next leg, or to no vendor at all.
# ---------------------------------------------------------------------------


def store_login_key(manager, provider: str, key: str, *, replace: bool = False) -> None:
    """Write a pasted-key row EXACTLY as ``ProviderController.login`` writes it.

    ``replace`` is how a ROTATION is expressed. The store resolves the first
    login row it has for a provider, so a second ``upsert_credential`` for the
    same provider adds a row that the resolver then ignores — that is the
    provider stack's own behaviour (a re-login without an identity key leaves the
    older row winning), not something this leg introduces, and it is why a test
    that means "the key changed" must delete the old row rather than add one.
    """
    from local_operator.providers.auth_store import AuthStore

    store = AuthStore(manager.config_dir / "auth.db", credential_manager=manager)
    try:
        if replace:
            for row in store.list_credentials():
                if row.provider == provider:
                    store.delete_credential(row.id)
        store.upsert_credential(provider, {"key": key, "source": "login", "type": "api_key"})
    finally:
        store.close()


@pytest.mark.parametrize(
    ("provider", "vendor_class"),
    [("typesafe", TypeSafeVendor), ("openrouter", OpenRouterVendor)],
    ids=["typesafe", "openrouter"],
)
async def test_a_key_stored_by_login_is_used_when_no_env_key_exists(
    bare_manager, provider: str, vendor_class: type
) -> None:
    """THE regression: nothing in the environment, everything in the store.

    Before the fix this leg resolved ``None`` — the login row was invisible — and
    the cascade either fell to a later leg or reported no vendor at all.
    """
    store_login_key(bare_manager, provider, f"{provider}-login-key")
    vendor = vendor_class(bare_manager)
    assert await credential_of(vendor, bare_manager) == f"{provider}-login-key"


async def test_the_credential_manager_tier_still_answers(bare_manager) -> None:
    """Regression guard for the env / ``credentials.env`` path, with the store empty."""
    assert not (bare_manager.config_dir / "auth.db").exists()
    bare_manager.set_credential("TYPESAFE_API_KEY", "env-key", write=False)
    assert await credential_of(TypeSafeVendor(bare_manager), bare_manager) == "env-key"
    # A first resolve created the store; the assertion above only holds because
    # nothing wrote a row into it.
    assert await credential_of(TypeSafeVendor(bare_manager), bare_manager) == "env-key"


async def test_the_login_row_wins_over_the_environment_key(bare_manager) -> None:
    """THE precedence rule: the store row first, the env tiers behind it.

    Both are "the credential", and they disagree, so the order has to be stated
    rather than left to whichever tier a reader happened to look at first: the
    login row is the credential the operator actively stored in this harness,
    while an exported variable is ambient and may belong to another tool.
    """
    store_login_key(bare_manager, "typesafe", "login-row-key")
    bare_manager.set_credential("TYPESAFE_API_KEY", "env-key", write=False)
    assert await credential_of(TypeSafeVendor(bare_manager), bare_manager) == "login-row-key"

    store_login_key(bare_manager, "openrouter", "login-row-key")
    bare_manager.set_credential("OPENROUTER_API_KEY_DEV", "dev-env-key", write=False)
    assert await credential_of(OpenRouterVendor(bare_manager), bare_manager) == "login-row-key"


async def test_the_alternates_still_answer_behind_the_store_and_the_primary_env_key(
    bare_manager,
) -> None:
    assert not (bare_manager.config_dir / "auth.db").exists()
    bare_manager.set_credential("JEV_API_KEY", "jev-key", write=False)
    assert await credential_of(TypeSafeVendor(bare_manager), bare_manager) == "jev-key"


async def test_a_login_stored_key_is_memoized_and_re_resolved_once_on_a_401(bare_manager) -> None:
    """The store tier goes through the same memo and the same once-only retry.

    The retry is proven to have RE-READ the store: the row is rotated between the
    two attempts, and the second request's bearer is the rotated key.
    """
    store_login_key(bare_manager, "openrouter", "first-key")
    resolves: list[int] = []
    bearers: list[str] = []

    class _Counting(OpenRouterVendor):
        async def _resolve_key(self, manager):  # type: ignore[no-untyped-def]
            resolves.append(1)
            return await super()._resolve_key(manager)

    def handler(request: httpx.Request) -> httpx.Response:
        bearers.append(request.headers["Authorization"])
        if len(bearers) == 1:
            store_login_key(bare_manager, "openrouter", "rotated-key", replace=True)
            return httpx.Response(401, json={"error": {"message": "stale"}})
        return json_response(
            {"answers": {"recommend_skill": {"type": "choice", "choice": "minerva-deploy"}}}
        )

    vendor = _Counting(bare_manager, client=client_for(handler))
    response = await vendor.decide(request_of(choice_question()), timeout_s=5.0)
    assert response.answers["recommend_skill"].value == "minerva-deploy"
    assert bearers == ["Bearer first-key", "Bearer rotated-key"]
    assert len(resolves) == 2  # one initial + one re-resolve, never a loop

    # The refreshed row is memoized: a third call resolves nothing new.
    await vendor.decide(request_of(choice_question()), timeout_s=5.0)
    assert len(resolves) == 2
    assert bearers[2] == "Bearer rotated-key"


async def test_a_rotated_login_key_is_picked_up_after_the_ttl(bare_manager) -> None:
    now = [0.0]
    store_login_key(bare_manager, "typesafe", "old-row-key")
    vendor = TypeSafeVendor(bare_manager, credential_ttl_s=300.0, clock=lambda: now[0])
    assert await credential_of(vendor, bare_manager) == "old-row-key"

    store_login_key(bare_manager, "typesafe", "new-row-key", replace=True)
    now[0] = 299.0
    assert await credential_of(vendor, bare_manager) == "old-row-key"
    now[0] = 300.5
    assert await credential_of(vendor, bare_manager) == "new-row-key"


async def test_the_storage_id_follows_the_registrys_store_credentials_as(bare_manager) -> None:
    """Derived, not hardcoded: a registry alias would otherwise be missed.

    ``xai-oauth`` is the registry's own example (it stores under ``xai``), and
    the login path resolves the same expression before writing.
    """
    from local_operator.classification.vendors import storage_provider_id

    assert storage_provider_id("xai-oauth") == "xai"
    assert storage_provider_id("openrouter") == "openrouter"
    # A provider with no registry row degrades to its own id rather than raising,
    # so this leg still works in a tree where the registry entry has not landed.
    assert storage_provider_id("not-a-provider") == "not-a-provider"


# ---------------------------------------------------------------------------
# The preference order, and the fallback behind it (operator requirement,
# 2026-09-18): the login row is PREFERRED, the static key is LEGACY.
# ---------------------------------------------------------------------------


def _answers_choice() -> httpx.Response:
    return json_response(
        {"answers": {"recommend_skill": {"type": "choice", "choice": "minerva-deploy"}}}
    )


async def test_the_radient_leg_prefers_the_login_session_over_the_static_key(
    bare_manager,
) -> None:
    """Both credits present and disagreeing: the LOGIN row is what gets sent.

    Radient's own case of the documented precedence — the OAuth session an
    interactive login wrote is the preferred credential, and ``RADIENT_API_KEY``
    is the legacy static tier behind it. Asserted on the WIRE rather than on the
    resolver, because "which one is preferred" is only observable as the bearer
    the vendor receives.
    """
    store_login_key(bare_manager, "radient", "oauth-session-bearer")
    bare_manager.set_credential("RADIENT_API_KEY", "legacy-static-key", write=False)
    bearers: list[str] = []

    def handler(request: httpx.Request) -> httpx.Response:
        bearers.append(request.headers["Authorization"])
        return _answers_choice()

    vendor = RadientVendor(bare_manager, client=client_for(handler))
    await vendor.decide(request_of(choice_question()), timeout_s=5.0)

    assert bearers == ["Bearer oauth-session-bearer"]


async def test_a_dead_login_row_falls_back_to_the_legacy_static_key(bare_manager) -> None:
    """The login row is preferred, not absolute: a dead one stops hiding the key.

    Before the fallback, the once-only retry re-resolved from the top, got the
    SAME stored row back, and the leg failed for the whole session — so a legacy
    static key the operator had exported was unreachable in practice. The
    sequence asserted here is: refused login row, one re-read (which proves the
    value did not rotate), then the tier behind it.
    """
    store_login_key(bare_manager, "radient", "dead-oauth-bearer")
    bare_manager.set_credential("RADIENT_API_KEY", "legacy-static-key", write=False)
    bearers: list[str] = []

    def handler(request: httpx.Request) -> httpx.Response:
        bearer = request.headers["Authorization"]
        bearers.append(bearer)
        if bearer == "Bearer dead-oauth-bearer":
            return httpx.Response(401, json={"error": {"message": "revoked"}})
        return _answers_choice()

    vendor = RadientVendor(bare_manager, client=client_for(handler))
    response = await vendor.decide(request_of(choice_question()), timeout_s=5.0)

    assert response.answers["recommend_skill"].value == "minerva-deploy"
    assert bearers == [
        "Bearer dead-oauth-bearer",
        "Bearer dead-oauth-bearer",
        "Bearer legacy-static-key",
    ]

    # The advance is MEMOIZED for the session: the next message starts at the
    # fallback tier instead of paying the dead row again.
    await vendor.decide(request_of(choice_question()), timeout_s=5.0)
    assert bearers[3] == "Bearer legacy-static-key"
    assert len(bearers) == 4


async def test_a_rotated_login_row_still_heals_without_advancing_a_tier(
    bare_manager,
) -> None:
    """The fallback must not fire when the preferred tier merely rotated.

    Same shape as the store-rotation test above, asserted through the TIER INDEX:
    a refreshed value is not a dead tier, so the session keeps using the login
    row rather than silently downgrading to the legacy key.
    """
    store_login_key(bare_manager, "radient", "first-bearer")
    bare_manager.set_credential("RADIENT_API_KEY", "legacy-static-key", write=False)
    bearers: list[str] = []

    def handler(request: httpx.Request) -> httpx.Response:
        bearer = request.headers["Authorization"]
        bearers.append(bearer)
        if len(bearers) == 1:
            store_login_key(bare_manager, "radient", "rotated-bearer", replace=True)
            return httpx.Response(401, json={"error": {"message": "stale"}})
        return _answers_choice()

    vendor = RadientVendor(bare_manager, client=client_for(handler))
    await vendor.decide(request_of(choice_question()), timeout_s=5.0)

    assert bearers == ["Bearer first-bearer", "Bearer rotated-bearer"]
    assert vendor._tier == 0


async def test_every_tier_dead_walks_each_of_them_at_most_once(bare_manager) -> None:
    """Bounded when nothing works: no per-message hammering of a dead store.

    The login row is the only credential here, so the walk runs out after
    advancing past it. The second call must therefore make NO request at all
    (the leg now reports "no credential"), which is what stops a genuinely
    revoked session from costing two store reads and two POSTs per message.
    """
    store_login_key(bare_manager, "radient", "dead-oauth-bearer")
    requests: list[httpx.Request] = []

    def handler(request: httpx.Request) -> httpx.Response:
        requests.append(request)
        return httpx.Response(403, json={"error": {"message": "forbidden"}})

    vendor = RadientVendor(bare_manager, client=client_for(handler))
    with pytest.raises(DecisionVendorError) as first:
        await vendor.decide(request_of(choice_question()), timeout_s=5.0)
    assert first.value.kind == "auth"
    assert len(requests) == 2

    with pytest.raises(DecisionVendorError) as second:
        await vendor.decide(request_of(choice_question()), timeout_s=5.0)
    assert second.value.kind == "auth"
    assert "no credential" in str(second.value)
    assert len(requests) == 2
    assert vendor._tier == 1


async def test_the_walk_restarts_at_the_preferred_tier_when_the_memo_expires(
    bare_manager,
) -> None:
    """A re-login must revive a RUNNING session, so the fallback is not one-way.

    The tier index advances only on a refusal, so without the memo-expiry reset a
    session that fell back once would keep using the legacy key for the rest of its
    life — `lop login` could not bring the preferred tier back without a restart
    (agent review round 1). The reset rides the expiry that is already happening, so
    the extra consult costs no request of its own.
    """
    now = [0.0]
    store_login_key(bare_manager, "radient", "first-login-bearer")
    bare_manager.set_credential("RADIENT_API_KEY", "legacy-static-key", write=False)
    bearers: list[str] = []

    def handler(request: httpx.Request) -> httpx.Response:
        bearer = request.headers["Authorization"].replace("Bearer ", "")
        bearers.append(bearer)
        if bearer == "first-login-bearer":
            return httpx.Response(401, json={"error": {"message": "revoked"}})
        return _answers_choice()

    vendor = RadientVendor(
        bare_manager,
        client=client_for(handler),
        credential_ttl_s=300.0,
        clock=lambda: now[0],
    )

    # The login row refuses twice, so the walk falls back inside this call.
    await vendor.decide(request_of(choice_question()), timeout_s=5.0)
    assert bearers[-1] == "legacy-static-key"
    assert vendor._tier == 1

    # The operator logs in again while the session is running, and the memo expires.
    store_login_key(bare_manager, "radient", "second-login-bearer", replace=True)
    now[0] = 301.0
    await vendor.decide(request_of(choice_question()), timeout_s=5.0)

    assert bearers[-1] == "second-login-bearer"
    assert vendor._tier == 0
