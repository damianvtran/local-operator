"""The three cascade legs: Radient, TypeSafe, OpenRouter.

They are three `POST`s with byte-identical bodies (`docs/design/classification-layer.md`
§3), which is the whole reason the cascade is a list and not a branch. This
module owns only the parts that genuinely differ per leg — endpoint, model id,
credential resolution — plus the one shared wire codec and the one shared error
taxonomy. Everything about *ordering*, fallback and disabling lives in
:mod:`local_operator.classification.cascade` and
:mod:`local_operator.classification.service`.

WHY THE CLIENT IS INJECTED, AND WHO OWNS IT
===========================================

A leg takes an ``httpx.AsyncClient`` rather than making one per request. A
connection plus a TLS handshake to ``api.radienthq.com`` is 50-150 ms on its
own, and this layer runs once per user message on the critical path before the
turn's first token — so paying that per message would spend the entire latency
budget on a handshake. :class:`~local_operator.classification.service.ClassificationService`
therefore owns ONE keep-alive client for the session and injects it here (and
``httpx.MockTransport`` is how the tests inject a fake one). A leg used without
an injected client still works — it opens one for the call — which keeps the
classes usable on their own in a diagnostic or a one-off script.

HOW A LEG FINDS ITS CREDENTIAL
===============================

The order is the provider stack's, and it is the same for all three legs
(:meth:`_HttpDecisionVendor._resolve_key`):

1. **The ``AuthStore`` row for the provider** — ``auth.db``, read-only. This is
   where an interactive login puts a key: ``ProviderController.login`` does
   ``upsert_credential(store_credentials_as or provider_id, {"key": <pasted>,
   "source": "login", "type": "api_key"})``. A leg that skipped this store
   silently ignored ``lop login typesafe`` — the user saw "Stored API key" and
   the classifier then fell through to the next leg, or to no vendor at all.
2. **``CredentialManager.get_credential(<env key>)``** — the process environment
   and the legacy ``credentials.env``. Still needed explicitly: the store's own
   env tier reads only a provider's *single-string* ``env_keys``, so a tuple
   like TypeSafe's ``("TYPESAFE_API_KEY", "JEV_API_KEY")`` is not covered there,
   and neither is ``OPENROUTER_API_KEY_DEV``.
3. **The vendor-specific alternates** — ``JEV_API_KEY`` for TypeSafe,
   ``OPENROUTER_API_KEY_DEV`` behind the production key for OpenRouter.
4. ``None`` — "this leg has no credential", which the cascade treats as "skip".

``read_only=True`` for the same reason ``providers/radient_credentials.py`` uses
it: a decorative classifier call must not block a credential, move account
stickiness or decide routing. A required OAuth refresh still persists centrally
inside the store, which is that account's own bookkeeping.

CREDENTIALS ARE MEMOIZED WITH A SHORT TTL
=========================================

``AuthStore.get_api_key`` walks a 7-step cascade over SQLite and can refresh an
OAuth grant over the network; on this path that may happen once per user
message. So a resolved credential is memoized on the LEG INSTANCE — which lives
on the service instance, never in a module-level global, because two sessions in
one process may be signed into different accounts and a global would send one
account's key in another's request.

The memo expires after :data:`CREDENTIAL_TTL_S` (5 minutes), matching the
agent-server's own auth cache (``internal/cache/auth_cache.go``,
``defaultAuthCacheTTL = 5 * time.Minute``). The TTL is what bounds staleness: a
memo with no TTL keeps using a key the operator has since rotated or logged out
of, for the whole session. And when a leg answers 401/403, the memo is dropped
and the credential re-resolved ONCE — never in a loop — so a rotated key heals
on the next message while a genuinely dead one stops costing a request per
message.

NO KEY EVER LEAVES THIS MODULE
==============================

Credential values are read here and handed straight to an ``Authorization``
header. They are never logged, never interpolated into an error message and
never returned in :class:`~local_operator.classification.types.DecisionResponse`.
:class:`DecisionVendorError` carries the upstream status and a truncated body —
and that body goes through the shared ``scrub_secrets`` (with THIS leg's own
credential passed as an exact value), because an upstream that echoes the
``Authorization`` header back in a refusal is the one realistic route by which a
key could otherwise reach a log line from here.
"""

from __future__ import annotations

import json
import logging
import time
from collections.abc import Callable
from typing import TYPE_CHECKING, Any, Mapping, Sequence

import httpx
from pydantic import SecretStr

from local_operator.classification.types import (
    Answer,
    DecisionRequest,
    DecisionResponse,
    DecisionSchemaError,
    DecisionVendorError,
    Question,
)
from local_operator.clients._http import scrub_secrets
from local_operator.providers.registry import get_provider_definition

if TYPE_CHECKING:
    from local_operator.credentials import CredentialManager

logger = logging.getLogger(__name__)

RADIENT_ENDPOINT = "https://api.radienthq.com/v1/decisions"
"""Radient's passthrough of the TypeSafe shape (§10)."""

TYPESAFE_ENDPOINT = "https://api.typesafe.ai/v1/systemone"
"""TypeSafe's native "System One" endpoint — the shape the other two copy."""

OPENROUTER_ENDPOINT = "https://openrouter.ai/api/alpha/decisions"
"""OpenRouter's ALPHA decisions route.

NOT ``/api/v1/chat/completions``: OpenRouter rejects this model there with a 400
that names the alpha path, so moving the leg onto the chat route "to simplify"
turns every OpenRouter call into a fall-through (§3).
"""

RADIENT_MODEL = "jev-1.13"
TYPESAFE_MODEL = "jev-1.13.0"
OPENROUTER_MODEL = "typesafe/jev-1.13"

#: How long a resolved credential is reused before it is resolved again.
#: Matches the agent-server's auth cache TTL (``internal/cache/auth_cache.go``,
#: ``defaultAuthCacheTTL = 5 * time.Minute``) so the two halves of the system age
#: a credential out on the same schedule.
CREDENTIAL_TTL_S = 300.0

#: Connection-pool policy for a decision client: keep-alive is the whole point
#: (the cascade is at most one request per user message, so a wide pool would
#: only hold sockets the process is not using), and the idle expiry keeps a
#: long-lived session from holding a server-closed connection open.
_LIMITS = httpx.Limits(max_connections=4, max_keepalive_connections=2, keepalive_expiry=30.0)

#: Bodies above this length are truncated in an error message. An upstream
#: error body is prose for a human; 400 characters is enough to see the shape
#: of a schema complaint and short enough that a stray HTML error page cannot
#: flood a log line.
_ERROR_BODY_CHARS = 400


def questions_payload(questions: Sequence[Question]) -> dict[str, dict[str, Any]]:
    """Serialize questions into the vendor's ``{id: {type, instructions, criteria}}`` map.

    The two shapes this must get right, both measured against the live alpha
    route on 2026-09-18:

    * a ``choice``/``noul`` criterion VALUE is a **string** — the rich
      ``{what, includes, not_for}`` object form is what TypeSafe's own docs show,
      and the contract forbids it here (§3);
    * a ``score`` criterion is an **array**, in level order, because the vendor
      answers a score question with a float index into it.

    Both are asserted in the unit tests, and the live test in
    ``tests/unit/classification/test_live_openrouter.py`` exercises the string
    form against the real route.
    """
    payload: dict[str, dict[str, Any]] = {}
    for question in questions:
        payload[question.id] = {
            "type": question.kind,
            "instructions": question.instructions,
            "criteria": _criteria_payload(question),
        }
    return payload


def _criteria_payload(question: Question) -> Any:
    """The JSON value of one question's criteria: a map for choice/noul, an array for score.

    The shape is checked here rather than assumed from the annotation for the
    same reason ``Question.__post_init__`` checks it at construction: a wrong
    shape must fail as a local error in a stack trace we own, not as a vendor
    refusal whose only trace is a status code. Reaching a raise here means
    something built a ``Question`` without going through the dataclass.
    """
    criteria = question.criteria
    if question.kind == "score":
        if isinstance(criteria, dict):
            raise DecisionSchemaError(
                f"score question {question.id!r} carries a mapping; it needs an array of levels"
            )
        return list(criteria)
    if not isinstance(criteria, dict):
        raise DecisionSchemaError(
            f"{question.kind} question {question.id!r} carries an array; it needs option "
            "descriptions"
        )
    return dict(criteria)


def request_body(request: DecisionRequest, model: str) -> dict[str, Any]:
    """The complete vendor body: model, state, questions."""
    return {
        "model": model,
        "state": request.state,
        "questions": questions_payload(request.questions),
    }


def _error_text(response: httpx.Response, secret: str) -> str:
    """The upstream's own words, truncated and SCRUBBED, for a human reading a log line.

    Scrubbed with this package's own credential passed as an exact value, on top
    of the shared shape-based scrubber: an upstream that echoes the
    ``Authorization`` header back in an error body is the one realistic way a
    key could reach a log line from here, and the fix is cheaper than the audit.
    """
    text = response.text or ""
    return scrub_secrets(text[:_ERROR_BODY_CHARS], (secret,))


def _schema_error(response: httpx.Response, secret: str) -> DecisionSchemaError | None:
    """A request-shape rejection, or ``None`` when the status is not one.

    Two rules, because the two statuses are not the same event — and the live
    route proved it (measured 2026-09-18):

    * **422 — ours, always.** §4 names a 422 as our bug, so it is a
      :class:`DecisionSchemaError` whatever its body says. The body is still read
      for a question id, because disabling one shape beats disabling nothing.
    * **400 — ours only when the body names one of OUR question fields.** The
      alpha route answers a malformed question with 400, and it answers an
      unknown model id with 400 too — and that second one is weather, i.e. the
      next leg's turn. What separates them is a field ``path`` pointing into
      ``questions.<question id>...``; a 400 without one stays a
      :class:`DecisionVendorError` and falls through, exactly as §3 says.

    See ``_schema_problem`` for the shapes the complaint arrives in.
    """
    status = response.status_code
    if status not in (400, 422):
        return None
    shape, detail = _schema_problem(response)
    if status == 400 and shape is None:
        return None
    named = f" {shape!r}" if shape else ""
    return DecisionSchemaError(
        scrub_secrets(
            f"vendor rejected our request shape{named} with {status}: {detail}", (secret,)
        ),
        shape=shape,
        status=status,
    )


def _schema_problem(response: httpx.Response) -> tuple[str | None, str]:
    """The rejected question id and the vendor's detail, from whatever shape the body took.

    Tolerant on purpose. On the live route the complaint arrives as a JSON
    *string* holding a Zod issue list:

        {"error":
          {"message": "[{\"path\": [\"questions\", \"recommend_skill\", \"criteria\"], ...}]"}}

    but the same information is a plain list on a 422 from another surface, so
    both are read rather than one being assumed. A body that cannot be parsed at
    all yields ``(None, "")``: no shape to disable, and the caller's status rule
    decides what the refusal means.
    """
    try:
        body = response.json()
    except (ValueError, json.JSONDecodeError):
        return None, ""
    if not isinstance(body, Mapping):
        return None, ""
    error = body.get("error", body)
    message = error.get("message") if isinstance(error, Mapping) else None
    problems: Any = message
    if isinstance(message, str):
        try:
            problems = json.loads(message)
        except (ValueError, json.JSONDecodeError):
            return None, message
    if isinstance(problems, Mapping):
        problems = [problems]
    if not isinstance(problems, list):
        return None, ""
    for problem in problems:
        if not isinstance(problem, Mapping):
            continue
        path = problem.get("path")
        if isinstance(path, list) and path and path[0] == "questions":
            # The whole path is carried into the message, not just the question
            # id: when this fires the operator needs to know WHICH field of the
            # question was refused (criteria? type?), and that is the one detail
            # the vendor was willing to give us.
            field_path = ".".join(str(part) for part in path)
            return (
                str(path[1]) if len(path) > 1 else None
            ), f"{field_path}: {problem.get('message', '')}"
    return None, ""


def _vendor_error(response: httpx.Response, secret: str) -> DecisionVendorError:
    """Classify a non-200 into the transport-class error the cascade falls through on."""
    status = response.status_code
    if status in (401, 403):
        kind = "auth"
    elif status == 429:
        kind = "rate-limit"
    elif status == 529:
        kind = "overloaded"
    elif status >= 500:
        kind = "server"
    else:
        kind = "http"
    return DecisionVendorError(
        f"decision vendor returned {status}: {_error_text(response, secret)}",
        kind=kind,
        status=status,
    )


def _refusal(
    response: httpx.Response, secret: str
) -> DecisionSchemaError | DecisionVendorError | None:
    """Why this response is not an answer, or ``None`` when it is a 200 we can parse.

    One place decides, so ``decide`` can ask the question twice (before and after
    a credential retry) without re-deriving the order: schema error first, then
    the transport-class refusal, then "this was fine".
    """
    schema_error = _schema_error(response, secret)
    if schema_error is not None:
        return schema_error
    if response.status_code == 200:
        return None
    return _vendor_error(response, secret)


def _answer_from_payload(question: Question, payload: Any) -> Answer:
    """One answer, shaped by what we ASKED, not by what came back.

    Taking the kind from our own question is what makes an unexpected body a
    loud failure (``DecisionVendorError``, next leg) rather than a silently
    mis-typed answer, and it is also what lets the ``noul`` answer — which
    carries neither probabilities nor confidence — come back as a plain
    probability instead of a ``None`` value.
    """
    if not isinstance(payload, Mapping):
        raise DecisionVendorError(f"answer for {question.id!r} is not an object", kind="response")
    if question.kind == "choice":
        choice = payload.get("choice")
        if not isinstance(choice, str):
            raise DecisionVendorError(
                f"choice answer for {question.id!r} carries no choice id", kind="response"
            )
        if isinstance(question.criteria, dict) and choice not in question.criteria:
            # The vendor cannot invent an option; if it names one we never
            # offered, the answer is unusable and the leg is not to be trusted
            # for this turn. Falling through is the contract's response to any
            # unusable answer, so it is the response here too.
            raise DecisionVendorError(
                f"choice answer for {question.id!r} named an option we never offered",
                kind="response",
            )
        probabilities = payload.get("probabilities")
        confidence = payload.get("confidence")
        return Answer(
            id=question.id,
            kind="choice",
            value=choice,
            probabilities=_probabilities(probabilities),
            confidence=float(confidence) if isinstance(confidence, (int, float)) else None,
        )
    if question.kind == "noul":
        noul = payload.get("noul")
        if not isinstance(noul, (int, float)):
            raise DecisionVendorError(
                f"noul answer for {question.id!r} carries no probability", kind="response"
            )
        return Answer(id=question.id, kind="noul", value=float(noul))
    score = payload.get("score")
    if not isinstance(score, (int, float)):
        raise DecisionVendorError(
            f"score answer for {question.id!r} carries no value", kind="response"
        )
    # ``legend`` is the vendor's index -> level-name map; it is kept inside
    # ``probabilities`` (keyed by index, as the per-level distribution already
    # is) rather than re-shaped into a second field, so nothing here guesses
    # which level a float landed on.
    return Answer(
        id=question.id,
        kind="score",
        value=float(score),
        probabilities=_probabilities(payload.get("probabilities")),
        confidence=(
            float(payload["confidence"])
            if isinstance(payload.get("confidence"), (int, float))
            else None
        ),
    )


def _probabilities(raw: Any) -> dict[str, float]:
    """Whatever the vendor sent as a distribution, defensively."""
    if not isinstance(raw, Mapping):
        return {}
    out: dict[str, float] = {}
    for key, value in raw.items():
        if isinstance(value, (int, float)) and not isinstance(value, bool):
            out[str(key)] = float(value)
    return out


def storage_provider_id(provider: str) -> str:
    """The provider id a pasted key is STORED under: ``store_credentials_as or id``.

    Mirrors ``ProviderController.login`` (``providers/controller.py``), which
    resolves the same expression before writing the row — deriving it rather than
    hardcoding each name means an alias added to the registry (``xai-oauth``
    stores under ``xai``) is honoured here without an edit. A provider id with no
    registry row falls back to the id itself, so this leg behaves sensibly in a
    tree where the registry entry has not landed yet.
    """
    definition = get_provider_definition(provider)
    if definition is None:
        return provider
    return definition.store_credentials_as or definition.id


async def auth_store_api_key(manager: "CredentialManager", provider: str) -> str | None:
    """The ``AuthStore`` row a login wrote for ``provider``, or ``None``.

    Read-only (see the module docstring), and every failure degrades to ``None``:
    a corrupt or half-migrated ``auth.db``, or a network failure while refreshing
    an expired grant, must land on "this tier said nothing" so the static tiers
    below still get their turn. This layer is advisory; it may not fail a turn.
    """
    from local_operator.providers.auth_store import AuthStore

    store: AuthStore | None = None
    try:
        store = AuthStore(manager.config_dir / "auth.db", credential_manager=manager)
        return await store.get_api_key(storage_provider_id(provider), read_only=True)
    except Exception:  # noqa: BLE001 — a leg that cannot resolve is not this leg
        logger.warning("classification: %s auth store unavailable", provider, exc_info=True)
        return None
    finally:
        if store is not None:
            store.close()


class _HttpDecisionVendor:
    """Shared wire behaviour; subclasses supply endpoint, model and credential.

    Not part of the public surface — the three named legs below are.
    """

    #: The cascade's identifier for this leg; also what lands in ``DecisionResponse.vendor``.
    name: str = ""
    endpoint: str = ""
    default_model: str = ""
    #: The provider whose login row this leg reads from ``auth.db``. Equals
    #: :attr:`name` for all three legs; kept separate because they are different
    #: vocabularies (a cascade id versus a provider id) that happen to coincide.
    provider_id: str = ""
    #: The env / ``credentials.env`` names to try after the store row, in order.
    env_key_names: tuple[str, ...] = ()

    def __init__(
        self,
        manager: "CredentialManager",
        *,
        model: str = "",
        client: httpx.AsyncClient | None = None,
        credential_ttl_s: float = CREDENTIAL_TTL_S,
        clock: Callable[[], float] = time.monotonic,
    ) -> None:
        self._manager = manager
        # An empty override means "the vendor's own id" — §8's ``model`` key
        # defaults to "", and an empty string is not a model id to send.
        self.model_id = model or self.default_model
        self._client = client
        self._credential_ttl_s = credential_ttl_s
        # ``time.monotonic`` and not the wall clock: a TTL is about elapsed time,
        # and a clock that steps (NTP, a laptop resuming) must not expire a
        # credential early or keep a dead one alive. Injectable so the TTL is
        # testable without sleeping for five minutes.
        self._clock = clock
        self._key: SecretStr | None = None
        self._key_expires_at = 0.0

    async def _resolve_key(self, manager: "CredentialManager") -> SecretStr | None:
        """``AuthStore`` row → ``credentials.env``/env key → alternates → ``None``.

        One implementation for all three legs because the order is the provider
        stack's, not a per-vendor detail; the legs differ only in the ids and key
        names they declare. See the module docstring for why each tier is needed.
        """
        stored = await auth_store_api_key(manager, self.provider_id)
        if stored:
            return SecretStr(stored)
        for name in self.env_key_names:
            value = manager.get_credential(name)
            if value:
                return value
        return None

    async def credential(self, manager: "CredentialManager") -> SecretStr | None:
        """The bearer for this leg, memoized for :data:`CREDENTIAL_TTL_S`.

        Memoized on the instance — i.e. per vendor, per service, per session.
        NOT in a module-level global: two sessions in one process may be signed
        into different accounts, and a global cache would put one account's key
        into another's request. The TTL is what stops a memo from outliving a
        rotated or logged-out credential; see the module docstring.
        """
        if self._key is not None and self._clock() < self._key_expires_at:
            return self._key
        self._key = await self._resolve_key(manager)
        self._key_expires_at = self._clock() + self._credential_ttl_s
        return self._key

    def invalidate_credential(self) -> None:
        """Drop the memo so the next :meth:`credential` re-reads the store.

        Called on a 401/403 (see :meth:`decide`) and nowhere else, and only once
        per call: a leg that re-resolves in a loop turns a revoked key into a
        request storm against the credential store.
        """
        self._key = None
        self._key_expires_at = 0.0

    async def decide(self, request: DecisionRequest, *, timeout_s: float) -> DecisionResponse:
        key = await self.credential(self._manager)
        if key is None or not key.get_secret_value():
            raise DecisionVendorError(f"{self.name} has no credential", kind="auth", status=401)
        body = request_body(request, self.model_id)
        started = time.monotonic()

        response = await self._send_with(
            self._client, body, self._headers(key.get_secret_value()), timeout_s
        )
        refusal = _refusal(response, key.get_secret_value())
        if isinstance(refusal, DecisionSchemaError):
            # Ours, not the vendor's: never quietly retried, never another leg's
            # problem (§4). The cascade turns this into a disabled question shape.
            raise refusal
        if isinstance(refusal, DecisionVendorError) and refusal.status in (401, 403):
            # Exactly one re-resolve-and-retry. Written out rather than looped so
            # the bound is visible: a revoked key costs one extra store read per
            # message, not a request storm.
            logger.info(
                "classification: %s rejected the memoized credential; re-resolving once",
                self.name,
            )
            self.invalidate_credential()
            refreshed = await self.credential(self._manager)
            if refreshed is None or not refreshed.get_secret_value():
                raise refusal
            response = await self._send_with(
                self._client, body, self._headers(refreshed.get_secret_value()), timeout_s
            )
            refusal = _refusal(response, refreshed.get_secret_value())
        if refusal is not None:
            raise refusal

        latency_s = time.monotonic() - started
        try:
            payload = response.json()
        except (ValueError, json.JSONDecodeError) as exc:
            raise DecisionVendorError(
                f"{self.name} returned a body that is not JSON", kind="response"
            ) from exc
        return _parse_response(self.name, self.model_id, payload, request, latency_s)

    def _headers(self, secret: str) -> dict[str, str]:
        """The two headers every leg sends. The credential never leaves this dict."""
        return {"Authorization": f"Bearer {secret}", "Content-Type": "application/json"}

    async def _send_with(
        self,
        client: httpx.AsyncClient | None,
        body: dict[str, Any],
        headers: dict[str, str],
        timeout_s: float,
    ) -> httpx.Response:
        """Send on the injected client, or on a client made for this one call.

        The fallback exists so a leg is usable without a service around it; the
        service always injects, because a per-message TLS handshake would spend
        the whole latency budget before the vendor was even asked.
        """
        if client is not None:
            return await self._send(client, body, headers, timeout_s)
        async with httpx.AsyncClient(timeout=httpx.Timeout(timeout_s), limits=_LIMITS) as own:
            return await self._send(own, body, headers, timeout_s)

    async def _send(
        self,
        client: httpx.AsyncClient,
        body: dict[str, Any],
        headers: dict[str, str],
        timeout_s: float,
    ) -> httpx.Response:
        """One POST, with every transport failure mapped into the fall-through class.

        ``httpx.HTTPError`` covers connect/read/write timeouts and every request
        error, so the caller's timeout and a refused connection arrive here in
        the same shape — which is correct: from the cascade's point of view both
        mean "ask the next leg".
        """
        try:
            return await client.post(self.endpoint, json=body, headers=headers, timeout=timeout_s)
        except httpx.HTTPError as exc:
            raise DecisionVendorError(
                f"{self.name} transport failure: {type(exc).__name__}", kind="transport"
            ) from exc


def _parse_response(
    vendor: str,
    model: str,
    payload: Any,
    request: DecisionRequest,
    latency_s: float,
) -> DecisionResponse:
    """Turn a 200 body into a :class:`DecisionResponse`, or refuse it.

    Refusing is the point: a 200 whose ``answers`` block is missing or is not an
    object is not "no recommendations", it is a vendor that did not answer the
    question we asked, and the cascade should try the next leg.
    """
    if not isinstance(payload, Mapping):
        raise DecisionVendorError(f"{vendor} returned a non-object body", kind="response")
    raw_answers = payload.get("answers")
    if not isinstance(raw_answers, Mapping):
        raise DecisionVendorError(f"{vendor} returned no answers block", kind="response")
    answers: dict[str, Answer] = {}
    for question in request.questions:
        candidate = raw_answers.get(question.id)
        if candidate is None:
            # A question the vendor skipped is not a failure of the whole leg:
            # the caller renders what came back, and an unanswered kind simply
            # contributes no recommendations.
            continue
        answers[question.id] = _answer_from_payload(question, candidate)
    usage = payload.get("usage")
    usage_map: Mapping[str, Any] = usage if isinstance(usage, Mapping) else {}
    raw_cost = usage_map.get("cost")
    return DecisionResponse(
        vendor=vendor,
        model=str(payload.get("model") or model),
        answers=answers,
        input_tokens=_int(usage_map.get("input_tokens")),
        output_tokens=_int(usage_map.get("output_tokens")),
        # Absent cost stays absent. The contract's alternative — derive it from
        # a configured price row — has no row to read for a decision model in
        # this repo, and inventing a number would put a fabricated figure in the
        # PR's cost arithmetic.
        cost_usd=float(raw_cost) if isinstance(raw_cost, (int, float)) else None,
        latency_s=latency_s,
    )


def _int(value: Any) -> int:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return 0
    return int(value)


class RadientVendor(_HttpDecisionVendor):
    """Leg one: Radient's own route, billed to the signed-in Radient account.

    The pattern the other two copy: the OAuth session (or a pasted key) lives in
    the store, and ``RADIENT_API_KEY`` is only the static tier behind it.
    """

    name = "radient"
    endpoint = RADIENT_ENDPOINT
    default_model = RADIENT_MODEL
    provider_id = "radient"
    env_key_names = ("RADIENT_API_KEY",)


class TypeSafeVendor(_HttpDecisionVendor):
    """Leg two: TypeSafe's native endpoint, the leg that needs no proxy."""

    name = "typesafe"
    endpoint = TYPESAFE_ENDPOINT
    default_model = TYPESAFE_MODEL
    provider_id = "typesafe"
    # ``JEV_API_KEY`` is the vendor's older name for the same credential (§3); it
    # stays behind the current key so a machine carrying both uses the one the
    # registry advertises.
    env_key_names = ("TYPESAFE_API_KEY", "JEV_API_KEY")


class OpenRouterVendor(_HttpDecisionVendor):
    """Leg three: OpenRouter's alpha decisions route, the leg that is always available."""

    name = "openrouter"
    endpoint = OPENROUTER_ENDPOINT
    default_model = OPENROUTER_MODEL
    provider_id = "openrouter"
    # ``OPENROUTER_API_KEY_DEV`` is the documented second tier (§3) — a developer
    # key, deliberately behind the production one so a machine that has both
    # never spends the dev quota for real work.
    env_key_names = ("OPENROUTER_API_KEY", "OPENROUTER_API_KEY_DEV")


#: The leg classes by name, so the cascade can build a pinned leg without a
#: branch of its own and a typo in the pin is visible in one place.
VENDOR_CLASSES: dict[str, type[_HttpDecisionVendor]] = {
    RadientVendor.name: RadientVendor,
    TypeSafeVendor.name: TypeSafeVendor,
    OpenRouterVendor.name: OpenRouterVendor,
}


def build_vendor(
    name: str,
    manager: "CredentialManager",
    *,
    model: str = "",
    client: httpx.AsyncClient | None = None,
) -> _HttpDecisionVendor:
    """Construct one leg by name. Raises ``KeyError`` for an unknown name."""
    return VENDOR_CLASSES[name](manager, model=model, client=client)
