"""H3 — the one-shot screener: pure render/parse plus the shared pipeline.

WHAT LIVES HERE. Deterministic pieces testable without a provider, and the
ONE pipeline that turns a fetched message into a metadata-only outcome:

1. :func:`render_prompt` — single-pass placeholder substitution over the
   FROZEN embedded template (``bridge-screen-v1``/``v2``, selected by the served
   version). A fresh cryptographic
   nonce delimits the submission; substituted values are never re-scanned,
   so text inserted from the (hostile) message cannot forge a closing tag or
   grow a new placeholder. The rendered text is a PRIVATE carrier: it is not
   a dataclass field, so neither ``repr`` nor ``dataclasses.asdict`` nor any
   generic serializer can leak the message body (H3-Q7).
2. :func:`parse_verdict` — the frozen two-line reply grammar, strict.
   Anything that is not exactly ``VERDICT: deliver|hold|bounce`` then one
   nonempty ``REASON:`` line (<=200 chars) holds: extra prose, markdown,
   missing/duplicate fields, tool calls, lone surrogates, unknown categories
   on a bounce.
3. :func:`screen_message` — the shared pipeline: exact-owner generation,
   frozen template byte-equality, envelope/participant validation and
   per-type caps BEFORE any spend, H2 intent reconciliation (never a fresh
   admission), fingerprint cache keyed on the FULL admission tuple (body +
   template + model + revision + envelope digest), durable logical budget
   reservation BEFORE inference I/O, the single model dispatch, strict
   parse, and a WRITE-ONCE metadata-only outcome fenced on the published
   policy epoch inside the store's own transaction.

CONTENT LAW. The model reply and the message body are ephemeral: they may
exist in memory for one call and are never written to the store, logs,
analytics, receipts or error text. Durable records carry only hashes, closed
enums, model identity, counts and timestamps. Both carry carriers here
(:class:`RenderedPrompt` text, :class:`ScreenResult` reason) are private
attributes with accessor methods, deliberately NOT fields.

CONTROL LAW. A ``deliver``/allow is evidence for the H2 ack path only;
working/done, approvals, delivery and any H4 connector remain other lanes'
work. Budget exhaustion, breaker-open, unknown routes and any local refusal
are HOLDS, never silent resets. One logical screening may reach up to three
physical requests (auth re-resolve + fast resend); the wire decorator
(``reserve_http_attempt``) charges each of those separately.
"""

from __future__ import annotations

import asyncio
import hashlib
import inspect
import json
import re
import secrets
import weakref as _weakref
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Mapping, Protocol

from local_operator.team_bridge.client import (
    SCREENING_TEMPLATES,
    SCREENING_VERSION_V1,
    screening_template_sha256,
)
from local_operator.team_bridge.store import Store, StoreRefused
from local_operator.team_bridge.types import (
    DELIVERY_STATES,
    SCREENING_CATEGORIES,
    OwnerSnapshot,
    ScreeningReceipt,
)

__all__ = [
    "HOLD_CODES",
    "Envelope",
    "PolicySnapshot",
    "ScreenResult",
    "ScreenRunner",
    "parse_verdict",
    "render_prompt",
    "screen_message",
    "validate_envelope",
]

#: The closed category enum (frozen §8) is imported from :mod:`....types`, so the
#: persisted-proof validator uses the same vocabulary.

#: The closed hold-code set. Every non-admission is ONE of these; none of
#: them ever carries model/vendor prose.
HOLD_CODES = (
    "owner_mismatch",
    "owner_changed",
    "template_mismatch",
    "body_hash_mismatch",
    "envelope_unbound",
    "envelope_malformed",
    "invalid_content",
    "invalid_content_type",
    "size_exceeded",
    "ack_recovery",
    "budget_exhausted",
    "breaker_open",
    "runner_unavailable",
    "provider_error",
    "model_identity_unknown",
    "model_identity_mismatch",
    "parse_failed",
    "bounce_without_category",
    "model_hold",
    "policy_stale",
    "local_quarantine",
    "retry_later",
    "write_once_conflict",
    "schema_damaged",
    # Not a screening outcome: the connector's wait for the owner to SEND a
    # message to a never-used chief-of-staff session whose accepted consent does not allow a
    # card to engage it (team_bridge.engagement). Listed so the owner's surface
    # renders the specific reason rather than ``unknown_hold``.
    "open_chief_of_staff",
)

#: Per-type RAW utf8 caps on the fetched body (§8.3 screening inputs).
_TYPE_CAPS = {"note": 16 * 1024, "question": 16 * 1024, "request": 64 * 1024}
#: Bound on each scalar a request block may carry, and on their count.
_REQUEST_FIELD_MAX = 2000
_REQUEST_FIELDS_MAX = 24
#: Bounds on server-resolved display metadata (sender id / org label).
_ENVELOPE_LABEL_MAX = 200
#: Bound on the rendered type line (type + canonical request metadata).
_TYPE_LABEL_MAX = 2000

_MAX_REASON_CHARS = 200
_NONCE_BYTES = 16
_VERDICT_LINE = re.compile(r"^VERDICT: (deliver|hold|bounce)$")
_REASON_LINE = re.compile(r"^REASON: (\S.*)$")
_PLACEHOLDER = re.compile(r"\{\{(nonce|body|sender|organization|type)\}\}")


class ScreeningPolicy(Protocol):
    """Trusted LOCAL plumbing — never agent- or server-supplied policy.

    ``screening_version``/``template_sha256`` come from the caller's current
    accepted status metadata (H1's monotonic, capped, byte-equality rules);
    ``default_model`` is the exact user-configured default resolution (see
    the H3 handoff: NOT an active CoS override, not a classifier tier, not a
    different-model fallback); ``admission_revision`` names the trusted
    publisher's current revision of that configuration. The pipeline
    SNAPSHOTS these at entry and re-reads them after the awaited work, so a
    mutation during a run cannot be laundered into an admission (H3-Q3).
    """

    screening_version: str
    template_sha256: str
    default_model: str | None
    admission_revision: str


@dataclass(frozen=True)
class PolicySnapshot:
    """The IMMUTABLE admission policy captured before any model work.

    ``approved_serving_identities`` is how an explicitly approved alias
    resolution crosses: the configured selector is the capture, and the
    resolver's APPROVED concrete serving identities are listed so a resolved
    versioned id is not rejected merely for differing spelling (R4). An
    unlisted identity still holds — never guess a fallback.
    """

    screening_version: str
    template_sha256: str
    default_model: str | None
    admission_revision: str
    approved_serving_identities: frozenset[str] = field(default_factory=frozenset)
    #: Canonical configured provider captured ONCE from the trusted policy
    #: source (resolver/config + accepted status). Carried through the
    #: snapshot so the job mint never re-resolves a second, later config
    #: (R7/Q3: one capture certifies the dispatch).
    provider: str = ""

    @classmethod
    def capture(cls, policy: ScreeningPolicy) -> "PolicySnapshot":
        identities = getattr(policy, "approved_serving_identities", ()) or ()
        provider = getattr(policy, "provider", "") or ""
        return cls(
            screening_version=str(getattr(policy, "screening_version", "")),
            template_sha256=str(getattr(policy, "template_sha256", "")),
            default_model=(policy.default_model if isinstance(policy.default_model, str) else None),
            admission_revision=str(getattr(policy, "admission_revision", "")),
            approved_serving_identities=frozenset(str(i) for i in identities),
            provider=provider,
        )

    def matches_live(self, policy: ScreeningPolicy) -> bool:
        """Re-read the LIVE policy object and compare every admission field."""

        return self == PolicySnapshot.capture(policy)

    def accepts_serving_identity(self, actual_model: str) -> bool:
        return actual_model == self.default_model or (
            actual_model in self.approved_serving_identities
        )


@dataclass
class RenderedPrompt:
    """One rendered screening request. The text is PRIVATE (accessor only).

    ``text`` is deliberately NOT a field: ``repr()`` and every generic
    serializer (``dataclasses.asdict``, ``json``) see only the nonce and the
    hashes, never the body (H3-Q7). Use :meth:`private_text` for the single
    dispatch and drop the reference.
    """

    nonce: str = field(repr=False)
    template_sha256: str
    body_sha256: str
    screened_bytes: int

    def private_text(self) -> str:
        return self._private_text  # type: ignore[attr-defined]


@dataclass
class ScreenResult:
    """The pipeline's outcome. The reason is PRIVATE (accessor only).

    ``ephemeral_reason`` is deliberately NOT a field for the same reason as
    :class:`RenderedPrompt`: it must never ride ``asdict``/JSON/tool results.
    """

    disposition: str  # deliver | bounce | hold
    receipt: ScreeningReceipt | None = None
    hold_code: str | None = None
    retry_at: float | None = None
    screened_bytes: int | None = None

    def __post_init__(self) -> None:
        self._private_reason: str | None = None

    def private_reason(self) -> str | None:
        return self._private_reason

    def set_private_reason(self, reason: str | None) -> "ScreenResult":
        self._private_reason = reason
        return self


class ScreenRunner(Protocol):
    """One model dispatch, injected. Returns ``(reply_text, actual_model)``.

    The runner is the WHOLE provider boundary for the pipeline: production
    supplies an adapter over the harness stream function (controlled QA
    supplies a scripted callable). It must return the actual serving model
    identity or ``None`` when it cannot be evidenced — never a guess. Errors
    it raises are sanitized here into typed holds; nothing of its prose is
    persisted.
    """

    def __call__(self, prompt: str, *, default_model: str) -> tuple[str, str | None]: ...


def _body_sha256_bytes(body_bytes: bytes) -> str:
    return hashlib.sha256(body_bytes).hexdigest()


@dataclass(frozen=True)
class Envelope:
    """Validated, bounded admission input derived from the fetched item.

    Everything here is inert DATA: the digest binds the envelope to the
    admission so a same-body message with a different sender/type/request
    cannot reuse another context's verdict.
    """

    message_id: str
    body_sha256: str
    message_type: str
    #: The bounded value the CLASSIFIER sees for the type line: the closed
    #: type plus (for request messages) canonical request metadata, so the
    #: required request fields reach the model input, not merely the digest.
    type_label: str
    sender_account: str
    sender_display: str
    organization: str
    delivery_state: str | None
    envelope_digest: str

    def private_body(self) -> str:
        """The raw body: a PRIVATE carrier, never a dataclass field (H3-Q7).

        Kept off ``repr``/``asdict``/JSON so no generic serializer can turn
        an Envelope into an unscreened body export; the single dispatch is
        its only reader.
        """

        return self._private_body  # type: ignore[attr-defined]


def _canonicalize_request(request: Mapping[str, Any]) -> dict[str, Any] | None:
    """Bound the request block to closed scalar metadata; None if malformed."""

    if len(request) > _REQUEST_FIELDS_MAX:
        return None
    canonical: dict[str, Any] = {}
    for key, value in request.items():
        if not isinstance(key, str) or not key or len(key) > 64:
            return None
        if isinstance(value, bool) or not isinstance(value, (str, int, float)):
            return None
        if isinstance(value, str) and len(value) > _REQUEST_FIELD_MAX:
            return None
        canonical[key] = value
    if not canonical:
        return None
    return canonical


def validate_envelope(
    owner: OwnerSnapshot, item: Mapping[str, Any]
) -> tuple[Envelope | None, str | None]:
    """Authenticate + bound the fetched item BEFORE any spend.

    Checks, in order: message identity/type against the closed type set;
    the body's exact utf8 bytes and hash; per-type raw size cap; thread and
    delivery bindings against the EXACT owner when the server resolved them
    (absent blocks are allowed — the canonical H2 DTO may omit them — but a
    PRESENT binding that names another tenant/recipient is a hold, never an
    admission); and the request block for request-type messages. Malformed
    Unicode is a typed, content-free hold (H3-Q9).
    """

    raw_message = item.get("message")
    message: Mapping[str, Any] = raw_message if isinstance(raw_message, Mapping) else {}
    message_id = message.get("id")
    if not isinstance(message_id, str) or not message_id:
        return None, "envelope_malformed"
    message_type = message.get("type")
    if not isinstance(message_type, str) or message_type not in _TYPE_CAPS:
        return None, "invalid_content_type"

    raw_body = message.get("body")
    body: Mapping[str, Any] = raw_body if isinstance(raw_body, Mapping) else {}
    body_text = body.get("text")
    if not isinstance(body_text, str):
        return None, "envelope_malformed"
    try:
        body_bytes = body_text.encode("utf-8")
    except UnicodeEncodeError:
        return None, "invalid_content"
    if len(body_bytes) > _TYPE_CAPS[message_type]:
        return None, "size_exceeded"
    body_sha = _body_sha256_bytes(body_bytes)
    served_hash = message.get("body_hash")
    if not isinstance(served_hash, str) or served_hash != body_sha:
        return None, "body_hash_mismatch"

    raw_thread = item.get("thread")
    thread: Mapping[str, Any] = raw_thread if isinstance(raw_thread, Mapping) else {}
    thread_tenant = thread.get("tenant_id", message.get("tenant_id"))
    if thread_tenant is not None and thread_tenant != owner.tenant_id:
        return None, "envelope_unbound"
    # Delivery (canonical H2 DTO: state/state_at live here; the recipient is
    # message.recipient — a delivery-nested recipient is only a fallback). A
    # PRESENT delivery block must carry a closed state; missing/null/wrong
    # state fails closed (A6). An ABSENT block is a legacy/minimal envelope.
    raw_delivery = item.get("delivery")
    delivery: Mapping[str, Any] = raw_delivery if isinstance(raw_delivery, Mapping) else {}
    delivery_state: str | None = None
    if "delivery" in item:
        state = delivery.get("state")
        if not isinstance(state, str) or state not in DELIVERY_STATES:
            return None, "envelope_malformed"
        delivery_state = state
    raw_recipient = message.get("recipient")
    if raw_recipient is None:
        raw_recipient = delivery.get("recipient")
    if isinstance(raw_recipient, Mapping):
        if raw_recipient.get("account_id") != owner.account_id:
            return None, "envelope_unbound"
        endpoint = raw_recipient.get("endpoint_id")
        if endpoint is not None and endpoint != owner.endpoint_id:
            return None, "envelope_unbound"

    raw_sender = item.get("sender")
    sender: Mapping[str, Any] = raw_sender if isinstance(raw_sender, Mapping) else {}
    sender_account = sender.get("account_id")
    if sender_account is None:
        # An ABSENT sender block is UNKNOWN provenance, not false provenance:
        # the participant bindings that matter (thread tenant, recipient) are
        # checked independently. A PRESENT but malformed sender still refuses.
        sender_account = ""
    if not isinstance(sender_account, str) or len(sender_account) > _ENVELOPE_LABEL_MAX:
        return None, "envelope_malformed"
    sender_name = sender.get("name")
    if sender_name is None or sender_name == "":
        sender_display = sender_account
    elif isinstance(sender_name, str) and len(sender_name) <= _ENVELOPE_LABEL_MAX:
        sender_display = sender_name
    else:
        return None, "envelope_malformed"

    organization = message.get("organization", sender.get("organization"))
    if not isinstance(organization, str) or not organization:
        organization = owner.tenant_id
    if len(organization) > _ENVELOPE_LABEL_MAX:
        return None, "envelope_malformed"

    # A request block is REQUIRED for request-type messages and included in
    # the admission digest WHENEVER present: the digest must cover every
    # envelope input the classifier can see, so a changed request
    # specification can never reuse another context's cached verdict
    # (H3-Q2's request-spec case; a stricter fingerprint only ever misses).
    request_canonical: dict[str, Any] | None = None
    raw_request = message.get("request")
    if message_type == "request" and not isinstance(raw_request, Mapping):
        return None, "envelope_malformed"
    if isinstance(raw_request, Mapping):
        request_canonical = _canonicalize_request(raw_request)
        if request_canonical is None:
            return None, "envelope_malformed"
    type_label = message_type
    if request_canonical is not None:
        # The classifier must SEE the request metadata: the frozen template
        # has no request placeholder, so the bounded canonical form rides in
        # the Type line (inert data, single-pass rendered like every value).
        type_label = (
            message_type
            + " request="
            + json.dumps(request_canonical, sort_keys=True, separators=(",", ":"))
        )
        if len(type_label) > _TYPE_LABEL_MAX:
            return None, "size_exceeded"

    digest_payload = json.dumps(
        {
            "sender": sender_account,
            # RENDERED identity is part of the admission digest (A6): a
            # display-name change must miss the cache, not ride another
            # context's verdict.
            "sender_display": sender_display,
            "organization": organization,
            "type": message_type,
            "request": request_canonical,
        },
        sort_keys=True,
        separators=(",", ":"),
    )
    envelope = Envelope(
        message_id=message_id,
        body_sha256=body_sha,
        message_type=message_type,
        type_label=type_label,
        sender_account=sender_account,
        sender_display=sender_display,
        organization=organization,
        delivery_state=delivery_state,
        envelope_digest=hashlib.sha256(digest_payload.encode("utf-8")).hexdigest(),
    )
    # Private carrier: never a field, so repr/asdict/JSON stay content-free.
    object.__setattr__(envelope, "_private_body", body_text)
    return envelope, None


def render_prompt(
    *,
    body_text: str,
    sender: str,
    organization: str,
    msg_type: str,
    nonce: str | None = None,
    version: str = SCREENING_VERSION_V1,
) -> RenderedPrompt:
    """Single-pass substitution over an EMBEDDED template — the ONLY renderer.

    ``version`` selects the bytes from :data:`SCREENING_TEMPLATES`; a version
    this client does not embed is refused. The served status value is only a
    selector (``_screening_is_trusted``), never content, so a compromised status
    response cannot rewrite the screener.

    The nonce must be absent from every substituted value and every value is
    inserted ONCE, left-to-right, without re-scanning: a message body that
    contains ``{{nonce}}`` (or an imitated closing tag) stays inert data.
    The template's three nonce occurrences are asserted, so a future template
    edit cannot silently change the delimiter contract.
    """

    template = SCREENING_TEMPLATES.get(version)
    if template is None:
        raise ValueError(f"no embedded screening template for version {version!r}")
    token = nonce if nonce is not None else secrets.token_hex(_NONCE_BYTES)
    values = {
        "sender": sender,
        "organization": organization,
        "type": msg_type,
        "body": body_text,
    }
    for name, value in values.items():
        if token in value:
            # A forged nonce literal inside untrusted data would let the
            # submission frame itself; regenerate is the caller's job, so
            # this renderer refuses outright (fail closed, never sanitize).
            raise ValueError(f"nonce literal present in substituted {name!r}")
    if template.count("{{nonce}}") != 3 or template.count("{{body}}") != 1:
        raise ValueError("embedded template delimiter contract changed")
    # SINGLE pass: scan the TEMPLATE only; inserted values are appended
    # verbatim and never rescanned (no recursive placeholder expansion).
    parts: list[str] = []
    cursor = 0
    for match in _PLACEHOLDER.finditer(template):
        parts.append(template[cursor : match.start()])
        name = match.group(1)
        parts.append(token if name == "nonce" else values[name])
        cursor = match.end()
    parts.append(template[cursor:])
    text = "".join(parts)
    if text.count(token) != 3:
        raise ValueError("rendered prompt lost its nonce delimiters")
    template_sha256 = screening_template_sha256(version)
    if template_sha256 is None:  # unreachable: the template map selected it
        raise ValueError(f"no trusted hash for screening version {version!r}")
    rendered = RenderedPrompt(
        nonce=token,
        template_sha256=template_sha256,
        body_sha256=_body_sha256_bytes(body_text.encode("utf-8")),
        screened_bytes=len(body_text.encode("utf-8")),
    )
    # Private carrier: set outside the field set so no generic serializer
    # can reach the body text (H3-Q7).
    rendered.__dict__["_private_text"] = text
    return rendered


def parse_verdict(reply: str) -> tuple[str, tuple[str, ...], str, str] | None:
    """Frozen two-line grammar → ``(disposition, categories, reason, verdict)``.

    ``None`` means the reply is NOT a valid screening answer and must hold.
    The reason is returned for the caller's ephemeral use only.
    """

    if not isinstance(reply, str):
        return None
    if any(0xD800 <= ord(character) <= 0xDFFF for character in reply):
        # A lone surrogate could not round-trip to UTF-8 storage; treating it
        # as a valid verdict would smuggle malformed content downstream.
        return None
    # Exactly two lines: strip the ONE conventional trailing terminator, but
    # an extra blank line is extra output and must reject (strict grammar).
    text = reply[:-1] if reply.endswith("\n") else reply
    if "\r" in text:
        return None
    # Trailing spaces/tabs per line are tolerated, nothing else (live E2E on
    # local-operator-ui #1314): chat models end lines with a markdown hard break
    # ("VERDICT: deliver␠␠"), which carries no content but held every real
    # screen as ``parse_failed``. Leading text, extra lines and CR stay refused.
    lines = [line.rstrip(" \t") for line in text.split("\n")]
    if len(lines) != 2:
        return None
    verdict_match = _VERDICT_LINE.match(lines[0])
    reason_match = _REASON_LINE.match(lines[1])
    if verdict_match is None or reason_match is None:
        return None
    verdict = verdict_match.group(1)
    reason = reason_match.group(1)
    if not reason or len(reason) > _MAX_REASON_CHARS:
        return None
    if verdict == "deliver":
        return ("deliver", (), reason, verdict)
    if verdict == "hold":
        return ("hold", (), reason, verdict)
    # bounce: categories are extracted ONLY as exact known tokens, canonical
    # order, de-duplicated; none unambiguous -> the caller holds.
    categories = tuple(
        category
        for category in SCREENING_CATEGORIES
        if re.search(rf"(?<![A-Za-z_]){re.escape(category)}(?![A-Za-z_])", reason)
    )
    return ("bounce", categories, reason, verdict)


_HOLD_FOR_SCHEMA = {
    "schema_damaged": "schema_damaged",
    "schema_unversioned": "schema_damaged",
    "schema_unsupported": "schema_damaged",
    "state_unreadable": "schema_damaged",
}
_HOLD_FOR_COMMIT = {
    "stale_generation": "owner_changed",
    "not_enrolled": "owner_changed",
    "busy": "retry_later",
    "write_once_conflict": "write_once_conflict",
    "attempt_mismatch": "retry_later",
    "policy_stale": "policy_stale",
    "unknown_attempt": "retry_later",
}


def _persist_hold(
    store: Store, owner: OwnerSnapshot, message_id: str, code: str, retry_at: float | None = None
) -> None:
    """Best-effort durable hold for a VALID live owner; never raises."""

    try:
        store.persist_hold(owner, message_id, code, retry_at=retry_at)
    except StoreRefused:
        # An owner that went stale between validation and the hold write
        # simply gets no row: durable state must never hold another
        # generation's data, and H4 reconciles from the ack path.
        pass


#: Per-account dispatch locks, keyed by (state db path, HUB, account id) —
#: the SAME account scope as the native turn and the aggregate budget, and
#: held by LOOP lifetime (a WeakKeyDictionary keyed by the running loop): a
#: lock is an asyncio primitive bound to the loop that contended it, so a
#: permanent module-global map hands a dead loop's lock to a fresh loop and
#: raises "bound to a different event loop" (QA's actual two-loop control).
#: Cross-call/cross-process exclusion is the NATIVE account turn; this lock
#: only serializes same-loop contenders so they queue rather than race the
#: claim.
#: Fallback lock table for loops that refuse attribute assignment (rare):
#: kept OUT of the default path because a WeakKeyDictionary keyed by the
#: loop can never collect — each value (dict -> asyncio.Lock -> waiter
#: futures) holds a STRONG reference back to its own key (R6/Q15).
#: Renewal cadence + lifetime for the in-flight dispatch keeper (R7/Q15-5):
#: the store's TTL floor is 60s, so a 20s cadence keeps a long callback ahead
#: of expiry while an ownership loss is detected within one interval.
_TURN_KEEPALIVE_S = 20.0
_TURN_KEEPALIVE_TTL = 90.0

_ACCOUNT_LOCKS: "_weakref.WeakKeyDictionary[Any, dict[tuple[str, str, str], asyncio.Lock]]"
_ACCOUNT_LOCKS = _weakref.WeakKeyDictionary()


def _account_lock(config_dir: Path, owner: OwnerSnapshot) -> asyncio.Lock:
    """Per-(config, hub, account) dispatch lock, owned BY THE LOOP.

    The table is attached to the running loop as an ordinary attribute, so
    it is a plain object graph that becomes unreachable (and collectible)
    when the loop does; contention safety and two-loop reuse are both
    preserved (R6/Q15). No cross-loop lock object is ever shared.
    """
    loop = asyncio.get_running_loop()
    per_loop = getattr(loop, "_team_bridge_account_locks", None)
    if per_loop is None:
        per_loop = {}
        try:
            loop._team_bridge_account_locks = per_loop  # type: ignore[attr-defined]
        except AttributeError:  # pragma: no cover — exotic loop classes only
            per_loop = _ACCOUNT_LOCKS.get(loop)
            if per_loop is None:
                per_loop = {}
                _ACCOUNT_LOCKS[loop] = per_loop
    key = (str(config_dir), owner.canonical_base_url, owner.account_id)
    lock = per_loop.get(key)
    if lock is None:
        lock = asyncio.Lock()
        per_loop[key] = lock
    return lock


async def screen_message_async(
    owner: OwnerSnapshot,
    item: Mapping[str, Any],
    *,
    config_dir: Path,
    policy: ScreeningPolicy,
    runner: ScreenRunner | None = None,
    nonce: str | None = None,
) -> ScreenResult:
    """One account's serialized screening entry (ONE in flight per account).

    Concurrent callers for the same account wait on the in-process lock —
    the cross-process equivalent is the native screen_turn lease; the frozen
    contract allows exactly one screening in flight per account, and a
    parallel screening amplifier is a defect, not a feature.
    """

    import os as _os
    import time as _time

    from local_operator.team_bridge import scheduler as _scheduler

    async def _close_owned_once() -> None:
        # Idempotent close of whatever THIS call owns (the runner): used on
        # every early path so no prebuilt owned client survives a lock wait,
        # a claim refusal or a cancelled wait (R6/Q10); the lazy production
        # path has nothing built yet and this is a no-op.
        _close = getattr(runner, "aclose", None)
        if callable(_close):
            _res = _close()
            if inspect.isawaitable(_res):
                await _res

    _lock = _account_lock(config_dir, owner)
    try:
        await _lock.acquire()
    except BaseException:
        # Cancelled/failed while WAITING for the account lock: close what
        # this call owns before propagating (R6/Q10).
        await _close_owned_once()
        raise
    try:
        store = Store(config_dir)
        scope = f"{owner.canonical_base_url}|{owner.account_id}"
        existing = _scheduler.current_turn()
        turn: str | None = None
        reuse = False
        if (
            existing is not None
            and existing[0] == scope
            and store.screen_turn_matches(owner, turn=existing[1], now=_time.time())
        ):
            # A cycle (or enclosing scope) already holds THIS account's turn:
            # reuse the exact token, never claim a competing second turn.
            turn = existing[1]
            reuse = True
        if turn is None:
            # The native account turn is INSEPARABLE from admitted dispatch
            # (R5): without it, another worker/process owns this account and
            # a direct public call must HOLD, not race past the lease.
            try:
                turn = store.claim_screen_turn(
                    owner, holder_prefix=f"pid{_os.getpid()}", now=_time.time(), ttl=90.0
                )
            except StoreRefused as failure:
                # A damaged/unreadable schedule store fails CLOSED as a typed
                # hold — never an uncaught refusal escaping the public entry.
                await _close_owned_once()
                return ScreenResult(
                    "hold", hold_code=_HOLD_FOR_SCHEMA.get(failure.code, "retry_later")
                )
        if turn is None:
            await _close_owned_once()
            return ScreenResult("hold", hold_code="retry_later")
        try:
            return await _screen_core(
                owner,
                item,
                config_dir=config_dir,
                policy=policy,
                runner=runner,
                nonce=nonce,
                turn=turn,
            )
        finally:
            # Whole-ENTRY lifetime (R5/R6): the runner's owned client/stream
            # fn close on success, every hold, deadline and cancellation; the
            # exact owned turn releases even if the close RAISES (release is
            # the nested finally, never skipped).
            try:
                await _close_owned_once()
            finally:
                if not reuse:
                    store.release_screen_turn(owner, turn=turn)

    except BaseException:
        # Setup failures — a damaged schedule store refusing inside
        # screen_turn_matches, a claim refusal raised as an exception — can
        # occur BEFORE _screen_core's own finally; owned resources close on
        # ANY setup exception too (R7/Q15-7). The close is idempotent.
        await _close_owned_once()
        raise
    finally:
        _lock.release()


async def _screen_core(
    owner: OwnerSnapshot,
    item: Mapping[str, Any],
    *,
    config_dir: Path,
    policy: ScreeningPolicy,
    runner: ScreenRunner | None = None,
    nonce: str | None = None,
    turn: str = "",
) -> ScreenResult:
    """The ONE screening pipeline. At most one runner dispatch per call.

    Order is the contract: owner authority → template floor → envelope
    validation (all pre-spend) → policy publish/epoch → H2 intent
    reconciliation (never a fresh admission, never a rebill) → full-fingerprint
    cache → breaker → availability → durable LOGICAL reservation → render →
    dispatch → live-policy recheck → identity → strict parse → WRITE-ONCE
    outcome fenced on the published epoch. Every refusal is a hold with a
    closed code; nothing durable ever carries body text, reasons or replies.
    """

    store = Store(config_dir)

    # 1. Exact owner + latest committed generation (native guard). Schema
    #    damage must surface as its own typed hold, never as a provider
    #    fault and never as a silent repair.
    try:
        committed = store.resolve_owner(
            credential_id=owner.credential_id,
            account_id=owner.account_id,
            tenant_id=owner.tenant_id,
            endpoint_id=owner.endpoint_id,
            canonical_base_url=owner.canonical_base_url,
        )
    except StoreRefused as failure:
        return ScreenResult("hold", hold_code=_HOLD_FOR_SCHEMA.get(failure.code, "owner_mismatch"))
    if committed is None or committed != owner:
        # A damaged/unknown store also answers None (no authority); name that
        # reason so the hold is typed and actionable rather than a false
        # "owner" claim (H3-Q4/Q8: never repair, but never mislabel either).
        if committed is None and store.schema_problem() is not None:
            return ScreenResult("hold", hold_code="schema_damaged")
        return ScreenResult("hold", hold_code="owner_mismatch")

    # 2. Template authority: an EMBEDDED version and its byte-equal hash,
    #    checked against an IMMUTABLE snapshot so nothing mutated later can
    #    launder in. v2 adds the scope categories; v1 stays accepted so a
    #    deployment still serving it keeps working.
    snapshot = PolicySnapshot.capture(policy)
    trusted_hash = screening_template_sha256(snapshot.screening_version)
    if trusted_hash is None or snapshot.template_sha256 != trusted_hash:
        # The owner resolved live, so this early refusal becomes a durable
        # hold during the message's lifetime (H3-Q8), separate from the
        # inbox projection's own state.
        raw_id = item.get("message")
        held_id = raw_id.get("id") if isinstance(raw_id, Mapping) else None
        if isinstance(held_id, str) and held_id:
            _persist_hold(store, owner, held_id, "template_mismatch")
        return ScreenResult("hold", hold_code="template_mismatch")

    # 3. Envelope validation before any store write or spend (H3-Q9).
    envelope, bad_code = validate_envelope(owner, item)
    if envelope is None:
        # A None/str message block must reach persist_hold as a TYPED hold
        # with a bounded placeholder id — never an AttributeError (R6/Q9).
        message_block = item.get("message")
        held_id = "unparsed"
        if isinstance(message_block, Mapping):
            candidate_id = message_block.get("id")
            if isinstance(candidate_id, str) and candidate_id:
                held_id = candidate_id
        _persist_hold(store, owner, held_id, bad_code or "envelope_malformed")
        return ScreenResult("hold", hold_code=bad_code or "envelope_malformed")
    message_id = envelope.message_id

    # 4. CONSUME the published admission authority — the pipeline is never
    #    its own publisher (A2/A3): a row is bootstrapped from the caller
    #    ONLY when none exists (first screening for this namespace); once a
    #    row exists, the caller must EQUAL it, so a stale retained caller
    #    can neither screen against it nor rewind it (H3-Q3).
    try:
        current = store.current_screening_policy(owner)
        if current is None:
            try:
                epoch = store.publish_screening_policy(
                    owner,
                    version=snapshot.screening_version,
                    template_sha256=snapshot.template_sha256,
                    default_model=snapshot.default_model,
                    revision=snapshot.admission_revision,
                    create_only=True,
                )
            except StoreRefused as race:
                if race.code != "policy_exists":
                    raise
                current = store.current_screening_policy(owner)
                if current is None or not _policy_row_matches(current, snapshot):
                    _persist_hold(store, owner, message_id, "policy_stale")
                    return ScreenResult("hold", hold_code="policy_stale")
                epoch = int(current["epoch"])
        else:
            if not _policy_row_matches(current, snapshot):
                _persist_hold(store, owner, message_id, "policy_stale")
                return ScreenResult("hold", hold_code="policy_stale")
            epoch = int(current["epoch"])
    except StoreRefused as failure:
        code = _HOLD_FOR_SCHEMA.get(failure.code, "owner_changed")
        return ScreenResult("hold", hold_code=code)

    # 5. Reconcile H2's immutable tuple FIRST: an existing intent is NEVER a
    #    fresh admission (H3-Q1/A3). Exact replays reconcile through the ack
    #    path; this pipeline just refuses to bill or admit behind its back.
    existing = store.find_pending_intent(owner, message_id, "read")
    if existing is not None:
        receipt_data = existing["receipt"]
        if receipt_data is None:
            _persist_hold(store, owner, message_id, "ack_recovery")
            return ScreenResult("hold", hold_code="ack_recovery")
        if receipt_data.get("body_sha256") != envelope.body_sha256:
            _persist_hold(store, owner, message_id, "body_hash_mismatch")
            return ScreenResult("hold", hold_code="body_hash_mismatch")
        _persist_hold(store, owner, message_id, "ack_recovery")
        return ScreenResult("hold", hold_code="ack_recovery")

    # 6. Fingerprint cache: reuse ONLY under the FULL current admission tuple
    #    (body + template + model + revision + envelope digest). A changed
    #    sender/type/request/revision misses and re-screens; a reject is
    #    reused as the same bounce rather than silently re-billed (H3-Q2).
    fingerprint = (
        f"{snapshot.admission_revision}|{envelope.envelope_digest}"
        f"|{_identities_digest(snapshot.approved_serving_identities)}"
    )
    if snapshot.default_model is not None:
        # ONE atomic native consumption (A2/A3): owner generation, current
        # published policy, receipt epoch and creator generation all verified
        # inside a single transaction; the live object is re-read LAST.
        try:
            cached = store.consume_screening_receipt(
                owner,
                body_sha256=envelope.body_sha256,
                template_sha256=snapshot.template_sha256,
                model=snapshot.default_model,
                admission_revision=snapshot.admission_revision,
                envelope_digest=envelope.envelope_digest,
            )
        except StoreRefused:
            # The consumption could not be LINEARIZED (an interleaved writer
            # or damaged store left the authority unverified): this is a
            # refusal exactly like a stale policy — never a silent cache
            # miss that lets a stale caller re-screen behind the change.
            _persist_hold(store, owner, message_id, "policy_stale")
            return ScreenResult("hold", hold_code="policy_stale")
        if cached is not None and not snapshot.matches_live(policy):
            _persist_hold(store, owner, message_id, "policy_stale")
            return ScreenResult("hold", hold_code="policy_stale")
        if cached is not None and cached["verdict"] == "reject":
            # A cached reject must respect the SAME historical-read rule as
            # a fresh one (A3): with a confirmed upstream read this stays a
            # LOCAL hold, never a bounce replay.
            if store.find_intent_state(owner, message_id, "read") == "confirmed":
                _persist_hold(store, owner, message_id, "local_quarantine")
                return ScreenResult("hold", hold_code="local_quarantine")
        if cached is not None:
            receipt = ScreeningReceipt(
                verdict=cached["verdict"],
                model=cached["model"],
                prompt_version=cached["prompt_version"],
                template_sha256=cached["template_sha256"],
                body_sha256=cached["body_sha256"],
                categories=tuple(cached["categories"]),
            )
            disposition = "deliver" if cached["verdict"] == "allow" else "bounce"
            return ScreenResult(
                disposition,
                receipt=receipt,
                screened_bytes=len(envelope.private_body().encode("utf-8")),
            )

    # 7. Account breaker: an open breaker is a hold with its own retry time,
    #    BEFORE any reservation (no spend while the route is known-broken).
    breaker = store.screening_breaker_state(owner)
    import time as _time

    now = _time.time()
    if breaker is not None and float(breaker.get("open_until", 0) or 0) > now:
        _persist_hold(
            store, owner, message_id, "breaker_open", retry_at=float(breaker["open_until"])
        )
        return ScreenResult("hold", hold_code="breaker_open", retry_at=float(breaker["open_until"]))

    # 8. Availability before the reservation: a known-missing runner or an
    #    unusable default is a durable hold, never a spent budget unit
    #    (H3-Q8 — no charge for a dispatch that provably cannot happen).
    if snapshot.default_model is None or runner is None:
        _persist_hold(store, owner, message_id, "runner_unavailable")
        return ScreenResult("hold", hold_code="runner_unavailable")

    # Canonical provider (R6/R8/Q3): the job must carry a REAL configured
    # provider, never an empty string from a fixture snapshot. The trusted
    # source is the same resolver the production factory uses — the
    # operative config — with a snapshot override when the published
    # authority supplies one (the ONE-capture snapshot does). No
    # caller-supplied strings certify here. Derived BEFORE the reservation
    # so the registered input proof pins the same value the job will carry.
    _canonical_provider = getattr(snapshot, "provider", None) or ""
    if not _canonical_provider:
        from local_operator.team_bridge.model_boundary import resolve_default_selection

        _resolved = resolve_default_selection(config_dir)
        _canonical_provider = _resolved.provider if _resolved is not None else ""

    # 8b. IMMUTABLE INPUT PROOF (R8/Q3): every canonical field of this
    #     dispatch is registered at reservation — owner namespace/generation,
    #     body/template/version, provider/model/approved map, envelope,
    #     request, display digests and revision/epoch — so the ONE bind can
    #     compare the DispatchJob against the record BEFORE any I/O and
    #     refuse a field mutated in between.
    input_proof = {
        "owner": (
            f"{owner.credential_id}|{owner.account_id}|{owner.tenant_id}"
            f"|{owner.endpoint_id}|{owner.canonical_base_url}"
            f"|{owner.enrollment_generation}"
        ),
        "body": envelope.body_sha256,
        "template": snapshot.template_sha256,
        "version": snapshot.screening_version,
        "provider": _canonical_provider,
        "model": snapshot.default_model or "",
        "approved": sorted(snapshot.approved_serving_identities),
        "envelope": envelope.envelope_digest,
        "request": hashlib.sha256(envelope.type_label.encode("utf-8")).hexdigest(),
        "display": hashlib.sha256(
            f"{envelope.sender_display}|{envelope.organization}".encode("utf-8")
        ).hexdigest(),
        "revision": snapshot.admission_revision,
        "epoch": int(epoch),
        "message": message_id,
        "tenant": owner.tenant_id,
        "logical": fingerprint,
    }

    # 9. Durable LOGICAL reservation BEFORE inference I/O. Exhaustion/busy/
    #    schema failures are typed holds; never a reset.
    try:
        attempt_id = store.reserve_screening_attempt(
            owner, message_id=message_id, fingerprint=fingerprint, input_proof=input_proof
        )
    except StoreRefused as failure:
        code = _HOLD_FOR_SCHEMA.get(failure.code)
        if code is None:
            code = {"budget_exhausted": "budget_exhausted"}.get(failure.code, "retry_later")
        _persist_hold(store, owner, message_id, code)
        return ScreenResult("hold", hold_code=code)

    # 9b. MINT the immutable AdmissionJob (R5): the logical row is bound to
    #     its canonical digest, the runner binds it, and the native dispatch
    #     reservation re-validates owner/policy/attempt/turn before EVERY
    #     actual send. The digest is the ONE fingerprint shared by the
    #     logical row, every physical row and the finish/receipt binding.
    from local_operator.team_bridge.model_boundary import AdmissionJob

    job = AdmissionJob(
        owner_generation=owner.enrollment_generation,
        credential_id=owner.credential_id,
        account_id=owner.account_id,
        tenant_id=owner.tenant_id,
        endpoint_id=owner.endpoint_id,
        hub_url=owner.canonical_base_url,
        message_id=message_id,
        attempt_id=attempt_id,
        body_sha256=envelope.body_sha256,
        template_sha256=snapshot.template_sha256,
        template_version=snapshot.screening_version,
        provider=_canonical_provider,
        model=snapshot.default_model or "",
        approved=tuple(sorted(snapshot.approved_serving_identities)),
        envelope_digest=envelope.envelope_digest,
        request_digest=hashlib.sha256(envelope.type_label.encode("utf-8")).hexdigest(),
        display_digest=hashlib.sha256(
            f"{envelope.sender_display}|{envelope.organization}".encode("utf-8")
        ).hexdigest(),
        revision=snapshot.admission_revision,
        epoch=int(epoch),
        turn=turn,
    )
    fingerprint = job.digest
    try:
        store.bind_attempt_fingerprint(
            owner, attempt_id=attempt_id, fingerprint=fingerprint, job=job
        )
    except StoreRefused:
        _persist_hold(store, owner, message_id, "retry_later")
        return ScreenResult("hold", hold_code="retry_later")
    bind = getattr(runner, "bind_job", None)
    if callable(bind):
        bind(job)

    # 10. Fresh nonce render. A polluted token or a changed delimiter
    #     contract refuses; regeneration is bounded (H3 hostile-input control).
    rendered: RenderedPrompt | None = None
    for _ in range(8):
        try:
            rendered = render_prompt(
                body_text=envelope.private_body(),
                sender=envelope.sender_display,
                organization=envelope.organization,
                msg_type=envelope.type_label,
                nonce=nonce,
                version=snapshot.screening_version,
            )
            break
        except ValueError:
            if nonce is not None:
                break
            continue
    if rendered is None:
        _commit_hold(store, owner, message_id, attempt_id, fingerprint, epoch, "template_mismatch")
        return ScreenResult("hold", hold_code="template_mismatch")

    # 10b. Ownership lifetime (R6/Q15): renew the exact account turn before
    #      the dispatch; if the successor already holds it, ABORT — no
    #      inference under a lost lease, and no release of someone else's.
    if not store.renew_screen_turn(owner, turn=turn, now=_time.time(), ttl=120.0):
        code = "owner_changed" if not store.current_generation_is_live(owner) else "retry_later"
        _commit_hold(store, owner, message_id, attempt_id, fingerprint, epoch, code)
        return ScreenResult("hold", hold_code=code)

    # 11. ONE dispatch. Provider failures are typed holds AND advance the
    #     durable breaker; nothing of the exception's prose is kept.
    #     WHOLE-DURATION lease protection (R7/Q15-5): a callback can outlast
    #     the TTL floor, so the account turn is renewed WHILE the dispatch is
    #     in flight and ownership loss CANCELS the inference — no send keeps
    #     running under a lease a successor now holds (fencing, not luck).
    lost = {"flag": False}
    try:
        outcome = _invoke_runner(runner, rendered.private_text(), snapshot.default_model)
        if inspect.isawaitable(outcome):
            dispatch = asyncio.ensure_future(_maybe_await(outcome))
            keepalive = asyncio.ensure_future(_keep_turn_alive(store, owner, turn, dispatch, lost))
            try:
                reply, actual_model = await dispatch
            finally:
                keepalive.cancel()
                try:
                    await keepalive
                except BaseException:  # noqa: BLE001 — the keeper is already done
                    pass
        else:
            reply, actual_model = outcome
    except asyncio.CancelledError:
        if lost["flag"]:
            # The lease moved mid-dispatch: typed hold, never a receipt, and
            # never a release of the successor's turn (release/renew are
            # token+frame fenced).
            code = "owner_changed" if not store.current_generation_is_live(owner) else "retry_later"
            _commit_hold(store, owner, message_id, attempt_id, fingerprint, epoch, code)
            return ScreenResult("hold", hold_code=code)
        raise
    except Exception:  # noqa: BLE001 - sanitized hold; never prose
        retry_at = _record_failure(store, owner)
        _commit_hold(store, owner, message_id, attempt_id, fingerprint, epoch, "provider_error")
        return ScreenResult("hold", hold_code="provider_error", retry_at=retry_at)

    # 11b. A provider RESPONSE — even a valid model HOLD — is a provider
    #      SUCCESS: it resets consecutive failures (A5). Only the exception
    #      path above counts toward the breaker; cached proofs never do.
    try:
        store.record_screening_success(owner)
    except StoreRefused:
        pass

    # 12. Post-await live-policy recheck: the LIVE policy must still equal the
    #     snapshot that authorized the dispatch, or the outcome is refused
    #     (H3-Q3 — a mid-run default/template/revision switch cannot be
    #     laundered into an admission, however the runner replies).
    if not snapshot.matches_live(policy):
        _commit_hold(store, owner, message_id, attempt_id, fingerprint, epoch, "policy_stale")
        return ScreenResult("hold", hold_code="policy_stale")

    # 13. Identity: the actual serving model must be the captured default or
    #     an APPROVED resolution of it; anything else holds (never a guess).
    if actual_model is None:
        _commit_hold(
            store, owner, message_id, attempt_id, fingerprint, epoch, "model_identity_unknown"
        )
        return ScreenResult("hold", hold_code="model_identity_unknown")
    if not snapshot.accepts_serving_identity(actual_model):
        _commit_hold(
            store, owner, message_id, attempt_id, fingerprint, epoch, "model_identity_mismatch"
        )
        return ScreenResult("hold", hold_code="model_identity_mismatch")

    # 14. Strict parse → write-once metadata-only outcome.
    parsed = parse_verdict(reply)
    if parsed is None:
        _commit_hold(store, owner, message_id, attempt_id, fingerprint, epoch, "parse_failed")
        return ScreenResult("hold", hold_code="parse_failed")
    disposition, categories, reason, _verdict = parsed
    if disposition == "hold":
        _commit_hold(store, owner, message_id, attempt_id, fingerprint, epoch, "model_hold")
        return ScreenResult("hold", hold_code="model_hold").set_private_reason(reason)
    if disposition == "bounce" and not categories:
        _commit_hold(
            store, owner, message_id, attempt_id, fingerprint, epoch, "bounce_without_category"
        )
        return ScreenResult("hold", hold_code="bounce_without_category").set_private_reason(reason)
    if disposition == "bounce":
        # H3-Q1: an UPSTREAM transport record that is already CONFIRMED (a
        # historical read) must never be turned into a new conflicting
        # transport reject. A current local reject here stays LOCAL: a hold
        # with the quarantine reason, no receipt rewrite, original tuple
        # untouched.
        transport_state = store.find_intent_state(owner, message_id, "read")
        if transport_state == "confirmed":
            _commit_hold(
                store, owner, message_id, attempt_id, fingerprint, epoch, "local_quarantine"
            )
            return ScreenResult("hold", hold_code="local_quarantine").set_private_reason(reason)
    # 12b. Final admission fence (R6/Q15): the receipt may only be admitted
    #      while THIS call still owns the account turn — a successor that
    #      took over mid-callback must never inherit our outcome.
    if not store.screen_turn_matches(owner, turn=turn, now=_time.time()):
        code = "owner_changed" if not store.current_generation_is_live(owner) else "retry_later"
        _commit_hold(store, owner, message_id, attempt_id, fingerprint, epoch, code)
        return ScreenResult("hold", hold_code=code)
    receipt = ScreeningReceipt(
        verdict="allow" if disposition == "deliver" else "reject",
        model=actual_model,
        prompt_version=snapshot.screening_version,
        template_sha256=rendered.template_sha256,
        body_sha256=envelope.body_sha256,
        categories=categories,
    )
    try:
        store.finish_screening(
            owner,
            message_id,
            attempt_id=attempt_id,
            fingerprint=fingerprint,
            expected_epoch=epoch,
            receipt=receipt,
            envelope_digest=envelope.envelope_digest,
            admission_revision=snapshot.admission_revision,
            # The turn fence runs INSIDE the receipt transaction (R7/Q15-6):
            # a successor claiming between the outside check and this commit
            # must not let the old pipeline admit.
            turn=turn,
            verify=lambda: snapshot.matches_live(policy),
        )
    except StoreRefused as failure:
        code = _HOLD_FOR_COMMIT.get(failure.code, "retry_later")
        _persist_hold(store, owner, message_id, code)
        return ScreenResult("hold", hold_code=code)
    return ScreenResult(
        disposition,
        receipt=receipt,
        screened_bytes=rendered.screened_bytes,
    ).set_private_reason(reason)


def _policy_row_matches(row: Mapping[str, Any], snapshot: "PolicySnapshot") -> bool:
    """Whether the PUBLISHED policy row equals the caller's frozen snapshot.

    The store row is the authority; the caller object is only a consumer's
    view. Equality here (not "newer than") is the admission condition — any
    difference means the consumer is stale and must hold.
    """

    return (
        row.get("version") == snapshot.screening_version
        and row.get("template_sha256") == snapshot.template_sha256
        and row.get("default_model") == snapshot.default_model
        and row.get("revision") == snapshot.admission_revision
    )


def _record_failure(store: Store, owner: OwnerSnapshot) -> float | None:
    """Advance the durable breaker; return its retry time (best effort)."""

    try:
        return store.record_screening_failure(owner)
    except StoreRefused:
        return None


def _commit_hold(
    store: Store,
    owner: OwnerSnapshot,
    message_id: str,
    attempt_id: str,
    fingerprint: str,
    epoch: int,
    hold_code: str,
) -> None:
    """Record ONE hold against the reserved attempt; never raise."""

    try:
        store.finish_screening(
            owner,
            message_id,
            attempt_id=attempt_id,
            fingerprint=fingerprint,
            expected_epoch=epoch,
            hold_code=hold_code,
        )
    except StoreRefused:
        _persist_hold(store, owner, message_id, hold_code)


async def _keep_turn_alive(
    store: Store,
    owner: OwnerSnapshot,
    turn: str,
    dispatch: "asyncio.Future[Any]",
    lost: dict[str, bool],
) -> None:
    """Renew the account turn periodically while a dispatch is in flight.

    On ownership loss (renew refused — successor, expiry or generation
    move) the DISPATCH is cancelled and the loss is flagged so the caller
    emits a typed hold instead of leaking a bare CancelledError. The keeper
    never touches the lease itself beyond renew.
    """

    import time as _time

    while True:
        await asyncio.sleep(_TURN_KEEPALIVE_S)
        try:
            ok = store.renew_screen_turn(
                owner, turn=turn, now=_time.time(), ttl=_TURN_KEEPALIVE_TTL
            )
        except StoreRefused:
            ok = False
        if not ok:
            lost["flag"] = True
            dispatch.cancel()
            return


def _invoke_runner(runner: Any, prompt: str, default_model: str) -> Any:
    """Call the runner through its PUBLIC async interface when it has one.

    The production runner is a ScreenModelRunner whose sync ``__call__``
    cannot drive a loop — inside the async core that returned the unevidenced
    ("", None) hold without ever dispatching (QA Q10). A controlled
    synchronous fixture (a plain callable) is called directly; both shapes
    are awaited through ``_maybe_await``.
    """

    acall = getattr(runner, "acall", None)
    if callable(acall):
        return acall(prompt, default_model=default_model)
    return runner(prompt, default_model=default_model)


async def _maybe_await(value: Any) -> Any:
    """Await the runner's result when it is awaitable; pass through otherwise."""

    if inspect.isawaitable(value):
        return await value
    return value


def _identities_digest(identities: frozenset[str]) -> str:
    """Bounded digest of the approved serving identities (fingerprint input)."""

    payload = ",".join(sorted(identities))
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()[:16]


def screen_message(
    owner: OwnerSnapshot,
    item: Mapping[str, Any],
    *,
    config_dir: Path,
    policy: ScreeningPolicy,
    runner: ScreenRunner | None = None,
    nonce: str | None = None,
) -> ScreenResult:
    """Synchronous facade over :func:`screen_message_async`.

    It runs the async core on a PRIVATE loop only when this thread has none
    (never nested, never a thread workaround); inside an active loop it
    refuses — the caller should await :func:`screen_message_async`.
    """

    try:
        asyncio.get_running_loop()
    except RuntimeError:
        pass
    else:
        raise RuntimeError(
            "screen_message cannot run inside an active event loop; "
            "await screen_message_async instead"
        )
    loop = asyncio.new_event_loop()
    try:
        return loop.run_until_complete(
            screen_message_async(
                owner, item, config_dir=config_dir, policy=policy, runner=runner, nonce=nonce
            )
        )
    finally:
        loop.close()
