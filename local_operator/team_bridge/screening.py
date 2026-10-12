"""H3 — the one-shot screener: pure render/parse plus the shared pipeline.

WHAT LIVES HERE. Three separable pieces, all deterministic and testable
without a provider:

1. :func:`render_prompt` — single-pass placeholder substitution over the
   FROZEN embedded template (``bridge-screen-v1``). A fresh cryptographic
   nonce delimits the submission; substituted values are never re-scanned,
   so text inserted from the (hostile) message cannot forge a closing tag or
   grow a new placeholder.
2. :func:`parse_verdict` — the frozen two-line reply grammar, strict. Anything
   that is not exactly ``VERDICT: deliver|hold|bounce`` followed by one
   nonempty ``REASON:`` line (<=200 chars) is a HOLD: extra prose, markdown,
   missing/duplicate fields, tool calls, unknown categories on a bounce.
3. :func:`screen_message` — the ONE shared pipeline: exact-owner and template
   authority, body-hash binding, H2 intent/cache recovery before any model
   spend, durable budget reservation BEFORE inference I/O, the single model
   dispatch, strict parse, and a metadata-only durable outcome. No delivery,
   no ack, no wake, no body/reason/text ever reaches durable state.

CONTENT LAW. The model reply and the message body are ephemeral: they may
exist in memory for the duration of one call and are never written to the
store, logs, analytics, receipts or error text. Durable records carry only
hashes, closed-enum metadata, model identity, counts and timestamps
(§13.9's operational metadata; the contract's no-content-cache rule).

H3 grants nothing: a ``deliver``/``allow`` here is evidence for the H2 ack
path only. Working/done, approvals, delivery and any H4 connector remain
other lanes' work.
"""

from __future__ import annotations

import hashlib
import re
import secrets
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Mapping, Protocol

from local_operator.team_bridge.client import (
    SCREENING_TEMPLATE_V1,
    SCREENING_TEMPLATE_V1_SHA256,
    SCREENING_VERSION_V1,
)
from local_operator.team_bridge.store import Store, StoreRefused
from local_operator.team_bridge.types import OwnerSnapshot, ScreeningReceipt

__all__ = [
    "HOLD_CODES",
    "ScreeningPolicy",
    "ScreenResult",
    "ScreenRunner",
    "parse_verdict",
    "render_prompt",
    "screen_message",
]

#: The closed category enum (frozen §8). Bounce receipts carry ONLY these,
#: de-duplicated; a bounce without one unambiguous known token holds.
SCREENING_CATEGORIES = (
    "prompt_injection",
    "social_engineering",
    "harassment",
    "malicious_link",
    "spam",
)

#: The closed hold-code set. Every non-admission is ONE of these; none of
#: them ever carries model/vendor prose.
HOLD_CODES = (
    "owner_mismatch",
    "template_mismatch",
    "body_hash_mismatch",
    "intent_conflict",
    "budget_exhausted",
    "runner_unavailable",
    "provider_error",
    "model_identity_unknown",
    "model_identity_mismatch",
    "parse_failed",
    "bounce_without_category",
    "model_hold",
)

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
    different-model fallback); ``admission_revision`` names the snapshot the
    decision is bound to.
    """

    screening_version: str
    template_sha256: str
    default_model: str | None
    admission_revision: str


@dataclass
class RenderedPrompt:
    """One rendered screening request. ``text`` is ephemeral; send it, drop it."""

    text: str
    nonce: str
    template_sha256: str
    body_sha256: str
    screened_bytes: int


@dataclass
class ScreenResult:
    """The pipeline's outcome. Content-free by construction.

    ``ephemeral_reason`` is for the CALLER's immediate use (e.g. a local
    surface) and is deliberately NOT part of any store write: nothing here
    is ever handed to an agent tool, logged, or persisted.
    """

    disposition: str  # deliver | bounce | hold
    hold_code: str | None = None
    receipt: ScreeningReceipt | None = None
    retry_at: float | None = None
    screened_bytes: int | None = None
    ephemeral_reason: str | None = field(default=None, repr=False)


class ScreenRunner(Protocol):
    """The one model dispatch H3 allows itself.

    Implementations MUST send exactly one user turn derived from the rendered
    prompt, with no tools, no tool choice, no session/history/system/cache
    lineage and no callbacks, and MUST report the ACTUAL serving model id (or
    ``None`` when the runtime cannot evidence it — which holds). Returns
    ``(reply_text, actual_model_or_none)``. Raises only sanitized, typed
    errors; vendor prose never escapes.
    """

    def __call__(self, prompt: str, *, default_model: str) -> tuple[str, str | None]: ...


def _body_sha256(body_text: str) -> str:
    # §8: hash the EXACT UTF-8 stored body.text — no NFC re-normalization,
    # no JSON re-serialization, no envelope.
    return hashlib.sha256(body_text.encode("utf-8")).hexdigest()


def render_prompt(
    *,
    body_text: str,
    sender: str,
    organization: str,
    msg_type: str,
    nonce: str | None = None,
) -> RenderedPrompt:
    """Single-pass substitution over the frozen template — the ONLY renderer.

    The nonce must be absent from every substituted value and every value is
    inserted ONCE, left-to-right, without re-scanning: a message body that
    contains ``{{nonce}}`` (or an imitated closing tag) stays inert data.
    The template's three nonce occurrences are asserted, so a future template
    edit cannot silently change the delimiter contract.
    """

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
    template = SCREENING_TEMPLATE_V1
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
    return RenderedPrompt(
        text=text,
        nonce=token,
        template_sha256=SCREENING_TEMPLATE_V1_SHA256,
        body_sha256=_body_sha256(body_text),
        screened_bytes=len(body_text.encode("utf-8")),
    )


def parse_verdict(reply: str) -> tuple[str, tuple[str, ...], str, str] | None:
    """Frozen two-line grammar → ``(disposition, categories, reason, verdict)``.

    ``None`` means the reply is NOT a valid screening answer and must hold.
    The reason is returned for the caller's ephemeral use only.
    """

    if not isinstance(reply, str):
        return None
    # Exactly two lines: strip the ONE conventional trailing terminator, but
    # an extra blank line is extra output and must reject (strict grammar).
    text = reply[:-1] if reply.endswith("\n") else reply
    if "\r" in text:
        return None
    lines = text.split("\n")
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


def screen_message(
    owner: OwnerSnapshot,
    item: Mapping[str, Any],
    *,
    config_dir: Path,
    policy: ScreeningPolicy,
    runner: ScreenRunner | None = None,
    nonce: str | None = None,
) -> ScreenResult:
    """The ONE screening pipeline. At most one runner dispatch per call.

    Order is the contract: authority → template → body hash → H2 intent/cache
    recovery (no model cost) → durable budget reservation → dispatch → strict
    parse → metadata-only durable outcome. Every refusal is a hold with a
    closed code; nothing durable ever carries body text, reasons or replies.
    """

    store = Store(config_dir)

    # 1. Exact owner + latest committed generation (native guard).
    try:
        committed = store.resolve_owner(
            credential_id=owner.credential_id,
            account_id=owner.account_id,
            tenant_id=owner.tenant_id,
            endpoint_id=owner.endpoint_id,
            canonical_base_url=owner.canonical_base_url,
        )
    except StoreRefused:
        return ScreenResult("hold", hold_code="owner_mismatch")
    if committed is None or committed != owner:
        return ScreenResult("hold", hold_code="owner_mismatch")

    # 2. Template authority: byte-equal frozen v1 only.
    if (
        policy.screening_version != SCREENING_VERSION_V1
        or policy.template_sha256 != SCREENING_TEMPLATE_V1_SHA256
    ):
        return ScreenResult("hold", hold_code="template_mismatch")

    # 3. Body binding BEFORE any model request.
    raw_message = item.get("message")
    message: Mapping[str, Any] = raw_message if isinstance(raw_message, Mapping) else {}
    message_id = message.get("id")
    raw_body = message.get("body")
    body: Mapping[str, Any] = raw_body if isinstance(raw_body, Mapping) else {}
    body_text = body.get("text")
    if not isinstance(message_id, str) or not message_id or not isinstance(body_text, str):
        return ScreenResult("hold", hold_code="body_hash_mismatch")
    body_sha = _body_sha256(body_text)
    served_hash = message.get("body_hash")
    if not isinstance(served_hash, str) or served_hash != body_sha:
        return ScreenResult("hold", hold_code="body_hash_mismatch")

    # 4. Reconcile H2's immutable tuple FIRST: a pending/recorded intent for
    #    this exact tuple recovers without a new screening round (no rebill).
    existing = store.find_pending_intent(owner, message_id, "read")
    if existing is not None and existing["receipt"] is not None:
        receipt_data = existing["receipt"]
        if receipt_data.get("body_sha256") == body_sha:
            return ScreenResult(
                "deliver",
                receipt=ScreeningReceipt(
                    verdict=receipt_data["verdict"],
                    model=receipt_data["model"],
                    prompt_version=receipt_data["prompt_version"],
                    template_sha256=receipt_data["template_sha256"],
                    body_sha256=receipt_data["body_sha256"],
                    categories=tuple(receipt_data["categories"]),
                ),
            )

    # 4b. Same-body cache: reuse ONLY when model+template are the SAME
    #     (a changed fingerprint must miss rather than replay a stale allow).
    if policy.default_model is not None:
        cached = store.find_screening_receipt(
            owner,
            body_sha256=body_sha,
            template_sha256=SCREENING_TEMPLATE_V1_SHA256,
            model=policy.default_model,
        )
        if cached is not None and cached.get("verdict") == "allow":
            return ScreenResult(
                "deliver",
                receipt=ScreeningReceipt(
                    verdict="allow",
                    model=cached["model"],
                    prompt_version=SCREENING_VERSION_V1,
                    template_sha256=cached["template_sha256"],
                    body_sha256=cached["body_sha256"],
                    categories=tuple(cached["categories"]),
                ),
            )

    # 5. Fresh nonce render (bounded regeneration; the renderer refuses a
    #    polluted token outright).
    raw_sender = item.get("sender")
    sender_data: Mapping[str, Any] = raw_sender if isinstance(raw_sender, Mapping) else {}
    rendered: RenderedPrompt | None = None
    for _ in range(8):
        try:
            rendered = render_prompt(
                body_text=body_text,
                sender=str(sender_data.get("account_id", "")),
                organization=str(owner.tenant_id),
                msg_type=str(message.get("type", "")),
                nonce=nonce,
            )
            break
        except ValueError:
            if nonce is not None:
                return ScreenResult("hold", hold_code="template_mismatch")
            continue
    if rendered is None:
        return ScreenResult("hold", hold_code="template_mismatch")

    # 6. Durable reservation BEFORE inference I/O: exactly one charge per
    #    dispatch; exhaustion is a HOLD, never a reset.
    if policy.default_model is None:
        return ScreenResult("hold", hold_code="runner_unavailable")
    try:
        attempt_id = store.reserve_screening_attempt(owner)
    except StoreRefused as failure:
        return ScreenResult(
            "hold",
            hold_code=(
                "budget_exhausted" if failure.code == "budget_exhausted" else "provider_error"
            ),
        )
    if runner is None:
        store.finish_screening(
            owner, message_id, attempt_id=attempt_id, hold_code="runner_unavailable"
        )
        return ScreenResult("hold", hold_code="runner_unavailable")

    # 7. ONE dispatch; sanitized typed failures only (the runner's contract).
    try:
        reply, actual_model = runner(rendered.text, default_model=policy.default_model)
    except Exception:  # noqa: BLE001 - sanitized hold; never prose
        store.finish_screening(owner, message_id, attempt_id=attempt_id, hold_code="provider_error")
        return ScreenResult("hold", hold_code="provider_error")

    if actual_model is None:
        store.finish_screening(
            owner, message_id, attempt_id=attempt_id, hold_code="model_identity_unknown"
        )
        return ScreenResult("hold", hold_code="model_identity_unknown")
    if actual_model != policy.default_model:
        # The runtime resolved a DIFFERENT model than the captured default:
        # no fallback is authorized, so this is a hold, never an allow.
        store.finish_screening(
            owner, message_id, attempt_id=attempt_id, hold_code="model_identity_mismatch"
        )
        return ScreenResult("hold", hold_code="model_identity_mismatch")

    # 8. Strict parse → metadata-only durable outcome.
    parsed = parse_verdict(reply)
    if parsed is None:
        store.finish_screening(owner, message_id, attempt_id=attempt_id, hold_code="parse_failed")
        return ScreenResult("hold", hold_code="parse_failed")
    disposition, categories, reason, _verdict = parsed
    if disposition == "hold":
        store.finish_screening(owner, message_id, attempt_id=attempt_id, hold_code="model_hold")
        return ScreenResult("hold", hold_code="model_hold", ephemeral_reason=reason)
    if disposition == "bounce" and not categories:
        store.finish_screening(
            owner, message_id, attempt_id=attempt_id, hold_code="bounce_without_category"
        )
        return ScreenResult("hold", hold_code="bounce_without_category")
    receipt = ScreeningReceipt(
        verdict="allow" if disposition == "deliver" else "reject",
        model=actual_model,
        prompt_version=SCREENING_VERSION_V1,
        template_sha256=rendered.template_sha256,
        body_sha256=body_sha,
        categories=categories,
    )
    store.finish_screening(
        owner, message_id, attempt_id=attempt_id, receipt=receipt, hold_code=None
    )
    return ScreenResult(
        disposition,
        receipt=receipt,
        screened_bytes=rendered.screened_bytes,
        ephemeral_reason=reason,
    )
